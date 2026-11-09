"""Experimental FP8 DSpine projections using Saber's shipped gfx1201 kernels.

Weights use independent 128x128 scale blocks; activations use row-wise groups
of 128. Bridge and serial-table arithmetic stay BF16. This changes the draft
model numerically and requires acceptance measurements against the BF16 model.
It does not select a serving backend or modify checkpoint files.
"""
import torch
import hip_module


def preshuffle(weight):
    """Map [N,K] bytes to the native WMMA [N/16,K*16] layout."""
    n,k=weight.shape
    if weight.dtype!=torch.uint8 or n%16 or k%32:
        raise ValueError('Expected byte weights with N%16=K%32=0')
    return weight.reshape(n//16,16,k//32,2,16).permute(0,2,3,1,4).contiguous().reshape(n//16,k*16)


@torch.no_grad()
def quantize_weight(weight):
    """Return packed E4M3FN bytes and FP32 scales without changing the source."""
    if weight.ndim!=2 or weight.dtype!=torch.bfloat16:
        raise ValueError('Expected a BF16 matrix')
    n,k=weight.shape
    if n%128 or k%128:
        raise ValueError('Blockscale projections require N and K multiples of128')
    blocks=weight.float().reshape(n//128,128,k//128,128)
    scales=(blocks.abs().amax(dim=(1,3))/448).clamp_min(1/(448*512))
    quant=(blocks/scales[:,None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    return preshuffle(quant.reshape(n,k).view(torch.uint8)),scales.contiguous()


class NativeLinear:
    """Own quantized projections and use PyTorch's current HIP stream."""
    def __init__(self,weights,prebuilt,device=0):
        self.device=device
        self.gemm=hip_module.HipModule(prebuilt,'dflash_gemm',device)
        self.quant=hip_module.HipModule(prebuilt,'dflash_mlp',device)
        self.weights={}
        for name,weight in weights.items():
            if name=='fc.weight' or (name.startswith('layers.') and name.endswith('_proj.weight')):
                if not weight.is_cuda or weight.device.index!=device:
                    raise ValueError('Projection weights must be on the module device')
                packed,scales=quantize_weight(weight)
                self.weights[weight.data_ptr()]=(weight.shape,packed,scales)

    @torch.no_grad()
    def quantize_activation(self,inputs):
        """Expose the native quantizer for independent numerical comparison."""
        if inputs.ndim!=2 or inputs.dtype!=torch.bfloat16 or inputs.shape[1]%128:
            raise ValueError('Expected [rows,K] BF16 with K%128=0')
        if not inputs.is_cuda or inputs.device.index!=self.device or not inputs.shape[0]:
            raise ValueError('Nonempty inputs must share the module device')
        x=inputs.contiguous();m,k=x.shape
        q=torch.empty((m,k),dtype=torch.uint8,device=x.device)
        scales=torch.empty((m,k//128),dtype=torch.float32,device=x.device)
        self.quant.launch('dflash_act_quant_g128',((m*(k//128)+3)//4,1,1),(128,1,1),
                          {'x':x,'q':q,'scales':scales,'t_rows':m,'n':k})
        return q,scales

    @torch.no_grad()
    def __call__(self,inputs,weight):
        entry=self.weights.get(weight.data_ptr())
        if entry is None:
            raise ValueError('Unregistered FP8 projection')
        (n,k),packed,scales=entry
        if inputs.ndim!=2 or inputs.shape[1]!=k or inputs.dtype!=torch.bfloat16:
            raise ValueError('Projection inputs must be [rows,K] BF16')
        if not inputs.is_cuda or inputs.device.index!=self.device:
            raise ValueError('Projection inputs must share the module device')
        m=inputs.shape[0]
        if not m:
            raise ValueError('Empty projections are not supported')
        x=inputs.contiguous()
        q,a_scales=self.quantize_activation(x)
        output=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
        name='dflash_gemm_a8w8_bs128_m16n128k128' if m<=16 else 'dflash_gemm_a8w8_bs128_mtn128k128'
        self.gemm.launch(name,(n//128,(m+15)//16,1),(128,1,1),
                         {'a':q,'w':packed,'as':a_scales,'ws':scales,'c':output,'m':m,'n':n,'k':k})
        return output


@torch.no_grad()
def quantize_channel_weight(weight):
    """W8A16 experiment: row scales and BF16 dequantization in the GEMM."""
    if weight.ndim!=2 or weight.dtype!=torch.bfloat16 or weight.shape[0]%16 or weight.shape[1]%16:
        raise ValueError('Expected a BF16 matrix with N and K multiples of16')
    scales=(weight.float().abs().amax(-1)/448).clamp_min(1/(448*512))
    q=(weight.float()/scales[:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).contiguous(),scales.contiguous()


class NativeChannelLinear:
    """Compare the target head's W8A16 kernel with group-128 W8A8 drafting."""
    def __init__(self,weights,prebuilt,device=0):
        self.device=device
        self.module=hip_module.HipModule(prebuilt,'dflash_bgemm',device)
        self.weights={}
        for name,weight in weights.items():
            if name=='fc.weight' or (name.startswith('layers.') and name.endswith('_proj.weight')):
                if not weight.is_cuda or weight.device.index!=device:
                    raise ValueError('Projection weights must share the module device')
                q,scales=quantize_channel_weight(weight)
                self.weights[weight.data_ptr()]=(weight.shape,q,scales)

    @torch.no_grad()
    def __call__(self,inputs,weight):
        entry=self.weights.get(weight.data_ptr())
        if entry is None:
            raise ValueError('Unregistered channel-FP8 projection')
        (n,k),q,scales=entry
        if (inputs.ndim!=2 or inputs.shape[1]!=k or inputs.shape[0]==0
                or inputs.dtype!=torch.bfloat16 or not inputs.is_cuda or inputs.device.index!=self.device):
            raise ValueError('Expected nonempty [rows,K] BF16 on the module device')
        m=inputs.shape[0]
        output=torch.empty((m,n),device=inputs.device,dtype=torch.bfloat16)
        # This symbol's body is the general 16-row slice: blockIdx.y selects
        # its m0, and the store bounds every row. It also serves a partial tile.
        self.module.launch('dflash_bgemm_w8_m16n16_w128_u16',((n+63)//64,(m+15)//16,1),(128,1,1),
                           {'a':inputs.contiguous(),'w':q,'ws':scales,'c':output,'m':m,'n':n,'k':k})
        return output
