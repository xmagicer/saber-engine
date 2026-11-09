"""Standalone BF16 DSpine draft forward for numerical/replay experiments.

Based on Tencent/DSpine 0cc92d85 and SGLang v0.5.22 (Apache-2.0).
This eager reference has no serving or tokenizer dependency. RoPE uses FP32
angles and NeoX half rotation. Teacher K/V is projected from target auxiliary
features and never replaced by the speculative query block's K/V.
"""
import json
from pathlib import Path
import torch
import reference as Ref
import target_layers as TargetLayers


def rms_norm(x, weight, eps):
    """SGLang RMSNorm default: multiply the weight in FP32, then round once.

    The DSpine bridge's fused next-norm has its own explicitly two-round BF16
    contract in reference.feature_finish; it must not be reused here.
    """
    v = x.float()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16)


def add_rms_norm(x,residual,weight,eps):
    """Normalize the FP32 residual sum; separately store its BF16 residual."""
    full=x.float()+residual.float()
    return rms_norm(full,weight,eps),full.to(torch.bfloat16)


class TorchDspine:
    """Plain PyTorch DSpine bridge plus five causal sliding-attention layers."""
    def __init__(self, config, weights):
        self.config = config
        self.w = weights
        d = config['dflash_config']
        self.h = int(config['hidden_size'])
        self.layers = int(config['num_hidden_layers'])
        self.qheads = int(config['num_attention_heads'])
        self.kheads = int(config['num_key_value_heads'])
        self.dim = int(config['head_dim'])
        self.block = int(config['block_size'])
        self.window = int(config['sliding_window'])
        self.eps = float(config.get('rms_norm_eps',1e-6))
        self.bridge_eps = float(d['selected_prefix_code_bridge'].get('eps',1e-6))
        self.theta = float(config.get('rope_theta',config.get('rope_parameters',{}).get('rope_theta',10000000)))
        if (not d.get('draft_block_causal') or d.get('conv_kernel_size')
                or d.get('selected_prefix_terminal',True)
                or d.get('selected_prefix_feature_write_mode')!='independent'
                or d.get('selected_prefix_feature_writes') is not True
                or sorted(d['selected_prefix_feature_layers'])!=list(range(1,self.layers+1))
                or self.qheads % self.kheads or self.dim % 2 or self.window<=0):
            raise ValueError('Unsupported DSpine architecture')
        refinement=config['code_bridge_second_pass']
        if refinement['mode']!='serial_table' or refinement['first_exact']!=self.layers-1:
            raise ValueError('Unsupported DSpine refinement')
        for name in ['draft_block_causal_version','last_write_refine_version',
                     'selected_prefix_bridge.independent_write_version',
                     'selected_prefix_bridge.terminal_disabled_version']:
            if int(weights[name].item())!=1:
                raise ValueError('Unsupported checkpoint version: '+name)
        if bool((weights['selected_prefix_bridge.out.weight']!=0).any()):
            raise ValueError('DSpine terminal output must be zero')
        for name,tensor in weights.items():
            if tensor.is_floating_point() and tensor.dtype!=torch.bfloat16:
                raise ValueError('This reference supports BF16 only: '+name)
        self.device = weights['fc.weight'].device
        self.target_features = len(d['target_layer_ids'])
        self.capture_layer_ids = TargetLayers.capture_layers_from_dspine(d['target_layer_ids'])
        if weights['fc.weight'].shape!=(self.h,self.target_features*self.h):
            raise ValueError('Target auxiliary projection shape mismatch')
        self.packed = Ref.pack_bridge_weights(weights['selected_prefix_bridge.gate.weight'],
                                              weights['selected_prefix_bridge.predecessor_gate.weight'],
                                              weights['selected_prefix_bridge.prev.weight'])
        self.inv_freq = self.theta ** (-torch.arange(0,self.dim,2,device=self.device,dtype=torch.float32)/self.dim)
        self.cache = {}
        self.linear = torch.nn.functional.linear
        self.bridge = Ref.bridge_layer

    @classmethod
    def from_checkpoint(cls, directory, device='cpu'):
        """Load the draft checkpoint only; target embeddings/head are supplied by the caller."""
        from safetensors import safe_open
        directory=Path(directory)
        config=json.loads((directory/'config.json').read_text())
        with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as f:
            weights={name:f.get_tensor(name).to(device) for name in f.keys()}
        return cls(config,weights)

    def rope(self, value, positions):
        """Rotate [rows, heads, dim] at the supplied absolute positions."""
        angles=positions.float()[:,None]*self.inv_freq[None,:]
        cosine,sine=angles.cos()[:,None,:],angles.sin()[:,None,:]
        a,b=value[...,:self.dim//2].float(),value[...,self.dim//2:].float()
        return torch.cat((a*cosine-b*sine,b*cosine+a*sine),-1).to(torch.bfloat16)

    def per_head_norm(self, value, weight):
        return rms_norm(value,weight,self.eps)

    @torch.no_grad()
    def append_context(self,seq,aux,positions):
        """Append only contiguous, committed target features; retain the trained sliding window."""
        if aux.ndim!=2 or aux.shape[1]!=self.target_features*self.h or positions.shape!=(aux.shape[0],):
            raise ValueError('Teacher context shape mismatch')
        Ref.require_bf16(aux)
        if not aux.shape[0]:
            return
        entry=self.cache.get(seq)
        next_position=0 if entry is None else entry['next']
        expected=torch.arange(next_position,next_position+aux.shape[0],device=positions.device,dtype=positions.dtype)
        if positions.dtype!=torch.int64 or not torch.equal(positions,expected):
            raise ValueError('Teacher context positions must be contiguous from zero')
        if aux.device!=self.device or positions.device!=self.device:
            raise ValueError('Context and weights must share a device')
        projected=rms_norm(self.linear(aux,self.w['fc.weight']),self.w['hidden_norm.weight'],self.eps)
        new_keys=[];new_values=[]
        for layer in range(self.layers):
            prefix=f'layers.{layer}.self_attn.'
            key=self.linear(projected,self.w[prefix+'k_proj.weight']).reshape(-1,self.kheads,self.dim)
            key=self.rope(self.per_head_norm(key,self.w[prefix+'k_norm.weight']),positions)
            value=self.linear(projected,self.w[prefix+'v_proj.weight']).reshape(-1,self.kheads,self.dim)
            if entry is not None:
                key=torch.cat((entry['keys'][layer],key),0)
                value=torch.cat((entry['values'][layer],value),0)
            new_keys.append(key[-self.window:].contiguous())
            new_values.append(value[-self.window:].contiguous())
        pos=positions if entry is None else torch.cat((entry['positions'],positions))
        self.cache[seq]={'next':next_position+aux.shape[0],'positions':pos[-self.window:].contiguous(),
                         'keys':new_keys,'values':new_values}

    @torch.no_grad()
    def forward_block(self,seq,input_embeds,input_ids,positions):
        """Return normed draft states and refinement stash without mutating teacher K/V."""
        t=input_embeds.shape[0]
        if input_embeds.shape!=(t,self.h) or not 2<=t<=self.block or input_ids.shape!=(t,) or positions.shape!=(t,):
            raise ValueError('Draft block shape mismatch')
        Ref.require_bf16(input_embeds)
        if input_ids.dtype!=torch.int64 or positions.dtype!=torch.int64:
            raise ValueError('Draft ids and positions must be int64')
        entry=self.cache.get(seq)
        start=0 if entry is None else entry['next']
        if not torch.equal(positions,torch.arange(start,start+t,device=positions.device,dtype=positions.dtype)):
            raise ValueError('Query block must follow committed teacher context')
        return self.forward_block_tensors(input_embeds,input_ids,positions,entry)

    def forward_block_tensors(self,input_embeds,input_ids,positions,entry):
        """Tensor-only body for a validated, fixed-shape graph caller.

        Entry contains only committed teacher K/V and absolute positions. The
        caller owns shape, dtype, position and device validation. No cache state
        is changed here; padded teacher rows need invalid absolute positions.
        """
        t=input_embeds.shape[0]
        residual=input_embeds
        hidden=rms_norm(input_embeds,self.w['layers.0.input_layernorm.weight'],self.eps)
        key_positions=positions if entry is None else torch.cat((entry['positions'],positions))
        mask=(key_positions[None,:]<=positions[:,None]) & (key_positions[None,:]>positions[:,None]-self.window)
        stash=None
        for layer in range(self.layers):
            prefix=f'layers.{layer}.';attn=prefix+'self_attn.'
            query=self.linear(hidden,self.w[attn+'q_proj.weight']).reshape(t,self.qheads,self.dim)
            key=self.linear(hidden,self.w[attn+'k_proj.weight']).reshape(t,self.kheads,self.dim)
            value=self.linear(hidden,self.w[attn+'v_proj.weight']).reshape(t,self.kheads,self.dim)
            query=self.rope(self.per_head_norm(query,self.w[attn+'q_norm.weight']),positions)
            key=self.rope(self.per_head_norm(key,self.w[attn+'k_norm.weight']),positions)
            if entry is not None:
                key=torch.cat((entry['keys'][layer],key),0)
                value=torch.cat((entry['values'][layer],value),0)
            q=query.transpose(0,1)[None]
            k=key.transpose(0,1)[None].repeat_interleave(self.qheads//self.kheads,dim=1)
            v=value.transpose(0,1)[None].repeat_interleave(self.qheads//self.kheads,dim=1)
            attended=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask[None,None],dropout_p=0.0)
            attended=attended[0].transpose(0,1).reshape(t,self.qheads*self.dim).contiguous()
            projected=self.linear(attended,self.w[attn+'o_proj.weight'])
            mlp_input,residual=add_rms_norm(projected,residual,self.w[prefix+'post_attention_layernorm.weight'],self.eps)
            gate=self.linear(mlp_input,self.w[prefix+'mlp.gate_proj.weight'])
            up=self.linear(mlp_input,self.w[prefix+'mlp.up_proj.weight'])
            glu=(torch.nn.functional.silu(gate.float())*up.float()).to(torch.bfloat16)
            mlp=self.linear(glu,self.w[prefix+'mlp.down_proj.weight'])
            read=self.w['selected_prefix_bridge.read_weight'] if layer==self.layers-1 else self.w['selected_prefix_bridge.early_read_weights'][layer]
            next_norm=self.w['norm.weight'] if layer==self.layers-1 else self.w[f'layers.{layer+1}.input_layernorm.weight']
            hidden,residual,stash=self.bridge(mlp,residual,input_ids,read,self.w['selected_prefix_bridge.codebook'],
                self.packed,self.w['selected_prefix_bridge.gate.bias'],self.w['selected_prefix_bridge.feature_write_weights'][layer],
                next_norm,t,self.bridge_eps,self.eps)
        return hidden,stash

    def clear(self,seq):
        """Release one request's committed teacher context."""
        self.cache.pop(seq,None)
