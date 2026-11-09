"""Select coarse candidates by descending FP32 value, then ascending token ID."""
import torch
import triton
import triton.language as tl


@triton.jit
def _keys(values,ids,out,count,BLOCK:tl.constexpr):
    at=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    live=at<count
    f=tl.load(values+at,mask=live,other=0.0)
    ident=tl.load(ids+at,mask=live,other=0).to(tl.uint32)
    bits=f.to(tl.uint32,bitcast=True)
    # +0 and -0 have equal score. Put all NaNs first, as torch.topk largest does.
    bits=tl.where((bits & 0x7fffffff)==0,0,bits)
    sortable=tl.where((bits & 0x80000000)!=0,bits ^ 0xffffffff,bits ^ 0x80000000)
    sortable=tl.where((bits & 0x7fffffff)>0x7f800000,0xffffffff,sortable).to(tl.uint32)
    # Map unsigned high bits to signed for PyTorch's int64 comparison. Every
    # valid block-max slot has a distinct token ID, so no call-site tie remains.
    hi=(sortable ^ 0x80000000).to(tl.uint32).to(tl.int32,bitcast=True).to(tl.int64)
    key=(hi << 32) | (ident ^ 0xffffffff).to(tl.uint32).to(tl.int64)
    tl.store(out+at,key,mask=live)


def positions(values,ids,k):
    if (values.ndim!=2 or values.shape!=ids.shape or values.dtype!=torch.float32
            or ids.dtype!=torch.int32 or ids.device!=values.device
            or not values.is_contiguous() or not ids.is_contiguous() or not 0<=k<=values.shape[1]):
        raise ValueError("Canonical coarse top-k expects contiguous FP32 scores and unique INT32 token IDs")
    if k==0 or values.shape[0]==0:
        return torch.empty((values.shape[0],k),device=values.device,dtype=torch.int64)
    out=torch.empty(values.shape,device=values.device,dtype=torch.int64)
    _keys[(triton.cdiv(values.numel(),256),)](values,ids,out,values.numel(),BLOCK=256)
    return torch.topk(out,k,dim=-1,sorted=True).indices
