"""Experimental fused BF16 DSpine bridge on Triton/HIP.

Four arithmetic kernels are adapted from Tencent/DSpine, Apache-2.0,
commit 0cc92d857a8f7bf252de67995ad4c19251620143,
sglang_dspine/kernels/dspine.py. Copyright (C) 2026 Tencent.
See LICENSE and NOTICE. No model or serving backend is selected by importing it.
"""
import torch
import triton
import triton.language as tl
import reference as Ref

@triton.jit
def prepare_read(
    X,
    Residual,
    Raw,
    Normalized,
    RMS,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, WIDTH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    if ADD:
        residual = tl.load(Residual + row * H + h, h < H, 0).to(tl.float32)
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
    rms = tl.sqrt(tl.maximum(tl.sum(x * x, 0) / H, 1.1754943508222875e-38))
    tl.store(Raw + row * H + h, x, h < H)
    tl.store(Normalized + row * H + h, x / (rms + EPS), h < H)
    tl.store(RMS + row, rms)


@triton.jit
def normalize_codes(
    Projected,
    Codebook,
    Anchors,
    Codes,
    C: tl.constexpr,
    T: tl.constexpr,
    ANCHOR_ROW: tl.constexpr,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    if ANCHOR_ROW and row % T == 0:
        token = tl.load(Anchors + row // T)
        z = tl.load(Codebook + token * C + k, k < C, 0).to(tl.float32)
        z = z * tl.sqrt(float(C))
    else:
        z = tl.load(Projected + row * C + k, k < C, 0).to(tl.float32)
        norm = tl.sqrt(tl.sum(z * z, 0))
        z = tl.sqrt(float(C)) * z / (norm + EPS)
    tl.store(Codes + row * C + k, z, k < C)


@triton.jit
def feature_message(
    Packed, Bias, Message, R: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    query = tl.load(Packed + row * (3 * R) + k, k < R, 0).to(tl.float32)
    prev = tl.maximum(row - 1, 0)
    sender = tl.load(Packed + prev * (3 * R) + R + k, k < R, 0).to(tl.float32)
    value = tl.load(Packed + prev * (3 * R) + 2 * R + k, k < R, 0).to(tl.float32)
    bias = tl.load(Bias + k, k < R, 0).to(tl.float32)
    gate = tl.sigmoid(query + sender + bias)
    message = tl.where(row % T == 0, 0.0, gate * value)
    tl.store(Message + row * R + k, message, k < R)


@triton.jit
def feature_finish(
    Raw,
    Delta,
    RMS,
    Valid,
    Weight,
    Output,
    Normed,
    H: tl.constexpr,
    T: tl.constexpr,
    WRITE: tl.constexpr,
    NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    raw = tl.load(Raw + row * H + k, k < H, 0).to(tl.float32)
    delta = tl.load(Delta + row * H + k, k < H, 0).to(tl.float32)
    rms = tl.load(RMS + row)
    previous = tl.maximum(row - 1, 0)
    edge = (row % T != 0) & tl.load(Valid + row) & tl.load(Valid + previous)
    update = rms * delta * tl.cast(edge & WRITE, tl.float32)
    result = (raw + update).to(Output.dtype.element_ty)
    tl.store(Output + row * H + k, result, k < H)
    if NORM:
        x = result.to(tl.float32)
        variance = tl.sum(x * x, 0) / H
        y = (x * tl.rsqrt(variance + NORM_EPS)).to(Output.dtype.element_ty)
        weight = tl.load(Weight + k, k < H, 0)
        y = (y.to(tl.float32) * weight.to(tl.float32)).to(Output.dtype.element_ty)
        tl.store(Normed + row * H + k, y, k < H)


class FusedBridge:
    """Inject into TorchDspine.bridge; projections remain BF16 by default.

    Reduction order can change BF16 states. Acceptance against actual target
    trajectories must be remeasured even when isolated kernels pass parity.
    An optional linear callback may replace only the three bridge GEMMs.
    """
    def __init__(self,linear=None):
        self.linear=torch.nn.functional.linear if linear is None else linear

    @torch.no_grad()
    def __call__(self,hidden,residual,input_ids,read_weight,codebook,packed_weight,
                 gate_bias,write_weight,next_norm_weight,block_size,eps=1e-6,norm_eps=1e-6):
        Ref.require_bf16(hidden,residual,read_weight,codebook,packed_weight,gate_bias,write_weight,next_norm_weight)
        m,h=hidden.shape;c=codebook.shape[1];r=gate_bias.numel()
        Ref.whole_blocks(m,block_size)
        if (not hidden.is_cuda or residual.shape!=hidden.shape or input_ids.shape!=(m,)
                or input_ids.dtype!=torch.int64 or read_weight.shape!=(c,h)
                or packed_weight.shape!=(3*r,c) or write_weight.shape!=(h,r)
                or next_norm_weight.shape!=(h,)):
            raise ValueError('Unsupported fused bridge geometry')
        tensors=(hidden,residual,input_ids,read_weight,codebook,packed_weight,gate_bias,write_weight,next_norm_weight)
        if any(x.device!=hidden.device or not x.is_contiguous() for x in tensors):
            raise ValueError('Fused bridge tensors must be contiguous on one HIP device')
        raw=torch.empty_like(hidden);normalized=torch.empty_like(hidden)
        rms=torch.empty(m,device=hidden.device,dtype=torch.float32)
        prepare_read[(m,)](hidden,residual,raw,normalized,rms,h,eps,True,triton.next_power_of_2(h),enable_fp_fusion=False)
        projected=self.linear(normalized,read_weight)
        anchors=input_ids.reshape(-1,block_size)[:,0].contiguous()
        codes=torch.empty((m,c),device=hidden.device,dtype=torch.bfloat16)
        normalize_codes[(m,)](projected,codebook,anchors,codes,c,block_size,True,eps,triton.next_power_of_2(c),enable_fp_fusion=False)
        packed=self.linear(codes,packed_weight)
        message=torch.empty((m,r),device=hidden.device,dtype=torch.bfloat16)
        feature_message[(m,)](packed,gate_bias,message,r,block_size,triton.next_power_of_2(r),enable_fp_fusion=False)
        delta=self.linear(message,write_weight)
        valid=torch.ones(m,device=hidden.device,dtype=torch.bool)
        output=torch.empty_like(hidden);normed=torch.empty_like(hidden)
        feature_finish[(m,)](raw,delta,rms,valid,next_norm_weight,output,normed,h,block_size,True,True,norm_eps,triton.next_power_of_2(h),enable_fp_fusion=False)
        return normed,output,{'raw':raw,'rms':rms,'packed':packed}
