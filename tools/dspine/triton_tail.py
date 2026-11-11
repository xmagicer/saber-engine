"""Experimental GPU serial refinement and proposer transition probabilities.

Adapted from Tencent/DSpine, Apache-2.0, commit
0cc92d857a8f7bf252de67995ad4c19251620143,
sglang_dspine/kernels/dspine.py. Copyright (C) 2026 Tencent.
See LICENSE and NOTICE. The optional log-probability output is a local extension;
it never consumes target future tokens. This is not a serving backend.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _serial_table(
    Raw,
    stride_raw,
    NormW,
    Cand,
    InputIds,
    LmW,
    stride_w,
    TG,
    TSV,
    Query,
    stride_q,
    Rms,
    GateB,
    Out,
    Confidence,
    LogP,
    H: tl.constexpr,
    RK: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TM1: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
    WITH_LOGP: tl.constexpr,
):
    n = tl.program_id(0)
    b = n // TM1
    r = n % TM1
    row = b * (TM1 + 1) + 1 + r
    kp = tl.arange(0, KP)
    live = kp < K
    ids = tl.load(Cand + n * K + kp, mask=live, other=0)
    prev = tl.load(Cand + tl.maximum(n - 1, 0) * K + kp, mask=live, other=0)
    anchor = tl.load(InputIds + b * (TM1 + 1)).to(tl.int64)
    sid = tl.where(r == 0, anchor, prev)
    base = tl.zeros([KP], dtype=tl.float32)
    for h0 in tl.static_range(0, H, BH):
        off = h0 + tl.arange(0, BH)
        u = tl.load(Raw + row * stride_raw + off).to(tl.float32) * tl.load(
            NormW + off
        ).to(tl.float32)
        w = tl.load(
            LmW + ids[:, None] * stride_w + off[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        base += tl.sum(w * u[None, :], axis=1)
    s = tl.zeros([KP, KP], dtype=tl.float32)
    for r0 in tl.static_range(0, RK, BR):
        rr = r0 + tl.arange(0, BR)
        gate = tl.load(
            TSV + sid[:, None] * (2 * RK) + rr[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        val = tl.load(
            TSV + sid[:, None] * (2 * RK) + RK + rr[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        q = tl.load(Query + row * stride_q + rr).to(tl.float32)
        gb = tl.load(GateB + rr).to(tl.float32)
        m = tl.sigmoid(q[None, :] + gate + gb[None, :]) * val
        g = tl.load(
            TG + ids[:, None] * RK + rr[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        s += tl.dot(m, tl.trans(g), input_precision="ieee")
    base = tl.where(live, base, -1e30)
    s = base[None, :] + tl.load(Rms + row).to(tl.float32) * s
    (best_value, best) = tl.max(
        s, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(Out + n * K + kp, best, mask=live)
    if WITH_CONFIDENCE or WITH_LOGP:
        scale = tl.maximum(tl.load(Rms + row).to(tl.float32), 1e-06)
        normalized = (s - best_value[:, None]) / scale
        mass = tl.sum(tl.exp(normalized), 1)
        if WITH_CONFIDENCE:
            tl.store(Confidence + n * K + kp, 1.0 / mass, mask=live)
        if WITH_LOGP:
            logp = normalized - tl.log(mass)[:, None]
            offset = n * K * K + kp[:, None] * K + kp[None, :]
            tl.store(LogP + offset, logp, mask=live[:, None] & live[None, :])


@triton.jit
def _serial_chain(
    Idx,
    Cand,
    Out,
    Confidence,
    ConfidenceOut,
    K: tl.constexpr,
    TM1: tl.constexpr,
    RPAD: tl.constexpr,
    WITH_CONFIDENCE: tl.constexpr,
):
    b = tl.program_id(0)
    rr = tl.arange(0, RPAD)
    kk = tl.arange(0, K)
    inside = rr[:, None] < TM1
    at = (b * TM1 + rr[:, None]) * K + kk[None, :]
    tab = tl.load(Idx + at, mask=inside, other=0)
    cnd = tl.load(Cand + at, mask=inside, other=0)
    if WITH_CONFIDENCE:
        probs = tl.load(Confidence + at, mask=inside, other=0.0)
    j = tl.sum(tl.where((rr[:, None] == 0) & (kk[None, :] == 0), tab, 0))
    previous = tl.full((), 0, tl.int64)
    for r in tl.static_range(TM1):
        if WITH_CONFIDENCE:
            probability = tl.sum(
                tl.where((rr[:, None] == r) & (kk[None, :] == previous), probs, 0.0)
            )
            tl.store(ConfidenceOut + b * TM1 + r, probability)
        tl.store(
            Out + b * TM1 + r,
            tl.sum(tl.where((rr[:, None] == r) & (kk[None, :] == j), cnd, 0)),
        )
        previous = j
        j = tl.sum(tl.where((rr[:, None] == r + 1) & (kk[None, :] == j), tab, 0))


def build_serial_tables(lm_head_w, norm_w, write_w, pv_w, codebook, chunk: int = 8192):
    """(TG [V, R] bf16, TSV [V, 2R] bf16).  lm_head_w [V, H], norm_w [H], write_w [H, R] (the last
    layer's write), pv_w [2R, C] = [G_p; A], codebook [V, C].
      TG[v]  = bf16(LmHead[v] * norm_w) @ W_last
      TSV[v] = [G_p; A] @ bf16(sqrt(C) * Codebook[v])"""
    (v, h) = lm_head_w.shape
    r = write_w.shape[1]
    c = codebook.shape[1]
    if (
        codebook.shape[0] != v
        or tuple(write_w.shape) != (h, r)
        or tuple(pv_w.shape) != (2 * r, c)
        or (norm_w.shape != (h,))
    ):
        raise ValueError("build_serial_tables: shape contract violated")
    tg = torch.empty((v, r), device=lm_head_w.device, dtype=torch.bfloat16)
    tsv = torch.empty((v, 2 * r), device=lm_head_w.device, dtype=torch.bfloat16)
    (nwf, wb, scale) = (norm_w.float(), write_w.to(torch.bfloat16), float(c) ** 0.5)
    for s in range(0, v, chunk):
        e = min(s + chunk, v)
        tg[s:e] = torch.matmul((lm_head_w[s:e].float() * nwf).to(torch.bfloat16), wb)
        tsv[s:e] = torch.nn.functional.linear(
            (scale * codebook[s:e].float()).to(torch.bfloat16), pv_w
        )
    return (tg, tsv)


def serial_table(
    raw,
    norm_w,
    cand,
    input_ids,
    lm_head_w,
    tg,
    tsv,
    packed,
    rms,
    gate_b,
    block_size: int,
    confidence=None,
    with_log_probabilities=False,
):
    """raw [bs*T, H] bf16 pre-write state of the last write; cand int64 [bs*(T-1), K] ascending;
    input_ids [bs*T]; packed [bs*T, >=R] bf16 with the receiver query in columns :R; rms fp32
    [bs*T]  ->  idx int64 [bs*(T-1), K]: idx[n, j] = the best candidate lane of row n when the
    previous row chose lane j (all j equal on block row 1, whose sender is the anchor)."""
    t = int(block_size)
    (m, h) = raw.shape
    (n, k) = cand.shape
    rk = tg.shape[1]
    if m % t or n != m // t * (t - 1) or input_ids.numel() != m or (rms.numel() != m):
        raise ValueError(
            f"serial_table row mismatch: raw {raw.shape}, cand {cand.shape}, T={t}"
        )
    if (
        raw.dtype != torch.bfloat16
        or raw.stride(1) != 1
        or packed.dtype != torch.bfloat16
        or (packed.stride(1) != 1)
    ):
        raise ValueError("serial_table expects row-contiguous BF16 state and query")
    if (
        packed.shape[0] != m
        or packed.shape[1] < rk
        or rms.dtype != torch.float32
        or (not rms.is_contiguous())
    ):
        raise ValueError("serial_table query/rms contract violated")
    if (
        not cand.is_contiguous()
        or cand.dtype != torch.int64
        or input_ids.dtype != torch.int64
        or (not input_ids.is_contiguous())
    ):
        raise ValueError(
            "serial_table expects contiguous int64 candidates and input ids"
        )
    if (
        lm_head_w.stride(1) != 1
        or tuple(tsv.shape) != (tg.shape[0], 2 * rk)
        or (not (tg.is_contiguous() and tsv.is_contiguous()))
    ):
        raise ValueError("serial_table table layout mismatch")
    if (
        h % 512
        or rk % 128
        or k & k - 1
        or (not 2 <= k <= 64)
        or (gate_b.numel() != rk)
        or (norm_w.numel() != h)
    ):
        raise ValueError("serial_table size contract violated")
    idx = torch.empty((n, k), dtype=torch.int64, device=raw.device)
    if confidence is not None and (
        tuple(confidence.shape) != (n, k)
        or confidence.dtype != torch.float32
        or (not confidence.is_contiguous())
    ):
        raise ValueError("Confidence output must be contiguous float32 [N, K]")
    logp = torch.empty((n,k,k),dtype=torch.float32,device=raw.device) if with_log_probabilities else None
    if n:
        _serial_table[n,](
            raw,
            raw.stride(0),
            norm_w,
            cand,
            input_ids,
            lm_head_w,
            lm_head_w.stride(0),
            tg,
            tsv,
            packed,
            packed.stride(0),
            rms,
            gate_b,
            idx,
            confidence if confidence is not None else idx,
            logp if logp is not None else idx,
            H=h,
            RK=rk,
            K=k,
            KP=max(k, 16),
            TM1=t - 1,
            BH=512,
            BR=128,
            WITH_CONFIDENCE=confidence is not None,
            WITH_LOGP=with_log_probabilities,
            num_warps=8,
            num_stages=2,
        )
    return (idx,logp) if with_log_probabilities else idx


def serial_chain(idx, cand, out, block_size: int, confidence=None, confidence_out=None):
    """Walk every request's greedy table: out[:bs*(T-1)] written in place (int64)."""
    t = int(block_size)
    (n, k) = cand.shape
    if (
        n % (t - 1)
        or tuple(idx.shape) != (n, k)
        or (not (idx.is_contiguous() and cand.is_contiguous()))
    ):
        raise ValueError("serial_chain layout mismatch")
    if out.dtype != torch.int64 or out.numel() < n or idx.dtype != torch.int64:
        raise ValueError("serial_chain dtype/capacity contract violated")
    if confidence is not None and (
        confidence_out is None
        or confidence_out.dtype != torch.float32
        or confidence_out.numel() < n
        or (tuple(confidence.shape) != (n, k))
    ):
        raise ValueError(
            "Serial confidence needs a float32 [N, K] table and [N] output"
        )
    if n:
        _serial_chain[n // (t - 1),](
            idx,
            cand,
            out,
            confidence if confidence is not None else idx,
            confidence_out if confidence_out is not None else out,
            K=k,
            TM1=t - 1,
            RPAD=triton.next_power_of_2(t - 1),
            WITH_CONFIDENCE=confidence is not None,
            num_warps=1,
        )
    return out

