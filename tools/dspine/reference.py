"""PyTorch numerical reference for Tencent DSpine's bridge and greedy tail.

Arithmetic derived from Tencent/DSpine, Apache-2.0, commit
0cc92d857a8f7bf252de67995ad4c19251620143, kernels/dspine.py.
No SGLang/Triton dependency. This is a correctness reference, not a serving backend.
CPU tests permit small dimensions; the optimized upstream kernels require tiled sizes.
Sampling is deliberately outside this greedy reference's contract.
"""
import math
import torch


def require_bf16(*tensors):
    if any(x.dtype != torch.bfloat16 for x in tensors):
        raise ValueError("DSpine bridge requires BF16 tensors")


def whole_blocks(rows, block_size):
    if block_size < 2 or rows % block_size:
        raise ValueError("DSpine requires whole request blocks of at least two rows")


@torch.no_grad()
def prepare_read(hidden, residual=None, eps=1e-6):
    require_bf16(hidden)
    if hidden.ndim != 2:
        raise ValueError("hidden must be [rows, hidden]")
    if residual is not None:
        require_bf16(residual)
        if residual.shape != hidden.shape:
            raise ValueError("residual shape mismatch")
        raw = (hidden.float() + residual.float()).to(torch.bfloat16)
    else:
        raw = hidden.clone()
    x = raw.float()
    rms = torch.sqrt(torch.clamp(x.square().mean(-1), min=torch.finfo(torch.float32).tiny))
    normalized = (x / (rms[:, None] + eps)).to(torch.bfloat16)
    return raw, normalized, rms


@torch.no_grad()
def normalize_codes(projected, codebook, anchors, block_size, eps=1e-6):
    require_bf16(projected, codebook)
    m, c = projected.shape
    whole_blocks(m, block_size)
    if codebook.ndim != 2 or codebook.shape[1] != c or anchors.numel() != m // block_size:
        raise ValueError("codebook/anchor shape mismatch")
    z = projected.float()
    result = (math.sqrt(c) * z / (torch.linalg.vector_norm(z, dim=-1)[:, None] + eps)).to(torch.bfloat16)
    result[::block_size] = (codebook[anchors.reshape(-1)].float() * math.sqrt(c)).to(torch.bfloat16)
    return result


def pack_bridge_weights(gate, predecessor_gate, prev):
    require_bf16(gate, predecessor_gate, prev)
    if gate.ndim != 2 or gate.shape != predecessor_gate.shape or gate.shape != prev.shape:
        raise ValueError("bridge projection shapes must match")
    return torch.cat((gate, predecessor_gate, prev), dim=0).contiguous()


@torch.no_grad()
def feature_message(packed, gate_bias, block_size):
    require_bf16(packed, gate_bias)
    m, width = packed.shape
    whole_blocks(m, block_size)
    r = gate_bias.numel()
    if width != 3 * r:
        raise ValueError("packed columns must be gate | predecessor gate | value")
    previous = packed.roll(1, dims=0).float()
    gate = torch.sigmoid(packed[:, :r].float() + previous[:, r:2*r] + gate_bias.float())
    message = (gate * previous[:, 2*r:]).to(torch.bfloat16)
    message[::block_size] = 0
    return message


@torch.no_grad()
def feature_finish(raw, delta, rms, valid, block_size, norm_weight=None, norm_eps=1e-6, write=True):
    require_bf16(raw, delta)
    whole_blocks(raw.shape[0], block_size)
    if delta.shape != raw.shape or rms.shape != (raw.shape[0],) or valid.shape != rms.shape:
        raise ValueError("feature write shape mismatch")
    edge = valid.bool() & valid.roll(1).bool()
    edge[::block_size] = False
    update = rms[:, None].float() * delta.float() * (edge & write)[:, None].float()
    output = (raw.float() + update).to(torch.bfloat16)
    if norm_weight is None:
        return output, None
    require_bf16(norm_weight)
    x = output.float()
    # The upstream kernel rounds once before and once after multiplying the norm weight.
    y = (x * torch.rsqrt(x.square().mean(-1)[:, None] + norm_eps)).to(torch.bfloat16)
    normed = (y.float() * norm_weight.float()).to(torch.bfloat16)
    return output, normed


@torch.no_grad()
def bridge_layer(hidden, residual, input_ids, read_weight, codebook, packed_weight,
                 gate_bias, write_weight, next_norm_weight, block_size, eps=1e-6, norm_eps=1e-6):
    raw, normalized, rms = prepare_read(hidden, residual, eps)
    projected = torch.nn.functional.linear(normalized, read_weight)
    anchors = input_ids.reshape(-1, block_size)[:, 0]
    codes = normalize_codes(projected, codebook, anchors, block_size, eps)
    packed = torch.nn.functional.linear(codes, packed_weight)
    message = feature_message(packed, gate_bias, block_size)
    delta = torch.nn.functional.linear(message, write_weight)
    valid = torch.ones(raw.shape[0], dtype=torch.bool, device=raw.device)
    output, normed = feature_finish(raw, delta, rms, valid, block_size, next_norm_weight, norm_eps)
    return normed, output, {"raw": raw, "rms": rms, "packed": packed}


@torch.no_grad()
def build_serial_tables(lm_head, norm_weight, write_weight, predecessor_value_weight, codebook, chunk=8192):
    require_bf16(lm_head, norm_weight, write_weight, predecessor_value_weight, codebook)
    v, h = lm_head.shape
    r = write_weight.shape[1]
    c = codebook.shape[1]
    if (write_weight.shape != (h, r) or predecessor_value_weight.shape != (2*r, c)
            or norm_weight.shape != (h,) or codebook.shape[0] != v or chunk < 1):
        raise ValueError("serial table shape contract violated")
    tg = torch.empty((v, r), dtype=torch.bfloat16, device=lm_head.device)
    tsv = torch.empty((v, 2*r), dtype=torch.bfloat16, device=lm_head.device)
    for start in range(0, v, chunk):
        end = min(start + chunk, v)
        weighted = (lm_head[start:end].float() * norm_weight.float()).to(torch.bfloat16)
        tg[start:end] = weighted @ write_weight
        codes = (math.sqrt(c) * codebook[start:end].float()).to(torch.bfloat16)
        tsv[start:end] = torch.nn.functional.linear(codes, predecessor_value_weight)
    return tg, tsv


@torch.no_grad()
def serial_table(raw, norm_weight, candidates, input_ids, lm_head, tg, tsv, packed,
                 rms, gate_bias, block_size):
    require_bf16(raw, packed, lm_head, norm_weight, tg, tsv)
    m, h = raw.shape
    whole_blocks(m, block_size)
    n, k = candidates.shape
    r = tg.shape[1]
    if (n != m//block_size*(block_size-1) or input_ids.numel() != m or rms.shape != (m,)
            or tsv.shape != (tg.shape[0], 2*r) or packed.shape[0] != m or packed.shape[1] < r
            or gate_bias.numel() != r or norm_weight.shape != (h,) or lm_head.shape[1] != h):
        raise ValueError("serial table state shape mismatch")
    if candidates.dtype != torch.int64 or input_ids.dtype != torch.int64:
        raise ValueError("candidate and input ids must be int64")
    if k < 2 or k > 64 or k & (k-1) or bool((candidates[:, 1:] < candidates[:, :-1]).any()):
        raise ValueError("candidate lanes must be sorted by id; K is a power of two in [2,64]")
    idx = torch.empty_like(candidates)
    confidence = torch.empty((n, k), dtype=torch.float32, device=raw.device)
    ids_flat = input_ids.reshape(-1)
    for at in range(n):
        b, pos = divmod(at, block_size-1)
        row = b*block_size + 1 + pos
        ids = candidates[at]
        if pos == 0:
            previous = ids_flat[b*block_size].expand(k)
        else:
            previous = candidates[at-1]
        sender = tsv[previous].float()
        message = torch.sigmoid(packed[row, :r].float() + sender[:, :r] + gate_bias.float()) * sender[:, r:]
        base = (lm_head[ids].float() * (raw[row].float() * norm_weight.float())).sum(-1)
        score = base[None, :] + rms[row].float() * (message @ tg[ids].float().T)
        best, idx[at] = torch.max(score, dim=1)
        scale = torch.clamp(rms[row].float(), min=1e-6)
        confidence[at] = 1.0 / torch.exp((score-best[:, None])/scale).sum(-1)
    return idx, confidence


@torch.no_grad()
def serial_chain(idx, candidates, block_size, confidence=None):
    n, k = candidates.shape
    if block_size < 2 or n % (block_size-1) or idx.shape != candidates.shape:
        raise ValueError("greedy chain requires whole request blocks")
    output = torch.empty((n,), dtype=torch.int64, device=candidates.device)
    selected_confidence = torch.empty((n,), dtype=torch.float32, device=candidates.device) if confidence is not None else None
    for b in range(n//(block_size-1)):
        previous = 0
        for pos in range(block_size-1):
            row = b*(block_size-1)+pos
            lane = int(idx[row, previous])
            output[row] = candidates[row, lane]
            if confidence is not None:
                selected_confidence[row] = confidence[row, previous]
            previous = lane
    return output, selected_confidence
