"""Fixed 15-node best-first selection over DSpine's 15 x 16 x 16 lattice."""
import torch
import triton
import triton.language as tl


@triton.jit
def _select(Candidates, LogP, Chain, Output):
    # Check the complete lattice, including conditionals that the tree never visits.
    row = tl.arange(0, 256)
    lane = tl.arange(0, 16)
    ids = tl.load(Candidates + row, mask=row < 240, other=0)
    previous = tl.load(Candidates + tl.maximum(row - 1, 0), mask=row < 240, other=0)
    bad_ids = tl.sum(((row < 240) & ((ids < 0) | ((row % 16 != 0) & (ids <= previous)))).to(tl.int32), 0) != 0
    conditional = tl.load(LogP + row[:, None] * 16 + lane[None, :],
                          mask=row[:, None] < 240, other=-float("inf"))
    bad_values = tl.sum(((conditional != conditional) | (conditional > 0)).to(tl.int32), 1) != 0
    mass = tl.sum(tl.exp(conditional.to(tl.float64)), 1)
    bad_logp = tl.sum(((row < 240) & (bad_values | (tl.abs(mass - 1.) > 1.e-4))).to(tl.int32), 0) != 0
    status = bad_ids.to(tl.int32) | (bad_logp.to(tl.int32) * 2)

    # Slot parent*16+lane is a unique heap entry. Scores are sums of FP32
    # conditional log probabilities widened before addition, as in Python.
    slot = tl.arange(0, 256)
    owner = slot // 16
    child = slot % 16
    scores = tl.load(LogP + child, mask=owner == 0, other=-float("inf")).to(tl.float64)
    tokens = tl.load(Candidates + child, mask=owner == 0, other=0)
    depths = tl.full((256,), 1, tl.int32)
    for step in range(15):
        best = tl.max(scores, 0)
        live = best > -float("inf")
        eligible = (scores == best) & live
        depth = tl.min(tl.where(eligible, depths, 16), 0)
        eligible = eligible & (depths == depth)
        token = tl.min(tl.where(eligible, tokens, 0x7fffffffffffffff), 0)
        winner = tl.min(tl.where(eligible & (tokens == token), slot, 256), 0)
        parent = winner // 16
        selected_lane = winner % 16
        status = status | ((~live).to(tl.int32) * 4)
        out = Output + step * 6
        tl.store(out, token)
        tl.store(out + 1, parent)
        tl.store(out + 2, depth)
        tl.store(out + 3, selected_lane)
        tl.store(out + 4, best.to(tl.int64, bitcast=True))
        tl.store(out + 5, tl.load(Chain + step))

        expand = (owner == step + 1) & (depth < 15) & live
        next_logp = tl.load(LogP + (depth * 16 + selected_lane) * 16 + child,
                            mask=expand, other=-float("inf")).to(tl.float64)
        next_tokens = tl.load(Candidates + depth * 16 + child, mask=expand, other=0)
        scores = tl.where(expand, best + next_logp, tl.where(slot == winner, -float("inf"), scores))
        tokens = tl.where(expand, next_tokens, tokens)
        depths = tl.where(expand, depth + 1, depths)
    # The host reads precisely one small buffer per proposal, including the chain and error mask.
    fields = tl.arange(0, 8)
    tl.store(Output + 15 * 6 + fields, tl.where(fields == 0, status, 0), mask=fields < 6)


def select(candidates, logp, chain):
    """Return device int64 rows (token, parent, depth, lane, FP64 score bits, chain token)."""
    if (candidates.shape != (15, 16) or logp.shape != (15, 16, 16) or chain.shape != (15,)
            or candidates.dtype != torch.int64 or logp.dtype != torch.float32 or chain.dtype != torch.int64):
        raise ValueError("GPU tree requires int64 candidates[15,16], FP32 logp[15,16,16] and int64 chain[15]")
    if not candidates.is_cuda or any(x.device != candidates.device or not x.is_contiguous()
                                     for x in (candidates, logp, chain)):
        raise ValueError("GPU tree tensors must be contiguous on one device")
    output = torch.empty((16, 6), device=candidates.device, dtype=torch.int64)
    _select[(1,)](candidates, logp, chain, output, num_warps=4, enable_fp_fusion=False)
    return output


def read_rows(output):
    rows = output.cpu().tolist()
    status = rows[-1][0]
    if status:
        errors = [message for flag, message in
                  [(1, "invalid candidate IDs"), (2, "invalid conditional probabilities"), (4, "fewer than 15 finite nodes")]
                  if status & flag]
        raise ValueError("Invalid DSpine GPU tree: " + ", ".join(errors))
    return rows[:15]


def wire_result(candidates, logp, chain, width):
    rows = read_rows(select(candidates, logp, chain))
    tree = [{"token": r[0], "parent": None if r[1] == 0 else r[1] - 1} for r in rows]
    return [r[5] for r in rows[:width]], tree
