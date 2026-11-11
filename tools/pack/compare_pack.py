#!/usr/bin/env python3
"""Compare two device-final packs record by record (``build_pack.py``'s output vs a reference).

  tools/pack/compare_pack.py <pack_dir> <reference_dir> [--model DIR] [--plan-json PLAN]

Per rank, every record of the reference is classified, as ``check_devfinal.py`` does:

* **identical** -- same name, layout, dtype, shape, shard kind, offset, length and sha256;
* **near-tie** -- an MXFP4 trio (``w`` / ``ws`` / ``wref`` of one op) whose only differences are
  whole 32-wide blocks with a block exponent one apart (the SBRPACK1 format § "Building the
  device-final tier": the ``mse`` exponent is not reproducible across reduction orders). The
  signature is exact: every differing ``w`` block is a block whose ``ws`` byte differs, by 1;
  ``wref`` differs only on a row holding such a block and equals that row's max of ``ws``; and at
  most 1 % of the record's blocks move. With ``--model`` + ``--plan-json`` each moved block is
  also re-scored from the checkpoint in f64: both packs' exponents must be the block's two ``mse``
  candidates and their errors must agree to 1e-6 relative (the near-tie, measured);
* **DIFFERS** -- anything else, including any metadata difference.

The header's plan hash, tier id, tp size, rank, record count and geometry must match too. Exit 1
on any DIFFERS.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pack as bp  # noqa: E402

META_KEYS = ("dtype", "shape", "shard_kind", "layout_id", "data_off", "length", "sources")
NEAR_TIE_REL = 1e-6


def header(path: str) -> dict:
    with open(path, "rb") as f:
        h = f.read(256)
    cnt = struct.unpack_from("<I", h, 20)[0]
    to, nbo, nbb, do, fb = struct.unpack_from("<QQQQQ", h, 24)
    tp, rank = struct.unpack_from("<II", h, 64)
    cbo, cbb = struct.unpack_from("<QQ", h, 176)
    return {"magic": h[:8], "version": struct.unpack_from("<I", h, 8)[0], "count": cnt,
            "table_off": to, "name_blob_off": nbo, "name_blob_bytes": nbb, "data_off": do,
            "file_bytes": fb, "tp": tp, "rank": rank, "plan_hash": h[80:112].hex(),
            "tier_id": h[112:144].hex(), "chunk_blob_off": cbo, "chunk_blob_bytes": cbb,
            "chunk_bytes": struct.unpack_from("<I", h, 224)[0], "reserved": h[228:]}


def read(fd: int, r: dict) -> np.ndarray:
    b = os.pread(fd, r["length"], r["data_off"])
    if len(b) != r["length"]:
        raise SystemExit(f"{r['name']}: short read")
    return np.frombuffer(b, np.uint8)


class Rescorer:
    """f64 errors of a block's two ``mse`` candidates, from the checkpoint (``--model``)."""

    def __init__(self, model: str, plan_json: str, tp: int):
        with open(plan_json) as f:
            plan = json.load(f)
        files = sorted(os.path.join(model, f) for f in os.listdir(model)
                       if f.endswith(".safetensors"))
        tl, _ = bp.collect_tensors(files)
        with open(os.path.join(model, "config.json")) as f:
            c = json.load(f)
        bp._G.update({"tensors": {t.name: t for t in tl}, "cfg": c.get("text_config", c),
                      "fds": {}, "tp": tp})
        self.w_of = {r["op"]: r for r in plan["records"]
                     if r["layout_id"] in bp.MXFP4_W}
        self.tp = tp

    def rows(self, op: str, rank: int) -> np.ndarray:
        """The rank's f32 source rows of an op, in the record's row order."""
        wrec = self.w_of[op]
        multi = len(wrec["sources"]) > 1
        out = []
        for src in wrec["sources"]:
            name = src["weight"]
            kind = bp.policy_kind(name) if multi else wrec["shard_kind"]
            for a, b in bp.row_pieces(name, kind, bp._G["tensors"][name].shape[0], rank, self.tp):
                out.append(bp.dequant_rows(src, kind, a, b, rank, self.tp))
        return np.concatenate(out)

    def score(self, x: np.ndarray, rows, blocks, e_a, e_b) -> float:
        """max relative f64 error gap between the two packs' exponents over the given blocks."""
        worst = 0.0
        for r, kb, ea, eb in zip(rows, blocks, e_a, e_b):
            v = x[r, kb * 32:(kb + 1) * 32].astype(np.float64)
            amax = float(np.abs(v).max())
            e0 = int(np.floor(np.log2(amax))) - 2 if amax > 0 else -127
            if sorted((int(ea), int(eb))) != [e0, e0 + 1]:
                return float("inf")
            errs = []
            for e in (ea, eb):
                s = 2.0 ** int(e)
                c = bp.e2m1_codes(np.abs(v / s).astype(np.float32))
                q = bp.E2M1[c].astype(np.float64) * s * np.sign(v)
                errs.append(float(((q - v) ** 2).sum()))
            worst = max(worst, abs(errs[0] - errs[1]) / max(errs[0], errs[1], 1e-300))
        return worst


def near_tie(op_recs, fd_a, fd_b, resc, rank):
    """(near_tie_ok, moved_blocks, total_blocks, why, worst_rel) for one op's trio."""
    w, ws, wref = op_recs["w"], op_recs["ws"], op_recs["wref"]
    n, k = w[0]["shape"]
    kb = k // 32
    sa = read(fd_a, ws[0]).reshape(kb, n).T.astype(np.int32)
    sb = read(fd_b, ws[1]).reshape(kb, n).T.astype(np.int32)
    dws = sa != sb
    if np.any(np.abs(sa - sb)[dws] != 1):
        return False, int(dws.sum()), n * kb, "a block exponent moved by more than 1", None
    wa = bp.unpermute_w(read(fd_a, w[0]), n, k).reshape(n, kb, 16)
    wb = bp.unpermute_w(read(fd_b, w[1]), n, k).reshape(n, kb, 16)
    dw = (wa != wb).any(axis=-1)
    if np.any(dw & ~dws):
        return False, int(dw.sum()), n * kb, "w moved in a block whose exponent did not", None
    ra, rb = read(fd_a, wref[0]), read(fd_b, wref[1])
    if not (np.array_equal(ra, sa.max(axis=1)) and np.array_equal(rb, sb.max(axis=1))):
        return False, int(dws.sum()), n * kb, "wref is not the row max of ws", None
    moved = int(dws.sum())
    if moved > 0.01 * n * kb:
        return False, moved, n * kb, f"{moved} of {n * kb} blocks moved (> 1 %)", None
    worst = None
    if resc is not None and moved:
        x = resc.rows(w[0]["op"], rank)
        rr, bb = np.nonzero(dws)
        worst = resc.score(x, rr, bb, sa[rr, bb] - 127, sb[rr, bb] - 127)
        if not worst <= NEAR_TIE_REL:
            return False, moved, n * kb, f"re-scored gap {worst:.3g} > {NEAR_TIE_REL}", worst
    return True, moved, n * kb, "", worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("reference")
    ap.add_argument("--model", default=None)
    ap.add_argument("--plan-json", default=None)
    ap.add_argument("--ranks", default=None, help="comma list (default: all)")
    a = ap.parse_args()
    ma = json.load(open(os.path.join(a.pack, "manifest.json")))
    mb = json.load(open(os.path.join(a.reference, "manifest.json")))
    bad = 0
    for key in ("tp_size",):
        if ma[key] != mb[key]:
            print(f"DIFFERS manifest {key}: {ma[key]} vs {mb[key]}")
            bad += 1
    if ma["tier"] != mb["tier"]:
        print("DIFFERS manifest tier (id or descriptor)")
        bad += 1
    if ma["plan"] != mb["plan"]:
        print("DIFFERS manifest plan (hash or descriptor)")
        bad += 1
    if ma["shard_policy"] != mb["shard_policy"]:
        print("DIFFERS manifest shard_policy")
        bad += 1
    partial = "only" in ma
    resc = (Rescorer(a.model, a.plan_json, ma["tp_size"])
            if a.model and a.plan_json else None)
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else range(ma["tp_size"])
    print(f"{'rank':>4} {'records':>8} {'identical':>9} {'near-tie':>8} {'DIFFERS':>7} "
          f"{'blocks moved':>14} {'of blocks':>12} {'worst rel':>10}")
    for rank in ranks:
        fa = os.path.join(a.pack, f"rank{rank}.pack")
        fb = os.path.join(a.reference, f"rank{rank}.pack")
        ha, hb = header(fa), header(fb)
        if not partial:
            for k in ha:
                if ha[k] != hb[k]:
                    print(f"DIFFERS rank {rank} header {k}: {ha[k]} vs {hb[k]}")
                    bad += 1
            if os.path.getsize(fa) != os.path.getsize(fb):
                print(f"DIFFERS rank {rank} file size")
                bad += 1
        else:
            for k in ("plan_hash", "tier_id", "tp", "rank"):
                if ha[k] != hb[k]:
                    print(f"DIFFERS rank {rank} header {k}")
                    bad += 1
        ra = {r["name"]: r for r in ma["ranks"][rank]["records"]}
        rb = {r["name"]: r for r in mb["ranks"][rank]["records"]}
        names = list(ra) if partial else list(rb)
        if not partial and list(ra) != list(rb):
            print(f"DIFFERS rank {rank}: record order / set")
            bad += 1
        fd_a, fd_b = os.open(fa, os.O_RDONLY), os.open(fb, os.O_RDONLY)
        ident = near = diff = 0
        moved = total = 0
        worst = 0.0
        ops = {}
        for nm in names:
            x, y = ra.get(nm), rb.get(nm)
            if y is None:
                print(f"DIFFERS rank {rank} {nm}: not in the reference")
                diff += 1
                continue
            keys = [k for k in META_KEYS if not (partial and k == "data_off")]
            if any(x[k] != y[k] for k in keys):
                print(f"DIFFERS rank {rank} {nm}: metadata "
                      f"{[(k, x[k], y[k]) for k in keys if x[k] != y[k]][:2]}")
                diff += 1
                continue
            if x["sha256"] == y["sha256"]:
                ident += 1
                continue
            if x["layout_id"] in bp.MXFP4_TRIO:
                op, role = nm.rsplit(".", 1)
                ops.setdefault(op, {})[role] = (dict(x, op=op), y)
                continue
            print(f"DIFFERS rank {rank} {nm}: {x['layout']} bytes, no near-tie form")
            diff += 1
        for op, trio in ops.items():
            full = {}
            for role in ("w", "ws", "wref"):
                if role in trio:
                    full[role] = trio[role]
                else:  # an identical member, read for the signature
                    full[role] = (dict(ra[f"{op}.{role}"], op=op), rb[f"{op}.{role}"])
            ok, mv, tot, why, wr = near_tie(full, fd_a, fd_b, resc, rank)
            if ok:
                near += len(trio)
                moved += mv
                total += tot
                if wr is not None:
                    worst = max(worst, wr)
            else:
                print(f"DIFFERS rank {rank} {op}: {why}")
                diff += len(trio)
        os.close(fd_a)
        os.close(fd_b)
        bad += diff
        print(f"{rank:>4} {len(names):>8} {ident:>9} {near:>8} {diff:>7} {moved:>14} "
              f"{total:>12} {worst if resc else float('nan'):>10.3g}")
    print("PASS" if bad == 0 else f"FAIL: {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
