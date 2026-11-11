#!/usr/bin/env python3
"""Check a Quark OCP-MXFP4 27B pack against its checkpoint with an INDEPENDENT dequant.

For every MXFP4 trio record (or ``--sample N`` of them): Quark's own dequant of the checkpoint
bytes (``flashnext_mx.quark_dequant``, the AMD reference port), sharded by the plan's policy,
must equal Saber's reading of the PACK bytes (``flashnext_mx.saber_dequant``) bit for bit, and
``wref`` must be the per-row max of ``ws``. ``lm_head.w`` must be the checkpoint's bf16 rows byte
for byte, every ``k_scale`` / ``v_scale`` 1.0. ``--control`` swaps the checkpoint's nibbles
(``nibble_swap_control``) and must FAIL every checked record.

    tools/pack/check_quark27b.py --model MODEL --pack PACKDIR --plan plan-quark.json [--sample 64]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pack as bp  # noqa: E402
import flashnext_mx as fm  # noqa: E402


def pack_bytes(pack: str, rank: int, rec: dict) -> bytes:
    with open(os.path.join(pack, f"rank{rank}.pack"), "rb") as f:
        f.seek(rec["data_off"])
        return f.read(rec["length"])


def expected_trio(w_rec: dict, rank: int, tp: int, control: bool) -> np.ndarray:
    T = bp._G["tensors"]
    rows = []
    multi = len(w_rec["sources"]) > 1
    for src in w_rec["sources"]:
        name = src["weight"]
        kind = bp.policy_kind(name) if multi else w_rec["shard_kind"]
        w, s = T[name], T[src["scale"]]
        k = w.shape[1] * 2
        c0, c1 = bp.col_range(name, kind, k, rank, tp, 32)
        for a, b in bp.row_pieces(name, kind, w.shape[0], rank, tp):
            p = bp.read_rows(w, a, b)[:, c0 // 2:c1 // 2]
            sc = bp.read_rows(s, a, b)[:, c0 // 32:c1 // 32]
            if control:
                p = fm.nibble_swap_control(p)
            rows.append(fm.quark_dequant(np.ascontiguousarray(p), np.ascontiguousarray(sc)))
    return np.concatenate(rows, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sample", type=int, default=0, help="check N random trios (0 = all)")
    ap.add_argument("--control", action="store_true", help="nibble-swapped checkpoint: must fail")
    a = ap.parse_args(argv)
    plan = json.load(open(a.plan))
    man = json.load(open(os.path.join(a.pack, "manifest.json")))
    tp = man["tp_size"]
    cfgj = json.load(open(os.path.join(a.model, "config.json")))
    files = sorted(os.path.join(a.model, f) for f in os.listdir(a.model)
                   if f.endswith(".safetensors"))
    tl, _ = bp.collect_tensors(files)
    bp._G.update({"tensors": {t.name: t for t in tl}, "cfg": cfgj.get("text_config", cfgj),
                  "fds": {}})
    by_rank = [{r["name"]: r for r in rk["records"]} for rk in man["ranks"]]
    trios = {}
    for r in plan["records"]:
        if r["layout_id"] in bp.MXFP4_TRIO:
            role = {bp.LAYOUT_MXFP4_BLOCK_SCALE: "ws", bp.LAYOUT_MXFP4_ROW_REF: "wref"}.get(
                r["layout_id"], "w")
            trios.setdefault(r["op"], {})[role] = r
    ops = sorted(o for o, g in trios.items() if g["w"]["name"] in by_rank[0])
    if a.sample and a.sample < len(ops):
        ops = sorted(random.Random(27).sample(ops, a.sample))
    bad = checked = 0
    for op in ops:
        g = trios[op]
        n, k = g["w"]["shape"]
        for rank in range(tp):
            wp = np.frombuffer(pack_bytes(a.pack, rank, by_rank[rank][g["w"]["name"]]), np.uint8)
            ws = np.frombuffer(pack_bytes(a.pack, rank, by_rank[rank][g["ws"]["name"]]),
                               np.uint8).reshape(k // 32, n)
            wref = np.frombuffer(pack_bytes(a.pack, rank, by_rank[rank][g["wref"]["name"]]),
                                 np.uint8)
            got = fm.saber_dequant(wp.reshape(n, k // 2), ws, n, k)
            want = expected_trio(g["w"], rank, tp, a.control)
            ok = fm.bits_equal(got, want) and np.array_equal(wref, ws.max(axis=0))
            checked += 1
            if not ok:
                bad += 1
                if not a.control:
                    print(f"MISMATCH {op} rank {rank}", file=sys.stderr)
    extra = []
    if not a.control and "lm_head.w" in by_rank[0]:
        T = bp._G["tensors"]
        for rank in range(tp):
            want = b"".join(bp.raw_pieces(T["lm_head.weight"], "Dim0", rank, tp))
            extra.append(("lm_head.w", rank,
                          pack_bytes(a.pack, rank, by_rank[rank]["lm_head.w"]) == want))
        for name in by_rank[0]:
            if name.endswith(("self_attn.k_scale", "self_attn.v_scale")):
                for rank in range(tp):
                    extra.append((name, rank,
                                  pack_bytes(a.pack, rank, by_rank[rank][name]) == b"\x80\x3f"))
    xbad = [e for e in extra if not e[2]]
    print(f"check_quark27b: {checked - bad}/{checked} trio x rank bit-equal"
          f"{' (CONTROL: nibble-swapped)' if a.control else ''}; "
          f"{len(extra) - len(xbad)}/{len(extra)} head / kv-scale records exact")
    if a.control:
        return 0 if bad == checked and checked else 1
    return 0 if bad == 0 and not xbad and checked else 1


if __name__ == "__main__":
    sys.exit(main())
