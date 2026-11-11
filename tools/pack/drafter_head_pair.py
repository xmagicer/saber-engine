#!/usr/bin/env python3
"""Pair a DFlash2 drafter pack with ANOTHER target checkpoint's bf16 head.

The drafter checkpoint (Qwen3.8-27B-DFlash2-FP8) has no embedding and no head: its pack carries
the TARGET's (``input_embedding.weight``, ``head.rerank.w`` and the int2 draft head
``head.int2.{wq,s,zs}`` quantised from it), and the engine's pairing law (``saber_engine::
head_share``) refuses a drafter whose ``head.rerank.w`` / ``input_embedding.weight`` sha256 differ
from the target pack's ``lm_head.w`` / embedding record. A target with a different head (Quark's
own bf16 ``lm_head``) therefore needs a drafter pack whose head records are THAT head.

This tool writes such a pack from two packs, no checkpoint read and no torch:

* every drafter record whose bytes come from the drafter checkpoint (and the embedding) is copied
  byte for byte from ``--drafter-pack`` (the bf16-rerank pack, ``qwen38-dflash2-drafter``) and its
  sha256 re-taken while it streams -- it must equal the source manifest's;
* ``head.rerank.w`` is the target pack's ``lm_head.w`` record of the same rank, byte for byte
  (its sha256 must equal the target manifest's), and ``input_embedding.weight`` must already
  equal the target's embedding (refused otherwise: this tool swaps the head, nothing else);
* ``head.int2.{wq,s,zs}`` are re-quantised from that head with the int2 draft-head algorithm
  (asymmetric 2-bit min/max over 128-wide groups, quarter-split packing, the scale and the zero
  premultiplied by 4 and rounded to bf16 -- the from-bf16-head path: the quantiser reads the
  bf16 matrix widened to f32). ``--check-int2`` first re-derives the SOURCE pack's int2 records
  from its own ``head.rerank.w`` and requires sha256 equality, which proves this port is the
  builder's arithmetic.

The table order, shapes, layouts, data offsets and chunk size are the source pack's; the plan
hash is the source's (same plan); the tier id is new (sha256 over the source tier id and the
target tier id). Host-only, CPU, numpy.

    tools/pack/drafter_head_pair.py --drafter-pack $SABER_PACKS/qwen38-dflash2-drafter \\
        --target-pack $SABER_PACKS/qwen38-27b-quark-awq-mxfp4 \\
        --out $SABER_PACKS/qwen38-dflash2-drafter-quarkhead --check-int2

The source must be a drafter pack with a **bf16** ``head.rerank.w``; the hosted ``-fp8head``
drafter packs carry ``head.rerank.w8`` + ``head.rerank.ws`` instead and are refused. The hosted ``qwen38-dflash2-drafter-quarkhead[-tp1]`` packs on Hugging Face are this
tool's output for the Quark checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pack import (  # noqa: E402
    F32, HashPool, Record, RankWriter, bf16_to_f32, canonical, f32_to_bf16, plan_rank,
)

HEAD_GROUP = 128
HEAD_LV = 3
HEAD_BIAS = F32(4.0)
RERANK = "head.rerank.w"
INT2 = ("head.int2.wq", "head.int2.s", "head.int2.zs")
EMBED = "input_embedding.weight"
TARGET_HEAD = "lm_head.w"
TARGET_EMBED = "model.language_model.embed_tokens.weight"
ROWS = 2048


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load(pack: str) -> dict:
    with open(os.path.join(pack, "manifest.json")) as f:
        return json.load(f)


def recs_of(man: dict, rank: int) -> list:
    (rk,) = [r for r in man["ranks"] if r["rank"] == rank]
    return rk["records"]


def header(path: str) -> dict:
    with open(path, "rb") as f:
        h = f.read(256)
    if h[:8] != b"SBRPACK1":
        raise SystemExit(f"{path}: not an SBRPACK1 file")
    return {"plan_hash": h[80:112], "tier_id": h[112:144],
            "chunk_bytes": struct.unpack_from("<I", h, 224)[0]}


def stream(path: str, off: int, length: int, step: int = 64 << 20):
    fd = os.open(path, os.O_RDONLY)
    try:
        pos = 0
        while pos < length:
            b = os.pread(fd, min(step, length - pos), off + pos)
            if not b:
                raise SystemExit(f"{path}: short read at {off + pos}")
            pos += len(b)
            yield b
    finally:
        os.close(fd)


def int2_rows(w16: np.ndarray):
    """The int2 draft head over a [n, K] bf16 block (uint16 bits): (packed u8 [n, K/4],
    s bf16 [n, K/128], zs bf16 [n, K/128])."""
    n, k = w16.shape
    wg = bf16_to_f32(w16).reshape(n, k // HEAD_GROUP, HEAD_GROUP)
    mn, mx = wg.min(axis=2), wg.max(axis=2)
    sc = np.maximum((mx - mn) / F32(HEAD_LV), F32(1e-8)).astype(F32)
    zp = np.clip(np.round(-mn / sc), 0, HEAD_LV).astype(F32)
    q = np.clip(np.round(wg / sc[:, :, None] + zp[:, :, None]), 0, HEAD_LV).astype(np.uint8)
    q = q.reshape(n, k)
    qq = k // 4
    packed = q[:, :qq] | (q[:, qq:2 * qq] << 2) | (q[:, 2 * qq:3 * qq] << 4) | (q[:, 3 * qq:] << 6)
    s = f32_to_bf16(HEAD_BIAS * sc)
    zs = f32_to_bf16(HEAD_BIAS * sc + zp * sc)
    return np.ascontiguousarray(packed), s, zs


def int2_of_head(path: str, rec: dict):
    """The three int2 records' bytes for one rank's head record (streamed in row blocks)."""
    n, k = rec["shape"]
    outs = ([], [], [])
    fd = os.open(path, os.O_RDONLY)
    try:
        for r0 in range(0, n, ROWS):
            r1 = min(n, r0 + ROWS)
            b = os.pread(fd, (r1 - r0) * k * 2, rec["data_off"] + r0 * k * 2)
            parts = int2_rows(np.frombuffer(b, np.uint16).reshape(r1 - r0, k))
            for i in range(3):
                outs[i].append(parts[i].tobytes())
    finally:
        os.close(fd)
    return tuple(b"".join(o) for o in outs)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drafter-pack", required=True)
    ap.add_argument("--target-pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-int2", action="store_true",
                    help="re-derive the source pack's int2 records from its own head first")
    a = ap.parse_args()
    dman, tman = load(a.drafter_pack), load(a.target_pack)
    tp = dman["tp_size"]
    if tman["tp_size"] != tp:
        raise SystemExit(f"tp {dman['tp_size']} vs {tman['tp_size']}")
    for rank in range(tp):
        names = {r["name"] for r in recs_of(dman, rank)}
        if RERANK not in names:
            raise SystemExit(f"{a.drafter_pack}: no {RERANK} (bf16) record; this tool needs the "
                             "bf16-head drafter pack, not an -fp8head one")
        if TARGET_HEAD not in {r["name"] for r in recs_of(tman, rank)}:
            raise SystemExit(f"{a.target_pack}: no bf16 {TARGET_HEAD} record (a target with an fp8 "
                             "head needs no pairing: use the stock drafter pack)")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite a pack")

    # 0. the int2 port against the source pack's own records.
    if a.check_int2:
        for rank in range(tp):
            path = os.path.join(a.drafter_pack, f"rank{rank}.pack")
            by = {r["name"]: r for r in recs_of(dman, rank)}
            got = int2_of_head(path, by[RERANK])
            for name, b in zip(INT2, got):
                want = by[name]["sha256"]
                ok = sha(b) == want
                log(f"check-int2 rank {rank} {name}: {sha(b)[:16]} vs source {want[:16]} "
                    f"{'EQUAL' if ok else 'DIFFER'}")
                if not ok:
                    raise SystemExit("the int2 port does not reproduce the source pack")

    os.makedirs(a.out)
    hp = HashPool(8, 4 << 30)
    ranks_out = []
    for rank in range(tp):
        src = os.path.join(a.drafter_pack, f"rank{rank}.pack")
        tgt = os.path.join(a.target_pack, f"rank{rank}.pack")
        hdr = header(src)
        drecs = recs_of(dman, rank)
        tby = {r["name"]: r for r in recs_of(tman, rank)}
        dby = {r["name"]: r for r in drecs}
        th, te = tby[TARGET_HEAD], tby[TARGET_EMBED]
        for d, t in ((dby[RERANK], th), (dby[EMBED], te)):
            if (d["dtype"], d["shape"], d["length"], d["layout_id"]) != \
                    (t["dtype"], t["shape"], t["length"], t["layout_id"]):
                raise SystemExit(f"rank {rank}: {d['name']} {d['shape']} L{d['layout_id']} vs "
                                 f"{t['name']} {t['shape']} L{t['layout_id']}")
        if dby[EMBED]["sha256"] != te["sha256"]:
            raise SystemExit(f"rank {rank}: the embeddings differ too; this tool swaps the head only")
        records = [Record(r["name"], r["dtype"], r["shape"], r["shard_kind"], r["layout_id"],
                          length=r["length"], sources=r.get("sources"), rank=rank)
                   for r in drecs]
        rp = plan_rank(records, hdr["chunk_bytes"])
        for r, d in zip(records, drecs):
            if r.data_off != d["data_off"]:
                raise SystemExit(f"{r.name}: offset {r.data_off} vs source {d['data_off']}")
        log(f"rank {rank}: int2 head from {TARGET_HEAD} ({th['sha256'][:16]})")
        int2 = dict(zip(INT2, int2_of_head(tgt, th)))
        out = os.path.join(a.out, f"rank{rank}.pack")
        w = RankWriter(out, rp, hp)
        for r, d in zip(records, drecs):
            w.begin(r)
            if r.name == RERANK:
                for b in stream(tgt, th["data_off"], th["length"]):
                    w.write(b)
                r.sources = [f"target:{s}" for s in th.get("sources", [])]
            elif r.name in int2:
                w.write(int2[r.name])
            else:
                for b in stream(src, d["data_off"], d["length"]):
                    w.write(b)
            w.end()
        tier_id = hashlib.sha256(canonical({
            "drafter_tier": dman["tier"]["id"], "head_from_tier": tman["tier"]["id"],
            "tool": "tools/pack/drafter_head_pair.py"})).hexdigest()
        w.finish(tp, rank, hdr["plan_hash"], bytes.fromhex(tier_id), int(time.time()))
        ranks_out.append((rank, out, rp, records, tier_id))
        log(f"rank {rank}: written {rp.file_bytes} B")
    hp.close()

    # The record law: every record equals the source's except the head family; the head equals
    # the target's lm_head.w.
    bad = 0
    man_ranks = []
    for rank, out, rp, records, tier_id in ranks_out:
        drecs = recs_of(dman, rank)
        th = {r["name"]: r for r in recs_of(tman, rank)}[TARGET_HEAD]
        rows = []
        for r, d in zip(records, drecs):
            h = r.sha256.hex()
            if r.name == RERANK:
                ok, what = h == th["sha256"], f"== target {TARGET_HEAD}"
            elif r.name in INT2:
                ok, what = True, f"re-quantised (source {d['sha256'][:16]})"
            else:
                ok, what = h == d["sha256"], "== source"
            bad += not ok
            if r.name in INT2 or r.name == RERANK or not ok:
                log(f"rank {rank} {r.name}: {h[:16]} {what} {'OK' if ok else 'FAIL'}")
            rows.append({"name": r.name, "dtype": r.dtype, "shape": r.shape,
                         "shard_kind": d["shard_kind"], "layout_id": r.layout_id,
                         "layout": d["layout"], "data_off": r.data_off, "length": r.length,
                         "sha256": h, **({"chunks": len(r.chunk_digests)}
                                         if r.chunk_digests else {}),
                         "sources": r.sources})
        same = sum(1 for r, d in zip(records, drecs)
                   if r.name not in INT2 and r.name != RERANK and r.sha256.hex() == d["sha256"])
        log(f"rank {rank}: {same} of {len(records) - 4} non-head records == source; "
            f"head == target: {records[[x.name for x in records].index(RERANK)].sha256.hex() == th['sha256']}")
        man_ranks.append({"rank": rank, "file": os.path.basename(out), "file_bytes": rp.file_bytes,
                          "data_off": rp.data_off, "record_count": len(records), "records": rows})
    manifest = {k: dman[k] for k in ("magic", "format_version", "header_bytes", "record_bytes",
                                     "align")}
    manifest.update({
        "built_unix": int(time.time()), "builder": "tools/pack/drafter_head_pair.py",
        "tp_size": tp,
        "tier": {"id": ranks_out[0][4], "descriptor": {
            "derived_from": {"pack": os.path.abspath(a.drafter_pack), "tier": dman["tier"]},
            "head_from": {"pack": os.path.abspath(a.target_pack), "tier_id": tman["tier"]["id"],
                          "record": TARGET_HEAD}}},
        "plan": dman["plan"], "shard_policy": dman.get("shard_policy", []), "ranks": man_ranks,
    })
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        f.write(json.dumps(manifest, indent=1) + "\n")
    if bad:
        raise SystemExit(f"{bad} record(s) failed the record law")
    log("done: the record law holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
