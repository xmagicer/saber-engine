#!/usr/bin/env python3
"""Build the Qwen3.8 Flash-Next (qwen4_exp) EP=2 tier pack from the Quark MXFP4 checkpoint.

Host tool, numpy only, CPU only: no torch, no GPU, no container, no lock. The record SET comes from
``saber-flashnext-packplan`` (``crates/saber-model/src/qwen4_exp/pack_plan.rs``); this file does
the bytes. Layout under ``--out`` (default ``$SABER_PACKS/qwen38-flashnext-quark-ep2``, SABER_PACKS default ``/mnt/nvme/saber-packs``):

    manifest.json            everything below, the ledger, the build's verification counts
    rank0/experts.xtier      rank 0's routed experts [0, 256) of every layer: the expert tier
    rank1/experts.xtier      rank 1's routed experts [256, 512)
    rank0/dense.pack         rank 0's WRITTEN dense records (SBRPACK1 v2, build_pack.py's format)
    rank1/dense.pack
    shared/dense.pack        replicated written records (the shared expert trios, ...)
    rank{0,1}/dense.json     every dense record of the placement: written ones point into
    shared/dense.json          dense.pack, identity ones (bf16 passthrough / merge, raw) are extent
                             lists into the checkpoint's own shards (``--dense ref``, the default)
    ple/ple_index.json       the PLE n-gram table: an index over the checkpoint's 128 shards
                             (bf16, in place) + the optional first-party fp8 source

Source kinds (``docs/EXEC.md`` "Flash-Next pack law"):

* ``quark_mxfp4`` — the expert tier and the shared expert: Quark's e2m1 nibbles and e8m0 scales
  BYTE COPIED, ``permute_w`` (WMMA fragment order), the scales transposed to ``[K/32, N]``, ``wref``
  = the row max of the scales. NO requantization. Asserted on EVERY expert during the build:
  Quark's own dequant of the checkpoint bytes == Saber's dequant of the pack bytes, bit for bit
  (``flashnext_mx.py``), with the nibble-swapped bytes as the negative control that must differ;
  and FN1's e8m0 scan (values, per-row spread) re-run over every scale tensor.
* ``bf16`` / ``bf16_merged`` / ``raw`` — identity layouts: the checkpoint's bytes in order.
  ``--dense ref`` records them as extents (file, offset, length, stride, count) + the sha256 of the
  bytes they name; ``--dense copy`` writes them into ``dense.pack`` instead.
* ``f32_from_bf16`` — ``A_log`` / ``dt_bias`` widened to f32 (the 27B's layout 5). Written.

The disk decision (2026-09-28: 172.6 GB free of 999.7 GB, the 90 % line leaves 72.6 GB): the
expert tier (64.3 GB, the only large transformed class) is written; every identity record is
referenced in place (a copy would add 14.8 GB and cross the line); the PLE table (102.4 GB bf16)
is read in place through its index. ``--max-fill`` refuses a build that would cross the line.

    cargo run -q -p saber-model --bin saber-flashnext-packplan -- \\
        --checkpoint CKPT > plan.json
    python3 tools/pack/build_flashnext_pack.py --plan plan.json \\
        --model CKPT --out $SABER_PACKS/qwen38-flashnext-quark-ep2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pack as bp  # noqa: E402
import flashnext_mx as mx  # noqa: E402

XTIER_MAGIC = b"SBRXPRT1"
XTIER_VERSION = 1
XTIER_HEADER = 4096
ALIGN = 4096
DEFAULT_OUT = os.path.join(os.environ.get("SABER_PACKS", "/mnt/nvme/saber-packs"),
                           "qwen38-flashnext-quark-ep2")
# The optional first-party fp8 PLE table (a file of the NVFP4 sibling checkpoint); off by default.
DEFAULT_FP8 = ""
ESIZE = {"BF16": 2, "F32": 4, "I64": 8, "U8": 1, "F8_E4M3": 1}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


# ------------------------------------------------------------------------------------------
# checkpoint
# ------------------------------------------------------------------------------------------


def load_checkpoint(model: str):
    files = sorted(
        os.path.join(model, f)
        for f in os.listdir(model)
        if f.endswith(".safetensors") and f.startswith("model-")
    )
    tensors, file_meta = bp.collect_tensors(files)
    return {t.name: t for t in tensors}, file_meta


def extents_of(t, rows, cols):
    """Byte extents of a slice: [(offset, length, stride, count)] with stride = the row bytes."""
    shape = t.shape
    ncols = 1
    for d in shape[1:]:
        ncols *= d
    es = ESIZE[t.dtype]
    row_bytes = ncols * es
    r0, r1 = rows
    c0, c1 = cols
    if (c0, c1) == (0, ncols):
        return [(t.offset + r0 * row_bytes, (r1 - r0) * row_bytes, row_bytes, 1)]
    return [(t.offset + r0 * row_bytes + c0 * es, (c1 - c0) * es, row_bytes, r1 - r0)]


def read_extent(fd, off, length, stride, count) -> bytes:
    if count == 1:
        b = os.pread(fd, length, off)
        if len(b) != length:
            raise SystemExit(f"short read at {off}")
        return b
    # One read over the span, then the strided gather (the rows are adjacent on disk).
    span = os.pread(fd, stride * (count - 1) + length, off)
    a = np.frombuffer(span + b"\0" * (stride - length), dtype=np.uint8).reshape(count, stride)
    return a[:, :length].tobytes()


# ------------------------------------------------------------------------------------------
# the expert tier
# ------------------------------------------------------------------------------------------

_W = {}


def _worker_init(meta_min, model):
    _W["meta"] = meta_min
    _W["fds"] = {}
    _W["model"] = model


def _fd(path):
    fd = _W["fds"].get(path)
    if fd is None:
        fd = os.open(path, os.O_RDONLY)
        _W["fds"][path] = fd
    return fd


def _tensor(name):
    """(path, offset, nbytes, shape) from the worker's copy of the header table."""
    return _W["meta"][name]


def _read(name) -> np.ndarray:
    path, off, n, shape = _tensor(name)
    fd = _fd(path)
    b = os.pread(fd, n, off)
    if len(b) != n:
        raise RuntimeError(f"{name}: short read")
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    return np.frombuffer(b, dtype=np.uint8).reshape(shape)


def quark_expert(prefix: str, hidden: int, inter: int, rec: dict, buf: memoryview, stats: dict):
    """Build one expert record into ``buf`` from checkpoint tensors ``prefix.{gate,up,down}_proj``;
    verify it; accumulate the scan. Returns nothing; raises on a mismatch."""
    g = _read(prefix + ".gate_proj.weight")
    gs = _read(prefix + ".gate_proj.weight_scale")
    u = _read(prefix + ".up_proj.weight")
    us = _read(prefix + ".up_proj.weight_scale")
    d = _read(prefix + ".down_proj.weight")
    ds = _read(prefix + ".down_proj.weight_scale")
    for arr, want in ((g, (inter, hidden // 2)), (u, (inter, hidden // 2)), (d, (hidden, inter // 2)),
                      (gs, (inter, hidden // 32)), (us, (inter, hidden // 32)), (ds, (hidden, inter // 32))):
        if arr.shape != want:
            raise RuntimeError(f"{prefix}: shape {arr.shape}, want {want}")
    gu = np.concatenate([g, u])
    gus = np.concatenate([gs, us])
    n_gu, k_gu = 2 * inter, hidden
    parts = {
        "gate_up_w": mx.permute_w(gu, n_gu, k_gu),
        "gate_up_ws": np.ascontiguousarray(gus.T),
        "gate_up_wref": gus.max(axis=1),
        "down_w": mx.permute_w(d, hidden, inter),
        "down_ws": np.ascontiguousarray(ds.T),
        "down_wref": ds.max(axis=1),
    }
    np.frombuffer(buf, dtype=np.uint8)[:] = 0
    for key, arr in parts.items():
        off, n = rec[key]
        b = np.ascontiguousarray(arr).reshape(-1)
        if b.size != n:
            raise RuntimeError(f"{prefix}.{key}: {b.size} bytes, the plan says {n}")
        buf[off:off + n] = b.tobytes()
    # ---- verification: read the RECORD back, dequantize both ways, compare bit for bit ----
    rb = np.frombuffer(buf, dtype=np.uint8)

    def sub(key, shape):
        off, n = rec[key]
        return rb[off:off + n].reshape(shape)

    sab_gu = mx.saber_dequant(sub("gate_up_w", (n_gu, k_gu // 2)),
                              sub("gate_up_ws", (k_gu // 32, n_gu)), n_gu, k_gu)
    sab_d = mx.saber_dequant(sub("down_w", (hidden, inter // 2)),
                             sub("down_ws", (inter // 32, hidden)), hidden, inter)
    ok = (mx.bits_equal(sab_gu[:inter], mx.quark_dequant(g, gs))
          and mx.bits_equal(sab_gu[inter:], mx.quark_dequant(u, us))
          and mx.bits_equal(sab_d, mx.quark_dequant(d, ds)))
    if not ok:
        raise RuntimeError(f"{prefix}: Saber's dequant of the pack != Quark's dequant of the checkpoint")
    # wref is the row max of the transposed scales, re-derived from the record itself.
    if not (np.array_equal(sub("gate_up_wref", (n_gu,)), sub("gate_up_ws", (k_gu // 32, n_gu)).max(axis=0))
            and np.array_equal(sub("down_wref", (hidden,)), sub("down_ws", (inter // 32, hidden)).max(axis=0))):
        raise RuntimeError(f"{prefix}: wref is not the row max of ws")
    for s in (gs, us, ds):
        mx.scale_scan(s, stats["hist"], stats["spread"])
    stats["tensors"] += 3
    stats["verified"] += 1
    return g, gs


def tier_task(args):
    rank, layer, first, count, hidden, inter, rec, stride, path, data_off, prefix_fmt = args
    fd = os.open(path, os.O_WRONLY | os.O_DIRECT)
    buf_map = mmap.mmap(-1, stride)
    buf = memoryview(buf_map)
    stats = {"hist": np.zeros(256, np.int64), "spread": np.zeros(256, np.int64),
             "tensors": 0, "verified": 0, "controls": 0, "control_escaped": 0}
    shas = []
    t0 = time.time()
    for i, e in enumerate(range(first, first + count)):
        prefix = prefix_fmt.format(layer=layer, expert=e)
        g, gs = quark_expert(prefix, hidden, inter, rec, buf, stats)
        idx = layer * count + i
        n = os.pwrite(fd, buf, data_off + idx * stride)
        if n != stride:
            raise RuntimeError(f"{path}: short O_DIRECT write {n}")
        shas.append(hashlib.sha256(buf).digest())
        if i % 64 == 0:
            # Negative controls, one expert in 64 (4 per layer per rank): the nibble-swapped
            # checkpoint bytes and a neighbouring expert's bytes must NOT match the record.
            rb = np.frombuffer(buf, dtype=np.uint8)
            off, nb = rec["gate_up_w"]
            soff, sn = rec["gate_up_ws"]
            sab = mx.saber_dequant(rb[off:off + nb].reshape(2 * inter, hidden // 2),
                                   rb[soff:soff + sn].reshape(hidden // 32, 2 * inter),
                                   2 * inter, hidden)[:inter]
            swapped = mx.quark_dequant(mx.nibble_swap_control(g), gs)
            other = prefix_fmt.format(layer=layer, expert=first + (i + 1) % count)
            og, ogs = _read(other + ".gate_proj.weight"), _read(other + ".gate_proj.weight_scale")
            for ctl in (swapped, mx.quark_dequant(og, ogs)):
                stats["controls"] += 1
                if mx.bits_equal(sab, ctl):
                    stats["control_escaped"] += 1
    os.close(fd)
    # numpy views of the buffer may outlive the loop; the mapping dies with the task.
    return rank, layer, shas, stats, time.time() - t0


def write_tier_header(path, t, plan_sha, table_sha, table_off, table_bytes, data_off, file_bytes,
                      hidden, inter):
    r = t["record"]
    b = bytearray(XTIER_HEADER)
    b[0:8] = XTIER_MAGIC
    struct.pack_into("<IIIIII", b, 8, XTIER_VERSION, XTIER_HEADER, t["rank"], t["layers"],
                     t["first_expert"], t["experts"])
    struct.pack_into("<QQ", b, 32, r["stride"], r["payload"])
    o = 48
    for key in ("gate_up_w", "gate_up_ws", "gate_up_wref", "down_w", "down_ws", "down_wref"):
        struct.pack_into("<QQ", b, o, r[key][0], r[key][1])
        o += 16
    struct.pack_into("<II", b, o, hidden, inter)
    o += 8
    struct.pack_into("<QQQQ", b, o, table_off, table_bytes, data_off, file_bytes)
    o += 32
    b[o:o + 32] = table_sha
    b[o + 32:o + 64] = plan_sha
    fd = os.open(path, os.O_WRONLY)
    os.pwrite(fd, bytes(b), 0)
    os.fsync(fd)
    os.close(fd)


def tier_geometry(t):
    records = t["records"]
    table_off = XTIER_HEADER
    table_bytes = records * 32
    data_off = align_up(table_off + table_bytes)
    file_bytes = data_off + records * t["record"]["stride"]
    return table_off, table_bytes, data_off, file_bytes


# ------------------------------------------------------------------------------------------
# dense records
# ------------------------------------------------------------------------------------------

LAYOUT_OF_KIND = {
    "bf16": bp.LAYOUT_RAW,
    "raw": bp.LAYOUT_RAW,
    "bf16_merged": bp.LAYOUT_BF16_MERGED,
    "f32_from_bf16": bp.LAYOUT_F32_FROM_BF16,
}


def shard_kind(r):
    if r["placement"] == "shared":
        return bp.REPLICATED
    srcs = r["sources"]
    if any(tuple(s["cols"]) != (0, _ncols(s["shape"])) for s in srcs):
        return bp.DIM1
    if len(srcs) > 1 or r["kind"] == "bf16_merged":
        return bp.DIM0SEG
    return bp.DIM0


def _ncols(shape):
    n = 1
    for d in shape[1:]:
        n *= d
    return n


def mxfp4_layout(r):
    return {"w": bp.LAYOUT_MXFP4_WPERM_MERGED if len(r["sources"]) > 1 else bp.LAYOUT_MXFP4_WPERM,
            "ws": bp.LAYOUT_MXFP4_BLOCK_SCALE, "wref": bp.LAYOUT_MXFP4_ROW_REF}[r["part"]]


def build_trio(ws_recs, meta, fds, stats):
    """The shared expert's (w, ws, wref) from Quark bytes; verified like the expert tier."""
    r = ws_recs[0]
    packed, scales, pieces = [], [], []
    for s in r["sources"]:
        t = meta[s["tensor"]]
        st = meta[s["tensor"] + "_scale"]
        w = np.frombuffer(os.pread(fds[t.path], t.nbytes, t.offset), np.uint8).reshape(t.shape)
        sc = np.frombuffer(os.pread(fds[st.path], st.nbytes, st.offset), np.uint8).reshape(st.shape)
        packed.append(w)
        scales.append(sc)
        pieces.append((w, sc))
        mx.scale_scan(sc, stats["hist"], stats["spread"])
        stats["tensors"] += 1
    n, k = r["shape"]
    p = np.concatenate(packed)
    s = np.concatenate(scales)
    w_perm = mx.permute_w(p, n, k)
    ws = np.ascontiguousarray(s.T)
    wref = ws.max(axis=0)
    sab = mx.saber_dequant(w_perm, ws, n, k)
    row = 0
    for w, sc in pieces:
        if not mx.bits_equal(sab[row:row + w.shape[0]], mx.quark_dequant(w, sc)):
            raise SystemExit(f"{r['name']}: Saber's dequant != Quark's")
        row += w.shape[0]
    stats["shared_verified"] += 1
    return {"w": w_perm.tobytes(), "ws": ws.tobytes(), "wref": wref.tobytes()}


def build_dense(plan, meta, fds, out, mode, stats, dry):
    """Write each placement's dense.pack + dense.json. Returns {placement: summary}."""
    by_place: dict[str, list[dict]] = {}
    for r in plan["dense"]:
        by_place.setdefault(r["placement"], []).append(r)
    summaries = {}
    trio_cache: dict[str, dict] = {}
    for place, recs in by_place.items():
        os.makedirs(os.path.join(out, place), exist_ok=True)
        written = [r for r in recs if mode == "copy" or not r["identity"]]
        pack_records = []
        for r in written:
            layout = mxfp4_layout(r) if r["kind"] == "quark_mxfp4" else LAYOUT_OF_KIND[r["kind"]]
            pack_records.append(bp.Record(
                name=r["name"], dtype=r["dtype"], shape=[int(x) for x in r["shape"]],
                shard_kind=shard_kind(r), layout_id=layout, ranges=[], length=int(r["bytes"])))
        entries = []
        pack_path = os.path.join(out, place, "dense.pack")
        rp = bp.plan_rank(pack_records, bp.DEFAULT_CHUNK_BYTES) if pack_records else None
        if rp is not None and not dry:
            hp = bp.HashPool(8, 2 << 30, bp.DEFAULT_CHUNK_BYTES)
            wr = bp.RankWriter(pack_path, rp, hp)
            for r, rec in zip(written, pack_records):
                if r["kind"] == "quark_mxfp4":
                    op = r["name"].rsplit(".", 1)[0]
                    if op not in trio_cache:
                        trio = [x for x in recs if x["kind"] == "quark_mxfp4"
                                and x["name"].rsplit(".", 1)[0] == op]
                        trio_cache[op] = build_trio(trio, meta, fds, stats)
                    data = trio_cache[op][r["part"]]
                elif r["kind"] == "f32_from_bf16":
                    raw = b"".join(read_slice(meta, fds, s) for s in r["sources"])
                    bf = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                    data = bf.view(np.float32).tobytes()
                else:
                    data = b"".join(read_slice(meta, fds, s) for s in r["sources"])
                wr.begin(rec)
                wr.write(data)
                wr.end()
            hp.close()
            tp, rank = (1, 0) if place == "shared" else (2, int(place[4:]))
            wr.finish(tp, rank, bytes.fromhex(plan["_sha"]), bytes.fromhex(plan["_sha"]),
                      int(time.time()))
        for r in recs:
            e = {k: r[k] for k in ("name", "class", "kind", "dtype", "shape", "bytes", "part")}
            if mode == "copy" or not r["identity"]:
                rec = pack_records[written.index(r)]
                e["storage"] = {"pack": "dense.pack", "data_off": rec.data_off,
                                "length": rec.length,
                                "sha256": rec.sha256.hex() if rec.sha256 else None}
            else:
                exts = []
                h = hashlib.sha256()
                for s in r["sources"]:
                    t = meta[s["tensor"]]
                    for off, ln, stride, cnt in extents_of(t, s["rows"], s["cols"]):
                        exts.append({"file": os.path.basename(t.path), "offset": off,
                                     "length": ln, "stride": stride, "count": cnt})
                        if not dry:
                            h.update(read_extent(fds[t.path], off, ln, stride, cnt))
                e["storage"] = {"checkpoint": exts, "sha256": None if dry else h.hexdigest()}
            entries.append(e)
        with open(os.path.join(out, place, "dense.json"), "w") as f:
            json.dump({"placement": place, "mode": mode, "records": entries}, f, indent=1)
        summaries[place] = {
            "records": len(recs), "written": len(written),
            "written_bytes": sum(int(r["bytes"]) for r in written),
            "pack_file_bytes": rp.file_bytes if rp else 0,
            "referenced": len(recs) - len(written),
            "referenced_bytes": sum(int(r["bytes"]) for r in recs) - sum(int(r["bytes"]) for r in written),
        }
        log(f"   {place}: {summaries[place]}")
    return summaries


def build_aliases(plan, meta, out, dry) -> dict:
    """FN9's loader aliases (the QSA merged projection, N 7,296 per rank): identity extents over
    bytes the dense records already own, + the sha256 of the bytes they name. No byte is written but
    ``rank<r>/aliases.json``."""
    fds = {}
    by_place = {}
    for a in plan.get("aliases", []):
        exts, h = [], hashlib.sha256()
        for s_ in a["sources"]:
            t = meta[s_["tensor"]]
            if t.path not in fds:
                fds[t.path] = os.open(t.path, os.O_RDONLY)
            for off, ln, stride, cnt in extents_of(t, s_["rows"], s_["cols"]):
                exts.append({"file": os.path.basename(t.path), "offset": off, "length": ln,
                             "stride": stride, "count": cnt})
                if not dry:
                    h.update(read_extent(fds[t.path], off, ln, stride, cnt))
        rows = sum(x["rows"][1] - x["rows"][0] for x in a["sources"])
        assert rows == a["shape"][0], a["name"]
        by_place.setdefault(a["placement"], []).append({
            "name": a["name"], "kind": a["kind"], "dtype": a["dtype"], "shape": a["shape"],
            "bytes": a["shape"][0] * a["shape"][1] * 2,
            "members": [{"tensor": x["tensor"], "rows": x["rows"]} for x in a["sources"]],
            "storage": {"checkpoint": exts, "sha256": None if dry else h.hexdigest()}})
    for fd in fds.values():
        os.close(fd)
    if not dry:
        for place, recs in by_place.items():
            with open(os.path.join(out, place, "aliases.json"), "w") as f:
                json.dump({"placement": place, "aliases": recs}, f, indent=1)
    return {k: len(v) for k, v in by_place.items()}


def read_slice(meta, fds, s) -> bytes:
    t = meta[s["tensor"]]
    return b"".join(read_extent(fds[t.path], *x) for x in extents_of(t, s["rows"], s["cols"]))


# ------------------------------------------------------------------------------------------
# PLE index
# ------------------------------------------------------------------------------------------


def _hash_region(args):
    path, off, n = args
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    step = 64 << 20
    pos = 0
    while pos < n:
        b = os.pread(fd, min(step, n - pos), off + pos)
        h.update(b)
        os.posix_fadvise(fd, off + pos, len(b), os.POSIX_FADV_DONTNEED)
        pos += len(b)
    os.close(fd)
    return h.hexdigest()


def straddle_fraction(base: int, row_bytes: int, block: int) -> float:
    """Fraction of rows whose bytes cross a ``block`` boundary (exact over one period)."""
    from math import gcd
    period = block // gcd(row_bytes, block)
    cross = sum(1 for i in range(period)
                if (base + i * row_bytes) // block != (base + i * row_bytes + row_bytes - 1) // block)
    return cross / period


def build_ple_index(plan, meta, model, out, fp8_path, workers, dry, hash_it):
    ple = plan["ple"]
    shards = []
    jobs = []
    for k in range(ple["shards"]):
        name = f"{ple['tensor_prefix']}{k}.weight"
        t = meta[name]
        rows = t.shape[0]
        if t.dtype != "BF16" or t.shape[1] != ple["row_dim"] or rows != ple["rows_per_shard"]:
            raise SystemExit(f"{name}: {t.dtype} {t.shape}, the plan says BF16 "
                             f"[{ple['rows_per_shard']}, {ple['row_dim']}]")
        shards.append({"shard": k, "tensor": name, "file": os.path.basename(t.path),
                       "offset": t.offset, "rows": rows, "bytes": t.nbytes,
                       "first_row": k * ple["rows_per_shard"],
                       "offset_mod_512": t.offset % 512, "offset_mod_4096": t.offset % 4096,
                       "straddle_512": straddle_fraction(t.offset, ple["row_bytes_bf16"], 512),
                       "straddle_4096": straddle_fraction(t.offset, ple["row_bytes_bf16"], 4096)})
        jobs.append((t.path, t.offset, t.nbytes))
    if hash_it and not dry:
        log(f"   hashing the 128 bf16 table shards in place ({sum(j[2] for j in jobs) / 1e9:.1f} GB)")
        with mp.Pool(workers) as pool:
            for s, hsh in zip(shards, pool.map(_hash_region, jobs)):
                s["sha256"] = hsh
    idx = {
        "table": {"rows": ple["rows"], "rows_per_shard": ple["rows_per_shard"],
                  "shards": ple["shards"], "row_dim": ple["row_dim"],
                  "rows_per_token": ple["rows_per_token"]},
        "sources": {
            "ckpt_bf16": {
                "default": True,
                "checkpoint": os.path.abspath(model),
                "dtype": "BF16", "row_bytes": ple["row_bytes_bf16"],
                "bytes_per_token": ple["row_bytes_bf16"] * ple["rows_per_token"],
                "read_law": "row r -> shard r // rows_per_shard, local r % rows_per_shard; byte "
                            "offset = shard.offset + local * row_bytes; one O_DIRECT read of the "
                            "512-aligned span covering it (512 B, or 1024 B when it straddles)",
                "shards": shards,
            },
        },
    }
    if fp8_path and os.path.exists(fp8_path):
        header, data_off = bp.read_safetensors_header(fp8_path)
        header.pop("__metadata__", None)
        fshards = []
        for k in range(ple["shards"]):
            name = f"{ple['tensor_prefix']}{k}.weight"
            e = header.get(name)
            sname = f"{ple['tensor_prefix']}{k}.weight_scale"
            se = header.get(sname)
            if e is None:
                fshards = None
                break
            off = data_off + e["data_offsets"][0]
            fs = {"shard": k, "tensor": name, "dtype": e["dtype"], "shape": e["shape"],
                  "offset": off, "bytes": e["data_offsets"][1] - e["data_offsets"][0],
                  "straddle_512": straddle_fraction(off, ple["row_bytes_fp8"], 512)}
            if se is not None:
                fs["scale"] = {"tensor": sname, "dtype": se["dtype"], "shape": se["shape"],
                               "offset": data_off + se["data_offsets"][0]}
            fshards.append(fs)
        # Qwen's fp8 table carries ONE per-tensor scale for all 128 shards
        # (`ngram_embedding.weight_scale`, bf16 [1]); a per-shard scale would sit on each shard.
        tname = f"{ple['tensor_prefix'].rsplit('.shard_', 1)[0]}.weight_scale"
        te = header.get(tname)
        table_scale = None
        if te is not None:
            with open(fp8_path, "rb") as f:
                f.seek(data_off + te["data_offsets"][0])
                raw = f.read(te["data_offsets"][1] - te["data_offsets"][0])
            val = float(np.frombuffer(raw, np.uint16).astype(np.uint32).__lshift__(16).view(np.float32)[0]) \
                if te["dtype"] == "BF16" else None
            table_scale = {"tensor": tname, "dtype": te["dtype"], "shape": te["shape"],
                           "offset": data_off + te["data_offsets"][0], "value": val}
        if fshards:
            bad = [s for s in fshards if s["dtype"] != "F8_E4M3"
                   or s["shape"] != [ple["rows_per_shard"], ple["row_dim"]]]
            if bad:
                raise SystemExit(f"fp8 PLE shard {bad[0]['tensor']}: {bad[0]['dtype']} {bad[0]['shape']}")
            idx["sources"]["qwen_fp8"] = {
                "default": False,
                "gated": "a numerics choice (the first-party fp8 bytes of Qwen/Qwen3.8-Flash-Next-FP8, "
                         "per-tensor scale per shard): F6's gates and the user's word before any "
                         "forward reads it",
                "file": fp8_path, "dtype": "F8_E4M3", "row_bytes": ple["row_bytes_fp8"],
                "table_scale": table_scale,
                "dequant": "row_fp32 = f32(e4m3) * table_scale.value (one scale for the whole table)",
                "bytes_per_token": ple["row_bytes_fp8"] * ple["rows_per_token"],
                "shards": fshards,
            }
    os.makedirs(os.path.join(out, "ple"), exist_ok=True)
    with open(os.path.join(out, "ple", "ple_index.json"), "w") as f:
        json.dump(idx, f, indent=1)
    return idx


# ------------------------------------------------------------------------------------------
# plan checks
# ------------------------------------------------------------------------------------------


def check_plan_against_checkpoint(plan, meta):
    """Every source exists with the plan's shape and a dtype its kind accepts, and every checkpoint
    tensor is consumed exactly once (dense records, the expert tier, the PLE index, vision)."""
    used: dict[str, list] = {}
    want_dtype = {"bf16": "BF16", "bf16_merged": "BF16", "f32_from_bf16": "BF16",
                  "raw": "I64", "quark_mxfp4": "U8"}
    for r in plan["dense"]:
        for s in r["sources"]:
            t = meta.get(s["tensor"])
            if t is None:
                raise SystemExit(f"{r['name']}: source {s['tensor']} is not in the checkpoint")
            if [int(x) for x in s["shape"]] != t.shape:
                raise SystemExit(f"{s['tensor']}: header {t.shape}, the plan says {s['shape']}")
            if t.dtype != want_dtype[r["kind"]]:
                raise SystemExit(f"{s['tensor']}: {t.dtype}, kind {r['kind']} wants {want_dtype[r['kind']]}")
            if r["kind"] == "quark_mxfp4":
                if r["part"] != "w":
                    continue
                sc = s["tensor"] + "_scale"
                if sc not in meta:
                    raise SystemExit(f"{sc}: missing")
                used.setdefault(sc, []).append(("all", [0, 0]))
                used.setdefault(s["tensor"], []).append(("all", [0, 0]))
                continue
            used.setdefault(s["tensor"], []).append((tuple(s["rows"]), tuple(s["cols"])))
    # the expert tier
    for t in plan["experts"]:
        for layer in range(t["layers"]):
            for e in range(t["first_expert"], t["first_expert"] + t["experts"]):
                p = f"model.language_model.layers.{layer}.mlp.experts.{e}"
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for suf in (".weight", ".weight_scale"):
                        n = f"{p}.{proj}{suf}"
                        if n not in meta or meta[n].dtype != "U8":
                            raise SystemExit(f"{n}: missing or not U8")
                        used.setdefault(n, []).append(("all", [0, 0]))
    for k in range(plan["ple"]["shards"]):
        used.setdefault(f"{plan['ple']['tensor_prefix']}{k}.weight", []).append(("all", [0, 0]))
    # Coverage: every byte of every tensor once. Slices of one tensor must tile it exactly.
    unused, double = [], []
    for name, t in meta.items():
        if name.startswith("model.visual."):
            continue
        u = used.get(name)
        if not u:
            unused.append(name)
            continue
        if u[0][0] == "all":
            if len(u) != 1:
                double.append(name)
            continue
        rows, ncols = t.shape[0], _ncols(t.shape)
        area = sum((r[1] - r[0]) * (c[1] - c[0]) for r, c in u)
        if area != rows * max(ncols, 1):
            double.append(f"{name} (area {area} vs {rows * max(ncols, 1)})")
        # overlap check: sort by (row, col) start and require disjointness
        boxes = sorted(u)
        for a, b in zip(boxes, boxes[1:]):
            (ar, ac), (br, bc) = a, b
            if ar[0] < br[1] and br[0] < ar[1] and ac[0] < bc[1] and bc[0] < ac[1]:
                double.append(f"{name} (overlap)")
                break
    if unused or double:
        raise SystemExit(f"coverage: {len(unused)} unconsumed (e.g. {unused[:3]}), "
                         f"{len(double)} doubly consumed (e.g. {double[:3]})")
    vision = [n for n in meta if n.startswith("model.visual.")]
    return {"tensors": len(meta), "consumed": len(meta) - len(vision), "vision_skipped": len(vision)}


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dense", choices=("ref", "copy"), default="ref")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layers", default="", help="a:b, build only these layers' experts (smoke)")
    ap.add_argument("--max-fill", type=float, default=0.90)
    ap.add_argument("--fp8-ple", default=DEFAULT_FP8,
                    help="optional: model-fp8-mtp-ple.safetensors, an extra PLE source (never the default one)")
    ap.add_argument("--no-ple-hash", action="store_true")
    ap.add_argument("--skip-experts", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-aliases", default="", metavar="BASE_PLAN",
                    help="FN9: write rank<r>/aliases.json into an existing pack built from BASE_PLAN "
                         "(the new --plan must equal BASE_PLAN except for its `aliases`)")
    ap.add_argument("--only-ple", action="store_true",
                    help="rebuild ple/ple_index.json of an existing pack and update its manifest")
    args = ap.parse_args(argv)

    t_start = time.time()
    raw_plan = open(args.plan, "rb").read()
    plan = json.loads(raw_plan)
    plan["_sha"] = hashlib.sha256(raw_plan).hexdigest()
    log(f"== plan {args.plan} sha256 {plan['_sha'][:16]}…: {len(plan['dense'])} dense records, "
        f"{sum(t['records'] for t in plan['experts'])} expert records")
    meta, file_meta = load_checkpoint(args.model)
    log(f"== checkpoint {args.model}: {len(file_meta)} shards, {len(meta)} tensors")
    cov = check_plan_against_checkpoint(plan, meta)
    log(f"   coverage: {cov}")
    if args.only_aliases:
        mpath = os.path.join(args.out, "manifest.json")
        manifest = json.load(open(mpath))
        base_raw = open(args.only_aliases, "rb").read()
        if hashlib.sha256(base_raw).hexdigest() != manifest["plan_sha256"]:
            raise SystemExit("--only-aliases: BASE_PLAN is not the plan the pack was built from")
        base = json.loads(base_raw)
        new = {k: v for k, v in plan.items() if k not in ("aliases", "_sha")}
        if base != new:
            diff = sorted(k for k in set(base) | set(new) if base.get(k) != new.get(k))
            raise SystemExit(f"--only-aliases: the plan differs from BASE_PLAN outside `aliases`: {diff}")
        n = build_aliases(plan, meta, args.out, args.dry_run)
        if not args.dry_run:
            manifest["aliases"] = {"plan_sha256": plan["_sha"], "files": {p_: f"{p_}/aliases.json" for p_ in n},
                                   "counts": n}
            with open(mpath, "w") as f:
                json.dump(manifest, f, indent=1)
        log(f"== aliases written: {n}")
        return 0
    if args.only_ple:
        mpath = os.path.join(args.out, "manifest.json")
        manifest = json.load(open(mpath))
        if manifest["plan_sha256"] != plan["_sha"]:
            raise SystemExit("--only-ple: the pack was built from a different plan")
        ple = build_ple_index(plan, meta, args.model, args.out, args.fp8_ple, args.workers, False,
                              not args.no_ple_hash)
        manifest["ple"] = {"index": "ple/ple_index.json",
                           "sources": {k: {kk: vv for kk, vv in v.items() if kk != "shards"}
                                       for k, v in ple["sources"].items()}}
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=1)
        log("== PLE index rebuilt")
        return 0

    # ---- the disk decision ------------------------------------------------------------------
    tiers = []
    for t in plan["experts"]:
        table_off, table_bytes, data_off, file_bytes = tier_geometry(t)
        tiers.append((t, table_off, table_bytes, data_off, file_bytes))
    dense_written = sum(int(r["bytes"]) for r in plan["dense"]
                        if args.dense == "copy" or not r["identity"])
    projected = (0 if args.skip_experts else sum(x[4] for x in tiers)) + dense_written
    st = os.statvfs(os.path.dirname(os.path.abspath(args.out)) or "/")
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    existing = 0
    for root, _, files in os.walk(args.out) if os.path.isdir(args.out) else []:
        for f in files:
            existing += os.stat(os.path.join(root, f)).st_blocks * 512
    after = total - free - existing + projected
    log(f"== disk: total {total / 1e9:.1f} GB, free {free / 1e9:.1f} GB, this pack {projected / 1e9:.2f} GB "
        f"(replacing {existing / 1e9:.2f}) -> {after / total * 100:.2f}% used (limit {args.max_fill * 100:.0f}%)")
    if after / total > args.max_fill:
        raise SystemExit("refusing: the build would fill the drive past the limit")
    if args.dry_run:
        log("dry run: stopping before any write")
        return 0
    os.makedirs(args.out, exist_ok=True)

    fds = {}
    for t in meta.values():
        if t.path not in fds:
            fds[t.path] = os.open(t.path, os.O_RDONLY)

    stats = {"hist": np.zeros(256, np.int64), "spread": np.zeros(256, np.int64),
             "tensors": 0, "verified": 0, "controls": 0, "control_escaped": 0,
             "shared_verified": 0}
    manifest_tiers = []
    lo, hi = 0, plan["experts"][0]["layers"]
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split(":"))
    if not args.skip_experts:
        meta_min = {n: (t.path, t.offset, t.nbytes, tuple(t.shape)) for n, t in meta.items()
                    if ".mlp.experts." in n and n.startswith("model.language_model.")}
        jobs = []
        for t, table_off, table_bytes, data_off, file_bytes in tiers:
            d = os.path.join(args.out, f"rank{t['rank']}")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "experts.xtier")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
            os.ftruncate(fd, file_bytes)
            if not args.layers:
                # One extent per tier file; a smoke build (--layers) stays sparse.
                os.posix_fallocate(fd, 0, file_bytes)
            os.close(fd)
            rec = {k: tuple(v) for k, v in t["record"].items() if isinstance(v, list)}
            for layer in range(lo, hi):
                jobs.append((t["rank"], layer, t["first_expert"], t["experts"], 2560, 640, rec,
                             t["record"]["stride"], path, data_off,
                             "model.language_model.layers.{layer}.mlp.experts.{expert}"))
        log(f"== expert tier: {len(jobs)} (rank, layer) tasks on {args.workers} workers")
        shas = {t["rank"]: [b"\0" * 32] * t["records"] for t in plan["experts"]}
        t0 = time.time()
        done = 0
        with mp.Pool(args.workers, initializer=_worker_init, initargs=(meta_min, args.model)) as pool:
            for rank, layer, s, st_, dt in pool.imap_unordered(tier_task, jobs):
                cnt = len(s)
                shas[rank][layer * cnt:(layer + 1) * cnt] = s
                for k in ("hist", "spread"):
                    stats[k] += st_[k]
                for k in ("tensors", "verified", "controls", "control_escaped"):
                    stats[k] += st_[k]
                done += 1
                if done % 8 == 0 or done == len(jobs):
                    el = time.time() - t0
                    log(f"   {done}/{len(jobs)} tasks, {stats['verified']} experts verified, "
                        f"{el:.0f} s, {stats['verified'] * 2.617344e-3 / max(el, 1e-9):.2f} GB/s")
        if stats["control_escaped"]:
            raise SystemExit(f"{stats['control_escaped']} negative controls MATCHED the record")
        for t, table_off, table_bytes, data_off, file_bytes in tiers:
            path = os.path.join(args.out, f"rank{t['rank']}", "experts.xtier")
            table = b"".join(shas[t["rank"]])
            fd = os.open(path, os.O_WRONLY)
            os.pwrite(fd, table, table_off)
            os.fsync(fd)
            os.close(fd)
            write_tier_header(path, t, bytes.fromhex(plan["_sha"]), hashlib.sha256(table).digest(),
                              table_off, table_bytes, data_off, file_bytes, 2560, 640)
            manifest_tiers.append({
                "rank": t["rank"], "file": f"rank{t['rank']}/experts.xtier",
                "first_expert": t["first_expert"], "experts": t["experts"], "layers": t["layers"],
                "built_layers": [lo, hi], "records": t["records"], "record": t["record"],
                "table_off": table_off, "table_bytes": table_bytes, "data_off": data_off,
                "file_bytes": file_bytes, "table_sha256": hashlib.sha256(table).hexdigest()})

    log("== dense records")
    dense = build_dense(plan, meta, fds, args.out, args.dense, stats, False)
    log("== PLE index")
    ple = build_ple_index(plan, meta, args.model, args.out, args.fp8_ple, args.workers, False,
                          not args.no_ple_hash)

    nz = lambda a: {int(i): int(v) for i, v in enumerate(a) if v}  # noqa: E731
    manifest = {
        "pack": os.path.basename(os.path.abspath(args.out)),
        "model": "qwen4_exp",
        "checkpoint": {"dir": os.path.abspath(args.model), "shards": file_meta},
        "plan_sha256": plan["_sha"],
        "layout": plan["layout"],
        "declared_defaults": plan["declared_defaults"],
        "dense_mode": args.dense,
        "dense": dense,
        "expert_tiers": manifest_tiers,
        "ple": {"index": "ple/ple_index.json",
                "sources": {k: {kk: vv for kk, vv in v.items() if kk != "shards"}
                            for k, v in ple["sources"].items()}},
        "ledger": plan["ledger"],
        "consumed_by_class": plan["consumed_by_class"],
        "coverage": cov,
        "verification": {
            "experts_verified": stats["verified"],
            "shared_expert_trios_verified": stats["shared_verified"],
            "scale_tensors_scanned": stats["tensors"],
            "e8m0_values": nz(stats["hist"]),
            "per_row_spread": nz(stats["spread"]),
            "negative_controls": stats["controls"],
            "negative_controls_matched": stats["control_escaped"],
        },
        "built_unix": int(time.time()),
        "build_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"== done in {time.time() - t_start:.0f} s: {json.dumps(manifest['verification'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
