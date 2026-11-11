#!/usr/bin/env python3
"""Build the Qwen3.8-27B device-final tier pack (``SBRPACK1`` v2) from the NVFP4 checkpoint, or
from AMD's OCP-MXFP4 (Quark AWQ) checkpoint through ``quark27b_plan.py``'s re-sourced plan.

Written from the SBRPACK1 format (header, record table, layout ids, TP shard policy; the reader is
``crates/saber-mem/src/pack/format.rs``). Host tool, **numpy only, CPU only**: no torch, no GPU.
Most users run it through ``tools/build_packs.sh``, which also makes the plan and runs the checks.

    saber-packplan --tp 2 --checkpoint MODEL > plan.json        # the record SET (saber-model)
    tools/pack/build_pack.py --model MODEL --out DIR --tp 2 \\
        --layout device-final --plan-json plan.json              # the bytes

The split is the one the SBRPACK1 format § "Building the device-final tier" states: the plan names every
record, its layout id, its rank-local shape and dtype and the checkpoint tensors that feed it, in
concatenation order; this file only does the numerics and the file format. Checkpoint tensors no
plan record reads (the vision tower) are carried ``raw`` under the shard policy, sorted by name.

Transforms, by layout id (the SBRPACK1 format § Layout ids):

* 0 ``raw`` -- the TP shard of the checkpoint's bytes and nothing else;
* 1 / 4 ``mxfp4_wperm[_merged]`` + 2 ``mxfp4_block_scale`` + 3 ``mxfp4_row_ref`` -- the sources
  dequantized to f32 (NVFP4: ``e2m1 x (f32(e4m3 scale) / weight_global_scale)``; per-channel fp8:
  ``f32(e4m3) x f32(weight_scale)``; bf16 widened), dim-0 concatenated, then MXFP4 at the ``mse``
  block exponent (``quantize_mxfp4`` below), the nibbles in WMMA fragment order (``permute_w``),
  the e8m0 scales transposed to ``[K/32, N]`` and ``wref`` = their max per row;
* 5 ``f32_from_bf16`` -- the bf16 widened (exact);
* 6 ``bf16_merged`` -- a dim-0 concatenation of bf16 shards, no requant;
* 14 ``bf16_from_fp8_channel`` -- ``bf16(f32(w8) x f32(weight_scale))``, round to nearest even;
* 15 ``fp8_channel`` -- the e4m3 rows, untouched; 16 ``f32_channel_scale`` -- the bf16 per-row
  scale widened to f32.

The ``mse`` block exponent. Per 32-element block along K: ``e0 = floor(log2(amax)) - 2`` (the OCP
exponent, clamped to [-127, 127]); the two candidates ``e0`` and ``e0 + 1`` are each encoded on the
e2m1 grid (round to nearest, ties to the even code, saturating at 6) and scored by their sum of
squared errors in f32, accumulated as eight lanes of four (lane ``i`` holds ``k = i, i+8, i+16,
i+24`` in that order) then the lanes in order; ``e0 + 1`` wins when its sum is smaller, or equal
and ``e0`` would clip (``amax > 6 x 2^e0``). Note that this decision is not
reproducible across reduction orders: a block whose two sums are within f32 rounding goes
whichever way the order goes (``tools/pack/compare_pack.py`` classifies such blocks).

Records are written straight into ``rank<r>.pack`` at offsets fixed before any byte is computed
(the record lengths are the plan's), by a pool of worker processes that also take each record's
sha256 and its 4 MiB chunk digests; the parent then writes the header, table, name blob and chunk
blob. The report goes to stderr; ``manifest.json`` is the only machine-readable output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ------------------------------------------------------------------------------------------
# the SBRPACK1 format
# ------------------------------------------------------------------------------------------

MAGIC = b"SBRPACK1"
FORMAT_VERSION = 2
HEADER_BYTES = 256
RECORD_BYTES = 160
ALIGN = 4096
CHUNK_ENTRY_HEADER = 40
FLAG_CHUNK_DIGESTS = 1
DEFAULT_CHUNK_BYTES = 4 << 20

# name -> (tag, bits per element); the names are safetensors' where it has one.
DTYPES = {
    "U8": (0, 8), "I8": (1, 8), "U16": (2, 16), "I16": (3, 16), "U32": (4, 32), "I32": (5, 32),
    "U64": (6, 64), "I64": (7, 64), "F16": (8, 16), "BF16": (9, 16), "F32": (10, 32),
    "F64": (11, 64), "F8_E4M3": (12, 8), "F8_E5M2": (13, 8), "FP4_E2M1_X2": (14, 4),
    "BOOL": (15, 8), "E8M0": (16, 8),
}

REPLICATED, DIM0, DIM1, DIM0SEG = 0, 1, 2, 3
SHARD_NAMES = {REPLICATED: "Replicated", DIM0: "Dim0", DIM1: "Dim1", DIM0SEG: "Dim0Segments"}
SHARD_OF_NAME = {v: k for k, v in SHARD_NAMES.items()}

LAYOUT_RAW = 0
LAYOUT_MXFP4_WPERM = 1
LAYOUT_MXFP4_BLOCK_SCALE = 2
LAYOUT_MXFP4_ROW_REF = 3
LAYOUT_MXFP4_WPERM_MERGED = 4
LAYOUT_F32_FROM_BF16 = 5
LAYOUT_BF16_MERGED = 6
LAYOUT_BF16_FROM_FP8_CHANNEL = 14
LAYOUT_FP8_CHANNEL = 15
LAYOUT_F32_CHANNEL_SCALE = 16
LAYOUT_NAMES = {
    0: "raw", 1: "mxfp4_wperm", 2: "mxfp4_block_scale", 3: "mxfp4_row_ref",
    4: "mxfp4_wperm_merged", 5: "f32_from_bf16", 6: "bf16_merged", 7: "fp8_preshuffle",
    8: "fp8_block_scale", 9: "int2_head", 10: "int2_head_scale", 11: "int2_head_zero",
    12: "bf16_from_fp8_block", 13: "rope_cos_sin", 14: "bf16_from_fp8_channel",
    15: "fp8_channel", 16: "f32_channel_scale",
}
MXFP4_W = (LAYOUT_MXFP4_WPERM, LAYOUT_MXFP4_WPERM_MERGED)
MXFP4_TRIO = MXFP4_W + (LAYOUT_MXFP4_BLOCK_SCALE, LAYOUT_MXFP4_ROW_REF)

# The plan descriptor's transform list (the plan hash's pre-image, the SBRPACK1 format § Plan hash).
DEVICE_FINAL_TRANSFORMS = [
    "dequant(nvfp4|fp8_channel|bf16)->fp32",
    "quant_mxfp4(exp=mse)",
    "make_row_ref",
    "permute_w(wperm=1)",
    "merge_dim0",
    "cast_bf16_to_f32",
]
# Hashed into the tier descriptor beside the safetensors headers (the SBRPACK1 format § Tier id).
TIER_EXTRAS = ["config.json", "chat_template.jinja", "tokenizer.json", "generation_config.json"]

# the SBRPACK1 format § "TP shard policy": every checkpoint tensor matches exactly one rule.
SHARD_POLICY = [
    (r"^model\.language_model\.embed_tokens\.weight$", "Dim0", "vocab parallel"),
    (r"^lm_head\.weight$", "Dim0", "vocab parallel"),
    (r"^lm_head\.weight_scale$", "Dim0", "per output row"),
    (r"^model\.language_model\.norm\.weight$", "Replicated", ""),
    (r"model\.language_model\.layers\.\d+\.input_layernorm\.weight$", "Replicated", ""),
    (r"model\.language_model\.layers\.\d+\.post_attention_layernorm\.weight$", "Replicated", ""),
    (r"model\.language_model\.layers\.\d+\.self_attn\.[qk]_norm\.weight$", "Replicated",
     "per head_dim"),
    (r"model\.language_model\.layers\.\d+\.self_attn\.[kv]_scale$", "Replicated",
     "scalar KV scale"),
    (r"model\.language_model\.layers\.\d+\.mlp\.\w+\.(input|weight)_global_scale$", "Replicated",
     "scalar"),
    (r"model\.language_model\.layers\.\d+\.self_attn\.q_proj\.weight$", "Dim0", "q+gate per head"),
    (r"model\.language_model\.layers\.\d+\.self_attn\.q_proj\.weight_scale$", "Dim0", ""),
    (r"model\.language_model\.layers\.\d+\.self_attn\.[kv]_proj\.weight$", "Dim0", "kv head"),
    (r"model\.language_model\.layers\.\d+\.self_attn\.[kv]_proj\.weight_scale$", "Dim0", ""),
    (r"model\.language_model\.layers\.\d+\.self_attn\.o_proj\.weight$", "Dim1", "row parallel"),
    (r"model\.language_model\.layers\.\d+\.self_attn\.o_proj\.weight_scale$", "Replicated",
     "per output row"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_qkv\.weight$", "Dim0Segments",
     "concat(q,k,v) along dim 0"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_qkv\.weight_scale$",
     "Dim0Segments", ""),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.conv1d\.weight$", "Dim0Segments",
     "same concat as in_proj_qkv"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_z\.weight$", "Dim0",
     "value heads"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_z\.weight_scale$", "Dim0", ""),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_[ab]\.weight$", "Dim0",
     "one row per value head"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.(A_log|dt_bias)$", "Dim0",
     "one entry per value head"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.norm\.weight$", "Replicated",
     "per head_dim"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.out_proj\.weight$", "Dim1",
     "row parallel"),
    (r"model\.language_model\.layers\.\d+\.linear_attn\.out_proj\.weight_scale$", "Replicated",
     "per output row"),
    (r"model\.language_model\.layers\.\d+\.mlp\.(gate|up)_proj\.(weight|weight_packed|weight_scale)$",
     "Dim0", ""),
    (r"model\.language_model\.layers\.\d+\.mlp\.down_proj\.(weight|weight_packed)$", "Dim1",
     "row parallel"),
    (r"model\.language_model\.layers\.\d+\.mlp\.down_proj\.weight_scale$", "Dim1",
     "per-K block scales, or per-row"),
    (r"model\.visual\..*$", "Replicated", "vision tower not TP-sharded in v0"),
    (r"mtp\.fc\.weight$", "Replicated", "drafter fc, replicated in v0"),
    (r"mtp\.(norm|pre_fc_norm_embedding|pre_fc_norm_hidden)\.weight$", "Replicated", ""),
    (r"mtp\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight$", "Replicated", ""),
    (r"mtp\.layers\.\d+\.self_attn\.[qk]_norm\.weight$", "Replicated", ""),
    (r"mtp\.layers\.\d+\.self_attn\.q_proj\.weight$", "Dim0", ""),
    (r"mtp\.layers\.\d+\.self_attn\.[kv]_proj\.weight$", "Dim0", ""),
    (r"mtp\.layers\.\d+\.self_attn\.o_proj\.weight$", "Dim1", ""),
    (r"mtp\.layers\.\d+\.mlp\.(gate|up)_proj\.weight$", "Dim0", ""),
    (r"mtp\.layers\.\d+\.mlp\.down_proj\.weight$", "Dim1", ""),
]
_POLICY_RE = [(re.compile(p), k) for p, k, _ in SHARD_POLICY]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def policy_kind(name: str) -> str:
    """The one shard-policy rule a checkpoint tensor matches; zero or two is a build error."""
    hits = [k for rx, k in _POLICY_RE if rx.search(name)]
    if len(hits) != 1:
        raise SystemExit(f"shard policy: {name} matches {len(hits)} rules (exactly one required)")
    return hits[0]


# ------------------------------------------------------------------------------------------
# safetensors
# ------------------------------------------------------------------------------------------


class Tensor:
    """One checkpoint tensor: where its bytes are, nothing read."""

    __slots__ = ("name", "path", "offset", "nbytes", "dtype", "shape")

    def __init__(self, name, path, offset, nbytes, dtype, shape):
        self.name, self.path, self.offset, self.nbytes = name, path, offset, nbytes
        self.dtype, self.shape = dtype, list(shape)


def read_safetensors_header(path: str):
    """(the JSON header as a dict, ``__metadata__`` included; the absolute data offset)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def collect_tensors(files):
    """Every tensor of ``files`` + one tier-descriptor entry per file (the SBRPACK1 format § Tier id)."""
    tensors, meta = [], []
    for path in files:
        with open(path, "rb") as f:
            raw8 = f.read(8)
            n = struct.unpack("<Q", raw8)[0]
            raw = f.read(n)
        header = json.loads(raw)
        md = header.pop("__metadata__", None)
        base = 8 + n
        for name, e in header.items():
            a, b = e["data_offsets"]
            tensors.append(Tensor(name, path, base + a, b - a, e["dtype"], e["shape"]))
        meta.append({
            "name": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "header_sha256": hashlib.sha256(raw8 + raw).hexdigest(),
            "metadata": md,
            "tensor_count": len(header),
        })
    return tensors, meta


# ------------------------------------------------------------------------------------------
# records, the rank's geometry, the metadata block
# ------------------------------------------------------------------------------------------


class Record:
    """One table entry. ``data_off`` is fixed by ``plan_rank``; ``sha256`` by the writer."""

    def __init__(self, name, dtype, shape, shard_kind, layout_id, ranges=None, length=None,
                 sources=None, rank=0):
        self.name = name
        self.dtype = dtype
        self.shape = [int(x) for x in shape]
        self.shard_kind = shard_kind if isinstance(shard_kind, int) else SHARD_OF_NAME[shard_kind]
        self.layout_id = int(layout_id)
        self.ranges = ranges or []
        want = expected_length(dtype, self.shape)
        if length is not None and int(length) != want:
            raise SystemExit(f"{name}: length {length} != {dtype} {self.shape} = {want}")
        self.length = want
        self.sources = list(sources or [])
        self.rank = rank
        self.data_off = None
        self.sha256 = None
        self.chunk_digests = None

    def flagged(self, chunk_bytes: int) -> bool:
        return chunk_bytes > 0 and self.length > chunk_bytes

    def n_chunks(self, chunk_bytes: int) -> int:
        return -(-self.length // chunk_bytes)


def expected_length(dtype: str, shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return -(-n * DTYPES[dtype][1] // 8)


class RankPlan:
    """A rank file's geometry, decided before any payload byte exists."""

    def __init__(self, records, chunk_bytes):
        if not 1 <= len(records):
            raise SystemExit("a pack needs at least one record")
        if chunk_bytes % ALIGN:
            raise SystemExit(f"chunk bytes {chunk_bytes} is not a multiple of {ALIGN}")
        names = set()
        blob = bytearray()
        self.name_offs = []
        for r in records:
            if r.name in names:
                raise SystemExit(f"duplicate record name {r.name}")
            names.add(r.name)
            nb = r.name.encode()
            blob += struct.pack("<H", len(nb))
            self.name_offs.append(len(blob))
            blob += nb
        self.records = records
        self.chunk_bytes = chunk_bytes
        self.name_blob = bytes(blob)
        self.table_off = HEADER_BYTES
        self.name_blob_off = self.table_off + RECORD_BYTES * len(records)
        self.chunk_blob_bytes = sum(CHUNK_ENTRY_HEADER + 32 * r.n_chunks(chunk_bytes)
                                    for r in records if r.flagged(chunk_bytes))
        meta_end = self.name_blob_off + len(self.name_blob)
        self.chunk_blob_off = meta_end if self.chunk_blob_bytes else 0
        meta_end += self.chunk_blob_bytes
        self.data_off = max(align_up(meta_end), ALIGN)
        cur = self.data_off
        for r in records:
            r.data_off = cur
            cur = align_up(cur + r.length)
        self.file_bytes = cur


def plan_rank(records, chunk_bytes=DEFAULT_CHUNK_BYTES) -> RankPlan:
    return RankPlan(records, chunk_bytes)


def encode_entry(r: Record, name_off: int, rank: int, chunk_bytes: int) -> bytes:
    if not 1 <= len(r.shape) <= 8:
        raise SystemExit(f"{r.name}: ndim {len(r.shape)}")
    e = bytearray(RECORD_BYTES)
    flags = FLAG_CHUNK_DIGESTS if r.flagged(chunk_bytes) else 0
    struct.pack_into("<IIII", e, 0, name_off, len(r.name.encode()), rank, r.layout_id)
    struct.pack_into("<HBBI", e, 16, DTYPES[r.dtype][0], len(r.shape), r.shard_kind, flags)
    struct.pack_into("<QQ", e, 24, r.data_off, r.length)
    struct.pack_into(f"<{len(r.shape)}Q", e, 40, *r.shape)
    e[104:136] = r.sha256
    return bytes(e)


def write_metadata(path, rp: RankPlan, tp_size, rank, plan_hash, tier_id, build_unix) -> None:
    """Header + table + name blob + chunk blob, once every record has its digests."""
    table = b"".join(encode_entry(r, off, rank, rp.chunk_bytes)
                     for r, off in zip(rp.records, rp.name_offs))
    table_sha = hashlib.sha256(table + rp.name_blob).digest()
    blob = bytearray()
    for r in rp.records:
        if r.flagged(rp.chunk_bytes):
            d = r.chunk_digests
            if len(d) != r.n_chunks(rp.chunk_bytes):
                raise SystemExit(f"{r.name}: {len(d)} chunk digests, want {r.n_chunks(rp.chunk_bytes)}")
            blob += struct.pack("<II", len(d), 0) + r.sha256 + b"".join(d)
    if len(blob) != rp.chunk_blob_bytes:
        raise SystemExit("chunk blob length disagrees with the plan")
    h = bytearray(HEADER_BYTES)
    h[0:8] = MAGIC
    struct.pack_into("<IIII", h, 8, FORMAT_VERSION, HEADER_BYTES, RECORD_BYTES, len(rp.records))
    struct.pack_into("<QQQQQ", h, 24, rp.table_off, rp.name_blob_off, len(rp.name_blob),
                     rp.data_off, rp.file_bytes)
    struct.pack_into("<IIQ", h, 64, tp_size, rank, build_unix)
    h[80:112] = plan_hash
    h[112:144] = tier_id
    h[144:176] = table_sha
    struct.pack_into("<QQ", h, 176, rp.chunk_blob_off, rp.chunk_blob_bytes)
    h[192:224] = hashlib.sha256(bytes(blob)).digest() if blob else bytes(32)
    struct.pack_into("<I", h, 224, rp.chunk_bytes if blob else 0)
    meta = bytes(h) + table + rp.name_blob + bytes(blob)
    fd = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(fd, meta, 0)
        if os.fstat(fd).st_size != rp.file_bytes:
            os.ftruncate(fd, rp.file_bytes)
        os.fsync(fd)
    finally:
        os.close(fd)


def create_rank_file(path: str, file_bytes: int) -> None:
    """One extent per rank (the SBRPACK1 format: the interleaved writes otherwise braid the two files)."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        try:
            os.posix_fallocate(fd, 0, file_bytes)
        except OSError:
            os.ftruncate(fd, file_bytes)
    finally:
        os.close(fd)


class Digest:
    """A record's sha256 and its chunk digests from one pass over bytes fed in any cut."""

    def __init__(self, chunk_bytes: int, flagged: bool):
        self.whole = hashlib.sha256()
        self.chunk_bytes = chunk_bytes
        self.flagged = flagged
        self.chunks = []
        self.cur = hashlib.sha256()
        self.fill = 0

    def update(self, b) -> None:
        mv = memoryview(b).cast("B")
        self.whole.update(mv)
        if not self.flagged:
            return
        i = 0
        while i < len(mv):
            take = min(len(mv) - i, self.chunk_bytes - self.fill)
            self.cur.update(mv[i:i + take])
            self.fill += take
            i += take
            if self.fill == self.chunk_bytes:
                self.chunks.append(self.cur.digest())
                self.cur, self.fill = hashlib.sha256(), 0

    def finish(self):
        if self.flagged and self.fill:
            self.chunks.append(self.cur.digest())
        return self.whole.digest(), (self.chunks if self.flagged else None)


class HashPool:
    """Background hashing for the streaming writer (``RankWriter``)."""

    def __init__(self, threads: int, budget_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        self.ex = ThreadPoolExecutor(max_workers=max(1, threads))
        self.chunk_bytes = chunk_bytes
        self.budget = threading.Semaphore(max(1, budget_bytes // (64 << 20)))
        self.futures = []

    def submit(self, rec: Record, parts) -> None:
        self.budget.acquire()

        def job():
            try:
                d = Digest(self.chunk_bytes, rec.flagged(self.chunk_bytes))
                for p in parts:
                    d.update(p)
                rec.sha256, rec.chunk_digests = d.finish()
            finally:
                self.budget.release()

        self.futures.append(self.ex.submit(job))

    def drain(self) -> None:
        for f in self.futures:
            f.result()
        self.futures = []

    def close(self) -> None:
        self.drain()
        self.ex.shutdown()


class RankWriter:
    """Stream records into a rank file in table order: ``begin`` / ``write``... / ``end``, then
    ``finish`` writes the metadata block. Used by ``build_flashnext_pack.py``."""

    def __init__(self, path: str, rp: RankPlan, hp: HashPool):
        if rp.chunk_bytes != hp.chunk_bytes:
            raise SystemExit("the hash pool and the rank plan cut chunks differently")
        self.path, self.rp, self.hp = path, rp, hp
        create_rank_file(path, rp.file_bytes)
        self.fd = os.open(path, os.O_WRONLY)
        self.rec, self.pos, self.parts, self.next = None, 0, [], 0

    def begin(self, rec: Record) -> None:
        if self.rec is not None or rec is not self.rp.records[self.next]:
            raise SystemExit(f"{rec.name}: records are written in table order, one at a time")
        self.rec, self.pos, self.parts = rec, 0, []

    def write(self, data) -> None:
        b = bytes(data)
        if self.pos + len(b) > self.rec.length:
            raise SystemExit(f"{self.rec.name}: more than {self.rec.length} bytes")
        os.pwrite(self.fd, b, self.rec.data_off + self.pos)
        self.pos += len(b)
        self.parts.append(b)

    def end(self) -> None:
        if self.pos != self.rec.length:
            raise SystemExit(f"{self.rec.name}: {self.pos} of {self.rec.length} bytes")
        self.hp.submit(self.rec, self.parts)
        self.rec, self.parts = None, []
        self.next += 1

    def finish(self, tp_size, rank, plan_hash, tier_id, build_unix) -> None:
        if self.next != len(self.rp.records):
            raise SystemExit(f"{self.path}: {self.next} of {len(self.rp.records)} records written")
        self.hp.drain()
        os.close(self.fd)
        write_metadata(self.path, self.rp, tp_size, rank, plan_hash, tier_id, build_unix)


# ------------------------------------------------------------------------------------------
# numerics
# ------------------------------------------------------------------------------------------

F32 = np.float32
MX_BLOCK = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)
# NVFP4 nibble -> value: bit 3 is the sign.
FP4_VALUES = np.concatenate([E2M1, -E2M1]).astype(F32)


def _e4m3_table() -> np.ndarray:
    """OCP e4m3 (``F8_E4M3``): bias 7, subnormals at 2^-6, 0x7F / 0xFF NaN, no infinity."""
    out = np.empty(256, F32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            out[b] = np.nan
        elif e == 0:
            out[b] = sign * (m / 8.0) * 2.0 ** -6
        else:
            out[b] = sign * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out


E4M3 = _e4m3_table()


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(F32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest, ties to even (no NaN in a weight; a NaN would be refused upstream)."""
    u = x.astype(F32).view(np.uint32).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def e2m1_codes(y: np.ndarray) -> np.ndarray:
    """|x| / 2^e -> the e2m1 magnitude code: nearest, a tie to the even code, saturating at 6.

    The midpoints 0.25 1.25 2.5 5 sit below an even code (a tie rounds DOWN), 0.75 1.75 3.5
    below an odd one (a tie rounds UP)."""
    c = (y > F32(0.25)).astype(np.uint8)
    c += y >= F32(0.75)
    c += y > F32(1.25)
    c += y >= F32(1.75)
    c += y > F32(2.5)
    c += y >= F32(3.5)
    c += y > F32(5.0)
    return c


def _lane8_sum(d2: np.ndarray) -> np.ndarray:
    """Sum over the last axis (32) in f32: eight lanes of four, then the lanes in order."""
    a = d2[..., 0:8] + d2[..., 8:16]
    a = a + d2[..., 16:24]
    a = a + d2[..., 24:32]
    s = a[..., 0] + a[..., 1]
    for i in range(2, 8):
        s = s + a[..., i]
    return s


def _at(ax: np.ndarray, e: np.ndarray):
    """Codes and f32 squared-error sum of |x| (``[..., 32]``) encoded at block exponent ``e``."""
    up = np.ldexp(F32(1), -e).astype(F32)[..., None]
    c = e2m1_codes(ax * up)
    q = E2M1[c] * np.ldexp(F32(1), e).astype(F32)[..., None]
    d = q - ax
    return c, _lane8_sum(d * d)


def quantize_mxfp4(x: np.ndarray):
    """f32 ``[R, K]`` -> (codes ``[R, K]`` u8 with the sign in bit 3, e8m0 ``[R, K/32]`` u8).

    The ``mse`` block exponent (module docstring)."""
    r, k = x.shape
    if k % MX_BLOCK:
        raise SystemExit(f"MXFP4 needs K % 32 == 0, got {k}")
    if not np.isfinite(x).all():
        raise SystemExit("non-finite weight in an MXFP4 source")
    xb = x.reshape(r, k // MX_BLOCK, MX_BLOCK)
    ax = np.abs(xb)
    amax = ax.max(axis=-1)
    eb = ((amax.view(np.uint32) >> 23) & 0xFF).astype(np.int32) - 127
    e0 = np.where(amax == 0, -127, np.clip(eb - 2, -127, 127)).astype(np.int32)
    e1 = np.minimum(e0 + 1, 127)
    clips = amax > np.ldexp(F32(6), e0).astype(F32)
    c0, s0 = _at(ax, e0)
    c1, s1 = _at(ax, e1)
    take1 = ((s1 < s0) | ((s1 == s0) & clips)) & (e1 > e0)
    codes = np.where(take1[..., None], c1, c0)
    codes |= ((xb < 0) & (codes != 0)).astype(np.uint8) << 3
    e = np.where(take1, e1, e0)
    return codes.reshape(r, k), (e + 127).astype(np.uint8)


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """``[R, K]`` codes -> ``[R, K/2]`` bytes, the even element in the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def permute_w(packed: np.ndarray, n: int, k: int) -> np.ndarray:
    """``[N, K/2]`` -> WMMA fragment order: (N/16, 16, K/16, 2, 4) -> (N/16, K/16, 2, 16, 4)."""
    if n % 16 or k % 16:
        raise SystemExit(f"permute_w needs N, K % 16 == 0, got N={n} K={k}")
    return packed.reshape(n // 16, 16, k // 16, 2, 4).transpose(0, 2, 3, 1, 4).reshape(n, k // 2)


def unpermute_w(perm: np.ndarray, n: int, k: int) -> np.ndarray:
    return perm.reshape(n // 16, k // 16, 2, 16, 4).transpose(0, 3, 1, 2, 4).reshape(n, k // 2)


# ------------------------------------------------------------------------------------------
# the checkpoint, sharded
# ------------------------------------------------------------------------------------------

NP_OF = {"BF16": np.uint16, "F8_E4M3": np.uint8, "U8": np.uint8, "F32": np.float32,
         "F16": np.uint16, "I64": np.int64, "I32": np.int32}
ESIZE = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I64": 8, "I32": 4}

_G: dict = {}  # the worker's view: tensors, config segments, fds, rank plans


def _fd(path: str, write: bool = False) -> int:
    key = (path, write)
    fd = _G["fds"].get(key)
    if fd is None:
        fd = os.open(path, os.O_WRONLY if write else os.O_RDONLY)
        _G["fds"][key] = fd
    return fd


def read_rows(t: Tensor, r0: int, r1: int) -> np.ndarray:
    """Rows ``[r0, r1)`` of a tensor, as its own dtype, shape ``[r1 - r0, *shape[1:]]``."""
    row = t.nbytes // t.shape[0]
    fd = _fd(t.path)
    off, n = t.offset + r0 * row, (r1 - r0) * row
    b = os.pread(fd, n, off)
    if len(b) != n:
        raise SystemExit(f"{t.name}: short read")
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    return np.frombuffer(b, NP_OF[t.dtype]).reshape([r1 - r0] + t.shape[1:])


def segments_of(name: str, rows: int) -> list:
    """Dim-0 segments of a tensor whose dim 0 is a concatenation (the SBRPACK1 format shard policy)."""
    cfg = _G["cfg"]
    if re.search(r"linear_attn\.(in_proj_qkv\.weight(_scale)?|conv1d\.weight)$", name):
        qk = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        v = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        seg = [qk, qk, v]
        if sum(seg) != rows:
            raise SystemExit(f"{name}: segments {seg} do not cover {rows} rows")
        return seg
    return [rows]


def row_pieces(name: str, kind: str, rows: int, rank: int, tp: int) -> list:
    """The source rows ``[(r0, r1), ...]`` this rank holds, in order."""
    if kind in ("Replicated", "Dim1"):
        return [(0, rows)]
    out, base = [], 0
    for s in (segments_of(name, rows) if kind == "Dim0Segments" else [rows]):
        if s % tp:
            raise SystemExit(f"{name}: segment of {s} rows does not split {tp} ways")
        per = s // tp
        out.append((base + rank * per, base + (rank + 1) * per))
        base += s
    return out


def col_range(name: str, kind: str, cols: int, rank: int, tp: int, quantum: int = 1):
    if kind != "Dim1":
        return 0, cols
    if cols % (tp * quantum):
        raise SystemExit(f"{name}: {cols} columns do not split {tp} ways on a {quantum} boundary")
    per = cols // tp
    return rank * per, (rank + 1) * per


def raw_pieces(t: Tensor, kind: str, rank: int, tp: int, max_bytes: int = 64 << 20):
    """The rank's bytes of a tensor, in order, in pieces of at most ~``max_bytes``."""
    if not t.shape:
        yield bytes(os.pread(_fd(t.path), t.nbytes, t.offset))
        return
    rows = t.shape[0]
    row_bytes = t.nbytes // rows
    cols = row_bytes // ESIZE[t.dtype]
    c0, c1 = col_range(t.name, kind, cols, rank, tp)
    step = max(1, max_bytes // max(1, row_bytes))
    for r0, r1 in row_pieces(t.name, kind, rows, rank, tp):
        for a in range(r0, r1, step):
            b = min(r1, a + step)
            x = read_rows(t, a, b).reshape(b - a, -1).view(np.uint8)
            if (c0, c1) != (0, cols):
                es = ESIZE[t.dtype]
                x = np.ascontiguousarray(x[:, c0 * es:c1 * es])
            yield x.tobytes()


def dequant_rows(src: dict, kind: str, r0: int, r1: int, rank: int, tp: int) -> np.ndarray:
    """f32 rows ``[r0, r1)`` of a source, the rank's columns."""
    T = _G["tensors"]
    w = T[src["weight"]]
    q = src["quant"]
    if q == "nvfp4":
        k = w.shape[1] * 2
        c0, c1 = col_range(w.name, kind, k, rank, tp, 32)
        p = read_rows(w, r0, r1)[:, c0 // 2:c1 // 2]
        s = read_rows(T[src["scale"]], r0, r1)[:, c0 // 16:c1 // 16]
        gs = np.frombuffer(os.pread(_fd(T[src["global_scale"]].path), 4,
                                    T[src["global_scale"]].offset), F32)[0]
        codes = np.empty((r1 - r0, c1 - c0), np.uint8)
        codes[:, 0::2] = p & 0xF
        codes[:, 1::2] = p >> 4
        bs = (E4M3[s] / gs).astype(F32)
        x = FP4_VALUES[codes].reshape(r1 - r0, -1, 16) * bs[..., None]
        return x.reshape(r1 - r0, c1 - c0)
    if q == "mxfp4_ocp":
        # an OCP MXFP4 checkpoint (AMD Quark `weight_format: real_quantized`): `weight` U8
        # [N, K/2] (the LOW nibble is element 2i, sign in bit 3) + `weight_scale` U8 e8m0
        # [N, K/32]; value = e2m1 x 2^(s - 127). `mxfp4_trio` asserts the requant is value-exact.
        if w.dtype != "U8" or T[src["scale"]].dtype != "U8":
            raise SystemExit(f"{w.name}: mxfp4_ocp source is {w.dtype}/{T[src['scale']].dtype}")
        k = w.shape[1] * 2
        c0, c1 = col_range(w.name, kind, k, rank, tp, 32)
        p = read_rows(w, r0, r1)[:, c0 // 2:c1 // 2]
        s = read_rows(T[src["scale"]], r0, r1)[:, c0 // MX_BLOCK:c1 // MX_BLOCK]
        if (s == 0xFF).any():
            raise SystemExit(f"{w.name}: e8m0 NaN scale (0xFF)")
        codes = np.empty((r1 - r0, c1 - c0), np.uint8)
        codes[:, 0::2] = p & 0xF
        codes[:, 1::2] = p >> 4
        bs = np.ldexp(F32(1), s.astype(np.int32) - 127).astype(F32)
        x = FP4_VALUES[codes].reshape(r1 - r0, -1, MX_BLOCK) * bs[..., None]
        return x.reshape(r1 - r0, c1 - c0)
    if q == "fp8_channel":
        c0, c1 = col_range(w.name, kind, w.shape[1], rank, tp, 32)
        s = bf16_to_f32(read_rows(T[src["scale"]], r0, r1).reshape(r1 - r0, 1))
        return E4M3[read_rows(w, r0, r1)[:, c0:c1]] * s
    if q in ("bf16", "as_stored"):
        if w.dtype != "BF16":
            raise SystemExit(f"{w.name}: {q} source is {w.dtype}, not BF16")
        c0, c1 = col_range(w.name, kind, w.shape[1], rank, tp, 32)
        return bf16_to_f32(read_rows(w, r0, r1)[:, c0:c1])
    raise SystemExit(f"{w.name}: unknown source quant {q}")


def mxfp4_trio(group: dict, rank: int, tp: int) -> dict:
    """{role: bytes} of one op's (w, ws, wref) for one rank."""
    wrec = group["w"]
    n, k = wrec["shape"]
    kind = wrec["shard_kind"]
    T = _G["tensors"]
    packed = np.empty((n, k // 2), np.uint8)
    ws = np.empty((n, k // MX_BLOCK), np.uint8)
    row = 0
    multi = len(wrec["sources"]) > 1
    for src in wrec["sources"]:
        name = src["weight"]
        # A merge concatenates each source's own split (the SBRPACK1 format: Dim0Segments); a single source
        # takes the record's.
        skind = policy_kind(name) if multi else kind
        for a, b in row_pieces(name, skind, T[name].shape[0], rank, tp):
            step = max(16, (4 << 20) // k)
            for r0 in range(a, b, step):
                r1 = min(b, r0 + step)
                x = dequant_rows(src, skind, r0, r1, rank, tp)
                if x.shape[1] != k:
                    raise SystemExit(f"{wrec['name']}: {name} gives {x.shape[1]} columns, plan {k}")
                codes, e = quantize_mxfp4(np.ascontiguousarray(x, dtype=F32))
                if src["quant"] == "mxfp4_ocp":
                    # An MXFP4 source re-encodes losslessly (one of the two candidate exponents
                    # carries every value exactly): prove it, block by block.
                    back = FP4_VALUES[codes].reshape(r1 - r0, -1, MX_BLOCK) * np.ldexp(
                        F32(1), e.astype(np.int32) - 127).astype(F32)[..., None]
                    if not np.array_equal(back.reshape(r1 - r0, k), x):
                        raise SystemExit(f"{wrec['name']}: {name} rows {r0}..{r1}: the MXFP4 "
                                         "re-encode is not value-exact")
                packed[row:row + r1 - r0] = pack_nibbles(codes)
                ws[row:row + r1 - r0] = e
                row += r1 - r0
    if row != n:
        raise SystemExit(f"{wrec['name']}: sources give {row} rows, plan {n}")
    out = {"w": permute_w(packed, n, k).tobytes()}
    if "ws" in group:
        out["ws"] = np.ascontiguousarray(ws.T).tobytes()
    if "wref" in group:
        out["wref"] = ws.max(axis=1).tobytes()
    return out


def record_pieces(rec: dict, rank: int, tp: int):
    """Byte pieces of a non-MXFP4 record for one rank."""
    T = _G["tensors"]
    lid = rec["layout_id"]
    kind = rec["shard_kind"]
    srcs = rec["sources"]
    if lid == LAYOUT_RAW and srcs[0]["quant"] == "unit_bf16":
        # a KV scale the checkpoint does not carry (no kv_cache scheme). The engine binds
        # 1.0 for every `k_scale` / `v_scale` (saber-exec forward.rs index table); the record
        # says so in bf16 rather than borrowing another checkpoint's calibration.
        if rec["dtype"] != "BF16" or rec["shape"] != [1]:
            raise SystemExit(f"{rec['name']}: unit_bf16 is a [1] BF16 record")
        yield np.array([0x3F80], np.uint16).tobytes()
    elif lid == LAYOUT_RAW:
        (src,) = srcs
        yield from raw_pieces(T[src["weight"]], kind, rank, tp)
    elif lid == LAYOUT_BF16_MERGED:
        for src in srcs:
            t = T[src["weight"]]
            if t.dtype != "BF16":
                raise SystemExit(f"{rec['name']}: {t.name} is {t.dtype}")
            yield from raw_pieces(t, policy_kind(t.name) if kind != "Dim1" else "Dim1", rank, tp)
    elif lid == LAYOUT_F32_FROM_BF16:
        (src,) = srcs
        for b in raw_pieces(T[src["weight"]], kind, rank, tp):
            yield bf16_to_f32(np.frombuffer(b, np.uint16)).tobytes()
    elif lid == LAYOUT_FP8_CHANNEL:
        (src,) = srcs
        t = T[src["weight"]]
        if t.dtype != "F8_E4M3":
            raise SystemExit(f"{rec['name']}: {t.name} is {t.dtype}")
        yield from raw_pieces(t, kind, rank, tp)
    elif lid == LAYOUT_F32_CHANNEL_SCALE:
        (src,) = srcs
        for b in raw_pieces(T[src["scale"]], kind, rank, tp):
            yield bf16_to_f32(np.frombuffer(b, np.uint16)).tobytes()
    elif lid == LAYOUT_BF16_FROM_FP8_CHANNEL and srcs[0]["quant"] == "bf16":
        # a checkpoint whose head is ALREADY bf16 (Quark's `lm_head` is excluded): the
        # record's bytes are the checkpoint's own rows, untouched (never requantized).
        (src,) = srcs
        t = T[src["weight"]]
        if t.dtype != "BF16":
            raise SystemExit(f"{rec['name']}: {t.name} is {t.dtype}, the plan says bf16")
        yield from raw_pieces(t, kind, rank, tp)
    elif lid == LAYOUT_BF16_FROM_FP8_CHANNEL:
        (src,) = srcs
        t, ts = T[src["weight"]], T[src["scale"]]
        for r0, r1 in row_pieces(t.name, kind, t.shape[0], rank, tp):
            step = max(1, (64 << 20) // t.shape[1])
            for a in range(r0, r1, step):
                b = min(r1, a + step)
                s = bf16_to_f32(read_rows(ts, a, b).reshape(b - a, 1))
                yield f32_to_bf16(E4M3[read_rows(t, a, b)] * s).tobytes()
    else:
        raise SystemExit(f"{rec['name']}: layout {lid} ({LAYOUT_NAMES.get(lid)}) is not built here")


# ------------------------------------------------------------------------------------------
# the worker: compute, write at the fixed offset, hash
# ------------------------------------------------------------------------------------------


def _emit(rank: int, name: str, pieces) -> tuple:
    rp_rec = _G["recs"][rank][name]
    cb = _G["chunk_bytes"]
    d = Digest(cb, rp_rec["length"] > cb > 0)
    fd = _fd(_G["paths"][rank], write=True)
    pos = rp_rec["data_off"]
    n = 0
    for b in pieces:
        os.pwrite(fd, b, pos + n)
        d.update(b)
        n += len(b)
    if n != rp_rec["length"]:
        raise SystemExit(f"rank {rank} {name}: wrote {n} bytes, the plan says {rp_rec['length']}")
    sha, chunks = d.finish()
    return rank, name, sha, chunks


def _task(task) -> list:
    # A SystemExit inside a pool worker kills the worker and loses the task (the pool hangs):
    # hand it back as an ordinary exception so the parent fails loudly.
    try:
        return _task_body(task)
    except SystemExit as e:
        raise RuntimeError(f"{task[0]} rank {task[1]}: {e}") from None


def _task_body(task) -> list:
    kind, rank, payload = task
    tp = _G["tp"]
    if kind == "trio":
        out = mxfp4_trio(payload, rank, tp)
        return [_emit(rank, payload[role]["name"], [b]) for role, b in out.items()]
    return [_emit(rank, payload["name"], record_pieces(payload, rank, tp))]


def _init_worker(state):
    _G.update(state)
    _G["fds"] = {}


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def build_records(plan: dict, tensors: dict, tp: int, only):
    """The rank's records in table order (the plan's, then the raw leftovers) + the task list."""
    used = set()
    for r in plan["records"]:
        for s in r["sources"]:
            if s["quant"] == "unit_bf16":
                continue  # a constant, not a checkpoint tensor (record_pieces)
            for key in ("weight", "scale", "global_scale", "input_scale"):
                if s.get(key):
                    used.add(s[key])
    missing = sorted(n for n in used if n not in tensors)
    if missing:
        raise SystemExit(f"plan names {len(missing)} tensors the checkpoint lacks: {missing[:4]}")
    rows = [dict(r) for r in plan["records"]]
    for name in sorted(set(tensors) - used):
        t = tensors[name]
        kind = policy_kind(name)
        if kind != "Replicated":
            raise SystemExit(f"{name}: read by no plan record and not Replicated ({kind})")
        rows.append({"name": name, "layout_id": LAYOUT_RAW, "dtype": t.dtype, "shard_kind": kind,
                     "shape": t.shape, "transformed": False, "leftover": True,
                     "sources": [{"weight": name, "quant": "as_stored", "scale": None,
                                  "global_scale": None, "input_scale": None}]})
    for r in rows:
        check_shard_kind(r, tensors)
    groups = {}
    for r in rows:
        if r["layout_id"] in MXFP4_TRIO:
            role = {LAYOUT_MXFP4_BLOCK_SCALE: "ws", LAYOUT_MXFP4_ROW_REF: "wref"}.get(
                r["layout_id"], "w")
            g = groups.setdefault(r["op"], {})
            if role in g:
                raise SystemExit(f"op {r['op']} has two {role} records")
            g[role] = r
    for op, g in groups.items():
        if set(g) != {"w", "ws", "wref"}:
            raise SystemExit(f"op {op}: MXFP4 records {sorted(g)}, want w, ws and wref")
    if only:
        # An MXFP4 record pulls in its op's whole trio: the three come out of one quantization.
        rx = re.compile(only)
        ops = {r["op"] for r in rows if r["layout_id"] in MXFP4_TRIO and rx.search(r["name"])}
        rows = [r for r in rows if rx.search(r["name"])
                or (r["layout_id"] in MXFP4_TRIO and r["op"] in ops)]
        if not rows:
            raise SystemExit(f"--only {only!r} selects no record")
    tasks, seen = [], set()
    for r in rows:
        if r["layout_id"] in MXFP4_TRIO:
            if r["op"] not in seen:
                seen.add(r["op"])
                tasks.append(("trio", groups[r["op"]]))
        else:
            tasks.append(("rec", r))
    return rows, tasks


def check_shard_kind(r: dict, tensors: dict) -> None:
    """The plan's shard kind is the one the policy gives its sources (the SBRPACK1 format: asserted loudly)."""
    if any(s["quant"] == "unit_bf16" for s in r["sources"]):
        return  # a constant: no checkpoint tensor to shard
    kinds = [policy_kind(s["weight"]) for s in r["sources"]]
    if r["layout_id"] in (LAYOUT_F32_CHANNEL_SCALE,):
        kinds = [policy_kind(r["sources"][0]["scale"])]
    if len(kinds) > 1:
        want = "Dim1" if all(k == "Dim1" for k in kinds) else "Dim0Segments"
        if want == "Dim0Segments" and any(k not in ("Dim0", "Dim0Segments") for k in kinds):
            raise SystemExit(f"{r['name']}: merge of {kinds}")
    else:
        want = kinds[0]
    if want != r["shard_kind"]:
        raise SystemExit(f"{r['name']}: plan says {r['shard_kind']}, the policy {want}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--out", required=True, help="pack directory (rank<r>.pack + manifest.json)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--layout", default="device-final", choices=["device-final"])
    ap.add_argument("--plan-json", required=True, help="saber-packplan's output")
    ap.add_argument("--only", default=None, help="regex: a partial pack of the matching records")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--digest-chunk-mib", type=int, default=DEFAULT_CHUNK_BYTES >> 20,
                    help="chunk digest size (0 = no chunk blob)")
    ap.add_argument("--plan-hash", default=None, help="a planner's plan hash (hex) to stamp")
    ap.add_argument("--build-unix", type=int, default=None,
                    help="header build time (informational; default now)")
    a = ap.parse_args(argv)
    t_start = time.time()

    with open(a.plan_json) as f:
        plan = json.load(f)
    if plan.get("producer") != "saber-packplan":
        raise SystemExit(f"{a.plan_json}: producer {plan.get('producer')!r}, want saber-packplan")
    if int(plan["tp_size"]) != a.tp:
        raise SystemExit(f"plan is tp {plan['tp_size']}, --tp {a.tp}")
    with open(os.path.join(a.model, "config.json")) as f:
        cfgj = json.load(f)
    cfg = cfgj.get("text_config", cfgj)

    files = sorted(os.path.join(a.model, f) for f in os.listdir(a.model)
                   if f.endswith(".safetensors"))
    tlist, file_meta = collect_tensors(files)
    tensors = {}
    for t in tlist:
        if t.name in tensors:
            raise SystemExit(f"{t.name} is in two checkpoint files")
        tensors[t.name] = t

    extras = {}
    for fn in TIER_EXTRAS:
        p = os.path.join(a.model, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                extras[fn] = hashlib.sha256(f.read()).hexdigest()
    tier_desc = {"model": os.path.basename(os.path.normpath(a.model)), "files": file_meta,
                 "extras": extras, "format": "safetensors"}
    tier_id = hashlib.sha256(canonical(tier_desc)).hexdigest()
    plan_desc = {"format_version": FORMAT_VERSION, "tp_size": a.tp,
                 "layouts": {"by_record": "device-final"},
                 "transforms": list(DEVICE_FINAL_TRANSFORMS), "source": {"tier_id": tier_id}}
    # an OCP MXFP4 source (Quark) adds its dequant to the pre-image; NVFP4 packs keep the
    # list (and so the plan hash) they always had.
    if any(s["quant"] == "mxfp4_ocp" for r in plan["records"] for s in r["sources"]):
        plan_desc["transforms"].append("dequant(mxfp4_ocp)->fp32(value-exact)")
    plan_hash = a.plan_hash or hashlib.sha256(canonical(plan_desc)).hexdigest()
    plan_desc["supplied"] = a.plan_hash is not None
    if len(bytes.fromhex(plan_hash)) != 32:
        raise SystemExit("--plan-hash must be 32 bytes of hex")

    rows, tasks = build_records(plan, tensors, a.tp, a.only)
    chunk_bytes = a.digest_chunk_mib << 20
    os.makedirs(a.out, exist_ok=True)
    rplans, paths = [], []
    for rank in range(a.tp):
        recs = [Record(r["name"], r["dtype"], r["shape"], r["shard_kind"], r["layout_id"],
                       sources=[s["weight"] for s in r["sources"]]
                       if r.get("transformed") else [], rank=rank) for r in rows]
        rp = plan_rank(recs, chunk_bytes)
        path = os.path.join(a.out, f"rank{rank}.pack")
        create_rank_file(path, rp.file_bytes)
        rplans.append(rp)
        paths.append(path)
    total = sum(rp.file_bytes for rp in rplans)
    log(f"build_pack: {len(rows)} records a rank, {a.tp} ranks, {total / 1e9:.2f} GB, "
        f"{len(tasks)} tasks x {a.tp} ranks on {a.workers} workers -> {a.out}")

    state = {
        "tensors": tensors, "cfg": cfg, "tp": a.tp, "paths": paths, "chunk_bytes": chunk_bytes,
        "recs": [{r.name: {"data_off": r.data_off, "length": r.length} for r in rp.records}
                 for rp in rplans],
    }
    by_name = [{r.name: r for r in rp.records} for rp in rplans]
    work = [(k, rank, p) for k, p in tasks for rank in range(a.tp)]
    # Largest first, so the long tasks do not straggle at the end.
    work.sort(key=lambda w: -_task_bytes(w))
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(state,)) as pool:
        for res in pool.imap_unordered(_task, work, chunksize=1):
            for rank, name, sha, chunks in res:
                rec = by_name[rank][name]
                rec.sha256, rec.chunk_digests = sha, chunks
            done += 1
            if done % 200 == 0 or done == len(work):
                log(f"   {done}/{len(work)} tasks, {time.time() - t_start:.0f} s")
    build_unix = a.build_unix if a.build_unix is not None else int(time.time())
    for rank, (rp, path) in enumerate(zip(rplans, paths)):
        for r in rp.records:
            if r.sha256 is None:
                raise SystemExit(f"rank {rank} {r.name}: never written")
        write_metadata(path, rp, a.tp, rank, bytes.fromhex(plan_hash), bytes.fromhex(tier_id),
                       build_unix)

    manifest = {
        "magic": MAGIC.decode(), "format_version": FORMAT_VERSION, "header_bytes": HEADER_BYTES,
        "record_bytes": RECORD_BYTES, "align": ALIGN, "built_unix": build_unix,
        "builder": "tools/pack/build_pack.py", "tp_size": a.tp,
        "tier": {"id": tier_id, "descriptor": tier_desc},
        "plan": {"hash": plan_hash, "descriptor": plan_desc},
        "shard_policy": [{"pattern": p, "kind": k, "why": w} for p, k, w in SHARD_POLICY],
        "ranks": [],
    }
    if a.only:
        manifest["only"] = a.only
    for rank, (rp, path) in enumerate(zip(rplans, paths)):
        recs = []
        for r in rp.records:
            e = {"name": r.name, "dtype": r.dtype, "shape": r.shape,
                 "shard_kind": SHARD_NAMES[r.shard_kind], "layout_id": r.layout_id,
                 "layout": LAYOUT_NAMES[r.layout_id], "data_off": r.data_off,
                 "length": r.length, "sha256": r.sha256.hex()}
            if r.flagged(chunk_bytes):
                e["chunks"] = r.n_chunks(chunk_bytes)
            e["sources"] = r.sources
            recs.append(e)
        manifest["ranks"].append({"rank": rank, "file": os.path.basename(path),
                                  "file_bytes": rp.file_bytes, "data_off": rp.data_off,
                                  "record_count": len(rp.records), "records": recs})
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        f.write(json.dumps(manifest, indent=1) + "\n")
    log(f"build_pack: done in {time.time() - t_start:.1f} s; tier {tier_id[:16]} plan "
        f"{plan_hash[:16]}")
    return 0


def _task_bytes(w) -> int:
    kind, _, p = w
    r = p["w"] if kind == "trio" else p
    return expected_length(r["dtype"], r["shape"]) * (8 if kind == "trio" else 1)


if __name__ == "__main__":
    sys.exit(main())
