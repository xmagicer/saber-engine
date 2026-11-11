#!/usr/bin/env python3
"""Check a Flash-Next (qwen4_exp) EP=2 tier pack against the checkpoint.

The builder verifies as it writes; this re-reads what landed, through a different path (an
independent header parse, not the builder's), and checks, in the `check_devfinal.py` style:

  1. every dense record's shape / dtype / source slices vs the survey's tensor headers
     (the checkpoint's safetensors headers, or `--ref-dir`'s survey copy), and every referenced extent lies inside
     its source tensor;
  2. every written dense record: the SBRPACK1 table entry (name, dtype, shape, layout, sha256) and
     the bytes; the f32 casts recomputed from the checkpoint; the shared-expert trios re-verified
     (Quark's own dequant of the checkpoint == Saber's dequant of the pack, bit for bit);
  3. every referenced dense record: the extents' bytes hash to the manifest's sha256;
  4. the expert tier: header geometry vs the plan, the sha256 table, EVERY record's sha256, and the
     byte-copy verification re-run on a sample (every expert with `--full`), with negative controls
     (a nibble-swapped record, a neighbouring expert, a one-byte flip) that must FAIL;
  5. the PLE index: geometry, every shard's offset vs the survey's headers, the row read law on random rows,
     the hash buffers vs upstream's formula (splitmix64 multipliers, next-prime head sizes), and the
     optional fp8 source's rows against the bf16 rows (same table, same order);
  6. the byte ledger: the plan's consumed bytes per class == the survey's category totals TO THE BYTE
     (recomputed with the survey's own classification from the headers, and equal to `budget.txt` at its
     printed precision); the per-rank resident lines vs the survey's EP=2 per-card lines with every delta
     named;
  7. the e8m0 scan in the manifest == the survey's `scale_scan.txt` exactly (all 73,872 scale tensors);
  8. the nibble-order witness: the survey's correlation against the NVFP4 checkpoint's dequant for the
     layer-0 experts whose NVFP4 bytes are on disk (low-first ~0.99, high-first ~0).

    python3 tools/pack/check_flashnext_pack.py $SABER_PACKS/qwen38-flashnext-quark-trunk8 \
        [--plan plan.json] [--full]

A pack whose manifest says ``"source": "fp4_local"`` is an xtier **v2** pack
(``saber flashnext-pack --source fp4-local``) and is checked by ``check_fp4local``: the v2 header
(version 2, format tag 2 = NVFP4_E4M3G16) and geometry vs the manifest, the sha256 table, EVERY
record's sha256, and the records re-derived from the checkpoint through an independent path
(``fp4local_csf``: numpy CSF decode + ModelOpt's NVFP4 dequant of the CHECKPOINT bytes == Saber's
reading of the PACK bytes -- unpermute, E4M3[ws^T], x gs -- bit for bit, f32), sampled or every
expert with ``--full``; controls (nibble swap, neighbour expert, the neighbour's exceptions,
one flipped byte) must FAIL. A v1 tier must be version 1 (a v2 file is refused by the v1 path).

Exit 0 only if every check passes and every negative control fails.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pack as bp  # noqa: E402
import flashnext_mx as mx  # noqa: E402
try:  # only the FP4-Local (v2) pack path needs it; not part of the public release
    import fp4local_csf as csf  # noqa: E402
except ImportError:
    csf = None

# A directory of the original survey's measurements (meta/quark/tensor_headers.json, budget.txt,
# scale_scan.txt, meta/nvfp4/...), optional: --ref-dir or SABER_FN_REFDIR. Without it the tensor
# headers are read from the checkpoint's own safetensors headers and the survey-only cross-checks
# (budget.txt, scale_scan.txt, the NVFP4 nibble witness) are skipped.
FN1 = os.environ.get("SABER_FN_REFDIR") or None
GI = 2 ** 30
FAIL: list[str] = []


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def check(cond: bool, what: str) -> bool:
    if not cond:
        FAIL.append(what)
        log(f"FAIL {what}")
    return cond


def control(escaped: bool, what: str) -> None:
    """A negative control: `escaped` True means the check did NOT catch the corruption."""
    check(not escaped, f"negative control escaped: {what}")


class Headers:
    """the survey's tensor headers: name -> (dtype, shape, file, absolute data offset, nbytes)."""

    def __init__(self, model: str):
        hp = f"{FN1}/meta/quark/tensor_headers.json" if FN1 else None
        if hp and os.path.exists(hp):
            raw = json.load(open(hp))
        else:
            raw = {}
            for fn in sorted(os.listdir(model)):
                if not fn.endswith(".safetensors"):
                    continue
                with open(os.path.join(model, fn), "rb") as f:
                    (hl,) = struct.unpack("<Q", f.read(8))
                    h = json.loads(f.read(hl))
                for k, v in h.items():
                    if k != "__metadata__":
                        raw[k] = (v["dtype"], v["shape"], fn, v["data_offsets"])
        base: dict[str, int] = {}
        self.t = {}
        for name, (dt, shape, fn, (beg, end)) in raw.items():
            if fn not in base:
                with open(os.path.join(model, fn), "rb") as f:
                    (hl,) = struct.unpack("<Q", f.read(8))
                base[fn] = 8 + hl
            self.t[name] = (dt, shape, fn, base[fn] + beg, end - beg)
        self.model = model
        self.fds: dict[str, int] = {}

    def fd(self, fn: str) -> int:
        if fn not in self.fds:
            self.fds[fn] = os.open(os.path.join(self.model, fn), os.O_RDONLY)
        return self.fds[fn]

    def read(self, name: str) -> np.ndarray:
        dt, shape, fn, off, n = self.t[name]
        b = os.pread(self.fd(fn), n, off)
        return np.frombuffer(b, dtype=np.uint8).reshape(shape) if dt == "U8" else np.frombuffer(b, np.uint8)


# ------------------------------------------------------------------------------------------
# SBRPACK1 tables
# ------------------------------------------------------------------------------------------

def read_pack_table(path: str):
    with open(path, "rb") as f:
        head = f.read(256)
        if head[:8] != bp.MAGIC:
            raise SystemExit(f"{path}: bad magic")
        ver, hb, rb, count = struct.unpack_from("<IIII", head, 8)
        table_off, name_off, name_len, data_off, file_bytes = struct.unpack_from("<QQQQQ", head, 24)
        f.seek(table_off)
        table = f.read(count * rb)
        f.seek(name_off)
        names = f.read(name_len)
    out = {}
    tags = {v[0]: k for k, v in bp.DTYPES.items()}
    for i in range(count):
        e = table[i * rb:(i + 1) * rb]
        noff, nlen, rank, layout = struct.unpack_from("<IIII", e, 0)
        tag, ndim, shard = struct.unpack_from("<HBB", e, 16)
        doff, length = struct.unpack_from("<QQ", e, 24)
        shape = list(struct.unpack_from(f"<{ndim}Q", e, 40))
        name = names[noff:noff + nlen].decode()
        out[name] = {"layout": layout, "dtype": tags[tag], "shape": shape, "shard": shard,
                     "data_off": doff, "length": length, "sha256": e[104:136].hex()}
    return out, file_bytes


# ------------------------------------------------------------------------------------------
# 1-3 dense
# ------------------------------------------------------------------------------------------

def check_dense(pack: str, man: dict, H: Headers) -> dict:
    counts = {"records": 0, "written_checked": 0, "referenced_checked": 0, "trios_verified": 0}
    want_dtype = {"bf16": "BF16", "bf16_merged": "BF16", "f32_from_bf16": "BF16", "raw": "I64",
                  "quark_mxfp4": "U8"}
    for place in man["dense"]:
        dj = json.load(open(os.path.join(pack, place, "dense.json")))
        pack_path = os.path.join(pack, place, "dense.pack")
        table, fbytes = read_pack_table(pack_path) if os.path.exists(pack_path) else ({}, 0)
        if table:
            check(os.path.getsize(pack_path) == fbytes, f"{pack_path}: length {os.path.getsize(pack_path)} != {fbytes}")
        pfd = os.open(pack_path, os.O_RDONLY) if table else None
        trios: dict[str, dict] = {}
        for r in dj["records"]:
            counts["records"] += 1
            st = r["storage"]
            if "pack" in st:
                e = table.get(r["name"])
                if not check(e is not None, f"{place}/{r['name']}: not in dense.pack"):
                    continue
                check(e["shape"] == r["shape"] and e["dtype"] == r["dtype"] and e["length"] == r["bytes"],
                      f"{place}/{r['name']}: table {e['dtype']} {e['shape']} {e['length']} vs manifest")
                data = os.pread(pfd, e["length"], e["data_off"])
                check(hashlib.sha256(data).hexdigest() == e["sha256"], f"{place}/{r['name']}: sha256")
                if r["kind"] == "f32_from_bf16":
                    m = re.match(r"(.*\.linear_attn\.(A_log|dt_bias))$", r["name"])
                    full = np.frombuffer(H.read(m.group(1)).tobytes(), np.uint16)
                    per = full.size // 2
                    rank = int(place[4:])
                    want = (full[rank * per:(rank + 1) * per].astype(np.uint32) << 16).view(np.float32)
                    check(data == want.tobytes(), f"{place}/{r['name']}: f32 cast")
                    check(e["layout"] == bp.LAYOUT_F32_FROM_BF16, f"{r['name']}: layout {e['layout']}")
                elif r["kind"] == "quark_mxfp4":
                    op = r["name"].rsplit(".", 1)[0]
                    trios.setdefault(op, {})[r["part"]] = (data, r)
                counts["written_checked"] += 1
            else:
                h = hashlib.sha256()
                for x in st["checkpoint"]:
                    fd = H.fd(x["file"])
                    if x["count"] == 1:
                        h.update(os.pread(fd, x["length"], x["offset"]))
                    else:
                        span = os.pread(fd, x["stride"] * (x["count"] - 1) + x["length"], x["offset"])
                        a = np.frombuffer(span + b"\0" * (x["stride"] - x["length"]), np.uint8)
                        h.update(a.reshape(x["count"], x["stride"])[:, :x["length"]].tobytes())
                check(h.hexdigest() == st["sha256"], f"{place}/{r['name']}: referenced sha256")
                counts["referenced_checked"] += 1
        # Every source vs the survey's headers, and the extents inside their tensors.
        plan_recs = [x for x in PLAN["dense"] if x["placement"] == place]
        check(len(plan_recs) == len(dj["records"]), f"{place}: {len(dj['records'])} records, plan {len(plan_recs)}")
        for pr, r in zip(plan_recs, dj["records"]):
            check(pr["name"] == r["name"] and pr["shape"] == r["shape"], f"{place}: order/shape {r['name']}")
            for s in pr["sources"]:
                dt, shape, fn, off, n = H.t[s["tensor"]]
                check(dt == want_dtype[pr["kind"]] and shape == s["shape"],
                      f"{s['tensor']}: FN1 header {dt} {shape} vs plan {s['shape']}")
            if "checkpoint" in r["storage"]:
                for x in r["storage"]["checkpoint"]:
                    lo = x["offset"]
                    hi = x["offset"] + x["stride"] * (x["count"] - 1) + x["length"]
                    inside = any(fn == x["file"] and off <= lo and hi <= off + n
                                 for (dt, shape, fn, off, n) in
                                 (H.t[s["tensor"]] for s in pr["sources"]))
                    check(inside, f"{r['name']}: extent {lo}..{hi} outside its sources")
        for op, parts in trios.items():
            w, wr = parts["w"]
            ws, _ = parts["ws"]
            wref, _ = parts["wref"]
            n, k = wr["shape"]
            w_perm = np.frombuffer(w, np.uint8).reshape(n, k // 2)
            ws_t = np.frombuffer(ws, np.uint8).reshape(k // 32, n)
            check(np.array_equal(np.frombuffer(wref, np.uint8), ws_t.max(axis=0)), f"{op}: wref")
            sab = mx.saber_dequant(w_perm, ws_t, n, k)
            row = 0
            src = [s for x in PLAN["dense"] if x["name"] == op + ".w" for s in x["sources"]]
            for s in src:
                q = mx.quark_dequant(H.read(s["tensor"]), H.read(s["tensor"] + "_scale"))
                check(mx.bits_equal(sab[row:row + q.shape[0]], q), f"{op}: dequant vs Quark")
                if row == 0:
                    ctl = mx.quark_dequant(mx.nibble_swap_control(H.read(s["tensor"])),
                                           H.read(s["tensor"] + "_scale"))
                    control(mx.bits_equal(sab[:q.shape[0]], ctl), f"{op}: nibble swap")
                row += q.shape[0]
            counts["trios_verified"] += 1
    return counts


# ------------------------------------------------------------------------------------------
# 4 the expert tier
# ------------------------------------------------------------------------------------------

def _sha_chunk(args):
    path, data_off, stride, first, count = args
    fd = os.open(path, os.O_RDONLY)
    out = []
    for i in range(first, first + count):
        b = os.pread(fd, stride, data_off + i * stride)
        out.append(hashlib.sha256(b).digest())
        os.posix_fadvise(fd, data_off + i * stride, stride, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    return first, out


def check_tier(pack: str, t: dict, H: Headers, full: bool, workers: int) -> dict:
    path = os.path.join(pack, t["file"])
    with open(path, "rb") as f:
        head = f.read(4096)
    check(head[:8] == b"SBRXPRT1", f"{path}: magic")
    ver, hb, rank, layers, first, experts = struct.unpack_from("<IIIIII", head, 8)
    check(ver == 1, f"{path}: version {ver} (the v1 path; FN144's v2 packs go through check_fp4local)")
    stride, payload = struct.unpack_from("<QQ", head, 32)
    subs = [struct.unpack_from("<QQ", head, 48 + 16 * i) for i in range(6)]
    hidden, inter = struct.unpack_from("<II", head, 144)
    table_off, table_bytes, data_off, file_bytes = struct.unpack_from("<QQQQ", head, 152)
    table_sha = head[184:216]
    rec = t["record"]
    keys = ("gate_up_w", "gate_up_ws", "gate_up_wref", "down_w", "down_ws", "down_wref")
    check((rank, layers, first, experts) == (t["rank"], t["layers"], t["first_expert"], t["experts"]),
          f"{path}: header geometry")
    check(stride == rec["stride"] and payload == rec["payload"], f"{path}: stride/payload")
    check(all(tuple(rec[k]) == s for k, s in zip(keys, subs)), f"{path}: sub-record offsets")
    check(os.path.getsize(path) == file_bytes and data_off % 4096 == 0 and stride % 4096 == 0,
          f"{path}: size/alignment")
    with open(path, "rb") as f:
        f.seek(table_off)
        table = f.read(table_bytes)
    check(hashlib.sha256(table).digest() == table_sha, f"{path}: table sha256")
    lo, hi = t.get("built_layers", [0, layers])
    n_rec = (hi - lo) * experts
    per = max(1, n_rec // (workers * 4))
    jobs = [(path, data_off, stride, lo * experts + i, min(per, n_rec - i)) for i in range(0, n_rec, per)]
    bad = 0
    with mp.Pool(workers) as pool:
        for firstrec, shas in pool.imap_unordered(_sha_chunk, jobs):
            for j, d in enumerate(shas):
                i = firstrec + j
                if d != table[i * 32:(i + 1) * 32]:
                    bad += 1
    check(bad == 0, f"{path}: {bad} records' sha256 differ from the table")
    # Byte-copy verification, sample or full.
    fd = os.open(path, os.O_RDONLY)
    rng = random.Random(0xF12)
    todo = [(l, e) for l in range(lo, hi) for e in range(experts)] if full else \
        [(l, e) for l in range(lo, hi) for e in rng.sample(range(experts), 2)]
    verified = 0
    for l, e in todo:
        g_id = first + e
        b = np.frombuffer(os.pread(fd, stride, data_off + (l * experts + e) * stride), np.uint8)
        sub = lambda k, shape: b[rec[k][0]:rec[k][0] + rec[k][1]].reshape(shape)  # noqa: E731
        sab_gu = mx.saber_dequant(sub("gate_up_w", (2 * inter, hidden // 2)),
                                  sub("gate_up_ws", (hidden // 32, 2 * inter)), 2 * inter, hidden)
        sab_d = mx.saber_dequant(sub("down_w", (hidden, inter // 2)),
                                 sub("down_ws", (inter // 32, hidden)), hidden, inter)
        p = f"model.language_model.layers.{l}.mlp.experts.{g_id}"
        q = {proj: mx.quark_dequant(H.read(f"{p}.{proj}.weight"), H.read(f"{p}.{proj}.weight_scale"))
             for proj in ("gate_proj", "up_proj", "down_proj")}
        ok = (mx.bits_equal(sab_gu[:inter], q["gate_proj"]) and mx.bits_equal(sab_gu[inter:], q["up_proj"])
              and mx.bits_equal(sab_d, q["down_proj"]))
        check(ok, f"{path}: layer {l} expert {g_id} dequant vs Quark")
        check(np.array_equal(sub("down_wref", (hidden,)), sub("down_ws", (inter // 32, hidden)).max(axis=0)),
              f"{path}: layer {l} expert {g_id} wref")
        verified += ok
        if e == todo[0][1] or not full:
            # controls: neighbour expert, nibble swap, one flipped byte (sha)
            nb = f"model.language_model.layers.{l}.mlp.experts.{first + (e + 1) % experts}"
            qn = mx.quark_dequant(H.read(f"{nb}.gate_proj.weight"), H.read(f"{nb}.gate_proj.weight_scale"))
            control(mx.bits_equal(sab_gu[:inter], qn), f"layer {l} expert {g_id} vs its neighbour")
            qs = mx.quark_dequant(mx.nibble_swap_control(H.read(f"{p}.gate_proj.weight")),
                                  H.read(f"{p}.gate_proj.weight_scale"))
            control(mx.bits_equal(sab_gu[:inter], qs), f"layer {l} expert {g_id} nibble swap")
            flip = bytearray(b.tobytes())
            flip[rec["down_w"][0] + 7] ^= 0x10
            i = l * experts + e
            control(hashlib.sha256(bytes(flip)).digest() == table[i * 32:(i + 1) * 32],
                    f"layer {l} expert {g_id} one-byte flip vs sha256")
    os.close(fd)
    return {"records_hashed": n_rec, "experts_reverified": verified, "sampled": not full}


# ------------------------------------------------------------------------------------------
# 5 PLE
# ------------------------------------------------------------------------------------------

M64 = (1 << 64) - 1


def splitmix64(v: int) -> int:
    v = (v + 0x9E3779B97F4A7C15) & M64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & M64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & M64
    return v ^ (v >> 31)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def upstream_hash_buffers(cfg: dict, dense_id: int = 0):
    """amd/ple_layer.py `_make_layer_multipliers` + `_make_vocab_layout`, transcribed."""
    t = cfg["text_config"]
    n = t["ngram_size"]
    vocab = t["vocab_size"]
    seed = t.get("seed", 1234)
    max_mult = ((1 << 63) - 1) // vocab
    half = max(1, max_mult // 2)
    base = seed + 10007 * dense_id
    mults = [2 * (splitmix64(base + 0x9E3779B97F4A7C15 * (i + 1)) % half) + 1 for i in range(n)]
    heads = (n - 1) * t["heads_per_ngram"]
    sizes, offs, off = [], [], 0
    for local in range(heads):
        gh = dense_id * heads + local
        p = t["ngram_vocab_size_base"] - 1
        for _ in range(gh + 1):
            c = p + 1
            while not is_prime(c):
                c += 1
            p = c
        sizes.append(p)
        offs.append(off)
        off += p
    return mults, sizes, offs, off


def check_ple(pack: str, H: Headers, cfg: dict) -> dict:
    idx = json.load(open(os.path.join(pack, "ple", "ple_index.json")))
    tab = idx["table"]
    src = idx["sources"]["ckpt_bf16"]
    pref = PLAN["ple"]["tensor_prefix"]
    check(tab["shards"] == 128 and tab["rows_per_shard"] == 2_500_012 and tab["row_dim"] == 160
          and tab["rows_per_token"] == 16, f"PLE geometry {tab}")
    check(src["row_bytes"] == 320 and src["bytes_per_token"] == 5120, "PLE row bytes")
    for s in src["shards"]:
        dt, shape, fn, off, n = H.t[s["tensor"]]
        check(dt == "BF16" and shape == [2_500_012, 160] and fn == s["file"] and off == s["offset"]
              and n == s["bytes"], f"PLE shard {s['shard']} vs FN1 headers")
    # hash buffers vs upstream's formula
    mults, sizes, offs, total = upstream_hash_buffers(cfg)
    lp = pref.split(".ngram_embedding.")[0]
    got = {k: np.frombuffer(H.read(f"{lp}.{k}").tobytes(), np.int64).tolist()
           for k in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")}
    check(got["layer_multipliers"] == mults, f"PLE multipliers {got['layer_multipliers']} vs {mults}")
    check(got["ngram_heads_vocab_sizes"] == sizes, "PLE head sizes vs next-prime law")
    check(got["ngram_heads_offsets"] == offs, "PLE head offsets")
    padded = -(-total // 128) * 128
    check(padded == tab["rows"] and -(-padded // 128) == tab["rows_per_shard"],
          f"PLE rows: sum of primes {total} -> padded {padded}, index says {tab['rows']}")
    control(upstream_hash_buffers(cfg, dense_id=1)[0] == mults, "PLE multipliers at the wrong dense id")
    # the row read law on random rows: index path vs FN1-header path
    rng = random.Random(7)
    rows = [rng.randrange(total) for _ in range(64)] + [0, tab["rows_per_shard"], total - 1]
    sh = {s["shard"]: s for s in src["shards"]}
    fp8 = idx["sources"].get("qwen_fp8") if FP8FD else None
    corr = []
    for r in rows:
        k, local = r // tab["rows_per_shard"], r % tab["rows_per_shard"]
        s = sh[k]
        a = os.pread(H.fd(s["file"]), 320, s["offset"] + local * 320)
        dt, shape, fn, off, n = H.t[f"{pref}{k}.weight"]
        # O_DIRECT law: the 512-aligned span covering the row
        span_lo = (off + local * 320) // 512 * 512
        span_hi = -(-(off + local * 320 + 320) // 512) * 512
        span = os.pread(H.fd(fn), span_hi - span_lo, span_lo)
        b = span[off + local * 320 - span_lo:][:320]
        check(a == b and span_hi - span_lo in (512, 1024), f"PLE row {r}: index vs header read")
        if fp8:
            fs = fp8["shards"][k]
            f8 = np.frombuffer(os.pread(FP8FD[0], 160, fs["offset"] + local * 160), np.uint8)
            v8 = e4m3(f8)
            vb = (np.frombuffer(a, np.uint16).astype(np.uint32) << 16).view(np.float32)
            if np.abs(vb).sum() > 0:
                corr.append(float(np.corrcoef(v8, vb)[0, 1]))
    out = {"rows_checked": len(rows), "straddle_512": sorted({s["straddle_512"] for s in src["shards"]}),
           "offset_mod_512": sorted({s["offset_mod_512"] for s in src["shards"]}),
           "hashed_shards": sum(1 for s in src["shards"] if "sha256" in s)}
    if fp8:
        ts = fp8.get("table_scale") or {}
        check(ts.get("dtype") == "BF16" and ts.get("shape") == [1] and (ts.get("value") or 0) > 0,
              f"fp8 PLE: the per-table scale is missing or not positive ({ts})")
        out["fp8_table_scale"] = ts.get("value")
        out["fp8_row_corr_min"] = min(corr)
        out["fp8_row_corr_median"] = float(np.median(corr))
        check(min(corr) > 0.95, f"fp8 PLE rows do not match the bf16 rows (min corr {min(corr):.3f})")
        # control: the same fp8 rows against a DIFFERENT bf16 row must not correlate
        k0 = sh[0]
        a0 = np.frombuffer(os.pread(H.fd(k0["file"]), 320, k0["offset"] + 5 * 320), np.uint16)
        b0 = e4m3(np.frombuffer(os.pread(FP8FD[0], 160, fp8["shards"][0]["offset"] + 6 * 160), np.uint8))
        c = float(np.corrcoef((a0.astype(np.uint32) << 16).view(np.float32), b0)[0, 1])
        control(c > 0.95, f"fp8 row 6 vs bf16 row 5 correlate ({c:.3f})")
    return out


FP8FD: list[int] = []


def e4m3(u8: np.ndarray) -> np.ndarray:
    s = (u8 >> 7) & 1
    e = ((u8 >> 3) & 0xF).astype(np.int32)
    m = (u8 & 7).astype(np.float64)
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    return np.where(s == 1, -v, v)


# ------------------------------------------------------------------------------------------
# 6-8 ledger, scan, nibble witness
# ------------------------------------------------------------------------------------------

def fn1_categories(H: Headers) -> dict:
    """budget.py's classification, verbatim, over the survey's headers."""
    by: dict[str, int] = {}
    for n, (dt, sh, fn, off, b) in H.t.items():
        if n.startswith('model.visual'):
            k = 'vision'
        elif n.startswith('mtp.'):
            if '.mlp.experts.' in n: k = 'mtp.experts'  # noqa: E701
            elif 'hyper_connection' in n or n.startswith('mtp.pre_fc_norm_hidden'): k = 'mtp.hc'  # noqa: E701
            elif '.mlp.gate.' in n or 'shared_expert_gate' in n: k = 'mtp.router'  # noqa: E701
            elif '.self_attn.k_proj' in n or '.self_attn.v_proj' in n: k = 'mtp.attn.kv'  # noqa: E701
            elif '.self_attn.indexer' in n or '_norm' in n: k = 'mtp.repl.small'  # noqa: E701
            else: k = 'mtp.tp'  # noqa: E701
        elif n == 'lm_head.weight': k = 'lm_head'  # noqa: E701
        elif 'embed_tokens' in n: k = 'embed'  # noqa: E701
        elif 'ngram_embedding.shard' in n: k = 'ple.table'  # noqa: E701
        elif '.ple.' in n: k = 'ple.dense'  # noqa: E701
        elif '.mlp.experts.' in n: k = 'experts'  # noqa: E701
        elif '.mlp.shared_expert.' in n: k = 'shared'  # noqa: E701
        elif '.mlp.gate.' in n or 'shared_expert_gate' in n: k = 'router'  # noqa: E701
        elif 'hyper_connection' in n or 'norm.weight' in n and 'layers' not in n: k = 'hc'  # noqa: E701
        elif '.linear_attn.' in n: k = 'gdn'  # noqa: E701
        elif 'self_attn.indexer' in n: k = 'indexer'  # noqa: E701
        elif '.self_attn.k_proj' in n or '.self_attn.v_proj' in n: k = 'attn.kv'  # noqa: E701
        elif '.self_attn.q_norm' in n or '.self_attn.k_norm' in n: k = 'indexer'  # noqa: E701
        elif '.self_attn.' in n: k = 'attn.qo'  # noqa: E701
        else: raise SystemExit('unclassified ' + n)  # noqa: E701
        by[k] = by.get(k, 0) + b
    return by


MTP_FN1 = ("mtp.tp", "mtp.hc", "mtp.router", "mtp.attn.kv", "mtp.repl.small")


def check_ledger(man: dict, H: Headers) -> list[str]:
    by = fn1_categories(H)
    # budget.txt's first line, parsed
    printed = {}
    if FN1 and os.path.exists(f"{FN1}/budget.txt"):
        line = open(f"{FN1}/budget.txt").readline()
        printed = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
    for k, v in printed.items():
        check(round(by[k] / GI, 4) == v, f"FN1 category {k}: headers {by[k] / GI:.4f} vs budget.txt {v}")
    consumed = {x["class"]: x["bytes"] for x in man["consumed_by_class"]}
    fn1_of = {"experts": "experts", "gdn": "gdn", "attn.qo": "attn.qo", "attn.kv": "attn.kv",
              "indexer": "indexer", "hc": "hc", "router": "router", "shared_expert": "shared",
              "ple.dense": "ple.dense", "embed": "embed", "lm_head": "lm_head",
              "mtp.experts": "mtp.experts"}
    lines = []
    for mine, theirs in fn1_of.items():
        ok = check(consumed[mine] == by[theirs], f"ledger {mine}: plan {consumed[mine]} vs FN1 {by[theirs]}")
        lines.append(f"| {mine} | {consumed[mine]:,} | {by[theirs]:,} | {'=' if ok else 'DIFF'} |")
    mtp = sum(by[k] for k in MTP_FN1)
    ok = check(consumed["mtp.dense"] == mtp, f"ledger mtp.dense {consumed['mtp.dense']} vs FN1 {mtp}")
    lines.append(f"| mtp.dense | {consumed['mtp.dense']:,} | {mtp:,} (tp+hc+router+attn.kv+repl.small) | {'=' if ok else 'DIFF'} |")
    ple_bytes = 128 * 2_500_012 * 320
    ok = check(ple_bytes == by["ple.table"], "PLE table bytes vs FN1")
    lines.append(f"| ple.table (index, in place) | {ple_bytes:,} | {by['ple.table']:,} | {'=' if ok else 'DIFF'} |")
    total_plan = sum(consumed.values()) + ple_bytes + by["vision"]
    ok = check(total_plan == sum(by.values()), "every checkpoint byte is consumed once")
    lines.append(f"| **all (+ vision {by['vision']:,})** | {total_plan:,} | {sum(by.values()):,} | {'=' if ok else 'DIFF'} |")
    # control: a ledger one byte off must be caught
    control(consumed["experts"] + 1 == by["experts"], "ledger +1 byte")
    return lines


def per_rank_lines(man: dict, H: Headers) -> list[str]:
    """Resident bytes per rank (rank + shared) vs the survey's EP=2 per-card line, deltas named."""
    by = fn1_categories(H)
    fn1 = {  # budget.py per_card(tp=2, ep=2)
        "experts": by["experts"] / 2, "gdn": by["gdn"] / 2, "attn.qo": by["attn.qo"] / 2,
        "attn.kv": by["attn.kv"] / 2, "indexer": by["indexer"], "hc": by["hc"], "router": by["router"],
        "shared_expert": by["shared"] / 2, "ple.dense": by["ple.dense"], "embed": by["embed"] / 2,
        "lm_head": by["lm_head"] / 2, "mtp.experts": by["mtp.experts"] / 2,
        "mtp.dense": by["mtp.tp"] / 2 + by["mtp.attn.kv"] / 2 + by["mtp.hc"] + by["mtp.router"]
        + by["mtp.repl.small"],
    }
    why = {
        "experts": "+ wref (3,840 B per expert-layer, derived from ws; not a checkpoint byte)",
        "gdn": "+ A_log/dt_bias widened to f32 (the kernels' dtype) + the 128-wide norm replicated (FN1 halved it)",
        "shared_expert": "replicated (§6: never split by K; FN1 halved it) + wref",
        "mtp.dense": "the bf16 shared expert replicated (FN1 halved it with mtp.tp)",
    }
    res: dict[str, int] = {}
    for x in man["ledger"]:
        if x["placement"] in ("rank0", "shared"):
            res[x["class"]] = res.get(x["class"], 0) + x["bytes"]
    lines = []
    tot_mine = tot_fn1 = 0
    for k, v in fn1.items():
        mine = res.get(k, 0)
        tot_mine += mine
        tot_fn1 += v
        d = mine - v
        lines.append(f"| {k} | {mine / GI:.3f} | {v / GI:.3f} | {d:+,.0f} B | {why.get(k, '=' if d == 0 else '?')} |")
        if d != 0:
            check(k in why, f"per-rank {k}: unexplained delta {d}")
    lines.append(f"| **per rank** | **{tot_mine / GI:.3f}** | **{tot_fn1 / GI:.3f}** | {tot_mine - tot_fn1:+,.0f} B | |")
    return lines


def check_aliases(pack: str, man: dict, H: Headers) -> None:
    """FN9's loader aliases: every QSA layer's merged projection per rank names exactly the rows
    [q|gate half r | k half r | v half r | index_qk whole] (N 7,296) and its extents' bytes hash to
    the recorded sha256, which equals the hash of the member tensors' rows read independently;
    the CONTROL (the other rank's k half) must hash differently."""
    info = man.get("aliases")
    if not check(info is not None, "the pack carries the FN9 aliases (build_flashnext_pack.py --only-aliases)"):
        return

    def rows_bytes(name, r0, r1):
        dt, shape, fn, off, _ = H.t[name]
        rb = int(np.prod(shape[1:])) * 2
        return os.pread(H.fd(fn), (r1 - r0) * rb, off + r0 * rb)

    n = 0
    for place in ("rank0", "rank1"):
        d = json.load(open(os.path.join(pack, place, "aliases.json")))
        r = int(place[-1])
        for a in d["aliases"]:
            check(a["shape"] == [7296, 2560], f"{a['name']} {place}: shape {a['shape']}")
            mem = a["members"]
            want = [(6144 * r, 6144 * (r + 1)), (256 * r, 256 * (r + 1)), (256 * r, 256 * (r + 1)), (0, 640)]
            check([tuple(m["rows"]) for m in mem] == want, f"{a['name']} {place}: member rows")
            h = hashlib.sha256()
            for m in mem:
                h.update(rows_bytes(m["tensor"], *m["rows"]))
            check(h.hexdigest() == a["storage"]["sha256"], f"{a['name']} {place}: bytes == members")
            hc = hashlib.sha256()
            for i, m in enumerate(mem):
                rr = m["rows"] if i != 1 else (256 * (1 - r), 256 * (2 - r))
                hc.update(rows_bytes(m["tensor"], *rr))
            control(hc.hexdigest() == a["storage"]["sha256"], f"{a['name']} {place}: the other rank's k half")
            n += 1
    check(n == 26, f"26 aliases (12 QSA layers + MTP, x 2 ranks): {n}")
    log(f"   aliases: {n} checked (bytes == members, controls differ)")


def check_scan(man: dict) -> None:
    if not (FN1 and os.path.exists(f"{FN1}/scale_scan.txt")):
        v = man["verification"]
        check(v["negative_controls_matched"] == 0 and v["negative_controls"] > 0, "build controls")
        log("   (no --ref-dir scale_scan.txt: the build's own controls only)")
        return
    txt = open(f"{FN1}/scale_scan.txt").read()
    fn1_vals = {str(k): v for k, v in ast.literal_eval(re.search(r"present (\{.*?\})", txt).group(1)).items()}
    fn1_spread = {str(k): v for k, v in ast.literal_eval(re.search(r"histogram (\{.*?\})", txt).group(1)).items()}
    v = man["verification"]
    check(v["scale_tensors_scanned"] == 73872, f"scan covered {v['scale_tensors_scanned']} tensors, FN1 73,872")
    check(v["e8m0_values"] == fn1_vals, f"e8m0 histogram differs from the survey's: {v['e8m0_values']}")
    check(v["per_row_spread"] == fn1_spread, f"spread histogram differs from the survey's: {v['per_row_spread']}")
    check(v["negative_controls_matched"] == 0 and v["negative_controls"] > 0, "build controls")


def nibble_witness(H: Headers, n: int = 6) -> list[str]:
    nv = f"{FN1}/meta/nvfp4/tensor_headers.json" if FN1 else ""
    nd = os.environ.get("SABER_FN_NVFP4_CKPT", "")
    if not (nv and os.path.exists(nv) and nd and os.path.isdir(nd)):
        return ["skipped (needs --ref-dir with meta/nvfp4 and SABER_FN_NVFP4_CKPT, the NVFP4 sibling checkpoint)"]
    nh = json.load(open(nv))
    out = []
    got = 0
    for e in range(512):
        name = f"model.language_model.layers.0.mlp.experts.{e}.gate_proj"
        parts = [name + s for s in (".weight", ".weight_scale", ".weight_scale_2")]
        if not all(p in nh and os.path.exists(os.path.join(nd, nh[p][2])) for p in parts):
            continue

        def raw(p):
            dt, sh, fn, (b0, b1) = nh[p]
            with open(os.path.join(nd, fn), "rb") as f:
                (hl,) = struct.unpack("<Q", f.read(8))
                f.seek(8 + hl + b0)
                return np.frombuffer(f.read(b1 - b0), np.uint8), sh
        w, sh = raw(parts[0])
        s, ss = raw(parts[1])
        g, _ = raw(parts[2])
        g = np.frombuffer(g.tobytes(), np.float32)[0]
        codes = np.stack([w & 0xF, w >> 4], -1).reshape(sh[0], -1)
        mag = mx.E2M1[codes & 7].astype(np.float64)
        v = np.where(codes & 8, -mag, mag).reshape(sh[0], -1, 16)
        ref = (v * e4m3(s.reshape(ss))[..., None] * g).reshape(sh[0], -1)
        qw, qs = H.read(name + ".weight"), H.read(name + ".weight_scale")
        low = mx.quark_dequant(qw, qs).astype(np.float64)
        high = mx.quark_dequant(mx.nibble_swap_control(qw), qs).astype(np.float64)
        cl = float(np.corrcoef(low.ravel(), ref.ravel())[0, 1])
        ch = float(np.corrcoef(high.ravel(), ref.ravel())[0, 1])
        check(cl > 0.98 and abs(ch) < 0.05, f"nibble witness l0 e{e}: low {cl:.4f} high {ch:.4f}")
        out.append(f"l0 e{e} gate_proj: corr low-first {cl:.5f}, high-first {ch:+.5f}")
        got += 1
        if got >= n:
            break
    return out


PLAN: dict = {}


# ------------------------------------------------------------------------------------------
# FN144: the FP4-Local v2 tier
# ------------------------------------------------------------------------------------------

def _pack_dequant(b: np.ndarray, rec: dict, key: str, n: int, k: int, gsi: int) -> np.ndarray:
    """Saber's reading of a v2 record's projection block: unpermute w, E4M3(ws^T) per 16, x gs."""
    o, ln = rec[f"{key}_w"]
    w = mx.unpermute_w(b[o:o + ln].reshape(n, k // 2), n, k)
    o, ln = rec[f"{key}_ws"]
    ws = b[o:o + ln].reshape(k // 16, n).T
    o, ln = rec[f"{key}_gs"]
    gs = b[o:o + ln].view(np.float32)
    codes = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(n, k)
    v = csf.E2M1_SIGNED[codes].reshape(n, k // 16, 16) * csf.E4M3[ws].reshape(n, k // 16, 1)
    if key == "gate_up":
        g = np.where(np.arange(n)[:, None, None] < n // 2, gs[0], gs[1]).astype(np.float32)
    else:
        g = np.float32(gs[gsi])
    return (v * g).reshape(n, k).astype(np.float32)


def _ckpt_dequant(S, l: int, e: int, proj: str, swap: bool = False, exc_of: int | None = None):
    x = f"model.language_model.layers.{l}.mlp.experts.{e}.{proj}"
    w = S.read(f"{x}.weight")
    n, k = w.shape[0], w.shape[1] * 2
    exc_src = x if exc_of is None else f"model.language_model.layers.{l}.mlp.experts.{exc_of}.{proj}"
    sc = csf.csf_decode(S.read(f"{x}.weight_scale.nvfp4_csf_fixed"),
                        S.read(f"{exc_src}.weight_scale.nvfp4_csf_exceptions"), n, k // 16)[0]
    if swap:
        w = (w >> 4) | ((w & 0xF) << 4)
    return csf.nvfp4_dequant(w, sc, float(S.read(f"{x}.weight_scale_2").reshape(-1)[0]))


def check_fp4local(pack: str, man: dict, full: bool, workers: int) -> None:
    S = csf.Shards(man["checkpoint"]["dir"])
    for t in man["expert_tiers"]:
        path = os.path.join(pack, t["file"])
        with open(path, "rb") as f:
            head = f.read(4096)
        check(head[:8] == b"SBRXPRT1", f"{path}: magic")
        ver, hb, rank, layers, first, experts = struct.unpack_from("<IIIIII", head, 8)
        (tag,) = struct.unpack_from("<I", head, 256)
        check(ver == 2 and tag == 2, f"{path}: v2 NVFP4_E4M3G16 (version {ver}, tag {tag})")
        stride, payload = struct.unpack_from("<QQ", head, 32)
        subs = [struct.unpack_from("<QQ", head, 48 + 16 * i) for i in range(6)]
        hidden, inter = struct.unpack_from("<II", head, 144)
        table_off, table_bytes, data_off, file_bytes = struct.unpack_from("<QQQQ", head, 152)
        rec = t["record"]
        keys = ("gate_up_w", "gate_up_ws", "gate_up_gs", "down_w", "down_ws", "down_gs")
        check((rank, layers, first, experts) == (t["rank"], t["layers"], t["first_expert"], t["experts"]),
              f"{path}: header geometry")
        check(stride == rec["stride"] == 676 * 4096 and payload == rec["payload"], f"{path}: stride/payload")
        check(all(tuple(rec[k]) == sv for k, sv in zip(keys, subs)), f"{path}: sub-record offsets")
        check(subs[1][1] == (hidden // 16) * 2 * inter and subs[4][1] == (inter // 16) * hidden
              and subs[2][1] == 8 and subs[5][1] == 4, f"{path}: E4M3 g16 + f32 gs sizes")
        check(os.path.getsize(path) == file_bytes == data_off + layers * experts * stride,
              f"{path}: size")
        with open(path, "rb") as f:
            f.seek(table_off)
            table = f.read(table_bytes)
        check(hashlib.sha256(table).digest() == head[184:216], f"{path}: table sha256")
        n_rec = layers * experts
        per = max(1, n_rec // (workers * 4))
        jobs = [(path, data_off, stride, i, min(per, n_rec - i)) for i in range(0, n_rec, per)]
        bad = 0
        with mp.Pool(workers) as pool:
            for firstrec, shas in pool.imap_unordered(_sha_chunk, jobs):
                for j, d in enumerate(shas):
                    i = firstrec + j
                    bad += d != table[i * 32:(i + 1) * 32]
        check(bad == 0, f"{path}: {bad} records' sha256 differ from the table")
        fd = os.open(path, os.O_RDONLY)
        rng = random.Random(0xF144)
        lo = t["layer_offset"]
        todo = [(li, e) for li in range(layers) for e in range(experts)] if full else \
            [(li, e) for li in range(layers) for e in rng.sample(range(experts), 4)]
        verified = 0
        for li, e in todo:
            l, g = lo + li, first + e
            b = np.frombuffer(os.pread(fd, stride, data_off + (li * experts + e) * stride), np.uint8)
            gu = _pack_dequant(b, rec, "gate_up", 2 * inter, hidden, 0)
            dn = _pack_dequant(b, rec, "down", hidden, inter, 0)
            ref = {p: _ckpt_dequant(S, l, g, p) for p in ("gate_proj", "up_proj", "down_proj")}
            ok = (mx.bits_equal(gu[:inter], ref["gate_proj"]) and mx.bits_equal(gu[inter:], ref["up_proj"])
                  and mx.bits_equal(dn, ref["down_proj"]))
            check(ok, f"{path}: layer {l} expert {g} pack dequant vs the checkpoint's")
            verified += ok
            if not full or e == todo[0][1]:
                nb = first + (e + 1) % experts
                control(mx.bits_equal(gu[:inter], _ckpt_dequant(S, l, nb, "gate_proj")),
                        f"layer {l} expert {g} vs its neighbour")
                control(mx.bits_equal(gu[:inter], _ckpt_dequant(S, l, g, "gate_proj", swap=True)),
                        f"layer {l} expert {g} nibble swap")
                control(mx.bits_equal(dn, _ckpt_dequant(S, l, g, "down_proj", exc_of=nb)),
                        f"layer {l} expert {g} the neighbour's exceptions")
                flip = bytearray(b.tobytes())
                flip[rec["down_ws"][0] + 3] ^= 0x01
                i = li * experts + e
                control(hashlib.sha256(bytes(flip)).digest() == table[i * 32:(i + 1) * 32],
                        f"layer {l} expert {g} one-byte flip vs sha256")
        os.close(fd)
        log(f"   {t['file']}: v2 {n_rec} records hashed, {verified}/{len(todo)} experts re-derived "
            f"bit-exact ({'full' if full else 'sampled'})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--plan", default=None, help="the plan the pack was built from (default PACK/plan.json)")
    ap.add_argument("--checkpoint", default=None,
                    help="the checkpoint directory (default: the manifest's checkpoint.dir)")
    ap.add_argument("--ref-dir", default=None, help="optional survey measurements (see FN1 above)")
    ap.add_argument("--full", action="store_true", help="re-verify every expert, not a sample")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-scan", action="store_true", help="a partial (--layers) build")
    args = ap.parse_args()
    global FN1
    FN1 = args.ref_dir or FN1
    args.plan = args.plan or os.path.join(args.pack, "plan.json")
    man = json.load(open(os.path.join(args.pack, "manifest.json")))
    if man.get("source") == "fp4_local":
        raw = open(os.path.join(args.pack, "plan.json"), "rb").read()
        check(hashlib.sha256(raw).hexdigest() == man["plan_sha256"], "the plan is the one the pack was built from")
        log(f"== {args.pack}: FN144 FP4-Local v2 pack, layers {man['layers']}")
        log("== 4 expert tier (v2)")
        check_fp4local(args.pack, man, args.full, args.workers)
        if FAIL:
            log(f"\nFAILED: {len(FAIL)} checks")
            for f in FAIL[:20]:
                log(f"  {f}")
            return 1
        log("\nALL CHECKS PASS (every negative control failed as it must)")
        return 0
    raw = open(args.plan, "rb").read()
    PLAN.update(json.loads(raw))
    check(hashlib.sha256(raw).hexdigest() == man["plan_sha256"], "the plan is the one the pack was built from")
    model = args.checkpoint or man["checkpoint"]["dir"]
    cfg = json.load(open(os.path.join(model, "config.json")))
    H = Headers(model)
    log(f"== {args.pack}: plan {man['plan_sha256'][:16]}…, {len(H.t)} checkpoint tensors (FN1 headers)")
    fp8 = json.load(open(os.path.join(args.pack, "ple", "ple_index.json")))["sources"].get("qwen_fp8")
    if fp8 and os.path.exists(fp8["file"]):
        FP8FD.append(os.open(fp8["file"], os.O_RDONLY))
    elif fp8:
        log(f"   note: the optional fp8 PLE source {fp8['file']} is absent; its rows are not checked")

    log("== 1-3 dense records")
    d = check_dense(args.pack, man, H)
    log(f"   {d}")
    log("== 4 expert tier")
    for t in man["expert_tiers"]:
        log(f"   {t['file']}: {check_tier(args.pack, t, H, args.full, args.workers)}")
    log("== 5 PLE index")
    log(f"   {check_ple(args.pack, H, cfg)}")
    log("== 6 ledger (consumed bytes vs FN1, to the byte)")
    log("| class | plan | FN1 | |")
    for ln in check_ledger(man, H):
        log(ln)
    log("   per-rank resident (rank0 + shared) vs the survey's EP=2 per-card line")
    log("| class | pack GiB | FN1 GiB | delta | why |")
    for ln in per_rank_lines(man, H):
        log(ln)
    if not args.skip_scan:
        log("== 7 e8m0 scan vs the survey's scale_scan.txt")
        check_scan(man)
        log(f"   {man['verification']}")
    log("== 9 FN9 loader aliases (the QSA merged projection, N 7,296 per rank)")
    check_aliases(args.pack, man, H)
    log("== 8 nibble-order witness vs the NVFP4 checkpoint's dequant")
    for ln in nibble_witness(H):
        log(f"   {ln}")
    if FAIL:
        log(f"\nFAILED: {len(FAIL)} checks")
        for f in FAIL[:20]:
            log(f"  {f}")
        return 1
    log("\nALL CHECKS PASS (every negative control failed as it must)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
