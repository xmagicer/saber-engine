#!/usr/bin/env python3
"""Check an fp8-head pack's `lm_head.w8` / `lm_head.ws` against the checkpoint, on the host.

  python3 tools/pack/check_fp8_head.py <pack dir> --ckpt <checkpoint dir | .safetensors> [--bf16-pack DIR]

`lm_head.w8` must be the checkpoint's e4m3 bytes of each rank's dim-0 slice, byte for byte, and
`lm_head.ws` the slice's bf16 `weight_scale` widened to f32 (compared as bits). With `--bf16-pack`
it also dequantizes a spread of rows from the fp8 pack's OWN records (numpy: e4m3 -> f32, times the
f32 scale, bf16 round-to-nearest-even) and compares them against the bf16 head's `lm_head.w` --
the host half of the claim the kernel gate makes on the card. The CONTROL is the same rows rounded
through f16 first, which must differ.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def find_head_file(ckpt):
    """The .safetensors file of `ckpt` (a file, or a checkpoint directory) holding lm_head.weight."""
    if os.path.isfile(ckpt):
        return ckpt
    for f in sorted(os.listdir(ckpt)):
        if f.endswith(".safetensors") and "lm_head.weight" in st_header(os.path.join(ckpt, f))[0]:
            return os.path.join(ckpt, f)
    raise SystemExit(f"{ckpt}: no .safetensors file holds lm_head.weight")


def ckpt_tensors(path):
    """(offset of lm_head.weight, offset of lm_head.weight_scale, N, K) in `path`."""
    h, base = st_header(path)
    if "lm_head.weight_scale" not in h:
        raise SystemExit(f"{path}: no lm_head.weight_scale -- not an fp8-head checkpoint")
    w, s = h["lm_head.weight"], h["lm_head.weight_scale"]
    n, k = w["shape"]
    assert w["dtype"] == "F8_E4M3", w
    assert s["dtype"] == "BF16" and s["shape"] == [n, 1], s
    return base + w["data_offsets"][0], base + s["data_offsets"][0], n, k


def e4m3_lut():
    """Every e4m3fn code as f32 (NaN at S.1111.111)."""
    out = np.zeros(256, np.float32)
    for c in range(256):
        sg, e, m = c >> 7, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = m * 2.0 ** -9
        else:
            v = (1 + m / 8) * 2.0 ** (e - 7)
        out[c] = -v if sg else v
    return out


def bf16_rne(f32):
    b = f32.view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def record(pack, rank, name):
    m = json.load(open(os.path.join(pack, "manifest.json")))
    rk = m["ranks"][rank]
    for r in rk["records"]:
        if r["name"] == name:
            return os.path.join(pack, rk["file"]), r
    raise SystemExit(f"{pack} rank {rank}: no record {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack", help="the fp8-head pack directory (manifest.json + rank<r>.pack)")
    ap.add_argument("--ckpt", required=True,
                    help="the checkpoint directory (or the .safetensors file holding lm_head)")
    ap.add_argument("--bf16-pack", default="", help="also compare against a bf16-head pack")
    a = ap.parse_args()
    a.ckpt = find_head_file(a.ckpt)
    off_w, off_s, n_full, K = ckpt_tensors(a.ckpt)
    TP = json.load(open(os.path.join(a.pack, "manifest.json")))["tp_size"]
    if n_full % TP:
        raise SystemExit(f"lm_head rows {n_full} do not split over tp {TP}")
    NLOC = n_full // TP
    bad = 0
    lut = e4m3_lut()
    rng = np.random.default_rng(0x4E34)
    with open(a.ckpt, "rb") as ck:
        for rank in range(TP):
            fw, rw = record(a.pack, rank, "lm_head.w8")
            fs, rs = record(a.pack, rank, "lm_head.ws")
            assert (rw["dtype"], rw["shape"], rw["layout_id"]) == ("F8_E4M3", [NLOC, K], 15), rw
            assert (rs["dtype"], rs["shape"], rs["layout_id"]) == ("F32", [NLOC], 16), rs
            ck.seek(off_w + rank * NLOC * K)
            src = np.frombuffer(ck.read(NLOC * K), np.uint8)
            with open(fw, "rb") as f:
                f.seek(rw["data_off"])
                got = np.frombuffer(f.read(rw["length"]), np.uint8)
            dw = int((src != got).sum()) if src.size == got.size else -1
            ck.seek(off_s + rank * NLOC * 2)
            sb = np.frombuffer(ck.read(NLOC * 2), np.uint16)
            want = (sb.astype(np.uint32) << 16)
            with open(fs, "rb") as f:
                f.seek(rs["data_off"])
                gs = np.frombuffer(f.read(rs["length"]), np.uint32)
            ds = int((want != gs).sum()) if want.size == gs.size else -1
            bad += (dw != 0) + (ds != 0)
            print(f"rank {rank}: lm_head.w8 {rw['length']} B, {dw} differing bytes vs the "
                  f"checkpoint; lm_head.ws {rs['length']} B, {ds} differing f32 values")
            if a.bf16_pack:
                fb, rb = record(a.bf16_pack, rank, "lm_head.w")
                rows = np.unique(np.concatenate([np.arange(32), rng.integers(0, NLOC, 512)]))
                w8 = got.reshape(NLOC, K)[rows]
                s32 = gs.view(np.float32)[rows]
                deq = bf16_rne((lut[w8] * s32[:, None]).astype(np.float32))
                ctl = bf16_rne((lut[w8] * s32[:, None]).astype(np.float16).astype(np.float32))
                rec = np.empty_like(deq)
                with open(fb, "rb") as f:
                    for i, r in enumerate(rows):
                        f.seek(rb["data_off"] + int(r) * K * 2)
                        rec[i] = np.frombuffer(f.read(K * 2), np.uint16)
                d, dc = int((deq != rec).sum()), int((ctl != rec).sum())
                bad += (d != 0) + (dc == 0)
                print(f"        {len(rows)} rows dequantized from THIS pack vs the bf16 pack's "
                      f"lm_head.w: {d} differing of {rec.size}; f16-first control {dc} "
                      f"({'fires' if dc else 'BLIND'})")
    print("OK" if bad == 0 else f"FAILED ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
