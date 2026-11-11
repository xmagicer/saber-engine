#!/usr/bin/env python3
"""FN110: the MTP drafter's experts in the expert tier's NATIVE record form (MXFP4, the trunk's
grouped W4A8 GEMM) as an ADDITIONAL pack file per rank -- `rank<r>/mtp_x4.{pack,json}` (the pack is
never rewritten in place; FN50's `mtp_fp8.*` beside it is untouched).

Host tool, numpy only, no GPU, no lock:

    python3 tools/pack/build_mtp_x4.py --model CKPT --pack $SABER_PACKS/qwen38-flashnext-quark-ep2

The quantisation (the host gate's `mx4s` form, `tools/capture/fn110/mtp_accept.py`): the checkpoint's
bf16 MTP experts (routed + the shared expert) RTN to MXFP4 -- per 32 along K the e8m0 exponent
floor(log2 amax) - 2 (OCP), e2m1 round-half-to-even with saturation at 6 -- then Quark's byte form
(evens in the LOW nibble, e8m0 codes) and FN2's record law exactly as `build_flashnext_pack.py`'s
`quark_expert` (`permute_w` fragment order, scales transposed to [K/32, N], `wref` = their row max).
Verified per expert: Quark's dequant of the produced bytes == the RTN values, Saber's dequant of the
record == Quark's dequant, `wref` re-derived; negative control: the nibble-swapped bytes must differ.

Records per routed expert `e` (rank r holds e in [256 r, 256 r + 256)):
    mtp.layer.0.expert.<e>.x4.{gate_up,down}.{w,ws,wref}
and on rank 0 the shared expert (entry 256 of rank 0's grouped table, the trunk's convention):
    mtp.layer.0.shared_expert.x4.{gate_up,down}.{w,ws,wref}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashnext_mx as mx  # noqa: E402

STEM = "mtp_x4"
H, I, ELOC = 2560, 640, 256
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


class St:
    def __init__(self, model):
        self.model = model
        self.map = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.hdr:
            fd = os.open(os.path.join(self.model, f), os.O_RDONLY)
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.hdr[f] = (fd, json.loads(os.pread(fd, n, 8)), 8 + n)
        fd, h, base = self.hdr[f]
        m = h[name]
        assert m["dtype"] == "BF16", (name, m["dtype"])
        a, b = m["data_offsets"]
        raw = np.frombuffer(os.pread(fd, b - a, base + a), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])


def mxfp4_rtn_bytes(w):
    """[N, K] fp32 -> (Quark bytes w [N, K/2] u8, e8m0 s [N, K/32] u8, the dequantised values)."""
    n, k = w.shape
    g = w.reshape(n, k // 32, 32)
    amax = np.abs(g).max(-1, keepdims=True)
    e = np.floor(np.log2(np.maximum(amax, np.float32(2.0 ** -126)))) - 2.0
    e = e.astype(np.float32)
    x = g / np.power(np.float32(2.0), e)
    a = np.minimum(np.abs(x), np.float32(6.0))
    step = np.where(a < 2.0, np.float32(0.5), np.where(a < 4.0, np.float32(1.0), np.float32(2.0))).astype(np.float32)
    r = np.minimum(np.round(a / step) * step, np.float32(6.0)).astype(np.float32)  # np.round: half to even
    idx = np.searchsorted(E2M1, r)
    assert np.array_equal(E2M1[idx], r)
    code = idx.astype(np.uint8) | np.where((x < 0) & (r > 0), np.uint8(8), np.uint8(0))
    code = code.reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    ec = e.reshape(n, k // 32) + 127.0
    assert ec.min() >= 0 and ec.max() <= 254, (ec.min(), ec.max())
    deq = (np.sign(x) * r * np.power(np.float32(2.0), e)).reshape(n, k).astype(np.float32)
    return packed, ec.astype(np.uint8), deq


def record_parts(gw, uw, dw):
    """Quark-form bytes of gate/up/down -> FN2's six sub-records (`quark_expert`'s law), verified."""
    g, gs, gd = mxfp4_rtn_bytes(gw)
    u, us, ud = mxfp4_rtn_bytes(uw)
    d, ds, dd = mxfp4_rtn_bytes(dw)
    for b, s, ref in ((g, gs, gd), (u, us, ud), (d, ds, dd)):
        if not np.array_equal(mx.quark_dequant(b, s), ref):
            raise RuntimeError("Quark's dequant of the produced bytes != the RTN values")
        if np.array_equal(mx.quark_dequant(mx.nibble_swap_control(b), s), ref):
            raise RuntimeError("negative control escaped: the nibble-swapped bytes dequantise the same")
    gu, gus = np.concatenate([g, u]), np.concatenate([gs, us])
    parts = {
        "gate_up.w": mx.permute_w(gu, 2 * I, H),
        "gate_up.ws": np.ascontiguousarray(gus.T),
        "gate_up.wref": gus.max(axis=1),
        "down.w": mx.permute_w(d, H, I),
        "down.ws": np.ascontiguousarray(ds.T),
        "down.wref": ds.max(axis=1),
    }
    sab_gu = mx.saber_dequant(parts["gate_up.w"], parts["gate_up.ws"], 2 * I, H)
    sab_d = mx.saber_dequant(parts["down.w"], parts["down.ws"], H, I)
    if not (mx.bits_equal(sab_gu[:I], mx.quark_dequant(g, gs)) and mx.bits_equal(sab_gu[I:], mx.quark_dequant(u, us))
            and mx.bits_equal(sab_d, mx.quark_dequant(d, ds))):
        raise RuntimeError("Saber's dequant of the record != Quark's dequant")
    return {k: np.ascontiguousarray(v).reshape(-1).tobytes() for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the amd/Qwen3.8-Flash-Next-Quark-MXFP4 checkpoint directory")
    ap.add_argument("--pack", required=True)
    a = ap.parse_args()
    st = St(a.model)
    for r in (0, 1):
        d = os.path.join(a.pack, f"rank{r}")
        pf, jf = os.path.join(d, STEM + ".pack"), os.path.join(d, STEM + ".json")
        if os.path.exists(pf) or os.path.exists(jf):
            raise SystemExit(f"{pf}: exists (never rewritten in place)")
        tmp = pf + ".tmp"
        recs, off = [], 0
        with open(tmp, "wb") as f:
            items = [(f"mtp.layer.0.expert.{r * ELOC + i}.x4", f"mtp.layers.0.mlp.experts.{r * ELOC + i}.")
                     for i in range(ELOC)]
            if r == 0:
                items.append(("mtp.layer.0.shared_expert.x4", "mtp.layers.0.mlp.shared_expert."))
            for j, (name, src) in enumerate(items):
                ws = [st.get(src + p + "_proj.weight") for p in ("gate", "up", "down")]
                for k, b in record_parts(*ws).items():
                    recs.append({"name": f"{name}.{k}", "class": "mtp.experts", "storage": {
                        "pack": STEM + ".pack", "data_off": off, "length": len(b),
                        "sha256": hashlib.sha256(b).hexdigest()}})
                    pad = (256 - len(b) % 256) % 256
                    f.write(b + bytes(pad))
                    off += len(b) + pad
                if j % 32 == 0:
                    print(f"rank{r}: {j}/{len(items)} records verified", flush=True)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, pf)
        json.dump({"records": recs, "source": a.model, "class": "mtp.experts", "bytes": off,
                   "layout": "mxfp4 RTN of the bf16 MTP experts in FN2's record form (FN110)"},
                  open(jf, "w"), indent=1)
        print(f"rank{r}: {len(recs)} records, {off / 2 ** 30:.3f} GiB -> {pf}")


if __name__ == "__main__":
    main()
