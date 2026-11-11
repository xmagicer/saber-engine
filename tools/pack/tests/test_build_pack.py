#!/usr/bin/env python3
"""Unit tests for ``tools/pack/build_pack.py`` (numpy only, no checkpoint, no GPU).

    python3 -m unittest discover -s tools/pack/tests -v

The end-to-end case writes a tiny synthetic checkpoint (NVFP4, per-channel fp8 and bf16 tensors,
a segmented ``in_proj_qkv``, a row-parallel ``down_proj``) plus a plan in ``saber-packplan``'s
shape, builds a TP=2 pack from it and checks the bytes against references computed here
independently: the e2m1 table read back through ``flashnext_mx.saber_dequant`` (Saber's pack
reader in numpy), a brute-force f64 exponent search, and the shard rules of the SBRPACK1 format.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_pack as bp  # noqa: E402
import flashnext_mx as mx  # noqa: E402

F32 = np.float32


def lane8_reference(v) -> np.float32:
    """The stated order, one scalar add at a time: lanes i = k mod 8, then the lanes in order."""
    lanes = [F32(v[i]) for i in range(8)]
    for blk in range(1, 4):
        for i in range(8):
            lanes[i] = F32(lanes[i] + F32(v[8 * blk + i]))
    s = lanes[0]
    for i in range(1, 8):
        s = F32(s + lanes[i])
    return s


class Numerics(unittest.TestCase):
    def test_e2m1_rounds_to_nearest_ties_to_even_code_and_saturates(self):
        y = np.array([0, .24, .25, .26, .74, .75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 9, 1e30], F32)
        want = [0, 0, 0, 1, 1, 2, 2, 4, 4, 6, 6, 7, 7, 7]
        self.assertEqual(bp.e2m1_codes(y).tolist(), want)

    def test_lane_sum_is_the_stated_order_and_not_the_sequential_one(self):
        rng = np.random.default_rng(7)
        d = (rng.standard_normal((4000, 32)) * np.exp2(rng.integers(-20, 20, (4000, 32))))
        d = (d * d).astype(F32)
        got = bp._lane8_sum(d)
        for r in range(0, 4000, 97):
            self.assertEqual(got[r].tobytes(), lane8_reference(d[r]).tobytes())
        seq = d[:, 0].copy()
        for j in range(1, 32):
            seq = seq + d[:, j]
        self.assertTrue((seq != got).any(), "control: the sequential order must differ somewhere")

    def test_mse_tie_goes_to_the_exponent_that_does_not_clip(self):
        # all 7.0: e0 = 0 clips 7 -> 6, e0 + 1 rounds 3.5 -> 4 (x 2 = 8); both errors 32 exactly.
        codes, ws = bp.quantize_mxfp4(np.full((1, 32), 7.0, F32))
        self.assertEqual(int(ws[0, 0]), 128)
        self.assertEqual(int(codes[0, 0]), 6)
        # all 5.0: e0 = 0 rounds to 4, e0 + 1 rounds 2.5 -> 2 (x 2 = 4); a tie, nothing clips.
        codes, ws = bp.quantize_mxfp4(np.full((1, 32), 5.0, F32))
        self.assertEqual(int(ws[0, 0]), 127)
        self.assertEqual(int(codes[0, 0]), 6)

    def test_exponent_is_the_f64_argmin_away_from_near_ties(self):
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((256, 1024)) * 0.02).astype(F32)
        codes, ws = bp.quantize_mxfp4(x)
        xb = x.reshape(256, 32, 32).astype(np.float64)
        amax = np.abs(xb).max(-1)
        e0 = np.floor(np.log2(amax)).astype(int) - 2
        errs = []
        for e in (e0, e0 + 1):
            s = np.exp2(e)[..., None]
            c = bp.e2m1_codes((np.abs(xb) / s).astype(F32))
            errs.append(((bp.E2M1[c] * s * np.sign(xb) - xb) ** 2).sum(-1))
        best = np.where(errs[1] < errs[0], e0 + 1, e0)
        gap = np.abs(errs[1] - errs[0]) / np.maximum(errs[0], errs[1])
        clear = gap > 1e-5
        self.assertTrue(clear.mean() > 0.99)
        np.testing.assert_array_equal((ws.astype(int) - 127)[clear], best[clear])
        # and the codes decode (Saber's reader) to the chosen grid point of every element
        n, k = x.shape
        perm = bp.permute_w(bp.pack_nibbles(codes), n, k)
        deq = mx.saber_dequant(perm, np.ascontiguousarray(ws.T), n, k)
        scale = np.repeat(np.exp2(ws.astype(np.float64) - 127), 32, axis=1)
        self.assertTrue(np.all(np.abs(deq - x) <= 2 * scale + 1e-30))

    def test_negative_values_carry_the_sign_but_zero_codes_do_not(self):
        x = np.zeros((1, 32), F32)
        x[0, :4] = [-1.0, -1e-9, 1.0, -0.0]
        codes, _ = bp.quantize_mxfp4(x)
        self.assertEqual(codes[0, :4].tolist(), [8 | 6, 0, 6, 0])  # amax 1: e0 = -2, 1.0 -> 4

    def test_permute_is_the_fragment_order_and_inverts(self):
        n, k = 32, 64
        p = np.arange(n * k // 2, dtype=np.uint32).astype(np.uint8).reshape(n, k // 2)
        q = bp.permute_w(p, n, k)
        np.testing.assert_array_equal(bp.unpermute_w(q, n, k), p)
        np.testing.assert_array_equal(q, mx.permute_w(p, n, k))
        # the first 64 output bytes: rows 0..15, the first 4 bytes of each
        np.testing.assert_array_equal(q.reshape(-1)[:64].reshape(16, 4), p[:16, :4])

    def test_bf16_rounding_is_nearest_even(self):
        x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5], F32)
        self.assertEqual(bp.f32_to_bf16(x).tolist(), [0x3F80, 0x3F80, 0x3F82, 0xC020])

    def test_ocp_mxfp4_reencodes_value_exact_and_a_nudge_does_not(self):
        # any OCP MXFP4 block (any code set, any scale) dequantizes to values one of the
        # `mse` candidates carries exactly -- the builder's assertion for `mxfp4_ocp` sources.
        rng = np.random.default_rng(27)
        codes = rng.integers(0, 16, size=(64, 256), dtype=np.uint8)
        codes[0, :32] = 0  # an all-zero block
        codes[1, :32] = rng.choice([0, 1, 8, 9], 32)  # amax 0.5: four steps below the top code
        scales = rng.integers(100, 140, size=(64, 8), dtype=np.uint8)
        x = (bp.FP4_VALUES[codes].reshape(64, 8, 32)
             * np.ldexp(F32(1), scales.astype(np.int32) - 127)[..., None]).reshape(64, 256)
        c, e = bp.quantize_mxfp4(x.astype(F32))
        back = (bp.FP4_VALUES[c].reshape(64, 8, 32)
                * np.ldexp(F32(1), e.astype(np.int32) - 127)[..., None]).reshape(64, 256)
        self.assertTrue(np.array_equal(back, x))
        # the negative control: a value off the grid does not survive
        y = x.copy()
        y[2, 3] = np.nextafter(y[2, 3] if y[2, 3] else F32(1e-3), F32(np.inf))
        c, e = bp.quantize_mxfp4(y.astype(F32))
        back = (bp.FP4_VALUES[c].reshape(64, 8, 32)
                * np.ldexp(F32(1), e.astype(np.int32) - 127)[..., None]).reshape(64, 256)
        self.assertFalse(np.array_equal(back, y))

    def test_e4m3_table_is_ocp(self):
        self.assertEqual(float(bp.E4M3[0x7E]), 448.0)
        self.assertEqual(float(bp.E4M3[0x01]), 2.0 ** -9)
        self.assertTrue(np.isnan(bp.E4M3[0x7F]) and np.isnan(bp.E4M3[0xFF]))
        self.assertEqual(float(bp.E4M3[0xB8]), -1.0)


class Format(unittest.TestCase):
    def test_writer_layout_and_digests(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rank0.pack")
            big = os.urandom(3 * 4096 + 5)
            recs = [bp.Record("a", "U8", [3], bp.REPLICATED, 0),
                    bp.Record("big", "BF16", [len(big) // 2 + 1], bp.DIM0, 0)]
            big += b"\0" * (recs[1].length - len(big))
            rp = bp.plan_rank(recs, 4096)
            hp = bp.HashPool(2, 1 << 30, 4096)
            w = bp.RankWriter(path, rp, hp)
            for r, data in zip(recs, [b"xyz", big]):
                w.begin(r)
                w.write(data[: len(data) // 2])
                w.write(data[len(data) // 2:])
                w.end()
            hp.close()
            w.finish(2, 0, b"P" * 32, b"T" * 32, 123)
            with open(path, "rb") as f:
                raw = f.read()
        self.assertEqual(len(raw) % 4096, 0)
        self.assertEqual(raw[:8], bp.MAGIC)
        ver, hb, rb, cnt = struct.unpack_from("<IIII", raw, 8)
        self.assertEqual((ver, hb, rb, cnt), (2, 256, 160, 2))
        to, nbo, nbb, do, fb = struct.unpack_from("<QQQQQ", raw, 24)
        self.assertEqual((to, fb), (256, len(raw)))
        self.assertEqual(raw[80:112], b"P" * 32)
        self.assertEqual(raw[112:144], b"T" * 32)
        self.assertEqual(hashlib.sha256(raw[to:nbo + nbb]).digest(), raw[144:176])
        cbo, cbb = struct.unpack_from("<QQ", raw, 176)
        self.assertEqual(cbo, nbo + nbb)
        self.assertEqual(hashlib.sha256(raw[cbo:cbo + cbb]).digest(), raw[192:224])
        self.assertEqual(struct.unpack_from("<I", raw, 224)[0], 4096)
        self.assertEqual(raw[228:256], bytes(28))
        names = []
        for i, data in enumerate([b"xyz", big]):
            e = raw[256 + 160 * i: 256 + 160 * (i + 1)]
            noff, nlen = struct.unpack_from("<II", e, 0)
            self.assertEqual(struct.unpack_from("<H", raw, nbo + noff - 2)[0], nlen)
            names.append(raw[nbo + noff: nbo + noff + nlen].decode())
            off, ln = struct.unpack_from("<QQ", e, 24)
            self.assertEqual(off % 4096, 0)
            self.assertEqual(raw[off:off + ln], data)
            self.assertEqual(e[104:136], hashlib.sha256(data).digest())
            self.assertEqual(struct.unpack_from("<I", e, 20)[0], 1 if ln > 4096 else 0)
        self.assertEqual(names, ["a", "big"])
        n_chunks, flags = struct.unpack_from("<II", raw, cbo)
        self.assertEqual((n_chunks, flags), (-(-len(big) // 4096), 0))
        self.assertEqual(raw[cbo + 8:cbo + 40], hashlib.sha256(big).digest())
        for c in range(n_chunks):
            self.assertEqual(raw[cbo + 40 + 32 * c:cbo + 72 + 32 * c],
                             hashlib.sha256(big[4096 * c:4096 * (c + 1)]).digest())

    def test_every_policy_rule_is_reachable_and_unknown_names_refuse(self):
        for p, k, _ in bp.SHARD_POLICY:
            self.assertIn(k, bp.SHARD_OF_NAME)
        self.assertEqual(bp.policy_kind("model.language_model.layers.7.mlp.down_proj.weight_packed"),
                         "Dim1")
        with self.assertRaises(SystemExit):
            bp.policy_kind("model.language_model.layers.7.mystery.weight")


# ------------------------------------------------------------------------------------------
# end to end on a synthetic checkpoint
# ------------------------------------------------------------------------------------------

H, KH, KD, VH, VD, INTER = 64, 2, 16, 16, 16, 128  # qkv = 32 + 32 + 256 rows, z = 256
L = "model.language_model.layers.0."


def write_safetensors(path, tensors):
    hdr, blobs, off = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        b = np.ascontiguousarray(data).tobytes()
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    j = json.dumps(hdr, sort_keys=True).encode()
    j += b" " * (-len(j) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(j)) + j + b"".join(blobs))


def rand_bf16(rng, shape, scale=0.05):
    x = (rng.standard_normal(shape) * scale).astype(F32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def synthetic(d):
    rng = np.random.default_rng(11)
    t = {}
    # NVFP4 MLP: gate/up [INTER, H], down [H, INTER]
    for proj, (n, k) in {"gate_proj": (INTER, H), "up_proj": (INTER, H),
                         "down_proj": (H, INTER)}.items():
        t[L + f"mlp.{proj}.weight_packed"] = ("U8", (n, k // 2),
                                               rng.integers(0, 256, (n, k // 2), dtype=np.uint8))
        t[L + f"mlp.{proj}.weight_scale"] = ("F8_E4M3", (n, k // 16),
                                              rng.integers(0x20, 0x50, (n, k // 16), dtype=np.uint8))
        t[L + f"mlp.{proj}.weight_global_scale"] = ("F32", (1,), np.array([37.0], F32))
        t[L + f"mlp.{proj}.input_global_scale"] = ("F32", (1,), np.array([1.0], F32))
    # per-channel fp8 GDN projections, bf16 b / a
    q = 2 * KH * KD + VH * VD
    for nm, n in {"in_proj_qkv": q, "in_proj_z": VH * VD}.items():
        w = rng.integers(0, 256, (n, H), dtype=np.uint8)
        w[(w & 0x7F) == 0x7F] = 0x10
        t[L + f"linear_attn.{nm}.weight"] = ("F8_E4M3", (n, H), w)
        t[L + f"linear_attn.{nm}.weight_scale"] = ("BF16", (n, 1), rand_bf16(rng, (n, 1), 0.01))
    for nm in ("in_proj_b", "in_proj_a"):
        t[L + f"linear_attn.{nm}.weight"] = ("BF16", (VH, H), rand_bf16(rng, (VH, H)))
    t[L + "linear_attn.conv1d.weight"] = ("BF16", (q, 1, 4), rand_bf16(rng, (q, 1, 4)))
    t[L + "linear_attn.A_log"] = ("BF16", (VH,), rand_bf16(rng, (VH,)))
    t[L + "input_layernorm.weight"] = ("BF16", (H,), rand_bf16(rng, (H,)))
    t["model.visual.blocks.0.norm1.weight"] = ("BF16", (8,), rand_bf16(rng, (8,)))
    write_safetensors(os.path.join(d, "model.safetensors"), t)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": {"linear_num_key_heads": KH, "linear_key_head_dim": KD,
                                   "linear_num_value_heads": VH, "linear_value_head_dim": VD}}, f)
    return t


def src(w, quant, scale=None, gs=None):
    return {"weight": w, "quant": quant, "scale": scale, "global_scale": gs, "input_scale": None}


def nv(proj):
    p = L + f"mlp.{proj}."
    return dict(src(p + "weight_packed", "nvfp4", p + "weight_scale", p + "weight_global_scale"),
                input_scale=p + "input_global_scale")


def trio(op, layout, shape, kind, sources):
    n, k = shape
    return [
        {"name": f"{op}.w", "layout_id": layout, "op": op, "dtype": "FP4_E2M1_X2",
         "shard_kind": kind, "shape": [n, k], "transformed": True, "sources": sources},
        {"name": f"{op}.ws", "layout_id": 2, "op": op, "dtype": "E8M0", "shard_kind": kind,
         "shape": [k // 32, n], "transformed": True, "sources": sources},
        {"name": f"{op}.wref", "layout_id": 3, "op": op, "dtype": "E8M0", "shard_kind": kind,
         "shape": [n], "transformed": True, "sources": sources},
    ]


def synthetic_plan():
    la = L + "linear_attn."
    recs = []
    recs += trio("layer.0.gdn.in_proj_qkvz", 4, [(2 * KH * KD + VH * VD + VH * VD) // 2, H],
                 "Dim0Segments", [src(la + "in_proj_qkv.weight", "fp8_channel",
                                      la + "in_proj_qkv.weight_scale"),
                                  src(la + "in_proj_z.weight", "fp8_channel",
                                      la + "in_proj_z.weight_scale")])
    recs += trio("layer.0.gdn.in_proj_ba", 4, [VH, H], "Dim0Segments",
                 [src(la + "in_proj_b.weight", "bf16"), src(la + "in_proj_a.weight", "bf16")])
    recs += trio("layer.0.mlp.gate_up", 4, [INTER, H], "Dim0Segments",
                 [nv("gate_proj"), nv("up_proj")])
    recs += trio("layer.0.mlp.down_proj", 1, [H, INTER // 2], "Dim1", [nv("down_proj")])
    recs.append({"name": la + "conv1d.weight", "layout_id": 0, "op": "c", "dtype": "BF16",
                 "shard_kind": "Dim0Segments", "shape": [(2 * KH * KD + VH * VD) // 2, 1, 4],
                 "transformed": False, "sources": [src(la + "conv1d.weight", "as_stored")]})
    recs.append({"name": la + "A_log", "layout_id": 5, "op": "g", "dtype": "F32",
                 "shard_kind": "Dim0", "shape": [VH // 2], "transformed": True,
                 "sources": [src(la + "A_log", "as_stored")]})
    recs.append({"name": L + "input_layernorm.weight", "layout_id": 0, "op": "n",
                 "dtype": "BF16", "shard_kind": "Replicated", "shape": [H],
                 "transformed": False, "sources": [src(L + "input_layernorm.weight", "as_stored")]})
    return {"producer": "saber-packplan", "tp_size": 2, "head": "fp8", "records": recs}


def read_pack(path):
    with open(path, "rb") as f:
        raw = f.read()
    cnt = struct.unpack_from("<I", raw, 20)[0]
    nbo = struct.unpack_from("<Q", raw, 32)[0]
    out = {}
    for i in range(cnt):
        e = raw[256 + 160 * i: 256 + 160 * (i + 1)]
        noff, nlen = struct.unpack_from("<II", e, 0)
        off, ln = struct.unpack_from("<QQ", e, 24)
        out[raw[nbo + noff:nbo + noff + nlen].decode()] = raw[off:off + ln]
    return out


def ref_dequant(t, name):
    """The source's f32 values, written here from the checkpoint's own definitions."""
    dt, shape, data = t[name]
    if dt == "BF16":
        return (data.astype(np.uint32) << 16).view(F32).reshape(shape)
    if dt == "F8_E4M3":
        s = (t[name + "_scale"][2].astype(np.uint32) << 16).view(F32).reshape(-1, 1)
        return bp.E4M3[data] * s
    base = name[: -len("weight_packed")]
    codes = np.stack([data & 0xF, data >> 4], -1).reshape(data.shape[0], -1)
    sc = bp.E4M3[t[base + "weight_scale"][2]] / t[base + "weight_global_scale"][2][0]
    return bp.FP4_VALUES[codes] * np.repeat(sc.astype(F32), 16, axis=1)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        cls.model = os.path.join(d, "Synth")
        os.makedirs(cls.model)
        cls.t = synthetic(cls.model)
        plan = os.path.join(d, "plan.json")
        with open(plan, "w") as f:
            json.dump(synthetic_plan(), f)
        cls.out = os.path.join(d, "pack")
        rc = bp.main(["--model", cls.model, "--out", cls.out, "--tp", "2", "--plan-json", plan,
                      "--workers", "2", "--digest-chunk-mib", "0", "--build-unix", "0"])
        assert rc == 0
        cls.ranks = [read_pack(os.path.join(cls.out, f"rank{r}.pack")) for r in (0, 1)]
        with open(os.path.join(cls.out, "manifest.json")) as f:
            cls.man = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def decode(self, rank, op, n, k):
        r = self.ranks[rank]
        w = np.frombuffer(r[op + ".w"], np.uint8).reshape(n, k // 2)
        ws = np.frombuffer(r[op + ".ws"], np.uint8).reshape(k // 32, n)
        wref = np.frombuffer(r[op + ".wref"], np.uint8)
        np.testing.assert_array_equal(wref, ws.max(axis=0))
        return mx.saber_dequant(w, ws, n, k), ws

    def check_close(self, got, want, ws):
        scale = np.repeat(np.exp2(ws.T.astype(np.float64) - 127), 32, axis=1)
        self.assertTrue(np.all(np.abs(got - want) <= 2 * scale), "outside the MXFP4 grid step")
        self.assertTrue(np.abs(got - want).mean() < 0.25 * np.abs(want).mean())

    def test_segments_are_split_per_segment_then_concatenated(self):
        la = L + "linear_attn."
        qkv = ref_dequant(self.t, la + "in_proj_qkv.weight")
        z = ref_dequant(self.t, la + "in_proj_z.weight")
        qk = KH * KD
        segs = [qkv[:qk], qkv[qk:2 * qk], qkv[2 * qk:]]
        n = (2 * qk + 2 * VH * VD) // 2
        for rank in (0, 1):
            want = np.concatenate([s[rank * len(s) // 2:(rank + 1) * len(s) // 2]
                                   for s in segs + [z]])
            got, ws = self.decode(rank, "layer.0.gdn.in_proj_qkvz", n, H)
            self.check_close(got, want, ws)
            conv = self.t[la + "conv1d.weight"][2]
            cw = np.concatenate([conv[a + rank * (b - a) // 2:a + (rank + 1) * (b - a) // 2]
                                 for a, b in ((0, qk), (qk, 2 * qk), (2 * qk, len(conv)))])
            self.assertEqual(self.ranks[rank][la + "conv1d.weight"], cw.tobytes())

    def test_a_contiguous_half_split_is_not_what_was_built(self):
        # the negative control of the segment rule
        la = L + "linear_attn."
        qkv = ref_dequant(self.t, la + "in_proj_qkv.weight")
        z = ref_dequant(self.t, la + "in_proj_z.weight")
        n = len(qkv) // 2 + len(z) // 2
        got, ws = self.decode(0, "layer.0.gdn.in_proj_qkvz", n, H)
        wrong = np.concatenate([qkv[: len(qkv) // 2], z[: len(z) // 2]])
        scale = np.repeat(np.exp2(ws.T.astype(np.float64) - 127), 32, axis=1)
        self.assertFalse(np.all(np.abs(got - wrong) <= 2 * scale))

    def test_row_parallel_down_splits_columns_and_bytes_match_a_whole_tensor_quant(self):
        x = ref_dequant(self.t, L + "mlp.down_proj.weight_packed")
        codes, e = bp.quantize_mxfp4(np.ascontiguousarray(x))
        for rank in (0, 1):
            c0, c1 = rank * INTER // 2, (rank + 1) * INTER // 2
            got, ws = self.decode(rank, "layer.0.mlp.down_proj", H, INTER // 2)
            self.check_close(got, x[:, c0:c1], ws)
            # blocks are independent: the whole tensor's quantization, sliced, is the rank's
            want_w = bp.permute_w(bp.pack_nibbles(np.ascontiguousarray(codes[:, c0:c1])), H,
                                  INTER // 2)
            self.assertEqual(self.ranks[rank]["layer.0.mlp.down_proj.w"], want_w.tobytes())
            self.assertEqual(self.ranks[rank]["layer.0.mlp.down_proj.ws"],
                             np.ascontiguousarray(e[:, c0 // 32:c1 // 32].T).tobytes())

    def test_merges_and_casts(self):
        gate = ref_dequant(self.t, L + "mlp.gate_proj.weight_packed")
        up = ref_dequant(self.t, L + "mlp.up_proj.weight_packed")
        b = ref_dequant(self.t, L + "linear_attn.in_proj_b.weight")
        a = ref_dequant(self.t, L + "linear_attn.in_proj_a.weight")
        for rank in (0, 1):
            h = INTER // 2
            got, ws = self.decode(rank, "layer.0.mlp.gate_up", INTER, H)
            self.check_close(got, np.concatenate([gate[rank * h:(rank + 1) * h],
                                                  up[rank * h:(rank + 1) * h]]), ws)
            v = VH // 2
            got, ws = self.decode(rank, "layer.0.gdn.in_proj_ba", VH, H)
            self.check_close(got, np.concatenate([b[rank * v:(rank + 1) * v],
                                                  a[rank * v:(rank + 1) * v]]), ws)
            alog = ref_dequant(self.t, L + "linear_attn.A_log")[rank * v:(rank + 1) * v]
            self.assertEqual(self.ranks[rank][L + "linear_attn.A_log"], alog.tobytes())
            self.assertEqual(self.ranks[rank][L + "input_layernorm.weight"],
                             self.t[L + "input_layernorm.weight"][2].tobytes())

    def test_unplanned_tensors_ride_raw_and_transform_inputs_do_not(self):
        names = [r["name"] for r in self.man["ranks"][0]["records"]]
        self.assertEqual(names[-1], "model.visual.blocks.0.norm1.weight")
        self.assertFalse(any("global_scale" in n or n.endswith("weight_scale") for n in names))

    def test_manifest_hashes_are_the_descriptors(self):
        def canon(o):
            return hashlib.sha256(json.dumps(o, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        tier = self.man["tier"]
        self.assertEqual(canon(tier["descriptor"]), tier["id"])
        desc = {k: v for k, v in self.man["plan"]["descriptor"].items() if k != "supplied"}
        self.assertEqual(canon(desc), self.man["plan"]["hash"])
        for rank in (0, 1):
            with open(os.path.join(self.out, f"rank{rank}.pack"), "rb") as f:
                raw = f.read(256)
            self.assertEqual(raw[80:112].hex(), self.man["plan"]["hash"])
            self.assertEqual(raw[112:144].hex(), tier["id"])
            self.assertEqual(struct.unpack_from("<II", raw, 64), (2, rank))


if __name__ == "__main__":
    unittest.main()
