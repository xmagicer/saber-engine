#!/usr/bin/env python3
"""Re-source a ``saber-packplan`` 27B plan onto an AMD Quark OCP-MXFP4 checkpoint.

``saber-packplan`` names the NVFP4 checkpoint's tensors (``weight_packed`` + e4m3
``weight_scale`` + globals for the MLPs, per-channel fp8 ``weight`` + ``weight_scale`` for the
attention / GDN linears, bf16 ``in_proj_a/b``, an fp8 ``lm_head``, bf16 ``k_scale``/``v_scale``).
``amd/Qwen3.8-27B-Quark-AWQ-MXFP4`` stores EVERY decoder linear as OCP MXFP4 (``weight`` U8
``[N, K/2]`` + ``weight_scale`` U8 e8m0 ``[N, K/32]``; AWQ already folded into the stored weights
and the preceding norms -- no extra tensors), a bf16 ``lm_head`` (excluded from quantization), a
bf16 MTP layer, and no KV scales. The record SET, names, layouts, shapes and shard kinds are the
plan's, untouched; only the SOURCES change:

* every decoder-linear source -> ``quant: mxfp4_ocp`` on ``<prefix>.weight`` + ``<prefix>.weight_scale``
  (``build_pack.py`` dequantizes it and proves the MXFP4 re-encode value-exact);
* ``lm_head.w`` (layout 14) -> ``quant: bf16``: the checkpoint's own bf16 rows;
* ``self_attn.{k,v}_scale`` -> ``quant: unit_bf16``: 1.0, the value the engine binds anyway.

Every rewritten source is checked against the checkpoint header (dtype, shape), and every U8
tensor of the checkpoint must be read by exactly one source.

    saber-packplan --tp 2 --checkpoint MODEL > plan.json
    tools/pack/quark27b_plan.py --model MODEL --plan plan.json > plan-quark.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

LAYERS = re.compile(r"^model\.language_model\.layers\.\d+\.")


def header(model: str) -> dict:
    out = {}
    for f in sorted(os.listdir(model)):
        if not f.endswith(".safetensors"):
            continue
        with open(os.path.join(model, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], v["shape"])
    return out


def strip(name: str) -> str:
    for suf in (".weight_packed", ".weight"):
        if name.endswith(suf):
            return name[: -len(suf)]
    raise SystemExit(f"{name}: not a weight tensor name")


def rewrite(plan: dict, hdr: dict) -> dict:
    consumed: dict[str, str] = {}

    def take(t: str, rec: str) -> None:
        if t in consumed and consumed[t] != rec:
            # The trio's three records (w, ws, wref) share sources; another record may not.
            if consumed[t].rsplit(".", 1)[0] != rec.rsplit(".", 1)[0]:
                raise SystemExit(f"{t} read by {consumed[t]} and {rec}")
        consumed[t] = rec

    for r in plan["records"]:
        for s in r["sources"]:
            w = s["weight"]
            if r["name"] == "lm_head.w":
                if hdr.get("lm_head.weight", ("",))[0] != "BF16":
                    raise SystemExit(f"lm_head.weight is {hdr.get('lm_head.weight')}, want BF16")
                s.update(weight="lm_head.weight", quant="bf16", scale=None, global_scale=None,
                         input_scale=None)
                continue
            if re.search(r"self_attn\.[kv]_scale$", w):
                if w in hdr:
                    raise SystemExit(f"{w} exists in the checkpoint: use it, not unit_bf16")
                s.update(quant="unit_bf16", scale=None, global_scale=None, input_scale=None)
                continue
            if not LAYERS.match(w) or s["quant"] not in ("nvfp4", "fp8_channel", "bf16"):
                if w not in hdr:
                    raise SystemExit(f"{r['name']}: {w} is not in the checkpoint")
                continue
            prefix = strip(w)
            wt, st = f"{prefix}.weight", f"{prefix}.weight_scale"
            if wt not in hdr or st not in hdr:
                raise SystemExit(f"{r['name']}: {wt} / {st} missing")
            (wd, wsh), (sd, ssh) = hdr[wt], hdr[st]
            if wd != "U8" or sd != "U8" or len(wsh) != 2 or ssh != [wsh[0], wsh[1] * 2 // 32]:
                raise SystemExit(f"{wt}: {wd}{wsh} + {sd}{ssh} is not OCP MXFP4 (U8 [N,K/2] + "
                                 f"U8 [N,K/32])")
            s.update(weight=wt, quant="mxfp4_ocp", scale=st, global_scale=None, input_scale=None)
            take(wt, r["name"])
            take(st, r["name"])
    u8 = sorted(k for k, (d, _) in hdr.items() if d == "U8")
    unread = [k for k in u8 if k not in consumed]
    if unread:
        raise SystemExit(f"{len(unread)} MXFP4 tensors read by no record: {unread[:4]}")
    plan["sources_rewritten_by"] = "tools/pack/quark27b_plan.py (Q27: OCP MXFP4 checkpoint)"
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True, help="saber-packplan's output")
    a = ap.parse_args(argv)
    with open(a.plan) as f:
        plan = json.load(f)
    if plan.get("producer") != "saber-packplan":
        raise SystemExit("not a saber-packplan plan")
    out = rewrite(plan, header(a.model))
    n = sum(1 for r in out["records"] for s in r["sources"] if s["quant"] == "mxfp4_ocp")
    print(f"quark27b_plan: {n} mxfp4_ocp sources, head {out['head']}", file=sys.stderr)
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
