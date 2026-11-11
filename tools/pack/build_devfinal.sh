#!/usr/bin/env bash
# Build a Qwen3.8-27B device-final pack (SBRPACK1 v2): the records the engine binds, with every
# load-time transform already applied. Most users run this through tools/build_packs.sh, which
# runs it inside the dev container and then runs the self-checks.
#
#   tools/pack/build_devfinal.sh --checkpoint DIR [--source auto|nvfp4|quark] [--tp 1|2]
#       [--out DIR] [--head auto|bf16|fp8|mxfp4] [--only REGEX] [--workers N] [--build-unix N]
#       [--plan-only]   (write the plan JSON and stop)
#
#   1. `saber-packplan` (crates/saber-model) prints the record SET from the engine's own layer
#      graph: which merges exist, what layout each record is, and which checkpoint tensors feed it.
#      The builder cannot invent a record.
#   2. quark only: `tools/pack/quark27b_plan.py` re-sources that plan onto the OCP-MXFP4 tensors
#      (`weight` U8 + `weight_scale` e8m0) and the checkpoint's own bf16 `lm_head`.
#   3. `tools/pack/build_pack.py --layout device-final` does the numerics (NVFP4 / fp8 -> MXFP4 at
#      the `mse` block exponent; OCP MXFP4 -> the same blocks, value-exact) and writes
#      `rank<r>.pack` + `manifest.json`.
#
# CPU only, numpy only: no GPU. Runs `nice -n 19` on SABER_PACK_WORKERS (default 8) processes.
#
# --source auto (the default) reads the checkpoint's safetensors headers: `*.weight_packed`
#   tensors = NVFP4 (e.g. unsloth/Qwen3.8-27B-NVFP4); U8 `weight` + U8 `weight_scale` pairs = OCP
#   MXFP4 (amd/Qwen3.8-27B-Quark-AWQ-MXFP4). Anything else is refused.
# --head auto (the default) serves the head at the CHECKPOINT's own precision, never requantized:
#   an fp8 `lm_head.weight` + per-channel `weight_scale` -> `fp8` (the raw e4m3 rows + the scale as
#   f32); a bf16 one -> `bf16`. `--head bf16|fp8|mxfp4` are explicit overrides (NVFP4 only).
# --out default: $SABER_PACKS/<the name the launchers use>, SABER_PACKS default /mnt/nvme/saber-packs:
#   nvfp4  qwen38-27b-nvfp4-devfinal[-tp1][-fp8head|-bf16head]
#   quark  qwen38-27b-quark-awq-mxfp4[-tp1]
# --build-unix N stamps N as the build time (default: now); pass a reference manifest's
#   `built_unix` to reproduce its rank files byte for byte.
#
# Env: SABER_PACKS, SABER_PACK_WORKERS, SABER_PACKPLAN (a prebuilt saber-packplan binary; default
# `cargo run -p saber-model --bin saber-packplan`), SABER_PACK_PLANS (where the generated plan
# JSONs go; default <repo>/.cache/pack -- they are build inputs, never published).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }
MODEL=""
SOURCE=auto
OUT=""
TP=2
ONLY=""
HEAD=auto
WORKERS=${SABER_PACK_WORKERS:-8}
BUILD_UNIX=""
PLAN_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) MODEL=${2:?--checkpoint DIR}; shift 2 ;;
    --source) SOURCE=${2:?}; shift 2 ;;
    --tp) TP=${2:?}; shift 2 ;;
    --only) ONLY=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --head) HEAD=${2:?}; shift 2 ;;
    --workers) WORKERS=${2:?}; shift 2 ;;
    --build-unix) BUILD_UNIX=${2:?}; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build_devfinal.sh: unknown argument '$1' (--help)" >&2; exit 2 ;;
  esac
done
die() { echo "build_devfinal.sh: $*" >&2; exit 1; }
[ -n "$MODEL" ] || die "--checkpoint DIR is required (the Hugging Face checkpoint directory)"
[ -f "$MODEL/config.json" ] || die "$MODEL has no config.json: not a checkpoint directory"
ls "$MODEL"/*.safetensors >/dev/null 2>&1 || die "$MODEL has no .safetensors files"
case "$TP" in 1|2) ;; *) die "--tp $TP: 1 or 2" ;; esac
case "$HEAD" in auto|bf16|mxfp4|fp8) ;; *) die "--head $HEAD: auto, bf16, fp8 or mxfp4" ;; esac
command -v python3 >/dev/null || die "python3 not found"
python3 -c 'import numpy' 2>/dev/null || die "python3 has no numpy (pip install numpy)"

if [ "$SOURCE" = auto ]; then
  SOURCE=$(python3 - "$MODEL" <<'EOF'
import json, os, struct, sys
m = sys.argv[1]
names = {}
for f in sorted(os.listdir(m)):
    if f.endswith(".safetensors"):
        with open(os.path.join(m, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            names.update({k: v["dtype"] for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"})
if any(k.endswith(".weight_packed") for k in names):
    print("nvfp4")
elif any(d == "U8" and k.endswith(".weight") and names.get(k[:-6] + "weight_scale") == "U8"
         for k, d in names.items()):
    print("quark")
else:
    print("unknown")
EOF
)
fi
case "$SOURCE" in
  nvfp4) ;;
  quark) [ "$HEAD" = auto ] || [ "$HEAD" = bf16 ] || die "--head $HEAD: a Quark pack keeps the checkpoint's own bf16 head" ;;
  *) die "$MODEL is neither an NVFP4 (weight_packed) nor an OCP-MXFP4 (U8 weight + weight_scale) Qwen3.8-27B checkpoint" ;;
esac

PLANS=${SABER_PACK_PLANS:-$REPO/.cache/pack}
mkdir -p "$PLANS"
PLAN0=$PLANS/plan-$SOURCE-tp$TP-$HEAD.json
echo "== 1/3 record plan from saber-model's graph (source $SOURCE, tp $TP, --head $HEAD)"
if [ -n "${SABER_PACKPLAN:-}" ]; then
  "$SABER_PACKPLAN" --tp "$TP" --head "$HEAD" --checkpoint "$MODEL" > "$PLAN0"
else
  command -v cargo >/dev/null || die "cargo not found (run through tools/build_packs.sh, or set SABER_PACKPLAN)"
  nice -n 19 cargo run -q --release --manifest-path "$REPO/Cargo.toml" -p saber-model \
    --bin saber-packplan -- --tp "$TP" --head "$HEAD" --checkpoint "$MODEL" > "$PLAN0"
fi
# The head the plan RESOLVED (auto -> the checkpoint's) names the plan and the directory.
HEAD=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['head'])" "$PLAN0")
PLAN=$PLANS/plan-$SOURCE-tp$TP-$HEAD.json
[ "$PLAN0" = "$PLAN" ] || mv -f "$PLAN0" "$PLAN"
if [ "$SOURCE" = quark ]; then
  echo "== 2/3 re-source the plan onto the OCP-MXFP4 tensors (quark27b_plan.py)"
  python3 "$REPO/tools/pack/quark27b_plan.py" --model "$MODEL" --plan "$PLAN" > "$PLANS/plan-quark-tp$TP.json"
  PLAN=$PLANS/plan-quark-tp$TP.json
else
  echo "== 2/3 (NVFP4: the plan's sources are the checkpoint's as named)"
fi
if [ "$PLAN_ONLY" = 1 ]; then echo "plan: $PLAN"; exit 0; fi
if [ -z "$OUT" ]; then
  T=""; [ "$TP" = 1 ] && T=-tp1
  if [ "$SOURCE" = quark ]; then
    OUT=${SABER_PACKS:-/mnt/nvme/saber-packs}/qwen38-27b-quark-awq-mxfp4$T
  else
    H=""; [ "$HEAD" = bf16 ] && H=-bf16head; [ "$HEAD" = fp8 ] && H=-fp8head
    OUT=${SABER_PACKS:-/mnt/nvme/saber-packs}/qwen38-27b-nvfp4-devfinal$T$H
  fi
fi
[ ! -e "$OUT/manifest.json" ] || die "$OUT already holds a pack; remove it or pass another --out"
echo "   head $HEAD, $(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['records']))" "$PLAN") records -> $OUT"
echo "   plan: $PLAN"

echo "== 3/3 build (CPU only, numpy, $WORKERS workers)"
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 19 python3 "$REPO/tools/pack/build_pack.py" --model "$MODEL" --out "$OUT" --tp "$TP" \
  --layout device-final --plan-json "$PLAN" --workers "$WORKERS" \
  ${ONLY:+--only "$ONLY"} ${BUILD_UNIX:+--build-unix "$BUILD_UNIX"}
echo "built $OUT"
