#!/usr/bin/env bash
# Build a Saber weight pack from a Hugging Face checkpoint, then self-check it. CPU only, no GPU.
#
#   tools/build_packs.sh 27b-quark  --checkpoint DIR [--tp 1|2] [--out DIR]   # amd/Qwen3.8-27B-Quark-AWQ-MXFP4
#   tools/build_packs.sh 27b-nvfp4  --checkpoint DIR [--tp 1|2] [--out DIR]   # unsloth/Qwen3.8-27B-NVFP4
#   tools/build_packs.sh 27b-quark-drafter --drafter-pack DIR --target-pack DIR [--out DIR]
#   tools/build_packs.sh flashnext  --checkpoint DIR [--out DIR] [--dry-run|--layers A:B]  # amd/Qwen3.8-Flash-Next-Quark-MXFP4:
#                                   the EP=2 BASE pack only (experts, dense, PLE index, aliases, mtp_x4);
#                                   see the README for the remaining `saber flashnext-pack` steps
#   tools/build_packs.sh check 27b-quark|27b-nvfp4 --pack DIR --checkpoint DIR   # the self-check alone
#   tools/build_packs.sh test                                                 # the builder's unit tests
#
# Runs inside the same dev container as the other launchers (localhost/saber-dev:latest, built by
# tools/build-image.sh; podman or docker, SABER_CONTAINER picks), builds `saber-packplan` from this
# repo's crates on first use, runs the builder, then the self-check. `--host` runs on this machine
# instead (needs python3 + numpy, and cargo).
#
# Options:
#   --checkpoint DIR   the checkpoint directory (`hf download <repo> --local-dir DIR`)
#   --tp 1|2           tensor-parallel degree of the pack: 2 = two cards (default), 1 = one card
#   --out DIR          the pack directory (default $SABER_PACKS/<the name the launchers use>)
#   --reference FILE   also compare against a reference manifest.json (e.g. the hosted pack's):
#                      tools/pack/check_manifest.py; PASS = byte-identical payload
#   --tp2-pack DIR     (27b-quark --tp 1) also check the TP=1 pack against this TP=2 pack;
#                      default: the TP=2 pack beside the output, if there is one
#   --workers N        builder processes (default 8; ~1.5 GB RAM each at the peak)
#   --build-unix N     stamp build time N into the headers (the reference's `built_unix` gives
#                      byte-identical rank files; default: now)
#   --dry-run          (flashnext) plan + ledger only, write nothing
#   --layers A:B       (flashnext) a smoke pack of those layers' experts only
#   --quick            check a 32-op sample instead of every MXFP4 record (27b-quark)
#   --no-check         build only
#   --host             run on the host, not in the container
#
# Env: SABER_PACKS (default /mnt/nvme/saber-packs), SABER_IMAGE (default localhost/saber-dev:latest),
#      SABER_CONTAINER (podman|docker), SABER_PACK_TARGET (cargo target dir; default <repo>/.cache/target-container).
#
# Disk: the pack is written next to nothing else -- 27b-quark 20.9 GB (TP=2) / 19.9 GB (TP=1), 27b-nvfp4
# 19.6 GB / 18.6 GB, the drafter pairing 8.2 GB / 7.7 GB. The build refuses to start without that much
# free space plus 2 GB; flashnext's base pack ~67 GB (it references the checkpoint's bf16 tensors and
# the 102 GB PLE table in place, so the checkpoint must stay where it was when the pack was built). Time: a few minutes on a 12-core desktop CPU (+ the first saber-packplan build),
# then the self-check (a full 27b-quark check reads every MXFP4 record: a few minutes).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }
die() { echo "build_packs.sh: $*" >&2; exit 1; }

[ $# -gt 0 ] || { usage; exit 2; }
CMD=$1; shift
case "$CMD" in -h|--help) usage; exit 0 ;; esac
CHECK_ONLY=0
if [ "$CMD" = check ]; then CHECK_ONLY=1; CMD=${1:-}; shift || true; fi
case "$CMD" in
  27b-quark|27b-nvfp4|27b-quark-drafter|flashnext|test) ;;
  *) die "unknown model '$CMD' (27b-quark, 27b-nvfp4, 27b-quark-drafter, flashnext, check, test; --help)" ;;
esac

CKPT=""; TP=2; OUT=""; REF=""; TP2=""; WORKERS=${SABER_PACK_WORKERS:-8}; BUILD_UNIX=""
DRY=0; LAYERS=""; QUICK=0; NOCHECK=0; HOST=0; INSIDE=0; PACK=""; DPACK=""; TPACK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT=${2:?--checkpoint DIR}; shift 2 ;;
    --tp) TP=${2:?--tp 1|2}; shift 2 ;;
    --out) OUT=${2:?--out DIR}; shift 2 ;;
    --pack) PACK=${2:?--pack DIR}; shift 2 ;;
    --reference) REF=${2:?--reference manifest.json}; shift 2 ;;
    --tp2-pack) TP2=${2:?--tp2-pack DIR}; shift 2 ;;
    --drafter-pack) DPACK=${2:?}; shift 2 ;;
    --target-pack) TPACK=${2:?}; shift 2 ;;
    --workers) WORKERS=${2:?}; shift 2 ;;
    --build-unix) BUILD_UNIX=${2:?}; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --layers) LAYERS=${2:?--layers A:B}; shift 2 ;;
    --no-check) NOCHECK=1; shift ;;
    --host) HOST=1; shift ;;
    --in-container) INSIDE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (--help)" ;;
  esac
done
case "$TP" in 1|2) ;; *) die "--tp $TP: 1 (one card) or 2 (two cards)" ;; esac
PACKS=${SABER_PACKS:-/mnt/nvme/saber-packs}
T=""; [ "$TP" = 1 ] && T=-tp1

# ---------------------------------------------------------------------------------------------
# The work itself (in the container, or on the host with --host). Paths here are the ones this
# side sees.
# ---------------------------------------------------------------------------------------------
inner() {
  ulimit -c 0
  if ! python3 -c 'import numpy' 2>/dev/null; then
    local lib=$REPO/.cache/pack-pylib
    if ! PYTHONPATH=$lib python3 -c 'import numpy' 2>/dev/null; then
      echo "== numpy is not in this image; installing it into .cache/pack-pylib (once)"
      python3 -m pip install -q --disable-pip-version-check --target "$lib" numpy \
        || die "could not install numpy (no network?); rebuild the image with tools/build-image.sh"
    fi
    export PYTHONPATH=$lib${PYTHONPATH:+:$PYTHONPATH}
  fi
  export SABER_PACK_PLANS=${SABER_PACK_PLANS:-$REPO/.cache/pack}
  local P=$REPO/tools/pack
  case "$CMD" in
    test)
      python3 -m unittest discover -s "$P/tests" ;;
    27b-quark|27b-nvfp4)
      local src=quark; [ "$CMD" = 27b-nvfp4 ] && src=nvfp4
      if [ "$CHECK_ONLY" = 0 ]; then
        "$P/build_devfinal.sh" --checkpoint "$CKPT" --source "$src" --tp "$TP" --out "$OUT" \
          --workers "$WORKERS" ${BUILD_UNIX:+--build-unix "$BUILD_UNIX"}
        PACK=$OUT
      fi
      [ "$NOCHECK" = 1 ] && return 0
      check_27b "$src" ;;
    flashnext)
      if [ "$CHECK_ONLY" = 0 ]; then build_flashnext; fi
      [ "$NOCHECK" = 1 ] || [ "$DRY" = 1 ] || [ -n "$LAYERS" ] && return 0
      echo "== self-check $PACK (check_flashnext_pack.py: dense records, expert tier with controls, PLE index, ledger, aliases)"
      nice -n 19 python3 "$P/check_flashnext_pack.py" "$PACK" --checkpoint "$CKPT" --workers "$WORKERS" \
        && verdict 0 || verdict 1 ;;
    27b-quark-drafter)
      if [ "$CHECK_ONLY" = 0 ]; then
        echo "== pair the drafter with the target's bf16 head (drafter_head_pair.py --check-int2)"
        nice -n 19 python3 "$P/drafter_head_pair.py" --drafter-pack "$DPACK" --target-pack "$TPACK" \
          --out "$OUT" --check-int2
        PACK=$OUT
      fi
      [ "$NOCHECK" = 1 ] && return 0
      check_pair ;;
  esac
}

build_flashnext() {
  local P=$REPO/tools/pack pl=$SABER_PACK_PLANS
  echo "== 1/4 record plan (saber-flashnext-packplan)"
  nice -n 19 cargo run -q --release --manifest-path "$REPO/Cargo.toml" -p saber-model \
    --bin saber-flashnext-packplan -- --checkpoint "$CKPT" > "$pl/flashnext-plan.json"
  # The pack is built from the plan WITHOUT the loader aliases; the aliases are a second pass
  # against that base plan (its sha256 is the manifest's plan_sha256).
  python3 -c "import json,sys;p=json.load(open(sys.argv[1]));p.pop('aliases',None);json.dump(p,open(sys.argv[2],'w'))" \
    "$pl/flashnext-plan.json" "$pl/flashnext-plan-base.json"
  local extra=(); [ "$DRY" = 1 ] && extra+=(--dry-run); [ -n "$LAYERS" ] && extra+=(--layers "$LAYERS")
  echo "== 2/4 base pack: expert tier (byte-copied MXFP4, verified per expert), dense records, PLE index"
  nice -n 19 python3 "$P/build_flashnext_pack.py" --plan "$pl/flashnext-plan-base.json" --model "$CKPT" \
    --out "$OUT" --workers "$WORKERS" "${extra[@]}"
  [ "$DRY" = 1 ] && return 0
  cp "$pl/flashnext-plan-base.json" "$OUT/plan.json"
  echo "== 3/4 loader aliases"
  nice -n 19 python3 "$P/build_flashnext_pack.py" --plan "$pl/flashnext-plan.json" --model "$CKPT" \
    --out "$OUT" --only-aliases "$pl/flashnext-plan-base.json"
  [ -n "$LAYERS" ] && return 0
  echo "== 4/4 MTP experts, MXFP4 (mtp_x4)"
  nice -n 19 python3 "$P/build_mtp_x4.py" --model "$CKPT" --pack "$OUT"
  PACK=$OUT
}

check_27b() {
  local src=$1 P=$REPO/tools/pack fails=0 tp
  tp=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['tp_size'])" "$PACK/manifest.json")
  echo "== self-check $PACK (tp $tp) against $CKPT"
  if [ "$src" = quark ]; then
    local plan=$SABER_PACK_PLANS/plan-quark-tp$tp.json
    if [ ! -f "$plan" ]; then
      "$P/build_devfinal.sh" --checkpoint "$CKPT" --source quark --tp "$tp" --plan-only >/dev/null
    fi
    local sample=(); [ "$QUICK" = 1 ] && sample=(--sample 32)
    echo "-- 1. every MXFP4 block == AMD Quark's own dequant of the checkpoint; lm_head + KV scales exact"
    nice -n 19 python3 "$P/check_quark27b.py" --model "$CKPT" --pack "$PACK" --plan "$plan" "${sample[@]}" || fails=$((fails + 1))
    echo "-- 2. negative control: the nibble-swapped checkpoint must FAIL every record"
    nice -n 19 python3 "$P/check_quark27b.py" --model "$CKPT" --pack "$PACK" --plan "$plan" --sample 16 --control || fails=$((fails + 1))
  else
    local head
    head=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print('fp8' if any(r['name']=='lm_head.w8' for r in m['ranks'][0]['records']) else 'other')" "$PACK/manifest.json")
    if [ "$head" = fp8 ]; then
      echo "-- 1. lm_head.w8 / lm_head.ws == the checkpoint's fp8 head bytes"
      nice -n 19 python3 "$P/check_fp8_head.py" "$PACK" --ckpt "$CKPT" || fails=$((fails + 1))
    fi
    [ -n "$REF" ] || echo "-- note: NVFP4 -> MXFP4 is a requantization with no independent oracle here; pass --reference <hosted manifest.json> for the byte-identity check"
  fi
  if [ "$tp" = 1 ] && [ -n "$TP2" ]; then
    echo "-- 3. the TP=1 pack == the TP=2 pack's ranks recombined"
    nice -n 19 python3 "$P/check_tp1_vs_tp2.py" --tp1 "$PACK" --tp2 "$TP2" || fails=$((fails + 1))
  fi
  if [ -n "$REF" ]; then
    echo "-- 4. record table + every record's sha256 == the reference manifest"
    python3 "$P/check_manifest.py" "$PACK" "$REF" || fails=$((fails + 1))
  fi
  verdict "$fails"
}

check_pair() {
  local fails=0
  echo "== self-check $PACK: the drafter's head.rerank.w must be the target's lm_head.w"
  python3 - "$PACK" "$TPACK" <<'EOF' || fails=$((fails + 1))
import json, os, sys
d, t = (json.load(open(os.path.join(p, "manifest.json"))) for p in sys.argv[1:3])
bad = 0
for rd, rt in zip(d["ranks"], t["ranks"]):
    a = {r["name"]: r["sha256"] for r in rd["records"]}
    b = {r["name"]: r["sha256"] for r in rt["records"]}
    ok = a.get("head.rerank.w") == b.get("lm_head.w") and \
        a.get("input_embedding.weight") == b.get("model.language_model.embed_tokens.weight")
    bad += not ok
    print(f"rank {rd['rank']}: head.rerank.w {a.get('head.rerank.w', '?')[:16]} vs lm_head.w "
          f"{b.get('lm_head.w', '?')[:16]}: {'PAIRED' if ok else 'NOT PAIRED'}")
sys.exit(1 if bad else 0)
EOF
  if [ -n "$REF" ]; then
    python3 "$REPO/tools/pack/check_manifest.py" "$PACK" "$REF" || fails=$((fails + 1))
  fi
  verdict "$fails"
}

verdict() {
  if [ "$1" = 0 ]; then echo "== SELF-CHECK PASS: $PACK"; else echo "== SELF-CHECK FAILED ($1 check(s)): $PACK" >&2; return 1; fi
}

if [ "$INSIDE" = 1 ]; then
  inner; exit $?
fi

# ---------------------------------------------------------------------------------------------
# Pre-flight on the host: inputs, names, disk; then the container (or --host).
# ---------------------------------------------------------------------------------------------
need_dir() { [ -d "$1" ] || die "$2: no directory '$1'"; }
GB=$((1000 * 1000 * 1000))
NEED=0
CANON=""
case "$CMD" in
  27b-quark|27b-nvfp4)
    [ -n "$CKPT" ] || die "--checkpoint DIR is required (hf download $([ "$CMD" = 27b-quark ] && echo amd/Qwen3.8-27B-Quark-AWQ-MXFP4 || echo unsloth/Qwen3.8-27B-NVFP4) --local-dir DIR)"
    need_dir "$CKPT" --checkpoint
    [ -f "$CKPT/config.json" ] && ls "$CKPT"/*.safetensors >/dev/null 2>&1 \
      || die "$CKPT: no config.json / *.safetensors (an incomplete download?)"
    CKPT=$(cd "$CKPT" && pwd)
    if [ "$CMD" = 27b-quark ]; then
      CANON=Qwen3.8-27B-Quark-AWQ-MXFP4; DEF=qwen38-27b-quark-awq-mxfp4$T
      NEED=$([ "$TP" = 1 ] && echo 19806 || echo 20838)
    else
      CANON=Qwen3.8-27B-NVFP4; DEF=qwen38-27b-nvfp4-devfinal$T-fp8head
      NEED=$([ "$TP" = 1 ] && echo 18536 || echo 19567)
    fi
    if [ "$CHECK_ONLY" = 1 ]; then
      [ -n "$PACK" ] || PACK=$PACKS/$DEF
      need_dir "$PACK" --pack; PACK=$(cd "$PACK" && pwd); NEED=0
    else
      OUT=${OUT:-$PACKS/$DEF}
    fi
    if [ "$CMD" = 27b-quark ] && [ -z "$TP2" ]; then
      if [ "$CHECK_ONLY" = 1 ]; then base=$PACK; else base=$OUT; fi
      [ "$TP" = 1 ] && [ -f "${base%-tp1}/rank1.pack" ] && TP2=${base%-tp1}
    fi ;;
  flashnext)
    [ -n "$CKPT" ] || die "--checkpoint DIR is required (hf download amd/Qwen3.8-Flash-Next-Quark-MXFP4 --local-dir DIR, ~170 GB)"
    need_dir "$CKPT" --checkpoint; CKPT=$(cd "$CKPT" && pwd)
    [ -f "$CKPT/config.json" ] || die "$CKPT: no config.json"
    [ "$TP" = 2 ] || die "flashnext: the pack is EP=2 (two cards); --tp 1 is not supported"
    if [ "$CHECK_ONLY" = 1 ]; then
      PACK=${PACK:-$PACKS/qwen38-flashnext-quark-ep2}; need_dir "$PACK" --pack; PACK=$(cd "$PACK" && pwd)
    else
      OUT=${OUT:-$PACKS/qwen38-flashnext-quark-ep2}
      NEED=$([ "$DRY" = 1 ] && echo 0 || echo 67000)
    fi ;;
  27b-quark-drafter)
    [ -n "$TPACK" ] || die "--target-pack DIR is required (the Quark target pack of the same --tp)"
    need_dir "$TPACK" --target-pack; TPACK=$(cd "$TPACK" && pwd)
    if [ "$CHECK_ONLY" = 1 ]; then
      [ -n "$PACK" ] || die "check 27b-quark-drafter needs --pack DIR (the paired drafter) and --target-pack DIR"
      need_dir "$PACK" --pack; PACK=$(cd "$PACK" && pwd)
    else
      [ -n "$DPACK" ] || die "--drafter-pack DIR is required (a DFlash2 drafter pack with a bf16 head.rerank.w)"
      need_dir "$DPACK" --drafter-pack; DPACK=$(cd "$DPACK" && pwd)
      tpt=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['tp_size'])" "$TPACK/manifest.json")
      [ "$tpt" = 1 ] && T=-tp1 || T=""
      OUT=${OUT:-$PACKS/qwen38-dflash2-drafter-quarkhead$T}
      NEED=$([ "$tpt" = 1 ] && echo 7669 || echo 8190)
    fi ;;
esac
if [ -n "$OUT" ] && [ "$CHECK_ONLY" = 0 ]; then
  [ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT" 2>/dev/null)" ] || die "$OUT exists and is not empty; remove it or pass another --out"
  mkdir -p "$OUT" || die "cannot create $OUT"
  OUT=$(cd "$OUT" && pwd)
  avail=$(( $(df -Pk "$OUT" | awk 'NR==2{print $4}') / 1000 ))   # MB
  if [ "$NEED" -gt 0 ] && [ "$avail" -lt $((NEED + 2000)) ]; then
    rmdir "$OUT" 2>/dev/null || true
    die "not enough disk under $(dirname "$OUT"): the pack needs $((NEED / 1000)).$(( (NEED % 1000) / 100 )) GB (+2 GB margin), $((avail / 1000)) GB free"
  fi
  [ "$CMD" = 27b-quark-drafter ] && rmdir "$OUT"   # the pairing tool creates it and refuses an existing one
  echo "== $CMD tp $TP -> $OUT (needs ~$((NEED / 1000)).$(( (NEED % 1000) / 100 )) GB; $((avail / 1000)) GB free)"
fi
[ -z "$REF" ] || { [ -f "$REF" ] || [ -f "$REF/manifest.json" ] || die "--reference: no manifest at '$REF'"; }
[ -z "$TP2" ] || need_dir "$TP2" --tp2-pack

if [ "$HOST" = 1 ]; then
  if [ -n "$CANON" ] && [ "$(basename "$CKPT")" != "$CANON" ]; then
    echo "note: the checkpoint directory's name is part of the pack's tier id; '$(basename "$CKPT")' is not" \
         "'$CANON', so the headers (not the weights) will differ from the hosted pack's" >&2
  fi
  inner; exit $?
fi

# The container sees: the repo at /work, the checkpoint read-only at /models/<its canonical
# name> (the name is part of the tier id, so a renamed download still reproduces the hosted
# pack), the output's parent read-write at /out, other packs read-only under /in.
. "$REPO/tools/lib/container.sh"
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
"$CTR" image inspect "$IMG" >/dev/null 2>&1 || die "no image $IMG: build it with tools/build-image.sh"
TGT=${SABER_PACK_TARGET:-$REPO/.cache/target-container}
mkdir -p "$TGT" "$REPO/.cache/cargo-registry" "$REPO/.cache/pack"
M=(-v "$REPO":/work -v "$TGT":/target -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry)
A=("$CMD" --in-container --tp "$TP" --workers "$WORKERS")
[ "$CHECK_ONLY" = 1 ] && A=(check "${A[@]}")
if [ "$CMD" = flashnext ]; then
  M+=(-v "$CKPT":"$CKPT":ro); A+=(--checkpoint "$CKPT")
elif [ -n "$CKPT" ]; then
  name=${CANON:-$(basename "$CKPT")}
  M+=(-v "$CKPT":/models/"$name":ro); A+=(--checkpoint /models/"$name")
fi
if [ -n "$OUT" ] && [ "$CHECK_ONLY" = 0 ]; then
  M+=(-v "$(dirname "$OUT")":/out); A+=(--out /out/"$(basename "$OUT")")
fi
[ -z "$PACK" ] || { M+=(-v "$PACK":/in/pack:ro); A+=(--pack /in/pack); }
[ -z "$DPACK" ] || { M+=(-v "$DPACK":/in/drafter:ro); A+=(--drafter-pack /in/drafter); }
[ -z "$TPACK" ] || { M+=(-v "$TPACK":/in/target:ro); A+=(--target-pack /in/target); }
[ -z "$TP2" ] || { M+=(-v "$(cd "$TP2" && pwd)":/in/tp2:ro); A+=(--tp2-pack /in/tp2); }
if [ -n "$REF" ]; then
  [ -d "$REF" ] && REF=$REF/manifest.json
  M+=(-v "$(cd "$(dirname "$REF")" && pwd)/$(basename "$REF")":/in/reference.json:ro); A+=(--reference /in/reference.json)
fi
[ -z "$BUILD_UNIX" ] || A+=(--build-unix "$BUILD_UNIX")
[ "$QUICK" = 0 ] || A+=(--quick)
[ "$DRY" = 0 ] || A+=(--dry-run)
[ -z "$LAYERS" ] || A+=(--layers "$LAYERS")
[ "$NOCHECK" = 0 ] || A+=(--no-check)
CNAME=saber-pack-build-$$
trap '"$CTR" rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
"$CTR" run --rm --name "$CNAME" "${CTR_USER[@]}" --security-opt label=disable "${M[@]}" \
  -e CARGO_TARGET_DIR=/target -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES= \
  -w /work --entrypoint /bin/bash "$IMG" /work/tools/build_packs.sh "${A[@]}"
