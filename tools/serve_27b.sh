#!/usr/bin/env bash
# Boot Saber's served Qwen3.8-27B (the main-branch defaults: NVFP4->MXFP4 devfinal pack, fp8 head,
# the DFlash2 drafter, TP=2) for a client to point at. Mirrors tools/gates/betterbench.sh's server half.
#
#   tools/serve_27b.sh              # build the release binary in the dev image, then serve on :8080
#   tools/serve_27b.sh --no-build   # reuse the last container build
#   tools/serve_27b.sh stop         # stop the server container (whichever of TP=2 / TP=1 is running)
#   tools/serve_27b.sh --tp 1 [--no-build]   # ONE card (TP=1; or TP=1 in the environment): the -tp1-fp8head
#     packs from `tools/fetch_packs.sh 27b-tp1`, on HIP device DEVICE (default 0), container
#     saber-27b-tp1-serve; defaults MAX_CTX 65536, MAX_SEQS 4, BATCHES 1,2,4 (every env knob below
#     still overrides). `--tp 1 stop` stops only the TP=1 container. TP=2 (`--tp 2`) is the default.
#   SERVE_PREFIX="/opt/rocm/bin/rocprofv3 --kernel-trace -f csv -d DIR --" tools/serve_27b.sh ...   # CMP27D: a DIAGNOSTIC tool wrapped around `saber serve` (both ranks)
#   PORT=8100 MAX_CTX=16384 WIDTHS=1,5,8 tools/serve_27b.sh ...   # port / per-sequence context (default 262144) / verify-width ladder (default 1,8)
#   DRAFT_BLOCK=4 WIDTHS=1,4 tools/serve_27b.sh ...   # P27-7: the drafter's block (default 8; 4 = the priced option, GDN slots 65 -> 33)
#   VISION (P27-11, 2026-10-07): `--vision` = `--vision lazy` -- the tower streams from pinned host RAM through the
#   idle prefill arena per image (no VRAM billed, the full 262k KV pool: 47,335 blocks a rank; images up to
#   16,384 patches, ~+30 ms a rank per image). `--vision on` = the resident tower (~1.9 GiB a rank; needs
#   MAX_CTX=131072 beside the width-8 verify arenas).
#   KV (P27-6, default since 2026-10-05): the drafter's KV is paged by admission -> at 262,144 the pool is
#   47,335 blocks = 757,360 tokens a rank (2 x 262k sessions); `--engine-draft-kv dense` = the control (21,169 = 338,704).
#   LEDGER (P27-7, default since 2026-10-07): `--engine-ledger lean` replaces --engine-budget-mib / headroom with
#   31,824 / 848 MiB -> 51,906 blocks = 830,496 tokens a rank (3 x 262k); `--engine-ledger today` = the control.
#
#   tools/serve_27b.sh --no-build --warm-prompts ~/pi-system.txt   # FN66: prefill your system prompt(s)
#     once at boot (prefix caching turns on); see docs/HOWTO-SERVE.md "Warm prompts"
# Extra arguments after the subcommand are passed to `saber serve`. OpenAI-compatible: /v1/models,
# /v1/chat/completions, /v1/completions; model id Qwen3.8. Logs: podman logs -f saber-27b-serve (or docker logs ...). Docker works too: SABER_CONTAINER=docker.
set -uo pipefail
REPO=${SABER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # SABER_REPO: a hold run from a worktree serves from the main checkout's .cache
. "$REPO/tools/lib/container.sh"   # CTR = podman or docker (SABER_CONTAINER=docker to force)
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
SHARED=${SABER_SHARED:-$HOME/saber-cache}
# The launcher's own leading words, in any order: `--tp 1|2`, `--no-build`, `stop`.
TP_GIVEN=${TP:+env}
TP=${TP:-2}
BUILD=1
STOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tp) TP=${2:-}; TP_GIVEN=flag; shift; [ $# -gt 0 ] && shift ;;
    --tp=*) TP=${1#*=}; TP_GIVEN=flag; shift ;;
    --no-build) BUILD=0; shift ;;
    stop) STOP=1; shift ;;
    *) break ;;
  esac
done
case "$TP" in
  2) TP_DEF_CNAME=saber-27b-serve ;;
  1) TP_DEF_CNAME=saber-27b-tp1-serve ;;
  *) echo "--tp $TP: the 27B launcher serves TP=2 (two cards) or TP=1 (one card)" >&2; exit 2 ;;
esac
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
PACKS=${SABER_PACKS:-/mnt/nvme/saber-packs}
MODEL=${MODEL:-Qwen3.8}
PREBUILT=${PREBUILT:-prebuilt/gfx1201}   # CMP27E: a hold's frozen prebuilt dir (a path the container sees: under /work or $SHARED)
SABER=${SABER_BIN:-.cache/target-container/release/saber}   # P27-9d: SABER_BIN = another build, a path under the repo as the container sees /work
OUT=$SHARED/serve-27b
ulimit -c 0
mkdir -p "$OUT"

if [ "$STOP" = 1 ]; then
  # `stop` stops the CNAME given, else the TP the command names, else whichever 27B server runs.
  if [ -n "${CNAME:-}" ]; then names=("$CNAME")
  elif [ -n "$TP_GIVEN" ]; then names=("$TP_DEF_CNAME")
  else names=(saber-27b-serve saber-27b-tp1-serve); fi
  stopped=0
  for n in "${names[@]}"; do
    $CTR container exists "$n" >/dev/null 2>&1 || $CTR inspect "$n" >/dev/null 2>&1 || continue
    $CTR rm -f "$n" >/dev/null 2>&1 && { echo "stopped $n"; stopped=1; }
  done
  j="${names[*]}"
  [ "$stopped" = 1 ] || echo "${j// / or } was not running"
  exit 0
fi

CNAME=${CNAME:-$TP_DEF_CNAME}
if [ "$TP" = 2 ]; then
  MAX_CTX=${MAX_CTX:-262144}
  TIER=${TIER:-qwen38-27b-nvfp4-devfinal-fp8head}
  DRAFT=${DRAFT:-qwen38-dflash2-drafter-fp8head}
  TP_ARGS="--engine-tp 2 --engine-device 0 --engine-peer-device 1"
  HIP_DEVICES=${HIP_DEVICES:-0,1}
  BATCHES=${BATCHES:-1,2,3,4,8}
  MAX_SEQS=${MAX_SEQS:-8}
  FETCH=27b
else
  # One card. The container sees only HIP device DEVICE, so inside it the engine's device is 0.
  [ -z "${HIP_DEVICES:-}" ] || { echo "TP=1 takes DEVICE=<one HIP device>, not HIP_DEVICES" >&2; exit 2; }
  DEVICE=${DEVICE:-0}
  [[ "$DEVICE" =~ ^[0-9]+$ ]] || { echo "DEVICE=$DEVICE: TP=1 serves on ONE HIP device index" >&2; exit 2; }
  MAX_CTX=${MAX_CTX:-65536}
  TIER=${TIER:-qwen38-27b-nvfp4-devfinal-tp1-fp8head}
  DRAFT=${DRAFT:-qwen38-dflash2-drafter-tp1-fp8head}
  TP_ARGS="--engine-tp 1 --engine-device 0"
  HIP_DEVICES=$DEVICE
  BATCHES=${BATCHES:-1,2,4}
  MAX_SEQS=${MAX_SEQS:-4}
  FETCH=27b-tp1
fi
# Explicit compatibility profile for hosts without GPU peer access. The runtime image must
# provide RCCL >= 2.30.4; it need not be installed in the build image. Native P2P is unchanged.
COLLECTIVE_ARGS=()
COLLECTIVE_ENV=()
if [ -n "${RCCL_LIB:-}" ]; then
  [ "$TP" = 2 ] || { echo "RCCL_LIB requires TP=2" >&2; exit 2; }
  COLLECTIVE_ARGS+=(--engine-rccl-lib "$RCCL_LIB")
  COLLECTIVE_ENV=(-e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 -e NCCL_PROTO=Simple)
fi
# FN66: `--warm-prompts FILE` -- the file is copied where the container sees it ($OUT) and the
# launcher reports ready only after the server's `warmprompts done` line.
ARGS=()
WARM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --warm-prompts) WARM=${2:-}; shift; [ $# -gt 0 ] && shift ;;
    --warm-prompts=*) WARM=${1#*=}; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -n "$WARM" ]; then
  [ -f "$WARM" ] || { echo "--warm-prompts: no file '$WARM'" >&2; exit 1; }
  cp -- "$WARM" "$OUT/warm-prompts.txt" || exit 1
  ARGS+=(--warm-prompts "$OUT/warm-prompts.txt")
fi

if systemctl --user is-active --quiet qwen_prod_nvfp4; then
  echo "prod's unit qwen_prod_nvfp4 is ACTIVE; refusing to boot a second server on the cards" >&2; exit 1
fi
if [ "$TP" = 2 ]; then
  if $CTR ps --format '{{.Names}}' | grep -qE '^saber-fn|^saber-flashnext-serve|^saber-27b-tp1-serve'; then
    echo "another saber server or an agent hold holds the cards; stop it or wait:" >&2
    $CTR ps --format '  {{.Names}} {{.Status}}' | grep saber >&2; exit 1
  fi
else
  # TP=1: refuse if any running saber container holds this card (one with no HIP_VISIBLE_DEVICES
  # is taken to hold every card), our own name included -- stop it first, or pick another DEVICE.
  for n in $($CTR ps --format '{{.Names}}' | grep '^saber'); do
    hv=$($CTR inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$n" 2>/dev/null | sed -n 's/^HIP_VISIBLE_DEVICES=//p' | head -1)
    case ",${hv:-all}," in
      *",$DEVICE,"* | ",all,")
        echo "container $n holds HIP device $DEVICE (HIP_VISIBLE_DEVICES=${hv:-unset}); stop it, wait, or pick another DEVICE" >&2
        exit 1 ;;
    esac
  done
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT\$"; then echo "port $PORT is already in use" >&2; exit 1; fi
for f in "$REPO/.cache/template_corpus.tokenizer.json" "$REPO/.cache/template_corpus.jinja"; do
  [ -f "$f" ] || { echo "missing $f (the 27B serve plane's tokenizer / chat template; run tools/fetch_packs.sh $FETCH)" >&2; exit 1; }
done
[ -d "$PACKS/$TIER" ] && [ -d "$PACKS/$DRAFT" ] || { echo "missing pack $PACKS/$TIER or $PACKS/$DRAFT (run tools/fetch_packs.sh $FETCH)" >&2; exit 1; }
if [ "$TP" = 1 ]; then
  for f in "$PACKS/$TIER/rank0.pack" "$PACKS/$DRAFT/rank0.pack"; do
    [ -f "$f" ] || { echo "missing $f (TP=1 needs the single-card packs: run tools/fetch_packs.sh 27b-tp1)" >&2; exit 1; }
  done
  [ ! -e "$PACKS/$TIER/rank1.pack" ] || { echo "$PACKS/$TIER is a TP=2 pack (it has rank1.pack); TP=1 needs the -tp1 packs from tools/fetch_packs.sh 27b-tp1" >&2; exit 1; }
fi

if [ "$BUILD" = 1 ]; then
  mkdir -p "$REPO/.cache/target-container" "$REPO/.cache/cargo-registry"
  echo "== $(date '+%F %T') building the release binary in the dev image ($(git -C "$REPO" rev-parse --short HEAD))"
  $CTR run --rm "${CTR_USER[@]}" --security-opt label=disable -v "$REPO":/work \
    -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry -e CARGO_TARGET_DIR=/work/.cache/target-container \
    -w /work --entrypoint bash "$IMG" -c "cargo build --release -p saber-cli 2>&1 | tail -3" | tee "$OUT/build.log"
fi
[ -x "$REPO/$SABER" ] || { echo "no release binary at $SABER" >&2; exit 1; }

CACHE_REAL=$(readlink -f "$REPO/.cache")
SERVE_CMD="$SABER serve --engine real $TP_ARGS \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7 --pack /packs/$TIER --prebuilt ${PREBUILT:-prebuilt/gfx1201} \
  --tokenizer /work/.cache/template_corpus.tokenizer.json --chat-template /work/.cache/template_corpus.jinja \
  --engine-max-ctx $MAX_CTX --engine-prefill-rows 96,128,192,256,320,384,512,768,1024,1536,2048 \
  --engine-decode-batches $BATCHES --engine-decode-widths ${WIDTHS:-1,8} --engine-max-seqs $MAX_SEQS --engine-draft-block ${DRAFT_BLOCK:-8} --engine-draft-pack /packs/$DRAFT \
  --engine-budget-mib ${BUDGET_MIB:-30500} --engine-rectangular-widths --served-model-name $MODEL --host $HOST --port $PORT ${COLLECTIVE_ARGS[*]} ${ARGS[*]}"
# FN73: a server that exits is reported with ITS exit code (86 = a rank failed: the log's last
# line is `saber: FATAL rank R: <reason>`), and this script exits with it -- never a masked 1.
died() {
  local rc
  rc=$($CTR inspect -f '{{.State.ExitCode}}' "$CNAME" 2>/dev/null)
  [[ "$rc" =~ ^[0-9]+$ ]] || rc=1
  echo "the server exited $1 with code $rc$([ "$rc" = 86 ] && echo ' (a rank failed: FN73)'):" >&2
  $CTR logs "$CNAME" 2>&1 | tail -30 >&2
  [ "$rc" -ne 0 ] && exit "$rc"
  exit 1
}
$CTR rm -f "$CNAME" >/dev/null 2>&1 || true
$CTR run -d --name "$CNAME" --security-opt label=disable --security-opt seccomp=unconfined \
  "${CTR_GPU[@]}" "${CTR_USER[@]}" --ipc=host --network=host \
  -v "$REPO":/work -v "$PACKS":/packs:ro -v "$CACHE_REAL":"$CACHE_REAL" -v "$SHARED":"$SHARED" \
  -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry -w /work \
  "${COLLECTIVE_ENV[@]}" -e HIP_VISIBLE_DEVICES="$HIP_DEVICES" -e GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}" -e CUDA_VISIBLE_DEVICES= -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e CARGO_TARGET_DIR=/work/.cache/target-container --entrypoint /bin/bash "$IMG" \
  -c "ulimit -c 0; exec ${SERVE_PREFIX:-} $SERVE_CMD" >/dev/null || { echo "$CTR run failed" >&2; exit 1; }
echo "== $(date '+%T') booting $CNAME on :$PORT ($([ "$TP" = 1 ] && echo "TP=1 on HIP device $DEVICE, ")max ctx $MAX_CTX per sequence; the pack loads from NVMe)"
for i in $(seq 1 900); do
  curl -s --max-time 2 "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  if [ "$($CTR inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" != true ]; then
    died "before it listened"
  fi
  sleep 1
done
curl -s --max-time 2 "localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "never listened in 900 s" >&2; exit 1; }
if [ -n "$WARM" ]; then
  echo "== $(date '+%T') warming the prompt cache from $WARM (a request sent now queues behind it)"
  for j in $(seq 1 1800); do
    $CTR logs "$CNAME" 2>&1 | grep -q '^warmprompts done' && break
    if [ "$($CTR inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" != true ]; then
      died "while warming"
    fi
    sleep 2
  done
  $CTR logs "$CNAME" 2>&1 | grep -E '^(fnserve: --warm-prompts|prefixcache|warmprompt)'
fi
echo "== $(date '+%T') serving after ${i} s: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1  (model: $MODEL)"
curl -s "localhost:$PORT/v1/models" | head -c 300; echo
echo "== a later exit: \`$CTR wait $CNAME\` prints the server's exit code (86 = a rank failed; the log's last line names it)"
