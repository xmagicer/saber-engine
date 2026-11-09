#!/usr/bin/env bash
# Boot Saber's served Qwen3.8 Flash-Next (the main-branch defaults) for a client to point at.
#
#   tools/serve_flashnext.sh            # build the release binary in the dev image, then serve on :8080 (prod's usual port)
#   tools/serve_flashnext.sh --no-build # reuse the last container build
#   tools/serve_flashnext.sh stop       # stop the server container
#   PORT=8100 tools/serve_flashnext.sh  # another port
#   SERVE_PREFIX="/opt/rocm/bin/rocprofv3 --kernel-trace -f csv -d DIR --" tools/serve_flashnext.sh --no-build   # ATTN-OPT-3: a DIAGNOSTIC tool around `saber engine` (both ranks)
#   tools/serve_flashnext.sh --no-build --record-routes   # (or SABER_RECORD_ROUTES=1) record the routes YOUR traffic takes
#
# Expert placement from your own traffic (FN65, 2026-10-06): `--record-routes` writes the served
# engine's routes to /mnt/nvme/saber-kv/routes/<date>/ (rotated files every 20,000 steps, a 16 GiB
# ring; encoded and written by a writer thread OFF the step path -- the rank thread only hands the
# route log it already read over a bounded queue, ~1 us a step; bytes and placement unchanged).
# After a day of real use:
#   target/release/saber flashnext-profile rebuild --trace /mnt/nvme/saber-kv/routes/<date> \
#       --pack /mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2        # -> profile/route_profile.user.json (an ADDED member)
#   tools/serve_flashnext.sh --no-build --fn-profile user                # warm-start the expert tier from it
# `--fn-profile pack` (= no flag) is the DEFAULT; the pack's own profile/route_profile.json is never rewritten.
#
# Any extra arguments after the subcommand are passed to `saber engine` (e.g. --fn-thinking off).
#   tools/serve_flashnext.sh --no-build --warm-prompts ~/pi-system.txt   # FN66: prefill your system
#     prompt(s) once at boot (the prompt cache turns on; max ctx 10,240 unless --fn-max-ctx); see
#     docs/HOWTO-SERVE.md "Warm prompts"
# The server is OpenAI-compatible: /v1/models, /v1/chat/completions, /v1/completions.
# Logs: podman logs -f saber-flashnext-serve. Refuses to start while prod's unit is active.
# --prompt-ids is optional under --fn-serve (the serve plane does not run the set); passed only if $SHARED/ws-fn3/ref/run1 exists.
# Expert tier (FN54c, 2026-10-05): second-chance insertion at b4/b8 (`--xtier-second-chance 4:8,8:8`) is the
# DEFAULT -- conc-8 +5 %, link bytes -20 %, bit-exact; `--xtier-second-chance 0` = the pre-FN54c law.
#
# Context (FN51b/FN51c, 2026-10-02): `--fn-max-ctx 262144` serves the checkpoint's 262,144 tokens (THE SERVED DEFAULT since FN51g, 2026-10-02 -- a plain `--fn-serve` boot runs 262,144 with MTP on; `--fn-max-ctx 10240` = the old control;
# `run::DEFAULT_FN_SERVED_MAX_CTX`), served by ADMISSION BY RESERVATION over one pinned RAM slab a
# rank sized by `kvwire::admit_slab_bytes` (FN51e: one max-context request + one 10,240 beside it =
# 3.386 GiB / 272,528 tokens at 262,144; FN51's 3.65 = `--fn-kv-ram-gib 3.65`): each request reserves
# prompt + max_tokens (+ 64 tokens of open pages); a request without max_tokens reserves up to 10,240
# tokens as before, or prompt + 16,384 when its prompt is longer (`serve::IMPLICIT_HEADROOM`). One
# 262k request runs beside one 10,240 request, or 26 x 10,240 without it; FN51e (MTP fixed by FN51d):
# the 262k + MTP boot leaves MemAvailable ~16 GiB idle (the box's other ~10 GiB + saber's 34.1-35.1
# GiB RSS) and conc-8 trails the 10,240 boot ~5 %, so 10,240 stays the default; a request that does not fit QUEUES until the slab
# frees, a prompt + max_tokens past 262,144 is REFUSED by name. The plain boot (10,240) is the
# pre-FN51b law (8 x 10,240 sequences, no admission). FN51g: every gate but host RAM passes (conc-8 -1.6 %);
# MemAvailable 14.5-16.4 GiB because ~6 GiB of RAM is zram holding leaked /tmp/saber-fn*-<pid>-* test dirs --
# clear them first (free -g should show ~55 GiB available with no server) before a 262,144 boot.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$REPO/tools/lib/container.sh"   # CTR = podman or docker (SABER_CONTAINER=docker to force)
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
SHARED=${SABER_SHARED:-$HOME/saber-cache}
NVME=${SABER_NVME:-/mnt/nvme}              # where models/, saber-packs/ and saber-kv/ live
PORT=${PORT:-8080}                       # the port the clients and BetterBench use (prod's)
# FN61f / user's word 2026-10-07: the fp8 W8A8 TRUNK is the served default (GDN + QSA at W8A8 incl. prefill,
# head + HC bf16): +7 % single / +12.5 % conc-8, GSM8K equal, +492 expert slots a rank; its harness KL is
# under the 0.0227 bar on 23/25 prompts (max 0.045). It needs the `-trunk8` pack variant (the served pack's
# members linked + the W8 records). The bf16 trunk = the control: `FN_TRUNK=bf16 tools/serve_flashnext.sh --no-build`.
# FN108 / the user's word 2026-10-09 ~16:55 ("make our mixers and head fp8 also"), FLIPPED on the gate:
# `w8-hc8-head` = w8 + the HC mixing matrices at fp8 (the reference's hc8m, the pack's shared/hc8.*) + the W8 head
# (FN61g): BetterBench --quick 4 + 4 single 140.0 vs 134.7, conc-8 347.3 vs 323.5, update p99 22.9 vs 24.3,
# TTFT p50 73 vs 80 ms; +355 expert slots a rank; KL 23/25 (p04 = FN80c's EP-draw near-tie), GSM8K 97.80.
# Controls: FN_TRUNK=w8 (the pre-FN108 default), w8-hc8 (mixers only), w8-head (head only), bf16.
# `tools/fetch_packs.sh flashnext` installs the hosted pack as qwen38-flashnext-quark-trunk8 (the same
# files as the maintainers' qwen38-flashnext-quark-ep2-trunk8, which is used when present).
if [ -d "$NVME/saber-packs/qwen38-flashnext-quark-ep2-trunk8" ]; then PACK_DEFAULT=$NVME/saber-packs/qwen38-flashnext-quark-ep2-trunk8
else PACK_DEFAULT=$NVME/saber-packs/qwen38-flashnext-quark-trunk8; fi
PACK=${PACK:-$PACK_DEFAULT}
FN_TRUNK=${FN_TRUNK:-w8-hc8-head}
# FN68b: the QSA K/V page dtype (`--fn-kv-dtype`; fp8 reads the pack's calibrated kv_scales.json).
# fp8 = THE DEFAULT since 2026-10-08 (the user's word; KL 25/25 vs bf16's 23, conc-8 +3.5 %, single -3.8 %, 4.3 GiB pinned RAM back, vision at 262k); FN_KV=bf16 = the control.
FN_KV=${FN_KV:-fp8}
# FN76: the elastic KV borrow at the served boot. 0.5 GiB keeps the tier COMPLETE: the static
# slots (6,133 a rank at fp8 KV, 262k) leave 6,155 records for the RAM L2, which is the pinned law's
# ceiling (34.90 of 36.40 GiB); 1.5 carves ~410 static slots and the L2 no longer covers the rest
# (NVMe on the step path: served single -20 % in FN76 h1). The P2PDMA pool's slots (FN76) add VRAM
# without touching the pinned law.
FN_ELASTIC_KV_MAX=${FN_ELASTIC_KV_MAX:-0.5}
# FN134 (2026-10-15, hold h2, the orchestrator's merge = the flip): the routed experts TP-split
# (`--fn-experts tp2`: every rank holds half of every expert, the -trunk8 pack links
# rank<r>/experts.tp2.xtier; the MTP experts stay the slab4 records, EP): BetterBench --quick 4 + 4
# single 162.1 -> 166.9, conc-8 413.9 -> 420.2, update p99 21.95 -> 20.38 ms, TTFT p50 65 -> 63 ms;
# envelope PASS; the MoE-out imbalance b8 2.42 -> 0.06 ms. FN_EXPERTS=ep = the control.
FN_EXPERTS=${FN_EXPERTS:-tp2}
CNAME=${CNAME:-saber-flashnext-serve}
SABER=/work/.cache/target-container/release/saber
# --prompt-ids is optional for a served boot; the maintainers' FN3 set is passed when present (identical geometry either way)
PROMPTS=$SHARED/ws-fn3/ref/run1
PROMPT_IDS_ARG=""; [ -d "$PROMPTS" ] && PROMPT_IDS_ARG="--prompt-ids $PROMPTS"
OUT=$SHARED/serve-flashnext
ulimit -c 0
mkdir -p "$OUT"

if [ "${1:-}" = stop ]; then
  $CTR rm -f "$CNAME" >/dev/null 2>&1 && echo "stopped $CNAME" || echo "$CNAME was not running"
  exit 0
fi
BUILD=1
RECORD=${SABER_RECORD_ROUTES:-0}
while :; do
  case "${1:-}" in
    --no-build) BUILD=0; shift ;;
    --record-routes) RECORD=1; shift ;;
    *) break ;;
  esac
done
REC_ARGS=""
if [ "$RECORD" = 1 ]; then
  ROUTES=$NVME/saber-kv/routes/$(date +%F)
  mkdir -p "$ROUTES" || { echo "cannot create $ROUTES" >&2; exit 1; }
  REC_ARGS="--fn-route-record $ROUTES"
  echo "== recording the served routes to $ROUTES (rebuild: saber flashnext-profile rebuild --trace $ROUTES --pack $PACK)"
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
# LR-F / LR-G / the user's word 2026-10-24: THE AGENT-LOAD DEFAULTS. A/B on 16 staggered agent sessions (same
# binary, A C D A C D): `--fn-kv-lazy 4096` (a request reserves prompt + min(max_tokens, 4096) KV and takes blocks
# as it decodes; the youngest lazy sequence is preempted and recomputed if growth is refused -- 0 preemptions on
# the agent load) + `--fn-prefix-tree-ram-gib 4.5` (the tree tier the 3.2 GiB slab had to give up) +
# `--fn-prefill-yield 1` (decoders step between prefill chunks: longest gap 5-9.6 s -> ~1.6 s) = tok/s +10.3 %
# over the 3.2 GiB-slab + 1.5 GiB-tree pair (125.1 vs 113.4; +18 % over the 10-23 prod flags), decoder stall
# time -39 %, tree reload 62 -> 76 %. Gate: temp-0 bytes identical A vs C (5 prompts incl. a 14k-token one, all
# 6 pairs), BetterBench --quick A vs C single 166.1 vs 168.7, conc-8 388.4 vs 388.8 (+0.1 %).
# Controls: `--fn-kv-ram-gib 3.2 --fn-prefix-tree-ram-gib 1.5` (the 3.2 GiB slab, +7 % over the 10-23 flags), or
# FN_AGENT_DEFAULTS=0 for none of these. A flag you pass wins.
if [ "${FN_AGENT_DEFAULTS:-1}" = 1 ]; then
  # 10-25 LR-K: the admission slab at the host's max for tree 4.5 (agent load +6 % chars/s, KV refusals -70 %, bytes identical,
  # BetterBench neutral). Re-derive the number after any change to the expert slots, the L2, the tree or MemTotal.
  # 10-25 LR-N: coupled drafting (sampled rows draft with the target's own seeded noise): single +5.7 %, conc-8 +1.7 %, agent chars/s +10 %,
  # temp-0 bytes and seeded outputs identical on/off (every committed token is still the target's draw). `--spec-couple off` = the control.
  for d in "--fn-kv-lazy 1024" "--fn-prefix-tree-ram-gib 4.5" "--fn-prefill-yield 1" "--fn-ring-idle" "--fn-kv-admit-slab-tokens 338592" "--spec-couple on"; do
    set -- $d
    case " ${ARGS[*]:-} " in *" --fn-kv-ram-gib "*) case "$1" in --fn-kv-lazy|--fn-kv-admit-slab-tokens) continue ;; esac ;; esac
    case " ${ARGS[*]:-} " in *" $1 "*|*" $1="*) ;; *) ARGS+=("$@") ;; esac
  done
fi
if [ -n "$WARM" ]; then
  [ -f "$WARM" ] || { echo "--warm-prompts: no file '$WARM'" >&2; exit 1; }
  cp -- "$WARM" "$OUT/warm-prompts.txt" || exit 1
  ARGS+=(--warm-prompts "$OUT/warm-prompts.txt")
fi

if systemctl --user is-active --quiet qwen_prod_nvfp4; then
  echo "prod's unit qwen_prod_nvfp4 is ACTIVE; refusing to boot a second server on the cards" >&2; exit 1
fi
if $CTR ps --format '{{.Names}}' | grep -qE '^saber-fn'; then
  echo "a saber-fn* hold container is running (an agent holds the cards); stop it or wait" >&2
  $CTR ps --format '  {{.Names}} {{.Status}}' | grep saber-fn >&2; exit 1
fi

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT\$"; then echo "port $PORT is already in use" >&2; ss -ltnp 2>/dev/null | grep ":$PORT " >&2; exit 1; fi
if [ "$BUILD" = 1 ]; then
  mkdir -p "$REPO/.cache/target-container" "$REPO/.cache/cargo-registry"
  echo "== $(date '+%F %T') building the release binary in the dev image ($(git -C "$REPO" rev-parse --short HEAD))"
  $CTR run --rm "${CTR_USER[@]}" --security-opt label=disable -v "$REPO":/work \
    -v "$REPO/.cache/cargo-registry":/opt/rust/cargo/registry -e CARGO_TARGET_DIR=/work/.cache/target-container \
    -w /work --entrypoint bash "$IMG" -c "cargo build --release -p saber-cli 2>&1 | tail -3" | tee "$OUT/build.log"
fi
[ -x "$REPO/.cache/target-container/release/saber" ] || { echo "no release binary at .cache/target-container/release/saber" >&2; exit 1; }

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
# FN71c: each card's P2PDMA pool (amdgpu.p2pmem_mb) handed to the container rw at /saber-p2p/<bdf>
# (the engine's --xtier-p2p-dir): the boot fill / the valve's pool bounce use it; no pool = no mount
# and the engine names the host route on its `fnfill` lines.
P2P=()
if [ -f "$REPO/tools/capture/p2p_mounts.sh" ]; then
  # shellcheck source=capture/p2p_mounts.sh
  . "$REPO/tools/capture/p2p_mounts.sh"
  p2p_mounts P2P
fi
# GPU_MAX_HW_QUEUES=3: the user's rule 10-26 (other devs report a slowdown at 4 or more); the env var overrides for a probe
$CTR rm -f "$CNAME" >/dev/null 2>&1 || true
$CTR run -d --name "$CNAME" --security-opt label=disable --security-opt seccomp=unconfined \
  "${CTR_GPU[@]}" "${CTR_USER[@]}" --ipc=host --network=host \
  -e HIP_VISIBLE_DEVICES="${HIP_DEVICES:-0,1}" -e GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}" -e LD_LIBRARY_PATH=/opt/rocm/lib -e PYTHONDONTWRITEBYTECODE=1 "${P2P[@]}" \
  -v "$SHARED":"$SHARED" -v "$NVME"/models:"$NVME"/models:ro -v "$NVME"/saber-packs:"$NVME"/saber-packs:ro \
  -v "$NVME"/saber-kv:"$NVME"/saber-kv -v "$REPO":/work:ro -w /work --entrypoint /bin/bash "$IMG" \
  -c "ulimit -c 0; exec ${SERVE_PREFIX:-} $SABER engine --pack $PACK --fn-trunk $FN_TRUNK --fn-kv-dtype $FN_KV --fn-vram-elastic-kv-max-gib $FN_ELASTIC_KV_MAX --fn-experts $FN_EXPERTS ${PROMPT_IDS_ARG} --tp 2 \
      --hip-lib /opt/rocm/lib/libamdhip64.so.7 --prebuilt /work/prebuilt/gfx1201 --fn-out $OUT/run \
      --fn-serve $PORT --fn-thinking on $REC_ARGS ${ARGS[*]}" >/dev/null || { echo "$CTR run failed" >&2; exit 1; }
echo "== $(date '+%T') booting $CNAME on :$PORT (the pack loads from NVMe; usually 1-3 min)"
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
echo "== $(date '+%T') serving after ${i} s: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1  (model id: see /v1/models)"
curl -s "localhost:$PORT/v1/models" | head -c 400; echo
echo "== a later exit: \`$CTR wait $CNAME\` prints the server's exit code (86 = a rank failed; the log's last line names it)"
