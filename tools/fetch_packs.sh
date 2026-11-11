#!/usr/bin/env bash
# Download the ready-made weight packs from Hugging Face, so a fresh clone can serve without
# converting a checkpoint (the CPU repack of tools/pack/build_devfinal.sh).
#
#   tools/fetch_packs.sh 27b          # ~26 GB: the 27B target pack + the DFlash2 drafter pack,
#                                     #   and the tokenizer / chat template into .cache/
#   tools/fetch_packs.sh 27b-tp1      # ~25 GB: the SINGLE-CARD (TP=1) 27B target + drafter packs
#   tools/fetch_packs.sh 27b-quark    # ~29 GB: the Quark AWQ MXFP4 27B target + its paired drafter (TP=2)
#   tools/fetch_packs.sh 27b-quark-tp1  # ~28 GB: the same for ONE card (TP=1)
#   tools/fetch_packs.sh flashnext    # ~70 GB pack + the ~170 GB original checkpoint it points into
#   tools/fetch_packs.sh all
#
# Packs land in $SABER_PACKS (default /mnt/nvme/saber-packs), the directory the launchers read.
# The download resumes if interrupted. Env:
#   SABER_PACKS     destination directory
#   SABER_HF_OWNER  Hugging Face account that hosts the packs (default Launch80)
#   SABER_FN_CHECKPOINT_REPO  Flash-Next's original checkpoint (default amd/Qwen3.8-Flash-Next-Quark-MXFP4)
#   HF_TOKEN        only needed while a repo is private
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PACKS=${SABER_PACKS:-/mnt/nvme/saber-packs}
OWNER=${SABER_HF_OWNER:-Launch80}
R27=$OWNER/Qwen3.8-27B-NVFP4-SABER
RFN=$OWNER/Qwen3.8-Flash-Next-SABER
RQK=$OWNER/Qwen3.8-27B-Quark-AWQ-MXFP4-SABER
CKREPO=${SABER_FN_CHECKPOINT_REPO:-amd/Qwen3.8-Flash-Next-Quark-MXFP4}

HF=$(command -v hf || command -v huggingface-cli || true)
[ -n "$HF" ] || { echo "needs the Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2; exit 1; }

want=${1:-}
case "$want" in 27b | 27b-tp1 | 27b-quark | 27b-quark-tp1 | flashnext | all) ;; *) echo "usage: $0 27b|27b-tp1|27b-quark|27b-quark-tp1|flashnext|all" >&2; exit 2 ;; esac

# A pack the engine would refuse fails here, by name, instead of mid-boot: every rank file
# must start with the SBRPACK1 magic (docs/abi/PACK.md).
check_packs() {
  local f bad=0
  for f in "$@"; do
    [ -f "$f" ] && [ "$(head -c 8 "$f")" = "SBRPACK1" ] || { echo "bad or missing pack: $f" >&2; bad=1; }
  done
  [ "$bad" = 0 ]
}

# The Flash-Next trunk8 / mtp_fp8 / mtp_x4 / hc8 files are raw record blobs, NOT SBRPACK1 files: they have no
# magic, and the engine reads them through the sibling <name>.json, whose "bytes" is the blob's size. So check
# that the index exists and the size matches (a truncated download fails here). dense.pack and vision/rank*.pack
# in the same repo are real SBRPACK1 files and keep the magic check above.
check_blobs() {
  local f j want have bad=0
  for f in "$@"; do
    j=${f%.pack}.json
    if [ ! -f "$f" ] || [ ! -f "$j" ]; then echo "bad or missing blob: $f (needs $j)" >&2; bad=1; continue; fi
    want=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["bytes"])' "$j" 2>/dev/null) || want=
    have=$(wc -c < "$f" | tr -d ' ')
    [ -n "$want" ] && [ "$want" = "$have" ] || { echo "bad blob size: $f is $have bytes, $j says ${want:-?}" >&2; bad=1; }
  done
  [ "$bad" = 0 ]
}

mkdir -p "$PACKS"

if [ "$want" = 27b ] || [ "$want" = all ]; then
  echo "== $R27 -> $PACKS"
  "$HF" download "$R27" --local-dir "$PACKS" \
    --include 'qwen38-27b-nvfp4-devfinal-fp8head/*' --include 'qwen38-dflash2-drafter-fp8head/*'
  mkdir -p "$REPO/.cache"
  "$HF" download "$R27" --local-dir "$REPO/.cache" \
    --include 'template_corpus.tokenizer.json' --include 'template_corpus.jinja'
  check_packs "$PACKS"/qwen38-27b-nvfp4-devfinal-fp8head/rank{0,1}.pack \
              "$PACKS"/qwen38-dflash2-drafter-fp8head/rank{0,1}.pack
  echo "27B ready: tools/serve_27b.sh --enable-prefix-caching on"
fi

if [ "$want" = 27b-tp1 ]; then
  # A pack is cut for one TP degree (rows/columns are already sharded per rank), so the TP=1 packs
  # are separate folders from the TP=2 ones above. One rank file each: rank0.pack.
  echo "== $R27 (TP=1 packs) -> $PACKS"
  "$HF" download "$R27" --local-dir "$PACKS" \
    --include 'qwen38-27b-nvfp4-devfinal-tp1-fp8head/*' --include 'qwen38-dflash2-drafter-tp1-fp8head/*'
  mkdir -p "$REPO/.cache"
  "$HF" download "$R27" --local-dir "$REPO/.cache" \
    --include 'template_corpus.tokenizer.json' --include 'template_corpus.jinja'
  check_packs "$PACKS"/qwen38-27b-nvfp4-devfinal-tp1-fp8head/rank0.pack \
              "$PACKS"/qwen38-dflash2-drafter-tp1-fp8head/rank0.pack
  echo "27B TP=1 packs ready: tools/serve_27b.sh --tp 1 --enable-prefix-caching on"
fi

if [ "$want" = 27b-quark ] || [ "$want" = 27b-quark-tp1 ]; then
  # amd/Qwen3.8-27B-Quark-AWQ-MXFP4 packed for Saber: the target and a drafter pack whose head is
  # THIS checkpoint's lm_head (the engine refuses a drafter whose head differs from the target's).
  T=""; [ "$want" = 27b-quark-tp1 ] && T=-tp1
  echo "== $RQK${T:+ (TP=1 packs)} -> $PACKS"
  "$HF" download "$RQK" --local-dir "$PACKS" \
    --include "qwen38-27b-quark-awq-mxfp4$T/*" --include "qwen38-dflash2-drafter-quarkhead$T/*"
  # The checkpoint's own tokenizer (it differs from the NVFP4 one); the chat template is the same file.
  mkdir -p "$REPO/.cache/quark27b-dl"
  "$HF" download "$RQK" --local-dir "$REPO/.cache/quark27b-dl" --include 'tokenizer.json' --include 'qwen-fixed-v22.3.jinja'
  cp -f "$REPO/.cache/quark27b-dl/tokenizer.json" "$REPO/.cache/quark27b.tokenizer.json"
  [ -f "$REPO/.cache/template_corpus.jinja" ] || cp -f "$REPO/.cache/quark27b-dl/qwen-fixed-v22.3.jinja" "$REPO/.cache/template_corpus.jinja"
  if [ -n "$T" ]; then
    check_packs "$PACKS"/qwen38-27b-quark-awq-mxfp4-tp1/rank0.pack "$PACKS"/qwen38-dflash2-drafter-quarkhead-tp1/rank0.pack
    echo "27B Quark TP=1 packs ready: tools/serve_27b.sh --weights quark --tp 1"
  else
    check_packs "$PACKS"/qwen38-27b-quark-awq-mxfp4/rank{0,1}.pack "$PACKS"/qwen38-dflash2-drafter-quarkhead/rank{0,1}.pack
    echo "27B Quark packs ready: tools/serve_27b.sh --weights quark"
  fi
fi

if [ "$want" = flashnext ] || [ "$want" = all ]; then
  echo "== $RFN -> $PACKS"
  "$HF" download "$RFN" --local-dir "$PACKS"
  P=$PACKS/qwen38-flashnext-quark-trunk8
  check_blobs "$P"/rank{0,1}/trunk8.pack "$P"/rank{0,1}/mtp_fp8.pack "$P"/rank{0,1}/mtp_x4.pack \
              "$P"/shared/hc8.pack
  check_packs "$P"/rank{0,1}/dense.pack "$P"/shared/dense.pack "$P"/vision/rank{0,1}.pack
  [ -s "$P/rank0/experts.tp2.xtier" ] && [ -s "$P/rank1/experts.tp2.xtier" ] \
    || { echo "missing rank*/experts.tp2.xtier under $P" >&2; exit 1; }
  # The Flash-Next pack is NOT self-contained: its manifest points into the original checkpoint
  # (embeddings, norms, router, and the 102 GB PLE table are read from its shards in place), at the
  # absolute path the manifest names. Download the checkpoint exactly there.
  CK=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"]["dir"])' "$P/manifest.json")
  echo "== $CKREPO -> $CK (~170 GB; resumable)"
  mkdir -p "$CK"
  "$HF" download "$CKREPO" --local-dir "$CK"
  [ -f "$CK/model-00131-of-00131.safetensors" ] && [ -f "$CK/config.json" ] \
    || { echo "checkpoint incomplete under $CK" >&2; exit 1; }
  echo "Flash-Next ready: tools/serve_flashnext.sh"
fi
