#!/usr/bin/env bash
# Download the ready-made weight packs from Hugging Face, so a fresh clone can serve without
# converting a checkpoint (the CPU repack of tools/pack/build_devfinal.sh).
#
#   tools/fetch_packs.sh 27b          # ~26 GB: the 27B target pack + the DFlash2 drafter pack,
#                                     #   and the tokenizer / chat template into .cache/
#   tools/fetch_packs.sh 27b-tp1      # ~25 GB: the SINGLE-CARD (TP=1) 27B target + drafter packs
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
CKREPO=${SABER_FN_CHECKPOINT_REPO:-amd/Qwen3.8-Flash-Next-Quark-MXFP4}

HF=$(command -v hf || command -v huggingface-cli || true)
[ -n "$HF" ] || { echo "needs the Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2; exit 1; }

want=${1:-}
case "$want" in 27b | 27b-tp1 | flashnext | all) ;; *) echo "usage: $0 27b|27b-tp1|flashnext|all" >&2; exit 2 ;; esac

# The 27B rank files use the SBRPACK1 container (docs/abi/PACK.md).
check_packs() {
  local f bad=0
  for f in "$@"; do
    [ -f "$f" ] && [ "$(head -c 8 "$f")" = "SBRPACK1" ] || { echo "bad or missing pack: $f" >&2; bad=1; }
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

if [ "$want" = flashnext ] || [ "$want" = all ]; then
  echo "== $RFN -> $PACKS"
  "$HF" download "$RFN" --local-dir "$PACKS"
  P=$PACKS/qwen38-flashnext-quark-trunk8
  # Mixed formats: the trunk, MTP and HC files are raw indexed payloads. The
  # TP experts and vision use different container headers.
  python3 "$REPO/tools/verify_flashnext_packs.py" "$P"
  # The Flash-Next pack is NOT self-contained: its manifest points into the original checkpoint
  # (embeddings, norms, router, and the 102 GB PLE table are read from its shards in place), at the
  # absolute path the manifest names. Download the checkpoint exactly there.
  CK=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"]["dir"])' "$P/manifest.json")
  echo "== $CKREPO -> $CK (~170 GB; resumable)"
  mkdir -p "$CK"
  "$HF" download "$CKREPO" --local-dir "$CK"
  python3 "$REPO/tools/verify_flashnext_packs.py" "$P" --checkpoint "$CK"
  echo "Flash-Next ready: tools/serve_flashnext.sh"
fi
