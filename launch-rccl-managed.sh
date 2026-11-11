#!/usr/bin/env bash
set -euo pipefail
cd /home/dev/Work/saber-migration-20261108/engine-compat/saber-source
export SABER_CONTAINER=docker
export GPU_MAX_HW_QUEUES=1
export TIER=qwen38-27b-quark-matched-core-fp8head
export DRAFT=qwen38-dflash2-tandem-b16-fp8head
export SABER_BIN=.cache/target-container/release/saber-merged-rccl-ae5e62ed879c
export SABER_IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
export SABER_PACKS=/home/dev/models/saber-packs
export SABER_SHARED=/home/dev/.cache/saber-rccl-runtime
export HOST=127.0.0.1 RCCL_LIB=/opt/rocm/lib/librccl.so
export PORT=18080 MODEL=qwen3.8-27b-mxfp4 HIP_DEVICES=0,1 MAX_CTX=262144 MAX_SEQS=8 CNAME=saber-rccl-serve
cleanup(){ docker stop -t 60 "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
tools/serve_27b.sh --tp 2 --no-build --enable-prefix-caching on --engine-host-ar
rc="$(docker wait "$CNAME")"
[[ "$rc" =~ ^[0-9]+$ ]] || exit 1
exit "$rc"
