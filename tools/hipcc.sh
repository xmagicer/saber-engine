#!/usr/bin/env bash
# Compile one HIP source to a gfx1201 code object inside the prod image (no host ROCm).
# Usage: tools/hipcc.sh <src.hip> <out.hsaco> [extra hipcc flags...]
set -euo pipefail
src=$1; out=$2; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
# An explicit compiler path lets saber-kbuild run inside an already isolated build container.
if [ -n "${SABER_HIPCC:-}" ]; then
  exec "$SABER_HIPCC" --genco --offload-arch=gfx1201 -O3 "$@" "$src" -o "$out"
fi
. "$REPO/tools/lib/container.sh"
IMG=${SABER_IMAGE:-localhost/saber-dev:latest}
# A WORKTREE'S `.cache` IS A SYMLINK, and a container has to be told what it points at.
# `saber-kbuild` writes its preprocessed source under `$REPO/.cache/kbuild`, which in a worktree is
# `/work/.cache -> ~/saber/.cache` and dangles inside the container unless the TARGET is
# mounted at the path the link names. A no-op in the main checkout (ws-s2's finding, the other half
# of ws-f27's).
CACHE=$(readlink -f "$REPO/.cache" 2>/dev/null || true)
MOUNT=()
[ -n "$CACHE" ] && [ -d "$CACHE" ] && [ "$CACHE" != "$REPO/.cache" ] && MOUNT=(-v "$CACHE":"$CACHE")
# label=disable, not :z -- the repo lives under a user_home_t path and SELinux is enforcing;
# relabelling the checkout would be a side effect on the user's tree, disabling the label for this
# one short-lived container is not.
exec $CTR run --rm "${CTR_USER[@]}" --security-opt label=disable -v "$REPO":/work "${MOUNT[@]}" -w /work \
  --entrypoint /opt/rocm/bin/hipcc "$IMG" \
  --genco --offload-arch=gfx1201 -O3 "$@" "$src" -o "$out"
