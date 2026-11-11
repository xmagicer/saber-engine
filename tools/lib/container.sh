#!/usr/bin/env bash
# Container-runtime shim shared by the launchers: podman or docker, same scripts.
#
#   . "$REPO/tools/lib/container.sh"     # sets CTR, CTR_GPU[], CTR_USER[]
#   "$CTR" run "${CTR_GPU[@]}" "${CTR_USER[@]}" ... image
#
# Pick the runtime with SABER_CONTAINER=podman|docker (default: podman if installed, else docker).
#   CTR_GPU   the GPU device nodes and the groups that may open them
#             podman: `--group-add keep-groups` (rootless podman passes the caller's groups through)
#             docker: the numeric gids of /dev/kfd and /dev/dri/render* (docker has no keep-groups)
#   CTR_USER  docker only (empty under rootless podman), the per-run options docker needs:
#             `--user UID:GID` so files written into the bind-mounted repo (.cache, build output)
#             stay yours instead of root's (rootless podman's container root already maps to you);
#             `--ulimit nofile=1048576:1048576`, because docker's default open-file limit is 1024
#             and the engine / pack builders hold every checkpoint shard and pack file open
#             ("too many open files" otherwise). Not passed to podman: rootless podman already
#             inherits a high limit, and asking above the host hard limit fails the run.
#             Every `$CTR run` that opens shards, checkpoints or packs expands "${CTR_USER[@]}"
#             (dev.sh, hipcc.sh, build_packs.sh, serve_27b.sh, serve_flashnext.sh).
if [ -z "${SABER_CONTAINER:-}" ]; then
  if command -v podman >/dev/null 2>&1; then SABER_CONTAINER=podman
  elif command -v docker >/dev/null 2>&1; then SABER_CONTAINER=docker
  else echo "saber: neither podman nor docker is installed (set SABER_CONTAINER=<runtime>)" >&2; exit 1; fi
fi
command -v "$SABER_CONTAINER" >/dev/null 2>&1 || { echo "saber: container runtime '$SABER_CONTAINER' not found" >&2; exit 1; }
CTR=$SABER_CONTAINER
CTR_GPU=(--device /dev/kfd --device /dev/dri)
CTR_USER=()
if [ "$("$CTR" --version 2>/dev/null | head -c 6)" = podman ] || [ "$CTR" = podman ]; then
  CTR_GPU+=(--group-add keep-groups)
else
  _g=$(stat -c %g /dev/kfd 2>/dev/null) && CTR_GPU+=(--group-add "$_g")
  for _n in /dev/dri/renderD* /dev/dri/card*; do
    [ -e "$_n" ] || continue
    _g=$(stat -c %g "$_n"); case " ${CTR_GPU[*]} " in *" $_g "*) ;; *) CTR_GPU+=(--group-add "$_g") ;; esac
  done
  CTR_USER=(--user "$(id -u):$(id -g)" -e HOME=/tmp --ulimit nofile=1048576:1048576)
fi
unset _g _n
