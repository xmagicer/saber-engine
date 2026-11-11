#!/usr/bin/env bash
# FN71c: the podman arguments that hand a container each card's amdgpu P2PDMA pool (FN71's
# `amdgpu.p2pmem_mb`): sysfs is read-only inside a container, so every published pool's
# `p2pmem/allocate` (rw for the user by FN71b's udev rule) is bind-mounted rw at
# /saber-p2p/<bdf> -- the engine's `--xtier-p2p-dir` default. No pool on this boot (size 0 or no
# p2pmem dir) = no argument: the engine's boot fill then runs the host route by name.
#   . tools/capture/p2p_mounts.sh; P2P=(); p2p_mounts P2P; podman run ... "${P2P[@]}" ...
p2p_mounts() {
  local -n _out=$1
  local d bdf
  _out=()
  for d in /sys/bus/pci/devices/*/p2pmem; do
    [ -d "$d" ] || continue
    [ "$(cat "$d/published" 2>/dev/null || echo 0)" = 1 ] || continue
    [ "$(cat "$d/size" 2>/dev/null || echo 0)" != 0 ] || continue
    [ -r "$d/allocate" ] && [ -w "$d/allocate" ] || continue
    bdf=$(basename "$(dirname "$d")")
    _out+=(--mount "type=bind,src=$d/allocate,dst=/saber-p2p/$bdf,rw=true")
  done
}
