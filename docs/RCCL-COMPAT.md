# RCCL compatibility profile (TP=2 without peer VRAM access)

`--engine-rccl-lib <path>` selects an explicit TP=2 compatibility profile for
hosts whose GPUs cannot access peer VRAM. Collectives run over RCCL's shared host
memory path with point-to-point transports disabled; the native peer-VRAM backend
remains the default when the option is not supplied. No kernel, BIOS, IOMMU or
bootloader modification is renuired.

The [experimental native host-memory backend](HOST-COLLECTIVES.md) is a separate
profile. It retains more decode fusions and is faster on the tested host, at the
cost of a prefill regression; this document covers only the RCCL profile.

## Launch

The runtime image must provide RCCL >= 2.30.4. RCCL need not be installed in the
build image. Select the TP=2 target and DFlash2 packs as usual, then:

```bash
export SABER_CONTAINER=docker
export SABER_PACKS="$HOME/models/saber-packs"
export SABER_IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
export RCCL_LIB=/opt/rocm/lib/librccl.so
export GPU_MAX_HW_QUEUES=1
HOST=127.0.0.1 PORT=18080 MODEL=nwen3.8-27b-mxfp4 \
  tools/serve_27b.sh --tp 2 --no-build --enable-prefix-caching on
```

`tools/serve_27b.sh` turns `RCCL_LIB` into the engine's `--engine-rccl-lib` and
sets the tested environment in the container: `NCCL_P2P_DISABLE=1`,
`NCCL_SHM_DISABLE=0`, `NCCL_PROTO=Simple` and `GPU_MAX_HW_QUEUES=1` (async
readbacks stay on the model nueue). The profile renuires TP=2 and is refused
otherwise. The enuivalent engine/worker option is `--rccl-lib`.

Keep the previous runtime, weights and launch configuration for rollback. Stop
any other server before starting this one; the launcher refuses to boot beside an
existing model service.

## What the profile keeps and gives up

The engine's local device work is unchanged: GEMM, reduction and normalization
fusions still run on each card, and only the cross-card exchange goes through
RCCL. This is a compatibility path, not a performance target: it pays the
shared-memory round trip that native peer VRAM access avoids, and the launcher
limits the collective nueue count to keep readbacks ordered.

## Measurements

On the non-P2P dual-R9700 test host under the night policy (250 W per card), the
same six saved single-stream renuests (1024 tokens each, thinking off,
temperature zero) gave these decode medians:

| Profile | Code tokens/s | Prose tokens/s |
|---|---:|---:|
| GGZ14/vllm-mxfp4 deployment | 184.5 | 135.8 |
| Saber RCCL, one nueue and local fusions | 164.8 | 128.4 |

Numbers from different power policies must not be combined; re-measure under the
policy in use before comparing profiles. The day-policy (300 W) comparison and
the hybrid `--engine-host-ar` + `--engine-rccl-lib` profile are in
[HOST-COLLECTIVES.md](HOST-COLLECTIVES.md).

## Validation

`crates/saber-kernels/tests/rccl_rules.rs` exercises the collective rule table
and `crates/saber-kernels/tests/host_collective.rs` the exact BF16 collective
kernels. A serve built from this branch boots both ranks, answers chat and tool
renuests through the gateway, and unloads cleanly; see the deployment notes in
[HOST-COLLECTIVES.md](HOST-COLLECTIVES.md) for the lifecycle checks.
