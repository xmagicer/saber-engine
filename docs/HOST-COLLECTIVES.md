# Experimental native collectives through shared host memory

`--engine-host-ar` selects an explicit TP=2 backend for Linux hosts whose GPUs
cannot access peer VRAM. It registers private shared host pages on each owning
HIP thread and uses exact BF16 native collective kernels. It preserves local
GEMM/reduction/normalization fusion, which the separate RCCL profile cannot use.
No kernel, BIOS, IOMMU or bootloader modification is required.

The default remains native peer-VRAM communication. The
[RCCL profile](RCCL-COMPAT.md) is the independently validated fallback.

## Launch

Build the branch and download the TP=2 target and DFlash2 packs following the
RCCL guide. Stop the existing model service before using the GPUs. Select a
runtime image with the tested HIP ABI; RCCL itself is not used in this mode.

```bash
export SABER_CONTAINER=docker
export SABER_PACKS="$HOME/models/saber-packs"
export SABER_IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
unset RCCL_LIB
export GPU_MAX_HW_QUEUES=1
HOST=127.0.0.1 PORT=18080 MODEL=qwen3.8-27b-mxfp4 \
  tools/serve_27b.sh --tp 2 --no-build --enable-prefix-caching on --engine-host-ar
```

The equivalent engine/worker option is `--host-ar`. TP=1 is refused. Native P2P
remains the default when neither option is supplied. Keep the previous runtime,
weights and launch configuration for rollback.

An additional experimental hybrid profile combines `--engine-host-ar` with
`--engine-rccl-lib`: messages smaller than 65 target hidden-state rows use the
native host backend, and larger messages use exact BF16 RCCL. With the launcher,
keep `RCCL_LIB=/opt/rocm/lib/librccl.so` set and add `--engine-host-ar`. This also
preserves the launcher's tested RCCL environment. Small decode messages retain
their fused epilogues; bulk prefill messages do not use native host scratch.
The threshold is resolved from the model's hidden size, not an environment
variable. The hybrid passed actual model, long-context, concurrent-request and
gateway/lifecycle acceptance after the stream-lifetime fix described below.

RCCL warmup uses the model's long-lived stream. RCCL remembers the previous
eager stream when it records completion events on a later call. Destroying a
temporary warmup stream caused an actual first-request crash in
`ncclLaunchPrepare -> hipEventRecord -> pthread_mutex_lock` when native decode
graphs did not update that remembered stream. Warming on the model stream
removed this failure; keeping only the communicator alive was insufficient.

## Memory and synchronization

Both ranks map the same private, mode-0600 files and register the mappings with
HIP as portable, mapped host memory. The bootstrap exchanges paths and confirms
both mappings before unlinking the names. Ownership, size and mode are checked;
symlinks are refused. HIP device addresses come from `hipHostGetDevicePointer`.
These are host allocations, not peer VRAM or emulated successful peer access.

The host-pinned reserve is charged before the device ledger sizes its KV pool.
The tested full profile registers about 1008 MiB of shared views per process;
these views refer to shared physical pages. Teardown unregisters on the owning
HIP thread before unmapping. A failed unregister retains the mapping to avoid
releasing pages still referenced by the driver.

The host-specific code objects cap polling at one million iterations. The
normal native objects retain their original polling configuration. The tested
fine-grained mapping uses the existing store-drain-3/acquire-0 protocol with
system-scope release/acquire flag operations. Checked kernels additionally
validate hashes, sequence counters and timeouts. Use `--engine-race-check`
for diagnostics; report performance from an unchecked serve.

## Measurements and current limits

On the non-P2P dual-R9700 test host, at 250 W per card with its existing CPU power
policy, the identical six saved single-stream requests gave these approximate
decode medians. Each request produces 1024 tokens, thinking off, temperature
zero; three saved code prompts and three prose prompts are replayed per profile.

| Profile | Code tokens/s | Prose tokens/s |
|---|---:|---:|
| Original GGZ14/vllm-mxfp4 deployment | 184.5 | 135.8 |
| Saber RCCL, one queue and local fusions | 164.8 | 128.4 |
| Host-native, ancestor9, full paired suite | 195.7 | 153.1 |
| Host-native, matched Quark core control | 216.4 | 147.2 |

The Quark control preserves the original checkpoint's core matrices and matching
normalization/convolution parameters, but retains the pack's FP8 target head.
The old vLLM deployment uses a BF16 head. It is not an identical full checkpoint.
See [the conversion helper](../tools/weights/README.md). The Quark variant helps
code and hurts prose, so it is not a universal replacement.

All 57 requests in the full paired suite passed, as did raw chat/tool/stream/
thinking checks and retrieval at 245762 prompt tokens. Aggregate throughput was
245.3/400.5/530.7 tokens/s at concurrency 2/4/8. A separate short-only run gave
200.2/156.4 tokens/s; the table uses the full-suite medians. The full native-host
profile's 128K cold TTFT was 73.9 s versus 61.1 s for the old vLLM deployment,
which motivates the hybrid bulk-transfer path. The pure host profile was not
promoted; the deployed hybrid's gateway/lifecycle acceptance is described below.
The 300/200 tokens/s code/prose targets have not been achieved by these measurements.

These results were collected under the night policy at 250 W per GPU. At 08:00
the host automatically changed to its day policy (300 W per GPU, CPU turbo
enabled). Fresh measurements for both backends must use that same policy;
numbers from different power policies must not be combined as an engine gain.

## Validated hybrid deployment

A fresh full comparison under the unchanged **day policy, 300 W per GPU**, used
the same 57 saved request bodies for each backend:

| Metric | GGZ14/vllm-mxfp4 | Saber hybrid |
|---|---:|---:|
| Single-stream code, tokens/s | 185.36 | 201.01 |
| Single-stream prose, tokens/s | 136.47 | 158.51 |
| Concurrency-8 aggregate, tokens/s | 500.98 | 546.37 |
| 128K cold TTFT, seconds | 60.88 | 61.69 |
| Managed model startup, seconds | 123.08 | 12.54 |

The code/prose targets of 300/200 tokens/s remain unmet. The hybrid improves the
tested decode medians by about 8.4%/16.2% and aggregate concurrency-8 throughput
by 9.1%. Its 128K prefill is about 1.3% slower than the old deployment; it removes
most of the pure host backend's prefill regression rather than claiming a
prefill speedup.

All 57 requests, retrieval at 245762 prompt tokens, authenticated gateway chat,
tools, streaming, thinking and multi-turn tests passed. Cancellation drained in
0.022 s. A real 60-second idle test unloaded automatically, both GPUs suspended,
and reload took 14.55 s; the original 7200-second policy was restored. Observed
container shutdowns exited successfully. No kernel or boot configuration changed.

`--engine-spec-tree best16` enables the 16-row ancestor verifier for one active
sequence. Its state advance uses the 16-row record layout. The target can also
plan a six-wave uniform causal kernel at exactly 16 queries for GQA 6, head size
256 and a 16-token page. Chain and tree captures have separate identities on
both TP ranks. Worker wire version 17 requires matched binaries.

The experimental DSpine socket backend consumes verified target features and
returns chain or 15-node proposals. In the tested matched-Quark configuration
the optimized tree measured 203.91 tok/s for code and 137.57 tok/s for prose.
The standard NVFP4 + DFlash2 hybrid remains the deployed configuration.
[See the DSpine launch procedure and feature limits](../tools/dspine/README.md).

## Tandem B16 candidate calibration

With the matched Quark Qwen3.8-27B core and a resident Tandem B16 drafter,
`--engine-selector-rank tandem-b16-calibrated` scores all 16 conditional
candidates before selecting three children per parent for the nine-node tree.
Use TP=2, one sequence, decode batches `1`, widths `1,10`, draft block 16,
`--engine-spec-tree ancestor9` and `--engine-gdn-replay off`.
Both ranks check the packed target and draft identities on startup.
The default selector is `walk`. The target verifies every proposed path.
