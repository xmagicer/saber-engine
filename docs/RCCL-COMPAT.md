# RCCL compatibility for TP=2 without GPU peer access

Use this opt-in backend when HIP reports no peer access between the two R9700s.
Tensor parallelism does not require GPU P2P: vLLM can use RCCL shared host memory,
while Saber's native all-reduce kernels directly address peer VRAM. A TP=2 vLLM
service working on the same host therefore does not prove native P2P works.

The native backend remains the default. `--engine-rccl-lib PATH` selects RCCL for
`serve`; `--rccl-lib PATH` is its equivalent for the engine/worker commands. The
27B launcher accepts `RCCL_LIB=PATH` and selects `NCCL_P2P_DISABLE=1`,
`NCCL_SHM_DISABLE=0`, `NCCL_PROTO=Simple` inside the runtime container.

## Requirements and behavior

- Linux x86_64, two gfx1201 R9700s, ROCm 7.x, the usual KFD/DRI permissions and TP=2 packs.
- The runtime image must contain RCCL >=2.30.4 and its matching HIP runtime.
  The public build image alone is not evidence of a compatible RCCL installation.
  Build and runtime images can differ because RCCL is loaded dynamically. The
  serving image supplies HIP/RCCL only; inference runs this branch's Rust Saber binary.
- RCCL uses its validated v2.30.4 config ABI: blocking initialization and
  `graphUsageMode=2`, permitting eager prefill and captured decode together.
- The profile uses BF16 sum collectives, disables native peer-memory all-reduce fusion, retains local decode fusions and
  selects Saber's existing host candidate exchange. It never opens peer VRAM or
  pretends that peer access succeeded. P2P race diagnostics and dispatch-record
  replay are refused explicitly.
- Native communicator memory is warmed and charged to the VRAM ledger before the
  KV pool is sized. The HIP thread owns communicator destruction after queued
  graph destruction; a borrowed communicator token cannot migrate to another HIP thread.
- A captured RCCL call contributes one kernel and one completion-event node in
  the validated runtime. Kernel and synchronization budgets are checked separately,
  including node kinds. An unexpected library graph shape refuses startup.

## Build, download and launch

```bash
git clone --branch codex/rccl-compat https://github.com/xmagicer/saber-engine.git
cd saber-engine
export SABER_CONTAINER=docker
# One-time compiler image; image tag defaults to localhost/saber-dev:latest.
tools/build-image.sh
export SABER_PACKS="$HOME/models/saber-packs"
# Installs the TP=2 target/drafter packs and tokenizer/template. About 26.5 GB.
tools/fetch_packs.sh 27b

# Build separately if your serving image does not have cargo.
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD":/work -w /work \
  -e CARGO_TARGET_DIR=/work/.cache/target-container \
  --entrypoint bash localhost/saber-dev:latest \
  -c 'cargo build --locked --release -p saber-cli'

# This immutable runtime was tested: HIP 7.14.60850 and RCCL 2.30.4.
export SABER_IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
export RCCL_LIB=/opt/rocm/lib/librccl.so
# Bind to loopback behind an authenticated gateway. Omit HOST to bind all interfaces.
HOST=127.0.0.1 PORT=18080 MODEL=qwen3.8-27b-mxfp4 \
  tools/serve_27b.sh --tp 2 --no-build --enable-prefix-caching on
curl -fsS http://127.0.0.1:18080/v1/models
# Stop with the same container name/profile; restore your previous launch command to roll back.
tools/serve_27b.sh --tp 2 stop
```

Stop the existing model manager/container before the real GPU trial. Keep the
previous service configuration and weights for rollback. After raw HTTP tests,
change the gateway's launch/stop commands to this profile, preserve its model ID,
authentication and idle policy, then test gateway requests and idle unload/reload.
No kernel, BIOS or bootloader change is needed for this route.

## Reproduce the collective probe

`tools/probes/rccl_no_p2p.hip` uses two processes and a temporary bootstrap ID file.
Compile/run inside the same GPU-enabled runtime image, with shared IPC and both
GPUs exposed. Set `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_PROTO=Simple`:

```bash
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_PROTO=Simple
hipcc --offload-arch=gfx1201 -O2 tools/probes/rccl_no_p2p.hip -lrccl -o /tmp/rccl-probe
# Use a new private temporary directory for each run; start both ranks together.
probe_dir=$(mktemp -d)
/tmp/rccl-probe 0 "$probe_dir/id" >"$probe_dir/rank0.log" 2>&1 & rank0=$!
/tmp/rccl-probe 1 "$probe_dir/id" >"$probe_dir/rank1.log" 2>&1 & rank1=$!
wait "$rank0"; status0=$?
wait "$rank1"; status1=$?
cat "$probe_dir/rank0.log" "$probe_dir/rank1.log"
test "$status0" = 0 && test "$status1" = 0
```

The probe checks eager and graph BF16 sums of 5,120, 131,072 and 4,194,304 elements
against an independent float/RNE reference, using 20 different inputs per case.
Its graph alternates 16 collectives with unrelated kernels; this catches the
completion events that an isolated-collective capture misses. It also verifies a
16-KiB/rank byte AllGather. Both ranks must report `all_passed: true`.

## Paired HTTP measurements

`tools/bench/paired_http.py` prepares and hashes fixed request bodies once, then
replays the identical files against each deployment. Preserve the served model
alias, GPU power caps and CPU profile. Finish compilation/downloads before timing.
Prepare using the original server's `/tokenize` endpoint:

```bash
python3 tools/bench/paired_http.py --prepare --payload-dir /tmp/paired-requests
python3 tools/bench/paired_http.py --payload-dir /tmp/paired-requests \
  --output-dir /tmp/baseline --label baseline
# Switch backend, warm and verify it, then use the same payload directory:
python3 tools/bench/paired_http.py --payload-dir /tmp/paired-requests \
  --output-dir /tmp/rccl --label saber-rccl
```

The default suite is 57 requests in 24 batches: three repeats of single-stream
code/prose, concurrency 2/4/8, and roughly 16K/64K/128K prompts. It requests exact
1,024 output tokens for short workloads and 512 for long ones with thinking off,
temperature zero and `ignore_eos`. Persisted SSE and usage distinguish tokens from
chunks. Decode rate is approximate (usage tokens divided by first-to-last text
chunk time); concurrent throughput includes end-to-end latency. Report the arithmetic mean, median and highest complete-request decode rate across the three repeats of each workload, alongside actual token counts. Also test retrieval near the
advertised context limit separately; startup alone does not verify that limit.

For a within-request peak, run both arms with `--stream-timing`. The client adds `return_token_ids: true` to the transmitted payload and records its SHA-256 alongside the saved payload hash. It counts generated token IDs as SSE frames arrive and reports the busiest complete one-second window in tok/s. The per-request timing record lists its frame counts and the window boundaries. Missing step IDs, mismatched final usage, or less than one second of ID arrivals leave this peak unavailable. To recover whole-request statistics from an existing run, use `--summarize-existing RESULT_DIR --payload-dir PAYLOAD_DIR`.

The old Quark MXFP4 checkpoint and Saber NVFP4-to-MXFP4 packs have different
quantization pipelines. These are deployment comparisons, not an isolated measure
of engine speed or a proof of equal model quality.

## Validation scope

The added compatibility checks plus existing HIP, execution and server tests pass
317 tests (one existing documentation test ignored). The native descriptor path
is included in these regressions. Native P2P GPU execution cannot be tested on a
host whose chipset rejects it; it remains a separate hardware validation item.

## Measured queue and fusion regression

On this non-P2P R9700 pair, the launcher now defaults to one HIP hardware queue
for RCCL (`GPU_MAX_HW_QUEUES=1`); explicit overrides remain available. The native
backend keeps its existing three-queue default. In A/B/A tests with identical
six request bodies, three queues produced about 118/92 tokens/s for code/prose,
one queue 163/127, and returning to three queues reproduced 118/92. All generated
content bytes matched across those three runs. Local GEMM-pair and rope/append
fusion increased the one-queue medians to about 167/129. These are approximate
decode rates, not total concurrent throughput.

Native AR-coupled fusions are explicitly refused in the planner for RCCL. Ordinary
local fusions retain their configured setting; `--engine-no-decode-fusion` remains
the unfused control. A plan regression checks both ranks, verifies that all 129
collectives remain external, and that local fusions survive.

Reducing the drafter block from eight to four made the tested workloads slower;
it is not the default. A separate-communicator experiment using mode 1 was also
rejected by real model acceptance and is not included. NCCL's mode 1 is a single
graph hint, while this engine retains multiple bucket graphs; mixed mode 2 is the
validated contract. See [NCCL graph usage types](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html).
