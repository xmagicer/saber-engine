# DSpine validation tools

This directory includes numerical reference tools and an experimental DSpine serving backend for the 27B target. The following capture and replay sections describe the offline tools. The final section gives the live TP=2 launch procedure.

- `reference.py` implements the four selected-prefix bridge operations and greedy serial refinement in PyTorch.
- `plain_model.py` loads the five-layer BF16 DSpine checkpoint, keeps only the trained 8192-token teacher window, and never persists speculative query KV. Ordinary backbone norms follow SGLang's default single final BF16 cast and FP32 residual sum; the bridge retains its separately verified two-round BF16 next-norm contract.
- `capture_reader.py` checks the committed-token journal against the actual API token stream. It trims internal terminal work beyond the requested output budget.
- `pack_io.py` reads target embeddings and the checkpoint-precision head from SBR packs.
- `hip_module.py` and `native_head.py` are GPU experiments. The int2 candidate filter is approximate and has not passed the drafter-quality adoption gate.
- `native_linear.py` experimentally replaces the draft's large projections with group-128 W8A8 code objects. This changes draft numerics and needs measured acceptance validation. BF16 bridge and serial-table math stay unchanged.
- `graph_model.py` captures fixed proposal shapes with padded teacher-cache staging. Invalid absolute positions mask padding; speculative query K/V never enters the committed cache. The group-128 model passed GPU comparisons with changing teacher histories, future queries and padding contents.

Run CPU checks with PyTorch installed:

```sh
python3 -m unittest discover -s tools/dspine -p 'test_*.py' -v
```

The opt-in target capture requires one sequence, prefix caching off and the chain verifier:

```sh
saber serve ... --engine-max-seqs 1 --enable-prefix-caching off --engine-spec-tree off \
  --engine-draft-capture-dir /private/new-directory \
  --engine-draft-capture-layers 2,17,32,47,62
```

The checkpoint's `[1,16,31,46,61]` are **zero-based decoder output** ids. Saber's `lN.resid.a` is the residual entering layer N, so the actual capture points are **`[2,17,32,47,62]`**. `target_layers.py` performs this mapping and rejects captures using the unshifted list. Do not apply this DSpine-specific convention to an older DFlash2 pack automatically. Capture uses an independently allocated plane so the resident DFlash2 proposal remains unchanged. Export directories/files use modes 0700/0600; existing directories are refused.

Verified on a real gfx1201 GPU: upstream Triton bridge parity with the BF16 checkpoint, and no-future-feature replay against captured target outputs. Historical eager timing preceded the corrected layer and normalization checks and is superseded. Acceptance experiments and isolated proposal timing must not be presented as end-to-end serving throughput.

Native group-128 W8A8 projections passed independent logical-matrix and varying-input graph checks. **Earlier acceptance measurements used the unshifted capture ids and are invalid as DSpine quality results.** The corrected replay values below use the shifted capture points. The live backend uses channel FP8 projections.

The channel-FP8 W8A16 variant is a separate numerical experiment. Its isolated projections passed matrix and graph checks, but its actual model comparison between padded graphs and the unpadded eager reference exceeded the current absolute-error gate. That graph variant has not passed acceptance; do not enable it based on the isolated matrix checks.

A same-power daytime comparison previously measured roughly 10.6 ms per proposal on one GPU, but used superseded feature inputs and normalization. Do not treat it as a correct deployment result.

With corrected layer mapping and SGLang backbone norm semantics, an offline reference replay on **one saved code and one prose trajectory** produced these mean committed tokens per round:

| Target / proposer | Chain, block 16 | 15-node tree, depth up to 15 |
|---|---:|---:|
| NVFP4 pack / BF16 dense head | 4.61 / 3.66 | 5.32 / 4.12 |
| Matched Quark core / BF16 dense head | 5.87 / 3.54 | 6.31 / 4.04 |
| Matched Quark core / channel-FP8, int2/80 candidates | 5.77 / 3.57 | 6.16 / 4.09 |

Each cell is code / prose. Tree construction reads only proposer probabilities; canonical future tokens are supplied afterward to an independent acceptance walk. `tree_proposal.py` checks normalized conditional probabilities, parent-before-child rows and 16-bit ancestor masks. The optional full transition output in `reference.serial_table` is a slow correctness path, not a serving implementation.

These corrected results measure acceptance on the two captured trajectories. The sections below cover isolated proposal timing and the live TP=2 experiment.


## Corrected GPU proposal experiments

`NativeChannelLinear(..., row_band=32)` selects the existing 32-row W8A16
kernel and bounds stores by the actual number of rows. The default remains 16.
On actual DSpine matrices at 8/16 rows, the 32-row form was about 1.6-2.0x
faster for the larger projections and bit-identical in eager and changing-input
graph tests. Whole proposals improved only about 3-5%; the 32-row int2 head was
slower and was rejected.

`triton_bridge.FusedBridge` can be assigned to `TorchDspine.bridge`. Its four
fused kernels retain BF16 bridge projections and the bridge-specific two-round
next-norm. Graph callers must retain the captured linear **and bridge** callback.
In 12 corrected-input cases (code/prose, three teacher lengths, blocks 8/16),
fusion saved approximately 0.8-1.0 ms per isolated proposal; final greedy drafts
matched the reference in these cases, although intermediate BF16 states differed.
Full saved-trajectory chain replay also retained reference acceptance for both
NVFP4 and matched Quark target packs. A more aggressive native BF16 projection
experiment slightly reduced Quark prose acceptance and was not selected.
These are one-trajectory diagnostics, not a claim of reaching 300/200 tok/s.

`triton_tail.py` computes serial greedy transitions, confidence and optionally
normalized conditional log-probabilities on the GPU. Tree construction receives
only those proposer outputs; target future tokens enter the acceptance walk
after the tree has been built. CPU `reference.py` remains the independent oracle.
Both physical gfx1201 GPUs passed changing-input graph checks at K=2/8/16/32;
transition log-probability maximum error was below 2.3e-5 in that gate, and greedy
indices/walks matched exactly.

Reproduce the primitive GPU checks (one process per physical GPU):

```sh
HIP_VISIBLE_DEVICES=0 python3 tools/dspine/gpu_bridge_gate.py --checkpoint /models/Qwen3.8-27B-DSpine-b16
HIP_VISIBLE_DEVICES=0 python3 tools/dspine/gpu_tail_gate.py --checkpoint /models/Qwen3.8-27B-DSpine-b16
# Repeat with HIP_VISIBLE_DEVICES=1; each process still uses logical cuda:0.
```

Replay a real capture with a fresh private evidence file:

```sh
python3 tools/dspine/offline_replay.py \
  --capture /evidence/correct-target-capture \
  --checkpoint /models/Qwen3.8-27B-DSpine-b16 \
  --target-pack /packs/qwen38-27b-nvfp4-devfinal-fp8head \
  --draft-pack /packs/qwen38-dflash2-drafter-fp8head \
  --prebuilt prebuilt/gfx1201 --projection channel --row-band 32 \
  --fused-bridge --block-size 16 --tree-nodes 15 --max-depth 15 \
  --output /evidence/new-tree-result.json
```

Use `--tree-nodes 0` for the chain control, omit `--fused-bridge` for the
unfused control, or select `--candidate-pool 0 --projection bf16` for the dense
BF16 control. The int2 pool and FP8 projections are experimental approximations;
target verification is unchanged. Saved output includes per-round proposals and
acceptance; it refuses overwriting an existing evidence file. No serving wrapper
or production model selection is changed by these utilities.

The standalone CLI passed six actual capture runs (NVFP4/Quark, chain control,
unfused tree and fused tree), each covering the saved code and prose trajectory.
The GPU tree results reproduced the CPU-reference means above, and bridge fusion
retained those means in both target variants. This validates the offline path;
it does not establish TP=2 serving performance or 300/200 tok/s.


## Live TP=2 experiment on two GPUs

Use an exclusive test window on the two cards. Keep the managed launcher and its
known-good binary available for restoration. Start in the repository after the
release build and pack verification described in the [host guide](../../docs/HOST-COLLECTIVES.zh-CN.md).
Download the draft weights with `hf download tencent/Qwen3.8-27B-DSpine-b16 --revision ea909c0f5fab8865f9e14600334f55f7ce47308f --local-dir "$HOME/models/Qwen3.8-27B-DSpine-b16"`. The tested target control used the matched Quark core pack and its existing FP8
head. The drafter checkpoint is `tencent/Qwen3.8-27B-DSpine-b16` at revision
`ea909c0f5fab8865f9e14600334f55f7ce47308f`.

```sh
export SABER_CONTAINER=docker
export SABER_IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
export SABER_PACKS="$HOME/models/saber-packs"
export TIER=qwen38-27b-quark-matched-core-fp8head
export SABER_SHARED="$HOME/.cache/dspine-test"
mkdir -p -m 700 "$SABER_SHARED"
export RCCL_LIB=/opt/rocm/lib/librccl.so GPU_MAX_HW_QUEUES=1

docker run -d --rm --name dspine-draft --device /dev/kfd --device /dev/dri \
  --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /dev/kfd)" --ipc=host \
  -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0,1 -e GPU_MAX_HW_QUEUES=1 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e TRITON_CACHE_DIR=/shared/triton-cache \
  -v "$PWD:/work:ro" -w /work -v "$SABER_SHARED:/shared" \
  -v "$SABER_PACKS:/packs:ro" -v "$HOME/models/Qwen3.8-27B-DSpine-b16:/checkpoint:ro" \
  --entrypoint python3 "$SABER_IMAGE" -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/dspine/tp_service.py --checkpoint /checkpoint --prebuilt /work/prebuilt/gfx1201 \
  --target-pack "/packs/$TIER" --draft-pack /packs/qwen38-dflash2-drafter-fp8head \
  --socket /shared/draft.sock --max-seqs 1 --graph --fused-projections --gpu-tree
```

Wait for the ready record in `docker logs dspine-draft`. Start the target with
the socket's host path, which the target launcher mounts at the same absolute path.

```sh
HOST=127.0.0.1 PORT=18081 CNAME=dspine-target MODEL=qwen3.8-27b-mxfp4 \
  MAX_CTX=262144 MAX_SEQS=1 BATCHES=1 WIDTHS=1,16 BUDGET_MIB=21504 \
  tools/serve_27b.sh --tp 2 --no-build --engine-host-ar --engine-ledger today \
  --enable-prefix-caching off --engine-gdn-replay off --engine-spec-tree best16 \
  --engine-dspine-socket "$SABER_SHARED/draft.sock"
curl -fsS http://127.0.0.1:18081/v1/models
```

The experimental socket path supports a single active sequence, with prefix
caching, forking and session park disabled. It uses a private mode-0600 Unix
socket, an 8192-token teacher KV window and a 262144-token target context limit.
For the chain control, select `--engine-spec-tree off`. For the width-8 control,
set `WIDTHS=1,8` and omit `--gpu-tree`. The JSON response and the shutdown tree
counters together verify a live request and actual tree selection. Stop the
experiment with `docker stop dspine-target dspine-draft` and restore the managed
launcher before checking gateway health.
