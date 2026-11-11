# DSpine validation tools

These are experimental numerical, pack-reading and offline replay utilities. They do not select a DSpine serving backend.

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

Native group-128 W8A8 projections passed independent logical-matrix and varying-input graph checks. **Earlier acceptance measurements used the unshifted capture ids and are invalid as DSpine quality results.** The low-level numerical/causality checks remain useful, but acceptance must be remeasured with correctly captured decoder outputs. This work has not been adopted for serving.

The channel-FP8 W8A16 variant is a separate numerical experiment. Its isolated projections passed matrix and graph checks, but its actual model comparison between padded graphs and the unpadded eager reference exceeded the current absolute-error gate. That graph variant has not passed acceptance; do not enable it based on the isolated matrix checks.

A same-power daytime comparison previously measured roughly 10.6 ms per proposal on one GPU, but used superseded feature inputs and normalization. Do not treat it as a correct deployment result.

With corrected layer mapping and SGLang backbone norm semantics, an offline reference replay on **one saved code and one prose trajectory** produced these mean committed tokens per round:

| Target / proposer | Chain, block 16 | 15-node tree, depth up to 15 |
|---|---:|---:|
| NVFP4 pack / BF16 dense head | 4.61 / 3.66 | 5.32 / 4.12 |
| Matched Quark core / BF16 dense head | 5.87 / 3.54 | 6.31 / 4.04 |
| Matched Quark core / channel-FP8, int2/80 candidates | 5.77 / 3.57 | 6.16 / 4.09 |

Each cell is code / prose. Tree construction reads only proposer probabilities; canonical future tokens are supplied afterward to an independent acceptance walk. `tree_proposal.py` checks normalized conditional probabilities, parent-before-child rows and 16-bit ancestor masks. The optional full transition output in `reference.serial_table` is a slow correctness path, not a serving implementation.

These corrected results are acceptance diagnostics, not three-prompt medians or live tok/s. TP=2 serving integration remains incomplete; corrected isolated proposer timing is described below.


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
