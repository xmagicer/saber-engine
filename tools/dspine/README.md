# DSpine validation tools

These are experimental numerical, pack-reading and offline replay utilities. They do not select a DSpine serving backend.

- `reference.py` implements the four selected-prefix bridge operations and greedy serial refinement in PyTorch.
- `plain_model.py` loads the five-layer BF16 DSpine checkpoint, keeps only the trained 8192-token teacher window, and never persists speculative query KV.
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
  --engine-draft-capture-layers 1,16,31,46,61
```

The DSpine taps differ from DFlash2's `[5,19,33,47,61]`. Capture uses an independently allocated plane so the resident DFlash2 proposal remains unchanged. Export directories/files use modes 0700/0600; existing directories are refused.

Verified on a real gfx1201 GPU: upstream Triton bridge parity with the BF16 checkpoint, and no-future-feature replay against captured target outputs. The eager BF16 implementation takes about 16-17 ms per proposal on the tested setup; this is an acceptance experiment, not end-to-end serving throughput. It needs an optimized proposer before it can offer a serving speedup.

Native group-128 W8A8 projections passed independent logical-matrix and varying-input graph checks. On the matched Quark-core target trajectory, the B16 proposer committed about 5.71 code tokens or 3.40 prose tokens per verification round. A graph with an approximate int2/80-candidate head retained about 5.64/3.40 while reducing proposal wall time to about 13.5 ms on one GPU. This still exceeds the resident TP=2 DFlash2 proposer's cost and has not been adopted for serving.

The channel-FP8 W8A16 variant is a separate numerical experiment. Its isolated projections passed matrix and graph checks, but its actual model comparison between padded graphs and the unpadded eager reference exceeded the current absolute-error gate. That graph variant has not passed acceptance; do not enable it based on the isolated matrix checks.

A same-power daytime comparison therefore kept the channel variant **eager**: group-128 graph proposals took 12.85/12.72 ms for code/prose, while channel-FP8 eager proposals took 10.64/10.50 ms. Mean committed tokens were 5.64/3.40 and 5.74/3.42 respectively on the same two target trajectories. These are single-GPU offline measurements, not live service throughput or three-prompt medians; TP=2 serving integration remains incomplete.
