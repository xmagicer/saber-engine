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
  --engine-draft-capture-layers 2,17,32,47,62
```

The checkpoint's `[1,16,31,46,61]` are **zero-based decoder output** ids. Saber's `lN.resid.a` is the residual entering layer N, so the actual capture points are **`[2,17,32,47,62]`**. `target_layers.py` performs this mapping and rejects captures using the unshifted list. Do not apply this DSpine-specific convention to an older DFlash2 pack automatically. Capture uses an independently allocated plane so the resident DFlash2 proposal remains unchanged. Export directories/files use modes 0700/0600; existing directories are refused.

Verified on a real gfx1201 GPU: upstream Triton bridge parity with the BF16 checkpoint, and no-future-feature replay against captured target outputs. The eager BF16 implementation takes about 16-17 ms per proposal on the tested setup; this is an acceptance experiment, not end-to-end serving throughput. It needs an optimized proposer before it can offer a serving speedup.

Native group-128 W8A8 projections passed independent logical-matrix and varying-input graph checks. **Earlier acceptance measurements used the unshifted capture ids and are invalid as DSpine quality results.** The low-level numerical/causality checks remain useful, but acceptance must be remeasured with correctly captured decoder outputs. This work has not been adopted for serving.

The channel-FP8 W8A16 variant is a separate numerical experiment. Its isolated projections passed matrix and graph checks, but its actual model comparison between padded graphs and the unpadded eager reference exceeded the current absolute-error gate. That graph variant has not passed acceptance; do not enable it based on the isolated matrix checks.

A same-power daytime comparison kept the channel variant **eager** and measured roughly 10.6 ms per proposal on one GPU. This was also collected with the incorrect feature-layer inputs; it is not evidence of a correct DSpine deployment or live service throughput. Corrected acceptance and performance replay is pending; TP=2 serving integration remains incomplete.
