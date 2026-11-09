# Qwen3.8-Flash-Next DFlash2

Run Inco's block-eight draft model alongside Saber's Flash-Next offline greedy verifier. Saber sends the BF16 mean of the four completed HC residual lanes at zero-based layers 5, 15, 25, 35 and 45. The drafter attends over a 2,048-position sliding window. It scores each successor with the learned rank-256 selector over the top 16 tokens. Saber checks up to four proposals and commits the accepted prefix and its own next token.

## Weights and dependencies

Use Linux, Python 3.10+, a PyTorch build supporting BF16 on your GPU and `safetensors`. ROCm PyTorch uses the `cuda` device namespace. The drafter loads the target's original BF16 embedding on the CPU and copies the selected LM-head rows to the draft GPU. The AMD checkpoint index and the shards containing `model.language_model.embed_tokens.weight` and `lm_head.weight` must be present. The verifier also needs its regular Saber pack and checkpoint.

```sh
hf download incoai/Qwen3.8-Flash-Next-DFlash2 config.json model.safetensors \
  --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6 --local-dir weights/df2
echo 'd6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f  weights/df2/model.safetensors' | sha256sum -c -
```

The tested target checkpoint revision is `79909fc9c57414bc59d52bd79e3343ad44a03515`. The target pack revision is `c455e46a8b76c85bbc7ed4395d447b3cd5b851af`. [Draft model and Apache-2.0 license](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6) · [AMD target checkpoint](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4) · [Saber target pack](https://huggingface.co/Launch80/Qwen3.8-Flash-Next-SABER).

By default the drafter scores all 248,320 vocabulary IDs. To reproduce the tested 106,299-row subset, download the little-endian int32 ID list and pass `--vocab weights/draft_vocab.bin` to the service. A different subset can change proposal acceptance.

```sh
curl -fL https://raw.githubusercontent.com/Niko1221/Strata/d2f096475817d4666c364c820b0ca830f751cfc5/data/draft_vocab.bin -o weights/draft_vocab.bin
echo 'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  weights/draft_vocab.bin' | sha256sum -c -
```

## Run

From the repository root, choose a fresh directory on a local filesystem that both processes can access. If they run in different containers, bind-mount its parent into both. Set the two weight paths to your own installation.

```sh
export FN_CHECKPOINT=/path/to/Qwen3.8-Flash-Next-Quark-MXFP4
export FN_PACK=/path/to/qwen38-flashnext-quark-trunk8
export RUN="$(mktemp -d)"
HIP_VISIBLE_DEVICES=1 python3 tools/flashnext_dflash2/service.py \
  --weights weights/df2 --checkpoint "$FN_CHECKPOINT" --device cuda:0 \
  --vocab weights/draft_vocab.bin --socket "$RUN/draft.sock" \
  --ready "$RUN/ready.json" --stats "$RUN/stats.json" --prewarm-all
```

Wait for `ready.json` and use the same `RUN` path in the verifier terminal. `--prewarm-all` captures all five commit sizes and all twelve (context capacity, proposal width) graphs before opening the socket. It checks each replay against eager execution on a second set of features, checks the short-width predictions, and clears the synthetic state. The readiness file includes timings and PyTorch allocator peaks in bytes. Omit this option for lazy capture, or use `--eager` for eager execution. Budget the target's resident expert weights to leave room on the draft GPU.

```sh
saber engine --pack "$FN_PACK" --tp 2 --fn-ar host --fn-mtp 0 \
  --fn-trunk w8-hc8-head --fn-experts tp2 --fn-kv-dtype fp8 \
  --fn-budget-mib 25500 --fn-max-chunk 2048 --fn-chunk 128 --fn-vision off \
  --fn-conc 1 --fn-turns 1 --fn-repeat 1 --fn-decode 128 --fn-graph-split layer \
  --fn-dflash2-socket "$RUN/draft.sock" --fn-dflash2-k 4 \
  --prompt-ids /path/to/prompt-token-ids.json --fn-out "$RUN/result"
```

The offline harness accepts a JSON array of token IDs. For a greedy control, omit the two DFlash2 arguments and use a fresh output directory with all other target settings and prompt IDs held fixed. Compare the full committed token sequences. Reports include proposed and accepted counts, decode time and individual commit timestamps.

The socket accepts one session at a time. A newline-terminated JSON `reset` begins a new sequence. An `append` header supplies the absolute `position` and `rows` (1–2,048), immediately followed by little-endian BF16 bytes in `[rows,5,2560]` order. A `propose` header supplies the next position, its anchor token ID and a width from 1 to 4. Only committed target rows are appended between proposals.

CPU regression tests cover width truncation, predecessor-conditioned selection and equal-score token ordering.

```sh
(cd tools/flashnext_dflash2 && python3 -m unittest -v test_prewarm test_selector)
```

## Dual R9700 measurements

Six single requests generated 1,024 tokens each on two Radeon AI PRO R9700
cards using TP2, host reductions, per-layer HIP graphs and a 25,500 MiB
target budget per rank. Each table cell shows the mean decode rate across
the three requests in that workload and the highest one-second commit rate,
in tokens/s. Decode timing begins after the first token from prefill.

| Offline greedy target | Code mean / peak | Prose mean / peak |
|---|---:|---:|
| Target only | 55.14 / 60 | 56.91 / 60 |
| DFlash2, two proposals | 83.31 / 105 | 81.59 / 102 |
| DFlash2, four proposals | 100.84 / 135 | 89.93 / 128 |
| Native MTP, four proposals | 107.80 / 154 | 99.17 / 134 |

The two DFlash2 variants reproduced all 6,144 target-only token IDs.
With four proposals, the target committed 3.44 tokens per verification
round on code and 2.80 on prose. Mean target verification took 25.81 ms
and 23.38 ms per round respectively. Draft proposals took 8.13 ms and
7.30 ms. The prewarmed BF16 draft process reserved 3.46 GiB on one card.
These measurements use the offline single-request interface and the
106,299-token draft vocabulary.

The arithmetic was checked against the public [Strata DFlash2 reference](https://github.com/Niko1221/Strata/blob/d2f096475817d4666c364c820b0ca830f751cfc5/tools/dflash2/dflash2_ref.py). The upstream MIT notice is included in [NOTICE](NOTICE).
