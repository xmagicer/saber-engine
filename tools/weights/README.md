# Quark parameter control

`quark_control.py` builds a separate Qwen3.8-27B control pack from an existing two-rank Saber pack and a Quark group-32 MXFP4 checkpoint. It copies FP4 codes and E8M0 exponents without requantizing them, applies the TP split/merge and WMMA byte permutation, and also imports the matching target normalization, convolution and GDN rate parameters. Replacing the linear matrices alone is invalid because the calibration changes companion parameters.

The template's FP8 output head, vision and inactive MTP records remain unchanged. This control is not identical to an old deployment using a BF16 output head. Missing checkpoint KV-cache scales are set to the standard unit scale. Preserve the original pack; output must be a new directory.

```sh
python3 -m unittest discover -s tools/weights -p 'test_*.py' -v
python3 tools/weights/quark_control.py --checkpoint /models/old-quark \
  --template /models/saber-original --output /models/new-quark-control
cargo run --release -p saber-mem --bin saber-pack-verify -- \
  /models/new-quark-control/rank0.pack /models/new-quark-control/rank1.pack
```

The builder verifies every record after writing and rebuilds SBR record, chunk, table and tier digests. The Rust verifier independently opens the result with the runtime reader and hashes all records. Neither integrity check substitutes for generation and tool-call validation; run those before benchmarking or selecting the control for serving.
# Fine-tuned DFlash2 weights

The optional native drafter importer accepts the [TandemLLM B16 checkpoint](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16/tree/98759a4995e4949c462f9a6197f59e015fd82530). The author distributes the weights under Apache-2.0. The tested file has 3,848,817,896 bytes and SHA-256 `843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a`.

```sh
hf download 0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16 --revision 98759a4995e4949c462f9a6197f59e015fd82530 --local-dir /models/tandem-b16
python3 tools/weights/import_dflash2_drafter.py --check-roundtrip --checkpoint /models/Qwen3.8-27B-DFlash2-FP8 --template /packs/qwen38-dflash2-drafter-fp8head --target-pack /packs/qwen38-27b-nvfp4-devfinal-fp8head
python3 tools/weights/import_dflash2_drafter.py --checkpoint /models/tandem-b16 --template /packs/qwen38-dflash2-drafter-fp8head --target-pack /packs/qwen38-27b-nvfp4-devfinal-fp8head --expected-sha256 843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a --output /packs/qwen38-dflash2-tandem-b16-fp8head
cargo run --release -p saber-mem --bin saber-pack-verify -- /packs/qwen38-dflash2-tandem-b16-fp8head/rank0.pack /packs/qwen38-dflash2-tandem-b16-fp8head/rank1.pack
```

The old checkpoint control reproduced all 89 learned records per rank byte for byte. The importer consumed all 81 BF16 parameters of the new checkpoint. The runtime reader verified 96 records in each output shard. The teacher KV matrix follows the multiplication order of an identity input through the quantized projection before BF16 rounding.

The same converter accepts the [block-8 checkpoint](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8/tree/ecf60b9b0cfbf7b70e68d8d11b5eb2843c405a98). Select its exact revision `ecf60b9b0cfbf7b70e68d8d11b5eb2843c405a98`, pass its weight SHA-256 `32ed444f1b2e1d443f9330a5d1c6e3dc9dcfee648eba9f282a1a351dab582f02`, and use a fresh output directory. Both checkpoints have the same matrix shapes. The training block size comes from `dflash_config.block_size`.

For one active sequence set `DRAFT=qwen38-dflash2-tandem-b16-fp8head`, `DRAFT_BLOCK=16`, `MAX_SEQS=1` and `BATCHES=1`. Use `WIDTHS=1,16` with `--engine-spec-tree best16`, or `WIDTHS=1,10` with `--engine-spec-tree ancestor9`. Both runs use `--engine-gdn-replay off` and the host collective options in the [deployment guide](../../docs/HOST-COLLECTIVES.zh-CN.md).

### Block-32 trained checkpoint

The importer also accepts the BF16 JonasLoos checkpoint at revision `fc65843ba218eeaa8e0dbcc922034184acadd100`. Its 81 learned tensors have the same names, types and shapes as the B8/B16 training checkpoints. Download the pinned checkpoint and pass `--expected-sha256 01e8c919665f563339ff41ac788497f483da5418c860c051b5fee17b801b7423` to the import command above, using a fresh output directory. The independent pack reader verified all 96 records on each rank. Runtime draft blocks 8 and 16 were tested. [Checkpoint and training details](https://huggingface.co/JonasLoos/Qwen3.8-27B-DFlash2-b32/blob/fc65843ba218eeaa8e0dbcc922034184acadd100/README.md).

Single-request measurements with the matched Quark target, three fixed code and three fixed prose prompts, 1,024 output tokens each, give the following medians. At runtime block 16 the verifier uses 15 candidate nodes and one active sequence. At block 8 it uses nine nodes and allows eight active sequences.

| Trained draft | Runtime block | Code tok/s | Prose tok/s |
|---|---:|---:|---:|
| Tandem B16 | 16 | 214.58 | 151.27 |
| JonasLoos B32 | 16 | 234.24 | 146.32 |
| Tandem B16 | 8 | 221.52 | 162.20 |
| JonasLoos B32 | 8 | 221.14 | 153.51 |

At runtime block 16, median verification rounds per 1,024 tokens changed from 200 to 184 for code and from 283 to 294 for prose. Changing the drafter changed the generated text on all six prompts. In an additional run with the NVFP4 target and the B32-trained draft at runtime block 16, medians were 217.58 and 146.01 tok/s.
