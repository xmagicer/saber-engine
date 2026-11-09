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
# B16-trained DFlash2 weights

The optional native drafter importer accepts the [TandemLLM B16 checkpoint](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16/tree/98759a4995e4949c462f9a6197f59e015fd82530). The author distributes the weights under Apache-2.0. The tested file has 3,848,817,896 bytes and SHA-256 `843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a`.

```sh
hf download 0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16 --revision 98759a4995e4949c462f9a6197f59e015fd82530 --local-dir /models/tandem-b16
python3 tools/weights/import_dflash2_drafter.py --check-roundtrip --checkpoint /models/Qwen3.8-27B-DFlash2-FP8 --template /packs/qwen38-dflash2-drafter-fp8head --target-pack /packs/qwen38-27b-nvfp4-devfinal-fp8head
python3 tools/weights/import_dflash2_drafter.py --checkpoint /models/tandem-b16 --template /packs/qwen38-dflash2-drafter-fp8head --target-pack /packs/qwen38-27b-nvfp4-devfinal-fp8head --expected-sha256 843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a --output /packs/qwen38-dflash2-tandem-b16-fp8head
cargo run --release -p saber-mem --bin saber-pack-verify -- /packs/qwen38-dflash2-tandem-b16-fp8head/rank0.pack /packs/qwen38-dflash2-tandem-b16-fp8head/rank1.pack
```

The old checkpoint control reproduced all 89 learned records per rank byte for byte. The importer consumed all 81 BF16 parameters of the new checkpoint. The runtime reader verified 96 records in each output shard. The teacher KV matrix follows the multiplication order of an identity input through the quantized projection before BF16 rounding.

For one active sequence set `DRAFT=qwen38-dflash2-tandem-b16-fp8head`, `DRAFT_BLOCK=16`, `MAX_SEQS=1` and `BATCHES=1`. Use `WIDTHS=1,16` with `--engine-spec-tree best16`, or `WIDTHS=1,10` with `--engine-spec-tree ancestor9`. Both runs use `--engine-gdn-replay off` and the host collective options in the [deployment guide](../../docs/HOST-COLLECTIVES.zh-CN.md).
