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
