# Saber

A Rust inference engine for the Qwen3.8 family on AMD Radeon AI PRO R9700 (gfx1201 / RDNA4).
Kernels ship as HIP code objects plus a manifest, the host owns every launch decision, and one
resolved plan replaces a flat environment-variable namespace. The server speaks the OpenAI API.

**Qwen3.8-27B** runs at tensor-parallel 2 on two R9700s with the DFlash2 speculative drafter:
about 229 tok/s single-stream decode, TTFT around 47 ms, and 5,000+ tok/s prefill at a 210 W
per-card cap, with a 262,144-token context, prefix caching and optional vision.
It also runs on **one card** (TP=1, `tools/serve_27b.sh --tp 1`) with smaller context and concurrency
defaults; see "One card (TP=1)" below.
**Qwen3.8 Flash-Next** (MoE) is served by the same binary.

For TP=2 hosts without GPU peer access, see the opt-in [RCCL compatibility profile](docs/RCCL-COMPAT.md).
It uses shared host memory and preserves the native P2P backend as the default.
An [experimental native host-memory backend](docs/HOST-COLLECTIVES.md) retains
more decode fusions; its validation scope and measured tradeoffs are documented separately.
中文部署步骤与对比见[双 R9700 无 P2P 部署](docs/HOST-COLLECTIVES.zh-CN.md)。

## What you need

| | |
|---|---|
| GPUs | 2x Radeon AI PRO R9700 (gfx1201), PCIe; one card works for TP=1 |
| OS / runtime | Linux with the ROCm kernel driver (`/dev/kfd`, `/dev/dri`) and **podman or docker** |
| Host | a fast NVMe for the weight packs (`/mnt/nvme/saber-packs`; 27B about 26 GB; Flash-Next about 70 GB, with its 170 GB checkpoint under `models/`) |
| Weights | the prebuilt packs from `tools/fetch_packs.sh`: 27B ~26 GB (TP=2) or ~25 GB (`27b-tp1`, one card); Flash-Next ~70 GB of packs plus the original 170 GB checkpoint |

All builds and GPU runs happen inside the dev container, so the host needs no ROCm toolchain.

## Podman or docker

Every launcher in `tools/` runs on either. It uses podman when installed, otherwise docker; force
one with `SABER_CONTAINER`:

```bash
SABER_CONTAINER=docker tools/serve_27b.sh --no-build
export SABER_CONTAINER=docker      # or set it once for the shell
```

| | podman (rootless) | docker |
|---|---|---|
| GPU access | `--device /dev/kfd --device /dev/dri --group-add keep-groups` | same devices, plus `--group-add <gid>` for the owners of `/dev/kfd` and `/dev/dri/*` (usually `video` and `render`) |
| File ownership | container root maps to you | the launchers pass `--user $(id -u):$(id -g)`, so `.cache/` stays yours |
| Logs / stop | `podman logs -f saber-27b-serve` | `docker logs -f saber-27b-serve` |

Your user needs read/write on `/dev/kfd` and `/dev/dri/renderD*` (`sudo usermod -aG video,render $USER`,
then log in again). On an SELinux host the launchers already pass `--security-opt label=disable`.
The shim is `tools/lib/container.sh` (it sets `CTR`, `CTR_GPU` and `CTR_USER`); a new script sources it
and calls `"$CTR" run "${CTR_GPU[@]}" ...`.

`tools/build-image.sh` needs nothing but the container runtime and network: it builds
[`Containerfile.public`](Containerfile.public) on a public ROCm 7.x base
(`docker.io/rocm/dev-ubuntu-24.04:7.2.4`; pick another with `SABER_ROCM_IMAGE`) plus Rust and
bindgen. The image compiles the whole
workspace and builds kernels with `hipcc`. It has not yet booted a model; the project's own image
carries a newer ROCm (HIP 7.16, ROCr 1.21), and on the public 7.x tags each rank may spin more CPU
in ROCr's async-event loop.

## Guided setup (terminal UI)

```bash
tools/saber-setup
```

A text-mode wizard (Python 3 standard library only, no installs) that does the setup for you:

1. **System** checks the GPU driver, that you have two supported GPUs (one for the 27B at TP=1), podman
   or docker, the dev image, the Hugging Face CLI, and free disk space.
2. **Configure** picks the model (27B or Flash-Next), the parallelism (TP=2 on two GPUs, or TP=1 on one
   GPU for the 27B) and shows a checklist of the common options with our recommended defaults already
   selected: prefix caching, images (vision) off / lazy / on, thinking, max context, concurrent
   sequences, port, packs directory, and which GPU(s) to use. `d` puts everything back to the defaults.
3. **Review & run** shows the exact launch command, what is installed and what is missing, and can
   install the missing pieces (builds the image, downloads the weights, each step asks first), save a
   re-runnable launch script, start the server, or stop it.

Keys: arrows or `j`/`k` move, Space/Enter change a row, `←`/`→` cycle choices, Tab switches page, `q`
quits. Settings are saved to `~/.config/saber/setup.json` and the launch script to
`~/.config/saber/launch-<model>.sh`. `tools/saber-setup --print` shows the plan without the UI.

Parallelism is **TP=2** (both cards) by default, and the wizard lets you choose which two GPUs. For the
27B you can pick **TP=1** instead: it then asks for ONE GPU, downloads the single-card packs
(`tools/fetch_packs.sh 27b-tp1`) and launches `tools/serve_27b.sh --tp 1` with the TP=1 context and
sequence defaults (see "One card (TP=1)" below; `tools/saber-setup --tp 1 --print` shows that plan).
Flash-Next is TP=2 only (no single-card Flash-Next pack is hosted), and the wizard says so on its TP=1
row. The engine refuses any other TP (3, 4, ...) by name: it is built for TP=1 and TP=2. The manual
steps below do the same thing without the wizard.

## Get running

Saber runs **two models**, each from prebuilt weight packs hosted on Hugging Face. Pick one (or both;
only one server runs at a time, they share the cards).

| | Qwen3.8-27B | Qwen3.8 Flash-Next (MoE) |
|---|---|---|
| Download | about 26 GB | about 70 GB of packs **plus the original 170 GB checkpoint** (see below) |
| Hugging Face | [`Launch80/Qwen3.8-27B-NVFP4-SABER`](https://huggingface.co/Launch80/Qwen3.8-27B-NVFP4-SABER) | [`Launch80/Qwen3.8-Flash-Next-SABER`](https://huggingface.co/Launch80/Qwen3.8-Flash-Next-SABER) + [`amd/Qwen3.8-Flash-Next-Quark-MXFP4`](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4) |
| Fetch | `tools/fetch_packs.sh 27b` (two cards, TP=2); `tools/fetch_packs.sh 27b-tp1` (one card, TP=1, about 25 GB) | `tools/fetch_packs.sh flashnext` |
| Serve | `tools/serve_27b.sh` (two cards); `tools/serve_27b.sh --tp 1` (one card) | `tools/serve_flashnext.sh` |

```bash
git clone git@github.com:GGZ14/saber-engine.git && cd saber-engine

# 1. Build the dev image once (ROCm, Rust, hipcc).
tools/build-image.sh

# 2. Pull the weights. Downloads resume if interrupted. Packs land in /mnt/nvme/saber-packs
#    (override with SABER_PACKS); it also needs `pip install -U 'huggingface_hub[cli]'`.
tools/fetch_packs.sh 27b            # or: flashnext   (or: all = 27b + flashnext, both TP=2)
#    One card? use the TP=1 packs instead:  tools/fetch_packs.sh 27b-tp1   (see "One card (TP=1)" below;
#    `all` does not include them)

# 3. Serve. Builds the release binary, boots both cards, waits for /v1/models.
tools/serve_27b.sh --enable-prefix-caching on        # or: tools/serve_flashnext.sh
#    One card (with the 27b-tp1 packs):  tools/serve_27b.sh --tp 1 --enable-prefix-caching on

# 4. Talk to it.
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen3.8",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 128
}'
```

(Use model id `Qwen3.8-Flash-Next` for Flash-Next.)

**One card (TP=1).** A pack is cut for one TP degree: its rows and columns are already sharded per
rank, so the TP=2 packs above (`rank0.pack` + `rank1.pack`) cannot be loaded on one card. The same
Hugging Face repo hosts the single-card packs as separate folders, each with one `rank0.pack`:
`qwen38-27b-nvfp4-devfinal-tp1-fp8head` (18.5 GB) and `qwen38-dflash2-drafter-tp1-fp8head` (6.4 GB).
`tools/fetch_packs.sh 27b-tp1` downloads them into the packs directory (the tokenizer and chat template
are the same files as TP=2's). Then:

```bash
tools/serve_27b.sh --tp 1 --enable-prefix-caching on          # card 0
DEVICE=1 tools/serve_27b.sh --tp 1 --no-build                 # or card 1 (container saber-27b-tp1-serve)
```

One card holds the 18.5 GB target pack and the 6.4 GB drafter pack, so the KV pool is far smaller than
TP=2's, and the TP=1 defaults are **`MAX_CTX=65536` per sequence, `MAX_SEQS=4`, decode batches
`1,2,4`** (draft block 8 and verify widths `1,8` as at TP=2). At those defaults the pool holds 10,548
blocks = 168,768 tokens, about 2.5 full-length sequences. Measured on one R9700 (card 0 at a 210 W cap;
ROCm 10.1 dev image; the optional 2 GiB P2P pool carved out of the card):

| TP=1, one R9700 | measured |
|---|---|
| Single-stream decode, 256 tokens, temperature 0 | 67 tok/s (prose) / 148 tok/s (code) / 231 tok/s (counting): the speculative drafter's acceptance sets the rate (card 1 at a 300 W cap: 68 / 152 / 237) |
| Max context per sequence | 65,536 by default; 131,072 and 163,840 also boot (a 145k-token prompt answered, prefill ~2,650 tok/s); 196,608 is refused at boot (one sequence would not fit the pool) |
| Max sequences | 4 by default; 8 boots and serves (`MAX_SEQS=8 BATCHES=1,2,4,8`: the GDN state doubles, the pool drops to 101,120 tokens) |
| VRAM | 29.5 GiB used at the served peak (31.5 GiB with the P2P pool), of 32 GiB |
| Images | `--vision lazy` and `--vision on` both work (`on` keeps the tower resident: the pool drops to 122k tokens, so keep `MAX_CTX` at 65,536 or below and `MAX_SEQS` at 4) |

Boot takes under 10 s with the packs in the page cache. The chat, streaming, 1k- and 2.6k-token prompts
and a 3-request burst were checked on both cards; no TP=1 BetterBench run, prefill sweep or long soak
has been done, and the speculative tree verify form is planned for TP=2 only (a TP=1 boot runs the
chain form and says so in its log). No single-card Flash-Next pack is hosted.

**What the 27B download contains.** The target pack (`qwen38-27b-nvfp4-devfinal-fp8head`), the DFlash2
drafter pack (`qwen38-dflash2-drafter-fp8head`), and the tokenizer and chat template, which
`fetch_packs.sh` puts in `.cache/`. Nothing else is needed.

**What Flash-Next needs, and why it is bigger.** The Flash-Next pack is not self-contained: for the
embeddings, norms and router, and for the 102 GB per-layer-embedding (PLE) table, it points into the
original checkpoint's safetensors files and reads them in place. `fetch_packs.sh flashnext` therefore
downloads both the pack (`qwen38-flashnext-quark-trunk8`) and the checkpoint. The pack's manifest names
the checkpoint by absolute path, so it goes in `/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4`; the
script reads the path from the manifest and downloads there. The launcher mounts
`$SABER_NVME/models` into the container, so keep `SABER_NVME` at its default (`/mnt/nvme`) unless you
also change that path. The checkpoint is AMD's, under its own license.

**The n-gram (PLE) table.** Flash-Next's per-layer-embedding table is the n-gram lookup behind every
token: 320,001,536 rows of 160 bf16 values (95.4 GiB, 128 shards, tensors named
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_N.weight`). It is not a separate
download and not in the pack: it lives inside 33 of the checkpoint's 131 safetensors files, and the
pack's `ple/ple_index.json` maps each row to its file and byte offset. Each token reads 16 rows
(5,120 bytes) from the drive with `O_DIRECT`, so keep the checkpoint on a fast NVMe, which is where the
default path puts it. The location is fixed in two places, both absolute paths:
`manifest.json` (`checkpoint.dir`) and `ple/ple_index.json` (`sources.ckpt_bf16.checkpoint`). The
index also lists a first-party fp8 copy of the table (half the size), but it is off by default,
gated on a numerics check, and not part of the hosted files.

## Can I run any MXFP4 model?

No. There are two separate things in the way, and neither is the quantization format.

1. **Saber does not load checkpoints directly.** It boots from a *pack* (`SBRPACK1`, one
   4096-byte-aligned file per GPU with a sha256 per record): the weights pre-arranged for its GPU kernels (WMMA fragment
   order, scales transposed, tensor-parallel split, projections merged, fp8 trunk where used). That is
   why even a checkpoint that is already MXFP4, like AMD's Flash-Next one, gets packed first. For
   that checkpoint the packer copies the 4-bit values and scales byte for byte (no requantization); it
   only rearranges them. The 27B packer starts from an NVFP4, per-channel fp8 or bf16 checkpoint and
   requantizes it to MXFP4.
2. **Only two model architectures are implemented:** Qwen3.8-27B (dense, with the DFlash2 drafter) and
   Qwen3.8 Flash-Next (MoE). The engine, kernels and plans are written for those layer graphs. A different
   model family (Llama, Mistral, another Qwen) is not a repack: it needs a new model definition in
   `crates/saber-model` and new kernel tuning rules.

So: for the two supported models, download the prebuilt packs (above). **Pack builders are not part of
this release**, so a model we do not host cannot be packed with the tools here yet.

The server listens on port 8080 and exposes `/v1/models`, `/v1/chat/completions`,
`/v1/completions` and `/metrics`. Only one server runs at a time; they share the cards.

## Flags and options

Both launchers take their own flags first, then pass everything else straight to `saber`.

### 27B: `tools/serve_27b.sh [flags]` (model id `Qwen3.8`)

| Flag | What it does |
|---|---|
| `--no-build` | Skip the `cargo build`; reuse `.cache/target-container/release/saber`. Use this for every boot after the first (a build takes about two minutes). |
| `--tp 1` | Serve on ONE card (TP=1) from the `-tp1-fp8head` packs (`tools/fetch_packs.sh 27b-tp1`), on HIP device `DEVICE` (default 0), container `saber-27b-tp1-serve`. `TP=1` in the environment does the same; `--tp 2` (the default) is the two-card server. See "One card (TP=1)". |
| `--vision` | Accept images (`image_url` parts). Same as `--vision lazy`: the vision tower lives in pinned host RAM and streams through the idle prefill arena per image. No VRAM is billed and the full 262k context stays, at about +30 ms per image. |
| `--vision on` | Keep the tower resident (about 1.9 GiB a rank). Needs `MAX_CTX=131072`. |
| `--vision off` | Text only (the default). |
| `--vision-max-patches N` | Largest image, in patches, the resident tower is sized for. |
| `--enable-prefix-caching on` | Cache finished prompts, so a returning conversation prefills only its new tail. Default `off`. |
| `--warm-prompts FILE` | Prefill the system prompt(s) in `FILE` at boot and report ready only when done. Turns prefix caching on. |
| `--tool-call-parser NAME` / `--reasoning-parser NAME` | Tool-call and reasoning-block parsing. Default `qwen3_coder` and `qwen3`. |
| `stop` | Stop the server container: whichever 27B server (TP=2 or TP=1) is running. `--tp 1 stop` stops only the TP=1 one. |

Environment variables (put them before the command):

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | Listen port |
| `MAX_CTX` | `262144` (TP=1: `65536`) | Context per sequence; lower it for more concurrent sessions or `--vision on` |
| `MAX_SEQS` | `8` (TP=1: `4`) | Concurrent sequences |
| `BATCHES` | `1,2,3,4,8` (TP=1: `1,2,4`) | Captured decode batch sizes |
| `DEVICE` | `0` | TP=1 only: the one HIP device to serve on (the container sees only that card) |
| `HIP_DEVICES` | `0,1` | TP=2 only: the two HIP devices |
| `WIDTHS` | `1,8` | Speculative verify widths |
| `DRAFT_BLOCK` | `8` | Drafter block size (`4` halves the GDN slots) |
| `MODEL` | `Qwen3.8` | Served model id |
| `SABER_PACKS` | `/mnt/nvme/saber-packs` | Where the weight packs live |
| `SABER_SHARED` | `~/saber-cache` | Scratch and logs shared between runs |
| `SABER_IMAGE` | `localhost/saber-dev:latest` | The dev image |
| `SABER_CONTAINER` | autodetect | `podman` or `docker` |
| `CNAME` | `saber-27b-serve` (TP=1: `saber-27b-tp1-serve`) | Container name |

### Flash-Next: `tools/serve_flashnext.sh [flags]` (model id `Qwen3.8-Flash-Next`)

| Flag | What it does |
|---|---|
| `--no-build` | Reuse the last release build, as above. |
| `--fn-vision [lazy\|on\|off]` | Images. `lazy` (the default boot) loads the tower on the first image request and unloads it after 300 s idle. `off` is text only. Image prompts are prefix-cached by content hash (`--fn-image-cache`, on by default), and oversized images are downscaled, not refused. |
| `--fn-thinking off` | Do not think by default (a request can still set `enable_thinking`). Default `on`. |
| `--fn-max-ctx N` | Context limit; the default boot serves 262,144. |
| `--fn-prefix-tree off` | Disable the tiered prefix cache (on by default). |
| `--warm-prompts FILE` | As above. |
| `--record-routes` | Record expert routes from your traffic under `$SABER_NVME/saber-kv/routes/` (see the script header for the rebuild step). |
| `stop` | Stop the server container. |

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | Listen port |
| `SABER_NVME` | `/mnt/nvme` | Parent of `models/`, `saber-packs/`, `saber-kv/` |
| `PACK` | `$SABER_NVME/saber-packs/qwen38-flashnext-quark-trunk8` | Weight pack |
| `FN_TRUNK` | `w8-hc8-head` | `w8`, `w8-hc8` or `bf16` select slower, higher-precision trunks |
| `FN_KV` | `fp8` | `bf16` for a bf16 KV cache |
| `FN_EXPERTS` | `tp2` | `ep` for expert parallelism |
| `FN_AGENT_DEFAULTS` | `1` | `0` drops the agent-load tuning flags (lazy KV, bigger prefix tree, prefill yield) |
| `GPU_MAX_HW_QUEUES` | `3` | ROCm hardware queues per process |

Every other `saber engine` flag goes after the launcher's own; `saber engine --help` lists them.

### Images

```bash
tools/serve_27b.sh --no-build --vision          # or: tools/serve_flashnext.sh --no-build
```

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen3.8",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}
  ]}],
  "max_tokens": 200
}'
```

Without a vision flag the 27B refuses an image part by name.

### Everyday commands

| Task | Command |
|---|---|
| Stop the server | `tools/serve_27b.sh stop` |
| Reuse the last build | `tools/serve_27b.sh --no-build` |
| Another port | `PORT=8100 tools/serve_27b.sh ...` |
| Smaller context | `MAX_CTX=65536 tools/serve_27b.sh ...` |
| One card | `tools/serve_27b.sh --tp 1` (card 1: `DEVICE=1 tools/serve_27b.sh --tp 1`) |
| Logs | `podman logs -f saber-27b-serve` (or `docker logs -f ...`) |
| Exit code of a dead server | `podman wait saber-27b-serve` |
| Flash-Next | `tools/serve_flashnext.sh` |

### Troubleshooting

| You see | Cause |
|---|---|
| `missing pack ...` / `missing .cache/template_corpus...` | A step in "Get running" was skipped; the message names the file. |
| `port 8080 is already in use` | Another server is up; stop it or set `PORT`. |
| `another saber server or an agent hold holds the cards` | A `saber-*` container is running; `ps` it and stop it. |
| `container ... holds HIP device N` (TP=1) | A running `saber-*` container already uses that card; stop it or pick another `DEVICE`. |
| `... is a TP=2 pack (it has rank1.pack)` | TP=1 needs the single-card packs: `tools/fetch_packs.sh 27b-tp1`. |
| Exit code **86**, last log line `saber: FATAL rank R: <reason>` | A rank failed. The server exits, answers queued requests 503 and frees both cards within 10 s. |
| `permission denied` on `/dev/kfd` | Add yourself to `video` and `render`, log in again. Under docker, check the printed `--group-add` gids. |
| `neither podman nor docker is installed` | Install one, or set `SABER_CONTAINER`. |

## Optional: SSD to VRAM peer-to-peer

Stock `amdgpu` cannot be the target of an NVMe read, so everything loaded from the drive
reaches VRAM through host RAM and costs host memory bandwidth twice. `tools/kernel/fn71/`
carries a patch that adds `amdgpu.p2pmem_mb=N`: N MiB of each card's VRAM is registered at boot
as a PCI P2PDMA pool, so the drive writes it directly and the GPU reads the same bytes.
Measured on the R9700s: 9.5 to 10.7 GB/s drive to VRAM, byte-equal, with host memory bandwidth
untouched (a CPU stream keeps 51.7 of 52 GB/s, against 32.4 beside the bounce copy).

- It needs a custom kernel build (`CONFIG_PCI_P2PDMA`, tested on 7.1.3) and a reboot:
  `sudo FN71_P2PMEM_MB=2048 bash tools/kernel/fn71/install.sh`. To turn it off, boot with
  `amdgpu.p2pmem_mb=0`.
- Cost: the pool is carved out of VRAM (2 GiB a card at 2048). Both servers subtract it from their
  budgets on their own.
- Nothing requires it. `saber_hsa::NvmeVram` uses the pool route when it exists and falls back to
  the bounce copy otherwise. It speeds up loads and reloads; the decode step path does not read
  from the drive.

## Develop

```bash
tools/dev.sh cargo test -p saber-plan        # any crate, inside the dev image
tools/dev.sh cargo build --release -p saber-cli
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
```

`tools/dev.sh` and `tools/hipcc.sh` use podman or docker like the launchers. Host `cargo` is fine for pure-Rust unit tests; anything that touches HIP or a GPU goes through
`tools/dev.sh`. Compile a kernel with `tools/hipcc.sh crates/saber-kernels/hip/<k>.hip out.hsaco`.
`saber --help` lists the CLI: `serve`, `engine` (one-prompt gate with ms/step), `probe`, `gate`,
`explain` and the per-feature gates.

## Reproducing the code objects

`prebuilt/gfx1201/` holds 116 compiled GPU code objects. Each has a manifest next to it recording the
source file, its sha256, the compiler flags, a build hash (sha256 of the preprocessed source plus
flags) and the object's own sha256. The sources are in `crates/saber-kernels/hip`.

**Two ROCm toolchains built them**, both AMD "TheRock" builds. Each object records its compiler in
the file; `tools/verify_prebuilt.py --table` lists it per object.

| Objects | ROCm | TheRock commit | LLVM (`ROCm/llvm-project`) | Patches on LLVM |
|---|---|---|---|---|
| 64 | 10.1.0rc3 (HIP 7.16.26385) | `b324869fb311` | `d6f6cb69` (clang 24.0.0git) | none |
| 52 | 7.14.0rc3 (HIP 7.14.60850) | `418cd5f63abb` | `46fcb339` (clang 23.0.0git) | 2 TheRock patches, both `hipcc` driver path handling (Windows include path, `HIP_PATH` precedence); the clang string carries `+PATCHED:` for them |

Saber applies no patches to LLVM. The container image from `tools/build-image.sh` uses the public
ROCm 7.2.4 image, whose compiler (clang 22) emits different code, so it will not reproduce these
objects. To check them you need the matching toolchain:

```bash
# rebuild with a toolchain that has the right ROCm (SABER_IMAGE names the image tools/hipcc.sh uses)
SABER_IMAGE=<image with that ROCm> cargo run -p saber-kernels --bin saber-kbuild -- --out /abs/path/inside/repo/rebuilt
tools/verify_prebuilt.py --rebuilt /abs/path/inside/repo/rebuilt
```

**Compare the machine code, not the files.** Two builds of one source with one compiler differ in
the dynamic symbol-table sections, so file hashes never match. `verify_prebuilt.py` compares `.text`,
`.rodata` and the register limits, and only against a rebuild by the same compiler. When we ran it,
every shipped object matched: 64 of 64 on ROCm 10.1 and 52 of 52 on ROCm 7.14, none different.

The MXFP4 GEMM sources were derived from `tools/reference/ref_mxfp4_fp8.hip` by
`tools/gen_hip_sources.py`. The committed `.hip` files are the inputs that were compiled, not the
generator output: they have been renamed since, and the generator no longer reproduces every file
(its header says which). Its reference differs from the one the file headers record by one comment and an identifier-prefix rename (the generator was renamed to match).

## Layout

| Path | What |
|---|---|
| `crates/saber-cli` | the `saber` binary |
| `crates/saber-engine`, `-exec`, `-plan`, `-sched` | engine, execution graphs, launch plan, scheduler |
| `crates/saber-serve` | the OpenAI HTTP plane |
| `crates/saber-kernels`, `prebuilt/gfx1201` | HIP kernel sources and their compiled code objects with manifests |
| `crates/saber-mem`, `-model` | weight packs and the model graph |
| `tools/` | launchers, the setup wizard, kernel generators and their MXFP4 reference, `verify_prebuilt.py`, the optional kernel patch |

## License

Saber is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 GGZ14. If you redistribute it
or a fork, keep the [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) files with it: the Apache License
requires the `NOTICE` attribution to be carried into your distribution.

Two things in this repository are not covered by that license. `tools/kernel/fn71/` holds a patch to the
Linux kernel, which is under the kernel's own license (GPL-2.0). The model weights the launchers download
(Qwen3.8 and AMD's Flash-Next checkpoint) keep their own licenses.
