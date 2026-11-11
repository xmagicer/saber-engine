//! `saber` — explain | probe | gate | bench-steploop | load | pack | serve. Subcommands land as
//! their crates do.
//!
//! Five are implemented:
//!
//! * `probe` — `dlopen` the HIP runtime and print what the machine actually is, rather than what a
//!   config file claims. Everything comes from `hipDeviceGetAttribute` and friends: no
//!   `hipDeviceProp_t`, no ROCm linked into this binary, no environment variable in the path.
//! * `explain` — read one manifest, build the registry from the committed artifacts, resolve the
//!   model, and print every route with its promotion state, workspace kinds, graph-safety and
//!   launch count. With a cold autotune cache that is the untuned work list, which is the P0 state.
//! * `gate` — the P0 bit-exact gate: replay the captured reference cases through the engine's own
//!   loader and launcher and byte-compare. Replaces `hip/harness/run_hsaco.cpp`.
//! * `bench-steploop` — P0.5: capture N tiny kernels, replay per step request off a bounded
//!   channel, report host microseconds per step.
//! * `tune` — the P0 autotune sweep: measure every untuned route's legal variants, interleaved,
//!   and write the cache `explain` then binds against.
//! * `load` — the cold-start path: one rank's pack, `O_DIRECT` at qd≥8, `sha256`-verified, through
//!   a fixed pinned staging ring into one VRAM arena, with per-record readiness and the byte
//!   ledger in front of the allocation.
//!
//! There is **no environment variable anywhere in this binary**. Verbosity, the HIP runtime to
//! load, the device to use and every path are command-line arguments, so a run is reproducible
//! from its command line alone.

mod ar_gate;
mod bench;
mod block_capture;
mod capture_gate;
mod census_parity;
mod chain_bench;
mod d2d_gate;
mod device_row_gate;
mod device_row_shards;
mod draft_proposal;
mod draft_replay;
mod draft_seq;
mod draft_step;
mod engine;
mod explain;
mod flashnext;
mod fn63_hsa;
mod fn_moe_bench;
mod fn_profile;
mod forward;
mod forward_tp2;
mod gate;
mod gdn_race;
mod gpu_idle;
mod layer;
mod ledger_law;
mod load;
mod npy;
mod park_bench;
mod ragged_attn_gate;
mod sample_check;
mod serve;
mod step_capture;
mod step_tp2;
mod tune;
mod verify_step;
mod xtier_flags;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use saber_hip::{HipThread, Topology, DEFAULT_HIP_SONAME};

#[derive(Parser)]
#[command(name = "saber", version)]
struct Cli {
    /// Log at debug level. Verbosity is a flag, not `RUST_LOG`: the engine reads no environment
    /// variables, and the CLI keeps the same rule so that a run is reproducible from its command
    /// line alone.
    #[arg(long, short, global = true)]
    verbose: bool,

    /// FN73: seconds the process may stay alive after a rank failure (the fatal) before the
    /// watchdog `_exit`s it with code 86.
    #[arg(long, global = true, hide = true, default_value_t = 10)]
    fatal_watchdog_s: u64,

    #[command(subcommand)]
    cmd: Cmd,
}

/// The `--hip-lib` help, shared by every subcommand that touches a GPU.
/// The prefill row buckets every serving launcher plans by default (C5, 2026-09-23): a chunk
/// runs every GEMM at its BUCKET's row count, so a prompt one row past a bucket pays the next
/// bucket's whole cost; `256,2048` (the pre-C5 rows) ran 144 of 400 conc-8 corpus prompts at 2,048
/// rows and this ladder took the corpus from 104.2 s to 68.3 s with GSM8K inside the band. The
/// floor is 96 because a 64-row bucket selects the split-K decode GEMM. `256,2048` stays the
/// control. The engine worker's copy must match rank 0's, so all three flags share this one.
/// C6's small-chunk law is the default (2026-09-23): prefill rungs of at most this many rows are
/// planned on the split-K decode GEMM at ONE split, which sums K in the folded kernel's order and
/// is gated bit-exact against it (p96 56.3 -> 42.2 ms; conc-8 corpus 65.9 -> 63.6 s). 0 is the
/// pre-C6 control. The worker's copy is forwarded by the leader, so all three flags share this.
pub const DEFAULT_SMALL_CHUNK_GEMM: u32 = 128;

/// P27-1d: the split all-reduce's push blocks when `--ar-form split` names none (P27-1c's pick:
/// 16 x 1,024 threads, fastest or tied in every op-level cell). The FORM's default is `oneshot`.
pub const DEFAULT_AR_PUSH_BLOCKS: u32 = 16;

/// `--ar-form` + `--ar-push-blocks` as the one number the engine carries: 0 = the one-shot (the
/// control), N = the split form with N push blocks.
#[must_use]
pub fn ar_push_blocks_of(form: &str, push_blocks: u32) -> u32 {
    if form == "split" {
        push_blocks
    } else {
        0
    }
}

/// The mixed step's default form (2026-09-24, the user: "lets go with packed"): several queued
/// prompts' whole chunks ride in one fused forward with the decode rows (C10/C11). `off` is the
/// unmixed schedule, the pairing control.
pub const DEFAULT_MIXED_PREFILL: &str = "packed";

/// P27-6: the drafter's KV law. **`paged` IS THE DEFAULT since 2026-10-05** (the user's word of
/// 2026-10-03, every gate green: engine gate ids / acceptance / every compute stage identical,
/// served greedy text + logprobs identical at 1.5k..250k incl. two needles and an n=2 fork,
/// BetterBench --quick 4 + 4 tied; 262k KV pool 21,169 -> 47,335 blocks). `dense` (`max_seqs` runs
/// of `max_ctx` pages billed before the arenas) is the control.
pub const DEFAULT_DRAFT_KV: &str = "paged";

/// P27-9b: the drafter's input embedding SHARED with the target's `embed_tokens` (one 1.27 GB
/// record a rank at TP=2 instead of two; the pair checked byte-identical at boot, a mismatched pair
/// refused by name). **THE DEFAULT since P27-9d (2026-10-07)**; `off` is the control. P27-9c passed
/// the share-on engine gate twice but the served 262k boot FAULTED in `dflash_rope_neox_k_inplace`:
/// the drafter's rope table was the pack's 4,096-row `context_kv.rope.cos_sin`, read by position
/// with no bound (with two copies the over-read landed in the drafter's own embedding; with the
/// share it left the upload window). P27-9d builds that table for `max_ctx` at load, refuses a
/// shorter one by name and checks every seeded position against it (LEDGER P27-9d).
pub const DEFAULT_DRAFT_EMBED_SHARE: &str = "on";

pub const DEFAULT_PREFILL_ROWS: &str = "96,128,192,256,320,384,512,768,1024,1536,2048";

const HIP_LIB_HELP: &str = "Soname or full path of the HIP runtime to load. Swapping this is the \
free A/B axis across the ROCm userspaces on this box; the binary is identical. Inside the dev \
image pass --hip-lib /opt/rocm/lib/libamdhip64.so.7 (ROCm is not on the loader path there).";

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Print the device profile (dlopen HIP, no ROCm at build time)
    Probe {
        /// Soname or full path of the HIP runtime to load. Swapping this is the free A/B axis
        /// across the ROCm userspaces on this box; the binary is identical. The bare soname works
        /// wherever the dynamic loader knows about ROCm; inside the saber-dev image it does not
        /// (ROCm is at /opt/rocm and is not on the loader path), so pass
        /// `--hip-lib /opt/rocm/lib/libamdhip64.so.7` there.
        #[arg(long, default_value = DEFAULT_HIP_SONAME)]
        hip_lib: String,

        /// Skip `hipMemGetInfo`. Free VRAM is the one figure that needs a context on each device;
        /// pass this when sharing the box with a server that has claimed nearly all of it.
        #[arg(long)]
        no_free_mem: bool,
    },

    /// Resolve a manifest against the committed artifacts and print every route
    Explain {
        /// The Saber manifest, e.g. examples/qwen38-27b-nvfp4-tp2.toml
        manifest: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml` the registry is derived from
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// An autotune cache to bind against. Without one the cache is cold, which is the P0 state
        /// and prints the untuned work list.
        #[arg(long)]
        autotune: Option<PathBuf>,

        /// Emit JSON instead of the human dump
        #[arg(long)]
        json: bool,
    },

    /// The P0 bit-exact gate: replay the captured reference cases and byte-compare
    Gate {
        /// Where a capture script wrote `cases.txt` and its `.npy` files. Repeatable (comma or
        /// a second `--data`): `--autotune`'s coverage line counts every route the PLAN binds,
        /// and the plan's non-GEMM routes are split across `capture-saber` and `capture-glue`.
        #[arg(long, value_delimiter = ',', default_value = ".cache/capture")]
        data: Vec<PathBuf>,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Only run cases whose task line contains this substring
        #[arg(long)]
        only: Option<String>,

        /// Refuse a case whose device buffers exceed this. Prod owns the card.
        #[arg(long, default_value_t = 520)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// An autotune cache. The gate then replays the variant the CACHE binds for a shape
        /// instead of the one `resolve` picks, and reports every bound shape no case covered.
        #[arg(long)]
        autotune: Option<PathBuf>,

        /// The manifest the cache is read against. Required with `--autotune`.
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// With `--autotune`: write the capture case line for every bound GEMM shape here and
        /// stop, so `the internal reference harness` can capture exactly what the plan binds.
        #[arg(long)]
        emit_cases: Option<PathBuf>,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// P4: the served stream against the rows the head wrote — replay every sampled row through
    /// the shipped sampler, and draw from a few of them for a distribution test
    SampleCheck {
        /// The JSONL `saber serve --sample-dump` wrote
        #[arg(long)]
        rows: PathBuf,

        /// Draws per histogram row
        #[arg(long, default_value_t = 10_000)]
        draws: u64,

        /// How many of the dumped rows to draw from
        #[arg(long, default_value_t = 8)]
        histogram_rows: usize,

        /// Where to write the histograms, for `tools/gates/sampler_stream.py` to test
        #[arg(long)]
        out: Option<PathBuf>,

        /// Where to write every seam row's truncated distribution p' (ws-f41), for
        /// `the internal reference harness` to compare with prod's sampler row for row
        #[arg(long)]
        law_out: Option<PathBuf>,
    },

    /// ws-v8: a `/metrics` scrape against the census a `saber serve --engine real` printed at
    /// shutdown, counter for counter (every `saber:` census family, by its def's pattern)
    CensusParity {
        /// The scrape (`curl localhost:<port>/metrics`)
        #[arg(long)]
        metrics: PathBuf,

        /// The server's stdout, census block included
        #[arg(long)]
        census: PathBuf,

        /// Counters that move while the server idles between the two readings (repeatable)
        #[arg(long)]
        ignore: Vec<String>,
    },

    /// C2: the RAGGED decode attention's card gate — (a) bit-exact against the uniform kernel at
    /// uniform widths, (b) ragged widths against a host f64 reference with two negative controls
    /// (the uniform kernel fed the ragged batch, an off-by-one `cu_q`), (c) every slot alone at
    /// its own width byte-identical to its rows in the batch
    RaggedAttnGate {
        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Refuse a case whose device buffers exceed this
        #[arg(long, default_value_t = 2048)]
        budget_mib: u64,

        /// Write the verdict JSON here
        #[arg(long)]
        out: Option<PathBuf>,

        /// Seed of the synthetic inputs
        #[arg(long, default_value_t = 0xC2)]
        seed: u64,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// P4: the sampler seam's gate — `saber_device_row_bf16` against the host rule, its two
    /// log-sum-exps in ULPs, and the argmax the row implies against `saber_argmax_bf16`
    DeviceRowGate {
        /// Where `tools/capture/capture_device_row.py` wrote `cases.txt` and its `.npy` files
        #[arg(long, default_value = ".cache/capture-device-row")]
        data: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Refuse a case whose device buffers exceed this. Prod owns the card.
        #[arg(long, default_value_t = 520)]
        budget_mib: u64,

        /// The largest log-sum-exp deviation, in f32 ULPs, this gate accepts. Not a knob to turn
        /// when a run is red: the number it prints is the measurement.
        #[arg(long, default_value_t = 2)]
        lse_ulp: u32,

        /// Vocabulary shards to split each captured row into. 1 is this card alone; 2 runs the
        /// TP=2 seam — a rank per card, the follower's half over the `tp2` wire, and the shipped
        /// merge law against the one-rank gate's own reference
        #[arg(long, default_value_t = 1)]
        shards: u32,

        /// Set on a CHILD of `--shards 2`: which rank this process is. The parent spawns two
        #[arg(long)]
        rank: Option<u32>,

        /// Set on a child of `--shards 2`: the loopback port rank 0 listens on
        #[arg(long)]
        port: Option<u16>,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// P1: run one decoder block (layers 0-4) and byte-compare it against the reference chain
    Layer {
        /// Where `the internal reference harness` wrote `block.json` and its `.npy` files
        #[arg(long, default_value = ".cache/capture-layer")]
        data: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// The tier pack directory. The block's already-device-final records (the norms and the
        /// GDN conv filter) are bound out of a `saber-mem::upload` arena by record name.
        #[arg(long)]
        pack: Option<PathBuf>,

        /// TP rank of the pack
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// Only run the case whose name contains this substring (`decode` / `prefill`)
        #[arg(long)]
        only: Option<String>,

        /// Print the provenance table: every weight record, its layout id, and the checkpoint
        /// tensors it was built from
        #[arg(long)]
        provenance: bool,

        /// NEGATIVE CONTROL: bind this record at the wrong layout id. The block must refuse it
        /// by name rather than launching a kernel over bytes that mean something else.
        #[arg(long)]
        corrupt_layout: Option<String>,

        /// Upload the whole rank rather than the window this block needs. What a serving load
        /// does; it needs the card to itself (a device-final rank is 9.48 GB).
        #[arg(long)]
        full_rank: bool,

        /// Refuse to start if the block's device buffers would exceed this. Prod owns the card.
        #[arg(long, default_value_t = 800)]
        budget_mib: u64,

        /// Replay the block and report µs/block
        #[arg(long)]
        timing: bool,

        /// Timed replays
        #[arg(long, default_value_t = 50)]
        iters: usize,

        /// Replays discarded before timing starts
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// One whole forward pass: token ids in, a token id out, every weight from the pack
    Forward {
        /// The TP=1 device-final tier pack (directory or `rank<r>.pack`)
        #[arg(long)]
        pack: PathBuf,

        /// TP rank of the pack
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`
        #[arg(long, default_value = ".cache/template_corpus.jsonl")]
        prompt_ids: PathBuf,

        /// Row of the corpus to take
        #[arg(long, default_value_t = 0)]
        prompt_index: usize,

        /// Run this many corpus rows in ONE process instead of one prompt. The weights load once;
        /// each prompt then costs only its own arenas, which is what makes a 200-prompt agreement
        /// gate minutes rather than hours
        #[arg(long, default_value_t = 1)]
        prompts: usize,

        /// Skip corpus rows longer than this when --prompts selects them. It must match the
        /// reference capture's own limit, or the two sides compare different lists
        #[arg(long, default_value_t = 4096)]
        max_prompt: u32,

        /// Run BOTH `_nq` fusion arms at --tp 2 and compare each against the reference. The
        /// fusion needs the EXACT all-reduce family, so at a prompt's row count it is only
        /// reachable together with --ar-exact (ws-h23)
        #[arg(long)]
        both_arms: bool,

        /// Force every all-reduce at --tp 2 to the EXACT family, by raising the rules'
        /// exact/lossy threshold above the largest message the run sends. Saber is then lossless
        /// while the no-speculation reference stays lossy, which is what says whether the
        /// residual in the token gate's undecided tail is prod's reduce (ws-h24)
        #[arg(long)]
        ar_exact: bool,

        /// At --tp 2, write the residual stream after every decoder layer (one row per prompt,
        /// the last position, f32) into this directory, for a per-layer comparison against a
        /// captured reference
        #[arg(long)]
        dump_layers: Option<PathBuf>,

        /// Run only these corpus row ids out of the ones --prompts selects, comma separated. A
        /// per-layer comparison is about named prompts, not about a count
        #[arg(long, value_delimiter = ',')]
        only_ids: Vec<u64>,

        /// A `tools/capture/capture_nospec.py` reference to compare the greedy tokens against
        #[arg(long)]
        compare: Option<PathBuf>,

        /// Truncate the prompt to this many tokens (0 = the whole prompt)
        #[arg(long, default_value_t = 0)]
        tokens: u32,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Refuse to start if the pass's device buffers would exceed this
        #[arg(long, default_value_t = 24576)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Replay the pass and report µs/forward
        #[arg(long)]
        timing: bool,

        /// Timed replays
        #[arg(long, default_value_t = 20)]
        iters: usize,

        /// Replays discarded before timing starts
        #[arg(long, default_value_t = 3)]
        warmup: usize,

        /// Refuse rather than run a pass whose kernel set is incomplete
        #[arg(long)]
        require_complete: bool,

        /// Read this pass's own logit row back and report ITS top-2 ids and margin beside the
        /// reference's. A flip can only be attributed with BOTH margins (workstream P8)
        #[arg(long)]
        top2: bool,

        /// Tensor-parallel world size. 2 runs the pass across BOTH cards as two processes, with
        /// every all-reduce executed for real through the launcher's peer-visible workspaces.
        #[arg(long, default_value_t = 1)]
        tp: u32,

        /// TP=2 only: this process's rank. Omit it and the process spawns both ranks itself.
        #[arg(long = "rank-of", value_name = "RANK")]
        tp_rank: Option<u32>,

        /// TP=2 only: the loopback port the two ranks meet on (the parent picks one).
        #[arg(long)]
        port: Option<u16>,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// T9: the GDN prefill repeat-and-compare harness (one card; a diagnostic)
    GdnRace {
        /// The device-final tier pack (directory or `rank<r>.pack`)
        #[arg(long, default_value = "/packs/qwen38-27b-nvfp4-devfinal-fp8head")]
        pack: PathBuf,

        /// The pack's TP world size
        #[arg(long, default_value_t = 2)]
        world: u32,

        /// TP rank of the pack
        #[arg(long, default_value_t = 1)]
        rank: u32,

        /// Plan layers 0..LAYERS
        #[arg(long, default_value_t = 12)]
        layers: u32,

        /// The prefill rung (bucket rows)
        #[arg(long, default_value_t = 384)]
        rows: u32,

        /// The chunk's live rows
        #[arg(long, default_value_t = 321)]
        live: u32,

        /// A JSON id array or a `.jsonl` corpus (default: synthetic ids)
        #[arg(long)]
        prompt_ids: Option<PathBuf>,

        /// Corpus row
        #[arg(long, default_value_t = 0)]
        prompt_index: usize,

        /// Reps (rep 0 is the reference)
        #[arg(long, default_value_t = 1000)]
        reps: u64,

        /// GDN state dtype: f16 (the default) or f32 (the control)
        #[arg(long, default_value = "f16")]
        state: String,

        /// `step` (a digest after every op) or `end` (the chunk undisturbed, one digest)
        #[arg(long, default_value = "step")]
        digest: String,

        /// Progress line every N reps
        #[arg(long, default_value_t = 1000)]
        every: u64,

        /// Report label
        #[arg(long, default_value = "gdnrace")]
        label: String,

        /// Idle milliseconds before every rep (a load step each chunk)
        #[arg(long, default_value_t = 0)]
        gap_ms: u64,

        /// NEGATIVE CONTROL: change one non-tail id on every odd rep (must report every one)
        #[arg(long)]
        control: bool,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Device buffers budget
        #[arg(long, default_value_t = 12288)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// The P1 decoder block as ONE captured graph: replay it, byte-compare it, and time it
    /// against the eager path
    BlockCapture {
        /// Where `the internal reference harness` wrote `block.json` and its `.npy` files
        #[arg(long, default_value = ".cache/capture-layer")]
        data: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// The device-final tier pack every weight is bound out of
        #[arg(long, default_value = "/packs/qwen38-27b-nvfp4-devfinal")]
        pack: PathBuf,

        /// TP rank of the pack
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// Only run the case whose name contains this substring (`decode` / `prefill`)
        #[arg(long)]
        only: Option<String>,

        /// Replays to byte-compare, each after the state is restored
        #[arg(long, default_value_t = 3)]
        replays: usize,

        /// Timed graph/eager pairs
        #[arg(long, default_value_t = 50)]
        iters: usize,

        /// Untimed pairs first
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// Refuse to start if the block's device buffers would exceed this
        #[arg(long, default_value_t = 800)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// The whole DECODE step as one captured graph: replay it, byte-compare it, and time it
    /// against the eager step and the P-1b prod comparator
    StepCapture {
        /// The TP=1 device-final tier pack
        #[arg(long, default_value = "/packs/qwen38-27b-nvfp4-devfinal-tp1")]
        pack: PathBuf,

        /// P27-16: the decode verify width `--rows` is cut by (8 = the served b1s8 / b8s8;
        /// default the graph's own 5, prod's old b1s5)
        #[arg(long)]
        verify_width: Option<u32>,

        /// TP rank of the pack
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`
        #[arg(long, default_value = ".cache/prompt64.json")]
        prompt_ids: PathBuf,

        /// Row of the corpus to take
        #[arg(long, default_value_t = 0)]
        prompt_index: usize,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Rows per step: 5 is b1s5, 40 is b8s5. Repeatable.
        #[arg(long = "rows", default_values_t = [5u32])]
        rows: Vec<u32>,

        /// Prefill chunk lengths to run instead: 2048 is `p2048`, 8192 is `p8192`. Repeatable.
        ///
        /// Pass `--rows` with no value to run prefill only (`--rows` defaults to 5 otherwise).
        #[arg(long = "prefill")]
        prefill: Vec<u32>,

        /// Sequences whose chunks share ONE prefill step.
        ///
        /// `--prefill 1024 --prefill-seqs 2` is two sequences of 1024 in a single 2048-row step.
        /// 1 is the shape the engine runs today, one chunk per step.
        #[arg(long = "prefill-seqs", default_value_t = 1)]
        prefill_seqs: u32,

        /// Plan the head on EVERY row instead of the sampled ones, as one combined plan.
        ///
        /// The old shape of this command, kept so the old figure stays reproducible. A prefill
        /// chunk's head belongs on one row per sequence; on every row it computes 2048 x 248320
        /// logits and reads one of them, which is ~26 ms of a p2048 step.
        #[arg(long = "head-on-every-row")]
        head_on_every_row: bool,

        /// Context band the paged KV cache is sized for
        #[arg(long, default_value_t = 8192)]
        ctx: u32,

        /// Refuse to start if the step's device buffers would exceed this
        #[arg(long, default_value_t = 30000)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Replays to byte-compare against the eager step
        #[arg(long, default_value_t = 3)]
        replays: usize,

        /// Timed graph/eager pairs
        #[arg(long, default_value_t = 50)]
        iters: usize,

        /// Untimed pairs first
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// Capture the step as a chain of per-layer sub-graphs and time each layer IN PLACE, with
        /// the previous layers' weights having evicted the Infinity Cache
        #[arg(long)]
        per_layer: bool,

        /// Rebuild the step with one family's ops omitted, per family, for each family's MARGINAL
        /// cost. The outputs are meaningless; the timing is the point
        #[arg(long)]
        ablate: bool,

        /// Time every op as its own sub-graph, in place, and report achieved GB/s per GEMM shape
        /// against the DRAM floor
        #[arg(long)]
        per_op: bool,

        /// With --per-op: detail every link whose op name contains this, with its neighbours and
        /// the same op timed eagerly in place, so the LINK's own cost is a measurement
        #[arg(long = "detail")]
        capture_detail: Option<String>,

        /// Measured DRAM peak for the weight-stream floor (635 GB/s on this box, not the 576 the
        /// startup sweep reports)
        #[arg(long, default_value_t = 635.0)]
        dram_gbs: f64,

        /// The arithmetic peak for the compute floor, TFLOP/s. RDNA4's fp8 WMMA is 325, and the
        /// MXFP4 kernels upconvert into that path; it is what the `bound` column compares against.
        #[arg(long, default_value_t = 325.0)]
        peak_tflops: f64,

        /// FN63: after the HIP replays, replay the step as an HSA AQL packet array beside the
        /// HIP graph (bytes, per-node overhead, host CPU, wake latency, SDMA) and write DIR/fn63.txt.
        #[arg(long)]
        fn63: Option<PathBuf>,

        /// FN63: timed reps per arm
        #[arg(long, default_value_t = 200)]
        fn63_reps: usize,

        /// FN63: reps per interleaved block
        #[arg(long, default_value_t = 25)]
        fn63_block: usize,

        /// FN63: the ROCr runtime
        #[arg(long, default_value = saber_hsa::DEFAULT_HSA_SONAME)]
        hsa_lib: String,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// The TP=2 decode step as ONE graph per rank, its 129 all-reduces captured as nodes
    StepTp2 {
        /// The TP=2 device-final tier pack directory
        #[arg(long, default_value = "/packs/qwen38-27b-nvfp4-devfinal")]
        pack: PathBuf,

        /// Plan the head on EVERY row instead of the sampled ones, as one combined plan.
        ///
        /// The pre-N6 shape of this command, kept so the old figures stay reproducible.
        #[arg(long = "head-on-every-row")]
        head_on_every_row: bool,

        /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`
        #[arg(long, default_value = ".cache/prompt64.json")]
        prompt_ids: PathBuf,

        /// Row of the corpus to take
        #[arg(long, default_value_t = 0)]
        prompt_index: usize,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Rows per step: at the graph's verify width, 5 is b1s5 and 40 is b8s5. Repeatable
        #[arg(long = "rows", default_values_t = [5u32])]
        rows: Vec<u32>,

        /// Decode ROW BANDS to run as well (C2), in rows: each a `b{slots}r{rows}` band at the
        /// graph's max width over `--ragged-widths`' slots, the ragged attention pair included.
        /// Repeatable; none by default
        #[arg(long = "ragged-rows")]
        ragged_rows: Vec<u32>,

        /// The per-slot widths every `--ragged-rows` band runs (C2): a realistic conc-8 multiset
        /// under prod's cap by default. Their sum must fit each band
        #[arg(
            long = "ragged-widths",
            value_delimiter = ',',
            default_value = "5,5,5,3,5,2,5,4"
        )]
        ragged_widths: Vec<u32>,

        /// The graph's verify width — `1 + SPEC`, so prod's SPEC=7 is 8. It is what `--rows`
        /// resolves against (`rows / width` is the batch), what the GDN conv slot and
        /// `gdn_shape`'s strides are derived from, and therefore the knob that makes the
        /// conc-8 width question askable at all: at width 5, `--rows 40` is b8s5; at width 8 it
        /// is b5s8 and `--rows 64` is b8s8
        #[arg(long, default_value_t = saber_model::graph::VERIFY_WIDTH)]
        spec_width: u32,

        /// The GDN recurrent state's STORED dtype (C1): `f16` (the serving default since N8,
        /// 2026-09-23) or `f32` (the control). Forwarded to the child rank by value
        #[arg(long, default_value = "f16", value_parser = ["f32", "f16"])]
        gdn_state_dtype: String,

        /// Prefill chunk lengths to run instead: 2048 is `p2048`, 8192 is `p8192`. Repeatable.
        #[arg(long = "prefill")]
        prefill: Vec<u32>,

        /// Context band the paged KV cache is sized for
        #[arg(long, default_value_t = 8192)]
        ctx: u32,

        /// Refuse to start if a rank's device buffers would exceed this
        #[arg(long, default_value_t = 26000)]
        budget_mib: u64,

        /// Replays to byte-compare against the eager two-rank step
        #[arg(long, default_value_t = 3)]
        replays: usize,

        /// Timed graph/eager pairs
        #[arg(long, default_value_t = 50)]
        iters: usize,

        /// Untimed pairs first
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// Measured DRAM peak for the per-card weight-stream floor
        #[arg(long, default_value_t = 635.0)]
        dram_gbs: f64,

        /// The compute roof the W4A8 kernels upconvert into (RDNA4 fp8 WMMA)
        #[arg(long, default_value_t = 325.0)]
        peak_tflops: f64,

        /// Time every op as its own sub-graph, both ranks in lockstep, and print the non-linear
        /// table with each op's own floor
        #[arg(long)]
        per_op: bool,

        /// H36: at a prefill bucket, byte-compare and time the prefill all-reduce overlap at this
        /// slice row count against the sequential body and the body without its reduces. 0 = off
        #[arg(long, default_value_t = 0)]
        ar_overlap: u32,

        /// P27-1d: with --ar-overlap, the overlapped arm's reduce form -- `oneshot` (H36's, the
        /// control) or `split` (an --ar-push-blocks-block push + a command-processor wait + the
        /// reduce)
        #[arg(long, default_value = "oneshot", value_parser = ["oneshot", "split"])]
        ar_form: String,

        /// P27-1d: --ar-form split's push blocks
        #[arg(long, default_value_t = DEFAULT_AR_PUSH_BLOCKS)]
        ar_push_blocks: u32,

        /// With --per-op: detail every link whose op name contains this, with its neighbours and
        /// the same op timed eagerly in place. Like --per-op itself, it replays one op many times
        /// over live state: the outputs are meaningless afterwards and the timing is the point
        #[arg(long)]
        detail: Option<String>,

        /// C6: plan every prefill bucket of at most this many rows (96..=128) on the decode GEMM
        /// family in the fusion gate's FUSED arm (timed, `--per-op`), the unfused arm on the
        /// committed rules: the gate is then the class comparison. 0 = off
        #[arg(long, default_value_t = 0)]
        small_chunk_gemm: u32,

        /// C6: `exact` (every small-rung GEMM at one split: the folded kernel's summation order)
        /// or `band` (the reference's M > 64 splits: a numerics-class change, the control)
        #[arg(long, default_value = "exact", value_parser = ["exact", "band"])]
        small_chunk_class: String,

        /// G5: plan the fusion gate's FUSED arm with the folded GEMM at the mid-M tile heights
        /// (`[[folded_mt]]`, the `folded_mt` knob on), the unfused arm on the committed rules:
        /// the gate is then the class comparison at every prefill bucket in a band
        #[arg(long)]
        mid_chunk_gemm: bool,

        /// G6: plan the fusion gate's FUSED arm under the layout law's mid band (the `atiled_mid`
        /// knob on: tiled producers and the A-tiled family from `atiled_mid_min_m` rows,
        /// `[[atiled_mt]]`), the unfused arm on the committed rules (the class comparison)
        #[arg(long)]
        mid_chunk_atiled: bool,

        /// G6: the fusion gate's FUSED arm runs the 48-wide gate on the skinny-N body (`[[skinny]]`)
        #[arg(long)]
        skinny_gate: bool,

        /// Plan without slot reuse: the fusion gate's two plans then cannot share an arena
        /// buffer, which is how a comparison difference is told apart from an arithmetic one
        #[arg(long)]
        no_slot_reuse: bool,

        /// Which fusions the fused arm of the fusion gate plans: `both` / `all` (every
        /// launch-train fusion, what `--decode-fusion` serves: the `_nq` epilogue, the decode GEMM
        /// pair, fusion-map row 3 and the rope + KV append; row 6 is NOT in it: measured slower),
        /// `h37` (the `_nq` epilogue and the pair), `nq`, `pair`, `gemm-ar` (row 3 -- the split
        /// decode GEMM + exact reduce + `_nq` epilogue in one dispatch -- with the plain `_nq`,
        /// ws-h38), `rope-append` (`qk_norm_rope` + `kv_append` in one dispatch) and
        /// `attn-combine` (the decode attention's combine in its last-arriving split block)
        #[arg(long, default_value = "both")]
        fusion: String,

        /// The fusion gate's NEGATIVE CONTROL (ws-h37), which must read MISMATCH: `pair-drop` never
        /// issues the pair's second half (caught only through the poison of folded outputs),
        /// `nq-eps` runs the fused epilogue at ten times its epsilon; the WEAK ones are `pair-ks`
        /// (the second half at the first half's split) and `nq-verbatim` (the reference's contract-off
        /// `_nq`). ws-h38, all decisive: `gemmar-eps` (row 3's epilogue at ten times epsilon),
        /// `gemmar-rows` (the last row gets no epilogue owner, so its outputs keep the poison),
        /// `rope-append-drop` (the fused append never writes V; the step's KV slots are poisoned
        /// in state first) and `combine-drop` (the last arriver merges every segment but the
        /// last). `none` is the gate itself
        #[arg(long, default_value = "none")]
        fusion_control: String,

        /// This process's rank. Omit it and the process spawns both ranks itself
        #[arg(long = "rank-of", value_name = "RANK")]
        rank_of: Option<u32>,

        /// The loopback port the two ranks meet on (the parent picks one)
        #[arg(long)]
        port: Option<u16>,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// P5's drafter side: the DFlash2 drafter as a PLAN -- every launch declared, every route
    /// resolved through the kernel plane, the count checked against the budget -- and, with a live
    /// capture, its selector run through the launcher and byte-compared against prod's own output
    DraftStep {
        /// Ranks the shapes are sharded for
        #[arg(long, default_value_t = 2)]
        tp: u32,

        /// `num_speculative_tokens`; the drafter's block is one more than this
        #[arg(long, default_value_t = 7)]
        spec: u32,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// A live-serve selector capture (`the internal reference harness`). Without it this
        /// is the plan and the routes only, and needs no device.
        #[arg(long)]
        sel_data: Option<PathBuf>,

        /// A proposal capture. TWO KINDS reach this flag and the directory says which:
        /// `capture_dflash_ctx.py`'s (a `ctx.json`) runs the context-KV sub-plan through the real
        /// launcher in prod's own order; `proposal_capture.sh`'s (an `index.json`) is one whole
        /// decode step of prod's `DFlash2Speculator.propose` and drives the staged k=7 gate --
        /// the exact rerank, the candidate selector, the walk and the whole-proposal us. The
        /// second needs `--pack` as well, because every launch after the context store reads a
        /// drafter weight.
        #[arg(long)]
        proposal: Option<PathBuf>,

        /// A DRAFTER PACK directory (`build_pack.py --layout dflash`): check every record the plan
        /// declares against the pack's own table, and -- with a device -- upload the rank and run
        /// every launch whose operands are a pack weight and an activation
        #[arg(long)]
        pack: Option<PathBuf>,

        /// Which rank of `--pack` to check and load
        #[arg(long, default_value_t = 0)]
        pack_rank: u32,

        /// `--pack`: upload the rank and dispatch the launches that read it. Without it the pack
        /// arm is the RECORD gate alone, which needs no device
        #[arg(long)]
        pack_launches: bool,

        /// Positions the pack's rope table covers. It must match the value the pack was BUILT
        /// with: the record's length is a function of it, so a mismatch is a shape mismatch and
        /// the gate says so rather than loading a shorter table.
        #[arg(long, default_value_t = 4096)]
        rope_max_pos: u32,

        /// Timed passes over `--pack`'s weight-reading launches, or over the whole proposal's
        /// covered launch set when `--proposal` names a proposal capture
        #[arg(long, default_value_t = 30)]
        iters: u32,

        /// `--proposal`: run the chain instead of the seeded stages. Every launch reads what the
        /// previous one wrote and only prod's own INPUTS come from the capture, which is the one
        /// construction a missing producer cannot hide from
        #[arg(long)]
        no_seed: bool,

        /// `--proposal --no-seed`: also write the chain's own intermediates here, one directory
        /// per prompt, under the engine's arena-region names. It is the reference
        /// `saber engine --draft-replay --draft-chain-dump` compares the ENGINE's plumbing
        /// against, and the only comparison of the two that admits no rounding class
        #[arg(long)]
        dump_chain: Option<PathBuf>,

        /// `--proposal --no-seed`: seed the drafter's KV cache with ZEROS rather than prod's
        /// `kv.before_ctx`. That is exactly what Saber's engine holds at its first decode step,
        /// because prod writes the drafter's context K/V at every target step INCLUDING the
        /// prefill chunks and Saber's engine proposes only on a decode step — so this arm prices
        /// the prompt's context in the block it produces (ws-s8)
        #[arg(long)]
        empty_draft_cache: bool,

        /// List the launches that are not code objects yet, with what each needs
        #[arg(long)]
        explain: bool,

        /// Refuse to start if the device buffers would exceed this
        #[arg(long, default_value_t = 26000)]
        budget_mib: u64,

        #[arg(long, default_value_t = 1)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// P5's target side: a decode step whose rows are DRAFT tokens, gated against the chain of
    /// single-row steps it has to equal
    VerifyStep {
        /// The device-final tier pack: the TP=1 pack at `--tp 1`, the TP=2 directory at `--tp 2`
        #[arg(long, default_value = "/packs/qwen38-27b-nvfp4-devfinal-tp1")]
        pack: PathBuf,

        /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`
        #[arg(long, default_value = ".cache/prompt64.json")]
        prompt_ids: PathBuf,

        /// Row of the corpus to take
        #[arg(long, default_value_t = 0)]
        prompt_index: usize,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Context band the paged KV cache is sized for
        #[arg(long, default_value_t = 8192)]
        ctx: u32,

        /// Draft width: rows of the verify step. 5 is the b1s5 bucket
        #[arg(long, default_value_t = 5)]
        spec: u32,

        /// Refuse to start if the device buffers would exceed this
        #[arg(long, default_value_t = 26000)]
        budget_mib: u64,

        /// HIP device, at TP=1
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Tensor parallelism: 1 (one process) or 2 (two, one card each)
        #[arg(long, default_value_t = 1)]
        tp: u32,

        /// This process's rank at TP=2. Omit it and the process spawns both ranks itself
        #[arg(long = "rank-of", value_name = "RANK")]
        rank_of: Option<u32>,

        /// The loopback port the two ranks meet on (the parent picks one)
        #[arg(long)]
        port: Option<u16>,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,

        /// P27-W2-5: arm A on the GDN REPLAY form (base + stash + replay) against the
        /// snapshot-law chain, both on the f16 stored state
        #[arg(long)]
        gdn_replay: bool,
    },

    /// The two-rank all-reduce gate: both cards, IPC scratch, through the engine's Launcher
    ArGate {
        /// The case list the reference all-reduce gate (`the internal reference harness`) writes
        #[arg(long, default_value = ".cache/capture-saber/ar_small_cases.txt")]
        cases: PathBuf,

        /// The rules file it writes alongside, with the IPC slot shrunk to fit beside prod.
        /// Without one the production slot is used, which does not fit next to a 0.95-utilisation
        /// server.
        #[arg(long)]
        rules: Option<PathBuf>,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Only run cases whose line contains this substring
        #[arg(long)]
        only: Option<String>,

        /// Refuse a case whose per-rank device buffers exceed this
        #[arg(long, default_value_t = 128)]
        budget_mib: u64,

        /// The relative-RMS bound a WHT-quantised case must stay inside.
        ///
        /// MEASURED on this harness's uniform [-1, 1) payload, 2026-09-18: 2.47% at 6 bits,
        /// 10.87% at 4. (The ledger's 1.6% / 7.3% are the same kernels on real activations; a
        /// uniform payload has no small values for the per-group scale to favour, so it reads
        /// higher. The ordering and the magnitude agree.) 0.15 admits the wider of the two with
        /// ~40% of margin and still rejects a missed contribution, which lands near 70%.
        #[arg(long, default_value_t = 0.15)]
        quant_rel_rms: f64,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,

        /// Internal: which rank this process is. The parent spawns two children with it.
        /// `hipIpcOpenMemHandle` only opens handles from ANOTHER process, so the two ranks cannot
        /// be two threads.
        #[arg(long, hide = true)]
        rank: Option<u32>,

        /// Internal: the loopback port rank 0 listens on.
        #[arg(long, hide = true)]
        port: Option<u16>,

        /// Time every case as well as checking it, and print one `AR-TIMING` line per case
        #[arg(long)]
        timing: bool,

        /// Untimed rounds before timing starts
        #[arg(long, default_value_t = 5)]
        warmup: usize,

        /// Timed rounds; each is preceded by a rendezvous so peer skew cannot inflate it
        #[arg(long, default_value_t = 20)]
        samples: usize,

        /// Back-to-back launches per timed round. Above 1 the ranks self-synchronise, so the
        /// arrival skew is paid once and amortised instead of landing in every sample.
        #[arg(long, default_value_t = 10)]
        iters: usize,

        /// Shrink the IPC slot to this many bytes (prod's is 84 MiB, and two of them plus the
        /// local pack are allocated per rank)
        #[arg(long)]
        ar_slot_bytes: Option<u64>,

        /// NEGATIVE CONTROL: desynchronise this rank's slot parity on purpose. The gate must
        /// refuse before launching; without the check both ranks would spin forever.
        #[arg(long)]
        desync_rank: Option<u32>,

        /// T4: run every case through the all-reduce's RACE-CHECK twin (`_chk`), check its bytes
        /// against the host reference AND the served kernel's, and read its diagnostic words
        /// (payload mismatches, flag drift, spin timeouts must all be 0)
        #[arg(long)]
        race_check: bool,

        /// T4 NEGATIVE CONTROL: this rank runs the SERVED kernel while its peer runs the twin, so
        /// the twin's receiver finds no check word beside the data: every block must count a
        /// mismatch that never converges, and the reduced bytes must still be right
        #[arg(long)]
        race_control_rank: Option<u32>,
    },

    /// Does a captured graph compute what eager launches compute?
    CaptureGate {
        /// Capture directories to run, read with the bit-exact gate's own case loader
        #[arg(long, value_delimiter = ',', default_values_t = [
            String::from(".cache/capture"), String::from(".cache/capture-saber")])]
        data: Vec<String>,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Only run cases whose line contains this substring
        #[arg(long)]
        only: Option<String>,

        /// Per-case limit, and the cap on the sequence graph's running total
        #[arg(long, default_value_t = 520)]
        budget_mib: u64,

        /// A case joins the sequence graph only if its own buffers are at most this. The
        /// sequence test measures node count and workspace sharing, so many small cases are worth
        /// more than one 136 MiB prefill case.
        #[arg(long, default_value_t = 12)]
        sequence_case_mib: u64,

        /// Replays of the sequence graph. Three is the minimum that can catch a Persistent
        /// workspace being re-zeroed: pass 1 would pass either way.
        #[arg(long, default_value_t = 3)]
        replays: usize,

        /// Timed replays for the per-node figure. Zero skips the timing.
        #[arg(long, default_value_t = 50)]
        timed: usize,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// How much does a DEPENDENT node cost over and above its kernel? (P6 sizing)
    ChainBench {
        /// Layers of the 9-op decoder cycle. 64 is the model's real depth.
        #[arg(long, default_value_t = 64)]
        layers: u32,

        /// Timed interleaved graph/eager pairs
        #[arg(long, default_value_t = 50)]
        passes: usize,

        /// Solo event-bracketed launches per distinct kernel
        #[arg(long, default_value_t = 30)]
        solo: usize,

        /// Refuse if the chain's buffers would exceed this
        #[arg(long, default_value_t = 520)]
        budget_mib: u64,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Measure only the null arms (skip the real decoder chain and its ~96 MiB)
        #[arg(long)]
        nulls_only: bool,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// The device-to-device copy binding, on a card: a known pattern back byte-exact, and the
    /// out-of-range and overlapping copies refused (V4, the sequence-group fork's copy)
    D2dGate {
        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Bytes per region
        #[arg(long, default_value_t = 1 << 20)]
        bytes: u64,

        /// Regions in each buffer
        #[arg(long, default_value_t = 4)]
        regions: u64,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// Load one rank's tier pack into VRAM through the pinned staging ring
    Load {
        /// The `rank<r>.pack` file, or the pack directory holding it
        pack: PathBuf,

        /// TP rank to load. The pack's header must agree; a mismatch is refused.
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// First table index to upload. The NVFP4 tier's first 337 records are 4-byte scalars
        /// and record 337 is the 1.21 GiB `embed_tokens`, so a byte limit alone selects either
        /// ~0 bytes or ≥1.2 GiB; start past it to measure ordinary weight records.
        #[arg(long, default_value_t = 0)]
        from_record: usize,

        /// Upload only the records from `--from-record` that fit in this many bytes (`768M`, `2G`).
        /// Bare K/M/G are binary. Prod holds both cards at 0.95 utilisation, so a full 12 GB rank
        /// does not fit beside it.
        #[arg(long, value_parser = load::parse_size)]
        limit_bytes: Option<u64>,

        /// Where in this rank's record window to start reading, as a fraction of it; it wraps at
        /// the end. Unset reads front to back, which measured fastest; staggering the two rank
        /// loaders by r/R was measured and lost (docs/LEDGER.md, workstream K6)
        #[arg(long)]
        read_offset: Option<f64>,

        /// Use the staging ring asked for instead of sizing it from the pack. A benchmark arm: the
        /// load path raises the ring to 3x the largest record, because one record's sha256 cannot
        /// be parallelised and the readers otherwise stall behind it
        #[arg(long)]
        fixed_ring: bool,

        /// Pinned staging slots. Allocated once; the transfer path never allocates.
        #[arg(long, default_value_t = 16)]
        slots: usize,

        /// Bytes per staging slot
        #[arg(long, default_value = "8M", value_parser = load::parse_size)]
        slot_bytes: u64,

        /// `O_DIRECT` reads in flight. 8 is the drive's knee; 16 costs the hash pool its cores.
        #[arg(long, default_value_t = 8)]
        qd: usize,

        /// Bytes per `preadv`
        #[arg(long, default_value = "1M", value_parser = load::parse_size)]
        chunk_bytes: u64,

        /// Threads checking `sha256`
        #[arg(long, default_value_t = 8)]
        hash_threads: usize,

        /// Host bytes the pack stream may hold in read buffers
        #[arg(long, default_value = "2G", value_parser = load::parse_size)]
        stream_budget: u64,

        /// Skip `sha256`. A benchmark mode: a corrupt record would then reach VRAM.
        #[arg(long)]
        no_verify: bool,

        /// VRAM the byte ledger is told this rank has. Deliberately not `hipMemGetInfo`: with prod
        /// resident it reports 0 free while a 520 MiB allocation still succeeds.
        #[arg(long, default_value = "1G", value_parser = load::parse_size)]
        vram_budget: u64,

        /// Pinned host memory the byte ledger is told the process has
        #[arg(long, default_value = "4G", value_parser = load::parse_size)]
        pinned_budget: u64,

        /// Skip the read-only pass that measures the drive's own rate for the same records
        #[arg(long)]
        no_io_pass: bool,

        /// Copies allowed in flight at once. Defaults to one per staging slot. Slots bound how
        /// far the readers run ahead; events bound how deep the copy queue gets, which is a tail
        /// at the end of the load where no read is left to overlap it.
        #[arg(long)]
        events: Option<usize>,

        /// Re-read this many uploaded records out of VRAM and check their `sha256` against the
        /// table. The host tests prove the pipeline asks for the right copies; this proves the
        /// bytes landed, at the right offset, on hardware.
        #[arg(long, default_value_t = 0)]
        check_records: usize,

        /// Negative control for `--check-records`: read each sampled record back from this many
        /// bytes past where it was written. The check must then FAIL; a gate nobody has watched
        /// fail is not a gate (CLAUDE.md rule 8).
        #[arg(long, default_value = "0", value_parser = load::parse_size)]
        check_shift: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// Flash-Next's memory-plane placement intelligence on the host (FN18): the real-traffic
    /// profile from a served route trace, the tier's replay, the residency ceiling, the owner
    /// permutation estimate, the PLE cache sizing. No GPU.
    FlashnextProfile(fn_profile::ProfileArgs),

    /// FN119: the device sampler twin's host oracle -- glibc's expf port vs the linked libm over
    /// all 2^32 patterns (+ the card probe's hash table), the walk + distribution cases with the
    /// real sampler's answers. No GPU.
    Fn119Oracle(flashnext::dstep::OracleArgs),

    /// FN50: add Qwen's first-party fp8 MTP experts to a Flash-Next pack as an ADDITIONAL file per
    /// rank (`rank<r>/mtp_fp8.{pack,json}`; the pack's existing files are never rewritten). No GPU.
    FlashnextPack {
        /// The Flash-Next pack.
        #[arg(
            long,
            default_value = "/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2"
        )]
        pack: PathBuf,
        /// The checkpoint directory holding `model-fp8-mtp-ple.safetensors` + its index.
        #[arg(long, default_value = "/mnt/nvme/models/Qwen3.8-Flash-Next-NVFP4")]
        fp8_ckpt: PathBuf,
        /// What to add: `fp8` (the only form).
        #[arg(long)]
        mtp_experts: Option<String>,
        /// FN61 / FN61b: build the dense-trunk variant `--out-pack` at `w4` (MXFP4 W4: adds
        /// `trunk4.{pack,json}` per directory) or `w8` (fp8 e4m3, 128 x 128 block scales: adds
        /// `trunk8.{pack,json}` per rank; HC stays bf16); the variant links every file of `--pack`.
        #[arg(long)]
        trunk: Option<String>,
        /// FN61: the variant directory (`--trunk`; default `<pack>-trunk4` / `<pack>-trunk8`).
        #[arg(long)]
        out_pack: Option<PathBuf>,
        /// FN108: ADD `shared/hc8.{pack,json}` to `--pack` (the `-trunk8` variant): the target HC
        /// mixing matrices at the reference's hc8m law (e4m3 + f32 row-group scales, 128 / 80).
        #[arg(long)]
        hc8: bool,
        /// FN61: the block-exponent rule, `ocp` (Quark's) or `mse`; FN67: `gptq` = the calibrated
        /// codes of `--calib DIR` (variant `-trunk4c`).
        #[arg(long, default_value = "ocp")]
        w4_rule: String,
        /// FN67: the calibration directory (`the internal reference harness calib`) for `--w4-rule gptq`.
        #[arg(long)]
        calib: Option<PathBuf>,
        /// FN80a: `tp2` = build the TP=2 expert-split variant `--out-pack` (default `<pack>-tp2`):
        /// every rank holds half of EVERY expert (`rank<r>/experts.tp2.xtier`), every other file
        /// linked, then the byte gate; `tp2-check` = the byte gate alone on an existing variant.
        #[arg(long)]
        experts: Option<String>,
        /// FN55: add the vision tower (`bf16`, the only form) as the pack's `vision/` member
        /// (`vision/rank<r>.pack` + `vision/vision.json`, read from `--vision-ckpt`).
        #[arg(long)]
        vision: Option<String>,
        /// FN55: the checkpoint holding Flash-Next's `model.visual.*` records.
        #[arg(
            long,
            default_value = "/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4"
        )]
        vision_ckpt: PathBuf,
        /// FN64e: refresh the pack's recorded lazy graph set (`profile/lazy_runs.rank<r>.json`,
        /// what the default `--fn-graph-lazy-pre recorded` pre-captures) from served boots'
        /// `--fn-graph-lazy-record DIR`s (repeat the flag): the union with the pack's current set,
        /// a run's uses summed, most used first.
        #[arg(long)]
        lazy_runs: Vec<PathBuf>,
        /// FN64e: write the merged set here instead of the pack's `profile/` (the pack untouched).
        #[arg(long)]
        lazy_runs_out: Option<PathBuf>,
        /// FN64e: the suffix the pack's replaced set is kept under (`lazy_runs.rank<r>.json.<s>`).
        #[arg(long, default_value = "prev")]
        lazy_runs_backup: String,
        /// FN64e: drop the pack's current set (the records alone) instead of merging with it.
        #[arg(long)]
        lazy_runs_fresh: bool,
        /// FN144: `fp4-local` = the FP4-Local LOSSLESS path (`flashnext::fp4local`): with
        /// `--dry-run` the plan + header check + byte accounting into `--report`; with
        /// `--layers A:B --out-pack DIR` a small xtier v2 pack (the P3 kernel fixture).
        #[arg(long)]
        source: Option<String>,
        /// FN144: the FP4-Local checkpoint.
        #[arg(long, default_value = "/mnt/nvme/models/Qwen3.8-Flash-Next-FP4-Local")]
        fp4_ckpt: PathBuf,
        /// FN144: plan + account only; no pack byte written.
        #[arg(long)]
        dry_run: bool,
        /// FN144: the layer range `A:B` of the small pack.
        #[arg(long)]
        layers: Option<String>,
        /// FN144: where `--dry-run` writes `plan.json` + `accounting.json`.
        #[arg(long, default_value = "~/saber-cache/ws-fn144")]
        report: PathBuf,
    },

    /// FN150: the engine-path byte test of the NVFP4 expert block on one card (the layer-0
    /// FP4-Local smoke pack through the engine's emitter + executor; dumps for
    /// `tools/capture/fn150/fn150_check.py`).
    #[command(hide = true)]
    Fn150MoeCheck {
        /// The layer-0 FP4-Local smoke pack (FN144).
        #[arg(
            long,
            default_value = "/mnt/nvme/saber-packs-smoke/qwen38-flashnext-fp4local-l0"
        )]
        pack: PathBuf,
        /// `fn150_prep.py`'s inputs (xq.bin, xas.bin, rid.bin).
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "/work/prebuilt/gfx1201")]
        prebuilt: PathBuf,
        #[arg(long, default_value_t = 0)]
        device: i32,
        /// Single-token cases (`b1_<t>`, t < N).
        #[arg(long, default_value_t = 8)]
        singles: u32,
        /// FN152: the rank whose records run (rank<r>/experts.xtier).
        #[arg(long, default_value_t = 0)]
        rank: u32,
        /// FN152: FN150's plain route (FN148's cells only; the control).
        #[arg(long)]
        plain: bool,
        /// FN152: the fused train (front + FN152's NVFP4 back half) on the decode cases.
        #[arg(long)]
        fused: bool,
        /// FN152: 8-token cases (`b8_<k>`, k < N).
        #[arg(long, default_value_t = 0)]
        b8: u32,
        #[arg(long, default_value = "libamdhip64.so")]
        hip_lib: String,
    },

    /// Flash-Next's MoE block over the memory-plane tier on one card, through the real mover
    /// (FN8): the byte gates, the per-path table, the conc-8 insert-policy sweep, the prefill
    /// staging crossover, the PLE staged buffer; `boot` prints the boot lines and the metrics
    /// exposition with no GPU.
    FlashnextMoeBench {
        /// gate | paths | sweep | stage | ple | boot
        #[arg(long, default_value = "gate")]
        mode: String,
        /// The Flash-Next pack (read-only).
        #[arg(
            long,
            default_value = "/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2"
        )]
        pack: PathBuf,
        /// The EP rank whose experts this card holds.
        #[arg(long, default_value_t = 0)]
        rank: u32,
        /// FN3's replay routes (`tools/capture/fn7/fn7_prep.py routes`).
        #[arg(long, default_value = "~/saber-cache/ws-fn7/routes.bin")]
        routes: PathBuf,
        /// The warm-start routing profile (`fn7-route-profile/1`), unless `--xtier-profile`.
        #[arg(
            long,
            default_value = "~/saber-cache/ws-fn7/profile/route_profile.json"
        )]
        profile: PathBuf,
        /// Token counts per step (1 = b1 decode, 8 = conc-8, 64 = conc-8 x 8 rows, >= 512 = a
        /// prefill chunk of the concatenated prompts).
        #[arg(long, value_delimiter = ',', default_value = "1,8,64,2048")]
        tokens: Vec<u32>,
        /// Steps per run (after the warm-up).
        #[arg(long, default_value_t = 400)]
        steps: usize,
        /// Warm-up steps (not timed).
        #[arg(long, default_value_t = 50)]
        warmup: usize,
        /// PLE row ids (`fn7_prep.py ple`: u64 [N][16]).
        #[arg(long, default_value = "~/saber-cache/ws-fn7/ple_fn3.bin")]
        ple_ids: PathBuf,
        /// The checkpoint directory as mounted here (the PLE index names the host path).
        #[arg(long)]
        ckpt_root: Option<PathBuf>,
        /// The sweep grid: `full` (max-inserts 0/8/16/32/64 x read-through 0.25/0.5/1.0) or
        /// `inserts` (max-inserts 0/1/2/4/8 at read-through 0.5).
        #[arg(long, default_value = "full")]
        sweep: String,
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[command(flatten)]
        xtier: xtier_flags::XtierFlags,
        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// Park and restore a synthetic session through the RAM and NVMe tiers, and time it
    ParkBench {
        /// KV bytes in the synthetic session. 4k tokens of fp8 KV at TP=2 is ~64 MiB.
        #[arg(long, default_value = "64M", value_parser = load::parse_size)]
        kv_bytes: u64,

        /// GDN state bytes. 72 MiB per rank at prod's shapes, and the half that makes a parked
        /// session worth its own tier: KV and GDN state are useless apart.
        #[arg(long, default_value = "72M", value_parser = load::parse_size)]
        gdn_bytes: u64,

        /// Timed park/restore cycles per tier
        #[arg(long, default_value_t = 50)]
        iters: usize,

        /// Cycles discarded before timing starts
        #[arg(long, default_value_t = 3)]
        warmup: usize,

        /// Slots in each tier
        #[arg(long, default_value_t = 2)]
        slots: usize,

        /// The parking file. One preallocated `O_DIRECT` file with a slot table.
        #[arg(long, default_value = "/mnt/nvme/saber-park/park.bin")]
        nvme_path: PathBuf,

        /// Pinned host memory the byte ledger is told the process has
        #[arg(long, default_value = "1G", value_parser = load::parse_size)]
        pinned_budget: u64,

        /// NVMe bytes the byte ledger is told the parking file may hold
        #[arg(long, default_value = "4G", value_parser = load::parse_size)]
        nvme_budget: u64,

        /// Keep the parking file instead of deleting it at the end
        #[arg(long)]
        keep_file: bool,

        /// Skip the corrupt-image negative control
        #[arg(long)]
        no_corrupt: bool,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Threads the chunked digest may use; 0 is every logical core
        #[arg(long, default_value_t = 0)]
        hash_threads: usize,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// P0.5: capture N dummy kernels per bucket and time the host chain per replayed step
    BenchSteploop {
        /// Manifest whose buckets are measured. Without one, the default 1x5 and 8x5 are used.
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Launches per captured graph. Prod runs 1084 per step.
        #[arg(long, value_delimiter = ',', default_value = "100,500,1084")]
        launches: Vec<u32>,

        /// Timed replays per row
        #[arg(long, default_value_t = 200)]
        steps: usize,

        /// Replays discarded before timing starts
        #[arg(long, default_value_t = 20)]
        warmup: usize,

        /// Bounded-channel depth between the tokio feeder and the step loop
        #[arg(long, default_value_t = 8)]
        channel_capacity: usize,

        /// Width of the dummy kernel (N % 8, N <= 20480). Lower it to separate the per-node cost
        /// from the kernel's own work.
        #[arg(long, default_value_t = saber_exec::steploop::DUMMY_N)]
        dummy_n: u32,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },

    /// Serve the OpenAI /v1 plane over the real scheduler, with a mock executor or the real engine
    Serve {
        /// Which executor: `mock` (a scripted response, no GPU) or `real` (the whole-model forward
        /// at TP=1 from a device-final pack — the P3 seam)
        #[arg(long, default_value = "mock")]
        engine: String,

        /// The mock script: the text the engine "generates" and how many tokens a step commits.
        /// Required for --engine mock, unused for --engine real.
        #[arg(long)]
        mock: Option<PathBuf>,

        /// --engine real, DIAGNOSTIC: write every SAMPLED row — its seam, its parameters, its
        /// noise key and the token it produced — to this file as JSONL, so a checker can replay
        /// the sampler against the served stream (`tools/gates/sampler_stream.sh`). A greedy row
        /// writes nothing, because nothing sampled it.
        #[arg(long)]
        sample_dump: Option<PathBuf>,

        /// --engine real, THE CONTROL for the stale-anchor law (ws-f35): offer a draft block whose
        /// anchor the step did not commit, which is what this engine did before ws-f35. The law —
        /// the default — drops it, because the device proposes from the ARGMAX and a row with a
        /// temperature then commits the host sampler's draw instead, so the block is a
        /// continuation of a token the sequence never had. Inert at temperature 0 either way.
        #[arg(long)]
        keep_stale_anchor_blocks: bool,

        /// --engine real, THE CONTROL for the drawn acceptance law (ws-f40; ws-q3 put it on the
        /// serve plane): a sampled verify row accepts a draft iff it equals the target's ARGMAX,
        /// only the last row is drawn, and the draw is keyed by the STEP — the engine before
        /// F40. The default is the drawn law (prod's rejection sampler in distribution). Counted
        /// in `step_keyed_rows` on `/metrics`; inert at temperature 0. Never a serving setting.
        #[arg(long)]
        step_keyed_draws: bool,

        /// --engine real, FN52: PREFIX (conversation) CACHING — `on` retains a request's KV
        /// pages + a GDN-state snapshot at its prompt's last whole block and lets a later prompt
        /// that extends it prefill only the tail (the reference server's `--enable-prefix-caching --mamba-cache-mode
        /// align`). `off` = today's engine = the control, and the default.
        /// Prod's bare spelling (`--enable-prefix-caching` with no value) means `on`.
        #[arg(long, value_parser = ["on", "off"], default_value = "off", num_args = 0..=1, default_missing_value = "on")]
        enable_prefix_caching: String,

        /// --engine real, FN52's verify arm: a hit is not adopted; the prompt is prefilled fresh
        /// with a cut at the hit and its state + pages compared byte for byte with the entry.
        #[arg(long)]
        prefix_cache_verify: bool,

        /// --engine real, FN69: PROMPT-LOOKUP speculation -- a per-sequence n-gram index over
        /// the prompt + the committed output; a match of at least `--spec-lookup-min-match`
        /// tokens REPLACES the DFlash2 block with what followed it earlier in the sequence. The
        /// verify step is unchanged (same width): greedy ids are the target's own. `off` = the
        /// control and the default.
        #[arg(long, value_parser = ["on", "off"], default_value = flashnext::run::DEFAULT_SPEC_LOOKUP)]
        spec_lookup: String,

        /// --engine real, FN69: the n-gram lengths indexed (`3..4`: the longest match wins).
        #[arg(long, default_value = "3..4")]
        spec_lookup_ngram: String,

        /// --engine real, FN69: the merge law's bar (match length, extended backwards).
        #[arg(long, default_value_t = 3)]
        spec_lookup_min_match: u32,

        /// --engine real, FN52: host bytes (MiB, per rank) the prefix snapshots may hold.
        #[arg(long, default_value_t = 6144)]
        prefix_cache_host_mib: u64,

        /// --engine real, FN52: prompts shorter than this are not snapshotted.
        #[arg(long, default_value_t = 512)]
        prefix_cache_min_tokens: u32,

        /// --engine real, FN66: WARM PROMPTS -- after the server listens, every conversation head
        /// in FILE (`saber_serve::warm`: a JSON array / JSON lines / plain text = one system
        /// prompt) is rendered through the chat path and prefilled once, so a session's first
        /// turn that begins with it adopts the retained prefix. Turns prefix caching ON (FN52's
        /// policy) and says so at boot.
        #[arg(long)]
        warm_prompts: Option<PathBuf>,

        /// --engine real, THE PAIRING CONTROL for ws-f37: propose INSIDE the decode step, from the
        /// device's argmax, the way this engine did before ws-f37. The default is prod's order —
        /// the proposal runs after the host sampler, from the token each slot COMMITTED — so a
        /// block is never the continuation of a token the sequence did not have. Refused without
        /// a resident drafter (`--engine-draft-pack`), where there is no proposal to move
        #[arg(long)]
        propose_in_step: bool,

        /// --engine real: the TP=1 device-final pack (directory or rank<r>.pack)
        #[arg(long)]
        pack: Option<PathBuf>,

        /// --engine real: ranks in the tier, 1 or 2
        #[arg(long, default_value_t = 1)]
        engine_tp: u32,

        /// --engine real (D3): serve `image_url` parts, the vision tower resident with its arena
        /// sized for this many patches per image (prod's ceiling is 65,536 = 16.7 MP). 0, the
        /// default, is text-only: an image part is refused by name. TP = 1 only until D4.
        #[arg(long, default_value_t = 0)]
        vision_max_patches: u32,

        /// --engine real (D3 / P27-11): serve images with prod's 65,536-patch ceiling.
        /// `--vision` alone = `lazy` (P27-11, the user's word 2026-10-07): the tower's weights stay
        /// in pinned host RAM and stream through the idle prefill arena per image (no VRAM billed,
        /// the text-only boot's KV pool; images up to what the arena holds). `on` = the resident
        /// tower (D2, ~2 GiB a rank billed before the KV pool), `off` = text-only.
        #[arg(long, num_args = 0..=1, default_missing_value = "lazy",
              value_parser = ["lazy", "on", "off"])]
        vision: Option<String>,

        /// --engine real (D6): an image sequence's DRAFTER rope (and so its whole table set) at
        /// the token INDEX -- prod's V2 runner, whose DFlash speculator reads
        /// `input_batch.positions`. THE DEFAULT since 2026-09-23 (the user's decision over D6's
        /// measurement: prod's law, +0.5 % image acceptance, inside noise). This flag is an
        /// accepted no-op that names the default; `--drafter-rope-temporal` is the opt-out.
        #[arg(long)]
        drafter_rope_index: bool,

        /// --engine real: the pairing CONTROL -- D4's temporal M-RoPE axis for the drafter's rope
        /// (the pre-D6 law). Refused together with `--drafter-rope-index`.
        #[arg(long, conflicts_with = "drafter_rope_index")]
        drafter_rope_temporal: bool,

        /// Images per prompt (the reference server's `--limit-mm-per-prompt '{"image": N}'`; its default 999).
        #[arg(long, default_value_t = 999)]
        limit_mm_images: u32,

        /// --engine real, --engine-tp 2: HIP device rank 1 takes
        #[arg(long, default_value_t = 1)]
        engine_peer_device: i32,

        /// --engine real: TP rank of the pack
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// --engine real: HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        engine_device: i32,

        /// --engine real: sequences the engine holds session state for. One GDN state slot each;
        /// the engine runs ONE sequence per step whatever this is.
        #[arg(long, default_value_t = 2)]
        engine_max_seqs: u32,

        /// --engine real: tokens one sequence may reach. Sets the block-table width.
        #[arg(long, default_value_t = 4096)]
        engine_max_ctx: u32,

        /// --engine real: prefill chunk sizes to plan; a prompt is padded up into the smallest one
        /// that holds it (the C5 ladder by default; `256,2048` is the pre-C5 control)
        #[arg(long, value_delimiter = ',', default_value = DEFAULT_PREFILL_ROWS)]
        engine_prefill_rows: Vec<u32>,

        /// --engine real: decode batches to plan and capture; a step of N live sequences replays
        /// the smallest planned batch >= N and runs the rest as padding. `1,2,3,4,8` by default
        /// since CMP27AF (`1,2,4,8` = the control and the rollback);
        /// B1's ladder `1,2,3,4,5,6,7,8` captures every live count (no padding slot at any count)
        #[arg(long, value_delimiter = ',', default_value = saber_sched::decode_law::DEFAULT_DECODE_BATCHES_ARG)]
        engine_decode_batches: Vec<u32>,

        /// --engine real: verify widths to plan and capture; 1 is SPEC=0
        #[arg(long, value_delimiter = ',', default_value = "1")]
        engine_decode_widths: Vec<u32>,

        /// --engine real: which rule picks a step's verify width out of those — `batch` (the
        /// default table `1:8,4:5`, carry-forward on the running count), `batch=<table>` for
        /// another one, or `ema` for prod's acceptance cap (measured, and it loses: ws-t2)
        #[arg(long, default_value = "batch")]
        engine_width_rule: String,

        /// --engine real: RAGGED per-slot verify widths (workstream C2), OPT-IN since C3: every
        /// slot at its own width (prod's acceptance cap under the ceiling), the step in the
        /// smallest captured row band that holds their sum. Off by default: C3's paired
        /// BetterBench at the N8 defaults read rectangles 501.8 / ragged-block 501.3 /
        /// ragged-table 495.9 conc-8 — no arm past the MDE, so the standing rectangles stay
        #[arg(long, conflicts_with = "engine_rectangular_widths")]
        engine_ragged_widths: bool,

        /// --engine real: the decode ROW BANDS a ragged schedule plans and captures (C2), in
        /// rows: each is clipped to `batch x max width` and that product is always added.
        /// Only with `--engine-ragged-widths`. FORWARDED to rank 1
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "16,24,32,40,48,56,64",
            requires = "engine_ragged_widths"
        )]
        engine_row_bands: Vec<u32>,

        /// --engine real: `batch x width` rectangles, every slot at the bucket's width (THE
        /// DEFAULT since C3; the flag is accepted and says so explicitly)
        #[arg(long)]
        engine_rectangular_widths: bool,

        /// --engine real: a ragged step's per-slot ceiling (C2): `table` (the batch table's cell)
        /// or `block` (the plan's max width, prod's own ceiling). Only with
        /// `--engine-ragged-widths`
        #[arg(long, default_value = "table", requires = "engine_ragged_widths")]
        engine_ragged_ceiling: String,

        /// --engine real: the MIXED STEP — a decode step also carries queued prompts' prefill
        /// chunks (`saber_sched::MixedPrefill`) instead of a chunk taking a step of its own while
        /// the decoding sequences wait. `packed` (C10/C11) is the DEFAULT since 2026-09-24: every
        /// queued prompt's chunk that fits whole, in arrival order, in class, in ONE fused forward
        /// with the decode rows (a numerics change on the decode rows of a mixed step — LEDGER
        /// C8/C9/C11). `fused` (C9) is one chunk a forward; `serial` is C4's bit-exact form,
        /// chunk forward then decode forward; `off` is the unmixed schedule, the pairing control
        #[arg(long, num_args = 0..=1, default_missing_value = "packed", value_name = "FORM")]
        engine_mixed_prefill: Option<String>,

        /// --engine real: the mixed step's token budget, decode rows plus chunk rows (prod's
        /// `max_num_batched_tokens`, 8192)
        #[arg(long, default_value_t = saber_sched::MixedPrefill::PROD_MAX_BATCHED_TOKENS)]
        engine_max_batched_tokens: u32,

        /// --engine real (C7): the prefill FILL RULE — prefill rungs above this many rows are taken
        /// only by a chunk that fills them (6,000 rows at `…,2048,4096,8192` = 4096 + 1904), so a
        /// rung above 2048 never pads. 0 = off (the padded law, the control). A chunk boundary is
        /// not bit-neutral (G4): a numerics change for prompts above the padded maximum
        #[arg(long, default_value_t = 0)]
        engine_prefill_fill_above: u32,

        /// --engine real (C7): every prefill rung over ONE activation arena sized to the widest
        /// rung (ledger line `engine.arena.prefill`) instead of an arena per rung; the difference
        /// goes back to the KV pool. Bit-exact (no kernel, plan or row moves). The DEFAULT since
        /// 2026-09-24; accepted as a no-op
        #[arg(long, conflicts_with = "no_engine_shared_prefill_arena")]
        engine_shared_prefill_arena: bool,

        /// --engine real: an arena per prefill rung (the pre-C7 control)
        #[arg(long)]
        no_engine_shared_prefill_arena: bool,

        /// --engine real (H46): every decode HEAD over ONE arena sized to the widest head (ledger
        /// line `engine.arena.head.decode`) instead of an arena per decode row count; the
        /// difference goes back to the KV pool. Bit-exact (no kernel, plan or row moves). The
        /// DEFAULT since 2026-09-24; accepted as a no-op; forwarded to rank 1
        #[arg(long, conflicts_with = "no_engine_shared_head_arena")]
        engine_shared_head_arena: bool,

        /// --engine real: an arena per decode row count (the pre-H46 control)
        #[arg(long)]
        no_engine_shared_head_arena: bool,

        /// --engine real: run every step eagerly, with no captured graphs
        #[arg(long)]
        engine_no_capture: bool,

        /// --engine real: plan a DFlash2 drafter tier at this block size (prod's is 8 = 1 + 7
        /// speculative tokens; it must be a power of two), and 0 for none. It BILLS the drafter's
        /// bytes before the KV pool is derived from what is left, so a tier that does not fit is a
        /// refusal to start rather than an allocation failure 7 GiB in
        #[arg(long, default_value_t = 0)]
        engine_draft_block: u32,

        /// --engine real: the DRAFTER's own device-final pack. With it the tier is RESIDENT and
        /// every decode step makes its own proposal, which the next step's verify checks; without
        /// it `--engine-draft-block` bills the tier and nothing is uploaded
        #[arg(long)]
        engine_draft_pack: Option<PathBuf>,

        /// --engine real, --engine-tp 2: force every all-reduce to the EXACT family, by raising
        /// the rules' exact/lossy threshold above the largest message this server can send.
        ///
        /// The SAME flag `saber engine --ar-exact` takes, resolved the same way and forwarded to
        /// rank 1 by the same spawn, so the two planes can be made to agree on the arithmetic by
        /// configuration. DEFAULT OFF, which is prod's band: every serve-side number is taken in
        /// the lossy band at the widths whose messages cross it unless this is passed (ws-q1's
        /// open item 3).
        #[arg(long)]
        ar_exact: bool,

        /// --engine real, --engine-tp 2 (H36): overlap each prefill chunk's all-reduces with the
        /// GEMM that produces them, cutting the producer into row slices of this many rows (a
        /// second stream per rank, events not spin-waits). 0 = off, the default; buckets below
        /// 2048 rows are never cut. Bit-exact: only the reduces' placement moves
        #[arg(long, default_value_t = 0)]
        engine_prefill_ar_overlap: u32,

        /// --engine real, --engine-tp 2 (P27-1d): the reduce's form inside the prefill overlap's
        /// slices. `oneshot` = H36's one-shot (the default, the control); `split` = the push
        /// (`--engine-ar-push-blocks` blocks) + a command-processor wait + the reduce, bit-exact.
        /// Needs `--engine-prefill-ar-overlap`. Decode keeps the one-shot
        #[arg(long, default_value = "oneshot", value_parser = ["oneshot", "split"])]
        engine_ar_form: String,

        /// --engine-ar-form split: the push's physical blocks (P27-1c's pick: 16)
        #[arg(long, default_value_t = DEFAULT_AR_PUSH_BLOCKS)]
        engine_ar_push_blocks: u32,

        /// --engine real (C6): plan every PREFILL rung of at most this many rows (96..=128) on the
        /// split-K decode GEMM family at ONE split (the folded kernel's summation order: gated
        /// bit-exact) instead of the folded 256-row tile. 128 is the default (C6); 0 = the control
        #[arg(long, default_value_t = DEFAULT_SMALL_CHUNK_GEMM)]
        engine_small_chunk_gemm: u32,

        /// --engine real (C6): those rungs at the reference's M > 64 band WITH its splits (down,
        /// the GDN gate): a numerics-class change, the pairing arm
        #[arg(long)]
        engine_small_chunk_band: bool,

        /// --engine real (G5): the folded prefill GEMM at the mid-M tile heights of `[[folded_mt]]`
        /// (a 320-row tile at p320, 192-row tiles at p192 / p384, where the reference pays one or two
        /// 256-row tiles): one summation class with the reference tile (gated bit-exact). The
        /// DEFAULT since 2026-09-24; this flag is accepted as a no-op
        #[arg(long, conflicts_with = "no_engine_mid_chunk_gemm")]
        engine_mid_chunk_gemm: bool,

        /// --engine real: the reference 256-row folded tile in the `[[folded_mt]]` bands (the pre-G5
        /// control)
        #[arg(long)]
        no_engine_mid_chunk_gemm: bool,

        /// --engine real (G6): the layout law's mid band. Every fused producer writes the
        /// fragment-tiled A layout from `atiled_mid_min_m` rows (not only from 513) and the prefill
        /// GEMMs of the 129..512-row rungs take the A-tiled family at mid-M tile heights
        /// (`[[atiled_mt]]`): one summation class with the folded tiles they replace (gated
        /// bit-exact). The DEFAULT since 2026-09-24; this flag is accepted as a no-op
        #[arg(long, conflicts_with = "no_engine_mid_chunk_atiled")]
        engine_mid_chunk_atiled: bool,

        /// --engine real: the pre-G6 layout law (row-major producers and the folded tiles at
        /// 129..512 rows) — the control
        #[arg(long)]
        no_engine_mid_chunk_atiled: bool,

        /// --engine real (G6): the 48-wide GDN gate on the skinny-N body (`[[skinny]]`: one wave
        /// per 16-row fragment, all of K in the reference's order; one summation class). The
        /// DEFAULT since 2026-09-24; accepted as a no-op
        #[arg(long, conflicts_with = "no_engine_skinny_gate")]
        engine_skinny_gate: bool,

        /// --engine real: the 48-wide gate on the reference K-serial body — the control
        #[arg(long)]
        no_engine_skinny_gate: bool,

        /// --engine real (CMP27E; THE DEFAULT since CMP27F): the b8 (tm4) decode GEMMs and the GDN
        /// in_proj pair on the register-prefetch twins (`mxfp4_gemm_decode_pf`; bit-exact)
        #[arg(long, conflicts_with = "no_engine_decode_pf")]
        engine_decode_pf: bool,

        /// --engine real: the shipped decode bodies at b8 -- the CMP27F control
        #[arg(long)]
        no_engine_decode_pf: bool,

        /// --engine real (CMP27F; THE DEFAULT): the fp8 head at 49..64 rows (the b8 verify) in ONE
        /// weight pass (`saber_head_w8_m64`; bit-exact)
        #[arg(long, conflicts_with = "no_engine_head_m64")]
        engine_head_m64: bool,

        /// --engine real: the fp8 head's two 32-row passes at 49..64 rows -- the CMP27F control
        #[arg(long)]
        no_engine_head_m64: bool,

        /// --engine real (CMP27K; DEFAULT 17 since CMP27O): the drafter's gate_up at MIN_ROWS..64
        /// rows in ONE weight pass (`dflash_gemm_m64`; bit-exact); `0` = the control and the
        /// rollback (the m-tiled kernel); a bare flag = 33 (the b8 cells only)
        #[arg(long, num_args = 0..=1, default_missing_value = "33", value_name = "MIN_ROWS")]
        engine_drafter_gateup_m64: Option<u32>,
        /// --engine real, --engine-tp 2 (CMP27S; ON BY DEFAULT since the CMP27S flip): the wht6
        /// reduce's WAVE-PARALLEL body (`saber_ar_wht_wpar`: the served bytes, grid and kernargs; one
        /// (chunk, iteration) item a wave) for every rotated call whose chunks a block fit its LDS;
        /// `off` = the served one-shot body (the control and the rollback). Forwarded to rank 1
        #[arg(long, num_args = 0..=1, default_missing_value = "on",
              default_value = saber_engine::device::DEFAULT_AR_WPAR, value_parser = ["on", "off"])]
        engine_ar_wpar: String,
        /// --engine real, --engine-tp 2 (CMP27AE; ON BY DEFAULT since the CMP27AE flip): the N 5120 dwn4 decode
        /// GEMM and the wht6 reduce after it as ONE launch (`mxfp4_gemm_decode_arw`, bit-exact,
        /// the target's 17..64-row decode plans; stands down under `--engine-race-check`); `off` =
        /// the two launches (the control). Forwarded to rank 1
        #[arg(long, num_args = 0..=1, default_missing_value = "on",
              default_value = saber_engine::device::DEFAULT_AR_FUSE, value_parser = ["on", "off"])]
        engine_ar_fuse: String,
        /// --engine real (CMP27L; ALL ON BY DEFAULT since CMP27O): b1 levers, bit-exact, a comma
        /// list of `conv` (the rows-parallel tree conv), `adv` (the R2 flush's advances in ONE
        /// launch), `head` (the 1..16-row fp8 head on the 32-row nontemporal band); `all` = every
        /// one; `off` = the control and the rollback
        #[arg(long, default_value = saber_engine::device::DEFAULT_B1_ARMS)]
        engine_b1_arms: String,
        /// --engine real (CMP27T; ALL ON BY DEFAULT since its flip; `off` = the control and the
        /// rollback): conc-4 levers, bit-exact, a
        /// comma list of `replay` (the GDN replay update with q / k, gates and v staged once in
        /// LDS), `pf2` (the register-prefetch decode body at the tm2 cells + pair), `picks` (the
        /// TP > 1 argmax also writes its winner's logit: one pick-table copy); `all` = every one
        #[arg(long, default_value = saber_engine::device::DEFAULT_C4_ARMS)]
        engine_c4_arms: String,
        /// --engine real (CMP27X; BOTH ON BY DEFAULT since its flip; `off` = the control and the
        /// rollback): step levers, bit-exact, a
        /// comma list of `devrow` (the sampler seam's row kernel with a thread's loads issued
        /// ahead, `saber_device_row_ilp_bf16`), `dwn4` (the prefetch decode GEMMs at N 5120 on 4
        /// waves a workgroup, 2x the workgroups); since the CMP27AG flip ALSO ON (`devrow,dwn4` = the
        /// control and the rollback): `treeblob` (CMP27AF: the
        /// tree form's own index tables narrowed + written only at a tree step), `slotdedup`
        /// (CMP27AG: one index slot per (side, suffix) group), `slotdedupcheck=N` (diagnostic:
        /// every key's own build compared with its shared slot for N steps); since the CMP27AA
        /// flip ALSO ON (`devrow,dwn4,treeblob,slotdedup` = the control and the rollback): `dh64`
        /// (CMP27Z: the drafter's int2 head at 64 rows in one weight pass), `rowsplit` (CMP27Y: the
        /// seam row over 2 workgroups, `saber_device_row_split_g2_bf16`), and `cuf` (CMP27AI FLIP
        /// 2026-10-30: the GDN conv update fused into the replay update, `p27w2_gdn_cuf_*`, routed
        /// at b3 / b4 only; `devrow,dwn4,treeblob,slotdedup,dh64,rowsplit` = the control and the
        /// rollback), and `rowstage` (CMP27AJ FLIP 2026-10-30: the seam row in the staged-term form,
        /// `saber_device_row_stage_t1024_bf16`, supersedes rowsplit / devrow at the seam;
        /// `devrow,dwn4,treeblob,slotdedup,dh64,rowsplit,cuf` = the control and the rollback);
        /// `all` = every one
        #[arg(long, default_value = saber_engine::device::DEFAULT_STEP_ARMS)]
        engine_step_arms: String,

        /// --engine real (C8 GATE CONTROL, never a serving mode): every DECODE plan in the fused
        /// mixed step's decode-row class (every decode GEMM cell at one split, the lossy reduce at
        /// every size, decode fusions off; `saber_engine::mixed_class`) — the quality gates' proxy
        /// for `--engine-mixed-prefill fused`. Forwarded to rank 1
        #[arg(long)]
        engine_mixed_class_proxy: bool,

        /// --engine real (C5): record the per-step HOST-time tables (`saber_engine::host_trace`,
        /// decode AND prefill, with every prefill chunk by planned bucket) and print them after
        /// the shutdown census; at TP=2 the follower prints its own. A DIAGNOSTIC serve: the
        /// instrument sits upstream of the tok/s, so a number from this serve is never published
        /// (GATES.md blind spot 19). OFF by default
        #[arg(long)]
        engine_host_trace: bool,

        /// --engine real: WHICH decode launch-train fusions the serve plane's engine plans. ON by
        /// default since N8 (2026-09-23, the user's decision) with ws-h37's set (`h37`: the `_nq`
        /// epilogue + the decode GEMM pair, rows 1-2). `all` is the ws-h38 served set (rows 1-3 +
        /// 7: adds the split decode GEMM + reduce + epilogue in one dispatch and the rope + KV
        /// append; row 6 was measured slower and is not in it), opt-in. A bare flag is N8's
        /// accepted no-op (it prints what the default resolved to). `--engine-no-decode-fusion`
        /// is the unfused control
        #[arg(long, num_args = 0..=1, default_missing_value = "default",
              value_parser = ["default", "h37", "all", "h40", "h42"])]
        engine_decode_fusion: Option<String>,

        /// --engine real, **PAIRING CONTROL** (N8): `saber engine --no-decode-fusion` on the serve
        /// plane -- plan the decode step unfused (the op sequence before ws-h37). Bit-exact with
        /// the default: only the dispatch count moves
        #[arg(long)]
        engine_no_decode_fusion: bool,

        /// --engine real, **P27-16 ARM**: the 27B's launch-fusion families on top of the served
        /// decode set (`none` = the default = today's step; `smq` = fusion-map row 4, `gdnnq` =
        /// row 5, `all` = both; a comma list). Bit-exact families only; resolved on the leader
        /// into the set rank 1 is told by value. Boot line `fuse27`
        #[arg(long, default_value = saber_engine::fuse27::DEFAULT_FUSE27, value_parser = parse_fuse27_arg)]
        engine_fuse: String,

        /// --engine real, --engine-tp 2, **CONTROL** (ws-p12): `saber engine --whole-step-frame`
        /// on the serve plane — the peer is sent the WHOLE step-varying index region every step
        /// instead of the runs the step wrote. Forwarded to rank 1 by the same spawn. A gate
        /// switch for "is the compacted frame why eight-in-flight serving differs?", never a
        /// serving one
        #[arg(long)]
        engine_whole_step_frame: bool,

        /// --engine real, --engine-tp 2, **CONTROL** (ws-p12): `saber engine --frame-digest` on
        /// the serve plane — the follower checks an FNV of the step-varying region after applying
        /// each frame, so a frame that leaves the two ranks' index blobs different is a named
        /// error at the step it happens
        #[arg(long)]
        engine_frame_digest: bool,

        /// --engine real, **DIAGNOSTIC** (T4): `saber engine --race-check` on the serve plane --
        /// the all-reduces run their race-check twins, every step reads their counters and swaps
        /// the sequence-counter digest and the frame serial with the peer, every wait re-queries
        /// an event recorded before it, and every proposal read checks a sentinel. Never a
        /// serving mode: the checked kernels hash every word they move. `shadow` (T6) is the
        /// non-perturbing form: none of those checks, only the per-proposal stage digests
        /// computed on the card after the proposal and read in its own pinned batch
        #[arg(long, num_args = 0..=1, default_missing_value = "full", value_name = "MODE")]
        engine_race_check: Option<String>,

        /// --engine real, --engine-tp 2, **CONTROL** (ws-p12): `saber engine --gather-per-row` on
        /// the serve plane — the logits gather one row per host round trip, as before ws-h31
        #[arg(long)]
        engine_gather_per_row: bool,

        /// FN73 (DEBUG, the teardown law's injection): `R:N` -- rank R fails after N steps (the
        /// 27B pair: R = 1, rank 1's follower loop errors after N steps).
        #[arg(long, hide = true, value_name = "R:N")]
        fail_rank_after_steps: Option<String>,

        /// FN57: how a step's host thread waits for the card -- `aimed` (ws-h32's sleep-then-poll,
        /// THE 27B DEFAULT), `spin` (the HIP runtime's active wait, one bounded
        /// `hipEventSynchronize`) or `block` (`hipDeviceScheduleBlockingSync`: every runtime wait
        /// sleeps on the signal's interrupt). Absent = the engine's default (27B `aimed`,
        /// Flash-Next `spin`). Forwarded to rank 1.
        #[arg(long)]
        host_wait: Option<String>,

        /// --engine real, **PAIRING CONTROL** (H35): `saber engine --draft-store-narrow` on the
        /// serve plane — each prefill chunk's drafter context stored at 8 rows a tile (main's law)
        #[arg(long)]
        engine_draft_store_narrow: bool,

        /// --engine real, **PAIRING CONTROL** (C6): a prefill chunk below the drafter store's class
        /// boundary (256 rows) stored as 8-row tiles (H35's law) instead of ONE wide tile at its
        /// own norm class
        #[arg(long)]
        engine_draft_store_natural_off: bool,

        /// --engine real, **GATE** (H35): `saber engine --draft-store-check` on the serve plane —
        /// every prefill chunk's wide store compared byte for byte against a narrow re-store
        #[arg(long)]
        engine_draft_store_check: bool,

        /// --engine real, **ARM** (P27-3): `saber engine --drafter-store` on the serve plane —
        /// `on-path` (today's law, the control) or `deferred` (the prompt's last-chunk drafter
        /// store runs with the first proposal, after the first token is handed back)
        #[arg(long, default_value = "on-path", value_parser = ["on-path", "deferred"])]
        engine_drafter_store: String,

        /// --engine real (P27-6): the drafter's KV law — `paged` (the drafter's pages sized by
        /// ADMISSION from the target's own KV ledger: one block costs the target's block + the
        /// drafter's page, a session's run is its reservation's length) or `dense` (the control:
        /// `max_seqs` runs of `max_ctx` pages billed before the arenas, 10,240 MiB a rank at
        /// max-ctx 262,144)
        #[arg(long, default_value = DEFAULT_DRAFT_KV, value_parser = ["paged", "dense"])]
        engine_draft_kv: String,

        /// --engine real (P27-9b): the drafter's input embedding — `on` (the drafter reads the
        /// target's `embed_tokens` record, checked byte-identical at boot; one copy a rank) or
        /// `off` (the control: the drafter uploads its own copy)
        #[arg(long, default_value = DEFAULT_DRAFT_EMBED_SHARE, value_parser = ["on", "off"])]
        engine_draft_embed_share: String,

        /// --engine real (P27-12): how a proposal ranks the DFlash2 selector's lattice -- `walk`
        /// (the device's greedy walk, prod's; the default and the control) or `explen` (the
        /// host's expected-length chain over the same 7 x 16 x 16 scores, read back in the
        /// proposal's batch), or `tandem-b16-calibrated` for the matched Quark/Tandem B16 TP=2 ancestor9 profile.
        #[arg(long, default_value = "walk", value_parser = ["walk", "explen", "tandem-b16-calibrated"])]
        engine_selector_rank: String,

        /// Score the full FP8 vocabulary for single-request native TP2 B8 proposals.
        #[arg(long)]
        engine_draft_full_vocab: bool,

        /// --engine real: `ancestor9` (default, root + 9 candidates), `best16` (root + 15
        /// candidates using the existing 6-wave attention/tree objects, requires max-seqs=1),
        /// or `off` (the chain). The ordinary decode width limit is unchanged.
        #[arg(long, default_value = "ancestor9", value_parser = ["off", "ancestor9", "best16"])]
        engine_spec_tree: String,

        /// --engine real (P27-W2-3): the SAMPLED tree acceptance law -- `off` (the default: a
        /// sampled slot runs the scheduler's draft as a chain plan inside the tree form) or `on`
        /// (THE DEFAULT since P27-W2-4, the user's flip 10-22; `off` = the control: a sampled slot verifies its lattice plan and commits the keyed draw walked
        /// over the tree; the output distribution is the target's, seeded draws stay a function
        /// of the seed). Inert unless the tree form runs
        #[arg(long, default_value = "on", value_parser = ["off", "on", "control"])]
        engine_spec_tree_sampled: String,
        /// --engine real (P27-W2-5 / CMP27G): the GDN decode update as BASE + STASH + REPLAY.
        /// `auto` (THE DEFAULT since CMP27G) = `on` unless the boot cannot carry it (the f32
        /// state, `--engine-spec-tree best16`, the prefix cache, a park tier: then `off`, said
        /// on the `gdnreplay` boot line); `on` = the arm, refused by name where it cannot run;
        /// `off` = the per-row snapshot law (the control and the rollback). Bit-exact.
        #[arg(long, default_value = "auto", value_parser = ["off", "on", "auto"])]
        engine_gdn_replay: String,

        /// --engine real (P27-14): the 27B's BIT-EXACT host levers, `none` or a comma list of
        /// `picks` (a sampled decode step's pick + seam in one frame), `table` (the compact
        /// block-table frame at a page crossing), `socket` (the proposal's frame inside its
        /// `Command::Propose`, the single proposal's terminator folded); `all` = the three
        #[arg(long, default_value = saber_engine::host_levers::DEFAULT_HOST_LEVERS, value_parser = parse_host_levers_arg)]
        engine_host_levers: String,

        /// --engine real (P27-W2-2): the 27B's device-step form: `auto` (THE DEFAULT since CMP27P:
        /// `cand` at TP = 2 without the T4/T5 race check, `off` elsewhere), `cand` (the drafter's
        /// candidate seam merged on the card through a P2P window, no host wait between the
        /// backbone and the tail; TP = 2, bit-exact) or `off` (the control and the rollback)
        #[arg(long, default_value = saber_engine::device_step::DEFAULT_DEVICE_STEP, value_parser = parse_device_step_arg)]
        engine_device_step: String,

        /// The GDN recurrent state's STORED dtype (C1): `f16` (THE DEFAULT since N8, 2026-09-23,
        /// the user's decision; prod's `--mamba-ssm-cache-dtype float16`: half the state bytes,
        /// fp32 accumulation, only the store narrows) or `f32` (the pairing control, the HF
        /// config's `mamba_ssm_dtype`). Moves greedy ids, so it is gated by end-task quality, not
        /// byte-compare. --engine real: `saber engine --gdn-state-dtype`
        #[arg(long, default_value = "f16", value_parser = ["f32", "f16"])]
        engine_gdn_state_dtype: String,

        /// ACCEPTED, DOES NOTHING (N5, N6): the default `lm_head` is the checkpoint's own, as the
        /// pack carries it. Kept so older command lines still parse; it prints what the default
        /// resolved to
        #[arg(long)]
        engine_bf16_head: bool,

        /// --engine real, **PAIRING CONTROL** (N5): plan the MXFP4 `lm_head` (the head before
        /// 2026-09-22) instead of the default exact head (the checkpoint's own, as the pack carries
        /// it). Needs the MXFP4 pack (`/packs/qwen38-27b-nvfp4-devfinal`); the default refuses it by
        /// name, and this flag refuses an exact-head pack by name
        #[arg(long, alias = "engine-mxfp4-head")]
        mxfp4_head: bool,

        /// ACCEPTED, DOES NOTHING (N6): the fp8-weight head is what the default resolves to on the
        /// fp8-head pack (this checkpoint's). It prints what the default resolved to
        #[arg(long)]
        engine_fp8_head: bool,

        /// --engine real: refuse to start if the engine's device buffers would exceed this
        #[arg(long, default_value_t = 26000)]
        engine_budget_mib: u64,

        /// --engine real: page-locked host MiB for the session-park staging buffer. 0 disables
        /// parking on the engine side, and `--park-ram-gib` without it is a refusal to start:
        /// a tier with nowhere to copy from refuses every eviction at the moment it matters
        #[arg(long, default_value_t = 0)]
        engine_park_staging_mib: u64,

        /// --engine real: MiB kept back from the derived KV pool. Raising it SHRINKS the pool,
        /// which is how an under-pressure arm makes the scheduler evict (ws-k14)
        #[arg(long, default_value_t = 1024)]
        engine_kv_headroom_mib: u64,

        /// --engine real (P27-7): the byte ledger's law, `lean` by default. `today` (the control) runs
        /// `--engine-budget-mib` / `--engine-kv-headroom-mib` as given; `lean` replaces both with
        /// the pair measured on ROCm 10.1 (`ledger_law::LEAN_*`), said on the `engledger` line
        #[arg(long, default_value = ledger_law::DEFAULT_LEDGER)]
        engine_ledger: String,

        /// --engine real: KV pool blocks, explicitly. 0 derives the pool from the byte ledger.
        /// A small pool is what makes the arenas refuse an admission, which is the only thing
        /// the on-pressure eviction policy fires on
        #[arg(long, default_value_t = 0)]
        engine_kv_pool_blocks: u32,

        /// The name clients ask for. The reference server takes several (prod's unit serves four aliases); Saber
        /// serves ONE and refuses a second by name rather than dropping it
        #[arg(long, default_value = "Qwen3.8", num_args = 1..)]
        served_model_name: Vec<String>,

        /// The reference server's `--enable-auto-tool-choice` (V9): parse tool calls out of the reply. ON unless
        /// `--no-enable-auto-tool-choice` or `--parser-defaults compat`; needs a tool parser
        #[arg(long, overrides_with = "no_enable_auto_tool_choice")]
        enable_auto_tool_choice: bool,

        /// The reference server's `--no-enable-auto-tool-choice`: no tool parser is constructed, and a request
        /// that sends `tools` without `tool_choice: "none"` gets the reference server's 400
        #[arg(long, overrides_with = "enable_auto_tool_choice")]
        no_enable_auto_tool_choice: bool,

        /// The reference server's `--tool-call-parser` (default `qwen3_coder` = `qwen3_xml` = `mimo`, one parser in
        /// prod's image). Any other of the reference server's registered names is refused by name; an unknown one
        /// gets the reference server's own error. Inert without auto tool choice, as in the reference server
        #[arg(long)]
        tool_call_parser: Option<String>,

        /// The reference server's `--reasoning-parser` (default `qwen3` = `mimo`). `""` is the reference server's default: no
        /// reasoning parser, the think block stays in `content`
        #[arg(long)]
        reasoning_parser: Option<String>,

        /// What an ABSENT parser flag means: `prod` (the default: prod's unit, auto tool choice on,
        /// `qwen3_coder`, `qwen3`) or `compat` (the reference server's own: everything off), for a launcher written
        /// for the reference server's `serve`
        #[arg(long, default_value = "prod")]
        parser_defaults: String,

        /// The reference server's `--max-logprobs`: the cap on `logprobs`, `top_logprobs` and `prompt_logprobs`,
        /// refused at validation with the reference server's 400. -1 is the whole vocabulary. The reference server's default, 20,
        /// is what the prod unit runs
        #[arg(long, default_value_t = 20, allow_negative_numbers = true)]
        max_logprobs: i32,

        /// --engine real: seconds one engine call may take before the engine is unresponsive
        #[arg(long, default_value_t = 120)]
        engine_deadline_s: u64,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,

        /// Explicit RCCL compatibility transport (TP=2, host-capable collectives).
        #[arg(long)]
        engine_rccl_lib: Option<String>,

        /// Native exact collectives over shared host pages (TP=2, no P2P).
        /// With --engine-rccl-lib, messages larger than 64 target rows use RCCL.
        #[arg(long)]
        engine_host_ar: bool,

        /// Diagnostic target-feature export (single sequence, prefix caching off; no performance figures).
        #[arg(long)]
        engine_draft_capture_dir: Option<PathBuf>,

        /// Use an external DSpine proposer over a local Unix-domain socket.
        #[arg(long)]
        engine_dspine_socket: Option<PathBuf>,

        /// Independent capture layers in checkpoint order (five comma-separated ids).
        #[arg(long, value_delimiter = ',', requires = "engine_draft_capture_dir")]
        engine_draft_capture_layers: Vec<u32>,

        /// The Saber manifest, resolved exactly as `saber explain` resolves it
        #[arg(long, default_value = "examples/qwen38-27b-nvfp4-tp2.toml")]
        manifest: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// An autotune cache. Covering every route binds the plan; otherwise the mock says so
        /// and continues on the manifest's bucket geometry, because it launches no kernel.
        #[arg(long)]
        autotune: Option<PathBuf>,

        /// The checkpoint's tokenizer.json
        #[arg(long, default_value = ".cache/template_corpus.tokenizer.json")]
        tokenizer: PathBuf,

        /// The chat template (qwen-fixed-v22.3.jinja)
        #[arg(long, default_value = ".cache/template_corpus.jinja")]
        chat_template: PathBuf,

        /// Address to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind; 0 takes an ephemeral one and prints it
        #[arg(long, default_value_t = 8099)]
        port: u16,

        /// KV arena per rank, GiB
        #[arg(long, default_value_t = 8)]
        kv_gib: u64,

        /// GDN state arena per rank, GiB
        #[arg(long, default_value_t = 8)]
        state_gib: u64,

        /// Milliseconds one engine step takes; 0 is as fast as the host allows
        #[arg(long, default_value_t = 0)]
        step_delay_ms: u64,

        /// Host RAM the session-parking tier may hold, GiB. 0 disables parking.
        #[arg(long, default_value_t = 0.0)]
        park_ram_gib: f64,

        /// Parking-file size, GiB. Needs --park-nvme-path. 0 means no spill tier.
        #[arg(long, default_value_t = 0.0)]
        park_nvme_gib: f64,

        /// Where the parking file lives
        #[arg(long)]
        park_nvme_path: Option<PathBuf>,

        /// Let the scheduler evict an idle session when an admission is refused for KV or GDN
        /// state. Without it, a parking tier is an operator hook and nothing evicts on its own.
        #[arg(long)]
        park_on_pressure: bool,

        /// Which running session the on-pressure policy evicts: `newest` (prod's — the back of the
        /// running queue, `upstream/v1/core/sched/scheduler.py`'s `self.running.pop()`) or
        /// `most-work` (K5..K17's — the session with the most tokens left).
        #[arg(long, default_value = "newest")]
        park_victim: String,

        /// Times one session may be evicted over its life (1 by default, ws-k18's measured
        /// deviation from prod's unbounded preemption). 0 means unbounded.
        #[arg(long, default_value_t = 1)]
        park_max_per_session: u32,

        /// Failed restore attempts after which a parked session's request ends with a named error.
        #[arg(long, default_value_t = 3)]
        park_max_restore_attempts: u32,

        /// THE CONTROL for the bounded-restore law, never a serving flag: the real executor refuses
        /// every restore (image handed back), so the retry-then-give-up path runs on the card.
        #[arg(long)]
        engine_park_fail_restores: bool,
    },

    /// The P3 seam's gate: one prompt through the pass and through the serve path, plus ms/step
    Engine {
        /// The device-final pack (directory holding rank<r>.pack, or one such file)
        #[arg(long)]
        pack: PathBuf,

        /// Ranks in the tier: 1 (one card) or 2 (this process plus a worker on the other)
        #[arg(long, default_value_t = 1)]
        tp: u32,

        /// TP rank of the pack
        #[arg(long, default_value_t = 0)]
        rank: u32,

        /// --tp 2: HIP device rank 1 takes, within the visible set
        #[arg(long, default_value_t = 1)]
        peer_device: i32,

        /// --tp 2: the loopback port the two ranks meet on; 0 asks the OS for a free one
        #[arg(long, default_value_t = 0)]
        peer_port: u16,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`. Required except for a Flash-Next
        /// `--fn-serve` boot, which sizes itself from a built-in prompt when this is absent
        #[arg(long)]
        prompt_ids: Option<PathBuf>,

        /// Row of the corpus to take
        #[arg(long, default_value_t = 0)]
        prompt_index: usize,

        /// Truncate the prompt to this many tokens (0 = all of it). The comparator is at ctx ~ 0.
        ///
        /// Truncating below the reference's `n_prompt` is REFUSED: a 25-token slice of a 234-token
        /// chat prompt ends mid-sentence, and the continuation of a fragment is not something prod
        /// has an answer for.
        #[arg(long, default_value_t = 0)]
        tokens: u32,

        /// Prod's first token per corpus row (workstream P's `nospec_ref.jsonl`)
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Least top-1-vs-top-2 margin, in nats, a prompt may have and still be used
        #[arg(long, default_value_t = 1.0)]
        min_margin: f64,

        /// Decode steps to run and to time
        #[arg(long, default_value_t = 400)]
        gen: usize,

        /// Timed steps discarded at the front
        #[arg(long, default_value_t = 5)]
        warmup: usize,

        /// Sequences the engine holds session state for
        #[arg(long, default_value_t = 2)]
        max_seqs: u32,

        /// Tokens one sequence may reach; sets the block-table width
        #[arg(long, default_value_t = 4096)]
        max_ctx: u32,

        /// Prefill chunk sizes to plan (the C5 ladder by default; `256,2048` is the pre-C5 control)
        #[arg(long, value_delimiter = ',', default_value = DEFAULT_PREFILL_ROWS)]
        prefill_rows: Vec<u32>,

        /// Decode batches to plan and capture (`1,2,3,4,8` since CMP27AF, `1,2,4,8` = the
        /// control; B1's ladder is `1,2,3,4,5,6,7,8`)
        #[arg(long, value_delimiter = ',', default_value = saber_sched::decode_law::DEFAULT_DECODE_BATCHES_ARG)]
        decode_batches: Vec<u32>,

        /// Verify widths to plan and capture: 1 is SPEC=0, 5 is the b*s5 buckets prod runs
        #[arg(long, value_delimiter = ',', default_value = "1")]
        decode_widths: Vec<u32>,

        /// Which rule picks a step's verify width out of those — `batch` (the default table
        /// `1:8,4:5`, carry-forward on the running count), `batch=<table>` for another one, or
        /// `ema` for prod's acceptance cap (measured, and it loses: ws-t2)
        #[arg(long, default_value = "batch")]
        width_rule: String,

        /// RAGGED per-slot verify widths (C2), OPT-IN since C3 (rectangles are the default: C3's
        /// paired BetterBench found no arm past the MDE). The scheduler-driven arms (concurrency,
        /// mixed arrival) run it at 3+ running; the direct arms call the rectangles as before
        #[arg(long, conflicts_with = "rectangular_widths")]
        ragged_widths: bool,

        /// The decode ROW BANDS a ragged schedule plans and captures (C2), in rows: each is
        /// clipped to `batch x max width` and that product is always added. Only with
        /// `--ragged-widths`
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "16,24,32,40,48,56,64",
            requires = "ragged_widths"
        )]
        row_bands: Vec<u32>,

        /// Today's rectangles (THE DEFAULT since C3; accepted, explicit)
        #[arg(long)]
        rectangular_widths: bool,

        /// A ragged step's per-slot ceiling (C2): `table` or `block`. Only with `--ragged-widths`
        #[arg(long, default_value = "table", requires = "ragged_widths")]
        ragged_ceiling: String,

        /// The MIXED STEP on the serve-path arms (`saber_sched::MixedPrefill`): `packed` (the
        /// default since 2026-09-24, also the bare flag), `fused` (C9), `serial` (C4, bit-exact)
        /// or `off` (the unmixed schedule, the control)
        #[arg(long, num_args = 0..=1, default_missing_value = "packed", value_name = "FORM")]
        mixed_prefill: Option<String>,

        /// The mixed step's token budget, decode rows plus chunk rows (prod's 8192)
        #[arg(long, default_value_t = saber_sched::MixedPrefill::PROD_MAX_BATCHED_TOKENS)]
        max_batched_tokens: u32,

        /// C7: the prefill fill rule on the serve-path arms (rungs above this many rows only
        /// when filled; 0 = off)
        #[arg(long, default_value_t = 0)]
        prefill_fill_above: u32,

        /// Decode steps per batch in the batch sweep
        #[arg(long, default_value_t = 100)]
        batch_gen: usize,

        /// Sequences in the batched-identity gate
        #[arg(long, default_value_t = 4)]
        gate_seqs: usize,

        /// Concurrency levels to drive through the scheduler
        #[arg(long, value_delimiter = ',', default_value = "1,4,8")]
        conc: Vec<usize>,

        /// Tokens each concurrent stream asks for
        #[arg(long, default_value_t = 64)]
        conc_tokens: usize,

        /// A transcript of chat turns to run instead of the corpus: one JSON object per line with
        /// a `messages` array, which is what `tools/gates/multiturn_continuity.py` writes. The
        /// turns it recorded as DIFFERING are taken first. Needs --tokenizer and --chat-template
        #[arg(long)]
        messages: Option<PathBuf>,

        /// The tokenizer --messages renders through (the serve plane's own)
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// The chat template --messages renders through (the serve plane's own)
        #[arg(long)]
        chat_template: Option<PathBuf>,

        /// Tokens the CONTINUITY arm asks its re-issued request for. 0 (the default) keeps the
        /// bound the arm has always had — `conc_tokens` clamped to 4..16 — which is too short to
        /// reach a divergence that first appears at token 29 (ws-l15's four multi-turn turns)
        #[arg(long, default_value_t = 0)]
        continuity_tokens: usize,

        /// Branch counts to run the SEQUENCE-GROUP arm at (`2,4`). Empty skips it, which is the
        /// default: `n > 1` is a request shape and every other arm of this gate is `n = 1`.
        #[arg(long, value_delimiter = ',')]
        fork_n: Vec<usize>,

        /// Tokens each branch of the fork arm asks for
        #[arg(long, default_value_t = 64)]
        fork_tokens: usize,

        /// Write every sampled row's seam, parameters, noise key and token here as JSONL, so that
        /// `saber sample-check` can replay the fork's draws against the recorded keys
        #[arg(long)]
        sample_dump: Option<PathBuf>,

        /// Run ONLY the chunk bisect, at this chunk size: one prompt fed as N-row chunks against
        /// the same prompt fed as ONE bucket, with the residual stream digested at every layer at
        /// each chunk's last live row and at the SAME absolute rows of the one-bucket run.
        ///
        /// The serve path chunks and `saber forward` does not, and on prompt 8 of the long-context
        /// corpus the two answer different tokens (ws-h28). This names the first chunk and layer
        /// that move. Needs a planned prefill shape of exactly this size AND one that holds the
        /// whole prompt: `--prefill-rows 2048,14301 --prompt-index 8 --tokens 0`
        #[arg(long, default_value_t = 0)]
        chunk_bisect: u32,

        /// Skip the bucket ATTRIBUTION of a red batched/concurrency arm, and report the red
        /// without it.
        ///
        /// The attribution is ws-p9's and it re-runs the diverging prompt alone and in the batch
        /// with the head's sampling seam on. It is correct and it is not free: it costs this
        /// engine a dozen extra admissions and ~50 extra steps, and ws-t3 measured every arm
        /// AFTER it in the same boot reading **+2.7 ms/step** while the same binary's own timed
        /// arm — taken before it — was unmoved. This switch is the CONTROL for that: one boot
        /// with it and one without, and the pair says what the attribution costs the arms
        /// downstream of it.
        #[arg(long)]
        no_attribute: bool,

        /// Record and print the per-step HOST-time breakdown: index build, blob upload, replay
        /// submit, each rank swap, the argmax and seam read-backs, the wait, the acceptance law
        /// and the proposal's two phases, as p50/p90 and ms/step over each timed arm (ws-h31).
        ///
        /// **A DIAGNOSTIC.** The instrument costs two clock reads and a mutex push per phase
        /// boundary, so every number a traced boot prints is a traced number and none of them is
        /// a published one (`docs/GATES.md` blind spot 19). It also runs the RESIDUE arm at the
        /// very end of the gate — the same steps before and after ws-p9's bucket attribution,
        /// per phase, with `--no-attribute` as the control that must read zero.
        #[arg(long)]
        host_trace: bool,

        /// **CONTROL**: send the WHOLE step-varying index region to the peer every step, the way
        /// every step did before ws-h31 compacted the frame to the runs the step actually wrote
        /// (1,932 KiB out and, on the old symmetric exchange, 1,932 KiB of zeros back). Two boots
        /// of one binary, with and without, are the A side of "is the compacted frame why two
        /// runs differ?"
        #[arg(long)]
        whole_step_frame: bool,

        /// **CONTROL**: carry an FNV of the step-varying region in every step frame and make the
        /// follower check it after applying the runs. The frame's whole claim is that its runs
        /// leave the two ranks' index blobs equal; this turns that induction into an assertion at
        /// the exact step it would break. It costs an FNV over 1.9 MiB on each rank per step, so
        /// it is a gate switch and never a serving one
        #[arg(long)]
        frame_digest: bool,

        /// **DIAGNOSTIC** (T4): the race checks -- every all-reduce runs its `_chk` twin (payload
        /// check words, the flag after the spin, row 3's split-K counter), every step reads the
        /// counts and swaps the sequence-counter digest and frame serial with the peer, every
        /// wait re-queries an event recorded before it, every proposal read checks a sentinel.
        /// Forwarded to rank 1. `racecheck rank R:` is the end-of-run line. `--race-check shadow`
        /// (T6) runs NONE of those: only the per-proposal stage digests, computed on the card
        /// after the proposal's last kernel and read in the proposal's own pinned batch (no sync,
        /// no read-back); `racecheck shadow rank R:` is its end-of-run line
        #[arg(long, num_args = 0..=1, default_missing_value = "full", value_name = "MODE")]
        race_check: Option<String>,

        /// **DIAGNOSTIC** (T6): at the end of boot fill a set of device buffers with a byte --
        /// `arenas:BYTE` (every target arena and `aux.cat`), `kv:BYTE` (the K/V pool), `all:BYTE`.
        /// A first request whose ids move reads memory it never wrote. Forwarded to rank 1
        #[arg(long, value_name = "SET:BYTE", value_parser = parse_poison)]
        poison_boot: Option<String>,

        /// **GATE CONTROL** (T7): the pre-T7 constant re-seed policy -- only the shared prefill
        /// arena's zero residual is re-seeded, every decode bucket reads its boot zeros. The
        /// card-side failing control of the stale-read law (with `--poison-boot decode:0x3c` it
        /// must move the run) and the paired-step control. Forwarded to rank 1
        #[arg(long)]
        constant_seed_control: bool,

        /// **CONTROL**: read the logits gather one ROW at a time, the way it did before ws-h31
        /// batched it into two host round trips. At TP=1 this round's binaries are bit-identical
        /// and at TP=2 they commit 2.4 % differently with the step frame ruled out by two other
        /// controls, so this is the one arm that says whether the batch is the cause
        #[arg(long)]
        gather_per_row: bool,

        /// **PAIRING CONTROL** (ws-f37): propose inside the decode step from the argmax, the order
        /// before ws-f37. Refused without `--draft-pack`: with no resident drafter nothing proposes
        #[arg(long)]
        propose_in_step: bool,

        /// **CONTROL** (ws-h32): every step wait on the fixed 200 µs-spin / 250 µs-sleep window
        /// instead of aiming it at the site's recent waits (the pre-ws-h32 schedule, one switch)
        #[arg(long)]
        fixed_step_wait: bool,

        /// FN57: how a step's host thread waits for the card -- `aimed` (ws-h32's sleep-then-poll,
        /// THE 27B DEFAULT), `spin` (the HIP runtime's active wait, one bounded
        /// `hipEventSynchronize`) or `block` (`hipDeviceScheduleBlockingSync`: every runtime wait
        /// sleeps on the signal's interrupt). Absent = the engine's default (27B `aimed`,
        /// Flash-Next `spin`). Forwarded to rank 1.
        #[arg(long)]
        host_wait: Option<String>,

        /// The GDN recurrent state's STORED dtype (C1): `f16` (THE DEFAULT since N8, 2026-09-23,
        /// the user's decision; prod's `--mamba-ssm-cache-dtype float16`: half the state bytes,
        /// fp32 accumulation, only the store narrows) or `f32` (the pairing control, the HF
        /// config's `mamba_ssm_dtype`). Moves greedy ids, so it is gated by end-task quality, not
        /// byte-compare. Forwarded to rank 1 by value
        #[arg(long, default_value = "f16", value_parser = ["f32", "f16"])]
        gdn_state_dtype: String,

        /// ACCEPTED, DOES NOTHING (N5, N6): the default `lm_head` is the checkpoint's own, as the
        /// pack carries it. Kept so older command lines still parse; it prints what the default
        /// resolved to
        #[arg(long)]
        bf16_head: bool,

        /// **PAIRING CONTROL** (N5): plan the MXFP4 `lm_head` (the head before 2026-09-22)
        /// instead of the default exact head (the checkpoint's own, as the pack carries it). Needs
        /// the MXFP4 pack; the default refuses it by name, and this flag refuses an exact-head pack
        #[arg(long)]
        mxfp4_head: bool,

        /// ACCEPTED, DOES NOTHING (N6): the fp8-weight head is what the default resolves to on the
        /// fp8-head pack (this checkpoint's). It prints what the default resolved to
        #[arg(long)]
        fp8_head: bool,

        /// Run every step eagerly: no capture, no replay
        #[arg(long)]
        no_capture: bool,

        /// Upload the step's ids from the host every step instead of letting the argmax feed them
        /// on the device — the A side of the token-feed measurement
        #[arg(long)]
        host_feed: bool,

        /// Do NOT point a padded prefill chunk's attention at the rows the chunk feeds — the
        /// NEGATIVE CONTROL for the continuity arms, which fails where the default passes. The
        /// plan then keeps the bucket's row count, `seqused_k` follows it, and every padded row is
        /// a live query row computed out of the previous request's arena again (ws-l16, ws-h27).
        /// `--no-pad-blank` is the ws-l16 name for the same control and still works.
        #[arg(long, alias = "no-pad-blank")]
        no_live_rows: bool,

        /// Page-locked host MiB for the session-park staging buffer; 0 (the default) means this run
        /// cannot park. One image is `max_ctx / block_size` KV pages plus every GDN state slot, and
        /// the engine refuses to start if this cannot hold one
        #[arg(long, default_value_t = 0)]
        park_staging_mib: u64,

        /// Staging slots to divide the park buffer into. 1 (the default) means a park and a restore
        /// take turns on one buffer; 2 lets them overlap at twice the pinned bytes
        #[arg(long, default_value_t = 1)]
        park_staging_slots: u32,

        /// THE PARK ARM'S NEGATIVE CONTROL: image the state-slot run's BASE for every state,
        /// which is the image a park carried before ws-k14. A restored session then reads a
        /// recurrent snapshot nothing wrote unless it landed back on its own slots, so the park
        /// arm MUST FAIL with this on
        #[arg(long, default_value_t = false)]
        park_image_base: bool,

        /// THE PARK PROBE (ws-k15): every rank dumps the DRAFTER half of every park image per
        /// cache, per layer, per page, and the park arm re-parks the restored session so the two
        /// images can be held against each other byte for byte. Forwarded to rank 1
        #[arg(long, default_value_t = false)]
        park_probe: bool,

        /// MiB kept back from the derived KV pool. The pool takes what the byte ledger has left
        /// after everything else, minus this; the reservations made after it is sized (the index
        /// blob, the ids buffer, and at TP>1 the 504 MiB peer workspace) come out of it
        #[arg(long, default_value_t = 1024)]
        kv_headroom_mib: u64,

        /// P27-7: the byte ledger's law, `lean` by default; `today` (the control: `--budget-mib` /
        /// `--kv-headroom-mib` as given) or `lean` (the pair measured on ROCm 10.1); 27B packs only
        #[arg(long, default_value = ledger_law::DEFAULT_LEDGER)]
        ledger: String,

        /// KV pool blocks, explicitly. 0 (the default) derives the pool from the byte ledger.
        /// A small pool is how an UNDER-PRESSURE arm makes the arenas refuse an admission, which
        /// is the one thing the eviction policy fires on
        #[arg(long, default_value_t = 0)]
        kv_pool_blocks: u32,

        /// ACCEPTANCE: corpus prompts to drive through the pass with this run's own drafter,
        /// one at a time, reporting accepted draft tokens per draft and the distribution. 0 skips
        /// the arm; the ids cannot see what it measures
        #[arg(long, default_value_t = 0)]
        accept_prompts: usize,

        /// Tokens each acceptance prompt commits
        #[arg(long, default_value_t = 64)]
        accept_tokens: usize,

        /// ATTRIBUTION: tokens each acceptance prompt is truncated to for a second pass. The
        /// drafter's context store is decode-only, so a long prompt leaves most of its attention
        /// window unwritten; a short one does not, and the difference is the missing prefill
        /// store. 0 skips the second pass
        #[arg(long, default_value_t = 8)]
        accept_short: usize,

        /// CONTROL: re-run the acceptance pass over the first N prompts with the LAST PREFILL
        /// CHUNK's own proposal turned off (ws-s9). That is the shape this tree had before the
        /// first proposal existed — the mailbox empty once per request, the first decode step at
        /// width 1 — so the difference between the two arms is what the first proposal earns, in
        /// ONE boot and therefore paired. 0 skips it
        #[arg(long, default_value_t = 0)]
        accept_control: usize,

        /// Write the acceptance pass's own COMMITTED ids here, one JSON row a prompt (ws-f36).
        /// Both stacks are greedy over the same corpus ids, so a position where Saber and prod
        /// commit different tokens is a position where a correct draft would have been rejected
        /// — the TARGET's half of ws-f35's uniform per-token deficit.
        /// `tools/gates/target_divergence.py` pairs it with `prod_spec_accept.py --ids-out`
        #[arg(long)]
        accept_ids: Option<PathBuf>,

        /// P27-10 (diagnostic): write the acceptance pass's DRAFTER LATTICES here -- every
        /// proposal's selector candidates + edge scores (`lattice.bin`) beside the committed ids
        /// (`ids.json`) -- the input `tools/capture/p27_10/tree_sim.py` prices tree shapes over
        #[arg(long)]
        draft_lattice_dump: Option<PathBuf>,

        /// H46 (diagnostic): end the gate right after the acceptance pass (the arms before it
        /// still run, so the pass sees the same engine state), for a short paired attribution run
        #[arg(long)]
        stop_after_accept: bool,

        /// Write the SOLO arm's greedy ids (the `ids       solo:` line's tokens) here as one JSON
        /// array (ws-h43): two builds whose solo digests differ are then diffed token by token,
        /// and the first differing position names the decode step to dump. Diagnostic only
        #[arg(long)]
        solo_ids_out: Option<PathBuf>,

        /// RUN ONLY the solo-prompt probe (H44, diagnostic): prompt 0 is prefilled and decoded for
        /// `--solo-probe-steps` steps EAGERLY with a digest after EVERY OP of every layer (all
        /// rows), and each step's committed ids + digests go here as one JSON line. Two builds'
        /// files are diffed offline (`tools/capture/h44_probe_diff.py`): the first (step, op) whose
        /// bytes differ names the kernel that wrote them. The gate does not run
        #[arg(long)]
        solo_probe_out: Option<PathBuf>,

        /// Steps the solo probe runs (H44)
        #[arg(long, default_value_t = 48)]
        solo_probe_steps: usize,

        /// The solo probe step at which every op's input and output rows are ALSO kept raw and
        /// written to `<solo-probe-out>.raw/` (H44: the bytes an off-engine gate replays)
        #[arg(long)]
        solo_probe_raw_step: Option<usize>,

        /// Only ops whose name contains this are written raw (H44)
        #[arg(long, default_value = "")]
        solo_probe_raw_op: String,

        /// THE FIRST PROPOSAL, against prod's own (ws-s9). Corpus indices, comma separated: each
        /// one is admitted, prefilled and released, and the block its LAST PREFILL CHUNK proposed
        /// is printed as ids. `tools/gates/s9_first_block.py` diffs them against
        /// `out.draft_tokens` of step 1 of the multi-step capture, which is prod's own first
        /// proposal for the same prompt
        #[arg(long, default_value = "")]
        first_block_ids: String,

        /// Plan a DFlash2 drafter tier at this block size (prod's is 8 = 1 + 7 speculative
        /// tokens; it must be a power of two). 0, the default, plans no tier. A non-zero value
        /// BILLS the drafter's bytes to the byte ledger before the KV pool is derived from what is
        /// left, checks that every one of its 107 launch sites has a code object, and gives every
        /// admitted session its own drafter context slot and KV pages
        #[arg(long, default_value_t = 0)]
        draft_block: u32,

        /// The DRAFTER's own device-final pack directory, which makes the tier RESIDENT: its
        /// weights uploaded through the same loader the target's take, its two contiguous KV
        /// caches allocated from the plan's own geometry, the proposal's arena laid out and every
        /// workspace of both phases sized before the first capture. Without it `--draft-block`
        /// bills the tier and uploads nothing
        #[arg(long)]
        draft_pack: Option<PathBuf>,

        /// Replay the proposal against a captured decode step's own inputs, through the ENGINE's
        /// launcher, and report every intermediate beside the capture's. Needs `--draft-pack`
        #[arg(long)]
        draft_replay: Option<PathBuf>,

        /// The STANDALONE chain's own dumped intermediates (`saber draft-step --proposal
        /// --no-seed --dump-chain DIR`). With it, `--draft-replay` compares the ENGINE's arena
        /// against the chain's operand by operand and names the first one that departs — which is
        /// the only comparison of the two that admits no rounding class, prod's own taps being an
        /// eager shadow in the body and a few ULPs elsewhere
        #[arg(long)]
        draft_chain_dump: Option<PathBuf>,

        /// Replay the proposal at EVERY step of a page-carrying prod SEQUENCE capture
        /// (`proposal_capture.sh --pages`) and write the block, the selector's scores and its
        /// candidate ids per step. The rate `ws-f35` predicted — how often the walk picks a
        /// different token from prod's — is a property of hundreds of positions and no one-step
        /// capture can carry it. Needs `--draft-pack`
        #[arg(long)]
        draft_seq: Option<PathBuf>,

        /// Where `--draft-seq` writes. Defaults to `<capture>/saber`
        #[arg(long)]
        draft_seq_out: Option<PathBuf>,

        /// Raise the all-reduce's exact/lossy band above every message this run can send, so that
        /// every shape resolves to the EXACT family (TP=2 only). Without it a decode step at a
        /// verify width above 2 crosses the band at batch 4 while the same sequence stepped alone
        /// does not, and the concurrency and mixed-arrival arms then compare two different
        /// arithmetics (ws-l17)
        #[arg(long)]
        ar_exact: bool,

        /// TP=2 (H36): overlap each prefill chunk's all-reduces with their producer GEMM, cut into
        /// row slices of this many rows on a second stream per rank. 0 = off (the default)
        #[arg(long, default_value_t = 0)]
        prefill_ar_overlap: u32,

        /// TP=2 (P27-1d): the reduce's form inside the prefill overlap's slices -- `oneshot`
        /// (H36's, the default, the control) or `split` (an `--ar-push-blocks`-block push, a
        /// command-processor wait, the reduce; bit-exact). Needs `--prefill-ar-overlap`
        #[arg(long, default_value = "oneshot", value_parser = ["oneshot", "split"])]
        ar_form: String,

        /// `--ar-form split`: the push's physical blocks (P27-1c's pick: 16)
        #[arg(long, default_value_t = DEFAULT_AR_PUSH_BLOCKS)]
        ar_push_blocks: u32,

        /// C6: plan every prefill rung of at most this many rows (96..=128) on the decode GEMM
        /// family at one split (bit-exact class). 128 is the default; 0 = the control
        #[arg(long, default_value_t = DEFAULT_SMALL_CHUNK_GEMM)]
        small_chunk_gemm: u32,

        /// C6: those rungs at the reference's split band (a numerics-class change)
        #[arg(long)]
        small_chunk_band: bool,

        /// C7: every prefill rung over one arena sized to the widest (bit-exact; frees KV). The
        /// default since 2026-09-24; accepted as a no-op
        #[arg(long, conflicts_with = "no_shared_prefill_arena")]
        shared_prefill_arena: bool,

        /// An arena per prefill rung (the pre-C7 control)
        #[arg(long)]
        no_shared_prefill_arena: bool,

        /// H46: every decode head over one arena sized to the widest head (bit-exact; frees KV).
        /// The default since 2026-09-24; accepted as a no-op; forwarded to rank 1
        #[arg(long, conflicts_with = "no_shared_head_arena")]
        shared_head_arena: bool,

        /// An arena per decode row count (the pre-H46 control)
        #[arg(long)]
        no_shared_head_arena: bool,

        /// G5: the folded prefill GEMM at the mid-M tile heights (`[[folded_mt]]`; bit-exact class).
        /// The default since 2026-09-24; accepted as a no-op
        #[arg(long, conflicts_with = "no_mid_chunk_gemm")]
        mid_chunk_gemm: bool,

        /// The reference 256-row folded tile in the `[[folded_mt]]` bands (the pre-G5 control)
        #[arg(long)]
        no_mid_chunk_gemm: bool,

        /// G6: the layout law's mid band (`atiled_mid`: tiled producers + the A-tiled family at
        /// 129..512 rows; bit-exact class). The default since 2026-09-24; accepted as a no-op
        #[arg(long, conflicts_with = "no_mid_chunk_atiled")]
        mid_chunk_atiled: bool,

        /// The pre-G6 layout law at 129..512 rows (the control)
        #[arg(long)]
        no_mid_chunk_atiled: bool,

        /// G6: the 48-wide GDN gate on the skinny-N body (`[[skinny]]`; bit-exact class). The
        /// default since 2026-09-24; accepted as a no-op
        #[arg(long, conflicts_with = "no_skinny_gate")]
        skinny_gate: bool,

        /// The 48-wide gate on the reference K-serial body (the control)
        #[arg(long)]
        no_skinny_gate: bool,

        /// CMP27E (the DEFAULT since CMP27F): the b8 (tm4) decode GEMMs on the register-prefetch
        /// twins (bit-exact)
        #[arg(long, conflicts_with = "no_decode_pf")]
        decode_pf: bool,

        /// The CMP27F control: the shipped decode bodies at b8
        #[arg(long)]
        no_decode_pf: bool,

        /// CMP27F (the DEFAULT): the fp8 head at 49..64 rows in one weight pass (bit-exact)
        #[arg(long, conflicts_with = "no_head_m64")]
        head_m64: bool,

        /// The CMP27F control: the fp8 head's two 32-row passes at 49..64 rows
        #[arg(long)]
        no_head_m64: bool,

        /// CMP27K (DEFAULT 17 since CMP27O): the drafter's gate_up at MIN_ROWS..64 rows in one
        /// weight pass (bit-exact); `0` = the control; bare = 33
        #[arg(long, num_args = 0..=1, default_missing_value = "33", value_name = "MIN_ROWS")]
        drafter_gateup_m64: Option<u32>,
        /// CMP27S (on by default): the wht6 reduce's wave-parallel body (`saber_ar_wht_wpar`,
        /// bit-exact); `off` = the control; bare = `on`. Forwarded to rank 1
        #[arg(long, num_args = 0..=1, default_missing_value = "on",
              default_value = saber_engine::device::DEFAULT_AR_WPAR, value_parser = ["on", "off"])]
        ar_wpar: String,
        /// CMP27AE (on by default): the N 5120 dwn4 decode GEMM + wht6 reduce as one
        /// launch (`mxfp4_gemm_decode_arw`, bit-exact); `off` = the control; bare = `on`.
        /// Forwarded to rank 1
        #[arg(long, num_args = 0..=1, default_missing_value = "on",
              default_value = saber_engine::device::DEFAULT_AR_FUSE, value_parser = ["on", "off"])]
        ar_fuse: String,
        /// CMP27L (all on by default since CMP27O): b1 levers, a comma list of conv / adv / head,
        /// or all / off (`off` = the control)
        #[arg(long, default_value = saber_engine::device::DEFAULT_B1_ARMS)]
        b1_arms: String,
        /// CMP27T arms (all on by default; `off` = the control): a comma list of replay / pf2 / picks, or
        /// all / off. Forwarded to rank 1
        #[arg(long, default_value = saber_engine::device::DEFAULT_C4_ARMS)]
        c4_arms: String,
        /// CMP27X / CMP27AG / CMP27AA arms (devrow,dwn4,treeblob,slotdedup,dh64,rowsplit by default;
        /// `devrow,dwn4,treeblob,slotdedup` = the CMP27AA control): a comma list of devrow / dwn4 /
        /// treeblob / slotdedup / slotdedupcheck=N / dh64 / rowsplit / rowstage (CMP27AJ arm) / cuf (CMP27AH / CMP27AI arm,
        /// off by default), or all / off.
        /// Forwarded to rank 1
        #[arg(long, default_value = saber_engine::device::DEFAULT_STEP_ARMS)]
        step_arms: String,

        /// C8 GATE CONTROL: every decode plan in the fused mixed step's decode-row class
        /// (`saber_engine::mixed_class`; never a serving mode). Forwarded to rank 1
        #[arg(long)]
        mixed_class_proxy: bool,

        /// WHICH decode launch-train fusions the decode plans take. ON by default since N8
        /// (2026-09-23, the user's decision) with ws-h37's set (`h37`: the `_nq` epilogue + the
        /// decode GEMM pair, rows 1-2). `all` is the ws-h38 served set (rows 1-3 + 7: adds the split
        /// decode GEMM + reduce + epilogue in one dispatch and the rope + KV append; row 6 was
        /// measured slower and is not in it), opt-in. A bare flag is N8's accepted no-op (it prints
        /// what the default resolved to). `--no-decode-fusion` is the unfused control. Forwarded
        /// to rank 1 by value: a pair where one rank fuses its reduces and the other does not is
        /// two protocols
        #[arg(long, num_args = 0..=1, default_missing_value = "default",
              value_parser = ["default", "h37", "all", "h40", "h42"])]
        decode_fusion: Option<String>,

        /// **PAIRING CONTROL** (N8): plan the decode step unfused (the op sequence before ws-h37).
        /// Bit-exact with the default: only the dispatch count moves. Forwarded to rank 1: a pair
        /// where one rank fuses its reduces and the other does not is two protocols
        #[arg(long)]
        no_decode_fusion: bool,

        /// **P27-16 ARM**: `saber serve --engine-fuse` on the engine (`none` = the default,
        /// `smq`, `gdnnq`, `all`, or a list), resolved into the decode set rank 1 is told
        #[arg(long, default_value = saber_engine::fuse27::DEFAULT_FUSE27, value_parser = parse_fuse27_arg)]
        fuse: String,

        /// Ask the BATCHED proposal whether each request's block is the one it would have got
        /// proposed alone: every proposal of the batched arm runs twice, once as the step's batch
        /// and once per request at the one-request shape out of the same `aux.cat` rows, compared
        /// row for row with a cross-request comparison as the control (ws-s10). It doubles that
        /// arm's proposals, so no acceptance number is taken from a run with it on
        #[arg(long)]
        draft_batch_check: bool,

        /// **PAIRING CONTROL** (H35): store each prefill chunk's context at the drafter's block of
        /// rows a tile (main's law: 256 tiles a 2048-row chunk) instead of the wide store's 256
        #[arg(long)]
        draft_store_narrow: bool,

        /// C6's PAIRING CONTROL: a prefill chunk below the store's class boundary (256 rows) as
        /// 8-row tiles (H35's law) instead of one wide tile at its own norm class
        #[arg(long)]
        draft_store_natural_off: bool,

        /// **GATE** (H35): after each chunk's wide context store, read its rows back on both
        /// ranks, poison them, store them again narrow and compare byte for byte. Doubles the
        /// store; no timing is taken from a run with it on
        #[arg(long)]
        draft_store_check: bool,

        /// **ARM** (P27-3): where the prompt's LAST-chunk drafter context store runs — `on-path`
        /// (inside the chunk, before its token is handed back: today's law, the control) or
        /// `deferred` (stashed with the first proposal and run after the token is handed back,
        /// ahead of the runner's next call: same tiles, same bytes, TTFT without the store)
        #[arg(long, default_value = "on-path", value_parser = ["on-path", "deferred"])]
        drafter_store: String,

        /// P27-6: the drafter's KV law — `paged` (sized by admission from the target's KV
        /// ledger) or `dense` (the control: `max_seqs` runs of `max_ctx` pages)
        #[arg(long, default_value = DEFAULT_DRAFT_KV, value_parser = ["paged", "dense"])]
        draft_kv: String,

        /// P27-9b: the drafter's input embedding — `on` (the target's record, checked
        /// byte-identical at boot) or `off` (the control: its own copy)
        #[arg(long, default_value = DEFAULT_DRAFT_EMBED_SHARE, value_parser = ["on", "off"])]
        draft_embed_share: String,

        /// P27-12: how a proposal ranks the selector's lattice -- `walk` (the device's greedy
        /// walk; the default), `explen` (expected-length chain), or `tandem-b16-calibrated` (nine-node tree)
        #[arg(long, default_value = "walk", value_parser = ["walk", "explen", "tandem-b16-calibrated"])]
        selector_rank: String,

        /// Score the full FP8 vocabulary for single-request native TP2 B8 proposals.
        #[arg(long)]
        draft_full_vocab: bool,

        /// P27-15c: the verify's row form, `ancestor9` (the tree; the default since P27-W2-4) or `off`
        /// (the chain; the control; refused by name and run as `off` where the boot cannot run it)
        /// (`ancestor9-pair` = P27-15f's two-record control: engine / worker only)
        #[arg(long, default_value = "ancestor9", value_parser = ["off", "ancestor9", "ancestor9-pair", "best16"])]
        spec_tree: String,

        /// P27-W2-3: the sampled tree acceptance law, `off` (default) or `on` (an ARM)
        #[arg(long, default_value = "on", value_parser = ["off", "on", "control"])]
        spec_tree_sampled: String,
        /// P27-W2-5 / CMP27G: the GDN decode update as base + stash + replay, `auto` (THE
        /// DEFAULT: `on` unless the f32 state or the best16 tree), `on` or `off` (the control)
        #[arg(long, default_value = "auto", value_parser = ["off", "on", "auto"])]
        gdn_replay: String,

        /// P27-14: the bit-exact host levers, `none` or a list of picks, table, socket (`all`)
        #[arg(long, default_value = saber_engine::host_levers::DEFAULT_HOST_LEVERS, value_parser = parse_host_levers_arg)]
        host_levers: String,

        /// P27-W2-2: the device-step form, `auto` (the default since CMP27P: `cand` at TP = 2
        /// without the T4/T5 race check), `cand` (the candidate seam on the card) or `off` (the
        /// control)
        #[arg(long, default_value = saber_engine::device_step::DEFAULT_DEVICE_STEP, value_parser = parse_device_step_arg)]
        device_step: String,

        /// Refuse to start if the engine's device buffers would exceed this
        #[arg(long, default_value_t = 26000)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// Seconds one engine call may take before the engine is declared unresponsive
        #[arg(long, default_value_t = 120)]
        deadline_s: u64,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,

        /// Explicit RCCL compatibility transport (TP=2, host-capable collectives).
        #[arg(long)]
        rccl_lib: Option<String>,

        /// Native exact collectives over shared host pages (TP=2, no P2P).
        /// With --rccl-lib, messages larger than 64 target rows use RCCL.
        #[arg(long)]
        host_ar: bool,

        /// The Flash-Next path's flags (FN12): used only when the pack's manifest names
        /// `qwen4_exp`; a 27B pack refuses any of them off its default.
        #[command(flatten)]
        fnx: flashnext::run::FnFlags,
    },

    /// Rank 1 of a TP=2 tier: the same engine on the other card, driven by rank 0
    ///
    /// Not run by hand in normal use — `saber engine --tp 2` and `saber serve --engine real
    /// --engine-tp 2` spawn it themselves and own its lifetime, because a worker whose leader is
    /// gone must not be left holding a card.
    EngineWorker {
        /// The TP=2 device-final pack (directory holding rank1.pack)
        #[arg(long)]
        pack: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// This process's rank. 1, and only 1.
        #[arg(long, default_value_t = 1)]
        rank: u32,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 1)]
        device: i32,

        /// The loopback port rank 0 is listening on
        #[arg(long)]
        peer_port: u16,

        /// Refuse to start if this rank's device buffers would exceed this
        #[arg(long, default_value_t = 26000)]
        budget_mib: u64,

        /// D4: the vision tower's arena in patches on this rank (0 = text-only); the leader
        /// forwards its own, because the tower is replicated.
        #[arg(long, default_value_t = 0)]
        vision_max_patches: u32,

        /// P27-11: the leader's `--vision lazy` (the streamed tower), forwarded.
        #[arg(long)]
        vision_lazy: bool,

        /// Sequences this rank holds session state for. Must match rank 0's.
        #[arg(long, default_value_t = 8)]
        max_seqs: u32,

        /// Tokens one sequence may reach. Must match rank 0's.
        #[arg(long, default_value_t = 4096)]
        max_ctx: u32,

        /// Prefill chunk sizes to plan. Must match rank 0's.
        #[arg(long, value_delimiter = ',', default_value = DEFAULT_PREFILL_ROWS)]
        prefill_rows: Vec<u32>,

        /// Decode batches to plan and capture. Must match rank 0's.
        #[arg(long, value_delimiter = ',', default_value = saber_sched::decode_law::DEFAULT_DECODE_BATCHES_ARG)]
        decode_batches: Vec<u32>,

        /// Verify widths to plan and capture. Must match rank 0's.
        #[arg(long, value_delimiter = ',', default_value = "1")]
        decode_widths: Vec<u32>,

        /// Decode row bands to plan and capture (C2). Must match rank 0's; the spawn forwards
        /// them, or `--rectangular-widths` when rank 0 planned none.
        #[arg(long, value_delimiter = ',', default_value = "16,24,32,40,48,56,64")]
        row_bands: Vec<u32>,

        /// Plan no row band (C2's pairing control). Forwarded by rank 0.
        #[arg(long)]
        rectangular_widths: bool,

        /// Page-locked host MiB for THIS rank's half of a session park. Must match rank 0's: a
        /// park at TP=2 images both halves and the follower images its own (ws-t2). 0 disables
        /// parking on this rank, which is what rank 0's own refusal then names.
        #[arg(long, default_value_t = 0)]
        park_staging_mib: u64,

        /// MiB kept back from the derived KV pool, rank 0's `--engine-kv-headroom-mib` forwarded
        /// (G2): the follower's reserves (the all-reduce IPC scratch at an 8192-row bucket is 672
        /// MiB) come out of the same headroom, and a follower pinned at 1024 refused to boot while
        /// the leader had been given more.
        #[arg(long, default_value_t = 1024)]
        kv_headroom_mib: u64,

        /// Staging slots to divide it into: 1 (a park and a restore take turns) or 2.
        #[arg(long, default_value_t = 1)]
        park_staging_slots: u32,

        /// The leader's park-image law, forwarded: both halves of an image are placed by the same
        /// arithmetic, so a follower on the other law would publish its half into another slot of
        /// the run. It is the park arm's negative control and it is a control on BOTH cards
        #[arg(long, default_value_t = false)]
        park_image_base: bool,

        /// The park probe, forwarded: this rank dumps its own half of every park image per cache,
        /// per layer, per page. The question it answers is a two-rank one (ws-k15)
        #[arg(long, default_value_t = false)]
        park_probe: bool,

        /// Run every step eagerly: no capture, no replay. Must match rank 0's.
        #[arg(long)]
        no_capture: bool,

        /// The drafter tier the leader planned, forwarded so this rank's byte ledger holds the
        /// same reservations
        #[arg(long, default_value_t = 0)]
        draft_block: u32,

        /// The drafter's pack, forwarded for the same reason `--draft-block` is: a leader that
        /// made the drafter resident and a follower that did not would declare different node
        /// counts for the same decode shape, because the aux tap is five launches of it
        #[arg(long)]
        draft_pack: Option<PathBuf>,

        /// The leader's all-reduce band, forwarded so both ranks resolve the same family
        #[arg(long)]
        ar_exact: bool,

        /// The leader's prefill all-reduce overlap (H36), forwarded: a pair where one rank slices
        /// its reduces and the other does not is two protocols, and hangs
        #[arg(long, default_value_t = 0)]
        prefill_ar_overlap: u32,

        /// The leader's reduce form in the overlap's slices (P27-1d), forwarded: one rank pushing
        /// split while the other runs the one-shot is two protocols
        #[arg(long, default_value = "oneshot", value_parser = ["oneshot", "split"])]
        ar_form: String,

        /// The leader's drafter KV law (P27-6), forwarded by value: both ranks size one pool
        #[arg(long, default_value = DEFAULT_DRAFT_KV, value_parser = ["paged", "dense"])]
        draft_kv: String,

        /// The leader's embedding share (P27-9b), forwarded by value: both ranks derive one pool
        #[arg(long, default_value = DEFAULT_DRAFT_EMBED_SHARE, value_parser = ["on", "off"])]
        draft_embed_share: String,

        /// The leader's selector rank form (P27-12), forwarded by value: one block digest
        #[arg(long, default_value = "walk", value_parser = ["walk", "explen", "tandem-b16-calibrated"])]
        selector_rank: String,

        /// Score the full FP8 vocabulary for single-request native TP2 B8 proposals.
        #[arg(long)]
        draft_full_vocab: bool,

        /// P27-15c: the verify's row form, `ancestor9` (the tree; the default since P27-W2-4) or `off`
        /// (the chain; the control; refused by name and run as `off` where the boot cannot run it)
        /// (`ancestor9-pair` = P27-15f's two-record control: engine / worker only)
        #[arg(long, default_value = "ancestor9", value_parser = ["off", "ancestor9", "ancestor9-pair", "best16"])]
        spec_tree: String,

        /// P27-W2-3: the sampled tree acceptance law, `off` (default) or `on` (an ARM)
        #[arg(long, default_value = "on", value_parser = ["off", "on", "control"])]
        spec_tree_sampled: String,
        /// P27-W2-5 / CMP27G: the GDN decode update as base + stash + replay, `auto` (THE
        /// DEFAULT: `on` unless the f32 state or the best16 tree), `on` or `off` (the control)
        #[arg(long, default_value = "auto", value_parser = ["off", "on", "auto"])]
        gdn_replay: String,

        /// The leader's host levers (P27-14), forwarded by value: one wire
        #[arg(long, default_value = saber_engine::host_levers::DEFAULT_HOST_LEVERS, value_parser = parse_host_levers_arg)]
        host_levers: String,

        /// The leader's device-step form (P27-W2-2), forwarded by value: one seam
        #[arg(long, default_value = saber_engine::device_step::DEFAULT_DEVICE_STEP, value_parser = parse_device_step_arg)]
        device_step: String,

        /// The leader's push blocks (P27-1d), forwarded
        #[arg(long, default_value_t = DEFAULT_AR_PUSH_BLOCKS)]
        ar_push_blocks: u32,

        /// The leader's small-chunk rungs (C6), forwarded: both ranks plan the same GEMM class
        #[arg(long, default_value_t = DEFAULT_SMALL_CHUNK_GEMM)]
        small_chunk_gemm: u32,

        /// The leader's small-chunk band law (C6), forwarded
        #[arg(long)]
        small_chunk_band: bool,

        /// The leader's shared prefill arena (C7), forwarded: both ranks derive one KV pool (the
        /// default; a no-op)
        #[arg(long, conflicts_with = "no_shared_prefill_arena")]
        shared_prefill_arena: bool,

        /// The leader's control, forwarded: an arena per prefill rung
        #[arg(long)]
        no_shared_prefill_arena: bool,

        /// The leader's shared decode-head arena (H46), forwarded: both ranks derive one KV pool
        /// (the default; a no-op)
        #[arg(long, conflicts_with = "no_shared_head_arena")]
        shared_head_arena: bool,

        /// The leader's control, forwarded: an arena per decode row count
        #[arg(long)]
        no_shared_head_arena: bool,

        /// The leader's mid-M folded tiles (G5), forwarded (the default; accepted as a no-op)
        #[arg(long, conflicts_with = "no_mid_chunk_gemm")]
        mid_chunk_gemm: bool,

        /// The leader's control, forwarded: the reference tile in the `[[folded_mt]]` bands
        #[arg(long)]
        no_mid_chunk_gemm: bool,

        /// The leader's layout-law mid band (G6), forwarded (the default; a no-op)
        #[arg(long, conflicts_with = "no_mid_chunk_atiled")]
        mid_chunk_atiled: bool,

        /// The leader's control, forwarded: the pre-G6 layout law at 129..512 rows
        #[arg(long)]
        no_mid_chunk_atiled: bool,

        /// The leader's skinny gate body (G6), forwarded (the default; a no-op)
        #[arg(long, conflicts_with = "no_skinny_gate")]
        skinny_gate: bool,

        /// The leader's control, forwarded: the 48-wide gate on the reference body
        #[arg(long)]
        no_skinny_gate: bool,

        /// The leader's CMP27E decode prefetch twins, forwarded (both ways: the default is ON)
        #[arg(long, conflicts_with = "no_decode_pf")]
        decode_pf: bool,

        /// The leader's control, forwarded
        #[arg(long)]
        no_decode_pf: bool,

        /// The leader's CMP27F one-pass 64-row head, forwarded (both ways: the default is ON)
        #[arg(long, conflicts_with = "no_head_m64")]
        head_m64: bool,

        /// The leader's control, forwarded
        #[arg(long)]
        no_head_m64: bool,

        /// The leader's CMP27K one-pass 64-row drafter gate_up (its MIN_ROWS), forwarded
        #[arg(long, num_args = 0..=1, default_missing_value = "33", value_name = "MIN_ROWS")]
        drafter_gateup_m64: Option<u32>,
        /// The leader's CMP27S wave-parallel wht6 body (`on` / `off`), forwarded
        #[arg(long, num_args = 0..=1, default_missing_value = "on",
              default_value = saber_engine::device::DEFAULT_AR_WPAR, value_parser = ["on", "off"])]
        ar_wpar: String,
        /// The leader's CMP27AE GEMM + wht6 fold (`on` / `off`), forwarded
        #[arg(long, num_args = 0..=1, default_missing_value = "on",
              default_value = saber_engine::device::DEFAULT_AR_FUSE, value_parser = ["on", "off"])]
        ar_fuse: String,
        /// The leader's CMP27L arms, forwarded (a comma list of conv / adv / head, or all / off)
        #[arg(long, default_value = saber_engine::device::DEFAULT_B1_ARMS)]
        b1_arms: String,
        /// CMP27T arms (all on by default; `off` = the control): a comma list of replay / pf2 / picks, or
        /// all / off. Forwarded to rank 1
        #[arg(long, default_value = saber_engine::device::DEFAULT_C4_ARMS)]
        c4_arms: String,
        /// CMP27X / CMP27AG / CMP27AA arms (devrow,dwn4,treeblob,slotdedup,dh64,rowsplit by default;
        /// `devrow,dwn4,treeblob,slotdedup` = the CMP27AA control): a comma list of devrow / dwn4 /
        /// treeblob / slotdedup / slotdedupcheck=N / dh64 / rowsplit / rowstage (CMP27AJ arm) / cuf (CMP27AH / CMP27AI arm,
        /// off by default), or all / off.
        /// Forwarded to rank 1
        #[arg(long, default_value = saber_engine::device::DEFAULT_STEP_ARMS)]
        step_arms: String,

        /// The leader's C8 gate control, forwarded: decode plans in the fused mixed class
        #[arg(long)]
        mixed_class_proxy: bool,

        /// The leader's decode launch-train fusion set (ws-h37/h38; ON by default since N8),
        /// `off`, `h37` or `all`, forwarded by value: a pair where one rank fuses its reduces and
        /// the other does not is two protocols. No default: the leader always says which
        #[arg(long, value_parser = parse_fusion_set, required = true)]
        decode_fusion: saber_engine::device::DecodeFusion,

        /// Record this rank's own per-step HOST-time breakdown and print it on the way out: a
        /// pair's step is the SLOWER rank's, so rank 1's frames, read-backs and waits belong on
        /// the same table (ws-h31). Forwarded by the leader's spawn
        #[arg(long)]
        host_trace: bool,

        /// The leader's step-frame shape, forwarded so both ranks agree about the protocol
        #[arg(long)]
        whole_step_frame: bool,

        /// Check every step frame's region digest. Forwarded by the leader's spawn; this rank is
        /// the one that makes the check
        #[arg(long)]
        frame_digest: bool,

        /// The race checks (T4), forwarded by the leader's spawn: both ranks run the checked
        /// twins or neither (the collective digest refuses a mixed pair); `shadow` (T6) likewise
        #[arg(long, num_args = 0..=1, default_missing_value = "full", value_name = "MODE")]
        race_check: Option<String>,

        /// T6's boot poison, forwarded by the leader's spawn (a diagnostic)
        #[arg(long, value_name = "SET:BYTE", value_parser = parse_poison)]
        poison_boot: Option<String>,

        /// **GATE CONTROL** (T7): the pre-T7 constant re-seed policy -- only the shared prefill
        /// arena's zero residual is re-seeded, every decode bucket reads its boot zeros. The
        /// card-side failing control of the stale-read law (with `--poison-boot decode:0x3c` it
        /// must move the run) and the paired-step control. Forwarded to rank 1
        #[arg(long)]
        constant_seed_control: bool,

        /// The leader's gather shape, forwarded so both ranks read their picks the same way
        #[arg(long)]
        gather_per_row: bool,

        /// The leader's wait schedule, forwarded so both ranks wait the same way (ws-h32)
        #[arg(long)]
        fixed_step_wait: bool,

        /// FN73 (DEBUG): this follower fails after N steps (the leader's `--fail-rank-after-steps 1:N`)
        #[arg(long, hide = true, default_value_t = 0)]
        fail_after_steps: u64,

        /// The leader's host-wait arm (FN57), forwarded by value
        #[arg(long, default_value = "aimed")]
        host_wait: String,

        /// The leader's GDN recurrent state STORED dtype (C1, N8), `f16` or `f32`, forwarded so
        /// both ranks plan the same graph. No default: the leader always says which
        #[arg(long, value_parser = ["f32", "f16"])]
        gdn_state_dtype: String,

        /// The leader's head (N2, N4, N5: `exact`, `fp8` or `mxfp4`), forwarded so both ranks plan
        /// the same graph. No default: the leader always says which
        #[arg(long, value_parser = parse_target_head)]
        head: saber_engine::head_share::TargetHead,

        /// Seconds one wait may take before this rank decides its leader is gone
        #[arg(long, default_value_t = 120)]
        deadline_s: u64,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,

        /// Explicit RCCL compatibility transport (TP=2, host-capable collectives).
        #[arg(long)]
        rccl_lib: Option<String>,

        /// Native exact collectives over shared host pages (TP=2, no P2P).
        /// With --rccl-lib, messages larger than 64 target rows use RCCL.
        #[arg(long)]
        host_ar: bool,
        /// Independent auxiliary capture plane; rank zero owns the output directory.
        #[arg(long, value_delimiter = ',')]
        draft_capture_layers: Vec<u32>,
        /// Mirror the leader's DSpine feature taps. Only rank zero opens the socket.
        #[arg(long)]
        dspine_socket: Option<PathBuf>,
    },

    /// The P0 autotune sweep: measure every untuned route and write the cache
    Tune {
        /// The Saber manifest, e.g. examples/qwen38-27b-nvfp4-tp2.toml
        manifest: PathBuf,

        /// Directory of committed `.hsaco` + `*.manifest.toml`
        #[arg(long, default_value = "prebuilt/gfx1201")]
        prebuilt: PathBuf,

        /// Where to write the cache
        #[arg(long, default_value = ".cache/autotune/gfx1201-210w.json")]
        out: PathBuf,

        /// An existing cache to extend; routes it already answers are not re-measured
        #[arg(long)]
        autotune_in: Option<PathBuf>,

        /// Only sweep routes whose model op name contains this substring
        #[arg(long)]
        only: Option<String>,

        /// Untimed rounds before timing starts; each launches every variant once
        #[arg(long, default_value_t = 5)]
        warmup: usize,

        /// Timed rounds; each contributes one sample per variant
        #[arg(long, default_value_t = 20)]
        samples: usize,

        /// Refuse a shape whose inputs exceed this. Prod owns the card.
        #[arg(long, default_value_t = 520)]
        budget_mib: u64,

        /// HIP device ordinal within the visible set
        #[arg(long, default_value_t = 0)]
        device: i32,

        /// The enforced cap in watts, when /sys/class/drm is not readable from here. Read it on
        /// the host and pass it; the run then says the cap was asserted rather than measured.
        #[arg(long)]
        power_cap_w: Option<u32>,

        /// host:port of the production engine's metrics endpoint, checked idle before timing
        #[arg(long, default_value = "127.0.0.1:8080")]
        metrics_addr: String,

        /// Print the work list and each route's variants without touching a GPU
        #[arg(long)]
        list: bool,

        /// Print every variant's median under each shape's row, not only winner and runner-up
        #[arg(long)]
        dump_variants: bool,

        /// Bind the all-reduce routes without a number instead of measuring them across two ranks
        #[arg(long)]
        no_ar_timing: bool,

        /// Measure with the Infinity Cache HOT (one set of buffers, re-read every sample).
        ///
        /// The pre-2026-09-19 behaviour, kept for the paired A/B only: the step streams 15.4 GB
        /// of weights per token and never reads a warm line, so a hot median is not its number.
        /// Records written this way carry `IC-HOT` in their provenance.
        #[arg(long)]
        ic_hot: bool,

        /// Do not check the production engine before timing. The run says, loudly, that the
        /// cards were ASSERTED idle rather than observed to be, and every record it writes
        /// carries that in its provenance.
        #[arg(long)]
        no_prod_check: bool,

        #[arg(long, default_value = DEFAULT_HIP_SONAME, help = HIP_LIB_HELP)]
        hip_lib: String,
    },
}

/// `saber serve`'s three parser flags, resolved with the reference server's checks and text (V9), and the boot
/// line that says what they resolved to. Resolved BEFORE anything loads, as the reference server validates its
/// argv before the engine starts.
fn serve_parsers(
    enable: bool,
    no_enable: bool,
    tool_call_parser: Option<String>,
    reasoning_parser: Option<String>,
    defaults: &str,
) -> anyhow::Result<saber_serve::state::ResolvedParsers> {
    use saber_serve::state::{ParserDefaults, ParserFlags};
    let defaults = match defaults {
        "prod" => ParserDefaults::Prod,
        "compat" => ParserDefaults::Compat,
        other => anyhow::bail!("--parser-defaults {other}: `prod` or `compat`"),
    };
    let flags = ParserFlags {
        enable_auto_tool_choice: match (enable, no_enable) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        },
        tool_call_parser,
        reasoning_parser,
        defaults,
    };
    let resolved = flags.resolve().map_err(anyhow::Error::msg)?;
    println!("{}", resolved.banner());
    Ok(resolved)
}

/// `--served-model-name`: the reference server takes a list, Saber serves one name.
/// **CMP27G: the GDN replay arm's default** (`--engine-gdn-replay` / `--gdn-replay`): `auto` = `on`
/// unless the boot cannot carry it -- the f32 stored state (the replay form is the f16 state's),
/// the 16-row best16 tree (its flush is not composed), or a `blockers` entry (the prefix cache, a
/// park tier) -- then `off`, said on one `gdnreplay` line (never silent). `on` / `off` pass
/// through: an explicit `on` the boot cannot carry is refused by name downstream.
fn gdn_replay_resolve(
    flag: &str,
    asked: &str,
    f16_state: bool,
    spec_tree: &str,
    blockers: &[(bool, &str)],
) -> String {
    if asked != "auto" {
        return asked.to_owned();
    }
    let why: Option<String> = if !f16_state {
        Some("the f32 stored GDN state (the replay form is the f16 state's)".to_owned())
    } else if spec_tree == "best16" {
        Some("--engine-spec-tree best16 (the 16-row flush is not composed)".to_owned())
    } else {
        blockers
            .iter()
            .find(|(b, _)| *b)
            .map(|(_, what)| (*what).to_owned())
    };
    match why {
        None => {
            println!("gdnreplay  {flag} auto -> on (CMP27G: the default)");
            "on".to_owned()
        }
        Some(w) => {
            println!("gdnreplay  {flag} auto -> off: STANDS DOWN beside {w}");
            "off".to_owned()
        }
    }
}

fn one_served_name(names: Vec<String>) -> anyhow::Result<String> {
    match names.as_slice() {
        [one] => Ok(one.clone()),
        many => anyhow::bail!(
            "--served-model-name {}: Saber serves ONE model name; the reference server's aliases (prod's unit \
             serves {} names) are not implemented",
            many.join(" "),
            many.len()
        ),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_max_level(if cli.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .with_target(false)
        .init();
    // FN73: the teardown law -- a failed rank sets `saber_serve::fatal`; the process exits
    // `FATAL_EXIT_CODE` (86) with `saber: FATAL rank R: <reason>`, a watchdog `_exit`s it
    // `--fatal-watchdog-s` after the fatal if a wait never returns.
    saber_serve::fatal::install(
        std::time::Duration::from_secs(cli.fatal_watchdog_s.max(1)),
        saber_hip::process::hard_exit,
    );
    let r = dispatch(cli.cmd);
    // every HIP handle is dropped by now (each arm drops its own); the exit is `_exit`
    saber_serve::fatal::exit_if_fatal();
    r
}

fn dispatch(cmd: Cmd) -> anyhow::Result<()> {
    match cmd {
        Cmd::Probe {
            hip_lib,
            no_free_mem,
        } => probe(&hip_lib, !no_free_mem),

        Cmd::Explain {
            manifest,
            prebuilt,
            autotune,
            json,
        } => explain::run(&explain::ExplainArgs {
            manifest,
            prebuilt,
            autotune,
            json,
        }),

        Cmd::ArGate {
            cases,
            rules,
            prebuilt,
            only,
            budget_mib,
            quant_rel_rms,
            hip_lib,
            rank,
            port,
            timing,
            warmup,
            samples,
            iters,
            ar_slot_bytes,
            desync_rank,
            race_check,
            race_control_rank,
        } => {
            // Both ranks spawn their own HipThread inside `ar_gate::run`, because each one has to
            // `hipSetDevice` to its own card and a HipThread is one thread with one current
            // device. Nothing is opened here.
            let failed = ar_gate::run(&ar_gate::ArGateArgs {
                cases,
                rules,
                prebuilt,
                only,
                budget_mib,
                quant_rel_rms,
                hip_lib,
                rank,
                port,
                timing,
                warmup,
                samples,
                iters,
                ar_slot_bytes,
                desync_rank,
                race_check,
                race_control_rank,
            })?;
            if failed > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::D2dGate {
            device,
            bytes,
            regions,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let r = d2d_gate::run(
                &hip,
                &d2d_gate::D2dArgs {
                    device,
                    bytes,
                    regions,
                },
            );
            drop(hip);
            match r? {
                0 => Ok(()),
                n => anyhow::bail!("{n} D2D case(s) failed"),
            }
        }

        Cmd::GdnRace {
            pack,
            world,
            rank,
            layers,
            rows,
            live,
            prompt_ids,
            prompt_index,
            reps,
            state,
            digest,
            every,
            label,
            gap_ms,
            control,
            prebuilt,
            budget_mib,
            device,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let r = gdn_race::run(
                &hip,
                &gdn_race::GdnRaceArgs {
                    pack,
                    world,
                    rank,
                    layers,
                    rows,
                    live,
                    prompt_ids,
                    prompt_index,
                    reps,
                    state,
                    digest,
                    every,
                    prebuilt,
                    budget_mib,
                    device,
                    label,
                    gap_ms,
                    control,
                },
            );
            drop(hip);
            match r {
                Ok(0) => Ok(()),
                Ok(_) => std::process::exit(1),
                Err(e) => Err(e),
            }
        }
        Cmd::ChainBench {
            layers,
            passes,
            solo,
            budget_mib,
            prebuilt,
            device,
            nulls_only,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let r = chain_bench::run(
                &hip,
                &chain_bench::ChainArgs {
                    layers,
                    passes,
                    solo,
                    budget_mib,
                    prebuilt,
                    device,
                    nulls_only,
                },
            );
            drop(hip);
            r
        }

        Cmd::CaptureGate {
            data,
            prebuilt,
            only,
            budget_mib,
            sequence_case_mib,
            replays,
            timed,
            device,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let failed = capture_gate::run(
                &hip,
                &capture_gate::CaptureGateArgs {
                    data: data.into_iter().map(PathBuf::from).collect(),
                    prebuilt,
                    only,
                    budget_mib,
                    sequence_case_mib,
                    replays,
                    timed,
                    device,
                },
            )?;
            drop(hip);
            if failed > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::Gate {
            data,
            prebuilt,
            only,
            budget_mib,
            device,
            autotune,
            manifest,
            emit_cases,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let failed = gate::run(
                &hip,
                &gate::GateArgs {
                    data,
                    prebuilt,
                    only,
                    budget_mib,
                    device,
                    autotune,
                    manifest,
                    emit_cases,
                },
            )?;
            // Shut the HIP thread down BEFORE exiting. `std::process::exit` skips destructors and
            // then runs the C runtime's atexit handlers -- including the HIP runtime's -- while our
            // worker thread is still inside the library, which segfaults on the way out and turns a
            // clean "1 MISMATCH" into exit code 139 that no script can interpret. Measured, not
            // theorised: the gate's negative control did exactly that.
            drop(hip);
            if failed > 0 {
                // A mismatch is the one outcome that must make a script stop.
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::CensusParity {
            metrics,
            census,
            ignore,
        } => {
            let bad = census_parity::run(&metrics, &census, &ignore)?;
            if bad > 0 {
                anyhow::bail!(
                    "{bad} census counter(s) differ between /metrics and the shutdown census"
                );
            }
            Ok(())
        }
        Cmd::SampleCheck {
            rows,
            draws,
            histogram_rows,
            out,
            law_out,
        } => {
            let bad = sample_check::run(&sample_check::SampleCheckArgs {
                rows,
                draws,
                histogram_rows,
                out,
                law_out,
            })?;
            if bad > 0 {
                anyhow::bail!(
                    "{bad} sampled row(s) replay to a different token than the stream committed"
                );
            }
            Ok(())
        }

        Cmd::RaggedAttnGate {
            prebuilt,
            device,
            budget_mib,
            out,
            seed,
            hip_lib,
        } => {
            let args = ragged_attn_gate::RaggedAttnGateArgs {
                prebuilt,
                device,
                budget_mib,
                out,
                seed,
            };
            let hip = open_hip(&hip_lib)?;
            // The HIP thread goes down BEFORE the exit (`Cmd::Gate`'s reason).
            let failed = match ragged_attn_gate::run(&hip, &args) {
                Ok(n) => n,
                Err(e) => {
                    drop(hip);
                    return Err(e);
                }
            };
            drop(hip);
            if failed > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::DeviceRowGate {
            data,
            prebuilt,
            device,
            budget_mib,
            lse_ulp,
            shards,
            rank,
            port,
            hip_lib,
        } => {
            let args = device_row_gate::DeviceRowGateArgs {
                data,
                prebuilt,
                device,
                budget_mib,
                lse_ulp,
                shards,
                rank,
                port,
                hip_lib,
            };
            if args.shards > 1 && args.rank.is_none() {
                // The PARENT of the two-rank arm opens no HIP context at all: it spawns a process
                // per card and reports what they exit with.
                let failed = device_row_shards::run_parent(&args)?;
                if failed > 0 {
                    std::process::exit(1);
                }
                return Ok(());
            }
            let hip = open_hip(&args.hip_lib)?;
            let failed = match (args.shards, args.rank, args.port) {
                (1, _, _) => device_row_gate::run(&hip, &args),
                (2, Some(rank), Some(port)) => device_row_shards::run_rank(&hip, &args, rank, port),
                (n, _, _) => Err(anyhow::anyhow!(
                    "--shards {n} with --rank/--port {:?}/{:?}: this gate splits a row into 1 or \
                     2 shards, and a child of the 2-shard arm needs both",
                    args.rank,
                    args.port
                )),
            };
            // The HIP thread goes down BEFORE the exit, for the reason `Cmd::Gate` states: a
            // `process::exit` with a live one turns a readable verdict into a 139.
            let failed = match failed {
                Ok(n) => n,
                Err(e) => {
                    drop(hip);
                    return Err(e);
                }
            };
            drop(hip);
            if failed > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::Forward {
            pack,
            rank,
            prompt_ids,
            prompt_index,
            prompts,
            max_prompt,
            both_arms,
            ar_exact,
            dump_layers,
            only_ids,
            compare,
            tokens,
            prebuilt,
            budget_mib,
            device,
            timing,
            iters,
            warmup,
            require_complete,
            top2,
            tp,
            tp_rank,
            port,
            hip_lib,
        } => {
            // TP=2 is a different executor, not a flag on this one: two processes, one card each,
            // and every all-reduce real. The parent spawns the ranks and opens no HIP context of
            // its own, so a spawn failure never leaves a context behind.
            if tp == 2 {
                if top2 {
                    anyhow::bail!(
                        "--top2 is a TP=1 flag: at TP=2 this rank's `logits.partial` is HALF the \
                         vocabulary and the two shards are combined on the host only for the \
                         argmax, so a top-2 read off it would be a top-2 of the wrong support"
                    );
                }
                let args = forward_tp2::ForwardTp2Args {
                    pack,
                    prompt_ids,
                    prompt_index,
                    tokens,
                    // The token gate's four: `saber forward` already takes them at TP=1, and the
                    // TP=2 pass is where the `_nq` fusion actually exists.
                    prompts,
                    max_prompt,
                    compare: compare.clone(),
                    both_arms,
                    ar_exact,
                    dump_layers,
                    only_ids,
                    require_complete,
                    prebuilt,
                    budget_mib,
                    timing,
                    iters,
                    warmup,
                    rank: tp_rank,
                    port,
                    hip_lib: hip_lib.clone(),
                };
                let bad = if args.rank.is_some() {
                    let hip = open_hip(&hip_lib)?;
                    let r = forward_tp2::run(Some(&hip), &args);
                    drop(hip);
                    r?
                } else {
                    forward_tp2::run(None, &args)?
                };
                if bad > 0 {
                    std::process::exit(1);
                }
                return Ok(());
            }
            if tp != 1 {
                anyhow::bail!("--tp {tp}: this engine is built for TP=1 and TP=2");
            }
            if dump_layers.is_some() || !only_ids.is_empty() {
                anyhow::bail!(
                    "--dump-layers and --only-ids are TP=2 flags on this command; the TP=1 pass \
                     has its own per-layer harness in `saber block-capture`"
                );
            }
            if ar_exact {
                anyhow::bail!(
                    "--ar-exact is a TP=2 flag: a TP=1 graph declares no `layer.allreduce` at \
                     all, so there is no family to force and accepting it silently would make \
                     two runs look like an A/B that never happened"
                );
            }
            let hip = open_hip(&hip_lib)?;
            let bad = forward::run(
                &hip,
                &forward::ForwardArgs {
                    pack,
                    rank,
                    prompt_ids,
                    prompt_index,
                    prompts,
                    max_prompt,
                    compare,
                    tokens,
                    prebuilt,
                    budget_mib,
                    device,
                    timing,
                    iters,
                    warmup,
                    require_complete,
                    top2,
                },
            )?;
            // The same law the verify arm above states and every other arm obeys: shut the HIP
            // thread down BEFORE `std::process::exit`, which skips destructors and then runs the
            // HIP runtime's atexit handler while our worker thread is still inside the library.
            // This arm was the one that did not, and it is the arm the TOKEN GATE runs: twelve
            // mismatches left as **139**, not 1, so `gate_token.sh` could not tell a disagreement
            // from a crash and `set -e` stopped on a gate that had printed its whole table.
            drop(hip);
            if bad > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
        Cmd::StepTp2 {
            pack,
            prompt_ids,
            prompt_index,
            prebuilt,
            rows,
            ragged_rows,
            ragged_widths,
            spec_width,
            gdn_state_dtype,
            prefill,
            ctx,
            budget_mib,
            replays,
            iters,
            warmup,
            dram_gbs,
            peak_tflops,
            per_op,
            ar_overlap,
            ar_form,
            ar_push_blocks,
            detail,
            small_chunk_gemm,
            small_chunk_class,
            mid_chunk_gemm,
            mid_chunk_atiled,
            skinny_gate,
            no_slot_reuse,
            fusion,
            fusion_control,
            head_on_every_row,
            rank_of,
            port,
            hip_lib,
        } => {
            let args = step_tp2::StepTp2Args {
                no_slot_reuse,
                fusion,
                fusion_control,
                head_on_every_row,
                pack,
                prompt_ids,
                prompt_index,
                prebuilt,
                rows,
                ragged_rows,
                ragged_widths,
                spec_width,
                gdn_state_f16: gdn_state_dtype == "f16",
                prefill,
                ctx,
                budget_mib,
                replays,
                iters,
                warmup,
                dram_gbs,
                peak_tflops,
                per_op,
                ar_overlap,
                ar_push_blocks: ar_push_blocks_of(&ar_form, ar_push_blocks),
                detail,
                small_chunk_gemm,
                small_chunk_class,
                mid_chunk_gemm,
                mid_chunk_atiled,
                skinny_gate,
                rank: rank_of,
                port,
                hip_lib: hip_lib.clone(),
            };
            let bad = if args.rank.is_some() {
                let hip = open_hip(&hip_lib)?;
                let r = step_tp2::run(Some(&hip), &args);
                drop(hip);
                r?
            } else {
                step_tp2::run(None, &args)?
            };
            if bad > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::DraftStep {
            tp,
            spec,
            prebuilt,
            sel_data,
            proposal,
            pack,
            pack_rank,
            pack_launches,
            rope_max_pos,
            iters,
            no_seed,
            dump_chain,
            empty_draft_cache,
            explain,
            budget_mib,
            device,
            hip_lib,
        } => {
            let args = draft_step::DraftStepArgs {
                tp,
                spec,
                prebuilt,
                sel_data,
                proposal,
                pack,
                pack_rank,
                pack_launches,
                rope_max_pos,
                iters,
                no_seed,
                dump_chain,
                empty_draft_cache,
                explain,
                budget_mib,
            };
            // The plan and the routes are pure: no device is opened unless a capture is being
            // replayed or a pack is being loaded, which is what makes this runnable in CI. A
            // `--pack` with no `--hip-lib` still runs the RECORD gate, which is host-only.
            let bad = if args.sel_data.is_some() || args.proposal.is_some() || args.pack_launches {
                let hip = open_hip(&hip_lib)?;
                hip.set_device(device)?;
                let r = draft_step::run(Some(&hip), &args);
                drop(hip);
                r?
            } else {
                draft_step::run(None, &args)?
            };
            if bad > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::VerifyStep {
            pack,
            prompt_ids,
            prompt_index,
            prebuilt,
            ctx,
            spec,
            budget_mib,
            device,
            tp,
            rank_of,
            port,
            hip_lib,
            gdn_replay,
        } => {
            let args = verify_step::VerifyStepArgs {
                pack,
                prompt_ids,
                prompt_index,
                prebuilt,
                ctx,
                spec,
                budget_mib,
                device,
                tp,
                rank_of,
                port,
                hip_lib: hip_lib.clone(),
                gdn_replay,
            };
            // The parent at TP=2 spawns two children and touches no card itself; every other case
            // is one process with one device.
            let bad = if args.tp == 1 || args.rank_of.is_some() {
                let hip = open_hip(&hip_lib)?;
                let r = verify_step::run(Some(&hip), &args);
                drop(hip);
                r?
            } else {
                verify_step::run(None, &args)?
            };
            if bad > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::StepCapture {
            pack,
            verify_width,
            rank,
            prompt_ids,
            prompt_index,
            prebuilt,
            rows,
            prefill,
            prefill_seqs,
            head_on_every_row,
            ctx,
            budget_mib,
            device,
            replays,
            iters,
            warmup,
            per_layer,
            ablate,
            per_op,
            capture_detail,
            dram_gbs,
            peak_tflops,
            fn63,
            fn63_reps,
            fn63_block,
            hsa_lib,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let failed = step_capture::run(
                &hip,
                &step_capture::StepCaptureArgs {
                    pack,
                    verify_width,
                    rank,
                    prompt_ids,
                    prompt_index,
                    prebuilt,
                    rows,
                    prefill,
                    prefill_seqs,
                    head_on_every_row,
                    ctx,
                    budget_mib,
                    device,
                    replays,
                    iters,
                    warmup,
                    per_layer,
                    ablate,
                    per_op,
                    detail: capture_detail,
                    dram_gbs,
                    peak_tflops,
                    fn63: fn63.map(|out| crate::fn63_hsa::Fn63Opts {
                        out,
                        reps: fn63_reps,
                        block: fn63_block,
                        hsa_lib,
                    }),
                },
            );
            drop(hip);
            if failed? > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::BlockCapture {
            data,
            prebuilt,
            pack,
            rank,
            only,
            replays,
            iters,
            warmup,
            budget_mib,
            device,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let failed = block_capture::run(
                &hip,
                &block_capture::BlockCaptureArgs {
                    data,
                    prebuilt,
                    pack,
                    rank,
                    only,
                    replays,
                    iters,
                    warmup,
                    budget_mib,
                    device,
                },
            );
            drop(hip);
            if failed? > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::Layer {
            data,
            prebuilt,
            pack,
            rank,
            only,
            provenance,
            corrupt_layout,
            full_rank,
            budget_mib,
            timing,
            iters,
            warmup,
            device,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let failed = layer::run(
                &hip,
                &layer::LayerArgs {
                    data,
                    prebuilt,
                    pack,
                    rank,
                    only,
                    provenance,
                    corrupt_layout,
                    full_rank,
                    budget_mib,
                    device,
                    timing,
                    iters,
                    warmup,
                },
            );
            // Same rule as `gate`: the HIP thread goes down before this function returns, never
            // after `std::process::exit` has started running atexit handlers underneath it.
            drop(hip);
            if failed? > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::FlashnextProfile(a) => fn_profile::run(&a),
        Cmd::Fn119Oracle(a) => flashnext::dstep::oracle(&a),
        Cmd::FlashnextPack {
            pack,
            fp8_ckpt,
            mtp_experts,
            trunk,
            out_pack,
            hc8,
            w4_rule,
            calib,
            vision,
            vision_ckpt,
            lazy_runs,
            lazy_runs_out,
            lazy_runs_backup,
            lazy_runs_fresh,
            experts,
            source,
            fp4_ckpt,
            dry_run,
            layers,
            report,
        } => match (mtp_experts.as_deref(), trunk.as_deref()) {
            _ if source.is_some() => {
                anyhow::ensure!(
                    source.as_deref() == Some("fp4-local"),
                    "--source fp4-local is the only source"
                );
                anyhow::ensure!(
                    mtp_experts.is_none()
                        && trunk.is_none()
                        && vision.is_none()
                        && experts.is_none()
                        && !hc8
                        && lazy_runs.is_empty(),
                    "flashnext-pack: --source is its own build"
                );
                match (dry_run, layers.as_deref(), out_pack.as_deref()) {
                    (true, None, None) => flashnext::fp4local::dry_run(&fp4_ckpt, &report).map(|_| ()),
                    (false, Some(l), Some(o)) => {
                        let (a, b) = l
                            .split_once(':')
                            .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                            .ok_or_else(|| anyhow::anyhow!("--layers A:B, not {l}"))?;
                        flashnext::fp4local::build_smoke(&fp4_ckpt, o, (a, b))
                    }
                    _ => anyhow::bail!(
                        "--source fp4-local: --dry-run, or --layers A:B --out-pack DIR (the full pack is FN133 P3+)"
                    ),
                }
            }
            _ if hc8 => {
                anyhow::ensure!(
                    mtp_experts.is_none()
                        && trunk.is_none()
                        && vision.is_none()
                        && experts.is_none()
                        && lazy_runs.is_empty(),
                    "flashnext-pack: --hc8 is its own build"
                );
                flashnext::hc8::build(&pack).map(|_| ())
            }
            _ if experts.is_some() => {
                anyhow::ensure!(
                    mtp_experts.is_none()
                        && trunk.is_none()
                        && vision.is_none()
                        && lazy_runs.is_empty(),
                    "flashnext-pack: --experts is its own build"
                );
                let out = out_pack.clone().unwrap_or_else(|| {
                    let mut p = pack.clone().into_os_string();
                    p.push("-tp2");
                    PathBuf::from(p)
                });
                match experts.as_deref() {
                    Some("tp2") => {
                        let b = flashnext::xsplit::build(&pack, &out)?;
                        println!(
                            "fn80a: {} records a rank -> {} ({} / {} B)",
                            b.records,
                            b.out.display(),
                            b.bytes[0],
                            b.bytes[1]
                        );
                        flashnext::xsplit::check(&pack, &b.out).map(|_| ())
                    }
                    Some("tp2-check") => flashnext::xsplit::check(&pack, &out).map(|_| ()),
                    Some(t) => anyhow::bail!("--experts tp2 | tp2-check, not {t}"),
                    None => unreachable!(),
                }
            }
            _ if !lazy_runs.is_empty() => {
                anyhow::ensure!(
                    mtp_experts.is_none() && trunk.is_none() && vision.is_none(),
                    "flashnext-pack: --lazy-runs is its own build"
                );
                flashnext::lazyset::build(
                    &pack,
                    &lazy_runs,
                    lazy_runs_out.as_deref(),
                    &lazy_runs_backup,
                    lazy_runs_fresh,
                )
                .map(|_| ())
            }
            _ if vision.is_some() => {
                anyhow::ensure!(
                    mtp_experts.is_none() && trunk.is_none(),
                    "flashnext-pack: --vision is its own build"
                );
                anyhow::ensure!(
                    vision.as_deref() == Some("bf16"),
                    "--vision bf16 is the only form"
                );
                flashnext::vision::build(&pack, &vision_ckpt).map(|_| ())
            }
            (Some(m), None) => {
                anyhow::ensure!(m == "fp8", "--mtp-experts fp8 is the only form");
                flashnext::mtp8::build(&pack, &fp8_ckpt).map(|_| ())
            }
            (None, Some(t)) => {
                let out = |sfx: &str| {
                    out_pack.clone().unwrap_or_else(|| {
                        let mut p = pack.clone().into_os_string();
                        p.push(sfx);
                        PathBuf::from(p)
                    })
                };
                match t {
                    "w4" => match (w4_rule.parse()?, calib.as_deref()) {
                        (flashnext::trunk4::Rule::Gptq, Some(c)) => {
                            flashnext::trunk4::build_calibrated(&pack, &out("-trunk4c"), c)
                                .map(|_| ())
                        }
                        (flashnext::trunk4::Rule::Gptq, None) => {
                            anyhow::bail!("--w4-rule gptq needs --calib DIR")
                        }
                        (_, Some(_)) => anyhow::bail!("--calib goes with --w4-rule gptq"),
                        (rule, None) => {
                            flashnext::trunk4::build(&pack, &out("-trunk4"), rule).map(|_| ())
                        }
                    },
                    "w8" => flashnext::trunk8::build(&pack, &out("-trunk8")).map(|_| ()),
                    _ => anyhow::bail!("--trunk w4 | w8, not {t}"),
                }
            }
            _ => anyhow::bail!(
                "flashnext-pack: exactly one of --mtp-experts fp8 / --trunk w4|w8 / --vision bf16"
            ),
        },
        Cmd::Fn150MoeCheck {
            pack,
            input,
            out,
            prebuilt,
            device,
            singles,
            rank,
            plain,
            fused,
            b8,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let r = flashnext::fn150::run(
                &hip,
                &flashnext::fn150::Fn150Args {
                    pack: &pack,
                    prebuilt: &prebuilt,
                    input: &input,
                    out: &out,
                    device,
                    singles,
                    rank,
                    plain,
                    fused,
                    b8,
                },
            );
            drop(hip);
            r
        }
        Cmd::FlashnextMoeBench {
            mode,
            pack,
            rank,
            routes,
            profile,
            tokens,
            steps,
            warmup,
            ple_ids,
            ckpt_root,
            sweep,
            prebuilt,
            device,
            xtier,
            hip_lib,
        } => {
            let args = fn_moe_bench::BenchArgs {
                mode,
                pack,
                rank,
                routes,
                profile,
                prebuilt,
                device,
                tokens,
                steps,
                warmup,
                ple_ids,
                ckpt_root,
                sweep,
                flags: xtier,
            };
            if args.mode == "boot" {
                return fn_moe_bench::boot(&args);
            }
            let hip = open_hip(&hip_lib)?;
            let r = fn_moe_bench::run(&hip, &args);
            drop(hip);
            r
        }

        Cmd::ParkBench {
            kv_bytes,
            gdn_bytes,
            iters,
            warmup,
            slots,
            nvme_path,
            pinned_budget,
            nvme_budget,
            keep_file,
            no_corrupt,
            device,
            hash_threads,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let result = park_bench::run(
                &hip,
                &park_bench::ParkBenchArgs {
                    kv_bytes,
                    gdn_bytes,
                    iters,
                    warmup,
                    slots,
                    nvme_path,
                    pinned_budget_bytes: pinned_budget,
                    nvme_budget_bytes: nvme_budget,
                    keep_file,
                    corrupt: !no_corrupt,
                    device,
                    hash_threads,
                },
            );
            // The HIP thread goes down before this returns, for the same reason the gate drops it
            // before `std::process::exit`.
            drop(hip);
            result
        }

        Cmd::Load {
            pack,
            rank,
            from_record,
            limit_bytes,
            read_offset,
            fixed_ring,
            slots,
            slot_bytes,
            qd,
            chunk_bytes,
            hash_threads,
            stream_budget,
            no_verify,
            vram_budget,
            pinned_budget,
            no_io_pass,
            events,
            check_records,
            check_shift,
            device,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            let result = load::run(
                &hip,
                &load::LoadArgs {
                    pack,
                    rank,
                    start_record: from_record,
                    limit_bytes,
                    read_offset,
                    fixed_ring,
                    slots,
                    slot_bytes: slot_bytes as usize,
                    queue_depth: qd,
                    chunk_bytes: chunk_bytes as usize,
                    hash_threads,
                    stream_budget_bytes: stream_budget,
                    no_verify,
                    vram_budget_bytes: vram_budget,
                    pinned_budget_bytes: pinned_budget,
                    io_pass: !no_io_pass,
                    events,
                    check_records,
                    check_shift,
                    device,
                },
            );
            // Drop the HIP thread before returning: the arena, the staging ring and the events are
            // freed by their handles, and an error path that skipped this would run the HIP
            // runtime's atexit handlers with the worker still inside the library (exit 139).
            drop(hip);
            result
        }

        Cmd::BenchSteploop {
            manifest,
            prebuilt,
            launches,
            steps,
            warmup,
            channel_capacity,
            dummy_n,
            device,
            hip_lib,
        } => {
            let hip = open_hip(&hip_lib)?;
            bench::run(
                &hip,
                &bench::BenchArgs {
                    manifest,
                    prebuilt,
                    launches,
                    steps,
                    warmup,
                    channel_capacity,
                    dummy_n,
                    device,
                },
            )
        }
        Cmd::Serve {
            engine,
            mock,
            pack,
            rank,
            engine_device,
            engine_tp,
            vision_max_patches,
            vision,
            drafter_rope_index,
            drafter_rope_temporal,
            limit_mm_images,
            engine_peer_device,
            engine_max_seqs,
            engine_max_ctx,
            engine_prefill_rows,
            engine_decode_batches,
            engine_decode_widths,
            engine_width_rule,
            engine_ragged_widths,
            engine_row_bands,
            engine_rectangular_widths,
            engine_ragged_ceiling,
            engine_mixed_prefill,
            engine_max_batched_tokens,
            engine_prefill_fill_above,
            engine_shared_prefill_arena,
            no_engine_shared_prefill_arena,
            engine_shared_head_arena,
            no_engine_shared_head_arena,
            engine_no_capture,
            engine_draft_block,
            engine_draft_pack,
            ar_exact,
            engine_prefill_ar_overlap,
            engine_ar_form,
            engine_ar_push_blocks,
            engine_small_chunk_gemm,
            engine_small_chunk_band,
            engine_mid_chunk_gemm,
            no_engine_mid_chunk_gemm,
            engine_mid_chunk_atiled,
            no_engine_mid_chunk_atiled,
            engine_skinny_gate,
            no_engine_skinny_gate,
            engine_decode_pf,
            no_engine_decode_pf,
            engine_head_m64,
            no_engine_head_m64,
            engine_drafter_gateup_m64,
            engine_ar_wpar,
            engine_ar_fuse,
            engine_b1_arms,
            engine_c4_arms,
            engine_step_arms,
            engine_mixed_class_proxy,
            engine_host_trace,
            engine_decode_fusion,
            engine_no_decode_fusion,
            engine_fuse,
            engine_whole_step_frame,
            engine_frame_digest,
            engine_race_check,
            engine_gather_per_row,
            fail_rank_after_steps,
            host_wait,
            engine_draft_store_narrow,
            engine_draft_store_natural_off,
            engine_draft_store_check,
            engine_drafter_store,
            engine_draft_kv,
            engine_draft_embed_share,
            engine_selector_rank,
            engine_draft_full_vocab,
            engine_spec_tree,
            engine_spec_tree_sampled,
            engine_gdn_replay,
            engine_host_levers,
            engine_device_step,
            engine_gdn_state_dtype,
            engine_bf16_head,
            mxfp4_head,
            engine_fp8_head,
            engine_budget_mib,
            engine_park_staging_mib,
            engine_kv_headroom_mib,
            engine_ledger,
            engine_kv_pool_blocks,
            served_model_name,
            enable_auto_tool_choice,
            no_enable_auto_tool_choice,
            tool_call_parser,
            reasoning_parser,
            parser_defaults,
            max_logprobs,
            sample_dump,
            keep_stale_anchor_blocks,
            step_keyed_draws,
            enable_prefix_caching,
            prefix_cache_verify,
            prefix_cache_host_mib,
            prefix_cache_min_tokens,
            warm_prompts,
            spec_lookup,
            spec_lookup_ngram,
            spec_lookup_min_match,
            propose_in_step,
            engine_deadline_s,
            hip_lib,
            engine_rccl_lib,
            engine_host_ar,
            engine_draft_capture_dir,
            engine_dspine_socket,
            engine_draft_capture_layers,
            manifest,
            prebuilt,
            autotune,
            tokenizer,
            chat_template,
            host,
            port,
            kv_gib,
            state_gib,
            step_delay_ms,
            park_ram_gib,
            park_nvme_gib,
            park_nvme_path,
            park_on_pressure,
            park_victim,
            park_max_per_session,
            park_max_restore_attempts,
            engine_park_fail_restores,
        } => match engine.as_str() {
            // The real engine: no manifest and no autotune cache on this path, because the
            // implementation choice comes from `saber-exec`'s own rule tables and not from a bound
            // plan (a divergence recorded in the branch report, not a decision hidden here).
            "real" => {
                let parsers = serve_parsers(
                    enable_auto_tool_choice,
                    no_enable_auto_tool_choice,
                    tool_call_parser,
                    reasoning_parser,
                    &parser_defaults,
                )?;
                let served_model_name = one_served_name(served_model_name)?;
                // P27-W2-5: the GDN replay arm is refused by name beside the prefix cache (its
                // host snapshot / reload is not gated under the arm) and beside parking (a park
                // of a session whose base owes a replay is refused at run time).
                // CMP27G: the replay arm COMPOSES with the tree default (`ancestor9`): the tree
                // reads the sequence's base (materialized first when it owes a chain step's rows)
                // and its flush advances the base in place. The 16-row form is refused at boot.
                let engine_gdn_replay = gdn_replay_resolve(
                    "--engine-gdn-replay",
                    &engine_gdn_replay,
                    engine_gdn_state_dtype == "f16",
                    &engine_spec_tree,
                    &[
                        (
                            engine_dspine_socket.is_some() && engine_decode_widths.contains(&16),
                            "DSpine width 16 (GDN replay holds at most 8 rows)",
                        ),
                        (enable_prefix_caching == "on", "--enable-prefix-caching on"),
                        (park_ram_gib > 0.0 || park_nvme_gib > 0.0, "a park tier"),
                    ],
                );
                if engine_gdn_replay == "on" && enable_prefix_caching == "on" {
                    anyhow::bail!(
                        "--engine-gdn-replay on with --enable-prefix-caching on: the prefix \
                         cache's GDN snapshot is not gated under the replay arm (P27-W2-5)"
                    );
                }
                if engine_gdn_replay == "on" && (park_ram_gib > 0.0 || park_nvme_gib > 0.0) {
                    anyhow::bail!(
                        "--engine-gdn-replay on with a park tier: parking is refused under the \
                         replay arm (P27-W2-5)"
                    );
                }
                if engine_spec_tree == "best16" && engine_max_seqs != 1 {
                    anyhow::bail!("--engine-spec-tree best16 currently requires --engine-max-seqs 1: its chain-after-tree flush is not implemented");
                }
                if engine_dspine_socket.is_some()
                    && (enable_prefix_caching != "off"
                        || park_ram_gib > 0.0
                        || park_nvme_gib > 0.0
                        || spec_lookup == "on"
                        || propose_in_step)
                {
                    anyhow::bail!("DSpine socket needs --enable-prefix-caching off, disabled park tiers, after-commit proposals and no lookup drafter");
                }
                if engine_draft_capture_dir.is_some()
                    && (enable_prefix_caching != "off"
                        || engine_spec_tree != "off"
                        || engine_max_seqs != 1)
                {
                    anyhow::bail!("target feature capture requires --enable-prefix-caching off, --engine-spec-tree off and --engine-max-seqs 1");
                }
                let Some(pack) = pack else {
                    anyhow::bail!(
                        "--engine real needs --pack <device-final pack>; build one with \
                         `tools/fetch_packs.sh`"
                    );
                };
                // P27-7: the ledger law resolves the two reserves ONCE, here; rank 1 is told the
                // resolved numbers by value (`--budget-mib` / `--kv-headroom-mib`).
                let (engine_budget_mib, engine_kv_headroom_mib) =
                    ledger_law::Ledger::parse(&engine_ledger)?.resolve_said(
                        engine_budget_mib,
                        engine_kv_headroom_mib,
                        ledger_law::box_p2pmem_mib(),
                    );
                let vision_on = matches!(vision.as_deref(), Some("lazy" | "on"));
                engine::serve(&engine::ServeArgs {
                    vision_max_patches: if vision.as_deref() == Some("off") {
                        0
                    } else if vision_on && vision_max_patches == 0 {
                        65_536
                    } else {
                        vision_max_patches
                    },
                    vision_lazy: vision.as_deref() == Some("lazy"),
                    image_limit: limit_mm_images,
                    engine: engine::EngineArgs {
                        vision_max_patches: 0,
                        vision_lazy: false,
                        // The index law is the default (the user, 2026-09-23); the temporal
                        // axis is the control. `drafter_rope_index` is a no-op that names it.
                        drafter_rope_index: drafter_rope_law(
                            drafter_rope_index,
                            drafter_rope_temporal,
                        ),
                        pack: pack.clone(),
                        world: engine_tp,
                        rank,
                        peer_port: 0,
                        follower: false,
                        peer_device: engine_peer_device,
                        prebuilt,
                        hip_lib,
                        rccl_lib: engine_rccl_lib,
                        host_ar: engine_host_ar,
                        dspine_socket: engine_dspine_socket,
                        draft_capture_dir: engine_draft_capture_dir,
                        draft_capture_layers: engine_draft_capture_layers,
                        device: engine_device,
                        budget_mib: engine_budget_mib,
                        max_seqs: engine_max_seqs,
                        max_ctx: engine_max_ctx,
                        prefill_rows: engine_prefill_rows,
                        decode_batches: engine_decode_batches,
                        decode_widths: engine_decode_widths,
                        decode_bands: decode_bands_of(
                            engine_ragged_widths,
                            engine_rectangular_widths,
                            engine_row_bands,
                        ),
                        ragged_ceiling: saber_sched::RaggedCeiling::parse(&engine_ragged_ceiling)?,
                        width_rule: saber_sched::WidthRule::parse(&engine_width_rule)?,
                        mixed_prefill: mixed_prefill_of(
                            engine_mixed_prefill.as_deref(),
                            engine_max_batched_tokens,
                        )?,
                        prefill_fill_above: engine_prefill_fill_above,
                        deadline_s: engine_deadline_s,
                        capture: !engine_no_capture,
                        device_feed: true,
                        live_row_attn: true,
                        // THE SERVE PLANE PARKS SINCE ws-k14. It was `--engine mock` only:
                        // a real serve held no staging buffer, so the one path an eviction
                        // actually takes had never run against the card. Both halves are flags
                        // now — this buffer and `--park-ram-gib` — and the engine refuses to
                        // start if what is asked for cannot hold one image at `max_ctx`.
                        park_staging_mib: engine_park_staging_mib,
                        // The TIER's bytes go with the staging buffer, because the engine is what
                        // declares the pinned pool both of them are billed against (ws-k14).
                        park_ram_gib,
                        park_staging_slots: 1,
                        park_image_base: false,
                        park_probe: false,
                        kv_headroom_mib: engine_kv_headroom_mib,
                        kv_pool_blocks: engine_kv_pool_blocks,
                        // A serve plans a drafter tier only when asked: its bytes come out of
                        // the KV pool, so a serve that does not speculate should not hold them.
                        // With `--engine-draft-pack` the tier is resident and every decode step
                        // proposes; with `--engine-draft-block` alone it is billed and nothing is
                        // uploaded, which is how the pool arithmetic is compared.
                        draft_block: engine_draft_block,
                        draft_pack: engine_draft_pack,
                        // A replay is a GATE's arm, not a server's: it seeds a capture's own
                        // inputs into the drafter's buffers.
                        draft_replay: None,
                        draft_chain_dump: None,
                        draft_seq: None,
                        draft_seq_out: None,
                        // THE SAME BAND AS `saber engine`. It was hardcoded `false` here while
                        // `saber engine` took the flag, so the two planes could not be made to
                        // resolve the same all-reduce family for the same shape by configuration
                        // (ws-q1 open item 3). `Worker::spawn` forwards it to rank 1, where one
                        // number in a rules table decides the SYMBOL each rank dispatches.
                        ar_exact,
                        draft_batch_check: false,
                        draft_store_narrow: engine_draft_store_narrow,
                        draft_store_natural_off: engine_draft_store_natural_off,
                        draft_store_check: engine_draft_store_check,
                        draft_store_deferred: engine_drafter_store == "deferred",
                        draft_kv_paged: engine_draft_kv == "paged",
                        draft_embed_share: engine_draft_embed_share == "on",
                        draft_full_vocab: engine_draft_full_vocab,
                        selector_rank: saber_engine::selector_rank::SelectorRank::parse(
                            &engine_selector_rank,
                        )
                        .unwrap_or_default(),
                        spec_tree: saber_engine::spec_tree::SpecTreeForm::parse(&engine_spec_tree)
                            .unwrap_or_default(),
                        spec_tree_sampled: engine_spec_tree_sampled != "off",
                        spec_tree_sampled_control: engine_spec_tree_sampled == "control",
                        gdn_replay: engine_gdn_replay == "on",
                        host_levers: saber_engine::host_levers::HostLevers::parse(
                            &engine_host_levers,
                        )
                        .unwrap_or_default(),
                        device_step: saber_engine::device_step::DeviceStep::parse(
                            &engine_device_step,
                        )
                        .unwrap_or_default(),
                        prefill_ar_overlap: engine_prefill_ar_overlap,
                        prefill_ar_push_blocks: ar_push_blocks_of(
                            &engine_ar_form,
                            engine_ar_push_blocks,
                        ),
                        small_chunk_gemm: engine_small_chunk_gemm,
                        small_chunk_band: engine_small_chunk_band,
                        shared_prefill_arena: on_unless_off(
                            engine_shared_prefill_arena,
                            no_engine_shared_prefill_arena,
                        ),
                        shared_head_arena: on_unless_off(
                            engine_shared_head_arena,
                            no_engine_shared_head_arena,
                        ),
                        mid_chunk_gemm: mid_chunk_gemm_law(
                            engine_mid_chunk_gemm,
                            no_engine_mid_chunk_gemm,
                        ),
                        mid_chunk_atiled: on_unless_off(
                            engine_mid_chunk_atiled,
                            no_engine_mid_chunk_atiled,
                        ),
                        skinny_gate: on_unless_off(engine_skinny_gate, no_engine_skinny_gate),
                        decode_pf: on_unless_off(engine_decode_pf, no_engine_decode_pf),
                        head_m64: on_unless_off(engine_head_m64, no_engine_head_m64),
                        drafter_gateup_m64: engine_drafter_gateup_m64
                            .unwrap_or(saber_engine::device::DEFAULT_DRAFTER_GATEUP_M64),
                        ar_wpar: engine_ar_wpar == "on",
                        ar_fuse: engine_ar_fuse == "on",
                        b1_arms: saber_engine::device::B1Arms::parse(&engine_b1_arms)
                            .map_err(anyhow::Error::msg)?,
                        c4_arms: saber_engine::device::C4Arms::parse(&engine_c4_arms)
                            .map_err(anyhow::Error::msg)?,
                        step_arms: saber_engine::device::StepArms::parse(&engine_step_arms)
                            .map_err(anyhow::Error::msg)?,
                        mixed_class_proxy: engine_mixed_class_proxy,
                        decode_fusion: saber_engine::fuse27::Fuse27::parse(&engine_fuse)
                            .and_then(|f| {
                                f.check_mixed(
                                    mixed_prefill_of(
                                        engine_mixed_prefill.as_deref(),
                                        engine_max_batched_tokens,
                                    )
                                    .is_ok_and(|m| m.enabled && m.fused),
                                )
                            })
                            .map_err(anyhow::Error::msg)?
                            .resolve(decode_fusion_of(
                                engine_decode_fusion.as_deref(),
                                engine_no_decode_fusion,
                                "--engine-decode-fusion",
                                "--engine-no-decode-fusion",
                            )?)
                            .map_err(anyhow::Error::msg)?,
                        // THE SERVE PLANE IS NOT TRACED unless a diagnostic serve names
                        // `--engine-host-trace` (C5). `saber serve` is where the published tok/s
                        // comes from, and an instrument armed underneath it would be exactly the
                        // ordering blind spot ws-t3 closed (GATES.md blind spot 19): a traced
                        // serve's numbers are attribution, never a published cell.
                        host_trace: engine_host_trace,
                        // The serve plane's controls are OFF unless named (ws-p12): they exist here
                        // because the eight-in-flight contamination is a SERVE-plane red that the
                        // engine's own arms do not reproduce, so the controls have to run where
                        // the red is.
                        whole_step_frame: engine_whole_step_frame,
                        frame_digest: engine_frame_digest,
                        race_check: engine::race_mode(engine_race_check.as_deref())
                            .map_err(anyhow::Error::msg)?
                            .0,
                        race_shadow: engine::race_mode(engine_race_check.as_deref())
                            .map_err(anyhow::Error::msg)?
                            .1,
                        poison_boot: None,
                        constant_seed_control: false,
                        gather_per_row: engine_gather_per_row,
                        propose_in_step,
                        fixed_step_wait: false,
                        fail_after_steps: engine::fail_rank1_after(
                            fail_rank_after_steps.as_deref(),
                        )?,
                        host_wait: engine::host_wait_27b(host_wait.as_deref())
                            .map_err(anyhow::Error::msg)?,
                        gdn_state_f16: engine_gdn_state_dtype == "f16",
                        head: target_head(
                            mxfp4_head,
                            &[
                                (engine_fp8_head, "--engine-fp8-head"),
                                (engine_bf16_head, "--engine-bf16-head"),
                            ],
                            &pack,
                        )?,
                    },
                    tokenizer,
                    chat_template,
                    host,
                    port,
                    served_model_name,
                    max_logprobs,
                    parsers,
                    sample_dump,
                    keep_stale_anchor_blocks,
                    step_keyed_draws,
                    spec_lookup: if spec_lookup == "on" {
                        Some(
                            saber_engine::LookupConfig::parse(
                                &spec_lookup_ngram,
                                spec_lookup_min_match,
                            )
                            .map_err(|e| anyhow::anyhow!(e))?,
                        )
                    } else {
                        None
                    },
                    prefix_cache: if enable_prefix_caching == "on"
                        || prefix_cache_verify
                        || warm_prompts.is_some()
                    {
                        saber_sched::PrefixCachePolicy {
                            verify: prefix_cache_verify,
                            host_budget_bytes: prefix_cache_host_mib << 20,
                            min_tokens: prefix_cache_min_tokens,
                            ..saber_sched::PrefixCachePolicy::on()
                        }
                    } else {
                        saber_sched::PrefixCachePolicy::OFF
                    },
                    warm_prompts,
                    park_nvme_gib,
                    park_nvme_path,
                    park_on_pressure,
                    park_policy: park_policy_of(
                        park_on_pressure,
                        &park_victim,
                        park_max_per_session,
                        park_max_restore_attempts,
                    )?,
                    park_fail_restores: engine_park_fail_restores,
                })
            }
            "mock" => {
                let parsers = serve_parsers(
                    enable_auto_tool_choice,
                    no_enable_auto_tool_choice,
                    tool_call_parser,
                    reasoning_parser,
                    &parser_defaults,
                )?;
                let _ = one_served_name(served_model_name)?;
                let Some(mock) = mock else {
                    anyhow::bail!("--engine mock needs --mock <script.toml>");
                };
                if ar_exact {
                    anyhow::bail!(
                        "--ar-exact is an `--engine real` flag: the mock executor launches no \
                         collective, so there is no family to force and accepting it would make \
                         two mock runs look like an arithmetic A/B that never happened"
                    );
                }
                if keep_stale_anchor_blocks {
                    anyhow::bail!(
                        "--keep-stale-anchor-blocks is an `--engine real` flag: the mock executor \
                         has no drafter and proposes nothing, so there is no block to keep and \
                         accepting it would make a control arm that never ran look like one that \
                         did"
                    );
                }
                if step_keyed_draws {
                    anyhow::bail!(
                        "--step-keyed-draws is an `--engine real` flag: the mock executor draws \
                         nothing through the real executor's acceptance law, so a control arm \
                         on it would never have run"
                    );
                }
                if propose_in_step {
                    anyhow::bail!(
                        "--propose-in-step is an `--engine real` flag: the mock executor has no \
                         drafter of its own, so there is no proposal to put back inside the step"
                    );
                }
                serve::run(&serve::ServeArgs {
                    mock,
                    manifest,
                    prebuilt,
                    autotune,
                    tokenizer,
                    chat_template,
                    host,
                    port,
                    kv_gib,
                    state_gib,
                    step_delay_ms,
                    park_ram_gib,
                    park_nvme_gib,
                    park_nvme_path,
                    park_on_pressure,
                    max_logprobs,
                    parsers,
                })
            }
            other => anyhow::bail!("--engine {other}: the executors are `mock` and `real`"),
        },

        Cmd::Engine {
            pack,
            rank,
            prebuilt,
            tp,
            reference,
            min_margin,
            peer_device,
            peer_port,
            prompt_ids,
            prompt_index,
            tokens,
            gen,
            warmup,
            max_seqs,
            max_ctx,
            prefill_rows,
            decode_batches,
            decode_widths,
            width_rule,
            ragged_widths,
            row_bands,
            rectangular_widths,
            ragged_ceiling,
            mixed_prefill,
            max_batched_tokens,
            prefill_fill_above,
            batch_gen,
            gate_seqs,
            conc,
            conc_tokens,
            messages,
            tokenizer,
            chat_template,
            continuity_tokens,
            fork_n,
            fork_tokens,
            sample_dump,
            chunk_bisect,
            no_attribute,
            host_trace,
            whole_step_frame,
            frame_digest,
            race_check,
            poison_boot,
            constant_seed_control,
            gather_per_row,
            propose_in_step,
            fixed_step_wait,
            host_wait,
            gdn_state_dtype,
            bf16_head,
            mxfp4_head,
            fp8_head,
            no_capture,
            host_feed,
            no_live_rows,
            park_staging_mib,
            park_staging_slots,
            park_image_base,
            park_probe,
            kv_headroom_mib,
            ledger,
            kv_pool_blocks,
            accept_prompts,
            accept_tokens,
            accept_short,
            accept_control,
            accept_ids,
            draft_lattice_dump,
            stop_after_accept,
            solo_ids_out,
            solo_probe_out,
            solo_probe_steps,
            solo_probe_raw_step,
            solo_probe_raw_op,
            first_block_ids,
            draft_block,
            draft_pack,
            draft_replay,
            draft_chain_dump,
            draft_seq,
            draft_seq_out,
            ar_exact,
            draft_batch_check,
            draft_store_narrow,
            draft_store_natural_off,
            draft_store_check,
            drafter_store,
            draft_kv,
            draft_embed_share,
            selector_rank,
            draft_full_vocab,
            spec_tree,
            spec_tree_sampled,
            gdn_replay,
            host_levers,
            device_step,
            prefill_ar_overlap,
            ar_form,
            ar_push_blocks,
            small_chunk_gemm,
            small_chunk_band,
            shared_prefill_arena,
            no_shared_prefill_arena,
            shared_head_arena,
            no_shared_head_arena,
            mid_chunk_gemm,
            no_mid_chunk_gemm,
            mid_chunk_atiled,
            no_mid_chunk_atiled,
            skinny_gate,
            no_skinny_gate,
            decode_pf,
            no_decode_pf,
            head_m64,
            no_head_m64,
            drafter_gateup_m64,
            ar_wpar,
            ar_fuse,
            b1_arms,
            c4_arms,
            step_arms,
            mixed_class_proxy,
            decode_fusion,
            no_decode_fusion,
            fuse,
            budget_mib,
            device,
            deadline_s,
            hip_lib,
            rccl_lib,
            host_ar,
            fnx,
        } => {
            // FN12, the model selection law: the pack's manifest decides the path. A Flash-Next
            // pack (`"model": "qwen4_exp"`) boots the Flash-Next engine and returns; every other
            // pack takes the 27B's path below, unchanged (and refuses the Flash-Next flags).
            if flashnext::is_flashnext_pack(&pack)? {
                // FN57: the host-wait arm crosses into the Flash-Next flags by value
                let mut fnx = fnx;
                fnx.host_wait = flashnext::run::host_wait_fn(host_wait.as_deref())
                    .map_err(anyhow::Error::msg)?;
                // P27-7's `--ledger` is the 27B's law; a Flash-Next pack's fence is
                // `--fn-budget-mib` (FN18c's exact ledger), so the flag is parsed and unused here.
                ledger_law::Ledger::parse(&ledger)?;
                return flashnext::run::main(
                    &pack,
                    tp,
                    &prebuilt,
                    prompt_ids.as_deref(),
                    &hip_lib,
                    &fnx,
                );
            }
            fnx.refuse_on_27b()?;
            let prompt_ids = prompt_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "--prompt-ids is required (only a Flash-Next --fn-serve boot may omit it)"
                )
            })?;
            // P27-7: the ledger law, resolved once; rank 1 is told the numbers by value.
            let (budget_mib, kv_headroom_mib) = ledger_law::Ledger::parse(&ledger)?.resolve_said(
                budget_mib,
                kv_headroom_mib,
                ledger_law::box_p2pmem_mib(),
            );
            let failures = engine::gate(&engine::GateArgs {
                engine: engine::EngineArgs {
                    vision_max_patches: 0,
                    vision_lazy: false,
                    drafter_rope_index: false,
                    pack: pack.clone(),
                    world: tp,
                    rank,
                    peer_port,
                    follower: false,
                    peer_device,
                    prebuilt,
                    hip_lib,
                    rccl_lib: rccl_lib,
                    host_ar,
                    dspine_socket: None,
                    draft_capture_dir: None,
                    draft_capture_layers: Vec::new(),
                    device,
                    budget_mib,
                    max_seqs,
                    max_ctx,
                    prefill_rows,
                    decode_batches,
                    decode_widths,
                    decode_bands: decode_bands_of(ragged_widths, rectangular_widths, row_bands),
                    ragged_ceiling: saber_sched::RaggedCeiling::parse(&ragged_ceiling)?,
                    width_rule: saber_sched::WidthRule::parse(&width_rule)?,
                    mixed_prefill: mixed_prefill_of(mixed_prefill.as_deref(), max_batched_tokens)?,
                    prefill_fill_above,
                    deadline_s,
                    capture: !no_capture,
                    device_feed: !host_feed,
                    live_row_attn: !no_live_rows,
                    park_staging_mib,
                    // `saber engine` drives the park hook itself; nothing above it holds a tier —
                    // except the fork arm's park leg (ws-f38), which parks whole groups through
                    // the scheduler into a RAM tier the ledger has to be told about. The tier is
                    // cut into SLOTS of one staging slot each (the largest image this engine can
                    // make), so it must hold the widest group whole: ws-f39's hold found the
                    // flat 2 GiB tier was two 768 MiB slots, and every n = 4 group's third
                    // branch refused "full" with 1.9 GB of the pool free.
                    park_ram_gib: fork_park_ram_gib(&fork_n, park_staging_mib, park_staging_slots),
                    park_staging_slots,
                    park_image_base,
                    park_probe,
                    kv_headroom_mib,
                    kv_pool_blocks,
                    draft_block,
                    draft_pack,
                    draft_replay,
                    draft_chain_dump,
                    draft_seq,
                    draft_seq_out,
                    ar_exact,
                    draft_batch_check,
                    draft_store_narrow,
                    draft_store_natural_off,
                    draft_store_check,
                    draft_store_deferred: drafter_store == "deferred",
                    draft_kv_paged: draft_kv == "paged",
                    draft_embed_share: draft_embed_share == "on",
                    draft_full_vocab,
                    host_levers: saber_engine::host_levers::HostLevers::parse(&host_levers)
                        .unwrap_or_default(),
                    device_step: saber_engine::device_step::DeviceStep::parse(&device_step)
                        .unwrap_or_default(),
                    selector_rank: saber_engine::selector_rank::SelectorRank::parse(&selector_rank)
                        .unwrap_or_default(),
                    spec_tree: saber_engine::spec_tree::SpecTreeForm::parse(&spec_tree)
                        .unwrap_or_default(),
                    spec_tree_sampled: spec_tree_sampled != "off",
                    spec_tree_sampled_control: spec_tree_sampled == "control",
                    gdn_replay: gdn_replay_resolve(
                        "--gdn-replay",
                        &gdn_replay,
                        gdn_state_dtype == "f16",
                        &spec_tree,
                        &[],
                    ) == "on",
                    prefill_ar_overlap,
                    prefill_ar_push_blocks: ar_push_blocks_of(&ar_form, ar_push_blocks),
                    small_chunk_gemm,
                    small_chunk_band,
                    shared_prefill_arena: on_unless_off(
                        shared_prefill_arena,
                        no_shared_prefill_arena,
                    ),
                    shared_head_arena: on_unless_off(shared_head_arena, no_shared_head_arena),
                    mid_chunk_gemm: mid_chunk_gemm_law(mid_chunk_gemm, no_mid_chunk_gemm),
                    mid_chunk_atiled: on_unless_off(mid_chunk_atiled, no_mid_chunk_atiled),
                    skinny_gate: on_unless_off(skinny_gate, no_skinny_gate),
                    decode_pf: on_unless_off(decode_pf, no_decode_pf),
                    head_m64: on_unless_off(head_m64, no_head_m64),
                    drafter_gateup_m64: drafter_gateup_m64
                        .unwrap_or(saber_engine::device::DEFAULT_DRAFTER_GATEUP_M64),
                    ar_wpar: ar_wpar == "on",
                    ar_fuse: ar_fuse == "on",
                    b1_arms: saber_engine::device::B1Arms::parse(&b1_arms)
                        .map_err(anyhow::Error::msg)?,
                    c4_arms: saber_engine::device::C4Arms::parse(&c4_arms)
                        .map_err(anyhow::Error::msg)?,
                    step_arms: saber_engine::device::StepArms::parse(&step_arms)
                        .map_err(anyhow::Error::msg)?,
                    mixed_class_proxy,
                    decode_fusion: saber_engine::fuse27::Fuse27::parse(&fuse)
                        .and_then(|f| {
                            f.check_mixed(
                                mixed_prefill_of(mixed_prefill.as_deref(), max_batched_tokens)
                                    .is_ok_and(|m| m.enabled && m.fused),
                            )
                        })
                        .map_err(anyhow::Error::msg)?
                        .resolve(decode_fusion_of(
                            decode_fusion.as_deref(),
                            no_decode_fusion,
                            "--decode-fusion",
                            "--no-decode-fusion",
                        )?)
                        .map_err(anyhow::Error::msg)?,
                    host_trace,
                    whole_step_frame,
                    frame_digest,
                    race_check: engine::race_mode(race_check.as_deref())
                        .map_err(anyhow::Error::msg)?
                        .0,
                    race_shadow: engine::race_mode(race_check.as_deref())
                        .map_err(anyhow::Error::msg)?
                        .1,
                    poison_boot,
                    constant_seed_control,
                    gather_per_row,
                    propose_in_step,
                    fixed_step_wait,
                    fail_after_steps: 0,
                    host_wait: engine::host_wait_27b(host_wait.as_deref())
                        .map_err(anyhow::Error::msg)?,
                    gdn_state_f16: gdn_state_dtype == "f16",
                    head: target_head(
                        mxfp4_head,
                        &[(fp8_head, "--fp8-head"), (bf16_head, "--bf16-head")],
                        &pack,
                    )?,
                },
                prompt_ids,
                prompt_index,
                tokens,
                reference,
                min_margin,
                gen,
                batch_gen,
                warmup,
                gate_seqs,
                conc,
                conc_tokens,
                messages,
                tokenizer,
                chat_template,
                accept_prompts,
                accept_tokens,
                accept_short,
                accept_control,
                accept_ids,
                draft_lattice_dump,
                stop_after_accept,
                solo_ids_out,
                solo_probe_out,
                solo_probe_steps,
                solo_probe_raw_step,
                solo_probe_raw_op,
                first_block_ids,
                continuity_tokens,
                fork_n,
                fork_tokens,
                sample_dump,
                chunk_bisect,
                no_attribute,
            })?;
            // The engine's own thread owns every HIP handle and is joined when `gate` returns, so
            // the exit below happens with no live HIP thread (a `process::exit` with one is a
            // segfault, 09-18).
            if failures > 0 {
                std::process::exit(1);
            }
            Ok(())
        }

        Cmd::EngineWorker {
            pack,
            prebuilt,
            rank,
            device,
            peer_port,
            budget_mib,
            vision_max_patches,
            vision_lazy,
            max_seqs,
            max_ctx,
            prefill_rows,
            decode_batches,
            decode_widths,
            row_bands,
            rectangular_widths,
            park_staging_mib,
            kv_headroom_mib,
            park_staging_slots,
            park_image_base,
            park_probe,
            no_capture,
            draft_block,
            draft_pack,
            ar_exact,
            prefill_ar_overlap,
            ar_form,
            draft_kv,
            draft_embed_share,
            selector_rank,
            draft_full_vocab,
            spec_tree,
            spec_tree_sampled,
            gdn_replay,
            host_levers,
            device_step,
            ar_push_blocks,
            small_chunk_gemm,
            small_chunk_band,
            shared_prefill_arena,
            no_shared_prefill_arena,
            shared_head_arena,
            no_shared_head_arena,
            mid_chunk_gemm,
            no_mid_chunk_gemm,
            mid_chunk_atiled,
            no_mid_chunk_atiled,
            skinny_gate,
            no_skinny_gate,
            decode_pf,
            no_decode_pf,
            head_m64,
            no_head_m64,
            drafter_gateup_m64,
            ar_wpar,
            ar_fuse,
            b1_arms,
            c4_arms,
            step_arms,
            mixed_class_proxy,
            decode_fusion,
            host_trace,
            whole_step_frame,
            frame_digest,
            race_check,
            poison_boot,
            constant_seed_control,
            gather_per_row,
            fixed_step_wait,
            fail_after_steps,
            host_wait,
            gdn_state_dtype,
            head,
            deadline_s,
            hip_lib,

            rccl_lib,
            host_ar,
            draft_capture_layers,
            dspine_socket,
        } => engine::worker(&engine::WorkerArgs {
            engine: engine::EngineArgs {
                vision_max_patches,
                vision_lazy,
                // Leader-only (the drafter's positions are in the leader's frames).
                drafter_rope_index: false,
                pack,
                world: 2,
                rank,
                peer_port,
                follower: true,
                // Rank 1 never spawns anything, so this is its own device and not a peer's.
                peer_device: device,
                prebuilt,
                hip_lib,
                rccl_lib: rccl_lib,
                host_ar,
                dspine_socket,
                draft_capture_dir: None,
                draft_capture_layers,
                device,
                budget_mib,
                max_seqs,
                max_ctx,
                prefill_rows,
                decode_batches,
                decode_widths,
                // The row bands are PLANS, so rank 1 plans rank 0's (forwarded on this argv).
                decode_bands: if rectangular_widths {
                    Vec::new()
                } else {
                    row_bands
                },
                // Rank 1 has no scheduler, so no ceiling of its own.
                ragged_ceiling: saber_sched::RaggedCeiling::Table,
                // Rank 1 has no scheduler: the width rule picks a step's shape and rank 1 is
                // TOLD its shape, frame by frame. The default is carried so the struct is whole.
                width_rule: saber_sched::WidthRule::default(),
                mixed_prefill: saber_sched::MixedPrefill::OFF,
                prefill_fill_above: 0,
                deadline_s,
                capture: !no_capture,
                // There is no device feed above TP=1: the argmax is over half a vocabulary and the
                // token is decided on the host.
                device_feed: false,
                live_row_attn: true,
                // PARKING AT TWO RANKS (ws-t2): a session's KV and state are half on each card,
                // so an image is two halves and this rank makes its own — on the leader's command,
                // over the same regions, into its own staging buffer. The size is rank 0's,
                // forwarded on this argv by the spawn; 0 (the default, and what a worker started
                // by hand gets) means this rank cannot park and the leader's first park says so
                // by name instead of imaging half a session.
                park_staging_mib,
                park_ram_gib: 0.0,
                park_staging_slots,
                park_image_base,
                park_probe,
                kv_headroom_mib,
                kv_pool_blocks: 0,
                draft_block,
                draft_pack,
                draft_replay: None,
                draft_chain_dump: None,
                draft_seq: None,
                draft_seq_out: None,
                ar_exact,
                draft_batch_check: false,
                draft_store_narrow: false,
                draft_store_natural_off: false,
                draft_store_check: false,
                draft_store_deferred: false,
                draft_kv_paged: draft_kv == "paged",
                draft_embed_share: draft_embed_share == "on",
                draft_full_vocab,
                host_levers: saber_engine::host_levers::HostLevers::parse(&host_levers)
                    .unwrap_or_default(),
                device_step: saber_engine::device_step::DeviceStep::parse(&device_step)
                    .unwrap_or_default(),
                selector_rank: saber_engine::selector_rank::SelectorRank::parse(&selector_rank)
                    .unwrap_or_default(),
                spec_tree: saber_engine::spec_tree::SpecTreeForm::parse(&spec_tree)
                    .unwrap_or_default(),
                spec_tree_sampled: spec_tree_sampled != "off",
                spec_tree_sampled_control: false,
                gdn_replay: gdn_replay_resolve(
                    "--gdn-replay",
                    &gdn_replay,
                    gdn_state_dtype == "f16",
                    &spec_tree,
                    &[],
                ) == "on",
                prefill_ar_overlap,
                prefill_ar_push_blocks: ar_push_blocks_of(&ar_form, ar_push_blocks),
                small_chunk_gemm,
                small_chunk_band,
                shared_prefill_arena: on_unless_off(shared_prefill_arena, no_shared_prefill_arena),
                shared_head_arena: on_unless_off(shared_head_arena, no_shared_head_arena),
                mid_chunk_gemm: mid_chunk_gemm_law(mid_chunk_gemm, no_mid_chunk_gemm),
                mid_chunk_atiled: on_unless_off(mid_chunk_atiled, no_mid_chunk_atiled),
                skinny_gate: on_unless_off(skinny_gate, no_skinny_gate),
                decode_pf: on_unless_off(decode_pf, no_decode_pf),
                head_m64: on_unless_off(head_m64, no_head_m64),
                drafter_gateup_m64: drafter_gateup_m64
                    .unwrap_or(saber_engine::device::DEFAULT_DRAFTER_GATEUP_M64),
                ar_wpar: ar_wpar == "on",
                ar_fuse: ar_fuse == "on",
                b1_arms: saber_engine::device::B1Arms::parse(&b1_arms)
                    .map_err(anyhow::Error::msg)?,
                c4_arms: saber_engine::device::C4Arms::parse(&c4_arms)
                    .map_err(anyhow::Error::msg)?,
                step_arms: saber_engine::device::StepArms::parse(&step_arms)
                    .map_err(anyhow::Error::msg)?,
                mixed_class_proxy,
                decode_fusion,
                host_trace,
                whole_step_frame,
                frame_digest,
                race_check: engine::race_mode(race_check.as_deref())
                    .map_err(anyhow::Error::msg)?
                    .0,
                race_shadow: engine::race_mode(race_check.as_deref())
                    .map_err(anyhow::Error::msg)?
                    .1,
                poison_boot,
                constant_seed_control,
                gather_per_row,
                // Rank 1 is TOLD the order, step by step (`Command::Step`'s `defer_proposals`).
                propose_in_step: false,
                fixed_step_wait,
                fail_after_steps,
                host_wait: host_wait.parse().map_err(anyhow::Error::msg)?,
                gdn_state_f16: gdn_state_dtype == "f16",
                head,
            },
        }),

        Cmd::Tune {
            manifest,
            prebuilt,
            out,
            autotune_in,
            only,
            warmup,
            samples,
            budget_mib,
            device,
            power_cap_w,
            metrics_addr,
            list,
            dump_variants,
            no_ar_timing,
            ic_hot,
            no_prod_check,
            hip_lib,
        } => {
            let args = tune::TuneArgs {
                manifest,
                prebuilt,
                out,
                autotune_in,
                only,
                warmup,
                samples,
                budget_mib,
                device,
                power_cap_w,
                metrics_addr,
                list,
                dump_variants,
                no_ar_timing,
                ic_hot,
                no_prod_check,
                hip_lib: hip_lib.clone(),
            };
            if args.list {
                // No GPU: enumeration is a pure function of the rules and the manifests.
                return tune::list(&args);
            }
            let hip = open_hip(&hip_lib)?;
            let result = tune::run(&hip, &args);
            // Same rule as `gate`: the HIP thread goes down before this function returns, never
            // after `std::process::exit` has started running atexit handlers underneath it.
            drop(hip);
            result
        }
    }
}

/// `dlopen` the runtime, with the one error message that is worth writing out.
/// The head a `serve` / `engine` command line asks for (N5, N6). **The default is the exact head
/// AS THE PACK CARRIES IT** — the checkpoint's own precision (the user's law of 2026-09-22) —
/// resolved here against rank 0's pack file when it exists (`head_share::resolve_pack`: the fp8
/// `lm_head.w8` pack -> `Fp8`, the bf16 `lm_head.w` pack -> `Bf16`, the MXFP4 pack -> a refusal
/// naming `--mxfp4-head`), so the leader forwards rank 1 a resolved kind. `--mxfp4-head` is the
/// pairing control. The old head flags (`noop`: the fp8 and bf16 ones) are accepted, do nothing,
/// and print what the default resolved to; `--mxfp4-head` beside any of them is a refusal.
fn target_head(
    mxfp4_head: bool,
    noop: &[(bool, &str)],
    pack: &std::path::Path,
) -> anyhow::Result<saber_engine::head_share::TargetHead> {
    use saber_engine::head_share::TargetHead;
    let given: Vec<&str> = noop.iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect();
    if mxfp4_head && !given.is_empty() {
        anyhow::bail!(
            "`--mxfp4-head` contradicts {}: the default head is the checkpoint's own (the pack's \
             exact head), `--mxfp4-head` asks for the MXFP4 one",
            given.join(", ")
        );
    }
    if mxfp4_head {
        return Ok(TargetHead::Mxfp4);
    }
    let rank0 = if pack.is_dir() {
        pack.join("rank0.pack")
    } else {
        pack.to_path_buf()
    };
    let head = if rank0.is_file() {
        saber_engine::head_share::resolve_pack(TargetHead::Exact, pack, 0)
            .map_err(anyhow::Error::msg)?
    } else {
        // No pack on this host (a test's argv): the engine resolves it at boot.
        TargetHead::Exact
    };
    for flag in given {
        eprintln!(
            "{flag}: accepted and does nothing (N6): the head is the checkpoint's own as the pack \
             carries it, which resolved to the {} head for `{}`",
            head.as_arg(),
            pack.display()
        );
    }
    Ok(head)
}

/// N8 + ws-h39: the decode launch-train fusions are ON by default (the user's decision of
/// 2026-09-23) with the ws-h38/h39 served set `all` (rows 1-3 + 7; the user's default over
/// H39's paired 22.05 ms/step, bit-exact). `set` is `--decode-fusion [h37|all]` (`default` is the bare flag:
/// N8's accepted no-op, which prints what the default resolved to); `off`
/// (`--no-decode-fusion` / `--engine-no-decode-fusion`) is the pairing control; a set named
/// together with the control is a contradiction and refuses by name.
fn decode_fusion_of(
    set: Option<&str>,
    off: bool,
    name: &str,
    off_name: &str,
) -> anyhow::Result<saber_engine::device::DecodeFusion> {
    use saber_engine::device::DecodeFusion;
    if off {
        if set.is_some() {
            anyhow::bail!(
                "`{off_name}` contradicts `{name}`: the decode fusions are ON by default (N8), \
                 `{off_name}` asks for the unfused decode step"
            );
        }
        return Ok(DecodeFusion::Off);
    }
    if set == Some("default") {
        eprintln!(
            "{name}: a bare flag is accepted and does nothing (N8): the decode launch-train \
             fusions are ON by default with the ws-h38/h39 served set (`all`, rows 1-3 + 7, the \
             user's default since 2026-09-23); `{name} h37` is the rows 1-2 set, `{off_name}` \
             the control"
        );
    }
    DecodeFusion::from_flag(set.filter(|s| *s != "default")).map_err(anyhow::Error::msg)
}

/// `--engine-fuse` / `--fuse` (P27-16): parsed once at the argv, kept canonical.
fn parse_fuse27_arg(s: &str) -> Result<String, String> {
    saber_engine::fuse27::Fuse27::parse(s).map(saber_engine::fuse27::Fuse27::as_arg)
}

/// `saber engine-worker --decode-fusion <off|h37|all>` (N8, ws-h39): the leader forwards the
/// resolved set.
fn parse_fusion_set(s: &str) -> Result<saber_engine::device::DecodeFusion, String> {
    saber_engine::device::DecodeFusion::from_worker_arg(s)
}

/// `saber engine-worker --head <exact|bf16|fp8|mxfp4>`.
fn parse_target_head(s: &str) -> Result<saber_engine::head_share::TargetHead, String> {
    saber_engine::head_share::TargetHead::from_arg(s)
}

fn open_hip(soname: &str) -> anyhow::Result<HipThread> {
    match HipThread::spawn(soname) {
        Ok(hip) => Ok(hip),
        Err(e) if e.is_not_available() => anyhow::bail!(
            "{e}\n\
             saber binds to HIP by dlopen, never by linking, so this is a run-time refusal and not \
             a build problem. There is no ROCm on this host; run inside the dev image and pass the \
             full path:\n    \
             --hip-lib /opt/rocm/lib/libamdhip64.so.7"
        ),
        Err(e) => anyhow::bail!(e),
    }
}

fn probe(soname: &str, with_free_mem: bool) -> anyhow::Result<()> {
    let hip = open_hip(soname)?;

    let topo = Topology::probe(&hip, with_free_mem)?;

    println!("runtime");
    println!("  soname            {}", topo.soname);
    println!("  driver version    {}", topo.driver);
    println!("  runtime version   {}", topo.runtime);
    println!("  devices           {}", topo.devices.len());

    for d in &topo.devices {
        println!();
        println!("device {}", d.index);
        println!("  name              {}", d.name);
        println!(
            "  gfx target        {}   [{}; HIP reports compute capability {}.{}, ASIC revision {}]",
            d.arch, d.arch_source, d.cc_major, d.cc_minor, d.asic_revision
        );
        match d.free_mem {
            Some(free) => println!(
                "  VRAM              {} total, {} free ({} MiB)",
                gib(d.total_mem),
                gib(free),
                free / (1024 * 1024)
            ),
            None => println!(
                "  VRAM              {} total, free unknown{}",
                gib(d.total_mem),
                if with_free_mem {
                    " (hipMemGetInfo failed; see the log)"
                } else {
                    " (--no-free-mem)"
                }
            ),
        }
        println!(
            "  compute units     {} reported, {} physical",
            d.cu_count, d.physical_cu_count
        );
        println!(
            "  wavefront         {}   (device default only; each kernel declares its own)",
            d.warp_size
        );
        println!(
            "  LDS               {} B per workgroup, {} B per CU",
            d.lds_per_block, d.lds_per_cu
        );
        println!(
            "  max threads/block {}, registers/block {}",
            d.max_threads_per_block, d.max_registers_per_block
        );
        println!("  core clock        {} MHz peak", d.clock_khz / 1000);
        println!(
            "  memory            {} MHz peak, {}-bit bus, {} MiB L2",
            d.memory_clock_khz / 1000,
            d.memory_bus_bits,
            d.l2_bytes / (1024 * 1024)
        );
        println!("  PCI               {}", d.pci_bus_id);
        println!(
            "  fine-grained      {}     large BAR: {}",
            yn(d.fine_grain),
            yn(d.large_bar)
        );
    }

    println!();
    println!("P2P access (hipDeviceCanAccessPeer; rows = from, columns = to)");
    print!("        ");
    for j in 0..topo.devices.len() {
        print!("{:>6}", format!("d{j}"));
    }
    println!();
    for (i, row) in topo.peer_access.iter().enumerate() {
        print!("  {:>5} ", format!("d{i}"));
        for (j, &can) in row.iter().enumerate() {
            let cell = if i == j { "self".to_string() } else { yn(can) };
            print!("{cell:>6}");
        }
        println!();
    }

    Ok(())
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn yn(b: bool) -> String {
    if b { "yes" } else { "no" }.to_string()
}

/// The on-pressure park policy out of the serve flags (ws-k18): the victim order, the per-session
/// cap (0 = unbounded) and the restore bound, on top of `ParkPolicy::on_pressure()`, or the default
/// (nothing evicts) without `--park-on-pressure`.
fn park_policy_of(
    on_pressure: bool,
    victim: &str,
    max_per_session: u32,
    max_restore_attempts: u32,
) -> anyhow::Result<saber_sched::ParkPolicy> {
    let base = if on_pressure {
        saber_sched::ParkPolicy::on_pressure()
    } else {
        saber_sched::ParkPolicy::default()
    };
    let victim = saber_sched::VictimOrder::parse(victim).map_err(anyhow::Error::msg)?;
    if max_restore_attempts == 0 {
        anyhow::bail!(
            "--park-max-restore-attempts 0: a parked session is always given its first restore"
        );
    }
    Ok(base
        .victim(victim)
        .max_parks_per_session(if max_per_session == 0 {
            u32::MAX
        } else {
            max_per_session
        })
        .max_restore_attempts(max_restore_attempts))
}

#[cfg(test)]
mod cmp27g_tests {
    use super::gdn_replay_resolve;

    /// CMP27G: `auto` (the default) runs the replay arm on the served line and stands down BY
    /// NAME where the arm cannot run; an explicit `on` / `off` passes through (`on` is refused
    /// downstream where it cannot run, `off` is the control).
    #[test]
    fn cmp27g_the_replay_default_is_on_where_it_can_run() {
        let r = |a, f16, t, b: &[(bool, &str)]| gdn_replay_resolve("--x", a, f16, t, b);
        assert_eq!(r("auto", true, "ancestor9", &[]), "on");
        assert_eq!(r("auto", true, "off", &[]), "on");
        assert_eq!(r("auto", false, "ancestor9", &[]), "off");
        assert_eq!(r("auto", true, "best16", &[]), "off");
        assert_eq!(
            r("auto", true, "ancestor9", &[(true, "a park tier")]),
            "off"
        );
        assert_eq!(
            r("auto", true, "ancestor9", &[(false, "a park tier")]),
            "on"
        );
        assert_eq!(r("off", true, "ancestor9", &[]), "off");
        assert_eq!(r("on", false, "best16", &[(true, "x")]), "on");
    }
}

#[cfg(test)]
mod exit_tests {
    /// **No `std::process::exit` in this file may run with a live `HipThread`**, and this reads
    /// the file to say so.
    ///
    /// `process::exit` skips destructors and goes straight to the C runtime's atexit handlers.
    /// The HIP runtime registers one, so it tears itself down while our worker thread is still
    /// inside the library — servicing the `hipFree` / `hipHostFree` jobs that every dropped
    /// handle posts. `docs/EXEC.md` sec.9 states the rule ("`process::exit` with a live HIP thread
    /// segfaults — drop the context") and the gate's negative control measured it as exit 139.
    ///
    /// The one this test was written for is quieter and took two workstreams to place:
    /// `saber forward --prompts 200 --compare` disagrees on some prompts, returns a non-zero
    /// count, and exits — with thousands of destructor jobs still queued behind 200 passes. The
    /// symptom is `double free or corruption (out)` printed after the agreement table (H part 13
    /// recorded it as an unattributed defect; H part 19 read it out of this file). At one prompt
    /// the queue is empty and nothing happens, which is why it never showed up in a smoke run.
    ///
    /// **The rule, as the scan applies it**: for each `std::process::exit`, find the nearest
    /// preceding `open_hip(`. If that binding was made in the SAME `Cmd::` arm and no `drop(hip`
    /// sits between the two, the thread is still alive at the exit. An arm with no `open_hip` of
    /// its own is not in scope — `Cmd::Engine` exits after joining a thread it owns internally,
    /// and says so at the call site.
    #[test]
    fn no_process_exit_runs_with_a_live_hip_thread() {
        let src = include_str!("main.rs");
        // Only the dispatcher: the test module below contains the very shape being looked for,
        // as a string, on purpose.
        let src = src.split("#[cfg(test)]").next().unwrap_or(src);
        let lines: Vec<&str> = src.lines().collect();
        let code = |i: usize| !lines[i].trim_start().starts_with("//");
        let arm_of = |i: usize| {
            (0..=i)
                .rev()
                .find(|j| lines[*j].starts_with("        Cmd::"))
        };
        let mut bad: Vec<String> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("std::process::exit") || !code(i) {
                continue;
            }
            let Some(o) = (0..i)
                .rev()
                .find(|j| lines[*j].contains("open_hip(") && code(*j))
            else {
                continue;
            };
            if arm_of(o) != arm_of(i) {
                continue;
            }
            if (o + 1..i).any(|k| lines[k].contains("drop(hip") && code(k)) {
                continue;
            }
            bad.push(format!(
                "line {}: `std::process::exit` with the `hip` opened at line {} still alive",
                i + 1,
                o + 1
            ));
        }
        assert!(
            bad.is_empty(),
            "a `process::exit` with a live HIP thread is a teardown race, not a clean exit \
             (docs/EXEC.md sec.9):\n  {}",
            bad.join("\n  ")
        );
    }

    /// The scan has teeth: the shape it is looking for is detected when it is present.
    ///
    /// Rule 8 — a bounded-numerics test ships with a negative control that is known to fail — and
    /// the same argument applies to a lint. This runs the scan's own predicate over a snippet that
    /// has the defect, so a future edit that softens the scan into something that always passes
    /// fails here instead of going unnoticed.
    #[test]
    fn the_scan_finds_a_live_hip_thread_when_there_is_one() {
        let sample = "        Cmd::Forward {\n            let hip = open_hip(&hip_lib)?;\n                                  if bad > 0 {\n                std::process::exit(1);\n";
        let lines: Vec<&str> = sample.lines().collect();
        let code = |i: usize| !lines[i].trim_start().starts_with("//");
        let arm_of = |i: usize| {
            (0..=i)
                .rev()
                .find(|j| lines[*j].starts_with("        Cmd::"))
        };
        let mut found = false;
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("std::process::exit") || !code(i) {
                continue;
            }
            let Some(o) = (0..i)
                .rev()
                .find(|j| lines[*j].contains("open_hip(") && code(*j))
            else {
                continue;
            };
            if arm_of(o) == arm_of(i) && !(o + 1..i).any(|k| lines[k].contains("drop(hip")) {
                found = true;
            }
        }
        assert!(
            found,
            "the scan no longer detects the pattern it exists for"
        );
    }
}

#[cfg(test)]
mod prod_argv_tests {
    //! Prod's launcher argv through `saber serve`'s clap (V9): every the reference server flag is either ACCEPTED
    //! (it means the same thing here) or REFUSED BY NAME (clap's "unexpected argument '--x'"), and
    //! the classification is asserted, not printed.
    use super::{one_served_name, serve_parsers, Cli, Cmd};
    use clap::error::{ContextKind, ContextValue, ErrorKind};
    use clap::Parser;

    /// `serve-mxfp4.sh:1062-1074` as the `qwen_prod_nvfp4` unit runs it (TP=2, `SPEC_METHOD=dflash`,
    /// `SERVED_NAMES="Qwen3.8 Qwen3.6 Qwen3.8-NVFP4 Qwen3.8-MXFP4"`, `KV_MEM=0`, async on), with
    /// the in-container paths.
    fn prod_argv() -> Vec<String> {
        [
            "saber", "serve",
            "/models/Qwen3.8-27B-NVFP4",
            "--served-model-name", "Qwen3.8", "Qwen3.6", "Qwen3.8-NVFP4", "Qwen3.8-MXFP4",
            "--host", "0.0.0.0", "--port", "8080",
            "--kv-cache-dtype", "fp8", "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "262144", "--max-num-seqs", "8", "--max-num-batched-tokens", "8192",
            "--attention-backend", "TRITON_ATTN",
            "--speculative-config",
            "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-FP8\",\"num_speculative_tokens\":7}",
            "--async-scheduling",
            "--enable-prefix-caching", "--mamba-cache-mode", "align",
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
            "--override-generation-config", "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":20}",
            "--chat-template", "/templates/qwen-fixed-v22.3.jinja",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
    }

    /// Parse, and on each refusal record the argument clap NAMED and drop it (with its values),
    /// until the rest parses.
    fn classify(mut argv: Vec<String>) -> (Vec<String>, Cli) {
        let mut refused = Vec::new();
        loop {
            match Cli::try_parse_from(&argv) {
                Ok(cli) => return (refused, cli),
                Err(e) => {
                    assert_eq!(e.kind(), ErrorKind::UnknownArgument, "{e}");
                    let Some(ContextValue::String(arg)) = e.get(ContextKind::InvalidArg) else {
                        panic!("clap did not name the argument: {e}");
                    };
                    let text = e.to_string();
                    assert!(
                        text.contains(&format!("unexpected argument '{arg}'")),
                        "{text}"
                    );
                    let at = argv
                        .iter()
                        .position(|a| a == arg)
                        .expect("named arg is in argv");
                    let mut end = at + 1;
                    if arg.starts_with("--") {
                        while end < argv.len() && !argv[end].starts_with("--") {
                            end += 1;
                        }
                    }
                    argv.drain(at..end);
                    refused.push(arg.clone());
                }
            }
        }
    }

    /// `Cli`'s derived command is large enough to overflow a debug test thread's 2 MiB stack.
    fn on_a_big_stack(f: fn()) {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("the test body panicked");
    }

    #[test]
    fn prods_argv_is_accepted_or_refused_by_name_flag_by_flag() {
        on_a_big_stack(prods_argv_body);
    }

    #[test]
    fn the_parser_flags_parse_as_the_reference_spells_them() {
        on_a_big_stack(parser_flags_body);
    }

    /// ws-q3: the acceptance-law CONTROL reaches `saber serve` off by default and on by name.
    #[test]
    fn the_step_keyed_control_is_a_serve_flag_off_by_default() {
        on_a_big_stack(|| {
            let read = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    step_keyed_draws, ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                step_keyed_draws
            };
            assert!(!read(&[]));
            assert!(read(&["--step-keyed-draws"]));
        });
    }

    /// FN52: prefix caching reaches `saber serve` OFF by default (the 27B's served bytes never
    /// move without the user's word) and on by name.
    #[test]
    fn prefix_caching_is_a_serve_flag_off_by_default() {
        on_a_big_stack(|| {
            let read = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    enable_prefix_caching,
                    prefix_cache_verify,
                    ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                (enable_prefix_caching, prefix_cache_verify)
            };
            assert_eq!(read(&[]), ("off".to_owned(), false));
            assert_eq!(read(&["--enable-prefix-caching", "on"]).0, "on");
            assert!(read(&["--prefix-cache-verify"]).1);
            // FN66: `--warm-prompts FILE` parses on `saber serve` (it turns the cache on)
            let Cmd::Serve { warm_prompts, .. } =
                Cli::try_parse_from(["saber", "serve", "--warm-prompts", "/w.json"])
                    .expect("parses")
                    .cmd
            else {
                panic!("not serve");
            };
            assert_eq!(warm_prompts, Some(std::path::PathBuf::from("/w.json")));
        });
    }

    /// CMP27O: the drafter gate_up m64 (17) and the b1 arms (`conv,adv,head`) are the serve
    /// defaults, `0` / `off` are the controls, and every path names the one constant.
    #[test]
    fn cmp27o_the_stacked_arms_are_the_default_and_0_off_are_the_controls() {
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    engine_drafter_gateup_m64,
                    engine_b1_arms,
                    ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                (
                    engine_drafter_gateup_m64
                        .unwrap_or(saber_engine::device::DEFAULT_DRAFTER_GATEUP_M64),
                    saber_engine::device::B1Arms::parse(&engine_b1_arms).expect("arms"),
                )
            };
            let (g, a) = serve(&[]);
            assert_eq!(g, 17);
            assert!(a.conv_rows && a.adv_batch && a.head_nt);
            let (g, a) = serve(&[
                "--engine-drafter-gateup-m64",
                "0",
                "--engine-b1-arms",
                "off",
            ]);
            assert_eq!(g, 0, "the gate_up control");
            assert_eq!(a, saber_engine::device::B1Arms::default(), "the b1 control");
            assert_eq!(
                serve(&["--engine-drafter-gateup-m64"]).0,
                33,
                "bare = the b8 cells"
            );
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert_eq!(
                src.matches("default_value = saber_engine::device::DEFAULT_B1_ARMS")
                    .count(),
                3,
                "serve, engine and engine-worker all default to the one constant"
            );
            assert_eq!(
                src.matches(".unwrap_or(saber_engine::device::DEFAULT_DRAFTER_GATEUP_M64)")
                    .count(),
                3,
                "serve, engine and engine-worker all default to the one constant"
            );
        });
    }

    /// CMP27AE: the GEMM + wht6 fold is the default on every path (serve, the engine harness, rank
    /// 1's worker), `off` is the control, a bare flag is `on`, and anything else is refused.
    #[test]
    fn cmp27ae_the_gemm_wht_fold_is_the_default_and_off_is_the_control() {
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve { engine_ar_fuse, .. } =
                    Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                engine_ar_fuse
            };
            assert_eq!(saber_engine::device::DEFAULT_AR_FUSE, "on");
            assert_eq!(serve(&[]), "on", "the default");
            assert_eq!(serve(&["--engine-ar-fuse", "off"]), "off", "the control");
            assert_eq!(serve(&["--engine-ar-fuse"]), "on", "bare = on");
            assert!(Cli::try_parse_from(["saber", "serve", "--engine-ar-fuse", "maybe"]).is_err());
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert_eq!(
                src.matches("default_value = saber_engine::device::DEFAULT_AR_FUSE")
                    .count(),
                3,
                "serve, engine and engine-worker all default to the one constant"
            );
        });
    }

    /// CMP27S: the wave-parallel wht6 body is the default on every path (serve, the engine harness,
    /// rank 1's worker), `off` is the control, a bare flag is `on`, and anything else is refused.
    #[test]
    fn cmp27s_the_wpar_body_is_the_default_and_off_is_the_control() {
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve { engine_ar_wpar, .. } =
                    Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                engine_ar_wpar
            };
            assert_eq!(saber_engine::device::DEFAULT_AR_WPAR, "on");
            assert_eq!(serve(&[]), "on", "the default");
            assert_eq!(serve(&["--engine-ar-wpar", "off"]), "off", "the control");
            assert_eq!(serve(&["--engine-ar-wpar"]), "on", "bare = on");
            assert!(Cli::try_parse_from(["saber", "serve", "--engine-ar-wpar", "maybe"]).is_err());
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert_eq!(
                src.matches("default_value = saber_engine::device::DEFAULT_AR_WPAR")
                    .count(),
                3,
                "serve, engine and engine-worker all default to the one constant"
            );
        });
    }

    /// P27-14: the host levers default to `table,socket` (the flip) on every path (serve, the engine harness, rank
    /// 1's worker), each lever is selectable, and an unknown lever is refused at the parser.
    #[test]
    fn p27_14_the_host_levers_default_to_none_and_parse_each_lever() {
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    engine_host_levers, ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                engine_host_levers
            };
            assert_eq!(serve(&[]), saber_engine::host_levers::DEFAULT_HOST_LEVERS);
            assert_eq!(
                serve(&["--engine-host-levers", "table,socket"]),
                "table,socket"
            );
            assert_eq!(
                serve(&["--engine-host-levers", "picks,table,socket"]),
                "picks,table,socket"
            );
            assert!(
                Cli::try_parse_from(["saber", "serve", "--engine-host-levers", "picks,tree"])
                    .is_err(),
                "an unknown lever is refused"
            );
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert_eq!(
                src.matches("default_value = saber_engine::host_levers::DEFAULT_HOST_LEVERS")
                    .count(),
                3,
                "serve, engine and engine-worker all default to the one constant"
            );
        });
    }

    /// P27-12: the selector's rank form is `walk` (the device's greedy walk, prod's) by default on
    /// every path, `explen` is the arm, and an unknown form is refused.
    #[test]
    fn p27_12_the_selector_rank_defaults_to_the_walk_on_every_path() {
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    engine_selector_rank,
                    ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                engine_selector_rank
            };
            assert_eq!(serve(&[]), "walk");
            assert_eq!(serve(&["--engine-selector-rank", "explen"]), "explen");
            assert_eq!(serve(&["--engine-selector-rank", "tandem-b16-calibrated"]), "tandem-b16-calibrated");
            assert!(
                Cli::try_parse_from(["saber", "serve", "--engine-selector-rank", "viterbi"])
                    .is_err(),
                "an unknown form is refused"
            );
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert_eq!(
                src.matches(r#"default_value = "walk", value_parser = ["walk", "explen", "tandem-b16-calibrated"]"#)
                    .count(),
                3,
                "serve, engine and engine-worker all default to the walk"
            );
            assert_eq!(
                saber_engine::selector_rank::SelectorRank::parse("walk"),
                Ok(saber_engine::selector_rank::SelectorRank::default())
            );
        });
    }

    /// P27-15c: the tree is `ancestor9` by default on every path since P27-W2-4 (the user's flip
    /// 10-22), `off` (the chain of 7) is the control, an unknown form is refused, the worker is told the leader's form by value, and in
    /// THIS binary an `ancestor9` boot is refused by name and runs `off` (the tree verify form is
    /// not built) while its plans are built from the lattice.
    #[test]
    fn p27_15c_the_spec_tree_is_ancestor9_on_every_path_and_off_is_the_control() {
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    engine_spec_tree, ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                engine_spec_tree
            };
            assert_eq!(serve(&[]), "ancestor9");
            assert_eq!(serve(&["--engine-spec-tree", "off"]), "off");
            assert!(
                Cli::try_parse_from(["saber", "serve", "--engine-spec-tree", "chains"]).is_err(),
                "an unknown form is refused"
            );
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert_eq!(
                src.matches(
                    r#"default_value = "ancestor9", value_parser = ["off", "ancestor9", "best16"]"#
                )
                .count(),
                1,
                "serve defaults to the tree"
            );
            // P27-15f: engine and engine-worker also take the two-record control
            assert_eq!(
                src.matches(
                    r#"default_value = "ancestor9", value_parser = ["off", "ancestor9", "ancestor9-pair", "best16"]"#
                )
                .count(),
                2,
                "engine and engine-worker default to the tree"
            );
            assert_eq!(
                saber_engine::spec_tree::SpecTreeForm::parse("off"),
                Ok(saber_engine::spec_tree::SpecTreeForm::default())
            );
            let eng = include_str!("engine.rs");
            assert!(
                eng.contains(r#"cmd.arg("--spec-tree").arg(args.spec_tree.as_str());"#),
                "rank 1 is told the leader's form by value"
            );
            let arm = saber_engine::spec_tree_step::resolve_arm(
                saber_engine::spec_tree::SpecTreeForm::Ancestor9,
                saber_engine::spec_tree_step::ArmContext {
                    tp: 2,
                    drafter: true,
                    draft_block: 8,
                    selector_reranks: false,
                    form_built: saber_engine::spec_tree_step::TREE_VERIFY_FORM_BUILT,
                    max_seqs: 8,
                },
            );
            // P27-W2-4 (R2): `ancestor9` flushes every tree step, so a served boot runs it; the
            // pair control has no flush and still falls back at max seqs 8.
            assert_eq!(arm.runs, saber_engine::spec_tree::SpecTreeForm::Ancestor9);
            let pair = saber_engine::spec_tree_step::resolve_arm(
                saber_engine::spec_tree::SpecTreeForm::Ancestor9Pair,
                saber_engine::spec_tree_step::ArmContext {
                    tp: 2,
                    drafter: true,
                    draft_block: 8,
                    selector_reranks: false,
                    form_built: saber_engine::spec_tree_step::TREE_VERIFY_FORM_BUILT,
                    max_seqs: 8,
                },
            );
            assert_eq!(pair.runs, saber_engine::spec_tree::SpecTreeForm::Off);
            assert!(pair.fell_back() && pair.plans);
        });
    }

    /// P27-9d: the drafter's embedding share is ON by default on every path that reads it (serve,
    /// the engine harness, rank 1's worker -- which is told the leader's value by name), `off` is
    /// the control, and the three paths read the one constant.
    #[test]
    fn the_embedding_share_is_the_default_on_every_path_and_off_is_the_control() {
        assert_eq!(super::DEFAULT_DRAFT_EMBED_SHARE, "on");
        on_a_big_stack(|| {
            let serve = |extra: &[&str]| {
                let mut argv = vec!["saber", "serve"];
                argv.extend_from_slice(extra);
                let Cmd::Serve {
                    engine_draft_embed_share,
                    ..
                } = Cli::try_parse_from(argv).expect("parses").cmd
                else {
                    panic!("not serve");
                };
                engine_draft_embed_share
            };
            assert_eq!(serve(&[]), "on");
            assert_eq!(serve(&["--engine-draft-embed-share", "off"]), "off");
            assert!(
                Cli::try_parse_from(["saber", "serve", "--engine-draft-embed-share", "both"])
                    .is_err(),
                "an unknown value is refused"
            );
            let src = include_str!("main.rs");
            let src = src.split("#[cfg(test)]").next().unwrap_or(src);
            let n = src
                .matches("default_value = DEFAULT_DRAFT_EMBED_SHARE")
                .count();
            assert_eq!(
                n, 3,
                "serve, engine and engine-worker all default to the one constant"
            );
        });
    }

    fn prods_argv_body() {
        let (refused, cli) = classify(prod_argv());
        assert_eq!(
            refused,
            [
                "/models/Qwen3.8-27B-NVFP4",
                "--kv-cache-dtype",
                "--tensor-parallel-size",
                "--gpu-memory-utilization",
                "--max-model-len",
                "--max-num-seqs",
                "--max-num-batched-tokens",
                "--attention-backend",
                "--speculative-config",
                "--async-scheduling",
                // FN52: `--enable-prefix-caching` is OURS now (prod's bare spelling = on)
                "--mamba-cache-mode",
                "--override-generation-config",
            ]
        );
        let Cmd::Serve {
            served_model_name,
            host,
            port,
            enable_auto_tool_choice,
            no_enable_auto_tool_choice,
            tool_call_parser,
            reasoning_parser,
            parser_defaults,
            chat_template,
            ..
        } = cli.cmd
        else {
            panic!("not a serve command");
        };
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 8080);
        assert_eq!(
            chat_template.to_str(),
            Some("/templates/qwen-fixed-v22.3.jinja")
        );
        // The parser trio reads as prod's profile.
        let parsers = serve_parsers(
            enable_auto_tool_choice,
            no_enable_auto_tool_choice,
            tool_call_parser,
            reasoning_parser,
            &parser_defaults,
        )
        .expect("prod's parser flags resolve");
        assert_eq!(parsers, saber_serve::state::ResolvedParsers::prod());
        // `--served-model-name` PARSES with prod's four names and is then refused by name: Saber
        // serves one.
        assert_eq!(served_model_name.len(), 4);
        let e = one_served_name(served_model_name).unwrap_err().to_string();
        assert!(e.starts_with("--served-model-name Qwen3.8 Qwen3.6"), "{e}");
    }

    fn parser_flags_body() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["saber", "serve"];
            argv.extend_from_slice(extra);
            let Cmd::Serve {
                enable_auto_tool_choice,
                no_enable_auto_tool_choice,
                tool_call_parser,
                reasoning_parser,
                parser_defaults,
                ..
            } = Cli::try_parse_from(argv).expect("parses").cmd
            else {
                panic!("not serve");
            };
            serve_parsers(
                enable_auto_tool_choice,
                no_enable_auto_tool_choice,
                tool_call_parser,
                reasoning_parser,
                &parser_defaults,
            )
        };
        let off = parse(&["--no-enable-auto-tool-choice", "--reasoning-parser", ""]).expect("ok");
        assert!(!off.enable_auto_tool_choice && off.tool_parser.is_none());
        assert!(off.reasoning_parser.is_none());
        // argparse's BooleanOptionalAction: the last of the pair wins.
        let on = parse(&["--no-enable-auto-tool-choice", "--enable-auto-tool-choice"]).expect("ok");
        assert!(on.enable_auto_tool_choice && on.tool_parser.is_some());
        let e = parse(&["--parser-defaults", "compat", "--enable-auto-tool-choice"]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "Error: --enable-auto-tool-choice requires --tool-call-parser"
        );
        let e = parse(&["--tool-call-parser", "nope"])
            .unwrap_err()
            .to_string();
        assert!(
            e.starts_with("invalid tool call parser: nope (chose from { apertus,"),
            "{e}"
        );
        assert!(parse(&["--parser-defaults", "neither"]).is_err());
    }
}

/// The fork arm's RAM tier, in GiB: whole staging slots (one image each at `max_ctx`) for the
/// widest group plus one, and never below the 2 GiB F38 declared. Zero without the arm or a
/// staging buffer.
fn fork_park_ram_gib(fork_n: &[usize], staging_mib: u64, staging_slots: u32) -> f64 {
    if fork_n.is_empty() || staging_mib == 0 {
        return 0.0;
    }
    let widest = fork_n.iter().copied().max().unwrap_or(1) as f64;
    let slot_mib = staging_mib as f64 / f64::from(staging_slots.max(1));
    (((widest + 1.0) * slot_mib) / 1024.0).max(2.0)
}

#[cfg(test)]
mod fork_tier_tests {
    use super::fork_park_ram_gib;

    /// The widest group fits whole: at the gate's 768 MiB / 1 slot and n = 4 the tier is five
    /// slots, where the flat 2 GiB was two (the control: 2 GiB / 768 MiB = 2 < 4).
    #[test]
    fn the_fork_tier_holds_the_widest_group_whole() {
        let gib = fork_park_ram_gib(&[2, 4], 768, 1);
        let slots = (gib * 1024.0 / 768.0).floor() as usize;
        assert!(slots >= 4, "{gib} GiB is {slots} slots of 768 MiB");
        assert_eq!(
            (2.0f64 * 1024.0 / 768.0).floor() as usize,
            2,
            "the old tier's slot count"
        );
        assert_eq!(fork_park_ram_gib(&[], 768, 1), 0.0);
        assert_eq!(fork_park_ram_gib(&[4], 0, 1), 0.0);
        assert!(fork_park_ram_gib(&[2], 64, 2) >= 2.0);
    }
}

/// **The decode row bands a serve / engine plane plans (C2, C3)**: none — `batch x width`
/// rectangles, THE DEFAULT since C3 — unless `--ragged-widths` (`--engine-ragged-widths`) asks for
/// the ragged schedule, and never beside an explicit `--rectangular-widths` (clap refuses the
/// pair; this is the belt to that brace). Rank 1 does not come through here: rank 0 forwards its
/// resolved bands, or `--rectangular-widths` when it planned none.
pub(crate) fn decode_bands_of(ragged: bool, rectangular: bool, bands: Vec<u32>) -> Vec<u32> {
    if ragged && !rectangular {
        bands
    } else {
        Vec::new()
    }
}

/// D6/the user's default (2026-09-23): the drafter's rope for an image sequence follows the token
/// INDEX (prod's V2 runner) unless `--drafter-rope-temporal` asks for the pre-D6 temporal axis.
/// `--drafter-rope-index` is accepted as a no-op that names the default; clap refuses both together.
/// G5's mid-M folded tiles are the DEFAULT (2026-09-24): the `on` flag is accepted for scripts that
/// already pass it, and only the explicit `off` flag turns the `[[folded_mt]]` bands back to the
/// reference 256-row tile (the control).
/// G6's two levers are the DEFAULT (2026-09-24): the layout law's mid band (tiled producers + the
/// A-tiled family at 129..512 rows) and the skinny 48-wide GDN gate body. Same shape as the other
/// default laws: the `on` flag is a no-op, only the explicit `off` flag is the control.
pub fn on_unless_off(on_flag: bool, off_flag: bool) -> bool {
    let _ = on_flag;
    !off_flag
}

pub fn mid_chunk_gemm_law(on_flag: bool, off_flag: bool) -> bool {
    let _ = on_flag;
    !off_flag
}

pub fn drafter_rope_law(index_flag: bool, temporal_flag: bool) -> bool {
    let _ = index_flag;
    !temporal_flag
}

#[cfg(test)]
mod drafter_rope_law_tests {
    use super::{drafter_rope_law, mid_chunk_gemm_law};

    #[test]
    fn the_packed_mixed_step_is_the_default_and_off_is_the_control() {
        use super::mixed_prefill_of;
        let d = mixed_prefill_of(None, 8192).expect("the default parses");
        let p = mixed_prefill_of(Some("packed"), 8192).expect("packed parses");
        assert_eq!(d, p, "no flag: the packed mixed step (the default)");
        assert!(
            d.fused && d.max_chunks > 1,
            "packed = fused with several chunks"
        );
        let off = mixed_prefill_of(Some("off"), 8192).expect("off parses");
        assert_eq!(
            off,
            saber_sched::MixedPrefill::OFF,
            "--engine-mixed-prefill off: the control"
        );
        assert!(
            mixed_prefill_of(Some("sideways"), 8192).is_err(),
            "an unknown form is refused"
        );
    }

    #[test]
    fn g6s_levers_are_the_default_and_the_off_flags_are_the_controls() {
        use super::on_unless_off;
        assert!(
            on_unless_off(false, false),
            "no flag: the mid band / skinny gate (the default)"
        );
        assert!(
            on_unless_off(true, false),
            "the on flag: a no-op naming the default"
        );
        assert!(!on_unless_off(false, true), "the off flag: the pre-G6 law");
    }

    #[test]
    fn the_mid_chunk_tiles_are_the_default_and_the_off_flag_is_the_control() {
        assert!(
            mid_chunk_gemm_law(false, false),
            "no flag: the mid-M tiles (the default)"
        );
        assert!(
            mid_chunk_gemm_law(true, false),
            "--engine-mid-chunk-gemm: a no-op naming the default"
        );
        assert!(
            !mid_chunk_gemm_law(false, true),
            "--no-engine-mid-chunk-gemm: the reference tile"
        );
    }

    #[test]
    fn the_index_law_is_the_default_and_temporal_is_the_control() {
        assert!(
            drafter_rope_law(false, false),
            "no flag: the index law (the default)"
        );
        assert!(
            drafter_rope_law(true, false),
            "--drafter-rope-index: a no-op naming the default"
        );
        assert!(
            !drafter_rope_law(false, true),
            "--drafter-rope-temporal: the control"
        );
    }
}

/// C4: the mixed step's switch and budget. A budget of 0 would make every mixed step run its
/// decode half alone forever while the chunk waited — a schedule that never admits — so it refuses
/// by name rather than serve it.
/// P27-W2-2: `--engine-device-step` / `--device-step` checked at parse time; the canonical spelling
/// is what the binary carries.
fn parse_device_step_arg(s: &str) -> Result<String, String> {
    saber_engine::device_step::DeviceStep::parse(s).map(|f| f.as_arg().to_owned())
}

/// P27-14: `--engine-host-levers` / `--host-levers` checked at parse time; the canonical spelling
/// is what is kept (and forwarded to rank 1).
fn parse_host_levers_arg(s: &str) -> Result<String, String> {
    saber_engine::host_levers::HostLevers::parse(s)
        .map(saber_engine::host_levers::HostLevers::as_arg)
}

/// `--poison-boot SET:BYTE` checked at parse time (T6's diagnostic).
fn parse_poison(s: &str) -> Result<String, String> {
    saber_engine::device::PoisonBoot::parse(s).map(|_| s.to_owned())
}

fn mixed_prefill_of(
    form: Option<&str>,
    max_batched_tokens: u32,
) -> anyhow::Result<saber_sched::MixedPrefill> {
    // `packed` is the DEFAULT since 2026-09-24 (the user's decision on C11's three-arm pair: corpus
    // 99.3% of prod, conc-8 TTFT p99 393 -> 290 ms, p50 87 -> 111 ms still under prod's 119.5,
    // GSM8K 97.60 in band). `off` is the unmixed schedule, the pairing control.
    let form = form.unwrap_or(DEFAULT_MIXED_PREFILL);
    if form == "off" {
        return Ok(saber_sched::MixedPrefill::OFF);
    }
    if max_batched_tokens == 0 {
        anyhow::bail!("a mixed step with a token budget of 0 never feeds a chunk");
    }
    match form {
        "serial" => Ok(saber_sched::MixedPrefill::on(max_batched_tokens)),
        "fused" => Ok(saber_sched::MixedPrefill::fused(max_batched_tokens)),
        "packed" => Ok(saber_sched::MixedPrefill::packed(max_batched_tokens)),
        other => anyhow::bail!(
            "--engine-mixed-prefill {other}: the forms are `packed` (C10/C11, the default), \
             `fused` (C9, one chunk a forward), `serial` (C4, bit-exact) and `off` (the unmixed \
             schedule, the control)"
        ),
    }
}

#[cfg(test)]
mod best16_serve_tests {
    use clap::Parser;
    #[test]
    fn serve_exposes_the_implemented_sixteen_row_tree() {
        let form = std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn(|| {
                match crate::Cli::try_parse_from(["saber", "serve", "--engine-spec-tree", "best16"])
                    .expect("implemented tree accepted by serve CLI")
                    .cmd
                {
                    crate::Cmd::Serve {
                        engine_spec_tree, ..
                    } => engine_spec_tree,
                    _ => panic!("serve command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(form, "best16");
    }

    #[test]
    fn host_and_rccl_flags_expose_the_explicit_hybrid_profile() {
        std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn(|| {
                match crate::Cli::try_parse_from([
                    "saber",
                    "serve",
                    "--engine-host-ar",
                    "--engine-rccl-lib",
                    "/runtime/librccl.so",
                ])
                .expect("hybrid transport accepted")
                .cmd
                {
                    crate::Cmd::Serve {
                        engine_host_ar,
                        engine_rccl_lib,
                        ..
                    } => {
                        assert!(engine_host_ar);
                        assert_eq!(engine_rccl_lib.as_deref(), Some("/runtime/librccl.so"));
                    }
                    _ => panic!("serve command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
