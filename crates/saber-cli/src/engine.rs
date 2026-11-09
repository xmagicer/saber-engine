//! `saber engine` and `saber serve --engine real` — the P3 seam on a card.
//!
//! `saber forward` runs the whole model once. `saber serve --mock` runs the whole serving plane over
//! a scripted executor. This is the join: `saber-engine`'s [`RealExecutor`] under `saber-sched`'s
//! scheduler, driving the same pass `saber forward` drives, one step at a time, for as many steps as
//! a request takes — and now for as many **sequences at once** as the bucket's batch.
//!
//! # The gate is structural, and it is relative
//!
//! Two ops have no code object on this branch (`qk_norm_rope`, `attn_gate_quant`) and the K/V append
//! is state no op writes, so the tokens are **not the model's tokens** — exactly as `saber forward`
//! says with the same banner. What is therefore gated here is everything that is not arithmetic:
//!
//! * **batched identity**: four prompts decoded together give each sequence the ids it gets alone;
//! * **path identity**: a replayed graph gives the ids the eager op-by-op walk gives;
//! * **serve identity**: the same prompt through the pass directly and through the serve path gives
//!   the same ids, one `StepEvent` per engine step;
//! * every step's dispatch count equals the budget the plan declares for the shape it ran
//!   (`CLAUDE.md` rule 7), asserted at capture as the graph's node count and per step against the
//!   count issued;
//! * every protocol fault is counted, and the run is only clean if every counter is zero.
//!
//! And the measurements: **ms/step at batch 1, 4 and 8**, graph and eager, against the P-1b
//! comparator (18.7 ms/step, prod stack without speculation, 210 W, GPU idle — `docs/LEDGER.md`).
//!
//! GPU rules (CLAUDE.md): one HIP device, under the shared lock, prod asserted idle by VRAM first,
//! never touching the power cap.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use saber_engine::host_trace::HostTraceReport;
use saber_engine::{
    DecodeSlot, Drafter, EngineConfig, EngineDrafter, FixedDrafter, ForwardRunner, HipForward,
    NoDrafter, ParkImageLaw, PeerConfig, PeerRole, ProbeDepth, RealExecutor, RecordedDrafter,
    RowProbe, RowRequest, StepRows,
};
use saber_mem::ledger::{Ledger, Pool};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    attach_park_tier, spawn_step_loop, stop_with_deadline, ArenaBytes, KvGeometry, Scheduler,
    SchedulerConfig, SeqId, ServeParkOptions, WidthRule,
};
use saber_serve::detok::ModelCodec;
use saber_serve::engine::{Engine, EngineRequest, ModelInfo, TokenId};
use saber_serve::sampling::SamplingParams;
use saber_serve::state::{ChatRenderer, TemplateRenderer};

const GIB: u64 = 1 << 30;

/// The token the `fixed` mock proposes. Inside every vocabulary (so it is a legal id the model could
/// in principle want) and arbitrary (so it almost never does).
const FIXED_DRAFT_TOKEN: TokenId = 4321;

/// How to build the engine. Shared by the gate and by the real serve.
#[derive(Clone, Debug)]
pub struct EngineArgs {
    /// The device-final pack: a directory holding `rank<r>.pack`, or one such file.
    pub pack: PathBuf,
    /// Ranks in the tier: 1 (one card) or 2 (this process plus a worker on the other).
    pub world: u32,
    /// TP rank of the pack this process opens.
    pub rank: u32,
    /// At `world > 1`: the loopback port the two ranks meet on. 0 asks the OS for a free one.
    pub peer_port: u16,
    /// At `world > 1`: this process's role. Rank 0 leads and hosts everything above the seam.
    pub follower: bool,
    /// At `world > 1`: HIP device the worker takes. Ignored at `world == 1`.
    pub peer_device: i32,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// The HIP runtime's soname.
    pub hip_lib: String,
    /// Optional RCCL compatibility library; selects host-capable collectives.
    pub rccl_lib: Option<String>,
    /// Explicit native exact collectives over registered shared host pages.
    pub host_ar: bool,
    /// Diagnostic rank-zero target-feature capture for offline draft validation.
    pub draft_capture_dir: Option<PathBuf>,
    /// An explicit Unix socket for external DSpine proposals.
    pub dspine_socket: Option<PathBuf>,
    /// Independent target feature plane used by offline draft validation.
    pub draft_capture_layers: Vec<u32>,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Refuse to start if the engine's device buffers would exceed this.
    pub budget_mib: u64,
    /// Sequences the engine holds session state for.
    pub max_seqs: u32,
    /// Tokens one sequence may reach.
    pub max_ctx: u32,
    /// Prefill chunk sizes to plan. A prompt is padded up into the smallest one that holds it.
    pub prefill_rows: Vec<u32>,
    /// Decode batches to plan and capture.
    pub decode_batches: Vec<u32>,
    /// Verify widths to plan and capture. `[1]` is SPEC=0.
    pub decode_widths: Vec<u32>,
    /// **The decode ROW BANDS (C2)**, in rows: empty is `--rectangular-widths`, the pairing
    /// control, under which the engine plans no band and the scheduler runs today's rectangles.
    /// FORWARDED to rank 1 (a pair that planned different bands would disagree on the collective
    /// digests before a launch).
    pub decode_bands: Vec<u32>,
    /// Where a ragged step's per-slot ceiling comes from (`--ragged-ceiling table|block`): the
    /// SCHEDULER's, like [`EngineArgs::width_rule`], so rank 1 has no use for it.
    pub ragged_ceiling: saber_sched::RaggedCeiling,
    /// Which rule picks a step's verify width out of those: the batch table (the default) or
    /// prod's acceptance-EMA cap. The engine itself does not read it — it is the SCHEDULER's
    /// policy — so it is carried here rather than in [`saber_engine::EngineConfig`], and rank 1
    /// has no scheduler and no use for it.
    pub width_rule: WidthRule,
    /// C4: whether a decode step may carry the queue head's next prefill chunk, and the budget.
    /// The SCHEDULER's policy like `width_rule`; rank 1 is told its steps and never reads it.
    pub mixed_prefill: saber_sched::MixedPrefill,
    /// C7: the prefill FILL RULE (`saber_sched::PrefillChunking::with_fill_above`): rungs above
    /// this many rows are taken only by a chunk that fills them. 0 = off (the padded law). The
    /// SCHEDULER's policy; rank 1 is told its chunks.
    pub prefill_fill_above: u32,
    /// Seconds one engine call may take before the engine is declared unresponsive.
    pub deadline_s: u64,
    /// Capture each decode batch at startup and replay it per step.
    pub capture: bool,
    /// Let the argmax feed the next step's ids on the device.
    pub device_feed: bool,
    /// Point a padded prefill chunk's attention at the rows the CHUNK feeds (ws-h27). `false` is
    /// the NEGATIVE CONTROL for the continuity arms: with it the plan keeps the bucket's row
    /// count, every padded row is a live query row again, and the same gate reads 11/15 where it
    /// reads 15/15 with it on.
    pub live_row_attn: bool,
    /// Page-locked host MiB for the session-park staging buffer. 0 disables parking.
    pub park_staging_mib: u64,
    /// Host GiB the SCHEDULER's parking tier will hold, so the engine declares `Pool::HostPinned`
    /// wide enough for it (ws-k14). `ServeArgs::park_ram_gib` is where the serve plane sets it.
    pub park_ram_gib: f64,
    /// Staging slots to divide it into: 1 (a park and a restore take turns) or 2 (they overlap).
    pub park_staging_slots: u32,
    /// **The park arm's NEGATIVE CONTROL**: image the state-slot run's BASE for every state,
    /// which is what a park image carried before ws-k14.
    ///
    /// A session that lands back on its own slots restores correctly under either law, so the only
    /// way to see the difference is to park a session that MOVES and image the wrong slot on
    /// purpose. With this on the park arm must FAIL; with it off it must pass.
    pub park_image_base: bool,
    /// **The park probe** (ws-k15): dump the DRAFTER half of every park image, per cache, per
    /// layer, per page, on the rank that made it — and, in the park arm, re-park the restored
    /// session so the two dumps can be held against each other.
    pub park_probe: bool,
    /// MiB kept back from the derived KV pool for the reservations made after it is sized.
    pub kv_headroom_mib: u64,
    /// KV pool blocks, **explicitly**, or 0 to derive it from the byte ledger (the default).
    ///
    /// The engine has always taken this; nothing exposed it. An UNDER-PRESSURE arm needs a pool
    /// the scheduler can exhaust, and shrinking it through `--kv-headroom-mib` means guessing at a
    /// subtraction from a budget — a few MiB either way is the difference between "the arenas
    /// refuse" and "they never do". This states the number (ws-k14).
    pub kv_pool_blocks: u32,
    /// Plan a DFlash2 drafter tier at this `block_size`, and 0 for none.
    ///
    /// Prod's block is 8 (`1 + num_speculative_tokens`, and it must be a power of two). A non-zero
    /// value makes the tier the engine's own -- billed to the byte ledger before the KV pool is
    /// derived from what is left, and admitted and released with every session -- instead of an
    /// accounting line this gate prints beside a mock.
    pub draft_block: u32,
    /// D4: the vision tower's arena in patches, on EVERY rank of the tier (0 = text-only). At TP 2
    /// it crosses rank 1's argv like every other plan input: the tower is replicated and rank 1
    /// runs it on the pixels `Command::Vision` hands it.
    pub vision_max_patches: u32,
    /// P27-11 `--vision lazy`: the STREAMED tower (no VRAM billed; forwarded to rank 1).
    pub vision_lazy: bool,
    /// D6 `--drafter-rope-index`: an image sequence's drafter tables at the token INDEX (prod's V2
    /// runner) instead of D4's temporal M-RoPE axis. Leader-only: the follower's drafter positions
    /// travel in the leader's frames, so rank 1's argv does not carry it.
    pub drafter_rope_index: bool,
    /// The DRAFTER's own device-final pack, so the tier is RESIDENT rather than only billed.
    ///
    /// With it the engine uploads the drafter through the same `Upload` the target takes, allocates
    /// its two contiguous KV caches from `DraftKvCache`'s geometry, lays the proposal's arena out
    /// and sizes every workspace before the first capture. Without it `--draft-block` bills the
    /// tier and nothing is uploaded, which is what ws-s2 shipped.
    pub draft_pack: Option<PathBuf>,
    /// Replay the proposal against a captured step's own inputs and report every intermediate.
    pub draft_replay: Option<PathBuf>,
    /// The STANDALONE chain's own dumped intermediates (`saber draft-step --proposal --no-seed
    /// --dump-chain`), which `--draft-replay` compares the engine's arena against operand by
    /// operand. Without it the replay's only comparator is prod, which cannot separate the
    /// engine's plumbing from ws-f's chain.
    pub draft_chain_dump: Option<PathBuf>,
    /// Replay the proposal at EVERY step of a page-carrying prod SEQUENCE capture (ws-f36), so
    /// that "how often does the walk pick a different token from prod's" is a rate.
    pub draft_seq: Option<PathBuf>,
    /// Where `--draft-seq` writes its per-step blocks, scores and candidate ids.
    pub draft_seq_out: Option<PathBuf>,
    /// Raise the all-reduce band above every message this run can send, so that every shape
    /// resolves to the exact family. TP=2 only.
    pub ar_exact: bool,
    /// **Ask the batched proposal whether it is the one-at-a-time proposal** (ws-s10).
    ///
    /// With it on, the gate's batched pass runs every proposal TWICE — once as the batch the step
    /// schedules and once per request at the one-request shape, out of the same `aux.cat` rows —
    /// and compares the blocks row for row, with a cross-request comparison as the control. It
    /// doubles that arm's proposals, so it is off by default and the acceptance numbers are never
    /// taken from a run that had it on.
    pub draft_batch_check: bool,
    /// **PAIRING CONTROL** (H35): every prefill chunk's drafter context stored at `block` rows a
    /// tile, main's law. The leader's alone, never forwarded: rank 1 runs whatever width a frame names.
    pub draft_store_narrow: bool,
    /// C6's pairing control: a chunk below the class boundary as 8-row tiles (H35's law).
    pub draft_store_natural_off: bool,
    /// **GATE** (H35): the wide store compared byte for byte against a narrow re-store, per chunk,
    /// on both ranks. The leader's alone: rank 1 acts on the two markers it is sent.
    pub draft_store_check: bool,
    /// P27-3: `--drafter-store deferred` — the last chunk's drafter store after the first token.
    /// The leader's alone: rank 1 is told per proposal.
    pub draft_store_deferred: bool,
    /// P27-6: `--draft-kv paged` — the drafter's KV paged by admission. FORWARDED to rank 1 by
    /// value: both ranks derive one KV pool and one drafter pool from it.
    pub draft_kv_paged: bool,
    /// P27-9b: `--draft-embed-share on` — the drafter reads the target's embedding record (one
    /// copy a rank). FORWARDED to rank 1 by value: both ranks derive one KV pool.
    pub draft_embed_share: bool,
    /// P27-12: `--selector-rank walk|explen` -- how a proposal ranks the selector's lattice.
    /// FORWARDED to rank 1 by value: the two ranks compare one block digest.
    pub selector_rank: saber_engine::selector_rank::SelectorRank,
    /// Score the full FP8 vocabulary for a single B8 proposal.
    pub draft_full_vocab: bool,
    /// P27-15c: `--spec-tree off|ancestor9` -- the verify's row form (the tree is an ARM; a form
    /// the boot cannot run is refused by name and runs `off`). FORWARDED to rank 1 by value.
    pub spec_tree: saber_engine::spec_tree::SpecTreeForm,
    /// P27-W2-3: `--spec-tree-sampled on|off` -- the sampled tree acceptance law (an ARM, default
    /// off). FORWARDED to rank 1 by value.
    pub spec_tree_sampled: bool,
    /// P27-W2-4: `--spec-tree-sampled control` -- the biased control (leader only, not forwarded).
    pub spec_tree_sampled_control: bool,
    /// P27-W2-5: `--gdn-replay on|off` -- the GDN decode update as base + stash + replay (an ARM,
    /// off by default). FORWARDED to rank 1 by value: both ranks plan one recurrent form.
    pub gdn_replay: bool,
    /// P27-14: `--host-levers picks,table,socket|none` -- the 27B's bit-exact host levers.
    /// FORWARDED to rank 1 by value: every lever changes the pair's frames.
    pub host_levers: saber_engine::host_levers::HostLevers,
    /// P27-W2-2: `--device-step auto|off|cand` -- the 27B's device-step form (forwarded by value;
    /// `auto` is the default since CMP27P, resolved on each rank at the engine's config head).
    pub device_step: saber_engine::device_step::DeviceStep,
    /// H36: slice rows of the prefill all-reduce overlap; 0 = off. FORWARDED to rank 1.
    pub prefill_ar_overlap: u32,
    /// P27-1d: the split form's push blocks inside the overlap's slices; 0 = the one-shot (the
    /// control). FORWARDED to rank 1 as `--ar-form split --ar-push-blocks N`.
    pub prefill_ar_push_blocks: u32,
    /// C6: `--small-chunk-gemm`, the prefill rungs at or below this many rows on the decode GEMM
    /// family (0 = off). Forwarded to rank 1 by value.
    pub small_chunk_gemm: u32,
    /// C6: `--small-chunk-band`, those rungs at the reference's split band (numerics class).
    pub small_chunk_band: bool,
    /// C7: `--shared-prefill-arena`, every prefill rung over one arena sized to the widest.
    /// Forwarded to rank 1.
    pub shared_prefill_arena: bool,
    /// H46: `--shared-head-arena`, every decode head over one arena sized to the widest head.
    /// Forwarded to rank 1.
    pub shared_head_arena: bool,
    /// G5: `--mid-chunk-gemm`, the folded GEMM at the mid-M tile heights (`[[folded_mt]]`).
    /// Forwarded to rank 1.
    pub mid_chunk_gemm: bool,
    /// G6: `--mid-chunk-atiled`, the layout law's mid band (tiled producers + the A-tiled family
    /// at 129..512 rows). Forwarded to rank 1.
    pub mid_chunk_atiled: bool,
    /// G6: `--skinny-gate`, the 48-wide GDN gate on the skinny-N body. Forwarded to rank 1.
    pub skinny_gate: bool,
    /// CMP27E ARM: `--decode-pf`, the b8 (tm4) decode GEMMs on the register-prefetch twins
    /// (`mxfp4_gemm_decode_pf`, bit-exact). Forwarded to rank 1.
    pub decode_pf: bool,
    /// CMP27F ARM: `--head-m64`, the fp8 head at 49..64 rows in ONE weight pass
    /// (`saber_head_w8_m64`, bit-exact). Forwarded to rank 1.
    pub head_m64: bool,
    /// CMP27K ARM: `--drafter-gateup-m64 [MIN_ROWS]`, the drafter's gate_up at MIN_ROWS..64 rows
    /// in ONE weight pass (`dflash_gemm_m64`, bit-exact); 0 = off. Forwarded to rank 1.
    pub drafter_gateup_m64: u32,
    /// CMP27S (DEFAULT on): `--ar-wpar on|off`, the wht6 reduce's wave-parallel body
    /// (`saber_ar_wht_wpar`, bit-exact, the served grid); `off` = the control. Forwarded to rank 1
    /// by value.
    pub ar_wpar: bool,
    /// CMP27AE (DEFAULT on since its flip; `off` = the control): `--ar-fuse on|off`, the N 5120 dwn4 decode GEMM + wht6 reduce as one
    /// launch (`mxfp4_gemm_decode_arw`, bit-exact). Forwarded to rank 1 by value (a fused rank's
    /// tile-contiguous wire does not interoperate with a two-launch rank).
    pub ar_fuse: bool,
    /// CMP27L ARMS: `--b1-arms`, the b1 levers (tree conv rows, batched flush advance, the
    /// 1..16-row head on the 32-row band; bit-exact). Forwarded.
    pub b1_arms: saber_engine::device::B1Arms,
    /// CMP27T ARMS: `--c4-arms`, the conc-4 levers (the LDS replay, the tm2 prefetch GEMMs, the
    /// argmax with its winner's logit; bit-exact). Forwarded.
    pub c4_arms: saber_engine::device::C4Arms,
    /// CMP27X ARMS: `--step-arms`, the step levers (the seam row's load-ahead twin, the 4-wave
    /// N 5120 prefetch GEMMs; bit-exact). Forwarded.
    pub step_arms: saber_engine::device::StepArms,
    /// C8 GATE CONTROL: `--mixed-class-proxy`, every decode plan in the fused mixed step's
    /// decode-row class (`saber_engine::mixed_class`). Forwarded to rank 1.
    pub mixed_class_proxy: bool,
    /// ws-h37: the decode launch-train fusions ([`saber_engine::device::EngineConfig::decode_fusion`]).
    pub decode_fusion: saber_engine::device::DecodeFusion,
    /// **Record and print the per-step HOST-time breakdown** (ws-h31, `saber_engine::host_trace`).
    ///
    /// A DIAGNOSTIC: every number a traced boot prints is a traced number, so this is off by
    /// default and no published figure is taken with it on (`docs/GATES.md` blind spot 19). It
    /// turns on three things: a phase table over the timed arm, one per batch of the sweep, and
    /// the RESIDUE arm at the very end of the gate — the same steps before and after ws-p9's
    /// bucket attribution, per phase, with `--no-attribute` as the control that must read zero.
    pub host_trace: bool,
    /// **GATE CONTROL**: send the whole step-varying index region to the peer every step, the way
    /// every step did before ws-h31 compacted it to the runs the step wrote.
    pub whole_step_frame: bool,
    /// **GATE CONTROL**: carry an FNV of that region in every frame and make the follower check it.
    pub frame_digest: bool,
    /// T4's race checks (`--race-check`), forwarded to rank 1.
    pub race_check: bool,
    /// T6's shadow digest (`--race-check shadow`), forwarded to rank 1. Never with `race_check`.
    pub race_shadow: bool,
    /// T6's boot poison (`--poison-boot SET:BYTE`, a diagnostic), forwarded to rank 1.
    pub poison_boot: Option<String>,
    /// T7's gate control (`--constant-seed-control`): the pre-T7 re-seed policy, forwarded to
    /// rank 1.
    pub constant_seed_control: bool,
    /// **GATE CONTROL**: read the logits gather one ROW at a time, the way it did before ws-h31
    /// batched it. One arm says whether the batch is why two binaries commit differently at TP=2.
    pub gather_per_row: bool,
    /// **PAIRING CONTROL** (ws-f37): propose inside the decode step from the argmax, the order this
    /// engine had before ws-f37. The default runs the proposal after the commit.
    pub propose_in_step: bool,

    /// **CONTROL** (ws-h32): every step wait on the fixed `WaitWindow::STEP` instead of aimed.
    pub fixed_step_wait: bool,
    /// FN73 (DEBUG): rank 1 fails after this many steps (0 = never; `--fail-rank-after-steps 1:N`).
    pub fail_after_steps: u64,
    /// FN57: the host-wait arm (`--host-wait`; [`DEFAULT_HOST_WAIT_27B`] when absent).
    pub host_wait: saber_exec::HostWait,
    /// The GDN recurrent state STORED as f16 (C1, `--gdn-state-dtype f16`). Forwarded to rank 1.
    pub gdn_state_f16: bool,
    /// The target's `lm_head`: the EXACT head by default (N5; today the bf16 record, shared with
    /// the drafter's), `Mxfp4` under `--mxfp4-head`. Forwarded to rank 1 as `--head`.
    pub head: saber_engine::head_share::TargetHead,
}

/// FN57: the 27B's host-wait arm when `--host-wait` is absent: ws-h32's aimed wait (today's).
pub const DEFAULT_HOST_WAIT_27B: saber_exec::HostWait = saber_exec::HostWait::Aimed;

/// FN57: `--host-wait` for the 27B (absent = [`DEFAULT_HOST_WAIT_27B`]).
///
/// # Errors
/// An unknown arm name.
pub fn host_wait_27b(v: Option<&str>) -> Result<saber_exec::HostWait, String> {
    v.map_or(Ok(DEFAULT_HOST_WAIT_27B), str::parse)
}

/// `--race-check [MODE]` (T4/T5 `full`, T6 `shadow`) as `(race_check, race_shadow)`.
///
/// # Errors
/// A mode that is neither.
pub fn race_mode(v: Option<&str>) -> Result<(bool, bool), String> {
    match v {
        None => Ok((false, false)),
        Some("full") => Ok((true, false)),
        Some("shadow") => Ok((false, true)),
        Some(other) => Err(format!(
            "--race-check {other}: the modes are `full` (T4/T5's checks, the bare flag) and \
             `shadow` (T6's on-card proposal digests, no sync)"
        )),
    }
}

impl EngineArgs {
    fn config(&self) -> EngineConfig {
        let peer = if self.world > 1 {
            Some(PeerConfig {
                role: if self.follower {
                    PeerRole::Follower
                } else {
                    PeerRole::Leader
                },
                port: self.peer_port,
                // Generous, because it also covers the other rank's whole startup: a 7.7 GiB weight
                // upload, 64 plans and four captures happen between the socket opening and the first
                // frame crossing it. Bounded all the same — that is the rule.
                deadline_s: 600,
            })
        } else {
            None
        };
        EngineConfig {
            pack: self.pack.clone(),
            world: self.world,
            peer,
            rank: self.rank,
            prebuilt: self.prebuilt.clone(),
            hip_lib: self.hip_lib.clone(),
            rccl_lib: self.rccl_lib.clone(),
            host_ar: self.host_ar,
            dspine_socket: self.dspine_socket.clone(),
            draft_capture_layers: self.draft_capture_layers.clone(),
            draft_capture_dir: if self.rank == 0 {
                self.draft_capture_dir.clone()
            } else {
                None
            },
            device: self.device,
            budget_mib: self.budget_mib,
            max_seqs: self.max_seqs,
            max_ctx: self.max_ctx,
            prefill_rows: self.prefill_rows.clone(),
            decode_batches: self.decode_batches.clone(),
            decode_widths: self.decode_widths.clone(),
            decode_bands: self.decode_bands.clone(),
            // 0: DERIVED from the byte ledger (ws-k13). `max_ctx x max_seqs` is a ceiling on what
            // one admission may ask for, not a size for the arena everything shares — sizing the
            // pool from it left 14 GiB of a 32 GiB card reserved by nobody (ws-k12). A non-zero
            // `--kv-pool-blocks` states it instead, which is how an arm makes the arenas refuse
            // (ws-k14).
            kv_pool_blocks: self.kv_pool_blocks,
            kv_headroom_mib: self.kv_headroom_mib,
            step_deadline: Duration::from_secs(self.deadline_s.max(1)),
            capture: self.capture,
            device_feed: self.device_feed,
            live_row_attn: self.live_row_attn,
            // 0 disables parking, which is the default: the buffer is page-locked host memory the
            // engine declares and reserves for itself, and a command that is not going to park
            // should not hold it. `--park-staging-mib` turns it on, and the engine refuses to START
            // if what is asked for cannot hold one image at `max_ctx` -- with the arithmetic in the
            // message.
            park_staging_bytes: self.park_staging_mib << 20,
            park_staging_slots: self.park_staging_slots,
            // The parking TIER's bytes, declared with the engine's own pinned pool because the
            // engine is what declares it (ws-k14). A tier attached to a pool that has no room for
            // it refuses every eviction and the run looks like one that simply never evicted.
            park_tier_bytes: (self.park_ram_gib.max(0.0) * (1u64 << 30) as f64) as u64,
            park_probe: self.park_probe,
            park_image_law: if self.park_image_base {
                ParkImageLaw::BaseOnly
            } else {
                ParkImageLaw::Live
            },
            draft_block: self.draft_block,
            draft_pack: self.draft_pack.clone(),
            draft_store_narrow: self.draft_store_narrow,
            draft_store_natural_off: self.draft_store_natural_off,
            draft_store_check: self.draft_store_check,
            draft_store_deferred: self.draft_store_deferred,
            draft_kv_paged: self.draft_kv_paged,
            draft_embed_share: self.draft_embed_share,
            selector_rank: self.selector_rank,
            draft_full_vocab: self.draft_full_vocab,
            spec_tree: self.spec_tree,
            spec_tree_sampled: self.spec_tree_sampled,
            spec_tree_sampled_control: self.spec_tree_sampled_control,
            gdn_replay: self.gdn_replay,
            host_levers: self.host_levers,
            device_step: self.device_step,
            prefill_ar_overlap: self.prefill_ar_overlap,
            prefill_ar_push_blocks: self.prefill_ar_push_blocks,
            small_chunk_gemm: self.small_chunk_gemm,
            small_chunk_band: self.small_chunk_band,
            shared_prefill_arena: self.shared_prefill_arena,
            shared_head_arena: self.shared_head_arena,
            mid_chunk_gemm: self.mid_chunk_gemm,
            mid_chunk_atiled: self.mid_chunk_atiled,
            skinny_gate: self.skinny_gate,
            decode_pf: self.decode_pf,
            head_m64: self.head_m64,
            drafter_gateup_m64: self.drafter_gateup_m64,
            ar_wpar: self.ar_wpar,
            ar_fuse: self.ar_fuse,
            b1_arms: self.b1_arms,
            c4_arms: self.c4_arms,
            step_arms: self.step_arms,
            mixed_class_proxy: self.mixed_class_proxy,
            decode_fusion: self.decode_fusion,
            ar_exact: self.ar_exact,
            host_trace: self.host_trace,
            whole_step_frame: self.whole_step_frame,
            frame_digest: self.frame_digest,
            race_check: self.race_check,
            race_shadow: self.race_shadow,
            poison_boot: self
                .poison_boot
                .as_deref()
                .and_then(|s| saber_engine::device::PoisonBoot::parse(s).ok()),
            constant_seed_control: self.constant_seed_control,
            gather_per_row: self.gather_per_row,
            propose_at: if self.propose_in_step {
                saber_engine::ProposeAt::InStep
            } else {
                saber_engine::ProposeAt::AfterCommit
            },
            fixed_step_wait: self.fixed_step_wait,
            fail_after_steps: if self.follower {
                self.fail_after_steps
            } else {
                0
            },
            host_wait: self.host_wait,
            gdn_state_f16: self.gdn_state_f16,
            // D4: the tier's own field, forwarded to rank 1's argv (`Worker::command`).
            vision_max_patches: self.vision_max_patches,
            vision_lazy: self.vision_lazy,
            drafter_rope_index: self.drafter_rope_index,
            head: self.head,
        }
    }

    /// The ledger the engine bills itself to and the scheduler then reserves its arenas from: one
    /// view of one card.
    fn ledger(&self, vram_gib: u64) -> anyhow::Result<Ledger> {
        let mut l = Ledger::new();
        l.declare(Pool::Vram(self.rank), vram_gib * GIB)?;
        l.declare(Pool::HostPageable, 48 * GIB)?;
        l.declare(Pool::Nvme, 64 * GIB)?;
        Ok(l)
    }
}

/// Everything the engine says about itself, printed the way `saber forward` prints it.
/// Admit `max_ctx`-token sessions until the arenas refuse, and say how many got in.
///
/// Uses the **allocator**, not the ledger's arithmetic: `Arenas::allocate_group` is what admission
/// actually calls, and it is what says no. Every allocation is handed straight back, so this leaves
/// the scheduler exactly as it found it and can run at boot without costing a session anything.
fn admit_until_refusal<E: saber_sched::StepExecutor>(
    scheduler: &Scheduler<E>,
    ctx: u32,
    predicted_sessions: u64,
) -> String {
    use std::fmt::Write as _;

    let cfg = scheduler.config();
    // A throwaway ledger with the same arena bytes: the point is the ALLOCATOR's geometry, and
    // billing a second ledger against the real one would double-count the same pages.
    //
    // It has to be DECLARED. `Ledger::new()` declares nothing, so every reservation against it is
    // `Undeclared` -- and this probe caught that error and returned an EMPTY STRING, which prints
    // as nothing at all between the census and the arenas line. The whole probe, and with it
    // ws-k13's census-against-allocator check, had been silently absent from every boot log
    // (measured 2026-09-21, ws-k13b). A probe that cannot run says so; that is rule 5.
    let mut probe = Ledger::new();
    for r in 0..cfg.tp {
        if let Err(e) = probe.declare(
            Pool::Vram(r),
            cfg.arenas.kv_per_rank + cfg.arenas.gdn_state_per_rank,
        ) {
            return format!("admit     the probe could not declare its own ledger: {e}\n");
        }
    }
    let mut arenas = match saber_sched::Arenas::reserve(&cfg, &mut probe) {
        Ok(a) => a,
        Err(e) => return format!("admit     the probe could not reserve the arenas: {e}\n"),
    };
    let mut taken = Vec::new();
    let want = u64::from(ctx);
    let mut refusal = None;
    for _ in 0..4096 {
        match arenas.allocate_group(1, want, want) {
            Ok(pair) => taken.push(pair),
            Err(r) => {
                refusal = Some(r.to_string());
                break;
            }
        }
    }
    let n = taken.len();
    let (free_kv, free_gdn) = (arenas.kv().free_blocks(), arenas.gdn().free_sets());
    for (kv, gdn) in taken {
        for (k, g) in kv.into_iter().zip(gdn) {
            arenas.free(k, g);
        }
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "admit     {n} session(s) of {ctx} tokens before the arenas refused: {}",
        refusal.unwrap_or_else(|| "nothing refused in 4096 tries".to_owned())
    );
    // The census predicted a number; the allocator just produced one. They come from the same two
    // geometry figures, so anything but equality is a bug in one of them (ws-k13).
    let predicted = predicted_sessions;
    let g = saber_engine::Granted {
        predicted,
        granted: n as u64,
    };
    let _ = writeln!(out, "{}", g.render());
    let _ = writeln!(
        out,
        "admit     {free_kv} KV block(s) and {free_gdn} state slot set(s) left over — too small for \
         one more session, which is what a refusal means"
    );
    out
}

/// The ledger's every reservation on one rank, and what is left in KV tokens.
///
/// Reads the scheduler's ledger **after** its arenas are reserved, which is the only moment the
/// picture is complete: the engine bills its own buffers during `spawn` and the scheduler bills the
/// two arenas when it is built.
fn capacity_report<E: saber_sched::StepExecutor>(
    scheduler: &Scheduler<E>,
    rank: u32,
    ctx: u32,
    stride: u64,
    pool_blocks: u32,
) -> (String, u64) {
    use std::fmt::Write as _;

    let snap = scheduler.ledger_snapshot();
    let cfg = scheduler.config();
    // The census reports what the card HOLDS, so it uses the arena stride — the scheduler's own
    // geometry is per-layer and stays that way (see `scheduler_config`). The two differ by the
    // attention-layer count and conflating them is how ws-k13's first cut broke every sequence's
    // page count.
    let geom = saber_engine::KvGeometry {
        block_bytes: stride,
        block_tokens: cfg.kv.block_tokens,
        gdn_bytes_per_slot: cfg.gdn_bytes_per_slot,
        gdn_slots_per_seq: cfg.gdn_slots_per_seq(),
    };
    let mut cap = saber_engine::Capacity::from_snapshot(
        &snap,
        rank,
        &[saber_sched::KV_ARENA_RESERVATION],
        &[saber_sched::GDN_ARENA_RESERVATION],
        geom,
    );
    // THE LEDGER UNDER-COUNTS THE KV POOL, and the census corrects it rather than repeating it.
    //
    // `kv_arena` is reserved as `kv_pool_blocks * SchedulerConfig::kv.block_bytes()`, and that
    // geometry is per-layer by necessity (see `scheduler_config`), so the reservation is short of
    // what the pool costs the card by the attention-layer count. Dividing that short figure by the
    // true stride -- which is what this census did once ws-k13 substituted the stride here -- gives
    // a KV-token count 16x too SMALL: 15,059 tokens for a pool holding 240,944, and three
    // concurrent 4k sessions where the allocator grants eight. The two errors used to cancel (a
    // per-layer divisor against a per-layer reservation), which is why neither was visible.
    //
    // The card is the authority, so the card's number is reported: blocks times the stride. The
    // unreserved remainder loses the same bytes, because they are not free -- the engine has
    // already allocated them. The fix that would make this correction unnecessary is in
    // `saber-sched`: `ArenaBytes` needs the device bytes BESIDE `kv_per_rank`, one for the ledger
    // and one for the allocator's block count. That is another workstream's crate; it is named
    // here, not done here.
    let on_card = u64::from(pool_blocks) * stride;
    let undercount = on_card.saturating_sub(cap.kv);
    cap.kv = on_card;
    cap.unreserved = cap.unreserved.saturating_sub(undercount);
    let mut out = String::new();
    if let Some(pool) = snap.pools.iter().find(|p| p.pool == Pool::Vram(rank)) {
        let mut rows = pool.reservations.clone();
        rows.sort_by_key(|r| std::cmp::Reverse(r.bytes));
        let _ = writeln!(
            out,
            "ledger    vram(rank {rank}), every reservation, largest first:"
        );
        for r in &rows {
            let _ = writeln!(
                out,
                "ledger      {:<30} {:>8} MiB  {:>5.1}%",
                r.name,
                r.bytes >> 20,
                100.0 * r.bytes as f64 / pool.capacity_bytes.max(1) as f64
            );
        }
    }
    if undercount > 0 {
        let _ = writeln!(
            out,
            "ledger      kv_arena under-counts the pool by {} MiB: it is reserved in the \
             scheduler's per-layer units, and the pool costs the card {} MiB. The rows above are \
             the ledger's; the census below is the card's.",
            undercount >> 20,
            on_card >> 20
        );
    }
    out.push_str(&cap.render(ctx));
    (out, cap.sessions(ctx))
}

fn print_facts(forward: &HipForward, ledger: &Ledger, rank: u32) {
    let f = forward.facts();
    println!(
        "model     {} ({} layers, vocab {})",
        f.model, f.layers, f.vocab
    );
    println!(
        "weights   {} records bound by name; {} MiB of engine device buffers",
        f.weights,
        f.device_bytes >> 20
    );
    if f.tp > 1 {
        println!(
            "tier      TP={} rank {}: {} all-reduce(s) per step INSIDE the captured graph, {} MiB \
             peer-visible per rank",
            f.tp,
            f.rank,
            f.collectives,
            f.ipc_bytes >> 20
        );
    }
    println!(
        "session   {} sequence slot(s) + the kernels' null slot; KV pool {} blocks of {} KiB, block \
         table {} wide; GDN {} MiB per slot",
        f.max_seqs,
        f.kv_pool_blocks,
        f.kv_block_bytes >> 10,
        f.max_blocks,
        f.gdn_bytes_per_slot >> 20
    );
    println!(
        "ledger    vram(rank {rank}) {} MiB of {} MiB reserved after the engine billed itself",
        ledger.used(Pool::Vram(rank)) >> 20,
        ledger.capacity(Pool::Vram(rank)).unwrap_or(0) >> 20
    );
    // Printed even at 0, on purpose: this counted the GDN decode conv route until ws-h15 fixed
    // `saber-exec`'s descriptor, and a number that only appears when something is wrong cannot show
    // that it stopped being wrong.
    println!(
        "declared  {} launch declaration(s) corrected in the plan this engine built{}",
        f.corrected_declarations,
        if f.corrected_declarations == 0 {
            " — the routes declare what they dispatch"
        } else {
            " (a route carries a count its dispatch does not match; the per-op assertion stays on)"
        }
    );
    for (shape, declared) in &f.declared {
        let nodes = f
            .captured
            .iter()
            .find(|(s, _)| s == shape)
            .map(|(_, n)| format!(", captured as {n} nodes"))
            .unwrap_or_default();
        // WHICH ARITHMETIC THIS SHAPE RUNS. The exact/lossy band is a function of the MESSAGE, so
        // two shapes of one run can resolve two different all-reduce families and every arm that
        // compares them is then comparing two arithmetics (ws-l17). Printed per shape rather than
        // once, because that is the fact a reader needs and a single "the band is N" line hides.
        let ar = f
            .ar_family(*shape)
            .map(|fam| format!(", ar {fam}"))
            .unwrap_or_default();
        println!("shape     {shape}: {declared} dispatches declared{nodes}{ar}");
    }
    if f.ar_exact {
        println!(
            "ar        --ar-exact: the exact/lossy band is above every message this run can send, \
             so every shape resolves to the EXACT family"
        );
    }
    if let Some(banner) = f.partial_banner() {
        println!();
        println!("{banner}");
        println!(
            "          The ids below are therefore gated against EACH OTHER (batched vs solo, graph \
             vs eager, pass vs serve path), never against the model."
        );
        println!();
    }
}

/// The boot banner's RAGGED line (C2): the bands the engine captured, their arena, and the per-slot
/// rule — or the rectangular control.
fn print_ragged_banner(config: &SchedulerConfig, facts: &saber_engine::ForwardFacts) {
    let wp = &config.width;
    match wp.ragged {
        Some(ceiling) if !facts.decode_bands.is_empty() => {
            let rows: Vec<u32> = facts.decode_bands.iter().map(|b| b.2).collect();
            let (batch, width) = facts.decode_bands.first().map_or((0, 0), |b| (b.0, b.1));
            println!(
                "ragged         ON (C2): row bands {rows:?} at batch {batch} x max width {width}; \
                 per slot prod's acceptance cap (alpha {} margin {} min {} min_batch {}) under \
                 the {ceiling} ceiling, first step 1 row, the step in the smallest band >= its \
                 rows (a uniform step in its exact rectangle); no width floor",
                wp.prod.alpha, wp.prod.margin, wp.prod.min_width, wp.prod.min_batch,
            );
        }
        _ => println!(
            "ragged         OFF: rectangular widths (every slot at its bucket's width), the \
             default since C3 (`--ragged-widths` opts in)"
        ),
    }
}

/// The scheduler the real engine gets.
///
/// The buckets are the engine's own decode batches and prefill shapes, and the KV / GDN geometry is
/// the engine's **measured** one, so the scheduler's arena reservations are the pages the engine
/// actually allocated rather than a second guess at their size. That is what makes the byte ledger one
/// view: the engine bills its arenas, weights and index blob; the scheduler bills the KV pool and the
/// state slots it hands out.
fn scheduler_config(
    forward: &HipForward,
    ctx_band: u32,
    width_rule: &WidthRule,
    ragged_ceiling: saber_sched::RaggedCeiling,
    mixed: saber_sched::MixedPrefill,
    fill_above: u32,
) -> anyhow::Result<SchedulerConfig> {
    let f = forward.facts();
    if f.kv_block_bytes == 0 || f.gdn_bytes_per_slot == 0 {
        bail!("the engine reported no KV or GDN geometry; the scheduler cannot size its arenas");
    }
    let kv = KvGeometry {
        block_tokens: 16,
        // The PER-LAYER figure, and it has to stay that way: the scheduler's whole geometry is
        // expressed in these units, and `KvGeometry::block_bytes()` divides by it to decide how many
        // PAGES a sequence needs. Substituting the arena stride here (ws-k13's first cut) made every
        // sequence get a sixteenth of its pages, and the engine gate caught it as
        // `concurrency 1: stream 0 differs from its solo ids at token 4` — a wrong answer, not a
        // refusal. The census reports the true bytes separately, from `kv_block_stride`.
        bytes_per_token_per_rank: f.kv_block_bytes / 16,
    };
    let mut buckets: Vec<Bucket> = Vec::new();
    // The decode buckets come from the engine's (batch, width) PAIRS, not from its row counts: a
    // bucket IS (batch, spec_width), and `b5s1` and `b1s5` both run five rows. Rebuilding them from
    // `declared`'s row count would tell the scheduler every shape is SPEC=0 and the width controller
    // would have nothing to move between.
    for (batch, width) in &f.decode_shapes {
        // P27-15f: the tree arm's b1t10 form is the ENGINE's (every b1 step under the arm runs
        // it whatever width the scheduler picked); a width past the engine's chain widths is
        // not the scheduler's to pick, and counting it sized a sequence at 11 state slots
        // against an arena of `max_width + 1` -- 0 sessions admitted at --max-seqs 1.
        if *width > f.max_width {
            continue;
        }
        buckets.push(Bucket::decode(*batch, *width, ctx_band, true));
    }
    // The ROW BANDS (C2), from what the engine PLANNED rather than from the flag: a band the
    // engine did not capture is a shape the scheduler must never pick.
    for (batch, width, rows) in &f.decode_bands {
        buckets.push(Bucket::band(*batch, *width, *rows, ctx_band, true));
    }
    for (shape, _) in &f.declared {
        if shape.phase == Phase::Prefill
            && !buckets
                .iter()
                .any(|b| b.phase == Phase::Prefill && b.tokens == shape.rows)
        {
            buckets.push(Bucket::prefill(shape.rows, ctx_band, true));
        }
    }
    buckets.dedup();
    SchedulerConfig::for_mock(
        buckets,
        1,
        f.gdn_bytes_per_slot,
        kv,
        ArenaBytes {
            kv_per_rank: u64::from(f.kv_pool_blocks) * kv.block_bytes(),
            // `max_seqs` sequences at the scheduler's own slots-per-sequence, which is
            // `max_spec_width + 1` — the widest bucket's width plus the bonus slot a verified step
            // budgets even when it drafts nothing. At width 1 that is 2, which is why this was `* 2`
            // before speculation; sizing it at one slot per sequence let the scheduler admit only
            // half of `max_seqs` and the widest captured bucket was never chosen (ws-l12).
            //
            // The ENGINE spaces each sequence's snapshots by `max_width` and reserves the null slot,
            // so it holds `max_seqs * max_width + 1`. This number is one slot per sequence larger,
            // deliberately: it is the scheduler's admission arithmetic and it must not admit a
            // sequence the engine has no run of snapshots for.
            gdn_state_per_rank: u64::from(f.max_seqs)
                * u64::from(f.max_width + 1)
                * f.gdn_bytes_per_slot,
        },
        f.max_ctx as usize,
        format!("engine:real:tp{}:{}", f.tp, f.weights),
    )
    .map(|c| {
        // The width policy is the SCHEDULER's, so it is set here rather than derived from the
        // engine's facts: the engine's `decode_widths` say which shapes exist, and the rule says
        // which of them a step of a given batch runs (ws-t2).
        let width = c.width.clone().with_rule(width_rule.clone()).with_ragged(
            // RAGGED whenever the engine planned bands (the default); the rectangles alone under
            // `--rectangular-widths`, which plans none.
            (!f.decode_bands.is_empty()).then_some(ragged_ceiling),
        );
        // C7: the fill rule over the engine's own rungs (0 = the padded law).
        let prefill = c.prefill.clone().with_fill_above(fill_above);
        if fill_above > 0 {
            println!(
                "prefillfill: rungs above {fill_above} row(s) are fill-only (C7): 4000 -> {:?}, \
                 6000 -> {:?}, 32000 -> {:?}",
                prefill.cut(4000),
                prefill.cut(6000),
                prefill.cut(32000)
            );
        }
        // E4-3: a tree form wider than the chain's widths (the engine's own (rows, rows) shape,
        // skipped from the buckets above) writes its rows past the chain's speculative tail.
        let tree_reserve = f
            .decode_shapes
            .iter()
            .filter(|(b, w)| *b == 1 && *w > f.max_width)
            .map(|(_, w)| w.saturating_sub(1))
            .max()
            .unwrap_or(0);
        SchedulerConfig {
            width,
            mixed,
            prefill,
            tree_reserve,
            ..c
        }
    })
    .context("the scheduler configuration for the real engine")
}

// =====================================================================================
// `saber engine` — the structural gate and the ms/step numbers
// =====================================================================================

/// Arguments of `saber engine`.
pub struct GateArgs {
    /// How to build the engine.
    pub engine: EngineArgs,
    /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`.
    pub prompt_ids: PathBuf,
    /// Row of the corpus to take for the first prompt; the batched gate takes the next ones too.
    pub prompt_index: usize,
    /// Truncate each prompt to this many tokens (0 = the whole prompt).
    pub tokens: u32,
    /// Prod's answer to each corpus row: `{id, n_prompt, token_id, text, logprob, top5}` per line.
    ///
    /// This is what makes the identity arms mean something. Without it the gate compares the engine
    /// with itself, which a broken pass satisfies perfectly — and did: every arm ran a prompt
    /// TRUNCATED to `--tokens` (25 of 234, ending mid-sentence at "give the final answer on a new")
    /// and compared one degenerate continuation with another.
    pub reference: Option<PathBuf>,
    /// Least top-1-vs-top-2 margin, in nats, a prompt may have and still be used.
    ///
    /// An id comparison on a prompt whose top two logits are equal is measuring the tie-break, not
    /// the model. Prod's own margins on this corpus run from 13.12 nats down to **0.00** (rows 78,
    /// 140 and 153 are exactly tied), so this is not a hypothetical.
    pub min_margin: f64,
    /// Decode steps for the batch-1 comparator and the identity checks.
    pub gen: usize,
    /// Decode steps per batch in the batch sweep.
    pub batch_gen: usize,
    /// Timed steps discarded at the front.
    pub warmup: usize,
    /// Sequences in the batched identity gate.
    pub gate_seqs: usize,
    /// Concurrency levels to drive through the scheduler, over HTTP-shaped submissions.
    pub conc: Vec<usize>,
    /// Tokens each concurrent stream asks for.
    pub conc_tokens: usize,
    /// A transcript of CHAT TURNS to run instead of the corpus: one JSON object per line with a
    /// `messages` array, which is what `multiturn_continuity.py` writes.
    ///
    /// The four turns that gate fails on cannot be rebuilt from any corpus — turn 2 carries the
    /// model's own turn-1 reply — so this is the only way to put the failing REQUEST itself in
    /// front of the probes. Rows whose transcript says `same: false` are taken first.
    pub messages: Option<PathBuf>,
    /// The tokenizer `--messages` renders through. The serve plane's own.
    pub tokenizer: Option<PathBuf>,
    /// The chat template `--messages` renders through. The serve plane's own.
    pub chat_template: Option<PathBuf>,
    /// Tokens the CONTINUITY arms ask their re-issued request for; 0 keeps the historic bound.
    ///
    /// The arm has always asked for `conc_tokens.clamp(4, 16)`, and that is why it has never seen
    /// the four multi-turn turns that differ at tokens 2, 5, 11 and **29** (ws-l15). Raising it is
    /// what puts that failure inside this gate, where the probes bisect it.
    pub continuity_tokens: usize,
    /// Branch counts to run the SEQUENCE-GROUP arm at, e.g. `[2, 4]`. Empty skips it.
    ///
    /// Not on by default, so `run_engine.sh`'s arms are the arms they were: `n > 1` is a request
    /// shape and this gate's other arms are all `n = 1`.
    pub fork_n: Vec<usize>,
    /// Tokens each branch of the fork arm asks for.
    pub fork_tokens: usize,
    /// Write every sampled row's seam, parameters, noise key and token here as JSONL, so
    /// `saber sample-check` can replay the fork's draws against the recorded keys.
    pub sample_dump: Option<PathBuf>,
    /// ACCEPTANCE: corpus prompts to drive through the pass with the run's own drafter. 0 skips.
    ///
    /// The arm exists because the ids cannot see the thing it measures. A pass where only row 0
    /// answers correctly commits the SPEC=0 token every step -- perfect ids, acceptance zero --
    /// which is the one failure `docs/LEDGER.md`'s ws-l14 names as invisible. So the number that
    /// matters for a REAL drafter is accepted tokens per draft, and it is measured over the corpus
    /// rather than over one prompt because acceptance is a property of the text.
    pub accept_prompts: usize,
    /// Tokens each acceptance prompt commits.
    pub accept_tokens: usize,
    /// Tokens each acceptance prompt is TRUNCATED to for the second, attribution pass. 0 skips it.
    pub accept_short: usize,
    /// Prompts the CONTROL arm re-runs with the prefill's own proposal off (ws-s9). 0 skips it.
    pub accept_control: usize,
    /// Where the acceptance pass writes its COMMITTED ids (ws-f36). `None` skips it.
    pub accept_ids: Option<PathBuf>,
    /// P27-10 (diagnostic): the acceptance pass's drafter lattices + committed ids go here.
    pub draft_lattice_dump: Option<PathBuf>,
    /// H46 (diagnostic): end the gate right after the acceptance pass.
    pub stop_after_accept: bool,
    /// Where the solo arm's greedy ids go, one JSON array (ws-h43, diagnostic). `None` skips it.
    pub solo_ids_out: Option<PathBuf>,
    /// Run ONLY the solo-prompt op-deep probe and write it here (H44, diagnostic).
    pub solo_probe_out: Option<PathBuf>,
    /// Steps the solo probe decodes.
    pub solo_probe_steps: usize,
    /// The solo probe step whose op inputs/outputs are kept raw (H44).
    pub solo_probe_raw_step: Option<usize>,
    /// Only points whose op name contains this are written raw.
    pub solo_probe_raw_op: String,
    /// Corpus indices whose FIRST proposal — the block the last prefill chunk made — is printed
    /// as ids, for a diff against prod's own (ws-s9). Empty skips the arm.
    pub first_block_ids: String,
    /// Skip ws-p9's bucket attribution of a red batched/concurrency arm (the CONTROL for what it
    /// costs the arms after it — see the flag's own documentation and ws-t3).
    pub no_attribute: bool,
    /// Run ONLY the chunk bisect, at this chunk size; 0 runs the gate.
    ///
    /// ws-h28's open item: the serve path feeds a long prompt in 2048-row chunks and `saber
    /// forward` feeds it as one bucket, and on prompt 8 (14,301 tokens) the two answer `BALL` and
    /// `CELL` — 3.75 nats apart in prod's own reference, which is too wide to call rounding. The
    /// arm runs both and names the first CHUNK and LAYER whose residual moves.
    pub chunk_bisect: u32,
}

/// One sequence driven directly through the pass.
struct Driven {
    seq: SeqId,
    ids: Vec<TokenId>,
    /// Tokens the previous step committed for this sequence: `naccept`, which picks the recurrent
    /// snapshot the next step continues from. 1 after a prompt.
    committed_before: u32,
    /// Draft tokens accepted over this sequence's whole run, for the tokens/step report.
    accepted: u64,
    /// Tokens resident PLUS the one the next step feeds — so `context - 1` is what the engine's
    /// accounting expects, which is where the off-by-one lives and is named.
    context: u32,
    /// **The block the LAST PREFILL CHUNK proposed** (ws-s9), waiting to be offered.
    ///
    /// Prod's `execute_model` calls `speculator.propose(...)` after every target step, so its
    /// first decode step already verifies a block. `RealExecutor` offers a prefill's proposal the
    /// same way it offers a decode step's; this is the direct-drive path's half of that, and it
    /// is carried on the `Driven` rather than offered inside `admit_and_prefill` because that
    /// helper has no drafter and nineteen of its twenty callers have none either. Taken once, by
    /// [`offer_prefill_block`].
    first_draft: Vec<TokenId>,
}

/// Offer each sequence's prefill proposal to the drafter, once.
///
/// Every arm that drives the engine directly goes through one of the decode helpers, so this is
/// called there rather than at twenty call sites — and it CLEARS what it offers, because a mailbox
/// handed the same block twice would draft from a history the model has moved past
/// (`EngineDrafter`'s own law). A mock drafter ignores the offer, which is the honest answer for a
/// drafter that proposes out of its own state.
fn offer_prefill_block(driven: &mut [Driven], drafter: &mut dyn Drafter) {
    for d in driven.iter_mut() {
        if !d.first_draft.is_empty() {
            let block = std::mem::take(&mut d.first_draft);
            drafter.offer(d.seq, &block);
        }
    }
}

/// How many corpus rows the gate walks for (ws-p13): its own `gate_seqs`, or — when a concurrency
/// arm that will run (`c <= max_seqs`) is wider — one DISTINCT prompt per stream of the widest.
/// A transcript run keeps its own turns and walks for `gate_seqs` only.
fn prompt_rows_wanted(
    gate_seqs: usize,
    conc: &[usize],
    max_seqs: usize,
    transcript: bool,
) -> usize {
    let widest = if transcript {
        0
    } else {
        conc.iter()
            .copied()
            .filter(|c| *c <= max_seqs)
            .max()
            .unwrap_or(0)
    };
    gate_seqs.max(1).max(widest)
}

/// The batch a concurrency red is attributed in, and the subject's index in it (ws-p13): the
/// gate's own prompts when the red's prompt is one of them, otherwise the gate's prompts with the
/// red's prompt in the slot its stream index cycles to, so the batch keeps the gate's size.
fn attribution_subject(
    gate: &[Vec<TokenId>],
    conc: &[Vec<TokenId>],
    stream: usize,
) -> (Vec<Vec<TokenId>>, usize) {
    if gate.is_empty() {
        return (conc.to_vec(), stream);
    }
    if stream < gate.len() || conc.get(stream).is_none() {
        return (gate.to_vec(), stream % gate.len());
    }
    let mut set = gate.to_vec();
    let at = stream % gate.len();
    set[at] = conc[stream].clone();
    (set, at)
}

/// A page allocator as simple as it can be: sequence `i` gets a contiguous run of its own.
fn blocks_for(i: usize, per_seq: u32) -> Vec<u32> {
    let base = i as u32 * per_seq;
    (base..base + per_seq).collect()
}

/// Pages each of `live` sequences held AT ONCE is given, out of a pool of `pool_blocks` (H44).
///
/// `per_seq` is sized for the longest run one sequence drives, and its pool clamp divides by
/// `gate_seqs.max(4)` -- but the batch sweep holds `b` sequences at once, up to `max_seqs`. At TP=2
/// the derived pool (57,697 blocks) holds 8 x 257 pages and nothing moved; at TP=1 it is 1,371
/// blocks, so the sweep's b = 8 handed `seq305` (i = 5) pages 1,285..1,541 and every TP=1
/// `run_engine.sh` exited rc=1 with "admitting seq305 ... given block 1371, which is outside the
/// engine's pool of 1371 blocks" after the gate lines, hiding every arm after it. An arm that
/// holds `live` sequences takes `per_seq` when the pool holds that many runs of it (TP=2: the same
/// pages as before), otherwise its equal share of the pool; `need` is the pages its own run
/// reaches, and a share below it is refused by name rather than admitted.
fn pages_for_live(live: usize, per_seq: u32, pool_blocks: u32, need: u32) -> Result<u32, String> {
    let live = live.max(1) as u32;
    let share = if per_seq.saturating_mul(live) <= pool_blocks {
        per_seq
    } else {
        pool_blocks / live
    };
    if share == 0 || share < need.min(per_seq) {
        return Err(format!(
            "{live} sequence(s) held at once need {need} page(s) each and the pool of \
             {pool_blocks} block(s) gives {share}"
        ));
    }
    Ok(share)
}

/// Admit a sequence, feed it its prompt IN CHUNKS, and take the first token.
///
/// `chunk_rows` is the scheduler's own rule — `PrefillShapes::chunk_for` takes
/// `remaining.min(max_chunk)`, so a prompt is fed in pieces of the LARGEST planned prefill shape —
/// and every direct-drive arm here follows it rather than handing the whole prompt to one step.
/// At the default `--prefill-rows 256,2048` that is one chunk for every corpus prompt and nothing
/// moves; at a single 256-row bucket (what `multiturn.sh` boots, because the prefill is wrong
/// above 512 rows) a 258-token prompt is two chunks, `done_before > 0` on the second — which is
/// the shape the four failing continuity turns have and the one this gate could not run at all
/// before ws-l16 ("no planned prefill shape holds this chunk").
fn admit_and_prefill(
    forward: &mut HipForward,
    seq: SeqId,
    prompt: &[TokenId],
    blocks: &[u32],
    chunk_rows: u32,
) -> anyhow::Result<Driven> {
    forward
        .admit(seq, prompt.len() as u32, blocks)
        .with_context(|| format!("admitting {seq}"))?;
    let chunk = (chunk_rows.max(1) as usize).min(prompt.len().max(1));
    let mut done: u32 = 0;
    let mut first: Option<TokenId> = None;
    // THE LAST CHUNK'S OWN PROPOSAL (ws-s9). A pass with a resident drafter proposes out of the
    // prefill too — prod's does, after every target step — and the block comes back with the step,
    // exactly as a decode step's does. It is CARRIED, not offered: the drafter belongs to the arm.
    let mut first_draft = Vec::new();
    for (i, part) in prompt.chunks(chunk).enumerate() {
        let last = (i + 1) * chunk >= prompt.len();
        let mut out = forward.prefill(seq, part, done, last, &RowRequest::greedy())?;
        // ws-f37: the proposal runs after the commit; a greedy driver hands back the argmax.
        forward.propose_greedy(&mut out)?;
        done += part.len() as u32;
        if last {
            first_draft = out.drafts.drain(..).next().unwrap_or_default();
            first = Some(
                out.token()
                    .context("the last prefill chunk sampled nothing")?,
            );
        }
    }
    Ok(Driven {
        seq,
        ids: vec![first.context("a prompt with no last chunk")?],
        committed_before: 1,
        accepted: 0,
        context: prompt.len() as u32 + 1,
        first_draft,
    })
}

/// What one decode step DRAFTED, for a comparison between two paths through the same program.
///
/// The ids a run returns are not enough to compare two paths once a drafter is resident: a step
/// commits between 1 and `width` tokens, so two paths that agree on every token return lists of
/// different lengths the moment their acceptance differs, and a list comparison then names an
/// index at which both lists hold the SAME id (S4's open item 3). What discriminates is the
/// proposal itself and the counters it moved, per step — which is what this carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DraftStep {
    /// The block the ENGINE proposed for each live slot, in slot order. Empty for a slot whose
    /// proposal was refused, and empty for every slot when no drafter is resident.
    proposed: Vec<Vec<TokenId>>,
    /// What each live slot committed on this step.
    committed: Vec<Vec<TokenId>>,
    /// Draft tokens the acceptance law took, per slot.
    accepted: Vec<u32>,
    /// The engine's own drafter counters, as the DELTA this step moved them by.
    counts: saber_engine::DraftCounts,
}

/// The counters one step moved, as a difference of two snapshots.
fn counts_delta(
    a: saber_engine::DraftCounts,
    b: saber_engine::DraftCounts,
) -> saber_engine::DraftCounts {
    saber_engine::DraftCounts {
        proposals: b.proposals - a.proposals,
        tokens: b.tokens - a.tokens,
        eager_launches: b.eager_launches - a.eager_launches,
        replayed_nodes: b.replayed_nodes - a.replayed_nodes,
        refused: b.refused - a.refused,
        out_of_range: b.out_of_range - a.out_of_range,
        accepted: b.accepted - a.accepted,
        verified: b.verified - a.verified,
        pages_cleared: b.pages_cleared - a.pages_cleared,
        pages_dirty: b.pages_dirty - a.pages_dirty,
        ctx_tiles: b.ctx_tiles - a.ctx_tiles,
        ctx_rows: b.ctx_rows - a.ctx_rows,
        ctx_wide_tiles: b.ctx_wide_tiles - a.ctx_wide_tiles,
        ctx_wide_rows: b.ctx_wide_rows - a.ctx_wide_rows,
        store_deferred_chunks: b.store_deferred_chunks - a.store_deferred_chunks,
        store_deferred_ran: b.store_deferred_ran - a.store_deferred_ran,
        store_checked: b.store_checked - a.store_checked,
        store_check_rows: b.store_check_rows - a.store_check_rows,
        store_check_bytes: b.store_check_bytes - a.store_check_bytes,
        store_check_diff: b.store_check_diff - a.store_check_diff,
        store_check_blind: b.store_check_blind - a.store_check_blind,
        store_check_class: b.store_check_class - a.store_check_class,
        store_check_class_diff: b.store_check_class_diff - a.store_check_class_diff,
        store_check_class_silent: b.store_check_class_silent - a.store_check_class_silent,
        prefill_proposals: b.prefill_proposals - a.prefill_proposals,
        prefill_silent: b.prefill_silent - a.prefill_silent,
        batch_checked: b.batch_checked - a.batch_checked,
        batch_mismatch: b.batch_mismatch - a.batch_mismatch,
        batch_cross: b.batch_cross - a.batch_cross,
        batch_cross_same: b.batch_cross_same - a.batch_cross_same,
        batch_us: b.batch_us - a.batch_us,
        solo_us: b.solo_us - a.solo_us,
        batch_timed: b.batch_timed - a.batch_timed,
        in_step: b.in_step - a.in_step,
        after_commit: b.after_commit - a.after_commit,
        uncollected: b.uncollected - a.uncollected,
        fork_sessions: b.fork_sessions - a.fork_sessions,
        fork_pages: b.fork_pages - a.fork_pages,
        fork_proposals: b.fork_proposals - a.fork_proposals,
        selector_reranked: b.selector_reranked - a.selector_reranked,
        selector_moved: b.selector_moved - a.selector_moved,
        spectree: b.spectree.since(&a.spectree),
    }
}

/// What a comparison of two id lists from the SAME NUMBER OF STEPS can honestly say.
///
/// S4's open item 3. Under speculation a step commits a variable number of tokens, so a fixed
/// step count produces a variable id count and `a == b` is the wrong question. The right one is:
/// do they agree on the COMMON PREFIX, and how long is each? A length difference is an
/// ACCEPTANCE difference and is reported as one — with the per-step draft trace beside it, which
/// is the thing that can attribute it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefixVerdict {
    /// Ids compared: the shorter of the two lengths.
    common: usize,
    /// The first index at which the two prefixes differ, if any.
    first_diff: Option<usize>,
    len_a: usize,
    len_b: usize,
}

impl PrefixVerdict {
    fn of(a: &[TokenId], b: &[TokenId]) -> PrefixVerdict {
        let common = a.len().min(b.len());
        PrefixVerdict {
            common,
            first_diff: a[..common]
                .iter()
                .zip(&b[..common])
                .position(|(x, y)| x != y),
            len_a: a.len(),
            len_b: b.len(),
        }
    }

    /// Both paths produced the same tokens for as far as both went.
    fn prefix_agrees(&self) -> bool {
        self.first_diff.is_none()
    }

    /// ...and they went the same distance, which is the acceptance half of the claim.
    fn lengths_agree(&self) -> bool {
        self.len_a == self.len_b
    }

    /// The line a gate prints, whichever way it went: the number compared and BOTH lengths.
    fn line(&self) -> String {
        format!(
            "{} id(s) compared (the common prefix of {} and {})",
            self.common, self.len_a, self.len_b
        )
    }
}

/// Two paths' per-step proposals and counters, held against each other.
///
/// **The instrument S4's open item 4 asked for.** The claim "these two paths run the same program"
/// is made of two halves and only one of them is in the ids: the TOKENS the model committed, and
/// the BLOCK the drafter proposed for the next step. The second one is what decides how many of
/// the first each step commits, so a difference there shows up in an id list only as a length —
/// three hundred steps later, at an index where both lists hold the same token. This compares the
/// blocks directly, step by step, and names the first step at which they part company.
///
/// Returns the number of failures, and prints one line either way.
fn draft_trace_verdict(
    a: &[DraftStep],
    b: &[DraftStep],
    name_a: &str,
    name_b: &str,
    fail: &impl Fn(&str),
) -> usize {
    if a.is_empty() && b.is_empty() {
        println!(
            "draft     no proposals on either path: this run has no resident drafter, so the \
             draft trace is empty on both sides and says nothing"
        );
        return 0;
    }
    if a.len() != b.len() {
        fail(&format!(
            "the {name_a} path ran {} decode step(s) and the {name_b} path ran {}: the draft \
             traces are not comparable",
            a.len(),
            b.len()
        ));
        return 1;
    }
    let block_diff = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.proposed != y.proposed)
        .map(|i| (i, &a[i], &b[i]));
    let count_diff = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.counts != y.counts)
        .map(|i| (i, &a[i], &b[i]));
    let commit_diff = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.committed != y.committed)
        .map(|i| (i, &a[i], &b[i]));
    let mut failures = 0usize;
    if let Some((i, x, y)) = block_diff {
        // THE FIRST DIFFERING BLOCK NAMES THE CAUSE. A proposal is a function of the target's
        // tapped hidden states for the rows this step ran, this session's drafter pages, and the
        // index tables the acceptance produced — so a block that differs on step `i` while every
        // step before it agreed says the divergence is in that step and not in the history.
        fail(&format!(
            "the {name_a} and {name_b} paths proposed DIFFERENT blocks at step {i}: {name_a} \
             {:?}, {name_b} {:?} (the committed tokens up to there: {} vs {})",
            x.proposed,
            y.proposed,
            a[..i].iter().map(|s| s.counts.tokens).sum::<u64>(),
            b[..i].iter().map(|s| s.counts.tokens).sum::<u64>(),
        ));
        failures += 1;
    }
    if let Some((i, x, y)) = commit_diff {
        fail(&format!(
            "the {name_a} and {name_b} paths COMMITTED differently at step {i}: {name_a} {:?}, \
             {name_b} {:?}",
            x.committed, y.committed
        ));
        failures += 1;
    }
    if let Some((i, x, y)) = count_diff {
        fail(&format!(
            "the {name_a} and {name_b} paths moved DIFFERENT drafter counters at step {i}: \
             {name_a} [{}], {name_b} [{}]",
            x.counts.line(),
            y.counts.line()
        ));
        failures += 1;
    }
    if failures == 0 {
        let total = |t: &[DraftStep]| -> (u64, u64, u64) {
            (
                t.iter().map(|s| s.counts.proposals).sum(),
                t.iter().map(|s| s.counts.accepted).sum(),
                t.iter().map(|s| s.counts.verified).sum(),
            )
        };
        let (p, ac, v) = total(a);
        println!(
            "OK        {} step(s): the {name_a} and {name_b} paths proposed the SAME block, \
             committed the same tokens and moved the same drafter counters at every one \
             ({p} proposal(s), {ac} accepted of {v} verified on each)",
            a.len()
        );
    }
    failures
}

/// `steps` decode steps over every sequence in `driven`, at the bucket's batch. Returns the wall
/// clock of each step.
fn decode_together(
    forward: &mut HipForward,
    driven: &mut [Driven],
    batch: u32,
    width: u32,
    drafter: &mut dyn Drafter,
    steps: usize,
) -> anyhow::Result<Vec<f64>> {
    decode_traced(
        forward, driven, batch, width, drafter, steps, None, None, None,
    )
}

/// [`decode_together`], with every step's proposal and counter delta recorded.
///
/// One function rather than two so that a traced arm and an untraced one cannot drift into driving
/// the engine differently — which is exactly how the graph arm and the eager arm came to run two
/// different drafters (S4's open item 4).
#[allow(clippy::too_many_arguments)]
fn decode_traced(
    forward: &mut HipForward,
    driven: &mut [Driven],
    batch: u32,
    width: u32,
    drafter: &mut dyn Drafter,
    steps: usize,
    mut trace: Option<&mut Vec<DraftStep>>,
    mut margins: Option<&mut Vec<Vec<RowMargin>>>,
    // THE GDN-SIDE CONTROL (ws-p9): the slot whose `committed_before` this drive deliberately
    // gets wrong. `committed_before` is `naccept`, which picks the recurrent snapshot the step
    // continues from and the conv window it reads, so a wrong one in range is a state law read
    // wrongly — silent, the shape ws-l17's `cache_idx` binding had — and the arm's law must turn
    // red on it or it has no teeth. Never set by anything but the control.
    gdn_defect: Option<usize>,
) -> anyhow::Result<Vec<f64>> {
    let mut per_step = Vec::with_capacity(steps);
    offer_prefill_block(driven, drafter);
    // THE MARGIN IS A ROW REQUEST, and asking for one does not change what the step commits: the
    // device computes the argmax either way and the seam kernel — a node of the captured graph
    // whatever the row asks for — writes the top-K beside it. The direct-drive arms never sample
    // on the host, so a temperature here buys the distribution and nothing else. It is off in
    // every timed arm all the same, because it costs a per-step host write and a read-back
    // (ws-p9).
    let sampling = if margins.is_some() {
        RowRequest {
            temperature: Some(1.0),
            penalty_ids: Vec::new(),
        }
    } else {
        RowRequest::greedy()
    };
    if let Some(m) = margins.as_mut() {
        m.clear();
        m.resize(driven.len(), Vec::new());
    }
    for _ in 0..steps {
        let slots: Vec<DecodeSlot> = driven
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let input = *d.ids.last().expect("a prefilled sequence has a token");
                let mut draft = drafter.propose(d.seq, input, width - 1);
                // The bucket's width is the shape; a short proposal is padded with a token the model
                // cannot want, and rejected at the first of them.
                draft.resize((width - 1) as usize, RecordedDrafter::WRONG);
                DecodeSlot {
                    seq: d.seq,
                    input,
                    context_len: d.context - 1,
                    draft,
                    committed_before: if gdn_defect == Some(i) {
                        width
                    } else {
                        d.committed_before
                    },
                    sampling: sampling.clone(),
                    decider: None,
                }
            })
            .collect();
        let before = trace.as_ref().map(|_| forward.draft_counts());
        let t0 = Instant::now();
        // The RUN's width, not 1. A run plans exactly one verify width (the GDN conv cache's layout
        // is a function of it), so every arm of the gate — solo, batched, the sweep, the serve path —
        // runs at the same one, and `b1s5` against `b1s1` is two RUNS compared pairwise.
        let mut out = forward.decode(&slots, batch, width)?;
        // ws-f37: THE PROPOSAL AFTER THE COMMIT, inside the timed window — it is the same work
        // the step used to do at its end, so the ms/step it reports is still the whole step.
        forward.propose_greedy(&mut out)?;
        per_step.push(t0.elapsed().as_secs_f64() * 1000.0);
        if out.commits.len() != driven.len() {
            bail!(
                "a batch-{batch} step returned {} commit(s) for {} live sequence(s)",
                out.commits.len(),
                driven.len()
            );
        }
        // WHAT THE PASS ITSELF PROPOSED, offered in slot order. This is the direct-drive twin of
        // `RealExecutor`'s own offer: a run with a RESIDENT drafter makes its proposal inside the
        // step that produced the hidden states, and `EngineDrafter` is the mailbox that carries it
        // to the next one. Every mock ignores the offer, which is the honest answer for a drafter
        // that proposes out of its own state.
        let drafts = std::mem::take(&mut out.drafts);
        for (d, block) in driven.iter().zip(&drafts) {
            drafter.offer(d.seq, block);
        }
        if let (Some(t), Some(b0)) = (trace.as_mut(), before) {
            t.push(DraftStep {
                proposed: drafts.clone(),
                committed: out.commits.iter().map(|c| c.tokens.clone()).collect(),
                accepted: out.commits.iter().map(|c| c.accepted).collect(),
                counts: counts_delta(b0, forward.draft_counts()),
            });
        }
        for (i, (d, c)) in driven.iter_mut().zip(out.commits).enumerate() {
            drafter.commit(d.seq, &c.tokens);
            // ONE MARGIN PER COMMITTED TOKEN, in commit order — the same list `SlotCommit::rows`
            // carries, because the seam ran on every verify row and an accepted prefix has a
            // distribution at each of its positions too.
            if let Some(m) = margins.as_mut() {
                for (j, t) in c.tokens.iter().enumerate() {
                    m[i].push(RowMargin {
                        id: *t,
                        top: c
                            .rows
                            .get(j)
                            .and_then(Option::as_ref)
                            .map_or_else(Vec::new, |r| {
                                r.top_ids
                                    .iter()
                                    .copied()
                                    .zip(r.top_values.iter().copied())
                                    .collect()
                            }),
                    });
                }
            }
            d.committed_before = c.tokens.len() as u32;
            d.accepted += u64::from(c.accepted);
            d.ids.extend(&c.tokens);
            d.context += c.tokens.len() as u32;
        }
    }
    Ok(per_step)
}

/// ATTRIBUTE a divergence between two decode buckets, inside this boot (ws-p9).
///
/// The arms that find one — batched, concurrency, mixed arrival — all compare ONE sequence between
/// a batch-1 run and a batch-N one, so this is one question asked once: run the same prompt alone
/// and in the batch, with the head's seam on, and say where the two streams part, which op the two
/// buckets resolved differently there, and how wide the flip was.
///
/// It re-runs rather than instrumenting the arm, because the arm's `ms/step` is a published number
/// and a margin costs a host write and a read-back per step. The re-run reproduces the arm because
/// everything it depends on is deterministic: the same prompts, the same pages, the same drafter,
/// and a slot shift that the slot-shift arm says does not matter.
#[allow(clippy::too_many_arguments)]
fn attribute_buckets(
    forward: &mut HipForward,
    prompts: &[Vec<TokenId>],
    which: usize,
    batch: u32,
    width: u32,
    steps: usize,
    per_seq: u32,
    prefill_chunk: u32,
    drafter: &mut dyn Drafter,
    // The control's slot, or `None` for the case. See `decode_traced`.
    gdn_defect: Option<usize>,
    // WHERE THE SUBJECT SITS IN THE MARGIN RUN'S BATCH (ws-p11). `Subject::ArmSlot` keeps the
    // arm's own order, so the run that decides the verdict is the arm's configuration and not a
    // neighbour of it; `Subject::SlotZero` is the order the probes need. See `Subject`.
    subject: Subject,
) -> anyhow::Result<BucketDivergence> {
    let mut out = BucketDivergence {
        probed: forward.facts().tp == 1,
        ..BucketDivergence::default()
    };
    // The plan facts first: they need no card at all, and they are what the probe's answer is
    // held against.
    let shape = |b: u32| StepRows {
        phase: Phase::Decode,
        rows: b * width,
        width,
        samples: true,
        tree: false,
        ragged: false,
    };
    out.deltas = forward
        .facts()
        .bucket_op_deltas(shape(1), shape(batch.max(1)));

    // THE SUBJECT SITS IN SLOT 0 OF THE BATCH, and that is not cosmetic: it is what makes the
    // subject's rows `0..width` on BOTH sides, so the two probes compare the same tokens.
    //
    // It used to be load-bearing for a second reason that is now gone. An op-deep probe sliced an
    // op's output edge by `bytes / rows`, which is the truth for the residual stream and an
    // assumption for `l{n}.gdn.conv` — one edge holding q, k and v as three blocks — so row 0 was
    // the only index whose span was the same quantity in both buckets. It was not a ROW there
    // either: at TP=1 it is five rows of q, which is why row 0 alone named `gdn.conv_update` as
    // the first op to write differing bytes when the difference was four rows nothing compared.
    // The span now comes from the plan (`saber_exec::RowSpan`) and every one of the subject's
    // rows is read (ws-p9, ws-l18). The batch is the same four prompts either way.
    let order: Vec<usize> = std::iter::once(which)
        .chain((0..prompts.len()).filter(|i| *i != which))
        .collect();
    // THE MARGIN RUN'S ORDER, which is not necessarily the probes' (ws-p11). The verdict rests on
    // the flip this run finds, and a run with the subject moved to slot 0 is a DIFFERENT batch:
    // the batched proposal is not slot-invariant (ws-s10's own check: 10 of 60 requests got a
    // different block batched than alone), so a different slot is a different draft, a different
    // acceptance history and a different verify-row alignment at every later position — and so,
    // at the documented near-ties, different flips. That is why ws-k16's run could not reproduce
    // two of the arm's three reds ("the attribution's own two runs AGREED over 85 token(s)") and
    // measured the third at token 256 instead of 46: it read slot 0's flips and reported them
    // against slot 2's. `Subject::ArmSlot` runs the arm's own batch; the probes keep slot 0.
    let margin_order: Vec<usize> = match subject {
        Subject::ArmSlot => (0..prompts.len()).collect(),
        Subject::SlotZero => order.clone(),
    };
    let subject_slot = margin_order
        .iter()
        .position(|i| *i == which)
        .ok_or_else(|| anyhow::anyhow!("prompt {which} is not in the batch it is attributed in"))?;
    out.subject_slot = subject_slot;

    // ---- the two runs, with the seam on -------------------------------------------------------
    let mut solo_one = vec![admit_and_prefill(
        forward,
        SeqId(960),
        &prompts[which],
        &blocks_for(which, per_seq),
        prefill_chunk,
    )?];
    let mut solo_m: Vec<Vec<RowMargin>> = Vec::new();
    decode_traced(
        forward,
        &mut solo_one,
        1,
        width,
        drafter,
        steps,
        None,
        Some(&mut solo_m),
        // THE SOLO SIDE IS CLEAN. The control injects the defect into the BATCH, which is where
        // the arms compare from: a perturbation on both sides would agree with itself and prove
        // nothing (`docs/GATES.md`: a control that cannot fail is not a control).
        None,
    )?;
    forward.release(solo_one[0].seq);
    let mut many: Vec<Driven> = Vec::new();
    for (n, i) in margin_order.iter().copied().enumerate() {
        many.push(admit_and_prefill(
            forward,
            SeqId(970 + n as u64),
            &prompts[i],
            &blocks_for(i, per_seq),
            prefill_chunk,
        )?);
    }
    let mut batch_m: Vec<Vec<RowMargin>> = Vec::new();
    decode_traced(
        forward,
        &mut many,
        batch.max(1),
        width,
        drafter,
        steps,
        None,
        Some(&mut batch_m),
        gdn_defect,
    )?;
    for d in many.iter().rev() {
        forward.release(d.seq);
    }
    let (a, b) = (&solo_m[0], &batch_m[subject_slot]);
    out.at = a
        .iter()
        .zip(b)
        .position(|(x, y)| x.id != y.id)
        .unwrap_or_else(|| a.len().min(b.len()));
    if let (Some(x), Some(y)) = (a.get(out.at), b.get(out.at)) {
        out.solo_id = Some(x.id);
        out.batch_id = Some(y.id);
        out.solo_gap = x.gap();
        out.batch_gap = y.gap();
        out.solo_to_batch = x.gap_to(y.id);
        out.batch_to_solo = y.gap_to(x.id);
    }
    let lo = out.at.saturating_sub(4);
    out.ties = a.iter().skip(lo).take(9).filter(|r| r.exact_tie()).count()
        + b.iter().skip(lo).take(9).filter(|r| r.exact_tie()).count();

    // ---- and where the two buckets' streams part ----------------------------------------------
    // TP=1 only: at two ranks a dump point is one rank's half of a reduced stream and the engine
    // refuses the probe by name, which is why `verdict` has a two-rank branch.
    if !out.probed {
        return Ok(out);
    }
    let probe_steps = steps.clamp(1, 12);
    let mut one = vec![admit_and_prefill(
        forward,
        SeqId(980),
        &prompts[which],
        &blocks_for(which, per_seq),
        prefill_chunk,
    )?];
    let pa = decode_probed_at(
        forward,
        &mut one,
        1,
        width,
        drafter,
        probe_steps,
        None,
        None,
    )?;
    forward.release(one[0].seq);
    let mut group: Vec<Driven> = Vec::new();
    for (n, i) in order.iter().copied().enumerate() {
        group.push(admit_and_prefill(
            forward,
            SeqId(985 + n as u64),
            &prompts[i],
            &blocks_for(i, per_seq),
            prefill_chunk,
        )?);
    }
    let pb = decode_probed_at(
        forward,
        &mut group,
        batch.max(1),
        width,
        drafter,
        probe_steps,
        None,
        gdn_defect,
    )?;
    for d in group.iter().rev() {
        forward.release(d.seq);
    }
    // THE SUBJECT'S OWN ROWS on both sides: it is slot 0 of the batch (see `order`), so it holds
    // rows `0..width` of each step in both buckets. Comparing row 0 alone is what made the conv
    // update look batch-dependent (`first_probe_divergence_rows`, ws-l18).
    let subject_rows = width.max(1) as usize;
    out.layer_first = first_probe_divergence_rows(&pa, &pb, subject_rows);
    // THE OP-DEEP PASS, only as far as the layer-deep one had to go. It reads every op's own
    // output edge for the two compared rows, which is 700-odd device-to-host reads a step — worth
    // paying once a divergence is in hand, and not worth paying to look for one.
    if let Some((step, _, _, _, _)) = out.layer_first.clone() {
        let deep_steps = (step + 1).min(8);
        let mut one = vec![admit_and_prefill(
            forward,
            SeqId(990),
            &prompts[which],
            &blocks_for(which, per_seq),
            prefill_chunk,
        )?];
        let rows: Vec<usize> = (0..subject_rows).collect();
        let da = decode_probed_at(
            forward,
            &mut one,
            1,
            width,
            drafter,
            deep_steps,
            Some(&rows),
            None,
        )?;
        forward.release(one[0].seq);
        let mut group: Vec<Driven> = Vec::new();
        for (n, i) in order.iter().copied().enumerate() {
            group.push(admit_and_prefill(
                forward,
                SeqId(995 + n as u64),
                &prompts[i],
                &blocks_for(i, per_seq),
                prefill_chunk,
            )?);
        }
        let db = decode_probed_at(
            forward,
            &mut group,
            batch.max(1),
            width,
            drafter,
            deep_steps,
            Some(&rows),
            gdn_defect,
        )?;
        for d in group.iter().rev() {
            forward.release(d.seq);
        }
        // Both sides digested the SAME row list -- the subject's `width` rows -- so index `r` of
        // one is index `r` of the other.
        out.op_first = first_probe_divergence_rows(&da, &db, subject_rows);
    }
    if let Some((_, _, _, op, _)) = out.first() {
        out.op_delta = forward
            .facts()
            .bucket_delta_for_op(shape(1), shape(batch.max(1)), op);
    }
    Ok(out)
}

/// **THE RESIDUE ARM** (ws-h31): the same steps before and after ws-p9's bucket attribution, per
/// HOST phase, so that the thing ws-t3 could only see as "+2.7 ms/step somewhere" has a name.
///
/// ws-t3 paired the whole gate with `--no-attribute` and found that every arm downstream of the
/// attribution in the same boot read +2.65 ms at `b1s8` and +3.9 at `b4s8`, that a FRESH engine in
/// the same process was unaffected, and that the launcher's counters, the dispatch count and every
/// committed id were identical across the pair -- so the cost is host state inside the long-lived
/// `HipForward` and not the card, the clock, the plan or the arena. It left two candidates and no
/// instrument. This is the instrument, localised: two identical direct-drive arms with one thing
/// between them, and the phase table of each.
///
/// **`--no-attribute` is the CONTROL and it must read zero.** With it the middle is a no-op, so
/// the two halves are the same arm twice and every phase's delta is the session's own drift; a
/// control that moved would mean the arm cannot tell an attribution from a step count.
///
/// It runs at the very END of the gate, under `--host-trace` only, so nothing this gate publishes
/// is measured after it -- the ordering law ws-t3 wrote (`docs/GATES.md` blind spot 19) applies to
/// this arm exactly as it applies to the thing it is measuring.
/// One half of the arm at one shape: admit, prefill, RESET the trace, decode, take it.
///
/// The reset is after the prefills because a prefill chunk passes through the same phases a decode
/// step does (the blob, the submit, the wait, the read-back), and a table over both would report
/// `hits` above its own step count and a p50 mixed out of two shapes.
#[allow(clippy::too_many_arguments)]
fn host_trace_half(
    forward: &mut HipForward,
    prompts: &[Vec<TokenId>],
    base: u64,
    batch: u32,
    width: u32,
    steps: usize,
    per_seq: u32,
    prefill_chunk: u32,
    d: &mut dyn Drafter,
) -> anyhow::Result<(HostTraceReport, Vec<f64>)> {
    let mut live = Vec::new();
    for i in 0..batch as usize {
        let p = &prompts[i % prompts.len()];
        live.push(admit_and_prefill(
            forward,
            SeqId(base + i as u64),
            p,
            &blocks_for(i, per_seq),
            prefill_chunk,
        )?);
    }
    forward.host_trace().reset();
    let ms = decode_together(forward, &mut live, batch, width, d, steps)?;
    let rep = forward.host_trace().take();
    for x in &live {
        forward.release(x.seq);
    }
    Ok((rep, ms))
}

#[allow(clippy::too_many_arguments)]
fn host_trace_residue(
    forward: &mut HipForward,
    prompts: &[Vec<TokenId>],
    which: usize,
    batches: &[u32],
    width: u32,
    steps: usize,
    per_seq: u32,
    prefill_chunk: u32,
    drafter: &mut dyn Drafter,
    attribute: bool,
) -> anyhow::Result<String> {
    use std::fmt::Write as _;
    // EVERY SHAPE'S `before` FIRST, then ONE attribution, then every shape's `after`. Running the
    // pair per shape instead would put the second shape's `before` downstream of the first shape's
    // attribution, so the two cells would not be answering the same question -- and ws-t3's table
    // is exactly two cells (`b1s8` +2.65 ms, `b4s8` +3.9) against one diagnostic.
    let mut before = Vec::new();
    for (k, b) in batches.iter().copied().enumerate() {
        before.push((
            b,
            host_trace_half(
                forward,
                prompts,
                700 + (k as u64 * 16),
                b,
                width,
                steps,
                per_seq,
                prefill_chunk,
                drafter,
            )?,
        ));
    }
    let ran = if attribute {
        // The diagnostic itself, exactly as the gate runs it -- its own admissions, its own
        // prefills, the head's sampling seam on, and the wrong-snapshot control left off.
        attribute_buckets(
            forward,
            prompts,
            which,
            batches.iter().copied().max().unwrap_or(1),
            width,
            steps.min(16),
            per_seq,
            prefill_chunk,
            drafter,
            None,
            Subject::ArmSlot,
        )?;
        "ws-p9's bucket attribution"
    } else {
        "NOTHING (--no-attribute: the control)"
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "htrace    RESIDUE ARM: {steps} step(s) per shape per side, and between them {ran}"
    );
    for (k, (b, (rep_a, ms_a))) in before.into_iter().enumerate() {
        let (rep_b, ms_b) = host_trace_half(
            forward,
            prompts,
            760 + (k as u64 * 16),
            b,
            width,
            steps,
            per_seq,
            prefill_chunk,
            drafter,
        )?;
        let _ = writeln!(
            out,
            "htrace    b{b}s{width} the arm's own wall: p50 {:.3} ms/step before, {:.3} after \
             ({:+.3})",
            p50(&ms_a),
            p50(&ms_b),
            p50(&ms_b) - p50(&ms_a)
        );
        out.push_str(&rep_a.table(&format!("residue BEFORE b{b}s{width}")));
        out.push_str(&rep_b.table(&format!("residue AFTER  b{b}s{width}")));
        out.push_str(&rep_b.delta(&rep_a, &format!("residue DELTA b{b}s{width}")));
    }
    Ok(out)
}

/// `steps` PROBED decode steps: the same drive as [`decode_together`], with every row of the
/// residual stream digested at every layer.
///
/// Separate from `decode_together` rather than a flag on it because the probe forces the EAGER
/// walk: a caller must not be able to ask for a bisect and quietly get a different path than the
/// arm it is comparing against. Both probed arms below run this, so both run eager.
fn decode_probed(
    forward: &mut HipForward,
    driven: &mut [Driven],
    batch: u32,
    width: u32,
    drafter: &mut dyn Drafter,
    steps: usize,
) -> anyhow::Result<Vec<Vec<RowProbe>>> {
    decode_probed_at(forward, driven, batch, width, drafter, steps, None, None)
}

/// [`decode_probed`] with a dump point after EVERY OP, digesting only the named rows (ws-p9).
///
/// The bisect one level in: a layer-deep probe says the two buckets parted at `l32.resid.a`, and
/// this says which of that layer's ops wrote the first differing bytes — the op a plan-fact delta
/// is keyed on. It reads 700-odd edges a step, so it names its rows.
/// One driver for both probe depths: a caller must not be able to ask for a bisect and quietly get
/// a different drive than the arm it is comparing against (S4's open item 4).
#[allow(clippy::too_many_arguments)]
fn decode_probed_at(
    forward: &mut HipForward,
    driven: &mut [Driven],
    batch: u32,
    width: u32,
    drafter: &mut dyn Drafter,
    steps: usize,
    // `Some(rows)` digests every OP's output edge for those rows; `None` is one point per layer
    // for every row (ws-p9).
    deep: Option<&[usize]>,
    // The GDN-side control's slot. See `decode_traced`.
    gdn_defect: Option<usize>,
) -> anyhow::Result<Vec<Vec<RowProbe>>> {
    decode_probed_depth(
        forward,
        driven,
        batch,
        width,
        drafter,
        steps,
        deep,
        gdn_defect,
        ProbeDepth::Ops,
    )
}

/// [`decode_probed_at`] with the deep probe's depth named (H44: `OpsRaw` keeps rows raw).
#[allow(clippy::too_many_arguments)]
fn decode_probed_depth(
    forward: &mut HipForward,
    driven: &mut [Driven],
    batch: u32,
    width: u32,
    drafter: &mut dyn Drafter,
    steps: usize,
    deep: Option<&[usize]>,
    gdn_defect: Option<usize>,
    depth: ProbeDepth,
) -> anyhow::Result<Vec<Vec<RowProbe>>> {
    let mut out = Vec::with_capacity(steps);
    offer_prefill_block(driven, drafter);
    for _ in 0..steps {
        let slots: Vec<DecodeSlot> = driven
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let input = *d.ids.last().expect("a prefilled sequence has a token");
                let mut draft = drafter.propose(d.seq, input, width - 1);
                draft.resize((width - 1) as usize, RecordedDrafter::WRONG);
                DecodeSlot {
                    seq: d.seq,
                    input,
                    context_len: d.context - 1,
                    draft,
                    committed_before: if gdn_defect == Some(i) {
                        width
                    } else {
                        d.committed_before
                    },
                    sampling: RowRequest::greedy(),
                    decider: None,
                }
            })
            .collect();
        let (mut step, probe) = match deep {
            Some(rows) => forward.decode_probe_depth(&slots, batch, width, rows, depth)?,
            None => forward.decode_probe(&slots, batch, width)?,
        };
        forward.propose_greedy(&mut step)?;
        if step.commits.len() != driven.len() {
            bail!(
                "a probed batch-{batch} step returned {} commit(s) for {} live sequence(s)",
                step.commits.len(),
                driven.len()
            );
        }
        let drafts = std::mem::take(&mut step.drafts);
        for (d, block) in driven.iter().zip(&drafts) {
            drafter.offer(d.seq, block);
        }
        for (d, c) in driven.iter_mut().zip(step.commits) {
            drafter.commit(d.seq, &c.tokens);
            d.committed_before = c.tokens.len() as u32;
            d.accepted += u64::from(c.accepted);
            d.ids.extend(&c.tokens);
            d.context += c.tokens.len() as u32;
        }
        out.push(probe);
    }
    Ok(out)
}

/// THE AUX TAP GATE: `aux.cat`'s bytes against the engine's own probe of the five edges it reads,
/// with a control that must differ.
///
/// Both arms run a real decode step at the given shape. The CASE issues each copy at its tap point
/// and every tapped edge must come back byte for byte; the CONTROL issues each copy one layer late
/// while the probe stays put, so slot reuse has overwritten the buffer by the time the copy runs
/// and the bytes must differ. A control that also moved the probe would compare the late copy with
/// the late edge and pass — which is the way this control could have been written wrong, and why
/// the two points are separate laws in `saber_engine::aux_tap`.
///
/// Returns `(failures, one line per tap)`.
fn aux_tap_arm(
    forward: &mut HipForward,
    driven: &mut [Driven],
    batch: u32,
    width: u32,
) -> anyhow::Result<(usize, Vec<String>)> {
    let mut lines = Vec::new();
    let mut bad = 0usize;
    let slots_of = |driven: &[Driven]| -> Vec<DecodeSlot> {
        driven
            .iter()
            .map(|d| DecodeSlot {
                seq: d.seq,
                input: *d.ids.last().expect("a prefilled sequence has a token"),
                context_len: d.context - 1,
                draft: vec![RecordedDrafter::WRONG; (width - 1) as usize],
                committed_before: d.committed_before,
                sampling: RowRequest::greedy(),
                decider: None,
            })
            .collect()
    };
    let advance = |driven: &mut [Driven], step: &saber_engine::StepOutcome| {
        for (d, c) in driven.iter_mut().zip(step.commits.iter()) {
            d.committed_before = c.tokens.len() as u32;
            d.ids.extend(&c.tokens);
            d.context += c.tokens.len() as u32;
        }
    };

    let slots = slots_of(driven);
    let (step, case) = forward.aux_tap_step(&slots, batch, width, false)?;
    advance(driven, &step);
    for v in &case {
        let ok = v.exact();
        if !ok {
            bad += 1;
        }
        lines.push(format!(
            "          l{:<3} {:<14} {} of {} bytes",
            v.layer,
            v.edge,
            if ok {
                "identical".to_owned()
            } else {
                format!("DIFFERS in {}", v.differing)
            },
            v.bytes
        ));
    }

    // THE CONTROL, and it has to fire on every tap: the copies go one layer on, the probe does not.
    let slots = slots_of(driven);
    let (step, late) = forward.aux_tap_step(&slots, batch, width, true)?;
    advance(driven, &step);
    let fired = late.iter().filter(|v| !v.exact()).count();
    lines.push(format!(
        "          CONTROL (one layer late): {fired} of {} tap(s) differ",
        late.len()
    ));
    if fired != late.len() {
        bad += 1;
        lines.push(
            "          FAIL      a tap issued one layer late still matched its own edge, so this \
             gate cannot tell where the copy ran"
                .to_owned(),
        );
    }
    Ok((bad, lines))
}

/// A whole request through the PASS: admit on `blocks`, feed the prompt in `chunk_rows`-row chunks
/// the way the scheduler does, decode `steps` times, release.
///
/// **The chunking is the point.** Every other direct-drive arm in this file hands the whole prompt
/// to one `prefill` call, so a 258-token prompt runs in the 2048-row bucket and `done_before` is 0
/// at every step the gate takes. The serve plane chunks into the smallest planned prefill shape,
/// and the four continuity turns that fail (ws-l15) are all turns whose prompt takes more than one
/// chunk -- so an arm that never chunks cannot reproduce them whatever else it does.
///
/// `probe` digests the residual stream at every layer: one entry per prefill chunk (at that
/// chunk's last live row) followed by one per decode step (every row). The decode half is then
/// EAGER, as `decode_probed` requires.
#[allow(clippy::too_many_arguments)]
fn request_through_pass(
    forward: &mut HipForward,
    seq: SeqId,
    prompt: &[TokenId],
    blocks: &[u32],
    chunk_rows: u32,
    steps: usize,
    width: u32,
    drafter: &mut dyn Drafter,
    probe: bool,
) -> anyhow::Result<(Vec<TokenId>, Vec<Vec<RowProbe>>)> {
    forward
        .admit(seq, prompt.len() as u32, blocks)
        .with_context(|| format!("admitting {seq}"))?;
    let mut probes: Vec<Vec<RowProbe>> = Vec::new();
    let chunk = chunk_rows.max(1) as usize;
    let mut done: u32 = 0;
    let mut first: Option<TokenId> = None;
    // The last chunk's own proposal, carried the way `admit_and_prefill` carries it (ws-s9).
    let mut first_draft = Vec::new();
    for (i, part) in prompt.chunks(chunk).enumerate() {
        let last = (i + 1) * chunk >= prompt.len();
        let mut out = if probe {
            let (out, p) = forward.prefill_probe(seq, part, done, last, None)?;
            probes.push(p);
            out
        } else {
            forward.prefill(seq, part, done, last, &RowRequest::greedy())?
        };
        forward.propose_greedy(&mut out)?;
        done += part.len() as u32;
        if last {
            first_draft = out.drafts.drain(..).next().unwrap_or_default();
            first = Some(
                out.token()
                    .context("the last prefill chunk sampled nothing")?,
            );
        }
    }
    let mut driven = vec![Driven {
        seq,
        ids: vec![first.context("a prompt with no last chunk")?],
        committed_before: 1,
        accepted: 0,
        context: prompt.len() as u32 + 1,
        first_draft,
    }];
    if probe {
        probes.extend(decode_probed(
            forward,
            &mut driven,
            1,
            width,
            drafter,
            steps,
        )?);
    } else {
        decode_together(forward, &mut driven, 1, width, drafter, steps)?;
    }
    forward.release(seq);
    Ok((driven.remove(0).ids, probes))
}

/// What one acceptance run measured.
struct AcceptRun {
    /// Prompts driven.
    prompts: usize,
    /// Decode steps run.
    steps: u64,
    /// Steps whose slot carried a proposal at all. The denominator of `per_draft`.
    drafted: u64,
    /// `propose` calls that found nothing waiting: a sequence's FIRST decode step, and any step
    /// whose proposal the engine refused. Counted because a mailbox that is empty every step is a
    /// drafter that never ran, and that is invisible in the ids and in the acceptance rate alike.
    empty: u64,
    /// Tokens committed.
    committed: u64,
    /// Draft tokens accepted.
    accepted: u64,
    /// How many steps accepted exactly `i` draft tokens, over the steps that carried a proposal.
    hist: Vec<u64>,
    /// **The ids this arm COMMITTED, per prompt** (ws-f36), for the one comparison the acceptance
    /// rate cannot make on its own.
    ///
    /// ws-f35 left a uniform 4.3 % per-token deficit with two candidate homes: the drafter's walk
    /// and the TARGET's own argmax. The second is measurable without a drafter at all — prod and
    /// Saber are both greedy over the same prompt ids, so a position where the two commit
    /// different tokens is a position where a correct draft would have been rejected — and the
    /// statistic is a HAZARD (the first divergence per prompt, over the positions reached with
    /// the prefix still agreeing), because after it the two engines are answering different
    /// prompts (GATES.md blind spot 12). `tools/gates/target_divergence.py` computes it against
    /// prod's own ids from `prod_spec_accept.py --ids-out`.
    ids: Vec<(usize, Vec<TokenId>)>,
    /// **H46: per prompt, per decode step, (accepted, the proposal's ids)** — written beside the
    /// ids by `--accept-ids` (`"acc"`, `"drafts"`), so two boots whose ids agree and whose
    /// acceptance does not name the prompt and the first step whose PROPOSAL parted.
    trace: Vec<Vec<(u32, Vec<TokenId>)>>,
}

impl AcceptRun {
    /// Accepted draft tokens per draft — prod's `saber:spec_decode_num_accepted / num_drafts`.
    fn per_draft(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// Tokens committed per decode step, which is what the step time multiplies.
    fn per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.committed as f64 / self.steps as f64
        }
    }
}

/// ACCEPTANCE over the corpus, through the pass, with whatever drafter this run carries.
///
/// One request at a time on purpose: acceptance is a property of the sequence and its drafter, and
/// a batched arm would fold the width controller's cap into the number. Each prompt is admitted,
/// prefilled the way the scheduler chunks it, decoded until it has committed `tokens`, and
/// released — and the drafter is released with it, because a mailbox that kept a finished
/// sequence's block would propose from a history nothing has.
#[allow(clippy::too_many_arguments)]
fn acceptance_arm(
    forward: &mut HipForward,
    prompts: &[Vec<TokenId>],
    per_seq: u32,
    prefill_chunk: u32,
    width: u32,
    drafter: &mut dyn Drafter,
    tokens: usize,
    max_ctx: u32,
    mut lattice: Option<&mut LatticeSink>,
) -> anyhow::Result<AcceptRun> {
    let k = (width.max(1) - 1) as usize;
    if lattice.is_some() {
        forward.set_draft_lattice_tap(true);
        let _ = forward.take_draft_lattices();
    }
    let mut run = AcceptRun {
        prompts: 0,
        steps: 0,
        drafted: 0,
        empty: 0,
        committed: 0,
        accepted: 0,
        hist: vec![0; k + 1],
        ids: Vec::new(),
        trace: Vec::new(),
    };
    for (i, p) in prompts.iter().enumerate() {
        // A prompt that cannot hold its own continuation is skipped rather than truncated: a
        // truncated prompt is a different request and its acceptance is a different number.
        if p.len() + tokens + width as usize >= max_ctx as usize {
            continue;
        }
        let seq = SeqId(900_000 + i as u64);
        let mut d = admit_and_prefill(forward, seq, p, &blocks_for(0, per_seq), prefill_chunk)?;
        offer_prefill_block(std::slice::from_mut(&mut d), drafter);
        if let Some(l) = lattice.as_deref_mut() {
            l.record(i, d.ids.len() - 1, &d.ids, forward.take_draft_lattices());
        }
        let start = d.ids.len();
        let mut steps_trace: Vec<(u32, Vec<TokenId>)> = Vec::new();
        while d.ids.len() - start < tokens {
            let input = *d.ids.last().expect("a prefilled sequence has a token");
            let mut draft = drafter.propose(seq, input, width - 1);
            let proposed = draft.len();
            let proposal = draft.clone();
            draft.resize(k, RecordedDrafter::WRONG);
            let slots = vec![DecodeSlot {
                seq,
                input,
                context_len: d.context - 1,
                draft,
                committed_before: d.committed_before,
                sampling: RowRequest::greedy(),
                decider: None,
            }];
            let mut out = forward.decode(&slots, 1, width)?;
            forward.propose_greedy(&mut out)?;
            let blocks = std::mem::take(&mut out.drafts);
            if let Some(b) = blocks.first() {
                drafter.offer(seq, b);
            }
            let c = out
                .commits
                .into_iter()
                .next()
                .context("a decode step that committed nothing")?;
            run.steps += 1;
            run.committed += c.tokens.len() as u64;
            if proposed == 0 {
                run.empty += 1;
            } else {
                run.drafted += 1;
                run.accepted += u64::from(c.accepted);
                run.hist[(c.accepted as usize).min(k)] += 1;
            }
            steps_trace.push((c.accepted, proposal));
            drafter.commit(seq, &c.tokens);
            d.committed_before = c.tokens.len() as u32;
            d.context += c.tokens.len() as u32;
            d.ids.extend(&c.tokens);
            if let Some(l) = lattice.as_deref_mut() {
                l.record(i, d.ids.len() - 1, &d.ids, forward.take_draft_lattices());
            }
        }
        if let Some(l) = lattice.as_deref_mut() {
            l.ids
                .insert(i.to_string(), serde_json::json!([0, d.ids.clone()]));
        }
        forward.release(seq);
        drafter.release(seq);
        // THE WHOLE VECTOR, FROM INDEX 0, and `check_committed_ids` is why that sentence is
        // written down rather than sliced (ws-p10). `Driven::ids` holds this sequence's
        // COMMITTED tokens and nothing else.
        let committed = d.ids.clone();
        if let Err(why) = check_committed_ids(&committed, start, tokens, width) {
            anyhow::bail!("the acceptance arm's committed ids for prompt {i}: {why}");
        }
        run.ids.push((i, committed));
        run.trace.push(steps_trace);
        run.prompts += 1;
    }
    if lattice.is_some() {
        forward.set_draft_lattice_tap(false);
    }
    Ok(run)
}

/// P27-10: the acceptance pass's drafter lattices, as `tools/capture/p27_10/tree_sim.py` reads
/// them (`load_engine`): `lattice.bin` = records of `(prompt i32, n i32, anchor index i32, anchor
/// id i64, cand i64[7*16], scores f32[7*16*16])`, `ids.json` = `{prompt: [0, committed ids]}` --
/// the anchor index is into the committed ids, so the drafts predict `ids[index + 1 ..]`.
#[derive(Default)]
struct LatticeSink {
    bin: Vec<u8>,
    ids: serde_json::Map<String, serde_json::Value>,
    /// Proposals that tapped other than exactly one lattice (a refused / batch-checked proposal).
    odd: u64,
    records: u64,
}

impl LatticeSink {
    fn record(&mut self, prompt: usize, at: usize, ids: &[TokenId], tapped: Vec<Vec<u8>>) {
        if tapped.len() != 1 {
            self.odd += 1;
            return;
        }
        self.bin
            .extend_from_slice(&i32::try_from(prompt).unwrap_or(-1).to_le_bytes());
        self.bin.extend_from_slice(&1i32.to_le_bytes());
        self.bin
            .extend_from_slice(&i32::try_from(at).unwrap_or(-1).to_le_bytes());
        self.bin
            .extend_from_slice(&i64::from(ids.get(at).copied().unwrap_or(0)).to_le_bytes());
        self.bin.extend_from_slice(&tapped[0]);
        self.records += 1;
    }

    fn write(&self, dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("lattice.bin"), &self.bin)?;
        std::fs::write(
            dir.join("ids.json"),
            serde_json::to_vec(&serde_json::Value::Object(self.ids.clone()))?,
        )?;
        Ok(())
    }
}

/// The law an `--accept-ids` row has to satisfy, as a function — because this one slice has been
/// wrong twice, in two directions, and neither time did anything refuse (ws-p10).
///
/// `Driven::ids` holds the sequence's COMMITTED tokens and nothing else: `admit_and_prefill`
/// seeds it with the single token the LAST PREFILL CHUNK sampled — which is committed token 0 —
/// and every decode step extends it by what that step committed. The capture is therefore the
/// whole vector from index 0, and its length is bounded on both sides:
///
///   * at least `prefill_ids + tokens`, because the loop runs until it has `tokens` decode
///     tokens on top of the prefill's one;
///   * at most `prefill_ids + tokens + width - 1`, because the last step can overshoot the
///     budget by at most the verify width less the token that crossed it.
///
/// THE TWO DEFECTS THIS REFUSES, each of which is one of the tests below:
///
///   * ws-f36 sliced from `start` (the length after the prefill), dropping committed token 0 —
///     the position the TP=2 token gate reads 186/200 at. Against prod's ids that reads as all
///     200 prompts diverging at position 0, and the published `f = 0.0276` was taken with
///     `--prod-offset 1`, i.e. over positions 1.. only, with the first token excluded;
///   * its fix sliced from `p.len()` on a comment that says the vector holds the prompt. It does
///     not, so every prompt longer than the token budget captured an EMPTY row: 175 of 200 empty
///     and the surviving 25 diverging at position 0 (ws-p10's first hold, caught here).
///
/// Generalise before quoting: a capture whose ROWS are wrong is not a measurement with a bug in
/// it, it is 25 prompts wearing the name of 200 — and the only thing that catches it at the site
/// is a length law the arm asserts before it writes the file.
fn check_committed_ids(
    ids: &[TokenId],
    prefill_ids: usize,
    tokens: usize,
    width: u32,
) -> Result<(), String> {
    if prefill_ids != 1 {
        return Err(format!(
            "the prefill contributed {prefill_ids} committed token(s), not 1: the last chunk \
             samples exactly one and the capture's index 0 is it"
        ));
    }
    let lo = prefill_ids + tokens;
    let hi = lo + (width.max(1) as usize - 1);
    if ids.len() < lo || ids.len() > hi {
        return Err(format!(
            "{} committed id(s) captured, and a run of {tokens} token(s) at verify width {width} \
             commits between {lo} and {hi}",
            ids.len()
        ));
    }
    Ok(())
}

/// The first token index at which two id streams disagree; `None` when they are the same stream.
fn first_id_difference(a: &[TokenId], b: &[TokenId]) -> Option<usize> {
    let n = a.len().min(b.len());
    a[..n]
        .iter()
        .zip(&b[..n])
        .position(|(x, y)| x != y)
        .or(if a.len() == b.len() { None } else { Some(n) })
}

/// **CMP27G: the chain's b1 verify widths** a width-controller arm may move between: every planned
/// `(1, w)` decode shape at or below the chain's widest width (`max_width`). The tree form's own
/// `b1t10` (`(1, 10)` under `--engine-spec-tree ancestor9`) is the ENGINE's form for every b1 step,
/// never a width the scheduler picks (`saber-cli` skips it from the buckets since P27-15f).
fn chain_b1_widths(decode_shapes: &[(u32, u32)], max_width: u32) -> Vec<u32> {
    let mut w: Vec<u32> = decode_shapes
        .iter()
        .filter(|(b, w)| *b == 1 && *w <= max_width)
        .map(|(_, w)| *w)
        .collect();
    w.sort_unstable();
    w.dedup();
    w
}

/// The first `(step, layer, edge)` at which row `ra` of `a` and row `rb` of `b` disagree.
///
/// `None` is the pass: every dump point of every step digested the same bytes. A dump point that
/// one side has and the other does not is a disagreement about the PLAN and is reported as one,
/// because two probe sets of different shapes cannot be compared and must not silently be.
fn first_probe_divergence(
    a: &[Vec<RowProbe>],
    ra: usize,
    b: &[Vec<RowProbe>],
    rb: usize,
) -> Option<(usize, i32, String)> {
    first_probe_divergence_op(a, ra, b, rb).map(|(s, l, e, _)| (s, l, e))
}

/// [`first_probe_divergence`] with the OP that wrote the edge (ws-p9).
///
/// The op is what a plan-fact delta is keyed on, so an attribution that has to say "this op
/// resolves two different launches in the two buckets" needs the name and not only the layer.
/// It also refuses a comparison of two points whose ROW WIDTH differs: an op-deep probe reads an
/// op's own output edge, and an edge that is not `rows x row_bytes` would be digested over a
/// different span in a batch-4 step than in a batch-1 one — two different quantities, reported
/// rather than silently differing.
fn first_probe_divergence_op(
    a: &[Vec<RowProbe>],
    ra: usize,
    b: &[Vec<RowProbe>],
    rb: usize,
) -> Option<(usize, i32, String, String)> {
    for (n, (sa, sb)) in a.iter().zip(b).enumerate() {
        if sa.len() != sb.len() {
            return Some((
                n,
                i32::MIN,
                format!("{} dump point(s) vs {}", sa.len(), sb.len()),
                "the plans disagree".to_owned(),
            ));
        }
        for (pa, pb) in sa.iter().zip(sb) {
            if pa.edge != pb.edge || pa.row_bytes != pb.row_bytes {
                return Some((
                    n,
                    pa.layer,
                    format!(
                        "`{}` at {} B/row against `{}` at {} B/row",
                        pa.edge, pa.row_bytes, pb.edge, pb.row_bytes
                    ),
                    pa.op.clone(),
                ));
            }
            let (Some(x), Some(y)) = (pa.rows.get(ra), pb.rows.get(rb)) else {
                return Some((
                    n,
                    pa.layer,
                    format!("{} has no row {ra}/{rb}", pa.edge),
                    pa.op.clone(),
                ));
            };
            if x != y {
                return Some((n, pa.layer, pa.edge.clone(), pa.op.clone()));
            }
        }
    }
    None
}

/// The first `(step, layer, edge, op, row)` at which the two probe sets disagree over the FIRST
/// `n` rows of each — the subject's own rows, when the subject is slot 0 of both buckets.
///
/// **Row 0 alone is not the subject** (ws-l18). Every decode op but the GDN pair is row-parallel,
/// so a difference an upstream op makes in row 3 is invisible at row 0 until something mixes
/// rows — and the conv update is the first op in a layer that does, because its output row `t`
/// convolves input rows `t-3..t` and its cache carries the previous step's last rows. So an
/// attribution that compares row 0 only reports the conv update as the first op to write
/// differing bytes whatever wrote them, which is what ws-p9's red was. The subject holds `width`
/// rows in both buckets and all of them are compared here.
fn first_probe_divergence_rows(
    a: &[Vec<RowProbe>],
    b: &[Vec<RowProbe>],
    n: usize,
) -> Option<(usize, i32, String, String, usize)> {
    for (step, (sa, sb)) in a.iter().zip(b).enumerate() {
        if sa.len() != sb.len() {
            return Some((
                step,
                i32::MIN,
                format!("{} dump point(s) vs {}", sa.len(), sb.len()),
                "the plans disagree".to_owned(),
                0,
            ));
        }
        for (pa, pb) in sa.iter().zip(sb) {
            if pa.edge != pb.edge || pa.row_bytes != pb.row_bytes {
                return Some((
                    step,
                    pa.layer,
                    format!(
                        "`{}` at {} B/row against `{}` at {} B/row",
                        pa.edge, pa.row_bytes, pb.edge, pb.row_bytes
                    ),
                    pa.op.clone(),
                    0,
                ));
            }
            for r in 0..n {
                let (Some(x), Some(y)) = (pa.rows.get(r), pb.rows.get(r)) else {
                    return Some((
                        step,
                        pa.layer,
                        format!("{} has no row {r} on one side", pa.edge),
                        pa.op.clone(),
                        r,
                    ));
                };
                if x != y {
                    return Some((step, pa.layer, pa.edge.clone(), pa.op.clone(), r));
                }
            }
        }
    }
    None
}

/// Decode steps an attribution runs to reach a flip the arm found at token `at`.
///
/// A step commits between one and `width` tokens, so the only bound that always reaches `at` is
/// `at + 1` steps — and the arm's own step count is the floor, because a shorter run would compare
/// a different prefix than the arm did. Capped so that a late flip cannot turn one red into a
/// minute of card time (ws-p9).
fn attribution_steps(at: usize, floor: usize) -> usize {
    floor.max(at + 1).min(256)
}

/// Nats above which a disagreement is no longer two arithmetics' last bits (ws-p9).
///
/// The number is `docs/GATES.md` blind spot 11's: the token gate's twelve flips are all decided by
/// less than a nat on at least one side, and P8's attribution turns on "no flip exceeds 1 nat on
/// both sides". A flip a whole nat wide is not a rounding that crossed a near-tie — something
/// chose differently — so it fails whatever else is true of it.
const NEAR_TIE_NATS: f32 = 1.0;

/// The kernel SYMBOLS in a plan-fact fingerprint, without the grid (ws-p9).
///
/// `ForwardFacts::bucket_ops` fingerprints a resolution as `<symbol> grid[…] block[…]` and joins
/// an op's several resolutions with ` + `. Two buckets that differ only in the grid's sequence
/// axis are running the same arithmetic on each row; two that resolve different symbols are not.
fn symbols_of(fingerprint: &str) -> std::collections::BTreeSet<&str> {
    fingerprint
        .split(" + ")
        .map(|p| p.split(" grid[").next().unwrap_or(p))
        .collect()
}

/// One committed token's own distribution, as the head's sampling seam wrote it (ws-p9).
///
/// The margin is the LOGIT gap, which is the logprob gap — the softmax normaliser cancels — and it
/// is the same quantity `saber forward --top2` reports for the pass. `top` is empty for a row whose
/// seam did not run, and a margin nobody could measure is reported as absent rather than as zero.
#[derive(Clone, Debug, Default)]
struct RowMargin {
    /// The token the step committed at this position.
    id: TokenId,
    /// The head's top-K raw logits and ids, descending.
    top: Vec<(TokenId, f32)>,
}

impl RowMargin {
    /// `top1 - top2`: how far this row was from choosing something else.
    fn gap(&self) -> Option<f32> {
        match (self.top.first(), self.top.get(1)) {
            (Some(a), Some(b)) => Some(a.1 - b.1),
            _ => None,
        }
    }

    /// This row's gap to the token the OTHER side chose — the quantity that bounds a
    /// disagreement (`docs/GATES.md` blind spot 11), and `None` when that token is not in the
    /// top-K at all, which is itself a statement: it is more than the K-th gap away.
    fn gap_to(&self, id: TokenId) -> Option<f32> {
        let top1 = self.top.first()?.1;
        self.top
            .iter()
            .find(|(t, _)| *t == id)
            .map(|(_, v)| top1 - v)
    }

    /// An EXACT tie between the top two: the rounding did not decide this row, the tiebreak did.
    fn exact_tie(&self) -> bool {
        self.gap() == Some(0.0)
    }
}

/// WHAT AN ID DIVERGENCE BETWEEN TWO DECODE BUCKETS IS (ws-p9).
///
/// Every red the batched, concurrency and mixed-arrival arms have reported since ws-s8 is one
/// sequence compared between a batch-1 run and a batch-N one, and until this existed the only
/// sentence available about one was "sequence 3 differs at token 51". Three measurements make it
/// a verdict instead: the first dump point the two buckets part at (down to the OP), the two
/// buckets' own resolved launches for that op, and the MARGIN at the flip on both sides.
#[derive(Clone, Debug, Default)]
struct BucketDivergence {
    /// The token the two runs' ids first differ at.
    at: usize,
    solo_id: Option<TokenId>,
    batch_id: Option<TokenId>,
    /// Each side's own `top1 - top2` there.
    solo_gap: Option<f32>,
    batch_gap: Option<f32>,
    /// Each side's gap to the token the other side chose.
    solo_to_batch: Option<f32>,
    batch_to_solo: Option<f32>,
    /// Committed rows within four tokens of the flip whose top two logits are EQUAL.
    ties: usize,
    /// The first dump point the two buckets differ at, one point per layer:
    /// `(step, layer, edge, op, row)`.
    layer_first: Option<(usize, i32, String, String, usize)>,
    /// The same question one point per OP: `(step, layer, edge, op, row)`.
    op_first: Option<(usize, i32, String, String, usize)>,
    /// What the two buckets resolved for that op, when they resolved it differently.
    op_delta: Option<(String, String)>,
    /// Every op the two buckets resolved differently, whatever the probe said.
    deltas: Vec<(String, String, String, u32)>,
    /// Whether a probe ran at all (it is refused at TP > 1).
    probed: bool,
    /// The slot the subject held in the margin run's batch (ws-p11): the arm's own slot under
    /// [`Subject::ArmSlot`], 0 under [`Subject::SlotZero`].
    subject_slot: usize,
}

/// Where an attribution's margin run puts the sequence it is asked about (ws-p11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Subject {
    /// The arm's own batch, in the arm's own order: the run that decides the verdict is the
    /// configuration the arm found the red in. The batched arm drives the engine directly, so it
    /// can be reproduced exactly, and it is.
    ArmSlot,
    /// The subject moved to slot 0, the order the TP=1 probes compare rows in. The concurrency
    /// arms use it: their batch composition is the scheduler's, which a direct drive does not
    /// reproduce in any order, so the attribution there is a neighbouring run by construction.
    SlotZero,
}

/// Prod's own N-in-flight-against-one divergence rate, BY THE NUMBER IN FLIGHT: the rate at which
/// the MODEL's own arithmetic choices produce id divergences when nothing is wrong, and so the rate
/// an arm's divergences are held against. Each row is prod's own stack, 200 corpus prompts x 64
/// greedy tokens, the conc-N pass against the same boot's one-at-a-time pass, at the UPPER end of
/// its prompt-bootstrap 95 % confidence interval (`docs/GATES.md` blind spot 21):
///
/// * **4** — P10: point 0.0080, CI 0.0062–0.0100 (Saber's own twin read 0.0084). ws-p12 re-read
///   it on another boot at 0.0087, CI 0.0068–0.0110, which is inside it;
/// * **8** — ws-p12: point **0.0089**, CI 0.0071–**0.0110** (83 events over 9,345 positions). Eight
///   in flight is NOT a faster floor than four on prod: the two rows are the same size.
///
/// A TABLE because the question an arm asks is "N in flight against one", and P10 measured only
/// N = 4 while the concurrency arm runs N = 8; one number quoted for every N was a row borrowed
/// from another question ([`documented_rate`] picks the row).
const DOCUMENTED_RATES: &[(usize, f64)] = &[(4, 0.0100), (8, 0.0110)];

/// The documented rate for an arm with `in_flight` sequences in its steps: the narrowest row that
/// covers it (a batch of 2 is held to the four-in-flight row, which is the nearer measurement and
/// the stricter one), and past the widest row that row, flagged `extrapolated`.
fn documented_rate(in_flight: usize) -> (f64, usize, bool) {
    for (n, r) in DOCUMENTED_RATES {
        if in_flight <= *n {
            return (*r, *n, false);
        }
    }
    let (n, r) = DOCUMENTED_RATES[DOCUMENTED_RATES.len() - 1];
    (r, n, true)
}

/// The one-sided p-value below which an arm's divergence count is too many for its
/// [`documented_rate`] over the positions it reached.
const RATE_ALPHA: f64 = 0.05;

/// The widest a DECIDED flip's nearer side has ever been in the documented floor (ws-p11).
///
/// P10 read both sides' margins at all 153 of Saber-against-prod's events: 3 are above a nat both
/// ways, and their NEARER sides are 1.500 (row 161, 1.500 / 3.125), 1.250 (row 81) and 1.000
/// (row 164). A flip whose nearer side is two nats is outside everything the floor has produced —
/// the card's wrong-snapshot control is 2.75 at TP=2 and 4.5 at TP=1 — so it is a different
/// answer at any rate.
const DECIDED_CEILING_NATS: f32 = 2.0;

/// An arm's id divergences over the positions it compared, as a HAZARD (ws-p11).
///
/// Each sequence contributes the positions it reached with its prefix still agreeing — up to and
/// including the one it diverged at — and one event if it diverged; a sequence that agreed
/// contributes its whole compared prefix and no event. That is `target_divergence.py`'s estimator
/// (GATES.md blind spot 12: a first divergence ends a sequence's exposure, so it is not a mean).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ArmRate {
    events: usize,
    reached: usize,
    /// Sequences in flight in the arm's steps, which picks the [`DOCUMENTED_RATES`] row (ws-p12).
    /// 0 (the default) is read as one, which the narrowest row covers.
    in_flight: usize,
}

impl ArmRate {
    /// The hazard of `(ours, reference)` pairs, compared over their common length.
    fn of<'a>(pairs: impl IntoIterator<Item = (&'a [TokenId], &'a [TokenId])>) -> Self {
        let mut r = Self::default();
        for (a, b) in pairs {
            let n = a.len().min(b.len());
            match a[..n].iter().zip(&b[..n]).position(|(x, y)| x != y) {
                Some(at) => {
                    r.events += 1;
                    r.reached += at + 1;
                }
                None => r.reached += n,
            }
        }
        r
    }

    /// The hazard of `(prompt, ours, reference)` STREAMS, counting each distinct prompt ONCE
    /// (ws-p13): the first stream of a prompt is the sample and a later copy of the same prompt
    /// adds neither positions nor events. Two copies of one prompt in one batch are one question
    /// asked twice — prod's own copies diverge at the same token (ws-p12) — so counting both
    /// doubles the positions and halves the p-value of an arm that fed `prompts[i % 4]`.
    fn of_streams<'a>(
        streams: impl IntoIterator<Item = (usize, &'a [TokenId], &'a [TokenId])>,
    ) -> Self {
        let mut seen = std::collections::BTreeSet::new();
        Self::of(
            streams
                .into_iter()
                .filter(|(p, _, _)| seen.insert(*p))
                .map(|(_, a, b)| (a, b)),
        )
    }

    /// The same hazard, held against the row for `n` sequences in flight.
    fn in_flight(mut self, n: usize) -> Self {
        self.in_flight = n;
        self
    }

    /// Events per position reached.
    fn rate(&self) -> f64 {
        if self.reached == 0 {
            return 0.0;
        }
        self.events as f64 / self.reached as f64
    }

    /// P(X >= events) for X ~ Poisson(documented rate x reached): how surprising this many
    /// divergences would be if the arm diverged at the documented rate for its in-flight count
    /// and no faster.
    fn p_value(&self) -> f64 {
        if self.events == 0 {
            return 1.0;
        }
        let lambda = documented_rate(self.in_flight).0 * self.reached as f64;
        let mut term = (-lambda).exp();
        let mut below = 0.0;
        for i in 0..self.events {
            below += term;
            term *= lambda / (i + 1) as f64;
        }
        (1.0 - below).clamp(0.0, 1.0)
    }

    /// Whether the arm's divergences are what the documented rate produces.
    fn within_band(&self) -> bool {
        self.p_value() >= RATE_ALPHA
    }

    fn line(&self) -> String {
        let (rate, row, beyond) = documented_rate(self.in_flight);
        format!(
            "{} divergence(s) over {} position(s) reached = {:.4} per token, p = {:.3} against \
             prod's own {row}-in-flight-vs-1 rate {rate:.4}{}",
            self.events,
            self.reached,
            self.rate(),
            self.p_value(),
            if beyond {
                format!(
                    " (EXTRAPOLATED: {} in flight is past the table's widest row)",
                    self.in_flight
                )
            } else {
                String::new()
            }
        )
    }
}

impl BucketDivergence {
    /// The worst margin anybody measured at the flip.
    fn worst_nats(&self) -> Option<f32> {
        let all = [
            self.solo_gap,
            self.batch_gap,
            self.solo_to_batch,
            self.batch_to_solo,
        ];
        all.iter()
            .flatten()
            .copied()
            .fold(None, |a: Option<f32>, b| Some(a.map_or(b, |x| x.max(b))))
    }

    /// The first dump point worth naming: the op-deep one when it ran, the layer otherwise.
    fn first(&self) -> Option<&(usize, i32, String, String, usize)> {
        self.op_first.as_ref().or(self.layer_first.as_ref())
    }

    /// Deltas that move an op's KERNEL rather than only its grid — the ones that are a different
    /// arithmetic per row.
    fn arithmetic_deltas(&self) -> usize {
        self.deltas
            .iter()
            .filter(|(_, a, b, _)| symbols_of(a) != symbols_of(b))
            .count()
    }

    /// The NEARER side of the flip: the smaller of each side's gap to the token the OTHER side
    /// chose (`docs/GATES.md` blind spot 11). A token outside a side's top-K is further than its
    /// K-th gap, so it counts as infinitely far here — it cannot make a flip a near-tie.
    fn nearer_nats(&self) -> Option<f32> {
        if self.solo_to_batch.is_none() && self.batch_to_solo.is_none() {
            return None;
        }
        Some(
            self.solo_to_batch
                .unwrap_or(f32::INFINITY)
                .min(self.batch_to_solo.unwrap_or(f32::INFINITY)),
        )
    }

    /// [`Self::verdict_at`] with no arm rate: a DECIDED flip fails, because there is nothing to
    /// hold it against.
    #[cfg(test)]
    fn verdict(&self, tp: u32) -> Result<String, String> {
        self.verdict_at(tp, None)
    }

    /// **THE LAW.** `Ok` is the documented bucket-arithmetic class (or a decided flip at the
    /// documented rate), with its nats; `Err` is a defect and says why.
    ///
    /// Three things have to hold, and each one is a way this can fail:
    ///
    /// * **the margin** — the flip is a near-tie for AT LEAST ONE side: one side's gap to the
    ///   token the other chose is below [`NEAR_TIE_NATS`]. That is blind spot 11's own statement
    ///   of the class ("no flip exceeds 1 nat on both sides") and P10's ("150 of the 153 events
    ///   are a near-tie for at least one engine"). Until ws-p11 this read the WORST of all four
    ///   gaps, which is stricter than the law it cites: it called ws-k16's 1.75 / 0.50 flip a
    ///   different answer when the batched side sat half a nat from the solo side's token, and it
    ///   would have called 39 of P10's 153 floor events the same.
    ///
    ///   A flip above a nat BOTH ways is DECIDED, and the floor produces those too (3 of P10's
    ///   153). One is REPORTED, not failed, only when the arm it came from diverges at the
    ///   documented rate ([`ArmRate::within_band`]) and its nearer side is inside the widest the
    ///   floor has produced ([`DECIDED_CEILING_NATS`]); otherwise it is a different answer;
    /// * **the place** — the two buckets' streams first part INSIDE a layer, never at the
    ///   embedding (which would mean they were fed different ids);
    /// * **the op** — the op that wrote the first differing bytes is one the two buckets
    ///   RESOLVED DIFFERENTLY. An op resolved identically in both and writing different bytes is
    ///   the same launch reading something else, which is a defect and is exactly what ws-l17's
    ///   `cache_idx` binding was.
    ///
    /// At TP > 1 there is no probe — a dump point is one rank's half of a reduced stream — so the
    /// last two cannot be asked and the verdict rests on the margin and on the buckets differing
    /// at all. That is weaker, it is said out loud in the line it returns, and `docs/GATES.md`
    /// carries it as a blind spot.
    fn verdict_at(&self, tp: u32, rate: Option<&ArmRate>) -> Result<String, String> {
        if self.solo_id.is_none() && self.batch_id.is_none() {
            return Err(format!(
                "the attribution's own two runs AGREED over {} token(s): the arm's divergence did \
                 not reproduce here, so nothing below is about it",
                self.at
            ));
        }
        let Some(worst) = self.worst_nats() else {
            return Err(
                "no margin: the head's seam produced no distribution for the flip, so nothing \
                 here bounds the disagreement"
                    .to_owned(),
            );
        };
        if self.solo_gap.is_none() || self.batch_gap.is_none() {
            return Err(
                "one side has no distribution at the flip, so the two margins cannot be compared"
                    .to_owned(),
            );
        }
        let nearer = self.nearer_nats().unwrap_or(f32::INFINITY);
        let mut decided: Option<String> = None;
        if nearer >= NEAR_TIE_NATS {
            let Some(r) = rate else {
                return Err(format!(
                    "the flip is decided both ways (nearer side {nearer:.4} nats), which is a \
                     different answer and not two arithmetics' last bits, and there is no arm \
                     rate to hold it against"
                ));
            };
            if nearer >= DECIDED_CEILING_NATS {
                return Err(format!(
                    "the flip is decided by {nearer:.4} nats on its NEARER side, wider than any \
                     event the documented floor has produced ({DECIDED_CEILING_NATS:.1}): a \
                     different answer, whatever the arm's rate"
                ));
            }
            if !r.within_band() {
                return Err(format!(
                    "the flip is decided both ways (nearer side {nearer:.4} nats) and the arm \
                     diverges faster than the documented floor: {}",
                    r.line()
                ));
            }
            decided = Some(format!(
                "a DECIDED flip (nearer side {nearer:.4} nats, both above {NEAR_TIE_NATS:.1}), \
                 REPORTED and not failed because the arm diverges at the documented rate — {}; \
                 ",
                r.line()
            ));
        }
        let prefix = decided.unwrap_or_default();
        if !self.probed {
            let arith = self.arithmetic_deltas();
            if arith == 0 {
                return Err(
                    "the two buckets resolve every op to the same KERNEL, so a divergence between \
                     them is a defect and not an arithmetic"
                        .to_owned(),
                );
            }
            return Ok(format!(
                "{prefix}documented class ({worst:.4} nats, nearer side {nearer:.4}), attributed \
                 at TP={tp} by the MARGIN and the plan facts alone: {arith} op(s) resolve a \
                 different KERNEL between the two buckets and no probe can run at two ranks (a \
                 dump point is one rank's half of a reduced stream), so the layer is not named \
                 here — `docs/GATES.md` carries that as a blind spot"
            ));
        }
        let Some((step, layer, edge, op, row)) = self.first() else {
            return Err(
                "the two buckets' ids differ and every dump point of every probed step agreed: \
                 the divergence is downstream of the body, which no bucket arithmetic reaches"
                    .to_owned(),
            );
        };
        if *layer < 0 {
            return Err(format!(
                "the two buckets part at the EMBEDDING (`{edge}`, step {step}): they were fed \
                 different ids, which is upstream of anything a bucket decides"
            ));
        }
        match &self.op_delta {
            None => Err(format!(
                "`{op}` (step {step}, layer {layer}, `{edge}`, row {row}) resolves to the SAME \
                 launch in both buckets and wrote different bytes: one kernel reading something \
                 else is a defect"
            )),
            // THE SYMBOL, NOT THE GRID. A decode op's grid carries the sequence axis — `b4s8`
            // launches `saber_attn_decode` over four sequences where `b1s8` launches it over one —
            // and a workgroup that serves its own sequence computes the same bits either way,
            // which is what ws-l15's pinned split count made true. What changes a row's
            // ARITHMETIC is the instantiation: `ks4` against `ks2` is a different partition of K,
            // `tm1` against `tm2` a different M tile, `vs2` against `vs1` a different value split.
            // So a delta that moves only the grid is NOT an arithmetic difference, and an op that
            // wrote different bytes under it is a defect.
            Some((a, b)) if symbols_of(a) == symbols_of(b) => Err(format!(
                "`{op}` (step {step}, layer {layer}, `{edge}`, row {row}) resolves the SAME KERNEL \
                 in both buckets and only its grid moves ({a} against {b}): a workgroup that \
                 serves its own sequence computes the same bits whatever the grid's sequence axis \
                 is, so different bytes here are a defect"
            )),
            Some((a, b)) => Ok(format!(
                "{prefix}documented class ({worst:.4} nats, nearer side {nearer:.4}): the two \
                 buckets first part at `{op}` (step {step}, layer {layer}, `{edge}`, row {row}), \
                 which resolves `{a}` in one and `{b}` in the other"
            )),
        }
    }

    /// The lines a report prints under a red, whichever way the verdict went.
    fn lines(&self) -> Vec<String> {
        let nats = |x: Option<f32>| match x {
            Some(v) => format!("{v:.4}"),
            None => "not in the top-K".to_owned(),
        };
        let mut out = vec![format!(
            "          token {}: solo {:?} at {} nats, batched {:?} at {} nats; each side's gap to \
             the OTHER's token: solo→batched {}, batched→solo {}; {} exact tie(s) within four \
             tokens; the subject in slot {} of the margin run's batch",
            self.at,
            self.solo_id,
            nats(self.solo_gap),
            self.batch_id,
            nats(self.batch_gap),
            nats(self.solo_to_batch),
            nats(self.batch_to_solo),
            self.ties,
            self.subject_slot
        )];
        if let Some((step, layer, edge, op, row)) = self.first() {
            out.push(format!(
                "          first differing dump point: step {step}, layer {layer}, row {row} \
                 (`{edge}`, written by `{op}`)"
            ));
        } else if self.probed {
            out.push(
                "          first differing dump point: none — every probed point agreed".into(),
            );
        }
        if !self.deltas.is_empty() {
            out.push(format!(
                "          the two buckets resolve {} op(s) differently, {} of them a different \
                 KERNEL and the rest only a grid:",
                self.deltas.len(),
                self.arithmetic_deltas()
            ));
            for (op, a, b, n) in &self.deltas {
                let kind = if symbols_of(a) == symbols_of(b) {
                    "grid"
                } else {
                    "KERNEL"
                };
                out.push(format!(
                    "            {op} x{n} ({kind})\n              A: {a}\n              B: {b}"
                ));
            }
        }
        out
    }
}

/// How a probe divergence reads in a report line.
fn probe_where(d: Option<&(usize, i32, String)>) -> String {
    match d {
        Some((step, layer, edge)) => format!("step {step}, layer {layer} (`{edge}`)"),
        None => "none: every dump point agreed".to_owned(),
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// The median, which is the statistic a heavy tail does not move.
///
/// Both are printed for every timed arm because they answer different questions: the MEAN is the
/// throughput a caller sees and the P50 is what a step costs. At TP=1 they agree to two decimals; at
/// TP=2 a handful of multi-hundred-millisecond stalls per run separated them by 12 ms/step before the
/// wait's spin window was shortened (2026-09-20), and a run that printed only the mean would have
/// reported the stalls as the step time.
fn p50(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Prod's answer to one corpus row, out of `nospec_ref.jsonl` (workstream P).
///
/// The gate's identity arms all ask "does this engine agree with itself". That is worth asking and
/// it is not worth much alone: an engine that answers the same wrong token every time satisfies all
/// of them. This is the other question — "does it agree with the model" — and it is the one that
/// caught the gate running a truncated prompt for a week.
#[derive(Clone, Debug)]
struct Reference {
    /// Tokens of the prompt prod was given. A run that truncates below this is not comparable.
    n_prompt: usize,
    /// The token prod produced first, greedily.
    token_id: TokenId,
    /// And its text, for a message a human can read.
    text: String,
    /// Top-1 minus top-2, in nats. Small means an id comparison here measures the tie-break.
    margin: f64,
}

impl Reference {
    /// Ids the vocabulary has but the tokenizer does not use for text: `<|endoftext|>` and every
    /// control token above it. A reference in this range is a prompt prod had nothing to say about,
    /// and an identity gate over the continuation of one is a gate over a delimiter loop.
    const FIRST_SPECIAL: TokenId = 248_044;

    fn is_special(&self) -> bool {
        self.token_id >= Self::FIRST_SPECIAL
    }
}

/// Read `nospec_ref.jsonl`, keyed by corpus row.
///
/// Missing file is not an error — the reference is a fixture and a fresh worktree may not have it —
/// but the gate says so loudly, because every identity arm is weaker without it.
fn read_reference(path: &Path) -> anyhow::Result<BTreeMap<usize, Reference>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the reference {}", path.display()))?;
    let mut out = BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}: line {}", path.display(), n + 1))?;
        let id = v["id"].as_u64().unwrap_or(n as u64) as usize;
        let token_id = v["token_id"]
            .as_u64()
            .with_context(|| format!("{}: line {} has no token_id", path.display(), n + 1))?
            as TokenId;
        // The margin in nats, from the top-5 logprobs prod reported. Two entries are enough; a row
        // with one is a row whose runner-up nobody recorded, and `INFINITY` is the honest answer
        // for "nothing was close" rather than a zero that would refuse it.
        let mut lps: Vec<f64> = v["top5"]
            .as_object()
            .map(|o| o.values().filter_map(serde_json::Value::as_f64).collect())
            .unwrap_or_default();
        lps.sort_by(|a, b| b.total_cmp(a));
        let margin = if lps.len() >= 2 {
            lps[0] - lps[1]
        } else {
            f64::INFINITY
        };
        out.insert(
            id,
            Reference {
                n_prompt: v["n_prompt"].as_u64().unwrap_or(0) as usize,
                token_id,
                text: v["text"].as_str().unwrap_or("?").to_owned(),
                margin,
            },
        );
    }
    Ok(out)
}

/// One recorded chat turn, rendered to the ids the serve plane would have sent.
#[derive(Debug)]
struct MessageTurn {
    /// The transcript says this turn's two passes differed.
    differed: bool,
    /// `conv 3 turn 2 DIFFERED, 218 tokens`, for the report.
    label: String,
    ids: Vec<TokenId>,
}

/// A MESSAGES transcript, rendered through the serve plane's own renderer and tokenizer — the two
/// halves the corpus path skips entirely (blind spot 4 in `docs/GATES.md`).
///
/// One JSON object per line, each with a `messages` array; `multiturn_continuity.py` writes
/// exactly that, plus `same`, `conv`, `turn` and `thinking`. This path exists because a turn that
/// FAILS the continuity gate cannot be rebuilt from the conversation corpus: its history carries
/// the model's own earlier replies, so the transcript is the only record of the request.
///
/// **The order is the transcript's own, reversed.** `multiturn_continuity.py` writes its rows as
/// it re-issues them, which is pass B — the REVERSE of the order they were first served in. A
/// replay that takes the file order replays pass B twice, and the whole property under test is
/// about order, so this returns pass A's order and the caller says so.
///
/// `enable_thinking` comes from the row when it carries one and is **false** otherwise, which is
/// what `multiturn.py` and `multiturn_continuity.py` both send: a different value renders a
/// different prompt and would be a different request.
fn read_message_turns(
    path: &Path,
    tokenizer: &Path,
    template: &Path,
) -> anyhow::Result<Vec<MessageTurn>> {
    use saber_serve::detok::TextCodec;
    let tok = saber_model::Tokenizer::from_file(tokenizer)
        .with_context(|| format!("the tokenizer {}", tokenizer.display()))?;
    let renderer = TemplateRenderer::from_file(template)
        .with_context(|| format!("the chat template {}", template.display()))?;
    let codec = ModelCodec::new(Arc::new(tok));
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the transcript {}", path.display()))?;
    let mut turns: Vec<MessageTurn> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}: line {}", path.display(), n + 1))?;
        let Some(messages) = v.get("messages").and_then(serde_json::Value::as_array) else {
            bail!(
                "{}: line {} has no `messages` array — this file is a transcript from \
                 `multiturn_continuity.py`, which records one per turn",
                path.display(),
                n + 1
            );
        };
        let thinking = v
            .get("thinking")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let mut kwargs = serde_json::Map::new();
        kwargs.insert(
            "enable_thinking".to_owned(),
            serde_json::Value::Bool(thinking),
        );
        let prompt = renderer
            .render(messages, None, true, &kwargs)
            .map_err(|e| anyhow::anyhow!("rendering a turn: {e:?}"))?;
        let ids = codec
            .encode(&prompt)
            .map_err(|e| anyhow::anyhow!("tokenizing a turn: {e:?}"))?;
        let differed = !v
            .get("same")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        turns.push(MessageTurn {
            differed,
            label: format!(
                "conv {} turn {}{}, {} tokens",
                v.get("conv")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                v.get("turn")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                if differed { " DIFFERED" } else { "" },
                ids.len()
            ),
            ids,
        });
    }
    if turns.is_empty() {
        bail!("{}: no turns in the transcript", path.display());
    }
    turns.reverse();
    Ok(turns)
}

/// Run the gate. Returns the number of failures, so the caller's exit code is the gate's verdict.
/// **The chunked prefill against the one-bucket prefill, layer by layer** (ws-h29).
///
/// One prompt, two ways to feed it, on one engine in one process:
///
/// * arm A runs it as ONE bucket — the shape `saber forward` plans, a bucket of the prompt's own
///   length — and digests the residual stream at every layer at rows 2047, 4095, …, the last row
///   of each chunk the other arm will feed;
/// * arm B runs it as `chunk_rows`-row chunks the way the scheduler does, digesting each chunk's
///   last live row at every layer.
///
/// Row *i* of arm A and chunk *i* of arm B are then **the same absolute token**, so the first
/// chunk at which they disagree is the first chunk whose inputs differ, and the first layer within
/// it is where. Everything the two arms do differently lives in the chunk boundary:
/// `done_before`-dependent `pos` and `seqused_k`, the `has_init` byte, the GDN recurrent carry
/// through the state slot, the conv cache's tail, and the GEMM's row-tile count.
///
/// The host pre-flight says the last of those is not a suspect:
/// `layer::tests::a_chunk_and_a_whole_prompt_resolve_the_same_kernels` asserts that a 2048-row and
/// a 14,301-row prefill resolve the SAME kernel, the same block shape and the same scalar kernargs
/// in all 882 ops — only the row-tile count of the grid moves, and a GEMM row does not read another
/// row's tile.
///
/// TP=1 only, because the probe is TP=1 only (`device.rs` refuses one rank's half of a reduced
/// residual by name).
fn chunk_bisect(
    forward: &mut HipForward,
    chunk_rows: u32,
    prompt: &[TokenId],
    per_seq: u32,
) -> anyhow::Result<usize> {
    let len = prompt.len() as u32;
    let mut shapes: Vec<u32> = forward
        .facts()
        .declared
        .iter()
        .filter(|(s, _)| s.phase == Phase::Prefill)
        .map(|(s, _)| s.rows)
        .collect();
    shapes.sort_unstable();
    shapes.dedup();
    if !shapes.contains(&chunk_rows) {
        bail!(
            "--chunk-bisect {chunk_rows} needs a planned prefill shape of exactly {chunk_rows} \
             row(s); this engine planned {shapes:?}"
        );
    }
    let whole = shapes
        .iter()
        .copied()
        .find(|r| *r >= len)
        .filter(|r| *r != chunk_rows)
        .with_context(|| {
            format!(
                "a {len}-token prompt needs a planned prefill shape that holds it WHOLE and is not \
                 the chunk size; this engine planned {shapes:?}. Pass \
                 `--prefill-rows {chunk_rows},{len}`"
            )
        })?;
    // The absolute row each chunk of arm B ends on, which is what arm A has to digest.
    let ends: Vec<u32> = (0..len.div_ceil(chunk_rows))
        .map(|i| ((i + 1) * chunk_rows).min(len) - 1)
        .collect();
    println!(
        "bisect    {len} token(s) as ONE {whole}-row bucket against {} chunk(s) of {chunk_rows}, \
         digesting rows {:?} on both sides",
        ends.len(),
        ends
    );
    let blocks = blocks_for(0, per_seq);
    if (blocks.len() as u32) * 16 < len {
        bail!(
            "{} page(s) of 16 tokens for a {len}-token prompt: the bisect would ask `kv_append` \
             for a page the sequence does not hold",
            blocks.len()
        );
    }

    // ---- arm A: one bucket, the shape `saber forward` runs ------------------------------------
    forward.admit(SeqId(970), len, &blocks)?;
    let (mut whole_out, whole_probe) =
        forward.prefill_probe_at(SeqId(970), prompt, 0, true, Some(whole), &ends)?;
    forward.propose_greedy(&mut whole_out)?;
    forward.release(SeqId(970));
    println!(
        "bisect    one bucket: {} layer(s) probed, sampled {:?}",
        whole_probe.len(),
        whole_out.token()
    );

    // ---- arm B: the scheduler's chunks ---------------------------------------------------------
    forward.admit(SeqId(971), len, &blocks)?;
    let mut failures = 0usize;
    let mut first: Option<(usize, i32, String)> = None;
    let mut chunk_token = None;
    let mut done = 0u32;
    for (i, part) in prompt.chunks(chunk_rows as usize).enumerate() {
        let last = done + part.len() as u32 >= len;
        let live = part.len() as u32;
        let (mut out, probe) = forward.prefill_probe_at(
            SeqId(971),
            part,
            done,
            last,
            Some(chunk_rows),
            &[live - 1],
        )?;
        forward.propose_greedy(&mut out)?;
        done += live;
        if last {
            chunk_token = out.token();
        }
        if probe.len() != whole_probe.len() {
            bail!(
                "chunk {i} probed {} layer(s) and the one-bucket arm probed {}: the two arms ran \
                 different plans",
                probe.len(),
                whole_probe.len()
            );
        }
        let moved = probe
            .iter()
            .zip(&whole_probe)
            .find(|(b, a)| b.rows.first() != a.rows.get(i))
            .map(|(b, _)| (b.layer, b.edge.clone()));
        match moved {
            None => println!(
                "OK        chunk {i} (rows {}..{}): bit-identical to the one-bucket run at all {} \
                 layers, at absolute row {}",
                done - live,
                done - 1,
                probe.len(),
                ends[i]
            ),
            Some((layer, edge)) => {
                println!(
                    "DIFFERS   chunk {i} (rows {}..{}): first layer to move is {layer} (`{edge}`) \
                     at absolute row {}",
                    done - live,
                    done - 1,
                    ends[i]
                );
                if first.is_none() {
                    first = Some((i, layer, edge));
                }
            }
        }
    }
    forward.release(SeqId(971));

    match &first {
        None => println!(
            "OK        every chunk's last live row is bit-identical to the one-bucket run's same \
             row at every layer"
        ),
        Some((chunk, layer, edge)) => println!(
            "bisect    FIRST DIVERGENCE: chunk {chunk} (absolute row {}), layer {layer} \
             (`{edge}`) — {} chunk(s) before it agreed at every layer",
            ends[*chunk], chunk
        ),
    }
    if whole_out.token() == chunk_token {
        println!(
            "OK        both arms sample {:?}: the chunking does not move this prompt's token",
            chunk_token
        );
    } else {
        println!(
            "FAIL      the one-bucket arm samples {:?} and the {}-chunk arm {:?} — ws-h28's \
             disagreement, reproduced through the pass with no HTTP and no scheduler in it",
            whole_out.token(),
            ends.len(),
            chunk_token
        );
        failures += 1;
    }
    Ok(failures)
}

#[allow(clippy::too_many_lines)]
pub fn gate(args: &GateArgs) -> anyhow::Result<usize> {
    // ws-f37's pairing control, refused when it cannot be honoured: with no resident drafter
    // nothing proposes, in the step or after it.
    if args.engine.propose_in_step && args.engine.draft_pack.is_none() {
        anyhow::bail!(
            "--propose-in-step needs --draft-pack: without a resident drafter there is no \
             proposal to put back inside the step"
        );
    }
    let mut failures = 0usize;
    let fail = |what: &str| println!("FAIL      {what}");

    // ---- the prompts, and prod's answer to each -------------------------------------------------
    //
    // Choosing the rows is part of the gate now. Every identity arm below asks whether this engine
    // agrees with ITSELF, which an engine that answers the same wrong token every time satisfies
    // perfectly — so the rows are chosen to be rows where agreeing with itself and agreeing with the
    // model are the same question. Two ways that fails, both refused by name:
    //
    //   * the reference's first token is a CONTROL token. Prod had nothing to say, and the arms
    //     would be comparing two continuations of a turn that already ended. (Nothing on the current
    //     corpus trips this: all 200 rows answer with text. It is a guard, not a workaround.)
    //   * the reference's margin is under `--min-margin`. Prod's own top two are that close, so the
    //     token is decided by rounding and an id comparison measures the tie-break. Real on this
    //     corpus: rows 78, 140 and 153 are tied at 0.00 nats.
    //
    // And the refusal that actually bit: `--tokens` truncating the prompt below the length the
    // reference was taken at. Until 2026-09-20 the harness passed `--tokens 25`, so every arm ran
    // the first 25 tokens of a 234-token chat prompt — ending mid-sentence on "give the final answer
    // on a new" — and the gate compared one continuation of a fragment with another.
    // A MESSAGES transcript replaces the corpus and the reference alike: prod has no recorded
    // first token for a turn this engine's own reply is half of, so the arms below are this
    // engine against itself and the banner says so. That is the right trade for the one question
    // these rows answer — whether a request that FAILED the continuity gate fails it here, where
    // the probes can bisect it.
    let from_messages = match (&args.messages, &args.tokenizer, &args.chat_template) {
        (Some(m), Some(t), Some(c)) => Some(read_message_turns(m, t, c)?),
        (Some(_), _, _) => bail!(
            "--messages needs --tokenizer and --chat-template: a transcript is text, and the ids \
             it becomes are the serve plane's renderer and tokenizer, not this gate's guess"
        ),
        _ => None,
    };
    let reference = match (&args.reference, &from_messages) {
        (Some(path), None) => Some(read_reference(path)?),
        _ => None,
    };
    let mut prompts: Vec<Vec<TokenId>> = Vec::new();
    let mut chosen: Vec<usize> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut row = args.prompt_index;
    // The transcript feeds TWO lists, and they are different questions. `prompts` is what every
    // other arm runs, so it is the turns that FAILED, first — a small set, and the sharpest one.
    // `continuity_prompts` is the whole transcript in pass-A order, because the continuity arm's
    // property is about ORDER: replaying every turn in order and then in reverse is exactly what
    // `multiturn_continuity.py` does, and anything less is a different experiment.
    let mut continuity_prompts: Vec<Vec<TokenId>> = Vec::new();
    if let Some(turns) = &from_messages {
        let mut order: Vec<usize> = (0..turns.len()).collect();
        // Stable, so turns that agree keep their pass-A order behind the ones that did not.
        order.sort_by_key(|i| usize::from(!turns[*i].differed));
        println!(
            "prompts   {} turn(s) from {}, rendered and tokenized by the serve plane's own \
             template and tokenizer; {} of them DIFFERED and are taken first:",
            turns.len(),
            args.messages
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string()),
            turns.iter().filter(|t| t.differed).count()
        );
        for (i, at) in order.iter().take(args.gate_seqs.max(1)).enumerate() {
            println!("          subject {i}: {}", turns[*at].label);
            prompts.push(turns[*at].ids.clone());
            chosen.push(*at);
        }
        continuity_prompts = turns.iter().map(|t| t.ids.clone()).collect();
        println!(
            "          the continuity arm replays all {} in the order they were first served, \
             then in reverse",
            continuity_prompts.len()
        );
    }
    // THE CONCURRENCY ARMS' OWN PROMPTS (ws-p13). Those arms fed `prompts[i % 4]` until ws-p13, so
    // concurrency 8 was four prompts TWICE, both copies in one batch, and its hazard counted eight
    // independent streams where prod's own duplicated copies diverge identically (ws-p12: the
    // arm's eight streams on prod, rows 2 and 4 each diverging at the same token in both copies).
    // So the corpus walk below keeps going past `gate_seqs` until the widest concurrency arm that
    // will run has a DISTINCT prompt per stream, by the same eligibility law (no control-token
    // answer, prod's margin at least `min_margin`); every other arm still takes the first
    // `gate_seqs`. A transcript run keeps its own turns: the arms cycle them and the hazard counts
    // each distinct prompt once (`ArmRate::of_streams`).
    let want_rows = prompt_rows_wanted(
        args.gate_seqs,
        &args.conc,
        args.engine.max_seqs as usize,
        from_messages.is_some(),
    );
    while prompts.len() < want_rows {
        let mut ids = match crate::forward::read_prompt_ids(&args.prompt_ids, row) {
            Ok(v) => v,
            // Past the gate's own rows the corpus running out is not fatal: the concurrency arms
            // cycle what there is, and their hazard counts each distinct prompt once.
            Err(_) if prompts.len() >= args.gate_seqs.max(1) => break,
            Err(e) if row > args.prompt_index => {
                bail!(
                    "ran out of corpus rows at {row} looking for {} usable prompt(s) (skipped: \
                     {}): {e}",
                    args.gate_seqs.max(1),
                    skipped.join("; ")
                );
            }
            Err(e) => return Err(e),
        };
        if let Some(r) = reference.as_ref().and_then(|m| m.get(&row)) {
            if r.is_special() {
                skipped.push(format!(
                    "row {row} (prod answers {} ({:?}), a control token)",
                    r.token_id, r.text
                ));
                row += 1;
                continue;
            }
            if r.margin < args.min_margin {
                skipped.push(format!(
                    "row {row} (prod's margin {:.2} nats < {:.2})",
                    r.margin, args.min_margin
                ));
                row += 1;
                continue;
            }
            if args.tokens > 0 && (args.tokens as usize) < r.n_prompt {
                bail!(
                    "--tokens {} truncates corpus row {row} below the {} tokens its reference was \
                     taken at: the continuation of a fragment is not something prod has an answer \
                     for, so every identity arm would be comparing this engine with itself. Pass \
                     --tokens 0.",
                    args.tokens,
                    r.n_prompt
                );
            }
        }
        if args.tokens > 0 && ids.len() > args.tokens as usize {
            ids.truncate(args.tokens as usize);
        }
        prompts.push(ids.iter().map(|v| *v as TokenId).collect());
        chosen.push(row);
        row += 1;
    }
    // Every arm but the concurrency ones runs the gate's own `gate_seqs` rows.
    let conc_prompts: Vec<Vec<TokenId>> = prompts.clone();
    let conc_chosen: Vec<usize> = chosen.clone();
    prompts.truncate(args.gate_seqs.max(1));
    chosen.truncate(args.gate_seqs.max(1));
    match reference.as_ref() {
        Some(m) => {
            let r0 = m.get(&chosen[0]);
            println!(
                "reference {} row(s) {:?} against {}; prod answers {} first{}",
                chosen.len(),
                chosen,
                args.reference
                    .as_ref()
                    .map_or_else(String::new, |p| p.display().to_string()),
                r0.map_or_else(|| "?".to_owned(), |r| format!("{:?}", r.text)),
                r0.map_or_else(String::new, |r| format!(
                    " (token {}, margin {:.2} nats)",
                    r.token_id, r.margin
                ))
            );
            if !skipped.is_empty() {
                println!("          skipped {}", skipped.join("; "));
            }
        }
        None => println!(
            "reference NONE: pass --reference .cache/forward/nospec_ref.jsonl, or every identity \
             arm below is this engine agreeing with itself"
        ),
    }
    if conc_prompts.len() > prompts.len() {
        println!(
            "conc      the concurrency arms run {} DISTINCT prompt(s), corpus row(s) {:?} (ws-p13: \
             they cycled the gate's {} before)",
            conc_prompts.len(),
            conc_chosen,
            prompts.len()
        );
    }

    // ---- the engine, with its shapes captured --------------------------------------------------
    // At TP=2 `start_tier` spawns rank 1 first: it has to be connecting while this rank listens.
    let (eargs, mut worker) = start_tier(&args.engine)?;
    let (mut forward, ledger) = match saber_engine::spawn(eargs.config(), eargs.ledger(32)?) {
        Ok(v) => v,
        Err(e) => {
            if let Some(w) = worker.take() {
                println!("tier      {}", w.stop());
            }
            return Err(e.into());
        }
    };
    if let Some(w) = worker.as_mut() {
        w.check()?;
    }
    print_facts(&forward, &ledger, eargs.rank);
    println!(
        "prompts   {} of {} tokens from {} (rows {}..), {} decode steps, {} per batch",
        prompts.len(),
        prompts
            .iter()
            .map(|p| p.len().to_string())
            .collect::<Vec<_>>()
            .join("/"),
        args.prompt_ids.display(),
        args.prompt_index,
        args.gen,
        args.batch_gen
    );
    // THE RUN'S WIDTH, and the drafter every arm of the gate uses. A run plans exactly one verify
    // width — the GDN conv cache's layout is a function of it — so `b1s5` against `b1s1` is two RUNS
    // compared pairwise, and inside one run every arm (solo, batched, sweep, serve path, concurrency)
    // is at the same width and therefore comparable with every other.
    let width = forward.facts().max_width.max(1);
    // A copy, because the scheduler takes the runner and the arms below still have to be able to
    // say WHICH ARITHMETIC each shape ran (`ar_band_note`).
    let facts = forward.facts().clone();
    // THE PASS'S OWN DRAFTER, when this run has one. `EngineDrafter` proposes nothing itself: the
    // step makes the proposal (it has to -- the drafter reads the target's tapped hidden states for
    // the rows that step committed, and they live in the arena only until the next step) and this
    // is the mailbox that carries each block to the next step's verify. Without `--draft-pack` the
    // arms drive a MOCK, whose acceptance is the mock's own and says nothing about the model.
    let engine_drafts = forward.facts().draft_resident.is_some();
    // **EVERY PROBED ARM DRIVES A MOCK, and it has to.** A probe digests every ROW of the step,
    // draft rows included, and two runs of the same prompt with the engine's own drafter get two
    // different blocks the moment anything about their session state differs -- a different context
    // slot, a different run of drafter pages, a different history in them. The rows it proposes are
    // then legitimately different residual streams, and an arm that compared them would be
    // measuring the drafter rather than the thing it was written to measure (the slot shift, the
    // twin, the chunk bisect: all claims about the TARGET's determinism). `fixed` proposes the same
    // token every step for every sequence, so the two arms of a probe comparison feed the same rows.
    // The IDS are compared with the real drafter, which is where the equivalence claim lives.
    let mut probe_drafter: Box<dyn Drafter> = if width <= 1 {
        Box::new(NoDrafter)
    } else {
        Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, width - 1))
    };
    let mut run_drafter: Box<dyn Drafter> = if width <= 1 {
        Box::new(NoDrafter)
    } else if engine_drafts {
        Box::new(EngineDrafter::new(width - 1))
    } else {
        Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, width - 1))
    };
    if width > 1 {
        println!(
            "spec      verify width {width}: every arm of this gate runs {width} rows per sequence, \
             with the `{}` drafter",
            run_drafter.name()
        );
        // THE ENGINE'S OWN TIER, when this run asked for one (`--draft-block`). It is not an
        // accounting copy: its bytes are reserved in the byte ledger before the KV pool is derived
        // from what is left, its sessions are admitted and released with the target's, and the
        // engine refused to boot if any of its 107 launch sites had lost its code object. What it
        // does NOT do yet is launch -- so the arms below still drive a mock, and the line says
        // which of the two a reader is looking at.
        if let Some(line) = facts.draft_tier.as_ref() {
            println!("drafter   TIER PLANNED AND BILLED: {line}");
            // ws-s3's plumbing, when this run also carried `--draft-pack`. Every number is the
            // plan's own: the launch counts come off the two phases, the arena size off the
            // layout, the node count off the capture, and the round trips off the captured phase.
            if let Some(r) = facts.draft_resident.as_ref() {
                println!(
                    "          RESIDENT: {} record(s) bound, arena {} MiB in {} region(s), KV \
                     {} blocks x 2 caches of {} MiB, workspaces {} B sized once before the first \
                     capture",
                    r.records,
                    r.arena_bytes >> 20,
                    r.arena_regions,
                    r.kv_blocks,
                    r.kv_cache_bytes >> 20,
                    r.workspace_bytes,
                );
                println!(
                    "          the proposal is TWO PHASES: {} eager launch(es) (the context \
                     store, whose kernargs are a function of the host's acceptance) + {} captured \
                     ({} node(s) in one graph), {} host round trip(s) inside the captured phase",
                    r.eager_launches, r.captured_launches, r.captured_nodes, r.host_round_trips,
                );
            }
            if let Some(t) = facts.aux_tap.as_ref() {
                println!(
                    "          aux tap: {} copies a decode step at layer(s) {:?}, into one \
                     [{} rows, 5 x hidden] bf16 buffer of {} B, issued AT the tap points inside \
                     the step's op list",
                    t.launches, t.layers, t.rows, t.bytes,
                );
            }
            if engine_drafts {
                println!(
                    "          THE PROPOSAL LAUNCHES: every decode step runs it once per live \
                     slot after the verify commits -- {} eager launch(es) with `draft_inputs`' \
                     tables for the committed rows, then the {}-node backbone replayed -- and \
                     hands the k={} block to the `{}` mailbox for the next step's verify. The \
                     ids cannot move: `Greedy` decides and every committed token is the model's \
                     own at a position whose prefix the model also produced",
                    facts
                        .draft_resident
                        .as_ref()
                        .map_or(0, |r| r.eager_launches),
                    facts
                        .draft_resident
                        .as_ref()
                        .map_or(0, |r| r.captured_nodes),
                    width - 1,
                    run_drafter.name()
                );
            } else {
                println!(
                    "          the proposal does NOT launch: this run was started without \
                     `--draft-pack`, so the tier is billed and nothing is uploaded, and the arms \
                     below drive the `{}` mock",
                    run_drafter.name()
                );
            }
        }
        // THE REAL DRAFTER'S ACCOUNTING, on the host, beside the mock that is actually running.
        //
        // Every arm below drives a MOCK, and a reader has no way to tell from the output whether
        // that is because the real one is slower, absent or unfinished. So the tier says what it
        // would cost and why it cannot propose, in the same block, with the launch sites named.
        //
        // THE BLOCK IS NOT THE VERIFY WIDTH. The drafter's `block_size` is
        // `1 + num_speculative_tokens` and the committed dflash rules refuse any block that is not
        // a power of two (`grouped_conv_quant`/`conv_add_rms` index the conv ring with
        // `position & (block_size - 1)`), so it is one of 2, 4, 8, 16 and prod's is 8. The target's
        // verify width is the SEPARATE, narrower number the scheduler's acceptance EMA caps — this
        // run's `width`. So the tier is planned at the smallest legal block that can feed this
        // run's width, and the relation between the two is checked rather than assumed.
        let block = width.max(2).next_power_of_two();
        match saber_engine::draft_tier::DraftTier::plan(
            forward.facts().tp,
            block,
            args.engine.max_ctx,
            args.engine.max_seqs,
        ) {
            Ok(tier) => {
                println!(
                    "drafter   DFlash2 tier at block {block} (num_speculative_tokens={}) feeding \
                     verify width {width}: {} of {} launch(es) covered, {} absent",
                    block - 1,
                    tier.budget(),
                    tier.graph().total_launches(),
                    tier.host_round_trips(),
                );
                if let Err(e) = tier.check_verify_width(width) {
                    println!("          WIDTH: {e}");
                }
                println!("          would hold {}", tier.demand().line());
                if tier.readiness().is_err() {
                    // The names and counts, one per line. `readiness()`'s own message carries the
                    // full reason for each — several hundred characters apiece — which is right for
                    // an error a caller handles and wrong for a gate's output, where it is one
                    // unreadable line. What a reader needs here is WHICH sites and HOW MANY.
                    println!(
                        "          NOT RUNNING: {} site(s) with no code object ({} launch(es)). A \
                         proposal is a CHAIN: the covered {} would run over buffers nothing wrote, \
                         the block would be a function of the arena, the law would reject it, and \
                         the ids would not move while acceptance collapsed",
                        tier.uncovered().len(),
                        tier.host_round_trips(),
                        tier.budget(),
                    );
                    for (name, _, count) in tier.uncovered() {
                        println!("            {name} x{count}");
                    }
                }
            }
            Err(e) => println!("drafter   the DFlash2 tier could not be planned: {e}"),
        }
    }
    // Pages one direct-drive sequence is given. Every term is a run this gate actually drives, and
    // `continuity_tokens` is here because it can be the LONGEST of them: a 192-token continuity run
    // on a 258-token prompt reaches row 450, and a `per_seq` sized from `gen` alone refuses it by
    // name ("a sequence at row 450 holds 21 page(s)") rather than quietly reading page 0.
    let longest_prompt = prompts
        .iter()
        .chain(conc_prompts.iter())
        .chain(continuity_prompts.iter())
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    //
    // A VERIFY STEP COMMITS UP TO `width` TOKENS, so `gen` steps reach `gen * width` rows. Sizing
    // from the step count alone was right for as long as every drafter in this file was a mock
    // whose acceptance is 0; the first run with the engine's OWN proposal in it refused by name at
    // row 528 of a 33-page sequence (ws-s4). Clamped twice, because a bound that cannot be
    // allocated is not a bound: to what the block table covers (`max_ctx`), and to the share of
    // the DERIVED KV pool the direct-drive sequences may take between them.
    let cap_ctx = (args.engine.max_ctx as usize / 16).max(1);
    let cap_pool = (forward.facts().kv_pool_blocks as usize / args.gate_seqs.max(4)).max(1);
    let per_seq = ((args.tokens as usize).max(longest_prompt)
        + (args.gen.max(args.continuity_tokens) + args.batch_gen) * width.max(1) as usize
        + 2)
    .div_ceil(16)
    .min(cap_ctx)
    .min(cap_pool) as u32;
    // HOW A PROMPT IS FED, and it is the scheduler's rule, not this file's: `chunk_for` takes
    // `remaining.min(max_chunk)`, so a prompt is fed in pieces of the LARGEST planned prefill
    // shape. Every direct-drive arm below chunks the same way. At `--prefill-rows 256,2048` that
    // is one chunk for every corpus prompt and no arm's numbers move; at a single 256-row bucket
    // it is two chunks for a 258-token prompt, which is both the shape `multiturn.sh` runs and the
    // only shape in which `done_before > 0` is ever reached here.
    let prefill_chunk = args
        .engine
        .prefill_rows
        .iter()
        .copied()
        .max()
        .unwrap_or(2048)
        .max(1);
    // ---- the chunk bisect, and nothing else (ws-h29) -------------------------------------------
    // It runs alone: the arms below drive four sequences through 400 decode steps each, and this
    // question is about ONE prefill. It also needs its own `--prefill-rows`, so a run that asked
    // for it is a run that was configured for it.
    if args.chunk_bisect > 0 {
        let n = chunk_bisect(&mut forward, args.chunk_bisect, &prompts[0], per_seq);
        drop(forward);
        if let Some(w) = worker.take() {
            println!("tier      {}", w.stop());
        }
        return n;
    }

    // ---- THE WALK, AT EVERY STEP OF A PROD SEQUENCE (ws-f36) ------------------------------------
    //
    // **IT RUNS ALONE, like the chunk bisect above, and for a reason this round paid for.** It was
    // placed among the drafter arms at first, two thirds of the way down the gate, and a KV-page
    // exhaustion in the concurrency arm (`admitting seq703`) ended the run with `?` before it ever
    // ran -- one lock hold, on a card, for an arm that needs nothing any of those arms produce.
    // The replay's inputs are all PROD's: the capture's `aux.cat`, its tables, its cache. So it
    // takes the boot, answers its question and returns.
    //
    // `--draft-replay` is ONE step of three prompts -- 21 walk positions, which cannot tell 0 %
    // from the 4-9 % ws-f35's uniform `r = 0.957` predicts. This is the same replay over a
    // page-carrying sequence capture, every step, so the rate is measured rather than assumed.
    if let Some(dir) = args.engine.draft_seq.as_ref() {
        let out = args
            .engine
            .draft_seq_out
            .clone()
            .unwrap_or_else(|| dir.join("saber"));
        let n = match crate::draft_seq::run(&mut forward, dir, &out) {
            Ok(0) => {
                println!("OK        every replayed step of the sequence proposes prod's own block");
                0
            }
            Ok(n) => {
                println!(
                    "walk      {n} step(s) of the sequence propose a block that is not prod's -- \
                     the rate is the verdict, see the table above"
                );
                0
            }
            Err(e) => {
                fail(&format!("the sequence replay: {e:#}"));
                1
            }
        };
        drop(forward);
        if let Some(w) = worker.take() {
            println!("tier      {}", w.stop());
        }
        return Ok(failures + n);
    }

    // The engine's decode BATCHES, from the (batch, width) pairs — not from `declared`'s row count,
    // which is `batch * width`. Reading the rows gave `[5, 20]` for `--decode-batches 1,4
    // --decode-widths 5` and the batched arm then asked for "batch 5", which was never planned:
    // `no plan for decode at 25 row(s): batch 5 at verify width 5`. A row count is not a bucket, one
    // level up from where that already bit.
    let batches: Vec<u32> = {
        let mut b: Vec<u32> = forward
            .facts()
            .decode_shapes
            .iter()
            .map(|(batch, _)| *batch)
            .collect();
        b.sort_unstable();
        b.dedup();
        b
    };

    // ---- THE SOLO PROBE, ALONE (H44, diagnostic) ------------------------------------------------
    //
    // H43 bisected an engine-level move (TP=1 solo ids `0x3643...` -> `0x7193...`) to
    // `add_rms_quant`'s static-loop object while every unit byte matched. This drives prompt 0 the
    // way the solo arm does (same pages, same drafter) but EAGER with a digest after every op of
    // every layer over all rows, and writes one JSON line a step: the step's committed ids and
    // every (layer, edge, op, row digests). Two builds' files, diffed, name the first op whose
    // output bytes part. It runs alone, like the chunk bisect: nothing else in the gate is needed.
    if let Some(path) = args.solo_probe_out.as_ref() {
        use std::io::Write as _;
        let mut one = vec![admit_and_prefill(
            &mut forward,
            SeqId(100),
            &prompts[0],
            &blocks_for(0, per_seq),
            prefill_chunk,
        )?];
        let rows: Vec<usize> = (0..width as usize).collect();
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("creating the solo probe {}", path.display()))?,
        );
        let first: Vec<i64> = one[0].ids.iter().map(|&t| t as i64).collect();
        writeln!(file, "{}", serde_json::json!({ "step": -1, "ids": first }))?;
        let steps = args.solo_probe_steps.max(1);
        let mut probe_err = None;
        for k in 0..steps {
            let before = one[0].ids.len();
            let raw_here = args.solo_probe_raw_step == Some(k);
            let probes = match decode_probed_depth(
                &mut forward,
                &mut one,
                1,
                width,
                run_drafter.as_mut(),
                1,
                Some(&rows),
                None,
                if raw_here {
                    ProbeDepth::OpsRaw
                } else {
                    ProbeDepth::Ops
                },
            ) {
                Ok(p) => p,
                Err(e) => {
                    probe_err = Some(e);
                    break;
                }
            };
            let committed: Vec<i64> = one[0].ids[before..].iter().map(|&t| t as i64).collect();
            // H44: the raw step's rows, for the points whose op names the filter, one file each
            // (`<out>.raw/p<n>.bin`) with an index of (point, layer, edge, op, row bytes).
            if raw_here {
                let dir = path.with_extension("raw");
                std::fs::create_dir_all(&dir)?;
                let mut index = Vec::new();
                for (n, p) in probes.iter().flatten().enumerate() {
                    if p.raw.is_empty() || !p.op.contains(args.solo_probe_raw_op.as_str()) {
                        continue;
                    }
                    std::fs::write(dir.join(format!("p{n}.bin")), &p.raw)?;
                    index.push(serde_json::json!([n, p.layer, p.edge, p.op, p.row_bytes]));
                }
                std::fs::write(dir.join("index.json"), serde_json::to_string(&index)?)?;
                println!(
                    "probe     step {k}: {} raw point(s) matching {:?} -> {}",
                    index.len(),
                    args.solo_probe_raw_op,
                    dir.display()
                );
            }
            let pts: Vec<serde_json::Value> = probes
                .into_iter()
                .flatten()
                .map(|p| {
                    let digests: Vec<String> = p.rows.iter().map(|r| format!("{r:016x}")).collect();
                    serde_json::json!([p.layer, p.edge, p.op, digests])
                })
                .collect();
            writeln!(
                file,
                "{}",
                serde_json::json!({ "step": k, "ids": committed, "probes": pts })
            )?;
        }
        file.flush()?;
        forward.release(one[0].seq);
        println!(
            "probe     solo prompt 0: {} step(s) op-deep over {} row(s), {} id(s) committed -> {}",
            steps,
            rows.len(),
            one[0].ids.len(),
            path.display()
        );
        if let Some(e) = probe_err {
            fail(&format!("the solo probe: {e:#}"));
            failures += 1;
        }
        drop(forward);
        if let Some(w) = worker.take() {
            println!("tier      {}", w.stop());
        }
        return Ok(failures);
    }

    // ---- solo: every prompt on its own, one row per step ---------------------------------------
    let mut solo: Vec<Vec<TokenId>> = Vec::new();
    let mut solo_ms = Vec::new();
    // PROMPT 0's PROPOSALS, step by step. The graph half of the path-identity comparison: the
    // eager engine below runs the same prompt with the same drafter and the two traces are held
    // against each other BEFORE the ids are, because a proposal that moved is what makes the id
    // lists different lengths and the trace is the only thing that names it.
    let mut solo_trace: Vec<DraftStep> = Vec::new();
    // The concurrency arms' references: the gate's own solos, then one per EXTRA distinct prompt
    // those arms run (ws-p13), each alone at batch 1 exactly as the gate's own are. Their pages
    // reuse the gate's runs (a solo is released before the next is admitted), so the pool share
    // `per_seq` was sized for is never exceeded.
    let mut conc_solo: Vec<Vec<TokenId>> = Vec::new();
    for (i, p) in conc_prompts.iter().enumerate() {
        let mut one = vec![admit_and_prefill(
            &mut forward,
            SeqId(100 + i as u64),
            p,
            &blocks_for(i % args.gate_seqs.max(4), per_seq),
            prefill_chunk,
        )?];
        let steps = if i == 0 { args.gen } else { args.gen.min(16) };
        // THE TRACE IS RESET AFTER THE PREFILLS, not at the arm's start: a prefill chunk passes
        // through the same phases a decode step does (the blob, the submit, the wait, the
        // read-back), so a table taken over both would report `hits` above its own step count and
        // a p50 mixed out of two shapes (ws-h31).
        forward.host_trace().reset();
        let ms = decode_traced(
            &mut forward,
            &mut one,
            1,
            width,
            run_drafter.as_mut(),
            steps.saturating_sub(1),
            (i == 0).then_some(&mut solo_trace),
            None,
            None,
        )?;
        if i == 0 {
            solo_ms = ms;
            if args.engine.host_trace {
                let tp = forward.facts().tp;
                let label = format!("time b1s{width} TP={tp}");
                print!("{}", forward.host_trace().take().table(&label));
            }
        }
        forward.release(one[0].seq);
        let ids = one.remove(0).ids;
        if i < prompts.len() {
            solo.push(ids.clone());
        }
        conc_solo.push(ids);
    }
    // The ids this run produced, as one number, so two RUNS can be compared — which is the only way
    // to ask "does TP=2 give TP=1's tokens?" when the two are different processes on different
    // cards. It is an open question, not a claim: the exact-family reduce is `a + b` in bf16 and the
    // TP=1 GEMM accumulates the same K in one kernel, so the two need not agree in the last bit, and
    // the wht6 family above the size threshold certainly does not. Printed either way.
    let digest = |ids: &[TokenId]| -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in ids {
            for b in (*t as i64).to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    };
    // AGAINST THE MODEL, not against itself. This is the one assertion in the gate that a
    // self-consistent wrong engine cannot pass, and it is the first token — which comes out of
    // prefill, before a single decode step, so a failure here is upstream of everything below it
    // and every arm after it is being read in that light.
    if let Some(r) = reference.as_ref().and_then(|m| m.get(&chosen[0])) {
        match solo[0].first() {
            Some(&got) if got == r.token_id => println!(
                "OK        token 0 is prod's: {} ({:?}), margin {:.2} nats",
                got, r.text, r.margin
            ),
            got => {
                fail(&format!(
                    "token 0 is {:?} and prod's is {} ({:?}) at a margin of {:.2} nats — this is \
                     the PREFILL answer, so it is upstream of every decode arm below",
                    got, r.token_id, r.text, r.margin
                ));
                failures += 1;
            }
        }
    }
    println!(
        "ids       solo: {} token(s), fnv1a {:#018x}, first 12 {:?}",
        solo[0].len(),
        digest(&solo[0]),
        &solo[0][..solo[0].len().min(12)]
    );
    if let Some(path) = args.solo_ids_out.as_ref() {
        let row: Vec<i64> = solo[0].iter().map(|&t| t as i64).collect();
        std::fs::write(path, serde_json::to_string(&row)?)
            .with_context(|| format!("writing the solo ids to {}", path.display()))?;
        println!("ids       solo ids written to {}", path.display());
        // H44: and what each GRAPH step committed and proposed, so the first differing id is
        // named to its step (`<path>.steps.json`, one [committed, proposed] pair a step).
        let steps: Vec<serde_json::Value> = solo_trace
            .iter()
            .map(|st| serde_json::json!([st.committed, st.proposed]))
            .collect();
        let sp = path.with_extension("steps.json");
        std::fs::write(&sp, serde_json::to_string(&steps)?)
            .with_context(|| format!("writing the solo steps to {}", sp.display()))?;
    }
    // THE BLOCKS, not the mean. An acceptance figure is an average over thousands of proposals and
    // two runs can reach the same one from different blocks; this is every block the drafter
    // proposed on this prompt, in step order, as one number — so the SAME prompt run at TP=1 and
    // at TP=2 is a comparison of the proposals themselves and not of their summary. It is printed
    // for the run that has a resident drafter and says so when it does not (ws-s6).
    if !solo_trace.is_empty() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut blocks = 0usize;
        let mut ids = 0usize;
        for st in &solo_trace {
            for b in &st.proposed {
                blocks += 1;
                ids += b.len();
                for t in b {
                    for byte in (*t as u64).to_le_bytes() {
                        h ^= u64::from(byte);
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                // The LENGTH is part of the claim: a refused proposal is an empty block and two
                // runs that differ only in which slot refused would otherwise hash the same.
                h ^= b.len() as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        println!(
            "draft     the blocks this prompt proposed, in step order: {blocks} block(s), {ids} \
             id(s) over {} step(s), fnv1a {h:#018x} — the number two runs at different RANK \
             COUNTS are compared by",
            solo_trace.len()
        );
    }

    let timed: Vec<f64> = solo_ms.iter().copied().skip(args.warmup).collect();
    let b1 = p50(&timed);
    let b1_mean = mean(&timed);
    // The shape this arm actually ran: batch 1 at the run's width, so `width` rows. Asking for
    // `rows: 1` at width 5 found nothing and the report read "0 dispatches/step = 29409.85 µs/launch".
    let decode1 = StepRows {
        phase: Phase::Decode,
        rows: width,
        width,
        samples: true,
        tree: false,
        ragged: false,
    };
    let per_step_launches = forward.declared_launches(decode1).unwrap_or(0);
    println!();
    println!(
        "time      batch 1: {b1:.2} ms/step (p50; mean {b1_mean:.2}) over {} greedy steps at ctx \
         {}..{} ({} warm-up discarded), GRAPH and {}",
        timed.len(),
        prompts[0].len(),
        prompts[0].len() + args.gen,
        args.warmup,
        // Read off the plan rather than asserted: this line said "PARTIAL" unconditionally from
        // ws-l10 until 2026-09-20, which was true for as long as it was true and a lie the hour
        // ws-p4's glue landed.
        if forward.facts().complete() {
            "COMPLETE"
        } else {
            "PARTIAL"
        }
    );
    println!(
        "          {per_step_launches} dispatches/step = {:.2} µs/launch; the P-1b comparator is \
         18.7 ms/step (prod stack, SPEC=0, TP=2, 210 W, GPU idle — docs/LEDGER.md)",
        b1 * 1000.0 / f64::from(per_step_launches.max(1))
    );
    // The SHAPE of the first run, not just its mean. A mean over the first two hundred steps of a
    // process is a mean over whatever the first two hundred steps happen to include, and at TP=2 the
    // difference between this number and the same shape measured later in the same process was 12
    // ms/step (2026-09-20) — which is a transient, not a step time, and a mean hides which.
    {
        let head: f64 = mean(&timed[..timed.len().min(20)]);
        let tail: f64 = mean(&timed[timed.len().saturating_sub(20)..]);
        let mut sorted = timed.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "          per-step spread: min {:.2} / p50 {:.2} / p90 {:.2} / max {:.2} ms; first 20 \
             {head:.2}, last 20 {tail:.2} — a gap here is a WARM-UP. THIS is the boot's step \
             time: the sweep below is the same shape hundreds of steps and several arms later \
             and carries whatever they left behind (ws-t3)",
            sorted.first().copied().unwrap_or(0.0),
            sorted[sorted.len() / 2],
            sorted[sorted.len() * 9 / 10],
            sorted.last().copied().unwrap_or(0.0),
        );
    }

    // ---- the prefill's own residual stream, layer by layer, at the row the head samples ---------
    //
    // A prompt is padded UP into its bucket, and "the padding is inert" is a claim about every op
    // in the body, not a property of the index tables alone. This prints one digest per layer for
    // the chunk's LAST LIVE ROW — the row the head reads — so two runs whose only difference is
    // `--prefill-rows` can be diffed and the first layer that moves is the op that read the
    // padding. It is a report, not an assertion: the number it would have to be asserted against
    // lives in the other run.
    //
    // It is an ASSERTION now, and it is the one this arm could not make on 2026-09-21: the number
    // it has to be asserted against lives in the other BUCKET, not in the other run, so the arm
    // runs the same chunk in every planned prefill shape and holds them against the narrowest
    // one. `--prefill-rows 256,512,1024,2048` is therefore one command and not four, and the
    // padding claim is tested at every band edge the plan crosses at once (ws-p6).
    if forward.facts().tp > 1 {
        // The prefill probe reads one rank's residual, and at TP > 1 that is half of a reduced
        // stream (`device.rs` refuses it by name). The padding claim is a property of the body
        // plan, which is the same object per rank, so it is gated at TP=1 and SAID here rather
        // than failing the whole TP=2 gate on an arm that cannot run.
        println!(
            "prefill   padding arm skipped at TP={}: the residual stream is reduced across ranks \
             and one rank's dump point is half of it; the bucket-invariance claim is gated at TP=1",
            forward.facts().tp
        );
    } else {
        let subject = &prompts[0];
        // The one band edge on this path that is a DIFFERENT KERNEL rather than more padding.
        // Read from the rules table rather than written here, so the gate cannot drift from the
        // number the launcher uses.
        let quant_band = saber_kernels::rules::LaunchRules::prod_mxfp4()
            .map(|r| r.knobs.quant_large_min_m)
            .unwrap_or(0);
        let mut shapes: Vec<u32> = forward
            .facts()
            .declared
            .iter()
            .filter(|(s, _)| s.phase == Phase::Prefill && s.rows >= subject.len() as u32)
            .map(|(s, _)| s.rows)
            .collect();
        shapes.sort_unstable();
        shapes.dedup();
        let mut reference: Option<(u32, Vec<RowProbe>, Option<TokenId>)> = None;
        for rows in shapes {
            forward.admit(SeqId(940), subject.len() as u32, &blocks_for(0, per_seq))?;
            let (mut out, probe) =
                forward.prefill_probe(SeqId(940), subject, 0, true, Some(rows))?;
            forward.propose_greedy(&mut out)?;
            forward.release(SeqId(940));
            println!(
                "prefill   {} live row(s) in the `{:?}` bucket, sampled {:?}; residual stream per \
                 layer at row {} (fnv1a):",
                subject.len(),
                out.rows.rows,
                out.token(),
                subject.len() - 1,
            );
            for chunk in probe.chunks(4) {
                let line: Vec<String> = chunk
                    .iter()
                    .map(|p| {
                        format!(
                            "L{:>3} {:016x}",
                            p.layer,
                            p.rows.first().copied().unwrap_or(0)
                        )
                    })
                    .collect();
                println!("          {}", line.join("  "));
            }
            match &reference {
                None => reference = Some((rows, probe, out.token())),
                Some((base_rows, base, base_token)) => {
                    let first = first_probe_divergence(
                        std::slice::from_ref(base),
                        0,
                        std::slice::from_ref(&probe),
                        0,
                    );
                    // A bucket pair that straddles `quant_large_min_m` resolves a DIFFERENT
                    // `add_rms_quant` instantiation (256 threads against 512), and two thread
                    // counts sum one row's `ssq` in two orders. The difference is a handful of
                    // bf16 roundings in 1.2 M values, measured 2026-09-21 by raising the knob to
                    // 99999: p2048 is then bit-identical to p256 at all 64 layers. It is a band
                    // the rules table declares and prod's launcher has too, so the claim across
                    // it is the TOKEN and not the bytes.
                    let band = quant_band > 0 && rows >= quant_band && *base_rows < quant_band;
                    if let Some((_, layer, edge)) = first {
                        if band && out.token() == *base_token {
                            println!(
                                "OK        prefill padding: p{rows} and p{base_rows} straddle \
                                 `quant_large_min_m` = {quant_band}, so `add_rms_quant` is a \
                                 different instantiation; first layer to move is {layer} \
                                 (`{edge}`) and the sampled token is the same {:?}",
                                out.token(),
                            );
                        } else {
                            fail(&format!(
                                "prefill padding: the same {} live row(s) in the p{rows} bucket \
                                 diverge from p{base_rows} at LAYER {layer} (`{edge}`) — token \
                                 {:?} against {base_token:?}",
                                subject.len(),
                                out.token(),
                            ));
                            failures += 1;
                        }
                    } else if out.token() != *base_token {
                        fail(&format!(
                            "prefill padding: p{rows} and p{base_rows} agree at every layer and \
                             sample {:?} against {base_token:?}",
                            out.token(),
                        ));
                        failures += 1;
                    } else {
                        println!(
                            "OK        prefill padding: p{rows} is bit-identical to p{base_rows} \
                             at every layer, token {:?}",
                            out.token(),
                        );
                    }
                }
            }
        }
    }

    // The three probed arms below (slot shift, twin, census) read one rank's residual stream
    // per layer. At TP > 1 that is half of a reduced stream and `device.rs` refuses the probe by
    // name, so the arms are gated at TP=1 and SAID here; the TP=2 gate keeps its remaining arms.
    if forward.facts().tp > 1 {
        println!(
            "probes    slot-shift, twin and census arms skipped at TP={}: the residual stream is \
             reduced across ranks and one rank's dump point is half of it (gated at TP=1)",
            forward.facts().tp
        );
    } else {
        // ---- the two isolation arms: is it the SLOT, or is it the BATCH? -----------------------------
        //
        // The batched and concurrent arms below have been failing for the sequences after the first
        // (2026-09-21, L14 on main), and "sequence 1 differs from its solo ids at token 5" names three
        // suspects at once: that sequence runs in a different BUCKET (b4s1 against b1s1), on a
        // different GDN state SLOT, and in different KV PAGES. These two arms separate them, and each
        // is one prompt against itself, so nothing else is left in the difference.
        //
        //   * SLOT SHIFT — one sequence, alone, in the b1s1 bucket, run twice: once on the slot it
        //     gets when nothing else is admitted, and once with a filler holding that slot so it lands
        //     on the next one. Same bucket, same rows, same prompt, same pages; only the state slot.
        //   * TWIN — the same prompt admitted TWICE and decoded together, so rows 0 and 1 of one step
        //     feed identical ids out of identically-prepared state. Row 1 must be row 0.
        //
        // Both are probed: every row of the residual stream is digested at every layer, so a failure
        // names the first LAYER that disagrees instead of the token several steps later where it
        // became a different word (`saber_engine::RowProbe`; the walk is eager on both sides).
        let probe_steps = args.gen.clamp(2, 12);
        {
            let subject = &prompts[0];
            let mut a = vec![admit_and_prefill(
                &mut forward,
                SeqId(900),
                subject,
                &blocks_for(0, per_seq),
                prefill_chunk,
            )?];
            let probe_a = decode_probed(
                &mut forward,
                &mut a,
                1,
                width,
                probe_drafter.as_mut(),
                probe_steps,
            )?;
            forward.release(a[0].seq);
            // A filler takes the slot the first run used. It is admitted and never stepped: `admit`
            // reserves a state slot and clears it, which is all this needs of it.
            forward.admit(SeqId(901), 1, &blocks_for(3, per_seq))?;
            let mut b = vec![admit_and_prefill(
                &mut forward,
                SeqId(902),
                subject,
                &blocks_for(0, per_seq),
                prefill_chunk,
            )?];
            let probe_b = decode_probed(
                &mut forward,
                &mut b,
                1,
                width,
                probe_drafter.as_mut(),
                probe_steps,
            )?;
            // Released in the REVERSE of admission, so the free list comes back in the order the arms
            // below expect: it is a LIFO, and handing it back the other way would move every later
            // arm's slots — this arm would then be the thing that perturbed them.
            forward.release(b[0].seq);
            forward.release(SeqId(901));
            let first = first_probe_divergence(&probe_a, 0, &probe_b, 0);
            if a[0].ids == b[0].ids && first.is_none() {
                println!(
                "OK        slot shift: {} ids and every layer of {probe_steps} step(s) identical \
                 on two different state slots",
                a[0].ids.len()
            );
            } else {
                let at = a[0].ids.iter().zip(&b[0].ids).position(|(x, y)| x != y);
                fail(&format!(
                    "slot shift: the same prompt, alone, in the same bucket and the same pages, \
                 answers differently on two state slots — ids differ at {at:?} ({:?} vs {:?}), \
                 first {}",
                    at.and_then(|i| a[0].ids.get(i)),
                    at.and_then(|i| b[0].ids.get(i)),
                    match &first {
                        Some((step, layer, edge)) =>
                            format!("step {step}, layer {layer} (`{edge}`)"),
                        None => "none: every dump point agreed".to_owned(),
                    }
                ));
                failures += 1;
            }
        }
        match batches.iter().copied().find(|b| *b >= 2) {
            None => println!("twin      skipped: no decode bucket at batch 2 or more was planned"),
            Some(tb) => {
                let subject = &prompts[0];
                let mut twins = Vec::new();
                for i in 0..2usize {
                    twins.push(admit_and_prefill(
                        &mut forward,
                        SeqId(910 + i as u64),
                        subject,
                        &blocks_for(i, per_seq),
                        prefill_chunk,
                    )?);
                }
                // Two prefills of one prompt: they must agree before a decode step has run at all.
                if twins[0].ids != twins[1].ids {
                    fail(&format!(
                    "twin: two identical prompts sampled {:?} and {:?} out of PREFILL, which is \
                     upstream of every decode row below",
                    twins[0].ids.first(),
                    twins[1].ids.first()
                ));
                    failures += 1;
                }
                let probe_t = decode_probed(
                    &mut forward,
                    &mut twins,
                    tb,
                    width,
                    probe_drafter.as_mut(),
                    probe_steps,
                )?;
                let row_div = first_probe_divergence(&probe_t, 0, &probe_t, width as usize);
                if twins[0].ids == twins[1].ids && row_div.is_none() {
                    println!(
                        "OK        twin: the same prompt in row 0 and row {width} of a batch-{tb} \
                     step agrees with itself over {probe_steps} step(s), layer by layer"
                    );
                } else {
                    let at = twins[0]
                        .ids
                        .iter()
                        .zip(&twins[1].ids)
                        .position(|(x, y)| x != y);
                    fail(&format!(
                        "twin: two identical prompts in one batch-{tb} step disagree — ids at \
                     {at:?} ({:?} vs {:?}), first {}",
                        at.and_then(|i| twins[0].ids.get(i)),
                        at.and_then(|i| twins[1].ids.get(i)),
                        match &row_div {
                            Some((step, layer, edge)) =>
                                format!("step {step}, layer {layer} (`{edge}`)"),
                            None => "none: every dump point agreed".to_owned(),
                        }
                    ));
                    failures += 1;
                }
                // The other half of the question: row 0 of the batched step against the same prompt
                // ALONE. A difference here while the two rows agree with each other is the BUCKET's
                // arithmetic and not a row index — a different finding and a different fix.
                for (i, t) in twins.iter().enumerate() {
                    let n = t.ids.len().min(solo[0].len());
                    if t.ids[..n] == solo[0][..n] {
                        continue;
                    }
                    let at = t.ids[..n]
                        .iter()
                        .zip(&solo[0][..n])
                        .position(|(x, y)| x != y);
                    fail(&format!(
                    "twin: row {i} of the batch-{tb} step differs from the same prompt run alone \
                     at token {at:?} ({:?} vs {:?})",
                    at.and_then(|k| t.ids.get(k)),
                    at.and_then(|k| solo[0].get(k))
                ));
                    failures += 1;
                    break;
                }
                for t in twins.iter().rev() {
                    forward.release(t.seq);
                }
            }
        }

        // ---- the census: every sequence, alone and in the batch, LAYER BY LAYER ---------------------
        //
        // The two arms above say what the failure is NOT: not the state slot (the same prompt alone on
        // two slots is identical to the last layer) and not a row index (the same prompt in rows 0 and
        // 1 of one step is identical to the last layer, and row 0 is identical to the solo run). What
        // is left is the step with FOUR DIFFERENT prompts in it — different contexts, different page
        // lists, a wider bucket — and "sequence 1 differs at token 5" does not say which.
        //
        // So: each prompt probed alone, then all of them probed together, and the first dump point at
        // which a sequence's row disagrees with its own solo run. A divergence at layer -1 is the
        // EMBEDDING, which means the ids fed in already differed; at layer L it is layer L's output.
        // This is a report, not an assertion — the batched arm below is the assertion — because its
        // job is to say where to look.
        let census_batch = batches
            .iter()
            .copied()
            .filter(|b| *b >= u32::try_from(prompts.len()).unwrap_or(1))
            .min();
        if let Some(cb) = census_batch {
            let mut solo_probe = Vec::new();
            let mut solo_ids = Vec::new();
            for (i, p) in prompts.iter().enumerate() {
                let mut one = vec![admit_and_prefill(
                    &mut forward,
                    SeqId(920 + i as u64),
                    p,
                    &blocks_for(i, per_seq),
                    prefill_chunk,
                )?];
                solo_probe.push(decode_probed(
                    &mut forward,
                    &mut one,
                    1,
                    width,
                    probe_drafter.as_mut(),
                    probe_steps,
                )?);
                forward.release(one[0].seq);
                solo_ids.push(one.remove(0).ids);
            }
            let mut many: Vec<Driven> = Vec::new();
            for (i, p) in prompts.iter().enumerate() {
                many.push(admit_and_prefill(
                    &mut forward,
                    SeqId(930 + i as u64),
                    p,
                    &blocks_for(i, per_seq),
                    prefill_chunk,
                )?);
            }
            let batched_probe = decode_probed(
                &mut forward,
                &mut many,
                cb,
                width,
                probe_drafter.as_mut(),
                probe_steps,
            )?;
            println!(
            "census    each prompt alone against the same prompt in a batch-{cb} step, {probe_steps} \
             probed step(s), eager on both sides:"
        );
            for (i, d) in many.iter().enumerate() {
                let at = d
                    .ids
                    .iter()
                    .zip(&solo_ids[i])
                    .position(|(x, y)| x != y)
                    .map_or("all identical".to_owned(), |k| {
                        format!("token {k} ({:?} vs {:?})", d.ids.get(k), solo_ids[i].get(k))
                    });
                let div =
                    first_probe_divergence(&solo_probe[i], 0, &batched_probe, i * width as usize)
                        .map_or("no dump point differs".to_owned(), |(step, layer, edge)| {
                            format!("step {step}, layer {layer} (`{edge}`)")
                        });
                println!(
                "          seq {i}: prompt {:>5} tokens, ctx {}..{} — ids {at}; first divergence {div}",
                prompts[i].len(),
                prompts[i].len(),
                d.context - 1,
            );
            }
            for d in many.iter().rev() {
                forward.release(d.seq);
            }
        }
    }

    // ---- batched: the same prompts, one step ----------------------------------------------------
    let batch = u32::try_from(prompts.len()).unwrap_or(1);
    let use_batch = batches
        .iter()
        .copied()
        .filter(|b| *b >= batch)
        .min()
        .unwrap_or(batch);
    let mut together: Vec<Driven> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        together.push(admit_and_prefill(
            &mut forward,
            SeqId(200 + i as u64),
            p,
            &blocks_for(i, per_seq),
            prefill_chunk,
        )?);
    }
    // ---- THE BATCHED PROPOSAL'S EQUIVALENCE ARM (ws-s10) ---------------------------------
    //
    // The proposal is a BATCH now — one eager context store over the step's rows and one captured
    // backbone over `reqs * block` query rows — and the claim is that each request's block is the
    // one it would have got proposed alone. That is asked INSIDE this pass, paired: every batched
    // proposal is followed by the same requests proposed one at a time out of the same `aux.cat`
    // rows at the one-request shape, and the two are compared row for row with a cross comparison
    // as the control. It doubles this arm's proposals, which is why it is a switch and why the
    // acceptance figures are taken from a run that does not have it on.
    let batch_equiv = args.engine.draft_batch_check && forward.facts().draft_resident.is_some();
    // The counters as a HANDLE, taken here because the gate hands `forward` to the scheduler a few
    // hundred lines below and the PADDED half of this arm runs on the far side of that move.
    let draft_counters = forward.draft_counter_handle();
    let dc_before = forward.draft_counts();
    if batch_equiv {
        forward.set_draft_batch_check(true);
    }
    let batched_ms = decode_together(
        &mut forward,
        &mut together,
        use_batch,
        width,
        run_drafter.as_mut(),
        args.gen.min(16).saturating_sub(1),
    )?;
    if batch_equiv {
        forward.set_draft_batch_check(false);
        let d = counts_delta(dc_before, forward.draft_counts());
        if d.batch_checked == 0 {
            fail(
                "batched proposal: the equivalence arm ran no comparisons — this pass made no \
                 batched proposal at all",
            );
            failures += 1;
        } else if d.batch_mismatch > 0 {
            fail(&format!(
                "batched proposal: {} of {} request(s) got a DIFFERENT block batched than they \
                 got proposed one at a time",
                d.batch_mismatch, d.batch_checked
            ));
            failures += 1;
        } else {
            println!(
                "OK        batched proposal: {} request-proposal(s) identical row for row to the \
                 SAME request proposed alone, over {} sequences at batch {use_batch}",
                d.batch_checked,
                together.len()
            );
            // THE PAIRED COST, in the same steps: one proposal at M = reqs x block against the
            // same requests' proposals at M = block. F33's per-launch table predicts about 0.53
            // for the twenty-one blockscale GEMMs alone.
            println!(
                "proposal  {} step(s): BATCHED {:.1} us against {:.1} us one at a time = {:.3}x",
                d.batch_timed,
                d.batch_mean_us(),
                d.solo_mean_us(),
                d.batch_ratio()
            );
        }
        // THE CONTROL, and it is the half that makes the line above mean anything: a comparison
        // that says "identical" whatever it is handed proves nothing, so the same blocks are
        // compared ACROSS requests and must not agree.
        if d.batch_cross == 0 {
            println!(
                "          batched proposal CONTROL: not asked — a batch of one has no other \
                 request to compare against"
            );
        } else if d.batch_cross_same == d.batch_cross {
            fail(&format!(
                "batched proposal CONTROL: all {} cross comparison(s) MATCHED, so the arm above \
                 compares nothing",
                d.batch_cross
            ));
            failures += 1;
        } else {
            println!(
                "OK        batched proposal CONTROL: {} of {} cross comparison(s) differ — four \
                 prompts draft four continuations",
                d.batch_cross - d.batch_cross_same,
                d.batch_cross
            );
        }
    }
    println!(
        "batched   {} sequences in a batch-{use_batch} step, {:.2} ms/step = {:.2} ms/step/sequence",
        together.len(),
        mean(&batched_ms),
        mean(&batched_ms) / together.len() as f64
    );
    // THE COMMON PREFIX, with both lengths beside it (S5). This arm ran `gen.min(16) - 1` STEPS
    // and compared `solo[i][..batched.len()]`, which is a slice of a list that is only as long as
    // the SOLO run's own acceptance made it — so under speculation, where a step commits a
    // variable number of tokens, the batched run outrunning the solo one panicked the harness on
    // a range rather than reporting the difference it is there to report (measured at TP=2,
    // ws-s6: 28 batched ids against 26 solo). A length difference is an ACCEPTANCE difference and
    // it is still a failure; it is just a different one from a token difference.
    //
    // The two arms run DIFFERENT NUMBERS OF STEPS on purpose — the solo one decodes `gen` and this
    // one `gen.min(16) - 1` — so the claim is about the PREFIX and never about the length, which
    // is why this is `prefix_agrees` and not `PrefixVerdict::agrees`.
    let mut batched_bad: Vec<(usize, usize)> = Vec::new();
    let mut compared = 0usize;
    for (i, d) in together.iter().enumerate() {
        let v = PrefixVerdict::of(&d.ids, &solo[i]);
        compared = compared.max(v.common);
        if v.prefix_agrees() {
            continue;
        }
        let at = v.first_diff.unwrap_or(0);
        println!(
            "batched   sequence {i} differs at token {at} batched ({:?}) vs solo ({:?}); {} id(s) \
             compared, batched {} solo {} — attributing it",
            d.ids.get(at),
            solo[i].get(at),
            v.common,
            v.len_a,
            v.len_b
        );
        batched_bad.push((i, at));
    }
    // THE ARM'S HAZARD (ws-p11): every sequence's divergence over the positions it reached, which
    // is what a DECIDED flip among them is held against. Printed red or green: a rate is a
    // measurement, and a gate that only states one when something failed has no baseline for it.
    let batched_rate = ArmRate::of(
        together
            .iter()
            .enumerate()
            .map(|(i, d)| (&d.ids[..], &solo[i][..])),
    )
    .in_flight(together.len());
    println!("batched   the arm's hazard: {}", batched_rate.line());
    // B1: each batched sequence's ids as one digest, so two boots that ran the same prompts at
    // DIFFERENT captured batches (the ladder's b5 against the control's padded b8) compare row by
    // row: equal digests are the bucket not moving a row's committed tokens.
    println!(
        "batched   ids at batch-{use_batch}: {}",
        together
            .iter()
            .map(|d| format!("{}:{:#018x}", d.ids.len(), digest(&d.ids)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if batched_bad.is_empty() {
        println!(
            "OK        {} sequences x {compared} ids identical batched and solo (the common \
             prefix: the two arms decode different numbers of STEPS by design, and under \
             speculation that is different numbers of TOKENS)",
            together.len(),
        );
    }
    for d in &together {
        forward.release(d.seq);
    }
    // The GDN-side control needs a verify width to get wrong (at width 1 `committed_before` is 1
    // and there is no other in-range value) and a batch to put the defect in.
    let batch_control = width > 1 && prompts.len() > 1 && use_batch > 1 && !args.no_attribute;
    // THE ATTRIBUTION IS NOT FREE TO THE ARMS AFTER IT, and `--no-attribute` is the control that
    // says so (ws-t3). Every arm downstream of it in the same boot — the sweep, the mock
    // drafters, the serve path — read **+2.7 ms/step** at `b1s8` on `main` while this binary's
    // own timed arm, taken before it, was unmoved and the fresh eager engine at the end of the
    // same process was unmoved. The red is still REPORTED without it; only the attribution of it
    // is skipped.
    if args.no_attribute && !batched_bad.is_empty() {
        println!(
            "batched   {} red(s) NOT attributed (--no-attribute): this boot's later timings are \
             the control for what the attribution costs them (ws-t3)",
            batched_bad.len()
        );
        batched_bad.clear();
    }
    // ---- AND WHAT EACH OF THOSE IS: ws-p9's ATTRIBUTION, DEFERRED TO THE END OF THE GATE --------
    //
    // It used to run here, where the arm found the red. That cost every timed arm after it in the
    // same boot, and nothing said so: paired in one hold on one binary, `b1s8`'s `sweep` read
    // **26.76 ms/step** with the attribution upstream of it and **24.11** with `--no-attribute`,
    // and `b4s8` **36.55** against **33.05** — which is ws-s11's own recorded 33.06. The `time`
    // arm, taken BEFORE it either way, read 24.63 / 25.09 across the same pair, and the eager arm
    // — a FRESH engine at the end of the same process — was unmoved at both. So the step never
    // moved; the measurement did, and four rounds' `b1s8 (sweep p50)` figures were read through
    // it (ws-f35's 26.90 is this, reproduced twice here at 26.93 / 26.84).
    //
    // **A DIAGNOSTIC MUST NOT BE UPSTREAM OF A PUBLISHED NUMBER.** The attribution needs nothing
    // the arm leaves behind — it re-admits its own sequences and re-prefills them — so it runs at
    // the end of the gate now, beside the concurrency attribution that was already there, after
    // the sweep, the mock drafters and the serve path. The reds are still reported HERE, at the
    // arm that found them, with the token they differ at.
    if !batched_bad.is_empty() {
        println!(
            "batched   {} red(s) recorded; attributed at the END of the gate, after every timed \
             arm (ws-t3)",
            batched_bad.len()
        );
    }

    // ---- the AUX TAP, at b1 and at this run's batch ----------------------------------------------
    //
    // The drafter's one non-weight input, gated against the engine's own probe of the five edges
    // the copies read — an EXTERNAL reference in the only sense available here, since the probe
    // reads the planner's buffer at the planner's own point and the tap reads it through a kernel.
    // The control (one layer late) is what says the gate can tell where the copy ran.
    // THE TAP GATE IS TP=1's, and the engine refuses it at two ranks by name (ws-s6): its step
    // deliberately proposes nothing, and a step that proposes nothing while the other rank waits
    // for a proposal frame is a pair that never speaks again. Said here rather than skipped
    // silently, so a TP=2 run's output states which arm did not run and why.
    if forward.facts().aux_tap.is_some() && args.engine.world > 1 {
        println!(
            "aux tap   SKIPPED at TP {}: the tap's claim is about a COPY inside one rank's step \
             and its control feeds `aux.cat` a layer the drafter was never trained on, so its \
             step makes no proposal — which at two ranks would leave the follower waiting. It is \
             green at TP=1 in the same round.",
            args.engine.world
        );
    }
    if forward.facts().aux_tap.is_some() && args.engine.world == 1 {
        for b in [1u32, use_batch] {
            if b > args.engine.max_seqs || !batches.contains(&b) {
                continue;
            }
            let mut live = Vec::new();
            for i in 0..b as usize {
                let p = &prompts[i % prompts.len()];
                live.push(admit_and_prefill(
                    &mut forward,
                    SeqId(700 + i as u64),
                    p,
                    &blocks_for(i, per_seq),
                    prefill_chunk,
                )?);
            }
            let (bad, lines) = aux_tap_arm(&mut forward, &mut live, b, width)?;
            println!("aux tap   b{b}s{width}: `aux.cat` against the engine's own probe");
            for l in &lines {
                println!("{l}");
            }
            if bad == 0 {
                println!(
                    "OK        all {} tap(s) bit-exact at b{b}s{width}, and the control fires",
                    forward.facts().aux_tap.as_ref().map_or(0, |t| t.launches)
                );
            } else {
                fail(&format!("the aux tap at b{b}s{width}: {bad} arm(s) red"));
                failures += 1;
            }
            for d in &live {
                forward.release(d.seq);
            }
        }
    }

    // ---- the PROPOSAL, against a capture's own inputs, through the engine's launcher -------------
    if let Some(dir) = args.engine.draft_replay.as_ref() {
        match crate::draft_replay::run(&mut forward, dir, args.engine.draft_chain_dump.as_deref()) {
            Ok(0) => println!(
                "OK        the proposal's plumbing through the engine reproduces the staged arm"
            ),
            Ok(n) => {
                fail(&format!("the proposal replay: {n} site(s) red"));
                failures += n;
            }
            Err(e) => {
                fail(&format!("the proposal replay: {e:#}"));
                failures += 1;
            }
        }
    }

    // ---- THE FIRST PROPOSAL OF A REQUEST, as ids, for a diff against PROD's (ws-s9) -------------
    //
    // Prod's `propose` runs after every target step, so its FIRST decode step verifies a block made
    // out of the LAST PREFILL CHUNK. `the internal reference harness --seq` records that block
    // for three corpus prompts (`<tag>.s1/out.draft_tokens`); this prints the engine's own for the
    // same prompts, so the two can be diffed off the card. It is an EXTERNAL comparison and the
    // only one this round has for the block itself: `--draft-replay` seeds a capture whose context
    // rows are the drafter's block, and a prefill's are the whole prompt.
    //
    // UNSEEDED on purpose, which is also its limit: the engine's target activations are its own,
    // so a prompt whose first block differs may be differing at the known projection-ULP class
    // rather than at the law. The number is reported as what it is.
    if !args.first_block_ids.trim().is_empty() {
        println!();
        println!(
            "first-block  the block each prompt's LAST PREFILL CHUNK proposed (ws-s9), as ids"
        );
        for tok in args.first_block_ids.split(',') {
            let Ok(idx) = tok.trim().parse::<usize>() else {
                continue;
            };
            let prompt: Vec<TokenId> = match crate::forward::read_prompt_ids(&args.prompt_ids, idx)
            {
                Ok(v) => v.iter().map(|x| *x as TokenId).collect(),
                Err(e) => {
                    fail(&format!("corpus prompt {idx}: {e:#}"));
                    failures += 1;
                    continue;
                }
            };
            let seq = SeqId(800_000 + idx as u64);
            match admit_and_prefill(
                &mut forward,
                seq,
                &prompt,
                &blocks_for(0, per_seq),
                prefill_chunk,
            ) {
                Ok(d) => {
                    println!(
                        "             p{idx}  prompt {} tok, sampled {}, block {:?}",
                        prompt.len(),
                        d.ids.first().copied().unwrap_or_default(),
                        d.first_draft
                    );
                    forward.release(seq);
                }
                Err(e) => {
                    fail(&format!("corpus prompt {idx}: {e:#}"));
                    failures += 1;
                }
            }
        }
    }

    // ---- ACCEPTANCE: what this run's drafter EARNS, over the corpus -----------------------------
    //
    // The one number the id arms cannot see. Reported with the empty-mailbox count beside it,
    // because "the drafter proposed and the model disagreed" and "the drafter never proposed" are
    // the same zero in an acceptance rate and are not the same fact.
    if args.accept_prompts > 0 && width > 1 {
        let mut corpus: Vec<Vec<TokenId>> = Vec::with_capacity(args.accept_prompts);
        for r in 0..args.accept_prompts {
            match crate::forward::read_prompt_ids(&args.prompt_ids, r) {
                Ok(v) => corpus.push(v.iter().map(|x| *x as TokenId).collect()),
                Err(_) => break,
            }
        }
        let want = args.accept_tokens.max(1);
        // TWO PASSES, and the second one is the ATTRIBUTION rather than a second measurement.
        //
        // The drafter attends its own K/V over `last_valid_pos + 1 + block` positions, and this
        // engine's context store runs on DECODE steps only -- the aux tap is decode-only by name
        // (ws-s3's open item 4). So on a 250-token prompt the drafter's window is ~250 positions
        // of cache nothing wrote plus the handful this run committed, and acceptance is a function
        // of that ratio and not of the drafter. Running the same corpus TRUNCATED to a few tokens
        // makes the ratio the other way round: after `want` committed tokens almost the whole
        // window is real. Two numbers from one hold, and the difference between them is the size
        // of the missing prefill store.
        let mut arms: Vec<(String, Vec<Vec<TokenId>>, bool)> = vec![(
            format!("whole prompt x {want} token(s)"),
            corpus.clone(),
            true,
        )];
        if args.accept_short > 0 {
            arms.push((
                format!(
                    "ATTRIBUTION: the same prompts truncated to {} token(s) x {want}",
                    args.accept_short
                ),
                corpus
                    .iter()
                    .map(|p| p[..args.accept_short.min(p.len())].to_vec())
                    .collect(),
                true,
            ));
        }
        // THE CONTROL (ws-s9), in the SAME boot so the comparison is paired. With the last prefill
        // chunk's proposal off, a request's first decode step finds the mailbox empty and runs at
        // width 1 — which is what every request in this tree did until this round — so the
        // difference between this arm and the first is what the first proposal earns. It is a
        // control in the strict sense too: it must READ WORSE, and `empty` must rise by exactly
        // one per prompt.
        if args.accept_control > 0 {
            arms.push((
                format!(
                    "CONTROL: the first {} prompt(s) with NO proposal out of the prefill x {want}",
                    args.accept_control.min(corpus.len())
                ),
                corpus[..args.accept_control.min(corpus.len())].to_vec(),
                false,
            ));
        }
        let mut first_arm = true;
        for (label, corpus, first_proposal) in arms {
            if let Err(e) = forward.set_prefill_proposal(first_proposal) {
                fail(&format!("the first-proposal switch: {e:#}"));
                failures += 1;
                break;
            }
            println!();
            println!(
                "accept    {} corpus prompt(s), {label}, at b1s{width} with the `{}` drafter",
                corpus.len(),
                run_drafter.name()
            );
            let before = forward.draft_counts();
            let mut sink =
                (first_arm && args.draft_lattice_dump.is_some()).then(LatticeSink::default);
            match acceptance_arm(
                &mut forward,
                &corpus,
                per_seq,
                prefill_chunk,
                width,
                run_drafter.as_mut(),
                want,
                args.engine.max_ctx,
                sink.as_mut(),
            ) {
                Ok(run) => {
                    if let (Some(sk), Some(dir)) = (sink.as_ref(), args.draft_lattice_dump.as_ref())
                    {
                        match sk.write(dir) {
                            Ok(()) => println!(
                                "          lattice   {} record(s), {} odd proposal(s) -> {}",
                                sk.records,
                                sk.odd,
                                dir.display()
                            ),
                            Err(e) => {
                                fail(&format!("the lattice dump: {e:#}"));
                                failures += 1;
                            }
                        }
                    }
                    let after = forward.draft_counts();
                    println!(
                    "          {} prompt(s) ran: {} step(s), {} token(s) committed = {:.3} per \
                     step; {} step(s) carried a proposal, {} found the mailbox empty",
                    run.prompts,
                    run.steps,
                    run.committed,
                    run.per_step(),
                    run.drafted,
                    run.empty
                );
                    println!(
                        "          ACCEPTED {} of {} draft(s) = {:.3} accepted per draft",
                        run.accepted,
                        run.drafted,
                        run.per_draft()
                    );
                    let k_pos = run.hist.len();
                    let dist: Vec<String> = run
                        .hist
                        .iter()
                        .enumerate()
                        .map(|(i, n)| {
                            let pct = if run.drafted == 0 {
                                0.0
                            } else {
                                100.0 * *n as f64 / run.drafted as f64
                            };
                            format!("{i}:{n} ({pct:.1}%)")
                        })
                        .collect();
                    println!("          distribution  {}", dist.join("  "));
                    // THE SURVIVAL CURVE, beside the PMF, because prod exports the first and
                    // Saber's arm has only ever printed the second — and the paired reading of
                    // the two is the one conditional curve that is NOT confounded by the two
                    // stacks writing different text (ws-f36, over `prod_spec_accept.py`'s
                    // `per_pos`). `survival[k]` is the drafts whose first `k+1` tokens were ALL
                    // accepted, which is prod's own definition, and the CONDITIONAL rate below
                    // it is the only normalisation that can carry a shape (blind spot 18).
                    let mut survival: Vec<u64> = Vec::with_capacity(k_pos);
                    let mut left: u64 = run.drafted;
                    // `hist` has `k + 1` buckets and the last survival entry would always be
                    // zero (a draft cannot accept more than `k`), so the curve is `k` long —
                    // which is the length prod's own `..._per_pos_total` exports.
                    for n in run.hist.iter().take(k_pos.saturating_sub(1)) {
                        left -= *n;
                        survival.push(left);
                    }
                    let mut entering = run.drafted;
                    let cond: Vec<String> = survival
                        .iter()
                        .map(|s| {
                            let c = if entering == 0 {
                                0.0
                            } else {
                                *s as f64 / entering as f64
                            };
                            entering = *s;
                            format!("{c:.4}")
                        })
                        .collect();
                    println!(
                        "          survival      {}",
                        survival
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                    println!("          conditional   {}", cond.join(" "));
                    println!(
                    "          engine    proposals {}, tokens {}, refused {}, out of range {}, \
                     eager launch(es) {}, replayed node(s) {}",
                    after.proposals - before.proposals,
                    after.tokens - before.tokens,
                    after.refused - before.refused,
                    after.out_of_range - before.out_of_range,
                    after.eager_launches - before.eager_launches,
                    after.replayed_nodes - before.replayed_nodes,
                );
                    // WHICH ROUTE THIS ARM TOOK, at the site rather than from the flag
                    // (`CLAUDE.md` rule 5). An arm that claims the first proposal and shows zero
                    // of them is a switch that did not reach the engine.
                    println!(
                        "          prefill   first proposal(s) {}, silent last chunk(s) {}, \
                         context {} tile(s) / {} row(s)",
                        after.prefill_proposals - before.prefill_proposals,
                        after.prefill_silent - before.prefill_silent,
                        after.ctx_tiles - before.ctx_tiles,
                        after.ctx_rows - before.ctx_rows,
                    );
                    // AND HOW THAT CONTEXT WAS TILED (H35): wide and narrow, and under
                    // `--draft-store-check` the byte-for-byte verdict on THIS rank.
                    let st = counts_delta(before, after);
                    println!("          store     {}", st.store_line());
                    // P27-12: WHICH CHAIN the proposals committed to, at the site (rule 5).
                    println!(
                        "          selrank   form {}, re-ranked {} block(s), moved {} from the walk",
                        args.engine.selector_rank.as_str(),
                        st.selector_reranked,
                        st.selector_moved,
                    );
                    // P27-15c: THE TREE ARM, at the site (rule 5): the form asked, the form that
                    // ran (a refusal names itself), and the plans the lattice reads built.
                    if args.engine.spec_tree != saber_engine::spec_tree::SpecTreeForm::Off {
                        let arm = saber_engine::device::spec_tree_arm(&args.engine.config());
                        println!("          {}", arm.line());
                        println!("          {}", st.spectree.line());
                    }
                    if st.store_check_diff > 0
                        || st.store_check_blind > 0
                        || st.store_check_class_silent > 0
                    {
                        fail(&format!(
                            "the wide context store is not the narrow one: {}",
                            st.store_line()
                        ));
                        failures += 1;
                    }
                    if after.out_of_range > before.out_of_range {
                        fail("the drafter's walk wrote an id outside the vocabulary");
                        failures += 1;
                    }
                    // THE COMMITTED IDS, for the TARGET half of the deficit (ws-f36). Only the
                    // first arm writes them: the attribution pairs this engine's greedy text
                    // against prod's over the SAME prompts, and a truncated or control arm is a
                    // different run of different prompts.
                    if let (Some(path), true) = (args.accept_ids.as_ref(), first_arm) {
                        let mut out = String::new();
                        for (n, (i, ids)) in run.ids.iter().enumerate() {
                            // H46: each step's accepted count and proposal beside the ids.
                            let (acc, drafts): (Vec<u32>, Vec<&Vec<TokenId>>) = run
                                .trace
                                .get(n)
                                .map(|t| t.iter().map(|(a, d)| (*a, d)).unzip())
                                .unwrap_or_default();
                            out.push_str(&format!(
                                "{}\n",
                                serde_json::json!({
                                    "row": i, "ids": ids, "acc": acc, "drafts": drafts
                                })
                            ));
                        }
                        match std::fs::write(path, out) {
                            Ok(()) => println!(
                                "          ids       wrote {} prompt(s) of committed ids to {}",
                                run.ids.len(),
                                path.display()
                            ),
                            Err(e) => {
                                fail(&format!("writing {}: {e}", path.display()));
                                failures += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    fail(&format!("the acceptance arm: {e:#}"));
                    failures += 1;
                }
            }
            first_arm = false;
        }
        // The switch goes back whatever the arms did: every arm after this one — the sweep, the
        // dial, the serve path — is the engine as it ships.
        if let Err(e) = forward.set_prefill_proposal(true) {
            fail(&format!("restoring the first-proposal switch: {e:#}"));
            failures += 1;
        }
        // H46 (diagnostic): a short attribution run ends here, the way the solo probe does.
        if args.stop_after_accept {
            println!("accept    --stop-after-accept: the gate ends after the acceptance pass");
            drop(forward);
            if let Some(w) = worker.take() {
                println!("tier      {}", w.stop());
            }
            return Ok(failures);
        }
    }

    // ---- the batch sweep ------------------------------------------------------------------------
    //
    // **THIS ARM'S p50 IS NOT THE STEP TIME, IT IS THIS BOOT'S STEP TIME AT THIS POINT IN THE
    // BOOT**, and until ws-t3 it was the one reading in the gate that said neither. The timed arm
    // above prints its own min/p50/p90/max and calls a gap a warm-up; this printed one number,
    // named it "the steady state" in the timed arm's own comment, and is the number four rounds
    // quoted as `b1s8 ms/step`. It is taken hundreds of steps and several arms later, so anything
    // an earlier arm leaves behind is IN it: ws-t3 paired ws-s9's tree against `main` A B A B in
    // one hold and read the timed arm at 24.87 / 24.92 (unmoved, inside the floor) with this arm
    // at 24.22 / 26.89 — +2.67 ms that is not the step and not the session, but ws-p9's bucket
    // attribution, which runs in between on `main` and not on ws-s9's tree.
    //
    // So this arm prints the same spread the timed arm does, and says what its distance from the
    // timed arm means. A p50 here that is far from the timed arm's is a fact about the BOOT.
    let mut sweep: Vec<(u32, f64, f64, f64, f64, f64)> = Vec::new();
    for b in batches.iter().copied() {
        if b > args.engine.max_seqs {
            continue;
        }
        // The sweep holds `b` sequences at once for `batch_gen` steps: its own need, and its own
        // share of the pool when `b` runs of `per_seq` do not fit (H44, `pages_for_live`).
        let sweep_longest = (0..b as usize)
            .map(|i| prompts[i % prompts.len()].len())
            .max()
            .unwrap_or(0);
        let need =
            (sweep_longest + (args.batch_gen + 1) * width.max(1) as usize + 2).div_ceil(16) as u32;
        let pool = forward.facts().kv_pool_blocks;
        let sweep_pages = match pages_for_live(b as usize, per_seq, pool, need) {
            Ok(n) => n,
            Err(e) => {
                fail(&format!("sweep b{b}s{width}: {e}; the batch is not run"));
                failures += 1;
                continue;
            }
        };
        if sweep_pages != per_seq {
            println!(
                "sweep     b{b}: {b} x {per_seq} page(s) exceed the pool of {pool} block(s); each \
                 sequence takes {sweep_pages} (this arm reaches {need})"
            );
        }
        let mut live = Vec::new();
        for i in 0..b as usize {
            let p = &prompts[i % prompts.len()];
            live.push(admit_and_prefill(
                &mut forward,
                SeqId(300 + i as u64),
                p,
                &blocks_for(i, sweep_pages),
                prefill_chunk,
            )?);
        }
        forward.host_trace().reset();
        let ms = decode_together(
            &mut forward,
            &mut live,
            b,
            width,
            run_drafter.as_mut(),
            args.batch_gen,
        )?;
        if args.engine.host_trace {
            let tp = forward.facts().tp;
            let label = format!("sweep b{b}s{width} TP={tp}");
            print!("{}", forward.host_trace().take().table(&label));
        }
        let timed: Vec<f64> = ms.iter().copied().skip(args.warmup).collect();
        let mut sorted = timed.clone();
        sorted.sort_by(f64::total_cmp);
        let at = |q: f64| -> f64 {
            sorted
                .get(((sorted.len() as f64 - 1.0) * q).round() as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        sweep.push((b, mean(&timed), p50(&timed), at(0.0), at(0.9), at(1.0)));
        for d in &live {
            forward.release(d.seq);
        }
    }
    println!();
    println!(
        "sweep     GRAPH, {} steps per batch (mean / p50):",
        args.batch_gen
    );
    for (b, ms, med, lo, p90, hi) in &sweep {
        println!(
            "          batch {b}: {ms:.2} / {med:.2} ms/step, {:.2} ms/step/sequence at the p50",
            med / f64::from(*b)
        );
        println!(
            "          batch {b}: per-step spread min {lo:.2} / p50 {med:.2} / p90 {p90:.2} / \
             max {hi:.2} ms — this arm runs LATE in the boot, so its distance from the `time` \
             line above is a fact about the boot and not about the step (ws-t3)"
        );
    }

    // ---- SPECULATION: what each mock drafter costs and commits, at this run's width -------------
    //
    // What this measures, and what it deliberately does NOT assert.
    //
    // MEASURES: ms/step and tokens/step per drafter at the run's verify width. Tokens/step is the
    // number P5 multiplies by a real drafter's acceptance, and ms/step at `b1s{width}` against a
    // `--decode-widths 1` run of the same tree is the cost of `width` rows against one.
    //
    // ASSERTS, once the pass is COMPLETE, that a verified run commits the SPEC=0 run's ids, and
    // — the arm that matters — that acceptance is the DIAL. The second one is not a refinement of
    // the first: the law commits `rows[accepted - 1]`, so a pass in which only row 0 answers
    // correctly commits row 0's answer every step, which IS the SPEC=0 token. Perfect ids, zero
    // acceptance. Measured here 2026-09-20 at b1s5: `oracle 4/4` accepted 1.80 of 4 per step with 89
    // matching ids in front of it.
    //
    // It was reported rather than asserted for one day, while `qk_norm_rope` and `attn_gate_quant`
    // had no code object: their output edges were never written, and with `reuse_layer_slots` on they
    // were arena buffers the step's other rows had written into, so row 0's token depended on the
    // DRAFT — four mocks, four different tokens at the same position. ws-p4's glue closed that; when
    // a pass is still partial this prints the reason instead of failing on arithmetic over arena
    // garbage.
    //
    // Read the id arms with the run's own output in hand. They compare a token sequence, and on this
    // pass that sequence is `<|im_end|>`, `\n`, `<|im_start|>` repeating — the model ends its turn on
    // token 0 of a GSM8K prompt and the gate keeps asking. Every divergence any arm has found is
    // between those same two delimiters, whose bf16 logits in that loop are as close as two logits
    // get, so an id comparison across two reduction orders is partly measuring the argmax's
    // tie-break. See LEDGER, ws-l14, "the acceptance dial is the gate".
    if width > 1 {
        // Named here because it is the reason the ids are reported rather than asserted.
        let partial_pass: String = forward
            .facts()
            .supplied
            .iter()
            .map(|(op, n)| format!("{op}x{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!();
        println!(
            "spec      drafters at b1s{width}, {} steps each. Tokens/step is the number a real \\
             drafter's acceptance multiplies; the ids are NOT compared across drafters — see below.",
            args.batch_gen
        );
        let k = width - 1;
        // Built FRESH for each run rather than re-armed, so "the same drafter twice" means the same
        // drafter in the same state twice — a re-armed oracle whose cursor had moved would be a
        // different drafter wearing the same name.
        let script: Vec<TokenId> = solo[0][1..].to_vec();
        let build = |dial: Option<u32>| -> Box<dyn Drafter> {
            match dial {
                None => Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, k)),
                Some(d) => {
                    let mut r = RecordedDrafter::new(k, d);
                    r.record(SeqId(600), script.clone());
                    Box::new(r)
                }
            }
        };
        let mut dials: Vec<(String, Option<u32>)> = vec![
            ("fixed".to_owned(), None),
            (format!("oracle 0/{k}"), Some(0)),
        ];
        if k >= 2 {
            dials.push((format!("oracle {}/{k}", k / 2), Some(k / 2)));
        }
        dials.push((format!("oracle {k}/{k}"), Some(k)));
        // Each mock twice, and the two runs must agree token for token. A verified step is a graph
        // replay over a recurrent state and a per-candidate snapshot pick; if any of that were
        // reading uninitialised state or racing, the SECOND run of the same drafter would differ.
        // This is the id assertion that survives a partial pass, and it is not a weak one.
        for (name, dial) in dials {
            let mut runs: Vec<(Vec<TokenId>, f64, f64, u64)> = Vec::with_capacity(2);
            for _ in 0..2 {
                let mut drafter = build(dial);
                let mut one = vec![admit_and_prefill(
                    &mut forward,
                    SeqId(600),
                    &prompts[0],
                    &blocks_for(0, per_seq),
                    prefill_chunk,
                )?];
                let ms = decode_together(
                    &mut forward,
                    &mut one,
                    1,
                    width,
                    drafter.as_mut(),
                    args.batch_gen,
                )?;
                let timed: Vec<f64> = ms.iter().copied().skip(args.warmup).collect();
                let d = one.remove(0);
                forward.release(d.seq);
                let steps = ms.len() as f64;
                runs.push((d.ids, mean(&timed), p50(&timed), d.accepted));
                let _ = steps;
            }
            let (ids, m, med, accepted) = runs[0].clone();
            let tokens_per_step = ids.len().saturating_sub(1) as f64 / args.batch_gen as f64;
            println!(
                "          {name:>16}: {m:.2} / {med:.2} ms/step, {tokens_per_step:.2} tokens/step, \\
                 {:.2} ms/token, {accepted} draft token(s) accepted over {} step(s)",
                med / tokens_per_step.max(1e-9),
                args.batch_gen
            );
            // THE DIAL, asserted — the arm the ids cannot replace.
            //
            // A truthful position IS the token the model is about to produce, so the model agrees and
            // the law accepts it: with the oracle dialled to `d`, every step accepts EXACTLY `d` and
            // commits `d + 1`. Proven on the host over a stub whose model is a true function of the
            // fed history (`draft::tests::a_verified_run_commits_the_spec0_run_whatever_the_draft`,
            // widths 1..5 x every dial), so a device that accepts fewer is a device whose rows 1..k
            // do not answer what a SPEC=0 step answers.
            //
            // This is the whole reason the mock exists, and it catches what the ids CANNOT: the law
            // commits `rows[accepted - 1]`, so a pass where only row 0 is right commits row 0's
            // answer every step — the SPEC=0 token — and the ids match while acceptance collapses.
            // Reported but not asserted until 2026-09-20; the b1s5 arm was accepting 1.8 of 4 with
            // 89 matching ids in front of it.
            //
            // The bound is what the script can pay for: it is `solo[0][1..]`, so once a sequence's
            // cursor is within `dial` of its end the oracle proposes the sentinel and is rightly
            // rejected. Step `j` sits at cursor `j * (dial + 1)`.
            // The ids this run committed against the SPEC=0 run's, computed here because BOTH arms
            // below need it: the acceptance arm to know how many steps the script still describes,
            // and the identity arm to say where they parted.
            let n = ids.len().min(solo[0].len());
            let first_div = ids[..n].iter().zip(&solo[0][..n]).position(|(a, b)| a != b);
            if let Some(d) = dial {
                let scripted = if d == 0 || script.len() < d as usize {
                    0
                } else {
                    ((script.len() - d as usize) / (d as usize + 1) + 1).min(args.batch_gen)
                };
                // AND THE SCRIPT DESCRIBES THE RUN ONLY WHILE THE RUN IS THE SPEC=0 RUN. The oracle
                // reads `solo[0][1..]`, so the step after the ids part company is drafting from a
                // continuation this run is no longer on: its proposals are then wrong by
                // construction and rightly rejected, which is a fact about the script and not
                // about rows 1..k. Each step commits `d + 1` tokens from index 1, so the
                // divergence at token `at` leaves `(at - 1) / (d + 1)` steps the script covered
                // whole. Without this the arm charges a run for the steps AFTER the one flip the
                // identity arm below reports (ws-l17: `oracle 4/4` accepted 229 where 228 is four
                // per step over the 57 steps before token 287).
                let valid = first_div
                    .map(|at| at.saturating_sub(1) / (d as usize + 1))
                    .unwrap_or(scripted);
                let owed_steps = scripted.min(valid);
                let owed = d as u64 * owed_steps as u64;
                if accepted < owed {
                    fail(&format!(
                        "{name}: {accepted} draft token(s) accepted where the dial owes {owed} \
                         ({d} per step over the {owed_steps} step(s) the script covers) — every \
                         truthful draft IS the model's own next token, so rows 1..{k} of the verify \
                         step are not answering what a SPEC=0 step answers. The ids cannot see this: \
                         the law commits row 0's answer, which is right either way."
                    ));
                    failures += 1;
                } else {
                    println!(
                        "          {:>16}  acceptance is the dial: {accepted} over the {owed_steps} \
                         scripted step(s), {d} per step",
                        ""
                    );
                }
            }
            // THE IDENTITY, asserted again now that the pass is COMPLETE.
            //
            // At temperature 0 every committed token is the model's own at a position whose entire
            // prefix the model also produced, so a verified run commits the SPEC=0 run's tokens
            // whatever the drafter guessed — all right, all wrong, or right for the first `dial` of
            // them. Speculation buys STEPS, never tokens.
            //
            // It was reported rather than asserted for one day, while `qk_norm_rope` and
            // `attn_gate_quant` had no code object: their output edges were never written, and with
            // slot reuse on they were arena buffers the step's other rows had written into, so row
            // 0's token depended on the DRAFT. Four mocks, four different tokens at the same
            // position. ws-p4's glue closed that; when a pass is still partial this prints the
            // reason instead of failing on arithmetic over arena garbage.
            if partial_pass.is_empty() {
                match first_div {
                    None if n > 0 => println!("          {:>16}  {n} ids MATCH the SPEC=0 run", ""),
                    // WHERE the runs part company is the whole question, and the answer is a
                    // different sentence inside the first step than deep in the run.
                    //
                    // Inside it, nothing has accumulated: the two runs fed the same token to the
                    // same state, so a different answer is a STATE law — which is exactly the
                    // shape of ws-s1's defect (token 1, step 0, layer 0) and stays a failure.
                    //
                    // Deep in the run it is the last bits. A token at position `P` is computed at
                    // ROW `t` of a step whose `seqused_k` is `P - t + width`, and the decode
                    // attention's split-KV partition is `tps = ceil(ceil(ctx / TILE) / splits)` —
                    // so once `ctx` passes `splits * TILE` the same keys are grouped differently
                    // for the two runs and the online-softmax combine folds them in a different
                    // order. At 32 splits and TILE 16 that is 512 keys, and `oracle 4/4` on a
                    // 234-token prompt parts company at token 287 = position 521. The arms that
                    // assert exactness rather than identity are the acceptance dial above (which
                    // now measures the steps the script still describes), the width arm's
                    // two-bucket probe, and `saber verify-step`, which compares the state and the
                    // logits row bit-for-bit at every acceptance.
                    Some(at) if at <= (k as usize + 1) => {
                        fail(&format!(
                            "{name}: the verified run diverges from the SPEC=0 run at token {at} \
                             ({:?} vs {:?}), INSIDE the first verify step — nothing has \
                             accumulated there, so this is a state law and not the last bits",
                            ids.get(at),
                            solo[0].get(at)
                        ));
                        failures += 1;
                    }
                    Some(at) => println!(
                        "          {:>16}  {at} ids MATCH the SPEC=0 run and token {at} differs \
                         ({:?} vs {:?}) — past `splits * TILE` keys the two runs' attention groups \
                         the same keys differently (see the comment); the dial above measures the \
                         {at} tokens the script still describes",
                        "",
                        ids.get(at),
                        solo[0].get(at)
                    ),
                    None => {}
                }
            }
            if runs[0].0 == runs[1].0 {
                println!(
                    "          {:>16}  {} ids REPRODUCED run to run",
                    "",
                    runs[0].0.len()
                );
            } else {
                let at = runs[0]
                    .0
                    .iter()
                    .zip(&runs[1].0)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                fail(&format!(
                    "{name}: the same drafter gave different ids on two runs, first at token {at} \\
                     ({:?} vs {:?}) — a verify step must be deterministic whatever it accepts",
                    runs[0].0.get(at),
                    runs[1].0.get(at)
                ));
                failures += 1;
            }
        }
        if !partial_pass.is_empty() {
            println!(
            "          ids across drafters are NOT compared: the pass is PARTIAL ({partial_pass}), so \\
             the unwritten attention output is a shared arena buffer and row 0's token depends on \\
             what rows 1..{k} of the same step wrote into it. The identity is gated on the host \\
             (`cargo test -p saber-engine`) and by ws-h18's state-side chain gate."
            );
        }
    }
    // ---- the width controller's move: 5 -> 1 -> 5 on one sequence ------------------------------
    //
    // The ws-l6 controller narrows a sequence's verify width when its acceptance EMA falls and widens
    // it again when it recovers, BETWEEN steps and without moving its state. That is only sound if
    // the GDN conv cache's layout is the same in every bucket — it is `[slot][qkv_n][state_len]`, and
    // until 2026-09-20 `state_len` came from the bucket's own `q`, so a width-1 step read the
    // dimension at stride 3 out of a buffer a width-5 step had laid out at stride 7. The dims
    // collided, every dispatch still issued, and the tokens were quietly wrong from there.
    //
    // The layout is now the GRAPH's (`saber_model::graph::VERIFY_WIDTH`), identical in every bucket.
    // This is the gate for that: the same prompt, stepped 5 / 1 / 5 / 1 …, must commit what the same
    // prompt commits at a single width. It is the only arm that can catch a stride that disagrees
    // between two buckets, because every other arm stays in one.
    //
    // CMP27G: the CHAIN's b1 widths only. Under the tree default the engine also plans its own
    // `b1t10` form (`(1, 10)` in `decode_shapes`), which no scheduler picks (P27-15f skips it from
    // the buckets) and no chain verifies: taking it as the "wide" width asked a FixedDrafter for a
    // 9-token chain, which the tree form's chain plan refuses (`seq700: ... node 7 at depth 8`).
    let widths = chain_b1_widths(&forward.facts().decode_shapes, forward.facts().max_width);
    if widths.len() > 1 {
        let narrow = widths[0];
        let wide = *widths.last().expect("non-empty");
        let steps = args.batch_gen.max(8);
        // The reference: the same prompt at the WIDE width throughout, with the same drafter.
        let reference = {
            let mut d = FixedDrafter::new(FIXED_DRAFT_TOKEN, wide - 1);
            let mut one = vec![admit_and_prefill(
                &mut forward,
                SeqId(700),
                &prompts[0],
                &blocks_for(0, per_seq),
                prefill_chunk,
            )?];
            decode_together(&mut forward, &mut one, 1, wide, &mut d, steps)?;
            let ids = one.remove(0).ids;
            forward.release(SeqId(700));
            ids
        };
        // The switcher: the same prompt and the same drafter, alternating width every step.
        let mut d = FixedDrafter::new(FIXED_DRAFT_TOKEN, wide - 1);
        let mut one = vec![admit_and_prefill(
            &mut forward,
            SeqId(701),
            &prompts[0],
            &blocks_for(0, per_seq),
            prefill_chunk,
        )?];
        let mut switched = 0usize;
        for i in 0..steps {
            let w = if i % 2 == 0 { wide } else { narrow };
            decode_together(&mut forward, &mut one, 1, w, &mut d, 1)?;
            switched += usize::from(w != wide);
        }
        let ids = one.remove(0).ids;
        forward.release(SeqId(701));

        // ---- THE SAME STEP IN TWO BUCKETS, LAYER BY LAYER ---------------------------------------
        //
        // The arm above compares two RUNS, so it can only say which TOKEN moved — several steps and
        // sixty-four layers after whatever moved it, and only once a difference has grown into a
        // different argmax. This compares one STEP: two sequences, the same prompt, the same
        // history (the `fixed` drafter accepts nothing, so both commit exactly one token a step)
        // and the same state law (the slot-shift arm says the slot itself does not matter). The
        // only thing that differs is the BUCKET the step ran in.
        //
        // What it is for is the difference between two sentences a token comparison cannot tell
        // apart: **a state law broke** — the embedding or a GDN layer, which is where a stride, a
        // snapshot index or a conv window shows up, and ws-l17's `cache_idx` binding did — and
        // **two attention kernels round differently**, which is what `b1s1` and `b1s5` have by
        // construction: the ONLY op whose kernel differs between the two buckets is
        // `layer.<n>.attn.decode` (`saber_attn_decode_h256_gqa6_fp8kv_nw1` against `nw2`, and a
        // combine over 24 jobs against 120). Every GEMM, every producer and every GDN op is the
        // same instantiation at the same per-row arithmetic.
        // Probed, so TP=1 only — at TP > 1 a dump point is one rank's half of a reduced stream and
        // `device.rs` refuses the probe by name, as it does for the slot-shift, twin and census
        // arms. Said out loud rather than skipped quietly, and the id arm below then keeps the
        // verdict it had before there was a probe to attribute with.
        let probes_available = forward.facts().tp == 1;
        let probe_steps = steps.min(24);
        let mut bucket_div: Option<(usize, i32, String)> = None;
        println!();
        if probes_available {
            let mut wide_one = vec![admit_and_prefill(
                &mut forward,
                SeqId(710),
                &prompts[0],
                &blocks_for(0, per_seq),
                prefill_chunk,
            )?];
            let mut narrow_one = vec![admit_and_prefill(
                &mut forward,
                SeqId(711),
                &prompts[0],
                &blocks_for(1, per_seq),
                prefill_chunk,
            )?];
            let mut dw = FixedDrafter::new(FIXED_DRAFT_TOKEN, wide - 1);
            let mut dn = FixedDrafter::new(FIXED_DRAFT_TOKEN, narrow - 1);
            let mut pw: Vec<Vec<RowProbe>> = Vec::with_capacity(probe_steps);
            let mut pn: Vec<Vec<RowProbe>> = Vec::with_capacity(probe_steps);
            for _ in 0..probe_steps {
                pw.extend(decode_probed(
                    &mut forward,
                    &mut wide_one,
                    1,
                    wide,
                    &mut dw,
                    1,
                )?);
                pn.extend(decode_probed(
                    &mut forward,
                    &mut narrow_one,
                    1,
                    narrow,
                    &mut dn,
                    1,
                )?);
            }
            bucket_div = first_probe_divergence(&pw, 0, &pn, 0);
            let v = PrefixVerdict::of(&wide_one[0].ids, &narrow_one[0].ids);
            let same_ids = v.prefix_agrees();
            // THE TWO BUCKETS' OWN IDS, as digests and as a first divergence. Printed because a
            // bisect across two BOOTS (a width-`narrow` build against a width-`wide` one) can only
            // be done by comparing these numbers with the other boot's — and the within-run arm is
            // the half that holds the GDN conv layout, the captured shape set and the derived KV
            // pool FIXED, so the two halves together separate the bucket from the build (ws-s5).
            println!(
                "width     b1s{wide} ids fnv1a {:#018x} ({} token(s)), b1s{narrow} ids fnv1a \
                 {:#018x} ({} token(s)); first {:?}",
                digest(&wide_one[0].ids),
                wide_one[0].ids.len(),
                digest(&narrow_one[0].ids),
                narrow_one[0].ids.len(),
                &narrow_one[0].ids[..narrow_one[0].ids.len().min(12)]
            );
            if let Some(at) = v.first_diff {
                println!(
                    "width     the two buckets' ids part company at token {at}: b1s{wide} {:?} vs \
                     b1s{narrow} {:?}",
                    wide_one[0].ids.get(at),
                    narrow_one[0].ids.get(at)
                );
            }
            forward.release(SeqId(710));
            forward.release(SeqId(711));
            println!(
                "width     row 0 of a b1s{wide} step against row 0 of a b1s{narrow} step, same \
                 prompt, {probe_steps} step(s), eager on both sides: first dump point that differs \
                 is {}; ids {}",
                probe_where(bucket_div.as_ref()),
                if same_ids { "identical" } else { "DIFFER" }
            );
        } else {
            println!(
                "width     the two-bucket probe is skipped at TP={}: a dump point is one rank's \
                 half of a reduced stream (gated at TP=1, like slot-shift, twin and the census)",
                forward.facts().tp
            );
        }

        let n = ids.len().min(reference.len());
        if n > 0 && ids[..n] == reference[..n] {
            println!(
                "width     a sequence stepped {wide}/{narrow} alternately ({switched} narrow steps \
                 of {steps}) commits the same {n} ids as one stepped {wide} throughout — the conv \
                 cache's layout is the same in both buckets"
            );
        } else {
            let at = ids
                .iter()
                .zip(&reference)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            // WHICH SENTENCE THIS IS, from the probe above rather than from the token. A dump
            // point at an ATTENTION layer's output is the two kernels' last bits and the ids only
            // move once that crosses a near-tie; anything earlier — the embedding, a GDN layer — is
            // a state law, and that is a failure whatever the token says.
            // `(l + 1) % interval == 0` is the model's own layer_kind, and the interval comes
            // from the engine rather than from this file.
            let interval = forward.facts().attention_interval.max(1);
            let attn_only = matches!(&bucket_div, Some((_, layer, _))
                if *layer >= 0 && (*layer as u32 + 1) % interval == 0);
            if attn_only {
                println!(
                    "width     switching between {wide} and {narrow} changed the ids at token {at} \
                     ({:?} vs {:?}) — and the first dump point the two buckets differ at is {}, an \
                     ATTENTION layer, which is the one op whose KERNEL differs between them. Every \
                     GDN layer and the embedding agree, so this is two attention kernels' last \
                     bits crossing a near-tie and not a state law: `saber verify-step` compares \
                     the same two buckets' state and logits row bit-for-bit at every acceptance",
                    ids.get(at),
                    reference.get(at),
                    probe_where(bucket_div.as_ref())
                );
            } else if probes_available {
                fail(&format!(
                    "width: switching between {wide} and {narrow} changed the ids at token {at} \
                     ({:?} vs {:?}), and the two buckets' first divergence is {} — NOT an attention \
                     layer, so a state law is being read differently in the two buckets",
                    ids.get(at),
                    reference.get(at),
                    probe_where(bucket_div.as_ref())
                ));
                failures += 1;
            } else {
                // No probe to attribute with, so the arm keeps the verdict it had before there was
                // one. A TP > 1 divergence here is a thing to look at at TP=1, where the bisect is.
                fail(&format!(
                    "width: switching between {wide} and {narrow} changed the ids at token {at} \
                     ({:?} vs {:?}), and at TP={} there is no two-bucket probe to say which layer \
                     they parted at — run the same arm at TP=1, where it bisects",
                    ids.get(at),
                    reference.get(at),
                    forward.facts().tp
                ));
                failures += 1;
            }
        }
    }

    // ---- parking, if this run was given a staging buffer ---------------------------------------
    // Before the engine is moved into the scheduler, because this drives it directly.
    if forward.can_park() {
        // BOTH PATHS, and they are different code: the synchronous one is `Req::Park` /
        // `Req::Restore` (what a tier's explicit hook takes) and the asynchronous one is
        // `ParkBegin` / `ParkPoll` / `RestoreBegin` / `RestorePoll` — the SERVE PLANE's path,
        // which ws-s5 wired and nobody had run (ws-t2). At TP=2 each of them now also drives the
        // other rank's half over the `engine.park` frames, so this arm is the gate for that too.
        let law = args.engine.config().park_image_law;
        let mut park_failures = 0usize;
        // THE STATE-SLOT LAW'S OWN NUMBER, read before the arm and again after it (ws-k15): a
        // restore hands the engine a `committed` out of the image and the next step's
        // `committed_before` comes from the caller, so the park arm is where the two can part.
        let slots_before = forward.slot_counts();
        for async_path in [false, true] {
            park_failures += park_round_trip(
                &mut forward,
                &prompts[0],
                per_seq,
                prefill_chunk,
                width,
                engine_drafts,
                &solo[0],
                &solo_trace,
                async_path,
                law,
                args.engine.park_probe,
                1,
                &fail,
            );
        }
        // THE SAME SESSION PARKED TWICE (ws-k17), on the serve plane's own asynchronous path:
        // `ParkPolicy::max_parks_per_session` no longer caps a session at one eviction, so a
        // session restored into other pages and slots has to survive being parked AGAIN out of
        // them -- and still commit exactly the unparked run's ids and proposals.
        park_failures += park_round_trip(
            &mut forward,
            &prompts[0],
            per_seq,
            prefill_chunk,
            width,
            engine_drafts,
            &solo[0],
            &solo_trace,
            true,
            law,
            args.engine.park_probe,
            2,
            &fail,
        );
        // AN IMAGE ABORTED WHILE PARKED (ws-k18): the follower's half must go back with it. Its own
        // tally rather than `park_failures`, because the base-only control is about the image law
        // and this arm is not.
        if law != ParkImageLaw::BaseOnly {
            failures += park_discard_arm(
                &mut forward,
                &prompts[0],
                per_seq,
                prefill_chunk,
                width,
                engine_drafts,
                &fail,
            );
        }
        // THE CONTROL'S VERDICT IS INVERTED, and it is the arm's own (`CLAUDE.md` rule 8): with
        // `--park-image-base` the image carries the run's BASE for every state, which is right
        // only for a session that lands back on its own slots — and this arm restores into
        // different ones on purpose. A control that passed would mean the arm cannot see the law.
        if law == ParkImageLaw::BaseOnly {
            if park_failures == 0 {
                fail(
                    "park: the NEGATIVE CONTROL (`--park-image-base`, the pre-ws-k14 image) \
                     PASSED. Either the restored session landed back on its own slots, or the \
                     step before the park committed one token and the base IS the live snapshot \
                     — the park line says which. This arm cannot see the image law.",
                );
                failures += 1;
            } else {
                println!(
                    "OK        the negative control failed as it must: {park_failures} park \
                     arm failure(s) under the pre-ws-k14 base-only image"
                );
            }
        } else {
            failures += park_failures;
        }
        let slots_after = forward.slot_counts();
        let moved = slots_after.committed_mismatched - slots_before.committed_mismatched;
        println!("park      slot law: {}", slots_after.line());
        if moved == 0 {
            println!(
                "OK        every step of the park arm continued from the snapshot this engine's \
                 own last step wrote: the image's `committed` and the caller's \
                 `committed_before` agree across a restore"
            );
        } else {
            fail(&format!(
                "{moved} step(s) of the park arm ran with a `committed_before` that is not what \
                 this engine's last step committed: a restored session is continuing from another \
                 of its recurrent snapshots"
            ));
            failures += 1;
        }
    } else if args.engine.park_staging_mib > 0 {
        // Asked for and refused, by name — which now means one thing only: this engine has no
        // staging buffer. Parking used to be TP = 1 only (half a session is not a session); at
        // TP > 1 the follower images its own half on the leader's command (ws-t2).
        println!(
            "park      REFUSED at TP={}: this engine has no park staging buffer",
            forward.facts().tp
        );
    }

    // ---- `clear_pages`, the arm S4 committed it without ------------------------------------------
    failures += draft_page_clear_arm(&mut forward, &prompts, per_seq, prefill_chunk, width, &fail);

    // ---- C9: the fused mixed step's per-row class gate, when the run serves it ------------------
    if args.engine.mixed_prefill.fused {
        failures += mixed_fused_arm(
            &mut forward,
            &prompts,
            per_seq,
            prefill_chunk,
            width,
            args.engine.mixed_class_proxy,
        )?;
        // C10: the same gate for a PACKED step of two and of three chunks.
        if args.engine.mixed_prefill.max_chunks > 1 {
            for chunks in [2usize, 3] {
                failures += mixed_packed_arm(
                    &mut forward,
                    &prompts,
                    per_seq,
                    prefill_chunk,
                    width,
                    args.engine.mixed_class_proxy,
                    chunks,
                    &batches,
                )?;
            }
        }
    }

    // ---- the serve path, over the same engine ---------------------------------------------------
    let config = scheduler_config(
        &forward,
        ctx_band_for(forward.facts().max_ctx),
        &args.engine.width_rule,
        args.engine.ragged_ceiling,
        args.engine.mixed_prefill,
        args.engine.prefill_fill_above,
    )?;
    let model = ModelInfo {
        served_model_name: "Qwen3.8".to_owned(),
        aliases: Vec::new(),
        max_model_len: config.max_context_tokens,
        root: args.engine.pack.display().to_string(),
        // No EOS: the gate wants the same number of steps on both paths, and a partial pass's ids
        // are arbitrary — one of them landing on `<|im_end|>` would end the request early.
        eos_token_ids: Vec::new(),
    };
    let want = solo[0].len();
    // Taken before the engine moves into the executor: the tier is a fact of the run and the
    // concurrency comparison names it.
    let forward_tp = forward.facts().tp;
    let forward_max_width = forward.facts().max_width;
    // ws-f38: the fork arm's PARK leg needs a tier to park a group into. Attached only when that
    // arm runs on an engine with a staging buffer, sized from the engine's own slot as the serve
    // plane sizes it, and with the default policy (nothing parks unless the arm asks).
    let fork_park_image = {
        let f = forward.facts();
        (f.park_staging_bytes / u64::from(f.park_staging_slots.max(1))).max(1)
    };
    let fork_parks = !args.fork_n.is_empty() && forward.facts().park_staging_bytes > 0;
    let fused_mixed = config.mixed.fused;
    let mut executor = RealExecutor::new(forward);
    // C9: the fused mixed forward, when the mixed step is on in that form.
    executor.set_fused_mixed(fused_mixed);
    let mut built = Scheduler::new(config, executor, model, ledger)
        .context("building the scheduler over the real engine")?;
    if fork_parks {
        // The pinned bytes the ENGINE declared for a scheduler tier (`park_ram_gib`, which
        // `saber engine` sets for the fork arm): a tier sized from anything else is billed
        // against a pool that is not there, and the first run of this leg found every group's
        // finished park refused as "host.pinned is full: 0 free".
        let ram = (args.engine.park_ram_gib.max(0.0) * (1u64 << 30) as f64) as u64;
        let store = saber_sched::TieredParkStore::host(&saber_sched::TieredParkConfig::ram(
            fork_park_image.min(ram.max(1)),
            ram,
        ))
        .map_err(|e| anyhow::anyhow!("the fork arm's park tier: {e:?}"))?;
        built = built.with_park_tier(Box::new(store));
    }
    let sched = Arc::new(built);
    // A drafter on the SCHEDULER, when a verify width was planned. Nothing else changes: the width
    // controller starts every cold sequence at the widest captured width (`ceiling_width`), so the
    // serve-identity and concurrency sections below run the b*s5 shape without being told to — and
    // the ids they assert are still the SPEC=0 run's, which is the whole claim. `fixed` is used
    // rather than the oracle because the scheduler assigns the sequence ids, so there is no
    // recording to key a script on: acceptance 0 through the serve plane is the honest arm, and the
    // dialled ones are measured by the direct-drive sweep above.
    let spec_through_serve = forward_max_width > 1;
    if spec_through_serve {
        let old = sched.with_executor(|e| {
            // With a RESIDENT drafter the serve path takes the PASS's own proposals, which is what
            // makes the serve-identity and concurrency arms a claim about the real drafter rather
            // than about a mock that never accepts. `fixed` stays the NEGATIVE CONTROL: acceptance
            // 0, and the same ids, through the same machinery.
            e.set_drafter(if engine_drafts {
                Box::new(EngineDrafter::new(forward_max_width - 1)) as Box<dyn Drafter>
            } else {
                Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, forward_max_width - 1))
            })
        });
        println!(
            "spec      the serve path and the concurrency sweep run with the `{}` drafter at width \
             up to {} (was `{}`)",
            sched.with_executor(|e| e.drafter_name()),
            forward_max_width,
            old.name()
        );
    }
    let sub = sched
        .submit(EngineRequest {
            prompt_token_ids: prompts[0].clone(),
            params: SamplingParams {
                max_tokens: Some(want),
                // TEMPERATURE 0, deliberately: this is an ID gate. `SamplingParams::default()` is
                // The reference server's own 1.0, and since the sampler was wired into the step loop that default
                // draws — two runs of it would be two draws rather than two answers.
                temperature: 0.0,
                ..SamplingParams::default()
            },
        })
        .map_err(|e| anyhow::anyhow!("submitting through the serve path: {e}"))?;
    let t0 = Instant::now();
    let steps = sched.run_steps(want * 4 + 16);
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("a runtime to drain the stream")?;
    let events = runtime.block_on(async move {
        use futures_util::StreamExt;
        sub.events.collect::<Vec<_>>().await
    });
    let served: Vec<TokenId> = events.iter().flat_map(|e| e.new_tokens.clone()).collect();
    let counters = sched.with_executor(|e| e.counters());
    println!();
    println!(
        "serve     {steps} engine steps, {} events, {} tokens, {:.2} ms/step through the scheduler",
        events.len(),
        served.len(),
        wall / steps.max(1) as f64
    );
    println!("counters  {}", counters.line());
    if served == solo[0] {
        println!(
            "OK        {} ids identical through the pass and the serve path",
            served.len()
        );
    } else {
        fail("the pass and the serve path disagree");
        failures += 1;
    }
    if events.len() == steps {
        println!("OK        one event per engine step");
    } else {
        fail(&format!("{} events for {steps} engine steps", events.len()));
        failures += 1;
    }
    if counters.clean() {
        println!("OK        every step ran a declared shape at its declared dispatch count");
    } else {
        fail(&format!(
            "the engine counters are not clean: {}",
            counters.line()
        ));
        failures += 1;
    }
    if counters.graph_steps == 0 && args.engine.capture {
        fail("capture was asked for and no step replayed a graph");
        failures += 1;
    }
    // ---- CONTINUITY: a request is a function of itself, whatever the engine served before it ---
    //
    // Every other identity arm here compares two runs that see the same history. None of them can
    // see a fresh request whose answer depends on what the engine did BEFORE it — a state slot
    // handed back dirty, a KV page reused before it is overwritten, an arena edge read before it is
    // written, an index blob still describing the last step's rows. That class took multi-turn chat
    // on the prod stack on 2026-09-17 and it is invisible to every gate that asks its question once.
    //
    // TWO THINGS ABOUT THE BOUND. This arm asked for `conc_tokens.clamp(4, 16)` tokens until
    // ws-l16, and the four turns `multiturn_continuity.py` fails on first differ at tokens 2, 5,
    // 11 and **29** — so the arm could not have seen three of the four however often it ran.
    // `--continuity-tokens` raises it and the default is unchanged, which is why the gate's own
    // number did not move when the flag landed.
    //
    // And the SHAPE is the gate's: pass A over every subject in order, pass B over them in
    // REVERSE. Re-issuing a request straight after itself is the easiest case for any engine —
    // same pages, same slot, still warm; running it last, after every other subject has been
    // through, is the case that finds a page handed on dirty. The first sampled token is still the
    // sharpest single assertion (it comes out of prefill, upstream of every decode step), but it
    // is no longer the only one: a divergence at token 29 is a decode fact and needs the tokens.
    let cont_want = if args.continuity_tokens > 0 {
        args.continuity_tokens
    } else {
        args.conc_tokens.clamp(4, 16)
    };
    // The transcript's whole list when there is one, and the gate's subjects otherwise.
    let cont: Vec<Vec<TokenId>> = if continuity_prompts.is_empty() {
        prompts.clone()
    } else {
        continuity_prompts.clone()
    };
    let mut cont_failed: Option<usize> = None;
    {
        println!();
        println!(
            "continuity {} request(s) x up to {cont_want} token(s) through the scheduler: pass A \
             in order, pass B in REVERSE",
            cont.len()
        );
        let mut a: Vec<Vec<TokenId>> = Vec::with_capacity(cont.len());
        for p in &cont {
            a.push(run_once(&sched, p, cont_want)?);
        }
        let mut b: Vec<Vec<TokenId>> = vec![Vec::new(); cont.len()];
        for i in (0..cont.len()).rev() {
            b[i] = run_once(&sched, &cont[i], cont_want)?;
        }
        let mut bad = 0usize;
        for i in 0..cont.len() {
            match first_id_difference(&a[i], &b[i]) {
                None => println!(
                    "          request {i}: {} prompt token(s), {} ids identical",
                    cont[i].len(),
                    a[i].len()
                ),
                Some(at) => {
                    bad += 1;
                    if cont_failed.is_none() {
                        cont_failed = Some(i);
                    }
                    println!(
                        "          request {i}: {} prompt token(s), DIFFERS at token {at} ({:?} \
                         vs {:?}), {} ids against {}",
                        cont[i].len(),
                        a[i].get(at),
                        b[i].get(at),
                        a[i].len(),
                        b[i].len()
                    );
                }
            }
        }
        if bad == 0 {
            println!(
                "OK        {} re-issued request(s) of up to {cont_want} token(s) are each exactly \
                 their own single-shot",
                cont.len()
            );
        } else {
            fail(&format!(
                "{bad} of {} re-issued requests differ from their own first pass — a fresh \
                 request's ids are a function of the request alone, so the engine kept something \
                 across a release",
                cont.len()
            ));
            failures += 1;
        }
    }

    // ---- the continuity BISECT: the failing request, probed, inside the failing REPLAY ---------
    //
    // The arm above names a TOKEN. This one names the first STEP and the first LAYER, which is
    // what ws-l15's census did for the batched divergence.
    //
    // It is an EXACT REPLAY of the two passes, with one request swapped for a probed one: every
    // other request goes through the scheduler exactly as it just did, and the subject is driven
    // through the pass with every row of the residual stream digested at every layer. Anything
    // less is a different experiment — the property under test is about what ran BEFORE the
    // request, so the traffic has to be the traffic, in its order, generating what it generated.
    //
    // The control runs first and has nothing between its two runs: a bisect that disagrees with
    // itself back to back is measuring the probe, not the engine.
    if args.continuity_tokens > 0 {
        let subject_i = cont_failed.unwrap_or(0);
        let subject = cont[subject_i].clone();
        let steps = args.continuity_tokens.min(48);
        let pages = blocks_for(0, per_seq);
        println!();
        println!(
            "bisect    request {subject_i} of {} ({} prompt token(s)), {steps} probed decode \
             step(s), chunked at {prefill_chunk} rows on pages {}..{}",
            cont.len(),
            subject.len(),
            pages.first().copied().unwrap_or(0),
            pages.last().copied().unwrap_or(0)
        );
        let probe_once = |sched: &Arc<Scheduler<RealExecutor<HipForward>>>,
                          seq: u64,
                          drafter: &mut dyn Drafter|
         -> anyhow::Result<(Vec<TokenId>, Vec<Vec<RowProbe>>)> {
            sched.with_executor(|e| {
                request_through_pass(
                    e.forward_mut(),
                    SeqId(seq),
                    &subject,
                    &pages,
                    prefill_chunk,
                    steps,
                    width,
                    drafter,
                    true,
                )
            })
        };
        let (ids_x, probe_x) = probe_once(&sched, 960, probe_drafter.as_mut())?;
        let (ids_y, probe_y) = probe_once(&sched, 961, probe_drafter.as_mut())?;
        match (
            first_id_difference(&ids_x, &ids_y),
            first_probe_divergence(&probe_x, 0, &probe_y, 0),
        ) {
            (None, None) => println!(
                "OK        control: the request twice, back to back, on the same pages — {} ids \
                 and every layer identical",
                ids_x.len()
            ),
            (at, first) => {
                fail(&format!(
                    "control: the same request twice with NOTHING between them differs at token \
                     {at:?}, first {}",
                    probe_where(first.as_ref())
                ));
                failures += 1;
            }
        }
        // PASS A, in order, with the subject probed.
        let mut probe_a = Vec::new();
        let mut ids_a = Vec::new();
        for (i, p) in cont.iter().enumerate() {
            if i == subject_i {
                let (ids, probe) = probe_once(&sched, 962, probe_drafter.as_mut())?;
                ids_a = ids;
                probe_a = probe;
            } else {
                let _ = run_once(&sched, p, cont_want)?;
            }
        }
        // PASS B, in reverse, with the subject probed again.
        let mut probe_b = Vec::new();
        let mut ids_b = Vec::new();
        for i in (0..cont.len()).rev() {
            if i == subject_i {
                let (ids, probe) = probe_once(&sched, 963, probe_drafter.as_mut())?;
                ids_b = ids;
                probe_b = probe;
            } else {
                let _ = run_once(&sched, &cont[i], cont_want)?;
            }
        }
        let chunks = probe_a.len().saturating_sub(steps);
        let at = first_id_difference(&ids_a, &ids_b);
        let first = first_probe_divergence(&probe_a, 0, &probe_b, 0);
        println!(
            "bisect    {chunks} prefill chunk(s) then {steps} decode step(s) probed per pass, {} \
             dump point(s) per step",
            probe_a.last().map_or(0, Vec::len)
        );
        println!(
            "          the same request in pass A and pass B: ids differ at token {at:?}, \
             residual {}",
            probe_where(first.as_ref())
        );
        if at.is_none() && first.is_none() {
            println!(
                "OK        the probed replay reproduces nothing: the request is its own \
                 single-shot at every layer of every probed step"
            );
        } else if let Some((step, layer, edge)) = &first {
            // The dump points are the residual stream at each layer's INPUT, in plan order, so the
            // index of the first differing one is the first layer whose input moved — and the step
            // index says whether it was the PREFILL (below `chunks`) or a decode step.
            let where_ = if *step < chunks {
                format!("prefill chunk {step}")
            } else {
                format!("decode step {}", step - chunks)
            };
            println!(
                "          FIRST DIVERGENCE: {where_}, layer {layer} (`{edge}`) — {} probed \
                 step(s) before it agreed at every layer",
                step
            );
        }
    }

    /// The all-reduce family two decode shapes resolved to, when they are not the same one.
    ///
    /// Printed beside a divergence, and it is the difference between a gate that reports a defect and
    /// one that reports an experiment. A decode step's message is `rows x hidden x 2` and the rules'
    /// `quant_min_bytes` is 131,072, so at TP=2 and hidden 5120 a verify width above 2 puts a batch-4
    /// step (204,800 B) onto the LOSSY `allreduce_wht` family while the same sequence stepped alone
    /// (51,200 B) keeps the exact one -- and an arm that compares the two is comparing two different
    /// arithmetics. `run_engine.sh --tp 2 --widths 5`'s concurrency 4, concurrency 8 and mixed-arrival
    /// arms are exactly that (ws-l17), and `--ar-exact` is the answer: it raises the band above every
    /// message the run can send, so both sides are exact and the arm asks its own question again.
    ///
    /// `None` at TP=1, where a step has no collective, and `None` when the families agree — in which
    /// case the divergence is something else and saying nothing is the honest output.
    fn ar_band_note(
        facts: &saber_engine::ForwardFacts,
        width: u32,
        batches: &[u32],
    ) -> Option<String> {
        if facts.tp < 2 {
            return None;
        }
        let named: Vec<(u32, String)> = batches
            .iter()
            .copied()
            .filter_map(|b| {
                let rows = StepRows {
                    phase: Phase::Decode,
                    rows: b * width,
                    width,
                    samples: true,
                    tree: false,
                    ragged: false,
                };
                facts.ar_family(rows).map(|f| (b, f))
            })
            .collect();
        let distinct: std::collections::BTreeSet<&String> = named.iter().map(|(_, f)| f).collect();
        if distinct.len() < 2 {
            return None;
        }
        let cells: Vec<String> = named
            .iter()
            .map(|(b, f)| format!("b{b}s{width} {f}"))
            .collect();
        Some(format!(
        "AR BAND: {} \u{2014} the two arms resolved DIFFERENT all-reduce families, so this is a \
         comparison of two arithmetics and not a defect. Re-run with --ar-exact{}",
        cells.join(" | "),
        if facts.ar_exact {
            " (which is already on: this is a real divergence)"
        } else {
            ""
        }
    ))
    }

    // ---- several requests in flight, through the scheduler -----------------------------------------
    //
    // This is the serving shape the engine exists for: the scheduler admits, batches, picks the
    // bucket and replays it, and the streams are the ones a client sees. The claim is that NONE of
    // that moves a sequence's ids — the same prompt that ran alone above comes back token for token
    // whether it shares its steps with none, three or seven others, and whether the others arrived
    // before it or in the middle of its response.
    // ---- AND THE BATCHED PROPOSAL'S PADDED CASE (ws-s10) ----------------------------------
    //
    // The equivalence arm above ran at `live == reqs`, which is the batch with NO padding in it.
    // These arms are where the padding runs: a concurrency below the planned batch, and a mixed
    // arrival where the live count moves between steps. So the arm is turned on again here, over
    // the SAME switch and the same comparison, and what it adds is the rows a shape has and the
    // step does not. It doubles these arms' proposals, so their tok/s is not a serving figure in
    // a run that has it on — which the line below says rather than leaving the reader to infer.
    let dc_conc = draft_counters.snapshot();
    if batch_equiv {
        draft_counters
            .batch_check
            .store(true, std::sync::atomic::Ordering::Relaxed);
        println!(
            "          (the batched-proposal equivalence arm is ON for the concurrency arms: \
             their tok/s is not a serving figure in this run)"
        );
    }
    let mut rows: Vec<ConcRow> = Vec::new();
    // Divergences these arms found, as `(concurrency, prompt, token)`. They are attributed after
    // the scheduler has given the engine back, because the attribution drives it directly (ws-p9).
    let mut conc_bad: Vec<(usize, usize, usize, ArmRate)> = Vec::new();
    for c in args.conc.iter().copied().filter(|c| *c > 0) {
        if c > args.engine.max_seqs as usize {
            continue;
        }
        let row = run_concurrent(&sched, &conc_prompts, c, args.conc_tokens, None)?;
        // ONE STREAM PER DISTINCT PROMPT is what the hazard counts (ws-p13): a prompt the arm
        // had to run twice is one sample, not two.
        let rate = ArmRate::of_streams(row.ids.iter().enumerate().map(|(i, ids)| {
            (
                i % conc_prompts.len(),
                &ids[..],
                &conc_solo[i % conc_solo.len()][..],
            )
        }))
        .in_flight(c);
        println!(
            "conc      concurrency {c}: the arm's hazard: {}",
            rate.line()
        );
        for (i, ids) in row.ids.iter().enumerate() {
            let want = &conc_solo[i % conc_solo.len()];
            let n = ids.len().min(want.len());
            if ids[..n] != want[..n] {
                let at = ids[..n]
                    .iter()
                    .zip(&want[..n])
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                println!(
                    "conc      concurrency {c}: stream {i} differs from its solo ids at token \
                     {at} — attributing it"
                );
                if let Some(note) = ar_band_note(&facts, width, &[1, c as u32]) {
                    println!("          {note}");
                }
                // THE SAME QUESTION THE BATCHED ARM ASKS, and the same law (ws-p9). A stream in a
                // concurrency-`c` run is one sequence in a step of up to `c`, so the attribution
                // is the direct-drive form of exactly this comparison: the same prompt alone
                // against the same prompt in a batch-`c` step, probed on both sides.
                conc_bad.push((c, i % conc_prompts.len(), at, rate));
                break;
            }
        }
        rows.push(row);
    }
    // The mixed arrival: one request joins after the others are already decoding, so its prefill
    // runs BETWEEN their decode steps and its admission changes their slot set mid-flight.
    let mixed = run_concurrent(&sched, &conc_prompts, 4, args.conc_tokens, Some(10))?;
    if batch_equiv {
        draft_counters
            .batch_check
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let d = counts_delta(dc_conc, draft_counters.snapshot());
        if d.batch_checked == 0 {
            println!(
                "          batched proposal, PADDED: no comparison ran — these arms made no \
                 batched proposal"
            );
        } else if d.batch_mismatch > 0 {
            fail(&format!(
                "batched proposal, PADDED: {} of {} request(s) got a DIFFERENT block in a batch \
                 whose shape is wider than the step than they got proposed one at a time — the \
                 padding rows are not inert",
                d.batch_mismatch, d.batch_checked
            ));
            failures += 1;
        } else {
            println!(
                "OK        batched proposal, PADDED: {} request-proposal(s) identical row for row \
                 through the concurrency and mixed-arrival arms, where the step is NARROWER than \
                 the shape it is padded up to",
                d.batch_checked
            );
        }
    }
    let mut mixed_ok = true;
    let mixed_rate = ArmRate::of_streams(mixed.ids.iter().enumerate().map(|(i, ids)| {
        (
            i % conc_prompts.len(),
            &ids[..],
            &conc_solo[i % conc_solo.len()][..],
        )
    }))
    .in_flight(mixed.ids.len());
    println!("mixed     the arm's hazard: {}", mixed_rate.line());
    for (i, ids) in mixed.ids.iter().enumerate() {
        let want = &conc_solo[i % conc_solo.len()];
        let n = ids.len().min(want.len());
        if ids[..n] != want[..n] {
            let at = ids[..n]
                .iter()
                .zip(&want[..n])
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            println!(
                "mixed     stream {i} was perturbed by a request that joined mid-run, at token \
                 {at} — attributing it"
            );
            if let Some(note) = ar_band_note(&facts, width, &[1, 4]) {
                println!("          {note}");
            }
            conc_bad.push((4, i % conc_prompts.len(), at, mixed_rate));
            mixed_ok = false;
            break;
        }
    }
    println!();
    println!(
        "conc      through the scheduler, {} tokens per stream:",
        args.conc_tokens
    );
    for r in &rows {
        println!(
            "          {:>2} stream(s): {:7.1} tok/s aggregate, {:6.1} tok/s per stream, TTFT p50 \
             {:6.0} ms / max {:6.0} ms, {} steps",
            r.conc, r.aggregate_tps, r.per_stream_tps, r.ttft_p50_ms, r.ttft_max_ms, r.steps
        );
        println!(
            "              width: {} of {} step(s) ran narrower than the ceiling, {} slot cap(s) \
             absorbed by the bucket; decode shapes so far {}",
            r.width_narrowed_steps,
            r.steps,
            r.slot_caps_absorbed,
            r.shapes
                .iter()
                .map(|(b, w)| format!("b{b}s{w}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    // What the width rule can do here AT ALL, printed beside the numbers it explains. Either
    // rule is rounded UP to a captured width, so a plan that captured only `1` and the ceiling has
    // exactly one reachable width and the rule is inert however the run goes (ws-t1, ws-t2).
    {
        let wp = sched.config().width;
        println!(
            "          width     rule {}; ladder {:?}, reachable {:?} — {}",
            wp.rule,
            wp.widths,
            wp.reachable_widths(),
            if wp.can_narrow() {
                "the rule has somewhere to go"
            } else {
                "INERT: one reachable captured width"
            }
        );
        if wp.rule == WidthRule::AcceptanceEma {
            println!(
                "                    EMA knobs alpha {} margin {} min {} min_batch {} (prod's)",
                wp.prod.alpha, wp.prod.margin, wp.prod.min_width, wp.prod.min_batch,
            );
        }
        // C2: the ragged schedule's own numbers for the arm, beside the rule.
        let n = sched.counters();
        println!(
            "          ragged    {}; steps {} ({} in a rectangle), rows per step {:.2} live / \
             {:.2} run, slots narrowed by the cap {}",
            match wp.ragged {
                Some(c) => format!("ON, ceiling {c}"),
                None => "OFF (--rectangular-widths, or no band captured)".to_owned(),
            },
            n.ragged_steps,
            n.ragged_rect_steps,
            n.ragged_live_rows as f64 / n.ragged_steps.max(1) as f64,
            n.ragged_step_rows as f64 / n.ragged_steps.max(1) as f64,
            n.ragged_slots_narrowed,
        );
    }
    println!(
        "          mixed arrival (one joins after {} steps): {:7.1} tok/s aggregate, {} streams \
         unperturbed",
        10,
        mixed.aggregate_tps,
        if mixed_ok { "all" } else { "NOT all" }
    );
    println!(
        "          prod's BetterBench conc-8 is 597 t/s WITH speculation (dflash SPEC=7, ~3 tokens \
         per step, TP=2); this engine is SPEC=0 at TP={}, so the comparison is a shape, not a race.",
        forward_tp
    );
    if let Some(widest) = rows
        .iter()
        .filter(|r| r.conc > 1)
        .map(|r| r.max_batch)
        .max()
    {
        println!("          widest decode shape the scheduler chose: batch {widest}");
    }
    // ---- ATTRIBUTING WHAT EVERY BUCKET ARM FOUND (ws-p9), AT THE END OF THE GATE (ws-t3) --------
    //
    // Every red — the batched arm's and these two — is one stream compared between a batch-1 run
    // and a batch-`c` one. The attribution is the direct-drive form of that question, and it runs
    // HERE for two reasons: with the scheduler idle and its executor's lock held nothing steps
    // underneath it, and NOTHING THIS GATE PUBLISHES IS MEASURED AFTER IT. The second reason is
    // ws-t3's: run at the batched arm, where it used to be, it put +2.7 ms/step on every timed arm
    // that followed it in the same boot (see the batched arm above for the paired numbers).
    // The drafter is a fresh one of the run's own kind: `EngineDrafter` is a mailbox and a new
    // sequence's is empty either way.
    let fresh_drafter = || -> Box<dyn Drafter> {
        if !spec_through_serve {
            Box::new(NoDrafter)
        } else if engine_drafts {
            Box::new(EngineDrafter::new(width - 1))
        } else {
            Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, width - 1))
        }
    };
    for (i, at) in batched_bad.clone() {
        let mut d = fresh_drafter();
        let out = sched.with_executor(|e| {
            attribute_buckets(
                e.forward_mut(),
                &prompts,
                i,
                use_batch,
                width,
                // ENOUGH STEPS TO REACH THE FLIP. The arm found it at token `at` with a drafter
                // that commits several tokens a step; the attribution must not stop short of it,
                // or it reports that its own two runs agreed (ws-p9).
                attribution_steps(at, args.gen.min(16).saturating_sub(1)),
                per_seq,
                prefill_chunk,
                d.as_mut(),
                None,
                // THE ARM'S OWN BATCH (ws-p11): the batched arm drives the engine directly, so
                // its configuration is reproducible exactly and the verdict is taken on it.
                Subject::ArmSlot,
            )
        });
        match out {
            Err(e) => {
                fail(&format!(
                    "batched: sequence {i} differs at token {at} and the attribution could not \
                     run: {e}"
                ));
                failures += 1;
            }
            Ok(v) => {
                for l in v.lines() {
                    println!("{l}");
                }
                match v.verdict_at(facts.tp, Some(&batched_rate)) {
                    Ok(line) => println!("OK        batched sequence {i}: {line}"),
                    Err(why) => {
                        fail(&format!("batched: sequence {i} at token {at} — {why}"));
                        failures += 1;
                    }
                }
            }
        }
    }
    // ---- THE CONTROL FOR THAT LAW (ws-p9) -------------------------------------------------------
    //
    // The law above lets an id divergence between two buckets pass when it is a near-tie crossed
    // by two arithmetics. A law that lets something pass needs a case it must NOT let pass, or it
    // is a gate that cannot fail. So: the same attribution, over a batch whose slot 0 reads the
    // WRONG recurrent snapshot — `committed_before` is `naccept`, in range and silent, which is
    // the shape ws-l17's `cache_idx` binding had — and the verdict must come back a DEFECT.
    //
    // It runs whenever the batched arm can run, red or green, because a control that only runs
    // when something is already broken is not a control either.
    if batch_control && use_batch > 1 {
        let mut d = fresh_drafter();
        let out = sched.with_executor(|e| {
            attribute_buckets(
                e.forward_mut(),
                &prompts,
                0,
                use_batch,
                width,
                args.gen.min(16).saturating_sub(1),
                per_seq,
                prefill_chunk,
                d.as_mut(),
                Some(0),
                Subject::SlotZero,
            )
        });
        match out {
            Err(e) => {
                fail(&format!("the GDN-side control could not run: {e}"));
                failures += 1;
            }
            Ok(v) => {
                for l in v.lines() {
                    println!("{l}");
                }
                // THE CONTROL IS HELD TO THE SAME RATE LAW, over its own exposure: one sequence,
                // one divergence at the position it reached (ws-p11). A law that excused a
                // decided flip at the documented rate must still refuse this one.
                let control_rate = ArmRate {
                    events: 1,
                    reached: v.at + 1,
                    in_flight: use_batch as usize,
                };
                match v.verdict_at(facts.tp, Some(&control_rate)) {
                    Err(why) => println!(
                        "OK        CONTROL: a batch whose slot 0 continues from the wrong \
                         recurrent snapshot is a DEFECT to the same law — {why}"
                    ),
                    Ok(line) => {
                        fail(&format!(
                            "the GDN-side CONTROL passed the arm's law as `{line}`: a state law \
                             read wrongly must not read as a bucket arithmetic, or the arm has no \
                             teeth"
                        ));
                        failures += 1;
                    }
                }
            }
        }
    }
    if args.no_attribute && !conc_bad.is_empty() {
        println!(
            "conc      {} red(s) NOT attributed (--no-attribute), as above",
            conc_bad.len()
        );
        conc_bad.clear();
    }
    for (c, i, at, rate) in conc_bad.clone() {
        let batch = batches
            .iter()
            .copied()
            .filter(|b| *b as usize >= c)
            .min()
            .unwrap_or(u32::try_from(c).unwrap_or(1));
        let mut d = fresh_drafter();
        // The attribution batch is the gate's own prompts with the subject put in (ws-p13): a
        // stream past `gate_seqs` is a prompt the gate's list does not hold, and the batch stays
        // the size `per_seq` was sized for. It is a neighbouring run either way (below).
        let (subject_set, subject_i) = attribution_subject(&prompts, &conc_prompts, i);
        let out = sched.with_executor(|e| {
            attribute_buckets(
                e.forward_mut(),
                &subject_set,
                subject_i,
                batch,
                width,
                attribution_steps(at, args.conc_tokens.max(8)),
                per_seq,
                prefill_chunk,
                d.as_mut(),
                None,
                // The scheduler's batch is not one a direct drive reproduces in any order, so
                // this is a neighbouring run either way; it keeps the probes' order (ws-p11).
                Subject::SlotZero,
            )
        });
        match out {
            Err(e) => {
                fail(&format!(
                    "concurrency {c}: stream over prompt {i} differs at token {at} and the \
                     attribution could not run: {e}"
                ));
                failures += 1;
            }
            Ok(v) => {
                println!("conc      concurrency {c}, prompt {i} (the arm's token {at}):");
                for l in v.lines() {
                    println!("{l}");
                }
                match v.verdict_at(facts.tp, Some(&rate)) {
                    Ok(line) => println!("OK        concurrency {c}, prompt {i}: {line}"),
                    Err(why) => {
                        fail(&format!(
                            "concurrency {c}: prompt {i} at token {at} — {why}"
                        ));
                        failures += 1;
                    }
                }
            }
        }
    }
    // ---- THE RESIDUE ARM (ws-h31), LAST OF ALL --------------------------------------------------
    //
    // Under `--host-trace` only, and after every arm this gate publishes: the same steps before
    // and after the attribution, per HOST phase, with `--no-attribute` as the control.
    if args.engine.host_trace {
        let mut d = fresh_drafter();
        let out = sched.with_executor(|e| {
            host_trace_residue(
                e.forward_mut(),
                &prompts,
                0,
                // BOTH CELLS ws-t3 PUBLISHED, in that order: batch 1 is the headline and the
                // batch the gate ran is the one whose step is 1.4x larger, which is what says
                // whether the residue scales with the shape's work or is flat in it.
                &if use_batch > 1 {
                    vec![1, use_batch]
                } else {
                    vec![1]
                },
                width,
                args.batch_gen.max(20),
                per_seq,
                prefill_chunk,
                d.as_mut(),
                !args.no_attribute,
            )
        });
        match out {
            Ok(text) => print!("{text}"),
            Err(e) => {
                fail(&format!("the host-trace residue arm could not run: {e:#}"));
                failures += 1;
            }
        }
    }
    let c = sched.with_executor(|e| e.counters());
    println!("counters  {}", c.line());
    if c.clean() {
        println!("OK        every concurrent step ran a declared shape at its declared count");
    } else {
        fail(&format!("the engine counters are not clean: {}", c.line()));
        failures += 1;
    }
    if failures == 0 {
        println!(
            "OK        every stream's ids are its solo ids, at concurrency {:?} and under a mixed \
             arrival",
            args.conc
        );
    }

    // ---- SEQUENCE GROUPS: `n > 1`, forked after the leader's prefill (V4) ----------------------
    if !args.fork_n.is_empty() {
        failures += fork_arm(&sched, &prompts, args, fail);
    }

    // The engine's thread is joined here, before the second one takes the card. Joining it is also
    // what sends rank 1 its `Stop`, so the worker is waited on immediately afterwards.
    drop(sched);
    if let Some(w) = worker.take() {
        println!("tier      {}", w.stop_checked());
    }

    // ---- the eager side of the path gate ---------------------------------------------------------
    if args.engine.capture {
        let mut eager_args = args.engine.clone();
        eager_args.capture = false;
        let (eager_args, mut eager_worker) = start_tier(&eager_args)?;
        let (mut eager, _ledger) =
            match saber_engine::spawn(eager_args.config(), eager_args.ledger(32)?) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(w) = eager_worker.take() {
                        println!("tier      {}", w.stop());
                    }
                    return Err(anyhow::Error::from(e).context("the eager engine"));
                }
            };
        // THE SAME DRAFTER THE GRAPH ARM RAN, and it has to be the same or the arm is not a path
        // comparison at all. Until S5 this was a `FixedDrafter` unconditionally, while the graph
        // half ran the engine's own `EngineDrafter` whenever `--draft-pack` made the drafter
        // resident: the eager engine then verified seven copies of `FIXED_DRAFT_TOKEN` every step,
        // rejected all of them by construction, and the arm reported the drafter's own acceptance
        // as **0 accepted of 2,513** — which is exactly `359 steps x 7`, and is the mock's number
        // and not the eager path's. It also made the arm's id comparison meaningless: a
        // zero-acceptance arm commits one token a step where the graph arm commits ~1.8, so the
        // two lists were different lengths for the same number of STEPS (S4's open items 3 and 4).
        let eager_resident = eager.facts().draft_resident.is_some();
        if eager_resident != engine_drafts {
            fail(&format!(
                "the graph engine's drafter is {} and the eager engine's is {}: two programs",
                if engine_drafts { "RESIDENT" } else { "absent" },
                if eager_resident { "RESIDENT" } else { "absent" },
            ));
            failures += 1;
        }
        let mut eager_drafter: Box<dyn Drafter> = if width <= 1 {
            Box::new(NoDrafter)
        } else if eager_resident {
            Box::new(EngineDrafter::new(width - 1))
        } else {
            Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, width - 1))
        };
        println!(
            "eager     the eager engine drives the `{}` drafter, the same one the graph arm ran",
            eager_drafter.name()
        );
        let mut one = vec![admit_and_prefill(
            &mut eager,
            SeqId(400),
            &prompts[0],
            &blocks_for(0, per_seq),
            prefill_chunk,
        )?];
        let mut eager_trace: Vec<DraftStep> = Vec::new();
        let ms = decode_traced(
            &mut eager,
            &mut one,
            1,
            width,
            eager_drafter.as_mut(),
            args.gen.saturating_sub(1),
            Some(&mut eager_trace),
            None,
            None,
        )?;
        let timed: Vec<f64> = ms.iter().copied().skip(args.warmup).collect();
        let eager_b1 = p50(&timed);
        // Released before the sweep admits its own: the engine has exactly `max_seqs` slots and
        // refuses a ninth by name (`AtCapacity`), which is how this leak announced itself.
        eager.release(one[0].seq);
        let eager_ids = one.remove(0).ids;
        println!();
        println!(
            "eager     batch 1: {eager_b1:.2} ms/step (p50) over {} steps, against {b1:.2} graph \
             — the capture saves {:.2} ms/step ({:.1}%)",
            timed.len(),
            eager_b1 - b1,
            (eager_b1 - b1) / eager_b1 * 100.0
        );
        let mut eager_sweep: Vec<(u32, f64, f64)> = Vec::new();
        for b in batches.iter().copied().filter(|b| *b > 1) {
            if b > eager_args.max_seqs {
                continue;
            }
            // The eager sweep holds `b` sequences too: the same pool law as the graph sweep (H44).
            let longest = (0..b as usize)
                .map(|i| prompts[i % prompts.len()].len())
                .max()
                .unwrap_or(0);
            let need =
                (longest + (args.batch_gen + 1) * width.max(1) as usize + 2).div_ceil(16) as u32;
            let pool = eager.facts().kv_pool_blocks;
            let pages = match pages_for_live(b as usize, per_seq, pool, need) {
                Ok(n) => n,
                Err(e) => {
                    fail(&format!(
                        "eager sweep b{b}s{width}: {e}; the batch is not run"
                    ));
                    failures += 1;
                    continue;
                }
            };
            let mut live = Vec::new();
            for i in 0..b as usize {
                let p = &prompts[i % prompts.len()];
                live.push(admit_and_prefill(
                    &mut eager,
                    SeqId(500 + i as u64),
                    p,
                    &blocks_for(i, pages),
                    prefill_chunk,
                )?);
            }
            let ms = decode_together(
                &mut eager,
                &mut live,
                b,
                width,
                eager_drafter.as_mut(),
                args.batch_gen,
            )?;
            let timed: Vec<f64> = ms.iter().copied().skip(args.warmup).collect();
            eager_sweep.push((b, mean(&timed), p50(&timed)));
            for d in &live {
                eager.release(d.seq);
            }
        }
        for (b, ms, med) in &eager_sweep {
            println!(
                "          batch {b}: {ms:.2} / {med:.2} ms/step eager, {:.2} ms/step/sequence at \
                 the p50",
                med / f64::from(*b)
            );
        }
        drop(eager);
        if let Some(w) = eager_worker.take() {
            println!("tier      {}", w.stop_checked());
        }
        // THE DISCRIMINATING HALF, FIRST. The proposal and the counters it moved, step by step:
        // if the two paths' drafters ever propose a different block, that is the defect, and it is
        // named at the step it happened on rather than inferred from an id list whose LENGTH moved
        // three hundred steps later.
        failures += draft_trace_verdict(&solo_trace, &eager_trace, "graph", "eager", &fail);
        // ...and then the ids, on the COMMON PREFIX with both lengths beside it (S4 item 3). A
        // fixed number of STEPS is a variable number of TOKENS under speculation, so the old
        // whole-list equality named an index at which both lists held the same id.
        let v = PrefixVerdict::of(&eager_ids, &solo[0]);
        match (v.prefix_agrees(), v.lengths_agree()) {
            (true, true) => println!(
                "OK        {} ids identical replayed (graph) and eager, and the two paths \
                 committed the same number of them",
                v.common
            ),
            (true, false) => {
                fail(&format!(
                    "the graph and the eager path agree on every one of the {} ids they both \
                     produced and committed a DIFFERENT NUMBER of them in {} step(s) — graph {}, \
                     eager {}. That is an ACCEPTANCE difference, not a token disagreement; the \
                     draft trace above names the step it starts on",
                    v.common,
                    args.gen.saturating_sub(1),
                    v.len_b,
                    v.len_a
                ));
                failures += 1;
            }
            (false, _) => {
                let at = v.first_diff.unwrap_or(0);
                fail(&format!(
                    "the graph and the eager path disagree at token {at}: eager {:?} vs graph \
                     {:?}; {}",
                    eager_ids.get(at),
                    solo[0].get(at),
                    v.line()
                ));
                failures += 1;
            }
        }
    }

    // THE PROPOSAL AFTER THE COMMIT WAS COLLECTED EVERY TIME (ws-f37). A stash the engine had to
    // abandon is a caller that forgot `propose_after_commit` — a drafter that silently stopped for
    // a step — and every arm above is such a caller, so the count is the gate's to read.
    {
        let dc = draft_counters.snapshot();
        if dc.uncollected > 0 {
            fail(&format!(
                "{} slot(s) had a proposal stashed for the committed anchor that was never \
                 collected (after the commit {}, in the step {})",
                dc.uncollected, dc.after_commit, dc.in_step
            ));
            failures += 1;
        } else if dc.after_commit + dc.in_step > 0 {
            println!(
                "OK        proposal order: {} proposal(s) after the commit, {} in the step, 0 \
                 uncollected",
                dc.after_commit, dc.in_step
            );
        }
    }

    if failures == 0 {
        println!();
        println!(
            "PASS      the P3 seam is structurally sound at batch 1..{} (numerics partial, see the \
             banner)",
            batches.last().copied().unwrap_or(1)
        );
    }
    Ok(failures)
}

/// `n > 1` on the card: one prefill, `n` branches, each a draw of its own from the leader's row.
///
/// Five questions, in one boot and on one scheduler, because each of them can only be asked of a
/// real group:
///
/// 1. **the prefill runs ONCE.** `prefill_steps` is the engine's own counter and it must grow by
///    one per REQUEST, not per branch. A branch that quietly re-prefilled would produce perfectly
///    plausible text and cost `n` prompts;
/// 2. **every branch's first token is a draw from the leader's row at that branch's key.** The
///    sample dump carries the row, the key and the token for every sampled row including the
///    forked ones, and `saber sample-check` replays it through the shipped sampler — which is the
///    same instrument V2 and V3 used, and the only one that can see past the serving plane;
/// 3. **temperature 0 is one answer `n` times** (the control), and it is the same answer the
///    prompt gets at `n = 1`. It is asked here rather than over HTTP because prod's own
///    `_verify_greedy_sampling` refuses `n > 1` with temperature 0 at the API, and `validate()`
///    reproduces that;
/// 4. **the census**: the group's KV is one prompt and `n` tails, and its state is `n` slot sets
///    whatever the prompt costs — the allocator's own numbers, which is what admission asks;
/// 5. **the use-after-free control**: the same seeded group run twice, the second time with
///    another request admitted into the pages a finished branch hands back. The survivor's ids
///    must not move. The seed is what makes the two runs comparable at all.
fn fork_arm(
    sched: &Arc<Scheduler<RealExecutor<HipForward>>>,
    prompts: &[Vec<TokenId>],
    args: &GateArgs,
    fail: impl Fn(&str),
) -> usize {
    let mut failures = 0usize;
    let tokens = args.fork_tokens.max(1);
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            fail(&format!("a runtime to drain the fork arm's streams: {e}"));
            return 1;
        }
    };
    println!();
    println!(
        "fork      n > 1 over {} prompt(s) x {tokens} tokens at prod's own defaults \
         (0.7 / 0.95 / 20), seeded",
        prompts.len().min(8)
    );
    if let Some(path) = &args.sample_dump {
        match sched.with_executor(|e| e.dump_samples_to(path)) {
            Ok(()) => println!(
                "          every sampled row is dumped to {}",
                path.display()
            ),
            Err(e) => {
                fail(&format!("the sample dump could not be opened: {e}"));
                failures += 1;
            }
        }
    }

    // A group's parameters: prod's own, plus a seed per request so the whole arm is reproducible.
    let group_params = |n: usize, seed: u64, temperature: f32| SamplingParams {
        n,
        temperature,
        top_p: 0.95,
        top_k: 20,
        seed: Some(seed),
        max_tokens: Some(tokens),
        ..SamplingParams::default()
    };

    // Drain the scheduler until nothing is running or queued, with a hard cap, and abort what is
    // left. A stream whose request never finished never ends, so a collect on it would hang this
    // gate behind its container timeout instead of failing it.
    let drain_to_idle = |cap: usize, ids: &[saber_serve::engine::RequestId]| {
        let mut spent = 0usize;
        while sched.has_work() && spent < cap {
            let ran = sched.run_steps(64);
            spent += ran.max(1);
            if ran == 0 {
                break;
            }
        }
        if sched.has_work() {
            println!("          WARN the step budget ran out with work in flight; aborting it");
            for id in ids {
                sched.abort(*id);
            }
            sched.run_steps(16);
        }
    };

    // Run `requests` and return each (request index, choice) stream, in submission order.
    let run_group = |params: Vec<SamplingParams>,
                     prompts: &[Vec<TokenId>]|
     -> Option<Vec<BTreeMap<u32, Vec<TokenId>>>> {
        let mut subs = Vec::new();
        for (i, p) in params.into_iter().enumerate() {
            match sched.submit(EngineRequest {
                prompt_token_ids: prompts[i % prompts.len()].clone(),
                params: p,
            }) {
                Ok(s) => subs.push(s),
                Err(e) => {
                    fail(&format!("a group was refused at submit: {e}"));
                    return None;
                }
            }
        }
        let ids: Vec<saber_serve::engine::RequestId> = subs.iter().map(|s| s.id).collect();
        drain_to_idle(tokens * 64 + 4096, &ids);
        let mut out = Vec::new();
        for sub in subs {
            let events = runtime.block_on(async move {
                use futures_util::StreamExt;
                sub.events.collect::<Vec<_>>().await
            });
            let mut by_choice: BTreeMap<u32, Vec<TokenId>> = BTreeMap::new();
            for e in &events {
                by_choice
                    .entry(e.choice_index)
                    .or_default()
                    .extend(e.new_tokens.iter().copied());
            }
            out.push(by_choice);
        }
        Some(out)
    };

    let corpus: Vec<Vec<TokenId>> = prompts.iter().take(8).cloned().collect();
    for &n in &args.fork_n {
        if n < 2 {
            continue;
        }
        let before = sched.with_executor(|e| e.counters());
        let free_before = sched.arena_free();
        let params: Vec<SamplingParams> = (0..corpus.len())
            .map(|i| group_params(n, 90_210 + i as u64, 0.7))
            .collect();
        let Some(runs) = run_group(params.clone(), &corpus) else {
            return failures + 1;
        };
        let after = sched.with_executor(|e| e.counters());
        let sc = sched.counters();

        let branches: usize = runs.iter().map(BTreeMap::len).sum();
        let empty = runs
            .iter()
            .flat_map(|r| r.values())
            .filter(|ids| ids.is_empty())
            .count();
        println!(
            "          n = {n}: {branches} branches over {} requests, {} prefill step(s), \
             forks {} ({} refused)",
            runs.len(),
            after.prefill_steps - before.prefill_steps,
            after.forks - before.forks,
            after.fork_refusals - before.fork_refusals,
        );
        if branches == n * runs.len() && empty == 0 {
            println!("OK        every request returned {n} non-empty branches");
        } else {
            fail(&format!(
                "n = {n}: {branches} branches for {} requests, {empty} of them empty",
                runs.len()
            ));
            failures += 1;
        }
        // THE CLAIM THE WHOLE DESIGN IS FOR.
        let prefills = after.prefill_steps - before.prefill_steps;
        let want_prefills = runs.len() as u64;
        if prefills == want_prefills {
            println!("OK        ONE prefill per request, not one per branch ({prefills})");
        } else {
            fail(&format!(
                "n = {n}: {prefills} prefill steps for {want_prefills} requests — a branch is \
                 prefilling"
            ));
            failures += 1;
        }
        let want_forks = (n as u64 - 1) * runs.len() as u64;
        if after.forks - before.forks == want_forks && after.fork_refusals == before.fork_refusals {
            println!("OK        {want_forks} branches forked, none refused");
        } else {
            fail(&format!(
                "n = {n}: {} forks and {} refusals, expected {want_forks} and 0",
                after.forks - before.forks,
                after.fork_refusals - before.fork_refusals
            ));
            failures += 1;
        }
        if sc.fork_branches_orphaned == 0 && sc.fork_failures == 0 {
            println!("OK        no branch was orphaned or failed in the scheduler");
        } else {
            fail(&format!(
                "n = {n}: {} orphaned, {} failed",
                sc.fork_branches_orphaned, sc.fork_failures
            ));
            failures += 1;
        }
        if after.clean() {
            println!(
                "OK        every step of the group ran a declared shape at its declared count"
            );
        } else {
            fail(&format!(
                "the engine counters are not clean: {}",
                after.line()
            ));
            failures += 1;
        }
        // The branches are DRAWS: within a request, at least two of them differ. (Weak on
        // purpose — two draws of a peaked row may agree, and demanding `n` distinct streams would
        // be asserting a property of the noise rather than of the fork.)
        let all_same = runs
            .iter()
            .filter(|r| r.len() > 1)
            .filter(|r| {
                let first = r.values().next().expect("non-empty");
                r.values().all(|v| v == first)
            })
            .count();
        println!(
            "          {} of {} requests produced {n} identical branches",
            all_same,
            runs.len()
        );
        if all_same < runs.len() {
            println!("OK        the branches are draws, not copies of the leader");
        } else {
            fail("every request's branches were identical: the choice index is not in the key");
            failures += 1;
        }
        // Reproducible, branch for branch.
        let Some(again) = run_group(params, &corpus) else {
            return failures + 1;
        };
        if again == runs {
            println!("OK        the same seeds give the same {n} streams, branch for branch");
        } else {
            fail(&format!("n = {n}: a seeded group did not repeat itself"));
            failures += 1;
        }
        let free_after = sched.arena_free();
        if free_after == free_before {
            println!("OK        every page and slot set the group held came back");
        } else {
            fail(&format!(
                "n = {n}: the arenas were {free_before:?} before the group and {free_after:?} after"
            ));
            failures += 1;
        }
    }

    // ---- the greedy control: one answer n times, and it is the n = 1 answer --------------------
    //
    // LIKE WITH LIKE (ws-f39). A lone n = 1 request decodes at batch 1, which at TP=2 resolves the
    // EXACT all-reduce, and a group of two decodes at batch >= 2, which on the default band is
    // the lossy `allreduce_wht` — two arithmetics, and F38's hold 1 read that as a FAIL of this
    // control on the default band (it passed under `--ar-exact` and at TP=1). So both sides now
    // run beside the same COMPANION request (another prompt, twice as long), which holds every
    // decode step of the compared stream at batch >= 2 in the same bucket; the control then asks
    // only what it means to ask — is a greedy branch the single request's stream.
    let companion = SamplingParams {
        max_tokens: Some(tokens * 2),
        ..group_params(1, 1, 0.0)
    };
    let pair: Vec<Vec<TokenId>> = corpus.iter().take(2).cloned().collect();
    let Some(greedy) = run_group(vec![group_params(2, 1, 0.0), companion.clone()], &pair) else {
        return failures + 1;
    };
    let Some(alone) = run_group(vec![group_params(1, 1, 0.0), companion], &pair) else {
        return failures + 1;
    };
    let two = &greedy[0];
    if two.len() == 2 && two[&0] == two[&1] && !two[&0].is_empty() {
        println!(
            "OK        temperature 0, n = 2: two identical branches of {} tokens",
            two[&0].len()
        );
    } else {
        fail("the greedy control's two branches are not identical");
        failures += 1;
    }
    if two.get(&0) == alone[0].get(&0) {
        println!(
            "OK        and it is the stream the same prompt gets at n = 1 (both beside one \
             companion request: batch >= 2 on both sides)"
        );
    } else {
        fail("a greedy group moved the stream the prompt gets alone");
        failures += 1;
    }

    // ---- THE CARD EQUIVALENCE LAW (ws-f38) ------------------------------------------------------
    failures += fork_equivalence(sched, &corpus, tokens, &run_group, &drain_to_idle, &fail);

    // ---- the census ----------------------------------------------------------------------------
    // The allocator's own numbers, taken while the group is resident: the prompt's WHOLE pages are
    // billed once and each branch's tail to itself.
    let free_before = sched.arena_free();
    let prompt = corpus[0].clone();
    let census = sched.submit(EngineRequest {
        prompt_token_ids: prompt.clone(),
        params: group_params(3, 5, 0.0),
    });
    match census {
        Ok(sub) => {
            // Admission, the leader's prefill chunks and the fork.
            sched.run_steps(8);
            let (kv_now, state_now) = sched.arena_free();
            let kv_used = free_before.0 - kv_now;
            let state_used = free_before.1 - state_now;
            let block_tokens = 16u64;
            // The scheduler's own reservation (`tokens_to_reserve`): prompt + max_tokens + the
            // widest verify width + 1. At width 1 that is +2 tokens, which V4's census left out
            // and never saw (234 + 64 rounds to the same 19 blocks either way); at width 8 it is
            // +9 and a block more (ws-f38, the first run of this arm under the drafter).
            let slack = u64::from(sched.config().max_spec_width()) + 1;
            let per_seq =
                (prompt.len() as u64 + tokens as u64 + slack).div_ceil(block_tokens) as u32;
            let shared = (prompt.len() as u64 / block_tokens) as u32;
            let alone = 3 * per_seq;
            println!(
                "census    group of 3 over a {}-token prompt: {kv_used} KV block(s) and \
                 {state_used} state slot set(s); three single submits would hold {alone} blocks \
                 ({per_seq} each), of which {} pages are shared",
                prompt.len(),
                2 * shared
            );
            if state_used == 3 {
                println!("OK        one GDN state slot set per branch, always");
            } else {
                fail(&format!("{state_used} state slot sets for three branches"));
                failures += 1;
            }
            if kv_used == alone - 2 * shared {
                println!(
                    "OK        the prompt's {shared} whole pages are billed ONCE and each \
                     branch's tail to itself"
                );
            } else {
                fail(&format!(
                    "{kv_used} KV blocks; the sharing law says {}",
                    alone - 2 * shared
                ));
                failures += 1;
            }
            drain_to_idle(tokens * 64 + 4096, &[sub.id]);
            let _ = runtime.block_on(async move {
                use futures_util::StreamExt;
                sub.events.collect::<Vec<_>>().await
            });
            if sched.arena_free() == free_before {
                println!("OK        every shared block came back when the last branch released it");
            } else {
                fail("the group's arenas did not come back");
                failures += 1;
            }
        }
        Err(e) => {
            fail(&format!("the census group was refused: {e}"));
            failures += 1;
        }
    }

    // ---- the use-after-free control --------------------------------------------------------------
    // The same seeded group twice: alone, and with another request admitted beside it and into the
    // pages its first finished branch hands back. The survivor's ids must not move.
    let Some(solo) = run_group(vec![group_params(2, 777, 0.7)], &corpus[..1]) else {
        return failures + 1;
    };
    let intruded = {
        let sub = sched.submit(EngineRequest {
            prompt_token_ids: corpus[0].clone(),
            params: group_params(2, 777, 0.7),
        });
        match sub {
            Ok(sub) => {
                // Let the group get going, then put an intruder in beside it. It asks for a few
                // tokens, so it finishes and is re-admitted repeatedly into whatever comes free —
                // which is what makes it an intruder rather than a neighbour.
                sched.run_steps(tokens / 2 + 4);
                let mut intruders = Vec::new();
                for k in 0..4u64 {
                    if let Ok(s) = sched.submit(EngineRequest {
                        prompt_token_ids: corpus[(1 + k as usize) % corpus.len()].clone(),
                        params: SamplingParams {
                            n: 1,
                            temperature: 0.0,
                            max_tokens: Some(4),
                            ..SamplingParams::default()
                        },
                    }) {
                        intruders.push(s);
                    }
                    sched.run_steps(8);
                }
                let mut ids: Vec<saber_serve::engine::RequestId> =
                    intruders.iter().map(|s| s.id).collect();
                ids.push(sub.id);
                drain_to_idle(tokens * 64 + 4096, &ids);
                for s in intruders {
                    let _ = runtime.block_on(async move {
                        use futures_util::StreamExt;
                        s.events.collect::<Vec<_>>().await
                    });
                }
                let events = runtime.block_on(async move {
                    use futures_util::StreamExt;
                    sub.events.collect::<Vec<_>>().await
                });
                let mut by_choice: BTreeMap<u32, Vec<TokenId>> = BTreeMap::new();
                for e in &events {
                    by_choice
                        .entry(e.choice_index)
                        .or_default()
                        .extend(e.new_tokens.iter().copied());
                }
                Some(by_choice)
            }
            Err(e) => {
                fail(&format!("the use-after-free control was refused: {e}"));
                failures += 1;
                None
            }
        }
    };
    if let Some(intruded) = intruded {
        if intruded == solo[0] {
            println!(
                "OK        four sessions admitted into the pages this group freed changed \
                 neither branch's ids"
            );
        } else {
            let which: Vec<u32> = intruded
                .iter()
                .filter(|(c, ids)| solo[0].get(*c) != Some(*ids))
                .map(|(c, _)| *c)
                .collect();
            fail(&format!(
                "a group's ids moved when other sessions reused its pages: choice(s) {which:?}"
            ));
            failures += 1;
        }
    }

    let c = sched.with_executor(|e| e.counters());
    println!("counters  {} | {}", c.fork_line(), c.sampler_line());
    if failures == 0 {
        println!("OK        the fork arm passes: one prefill, n draws, and the pages behave");
    }
    failures
}

/// The fork arm's group driver: requests in, each request's streams by choice out.
type RunGroup<'a> =
    dyn Fn(Vec<SamplingParams>, &[Vec<TokenId>]) -> Option<Vec<BTreeMap<u32, Vec<TokenId>>>> + 'a;

/// **The fork's equivalence law on the card** (ws-f38), with the drafter resident or not.
///
/// * **temperature 0**: every branch of an `n = 2` and an `n = 4` group is the stream the same
///   prompt gets as a single request, over four prompts — which, under a drafter, is the whole of
///   (a) and (b): a branch whose drafter pages were not the leader's, or whose first block was not
///   its own, would still produce these ids (`draft::Greedy`) but would propose different blocks,
///   so the drafter counters below are the other half of the claim;
/// * **seeded, temperature 0.7**: choice 0 of an `n = 2` group is the single request with the same
///   seed (the leader runs the single request's code), and the group repeats itself;
/// * **with a park in between** (when the engine has a park staging buffer): the same groups,
///   parked after a few steps and restored into fresh pages, slots and drafter runs, are the
///   unparked groups branch for branch;
/// * **the census**: every fork made a drafter session and a first block from the branch's own
///   token, no proposal went uncollected, no anchor was stale, nothing was refused.
fn fork_equivalence(
    sched: &Arc<Scheduler<RealExecutor<HipForward>>>,
    corpus: &[Vec<TokenId>],
    tokens: usize,
    run_group: &RunGroup<'_>,
    drain_to_idle: &dyn Fn(usize, &[saber_serve::engine::RequestId]),
    fail: &dyn Fn(&str),
) -> usize {
    let mut failures = 0usize;
    let params = |n: usize, seed: u64, temperature: f32| SamplingParams {
        n,
        temperature,
        top_p: 0.95,
        top_k: 20,
        seed: Some(seed),
        max_tokens: Some(tokens),
        ..SamplingParams::default()
    };
    let drafted = sched.with_executor(|e| e.facts().draft_resident.is_some());
    let draft0 = sched.with_executor(|e| e.forward_mut().draft_counts());
    let exec0 = sched.with_executor(|e| e.counters());
    println!();
    println!(
        "equiv     ws-f38: branches against single requests, {} prompt(s) x {tokens} tokens, \
         drafter {}",
        corpus.len().min(4),
        if drafted { "RESIDENT" } else { "absent" }
    );
    let prompts: Vec<Vec<TokenId>> = corpus.iter().take(4).cloned().collect();

    // Temperature 0: every branch is the single request's stream.
    let Some(alone) = run_group(
        prompts.iter().map(|_| params(1, 1, 0.0)).collect(),
        &prompts,
    ) else {
        return failures + 1;
    };
    let mut groups_t0: BTreeMap<usize, Vec<BTreeMap<u32, Vec<TokenId>>>> = BTreeMap::new();
    for n in [2usize, 4] {
        let Some(g) = run_group(
            prompts.iter().map(|_| params(n, 1, 0.0)).collect(),
            &prompts,
        ) else {
            return failures + 1;
        };
        let mut bad = Vec::new();
        for (i, (group, solo)) in g.iter().zip(&alone).enumerate() {
            let want = solo.get(&0);
            if group.len() != n || group.values().any(|ids| Some(ids) != want) {
                bad.push(i);
            }
        }
        if bad.is_empty()
            && alone
                .iter()
                .all(|a| a.get(&0).is_some_and(|v| !v.is_empty()))
        {
            println!(
                "OK        temperature 0, n = {n}: all {} branches over {} prompts are the single \
                 request's stream, token for token",
                n * prompts.len(),
                prompts.len()
            );
        } else {
            fail(&format!(
                "temperature 0, n = {n}: prompts {bad:?} have a branch that is not the single \
                 request's stream"
            ));
            failures += 1;
        }
        groups_t0.insert(n, g);
    }

    // Seeded, temperature 0.7: choice 0 is the single request with that seed; the group repeats.
    let seeded = |n: usize| -> Vec<SamplingParams> {
        (0..prompts.len())
            .map(|i| params(n, 31_337 + i as u64, 0.7))
            .collect()
    };
    let (Some(solo7), Some(g7), Some(g7b)) = (
        run_group(seeded(1), &prompts),
        run_group(seeded(2), &prompts),
        run_group(seeded(2), &prompts),
    ) else {
        return failures + 1;
    };
    let leader_is_solo = g7
        .iter()
        .zip(&solo7)
        .filter(|(g, s)| g.get(&0) == s.get(&0))
        .count();
    if leader_is_solo == prompts.len() {
        println!(
            "OK        seeded 0.7, n = 2: choice 0 of every group is the single request with the \
             same seed ({leader_is_solo} of {})",
            prompts.len()
        );
    } else {
        fail(&format!(
            "seeded 0.7, n = 2: choice 0 is the single request's stream on {leader_is_solo} of {} \
             prompts",
            prompts.len()
        ));
        failures += 1;
    }
    if g7 == g7b {
        println!("OK        seeded 0.7, n = 2: the group repeats itself branch for branch");
    } else {
        fail("seeded 0.7, n = 2: a seeded group did not repeat itself");
        failures += 1;
    }
    let differ = g7.iter().filter(|g| g.get(&0) != g.get(&1)).count();
    println!(
        "          seeded 0.7, n = 2: {differ} of {} groups have two different branches (draws, \
         not copies)",
        g7.len()
    );

    // The same groups with a PARK in between: after a few steps, both branches imaged and
    // released, then restored into whatever the arenas hand out next.
    let can_park = sched.with_executor(|e| e.facts().park_staging_bytes > 0);
    if can_park {
        let park0 = sched.with_executor(|e| e.counters());
        let parked_run =
            |p: SamplingParams, prompt: &Vec<TokenId>| -> Option<BTreeMap<u32, Vec<TokenId>>> {
                let sub = match sched.submit(EngineRequest {
                    prompt_token_ids: prompt.clone(),
                    params: p,
                }) {
                    Ok(s) => s,
                    Err(e) => {
                        fail(&format!("a parked group was refused at submit: {e}"));
                        return None;
                    }
                };
                // Admission, the prefill, the fork, and a few decode steps with speculation.
                sched.run_steps(6);
                if let Err(e) = sched.park(sub.id) {
                    fail(&format!("the group could not be parked: {e:?}"));
                    sched.abort(sub.id);
                    drain_to_idle(4096, &[sub.id]);
                    return None;
                }
                // The park runs OFF the step loop, a branch at a time through the one staging
                // slot (ws-f38's queue); a step with nothing steppable is idle and returns, and
                // each call polls the transfers once. Bounded, and a group that never lands is
                // a failure rather than a hang.
                let mut polls = 0usize;
                while !sched.parked_ids().contains(&sub.id) && polls < 20_000 {
                    sched.run_steps(1);
                    polls += 1;
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                if !sched.parked_ids().contains(&sub.id) {
                    fail("the group's park never landed");
                    sched.abort(sub.id);
                    drain_to_idle(4096, &[sub.id]);
                    return None;
                }
                if let Err(e) = sched.unpark(sub.id) {
                    fail(&format!("the group could not be unparked: {e:?}"));
                    return None;
                }
                // The restore, the same way; then the rest of the stream.
                let mut polls = 0usize;
                while sched.has_work() && polls < 200_000 {
                    if sched.run_steps(64) == 0 {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                    polls += 1;
                }
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .ok()?;
                let events = runtime.block_on(async move {
                    use futures_util::StreamExt;
                    sub.events.collect::<Vec<_>>().await
                });
                let mut by_choice: BTreeMap<u32, Vec<TokenId>> = BTreeMap::new();
                for e in &events {
                    by_choice
                        .entry(e.choice_index)
                        .or_default()
                        .extend(e.new_tokens.iter().copied());
                }
                Some(by_choice)
            };
        let mut same = 0usize;
        let mut total = 0usize;
        // THE REFERENCE IS THE SAME GROUP RUN ALONE, unparked (ws-f39). The groups above ran four
        // prompts at once (eight sequences through four seats), and a SEEDED draw is keyed by the
        // sequence's step, whose boundaries are the drafter's acceptance — which a batch moves
        // (P11: a batched step is not slot-invariant at the drafter). So a seeded group parked
        // alone and compared with the same group run in a crowd differs on a different key, not
        // on a different state: both F39 holds read exactly that on prompt 1's seeded group while
        // every temperature-0 group matched. Against the group alone, the park is the ONLY
        // difference; the crowd's groups stay the temperature-0 cross-check.
        for (i, prompt) in prompts.iter().enumerate().take(2) {
            for (label, p, crowd) in [
                (
                    "temperature 0, n = 2",
                    params(2, 1, 0.0),
                    Some(&groups_t0[&2][i]),
                ),
                (
                    "temperature 0, n = 4",
                    params(4, 1, 0.0),
                    Some(&groups_t0[&4][i]),
                ),
                ("seeded 0.7, n = 2", params(2, 31_337 + i as u64, 0.7), None),
            ] {
                total += 1;
                let Some(alone) = run_group(vec![p.clone()], std::slice::from_ref(prompt)) else {
                    continue;
                };
                let want = &alone[0];
                if let Some(c) = crowd {
                    if c != want {
                        fail(&format!(
                            "prompt {i}, {label}: at temperature 0 the group alone is not the group \
                             in a crowd"
                        ));
                    }
                }
                match parked_run(p, prompt) {
                    Some(got) if &got == want => same += 1,
                    Some(_) => fail(&format!(
                        "prompt {i}, {label}: the group parked and restored is not the same group \
                         run alone"
                    )),
                    None => {}
                }
            }
        }
        let park1 = sched.with_executor(|e| e.counters());
        println!(
            "          park arm: {} park(s), {} restore(s), {} park / {} restore refusal(s)",
            park1.parks - park0.parks,
            park1.restores - park0.restores,
            park1.park_failures - park0.park_failures,
            park1.restore_failures - park0.restore_failures
        );
        // THE FOLLOWER'S HALVES, AFTER (ws-f39). At TP=2 every branch image is two halves and the
        // follower holds its own until the leader restores or forgets it; a leg that restored
        // everything leaves it holding NONE. Asked with a discard of a name nothing was parked
        // under, which drops nothing and reports what the follower still holds (ws-k18's probe;
        // at TP=1 there is no follower and it reports 0).
        let probe = sched.with_executor(|e| e.forward_mut().discard_parked(SeqId(u64::MAX - 39)));
        println!(
            "          park arm: after it, the follower holds {} half-image(s), the leader {} \
             receipt(s)",
            probe.peer_remaining, probe.receipts_remaining
        );
        if probe.held() || probe.peer_remaining != 0 || probe.receipts_remaining != 0 {
            fail(&format!(
                "the park leg left parked halves behind: follower {}, leader receipts {}",
                probe.peer_remaining, probe.receipts_remaining
            ));
            failures += 1;
        } else {
            println!(
                "OK        every half-image the park leg made was restored: 0 held on either rank"
            );
        }
        if park1.parks - park0.parks != park1.restores - park0.restores {
            fail(&format!(
                "the park leg parked {} image(s) and restored {}",
                park1.parks - park0.parks,
                park1.restores - park0.restores
            ));
            failures += 1;
        }
        if same == total && park1.restores > park0.restores {
            println!(
                "OK        with a park in between: {same} of {total} groups (every branch parked and \
                 restored into fresh pages, slots and drafter runs) are the unparked groups"
            );
        } else {
            fail(&format!(
                "with a park in between: {same} of {total} groups match the unparked run"
            ));
            failures += 1;
        }
    } else {
        println!("          park arm SKIPPED: this engine has no park staging buffer");
    }

    // The census of the drafter's half, and every counter that says a fork went wrong.
    let draft1 = sched.with_executor(|e| e.forward_mut().draft_counts());
    let exec1 = sched.with_executor(|e| e.counters());
    let forks = exec1.forks - exec0.forks;
    let sessions = draft1.fork_sessions - draft0.fork_sessions;
    let pages = draft1.fork_pages - draft0.fork_pages;
    let firsts = draft1.fork_proposals - draft0.fork_proposals;
    let blocks = exec1.fork_blocks - exec0.fork_blocks;
    println!(
        "census    {forks} fork(s): {sessions} drafter session(s), {pages} drafter page(s) copied \
         ({:.1} per branch), {firsts} first block(s) proposed, {blocks} offered; {}",
        pages as f64 / sessions.max(1) as f64,
        exec1.fork_line()
    );
    println!("          drafter: {}", draft1.line());
    println!("          engine:  {}", exec1.proposal_line());
    let want_draft = if drafted { forks } else { 0 };
    let counters_clean = sessions == want_draft
        && firsts == want_draft
        && blocks == want_draft
        && exec1.fork_block_failures == exec0.fork_block_failures
        && exec1.fork_refusals == exec0.fork_refusals
        && exec1.stale_anchor_defects == exec0.stale_anchor_defects
        && exec1.proposal_collect_failures == exec0.proposal_collect_failures
        && draft1.uncollected == draft0.uncollected;
    if counters_clean && forks > 0 {
        println!(
            "OK        every branch got its own drafter session and a first block from its own \
             token ({want_draft} of {forks}); uncollected, stale, refused and failed all 0"
        );
    } else {
        fail(&format!(
            "the fork census: {forks} forks, {sessions} drafter sessions, {firsts} first blocks, \
             {blocks} offered (want {want_draft} each); {} / {}",
            exec1.fork_line(),
            draft1.line()
        ));
        failures += 1;
    }
    failures
}

/// `clear_pages`, RETIRED (ws-p9) — and the law that replaced it.
///
/// **What the arm was.** ws-s5 gated `DraftResident::clear_pages` — a `hipMemsetAsync` per layer
/// per cache at admission — with a pair of pairs: one request admitted after ITSELF and after a
/// 32-token predecessor that wrote INSIDE its read window, with the memset on (they must propose
/// the same blocks) and with it off (they must DIFFER). The claim was narrow and true at the time:
/// the drafter's attention reads `last_valid_pos + 1 + block` positions of its own cache, the
/// context store ran on DECODE steps only, so a new session read a prompt's worth of positions it
/// had never written and whatever they held was what the proposal was a function of.
///
/// **Why it is retired.** ws-s8 put the context store on the PREFILL steps, and with it the
/// request writes every position of its own window before it reads any of it: the store covers
/// `0 ..= last_valid_pos` (the prompt's rows on the prefill chunks, each committed token's on the
/// step that committed it) and the proposal's own `block` query rows cover
/// `last_valid_pos + 1 ..= last_valid_pos + block`, which is the window exactly. Nothing at or
/// past `seq_lens` is read at all, so a predecessor that wrote there cannot be seen either. There
/// is no position left for a stale page to be read at — so the CONTROL could not fail, and
/// ws-s8, ws-s9, Q1 and ws-s10 each recorded it failing to fail. A control that cannot fail is not
/// a control (`docs/GATES.md`), and an arm whose control cannot fail proves nothing.
///
/// **What carries the claim now**: `saber_engine::draft_inputs`' `window_law` tests — every
/// position the drafter reads is written by this request first, over seven prompt lengths and
/// three acceptance patterns, with ws-s8's own decode-only store as the control that leaves 274
/// positions unwritten. It is a law over the tables and needs no card.
///
/// **The memset stays**, unchanged and counted (`pages_cleared` / `pages_dirty`): it costs one
/// async memset per admission, it is what makes an unwritten page deterministic if the store ever
/// regresses, and ws-s6 found a real defect with it at two ranks. What is gone is the claim that a
/// gate arm was proving something about it.
fn draft_page_clear_arm(
    forward: &mut HipForward,
    _prompts: &[Vec<TokenId>],
    _per_seq: u32,
    _prefill_chunk: u32,
    width: u32,
    _fail: &impl Fn(&str),
) -> usize {
    use saber_engine::ForwardRunner;

    if forward.facts().draft_resident.is_none() {
        return 0;
    }
    let c = forward.draft_counts();
    println!();
    println!(
        "clear     RETIRED (ws-p9): with ws-s8's prefill context store a request writes every \
         position of its own drafter window before it reads any of it — the store covers \
         `0..=last_valid_pos` and the proposal's own {width} query rows cover the rest of \
         `last_valid_pos + 1 + block` — so no position is left for a predecessor's K/V to be read \
         at, and this arm's CONTROL could not fail (ws-s8, ws-s9, Q1 and ws-s10 each recorded it)"
    );
    println!(
        "          the claim is now `saber_engine::draft_inputs`' window law (host, with the \
         decode-only store as its control); the memset itself is unchanged and counted: {} \
         admission(s) cleared, {} left dirty",
        c.pages_cleared, c.pages_dirty
    );
    0
}

/// **C9's per-row class gate on the card**: the same two sequences stepped twice, once with the
/// mix step SERIAL (the chunk's forward, then the decode half's) and once FUSED (one forward),
/// everything else identical; every sampled row's top-K logits, log-sum-exp and token compared.
///
/// Sequence A decodes (a 40-token prompt, then two steps); sequence B's 150-token prompt is the
/// chunk that rides beside A's step; then A and B decode together for four steps. The rows are:
///
/// * **A's row at the mix step** — a decode row of a fused step. Its class is the rung's (C8's
///   proof: the class proxy's), so under `--mixed-class-proxy` it must be 0 bytes from the serial
///   form's (whose decode half then runs the proxy's plan) and under the default plans it must
///   DIFFER (the control: the comparison sees the class move);
/// * **B's first token** — the chunk's row, whose class the fused step keeps below p2048: 0 bytes
///   under both engines;
/// * **the four steps after** — what the fused step wrote into KV pages and GDN state, read back
///   through attention and the recurrence (reported; under the proxy they must match too).
///
/// The all-reduce CLASS a planned shape's reduces sum in, read off the engine's own plans
/// ([`saber_engine::ForwardFacts::ar_families`]): `exact`, `lossy` (the `wht` families) or `mixed`;
/// `none` when the shape has no reduce (TP=1). Matched on phase and rows (a prefill rung is planned
/// body-only and with the head; both carry the same reduces).
fn ar_class_of(facts: &saber_engine::ForwardFacts, phase: Phase, rows: u32, width: u32) -> String {
    let syms: Vec<&str> = facts
        .ar_families
        .iter()
        .filter(|(s, _, _)| s.phase == phase && s.rows == rows && s.width == width && !s.ragged)
        .map(|(_, sym, _)| sym.as_str())
        .collect();
    let exact = syms.iter().any(|s| s.contains("exact"));
    let lossy = syms.iter().any(|s| !s.contains("exact"));
    match (exact, lossy) {
        (false, false) => "none".to_owned(),
        (true, false) => "exact".to_owned(),
        (false, true) => "lossy".to_owned(),
        (true, true) => "mixed".to_owned(),
    }
}

/// **H46: what the `mixfused` arms PREDICT for the decode row, from the plans** rather than a
/// constant. The fused step's decode row takes the fused RUNG's reduce; its GEMMs are the class
/// proxy's, which the card has read 0 bytes from the decode bucket's own at b1s8 whenever the two
/// reduces agree (B1's `--ar-exact` arm: every `mixfused` decode row 0 bytes). So the row keeps
/// its bytes exactly when the decode bucket and the rung reduce in the same class — under the class
/// proxy (lossy on both sides) and under `--ar-exact` (exact on both sides) — and moves under the
/// default plans at <= 12 decode rows (exact) beside a rung (lossy). Returns `(differs, why)`.
fn predict_mixed_decode_row(
    facts: &saber_engine::ForwardFacts,
    width: u32,
    rung: u32,
) -> (bool, String) {
    let dec = ar_class_of(facts, Phase::Decode, width, width);
    let mix = ar_class_of(facts, Phase::Prefill, rung, 1);
    (
        dec != mix,
        format!("b1s{width} reduces {dec}, the p{rung} rung {mix}"),
    )
}

/// Returns the failure count: 1 when the verdict is not the one this engine's plans predict.
#[allow(clippy::too_many_lines)]
fn mixed_fused_arm(
    forward: &mut HipForward,
    prompts: &[Vec<TokenId>],
    per_seq: u32,
    prefill_chunk: u32,
    width: u32,
    proxy: bool,
) -> anyhow::Result<usize> {
    use saber_engine::ForwardRunner;
    println!();
    if prompts.len() < 2 || prompts[0].is_empty() || prompts[1].is_empty() {
        println!("FAIL      mixfused: needs two non-empty prompts");
        return Ok(1);
    }
    // A's 40-token prompt and B's 150-token chunk, cycled out of the run's first two prompts.
    let pa: Vec<TokenId> = prompts[0].iter().copied().cycle().take(40).collect();
    let pb: Vec<TokenId> = prompts[1].iter().copied().cycle().take(150).collect();
    let k = (width.max(1) - 1) as usize;
    let logprob_row = RowRequest {
        temperature: Some(1.0),
        penalty_ids: Vec::new(),
    };
    // One record per sampled row: (label, token, the seam's row in Debug form — exact f32 bits
    // round-trip through `{:?}`).
    type Rec = Vec<(String, TokenId, String)>;
    let slot_of = |d: &Driven, req: &RowRequest| DecodeSlot {
        seq: d.seq,
        input: *d.ids.last().expect("a prefilled sequence has a token"),
        context_len: d.context - 1,
        draft: vec![RecordedDrafter::WRONG; k],
        committed_before: d.committed_before,
        sampling: req.clone(),
        decider: None,
    };
    let advance = |d: &mut Driven, c: &saber_engine::SlotCommit| {
        d.committed_before = c.tokens.len() as u32;
        d.context += c.tokens.len() as u32;
        d.ids.extend(&c.tokens);
    };
    let mut run = |fused: bool| -> anyhow::Result<(Rec, u32)> {
        let base = if fused { 880_100 } else { 880_000 };
        let mut rec: Rec = Vec::new();
        let mut a = admit_and_prefill(
            forward,
            SeqId(base),
            &pa,
            &blocks_for(0, per_seq),
            prefill_chunk,
        )?;
        for step in 0..2 {
            let mut out = forward.decode(&[slot_of(&a, &logprob_row)], 1, width)?;
            forward.propose_greedy(&mut out)?;
            let c = out
                .commits
                .into_iter()
                .next()
                .context("a decode step committed nothing")?;
            rec.push((
                format!("A step {step}"),
                c.tokens[0],
                format!("{:?}", c.rows.first()),
            ));
            advance(&mut a, &c);
        }
        let b_prompt = &pb[..];
        let bseq = SeqId(base + 1);
        forward.admit(bseq, b_prompt.len() as u32, &blocks_for(1, per_seq))?;
        let a_slot = slot_of(&a, &logprob_row);
        let (mut dout, mut cout, rung) = if fused {
            if !forward.stage_mixed(saber_engine::MixedChunk {
                seq: bseq,
                tokens: b_prompt.to_vec(),
                done_before: 0,
                last: true,
                sampling: logprob_row.clone(),
            }) {
                bail!("the engine refused to stage a fused chunk");
            }
            let d = forward.decode(std::slice::from_ref(&a_slot), 1, width)?;
            let c = forward.prefill(bseq, b_prompt, 0, true, &logprob_row)?;
            match forward.take_mixed_info() {
                Some((true, rung)) => (d, c, rung),
                other => bail!("the mix step did not FUSE ({other:?}): nothing was compared"),
            }
        } else {
            // The serial form's two calls, each proposal collected before the next call (a
            // stash outlives only the host sampler).
            let mut c = forward.prefill(bseq, b_prompt, 0, true, &logprob_row)?;
            forward.propose_greedy(&mut c)?;
            let d = forward.decode(std::slice::from_ref(&a_slot), 1, width)?;
            (d, c, 0)
        };
        // The decode half's proposal first, then the chunk's (the fused step's queue order; the
        // serial form's chunk proposal is already collected).
        forward.propose_greedy(&mut dout)?;
        forward.propose_greedy(&mut cout)?;
        let ca = dout
            .commits
            .into_iter()
            .next()
            .context("the mix step's decode half")?;
        let cb = cout
            .commits
            .into_iter()
            .next()
            .context("the mix step's chunk")?;
        rec.push((
            "A mix".to_owned(),
            ca.tokens[0],
            format!("{:?}", ca.rows.first()),
        ));
        rec.push((
            "B chunk".to_owned(),
            cb.tokens[0],
            format!("{:?}", cb.rows.first()),
        ));
        advance(&mut a, &ca);
        let mut b = Driven {
            seq: bseq,
            ids: b_prompt.to_vec(),
            committed_before: 1,
            accepted: 0,
            context: b_prompt.len() as u32 + 1,
            first_draft: Vec::new(),
        };
        b.ids.extend(&cb.tokens);
        for step in 0..4 {
            let slots = vec![slot_of(&a, &logprob_row), slot_of(&b, &logprob_row)];
            let mut out = forward.decode(&slots, 2, width)?;
            forward.propose_greedy(&mut out)?;
            for (c, (name, d)) in out.commits.iter().zip([("A", &mut a), ("B", &mut b)]) {
                rec.push((
                    format!("{name} after {step}"),
                    c.tokens[0],
                    format!("{:?}", c.rows.first()),
                ));
                advance(d, c);
            }
        }
        forward.release(a.seq);
        forward.release(bseq);
        Ok((rec, rung))
    };
    let (serial, _) = run(false)?;
    let (fused, rung) = run(true)?;
    let same = |label: &str| {
        let s = serial.iter().find(|r| r.0 == label);
        let f = fused.iter().find(|r| r.0 == label);
        s.is_some() && s == f
    };
    let before = same("A step 0") && same("A step 1");
    let decode_row = same("A mix");
    let chunk_row = same("B chunk");
    let after = serial
        .iter()
        .zip(&fused)
        .filter(|(s, _)| s.0.contains("after"))
        .all(|(s, f)| s == f);
    let first_after = serial
        .iter()
        .zip(&fused)
        .find(|(s, f)| s.0.contains("after") && s != f)
        .map(|(s, _)| s.0.clone());
    println!(
        "mixfused  p{rung} (150-row chunk beside b1s{width}), plans {}: before the mix {} | A's \
         decode row {} | B's chunk row {} | the 4 steps after {}{}",
        if proxy { "CLASS PROXY" } else { "default" },
        if before { "identical" } else { "DIFFER" },
        if decode_row { "0 bytes" } else { "DIFFERS" },
        if chunk_row { "0 bytes" } else { "DIFFERS" },
        if after { "identical" } else { "differ" },
        first_after.map_or(String::new(), |l| format!(" (first: {l})")),
    );
    // The verdict each engine's plans predict: the chunk row always 0 bytes; the decode row 0
    // bytes (and the steps after with it) exactly when the decode bucket and the fused rung reduce
    // in one class (H46: computed from the plans — the proxy and `--ar-exact` keep it, the default
    // plans move it — never a constant), DIFFERENT otherwise: the control that the comparison can
    // see the class.
    let (differs, why) = predict_mixed_decode_row(forward.facts(), width, rung);
    println!(
        "          predicted decode row: {} ({why})",
        if differs { "DIFFERS" } else { "0 bytes" }
    );
    let ok = before
        && chunk_row
        && if differs {
            !decode_row
        } else {
            decode_row && after
        };
    if ok {
        println!(
            "OK        mixfused: {}",
            if differs {
                "the control moved the decode row (the reduce's class moved) and kept the chunk row"
            } else if proxy {
                "the fused step's decode row is the class proxy's to the byte and its chunk row is \
                 the chunk alone's"
            } else {
                "the decode row kept its bytes (one reduce class on both sides) and so did the \
                 chunk row"
            }
        );
        Ok(0)
    } else {
        println!("FAIL      mixfused: the verdict is not the one these plans predict");
        Ok(1)
    }
}

/// **C10's per-row class gate for a PACKED step** (`mixfused Kn`): [`mixed_fused_arm`] with
/// `chunks` prompts (150, 90 and 60 rows, cycled from the run's prompts) riding beside A's decode
/// row in ONE fused forward, against the serial form (each chunk alone, in order, then A's step).
/// Every chunk's first token row must be 0 bytes from its chunk alone's (the packed step moves no
/// chunk row: its rows keep the chunk's class, its attention / GDN / store run over its own rows
/// and slot); A's row is the class proxy's under `--mixed-class-proxy` (0 bytes) and differs under
/// the default plans (the control); the four steps after, over A and every chunk's sequence
/// together, read back what each chunk wrote into its own pages and state (identical under the
/// proxy).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn mixed_packed_arm(
    forward: &mut HipForward,
    prompts: &[Vec<TokenId>],
    per_seq: u32,
    prefill_chunk: u32,
    width: u32,
    proxy: bool,
    chunks: usize,
    batches: &[u32],
) -> anyhow::Result<usize> {
    use saber_engine::ForwardRunner;
    println!();
    let lens = [150usize, 90, 60];
    if prompts.is_empty() || prompts.iter().any(Vec::is_empty) || !(2..=3).contains(&chunks) {
        println!("FAIL      mixfused K{chunks}: needs non-empty prompts and 2 or 3 chunks");
        return Ok(1);
    }
    let pa: Vec<TokenId> = prompts[0].iter().copied().cycle().take(40).collect();
    let pbs: Vec<Vec<TokenId>> = (0..chunks)
        .map(|i| {
            prompts[(i + 1) % prompts.len()]
                .iter()
                .copied()
                .cycle()
                .skip(i * 7)
                .take(lens[i])
                .collect()
        })
        .collect();
    let k = (width.max(1) - 1) as usize;
    let logprob_row = RowRequest {
        temperature: Some(1.0),
        penalty_ids: Vec::new(),
    };
    type Rec = Vec<(String, TokenId, String)>;
    let slot_of = |d: &Driven, req: &RowRequest| DecodeSlot {
        seq: d.seq,
        input: *d.ids.last().expect("a prefilled sequence has a token"),
        context_len: d.context - 1,
        draft: vec![RecordedDrafter::WRONG; k],
        committed_before: d.committed_before,
        sampling: req.clone(),
        decider: None,
    };
    let advance = |d: &mut Driven, c: &saber_engine::SlotCommit| {
        d.committed_before = c.tokens.len() as u32;
        d.context += c.tokens.len() as u32;
        d.ids.extend(&c.tokens);
    };
    let mut run = |fused: bool| -> anyhow::Result<(Rec, u32)> {
        let base = if fused { 890_100 } else { 890_000 };
        let mut rec: Rec = Vec::new();
        let mut a = admit_and_prefill(
            forward,
            SeqId(base),
            &pa,
            &blocks_for(0, per_seq),
            prefill_chunk,
        )?;
        for step in 0..2 {
            let mut out = forward.decode(&[slot_of(&a, &logprob_row)], 1, width)?;
            forward.propose_greedy(&mut out)?;
            let c = out
                .commits
                .into_iter()
                .next()
                .context("a decode step committed nothing")?;
            rec.push((
                format!("A step {step}"),
                c.tokens[0],
                format!("{:?}", c.rows.first()),
            ));
            advance(&mut a, &c);
        }
        let seqs: Vec<SeqId> = (0..chunks).map(|i| SeqId(base + 1 + i as u64)).collect();
        for (i, (s, p)) in seqs.iter().zip(&pbs).enumerate() {
            forward.admit(*s, p.len() as u32, &blocks_for(i + 1, per_seq))?;
        }
        let a_slot = slot_of(&a, &logprob_row);
        let mixed: Vec<saber_engine::MixedChunk> = seqs
            .iter()
            .zip(&pbs)
            .map(|(s, p)| saber_engine::MixedChunk {
                seq: *s,
                tokens: p.clone(),
                done_before: 0,
                last: true,
                sampling: logprob_row.clone(),
            })
            .collect();
        let (mut dout, mut couts, rung) = if fused {
            if !forward.can_fuse(&mixed, 1, width) {
                bail!("the fused law refused a {chunks}-chunk packed step");
            }
            for m in mixed {
                if !forward.stage_mixed(m) {
                    bail!("the engine refused to stage a packed chunk");
                }
            }
            let d = forward.decode(std::slice::from_ref(&a_slot), 1, width)?;
            let mut cs = Vec::new();
            for (s, p) in seqs.iter().zip(&pbs) {
                cs.push(forward.prefill(*s, p, 0, true, &logprob_row)?);
            }
            match forward.take_mixed_info() {
                Some((true, rung)) => (d, cs, rung),
                other => bail!("the packed step did not FUSE ({other:?}): nothing was compared"),
            }
        } else {
            let mut cs = Vec::new();
            for (s, p) in seqs.iter().zip(&pbs) {
                let mut c = forward.prefill(*s, p, 0, true, &logprob_row)?;
                forward.propose_greedy(&mut c)?;
                cs.push(c);
            }
            let d = forward.decode(std::slice::from_ref(&a_slot), 1, width)?;
            (d, cs, 0)
        };
        // The decode half's proposal first, then each chunk's (the fused queue order).
        forward.propose_greedy(&mut dout)?;
        for c in &mut couts {
            forward.propose_greedy(c)?;
        }
        let ca = dout
            .commits
            .into_iter()
            .next()
            .context("the mix step's decode half")?;
        rec.push((
            "A mix".to_owned(),
            ca.tokens[0],
            format!("{:?}", ca.rows.first()),
        ));
        advance(&mut a, &ca);
        let mut bs = Vec::new();
        for (i, ((s, p), c)) in seqs.iter().zip(&pbs).zip(couts).enumerate() {
            let cb = c
                .commits
                .into_iter()
                .next()
                .context("a packed chunk committed nothing")?;
            rec.push((
                format!("B{i} chunk"),
                cb.tokens[0],
                format!("{:?}", cb.rows.first()),
            ));
            let mut b = Driven {
                seq: *s,
                ids: p.clone(),
                committed_before: 1,
                accepted: 0,
                context: p.len() as u32 + 1,
                first_draft: Vec::new(),
            };
            b.ids.extend(&cb.tokens);
            bs.push(b);
        }
        for step in 0..4 {
            let mut slots = vec![slot_of(&a, &logprob_row)];
            slots.extend(bs.iter().map(|b| slot_of(b, &logprob_row)));
            // C11: A + K chunks' sequences is 3 at K2. H46: the batch is the SCHEDULER'S pick over
            // this run's planned batches (`decode_law::pick_batch`, `select`'s batch axis) — b3 on
            // the ladder, b4 at `1,2,4,8` — never a power of two this file assumes.
            let live = slots.len() as u32;
            let batch = saber_sched::decode_law::pick_batch(batches, live).ok_or_else(|| {
                anyhow::anyhow!("no planned decode batch holds {live} sequences ({batches:?})")
            })?;
            let mut out = forward.decode(&slots, batch, width)?;
            forward.propose_greedy(&mut out)?;
            let commits = out.commits;
            advance(&mut a, &commits[0]);
            rec.push((
                format!("A after {step}"),
                commits[0].tokens[0],
                format!("{:?}", commits[0].rows.first()),
            ));
            for (i, (c, b)) in commits[1..].iter().zip(bs.iter_mut()).enumerate() {
                rec.push((
                    format!("B{i} after {step}"),
                    c.tokens[0],
                    format!("{:?}", c.rows.first()),
                ));
                advance(b, c);
            }
        }
        forward.release(a.seq);
        for s in seqs {
            forward.release(s);
        }
        Ok((rec, rung))
    };
    let (serial, _) = run(false)?;
    let (fused, rung) = run(true)?;
    let same = |label: &str| {
        let s = serial.iter().find(|r| r.0 == label);
        let f = fused.iter().find(|r| r.0 == label);
        s.is_some() && s == f
    };
    let before = same("A step 0") && same("A step 1");
    let decode_row = same("A mix");
    let chunk_rows: Vec<bool> = (0..chunks).map(|i| same(&format!("B{i} chunk"))).collect();
    let after = serial
        .iter()
        .zip(&fused)
        .filter(|(s, _)| s.0.contains("after"))
        .all(|(s, f)| s == f);
    let first_after = serial
        .iter()
        .zip(&fused)
        .find(|(s, f)| s.0.contains("after") && s != f)
        .map(|(s, _)| s.0.clone());
    println!(
        "mixfused  K{chunks} p{rung} ({:?}-row chunks beside b1s{width}), plans {}: before the mix \
         {} | A's decode row {} | chunk rows {} | the 4 steps after {}{}",
        &lens[..chunks],
        if proxy { "CLASS PROXY" } else { "default" },
        if before { "identical" } else { "DIFFER" },
        if decode_row { "0 bytes" } else { "DIFFERS" },
        chunk_rows
            .iter()
            .map(|b| if *b { "0 bytes" } else { "DIFFERS" })
            .collect::<Vec<_>>()
            .join(" / "),
        if after { "identical" } else { "differ" },
        first_after.map_or(String::new(), |l| format!(" (first: {l})")),
    );
    // H46: the decode row's prediction from the plans (see `predict_mixed_decode_row`).
    let (differs, why) = predict_mixed_decode_row(forward.facts(), width, rung);
    println!(
        "          predicted decode row: {} ({why})",
        if differs { "DIFFERS" } else { "0 bytes" }
    );
    let ok = before
        && chunk_rows.iter().all(|b| *b)
        && if differs {
            !decode_row
        } else {
            decode_row && after
        };
    if ok {
        println!(
            "OK        mixfused K{chunks}: {}",
            if differs {
                "the control moved the decode row and kept every chunk row"
            } else if proxy {
                "every chunk row is its chunk alone's and the decode row the class proxy's, to \
                 the byte; the steps after read back the same pages and state"
            } else {
                "the decode row kept its bytes (one reduce class on both sides) and so did every \
                 chunk row"
            }
        );
        Ok(0)
    } else {
        println!("FAIL      mixfused K{chunks}: the verdict is not the one these plans predict");
        Ok(1)
    }
}

/// Park the first prompt's session mid-generation, restore it into **other** pages, and check that
/// it continues the ids the unparked run produced — **at the run's own verify width, with the
/// run's own drafter**.
///
/// The CLI's own version of `park_seam.rs`'s property, on the card: the host gate proves it against
/// a stub, and this proves the same thing against the real KV pages, the real state slot and (since
/// ws-s5) the real DRAFTER pages. It also prints what the two transfers cost next to the step they
/// displace, which is the number the eviction policy is set from.
///
/// **Two things changed in ws-s5 and both were the arm's shape rather than the park's.**
///
/// 1. It stepped at width 1 (`decode(&[…], 1, 1)`) whatever the run planned. A run plans exactly
///    one verify width — the GDN conv cache's layout is a function of it — so on a width-8 run the
///    arm asked for a bucket that was never planned and the engine refused by name: *"no plan for
///    decode at 1 row(s): batch 1 at verify width 1 was never planned"* (S4's open item 5). It
///    takes the run's width now, and drives it with the run's own drafter, so "a restored session
///    continues" is asked of the session speculation actually produces.
/// 2. A parked image carried the target's KV and GDN state and NOT the drafter's pages. Those are
///    session state exactly as the others are (`DraftTier::admit`/`release` hand them out and take
///    them back), and a restored session proposing out of pages another request wrote changes no
///    TOKEN — [`saber_engine::EngineDrafter`]'s blocks are only ever guesses and `Greedy` commits
///    the model's own answer — and changes the ACCEPTANCE, which is the one thing an id list
///    cannot see. The image carries them now, and the engine refuses by name when it cannot
///    (`DraftPark`, `crates/saber-engine/src/device.rs`).
///
/// So the verdict has two halves: the ids across the restore, on the COMMON PREFIX with both
/// lengths beside it (S4's item 3 — a fixed number of STEPS is a variable number of TOKENS under
/// speculation), and the per-step DRAFT TRACE against the same steps of the unparked run, which is
/// the half that can see whether the drafter's pages came back.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
/// Polls one asynchronous park or restore may take before it is declared stuck.
///
/// A poll hashes at most 8 MiB and a whole image is ~270 MiB at width 8, so a park is ~35 polls
/// plus whatever the copy takes; the bound is far above that because what it is for is a path
/// that has STOPPED, and a gate that hangs is a card nobody gets back.
const MAX_PARK_POLLS: usize = 200_000;

/// The serve plane's own park: issue, then poll until the blob is there.
///
/// This is `ParkBegin`/`ParkPoll` — the path `saber serve` takes when the scheduler evicts a
/// session — as opposed to `Req::Park`, which does the whole thing in one call. S5 wired it and
/// nothing had ever driven it (ws-t2).
fn park_async(
    forward: &mut saber_engine::HipForward,
    seq: saber_sched::SeqId,
) -> Result<Vec<u8>, saber_engine::ForwardError> {
    use saber_engine::ForwardRunner;
    forward.park_begin(seq)?;
    for _ in 0..MAX_PARK_POLLS {
        if let Some(blob) = forward.park_poll(seq)? {
            return Ok(blob);
        }
    }
    Err(saber_engine::ForwardError::Device(format!(
        "{seq}: an asynchronous park did not finish in {MAX_PARK_POLLS} polls"
    )))
}

/// The same for the restore half: `RestoreBegin` and then `RestorePoll` to done.
fn restore_async(
    forward: &mut saber_engine::HipForward,
    seq: saber_sched::SeqId,
    blob: Vec<u8>,
) -> Result<(), saber_engine::ForwardError> {
    use saber_engine::ForwardRunner;
    forward.restore_begin(seq, blob)?;
    for _ in 0..MAX_PARK_POLLS {
        if forward.restore_poll(seq)? {
            return Ok(());
        }
    }
    Err(saber_engine::ForwardError::Device(format!(
        "{seq}: an asynchronous restore did not finish in {MAX_PARK_POLLS} polls"
    )))
}

/// THE DISCARD HOOK ON THE CARD (ws-k18): park a session, release it the way the scheduler
/// releases a parked sequence (`release_parked`, which KEEPS the other rank's half for a restore),
/// then drop the image unrestored — the abort-while-parked path — and read what each rank held.
///
/// At TP=2 the follower must answer that it HELD a half under that name and holds NONE after, and
/// the leader's receipt must be gone too. The negative control runs first and is the same call for
/// a name nothing was parked under: it must find nothing and leave the follower's one half in
/// place — a hook that dropped every half, or none, fails one of the two. At TP=1 an image has no
/// other half and the arm says so rather than passing on zeros.
fn park_discard_arm(
    forward: &mut saber_engine::HipForward,
    prompt: &[TokenId],
    blocks_per_seq: u32,
    prefill_chunk: u32,
    width: u32,
    engine_drafts: bool,
    fail: &impl Fn(&str),
) -> usize {
    use saber_engine::ForwardRunner;
    use saber_sched::SeqId;

    let tp = forward.facts().tp;
    let seq = SeqId(951);
    let pages: Vec<u32> = (0..blocks_per_seq).collect();
    let mut driven = match admit_and_prefill(forward, seq, prompt, &pages, prefill_chunk) {
        Ok(d) => vec![d],
        Err(e) => {
            fail(&format!("park discard: admit/prefill: {e}"));
            return 1;
        }
    };
    // Two decode steps at the run's own width, so the image is of a session that has COMMITTED
    // (an image with `committed 0` is refused by name) -- the drafter as the park arm builds it.
    let mut drafter: Box<dyn Drafter> = if width <= 1 {
        Box::new(NoDrafter)
    } else if engine_drafts {
        Box::new(EngineDrafter::new(width - 1))
    } else {
        Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, width - 1))
    };
    if let Err(e) = decode_traced(
        forward,
        &mut driven,
        1,
        width,
        drafter.as_mut(),
        2,
        None,
        None,
        None,
    ) {
        fail(&format!("park discard: decode: {e}"));
        forward.release(seq);
        return 1;
    }
    let parked = if forward.parks_async() {
        park_async(forward, seq)
    } else {
        forward.park_encoded(seq)
    };
    let blob = match parked {
        Ok(b) => b,
        Err(e) => {
            fail(&format!("park discard: park: {e}"));
            forward.release(seq);
            return 1;
        }
    };
    // The scheduler's release of a parked sequence: pages and slot back, the image's halves kept.
    forward.release_parked(seq);
    let control = forward.discard_parked(SeqId(seq.0 + 1000));
    let case = forward.discard_parked(seq);
    drop(blob);
    println!(
        "park      discard at TP={tp}: control (a name never parked) {control:?}; case (the parked name, after release_parked) {case:?}"
    );
    if tp < 2 {
        if control.held() || case.held() {
            fail("park discard: a ONE-rank engine reported a held half; it has no other rank");
            return 1;
        }
        println!("park      discard SKIP at TP=1: an image has no half on another rank");
        return 0;
    }
    let mut failures = 0;
    if control.held() || control.peer_remaining != 1 {
        fail(&format!(
            "park discard: the CONTROL found something under a name nothing was parked under, or dropped the real half: {control:?} (want nothing held and the follower still holding 1)"
        ));
        failures += 1;
    }
    if case.receipts != 1
        || !case.peer_held
        || case.peer_remaining != 0
        || case.receipts_remaining != 0
    {
        fail(&format!(
            "park discard: dropping the parked image left something held: {case:?} (want 1 receipt dropped, the follower held it, 0 halves and 0 receipts left)"
        ));
        failures += 1;
    }
    if failures == 0 {
        println!(
            "OK        an image dropped while parked is dropped on BOTH ranks: the follower held its half and holds 0 after, the leader's receipt is gone; the control left the half alone"
        );
    }
    failures
}

/// The first byte at which two park images differ, named by the REGION it lands in (ws-k15).
///
/// **The instrument for a two-rank park.** A restore that put every region back where it was
/// makes the NEXT park of the restored session byte-identical to the first: the same regions in
/// the same order, the same `resident` and `committed`, and the same payload — the pages and
/// slots moved but their contents did not. So "park, restore into other pages, park again" is a
/// comparison of the device's own bytes with no second read-back path, and the first differing
/// byte names which region the restore missed. `None` means identical.
///
/// A `draft_kv` region is named further: the drafter's regions are one per (cache, layer) in the
/// order `k` layers then `v` layers, so the region's ordinal inside that run says which cache and
/// which layer — and the offset inside it, against the engine's own per-page dump, says which
/// page.
fn image_diff(before: &[u8], after: &[u8]) -> Option<String> {
    use saber_mem::park::RegionKind;

    let (a, b) = match (
        saber_engine::ParkImage::decode(before),
        saber_engine::ParkImage::decode(after),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            return Some(format!(
                "one of the two images does not decode: before {:?}, after {:?}",
                a.err().map(|e| e.to_string()),
                b.err().map(|e| e.to_string())
            ))
        }
    };
    if a.resident != b.resident || a.committed != b.committed {
        return Some(format!(
            "the re-parked image says resident {} / committed {}; the parked one said {} / {}",
            b.resident, b.committed, a.resident, a.committed
        ));
    }
    if a.digests.layout != b.digests.layout {
        return Some(format!(
            "the two images have different shapes: {} region(s) against {}",
            a.digests.layout.len(),
            b.digests.layout.len()
        ));
    }
    let at = a.bytes.iter().zip(&b.bytes).position(|(x, y)| x != y)?;
    let mut base = 0usize;
    let mut drafts = 0usize;
    let layers = a
        .digests
        .layout
        .iter()
        .filter(|r| r.kind == RegionKind::DraftKv)
        .count()
        / 2;
    for (i, r) in a.digests.layout.iter().enumerate() {
        let end = base + r.len as usize;
        if at < end {
            let inside = at - base;
            let where_ = if r.kind == RegionKind::DraftKv && layers > 0 {
                let cache = if drafts < layers { "k" } else { "v" };
                format!(
                    " = the drafter's {cache} cache, layer {}, byte {inside} of its {}-byte run",
                    drafts % layers,
                    r.len
                )
            } else {
                format!(" = byte {inside} of a {}-byte region", r.len)
            };
            return Some(format!(
                "the re-parked image differs from the parked one at byte {at} of {}:                  region {i} of {}, kind {}{where_}; {:#04x} against {:#04x}",
                a.bytes.len(),
                a.digests.layout.len(),
                r.kind,
                b.bytes[at],
                a.bytes[at],
            ));
        }
        if r.kind == RegionKind::DraftKv {
            drafts += 1;
        }
        base = end;
    }
    Some(format!(
        "the images differ at byte {at}, which is past every region"
    ))
}

#[allow(clippy::too_many_arguments)]
fn park_round_trip(
    forward: &mut saber_engine::HipForward,
    prompt: &[TokenId],
    blocks_per_seq: u32,
    prefill_chunk: u32,
    width: u32,
    engine_drafts: bool,
    control: &[TokenId],
    control_trace: &[DraftStep],
    async_path: bool,
    law: ParkImageLaw,
    probe: bool,
    parks: usize,
    fail: &impl Fn(&str),
) -> usize {
    use std::time::Instant;

    use saber_engine::ForwardRunner;
    use saber_sched::SeqId;

    let mut failures = 0usize;
    let arm = match (async_path, parks) {
        (false, 1) => "sync",
        (true, 1) => "async",
        (false, _) => "sync x2",
        (true, _) => "async x2",
    };
    let facts = forward.facts().clone();
    if async_path && !forward.parks_async() {
        println!("park      SKIP: this engine has no asynchronous park path");
        return 0;
    }
    // STEPS, not tokens. Under speculation a step commits between 1 and `width` tokens, so a
    // bound taken off the control's id count would run a different number of steps than the run
    // it is compared with. Eight steps either way, capped by what the control actually ran.
    let steps = control_trace.len().clamp(1, 8).min(control.len().max(1));
    let from: Vec<u32> = (0..blocks_per_seq).collect();
    let into: Vec<u32> = (blocks_per_seq..2 * blocks_per_seq).collect();
    if 2 * blocks_per_seq > facts.kv_pool_blocks {
        println!(
            "park      SKIP: the pool is {} blocks and a park+restore needs two sets of {}",
            facts.kv_pool_blocks, blocks_per_seq
        );
        return 0;
    }

    // The run's own drafter, so that the steps either side of the restore are the steps the run
    // makes. `EngineDrafter` is a mailbox keyed on the sequence: the restored session is a NEW
    // sequence, so the block the pre-park step proposed is re-offered under its new name below —
    // which is what the drafter's state crossing a restore MEANS.
    let mut drafter: Box<dyn Drafter> = if width <= 1 {
        Box::new(NoDrafter)
    } else if engine_drafts {
        Box::new(EngineDrafter::new(width - 1))
    } else {
        Box::new(FixedDrafter::new(FIXED_DRAFT_TOKEN, width - 1))
    };

    let mut seq = SeqId(901);
    let mut driven = match admit_and_prefill(forward, seq, prompt, &from, prefill_chunk) {
        Ok(d) => vec![d],
        Err(e) => {
            fail(&format!("park: admit/prefill: {e}"));
            return 1;
        }
    };

    let mut trace: Vec<DraftStep> = Vec::new();
    // One entry per park, in order: an arm that parks the same session TWICE (ws-k17) reports both
    // round trips rather than the last one.
    let mut park_ms: Vec<f64> = Vec::new();
    let mut restore_ms: Vec<f64> = Vec::new();
    let mut blob_bytes = 0usize;
    let mut image_line = String::new();
    // Tokens the step before the park committed, from the image itself. **The whole arm turns on
    // it** (ws-k14): the conv cache lives in the run's base and the recurrent snapshot in
    // `base + committed - 1`, so at `committed == 1` the two coincide and a restore onto different
    // slots is sound under the OLD image too. An arm that parked there could not fail with
    // `--park-image-base` and therefore proves nothing about the law.
    let mut committed_at_park = 0u32;
    // WHERE THE PARKS LAND. One park at the middle of the run, as the arm has always done; two
    // (ws-k17, `ParkPolicy::max_parks_per_session` is unbounded now) at a third and two thirds,
    // with at least one decode step between them so the second image is of a session that has
    // MOVED since it was restored -- a re-park before any step is the probe's question, not this.
    let park_points: Vec<usize> = if parks <= 1 {
        vec![(steps / 2).max(1)]
    } else {
        if steps < 3 {
            println!(
                "park      SKIP ({arm}): {steps} step(s) cannot hold two parks with a step between"
            );
            return 0;
        }
        (1..=parks)
            .map(|k| (k * steps / (parks + 1)).max(1))
            .collect()
    };
    // The pages each restore lands in, alternating: the first restore goes into the second set,
    // the second back into the first (freed when the first parked name was released) -- still
    // DIFFERENT pages from the ones the session was parked out of, every time.
    let mut pages_now_second = false;
    for i in 0..steps {
        if park_points.contains(&i) {
            let t = Instant::now();
            let blob = match if async_path {
                park_async(forward, seq)
            } else {
                forward.park_encoded(seq)
            } {
                Ok(b) => b,
                Err(e) => {
                    fail(&format!("park ({arm}): {e}"));
                    forward.release(seq);
                    return failures + 1;
                }
            };
            park_ms.push(t.elapsed().as_secs_f64() * 1e3);
            blob_bytes = blob.len();
            // THE PROBE'S "BEFORE" (ws-k15). Kept whole rather than hashed: the comparison below
            // wants the first differing BYTE, and a digest can only say that there is one.
            let before = probe.then(|| blob.clone());
            // WHAT THE IMAGE CARRIES, from the image itself. A park that dropped the drafter's
            // half would look exactly like a park that had none, and this is the line that tells
            // them apart before any id is compared.
            match saber_engine::ParkImage::decode(&blob) {
                Ok(img) => {
                    let (kv, gdn, bytes) = img.shape();
                    let (dr, dbytes) = img.draft_shape();
                    let mib = |b: u64| b as f64 / (1 << 20) as f64;
                    // PER KIND, because the size question this round had to answer is what the GDN
                    // half costs: one slot per state buffer against the whole run of `max_width`
                    // (ws-k14). The image's own layout says it; nothing here derives it.
                    let of = |k: saber_mem::park::RegionKind| -> u64 {
                        img.digests
                            .layout
                            .iter()
                            .filter(|r| r.kind == k)
                            .map(|r| r.len)
                            .sum()
                    };
                    let gdn_bytes = of(saber_mem::park::RegionKind::GdnState);
                    committed_at_park = committed_at_park.max(img.committed);
                    image_line = format!(
                        "{} region(s): {kv} kv + {gdn} gdn + {dr} drafter, {:.1} MiB = {:.1} kv + \
                         {:.1} gdn + {:.1} drafter; law {}, committed {} at the park",
                        img.digests.layout.len(),
                        mib(bytes),
                        mib(of(saber_mem::park::RegionKind::Kv)),
                        mib(gdn_bytes),
                        mib(dbytes),
                        law.label(),
                        img.committed,
                    );
                    if engine_drafts && dr == 0 {
                        fail(
                            "park: this run has a RESIDENT drafter and its image carries no \
                             drafter region: the restored session would propose out of whatever \
                             its new pages held",
                        );
                        failures += 1;
                    }
                }
                Err(e) => {
                    fail(&format!("park: the image does not decode: {e}"));
                    failures += 1;
                }
            }
            // The block the parked step proposed belongs to the session, not to its name.
            let pending = drafter.propose(seq, *driven[0].ids.last().expect("a token"), width - 1);
            // RELEASED AFTER THE RESTORE, not before it (ws-t2). At TP=2 an image has a half on
            // the other rank, and the only signal this engine has that an image is dead is the
            // sequence being released — so a release here would tell the follower to drop the
            // half this arm is about to restore. Both sessions are alive for the length of the
            // restore, which is what the pool check above (two sets of pages) already sized for.
            let parked = seq;
            seq = SeqId(seq.0 + 1);
            driven[0].seq = seq;
            pages_now_second = !pages_now_second;
            let target = if pages_now_second { &into } else { &from };
            if !pending.is_empty() {
                drafter.offer(seq, &pending);
            }
            if let Err(e) = forward.admit(seq, prompt.len() as u32, target) {
                fail(&format!("park: re-admit: {e}"));
                forward.release(parked);
                return failures + 1;
            }
            let t = Instant::now();
            let restored = if async_path {
                restore_async(forward, seq, blob)
            } else {
                forward.restore_encoded(seq, blob)
            };
            if let Err(e) = restored {
                fail(&format!("restore ({arm}): {e}"));
                forward.release(seq);
                forward.release(parked);
                return failures + 1;
            }
            restore_ms.push(t.elapsed().as_secs_f64() * 1e3);
            // The parked session's name is done with now that its image has been published into
            // another one: at TP=2 this is also what tells the follower to drop any half of it
            // that was NOT restored.
            forward.release(parked);
            // AND THE PROBE'S "AFTER": park the restored session again, before it takes a step,
            // and hold the two images against each other. Every rank dumps its own drafter half
            // per cache, per layer, per page as it parks, so this line is the leader's verdict and
            // the two dumps in the log are where a two-rank difference is localised (ws-k15).
            if let Some(before) = before {
                match forward.park_encoded(seq) {
                    Ok(after) => match image_diff(&before, &after) {
                        None => println!(
                            "park      PROBE: re-parking the restored session gives a \
                             BYTE-IDENTICAL image ({} bytes): every region came back where it was",
                            after.len()
                        ),
                        Some(line) => {
                            fail(&format!("park probe ({arm}): {line}"));
                            failures += 1;
                        }
                    },
                    Err(e) => {
                        fail(&format!("park probe ({arm}): the re-park failed: {e}"));
                        failures += 1;
                    }
                }
            }
        }
        // THE RUN'S WIDTH. A run plans one verify width and this arm used to ask for 1.
        if let Err(e) = decode_traced(
            forward,
            &mut driven,
            1,
            width,
            drafter.as_mut(),
            1,
            Some(&mut trace),
            None,
            None,
        ) {
            fail(&format!("park: decode after restore: {e}"));
            forward.release(seq);
            return failures + 1;
        }
    }
    let ids = std::mem::take(&mut driven[0].ids);
    forward.release(seq);

    println!();
    // The region count is the IMAGE's own, not `max_blocks + 1`. That guess was one region per
    // page plus the state slot, and it stopped being the shape of an image the moment a session
    // had more than one paged buffer -- let alone a drafter's ten (ws-s5: it printed 257 for an
    // image of 2,458).
    let ms = |v: &[f64]| {
        v.iter()
            .map(|m| format!("{m:.2}"))
            .collect::<Vec<_>>()
            .join(" / ")
    };
    println!(
        "park      {arm} at TP={}: image {:.1} MiB [{image_line}]; park {} ms, restore {} ms \
         ({} park(s) at step(s) {park_points:?}), into DIFFERENT pages AND DIFFERENT SLOTS, \
         {steps} step(s) at verify width {width}",
        facts.tp,
        blob_bytes as f64 / (1 << 20) as f64,
        ms(&park_ms),
        ms(&restore_ms),
        park_ms.len(),
    );
    // **WHETHER THIS ARM COULD HAVE SEEN THE SLOT LAW AT ALL.** At one committed token the base
    // and the live snapshot are the same slot, so the arm is green under either image and says so
    // rather than being quoted as evidence for the law (ws-k14).
    if committed_at_park <= 1 {
        println!(
            "park      NOTE: the step before the park committed {committed_at_park} token(s), so \
             the run's base IS its live recurrent snapshot and this arm cannot tell the image \
             laws apart. Run it at a width and an acceptance that commit more than one."
        );
    }
    // THE DRAFTER, ACROSS THE RESTORE. The same steps of the unparked run, proposal for proposal:
    // this is the half of the claim the ids cannot make, because a drafter reading the wrong pages
    // proposes a worse block and the model still commits its own token.
    failures += draft_trace_verdict(
        &control_trace[..trace.len().min(control_trace.len())],
        &trace,
        "unparked",
        "restored",
        fail,
    );
    let v = PrefixVerdict::of(&ids, control);
    match (
        v.prefix_agrees(),
        v.lengths_agree() || ids.len() <= control.len(),
    ) {
        (true, true) => println!(
            "OK        {} id(s) identical across a {arm} park and a restore into other pages ({} \
             step(s) at width {width}, TP={})",
            v.common, steps, facts.tp
        ),
        (true, false) => {
            fail(&format!(
                "a parked session committed MORE tokens than the unparked run it is compared \
                 with: {}; the two agree on every id they share",
                v.line()
            ));
            failures += 1;
        }
        (false, _) => {
            let at = v.first_diff.unwrap_or(0);
            fail(&format!(
                "a parked session diverged at token {at}: {:?} vs the unparked {:?}; {}",
                ids.get(at),
                control.get(at),
                v.line()
            ));
            failures += 1;
        }
    }
    failures
}

/// What one concurrency level measured.
struct ConcRow {
    conc: usize,
    /// What each stream received, in submission order.
    ids: Vec<Vec<TokenId>>,
    aggregate_tps: f64,
    per_stream_tps: f64,
    ttft_p50_ms: f64,
    ttft_max_ms: f64,
    steps: usize,
    max_batch: u32,
    /// Every DECODE shape the engine has replayed so far, `(batch, width)` ascending —
    /// cumulative, because `EngineCounters::launches_by_shape` is keyed per shape and not
    /// counted per step. It is the dynamic verify width as a CENSUS rather than as a claim: a
    /// run whose widths are all the ceiling narrowed nothing, whatever the EMAs did.
    shapes: Vec<(u32, u32)>,
    /// Decode steps in THIS run whose bucket was narrower than the widest captured width.
    width_narrowed_steps: u64,
    /// Slot-steps in this run whose own acceptance EMA asked for a narrower width than the bucket
    /// they ran in (`SchedCounters::slot_caps_absorbed`) — the rectangular-bucket tax.
    slot_caps_absorbed: u64,
}

/// One request through the scheduler, start to finish, and the ids it produced.
///
/// Sequential on purpose: `run_concurrent` measures streams that overlap, and this measures a
/// request that has the engine to itself — which is what makes the difference between two calls of
/// it a fact about what the engine REMEMBERED rather than about what it was sharing.
fn run_once(
    sched: &Arc<Scheduler<RealExecutor<HipForward>>>,
    prompt: &[TokenId],
    max_tokens: usize,
) -> anyhow::Result<Vec<TokenId>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("a runtime for the request")?;
    let steps = spawn_step_loop(Arc::clone(sched));
    let prompt = prompt.to_vec();
    let sched2 = Arc::clone(sched);
    let out = runtime.block_on(async move {
        use futures_util::StreamExt;
        let sub = sched2
            .submit(EngineRequest {
                prompt_token_ids: prompt,
                params: SamplingParams {
                    max_tokens: Some(max_tokens),
                    // An ID gate: temperature 0, for the reason above.
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
            })
            .map_err(|e| format!("submit: {e}"))?;
        let mut events = sub.events;
        let mut ids: Vec<TokenId> = Vec::new();
        while let Some(ev) = events.next().await {
            ids.extend(ev.new_tokens);
        }
        Ok::<_, String>(ids)
    });
    let _ = steps.stop();
    out.map_err(|e| anyhow::anyhow!("a request failed: {e}"))
}

/// One stream of a harness concurrency arm: its ids, its TTFT (submit to first token) and its time
/// in flight (submit to its last event).
type StreamOut = (Vec<TokenId>, Duration, Duration);

/// What [`drive_streams`] hands back.
struct ArmStreams {
    /// Every stream's result, in STREAM order (the order of the requests), whatever order the
    /// consuming tasks happened to start or finish in.
    out: Vec<StreamOut>,
    /// Engine steps the arm's step thread ran.
    steps: u64,
    /// The late joiner's admission: the number of steps the arm had run when it was submitted,
    /// and whether that was the `join_after` it asked for (`false`: the others finished first).
    late: Option<(u64, bool)>,
}

/// A submitted stream and the instant its `submit` was called.
type Ready = (saber_serve::engine::Submission, Instant);

/// **FN29: THE ADMISSION LAW of the gate's scheduler-driven arms** — the schedule is a function of
/// the inputs only.
///
/// Before FN29 each stream's tokio task submitted its own request AFTER the step loop had started,
/// so which requests the arm's first steps admitted was a host race (`docs/LEDGER.md` "FN14c"): one
/// binary produced three different concurrency-4 proposal streams, and the shadow gate's signature
/// had to stop at the arms' start. Here:
///
/// * every request but the late joiner is SUBMITTED HERE, in stream order `0, 1, ..`, before the
///   arm's step thread exists — the barrier is that the thread is spawned only after the last
///   `submit` returned, so its first `step()` sees the whole queue, in that order;
/// * the late joiner (`join_after = Some(k)`, the last request) is submitted BY THE STEP THREAD,
///   between steps, once the arm has run exactly `k` steps — never by a task polling a counter.
///   If the others finish first (the scheduler goes idle before step `k`) it is submitted at that
///   idle point, which is again a function of the inputs, and the caller says so;
/// * the tasks that CONSUME the streams start in `task_order` (the identity in the gate; a
///   shuffle in the host test) and can no longer move anything: the event channels are unbounded,
///   so a consumer that starts late only reads later.
///
/// The step thread is `saber_sched::spawn_step_loop`'s loop (step, yield after a step, sleep
/// `IDLE_POLL` when idle) plus the late joiner's submit; the scheduler, the executor and the serve
/// path are untouched.
fn drive_streams<E: saber_sched::StepExecutor>(
    sched: &Arc<Scheduler<E>>,
    requests: Vec<EngineRequest>,
    join_after: Option<usize>,
    task_order: &[usize],
) -> anyhow::Result<ArmStreams> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let n = requests.len();
    anyhow::ensure!(n > 0, "a concurrency arm with no streams");
    let mut sorted: Vec<usize> = task_order.to_vec();
    sorted.sort_unstable();
    anyhow::ensure!(
        sorted == (0..n).collect::<Vec<_>>(),
        "task_order must be a permutation of 0..{n}"
    );
    let late_ix = join_after.map(|_| n - 1);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("a runtime for the concurrent streams")?;

    // 1. The fixed submission order, before any step can run.
    let mut ready: Vec<Option<Ready>> = Vec::with_capacity(n);
    let mut late_req: Option<EngineRequest> = None;
    for (i, req) in requests.into_iter().enumerate() {
        if Some(i) == late_ix {
            late_req = Some(req);
            ready.push(None);
            continue;
        }
        let submitted = Instant::now();
        let sub = sched
            .submit(req)
            .map_err(|e| anyhow::anyhow!("submit stream {i}: {e}"))?;
        ready.push(Some((sub, submitted)));
    }

    // 2. The arm's step thread, spawned only now: the barrier.
    let (late_tx, late_rx) = tokio::sync::oneshot::channel::<Result<Ready, String>>();
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let sched = Arc::clone(sched);
        let stop = Arc::clone(&stop);
        let after = join_after.unwrap_or(0) as u64;
        std::thread::Builder::new()
            .name("saber-gate-step".to_owned())
            .spawn(move || {
                let mut steps = 0u64;
                let mut late = late_req.map(|r| (r, late_tx));
                let mut late_at: Option<(u64, bool)> = None;
                while !stop.load(Ordering::Relaxed) {
                    if late.is_some() && steps >= after {
                        late_at = submit_late(&sched, &mut late, steps, true);
                    }
                    match sched.step() {
                        saber_sched::StepOutcome::Ran(_) => {
                            steps += 1;
                            std::thread::yield_now();
                        }
                        saber_sched::StepOutcome::Idle => {
                            if late.is_some() {
                                late_at = submit_late(&sched, &mut late, steps, false);
                            } else {
                                std::thread::sleep(saber_sched::IDLE_POLL);
                            }
                        }
                        saber_sched::StepOutcome::NoBucket(e) => {
                            tracing::error!(error = %e, "gate step loop: no captured bucket");
                            std::thread::sleep(saber_sched::IDLE_POLL);
                        }
                    }
                }
                (steps, late_at)
            })
            .context("spawn the arm's step thread")?
    };

    // 3. The consumers, in `task_order`.
    let out = runtime.block_on(async {
        use futures_util::StreamExt;
        let mut late_rx = Some(late_rx);
        let mut tasks = Vec::with_capacity(n);
        for &i in task_order {
            let pre = ready[i].take();
            let rx = if pre.is_none() { late_rx.take() } else { None };
            tasks.push(tokio::spawn(async move {
                let (sub, submitted) = match (pre, rx) {
                    (Some(r), _) => r,
                    (None, Some(rx)) => rx
                        .await
                        .map_err(|_| "the step thread dropped the late joiner".to_owned())??,
                    (None, None) => return Err(format!("stream {i} has no submission")),
                };
                let mut events = sub.events;
                let mut ids: Vec<TokenId> = Vec::new();
                let mut ttft = None;
                while let Some(ev) = events.next().await {
                    if ttft.is_none() && !ev.new_tokens.is_empty() {
                        ttft = Some(submitted.elapsed());
                    }
                    ids.extend(ev.new_tokens);
                }
                Ok::<_, String>((i, (ids, ttft.unwrap_or_default(), submitted.elapsed())))
            }));
        }
        let mut done: Vec<Option<StreamOut>> = (0..n).map(|_| None).collect();
        for t in tasks {
            let (i, row) = t.await.map_err(|e| e.to_string())??;
            done[i] = Some(row);
        }
        Ok::<_, String>(done.into_iter().flatten().collect::<Vec<_>>())
    });
    stop.store(true, Ordering::Relaxed);
    let (steps, late) = thread
        .join()
        .map_err(|_| anyhow::anyhow!("the arm's step thread panicked"))?;
    let out = out.map_err(|e| anyhow::anyhow!("a concurrent stream failed: {e}"))?;
    anyhow::ensure!(out.len() == n, "{} of {n} streams came back", out.len());
    Ok(ArmStreams { out, steps, late })
}

/// The late joiner's submit, on the arm's step thread between two steps: returns
/// `(steps run so far, on time)` and hands the submission to its consumer.
fn submit_late<E: saber_sched::StepExecutor>(
    sched: &Scheduler<E>,
    late: &mut Option<(
        EngineRequest,
        tokio::sync::oneshot::Sender<Result<Ready, String>>,
    )>,
    steps: u64,
    on_time: bool,
) -> Option<(u64, bool)> {
    let (req, tx) = late.take()?;
    let submitted = Instant::now();
    let r = sched
        .submit(req)
        .map(|s| (s, submitted))
        .map_err(|e| format!("submit the late joiner: {e}"));
    // A consumer that is gone has already failed its arm; nothing to add here.
    let _ = tx.send(r);
    Some((steps, on_time))
}

/// Drive `conc` streams through the scheduler at once and measure what a client would see.
///
/// `join_after` submits the LAST stream only once that many engine steps have run, so its prefill
/// lands between the others' decode steps — the mixed arrival. Everything is measured from the
/// client side: TTFT is submit → first event, and tok/s is the tokens a stream received over the
/// time it was in flight. THE ADMISSION IS FIXED (FN29, [`drive_streams`]): the streams are
/// submitted in stream order before the arm's first step, and the late joiner by the step thread
/// after exactly `join_after` steps, so the arm's schedule — and every proposal in it — is a
/// function of the inputs, and the shadow gate's signature covers it.
fn run_concurrent(
    sched: &Arc<Scheduler<RealExecutor<HipForward>>>,
    prompts: &[Vec<TokenId>],
    conc: usize,
    max_tokens: usize,
    join_after: Option<usize>,
) -> anyhow::Result<ConcRow> {
    let sc_before = sched.counters();
    let requests: Vec<EngineRequest> = (0..conc)
        .map(|i| EngineRequest {
            prompt_token_ids: prompts[i % prompts.len()].clone(),
            params: SamplingParams {
                max_tokens: Some(max_tokens),
                // An ID gate: temperature 0, for the reason above.
                temperature: 0.0,
                ..SamplingParams::default()
            },
        })
        .collect();
    let identity: Vec<usize> = (0..conc).collect();
    let t0 = Instant::now();
    let arm = drive_streams(sched, requests, join_after, &identity)?;
    let wall = t0.elapsed().as_secs_f64();
    let late_note = match (join_after, arm.late) {
        (Some(k), Some((at, true))) => format!(", the late joiner after step {at} (asked {k})"),
        (Some(k), Some((at, false))) => {
            // ws-p10's case, now decided by the schedule rather than a 30 s host deadline: the
            // other streams' token budget is shorter than `k` steps. An arm that silently became a
            // plain concurrency arm is a gate reporting the wrong claim, so it says so.
            println!(
                "mixed     the other streams finished after {at} step(s), before step {k}: the \
                 late joiner was submitted then, so this arm is a concurrency arm and not a \
                 mid-run arrival (their token budget is shorter than {k} steps)"
            );
            format!(", the late joiner at idle after step {at} (asked {k})")
        }
        _ => String::new(),
    };
    println!(
        "conc      admission: fixed order (FN29): {} stream(s) submitted in stream order before \
         the arm's first step{late_note}; {} step(s)",
        conc - usize::from(join_after.is_some()),
        arm.steps
    );
    let out = arm.out;
    let ran = arm.steps;

    let tokens: usize = out.iter().map(|(ids, _, _)| ids.len()).sum();
    let mut ttfts: Vec<f64> = out
        .iter()
        .map(|(_, t, _)| t.as_secs_f64() * 1000.0)
        .collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let per_stream: f64 = out
        .iter()
        .map(|(ids, _, in_flight)| ids.len() as f64 / in_flight.as_secs_f64().max(1e-9))
        .sum::<f64>()
        / out.len() as f64;
    let after = sched.with_executor(|e| e.counters());
    let max_batch = after
        .launches_by_shape
        .keys()
        .filter(|k| k.phase == Phase::Decode)
        .map(|k| k.rows)
        .max()
        .unwrap_or(0);
    // The decode shapes the engine has replayed. `launches_by_shape` is keyed per shape and holds
    // that shape's LAUNCH count, not a step count, so this is cumulative over the engine's life
    // and is reported as such. `rows / width` is the batch (EXEC.md §2: a row count is not a
    // bucket). The per-run question — did the width narrow at all — is the scheduler's own
    // counter below.
    let mut shapes: Vec<(u32, u32)> = after
        .launches_by_shape
        .keys()
        .filter(|k| k.phase == Phase::Decode)
        .map(|k| (k.rows / k.width.max(1), k.width))
        .collect();
    shapes.sort_unstable();
    shapes.dedup();
    Ok(ConcRow {
        conc,
        aggregate_tps: tokens as f64 / wall.max(1e-9),
        per_stream_tps: per_stream,
        ttft_p50_ms: ttfts[ttfts.len() / 2],
        ttft_max_ms: *ttfts.last().unwrap_or(&0.0),
        steps: ran as usize,
        max_batch,
        shapes,
        width_narrowed_steps: sched.counters().width_narrowed_steps
            - sc_before.width_narrowed_steps,
        slot_caps_absorbed: sched.counters().slot_caps_absorbed - sc_before.slot_caps_absorbed,
        ids: out.into_iter().map(|(ids, _, _)| ids).collect(),
    })
}

/// The smallest prod-shaped context band that holds `tokens`.
fn ctx_band_for(tokens: u32) -> u32 {
    [8192u32, 32_768, 65_536, 262_144]
        .into_iter()
        .find(|b| *b >= tokens)
        .unwrap_or(262_144)
}

// =====================================================================================
// `saber serve --engine real`
// =====================================================================================

/// Arguments of `saber serve --engine real`.
pub struct ServeArgs {
    /// How to build the engine.
    pub engine: EngineArgs,
    /// D3: `--vision-max-patches` (0 = text-only, the default): the vision tower is uploaded and
    /// its arena sized for this many patches per image, and the OpenAI plane accepts `image_url`
    /// parts. TP = 1 only until D4 (refused by name at TP 2).
    pub vision_max_patches: u32,
    /// P27-11: `--vision lazy` (the streamed tower) rather than `on` (resident).
    pub vision_lazy: bool,
    /// D3: `--limit-mm-per-prompt` for images (the reference server's default 999).
    pub image_limit: u32,
    /// The checkpoint's `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// The chat template (`qwen-fixed-v22.3.jinja`).
    pub chat_template: PathBuf,
    /// Address to bind.
    pub host: String,
    /// Port to bind; 0 takes an ephemeral one and prints it.
    pub port: u16,
    /// The name clients ask for.
    pub served_model_name: String,
    /// `--max-logprobs` (V7).
    pub max_logprobs: i32,
    /// The parser flags, resolved (V9).
    pub parsers: saber_serve::state::ResolvedParsers,
    /// A DIAGNOSTIC arm: write every sampled row's seam, its parameters, its noise key and the
    /// token it produced to this file as JSONL, for `tools/gates/sampler_stream.sh` to replay.
    /// The serving path never sets it.
    pub sample_dump: Option<PathBuf>,
    /// Parking-file bytes, GiB. Needs `park_nvme_path`. 0 means no spill tier.
    pub park_nvme_gib: f64,
    /// Where the parking file lives.
    pub park_nvme_path: Option<PathBuf>,
    /// Let the scheduler EVICT on its own when an admission is refused for KV or GDN state
    /// (K5's on-pressure policy). Without it a park only happens when something asks for one.
    pub park_on_pressure: bool,
    /// The whole park policy the flags asked for (ws-k18): `park_on_pressure`, the victim order,
    /// the per-session cap and the restore bound.
    pub park_policy: saber_sched::ParkPolicy,
    /// THE CONTROL for the bounded-restore law: refuse every restore on the executor.
    pub park_fail_restores: bool,
    /// THE CONTROL for the stale-anchor law (ws-f35), and never the serving default.
    ///
    /// The engine proposes inside the decode step, from the device's ARGMAX; a row with a
    /// temperature then has that token replaced by the host sampler's draw, and the block it made
    /// is the continuation of a token the sequence never had. The law drops such a block. With
    /// this, it is offered anyway — the engine as it was before ws-f35 — so the two arms can be
    /// driven over the same corpus and the difference measured instead of argued.
    pub keep_stale_anchor_blocks: bool,
    /// THE CONTROL for the drawn acceptance law (ws-f40), never the serving default: the pre-F40
    /// law — argmax acceptance of the drafted prefix, the last row drawn, noise keyed by the step.
    /// `RealExecutor::step_keyed_draws`; every row it draws is counted in `step_keyed_rows`.
    pub step_keyed_draws: bool,
    /// FN52: prefix caching (`--enable-prefix-caching`, `--prefix-cache-*`); OFF = the control.
    pub prefix_cache: saber_sched::PrefixCachePolicy,
    /// FN66: `--warm-prompts FILE` (it turns `prefix_cache` on).
    pub warm_prompts: Option<std::path::PathBuf>,
    /// FN69: `--spec-lookup on` -- prompt-lookup speculation in front of the DFlash2 mailbox
    /// (`saber_engine::LookupDrafter`); `None` = off = the control and the default.
    pub spec_lookup: Option<saber_engine::LookupConfig>,
}

/// Serve the OpenAI `/v1` plane over the real engine.
pub fn serve(args: &ServeArgs) -> anyhow::Result<()> {
    // Cold start, from as close to process start as this function can see. Argument parsing above it
    // is microseconds; the number that matters to a client is the EXTERNAL wall from `exec` to the
    // first token, which the harness measures around the whole command, and the phase ledger below
    // accounts for the inside of it.
    let boot = std::time::Instant::now();

    // ---- the host plane, started FIRST and joined last -----------------------------------------
    // The tokenizer and the chat template are pure host work: files parsed on the CPU, nothing that
    // touches HIP, nothing the engine needs and nothing that needs the engine. Loading them after
    // the engine was ready put them on the critical path for ~300 ms of a 3.1 s boot (ws-k11's
    // first measurement put them in `unaccounted`, which is what a phase ledger is for). Started
    // here, they finish long before the 2.4 s weight upload does and cost the boot nothing.
    let (tok_path, tpl_path) = (args.tokenizer.clone(), args.chat_template.clone());
    // The thread times ITSELF and hands the duration back. Timing it from out here would measure the
    // wall until the join — which is the whole engine startup — and report the tokenizer as having
    // taken 2.7 seconds. The first cut did exactly that, and the ledger's own `unaccounted` line
    // going to zero is what showed it up.
    type HostPlane = anyhow::Result<(
        saber_model::Tokenizer,
        TemplateRenderer,
        std::time::Duration,
    )>;
    let host_plane = std::thread::Builder::new()
        .name("saber-host-plane".to_owned())
        .spawn(move || -> HostPlane {
            let t = std::time::Instant::now();
            let tokenizer = saber_model::Tokenizer::from_file(&tok_path)
                .with_context(|| format!("loading {}", tok_path.display()))?;
            let renderer = TemplateRenderer::from_file(&tpl_path)
                .with_context(|| format!("loading {}", tpl_path.display()))?;
            Ok((tokenizer, renderer, t.elapsed()))
        })
        .context("spawning the host-plane loader")?;

    // D4 lifted D3's TP > 1 refusal: the tower runs on both ranks, rank 1 fed by `Command::Vision`.
    let mut tier_args = args.engine.clone();
    tier_args.vision_max_patches = args.vision_max_patches;
    tier_args.vision_lazy = args.vision_lazy && args.vision_max_patches > 0;
    let (eargs, mut worker) = start_tier(&tier_args)?;
    let ecfg = eargs.config();
    let (forward, ledger) = match saber_engine::spawn(ecfg, eargs.ledger(32)?) {
        Ok(v) => v,
        Err(e) => {
            if let Some(w) = worker.take() {
                println!("tier           {}", w.stop());
            }
            return Err(e.into());
        }
    };
    if let Some(w) = worker.as_mut() {
        w.check()?;
    }
    // Cloned before the engine is moved into the executor. It is a handful of durations.
    let mut phases = forward.phases().clone();
    print_facts(&forward, &ledger, eargs.rank);

    // Joined, not loaded: by now it has been done for two seconds.
    let (tokenizer, renderer, host_ms) = host_plane
        .join()
        .map_err(|_| anyhow::anyhow!("the host-plane loader panicked"))??;
    phases.record(
        saber_engine::Phase::HostPlane,
        host_ms,
        "tokenizer + chat template, on their own thread",
    );
    phases.overlapped(saber_engine::Phase::HostPlane);
    let tokenizer = Arc::new(tokenizer);
    let renderer = Arc::new(renderer);
    // Beside the tokenizer, which is where a checkpoint keeps it. A pack that ships no
    // `generation_config.json` falls back to `<|im_end|>` alone, and the banner below says which
    // happened -- an end-of-turn the server does not know about is a request that runs its whole
    // budget emitting whatever comes after the conversation.
    let eos_dir = args
        .tokenizer
        .parent()
        .map_or_else(|| std::path::PathBuf::from("."), Path::to_path_buf);
    let eos = crate::serve::eos_ids(&tokenizer, &eos_dir)?;

    let mut config = scheduler_config(
        &forward,
        ctx_band_for(args.engine.max_ctx),
        &args.engine.width_rule,
        args.engine.ragged_ceiling,
        args.engine.mixed_prefill,
        args.engine.prefill_fill_above,
    )?;
    config.prefix_cache = args.prefix_cache;
    println!(
        "prefixcache    {} (FN52: verify {}, host budget {} MiB/rank, min {} tokens){}",
        if args.prefix_cache.enabled {
            "on"
        } else {
            "off"
        },
        args.prefix_cache.verify,
        args.prefix_cache.host_budget_bytes >> 20,
        args.prefix_cache.min_tokens,
        if args.warm_prompts.is_some() {
            " -- on by --warm-prompts (FN66)"
        } else {
            ""
        }
    );
    // FN66: the warm prompts, parsed before the boot spends the cards' time (a bad file refuses)
    let warm = args
        .warm_prompts
        .as_ref()
        .map(|p| saber_serve::warm::load(p).map(|w| (p.clone(), w)))
        .transpose()?;
    let model = ModelInfo {
        served_model_name: args.served_model_name.clone(),
        aliases: Vec::new(),
        max_model_len: config.max_context_tokens,
        root: args.engine.pack.display().to_string(),
        eos_token_ids: eos.clone(),
    };
    println!("eos            {eos:?} (from {})", eos_dir.display());
    println!(
        "mixed          {} (C4: {})",
        match (config.mixed.enabled, config.mixed.fused) {
            (false, _) => "OFF",
            (true, false) => "ON serial",
            (true, true) => "ON fused",
        },
        if config.mixed.enabled {
            format!(
                "a decode step carries the queue head's next prefill chunk, budget {} rows \
                 (decode rows + chunk rows; prod's max_num_batched_tokens is 8192){}",
                config.mixed.max_batched_tokens,
                if config.mixed.fused {
                    "; C9 FUSED: chunk + decode rows as one forward at the smallest prefill rung \
                     holding both (decode rows at the rung's class), the chunk capped at the \
                     widest rung minus the decode rows"
                } else {
                    ""
                }
            )
        } else {
            "a prefill chunk takes a step of its own while the decoding sequences wait; \
             --engine-mixed-prefill turns the mixed step on"
                .to_owned()
        }
    );
    println!(
        "buckets        {} captured, widths {:?}, max batch {}",
        config.buckets.len(),
        config.width.widths,
        config.max_batch()
    );
    // THE VERIFY-WIDTH RULE, and whether it can do anything on this plan.
    //
    // Either rule can only ask for a width the plan CAPTURED (`bucket_width` rounds up), so a
    // rule with one reachable width is INERT and that is a boot-time fact — said at boot, beside
    // the widths it is true of. The default is the batch table (ws-t2): its cells are measured
    // numbers, so they are printed as a table and not as a sentence. `--width-rule ema` is prod's
    // acceptance cap, which is available and off because it was measured and lost (153.0 / 405.1
    // against static 8's 158.0 / 416.2 on the paired BetterBench).
    {
        let wp = &config.width;
        match &wp.rule {
            WidthRule::ByBatch(t) => {
                println!(
                    "verify width   BY BATCH {t} (batch -> width, carry-forward); reachable {:?} \
                     of captured {:?}",
                    wp.reachable_widths(),
                    wp.widths,
                );
                for (batch, want) in t.rows() {
                    let runs = wp.bucket_width(*want);
                    println!(
                        "               batch >= {batch}: width {want}{}",
                        if runs == *want {
                            String::new()
                        } else {
                            format!(" -> runs {runs} (no captured shape at {want})")
                        }
                    );
                }
            }
            WidthRule::AcceptanceEma => println!(
                "verify width   EMA cap alpha {} margin {} min {} min_batch {} (prod's own \
                 knobs); reachable {:?} of captured {:?}",
                wp.prod.alpha,
                wp.prod.margin,
                wp.prod.min_width,
                wp.prod.min_batch,
                wp.reachable_widths(),
                wp.widths,
            ),
        }
        if !wp.can_narrow() {
            println!(
                "               INERT on this plan: the rule can reach only width {}, so no step \
                 will ever run narrower than the ceiling. Capture the width its table asks for \
                 (`--engine-decode-widths 1,5,8`) to give it somewhere to go.",
                wp.ceiling_width()
            );
        }
    }
    // THE RAGGED SCHEDULE (C2): which bands the engine captured and the per-slot rule, or the
    // rectangles alone under `--rectangular-widths`.
    print_ragged_banner(&config, forward.facts());
    println!(
        "prefill        chunk sizes {:?}",
        config
            .prefill_buckets
            .iter()
            .map(|b| b.tokens)
            .collect::<Vec<_>>()
    );
    let pool_blocks = forward.facts().kv_pool_blocks;
    let stride = forward.facts().kv_block_stride;
    // Taken before the engine is moved into the executor: the parking tier is sized from the
    // engine's own staging slot (ws-k14).
    let forward_facts = forward.facts().clone();
    let t_sched = std::time::Instant::now();
    // THE PASS'S OWN DRAFTER on the serve plane. A run with `--engine-draft-pack` proposes inside
    // every decode step and this is the mailbox the next step's verify empties; without one the
    // executor keeps `NoDrafter` and a width-`w` bucket runs `w` rows of which `w - 1` are a token
    // the model cannot want -- SPEC=0 reached through the verify path, which is what it always was.
    let serve_width = forward.facts().max_width;
    let serve_drafts =
        forward.facts().draft_resident.is_some() || args.engine.dspine_socket.is_some();
    // THE PAIRING CONTROLS, REFUSED WHEN THEY CANNOT BE HONOURED (ws-f37). `--propose-in-step`
    // moves the engine's own proposal back inside the step; with no resident drafter there is no
    // proposal to move. `--keep-stale-anchor-blocks` offers a block the in-step order anchored on
    // an argmax the sampler replaced; after the commit no such block exists, so the control has
    // nothing to keep unless the order is the in-step one.
    if args.engine.propose_in_step && !(serve_width > 1 && serve_drafts) {
        anyhow::bail!(
            "--propose-in-step needs the engine's own drafter (`--engine-draft-pack` and a verify \
             width above 1): without one nothing proposes, in the step or after it"
        );
    }
    if args.keep_stale_anchor_blocks && !args.engine.propose_in_step {
        anyhow::bail!(
            "--keep-stale-anchor-blocks is the control for the IN-STEP order's stale-anchor law: \
             after ws-f37 the proposal runs from the committed token and no block is stale. Pair \
             it with --propose-in-step"
        );
    }
    let mut executor = RealExecutor::new(forward);
    // C9: the fused mixed forward, when the mixed step is on in that form.
    executor.set_fused_mixed(config.mixed.fused);
    // THE BOUNDED-RESTORE CONTROL (ws-k18), off unless the flag asks: every restore refused.
    executor.fail_restores(args.park_fail_restores);
    // THE ACCEPTANCE-LAW CONTROL (ws-f40 / ws-q3), off unless the flag asks. Printed at boot and
    // counted per row (`step_keyed_rows`), so a run can never be mistaken for the other law.
    executor.step_keyed_draws(args.step_keyed_draws);
    println!(
        "draw law       {}",
        if args.step_keyed_draws {
            "STEP-KEYED (--step-keyed-draws: the pre-F40 CONTROL — argmax acceptance, the last \
             row drawn; not prod's law)"
        } else {
            "every sampled verify row a draw keyed by (seed, output position, choice); a draft \
             is accepted iff it IS that draw (ws-f40, prod's rejection sampler in distribution)"
        }
    );
    if serve_width > 1 && serve_drafts {
        executor.set_drafter(Box::new(EngineDrafter::new(serve_width - 1)));
        if args.engine.dspine_socket.is_some()
            || (matches!(
                args.engine.spec_tree,
                saber_engine::spec_tree::SpecTreeForm::Best16
                    | saber_engine::spec_tree::SpecTreeForm::Ancestor9
            ) && args.engine.draft_block == 16)
        {
            executor.enable_scalar_context_tail();
        }
        if args.engine.dspine_socket.is_some() {
            println!("drafter        DSpine Unix socket, scheduled verify width {serve_width}");
        } else {
            println!("drafter        the engine's own DFlash2 proposal, block {serve_width}, one per live slot per decode step");
        }
        // FN69: prompt-lookup speculation in front of the mailbox (the verify is unchanged).
        if let Some(cfg) = args.spec_lookup {
            executor.set_drafter(Box::new(saber_engine::LookupDrafter::new(
                Box::new(EngineDrafter::new(serve_width - 1)),
                cfg,
            )));
            println!(
                "speclookup     ON: prompt-lookup n-grams {}..{}, merge bar {} (FN69; a confident \
                 match REPLACES the DFlash2 block, verify unchanged; census line `lookup`)",
                cfg.nmin, cfg.nmax, cfg.min_match
            );
        } else {
            println!("speclookup     off (FN69's control and the default)");
        }
        // WHICH ORDER THE PROPOSAL TAKES, and which route the stale-anchor law takes, printed at
        // boot rather than inferred from the flags (`CLAUDE.md` rule 5).
        executor.keep_stale_anchor_blocks(args.keep_stale_anchor_blocks);
        if args.engine.propose_in_step {
            println!(
                "proposal       INSIDE the step, from the device's argmax (--propose-in-step: the \
                 pairing CONTROL, not prod's order)"
            );
            println!(
                "anchor         a block whose anchor the step did not commit is {}",
                if args.keep_stale_anchor_blocks {
                    "OFFERED (--keep-stale-anchor-blocks: the CONTROL, not the law)"
                } else {
                    "DROPPED and counted (the law)"
                }
            );
        } else {
            println!(
                "proposal       AFTER the host sampler, from each slot's COMMITTED token (prod's \
                 order, ws-f37); a stale anchor is a DEFECT, refused and counted"
            );
        }
    }
    if let Some(path) = &args.sample_dump {
        executor
            .dump_samples_to(path)
            .with_context(|| format!("opening the sample dump {}", path.display()))?;
        println!("sample dump    every sampled row -> {}", path.display());
    }
    // ---- THE PARKING TIER, on the real engine (ws-k14) -----------------------------------------
    //
    // `image_bytes` is the ENGINE's own slot size and not a guess: the engine refused to start
    // unless `--engine-park-staging-mib` holds `park_staging_slots` whole images at `max_ctx`, so
    // one slot of that buffer is exactly the largest image this engine can produce. A tier sized
    // from anything else would either refuse a park the engine can make or accept one it cannot.
    let park_image_bytes = {
        let slots = u64::from(forward_facts.park_staging_slots.max(1));
        (forward_facts.park_staging_bytes / slots).max(1)
    };
    let park = ServeParkOptions::from_gib(
        args.engine.park_ram_gib,
        args.park_nvme_gib,
        args.park_nvme_path.clone(),
    )
    .with_image_bytes(park_image_bytes)
    .with_policy(args.park_policy);
    if args.park_policy.on_pressure != args.park_on_pressure {
        bail!("the park policy's on-pressure disagrees with --park-on-pressure");
    }
    if park.enabled() && forward_facts.park_staging_bytes == 0 {
        bail!(
            "--park-ram-gib was given and the engine has no park staging buffer: a tier with \
             nowhere to copy from would refuse every eviction at the moment it mattered. Pass \
             `--engine-park-staging-mib N` as well."
        );
    }
    let scheduler = Arc::new(attach_park_tier(
        Scheduler::new(config, executor, model, ledger)?,
        &park,
    )?);
    if park.enabled() {
        // THE SCHEDULER'S POLICY, not the options' (ws-k17): this line printed the options'
        // `on-pressure true` for a whole run whose scheduler never received it.
        let policy = scheduler.config().park;
        println!(
            "parking        ram {} MiB, nvme {} MiB at {}, image {} MiB (the engine's own \
             staging slot), on-pressure {} (the scheduler's), max parks per session {}, victim \
             {}, restore attempts {}{}",
            park.ram_bytes / (1024 * 1024),
            park.nvme_bytes / (1024 * 1024),
            park.nvme_path
                .as_ref()
                .map_or_else(|| "-".to_owned(), |p| p.display().to_string()),
            park.image_bytes / (1024 * 1024),
            policy.on_pressure,
            if policy.max_parks_per_session == u32::MAX {
                "unbounded".to_owned()
            } else {
                policy.max_parks_per_session.to_string()
            },
            policy.victim.label(),
            policy.max_restore_attempts,
            if args.park_fail_restores {
                " -- EVERY RESTORE REFUSED (--engine-park-fail-restores, the control)"
            } else {
                ""
            },
        );
        if policy != park.policy {
            bail!(
                "the scheduler runs park policy {policy:?} and the options asked for {:?}",
                park.policy
            );
        }
    } else {
        println!("parking        OFF (no --park-ram-gib): an eviction refuses by name");
    }
    phases.record(
        saber_engine::Phase::Scheduler,
        t_sched.elapsed(),
        "arenas reserved against the engine's geometry",
    );
    // ---- the capacity side of the ledger (ws-k12) ----------------------------------------------
    // Every reservation on this rank, then the one conversion an operator actually decides on: how
    // many resident KV tokens are left after everything, and how many concurrent sessions that is.
    // Prod's comparators are in tokens, so the arithmetic is done here, once, with its terms shown.
    let (report, predicted_sessions) = capacity_report(
        &scheduler,
        eargs.rank,
        args.engine.max_ctx,
        stride,
        pool_blocks,
    );
    print!("{report}");
    // The arithmetic above is checked against the allocator, which is the authority: admit sessions
    // of `max_ctx` until the arenas refuse, and count what got in. A census that disagrees with the
    // thing it describes is worse than no census.
    print!(
        "{}",
        admit_until_refusal(&scheduler, args.engine.max_ctx, predicted_sessions)
    );
    let (free_kv, free_state) = scheduler.arena_free();
    println!(
        "arenas         kv {free_kv} blocks of the engine's {pool_blocks}, {free_state} GDN state \
         slot sets — one ledger, the engine's own geometry"
    );
    if free_kv > pool_blocks {
        bail!(
            "the scheduler can hand out {free_kv} KV blocks and the engine's pool has \
             {pool_blocks}: a block id past the pool is a kernel reading a page that is not there"
        );
    }

    let steps = spawn_step_loop(Arc::clone(&scheduler));
    let state = saber_sched::build_state_with_vision(
        Arc::clone(&scheduler),
        Arc::new(ModelCodec::new(Arc::clone(&tokenizer))),
        renderer,
        args.max_logprobs,
        &args.parsers,
        args.vision_max_patches,
        args.image_limit,
    );
    if args.vision_max_patches > 0 {
        println!(
            "vision         image parts ON: tower {}, arena {} patches per image requested, {} \
             image(s) per prompt, data: URLs only (remote fetch refused by name)",
            if args.vision_lazy {
                "STREAMED (`--vision lazy`, P27-11: weights in pinned host RAM, the prefill arena \
                 lent per image; the boot's `visionlazy` line names the patches it holds)"
            } else {
                "resident (`--vision on`)"
            },
            args.vision_max_patches,
            args.image_limit
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the tokio runtime")?;
    let addr = format!("{}:{}", args.host, args.port);
    let warm_sched = Arc::clone(&scheduler);
    runtime.block_on(async move {
        let t_bind = std::time::Instant::now();
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        let bound = listener.local_addr().context("local addr")?;
        phases.record(saber_engine::Phase::ServerBind, t_bind.elapsed(), &addr);
        // The clock the ledger's shares are taken against is the CLI's, not the engine thread's, so
        // that everything between them — the tokenizer, the template, the scheduler, the runtime —
        // lands in `unaccounted` rather than vanishing.
        phases.total = boot.elapsed();
        print!("{}", phases.render("boot           phase ledger (ws-k11)"));
        println!(
            "boot           TOTAL {:.0} ms to listening; the phases above account for {:.0} ms of it",
            phases.total.as_secs_f64() * 1e3,
            phases.serial().as_secs_f64() * 1e3
        );
        println!("listening      http://{bound}/v1  (SIGINT or SIGTERM to stop)");
        // FN66: the warm prompts run beside the listener (a client's request queues behind them)
        if let Some((path, prompts)) = warm {
            println!(
                "warmprompts    {}: {} prompt(s), each head prefilled once into the prefix cache",
                path.display(),
                prompts.len()
            );
            let probe: saber_serve::warm::CacheProbe = Arc::new(move || {
                let (pc, entries, held) = warm_sched.prefix_counters();
                (
                    held,
                    format!(
                        "{entries} entries, {:.1} MiB host snapshots a rank, {} saves",
                        held as f64 / f64::from(1u32 << 20),
                        pc.saves
                    ),
                )
            });
            tokio::spawn(saber_serve::warm::run(Arc::clone(&state), prompts, probe));
        }
        // The signal is what makes this server stoppable. Without it a `kill` lands on a process
        // whose engine thread is inside a device call, nothing exits, and whoever launched it waits
        // forever holding the card — which is exactly what happened on 2026-09-19 (2 h 55 m).
        // FN73: and a failed rank (the fatal) stops it the same way.
        saber_serve::server::serve_with_shutdown(state, listener, shutdown_or_fatal())
            .await
            .context("serving")
    })?;
    // Counters first: the scheduler is about to be dropped on the teardown thread. The block is
    // `saber_engine::shutdown_census` (ws-v8), and every number in it is also live on `/metrics`
    // under the `saber:` name `docs/SHADOW.md` §2's table gives it — the prompt, counters,
    // sampler, proposal, spec, session, forks, clean, drafted, parking, admission and sched lines.
    // WHICH BRANCH DECIDED THE TOKENS, WHICH ROUTE THE PROPOSAL TOOK and WHETHER ANYTHING WAS EVER
    // PARKED are all in it: a serve that sampled every row and one that fell back to the argmax on
    // all of them print the same everything else (`CLAUDE.md` rule 5).
    {
        let sc = scheduler.counters();
        let ac = scheduler.admission_counters();
        let (ec, dc) = scheduler.with_executor(|e| {
            (
                e.counters(),
                saber_engine::ForwardRunner::draft_counts(e.forward_mut()),
            )
        });
        print!(
            "{}",
            saber_engine::shutdown_census(&sc, &ac, &ec, dc.as_ref())
        );
        // FN52: the prefix cache's which-path counters (all zero with the cache off).
        let (pc, entries, held) = scheduler.prefix_counters();
        println!(
            "prefixcache    lookups {} hits {} adopted {} hit_tokens {} saves {} save_fail {} \
             load_fail {} evict_budget {} evict_pressure {} verify {}/{} skipped {} entries {} \
             held {} MiB",
            pc.lookups,
            pc.hits,
            pc.adopted,
            pc.hit_tokens,
            pc.saves,
            pc.save_failures,
            pc.load_failures,
            pc.evictions_budget,
            pc.evictions_pressure,
            pc.verify_match,
            pc.verify_match + pc.verify_mismatch,
            pc.verify_skipped,
            entries,
            held >> 20
        );
        // C5: `--engine-host-trace`'s tables (a DIAGNOSTIC serve: its tok/s is not published),
        // decode and PREFILL, with the chunks by bucket. Nothing is printed without the flag.
        if let Some(t) = scheduler.with_executor(|e| {
            saber_engine::ForwardRunner::host_trace_table(e.forward_mut(), "serve rank 0 (leader)")
        }) {
            print!("{t}");
        }
    }
    if saber_serve::fatal::get().is_none() {
        // FN73: the orderly stop -- rank 1's exit from here on is the one the leader asked for
        if let Some(w) = worker.as_ref().and_then(|w| w.watch.as_ref()) {
            w.disarm();
        }
    }
    match stop_with_deadline(scheduler, steps, SHUTDOWN_DEADLINE) {
        Ok(ran) => {
            if let Some(f) = saber_serve::fatal::get() {
                if let Some(w) = worker.take() {
                    println!("tier           {}", w.stop());
                }
                bail!("{}", f.line());
            }
            // The engine's thread is joined by now, so rank 1 has its `Stop`; this is the bounded
            // wait for it to actually go, and it is the difference between "both cards released" and
            // "one of them is still held by a process nobody is watching".
            if let Some(w) = worker.take() {
                println!("tier           {}", w.stop_checked());
            }
            println!("stopped        {ran} engine steps, card released");
            Ok(())
        }
        Err(e) => {
            eprintln!("stuck          {e}");
            if let Some(w) = worker.take() {
                eprintln!("stuck          {}", w.stop());
            }
            eprintln!(
                "stuck          leaving without running destructors: `std::process::exit` with a \
                 live HIP thread segfaults, and this process must not keep the card"
            );
            // FN73: a fatal leaves with its own code (`_exit`, no atexit handler under the live
            // HIP thread)
            saber_serve::fatal::exit_if_fatal();
            // Non-zero, and no destructor runs: the HIP thread is still in there.
            std::process::abort();
        }
    }
}

/// How long the teardown may take before the process gives up and aborts.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(30);

/// FN73: SIGINT, SIGTERM or the fatal (a failed rank), whichever first.
pub(crate) async fn shutdown_or_fatal() {
    tokio::select! {
        () = shutdown_signal() => {
            // FN73b: a signal stop -- a rank that exits on it (the same process group's SIGINT,
            // or the leader's `Stop`) is the stop's, not a failure
            saber_serve::fatal::stop_requested();
        }
        () = saber_serve::fatal::wait() => {
            println!("fatal          the HTTP plane stops: listener closed, in-flight requests ended");
        }
    }
}

/// Resolves on SIGINT or SIGTERM.
///
/// Both, because a person types ctrl-c and a script sends `kill`; a server that only handles one of
/// them is a server that hangs under the other.
pub(crate) async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = term => {}
    }
    println!("signal         received: no new requests, stopping the step loop");
}

// =====================================================================================
// TP = 2: the other rank, as a child process
// =====================================================================================

/// FN73 (DEBUG): `--fail-rank-after-steps R:N` on the 27B pair -> rank 1's step count (R must be
/// 1: rank 1 is the follower whose loop fails; 0 = no injection).
pub fn fail_rank1_after(spec: Option<&str>) -> anyhow::Result<u64> {
    let Some(spec) = spec else { return Ok(0) };
    let (r, n) = spec
        .split_once(':')
        .with_context(|| format!("--fail-rank-after-steps {spec}: the form is R:N"))?;
    let (r, n): (u32, u64) = (
        r.parse().context("--fail-rank-after-steps R")?,
        n.parse().context("--fail-rank-after-steps N")?,
    );
    anyhow::ensure!(
        r == 1 && n > 0,
        "--fail-rank-after-steps {spec}: the 27B pair injects into rank 1 (the follower) after N > 0 \
         steps"
    );
    Ok(n)
}

/// Rank 1, spawned and owned by rank 0.
///
/// Two processes rather than two threads because `hipIpcOpenMemHandle` opens "an interprocess memory
/// handle exported from *another process*" and answers `hipErrorInvalidContext` to a handle this
/// process exported (measured 2026-09-18, `saber_exec::peer`). So the peer-visible all-reduce cannot
/// be exercised any other way, and the deployment shape is decided by the hardware.
///
/// Rank 0 owns the child's lifetime because rank 0 is the only process anything else talks to: one
/// command starts the tier and one stops it. [`Worker::stop`] is bounded at every step — SIGTERM, a
/// deadline, SIGKILL, another deadline — because the failure that cost 2 h 55 m of a shared GPU lock
/// on 2026-09-19 was a wait with nothing behind it.
struct Worker {
    child: std::process::Child,
    port: u16,
    device: i32,
    /// Whether this worker has already been through [`Worker::shutdown`].
    ///
    /// `Drop` is the whole point of this field: `stop()` is called on the paths that finish, and a
    /// TP=2 gate has a dozen that do not — every `?` between the spawn and the end. `Child`'s own
    /// `Drop` neither kills nor waits (documented, and the right default for a supervisor), so a
    /// leaked `Worker` leaves rank 1 running on the other card, and when it does exit it stays a
    /// ZOMBIE because nobody ever reaps it. Measured 2026-09-20: a TP=2 gate that failed in its
    /// concurrency arm left the pair up for 46 minutes on 30 s of CPU, both cards idle, the leader
    /// asleep on a zombie child, holding the shared GPU lock with four jobs queued behind it.
    reaped: bool,
    /// FN73: the watch that turns rank 1's unasked exit into the process's fatal.
    watch: Option<saber_serve::fatal::RankWatch>,
    /// FN73b: the status of the exit the leader asked for (reaped without a kill), for
    /// [`Worker::stop_checked`].
    asked_exit: Option<std::process::ExitStatus>,
}

impl Worker {
    /// Rank 1's command line: every plan input the follower needs, on its argv (the host tests
    /// parse it back through `Cli`).
    fn command(args: &EngineArgs, port: u16) -> anyhow::Result<std::process::Command> {
        let exe = std::env::current_exe().context("finding this executable")?;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("engine-worker")
            .arg("--pack")
            .arg(&args.pack)
            .arg("--prebuilt")
            .arg(&args.prebuilt)
            .arg("--hip-lib")
            .arg(&args.hip_lib)
            .arg("--rank")
            .arg("1")
            .arg("--device")
            .arg(args.peer_device.to_string())
            .arg("--peer-port")
            .arg(port.to_string())
            .arg("--budget-mib")
            .arg(args.budget_mib.to_string())
            .arg("--max-seqs")
            .arg(args.max_seqs.to_string())
            .arg("--max-ctx")
            .arg(args.max_ctx.to_string())
            .arg("--deadline-s")
            .arg(args.deadline_s.to_string());
        if !args.capture {
            // Keep the follower on the leader's explicitly selected transport.
            cmd.arg("--no-capture");
        }
        if args.host_ar {
            cmd.arg("--host-ar");
        }
        if let Some(path) = &args.rccl_lib {
            cmd.arg("--rccl-lib").arg(path);
        }
        if let Some(path) = &args.dspine_socket {
            cmd.arg("--dspine-socket").arg(path);
        }
        if !args.draft_capture_layers.is_empty() {
            cmd.arg("--draft-capture-layers").arg(
                args.draft_capture_layers
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        // D4: the tower on rank 1 too (replicated), sized as the leader's.
        if args.vision_max_patches > 0 {
            cmd.arg("--vision-max-patches")
                .arg(args.vision_max_patches.to_string());
            if args.vision_lazy {
                cmd.arg("--vision-lazy");
            }
        }
        for r in &args.prefill_rows {
            cmd.arg("--prefill-rows").arg(r.to_string());
        }
        for b in &args.decode_batches {
            cmd.arg("--decode-batches").arg(b.to_string());
        }
        // The WIDTHS too. Forgotten until 2026-09-20, and the shape of the bug is worth keeping: the
        // leader planned b1s5/b4s5 and rank 1, defaulting to width 1, planned b1s1/b4s1 -- a
        // different bucket set, so a different collective geometry, so a different protocol digest.
        // ws-h14's check refused before a single launch ("a one-shot all-reduce whose ranks disagree
        // hangs rather than answering wrongly") and named the op. Every plan input the follower needs
        // has to cross this argv, and a spawn that forwards all but one of them looks exactly like a
        // spawn that forwards them all.
        for w in &args.decode_widths {
            cmd.arg("--decode-widths").arg(w.to_string());
        }
        // AND THE ROW BANDS (C2), for the same reason: a band is a plan and a captured graph on
        // both ranks, and a follower that planned none could not replay the leader's band step.
        if args.decode_bands.is_empty() {
            cmd.arg("--rectangular-widths");
        } else {
            for r in &args.decode_bands {
                cmd.arg("--row-bands").arg(r.to_string());
            }
        }
        // AND THE BAND, for exactly the reason above. `--ar-exact` is one number in a rules table
        // and both ranks resolve their own copy of it: a leader that raised it and a follower that
        // did not would agree on every descriptor's shape and disagree on its SYMBOL, which
        // `agree_on_collectives` refuses by name before a launch -- and which, if it did not, would
        // be two halves of one reduce running two different kernels.
        if args.ar_exact {
            cmd.arg("--ar-exact");
        }
        // AND THE PREFILL OVERLAP (H36), for the same reason: a leader that slices a chunk's
        // reduces and a follower that issues them whole run two different reduce sequences over
        // one protocol, and that does not answer wrongly -- it never retires.
        if args.prefill_ar_overlap > 0 {
            cmd.arg("--prefill-ar-overlap")
                .arg(args.prefill_ar_overlap.to_string());
        }
        // AND ITS FORM (P27-1d): a leader that pushes split and a follower that runs the one-shot
        // (or pushes with another N) is two protocols; the boot's agree names it, this prevents it.
        if args.prefill_ar_push_blocks > 0 {
            cmd.args(["--ar-form", "split", "--ar-push-blocks"])
                .arg(args.prefill_ar_push_blocks.to_string());
        }
        // AND THE SMALL-CHUNK RUNGS (C6): a leader whose p96 runs the decode GEMM and a follower
        // whose p96 runs the folded one agree on every reduce's shape and on nothing else.
        if args.small_chunk_gemm > 0 {
            cmd.arg("--small-chunk-gemm")
                .arg(args.small_chunk_gemm.to_string());
            if args.small_chunk_band {
                cmd.arg("--small-chunk-band");
            }
        }
        // AND THE MID-M FOLDED TILES (G5): one summation class, but a different launch per GEMM
        // and a plan hash of their own, so both ranks take the same rules.
        // The follower's own default is ON (G5's flip), so the CONTROL must be forwarded too.
        if args.mid_chunk_gemm {
            cmd.arg("--mid-chunk-gemm");
        } else {
            cmd.arg("--no-mid-chunk-gemm");
        }
        // G6: the layout law's mid band and the skinny gate. The layout is a PROTOCOL between a
        // rank's producers and its GEMMs, and the plan hash differs, so both ranks take the same.
        if args.mid_chunk_atiled {
            cmd.arg("--mid-chunk-atiled");
        }
        if args.skinny_gate {
            cmd.arg("--skinny-gate");
        }
        // CMP27E / CMP27F: the decode prefetch twins and the one-pass 64-row head (bit-exact; both
        // ranks take the same plan). The follower's own default is ON (CMP27F's flip), so the
        // CONTROL is forwarded too.
        cmd.arg(if args.decode_pf {
            "--decode-pf"
        } else {
            "--no-decode-pf"
        });
        cmd.arg(if args.head_m64 {
            "--head-m64"
        } else {
            "--no-head-m64"
        });
        // CMP27K: the one-pass 64-row drafter gate_up (bit-exact; both ranks take the same plan).
        // Forwarded ALWAYS by value since CMP27O (the worker's default is 17: a leader's `0`, the
        // control, must reach rank 1).
        cmd.arg("--drafter-gateup-m64")
            .arg(args.drafter_gateup_m64.to_string());
        // CMP27S: the wht6 body, forwarded by value (both ranks launch one symbol; the bytes and
        // the protocol are the served ones either way, but a pair is gated as a pair).
        cmd.arg("--ar-wpar")
            .arg(if args.ar_wpar { "on" } else { "off" });
        // CMP27AE: the GEMM + wht6 fold, forwarded by value -- its wire is tile-contiguous, so a
        // fused rank cannot talk to a two-launch rank: both run the arm or neither does.
        cmd.arg("--ar-fuse")
            .arg(if args.ar_fuse { "on" } else { "off" });
        // CMP27L: the tree trio's arms (both ranks run the same tree kernels).
        cmd.arg("--b1-arms").arg(args.b1_arms.as_string());
        // CMP27T: the conc-4 arms (both ranks plan the same kernels and pick layout).
        cmd.arg("--c4-arms").arg(args.c4_arms.as_string());
        // CMP27X: the step arms (both ranks plan the same kernels).
        cmd.arg("--step-arms").arg(args.step_arms.as_string());
        // C8's gate control: the decode plans' class, which both ranks' plans must agree on.
        if args.mixed_class_proxy {
            cmd.arg("--mixed-class-proxy");
        }
        // AND THE SHARED PREFILL ARENA (C7): no plan or row moves, but the KV pool each rank
        // derives from its ledger does, and the two ranks' pools must be one pool.
        if args.shared_prefill_arena {
            cmd.arg("--shared-prefill-arena");
        }
        // AND THE SHARED DECODE-HEAD ARENA (H46), for the same reason.
        if args.shared_head_arena {
            cmd.arg("--shared-head-arena");
        }
        // AND THE DRAFTER'S KV LAW (P27-6), BY VALUE: a paged leader's runs live in a pool the
        // follower must size by the same law, or its clears and copies land outside its own.
        cmd.arg("--draft-kv").arg(if args.draft_kv_paged {
            "paged"
        } else {
            "dense"
        });
        // AND THE EMBEDDING SHARE (P27-9b), BY VALUE: a rank that shares bills 1.27 GB less and
        // derives a larger KV pool; the two ranks' pools must be one.
        cmd.arg("--draft-embed-share")
            .arg(if args.draft_embed_share { "on" } else { "off" });
        // AND THE SELECTOR'S RANK FORM (P27-12), BY VALUE: both ranks re-rank or neither, or the
        // block digests they compare differ.
        cmd.arg("--selector-rank").arg(args.selector_rank.as_str());
        if args.draft_full_vocab {
            cmd.arg("--draft-full-vocab");
        }
        // AND THE TREE ARM (P27-15c), BY VALUE: both ranks resolve one form, read one lattice.
        cmd.arg("--spec-tree").arg(args.spec_tree.as_str());
        // AND THE SAMPLED TREE LAW (P27-W2-3), BY VALUE: the follower takes the drawn path.
        cmd.arg("--spec-tree-sampled")
            .arg(if args.spec_tree_sampled { "on" } else { "off" });
        // AND THE GDN REPLAY ARM (P27-W2-5), BY VALUE: both ranks plan one recurrent form.
        cmd.arg("--gdn-replay")
            .arg(if args.gdn_replay { "on" } else { "off" });
        // AND THE HOST LEVERS (P27-14), BY VALUE: each one changes a frame both ranks read.
        cmd.arg("--host-levers").arg(args.host_levers.as_arg());
        // AND THE DEVICE-STEP FORM (P27-W2-2), BY VALUE: both ranks merge the seam on the card or
        // neither (one rank's kernel would spin on a window the other never writes).
        cmd.arg("--device-step").arg(args.device_step.as_arg());
        // THE DECODE FUSIONS (ws-h37; the default since N8), BY VALUE like the state dtype and
        // the head: the worker's `--decode-fusion` has no default, so a follower never falls back
        // to whichever default its argv parser has while the leader resolved the other one. And
        // it is the SET (ws-h38/h39: `off`, `h37` or `all`), not "on": rows 3/7 on one rank only
        // would hang (row 3 is a different reduce protocol) or diverge.
        cmd.arg("--decode-fusion")
            .arg(args.decode_fusion.worker_arg());
        // AND THE TRACE, because a pair's step time is the SLOWER rank's: rank 1 has no handle
        // above the seam, so the only way its frames, read-backs and waits are on a table is for
        // the flag to cross this argv with every other plan input (ws-h31). It prints its own
        // table on the way out, into the leader's log.
        if args.host_trace {
            cmd.arg("--host-trace");
        }
        // AND BOTH FRAME CONTROLS, for the reason every other plan input crosses this argv: the
        // step frame is a PROTOCOL and a pair that disagrees about its shape reads the other's
        // bytes as a length. The digest is the follower's check, so the follower is the rank that
        // has to be told to make it.
        if args.whole_step_frame {
            cmd.arg("--whole-step-frame");
        }
        if args.frame_digest {
            cmd.arg("--frame-digest");
        }
        if args.race_check {
            cmd.arg("--race-check");
        }
        if args.race_shadow {
            cmd.arg("--race-check").arg("shadow");
        }
        if let Some(p) = args.poison_boot.as_deref() {
            cmd.arg("--poison-boot").arg(p);
        }
        if args.constant_seed_control {
            cmd.arg("--constant-seed-control");
        }
        if args.gather_per_row {
            cmd.arg("--gather-per-row");
        }
        // AND THE WAIT SCHEDULE (ws-h32), so the control arm's two ranks wait the same way: a pair's
        // step is the slower rank's, and a control that changed one rank is not a control.
        if args.fixed_step_wait {
            cmd.arg("--fixed-step-wait");
        }
        // FN73 (DEBUG): the injected rank failure is rank 1's own
        if args.fail_after_steps > 0 {
            cmd.arg("--fail-after-steps")
                .arg(args.fail_after_steps.to_string());
        }
        // FN57: the host-wait arm, by value -- a pair whose ranks wait differently is not an arm
        cmd.arg("--host-wait").arg(args.host_wait.to_string());
        // THE STORED GDN STATE DTYPE (C1): both ranks plan the same graph or the collective
        // digests refuse at boot, and a pair with one narrow rank is not an arm.
        cmd.arg("--gdn-state-dtype")
            .arg(if args.gdn_state_f16 { "f16" } else { "f32" });
        // The head the pair plans (N2, N5): both ranks plan the same graph or neither binds. It is
        // ALWAYS forwarded, by value, and the worker's `--head` has no default: since N5 the two
        // planes default to the exact head, and a forward that only said "mxfp4" when it differed
        // would leave the follower's choice to whichever default its argv parser has.
        cmd.arg("--head").arg(args.head.as_arg());
        // THE PARK STAGING BUFFER, because at TP=2 a park is TWO halves and the follower images
        // its own (ws-t2). A leader with a staging buffer and a follower without one is a park
        // that refuses at the first eviction — on the rank that cannot say so to the scheduler —
        // so the size crosses the argv with every other plan input. It is the same number on both
        // ranks: an image is this rank's own half either way and the halves are the same size.
        // The KV headroom crosses too (G2): the follower bills the same reserves out of it.
        cmd.arg("--kv-headroom-mib")
            .arg(args.kv_headroom_mib.to_string());
        if args.park_staging_mib > 0 {
            cmd.arg("--park-staging-mib")
                .arg(args.park_staging_mib.to_string())
                .arg("--park-staging-slots")
                .arg(args.park_staging_slots.to_string());
        }
        // AND THE IMAGE LAW WITH IT (ws-k14). Both halves of a session's image are placed by the
        // same arithmetic — `slot + committed - 1` for every `sidx`-addressed state — so a
        // follower on the other law would publish its half into a different slot of the run and
        // the arm would be measuring two defects at once. The negative control is a control on
        // BOTH cards or it is not one.
        if args.park_image_base {
            cmd.arg("--park-image-base");
        }
        // AND THE PROBE, because the question it was added for is a TWO-RANK one: the leader's
        // half of a park and the follower's are imaged by the same arithmetic over each rank's own
        // shard, and a dump that only rank 0 made could not tell them apart (ws-k15).
        if args.park_probe {
            cmd.arg("--park-probe");
        }
        // The drafter tier is the follower's too: its shards are half the size and its byte ledger
        // has to know, or rank 1 derives a KV pool the leader does not have.
        if args.draft_block > 0 {
            cmd.arg("--draft-block").arg(args.draft_block.to_string());
        }
        // AND THE PACK, for the same reason: a leader that made the drafter resident and a
        // follower that did not would agree on every shape and declare different node counts — the
        // aux tap is five launches on a decode plan and both ranks' graphs carry them. The
        // follower's flag has no value in `main.rs`'s `EngineWorker` arm, so it is passed here.
        if let Some(p) = &args.draft_pack {
            cmd.arg("--draft-pack").arg(p);
        }
        Ok(cmd)
    }

    /// Start rank 1 on `--peer-device`, pointed at the port rank 0 is about to listen on.
    fn spawn(args: &EngineArgs, port: u16) -> anyhow::Result<Worker> {
        let mut cmd = Self::command(args, port)?;
        let exe = std::path::PathBuf::from(cmd.get_program());
        // FN73: rank 1's stderr comes through the leader (forwarded line by line), so its own
        // `saber: FATAL rank 1:` line is the reason of the leader's fatal
        cmd.stderr(std::process::Stdio::piped());
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning rank 1 ({})", exe.display()))?;
        let watch = Some(saber_serve::fatal::RankWatch::start(
            child.id(),
            1,
            child.stderr.take(),
        ));
        println!(
            "tier           TP=2: rank 0 on HIP device {}, rank 1 (pid {}) on device {}, meeting on \
             127.0.0.1:{port}",
            args.device,
            child.id(),
            args.peer_device
        );
        Ok(Worker {
            child,
            port,
            device: args.peer_device,
            reaped: false,
            watch,
            asked_exit: None,
        })
    }

    /// Has it died on its own? A rank that is gone is a tier that cannot step, and the leader should
    /// say so rather than block on a socket.
    fn check(&mut self) -> anyhow::Result<()> {
        match self.child.try_wait().context("polling rank 1")? {
            None => Ok(()),
            Some(status) => bail!(
                "rank 1 (device {}) exited {status} before the tier was done; the leader cannot \
                 step alone",
                self.device
            ),
        }
    }

    /// Stop it, bounded at every step. Returns what it exited with, or the reason it had to be killed.
    fn stop(mut self) -> String {
        self.shutdown_within(WORKER_STOP, WORKER_KILL)
    }

    /// FN73b: [`Worker::stop`] on a path that FINISHED (the gate's end, the server's orderly stop):
    /// rank 1's exit after the `Stop` is checked too -- a non-zero code or a signal the leader did
    /// not cause is still rank 1's failure (the fatal, exit 86); 0 is the clean stop. The error
    /// paths keep [`Worker::stop`]: there the leader's own error is the one to report.
    fn stop_checked(mut self) -> String {
        let said = self.shutdown_within(WORKER_STOP, WORKER_KILL);
        if let (Some(w), Some(st)) = (self.watch.as_ref(), self.asked_exit) {
            w.reaped(1, st);
        }
        said
    }

    /// The bounded reap, with the two deadlines passed in so a test can run it in milliseconds.
    ///
    /// Three phases, none of them unbounded: wait for the exit the leader has already asked for,
    /// `SIGKILL` if it does not come, and wait for THAT. A worker that survives the third phase is
    /// a worker holding a card this process cannot free, which is the one case where continuing is
    /// worse than stopping — see [`Worker::drop`].
    fn shutdown_within(&mut self, stop: Duration, kill: Duration) -> String {
        if self.reaped {
            return "rank 1 was already reaped".to_owned();
        }
        self.reaped = true;
        // FN73b: disarm FIRST -- the reap below takes rank 1's zombie, after which the watch's
        // `/proc` poll sees the pid gone (`ExitSeen::Gone`) and, armed, would call the exit the
        // leader asked for a rank failure (P27-3b's standing gate: exit 86 after rank 1's 0).
        if let Some(w) = &self.watch {
            w.disarm();
        }
        // The leader's engine thread has already sent `Command::Stop` by the time this runs (the
        // engine's Drop joins that thread), so in the healthy case rank 1 is already on its way out
        // and this is the wait for it.
        let deadline = Instant::now() + stop;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    self.asked_exit = Some(status);
                    return format!("rank 1 exited {status}");
                }
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Ok(None) => break,
                Err(e) => return format!("rank 1 could not be waited on: {e}"),
            }
        }
        // FN73: the leader kills it now: that exit is not a rank failure (disarmed above)
        let _ = self.child.kill();
        let hard = Instant::now() + kill;
        while Instant::now() < hard {
            // `wait`, not just `kill`: a killed child that is never waited on is a zombie, and a
            // zombie still owns its pid. This is the reap.
            if let Ok(Some(status)) = self.child.try_wait() {
                return format!("rank 1 ignored the stop for {stop:?} and was killed ({status})");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        format!(
            "rank 1 (port {}) survived SIGKILL for {kill:?}: THE CARD IS STILL HELD",
            self.port
        )
    }
}

impl Drop for Worker {
    /// Reap rank 1 however this scope is leaving, and **abort** if it cannot be reaped.
    ///
    /// Aborting the leader because the follower will not die looks drastic until the alternative is
    /// written down: a leader that returns normally here leaves two processes mapped onto two cards,
    /// holding whatever lock the harness gave them, answering nothing. That is the state that cost
    /// 46 minutes of two idle cards on 2026-09-20, and it is invisible — the run had already printed
    /// its error and looked finished. A crash is louder than a deadlock and the kernel reaps what it
    /// leaves. `abort`, not `exit`: this process has a live HIP thread and unwinding through it is
    /// its own failure mode.
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        let said = self.shutdown_within(WORKER_STOP, WORKER_KILL);
        eprintln!("tier           {said} (reaped on the way out of a failing scope)");
        if said.contains("THE CARD IS STILL HELD") {
            eprintln!(
                "tier           rank 1 cannot be reaped; aborting rank 0 rather than leaving two \
                 cards held by a tier that answers nothing"
            );
            std::process::abort();
        }
    }
}

/// How long rank 1 may take to exit after being told to, before it is killed.
const WORKER_STOP: Duration = Duration::from_secs(60);
/// And how long after SIGKILL before the leader reports the card as still held.
const WORKER_KILL: Duration = Duration::from_secs(20);

/// A loopback port nothing is listening on.
///
/// Bind-and-drop: the same race `saber step-tp2` runs and has never lost, and the alternative — the
/// leader holding the listener and handing it to the engine thread — would put a `TcpListener` in
/// `EngineConfig`, which is a value the config is meant to be able to print.
fn free_port() -> anyhow::Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").context("finding a free loopback port")?;
    let p = l.local_addr().context("local addr")?.port();
    drop(l);
    Ok(p)
}

/// Start the tier: pick a port, spawn rank 1, and give back the args rank 0 should build with.
///
/// At `world == 1` this is nothing at all, which is what keeps one code path for both tiers.
fn start_tier(args: &EngineArgs) -> anyhow::Result<(EngineArgs, Option<Worker>)> {
    if args.world == 1 {
        return Ok((args.clone(), None));
    }
    let mut leader = args.clone();
    leader.peer_port = if args.peer_port == 0 {
        free_port()?
    } else {
        args.peer_port
    };
    leader.follower = false;
    leader.rank = 0;
    let worker = Worker::spawn(&leader, leader.peer_port)?;
    Ok((leader, Some(worker)))
}

/// Arguments of `saber engine-worker` — rank 1, and nothing else.
pub struct WorkerArgs {
    /// How to build the engine. `world` is 2 and `follower` is true, or this refuses.
    pub engine: EngineArgs,
}

/// Run rank 1 until the leader says stop.
///
/// Prints what it is and then nothing: its stdout is the leader's log too, so a follower that
/// chattered per step would interleave with the numbers being measured.
pub fn worker(args: &WorkerArgs) -> anyhow::Result<()> {
    if args.engine.world != 2 || !args.engine.follower {
        bail!("`saber engine-worker` is rank 1 of a TP=2 tier: pass --tp 2 and --rank 1");
    }
    let (facts, ledger) =
        match saber_engine::run_worker(args.engine.config(), args.engine.ledger(32)?) {
            Ok(x) => x,
            Err(e) => {
                // FN73: rank 1's failure is the pair's fatal (`main` exits 86 with the line; the
                // leader forwards it from this process's stderr)
                saber_serve::fatal::set(1, e.to_string());
                return Err(e.into());
            }
        };
    println!(
        "worker         rank {} done: {} weights, {} MiB of device buffers, {} collectives per \
         step, {} MiB peer-visible, vram(rank {}) {} MiB reserved",
        facts.rank,
        facts.weights,
        facts.device_bytes >> 20,
        facts.collectives,
        facts.ipc_bytes >> 20,
        facts.rank,
        ledger.used(Pool::Vram(facts.rank)) >> 20
    );
    // Rank 1's own budget and its own node count, printed rather than assumed. Both ranks assert
    // this for themselves at capture — `capture_step` refuses a graph whose nodes are not its shape's
    // declared budget — but a run that never says the number leaves a reader taking it on trust.
    for (shape, declared) in &facts.declared {
        let nodes = facts
            .captured
            .iter()
            .find(|(s, _)| s == shape)
            .map(|(_, n)| format!(", captured as {n} nodes"))
            .unwrap_or_default();
        println!(
            "worker         rank {} {shape}: {declared} dispatches declared{nodes}",
            facts.rank
        );
    }
    Ok(())
}

#[cfg(test)]
mod worker_tests {
    use super::{Worker, WORKER_KILL, WORKER_STOP};
    use std::time::{Duration, Instant};

    /// A `Worker` around a child this test owns, so the reap can be exercised without a card.
    fn wrap(child: std::process::Child) -> Worker {
        Worker {
            child,
            port: 0,
            device: 1,
            reaped: false,
            watch: None,
            asked_exit: None,
        }
    }

    /// A worker that will not take the hint is killed and REAPED, inside the deadline.
    ///
    /// The negative control is the bug this exists for: before 2026-09-20 the only reap was in
    /// `stop()`, so a gate that returned early left `Child`'s own `Drop` to run — which neither
    /// kills nor waits — and rank 1 stayed up holding a card, then stayed a zombie holding a pid.
    /// `sh -c 'trap "" TERM; sleep 600'` is a follower wedged past any polite request.
    #[test]
    fn a_wedged_worker_is_killed_and_reaped_within_the_deadline() {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg("trap '' TERM; sleep 600")
            .spawn()
            .expect("spawning a wedged worker");
        let pid = child.id();
        let mut w = wrap(child);
        let t0 = Instant::now();
        let said = w.shutdown_within(Duration::from_millis(50), Duration::from_secs(10));
        let took = t0.elapsed();
        assert!(
            said.contains("was killed"),
            "a worker that ignores the stop must be killed, not waited on forever: {said}"
        );
        assert!(
            took < Duration::from_secs(10),
            "the reap took {took:?}, which is not a deadline"
        );
        // REAPED, not merely killed: a killed child nobody waits on is a zombie, and a zombie still
        // owns its pid. `kill -0` succeeds on a zombie, so ask the child table instead — `try_wait`
        // has already returned the status once, which it can only do for a process it reaped.
        assert!(w.reaped, "the worker must be marked reaped");
        let _ = pid;
    }

    /// And the healthy path: a worker that exits on its own is waited on, not killed.
    #[test]
    fn a_worker_that_exits_is_waited_on() {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .spawn()
            .expect("spawning a worker that exits");
        let mut w = wrap(child);
        let said = w.shutdown_within(WORKER_STOP, WORKER_KILL);
        assert!(
            said.contains("exited"),
            "a worker that exits on its own is waited on: {said}"
        );
        assert!(!said.contains("killed"), "and is not killed: {said}");
    }

    /// FN73b: the reap the leader asked for is not a rank failure. Before FN73b the watch was
    /// armed through `shutdown_within`'s `try_wait`, which took the zombie inside its 10 ms poll;
    /// the watch's 50 ms `/proc` poll then found the pid gone (`ExitSeen::Gone`) and set the fatal
    /// (P27-3b's standing gate: exit 86 after rank 1's clean 0). The watch is disarmed at the head
    /// of the reap now, and the clean status is handed back.
    #[test]
    fn fn73b_a_worker_reaped_after_its_stop_is_not_a_fatal() {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg("sleep 0.2; exit 0")
            .spawn()
            .expect("spawning a worker that exits");
        let watch =
            saber_serve::fatal::RankWatch::start(child.id(), 1, None::<std::process::ChildStderr>);
        // the leader's reap races the watch exactly as the gate's did: `try_wait` every 10 ms
        let mut w = wrap(child);
        w.watch = Some(watch);
        let said = w.shutdown_within(WORKER_STOP, WORKER_KILL);
        assert!(said.contains("exited"), "{said}");
        assert!(w.asked_exit.is_some_and(|s| s.success()), "{said}");
        // give the watch several polls to (wrongly) see the pid gone
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            saber_serve::fatal::get().is_none(),
            "the exit the leader asked for set the fatal: {:?}",
            saber_serve::fatal::get()
        );
    }

    /// Reaping twice is not an error, because `Drop` runs after `stop()` on every healthy path.
    #[test]
    fn reaping_twice_is_a_no_op() {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .spawn()
            .expect("spawning");
        let mut w = wrap(child);
        let _ = w.shutdown_within(WORKER_STOP, WORKER_KILL);
        let again = w.shutdown_within(WORKER_STOP, WORKER_KILL);
        assert!(again.contains("already reaped"), "{again}");
    }
}

#[cfg(test)]
mod reference_tests {
    use super::{blocks_for, pages_for_live, read_reference, Reference};

    /// The fixture in a scratch directory under `target/`, removed when the returned guard
    /// drops -- never `/tmp` (FN51h).
    fn write(body: &str) -> (saber_test_scratch::ScratchDir, std::path::PathBuf) {
        let d = saber_test_scratch::ScratchDir::new("saber_ref");
        let p = d.join(format!("saber_ref_{}.jsonl", body.len()));
        std::fs::write(&p, body).expect("writing the fixture");
        (d, p)
    }

    /// The margin is top-1 minus top-2 out of the recorded logprobs, in nats.
    #[test]
    fn the_margin_is_the_gap_between_the_top_two() {
        let (_d, p) = write(
            r#"{"id":0,"n_prompt":234,"token_id":9764,"text":"Let","logprob":-0.08,"top5":{"Let":-0.08,"The":-2.83,"N":-3.96}}
"#,
        );
        let m = read_reference(&p).expect("reading");
        let r = &m[&0];
        assert_eq!(r.token_id, 9764);
        assert_eq!(r.n_prompt, 234);
        assert!((r.margin - 2.75).abs() < 1e-6, "margin was {}", r.margin);
        assert!(!r.is_special());
        let _ = std::fs::remove_file(&p);
    }

    /// A control token as prod's answer is a prompt the identity arms must not be run on.
    ///
    /// The negative control for the guard: `248046` is `<|im_end|>`, and a gate that accepted this
    /// row would spend its whole run comparing two continuations of a turn that already ended.
    #[test]
    fn a_control_token_reference_is_recognised() {
        let (_d, p) = write(
            r#"{"id":7,"n_prompt":10,"token_id":248046,"text":"<|im_end|>","logprob":-0.01,"top5":{"a":-0.01,"b":-9.0}}
"#,
        );
        let m = read_reference(&p).expect("reading");
        assert!(m[&7].is_special(), "248046 is <|im_end|>");
        // And the boundary is where the tokenizer's text ends, not one either side of it.
        assert_eq!(Reference::FIRST_SPECIAL, 248_044);
        let _ = std::fs::remove_file(&p);
    }

    /// H44: the TP=1 pool of 1,371 blocks cannot hold the sweep's 8 x 257 pages; the sweep's
    /// sequences take an equal share (171, every block in range) instead of being handed block
    /// 1,371 at `seq305`. TP=2's 57,697 blocks keep 257 each (unchanged pages), and a share below
    /// the arm's own need is refused by name. Control: the old allocation overruns at i = 5.
    #[test]
    fn the_sweep_pages_fit_the_tp1_pool() {
        let (tp1, tp2, per_seq, need) = (1371u32, 57_697u32, 257u32, 67u32);
        let old_last = blocks_for(7, per_seq);
        assert!(
            old_last.iter().any(|&b| b >= tp1),
            "control: 8 x 257 overruns 1,371"
        );
        assert_eq!(
            blocks_for(5, per_seq).iter().find(|&&b| b >= tp1),
            Some(&1371)
        );
        let share = pages_for_live(8, per_seq, tp1, need).expect("fits");
        assert_eq!(share, 171);
        assert!(blocks_for(7, share).iter().all(|&b| b < tp1));
        assert_eq!(pages_for_live(8, per_seq, tp2, need), Ok(per_seq));
        assert_eq!(pages_for_live(1, per_seq, tp1, need), Ok(per_seq));
        let err = pages_for_live(8, per_seq, 400, need).expect_err("50 < 67 pages");
        assert!(err.contains("need 67 page(s)"), "{err}");
    }

    /// A row whose runner-up nobody recorded is infinitely confident, not tied.
    ///
    /// Zero would be the easy default and it is the wrong one: it would refuse the row for being
    /// too close when what is actually true is that nothing was close enough to record.
    #[test]
    fn a_single_candidate_is_not_a_tie() {
        let (_d, p) = write(
            r#"{"id":1,"n_prompt":5,"token_id":42,"text":"x","logprob":-0.0,"top5":{"x":-0.0}}
"#,
        );
        let m = read_reference(&p).expect("reading");
        assert!(m[&1].margin.is_infinite(), "margin was {}", m[&1].margin);
        let _ = std::fs::remove_file(&p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CMP27G: under the tree default the engine plans `(1, 10)` beside the chain's widths; the
    /// width-controller arm moves between the CHAIN's b1 widths only (the seq700 failure took 10
    /// as "wide" and asked the tree form's chain plan for 9 drafts). Control: without the cap the
    /// tree's shape is picked up, exactly as the failing gate did.
    #[test]
    fn cmp27g_the_width_arm_moves_between_chain_widths_only() {
        let shapes = [(1, 8), (2, 8), (4, 8), (8, 8), (1, 10)];
        assert_eq!(chain_b1_widths(&shapes, 8), vec![8]);
        let ladder = [(1, 1), (1, 5), (1, 8), (8, 8), (1, 10)];
        assert_eq!(chain_b1_widths(&ladder, 8), vec![1, 5, 8]);
        // the control: no cap = the pre-fix list, whose widest entry is the tree form's
        assert_eq!(chain_b1_widths(&shapes, u32::MAX), vec![8, 10]);
        // and that width is what the tree form's chain plan refuses (a 9-token chain)
        let draft: Vec<u32> = (0..9).collect();
        let e = saber_engine::spec_tree_step::draft_chain_plan(1, &draft).unwrap_err();
        assert!(e.to_string().contains("node 7 at depth 8"), "{e}");
    }

    /// The fixtures `--messages` needs. Gitignored, and a fresh worktree may not have them --
    /// which is a SKIP with a reason, never a silent pass.
    fn fixtures() -> Option<(PathBuf, PathBuf)> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join(".cache");
        let tok = root.join("template_corpus.tokenizer.json");
        let tpl = root.join("template_corpus.jinja");
        (tok.exists() && tpl.exists()).then_some((tok, tpl))
    }

    /// The host half of `--messages`: a transcript becomes ids through the SERVE PLANE's renderer
    /// and tokenizer, in the order the turns were FIRST served — which is the transcript's own
    /// order reversed, because it records its rows as it re-issues them.
    ///
    /// Pre-flighted here because the alternative is finding out on a shared card: the read, the
    /// render and the encode are all pure host work, and the only thing the GPU adds is the model.
    #[test]
    fn a_transcript_becomes_its_turns_in_the_order_they_were_first_served() {
        let Some((tok, tpl)) = fixtures() else {
            eprintln!(
                "skipped: .cache/template_corpus.{{tokenizer.json,jinja}} are not in this worktree"
            );
            return;
        };
        // Under `target/`, removed on drop -- never `/tmp` (FN51h).
        let dir = saber_test_scratch::ScratchDir::new("saber-msg");
        let path = dir.join("transcript.jsonl");
        // As `multiturn_continuity.py` writes them: pass B's order, which is the LAST turn served
        // first. The three-message turn here is the one that differed.
        std::fs::write(
            &path,
            concat!(
                r#"{"conv":2,"turn":3,"same":false,"messages":[{"role":"user","content":"a"},"#,
                r#"{"role":"assistant","content":"b"},{"role":"user","content":"c"}]}"#,
                "\n",
                r#"{"conv":1,"turn":1,"same":true,"messages":[{"role":"user","content":"hello"}]}"#,
                "\n",
            ),
        )
        .expect("writing the transcript");
        let turns = read_message_turns(&path, &tok, &tpl).expect("reading the transcript");
        assert_eq!(turns.len(), 2, "both turns");
        // REVERSED: the row written last is the request served first.
        assert!(turns[0].label.contains("conv 1"), "{}", turns[0].label);
        assert!(!turns[0].differed, "{}", turns[0].label);
        assert!(turns[1].label.contains("conv 2"), "{}", turns[1].label);
        assert!(turns[1].differed, "{}", turns[1].label);
        assert!(turns[1].label.contains("DIFFERED"), "{}", turns[1].label);
        // The three-message turn renders longer than the one-message turn, which is the cheapest
        // check that the HISTORY reached the template rather than only the last user message.
        assert!(
            turns[1].ids.len() > turns[0].ids.len(),
            "a three-message history rendered to {} ids and a one-message turn to {}",
            turns[1].ids.len(),
            turns[0].ids.len()
        );
        assert!(
            turns[0].ids.len() > 4,
            "a rendered turn is more than its text"
        );
        // Deterministic: the same file twice is the same ids.
        let again = read_message_turns(&path, &tok, &tpl).expect("the second read");
        assert_eq!(turns[0].ids, again[0].ids);
        assert_eq!(turns[1].ids, again[1].ids);

        // NEGATIVE CONTROL: a file that is not a transcript is refused by name, not read as an
        // empty list of turns -- which would run the gate on the corpus and report a pass.
        let bad = dir.join("not-a-transcript.jsonl");
        std::fs::write(&bad, "{\"token_ids\":[1,2,3]}\n").expect("writing the control");
        let err = read_message_turns(&bad, &tok, &tpl).expect_err("no `messages` array");
        assert!(format!("{err}").contains("messages"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod s5_tests {
    use saber_engine::DraftCounts;

    use super::{counts_delta, DraftStep, PrefixVerdict};

    fn step(proposed: &[&[u32]], committed: &[&[u32]], accepted: &[u32]) -> DraftStep {
        DraftStep {
            proposed: proposed.iter().map(|b| b.to_vec()).collect(),
            committed: committed.iter().map(|b| b.to_vec()).collect(),
            accepted: accepted.to_vec(),
            counts: DraftCounts {
                proposals: 1,
                tokens: proposed.iter().map(|b| b.len() as u64).sum(),
                accepted: u64::from(accepted.iter().sum::<u32>()),
                verified: proposed.iter().map(|b| b.len() as u64).sum(),
                ..DraftCounts::default()
            },
        }
    }

    /// **S4's open item 3, as a law.** Two paths that ran the same number of STEPS under
    /// speculation produce different numbers of TOKENS the moment their acceptance differs, so a
    /// whole-list comparison names an index at which both lists hold the same id. The verdict is
    /// the common prefix, with both lengths, and the two failures are different failures.
    #[test]
    fn a_fixed_step_count_is_a_variable_token_count_and_the_verdict_says_which() {
        // Same tokens, one path just got further: the prefix agrees and the LENGTHS do not.
        let v = PrefixVerdict::of(&[1, 2, 3], &[1, 2, 3, 4, 5]);
        assert!(v.prefix_agrees());
        assert!(!v.lengths_agree());
        assert_eq!(v.common, 3);
        assert_eq!((v.len_a, v.len_b), (3, 5));
        assert!(v.line().contains("3 and 5"), "{}", v.line());

        // A REAL disagreement inside the prefix is a different verdict and is not hidden by the
        // lengths agreeing.
        let v = PrefixVerdict::of(&[1, 2, 9], &[1, 2, 3]);
        assert!(!v.prefix_agrees());
        assert!(v.lengths_agree());
        assert_eq!(v.first_diff, Some(2));

        // And the control for the whole idea: the OLD comparison (`a == b`) would have called
        // the first case a disagreement at index 3, where one list has nothing at all.
        assert_ne!([1u32, 2, 3].as_slice(), [1u32, 2, 3, 4, 5].as_slice());

        // Two empty lists agree, and so do two identical ones.
        assert!(PrefixVerdict::of(&[], &[]).prefix_agrees());
        let v = PrefixVerdict::of(&[7, 8], &[7, 8]);
        assert!(v.prefix_agrees() && v.lengths_agree());
    }

    /// A counter delta is a difference, and every field of it is.
    #[test]
    fn a_counter_delta_is_per_field() {
        let a = DraftCounts {
            proposals: 3,
            tokens: 21,
            accepted: 4,
            verified: 21,
            pages_cleared: 2,
            ..DraftCounts::default()
        };
        let b = DraftCounts {
            proposals: 5,
            tokens: 35,
            accepted: 9,
            verified: 35,
            pages_cleared: 2,
            pages_dirty: 1,
            ..DraftCounts::default()
        };
        let d = counts_delta(a, b);
        assert_eq!(d.proposals, 2);
        assert_eq!(d.tokens, 14);
        assert_eq!(d.accepted, 5);
        assert_eq!(d.verified, 14);
        assert_eq!(d.pages_cleared, 0);
        assert_eq!(d.pages_dirty, 1);
    }

    /// **The instrument's own property**: two traces that differ in the PROPOSAL differ, and the
    /// difference is at the step it happened on — not at the step the ids finally moved.
    ///
    /// The control is the pair that differs only in the number of tokens COMMITTED: that is the
    /// same defect seen through the ids, and it must be visible here as well, at its own step.
    #[test]
    fn a_draft_trace_names_the_step_a_proposal_moved() {
        let a = vec![
            step(&[&[11, 12]], &[&[11]], &[1]),
            step(&[&[13, 14]], &[&[13]], &[1]),
        ];
        let mut b = a.clone();
        assert_eq!(a, b, "two identical traces must compare equal");
        b[1].proposed[0][1] = 99;
        assert_ne!(a[1], b[1]);
        assert_eq!(
            a.iter().zip(&b).position(|(x, y)| x.proposed != y.proposed),
            Some(1),
            "the first differing proposal is step 1"
        );
        // ...and step 0 still agrees, which is what makes the answer an attribution rather than a
        // report that something, somewhere, moved.
        assert_eq!(a[0], b[0]);

        let mut c = a.clone();
        c[0].committed[0].push(12);
        c[0].accepted[0] = 1;
        assert_eq!(
            a.iter()
                .zip(&c)
                .position(|(x, y)| x.committed != y.committed),
            Some(0)
        );
    }
}

#[cfg(test)]
mod accept_ids_tests {
    use super::*;

    /// A capture of 64 tokens at width 8 is the prefill's one token plus 64 decode tokens, and
    /// the last step may overshoot by up to `width - 1`.
    #[test]
    fn a_captured_row_is_the_prefill_token_plus_the_token_budget() {
        for n in 65..=72 {
            let ids: Vec<TokenId> = (0..n).collect();
            assert!(
                check_committed_ids(&ids, 1, 64, 8).is_ok(),
                "{n} committed ids must be accepted at tokens 64, width 8"
            );
        }
    }

    /// CONTROL 1 — ws-f36's slice, which dropped committed token 0. It is one token short and
    /// the capture it produced had to be read with `--prod-offset 1`.
    #[test]
    fn negative_control_the_slice_that_dropped_the_prefills_token_is_refused() {
        let ids: Vec<TokenId> = (0..64).collect();
        let why = check_committed_ids(&ids, 1, 64, 8).expect_err("64 ids is one short of 65");
        assert!(why.contains("64 committed id(s)"), "{why}");
    }

    /// CONTROL 2 — the slice that replaced it, which took `d.ids[p.len()..]` on a vector that
    /// never held the prompt: empty for every prompt longer than the token budget.
    #[test]
    fn negative_control_the_slice_that_took_the_prompts_length_is_refused() {
        let why = check_committed_ids(&[], 1, 64, 8).expect_err("an empty row is not a capture");
        assert!(why.contains("0 committed id(s)"), "{why}");
    }

    /// CONTROL 3 — a row longer than one overshoot, which is the shape a capture that appended
    /// two sequences' tokens would have.
    #[test]
    fn negative_control_a_row_longer_than_one_overshoot_is_refused() {
        let ids: Vec<TokenId> = (0..73).collect();
        check_committed_ids(&ids, 1, 64, 8).expect_err("73 ids is past the width-8 overshoot");
    }

    /// And the prefill's own contribution is asserted, because index 0 of the capture is it.
    #[test]
    fn negative_control_a_prefill_that_committed_two_tokens_is_refused() {
        let ids: Vec<TokenId> = (0..66).collect();
        let why = check_committed_ids(&ids, 2, 64, 8).expect_err("the prefill samples one token");
        assert!(why.contains("not 1"), "{why}");
    }
}

#[cfg(test)]
mod image_diff_tests {
    use saber_engine::ParkImage;
    use saber_mem::park::{ImageDigests, RegionKind, RegionSpec};

    use super::image_diff;

    /// A two-layer drafter's image: one KV region, one GDN region, then `k.l0 k.l1 v.l0 v.l1`.
    fn image() -> ParkImage {
        let lens = [(RegionKind::Kv, 64u64), (RegionKind::GdnState, 32)];
        let mut layout: Vec<RegionSpec> = lens
            .iter()
            .map(|(kind, len)| RegionSpec {
                kind: *kind,
                len: *len,
            })
            .collect();
        for _ in 0..4 {
            layout.push(RegionSpec {
                kind: RegionKind::DraftKv,
                len: 16,
            });
        }
        let bytes: u64 = layout.iter().map(|r| r.len).sum();
        ParkImage {
            digests: ImageDigests {
                layout,
                bytes,
                chunk_bytes: 4 << 20,
                chunks: vec![[3u8; 32]],
                sha256: [4u8; 32],
            },
            resident: 40,
            committed: 4,
            bytes: (0..bytes as u32).map(|i| (i % 251) as u8).collect(),
        }
    }

    /// **THE INSTRUMENT'S OWN CONTROL** (`CLAUDE.md` rule 8, ws-k15).
    ///
    /// The park arm's probe claims that a restore which put every region back makes the next image
    /// byte-identical, and reports the first byte that is not. The control is the byte: a single
    /// flip inside the drafter's THIRD region has to be named as the `v` cache's layer 0 — the
    /// drafter's regions are `k` layers then `v` layers — at its own offset, and the clean pair
    /// beside it has to report nothing.
    #[test]
    fn a_flipped_byte_is_named_by_the_region_it_lands_in() {
        let a = image().encode();
        assert_eq!(image_diff(&a, &a), None, "two copies of one image differ");

        let mut img = image();
        // Region 4 is the third `draft_kv` region = `v`, layer 0; byte 5 of its 16.
        let at = 64 + 32 + 16 * 2 + 5;
        img.bytes[at] ^= 0xFF;
        let line = image_diff(&a, &img.encode()).expect("a flipped byte is a difference");
        assert!(line.contains(&format!("at byte {at}")), "{line}");
        assert!(line.contains("v cache, layer 0"), "{line}");
        assert!(line.contains("byte 5 of its 16-byte run"), "{line}");

        // The counts are part of the claim: a restore that put the bytes back and the COUNTS
        // wrong is a session that reads the right pages from the wrong slot of its run.
        let mut img = image();
        img.committed += 1;
        let line = image_diff(&a, &img.encode()).expect("a moved `committed` is a difference");
        assert!(line.contains("committed"), "{line}");

        // And a shape that does not match is named as a shape, not as a byte.
        let mut img = image();
        img.digests.layout.pop();
        img.digests.bytes -= 16;
        img.bytes.truncate(img.digests.bytes as usize);
        let line = image_diff(&a, &img.encode()).expect("a different shape is a difference");
        assert!(line.contains("different shapes"), "{line}");
    }
}

#[cfg(test)]
mod ar_band_tests {
    use super::EngineArgs;
    use clap::Parser as _;
    use saber_engine::head_share::TargetHead;
    use saber_kernels::saber::{self, SaberRules};

    /// The flag as the two planes' own argv parsers see it.
    ///
    /// On its OWN thread with a 32 MiB stack: `Cmd` is a large enum (every subcommand's arguments
    /// in one union) and clap builds it by value, which overflows a test thread's default 2 MiB.
    /// `main` never sees this because a process's main thread gets 8 MiB and the binary is built
    /// with optimisations.
    fn parse(argv: &[String]) -> bool {
        let argv = argv.to_vec();
        std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn(move || match crate::Cli::parse_from(&argv).cmd {
                crate::Cmd::Serve { ar_exact, .. } | crate::Cmd::Engine { ar_exact, .. } => {
                    ar_exact
                }
                _ => panic!("neither a `serve` nor an `engine` command line"),
            })
            .expect("spawning the parser thread")
            .join()
            .expect("the argv parser panicked")
    }

    const SERVE: &[&str] = &[
        "saber",
        "serve",
        "--engine",
        "real",
        "--engine-tp",
        "2",
        "--pack",
        "/packs/tier",
    ];
    const ENGINE: &[&str] = &[
        "saber",
        "engine",
        "--tp",
        "2",
        "--pack",
        "/packs/tier",
        "--prompt-ids",
        "/cache/template_corpus.jsonl",
    ];

    fn with(base: &[&str], extra: &[&str]) -> Vec<String> {
        base.iter().chain(extra).map(|s| (*s).to_owned()).collect()
    }

    fn args_at(ar_exact: bool) -> EngineArgs {
        EngineArgs {
            vision_max_patches: 0,
            vision_lazy: false,
            drafter_rope_index: false,
            pack: std::path::PathBuf::from("/packs/tier"),
            world: 2,
            rank: 0,
            peer_port: 0,
            follower: false,
            peer_device: 1,
            prebuilt: std::path::PathBuf::from("prebuilt/gfx1201"),
            hip_lib: saber_hip::DEFAULT_HIP_SONAME.to_owned(),
            rccl_lib: None,
            host_ar: false,
            dspine_socket: None,
            draft_capture_dir: None,
            draft_capture_layers: Vec::new(),
            device: 0,
            budget_mib: 26000,
            max_seqs: 2,
            max_ctx: 4096,
            prefill_rows: vec![2048],
            decode_batches: vec![1],
            decode_widths: vec![8],
            decode_bands: saber_sched::DEFAULT_ROW_BANDS.to_vec(),
            ragged_ceiling: saber_sched::RaggedCeiling::Table,
            width_rule: saber_sched::WidthRule::default(),
            mixed_prefill: saber_sched::MixedPrefill::OFF,
            prefill_fill_above: 0,
            deadline_s: 120,
            capture: true,
            device_feed: true,
            live_row_attn: true,
            park_staging_mib: 0,
            park_ram_gib: 0.0,
            park_staging_slots: 1,
            park_image_base: false,
            park_probe: false,
            kv_pool_blocks: 0,
            kv_headroom_mib: 1024,
            draft_block: 0,
            draft_pack: None,
            draft_replay: None,
            draft_seq: None,
            draft_seq_out: None,
            draft_chain_dump: None,
            ar_exact,
            draft_batch_check: false,
            draft_store_narrow: false,
            draft_store_natural_off: false,
            draft_store_check: false,
            draft_store_deferred: false,
            draft_kv_paged: false,
            draft_embed_share: false,
            selector_rank: saber_engine::selector_rank::SelectorRank::Walk,
            draft_full_vocab: false,
            spec_tree: saber_engine::spec_tree::SpecTreeForm::Off,
            spec_tree_sampled: false,
            spec_tree_sampled_control: false,
            gdn_replay: false,
            host_levers: saber_engine::host_levers::HostLevers::default(),
            device_step: saber_engine::device_step::DeviceStep::default(),
            prefill_ar_overlap: 0,
            prefill_ar_push_blocks: 0,
            small_chunk_gemm: 0,
            small_chunk_band: false,
            shared_prefill_arena: false,
            shared_head_arena: false,
            mid_chunk_gemm: false,
            mid_chunk_atiled: false,
            skinny_gate: false,
            decode_pf: false,
            head_m64: false,
            drafter_gateup_m64: 0,
            ar_wpar: false,
            ar_fuse: false,
            b1_arms: Default::default(),
            c4_arms: Default::default(),
            step_arms: Default::default(),
            mixed_class_proxy: false,
            decode_fusion: saber_engine::device::DecodeFusion::All,
            host_trace: false,
            whole_step_frame: false,
            frame_digest: false,
            race_check: false,
            race_shadow: false,
            poison_boot: None,
            constant_seed_control: false,
            gather_per_row: false,
            propose_in_step: false,
            fixed_step_wait: false,
            fail_after_steps: 0,
            host_wait: crate::engine::DEFAULT_HOST_WAIT_27B,
            gdn_state_f16: true,
            head: saber_engine::head_share::TargetHead::default(),
        }
    }

    /// **ws-q1's open item 3.** `saber serve` passed `ar_exact: false` whatever an operator asked,
    /// while `saber engine` took `--ar-exact` — so the two planes could not be made to resolve the
    /// same all-reduce family for the same shape BY CONFIGURATION, and every serve-side number was
    /// in the lossy band at the widths whose messages cross it. They take the same flag now, and
    /// it reaches `EngineConfig` through the one constructor both planes use.
    #[test]
    fn the_serve_plane_and_the_engine_plane_take_the_same_all_reduce_flag() {
        assert!(!parse(&with(SERVE, &[])), "serve defaults to prod's band");
        assert!(!parse(&with(ENGINE, &[])), "engine defaults to prod's band");
        assert_eq!(
            parse(&with(SERVE, &["--ar-exact"])),
            parse(&with(ENGINE, &["--ar-exact"])),
            "the two planes disagree about the band the same flag asks for"
        );
        assert!(parse(&with(SERVE, &["--ar-exact"])));

        // Not cosmetic: it is the one field `EngineArgs::config` carries into the `EngineConfig`
        // the engine derives the band from, and rank 1 is spawned from the same `EngineArgs`.
        assert!(!args_at(false).config().ar_exact);
        assert!(args_at(true).config().ar_exact);
    }

    /// The property the flag exists for, checked on the host: the family a shape resolves to is
    /// decided by ONE number in the rules table, so two planes carrying the same `ar_exact`
    /// resolve the same SYMBOL for the same message.
    ///
    /// Its control is the half that must fail: at the default band a serve-sized prefill message
    /// resolves the LOSSY family, which is the band every ws-q1 serve number was taken in. If the
    /// two bands resolved the same kernel the flag would prove nothing.
    #[test]
    fn one_number_decides_the_family_and_both_planes_read_the_same_one() {
        let hidden = 5120u64; // Qwen3.8-27B's hidden size
        let rows = 2048u64; // the serve plane's largest prefill chunk
        let msg_bytes = rows * hidden * 2;

        let family = |rules: &SaberRules| -> String {
            saber::resolve(
                saber::Op::AllReduce {
                    dtype: saber::ArPayload::Bf16,
                },
                saber::Shape::Ar(saber::ArShape {
                    n_elem: rows * hidden,
                    elem_bytes: 2,
                    world_size: 2,
                }),
                rules,
            )
            .expect("resolving a 2048-row all-reduce")
            .kernel
        };

        let prod = SaberRules::prod_saber().expect("prod saber rules");
        let mut exact_rules = SaberRules::prod_saber().expect("prod saber rules");
        exact_rules.ar.quant_min_bytes = msg_bytes + 1;

        let lossy = family(&prod);
        let exact = family(&exact_rules);
        assert_ne!(
            lossy, exact,
            "a {msg_bytes} B message resolves the same kernel on both sides of the band: the \
             control cannot fail, so the flag would prove nothing"
        );
        assert!(exact.contains("exact"), "{exact}");
        assert!(!lossy.contains("exact"), "{lossy}");
        // One band, both planes: the serve plane's rules and the engine plane's rules are the same
        // table read twice, so the same shape resolves the same symbol on both.
        assert_eq!(family(&exact_rules), family(&exact_rules));
        assert_eq!(family(&prod), lossy);
    }

    /// The head both planes resolve from their own argv (N5), on a thread with a big stack for the
    /// reason `parse` gives.
    fn parse_head(argv: &[String]) -> anyhow::Result<TargetHead> {
        let argv = argv.to_vec();
        std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn(move || match crate::Cli::try_parse_from(&argv)?.cmd {
                crate::Cmd::Serve {
                    mxfp4_head,
                    engine_fp8_head,
                    engine_bf16_head,
                    pack,
                    ..
                } => crate::target_head(
                    mxfp4_head,
                    &[
                        (engine_fp8_head, "--engine-fp8-head"),
                        (engine_bf16_head, "--engine-bf16-head"),
                    ],
                    &pack.unwrap_or_default(),
                ),
                crate::Cmd::Engine {
                    mxfp4_head,
                    fp8_head,
                    bf16_head,
                    pack,
                    ..
                } => crate::target_head(
                    mxfp4_head,
                    &[(fp8_head, "--fp8-head"), (bf16_head, "--bf16-head")],
                    &pack,
                ),
                crate::Cmd::EngineWorker { head, .. } => Ok(head),
                _ => anyhow::bail!("not a serve / engine / engine-worker command line"),
            })
            .expect("spawning the parser thread")
            .join()
            .expect("the argv parser panicked")
    }

    /// G2: rank 1's argv, as `Worker::command` builds it and `Cli` parses it back, carries the
    /// leader's KV headroom. Until G2 the follower was pinned at 1024 MiB whatever the leader was
    /// given, and at the 8192-row bucket its all-reduce IPC scratch (672 MiB) did not fit what was
    /// left, so every such boot refused on rank 1 alone. The default (1024) is the control.
    #[test]
    fn rank_1_is_told_the_kv_headroom() {
        for mib in [1024u64, 2048, 3000] {
            let mut args = args_at(false);
            args.kv_headroom_mib = mib;
            let cmd = super::Worker::command(&args, 4242).expect("rank 1's command");
            let argv: Vec<String> = std::iter::once("saber".to_owned())
                .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
                .collect();
            let got = std::thread::Builder::new()
                .stack_size(32 << 20)
                .spawn(
                    move || match crate::Cli::try_parse_from(&argv).expect("parses").cmd {
                        crate::Cmd::EngineWorker {
                            kv_headroom_mib, ..
                        } => kv_headroom_mib,
                        _ => panic!("not an engine-worker command line"),
                    },
                )
                .expect("spawning the parser thread")
                .join()
                .expect("the argv parser panicked");
            assert_eq!(got, mib, "rank 1 dropped the KV headroom");
        }
    }

    /// **N5/N6: THE DEFAULT IS THE EXACT HEAD** on both planes (resolved against the pack when the
    /// pack is on the host — this argv's `/packs/tier` is not, so it stays `Exact` here and the
    /// real packs' resolution is `head_share`'s test); `--mxfp4-head` is the control; the old
    /// head flags parse and change nothing (N6: the fp8 one too); `--mxfp4-head` beside any of them
    /// is a refusal. Then the forward: rank 1's argv, as `Worker::command` builds it and `Cli`
    /// parses it back, carries the leader's head at EVERY value — the resolved kinds included.
    #[test]
    fn the_exact_head_is_the_default_and_rank_1_is_told_which_head() {
        for base in [SERVE, ENGINE] {
            assert_eq!(parse_head(&with(base, &[])).unwrap(), TargetHead::Exact);
            assert_eq!(
                parse_head(&with(base, &["--mxfp4-head"])).unwrap(),
                TargetHead::Mxfp4
            );
        }
        assert_eq!(
            parse_head(&with(SERVE, &["--engine-bf16-head"])).unwrap(),
            TargetHead::Exact
        );
        assert_eq!(
            parse_head(&with(SERVE, &["--engine-mxfp4-head"])).unwrap(),
            TargetHead::Mxfp4
        );
        assert_eq!(
            parse_head(&with(ENGINE, &["--bf16-head"])).unwrap(),
            TargetHead::Exact
        );
        let e = parse_head(&with(ENGINE, &["--bf16-head", "--mxfp4-head"])).unwrap_err();
        assert!(e.to_string().contains("contradict"), "{e}");
        let e = parse_head(&with(SERVE, &["--engine-bf16-head", "--mxfp4-head"])).unwrap_err();
        assert!(e.to_string().contains("contradict"), "{e}");
        // N6: the fp8 flags are no-ops too (the default resolves to fp8 on the fp8 pack).
        assert_eq!(
            parse_head(&with(ENGINE, &["--fp8-head"])).unwrap(),
            TargetHead::Exact
        );
        assert_eq!(
            parse_head(&with(SERVE, &["--engine-fp8-head"])).unwrap(),
            TargetHead::Exact
        );
        assert_eq!(
            parse_head(&with(ENGINE, &["--fp8-head", "--bf16-head"])).unwrap(),
            TargetHead::Exact
        );
        let e = parse_head(&with(ENGINE, &["--fp8-head", "--mxfp4-head"])).unwrap_err();
        assert!(e.to_string().contains("--fp8-head"), "{e}");

        for head in [
            TargetHead::Exact,
            TargetHead::Bf16,
            TargetHead::Fp8,
            TargetHead::Mxfp4,
        ] {
            let mut args = args_at(false);
            args.head = head;
            assert_eq!(
                args.config().head,
                head,
                "EngineArgs::config drops the head"
            );
            let cmd = super::Worker::command(&args, 4242).expect("rank 1's command");
            let argv: Vec<String> = std::iter::once("saber".to_owned())
                .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
                .collect();
            assert_eq!(
                parse_head(&argv).expect("rank 1's argv parses"),
                head,
                "{argv:?}"
            );
        }
        // The worker has no default: an argv without `--head` does not parse (the control).
        let cmd = super::Worker::command(&args_at(false), 4242).expect("rank 1's command");
        let mut argv: Vec<String> = std::iter::once("saber".to_owned())
            .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
            .collect();
        let at = argv
            .iter()
            .position(|a| a == "--head")
            .expect("--head forwarded");
        argv.drain(at..at + 2);
        assert!(parse_head(&argv).is_err(), "a follower with no head parsed");
    }

    /// H36: the prefill all-reduce overlap is a PROTOCOL fact -- a leader that slices a chunk's
    /// reduces beside a follower that issues them whole is two reduce sequences on one protocol, and
    /// it hangs. So it crosses rank 1's argv with every other plan input, both planes take it, and
    /// the default is off (the control: an argv without it parses to 0).
    #[test]
    fn the_prefill_overlap_reaches_both_planes_and_rank_one() {
        fn overlap_of(argv: &[String]) -> u32 {
            let argv = argv.to_vec();
            std::thread::Builder::new()
                .stack_size(32 << 20)
                .spawn(move || match crate::Cli::parse_from(&argv).cmd {
                    crate::Cmd::Serve {
                        engine_prefill_ar_overlap,
                        ..
                    } => engine_prefill_ar_overlap,
                    crate::Cmd::Engine {
                        prefill_ar_overlap, ..
                    }
                    | crate::Cmd::EngineWorker {
                        prefill_ar_overlap, ..
                    } => prefill_ar_overlap,
                    _ => panic!("not a serve / engine / engine-worker command line"),
                })
                .expect("spawning the parser thread")
                .join()
                .expect("the argv parser panicked")
        }
        assert_eq!(overlap_of(&with(SERVE, &[])), 0, "serve defaults to off");
        assert_eq!(overlap_of(&with(ENGINE, &[])), 0, "engine defaults to off");
        assert_eq!(
            overlap_of(&with(SERVE, &["--engine-prefill-ar-overlap", "1024"])),
            1024
        );
        assert_eq!(
            overlap_of(&with(ENGINE, &["--prefill-ar-overlap", "2048"])),
            2048
        );
        let mut args = args_at(false);
        args.prefill_ar_overlap = 1024;
        assert_eq!(args.config().prefill_ar_overlap, 1024, "config drops it");
        args.prefill_ar_push_blocks = 16;
        assert_eq!(
            args.config().prefill_ar_push_blocks,
            16,
            "config drops the form"
        );
        let argv = |args: &EngineArgs| -> Vec<String> {
            let cmd = super::Worker::command(args, 4242).expect("rank 1's command");
            std::iter::once("saber".to_owned())
                .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
                .collect()
        };
        assert_eq!(overlap_of(&argv(&args)), 1024, "rank 1 was not told");
        assert_eq!(overlap_of(&argv(&args_at(false))), 0, "the control");
    }

    /// P27-1d: the reduce's FORM inside the overlap crosses rank 1's argv too -- a leader pushing
    /// split beside a follower running the one-shot (or pushing with another N) is two protocols.
    /// Both planes take it; the default is the one-shot (the control: an argv without it parses
    /// to 0), and `--ar-form split` alone takes P27-1c's 16 blocks.
    #[test]
    fn the_split_form_reaches_both_planes_and_rank_one() {
        fn push_of(argv: &[String]) -> u32 {
            let argv = argv.to_vec();
            std::thread::Builder::new()
                .stack_size(32 << 20)
                .spawn(move || match crate::Cli::parse_from(&argv).cmd {
                    crate::Cmd::Serve {
                        engine_ar_form,
                        engine_ar_push_blocks,
                        ..
                    } => crate::ar_push_blocks_of(&engine_ar_form, engine_ar_push_blocks),
                    crate::Cmd::Engine {
                        ar_form,
                        ar_push_blocks,
                        ..
                    }
                    | crate::Cmd::EngineWorker {
                        ar_form,
                        ar_push_blocks,
                        ..
                    } => crate::ar_push_blocks_of(&ar_form, ar_push_blocks),
                    _ => panic!("not a serve / engine / engine-worker command line"),
                })
                .expect("spawning the parser thread")
                .join()
                .expect("the argv parser panicked")
        }
        assert_eq!(
            push_of(&with(SERVE, &[])),
            0,
            "serve defaults to the one-shot"
        );
        assert_eq!(
            push_of(&with(ENGINE, &[])),
            0,
            "engine defaults to the one-shot"
        );
        assert_eq!(
            push_of(&with(ENGINE, &["--ar-push-blocks", "8"])),
            0,
            "blocks without the form are the one-shot"
        );
        assert_eq!(push_of(&with(ENGINE, &["--ar-form", "split"])), 16);
        assert_eq!(
            push_of(&with(
                SERVE,
                &["--engine-ar-form", "split", "--engine-ar-push-blocks", "8"]
            )),
            8
        );
        let mut args = args_at(false);
        args.prefill_ar_overlap = 1024;
        args.prefill_ar_push_blocks = 16;
        let argv = |args: &EngineArgs| -> Vec<String> {
            let cmd = super::Worker::command(args, 4242).expect("rank 1's command");
            std::iter::once("saber".to_owned())
                .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
                .collect()
        };
        assert_eq!(push_of(&argv(&args)), 16, "rank 1 was not told the form");
        assert_eq!(push_of(&argv(&args_at(false))), 0, "the control");
    }

    /// C2 / C3: the ROW BANDS are plans, so they cross rank 1's argv with every other plan input.
    /// Since C3 both planes default to RECTANGLES (no band planned) and `--ragged-widths` /
    /// `--engine-ragged-widths` opt in to the ragged schedule with prod-shaped bands;
    /// `--row-bands` and `--ragged-ceiling` without the opt-in, and the opt-in beside
    /// `--rectangular-widths`, are refused. The control: a leader with no bands forwards
    /// `--rectangular-widths`, which rank 1 parses to none — never to its own default bands.
    #[test]
    fn the_row_bands_reach_both_planes_and_rank_one() {
        // The bands each plane RESOLVES (rank 1: what its argv says, rank 0 having resolved).
        fn resolved(argv: &[String]) -> Result<Vec<u32>, String> {
            let argv = argv.to_vec();
            std::thread::Builder::new()
                .stack_size(32 << 20)
                .spawn(move || match crate::Cli::try_parse_from(&argv) {
                    Err(e) => Err(e.kind().to_string()),
                    Ok(cli) => Ok(match cli.cmd {
                        crate::Cmd::Serve {
                            engine_ragged_widths,
                            engine_row_bands,
                            engine_rectangular_widths,
                            ..
                        } => crate::decode_bands_of(
                            engine_ragged_widths,
                            engine_rectangular_widths,
                            engine_row_bands,
                        ),
                        crate::Cmd::Engine {
                            ragged_widths,
                            row_bands,
                            rectangular_widths,
                            ..
                        } => crate::decode_bands_of(ragged_widths, rectangular_widths, row_bands),
                        crate::Cmd::EngineWorker {
                            row_bands,
                            rectangular_widths,
                            ..
                        } => {
                            if rectangular_widths {
                                Vec::new()
                            } else {
                                row_bands
                            }
                        }
                        _ => panic!("not a serve / engine / engine-worker command line"),
                    }),
                })
                .expect("spawning the parser thread")
                .join()
                .expect("the argv parser panicked")
        }
        let default = saber_sched::DEFAULT_ROW_BANDS.to_vec();
        // THE DEFAULT (C3): rectangles on both planes.
        assert_eq!(resolved(&with(SERVE, &[])), Ok(Vec::new()));
        assert_eq!(resolved(&with(ENGINE, &[])), Ok(Vec::new()));
        assert_eq!(
            resolved(&with(SERVE, &["--engine-rectangular-widths"])),
            Ok(Vec::new())
        );
        assert_eq!(
            resolved(&with(ENGINE, &["--rectangular-widths"])),
            Ok(Vec::new())
        );
        // The opt-in.
        assert_eq!(
            resolved(&with(SERVE, &["--engine-ragged-widths"])),
            Ok(default.clone())
        );
        assert_eq!(
            resolved(&with(ENGINE, &["--ragged-widths"])),
            Ok(default.clone())
        );
        assert_eq!(
            resolved(&with(
                ENGINE,
                &["--ragged-widths", "--ragged-ceiling", "block"]
            )),
            Ok(default.clone())
        );
        assert_eq!(
            resolved(&with(ENGINE, &["--ragged-widths", "--row-bands", "24,40"])),
            Ok(vec![24, 40])
        );
        // REFUSED: a ragged knob without the opt-in, and the opt-in beside the rectangles.
        for bad in [
            with(ENGINE, &["--row-bands", "24,40"]),
            with(ENGINE, &["--ragged-ceiling", "block"]),
            with(SERVE, &["--engine-ragged-ceiling", "block"]),
            with(ENGINE, &["--ragged-widths", "--rectangular-widths"]),
            with(
                SERVE,
                &["--engine-ragged-widths", "--engine-rectangular-widths"],
            ),
        ] {
            assert!(resolved(&bad).is_err(), "{bad:?} parsed");
        }
        let argv = |args: &EngineArgs| -> Vec<String> {
            let cmd = super::Worker::command(args, 4242).expect("rank 1's command");
            std::iter::once("saber".to_owned())
                .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
                .collect()
        };
        let mut args = args_at(false);
        args.decode_bands = vec![16, 40, 64];
        assert_eq!(
            args.config().decode_bands,
            vec![16, 40, 64],
            "config drops it"
        );
        assert_eq!(resolved(&argv(&args)), Ok(vec![16, 40, 64]));
        // THE CONTROL: no bands on rank 0 is `--rectangular-widths` on rank 1.
        args.decode_bands = Vec::new();
        assert_eq!(
            resolved(&argv(&args)),
            Ok(Vec::new()),
            "rank 1 would plan its default bands beside a rectangular rank 0"
        );
    }

    /// **N8: THE SERVING DEFAULTS are the f16 stored GDN state and the decode launch-train
    /// fusions** (the user's decision of 2026-09-23). Both planes resolve them with no flag;
    /// `--gdn-state-dtype f32` and `--no-decode-fusion` (`--engine-…` on the serve plane) are
    /// the controls; the old bare `--decode-fusion` spelling is accepted and changes nothing;
    /// asking for a set and the control refuses; `EngineArgs::config` carries each way; and rank
    /// 1's argv carries the RESOLVED value of both, by value, for the default and for the control
    /// — a follower on its parser's default beside a leader on the other value is two protocols
    /// (the fusion) or two graphs (the state). **ws-h38/h39: the fusion is a SET** — the default
    /// is ws-h37's (`h37`, rows 1-2), `--decode-fusion all` the ws-h38 served set (rows 1-3 + 7);
    /// rank 1 is told the set; a bare flag before another flag does not swallow it; a typo is
    /// refused.
    #[test]
    fn n8_the_f16_state_and_the_decode_fusion_are_the_default_on_both_planes_and_rank_one() {
        use saber_engine::device::DecodeFusion;
        /// (gdn state dtype, decode fusion RESOLVED) of a serve / engine / engine-worker argv.
        fn resolved(argv: &[String]) -> anyhow::Result<(String, DecodeFusion)> {
            let argv = argv.to_vec();
            std::thread::Builder::new()
                .stack_size(32 << 20)
                .spawn(move || match crate::Cli::try_parse_from(&argv)?.cmd {
                    crate::Cmd::Serve {
                        engine_gdn_state_dtype,
                        engine_decode_fusion,
                        engine_no_decode_fusion,
                        ..
                    } => Ok((
                        engine_gdn_state_dtype,
                        crate::decode_fusion_of(
                            engine_decode_fusion.as_deref(),
                            engine_no_decode_fusion,
                            "--engine-decode-fusion",
                            "--engine-no-decode-fusion",
                        )?,
                    )),
                    crate::Cmd::Engine {
                        gdn_state_dtype,
                        decode_fusion,
                        no_decode_fusion,
                        ..
                    } => Ok((
                        gdn_state_dtype,
                        crate::decode_fusion_of(
                            decode_fusion.as_deref(),
                            no_decode_fusion,
                            "--decode-fusion",
                            "--no-decode-fusion",
                        )?,
                    )),
                    crate::Cmd::EngineWorker {
                        gdn_state_dtype,
                        decode_fusion,
                        ..
                    } => Ok((gdn_state_dtype, decode_fusion)),
                    _ => panic!("not a serve / engine / engine-worker command line"),
                })
                .expect("spawning the parser thread")
                .join()
                .expect("the argv parser panicked")
        }
        let r = |argv: &[String]| resolved(argv).expect("resolves");
        let on = ("f16".to_owned(), DecodeFusion::All);
        let h37 = ("f16".to_owned(), DecodeFusion::H37);
        // The defaults, both planes (the `all` set since 2026-09-23).
        assert_eq!(r(&with(SERVE, &[])), on, "serve's default");
        assert_eq!(r(&with(ENGINE, &[])), on, "engine's default");
        // The accepted no-op spelling changes nothing.
        assert_eq!(r(&with(SERVE, &["--engine-decode-fusion"])), on);
        assert_eq!(r(&with(ENGINE, &["--decode-fusion"])), on);
        assert_eq!(r(&with(SERVE, &["--engine-gdn-state-dtype", "f16"])), on);
        // The controls.
        assert_eq!(
            r(&with(
                SERVE,
                &[
                    "--engine-gdn-state-dtype",
                    "f32",
                    "--engine-no-decode-fusion"
                ]
            )),
            ("f32".to_owned(), DecodeFusion::Off)
        );
        assert_eq!(
            r(&with(
                ENGINE,
                &["--gdn-state-dtype", "f32", "--no-decode-fusion"]
            )),
            ("f32".to_owned(), DecodeFusion::Off)
        );
        assert_eq!(
            r(&with(ENGINE, &["--no-decode-fusion"])),
            ("f16".to_owned(), DecodeFusion::Off)
        );
        assert_eq!(
            r(&with(ENGINE, &["--gdn-state-dtype", "f32"])),
            ("f32".to_owned(), DecodeFusion::All)
        );
        // ws-h38/h39: the sets, both planes; a bare flag before another flag is the no-op and
        // the next flag is still parsed; a typo is refused.
        let all = ("f16".to_owned(), DecodeFusion::All);
        assert_eq!(r(&with(SERVE, &["--engine-decode-fusion", "all"])), all);
        assert_eq!(r(&with(ENGINE, &["--decode-fusion", "all"])), all);
        assert_eq!(r(&with(ENGINE, &["--decode-fusion=all"])), all);
        assert_eq!(r(&with(SERVE, &["--engine-decode-fusion", "h37"])), h37);
        assert_eq!(r(&with(ENGINE, &["--decode-fusion", "h37"])), h37);
        assert_eq!(
            r(&with(
                ENGINE,
                &["--decode-fusion", "--gdn-state-dtype", "f32"]
            )),
            ("f32".to_owned(), DecodeFusion::All)
        );
        assert!(resolved(&with(ENGINE, &["--decode-fusion", "h38"])).is_err());
        assert!(resolved(&with(SERVE, &["--engine-decode-fusion", "nq"])).is_err());
        assert!(resolved(&with(
            ENGINE,
            &["--decode-fusion", "all", "--no-decode-fusion"]
        ))
        .is_err());
        // Both at once is a contradiction, refused by name.
        let e = resolved(&with(ENGINE, &["--decode-fusion", "--no-decode-fusion"]))
            .expect_err("a contradiction resolved");
        assert!(e.to_string().contains("--no-decode-fusion"), "{e}");
        assert!(resolved(&with(
            SERVE,
            &["--engine-decode-fusion", "--engine-no-decode-fusion"]
        ))
        .is_err());
        // `config()` carries each way, and rank 1's argv round-trips each way, by value.
        let argv = |args: &EngineArgs| -> Vec<String> {
            let cmd = super::Worker::command(args, 4242).expect("rank 1's command");
            std::iter::once("saber".to_owned())
                .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
                .collect()
        };
        for (f16, fusion) in [
            (true, DecodeFusion::H37),
            (false, DecodeFusion::Off),
            (true, DecodeFusion::Off),
            (false, DecodeFusion::H37),
            (true, DecodeFusion::All),
            (false, DecodeFusion::All),
        ] {
            let mut args = args_at(false);
            args.gdn_state_f16 = f16;
            args.decode_fusion = fusion;
            let cfg = args.config();
            assert_eq!(
                (cfg.gdn_state_f16, cfg.decode_fusion),
                (f16, fusion),
                "config drops one"
            );
            let w = argv(&args);
            assert!(
                w.iter().any(|a| a == "--decode-fusion"),
                "rank 1's fusion is not by value"
            );
            assert!(
                w.iter().any(|a| a == "--gdn-state-dtype"),
                "rank 1's state is not by value"
            );
            assert_eq!(
                r(&w),
                ((if f16 { "f16" } else { "f32" }).to_owned(), fusion),
                "rank 1 was told something else"
            );
        }
        // The helper mirrors the default (the other tests' arms run at it).
        let d = args_at(false);
        assert!(
            d.gdn_state_f16 && d.decode_fusion == DecodeFusion::All,
            "args_at is not the default"
        );
    }
}

/// **THE ARM'S LAW, and the four ways it must still fail** (ws-p9).
///
/// **THE ATTRIBUTION READS THE SUBJECT'S ROWS, NOT ROW 0** (ws-l18).
///
/// The host half of the conv-update finding. Every decode op but the GDN pair is row-parallel, so
/// a difference an upstream op makes in row 3 is invisible at row 0 — until an op mixes rows, or
/// until a point whose span is wider than one row picks it up. Both of those are the GDN conv:
/// its output row `t` convolves input rows `t-3..t`, and under the OLD slicing law its "row 0" was
/// five rows of q (`saber-exec/tests/edge_rows.rs`). So a row-0-only comparison names the conv
/// update whatever wrote the differing bytes, and these are the tests that say so.
#[cfg(test)]
mod probe_rows {
    use super::*;

    fn point(edge: &str, op: &str, rows: Vec<u64>) -> RowProbe {
        RowProbe {
            layer: 0,
            edge: edge.to_owned(),
            op: op.to_owned(),
            rows,
            row_bytes: 10240,
            raw: Vec::new(),
        }
    }

    /// One step of a plausible layer: a GEMM, then the conv. The GEMM differs in ROW 3 only; the
    /// conv differs in every row, because it convolves them.
    fn two_sides() -> (Vec<Vec<RowProbe>>, Vec<Vec<RowProbe>>) {
        let a = vec![vec![
            point(
                "l0.gdn.qkvz",
                "layer.0.gdn.in_proj_qkvz",
                vec![10, 11, 12, 13],
            ),
            point(
                "l0.gdn.conv",
                "layer.0.gdn.conv_update",
                vec![20, 21, 22, 23],
            ),
        ]];
        let b = vec![vec![
            point(
                "l0.gdn.qkvz",
                "layer.0.gdn.in_proj_qkvz",
                vec![10, 11, 12, 99],
            ),
            point(
                "l0.gdn.conv",
                "layer.0.gdn.conv_update",
                vec![70, 71, 72, 73],
            ),
        ]];
        (a, b)
    }

    /// THE CASE: over the subject's four rows the first differing point is the GEMM, at row 3.
    #[test]
    fn the_first_differing_point_over_every_row_is_the_upstream_op() {
        let (a, b) = two_sides();
        let (step, _, edge, op, row) =
            first_probe_divergence_rows(&a, &b, 4).expect("the two sides differ");
        assert_eq!((step, row), (0, 3));
        assert_eq!(edge, "l0.gdn.qkvz");
        assert_eq!(op, "layer.0.gdn.in_proj_qkvz");
    }

    /// THE NEGATIVE CONTROL, and it is the old instrument: comparing row 0 alone walks past the
    /// GEMM and names the conv update — a defect verdict for a difference the conv did not make.
    #[test]
    fn row_zero_alone_walks_past_it_and_names_the_conv() {
        let (a, b) = two_sides();
        let (_, _, edge, op, row) =
            first_probe_divergence_rows(&a, &b, 1).expect("row 0 differs somewhere");
        assert_eq!(row, 0);
        assert_eq!(edge, "l0.gdn.conv");
        assert_eq!(
            op, "layer.0.gdn.conv_update",
            "this is the reading ws-p9 reported, reproduced on the host"
        );
    }

    /// Two probe sets of different SHAPES are a disagreement about the plan, not about a row.
    #[test]
    fn a_point_the_two_sides_span_differently_is_refused_rather_than_compared() {
        let (a, mut b) = two_sides();
        b[0][0] = a[0][0].clone();
        b[0][0].row_bytes = 20480;
        let (_, _, edge, _, _) =
            first_probe_divergence_rows(&a, &b, 4).expect("a span mismatch is reported");
        assert!(edge.contains("B/row"), "{edge}");
    }
}

/// `BucketDivergence::verdict` is what lets a batched/concurrency red be reported as the
/// documented bucket-arithmetic class instead of failing the gate, so every way it could let
/// something through is a test here. The card-side control is `saber engine`'s own — a batch whose
/// slot 0 continues from the wrong recurrent snapshot — and these are its host half: the same
/// verdict function, fed the shapes a defect makes.
#[cfg(test)]
mod bucket_law {
    use super::*;

    fn near_tie() -> BucketDivergence {
        BucketDivergence {
            at: 51,
            solo_id: Some(220),
            batch_id: Some(318),
            solo_gap: Some(0.25),
            batch_gap: Some(0.375),
            solo_to_batch: Some(0.25),
            batch_to_solo: Some(0.375),
            ties: 0,
            layer_first: Some((
                3,
                0,
                "l1.resid.a".into(),
                "layer.1.input_layernorm".into(),
                0,
            )),
            op_first: Some((3, 0, "l0.mlp.out".into(), "layer.0.mlp.down_proj".into(), 3)),
            op_delta: Some((
                "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1 grid[40,1,4] block[256]".into(),
                "mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1 grid[40,1,2] block[256]".into(),
            )),
            deltas: vec![(
                "layer.N.mlp.down_proj".into(),
                "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1 grid[40,1,4] block[256]".into(),
                "mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1 grid[40,1,2] block[256]".into(),
                64,
            )],
            probed: true,
            subject_slot: 0,
        }
    }

    /// A flip decided BOTH ways: each side puts the other's token more than a nat down.
    fn decided(nearer: f32) -> BucketDivergence {
        let mut d = near_tie();
        d.solo_gap = Some(nearer + 0.5);
        d.batch_gap = Some(nearer);
        d.solo_to_batch = Some(nearer + 0.5);
        d.batch_to_solo = Some(nearer);
        d
    }

    /// ws-k16's batched arm, as the card read it: three divergences over the four sequences'
    /// prefixes (tokens 9, 46 and 2; sequence 0 agreed over ~100).
    fn k16_arm() -> ArmRate {
        ArmRate {
            events: 3,
            reached: 100 + 10 + 47 + 3,
            in_flight: 4,
        }
    }

    /// The case: a sub-nat flip at an op the two buckets resolve with different KERNELS.
    #[test]
    fn a_sub_nat_flip_at_an_op_the_buckets_resolve_differently_is_the_documented_class() {
        let line = near_tie().verdict(1).expect("the documented class");
        assert!(line.contains("0.3750"), "the nats are in the line: {line}");
        assert!(line.contains("mlp.down_proj"), "and the op is: {line}");
    }

    /// A DISAGREEMENT DECIDED BOTH WAYS, with no arm to hold it against, is a different answer
    /// whatever the layer says. This is the tooth that catches a state law read wrongly at a layer
    /// whose plans happen to differ.
    #[test]
    fn a_flip_decided_both_ways_fails_without_an_arm_rate() {
        let why = decided(1.25)
            .verdict(1)
            .expect_err("a flip a nat wide both ways is not a rounding");
        assert!(why.contains("1.2500"), "{why}");
    }

    /// THE CLASS IS A NEAR-TIE FOR AT LEAST ONE SIDE (blind spot 11, P10), and ws-k16's
    /// 1.75 / 0.50 flip is one: the batched side had the solo side's token half a nat down.
    #[test]
    fn a_flip_near_a_tie_on_one_side_is_the_class_however_wide_the_other() {
        let mut d = near_tie();
        d.solo_gap = Some(1.75);
        d.batch_gap = Some(0.5);
        d.solo_to_batch = Some(1.75);
        d.batch_to_solo = Some(0.5);
        let line = d
            .verdict(1)
            .expect("a one-sided near-tie is the documented class");
        assert!(line.contains("nearer side 0.5000"), "{line}");
    }

    /// THE NEGATIVE CONTROL for the change above is the law it replaces: the WORST of the four
    /// gaps. It calls ws-k16's flip a different answer — the reading this round corrects — so a
    /// verdict that went back to it would fail the test above and pass this one.
    #[test]
    fn the_worst_of_four_law_would_have_failed_the_same_flip() {
        let mut d = near_tie();
        d.solo_gap = Some(1.75);
        d.batch_gap = Some(0.5);
        d.solo_to_batch = Some(1.75);
        d.batch_to_solo = Some(0.5);
        assert!(d.worst_nats().expect("four gaps") >= NEAR_TIE_NATS);
        assert!(d.nearer_nats().expect("two gaps") < NEAR_TIE_NATS);
    }

    /// A DECIDED flip in an arm that diverges at the documented rate is REPORTED, not failed.
    #[test]
    fn a_decided_flip_at_the_documented_rate_is_reported_not_failed() {
        let r = k16_arm();
        assert!(r.within_band(), "{}", r.line());
        let line = decided(1.25)
            .verdict_at(1, Some(&r))
            .expect("a decided flip at the documented rate is the floor");
        assert!(line.contains("DECIDED"), "{line}");
        assert!(line.contains("REPORTED"), "{line}");
    }

    /// ...and the SAME flip in an arm diverging far faster than the floor fails.
    #[test]
    fn a_decided_flip_in_an_arm_above_the_documented_rate_fails() {
        let r = ArmRate {
            events: 10,
            reached: 160,
            in_flight: 4,
        };
        assert!(!r.within_band(), "{}", r.line());
        let why = decided(1.25)
            .verdict_at(1, Some(&r))
            .expect_err("ten divergences in 160 positions is not the floor");
        assert!(why.contains("faster than the documented floor"), "{why}");
    }

    /// THE CARD'S CONTROL, in the card's own numbers: the wrong-snapshot batch at TP=2 reads
    /// 19.9375 / 2.7500 at token 1. It is a DEFECT twice over: its nearer side is beyond anything
    /// the floor produced, and one divergence in two positions is not the documented rate.
    #[test]
    fn the_wrong_snapshot_control_stays_red_under_the_rate_law() {
        let mut d = near_tie();
        d.probed = false;
        d.op_first = None;
        d.layer_first = None;
        d.at = 1;
        d.solo_gap = Some(3.75);
        d.batch_gap = Some(2.125);
        d.solo_to_batch = Some(19.9375);
        d.batch_to_solo = Some(2.75);
        let own = ArmRate {
            events: 1,
            reached: d.at + 1,
            in_flight: 4,
        };
        assert!(!own.within_band(), "{}", own.line());
        // Beyond the ceiling at any rate, including the batched arm's.
        let why = d
            .verdict_at(2, Some(&k16_arm()))
            .expect_err("beyond the floor");
        assert!(why.contains("NEARER side"), "{why}");
        assert!(d.verdict_at(2, Some(&own)).is_err());
        // And the rate leg alone refuses it too: the same exposure with a nearer side INSIDE
        // the ceiling still fails.
        let mut e = d.clone();
        e.batch_to_solo = Some(1.5);
        e.batch_gap = Some(1.5);
        let why = e
            .verdict_at(2, Some(&own))
            .expect_err("one event in two positions");
        assert!(why.contains("faster than the documented floor"), "{why}");
    }

    /// The estimator: a diverged sequence reaches its flip INCLUSIVE and stops; an agreeing one
    /// contributes its common prefix.
    #[test]
    fn an_arm_rate_is_a_hazard_over_the_positions_reached() {
        let a: Vec<TokenId> = vec![1, 2, 3, 4, 5];
        let b: Vec<TokenId> = vec![1, 2, 9, 4, 5];
        let c: Vec<TokenId> = vec![7, 7, 7];
        let r = ArmRate::of([(&a[..], &b[..]), (&c[..], &c[..]), (&a[..], &a[..3])]);
        assert_eq!(
            r,
            ArmRate {
                events: 1,
                reached: 3 + 3 + 3,
                in_flight: 0,
            }
        );
        assert!((r.rate() - 1.0 / 9.0).abs() < 1e-12);
        // No events is never surprising; the Poisson tail is exact for one event.
        assert_eq!(ArmRate::default().p_value(), 1.0);
        let one = ArmRate {
            events: 1,
            reached: 100,
            in_flight: 4,
        };
        assert!((one.p_value() - (1.0 - (-1.0f64).exp())).abs() < 1e-12);
    }

    /// THE DOCUMENTED RATE IS A TABLE BY IN-FLIGHT COUNT (ws-p12): the narrowest measured row that
    /// covers the arm, and past the widest row that row, flagged as an extrapolation.
    #[test]
    fn the_concurrency_hazard_counts_each_distinct_prompt_once() {
        // ws-p13: the concurrency-8 arm fed `prompts[i % 4]`, so its eight streams were four
        // prompts twice, and prod's own copies diverge at the same token (ws-p12). Four prompts,
        // two diverging at tokens 10 and 39 over 64 compared, each run twice in the arm.
        let solo: Vec<Vec<TokenId>> = (0..4).map(|p| vec![p as TokenId; 64]).collect();
        let mut got = solo.clone();
        got[1][10] = 9;
        got[3][39] = 9;
        let cycled: Vec<(usize, &[TokenId], &[TokenId])> = (0..8)
            .map(|i| (i % 4, &got[i % 4][..], &solo[i % 4][..]))
            .collect();
        let honest = ArmRate::of_streams(cycled.iter().copied()).in_flight(8);
        // Each prompt once: 2 events over 64 + 11 + 64 + 40 positions.
        assert_eq!((honest.events, honest.reached), (2, 179));
        // THE CONTROL: the old count, every stream a sample, doubles both and makes the arm
        // look twice as well-measured as it is.
        let naive = ArmRate::of(cycled.iter().map(|(_, a, b)| (*a, *b))).in_flight(8);
        assert_eq!((naive.events, naive.reached), (4, 358));
        assert!(naive.p_value() < honest.p_value());
        // Eight DISTINCT prompts are eight samples: the dedup is by prompt, not by position.
        let distinct: Vec<(usize, &[TokenId], &[TokenId])> = (0..8)
            .map(|i| (i, &got[i % 4][..], &solo[i % 4][..]))
            .collect();
        let eight = ArmRate::of_streams(distinct.iter().copied());
        assert_eq!((eight.events, eight.reached), (4, 358));
    }

    #[test]
    fn the_concurrency_arms_walk_the_corpus_for_a_prompt_per_stream() {
        // The gate's four, widened to the widest concurrency arm that will run.
        assert_eq!(prompt_rows_wanted(4, &[1, 4, 8], 8, false), 8);
        // An arm past `max_seqs` does not run, so it asks for nothing.
        assert_eq!(prompt_rows_wanted(4, &[1, 4, 8], 4, false), 4);
        assert_eq!(prompt_rows_wanted(4, &[], 8, false), 4);
        assert_eq!(prompt_rows_wanted(0, &[1], 8, false), 1);
        // A transcript keeps its own turns.
        assert_eq!(prompt_rows_wanted(4, &[1, 4, 8], 8, true), 4);
        // A red on a stream past the gate's prompts is attributed in a gate-sized batch with
        // the red's own prompt in it; one inside the gate's prompts runs the gate's batch.
        let gate: Vec<Vec<TokenId>> = (0..4).map(|p| vec![p as TokenId]).collect();
        let conc: Vec<Vec<TokenId>> = (0..8).map(|p| vec![p as TokenId]).collect();
        let (set, at) = attribution_subject(&gate, &conc, 6);
        assert_eq!((set.len(), at), (4, 2));
        assert_eq!(set[2], vec![6 as TokenId]);
        assert_eq!(set[0], gate[0]);
        let (set, at) = attribution_subject(&gate, &conc, 3);
        assert_eq!((set, at), (gate.clone(), 3));
    }

    #[test]
    fn the_documented_rate_is_read_by_the_number_in_flight() {
        assert_eq!(documented_rate(0), (0.0100, 4, false));
        assert_eq!(documented_rate(1), (0.0100, 4, false));
        assert_eq!(documented_rate(4), (0.0100, 4, false));
        assert_eq!(documented_rate(5), (0.0110, 8, false));
        assert_eq!(documented_rate(8), (0.0110, 8, false));
        assert_eq!(documented_rate(16), (0.0110, 8, true));
        let wide = ArmRate {
            events: 1,
            reached: 10,
            in_flight: 16,
        };
        assert!(wide.line().contains("EXTRAPOLATED"), "{}", wide.line());
        // The rows are ordered and the table is prod's own: eight in flight is not a faster floor
        // than four by more than the two rows' own confidence intervals.
        assert!(DOCUMENTED_RATES.windows(2).all(|w| w[0].0 < w[1].0));
    }

    /// **THE NEGATIVE CONTROL for the table**: the concurrency-8 arm's own reading (ws-p11, 7
    /// divergences over 157 positions) is NOT excused by the eight-in-flight row. Prod's own
    /// eight-in-flight rate is the same size as its four, so the row moves the yardstick by a
    /// tenth and the arm stays outside it — a table that silently absorbed it would pass the test
    /// above and fail this one.
    #[test]
    fn the_eight_in_flight_row_does_not_absorb_the_concurrency_8_arm() {
        let arm = ArmRate {
            events: 7,
            reached: 157,
            in_flight: 8,
        };
        assert!(!arm.within_band(), "{}", arm.line());
        assert!(arm.p_value() < 0.01, "{}", arm.line());
        // ...while prod's own eight-in-flight reading, held to its own row, is inside it.
        let prod = ArmRate {
            events: 83,
            reached: 9345,
            in_flight: 8,
        };
        assert!(prod.within_band(), "{}", prod.line());
    }

    /// THE EMBEDDING can never be the class: the two buckets were fed different ids, which no
    /// bucket decides.
    #[test]
    fn a_divergence_at_the_embedding_fails() {
        let mut d = near_tie();
        d.op_first = Some((
            0,
            -1,
            "hidden.0".into(),
            "layer.0.input_layernorm".into(),
            0,
        ));
        d.layer_first = d.op_first.clone();
        let why = d
            .verdict(1)
            .expect_err("the embedding is upstream of every bucket fact");
        assert!(why.contains("EMBEDDING"), "{why}");
    }

    /// AN OP BOTH BUCKETS RESOLVE IDENTICALLY, writing different bytes, is ws-l17's own defect
    /// shape: the same launch reading something else.
    #[test]
    fn the_same_launch_writing_different_bytes_fails() {
        let mut d = near_tie();
        d.op_first = Some((
            0,
            0,
            "l0.gdn.conv".into(),
            "layer.0.gdn.conv_update".into(),
            0,
        ));
        d.op_delta = None;
        let why = d
            .verdict(1)
            .expect_err("an identical launch must not be excused");
        assert!(why.contains("SAME launch"), "{why}");
    }

    /// AND A DELTA THAT MOVES ONLY THE GRID is not an arithmetic difference: a workgroup that
    /// serves its own sequence computes the same bits whatever the grid's sequence axis is, which
    /// is exactly what ws-l15's pinned split count bought. This is the case the GDN-side control
    /// on the card lands in — `gdn.conv_update` keeps its kernel and only its grid moves.
    #[test]
    fn a_grid_only_delta_is_not_an_arithmetic_and_fails() {
        let mut d = near_tie();
        d.op_first = Some((
            0,
            0,
            "l0.gdn.conv".into(),
            "layer.0.gdn.conv_update".into(),
            0,
        ));
        d.op_delta = Some((
            "saber_gdn_conv_update_w4_h128_hpb1 grid[1,80,1] block[32]".into(),
            "saber_gdn_conv_update_w4_h128_hpb1 grid[4,80,1] block[32]".into(),
        ));
        let why = d.verdict(1).expect_err("a grid is not an arithmetic");
        assert!(why.contains("SAME KERNEL"), "{why}");
    }

    /// AT TWO RANKS there is no probe, so the verdict rests on the margin and on the buckets
    /// resolving different kernels at all — weaker, and it says so in the line it returns.
    #[test]
    fn at_two_ranks_the_verdict_names_what_it_could_not_ask() {
        let mut d = near_tie();
        d.probed = false;
        d.op_first = None;
        d.layer_first = None;
        let line = d
            .verdict(2)
            .expect("the documented class, attributed by the margin");
        assert!(line.contains("plan facts alone"), "{line}");
        // And with every op resolved identically there is nothing to attribute it to.
        let mut e = d.clone();
        e.deltas.clear();
        assert!(e.verdict(2).is_err());
        // A grid-only delta is not an arithmetic at two ranks either.
        let mut f = d.clone();
        f.deltas = vec![(
            "layer.N.gdn.conv_update".into(),
            "saber_gdn_conv_update_w4_h128_hpb1 grid[1,80,1] block[32]".into(),
            "saber_gdn_conv_update_w4_h128_hpb1 grid[4,80,1] block[32]".into(),
            48,
        )];
        assert!(f.verdict(2).is_err());
    }

    /// A flip nobody could measure is not a near-tie: it is an unmeasured flip.
    #[test]
    fn no_distribution_at_the_flip_fails() {
        let mut d = near_tie();
        d.solo_gap = None;
        d.solo_to_batch = None;
        assert!(d.verdict(1).is_err());
        let mut e = near_tie();
        e.solo_gap = None;
        e.batch_gap = None;
        e.solo_to_batch = None;
        e.batch_to_solo = None;
        assert!(e.verdict(1).is_err());
    }

    /// The margin is the LOGIT gap and the gap to the other engine's token is measured against
    /// that engine's own top1 (`docs/GATES.md` blind spot 11).
    #[test]
    fn a_rows_margins_are_the_gaps_its_own_row_states() {
        let r = RowMargin {
            id: 220,
            top: vec![(220, 3.5), (318, 3.25), (9, 1.0)],
        };
        assert_eq!(r.gap(), Some(0.25));
        assert_eq!(r.gap_to(318), Some(0.25));
        assert_eq!(r.gap_to(9), Some(2.5));
        assert_eq!(r.gap_to(1234), None);
        assert!(!r.exact_tie());
        assert!(RowMargin {
            id: 1,
            top: vec![(1, 2.0), (2, 2.0)],
        }
        .exact_tie());
    }
}

/// FN29: the gate's concurrency arms admit in a FIXED order, whatever order the tasks that consume
/// the streams start in. The executor is `saber_sched`'s deterministic fake wrapped in a recorder;
/// the scheduler is the real one over `saber-sched`'s own test fixture (a planner-resolved plan).
/// `saber-sched`'s own scheduling fixture, shared rather than copied: the FN29 test below drives
/// the real scheduler over it.
#[cfg(test)]
#[path = "../../saber-sched/tests/common/mod.rs"]
mod fn29_sched_fixture;

#[cfg(test)]
mod fn29_admission_tests {
    use std::sync::{Arc, Mutex};

    use super::fn29_sched_fixture as common;

    use saber_sched::{
        Acceptance, AdmittedSeq, RandomAcceptanceExecutor, SeqId, StepBatch, StepExecutor,
        StepResult,
    };
    use saber_serve::engine::{EngineRequest, TokenId};

    use super::drive_streams;

    /// Everything the scheduler did, in order: admissions (by stream) and steps (phase + streams).
    #[derive(Debug)]
    struct Recorder {
        inner: RandomAcceptanceExecutor,
        log: Arc<Mutex<Vec<String>>>,
        seq_stream: std::collections::BTreeMap<SeqId, usize>,
    }

    /// Stream `i`'s prompt opens with `STREAM_BASE + i`: the recorder names streams by it.
    const STREAM_BASE: TokenId = 50_000;

    impl StepExecutor for Recorder {
        fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
            let streams: Vec<usize> = batch
                .slots
                .iter()
                .map(|s| {
                    self.seq_stream
                        .get(&s.seq_id)
                        .copied()
                        .unwrap_or(usize::MAX)
                })
                .collect();
            self.log.lock().expect("log").push(format!(
                "step {} {:?} {:?} {streams:?}",
                batch.step_index, batch.phase, batch.bucket
            ));
            self.inner.execute(batch)
        }
        fn admit(&mut self, seq: &AdmittedSeq) {
            let stream = (seq.prompt_tokens[0] - STREAM_BASE) as usize;
            self.seq_stream.insert(seq.seq_id, stream);
            self.log
                .lock()
                .expect("log")
                .push(format!("admit stream {stream}"));
            self.inner.admit(seq);
        }
        fn release(&mut self, seq: SeqId) {
            self.inner.release(seq);
        }
    }

    fn requests(order: &[usize]) -> Vec<EngineRequest> {
        order
            .iter()
            .map(|&i| {
                // Distinct lengths, so the prefill chunks differ stream to stream.
                let mut p = vec![STREAM_BASE + i as TokenId];
                p.extend((0..(9 + 13 * i as TokenId)).map(|t| 1000 + t));
                common::request(&p, 24 + 4 * i)
            })
            .collect()
    }

    type Run = (Vec<String>, Vec<Vec<TokenId>>, Option<(u64, bool)>);

    /// One arm on a fresh scheduler: the recorded schedule, every stream's ids, the late joiner.
    fn run(stream_order: &[usize], join_after: Option<usize>, task_order: &[usize]) -> Run {
        let log = Arc::new(Mutex::new(Vec::new()));
        let exec = Recorder {
            inner: RandomAcceptanceExecutor::new(29, Acceptance::Bernoulli { p: 0.6 }),
            log: Arc::clone(&log),
            seq_stream: Default::default(),
        };
        // The fixture's plan resolves once (it is the slow part in a debug build); every run gets
        // a fresh scheduler over it, so no state crosses runs.
        static CONFIG: std::sync::OnceLock<saber_sched::SchedulerConfig> =
            std::sync::OnceLock::new();
        let config = CONFIG.get_or_init(|| {
            common::config_with(
                &common::decode_buckets(),
                2,
                saber_sched::ArenaBytes {
                    kv_per_rank: 8 * common::GIB,
                    gdn_state_per_rank: 8 * common::GIB,
                },
            )
        });
        let sched = Arc::new(
            saber_sched::Scheduler::new(
                config.clone(),
                exec,
                common::model_info(),
                common::boot_ledger(),
            )
            .expect("the arenas fit"),
        );
        let arm = drive_streams(&sched, requests(stream_order), join_after, task_order)
            .expect("the arm runs");
        let ids = arm.out.into_iter().map(|(ids, _, _)| ids).collect();
        let log = log.lock().expect("log").clone();
        (log, ids, arm.late)
    }

    /// A deterministic Fisher-Yates over `0..n` (no `rand` in this crate).
    fn shuffled(n: usize, seed: u64) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        let mut x = (seed + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        for i in (1..n).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v.swap(i, (x % (i as u64 + 1)) as usize);
        }
        v
    }

    fn assert_fixed(n: usize, join_after: Option<usize>) {
        let ident: Vec<usize> = (0..n).collect();
        let (want_log, want_ids, want_late) = run(&ident, join_after, &ident);
        // The admission is stream order.
        let admits: Vec<String> = want_log
            .iter()
            .filter(|l| l.starts_with("admit"))
            .cloned()
            .collect();
        let expect: Vec<String> = (0..n).map(|i| format!("admit stream {i}")).collect();
        assert_eq!(admits, expect);
        let mut distinct_orders = std::collections::BTreeSet::new();
        for seed in 0..100u64 {
            let order = shuffled(n, seed);
            distinct_orders.insert(order.clone());
            let (log, ids, late) = run(&ident, join_after, &order);
            assert_eq!(
                log, want_log,
                "task-start order {order:?} moved the schedule"
            );
            assert_eq!(
                ids, want_ids,
                "task-start order {order:?} moved a stream's ids"
            );
            assert_eq!(late, want_late);
        }
        // The shuffle is a shuffle: most of the n! orders are exercised.
        assert!(
            distinct_orders.len() >= 20,
            "{} orders",
            distinct_orders.len()
        );
    }

    #[test]
    fn concurrency_4_admits_in_stream_order_across_100_task_start_orders() {
        assert_fixed(4, None);
    }

    #[test]
    fn concurrency_8_admits_in_stream_order_across_100_task_start_orders() {
        assert_fixed(8, None);
    }

    /// The mixed arrival: the late joiner is admitted after exactly `join_after` steps.
    #[test]
    fn the_late_joiner_lands_after_the_same_step_across_100_task_start_orders() {
        assert_fixed(4, Some(3));
        let ident: Vec<usize> = (0..4).collect();
        let (log, _, late) = run(&ident, Some(3), &ident);
        assert_eq!(late, Some((3, true)));
        let steps_before = log
            .iter()
            .take_while(|l| l.as_str() != "admit stream 3")
            .filter(|l| l.starts_with("step"))
            .count();
        assert_eq!(steps_before, 3, "{log:#?}");
    }

    /// A late joiner whose peers finish first is admitted at the idle point, deterministically.
    #[test]
    fn a_late_joiner_after_the_others_finish_is_admitted_at_idle() {
        let ident: Vec<usize> = (0..2).collect();
        let (_, ids, late) = run(&ident, Some(10_000), &ident);
        let (at, on_time) = late.expect("admitted");
        assert!(!on_time && at > 0 && at < 10_000, "{late:?}");
        assert_eq!(ids[1].len(), 28);
    }

    /// The control: the recorder SEES the admission order — a different submission order is a
    /// different schedule, so the identity above is not a recorder that cannot tell.
    #[test]
    fn a_different_submission_order_is_a_different_schedule() {
        let ident: Vec<usize> = (0..4).collect();
        let (a, _, _) = run(&ident, None, &ident);
        let (b, _, _) = run(&[2, 0, 3, 1], None, &ident);
        assert_ne!(a, b);
    }
}
