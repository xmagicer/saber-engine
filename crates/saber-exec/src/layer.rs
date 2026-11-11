//! P1 — one decoder block, executed from the model graph.
//!
//! `saber-model`'s [`Graph`] describes the forward pass as data: ops in order, the edges between
//! them with a layout rule on every edge, the caller-owned state each op rewrites, the weights by
//! exact pack record name, and `n_launches` per op. This module turns the slice of it that covers
//! layers `L..M` into something that runs:
//!
//! ```text
//!   Graph::ops_for(bucket)          the op list, already filtered and layout-resolved
//!        │
//!        ▼
//!   BlockPlan::new                  one arena: a byte range per edge, per edge SCALE, and per
//!        │                          op-private intermediate; a route per op; the launch budget
//!        ▼
//!   BlockExecutor::run              resolve → bind → Launcher::launch_op on ONE stream
//! ```
//!
//! Three things are policy here, and each is a claim the gate can falsify:
//!
//! * **The plan is pure.** [`BlockPlan::new`] touches no GPU and reads no file: every byte count
//!   comes from the graph's own geometry and the bucket, which is why the host tests can pin the
//!   edge count, the total bytes and the layout of every edge at M=5 and M=128 with no device.
//! * **A route is a kernel, an identity, or a refusal — never a guess.** An op whose kernel has no
//!   code object ([`KernelOp::n_launches`] is `None`) is [`Route::Supplied`]: its outputs are
//!   uploaded from the reference capture and it is named in the report rather than silently
//!   skipped. The two-rank all-reduce is [`Route::Identity`] for the same reason: rank 0 of a TP=2
//!   tier has no peer here and `IpcPeerVisible` workspaces are not implemented, so the reduce is
//!   replaced by an ALIAS of its input edge — no copy, no launch, and the report says so.
//! * **The launch count is budgeted before the block runs.** [`BlockPlan::declared_launches`] sums
//!   the graph's own `n_launches` over exactly the ops in the block, the way
//!   `Graph::launch_count_lower_bound` sums it over all of them, and
//!   [`BlockPlan::expected_launches`] subtracts what identity replaced. `run` asserts the total the
//!   `Launcher` actually issued against that number (structural rule 7).
//!
//! What this module does NOT do is decide anything a rules table can decide. Every kernel comes
//! from `saber_kernels::rules::resolve` or `saber_kernels::saber::resolve`; the per-call geometry
//! those take is derived from the graph plus [`RuntimeParams`], and a value that cannot be derived
//! (the paged context length, the state-slot count) is an explicit input that the caller
//! cross-checks against its capture.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use saber_hip::{DeviceAddr, DevicePtr, HipThread, Job, PinnedBuf, Stream};
use saber_kernels::glue::{self, GlueRules};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};
use saber_kernels::rules::{
    producer_writes_tiled, resolve, resolve_with_layout, ALayout, ATiledLaw, LaunchRules,
    Op as GemmOp, Shape, NO_PARTITIONS,
};
use saber_kernels::saber::{
    self, attn_decode_combine, gdn_decode_pair_recurrent, AttnShape, GdnPrefillShape, GdnShape,
    KvDtype, SaberRules,
};
use saber_kernels::LaunchDescriptor;
use saber_model::graph::{
    glue_shape, Bucket, Geometry, Graph, KernelOp, Layout, Op as ModelOp, Phase, Site, WeightRole,
};

use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};

/// Alignment of every buffer inside the activation arena.
///
/// 256 B, not 4 kiB: the arena holds a hundred-odd small activations and the kernels only need
/// natural alignment, but a fragment-tiled `A` is read 128 bits at a time and a `hipMemcpy` into
/// the middle of an unaligned region is measurably slower.
pub const ARENA_ALIGN: u64 = 256;

/// `SOFTPLUS_THRESHOLD` — a saber constant, not a knob (`ref_gdn.py`).
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;

fn align_up(x: u64) -> u64 {
    x.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

// =====================================================================================
// Buffers
// =====================================================================================

/// What a planned buffer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// A graph edge's payload.
    EdgeData,
    /// A graph edge's per-token activation scale. An fp8 activation edge is TWO buffers: the
    /// `e4m3` bytes and the `f32` scale the consuming GEMM multiplies by. The graph models the
    /// pair as one edge because the layout rule applies to both.
    EdgeScale,
    /// An op-private intermediate with no edge: the GDN core's q/k/v split, the prefill gates,
    /// the zero residual layer 0 reads. `chunk_scan`'s final state is NOT one of these — it is
    /// written in place into the session's state slot (ws-h30).
    Internal,
    /// A block input: an edge consumed here whose producer is outside the block.
    Input,
}

/// How row `r` of a planned buffer is FOUND: the byte ranges, inside the buffer, that make it up.
///
/// **`bytes / rows` is the truth for the residual stream and an assumption everywhere else**
/// (`docs/GATES.md` blind spot 17). `l{n}.gdn.conv` is ONE edge holding q, k and v as three
/// row-major BLOCKS — the conv kernel writes `q[(row * Hg + h) * D]`, `k` the same and
/// `v[(row * H + h) * D]` into three separate tensors that the planner lays end to end — so at
/// TP=1 its `bytes / rows` is 20,480 bytes that span **five rows of q** and none of k or v. A
/// reader that slices by it is not reading a row; at row 0 it is reading five of them, which is
/// how an op-deep attribution came to name `gdn.conv_update` as the first op to write differing
/// bytes when what had differed was four rows nothing else compared (ws-l18).
///
/// So the layout is a fact of the PLAN, stated where the buffer is sized, and every reader that
/// wants a row asks for one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowSpan {
    /// One row-major block: row `r` is `[r * bytes, bytes)`.
    Contiguous {
        /// Bytes per row.
        bytes: u64,
    },
    /// Several row-major blocks in one buffer: row `r` is the concatenation of
    /// `[base + r * bytes, bytes)` over the blocks, in order.
    Blocks {
        /// `(base offset of the block, bytes per row of it)`.
        blocks: Vec<(u64, u64)>,
    },
}

impl RowSpan {
    /// Bytes one row holds, across every block.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            RowSpan::Contiguous { bytes } => *bytes,
            RowSpan::Blocks { blocks } => blocks.iter().map(|(_, b)| *b).sum(),
        }
    }

    /// Rows a buffer of `buffer_bytes` holds under this layout, or 0 if it holds none.
    #[must_use]
    pub fn rows(&self, buffer_bytes: u64) -> u64 {
        match self {
            RowSpan::Contiguous { bytes } => {
                if *bytes == 0 {
                    0
                } else {
                    buffer_bytes / bytes
                }
            }
            // Every block spans the same rows, so the first one answers for all of them.
            RowSpan::Blocks { blocks } => blocks
                .iter()
                .map(|(base, per)| {
                    if *per == 0 {
                        0
                    } else {
                        buffer_bytes.saturating_sub(*base) / per
                    }
                })
                .min()
                .unwrap_or(0),
        }
    }

    /// Row `row` as `(offset in the buffer, length)` ranges, in order.
    #[must_use]
    pub fn spans(&self, row: u64) -> Vec<(u64, u64)> {
        match self {
            RowSpan::Contiguous { bytes } => vec![(row * bytes, *bytes)],
            RowSpan::Blocks { blocks } => blocks
                .iter()
                .map(|(base, per)| (base + row * per, *per))
                .collect(),
        }
    }
}

/// One buffer in the activation arena.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferPlan {
    /// Unique name. An edge's payload uses the edge name; its scale appends `.scale`.
    pub name: String,
    /// What it is.
    pub kind: BufferKind,
    /// Byte offset in the arena.
    pub offset: u64,
    /// Byte length, before alignment.
    pub bytes: u64,
    /// The resolved layout of the edge this belongs to.
    pub layout: Layout,
    /// Where row `r` of it lives. See [`RowSpan`].
    pub rows: RowSpan,
    /// The op that writes it, or `"<block input>"`.
    pub producer: String,
}

/// Where a descriptor's named buffer gets its address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A byte range of the one activation arena, absolute.
    Arena {
        /// The [`BufferPlan`] it lands in, for messages.
        buffer: String,
        /// Absolute offset in the arena.
        offset: u64,
        /// Bytes the op reads or writes there.
        bytes: u64,
    },
    /// A weight, by **pack record name**: `"<op>.w" | "<op>.ws" | "<op>.wref"` for a matrix,
    /// otherwise the checkpoint's own name. `layout` is the id the record must carry
    /// (`docs/abi/PACK.md` § Layout ids) — it comes from
    /// [`saber_model::graph::Graph::device_final_records`], so the expectation is the model's and
    /// [`BlockExecutor::address`] refuses a record that is something else.
    Weight {
        /// The record name.
        key: String,
        /// The layout id the record must carry.
        layout: u32,
    },
    /// Caller-owned state, by the graph's [`saber_model::graph::StateRef`] name, and WHERE in
    /// it this op starts reading ([`StateAt`]).
    State {
        /// The state's name.
        key: String,
        /// The base the op is handed.
        at: StateAt,
    },
    /// An index table the scheduler would build (P4) and the capture supplies: `cu`, `sidx`,
    /// `cache_idx`, `naccept`, `has_init`, `block_table`, `seqused_k`. Keyed `"<op>.<buffer>"`.
    Index(String),
}

/// Where in a caller-owned state buffer an op starts reading — and the one law that is not the
/// same for every GDN op.
///
/// Most state bindings find their own slot: `cache_idx` and `sidx` are index tables, so
/// `saber_gdn_conv_prep`, `saber_gdn_conv_update`, `saber_gdn_fused_update` and
/// `saber_gdn_recurrent_update` are all handed the POOL's base and do the arithmetic themselves.
///
/// **`saber_gdn_chunk_scan` does not.** It indexes `h0` and `ht` as `(nq * H + hv) * VD * KD` with
/// `nq = blockIdx.z` — the sequence's index IN THE STEP, not a slot — so the pointer it is handed
/// has to start at the step's FIRST LIVE sequence's slot. That is a fact about the STEP, in
/// exactly the sense [`BlockPlan::retarget_prefill_rows`] is (ws-h27), and prefill steps are eager
/// so a per-step binding is legal where a captured graph's would not be
/// ([`BlockPlan::retarget_prefill_state`], ws-h30).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateAt {
    /// The buffer's base: the op finds its own session's slot from an index table.
    Pool,
    /// One session's slot.
    Slot {
        /// The slot, never 0 — slot 0 is the kernels' null slot and an op that WRITES it poisons
        /// every padding row that reads it.
        slot: u32,
        /// Its byte offset from the base: `slot * bytes_per_slot`.
        offset: u64,
    },
    /// A per-step binding no step has made yet.
    ///
    /// [`BlockExecutor::address`] refuses it BY NAME rather than falling back to the base, because
    /// the base is slot 0 and the fallback would be a prefill reading the null slot for its
    /// initial state and publishing its final one into it — which is the ws-h29 defect exactly.
    PerStep,
}

// =====================================================================================
// Routes and steps
// =====================================================================================

/// How one op is executed.
#[derive(Clone, Debug)]
pub enum Route {
    /// One or more dispatches. `descs[0]` is the head whose `n_launches` the launcher asserts.
    Kernel {
        /// The dispatch sequence.
        descs: Vec<LaunchDescriptor>,
        /// The shape the Persistent workspace key is built from.
        shape: Shape,
    },
    /// The output edge is an ALIAS of the input edge: nothing is copied and nothing is launched.
    Identity {
        /// Why.
        reason: &'static str,
    },
    /// No code object implements this op; its outputs are uploaded from the reference capture.
    Supplied {
        /// Why.
        reason: &'static str,
    },
    /// An EARLIER op's kernel already did this op's work, so it launches nothing.
    ///
    /// The one producer of this route today is the `_nq` epilogue: `saber_ar_oneshot_2rank_exact_nq`
    /// is the exact reduce with the residual add, the Gemma RMS norm and the per-token `e4m3`
    /// quant folded into its epilogue, so ONE dispatch covers the `allreduce` op and the
    /// `add_rms_quant` op that consumes it. The reduce carries the fused descriptor and the norm
    /// carries this.
    Fused {
        /// The op whose descriptor does the work.
        into: String,
        /// Why.
        reason: &'static str,
    },
}

impl Route {
    /// Dispatches this route issues.
    #[must_use]
    pub fn launches(&self) -> u32 {
        match self {
            Route::Kernel { descs, .. } => descs.iter().map(|d| d.n_launches).max().unwrap_or(0),
            Route::Identity { .. } | Route::Supplied { .. } | Route::Fused { .. } => 0,
        }
    }

    /// The kernel symbols, for the report.
    #[must_use]
    pub fn symbols(&self) -> Vec<&str> {
        match self {
            Route::Kernel { descs, .. } => descs.iter().map(|d| d.kernel.as_str()).collect(),
            Route::Identity { .. } | Route::Supplied { .. } | Route::Fused { .. } => vec![],
        }
    }
}

/// A fused dispatch: the descriptor that does two ops' work, and the buffers it names.
type Fusion = (LaunchDescriptor, Vec<(String, Source)>);

/// Which fusion folded an op into the step before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FoldKind {
    /// The `_nq` epilogue: `add_rms_quant` into the exact reduce.
    Nq,
    /// The decode GEMM pair: the second of two GEMMs over one activation.
    GemmPair,
    /// Fusion-map row 3 (ws-h38): the exact reduce AFTER a row-parallel decode GEMM, folded into
    /// the GEMM's split-K last block.
    GemmAr,
    /// ... and the `add_rms_quant` after that reduce, whose epilogue the same dispatch runs.
    GemmArNorm,
    /// ws-h38: `kv_append` into the `qk_norm_rope` before it.
    RopeAppend,
    /// Fusion-map row 4 (ws-h40): `silu_mul_quant` into the gate_up decode GEMM before it.
    GemmSmq,
    /// Fusion-map row 5 (ws-h42): `gdn_norm_quant` into the GDN fused update before it.
    GdnNq,
    /// CMP27AE: the rotated (wht6) reduce AFTER an N 5120 dwn4 decode GEMM, folded into the GEMM's
    /// tile epilogue (`--engine-ar-fuse`).
    GemmWht,
}

/// One op of the block, resolved and bound.
#[derive(Clone, Debug)]
pub struct Step<'g> {
    /// The graph op.
    pub op: &'g ModelOp,
    /// How it runs.
    pub route: Route,
    /// Every buffer the descriptor names, in descriptor order. A slot the descriptor binds null
    /// (an unused descale, the split-K partials at DKS=1) is absent, and so is a workspace: the
    /// launcher owns those by kind.
    pub bindings: Vec<(String, Source)>,
}

impl Step<'_> {
    /// The op's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.op.name
    }

    /// The source bound to `buffer`, if any.
    #[must_use]
    pub fn source(&self, buffer: &str) -> Option<&Source> {
        self.bindings
            .iter()
            .find(|(n, _)| n == buffer)
            .map(|(_, s)| s)
    }
}

// =====================================================================================
// Runtime parameters
// =====================================================================================

/// The two per-call values the graph cannot state, and the block therefore takes as inputs.
///
/// Everything else a saber entry point needs — every stride, both head counts, the conv window,
/// the state-slot pitch, the softmax and delta-rule scales — is derived from the graph's geometry
/// and the bucket inside this module. These two are properties of the *session*: how many KV
/// tokens the sequence already has, and how many state slots its allocator handed out. In P4 they
/// come from the scheduler; here they come from the capture and the caller checks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeParams {
    /// Paged KV tokens the attention layer reads.
    pub max_ctx: u32,
    /// GDN state slots the conv and recurrent caches are sized for.
    pub gdn_slots: u32,
}

impl RuntimeParams {
    /// Blocks in the page table for `max_ctx`.
    #[must_use]
    pub fn kv_blocks(&self, block_size: u32) -> u32 {
        self.max_ctx.div_ceil(block_size)
    }
}

// =====================================================================================
// The plan
// =====================================================================================

/// Which of the graph's sites a plan covers, and whether its layers share one set of buffers.
///
/// [`PlanOptions::BLOCK`] is what P1 planned: a handful of adjacent decoder layers, each edge its
/// own buffer, so the gate can byte-compare EVERY op's output after the block has run. A whole
/// forward pass cannot be planned that way — 64 layers of distinct edges is ~19 GiB of arena at a
/// 2048-token chunk — so [`PlanOptions::FORWARD`] turns on slot reuse, and the two settings are
/// different for a reason worth stating: reuse makes intermediate results UNREADABLE after the
/// fact, which is exactly what a bit-exact gate needs to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOptions {
    /// Include `Site::Embedding`'s ops: the vocab-parallel lookup and, at TP > 1, its reduce.
    pub embedding: bool,
    /// Include `Site::Head`'s ops: the final norm, `lm_head`, and at TP > 1 the vocab gather.
    pub head: bool,
    /// Give every layer-local edge ONE arena buffer, shared by every layer in the plan.
    ///
    /// Safe because the graph's only cross-layer edges are the two `push_layer` returns —
    /// `l<n>.out` and `l<n>.resid.b` — and layer `n+1` READS both in its first op
    /// (`input_layernorm`) and writes its own copies later (`post_attention_layernorm` and the
    /// MLP's last op). Every other edge is produced and consumed inside one layer. This module
    /// does not take that on faith: [`BlockPlan::check_slot_reuse`] re-derives it from the plan's
    /// own step order and refuses a graph where it stops holding.
    pub reuse_layer_slots: bool,
    /// This rank's index in the tier.
    ///
    /// The vocab-parallel embedding is the only op whose CODE depends on it: rank `r` owns rows
    /// `[r * vocab_local, (r + 1) * vocab_local)` and writes zeros for every id it does not own, so
    /// the reduce that follows sums exactly one non-zero contribution per token. The planner
    /// refused to plan the site at all above TP=1 until ws-h16, because it had no rank to give it.
    pub rank: u32,
    /// How `layer.allreduce` runs.
    pub collectives: Collectives,
    /// Plan the head site at the plan's own row count even when that is not the sampled row
    /// count.
    ///
    /// Off, and refused, by default. A prefill chunk's head belongs at
    /// `Bucket::sampled_rows` -- one row per sequence -- and planning it at the chunk's `M`
    /// computes a shard of logits per token, of which a sampler reads one row: 2048 x 248320 at
    /// `p2048`, 970 MiB, 27.6 ms of a 637 ms step (`docs/LEDGER.md`, workstream I15). No serving
    /// path wants that, so the planner refuses it rather than letting it look like a slow step.
    ///
    /// A **measurement** harness does want it, exactly once: to time the every-row head and to
    /// compare its sampled row against the row the split plan produces. That is what this flag is
    /// for, and it is spelled at the call site so that a reader can see which of the two plans a
    /// number came from.
    pub head_on_every_row: bool,
    /// Resolve the decode attention's split-KV count from the sequence's own context, never from
    /// how many sequences share the step.
    ///
    /// **This is what made a batched step answer differently from a solo one** (ws-l15, the
    /// 2026-09-21 census). `saber_attn_paged`'s split law is
    /// `raw = 192 / (kv_heads * num_seqs)`, snapped down to a power of two with a floor of 16, and
    /// the kernel then walks `tps = ceil(ceil(ctx / 16) / splits)` tiles per segment. At
    /// `kv_heads = 6` that is **32 splits for one sequence and 16 for two or more**, so a sequence
    /// past `16 x 16 = 256` resident keys is partitioned into segments of two tiles in a batch and
    /// of one tile alone. The online-softmax combine over a different partition is a different sum
    /// in a different order: the same request, with the same state, in the same pages, gets
    /// different bits — and a few steps later a different token. Measured: with four corpus
    /// prompts (234/258/251/223 tokens) the two whose context passes 256 diverge from their own
    /// solo run at `l4.resid.a` — the first full-attention layer — and nowhere earlier, sequence 1
    /// at step 0 (259 keys) and sequence 2 at step **5**, which is exactly the step its context
    /// reaches 257.
    ///
    /// The law is prod's, verbatim, and prod has the same property; what prod does not have is a
    /// gate that says a request's answer is a function of the request. This engine does
    /// (`tools/gates/multiturn_continuity.py`, `saber engine`'s batched and concurrent arms), and
    /// a serving engine whose tokens depend on its neighbours cannot pass it. So a caller that
    /// serves several sequences at once says so here, and the count becomes a PLAN fact: the law
    /// evaluated at one sequence, the same number in every bucket of the graph.
    ///
    /// Off by default, because the capture gates compare our launcher's choice with **prod's** for
    /// the shape prod recorded, and that comparison must keep asking prod's question.
    pub batch_invariant_attn: bool,
    /// **The paged attention's block-table ROW WIDTH**, in pages, when the caller's table is wider
    /// than `max_ctx` needs; 0 derives it from [`RuntimeParams::kv_blocks`].
    ///
    /// The kernels index a sequence's row as `block_table + seq * max_blocks`
    /// (`saber_attn.hip`), so this number and the width the index builder lays each row out at
    /// (`saber_exec::forward`'s `.block_table` law, `StepIndex::max_blocks`) are ONE fact. ws-k14
    /// widened the engine's table to `ceil((max_ctx + max_width + 1) / 16)` pages — 257 at
    /// `max_ctx` 4096 — so a session at the top of its context has a page for the speculative
    /// tail, and left this at `max_ctx / 16` = 256: row `i` of every batched step was then read
    /// `i` entries early, so sequence `i`'s first `i` pages were sequence `i - 1`'s LAST page and
    /// the rest its own pages shifted by `i`. Slot 0 is exact, which is why every one-sequence
    /// gate stayed green; at eight in flight 25 % of served replies answered another request's
    /// prompt (ws-p12). It is a separate field rather than a wider `max_ctx` because `max_ctx`
    /// also feeds the decode split law, and moving it would move the arithmetic.
    pub table_blocks: u32,
    /// Fold `kv_append` into the `qk_norm_rope` before it at DECODE: one
    /// `qk_rope_append_*` dispatch that runs the rope's own impl and then, in the same
    /// workgroup, appends the K rows that workgroup wrote plus their V rows
    /// ([`saber_kernels::glue::rope_append`]). Bit-exact against the two ops by construction and
    /// by `step-tp2 --fusion rope-append`. 16 nodes a step at TP=2. OFF by default (ws-h38).
    pub fuse_rope_append: bool,
    /// Run the decode attention's split-KV combine in the LAST-ARRIVING split block instead of a
    /// second dispatch ([`saber_kernels::saber::attn_decode_declc`]), wherever the split law answers
    /// more than one split. 16 nodes a step at TP=2. OFF by default (ws-h38).
    pub fuse_attn_combine: bool,
    /// Fold an `add_rms_quant` into the reduce before it where the `_nq` epilogue can do both
    /// (see [`Planner::nq_fusion`]).
    ///
    /// **OFF by default since 2026-09-20, and that is a correctness decision rather than a
    /// preference.** ws-h17 gated the fusion bit-exact against the two-op sequence and ws-h19
    /// re-ran the same gate once P4's attention glue was bound: 0 differing bytes became **627,393
    /// arena and 35,040,470 state**, reproducibly, on both ranks. The earlier 0 was not a pass —
    /// the unbound `qk_norm_rope` and `attn_gate_quant` had been feeding **zeros** into the
    /// residual stream, so the gate compared two computations over the same zeros. (A gate whose
    /// inputs are partly zeros has not measured what it names; the same failure as
    /// `CHECKALL 0.00000`, 2026-09-08.)
    ///
    /// And the two-op path is the **reference**, not the alternative: ws-f recorded that `_nq` has
    /// no call site in `ref_allreduce.py`, so prod runs the plain reduce plus the traced
    /// norm+quant, which is what `add_rms_quant` mirrors. The fusion is therefore a **deviation
    /// prod does not run**, and turning it on is a quality decision (GSM8K-class) rather than a
    /// bit-compare — which is exactly why it may not be the default while the difference is
    /// unexplained.
    ///
    /// The route is kept, behind this flag and its which-path counter, because the node saving is
    /// real (`fused_launches` per plan) and the localisation is now precise: the first divergent
    /// buffer is `l0.in_q8`, produced by the norm fused into the **embedding's** reduce.
    ///
    /// **ws-h37 closed the difference**: it was the build flag (`saber_*` is `-ffp-contract=off`,
    /// `add_rms_quant` contracts its sum of squares), and the rules now bind the `_ssqfma`
    /// epilogue, which writes that one line as an explicit fma and is gated bit-exact against the
    /// composition on the card. It stays OFF by default because the serving defaults do.
    pub fuse: bool,
    /// Fold a decode GEMM into the one before it when both read the SAME activation edge and
    /// neither reads the other's output: one `mxfp4_gemm_decode_pair_*` dispatch, each half its
    /// own unpaired descriptor (split count, partials, counter), laid end to end in one grid
    /// ([`saber_kernels::rules::gemm_decode_pair`]). At TP=2 the one site is the GDN's
    /// `in_proj_qkvz` + `in_proj_ba` (48 nodes a step). OFF by default (ws-h37).
    pub fuse_gemm_pair: bool,
    /// Fusion-map ROW 3 (ws-h38): fold `GEMM -> allreduce -> add_rms_quant` into ONE dispatch,
    /// `mxfp4_gemm_decode_arnq_*`, whose split-K last block pushes its tile to the peer and whose
    /// last `M` tile arrivals run the `_nq` epilogue ([`saber_kernels::rules::gemm_decode_ar_nq`]).
    /// Fires exactly where [`Planner::nq_fusion`] would fold the norm into the reduce (the EXACT
    /// family: `M * hidden * 2 < quant_min_bytes`, i.e. the `b1s*` buckets) AND the reduce's one
    /// input is a split decode GEMM's output with no other reader. Takes precedence over the
    /// plain `_nq` fold at those sites. OFF by default; [`PlanOptions::with_decode_fusion`]
    /// includes it.
    pub fuse_gemm_ar: bool,
    /// Fusion-map ROW 4 (ws-h40): fold `gate_up GEMM -> silu_mul_quant` into ONE dispatch,
    /// `mxfp4_gemm_decode_smq_*`, whose last `M` arriving n-tiles each run the producer on one
    /// row ([`saber_kernels::rules::gemm_decode_smq`]). Fires where the gate_up GEMM takes an
    /// unsplit DIRECT decode cell and `silu_mul_quant` is its output's one reader. OFF by
    /// default; [`PlanOptions::with_h40_fusion`] includes it.
    pub fuse_gemm_smq: bool,
    /// Fusion-map ROW 5 (ws-h42): fold `GDN fused update -> gdn_norm_quant` into ONE dispatch,
    /// `saber_gdn_fnq_*`, whose last `M` arriving workgroups each run the norm on one row
    /// ([`saber_kernels::saber::gdn_fused_update_nq`]). Fires where the decode GDN takes the
    /// fused update (one sequence at TP=2: `n_seqs x H <= fused_max_items`) on the f16 stored
    /// state and the norm is the only reader of its `o`. OFF by default;
    /// [`PlanOptions::with_h42_fusion`] includes it.
    pub fuse_gdn_nq: bool,
    /// **CMP27AE** (`--engine-ar-fuse`, the target's decode plans only): fold `GEMM -> allreduce`
    /// into ONE dispatch, `mxfp4_gemm_decode_arw_*`, where the GEMM takes an N 5120 dwn4 prefetch
    /// cell (tm2 / tm4: 17..64 rows) and the reduce is the served wht6 bf16 one-shot in one band
    /// ([`saber_kernels::rules::gemm_decode_arw`]); stands down (two launches) under the race
    /// check. Bit-exact (LEDGER CMP27AB / CMP27AE). Off in the plan constants; the engine's
    /// `--engine-ar-fuse` (DEFAULT on since CMP27AE) sets it on the target's decode plans.
    pub fuse_gemm_wht: bool,
}

/// What a plan does with the all-reduce ops the graph declares.
///
/// This is a property of the CALLER, not of the model: a rank with no peer cannot reduce, and a
/// plan cannot ask a launcher whether it has one. So the caller says, and says it where a reader
/// can see it — a plan built with [`Collectives::Identity`] produces partial sums and is honest
/// about it, and one built with [`Collectives::Real`] will refuse at launch if the launcher turns
/// out to have no peer (`Launcher` refuses an `IpcPeerVisible` workspace by name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collectives {
    /// The output edge ALIASES the input: nothing is launched and every row-parallel output stays a
    /// partial sum. The only correct choice for one rank.
    Identity,
    /// The real one-shot two-rank reduce, in place over the aliased edge, with the family chosen by
    /// the rules' message-size threshold — exact below `quant_min_bytes`, rotated above it.
    Real,
}

impl PlanOptions {
    /// This plan, for rank `r`.
    #[must_use]
    pub const fn rank(mut self, r: u32) -> Self {
        self.rank = r;
        self
    }

    /// This plan, with the all-reduces launched for real.
    #[must_use]
    pub const fn with_collectives(mut self, c: Collectives) -> Self {
        self.collectives = c;
        self
    }

    /// This plan with the head at the plan's own rows, sampled or not. For a measurement harness
    /// comparing the two plans; see [`Self::head_on_every_row`].
    #[must_use]
    pub const fn with_head_on_every_row(mut self) -> Self {
        self.head_on_every_row = true;
        self
    }

    /// This plan with every fusion turned off: one op, one dispatch, the way the graph declares
    /// it. The arena is IDENTICAL either way — a fusion moves a dispatch, not a buffer — which is
    /// what makes a byte comparison between the two a comparison of arithmetic alone.
    ///
    /// Now the default ([`PlanOptions::fuse`]), so this is kept for callers that want to say so
    /// explicitly and for the gate that compares the two paths.
    #[must_use]
    pub const fn without_fusion(mut self) -> Self {
        self.fuse = false;
        self.fuse_gemm_pair = false;
        self.fuse_gemm_ar = false;
        self.fuse_gemm_smq = false;
        self.fuse_gdn_nq = false;
        self.fuse_gemm_wht = false;
        self.fuse_rope_append = false;
        self.fuse_attn_combine = false;
        self
    }

    /// This plan with the decode attention's split count pinned to the one-sequence answer, so a
    /// sequence's tokens are a function of the sequence (see [`PlanOptions::batch_invariant_attn`]).
    #[must_use]
    pub const fn with_batch_invariant_attn(mut self) -> Self {
        self.batch_invariant_attn = true;
        self
    }

    /// This plan with the paged attention reading block-table rows `pages` wide
    /// ([`PlanOptions::table_blocks`]): the width the caller's index builder lays them out at.
    #[must_use]
    pub const fn with_table_blocks(mut self, pages: u32) -> Self {
        self.table_blocks = pages;
        self
    }

    /// This plan with the `_nq` fusion ON (see [`PlanOptions::fuse`]). A caller opts in by name.
    #[must_use]
    pub const fn with_fusion(mut self) -> Self {
        self.fuse = true;
        self
    }

    /// This plan with the decode GEMM pair ON (see [`PlanOptions::fuse_gemm_pair`]).
    #[must_use]
    pub const fn with_gemm_pair(mut self) -> Self {
        self.fuse_gemm_pair = true;
        self
    }

    /// This plan with fusion-map row 4 ON (see [`PlanOptions::fuse_gemm_smq`]).
    #[must_use]
    pub const fn with_gemm_smq(mut self) -> Self {
        self.fuse_gemm_smq = true;
        self
    }

    /// This plan with CMP27AE's GEMM + wht6 fold ON (see [`PlanOptions::fuse_gemm_wht`]).
    #[must_use]
    pub const fn with_gemm_wht(mut self) -> Self {
        self.fuse_gemm_wht = true;
        self
    }

    /// This plan with fusion-map row 5 ON (see [`PlanOptions::fuse_gdn_nq`]).
    #[must_use]
    pub const fn with_gdn_nq(mut self) -> Self {
        self.fuse_gdn_nq = true;
        self
    }

    /// The ws-h42 set: [`Self::with_h40_fusion`] (rows 1-4 + 7) plus row 5
    /// ([`Self::with_gdn_nq`]). What `--decode-fusion h42` serves; opt-in.
    #[must_use]
    pub const fn with_h42_fusion(self) -> Self {
        self.with_h40_fusion().with_gdn_nq()
    }

    /// The ws-h40 set: [`Self::with_decode_fusion`] (rows 1-3 + 7) plus row 4
    /// ([`Self::with_gemm_smq`]). What `--decode-fusion h40` serves; opt-in.
    #[must_use]
    pub const fn with_h40_fusion(self) -> Self {
        self.with_decode_fusion().with_gemm_smq()
    }

    /// This plan with fusion-map row 3 ON (see [`PlanOptions::fuse_gemm_ar`]).
    #[must_use]
    pub const fn with_gemm_ar(mut self) -> Self {
        self.fuse_gemm_ar = true;
        self
    }

    /// This plan with `qk_norm_rope` + `kv_append` in one decode dispatch (see
    /// [`PlanOptions::fuse_rope_append`]).
    #[must_use]
    pub const fn with_rope_append(mut self) -> Self {
        self.fuse_rope_append = true;
        self
    }

    /// This plan with the decode attention's combine in its last-arriving split block (see
    /// [`PlanOptions::fuse_attn_combine`]).
    #[must_use]
    pub const fn with_attn_combine(mut self) -> Self {
        self.fuse_attn_combine = true;
        self
    }

    /// The ws-h37 set alone: the `_nq` epilogue and the decode GEMM pair (fusion-map rows 1-2).
    /// What `--decode-fusion h37` serves; the paired control for the ws-h38 rows.
    #[must_use]
    pub const fn with_h37_fusion(self) -> Self {
        self.with_fusion().with_gemm_pair()
    }

    /// The served launch-train fusions: the `_nq` epilogue and the decode GEMM pair (ws-h37), row
    /// 3's GEMM + reduce + epilogue and the rope + KV append (ws-h38). Each fires only where its
    /// conditions hold, so on a prefill plan this is the plan it was. What `--decode-fusion`
    /// serves.
    ///
    /// NOT row 6 (the last-arriver attention combine, [`Self::with_attn_combine`]): it is
    /// bit-exact but measured SLOWER than the combine kernel it replaces (ws-h38 hold 1,
    /// `step-tp2 --fusion attn-combine`: b1s8 +2.9 ms, b4s8 +0.8 ms net of the head), because
    /// its merge runs serially in one block per (sequence, kv head). It stays selectable by name.
    #[must_use]
    pub const fn with_decode_fusion(self) -> Self {
        self.with_h37_fusion().with_gemm_ar().with_rope_append()
    }
}

impl PlanOptions {
    /// P1's setting: layers only, one buffer per edge, every intermediate readable afterwards.
    pub const BLOCK: Self = PlanOptions {
        rank: 0,
        head_on_every_row: false,
        collectives: Collectives::Identity,
        fuse: false,
        fuse_gemm_pair: false,
        fuse_gemm_ar: false,
        fuse_gemm_smq: false,
        fuse_gdn_nq: false,
        fuse_gemm_wht: false,
        batch_invariant_attn: false,
        table_blocks: 0,
        fuse_rope_append: false,
        fuse_attn_combine: false,
        embedding: false,
        head: false,
        reuse_layer_slots: false,
    };
    /// A forward pass's BODY: the embedding and every layer, one buffer per layer-local edge.
    ///
    /// The head is deliberately not here. It is planned separately at the row count that is
    /// actually SAMPLED — one row for one sequence — because `lm_head` at a 2048-token chunk
    /// would write 2048 x 248320 bf16 logits, 970 MiB, of which the sampler reads one row.
    /// Selecting the rows first is what prod's `LogitsProcessor` does and why the head plugin
    /// has a `row_gather`.
    pub const BODY: Self = PlanOptions {
        rank: 0,
        head_on_every_row: false,
        collectives: Collectives::Identity,
        fuse: false,
        fuse_gemm_pair: false,
        fuse_gemm_ar: false,
        fuse_gemm_smq: false,
        fuse_gdn_nq: false,
        fuse_gemm_wht: false,
        batch_invariant_attn: false,
        table_blocks: 0,
        fuse_rope_append: false,
        fuse_attn_combine: false,
        embedding: true,
        head: false,
        reuse_layer_slots: true,
    };
    /// A forward pass's HEAD: the final norm, `lm_head`, and nothing else.
    pub const HEAD: Self = PlanOptions {
        rank: 0,
        head_on_every_row: false,
        collectives: Collectives::Identity,
        fuse: false,
        fuse_gemm_pair: false,
        fuse_gemm_ar: false,
        fuse_gemm_smq: false,
        fuse_gdn_nq: false,
        fuse_gemm_wht: false,
        batch_invariant_attn: false,
        table_blocks: 0,
        fuse_rope_append: false,
        fuse_attn_combine: false,
        embedding: false,
        head: true,
        reuse_layer_slots: false,
    };
    /// Every site in one plan. Correct, and only affordable at a small row count.
    pub const FORWARD: Self = PlanOptions {
        rank: 0,
        head_on_every_row: false,
        collectives: Collectives::Identity,
        fuse: false,
        fuse_gemm_pair: false,
        fuse_gemm_ar: false,
        fuse_gemm_smq: false,
        fuse_gdn_nq: false,
        fuse_gemm_wht: false,
        batch_invariant_attn: false,
        table_blocks: 0,
        fuse_rope_append: false,
        fuse_attn_combine: false,
        embedding: true,
        head: true,
        reuse_layer_slots: true,
    };
}

/// The layer-local part of an edge or op name: `l7.gdn.qkvz` and `layer.7.gdn.norm.foo` both
/// answer `gdn.qkvz` / `gdn.norm.foo`, and a name with no layer prefix answers `None`.
///
/// This is the whole of the slot-reuse key. It is a function of the graph's naming convention —
/// `push_layer` writes every edge as `l{layer}.…` and every op as `layer.{layer}.…` — so a graph
/// that stopped following it would produce names with no prefix, which get their own buffers and
/// are therefore still correct, only larger.
fn layer_local(name: &str) -> Option<&str> {
    let rest = name
        .strip_prefix("layer.")
        .or_else(|| name.strip_prefix('l'))?;
    let (digits, tail) = rest.split_at(rest.find('.')?);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    tail.strip_prefix('.')
}

/// One decoder block, planned: every buffer, every route, and the launch budget.
#[derive(Debug)]
pub struct BlockPlan<'g> {
    /// The graph it came from.
    pub graph: &'g Graph,
    /// The bucket it is planned at.
    pub bucket: Bucket,
    /// The decoder layers in the block, in order.
    pub layers: Vec<u32>,
    /// The ops, in forward order.
    pub steps: Vec<Step<'g>>,
    /// Every buffer in the one arena, in allocation order.
    pub buffers: Vec<BufferPlan>,
    /// Total arena bytes, buffers aligned to [`ARENA_ALIGN`].
    pub arena_bytes: u64,
    /// Edges the block consumes whose producer is outside it.
    pub inputs: Vec<String>,
    /// Sum of the graph's own `n_launches` over the block's ops.
    pub declared_launches: u32,
    /// Launches the identity all-reduce removes from that sum.
    pub identity_launches: u32,
    /// Launches a FUSION removes from that sum: one per op carried as `Route::Fused` -- the
    /// `add_rms_quant` a `_nq` reduce folds, a pair's second GEMM, and BOTH the reduce and the
    /// norm a row-3 GEMM folds, a folded `kv_append`, and the combine launch a last-arriver attention
    /// drops. `fused_nq + fused_pairs + 2 * fused_gemm_ar + fused_rope_appends + fused_combines`.
    pub fused_launches: u32,
    /// Reduces that carry the `_nq` epilogue, and so the `add_rms_quant` ops that carry
    /// [`Route::Fused`].
    pub fused_nq: u32,
    /// Decode GEMMs carried as the second half of a pair ([`PlanOptions::fuse_gemm_pair`]).
    pub fused_pairs: u32,
    /// Row-parallel decode GEMMs that carry the reduce after them AND that reduce's `add_rms_quant`
    /// (fusion-map row 3, [`PlanOptions::fuse_gemm_ar`]): two folded ops, two launches, each.
    /// A reduce counted here is NOT counted in `fused_nq`, which is the plain `_nq` folds only.
    ///
    /// `fused_launches == fused_nq + fused_pairs + 2 * fused_gemm_ar + fused_rope_appends +
    /// fused_combines` (one launch per folded op).
    pub fused_gemm_ar: u32,
    /// Ops with no ported kernel, by op id and count.
    pub supplied: BTreeMap<&'static str, u32>,
    /// ws-h38: `kv_append` ops carried by the `qk_norm_rope` before them
    /// ([`PlanOptions::fuse_rope_append`]), one launch each.
    pub fused_rope_appends: u32,
    /// ws-h38: decode attentions whose combine runs in the last-arriving split block
    /// ([`PlanOptions::fuse_attn_combine`]): one of the op's two launches each.
    pub fused_combines: u32,
    /// ws-h40: `silu_mul_quant` ops carried by the gate_up decode GEMM before them
    /// ([`PlanOptions::fuse_gemm_smq`], fusion-map row 4), one launch each.
    pub fused_gemm_smq: u32,
    /// ws-h42: `gdn_norm_quant` ops carried by the GDN fused update before them
    /// ([`PlanOptions::fuse_gdn_nq`], fusion-map row 5), one launch each.
    pub fused_gdn_nq: u32,
    /// CMP27AE: wht6 reduces carried by the N 5120 dwn4 decode GEMM before them
    /// ([`PlanOptions::fuse_gemm_wht`]), one launch each.
    pub fused_gemm_wht: u32,
    /// The options it was planned with.
    pub options: PlanOptions,
    /// Every edge name, including the ones that share a buffer, to that buffer's index.
    pub index: BTreeMap<String, usize>,
    /// An all-reduce's output edge -> the input edge it aliases.
    ///
    /// The reduce is in place, so its output has no buffer of its own and every consumer reads the
    /// producer's. [`BlockPlan::locate`] follows this, which is the difference between a caller
    /// asking for `l63.out` and a caller having to know that the bytes are under `l63.mlp.partial`.
    pub aliases: BTreeMap<String, String>,
    /// The `q_len` this plan's prefill attention descriptors currently carry — the bucket's row
    /// count as planned, the CHUNK's after [`BlockPlan::retarget_prefill_rows`].
    ///
    /// It is a field rather than a re-derivation because two readers have to agree on it: the
    /// kernarg, and `seqused_k` (`saber_exec::forward::AttnQLen::Exact`). A step that moves one
    /// without the other gets a mask that is right-aligned on a row count the sequence does not
    /// have — which is ws-l15's garbage in one direction and ws-l16's arena leak in the other.
    prefill_q_len: u32,
    /// GDN state slots the caches this plan binds were sized for
    /// ([`RuntimeParams::gdn_slots`]). Kept so that
    /// [`BlockPlan::retarget_prefill_state`] can refuse a slot past the pool instead of writing
    /// 1.5 MiB per layer past the end of it.
    gdn_slots: u32,
    /// The GDN state slot this plan's `chunk_scan` currently reads `h0` from and writes `ht` to,
    /// or `None` while it is [`StateAt::PerStep`] — which is how every prefill plan is born.
    prefill_state_slot: Option<u32>,
}

impl<'g> BlockPlan<'g> {
    /// Plan `layers` of `graph` at `bucket`. Pure: no GPU, no file, no environment.
    ///
    /// # Errors
    /// A geometry the kernel plane refuses (an unaligned K, a batch the fused GDN kernel cannot
    /// serve), an edge with no producer and no block input, or an op whose buffers this module
    /// does not know how to bind.
    pub fn new(
        graph: &'g Graph,
        bucket: Bucket,
        layers: &[u32],
        params: RuntimeParams,
        mxfp4: &LaunchRules,
        saber_rules: &SaberRules,
    ) -> Result<Self> {
        Self::with_options(
            graph,
            bucket,
            layers,
            params,
            mxfp4,
            saber_rules,
            PlanOptions::BLOCK,
        )
    }

    /// Plan `layers` of `graph` at `bucket`, with the sites and the arena policy `opts` states.
    ///
    /// The head plugin's launch rules are read here rather than passed in, unlike the MXFP4 and
    /// saber ones. They carry no band edges and no tile ladder — one workgroup per output row,
    /// and a width the kernels are compiled for — so there is no variant of them a caller could
    /// legitimately want to substitute, and `head::resolve` refuses the one knob that matters.
    ///
    /// # Errors
    /// Everything [`BlockPlan::new`] refuses, plus a slot-reuse hazard: with
    /// `reuse_layer_slots` on, an edge that is read after the layer that shares its buffer has
    /// overwritten it.
    pub fn with_options(
        graph: &'g Graph,
        bucket: Bucket,
        layers: &[u32],
        params: RuntimeParams,
        mxfp4: &LaunchRules,
        saber_rules: &SaberRules,
        opts: PlanOptions,
    ) -> Result<Self> {
        let m = bucket.m();
        // The head site in a plan whose row count is not the sampled row count would compute
        // logits for every token and throw all but one row away: 2048 x 248320 at a p2048 chunk,
        // 970 MiB of a GEMM nobody reads. It is not a slow path, it is the wrong answer to a
        // question the caller did not mean to ask, so it is refused by name rather than planned.
        //
        // `sampled_rows == m` for EVERY decode bucket, so this never fires on a decode plan; it
        // fires on a prefill chunk longer than its sequence count, where the body and the head
        // belong to two plans joined by a row selection (`forward::Plans`, `forward::Head`).
        if opts.head && !opts.head_on_every_row && bucket.sampled_rows() < m {
            let sampled = bucket.sampled_rows();
            return Err(Error::Refused(format!(
                "one plan cannot hold the body at {m} rows and the head at {sampled} ({bucket:?}). \
                 The head runs on the SAMPLED rows, so planning it here would compute a shard of \
                 logits per token and read one row of them. Plan the body with `PlanOptions::BODY` \
                 and the head with `PlanOptions::HEAD` at `Graph::head_bucket(bucket)`; \
                 `saber_exec::forward::Plans` does both and launches the row selection between \
                 them. A harness that means to MEASURE the every-row head says \
                 `.with_head_on_every_row()`."
            )));
        }
        let want: BTreeSet<u32> = layers.iter().copied().collect();
        let ops: Vec<&ModelOp> = graph
            .ops_for(bucket)
            .into_iter()
            .map(|r| r.op)
            .filter(|op| match op.site {
                Site::Layer(n) => want.contains(&n),
                Site::Embedding => opts.embedding,
                Site::Head => opts.head,
                Site::Drafter => false,
            })
            .collect();
        if ops.is_empty() {
            return Err(Error::Refused(format!(
                "no ops for layers {layers:?} at bucket {bucket:?}"
            )));
        }

        let layouts: BTreeMap<String, u32> = graph
            .device_final_records()
            .map_err(|e| Error::Refused(format!("device-final record plan: {e}")))?
            .into_iter()
            .map(|r| (r.name, r.layout_id))
            .collect();
        let mut plan = Planner {
            m,
            law: ATiledLaw::of(mxfp4),
            bucket,
            params,
            graph,
            buffers: Vec::new(),
            index: BTreeMap::new(),
            cursor: 0,
            aliases: BTreeMap::new(),
            layouts,
            head: HeadRules::prod_head()
                .map_err(|e| Error::Refused(format!("head launch rules: {e}")))?,
            glue: GlueRules::prod()
                .map_err(|e| Error::Refused(format!("attn_glue launch rules: {e}")))?,
            reuse: opts.reuse_layer_slots,
            rank: opts.rank,
            collectives: opts.collectives,
            fuse: opts.fuse,
            fuse_gemm_pair: opts.fuse_gemm_pair,
            fuse_gemm_ar: opts.fuse_gemm_ar,
            fuse_gemm_smq: opts.fuse_gemm_smq,
            fuse_gdn_nq: opts.fuse_gdn_nq,
            fuse_gemm_wht: opts.fuse_gemm_wht,
            batch_invariant_attn: opts.batch_invariant_attn,
            table_blocks: opts.table_blocks,
            fuse_rope_append: opts.fuse_rope_append,
            fuse_attn_combine: opts.fuse_attn_combine,
            slots: BTreeMap::new(),
        };
        // Pass 1: the arena. Every op's outputs, then its private intermediates, in forward order,
        // so an address is a function of the plan and not of the order the caller asks in.
        for op in &ops {
            plan.plan_op_buffers(op)?;
        }
        // Pass 2: block inputs -- an edge some op reads that nothing in the block writes.
        let mut inputs = Vec::new();
        for op in &ops {
            for input in &op.inputs {
                if plan.lookup(&input.edge).is_none() {
                    let bytes = plan.edge_bytes_for_consumer(op, &input.edge)?;
                    // Only the residual stream enters a block from outside, and it is row-major
                    // `[rows, hidden]` -- the one case `bytes / rows` was always right for.
                    let per = bytes / u64::from(plan.m.max(1));
                    plan.push(
                        input.edge.clone(),
                        BufferKind::Input,
                        bytes,
                        Layout::Bf16,
                        RowSpan::Contiguous { bytes: per },
                        "<block input>",
                    );
                    inputs.push(input.edge.clone());
                }
            }
        }

        // How many ops READ each edge. The `_nq` fusion needs this: the fused kernel never
        // writes the reduced value to memory, so it is only legal when the reduce's output edge
        // has exactly one consumer -- the norm being folded into it.
        let mut consumers: BTreeMap<&str, usize> = BTreeMap::new();
        for op in &ops {
            for input in &op.inputs {
                *consumers.entry(input.edge.as_str()).or_default() += 1;
            }
        }

        // Pass 3: routes and bindings.
        let mut steps: Vec<Step<'g>> = Vec::new();
        let mut declared = 0u32;
        let mut identity = 0u32;
        let mut fused = 0u32;
        let mut fused_nq = 0u32;
        let mut fused_pairs = 0u32;
        let mut fused_gemm_ar = 0u32;
        let mut fused_rope_appends = 0u32;
        let mut fused_combines = 0u32;
        let mut fused_gemm_smq = 0u32;
        let mut fused_gdn_nq = 0u32;
        let mut fused_gemm_wht = 0u32;
        let mut supplied: BTreeMap<&'static str, u32> = BTreeMap::new();
        // The ops an earlier step folded into itself, IN ORDER, so each launches nothing, and
        // which fusion did it. A queue because row 3 folds the next TWO ops.
        let mut folded_into: std::collections::VecDeque<(String, FoldKind)> =
            std::collections::VecDeque::new();
        for (i, op) in ops.iter().enumerate() {
            // The all-reduce's declaration is a function of the bucket's rows (it is cut into
            // row bands above the IPC scratch slot), so the budget asks the OP, not the kernel.
            if let Some(n) = op.n_launches_at(bucket.m()) {
                declared += n;
            }
            if let Some((into, kind)) = folded_into.pop_front() {
                fused += op.n_launches_at(bucket.m()).unwrap_or(0);
                let reason = match kind {
                    FoldKind::Nq => {
                        fused_nq += 1;
                        "the reduce before it carries the `_nq` epilogue: one dispatch does the \
                         exact all-reduce, this op's residual add, its Gemma RMS norm and its \
                         per-token e4m3 quant"
                    }
                    FoldKind::GemmPair => {
                        fused_pairs += 1;
                        "the GEMM before it reads the same activation and carries this one as the \
                         second half of a decode pair: one dispatch, each half its own unpaired \
                         descriptor"
                    }
                    FoldKind::GemmAr => {
                        fused_gemm_ar += 1;
                        "the row-parallel decode GEMM before it carries this reduce (fusion-map \
                         row 3): its split-K last blocks push their tiles to the peer and its \
                         last M arrivals run the exact reduce"
                    }
                    FoldKind::GemmArNorm => {
                        "the row-parallel decode GEMM two ops before it carries the reduce and \
                         this op (fusion-map row 3): its row owners run the residual add, the \
                         Gemma RMS norm and the per-token e4m3 quant"
                    }
                    FoldKind::RopeAppend => {
                        fused_rope_appends += 1;
                        "the rope before it appends this layer's K and V rows in the same \
                         workgroup that roped them: one dispatch, the two ops' own arithmetic"
                    }
                    FoldKind::GdnNq => {
                        fused_gdn_nq += 1;
                        "the GDN fused update before it carries this norm (fusion-map row 5): its \
                         last M arriving workgroups each run the gated RMSNorm and the per-token \
                         e4m3 quant on one row"
                    }
                    FoldKind::GemmWht => {
                        fused_gemm_wht += 1;
                        "the N 5120 dwn4 decode GEMM before it carries this rotated reduce \
                         (CMP27AE): each tile's last split-K arriver quantizes its rows' rotation \
                         groups, pushes them to the peer and stores the reduced rows"
                    }
                    FoldKind::GemmSmq => {
                        fused_gemm_smq += 1;
                        "the gate_up decode GEMM before it carries this producer (fusion-map row \
                         4): its last M arriving n-tiles each run silu * up and the per-token e4m3 \
                         quant on one row"
                    }
                };
                let route = Route::Fused { into, reason };
                let bindings = plan.bindings(op, &route, saber_rules.ar.max_bytes)?;
                steps.push(Step {
                    op,
                    route,
                    bindings,
                });
                continue;
            }
            // A reduce whose consumer is the norm right after it becomes ONE op.
            let mut fusion = None;
            if op.kernel == KernelOp::AllReduce && plan.fuse {
                if let Some(next) = ops.get(i + 1) {
                    fusion = plan
                        .nq_fusion(op, next, &consumers, mxfp4, saber_rules)?
                        .map(|f| (f, vec![FoldKind::Nq]));
                }
            }
            if op.kernel == KernelOp::GemmW4A8 {
                // Row 3 first: where it fires, the GEMM carries the reduce and its norm.
                if let (Some(ar), Some(norm)) = (ops.get(i + 1), ops.get(i + 2)) {
                    fusion = plan
                        .gemm_ar_fusion(op, ar, norm, &consumers, mxfp4, saber_rules)?
                        .map(|f| (f, vec![FoldKind::GemmAr, FoldKind::GemmArNorm]));
                }
                if fusion.is_none() {
                    if let Some(next) = ops.get(i + 1) {
                        fusion = plan
                            .gemm_wht_fusion(op, next, &consumers, mxfp4, saber_rules)?
                            .map(|f| (f, vec![FoldKind::GemmWht]));
                    }
                }
                if fusion.is_none() {
                    if let Some(next) = ops.get(i + 1) {
                        fusion = plan
                            .smq_fusion(op, next, &consumers, mxfp4, saber_rules)?
                            .map(|f| (f, vec![FoldKind::GemmSmq]));
                    }
                }
                if fusion.is_none() {
                    if let Some(next) = ops.get(i + 1) {
                        fusion = plan
                            .pair_fusion(op, next, mxfp4, saber_rules)?
                            .map(|f| (f, vec![FoldKind::GemmPair]));
                    }
                }
            }
            if op.kernel == KernelOp::GdnFusedUpdate {
                if let Some(next) = ops.get(i + 1) {
                    fusion = plan
                        .gdn_nq_fusion(op, next, &consumers, mxfp4, saber_rules)?
                        .map(|f| (f, vec![FoldKind::GdnNq]));
                }
            }
            if op.kernel == KernelOp::QkNormRope {
                if let Some(next) = ops.get(i + 1) {
                    fusion = plan
                        .rope_append_fusion(op, next, mxfp4, saber_rules)?
                        .map(|f| (f, vec![FoldKind::RopeAppend]));
                }
            }
            let (route, bindings) = match fusion {
                Some(((desc, bindings), kinds)) => {
                    // Row 3's regions are keyed by N as well (the launcher keys Persistent and
                    // peer-visible workspaces by `(plugin, op, shape)`): its arrival counter's
                    // arithmetic needs every call of one region to have the same n-tile count.
                    // Row 5 (ws-h42) keys its region by the grid (`num_wg`), for the same reason.
                    let key = match kinds[0] {
                        FoldKind::GemmAr | FoldKind::GemmSmq => Some("n"),
                        FoldKind::GdnNq => Some("num_wg"),
                        _ => None,
                    };
                    let shape = if let Some(key) = key {
                        let n = match desc.arg(key).map(|s| &s.value) {
                            Some(saber_kernels::rules::ArgValue::I32 { ival }) => {
                                u32::try_from(*ival).unwrap_or(0)
                            }
                            _ => 0,
                        };
                        Shape::new(0, n, 0)
                    } else {
                        Shape::new(0, 0, 0)
                    };
                    for kind in kinds {
                        folded_into.push_back((op.name.clone(), kind));
                    }
                    (
                        Route::Kernel {
                            descs: vec![desc],
                            shape,
                        },
                        bindings,
                    )
                }
                None => {
                    let mut route = plan.route(op, mxfp4, saber_rules)?;
                    // ws-h38: the decode attention's pair becomes one dispatch whose last-arriving
                    // split block runs the combine -- one of the op's two declared launches.
                    if let Some(desc) = plan.combine_fusion(op, saber_rules)? {
                        fused += 1;
                        fused_combines += 1;
                        route = Route::Kernel {
                            descs: vec![desc],
                            shape: Shape::new(0, 0, 0),
                        };
                    }
                    match &route {
                        Route::Identity { .. } => {
                            identity += op.n_launches_at(bucket.m()).unwrap_or(0);
                        }
                        Route::Supplied { .. } => {
                            *supplied.entry(op.kernel.id()).or_default() += 1;
                        }
                        Route::Kernel { .. } | Route::Fused { .. } => {}
                    }
                    let bindings = plan.bindings(op, &route, saber_rules.ar.max_bytes)?;
                    (route, bindings)
                }
            };
            steps.push(Step {
                op,
                route,
                bindings,
            });
        }
        if let Some((into, _)) = folded_into.pop_front() {
            return Err(Error::Refused(format!(
                "{into}: a fusion folded the op after it, but the block ends there"
            )));
        }

        let arena_bytes = plan.cursor;
        let out = BlockPlan {
            graph,
            bucket,
            layers: layers.to_vec(),
            steps,
            buffers: plan.buffers,
            arena_bytes,
            inputs,
            declared_launches: declared,
            identity_launches: identity,
            fused_launches: fused,
            fused_nq,
            fused_pairs,
            fused_gemm_ar,
            supplied,
            fused_rope_appends,
            fused_combines,
            fused_gemm_smq,
            fused_gdn_nq,
            fused_gemm_wht,
            options: opts,
            index: plan.index,
            aliases: plan.aliases,
            prefill_q_len: bucket.q_len(),
            gdn_slots: params.gdn_slots,
            prefill_state_slot: None,
        };
        out.check_slot_reuse()?;
        Ok(out)
    }

    /// **C7: the arena buffers this plan READS before any of its own steps wrote them** — the
    /// buffers whose contents a run takes from the PREVIOUS run over the same arena. A plan whose
    /// list is non-empty cannot share its arena with another plan unless each such buffer is
    /// seeded (or is inert) at every run: over a shared arena, "the previous run" is another rung's,
    /// laid out differently. Reads and writes are told apart as [`Self::check_slot_reuse`] does
    /// (an op's declared outputs, and those of the ops fused into it, are its writes).
    #[must_use]
    pub fn reads_before_write(&self) -> Vec<(String, String)> {
        let mut written: BTreeSet<usize> = BTreeSet::new();
        let mut out: Vec<(String, String)> = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            let carried: Vec<_> = self.steps[i + 1..]
                .iter()
                .take_while(
                    |n| matches!(&n.route, Route::Fused { into, .. } if *into == step.op.name),
                )
                .map(|n| n.op)
                .collect();
            let mut reads: Vec<(usize, &str)> = Vec::new();
            let mut writes: Vec<usize> = Vec::new();
            for (_, src) in &step.bindings {
                let Source::Arena { buffer, .. } = src else {
                    continue;
                };
                let Some(&b) = self.index.get(buffer.as_str()) else {
                    continue;
                };
                let is_output = std::iter::once(step.op)
                    .chain(carried.iter().copied())
                    .any(|op| {
                        op.outputs.iter().any(|e| {
                            self.index.get(&e.name) == Some(&b)
                                || self.index.get(&format!("{}.scale", e.name)) == Some(&b)
                        })
                    })
                    // T7: an op's PRIVATE intermediate (`conv_prep.gates`, `qk_norm_rope.k`,
                    // `fused_update.qkv`) is written by the op that owns it before any read of
                    // it -- the kernel's own store, which the graph does not declare as an edge.
                    // A CONSTANT (`.zero_resid`) is private too but no op writes it: it stays a
                    // read, and only the executor's re-seed ([`ConstantSeeds`]) covers it.
                    || self.buffers.get(b).is_some_and(|bp| {
                        bp.kind == BufferKind::Internal
                            && !bp.name.ends_with(ZERO_CONSTANT_SUFFIX)
                            && std::iter::once(step.op)
                                .chain(carried.iter().copied())
                                .any(|op| bp.producer == op.name)
                    });
                if is_output {
                    writes.push(b);
                } else {
                    reads.push((b, buffer.as_str()));
                }
            }
            for (b, name) in reads {
                if !written.contains(&b) && !out.iter().any(|(n, _)| n == name) {
                    out.push((name.to_owned(), step.op.name.to_string()));
                }
            }
            written.extend(writes);
        }
        out
    }

    /// **H46: [`Self::reads_before_write`] minus the plan's own INPUTS** (`self.inputs`, the edges
    /// a caller writes into the arena before every run — a head's gathered hidden state and
    /// residual). What is left is a value the plan takes from the arena's PREVIOUS run, which over
    /// a shared arena is another plan's; empty is the condition for sharing without re-seeding.
    ///
    /// **T7: and minus the plan's CONSTANTS, because the executor re-seeds every one of them before
    /// EVERY run** ([`ConstantSeeds::EveryRun`], [`BlockExecutor::zero_constants`]). This is the
    /// law's form over every plan the engine builds -- prefill rungs, decode buckets, row bands,
    /// heads -- and the boot refuses a plan it leaves non-empty.
    #[must_use]
    pub fn stale_reads(&self) -> Vec<(String, String)> {
        self.stale_reads_under(ConstantSeeds::EveryRun)
    }

    /// [`Self::stale_reads`] under a given re-seed policy. [`ConstantSeeds::SharedArenaOnly`] is
    /// the pre-T7 rule (C7 re-seeded the constants over the SHARED prefill arena only), kept as
    /// the law's failing control: under it every decode bucket (an own arena) carries layer 0's
    /// `.zero_resid`, the buffer T7's boot poison named.
    #[must_use]
    pub fn stale_reads_under(&self, seeds: ConstantSeeds) -> Vec<(String, String)> {
        let seeded = |buf: &str| {
            buf.ends_with(ZERO_CONSTANT_SUFFIX)
                && match seeds {
                    ConstantSeeds::EveryRun => true,
                    ConstantSeeds::SharedArenaOnly { shared } => shared,
                }
        };
        self.reads_before_write()
            .into_iter()
            .filter(|(buf, _)| !self.inputs.iter().any(|i| i == buf) && !seeded(buf))
            .collect()
    }

    /// Refuse a plan whose shared buffers are read after something else has overwritten them.
    ///
    /// Walks the steps in order keeping, per arena buffer, the last step that WROTE it, and
    /// refuses an op that reads a buffer whose last writer is neither in its own layer nor the
    /// op that produced the value it is supposed to be reading. Written as a check rather than
    /// an argument because the safety of slot reuse is a property of the GRAPH — one that holds
    /// today because `push_layer` hands exactly two edges forward — and a graph change that
    /// broke it would otherwise show up as a wrong answer 40 layers later.
    fn check_slot_reuse(&self) -> Result<()> {
        if !self.options.reuse_layer_slots {
            return Ok(());
        }
        // buffer index -> (step that wrote it, the op's site)
        let mut writer: BTreeMap<usize, (usize, Site)> = BTreeMap::new();
        for (i, step) in self.steps.iter().enumerate() {
            let site = step.op.site;
            for (_, src) in &step.bindings {
                let Source::Arena { buffer, .. } = src else {
                    continue;
                };
                let Some(&b) = self.index.get(buffer.as_str()) else {
                    continue;
                };
                // An op's own outputs are writes; everything else it names is a read. The
                // binding table does not label them, so the op's declared outputs do -- and a
                // step that carries the op after it (a fusion: that op is `Route::Fused` into
                // this one) writes THAT op's outputs too.
                // Row 3 folds TWO ops (the reduce and its norm), so every consecutive step after
                // this one that is fused into it counts.
                let carried = self.steps[i + 1..]
                    .iter()
                    .take_while(
                        |n| matches!(&n.route, Route::Fused { into, .. } if *into == step.op.name),
                    )
                    .map(|n| n.op);
                let is_output = std::iter::once(step.op).chain(carried).any(|op| {
                    op.outputs.iter().any(|e| {
                        self.index.get(&e.name) == Some(&b)
                            || self.index.get(&format!("{}.scale", e.name)) == Some(&b)
                    })
                });
                if is_output {
                    writer.insert(b, (i, site));
                    continue;
                }
                if let Some(&(w, wsite)) = writer.get(&b) {
                    // A read of a buffer last written by a DIFFERENT site is only legal when
                    // that site is the immediately preceding layer -- the residual stream and
                    // the block output. Anything else means two layers' live ranges overlap.
                    let ok = wsite == site
                        || match (wsite, site) {
                            (Site::Embedding, Site::Layer(_)) => true,
                            (Site::Layer(a), Site::Layer(b2)) => a + 1 == b2,
                            (Site::Layer(_), Site::Head) => true,
                            _ => false,
                        };
                    if !ok {
                        return Err(Error::Refused(format!(
                            "slot reuse is unsafe here: `{}` (step {i}, {:?}) reads arena buffer                              `{buffer}`, last written by step {w} at {wsite:?}. The graph handed                              an edge across more than one layer boundary, which this arena                              policy does not model.",
                            step.op.name, site
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The `q_len` this plan's prefill attention descriptors carry, which `seqused_k` must agree
    /// with ([`crate::forward::AttnQLen::Exact`]).
    #[must_use]
    pub fn prefill_q_len(&self) -> u32 {
        self.prefill_q_len
    }

    /// Point this plan's prefill attention at the rows a CHUNK actually feeds.
    ///
    /// **The law (ws-h27).** `saber_attn_prefill` reads one `q_len` kernarg three ways — the row
    /// stride into `q`/`out`, the live predicate (`live = qpos < q_len`, and a row past it
    /// duplicates row `q_len - 1` and stores nothing), and the right-alignment of the causal mask
    /// (`klimit = ctx - q_len + qpos`). Planning it as the BUCKET's row count made every padded
    /// row a live query row computed out of two leftovers — a query the GDN ops never wrote,
    /// because `cu` stops at the live rows, and keys past the ones the sequence has written — and
    /// the online-softmax rescale is guarded by `wave_any`, so with GQA 6 the last live row of a
    /// chunk and the first padded row share a wave and the padded row decides when the live one
    /// rescales (ws-l16, which answered it with 1664 memsets a run).
    ///
    /// Passing the chunk's live rows closes it at the source: `klimit` becomes `done_before +
    /// qpos` for every live row, the rows the grid still covers past the chunk duplicate the last
    /// LIVE row rather than reading the arena, and the grid itself shrinks to
    /// `ceil(live / prefill_bq)` CTAs. Nothing then reads a row no token feeds, so the answer is a
    /// function of the request and not of what ran before it.
    ///
    /// Two properties this relies on, both checked by the callers' own laws:
    ///
    /// * **the row stride is only read at `seq`**, and a short chunk is legal only as the step's
    ///   ONLY live sequence (`StepIndex::check`) — so `seq` is 0, `seq * q_len` is 0, and no
    ///   address moves. With several live sequences every chunk is full and this is a no-op.
    /// * **`seqused_k` moves with it**: the index law reports `done_before + q_len`, and
    ///   `AttnQLen::Exact(plan.prefill_q_len())` is how a step says which number that is.
    ///
    /// Idempotent and re-entrant: the descriptor is always re-derived from the BUCKET's row count,
    /// never from whatever the previous chunk left it at.
    ///
    /// # Errors
    /// A decode plan, a `live_rows` outside `1..=bucket.q_len()`, or a prefill plan with no
    /// `attn_prefill` step at all — the last one loudly, because a plan whose attention op was
    /// renamed or fused would otherwise go on padding the way it did before this existed.
    pub fn retarget_prefill_rows(&mut self, live_rows: u32, rules: &SaberRules) -> Result<u32> {
        if self.bucket.phase() != Phase::Prefill {
            return Err(Error::Refused(format!(
                "retarget_prefill_rows on a {:?} plan: a decode step's `q_len` is its verify \
                 width and is also the `sidx` block width",
                self.bucket
            )));
        }
        let q = self.bucket.q_len();
        if live_rows == 0 || live_rows > q {
            return Err(Error::Refused(format!(
                "retarget_prefill_rows: {live_rows} live row(s) in a {q}-row bucket; a chunk is \
                 padded UP into its bucket, never truncated"
            )));
        }
        let mut moved = 0u32;
        for step in &mut self.steps {
            if step.op.kernel != KernelOp::AttnPrefillPaged {
                continue;
            }
            let Route::Kernel { descs, .. } = &mut step.route else {
                continue;
            };
            for desc in descs.iter_mut() {
                saber::retarget_attn_prefill(desc, live_rows, q, rules)
                    .map_err(|e| Error::Refused(format!("{}: {e}", step.op.name)))?;
            }
            moved += 1;
        }
        if moved == 0 {
            return Err(Error::Refused(format!(
                "retarget_prefill_rows: {:?} has no `attn_prefill` step to point at the chunk's \
                 {live_rows} live row(s); the padding would steer the live rows' softmax again \
                 and nothing would say so",
                self.bucket
            )));
        }
        self.prefill_q_len = live_rows;
        Ok(moved)
    }

    /// Point this prefill plan's recurrent state at the step's SESSION slot (ws-h30).
    ///
    /// `saber_gdn_chunk_scan` reads `h0` and writes `ht` at `(nq * H + hv) * VD * KD`, where `nq` is
    /// `blockIdx.z` — the sequence's index in the STEP. Every other GDN kernel finds a session's
    /// state through `cache_idx` or `sidx`; this one cannot, so the BASE it is handed carries the
    /// slot. That is a fact about the step and not about the plan's tuple, which is why it moves
    /// after the plan is built, exactly like [`BlockPlan::retarget_prefill_rows`] — and why it is
    /// legal at all: a prefill chunk is eager, so its kernargs are packed per step.
    ///
    /// A step's live sequences therefore have to hold **consecutive** slots from `slot`, because
    /// `nq` walks them by one. `saber_exec::forward::StepIndex::check` refuses a prefill step
    /// whose slots are not, and `StepIndex::prefill_state_slot` is where a caller gets this
    /// number from, so the two ends of the law cannot drift apart.
    ///
    /// Idempotent and re-entrant: the offset is always re-derived from the state's own
    /// `bytes_per_slot`, never from whatever the last chunk left it at.
    ///
    /// # Errors
    /// A decode plan; slot 0, which is the kernels' null slot and which a plan that now writes
    /// `ht` in place would POISON for every padding row that reads it; a slot whose sequences run
    /// past the pool the plan was sized for; a `chunk_scan` whose recurrent state does not state
    /// its `bytes_per_slot`; and — loudly — a prefill plan with no `chunk_scan` step at all, so
    /// that an op renamed or fused cannot go back to reading the null slot in silence.
    pub fn retarget_prefill_state(&mut self, slot: u32) -> Result<u32> {
        if slot == 0 {
            return Err(Error::Refused(
                "retarget_prefill_state(0): slot 0 is the kernels' null slot, the one every \
                 padding row is pointed at. A prefill publishes its final state in place, so a \
                 chunk scanned there would write what padding reads"
                    .to_owned(),
            ));
        }
        self.point_prefill_state(slot)
    }

    /// The rewrite both entry points share: everything except whether slot 0 is legal.
    fn point_prefill_state(&mut self, slot: u32) -> Result<u32> {
        let seqs = self.bucket.num_seqs().max(1);
        if u64::from(slot) + u64::from(seqs) > u64::from(self.gdn_slots) {
            return Err(Error::Refused(format!(
                "retarget_prefill_state({slot}): this bucket runs {seqs} sequence(s) from that \
                 slot and the caches were sized for {} slot(s)",
                self.gdn_slots
            )));
        }
        let moved = self.rewrite_prefill_state(Some(slot))?;
        self.prefill_state_slot = Some(slot);
        Ok(moved)
    }

    /// Rewrite every `chunk_scan`'s `h0`/`ht` binding: `Some(slot)` points it there,
    /// `None` puts it back to [`StateAt::PerStep`]. The one place the binding moves.
    fn rewrite_prefill_state(&mut self, slot: Option<u32>) -> Result<u32> {
        if self.bucket.phase() != Phase::Prefill {
            return Err(Error::Refused(format!(
                "retarget_prefill_state on a {:?} plan: a decode step finds its own slots through \
                 `sidx`, one per candidate ROW, and pointing its state buffer at one would move \
                 every snapshot it takes",
                self.bucket
            )));
        }
        let mut moved = 0u32;
        for step in &mut self.steps {
            if step.op.kernel != KernelOp::GdnChunkScan {
                continue;
            }
            let at = match slot {
                None => StateAt::PerStep,
                Some(slot) => {
                    let rec = step
                        .op
                        .state
                        .iter()
                        .find(|s| s.name.ends_with("recurrent_state"))
                        .ok_or_else(|| {
                            Error::Refused(format!("{}: no `recurrent_state` state", step.op.name))
                        })?;
                    let per_slot = rec.bytes_per_slot.ok_or_else(|| {
                        Error::Refused(format!(
                            "{}: `{}` states no `bytes_per_slot`, so a slot has no offset",
                            step.op.name, rec.name
                        ))
                    })?;
                    StateAt::Slot {
                        slot,
                        offset: u64::from(slot) * per_slot,
                    }
                }
            };
            for (name, src) in &mut step.bindings {
                if name != "h0" && name != "ht" {
                    continue;
                }
                let Source::State { key, .. } = src else {
                    return Err(Error::Refused(format!(
                        "{}: `{name}` is not bound to a state",
                        step.op.name
                    )));
                };
                *src = Source::State {
                    key: key.clone(),
                    at,
                };
            }
            moved += 1;
        }
        if moved == 0 {
            return Err(Error::Refused(format!(
                "retarget_prefill_state: {:?} has no `gdn_chunk_scan` step to point at a slot; \
                 the recurrent state would stay at the pool's base, which is the null slot, and \
                 the prompt would be forgotten between chunks with nothing to say so",
                self.bucket
            )));
        }
        Ok(moved)
    }

    /// Point the scan at slot 0 — the base of the state buffer the caller bound — for a
    /// REFERENCE-CAPTURE harness only.
    ///
    /// [`BlockPlan::retarget_prefill_state`] refuses slot 0 by name, because a serving caller has
    /// a session POOL whose slot 0 is the kernels' null slot and an in-place `ht` written there
    /// would be read by every padding row. `the internal reference harness` has no pool: it calls
    /// The reference's own `gdn_chunk_scan` with `h0 = state[:batch]`, sequence 0 at offset 0, so the
    /// REFERENCE's own layout puts this step's state at slot 0 and there is nothing there to
    /// poison. `saber layer` and `saber block-capture` are the two callers, and they say so by
    /// calling a differently-named method rather than by passing a 0 the other law rejects.
    ///
    /// # Errors
    /// As [`BlockPlan::retarget_prefill_state`], minus the null-slot refusal.
    pub fn retarget_prefill_state_at_reference_base(&mut self) -> Result<u32> {
        self.point_prefill_state(0)
    }

    /// Un-point the scan: the binding goes back to [`StateAt::PerStep`], which
    /// [`state_address`] refuses.
    ///
    /// For a caller that has to BIND a prefill plan for something other than RUNNING it. The
    /// engine's startup sizes every workspace and agrees every collective digest over every plan
    /// it can run, and a state ADDRESS is an input to neither — so it points each prefill plan at
    /// a slot, binds, and puts it back, and a step that then forgets to point it is still refused
    /// rather than quietly scanning through slot 1.
    ///
    /// # Errors
    /// A decode plan, or a prefill plan with no `gdn_chunk_scan` step — the same refusals
    /// [`BlockPlan::retarget_prefill_state`] makes, for the same reason.
    pub fn clear_prefill_state(&mut self) -> Result<u32> {
        let moved = self.rewrite_prefill_state(None)?;
        self.prefill_state_slot = None;
        Ok(moved)
    }

    /// The GDN state slot this plan's `chunk_scan` is pointed at, or `None` while no step has.
    #[must_use]
    pub fn prefill_state_slot(&self) -> Option<u32> {
        self.prefill_state_slot
    }

    /// The arena buffer an edge resolves to, following the identity all-reduce's aliases.
    #[must_use]
    pub fn locate(&self, edge: &str) -> Option<&BufferPlan> {
        let name = self.aliases.get(edge).map_or(edge, String::as_str);
        self.index.get(name).map(|i| &self.buffers[*i])
    }

    /// Dispatches the block will issue: the declared budget minus what identity replaced.
    #[must_use]
    pub fn expected_launches(&self) -> u32 {
        self.declared_launches - self.identity_launches - self.fused_launches
    }

    /// A planned buffer by name.
    #[must_use]
    pub fn buffer(&self, name: &str) -> Option<&BufferPlan> {
        self.buffers.iter().find(|b| b.name == name)
    }

    /// Bytes by buffer kind, for the plan table.
    #[must_use]
    pub fn bytes_by_kind(&self) -> BTreeMap<&'static str, u64> {
        let mut out = BTreeMap::new();
        for b in &self.buffers {
            let key = match b.kind {
                BufferKind::EdgeData => "edge",
                BufferKind::EdgeScale => "edge.scale",
                BufferKind::Internal => "internal",
                BufferKind::Input => "input",
            };
            *out.entry(key).or_default() += b.bytes;
        }
        out
    }

    /// Every distinct weight key the block binds.
    #[must_use]
    pub fn weight_keys(&self) -> BTreeSet<String> {
        self.keys(|s| match s {
            Source::Weight { key, .. } => Some(key.clone()),
            _ => None,
        })
    }

    /// Every distinct state name the block binds.
    #[must_use]
    pub fn state_keys(&self) -> BTreeSet<String> {
        self.keys(|s| match s {
            Source::State { key, .. } => Some(key.clone()),
            _ => None,
        })
    }

    /// Every distinct index-table key the block binds.
    #[must_use]
    pub fn index_keys(&self) -> BTreeSet<String> {
        self.keys(|s| match s {
            Source::Index(k) => Some(k.clone()),
            _ => None,
        })
    }

    fn keys(&self, f: impl Fn(&Source) -> Option<String>) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for step in &self.steps {
            for (_, src) in &step.bindings {
                if let Some(k) = f(src) {
                    out.insert(k);
                }
            }
        }
        out
    }
}

// =====================================================================================
// The planner
// =====================================================================================

struct Planner<'g> {
    m: u32,
    /// The layout law of the mxfp4 rules this plan resolves against (G6): what every fused
    /// producer writes at `m`, so the edge the planner declares, the producer's kernel and the
    /// GEMM that reads it answer ONE question.
    law: ATiledLaw,
    bucket: Bucket,
    params: RuntimeParams,
    graph: &'g Graph,
    buffers: Vec<BufferPlan>,
    index: BTreeMap<String, usize>,
    cursor: u64,
    /// Identity all-reduce: output edge -> the input edge it aliases.
    aliases: BTreeMap<String, String>,
    /// Pack record name -> the layout id the model's device-final plan gives it.
    layouts: BTreeMap<String, u32>,
    /// The head plugin's launch rules (embedding lookup, row gather, argmax).
    head: HeadRules,
    /// The attention glue's launch rules (q/k norm + rope, the gate quant, the K/V append).
    glue: GlueRules,
    /// Share one buffer per layer-local edge name across the plan's layers.
    reuse: bool,
    /// This rank, for the vocab-parallel embedding's window.
    rank: u32,
    /// What to do with `layer.allreduce`.
    collectives: Collectives,
    /// Whether an `add_rms_quant` may be folded into the reduce before it.
    fuse: bool,
    /// [`PlanOptions::fuse_gemm_pair`].
    fuse_gemm_pair: bool,
    /// [`PlanOptions::fuse_gemm_ar`].
    fuse_gemm_ar: bool,
    /// [`PlanOptions::fuse_gemm_smq`].
    fuse_gemm_smq: bool,
    /// [`PlanOptions::fuse_gdn_nq`].
    fuse_gdn_nq: bool,
    /// [`PlanOptions::fuse_gemm_wht`].
    fuse_gemm_wht: bool,
    /// Whether the decode attention's split count is the law's one-sequence answer
    /// ([`PlanOptions::batch_invariant_attn`]).
    batch_invariant_attn: bool,
    /// The block table's row width, or 0 for `params.kv_blocks` ([`PlanOptions::table_blocks`]).
    table_blocks: u32,
    /// [`PlanOptions::fuse_rope_append`].
    fuse_rope_append: bool,
    /// [`PlanOptions::fuse_attn_combine`].
    fuse_attn_combine: bool,
    /// (layer-local name, bytes, layout) -> the buffer that serves it.
    slots: BTreeMap<(String, u64, String), usize>,
}

/// Does this producer's resolved kernel write the fragment-tiled A layout at `m`?
///
/// A thin name over [`producer_writes_tiled`], which answers `None` for a non-producer: reaching
/// that arm here would mean a producer route was asked about a GEMM, so it is a refusal by
/// panic-free default rather than a silent `false`.
fn writes_tiled(op: GemmOp, m: u32, rules: &LaunchRules) -> bool {
    producer_writes_tiled(op, m, rules).unwrap_or(false)
}

/// Whether `op`'s GDN recurrent state is STORED as f16 (C1). Read off the op's own
/// [`saber_model::graph::StateRef`], which [`saber_model::graph::Graph::with_gdn_state_f16`] is the
/// one writer of, so the route and the arena cannot disagree about the width.
fn stores_f16_state(op: &ModelOp) -> bool {
    op.state.iter().any(|s| s.layout == Layout::F16State)
}

/// Bytes an element of `layout` occupies.
fn elem_bytes(layout: Layout) -> u64 {
    match layout {
        Layout::Bf16 => 2,
        // Int8 is the Paro tier's A side; this block is the NVFP4 tier, but the width is the
        // same one byte and naming it keeps the match exhaustive rather than defaulted.
        Layout::RowMajorFp8 | Layout::ATiledFp8 | Layout::Int8 => 1,
        Layout::Fp32State => 4,
        Layout::F16State => 2,
    }
}

impl<'g> Planner<'g> {
    fn push(
        &mut self,
        name: String,
        kind: BufferKind,
        bytes: u64,
        layout: Layout,
        rows: RowSpan,
        producer: &str,
    ) -> u64 {
        // Slot reuse: the same layer-local name at the same size, layout and ROW LAYOUT is the
        // same buffer. Keyed on all three, not on the name alone, so a graph where two layers
        // gave one name two shapes would allocate two buffers rather than silently truncate one
        // of them -- and so that a reader taking a row out of a shared slot cannot be handed the
        // other occupant's row law (ws-l18).
        if self.reuse {
            if let Some(local) = layer_local(&name) {
                let key = (local.to_owned(), bytes, format!("{layout:?}/{rows:?}"));
                if let Some(&i) = self.slots.get(&key) {
                    self.index.insert(name, i);
                    return self.buffers[i].offset;
                }
                let offset = self.push_owned(name, kind, bytes, layout, rows, producer);
                self.slots.insert(key, self.buffers.len() - 1);
                return offset;
            }
        }
        self.push_owned(name, kind, bytes, layout, rows, producer)
    }

    fn push_owned(
        &mut self,
        name: String,
        kind: BufferKind,
        bytes: u64,
        layout: Layout,
        rows: RowSpan,
        producer: &str,
    ) -> u64 {
        let offset = self.cursor;
        self.cursor += align_up(bytes.max(1));
        self.index.insert(name.clone(), self.buffers.len());
        self.buffers.push(BufferPlan {
            name,
            kind,
            offset,
            bytes,
            layout,
            rows,
            producer: producer.to_owned(),
        });
        offset
    }

    fn lookup(&self, name: &str) -> Option<&BufferPlan> {
        let name = self.aliases.get(name).map_or(name, String::as_str);
        self.index.get(name).map(|i| &self.buffers[*i])
    }

    fn at(&self, name: &str) -> Result<Source> {
        let b = self.lookup(name).ok_or_else(|| {
            Error::Refused(format!("`{name}` is not a planned buffer of this block"))
        })?;
        Ok(Source::Arena {
            buffer: b.name.clone(),
            offset: b.offset,
            bytes: b.bytes,
        })
    }

    /// A byte range INSIDE a planned buffer: the q/k/v thirds of a GDN conv output, the two halves
    /// of the gate pair, the `z` slice of a merged `in_proj` output.
    fn at_offset(&self, name: &str, offset: u64, bytes: u64) -> Result<Source> {
        let b = self.lookup(name).ok_or_else(|| {
            Error::Refused(format!("`{name}` is not a planned buffer of this block"))
        })?;
        if offset + bytes > b.bytes {
            return Err(Error::Refused(format!(
                "`{name}` sub-range [{offset}, {}) runs past its {} bytes",
                offset + bytes,
                b.bytes
            )));
        }
        Ok(Source::Arena {
            buffer: b.name.clone(),
            offset: b.offset + offset,
            bytes,
        })
    }

    fn scale_of(&self, edge: &str) -> Result<Source> {
        self.at(&format!("{edge}.scale"))
    }

    /// The hidden size an all-reduce's message is `rows` of. The graph states it
    /// (`Geometry::AllReduce::elems_per_token`) and this module never guesses it: the band law
    /// needs the ROW, and a row derived from the arena's byte count alone would be a division by
    /// whatever padding the buffer carries.
    fn ar_elems_per_row(&self, op: &ModelOp) -> Result<u32> {
        match op.geometry {
            Geometry::AllReduce {
                elems_per_token, ..
            } => Ok(elems_per_token),
            _ => Err(Error::Refused(format!(
                "{}: an all-reduce with no AllReduce geometry",
                op.name
            ))),
        }
    }

    /// How many rows the message carries, checked against the planned buffer.
    ///
    /// A refusal rather than a rounding: the band law cuts on rows, so a buffer that is not a
    /// whole number of them would put a band boundary inside a row and hand the two ranks
    /// different element counts.
    fn ar_rows(&self, op: &ModelOp, edge: &str, bytes: u64, elem: u64) -> Result<u64> {
        let row_bytes = u64::from(self.ar_elems_per_row(op)?) * elem;
        if row_bytes == 0 || bytes % row_bytes != 0 {
            return Err(Error::Refused(format!(
                "{}: `{edge}` is {bytes} bytes, which is not a whole number of {row_bytes}-byte                  rows",
                op.name
            )));
        }
        Ok(bytes / row_bytes)
    }

    /// Columns an op's output edge `idx` carries, and whether the edge has an activation scale.
    fn edge_cols(&self, op: &ModelOp, idx: usize) -> Result<u32> {
        let cols = match (op.kernel, &op.geometry, idx) {
            (
                KernelOp::GemmW4A8
                | KernelOp::GemmBf16
                | KernelOp::GemmBf16Head
                | KernelOp::GemmFp8Head,
                Geometry::Gemm { n, .. },
                0,
            ) => *n,
            (KernelOp::AddRmsQuant | KernelOp::AddRmsBf16, Geometry::Rms { k }, _) => *k,
            (KernelOp::SiluMulQuant, Geometry::SiluMul { n }, 0) => *n,
            (KernelOp::GdnNormQuant, Geometry::GdnNorm { n, .. }, 0) => *n,
            (
                KernelOp::GdnFusedUpdate
                | KernelOp::GdnRecurrentUpdate
                | KernelOp::GdnTreeUpdate
                | KernelOp::GdnChunkScan,
                Geometry::GdnCore { heads, head_v, .. },
                0,
            ) => heads * head_v,
            (
                KernelOp::GdnConvUpdate | KernelOp::GdnConvTree | KernelOp::GdnConvPrep,
                Geometry::GdnCore {
                    heads,
                    head_v,
                    key_heads,
                    head_k,
                    ..
                },
                0,
            ) => 2 * key_heads * head_k + heads * head_v,
            // `A` is [T, H, chunk]: the graph carries the prefill token block, so the width of
            // the chunked inverse factor is not a constant restated here.
            (KernelOp::GdnKktSolve, Geometry::GdnCore { heads, chunk, .. }, 0) => heads * chunk,
            (
                KernelOp::AttnPrefillPaged
                | KernelOp::AttnDecodePaged
                | KernelOp::AttnDecodeTree
                | KernelOp::QkNormRope,
                Geometry::Attn {
                    q_heads, head_dim, ..
                }
                | Geometry::QkNorm {
                    q_heads, head_dim, ..
                },
                0,
            ) => q_heads * head_dim,
            (KernelOp::AttnGateQuant, Geometry::GateQuant { n, .. }, 0) => *n,
            // The vocabulary all-gather: `n` is one rank's shard and the output is every rank's,
            // concatenated. Nothing on this branch launches it — the caller gathers the shards
            // itself — but the edge still has to be a size, because the buffer the caller writes
            // into is the plan's.
            (KernelOp::LogitsGather, Geometry::Gather { n }, 0) => *n * self.graph.tp,
            (KernelOp::Embedding, Geometry::Embedding { hidden, .. }, 0) => *hidden,
            (
                KernelOp::AllReduce,
                Geometry::AllReduce {
                    elems_per_token, ..
                },
                0,
            ) => *elems_per_token,
            (kernel, geom, idx) => {
                return Err(Error::Refused(format!(
                    "{}: output {idx} of a {kernel} with geometry {geom:?} has no width law here",
                    op.name
                )))
            }
        };
        Ok(cols)
    }

    /// The byte size a consumer implies for an edge nothing in the block produces.
    fn edge_bytes_for_consumer(&self, op: &ModelOp, edge: &str) -> Result<u64> {
        // Only the residual stream enters a block from outside, and it is bf16 `hidden`.
        let _ = edge;
        match op.geometry {
            Geometry::Rms { k } => Ok(u64::from(self.m) * u64::from(k) * 2),
            _ => Err(Error::Refused(format!(
                "{}: edge `{edge}` has no producer in this block and no size law",
                op.name
            ))),
        }
    }

    /// `q`, `k` and `v` of a GDN conv output, as (offset, bytes) inside the conv buffer.
    fn qkv_regions(&self, geom: &Geometry) -> Result<[(u64, u64); 3]> {
        let Geometry::GdnCore {
            heads,
            head_v,
            key_heads,
            head_k,
            ..
        } = geom
        else {
            return Err(Error::Refused("qkv_regions on a non-GDN geometry".into()));
        };
        let t = u64::from(self.m);
        let kb = t * u64::from(*key_heads) * u64::from(*head_k) * 2;
        let vb = t * u64::from(*heads) * u64::from(*head_v) * 2;
        Ok([(0, kb), (kb, kb), (2 * kb, vb)])
    }

    /// The conv q|k|v buffer's ROW law: three row-major blocks, one per tensor.
    fn qkv_row_span(&self, geom: &Geometry) -> Result<RowSpan> {
        let [(qo, kb), (ko, _), (vo, vb)] = self.qkv_regions(geom)?;
        let t = u64::from(self.m.max(1));
        Ok(RowSpan::Blocks {
            blocks: vec![(qo, kb / t), (ko, kb / t), (vo, vb / t)],
        })
    }

    /// Where row `r` of an op's output edge `idx` lives.
    ///
    /// Every edge but one is row-major `[rows, cols]`. The exception is the GDN conv's own output,
    /// which the graph names "convolved q/k/v" and which IS three tensors end to end — see
    /// [`RowSpan`].
    fn edge_rows(&self, op: &ModelOp, idx: usize, layout: Layout, cols: u32) -> Result<RowSpan> {
        // The fused update's own EDGE is the delta-rule core output and row-major; it writes the
        // q|k|v buffer to a PRIVATE intermediate, which is given the same span where it is sized.
        if idx == 0
            && matches!(
                op.kernel,
                KernelOp::GdnConvUpdate | KernelOp::GdnConvTree | KernelOp::GdnConvPrep
            )
        {
            return self.qkv_row_span(&op.geometry);
        }
        Ok(RowSpan::Contiguous {
            bytes: u64::from(cols) * elem_bytes(layout),
        })
    }

    fn plan_op_buffers(&mut self, op: &ModelOp) -> Result<()> {
        // The identity all-reduce aliases its input edge instead of allocating an output.
        if op.kernel == KernelOp::AllReduce {
            let from = op.inputs[0].edge.clone();
            let to = op.outputs[0].name.clone();
            self.aliases.insert(to, from);
            return Ok(());
        }
        for (i, edge) in op.outputs.iter().enumerate() {
            let layout = edge.rule.resolve_under(self.m, self.law);
            let cols = self.edge_cols(op, i)?;
            // The A-tiled layout pads to whole 16-row fragments; nothing else does.
            let rows = if layout == Layout::ATiledFp8 {
                u64::from(self.m).div_ceil(16) * 16
            } else {
                u64::from(self.m)
            };
            let bytes = rows * u64::from(cols) * elem_bytes(layout);
            let span = self.edge_rows(op, i, layout, cols)?;
            self.push(
                edge.name.clone(),
                BufferKind::EdgeData,
                bytes,
                layout,
                span,
                &op.name,
            );
            if matches!(layout, Layout::RowMajorFp8 | Layout::ATiledFp8) {
                self.push(
                    format!("{}.scale", edge.name),
                    BufferKind::EdgeScale,
                    u64::from(self.m) * 4,
                    layout,
                    RowSpan::Contiguous { bytes: 4 },
                    &op.name,
                );
            }
        }
        // Op-private intermediates.
        let t = u64::from(self.m);
        match op.kernel {
            KernelOp::GdnFusedUpdate => {
                let [(_, kb), _, (_, vb)] = self.qkv_regions(&op.geometry)?;
                // The SAME three blocks the conv edge is: the fused kernel writes q, k and v with
                // the pair's own indexing into its private buffer.
                self.push(
                    format!("{}.qkv", op.name),
                    BufferKind::Internal,
                    2 * kb + vb,
                    Layout::Bf16,
                    self.qkv_row_span(&op.geometry)?,
                    &op.name,
                );
            }
            KernelOp::QkNormRope => {
                let Geometry::QkNorm {
                    head_dim, kv_heads, ..
                } = op.geometry
                else {
                    return Err(Error::Refused(
                        "qk_norm_rope without a QkNorm geometry".into(),
                    ));
                };
                // The kernel writes q and k to SEPARATE tensors; the graph declares only the q
                // edge, because q is what attention reads and k is consumed by `kv_append`
                // inside the same layer. It is an op-private intermediate for exactly that
                // reason, sized like any other.
                self.push(
                    format!("{}.k", op.name),
                    BufferKind::Internal,
                    t * u64::from(kv_heads) * u64::from(head_dim) * 2,
                    Layout::Bf16,
                    RowSpan::Contiguous {
                        bytes: u64::from(kv_heads) * u64::from(head_dim) * 2,
                    },
                    &op.name,
                );
            }
            KernelOp::GdnConvPrep => {
                let Geometry::GdnCore { heads, .. } = op.geometry else {
                    return Err(Error::Refused("conv_prep without a GDN geometry".into()));
                };
                // `g` and `beta`, fp32 [T, H] each, consumed by kkt_solve and chunk_scan. TWO
                // row-major blocks in one buffer, like the conv edge and for the same reason: the
                // kernel writes them as two tensors.
                let per = u64::from(heads) * 4;
                self.push(
                    format!("{}.gates", op.name),
                    BufferKind::Internal,
                    2 * t * per,
                    Layout::Fp32State,
                    RowSpan::Blocks {
                        blocks: vec![(0, per), (t * per, per)],
                    },
                    &op.name,
                );
            }
            KernelOp::AddRmsQuant if op.inputs.len() == 1 => {
                // Layer 0 has no residual to add. Prod runs a no-residual instantiation of the
                // same op; the ported kernel takes a residual pointer, so the block hands it an
                // explicit zero buffer and `res_out` is then the hidden state itself.
                let Geometry::Rms { k } = op.geometry else {
                    return Err(Error::Refused("add_rms without an Rms geometry".into()));
                };
                self.push(
                    format!("{}.zero_resid", op.name),
                    BufferKind::Internal,
                    t * u64::from(k) * 2,
                    Layout::Bf16,
                    RowSpan::Contiguous {
                        bytes: u64::from(k) * 2,
                    },
                    &op.name,
                );
            }
            _ => {}
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------- fusion
    /// The `_nq` fusion for a reduce and the `add_rms_quant` right after it, or `None` when this
    /// pair does not qualify.
    ///
    /// `saber_ar_oneshot_2rank_exact_nq_bf16` is the exact one-shot reduce with the decoder's
    /// post-AR epilogue folded into it: residual add, Gemma RMS norm, per-token `e4m3` quant, one
    /// workgroup per row. Taking it deletes a whole dispatch per reduce — at b1s5 that is 128–129
    /// of the step's nodes — but it is only *correct* under five conditions, and every one of them
    /// is checked here rather than assumed:
    ///
    /// 1. **The rules picked the exact family.** The message size decides that (`quant_min_bytes`),
    ///    so the fusion is a property of the bucket, not of the model: above the threshold the
    ///    reduce is the rotated kernel and there is no `_nq` arm of it.
    /// 2. **The reduced value has exactly one reader.** The fused kernel keeps the sum in
    ///    registers and never writes it back, so the reduce's output edge — which aliases its
    ///    input — is left holding this rank's PARTIAL. That is invisible only while the norm being
    ///    folded in is its one consumer.
    /// 3. **The norm's own kernel is the one whose arithmetic the epilogue reproduces**:
    ///    `add_rms_quant_t256_tiled0`, 256 threads, row-major `q`. The epilogue's reduction tree
    ///    is the same shape at the same block width, so the two agree bit for bit — at `t512`
    ///    they would not (a different partial-sum order), and `tiled1` writes the fragment-tiled
    ///    layout the epilogue does not have.
    /// 4. **The row geometry is the epilogue's**: `K` a multiple of `k_multiple`, within the
    ///    register budget `max_k`, and one block per row within `max_blocks`.
    /// 5. **The norm's residual and outputs are bindable here** — the same sources the unfused op
    ///    would bind, including the explicit zero residual layer 0 uses.
    ///
    /// Anything else returns `None` and the pair stays two ops. That is a route choice, not a
    /// failure, which is why the count of fused reduces is a field on the plan
    /// ([`BlockPlan::fused_nq`]) and a launch-path counter
    /// (`LaunchCounts::ar_fused_nq` vs `ar_plain`).
    fn nq_fusion(
        &self,
        ar: &ModelOp,
        norm: &ModelOp,
        consumers: &BTreeMap<&str, usize>,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        // Row 3 ([`Self::gemm_ar_fusion`]) asks this same question for its reduce and norm, so
        // either flag opens it; the plain `_nq` call site checks `fuse` itself.
        if !(self.fuse || self.fuse_gemm_ar)
            || self.collectives != Collectives::Real
            || norm.kernel != KernelOp::AddRmsQuant
        {
            return Ok(None);
        }
        // ---- 2: the shape of the pair, and the one-reader rule ------------------------------
        let (Some(out), 1) = (ar.outputs.first(), ar.outputs.len()) else {
            return Ok(None);
        };
        let Some(in_edge) = ar.inputs.first().map(|i| i.edge.as_str()) else {
            return Ok(None);
        };
        if norm.inputs.first().map(|i| i.edge.as_str()) != Some(out.name.as_str()) {
            return Ok(None);
        }
        // The output edge aliases the input edge, so BOTH names address the buffer the fused
        // kernel leaves holding a partial: the reduce is the input's only reader and the norm is
        // the output's only reader, or the value is read after it stopped being the sum.
        if consumers.get(in_edge) != Some(&1) || consumers.get(out.name.as_str()) != Some(&1) {
            return Ok(None);
        }
        // ---- 1: the family is the rules' own answer for this message ------------------------
        let bytes = self
            .lookup(in_edge)
            .ok_or_else(|| Error::Refused(format!("{}: no buffer for `{in_edge}`", ar.name)))?
            .bytes;
        if bytes % 2 != 0 {
            return Ok(None);
        }
        // A hybrid profile keeps small host-native collectives fused and routes
        // larger messages to RCCL, whose external call cannot carry this tail.
        if saber_kernels::saber::uses_rccl(rules, bytes) {
            return Ok(None);
        }
        // A BANDED message cannot carry the epilogue (H28). A plain reduce above the IPC scratch
        // slot is cut into row bands, one dispatch each; the `_nq` epilogue is one dispatch that
        // keeps the sum in registers and never writes it, so there is nothing to hand the next
        // band. It would have been refused downstream anyway — `ar_exact_nq`'s grid is one block
        // per row and it stops at `ar.max_blocks` = 512 rows — but the size question is asked
        // here, where the answer is "no fusion" rather than "no plan".
        if bytes > rules.ar.max_bytes {
            return Ok(None);
        }
        let plain = saber::resolve(
            saber::Op::AllReduce {
                dtype: saber::ArPayload::Bf16,
            },
            saber::Shape::Ar(saber::ArShape {
                n_elem: bytes / 2,
                elem_bytes: 2,
                world_size: self.graph.tp,
            }),
            rules,
        )
        .map_err(|e| Error::Refused(format!("{}: {e}", ar.name)))?;
        if plain.op != "allreduce_exact" {
            return Ok(None);
        }
        // ---- 3: the norm's kernel is the arithmetic the epilogue reproduces ------------------
        let Geometry::Rms { k } = norm.geometry else {
            return Ok(None);
        };
        let rows = u64::from(self.m);
        if rows == 0 || bytes != rows * u64::from(k) * 2 {
            return Ok(None);
        }
        let unfused = resolve(
            GemmOp::AddRmsQuant {
                eps: self.graph.config.rms_norm_eps,
            },
            Shape::new(self.m, 0, k),
            mxfp4,
        )
        .map_err(|e| Error::Refused(format!("{}: {e}", norm.name)))?;
        if unfused.block[0] != rules.ar_nq.block || !unfused.kernel.ends_with("_tiled0") {
            return Ok(None);
        }
        // ---- 4: the epilogue's own geometry -------------------------------------------------
        let desc = match saber::resolve(
            saber::Op::AllReduceExactNq,
            saber::Shape::ArNq(saber::ArNqShape {
                m_rows: self.m,
                k_cols: k,
                eps_bits: self.graph.config.rms_norm_eps.to_bits(),
                world_size: self.graph.tp,
            }),
            rules,
        ) {
            Ok(d) => d,
            // A shape the epilogue is not compiled for. The pair stays two ops.
            Err(_) => return Ok(None),
        };
        // ---- 5: the bindings, from BOTH ops -------------------------------------------------
        let resid = match norm.inputs.get(1) {
            Some(res) => self.at(&res.edge)?,
            None => self.at(&format!("{}.zero_resid", norm.name))?,
        };
        let q_out = norm
            .outputs
            .first()
            .ok_or_else(|| Error::Refused(format!("{}: no quantized output", norm.name)))?;
        let res_out = norm
            .outputs
            .get(1)
            .ok_or_else(|| Error::Refused(format!("{}: no residual output", norm.name)))?;
        let bindings = vec![
            ("in".to_owned(), self.at(in_edge)?),
            ("residual".to_owned(), resid),
            (
                "norm_w".to_owned(),
                self.layer_weight(norm, WeightRole::NormWeight)?,
            ),
            ("q_out".to_owned(), self.at(&q_out.name)?),
            ("scale_out".to_owned(), self.scale_of(&q_out.name)?),
            ("res_out".to_owned(), self.at(&res_out.name)?),
        ];
        // The same coverage check the binding table ends with: a descriptor slot this does not
        // name would be packed from whatever the staging buffer last held.
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let known = bindings.iter().any(|(n, _)| n == buffer)
                || desc.workspaces.iter().any(|w| &w.name == buffer);
            if !known {
                return Err(Error::Refused(format!(
                    "{}: the `_nq` epilogue names buffer `{buffer}`, which the fusion does not bind",
                    ar.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// Fusion-map ROW 3 (ws-h38): the decode GEMM `gemm`, the reduce `ar` right after it, and the
    /// `add_rms_quant` `norm` right after that, as ONE dispatch -- or `None` when the triple does
    /// not qualify, and the ops plan as they would have.
    ///
    /// The reduce + norm half is [`Self::nq_fusion`]'s own answer, so the exactness decision (the
    /// rules' plain reduce is `allreduce_exact`: `M * hidden * 2 < quant_min_bytes`, unbanded),
    /// the one-reader rule on the reduce's edge (which is the GEMM's output, left holding this
    /// rank's PARTIAL), the norm kernel's arithmetic and the binding set are all decided there
    /// and not re-derived. Row 3 adds only: the reduce's one input is this GEMM's one output, the
    /// GEMM resolves to a split decode cell, and [`saber_kernels::rules::gemm_decode_ar_nq`]
    /// accepts the two descriptors (compiled cell, `N` whole n-tiles, `M <= N / 128`).
    ///
    /// A banded or rotated (wht) reduce is never fused here: `nq_fusion` refuses it, because the
    /// rotated kernel quantises over 64-element groups with a per-group amax and a rotation, and
    /// a message above the IPC slot is cut into row bands -- the group would have to be whole in
    /// one block, which a 128-column tile is not in general.
    fn gemm_ar_fusion(
        &self,
        gemm: &ModelOp,
        ar: &ModelOp,
        norm: &ModelOp,
        consumers: &BTreeMap<&str, usize>,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        if !self.fuse_gemm_ar
            || gemm.kernel != KernelOp::GemmW4A8
            || ar.kernel != KernelOp::AllReduce
        {
            return Ok(None);
        }
        let ([out], [input]) = (&gemm.outputs[..], &ar.inputs[..]) else {
            return Ok(None);
        };
        if input.edge != out.name {
            return Ok(None);
        }
        let Some((nq_desc, nq_bindings)) = self.nq_fusion(ar, norm, consumers, mxfp4, rules)?
        else {
            return Ok(None);
        };
        let route = self.route(gemm, mxfp4, rules)?;
        let Route::Kernel { descs, shape } = &route else {
            return Ok(None);
        };
        let [g] = &descs[..] else {
            return Ok(None);
        };
        let Ok(desc) = saber_kernels::rules::gemm_decode_ar_nq((g, *shape), &nq_desc, mxfp4) else {
            return Ok(None);
        };
        let mut bindings = self.bindings(gemm, &route, rules.ar.max_bytes)?;
        let c = bindings
            .iter()
            .find(|(n, _)| n == "c")
            .map(|(_, s)| s.clone())
            .ok_or_else(|| Error::Refused(format!("{}: the GEMM binds no `c`", gemm.name)))?;
        for (name, src) in nq_bindings {
            if name == "in" {
                // The reduce's local input IS the GEMM's output: one buffer, bound once as `c`.
                if src != c {
                    return Err(Error::Refused(format!(
                        "{}: row 3 found the reduce reading a different buffer than the GEMM \
                         writes",
                        ar.name
                    )));
                }
                continue;
            }
            bindings.push((name, src));
        }
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let known = bindings.iter().any(|(n, _)| n == buffer)
                || desc.workspaces.iter().any(|w| &w.name == buffer);
            if !known {
                return Err(Error::Refused(format!(
                    "{}: row 3 names buffer `{buffer}`, which the fusion does not bind",
                    gemm.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// **CMP27AE**: the N 5120 dwn4 decode GEMM `gemm` and the rotated reduce `ar` right after it
    /// as one dispatch ([`saber_kernels::rules::gemm_decode_arw`]). Fires only when the reduce
    /// reads the GEMM's one output, resolves to ONE band of the served wht6 bf16 one-shot (or its
    /// wave-parallel body), and the race check is off (the fold has no checked twin: the checked
    /// run keeps the two launches); anything else is `None`.
    fn gemm_wht_fusion(
        &self,
        gemm: &ModelOp,
        ar: &ModelOp,
        consumers: &BTreeMap<&str, usize>,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        if rules.ar.external_rccl
            || !self.fuse_gemm_wht
            || rules.ar.race_check
            || self.collectives != Collectives::Real
            || self.bucket.phase() != Phase::Decode
            || gemm.kernel != KernelOp::GemmW4A8
            || ar.kernel != KernelOp::AllReduce
        {
            return Ok(None);
        }
        let ([out], [input]) = (&gemm.outputs[..], &ar.inputs[..]) else {
            return Ok(None);
        };
        // The fold never writes the GEMM's pre-reduce value: legal only when the reduce is the
        // GEMM output's one reader.
        if input.edge != out.name || consumers.get(out.name.as_str()).copied() != Some(1) {
            return Ok(None);
        }
        let route = self.route(gemm, mxfp4, rules)?;
        let Route::Kernel { descs, shape } = &route else {
            return Ok(None);
        };
        let [g] = &descs[..] else {
            return Ok(None);
        };
        if !g.kernel.ends_with("_dwn4") {
            return Ok(None);
        }
        let ar_route = self.route(ar, mxfp4, rules)?;
        let Route::Kernel {
            descs: ar_descs, ..
        } = &ar_route
        else {
            return Ok(None);
        };
        let [a] = &ar_descs[..] else {
            return Ok(None);
        };
        let Ok(desc) = saber_kernels::rules::gemm_decode_arw((g, *shape), a, rules) else {
            return Ok(None);
        };
        let mut bindings = self.bindings(gemm, &route, rules.ar.max_bytes)?;
        let c = bindings
            .iter()
            .find(|(n, _)| n == "c")
            .map(|(_, s)| s.clone())
            .ok_or_else(|| Error::Refused(format!("{}: the GEMM binds no `c`", gemm.name)))?;
        for (name, src) in self.bindings(ar, &ar_route, rules.ar.max_bytes)? {
            if name == "in" || name == "out" {
                // The reduce is in place over the GEMM's output: one buffer, bound once as `c`.
                if src != c {
                    return Err(Error::Refused(format!(
                        "{}: the GEMM + wht6 fold found the reduce on a different buffer than \
                         the GEMM writes",
                        ar.name
                    )));
                }
                continue;
            }
            if !bindings.iter().any(|(n, _)| *n == name) {
                bindings.push((name, src));
            }
        }
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let known = bindings.iter().any(|(n, _)| n == buffer)
                || desc.workspaces.iter().any(|w| &w.name == buffer);
            if !known {
                return Err(Error::Refused(format!(
                    "{}: the GEMM + wht6 fold names buffer `{buffer}`, which it does not bind",
                    gemm.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// Fusion-map ROW 5 (ws-h42): the decode GDN fused update `fu` and the `gdn_norm_quant`
    /// `norm` right after it as one dispatch ([`saber_kernels::saber::gdn_fused_update_nq`]). Fires
    /// only when the norm reads the update's `o` edge and is its ONLY reader, and when the update
    /// resolves to the compiled f16-state instantiation; anything else is `None` and the two stay
    /// two dispatches.
    fn gdn_nq_fusion(
        &self,
        fu: &ModelOp,
        norm: &ModelOp,
        consumers: &BTreeMap<&str, usize>,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        if !self.fuse_gdn_nq
            || self.bucket.phase() != Phase::Decode
            || fu.kernel != KernelOp::GdnFusedUpdate
            || norm.kernel != KernelOp::GdnNormQuant
        {
            return Ok(None);
        }
        let ([out], Some(input)) = (&fu.outputs[..], norm.inputs.first()) else {
            return Ok(None);
        };
        if input.edge != out.name || consumers.get(out.name.as_str()).copied() != Some(1) {
            return Ok(None);
        }
        let route = self.route(fu, mxfp4, rules)?;
        let Route::Kernel { descs, .. } = &route else {
            return Ok(None);
        };
        let [f] = &descs[..] else {
            return Ok(None);
        };
        let norm_route = self.route(norm, mxfp4, rules)?;
        let Route::Kernel { descs: ndescs, .. } = &norm_route else {
            return Ok(None);
        };
        let [nd] = &ndescs[..] else {
            return Ok(None);
        };
        let Ok(desc) = saber_kernels::saber::gdn_fused_update_nq(f, nd, i64::from(nd.grid[0]))
        else {
            return Ok(None);
        };
        let mut bindings = self.bindings(fu, &route, rules.ar.max_bytes)?;
        let o = bindings
            .iter()
            .find(|(n, _)| n == "o")
            .map(|(_, s)| s.clone())
            .ok_or_else(|| Error::Refused(format!("{}: the update binds no `o`", fu.name)))?;
        for (name, src) in self.bindings(norm, &norm_route, rules.ar.max_bytes)? {
            let renamed = match name.as_str() {
                "x" => {
                    if src != o {
                        return Err(Error::Refused(format!(
                            "{}: row 5 found the norm reading a different buffer than the \
                             update writes",
                            norm.name
                        )));
                    }
                    continue;
                }
                "z" => "nq_z",
                "w" => "nq_w",
                "q" => "nq_q",
                "scale" => "nq_scale",
                other => {
                    return Err(Error::Refused(format!(
                        "{}: row 5 does not know the norm's binding `{other}`",
                        norm.name
                    )))
                }
            };
            bindings.push((renamed.to_owned(), src));
        }
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let known = bindings.iter().any(|(n, _)| n == buffer)
                || desc.workspaces.iter().any(|w| &w.name == buffer);
            if !known {
                return Err(Error::Refused(format!(
                    "{}: row 5 names buffer `{buffer}`, which the fusion does not bind",
                    fu.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// Fusion-map ROW 4 (ws-h40): the gate_up decode GEMM `gemm` and the `silu_mul_quant`
    /// `silu` right after it as one dispatch ([`saber_kernels::rules::gemm_decode_smq`]). Fires
    /// only when the producer reads the GEMM's one output edge and is its ONLY reader (the
    /// gate_up row is still written, but nothing else may be ordered after the owners' read of
    /// it), and when the GEMM resolves to a compiled unsplit DIRECT cell; anything else is `None`
    /// and the two stay two dispatches.
    fn smq_fusion(
        &self,
        gemm: &ModelOp,
        silu: &ModelOp,
        consumers: &BTreeMap<&str, usize>,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        if !self.fuse_gemm_smq
            || gemm.kernel != KernelOp::GemmW4A8
            || silu.kernel != KernelOp::SiluMulQuant
        {
            return Ok(None);
        }
        let ([out], [input]) = (&gemm.outputs[..], &silu.inputs[..]) else {
            return Ok(None);
        };
        if input.edge != out.name || consumers.get(out.name.as_str()).copied() != Some(1) {
            return Ok(None);
        }
        let route = self.route(gemm, mxfp4, rules)?;
        let Route::Kernel { descs, shape } = &route else {
            return Ok(None);
        };
        let [g] = &descs[..] else {
            return Ok(None);
        };
        let silu_route = self.route(silu, mxfp4, rules)?;
        let Route::Kernel { descs: sdescs, .. } = &silu_route else {
            return Ok(None);
        };
        let [sd] = &sdescs[..] else {
            return Ok(None);
        };
        let Ok(desc) = saber_kernels::rules::gemm_decode_smq((g, *shape), sd, mxfp4) else {
            return Ok(None);
        };
        let mut bindings = self.bindings(gemm, &route, rules.ar.max_bytes)?;
        let c = bindings
            .iter()
            .find(|(n, _)| n == "c")
            .map(|(_, s)| s.clone())
            .ok_or_else(|| Error::Refused(format!("{}: the GEMM binds no `c`", gemm.name)))?;
        for (name, src) in self.bindings(silu, &silu_route, rules.ar.max_bytes)? {
            if name == "gu" {
                if src != c {
                    return Err(Error::Refused(format!(
                        "{}: row 4 found the producer reading a different buffer than the GEMM \
                         writes",
                        silu.name
                    )));
                }
                continue;
            }
            bindings.push((name, src));
        }
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let known = bindings.iter().any(|(n, _)| n == buffer)
                || desc.workspaces.iter().any(|w| &w.name == buffer);
            if !known {
                return Err(Error::Refused(format!(
                    "{}: row 4 names buffer `{buffer}`, which the fusion does not bind",
                    gemm.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// The decode GEMM pair (ws-h37): `first` and `second` are W4A8 GEMMs reading the SAME
    /// single activation edge, so neither can read the other's output and both can run in one
    /// dispatch. Each half is resolved exactly as it would be alone; the pair only fires when
    /// [`saber_kernels::rules::gemm_decode_pair`] accepts both (decode band, same `bk`/`tm`,
    /// `wperm1_nt1`). Anything else returns `None` and the two stay two ops -- a route choice,
    /// counted as [`BlockPlan::fused_pairs`].
    fn pair_fusion(
        &self,
        first: &ModelOp,
        second: &ModelOp,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        if !self.fuse_gemm_pair
            || first.kernel != KernelOp::GemmW4A8
            || second.kernel != KernelOp::GemmW4A8
        {
            return Ok(None);
        }
        let ([a0], [a1]) = (&first.inputs[..], &second.inputs[..]) else {
            return Ok(None);
        };
        if a0.edge != a1.edge {
            return Ok(None);
        }
        let (r0, r1) = (
            self.route(first, mxfp4, rules)?,
            self.route(second, mxfp4, rules)?,
        );
        let (
            Route::Kernel {
                descs: d0,
                shape: s0,
            },
            Route::Kernel {
                descs: d1,
                shape: s1,
            },
        ) = (&r0, &r1)
        else {
            return Ok(None);
        };
        let ([d0], [d1]) = (&d0[..], &d1[..]) else {
            return Ok(None);
        };
        let Ok(desc) = saber_kernels::rules::gemm_decode_pair((d0, *s0), (d1, *s1), mxfp4) else {
            return Ok(None);
        };
        let mut bindings = self.bindings(first, &r0, rules.ar.max_bytes)?;
        for (name, src) in self.bindings(second, &r1, rules.ar.max_bytes)? {
            match name.as_str() {
                // The shared activation: the same edge, so the same source, bound once.
                "a" | "as" => {}
                "w" | "ws" | "wref" | "c" => bindings.push((format!("{name}_b"), src)),
                other => {
                    return Err(Error::Refused(format!(
                        "{}: the decode pair does not know the second half's binding `{other}`",
                        second.name
                    )))
                }
            }
        }
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let known = bindings.iter().any(|(n, _)| n == buffer)
                || desc.workspaces.iter().any(|w| &w.name == buffer);
            if !known {
                return Err(Error::Refused(format!(
                    "{}: the decode pair names buffer `{buffer}`, which the fusion does not bind",
                    first.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// `qk_norm_rope` + `kv_append` in ONE decode dispatch (ws-h38, fusion-map row 7).
    ///
    /// Fires only at a DECODE bucket, with [`PlanOptions::fuse_rope_append`], when `append` is the
    /// same layer's `kv_append` reading the edge `rope` writes, and when
    /// [`saber_kernels::glue::rope_append`] accepts both halves' shapes (each resolved exactly as
    /// it would be alone). The fused descriptor binds the rope's buffers and the append's, minus
    /// the append's `key` -- the fused kernel reads back the `k_out` its own workgroup wrote.
    /// Anything else returns `None` and the two stay two ops, counted as
    /// [`BlockPlan::fused_rope_appends`].
    fn rope_append_fusion(
        &self,
        rope: &ModelOp,
        append: &ModelOp,
        mxfp4: &LaunchRules,
        rules: &SaberRules,
    ) -> Result<Option<Fusion>> {
        if !self.fuse_rope_append
            || self.bucket.phase() != Phase::Decode
            || rope.kernel != KernelOp::QkNormRope
            || append.kernel != KernelOp::KvAppend
            || rope.site != append.site
        {
            return Ok(None);
        }
        let (Some(out), Some(input)) = (rope.outputs.first(), append.inputs.first()) else {
            return Ok(None);
        };
        if out.name != input.edge {
            return Ok(None);
        }
        let (
            Some((
                glue::Op::QkNormRope {
                    interleaved,
                    flavour,
                },
                glue::Shape::Qk(qk),
            )),
            Some((glue::Op::KvAppend, glue::Shape::Kv(kv))),
        ) = (glue_shape(rope, self.m), glue_shape(append, self.m))
        else {
            return Ok(None);
        };
        let Ok(desc) = glue::rope_append(qk, interleaved, flavour, kv, &self.glue) else {
            return Ok(None);
        };
        // Each half's bindings exactly as it would bind alone.
        let (r0, r1) = (
            self.route(rope, mxfp4, rules)?,
            self.route(append, mxfp4, rules)?,
        );
        let mut bindings = self.bindings(rope, &r0, rules.ar.max_bytes)?;
        for (name, src) in self.bindings(append, &r1, rules.ar.max_bytes)? {
            match name.as_str() {
                // The fused kernel reads K back from the rope's own `k_out`.
                "key" => {}
                "value" | "kv" | "slot" | "k_scale" | "v_scale" => bindings.push((name, src)),
                other => {
                    return Err(Error::Refused(format!(
                        "{}: the rope + append fusion does not know the append's binding `{other}`",
                        append.name
                    )))
                }
            }
        }
        for slot in &desc.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            if !bindings.iter().any(|(n, _)| n == buffer) {
                return Err(Error::Refused(format!(
                    "{}: the rope + append fusion names buffer `{buffer}`, which it does not bind",
                    rope.name
                )));
            }
        }
        Ok(Some((desc, bindings)))
    }

    /// The decode attention with its split-KV combine run by the last-arriving split block
    /// (ws-h38, fusion-map row 6): [`saber_kernels::saber::attn_decode_declc`] over the op's own
    /// [`AttnShape`] -- the same one the unfused pair resolves -- or `None` (the pair stays) when
    /// [`PlanOptions::fuse_attn_combine`] is off, the op is not the paged decode attention, or the
    /// builder refuses (one split, a cache it has no instantiation for).
    fn combine_fusion(&self, op: &ModelOp, rules: &SaberRules) -> Result<Option<LaunchDescriptor>> {
        if !self.fuse_attn_combine
            || op.kernel != KernelOp::AttnDecodePaged
            || self.bucket.phase() != Phase::Decode
        {
            return Ok(None);
        }
        let s = self.attn_shape(op, rules, true)?;
        Ok(saber::attn_decode_declc(s, rules).ok())
    }

    /// Which A-side layout a W4A8 GEMM's activation edge actually carries, from the edge the
    /// plan allocated rather than from `M`.
    ///
    /// **This is the fix for the 2026-09-21 prefill garbage above 512 rows (ws-p6).**
    /// `rules::resolve` assumes a fused producer wrote A and re-derives the layout from `M`
    /// against `a_tiled_min_m`. That is right for `add_rms_quant`, `silu_mul_quant` and
    /// `attn_gate_quant`, all of which carry `tiled0`/`tiled1` instantiations at the same band
    /// edge — and wrong for `gdn_norm_quant`, which has ONE instantiation and writes row-major at
    /// every M. At a 1024- or 2048-row prefill chunk `gdn.out_proj` therefore took
    /// `mxfp4_gemm_atiled` and read row-major bytes as WMMA fragments, in all 48 GDN layers,
    /// starting with layer 0.
    ///
    /// Prod does not re-derive it either: `ref_arnq.py` REGISTERS the buffer
    /// (`ref_mxfp4.a_tiled_register`) when a producer wrote the tiled layout and the GEMM
    /// POPS that registration (`a_tiled_take`), so a buffer nobody registered takes the row-major
    /// family however large M is. The edge's declared layout is this plane's registry.
    fn a_layout_of(&self, op: &ModelOp) -> Result<ALayout> {
        let [input] = &op.inputs[..] else {
            return Err(Error::Refused(format!(
                "{}: a W4A8 GEMM reads exactly one activation edge, not {}",
                op.name,
                op.inputs.len()
            )));
        };
        let b = self.lookup(&input.edge).ok_or_else(|| {
            Error::Refused(format!(
                "{}: `{}` is not a planned buffer, so its A layout is a guess",
                op.name, input.edge
            ))
        })?;
        match b.layout {
            Layout::RowMajorFp8 => Ok(ALayout::RowMajor),
            Layout::ATiledFp8 => Ok(ALayout::FollowsM),
            other => Err(Error::Refused(format!(
                "{}: its A edge `{}` carries {other:?}, which is not an fp8 activation layout",
                op.name, input.edge
            ))),
        }
    }

    /// The producer half of the same law: what this op's kernel WRITES must be what its output
    /// edge DECLARES.
    ///
    /// Run for every fused-quant producer at plan time, because the alternative is finding out
    /// sixty-four layers later. `saber_model::graph::check_layouts` cannot ask this — a W4A8
    /// GEMM `accepts` both fp8 layouts, so the declaration and the consumer agree with each other
    /// while the kernel the rules emit disagrees with both.
    fn check_producer_layout(
        &self,
        op: &ModelOp,
        desc: &LaunchDescriptor,
        writes_tiled: bool,
    ) -> Result<()> {
        // The fp8 output is output 0 for all four producers (`add_rms_quant`'s residual is
        // output 1).
        let Some(edge) = op.outputs.first() else {
            return Ok(());
        };
        let declared = edge.rule.resolve_under(self.m, self.law);
        let wrote = if writes_tiled {
            Layout::ATiledFp8
        } else {
            Layout::RowMajorFp8
        };
        if wrote != declared {
            return Err(Error::Refused(format!(
                "{}: `{}` writes {wrote:?} at M={} (kernel `{}`) but the graph declares the edge \
                 `{}` as {declared:?}; a consumer reading the declared layout would read the \
                 other one's bytes",
                op.name, op.kernel, self.m, desc.kernel, edge.name,
            )));
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------- routing
    fn route(&self, op: &ModelOp, mxfp4: &LaunchRules, rules: &SaberRules) -> Result<Route> {
        let m = self.m;
        let refused = |e: saber_kernels::Error| Error::Refused(format!("{}: {e}", op.name));
        Ok(match op.kernel {
            KernelOp::GemmW4A8 => {
                let Geometry::Gemm { n, k, .. } = op.geometry else {
                    return Err(Error::Refused(format!("{}: not a GEMM geometry", op.name)));
                };
                let shape = Shape::new(m, n, k);
                let desc = resolve_with_layout(
                    GemmOp::Gemm {
                        has_wref: true,
                        partitions: NO_PARTITIONS,
                    },
                    shape,
                    mxfp4,
                    self.a_layout_of(op)?,
                )
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape,
                }
            }
            KernelOp::AddRmsQuant => {
                let Geometry::Rms { k } = op.geometry else {
                    return Err(Error::Refused(format!("{}: not an Rms geometry", op.name)));
                };
                let shape = Shape::new(m, 0, k);
                let kop = GemmOp::AddRmsQuant {
                    eps: self.graph.config.rms_norm_eps,
                };
                let desc = resolve(kop, shape, mxfp4).map_err(refused)?;
                self.check_producer_layout(op, &desc, writes_tiled(kop, m, mxfp4))?;
                Route::Kernel {
                    descs: vec![desc],
                    shape,
                }
            }
            KernelOp::SiluMulQuant => {
                let Geometry::SiluMul { n } = op.geometry else {
                    return Err(Error::Refused(format!(
                        "{}: not a SiluMul geometry",
                        op.name
                    )));
                };
                let shape = Shape::new(m, n, 0);
                let desc = resolve(GemmOp::SiluMulQuant, shape, mxfp4).map_err(refused)?;
                self.check_producer_layout(
                    op,
                    &desc,
                    writes_tiled(GemmOp::SiluMulQuant, m, mxfp4),
                )?;
                Route::Kernel {
                    descs: vec![desc],
                    shape,
                }
            }
            KernelOp::GdnNormQuant => {
                let Geometry::GdnNorm { n, zs, .. } = op.geometry else {
                    return Err(Error::Refused(format!(
                        "{}: not a GdnNorm geometry",
                        op.name
                    )));
                };
                let shape = Shape::new(m, n, 0);
                let kop = GemmOp::GdnNormQuant {
                    zs: i64::from(zs),
                    eps: self.graph.config.rms_norm_eps,
                };
                let desc = resolve(kop, shape, mxfp4).map_err(refused)?;
                self.check_producer_layout(op, &desc, writes_tiled(kop, m, mxfp4))?;
                Route::Kernel {
                    descs: vec![desc],
                    shape,
                }
            }
            KernelOp::AttnPrefillPaged => {
                let s = self.attn_shape(op, rules, false)?;
                let desc = saber::resolve(
                    saber::Op::AttnPrefill { fp8_mode: 3 },
                    saber::Shape::Attn(s),
                    rules,
                )
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::AttnDecodePaged => {
                let s = self.attn_shape(op, rules, true)?;
                let head = saber::resolve(saber::Op::AttnDecode, saber::Shape::Attn(s), rules)
                    .map_err(refused)?;
                let combine = attn_decode_combine(s, rules).map_err(refused)?;
                Route::Kernel {
                    descs: vec![head, combine],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnFusedUpdate => {
                let s = self.gdn_shape(op)?;
                let desc = saber::resolve(
                    saber::Op::GdnFusedUpdate { ab_bf16: true },
                    saber::Shape::Gdn(s),
                    rules,
                )
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnConvUpdate => {
                let s = self.gdn_shape(op)?;
                // The op declares TWO launches: conv_update then recurrent_update. The graph
                // models them as two ops, so this one issues only the conv and the launch budget
                // is asserted per op -- hence `dispatch_one` is never used and the pair's head
                // descriptor is re-declared as a single dispatch here.
                let mut desc = saber::resolve(
                    saber::Op::GdnDecodePair { ab_bf16: true },
                    saber::Shape::Gdn(s),
                    rules,
                )
                .map_err(refused)?;
                // ...and here it is, at last. `saber::resolve` describes the PAIR, so its descriptor
                // declares the two dispatches the pair issues when one op drives both. This route
                // issues only the conv, because the graph models the recurrent update as its own op
                // and rule 7 is asserted per op — so the count has to say what THIS route
                // dispatches, or `launch_op` refuses it (which it did, for every plan that routed
                // this op: ws-h11 found it on the first whole-model decode step and three callers
                // carried a re-declaration stopgap until now).
                desc.n_launches = 1;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnRecurrentUpdate => {
                let s = self.gdn_shape(op)?;
                let desc = gdn_decode_pair_recurrent(s, true, None, rules).map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            // ---- P27-15d: the tree verify form (the `DecodeTree` bucket) ---------------------
            KernelOp::GdnConvTree => {
                let s = self.gdn_shape(op)?;
                // CMP27L: under the `gdn_conv_rows` knob the rows-parallel twin (same bytes).
                let desc = if mxfp4.knobs.gdn_conv_rows {
                    saber::tree_gdn_conv_rows(s, rules)
                } else {
                    saber::tree_gdn_conv(s, rules)
                }
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnTreeUpdate => {
                let s = self.gdn_shape(op)?;
                let desc = saber::tree_gdn_update(s, rules).map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::AttnDecodeTree => {
                let s = self.attn_shape(op, rules, true)?;
                let head = saber::tree_attn_decode(s, rules).map_err(refused)?;
                let combine = saber::tree_attn_combine(s, rules).map_err(refused)?;
                Route::Kernel {
                    descs: vec![head, combine],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnConvPrep => {
                let s = self.gdn_prefill_shape(op, rules)?;
                let desc = saber::resolve(
                    saber::Op::GdnConvPrep { ab_bf16: true },
                    saber::Shape::GdnPrefill(s),
                    rules,
                )
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnKktSolve => {
                let s = self.gdn_prefill_shape(op, rules)?;
                let desc =
                    saber::resolve(saber::Op::GdnKktSolve, saber::Shape::GdnPrefill(s), rules)
                        .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::GdnChunkScan => {
                let s = self.gdn_prefill_shape(op, rules)?;
                let desc =
                    saber::resolve(saber::Op::GdnChunkScan, saber::Shape::GdnPrefill(s), rules)
                        .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::AllReduce => match self.collectives {
                Collectives::Identity => Route::Identity {
                    reason: "one rank with no peer: the output edge ALIASES the input edge, so \
                             this plan's row-parallel outputs are partial sums. Build the plan \
                             with `Collectives::Real` to launch the reduce.",
                },
                Collectives::Real => {
                    // In place over the ALIASED edge. The alias is not a shortcut here, it is the
                    // reason this works: the planner gives the reduce's output edge no buffer and
                    // points every consumer at the producer's, and the reference's one-shot kernel is
                    // in-place safe by construction — its reduce phase is
                    // `out4[k] = in4[k] + my_scratch[k]`, one thread per index, after the push
                    // phase that read all of `in4` has completed and been flagged.
                    let edge = &op
                        .inputs
                        .first()
                        .ok_or_else(|| {
                            Error::Refused(format!("{}: an all-reduce with no input", op.name))
                        })?
                        .edge;
                    let bytes = self
                        .lookup(edge)
                        .ok_or_else(|| {
                            Error::Refused(format!(
                                "{}: no buffer for `{edge}`, so there is nothing to reduce in \
                                 place",
                                op.name
                            ))
                        })?
                        .bytes;
                    // bf16 payload: every activation edge a reduce sees is bf16, and the rules
                    // pick the FAMILY from the message size — exact below `quant_min_bytes`, the
                    // rotated one above it. That threshold is the plan's, not this module's.
                    //
                    // And the SLOT is not a threshold at all any more: above `ar.max_bytes` the
                    // message is cut into row bands that each fit, one dispatch each, because the
                    // reduce is elementwise and a `rows x hidden` message is the concatenation of
                    // its rows (`saber::ar_bands`). The band count is a function of the bucket, so
                    // it is known at plan time and the whole sequence captures as nodes.
                    let elem = 2u64;
                    let rows = self.ar_rows(op, edge, bytes, elem)?;
                    let descs = saber::resolve_ar_banded(
                        rows,
                        u64::from(self.ar_elems_per_row(op)?),
                        elem,
                        saber::ArPayload::Bf16,
                        self.graph.tp,
                        rules,
                        &saber::Registry::new(rules),
                    )
                    .map_err(refused)?;
                    Route::Kernel {
                        descs,
                        shape: Shape::new(0, 0, 0),
                    }
                }
            },
            // The attention layer's glue. The shape comes from `saber_model::graph::glue_shape`
            // — the SAME function `Graph::route` calls to report what the plan would launch — so
            // a plan that resolves one kernel and a run that launches another is not expressible.
            KernelOp::QkNormRope | KernelOp::AttnGateQuant | KernelOp::KvAppend => {
                let (gop, shape) = glue_shape(op, m).ok_or_else(|| {
                    Error::Refused(format!(
                        "{}: {} carries a geometry the glue shapes cannot be derived from",
                        op.name, op.kernel
                    ))
                })?;
                let desc =
                    glue::resolve_under(gop, shape, &self.glue, self.law).map_err(refused)?;
                if op.kernel == KernelOp::AttnGateQuant {
                    // The glue plugin carries its own copy of the band edge
                    // (`rules/attn_glue_gfx1201.toml`'s `a_tiled_min_m`), so this is where the two
                    // copies are held against each other on the live plan rather than by a test.
                    self.check_producer_layout(op, &desc, self.glue.tiled_under(m, self.law))?;
                }
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::Embedding => {
                let Geometry::Embedding { vocab, hidden } = op.geometry else {
                    return Err(Error::Refused(format!(
                        "{}: not an Embedding geometry",
                        op.name
                    )));
                };
                // The vocab-parallel slice: rank `r` owns `[r * vocab_local, (r + 1) *
                // vocab_local)` and the kernel writes ZERO for every id outside it, so the reduce
                // that follows sums exactly one non-zero contribution per token. `vocab` here is
                // already the per-rank count (`Dims::vocab_local`), so the window is a
                // multiplication and not a division. At TP = 1 rank 0 owns everything and this is
                // `(0, vocab)`, which is what it always was.
                let begin = self.rank * vocab;
                let shape = HeadShape {
                    rows: m,
                    cols: hidden,
                    src_stride: i64::from(hidden),
                    vocab: (begin, begin + vocab),
                };
                let desc = head::resolve(HeadOp::Embedding, shape, &self.head).map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(0, 0, 0),
                }
            }
            KernelOp::LogitsGather => Route::Supplied {
                reason: "no ported kernel: `logits.gather` is the TP > 1 all-GATHER of the \
                         vocabulary shards, which is not the row selection the head plugin's \
                         `row_gather` performs. The caller gathers the shards itself — over the \
                         rendezvous socket in `saber forward --tp 2` — and the plan says so \
                         instead of refusing to exist, which is what kept a TP>1 head from being \
                         plannable at all",
            },
            KernelOp::GemmBf16 => {
                return Err(Error::Refused(format!(
                    "{}: {} has no ported kernel (it is the drafter's)",
                    op.name, op.kernel
                )))
            }
            // ---- the bf16 head (workstream N2, `Graph::with_bf16_head`) ----------------------
            KernelOp::AddRmsBf16 => {
                let Geometry::Rms { k } = op.geometry else {
                    return Err(Error::Refused(format!("{}: not an Rms geometry", op.name)));
                };
                let desc = head::resolve(
                    HeadOp::AddRmsBf16 {
                        eps_bits: self.graph.config.rms_norm_eps.to_bits(),
                    },
                    HeadShape::packed(m, k),
                    &self.head,
                )
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(m, 0, k),
                }
            }
            KernelOp::GemmBf16Head => {
                let Geometry::Gemm { n, k, .. } = op.geometry else {
                    return Err(Error::Refused(format!("{}: not a GEMM geometry", op.name)));
                };
                let rules = saber_kernels::dflash::DflashRules::prod_dflash()
                    .map_err(|e| Error::Refused(format!("dflash launch rules: {e}")))?;
                let desc =
                    saber_kernels::dflash::resolve_head_bgemm(m, n, k, &rules).map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(m, n, k),
                }
            }
            // The fp8-weight head (N4, `Graph::with_fp8_head`): the bf16 head's bands over the
            // checkpoint's e4m3 matrix + per-channel scale, expanded in register.
            KernelOp::GemmFp8Head => {
                let Geometry::Gemm { n, k, .. } = op.geometry else {
                    return Err(Error::Refused(format!("{}: not a GEMM geometry", op.name)));
                };
                let rules = saber_kernels::dflash::DflashRules::prod_dflash()
                    .map_err(|e| Error::Refused(format!("dflash launch rules: {e}")))?;
                // CMP27F: under the `head_m64` knob the 49..64-row head is one weight pass.
                // CMP27L: under the `head_b1_nt` knob the 1..16-row head takes the 32-row band.
                let desc = if mxfp4.knobs.head_b1_nt && m <= rules.bgemm.m {
                    saber_kernels::dflash::resolve_head_w8_b1(m, n, k, &rules)
                } else if mxfp4.knobs.head_m64 {
                    saber_kernels::dflash::resolve_head_w8_m64(m, n, k, &rules)
                } else {
                    saber_kernels::dflash::resolve_head_w8(m, n, k, &rules)
                }
                .map_err(refused)?;
                Route::Kernel {
                    descs: vec![desc],
                    shape: Shape::new(m, n, k),
                }
            }
        })
    }

    fn attn_shape(&self, op: &ModelOp, rules: &SaberRules, decode: bool) -> Result<AttnShape> {
        let Geometry::Attn {
            head_dim,
            block_size,
            q_heads,
            kv_heads,
            kv_dtype,
            ..
        } = op.geometry
        else {
            return Err(Error::Refused(format!("{}: not an Attn geometry", op.name)));
        };
        Ok(AttnShape {
            num_seqs: self.bucket.num_seqs(),
            q_len: self.bucket.q_len(),
            // THE PLAN'S ANSWER IS "EVERY ROW IS LIVE", which is the truth for every decode bucket
            // and for a prefill chunk that fills its bucket — and a lie for a padded one. A plan
            // cannot know the chunk, so a step says so: [`BlockPlan::retarget_prefill_rows`] moves
            // this number (and the grid with it) before the step binds, and `seqused_k` follows it
            // through `saber_exec::forward`'s index law. Planning the bucket here keeps a plan
            // that is never retargeted running exactly what it ran before ws-h27.
            live_rows: self.bucket.q_len(),
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            // THE ROW WIDTH THE INDEX BUILDER LAYS THE TABLE OUT AT, which is the caller's to
            // state when it is wider than `max_ctx` needs (ws-p12, [`PlanOptions::table_blocks`]).
            max_blocks: if self.table_blocks > 0 {
                self.table_blocks
            } else {
                self.params.kv_blocks(block_size)
            },
            max_ctx: self.params.max_ctx,
            // 0 = let the split law choose, which is what prod passes — and what makes a
            // sequence's answer a function of its neighbours, because the law divides by
            // `kv_heads * num_seqs` (see [`PlanOptions::batch_invariant_attn`]). A caller that
            // serves several sequences at once pins it to the law's ONE-SEQUENCE answer instead,
            // which is the same number in every bucket of this graph and therefore a plan fact.
            //
            // Only the decode route reads it: `saber_attn_prefill` has no split-KV phase, its
            // chunk is one sequence's rows, and handing it a number here would be a kernarg
            // nobody reads pretending to be a decision.
            splits: if decode && self.batch_invariant_attn {
                rules.decode_splits(self.params.max_ctx, kv_heads, 1, 0)
            } else {
                0
            },
            kv_dtype: if kv_dtype == "bf16" {
                KvDtype::Bf16
            } else {
                KvDtype::Fp8E4m3
            },
            // The paged cache is [blocks, kv_heads, block_size, 2 * head_dim].
            kv_block_stride: i64::from(kv_heads) * i64::from(block_size) * 2 * i64::from(head_dim),
            kv_head_stride: i64::from(block_size) * 2 * i64::from(head_dim),
            scale_bits: (f64::from(head_dim).powf(-0.5) as f32).to_bits(),
            // Prod passes 0 for both descales when the per-tensor K/V scales are 1.0 and fold
            // away (`ref_paged_attn.py:_descale_ptrs`).
            has_descales: false,
            // A ROW BAND's decode attention is the RAGGED pair (C2): `q_len` above stays the
            // band's MAX width (the scratch stride and the NW choice), each sequence's own length
            // comes from the step's `cu` (bound as `cu_q` in `bindings`), and the combine covers
            // the band's `m` rows. Every rectangle and every prefill chunk passes 0.
            ragged_rows: if decode {
                self.bucket.band_rows().unwrap_or(0)
            } else {
                0
            },
        })
    }

    /// The width of the merged `in_proj_qkvz` output this layer's GDN core reads `x` out of.
    fn x_pitch(&self, op: &ModelOp) -> Result<i64> {
        let edge = &op.inputs[0].edge;
        let b = self
            .lookup(edge)
            .ok_or_else(|| Error::Refused(format!("{}: `{edge}` is not planned", op.name)))?;
        Ok(i64::try_from(b.bytes / u64::from(self.m) / 2).unwrap_or(0))
    }

    /// The `in_proj_ba` edge this op's gates come from.
    ///
    /// Normally the op's own input list names it. `gdn.conv_prep` is the exception: the graph
    /// declares only its `qkvz` input although the kernel also reads the `a`/`b` gates (and
    /// `A_log` / `dt_bias`, see [`Planner::layer_weight`]) — an incompleteness in
    /// `saber-model::graph`'s prefill trio, worked around here by asking the graph for the op of
    /// the SAME layer that writes the edge, never by spelling an edge name out.
    fn ba_edge(&self, op: &ModelOp) -> Result<String> {
        if let Some(e) = op
            .inputs
            .iter()
            .map(|i| i.edge.clone())
            .find(|e| e.ends_with(".ba"))
        {
            return Ok(e);
        }
        self.graph
            .ops
            .iter()
            .filter(|o| o.site == op.site)
            .flat_map(|o| o.outputs.iter())
            .map(|e| e.name.clone())
            .find(|e| e.ends_with(".ba"))
            .ok_or_else(|| {
                Error::Refused(format!(
                    "{}: no `.ba` edge on this op or anywhere in {:?}",
                    op.name, op.site
                ))
            })
    }

    /// The `b`/`a` gate pair's row stride, from the `in_proj_ba` edge.
    fn ab_stride(&self, op: &ModelOp) -> Result<i64> {
        let edge = self.ba_edge(op)?;
        let b = self
            .lookup(&edge)
            .ok_or_else(|| Error::Refused(format!("{}: `{edge}` is not planned", op.name)))?;
        Ok(i64::try_from(b.bytes / u64::from(self.m) / 2).unwrap_or(0))
    }

    /// A weight by pack record name, with the layout id the **model** says that record must
    /// carry.
    ///
    /// The expectation is not computed here and it is not a convention: it is
    /// `Graph::device_final_records()`, the same list `tools/pack` builds the pack from. A record
    /// the model does not know is a refusal rather than a guessed layout — binding a weight the
    /// plan never asked for is how a tier and an engine drift apart.
    fn weight(&self, record: &str) -> Result<Source> {
        let layout = *self.layouts.get(record).ok_or_else(|| {
            Error::Refused(format!(
                "`{record}` is not a record the model's device-final plan declares; the pack and \
                 the plan disagree about what this op reads"
            ))
        })?;
        Ok(Source::Weight {
            key: record.to_owned(),
            layout,
        })
    }

    /// A weight by role: the op's own binding if it has one, otherwise the binding another op of
    /// the same layer carries for that role.
    ///
    /// The fallback exists for `gdn.conv_prep`, which reads `A_log` and `dt_bias` but binds only
    /// the conv filter in the graph. Resolving it through a sibling op keeps the record NAME the
    /// model's property — this module never spells a checkpoint record out.
    fn layer_weight(&self, op: &ModelOp, role: WeightRole) -> Result<Source> {
        if let Some(b) = op.weights.iter().find(|b| b.role == role) {
            return self.weight(&b.record);
        }
        self.graph
            .ops
            .iter()
            .filter(|o| o.site == op.site)
            .flat_map(|o| o.weights.iter())
            .find(|b| b.role == role)
            .map(|b| self.weight(&b.record))
            .transpose()?
            .ok_or_else(|| {
                Error::Refused(format!(
                    "{}: no {role:?} weight on this op or anywhere in {:?}",
                    op.name, op.site
                ))
            })
    }

    fn gdn_dims(&self, op: &ModelOp) -> Result<GdnDims> {
        let Geometry::GdnCore {
            conv_width,
            head_k,
            head_v,
            heads,
            key_heads,
            chunk,
        } = op.geometry
        else {
            return Err(Error::Refused(format!(
                "{}: not a GdnCore geometry",
                op.name
            )));
        };
        Ok(GdnDims {
            conv_width,
            head_k,
            head_v,
            heads,
            key_heads,
            chunk,
            qkv_n: 2 * key_heads * head_k + heads * head_v,
        })
    }

    fn gdn_shape(&self, op: &ModelOp) -> Result<GdnShape> {
        let GdnDims {
            conv_width,
            head_k,
            head_v,
            heads,
            key_heads,
            qkv_n,
            ..
        } = self.gdn_dims(op)?;
        let q_len = self.bucket.q_len();
        // THE CONV CACHE'S LAYOUT IS A MODEL CONSTANT, NOT A BUCKET ONE.
        //
        // The cache is `[slot][qkv_n][state_len]`, and `state_len = conv_width - 1 + q - 1` — so
        // sizing the STRIDE from this bucket's `q` makes the layout a function of the verify width:
        // a width-1 step would index the dimension at stride 3 in a buffer a width-5 step laid out
        // at stride 7, the dims would collide, every dispatch would still issue, and the tokens
        // would be quietly wrong from then on. One allocation has to serve every bucket of a graph,
        // because the ws-l6 width controller moves a sequence between widths between steps.
        //
        // So the stride comes from the GRAPH's `verify_width` — the widest step this model plans
        // for — and is the same for every bucket. Safe by construction: every access in both conv
        // kernels is `sp[c * cs_dim + k * cs_tok]` with `k < slen_eff <= state_len_max`, so a wider
        // stride with a shorter walk simply leaves unused tail entries, and the convention is
        // self-consistent across widths because every step WRITES its window from entry 0 and READS
        // it at `off = naccept - 1`.
        // P27-15d: the tree form widens the CACHE (`Graph::with_tree_conv_rows`), never `sidx`.
        let layout_len =
            i64::from(conv_width - 1 + self.graph.conv_layout_rows(qkv_n).saturating_sub(1));
        // And the WALK LENGTH stays this bucket's own. This is load-bearing, not cosmetic: the
        // kernarg `state_len_max` that `saber.rs` publishes is `(conv_width - 1) + (maxq - 1)` from
        // `maxq` below, and the write-back is
        //     slen_eff = state_len_max - (maxq - slen);
        //     for (i = 0; i < slen_eff; ++i) val = hist[i + 1][c] ...
        // where `hist` is `hist[CP_ST][CP_DPL]` — **`conv_width - 1` = 3 REGISTERS**, not
        // `state_len_max`. Hand a `q = 1` step a width-5 `state_len` and it indexes `hist[4..6]` off
        // the end of a register array: another memory fault, not a wrong number. Stride from the
        // plan, walk length from the bucket.
        Ok(GdnShape {
            n_seqs: self.bucket.num_seqs(),
            h: heads,
            hg: key_heads,
            head_k,
            head_v,
            conv_width,
            maxq: q_len,
            o_rows: self.m,
            xpitch: self.x_pitch(op)?,
            // The conv cache is [slots, qkv_n, state_len_max_for_the_GRAPH] — see above.
            cs_seq: i64::from(qkv_n) * layout_len,
            cs_dim: layout_len,
            cs_tok: 1,
            // `cache_idx` IS `sidx[:, 0]`: the same allocation read with a stride of q_len.
            //
            // This is prod's own call -- `conv_update(..., sidx[:, 0][:nseq], ...)` passes
            // `cache_idx.data_ptr()` and `cache_idx.stride(0)`, so the POINTER is the `sidx`
            // tensor's and the STRIDE is its row stride (`ref_gdn.py`, the decode leg). A
            // kernarg that says "stride q_len" is therefore a statement about WHICH buffer is
            // bound, and `bindings` binds `sidx` for both decode ops because of it. The per-SLOT
            // `cache_idx` table is a contiguous table and belongs to `ci_stride = 1`, which is
            // the prefill shape below.
            ci_stride: i64::from(q_len),
            ab_stride: self.ab_stride(op)?,
            // The recurrent state is [slots, heads, head_v, head_k].
            st_slot: i64::from(heads) * i64::from(head_v) * i64::from(head_k),
            st_head: i64::from(head_v) * i64::from(head_k),
            sidx_stride: i64::from(q_len),
            scale_bits: (f64::from(head_k).powf(-0.5) as f32).to_bits(),
            softplus_thr_bits: SOFTPLUS_THRESHOLD.to_bits(),
            has_naccept: true,
            // The checkpoint has no `conv1d.bias` record: Qwen3.8's GDN conv is bias-free.
            has_bias: false,
            state_f16: stores_f16_state(op),
        })
    }

    fn gdn_prefill_shape(&self, op: &ModelOp, rules: &SaberRules) -> Result<GdnPrefillShape> {
        let GdnDims {
            conv_width,
            head_k,
            head_v,
            heads,
            key_heads,
            chunk,
            qkv_n,
        } = self.gdn_dims(op)?;
        // The graph and the rules both name the prefill token block; a disagreement would put the
        // registry query and the kernel's own tiling out of step, so it is an error, not a pick.
        if chunk != rules.gdn.chunk {
            return Err(Error::Refused(format!(
                "{}: the graph's chunk is {chunk} and `rules/saber_gfx1201.toml` says {}",
                op.name, rules.gdn.chunk
            )));
        }
        // The SAME layout as the decode path, and for the same reason: one `l<n>.gdn.conv_state`
        // buffer serves both phases, so a prompt that wrote its history at stride `conv_width - 1`
        // and a verify step that reads at the graph's wider stride is the dimension collision one
        // phase earlier. The prefill kernel takes no `state_len_max` kernarg — it writes the last
        // `CP_ST = conv_width - 1` entries of each row from a compile-time constant — so widening
        // the stride leaves its tail entries untouched and costs it nothing.
        let layout_len =
            i64::from(conv_width - 1 + self.graph.conv_layout_rows(qkv_n).saturating_sub(1));
        Ok(GdnPrefillShape {
            n_seqs: self.bucket.num_seqs(),
            t: self.m,
            h: heads,
            hg: key_heads,
            head_k,
            head_v,
            conv_width,
            chunk,
            xpitch: self.x_pitch(op)?,
            cs_seq: i64::from(qkv_n) * layout_len,
            cs_dim: layout_len,
            cs_tok: 1,
            // A prefill chunk's `cache_idx` is one slot per sequence, contiguous.
            ci_stride: 1,
            ab_stride: self.ab_stride(op)?,
            scale_bits: (f64::from(head_k).powf(-0.5) as f32).to_bits(),
            softplus_thr_bits: SOFTPLUS_THRESHOLD.to_bits(),
            has_bias: false,
            has_init: true,
            state_f16: stores_f16_state(op),
        })
    }

    // ---------------------------------------------------------------------------- binding
    /// Which arena range, weight, state or index table every buffer the descriptor names is.
    ///
    /// This is the one table in the module, and it is a table on purpose: a descriptor's buffer
    /// names are the kernel's ABI, the graph's edge names are the model's, and the mapping between
    /// them is data. A name this does not cover is an error -- `pack_kernarg` would otherwise
    /// write whatever the staging buffer last held into a pointer slot.
    fn bindings(
        &self,
        op: &ModelOp,
        route: &Route,
        ar_max_bytes: u64,
    ) -> Result<Vec<(String, Source)>> {
        let mut out: Vec<(String, Source)> = Vec::new();
        // A macro, not a closure: `bind_gates` needs `&mut out` in the middle of the table
        // and a closure holding it would make that a second mutable borrow.
        macro_rules! bind {
            ($name:expr, $src:expr) => {
                out.push(($name.to_owned(), $src))
            };
        }
        let weight = |role: WeightRole| -> Result<Source> { self.layer_weight(op, role) };
        let state_ref = |needle: &str| -> Result<&saber_model::graph::StateRef> {
            op.state
                .iter()
                .find(|s| s.name.ends_with(needle))
                .ok_or_else(|| Error::Refused(format!("{}: no `{needle}` state", op.name)))
        };
        let state = |needle: &str| -> Result<Source> {
            Ok(Source::State {
                key: state_ref(needle)?.name.clone(),
                at: StateAt::Pool,
            })
        };
        // The one state a STEP has to point: `saber_gdn_chunk_scan` indexes it by `blockIdx.z`.
        let state_per_step = |needle: &str| -> Result<Source> {
            Ok(Source::State {
                key: state_ref(needle)?.name.clone(),
                at: StateAt::PerStep,
            })
        };
        let index = |buf: &str| Source::Index(format!("{}.{buf}", op.name));
        let out_edge = |i: usize| -> Result<&str> {
            op.outputs
                .get(i)
                .map(|e| e.name.as_str())
                .ok_or_else(|| Error::Refused(format!("{}: no output {i}", op.name)))
        };

        match op.kernel {
            KernelOp::GemmW4A8 => {
                let a = &op.inputs[0].edge;
                bind!("a", self.at(a)?);
                bind!("as", self.scale_of(a)?);
                bind!("w", self.weight(&format!("{}.w", op.name))?);
                bind!("ws", self.weight(&format!("{}.ws", op.name))?);
                bind!("wref", self.weight(&format!("{}.wref", op.name))?);
                bind!("c", self.at(out_edge(0)?)?);
            }
            KernelOp::AddRmsQuant => {
                bind!("y", self.at(&op.inputs[0].edge)?);
                match op.inputs.get(1) {
                    Some(res) => bind!("res", self.at(&res.edge)?),
                    None => bind!("res", self.at(&format!("{}.zero_resid", op.name))?),
                }
                bind!("w", weight(WeightRole::NormWeight)?);
                bind!("q", self.at(out_edge(0)?)?);
                bind!("scale", self.scale_of(out_edge(0)?)?);
                bind!("res_out", self.at(out_edge(1)?)?);
            }
            KernelOp::SiluMulQuant => {
                bind!("gu", self.at(&op.inputs[0].edge)?);
                bind!("q", self.at(out_edge(0)?)?);
                bind!("scale", self.scale_of(out_edge(0)?)?);
            }
            KernelOp::AddRmsBf16 => {
                bind!("y", self.at(&op.inputs[0].edge)?);
                let res = op.inputs.get(1).ok_or_else(|| {
                    Error::Refused(format!(
                        "{}: the bf16 head's norm needs a residual",
                        op.name
                    ))
                })?;
                bind!("res", self.at(&res.edge)?);
                bind!("w", weight(WeightRole::NormWeight)?);
                bind!("out", self.at(out_edge(0)?)?);
                bind!("res_out", self.at(out_edge(1)?)?);
            }
            KernelOp::GemmBf16Head => {
                bind!("a", self.at(&op.inputs[0].edge)?);
                bind!("w", self.weight(&format!("{}.w", op.name))?);
                bind!("c", self.at(out_edge(0)?)?);
            }
            KernelOp::GemmFp8Head => {
                bind!("a", self.at(&op.inputs[0].edge)?);
                bind!("w", self.weight(&format!("{}.w8", op.name))?);
                bind!("ws", self.weight(&format!("{}.ws", op.name))?);
                bind!("c", self.at(out_edge(0)?)?);
            }
            KernelOp::GdnNormQuant => {
                let Geometry::GdnNorm { n, zs, .. } = op.geometry else {
                    return Err(Error::Refused(format!(
                        "{}: not a GdnNorm geometry",
                        op.name
                    )));
                };
                bind!("x", self.at(&op.inputs[0].edge)?);
                // `z` is a SLICE of the merged in_proj output: the pointer is qkvz + (zs - n)
                // elements and the row stride is the merged width, which is why `zs` is a
                // selection key and not a constant.
                let zoff = u64::from(zs - n) * 2;
                let zbytes = u64::from(self.m) * u64::from(zs) * 2 - zoff;
                bind!("z", self.at_offset(&op.inputs[1].edge, zoff, zbytes)?);
                bind!("w", weight(WeightRole::NormWeight)?);
                bind!("q", self.at(out_edge(0)?)?);
                bind!("scale", self.scale_of(out_edge(0)?)?);
            }
            KernelOp::AttnDecodeTree => {
                bind!("q", self.at(&op.inputs[0].edge)?);
                bind!("kv", state("kv_cache")?);
                bind!("block_table", index("block_table"));
                bind!("seqused_k", index("seqused_k"));
                bind!("out", self.at(out_edge(0)?)?);
                bind!("cu_q", index("cu"));
                // P27-15d: one u16 ancestor word a row (the tree's mask; a chain's IS causal).
                bind!("anc", index("anc"));
            }
            KernelOp::GdnConvTree => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let conv = out_edge(0)?.to_owned();
                bind!("x", self.at(&op.inputs[0].edge)?);
                bind!("wgt", weight(WeightRole::ConvWeight)?);
                bind!("cstate", state("conv_state")?);
                // The per-SLOT table at `ci_stride` 1: the sequence's conv cache is its base slot.
                bind!("cache_idx", index("cache_idx"));
                bind!("hsrc", index("hsrc"));
                bind!("tsrc", index("tsrc"));
                bind!("q", self.at_offset(&conv, qo, qb)?);
                bind!("k", self.at_offset(&conv, ko, kb)?);
                bind!("v", self.at_offset(&conv, vo, vb)?);
                bind!("cu", index("cu"));
            }
            KernelOp::GdnTreeUpdate => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let conv = op.inputs[0].edge.clone();
                bind!("q", self.at_offset(&conv, qo, qb)?);
                bind!("k", self.at_offset(&conv, ko, kb)?);
                bind!("v", self.at_offset(&conv, vo, vb)?);
                self.bind_gates(op, &mut out)?;
                bind!("a_log", weight(WeightRole::ALog)?);
                bind!("dt_bias", weight(WeightRole::DtBias)?);
                bind!("state", state("recurrent_state")?);
                bind!("o", self.at(out_edge(0)?)?);
                bind!("cu", index("cu"));
                // The ONE committed f16 slot a sequence, the ancestor words, the accepted row of
                // the previous tree (the ADV prologue; -1 = none).
                bind!("slot", index("tslot"));
                bind!("anc", index("anc"));
                bind!("acc", index("acc"));
                // The layer's tree record (engine-owned, `engine.spectree.records`): VS = 1 reads
                // its (sequence, head) record in the prologue and rewrites it after the
                // workgroup's barriers, so one record serves as `ts_in` and `ts_out`.
                let rec = Source::State {
                    key: tree_record_key(op)?,
                    at: StateAt::Pool,
                };
                bind!("ts_in", rec.clone());
                bind!("ts_out", rec);
            }
            KernelOp::AttnPrefillPaged | KernelOp::AttnDecodePaged => {
                bind!("q", self.at(&op.inputs[0].edge)?);
                bind!("kv", state("kv_cache")?);
                bind!("block_table", index("block_table"));
                bind!("seqused_k", index("seqused_k"));
                bind!("out", self.at(out_edge(0)?)?);
                // The ragged pair's one extra buffer (C2): the step's cumulative LIVE-row prefix,
                // the same `cu` law every GDN kernel walks. Its own key per attention layer (36
                // bytes of blob at batch 8), bound only where a row band's descriptors name it, so
                // a rectangle's key set -- and its index blob -- is byte for byte what it was.
                if let Route::Kernel { descs, .. } = route {
                    if descs.iter().any(|d| d.arg("cu_q").is_some()) {
                        bind!("cu_q", index("cu"));
                    }
                }
            }
            KernelOp::GdnFusedUpdate => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let qkv = format!("{}.qkv", op.name);
                bind!("x", self.at(&op.inputs[0].edge)?);
                bind!("wgt", weight(WeightRole::ConvWeight)?);
                bind!("cstate", state("conv_state")?);
                // `cache_idx` IS `sidx[:, 0]` at decode -- the SAME allocation, read at
                // `ci_stride = q_len` (`gdn_shape`). Binding the per-SLOT `cache_idx` table here
                // is right at width 1, where the two strides coincide, and reads slot `n * q`'s
                // entry for every sequence above the first at any width above it (ws-l17).
                bind!("cache_idx", index("sidx"));
                bind!("naccept", index("naccept"));
                bind!("cu", index("cu"));
                bind!("q", self.at_offset(&qkv, qo, qb)?);
                bind!("k", self.at_offset(&qkv, ko, kb)?);
                bind!("v", self.at_offset(&qkv, vo, vb)?);
                self.bind_gates(op, &mut out)?;
                bind!("a_log", weight(WeightRole::ALog)?);
                bind!("dt_bias", weight(WeightRole::DtBias)?);
                bind!("state", state("recurrent_state")?);
                bind!("o", self.at(out_edge(0)?)?);
                bind!("sidx", index("sidx"));
            }
            KernelOp::GdnConvUpdate => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let conv = out_edge(0)?.to_owned();
                bind!("x", self.at(&op.inputs[0].edge)?);
                bind!("wgt", weight(WeightRole::ConvWeight)?);
                bind!("cstate", state("conv_state")?);
                // The same law as the fused update's, and the same buffer: `sidx[:, 0]` at
                // `ci_stride = q_len`. This op is the one that RUNS at batch >= 2 (the fused
                // kernel's resident grid caps at `fused_max_items`), so this binding is where
                // ws-s1's batched verify step read slot `n * q`'s entry -- 0 past the table's
                // end, the kernels' null slot, so every sequence but slot 0 updated no conv
                // state at all and its rows read whatever the arena held.
                //
                // It names the PARTNER's key, not one of its own, because prod hands the pair's
                // two halves the one `sidx` tensor and because a key is a BLOB SLOT: a second
                // name for the same bytes is 8 KiB per layer of index blob re-uploaded every step
                // (48 layers = 372 KiB, a fifth of ws-h29's step-varying region) to hold a table
                // the step already carries. `GdnKktSolve` names its producer's `cu` for the same
                // reason.
                bind!(
                    "cache_idx",
                    Source::Index(format!("{}.sidx", pair_partner(op)?))
                );
                bind!("naccept", index("naccept"));
                bind!("q", self.at_offset(&conv, qo, qb)?);
                bind!("k", self.at_offset(&conv, ko, kb)?);
                bind!("v", self.at_offset(&conv, vo, vb)?);
                bind!("cu", index("cu"));
            }
            KernelOp::GdnRecurrentUpdate => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let conv = op.inputs[0].edge.clone();
                bind!("q", self.at_offset(&conv, qo, qb)?);
                bind!("k", self.at_offset(&conv, ko, kb)?);
                bind!("v", self.at_offset(&conv, vo, vb)?);
                self.bind_gates(op, &mut out)?;
                bind!("a_log", weight(WeightRole::ALog)?);
                bind!("dt_bias", weight(WeightRole::DtBias)?);
                bind!("state", state("recurrent_state")?);
                bind!("o", self.at(out_edge(0)?)?);
                bind!("cu", index("cu"));
                bind!("sidx", index("sidx"));
                bind!("naccept", index("naccept"));
            }
            KernelOp::GdnConvPrep => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let conv = out_edge(0)?.to_owned();
                let gates = format!("{}.gates", op.name);
                let gb = self.gate_bytes(op)?;
                bind!("x", self.at(&op.inputs[0].edge)?);
                bind!("wgt", weight(WeightRole::ConvWeight)?);
                bind!("cstate", state("conv_state")?);
                bind!("cache_idx", index("cache_idx"));
                bind!("has_init", index("has_init"));
                self.bind_gates(op, &mut out)?;
                bind!("a_log", weight(WeightRole::ALog)?);
                bind!("dt_bias", weight(WeightRole::DtBias)?);
                bind!("q", self.at_offset(&conv, qo, qb)?);
                bind!("k", self.at_offset(&conv, ko, kb)?);
                bind!("v", self.at_offset(&conv, vo, vb)?);
                bind!("g", self.at_offset(&gates, 0, gb)?);
                bind!("beta", self.at_offset(&gates, gb, gb)?);
                bind!("cu", index("cu"));
            }
            KernelOp::GdnKktSolve => {
                let [_, (ko, kb), _] = self.qkv_regions(&op.geometry)?;
                let conv = &op.inputs[0].edge;
                let prep = conv_producer(op)?;
                let gates = format!("{prep}.gates");
                let gb = self.gate_bytes(op)?;
                bind!("k", self.at_offset(conv, ko, kb)?);
                bind!("beta", self.at_offset(&gates, gb, gb)?);
                bind!("g", self.at_offset(&gates, 0, gb)?);
                bind!("a_out", self.at(out_edge(0)?)?);
                bind!("cu", Source::Index(format!("{prep}.cu")));
            }
            KernelOp::GdnChunkScan => {
                let [(qo, qb), (ko, kb), (vo, vb)] = self.qkv_regions(&op.geometry)?;
                let conv = &op.inputs[0].edge;
                let prep = conv_producer(op)?;
                let gates = format!("{prep}.gates");
                let gb = self.gate_bytes(op)?;
                bind!("q", self.at_offset(conv, qo, qb)?);
                bind!("k", self.at_offset(conv, ko, kb)?);
                bind!("v", self.at_offset(conv, vo, vb)?);
                bind!("a_mat", self.at(&op.inputs[1].edge)?);
                bind!("gcs", self.at_offset(&gates, 0, gb)?);
                bind!("beta", self.at_offset(&gates, gb, gb)?);
                // `ht` IN PLACE (ws-h30): the same slot `h0` was read from. Sound in this
                // kernel — a workgroup reads its own slice into registers before the chunk loop
                // and writes exactly that slice after it, and the `(nq, hv, bv)` grid partitions
                // the state, so the only write a read can race with is its own workgroup's. It
                // used to be an arena buffer `<op>.ht` that NOTHING copied back, which is why a
                // multi-chunk prompt restarted the delta rule at every boundary and every decode
                // step after any prompt began from a zero recurrent state (LEDGER H29 part 2).
                let h0 = state_per_step("recurrent_state")?;
                bind!("h0", h0.clone());
                bind!("o", self.at(out_edge(0)?)?);
                bind!("ht", h0);
                bind!("cu", Source::Index(format!("{prep}.cu")));
            }
            KernelOp::LogitsGather => {
                // Supplied: the caller gathers the vocabulary shards itself and writes the result
                // here. Named so that a caller can find the buffer without knowing the edge name.
                bind!("logits", self.at(out_edge(0)?)?);
                bind!("partial", self.at(&op.inputs[0].edge)?);
            }
            KernelOp::QkNormRope => {
                bind!("qkv", self.at(&op.inputs[0].edge)?);
                // TWO norm weights, and they are not interchangeable: `q_norm` is applied to
                // the q heads and `k_norm` to the k heads. `layer_weight` returns the first of
                // a role, so this asks for them by position in the op's own binding list.
                bind!("qw", self.nth_weight(op, WeightRole::NormWeight, 0)?);
                bind!("kw", self.nth_weight(op, WeightRole::NormWeight, 1)?);
                // The rope table and this chunk's mrope positions. Both are SESSION facts —
                // the cache is a constant of the config and the positions are where the
                // sequence is — so they come from the caller, like every index table.
                bind!("cos_sin", index("cos_sin"));
                bind!("pos", index("pos"));
                bind!("q_out", self.at(out_edge(0)?)?);
                bind!("k_out", self.at(&format!("{}.k", op.name))?);
            }
            KernelOp::AttnGateQuant => {
                bind!("core", self.at(&op.inputs[0].edge)?);
                // The gate is read IN PLACE out of the projection's [q|gate] block: the kernel
                // adds the `+head_dim` offset per head itself, so this is the projection's
                // base, not a materialised gate tensor.
                bind!("gate", self.at(&op.inputs[1].edge)?);
                bind!("q", self.at(out_edge(0)?)?);
                bind!("scale", self.scale_of(out_edge(0)?)?);
            }
            KernelOp::KvAppend => {
                let Geometry::Attn {
                    head_dim,
                    q_heads,
                    kv_heads,
                    ..
                } = op.geometry
                else {
                    return Err(Error::Refused(format!("{}: not an Attn geometry", op.name)));
                };
                // K is the roped k the sibling `qk_norm_rope` wrote; V is read in place out of
                // the projection, after the [q|gate] pairs and the k block.
                let qk = self.sibling(op, KernelOp::QkNormRope)?;
                bind!("key", self.at(&format!("{qk}.k"))?);
                let qkv = &op.inputs[1].edge;
                let voff = u64::from(2 * q_heads + kv_heads) * u64::from(head_dim) * 2;
                let total = self
                    .lookup(qkv)
                    .ok_or_else(|| Error::Refused(format!("{}: `{qkv}` is not planned", op.name)))?
                    .bytes;
                bind!("value", self.at_offset(qkv, voff, total - voff)?);
                bind!("kv", state("kv_cache")?);
                bind!("slot", index("slot"));
                // fp32 scalars. The pack carries `k_scale`/`v_scale` as **bf16** records, which
                // this kernel cannot read (`const float*`), so they come from the caller until
                // the pack carries them as `f32_from_bf16` — the same cast `A_log` and
                // `dt_bias` already get.
                bind!("k_scale", index("k_scale"));
                bind!("v_scale", index("v_scale"));
            }
            KernelOp::Embedding => {
                bind!("table", weight(WeightRole::Embedding)?);
                bind!("ids", index("ids"));
                bind!("out", self.at(out_edge(0)?)?);
            }
            KernelOp::AllReduce => {
                // `in` and `out` are ONE address: the reduce is in place over the aliased edge.
                // `at` resolves through the alias, so both sides name the producer's buffer.
                //
                // Above the IPC scratch slot the op is a SEQUENCE of row bands, and each band
                // binds its own pair at its own byte offset into that one buffer — `in`/`out` for
                // band 0, `in.b1`/`out.b1` for the next, exactly as `saber::ar_band_names` spells
                // them. A one-band message therefore binds what it always bound.
                if let Route::Kernel { .. } = route {
                    let edge = op.inputs.first().map(|i| i.edge.as_str()).ok_or_else(|| {
                        Error::Refused(format!("{}: an all-reduce with no input", op.name))
                    })?;
                    let bytes = self
                        .lookup(edge)
                        .ok_or_else(|| {
                            Error::Refused(format!("{}: `{edge}` is not planned", op.name))
                        })?
                        .bytes;
                    let elems = u64::from(self.ar_elems_per_row(op)?);
                    let rows = self.ar_rows(op, edge, bytes, 2)?;
                    let row_bytes = elems * 2;
                    for (i, band) in saber::ar_bands(rows, elems, 2, ar_max_bytes)
                        .map_err(|e| Error::Refused(format!("{}: {e}", op.name)))?
                        .iter()
                        .enumerate()
                    {
                        let (in_name, out_name) = saber::ar_band_names(i);
                        let src = self.at_offset(
                            edge,
                            band.first_row * row_bytes,
                            band.rows * row_bytes,
                        )?;
                        bind!(in_name.as_str(), src.clone());
                        bind!(out_name.as_str(), src);
                    }
                }
            }
            other => {
                return Err(Error::Refused(format!(
                    "{}: no binding table for {other}",
                    op.name
                )))
            }
        }
        // Every buffer the descriptor names must be covered, or the launcher would refuse it at
        // pack time with a less specific message. Checking here keeps the failure at plan time.
        if let Route::Kernel { descs, .. } = route {
            for desc in descs {
                for slot in &desc.args {
                    let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                        continue;
                    };
                    let known = out.iter().any(|(n, _)| n == buffer)
                        || desc.workspaces.iter().any(|w| &w.name == buffer);
                    if !known {
                        return Err(Error::Refused(format!(
                            "{}: `{}` names buffer `{buffer}`, which this block does not bind",
                            op.name, desc.kernel
                        )));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The `n`th weight of `role` in this op's own binding list.
    ///
    /// `layer_weight` answers "the weight of this role", which is right when there is one.
    /// `qk_norm_rope` has two norm weights and they are not interchangeable, so position is
    /// what distinguishes them — and it is the op's declared order, not a name match.
    fn nth_weight(&self, op: &ModelOp, role: WeightRole, n: usize) -> Result<Source> {
        let b = op
            .weights
            .iter()
            .filter(|w| w.role == role)
            .nth(n)
            .ok_or_else(|| {
                Error::Refused(format!(
                    "{}: no {role:?} weight at position {n}; the op declares {}",
                    op.name,
                    op.weights.len()
                ))
            })?;
        self.weight(&b.record)
    }

    /// The op of `kernel` at the same [`Site`] as `op`.
    ///
    /// `kv_append` reads the k tensor `qk_norm_rope` wrote, and that buffer is named after its
    /// producer. Asking the graph which op of this layer produced it keeps the name in one
    /// place instead of spelling `layer.{n}.attn.qk_norm_rope.k` out here.
    fn sibling(&self, op: &ModelOp, kernel: KernelOp) -> Result<String> {
        self.graph
            .ops
            .iter()
            .find(|o| o.site == op.site && o.kernel == kernel)
            .map(|o| o.name.clone())
            .ok_or_else(|| {
                Error::Refused(format!(
                    "{}: no {kernel} op at {:?}, so its output cannot be bound",
                    op.name, op.site
                ))
            })
    }

    fn gate_bytes(&self, op: &ModelOp) -> Result<u64> {
        let Geometry::GdnCore { heads, .. } = op.geometry else {
            return Err(Error::Refused(format!(
                "{}: not a GdnCore geometry",
                op.name
            )));
        };
        Ok(u64::from(self.m) * u64::from(heads) * 4)
    }

    /// `av` and `bv`: the two halves of the merged `in_proj_ba` output, `concat(b, a)`.
    fn bind_gates(&self, op: &ModelOp, out: &mut Vec<(String, Source)>) -> Result<()> {
        let Geometry::GdnCore { heads, .. } = op.geometry else {
            return Err(Error::Refused(format!(
                "{}: not a GdnCore geometry",
                op.name
            )));
        };
        let edge = self.ba_edge(op)?;
        let half = u64::from(heads) * 2;
        let total = self
            .lookup(&edge)
            .ok_or_else(|| Error::Refused(format!("{}: `{edge}` is not planned", op.name)))?
            .bytes;
        out.push(("av".into(), self.at_offset(&edge, half, total - half)?));
        out.push(("bv".into(), self.at_offset(&edge, 0, total - half)?));
        Ok(())
    }
}

/// **P27-15d: the engine-owned GDN tree record of `op`'s layer** (`l<L>.gdn.tree_record`, one
/// (sequence, v head) record of `TREE_REC_FLOATS` floats each, allocated with the ledger class
/// `engine.spectree.records`).
///
/// # Errors
/// An op outside a decoder layer.
pub fn tree_record_key(op: &ModelOp) -> Result<String> {
    match op.site {
        saber_model::graph::Site::Layer(l) => Ok(format!("l{l}.gdn.tree_record")),
        _ => Err(Error::Refused(format!(
            "{}: a GDN tree update outside a decoder layer",
            op.name
        ))),
    }
}

/// The GDN geometry a saber shape is built from, named rather than positional: six `u32`s in a
/// tuple is exactly the shape of a silent swap.
struct GdnDims {
    conv_width: u32,
    head_k: u32,
    head_v: u32,
    /// Value heads per rank.
    heads: u32,
    /// Key heads per rank.
    key_heads: u32,
    /// The prefill token block.
    chunk: u32,
    /// `2 * key_heads * head_k + heads * head_v`: the width of the conv/qkv half of `in_proj`.
    qkv_n: u32,
}

/// The op that produced a prefill GDN op's conv edge, from the edge name.
/// The other half of the decode PAIR: `layer.<n>.gdn.conv_update` -> `…recurrent_update`.
///
/// The two are one dispatch pair on one site and they read the same `sidx` table — prod hands them
/// the one tensor — so the conv half names the recurrent half's key rather than opening a second
/// blob slot for the same bytes. Derived from the op's own name rather than from the graph so that
/// this is a pure function like [`conv_producer`], and a name that is not the pair's is a refusal.
fn pair_partner(op: &ModelOp) -> Result<String> {
    op.name
        .strip_suffix(".conv_update")
        .map(|stem| format!("{stem}.recurrent_update"))
        .ok_or_else(|| {
            Error::Refused(format!(
                "{}: not a `<layer>.gdn.conv_update` op, so it has no recurrent partner to share \
                 `sidx` with",
                op.name
            ))
        })
}

fn conv_producer(op: &ModelOp) -> Result<String> {
    let edge = &op.inputs[0].edge;
    // `l12.gdn.conv` -> `layer.12.gdn.conv_prep`
    let rest = edge
        .strip_prefix('l')
        .and_then(|r| r.strip_suffix(".gdn.conv"))
        .ok_or_else(|| {
            Error::Refused(format!(
                "{}: `{edge}` is not a `l<N>.gdn.conv` edge",
                op.name
            ))
        })?;
    Ok(format!("layer.{rest}.gdn.conv_prep"))
}

// =====================================================================================
// Execution
// =====================================================================================

/// One weight record, as the caller bound it: where it landed and **what the pack says it is**.
///
/// The layout id travels with the address because that is the only way the engine can check it:
/// the arena is bytes, and a record in the wrong layout is not detectable from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundWeight {
    /// Where the record landed.
    pub addr: DeviceAddr,
    /// The `layout_id` the pack's record table carries (`docs/abi/PACK.md` § Layout ids).
    pub layout_id: u32,
}

impl BoundWeight {
    /// A record at `addr` carrying `layout_id`.
    #[must_use]
    pub fn new(addr: DeviceAddr, layout_id: u32) -> Self {
        BoundWeight { addr, layout_id }
    }
}

/// Everything the block reads that the arena does not own.
#[derive(Debug, Default)]
pub struct BlockInputs {
    /// Weight record name -> where it landed and what layout the pack says it is.
    pub weights: BTreeMap<String, BoundWeight>,
    /// State name -> device address.
    pub states: BTreeMap<String, DeviceAddr>,
    /// Index-table key -> device address.
    pub indices: BTreeMap<String, DeviceAddr>,
}

/// What one op did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpReport {
    /// The op's name.
    pub name: String,
    /// Kernel symbols it dispatched, in order.
    pub symbols: Vec<String>,
    /// Dispatches issued.
    pub launches: u32,
    /// Microseconds, when the caller asked for per-op timing.
    pub micros: Option<u64>,
}

/// What one block run did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRun {
    /// Per op, in forward order.
    pub ops: Vec<OpReport>,
    /// Total dispatches.
    pub launches: u32,
}

/// The address of a bound state, at the base or at one session's slot.
///
/// A free function, not a method, for the reason [`check_weight_layout`] is one: it needs no arena
/// and no device, so the negative control — a `chunk_scan` whose slot no step ever set — exercises
/// the **real** path as a host test rather than a copy of it.
///
/// # Errors
/// A state key the caller did not bind, a slot offset that overflows the base, and
/// [`StateAt::PerStep`] — which is a step that never pointed it anywhere.
pub fn state_address(key: &str, at: StateAt, inputs: &BlockInputs) -> Result<DeviceAddr> {
    let base = inputs
        .states
        .get(key)
        .copied()
        .ok_or_else(|| Error::Refused(format!("no device state bound for `{key}`")))?;
    match at {
        StateAt::Pool => Ok(base),
        StateAt::Slot { slot, offset } => base.offset(offset).ok_or_else(|| {
            Error::Refused(format!(
                "`{key}` slot {slot} is {offset} bytes past a base that cannot hold it"
            ))
        }),
        StateAt::PerStep => Err(Error::Refused(format!(
            "`{key}` is bound per STEP and this step never pointed it at a slot: \
             `saber_gdn_chunk_scan` indexes `h0`/`ht` by `blockIdx.z`, so a prefill step must call \
             `BlockPlan::retarget_prefill_state(slot)` with its session's GDN slot. Refusing \
             rather than using the base, which is slot 0 — the kernels' null slot"
        ))),
    }
}

/// The address of a bound weight, refusing a record whose layout is not the one the op reads.
///
/// A free function, not a method, for one reason: it needs no arena and no device, so the negative
/// control ("a record with the wrong layout id for its role is refused at bind") exercises the
/// **real** path as a host test rather than a copy of it.
///
/// # Errors
/// The record not being bound at all, or carrying a different `layout_id` than the model's
/// device-final plan gives it. A record in the wrong layout is bytes of the right length that
/// compute the wrong answer, so nothing downstream could attribute the failure.
pub fn check_weight_layout(src: &Source, inputs: &BlockInputs) -> Result<DeviceAddr> {
    let Source::Weight { key, layout } = src else {
        return Err(Error::Refused(format!("{src:?} is not a weight")));
    };
    let w = inputs
        .weights
        .get(key)
        .copied()
        .ok_or_else(|| Error::Refused(format!("no device weight bound for `{key}`")))?;
    if w.layout_id != *layout {
        return Err(Error::Refused(format!(
            "record `{key}` carries layout id {} but this op reads it as layout id {layout}; \
             the pack and the model's device-final plan disagree",
            w.layout_id
        )));
    }
    Ok(w.addr)
}

/// The suffix of an arena buffer that holds a CONSTANT zero no op writes (layer 0's residual).
pub const ZERO_CONSTANT_SUFFIX: &str = ".zero_resid";

/// **T7: when the executor re-seeds a plan's constants** ([`ZERO_CONSTANT_SUFFIX`] buffers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstantSeeds {
    /// Before every run of every plan (T7, the law): a constant is zero at every read whatever
    /// the arena held, so no run depends on boot-time memory.
    EveryRun,
    /// C7's rule, the CONTROL: only over a shared arena (`shared`); an own arena relied on its
    /// boot zeroing, and T7's poison (`decode:0x3c`, `buf=layer.0.input_layernorm.zero_resid`)
    /// moved every request through exactly this.
    SharedArenaOnly {
        /// Whether the plan's arena is shared.
        shared: bool,
    },
}

/// The block's activation arena and the one stream it runs on.
///
/// Invariant I1: the arena is allocated once and never reallocated; only its contents change
/// between runs. That is what makes the block replayable — and, at P3, capturable.
pub struct BlockExecutor {
    /// The activation arena. Its OWN allocation for every plan but a prefill rung under C7's
    /// shared-arena law, where every rung's executor holds a counted reference to ONE allocation
    /// sized to the widest rung ([`BlockExecutor::with_shared_arena`]).
    arena: Rc<DevicePtr>,
    staging: PinnedBuf,
    /// Has this executor already sized the launcher's local workspaces from a plan's maximum?
    ///
    /// The reservation is idempotent but not free — it walks every op and rebuilds its bindings —
    /// and `run_steps` is the timed region of every eager measurement, so it happens on the first
    /// call and never again.
    reserved: bool,
}

impl BlockExecutor {
    /// Allocate the arena for `plan` and a pinned staging buffer for host transfers.
    ///
    /// # Errors
    /// The allocation failing, which beside a 0.95-utilisation server is the expected way to be
    /// told the block does not fit.
    pub fn new(hip: &HipThread, plan: &BlockPlan<'_>, staging_bytes: usize) -> Result<Self> {
        let arena = hip.malloc(plan.arena_bytes.max(1))?;
        let staging = hip.host_malloc(staging_bytes.max(1) as u64)?;
        Ok(BlockExecutor {
            arena: Rc::new(arena),
            staging,
            reserved: false,
        })
    }

    /// **C7: run `plan` over a SHARED arena** — one allocation that several plans lay their
    /// buffers over, instead of one of its own.
    ///
    /// Legal only for plans that never run at the same time and keep nothing in the arena from one
    /// run to the next: the prefill rungs (one chunk a step, one sequence a chunk, every buffer
    /// written before it is read inside the chunk; the head reads the body's rows and the drafter's
    /// store reads the aux tap inside the same chunk). The shared allocation is sized to the
    /// widest rung (`saber_sched::prefill_law::SharedPrefillArena` is the sizing law), so a
    /// plan's every offset is inside it; a plan wider than the allocation is refused here by name,
    /// never laid over it.
    ///
    /// # Errors
    /// A plan wider than `arena`, or the staging allocation failing.
    pub fn with_shared_arena(
        hip: &HipThread,
        plan: &BlockPlan<'_>,
        arena: &Rc<DevicePtr>,
        staging_bytes: usize,
    ) -> Result<Self> {
        if plan.arena_bytes > arena.len() {
            return Err(Error::Refused(format!(
                "a {} B plan ({:?}) laid over a {} B shared arena: the shared arena is sized to the \
                 widest plan that shares it, and this one was not counted",
                plan.arena_bytes,
                plan.bucket,
                arena.len()
            )));
        }
        let staging = hip.host_malloc(staging_bytes.max(1) as u64)?;
        Ok(BlockExecutor {
            arena: Rc::clone(arena),
            staging,
            reserved: false,
        })
    }

    /// **Re-seed the plan's CONSTANT buffers before a run** (C7 over the shared prefill arena;
    /// **T7: before EVERY run of every plan**, [`ConstantSeeds::EveryRun`]): every `.zero_resid`
    /// (layer 0's explicit zero residual, [`BlockPlan::reads_before_write`]'s one buffer that no op
    /// writes) is zeroed at its offset, on `stream`, no sync. Until T7 an own arena (every decode
    /// bucket) relied on its boot zeroing, so a run's answer was a function of boot-time memory:
    /// T7's boot poison moved every request through the decode buckets' `.zero_resid` alone.
    ///
    /// # Errors
    /// The memset failing.
    pub fn zero_constants(
        &self,
        hip: &HipThread,
        plan: &BlockPlan<'_>,
        stream: &Stream,
    ) -> Result<u32> {
        let mut n = 0u32;
        for b in plan
            .buffers
            .iter()
            .filter(|b| b.name.ends_with(ZERO_CONSTANT_SUFFIX))
        {
            hip.memset_at(&self.arena, b.offset, b.bytes, 0, stream)?;
            n += 1;
        }
        Ok(n)
    }

    /// Whether this executor's arena is shared with another executor (C7).
    #[must_use]
    pub fn arena_is_shared(&self) -> bool {
        Rc::strong_count(&self.arena) > 1
    }

    /// Bytes of arena.
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        self.arena.len()
    }

    /// The arena's base address, for a caller that needs to name a sub-range itself.
    #[must_use]
    pub fn arena_base(&self) -> DeviceAddr {
        self.arena.addr()
    }

    /// The arena itself, for a caller that copies into a planned buffer with a `Job` (D4's
    /// embed-row scatter): offsets come from [`BlockPlan::locate`].
    #[must_use]
    pub fn arena(&self) -> &DevicePtr {
        &self.arena
    }

    /// Zero the whole arena, so that a kernel which fails to write a byte shows up as a mismatch
    /// rather than as whatever the allocator handed back.
    pub fn zero(&self, hip: &HipThread, stream: &Stream) -> Result<()> {
        hip.memset(&self.arena, 0, stream)?;
        hip.sync_stream(stream)?;
        Ok(())
    }

    /// **T6 DIAGNOSTIC** (`saber engine --poison-boot`): fill the whole arena with `byte` and wait.
    /// A run whose ids move under a poisoned boot reads arena bytes it never wrote.
    pub fn fill(&self, hip: &HipThread, stream: &Stream, byte: u8) -> Result<()> {
        hip.memset(&self.arena, byte, stream)?;
        hip.sync_stream(stream)?;
        Ok(())
    }

    /// Copy `bytes` into the arena at `offset`, through the pinned staging buffer.
    pub fn upload(
        &mut self,
        hip: &HipThread,
        offset: u64,
        bytes: &[u8],
        stream: &Stream,
    ) -> Result<()> {
        let chunk = self.staging.len();
        if chunk == 0 {
            return Err(Error::Refused("staging buffer is empty".into()));
        }
        for (i, part) in bytes.chunks(chunk).enumerate() {
            {
                let mut slice = self
                    .staging
                    .slice_mut(0, part.len())
                    .ok_or_else(|| Error::Refused("staging slice out of range".into()))?;
                slice.as_mut_slice().copy_from_slice(part);
            }
            let view = self
                .staging
                .slice(0, part.len())
                .ok_or_else(|| Error::Refused("staging slice out of range".into()))?;
            hip.memcpy_h2d_at(&self.arena, offset + (i * chunk) as u64, view, stream)?;
            hip.sync_stream(stream)?;
        }
        Ok(())
    }

    /// Read `out.len()` bytes out of the arena at `offset`.
    pub fn download(
        &self,
        hip: &HipThread,
        offset: u64,
        out: &mut [u8],
        stream: &Stream,
    ) -> Result<()> {
        hip.memcpy_d2h_at_blocking(out, &self.arena, offset, stream)?;
        Ok(())
    }

    /// Read several SMALL ranges out of the arena into one pinned buffer, in **one** round trip to
    /// the HIP thread.
    ///
    /// The engine's logits gather is the caller this exists for: `logits.gather` has no code
    /// object at TP > 1, so the host reads one winning logit per verify ROW at the column that
    /// row's local argmax named (`saber_engine::device::read_picks`). At `b1s8` that is eight
    /// two-byte reads a step and at `b4s8` thirty-two, and each one was its own
    /// `memcpy_d2h_at_blocking` — a channel round trip to the HIP thread (~3.24 µs, workstream H),
    /// a `hipMemcpyAsync` and a `hipStreamSynchronize`, nine times over for what is sixteen bytes.
    ///
    /// The offsets are only known AFTER the pick table comes back, so this cannot be folded into
    /// that read; what it can be is one batch. `Job::SyncStream` rides in the same batch, so the
    /// whole gather is one round trip and the copies are complete when it returns.
    ///
    /// # Errors
    /// A destination too small for `offsets.len() * width`, a range past the arena, or the batch
    /// failing — in which case the job index that failed is logged and nothing after it was
    /// issued.
    pub fn download_scattered(
        &self,
        hip: &HipThread,
        offsets: &[u64],
        width: usize,
        dst: &mut PinnedBuf,
        stream: &Stream,
    ) -> Result<()> {
        if offsets.is_empty() {
            return Ok(());
        }
        let want = offsets.len() * width;
        if dst.len() < want {
            return Err(Error::Refused(format!(
                "a scattered download of {} range(s) of {width} byte(s) into a {}-byte buffer",
                offsets.len(),
                dst.len()
            )));
        }
        // EVERY RANGE IS CHECKED BEFORE ANYTHING IS POSTED, so a bad offset is a named refusal on
        // the calling thread and never a fault inside a batch that has already moved half of it.
        for off in offsets {
            let end = off.checked_add(width as u64).ok_or_else(|| {
                Error::Refused(format!(
                    "a scattered download at {off} of {width} overflows"
                ))
            })?;
            if end > self.arena.len() {
                return Err(Error::Refused(format!(
                    "a scattered download at {off}..{end} of a {}-byte arena",
                    self.arena.len()
                )));
            }
        }
        let views = dst
            .chunks_mut(width)
            .ok_or_else(|| Error::Refused("the gather staging will not chunk".to_owned()))?;
        let mut jobs: Vec<Job<'_>> = Vec::with_capacity(offsets.len() + 1);
        for (off, view) in offsets.iter().zip(views) {
            jobs.push(Job::MemcpyD2h {
                src: &self.arena,
                src_offset: *off,
                dst: view,
                stream,
            });
        }
        jobs.push(Job::SyncStream { stream });
        let issued = hip.run_batch(&jobs)?;
        if issued != jobs.len() {
            return Err(Error::Refused(format!(
                "a scattered download issued {issued} of {} job(s)",
                jobs.len()
            )));
        }
        Ok(())
    }

    /// The device address a [`Source`] resolves to.
    ///
    /// # Errors
    /// A weight, state or index key the caller did not provide. Fail loud: the alternative is a
    /// kernel dereferencing whatever was in that kernarg slot.
    pub fn address(&self, src: &Source, inputs: &BlockInputs) -> Result<DeviceAddr> {
        match src {
            Source::Arena { buffer, offset, .. } => self.arena.sub_addr(*offset).ok_or_else(|| {
                Error::Refused(format!(
                    "`{buffer}` at arena offset {offset} runs past the {} byte arena",
                    self.arena.len()
                ))
            }),
            Source::Weight { .. } => check_weight_layout(src, inputs),
            Source::State { key, at } => state_address(key, *at, inputs),
            Source::Index(k) => inputs
                .indices
                .get(k)
                .copied()
                .ok_or_else(|| Error::Refused(format!("no index table bound for `{k}`"))),
        }
    }

    /// Run the block once on `stream`.
    ///
    /// `per_op` records a `hipEvent` pair around each op, which serialises the stream at op
    /// granularity; without it the whole block is timed by the caller and the ops run back to
    /// back, which is what the µs/block number must be measured with.
    ///
    /// # Errors
    /// Any resolve, binding or launch failure, and — structural rule 7 — a total dispatch count
    /// that is not [`BlockPlan::expected_launches`].
    pub fn run(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        plan: &BlockPlan<'_>,
        inputs: &BlockInputs,
        per_op: bool,
    ) -> Result<BlockRun> {
        let run = self.run_steps(ctx, launcher, stream, &plan.steps, inputs, per_op)?;
        let expected = plan.expected_launches();
        if run.launches != expected {
            return Err(Error::Refused(format!(
                "block issued {} dispatches, the graph's declared budget for these ops is \
                 {expected} ({} declared minus {} the identity all-reduce replaces)",
                run.launches, plan.declared_launches, plan.identity_launches
            )));
        }
        Ok(run)
    }

    /// Run a SLICE of a plan's steps, with no launch-budget assertion.
    ///
    /// [`BlockExecutor::run`] is this plus the assertion, and is what a block gate calls. A
    /// forward pass calls this one, because it runs the plan in segments — it has to stop between
    /// the layers and the head to select the sampled rows, and it stops at a layer boundary when
    /// a caller wants the residual stream read out mid-pass. The budget is then asserted by the
    /// caller over the segments it ran, which is the same statement made once instead of N times.
    ///
    /// # Errors
    /// Any resolve, binding or launch failure.
    pub fn run_steps(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        steps: &[Step<'_>],
        inputs: &BlockInputs,
        per_op: bool,
    ) -> Result<BlockRun> {
        let plan_steps = steps;
        // Size every local workspace from the MAX over the plan before the first launch, so no
        // step pays a `hipMalloc`/`hipFree` grow chain inside its own time. The graph path has had
        // this property by accident since it prepares every op before the capture; I15's prefill
        // measurement is what it costs an eager step not to have it (ws-h21).
        if !self.reserved {
            let mut reserve: Vec<(&LaunchDescriptor, ArgBindings)> = Vec::new();
            for step in plan_steps {
                let Route::Kernel { descs, shape } = &step.route else {
                    continue;
                };
                let mut args = ArgBindings::new(*shape);
                for (name, src) in &step.bindings {
                    args.bind(name, self.address(src, inputs)?);
                }
                for desc in descs {
                    reserve.push((desc, args.clone()));
                }
            }
            let refs: Vec<(&LaunchDescriptor, &ArgBindings)> =
                reserve.iter().map(|(d, a)| (*d, a)).collect();
            launcher.reserve_local(ctx, &refs, stream)?;
            self.reserved = true;
        }
        let mut ops = Vec::with_capacity(plan_steps.len());
        let mut total = 0u32;
        for step in plan_steps {
            let Route::Kernel { descs, shape } = &step.route else {
                continue;
            };
            let mut args = ArgBindings::new(*shape);
            for (name, src) in &step.bindings {
                args.bind(name, self.address(src, inputs)?);
            }
            let refs: Vec<&LaunchDescriptor> = descs.iter().collect();
            let micros = if per_op {
                let start = ctx.hip.create_event(saber_hip::EventFlags::Default)?;
                let end = ctx.hip.create_event(saber_hip::EventFlags::Default)?;
                ctx.hip.record_event(&start, stream)?;
                let report = launcher.launch_op(ctx, &refs, &args, stream)?;
                ctx.hip.record_event(&end, stream)?;
                ctx.hip.sync_event(&end)?;
                total += report.launches;
                Some((f64::from(ctx.hip.elapsed_ms(&start, &end)?) * 1000.0).round() as u64)
            } else {
                let report = launcher.launch_op(ctx, &refs, &args, stream)?;
                total += report.launches;
                None
            };
            ops.push(OpReport {
                name: step.op.name.clone(),
                symbols: descs.iter().map(|d| d.kernel.clone()).collect(),
                launches: descs.iter().map(|d| d.n_launches).max().unwrap_or(0),
                micros,
            });
        }
        Ok(BlockRun {
            ops,
            launches: total,
        })
    }

    /// Launch one descriptor that is not part of a plan, with explicit bindings.
    ///
    /// The forward pass needs exactly three of these — the two row selections that turn a
    /// 2048-row residual stream into the one row `lm_head` is asked for, and the argmax — and
    /// none of them is in the graph: the graph's `logits.gather` is the TP > 1 vocabulary
    /// all-gather, a different op with the same English name. Rather than invent graph ops for
    /// them, the forward pass states them here and counts them in its own launch budget.
    ///
    /// # Errors
    /// Any launch failure, or a descriptor naming a buffer the caller did not bind.
    pub fn launch_extra(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
    ) -> Result<u32> {
        let report = launcher.launch_op(ctx, &[desc], args, stream)?;
        Ok(report.launches)
    }
}

impl std::fmt::Debug for BlockExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockExecutor(arena {} B)", self.arena.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::ArgValue;
    use saber_model::graph::{LayoutRule, ModelConfig, Phase};

    fn graph() -> Graph {
        Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph")
    }

    /// Read an index table as `i32`s, whatever key it is under: [`crate::forward::index_table`]
    /// dispatches on the key's last dot-component, so `"x.cache_idx"` is the step's own law for
    /// that family and nothing else about the name matters.
    fn law_i32(
        key: &str,
        index: &crate::forward::StepIndex<'_>,
    ) -> std::result::Result<Vec<i32>, String> {
        let bytes = crate::forward::index_table(key, index).map_err(|e| e.to_string())?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Every strided index kernarg of one planned step, against the table it is bound to.
    ///
    /// `control` replaces the resolved key with the per-SLOT `cache_idx` table, which is what the
    /// plan bound before ws-l17: the control must FAIL wherever the case could.
    fn strided_index_bindings(
        plan: &BlockPlan<'_>,
        index: &crate::forward::StepIndex<'_>,
        control: bool,
    ) -> std::result::Result<usize, String> {
        let n_seqs = index.seqs.len();
        let q = index.q_len() as usize;
        // The two laws a stride kernarg can be a statement about: one conv slot per SLOT, and one
        // recurrent snapshot per ROW.
        let per_slot = law_i32("x.cache_idx", index)?;
        let per_row = law_i32("x.sidx", index)?;
        let mut checked = 0usize;
        for step in &plan.steps {
            let Route::Kernel { descs, .. } = &step.route else {
                continue;
            };
            for d in descs {
                for (buffer, stride_arg, per_candidate) in [
                    ("cache_idx", "ci_stride", false),
                    ("sidx", "sidx_stride", true),
                ] {
                    let (Some(buf), Some(st)) = (d.arg(buffer), d.arg(stride_arg)) else {
                        continue;
                    };
                    let ArgValue::Buffer { buffer: bound } = &buf.value else {
                        continue;
                    };
                    let ArgValue::I64 { ival } = st.value else {
                        return Err(format!("{}: `{stride_arg}` is not an i64", step.name()));
                    };
                    let Some(Source::Index(key)) = step.source(bound) else {
                        return Err(format!(
                            "{}: `{bound}` is bound to something that is not an index table",
                            step.name()
                        ));
                    };
                    let key = if control && buffer == "cache_idx" {
                        format!("{}.cache_idx", step.name())
                    } else {
                        key.clone()
                    };
                    let table = law_i32(&key, index)?;
                    let stride = usize::try_from(ival)
                        .map_err(|_| format!("{}: `{stride_arg}` is {ival}", step.name()))?;
                    // `t` is the candidate row for a per-ROW read and 0 for a per-slot one.
                    let rows = if per_candidate { q } else { 1 };
                    for n in 0..n_seqs {
                        for t in 0..rows {
                            let at = n * stride + t;
                            let want = if per_candidate {
                                per_row[n * q + t]
                            } else {
                                per_slot[n]
                            };
                            let got = table.get(at).copied().ok_or_else(|| {
                                format!(
                                    "{}: `{key}` holds {} entries and slot {n}'s state slot is \
                                     read at {at} (`{stride_arg}` {stride})",
                                    step.name(),
                                    table.len()
                                )
                            })?;
                            if got != want {
                                return Err(format!(
                                    "{}: `{key}[{at}]` is {got} and slot {n} row {t} owns state \
                                     slot {want} (`{stride_arg}` {stride})",
                                    step.name()
                                ));
                            }
                        }
                    }
                    checked += 1;
                }
            }
        }
        Ok(checked)
    }

    /// A STRIDE KERNARG IS A STATEMENT ABOUT WHICH BUFFER IS BOUND, and the two are only checkable
    /// together.
    ///
    /// Both decode GDN kernels read their state tables strided — `slot = cache_idx[n * ci_stride]`
    /// and `si = sidx[n * sidx_stride + t]` — and prod's own call passes `sidx[:, 0][:nseq]` as
    /// `cache_idx`, i.e. the `sidx` allocation's pointer with `cache_idx.stride(0) = maxq`
    /// (`ref_gdn.py`, the decode leg). So `ci_stride = q_len` means the SIDX table, and the
    /// contiguous per-slot `cache_idx` table belongs to `ci_stride = 1`.
    ///
    /// At `q_len = 1` the two coincide and every binding looks right. Above it the per-slot table
    /// read at a per-row stride hands sequence `n` the entry of slot `n * q` — past the end of an
    /// `n_seqs`-entry table for every sequence but the first, where the blob's zero padding is
    /// the kernels' NULL state slot, so `saber_gdn_conv_update` returns having updated nothing and
    /// those rows read whatever the arena held. That is ws-s1's batched verify step: slot 0 clean,
    /// every other slot emitting the same prompt-independent token at token 1, step 0, layer 0.
    #[test]
    fn a_strided_index_kernarg_names_the_buffer_that_is_true_of_it() {
        let pages: Vec<u32> = (0..8).collect();
        let case = |g: &Graph,
                    batch: u32,
                    width: u32,
                    live: u32,
                    control: bool|
         -> std::result::Result<usize, String> {
            let plan = BlockPlan::new(
                g,
                Bucket::Decode {
                    batch,
                    verify_width: width,
                },
                &[0, 1, 2, 3, 4],
                RuntimeParams {
                    max_ctx: 128,
                    gdn_slots: 1 + batch * width,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
            )
            .map_err(|e| e.to_string())?;
            // Slot `i` owns `width` consecutive snapshots from `1 + i * width`, which is the
            // engine's own free list (`device.rs`: the bases are spaced by the widest width).
            let seqs: Vec<crate::forward::SeqFacts<'_>> = (0..batch)
                .map(|i| {
                    if i < live {
                        crate::forward::SeqFacts {
                            state_slot: 1 + i * width,
                            done_before: 64,
                            kv_blocks: &pages,
                            live_rows: width,
                            rope: None,
                        }
                    } else {
                        crate::forward::SeqFacts::padding()
                    }
                })
                .collect();
            let ids = vec![7i32; (batch * width) as usize];
            let index = crate::forward::StepIndex {
                phase: Phase::Decode,
                rows: batch * width,
                max_blocks: 8,
                seqs: &seqs,
                ids: &ids,
                band_width: None,
            };
            index.check().map_err(|e| e.to_string())?;
            strided_index_bindings(&plan, &index, control)
        };

        // TP=1 runs the conv/recurrent PAIR at every batch (48 value heads against
        // `fused_max_items` 32); TP=2 runs the fused kernel at batch 1 and the pair above it. So
        // both readers of `cache_idx` are covered, and the batched shapes are the ones ws-s1 ran.
        let one = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph at TP=1");
        let two = graph();
        for g in [&one, &two] {
            for (batch, width, live) in [
                (1, 1, 1),
                (8, 1, 8),
                (1, 5, 1),
                (2, 5, 2),
                (8, 5, 4),
                (8, 5, 8),
                (4, 8, 4),
            ] {
                let checked = case(g, batch, width, live, false)
                    .unwrap_or_else(|e| panic!("b{batch}s{width} ({live} live): {e}"));
                assert!(
                    checked >= 5,
                    "b{batch}s{width}: {checked} strided binding(s) checked; five GDN layers bind \
                     at least one each"
                );
            }
        }

        // NEGATIVE CONTROL: the per-SLOT table under the same `ci_stride`, which is what the plan
        // bound until ws-l17. It must be right at width 1 and wrong at every width above it, or
        // this test cannot fail.
        for g in [&one, &two] {
            assert!(
                case(g, 8, 1, 8, true).is_ok(),
                "at width 1 the per-slot table IS the strided view: the control must agree here"
            );
            let err = case(g, 8, 5, 8, true)
                .expect_err("the per-slot table read at stride 5 must not pass");
            assert!(
                err.contains("cache_idx"),
                "the control must fail on the conv slot table: {err}"
            );
        }
    }

    /// **A ROW BAND's plan (C2)**: the same op list and the same launch budget as its rectangle
    /// (`b{batch}s{max width}`), whatever its row count — rule 7's "the op list does not depend on
    /// the width multiset" — with every decode attention on the RAGGED pair (its `cu_q` bound to
    /// the layer's own `.cu` table, its combine over the band's rows), every GDN kernel at the
    /// rectangle's strides (`ci_stride = sidx_stride = maxq =` the max width) with its output row
    /// count at the band's, and every strided index binding reading the right slot for every one
    /// of the max width's candidates at RAGGED live widths.
    /// **P27-15d: the b1 tree bucket plans the TREE FORM** -- every GDN layer's core is the tree
    /// conv -> the WY tree update (adv1) -> the shipped `gdn_norm_quant`, every attention layer's
    /// the ancestor-word ragged pair; no chain-only route survives; the conv cache is laid out
    /// for the tree's 10 rows in EVERY bucket of the graph (the chain's walk unchanged) while the
    /// recurrent snapshots stay the chain's; the tree tables are bound by name and the record by
    /// the layer's engine-owned key. The budget is the chain b1 bucket's + 1 a GDN layer.
    /// **E4-3: the 16-row tree bucket** (`--engine-spec-tree best16`) plans the 16-row objects:
    /// the GDN tree at `p27_tree_gdn16` (record 4,128 floats a head), the conv walk 3 + 15, the
    /// attention on the 6-warp `p27_attn_decode_anc16_*_nw6` (192 threads, `q_len` 16 in its
    /// kernargs, the anc table at the same offset) + the shipped ragged combine at 16 x 12 jobs.
    #[test]
    fn e4_3_the_16_row_tree_bucket_plans_the_16_row_objects() {
        let w = 8u32;
        let g = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, w)
            .expect("graph")
            .with_gdn_state_f16()
            .expect("f16 state")
            .with_tree_conv_rows(16)
            .expect("tree conv rows");
        let layers: Vec<u32> = (0..8).collect();
        let tree = BlockPlan::new(
            &g,
            Bucket::DecodeTree { rows: 16 },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 1 + 8 * w,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
        )
        .unwrap_or_else(|e| panic!("the 16-row tree bucket: {e}"));
        let ival = |d: &saber_kernels::rules::LaunchDescriptor, a: &str| -> i64 {
            match d.arg(a).map(|s| s.value.clone()) {
                Some(saber_kernels::rules::ArgValue::I64 { ival }) => ival,
                Some(saber_kernels::rules::ArgValue::I32 { ival }) => ival,
                other => panic!("{}: `{a}` = {other:?}", d.kernel),
            }
        };
        let (mut gdn, mut attn, mut conv) = (0u32, 0u32, 0u32);
        for s in &tree.steps {
            let n = s.name();
            let Route::Kernel { descs, .. } = &s.route else {
                continue;
            };
            if n.ends_with(".gdn.conv_tree") {
                conv += 1;
                assert_eq!(
                    ival(&descs[0], "state_len_max"),
                    3 + 15,
                    "the walk: 16 rows"
                );
                assert_eq!(ival(&descs[0], "cs_dim"), 3 + 15, "the layout: 16 rows");
            }
            if n.ends_with(".gdn.tree_update") {
                gdn += 1;
                let d = &descs[0];
                assert_eq!(d.kernel, "p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs1_adv1");
                assert_eq!(d.plugin, "p27_tree_gdn16");
                assert_eq!(ival(d, "ts_head"), 4128);
                assert_eq!(ival(d, "ts_seq"), 24 * 4128);
            }
            if n.ends_with(".attn.tree") {
                attn += 1;
                assert_eq!(descs[0].kernel, "p27_attn_decode_anc16_h256_gqa6_fp8kv_nw6");
                assert_eq!(descs[0].plugin, "p27_tree_attn16");
                assert_eq!(descs[0].block, [192, 1, 1]);
                let anc = descs[0].arg("anc").expect("anc");
                assert_eq!((anc.offset, descs[0].kernarg_bytes), (152, 160));
                assert_eq!(
                    descs[1].grid[0],
                    16 * 12,
                    "the combine: 16 rows x 12 q heads"
                );
            }
        }
        assert_eq!(
            (conv, gdn, attn),
            (6, 6, 2),
            "layers 0..8: six GDN, two attention"
        );
    }

    /// CMP27L: under the `gdn_conv_rows` knob the tree bucket's conv takes the rows-parallel twin
    /// with the serial kernel's kernargs byte for byte (one workgroup a unit, 10 waves); the
    /// control without the knob keeps the serial `hpb` kernel, and no other step moves.
    #[test]
    fn cmp27l_the_conv_rows_knob_moves_only_the_tree_conv() {
        let g = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8)
            .expect("graph")
            .with_gdn_state_f16()
            .expect("f16 state")
            .with_tree_conv_rows(10)
            .expect("tree conv rows");
        let layers: Vec<u32> = (0..8).collect();
        let plan_with = |knob: bool| {
            let mut rules = LaunchRules::prod_mxfp4().expect("rules");
            rules.knobs.gdn_conv_rows = knob;
            BlockPlan::new(
                &g,
                Bucket::DecodeTree { rows: 10 },
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 1 + 8 * 8,
                },
                &rules,
                &SaberRules::prod_saber().expect("saber rules"),
            )
            .expect("the tree plan")
            .steps
            .iter()
            .map(|s| {
                let d = match &s.route {
                    Route::Kernel { descs, .. } => descs.clone(),
                    _ => Vec::new(),
                };
                (s.name().to_owned(), d)
            })
            .collect::<Vec<_>>()
        };
        let (off, on) = (plan_with(false), plan_with(true));
        assert_eq!(off.len(), on.len());
        let mut convs = 0;
        for ((n0, d0), (n1, d1)) in off.iter().zip(&on) {
            assert_eq!(n0, n1);
            if n0.ends_with(".gdn.tree_update") {
                // the batched flush advance (CMP27L `adv`): the per-layer advance's strides
                let one = saber_kernels::saber::tree_gdn_advance_like(&d0[0]).expect("advance");
                let all = saber_kernels::saber::tree_gdn_advance_layers(&one, 48).expect("layers");
                assert_eq!(all.grid, [48, one.grid[1], 1]);
                for imm in ["st_slot", "st_head", "ts_seq", "ts_head"] {
                    assert_eq!(
                        all.arg(imm).map(|s| &s.value),
                        one.arg(imm).map(|s| &s.value),
                        "{imm}"
                    );
                }
                assert!(
                    saber_kernels::saber::tree_gdn_advance_layers(&d0[0], 48).is_err(),
                    "the control: the update descriptor is refused"
                );
            }
            if n0.ends_with(".gdn.conv_tree") {
                convs += 1;
                let (a, b) = (&d0[0], &d1[0]);
                assert!(
                    a.kernel.starts_with("p27_gdn_conv_tree_w4_h128_hpb"),
                    "{}",
                    a.kernel
                );
                assert_eq!(b.kernel, saber_kernels::saber::TREE_CONV_ROWS_SYMBOL);
                assert_eq!(a.args, b.args, "the kernargs are the serial kernel's");
                assert_eq!(b.plugin, "p27_tree_rows");
                assert_eq!(a.kernarg_bytes, b.kernarg_bytes);
                assert_eq!(b.block, [320, 1, 1]);
                assert_eq!(b.grid, [a.grid[0], a.grid[1] * a.block[0] / 32, 1]);
            } else {
                assert_eq!(d0, d1, "{n0} moved under the conv knob");
            }
        }
        assert_eq!(convs, 6, "the 8-layer slice's GDN layers");
    }

    #[test]
    fn p27_15d_the_tree_bucket_plans_the_tree_form() {
        let w = 8u32;
        let base = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, w)
            .expect("graph")
            .with_gdn_state_f16()
            .expect("f16 state");
        let g = base
            .clone()
            .with_tree_conv_rows(10)
            .expect("tree conv rows");
        let layers: Vec<u32> = (0..8).collect();
        fn plan_at<'g>(g: &'g Graph, bucket: Bucket, layers: &[u32], w: u32) -> BlockPlan<'g> {
            BlockPlan::new(
                g,
                bucket,
                layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 1 + 8 * w,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
            )
            .unwrap_or_else(|e| panic!("{bucket:?}: {e}"))
        }
        let plan = |g, b| plan_at(g, b, &layers, w);
        let tree = plan(&g, Bucket::DecodeTree { rows: 10 });
        let chain = plan(
            &g,
            Bucket::Decode {
                batch: 1,
                verify_width: w,
            },
        );
        let chain0 = plan(
            &base,
            Bucket::Decode {
                batch: 1,
                verify_width: w,
            },
        );
        let launches =
            |p: &BlockPlan<'_>| -> u32 { p.steps.iter().map(|s| s.route.launches()).sum() };
        let ival = |d: &saber_kernels::rules::LaunchDescriptor, a: &str| -> i64 {
            match d.arg(a).map(|s| &s.value) {
                Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => *ival,
                other => panic!("{}: `{a}` = {other:?}", d.kernel),
            }
        };
        let (mut gdn, mut attn) = (0u32, 0u32);
        for s in &tree.steps {
            let n = s.name();
            for chain_only in [
                ".gdn.fused_update",
                ".gdn.conv_update",
                ".gdn.recurrent_update",
                ".attn.decode",
            ] {
                assert!(!n.ends_with(chain_only), "{n} in the tree bucket");
            }
            let Route::Kernel { descs, .. } = &s.route else {
                continue;
            };
            if n.ends_with(".gdn.conv_tree") {
                let d = &descs[0];
                assert!(
                    d.kernel.starts_with("p27_gdn_conv_tree_w4_h128_hpb"),
                    "{}",
                    d.kernel
                );
                assert_eq!(d.plugin, "p27_tree_conv");
                assert_eq!(ival(d, "state_len_max"), 3 + 9, "the walk: 10 rows");
                assert_eq!(ival(d, "cs_dim"), 3 + 9, "the layout: 10 rows");
                assert_eq!(ival(d, "ci_stride"), 1);
                for t in ["hsrc", "tsrc", "cache_idx", "cu"] {
                    assert_eq!(
                        s.source(t),
                        Some(&Source::Index(format!("{n}.{t}"))),
                        "{n} {t}"
                    );
                }
            }
            if n.ends_with(".gdn.tree_update") {
                gdn += 1;
                let d = &descs[0];
                assert_eq!(d.kernel, "p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs1_adv1");
                assert_eq!(d.grid, [1, 24, 1]);
                assert_eq!(d.block, [256, 1, 1]);
                assert_eq!(ival(d, "ts_head"), 2592);
                assert_eq!(ival(d, "ts_seq"), 24 * 2592);
                let layer = n.split('.').nth(1).expect("layer");
                let rec = Source::State {
                    key: format!("l{layer}.gdn.tree_record"),
                    at: StateAt::Pool,
                };
                assert_eq!(s.source("ts_in"), Some(&rec));
                assert_eq!(s.source("ts_out"), Some(&rec));
                assert_eq!(s.source("slot"), Some(&Source::Index(format!("{n}.tslot"))));
                assert_eq!(s.source("acc"), Some(&Source::Index(format!("{n}.acc"))));
                assert_eq!(s.source("anc"), Some(&Source::Index(format!("{n}.anc"))));
            }
            if n.ends_with(".attn.tree") {
                attn += 1;
                assert!(descs[0]
                    .kernel
                    .starts_with("p27_attn_decode_anc_h256_gqa6_"));
                assert_eq!(descs[0].plugin, "p27_tree_attn");
                let anc = descs[0].arg("anc").expect("anc");
                assert_eq!((anc.offset, descs[0].kernarg_bytes), (152, 160));
                assert!(descs[1]
                    .kernel
                    .starts_with("saber_attn_splitkv_combine_ragged_"));
                assert_eq!(
                    descs[1].grid[0],
                    10 * 12,
                    "the combine: 10 rows x 12 q heads a rank"
                );
                assert_eq!(s.source("cu_q"), Some(&Source::Index(format!("{n}.cu"))));
            }
        }
        assert_eq!((gdn, attn), (6, 2), "layers 0..8: six GDN, two attention");
        assert_eq!(
            launches(&tree),
            launches(&chain) + gdn,
            "+1 launch a GDN layer"
        );
        // The chain's conv: the WALK is its own width, the LAYOUT the graph's (widened or not).
        let conv_of = |p: &BlockPlan<'_>| -> (i64, i64) {
            let s = p
                .steps
                .iter()
                .find(|s| s.name().ends_with(".gdn.fused_update"))
                .expect("the b1 chain's fused update");
            let Route::Kernel { descs, .. } = &s.route else {
                panic!("fused update unrouted")
            };
            (ival(&descs[0], "cs_dim"), ival(&descs[0], "state_len_max"))
        };
        assert_eq!(conv_of(&chain), (12, 10), "the chain at the widened layout");
        assert_eq!(conv_of(&chain0), (10, 10), "the chain's own layout");
        assert_eq!(g.conv_layout_rows(5120), 10);
        assert_eq!(base.conv_layout_rows(5120), 8);
        // The recurrent snapshots are untouched; only the conv cache grew (by 2 of 10 entries).
        let bytes = |g: &Graph, suffix: &str| -> u64 {
            g.ops
                .iter()
                .flat_map(|o| o.state.iter())
                .find(|st| st.name.ends_with(suffix))
                .and_then(|st| st.bytes_per_slot)
                .expect("state")
        };
        assert_eq!(
            bytes(&g, ".gdn.recurrent_state"),
            bytes(&base, ".gdn.recurrent_state")
        );
        assert_eq!(
            bytes(&g, ".gdn.conv_state") * 10,
            bytes(&base, ".gdn.conv_state") * 12
        );
        // The tree's tables are named keys of its blob and of no chain bucket's.
        for t in ["anc", "tsrc", "hsrc", "acc", "tslot"] {
            assert!(
                tree.index_keys()
                    .iter()
                    .any(|k| k.ends_with(&format!(".{t}"))),
                "{t}"
            );
            assert!(
                !chain
                    .index_keys()
                    .iter()
                    .any(|k| k.ends_with(&format!(".{t}"))),
                "{t}"
            );
        }
    }

    #[test]
    fn a_row_band_plans_the_rectangles_ops_with_the_ragged_attention() {
        let (batch, w) = (8u32, 8u32);
        for tp in [1u32, 2] {
            let g = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), tp, w).expect("graph");
            let layers: Vec<u32> = (0..8).collect();
            let at = |bucket: Bucket| {
                BlockPlan::new(
                    &g,
                    bucket,
                    &layers,
                    RuntimeParams {
                        max_ctx: 4096,
                        gdn_slots: 1 + batch * w,
                    },
                    &LaunchRules::prod_mxfp4().expect("rules"),
                    &SaberRules::prod_saber().expect("saber rules"),
                )
                .unwrap_or_else(|e| panic!("{bucket:?}: {e}"))
            };
            let rect = at(Bucket::Decode {
                batch,
                verify_width: w,
            });
            let launches =
                |p: &BlockPlan<'_>| -> u32 { p.steps.iter().map(|s| s.route.launches()).sum() };
            for rows in [16u32, 40, 64] {
                let band = at(Bucket::DecodeBand {
                    batch,
                    verify_width: w,
                    rows,
                });
                assert_eq!(
                    band.steps.len(),
                    rect.steps.len(),
                    "tp{tp} r{rows}: the op list"
                );
                assert_eq!(
                    launches(&band),
                    launches(&rect),
                    "tp{tp} r{rows}: the budget"
                );
                let mut attn = 0;
                for (b, r) in band.steps.iter().zip(&rect.steps) {
                    assert_eq!(b.name(), r.name());
                    let (Route::Kernel { descs: bd, .. }, Route::Kernel { descs: rd, .. }) =
                        (&b.route, &r.route)
                    else {
                        continue;
                    };
                    if b.name().ends_with(".attn.decode") {
                        attn += 1;
                        assert!(
                            bd[0].kernel.starts_with("saber_attn_decode_ragged_"),
                            "{}",
                            bd[0].kernel
                        );
                        assert_eq!(
                            bd[0].grid, rd[0].grid,
                            "the split law and the sequence axis"
                        );
                        assert!(bd[1]
                            .kernel
                            .starts_with("saber_attn_splitkv_combine_ragged_"));
                        let q_heads = rd[1].grid[0] / (batch * w);
                        assert_eq!(bd[1].grid[0], rows * q_heads, "the combine covers the band");
                        let cu_key = format!("{}.cu", b.name());
                        assert_eq!(b.source("cu_q"), Some(&Source::Index(cu_key.clone())));
                        assert!(band.index_keys().contains(&cu_key));
                        assert!(
                            !rect.index_keys().contains(&cu_key),
                            "a rectangle binds no cu_q"
                        );
                        continue;
                    }
                    // The GDN decode kernels resolve to the rectangle's symbols and keep every
                    // stride; only the output rows they zero past `cu[N]` move. (The GEMMs run
                    // at the band's M, so their tiles are the band's own.)
                    for (x, y) in bd.iter().zip(rd) {
                        if !y.kernel.starts_with("saber_gdn_") {
                            continue;
                        }
                        assert_eq!(x.kernel, y.kernel, "{}", b.name());
                        for key in [
                            "ci_stride",
                            "sidx_stride",
                            "maxq",
                            "state_len_max",
                            "cs_dim",
                        ] {
                            assert_eq!(
                                x.arg(key).map(|a| a.value.clone()),
                                y.arg(key).map(|a| a.value.clone()),
                                "{}: {key}",
                                b.name()
                            );
                        }
                    }
                }
                assert_eq!(attn, 2, "tp{tp}: eight layers hold two attention layers");
                // The strided bindings at RAGGED widths: every candidate of the max width reads
                // its own run's slot, whatever the slot feeds this step.
                let pages: Vec<u32> = (0..8).collect();
                let widths = [5u32, 3, 8, 2, 5, 1, 4, 5];
                let mut seqs: Vec<crate::forward::SeqFacts<'_>> = Vec::new();
                let mut used = 0u32;
                for (i, wi) in widths.iter().enumerate() {
                    let live = used + wi <= rows;
                    seqs.push(if live {
                        used += wi;
                        crate::forward::SeqFacts {
                            state_slot: 1 + i as u32 * w,
                            done_before: 64,
                            kv_blocks: &pages,
                            live_rows: *wi,
                            rope: None,
                        }
                    } else {
                        crate::forward::SeqFacts::padding()
                    });
                }
                // Live slots first: drop a live slot that follows a padding one.
                let mut seen_pad = false;
                for s in &mut seqs {
                    if !s.live() {
                        seen_pad = true;
                    } else if seen_pad {
                        *s = crate::forward::SeqFacts::padding();
                    }
                }
                let ids = vec![7i32; rows as usize];
                let index = crate::forward::StepIndex {
                    phase: Phase::Decode,
                    rows,
                    max_blocks: 8,
                    seqs: &seqs,
                    ids: &ids,
                    band_width: Some(w),
                };
                index.check().expect("a legal band");
                let checked = strided_index_bindings(&band, &index, false)
                    .unwrap_or_else(|e| panic!("tp{tp} r{rows}: {e}"));
                assert!(checked >= 6, "tp{tp} r{rows}: {checked} strided binding(s)");
            }
        }
    }

    fn plan_for(g: &Graph, bucket: Bucket) -> BlockPlan<'_> {
        BlockPlan::new(
            g,
            bucket,
            &[0, 1, 2, 3, 4],
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 7,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
        )
        .expect("plan")
    }

    /// The whole model at TP = 1, planned: every site, every layer, one arena.
    fn forward_plan(g: &Graph, bucket: Bucket) -> BlockPlan<'_> {
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        BlockPlan::with_options(
            g,
            bucket,
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 1,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions::BODY,
        )
        .expect("forward plan")
    }

    /// The decode attention's split count is a PLAN fact under
    /// [`PlanOptions::batch_invariant_attn`], and a BATCH fact without it.
    ///
    /// `saber_attn_decode`'s grid is `dim3(splits, kv_heads, num_seqs)`, so `grid[0]` IS the split
    /// count and this reads the number the kernel will be launched with rather than a field on the
    /// way to it. The negative control is the same two buckets with the flag off: they must
    /// DISAGREE, because that disagreement is the defect — 32 segments for a sequence that is
    /// alone and 16 for the same sequence in a batch of four, which partitions any context past
    /// 256 keys differently and gives the online-softmax combine a different sum to fold.
    #[test]
    fn the_decode_attention_split_count_is_the_plan_s_not_the_batch_s() {
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let splits_at = |batch: u32, opts: PlanOptions| -> u32 {
            let plan = BlockPlan::with_options(
                &g,
                Bucket::Decode {
                    batch,
                    verify_width: 1,
                },
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 41,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
                opts,
            )
            .expect("a decode plan");
            let step = plan
                .steps
                .iter()
                .find(|s| s.op.name.ends_with(".attn.decode"))
                .expect("a decode attention step");
            let Route::Kernel { descs, .. } = &step.route else {
                panic!("the decode attention is not a kernel route");
            };
            descs[0].grid[0]
        };
        let invariant = PlanOptions::BODY.with_batch_invariant_attn();
        assert_eq!(
            splits_at(1, invariant),
            splits_at(4, invariant),
            "a sequence's split-KV partition must not depend on how many sequences share the step"
        );
        assert_eq!(
            splits_at(4, invariant),
            splits_at(8, invariant),
            "and it must be the same number in every bucket of the graph"
        );
        // NEGATIVE CONTROL: prod's law, which is what every capture gate must keep asking. If this
        // ever stops differing, the flag above has stopped meaning anything and the test that says
        // it works has stopped being able to fail.
        assert_ne!(
            splits_at(1, PlanOptions::BODY),
            splits_at(4, PlanOptions::BODY),
            "prod's law divides by `kv_heads * num_seqs`: one sequence and four must differ, or \
             this test cannot fail"
        );
    }

    /// **THE PAGED ATTENTION READS A SEQUENCE'S BLOCK-TABLE ROW AT THE WIDTH THE CALLER LAYS IT
    /// OUT AT** (ws-p12). The kernel finds row `seq` at `block_table + seq * max_blocks`, so a
    /// caller whose table is wider than `max_ctx / 16` (the engine's, since ws-k14: 257 pages at
    /// `max_ctx` 4096) must be able to say so, and both attention routes — the decode and the
    /// prefill leg — must carry it. The NEGATIVE CONTROL is the plan without it: 256, which is
    /// the stride every batched serve step read a 257-wide table at, and which put sequence `i`'s
    /// first `i` pages on sequence `i - 1`'s last one. It must also leave the split law alone:
    /// `max_ctx` is not moved, so the decode grid is the same with and without.
    #[test]
    fn the_attention_reads_the_block_table_at_the_width_it_is_laid_out_at() {
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let read = |bucket: Bucket, opts: PlanOptions, suffix: &str| -> (i64, [u32; 3]) {
            let plan = BlockPlan::with_options(
                &g,
                bucket,
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 41,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
                opts,
            )
            .expect("a plan");
            let step = plan
                .steps
                .iter()
                .find(|s| s.op.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("no step ending {suffix}"));
            let Route::Kernel { descs, .. } = &step.route else {
                panic!("{suffix} is not a kernel route");
            };
            let Some(saber_kernels::rules::Slot {
                value: saber_kernels::rules::ArgValue::I32 { ival },
                ..
            }) = descs[0].arg("max_blocks")
            else {
                panic!("{suffix} carries no i32 `max_blocks`");
            };
            (*ival, descs[0].grid)
        };
        let decode = Bucket::Decode {
            batch: 8,
            verify_width: 8,
        };
        let prefill = g.bucket_for(2048, Phase::Prefill);
        let wide = PlanOptions::BODY
            .with_batch_invariant_attn()
            .with_table_blocks(257);
        let (d_wide, g_wide) = read(decode, wide, ".attn.decode");
        let (p_wide, _) = read(prefill, wide, ".attn.prefill");
        assert_eq!(
            d_wide, 257,
            "the decode attention must read the caller's row width"
        );
        assert_eq!(p_wide, 257, "and so must the prefill leg");
        // NEGATIVE CONTROL: the width `max_ctx` implies, which is what ws-k14's table was read at.
        let narrow = PlanOptions::BODY.with_batch_invariant_attn();
        let (d_narrow, g_narrow) = read(decode, narrow, ".attn.decode");
        assert_eq!(
            d_narrow, 256,
            "without it the row width is max_ctx / 16, or this cannot fail"
        );
        // And the arithmetic is untouched: the split count is the grid's x, a function of max_ctx.
        assert_eq!(
            g_wide, g_narrow,
            "the row width must not move the split law"
        );
    }

    #[test]
    fn the_layer_local_name_is_the_slot_key() {
        assert_eq!(layer_local("l7.gdn.qkvz"), Some("gdn.qkvz"));
        assert_eq!(layer_local("l63.resid.b"), Some("resid.b"));
        assert_eq!(
            layer_local("layer.12.gdn.norm.gates"),
            Some("gdn.norm.gates")
        );
        // No layer prefix: the embedding's output, the head's edges, and anything else that
        // exists once per forward pass. These get their own buffers, which is what they need.
        assert_eq!(layer_local("hidden.0"), None);
        assert_eq!(layer_local("head.in_q8"), None);
        assert_eq!(layer_local("logits.partial"), None);
        // A name that LOOKS like one but is not: no digits, or no dot after them.
        assert_eq!(layer_local("lm_head.w"), None);
        assert_eq!(layer_local("logits"), None);
    }

    /// The reason slot reuse exists: 64 layers of distinct edges do not fit, and 64 layers
    /// sharing one layer's worth of edges do.
    #[test]
    fn a_whole_forward_pass_reuses_one_layers_worth_of_arena() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph at TP=1");
        let bucket = g.bucket_for(2048, Phase::Prefill);
        let shared = forward_plan(&g, bucket);
        // Distinct buffers per edge, same ops: the arena the same plan would need without reuse.
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let distinct = BlockPlan::with_options(
            &g,
            bucket,
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 1,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions {
                reuse_layer_slots: false,
                ..PlanOptions::BODY
            },
        )
        .expect("plan without reuse");
        assert_eq!(
            shared.steps.len(),
            distinct.steps.len(),
            "reuse changes the arena, never the work"
        );
        // 64 layers collapse to roughly one layer's edges plus the two ends. The exact ratio is
        // data, not a target; what is asserted is the order of magnitude that makes the pass
        // possible at all, and that the unshared plan is the one that does not fit a 32 GiB card.
        println!(
            "BODY arena at 2048 tokens: shared {} MiB, distinct {} MiB",
            shared.arena_bytes >> 20,
            distinct.arena_bytes >> 20
        );
        assert!(
            shared.arena_bytes * 40 < distinct.arena_bytes,
            "shared {} MiB vs distinct {} MiB",
            shared.arena_bytes >> 20,
            distinct.arena_bytes >> 20
        );
        assert!(
            distinct.arena_bytes > 16 << 30,
            "the whole point: {} MiB does not fit beside 16.7 GiB of weights",
            distinct.arena_bytes >> 20
        );
    }

    /// The bf16 head (N2) plans at every row count the engine asks a head for — the prefill head's
    /// one row, the decode heads, the prompt-row block's 64 — as the head plugin's norm and the
    /// drafter's bf16 GEMM, two launches as the MXFP4 head has, binding ONE weight record at its
    /// layout-14 id (the MXFP4 head binds three). The control is the default graph at the same
    /// rows: the W4A8 trio.
    #[test]
    fn the_bf16_head_plans_its_two_launches_at_every_head_row_count() {
        let base = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8).expect("graph");
        let bf = base.clone().with_bf16_head().expect("bf16 head");
        fn plan(g: &Graph, rows: u32) -> BlockPlan<'_> {
            BlockPlan::with_options(
                g,
                saber_model::graph::Bucket::Decode {
                    batch: rows,
                    verify_width: 1,
                },
                &[],
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 1,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
                PlanOptions {
                    rank: 0,
                    ..PlanOptions::HEAD
                },
            )
            .expect("head plan")
        }
        for (rows, gemm) in [
            (1, "dflash_bgemm_bf16_m16n16"),
            (8, "dflash_bgemm_bf16_m16n16"),
            (32, "dflash_bgemm_bf16_m32n16"),
            (64, "dflash_bgemm_bf16_m32n16"),
        ] {
            let p = plan(&bf, rows);
            let kernels: Vec<&str> = p
                .steps
                .iter()
                .flat_map(|s| match &s.route {
                    Route::Kernel { descs, .. } => {
                        descs.iter().map(|d| d.kernel.as_str()).collect::<Vec<_>>()
                    }
                    _ => vec![],
                })
                .collect();
            assert_eq!(kernels, vec!["saber_add_rms_bf16", gemm], "rows {rows}");
            assert_eq!(
                p.weight_keys()
                    .into_iter()
                    .filter(|k| k.starts_with("lm_head"))
                    .collect::<Vec<_>>(),
                vec!["lm_head.w".to_owned()]
            );
            let control = plan(&base, rows);
            assert_eq!(
                control
                    .weight_keys()
                    .into_iter()
                    .filter(|k| k.starts_with("lm_head"))
                    .count(),
                3,
                "the MXFP4 head binds w/ws/wref"
            );
        }
    }

    /// The fp8-weight head (N4) plans the bf16 head's norm and the fp8 bands at every head row
    /// count, two launches like every head, binding its TWO records `lm_head.w8` (layout 15) and
    /// `lm_head.ws` (layout 16) -- the second being the kernarg the bf16 head does not have. The
    /// control is the bf16 head at the same rows: one record, and the bf16 symbols.
    #[test]
    fn the_fp8_head_plans_its_two_launches_and_binds_both_records() {
        let base = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8).expect("graph");
        let f8 = base.clone().with_fp8_head().expect("fp8 head");
        let bf = base.clone().with_bf16_head().expect("bf16 head");
        fn plan(g: &Graph, rows: u32) -> BlockPlan<'_> {
            BlockPlan::with_options(
                g,
                saber_model::graph::Bucket::Decode {
                    batch: rows,
                    verify_width: 1,
                },
                &[],
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 1,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
                PlanOptions {
                    rank: 0,
                    ..PlanOptions::HEAD
                },
            )
            .expect("head plan")
        }
        let kernels = |p: &BlockPlan<'_>| -> Vec<String> {
            p.steps
                .iter()
                .flat_map(|s| match &s.route {
                    Route::Kernel { descs, .. } => {
                        descs.iter().map(|d| d.kernel.clone()).collect::<Vec<_>>()
                    }
                    _ => vec![],
                })
                .collect()
        };
        let heads = |p: &BlockPlan<'_>| -> Vec<String> {
            p.weight_keys()
                .into_iter()
                .filter(|k| k.starts_with("lm_head"))
                .collect()
        };
        // H33: at 40 rows (b8s5) the fp8 head takes its one-pass 48-row band where the bf16 head
        // runs the 32-row band twice; at 64 both keep the 32-row band.
        for (rows, w8band, band) in [
            (1, "m16n16", "m16n16"),
            (8, "m16n16", "m16n16"),
            (32, "m32n16", "m32n16"),
            (40, "m48n16", "m32n16"),
            (64, "m32n16", "m32n16"),
        ] {
            let p = plan(&f8, rows);
            let k = kernels(&p);
            assert_eq!(k.len(), 2, "rows {rows}: {k:?}");
            assert_eq!(k[0], "saber_add_rms_bf16", "rows {rows}");
            assert!(
                k[1].starts_with(&format!("dflash_bgemm_w8_{w8band}")),
                "rows {rows}: {}",
                k[1]
            );
            let mut h = heads(&p);
            h.sort();
            assert_eq!(h, vec!["lm_head.w8".to_owned(), "lm_head.ws".to_owned()]);
            let c = plan(&bf, rows);
            assert_eq!(
                kernels(&c)[1],
                format!("dflash_bgemm_bf16_{band}"),
                "rows {rows}"
            );
            assert_eq!(heads(&c), vec!["lm_head.w".to_owned()]);
        }
    }

    /// Both ends are in the plan, and they are the ops the graph says they are.
    #[test]
    fn the_forward_plan_runs_from_the_embedding_to_the_logits() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph at TP=1");
        let p = forward_plan(&g, g.bucket_for(128, Phase::Prefill));
        assert_eq!(
            p.steps.first().map(|s| s.op.name.as_str()),
            Some("embed_tokens")
        );
        assert_eq!(
            p.steps.last().map(|s| s.op.name.as_str()),
            Some("layer.63.mlp.down_proj"),
            "the body ends at the last layer; the head is planned at the sampled row count"
        );
        // The embedding is a KERNEL now, not a stand-in: that is what the head plugin added.
        assert!(matches!(p.steps[0].route, Route::Kernel { .. }));
        // The head, planned on its own at one row. NO layers: `PlanOptions::HEAD` selects the
        // head SITE, and the layer list is what selects the layer site.
        let p = BlockPlan::with_options(
            &g,
            g.bucket_for(1, Phase::Decode),
            &[],
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 1,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions::HEAD,
        )
        .expect("head plan");
        assert_eq!(p.steps.last().map(|s| s.op.name.as_str()), Some("lm_head"));
        // TWO ops, and no more: `PlanOptions::HEAD` selects the head SITE, and the layer list is
        // what selects the layer site. Passing the body's layer list to a head plan planned the
        // whole model a second time, which cost one wrong run to notice.
        assert_eq!(
            p.steps
                .iter()
                .map(|s| s.op.name.as_str())
                .collect::<Vec<_>>(),
            vec!["model.norm", "lm_head"]
        );
        // The final norm needs NO new kernel: the graph models it as the same fused
        // add+RMS+quant every layer boundary uses, with the last layer's residual as its second
        // input. Asserting it here is the written confirmation the P3 brief asked for.
        let norm = p
            .steps
            .iter()
            .find(|s| s.op.name == "model.norm")
            .expect("the final norm is in the plan");
        assert_eq!(norm.op.kernel, KernelOp::AddRmsQuant);
        assert_eq!(norm.op.inputs.len(), 2, "hidden and the last residual");
        match &norm.route {
            Route::Kernel { descs, .. } => {
                assert!(
                    descs[0].kernel.starts_with("add_rms_quant"),
                    "{:?}",
                    descs[0].kernel
                )
            }
            other => panic!("the final norm routed to {other:?}"),
        }
        // At TP = 1 there is no all-reduce anywhere, so nothing is an identity alias and the
        // launch budget is the graph's own number.
        assert_eq!(p.identity_launches, 0);
        assert_eq!(p.expected_launches(), p.declared_launches);
    }

    /// NEGATIVE CONTROL for the arena policy: a plan whose steps are reordered so that a layer
    /// reads a shared buffer AFTER a later layer overwrote it must be refused. Built by hand
    /// because the real graph cannot produce it — which is the point of checking rather than
    /// arguing.
    #[test]
    fn slot_reuse_is_refused_when_a_value_would_be_read_after_it_is_overwritten() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph at TP=1");
        let mut p = forward_plan(&g, g.bucket_for(128, Phase::Prefill));
        // Move layer 1's first op to the END. It reads `l0.out` and `l0.resid.b`, which by then
        // hold LAYER 63's values -- the exact hazard slot reuse would introduce if the graph ever
        // handed an edge across more than one layer boundary.
        let victim = p
            .steps
            .iter()
            .position(|s| s.op.name == "layer.1.input_layernorm")
            .expect("layer 1's first op is in the plan");
        let moved = p.steps.remove(victim);
        p.steps.push(moved);
        match p.check_slot_reuse() {
            Err(Error::Refused(m)) => {
                assert!(m.contains("slot reuse is unsafe here"), "{m}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// ws-h27: a padded prefill chunk points EVERY attention layer at the rows it feeds, and the
    /// grid shrinks with them.
    ///
    /// `grid[0]` IS the query-CTA count, so this reads the number the kernel will be launched with
    /// rather than a field on the way to it. The whole-model plan has 16 full-attention layers and
    /// the count is asserted, because an attention op that was renamed or fused would otherwise go
    /// on padding the way it did before this existed.
    #[test]
    fn a_padded_prefill_chunk_points_every_attention_layer_at_its_live_rows() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph at TP=1");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let mut p = forward_plan(&g, g.bucket_for(2048, Phase::Prefill));
        let grids = |p: &BlockPlan<'_>| -> Vec<(u32, i64)> {
            p.steps
                .iter()
                .filter(|s| s.op.kernel == KernelOp::AttnPrefillPaged)
                .map(|s| {
                    let Route::Kernel { descs, .. } = &s.route else {
                        panic!("the prefill attention is a kernel route");
                    };
                    let q = match descs[0].arg("q_len").map(|a| a.value.clone()) {
                        Some(saber_kernels::rules::ArgValue::I32 { ival }) => ival,
                        other => panic!("q_len is an i32 kernarg, got {other:?}"),
                    };
                    (descs[0].grid[0], q)
                })
                .collect()
        };
        // As PLANNED: every row of the bucket is a live query row, which is the truth only when
        // the chunk fills it.
        assert_eq!(p.prefill_q_len(), 2048);
        assert_eq!(grids(&p), vec![(32, 2048); 16]);
        // Retargeted at a 234-token prompt: 4 CTAs instead of 32, in all 16 layers.
        assert_eq!(p.retarget_prefill_rows(234, &saber).expect("retarget"), 16);
        assert_eq!(p.prefill_q_len(), 234);
        assert_eq!(grids(&p), vec![(4, 234); 16]);
        // Re-entrant: the next chunk is measured against the BUCKET, not against this one.
        assert_eq!(p.retarget_prefill_rows(2048, &saber).expect("retarget"), 16);
        assert_eq!(grids(&p), vec![(32, 2048); 16]);
        assert_eq!(p.retarget_prefill_rows(1, &saber).expect("retarget"), 16);
        assert_eq!(grids(&p), vec![(1, 1); 16]);
        // NEGATIVE CONTROLS: a chunk is padded UP into its bucket, never truncated; and a DECODE
        // plan has no live/bucket split to make — its `q_len` is the verify width and the `sidx`
        // block width at once.
        assert!(p.retarget_prefill_rows(0, &saber).is_err());
        assert!(p.retarget_prefill_rows(2049, &saber).is_err());
        let mut d = forward_plan(&g, g.bucket_for(8, Phase::Decode));
        assert!(d.retarget_prefill_rows(4, &saber).is_err());
    }

    /// ws-h30: the prefill's recurrent state is the SESSION's slot, `ht` is `h0`, and a plan that
    /// was never pointed at a slot REFUSES rather than reading the null one.
    ///
    /// Three properties in one test because they are one law:
    ///
    /// * `h0` and `ht` name the same state at the same offset — the scan writes IN PLACE, and the
    ///   arena buffer `<op>.ht` that nothing ever copied back is gone;
    /// * the offset is `slot * bytes_per_slot`, re-derived every call, so a second chunk is never
    ///   measured against the first;
    /// * as PLANNED the binding is [`StateAt::PerStep`], which `state_address` refuses by name.
    ///   That is the whole point: the fallback it is refusing to take is the pool's BASE, which is
    ///   slot 0 — so a caller that forgets would read the kernels' null slot for its initial state
    ///   and publish the prompt's final state into it, which is ws-h29's defect with `ht` bound in
    ///   place to make it worse.
    #[test]
    fn a_prefill_chunk_scans_through_its_own_state_slot_and_refuses_until_a_step_says_which() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph at TP=1");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let plan_at = |bucket: Bucket, slots: u32| {
            BlockPlan::with_options(
                &g,
                bucket,
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: slots,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
                PlanOptions::BODY,
            )
            .expect("a plan")
        };
        let scans = |p: &BlockPlan<'_>| -> Vec<(Source, Source)> {
            p.steps
                .iter()
                .filter(|s| s.op.kernel == KernelOp::GdnChunkScan)
                .map(|s| {
                    (
                        s.source("h0").expect("h0 is bound").clone(),
                        s.source("ht").expect("ht is bound").clone(),
                    )
                })
                .collect()
        };
        let mut p = plan_at(
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            9,
        );
        let born = scans(&p);
        assert_eq!(born.len(), 48, "one `chunk_scan` per GDN layer");
        assert_eq!(p.prefill_state_slot(), None);
        // The arena buffer is gone, in every layer.
        assert!(
            p.buffers.iter().all(|b| !b.name.ends_with(".ht")),
            "`<op>.ht` is the state slot now, not an arena buffer"
        );
        // As planned: per-step, and unresolvable. The message has to name the null slot, because
        // that is what silently happened before.
        let mut inputs = BlockInputs::default();
        inputs.states.insert(
            "l0.gdn.recurrent_state".to_owned(),
            DeviceAddr::NULL.offset(1 << 20).expect("a base"),
        );
        for (h0, ht) in &born {
            assert_eq!(h0, ht, "`ht` is bound where `h0` is: in place");
            let Source::State { key, at } = h0 else {
                panic!("the scan's `h0` is a state");
            };
            assert_eq!(*at, StateAt::PerStep);
            if key == "l0.gdn.recurrent_state" {
                let e = state_address(key, *at, &inputs).expect_err("unbound per-step state");
                let text = format!("{e}");
                assert!(text.contains("null slot"), "{text}");
                assert!(text.contains("retarget_prefill_state"), "{text}");
            }
        }
        // Pointed at a slot: both bindings move together, to `slot * bytes_per_slot`.
        let per_slot = g
            .ops
            .iter()
            .flat_map(|o| &o.state)
            .find(|st| st.name == "l0.gdn.recurrent_state")
            .and_then(|st| st.bytes_per_slot)
            .expect("the recurrent state states its slot size");
        assert_eq!(p.retarget_prefill_state(3).expect("retarget"), 48);
        assert_eq!(p.prefill_state_slot(), Some(3));
        for (h0, ht) in scans(&p) {
            assert_eq!(h0, ht);
            let Source::State { key, at } = &h0 else {
                panic!("a state");
            };
            assert_eq!(
                *at,
                StateAt::Slot {
                    slot: 3,
                    offset: 3 * per_slot
                }
            );
            if key == "l0.gdn.recurrent_state" {
                let got = state_address(key, *at, &inputs).expect("a slot resolves");
                assert_eq!(
                    got.as_u64(),
                    (1u64 << 20) + 3 * per_slot,
                    "the slot is an offset from the pool's base"
                );
            }
        }
        // Idempotent and re-entrant: slot 1 is measured from the BASE, not from slot 3.
        assert_eq!(p.retarget_prefill_state(1).expect("retarget"), 48);
        assert!(scans(&p).iter().all(|(h0, _)| matches!(
            h0,
            Source::State {
                at: StateAt::Slot { slot: 1, offset },
                ..
            } if *offset == per_slot
        )));
        // NEGATIVE CONTROLS, each naming what it is refusing.
        let e = p.retarget_prefill_state(0).expect_err("the null slot");
        assert!(format!("{e}").contains("null slot"), "{e}");
        let e = p.retarget_prefill_state(9).expect_err("past the pool");
        assert!(format!("{e}").contains("9 slot(s)"), "{e}");
        // A batched prefill reaches `slot + seqs - 1`, so the bound is tighter there.
        let mut two = plan_at(Bucket::prefill_batch(2, 1024), 9);
        assert!(two.retarget_prefill_state(7).is_ok(), "slots 7 and 8 of 9");
        assert!(
            two.retarget_prefill_state(8).is_err(),
            "the second sequence would be slot 9 of 9"
        );
        // AND BACK: a caller that had to BIND a prefill plan for something that is not a step --
        // the engine sizes every workspace over every plan it can run before a session exists --
        // puts it back, so the next thing to bind one has to say which slot again.
        assert_eq!(p.clear_prefill_state().expect("clear"), 48);
        assert_eq!(p.prefill_state_slot(), None);
        assert!(scans(&p).iter().all(|(h0, ht)| h0 == ht
            && matches!(
                h0,
                Source::State {
                    at: StateAt::PerStep,
                    ..
                }
            )));
        // A DECODE plan finds its slots through `sidx`, one per candidate row.
        let mut d = plan_at(g.bucket_for(8, Phase::Decode), 9);
        let e = d.retarget_prefill_state(1).expect_err("a decode plan");
        assert!(format!("{e}").contains("sidx"), "{e}");
        // And a decode plan's own recurrent state stays at the POOL's base, which is what makes
        // `sidx` the right index into it.
        assert!(d
            .steps
            .iter()
            .filter(|s| matches!(
                s.op.kernel,
                KernelOp::GdnRecurrentUpdate | KernelOp::GdnFusedUpdate
            ))
            .all(|s| matches!(
                s.source("state"),
                Some(Source::State {
                    at: StateAt::Pool,
                    ..
                })
            )));
    }

    /// **What a CHUNKED prefill and a ONE-BUCKET prefill resolve differently, with no card.**
    ///
    /// The serve path feeds a 14,301-token prompt as seven 2,048-row chunks; `saber forward` feeds
    /// it as one 14,301-row bucket; ws-h28 found the two answering different tokens on prompt 8
    /// (`BALL` against `CELL`, 3.75 nats apart in prod's own reference). Before a bisect takes a
    /// card, this asks the cheap half of the question: do the two row counts resolve the same
    /// CODE? A different kernel, a different block shape or a different split-K count is a
    /// different summation order in every row of every layer, and the first divergence would then
    /// be chunk 0, layer 0 — arithmetic, not a law.
    ///
    /// The answer is asserted for the symbols and the block shapes and REPORTED for the scalars,
    /// because a scalar that is the row count is not a difference and a scalar that is not is
    /// exactly what a reader wants named.
    #[test]
    fn a_chunk_and_a_whole_prompt_resolve_the_same_kernels() {
        use saber_kernels::rules::ArgValue;
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let plan_at = |tokens: u32| {
            BlockPlan::with_options(
                &g,
                Bucket::Prefill { tokens, seqs: 1 },
                &layers,
                RuntimeParams {
                    max_ctx: 32768,
                    gdn_slots: 9,
                },
                &mx,
                &r4,
                PlanOptions::BODY,
            )
            .expect("plan")
        };
        let (m_chunk, m_whole) = (2048u32, 14301u32);
        let chunk = plan_at(m_chunk);
        let whole = plan_at(m_whole);
        assert_eq!(
            chunk.steps.len(),
            whole.steps.len(),
            "the two row counts plan different op lists"
        );
        let mut scalars: std::collections::BTreeMap<String, (i64, i64, usize)> =
            std::collections::BTreeMap::new();
        let mut grids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut tall = 0usize;
        for (a, b) in chunk.steps.iter().zip(&whole.steps) {
            assert_eq!(a.name(), b.name());
            let (Route::Kernel { descs: da, .. }, Route::Kernel { descs: db, .. }) =
                (&a.route, &b.route)
            else {
                continue;
            };
            assert_eq!(
                da.len(),
                db.len(),
                "{}: {} dispatch(es) at {m_chunk} rows and {} at {m_whole}",
                a.name(),
                da.len(),
                db.len()
            );
            for (x, y) in da.iter().zip(db) {
                // G1: the 2048-row chunk takes the 512-row A-tiled tile and a wider pass the
                // reference's; the two are ONE summation (bit-identical, gated), so the symbol is
                // compared by its arithmetic class and the tile's own block shape is allowed.
                use saber_kernels::rules::summation_class;
                if x.kernel != y.kernel {
                    assert_eq!(
                        summation_class(&x.kernel),
                        summation_class(&y.kernel),
                        "{}: `{}` and `{}` are different summations",
                        a.name(),
                        x.kernel,
                        y.kernel
                    );
                    tall += 1;
                    continue;
                }
                assert_eq!(
                    x.kernel,
                    y.kernel,
                    "{}: a {m_chunk}-row chunk resolves `{}` and a {m_whole}-row pass `{}` — the \
                     same rows would be summed in two different orders",
                    a.name(),
                    x.kernel,
                    y.kernel
                );
                assert_eq!(x.block, y.block, "{}: two block shapes", a.name());
                if x.grid[1..] != y.grid[1..] {
                    grids.insert(format!("{} {:?} vs {:?}", x.kernel, x.grid, y.grid));
                }
                for (sa, sb) in x.args.iter().zip(&y.args) {
                    assert_eq!(sa.name, sb.name, "{}: kernarg order", a.name());
                    let ((ArgValue::I32 { ival: va }, ArgValue::I32 { ival: vb })
                    | (ArgValue::I64 { ival: va }, ArgValue::I64 { ival: vb })) =
                        (&sa.value, &sb.value)
                    else {
                        continue;
                    };
                    if va == vb {
                        continue;
                    }
                    // A scalar that IS the row count is the row count, not a decision.
                    if *va == i64::from(m_chunk) && *vb == i64::from(m_whole) {
                        continue;
                    }
                    let e = scalars
                        .entry(format!("{} `{}`", x.kernel, sa.name))
                        .or_insert((*va, *vb, 0));
                    e.2 += 1;
                }
            }
        }
        println!(
            "chunked vs one-bucket prefill: {m_chunk} rows against {m_whole}, {} op(s), same \
             kernel symbol and block shape everywhere",
            chunk.steps.len()
        );
        for (what, (va, vb, n)) in &scalars {
            println!(
                "  scalar differs: {what} = {va} at {m_chunk} rows, {vb} at {m_whole} ({n} op(s))"
            );
        }
        for line in &grids {
            println!("  grid y/z differs: {line}");
        }
        if scalars.is_empty() {
            println!("  no scalar kernarg differs except the row count itself");
        }
        println!(
            "  {tall} dispatch(es) take the G1 512-row tile at {m_chunk} rows (same summation)"
        );
        assert!(tall > 0, "the 2048-row chunk no longer reaches the G1 tile");
    }

    #[test]
    fn the_block_is_four_gdn_layers_and_one_attention_layer() {
        let g = graph();
        let p = plan_for(&g, g.bucket_for(5, Phase::Decode));
        let gdn = p
            .steps
            .iter()
            .filter(|s| s.op.kernel == KernelOp::GdnFusedUpdate)
            .count();
        let attn = p
            .steps
            .iter()
            .filter(|s| s.op.kernel == KernelOp::AttnDecodePaged)
            .count();
        assert_eq!((gdn, attn), (4, 1), "layers 0-4 are GDN GDN GDN ATTN GDN");
    }

    #[test]
    fn the_launch_budget_is_the_graphs_own_and_identity_is_subtracted() {
        let g = graph();
        // The attention layer gained THREE launches when the `attn_glue` plugin landed:
        // qk_norm_rope, kv_append and gate_quant all declare one each, where before they
        // declared none and the block supplied their edges from a capture.
        for (m, phase, per_gdn, attn_launches) in [
            (5u32, Phase::Decode, 12u32, 14u32),
            (128, Phase::Prefill, 14, 13),
        ] {
            let p = plan_for(&g, g.bucket_for(m, phase));
            // 4 GDN layers + the attention layer, each counted out of KernelOp::n_launches().
            assert_eq!(p.declared_launches, 4 * per_gdn + attn_launches);
            // Two all-reduces per layer, all replaced by an alias.
            assert_eq!(p.identity_launches, 10);
            assert_eq!(p.expected_launches(), p.declared_launches - 10);
        }
    }

    #[test]
    fn the_block_budget_agrees_with_the_graphs_whole_model_budget() {
        // The per-block sum must be the same rule the graph applies to the whole forward pass:
        // planning every layer and adding the ops outside a layer reproduces
        // `Graph::launch_count_lower_bound` exactly.
        let g = graph();
        let bucket = g.bucket_for(5, Phase::Decode);
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let p = BlockPlan::new(
            &g,
            bucket,
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 7,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
        );
        // The whole model does not plan (lm_head's N is outside the decode band's reach and the
        // head is not a layer), so compare the DECLARED count the same way the graph does.
        let declared: u32 = g
            .ops_for(bucket)
            .iter()
            .filter(|r| matches!(r.op.site, Site::Layer(_)))
            .filter_map(|r| r.op.kernel.n_launches())
            .sum();
        let whole = g.launch_count_lower_bound(bucket).total;
        // Everything outside a layer at this bucket: the embedding's all-reduce, the final norm
        // and lm_head.
        let outside: u32 = g
            .ops_for(bucket)
            .iter()
            .filter(|r| !matches!(r.op.site, Site::Layer(_)))
            .filter_map(|r| r.op.kernel.n_launches())
            .sum();
        assert_eq!(declared + outside, whole);
        assert!(p.is_ok(), "every layer plans: {:?}", p.err());
    }

    /// **Every W4A8 GEMM reads the layout its A-side producer actually writes**, at every
    /// prefill bucket the engine plans — the law the 2026-09-21 garbage above 512 rows broke.
    ///
    /// Two halves, and the bug lived in the gap between them:
    ///
    /// * the PRODUCER half — the kernel the rules emit writes what the graph's edge declares.
    ///   `gdn_norm_quant` has one instantiation and writes row-major at every M (the reference's
    ///   `saber_gdn_norm_quant` is templated on the thread count alone, and `ref_arnq.py`
    ///   never calls `a_tiled_register` for it), while the edge used to be declared `fused_q8`;
    /// * the CONSUMER half — the GEMM's family follows that edge and not `M`. A W4A8 GEMM
    ///   `accepts` BOTH fp8 layouts, so `Graph::check_layouts` is green either way; what says the
    ///   answer is wrong is the kernel name.
    ///
    /// `BlockPlan::new` refuses the producer half by name now, so this test is the consumer half
    /// plus the standing assertion that no bucket refuses.
    #[test]
    fn every_w4a8_gemm_reads_the_layout_its_producer_wrote() {
        let g = graph();
        // 1024 and 2048 are the buckets that were wrong; 256 and 512 are the ones that were
        // right, and a law that only holds above the band edge is not the law.
        for m in [256u32, 512, 1024, 2048] {
            let p = forward_plan(&g, g.bucket_for(m, Phase::Prefill));
            let mut gemms = 0usize;
            let mut a_tiled = 0usize;
            for step in &p.steps {
                if step.op.kernel != KernelOp::GemmW4A8 {
                    continue;
                }
                let Route::Kernel { descs, .. } = &step.route else {
                    continue;
                };
                let a = p
                    .locate(&step.op.inputs[0].edge)
                    .unwrap_or_else(|| panic!("{}: unplanned A edge", step.op.name));
                let kernel = &descs[0].kernel;
                let reads_tiled = kernel.contains("atiled");
                assert_eq!(
                    reads_tiled,
                    a.layout == Layout::ATiledFp8,
                    "M={m} {}: kernel `{kernel}` against a {:?} A edge `{}`",
                    step.op.name,
                    a.layout,
                    step.op.inputs[0].edge,
                );
                gemms += 1;
                a_tiled += usize::from(reads_tiled);
            }
            // 5 W4A8 GEMMs in a GDN layer (in_proj_qkvz, in_proj_ba, out_proj, gate_up,
            // down_proj) and 4 in an attention layer (qkv, o_proj, gate_up, down_proj).
            // `lm_head` is a head-site op and is not in the body plan.
            assert_eq!(gemms, 48 * 5 + 16 * 4, "M={m}");
            // The 48 GDN layers' `out_proj` stays row-major above the band edge exactly when
            // `gdn_norm_quant` writes row-major there -- the reference's law, the committed
            // table's `gdn_norm_tiled = false`. With G2's tiled store it is A-tiled like the rest.
            let gdn_row_major = if LaunchRules::prod_mxfp4()
                .expect("rules")
                .knobs
                .gdn_norm_tiled
            {
                0
            } else {
                48
            };
            let want_tiled = if m >= saber_model::graph::a_tiled_min_m() {
                48 * 5 + 16 * 4 - gdn_row_major
            } else {
                0
            };
            assert_eq!(a_tiled, want_tiled, "M={m}");
        }
    }

    /// The negative control for the producer half (structural rule 8): declare the GDN norm's
    /// edge the OTHER way from what its producer writes and the planner must REFUSE, naming the
    /// op and both layouts. Under the reference's law (`gdn_norm_tiled = false`) the lie is the
    /// pre-P6 `FusedActivationFp8`; under G2's tiled store it is the old row-major-at-every-M.
    #[test]
    fn a_producer_that_does_not_write_what_its_edge_declares_is_refused() {
        let mut g = graph();
        let tiled = LaunchRules::prod_mxfp4()
            .expect("rules")
            .knobs
            .gdn_norm_tiled;
        let lie = if tiled {
            LayoutRule::Fixed(Layout::RowMajorFp8)
        } else {
            LayoutRule::FusedActivationFp8
        };
        let mut patched = 0usize;
        for op in &mut g.ops {
            if op.name.ends_with(".gdn.norm") {
                assert_ne!(op.outputs[0].rule, lie, "the control must change the edge");
                op.outputs[0].rule = lie;
                patched += 1;
            }
        }
        assert_eq!(
            patched, 48,
            "the control has to change the case it controls"
        );
        let bucket = g.bucket_for(2048, Phase::Prefill);
        let err = BlockPlan::new(
            &g,
            bucket,
            &[0, 1, 2, 3, 4],
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 7,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
        )
        .expect_err("a row-major producer under an A-tiled edge must not plan");
        let msg = format!("{err}");
        assert!(msg.contains("gdn.norm"), "{msg}");
        assert!(msg.contains("RowMajorFp8"), "{msg}");
        assert!(msg.contains("ATiledFp8"), "{msg}");

        // The SAME patched graph below the band edge still plans, so the control is about the
        // layout and not about the patch.
        assert!(!plan_for(&g, g.bucket_for(256, Phase::Prefill))
            .buffers
            .is_empty());
    }

    #[test]
    fn buffer_planning_is_a_pure_function_of_the_graph_and_the_bucket() {
        let g = graph();
        let decode = plan_for(&g, g.bucket_for(5, Phase::Decode));
        let prefill = plan_for(&g, g.bucket_for(128, Phase::Prefill));

        // Edge count, derived rather than observed. Per GDN layer: two norms at two edges each
        // (the quantized activation and the residual), `qkvz`, `ba`, the core output, the gated
        // norm's `out_q8`, `out_proj`'s partial, `gate_up`, `act_q8`, `down`'s partial = 12, and
        // the two all-reduces add none because identity ALIASES. The attention layer has the same
        // twelve with `roped` and the attention core in place of `qkvz`/`ba`. The prefill GDN
        // trio writes `conv` and `kkt` where the fused kernel writes neither, so +2 per GDN layer.
        let edges = |p: &BlockPlan<'_>| {
            p.buffers
                .iter()
                .filter(|b| b.kind == BufferKind::EdgeData)
                .count()
        };
        assert_eq!(edges(&decode), 5 * 12);
        assert_eq!(edges(&prefill), 5 * 12 + 4 * 2);

        // Every quantized activation edge carries a scale buffer of exactly 4 bytes per row.
        for p in [&decode, &prefill] {
            let m = u64::from(p.bucket.m());
            for b in &p.buffers {
                if b.kind == BufferKind::EdgeScale {
                    assert_eq!(b.bytes, m * 4, "{}", b.name);
                }
            }
        }

        // Layout per edge: at M=5 and M=128 every fused producer is below `a_tiled_min_m`, so
        // every activation edge is row-major fp8 and no edge is fragment-tiled.
        for p in [&decode, &prefill] {
            assert!(
                p.buffers.iter().all(|b| b.layout != Layout::ATiledFp8),
                "M < 513 keeps every edge row-major"
            );
        }
        assert_eq!(
            decode.buffer("l0.in_q8").map(|b| (b.layout, b.bytes)),
            Some((Layout::RowMajorFp8, 5 * 5120))
        );
        assert_eq!(
            prefill
                .buffer("l0.mlp.gate_up")
                .map(|b| (b.layout, b.bytes)),
            Some((Layout::Bf16, 128 * 17408 * 2))
        );

        // Total bytes: the arena is the sum of the aligned buffers and nothing else.
        for p in [&decode, &prefill] {
            let sum: u64 = p.buffers.iter().map(|b| align_up(b.bytes.max(1))).sum();
            assert_eq!(sum, p.arena_bytes);
        }
        assert!(
            decode.arena_bytes < prefill.arena_bytes,
            "M=128 needs more arena than M=5 ({} vs {})",
            decode.arena_bytes,
            prefill.arena_bytes
        );
    }

    /// **A prefill above the IPC scratch slot is banded, and below it nothing moves** (H28).
    ///
    /// The old rule refused a `tokens x hidden x 2` message over 88,080,384 B by name, so
    /// `saber forward --tp 2` stopped at 8,601 tokens. The reduce is elementwise, so the message
    /// is now cut into row bands that each fit — one dispatch each, all inside the plan, with the
    /// band count a function of the bucket. Two things have to hold for that to be safe, and this
    /// checks both: the bands TILE the edge (no gap, no overlap, nothing reduced twice), and a
    /// message that already fitted resolves to exactly one band.
    #[test]
    fn a_prefill_above_the_ipc_slot_is_banded_and_one_below_it_is_not() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let params = RuntimeParams {
            max_ctx: 32768,
            gdn_slots: 8,
        };
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let hidden = u64::from(g.config.hidden_size);
        let fits = r4.ar.max_bytes / (hidden * 2); // 8,601 at hidden 5120

        let mut declared: Vec<u32> = Vec::new();
        for (rows, want_bands) in [(fits as u32, 1usize), (14301, 2), (30740, 4)] {
            let bucket = g.bucket_for(rows, Phase::Prefill);
            let p = BlockPlan::with_options(
                &g,
                bucket,
                &layers,
                params,
                &mx,
                &r4,
                PlanOptions::BODY
                    .rank(0)
                    .with_collectives(Collectives::Real),
            )
            .expect("a TP=2 prefill plan at a long-context length");
            let mut reduces = 0;
            for step in p
                .steps
                .iter()
                .filter(|s| s.op.kernel == KernelOp::AllReduce)
            {
                reduces += 1;
                let Route::Kernel { descs, .. } = &step.route else {
                    panic!("{}: a reduce with a peer is a kernel", step.name());
                };
                assert_eq!(descs.len(), want_bands, "{} at {rows} rows", step.name());
                assert_eq!(
                    descs[0].n_launches as usize,
                    want_bands,
                    "{}: the head declares the whole op",
                    step.name()
                );
                // The bands tile the edge: contiguous, in order, and exactly its byte count.
                let edge = p
                    .locate(&step.op.inputs[0].edge)
                    .expect("the reduced edge is planned");
                let mut next = edge.offset;
                for i in 0..want_bands {
                    let (in_name, out_name) = saber_kernels::saber::ar_band_names(i);
                    let (Some(Source::Arena { offset, bytes, .. }), Some(out)) =
                        (step.source(&in_name), step.source(&out_name))
                    else {
                        panic!("{}: band {i} binds no arena range", step.name());
                    };
                    assert_eq!(
                        step.source(&in_name),
                        Some(out),
                        "{}: band {i} reduces in place",
                        step.name()
                    );
                    assert_eq!(*offset, next, "{}: band {i} is not contiguous", step.name());
                    assert!(
                        *bytes <= r4.ar.max_bytes,
                        "{}: band {i} is {bytes} B, over the slot",
                        step.name()
                    );
                    assert_eq!(
                        *bytes % (hidden * 2),
                        0,
                        "{}: band {i} splits a row",
                        step.name()
                    );
                    next += *bytes;
                }
                assert_eq!(
                    next,
                    edge.offset + edge.bytes,
                    "{}: the bands must cover the whole message exactly once",
                    step.name()
                );
            }
            // 2 per layer plus the embedding's: 129 at 64 layers.
            assert_eq!(reduces, 2 * g.config.num_hidden_layers + 1);
            // The fusion cannot fire at a prefill row count (the `_nq` epilogue's band is below
            // 12 rows), so `expected` is `declared` and the bands are the only thing that moved.
            assert_eq!(p.fused_launches, 0, "{rows} rows");
            assert_eq!(p.identity_launches, 0, "{rows} rows");
            assert_eq!(p.expected_launches(), p.declared_launches, "{rows} rows");
            declared.push(p.declared_launches);
        }
        // The extra dispatches are EXACTLY the extra bands, and nothing else in the plan moved:
        // 129 reduces, one more dispatch each per extra band.
        let reduces = 2 * g.config.num_hidden_layers + 1;
        assert_eq!(
            declared[1],
            declared[0] + reduces,
            "14k: one extra band each"
        );
        assert_eq!(
            declared[2],
            declared[0] + 3 * reduces,
            "30k: three extra bands each"
        );
    }

    /// A fused plan at a length that has to be BANDED still plans, and fuses nothing.
    ///
    /// `nq_fusion` asks the rules which family the WHOLE message takes, and above the IPC scratch
    /// slot that question has no answer — it was a hard refusal, so `--both-arms` at 14k tokens
    /// would have failed to plan rather than reporting zero folded norms.
    #[test]
    fn the_nq_fusion_declines_a_message_that_has_to_be_banded() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let layers: Vec<u32> = (0..2).collect();
        let params = RuntimeParams {
            max_ctx: 32768,
            gdn_slots: 8,
        };
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let p = BlockPlan::with_options(
            &g,
            g.bucket_for(14301, Phase::Prefill),
            &layers,
            params,
            &mx,
            &r4,
            PlanOptions::BODY
                .rank(0)
                .with_collectives(Collectives::Real)
                .with_fusion(),
        )
        .expect("a fused plan at a banded length plans");
        assert_eq!(p.fused_nq, 0, "no epilogue can ride a banded reduce");
    }

    /// The control the test above needs: the identity plan (no peer) accounts the SAME band count,
    /// so `expected_launches` subtracts what it declared rather than one per reduce.
    #[test]
    fn an_identity_reduce_removes_every_band_it_declared() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let layers: Vec<u32> = (0..2).collect();
        let params = RuntimeParams {
            max_ctx: 32768,
            gdn_slots: 8,
        };
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let bucket = g.bucket_for(30740, Phase::Prefill);
        let p = BlockPlan::with_options(
            &g,
            bucket,
            &layers,
            params,
            &mx,
            &r4,
            PlanOptions::BODY.rank(0),
        )
        .expect("an identity plan");
        let reduces = p
            .steps
            .iter()
            .filter(|s| s.op.kernel == KernelOp::AllReduce)
            .count() as u32;
        assert!(reduces > 0);
        // Four bands each at 30,740 rows, and every one of them is an identity.
        assert_eq!(p.identity_launches, 4 * reduces);
        assert_eq!(
            p.declared_launches - p.expected_launches(),
            p.identity_launches + p.fused_launches
        );
    }

    /// ws-h16's three asks, as one test: a TP=2 plan EXISTS (the embedding takes this rank's
    /// vocabulary window and `logits.gather` is Supplied instead of a refusal), its budget is the
    /// graph's own, and `layer.allreduce` is a real route with `in` and `out` at one address.
    #[test]
    fn a_tp2_plan_reduces_for_real_and_its_budget_is_the_graphs() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let bucket = g.bucket_for(5, Phase::Decode);
        let params = RuntimeParams {
            max_ctx: 8192,
            gdn_slots: 8,
        };
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        for rank in [0u32, 1] {
            let p = BlockPlan::with_options(
                &g,
                bucket,
                &layers,
                params,
                &mx,
                &r4,
                PlanOptions::FORWARD
                    .rank(rank)
                    .with_collectives(Collectives::Real),
            )
            .expect("a TP=2 plan, which refused to exist before ws-h16");
            // Nothing is an identity any more, and the budget is the graph's whole-model count
            // LESS what the `_nq` fusion folds away (ws-h17): the graph's bound counts the
            // reduce and the norm after it as two ops, and one dispatch does both.
            assert_eq!(p.identity_launches, 0);
            assert_eq!(
                p.expected_launches() + p.fused_launches,
                g.launch_count_lower_bound(bucket).total
            );
            // The gather is Supplied, not missing: the caller does it and the plan has its buffer.
            assert_eq!(p.supplied.get("logits_gather").copied(), Some(1));
            // Every reduce is a kernel, and its `in` and `out` are the SAME arena offset.
            let mut reduces = 0;
            for step in p
                .steps
                .iter()
                .filter(|s| s.op.kernel == KernelOp::AllReduce)
            {
                reduces += 1;
                assert!(
                    matches!(step.route, Route::Kernel { .. }),
                    "{}",
                    step.name()
                );
                // A FUSED reduce has no `out` at all -- the sum never reaches memory -- so the
                // in-place law is checked on the arm that still has one.
                if step.source("out").is_some() {
                    assert_eq!(
                        step.source("in"),
                        step.source("out"),
                        "{}: a reduce is in place",
                        step.name()
                    );
                }
            }
            // Two per layer plus the embedding's.
            assert_eq!(reduces, 2 * g.config.num_hidden_layers + 1);
            // The embedding's window is THIS rank's half of the vocabulary.
            let embed = p
                .steps
                .iter()
                .find(|s| s.op.kernel == KernelOp::Embedding)
                .expect("the embedding");
            let Route::Kernel { descs, .. } = &embed.route else {
                panic!("the embedding must be a kernel");
            };
            // The window is in the descriptor's scalar slots; the symbol is the same either way, so
            // what this checks is that the plan resolved AT ALL for rank 1, which it could not.
            assert!(descs[0].kernel.starts_with("saber_embed"));
        }
        // And the default is still Identity, which is the only correct choice for one rank.
        let one =
            BlockPlan::with_options(&g, bucket, &layers, params, &mx, &r4, PlanOptions::BLOCK)
                .expect("the identity plan");
        assert!(one.identity_launches > 0);
    }

    #[test]
    fn host_collectives_keep_fusions_in_bounded_code_objects() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).unwrap();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().unwrap();
        let mut rules = SaberRules::prod_saber().unwrap();
        rules.ar.host_mapped = true;
        rules.ar.quant_min_bytes = rules.ar.max_bytes + 1;
        for checked in [false, true] {
            rules.ar.race_check = checked;
            for rank in [0, 1] {
                for rows in [1, 8] {
                    let p = BlockPlan::with_options(
                        &g,
                        g.bucket_for(rows, Phase::Decode),
                        &layers,
                        RuntimeParams {
                            max_ctx: 8192,
                            gdn_slots: 8,
                        },
                        &mx,
                        &rules,
                        PlanOptions::FORWARD
                            .rank(rank)
                            .with_collectives(Collectives::Real)
                            .with_decode_fusion()
                            .with_gemm_wht(),
                    )
                    .unwrap();
                    assert!(
                        p.fused_gemm_ar > 0,
                        "bounded GEMM/reduce fusion must survive"
                    );
                    assert!(p.fused_pairs > 0);
                    let mut collectives = 0;
                    for step in &p.steps {
                        if let Route::Kernel { descs, .. } = &step.route {
                            for d in descs {
                                if d.workspaces.iter().any(|w| {
                                    w.kind == saber_kernels::rules::WorkspaceKind::IpcPeerVisible
                                }) {
                                    collectives += 1;
                                    assert!(
                                        matches!(
                                            d.plugin.as_str(),
                                            "saber_ar_host_exact"
                                                | "saber_ar_host_exact_chk"
                                                | "mxfp4_gemm_decode_host_arnq"
                                                | "mxfp4_gemm_decode_host_arnq_chk"
                                        ),
                                        "{}: {}",
                                        step.name(),
                                        d.plugin
                                    );
                                    assert_eq!(d.kernel.ends_with("_chk"), checked);
                                }
                            }
                        }
                    }
                    assert_eq!(collectives, 2 * g.config.num_hidden_layers + 1);
                }
            }
        }
    }

    #[test]
    fn hybrid_collectives_preserve_decode_fusion_and_external_prefill() {
        let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).unwrap();
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().unwrap();
        let mut rules = SaberRules::prod_saber().unwrap();
        rules.ar.host_mapped = true;
        rules.ar.external_rccl = true;
        rules.ar.external_rccl_min_bytes = 65 * 5120 * 2;
        rules.ar.quant_min_bytes = rules.ar.max_bytes + 1;
        for rank in [0, 1] {
            for (rows, phase) in [(8, Phase::Decode), (96, Phase::Prefill)] {
                let plan = BlockPlan::with_options(
                    &graph,
                    graph.bucket_for(rows, phase),
                    &layers,
                    RuntimeParams {
                        max_ctx: 8192,
                        gdn_slots: 8,
                    },
                    &mx,
                    &rules,
                    PlanOptions::BODY
                        .rank(rank)
                        .with_collectives(Collectives::Real)
                        .with_decode_fusion(),
                )
                .unwrap();
                if phase == Phase::Decode {
                    assert!(
                        plan.fused_gemm_ar > 0,
                        "hybrid decode must retain native fusion"
                    );
                } else {
                    assert_eq!((plan.fused_nq, plan.fused_gemm_ar), (0, 0));
                }
                let mut collectives = 0;
                for step in &plan.steps {
                    if let Route::Kernel { descs, .. } = &step.route {
                        for desc in descs {
                            let external = desc.plugin == saber_kernels::saber::RCCL_PLUGIN;
                            let host = desc.workspaces.iter().any(|w| {
                                w.kind == saber_kernels::rules::WorkspaceKind::IpcPeerVisible
                            });
                            if external || host {
                                collectives += 1;
                                assert_eq!(external, phase == Phase::Prefill);
                                assert_eq!(host, phase == Phase::Decode);
                            }
                        }
                    }
                }
                assert_eq!(collectives, 2 * graph.config.num_hidden_layers + 1);
            }
        }
    }

    #[test]
    fn rccl_retains_local_fusions_without_native_peer_reductions() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let mut rules = SaberRules::prod_saber().expect("rules");
        rules.ar.external_rccl = true;
        for rank in [0, 1] {
            for rows in [1, 8] {
                let p = BlockPlan::with_options(
                    &g,
                    g.bucket_for(rows, Phase::Decode),
                    &layers,
                    RuntimeParams {
                        max_ctx: 8192,
                        gdn_slots: 8,
                    },
                    &mx,
                    &rules,
                    PlanOptions::FORWARD
                        .rank(rank)
                        .with_collectives(Collectives::Real)
                        .with_decode_fusion()
                        .with_gemm_wht(),
                )
                .expect("RCCL plan with local fusions");
                assert_eq!((p.fused_nq, p.fused_gemm_ar, p.fused_gemm_wht), (0, 0, 0));
                assert!(p.fused_pairs > 0, "local GEMM pairs must remain enabled");
                assert!(
                    p.fused_rope_appends > 0,
                    "local rope/append must remain enabled"
                );
                let mut reduces = 0;
                for step in &p.steps {
                    if step.op.kernel == KernelOp::AllReduce {
                        reduces += 1;
                        let Route::Kernel { descs, .. } = &step.route else {
                            panic!("collective must remain explicit")
                        };
                        assert_eq!(descs.len(), 1);
                        assert_eq!(descs[0].plugin, saber_kernels::saber::RCCL_PLUGIN);
                        assert!(descs[0].workspaces.is_empty());
                    }
                }
                assert_eq!(reduces, 2 * g.config.num_hidden_layers + 1);
            }
        }
    }

    /// ws-h17: the `_nq` epilogue, taken by MESSAGE SIZE and only there.
    ///
    /// At b1s5 a row block is 5 x 5120 x bf16 = 50 KiB, under the rules' 128 KiB threshold, so
    /// every reduce is the exact family and every one of them folds the `add_rms_quant` after it
    /// into its epilogue. At b4s5 the same message is 200 KiB, the rules answer the rotated
    /// kernel, and there is no `_nq` arm of it — so the plan is the two-op one, unchanged.
    #[test]
    fn the_nq_epilogue_is_taken_below_the_rules_threshold_and_nowhere_else() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let params = RuntimeParams {
            max_ctx: 8192,
            gdn_slots: 8,
        };
        let plan = |rows: u32| {
            BlockPlan::with_options(
                &g,
                g.bucket_for(rows, Phase::Decode),
                &layers,
                params,
                &mx,
                &r4,
                PlanOptions::FORWARD
                    .rank(0)
                    .with_collectives(Collectives::Real)
                    // Opted into BY NAME: the fusion is off by default since ws-h19 measured it is
                    // not bit-identical to the two-op sequence once the attention glue is bound.
                    // This test is about the route, so it asks for it.
                    .with_fusion(),
            )
            .expect("a TP=2 plan")
        };

        // ---- below the threshold: every reduce carries the epilogue ------------------------
        let small = plan(5);
        let reduces = 2 * g.config.num_hidden_layers + 1;
        assert_eq!(small.fused_nq, reduces, "every reduce folds its norm");
        assert_eq!(small.fused_launches, small.fused_nq);
        for step in &small.steps {
            let fused = matches!(step.route, Route::Fused { .. });
            assert_eq!(
                fused,
                step.op.kernel == KernelOp::AddRmsQuant,
                "{}: only the norms are fused",
                step.name()
            );
        }
        // The reduce's descriptor is the epilogue, one workgroup per row, and it binds the
        // norm's weight, residual and BOTH of its outputs.
        let ar = small
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.gdn.all_reduce")
            .expect("layer 0's GDN reduce");
        let Route::Kernel { descs, .. } = &ar.route else {
            panic!("the reduce must be a kernel");
        };
        assert_eq!(
            descs[0].kernel,
            "saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16"
        );
        assert_eq!(descs[0].grid, [5, 1, 1]);
        assert_eq!(descs[0].n_launches, 1);
        for slot in ["in", "residual", "norm_w", "q_out", "scale_out", "res_out"] {
            assert!(ar.source(slot).is_some(), "the fusion binds `{slot}`");
        }
        // `q_out` is the norm's own quantized edge, at the norm's own arena offset: the fusion
        // moved the dispatch, not the data.
        let norm = small
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.post_attention_layernorm")
            .expect("the norm after it");
        assert_eq!(ar.source("q_out"), norm.source("q"));
        assert_eq!(ar.source("res_out"), norm.source("res_out"));
        assert_eq!(ar.source("residual"), norm.source("res"));

        // ---- above it: the rotated reduce, and the pair stays two ops ----------------------
        let big = plan(20);
        assert_eq!(big.fused_nq, 0, "200 KiB is above `quant_min_bytes`");
        assert_eq!(
            big.expected_launches(),
            g.launch_count_lower_bound(g.bucket_for(20, Phase::Decode))
                .total
        );
        let ar = big
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.gdn.all_reduce")
            .expect("layer 0's GDN reduce");
        let Route::Kernel { descs, .. } = &ar.route else {
            panic!("the reduce must be a kernel");
        };
        assert_eq!(descs[0].kernel, "saber_ar_oneshot_2rank_wht6_bf16");
        // And the budget difference between the two is exactly the norms folded away.
        assert_eq!(
            small.declared_launches - small.expected_launches(),
            reduces,
            "one dispatch per fused pair"
        );

        // ---- the control: the same bucket with the fusion off ------------------------------
        //
        // This is the plan the gate compares against, so it has to be the graph's own sequence:
        // nothing fused, the budget back to the graph's bound, and the ARENA identical — a fusion
        // moves a dispatch, not a buffer, and a byte comparison between the two runs depends on
        // that.
        let off = BlockPlan::with_options(
            &g,
            g.bucket_for(5, Phase::Decode),
            &layers,
            params,
            &mx,
            &r4,
            PlanOptions::FORWARD
                .rank(0)
                .with_collectives(Collectives::Real)
                .without_fusion(),
        )
        .expect("the unfused plan");
        assert_eq!(off.fused_nq, 0);
        assert_eq!(
            off.expected_launches(),
            g.launch_count_lower_bound(g.bucket_for(5, Phase::Decode))
                .total
        );
        assert_eq!(off.arena_bytes, small.arena_bytes);
        assert_eq!(off.buffers.len(), small.buffers.len());
        for (a, b) in off.buffers.iter().zip(&small.buffers) {
            assert_eq!(
                (&a.name, a.offset, a.bytes),
                (&b.name, b.offset, b.bytes),
                "the fusion moved a buffer"
            );
        }
        // The norm that was folded is an ordinary kernel again, and its bindings are the ones the
        // fused descriptor borrowed.
        let norm_off = off
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.post_attention_layernorm")
            .expect("the norm");
        assert!(matches!(norm_off.route, Route::Kernel { .. }));
        assert_eq!(norm_off.source("q"), norm.source("q"));
        assert_eq!(norm_off.source("res"), norm.source("res"));
    }

    #[test]
    fn the_identity_all_reduce_aliases_its_input_edge_rather_than_allocating() {
        let g = graph();
        let p = plan_for(&g, g.bucket_for(5, Phase::Decode));
        // `l0.out` is the all-reduce's output edge; it must resolve to the same bytes as the
        // GDN partial that feeds it, and cost nothing.
        assert!(
            p.buffer("l0.out").is_none(),
            "no buffer is allocated for it"
        );
        let next = p
            .steps
            .iter()
            .find(|s| s.name() == "layer.1.input_layernorm")
            .expect("layer 1 opens the next block");
        let partial = p.buffer("l0.mlp.partial").expect("the down_proj output");
        assert_eq!(
            next.source("y"),
            Some(&Source::Arena {
                buffer: "l0.mlp.partial".into(),
                offset: partial.offset,
                bytes: partial.bytes,
            })
        );
    }

    #[test]
    fn layer_zero_reads_an_explicit_zero_residual() {
        let g = graph();
        let p = plan_for(&g, g.bucket_for(5, Phase::Decode));
        let first = &p.steps[0];
        assert_eq!(first.name(), "layer.0.input_layernorm");
        let Some(Source::Arena { buffer, .. }) = first.source("res") else {
            panic!("layer 0's residual is not bound");
        };
        assert_eq!(buffer, "layer.0.input_layernorm.zero_resid");
    }

    #[test]
    fn every_op_binds_every_buffer_its_descriptor_names() {
        // The check `bindings` runs, restated as a test so that a future kernel with a new
        // argument fails here rather than inside `pack_kernarg` on a device.
        let g = graph();
        for (m, phase) in [(5u32, Phase::Decode), (128, Phase::Prefill)] {
            let p = plan_for(&g, g.bucket_for(m, phase));
            for step in &p.steps {
                let Route::Kernel { descs, .. } = &step.route else {
                    continue;
                };
                for desc in descs {
                    for slot in &desc.args {
                        if let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value {
                            assert!(
                                step.source(buffer).is_some()
                                    || desc.workspaces.iter().any(|w| &w.name == buffer),
                                "{}: {} wants `{buffer}`",
                                step.name(),
                                desc.kernel
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_gdn_norm_reads_z_at_the_merged_in_proj_offset() {
        let g = graph();
        let p = plan_for(&g, g.bucket_for(5, Phase::Decode));
        let norm = p
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.gdn.norm")
            .expect("the gated norm");
        let qkvz = p.buffer("l0.gdn.qkvz").expect("the merged in_proj output");
        let Some(Source::Arena { offset, .. }) = norm.source("z") else {
            panic!("z is not bound");
        };
        // z starts (8192 - 3072) elements into the merged output.
        assert_eq!(*offset, qkvz.offset + 5120 * 2);
    }

    /// There is nothing left to supply. This test used to assert that `qk_norm_rope` and
    /// `attn_gate_quant` were named as supplied and launched nothing; the `attn_glue` plugin
    /// gave both a code object, and the paged K/V write became an op of its own, so a decoder
    /// block now has a kernel for every op it contains. Asserting the EMPTY set is the same
    /// statement made the other way round, and it fails the day something new is supplied.
    #[test]
    fn nothing_in_a_decoder_block_is_supplied_any_more() {
        let g = graph();
        for (m, phase) in [(5u32, Phase::Decode), (128, Phase::Prefill)] {
            let p = plan_for(&g, g.bucket_for(m, phase));
            assert!(
                p.supplied.is_empty(),
                "{phase:?}: still supplied: {:?}",
                p.supplied
            );
            assert!(p
                .steps
                .iter()
                .all(|s| !matches!(s.route, Route::Supplied { .. })));
            // And the three glue ops are all there, as kernels.
            for want in [
                KernelOp::QkNormRope,
                KernelOp::AttnGateQuant,
                KernelOp::KvAppend,
            ] {
                let step = p
                    .steps
                    .iter()
                    .find(|s| s.op.kernel == want)
                    .unwrap_or_else(|| panic!("{want} is not in the block"));
                match &step.route {
                    Route::Kernel { descs, .. } => {
                        assert_eq!(descs.len(), 1, "{want} is one launch");
                        assert_eq!(descs[0].plugin, "attn_glue");
                    }
                    other => panic!("{want} routed to {other:?}"),
                }
            }
        }
    }

    /// A combined body+head plan at a prefill chunk is refused, not planned.
    ///
    /// `PlanOptions::FORWARD` is documented as affordable only at a small row count, and this is
    /// the reason: at `p2048` its head would compute 2048 x 248320 logits and read one row of
    /// them. The negative control matters as much as the refusal -- the same options at a DECODE
    /// bucket must still plan, because there every row is a sampled position and the combined
    /// plan is correct.
    #[test]
    fn a_combined_plan_will_not_put_the_head_on_a_whole_prefill_chunk() {
        let g = graph();
        let params = RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 7,
        };
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();

        let err = BlockPlan::with_options(
            &g,
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            &layers,
            params,
            &mxfp4,
            &saber,
            PlanOptions::FORWARD,
        );
        match err {
            Err(Error::Refused(m)) => {
                assert!(m.contains("SAMPLED"), "{m}");
                assert!(m.contains("2048"), "it names the row count it refused: {m}");
                assert!(
                    m.contains("PlanOptions::BODY") && m.contains("PlanOptions::HEAD"),
                    "it names the way out: {m}"
                );
            }
            other => panic!("a 2048-row head was not refused: {other:?}"),
        }

        // NEGATIVE CONTROL 1: the same bucket without the head site plans fine. The body is not
        // what the guard is about.
        BlockPlan::with_options(
            &g,
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            &layers,
            params,
            &mxfp4,
            &saber,
            PlanOptions::BODY,
        )
        .expect("the body at 2048 rows is exactly what a prefill chunk runs");

        // NEGATIVE CONTROL 2: a combined plan at a DECODE bucket still plans -- 756 tests and
        // every `step-capture` run depend on it.
        BlockPlan::with_options(
            &g,
            g.bucket_for(5, Phase::Decode),
            &layers,
            params,
            &mxfp4,
            &saber,
            PlanOptions::FORWARD,
        )
        .expect("every decode row is sampled, so one plan is correct there");

        // NEGATIVE CONTROL 3: a one-token prefill chunk has sampled_rows == m, so it is allowed.
        BlockPlan::with_options(
            &g,
            Bucket::Prefill { tokens: 1, seqs: 1 },
            &layers,
            params,
            &mxfp4,
            &saber,
            PlanOptions::FORWARD,
        )
        .expect("one token, one sampled row");
    }

    /// The head's cost does not scale with the chunk, and the budget says so.
    ///
    /// The body's launch count grows with neither (it is per layer), but the head's GEMM at a
    /// prefill bucket used to be planned at the chunk's M. Planning it at the sampled rows makes
    /// the head plan IDENTICAL for every prefill length, which is the property worth asserting:
    /// a future caller that re-introduces the chunk's M would break this before it reached a GPU.
    #[test]
    fn the_head_plan_is_the_same_at_every_prefill_length() {
        let g = graph();
        let params = RuntimeParams {
            max_ctx: 8192,
            gdn_slots: 7,
        };
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let head_for = |tokens: u32| {
            BlockPlan::with_options(
                &g,
                g.head_bucket(Bucket::Prefill { tokens, seqs: 1 }),
                &[],
                params,
                &mxfp4,
                &saber,
                PlanOptions::HEAD,
            )
            .expect("the head plans at one row")
        };
        let small = head_for(64);
        let big = head_for(8192);
        assert_eq!(
            small.expected_launches(),
            big.expected_launches(),
            "the head's launch count is a function of the SAMPLED rows, not the chunk"
        );
        assert_eq!(
            small.arena_bytes, big.arena_bytes,
            "and so is its arena: a p8192 head that allocated for 8192 rows would ask for              ~970 MiB of logits"
        );
        // One row of logits per rank, not 8192. 124160 vocab x 2 bytes = 242.5 KiB is the scale;
        // the chunk's would be three orders of magnitude more.
        assert!(
            big.arena_bytes < 64 << 20,
            "the head's arena is {} bytes",
            big.arena_bytes
        );
    }

    #[test]
    fn a_layer_outside_the_graph_is_refused_rather_than_planned_empty() {
        let g = graph();
        let err = BlockPlan::new(
            &g,
            g.bucket_for(5, Phase::Decode),
            &[900],
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 7,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
        );
        assert!(err.is_err());
    }

    #[test]
    fn every_weight_the_block_binds_is_a_record_the_model_declares_device_final() {
        // The plan's weight keys and the model's device-final record names are the same set, which
        // is what lets `saber layer` bind straight out of the upload arena with no mapping table.
        let g = graph();
        let records: BTreeMap<String, u32> = g
            .device_final_records()
            .expect("plan")
            .into_iter()
            .map(|r| (r.name, r.layout_id))
            .collect();
        for (m, phase) in [(5u32, Phase::Decode), (128, Phase::Prefill)] {
            let p = plan_for(&g, g.bucket_for(m, phase));
            assert!(!p.weight_keys().is_empty());
            for key in p.weight_keys() {
                assert!(
                    records.contains_key(&key),
                    "`{key}` is bound by the block and is not in the device-final plan"
                );
            }
            // And the expectation on the binding is the model's id, not a guess.
            for step in &p.steps {
                for (_, src) in &step.bindings {
                    if let Source::Weight { key, layout } = src {
                        assert_eq!(records.get(key), Some(layout), "{key}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_matrix_records_expect_the_merged_layout_exactly_where_the_merge_is() {
        let g = graph();
        let p = plan_for(&g, g.bucket_for(5, Phase::Decode));
        let expect = |op: &str, buf: &str| -> u32 {
            let step = p.steps.iter().find(|s| s.name() == op).expect(op);
            match step.source(buf) {
                Some(Source::Weight { layout, .. }) => *layout,
                other => panic!("{op}/{buf} is {other:?}"),
            }
        };
        // 4 = mxfp4_wperm_merged, 1 = mxfp4_wperm, 2 = block scale, 3 = row ref, 5 = the f32 cast,
        // 0 = raw (docs/abi/PACK.md § Layout ids).
        assert_eq!(expect("layer.0.gdn.in_proj_qkvz", "w"), 4, "qkv + z");
        assert_eq!(expect("layer.0.mlp.gate_up", "w"), 4, "gate + up");
        assert_eq!(expect("layer.3.attn.qkv", "w"), 4, "q + k + v");
        assert_eq!(expect("layer.0.mlp.down_proj", "w"), 1, "one projection");
        assert_eq!(expect("layer.0.gdn.out_proj", "w"), 1);
        assert_eq!(expect("layer.0.mlp.down_proj", "ws"), 2);
        assert_eq!(expect("layer.0.mlp.down_proj", "wref"), 3);
        assert_eq!(expect("layer.0.gdn.fused_update", "a_log"), 5);
        assert_eq!(expect("layer.0.gdn.fused_update", "dt_bias"), 5);
        assert_eq!(
            expect("layer.0.gdn.fused_update", "wgt"),
            0,
            "conv1d filter"
        );
        assert_eq!(expect("layer.0.input_layernorm", "w"), 0);
    }

    /// NEGATIVE CONTROL for the layout check (CLAUDE.md rule 8).
    #[test]
    fn a_record_with_the_wrong_layout_for_its_role_is_refused_at_bind() {
        let g = graph();
        let p = plan_for(&g, g.bucket_for(5, Phase::Decode));
        let step = p
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.mlp.gate_up")
            .expect("the merged MLP GEMM");
        let src = step.source("w").expect("its weight").clone();
        let Source::Weight { key, layout } = &src else {
            panic!("not a weight");
        };
        assert_eq!(*layout, 4);

        let mut inputs = BlockInputs::default();
        // Raw (0) where the op needs mxfp4_wperm_merged (4): exactly what binding a v0 pack to a
        // device-final plan would do, and the bytes would be the right LENGTH either way.
        inputs
            .weights
            .insert(key.clone(), BoundWeight::new(DeviceAddr::NULL, 0));
        let err = check_weight_layout(&src, &inputs).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains(key.as_str()), "{msg}");
        assert!(msg.contains("layout id 0"), "{msg}");
        assert!(msg.contains("layout id 4"), "{msg}");

        inputs
            .weights
            .insert(key.clone(), BoundWeight::new(DeviceAddr::NULL, 4));
        check_weight_layout(&src, &inputs).expect("the right layout binds");

        // And an unbound record is refused too, by name.
        let empty = BlockInputs::default();
        assert!(check_weight_layout(&src, &empty)
            .unwrap_err()
            .to_string()
            .contains(key.as_str()));
    }

    /// C1: the f16 STORED GDN state is a ROUTE change and a SIZE change, and nothing else.
    ///
    /// Planned over the whole body at TP=2 / width 8, at the three shapes the three state kernels
    /// serve — b1s8 (the fused update, 24 items), b8s5 (the conv + recurrent pair) and a
    /// 2048-row prefill chunk (the scan) — the f16 graph's plan is the f32 graph's plan step for
    /// step: the same ops, the same launch budget, the same grids, blocks and kernarg bytes. The
    /// ONLY differences are the three GDN state kernels' symbols (the rules' `state_f16` rows,
    /// `_f16s_` in the name, in their own code objects) and the prefill slot offset, which is
    /// `slot * bytes_per_slot` at HALF the bytes. The control is the f32 graph itself: it resolves
    /// no `_f16s_` symbol anywhere.
    #[test]
    fn the_f16_gdn_state_changes_three_symbols_and_the_slot_bytes_and_nothing_else() {
        let g32 = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8).expect("graph");
        let g16 = g32.clone().with_gdn_state_f16().expect("f16 state");
        assert!(g16.gdn_state_f16() && !g32.gdn_state_f16());
        assert_eq!(
            g16.clone().with_gdn_state_f16().expect("idempotent").ops,
            g16.ops,
            "applying it twice is applying it once"
        );
        let per_slot = |g: &Graph, name: &str| -> u64 {
            g.ops
                .iter()
                .flat_map(|o| &o.state)
                .find(|st| st.name == name)
                .and_then(|st| st.bytes_per_slot)
                .expect("a slotted state")
        };
        // 24 v heads x 128 x 128 at TP=2: fp32 1.5 MiB, f16 0.75 MiB; the conv cache is untouched.
        assert_eq!(per_slot(&g32, "l0.gdn.recurrent_state"), 1_572_864);
        assert_eq!(per_slot(&g16, "l0.gdn.recurrent_state"), 786_432);
        assert_eq!(
            per_slot(&g32, "l0.gdn.conv_state"),
            per_slot(&g16, "l0.gdn.conv_state")
        );
        fn plan(g: &Graph, bucket: Bucket) -> BlockPlan<'_> {
            let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
            BlockPlan::with_options(
                g,
                bucket,
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 65,
                },
                &LaunchRules::prod_mxfp4().expect("rules"),
                &SaberRules::prod_saber().expect("saber rules"),
                PlanOptions::BODY,
            )
            .expect("a plan")
        }
        let narrow = |k: &str| -> String {
            for (from, to) in [
                ("saber_gdn_fused_update_", "saber_gdn_fused_update_f16s_"),
                (
                    "saber_gdn_recurrent_update_",
                    "saber_gdn_recurrent_update_f16s_",
                ),
                ("saber_gdn_chunk_scan_", "saber_gdn_chunk_scan_f16s_"),
            ] {
                if let Some(rest) = k.strip_prefix(from) {
                    return format!("{to}{rest}");
                }
            }
            k.to_owned()
        };
        for (bucket, want) in [
            (
                Bucket::Decode {
                    batch: 1,
                    verify_width: 8,
                },
                "saber_gdn_fused_update_f16s_w4k128v128_ab1",
            ),
            (
                Bucket::Decode {
                    batch: 8,
                    verify_width: 5,
                },
                "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1",
            ),
            (
                Bucket::Prefill {
                    tokens: 2048,
                    seqs: 1,
                },
                "saber_gdn_chunk_scan_f16s_k128_v128_c64",
            ),
        ] {
            let (a, b) = (plan(&g32, bucket), plan(&g16, bucket));
            assert_eq!(a.expected_launches(), b.expected_launches(), "{bucket:?}");
            assert_eq!(a.steps.len(), b.steps.len());
            let mut narrowed = 0usize;
            for (sa, sb) in a.steps.iter().zip(&b.steps) {
                assert_eq!(sa.op.name, sb.op.name);
                match (&sa.route, &sb.route) {
                    (Route::Kernel { descs: da, .. }, Route::Kernel { descs: db, .. }) => {
                        assert_eq!(da.len(), db.len(), "{}", sa.op.name);
                        for (x, y) in da.iter().zip(db) {
                            assert!(
                                !x.kernel.contains("_f16s_"),
                                "the f32 control resolves no narrow symbol: {}",
                                x.kernel
                            );
                            assert_eq!(y.kernel, narrow(&x.kernel), "{}", sa.op.name);
                            narrowed += usize::from(y.kernel != x.kernel);
                            assert_eq!(
                                (x.grid, x.block, x.kernarg_bytes, x.n_launches),
                                (y.grid, y.block, y.kernarg_bytes, y.n_launches),
                                "{}",
                                sa.op.name
                            );
                        }
                    }
                    (ra, rb) => assert_eq!(
                        std::mem::discriminant(ra),
                        std::mem::discriminant(rb),
                        "{}",
                        sa.op.name
                    ),
                }
            }
            // 48 GDN layers, one state kernel each.
            assert_eq!(narrowed, 48, "{bucket:?}");
            assert!(
                b.steps.iter().any(|s| match &s.route {
                    Route::Kernel { descs, .. } => descs.iter().any(|d| d.kernel == want),
                    _ => false,
                }),
                "{bucket:?} resolves {want}"
            );
        }
        // The prefill publish lands at `slot * bytes_per_slot`, at the NARROW width.
        let mut p = plan(
            &g16,
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
        );
        assert_eq!(p.retarget_prefill_state(3).expect("retarget"), 48);
        for s in p
            .steps
            .iter()
            .filter(|s| s.op.kernel == KernelOp::GdnChunkScan)
        {
            let Some(Source::State { at, .. }) = s.source("h0") else {
                panic!("h0 is a state");
            };
            assert_eq!(
                *at,
                StateAt::Slot {
                    slot: 3,
                    offset: 3 * 786_432
                }
            );
        }
    }
}
