//! The DFlash2 proposal as **two phases**, built through the engine's own launcher.
//!
//! [`crate::draft_tier`] bills the drafter; [`crate::draft_inputs`] derives its index tables;
//! [`crate::aux_tap`] gets its one non-weight input out of the target's step. This module is the
//! launches: every dispatch of a proposal, in forward order, bound to addresses the ENGINE owns.
//!
//! # Why two phases, and which is which
//!
//! Prod's own arrangement is the answer, and Saber cannot do better than it for a reason that is
//! structural rather than a matter of effort. `precompute_and_store_context_kv` runs **eagerly**,
//! outside `_generate_draft`'s captured graph, because the context shape varies per step; only the
//! drafter's own forward is replayed. The same is forced here: the target's acceptance is decided
//! on the HOST after the argmax read-back, and every context-store table is a function of it — how
//! many rows are valid, which positions they carry, which slot each one lands in. A phase whose
//! kernargs are a function of a host decision cannot be a replayed graph, whose kernargs are baked
//! at capture.
//!
//! * [`DraftPhase::ContextStore`] — EAGER. Twelve launches, driven by [`crate::draft_inputs`]:
//!   the aux quant, the aux projection, the hidden norm, the fused KV projection, the layout copy,
//!   the grouped K norm, the batched rope and one cache insert per drafter layer.
//! * [`DraftPhase::Backbone`] — CAPTURED. The drafter's forward at a **shape fact**: `block` query
//!   rows, whatever the step accepted, because prod pads to the block for the captured graph's
//!   sake and the index tables carry where each row goes. Embedding, five layers, the head's seam,
//!   the int2 head, its two selections, the exact rerank, the selector's top-16, the anchor
//!   gather, the projection, `score_edges` and the walk.
//!
//! The split is also what makes `host_round_trips() == 0` a checkable claim about the captured
//! half rather than about the whole proposal: [`DraftPlan::host_round_trips`] is 0 by construction
//! — nothing in [`DraftPlan::captured`] reads a device buffer back — and the test says so.
//!
//! # The candidate ids are ONE region, and the day the kernel's dtype changed this plan did not
//!
//! `dflash_topk_bf16_k16` wrote the selector's candidate ids as **i32** when ws-s3 wrote this
//! module, and both `dflash_selector_score_r256_k16` and `dflash_selector_walk_greedy` read them
//! as `const long *`; the captured phase therefore carried one `head::WidenI64` of 7 x 16
//! elements to bridge them, because a captured graph cannot round-trip through the host without
//! making `host_round_trips() == 0` a lie. **ws-f28 changed the kernel to write `long *` (commit
//! `2eb58a1`) and this plan was not changed with it**, so for three rounds the selection wrote
//! 896 bytes into a 448-byte region and the widen then read those i64s back as i32 — the scorer
//! and the walk were handed `[id0, 0, id1, 0, …]`, which is a wrong candidate SET reached through
//! a context store and a body that are bit-exact. It is the defect ws-s7 found, and nothing in
//! the tree could see it: the descriptor law below compares kernels, kernargs and grids, and a
//! buffer's SIZE and its element WIDTH are in neither.
//!
//! So there is no widen and no `sel.idx32`: the selection writes `sel.cand64` and both its
//! consumers read that same region, which is what `saber-cli`'s standalone chain has done since
//! F28. At TP > 1 the candidate SEAM reads it too — the ids the host makes global and merges are
//! i64 on the way out as well as back in.
//! [`the_candidate_ids_are_one_i64_region_from_the_selection_to_the_walk`] is the law and
//! [`every_arena_region_is_bound_at_one_element_width`] is its general form; both fail on the old
//! layout.
//!
//! # What this module does NOT claim
//!
//! Not that the proposal is right. The chain from `aux.cat` to a draft block is ws-f's, and as of
//! F27 its unseeded backbone half is red at or before layer 0's post-rope `q`. What is claimed
//! here is PLUMBING: the same launches, in the same order, at the same shapes, over an arena the
//! engine allocated once and workspaces the engine prepared before capture — instead of
//! `draft_proposal.rs`'s `hipMalloc` per buffer and 18 host round trips before its first launch.

use std::collections::BTreeMap;

use saber_exec::ArgBindings;
use saber_hip::DeviceAddr;
use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape};
use saber_kernels::saber::{self, SaberRules};
use saber_model::dflash_graph::DflashConfig;

use crate::forward::ForwardError;

/// Which half of a proposal a launch belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftPhase {
    /// Eager: its kernargs are a function of the host's acceptance verdict.
    ContextStore,
    /// Captured: its shapes are facts of the drafter's block.
    Backbone,
    /// Captured, and only at TP > 1: the launches that come AFTER the candidate seam.
    ///
    /// The drafter's head scores this rank's vocabulary SHARD, so `selector.candidate_topk`
    /// returns the best 16 of HALF a vocabulary and the row's own best 16 is the merge of the
    /// two — prod's `get_top_k_tokens` all-gathers `2k` values and ids and takes the top-k of the
    /// concatenation, which is `saber_serve::sampler::merge_shards`'s law one level up
    /// (ws-v3: "a top-K over a shard is not the row's top-K, and two of them are"). Saber gathers
    /// that seam the way it gathers `logits.gather` — on the host, over the rendezvous socket,
    /// 896 bytes a rank — so the backbone is TWO graphs at two ranks with one host seam between
    /// them, and ONE graph with no seam at TP=1, where this phase is empty.
    BackboneTail,
}

/// One launch of a proposal: a name, the descriptor the rules resolved, and every buffer bound.
#[derive(Clone, Debug)]
pub struct DraftStep {
    /// The plan's own op name, so a launch report reads like [`saber_model::dflash_graph`].
    pub name: String,
    /// Which phase it belongs to.
    pub phase: DraftPhase,
    /// The resolved descriptor.
    pub desc: LaunchDescriptor,
    /// Its bindings.
    pub args: ArgBindings,
}

/// Where one activation lives in the proposal's arena: a byte offset and a length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Byte offset from the arena base.
    pub at: u64,
    /// Bytes.
    pub bytes: u64,
}

/// The proposal's activations as ONE allocation, laid out on the host.
///
/// `draft_proposal.rs` allocates every intermediate with its own `hipMalloc` (`Pool2`), which is
/// fine for a command that runs three prompts and exits and is not fine inside a step loop: a
/// `hipMalloc`/`hipFree` pair per buffer per step is tens of allocations a step, and an address
/// that moves between two replays is a captured graph pointing at freed memory. So the layout is
/// computed here, with no card in the room, and the engine makes one allocation of
/// [`DraftArena::len`] bytes whose base every binding is an offset from.
#[derive(Clone, Debug, Default)]
pub struct DraftArena {
    regions: BTreeMap<String, Region>,
    len: u64,
}

/// Device-pointer alignment for every region. 256 B is what a `hipMalloc` guarantees and what the
/// 16-byte vector loads in these kernels need several times over.
const ALIGN: u64 = 256;

impl DraftArena {
    fn alloc(&mut self, name: &str, bytes: u64) -> u64 {
        let at = self.len;
        self.regions.insert(
            name.to_owned(),
            Region {
                at,
                bytes: bytes.max(1),
            },
        );
        self.len = at + bytes.max(1).div_ceil(ALIGN) * ALIGN;
        at
    }

    /// Bytes the whole arena needs.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether anything was laid out.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Regions laid out.
    #[must_use]
    pub fn count(&self) -> usize {
        self.regions.len()
    }

    /// Every region, by name — for a caller that has to address one without knowing the layout.
    pub fn regions(&self) -> impl Iterator<Item = (&str, Region)> {
        self.regions.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// One region, by name.
    ///
    /// # Errors
    /// A name this layout does not carry — which is a plan that binds a buffer the layout never
    /// reserved, and an unreserved binding is whatever the arena held.
    pub fn region(&self, name: &str) -> Result<Region, ForwardError> {
        self.regions.get(name).copied().ok_or_else(|| {
            ForwardError::Device(format!(
                "the drafter's arena has no region `{name}`: a binding with no region reads \
                 whatever the arena held"
            ))
        })
    }
}

/// **Every shape one boot plans**, in the order the residency builds them: the PROMPT's store
/// shape first (one request, `block` rows) and then one per `(decode batch, verify width)` pair
/// the engine can schedule, with a duplicate of the store shape dropped.
///
/// It is a free function because it has TWO callers that must not drift: the residency, which
/// allocates the arena from it, and the byte ledger, which bills that arena at the drafter tier's
/// own site — **before** the KV pool is derived from what is left (ws-k13). The arena was a raw
/// `hipMalloc` the pool never saw until ws-s11, and it grows with the batch (20 MiB at one request
/// to 472 at thirty-two, TP=1), so a pool derived without it is a pool competing with it out of
/// `kv_headroom_mib`.
///
/// The ORDER is load-bearing: a shape's index is what the `engine.draft` wire header carries, so
/// two callers that built the same set in a different order would agree on the arena and disagree
/// on the frame.
#[must_use]
pub fn planned_shapes(
    cfg: &DflashConfig,
    kv_block: u32,
    max_ctx: u32,
    widths: &[u32],
    batches: &[u32],
    max_seqs: u32,
) -> Vec<DraftShape> {
    let per_seq_blocks = max_ctx.div_ceil(kv_block.max(1));
    let store = DraftShape {
        reqs: 1,
        ctx_rows: cfg.block_size,
        kv_block,
        bt_stride: per_seq_blocks.max(1),
        seq_len_bound: max_ctx + cfg.block_size,
    };
    let mut list = vec![store];
    for w in widths {
        for b in batches {
            let (b, w) = (*b, *w);
            if b == 0 || w == 0 || b > max_seqs {
                continue;
            }
            let sh = DraftShape {
                reqs: b,
                ctx_rows: b * w,
                ..store
            };
            if !list.contains(&sh) {
                list.push(sh);
            }
        }
    }
    list
}

/// What the arena costs over a shape list: the WIDEST, because two shapes never run in the same
/// step and every region of a proposal is written before it is read, so one allocation serves them
/// all and a boot pays the maximum rather than the sum.
#[must_use]
pub fn widest_arena(cfg: &DflashConfig, shapes: &[DraftShape]) -> u64 {
    shapes
        .iter()
        .map(|s| DraftPlan::arena_for(cfg, *s).len())
        .max()
        .unwrap_or(0)
}

/// The geometry one proposal runs at.
///
/// `ctx_rows` and `query_rows` are separate because they are separate facts: the context store
/// runs over the TARGET's rows this step and the drafter's forward runs over `reqs * block` rows
/// of its own, and only at one request in steady state are the two equal.
///
/// # The BATCHED shape (ws-s10)
///
/// `reqs` is the proposal's batch and it is a SHAPE, exactly as the target's `(batch, width)`
/// bucket is: the arena's regions are `reqs * block` rows tall, every GEMM resolves at that M
/// through the 32-row bands F33/F34 gated, and the captured backbone is one graph per shape. A
/// step with fewer live sequences than the shape pads the rows it does not have -- the head
/// refuses a partial tile, so a batch is padded UP to a planned shape and never trimmed down to a
/// grid nobody sized.
///
/// **`ctx_rows` is `reqs * ctx_per_req`, and `ctx_per_req` is the STEP'S VERIFY WIDTH** rather
/// than the drafter's block. The store reads the target's hidden states out of `aux.cat`, which
/// the tap writes at the step's own row layout (slot `i`'s rows at `i * width`), so a batched
/// store is ONE contiguous window only if the plan's per-request context is that same `width`. At
/// width 8 -- prod's block, and the shape every S3..S9 number was taken at -- the two coincide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftShape {
    /// Requests in this proposal.
    pub reqs: u32,
    /// Rows the context store covers: the target's own query rows this step, over every request.
    pub ctx_rows: u32,
    /// The drafter's KV page size, in tokens.
    pub kv_block: u32,
    /// Entries per sequence in the drafter's block table.
    pub bt_stride: u32,
    /// The largest absolute position the drafter's attention may read up to. It feeds a
    /// `k >= m` check in `dflash::resolve` and NOTHING the kernel reads — the real per-request
    /// lengths ride in the `seq_lens` BUFFER — which is what lets this phase be captured.
    pub seq_len_bound: u32,
}

/// Every device address a proposal binds that is not its own arena.
#[derive(Clone, Debug)]
pub struct DraftBindings {
    /// The arena's base.
    pub arena: DeviceAddr,
    /// `[ctx_rows, 5 * hidden]` bf16: the target's five tapped layers, written by
    /// [`crate::aux_tap`].
    pub aux_cat: DeviceAddr,
    /// `[layers * ctx_rows]` i64: the context rope's positions, the SAME table repeated per layer.
    pub ctx_positions: DeviceAddr,
    /// `[layers][ctx_rows]` i64, one per layer in order: where each context row's K/V lands.
    pub ctx_slots: Vec<DeviceAddr>,
    /// The drafter's K cache, one per layer.
    pub kv_k: Vec<DeviceAddr>,
    /// The drafter's V cache, one per layer.
    pub kv_v: Vec<DeviceAddr>,
    /// `kv_heads_local` f32 ones: the K/V append's per-head scales.
    pub kv_scales: DeviceAddr,
    /// One f32 one: the attention's `k_scale` / `v_scale`.
    pub one_f32: DeviceAddr,
    /// `[reqs * block]` i32: the anchor at each request's row 0, the mask token elsewhere.
    pub input_ids: DeviceAddr,
    /// `[reqs * block]` i64: the drafter's query positions, one per token.
    ///
    /// ONE row, not three. `attn_glue::qk_norm_rope` reads a three-row mrope section map and the
    /// drafter's rope is plain, so a replay through it had to replicate; `dflash_qk` is the
    /// drafter's own kernel and reads `pos[tok]` (ws-f28).
    pub query_pos: DeviceAddr,
    /// `[reqs * block]` i64: where each drafter query row's K/V lands.
    pub query_slots: DeviceAddr,
    /// `[reqs, bt_stride]` i32.
    pub block_table: DeviceAddr,
    /// `[reqs]` i32: `last_valid_pos + 1 + block`, the DRAFTER's own lengths.
    pub seq_lens: DeviceAddr,
    /// `[reqs * steps]` i64: which query row each proposal position samples from.
    pub sample_indices: DeviceAddr,
    /// `[reqs]` i64: each request's row 0, for the anchor gather.
    pub anchor_indices: DeviceAddr,
    /// `[reqs * steps]` i32: `sample_idx_mapping`, the per-request state slot.
    pub req_state: DeviceAddr,
    /// Weights by op name, each entry `(kernarg, address)` in the record set's own order.
    pub weights: BTreeMap<String, Vec<(String, DeviceAddr)>>,
}

/// The proposal, in two phases.
#[derive(Clone, Debug)]
pub struct DraftPlan {
    /// The arena layout it was built against.
    pub arena: DraftArena,
    /// Every launch, in forward order, each tagged with its phase.
    pub steps: Vec<DraftStep>,
    /// The shape it was built at.
    pub shape: DraftShape,
}

impl DraftPlan {
    /// Lay the arena out for `cfg` at `shape`. Pure: no card, no rules, no addresses.
    #[must_use]
    pub fn arena_for(cfg: &DflashConfig, shape: DraftShape) -> DraftArena {
        let mut a = DraftArena::default();
        let h = u64::from(cfg.hidden);
        let m = u64::from(shape.reqs) * u64::from(cfg.block_size);
        let nkv = u64::from(cfg.kv_heads_local());
        let hd = u64::from(cfg.head_dim);

        // ---- the context store ------------------------------------------------------------
        lay_context_store(&mut a, cfg, shape);

        // ---- the backbone ------------------------------------------------------------------
        let qn = u64::from(cfg.q_heads_local()) * hd;
        let qkv_n = u64::from(cfg.qkv_n());
        let kp_n = u64::from(cfg.kernel_projection_n());
        let inter = u64::from(cfg.intermediate / cfg.tp.max(1));
        a.alloc("bb.hidden", m * h * 2);
        a.alloc("bb.normed0", m * h * 2);
        for l in 0..cfg.layers {
            let p = |s: &str| format!("bb.l{l}.{s}");
            a.alloc(&p("coeff_a"), m * kp_n * 2);
            a.alloc(&p("coeff_m"), m * kp_n * 2);
            a.alloc(&p("aq"), m * h);
            a.alloc(&p("as"), m * h.div_ceil(128) * 4);
            a.alloc(&p("qkv"), m * qkv_n * 2);
            a.alloc(&p("q_rot"), m * qn * 2);
            a.alloc(&p("k_rot"), m * nkv * hd * 2);
            a.alloc(&p("attn_out"), m * qn * 2);
            a.alloc(&p("oq"), m * qn);
            a.alloc(&p("os"), m * qn.div_ceil(128) * 4);
            a.alloc(&p("o"), m * h * 2);
            a.alloc(&p("normed2"), m * h * 2);
            // FOUR bytes: the residual between the layer's two halves is fp32 in prod's compiled
            // drafter (F29) -- `conv_add_rms_open` writes it and `_close` reads it.
            a.alloc(&p("resid2"), m * h * 4);
            a.alloc(&p("mq"), m * h);
            a.alloc(&p("ms"), m * h.div_ceil(128) * 4);
            a.alloc(&p("gate_up"), m * 2 * inter * 2);
            a.alloc(&p("dq"), m * inter);
            a.alloc(&p("ds"), m * inter.div_ceil(128) * 4);
            a.alloc(&p("down"), m * h * 2);
            a.alloc(&p("out_norm"), m * h * 2);
            a.alloc(&p("res_out"), m * h * 2);
        }

        // ---- the head and the selector -----------------------------------------------------
        let steps = u64::from(shape.reqs) * u64::from(cfg.selector_positions());
        let m_pad = pad_rows(cfg, shape);
        let vloc = u64::from(cfg.vocab_local());
        let bm_w = u64::from(cfg.head_block_maxima());
        let rerank = u64::from(cfg.rerank);
        let top_k = u64::from(cfg.selector_top_k);
        a.alloc("head.hidden", steps * h * 2);
        a.alloc("head.x_pad", m_pad * h * 2);
        a.alloc("head.xs", m_pad * h.div_ceil(128) * 4);
        a.alloc("head.y", m_pad * vloc * 2);
        a.alloc("head.bmax", m_pad * bm_w * 4);
        a.alloc("head.bidx", m_pad * bm_w * 4);
        a.alloc("head.thresh", m_pad * 8);
        a.alloc("head.tvals", m_pad * rerank * 4);
        a.alloc("head.tpos", m_pad * rerank * 4);
        a.alloc("head.cand80", m_pad * rerank * 4);
        // The dense rerank output, for the arm whose kernel is not the scatter one. Laid out
        // either way: an arena sized by which symbol the rules happen to route to would change
        // size with a rules edit.
        a.alloc("head.rr_dense", m_pad * rerank * 4);
        a.alloc("sel.vals", steps * top_k * 4);
        // EIGHT bytes an id, because `dflash_topk_bf16_k16` writes `long *` and both of its
        // consumers read `const long *`. It was sized at four — see the module doc.
        a.alloc("sel.cand64", steps * top_k * 8);
        a.alloc("sel.anchor", u64::from(shape.reqs) * 4);
        a.alloc("sel.proj", steps * u64::from(cfg.selector_rank) * 2);
        a.alloc("sel.scores", steps * top_k * top_k * 4);
        a.alloc("out.tokens", steps * 8);
        a.alloc("out.realized", steps * top_k * 4);
        a
    }

    /// Build every launch of one proposal.
    ///
    /// # Errors
    /// A shape the committed rules refuse, a weight the pack does not carry, or an arena region a
    /// binding needs and the layout did not reserve. Each one is named.
    pub fn build(
        cfg: &DflashConfig,
        shape: DraftShape,
        binds: &DraftBindings,
        rules: &DflashRules,
        head_rules: &HeadRules,
        rank: u32,
        collectives: Option<&SaberRules>,
    ) -> Result<Self, ForwardError> {
        if cfg.tp > 1 && collectives.is_none() {
            return Err(ForwardError::Device(format!(
                "a drafter plan at TP {} with no all-reduce rules: every drafter layer is \
                 row-parallel and its reduce is a launch, so a plan built without one would \
                 propose from this rank's HALF of every attention and every MLP",
                cfg.tp
            )));
        }
        let arena = Self::arena_for(cfg, shape);
        let mut b = Builder {
            cfg,
            shape,
            binds,
            rules,
            head_rules,
            rank,
            collectives,
            arena: &arena,
            steps: Vec::new(),
        };
        b.context_store()?;
        b.backbone()?;
        let steps = b.steps;
        Ok(DraftPlan {
            arena,
            steps,
            shape,
        })
    }

    /// The arena of a STORE-ONLY plan ([`DraftPlan::build_store`]): the context store's regions
    /// and nothing else.
    #[must_use]
    pub fn store_arena_for(cfg: &DflashConfig, shape: DraftShape) -> DraftArena {
        let mut a = DraftArena::default();
        lay_context_store(&mut a, cfg, shape);
        a
    }

    /// **A context store and nothing else** (H35): the wide store's plan. The same twelve launches
    /// [`DraftPlan::build`]'s eager phase issues, at `shape.ctx_rows`, over an arena of its own
    /// ([`DraftPlan::store_arena_for`], whose base is `binds.arena`), with no backbone — a prompt's
    /// wide tiles never propose; the chunk's held-back last tile does, at the one-request store
    /// shape.
    ///
    /// # Errors
    /// [`DraftPlan::build`]'s, for the store's ops.
    pub fn build_store(
        cfg: &DflashConfig,
        shape: DraftShape,
        binds: &DraftBindings,
        rules: &DflashRules,
        head_rules: &HeadRules,
        rank: u32,
    ) -> Result<Self, ForwardError> {
        let arena = Self::store_arena_for(cfg, shape);
        let mut b = Builder {
            cfg,
            shape,
            binds,
            rules,
            head_rules,
            rank,
            collectives: None,
            arena: &arena,
            steps: Vec::new(),
        };
        b.context_store()?;
        let steps = b.steps;
        Ok(DraftPlan {
            arena,
            steps,
            shape,
        })
    }

    /// The eager phase, in order.
    #[must_use]
    pub fn eager(&self) -> Vec<&DraftStep> {
        self.steps
            .iter()
            .filter(|s| s.phase == DraftPhase::ContextStore)
            .collect()
    }

    /// The captured phase, in order — up to the candidate seam at TP > 1, the whole of it at TP=1.
    #[must_use]
    pub fn captured(&self) -> Vec<&DraftStep> {
        self.steps
            .iter()
            .filter(|s| s.phase == DraftPhase::Backbone)
            .collect()
    }

    /// The captured phase AFTER the candidate seam. Empty at TP=1.
    #[must_use]
    pub fn tail(&self) -> Vec<&DraftStep> {
        self.steps
            .iter()
            .filter(|s| s.phase == DraftPhase::BackboneTail)
            .collect()
    }

    /// Every collective this proposal launches: the ops that declare a peer-visible workspace.
    ///
    /// The engine hands these to `Launcher::prepare_workspaces` and to
    /// `saber_exec::tp2::agree_on_collectives` beside the target's own, because a reduce whose
    /// two ranks disagree about a size does not answer wrongly — it never retires.
    #[must_use]
    pub fn collectives(&self) -> Vec<&DraftStep> {
        self.steps
            .iter()
            .filter(|s| {
                s.desc
                    .workspaces
                    .iter()
                    .any(|w| w.kind == saber_kernels::rules::WorkspaceKind::IpcPeerVisible)
            })
            .collect()
    }

    /// Dispatches the eager phase issues.
    #[must_use]
    pub fn eager_launches(&self) -> u32 {
        self.eager().iter().map(|s| s.desc.n_launches).sum()
    }

    /// Dispatches the captured phase issues — and therefore the node count its capture must have.
    #[must_use]
    pub fn captured_launches(&self) -> u32 {
        self.captured().iter().map(|s| s.desc.n_launches).sum()
    }

    /// Dispatches the tail issues — the node count ITS capture must have. Zero at TP=1.
    #[must_use]
    pub fn tail_launches(&self) -> u32 {
        self.tail().iter().map(|s| s.desc.n_launches).sum()
    }

    /// Host round trips INSIDE the captured phase.
    ///
    /// Zero by construction, and it is a function rather than a constant because the thing that
    /// would make it non-zero is an edit — a widening on the host, a read-back between two
    /// launches — and an edit is exactly what this should catch. Nothing in [`DraftStep`] can
    /// express a read-back, so a non-zero answer here would mean a step whose bindings name no
    /// device address at all.
    #[must_use]
    pub fn host_round_trips(&self) -> u32 {
        self.captured()
            .iter()
            .chain(self.tail().iter())
            .filter(|s| {
                s.desc.args.iter().any(|a| {
                    matches!(&a.value, ArgValue::Buffer { buffer }
                        // A WORKSPACE is not an unbound buffer: the launcher assigns it by name
                        // out of its own IPC state (the all-reduce's peer scratch, flags and
                        // sequence counter), which is exactly the arrangement that lets a
                        // collective be an ordinary graph node.
                        if s.args.get(buffer).is_none()
                            && !s.desc.workspaces.iter().any(|w| w.name == *buffer))
                })
            })
            .count() as u32
    }

    /// Every descriptor and its bindings, for `Launcher::reserve_local` — which must be called
    /// with ALL of them, both phases, **before the first capture** (`docs/EXEC.md` section 5: a
    /// capture that grows a Scratch slab frees the address every earlier graph baked).
    #[must_use]
    pub fn workspace_set(&self) -> Vec<(&LaunchDescriptor, &ArgBindings)> {
        self.steps.iter().map(|s| (&s.desc, &s.args)).collect()
    }
}

/// The context store's regions, at `shape.ctx_rows` — the first thing every proposal's arena lays
/// out, and the WHOLE of the wide store's (H35).
fn lay_context_store(a: &mut DraftArena, cfg: &DflashConfig, shape: DraftShape) {
    let h = u64::from(cfg.hidden);
    let ctx = u64::from(shape.ctx_rows);
    let nkv = u64::from(cfg.kv_heads_local());
    let hd = u64::from(cfg.head_dim);
    let aux_k = u64::from(cfg.target_layers) * h;
    let kv_n = u64::from(cfg.context_kv_n());
    let half = u64::from(cfg.layers) * ctx * nkv * hd * 2;
    a.alloc("ctx.aux.q", ctx * aux_k);
    a.alloc("ctx.aux.scales", ctx * aux_k.div_ceil(128) * 4);
    a.alloc("ctx.states", ctx * h * 2);
    a.alloc("ctx.normed", ctx * h * 2);
    a.alloc("ctx.kv_flat", ctx * kv_n * 2);
    // K and V in ONE region: the layout copy writes `[2, L, ctx, nkv, hd]` and the V half is
    // exactly `half` bytes past the K half, which is what makes it one output buffer.
    a.alloc("ctx.all_kv", half * 2);
    a.alloc("ctx.all_k_normed", half);
}

/// **The WIDE store's rows** (H35): the widest M every M-dependent op of the context store carries,
/// in whole drafter blocks — or `None` when that is not wider than one block.
///
/// Two of the store's twelve launches resolve by M: the aux projection
/// ([`DflashOp::blockscale_for`], `gemm_a8w8_blockscale_mt` to `[knobs] max_m`) and the fused KV
/// projection ([`DflashOp::bgemm_for`], `bgemm_bf16_m32` to `[bgemm] max_m`). The other ten — the
/// act quant, both norms, the layout, the rope and the five cache inserts — are per row at any M.
/// Both limits are 256 in the committed rules (`max_seqs` 32 x the block of 8), so a 2048-row
/// prefill chunk is 8 wide tiles instead of 256 narrow ones. The number is READ from the rules
/// rather than written here: a rules edit that narrows either op narrows the store with it.
#[must_use]
pub fn wide_store_rows(cfg: &DflashConfig, rules: &DflashRules) -> Option<u32> {
    let block = cfg.block_size.max(1);
    let rows = rules.bgemm.max_m.min(rules.knobs.max_m) / block * block;
    (rows > block).then_some(rows)
}

/// The wide store's SHAPE: one request, [`wide_store_rows`] context rows, the store shape's
/// geometry otherwise. `None` when the rules carry no M wider than the block.
#[must_use]
pub fn wide_store_shape(
    cfg: &DflashConfig,
    kv_block: u32,
    max_ctx: u32,
    rules: &DflashRules,
) -> Option<DraftShape> {
    let rows = wide_store_rows(cfg, rules)?;
    let store = planned_shapes(cfg, kv_block, max_ctx, &[], &[], 1)[0];
    Some(DraftShape {
        ctx_rows: rows,
        ..store
    })
}

/// The CHUNK-CLASS store's shape (H35): the one-request store shape — `block` context rows — whose
/// plan is resolved under [`chunk_class_rules`], for the narrow tiles of a chunk at least
/// `[ctx] narrow_rows` long.
#[must_use]
pub fn chunk_class_shape(cfg: &DflashConfig, wide: DraftShape) -> DraftShape {
    DraftShape {
        ctx_rows: cfg.block_size,
        ..wide
    }
}

/// **The rules a chunk-class tile resolves under** (H35): the committed rules with
/// `[ctx] narrow_rows` at 1, so `ctx_rms_norm` takes its wide-row block (`narrow_block`, 256
/// threads) at ANY launch rows. That is the ONE op of the context store whose arithmetic depends
/// on M — the block decides how many 8-wide vectors of the 5120-wide hidden state one thread
/// chains into its accumulator — and prod's launcher picks it from the CALL's rows, which on a
/// prefill chunk are the chunk's. Nothing else in the store reads `narrow_rows`.
#[must_use]
pub fn chunk_class_rules(rules: &DflashRules) -> DflashRules {
    let mut r = rules.clone();
    r.ctx.narrow_rows = 1;
    r
}

/// **The rules a WIDE tile of a chunk below the class boundary resolves under** (C6): the
/// committed rules with `[ctx] narrow_rows` at `u32::MAX`, so `ctx_rms_norm` takes its narrow-row
/// block (`wide_block`) at ANY launch rows — the block prod's launcher gives a chunk of fewer than
/// `narrow_rows` rows stored in one call, and the block every one of main's 8-row tiles of that
/// chunk already ran. The mirror of [`chunk_class_rules`]; nothing else in the store reads it.
#[must_use]
pub fn natural_class_rules(rules: &DflashRules) -> DflashRules {
    let mut r = rules.clone();
    r.ctx.narrow_rows = u32::MAX;
    r
}

/// Bytes the wide store's OWN arenas take: the context store's seven regions at its rows, and the
/// chunk-class store's at `block` rows — no backbone in either. Billed beside [`widest_arena`]
/// (H35).
#[must_use]
pub fn wide_store_arena(cfg: &DflashConfig, shape: Option<DraftShape>) -> u64 {
    shape.map_or(0, |s| {
        DraftPlan::store_arena_for(cfg, s).len()
            + DraftPlan::store_arena_for(cfg, chunk_class_shape(cfg, s)).len()
    })
}

/// **Which planned shape a step of `live` sequences at `ctx_per_req` context rows each runs at.**
///
/// A free function over `(reqs, ctx_per_req)` pairs, and it is free for the reason `state_address`
/// is: the law is "the SMALLEST planned batch that holds the step, at that width's context rows",
/// and the alternative to stating it once here is a caller that clamps. **Padded UP and never
/// trimmed down** — the int2 head refuses a partial tile, so a step of three sequences runs the
/// four-request shape with one padding row-group rather than a grid nobody sized.
///
/// `None` is a refusal: a step whose width no shape plans would otherwise propose at the wrong
/// context rows, which is a store reading another slot's hidden states.
#[must_use]
pub fn pick_shape(shapes: &[(u32, u32)], live: u32, ctx_per_req: u32) -> Option<usize> {
    shapes
        .iter()
        .enumerate()
        .filter(|(_, (reqs, ctx))| *ctx == ctx_per_req && *reqs >= live)
        .min_by_key(|(_, (reqs, _))| *reqs)
        .map(|(i, _)| i)
}

/// The head's input rows padded up to `tl.dot`'s tile: `_pow2_at_least(max(steps, 16))`.
fn pad_rows(cfg: &DflashConfig, shape: DraftShape) -> u64 {
    let steps = shape.reqs * cfg.selector_positions();
    u64::from(steps.max(16).next_power_of_two())
}

struct Builder<'a> {
    cfg: &'a DflashConfig,
    shape: DraftShape,
    binds: &'a DraftBindings,
    rules: &'a DflashRules,
    head_rules: &'a HeadRules,
    rank: u32,
    /// The all-reduce rules, at TP > 1 only. `None` is a one-card proposal and every
    /// [`Builder::all_reduce`] is then an `Identity` that launches nothing — the same law
    /// `saber_exec::layer`'s `Collectives::Identity` states for the target.
    collectives: Option<&'a SaberRules>,
    arena: &'a DraftArena,
    steps: Vec<DraftStep>,
}

impl Builder<'_> {
    /// **Which bgemm instantiation this M asks for**, and the same question for the blockscale
    /// GEMM and the int2 head.
    ///
    /// The answer is the rules' own ([`DflashOp::bgemm_for`] and its two siblings) and it is asked
    /// here rather than hard-coded because a one-request proposal's M is 8 and a batched one's is
    /// `reqs * block`: the 16-row symbols the whole proposal was gated at REFUSE an M above their
    /// tile by name, and the bands F33/F34 gated are bit-identical to them per 16-row slice. So a
    /// batched plan is the same arithmetic on the same rows through a different symbol, which is
    /// what makes "each request's block is the one-at-a-time block" a property of the source and
    /// not a hope about a rewrite.
    fn bgemm(&self, m: u32) -> DflashOp {
        DflashOp::bgemm_for(m, self.rules)
    }

    fn blockscale(&self, m: u32) -> DflashOp {
        DflashOp::blockscale_for(m, self.rules)
    }

    /// An arena region's device address.
    fn at(&self, name: &str) -> Result<DeviceAddr, ForwardError> {
        let r = self.arena.region(name)?;
        self.binds.arena.offset(r.at).ok_or_else(|| {
            ForwardError::Device(format!(
                "the drafter's arena region `{name}` at +{} is past its own allocation",
                r.at
            ))
        })
    }

    /// A weight, by op and kernarg name.
    fn w(&self, op: &str, slot: &str) -> Result<DeviceAddr, ForwardError> {
        self.binds
            .weights
            .get(op)
            .and_then(|v| v.iter().find(|(n, _)| n == slot))
            .map(|(_, a)| *a)
            .ok_or_else(|| {
                ForwardError::Device(format!("the drafter's pack carries no `{op}.{slot}`"))
            })
    }

    /// A weight by POSITION within its op's record list — for `attn.qk_norm_rope`, whose two
    /// records are both called `weight` (the record map keys on the last dot-segment) and whose
    /// order is the plan's: q_norm then k_norm, which is what the kernel's `qw`/`kw` want.
    fn w_at(&self, op: &str, i: usize) -> Result<DeviceAddr, ForwardError> {
        self.binds
            .weights
            .get(op)
            .and_then(|v| v.get(i))
            .map(|(_, a)| *a)
            .ok_or_else(|| {
                ForwardError::Device(format!("the drafter's pack carries no `{op}` record #{i}"))
            })
    }

    /// Resolve a dflash op and bind every `Buffer` slot it declares, by name.
    fn push(
        &mut self,
        name: &str,
        phase: DraftPhase,
        op: DflashOp,
        shape: DflashShape,
        binds: &[(&str, DeviceAddr)],
    ) -> Result<(), ForwardError> {
        let desc = dflash::resolve(op, shape, self.rules)
            .map_err(|e| ForwardError::Device(format!("{name}: {e}")))?;
        let args = bind_all(&desc, Shape::new(shape.m, shape.n, shape.k), binds, name)?;
        self.steps.push(DraftStep {
            name: name.to_owned(),
            phase,
            desc,
            args,
        });
        Ok(())
    }

    /// One of the proposal's exact all-reduces, IN PLACE over the edge the row-parallel GEMM
    /// before it left this rank's partial sum in.
    ///
    /// At TP=1 it launches nothing and says so by launching nothing: the target's planner calls
    /// that `Route::Identity`, and here the absence of `collectives` is the same fact. At TP > 1
    /// it is an ordinary [`DraftStep`] and therefore an ordinary NODE of the captured backbone —
    /// `PeerCapture::Lockstep`'s assertion, the same one the target's 129 reduces ride on: the
    /// one-shot kernel's protocol state is a per-block sequence counter that a replay advances
    /// exactly as a fresh launch does, provided both ranks replay the same graph the same number
    /// of times in the same order. One proposal per rank per live slot, in slot order, does.
    ///
    /// The message is `rows x cols` bf16 — 80 KiB at the drafter's own block — which is below
    /// `quant_min_bytes` and therefore the EXACT family whatever `--ar-exact` says. One band by
    /// construction, and a message that needed two is refused by name rather than silently
    /// declaring a launch count nothing asserts.
    fn all_reduce(
        &mut self,
        name: &str,
        phase: DraftPhase,
        edge: DeviceAddr,
        rows: u64,
        cols: u64,
    ) -> Result<(), ForwardError> {
        let Some(rules) = self.collectives else {
            return Ok(());
        };
        let descs = saber::resolve_ar_banded(
            rows,
            cols,
            2,
            saber::ArPayload::Bf16,
            self.cfg.tp,
            rules,
            &saber::Registry::new(rules),
        )
        .map_err(|e| ForwardError::Device(format!("{name}: {e}")))?;
        if descs.len() != 1 {
            return Err(ForwardError::Device(format!(
                "{name}: a {rows} x {cols} bf16 reduce needs {} bands; the drafter's messages are \
                 one decode block wide and a banded one here would declare a launch count this \
                 plan does not assert",
                descs.len()
            )));
        }
        let desc = descs.into_iter().next().expect("one band");
        let (inn, outn) = saber::ar_band_names(0);
        self.push_desc(
            name,
            phase,
            desc,
            Shape::new(0, 0, 0),
            &[(inn.as_str(), edge), (outn.as_str(), edge)],
        )
    }

    /// The same for a descriptor another plugin resolved.
    fn push_desc(
        &mut self,
        name: &str,
        phase: DraftPhase,
        desc: LaunchDescriptor,
        shape: Shape,
        binds: &[(&str, DeviceAddr)],
    ) -> Result<(), ForwardError> {
        let args = bind_all(&desc, shape, binds, name)?;
        self.steps.push(DraftStep {
            name: name.to_owned(),
            phase,
            desc,
            args,
        });
        Ok(())
    }

    // ---- phase 1 -----------------------------------------------------------------------------

    fn context_store(&mut self) -> Result<(), ForwardError> {
        let cfg = self.cfg;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let nkv = cfg.kv_heads_local();
        let ctx = self.shape.ctx_rows;
        let aux_k = cfg.target_layers * h;
        let kv_n = cfg.context_kv_n();
        let half = u64::from(cfg.layers) * u64::from(ctx) * u64::from(nkv) * u64::from(hd) * 2;

        let aux = self.binds.aux_cat;
        let aq = self.at("ctx.aux.q")?;
        let aqs = self.at("ctx.aux.scales")?;
        let states = self.at("ctx.states")?;
        let normed = self.at("ctx.normed")?;
        let kv_flat = self.at("ctx.kv_flat")?;
        let all_kv = self.at("ctx.all_kv")?;
        let all_k_normed = self.at("ctx.all_k_normed")?;
        let all_v = all_kv.offset(half).ok_or_else(|| {
            ForwardError::Device("the context store's V half is past its own region".to_owned())
        })?;

        let p = DraftPhase::ContextStore;
        self.push(
            "aux_hidden_projection.quant",
            p,
            DflashOp::ActQuant,
            DflashShape::new(ctx, aux_k, aux_k),
            &[("x", aux), ("q", aq), ("scales", aqs)],
        )?;
        let w_aux = self.w("aux_hidden_projection", "w")?;
        let ws_aux = self.w("aux_hidden_projection", "ws")?;
        let op = self.blockscale(ctx);
        self.push(
            "aux_hidden_projection",
            p,
            op,
            DflashShape::new(ctx, h, aux_k),
            &[
                ("a", aq),
                ("as", aqs),
                ("w", w_aux),
                ("ws", ws_aux),
                ("c", states),
            ],
        )?;
        let w_hn = self.w("context_kv.hidden_norm", "weight")?;
        self.push(
            "context_kv.hidden_norm",
            p,
            DflashOp::CtxRmsNorm,
            DflashShape::ctx_norm(ctx, h, ctx),
            &[("x", states), ("weight", w_hn), ("out", normed)],
        )?;
        let w_kvp = self.w("context_kv.kv_projection", "w")?;
        let op = self.bgemm(ctx);
        self.push(
            "context_kv.kv_projection",
            p,
            op,
            DflashShape::new(ctx, kv_n, h),
            &[("a", normed), ("w", w_kvp), ("c", kv_flat)],
        )?;
        self.push(
            "context_kv.layout",
            p,
            DflashOp::CtxKvLayout,
            DflashShape::ctx_layout(ctx, cfg.layers, nkv, hd),
            &[("in", kv_flat), ("out", all_kv)],
        )?;
        let w_kn = self.w("context_kv.k_norm", "weight")?;
        self.push(
            "context_kv.k_norm",
            p,
            DflashOp::CtxRmsNorm,
            DflashShape::ctx_norm(cfg.layers * ctx * nkv, hd, ctx * nkv),
            &[("x", all_kv), ("weight", w_kn), ("out", all_k_normed)],
        )?;
        let cos_sin = self.w("context_kv.rope", "cos_sin")?;
        self.push(
            "context_kv.rope",
            p,
            DflashOp::CtxRope,
            DflashShape::ctx_rope(cfg.layers * ctx, nkv, hd, hd),
            &[
                ("positions", self.binds.ctx_positions),
                ("k", all_k_normed),
                ("cos_sin", cos_sin),
            ],
        )?;
        let per_layer = u64::from(ctx) * u64::from(nkv) * u64::from(hd) * 2;
        for l in 0..cfg.layers {
            let k = all_k_normed
                .offset(u64::from(l) * per_layer)
                .ok_or_else(|| {
                    ForwardError::Device(format!("layer {l}'s K slice is past its own region"))
                })?;
            let v = all_v.offset(u64::from(l) * per_layer).ok_or_else(|| {
                ForwardError::Device(format!("layer {l}'s V slice is past its own region"))
            })?;
            let (kc, vc, slots) = self.layer_cache(l)?;
            let mut shape = DflashShape::new(ctx, nkv, hd);
            shape.block_size = self.shape.kv_block;
            self.push(
                &format!("context_kv.cache_insert.l{l}"),
                p,
                DflashOp::KvAppendFp8,
                shape,
                &[
                    ("key", k),
                    ("value", v),
                    ("kc", kc),
                    ("vc", vc),
                    ("slots", slots),
                    ("scales", self.binds.kv_scales),
                ],
            )?;
        }
        Ok(())
    }

    fn layer_cache(&self, l: u32) -> Result<(DeviceAddr, DeviceAddr, DeviceAddr), ForwardError> {
        let i = l as usize;
        let kc = *self.binds.kv_k.get(i).ok_or_else(|| {
            ForwardError::Device(format!("the drafter has no K cache for layer {l}"))
        })?;
        let vc = *self.binds.kv_v.get(i).ok_or_else(|| {
            ForwardError::Device(format!("the drafter has no V cache for layer {l}"))
        })?;
        let slots = *self.binds.ctx_slots.get(i).ok_or_else(|| {
            ForwardError::Device(format!("the context store has no slot table for layer {l}"))
        })?;
        Ok((kc, vc, slots))
    }

    // ---- phase 2 -----------------------------------------------------------------------------

    #[allow(clippy::too_many_lines)]
    fn backbone(&mut self) -> Result<(), ForwardError> {
        let cfg = self.cfg;
        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let nkv = cfg.kv_heads_local();
        let nq = cfg.q_heads_local();
        let qn = nq * hd;
        let qkv_n = cfg.qkv_n();
        let kp_n = cfg.kernel_projection_n();
        let inter = cfg.intermediate / cfg.tp.max(1);
        let groups = h / cfg.conv_group;
        let vloc = cfg.vocab_local();
        let m = self.shape.reqs * cfg.block_size;
        let p = DraftPhase::Backbone;

        // 1. the embedding, at this rank's own vocabulary window.
        let table = self.w("input_embedding", "weight")?;
        let hidden = self.at("bb.hidden")?;
        let embed = head::resolve(
            HeadOp::Embedding,
            HeadShape {
                rows: m,
                cols: h,
                src_stride: i64::from(h),
                vocab: (self.rank * vloc, (self.rank + 1) * vloc),
            },
            self.head_rules,
        )
        .map_err(|e| ForwardError::Device(format!("input_embedding: {e}")))?;
        self.push_desc(
            "input_embedding",
            p,
            embed,
            Shape::new(m, h, 0),
            &[
                ("table", table),
                ("ids", self.binds.input_ids),
                ("out", hidden),
            ],
        )?;
        // THE EMBEDDING'S OWN REDUCE, which the drafter's declared graph did not carry until this
        // round. `head::embedding` writes a ZERO row for every id outside this rank's window
        // (`saber_exec::layer`'s vocab-parallel slice), so at two ranks each token's embedding is
        // the sum of one non-zero contribution and one zero — and a proposal that skipped it
        // would start its five layers from half the tokens' embeddings and zeros for the rest.
        // The target's own graph has always declared one; the drafter shares the target's
        // `VocabParallelEmbedding`, whose `forward` all-reduces at `tp_size > 1`.
        self.all_reduce(
            "input_embedding.all_reduce",
            p,
            hidden,
            u64::from(m),
            u64::from(h),
        )?;

        let w0 = self.w("layer.0.input_layernorm", "weight")?;
        let mut normed = self.at("bb.normed0")?;
        self.push(
            "layer.0.input_layernorm",
            p,
            DflashOp::RmsNorm,
            DflashShape::new(m, h, h),
            &[("x", hidden), ("weight", w0), ("out", normed)],
        )?;
        let mut resid = hidden;

        let cos_sin = self.w("context_kv.rope", "cos_sin")?;
        for l in 0..cfg.layers {
            let n = |s: &str| format!("layer.{l}.{s}");
            let r = |s: &str| format!("bb.l{l}.{s}");
            let coeff_a = self.at(&r("coeff_a"))?;
            let coeff_m = self.at(&r("coeff_m"))?;
            let aq = self.at(&r("aq"))?;
            let as_ = self.at(&r("as"))?;
            let qkv = self.at(&r("qkv"))?;
            let q_rot = self.at(&r("q_rot"))?;
            let k_rot = self.at(&r("k_rot"))?;
            let attn_out = self.at(&r("attn_out"))?;
            let oq = self.at(&r("oq"))?;
            let os = self.at(&r("os"))?;
            let o = self.at(&r("o"))?;
            let normed2 = self.at(&r("normed2"))?;
            let resid2 = self.at(&r("resid2"))?;
            let mq = self.at(&r("mq"))?;
            let ms = self.at(&r("ms"))?;
            let gate_up = self.at(&r("gate_up"))?;
            let dq = self.at(&r("dq"))?;
            let ds = self.at(&r("ds"))?;
            let down = self.at(&r("down"))?;
            let out_norm = self.at(&r("out_norm"))?;
            let res_out = self.at(&r("res_out"))?;
            let (kc, vc, _) = self.layer_cache(l)?;

            let w_kpa = self.w(&n("attention_conv.kernel_projection"), "w")?;
            let op = self.bgemm(m);
            self.push(
                &n("attention_conv.kernel_projection"),
                p,
                op,
                DflashShape::new(m, kp_n, h),
                &[("a", normed), ("w", w_kpa), ("c", coeff_a)],
            )?;
            let base_a = self.w(&n("attention_conv.prepare"), "base")?;
            self.push(
                &n("attention_conv.prepare"),
                p,
                DflashOp::GroupedConvQuant,
                DflashShape::conv(m, h, groups, cfg.block_size, 0),
                &[
                    ("x", normed),
                    ("coeff", coeff_a),
                    ("base", base_a),
                    ("q", aq),
                    ("scales", as_),
                ],
            )?;
            let w_qkv = self.w(&n("attn.qkv"), "w")?;
            let ws_qkv = self.w(&n("attn.qkv"), "ws")?;
            let op = self.blockscale(m);
            self.push(
                &n("attn.qkv"),
                p,
                op,
                DflashShape::new(m, qkv_n, h),
                &[
                    ("a", aq),
                    ("as", as_),
                    ("w", w_qkv),
                    ("ws", ws_qkv),
                    ("c", qkv),
                ],
            )?;
            // THE DRAFTER'S OWN q/k NORM + ROPE, which is `dflash_qk` and not `attn_glue`:
            // ws-f28 found that the glue op is the TARGET's GemmaRMSNorm (`1 + w`) over ATen's
            // reduction with a non-contracting rope, and the drafter's is the reference server's plain `RMSNorm`
            // over Triton's. Same kernarg names, one launch, and the positions are read once per
            // TOKEN (`pos[tok]`) rather than through a three-row mrope section map.
            let qw = self.w_at(&n("attn.qk_norm_rope"), 0)?;
            let kw = self.w_at(&n("attn.qk_norm_rope"), 1)?;
            self.push(
                &n("attn.qk_norm_rope"),
                p,
                DflashOp::QkNormRope,
                DflashShape::qk_rope(m, nq, nkv, hd),
                &[
                    ("qkv", qkv),
                    ("qw", qw),
                    ("kw", kw),
                    ("cos_sin", cos_sin),
                    ("pos", self.binds.query_pos),
                    ("q_out", q_rot),
                    ("k_out", k_rot),
                ],
            )?;
            // V is the tail of `[q | k | v]`, and the append writes k and v together.
            let v_off = qkv.offset(u64::from(qn + nkv * hd) * 2).ok_or_else(|| {
                ForwardError::Device(format!("layer {l}: the V view is past its own region"))
            })?;
            let mut ap = DflashShape::new(m, nkv, hd);
            ap.block_size = self.shape.kv_block;
            // **THE SOURCE STRIDES, and they are not the contiguous default.** `key` is the rope's
            // own contiguous `nkv * head_dim` output and `value` is a VIEW of `[q | k | v]`, so its
            // rows are the PROJECTION's `qkv_n` apart. ws-f29 found this from the other end and
            // paid a whole cache row for it (7,121 of 3,375,104 bytes at `kv.after_fwd`): an
            // append that assumes contiguity reads v row 0 correctly and every row after it out of
            // the NEXT row's q. `DraftGraph` declares the pair and `draft_proposal.rs` binds it;
            // this plan did not, which is the one thing the engine's replay could tell that
            // neither of those two can — a staged replay seeds the attention from prod's own
            // cache, and the context store's own append IS contiguous, so both of the gates that
            // exist were blind to it (ws-s4).
            ap.kv_src_stride = (nkv * hd, qkv_n);
            self.push(
                &n("attn.kv_append"),
                p,
                DflashOp::KvAppendFp8,
                ap,
                &[
                    ("key", k_rot),
                    ("value", v_off),
                    ("kc", kc),
                    ("vc", vc),
                    ("slots", self.binds.query_slots),
                    ("scales", self.binds.kv_scales),
                ],
            )?;
            // ---- THE ATTENTION ENTERS THE GRID (ws-s11) ------------------------------------
            //
            // It was the ONE site of nineteen that did not. `dflash_attn` served one sequence and
            // said so in its own header -- the program id WAS the local q-block index,
            // `seq_lens[0]` was the length and the block table had no row stride -- so ws-s10
            // expressed the batch as `reqs` LAUNCHES at static per-request offsets: bit-identical
            // by construction, and `5 * (reqs - 1)` extra nodes in the captured backbone.
            //
            // The kernel now carries the walk the reference does (`seq = qblock * BLOCK_Q /
            // q_per_seq`, a division rather than a binary search, exact because the drafter's
            // batch is RECTANGULAR -- every request contributes the block's rows), so a batched
            // shape is ONE launch with `num_seqs` in its grid and the same backbone node count at
            // every batch. `DflashOp::attn_for` is what picks: at one request this is the same
            // descriptor, the same grid and the same symbol the one-request proposal has always
            // made, which is what keeps every F-round gate against prod's own live call a gate
            // about the kernel this plan binds.
            self.push(
                &n("attn.attention"),
                p,
                DflashOp::attn_for(self.shape.reqs, true),
                DflashShape::attn_seqs(
                    m,
                    nq,
                    self.shape.seq_len_bound,
                    nkv,
                    self.shape.kv_block,
                    self.shape.bt_stride,
                    self.shape.reqs,
                ),
                &[
                    ("q", q_rot),
                    ("kc", kc),
                    ("vc", vc),
                    ("block_table", self.binds.block_table),
                    ("seq_lens", self.binds.seq_lens),
                    ("k_scale", self.binds.one_f32),
                    ("v_scale", self.binds.one_f32),
                    ("out", attn_out),
                ],
            )?;
            self.push(
                &n("attn.o_proj.quant"),
                p,
                // NOT `ActQuant`: this one is INSIDE the drafter's compiled region, where
                // inductor strength-reduces `absmax / 448` into a reciprocal multiply, and
                // `model.fc`'s (above) is eager and divides. One source line, two programs, and
                // `DflashRules` binds one symbol per op (F31).
                DflashOp::ActQuantRecip,
                DflashShape::new(m, qn, qn),
                &[("x", attn_out), ("q", oq), ("scales", os)],
            )?;
            let w_o = self.w(&n("attn.o_proj"), "w")?;
            let ws_o = self.w(&n("attn.o_proj"), "ws")?;
            let op = self.blockscale(m);
            self.push(
                &n("attn.o_proj"),
                p,
                op,
                DflashShape::new(m, h, qn),
                &[("a", oq), ("as", os), ("w", w_o), ("ws", ws_o), ("c", o)],
            )?;
            // ROW-PARALLEL, so the layer's first reduce. In place over `o`, which holds this
            // rank's partial sum until it does.
            self.all_reduce(&n("attn.all_reduce"), p, o, u64::from(m), u64::from(h))?;
            let w_fin_a = self.w(
                &n("attention_conv.finish+post_attention_layernorm"),
                "weight",
            )?;
            self.push(
                &n("attention_conv.finish"),
                p,
                DflashOp::ConvAddRmsOpen,
                DflashShape::conv(m, h, groups, cfg.block_size, 1),
                &[
                    ("x", o),
                    ("coeff", coeff_a),
                    ("base", base_a),
                    ("resid", resid),
                    ("weight", w_fin_a),
                    ("out", normed2),
                    ("res_out", resid2),
                ],
            )?;
            let w_kpm = self.w(&n("mlp_conv.kernel_projection"), "w")?;
            let op = self.bgemm(m);
            self.push(
                &n("mlp_conv.kernel_projection"),
                p,
                op,
                DflashShape::new(m, kp_n, h),
                &[("a", normed2), ("w", w_kpm), ("c", coeff_m)],
            )?;
            let base_m = self.w(&n("mlp_conv.prepare"), "base")?;
            self.push(
                &n("mlp_conv.prepare"),
                p,
                DflashOp::GroupedConvQuant,
                DflashShape::conv(m, h, groups, cfg.block_size, 0),
                &[
                    ("x", normed2),
                    ("coeff", coeff_m),
                    ("base", base_m),
                    ("q", mq),
                    ("scales", ms),
                ],
            )?;
            let w_gu = self.w(&n("mlp.gate_up"), "w")?;
            let ws_gu = self.w(&n("mlp.gate_up"), "ws")?;
            // CMP27K: the rules' blockscale route, or (the `gateup_m64` knob, 33..64 rows) the
            // one-weight-pass 64-row tile -- the same kernargs, 0 bytes differing.
            let gu = dflash::resolve_blockscale_gateup(m, 2 * inter, h, self.rules)
                .map_err(|e| ForwardError::Device(format!("{}: {e}", n("mlp.gate_up"))))?;
            self.push_desc(
                &n("mlp.gate_up"),
                p,
                gu,
                Shape::new(m, 2 * inter, h),
                &[
                    ("a", mq),
                    ("as", ms),
                    ("w", w_gu),
                    ("ws", ws_gu),
                    ("c", gate_up),
                ],
            )?;
            self.push(
                &n("mlp.silu_quant"),
                p,
                DflashOp::SiluQuant,
                DflashShape::new(m, inter, 2 * inter),
                &[("gate_up", gate_up), ("q", dq), ("scales", ds)],
            )?;
            let w_dn = self.w(&n("mlp.down"), "w")?;
            let ws_dn = self.w(&n("mlp.down"), "ws")?;
            let op = self.blockscale(m);
            self.push(
                &n("mlp.down"),
                p,
                op,
                DflashShape::new(m, h, inter),
                &[
                    ("a", dq),
                    ("as", ds),
                    ("w", w_dn),
                    ("ws", ws_dn),
                    ("c", down),
                ],
            )?;
            // The layer's second reduce, the same way, over `down`.
            self.all_reduce(&n("mlp.all_reduce"), p, down, u64::from(m), u64::from(h))?;
            let w_fin_m = self.w(&n("mlp_conv.finish+next_layernorm"), "weight")?;
            self.push(
                &n("mlp_conv.finish"),
                p,
                DflashOp::ConvAddRmsClose,
                DflashShape::conv(m, h, groups, cfg.block_size, 1),
                &[
                    ("x", down),
                    ("coeff", coeff_m),
                    ("base", base_m),
                    ("resid", resid2),
                    ("weight", w_fin_m),
                    ("out", out_norm),
                    ("res_out", res_out),
                ],
            )?;
            normed = out_norm;
            resid = res_out;
        }

        // 2. the seam between the drafter's forward and its head: the sample gather, the row pad
        //    and the group sum. Five launches ws-f27 found by writing a CHAIN, every one of them
        //    invisible to a staged replay because a staged replay seeds their outputs.
        let steps = self.shape.reqs * cfg.selector_positions();
        let head_hidden = self.at("head.hidden")?;
        self.push(
            "head.sample_rows",
            p,
            DflashOp::RowGather,
            DflashShape::new(steps, h * 2, m),
            &[
                ("in", normed),
                ("idx", self.binds.sample_indices),
                ("out", head_hidden),
            ],
        )?;
        let y = self.at("head.y")?;
        self.proposal_logits(p, head_hidden, y, steps)?;

        // 5. the selector's own top-16 over the reranked vocabulary row, the anchor gather and the
        //    projection. The selection writes the i64 ids its consumers declare, into the ONE
        //    region all of them bind — see the module doc.
        let svals = self.at("sel.vals")?;
        let cand64 = self.at("sel.cand64")?;
        self.push(
            "selector.candidate_topk",
            p,
            DflashOp::SelectorTopk,
            DflashShape::rerank(steps, vloc, h, cfg.selector_top_k),
            &[("x", y), ("vals", svals), ("idx", cand64)],
        )?;
        // ---- THE CANDIDATE SEAM ------------------------------------------------------------
        //
        // `selector.candidate_topk` has just written the best 16 of THIS RANK'S HALF of the
        // vocabulary, as f32 values and SHARD-LOCAL i64 columns. At TP=1 that is the row's own
        // top-16 and its consumers read it where it lies. At TP > 1 it is half an answer twice,
        // and what follows is the merge: the two ranks swap their 16 pairs over the rendezvous
        // socket, each adds `rank * vocab_local` to its own columns, and the top-16 of the 32 is
        // what the codebooks are then gathered by. That merge is HOST work — the same law and the
        // same transport as `logits.gather` — so the two ops that read a whole row are a captured
        // TAIL of their own and the host writes `sel.cand64` back, GLOBAL, in between.
        let seam = self.collectives.is_some();
        let tail = if seam { DraftPhase::BackboneTail } else { p };
        let anchor = self.at("sel.anchor")?;
        self.push(
            "selector.anchor_ids",
            p,
            DflashOp::RowGather,
            DflashShape::new(self.shape.reqs, 4, m),
            &[
                ("in", self.binds.input_ids),
                ("idx", self.binds.anchor_indices),
                ("out", anchor),
            ],
        )?;
        let proj = self.at("sel.proj")?;
        let w_hp = self.w("selector.hidden_projection", "w")?;
        let op = self.bgemm(steps);
        self.push(
            "selector.hidden_projection",
            p,
            op,
            DflashShape::new(steps, cfg.selector_rank, h),
            &[("a", head_hidden), ("w", w_hp), ("c", proj)],
        )?;
        let scores = self.at("sel.scores")?;
        let mut sb: Vec<(&str, DeviceAddr)> = vec![
            ("cand", cand64),
            ("anchor", anchor),
            ("unary", svals),
            ("hidden", proj),
            ("out", scores),
        ];
        let sc_weights: Vec<(String, DeviceAddr)> = self
            .binds
            .weights
            .get("selector.score_edges")
            .cloned()
            .unwrap_or_default();
        for (k, a) in &sc_weights {
            sb.push((k.as_str(), *a));
        }
        self.push(
            "selector.score_edges",
            tail,
            DflashOp::SelectorScore,
            DflashShape::selector(
                steps,
                cfg.selector_positions(),
                cfg.selector_rank,
                cfg.selector_top_k,
            ),
            &sb,
        )?;

        // 6. the walk: the draft block, one request per program, on the device.
        let tokens = self.at("out.tokens")?;
        let realized = self.at("out.realized")?;
        self.push(
            "selector.walk",
            tail,
            DflashOp::SelectorWalk,
            DflashShape::rerank(
                self.shape.reqs,
                cfg.selector_positions(),
                cfg.selector_top_k,
                cfg.selector_top_k,
            ),
            &[
                ("scores", scores),
                ("candidates", cand64),
                ("req_state", self.binds.req_state),
                ("tokens", tokens),
                ("realized", realized),
            ],
        )?;
        Ok(())
    }

    /// Fill the local vocabulary logits consumed by the selector and its TP merge.
    fn proposal_logits(
        &mut self,
        p: DraftPhase,
        head_hidden: DeviceAddr,
        y: DeviceAddr,
        steps: u32,
    ) -> Result<(), ForwardError> {
        let cfg = self.cfg;
        let h = cfg.hidden;
        let vloc = cfg.vocab_local();
        if self.rules.head.full_vocab_b1 && self.shape.reqs == 1 {
            let weight = self.w("head.rerank", "w8")?;
            let scale = self.w("head.rerank", "ws")?;
            let dense = dflash::resolve_head_w8_b1(steps, vloc, h, self.rules)
                .map_err(|e| ForwardError::Device(format!("full draft head: {e}")))?;
            return self.push_desc(
                "head.full_w8",
                p,
                dense,
                Shape::new(steps, vloc, h),
                &[("a", head_hidden), ("w", weight), ("ws", scale), ("c", y)],
            );
        }
        let m_pad = pad_rows(cfg, self.shape) as u32;
        let x_pad = self.at("head.x_pad")?;
        let xs = self.at("head.xs")?;
        self.push(
            "head.row_pad",
            p,
            DflashOp::HeadRowPad,
            DflashShape::rerank(m_pad, h, h, steps),
            &[("x", head_hidden), ("out", x_pad)],
        )?;
        self.push(
            "head.group_sum",
            p,
            DflashOp::HeadGroupSum,
            DflashShape::new(m_pad, h, h),
            &[("x", x_pad), ("xs", xs)],
        )?;

        // 3. the int2 head, its two-pass selection and the gather that selection needs.
        let bmax = self.at("head.bmax")?;
        let bidx = self.at("head.bidx")?;
        let wq = self.w("head.int2", "wq")?;
        let s = self.w("head.int2", "s")?;
        let zs = self.w("head.int2", "zs")?;
        // CMP27Z: the rules' route, or (the head's `m64` knob, whole 64-row tiles) the one-pass
        // 64-row twin -- the same kernargs, 0 bytes differing.
        let hd = dflash::resolve_draft_head_int2(m_pad, vloc, h, self.rules)
            .map_err(|e| ForwardError::Device(format!("head.int2: {e}")))?;
        self.push_desc(
            "head.int2",
            p,
            hd,
            Shape::new(m_pad, vloc, h),
            &[
                ("x", x_pad),
                ("xs", xs),
                ("wq", wq),
                ("s", s),
                ("zs", zs),
                ("y", y),
                ("bmax", bmax),
                ("bidx", bidx),
            ],
        )?;
        let bm_w = cfg.head_block_maxima();
        let sel_shape = DflashShape::rerank(m_pad, bm_w, h, cfg.rerank);
        let thresh = self.at("head.thresh")?;
        let tvals = self.at("head.tvals")?;
        let tpos = self.at("head.tpos")?;
        let cand80 = self.at("head.cand80")?;
        self.push(
            "head.candidate_topk.select",
            p,
            DflashOp::HeadTopkThresh,
            sel_shape,
            &[("x", bmax), ("thresh", thresh)],
        )?;
        self.push(
            "head.candidate_topk.gather",
            p,
            DflashOp::HeadTopkGather,
            sel_shape,
            &[
                ("x", bmax),
                ("thresh", thresh),
                ("vals", tvals),
                ("idx", tpos),
            ],
        )?;
        self.push(
            "head.candidate_topk.gather_ids",
            p,
            DflashOp::HeadCandGather,
            sel_shape,
            &[("bidx", bidx), ("pos", tpos), ("out", cand80)],
        )?;

        // 4. the exact rerank. WHICH ARM the rules route to decides where it writes: the scatter
        //    variant writes back into the head's own `[M, N]` output and the dense one into its
        //    own `[M, R]`. Asked of the descriptor rather than assumed.
        //    WHICH WEIGHT FORM is the drafter pack's fact (N6): its `head.rerank` records bind
        //    `w8` + `ws` (the checkpoint's fp8 head, shared with the target's `lm_head.w8`) or `w`
        //    (the bf16 expansion), and the op follows the records bound, never a flag.
        let rr_shape = DflashShape::rerank(m_pad, vloc, h, cfg.rerank);
        let rr_op = if self
            .binds
            .weights
            .get("head.rerank")
            .is_some_and(|ws| ws.iter().any(|(k, _)| k == "w8"))
        {
            DflashOp::RerankExactW8
        } else {
            DflashOp::RerankExactBf16
        };
        let rr = dflash::resolve(rr_op, rr_shape, self.rules)
            .map_err(|e| ForwardError::Device(format!("head.rerank: {e}")))?;
        let rr_out = if rr.kernel.contains("scatter") {
            y
        } else {
            self.at("head.rr_dense")?
        };
        let mut rb: Vec<(&str, DeviceAddr)> = vec![("x", x_pad), ("idx", cand80), ("out", rr_out)];
        let rr_weights: Vec<(String, DeviceAddr)> = self
            .binds
            .weights
            .get("head.rerank")
            .cloned()
            .unwrap_or_default();
        for (k, a) in &rr_weights {
            rb.push((k.as_str(), *a));
        }
        self.push_desc(
            "head.rerank",
            p,
            rr,
            Shape::new(rr_shape.m, rr_shape.n, rr_shape.k),
            &rb,
        )?;
        Ok(())
    }
}

/// Bind every `Buffer` slot a descriptor declares, and refuse one nothing was bound for.
///
/// An unbound buffer is whatever the arena held, which is the failure mode this whole module is
/// arranged against, so it is an error by name rather than a zero address.
fn bind_all(
    desc: &LaunchDescriptor,
    shape: Shape,
    binds: &[(&str, DeviceAddr)],
    name: &str,
) -> Result<ArgBindings, ForwardError> {
    let mut args = ArgBindings::new(shape);
    for slot in &desc.args {
        if let ArgValue::Buffer { buffer } = &slot.value {
            // A WORKSPACE is not a binding this plan owns. The all-reduce's peer scratch, its two
            // flag pages and its sequence counter are the LAUNCHER's — it assigns them by name out
            // of the IPC state it exchanged at boot (`pack_kernarg` looks in the bindings first
            // and in the workspaces second) — and a plan that tried to bind them would be
            // inventing an address for a region that has to be the same on both cards.
            if desc.workspaces.iter().any(|w| w.name == *buffer) {
                continue;
            }
            let (_, a) = binds.iter().find(|(n, _)| n == buffer).ok_or_else(|| {
                ForwardError::Device(format!(
                    "{name}: nothing bound for `{buffer}` — an unbound buffer is whatever the \
                     arena held"
                ))
            })?;
            args.bind(buffer, *a);
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arena is a pure function of the geometry, and it is ONE allocation.
    #[test]
    fn the_arena_is_one_allocation_laid_out_on_the_host() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let shape = DraftShape {
            reqs: 1,
            ctx_rows: 8,
            kv_block: 16,
            bt_stride: 8,
            seq_len_bound: 4104,
        };
        let a = DraftPlan::arena_for(&cfg, shape);
        assert!(!a.is_empty());
        // Every region is inside the arena and none overlaps the next.
        let mut seen: Vec<(u64, u64, String)> = Vec::new();
        for (name, r) in &a.regions {
            assert!(
                r.at + r.bytes <= a.len(),
                "{name} runs past the arena: {} + {} > {}",
                r.at,
                r.bytes,
                a.len()
            );
            seen.push((r.at, r.bytes, name.clone()));
        }
        seen.sort_by_key(|(at, _, _)| *at);
        for w in seen.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "`{}` overlaps `{}`",
                w[0].2,
                w[1].2
            );
            assert_eq!(w[1].0 % ALIGN, 0, "`{}` is not aligned", w[1].2);
        }
        // The five per-layer slabs plus the head's, at the drafter's own shape: tens of MiB, not
        // hundreds. A number in the report rather than an assertion on a magnitude.
        assert!(a.len() < 64 << 20, "{} bytes", a.len());
        assert!(a.region("bb.l4.gate_up").is_ok());
        assert!(a.region("bb.l5.gate_up").is_err());
    }

    /// The two phases, their launch counts, and the one launch prod does not make.
    ///
    /// The numbers are the plan's own: `DraftGraph` declares 117 launch sites of which **15 exist
    /// only at TP > 1** — eleven all-reduces (two a layer and the embedding's) and the four-launch
    /// candidate seam — so a TP=1 proposal is 102: 12 in the eager context store and 90 in the
    /// captured backbone. It was 103 while this module bridged the selection's ids to the scorer
    /// with a widen (module doc); the kernel writes i64 and there is nothing to bridge.
    #[test]
    fn the_proposal_is_twelve_eager_launches_and_a_captured_graph() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let g = saber_model::dflash_graph::DraftGraph::build(cfg.clone());
        let all_tp1: u32 = g
            .ops
            .iter()
            .filter(|o| !o.name.contains("all_reduce") && o.name != "selector.candidate_seam")
            .map(|o| o.count)
            .sum();
        assert_eq!(g.total_launches(), 117);
        assert_eq!(all_tp1, 102, "the TP=1 proposal, without the fifteen");

        // And the plan this module builds, over SYNTHETIC addresses: no card, no pack, and every
        // shape put through the committed rules. `DeviceAddr::NULL.offset(n)` is a legal address
        // to BIND and an illegal one to launch, which is exactly the half this test is about.
        let plan = fake_plan(&cfg, 1);
        assert_eq!(plan.eager_launches(), 12, "the context store");
        assert_eq!(plan.captured_launches(), 90, "the backbone");
        assert_eq!(plan.eager_launches() + plan.captured_launches(), all_tp1);
        assert_eq!(
            plan.host_round_trips(),
            0,
            "the captured phase reads nothing back"
        );
        // Every `Buffer` slot of every descriptor is bound: an unbound buffer is whatever the
        // arena held, and a plan that has one is a proposal that is a function of the arena.
        for s in &plan.steps {
            for a in &s.desc.args {
                if let ArgValue::Buffer { buffer } = &a.value {
                    assert!(s.args.get(buffer).is_some(), "{}: `{buffer}`", s.name);
                }
            }
        }
        // The workspace set is BOTH phases: `reserve_local` sizes every slab from the max over
        // the whole proposal before the first capture (`docs/EXEC.md` section 5).
        assert_eq!(plan.workspace_set().len(), plan.steps.len());
    }

    /// The five drafter layers each contribute the same fifteen launches, and the head's seam is
    /// where ws-f27's five extra ones live.
    #[test]
    fn the_backbone_is_fifteen_launches_a_layer_and_a_seam() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let plan = fake_plan(&cfg, 1);
        for l in 0..cfg.layers {
            let n = plan
                .captured()
                .iter()
                .filter(|s| s.name.starts_with(&format!("layer.{l}.")))
                .count();
            let want = if l == 0 { 16 } else { 15 };
            assert_eq!(n, want, "layer {l}");
        }
        for name in [
            "head.sample_rows",
            "head.row_pad",
            "head.group_sum",
            "head.candidate_topk.gather_ids",
            "selector.anchor_ids",
            "selector.walk",
        ] {
            assert!(
                plan.captured().iter().any(|s| s.name == name),
                "no `{name}` in the captured phase"
            );
        }
        // And nothing of the head is in the EAGER phase: the context store ends at the last cache
        // insert, which is the split prod itself makes.
        assert!(plan
            .eager()
            .iter()
            .all(|s| s.name.starts_with("aux_hidden_projection")
                || s.name.starts_with("context_kv.")));
    }

    /// **Every launch this plan makes is the one `DraftGraph` declares, to the kernarg.**
    ///
    /// The plan and the graph are two statements of the same program — the graph says what the
    /// drafter's forward IS and the plan says how the engine launches it — and nothing held them
    /// together. That cost a round: `DflashShape::kv_src_stride` was added by ws-f29 (the drafter's
    /// `value` is a view of `[q | k | v]`, so its rows are the projection's apart), declared in the
    /// graph, bound by `draft_proposal.rs`, and **silently left at the contiguous default here** —
    /// so the engine appended seven of every eight v rows out of the next row's q. Neither existing
    /// gate could see it: a staged replay seeds the attention from prod's own cache, and the
    /// context store's own append really is contiguous.
    ///
    /// So the two are compared here, on the host, with no card: for every op the graph covers with
    /// a `dflash` shape and this plan also launches, the resolved descriptors must be equal.
    #[test]
    fn every_launch_the_plan_makes_is_the_graphs_own_descriptor() {
        use saber_model::dflash_graph::{Coverage, DraftGraph};

        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let plan = fake_plan(&cfg, 1);
        let g = DraftGraph::build(cfg.clone());
        let rules = DflashRules::prod_dflash().expect("the committed dflash rules");
        let mut checked = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        let mut differ: Vec<String> = Vec::new();
        for op in &g.ops {
            let (Coverage::Dflash(dop), Some(shape)) = (&op.coverage, op.shape) else {
                continue;
            };
            let Some(step) = plan.steps.iter().find(|s| s.name == op.name) else {
                continue;
            };
            // SOME OF THE GRAPH'S SHAPES CANNOT RESOLVE, and that is a fact about the graph rather
            // than a hole in this test: `attn.attention` carries no sequence length and no
            // block-table stride, because both are properties of the STEP and not of the model
            // (`dflash::resolve` refuses a `bt_stride` of 0 by name for exactly that reason). An
            // op whose declared shape does not resolve is skipped and NAMED, so the day another
            // one joins them it is a failure here rather than a silent hole.
            let Ok(want) = dflash::resolve(*dop, shape, &rules) else {
                skipped.push(op.name.clone());
                continue;
            };
            if step.desc.kernel != want.kernel
                || step.desc.args != want.args
                || step.desc.grid != want.grid
            {
                differ.push(format!(
                    "{}: plan {} {:?} {:?} / graph {} {:?} {:?}",
                    op.name,
                    step.desc.kernel,
                    step.desc.grid,
                    step.desc.args,
                    want.kernel,
                    want.grid,
                    want.args
                ));
            }
            checked += 1;
        }
        assert!(
            checked >= 60,
            "only {checked} compared (skipped {skipped:?})"
        );
        assert!(
            skipped.iter().all(|n| n.ends_with("attn.attention")),
            "an op the graph declares stopped resolving on its own shape: {skipped:?}"
        );
        // TWO KNOWN DIFFERENCES, both the same one, and it is the PLAN that is right in both.
        // `head.row_pad` and `selector.hidden_projection` run over the rows the seam SAMPLED —
        // `selector_positions()` of them, 7 — because row 0 of the drafter's block is the ANCHOR
        // and is never sampled from; the gather before them writes exactly that many.
        // `DraftGraph` declares both at the block's 8. Named here rather than hidden behind a
        // looser comparison, so the day they are reconciled this test says so instead of quietly
        // passing.
        let names: Vec<&str> = differ
            .iter()
            .map(|d| d.split(':').next().unwrap_or(d))
            .collect();
        assert_eq!(
            names,
            vec!["head.row_pad", "selector.hidden_projection"],
            "the plan and the graph disagree: {differ:#?}"
        );
        // THE CONTROL: the append's source strides are exactly what this test exists to pin, so
        // the contiguous default must resolve to a DIFFERENT descriptor.
        let mut flat = saber_kernels::dflash::DflashShape::new(
            cfg.block_size,
            cfg.kv_heads_local(),
            cfg.head_dim,
        );
        flat.block_size = 16;
        let a =
            dflash::resolve(DflashOp::KvAppendFp8, flat, &rules).expect("the contiguous append");
        let mut view = flat;
        view.kv_src_stride = (cfg.kv_heads_local() * cfg.head_dim, cfg.qkv_n());
        let b = dflash::resolve(DflashOp::KvAppendFp8, view, &rules).expect("the strided append");
        assert_ne!(
            a.args, b.args,
            "the source strides do not reach the kernargs"
        );
    }

    /// **THE BATCHED SHAPE: one plan, `reqs` requests, and the launch count says what moved.**
    ///
    /// Four facts, and each one is a law rather than an arithmetic check:
    ///
    /// * every site resolves at `reqs = 4` -- which is F33's census turned into the plan that asks
    ///   for it, at the real shapes rather than at a hand-written list;
    /// * the three M-tiled families pick the BAND above the tile (F34) and the 16-row symbol at
    ///   one request, so the shape the whole proposal was gated at keeps the code object it was
    ///   gated with;
    /// * the ARENA is `reqs`-proportional in the rows and not in the launches;
    /// * and NO launch count moves at all (ws-s11). It used to: the attention was the one site
    ///   whose kernel served one sequence, so a batch cost `5 * (reqs - 1)` extra nodes. The
    ///   `num_seqs` walk put that site in the grid with the other eighteen, so the backbone is 90
    ///   nodes at every batch and the batched graph is the one-request graph at a taller M.
    #[test]
    fn the_batched_plan_is_one_shape_at_four_requests() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let one = fake_plan(&cfg, 1);
        let four = fake_plan(&cfg, 4);

        // The launch counts: the eager phase is unchanged (twelve launches over four times the
        // rows) and the backbone grows by the attention's fan-out alone.
        assert_eq!(one.eager_launches(), 12);
        assert_eq!(four.eager_launches(), 12, "the context store is ONE phase");
        assert_eq!(one.captured_launches(), 90);
        assert_eq!(
            four.captured_launches(),
            90,
            "the batched backbone is the one-request backbone at a taller M: the attention's \
             `num_seqs` walk is a grid extent, not a launch per request"
        );
        assert_eq!(four.host_round_trips(), 0);
        // ... and the attention is ONE step whose descriptor carries the batch. The CONTROL is
        // the one-sequence symbol: at four requests it is refused BY NAME, because handed this
        // shape it would compute request 0's attention four times and report success.
        let attn: Vec<&DraftStep> = four
            .steps
            .iter()
            .filter(|s| s.name.starts_with("layer.0.attn.attention"))
            .collect();
        assert_eq!(
            attn.len(),
            1,
            "one attention launch a layer at four requests"
        );
        assert_eq!(attn[0].desc.kernel, "dflash_attn_h128_gqa4_fp8kv_t32_ns");
        assert_eq!(
            one.steps
                .iter()
                .find(|s| s.name == "layer.0.attn.attention")
                .expect("the one-request attention")
                .desc
                .kernel,
            "dflash_attn_h128_gqa4_fp8kv_t32",
            "one request keeps the symbol every gate against prod's live call was taken with"
        );
        let rules = DflashRules::prod_dflash().expect("the committed dflash rules");
        let batched = DflashShape::attn_seqs(4 * cfg.block_size, 16, 4096, 4, 16, 8, 4);
        let e = dflash::resolve(DflashOp::AttnFp8Kv, batched, &rules)
            .expect_err("the one-sequence kernel must refuse a batch");
        assert!(e.to_string().contains("attn_fp8kv_ns"), "{e}");
        // ... and a RAGGED batch is refused too: the walk's division is exact only because every
        // request contributes the same whole number of BLOCK_Q groups.
        let ragged = DflashShape::attn_seqs(4 * cfg.block_size + 2, 16, 4096, 4, 16, 8, 4);
        dflash::resolve(DflashOp::AttnFp8KvNs, ragged, &rules)
            .expect_err("a ragged batch has no exact walk");

        // The symbols: the band above the tile, the tile at one request.
        let sym = |p: &DraftPlan, name: &str| -> String {
            p.steps
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no `{name}`"))
                .desc
                .kernel
                .clone()
        };
        for (site, at_one, at_four) in [
            (
                "context_kv.kv_projection",
                "dflash_bgemm_bf16_m16n16",
                "dflash_bgemm_bf16_m32n16",
            ),
            (
                "head.int2",
                "dflash_draft_head_int2_m16n64x2g128_lx",
                "dflash_draft_head_int2_m32n64x2g128_lx",
            ),
            (
                "layer.0.mlp.gate_up",
                "dflash_gemm_a8w8_bs128_m16n128k128",
                "dflash_gemm_a8w8_bs128_mtn128k128",
            ),
        ] {
            assert_eq!(sym(&one, site), at_one, "{site} at one request");
            assert_eq!(sym(&four, site), at_four, "{site} at four");
        }

        // THE ATTENTION'S BATCH IS IN THE GRID AND THE KERNARGS, and every per-request address is
        // gone: `q`, `out`, `block_table` and `seq_lens` are the BASES, and the kernel reaches
        // request `i`'s rows through its own walk. What must therefore differ between the two
        // plans is the grid's x extent (`rows / BLOCK_Q + num_seqs`) and the `q_per_seq` kernarg
        // the walk divides by -- if either were the one-request launch's, the batched shape would
        // be computing request 0's attention four times.
        let a = one
            .steps
            .iter()
            .find(|s| s.name == "layer.0.attn.attention")
            .expect("the one-request attention");
        let b = four
            .steps
            .iter()
            .find(|s| s.name == "layer.0.attn.attention")
            .expect("the batched attention");
        // The two plans are built over the same synthetic bindings, so the index TABLES are an
        // equality: the batched launch binds the table and not a row of it, and the length table
        // and not an entry of it.
        for k in ["block_table", "seq_lens"] {
            assert_eq!(
                b.args.get(k).expect(k).as_u64(),
                a.args.get(k).expect(k).as_u64(),
                "{k} is the table and not request `i`'s entry"
            );
        }
        // The two ARENA bindings are each region's own base — the arena's layout moves with the
        // batch, so this is stated against the layout rather than against the other plan.
        let base = a.args.get("q").expect("q").as_u64()
            - one.arena.region("bb.l0.q_rot").expect("q_rot").at;
        for (k, region) in [("q", "bb.l0.q_rot"), ("out", "bb.l0.attn_out")] {
            assert_eq!(
                b.args.get(k).expect(k).as_u64(),
                base + four.arena.region(region).expect(region).at,
                "{k} is `{region}`'s base and not request `i`'s window into it"
            );
        }
        assert_eq!(a.desc.grid[0], cfg.block_size / 4 + 1, "one sequence");
        assert_eq!(
            b.desc.grid[0],
            4 * cfg.block_size / 4 + 4,
            "the batch is `rows / BLOCK_Q + num_seqs`"
        );
        let qps = |d: &saber_kernels::rules::LaunchDescriptor| {
            d.args
                .iter()
                .find(|s| s.name == "q_per_seq")
                .map(|s| s.value.clone())
        };
        assert!(
            qps(&a.desc).is_none(),
            "the one-sequence kernel has no walk"
        );
        assert_eq!(
            qps(&b.desc),
            Some(saber_kernels::rules::ArgValue::I32 {
                ival: i64::from(cfg.block_size)
            }),
            "the walk divides by the drafter's own block"
        );

        // The arena: the backbone's rows are `reqs * block` and the head's are the padded sampled
        // rows, so `bb.hidden` scales exactly with the batch and the whole layout stays one
        // allocation with no overlap.
        let h = u64::from(cfg.hidden);
        assert_eq!(
            four.arena.region("bb.hidden").expect("bb.hidden").bytes,
            4 * u64::from(cfg.block_size) * h * 2
        );
        let mut seen: Vec<(u64, u64, String)> = four
            .arena
            .regions()
            .map(|(n, r)| (r.at, r.bytes, n.to_owned()))
            .collect();
        seen.sort_by_key(|(at, _, _)| *at);
        for w in seen.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "`{}` overlaps `{}`",
                w[0].2,
                w[1].2
            );
        }
    }

    /// **The shape a step runs at is the smallest planned batch that HOLDS it** (ws-s10), and a
    /// width nobody planned is a refusal rather than a proposal at the wrong context rows.
    #[test]
    fn a_step_runs_at_the_smallest_planned_shape_that_holds_it() {
        // The list a `--batches 1,4 --widths 8` boot builds: the PROMPT's store shape first, then
        // one per (decode batch, verify width) pair — with the duplicate of the store shape
        // dropped, which is why a b1s8 step still runs the one-request plan.
        let shapes = [(1u32, 8u32), (4, 8)];
        assert_eq!(pick_shape(&shapes, 1, 8), Some(0), "one request runs alone");
        assert_eq!(
            pick_shape(&shapes, 2, 8),
            Some(1),
            "two are PADDED up to four"
        );
        assert_eq!(pick_shape(&shapes, 3, 8), Some(1));
        assert_eq!(pick_shape(&shapes, 4, 8), Some(1));
        assert_eq!(pick_shape(&shapes, 5, 8), None, "five is nobody's shape");
        // THE WIDTH IS PART OF THE KEY, and that is the half a row count cannot carry: the store
        // reads `aux.cat` at the step's own row layout, so a width the boot did not plan has no
        // shape whose context rows are contiguous over the batch.
        assert_eq!(pick_shape(&shapes, 4, 5), None, "width 5 was never planned");
        let both = [(1u32, 8u32), (4, 8), (1, 5), (4, 5)];
        assert_eq!(pick_shape(&both, 3, 5), Some(3));
        assert_eq!(pick_shape(&both, 1, 5), Some(2));
        // ... and the SMALLEST that holds it, not the first that does.
        let wide = [(8u32, 8u32), (2, 8), (4, 8)];
        assert_eq!(pick_shape(&wide, 2, 8), Some(1));
        assert_eq!(pick_shape(&wide, 3, 8), Some(2));
        assert_eq!(pick_shape(&wide, 6, 8), Some(0));
    }

    /// **Every batch the engine can schedule plans, at BOTH rank counts**, and the arena is what
    /// it costs.
    ///
    /// The batches are the engine's `decode_batches`, so this is the census the boot would run,
    /// asked on the host with no card. The arena figure is a REPORT rather than a bar: one
    /// allocation serves every shape (they never run at once), so what the engine pays is the
    /// widest one and a number in a test is how that stops being a surprise.
    #[test]
    fn every_planned_batch_resolves_at_both_rank_counts() {
        let ar = SaberRules::prod_saber().expect("the committed saber rules");
        for tp in [1u32, 2] {
            let cfg = DflashConfig::qwen38_dflash2(tp, 7);
            let mut widest = 0u64;
            for reqs in [1u32, 2, 4, 8, 16, 32] {
                let plan = fake_plan_with(&cfg, reqs, (tp > 1).then_some(&ar));
                widest = widest.max(plan.arena.len());
                assert_eq!(plan.eager_launches(), 12, "tp{tp} reqs{reqs}");
                // FLAT IN THE BATCH since ws-s11: the attention's `5 * (reqs - 1)` was the whole
                // of the growth and the `num_seqs` walk retired it.
                let want = if tp > 1 { 99 } else { 90 };
                assert_eq!(plan.captured_launches(), want, "tp{tp} reqs{reqs}");
                assert_eq!(plan.host_round_trips(), 0, "tp{tp} reqs{reqs}");
                // A REPORT, not a bar: run the test with `--nocapture` to read what a batch costs
                // in arena bytes before a boot has to find out.
                eprintln!(
                    "tp{tp} reqs{reqs}: arena {} MiB, captured {} launch(es)",
                    plan.arena.len() >> 20,
                    plan.captured_launches()
                );
            }
            // 32 requests x the block of 8 is `max_m`, the widest any of the three M-tiled ops
            // will take, and the arena that serves it is tens of MiB rather than hundreds.
            assert!(widest < 512 << 20, "tp{tp}: {widest} bytes");
        }
    }

    /// **The arena the byte ledger bills is the arena the residency allocates**, and it grows with
    /// the batch (ws-s11).
    ///
    /// The two numbers come from ONE function now, so this test is about the number rather than
    /// about the agreement: `planned_shapes` is what `build_drafter` allocates from and what the
    /// tier's `bill(..., "engine.draft.arena")` is computed from, before the KV pool is derived.
    /// The table is S10's, re-derived here: a boot that plans a batch of eight at width 8 holds
    /// 118 MiB of arena at TP=1 that no pool derivation could see until this round.
    #[test]
    fn the_billed_arena_is_the_widest_planned_shapes() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        // The store shape alone: a boot with no decode shape at all still lays out one proposal.
        let store = planned_shapes(&cfg, 16, 4096, &[], &[], 8);
        assert_eq!(store.len(), 1, "the PROMPT's store shape is always planned");
        assert_eq!(store[0].reqs, 1);
        assert_eq!(store[0].ctx_rows, cfg.block_size);
        // S10's own table, at the widths a boot plans: one allocation serves every shape, so the
        // bill is the widest and never the sum.
        for (batch, want_mib) in [(1u32, 20u64), (2, 29), (4, 59), (8, 118)] {
            let shapes = planned_shapes(&cfg, 16, 4096, &[8], &[batch], 32);
            let bytes = widest_arena(&cfg, &shapes);
            assert_eq!(
                bytes >> 20,
                want_mib,
                "batch {batch}: {} shape(s)",
                shapes.len()
            );
            // ... and the widest is the one the residency would pick, region for region.
            let widest = shapes
                .iter()
                .copied()
                .max_by_key(|s| (s.reqs, s.ctx_rows))
                .expect("a shape");
            assert_eq!(bytes, DraftPlan::arena_for(&cfg, widest).len());
        }
        // THE CONTROL: the sum of the shapes' arenas is what a boot would pay if two shapes could
        // run in one step, and it is a DIFFERENT number — so "the widest" is a claim with teeth.
        let both = planned_shapes(&cfg, 16, 4096, &[8], &[1, 4], 8);
        assert_eq!(both.len(), 2, "the store shape is (1, 8) already");
        let sum: u64 = both
            .iter()
            .map(|s| DraftPlan::arena_for(&cfg, *s).len())
            .sum();
        assert!(
            sum > widest_arena(&cfg, &both),
            "the sum {sum} is not above the max"
        );
        // A width the boot does not plan is not in the list, which is the same fact
        // `pick_shape` refuses a step for: the ledger bills what can run.
        let w5 = planned_shapes(&cfg, 16, 4096, &[5], &[4], 8);
        assert_eq!(w5.len(), 2);
        assert_eq!(w5[1].ctx_rows, 20, "four requests at width 5");
        assert!(
            widest_arena(&cfg, &w5)
                < widest_arena(&cfg, &planned_shapes(&cfg, 16, 4096, &[8], &[4], 8)),
            "a narrower width is a narrower context store"
        );
        // A batch above `max_seqs` is not a shape this engine can schedule and is not billed.
        assert_eq!(planned_shapes(&cfg, 16, 4096, &[8], &[16], 8).len(), 1);
    }

    /// **A LADDER PLANS EVERY WIDTH IT CAN SCHEDULE** (ws-s11), which is the configuration law
    /// ws-s10's open item 4 named and `device::start` now refuses at boot.
    ///
    /// The scheduler's acceptance cap rounds UP to a captured width (ws-t1), so the widths it can
    /// pick are exactly the engine's decode buckets' — and `--engine-decode-widths 1,3,5,8` is a
    /// ladder only if every one of those widths has a drafter shape at every live count the
    /// batch can hold. The CONTROL is the other list: a drafter planned at one width serving a
    /// bucket at another has no shape for it, which is the refusal, not a padded proposal.
    #[test]
    fn a_ladder_plans_a_shape_for_every_width_it_can_schedule() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let widths = [1u32, 3, 5, 8];
        let batches = [1u32, 2, 4, 8];
        let shapes = planned_shapes(&cfg, 16, 4096, &widths, &batches, 8);
        let pairs: Vec<(u32, u32)> = shapes
            .iter()
            .map(|s| (s.reqs, s.ctx_rows / s.reqs.max(1)))
            .collect();
        for w in widths {
            for b in batches {
                for live in 1..=b {
                    assert!(
                        pick_shape(&pairs, live, w).is_some(),
                        "b{live}s{w} has no shape among {pairs:?}"
                    );
                }
            }
        }
        // THE CONTROL: the same engine whose drafter was planned at ONE width. Every bucket the
        // ladder can schedule above it is a step that would refuse by name.
        let narrow = planned_shapes(&cfg, 16, 4096, &[8], &batches, 8);
        let npairs: Vec<(u32, u32)> = narrow
            .iter()
            .map(|s| (s.reqs, s.ctx_rows / s.reqs.max(1)))
            .collect();
        let refused: Vec<(u32, u32)> = widths
            .iter()
            .flat_map(|w| batches.iter().map(move |b| (*b, *w)))
            .filter(|(b, w)| pick_shape(&npairs, *b, *w).is_none())
            .collect();
        assert_eq!(
            refused.len(),
            12,
            "three widths x four batches: {refused:?}"
        );
        // ... and the store shape is why width 8 still resolves at one request even there.
        assert_eq!(pick_shape(&npairs, 1, 8), Some(0));
    }

    /// **The head's M is a power of two and therefore never the row slice** -- which is what makes
    /// F34's "48 rows resolve under the fallback and not under the band" unreachable from here.
    ///
    /// `pad_rows` is `_pow2_at_least(max(reqs * (block - 1), 16))`, so every batch this engine can
    /// plan gives the head 16 or a multiple of 32. The row slice stays in the rules as the named
    /// fallback for an M that is whole tiles and not whole bands; no caller here produces one, and
    /// this test is the statement of that rather than a comment about it.
    #[test]
    fn the_heads_padded_rows_are_sixteen_or_whole_bands() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let rules = DflashRules::prod_dflash().expect("the committed dflash rules");
        for reqs in 1..=32u32 {
            let m_pad = pad_rows(&cfg, fake_shape(&cfg, reqs)) as u32;
            assert!(
                m_pad == 16 || m_pad % 32 == 0,
                "reqs {reqs}: the head is asked for {m_pad} rows"
            );
            let op = DflashOp::draft_head_for(m_pad, &rules);
            assert_ne!(
                op,
                DflashOp::DraftHeadInt2Mt,
                "reqs {reqs}: the row slice is unreachable from the engine's own padding"
            );
            dflash::resolve(
                op,
                DflashShape::new(m_pad, cfg.vocab_local(), cfg.hidden),
                &rules,
            )
            .unwrap_or_else(|e| panic!("reqs {reqs}: {e}"));
        }
    }

    /// A plan over synthetic addresses, for the host tests.
    fn fake_plan(cfg: &DflashConfig, reqs: u32) -> DraftPlan {
        fake_plan_with(cfg, reqs, None)
    }

    /// The shape every host test plans at: one request at prod's own block.
    fn fake_shape(cfg: &DflashConfig, reqs: u32) -> DraftShape {
        DraftShape {
            reqs,
            ctx_rows: reqs * cfg.block_size,
            kv_block: 16,
            bt_stride: 8,
            seq_len_bound: 4096 + cfg.block_size,
        }
    }

    /// Synthetic bindings, for the host tests. They come back whole rather than as a plan because
    /// one of the laws below is about the ARENA, and a region's address is `binds.arena + offset`.
    fn fake_binds(cfg: &DflashConfig) -> DraftBindings {
        use saber_model::dflash_graph::{DraftGraph, DraftRope};
        let mut next = 1u64 << 20;
        let mut fresh = || {
            next += 1 << 20;
            DeviceAddr::NULL.offset(next).expect("an address")
        };
        let g = DraftGraph::build(cfg.clone());
        let mut weights: BTreeMap<String, Vec<(String, DeviceAddr)>> = BTreeMap::new();
        for r in g.records(DraftRope::qwen38_dflash2(4096)) {
            if let Some(op) = r.op.clone() {
                weights
                    .entry(op)
                    .or_default()
                    .push((r.kernarg().to_owned(), fresh()));
            }
        }
        DraftBindings {
            arena: fresh(),
            aux_cat: fresh(),
            ctx_positions: fresh(),
            ctx_slots: (0..cfg.layers).map(|_| fresh()).collect(),
            kv_k: (0..cfg.layers).map(|_| fresh()).collect(),
            kv_v: (0..cfg.layers).map(|_| fresh()).collect(),
            kv_scales: fresh(),
            one_f32: fresh(),
            input_ids: fresh(),
            query_pos: fresh(),
            query_slots: fresh(),
            block_table: fresh(),
            seq_lens: fresh(),
            sample_indices: fresh(),
            anchor_indices: fresh(),
            req_state: fresh(),
            weights,
        }
    }

    #[test]
    fn full_vocab_b8_uses_the_fp8_head_at_b1_and_int2_at_b2() {
        let cfg = DflashConfig::qwen38_dflash2(2, 7);
        let mut binds = fake_binds(&cfg);
        binds.weights.insert(
            "head.rerank".into(),
            vec![("w8".into(), binds.arena), ("ws".into(), binds.one_f32)],
        );
        let mut rules = DflashRules::prod_dflash().unwrap();
        rules.head.full_vocab_b1 = true;
        let ar = SaberRules::prod_saber().unwrap();
        let head = HeadRules::prod_head().unwrap();
        for batch in [1, 2] {
            let plan = DraftPlan::build(&cfg, fake_shape(&cfg, batch), &binds, &rules, &head, 0, Some(&ar)).unwrap();
            let full = plan.steps.iter().find(|s| s.name == "head.full_w8");
            assert_eq!(full.is_some(), batch == 1);
            assert_eq!(plan.steps.iter().any(|s| s.name == "head.int2"), batch == 2);
            assert!(plan.steps.iter().any(|s| s.name == "selector.candidate_topk"));
            if let Some(step) = full {
                assert_eq!(step.args.get("w"), Some(binds.arena));
                assert_eq!(step.args.get("ws"), Some(binds.one_f32));
            }
        }
    }

    /// The same, with the all-reduce rules a TP > 1 plan needs.
    fn fake_plan_with(cfg: &DflashConfig, reqs: u32, ar: Option<&SaberRules>) -> DraftPlan {
        DraftPlan::build(
            cfg,
            fake_shape(cfg, reqs),
            &fake_binds(cfg),
            &DflashRules::prod_dflash().expect("the committed dflash rules"),
            &HeadRules::prod_head().expect("the committed head rules"),
            0,
            ar,
        )
        .expect("the proposal plans at prod's own shape")
    }

    /// **Every arena region is bound at ONE element width, and the ABI is what says which.**
    ///
    /// The general form of the defect the test below pins, and the piece the descriptor law was
    /// missing: `saber_kernels::abi` declares every pointer kernarg's ELEMENT width
    /// (`SigArg::elem`, which exists because F22 and F24 were both a slot declared narrower than
    /// the table behind it), the arena declares every region's BYTES — and until ws-s7 nothing
    /// held the two together. A region one op writes as `long *` and the next reads as `int *` is
    /// a contradiction every other gate in this tree is blind to: the addresses are right, the
    /// kernargs pack the same, the launch counts are unchanged, and the bytes are wrong.
    ///
    /// ONE is the ABI's byte view rather than an element width and is compatible with anything —
    /// `dflash_row_gather`'s `in` and `out` are declared 1 because the op has no arithmetic and
    /// serves bf16 rows and i32 ids alike — so the comparison is over the widths above it, which
    /// is where 4-against-8 lives. A 0 is "not declared yet" and is skipped, so this grows teeth
    /// as the signatures do rather than failing on an absence.
    #[test]
    fn every_arena_region_is_bound_at_one_element_width() {
        use saber_kernels::abi::signature_for;

        for (cfg, ar) in [
            (DflashConfig::qwen38_dflash2(1, 7), None),
            (
                DflashConfig::qwen38_dflash2(2, 7),
                Some(SaberRules::prod_saber().expect("the committed saber rules")),
            ),
        ] {
            let binds = fake_binds(&cfg);
            let base = binds.arena.as_u64();
            let plan = DraftPlan::build(
                &cfg,
                fake_shape(&cfg, 1),
                &binds,
                &DflashRules::prod_dflash().expect("the committed dflash rules"),
                &HeadRules::prod_head().expect("the committed head rules"),
                0,
                ar.as_ref(),
            )
            .expect("the proposal plans at prod's own shape");
            let regions: Vec<(u64, &str)> = plan
                .arena
                .regions()
                .map(|(n, r)| (base + r.at, n))
                .collect();
            let mut width: BTreeMap<&str, (u32, String)> = BTreeMap::new();
            let mut seen = 0usize;
            for step in &plan.steps {
                let Some(sig) = signature_for(&step.desc.kernel) else {
                    continue;
                };
                for slot in &step.desc.args {
                    let ArgValue::Buffer { buffer } = &slot.value else {
                        continue;
                    };
                    let Some(a) = step.args.get(buffer) else {
                        continue;
                    };
                    let Some((_, name)) = regions.iter().find(|(at, _)| *at == a.as_u64()) else {
                        continue;
                    };
                    let Some(w) = sig
                        .iter()
                        .find(|s| s.name == *buffer)
                        .and_then(saber_kernels::abi::SigArg::elem_bytes)
                    else {
                        continue;
                    };
                    if w <= 1 {
                        continue;
                    }
                    seen += 1;
                    let here = format!("{}.{buffer}", step.name);
                    match width.get(name) {
                        None => {
                            width.insert(name, (w, here));
                        }
                        Some((w0, first)) => assert_eq!(
                            *w0, w,
                            "tp{}: the arena region `{name}` is bound at {w0} bytes an element by \
                             `{first}` and at {w} by `{here}`: one of the two reads the right \
                             addresses and the wrong bytes",
                            cfg.tp
                        ),
                    }
                }
            }
            assert!(seen >= 80, "only {seen} region bindings carried a width");
        }
    }

    /// **The candidate ids are ONE i64 region from the selection to the walk.**
    ///
    /// ws-s7's defect, and the reason it survived three rounds is that every other law here is
    /// about descriptors: `dflash_topk_bf16_k16` changed its `idx` kernarg from `int *` to
    /// `long *` in ws-f28 (`2eb58a1`) and the plan kept a 448-byte `sel.idx32` region plus a
    /// `head::WidenI64` that read those i64s back as i32. Every launch issued, the descriptor law
    /// passed (a dtype changes neither the kernel, nor the kernargs, nor the grid), the context
    /// store and the whole body stayed bit-exact — and the scorer and the walk were handed
    /// `[id0, 0, id1, 0, …]`, which is a wrong candidate SET and a wrong draft block.
    ///
    /// So: the three ops that touch the ids bind the SAME region, and the region is eight bytes an
    /// id. Both halves are load-bearing — a same-region binding at four bytes would still be an
    /// overrun, and two regions at eight bytes each would still be two tensors.
    #[test]
    fn the_candidate_ids_are_one_i64_region_from_the_selection_to_the_walk() {
        let cfg = DflashConfig::qwen38_dflash2(1, 7);
        let plan = fake_plan(&cfg, 1);
        let addr = |op: &str, slot: &str| -> DeviceAddr {
            plan.steps
                .iter()
                .find(|s| s.name == op)
                .unwrap_or_else(|| panic!("the plan has no `{op}`"))
                .args
                .get(slot)
                .unwrap_or_else(|| panic!("`{op}` binds nothing for `{slot}`"))
        };
        let ids = addr("selector.candidate_topk", "idx");
        assert_eq!(ids, addr("selector.score_edges", "cand"));
        assert_eq!(ids, addr("selector.walk", "candidates"));
        let steps = u64::from(cfg.selector_positions());
        let k = u64::from(cfg.selector_top_k);
        assert_eq!(
            plan.arena
                .region("sel.cand64")
                .expect("the candidate region")
                .bytes,
            steps * k * 8,
            "the selection writes `long *`: eight bytes an id"
        );
        // THE CONTROL over the layout this replaced: a second, narrower region for the same tensor
        // is what the overrun was, so its absence is part of the law rather than an incidental.
        assert!(
            plan.arena.region("sel.idx32").is_err(),
            "`sel.idx32` is back: one tensor, one region"
        );
        assert!(
            !plan
                .steps
                .iter()
                .any(|s| s.name == "selector.candidate_widen"),
            "the widen is back: the selection writes what its consumers read"
        );
    }

    /// **THE TP=2 PROPOSAL'S SHAPE**: eleven reduces inside the captured backbone, and the two
    /// launches that need the whole row on the far side of the candidate seam.
    ///
    /// The numbers are the law stated as arithmetic. Ten layer reduces (two a layer, five layers)
    /// plus the embedding's, every one of them an ordinary graph node; the widen is GONE, because
    /// at two ranks the host writes the merged i64 ids and there is nothing to bridge; and the
    /// tail is exactly `score_edges` + `walk`, the two ops that read a candidate set which neither
    /// rank's own vocabulary shard contains.
    #[test]
    fn the_tp2_plan_reduces_eleven_times_and_splits_at_the_candidate_seam() {
        let one = DflashConfig::qwen38_dflash2(1, 7);
        let two = DflashConfig::qwen38_dflash2(2, 7);
        let ar = SaberRules::prod_saber().expect("the committed saber rules");
        let a = fake_plan(&one, 1);
        let b = fake_plan_with(&two, 1, Some(&ar));

        // TP=1: one graph and no reduce. There is no widen at either rank count any more (ws-s7,
        // module doc), so its absence is asserted on both plans below rather than its presence
        // here.
        assert!(
            a.collectives().is_empty(),
            "a one-card proposal reduces nothing"
        );
        assert!(a.tail().is_empty(), "there is no seam at one rank");
        assert_eq!(a.captured_launches(), 90);
        assert!(!a
            .captured()
            .iter()
            .any(|s| s.name == "selector.candidate_widen"));

        // TP=2: eleven reduces, all of them CAPTURED nodes rather than host calls.
        let ars: Vec<&str> = b.collectives().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(ars.len(), 11, "{ars:?}");
        assert_eq!(ars[0], "input_embedding.all_reduce");
        assert!(ars.contains(&"layer.0.attn.all_reduce"));
        assert!(ars.contains(&"layer.4.mlp.all_reduce"));
        assert!(
            b.collectives()
                .iter()
                .all(|s| s.phase == DraftPhase::Backbone),
            "every reduce is a node of the backbone graph, not of the eager phase"
        );
        // The split: `score_edges` and `walk` after the seam, the widen gone, and the eager phase
        // untouched (the context store is replicated input and column-parallel output at every
        // rank count, so it reduces nothing).
        let tail: Vec<&str> = b.tail().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(tail, vec!["selector.score_edges", "selector.walk"]);
        assert!(!b
            .captured()
            .iter()
            .any(|s| s.name == "selector.candidate_widen"));
        assert_eq!(b.eager_launches(), a.eager_launches());
        // 90 - 2 tail + 11 reduces = 99, and the tail is 2.
        assert_eq!(b.captured_launches(), 99);
        assert_eq!(b.tail_launches(), 2);
        // AND STILL NOTHING READS THE DEVICE BACK INSIDE A CAPTURED GRAPH. The reduce's five
        // peer-visible workspaces are bound by the launcher, not by the plan, and a plan that
        // counted them as unbound buffers would report five host round trips a layer.
        assert_eq!(b.host_round_trips(), 0);
    }

    /// A TP > 1 plan with no all-reduce rules is a refusal by NAME, not a plan with ten holes.
    #[test]
    fn a_two_rank_plan_without_collectives_is_refused() {
        let two = DflashConfig::qwen38_dflash2(2, 7);
        let mut next = 1u64 << 20;
        let mut fresh = || {
            next += 1 << 20;
            DeviceAddr::NULL.offset(next).expect("an address")
        };
        let binds = DraftBindings {
            arena: fresh(),
            aux_cat: fresh(),
            ctx_positions: fresh(),
            ctx_slots: (0..two.layers).map(|_| fresh()).collect(),
            kv_k: (0..two.layers).map(|_| fresh()).collect(),
            kv_v: (0..two.layers).map(|_| fresh()).collect(),
            kv_scales: fresh(),
            one_f32: fresh(),
            input_ids: fresh(),
            query_pos: fresh(),
            query_slots: fresh(),
            block_table: fresh(),
            seq_lens: fresh(),
            sample_indices: fresh(),
            anchor_indices: fresh(),
            req_state: fresh(),
            weights: BTreeMap::new(),
        };
        let err = DraftPlan::build(
            &two,
            DraftShape {
                reqs: 1,
                ctx_rows: two.block_size,
                kv_block: 16,
                bt_stride: 8,
                seq_len_bound: 4096 + two.block_size,
            },
            &binds,
            &DflashRules::prod_dflash().expect("rules"),
            &HeadRules::prod_head().expect("head rules"),
            0,
            None,
        )
        .expect_err("a two-rank plan with no reduce");
        assert!(err.to_string().contains("no all-reduce rules"), "{err}");
    }

    /// **The WIDE context store (H35)**: 256 rows from the committed rules, twelve eager launches
    /// and nothing captured, the two M-dependent ops routed to their M-tiled symbols, the other
    /// ten the SAME kernels as the narrow store at a taller grid — and an arena of its own that
    /// holds the context store's seven regions and no backbone.
    #[test]
    fn the_wide_store_is_the_same_twelve_ops_at_256_rows_over_its_own_arena() {
        let rules = DflashRules::prod_dflash().expect("the committed dflash rules");
        let head = HeadRules::prod_head().expect("the committed head rules");
        for tp in [1u32, 2] {
            let cfg = DflashConfig::qwen38_dflash2(tp, 7);
            assert_eq!(wide_store_rows(&cfg, &rules), Some(256), "tp {tp}");
            let wide = wide_store_shape(&cfg, 16, 4096, &rules).expect("a wide shape");
            let store = planned_shapes(&cfg, 16, 4096, &[8], &[1], 8)[0];
            assert_eq!(
                wide,
                DraftShape {
                    ctx_rows: 256,
                    ..store
                },
                "the store shape's geometry at 256 context rows"
            );
            let binds = fake_binds(&cfg);
            let w = DraftPlan::build_store(&cfg, wide, &binds, &rules, &head, 0)
                .expect("the wide store plans");
            let ar = SaberRules::prod_saber().expect("the committed saber rules");
            let ar = (tp > 1).then_some(&ar);
            let n = DraftPlan::build(&cfg, store, &binds, &rules, &head, 0, ar)
                .expect("the narrow proposal plans");
            assert_eq!(w.eager_launches(), 12, "the context store, tp {tp}");
            assert_eq!(w.captured_launches(), 0, "no backbone");
            assert_eq!(w.tail_launches(), 0);
            let (we, ne) = (w.eager(), n.eager());
            assert_eq!(we.len(), ne.len());
            for (a, b) in we.iter().zip(&ne) {
                assert_eq!(a.name, b.name, "the same ops in the same order");
                match a.name.as_str() {
                    "aux_hidden_projection" => {
                        assert_eq!(b.desc.kernel, "dflash_gemm_a8w8_bs128_m16n128k128");
                        assert_eq!(a.desc.kernel, "dflash_gemm_a8w8_bs128_mtn128k128");
                        assert_eq!(a.desc.grid[1], 256 / 16, "one 16-row tile per y");
                    }
                    "context_kv.kv_projection" => {
                        assert_eq!(b.desc.kernel, "dflash_bgemm_bf16_m16n16");
                        assert_eq!(a.desc.kernel, "dflash_bgemm_bf16_m32n16");
                        assert_eq!(a.desc.grid[1], 256 / 32, "one 32-row band per y");
                    }
                    _ => assert_eq!(a.desc.kernel, b.desc.kernel, "{}: a per-row op", a.name),
                }
            }
            // ITS OWN ARENA: the context store's regions at 256 rows, and no backbone.
            let a = DraftPlan::store_arena_for(&cfg, wide);
            assert_eq!(a.count(), 7);
            assert!(a.region("bb.hidden").is_err());
            let ca = DraftPlan::store_arena_for(&cfg, chunk_class_shape(&cfg, wide));
            assert_eq!(wide_store_arena(&cfg, Some(wide)), a.len() + ca.len());
            // THE CHUNK-CLASS STORE: the narrow store's twelve kernels at 8 rows, every
            // descriptor the same but the hidden norm's block — 256 threads, not 640.
            let crules = chunk_class_rules(&rules);
            let c = DraftPlan::build_store(
                &cfg,
                chunk_class_shape(&cfg, wide),
                &binds,
                &crules,
                &head,
                0,
            )
            .expect("the chunk-class store plans");
            for (a, b) in c.eager().iter().zip(n.eager()) {
                assert_eq!(a.desc.kernel, b.desc.kernel, "{}", a.name);
                if a.name == "context_kv.hidden_norm" {
                    assert_eq!((a.desc.block[0], b.desc.block[0]), (256, 640));
                } else {
                    assert_eq!(
                        a.desc.block, b.desc.block,
                        "{}: only the hidden norm moves",
                        a.name
                    );
                    assert_eq!(a.desc.grid, b.desc.grid, "{}", a.name);
                }
            }
            // ... and the wide tile's own hidden norm is at that class already (256 rows).
            let wn = w
                .eager()
                .into_iter()
                .find(|s| s.name == "context_kv.hidden_norm")
                .expect("the hidden norm");
            assert_eq!(wn.desc.block[0], 256);
            // C6: THE NATURAL-CLASS WIDE STORE — the wide store's twelve descriptors, grid for
            // grid, but the hidden norm at the narrow-row block (640) every 8-row tile runs; its
            // arena is the wide store's own regions (it runs over that arena).
            let nat =
                DraftPlan::build_store(&cfg, wide, &binds, &natural_class_rules(&rules), &head, 0)
                    .expect("the natural-class wide store plans");
            for (a, b) in nat.eager().iter().zip(w.eager()) {
                assert_eq!(a.desc.kernel, b.desc.kernel, "{}", a.name);
                assert_eq!(a.desc.grid, b.desc.grid, "{}", a.name);
                if a.name == "context_kv.hidden_norm" {
                    let narrow_norm = n
                        .eager()
                        .into_iter()
                        .find(|s| s.name == "context_kv.hidden_norm")
                        .expect("the narrow norm");
                    assert_eq!((a.desc.block[0], b.desc.block[0]), (640, 256));
                    assert_eq!(
                        a.desc.block, narrow_norm.desc.block,
                        "the 8-row tiles' class"
                    );
                } else {
                    assert_eq!(
                        a.desc.block, b.desc.block,
                        "{}: only the norm moves",
                        a.name
                    );
                }
            }
            assert_eq!(nat.arena.len(), w.arena.len());
            assert_eq!(wide_store_arena(&cfg, None), 0);
            // The CONTROL: the narrow store's context regions are 32x shorter.
            let na = DraftPlan::arena_for(&cfg, store);
            assert_eq!(
                a.region("ctx.aux.q").expect("aux.q").bytes,
                32 * na.region("ctx.aux.q").expect("aux.q").bytes
            );
            // A 2048-row chunk: 12 x 256 launches narrow, 12 x 8 wide.
            assert_eq!(n.eager_launches() * 2048 / store.ctx_rows, 3072);
            assert_eq!(w.eager_launches() * 2048 / wide.ctx_rows, 96);
        }
    }

    /// The wide width is READ from the rules: narrowing either M-tiled op narrows the store, and a
    /// rules table whose tiled ops reach no further than one block has no wide store at all.
    #[test]
    fn the_wide_stores_rows_follow_the_rules_and_vanish_below_two_blocks() {
        let cfg = DflashConfig::qwen38_dflash2(2, 7);
        let mut rules = DflashRules::prod_dflash().expect("the committed dflash rules");
        rules.bgemm.max_m = 100;
        assert_eq!(wide_store_rows(&cfg, &rules), Some(96), "whole blocks of 8");
        rules.knobs.max_m = 64;
        assert_eq!(wide_store_rows(&cfg, &rules), Some(64));
        rules.knobs.max_m = 8;
        assert_eq!(wide_store_rows(&cfg, &rules), None);
        assert!(wide_store_shape(&cfg, 16, 4096, &rules).is_none());
    }

    /// **P27-16: the drafter's launch census by family** (TP=2, the served block of 8, at one and
    /// eight requests), printed with `--nocapture`: every proposal launch by (phase, op family,
    /// kernel). The law it pins: the census sums to the plan's own steps, and the batch adds no
    /// launch (S10's item 3, the attention's `5 x (reqs - 1)`, retired by ws-s11's `num_seqs`
    /// walk).
    #[test]
    fn p27_16_drafter_census_by_family() {
        let ar = SaberRules::prod_saber().expect("the committed saber rules");
        let cfg = DflashConfig::qwen38_dflash2(2, 7);
        let mut totals = Vec::new();
        for reqs in [1u32, 8] {
            let plan = fake_plan_with(&cfg, reqs, Some(&ar));
            let mut by: BTreeMap<(String, String, String), u32> = BTreeMap::new();
            for s in &plan.steps {
                let fam = s
                    .name
                    .strip_prefix("layer.")
                    .and_then(|r| r.split_once('.'))
                    .map_or(s.name.clone(), |(_, r)| format!("layer.{r}"));
                *by.entry((format!("{:?}", s.phase), fam, s.desc.kernel.clone()))
                    .or_default() += s.desc.n_launches;
            }
            let total: u32 = by.values().sum();
            eprintln!("P27-16 drafter tp2 reqs{reqs}: {total} launches");
            for ((ph, fam, k), n) in &by {
                eprintln!("  {ph:<13} {fam:<40} {k:<52} {n:>4}");
            }
            let steps: u32 = plan.steps.iter().map(|s| s.desc.n_launches).sum();
            assert_eq!(total, steps, "the census is the plan's steps");
            totals.push(total);
        }
        assert_eq!(totals[0], totals[1], "the batch adds no drafter launch");
    }
}
