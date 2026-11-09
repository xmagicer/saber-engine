//! The seam: what the scheduler needs of a forward pass, and what a pass declares about itself.
//!
//! One trait, four verbs and one fact table. It exists because the whole-model pass is a command in
//! `saber-cli` rather than a library in `saber-exec`: this is the shape the engine wants, and the
//! shape a future `saber_exec::forward` should present.

use saber_plan::Phase;
use saber_sched::SeqId;
use saber_serve::engine::TokenId;

/// Why a forward step could not run.
#[derive(Debug, thiserror::Error)]
pub enum ForwardError {
    /// The device side is gone (its thread died, taking the reason with it into the log).
    #[error("the engine thread is not answering: {0}")]
    Engine(String),
    /// A shape the plan does not cover.
    #[error("no plan for {phase} at {rows} row(s): {detail}")]
    NoPlan {
        /// Which half of the step.
        phase: Phase,
        /// Rows asked for.
        rows: u32,
        /// What the planner said.
        detail: String,
    },
    /// A sequence the runner was never told about.
    #[error("sequence {0} was never admitted to the engine")]
    Unknown(SeqId),
    /// The engine has no session state left.
    #[error("the engine holds state for {live} sequence(s) and has room for {max}")]
    AtCapacity {
        /// Sequences already admitted.
        live: u32,
        /// Sequences the engine sized its state for.
        max: u32,
    },
    /// The device refused, and this is what it said.
    #[error("{0}")]
    Device(String),
    /// The other rank of a TP > 1 tier disagreed, drifted, or stopped answering.
    ///
    /// Separate from [`ForwardError::Device`] because it is a different fault with a different
    /// remedy: the card is fine and the PAIR is not. A collective whose two ranks disagree about a
    /// size or a step number must be a message before either kernel starts spinning on a flag that
    /// is not coming (`saber_exec::peer`), so every one of these is raised on the host.
    #[error("the two-rank tier: {0}")]
    Peer(String),
    /// This runner has no way to move a session's state off the device.
    #[error("this forward runner cannot park a session")]
    CannotPark,
    /// This runner has no way to branch a session: `n > 1` needs a device-to-device copy of the
    /// leader's GDN state slot set and of its one partial KV page, and a runner without one would
    /// have to start the branch from a zeroed slot and an empty page.
    #[error("this forward runner cannot fork a session")]
    CannotFork,
    /// A transfer is already in flight and there is one staging buffer. Retryable.
    #[error("a session transfer is already in flight")]
    TransferBusy,
    /// A session image was refused: the wrong shape for the destination, or it did not verify.
    #[error("session image refused: {0}")]
    Image(String),
    /// A restore was refused and the image is handed **back**.
    ///
    /// The restore seam takes the blob by value — the engine thread needs the bytes and copying them
    /// across the channel was 71.69 ms of host `memcpy` on a 403 MiB image (K9). Taking ownership
    /// means the runner must return them when it cannot use them, because the parking ticket was
    /// spent when the restore was asked for and these bytes are the only copy left. Boxed so the
    /// enum stays small: it is in the return type of every step.
    #[error("session image refused and handed back: {0}")]
    Rejected(String, Box<Vec<u8>>),
    /// A step whose write position would run past the block table the plans were built with.
    ///
    /// Found by ws-k8 on 2026-09-20: `context_len == max_ctx` is a *correct* residency count and
    /// passes every check the engine had, and then the attention kernel writes K/V at token
    /// `max_ctx`, which is page `max_ctx / block_size` of a table that is exactly that many pages
    /// wide. The kernel reads an unmapped page, the card raises "Memory access fault by GPU
    /// node-2", and the engine thread wedges behind its deadline. So the step is refused **by
    /// name, before the launch**: a sequence that has filled its context has no next token, and
    /// that is a stop rule, not a fault.
    #[error(
        "{seq}: a step at context {context} would write token {context} of a context sized for \
         {max_ctx} ({blocks} page(s) of {block_size}); this sequence is full"
    )]
    ContextFull {
        /// The sequence.
        seq: SeqId,
        /// Tokens already resident — which is also the position the step would write.
        context: u32,
        /// Tokens the plans' block table covers.
        max_ctx: u32,
        /// Pages that table is wide.
        blocks: u32,
        /// Tokens per page.
        block_size: u32,
    },
}

impl ForwardError {
    /// A refusal that hands the image back.
    #[must_use]
    pub fn rejected(what: String, blob: Vec<u8>) -> ForwardError {
        ForwardError::Rejected(what, Box::new(blob))
    }

    /// The image a refusal handed back, if it did.
    #[must_use]
    pub fn into_blob(self) -> Option<Vec<u8>> {
        match self {
            ForwardError::Rejected(_, blob) => Some(*blob),
            _ => None,
        }
    }
}

/// Rows a step runs at, which phase they belong to, and whether the step ends in a token.
///
/// The launch budget is declared per **triple**, not per row count: a prefill chunk that is not the
/// last one runs the body and stops there, because the head's `lm_head` and the three launches
/// around it belong to the step that samples. Keying the budget on `rows` alone would make every
/// non-final chunk a budget mismatch — or, worse, make the assertion so loose it never fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StepRows {
    /// Prefill chunk or decode step.
    pub phase: Phase,
    /// Rows the plan ran at — the **planned** count, padding included. For a prefill chunk that is
    /// the bucket's row count with the chunk padded up into it; for a decode step at verify width 1
    /// it is the bucket's **batch**, so one row per sequence whether or not every slot is live.
    pub rows: u32,
    /// The verify width these rows are grouped into — 1 for prefill and for a SPEC=0 decode step.
    ///
    /// Carried because **a row count is not a bucket**: `b1s5` and `b5s1` are both five rows and are
    /// different graphs, with different `cu` spans, different snapshot counts and different attention
    /// geometry. Keyed on `rows` alone they collide, and the second shape captured silently
    /// overwrites the first one's budget. `rows / width` is the batch.
    pub width: u32,
    /// Whether this step ran the head and sampled a token.
    pub samples: bool,
    /// **A decode ROW BAND (workstream C2)**: `rows` is the band's row count and `width` its MAX
    /// verify width, each slot feeding its own `1..=width` rows packed by `cu`.
    ///
    /// Part of the key because a band and a rectangle can share `(rows, width)` — the 40-row band
    /// at max width 8 and `b5s8` are both 40 rows at width 8 — and are different graphs (a batch-8
    /// `cu`, the ragged attention, another GDN route). Every map keyed on `StepRows` — the declared
    /// budgets, the captured graphs, the per-shape facts — keys on it.
    pub ragged: bool,
    /// The ancestor-mask verifier has different kernels from a rectangular chain of the same size.
    pub tree: bool,
}

impl std::fmt::Display for StepRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.tree {
            return write!(f, "{} tree ({} rows)", self.phase, self.rows);
        }
        if self.ragged {
            return write!(
                f,
                "{} r{} band (max w{}) {}",
                self.phase,
                self.rows,
                self.width,
                if self.samples { "+head" } else { "body-only" }
            );
        }
        write!(
            f,
            "{} b{}s{} (x{}) {}",
            self.phase,
            self.rows / self.width.max(1),
            self.width,
            self.rows,
            if self.samples { "+head" } else { "body-only" }
        )
    }
}

/// Which path ran a step. A fallback nobody counts is a fallback nobody notices, so this comes back
/// with every step and the executor counts both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPath {
    /// One captured graph, replayed: the shape was captured at startup.
    Graph,
    /// Op by op on the host, as `saber forward` runs it.
    Eager,
    /// C9: one FUSED mixed forward (a prompt chunk and a decode half), eager, its launch count
    /// asserted by the engine against the two plans it is built from (the prefill rung's body,
    /// the decode-side ops, the taps and both heads) — no single planned shape declares it.
    Fused,
}

impl std::fmt::Display for StepPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StepPath::Graph => "graph",
            StepPath::Eager => "eager",
            StepPath::Fused => "fused",
        })
    }
}

/// Which side put this step's input ids on the device.
///
/// A serving step's tokens come from the step before it, so at decode the device can close that loop
/// itself: the head's argmax writes the ids table the next step's embedding gathers, and the host
/// feeds nothing. It is an **optimisation, not a second source of truth** — the engine only takes it
/// when the step is the same shape, the same sequences in the same slots, and the same tokens the
/// scheduler is handing back, so a device feed cannot disagree with what the scheduler committed.
/// Everything else falls back to a host upload, and both are counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenFeed {
    /// The ids were already on the device: the previous step's argmax wrote them.
    Device,
    /// The host uploaded them — a prompt, a first decode step, or a changed slot set.
    Host,
}

impl std::fmt::Display for TokenFeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TokenFeed::Device => "device",
            TokenFeed::Host => "host",
        })
    }
}

/// One dump point of a probed decode step: a residual-stream edge, and a digest per ROW.
///
/// The instrument the multi-turn leak needed. Every id-level gate in this tree compares the END of
/// a computation, so a divergence is only ever reported where it became a different token — several
/// steps and sixty-four layers after whatever caused it. This reports the residual stream *as the
/// step walks it*, one digest per row, so two runs (or two rows of one run) can be bisected to the
/// first layer that disagrees. It is a DIAGNOSTIC: the digests are of our own bytes and say nothing
/// about whether either side is right.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowProbe {
    /// The layer whose input this edge is, `-1` for the embedding.
    pub layer: i32,
    /// The plan edge the digest was taken from.
    pub edge: String,
    /// The op the point sits next to: the WRITER of `edge` under [`ProbeDepth::Ops`], and the op
    /// the layer point is taken before or after under [`ProbeDepth::Layers`].
    ///
    /// Carried so that a bisect can name an OP and hold it against the two buckets' resolved
    /// launches for it ([`ForwardFacts::bucket_delta_for_op`]) rather than naming a layer and
    /// leaving the reader to guess which of its eleven ops moved (ws-p9).
    pub op: String,
    /// `fnv1a` of each row's bytes, in row order (padding rows included).
    pub rows: Vec<u64>,
    /// Bytes one row of this edge occupies, from the plan's own [`saber_exec::RowSpan`].
    ///
    /// Carried so that two probe sets taken in two BUCKETS can refuse to be compared when a point
    /// is not the same quantity on both sides. It is the plan's number and no longer
    /// `bytes / rows`: `l{n}.gdn.conv` is q, k and v as three blocks, so `bytes / rows` of it is
    /// five rows of q and a probe slicing by it compares five rows here and one everywhere else
    /// (ws-p9 found the symptom, ws-l18 the cause).
    pub row_bytes: u64,
    /// The rows' bytes, concatenated in row order, under [`ProbeDepth::OpsRaw`] only (H44).
    pub raw: Vec<u8>,
}

/// How deep a probed step reads.
///
/// [`ProbeDepth::Layers`] is the instrument every arm before ws-p9 used: one dump point per layer,
/// which names the first LAYER two runs part at and therefore whether the divergence is upstream
/// of the attention or inside it. [`ProbeDepth::Ops`] adds a point after **every** op of every
/// layer, which is what a bisect needs once the layer is known: the first differing OP, which is
/// the thing a plan-fact delta can be held against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeDepth {
    /// One point per layer: the residual stream, plus the embedding.
    Layers,
    /// A point after every op of every layer, on top of the layer points.
    Ops,
    /// [`ProbeDepth::Ops`] plus, for every op, a point on EACH of its input and output edges whose
    /// rows are kept RAW in [`RowProbe::raw`] (H44: the bytes an off-engine unit gate replays).
    /// Input points name the op as `<op> <in>`; an edge the plan does not hold (or with fewer
    /// rows than asked) is skipped rather than refused.
    OpsRaw,
}

/// What one forward step did: the tokens it committed, the shape it ran at, the dispatches it issued
/// and which path issued them.
///
/// The shape comes **back** from the pass rather than being assumed by the caller, because a step
/// runs at its bucket's shape and not at its own size: a 234-token prompt is one `p2048` chunk, and
/// two running sequences at the `b4s1` bucket are four rows. The caller then asserts what it can —
/// the phase, that the padding went up and never down, that a non-final chunk sampled nothing, that
/// there is one token per live slot — and checks `launches` against the budget declared for exactly
/// this shape.
#[derive(Clone, Debug, PartialEq)]
pub struct StepOutcome {
    /// One entry per **live** slot, in slot order. Empty for a prefill chunk that is not the last.
    ///
    /// A verify step commits between 1 and `verify_width` tokens per slot, so this is a list of
    /// lists rather than a list of tokens — and the list per slot is the accepted prefix plus
    /// exactly one more, which is the contract `saber_sched::SeqStep` counts.
    pub commits: Vec<SlotCommit>,
    /// The shape the pass ran.
    pub rows: StepRows,
    /// Dispatches issued by this step.
    pub launches: u32,
    /// Which path ran it.
    pub path: StepPath,
    /// Which side put its input ids on the device.
    pub feed: TokenFeed,
    /// What the pass's OWN drafter proposes for each live slot's **next** step, in slot order.
    ///
    /// Empty when the pass has no drafter of its own, which is every runner on this branch and is
    /// the SPEC=0 answer. It is on the outcome rather than behind a second call for a reason that
    /// is about ordering and not about tidiness: a real DFlash2 proposal reads the TARGET's hidden
    /// states at the five tapped layers for the rows this step just committed, and those live in
    /// the target's arena until the next step overwrites them. A drafter that ran between two
    /// engine calls would be reading whatever the step in between left there — and a prefill for
    /// another sequence is exactly such a step. So the proposal is part of the step that feeds it,
    /// and this is what that step hands back.
    ///
    /// The host-side [`crate::draft::Drafter`] seam still decides what a verify step is HANDED
    /// (`crate::draft::EngineDrafter` is the one that takes these); the pass never gets a vote on
    /// what is committed, because [`crate::draft::Greedy`] does.
    ///
    /// **Empty on the serving path since ws-f37**, which proposes AFTER the host sampler: see
    /// [`StepOutcome::proposal_pending`]. It is filled only under [`ProposeAt::InStep`], the old
    /// order kept as the pairing control, where the block is anchored on the device's ARGMAX.
    pub drafts: Vec<Vec<TokenId>>,
    /// The slots, in slot order, whose proposal this step STASHED rather than ran (ws-f37).
    ///
    /// Prod's order is sample, THEN propose: `execute_model` calls `speculator.propose` after its
    /// sampler, so a block is always the continuation of the token the sequence committed. Saber
    /// proposed inside the step from the ARGMAX, and a row with a temperature whose draw differed
    /// drafted a block for a text the sequence never had (ws-f35: position-0 conditional 0.683
    /// against prod's 0.800 at temperature 0.7). The proposal's inputs — `aux.cat`'s rows for
    /// this step and each slot's `(done_before, committed)` — are kept on the runner until the
    /// caller hands back the COMMITTED anchors through [`ForwardRunner::propose_after_commit`].
    ///
    /// **A caller that gets a non-empty list here MUST collect it** before its next call into the
    /// runner. One that does not is counted on the runner (`uncollected`) and logged at error
    /// level: the proposal is abandoned, which costs the next step its speculation and never a
    /// token.
    pub proposal_pending: Vec<SeqId>,
}

/// Where the pass's own drafter proposes (ws-f37).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProposeAt {
    /// **Prod's order, the default**: the step stashes the proposal's inputs and the caller runs
    /// it with the committed anchor ([`ForwardRunner::propose_after_commit`]), after the host
    /// sampler has decided every slot's last token.
    #[default]
    AfterCommit,
    /// The order this engine had before ws-f37, kept as the PAIRING CONTROL
    /// (`--propose-in-step`): the proposal runs inside the step, anchored on the device's argmax,
    /// and the executor drops a block whose anchor the sampler moved (ws-f35's law).
    InStep,
}

/// One slot's proposal, made after the commit, and the anchor it was made FROM.
///
/// The anchor is echoed from what the drafter actually consumed — on the device, the ids the
/// proposal's frame carries at each request's row 0 — so the executor can hold it against the
/// token the slot committed. After ws-f37 they are the same token by construction, and a
/// difference is a DEFECT, refused by name and counted (`EngineCounters::stale_anchor_defects`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    /// The slot's sequence, echoed so a caller cannot mis-pair a block with a stream.
    pub seq: SeqId,
    /// The token the block continues from, as the drafter was fed it.
    pub anchor: TokenId,
    /// The block: `block - 1` draft tokens, or empty when the drafter made none for this slot.
    pub block: Vec<TokenId>,
}

impl StepOutcome {
    /// The single token of a one-sequence, one-token step, if there is one.
    ///
    /// `None` rather than the first of many: a caller that wants one token and is handed a verify
    /// step's four should find out here and not three tokens later.
    #[must_use]
    pub fn token(&self) -> Option<TokenId> {
        match self.commits.as_slice() {
            [c] => match c.tokens.as_slice() {
                [t] => Some(*t),
                _ => None,
            },
            _ => None,
        }
    }

    /// Tokens committed across every slot.
    #[must_use]
    pub fn committed(&self) -> usize {
        self.commits.iter().map(|c| c.tokens.len()).sum()
    }

    /// Draft tokens accepted across every slot.
    #[must_use]
    pub fn accepted(&self) -> u32 {
        self.commits.iter().map(|c| c.accepted).sum()
    }
}

/// What the head must produce for a row **beyond the argmax**.
///
/// The argmax is the whole of greedy sampling and none of the rest, so a row that is going to be
/// sampled on the host needs the seam `saber_serve::sampler::DeviceRow` describes: the K largest
/// logits and ids, the row's log-sum-exp at T = 1 and at its own temperature, and the raw logits
/// at the ids the penalties can move. This is the request for it, per row.
///
/// [`RowRequest::greedy`] is the default and asks for nothing: the seam kernel's node is in the
/// captured graph either way and skips a row whose temperature is not positive, which is what
/// keeps a temperature-0 step the step it was.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowRequest {
    /// The row's temperature, when it is to be sampled. `None` is greedy.
    pub temperature: Option<f32>,
    /// The ids the penalties can move — the prompt's and the output's tokens. Empty when the row
    /// has no penalties, which is prod's own default.
    pub penalty_ids: Vec<TokenId>,
}

impl RowRequest {
    /// The greedy row: the argmax answers it and the seam kernel skips it.
    #[must_use]
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Whether this row wants the seam at all.
    #[must_use]
    pub fn wants_row(&self) -> bool {
        self.temperature.is_some_and(|t| t > 0.0)
    }
}

/// C9: a prompt chunk staged to ride in the next decode call ([`ForwardRunner::stage_mixed`]):
/// exactly the arguments of the [`ForwardRunner::prefill`] call it stands for.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedChunk {
    /// The prefilling sequence.
    pub seq: SeqId,
    /// The chunk's prompt tokens.
    pub tokens: Vec<TokenId>,
    /// Prompt tokens already resident before the chunk.
    pub done_before: u32,
    /// Whether this is the prompt's last chunk (its head samples the first token).
    pub last: bool,
    /// What the chunk's head must produce for its sampled row.
    pub sampling: RowRequest,
}

/// One live sequence in a decode step, as the engine is handed it.
///
/// Not `Copy`, because of `draft`: a verify step is the one step whose input is a *sequence* of
/// tokens per slot rather than one.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeSlot {
    /// The sequence.
    pub seq: SeqId,
    /// The token committed last step, which row 0 feeds.
    pub input: TokenId,
    /// Tokens already resident in this sequence's KV before the step.
    pub context_len: u32,
    /// The draft tokens rows `1..=k` feed. Empty at SPEC=0.
    ///
    /// Row 0 feeds [`Self::input`] and row `i` feeds `draft[i - 1]`, so the model's output at row
    /// `i` is its answer *given* `draft[..i]` — which is what makes the acceptance law a prefix
    /// comparison (`crate::draft`).
    pub draft: Vec<TokenId>,
    /// Tokens the previous step committed for this sequence, `1..=verify_width`.
    ///
    /// The one input that is a fact about the step BEFORE this one: `naccept - 1` picks which of
    /// this sequence's recurrent snapshots the step continues from (`crate::index`). 1 at SPEC=0
    /// and on a sequence's first decode step.
    pub committed_before: u32,
    /// What this slot's committed row needs beyond the argmax. [`RowRequest::greedy`] by default.
    pub sampling: RowRequest,
    /// Who decides this slot's verify rows when it is SAMPLED (ws-f40): `None` for a greedy slot,
    /// where the acceptance law compares the draft with the argmax exactly as before.
    ///
    /// With one, the pass walks the slot's rows through [`drawn_rows`] — each row's token is the
    /// host sampler's draw keyed by the output position that row decides, and a draft token is
    /// accepted iff it IS that draw — so the tokens a sampled slot commits are the tokens a width-1
    /// run commits, draw for draw, whatever the verify width, the batch or the drafter.
    pub decider: Option<Decider>,
}

/// Decides the token a SAMPLED slot's verify row commits (ws-f40).
///
/// The pass hands it one merged seam per row, in row order, and stops asking at the first row whose
/// token is not the draft at that row (that token is the recovery token and is committed), at a
/// row it cannot decide, or at a row whose token ends the request. Every row it answers is
/// therefore a committed token and is decided exactly once.
pub trait RowDecider: Send + Sync + std::fmt::Debug {
    /// The token row `j` commits if the walk reaches it — row `j` answers the input followed by
    /// `draft[..j]` — or `None` when this row cannot be decided from its seam (no seam, or the
    /// sampler refused it). A `None` ends the walk with the row committed on its ARGMAX as a
    /// placeholder, and the executor decides that last row densely.
    fn decide(&self, j: usize, seam: Option<&saber_serve::sampler::DeviceRow>) -> Option<TokenId>;

    /// **P27-W2-3**: the same decision for a row whose accepted prefix is `path` rather than
    /// the slot's chain draft (`draft[..j]`): a TREE's row at depth `j` (the sampled tree law,
    /// [`crate::spec_tree::TreePlan::drawn_verdict`]). The key is still `position + j`; only the
    /// history the penalties read changes. A decider with no history decides as [`Self::decide`].
    fn decide_path(
        &self,
        j: usize,
        path: &[TokenId],
        seam: Option<&saber_serve::sampler::DeviceRow>,
    ) -> Option<TokenId> {
        let _ = path;
        self.decide(j, seam)
    }

    /// Whether the token row `j` decided is the request's LAST (a stop id, or its `max_tokens`):
    /// no later row may then be drawn, because nothing after it is served.
    fn ends_at(&self, j: usize, token: TokenId) -> bool {
        let _ = (j, token);
        false
    }
}

/// A shared [`RowDecider`], compared by identity so [`DecodeSlot`] keeps its `PartialEq`.
#[derive(Clone)]
pub struct Decider(pub std::sync::Arc<dyn RowDecider>);

impl std::fmt::Debug for Decider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl PartialEq for Decider {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// THE DRAWN ACCEPTANCE LAW's walk over one sampled slot's verify rows (ws-f40).
///
/// Row `j`'s target is `decider`'s draw for it; the walk continues past row `j` only while that
/// draw IS `draft[j]` (the draft is accepted because the draw chose it) and the request has not
/// ended. Returns the targets of the rows it reached: `Greedy.accept(&draft[..n - 1], &targets)`
/// with `n = targets.len()` then commits exactly `targets` — the drafts it matched and the last
/// row's own token — and accepts `n - 1`.
///
/// An undecided row ends the walk on its ARGMAX (`argmax[j]`), which the executor replaces by a
/// dense draw of that row under the same key. At temperature 0 nothing calls this: a greedy slot
/// has no decider and the law is the argmax comparison, byte for byte what it was.
///
/// Why this is prod's distribution: the draw at each position is a sample from that row's
/// processed distribution (the exponential race, `argmax(p / Exp(1))`), so a draft is accepted
/// with probability `p(draft)` — prod's rejection sampler's rate for a greedy drafter
/// (`target_prob / 1 >= u`) — and a rejected row commits a sample from `p` conditioned on not the
/// draft, which is prod's recovered token (`p` with the draft's mass removed). Same law in
/// distribution; unlike prod's, a function of `(seed, position, choice)` and the row alone.
pub fn drawn_rows(
    decider: &dyn RowDecider,
    draft: &[TokenId],
    argmax: &[TokenId],
    seam: &mut dyn FnMut(usize) -> Option<saber_serve::sampler::DeviceRow>,
) -> Vec<TokenId> {
    let rows = (draft.len() + 1).min(argmax.len());
    let mut targets = Vec::with_capacity(rows);
    for j in 0..rows {
        let row = seam(j);
        match decider.decide(j, row.as_ref()) {
            None => {
                targets.push(argmax[j]);
                break;
            }
            Some(t) => {
                targets.push(t);
                if j == draft.len() || t != draft[j] || decider.ends_at(j, t) {
                    break;
                }
            }
        }
    }
    targets
}

impl DecodeSlot {
    /// The SPEC=0 shape: one row, no draft, one token committed last step.
    #[must_use]
    pub fn single(seq: SeqId, input: TokenId, context_len: u32) -> Self {
        DecodeSlot {
            seq,
            input,
            context_len,
            draft: Vec::new(),
            committed_before: 1,
            sampling: RowRequest::greedy(),
            decider: None,
        }
    }

    /// Rows this slot feeds: the draft plus the token it is continuing from.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.draft.len() as u32 + 1
    }

    /// The ids this slot's rows gather, in row order.
    #[must_use]
    pub fn row_ids(&self) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(self.rows() as usize);
        out.push(self.input);
        out.extend_from_slice(&self.draft);
        out
    }
}

/// What one slot's step committed.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotCommit {
    /// The sequence, echoed so a caller zipping slots to commits cannot mis-pair them silently.
    pub seq: SeqId,
    /// Draft tokens accepted, **excluding** the bonus token. 0 at SPEC=0.
    pub accepted: u32,
    /// The tokens committed, in order: always `accepted + 1` of them. **The argmax's**, always:
    /// the device computes them and the host's sampler replaces the LAST one when the row asked
    /// to be sampled, so a greedy row and a sampled row take the same path to here.
    pub tokens: Vec<TokenId>,
    /// One seam per COMMITTED token, in the same order as `tokens`, when
    /// [`RowRequest::wants_row`] asked for it. Empty on a greedy row, and empty from a runner that
    /// has no seam kernel, which the executor counts rather than samples around.
    ///
    /// It is a LIST and not one row because the seam kernel runs on every verify row and a step
    /// that commits an accepted prefix has a distribution for each of those tokens too — the
    /// model's answer at that position, which is the number a client asking for logprobs wants.
    /// The LAST entry is the one the host sampler draws from; the ones before it are rows whose
    /// token the acceptance law already chose, and they are reported, not re-drawn
    /// (`logprobs_skipped_on_verify` was what this replaced).
    ///
    /// An entry may be `None` while others are `Some`: the kernel skips a greedy row, and a row
    /// the merge refused is a fault the executor counts rather than a distribution.
    pub rows: Vec<Option<saber_serve::sampler::DeviceRow>>,
}

impl SlotCommit {
    /// The seam for the row that produced the LAST committed token — the recovery or bonus token,
    /// the one this step actually chose and the only one the sampler draws.
    #[must_use]
    pub fn last_row(&self) -> Option<&saber_serve::sampler::DeviceRow> {
        self.rows.last().and_then(Option::as_ref)
    }
}

/// One prompt row, scored for `prompt_logprobs` (V7): the row's seam and the target's counted rank.
///
/// The seam is the one the sampler reads — the K largest raw logits, the row's log-sum-exp at
/// T = 1, and the raw logit at the TARGET (the prompt token at the next position), which the
/// engine names in the penalty gather — merged across the vocabulary shards at TP = 2 by the
/// same law as a sampled row (`saber_serve::sampler::merge_shards`). `rank` is the reference server's
/// `(logprobs >= logprobs[target]).sum()`: from the top-K when it decides it, from a dense read of
/// the row when it does not (`saber_serve::prompt_logprobs::rank_from_seam`).
#[derive(Clone, Debug, PartialEq)]
pub struct PromptRow {
    /// The prompt token this row predicts.
    pub target: TokenId,
    /// The row's merged seam.
    pub row: saber_serve::sampler::DeviceRow,
    /// The target's counted rank over the whole vocabulary.
    pub rank: u32,
}

/// What a pass declares about itself, so the engine can assert rather than hope.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardFacts {
    /// Model name, for the log line.
    pub model: String,
    /// Tensor-parallel rank count the pack is sharded for: 1 or 2.
    pub tp: u32,
    /// This process's rank within that tier.
    pub rank: u32,
    /// All-reduces **inside** one step's captured graph, per rank. 0 at TP = 1, where the graph has
    /// no collective at all; 129 at TP = 2 — the embedding's and one per layer that ends in a
    /// row-parallel GEMM.
    pub collectives: u32,
    /// Peer-visible (IPC) bytes this rank's launcher holds for those collectives.
    pub ipc_bytes: u64,
    /// Decoder layers.
    pub layers: u32,
    /// `full_attention_interval`: every `n`-th layer is full attention and the rest are gated
    /// deltanet, so layer `l` is an attention layer exactly when `(l + 1) % n == 0`.
    ///
    /// Here because a caller that has to say WHICH KIND of layer a dump point belongs to should ask
    /// the engine rather than restate the model's own constant — and one that restates it is a
    /// second place for it to be wrong (ws-l17's two-bucket probe is the caller).
    pub attention_interval: u32,
    /// Vocabulary, as the head's GEMM declares it.
    pub vocab: u32,
    /// Weight records bound by name out of the pack.
    pub weights: usize,
    /// Ops with no code object on this branch, by op id and count. Non-empty means the tokens are
    /// **not** the model's tokens, and every report has to say so.
    pub supplied: Vec<(String, u32)>,
    /// Bytes of every device buffer the engine owns: arenas, session state, index tables, weights.
    pub device_bytes: u64,
    /// KV blocks in the engine's pool. A block id at or past this is refused at admission.
    pub kv_pool_blocks: u32,
    /// Block-table width the plans were built with: `max_ctx / block_size`, rounded up.
    pub max_blocks: u32,
    /// Tokens one sequence may reach, as the plans were built for it.
    pub max_ctx: u32,
    /// Sequences the engine sized its GDN state for.
    pub max_seqs: u32,
    /// Bytes of one KV block across every attention layer, as the engine allocated it. The
    /// scheduler's `KvGeometry` is built from this so the two do not each guess a different size for
    /// the same pages.
    pub kv_block_bytes: u64,
    /// Bytes one KV block occupies in the **arena**: [`ForwardFacts::kv_block_bytes`] summed over
    /// every attention layer, which is the stride the pool is actually indexed by.
    ///
    /// The two differ by the attention-layer count (16 here, so 32 KiB against 512 KiB) and reading
    /// the wrong one is not cosmetic: ws-k13 measured the scheduler reserving **472 MiB** for a pool
    /// the engine had allocated **7563 MiB** of, because the CLI built its `KvGeometry` from the
    /// per-layer figure. A byte ledger that under-counts by 16x cannot refuse when it should.
    pub kv_block_stride: u64,
    /// Bytes of one GDN state slot across every GDN layer, as the engine allocated it.
    pub gdn_bytes_per_slot: u64,
    /// Decode shapes the engine planned, as **(batch, verify width)** pairs.
    ///
    /// Separate from `declared`, whose key carries the total row count: `b1s5` and `b5s1` both run
    /// five rows and are not the same graph, so a caller that has to rebuild the scheduler's buckets
    /// cannot recover the pair from the rows. The scheduler needs the pair — its bucket IS
    /// (batch, spec_width) — and a bucket built as `(rows, 1)` would tell it every shape is SPEC=0.
    pub decode_shapes: Vec<(u32, u32)>,
    /// The widest verify width planned. Each sequence owns this many consecutive recurrent
    /// snapshots, which is what the scheduler's per-sequence state reservation has to agree with.
    pub max_width: u32,
    /// **The decode ROW BANDS planned (workstream C2)**, as `(batch, max width, rows)`: empty
    /// under `--rectangular-widths`. The scheduler's band buckets are built from these, exactly as
    /// its rectangles are built from [`ForwardFacts::decode_shapes`].
    pub decode_bands: Vec<(u32, u32, u32)>,
    /// Shapes captured as a graph at startup, with the node count each produced.
    pub captured: Vec<(StepRows, usize)>,
    /// Descriptors whose declared launch count the engine had to correct before it could launch
    /// them.
    ///
    /// **This reads 0 since ws-h15.** It used to count the GDN decode conv route, whose descriptor
    /// declared the pair's two dispatches while the route issued only the conv; `saber-exec` now
    /// declares what it dispatches, so nothing is corrected here. The field stays because a
    /// stopgap that vanished silently would be indistinguishable from the bug being fixed, and a
    /// number someone can read is the difference.
    pub corrected_declarations: u32,
    /// Shapes the engine planned, with the launch budget declared for each.
    pub declared: Vec<(StepRows, u32)>,
    /// Page-locked host bytes this runner holds for session parking — one image's worst case.
    ///
    /// 0 means it cannot park. Non-zero is a number the caller has **billed**: the bytes are
    /// reserved in the byte ledger under `Pool::HostPinned` before the runner is built, so "not
    /// enough pinned memory" is a refusal to park rather than an allocation nobody counted.
    pub park_staging_bytes: u64,
    /// Staging slots the park buffer is divided into: **1** (a park and a restore take turns) or
    /// **2** (they overlap, at twice the pinned cost). 0 when this runner cannot park.
    pub park_staging_slots: u32,
    /// The all-reduce family each planned shape RESOLVED to, as `(shape, symbol, dispatches)`.
    ///
    /// Empty at TP = 1, where a step has no collective at all. At TP = 2 it is the answer to a
    /// question every identity arm above a verify width of 2 has to ask: a decode step's message is
    /// `rows x hidden x 2` and `quant_min_bytes` is 131,072, so `b4s5` (204,800 B) resolves to the
    /// LOSSY `allreduce_wht` family while the same sequence stepped alone keeps the exact one — and
    /// two arms of the same gate then compare two different arithmetics and report a divergence
    /// (ws-l17's open item, `docs/GATES.md`). Carried rather than inferred by the caller, because
    /// the band is `saber-kernels`' data and a gate that restated it would be a second place for it
    /// to be wrong.
    pub ar_families: Vec<(StepRows, String, u32)>,
    /// Whether this run raised the exact/lossy band above every message it can send
    /// ([`crate::EngineConfig::ar_exact`]), so that every shape resolves to the exact family.
    pub ar_exact: bool,
    /// **What each planned shape RESOLVED TO, op by op** — the plan facts two buckets can be
    /// held against (ws-p9).
    ///
    /// [`ForwardFacts::ar_families`] answers the same question for the collective and nothing
    /// else, and the answer for a TP=1 batch divergence is not there: at `b1s8` against `b4s8`
    /// the decode attention keeps its symbol and its split count (ws-l15 pinned that), and what
    /// moves is the row-parallel GEMMs' **split-K** (`ks4` at M=8 against `ks2` at M=32, a
    /// different partition of K and therefore a different sum), their M tile and the GDN
    /// recurrent update's value split. An arm that reports a divergence without these is
    /// reporting a hypothesis.
    ///
    /// One entry per (shape, generic op name, fingerprint): the op name with its layer number
    /// replaced by `N`, so all 64 layers' `mlp.down_proj` collapse to one row, and the count of
    /// descriptors that resolved to it.
    pub bucket_ops: Vec<(StepRows, String, String, u32)>,
    /// The DFlash2 drafter tier this engine planned, as its own accounting line, or `None` when the
    /// run has no drafter tier.
    pub draft_tier: Option<String>,
    /// The drafter as it is RESIDENT — weights bound, caches allocated, arena laid out, both
    /// phases planned and the backbone captured — or `None` when the tier is billed and nothing
    /// was uploaded (`--draft-block` without `--draft-pack`).
    pub draft_resident: Option<DraftResidency>,
    /// The aux tap: its copies per decode step, the layers it reads and what `aux.cat` holds.
    pub aux_tap: Option<AuxTapFacts>,
}

/// What the drafter's residency came to, in numbers a gate can assert rather than a sentence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftResidency {
    /// Records bound out of the drafter's pack.
    pub records: u32,
    /// Bytes of the proposal's one activation arena.
    pub arena_bytes: u64,
    /// Regions laid out in it.
    pub arena_regions: u32,
    /// Launches of the EAGER context-store phase.
    pub eager_launches: u32,
    /// Launches of the CAPTURED backbone phase, and therefore its node count.
    pub captured_launches: u32,
    /// Nodes the capture actually took. Equal to `captured_launches` or the boot refused.
    pub captured_nodes: u32,
    /// Host round trips inside the captured phase. Zero, or the phase cannot be a graph.
    pub host_round_trips: u32,
    /// Drafter KV blocks the tier was sized for.
    pub kv_blocks: u32,
    /// Bytes of EACH of the two contiguous drafter KV caches.
    pub kv_cache_bytes: u64,
    /// Bytes the drafter's workspaces took, sized once before the first capture.
    pub workspace_bytes: u64,
    /// **The planned proposal SHAPES**, as `(requests, context rows per request)` pairs (ws-s10).
    ///
    /// One captured backbone graph each, exactly as the target has one per decode bucket: the
    /// first is the shape a PROMPT's context store and a first proposal run at (one request,
    /// `block` rows) and the rest are the engine's `(decode batch, verify width)` pairs.
    pub shapes: Vec<(u32, u32)>,
    /// Captured backbone nodes of the WIDEST planned shape.
    pub widest_captured_nodes: u32,
    /// Bytes of one `engine.draft` frame: the widest shape's, which is the one width on the wire.
    pub frame_bytes: u32,
}

/// The aux tap, as facts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuxTapFacts {
    /// The target layers whose `l{id}.resid.a` the taps read.
    pub layers: Vec<u32>,
    /// Copies issued per decode step.
    pub launches: u32,
    /// Rows `aux.cat` was sized for: the widest decode shape.
    pub rows: u32,
    /// Bytes of `aux.cat`.
    pub bytes: u64,
}

impl ForwardFacts {
    /// Whether every op this pass needs has a code object.
    pub fn complete(&self) -> bool {
        self.supplied.is_empty()
    }

    /// The MODEL's vocabulary, which is not this rank's.
    ///
    /// [`ForwardFacts::vocab`] is what `lm_head` produces HERE — the whole thing at TP = 1 and a
    /// **shard** at TP = 2, because the head is column-parallel. Anything that resolves a
    /// request's parameters reads this one instead: `top_k` outside `1..vocab` is the reference server's disabled
    /// form, and a `top_k` of 124,160 resolved against half a vocabulary would silently become
    /// "no truncation at all" — which is the one request shape a top-K seam cannot serve.
    ///
    /// Derived rather than stored, because the same number written down twice is the same number
    /// wrong twice.
    #[must_use]
    pub fn vocab_total(&self) -> u32 {
        self.vocab * self.tp.max(1)
    }

    /// The all-reduce family a shape resolved to, for a report that has to say which arithmetic
    /// an arm ran. `None` at TP = 1 and for a shape this run did not plan.
    #[must_use]
    pub fn ar_family(&self, rows: StepRows) -> Option<String> {
        let mut named: Vec<String> = self
            .ar_families
            .iter()
            .filter(|(s, _, _)| *s == rows)
            .map(|(_, sym, n)| format!("{sym} x{n}"))
            .collect();
        if named.is_empty() {
            return None;
        }
        named.sort();
        Some(named.join(", "))
    }

    /// **Every op whose resolved launch differs between two planned shapes** (ws-p9).
    ///
    /// This is the plan-fact half of a divergence report. An identity arm that finds sequence 3's
    /// ids moving between a batch-1 run and a batch-4 one can say one of two things: *two
    /// arithmetics were compared* or *a law is being read wrongly in one of them* — and the only
    /// thing that separates them is which ops the two buckets resolved differently. It is read off
    /// the engine's OWN plans ([`ForwardFacts::bucket_ops`]), never re-derived from the rules,
    /// because a gate that restated a rule would be a second place for it to be wrong.
    ///
    /// Returns `(generic op, fingerprint in `a`, fingerprint in `b`, descriptors)`, sorted by op.
    /// An op only one side has is reported with `—` for the other.
    #[must_use]
    pub fn bucket_op_deltas(&self, a: StepRows, b: StepRows) -> Vec<(String, String, String, u32)> {
        let side = |s: StepRows| -> std::collections::BTreeMap<String, (String, u32)> {
            let mut m: std::collections::BTreeMap<String, (String, u32)> =
                std::collections::BTreeMap::new();
            for (shape, op, fp, n) in &self.bucket_ops {
                if *shape != s {
                    continue;
                }
                let e = m.entry(op.clone()).or_insert_with(|| (String::new(), 0));
                if !e.0.is_empty() {
                    e.0.push_str(" + ");
                }
                e.0.push_str(fp);
                e.1 += n;
            }
            m
        };
        let (ma, mb) = (side(a), side(b));
        let mut out = Vec::new();
        for op in ma
            .keys()
            .chain(mb.keys())
            .collect::<std::collections::BTreeSet<_>>()
        {
            let x = ma.get(op);
            let y = mb.get(op);
            let (fa, fb) = (
                x.map_or("—", |v| v.0.as_str()),
                y.map_or("—", |v| v.0.as_str()),
            );
            if fa == fb {
                continue;
            }
            out.push((
                op.clone(),
                fa.to_owned(),
                fb.to_owned(),
                x.map_or(0, |v| v.1).max(y.map_or(0, |v| v.1)),
            ));
        }
        out
    }

    /// The two shapes' resolved launches for ONE op, when they differ.
    ///
    /// The sharp form of [`ForwardFacts::bucket_op_deltas`], asked at the op a probe named: an op
    /// the two buckets resolved IDENTICALLY writing different bytes is not a bucket-arithmetic
    /// difference at all — it is the same launch reading something else, which is a defect —
    /// while one they resolved differently is two arithmetics being compared. `op` is the op's own
    /// name, layer number and all; the deltas are keyed on the generic name, and that mapping is
    /// here rather than at the call site so the two cannot drift.
    #[must_use]
    pub fn bucket_delta_for_op(
        &self,
        a: StepRows,
        b: StepRows,
        op: &str,
    ) -> Option<(String, String)> {
        let generic: String = op
            .split('.')
            .map(|p| if p.parse::<u32>().is_ok() { "N" } else { p })
            .collect::<Vec<_>>()
            .join(".");
        self.bucket_op_deltas(a, b)
            .into_iter()
            .find(|(name, _, _, _)| *name == generic)
            .map(|(_, x, y, _)| (x, y))
    }

    /// The ops of ONE layer whose resolved launch differs between two shapes.
    ///
    /// The same question as [`ForwardFacts::bucket_op_deltas`] asked where a probe says the two
    /// buckets first parted: an op named `layer.<n>.…` belongs to layer `n`, and the deltas are
    /// stored under the generic name (`layer.N.…`), so this filters the model's site set by the
    /// layer's KIND rather than by its number — which is the only thing a generic name carries.
    /// `attention` selects the sites a full-attention layer runs and `gdn` the rest; every
    /// non-layer op (the embedding, the head) belongs to neither.
    #[must_use]
    pub fn layer_kind_of(&self, layer: i32) -> &'static str {
        if layer < 0 {
            return "embedding";
        }
        let interval = self.attention_interval.max(1);
        if (layer as u32 + 1) % interval == 0 {
            "attention"
        } else {
            "gdn"
        }
    }

    /// The one-line PARTIAL banner, or `None` when the pass is complete.
    pub fn partial_banner(&self) -> Option<String> {
        if self.complete() {
            return None;
        }
        let ops: Vec<String> = self
            .supplied
            .iter()
            .map(|(op, n)| format!("{op}x{n}"))
            .collect();
        Some(format!(
            "PARTIAL: {} op(s) have no code object on this branch ({}), so the attention layers do \
             not compute their value and these tokens are NOT the model's tokens",
            self.supplied.len(),
            ops.join(" ")
        ))
    }
}

/// What [`ForwardRunner::discard_parked`] found and freed (ws-k18).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParkDiscard {
    /// Receipts this rank held for the image's half on the OTHER rank, dropped.
    pub receipts: u32,
    /// Whether the other rank held a half under that name when it was told to drop it.
    pub peer_held: bool,
    /// Halves the other rank STILL holds after the drop, as it answered — the number the
    /// abort-while-parked card arm requires to be back at zero.
    pub peer_remaining: u64,
    /// Receipts this rank still holds after the drop.
    pub receipts_remaining: u32,
}

impl ParkDiscard {
    /// Whether anything was held for the image at all.
    #[must_use]
    pub fn held(&self) -> bool {
        self.receipts > 0 || self.peer_held
    }
}

/// A sequence's image facts (D4): where its tokens sit for the rope, and which tower run holds
/// the rows its prefill scatters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqImages {
    /// The M-RoPE table of the prompt, the delta past it, and the image runs.
    pub rope: std::sync::Arc<saber_exec::mrope::MropePositions>,
    /// The tower run (`VisionRows::epoch`) that left this request's rows in the arena. The
    /// prefill refuses a chunk with image rows once a later run has overwritten them.
    pub epoch: u64,
}

/// A whole-model forward pass, as the engine drives it.
///
/// Implementors: [`crate::NullForward`] on the host, [`crate::device::HipForward`] on a card. Every
/// method is called from the scheduler's step thread and may block; none of them may panic on a
/// device error, because the step loop must survive one (`CLAUDE.md` rule 5).
pub trait ForwardRunner: Send + std::fmt::Debug + 'static {
    /// What this pass is, for the log and for the report's banner.
    fn facts(&self) -> &ForwardFacts;

    /// D3: the largest image (in patches) this runner's vision tower holds; 0 = text-only.
    fn vision_max_patches(&self) -> u32 {
        0
    }

    /// D3/D4: run the vision tower over ONE REQUEST's images (`image_grid_thw` each,
    /// `[patches, 1536]` f32 concatenated). The rows stay in the tower's arena, one image after
    /// another, until the NEXT call (`VisionRows::epoch` names this run). The default is the
    /// text-only refusal, by name.
    fn run_vision(
        &mut self,
        grids: &[[u32; 3]],
        pixels: Vec<f32>,
    ) -> Result<crate::device::VisionRows, ForwardError> {
        let _ = (grids, pixels);
        crate::device::check_vision_enabled(0)?;
        Err(ForwardError::Device("vision: unreachable".to_owned()))
    }

    /// D4: give an admitted sequence its image facts -- the M-RoPE table and delta its rope
    /// positions come from, and the tower run whose rows its prefill scatters. Called after
    /// [`ForwardRunner::admit`] (a first admission and every restore); `None` is a text-only
    /// sequence, which is what an admission leaves. The default refuses `Some` by name: a runner
    /// with no tower cannot place an image's rows.
    ///
    /// # Errors
    /// A runner that cannot serve images, or a sequence it never admitted.
    fn set_images(&mut self, seq: SeqId, images: Option<SeqImages>) -> Result<(), ForwardError> {
        let _ = seq;
        match images {
            None => Ok(()),
            Some(_) => crate::device::check_vision_enabled(0),
        }
    }

    /// Give a sequence its session state: a GDN state slot, zeroed, and the KV pages the scheduler
    /// allocated it.
    fn admit(&mut self, seq: SeqId, prompt_len: u32, kv_blocks: &[u32])
        -> Result<(), ForwardError>;

    /// Drop a sequence's session state. Its GDN slot goes back to the engine's free list.
    fn release(&mut self, seq: SeqId);

    /// The same for a sequence that was just PARKED (ws-k17): its pages and slot go back, and any
    /// half of its image held on ANOTHER rank is kept for the restore. A plain
    /// [`ForwardRunner::release`] at TP > 1 tells the follower to drop that half, which is right
    /// for a sequence that is finished and wrong for one whose image is on the scheduler's tier.
    /// The default is a plain release: a one-rank runner keeps nothing outside the image.
    fn release_parked(&mut self, seq: SeqId) {
        self.release(seq);
    }

    /// The image parked under `seq` was dropped WITHOUT a restore (ws-k18): release whatever this
    /// runner kept for it. At TP=2 that is the follower's half — kept by
    /// [`ForwardRunner::release_parked`] for a restore that is now never coming — and the leader's
    /// receipt for it. The default keeps nothing outside the image and holds nothing.
    fn discard_parked(&mut self, _seq: SeqId) -> ParkDiscard {
        ParkDiscard::default()
    }

    /// Give a BRANCH of `leader` its session state, by copying the leader's.
    ///
    /// Called instead of [`ForwardRunner::admit`] for branch `i > 0` of an `n > 1` request, at
    /// the step the leader's last prefill chunk sampled. Three things happen and all three are
    /// device-to-device:
    ///
    /// * the branch takes a state slot of its own and the leader's whole slot SET is copied into
    ///   it — recurrent state is not addressable per token, so it cannot be shared the way a page
    ///   can;
    /// * the leader's one PARTIAL KV page (`cow_source`) is copied into the branch's first
    ///   private block, because it is the one page both would write into;
    /// * the branch's residency (`done`) starts at the leader's, because the prompt is already in
    ///   the pages it was given.
    ///
    /// The leading `shared_prefix` pages of `kv_blocks` are the leader's own and are shared
    /// **read-only for ever**: every branch appends only into pages it owns, so there is no
    /// copy-on-write after this call.
    ///
    /// The default is a refusal by name, which is what makes `n > 1` a 400 rather than a wrong
    /// answer on a runner that cannot do it.
    fn fork(
        &mut self,
        leader: SeqId,
        branch: SeqId,
        kv_blocks: &[u32],
        shared_prefix: u32,
        cow_source: Option<u32>,
    ) -> Result<(), ForwardError> {
        let _ = (leader, branch, kv_blocks, shared_prefix, cow_source);
        Err(ForwardError::CannotFork)
    }

    /// Whether this runner can branch a session at all — asked once, at construction, so that
    /// `n > 1` is refused at submit rather than at the fork.
    fn can_fork(&self) -> bool {
        false
    }

    /// FN52: whether this runner snapshots and restores prefix state (`saber_sched::prefix_cache`).
    fn prefix_cache_supported(&self) -> bool {
        false
    }

    /// FN52: snapshot `seq`'s GDN live state and its drafter pages over `[0, tokens)` (its prefill
    /// is exactly at `tokens`) into host memory under `entry`, on every rank. Returns this rank's
    /// bytes. The KV pages are retained in place by the scheduler and are not copied.
    ///
    /// # Errors
    /// A runner without the cache, or a session not at `tokens`.
    fn prefix_save(&mut self, seq: SeqId, entry: u32, tokens: u32) -> Result<u64, ForwardError> {
        let _ = (seq, entry, tokens);
        Err(ForwardError::Device(
            "this pass has no prefix cache".to_owned(),
        ))
    }

    /// FN52: restore `entry` into the freshly admitted `seq` (its first pages are the entry's,
    /// adopted), so its prefill resumes at `tokens`.
    ///
    /// # Errors
    /// A runner without the cache, an unknown entry, a session that has already stepped.
    fn prefix_load(&mut self, seq: SeqId, entry: u32, tokens: u32) -> Result<(), ForwardError> {
        let _ = (seq, entry, tokens);
        Err(ForwardError::Device(
            "this pass has no prefix cache".to_owned(),
        ))
    }

    /// FN52: drop `entry`'s snapshot on every rank.
    fn prefix_drop(&mut self, entry: u32) {
        let _ = entry;
    }

    /// FN52, the verify arm: compare `seq`'s state, drafter pages and first `tokens / 16` KV
    /// pages with `entry`'s snapshot and `entry_blocks` (rank 0's half). `Ok(true)` = identical.
    ///
    /// # Errors
    /// A runner without the cache.
    fn prefix_verify(
        &mut self,
        seq: SeqId,
        entry: u32,
        tokens: u32,
        entry_blocks: &[u32],
    ) -> Result<bool, ForwardError> {
        let _ = (seq, entry, tokens, entry_blocks);
        Err(ForwardError::Device(
            "this pass has no prefix cache".to_owned(),
        ))
    }

    /// **A forked branch's FIRST BLOCK, from the token the branch committed** (ws-f38).
    ///
    /// Called by the executor once per branch, right after [`ForwardRunner::fork`] and the
    /// branch's draw, and before anything else reaches the pass. The block is the leader's prompt
    /// proposal re-run over the same context rows into the BRANCH's drafter session, anchored on
    /// `anchor` — the branch's own first token — and it comes back with the anchor its drafter was
    /// FED (F37's echo), which the executor checks against the token exactly as it checks every
    /// block made after a commit.
    ///
    /// `Ok(None)` is a runner with no drafter, which proposes nothing for anyone — the default.
    ///
    /// # Errors
    /// A runner with a drafter that cannot make the block names why.
    fn propose_fork(
        &mut self,
        leader: SeqId,
        branch: SeqId,
        anchor: TokenId,
    ) -> Result<Option<Proposal>, ForwardError> {
        let _ = (leader, branch, anchor);
        Ok(None)
    }

    /// Feed one prompt chunk. Samples **only** on the last chunk: a chunk that is not the last
    /// returns no token at all, which is why a client hears nothing until the prompt is in. Prefill
    /// is one sequence per step (`Bucket::prefill` is batch 1), so the outcome carries at most one.
    fn prefill(
        &mut self,
        seq: SeqId,
        chunk: &[TokenId],
        tokens_done_before: u32,
        last: bool,
        sampling: &RowRequest,
    ) -> Result<StepOutcome, ForwardError>;

    /// One decode step for **every live slot of the batch at once**.
    ///
    /// `batch` is the bucket's batch and `width` its verify width — together they are the shape to
    /// run, `batch * width` rows — and `slots` are the live sequences, in slot order, at most
    /// `batch` of them. Every live slot must ask for the same width, because the bucket has one:
    /// a slot with fewer draft tokens than the bucket verifies is a slot the scheduler should have
    /// put in a narrower bucket, and the engine refuses it rather than padding a sequence's own
    /// rows (which would write KV at positions it then has to un-write).
    ///
    /// The rows past `slots.len() * width` are padding: the pass runs them because a captured graph
    /// has a fixed `M`, and the index tables point them at the null state slot and a zero context so
    /// they touch no live session.
    ///
    /// Returns one [`SlotCommit`] per live slot, in slot order.
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError>;

    /// **One decode step in a ROW BAND (workstream C2)**: every live slot at its OWN verify width
    /// ([`DecodeSlot::rows`], `1..=max_width`), the rows packed back to back by the step's `cu`
    /// prefix, the step run at the captured band of `rows` rows over `batch` slots.
    ///
    /// No slot's own rows are padded: a slot that verifies 3 feeds 3 rows and writes 3 K/V
    /// positions. The band's padding is its tail — rows past `Σ w_i`, which the index tables point
    /// at nothing (`slot` −1, position 0, outside every `cu` span) — and the slots past the live
    /// ones (zero rows, the null state slot, `seqused_k` 0).
    ///
    /// Returns one [`SlotCommit`] per live slot, in slot order; `StepOutcome::rows` is the band's
    /// [`StepRows`] (`ragged: true`). The default is the refusal: a runner that planned no bands.
    ///
    /// # Errors
    /// A runner with no band of `rows`, a slot wider than `max_width` or of no rows, or more rows
    /// than the band holds.
    fn decode_band(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        max_width: u32,
        rows: u32,
    ) -> Result<StepOutcome, ForwardError> {
        let _ = (slots, batch, max_width);
        Err(ForwardError::NoPlan {
            phase: Phase::Decode,
            rows,
            detail: "this runner planned no decode row bands (--rectangular-widths, or a runner \
                     that cannot run a ragged step)"
                .to_owned(),
        })
    }

    /// The engine's own drafter's counters, snapshotted (ws-v8: the `drafted` census line and its
    /// `saber:draft_*` twins on `/metrics`). `None` — the default — is a runner that keeps none;
    /// the census then reads zeros, as the line does.
    fn draft_counts(&self) -> Option<crate::device::DraftCounts> {
        None
    }

    /// The per-step host-time tables so far, drained (C5: `saber serve --engine-host-trace`, a
    /// DIAGNOSTIC arm, never a published one). `None` — the default — is a runner with no trace,
    /// or one whose trace is off.
    fn host_trace_table(&self, _label: &str) -> Option<String> {
        None
    }

    /// Whether this runner can move a session's state off the device and back.
    ///
    /// `false` is the honest default: a runner that cannot park makes the scheduler refuse and
    /// count (`ParkError::ExecutorCannotPark`), which is a loud no rather than an eviction whose
    /// state cannot be brought back.
    fn can_park(&self) -> bool {
        false
    }

    /// Copy a sequence's KV pages and GDN state slot out as one image.
    ///
    /// The sequence keeps its pages and its slot until the caller releases it: this reads, it does
    /// not free. A caller parks by calling this and *then* releasing, so a failure here leaves the
    /// sequence exactly as it was.
    fn park(&mut self, seq: SeqId) -> Result<crate::park::ParkImage, ForwardError> {
        let _ = seq;
        Err(ForwardError::CannotPark)
    }

    /// Put an image back into a **freshly admitted** sequence's pages and slot.
    ///
    /// The destinations are whatever that admission was given, which is the point — they are not
    /// the ones the image was taken from. The shape must match (same regions, same kinds, same
    /// lengths) and every chunk is verified before it is issued to the device.
    fn restore(&mut self, seq: SeqId, image: &crate::park::ParkImage) -> Result<(), ForwardError> {
        let _ = (seq, image);
        Err(ForwardError::CannotPark)
    }

    /// A park, as the wire blob [`saber_sched::SeqState`] carries.
    ///
    /// **This is the seam the scheduler uses**, and it exists for one reason: cost. Going through
    /// [`ForwardRunner::park`] means the payload is copied out of the staging buffer into
    /// `ParkImage::bytes` and then again into the blob — two full-image host memcpys per park, 806
    /// MiB of them at prod's 403 MiB image. A runner that holds the bytes already can build the
    /// blob in one copy, which is the least possible while `SeqState` owns what it carries.
    fn park_encoded(&mut self, seq: SeqId) -> Result<Vec<u8>, ForwardError> {
        self.park(seq).map(|i| i.encode())
    }

    /// The mirror: publish straight out of the blob, without decoding it into an owned image first.
    fn restore_encoded(&mut self, seq: SeqId, blob: Vec<u8>) -> Result<(), ForwardError> {
        match crate::park::ParkImage::decode(&blob) {
            Ok(image) => self.restore(seq, &image),
            Err(e) => Err(ForwardError::rejected(e.to_string(), blob)),
        }
    }

    /// Whether this runner moves session state **off the step loop**.
    ///
    /// `false` means the caller must use the blocking [`ForwardRunner::park_encoded`]. `true`
    /// promises the pair below: an issue that returns immediately, and a poll that never waits.
    fn parks_async(&self) -> bool {
        false
    }

    /// Issue a park's device-to-host copy and return. The sequence keeps its pages and its slot
    /// until the poll says the copy is done, because the copy is still reading them.
    fn park_begin(&mut self, seq: SeqId) -> Result<(), ForwardError> {
        let _ = seq;
        Err(ForwardError::CannotPark)
    }

    /// Advance an in-flight park by one bounded step. `Ok(None)` means "not finished".
    fn park_poll(&mut self, seq: SeqId) -> Result<Option<Vec<u8>>, ForwardError> {
        let _ = seq;
        Err(ForwardError::CannotPark)
    }

    /// Verify an image and issue its host-to-device copy, then return. Takes the blob by value and
    /// hands it back in [`ForwardError::Rejected`] when it cannot start.
    fn restore_begin(&mut self, seq: SeqId, blob: Vec<u8>) -> Result<(), ForwardError> {
        let _ = seq;
        Err(ForwardError::rejected(
            "this runner has no asynchronous restore".to_owned(),
            blob,
        ))
    }

    /// Has an in-flight restore landed? `Ok(false)` means "not yet".
    ///
    /// A failure carries the blob back ([`ForwardError::Rejected`]), so the caller can put the
    /// session back on its tier rather than losing it.
    fn restore_poll(&mut self, seq: SeqId) -> Result<bool, ForwardError> {
        let _ = seq;
        Err(ForwardError::CannotPark)
    }

    /// The **dense** logits of one row of the step that just ran, as f32.
    ///
    /// The fallback behind the seam, and it is here rather than hidden inside the runner because
    /// it is a cost the caller has to count: a row whose truncation is not narrower than K cannot
    /// be sampled from the top-K at all (`saber_serve::sampler::SamplerError::SeamUntruncated`),
    /// and the only honest answer is to read the whole 248,320-wide row back and sample it
    /// densely. That is 485 KiB per row and a host sort, so every use of it is counted.
    ///
    /// Valid only until the next step overwrites the head's arena. `row` is the row index within
    /// the step that just ran, not a slot.
    fn dense_row(&mut self, row: u32) -> Result<Vec<f32>, ForwardError> {
        let _ = row;
        Err(ForwardError::Device(
            "this runner cannot read a dense logit row back".to_owned(),
        ))
    }

    /// Score the rows of the prefill chunk that JUST ran against `targets` (V7, `prompt_logprobs`).
    ///
    /// `targets[i]` is the prompt token after row `i` of that chunk, so the chunk's hidden states
    /// are still where its body left them and nothing is recomputed: the head runs again over
    /// those rows, `R` at a time, with the seam at T = 1 and the targets named in its gather. It
    /// must be called before any other step; a call for a sequence other than the last prefill's,
    /// or for more rows than it fed, is refused by name.
    ///
    /// The default refuses: a runner that cannot score prompt rows makes the executor answer
    /// `None` from [`ForwardRunner::prompt_rows_k`], and the scheduler refuses the request.
    fn prompt_rows(
        &mut self,
        seq: SeqId,
        targets: &[TokenId],
    ) -> Result<Vec<PromptRow>, ForwardError> {
        let _ = (seq, targets);
        Err(ForwardError::Device(
            "this runner cannot score prompt rows".to_owned(),
        ))
    }

    /// The top-k width [`ForwardRunner::prompt_rows`] can answer, or `None` when it cannot score
    /// prompt rows at all. The seam's K on a runner that can.
    fn prompt_rows_k(&self) -> Option<u32> {
        None
    }

    /// What the plan declares for that shape, or `None` if it has never been planned.
    ///
    /// Compared against [`StepOutcome::launches`] on **every** step: launch count is a budgeted,
    /// gated number (`CLAUDE.md` rule 7), and a step that quietly issues one more than the plan
    /// declares is the regression that rule exists for.
    fn declared_launches(&self, rows: StepRows) -> Option<u32>;

    /// **C9: stage a prompt chunk to run FUSED with the next [`ForwardRunner::decode`] call**
    /// (`--engine-mixed-prefill fused`). `true` = the runner took it: the next `decode` runs the
    /// chunk and the decode half as one forward (or, where the fused law refuses the pair, the
    /// chunk and then the decode half, serially, in that call), and the next
    /// [`ForwardRunner::prefill`] for the same chunk returns the chunk's half without running it
    /// again. The default refuses (a runner with no fused forward), and the caller runs the serial
    /// form.
    fn stage_mixed(&mut self, _chunk: MixedChunk) -> bool {
        false
    }

    /// **C10: whether the fused law takes these chunks** (a packed step, in queue order) beside
    /// a `batch` x `width` decode half. `false` (the default, and every refusal) makes the caller
    /// run the chunks and the decode half serially, each through its own call — so each chunk's
    /// dense fallback reads its own head. Chunks staged after a `true` are run fused by the next
    /// [`ForwardRunner::decode`].
    fn can_fuse(&mut self, _chunks: &[MixedChunk], _batch: u32, _width: u32) -> bool {
        false
    }

    /// **P27-3: whether a prompt's last chunk leaves its drafter store in the stash** (the
    /// `--drafter-store deferred` arm): the executor then returns the chunk's token before it
    /// collects the stash, and collects it at the head of its next call into the runner. False
    /// (the default, and every runner without a drafter): the store runs inside the chunk.
    fn defers_prompt_store(&self) -> bool {
        false
    }

    /// C9: what the last mixed call did — `Some((fused, rung))` once per mixed call, `fused`
    /// false for a serial fallback. `None` on a runner that never mixes.
    fn take_mixed_info(&mut self) -> Option<(bool, u32)> {
        None
    }

    /// Run the proposal the last step STASHED ([`StepOutcome::proposal_pending`]), anchored on
    /// the tokens the slots COMMITTED — prod's order, sample then propose (ws-f37).
    ///
    /// `anchors` is `(sequence, committed token)` per pending slot, in the pending order; a list
    /// that names other sequences, or a call with nothing pending, is a refusal by name. The
    /// proposal reads the rows the step just wrote into `aux.cat`, which nothing overwrites until
    /// the runner's next forward — so a caller runs this before any other call into the runner,
    /// and a runner that is called for anything else first ABANDONS the stash and counts it.
    ///
    /// The default is the refusal: a runner with no drafter of its own never stashes anything.
    ///
    /// **P27-3:** under [`ForwardRunner::defers_prompt_store`] a prompt's LAST chunk stashes its
    /// drafter context store here too, so the call also stores the prompt's last chunk before it
    /// proposes — and the executor may then hand the chunk's token back FIRST and make this call
    /// at the head of its next call into the runner (still before any other).
    fn propose_after_commit(
        &mut self,
        anchors: &[(SeqId, TokenId)],
    ) -> Result<Vec<Proposal>, ForwardError> {
        Err(ForwardError::Device(format!(
            "no proposal is pending on this runner ({} anchor(s) handed back): it has no drafter \
             of its own",
            anchors.len()
        )))
    }

    /// **The greedy composition** of a step and its proposal, for a caller that drives the pass
    /// directly and never samples on the host (every `saber engine` arm but the serve path).
    ///
    /// At temperature 0 the committed token IS the argmax, so the anchor handed back is each
    /// commit's own last token and the block is the one the step would have made in place — the
    /// two orders agree bit for bit on a greedy row, which is what `stale_anchor.rs` gates. A
    /// no-op when nothing is pending.
    ///
    /// # Errors
    /// Whatever [`ForwardRunner::propose_after_commit`] refuses.
    fn propose_greedy(&mut self, out: &mut StepOutcome) -> Result<(), ForwardError> {
        if out.proposal_pending.is_empty() {
            return Ok(());
        }
        let anchors: Vec<(SeqId, TokenId)> = out
            .commits
            .iter()
            .filter_map(|c| c.tokens.last().map(|t| (c.seq, *t)))
            .collect();
        let proposals = self.propose_after_commit(&anchors)?;
        out.drafts = proposals.into_iter().map(|p| p.block).collect();
        out.proposal_pending.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(batch: u32, width: u32) -> StepRows {
        StepRows {
            phase: saber_plan::Phase::Decode,
            rows: batch * width,
            width,
            samples: true,
            tree: false,
            ragged: false,
        }
    }

    /// The facts say WHICH ALL-REDUCE FAMILY each shape resolved to, and the two shapes an identity
    /// arm compares can be two different ones.
    ///
    /// This is ws-l17's open item as a host test: `b1s5` is 51,200 bytes and `b4s5` is 204,800, the
    /// rules' `quant_min_bytes` is 131,072, and a gate that compares a sequence alone with the same
    /// sequence in a batch of four is then comparing the exact reduce with the lossy one. The
    /// lookup is what lets the gate PRINT that beside the divergence instead of reporting a defect.
    ///
    /// NEGATIVE CONTROL in the same test: a shape the run did not plan, and a TP=1 run, both answer
    /// `None` — because "no entry" has to mean "no collective" and not "I did not look".
    #[test]
    fn the_facts_say_which_all_reduce_family_each_shape_resolved_to() {
        let mut f = ForwardFacts {
            tp: 2,
            ..ForwardFacts::default()
        };
        f.ar_families.push((
            rows(1, 5),
            "saber_ar_oneshot_2rank_exact_bf16".to_owned(),
            129,
        ));
        f.ar_families.push((
            rows(4, 5),
            "saber_ar_oneshot_2rank_wht6_bf16".to_owned(),
            129,
        ));
        assert_eq!(
            f.ar_family(rows(1, 5)).as_deref(),
            Some("saber_ar_oneshot_2rank_exact_bf16 x129")
        );
        assert_eq!(
            f.ar_family(rows(4, 5)).as_deref(),
            Some("saber_ar_oneshot_2rank_wht6_bf16 x129")
        );
        assert_ne!(f.ar_family(rows(1, 5)), f.ar_family(rows(4, 5)));
        // A shape this run did not plan, and a width-1 bucket of the same row count, which is a
        // DIFFERENT shape and must not match by row count alone.
        assert_eq!(f.ar_family(rows(8, 5)), None);
        assert_eq!(f.ar_family(rows(5, 1)), None);
        // TP=1: no collective, so no family, and the field is empty rather than absent.
        let one = ForwardFacts::default();
        assert_eq!(one.tp, 0);
        assert!(one.ar_families.is_empty());
        assert_eq!(one.ar_family(rows(1, 5)), None);
    }

    /// Several families at one shape are listed, sorted, so the line is stable between runs.
    #[test]
    fn a_shape_with_two_families_lists_both() {
        let mut f = ForwardFacts {
            tp: 2,
            ..ForwardFacts::default()
        };
        f.ar_families.push((rows(1, 1), "zeta".to_owned(), 1));
        f.ar_families.push((rows(1, 1), "alpha".to_owned(), 128));
        assert_eq!(
            f.ar_family(rows(1, 1)).as_deref(),
            Some("alpha x128, zeta x1")
        );
    }
}

#[cfg(test)]
mod bucket_facts {
    use super::*;

    fn shape(rows: u32, width: u32) -> StepRows {
        StepRows {
            phase: crate::forward::Phase::Decode,
            rows,
            width,
            samples: true,
            tree: false,
            ragged: false,
        }
    }

    fn facts() -> ForwardFacts {
        ForwardFacts {
            attention_interval: 4,
            bucket_ops: vec![
                (
                    shape(8, 8),
                    "layer.N.mlp.down_proj".to_owned(),
                    "gemm_ks4_tm1 grid[40,1,4] block[256]".to_owned(),
                    64,
                ),
                (
                    shape(32, 8),
                    "layer.N.mlp.down_proj".to_owned(),
                    "gemm_ks2_tm2 grid[40,1,2] block[256]".to_owned(),
                    64,
                ),
                (
                    shape(8, 8),
                    "layer.N.attn.decode".to_owned(),
                    "attn_nw3 grid[32,4,1] block[96]".to_owned(),
                    16,
                ),
                (
                    shape(32, 8),
                    "layer.N.attn.decode".to_owned(),
                    "attn_nw3 grid[32,4,4] block[96]".to_owned(),
                    16,
                ),
                (
                    shape(8, 8),
                    "embed_tokens".to_owned(),
                    "embed grid[8,1,1] block[256]".to_owned(),
                    1,
                ),
                (
                    shape(32, 8),
                    "embed_tokens".to_owned(),
                    "embed grid[32,1,1] block[256]".to_owned(),
                    1,
                ),
            ],
            ..ForwardFacts::default()
        }
    }

    /// The deltas are every op the two shapes resolved differently, by generic name, and a shape
    /// against itself has none — which is what makes the comparison able to say "nothing".
    #[test]
    fn the_deltas_are_the_ops_two_shapes_resolved_differently() {
        let f = facts();
        let d = f.bucket_op_deltas(shape(8, 8), shape(32, 8));
        let names: Vec<&str> = d.iter().map(|(n, _, _, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "embed_tokens",
                "layer.N.attn.decode",
                "layer.N.mlp.down_proj"
            ]
        );
        assert!(f.bucket_op_deltas(shape(8, 8), shape(8, 8)).is_empty());
    }

    /// And one op's delta is found from its OWN name, layer number and all: the deltas are keyed
    /// on the generic name and a probe names `layer.31.mlp.down_proj`.
    #[test]
    fn an_ops_delta_is_found_from_the_name_a_probe_reports() {
        let f = facts();
        let (a, b) = f
            .bucket_delta_for_op(shape(8, 8), shape(32, 8), "layer.31.mlp.down_proj")
            .expect("the GEMM's delta");
        assert!(a.contains("ks4") && b.contains("ks2"));
        assert!(f
            .bucket_delta_for_op(shape(8, 8), shape(32, 8), "layer.31.gdn.norm")
            .is_none());
    }

    /// The layer KIND comes from the model's own interval, and the embedding is neither.
    #[test]
    fn the_layer_kind_is_the_models_own_interval() {
        let f = facts();
        assert_eq!(f.layer_kind_of(-1), "embedding");
        assert_eq!(f.layer_kind_of(3), "attention");
        assert_eq!(f.layer_kind_of(31), "attention");
        assert_eq!(f.layer_kind_of(0), "gdn");
        assert_eq!(f.layer_kind_of(30), "gdn");
    }
}
