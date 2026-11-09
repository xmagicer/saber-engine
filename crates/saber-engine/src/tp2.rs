//! The two-rank engine: how rank 0 drives rank 1, and how the vocabulary shards come back together.
//!
//! At TP = 2 the model is halved down the middle of every GEMM and every vocabulary. One card can
//! run neither half alone, so a served step is **two processes stepping in lockstep**: rank 0 hosts
//! the scheduler and the serve plane and rank 1 runs nothing but the same engine on the other card,
//! told what to do once per step.
//!
//! # What crosses the wire, and what does not
//!
//! Not the activations. The 129 all-reduces are `saber` kernels writing into the peer's
//! peer-visible scratch over PCIe, **inside each rank's captured graph** — the host never sees them
//! (`saber_exec::PeerCapture::Lockstep` is the assertion that makes that legal, and
//! [`crate::device`] pays its three prices where it passes it). What crosses the wire is:
//!
//! 1. one [`Command`] per step — the phase, the row count, and the part of the index blob the step
//!    REWROTE, which is **byte-identical on the two ranks** because the sequences, their KV pages
//!    and their GDN slots are the same on both cards;
//! 2. the `block_table` region, in a frame of its own, on the steps a sequence crossed a page
//!    boundary in ([`Command::Step::paged`]) — 2 MiB at `max_ctx` 32768 that fifteen decode steps in
//!    sixteen do not change, and that the pair used to swap every step;
//! 3. one [`Pick`] per row afterwards — this rank's local argmax and the logit it won with;
//! 4. on a step that SAMPLES, two more frames of the same kind — the leader's per-row temperatures
//!    and penalty ids before the replay ([`RankLink::lead_seam`]) and the follower's shard of
//!    every sampled row after it ([`RankLink::send_seam`]). A top-K over half a vocabulary is not
//!    the row's top-K, so the two halves are merged on the leader by
//!    `saber_serve::sampler::merge_shards` — 520 bytes a row against the 485 KiB a dense read
//!    would move, and **nothing at all on a step whose rows are all greedy**;
//! 5. on the 0.4% of sampled rows whose truncation does not fit inside K, a
//!    [`Command::DenseRow`] and one shard of raw logits each way — the dense fallback behind the
//!    seam, which at two ranks needs the other half of the row to exist at all.
//!
//! What does NOT cross it any more (ws-h29): the plan's **constants**. `cos_sin` is 4 MiB at
//! `max_ctx` 32768 and is a function of `rope_theta` and the context band, so both ranks build it
//! from their own identical plans at open and neither tells the other about it.
//!
//! Both ride `saber_exec::IpcExchange`, the same socket the IPC handles and the collective digests
//! went over, because "two transports between the same pair of ranks would interleave their
//! messages" (`saber_exec::peer`).
//!
//! # The logits gather is on the host, and it is counted
//!
//! `lm_head` is column-parallel: rank *r* holds vocabulary `[r * V/2, (r+1) * V/2)`, so neither
//! rank's argmax is the model's argmax. The graph's `logits.gather` — the all-gather that would fix
//! that on the device — **has no code object**, and `saber_exec::layer` refuses to plan it rather
//! than inventing a route (`KernelOp::LogitsGather`). So the combine is here, on the host, and it
//! moves 8 bytes per row instead of a vocabulary shard: each rank's argmax runs on its own
//! `[rows, V/2]` tile, each rank reads back the one logit its winner scored, the two ranks swap
//! `(index, value)` pairs and [`combine`] picks. That is exactly the answer a device all-gather
//! would give, at 16 bytes per row per step instead of 300 KiB, and every one of those reads is in
//! [`crate::device`]'s `host_gathers` counter.
//!
//! # The API this module wants
//!
//! * `IpcExchange::swap_fixed(name, &[u8])` — a symmetric swap of an agreed, constant width. Every
//!   payload here is one, and `swap_bytes` spends a round trip per call agreeing a length both
//!   ranks computed from the same plan at startup.
//! * An **asymmetric** `send_bytes` / `recv_bytes`. The command blob flows one way only; the
//!   follower answers it with an equal-length buffer of zeros purely to satisfy the symmetric
//!   contract, which DOUBLES the per-step wire bytes. That is 1.9 MiB a step of pure waste at
//!   `max_seqs` 8 today, and 2 MiB more on every page boundary — the largest thing left on this
//!   wire now that the constants and the unchanged block table are off it (ws-h29).
//! * `saber_exec::layer::Route::Supplied` for `KernelOp::LogitsGather`, which would let the head be
//!   PLANNED at TP > 1 and leave only the gather to the caller. Today the whole head plan fails and
//!   [`crate::device`] hand-resolves the final norm and `lm_head`.

use std::sync::Arc;

use saber_exec::RankPeer;
use saber_plan::Phase;
use saber_serve::engine::TokenId;

use crate::forward::ForwardError;

/// Bytes of the fixed command header that precedes every step payload.
///
/// **32 until ws-h30**, which added the prefill step's GDN state slot; **40 until V4**, which added
/// the fork's two page numbers — a [`Command::Fork`] names five things (two slots, the set's width
/// and the two ends of the page copy) and the header held four free words. In both cases the
/// follower cannot derive what was added: it keeps no session state and owns no slot table.
/// A header that GROWS is a wire change, so it carries [`VERSION`] beside it rather than being
/// discovered as a length mismatch in the middle of a run.
pub const HEADER_BYTES: usize = 48;

/// `SBT2`, so a frame that is not one says so instead of being decoded.
const MAGIC: u32 = 0x5342_5432;

/// The command protocol this build speaks.
///
/// 1 = the 32-byte header (ws-h29 and before). 2 = the 40-byte header with the prefill state slot
/// (ws-h30). 3 = the sampler's seam across the pair (V3): two flag bits and the two seam frames
/// they announce, plus [`Command::DenseRow`]. 4 = the 48-byte header and [`Command::Fork`] (V4):
/// a sequence group's branch is created on BOTH cards, because each holds half of the session's
/// KV and half of its recurrent state. 5 = the DRAFTER at two ranks (ws-s6): a decode step with a
/// resident drafter is followed by one `engine.draft` frame per proposal and a terminator, each
/// proposal swaps an `engine.draftcand` seam, and the blocks' digests ride back on the frames.
/// 6 = the PARK at two ranks (ws-t2): [`Command::Park`] and one `engine.park` descriptor frame
/// per op, because a session's KV, GDN state and drafter pages are half on each card and half a
/// session is not a session. 8 = [`Command::PromptRows`] (V7): `prompt_logprobs` re-runs the head
/// over the last prefill chunk's rows on both cards, because each holds half the vocabulary.
/// 9 = the proposal AFTER THE COMMIT (ws-f37): [`Command::Step::defer_proposals`] (flag bit 4)
/// says the step's proposal frames do NOT follow it, and [`Command::Propose`] is what starts them
/// once the leader's host sampler has decided every slot's token — so the committed anchor is in
/// the leader's frame before the follower's proposal reads it. 10 = the fork UNDER THE DRAFTER
/// (ws-f38): [`Command::ForkDraft`] copies the leader's drafter pages into a branch's own run on
/// both cards, and the branch's first proposal is a `Propose` of its own.
/// 11 = IMAGES at two ranks (D4): [`Command::Vision`] and its pixel frame run the replicated tower
/// on both cards (the two ranks then swap a digest of their rows), and [`Command::Scatter`] and
/// its runs frame precede a prefill chunk that carries image rows.
/// 12 = the decode ROW BAND (C2): [`Command::Step::ragged`] (flag bit 5) says `rows` is a band's
/// row count and `width` its MAX width, so the follower replays the band's graph rather than the
/// rectangle that shares `(rows, width)`; the per-slot widths are the `cu` table already in the
/// step's index frame, and `live` is `Σ w_i`.
/// 14 = the PACKED fused step (C10): [`Command::Mixed::packed`] (the header's last word) carries
/// the further chunks' count and per-chunk head / seam masks; their rows and slots ride in the
/// index frame's packed tables.
/// 15 = the PREFIX CACHE (FN52): [`Command::Prefix`] saves, loads and drops one rank's half of a
/// retained prefix's snapshot (GDN live slots + drafter pages; the KV pages stay in place).
/// 16 = the GDN replay arm under the tree (CMP27G): [`Command::GdnMaterialize`] applies a
/// sequence's pending replay to its GDN base on both cards before a tree step reads it.
/// 13 = the FUSED mixed step (C9): [`Command::Mixed`] runs a prompt chunk and a decode half as
/// ONE forward at a prefill rung over the combined index blob that follows it; its sampling,
/// seam and context-store frames follow in the leader's order (the chunk's head, the decode
/// head, the chunk's store), and both halves' proposals are deferred `Propose`s.
/// 17 adds bit 6 to select the tree graph independently of its ordinary chain twin.
/// It is in EVERY frame and checked on every decode,
/// because the failure it prevents is a follower reading the leader's step fields out of the wrong
/// offsets and replaying a shape that looks legal — and both ranks being the same binary is a
/// deployment convention, not a law.
///
/// It is ALSO agreed once at boot, before a frame crosses at all
/// ([`saber_exec::RankPeer::agree`], through [`crate::device`]'s startup handshake): a pair of
/// mismatched binaries is then a message while both cards are still idle, rather than a refused
/// frame with a graph already in flight on each of them.
pub const VERSION: u32 = 17;

/// **C10: a packed step's word** ([`Command::Mixed::packed`]): `more` further chunks, and per
/// chunk `k` (bit `k`) whether its head samples, its seam runs, and its seam carries penalties.
#[must_use]
pub fn packed_word(more: u32, samples: u8, seams: u8, penalties: u8) -> u32 {
    (more & 0xff)
        | (u32::from(samples) << 8)
        | (u32::from(seams) << 16)
        | (u32::from(penalties) << 24)
}

/// [`packed_word`] read back: `(more, samples, seams, penalties)`.
#[must_use]
pub fn packed_fields(word: u32) -> (u32, u8, u8, u8) {
    (
        word & 0xff,
        (word >> 8) as u8,
        (word >> 16) as u8,
        (word >> 24) as u8,
    )
}

/// Bytes one row's [`Pick`] occupies on the wire.
pub const PICK_BYTES: usize = 8;

/// Which rank of the pair this process is, and therefore what it does with the socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerRole {
    /// Rank 0: hosts the scheduler and the serve plane, and tells the other rank what to run.
    Leader,
    /// Rank 1: runs the same engine on the other card and does what it is told, until it is told
    /// to stop or the socket goes quiet for longer than its deadline.
    Follower,
}

/// How this process meets its peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerConfig {
    /// This rank's role.
    pub role: PeerRole,
    /// The loopback port the two ranks meet on. The leader listens; the follower connects.
    pub port: u16,
    /// How long the rendezvous may take, and how long any single socket read may block.
    ///
    /// It bounds the follower's idle wait too, which is why the leader sends [`Command::Ping`]
    /// while it has nothing to do: a follower that hears nothing for this long exits and gives the
    /// card back rather than sitting on it (`CLAUDE.md`: every wait is deadline-bounded).
    pub deadline_s: u64,
}

/// What rank 0 tells rank 1 to do.
///
/// Five verbs, because rank 1 keeps **no session state**: rank 0 owns the slot table, the block
/// tables and the token stream, and what rank 1 receives is the finished index blob rather than the
/// facts it was built from. The one piece of session state that is a device action — clearing a GDN
/// slot a finished sequence left dirty — is [`Command::ClearSlot`]. [`Command::DenseRow`] names the
/// head it wants by SHAPE for the same reason: there is nothing on rank 1 that remembers which head
/// last ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// Nothing to do; the leader is idle and this is what keeps the follower's socket read from
    /// timing out. Carries no payload.
    Ping,
    /// **CMP27G: apply one sequence's pending GDN replay to its base** (`--engine-gdn-replay on`
    /// under the tree default): `rows` stashed rows from region `par` of the stash, the base slot
    /// `slot`, the admission `tag` the stash header must carry -- the `_materialize` launch a GDN
    /// layer, on this card, before the tree step that reads the base. A fact only the leader
    /// holds (rank 1 keeps no session state), so it crosses as its own command.
    GdnMaterialize {
        /// The sequence's GDN base slot (its slot set's first slot). Never 0.
        slot: u32,
        /// Rows the base owes (the last chain step's committed count).
        rows: u32,
        /// The sequence's admission tag.
        tag: u32,
        /// The stash region the rows are in.
        par: u32,
    },
    /// Zero GDN state slot `slot` on this card, exactly as the leader is doing on its own.
    ClearSlot {
        /// The slot to zero. Never 0: slot 0 is the kernels' null slot.
        slot: u32,
    },
    /// Run one step at this shape over the index blob that follows.
    Step {
        /// Prefill chunk or decode step.
        phase: Phase,
        /// The GDN state slot a PREFILL chunk's `saber_gdn_chunk_scan` reads `h0` from and writes
        /// `ht` to — the step's first live sequence's slot. **0 on a decode step**, which finds
        /// every slot it touches through `sidx` instead.
        ///
        /// On the wire (ws-h30) because it is a fact the follower cannot derive: it keeps no
        /// session state and owns no slot table, and this is the one GDN binding that is not an
        /// index table the leader's blob already answers. A follower that guessed would scan the
        /// prompt through the null slot on its half of every reduce.
        slot: u32,
        /// The planned row count — the bucket's, padding included.
        rows: u32,
        /// The verify width those rows are grouped into: `rows / width` sequences of `width` rows.
        ///
        /// On the wire because **a row count is not a bucket**: `b1s5` and `b5s1` are both five rows
        /// and are different graphs. The follower resolves the leader's shape against its own plans,
        /// so without this it would find the pair's other bucket and replay the wrong graph in
        /// lockstep with a leader that is replaying the right one.
        width: u32,
        /// Live rows, for the follower's own assertion.
        live: u32,
        /// Whether the head runs and a token is sampled.
        samples: bool,
        /// Whether a second frame carries the `block_table` region of the index blob.
        ///
        /// On the wire because it is a fact the follower cannot derive: it has no scheduler and no
        /// page lists, so only the leader knows whether a sequence crossed a page boundary this
        /// step. At `max_ctx` 32768 and `max_seqs` 8 that region is 2 MiB, and a symmetric swap
        /// moves it TWICE — so sending it on the fifteen steps in sixteen that do not change it is
        /// 4 MiB of loopback socket per step to re-state the table the peer already holds
        /// (ws-h28's +6.6 ms/step, ws-h29's attribution).
        paged: bool,
        /// Whether the sampler's seam runs this step, and therefore whether the two extra frames
        /// ride with it: the leader's temperatures **before** the replay, and the follower's
        /// shard of every sampled row **after** it.
        ///
        /// On the wire because it is the leader's policy and not a shape: a step whose rows are
        /// all greedy is answered by the two argmaxes and the [`Pick`] swap alone, and pays
        /// nothing here. Both ranks' seam buffers are zeroed at boot and a zero temperature is
        /// the kernel's own "skip me", so `false` really is the step it was.
        seam: bool,
        /// Whether those frames also carry the penalty ids and the logits gathered at them.
        ///
        /// Separate from `seam` because it is by far the larger half — `rows * 256` ids each way
        /// against `rows` temperatures — and prod's own defaults carry no penalty at all.
        seam_penalties: bool,
        /// **The step's proposal is DEFERRED** (ws-f37): the follower does not loop proposal
        /// frames at the end of this step; it goes back to its command loop and runs them when
        /// the leader sends [`Command::Propose`].
        ///
        /// On the wire because it is the leader's order and not a shape. Prod proposes after its
        /// sampler, so the anchor is the token the sequence committed — and on the leader that
        /// token is decided on the host AFTER the step's picks and seam have crossed. A follower
        /// that went on waiting for proposal frames would be sitting on the socket while the
        /// leader needed it for something else (a [`Command::DenseRow`] behind a refused seam).
        /// `false` is the in-step order, byte for byte the frames this wire carried before.
        defer_proposals: bool,
        /// **A decode ROW BAND (C2)**: `rows` is the band's row count and `width` its MAX verify
        /// width, each slot at its own width packed by `cu`. On the wire because the band and the
        /// rectangle of the same `(rows, width)` are different graphs (the 40-row band at max
        /// width 8 and `b5s8`), and the follower resolves the leader's shape against its own plans.
        /// The per-slot widths need no field of their own: they are the `cu` table the leader's
        /// index frame already carries, which is what every kernel on the follower reads.
        ragged: bool,
        /// Select the ancestor graph when the ordinary chain uses the same (rows, width).
        tree: bool,
    },
    /// **C9: one FUSED mixed step**: a prompt chunk of `chunk_live` rows (rows `0..chunk_live`)
    /// and a decode bucket of `decode_rows` rows at verify `width` (rows `chunk_live..`) as ONE
    /// forward at the prefill rung of `rung` rows, over the combined index blob that follows. The
    /// follower builds the same launch list from its own plans (`saber_exec::mixed`), samples the
    /// chunk's head (one row, when `chunk_samples`) and then the decode head (`decode_live` rows),
    /// and loops the chunk's context-store frames; both halves' proposals are deferred to
    /// `Propose`s (a fused step is only ever sent with `defer_proposals`).
    Mixed {
        /// The prefill rung the row-local ops run at.
        rung: u32,
        /// The chunk's live rows (its attention's `q_len`, its head's row is the last of them).
        chunk_live: u32,
        /// Whether the chunk's head runs and samples (the prompt's last chunk).
        chunk_samples: bool,
        /// The chunk's GDN state slot (`Step::slot`'s law).
        slot: u32,
        /// The decode bucket's rows (padding included): the decode plan is `(rows, width)`.
        decode_rows: u32,
        /// The decode bucket's verify width.
        width: u32,
        /// Decode rows sampled.
        decode_live: u32,
        /// A `block_table` frame follows (always, on a fused step: both halves' tables move).
        paged: bool,
        /// The chunk head's seam runs, and whether with penalties.
        chunk_seam: bool,
        /// See `chunk_seam`.
        chunk_seam_penalties: bool,
        /// The decode head's seam runs, and whether with penalties.
        decode_seam: bool,
        /// See `decode_seam`.
        decode_seam_penalties: bool,
        /// Both halves' proposals wait for `Propose`s.
        defer_proposals: bool,
        /// **C10: a PACKED step** ([`packed_word`]): 0 for C9's one chunk; otherwise bits 0-7 are
        /// the further chunks (their rows, slots and tables are in the index frame under
        /// `saber_exec::mixed::packed_key`), and bits 8-15 / 16-23 / 24-31 are, per chunk `k`
        /// (bit `k`, chunk 0 included), whether its head samples, its seam runs, and its seam
        /// carries penalties. The chunk fields above are chunk 0's.
        packed: u32,
    },
    /// **Run the proposal the last step deferred** (ws-f37): the follower loops `engine.draft`
    /// frames exactly as it would have at the end of that step, one per proposal and a
    /// terminator. No payload — the frames carry the tables, and the tables carry the COMMITTED
    /// anchor at each request's row 0.
    ///
    /// A deferred proposal the leader never collected is simply never announced: the follower is
    /// in its command loop and the pair stays in step, and the leader counts the abandoned stash.
    ///
    /// `store_first` (P27-3, `--drafter-store deferred`): the prompt's last chunk left its context
    /// store in the stash, so the follower first loops that chunk's context frames to their `End`
    /// marker exactly as it would have inside the chunk (`follow_ctx_store`), then the proposal's.
    Propose {
        /// The prompt's last-chunk context tiles precede the proposal's frames.
        store_first: bool,
        /// P27-14 (`socket`): the proposal's FIRST drafter frame is this command's payload, not a
        /// frame of its own.
        inline: bool,
        /// P27-14 (`socket`): the block digest of the previous single-proposal step, whose
        /// terminator was not sent; the follower checks it against its own.
        carry: Option<u64>,
    },
    /// Read one row of the head's `logits.partial` back and swap it — the DENSE fallback behind
    /// the sampler's seam, at two ranks.
    ///
    /// A row whose truncation is not narrower than K cannot be sampled from a top-K at all
    /// (`saber_serve::sampler::SamplerError::SeamUntruncated`), and at TP = 2 neither rank's half
    /// of it is a distribution. So the leader asks for the other half and concatenates — 248 KiB
    /// each way for the 0.4% of rows at prod's own defaults that need it, against the 520 bytes a
    /// row usually costs.
    ///
    /// It carries the head's own shape rather than relying on the follower to remember which head
    /// last ran, because **rank 1 keeps no session state** and the prefill head and `b1s1`'s are
    /// both one row.
    DenseRow {
        /// Prefill head or decode head.
        phase: Phase,
        /// Rows that head ran, which is how the follower picks it.
        head_rows: u32,
        /// The row within it.
        row: u32,
    },
    /// Copy a session's GDN state slot SET, and one KV page, on this card: the device half of an
    /// `n > 1` fork (V4).
    ///
    /// Both ranks run it, over their own half of the model — this rank's KV pool is its own K/V
    /// shard and its GDN slots are its own state — so a branch exists on both cards or on
    /// neither. Everything else about a fork (which sequence, which choice, the token it draws)
    /// is the leader's bookkeeping and never crosses.
    Fork {
        /// The leader's state slot: the BASE of its run of `slots` consecutive snapshots.
        from_slot: u32,
        /// The branch's base, which the leader has just taken off its own free list. Never 0.
        to_slot: u32,
        /// Snapshots in a set — the planned verify width. On the wire because the follower's own
        /// `max_width` is the same number and a disagreement would copy a different amount of
        /// state on each card, which is exactly the kind of thing that changes tokens and reports
        /// nothing.
        slots: u32,
        /// The leader's partial KV page, or `None` when the prompt ended on a page boundary.
        cow_source: Option<u32>,
        /// The branch's first private page, which that one is copied into.
        cow_dest: Option<u32>,
    },
    /// Zero this session's DRAFTER pages on this card: the device half of an admission, at two
    /// ranks (ws-s6).
    ///
    /// The same shape as [`Command::ClearSlot`] and for the same reason. A session's drafter K/V
    /// is half on each card, the drafter's attention reads `last_valid_pos + 1 + block` positions
    /// of it whether this session has written them or not, and the clear is what keeps a new
    /// request from proposing out of the last one's history. The leader clears at admission, and
    /// a follower that did not would leave ITS half dirty — which changes no token (`draft::Greedy`
    /// commits the model's own answer) and changes the block, and therefore the acceptance. Found
    /// by `clear_pages`' own gate arm on the first two-rank run: the case failed and its control
    /// passed, which is the arm working.
    ClearDraftPages {
        /// The first page of this session's run in the drafter's own pool.
        base: u32,
        /// How many pages the run covers.
        pages: u32,
    },
    /// One step of a session PARK or RESTORE, over this rank's half (ws-t2).
    ///
    /// The `engine.park` frame that follows carries the session — which sequence, its GDN state
    /// slot, its KV pages, its drafter page run and its resident token count — because **rank 1
    /// keeps no session state**, exactly as [`Command::ClearDraftPages`] carries a page run and
    /// [`Command::Fork`] carries two slots. The same frame carries the follower's answer back:
    /// `swap_bytes` is symmetric, so the reply costs nothing extra.
    ///
    /// **The image does not cross the wire.** Each rank parks its own half into its own staging
    /// buffer and keeps its own blob; the leader holds the digest and the byte count of the
    /// other half, and a restore that cannot produce them is refused by name rather than
    /// publishing half a session. That is why a park at two ranks is five frames of 4 KiB and not
    /// 260 MiB of loopback socket.
    Park {
        /// Which step of it.
        op: ParkOp,
    },
    /// Re-run the HEAD over rows of the last prefill chunk, for `prompt_logprobs` (V7).
    ///
    /// The leader's scheduler asked for the prompt's own distribution at every row, and each rank
    /// holds half of every row's logits (`lm_head` is column-parallel), so both ranks run the
    /// widest decode head's gathers, final norm, `lm_head` and seam over rows
    /// `first .. first + count` of the prefill plan that just ran — `head_rows` wide, the tail
    /// padded with the last row and skipped by the seam — and the follower sends its shard back
    /// on the seam's own two frames. A decode head's `sel` and temperatures are put back before
    /// this rank reads another command, so the captured decode graphs see what they always saw.
    PromptRows {
        /// The prefill plan's row count, which is how the follower finds its body arena.
        prefill_rows: u32,
        /// The decode head that runs, by its row count.
        head_rows: u32,
        /// The first chunk row of this block.
        first: u32,
        /// Live rows in this block, `1..=head_rows`.
        count: u32,
    },
    /// Give a forked BRANCH its half of the leader's DRAFTER pages on this card (ws-f38): the
    /// drafter half of an `n > 1` fork under a resident drafter.
    ///
    /// The same shape as [`Command::ClearDraftPages`] and [`Command::Fork`], and for the same
    /// reason: each card holds half of every drafter page (its K/V heads are column-parallel),
    /// rank 1 keeps no session state, and a branch whose drafter context existed on one card
    /// only would propose one block from two different histories. For every drafter layer and
    /// both caches: pages `from_first .. from_first + copy` are copied to `to_first ..`, and the
    /// rest of the branch's run is zeroed (`crate::draft_tier::DraftForkCopy`).
    ForkDraft {
        /// The leader's first drafter page.
        from_first: u32,
        /// The branch's first drafter page.
        to_first: u32,
        /// Pages in each run.
        run: u32,
        /// Pages copied from the front of the leader's run.
        copy: u32,
    },
    /// **Run the vision tower over one request's images on this card** (D4). The frame that
    /// follows ([`vision_frame`]) carries the grids and the processed `pixel_values` -- rank 1 has
    /// no serving plane, so the pixels are the one input it cannot derive -- and both ranks then
    /// swap a digest of the `[tokens, 5120]` rows they produced: the tower is REPLICATED (D2's
    /// residency), so the two must be the same bytes, and a pair that disagreed would scatter two
    /// different images into the two halves of one residual stream.
    Vision {
        /// Images in the frame.
        images: u32,
        /// Patches, summed.
        patches: u32,
    },
    /// **The next prefill chunk replaces image rows** (D4): the frame that follows carries the
    /// chunk's scatter runs ([`scatter_frame`]), and the follower's next `Step` (a prefill) copies
    /// them out of its own tower arena after its embedding, exactly as the leader does. The runs
    /// ride beside the step rather than in the index blob because no kernel reads them: they are
    /// `Job` copies between two launches of the step.
    Scatter {
        /// Runs in the frame.
        runs: u32,
    },
    /// Shut down: release the card and exit 0.
    Stop,
    /// **FN52: a retained prefix's snapshot on this rank** (wire 15). `op` 0 = SAVE this rank's
    /// half (GDN live slots of the run based at `slot` after `committed` tokens, and drafter pages
    /// `draft_first .. + draft_pages`) under `entry`; 1 = LOAD it into the run based at `slot`
    /// (a fresh admission, `committed` 1) and those drafter pages; 2 = DROP it. Rank 1 keeps the
    /// bytes, as it keeps a parked half: they are this card's shard and worth nothing elsewhere.
    Prefix {
        /// 0 save, 1 load, 2 drop.
        op: u32,
        /// The entry's handle.
        entry: u32,
        /// The state slot run's base.
        slot: u32,
        /// Tokens the run's last step committed (which slot is live).
        committed: u32,
        /// The drafter run's first page.
        draft_first: u32,
        /// Drafter pages the prefix covers (0 = no drafter).
        draft_pages: u32,
    },
}

/// Which step of a two-rank park a [`Command::Park`] frame is.
///
/// The pair is a state machine per rank and the two run in lockstep: the leader issues its own
/// copy and tells the follower to issue its own, then each poll advances both halves by a bounded
/// amount. A park is done when BOTH are, which is what makes "the session is parked" mean the
/// same thing on both cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParkOp {
    /// Issue this rank's half of the park. Answers `Busy` when one is already in flight.
    ParkBegin,
    /// Advance it by at most one poll's worth. Answers `Pending` or `Done` with the half's digest.
    ParkPoll,
    /// Issue the restore of the half this rank parked earlier, from its own blob.
    RestoreBegin,
    /// Advance it. `Done` means this rank's half is published.
    RestorePoll,
    /// Discard the parked half this rank is holding: the image was dropped, not restored.
    Forget,
}

/// The pixel frame a [`Command::Vision`] announces (D4): `[images u32][grid 3 x u32 per
/// image][pixel_values f32 LE ...]`. The header's counts are checked against the grids and the
/// pixel count against the patches, so a torn frame is refused by name.
#[must_use]
pub fn vision_frame(grids: &[[u32; 3]], pixels: &[f32]) -> Vec<u8> {
    let mut f = Vec::with_capacity(4 + grids.len() * 12 + pixels.len() * 4);
    f.extend_from_slice(&(grids.len() as u32).to_le_bytes());
    for g in grids {
        for v in g {
            f.extend_from_slice(&v.to_le_bytes());
        }
    }
    for p in pixels {
        f.extend_from_slice(&p.to_le_bytes());
    }
    f
}

/// The pixel row width of the processor's `pixel_values`: `3 x 2 x 16 x 16`.
pub const PIXEL_ROW: usize = 1536;

/// Decode a [`vision_frame`] against its command's counts.
///
/// # Errors
/// A frame whose image count, patch count or length disagrees with the header.
pub fn decode_vision_frame(
    f: &[u8],
    images: u32,
    patches: u32,
) -> Result<(Vec<[u32; 3]>, Vec<f32>), ForwardError> {
    let word = |at: usize| -> Result<u32, ForwardError> {
        f.get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| ForwardError::Peer(format!("a vision frame of {} byte(s)", f.len())))
    };
    let n = word(0)?;
    if n != images {
        return Err(ForwardError::Peer(format!(
            "a vision frame of {n} image(s) under a command that announced {images}"
        )));
    }
    let mut grids = Vec::with_capacity(n as usize);
    let mut total = 0u64;
    for k in 0..n as usize {
        let g = [word(4 + k * 12)?, word(8 + k * 12)?, word(12 + k * 12)?];
        total += u64::from(g[0]) * u64::from(g[1]) * u64::from(g[2]);
        grids.push(g);
    }
    if total != u64::from(patches) {
        return Err(ForwardError::Peer(format!(
            "a vision frame whose grids make {total} patch(es) under a command that announced \
             {patches}"
        )));
    }
    let body = &f[4 + n as usize * 12..];
    if body.len() != total as usize * PIXEL_ROW * 4 {
        return Err(ForwardError::Peer(format!(
            "a vision frame carrying {} pixel byte(s) for {total} patch(es) of {PIXEL_ROW} f32",
            body.len()
        )));
    }
    let pixels = body
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((grids, pixels))
}

/// The runs frame a [`Command::Scatter`] announces (D4): `(dst_row, src_row, rows)` u32 triples.
#[must_use]
pub fn scatter_frame(runs: &[saber_exec::mrope::ScatterRun]) -> Vec<u8> {
    let mut f = Vec::with_capacity(runs.len() * 12);
    for r in runs {
        for v in [r.dst_row, r.src_row, r.rows] {
            f.extend_from_slice(&v.to_le_bytes());
        }
    }
    f
}

/// Decode a [`scatter_frame`].
///
/// # Errors
/// A frame that is not `runs` triples.
pub fn decode_scatter_frame(
    f: &[u8],
    runs: u32,
) -> Result<Vec<saber_exec::mrope::ScatterRun>, ForwardError> {
    if f.len() != runs as usize * 12 {
        return Err(ForwardError::Peer(format!(
            "a scatter frame of {} byte(s) under a command that announced {runs} run(s)",
            f.len()
        )));
    }
    Ok(f.chunks_exact(12)
        .map(|c| {
            let w = |i: usize| u32::from_le_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
            saber_exec::mrope::ScatterRun {
                dst_row: w(0),
                src_row: w(4),
                rows: w(8),
            }
        })
        .collect())
}

impl Command {
    /// Does this command carry a step payload?
    #[must_use]
    pub fn has_payload(&self) -> bool {
        matches!(
            self,
            Command::Step { .. } | Command::Mixed { .. } | Command::Propose { inline: true, .. }
        )
    }

    fn encode(&self, serial: u64) -> [u8; HEADER_BYTES] {
        let mut h = [0u8; HEADER_BYTES];
        let mut cow: Option<(u32, u32)> = None;
        h[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        // `slot` carries the width for a `Step` and the state slot for a `ClearSlot`: one word, one
        // meaning per op, and both ranks are the same binary so the layout cannot disagree with
        // itself. The protocol digest the pair compares before any launch covers it.
        let (op, phase, rows, live, slot, flags, state_slot) = match *self {
            Command::Ping => (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32),
            Command::ClearSlot { slot } => (1, 0, 0, 0, slot, 0, 0),
            // CMP27G: `phase` = the parity, `rows` = the rows owed, `live` = the tag.
            Command::GdnMaterialize {
                slot,
                rows,
                tag,
                par,
            } => (15, par, rows, tag, slot, 0, 0),
            Command::Step {
                phase,
                rows,
                width,
                live,
                samples,
                paged,
                seam,
                seam_penalties,
                slot,
                defer_proposals,
                ragged,
                tree,
            } => (
                2,
                match phase {
                    Phase::Prefill => 0,
                    Phase::Decode => 1,
                },
                rows,
                live,
                width,
                // `flags` is a bit field: bit 0 says the head runs, bit 1 says a `block_table`
                // frame follows this one, bit 2 says the sampler's seam runs, bit 3 that its
                // frames carry the penalty ids as well as the temperatures, and bit 4 that the
                // step's proposal waits for a `Propose` (ws-f37), and bit 5 that the rows are a
                // decode ROW BAND (C2).
                u32::from(samples)
                    | (u32::from(paged) << 1)
                    | (u32::from(seam) << 2)
                    | (u32::from(seam_penalties) << 3)
                    | (u32::from(defer_proposals) << 4)
                    | (u32::from(ragged) << 5)
                    | (u32::from(tree) << 6),
                slot,
            ),
            Command::Mixed {
                rung,
                chunk_live,
                chunk_samples,
                slot,
                decode_rows,
                width,
                decode_live,
                paged,
                chunk_seam,
                chunk_seam_penalties,
                decode_seam,
                decode_seam_penalties,
                defer_proposals,
                packed,
            } => {
                // `phase` carries the decode width, `slot` the decode rows, the state-slot word the
                // chunk's GDN slot and V4's two words the decode live rows and (C10) the packed
                // step's word; flags as `Step`'s for the chunk (bits 0-4) plus bits 6/7 for the
                // decode head's seam.
                cow = Some((decode_live, packed));
                (
                    13,
                    width,
                    rung,
                    chunk_live,
                    decode_rows,
                    u32::from(chunk_samples)
                        | (u32::from(paged) << 1)
                        | (u32::from(chunk_seam) << 2)
                        | (u32::from(chunk_seam_penalties) << 3)
                        | (u32::from(defer_proposals) << 4)
                        | (u32::from(decode_seam) << 6)
                        | (u32::from(decode_seam_penalties) << 7),
                    slot,
                )
            }
            Command::Stop => (3, 0, 0, 0, 0, 0, 0),
            Command::Prefix {
                op,
                entry,
                slot,
                committed,
                draft_first,
                draft_pages,
            } => (14, op, entry, slot, draft_first, draft_pages, committed),
            Command::Vision { images, patches } => (11, 0, images, patches, 0, 0, 0),
            Command::Scatter { runs } => (12, 0, runs, 0, 0, 0, 0),
            Command::Propose {
                store_first,
                inline,
                carry,
            } => {
                let c = carry.unwrap_or(0);
                (
                    9,
                    0,
                    u32::from(store_first),
                    u32::from(inline),
                    c as u32,
                    u32::from(carry.is_some()),
                    (c >> 32) as u32,
                )
            }
            // `rows` = the leader's first page, `live` = the branch's, `slot` = the run, and the
            // copied count in the state-slot word, as `PromptRows` does.
            Command::ForkDraft {
                from_first,
                to_first,
                run,
                copy,
            } => (10, 0, from_first, to_first, run, 0, copy),
            Command::Park { op } => (
                7,
                0,
                match op {
                    ParkOp::ParkBegin => 0,
                    ParkOp::ParkPoll => 1,
                    ParkOp::RestoreBegin => 2,
                    ParkOp::RestorePoll => 3,
                    ParkOp::Forget => 4,
                },
                0,
                0,
                0,
                0,
            ),
            Command::ClearDraftPages { base, pages } => (6, 0, base, pages, 0, 0, 0),
            // `rows` = the prefill plan, `live` = the first row, `slot` = the head, and the count
            // in the state-slot word, which a `PromptRows` has no other use for.
            Command::PromptRows {
                prefill_rows,
                head_rows,
                first,
                count,
            } => (8, 0, prefill_rows, first, head_rows, 0, count),
            Command::Fork {
                from_slot,
                to_slot,
                slots,
                cow_source,
                cow_dest,
            } => {
                // `cow` rides in `flags` bit 0 because `None` and block 0 are both legal and
                // indistinguishable in a u32; the two page numbers are the header's two new words.
                cow = cow_source.zip(cow_dest);
                (5, 0, from_slot, to_slot, slots, u32::from(cow.is_some()), 0)
            }
            Command::DenseRow {
                phase,
                head_rows,
                row,
            } => (
                4,
                match phase {
                    Phase::Prefill => 0,
                    Phase::Decode => 1,
                },
                head_rows,
                row,
                0,
                0,
                0,
            ),
        };
        h[4..8].copy_from_slice(&op.to_le_bytes());
        h[8..12].copy_from_slice(&phase.to_le_bytes());
        h[12..16].copy_from_slice(&rows.to_le_bytes());
        h[16..20].copy_from_slice(&live.to_le_bytes());
        h[20..24].copy_from_slice(&slot.to_le_bytes());
        h[24..28].copy_from_slice(&flags.to_le_bytes());
        // Only the low 32 bits: the serial is a drift detector, not an identifier, and 4 billion
        // steps at 18 ms is two and a half years.
        h[28..32].copy_from_slice(&(serial as u32).to_le_bytes());
        // ws-h30's two words, at the END so that every field the 32-byte header had keeps its
        // offset and this diff is one addition rather than a re-layout.
        h[32..36].copy_from_slice(&VERSION.to_le_bytes());
        h[36..40].copy_from_slice(&state_slot.to_le_bytes());
        // V4's two words, at the END for the same reason ws-h30's were: every offset the 40-byte
        // header had keeps it, and this diff is one addition rather than a re-layout.
        let (cow_src, cow_dst) = cow.unwrap_or((0, 0));
        h[40..44].copy_from_slice(&cow_src.to_le_bytes());
        h[44..48].copy_from_slice(&cow_dst.to_le_bytes());
        h
    }

    fn decode(h: &[u8], want_serial: u64) -> Result<Command, ForwardError> {
        if h.len() < HEADER_BYTES {
            return Err(ForwardError::Peer(format!(
                "a command frame of {} bytes; the header is {HEADER_BYTES}",
                h.len()
            )));
        }
        let word = |at: usize| u32::from_le_bytes([h[at], h[at + 1], h[at + 2], h[at + 3]]);
        if word(0) != MAGIC {
            return Err(ForwardError::Peer(format!(
                "a command frame whose magic is {:#010x}, not {MAGIC:#010x}: the two ranks are out \
                 of frame",
                word(0)
            )));
        }
        // The VERSION before any field is read out of an offset it may not be at.
        let version = word(32);
        if version != VERSION {
            return Err(ForwardError::Peer(format!(
                "a command frame at protocol version {version}; this build speaks {VERSION} \
                 (a {HEADER_BYTES}-byte header with the prefill state slot). The two ranks are \
                 not the same build"
            )));
        }
        let serial = u64::from(word(28));
        if serial != want_serial as u32 as u64 {
            return Err(ForwardError::Peer(format!(
                "the leader is on step {serial} and this rank is on step {}: the pair has drifted",
                want_serial as u32
            )));
        }
        Ok(match word(4) {
            0 => Command::Ping,
            1 => Command::ClearSlot { slot: word(20) },
            2 => Command::Step {
                phase: if word(8) == 0 {
                    Phase::Prefill
                } else {
                    Phase::Decode
                },
                rows: word(12),
                width: word(20).max(1),
                live: word(16),
                samples: word(24) & 1 != 0,
                paged: word(24) & 2 != 0,
                seam: word(24) & 4 != 0,
                seam_penalties: word(24) & 8 != 0,
                defer_proposals: word(24) & 16 != 0,
                ragged: word(24) & 32 != 0,
                tree: word(24) & 64 != 0,
                slot: word(36),
            },
            3 => Command::Stop,
            4 => Command::DenseRow {
                phase: if word(8) == 0 {
                    Phase::Prefill
                } else {
                    Phase::Decode
                },
                head_rows: word(12),
                row: word(16),
            },
            5 => {
                let cow = word(24) & 1 != 0;
                Command::Fork {
                    from_slot: word(12),
                    to_slot: word(16),
                    slots: word(20).max(1),
                    cow_source: cow.then(|| word(40)),
                    cow_dest: cow.then(|| word(44)),
                }
            }
            6 => Command::ClearDraftPages {
                base: word(12),
                pages: word(16),
            },
            8 => Command::PromptRows {
                prefill_rows: word(12),
                first: word(16),
                head_rows: word(20),
                count: word(36),
            },
            7 => Command::Park {
                op: match word(12) {
                    0 => ParkOp::ParkBegin,
                    1 => ParkOp::ParkPoll,
                    2 => ParkOp::RestoreBegin,
                    3 => ParkOp::RestorePoll,
                    4 => ParkOp::Forget,
                    other => {
                        return Err(ForwardError::Peer(format!(
                            "a park frame with sub-op {other}; this build speaks 0..=4"
                        )))
                    }
                },
            },
            9 => Command::Propose {
                store_first: word(12) != 0,
                inline: word(16) != 0,
                carry: (word(24) & 1 != 0)
                    .then(|| u64::from(word(20)) | (u64::from(word(36)) << 32)),
            },
            10 => Command::ForkDraft {
                from_first: word(12),
                to_first: word(16),
                run: word(20),
                copy: word(36),
            },
            11 => Command::Vision {
                images: word(12),
                patches: word(16),
            },
            12 => Command::Scatter { runs: word(12) },
            13 => Command::Mixed {
                rung: word(12),
                chunk_live: word(16),
                chunk_samples: word(24) & 1 != 0,
                slot: word(36),
                decode_rows: word(20),
                width: word(8).max(1),
                decode_live: word(40),
                paged: word(24) & 2 != 0,
                chunk_seam: word(24) & 4 != 0,
                chunk_seam_penalties: word(24) & 8 != 0,
                defer_proposals: word(24) & 16 != 0,
                decode_seam: word(24) & 64 != 0,
                decode_seam_penalties: word(24) & 128 != 0,
                packed: word(44),
            },
            14 => Command::Prefix {
                op: word(8),
                entry: word(12),
                slot: word(16),
                draft_first: word(20),
                draft_pages: word(24),
                committed: word(36),
            },
            15 => Command::GdnMaterialize {
                par: word(8),
                rows: word(12),
                tag: word(16),
                slot: word(20),
            },
            other => {
                return Err(ForwardError::Peer(format!(
                    "a command frame with op {other}; this build speaks 0..=15 at protocol \
                     version {VERSION}"
                )))
            }
        })
    }
}

/// One row's local answer: which column of **this rank's** vocabulary shard won, and with what.
///
/// `index == cols` is the argmax kernel's "every logit was NaN" answer, and it means this rank has
/// nothing to offer for that row — which is a fact the other rank can still act on, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    /// The winning column within this rank's shard, or `cols` for "no finite logit".
    pub index: u32,
    /// Its logit, as `f32` bits (the bf16 value widened, which is exact).
    pub bits: u32,
}

impl Pick {
    /// The "this rank has no finite logit for this row" answer.
    #[must_use]
    pub fn none(cols: u32) -> Pick {
        Pick {
            index: cols,
            bits: f32::NEG_INFINITY.to_bits(),
        }
    }

    /// The `f32` this row won with, or `-inf` when nothing did.
    #[must_use]
    pub fn value(&self) -> f32 {
        f32::from_bits(self.bits)
    }

    fn write(&self, out: &mut [u8]) {
        out[0..4].copy_from_slice(&self.index.to_le_bytes());
        out[4..8].copy_from_slice(&self.bits.to_le_bytes());
    }

    fn read(b: &[u8]) -> Pick {
        Pick {
            index: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            bits: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        }
    }
}

/// Pack one rank's picks for the wire.
#[must_use]
pub fn pack_picks(picks: &[Pick]) -> Vec<u8> {
    let mut out = vec![0u8; picks.len() * PICK_BYTES];
    for (p, slot) in picks.iter().zip(out.chunks_exact_mut(PICK_BYTES)) {
        p.write(slot);
    }
    out
}

/// Unpack the peer's picks.
///
/// # Errors
/// A buffer that is not a whole number of picks.
pub fn unpack_picks(bytes: &[u8]) -> Result<Vec<Pick>, ForwardError> {
    if bytes.len() % PICK_BYTES != 0 {
        return Err(ForwardError::Peer(format!(
            "{} pick byte(s) is not a multiple of {PICK_BYTES}",
            bytes.len()
        )));
    }
    Ok(bytes.chunks_exact(PICK_BYTES).map(Pick::read).collect())
}

/// The logits gather, on the host: the global argmax of a row whose vocabulary is split in two.
///
/// `rank` is whose `mine` is, `cols` is the shard width (`vocab / 2`), and the answer is a **global**
/// token id — rank *r*'s column *c* is token `r * cols + c`. Ties go to the lower global id, which is
/// what `torch.argmax` does over a concatenation and is the only tie rule that does not depend on
/// which rank ran first.
///
/// # Errors
/// Mismatched row counts, a column outside a shard, or a row on which **neither** rank had a finite
/// logit — the same condition [`crate::device`] refuses on one card, which is the forward pass
/// producing no number rather than a sampling question.
pub fn combine(
    rank: u32,
    cols: u32,
    mine: &[Pick],
    theirs: &[Pick],
) -> Result<Vec<TokenId>, ForwardError> {
    if mine.len() != theirs.len() {
        return Err(ForwardError::Peer(format!(
            "this rank picked {} row(s) and the peer picked {}",
            mine.len(),
            theirs.len()
        )));
    }
    let peer = 1 - rank;
    let mut out = Vec::with_capacity(mine.len());
    for (row, (m, t)) in mine.iter().zip(theirs).enumerate() {
        for (who, p) in [(rank, m), (peer, t)] {
            if p.index > cols {
                return Err(ForwardError::Peer(format!(
                    "row {row}: rank {who} picked column {} of a {cols}-wide shard",
                    p.index
                )));
            }
        }
        let live = |p: &Pick| p.index < cols && !p.value().is_nan();
        let id = match (live(m), live(t)) {
            (false, false) => {
                return Err(ForwardError::Device(format!(
                    "row {row}: neither vocabulary shard had a finite logit — the forward pass \
                     produced no number, which is the argmax kernel's NaN answer on both cards"
                )))
            }
            (true, false) => rank * cols + m.index,
            (false, true) => peer * cols + t.index,
            (true, true) => {
                let (mine_id, theirs_id) = (rank * cols + m.index, peer * cols + t.index);
                match m.value().partial_cmp(&t.value()) {
                    Some(std::cmp::Ordering::Greater) => mine_id,
                    Some(std::cmp::Ordering::Less) => theirs_id,
                    // Equal, or incomparable because one of them is a NaN `live` let through: the
                    // lower id, always, so both ranks compute the same token from the same pair.
                    _ => mine_id.min(theirs_id),
                }
            }
        };
        out.push(id as TokenId);
    }
    Ok(out)
}

// ── the sampler's seam, across the pair ──────────────────────────────────────────────────────

/// `SBS3`, so a seam frame that is not one says so instead of being decoded as logits.
const SEAM_MAGIC: u32 = 0x5342_5333;

/// Bytes of the header every seam frame opens with: magic, version, rows, and the geometry word.
pub const SEAM_HEADER_BYTES: usize = 16;

/// The seam frames' own version, checked by the rank that RECEIVES one.
///
/// [`VERSION`] is in every command frame and is agreed at boot; this catches it
/// in the frame, which is what a rank that was handed a socket it did not open has. They move
/// together and both are bumped by any change to the layout below.
pub const SEAM_FRAME_VERSION: u32 = 1;
/// The seam's geometry, as both ranks compute it from the plan they built identically.
///
/// It is not negotiated: `k` is `saber_kernels::head::DEVICE_ROW_K` and `penalties` is the head's
/// own kernarg, so a pair whose binaries disagree about either disagrees about
/// [`VERSION`] first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamGeometry {
    /// Candidates per row, per shard.
    pub k: u32,
    /// Penalty ids per row the head gathers.
    pub penalties: u32,
}

impl SeamGeometry {
    /// Bytes the leader's frame occupies for `rows` rows.
    #[must_use]
    pub fn input_bytes(&self, rows: u32, penalties: bool) -> usize {
        let rows = rows as usize;
        SEAM_HEADER_BYTES
            + rows * 4
            + if penalties {
                rows * self.penalties as usize * 4
            } else {
                0
            }
    }

    /// Bytes the follower's frame occupies for `rows` rows.
    #[must_use]
    pub fn output_bytes(&self, rows: u32, penalties: bool) -> usize {
        let rows = rows as usize;
        let k = self.k as usize;
        SEAM_HEADER_BYTES
            + rows * k * 4 // values
            + rows * k * 4 // ids
            + rows * 2 * 4 // the two log-sum-exps
            + if penalties {
                rows * self.penalties as usize * 4
            } else {
                0
            }
    }

    fn header(&self, rows: u32, kind: u32) -> [u8; SEAM_HEADER_BYTES] {
        let mut h = [0u8; SEAM_HEADER_BYTES];
        h[0..4].copy_from_slice(&SEAM_MAGIC.to_le_bytes());
        h[4..8].copy_from_slice(&((SEAM_FRAME_VERSION << 8) | kind).to_le_bytes());
        h[8..12].copy_from_slice(&rows.to_le_bytes());
        h[12..16].copy_from_slice(&((self.k << 16) | (self.penalties & 0xffff)).to_le_bytes());
        h
    }

    /// Check a received frame's header against what this rank believes the step is.
    ///
    /// # Errors
    /// A foreign frame, a version this build does not speak, or a geometry that is not this
    /// rank's — each named, because "the two ranks are out of frame" is the failure this whole
    /// header exists to turn into a sentence.
    fn check(&self, h: &[u8], rows: u32, kind: u32) -> Result<(), ForwardError> {
        if h.len() < SEAM_HEADER_BYTES {
            return Err(ForwardError::Peer(format!(
                "a seam frame of {} byte(s); the header alone is {SEAM_HEADER_BYTES}",
                h.len()
            )));
        }
        let word = |at: usize| u32::from_le_bytes([h[at], h[at + 1], h[at + 2], h[at + 3]]);
        if word(0) != SEAM_MAGIC {
            return Err(ForwardError::Peer(format!(
                "a seam frame whose magic is {:#010x}, not {SEAM_MAGIC:#010x}: the two ranks are \
                 out of frame",
                word(0)
            )));
        }
        let (version, got_kind) = (word(4) >> 8, word(4) & 0xff);
        if version != SEAM_FRAME_VERSION {
            return Err(ForwardError::Peer(format!(
                "the peer speaks seam frame version {version} and this build speaks \
                 {SEAM_FRAME_VERSION}"
            )));
        }
        if got_kind != kind {
            return Err(ForwardError::Peer(format!(
                "a seam frame of kind {got_kind} where kind {kind} was expected"
            )));
        }
        if word(8) != rows {
            return Err(ForwardError::Peer(format!(
                "a seam frame for {} row(s) and this step has {rows}",
                word(8)
            )));
        }
        let (k, pen) = (word(12) >> 16, word(12) & 0xffff);
        if k != self.k || pen != self.penalties {
            return Err(ForwardError::Peer(format!(
                "the peer's seam is K = {k} with {pen} penalty id(s) and this rank's is K = {} \
                 with {}",
                self.k, self.penalties
            )));
        }
        Ok(())
    }
}

/// What the leader tells the other rank about the rows it is sampling.
///
/// The follower has no scheduler and no requests, so every one of these is a fact only rank 0
/// holds. `pen_ids` carries **global** token ids: each rank maps them to its own columns, which
/// is one subtraction and keeps the frame independent of who is reading it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeamInputs {
    /// One temperature per ROW of the head, `0.0` for a row the kernel is to skip.
    pub temps: Vec<f32>,
    /// `rows * penalties` global ids, `-1` padded. Empty when no row is penalised.
    pub pen_ids: Vec<i32>,
}

/// One rank's shard of every sampled row, as the kernel wrote it.
///
/// The ids are **columns of that rank's shard**, not token ids: turning them into tokens is the
/// merge's job and doing it twice is how a base gets added twice
/// (`saber_serve::sampler::merge_shards`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeamOutputs {
    /// `rows * k` values, value descending within each row.
    pub values: Vec<f32>,
    /// `rows * k` columns, in the same order.
    pub ids: Vec<i32>,
    /// `rows * 2`: each row's log-sum-exp at T = 1 and at its own temperature.
    pub lse: Vec<f32>,
    /// `rows * penalties` raw logits at the ids the frame above carried, or empty.
    pub pen: Vec<f32>,
}

fn put_f32(out: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn put_i32(out: &mut Vec<u8>, xs: &[i32]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn take_f32(b: &[u8], at: &mut usize, n: usize) -> Vec<f32> {
    let end = (*at + n * 4).min(b.len());
    let out = b[*at..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *at = end;
    out
}

fn take_i32(b: &[u8], at: &mut usize, n: usize) -> Vec<i32> {
    let end = (*at + n * 4).min(b.len());
    let out = b[*at..end]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *at = end;
    out
}

/// Magic of an `engine.park` frame: it is its own message and not a command header.
const PARK_MAGIC: u32 = 0x5342_5450;

/// Fixed bytes at the front of an `engine.park` frame, before the refusal text and the page list.
pub const PARK_HEAD: usize = 96;

/// Bytes of refusal text a park frame carries. A refusal that arrives truncated is still a
/// refusal that names its rank and its op, which is what the leader reports.
pub const PARK_MSG: usize = 256;

/// How one rank answered a [`Command::Park`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParkStatus {
    /// The leader's outbound frame: a question, not an answer.
    Ask,
    /// Accepted; nothing to report yet.
    Ok,
    /// In flight, come back next poll.
    Pending,
    /// Finished. The digest and the byte count are this rank's half of the image.
    Done,
    /// Refused, with the reason in `message`.
    Refused,
}

/// The `engine.park` frame: a session's description on the way out and one rank's answer on the
/// way back, in one symmetric swap.
///
/// Every field in the first half is something **rank 1 cannot derive**, which is the same test
/// every other field of this protocol passes: it keeps no session state, so the slot, the page
/// list, the drafter's page run and the resident token count are the leader's to state. The
/// second half is the answer, and the pair of `bytes`/`digest` is what makes a restore checkable
/// — a follower that came back with a different image than it parked is refused by name instead
/// of publishing it over a live session's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParkFrame {
    /// The sequence, as the leader names it.
    pub seq: u64,
    /// Its GDN state slot — the BASE of its run of snapshots, the same number on both cards.
    pub slot: u32,
    /// Tokens resident when the park was issued. Part of the image on both ranks.
    pub resident: u32,
    /// Tokens the session's last step committed, `1..=max_width` (ws-k14).
    ///
    /// **Which slot of the run each rank's GDN regions are**: the conv cache is the run's base and
    /// the recurrent snapshot is `base + committed - 1` (`docs/EXEC.md` § 2). Rank 1 keeps no
    /// sessions, so like the slot and the page list this is the leader's to state, and both halves
    /// of an image are then placed by one piece of arithmetic.
    pub committed: u32,
    /// This session's first page in the DRAFTER's pool, when there is a resident drafter.
    pub draft_first_page: u32,
    /// Pages in its drafter run.
    pub draft_pages: u32,
    /// Whether the drafter's half is part of this image at all.
    pub has_draft: bool,
    /// The session's KV pages, in order — its block table, which is the same on both ranks
    /// because the leader's allocator hands out page numbers and both ranks index their own shard
    /// with them.
    pub blocks: Vec<u32>,
    /// The answer.
    pub status: ParkStatus,
    /// Bytes of this rank's half, once it is `Done`.
    pub bytes: u64,
    /// Its digest (`saber_mem::park::ImageDigests::sha256`), once it is `Done`.
    pub digest: [u8; 32],
    /// A refusal's reason.
    pub message: String,
}

impl ParkFrame {
    /// The leader's question about `seq`.
    #[must_use]
    pub fn ask(seq: u64, slot: u32, resident: u32, committed: u32, blocks: Vec<u32>) -> ParkFrame {
        ParkFrame {
            seq,
            slot,
            resident,
            committed,
            draft_first_page: 0,
            draft_pages: 0,
            has_draft: false,
            blocks,
            status: ParkStatus::Ask,
            bytes: 0,
            digest: [0u8; 32],
            message: String::new(),
        }
    }

    /// The same question, with the drafter's page run in it.
    #[must_use]
    pub fn with_draft(mut self, first_page: u32, pages: u32) -> ParkFrame {
        self.draft_first_page = first_page;
        self.draft_pages = pages;
        self.has_draft = true;
        self
    }

    /// An answer that carries no image: `Ok` or `Pending`.
    #[must_use]
    pub fn answer(seq: u64, status: ParkStatus) -> ParkFrame {
        ParkFrame {
            status,
            ..ParkFrame::ask(seq, 0, 0, 1, Vec::new())
        }
    }

    /// The follower's answer to a `Forget` (ws-k18): whether it held a half under `seq` (in
    /// `slot`, 1 or 0) and how many halves it still holds after dropping it (in `bytes`). An
    /// answer frame carries no session, so the two words are free; [`ParkFrame::forgotten`] is
    /// the one reader.
    #[must_use]
    pub fn forgotten_answer(seq: u64, held: bool, remaining: u64) -> ParkFrame {
        ParkFrame {
            slot: u32::from(held),
            bytes: remaining,
            ..ParkFrame::answer(seq, ParkStatus::Ok)
        }
    }

    /// `(held, remaining)` out of a [`ParkFrame::forgotten_answer`].
    #[must_use]
    pub fn forgotten(&self) -> (bool, u64) {
        (self.slot != 0, self.bytes)
    }

    /// A finished half.
    #[must_use]
    pub fn done(seq: u64, bytes: u64, digest: [u8; 32]) -> ParkFrame {
        ParkFrame {
            status: ParkStatus::Done,
            bytes,
            digest,
            ..ParkFrame::ask(seq, 0, 0, 1, Vec::new())
        }
    }

    /// A refusal, with its reason.
    #[must_use]
    pub fn refused(seq: u64, message: impl Into<String>) -> ParkFrame {
        ParkFrame {
            status: ParkStatus::Refused,
            message: message.into(),
            // `committed` is 1 rather than 0: it is `naccept`, which is never 0, and an answer
            // frame carries no session anyway.
            ..ParkFrame::ask(seq, 0, 0, 1, Vec::new())
        }
    }

    /// Bytes on the wire, given the room a link laid out for the page list.
    ///
    /// # Errors
    /// A page list that does not fit: the link sizes the frame from the block-table region both
    /// ranks already agreed on, so this is a session with more pages than the plan can hold and it
    /// is named rather than truncated into another session's pages.
    pub fn encode(&self, width: usize) -> Result<Vec<u8>, ForwardError> {
        let want = PARK_HEAD + PARK_MSG + self.blocks.len() * 4;
        if want > width {
            return Err(ForwardError::Peer(format!(
                "a park frame for {} page(s) needs {want} bytes and this link agreed {width}",
                self.blocks.len()
            )));
        }
        let mut f = vec![0u8; width];
        f[0..4].copy_from_slice(&PARK_MAGIC.to_le_bytes());
        f[4..8].copy_from_slice(&VERSION.to_le_bytes());
        f[8..12].copy_from_slice(
            &match self.status {
                ParkStatus::Ask => 0u32,
                ParkStatus::Ok => 1,
                ParkStatus::Pending => 2,
                ParkStatus::Done => 3,
                ParkStatus::Refused => 4,
            }
            .to_le_bytes(),
        );
        f[12..16].copy_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        f[16..24].copy_from_slice(&self.seq.to_le_bytes());
        f[24..28].copy_from_slice(&self.slot.to_le_bytes());
        f[28..32].copy_from_slice(&self.resident.to_le_bytes());
        f[32..36].copy_from_slice(&self.draft_first_page.to_le_bytes());
        f[36..40].copy_from_slice(&self.draft_pages.to_le_bytes());
        f[40..44].copy_from_slice(&u32::from(self.has_draft).to_le_bytes());
        let msg = self.message.as_bytes();
        let msg_len = msg.len().min(PARK_MSG);
        f[44..48].copy_from_slice(&(msg_len as u32).to_le_bytes());
        f[48..56].copy_from_slice(&self.bytes.to_le_bytes());
        f[56..88].copy_from_slice(&self.digest);
        f[88..92].copy_from_slice(&self.committed.to_le_bytes());
        f[PARK_HEAD..PARK_HEAD + msg_len].copy_from_slice(&msg[..msg_len]);
        let at = PARK_HEAD + PARK_MSG;
        for (i, b) in self.blocks.iter().enumerate() {
            f[at + i * 4..at + i * 4 + 4].copy_from_slice(&b.to_le_bytes());
        }
        Ok(f)
    }

    /// Read one back.
    ///
    /// # Errors
    /// A frame that is not one, a frame from another build, or a page count the frame cannot hold.
    pub fn decode(f: &[u8]) -> Result<ParkFrame, ForwardError> {
        if f.len() < PARK_HEAD + PARK_MSG {
            return Err(ForwardError::Peer(format!(
                "a park frame of {} bytes; the header is {}",
                f.len(),
                PARK_HEAD + PARK_MSG
            )));
        }
        let word = |at: usize| u32::from_le_bytes([f[at], f[at + 1], f[at + 2], f[at + 3]]);
        if word(0) != PARK_MAGIC {
            return Err(ForwardError::Peer(format!(
                "a park frame whose magic is {:#010x}, not {PARK_MAGIC:#010x}",
                word(0)
            )));
        }
        if word(4) != VERSION {
            return Err(ForwardError::Peer(format!(
                "a park frame at protocol version {}; this build speaks {VERSION}",
                word(4)
            )));
        }
        let status = match word(8) {
            0 => ParkStatus::Ask,
            1 => ParkStatus::Ok,
            2 => ParkStatus::Pending,
            3 => ParkStatus::Done,
            4 => ParkStatus::Refused,
            other => {
                return Err(ForwardError::Peer(format!(
                    "a park frame with status {other}; this build speaks 0..=4"
                )))
            }
        };
        let n = word(12) as usize;
        let at = PARK_HEAD + PARK_MSG;
        if at + n * 4 > f.len() {
            return Err(ForwardError::Peer(format!(
                "a park frame claiming {n} page(s) in {} bytes",
                f.len()
            )));
        }
        let msg_len = (word(44) as usize).min(PARK_MSG);
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&f[56..88]);
        Ok(ParkFrame {
            seq: u64::from_le_bytes([f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23]]),
            slot: word(24),
            resident: word(28),
            committed: word(88),
            draft_first_page: word(32),
            draft_pages: word(36),
            has_draft: word(40) != 0,
            blocks: (0..n).map(|i| word(at + i * 4)).collect(),
            status,
            bytes: u64::from_le_bytes([f[48], f[49], f[50], f[51], f[52], f[53], f[54], f[55]]),
            digest,
            message: String::from_utf8_lossy(&f[PARK_HEAD..PARK_HEAD + msg_len]).into_owned(),
        })
    }
}

/// **THE ACK a one-way frame is answered with**: this rank's step serial, eight bytes.
///
/// Every frame on this wire used to be SYMMETRIC, because `swap_bytes` is: the side with nothing
/// to say answered with a buffer of zeros as wide as the side that had something. At `max_seqs` 8
/// that made the step frame **1,932 KiB in each direction** to carry a leader-to-follower payload
/// (ws-h29's own census, and its "what is still on this wire" paragraph), and it cost a whole
/// round trip per call agreeing a length both ranks already knew.
///
/// `IpcExchange::xchg_bytes` carries its own length and needs no symmetry, so the passive side of
/// a one-way frame answers with this instead. It is still a RENDEZVOUS — neither rank returns
/// until both have sent — which is the property the step depends on: both ranks' captured graphs
/// hold each other's one-shot all-reduces, and a rank that ran ahead would leave the other's
/// bounded spin to expire (`docs/EXEC.md` §4).
const ACK_BYTES: usize = 8;

/// One rank's end of the step channel.
///
/// Holds the serial both ranks count steps with, which is the whole of the drift detection: a
/// follower that decodes a header from a step it is not on refuses instead of running the wrong
/// shape over the right buffers.
pub struct RankLink {
    peer: Arc<RankPeer>,
    rank: u32,
    payload_bytes: usize,
    paged_bytes: usize,
    park_bytes: usize,
    seam: SeamGeometry,
    serial: u64,
    /// Payload bytes the last `follow` brought in, for a caller that wants to say what crossed.
    last_payload: usize,
    /// P27-14: the host levers both ranks run (forwarded by value) and their counters.
    levers: crate::host_levers::HostLevers,
    lever_counts: Arc<crate::host_levers::HostLeverCounters>,
    /// P27-14 (`table`): a whole block-table frame has crossed on this link (the induction base
    /// every compact frame rests on).
    paged_whole_sent: bool,
    /// P27-14 (`socket`, leader): a `Command::Propose` is owed and rides the next drafter frame.
    propose_owed: bool,
    /// P27-14 (`socket`, leader): the owed proposal may mark its frame solo (no terminator).
    solo_ok: bool,
    /// P27-14 (`socket`): the last single proposal's block digest, carried to the next
    /// `Command::Propose` (both ranks keep their own).
    carry: Option<u64>,
}

impl std::fmt::Debug for RankLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankLink")
            .field("rank", &self.rank)
            .field("payload_bytes", &self.payload_bytes)
            .field("serial", &self.serial)
            .finish()
    }
}

impl RankLink {
    /// Wrap the pair's socket. `payload_bytes` is the step frame's **constant** width — the index
    /// tables every step rewrites plus the ids — and `paged_bytes` is the width of the second frame
    /// a page-crossing step sends. Both are sized at startup from plans both ranks built
    /// identically, so neither is negotiated.
    #[must_use]
    pub fn new(
        peer: Arc<RankPeer>,
        payload_bytes: usize,
        paged_bytes: usize,
        seam: SeamGeometry,
    ) -> RankLink {
        let rank = peer.rank();
        RankLink {
            peer,
            rank,
            payload_bytes,
            paged_bytes,
            // The park frame is sized from the block-table region both ranks already agreed on,
            // so it is not negotiated either: that region is `max_reqs` rows of `max_blocks`
            // pages, and one session's page list is one row of it (ws-t2).
            park_bytes: PARK_HEAD + PARK_MSG + paged_bytes,
            seam,
            serial: 0,
            last_payload: 0,
            levers: crate::host_levers::HostLevers::default(),
            lever_counts: Arc::new(crate::host_levers::HostLeverCounters::default()),
            paged_whole_sent: false,
            propose_owed: false,
            solo_ok: false,
            carry: None,
        }
    }

    /// P27-14: run the host levers on this link (both ranks, by value) with these counters.
    pub fn set_levers(
        &mut self,
        levers: crate::host_levers::HostLevers,
        counts: Arc<crate::host_levers::HostLeverCounters>,
    ) {
        self.levers = levers;
        self.lever_counts = counts;
    }

    /// The host levers this link runs.
    #[must_use]
    pub fn levers(&self) -> crate::host_levers::HostLevers {
        self.levers
    }

    /// The host levers' counters.
    #[must_use]
    pub fn lever_counts(&self) -> &crate::host_levers::HostLeverCounters {
        &self.lever_counts
    }

    /// P27-14 (`socket`, leader): owe the follower a `Command::Propose` that the next drafter
    /// frame will carry. `solo` = that proposal may fold its terminator.
    pub fn owe_propose(&mut self, solo: bool) {
        self.propose_owed = true;
        self.solo_ok = solo;
    }

    /// P27-14 (`socket`, leader): whether the proposal running now may be marked solo; reading
    /// it clears it (one proposal a `Command::Propose`).
    pub fn take_solo_ok(&mut self) -> bool {
        std::mem::take(&mut self.solo_ok)
    }

    /// P27-14 (`socket`): carry this rank's single-proposal digest to the next `Propose`.
    pub fn set_carry(&mut self, digest: u64) {
        self.carry = Some(digest);
    }

    /// P27-14 (`socket`): the carried digest, consumed.
    pub fn take_carry(&mut self) -> Option<u64> {
        self.carry.take()
    }

    /// P27-14 (`socket`, follower): check the leader's carried digest against this rank's own.
    ///
    /// # Errors
    /// The two ranks proposed different blocks, or only one of them carried a digest.
    pub fn check_carry(&mut self, theirs: Option<u64>) -> Result<(), ForwardError> {
        let mine = self.carry.take();
        if theirs != mine {
            return Err(ForwardError::Peer(format!(
                "the two ranks proposed different blocks: the leader carried {theirs:?} and this                  rank {mine:?} (P27-14's folded terminator)"
            )));
        }
        if mine.is_some() {
            self.lever_counts
                .carry_checked
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// This rank's answer to a one-way frame: the step serial, so a pair that has drifted says so
    /// at the frame rather than at the next collective.
    fn ack(&self) -> [u8; ACK_BYTES] {
        self.serial.to_le_bytes()
    }

    /// Check the peer's ack, by name.
    fn check_ack(&self, got: &[u8], what: &str) -> Result<(), ForwardError> {
        if got.len() != ACK_BYTES {
            return Err(ForwardError::Peer(format!(
                "{what} at step {}: the peer answered {} byte(s) and an ack is {ACK_BYTES}",
                self.serial,
                got.len()
            )));
        }
        let theirs = u64::from_le_bytes([
            got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7],
        ]);
        if theirs != self.serial {
            return Err(ForwardError::Peer(format!(
                "{what}: this rank is on step {} and the peer answered from step {theirs}",
                self.serial
            )));
        }
        Ok(())
    }

    /// Payload bytes the last followed frame carried.
    #[must_use]
    pub fn last_payload_bytes(&self) -> usize {
        self.last_payload
    }

    /// Bytes an `engine.park` frame occupies on this link.
    #[must_use]
    pub fn park_bytes(&self) -> usize {
        self.park_bytes
    }

    /// Leader: one whole park op — the command, this rank's descriptor, and the follower's
    /// answer.
    ///
    /// **Two park frames and not one**, because the exchange is symmetric: both ranks put their
    /// bytes in at once, so the follower cannot answer a question it has not read yet. The first
    /// swap carries the question out (and brings a placeholder back), the second brings the answer
    /// in (and carries a placeholder out). At a few kilobytes a frame that is the cheapest half of
    /// a park by three orders of magnitude, and it keeps `swap_bytes`' symmetry law intact rather
    /// than adding a second, asymmetric transport.
    ///
    /// # Errors
    /// The socket, or a frame that is not the agreed width.
    pub fn park_ask(&mut self, op: ParkOp, ask: &ParkFrame) -> Result<ParkFrame, ForwardError> {
        // ENCODED BEFORE THE COMMAND IS LED, because a frame that does not fit is a refusal this
        // rank can still take back: once the header has crossed, the follower is waiting for two
        // park frames and a leader that failed between them leaves it there until its socket
        // deadline. Every failure after this point is the socket's.
        let bytes = ask.encode(self.park_bytes)?;
        self.lead(Command::Park { op }, &[])?;
        let _ = self
            .peer
            .exchange()
            .swap_bytes("engine.park", &bytes)
            .map_err(|e| ForwardError::Peer(format!("swapping a park frame: {e}")))?;
        self.swap_park(&ParkFrame::answer(ask.seq, ParkStatus::Ask))
    }

    /// Follower: read the descriptor the [`Command::Park`] it just followed announced.
    ///
    /// # Errors
    /// The socket, or a frame this build cannot decode.
    pub fn park_take(&self) -> Result<ParkFrame, ForwardError> {
        self.swap_park(&ParkFrame::answer(0, ParkStatus::Ask))
    }

    /// Follower: answer it.
    ///
    /// # Errors
    /// The socket, or an answer that is not the agreed width.
    pub fn park_reply(&self, reply: &ParkFrame) -> Result<(), ForwardError> {
        self.swap_park(reply).map(|_| ())
    }

    /// Swap one `engine.park` frame: the leader's question for the follower's answer, or the
    /// other way round.
    ///
    /// Symmetric like every other exchange here — `swap_bytes` refuses two different widths — so
    /// the follower answers into a frame of the same size it was asked in.
    ///
    /// # Errors
    /// The socket, or a frame that is not the agreed width.
    pub fn swap_park(&self, mine: &ParkFrame) -> Result<ParkFrame, ForwardError> {
        let bytes = mine.encode(self.park_bytes)?;
        let got = self
            .peer
            .exchange()
            .swap_bytes("engine.park", &bytes)
            .map_err(|e| ForwardError::Peer(format!("swapping a park frame: {e}")))?;
        ParkFrame::decode(&got)
    }

    /// The seam geometry both ranks laid out.
    #[must_use]
    pub fn seam(&self) -> SeamGeometry {
        self.seam_geometry()
    }

    /// **Leader**: send one frame under `name` and take the follower's ack (D4: the pixel frame
    /// and the scatter runs).
    ///
    /// # Errors
    /// The socket, or an ack that is not this step's.
    pub fn lead_bytes(&self, name: &str, bytes: &[u8]) -> Result<(), ForwardError> {
        let ack =
            self.peer.exchange().xchg_bytes(name, bytes).map_err(|e| {
                ForwardError::Peer(format!("`{name}` at step {}: {e}", self.serial))
            })?;
        self.check_ack(&ack, name)
    }

    /// **Follower**: take the frame the leader's [`RankLink::lead_bytes`] sent under `name`.
    ///
    /// # Errors
    /// The socket.
    pub fn follow_bytes(&self, name: &str) -> Result<Vec<u8>, ForwardError> {
        self.peer
            .exchange()
            .xchg_bytes(name, &self.ack())
            .map_err(|e| ForwardError::Peer(format!("`{name}` at step {}: {e}", self.serial)))
    }

    /// Both ranks: swap one u64 (D4: the digest of the tower rows each rank produced).
    ///
    /// # Errors
    /// The socket, or an answer that is not eight bytes.
    pub fn swap_u64(&self, name: &str, mine: u64) -> Result<u64, ForwardError> {
        let got = self
            .peer
            .exchange()
            .xchg_bytes(name, &mine.to_le_bytes())
            .map_err(|e| ForwardError::Peer(format!("`{name}` at step {}: {e}", self.serial)))?;
        let b: [u8; 8] = got.as_slice().try_into().map_err(|_| {
            ForwardError::Peer(format!("`{name}`: the peer answered {} byte(s)", got.len()))
        })?;
        Ok(u64::from_le_bytes(b))
    }

    fn seam_geometry(&self) -> SeamGeometry {
        self.seam
    }

    /// This rank.
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Steps exchanged so far — the serial the next frame carries.
    #[must_use]
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// The peer, for the rendezvous and digest agreement the collectives need.
    #[must_use]
    pub fn peer(&self) -> &Arc<RankPeer> {
        &self.peer
    }

    /// Leader: tell the other rank what this step is, and hand it the blob.
    ///
    /// # Errors
    /// The socket, or a payload wider than the widest this link laid out.
    pub fn lead(&mut self, cmd: Command, blob: &[u8]) -> Result<(), ForwardError> {
        // A CEILING, not a width. The frame used to be padded to `payload_bytes` whatever it
        // carried, because the exchange was symmetric and the follower could not know what it was
        // about to read; it carries its own length now, so what this checks is that a frame never
        // exceeds what both ranks laid out from their identical plans (ws-h31).
        let want = if cmd.has_payload() {
            self.payload_bytes
        } else {
            0
        };
        if blob.len() > want {
            return Err(ForwardError::Peer(format!(
                "a {cmd:?} frame with {} payload byte(s); this link's widest is {want}",
                blob.len()
            )));
        }
        let mut frame = Vec::with_capacity(HEADER_BYTES + blob.len());
        frame.extend_from_slice(&cmd.encode(self.serial));
        frame.extend_from_slice(blob);
        // THE FRAME IS THIS FRAME'S WIDTH, not the widest one. It was padded to
        // `HEADER_BYTES + payload_bytes` every time — a `Command::Ping` moved 1.9 MiB — because
        // `swap_bytes` is symmetric and the follower could not know what it was about to receive.
        // `xchg_bytes` carries the length, so a command with no payload is 48 bytes and a decode
        // step's is its own runs (ws-h31).
        let ack = self
            .peer
            .exchange()
            .xchg_bytes("engine.step", &frame)
            .map_err(|e| ForwardError::Peer(format!("leading step {}: {e}", self.serial)))?;
        self.check_ack(&ack, "leading a step")?;
        self.serial += 1;
        Ok(())
    }

    /// Follower: wait for the next command and copy its blob into `blob`.
    ///
    /// # Errors
    /// The socket, a frame this build cannot decode, or a serial that says the pair has drifted.
    pub fn follow(&mut self, blob: &mut [u8]) -> Result<Command, ForwardError> {
        if blob.len() < self.payload_bytes {
            return Err(ForwardError::Peer(format!(
                "a follower buffer of {} byte(s); this link's widest frame is {}",
                blob.len(),
                self.payload_bytes
            )));
        }
        let frame = self
            .peer
            .exchange()
            .xchg_bytes("engine.step", &self.ack())
            .map_err(|e| ForwardError::Peer(format!("following step {}: {e}", self.serial)))?;
        let cmd = Command::decode(&frame, self.serial)?;
        if cmd.has_payload() {
            let payload = &frame[HEADER_BYTES..];
            if payload.len() > blob.len() {
                return Err(ForwardError::Peer(format!(
                    "a step payload of {} byte(s); this rank's buffer is {}",
                    payload.len(),
                    blob.len()
                )));
            }
            blob[..payload.len()].copy_from_slice(payload);
            self.last_payload = payload.len();
        } else {
            // A payload-less command leaves the buffer alone and SAYS SO: a `last_payload` that
            // kept the last step's count would make a `Stop` look like a step that carried one.
            self.last_payload = 0;
        }
        self.serial += 1;
        Ok(cmd)
    }

    /// Move the `block_table` region, on the steps whose [`Command::Step::paged`] says it moved.
    ///
    /// The leader passes its own bytes and ignores what comes back; the follower passes a buffer of
    /// zeros and keeps what comes back. That is the same half-measure `lead`/`follow` make — the
    /// exchange is symmetric by contract, so one direction is always wasted — and it is why this is
    /// a SEPARATE frame rather than part of the step's: the waste is paid on the steps that need it
    /// and not on the fifteen in sixteen that do not.
    ///
    /// # Errors
    /// The socket, or a buffer that is not the width both ranks laid out.
    pub fn swap_paged(&self, mine: &[u8]) -> Result<Vec<u8>, ForwardError> {
        if mine.len() != self.paged_bytes {
            return Err(ForwardError::Peer(format!(
                "a paged frame of {} byte(s); this link agreed {}",
                mine.len(),
                self.paged_bytes
            )));
        }
        let ack = self
            .peer
            .exchange()
            .xchg_bytes("engine.blocktable", mine)
            .map_err(|e| {
                ForwardError::Peer(format!("the block table at step {}: {e}", self.serial))
            })?;
        self.check_ack(&ack, "the block table")?;
        Ok(Vec::new())
    }

    /// P27-14 (`table`, leader): a COMPACT block-table frame (`host_levers::encode_paged`), only
    /// after a whole frame has crossed on this link.
    ///
    /// # Errors
    /// The socket, a frame that is not compact, or no whole frame yet.
    pub fn swap_paged_compact(&self, frame: &[u8]) -> Result<(), ForwardError> {
        if !self.paged_whole_sent
            || !crate::host_levers::is_compact(frame, self.paged_bytes)
            || !self.levers.table
        {
            return Err(ForwardError::Peer(
                "a compact block-table frame before the link's whole one, or without the lever"
                    .to_owned(),
            ));
        }
        let ack = self
            .peer
            .exchange()
            .xchg_bytes("engine.blocktable", frame)
            .map_err(|e| {
                ForwardError::Peer(format!("the block table at step {}: {e}", self.serial))
            })?;
        self.check_ack(&ack, "the block table")
    }

    /// P27-14 (`table`): a whole block-table frame crossed (either rank notes it).
    pub fn note_paged_whole(&mut self) {
        self.paged_whole_sent = true;
    }

    /// P27-14 (`table`): whether a whole block-table frame has crossed on this link.
    #[must_use]
    pub fn paged_whole_sent(&self) -> bool {
        self.paged_whole_sent
    }

    /// The agreed width of a whole block-table frame.
    #[must_use]
    pub fn paged_bytes(&self) -> usize {
        self.paged_bytes
    }

    /// **Follower**: take the block table the leader says moved, answering with an ack.
    ///
    /// # Errors
    /// The socket, or a frame that is not the width both ranks laid out.
    pub fn follow_paged(&self) -> Result<Vec<u8>, ForwardError> {
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.blocktable", &self.ack())
            .map_err(|e| {
                ForwardError::Peer(format!("the block table at step {}: {e}", self.serial))
            })?;
        // P27-14 (`table`): a compact frame is shorter by construction and starts with its
        // magic, and is accepted only on a link whose whole frame has crossed.
        let compact = self.levers.table
            && self.paged_whole_sent
            && crate::host_levers::is_compact(&got, self.paged_bytes);
        if got.len() != self.paged_bytes && !compact {
            return Err(ForwardError::Peer(format!(
                "a paged frame of {} byte(s); this link agreed {}",
                got.len(),
                self.paged_bytes
            )));
        }
        Ok(got)
    }

    /// **Leader**: hand the other rank the temperatures (and the penalty ids) of the rows this
    /// step samples, BEFORE either replays. Returns what came back, which is the follower's
    /// buffer of zeros and is ignored.
    ///
    /// One direction of this swap is waste, exactly as `lead`/`follow` and `swap_paged` are: the
    /// exchange is symmetric by contract. It is paid only on a step that samples.
    ///
    /// # Errors
    /// The socket, or a peer whose frame is not this step's.
    pub fn lead_seam(&self, rows: u32, inputs: &SeamInputs) -> Result<(), ForwardError> {
        let penalties = !inputs.pen_ids.is_empty();
        let want = self.seam.input_bytes(rows, penalties);
        let mut frame = Vec::with_capacity(want);
        frame.extend_from_slice(&self.seam.header(rows, 0));
        put_f32(&mut frame, &inputs.temps);
        put_i32(&mut frame, &inputs.pen_ids);
        if frame.len() != want {
            return Err(ForwardError::Peer(format!(
                "a seam input frame of {} byte(s) for {rows} row(s); this geometry says {want}",
                frame.len()
            )));
        }
        let ack = self
            .peer
            .exchange()
            .xchg_bytes("engine.seam.in", &frame)
            .map_err(|e| {
                ForwardError::Peer(format!("leading the seam at step {}: {e}", self.serial))
            })?;
        self.check_ack(&ack, "the seam's inputs")?;
        Ok(())
    }

    /// **Follower**: take the frame [`RankLink::lead_seam`] sent.
    ///
    /// # Errors
    /// The socket, a foreign or wrong-version frame, or a geometry that is not this rank's.
    pub fn follow_seam(&self, rows: u32, penalties: bool) -> Result<SeamInputs, ForwardError> {
        let want = self.seam.input_bytes(rows, penalties);
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.seam.in", &self.ack())
            .map_err(|e| {
                ForwardError::Peer(format!("following the seam at step {}: {e}", self.serial))
            })?;
        // THE WIDTH IS STILL THIS RANK'S OWN LAW. `xchg_bytes` does not require the two sides to
        // be the same width, which is the point, so the check the symmetric swap used to make for
        // free is made here instead: a seam frame that is not this geometry's is a pair that
        // disagrees about `K` or about the penalty width, and it is refused rather than parsed.
        if got.len() != want {
            return Err(ForwardError::Peer(format!(
                "a seam input frame of {} byte(s) for {rows} row(s); this geometry says {want}",
                got.len()
            )));
        }
        self.seam.check(&got, rows, 0)?;
        let mut at = SEAM_HEADER_BYTES;
        let temps = take_f32(&got, &mut at, rows as usize);
        let pen_ids = if penalties {
            take_i32(&got, &mut at, rows as usize * self.seam.penalties as usize)
        } else {
            Vec::new()
        };
        Ok(SeamInputs { temps, pen_ids })
    }

    /// **Follower**: send this rank's shard of every sampled row back, AFTER the replay.
    ///
    /// # Errors
    /// The socket, or a payload that is not the geometry both ranks laid out.
    pub fn send_seam(&self, rows: u32, out: &SeamOutputs) -> Result<(), ForwardError> {
        let frame = self.seam_frame(rows, out)?;
        self.peer
            .exchange()
            .xchg_bytes("engine.seam.out", &frame)
            .map_err(|e| {
                ForwardError::Peer(format!("sending the seam at step {}: {e}", self.serial))
            })?;
        Ok(())
    }

    /// The follower's seam output frame (`send_seam`'s bytes).
    fn seam_frame(&self, rows: u32, out: &SeamOutputs) -> Result<Vec<u8>, ForwardError> {
        let penalties = !out.pen.is_empty();
        let want = self.seam.output_bytes(rows, penalties);
        let mut frame = Vec::with_capacity(want);
        frame.extend_from_slice(&self.seam.header(rows, 1));
        put_f32(&mut frame, &out.values);
        put_i32(&mut frame, &out.ids);
        put_f32(&mut frame, &out.lse);
        put_f32(&mut frame, &out.pen);
        if frame.len() != want {
            return Err(ForwardError::Peer(format!(
                "a seam output frame of {} byte(s) for {rows} row(s); this geometry says {want}",
                frame.len()
            )));
        }
        Ok(frame)
    }

    /// **Leader**: take the follower's shard of every sampled row.
    ///
    /// # Errors
    /// The socket, a foreign or wrong-version frame, or a geometry that is not this rank's.
    pub fn recv_seam(&self, rows: u32, penalties: bool) -> Result<SeamOutputs, ForwardError> {
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.seam.out", &self.ack())
            .map_err(|e| {
                ForwardError::Peer(format!("receiving the seam at step {}: {e}", self.serial))
            })?;
        self.parse_seam(&got, rows, penalties)
    }

    /// Parse a follower's seam output frame (`recv_seam`'s law).
    fn parse_seam(
        &self,
        got: &[u8],
        rows: u32,
        penalties: bool,
    ) -> Result<SeamOutputs, ForwardError> {
        let want = self.seam.output_bytes(rows, penalties);
        if got.len() != want {
            return Err(ForwardError::Peer(format!(
                "a seam output frame of {} byte(s) for {rows} row(s); this geometry says {want}",
                got.len()
            )));
        }
        self.seam.check(got, rows, 1)?;
        let (rows, k, pen) = (
            rows as usize,
            self.seam.k as usize,
            self.seam.penalties as usize,
        );
        let mut at = SEAM_HEADER_BYTES;
        let values = take_f32(got, &mut at, rows * k);
        let ids = take_i32(got, &mut at, rows * k);
        let lse = take_f32(got, &mut at, rows * 2);
        let pen = if penalties {
            take_f32(got, &mut at, rows * pen)
        } else {
            Vec::new()
        };
        Ok(SeamOutputs {
            values,
            ids,
            lse,
            pen,
        })
    }

    /// Swap one rank's SHARD of a dense logit row for the other's.
    ///
    /// Both sides send `cols` f32s. The leader sends this after a [`Command::DenseRow`] and the
    /// follower answers it from its own head; neither reads the other's ids, because a shard's
    /// columns are ordered by the vocabulary and the concatenation is by rank.
    ///
    /// # Errors
    /// The socket, or a peer whose shard is a different width.
    pub fn swap_dense(&self, mine: &[f32]) -> Result<Vec<f32>, ForwardError> {
        let mut bytes = Vec::with_capacity(mine.len() * 4);
        put_f32(&mut bytes, mine);
        let got = self
            .peer
            .exchange()
            .swap_bytes("engine.dense", &bytes)
            .map_err(|e| {
                ForwardError::Peer(format!("swapping a {}-wide dense shard: {e}", mine.len()))
            })?;
        let mut at = 0usize;
        Ok(take_f32(&got, &mut at, mine.len()))
    }

    /// **THE DRAFTER'S TABLE FRAME.** The leader hands the follower one proposal's index tables;
    /// what comes back is the follower's own frame, whose first eight bytes carry the digest of
    /// the block IT last proposed.
    ///
    /// A frame of its own rather than a [`Command`], because the command frame is
    /// `HEADER + payload_bytes` wide whatever it carries — 1.9 MiB at `max_seqs` 8 — and a
    /// proposal's tables are nine kilobytes. The number of proposals is not in the step's header
    /// either: each frame says whether another follows, and the last one carries the count, which
    /// the follower checks against what it ran. A pair that disagreed about how many proposals a
    /// step made would advance the all-reduce sequence counters by different amounts and the next
    /// collective would never retire.
    ///
    /// # Errors
    /// The socket, or a frame that is not the agreed width.
    pub fn swap_draft(&mut self, mine: &[u8]) -> Result<Vec<u8>, ForwardError> {
        // P27-14 (`socket`): the owed `Command::Propose` carries this, the proposal's first
        // frame, as its payload -- one round trip where there were two. The follower takes the
        // payload as the frame; the leader ignores what a frame answers, so the answer it gets
        // here is the width it sent.
        if std::mem::take(&mut self.propose_owed) {
            let carry = self.carry.take();
            if mine.len() <= self.payload_bytes {
                self.lead(
                    Command::Propose {
                        store_first: false,
                        inline: true,
                        carry,
                    },
                    mine,
                )?;
                self.lever_counts
                    .propose_inline
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(vec![0u8; mine.len()]);
            }
            self.lead(
                Command::Propose {
                    store_first: false,
                    inline: false,
                    carry,
                },
                &[],
            )?;
        }
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.draft", mine)
            .map_err(|e| {
                ForwardError::Peer(format!("a proposal frame at step {}: {e}", self.serial))
            })?;
        if got.len() != mine.len() {
            return Err(ForwardError::Peer(format!(
                "a proposal frame of {} byte(s) against this rank's {}",
                got.len(),
                mine.len()
            )));
        }
        Ok(got)
    }

    /// **THE CANDIDATE SEAM.** This rank's `steps x top_k` (value, global id) pairs for the peer's.
    ///
    /// Symmetric and REAL in both directions, unlike every other frame on this wire: neither
    /// rank's half of the vocabulary contains the row's own top-k, so each one needs what the
    /// other has. 896 bytes a proposal at the drafter's own shape, which is prod's own
    /// `O(batch * 2k * tp_size)` and not an all-gather of 248,320 columns.
    ///
    /// # Errors
    /// The socket, or a peer whose frame is a different width — which is a pair that disagrees
    /// about `selector_top_k` or about the block.
    pub fn swap_cand(&self, mine: &[u8]) -> Result<Vec<u8>, ForwardError> {
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.draftcand", mine)
            .map_err(|e| {
                ForwardError::Peer(format!("the candidate seam at step {}: {e}", self.serial))
            })?;
        if got.len() != mine.len() {
            return Err(ForwardError::Peer(format!(
                "a candidate seam frame of {} byte(s) against this rank's {}",
                got.len(),
                mine.len()
            )));
        }
        Ok(got)
    }

    /// P27-14 (`picks`, LEADER): send this rank's picks and take the follower's picks AND its seam
    /// shard in ONE frame (`engine.pickseam`), where [`Self::swap_picks`] + [`Self::recv_seam`]
    /// were two. Same bytes, one round trip.
    ///
    /// # Errors
    /// The socket, or a frame that is not this geometry's.
    pub fn swap_picks_seam_leader(
        &self,
        mine: &[Pick],
        rows: u32,
        penalties: bool,
    ) -> Result<(Vec<Pick>, SeamOutputs), ForwardError> {
        let picks = pack_picks(mine);
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.pickseam", &picks)
            .map_err(|e| {
                ForwardError::Peer(format!(
                    "the pick + seam frame at step {}: {e}",
                    self.serial
                ))
            })?;
        let np = picks.len();
        if got.len() < np {
            return Err(ForwardError::Peer(format!(
                "a pick + seam frame of {} byte(s) for {np} pick byte(s)",
                got.len()
            )));
        }
        let theirs = unpack_picks(&got[..np])?;
        let seam = self.parse_seam(&got[np..], rows, penalties)?;
        Ok((theirs, seam))
    }

    /// P27-14 (`picks`, FOLLOWER): this rank's picks and its seam shard, one frame.
    ///
    /// # Errors
    /// The socket, or a seam that is not this geometry's.
    pub fn swap_picks_seam_follower(
        &self,
        mine: &[Pick],
        rows: u32,
        out: &SeamOutputs,
    ) -> Result<(), ForwardError> {
        let mut frame = pack_picks(mine);
        frame.extend_from_slice(&self.seam_frame(rows, out)?);
        self.peer
            .exchange()
            .xchg_bytes("engine.pickseam", &frame)
            .map_err(|e| {
                ForwardError::Peer(format!(
                    "the pick + seam frame at step {}: {e}",
                    self.serial
                ))
            })?;
        self.lever_counts
            .pickseam_frames
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Swap this rank's picks for the peer's. Both ranks send one per row, in row order.
    ///
    /// # Errors
    /// The socket, or a peer that answered a different number of rows.
    pub fn swap_picks(&self, mine: &[Pick]) -> Result<Vec<Pick>, ForwardError> {
        let bytes = pack_picks(mine);
        let got = self
            .peer
            .exchange()
            .xchg_bytes("engine.picks", &bytes)
            .map_err(|e| ForwardError::Peer(format!("swapping {} pick(s): {e}", mine.len())))?;
        unpack_picks(&got)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: f32) -> u32 {
        // What the engine does with the two bytes it reads back: widen, which is exact.
        f32::from_bits(v.to_bits() & 0xffff_0000).to_bits()
    }

    /// **THE WIRE, END TO END, AND WHAT IT NOW COSTS.**
    ///
    /// The step frame used to be `HEADER + payload_bytes` wide in BOTH directions whatever it
    /// carried — 1,932 KiB out and 1,932 KiB of the follower's zeros back, per step, per rank, to
    /// say where four sequences are (ws-h29's census; its "what is still on this wire" paragraph
    /// is this test's subject). It is now the step's own bytes out and an eight-byte ack back.
    ///
    /// The two ranks are two threads over a `ThreadPairExchange`, which is the same
    /// `IpcExchange` contract the TCP pair implements, so this is the protocol and not a mock of
    /// it. **The CONTROL is the byte count**: a test that only checked the payload arrived would
    /// pass on the old, padded frame too.
    #[test]
    fn a_step_frame_carries_the_steps_own_bytes_and_an_ack_comes_back() {
        use saber_exec::{RankPeer, ThreadPairExchange};
        use std::sync::Arc;

        const PAYLOAD: usize = 4096;
        const PAGED: usize = 64;
        let seam = SeamGeometry { k: 2, penalties: 0 };
        let (ex0, ex1) = ThreadPairExchange::pair();
        let mut leader =
            RankLink::new(Arc::new(RankPeer::new(Arc::new(ex0))), PAYLOAD, PAGED, seam);
        let mut follower =
            RankLink::new(Arc::new(RankPeer::new(Arc::new(ex1))), PAYLOAD, PAGED, seam);
        let cmd = Command::Step {
            phase: Phase::Decode,
            rows: 8,
            width: 8,
            live: 8,
            samples: true,
            paged: true,
            seam: true,
            seam_penalties: false,
            defer_proposals: false,
            ragged: false,
            tree: false,
            slot: 0,
        };
        // A decode step's frame: a handful of runs and eight ids, not the 4 KiB region.
        let payload: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        let pages: Vec<u8> = vec![7u8; PAGED];
        let temps = vec![0.7f32; 8];
        let out = SeamOutputs {
            values: vec![1.0f32; 16],
            ids: vec![3i32; 16],
            lse: vec![0.5f32; 16],
            pen: Vec::new(),
        };
        let mine = payload.clone();
        let theirs = pages.clone();
        let back = out.clone();
        let hand = std::thread::spawn(move || {
            let mut blob = vec![0u8; PAYLOAD];
            let got = follower.follow(&mut blob).expect("following the step");
            assert_eq!(got, cmd, "the command crossed");
            let carried = follower.last_payload_bytes();
            assert_eq!(&blob[..carried], &mine[..], "the payload crossed");
            // THE CONTROL: the frame is the STEP's bytes, not the region's. On the padded
            // protocol this was `PAYLOAD` on every frame, including a `Ping`.
            assert_eq!(
                carried,
                mine.len(),
                "the follower was handed {carried} byte(s) for a {}-byte step",
                mine.len()
            );
            let pg = follower.follow_paged().expect("the block table");
            assert_eq!(pg, theirs, "the paged frame crossed");
            let si = follower.follow_seam(8, false).expect("the seam's inputs");
            assert_eq!(si.temps, vec![0.7f32; 8]);
            follower.send_seam(8, &back).expect("the seam's outputs");
            let picks = vec![
                Pick {
                    index: 5,
                    bits: 0x3f80_0000
                };
                8
            ];
            let peers = follower.swap_picks(&picks).expect("the picks");
            assert_eq!(peers.len(), 8, "the leader's picks came back");
        });
        leader.lead(cmd, &payload).expect("leading the step");
        leader.swap_paged(&pages).expect("the block table");
        leader
            .lead_seam(
                8,
                &SeamInputs {
                    temps,
                    pen_ids: Vec::new(),
                },
            )
            .expect("the seam's inputs");
        let got = leader.recv_seam(8, false).expect("the seam's outputs");
        assert_eq!(got.values, out.values, "the follower's shard came back");
        let mine = vec![
            Pick {
                index: 9,
                bits: 0x4000_0000
            };
            8
        ];
        let peers = leader.swap_picks(&mine).expect("the picks");
        assert_eq!(peers.len(), 8);
        assert_eq!(
            peers[0].index, 5,
            "the follower's own shard, not this rank's"
        );
        hand.join().expect("the follower thread");
    }

    /// A command with no payload is a HEADER and nothing else — the other half of the same claim,
    /// and the one that used to move 1.9 MiB to say "stop".
    #[test]
    fn a_command_with_no_payload_carries_no_payload() {
        use saber_exec::{RankPeer, ThreadPairExchange};
        use std::sync::Arc;

        const PAYLOAD: usize = 4096;
        let seam = SeamGeometry { k: 2, penalties: 0 };
        let (ex0, ex1) = ThreadPairExchange::pair();
        let mut leader = RankLink::new(Arc::new(RankPeer::new(Arc::new(ex0))), PAYLOAD, 64, seam);
        let mut follower = RankLink::new(Arc::new(RankPeer::new(Arc::new(ex1))), PAYLOAD, 64, seam);
        let hand = std::thread::spawn(move || {
            let mut blob = vec![0u8; PAYLOAD];
            let got = follower.follow(&mut blob).expect("following");
            assert_eq!(got, Command::Stop);
            assert_eq!(
                follower.last_payload_bytes(),
                0,
                "a payload-less command must carry no payload"
            );
        });
        leader.lead(Command::Stop, &[]).expect("leading");
        hand.join().expect("the follower thread");
    }

    /// A payload wider than the widest the two ranks laid out is a refusal by NAME, not a frame
    /// the follower reads half of.
    #[test]
    fn a_frame_wider_than_the_link_is_refused() {
        use saber_exec::{RankPeer, ThreadPairExchange};
        use std::sync::Arc;

        let seam = SeamGeometry { k: 2, penalties: 0 };
        let (ex0, _ex1) = ThreadPairExchange::pair();
        let mut leader = RankLink::new(Arc::new(RankPeer::new(Arc::new(ex0))), 64, 16, seam);
        let cmd = Command::Step {
            phase: Phase::Decode,
            rows: 1,
            width: 1,
            live: 1,
            samples: true,
            paged: false,
            seam: false,
            seam_penalties: false,
            defer_proposals: false,
            ragged: false,
            tree: false,
            slot: 0,
        };
        let e = leader
            .lead(cmd, &[0u8; 65])
            .expect_err("a frame wider than the link was accepted");
        assert!(e.to_string().contains("this link's widest is 64"), "{e}");
    }

    /// Every command survives the wire, and the serial rides with it.
    #[test]
    fn a_command_round_trips() {
        for (i, cmd) in [
            Command::Ping,
            Command::ClearSlot { slot: 7 },
            // b4s2, not b8s1: the width is on the wire precisely so the follower can tell those
            // two apart, and a round-trip test that only ever sent width 1 would not have noticed.
            Command::Step {
                phase: Phase::Decode,
                rows: 8,
                width: 2,
                live: 6,
                samples: true,
                paged: false,
                seam: false,
                seam_penalties: false,
                defer_proposals: false,
                ragged: false,
                tree: false,
                slot: 0,
            },
            // A prefill chunk, WITH its state slot: the field ws-h30 added is the one the
            // follower cannot derive, so a round trip that always sent 0 would not test it.
            Command::Step {
                phase: Phase::Prefill,
                rows: 2048,
                width: 1,
                live: 25,
                samples: false,
                paged: true,
                seam: false,
                seam_penalties: false,
                defer_proposals: false,
                ragged: false,
                tree: false,
                slot: 6,
            },
            // EVERY flag bit at once: they share one word, so a decode that samples, crosses a
            // page boundary AND runs a penalised seam is the case that catches a mask written as
            // a comparison.
            Command::Step {
                phase: Phase::Decode,
                rows: 4,
                width: 1,
                live: 4,
                samples: true,
                paged: true,
                seam: true,
                seam_penalties: true,
                defer_proposals: false,
                ragged: false,
                tree: false,
                slot: 0,
            },
            // THE DEFERRED PROPOSAL (ws-f37): the flag alone, so a mask that aliased it onto
            // another bit would show here, and the command that starts the frames.
            Command::Step {
                phase: Phase::Decode,
                rows: 16,
                width: 8,
                live: 16,
                samples: true,
                paged: false,
                seam: false,
                seam_penalties: false,
                defer_proposals: true,
                ragged: false,
                tree: false,
                slot: 0,
            },
            Command::Propose {
                store_first: false,
                inline: false,
                carry: None,
            },
            // P27-3: a proposal whose prompt store was deferred into it.
            Command::Propose {
                store_first: true,
                inline: false,
                carry: None,
            },
            // P27-14: the proposal's first frame inline, with a carried digest.
            Command::Propose {
                store_first: false,
                inline: true,
                carry: Some(0xdead_beef_0123_4567),
            },
            // The common sampled step: a seam with no penalties, which is prod's own defaults.
            Command::Step {
                phase: Phase::Decode,
                rows: 5,
                width: 5,
                live: 5,
                samples: true,
                paged: false,
                seam: true,
                seam_penalties: false,
                defer_proposals: false,
                ragged: false,
                tree: false,
                slot: 0,
            },
            // A DENSE ROW, the fallback behind the seam: the head named by shape, because rank 1
            // remembers no step.
            Command::DenseRow {
                phase: Phase::Decode,
                head_rows: 8,
                row: 3,
            },
            // CMP27G: the replay arm's materialize before a tree step (every field its own word)
            Command::GdnMaterialize {
                slot: 9,
                rows: 7,
                tag: 0x00ab_cdef,
                par: 1,
            },
            // EVERY PARK OP (ws-t2): the sub-op is a word of the header and a frame that decoded
            // `ParkBegin` as `RestoreBegin` would publish a stale half over a live session.
            Command::Park {
                op: ParkOp::ParkBegin,
            },
            Command::Park {
                op: ParkOp::ParkPoll,
            },
            Command::Park {
                op: ParkOp::RestoreBegin,
            },
            Command::Park {
                op: ParkOp::RestorePoll,
            },
            Command::Park { op: ParkOp::Forget },
            Command::Vision {
                images: 3,
                patches: 70_000,
            },
            Command::Scatter { runs: 2 },
            Command::Stop,
        ]
        .into_iter()
        .enumerate()
        {
            let h = cmd.encode(i as u64);
            assert_eq!(Command::decode(&h, i as u64).expect("decode"), cmd);
        }
    }

    /// **C2 host test (d): a ROW BAND's step crosses the wire as a band.** The 40-row band at max
    /// width 8 and `b5s8` are the same `(rows, width)`, so the flag is the only thing that tells the
    /// follower which graph to replay; `live` is the slots' `Σ w_i` (here the ragged widths
    /// `[5, 3, 8, 2, 5, 1, 4, 5]` = 33 of the band's 40 rows). The per-slot widths themselves ride
    /// in the index frame's `cu` table, which is what every kernel on the follower reads.
    ///
    /// NEGATIVE CONTROLS: the same header with bit 5 cleared decodes to the RECTANGLE (so a wire
    /// that dropped the flag would replay `b5s8`), and a pre-C2 header (wire 11) is refused by the
    /// version check before any field is read.
    #[test]
    fn a_band_step_round_trips_and_its_flag_is_what_names_the_graph() {
        let widths = [5u32, 3, 8, 2, 5, 1, 4, 5];
        let live: u32 = widths.iter().sum();
        let band = Command::Step {
            phase: Phase::Decode,
            rows: 40,
            width: 8,
            live,
            samples: true,
            paged: true,
            seam: true,
            seam_penalties: false,
            defer_proposals: true,
            ragged: true,
            tree: false,
            slot: 0,
        };
        let h = band.encode(7);
        assert_eq!(Command::decode(&h, 7).expect("decode"), band);
        // A rectangular 16-row chain and a best16 target tree must select different GPU graphs.
        let mut chain = band;
        if let Command::Step {
            rows,
            width,
            live,
            ragged,
            tree,
            ..
        } = &mut chain
        {
            *rows = 16;
            *width = 16;
            *live = 16;
            *ragged = false;
            *tree = false;
        }
        let mut ancestor = chain;
        if let Command::Step { tree, .. } = &mut ancestor {
            *tree = true;
        }
        assert_ne!(chain.encode(7), ancestor.encode(7));
        assert_eq!(Command::decode(&ancestor.encode(7), 7).unwrap(), ancestor);
        // The flag is ONE bit of the flag word, beside the others: clear it and every other field
        // survives while the step becomes the rectangle of the same (rows, width).
        let mut cleared = h;
        let flags = u32::from_le_bytes([h[24], h[25], h[26], h[27]]);
        assert_eq!(flags & 32, 32, "bit 5 carries the band");
        cleared[24..28].copy_from_slice(&(flags & !32).to_le_bytes());
        match Command::decode(&cleared, 7).expect("decode") {
            Command::Step {
                ragged,
                rows,
                width,
                live: l,
                defer_proposals,
                ..
            } => {
                assert!(!ragged, "the control must lose the band");
                assert_eq!((rows, width, l, defer_proposals), (40, 8, live, true));
            }
            other => panic!("decoded {other:?}"),
        }
        assert_ne!(Command::decode(&cleared, 7).expect("decode"), band);
        // A header from the wire before the band: refused by version, not read.
        let mut old = h;
        old[32..36].copy_from_slice(&11u32.to_le_bytes());
        assert!(Command::decode(&old, 7).is_err(), "wire 11 must be refused");
        old[32..36].copy_from_slice(&16u32.to_le_bytes());
        assert!(
            Command::decode(&old, 7).is_err(),
            "a rank without a tree/chain discriminator must refuse this frame"
        );
        assert_eq!(VERSION, 17);
    }

    /// C10: a packed step's word round-trips K, the per-chunk sample / seam / penalty masks, and
    /// its frame; a C9 frame (packed 0) is the one-chunk step, and a wire-13 frame is refused.
    #[test]
    fn a_packed_mixed_step_round_trips() {
        let w = packed_word(3, 0b1011, 0b0011, 0b0001);
        assert_eq!(packed_fields(w), (3, 0b1011, 0b0011, 0b0001));
        let m = Command::Mixed {
            rung: 1536,
            chunk_live: 35,
            chunk_samples: true,
            slot: 4,
            decode_rows: 8,
            width: 1,
            decode_live: 8,
            paged: true,
            chunk_seam: true,
            chunk_seam_penalties: true,
            decode_seam: false,
            decode_seam_penalties: false,
            defer_proposals: true,
            packed: w,
        };
        let h = m.encode(11);
        assert_eq!(Command::decode(&h, 11).expect("decode"), m);
        // The control: the same frame with the word cleared is C9's one-chunk step.
        let mut one = h;
        one[44..48].copy_from_slice(&0u32.to_le_bytes());
        match Command::decode(&one, 11).expect("decode") {
            Command::Mixed { packed, rung, .. } => assert_eq!((packed, rung), (0, 1536)),
            other => panic!("decoded {other:?}"),
        }
        let mut old = h;
        old[32..36].copy_from_slice(&13u32.to_le_bytes());
        assert!(
            Command::decode(&old, 11).is_err(),
            "wire 13 must be refused"
        );
    }

    /// C9: the fused mixed step's frame round-trips every field, carries a payload like a
    /// `Step`, and its decode-seam bit is its own (the control: clearing it keeps the rest).
    #[test]
    fn a_mixed_step_round_trips() {
        let m = Command::Mixed {
            rung: 320,
            chunk_live: 222,
            chunk_samples: true,
            slot: 17,
            decode_rows: 35,
            width: 5,
            decode_live: 30,
            paged: true,
            chunk_seam: true,
            chunk_seam_penalties: false,
            decode_seam: true,
            decode_seam_penalties: true,
            defer_proposals: true,
            packed: 0,
        };
        assert!(m.has_payload());
        let h = m.encode(9);
        assert_eq!(Command::decode(&h, 9).expect("decode"), m);
        let mut cleared = h;
        let flags = u32::from_le_bytes([h[24], h[25], h[26], h[27]]);
        assert_eq!(flags & 64, 64, "bit 6 carries the decode seam");
        cleared[24..28].copy_from_slice(&(flags & !64).to_le_bytes());
        match Command::decode(&cleared, 9).expect("decode") {
            Command::Mixed {
                decode_seam,
                decode_seam_penalties,
                rung,
                chunk_live,
                decode_live,
                slot,
                ..
            } => {
                assert!(!decode_seam);
                assert!(decode_seam_penalties);
                assert_eq!((rung, chunk_live, decode_live, slot), (320, 222, 30, 17));
            }
            other => panic!("decoded {other:?}"),
        }
        // A frame from the wire before C9: refused by version.
        let mut old = h;
        old[32..36].copy_from_slice(&12u32.to_le_bytes());
        assert!(Command::decode(&old, 9).is_err(), "wire 12 must be refused");
    }

    /// D4: the pixel frame and the scatter frame round-trip, and a torn one is refused by name.
    #[test]
    fn the_vision_and_scatter_frames_round_trip_and_refuse_a_torn_frame() {
        let grids = vec![[1u32, 2, 4], [1, 4, 2]];
        let pixels: Vec<f32> = (0..16 * PIXEL_ROW).map(|i| i as f32 * 0.5 - 3.0).collect();
        let f = vision_frame(&grids, &pixels);
        let (g, p) = decode_vision_frame(&f, 2, 16).expect("round trip");
        assert_eq!(g, grids);
        assert_eq!(p, pixels);
        // The controls: a wrong image count, a wrong patch count, a short body.
        assert!(decode_vision_frame(&f, 1, 16).is_err());
        assert!(decode_vision_frame(&f, 2, 15).is_err());
        assert!(decode_vision_frame(&f[..f.len() - 4], 2, 16).is_err());
        let runs = vec![
            saber_exec::mrope::ScatterRun {
                dst_row: 5,
                src_row: 0,
                rows: 192,
            },
            saber_exec::mrope::ScatterRun {
                dst_row: 0,
                src_row: 192,
                rows: 7,
            },
        ];
        let f = scatter_frame(&runs);
        assert_eq!(decode_scatter_frame(&f, 2).expect("round trip"), runs);
        assert!(decode_scatter_frame(&f, 3).is_err());
    }

    /// **A PARK FRAME CARRIES A SESSION THE FOLLOWER CANNOT DERIVE, AND ITS ANSWER** (ws-t2).
    ///
    /// The page list is the half of it that varies, so it is what the round trip is written
    /// around: a frame that dropped or truncated it would park the wrong pages on one card and
    /// the right ones on the other, and the only symptom would be a restored session proposing
    /// out of somebody else's K/V.
    #[test]
    fn a_park_frame_round_trips_with_its_session() {
        let width = PARK_HEAD + PARK_MSG + 64 * 4;
        let ask =
            ParkFrame::ask(0xdead_beef_u64, 17, 2048, 6, (100..164).collect()).with_draft(96, 8);
        let bytes = ask.encode(width).expect("it fits");
        assert_eq!(bytes.len(), width, "the frame is the link's width, padded");
        assert_eq!(ParkFrame::decode(&bytes).expect("decode"), ask);

        // The three answers, each of which the leader acts on differently.
        for reply in [
            ParkFrame::answer(7, ParkStatus::Ok),
            ParkFrame::answer(7, ParkStatus::Pending),
            ParkFrame::done(7, 266_911_744, [0xa5u8; 32]),
            ParkFrame::refused(7, "this rank has no park staging buffer"),
        ] {
            let bytes = reply.encode(width).expect("it fits");
            assert_eq!(ParkFrame::decode(&bytes).expect("decode"), reply);
        }

        // THE FORGET ANSWER (ws-k18) carries what the follower held and what it still holds, and
        // the leader's discard receipt reads both back off the wire. The control is the answer for
        // a name the follower did not hold: the same frame with `held` false.
        for (held, remaining) in [(true, 0u64), (false, 1), (true, 3)] {
            let reply = ParkFrame::forgotten_answer(9, held, remaining);
            let bytes = reply.encode(width).expect("it fits");
            let back = ParkFrame::decode(&bytes).expect("decode");
            assert_eq!(back.status, ParkStatus::Ok);
            assert_eq!(back.forgotten(), (held, remaining));
        }
        assert_eq!(
            ParkFrame::answer(9, ParkStatus::Ok).forgotten(),
            (false, 0),
            "a plain answer reads as nothing held"
        );
    }

    /// NEGATIVE CONTROLS for the same frame: a page list that does not fit is a REFUSAL, and so
    /// is a frame from another build. A truncated page list would be a session parked over
    /// another session's pages, which is the one failure this protocol can produce silently.
    #[test]
    fn a_park_frame_that_does_not_fit_or_does_not_belong_is_refused() {
        let narrow = PARK_HEAD + PARK_MSG + 4;
        let ask = ParkFrame::ask(1, 1, 16, 1, (0..8).collect());
        let e = ask.encode(narrow).expect_err("8 pages in room for 1");
        assert!(e.to_string().contains("park frame for 8 page(s)"), "{e}");

        let mut bytes = ask.encode(PARK_HEAD + PARK_MSG + 32).expect("it fits");
        bytes[4] = VERSION as u8 + 1;
        let e = ParkFrame::decode(&bytes).expect_err("another build");
        assert!(e.to_string().contains("protocol version"), "{e}");
        bytes[0] ^= 0xff;
        let e = ParkFrame::decode(&bytes).expect_err("not a park frame");
        assert!(e.to_string().contains("magic"), "{e}");
        let e = ParkFrame::decode(&bytes[..PARK_HEAD]).expect_err("short");
        assert!(e.to_string().contains("park frame of"), "{e}");
    }

    /// NEGATIVE CONTROL: a frame from a step this rank is not on is refused, not run. Without the
    /// serial, a follower that missed a command would run the NEXT step's shape over the LAST
    /// step's buffers and the only symptom would be a wrong token.
    #[test]
    fn a_drifted_serial_is_refused() {
        let h = Command::Ping.encode(41);
        let e = Command::decode(&h, 42).expect_err("drift");
        assert!(format!("{e}").contains("drifted"), "{e}");
    }

    /// NEGATIVE CONTROL: a frame that is not one of ours says so.
    #[test]
    fn a_foreign_frame_is_refused() {
        let mut h = Command::Ping.encode(0);
        h[0] ^= 0xff;
        assert!(format!("{}", Command::decode(&h, 0).expect_err("magic")).contains("magic"));
        let short = [0u8; 4];
        assert!(Command::decode(&short, 0).is_err());
    }

    /// NEGATIVE CONTROL for ws-h30's header growth: a frame at another PROTOCOL VERSION is
    /// refused by name, and a 32-byte frame (the header before it) is too short to be one.
    ///
    /// The failure this prevents is the quiet one: the seven words the old header had all keep
    /// their offsets, so a follower on the old build would decode the phase, rows, width and
    /// flags of a new leader's frame perfectly and read a state slot that is not there.
    #[test]
    fn a_frame_at_another_protocol_version_is_refused() {
        let mut h = Command::Ping.encode(0);
        h[32..36].copy_from_slice(&1u32.to_le_bytes());
        let e = Command::decode(&h, 0).expect_err("version");
        let text = format!("{e}");
        assert!(text.contains("version 1"), "{text}");
        assert!(text.contains(&format!("{VERSION}")), "{text}");
        // Neither the 32-byte header of every build up to ws-h29 nor ws-h30's 40-byte one is long
        // enough to be one of these.
        assert_eq!(HEADER_BYTES, 48);
        assert!(Command::decode(&h[..32], 0).is_err());
        assert!(Command::decode(&h[..40], 0).is_err());
    }

    /// V4's verb, both ways: five numbers in, five numbers out, and the `None` that a prompt
    /// ending on a page boundary produces is not block 0.
    ///
    /// The distinction is the whole reason `cow_source` rides a flag bit rather than a sentinel:
    /// block 0 is a legal page and a fork that copied it into a branch that owed no copy would
    /// overwrite the branch's first private page with another session's.
    #[test]
    fn a_fork_frame_carries_its_five_numbers_and_the_absent_page_is_not_block_zero() {
        let with_page = Command::Fork {
            from_slot: 1,
            to_slot: 6,
            slots: 5,
            cow_source: Some(0),
            cow_dest: Some(41),
        };
        let h = with_page.encode(7);
        assert_eq!(Command::decode(&h, 7).expect("decode"), with_page);
        let no_page = Command::Fork {
            from_slot: 11,
            to_slot: 16,
            slots: 5,
            cow_source: None,
            cow_dest: None,
        };
        let h = no_page.encode(8);
        assert_eq!(Command::decode(&h, 8).expect("decode"), no_page);
    }

    /// ws-f38's verb, both ways: the two runs and the copied count a follower needs to give a
    /// branch its half of the leader's drafter pages, none of which it can derive.
    #[test]
    fn a_fork_draft_frame_carries_its_four_numbers() {
        let cmd = Command::ForkDraft {
            from_first: 257,
            to_first: 771,
            run: 257,
            copy: 16,
        };
        let h = cmd.encode(3);
        assert_eq!(Command::decode(&h, 3).expect("decode"), cmd);
        // A copy of nothing is a legal frame (a zero-length prompt is refused long before this),
        // and it is not the same frame as a copy of one page.
        let none = Command::ForkDraft {
            from_first: 0,
            to_first: 257,
            run: 257,
            copy: 0,
        };
        assert_ne!(Command::decode(&none.encode(4), 4).expect("decode"), cmd);
    }

    /// V7's verb, both ways: the four numbers a follower needs to re-run the head over a prefill
    /// chunk's rows, none of which it can derive (it keeps no record of which plan ran last).
    #[test]
    fn a_prompt_rows_frame_carries_its_four_numbers() {
        let cmd = Command::PromptRows {
            prefill_rows: 2048,
            head_rows: 64,
            first: 1984,
            count: 17,
        };
        let h = cmd.encode(9);
        assert_eq!(Command::decode(&h, 9).expect("decode"), cmd);
        assert!(!cmd.has_payload());
    }

    /// The gather: the bigger logit wins and the id is GLOBAL, whichever rank it came from.
    #[test]
    fn the_bigger_logit_wins_and_the_id_is_global() {
        let cols = 100;
        let mine = vec![
            Pick {
                index: 5,
                bits: bf16(2.0),
            },
            Pick {
                index: 7,
                bits: bf16(-1.0),
            },
        ];
        let theirs = vec![
            Pick {
                index: 9,
                bits: bf16(1.0),
            },
            Pick {
                index: 3,
                bits: bf16(4.0),
            },
        ];
        // Rank 0 owns [0, 100) and rank 1 owns [100, 200). Row 0 goes to rank 0's column 5 (2.0
        // beats 1.0) and row 1 to rank 1's column 3 (4.0 beats -1.0).
        assert_eq!(
            combine(0, cols, &mine, &theirs).expect("combine"),
            vec![5, 103]
        );
        // THE SAME TOKENS from rank 1's side of the same pair, with `mine` and `theirs` swapped —
        // which is the property that matters: neither rank's answer depends on which rank it is.
        assert_eq!(
            combine(1, cols, &theirs, &mine).expect("combine"),
            vec![5, 103]
        );
    }

    /// A rank with no finite logit yields to the one that has one.
    #[test]
    fn a_nan_shard_yields() {
        let cols = 64;
        let mine = vec![Pick::none(cols)];
        let theirs = vec![Pick {
            index: 11,
            bits: bf16(0.5),
        }];
        assert_eq!(combine(0, cols, &mine, &theirs).expect("combine"), vec![75]);
    }

    /// NEGATIVE CONTROL: neither shard had a number, so there is no token to commit and the step
    /// fails loudly rather than returning column zero.
    #[test]
    fn two_nan_shards_are_a_refusal() {
        let cols = 64;
        let n = vec![Pick::none(cols)];
        let e = combine(0, cols, &n, &n).expect_err("no logit");
        assert!(format!("{e}").contains("a finite logit"), "{e}");
    }

    /// A tie goes to the lower GLOBAL id, so both ranks compute the same token from the same pair.
    #[test]
    fn a_tie_goes_to_the_lower_global_id() {
        let cols = 50;
        let same = bf16(3.25);
        let a = vec![Pick {
            index: 40,
            bits: same,
        }];
        let b = vec![Pick {
            index: 1,
            bits: same,
        }];
        // Rank 0's column 40 is token 40; rank 1's column 1 is token 51.
        assert_eq!(combine(0, cols, &a, &b).expect("combine"), vec![40]);
        assert_eq!(combine(1, cols, &b, &a).expect("combine"), vec![40]);
    }

    /// NEGATIVE CONTROL: a column outside the shard is a protocol error, not a token.
    #[test]
    fn a_column_past_the_shard_is_refused() {
        let bad = vec![Pick {
            index: 101,
            bits: bf16(1.0),
        }];
        let ok = vec![Pick {
            index: 1,
            bits: bf16(1.0),
        }];
        assert!(combine(0, 100, &bad, &ok).is_err());
        assert!(combine(0, 100, &ok, &bad).is_err());
    }

    // ── the seam frames ─────────────────────────────────────────────────────────

    const GEOM: SeamGeometry = SeamGeometry { k: 4, penalties: 3 };

    /// The frame widths are a function of the geometry and the row count alone — which is what
    /// "sizes are shape facts, not a negotiation" means: both ranks compute the same number from
    /// plans they built identically, and `swap_bytes` refuses a mismatch by name.
    #[test]
    fn a_seam_frames_width_is_its_shape() {
        assert_eq!(GEOM.input_bytes(5, false), SEAM_HEADER_BYTES + 5 * 4);
        assert_eq!(
            GEOM.input_bytes(5, true),
            SEAM_HEADER_BYTES + 5 * 4 + 5 * 3 * 4
        );
        assert_eq!(
            GEOM.output_bytes(5, false),
            SEAM_HEADER_BYTES + 5 * 4 * 4 + 5 * 4 * 4 + 5 * 2 * 4
        );
        assert_eq!(
            GEOM.output_bytes(5, true),
            GEOM.output_bytes(5, false) + 5 * 3 * 4
        );
    }

    /// A seam frame survives its header: the magic, the version, the row count and the geometry
    /// are all checked by the rank that receives it.
    #[test]
    fn a_seam_header_round_trips() {
        for kind in [0u32, 1] {
            let h = GEOM.header(7, kind);
            GEOM.check(&h, 7, kind).expect("its own header");
            // NEGATIVE CONTROLS, one per field.
            let mut foreign = h;
            foreign[0] ^= 0xff;
            assert!(
                format!("{}", GEOM.check(&foreign, 7, kind).expect_err("magic")).contains("magic")
            );
            let mut old = GEOM.header(7, kind);
            old[4..8].copy_from_slice(&(((SEAM_FRAME_VERSION + 1) << 8) | kind).to_le_bytes());
            assert!(
                format!("{}", GEOM.check(&old, 7, kind).expect_err("version")).contains("version")
            );
            assert!(format!("{}", GEOM.check(&h, 8, kind).expect_err("rows")).contains("row(s)"));
            let other = SeamGeometry {
                k: 64,
                penalties: 3,
            };
            assert!(format!("{}", other.check(&h, 7, kind).expect_err("geometry")).contains("K ="));
            assert!(GEOM.check(&h[..4], 7, kind).is_err());
        }
        // A frame of the wrong KIND is refused: the two frames ride the same socket in a fixed
        // order, and a rank that read them out of order would decode temperatures as logits.
        let inputs = GEOM.header(7, 0);
        assert!(format!("{}", GEOM.check(&inputs, 7, 1).expect_err("kind")).contains("kind"));
    }

    /// The payloads survive the wire in the order the ranks write them, penalties and all.
    #[test]
    fn seam_payloads_round_trip() {
        let rows = 3usize;
        let (k, pen) = (GEOM.k as usize, GEOM.penalties as usize);
        let inputs = SeamInputs {
            temps: vec![0.7, 0.0, 1.3],
            pen_ids: (0..rows * pen).map(|i| i as i32 - 1).collect(),
        };
        let mut frame = Vec::new();
        frame.extend_from_slice(&GEOM.header(rows as u32, 0));
        put_f32(&mut frame, &inputs.temps);
        put_i32(&mut frame, &inputs.pen_ids);
        assert_eq!(frame.len(), GEOM.input_bytes(rows as u32, true));
        let mut at = SEAM_HEADER_BYTES;
        assert_eq!(take_f32(&frame, &mut at, rows), inputs.temps);
        assert_eq!(take_i32(&frame, &mut at, rows * pen), inputs.pen_ids);

        let out = SeamOutputs {
            values: (0..rows * k).map(|i| i as f32 * -0.5).collect(),
            ids: (0..rows * k).map(|i| i as i32).collect(),
            lse: (0..rows * 2).map(|i| 10.0 + i as f32).collect(),
            pen: (0..rows * pen).map(|i| i as f32 * 0.25).collect(),
        };
        let mut frame = Vec::new();
        frame.extend_from_slice(&GEOM.header(rows as u32, 1));
        put_f32(&mut frame, &out.values);
        put_i32(&mut frame, &out.ids);
        put_f32(&mut frame, &out.lse);
        put_f32(&mut frame, &out.pen);
        assert_eq!(frame.len(), GEOM.output_bytes(rows as u32, true));
        let mut at = SEAM_HEADER_BYTES;
        assert_eq!(take_f32(&frame, &mut at, rows * k), out.values);
        assert_eq!(take_i32(&frame, &mut at, rows * k), out.ids);
        assert_eq!(take_f32(&frame, &mut at, rows * 2), out.lse);
        assert_eq!(take_f32(&frame, &mut at, rows * pen), out.pen);
    }

    /// **P27-14: the three levers' frames over a real pair** (two threads, the `IpcExchange`
    /// contract): `picks` = picks + seam in ONE frame and the same values as the two-frame form;
    /// `table` = a compact block-table frame accepted only after the whole one crossed; `socket`
    /// = the proposal's first frame inside its `Command::Propose`, with the carried digest. THE
    /// CONTROL for `table`: a compact frame before the whole one is refused on both ranks.
    #[test]
    fn p27_14_the_lever_frames_cross_a_real_pair() {
        use crate::host_levers::{HostLeverCounters, HostLevers};
        use saber_exec::{RankPeer, ThreadPairExchange};
        use std::sync::Arc;
        const PAYLOAD: usize = 4096;
        const PAGED: usize = 4096;
        let geom = SeamGeometry { k: 2, penalties: 0 };
        let (ex0, ex1) = ThreadPairExchange::pair();
        let mut leader =
            RankLink::new(Arc::new(RankPeer::new(Arc::new(ex0))), PAYLOAD, PAGED, geom);
        let mut follower =
            RankLink::new(Arc::new(RankPeer::new(Arc::new(ex1))), PAYLOAD, PAGED, geom);
        leader.set_levers(HostLevers::ALL, Arc::new(HostLeverCounters::default()));
        follower.set_levers(HostLevers::ALL, Arc::new(HostLeverCounters::default()));
        // THE CONTROL first: no whole frame has crossed, so the leader refuses a compact one.
        let compact = crate::host_levers::encode_paged(
            &crate::host_levers::PagedRecipe {
                tables: vec![vec![7u8; 64]],
                spans: vec![(0, 0)],
            },
            PAGED,
        )
        .expect("compact");
        assert!(leader.swap_paged_compact(&compact).is_err());
        let rows = 2u32;
        let out = SeamOutputs {
            values: vec![3.0, 2.0, 1.5, 1.0],
            ids: vec![5, 9, 1, 2],
            lse: vec![10.0, 11.0, 12.0, 13.0],
            pen: Vec::new(),
        };
        let frame: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
        let (out2, frame2, compact2) = (out.clone(), frame.clone(), compact.clone());
        let hand = std::thread::spawn(move || {
            // picks + seam
            let mine = vec![
                Pick {
                    index: 4,
                    bits: bf16(2.0),
                },
                Pick {
                    index: 6,
                    bits: bf16(1.0),
                },
            ];
            follower
                .swap_picks_seam_follower(&mine, rows, &out2)
                .expect("pickseam");
            // the whole table, then a compact one
            let whole = follower.follow_paged().expect("whole");
            assert_eq!(whole.len(), PAGED);
            follower.note_paged_whole();
            let c = follower.follow_paged().expect("compact");
            assert_eq!(c, compact2);
            // the Propose with its first frame inline and a carried digest
            follower.set_carry(0x0123_4567_89ab_cdef);
            let mut blob = vec![0u8; PAYLOAD];
            let cmd = follower.follow(&mut blob).expect("propose");
            let Command::Propose {
                store_first,
                inline,
                carry,
            } = cmd
            else {
                panic!("not a propose: {cmd:?}");
            };
            assert!(!store_first && inline);
            follower
                .check_carry(carry)
                .expect("the carried digest matches");
            assert_eq!(&blob[..follower.last_payload_bytes()], &frame2[..]);
        });
        let mine = vec![
            Pick {
                index: 1,
                bits: bf16(3.0),
            },
            Pick {
                index: 2,
                bits: bf16(0.5),
            },
        ];
        let (theirs, seam) = leader
            .swap_picks_seam_leader(&mine, rows, false)
            .expect("pickseam");
        assert_eq!(theirs[0].index, 4);
        assert_eq!(seam.values, out.values);
        assert_eq!(seam.ids, out.ids);
        assert_eq!(seam.lse, out.lse);
        leader.swap_paged(&[0u8; PAGED]).expect("whole");
        leader.note_paged_whole();
        leader
            .swap_paged_compact(&compact)
            .expect("compact after the whole");
        leader.set_carry(0x0123_4567_89ab_cdef);
        leader.owe_propose(true);
        assert!(leader.take_solo_ok());
        let got = leader.swap_draft(&frame).expect("inline");
        assert_eq!(got.len(), frame.len());
        hand.join().expect("the follower");
    }

    /// Picks survive the wire in row order.
    #[test]
    fn picks_round_trip() {
        let picks: Vec<Pick> = (0..8)
            .map(|i| Pick {
                index: i * 3,
                bits: bf16(i as f32),
            })
            .collect();
        assert_eq!(unpack_picks(&pack_picks(&picks)).expect("unpack"), picks);
        assert!(unpack_picks(&[0u8; 7]).is_err());
    }

    /// FN52: the prefix frame round-trips every word.
    #[test]
    fn a_prefix_frame_round_trips() {
        let c = Command::Prefix {
            op: 1,
            entry: 77,
            slot: 13,
            committed: 1,
            draft_first: 4096,
            draft_pages: 250,
        };
        let h = c.encode(9);
        assert_eq!(Command::decode(&h, 9).expect("decode"), c);
    }
}
