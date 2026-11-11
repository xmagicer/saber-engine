//! [`RealExecutor`] — the scheduler's [`StepExecutor`] over a [`ForwardRunner`].
//!
//! Everything phase-specific in the contract is enforced here rather than assumed:
//!
//! * a **prefill** chunk commits no token unless it is the last, and exactly one when it is;
//! * a **decode** step is **one call for the whole batch**: every live slot goes down together, the
//!   bucket's batch is the shape, and one token comes back per live slot in slot order. Each slot
//!   commits exactly one token and declares **no draft** — the P3 SPEC=0 target, with the drafter
//!   absent, so `draft_tokens` and `accepted_tokens` are zero and `tokens.len() == accepted + 1`
//!   holds trivially;
//! * `StepBatch::padded_slots` **is** run, because a captured graph has a fixed `M`. The rows past
//!   the live slots are padding the index tables point at the null state slot and a zero context, so
//!   they compute a row nobody reads and touch no live session;
//! * every step's launch count is compared against the plan's declared budget for the shape the pass
//!   says it ran, and a mismatch is a counted fault rather than a log line nobody reads;
//! * which path ran each step — a replayed graph or the eager op-by-op walk — is counted, because a
//!   fallback nobody counts is a fallback nobody notices.
//!
//! **Where the token is chosen.** The device computes an argmax for every sampled row and the
//! executor commits it unchanged for a greedy row — that path is byte for byte what it was before
//! this file knew what a temperature is, and `greedy_rows` counts it. A row whose request asks for
//! a temperature is decided by `saber_serve::sampler` on the host, from the seam the head's
//! `saber_device_row_bf16` wrote (`crate::forward::RowRequest` asks for it, `SlotCommit::rows`
//! carries it back — one per COMMITTED token), and the committed token replaces the argmax's
//! **last** token, the recovery or bonus token, which is the one this step actually chose. Three
//! counters say which branch ran: `greedy_rows`, `sampled_rows` and `dense_rows`, and the refusals
//! the seam raises are `seam_refusals` beside them (`CLAUDE.md` rule 5).
//!
//! **What a step REPORTS is a different question from what it CHOSE** (V3). `logprobs` is not a
//! sampling knob: a greedy request may ask for it and prod answers, so a greedy row that asked for
//! logprobs still asks the head for a seam — at T = 1, which is the scale logprobs are reported at
//! — and its token is still the argmax's. And a verify step commits `accepted + 1` tokens, each
//! with a row of its own, so every one of them is reported rather than only the last
//! (`logprobs_skipped_on_verify` now counts a row the head skipped or the shard merge refused,
//! which is a fault, and not the ordinary case it used to count).

use std::collections::BTreeMap;

use saber_plan::Phase;
use saber_sched::{
    AdmittedSeq, BatchSlot, Forked, ParkPoll, RestorePoll, RestoreStart, SeqId, SeqState, SeqStep,
    SlotSampling, StepBatch, StepExecutor, StepResult, TransferStart,
};
use saber_serve::engine::{SampledLogprobs, TokenId};
use saber_serve::sampler::{
    EntropyNoise, NoiseSource, PenaltyContext, RowParams, Sampled, Sampler, SamplerError,
    SeededNoise,
};

use crate::draft::{Drafter, NoDrafter};
use crate::forward::{
    DecodeSlot, ForwardError, ForwardFacts, ForwardRunner, Proposal, RowRequest, SlotCommit,
    StepOutcome, StepPath, StepRows, TokenFeed,
};

/// Which-path counters for the engine seam. Every branch that can silently change behaviour has
/// one (`CLAUDE.md` rule 5).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineCounters {
    /// Prefill chunks run.
    pub prefill_steps: u64,
    /// Prefill chunks that were not the last and therefore committed nothing.
    pub silent_prefill_steps: u64,
    /// C4's split of a step's wall, microseconds, summed: the forward call (the target's
    /// forward plus the drafter's context store — `ForwardRunner::prefill` / `decode`) and what
    /// the executor does after it (the sampler, the proposal after commit, prompt logprobs).
    /// A measurement only; nothing branches on it.
    pub prefill_forward_us: u64,
    /// See [`EngineCounters::prefill_forward_us`]: a prefill chunk's work after its forward.
    pub prefill_after_us: u64,
    /// See [`EngineCounters::prefill_forward_us`]: a decode step's forward call.
    pub decode_forward_us: u64,
    /// See [`EngineCounters::prefill_forward_us`]: a decode step's work after its forward.
    pub decode_after_us: u64,
    /// C5: the prompt rows the prefill chunks carried, summed.
    pub prefill_live_rows: u64,
    /// C5: the rows the prefill chunks' PLANNED buckets ran, summed — the padding is this minus
    /// [`EngineCounters::prefill_live_rows`]. A 300-row prompt on `--engine-prefill-rows 256,2048`
    /// runs 2048 rows of GEMM, which is the row law this pair of counters makes visible.
    pub prefill_plan_rows: u64,
    /// C9: mixed steps the runner ran as ONE fused forward (`--engine-mixed-prefill fused`).
    pub mixed_fused_steps: u64,
    /// C9: mixed steps under `fused` that the fused law refused, run in the serial form.
    pub mixed_fused_fallbacks: u64,
    /// C9: the prefill rungs the fused steps ran at, summed (rows).
    pub mixed_fused_rung_rows: u64,
    /// C9: the fused steps' forward-call wall, microseconds, summed (the decode half's forward
    /// call carries the whole fused forward; the chunk's prefill call returns its half).
    pub mixed_fused_forward_us: u64,
    /// C10: fused mixed steps that carried two or more prompt chunks (`--engine-mixed-prefill
    /// packed`).
    pub mixed_packed_steps: u64,
    /// C10: the prompt chunks those packed steps carried, summed.
    pub mixed_packed_chunks: u64,
    /// Decode steps run. A step is one call for the whole batch, not one per sequence.
    pub decode_steps: u64,
    /// Sequences stepped, summed over decode steps: `decode_steps` × batch when every slot is live.
    pub decode_sequences: u64,
    /// Padding rows run, summed. A captured graph has a fixed `M`, so these are the rows the shape
    /// carried and no sequence filled.
    pub padded_rows: u64,
    /// Decode steps run in a ROW BAND (C2): every slot at its own width.
    pub band_steps: u64,
    /// Rows the slots FED over the band steps (`Σ w_i`), so "rows per step" is this over
    /// [`EngineCounters::band_steps`].
    pub band_live_rows: u64,
    /// Rows the band steps RAN (the bands' `M`), padding included.
    pub band_rows: u64,
    /// Steps a captured graph replayed.
    pub graph_steps: u64,
    /// Steps the eager op-by-op path ran.
    pub eager_steps: u64,
    /// Steps whose input ids were already on the device — the previous step's argmax wrote them.
    pub device_fed_steps: u64,
    /// Steps whose input ids the host uploaded: every prompt, every first decode step, and every
    /// step whose slot set or tokens were not the ones the device was left holding.
    pub host_fed_steps: u64,
    /// Dispatches issued, summed over every step.
    pub launches: u64,
    /// Steps whose launch count did not match the budget declared for the shape they ran.
    pub budget_mismatches: u64,
    /// Steps that ran at a shape the pass declares no budget for.
    pub undeclared_shapes: u64,
    /// Steps whose reported shape contradicted the batch: the wrong phase, fewer rows than the
    /// batch asked for, a sampling flag that disagreed with the chunk's `last`, or the wrong number
    /// of tokens for the live slots.
    pub shape_violations: u64,
    /// Steps the forward refused. The sequences are failed loudly rather than committed tokens they
    /// did not produce.
    pub forward_errors: u64,
    /// Sequences the scheduler stepped that the engine had never admitted.
    pub unknown_sequences: u64,
    /// Steps refused because the batch carried more sequences than the engine has state for.
    pub oversized_step_refusals: u64,
    /// Draft tokens the drafter PROPOSED, summed over slots and steps. The denominator of the
    /// acceptance rate, and the number a report multiplies by a real drafter's acceptance.
    pub draft_tokens: u64,
    /// Draft tokens ACCEPTED, excluding the bonus token every verified step commits.
    pub accepted_tokens: u64,
    /// Tokens committed across every slot and step: `accepted + one per slot per step`. Divided by
    /// `decode_steps` this is tokens/step, which is what speculation buys.
    pub committed_tokens: u64,
    /// Proposals wider than the bucket's verify width, truncated. A drafter that ignores the width
    /// it was asked for is a bug in the drafter, not a shape to run.
    pub draft_overruns: u64,
    /// Proposals the PASS made and handed to the drafter
    /// ([`crate::forward::StepOutcome::drafts`]), one per live slot per step.
    ///
    /// 0 on a pass with no drafter tier of its own, which is every runner on this branch. Counted
    /// because the difference between "the device drafter is off" and "the device drafter ran and
    /// proposed nothing" is invisible in `draft_tokens` — both read 0 there, and only one of them
    /// is a fault.
    pub pass_proposals: u64,
    /// Proposals the pass made and this executor DROPPED because their anchor was not committed
    /// (ws-f35).
    ///
    /// The device proposes inside the step, from the ARGMAX of the row the step is about to
    /// commit; a row with a temperature then has that token replaced by the host sampler's draw.
    /// When the draw differs, the block the device made is the continuation of a token the
    /// sequence never had — every one of its tokens is conditioned on the wrong history — and
    /// offering it spends a whole verify width on a guess the engine knows is about the wrong
    /// text. Zero at temperature 0 on every row, by construction: a greedy row commits the
    /// argmax.
    pub pass_proposals_stale: u64,
    /// Blocks with a stale anchor that were VERIFIED, and what they earned, under the CONTROL
    /// (`keep_stale_anchor_blocks`). Both are zero under the law, because such a block is never
    /// offered — which is the point: the pair says what the dropped blocks were worth, in the same
    /// run, rather than from a difference between two corpora.
    pub stale_anchor_drafts: u64,
    /// Tokens accepted out of those blocks. See [`EngineCounters::stale_anchor_drafts`].
    pub stale_anchor_accepted: u64,
    /// FN69 (`--spec-lookup on`): the prompt-lookup source's queries, hits and the drafting
    /// points the merge law gave it; draft tokens verified / accepted per source. All zero when
    /// the lookup is off.
    pub lookup_lookups: u64,
    /// See [`EngineCounters::lookup_lookups`].
    pub lookup_hits: u64,
    /// See [`EngineCounters::lookup_lookups`].
    pub lookup_used: u64,
    /// See [`EngineCounters::lookup_lookups`].
    pub lookup_drafted: u64,
    /// See [`EngineCounters::lookup_lookups`].
    pub lookup_accepted: u64,
    /// See [`EngineCounters::lookup_lookups`].
    pub lookup_drafter_drafted: u64,
    /// See [`EngineCounters::lookup_lookups`].
    pub lookup_drafter_accepted: u64,
    /// Proposals made AFTER the host sampler, from the committed anchor (ws-f37): prod's order,
    /// and the serving default. One per live slot per step that stashed one; each is also a
    /// [`EngineCounters::pass_proposals`] when it is offered.
    pub proposals_after_commit: u64,
    /// Blocks made after the commit whose anchor — as the drafter was FED it, echoed back by the
    /// pass — is not the token the slot committed. **A DEFECT, not a rate**: after ws-f37 the two
    /// are the same token by construction, so a non-zero count is a pass that fed its drafter
    /// something other than what it was handed. Refused by name, never offered, and it makes the
    /// run unclean. `stale_anchor.rs` injects one as the control.
    pub stale_anchor_defects: u64,
    /// Steps whose stashed proposal could not be collected: the pass named different slots from
    /// the ones the step ran, or refused the collection. The step's tokens stand; the next step
    /// verifies nothing for those slots, and the pass counts the stash it abandoned.
    pub proposal_collect_failures: u64,
    /// Sessions copied off the device as a park image.
    pub parks: u64,
    /// Park attempts the forward refused. The scheduler then refuses the park, which is a loud no
    /// rather than an eviction whose state cannot be brought back.
    pub park_failures: u64,
    /// Session images put back into a freshly admitted sequence.
    pub restores: u64,
    /// Restore attempts refused: an image that did not decode, did not verify, or was the wrong
    /// shape for the pages and slot this admission was given.
    pub restore_failures: u64,
    /// Images the scheduler dropped unrestored (an abort while parked, a restore given up on) whose
    /// hold on this engine — at TP=2 the follower's half — was released (ws-k18).
    pub parked_discards: u64,
    /// Of those, the ones for which some rank still HELD a half: a discard that found nothing held
    /// is a name the engine never parked under, and is counted apart.
    pub parked_discards_held: u64,
    /// Restores refused ON PURPOSE by `RealExecutor::fail_restores` — the diagnostic that is the
    /// card control for the bounded-restore law. Zero on every serving run.
    pub restores_refused_by_injection: u64,
    /// D3: images whose tower ran at admission, their patches, the image-token rows the tower
    /// left in its arena, the dispatches, and the image requests refused or failed there.
    pub vision_images: u64,
    /// Patches the tower ran over.
    pub vision_patches: u64,
    /// Image-token rows (patches / 4) left in the tower arena.
    pub vision_tokens: u64,
    /// Tower dispatches (one per image REQUEST since D4: a request's images run together).
    pub vision_runs: u64,
    /// Image sequences admitted with their M-RoPE table and scatter (D4).
    pub vision_seqs: u64,
    /// Image requests refused or failed at the tower.
    pub vision_failures: u64,
    /// Rows committed straight from the device argmax: `temperature < 1e-5`, the byte-identical
    /// path every temperature-0 gate in this tree runs.
    pub greedy_rows: u64,
    /// Rows the host sampler decided from the head's top-K seam.
    pub sampled_rows: u64,
    /// Rows the seam could not decide and that were read back DENSELY and sampled from the whole
    /// vocabulary. Correct, and 485 KiB plus a host sort each: a number that climbs is a request
    /// mix the seam's K does not cover.
    pub dense_rows: u64,
    /// Seam refusals, by name, from `saber_serve::sampler::SamplerError`. Each one is followed by
    /// a dense read or — if that is not available either — by the argmax below.
    pub seam_refusals: u64,
    /// Rows a sampler error left on the ARGMAX. Never silent: this is the last resort, it is
    /// logged at error level, and a non-zero count means requests were served greedily that did
    /// not ask to be.
    pub sampler_fallbacks_to_argmax: u64,
    /// Rows whose penalty-id list was wider than the seam carries, and which were therefore read
    /// densely rather than sampled from a gather that would have missed a moved logit.
    pub penalty_lists_too_wide: u64,
    /// Drawn rows with an EXACT tie at the top of the (bf16) logit row, from the sampler's own
    /// `top_ties` (ws-f39). Every mask keeps a tie whole, so even `top_k = 1` is not the argmax
    /// there — prod's `apply_top_k_only` has the same mask.
    pub top_ties: u64,
    /// Of those, the draws that committed a tied id other than the lowest (the argmax's): the
    /// token temperature 0 would not have committed, at a margin of exactly zero (ws-f39). Since
    /// ws-f40 every committed token of a sampled request is a draw, counted once — the accepted
    /// prefix of a verify step included, and a seam refusal read again densely not twice.
    pub top_tie_draws: u64,
    /// Rows drawn under THE CONTROL (`step_keyed_draws`): the pre-F40 law — the accepted prefix
    /// on the argmax, only the last row drawn, the noise keyed by the sequence's STEP. Zero on
    /// every serving run; the control exists so the seeded-invariance gate has a law that fails.
    pub step_keyed_rows: u64,
    /// Steps that committed more than one token for a slot that asked for logprobs. The seam
    /// carries the row that chose the LAST token, so the accepted prefix of a speculative step has
    /// no distribution to report and the step reports none at all rather than a partial list.
    pub logprobs_skipped_on_verify: u64,
    /// Branches forked from a leader's sampled prefill row: the GDN slot set and the one partial
    /// KV page copied device to device, and the branch's own draw from that row.
    pub forks: u64,
    /// Forks refused: no row was kept for that leader, or the device copy failed. The scheduler
    /// aborts the branch, which is a loud no rather than a branch decoding out of a zeroed slot.
    pub fork_refusals: u64,
    /// Forked branches whose FIRST BLOCK was proposed from their own committed token (ws-f38)
    /// and offered to the drafter's mailbox. Zero without a drafter.
    pub fork_blocks: u64,
    /// Forked branches under a drafter whose first block could not be made: the pass refused, or
    /// returned a block for another sequence. The branch stands (its first step verifies
    /// nothing), and the run is unclean — a branch whose speculation silently began one step late
    /// is exactly what the fork under the drafter was built to prevent.
    pub fork_block_failures: u64,
    /// Prefill chunks whose prompt rows were scored for `prompt_logprobs` (V7).
    pub prompt_logprob_chunks: u64,
    /// Prompt rows scored.
    pub prompt_logprob_rows: u64,
    /// Chunks whose request asked and whose rows could NOT be scored; each is a request the
    /// scheduler ends with an error rather than a `null`.
    pub prompt_logprob_refusals: u64,
    /// Per-shape launch counts, as measured. `explain`-shaped output for the report.
    pub launches_by_shape: BTreeMap<StepRows, u32>,
}

impl EngineCounters {
    /// One line for a log or a report.
    pub fn line(&self) -> String {
        format!(
            "prefill {} ({} silent), decode {} ({} sequences, {} padded rows), path graph {} / \
             eager {}, feed device {} / host {}, launches {}, budget mismatches {}, undeclared \
             shapes {}, shape violations {}, forward errors {}, unknown seqs {}, oversized \
             refusals {}, band steps {} (live rows {} / band rows {})",
            self.prefill_steps,
            self.silent_prefill_steps,
            self.decode_steps,
            self.decode_sequences,
            self.padded_rows,
            self.graph_steps,
            self.eager_steps,
            self.device_fed_steps,
            self.host_fed_steps,
            self.launches,
            self.budget_mismatches,
            self.undeclared_shapes,
            self.shape_violations,
            self.forward_errors,
            self.unknown_sequences,
            self.oversized_step_refusals,
            self.band_steps,
            self.band_live_rows,
            self.band_rows,
        )
    }

    /// The parking half, for a report that cares about it.
    pub fn park_line(&self) -> String {
        format!(
            "parks {} ({} refused), restores {} ({} refused)",
            self.parks, self.park_failures, self.restores, self.restore_failures
        )
    }

    /// The sampler's own which-path line.
    pub fn sampler_line(&self) -> String {
        format!(
            "rows greedy {} / sampled {} / dense {}, seam refusals {}, penalty lists too wide {},              argmax fallbacks {}, logprobs skipped on a verify step {}, ties at the top {}              (drawn off the argmax {}), step-keyed control rows {}",
            self.greedy_rows,
            self.sampled_rows,
            self.dense_rows,
            self.seam_refusals,
            self.penalty_lists_too_wide,
            self.sampler_fallbacks_to_argmax,
            self.logprobs_skipped_on_verify,
            self.top_ties,
            self.top_tie_draws,
            self.step_keyed_rows
        )
    }

    /// The `prompt_logprobs` half (V7). All zero on a serve nobody asked for them on — which is
    /// the claim that a request without the flag pays nothing.
    pub fn prompt_line(&self) -> String {
        format!(
            "prompt logprobs: chunks {}, rows {}, refused {}",
            self.prompt_logprob_chunks, self.prompt_logprob_rows, self.prompt_logprob_refusals
        )
    }

    /// The sequence-group half.
    pub fn fork_line(&self) -> String {
        format!(
            "forks {} ({} refused), first blocks {} ({} failed)",
            self.forks, self.fork_refusals, self.fork_blocks, self.fork_block_failures
        )
    }

    /// The proposal half, and the stale-anchor law's which-path line (ws-f35).
    ///
    /// `stale` is a rate, not a fault: at temperature 0 it must be 0 (a greedy row commits the
    /// argmax the drafter was anchored on), and above it the count is the number of steps whose
    /// host draw moved the anchor. `earned` is only ever non-zero under the control.
    pub fn proposal_line(&self) -> String {
        format!(
            "pass proposals {} offered ({} made after the commit), {} dropped with a STALE \
             ANCHOR (the in-step control's law), {} stale-anchor DEFECT(s) after the commit, {} \
             collection failure(s); stale blocks verified {} for {} accepted token(s)",
            self.pass_proposals,
            self.proposals_after_commit,
            self.pass_proposals_stale,
            self.stale_anchor_defects,
            self.proposal_collect_failures,
            self.stale_anchor_drafts,
            self.stale_anchor_accepted
        )
    }

    /// FN69: the prompt-lookup source's census line.
    pub fn lookup_line(&self) -> String {
        crate::lookup::LookupCounts {
            lookups: self.lookup_lookups,
            hits: self.lookup_hits,
            used: self.lookup_used,
            lookup_drafted: self.lookup_drafted,
            lookup_accepted: self.lookup_accepted,
            drafter_drafted: self.lookup_drafter_drafted,
            drafter_accepted: self.lookup_drafter_accepted,
        }
        .line()
    }

    /// Whether every step this executor ran was a clean one. The structural gate's single question.
    pub fn clean(&self) -> bool {
        self.budget_mismatches == 0
            && self.undeclared_shapes == 0
            && self.shape_violations == 0
            && self.forward_errors == 0
            && self.unknown_sequences == 0
            && self.oversized_step_refusals == 0
            && self.park_failures == 0
            && self.restore_failures == 0
            // A row that asked to be sampled and was answered by the argmax anyway is a wrong
            // answer, not a slow one.
            && self.sampler_fallbacks_to_argmax == 0
            // A branch the engine refused to fork is a choice the client asked for and will not
            // get; the scheduler aborts it, and that is not a clean run.
            && self.fork_refusals == 0
            && self.fork_block_failures == 0
            // A block made after the commit from a token the slot did not commit is the defect
            // ws-f37 exists to make impossible, and one that could not be collected is a step
            // whose speculation was silently lost.
            && self.stale_anchor_defects == 0
            && self.proposal_collect_failures == 0
    }
}

/// What a leader's first sampled row was, kept until its branches are forked from it.
///
/// One entry per sequence, written at the step that samples the PROMPT's first token (the last
/// prefill chunk, or — with no captured prefill bucket — the one decode-shaped step that feeds the
/// prompt) and dropped at `release`. It is not written on any other step: a fork happens once per
/// branch and at one known moment, and a row kept on every step would be a per-step host cost for
/// something almost no request uses.
#[derive(Clone, Debug)]
struct ForkRow {
    /// The seam the head wrote for the row the first token came from. `None` on a greedy row,
    /// which the kernel skips, and on a row the head produced nothing for.
    row: Option<saber_serve::sampler::DeviceRow>,
    /// Which row of that step it was, so the dense fallback can ask the pass for it by index.
    row_index: u32,
    /// The LEADER's sampling policy. A branch of the same request has the same parameters and the
    /// same seed and differs only in `choice_index` — which is the whole of "one distribution, `n`
    /// keys" and is why the branch does not get to supply its own.
    sampling: SlotSampling,
    /// The token the leader committed. A greedy fork's answer, unchanged: at temperature 0 every
    /// branch of a group is the same argmax.
    token: TokenId,
}

/// The noise that decides output position `position` of choice `choice` (ws-f40): a function of
/// the request's seed and of that decision alone, or entropy for an unseeded request.
fn position_noise(seed: Option<u64>, position: u64, choice: u32) -> Box<dyn NoiseSource> {
    match seed {
        Some(seed) => Box::new(SeededNoise::new(seed, position, choice)),
        None => Box::new(EntropyNoise::new()),
    }
}

/// One sampled slot's draws for one decode step — the [`crate::forward::RowDecider`] the pass
/// walks its verify rows with (ws-f40).
///
/// Row `j` decides output position `position + j`, so its draw is keyed by that and by nothing the
/// schedule controls. The penalties see the history as a width-1 run would at that position: the
/// output so far and the drafts the walk has already accepted (`draft[..j]`). Every row it decides
/// is kept, in order, for the executor to report, dump and count once the step is back.
///
/// Public since FN16: the Flash-Next serve plane walks its MTP verify rows with THIS decider (one
/// drawn-row law for both engines, not a fork).
#[derive(Debug)]
pub struct SlotDraws {
    sampler: std::sync::Arc<Sampler>,
    params: RowParams,
    seed: Option<u64>,
    choice: u32,
    /// The output position row 0 decides.
    position: u64,
    penalties: Option<std::sync::Arc<saber_sched::PenaltyHistory>>,
    /// The block this step verifies. EVERY row of it is walked, a request's stop id and its last
    /// position included (ws-f41): prod's rejection sampler verifies the whole block and counts
    /// what it accepts before the scheduler cuts at `max_tokens` or a stop, and so does Saber's
    /// greedy law. Rows past the end are drawn, never served (the scheduler drops them exactly as
    /// it drops a greedy step's), and since every draw is keyed by its own position no served
    /// token moves. F40's walk stopped at the end and counted the matched draft there as the
    /// recovery token: 156 + 164 of Q3's 400 corpus requests ended on one, with 1,077 drafts
    /// behind them never verified — the "sampled gap" at 64 tokens per request.
    draft: Vec<TokenId>,
    /// A row whose penalty list the seam could not carry: every row is read densely instead,
    /// which the walk expresses by deciding none (the executor draws row 0 from the dense read).
    dense: bool,
    decided: std::sync::Mutex<Vec<Sampled>>,
    /// P27-W2-3: the accepted path a TREE walk last offered
    /// ([`crate::forward::RowDecider::decide_path`]); `None` on a chain walk. The dump's per-row
    /// draft reads it in place of `draft` (row `j < depth` offered the path's node `j`; the last
    /// row's offer was the node's children, recorded as none).
    walked: std::sync::Mutex<Option<Vec<TokenId>>>,
}

impl SlotDraws {
    /// The draws of one sampled slot's verify rows (FN16's public constructor; the 27B's
    /// executor builds the same fields in place): row `j` decides output position
    /// `sampling.position + j`, keyed by `(seed, position, choice)`, the penalties over the output
    /// so far + `draft[..j]`.
    #[must_use]
    pub fn new(
        sampler: std::sync::Arc<Sampler>,
        params: RowParams,
        sampling: &saber_sched::SlotSampling,
        draft: Vec<TokenId>,
    ) -> Self {
        SlotDraws {
            sampler,
            params,
            seed: sampling.params.seed,
            choice: sampling.choice_index,
            position: sampling.position,
            penalties: sampling.penalties.clone(),
            draft,
            dense: false,
            decided: std::sync::Mutex::new(Vec::new()),
            walked: std::sync::Mutex::new(None),
        }
    }

    /// What the walk decided, in row order, taken.
    pub fn take(&self) -> Vec<Sampled> {
        std::mem::take(&mut *self.decided.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

impl crate::forward::RowDecider for SlotDraws {
    fn decide(&self, j: usize, seam: Option<&saber_serve::sampler::DeviceRow>) -> Option<TokenId> {
        self.decide_after(j, &self.draft[..j.min(self.draft.len())], seam)
    }

    fn decide_path(
        &self,
        j: usize,
        path: &[TokenId],
        seam: Option<&saber_serve::sampler::DeviceRow>,
    ) -> Option<TokenId> {
        // The dump's "draft this row was asked to accept" is the tree path's next node: kept
        // per depth as the walk descends (row `j`'s offer is unknown until the walk returns, so
        // the path's own tokens are what is recorded for rows `0..j`).
        {
            *self.walked.lock().unwrap_or_else(|p| p.into_inner()) = Some(path.to_vec());
        }
        self.decide_after(j, path, seam)
    }
}

impl SlotDraws {
    /// Row `j`'s draw over `seam` with `accepted` (the tokens accepted before row `j` in this
    /// step) as the penalties' history past the output.
    fn decide_after(
        &self,
        j: usize,
        accepted: &[TokenId],
        seam: Option<&saber_serve::sampler::DeviceRow>,
    ) -> Option<TokenId> {
        if self.dense {
            return None;
        }
        let seam = seam?;
        let history: Vec<TokenId>;
        let ctx = match &self.penalties {
            None => PenaltyContext::default(),
            Some(h) => {
                history = h.output.iter().chain(accepted).copied().collect();
                PenaltyContext {
                    prompt: &h.prompt,
                    output: &history,
                }
            }
        };
        let mut noise = position_noise(self.seed, self.position + j as u64, self.choice);
        match self
            .sampler
            .sample(seam, &self.params, &ctx, noise.as_mut())
        {
            Ok(out) => {
                let token = out.token_id;
                self.decided
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(out);
                Some(token)
            }
            // Counted by the sampler (`seam_refusals`); the executor reads this row densely.
            Err(_) => None,
        }
    }
}

impl<R: ForwardRunner> RealExecutor<R> {
    /// D4: a sequence left (released, or its parked image dropped). The request's grids go when
    /// no live or parked sequence of it remains.
    fn forget_image_seq(&mut self, seq: SeqId, parked: bool) {
        let r = if parked {
            self.parked_image_seqs.remove(&seq)
        } else {
            self.image_seqs.remove(&seq)
        };
        if let Some(r) = r {
            let held = self.image_seqs.values().any(|x| *x == r)
                || self.parked_image_seqs.values().any(|x| *x == r);
            if !held {
                self.vision_pending.remove(&r);
            }
        }
    }
}

/// A real executor: the scheduler's steps, a whole-model forward underneath.
#[derive(Debug)]
pub struct RealExecutor<R: ForwardRunner> {
    forward: R,
    scalar_context_tail: bool,
    counters: EngineCounters,
    /// Sequences the engine has session state for, and their prompt length.
    live: BTreeMap<SeqId, u32>,
    /// D4: a request's image grids and the tower run that left its rows, from `encode_images`
    /// until its leader is admitted.
    vision_pending: BTreeMap<saber_serve::engine::RequestId, (Vec<[u32; 3]>, u64)>,
    /// D4: live image sequences and their request, so the request's grids are forgotten when its
    /// last sequence goes.
    image_seqs: BTreeMap<SeqId, saber_serve::engine::RequestId>,
    /// D4: the same for PARKED image sequences (their group comes back under new sequence ids).
    parked_image_seqs: BTreeMap<SeqId, saber_serve::engine::RequestId>,
    /// The prompt's first sampled row, per sequence, for as long as a branch could still be
    /// forked from it. See [`ForkRow`].
    fork_rows: BTreeMap<SeqId, ForkRow>,
    /// What proposes the draft tokens a verify step checks.
    ///
    /// Owned here rather than inside the forward because drafting is a SCHEDULER-side concern: the
    /// verify width arrives on `BatchSlot`, the acceptance is reported back on `SeqStep`, and the
    /// pass's job is only to run `k + 1` rows and say what the model wanted at each. [`NoDrafter`]
    /// is the default and is what makes SPEC=0 a special case of the verify path rather than a
    /// separate one.
    drafter: Box<dyn Drafter>,
    /// Session transfers issued and not yet finished, of either kind. 0..=2, and a count rather
    /// than a flag so that a leak shows up as a number.
    in_flight: usize,
    /// Of those, the restores. Tracked separately because with two staging slots a restore is
    /// blocked only by another restore.
    restores_in_flight: usize,
    /// The reference server's V1 sampler, gated against prod's own on the host (`saber_serve::sampler`). Shared
    /// with each sampled slot's [`SlotDraws`], which the pass runs inside the step (ws-f40), so
    /// its counters are one set whoever drew.
    sampler: std::sync::Arc<Sampler>,
    /// THE CONTROL for the seeded-invariance law (ws-f40), and never the serving default.
    ///
    /// `false` — the law — decides every verify row of a sampled slot by a draw keyed by the
    /// output position it decides (`crate::forward::drawn_rows`). `true` is the engine before
    /// ws-f40: the accepted prefix on the argmax, only the last row drawn, keyed by the
    /// sequence's step — which a park, a crowd or a different verify width moves.
    step_keyed_draws: bool,
    /// The widest penalty-id list the head's seam carries. A row whose penalties move more ids
    /// than this is read densely instead, counted, rather than sampled from a gather that would
    /// have missed one.
    penalty_capacity: usize,
    /// A DIAGNOSTIC: every sampled row's seam, its parameters, its noise key and the token it
    /// produced, one JSON object per line.
    ///
    /// It exists because "the served stream is what the sampler decided" is otherwise unfalsifiable
    /// from outside: the row the device wrote and the token the client received are separated by
    /// the whole serving plane. With this, a checker rebuilds the row, re-runs the shipped sampler
    /// against the recorded noise key and compares ids — and draws ten thousand more from the same
    /// row for a distribution test (`tools/gates/sampler_stream.sh`). Off unless a flag asks for
    /// it; the serving path never writes it.
    dump: Option<std::io::BufWriter<std::fs::File>>,
    /// THE CONTROL for the stale-anchor law (ws-f35), and nothing else.
    ///
    /// `false` — the default and the law — drops a block whose anchor the step did not commit.
    /// `true` offers it anyway, which is what this engine did until ws-f35, so the two arms can be
    /// run back to back over the same corpus and the block's worth measured rather than assumed.
    keep_stale_anchor_blocks: bool,
    /// Sequences whose MAILBOX currently holds a block made from an anchor the step did not
    /// commit. Only ever non-empty under the control above; it is what lets the control arm say
    /// what those blocks earned instead of only how many there were.
    stale_blocks: std::collections::BTreeSet<SeqId>,
    /// THE CONTROL for the bounded-restore law (ws-k18), and never the serving default: every
    /// restore is refused with its image handed back, so the scheduler's retry-then-give-up path
    /// runs on the card exactly as a restore that always fails would drive it.
    fail_restores: bool,
    /// The live census (ws-v8): [`crate::census::engine_defs`], published at the end of every
    /// step once a `/metrics` scrape has armed it, and read by the scrape without a lock.
    census: std::sync::Arc<saber_serve::census::CensusMirror>,
    /// C9: run a mixed step's chunk and decode half as ONE fused forward
    /// (`--engine-mixed-prefill fused`); `false` is C4's serial form, the bit-exact control.
    fused_mixed: bool,
    /// **P27-3: a prompt's last-chunk drafter store + first proposal, owed** — stashed by the
    /// runner under [`ForwardRunner::defers_prompt_store`] and NOT collected before the chunk's
    /// token was handed back. [`Self::settle_deferred`] collects it at the head of every call
    /// that reaches the runner, so it still runs before any other runner call (the stash law)
    /// and before the next decode step asks the drafter for a block: the mailbox, the drafter's
    /// cache and the target's bytes are the on-path ones, and only the token left earlier.
    deferred_store: Option<DeferredCollect>,
    /// P27-3: how many such stashes were collected late (the arm's which-path number).
    deferred_store_settles: u64,
    /// P27-3: the prefill in flight is an UNMIXED one (`execute`'s prefill arm) — the only call
    /// whose collection may be deferred. A mixed or packed call runs further halves into the
    /// runner after its chunk, and the runner holds one stash at a time, so it collects in place.
    unmixed_prefill: bool,
}

/// P27-3: what [`RealExecutor::settle_deferred`] collects — the prefill's stash and the token the
/// sampler committed for it.
#[derive(Clone, Debug)]
struct DeferredCollect {
    seq: SeqId,
    pending: Vec<SeqId>,
    committed: Vec<(SeqId, TokenId)>,
    last: Option<TokenId>,
}

impl<R: ForwardRunner> RealExecutor<R> {
    /// Use the ordinary one-row verifier when a full rectangular proposal would cross max_ctx.
    pub fn enable_scalar_context_tail(&mut self) {
        self.scalar_context_tail = true;
    }

    /// C9: the fused mixed forward on or off (`--engine-mixed-prefill fused` / `serial`). Only
    /// meaningful when the scheduler mixes at all.
    pub fn set_fused_mixed(&mut self, on: bool) {
        self.fused_mixed = on;
    }

    /// Wraps a forward pass, with no speculation.
    pub fn new(forward: R) -> Self {
        Self::with_drafter(forward, Box::new(NoDrafter))
    }

    /// Wraps a forward pass, with a drafter.
    pub fn with_drafter(forward: R, drafter: Box<dyn Drafter>) -> Self {
        Self {
            forward,
            scalar_context_tail: false,
            counters: EngineCounters::default(),
            live: BTreeMap::new(),
            vision_pending: BTreeMap::new(),
            image_seqs: BTreeMap::new(),
            parked_image_seqs: BTreeMap::new(),
            fork_rows: BTreeMap::new(),
            drafter,
            in_flight: 0,
            restores_in_flight: 0,
            sampler: std::sync::Arc::new(Sampler::new()),
            step_keyed_draws: false,
            penalty_capacity: saber_kernels::head::HeadRules::prod_head()
                .map(|r| r.knobs.device_row_penalty_max as usize)
                .unwrap_or(0),
            dump: None,
            keep_stale_anchor_blocks: false,
            stale_blocks: std::collections::BTreeSet::new(),
            fail_restores: false,
            fused_mixed: false,
            deferred_store: None,
            deferred_store_settles: 0,
            unmixed_prefill: false,
            census: std::sync::Arc::new(saber_serve::census::CensusMirror::new(
                crate::census::engine_defs(),
            )),
        }
    }

    /// P27-3: prompt stores collected AFTER their token was handed back (`--drafter-store
    /// deferred`); 0 on the on-path control, one per prompt on the arm.
    pub fn deferred_store_settles(&self) -> u64 {
        self.deferred_store_settles
    }

    /// P27-3: whether a prompt's store is still owed (the readiness gate's question: a decode step
    /// never runs while it is — [`Self::settle_deferred`] runs first).
    pub fn store_owed(&self) -> bool {
        self.deferred_store.is_some()
    }

    /// **P27-3: collect the owed prompt store + first proposal**, before anything else reaches the
    /// runner. A no-op on the control. The proposal is offered to the mailbox exactly as the
    /// on-path chunk offers it, so the next decode step drafts from the same block.
    fn settle_deferred(&mut self) {
        let Some(d) = self.deferred_store.take() else {
            return;
        };
        let t = std::time::Instant::now();
        let made = self.collect_after_commit(&d.pending, &d.committed);
        for p in &made {
            self.offer_after_commit(d.seq, d.last, p);
        }
        self.deferred_store_settles += 1;
        self.counters.prefill_after_us += t.elapsed().as_micros() as u64;
    }

    /// THE CONTROL SWITCH for the bounded-restore law (ws-k18): refuse every restore, image handed
    /// back, as a restore that always fails would be refused. A diagnostic; see `fail_restores`.
    pub fn fail_restores(&mut self, fail: bool) {
        self.fail_restores = fail;
    }

    /// THE CONTROL SWITCH for the stale-anchor law (ws-f35).
    ///
    /// The law is that a block whose anchor the step did not commit is not a proposal for this
    /// sequence at all — it is the continuation of a token the sequence never had. The engine
    /// drops it. `keep(true)` offers it anyway, which is what this engine did before ws-f35, and
    /// it exists so that the two arms can be run over the same corpus in the same session: the
    /// difference between them is what a stale block was worth, measured.
    pub fn keep_stale_anchor_blocks(&mut self, keep: bool) {
        self.keep_stale_anchor_blocks = keep;
    }

    /// THE CONTROL SWITCH for the seeded-invariance law (ws-f40): `true` runs the pre-F40 law
    /// (argmax acceptance, the last row drawn, noise keyed by the step). A diagnostic; see
    /// `step_keyed_draws`. Every row it draws is counted in `step_keyed_rows`.
    pub fn step_keyed_draws(&mut self, control: bool) {
        self.step_keyed_draws = control;
    }

    /// Write every sampled row to `path` as JSONL. See [`RealExecutor::dump`].
    ///
    /// # Errors
    /// The file could not be created.
    pub fn dump_samples_to(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        self.dump = Some(std::io::BufWriter::new(std::fs::File::create(path)?));
        Ok(())
    }

    /// The drafter's name, for the report and the which-path line.
    pub fn drafter_name(&self) -> &'static str {
        self.drafter.name()
    }

    /// Swap the drafter. Returns the old one.
    ///
    /// For a gate that runs several mocks over one engine: a second engine means a second 9.6 GiB
    /// weight upload, and the thing being compared is the drafter and not the load. The sequences in
    /// flight are the caller's problem — a drafter with a script for a sequence the new one has never
    /// heard of proposes a token the model will reject, which is legal and counted, not a fault.
    pub fn set_drafter(&mut self, drafter: Box<dyn Drafter>) -> Box<dyn Drafter> {
        std::mem::replace(&mut self.drafter, drafter)
    }

    /// What the pass is, including the PARTIAL banner when it is one.
    pub fn facts(&self) -> &ForwardFacts {
        self.forward.facts()
    }

    /// The counters. The sampler's own are folded in, so one snapshot answers "which branch
    /// decided these tokens" without two places to look.
    pub fn counters(&self) -> EngineCounters {
        let mut c = self.counters.clone();
        for (name, value) in self.sampler.counters.snapshot() {
            match name {
                "seam_refusals" => c.seam_refusals = value,
                "top_ties" => c.top_ties = value,
                "top_tie_draws" => c.top_tie_draws = value,
                _ => {}
            }
        }
        c
    }

    /// The MODEL's vocabulary, which is not this rank's above TP = 1: `top_k` is normalised
    /// against it, and a `top_k` resolved against half a vocabulary would silently become "no
    /// truncation at all" — the one request shape a top-K seam cannot serve.
    fn vocab(&self) -> usize {
        (self.forward.facts().vocab_total() as usize).max(1)
    }

    /// How a row is to be sampled, resolved against the vocabulary this pass declares.
    ///
    /// `None` means greedy — the argmax decides it and the seam kernel skips the row.
    fn row_params(&self, sampling: &SlotSampling) -> Option<RowParams> {
        let params = RowParams::resolve(&sampling.params, self.vocab());
        (!params.is_greedy()).then_some(params)
    }

    /// The request the head is handed for this row, and whether the row must be read densely
    /// whatever the seam says (a penalty list wider than the seam can gather).
    fn row_request(&mut self, sampling: &SlotSampling) -> (RowRequest, bool) {
        let Some(params) = self.row_params(sampling) else {
            // A GREEDY ROW THAT ASKED FOR LOGPROBS still needs the seam. The token is the
            // argmax's and nothing below changes that — but the numbers a client asked to be told
            // about the token are the raw logits' `log_softmax`, and the argmax kernel does not
            // produce them. The kernel skips a row whose temperature is not positive, so this asks
            // for **T = 1**, which is exactly the scale logprobs are reported at. Prod reports
            // logprobs for a greedy request; before V3 this returned `RowRequest::greedy()` and
            // the reply carried none.
            let greedy = RowParams::resolve(&sampling.params, self.vocab());
            if greedy.logprobs.is_some() {
                return (
                    RowRequest {
                        temperature: Some(1.0),
                        penalty_ids: Vec::new(),
                    },
                    false,
                );
            }
            return (RowRequest::greedy(), false);
        };
        let ids = match &sampling.penalties {
            None => Vec::new(),
            Some(h) => params.penalty_ids(&PenaltyContext {
                prompt: &h.prompt,
                output: &h.output,
            }),
        };
        if ids.len() > self.penalty_capacity {
            // Fail loud rather than gather a prefix: a penalty the seam did not carry is a logit
            // the host would sample as if nothing had moved it.
            self.counters.penalty_lists_too_wide += 1;
            return (
                RowRequest {
                    temperature: None,
                    penalty_ids: Vec::new(),
                },
                true,
            );
        }
        (
            RowRequest {
                temperature: Some(params.temperature),
                penalty_ids: ids,
            },
            false,
        )
    }

    /// Decide one slot's committed tokens: the argmax as the device computed them for a greedy
    /// row; for a sampled row, the draws the pass's walk already made (ws-f40) — or, on a step with
    /// no walk (a prefill, a fork, the control), the host sampler's draw in place of the LAST one.
    ///
    /// `row_index` is the row the last token came from, within the step that just ran, so the
    /// dense fallback can ask the pass for exactly that row. `draws` is the slot's walk, when the
    /// step ran one.
    fn sample_commit(
        &mut self,
        sampling: &SlotSampling,
        commit: &mut SlotCommit,
        row_index: u32,
        force_dense: bool,
        draws: Option<&SlotDraws>,
    ) -> Option<Vec<SampledLogprobs>> {
        let Some(params) = self.row_params(sampling) else {
            // GREEDY. The token is the argmax's and stays the argmax's — nothing below runs, which
            // is the whole of "temperature 0 is the step it was". The one thing a greedy row may
            // still owe the client is the numbers: `logprobs` is not a sampling knob, it is a
            // report, and prod reports it for a greedy request too.
            self.counters.greedy_rows += 1;
            let greedy = RowParams::resolve(&sampling.params, self.vocab());
            greedy.logprobs?;
            return self.report_logprobs(commit, &greedy);
        };
        let (prompt, output): (&[TokenId], &[TokenId]) = match &sampling.penalties {
            None => (&[], &[]),
            Some(h) => (h.prompt.as_slice(), h.output.as_slice()),
        };
        let n = commit.tokens.len();
        // THE LAST ROW's position: the committed prefix's rows decide `position ..`, in order.
        let last_position = sampling.position + n.saturating_sub(1) as u64;

        // THE WALK'S DRAWS (ws-f40). Every row the pass's walk decided is a committed token, in
        // order; the last row is either among them or was left undecided (a seam refusal) and is
        // read densely below under the same position key.
        let mut from_walk: Option<Sampled> = None;
        let mut undecided_last = false;
        if let Some(d) = draws {
            let mut decided = d.take();
            let agrees = decided.len() <= n
                && decided
                    .iter()
                    .zip(&commit.tokens)
                    .all(|(o, t)| o.token_id == *t);
            if !agrees || decided.len() + 1 < n {
                // The pass committed tokens the walk did not decide: a protocol fault, counted.
                self.counters.shape_violations += 1;
                tracing::error!(
                    seq = %commit.seq,
                    decided = decided.len(),
                    committed = n,
                    "a sampled slot's committed tokens are not the walk's draws"
                );
            }
            self.counters.sampled_rows += decided.len() as u64;
            // P27-W2-3: a TREE walk's offers are its accepted path, not the chain draft.
            let walked = d.walked.lock().unwrap_or_else(|p| p.into_inner()).take();
            for (j, o) in decided.iter().enumerate().take(n) {
                let row = commit.rows.get(j).cloned().flatten();
                // The draft this row was asked to accept (ws-f41): `p'(draft)` is the law's rate,
                // and the dump is where it is measured against the walk's own outcome.
                let draft = match &walked {
                    Some(path) => path.get(j),
                    None => d.draft.get(j),
                }
                .copied()
                .filter(|t| *t != crate::draft::RecordedDrafter::WRONG);
                self.record(
                    sampling,
                    &params,
                    commit.seq,
                    row.as_ref(),
                    o.token_id,
                    false,
                    sampling.position + j as u64,
                    draft,
                );
            }
            if agrees && decided.len() == n {
                from_walk = decided.pop();
            } else {
                undecided_last = true;
            }
        }

        let sampled = if let Some(out) = from_walk {
            Some(out)
        } else {
            let ctx = PenaltyContext { prompt, output };
            // THE KEY. The law's is the position this row decides; the control's (the pre-F40
            // engine) is the sequence's step, which a park, a crowd or a width moves.
            let mut noise: Box<dyn NoiseSource> = if self.step_keyed_draws {
                self.counters.step_keyed_rows += 1;
                match sampling.params.seed {
                    Some(seed) => Box::new(SeededNoise::new(
                        seed,
                        sampling.step_index,
                        sampling.choice_index,
                    )),
                    None => Box::new(EntropyNoise::new()),
                }
            } else {
                position_noise(sampling.params.seed, last_position, sampling.choice_index)
            };
            // A row the walk already offered the seam and had refused goes straight to the dense
            // read: the seam's answer would be the same refusal, counted twice.
            let seam = if force_dense || undecided_last {
                Err(SamplerError::SeamEmpty)
            } else {
                match commit.last_row() {
                    Some(row) => self.sampler.sample(row, &params, &ctx, noise.as_mut()),
                    None => Err(SamplerError::SeamEmpty),
                }
            };
            let mut read_densely = false;
            let sampled = match seam {
                Ok(out) => {
                    self.counters.sampled_rows += 1;
                    Some(out)
                }
                Err(err) => {
                    // The seam could not decide this row. The dense read is the honest answer: the
                    // whole vocabulary, the same pipeline, and a counter that says it happened.
                    match self.forward.dense_row(row_index) {
                        Ok(mut dense) => {
                            match self.sampler.sample_dense(
                                &mut dense,
                                &params,
                                &ctx,
                                noise.as_mut(),
                            ) {
                                Ok(out) => {
                                    self.counters.dense_rows += 1;
                                    read_densely = true;
                                    Some(out)
                                }
                                Err(e) => {
                                    self.counters.sampler_fallbacks_to_argmax += 1;
                                    tracing::error!(
                                        seq = %commit.seq, error = %e,
                                        "the dense row could not be sampled either; committing \
                                         the argmax, which is NOT what this request asked for"
                                    );
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            self.counters.sampler_fallbacks_to_argmax += 1;
                            tracing::error!(
                                seq = %commit.seq, seam = %err, dense = %e,
                                "the seam refused and no dense row is available; committing the \
                                 argmax, which is NOT what this request asked for"
                            );
                            None
                        }
                    }
                }
            };
            let sampled = sampled?;
            if let Some(last) = commit.tokens.last_mut() {
                *last = sampled.token_id;
            }
            let row = commit.last_row().cloned();
            self.record(
                sampling,
                &params,
                commit.seq,
                row.as_ref(),
                sampled.token_id,
                read_densely,
                last_position,
                None,
            );
            Some(sampled)
        };
        let last = sampled?.logprobs?;
        if commit.tokens.len() == 1 {
            return Some(vec![last]);
        }
        // A VERIFY STEP'S ACCEPTED PREFIX. The seam kernel runs on every verify row, so each
        // committed token has a distribution of its own — the model's answer at that position,
        // which the acceptance law compared a draft against. They are REPORTED here off the raw
        // row (the walk's draws already chose them); only the LAST entry is the draw's own report.
        //
        // All or nothing, because the serving plane lines `logprobs.content` up with the text: a
        // prefix row the head skipped, or one the shard merge refused, leaves the whole step's
        // list unreported and bumps the counter rather than shipping a list that is one short.
        let mut out = self.report_logprobs_upto(commit, &params, commit.tokens.len() - 1)?;
        out.push(last);
        Some(out)
    }

    /// The reported logprobs of every committed token, for a row nobody drew from.
    ///
    /// The greedy path: the tokens are the argmax's and this says what the model thought of them.
    fn report_logprobs(
        &mut self,
        commit: &SlotCommit,
        params: &RowParams,
    ) -> Option<Vec<SampledLogprobs>> {
        self.report_logprobs_upto(commit, params, commit.tokens.len())
    }

    /// The first `upto` committed tokens' reported logprobs, or `None` with the counter bumped.
    ///
    /// ALL OR NOTHING, because the serving plane lines `logprobs.content` up with the text: a row
    /// the head skipped, or one the shard merge refused, leaves the whole step's list unreported
    /// rather than shipping a list that is one short.
    fn report_logprobs_upto(
        &mut self,
        commit: &SlotCommit,
        params: &RowParams,
        upto: usize,
    ) -> Option<Vec<SampledLogprobs>> {
        let mut out: Vec<SampledLogprobs> = Vec::with_capacity(commit.tokens.len());
        for (i, token) in commit.tokens.iter().enumerate().take(upto) {
            let Some(Some(row)) = commit.rows.get(i) else {
                self.counters.logprobs_skipped_on_verify += 1;
                return None;
            };
            match self.sampler.logprobs_at(row, params, *token) {
                Ok(lp) => out.push(lp),
                Err(_) => {
                    self.counters.logprobs_skipped_on_verify += 1;
                    return None;
                }
            }
        }
        Some(out)
    }

    /// Sequences the engine holds state for.
    pub fn live_sequences(&self) -> usize {
        self.live.len()
    }

    /// The forward, for a caller that owns the engine and wants to drive it directly — which is how
    /// the gate compares the serve path against the pass itself, and (ws-p9) how it ATTRIBUTES a
    /// divergence the concurrency arms found: the same prompt alone and in a batch with the probe
    /// on is a direct drive of the pass and cannot be expressed as a request. A caller holds the
    /// executor's lock while it drives, so nothing steps underneath it, and anything it admits it
    /// releases.
    pub fn forward_mut(&mut self) -> &mut R {
        &mut self.forward
    }

    /// One step's shape, token count and launch count against what the plan declares.
    ///
    /// `want` is what the batch says the step should have been; `out.rows` is what the pass says it
    /// ran. Every half is checked: a pass that ran the wrong shape and then issued exactly that
    /// shape's declared launches would otherwise pass a launch-count assertion.
    fn check_step(&mut self, want: StepRows, live_slots: usize, out: &StepOutcome) {
        self.counters.launches += u64::from(out.launches);
        self.counters
            .launches_by_shape
            .insert(out.rows, out.launches);
        match out.path {
            StepPath::Graph => self.counters.graph_steps += 1,
            StepPath::Eager | StepPath::Fused => self.counters.eager_steps += 1,
        }
        match out.feed {
            TokenFeed::Device => self.counters.device_fed_steps += 1,
            TokenFeed::Host => self.counters.host_fed_steps += 1,
        }
        let padded_up = out.rows.rows >= want.rows;
        let tokens_owed = if want.samples { live_slots } else { 0 };
        if out.rows.phase != want.phase
            || out.rows.samples != want.samples
            || !padded_up
            || out.commits.len() != tokens_owed
        {
            self.counters.shape_violations += 1;
            tracing::error!(
                ran = %out.rows,
                asked = %want,
                slots_committed = out.commits.len(),
                owed = tokens_owed,
                "a step ran at a shape that is not the one the batch asked for"
            );
        }
        match self.forward.declared_launches(out.rows) {
            // C9: a fused step's budget is no planned shape's; the engine asserted it.
            _ if out.path == StepPath::Fused => {}
            Some(declared) if declared == out.launches => {}
            Some(declared) => {
                self.counters.budget_mismatches += 1;
                tracing::error!(
                    shape = %out.rows,
                    issued = out.launches,
                    declared,
                    path = %out.path,
                    "a step issued a different number of dispatches than the plan declares"
                );
            }
            None => {
                self.counters.undeclared_shapes += 1;
                tracing::error!(
                    shape = %out.rows,
                    issued = out.launches,
                    "a step ran at a shape the plan never declared a budget for"
                );
            }
        }
    }

    /// One line of the sample dump, when one was asked for.
    ///
    /// Hand-rolled JSON: every field is a number or a fixed tag, this crate has no serialiser, and
    /// a diagnostic that pulled one in would be a dependency the engine carries for a flag nobody
    /// sets in production.
    ///
    /// One line per DECIDED row (ws-f40): at width > 1 every committed token of a sampled slot is a
    /// draw, and each carries the output `position` its noise was keyed by.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        sampling: &SlotSampling,
        params: &RowParams,
        seq: SeqId,
        row: Option<&saber_serve::sampler::DeviceRow>,
        token: TokenId,
        dense: bool,
        position: u64,
        draft: Option<TokenId>,
    ) {
        use std::io::Write as _;
        let Some(dump) = self.dump.as_mut() else {
            return;
        };
        let join = |v: &[f32]| -> String {
            v.iter()
                .map(|x| format!("{x:e}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let ids = row.map_or(String::new(), |r| {
            r.top_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        });
        let values = row.map_or(String::new(), |r| join(&r.top_values));
        let penalised = row.map_or(String::new(), |r| {
            r.penalised
                .iter()
                .map(|(id, v)| format!("[{id},{v:e}]"))
                .collect::<Vec<_>>()
                .join(",")
        });
        let seed = sampling
            .params
            .seed
            .map_or("null".to_owned(), |s| s.to_string());
        let lse = row.map_or("null".to_owned(), |r| format!("{:e}", r.logsumexp));
        let lse_t = row.map_or("null".to_owned(), |r| {
            r.logsumexp_at_temperature
                .map_or("null".to_owned(), |v| format!("{v:e}"))
        });
        // WHICH BRANCH ANSWERED, not whether a seam exists. A row the seam refused was read
        // densely and sampled from all 248,320 logits, and the seam it CARRIES cannot reproduce
        // that draw — the first run of this dump labelled four such rows `seam` and the replay
        // refused them by name, which is the instrument disagreeing with itself rather than the
        // engine disagreeing with the stream.
        let path = if dense || row.is_none() {
            "dense"
        } else {
            "seam"
        };
        let line = format!(
            concat!(
                "{{\"seq\":{},\"choice\":{},\"step\":{},\"position\":{},\"seed\":{},",
                "\"temperature\":{:e},",
                "\"top_p\":{:e},\"top_k\":{},\"min_p\":{:e},\"presence\":{:e},\"frequency\":{:e},",
                "\"repetition\":{:e},\"vocab\":{},\"path\":\"{}\",\"token\":{},\"draft\":{},",
                "\"lse\":{},\"lse_t\":{},\"top_ids\":[{}],\"top_values\":[{}],\"penalised\":[{}]}}"
            ),
            seq.0,
            sampling.choice_index,
            sampling.step_index,
            position,
            seed,
            params.temperature,
            params.top_p,
            params.top_k,
            params.min_p,
            params.presence_penalty,
            params.frequency_penalty,
            params.repetition_penalty,
            params.vocab_size,
            path,
            token,
            draft.map_or("null".to_owned(), |t| t.to_string()),
            lse,
            lse_t,
            ids,
            values,
            penalised,
        );
        if let Err(e) = writeln!(dump, "{line}") {
            tracing::error!(error = %e, "the sample dump could not be written");
            self.dump = None;
        }
    }

    /// Keep the row a sequence's FIRST token came from, so that a branch can be forked from it.
    ///
    /// Called at the two steps that sample a prompt's first token and at no others (see
    /// [`ForkRow`]). The row is the one the host sampler was handed — at TP = 2 the MERGED row,
    /// not a shard — so a branch's draw runs the same pipeline over the same numbers the leader's
    /// did.
    fn keep_fork_row(&mut self, slot: &BatchSlot, commit: &SlotCommit, row_index: u32) {
        let token = match commit.tokens.last() {
            Some(t) => *t,
            None => return,
        };
        self.fork_rows.insert(
            slot.seq_id,
            ForkRow {
                row: commit.last_row().cloned(),
                row_index,
                sampling: slot.sampling.clone(),
                token,
            },
        );
    }

    /// **THE PROPOSAL, AFTER THE SAMPLER** (ws-f37): run what the step stashed, anchored on the
    /// tokens the slots COMMITTED.
    ///
    /// This is prod's order — `execute_model` samples and only then calls `speculator.propose` —
    /// and it is what the in-step proposal could not be: that one ran from the device's ARGMAX
    /// because the target's hidden states for the step's rows live in `aux.cat` only until the
    /// next forward, and a row with a temperature then committed the host sampler's draw instead.
    /// Nothing between the step and this call writes `aux.cat` (the sampler is host arithmetic
    /// and the dense fallback reads the head's logits), so the same rows are here, and the anchor
    /// is now the token the sequence has.
    ///
    /// A pass that names different slots from the ones the step ran, or refuses, costs the next
    /// step its speculation and never a token; both are counted
    /// (`proposal_collect_failures`), and the pass counts the stash it then abandons.
    fn collect_after_commit(
        &mut self,
        pending: &[SeqId],
        committed: &[(SeqId, TokenId)],
    ) -> Vec<Proposal> {
        if pending.is_empty() {
            return Vec::new();
        }
        if !pending
            .iter()
            .copied()
            .eq(committed.iter().map(|(s, _)| *s))
        {
            self.counters.proposal_collect_failures += 1;
            self.counters.shape_violations += 1;
            tracing::error!(
                ?pending,
                committed = committed.len(),
                "the pass stashed a proposal for different slots from the ones this step \
                 committed; it is not collected"
            );
            return Vec::new();
        }
        match self.forward.propose_after_commit(committed) {
            Ok(p) => p,
            Err(e) => {
                self.counters.proposal_collect_failures += 1;
                tracing::error!(
                    error = %e,
                    "the proposal after the commit was refused; the next step verifies nothing \
                     for these slots"
                );
                Vec::new()
            }
        }
    }

    /// Offer one proposal made after the commit — under the invariant ws-f37 makes TRUE.
    ///
    /// ws-f35's stale-anchor law was a RATE on the in-step path (14 % of proposals at
    /// temperature 0.7). Here it is an invariant: the pass was handed the committed token and
    /// echoes the anchor its drafter was actually fed, so a block whose anchor is not `committed`
    /// is a pass that fed its drafter something else — a DEFECT. It is refused by name, never
    /// offered, counted in `stale_anchor_defects`, and it makes the run unclean.
    fn offer_after_commit(&mut self, seq: SeqId, committed: Option<TokenId>, p: &Proposal) {
        if p.seq != seq {
            self.counters.shape_violations += 1;
            tracing::error!(
                slot = %seq,
                proposal = %p.seq,
                "a proposal made after the commit came back for another slot; it is not offered"
            );
            return;
        }
        self.counters.proposals_after_commit += 1;
        if committed != Some(p.anchor) {
            self.counters.stale_anchor_defects += 1;
            tracing::error!(
                seq = %seq,
                anchor = p.anchor,
                committed = ?committed,
                "STALE ANCHOR DEFECT: a block made AFTER the commit continues a token the slot did \
                 not commit; refused"
            );
            return;
        }
        self.counters.pass_proposals += 1;
        self.drafter.offer(seq, &p.block);
    }

    /// A failed forward produced no slot results. An empty token list *for* a slot would
    /// acknowledge a successful non-final prefill chunk and advance its prompt cursor.
    fn failed() -> StepResult {
        StepResult::default()
    }

    /// One prefill chunk: one sequence, one chunk, a token only on the last.
    fn prefill_step(&mut self, slot: &BatchSlot) -> StepResult {
        // P27-3: belt and braces for the internal halves of a mixed call.
        self.settle_deferred();
        let chunk = slot.chunk.expect("a prefill step carries a chunk");
        let want = StepRows {
            phase: Phase::Prefill,
            rows: chunk.prompt_tokens,
            width: 1,
            samples: chunk.last,
            tree: false,
            ragged: false,
        };
        // Only the LAST chunk samples, so only it asks the head for a seam: a chunk that is not
        // the last would pay for a row nothing reads.
        let (request, force_dense) = if chunk.last {
            self.row_request(&slot.sampling)
        } else {
            (RowRequest::greedy(), false)
        };
        let clock = std::time::Instant::now();
        let out = match self.forward.prefill(
            slot.seq_id,
            &slot.input_tokens,
            chunk.tokens_done_before,
            chunk.last,
            &request,
        ) {
            Ok(o) => o,
            Err(e) => {
                self.counters.forward_errors += 1;
                tracing::error!(seq = %slot.seq_id, error = %e, "prefill chunk failed");
                return Self::failed();
            }
        };
        let forward_done = std::time::Instant::now();
        self.counters.prefill_forward_us += (forward_done - clock).as_micros() as u64;
        self.counters.prefill_live_rows += u64::from(chunk.prompt_tokens);
        self.counters.prefill_plan_rows += u64::from(out.rows.rows);
        self.counters.prefill_steps += 1;
        if !chunk.last {
            self.counters.silent_prefill_steps += 1;
        }
        self.check_step(want, 1, &out);
        let mut commits = out.commits;
        let one = commits
            .first()
            .is_some_and(|c: &SlotCommit| c.tokens.len() == 1);
        // The token the last chunk's PROPOSAL was anchored on: the device's argmax, read before
        // the sampler below can replace it (ws-f35, and the same law as the decode path).
        let proposed_from: Option<TokenId> = commits.first().and_then(|c| c.tokens.last().copied());
        let (tokens, logprobs) = match (chunk.last, one) {
            (true, true) => {
                // The prompt's first token, chosen by the same rule every later step uses: the
                // argmax for a greedy request, the sampler's draw for one with a temperature.
                let commit = commits.first_mut().expect("checked above");
                let lp = self.sample_commit(&slot.sampling, commit, 0, force_dense, None);
                // THE FORK POINT. This row is the prompt's own distribution, and a branch of an
                // `n > 1` request draws its first token from it with a key of its own. Kept until
                // the sequence is released; nothing else keeps a row.
                self.keep_fork_row(slot, commit, 0);
                (commit.tokens.clone(), lp)
            }
            (true, false) => {
                // The pass owes a token on the last chunk. Refusing here is what keeps a silent
                // zero-token finish from looking like a stop.
                self.counters.forward_errors += 1;
                tracing::error!(seq = %slot.seq_id, "the last prefill chunk sampled nothing");
                return Self::failed();
            }
            (false, _) => (Vec::new(), None),
        };
        // WHAT THE LAST CHUNK PROPOSED, offered to the drafter's mailbox (ws-s9).
        //
        // Prod's `execute_model` calls `speculator.propose(...)` after EVERY target step, so its
        // first decode step already verifies a block; Saber proposed on decode steps only and the
        // mailbox was empty once per request — 7.4% of steps over the 200-prompt corpus, each one
        // a whole block of speculation thrown away. A prefill chunk hands back at most ONE block,
        // for the one slot it ran, and only when it made one (`StepOutcome::drafts` is empty
        // otherwise), so the same zip that guards the decode path guards this one.
        //
        // **AND THE ANCHOR IS THE LAW HERE TOO** (ws-f35): the prefill proposal anchors on the
        // prompt's argmax, and a request with a temperature whose draw differs drafts from a token
        // the sequence never had. Same rule, same counter, and it is the request's FIRST block, so
        // a stale one costs the step that speculation buys least anyway.
        let stale = proposed_from.is_some_and(|argmax| tokens.last() != Some(&argmax));
        for (s, d) in std::slice::from_ref(slot).iter().zip(&out.drafts) {
            if stale && !self.keep_stale_anchor_blocks {
                self.counters.pass_proposals_stale += 1;
                continue;
            }
            self.counters.pass_proposals += 1;
            self.drafter.offer(s.seq_id, d);
            if stale {
                self.stale_blocks.insert(s.seq_id);
            }
        }
        // ...OR THE ORDER PROD HAS (ws-f37): the chunk stashed its proposal, and it runs now,
        // from the token the sampler above committed.
        if !out.proposal_pending.is_empty() {
            let last = tokens.last().copied();
            let committed: Vec<(SeqId, TokenId)> =
                last.map(|t| vec![(slot.seq_id, t)]).unwrap_or_default();
            // P27-3: under `--drafter-store deferred` the stash also holds the prompt's last
            // store, and the token goes back FIRST: the collection is owed to the head of the
            // next call (`settle_deferred`). A chunk that scores its prompt rows keeps the
            // on-path order, so the head never runs between the chunk and its proposal.
            if self.forward.defers_prompt_store()
                && self.unmixed_prefill
                && chunk.last
                && slot.sampling.params.prompt_logprobs.is_none()
            {
                self.deferred_store = Some(DeferredCollect {
                    seq: slot.seq_id,
                    pending: out.proposal_pending.clone(),
                    committed,
                    last,
                });
            } else {
                let made = self.collect_after_commit(&out.proposal_pending, &committed);
                for p in &made {
                    self.offer_after_commit(slot.seq_id, last, p);
                }
            }
        }
        // PROMPT LOGPROBS (V7), after the commit and the proposal offer so that neither sees a
        // different step: the chunk's rows are scored against the NEXT prompt token each, and
        // the head runs again over the hidden states the body just wrote. A request that did not
        // ask pays nothing here — not a launch, not a copy.
        let prompt_logprobs = self.score_prompt_rows(slot, chunk);
        self.counters.prefill_after_us += forward_done.elapsed().as_micros() as u64;
        StepResult {
            seqs: vec![SeqStep {
                seq_id: slot.seq_id,
                tokens,
                draft_tokens: 0,
                accepted_tokens: 0,
                logprobs,
                prompt_logprobs,
            }],
        }
    }

    /// The prompt positions one prefill chunk scores, in the reference server's shape, or `None` — which is
    /// either "the request did not ask" or a refusal the scheduler turns into an error.
    fn score_prompt_rows(
        &mut self,
        slot: &BatchSlot,
        chunk: saber_sched::PrefillChunk,
    ) -> Option<Vec<saber_serve::engine::PromptPositionLogprobs>> {
        let num = slot.sampling.params.prompt_logprobs?;
        let num = usize::try_from(num.max(0)).unwrap_or(0);
        let mut targets: Vec<TokenId> = slot.input_tokens.iter().skip(1).copied().collect();
        if !chunk.last {
            match chunk.next_prompt_token {
                Some(t) => targets.push(t),
                None => {
                    self.counters.prompt_logprob_refusals += 1;
                    tracing::error!(seq = %slot.seq_id, "a chunk that is not the last named no next prompt token");
                    return None;
                }
            }
        }
        self.counters.prompt_logprob_chunks += 1;
        if targets.is_empty() {
            return Some(Vec::new());
        }
        let rows = match self.forward.prompt_rows(slot.seq_id, &targets) {
            Ok(r) if r.len() == targets.len() => r,
            Ok(r) => {
                self.counters.prompt_logprob_refusals += 1;
                tracing::error!(seq = %slot.seq_id, got = r.len(), want = targets.len(), "the pass scored the wrong number of prompt rows");
                return None;
            }
            Err(e) => {
                self.counters.prompt_logprob_refusals += 1;
                tracing::error!(seq = %slot.seq_id, error = %e, "the pass could not score this chunk's prompt rows");
                return None;
            }
        };
        let mut out = Vec::with_capacity(rows.len());
        for (r, t) in rows.iter().zip(&targets) {
            if r.target != *t {
                self.counters.prompt_logprob_refusals += 1;
                tracing::error!(seq = %slot.seq_id, "a prompt row came back for a different target");
                return None;
            }
            match saber_serve::prompt_logprobs::position(&r.row, r.target, r.rank, num) {
                Ok(p) => out.push(Some(p)),
                Err(e) => {
                    self.counters.prompt_logprob_refusals += 1;
                    tracing::error!(seq = %slot.seq_id, error = %e, "a prompt row could not be reported");
                    return None;
                }
            }
        }
        self.counters.prompt_logprob_rows += out.len() as u64;
        Some(out)
    }

    /// One decode step for the whole batch.
    fn decode_step(&mut self, batch: &StepBatch<'_>) -> StepResult {
        // P27-3: belt and braces for the internal halves of a mixed call.
        self.settle_deferred();
        let shape = batch.bucket.batch;
        if batch.slots.len() as u32 > shape {
            self.counters.oversized_step_refusals += 1;
            tracing::error!(
                slots = batch.slots.len(),
                bucket = %batch.bucket.label(),
                "more live sequences than the bucket's batch"
            );
            return Self::failed();
        }
        // THE DRAFT. The bucket's verify width is `w = k + 1` rows per sequence: row 0 feeds the
        // token the last step committed and rows `1..=k` feed the drafter's guesses. The drafter may
        // propose fewer than `k` — its own width, or the end of its script — and then this slot would
        // run fewer rows than the bucket has, which the engine refuses rather than padding a
        // sequence's own rows. So the proposal is padded to the bucket's width with a token the model
        // cannot want, which costs the same rows and is honestly reported as drafted.
        let scheduled_width = batch.bucket.spec_width.max(1);
        // A rectangular bucket pads short proposals. At the end of the context those
        // padding rows have no valid KV positions. The single-sequence DSpine profile
        // always plans the ordinary one-row graph, including beside the best16 tree.
        let width = if self.scalar_context_tail
            && shape == 1
            && !batch.bucket.is_band()
            && batch.slots.first().is_some_and(|slot| {
                self.forward
                    .facts()
                    .max_ctx
                    .saturating_sub(slot.context_len)
                    < scheduled_width
            }) {
            1
        } else {
            scheduled_width
        };
        let k = width - 1;
        // A decode ROW BAND (C2): every slot verifies its OWN `BatchSlot::verify_width` rows (the
        // scheduler's per-slot cap, 1 on a first step), the drafter is asked for `w_i - 1` and
        // only TRUNCATES — prod's `del spec_token_ids[w:]`, no per-slot block length — and no
        // slot's own rows are padded. The band's padding is its tail, which the engine's index
        // tables point at nothing.
        let band = batch.bucket.is_band().then(|| batch.bucket.m());
        let mut drafted_rows = 0u64;
        let mut scheduled: Vec<u32> = Vec::with_capacity(batch.slots.len());
        // What each row needs beyond the argmax, decided before anything is launched: the seam
        // kernel reads its temperature and its penalty ids out of buffers this step uploads.
        let requests: Vec<(RowRequest, bool)> = batch
            .slots
            .iter()
            .map(|s| self.row_request(&s.sampling))
            .collect();
        // THE DRAWN LAW (ws-f40): each sampled slot's verify rows are decided inside the step by
        // a draw keyed by the output position each row decides. Not under the control, and never
        // for a greedy slot (no decider: the argmax law, byte for byte what it was).
        let drawn: Vec<Option<RowParams>> = batch
            .slots
            .iter()
            .map(|s| {
                (!self.step_keyed_draws)
                    .then(|| self.row_params(&s.sampling))
                    .flatten()
            })
            .collect();
        let sampler = std::sync::Arc::clone(&self.sampler);
        let mut draws: Vec<Option<std::sync::Arc<SlotDraws>>> =
            Vec::with_capacity(batch.slots.len());
        let slots: Vec<DecodeSlot> = batch
            .slots
            .iter()
            .zip(requests.iter())
            .zip(drawn)
            .map(|((s, (request, dense)), params)| {
                let input = s.input_tokens.last().copied().unwrap_or_default();
                // The SLOT's cap, not the bucket's width. The bucket is the captured shape and is an
                // upper bound; `BatchSlot::verify_width` is what this sequence's acceptance EMA has
                // earned, and the rows past it are padded with a token the model cannot want. Drafting
                // the bucket's width and REPORTING it would silently disable the EMA's climb-out,
                // because `accepted >= drafted` would then need a full block's acceptance rather than a
                // full scheduled one.
                let ask = if band.is_some() {
                    s.verify_width.clamp(1, width) - 1
                } else {
                    k.min(s.verify_width)
                };
                self.drafter.input(s.seq_id, input);
                let mut draft = if ask == 0 {
                    Vec::new()
                } else {
                    self.drafter.propose(s.seq_id, input, ask)
                };
                if draft.len() as u32 > ask {
                    self.counters.draft_overruns += 1;
                    draft.truncate(ask as usize);
                }
                drafted_rows += draft.len() as u64;
                scheduled.push(draft.len() as u32);
                // A band slot runs `1 + ask` rows whatever the drafter returned: a short proposal
                // is padded to ITS OWN width (the rows are scheduled, and the band was chosen for
                // them), never to the band's max width.
                let pad_to = if band.is_some() { ask } else { k };
                draft.resize(pad_to as usize, crate::draft::RecordedDrafter::WRONG);
                let slot_draws = params.map(|params| {
                    std::sync::Arc::new(SlotDraws {
                        sampler: std::sync::Arc::clone(&sampler),
                        params,
                        seed: s.sampling.params.seed,
                        choice: s.sampling.choice_index,
                        position: s.sampling.position,
                        penalties: s.sampling.penalties.clone(),
                        draft: draft.clone(),
                        dense: *dense,
                        decided: std::sync::Mutex::new(Vec::new()),
                        walked: std::sync::Mutex::new(None),
                    })
                });
                let decider = slot_draws.as_ref().map(|d| {
                    crate::forward::Decider(
                        std::sync::Arc::clone(d) as std::sync::Arc<dyn crate::forward::RowDecider>
                    )
                });
                draws.push(slot_draws);
                DecodeSlot {
                    seq: s.seq_id,
                    input,
                    context_len: s.context_len,
                    draft,
                    // What the previous step committed for this slot, which is what picks the
                    // recurrent snapshot this one continues from. The scheduler hands the tokens it
                    // is feeding back; at SPEC=0 that is one.
                    committed_before: (s.input_tokens.len() as u32).max(1),
                    sampling: request.clone(),
                    decider,
                }
            })
            .collect();
        let want = StepRows {
            phase: Phase::Decode,
            rows: band.unwrap_or(shape * width),
            width,
            samples: true,
            tree: false,
            ragged: band.is_some(),
        };
        let clock = std::time::Instant::now();
        let ran = match band {
            Some(rows) => self.forward.decode_band(&slots, shape, width, rows),
            None => self.forward.decode(&slots, shape, width),
        };
        let mut out = match ran {
            Ok(o) => o,
            Err(e) => {
                self.counters.forward_errors += 1;
                tracing::error!(error = %e, slots = slots.len(), width, "decode step failed");
                return Self::failed();
            }
        };
        let forward_done = std::time::Instant::now();
        self.counters.decode_forward_us += (forward_done - clock).as_micros() as u64;
        self.counters.decode_steps += 1;
        self.counters.decode_sequences += slots.len() as u64;
        let fed_rows: u32 = slots.iter().map(DecodeSlot::rows).sum();
        self.counters.padded_rows += u64::from(want.rows.saturating_sub(fed_rows));
        if band.is_some() {
            self.counters.band_steps += 1;
            self.counters.band_live_rows += u64::from(fed_rows);
            self.counters.band_rows += u64::from(want.rows);
        }
        self.counters.draft_tokens += drafted_rows;
        self.counters.accepted_tokens += u64::from(out.accepted());
        self.counters.committed_tokens += out.committed() as u64;
        self.check_step(want, slots.len(), &out);
        if out.commits.len() != slots.len() {
            // `check_step` has already counted this; the stream must not be fed guesses.
            return Self::failed();
        }
        // The commits come back in slot order and each one echoes its sequence, so a pass that
        // returned them permuted is a protocol error here rather than tokens on the wrong stream.
        if let Some((i, bad)) = out
            .commits
            .iter()
            .zip(batch.slots)
            .enumerate()
            .find(|(_, (c, s))| c.seq != s.seq_id)
            .map(|(i, (c, _))| (i, c.seq))
        {
            self.counters.shape_violations += 1;
            tracing::error!(
                at = i,
                got = %bad,
                "a decode step returned its commits out of slot order"
            );
            return Self::failed();
        }
        for (s, c) in batch.slots.iter().zip(&out.commits) {
            self.drafter.commit(s.seq_id, &c.tokens);
            // WHAT A STALE BLOCK EARNED, tallied where the acceptance is (ws-f35). Only ever
            // non-empty under the control; under the law those blocks were never offered.
            if self.stale_blocks.remove(&s.seq_id) {
                self.counters.stale_anchor_drafts += 1;
                self.counters.stale_anchor_accepted += u64::from(c.accepted);
            }
        }
        // THE ANCHOR THE DEVICE PROPOSED FROM, read before the sampler can move it (ws-f35).
        //
        // `decode_step` makes the block out of the ARGMAX of the row it is about to commit —
        // `saber_engine::device`'s `let anchor = *verdict.committed.last()` — and the host sampler
        // below replaces that same token for a row with a temperature. So this is the token the
        // drafter was told the sequence would have, and the loop after the sampler compares it
        // with the token the sequence actually got.
        let proposed_from: Vec<Option<TokenId>> = out
            .commits
            .iter()
            .map(|c| c.tokens.last().copied())
            .collect();
        // A pass that returned a different number of proposals than it ran slots is counted here;
        // the offer below is indexed by slot, so what it cannot pair it simply does not offer,
        // rather than pairing a proposal with the wrong sequence.
        if !out.drafts.is_empty() && out.drafts.len() != slots.len() {
            self.counters.shape_violations += 1;
            tracing::error!(
                proposals = out.drafts.len(),
                slots = slots.len(),
                "a decode step proposed for a different number of slots than it ran"
            );
        }
        // What the step STASHED rather than proposed (ws-f37): collected below, after every
        // slot's token has been decided.
        let pending = std::mem::take(&mut out.proposal_pending);
        // THE TOKEN. Every slot's commits are the device's argmax up to here; a row that asked
        // for a temperature has its LAST token replaced by the host sampler's draw, taken from
        // the seam the head wrote for exactly that row (`i * width + accepted`).
        let mut seqs = Vec::with_capacity(batch.slots.len());
        // Each slot's first row in the step: the `cu` prefix (`i * width` in a rectangle).
        let starts: Vec<u32> = slots
            .iter()
            .scan(0u32, |at, s| {
                let here = *at;
                *at += s.rows();
                Some(here)
            })
            .collect();
        for (i, ((s, mut c), drafted)) in batch
            .slots
            .iter()
            .zip(out.commits)
            .zip(scheduled)
            .enumerate()
        {
            let row_index = starts[i] + c.accepted;
            let force_dense = requests.get(i).is_some_and(|(_, dense)| *dense);
            let slot_draws = draws.get(i).cloned().flatten();
            let logprobs = self.sample_commit(
                &s.sampling,
                &mut c,
                row_index,
                force_dense,
                slot_draws.as_deref(),
            );
            // WHAT THE PASS ITSELF PROPOSED, offered to the drafter — AFTER the sampler, and only
            // when the token the drafter was anchored on is the token this slot committed.
            //
            // A pass with a drafter tier of its own runs the proposal inside the step, because a
            // real DFlash2 proposal reads the TARGET's hidden states for the rows that step just
            // committed and those live in the target's arena until the next step overwrites them.
            // So it comes back with the step, from the argmax, and the drafter the executor holds
            // is a mailbox for it (`crate::draft::EngineDrafter`). Every other drafter ignores the
            // offer, which is the default on the trait.
            //
            // **THE ANCHOR IS THE LAW** (ws-f35). Prod calls `speculator.propose` AFTER its
            // sampler, so prod's block is always the continuation of the token the sequence
            // committed. Saber's is the continuation of the ARGMAX, and for a row with a
            // temperature those are different tokens on a measurable fraction of steps — held-out
            // corpus, 160 prompts, temperature 0.7: the first drafted token survives 0.683 of the
            // time against prod's 0.800, while every deeper position sits at 0.96–0.99 of prod,
            // because a step whose anchor moved is drafting a different text entirely and is
            // filtered out at position 0. A block made from a token the sequence never had is not
            // a proposal for this sequence: it is dropped and counted.
            let stale = proposed_from
                .get(i)
                .copied()
                .flatten()
                .is_some_and(|argmax| c.tokens.last() != Some(&argmax));
            if let Some(d) = out.drafts.get(i) {
                if !stale || self.keep_stale_anchor_blocks {
                    self.counters.pass_proposals += 1;
                    self.drafter.offer(s.seq_id, d);
                    if stale {
                        self.stale_blocks.insert(s.seq_id);
                    }
                } else {
                    self.counters.pass_proposals_stale += 1;
                }
            }
            // THE OTHER FORK POINT: with no captured prefill bucket the prompt is fed by one
            // decode-shaped step, and that step is where the prompt's first token — and the row
            // a branch would draw from — is produced. `first_step` is the scheduler saying so.
            if s.first_step {
                self.keep_fork_row(s, &c, row_index);
            }
            seqs.push(SeqStep {
                seq_id: s.seq_id,
                tokens: c.tokens,
                // The SCHEDULED width for THIS slot, which is what the scheduler's acceptance EMA
                // is computed against — not the drafter's own block width and not the bucket's
                // (`saber_sched::SeqStep`).
                draft_tokens: drafted,
                accepted_tokens: c.accepted,
                logprobs,
                prompt_logprobs: None,
            });
        }
        // THE PROPOSAL FOR THE NEXT STEP, IN PROD'S ORDER (ws-f37): every slot's last token is
        // now the one it COMMITTED — the argmax on a greedy row, the sampler's draw otherwise —
        // and that is what the drafter is anchored on. One call for the whole batch, exactly as
        // the in-step proposal was one batched proposal.
        if !pending.is_empty() {
            let committed: Vec<(SeqId, TokenId)> = seqs
                .iter()
                .filter_map(|q: &SeqStep| q.tokens.last().map(|t| (q.seq_id, *t)))
                .collect();
            let made = self.collect_after_commit(&pending, &committed);
            for (q, p) in seqs.iter().zip(&made) {
                self.offer_after_commit(q.seq_id, q.tokens.last().copied(), p);
            }
        }
        // FN69: the lookup source's counters, level with the step that just committed.
        if let Some(l) = self.drafter.lookup_counts() {
            self.counters.lookup_lookups = l.lookups;
            self.counters.lookup_hits = l.hits;
            self.counters.lookup_used = l.used;
            self.counters.lookup_drafted = l.lookup_drafted;
            self.counters.lookup_accepted = l.lookup_accepted;
            self.counters.lookup_drafter_drafted = l.drafter_drafted;
            self.counters.lookup_drafter_accepted = l.drafter_accepted;
        }
        self.counters.decode_after_us += forward_done.elapsed().as_micros() as u64;
        StepResult { seqs }
    }
}

impl<R: ForwardRunner> StepExecutor for RealExecutor<R> {
    fn memory_plane(&self) -> Option<std::sync::Arc<std::sync::Mutex<String>>> {
        let text = &self.forward.facts().rank_memory_plane;
        (!text.is_empty()).then(|| std::sync::Arc::new(std::sync::Mutex::new(text.clone())))
    }

    fn census_mirror(&self) -> Option<std::sync::Arc<saber_serve::census::CensusMirror>> {
        Some(std::sync::Arc::clone(&self.census))
    }

    /// One relaxed load while nobody scrapes; once armed, a store per census field (about 70),
    /// with no allocation and no clone of the per-shape map (ws-v8).
    fn publish_census(&mut self) {
        if !self.census.armed() {
            return;
        }
        // `counters()` folds the sampler's seam refusals in; the stored copy is brought level with
        // it so the publish reads one struct without cloning it.
        self.counters.seam_refusals = self
            .sampler
            .counters
            .seam_refusals
            .load(std::sync::atomic::Ordering::Relaxed);
        // ws-f39's two, which live in the sampler too: without this /metrics read 0 while the
        // shutdown census read 19 / 9 (the first card run).
        self.counters.top_ties = self
            .sampler
            .counters
            .top_ties
            .load(std::sync::atomic::Ordering::Relaxed);
        self.counters.top_tie_draws = self
            .sampler
            .counters
            .top_tie_draws
            .load(std::sync::atomic::Ordering::Relaxed);
        let draft = self.forward.draft_counts().unwrap_or_default();
        let counters = &self.counters;
        self.census
            .publish(|put| crate::census::engine_values(counters, &draft, put));
    }

    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        if let Some(unknown) = batch
            .slots
            .iter()
            .find(|s| !self.live.contains_key(&s.seq_id))
        {
            self.counters.unknown_sequences += 1;
            tracing::error!(seq = %unknown.seq_id, "a step for a sequence the engine never admitted");
            return Self::failed();
        }
        match batch.phase {
            Phase::Prefill => match batch.slots {
                [slot] => {
                    self.unmixed_prefill = true;
                    let r = self.prefill_step(slot);
                    self.unmixed_prefill = false;
                    r
                }
                other => {
                    // The scheduler refuses to build one (`NoBucket::MixedPhase` and the prefill
                    // rule), and the index tables could not describe it: `cu` is a prefix over
                    // sequences and a prompt chunk is a bucket of batch 1.
                    self.counters.oversized_step_refusals += 1;
                    tracing::error!(
                        slots = other.len(),
                        "a prefill step with more than one sequence"
                    );
                    Self::failed()
                }
            },
            Phase::Decode => self.decode_step(batch),
        }
    }

    /// C9: under `--engine-mixed-prefill fused`, the chunk is STAGED on the runner and rides in
    /// the decode half's forward call (one fused forward where the law allows, the serial form in
    /// the same call where it does not); the chunk's own prefill call then returns its half. The
    /// executor's two halves run exactly as unmixed (the decode half commits first). Everything
    /// the fused form does not take — a row band, a prompt-logprobs chunk, a batch that is not one
    /// chunk — runs C4's serial default.
    fn execute_mixed(
        &mut self,
        decode: &StepBatch<'_>,
        chunk: &StepBatch<'_>,
    ) -> (StepResult, StepResult) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        let serial = |me: &mut Self| {
            let c = me.execute(chunk);
            let d = me.execute(decode);
            (d, c)
        };
        if !self.fused_mixed || decode.bucket.is_band() || decode.phase != Phase::Decode {
            return serial(self);
        }
        let [slot] = chunk.slots else {
            return serial(self);
        };
        let Some(ch) = slot.chunk else {
            return serial(self);
        };
        if slot.sampling.params.prompt_logprobs.is_some()
            || !self.live.contains_key(&slot.seq_id)
            || decode
                .slots
                .iter()
                .any(|s| !self.live.contains_key(&s.seq_id))
        {
            return serial(self);
        }
        let (request, _) = if ch.last {
            self.row_request(&slot.sampling)
        } else {
            (RowRequest::greedy(), false)
        };
        let staged = crate::forward::MixedChunk {
            seq: slot.seq_id,
            tokens: slot.input_tokens.clone(),
            done_before: ch.tokens_done_before,
            last: ch.last,
            sampling: request,
        };
        // C11: the one-chunk form asks the fused law first, as the packed form does. Since the
        // scheduler never cuts a chunk to fit the rung, a long prompt's whole 2048-row chunk
        // reaches here beside the decode rows with no in-class rung: it runs the serial form
        // through the executor's own calls (C4's bit-exact path), counted as a fallback.
        let width = decode.bucket.spec_width.max(1);
        if !self
            .forward
            .can_fuse(std::slice::from_ref(&staged), decode.bucket.batch, width)
        {
            self.counters.mixed_fused_fallbacks += 1;
            return serial(self);
        }
        if !self.forward.stage_mixed(staged) {
            return serial(self);
        }
        let clock = std::time::Instant::now();
        let d = self.decode_step(decode);
        let fused_us = clock.elapsed().as_micros() as u64;
        let c = self.prefill_step(slot);
        match self.forward.take_mixed_info() {
            Some((true, rung)) => {
                self.counters.mixed_fused_steps += 1;
                self.counters.mixed_fused_rung_rows += u64::from(rung);
                self.counters.mixed_fused_forward_us += fused_us;
            }
            Some((false, _)) => self.counters.mixed_fused_fallbacks += 1,
            None => {}
        }
        (d, c)
    }

    /// C10: under `--engine-mixed-prefill packed`, the chunks of a packed step are STAGED on the
    /// runner and ride in the decode half's forward call as ONE fused forward — after the runner
    /// said the fused law takes them ([`crate::forward::ForwardRunner::can_fuse`]). Anything the
    /// fused form does not take runs the serial default here (each chunk through its own call,
    /// then the decode half), so every chunk's dense fallback reads its own head.
    fn execute_packed(
        &mut self,
        decode: &StepBatch<'_>,
        chunks: &[StepBatch<'_>],
    ) -> (StepResult, Vec<StepResult>) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        let serial = |me: &mut Self| {
            let cs: Vec<StepResult> = chunks.iter().map(|c| me.execute(c)).collect();
            let d = me.execute(decode);
            (d, cs)
        };
        if !self.fused_mixed || decode.bucket.is_band() || decode.phase != Phase::Decode {
            return serial(self);
        }
        if decode
            .slots
            .iter()
            .any(|s| !self.live.contains_key(&s.seq_id))
        {
            return serial(self);
        }
        let mut staged = Vec::with_capacity(chunks.len());
        let mut slots = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let [slot] = chunk.slots else {
                return serial(self);
            };
            let Some(ch) = slot.chunk else {
                return serial(self);
            };
            if slot.sampling.params.prompt_logprobs.is_some()
                || !self.live.contains_key(&slot.seq_id)
            {
                return serial(self);
            }
            let (request, _) = if ch.last {
                self.row_request(&slot.sampling)
            } else {
                (RowRequest::greedy(), false)
            };
            staged.push(crate::forward::MixedChunk {
                seq: slot.seq_id,
                tokens: slot.input_tokens.clone(),
                done_before: ch.tokens_done_before,
                last: ch.last,
                sampling: request,
            });
            slots.push(slot);
        }
        let width = decode.bucket.spec_width.max(1);
        if !self.forward.can_fuse(&staged, decode.bucket.batch, width) {
            self.counters.mixed_fused_fallbacks += 1;
            return serial(self);
        }
        let n = staged.len() as u64;
        for m in staged {
            if !self.forward.stage_mixed(m) {
                return serial(self);
            }
        }
        let clock = std::time::Instant::now();
        let d = self.decode_step(decode);
        let fused_us = clock.elapsed().as_micros() as u64;
        let cs: Vec<StepResult> = slots.iter().map(|s| self.prefill_step(s)).collect();
        match self.forward.take_mixed_info() {
            Some((true, rung)) => {
                self.counters.mixed_fused_steps += 1;
                self.counters.mixed_fused_rung_rows += u64::from(rung);
                self.counters.mixed_fused_forward_us += fused_us;
                if n > 1 {
                    self.counters.mixed_packed_steps += 1;
                    self.counters.mixed_packed_chunks += n;
                }
            }
            Some((false, _)) => self.counters.mixed_fused_fallbacks += 1,
            None => {}
        }
        (d, cs)
    }

    fn serves_images(&self) -> bool {
        self.forward.vision_max_patches() > 0
    }

    /// D3/D4: the tower over ALL of a request's images in ONE run, at the head of the queue. The
    /// rows stay in the tower arena's `out` slot, image after image, and the request's grids and
    /// the run's epoch are kept until its leader is admitted -- where the M-RoPE table is built
    /// from the prompt and handed to the engine with them ([`ForwardRunner::set_images`]).
    fn encode_images(
        &mut self,
        request: saber_serve::engine::RequestId,
        images: &saber_serve::engine::ImageBatch,
    ) -> Result<(), String> {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        let row = 1536usize;
        let patches = images.patches() as usize;
        if images.pixel_values.len() != patches * row {
            self.counters.vision_failures += 1;
            return Err(format!(
                "vision: the image batch carries {} pixel rows for grids of {patches} patches",
                images.pixel_values.len() / row
            ));
        }
        match self
            .forward
            .run_vision(&images.grids, images.pixel_values.clone())
        {
            Ok(rows) => {
                self.counters.vision_images += images.grids.len() as u64;
                self.counters.vision_patches += patches as u64;
                self.counters.vision_tokens += u64::from(rows.tokens);
                self.counters.vision_runs += 1;
                tracing::info!(
                    request = %request,
                    grids = ?images.grids,
                    patches,
                    tokens = rows.tokens,
                    launches = rows.launches,
                    epoch = rows.epoch,
                    addr = format_args!("{:#x}", rows.addr),
                    "vision tower ran; rows left in the arena for the prefill's scatter"
                );
                self.vision_pending
                    .insert(request, (images.grids.clone(), rows.epoch));
                Ok(())
            }
            Err(e) => {
                self.counters.vision_failures += 1;
                Err(format!("vision: {e}"))
            }
        }
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        let prompt_len = u32::try_from(seq.prompt_tokens.len()).unwrap_or(u32::MAX);
        match self.forward.admit(seq.seq_id, prompt_len, &seq.kv_blocks) {
            Ok(()) => {
                self.live.insert(seq.seq_id, prompt_len);
                // FN69: the prompt-lookup source indexes the whole prompt (prefix-cached part too).
                self.drafter.prompt(seq.seq_id, &seq.prompt_tokens);
                // D4: the sequence's image facts, built from the PROMPT and the request's grids --
                // on a first admission and on every restore alike (a restored group comes back
                // under new sequence ids and the same request; its facts are host state, which a
                // park does not image, and the same prompt and grids give the same positions and
                // delta). The epoch is the tower run `encode_images` recorded: a restore is past
                // its prompt, so it scatters nothing and the epoch is never checked for it.
                let facts = self
                    .vision_pending
                    .get(&seq.request_id)
                    .map(|(grids, epoch)| {
                        saber_exec::mrope::for_prompt(
                            &seq.prompt_tokens,
                            saber_vision_input::IMAGE_PAD_ID,
                            grids,
                        )
                        .map(|rope| crate::forward::SeqImages {
                            rope: std::sync::Arc::new(rope),
                            epoch: *epoch,
                        })
                        .map_err(|e| e.to_string())
                    });
                match facts {
                    None => {}
                    Some(Ok(f)) => {
                        let delta = f.rope.delta;
                        match self.forward.set_images(seq.seq_id, Some(f.clone())) {
                            Ok(()) => {
                                self.counters.vision_seqs += 1;
                                tracing::info!(
                                    seq = %seq.seq_id,
                                    images = f.rope.spans.len(),
                                    tokens = f.rope.image_tokens(),
                                    mrope_delta = delta,
                                    epoch = f.epoch,
                                    "image sequence admitted: M-RoPE positions and scatter set"
                                );
                                self.image_seqs.insert(seq.seq_id, seq.request_id);
                            }
                            Err(e) => {
                                self.counters.forward_errors += 1;
                                tracing::error!(seq = %seq.seq_id, error = %e, "the engine refused an image sequence's facts");
                            }
                        }
                    }
                    Some(Err(e)) => {
                        // Positions nobody can place are refused rather than served as text: the
                        // step after this finds no image facts and the sequence is aborted by the
                        // scheduler's unknown-sequence path.
                        self.counters.forward_errors += 1;
                        self.live.remove(&seq.seq_id);
                        self.forward.release(seq.seq_id);
                        tracing::error!(seq = %seq.seq_id, error = %e, "an image prompt whose layout the grids do not account for");
                    }
                }
            }
            Err(e) => {
                // Not admitted here means every step for it will be a counted unknown-sequence
                // fault and the request will be aborted — loud, and not a wrong answer.
                self.counters.forward_errors += 1;
                tracing::error!(seq = %seq.seq_id, error = %e, "the engine could not admit a sequence");
            }
        }
    }

    fn release(&mut self, seq: SeqId) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        self.live.remove(&seq);
        self.fork_rows.remove(&seq);
        self.forget_image_seq(seq, false);
        // The control's outstanding-stale set goes with the sequence. Sequence ids are not reused,
        // so leaving it would only leak — which is the smell S9 named on `drafter.release` and is
        // not one to add a second instance of.
        self.stale_blocks.remove(&seq);
        // FN69: a drafter with per-sequence state (the lookup index) drops it here.
        self.drafter.release(seq);
        self.forward.release(seq);
    }

    fn release_parked(&mut self, seq: SeqId) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        // D4: the request's grids stay -- the group comes back -- and the sequence moves to the
        // parked set, from which a discard forgets it.
        if let Some(r) = self.image_seqs.remove(&seq) {
            self.parked_image_seqs.insert(seq, r);
        }
        self.live.remove(&seq);
        self.fork_rows.remove(&seq);
        self.stale_blocks.remove(&seq);
        // a restore comes back under a new sequence id, admitted (and re-indexed) afresh
        self.drafter.release(seq);
        self.forward.release_parked(seq);
    }

    fn discard_parked(&mut self, seq: SeqId) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        self.forget_image_seq(seq, true);
        let receipt = self.forward.discard_parked(seq);
        self.counters.parked_discards += 1;
        if receipt.held() {
            self.counters.parked_discards_held += 1;
        }
        tracing::debug!(seq = %seq, ?receipt, "a parked image was dropped unrestored");
    }

    /// Whether the pass underneath can branch a session: it needs the device-to-device copy
    /// (`saber-hip`'s `memcpy_d2d_blocking`) and a state slot to put it in. Asked once, at
    /// construction, so `n > 1` is a 400 on a runner that cannot rather than a refusal per branch.
    fn can_fork(&self) -> bool {
        self.forward.can_fork()
    }

    /// The pass's seam width when it can score prompt rows (V7), and `None` — a 400 at submit —
    /// when it cannot.
    fn max_prompt_logprobs(&self) -> Option<u32> {
        self.forward.prompt_rows_k()
    }

    /// Branch `branch` off `leader`, at the step the leader's prompt sampled.
    ///
    /// Two halves, in this order and for this reason. The DEVICE half first — the GDN state slot
    /// set and the leader's one partial KV page, both copied device to device by the pass — so
    /// that a branch whose state could not be copied never gets a token at all. Then the DRAW:
    /// the leader's own row, the leader's own parameters, and this branch's `choice_index` in the
    /// noise key, which is what makes two branches of one seeded request two different streams
    /// that are each reproducible.
    ///
    /// **A greedy group is one answer `n` times**, and that is not a special case here: the row
    /// the kernel skipped is `None`, `sample_commit` takes the argmax path, and the leader's own
    /// token comes back unchanged.
    fn prefix_cache_supported(&self) -> bool {
        self.forward.prefix_cache_supported()
    }

    fn prefix_save(&mut self, seq: SeqId, entry: u32, tokens: u32) -> Result<u64, String> {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        self.forward
            .prefix_save(seq, entry, tokens)
            .map_err(|e| e.to_string())
    }

    fn prefix_load(&mut self, seq: &AdmittedSeq, entry: u32, tokens: u32) -> Result<(), String> {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        self.forward
            .prefix_load(seq.seq_id, entry, tokens)
            .map_err(|e| e.to_string())
    }

    fn prefix_drop(&mut self, entry: u32) {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        self.forward.prefix_drop(entry);
    }

    fn prefix_verify(
        &mut self,
        seq: SeqId,
        entry: u32,
        tokens: u32,
        entry_blocks: &[u32],
    ) -> Result<bool, String> {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        self.forward
            .prefix_verify(seq, entry, tokens, entry_blocks)
            .map_err(|e| e.to_string())
    }

    fn fork(&mut self, leader: SeqId, branch: &AdmittedSeq) -> Option<Forked> {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        let Some(seed) = self.fork_rows.get(&leader).cloned() else {
            self.counters.fork_refusals += 1;
            tracing::error!(
                leader = %leader, branch = %branch.seq_id,
                "no sampled row is being held for this leader; the branch cannot be forked from a                  row that is gone"
            );
            return None;
        };
        if let Err(e) = self.forward.fork(
            leader,
            branch.seq_id,
            &branch.kv_blocks,
            branch.shared_prefix_blocks,
            branch.cow_source,
        ) {
            self.counters.fork_refusals += 1;
            tracing::error!(
                leader = %leader, branch = %branch.seq_id, error = %e,
                "the pass refused to fork this branch's session state"
            );
            return None;
        }
        let prompt_len = u32::try_from(branch.prompt_tokens.len()).unwrap_or(u32::MAX);
        self.live.insert(branch.seq_id, prompt_len);
        self.drafter.prompt(branch.seq_id, &branch.prompt_tokens);
        self.counters.forks += 1;
        // D4: the pass copied the leader's image facts to the branch; the bookkeeping follows.
        if let Some(r) = self.image_seqs.get(&leader).copied() {
            self.image_seqs.insert(branch.seq_id, r);
        }

        // The branch's own key over the leader's own row.
        let mut sampling = seed.sampling.clone();
        sampling.choice_index = branch.choice_index;
        let mut commit = SlotCommit {
            seq: branch.seq_id,
            accepted: 0,
            tokens: vec![seed.token],
            rows: vec![seed.row],
        };
        // What the draw REPORTS is kept (ws-f39): the branch's first choice chunk carries its
        // logprobs exactly as the leader's first chunk carries the leader's, off the same row.
        let reported = self.sample_commit(&sampling, &mut commit, seed.row_index, false, None);
        let token = commit.tokens.last().copied()?;
        let logprobs = reported.and_then(|mut v| v.pop());

        // THE BRANCH'S FIRST BLOCK (ws-f38), from ITS token — F37's law at the fork. The leader's
        // prompt proposal was anchored on the leader's first token, which a draw with a
        // temperature makes a different token from this one; so the pass proposes again, over the
        // same context rows, into this branch's own drafter pages, and the block is offered only
        // if the anchor its drafter was fed is the token this branch committed. Nothing reaches
        // the pass between the draw above and this call (a dense fallback is part of the draw).
        match self.forward.propose_fork(leader, branch.seq_id, token) {
            Ok(None) => {}
            Ok(Some(p)) if p.seq == branch.seq_id => {
                self.counters.fork_blocks += 1;
                self.offer_after_commit(branch.seq_id, Some(token), &p);
            }
            Ok(Some(p)) => {
                self.counters.fork_block_failures += 1;
                tracing::error!(
                    branch = %branch.seq_id, proposal = %p.seq,
                    "a branch's first block came back for another sequence; it is not offered"
                );
            }
            Err(e) => {
                self.counters.fork_block_failures += 1;
                tracing::error!(
                    leader = %leader, branch = %branch.seq_id, error = %e,
                    "a forked branch's first block could not be made; its first step verifies \
                     nothing"
                );
            }
        }
        Some(Forked { token, logprobs })
    }

    fn parks_async(&self) -> bool {
        self.forward.parks_async()
    }

    /// Whether a **restore** cannot start right now.
    ///
    /// With one staging slot a park and a restore share the buffer, so either blocks the other. With
    /// two they have a slot each and only another restore is in the way. Asked before the parking
    /// tier is touched, so the answer has to be right or a retry loses its tickets.
    fn transfer_in_flight(&self) -> bool {
        if self.forward.facts().park_staging_slots >= 2 {
            self.restores_in_flight > 0
        } else {
            self.in_flight > 0
        }
    }

    /// Issue the park and return. The sequence stays in `self.live`: its pages and its slot are
    /// still being read, and the scheduler frees them only when `park_poll` says the copy is done.
    fn park_begin(&mut self, seq: SeqId) -> TransferStart {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        if !self.live.contains_key(&seq) {
            self.counters.unknown_sequences += 1;
            tracing::error!(seq = %seq, "a park for a sequence the engine never admitted");
            return TransferStart::CannotPark;
        }
        if !self.forward.parks_async() {
            return TransferStart::Blocking;
        }
        match self.forward.park_begin(seq) {
            Ok(()) => {
                self.in_flight += 1;
                TransferStart::Started
            }
            Err(ForwardError::CannotPark) => TransferStart::CannotPark,
            Err(ForwardError::TransferBusy) => TransferStart::Busy,
            Err(e) => {
                self.counters.park_failures += 1;
                tracing::error!(seq = %seq, error = %e, "the engine could not start a park");
                TransferStart::Failed(e.to_string())
            }
        }
    }

    fn park_poll(&mut self, seq: SeqId) -> ParkPoll {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        match self.forward.park_poll(seq) {
            Ok(None) => ParkPoll::Pending,
            Ok(Some(blob)) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.counters.parks += 1;
                tracing::debug!(seq = %seq, bytes = blob.len(), "session parked");
                ParkPoll::Done(SeqState(blob))
            }
            Err(e) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.counters.park_failures += 1;
                tracing::error!(seq = %seq, error = %e, "a park in flight failed");
                ParkPoll::Failed(e.to_string())
            }
        }
    }

    fn restore_begin(&mut self, seq: &AdmittedSeq, state: SeqState) -> RestoreStart {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        if self.fail_restores {
            self.counters.restores_refused_by_injection += 1;
            return RestoreStart::Refused(
                "refused by --engine-park-fail-restores (the bounded-restore control)".to_owned(),
                Some(state),
            );
        }
        if !self.live.contains_key(&seq.seq_id) {
            self.counters.unknown_sequences += 1;
            tracing::error!(seq = %seq.seq_id, "a restore for a sequence the engine never admitted");
            return RestoreStart::Refused("not admitted".to_owned(), Some(state));
        }
        if !self.forward.parks_async() {
            return RestoreStart::Blocking(state);
        }
        // Moved in. Every answer below that is not `Started` gives the bytes back, which is the
        // contract that lets the engine own them instead of copying them.
        match self.forward.restore_begin(seq.seq_id, state.0) {
            Ok(()) => {
                self.in_flight += 1;
                self.restores_in_flight += 1;
                RestoreStart::Started
            }
            Err(ForwardError::Rejected(what, blob)) => {
                // `TransferBusy` is carried as a rejection with the bytes, because a busy answer
                // must be retryable and a retry needs the image.
                if what == "busy" {
                    RestoreStart::Busy(SeqState(*blob))
                } else {
                    self.counters.restore_failures += 1;
                    tracing::error!(seq = %seq.seq_id, error = %what, "a session image was refused");
                    RestoreStart::Refused(what, Some(SeqState(*blob)))
                }
            }
            Err(e) => {
                self.counters.restore_failures += 1;
                tracing::error!(seq = %seq.seq_id, error = %e, "a session image was refused and not handed back");
                RestoreStart::Refused(e.to_string(), None)
            }
        }
    }

    fn restore_poll(&mut self, seq: SeqId) -> RestorePoll {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        match self.forward.restore_poll(seq) {
            Ok(false) => RestorePoll::Pending,
            Ok(true) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.restores_in_flight = self.restores_in_flight.saturating_sub(1);
                self.counters.restores += 1;
                RestorePoll::Done
            }
            Err(e) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                self.restores_in_flight = self.restores_in_flight.saturating_sub(1);
                self.counters.restore_failures += 1;
                tracing::error!(seq = %seq, error = %e, "a restore in flight failed");
                let what = e.to_string();
                RestorePoll::Failed(what, e.into_blob().map(SeqState))
            }
        }
    }

    /// Copy a sequence's KV pages and GDN state slot off the device, as one image.
    ///
    /// `None` is what the scheduler reads as "this executor cannot park", and it means exactly
    /// that: either the runner has no way to move state (the default), or this attempt failed and
    /// was counted. Either way the sequence is left untouched and the scheduler refuses the park
    /// rather than evicting state it could not bring back.
    ///
    /// The bytes leave here as an opaque `SeqState` because that is the scheduler's contract; what
    /// is inside is [`crate::ParkImage`], which carries the region layout and a digest per 4 MiB
    /// chunk so the restore can verify before it issues anything.
    fn park_state(&mut self, seq: SeqId) -> Option<SeqState> {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        if !self.live.contains_key(&seq) {
            self.counters.unknown_sequences += 1;
            tracing::error!(seq = %seq, "a park for a sequence the engine never admitted");
            return None;
        }
        if !self.forward.can_park() {
            return None;
        }
        // The encoded seam, not the typed one: the blob is built straight out of the runner's
        // staging buffer, so the payload is copied once rather than twice.
        match self.forward.park_encoded(seq) {
            Ok(blob) => {
                self.counters.parks += 1;
                tracing::debug!(seq = %seq, bytes = blob.len(), "session parked");
                Some(SeqState(blob))
            }
            Err(e) => {
                self.counters.park_failures += 1;
                tracing::error!(seq = %seq, error = %e, "the engine could not park a session");
                None
            }
        }
    }

    /// Put an image back into a sequence the scheduler has just admitted.
    ///
    /// The pages and the slot are whatever that admission was given — different ones, in general,
    /// which is the point of parking. `false` leaves the session as `admit` made it (zeroed slot,
    /// empty pages) and the scheduler hands both allocations straight back without binding them.
    fn restore_state(&mut self, seq: &AdmittedSeq, state: &SeqState) -> bool {
        // P27-3: an owed prompt store is collected before anything else reaches the runner.
        self.settle_deferred();
        if !self.live.contains_key(&seq.seq_id) {
            self.counters.unknown_sequences += 1;
            tracing::error!(seq = %seq.seq_id, "a restore for a sequence the engine never admitted");
            return false;
        }
        match self.forward.restore_encoded(seq.seq_id, state.0.clone()) {
            Ok(()) => {
                self.counters.restores += 1;
                true
            }
            Err(e) => {
                self.counters.restore_failures += 1;
                tracing::error!(seq = %seq.seq_id, error = %e, "a session image was refused");
                false
            }
        }
    }
}
