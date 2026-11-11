//! The scheduler: the [`Engine`] the serving plane talks to.
//!
//! One step is one pass of [`Scheduler::step`]:
//!
//! 1. **Admit** from the head of the queue while a KV allocation *and* a GDN state slot set are
//!    both available and the running set is below the widest captured batch. First-come, first
//!    served, and a refusal stops the scan — head-of-line blocking is deliberate, because skipping
//!    a large request to admit small ones starves it forever and prod does not reorder either.
//! 2. **Choose the width** from every running sequence's acceptance EMA and the running count —
//!    prod's cap applies only at three or more requests in flight ([`crate::width`]).
//! 3. **Choose the bucket**: the smallest captured shape that fits the running set
//!    ([`crate::buckets`]). The batch is rounded up and the extra slots are padding.
//! 4. **Execute** one [`StepBatch`] through the [`StepExecutor`].
//! 5. **Commit**: append tokens, apply stop conditions, update the EMAs and the spec counters, emit
//!    exactly one [`StepEvent`] per running request, and free both allocations of anything that
//!    finished — immediately, in the same step, so the next admission sees the space.
//!
//! The step loop is synchronous and takes `&self`: the serving plane runs it on a plain
//! `std::thread` (`ARCHITECTURE.md` § Rust-specific hazards — the step loop wants fixed cadence,
//! and a work-stealing scheduler adds variance to exactly that path). Only the per-request event
//! channel is async, because that is what the SSE handler already consumes.

use std::collections::{BTreeMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures_core::Stream;
use saber_mem::ledger::{Ledger, LedgerError, LedgerSnapshot};
use saber_plan::{Bucket, Phase};
use saber_serve::engine::{
    Engine, EngineError, EngineRequest, EngineStats, FinishReason, ModelInfo,
    PromptPositionLogprobs, RequestId, SampledLogprobs, StepEvent, StepStream, Submission, TokenId,
};
use saber_serve::sampling::SamplingParams;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::alloc::{Arenas, GdnAllocation, KvAllocation, Refusal};
use crate::buckets::{
    select, select_band, select_prefill, BandRequest, BucketRequest, NoBucket, PrefillRequest,
};
use crate::config::SchedulerConfig;
use crate::executor::{
    AdmittedSeq, BatchSlot, Forked, ParkPoll, PenaltyHistory, PrefillChunk, RestorePoll,
    RestoreStart, SeqId, SeqState, SlotSampling, StepBatch, StepExecutor, StepResult, StepTiming,
    TransferStart,
};
use crate::park::{
    choose_victim, ParkCandidate, ParkCounters, ParkError, ParkTicket, Parkable, SessionImage,
};
use crate::width::{AcceptanceEma, RaggedSlot};
use saber_model::draft::SpecDecodeCounters;

/// Why a scheduler could not be built.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// The arenas do not fit. This is the startup error the byte ledger exists for: an over-commit
    /// here is a refusal to start, never an OOM at token 4000.
    #[error("cannot reserve the scheduler's arenas: {0}")]
    Ledger(#[from] LedgerError),
    /// FN52: `--enable-prefix-caching on` over an executor that cannot snapshot a prefix — a
    /// refusal at boot, never a cache that silently never hits.
    #[error("prefix caching is on and this executor cannot snapshot or restore a prefix")]
    PrefixCacheUnsupported,
}

/// What one step did. Returned by [`Scheduler::step`] so a harness can trace the step loop without
/// scraping logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepOutcome {
    /// Nothing running and nothing admissible.
    Idle,
    /// A bucket was replayed.
    Ran(StepSummary),
    /// The running set has no captured shape. Fail loud: nothing is executed.
    NoBucket(NoBucket),
}

impl StepOutcome {
    /// The summary, when the step ran.
    pub fn summary(&self) -> Option<StepSummary> {
        match self {
            StepOutcome::Ran(s) => Some(*s),
            _ => None,
        }
    }
}

/// One step's shape and yield.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepSummary {
    /// Engine-wide step index.
    pub step_index: u64,
    /// The bucket replayed. Always captured.
    pub bucket: Bucket,
    /// Live sequences in the step.
    pub sequences: u32,
    /// Padding slots the captured shape ran anyway.
    pub padded_slots: u32,
    /// The verify width asked for this step.
    pub verify_width: u32,
    /// Tokens committed across all sequences.
    pub committed_tokens: u32,
    /// Sequences admitted at the top of this step.
    pub admitted: u32,
    /// Sequences that finished in this step.
    pub finished: u32,
}

/// Which-path counters. Every branch that can silently change behaviour has one
/// (`CLAUDE.md` rule 5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedCounters {
    /// Steps that executed a bucket.
    pub steps: u64,
    /// Steps with nothing to do.
    pub idle_steps: u64,
    /// Steps refused because no captured bucket fitted.
    pub no_bucket_steps: u64,
    /// Results the executor returned that did not match the batch: wrong count, wrong order, a
    /// wider draft than was scheduled, or more accepted than drafted.
    pub executor_protocol_errors: u64,
    /// Results whose committed-token count broke `tokens.len() == accepted_tokens + 1`
    /// (`docs/abi/DFLASH2-SEMANTICS.md` § 1.4). Counted separately because it is the one that
    /// would desynchronise the spec-decode metrics from the token stream rather than corrupt a
    /// batch.
    pub executor_token_count_errors: u64,
    /// Requests aborted while running.
    pub aborts_running: u64,
    /// Requests aborted while queued.
    pub aborts_waiting: u64,
    /// Requests aborted while parked.
    pub aborts_parked: u64,
    /// Sessions parked.
    pub parks: u64,
    /// Park attempts refused (no tier, tier full, executor cannot park).
    pub park_refusals: u64,
    /// Sessions restored into a running sequence.
    pub restores: u64,
    /// Restore attempts that failed after the allocations were taken (the allocations are handed
    /// straight back and the session stays parked).
    pub restore_failures: u64,
    /// Parks issued off the step loop and not yet finished. The number that says the asynchronous
    /// path is the one being taken: a scheduler on the blocking path never raises it.
    pub parks_started: u64,
    /// Restores issued off the step loop and not yet finished.
    pub restores_started: u64,
    /// Steps that ran while a park or a restore was in flight — the step loop carrying on, which is
    /// the whole point of the asynchronous path.
    pub steps_during_transfer: u64,
    /// Parks the executor abandoned after they were in flight. The sequence goes back to running
    /// with its pages and slot intact.
    pub park_transfer_failures: u64,
    /// Transfers deferred because the executor already had one in flight. Retryable, and counted so
    /// that "the staging buffer is the bottleneck" is a number rather than a guess.
    pub park_transfer_busy: u64,
    /// Sessions parked by the eviction policy rather than by an explicit hook. A subset of
    /// [`SchedCounters::parks`].
    pub policy_parks: u64,
    /// Admissions refused for KV or GDN state where the policy wanted to evict but no running
    /// sequence was eligible — the `min_running` floor, the residency guard, or an empty running
    /// set. The number that says whether the policy is doing anything at all.
    pub policy_park_misses: u64,
    /// Admissions refused for KV or GDN while a policy park was already IN FLIGHT (the
    /// asynchronous path): the space is on its way, so the policy waits for it rather than
    /// evicting a second session or re-choosing the one whose copy is running (ws-k17 — before
    /// this, every step of a park's copy re-chose the same victim and had it refused as
    /// `NotRunning`, and the finished park was never marked as the policy's, so nothing restored it).
    pub policy_park_waits: u64,
    /// Parks of a group that had already been parked at least once before — a session evicted a
    /// second (third, ...) time. Zero whenever `ParkPolicy::max_parks_per_session` is 1, and the
    /// serve plane's own evidence that the unbounded default is exercised (ws-k17).
    pub reparks: u64,
    /// Policy-parked sessions queued for restore because a sequence finished and freed space.
    pub policy_restores_queued: u64,
    /// Parked sessions whose restore failed `ParkPolicy::max_restore_attempts` times and which
    /// were ENDED with a named error rather than retried again (ws-k18). Before the bound, such a
    /// session was retried on every freed slot for ever.
    pub restores_given_up: u64,
    /// Parked images dropped without a restore — an abort while parked, or a session given up on —
    /// for which the executor was told to drop what it holds FOR the image
    /// ([`StepExecutor::discard_parked`]; at TP=2 the follower's half). ws-k18.
    pub parked_discards: u64,
    /// Requests finished on an EOS or stop token.
    pub finished_stop: u64,
    /// Requests finished on `max_tokens`.
    pub finished_length: u64,
    /// Results the executor returned for a sequence that left the running set while the step was
    /// in flight — aborted or parked mid-step. Dropped, not committed.
    pub results_for_gone_sequences: u64,
    /// Aborts whose allocations could not be handed back immediately because a step was in
    /// flight. They go back in that step's commit, a few milliseconds later.
    pub retires_deferred: u64,
    /// Steps that fed a prompt chunk rather than decoding. A prompt of `n` chunks costs `n` of
    /// these before its first token, which is what `TTFT` is made of.
    pub prefill_steps: u64,
    /// Real prompt rows successfully processed, excluding padding and prefix cache hits.
    pub prefill_computed_tokens: u64,
    /// Output tokens committed after stop and length trimming, accumulated after every step.
    pub output_committed_tokens: u64,
    /// Prompts fed as one decode-shaped step because the plan captured no prefill bucket that
    /// fits them. The pre-P2 path, kept so a manifest without prefill buckets still serves —
    /// and counted, because it is a fallback and a silent one would look like success.
    pub prefill_bucket_misses: u64,
    /// Admissions that stopped at an IMAGE request because the tower's one output slot still
    /// holds another request's rows that its prefill has not consumed (D4's slot law).
    pub vision_waits: u64,
    /// Branches admitted to the arenas whose admission to the EXECUTOR is deferred to the fork
    /// (`Seq::pending_fork`). Every one of them is either forked or aborted, so this is `forks +
    /// fork_failures + fork_branches_orphaned` once nothing is running.
    pub forks_pending: u64,
    /// Branches the executor forked from their leader's sampled prefill row.
    pub forks: u64,
    /// Forks the executor refused. The branch is aborted — loudly — rather than started from a
    /// state nothing copied.
    pub fork_failures: u64,
    /// Branches whose leader left before it sampled (it was aborted, or its prefill failed), so
    /// there was never a row to fork from. They are aborted with it, because a group is one unit.
    pub fork_branches_orphaned: u64,
    /// Forked branches whose request asked for `logprobs` and whose FIRST token came back from
    /// [`StepExecutor::fork`] without a report, so that one entry of `logprobs.content` is
    /// missing. Since ws-f39 the real executor reports the branch's draw off the leader's row
    /// ([`crate::Forked::logprobs`]) and this is zero on a serving run; a mock with no
    /// distribution still counts here rather than papering over it.
    pub fork_first_token_logprobs_absent: u64,
    /// Slot-steps whose own acceptance EMA asked for a narrower width than the bucket the step
    /// ran in, and which therefore verified the bucket's width instead
    /// ([`crate::VerifyWidthPolicy::slot_in_bucket`]).
    ///
    /// This is the rectangular-bucket tax, as a number. prod's ragged batch would have dropped
    /// those rows from `M`; a captured shape launches them whatever the slot asks, so the slot
    /// verifies them rather than throwing away a draft it has already paid for. A large count
    /// with `VerifyWidthPolicy::can_narrow() == false` means the plan captured no width for the
    /// cap to reach.
    pub slot_caps_absorbed: u64,
    /// Decode steps whose bucket was NARROWER than the widest captured width — the dynamic verify
    /// width actually doing something. Zero over a whole run at concurrency says the ladder has
    /// nothing to narrow to, not that the EMAs are warm.
    pub width_narrowed_steps: u64,
    /// Decode steps whose width was RAISED to cover a slot's previous commit (ws-f38).
    ///
    /// A step resumes each sequence from `state_slot + naccept - 1`, INSIDE its own `sidx` block,
    /// so the block must be at least as wide as what the previous step committed. The batch table
    /// (`1:8, 4:5`) narrows on the batch alone: a sequence that committed 8 at batch 3 and then
    /// met a fourth sequence was handed a width-5 block and the engine refused the step by name
    /// (`saber_exec::forward::check_with`), aborting the whole batch. Groups made it routine (a
    /// branch that joins the batch late); a single request could hit it at any 3 -> 4 transition.
    ///
    /// **Zero under RAGGED widths (C2)**: a band's `sidx` block is the MAX width with every entry
    /// of a live run filled, so `naccept - 1` always lands inside it whatever the slot verifies
    /// now, and no floor is applied.
    pub width_floor_raised: u64,
    /// Decode steps that ran a RAGGED schedule (C2): every slot at its own width, in a row band
    /// or — when every slot asked the same width and that rectangle is captured — in it.
    pub ragged_steps: u64,
    /// Of [`SchedCounters::ragged_steps`], the ones whose slots all asked one width and ran in the
    /// captured RECTANGLE of that width (a band would have run the same rows at batch `max_seqs`,
    /// with the fused GDN route and the batch-1 attention split given up).
    pub ragged_rect_steps: u64,
    /// Rows the running slots FED over the ragged steps, `Σ_step Σ_slot w_i` — the census's
    /// "rows per step" is this over [`SchedCounters::ragged_steps`], against the rectangle's 40 at
    /// conc-8.
    pub ragged_live_rows: u64,
    /// Rows the ragged steps RAN (their band's or rectangle's `M`), padding included.
    pub ragged_step_rows: u64,
    /// Slot-steps whose own acceptance EMA narrowed them below the step's ceiling (prod's cap
    /// binding) under the ragged schedule — the rows [`SchedCounters::slot_caps_absorbed`] counts
    /// under the rectangle, here actually dropped.
    pub ragged_slots_narrowed: u64,
    /// C4: decode steps that carried a prefill chunk (`MixedPrefill`). Zero when it is off.
    pub mixed_steps: u64,
    /// C4: prompt rows fed inside mixed steps.
    pub mixed_chunk_rows: u64,
    /// C4: mixed-step candidates whose decode half left no room in `max_batched_tokens`, so the
    /// owed chunk waited a step.
    pub mixed_budget_spent: u64,
    /// C4: prefill steps that ran ALONE (the unmixed schedule, or nothing was decoding).
    pub prefill_alone_steps: u64,
    /// C4, THE STALL: decoding sequences that were ready and waited out a prefill step that ran
    /// alone, summed over those steps (slot-steps lost to admission).
    pub stalled_decoder_steps: u64,
    /// C4, THE STALL in time: `waiting decoders x the step's wall`, summed (slot-microseconds).
    pub stalled_decoder_us: u64,
    /// C4: the executor call's wall over prefill steps that ran alone, microseconds.
    pub prefill_alone_us: u64,
    /// C4: the executor call's wall over decode steps that carried no chunk, microseconds.
    pub decode_step_us: u64,
    /// C4: the executor call's wall over mixed steps, microseconds.
    pub mixed_step_us: u64,
    /// LR-B: decode steps run BETWEEN two prefill chunks under `--fn-prefill-yield` (0 when off).
    pub yield_decode_steps: u64,
}

impl SchedCounters {
    /// C4's step timeline: the executor call's wall, filed by the step's kind, and the stall a
    /// prefill step that ran alone cost the decoders it left waiting. A measurement only: nothing
    /// branches on it (`CLAUDE.md` rule 6).
    fn record_step_wall(&mut self, ready: &ReadyStep, wall_us: u64) {
        match (ready.phase, ready.chunk.is_some()) {
            (Phase::Prefill, _) => {
                self.prefill_alone_steps += 1;
                self.prefill_alone_us += wall_us;
                self.stalled_decoder_steps += u64::from(ready.waiting_decoders);
                self.stalled_decoder_us += u64::from(ready.waiting_decoders) * wall_us;
            }
            (Phase::Decode, true) => self.mixed_step_us += wall_us,
            (Phase::Decode, false) => self.decode_step_us += wall_us,
        }
    }
}

// The scheduler's half of the shutdown census, live on `/metrics` (ws-v8). The `census` column is
// the line `saber serve --engine real` prints at shutdown and the text around the number in it
// (`saber_engine::census::shutdown_census` renders the lines); `tests/census_parity.rs` in
// saber-engine reads one against the other, counter for counter.
saber_serve::census_counters!(
    SchedCounters,
    CENSUS,
    census_values,
    [
        (
            steps,
            "saber:sched_steps_total",
            "Steps that executed a bucket.",
            "sched: steps {},"
        ),
        (
            idle_steps,
            "saber:sched_idle_steps_total",
            "Step-loop polls with nothing to run.",
            "sched: idle {},"
        ),
        (
            no_bucket_steps,
            "saber:sched_no_bucket_steps_total",
            "Steps refused because no captured bucket fitted.",
            "sched: no bucket {},"
        ),
        (
            executor_protocol_errors,
            "saber:sched_executor_protocol_errors_total",
            "Executor results that did not match the batch.",
            "sched: protocol errors {},"
        ),
        (
            executor_token_count_errors,
            "saber:sched_executor_token_count_errors_total",
            "Executor results that broke tokens == accepted + 1.",
            "sched: token count errors {},"
        ),
        (
            aborts_running,
            "saber:sched_aborts_running_total",
            "Requests aborted while running.",
            "sched: aborts running {} /"
        ),
        (
            aborts_waiting,
            "saber:sched_aborts_waiting_total",
            "Requests aborted while queued.",
            "sched: waiting {} /"
        ),
        (
            aborts_parked,
            "saber:sched_aborts_parked_total",
            "Requests aborted while parked.",
            "sched: parked {},"
        ),
        (
            finished_stop,
            "saber:sched_finished_stop_total",
            "Requests finished on an EOS or stop token.",
            "sched: finished stop {} /"
        ),
        (
            finished_length,
            "saber:sched_finished_length_total",
            "Requests finished on max_tokens.",
            "sched: length {},"
        ),
        (
            results_for_gone_sequences,
            "saber:sched_results_for_gone_sequences_total",
            "Results for a sequence that left mid-step (dropped).",
            "sched: results for gone seqs {},"
        ),
        (
            retires_deferred,
            "saber:sched_retires_deferred_total",
            "Abort retirements deferred to the step in flight.",
            "sched: retires deferred {},"
        ),
        (
            prefill_steps,
            "saber:sched_prefill_steps_total",
            "Steps that fed a prompt chunk.",
            "sched: prefill steps {},"
        ),
        (
            prefill_computed_tokens,
            "saber:sched_prefill_computed_tokens_total",
            "Prompt tokens actually computed, excluding cached prefix rows and padding.",
            "sched: computed prompt tokens {},"
        ),
        (
            output_committed_tokens,
            "saber:sched_output_committed_tokens_total",
            "Output tokens committed by the scheduler, including the first sampled token.",
            "sched: committed output tokens {},"
        ),
        (
            prefill_bucket_misses,
            "saber:sched_prefill_bucket_misses_total",
            "Prompts fed as one decode-shaped step (no prefill bucket fitted).",
            "sched: prefill bucket misses {},"
        ),
        (
            vision_waits,
            "saber:sched_vision_waits_total",
            "Admissions held at an image request while the tower slot was owned (D4).",
            "sched: vision waits {},"
        ),
        (
            forks_pending,
            "saber:sched_forks_pending_total",
            "Branches admitted to the arenas awaiting their fork.",
            "sched: forks pending {} /"
        ),
        (
            forks,
            "saber:sched_forks_total",
            "Branches forked from their leader's sampled row.",
            "sched: done {} /"
        ),
        (
            fork_failures,
            "saber:sched_fork_failures_total",
            "Forks the executor refused (the branch is aborted).",
            "sched: failed {} /"
        ),
        (
            fork_branches_orphaned,
            "saber:sched_fork_branches_orphaned_total",
            "Branches whose leader left before it sampled.",
            "sched: orphaned {},"
        ),
        (
            fork_first_token_logprobs_absent,
            "saber:sched_fork_first_token_logprobs_absent_total",
            "Forked branches whose first token carries no logprobs.",
            "sched: fork first-token logprobs absent {},"
        ),
        (
            slot_caps_absorbed,
            "saber:sched_slot_caps_absorbed_total",
            "Slot-steps that verified the bucket's wider width.",
            "sched: slot caps absorbed {},"
        ),
        (
            width_narrowed_steps,
            "saber:sched_width_narrowed_steps_total",
            "Decode steps whose bucket was narrower than the widest width.",
            "sched: width narrowed steps {},"
        ),
        (
            width_floor_raised,
            "saber:sched_width_floor_raised_total",
            "Decode steps whose width was raised to cover a slot's previous commit.",
            "sched: width raised to a previous commit {},"
        ),
        (
            ragged_steps,
            "saber:sched_ragged_steps_total",
            "Decode steps under ragged per-slot widths (C2).",
            "sched: ragged steps {} ("
        ),
        (
            ragged_rect_steps,
            "saber:sched_ragged_rect_steps_total",
            "Ragged steps whose uniform widths ran in the captured rectangle.",
            "sched: (in a rectangle {}),"
        ),
        (
            ragged_live_rows,
            "saber:sched_ragged_live_rows_total",
            "Rows the slots fed over the ragged steps.",
            "sched: live rows {} /"
        ),
        (
            ragged_step_rows,
            "saber:sched_ragged_step_rows_total",
            "Rows the ragged steps ran, padding included.",
            "sched: / run {},"
        ),
        (
            ragged_slots_narrowed,
            "saber:sched_ragged_slots_narrowed_total",
            "Slot-steps prod's acceptance cap narrowed below the ragged ceiling.",
            "sched: slots narrowed by the cap {}"
        ),
        (
            mixed_steps,
            "saber:sched_mixed_steps_total",
            "Decode steps that carried a prefill chunk (C4).",
            "timeline: mixed steps {} ("
        ),
        (
            mixed_chunk_rows,
            "saber:sched_mixed_chunk_rows_total",
            "Prompt rows fed inside mixed steps (C4).",
            "timeline: ({} chunk rows,"
        ),
        (
            mixed_budget_spent,
            "saber:sched_mixed_budget_spent_total",
            "Owed chunks that waited because the decode half spent the token budget (C4).",
            "timeline: budget spent {},"
        ),
        (
            mixed_step_us,
            "saber:sched_mixed_step_us_total",
            "Executor wall over mixed steps, microseconds (C4).",
            "timeline: , wall us {}), prefill"
        ),
        (
            prefill_alone_steps,
            "saber:sched_prefill_alone_steps_total",
            "Prefill steps that ran alone (C4).",
            "timeline: prefill alone {} ("
        ),
        (
            prefill_alone_us,
            "saber:sched_prefill_alone_us_total",
            "Executor wall over prefill steps that ran alone, microseconds (C4).",
            "timeline: (wall us {}), decode"
        ),
        (
            decode_step_us,
            "saber:sched_decode_step_us_total",
            "Executor wall over unmixed decode steps, microseconds (C4).",
            "timeline: decode wall us {},"
        ),
        (
            stalled_decoder_steps,
            "saber:sched_stalled_decoder_steps_total",
            "Decoder slot-steps that waited out a prefill step run alone (C4 stall).",
            "timeline: stalled decoder slot-steps {} /"
        ),
        (
            stalled_decoder_us,
            "saber:sched_stalled_decoder_us_total",
            "Decoder slot-microseconds lost to prefill steps run alone (C4 stall).",
            "timeline: / slot-us {}"
        ),
        (
            yield_decode_steps,
            "saber:sched_yield_decode_steps_total",
            "Decode steps run between two prefill chunks (LR-B --fn-prefill-yield).",
            "timeline: yield decode steps {}"
        ),
        (
            parks,
            "saber:sched_parks_total",
            "Sessions parked.",
            "parking: parks {} ("
        ),
        (
            policy_parks,
            "saber:sched_policy_parks_total",
            "Sessions parked by the eviction policy.",
            "parking: (policy {},"
        ),
        (
            park_refusals,
            "saber:sched_park_refusals_total",
            "Park attempts refused.",
            "parking: refused {})"
        ),
        (
            restores,
            "saber:sched_restores_total",
            "Sessions restored.",
            "parking: restores {} ("
        ),
        (
            restore_failures,
            "saber:sched_restore_failures_total",
            "Restores that failed after allocation.",
            "parking: (failed {},"
        ),
        (
            policy_restores_queued,
            "saber:sched_policy_restores_queued_total",
            "Policy-parked sessions queued for restore.",
            "parking: queued {})"
        ),
        (
            parks_started,
            "saber:sched_parks_started_total",
            "Parks issued off the step loop.",
            "parking: transfers started {}/"
        ),
        (
            restores_started,
            "saber:sched_restores_started_total",
            "Restores issued off the step loop.",
            "parking: /{} busy"
        ),
        (
            park_transfer_busy,
            "saber:sched_park_transfer_busy_total",
            "Transfers deferred because one was in flight.",
            "parking: busy {} failed"
        ),
        (
            park_transfer_failures,
            "saber:sched_park_transfer_failures_total",
            "Parks abandoned in flight.",
            "parking: failed {}, steps during"
        ),
        (
            steps_during_transfer,
            "saber:sched_steps_during_transfer_total",
            "Steps that ran while a transfer was in flight.",
            "parking: steps during a transfer {},"
        ),
        (
            reparks,
            "saber:sched_reparks_total",
            "Sessions parked again.",
            "parking: sessions parked again {},"
        ),
        (
            restores_given_up,
            "saber:sched_restores_given_up_total",
            "Parked sessions ended after max restore attempts.",
            "parking: restores given up {},"
        ),
        (
            parked_discards,
            "saber:sched_parked_discards_total",
            "Parked images dropped unrestored.",
            "parking: dropped unrestored {} ("
        ),
        (
            policy_park_misses,
            "saber:sched_policy_park_misses_total",
            "KV/GDN refusals with no eligible victim.",
            "admission: policy misses {} ("
        ),
        (
            policy_park_waits,
            "saber:sched_policy_park_waits_total",
            "Refusals that waited on a park in flight.",
            "admission: waits on a park in flight {}"
        ),
    ]
);

/// How long the FIRST `/metrics` scrape waits for the step in flight to publish the executor's
/// census (ws-v8). Every later scrape reads the mirror without waiting. A step is ~22-40 ms and a
/// long prefill chunk well under a second; a scrape that times out renders without the engine's
/// counters, and the next one has them.
const FIRST_SCRAPE_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

type EventSink = UnboundedSender<StepEvent>;

/// A stream of one request's steps.
#[derive(Debug)]
struct EventStream {
    rx: UnboundedReceiver<StepEvent>,
}

impl Stream for EventStream {
    type Item = StepEvent;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<StepEvent>> {
        self.get_mut().rx.poll_recv(cx)
    }
}

/// What [`Core::prepare`] decided, carried across the unlocked device call.
#[derive(Debug)]
enum Prepared {
    Idle,
    NoBucket(NoBucket),
    Ready(ReadyStep),
}

/// One step's shape, built under the core lock and executed without it.
#[derive(Debug)]
struct ReadyStep {
    phase: Phase,
    bucket: Bucket,
    slots: Vec<BatchSlot>,
    running: u32,
    width: u32,
    admitted: u32,
    step_index: u64,
    /// C4: the prefill chunk this DECODE step carries (`MixedPrefill`), a complete one-slot
    /// prefill step of its own. `None` on every unmixed step.
    chunk: Option<Box<ReadyStep>>,
    /// C10: the FURTHER chunks a packed step carries (`MixedPrefill::packs`), each a complete
    /// one-slot prefill step of its own, in queue order after `chunk`. Empty unless packing.
    more: Vec<ReadyStep>,
    /// Decoding sequences that were ready to step and are NOT rows of this step: on a prefill
    /// step that ran alone, the slots that waited it out (C4's stall).
    waiting_decoders: u32,
}

impl ReadyStep {
    /// The executor's view of this step (its own half only).
    fn batch(&self) -> StepBatch<'_> {
        StepBatch {
            phase: self.phase,
            bucket: self.bucket,
            step_index: self.step_index,
            slots: &self.slots,
            padded_slots: self.bucket.batch - self.running,
        }
    }
}

/// A running sequence.
#[derive(Debug)]
struct Seq {
    request_id: RequestId,
    /// Which of the request's `n` choices this branch is. A request is a **sequence group**: one
    /// prompt, `n` branches that share its KV pages and decode independently.
    choice_index: u32,
    seq_id: SeqId,
    params: SamplingParams,
    prompt_len: usize,
    /// LR-G: the tokens the prefill feeds -- `prompt_len`, or for a sequence re-admitted after a
    /// lazy-growth preemption (recompute) the prompt AND everything it had committed.
    prefill_len: usize,
    /// LR-G: this sequence's KV was reserved lazily (`AdmitPolicy::lazy_headroom`): it grows
    /// block by block before the step that needs the next block, and it may be preempted.
    lazy: bool,
    /// LR-G: times this request's sequence was preempted for recompute (carried through the
    /// queue).
    preempts: u32,
    /// Prompt plus everything committed.
    tokens: Vec<TokenId>,
    generated: usize,
    max_tokens: usize,
    kv: KvAllocation,
    gdn: GdnAllocation,
    /// Per-request event index, monotone from 0.
    step_index: u32,
    /// Engine step at which this sequence was admitted (or restored). The eviction policy's
    /// residency guard reads it: a sequence that has just prefilled is the worst thing to evict.
    admitted_at_step: u64,
    /// Engine step at which it last committed a token. See `ParkPolicy::idle_steps` for why this
    /// is 0 or 1 today and what it is there for.
    last_active_step: u64,
    /// Times this session has already been parked, carried across every restore. The policy's
    /// anti-starvation bound reads it.
    parks: u32,
    /// Whether the park this sequence was restored from was the POLICY's, carried so that a
    /// restore which fails after it was issued puts the group back as the policy's — and
    /// `requeue_auto_parked` still owes it a restore (ws-k18; it went back as an explicit park and
    /// nothing ever asked for it again).
    auto_parked: bool,
    /// Restores of this session that have FAILED so far, carried across the attempt.
    restore_failures: u32,
    /// Tokens the next step feeds: the prompt, then the previous step's committed tokens.
    pending_input: Vec<TokenId>,
    first_step: bool,
    /// Prompt tokens already written into this sequence's KV by earlier prefill chunks.
    /// `prefilled == prompt_len` means the prompt is in and the sequence is decoding.
    prefilled: usize,
    ema: AcceptanceEma,
    events: EventSink,
    /// This sequence is a BRANCH of a group whose leader has not sampled yet, and this is the
    /// admission the executor has not been given.
    ///
    /// `n > 1` forks **after** the leader's last prefill chunk (`docs/LEDGER.md`, V3's design):
    /// the branch's first token is a second draw from the leader's own prefill row, and that row
    /// exists only at that step. So the branch holds its pages and its state slot set from
    /// admission — the group is all-or-nothing in the arenas — but the executor is not told about
    /// it until [`StepExecutor::fork`] can be called, and until then the sequence is not
    /// schedulable and owes no `release`.
    pending_fork: Option<Box<AdmittedSeq>>,
    /// A device transfer of this sequence's session state is in flight, so it is **not stepped**
    /// and its allocations are **not freed** until the executor says the copy is done.
    ///
    /// This is the whole of "parking does not stall the step loop": the park is issued, the
    /// sequence stops being a row of the step's shape, and the step loop carries on with everybody
    /// else. K9 measured a 403 MiB park at 101 ms against a 30 ms step — three steps the other
    /// sequences would otherwise spend inside a `memcpy`.
    transfer: Option<Transfer>,
    /// `prompt_logprobs` gathered so far, chunk by chunk (V7). `Some` from admission exactly when
    /// the request asked and this is the group's LEADER on a fresh admission — it starts as
    /// `[None]`, the reference server's entry for position 0 — and it is taken, whole, into the event that carries
    /// the first token. A chunk that should have scored rows and did not ends the request with an
    /// error rather than a list with a hole in it.
    prompt_lp: Option<Vec<PromptPositionLogprobs>>,
    /// FN52: prompt positions the prefill must CUT at (a chunk never crosses one): the prefix
    /// snapshot point and the verify arm's compare point. Empty with the cache off.
    cuts: Vec<u32>,
    /// FN52: snapshot the state here (a whole-block boundary) and retain the prefix.
    snap_at: Option<u32>,
    /// FN52, the verify arm: `(entry, tokens, entry's blocks)` to compare against at `tokens`.
    verify_at: Option<(u32, u32, Vec<u32>)>,
}

impl Seq {
    /// FN52: how many prompt tokens the next chunk may feed — the rest of the prompt, or up to
    /// the next cut.
    fn prefill_room(&self) -> usize {
        let end = self
            .cuts
            .iter()
            .map(|c| *c as usize)
            .filter(|c| *c > self.prefilled)
            .min()
            .unwrap_or(self.prefill_len)
            .min(self.prefill_len);
        end.saturating_sub(self.prefilled)
    }

    /// How this sequence's next token is to be chosen, as the executor is handed it.
    ///
    /// Everything here is a fact the SCHEDULER owns: the request's parameters, which of its `n`
    /// branches this sequence is, how many tokens it has committed (the noise key's position, so a
    /// seeded request draws the same stream wherever and however it is stepped — ws-f40), and —
    /// only when the penalties actually bind
    /// — the tokens they read. The executor resolves the rest against the vocabulary, which is the
    /// engine's fact and not this crate's.
    ///
    /// The history is built **only** under penalties because it is the one part of this whose size
    /// is the sequence's length: at prod's defaults (no penalty at all) a sampled row costs the
    /// same as a greedy one here.
    fn sampling(&self) -> SlotSampling {
        let penalties = (self.params.presence_penalty != 0.0
            || self.params.frequency_penalty != 0.0
            || self.params.repetition_penalty != 1.0)
            .then(|| {
                Arc::new(PenaltyHistory {
                    prompt: self.tokens[..self.prompt_len.min(self.tokens.len())].to_vec(),
                    output: self.tokens[self.prompt_len.min(self.tokens.len())..].to_vec(),
                })
            });
        SlotSampling {
            params: self.params.clone(),
            choice_index: self.choice_index,
            step_index: u64::from(self.step_index),
            // Everything committed so far — a park carries `tokens` whole and a branch starts from
            // its leader's prompt, so this is the decision's own index, not the schedule's.
            position: self.tokens.len().saturating_sub(self.prompt_len) as u64,
            penalties,
        }
    }
}

/// Which way a sequence's state is moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transfer {
    /// Off the device. Its pages and slot are still being read.
    Parking,
    /// Back onto the device. Its host-side bookkeeping is already the restored one.
    Restoring,
    /// A branch of a group whose park has STARTED on a sibling and could not start on this one
    /// yet: the executor has one staging slot and a sibling holds it (ws-f38). Started by
    /// `poll_transfers` as soon as the slot frees; the group completes when every branch has.
    ParkQueued,
    /// The same for a restore: the image (with its state) waits in `restoring_images` and the
    /// admission in `restore_admitted` until the slot frees.
    RestoreQueued,
}

impl Transfer {
    /// A park of this sequence is under way, started or queued behind a sibling's.
    fn parking(self) -> bool {
        matches!(self, Transfer::Parking | Transfer::ParkQueued)
    }
}

impl Seq {
    /// Whether this sequence still owes prefill steps.
    fn prefilling(&self) -> bool {
        self.prefilled < self.prefill_len
    }

    /// Whether it can be a row of this step. A sequence with a transfer in flight cannot: its KV
    /// is half-written or being read, and stepping it would race the copy. Nor can a branch whose
    /// fork has not happened: the executor has never heard of it, its pages hold the leader's
    /// prompt and nothing else, and its state slot set has not been copied into.
    fn steppable(&self) -> bool {
        self.transfer.is_none() && self.pending_fork.is_none()
    }

    /// GDN state slots this sequence's steps actually touch right now.
    ///
    /// **One while it is prefilling**, `spec_width + 1` once it is decoding — the planner's own
    /// split (`Bucket::gdn_state_slots`): a prefill chunk has no candidates, so it carries one
    /// recurrent state and scans the chunk through it, where a decode step holds one state per
    /// draft token. The *reservation* is the decode-width set either way; this is what the
    /// Infinity-Cache accounting and the executor see.
    fn active_state_slots(&self, decode_slots: u32) -> u32 {
        if self.prefilling() {
            1
        } else {
            decode_slots
        }
    }
}

/// One branch of a parked session.
#[derive(Debug)]
struct ParkedBranch {
    choice_index: u32,
    tokens: Vec<TokenId>,
    generated: usize,
    step_index: u32,
    pending_input: Vec<TokenId>,
    ema: AcceptanceEma,
    ticket: ParkTicket,
    /// The sequence this branch's image was PARKED under: the name an executor holds anything it
    /// keeps for the image by (at TP=2, the follower's half), and so the name
    /// [`StepExecutor::discard_parked`] is called with when the image is dropped unrestored.
    parked_as: SeqId,
}

/// A parked **group**: every branch of a request, evicted as one unit.
///
/// Half a group on the tier is not a state anything can resume from: the branches share the
/// prompt's KV pages, so restoring some of them would mean re-materialising a prefix the others
/// still need. Park and restore are therefore all-or-nothing over the group.
#[derive(Debug)]
struct ParkedSeq {
    request_id: RequestId,
    params: SamplingParams,
    prompt_len: usize,
    max_tokens: usize,
    events: EventSink,
    /// Every branch of the group, each with its own image on the tier. A group parks and
    /// restores whole: the branches share the prompt's KV pages, so half a group on the tier is
    /// not a state anything can resume from.
    branches: Vec<ParkedBranch>,
    /// Parked by the policy rather than by an explicit hook, so the scheduler owes it a restore
    /// once the pressure that evicted it passes. A **group** is auto-parked or not; the policy
    /// never evicts one branch of one.
    auto: bool,
    /// Parks this session has been through, including this one. Counted per group, because the
    /// branches are evicted and restored together and a per-branch count would read `n` times
    /// the truth.
    parks: u32,
    /// Restore attempts of this group that have failed. At `ParkPolicy::max_restore_attempts`
    /// the request is ended with a named error (ws-k18).
    restore_failures: u32,
}

/// A queued request, or a parked one on its way back in.
#[derive(Debug)]
struct Waiting {
    request_id: RequestId,
    params: SamplingParams,
    prompt: Vec<TokenId>,
    prompt_len: usize,
    max_tokens: usize,
    /// Branches this request wants: `params.n`. Admission is all-or-nothing over them.
    n: u32,
    events: EventSink,
    resume: Option<Box<ParkedSeq>>,
    /// D3: the request's processed images, run through the executor's vision tower once, at the
    /// head of the queue, before the group's allocations are taken. `None` for a text request
    /// and after the tower ran.
    images: Option<saber_serve::engine::ImageBatch>,
    /// FN157 (the admit-slack census): when the request reached the queue (`None` for a parked
    /// session on its way back in). Read by the served-step ledger only.
    arrived: Option<std::time::Instant>,
    /// LR-G: a sequence preempted by a lazy growth on its way back in (recompute: `prompt` is its
    /// prompt + committed tokens, `prompt_len` their count, `max_tokens` what it may still make).
    recompute: Option<Box<Recompute>>,
}

/// LR-G: what a preempted sequence carries back through the queue so its stream continues where
/// it stopped (the event index, the token budget, the sampling position).
#[derive(Debug, Clone, Copy)]
struct Recompute {
    /// The ORIGINAL prompt's length (the sampling position and penalty split read it).
    prompt_len: usize,
    /// Tokens it had committed.
    generated: usize,
    /// Its original `max_tokens`.
    max_tokens: usize,
    /// Its next event index.
    step_index: u32,
    /// Its preemptions so far (this one included).
    preemptions: u32,
}

/// A sequence that has left the running set but whose slots the executor may still be reading.
///
/// The allocations are **not** handed back here. A step in flight was handed this sequence's KV
/// block table and GDN slot set; returning them to the arenas before that step is committed would
/// let the next admission be given slots the executor is still using. They go back in
/// [`Core::drain_retiring`], which runs with both locks held and therefore with no step running.
#[derive(Debug)]
struct Retired {
    seq_id: SeqId,
    kv: KvAllocation,
    gdn: GdnAllocation,
    /// Whether the executor was ever told about this sequence. A branch whose leader was aborted
    /// mid-prefill was never admitted and never forked, so releasing it would be releasing a
    /// sequence no executor has heard of — which a mock counts and a real engine logs.
    admitted: bool,
}

/// FN75: the step thread's clock between steps (held in the stepper's lock).
#[derive(Debug, Default)]
struct StepClock {
    last_end: Option<std::time::Instant>,
    work_at_end: bool,
    idle_polls: u64,
}

#[derive(Debug)]
struct Core {
    config: SchedulerConfig,
    ledger: Ledger,
    arenas: Arenas,
    park_tier: Option<Box<dyn Parkable>>,
    waiting: VecDeque<Waiting>,
    /// FN157: the arrival instants of the requests this step's admission took (the census only).
    admit_arrivals: Vec<std::time::Instant>,
    running: Vec<Seq>,
    /// States collected from finished park copies, until every branch of the group has one: a group
    /// parks whole or not at all, and the branches do not finish together.
    parking_states: BTreeMap<SeqId, SeqState>,
    /// Images handed to in-flight restores, kept so that a restore which fails **after** it was
    /// issued can put them back on the tier and leave the group parked rather than losing it.
    restoring_images: BTreeMap<SeqId, SessionImage>,
    /// Admissions of branches whose restore is QUEUED behind a sibling's (ws-f38): what
    /// `restore_begin` needs when the staging slot frees.
    restore_admitted: BTreeMap<SeqId, AdmittedSeq>,
    parked: Vec<ParkedSeq>,
    next_seq: u64,
    step_index: u64,
    /// Parks the policy has performed in the step being admitted for. Reset at the top of a step.
    parks_this_step: u32,
    /// LR-B: decode steps still owed to the decoders before the next prefill chunk
    /// (`PrefillChunking::yield_steps`; always 0 with it off).
    yield_owed: u32,
    /// Groups the POLICY issued an asynchronous park for, until the park completes (ws-k17). An
    /// asynchronous park lands in `parked` steps later, in `finish_park`, and that is where it has
    /// to be marked `auto` — or `requeue_auto_parked` never brings it back.
    auto_parks_in_flight: std::collections::BTreeSet<RequestId>,
    counters: SchedCounters,
    /// The `saber:spec_decode_*` counters, in prod's own definitions — `saber-model`'s
    /// [`SpecDecodeCounters`], which carries the per-position acceptance histogram as well as the
    /// three totals `EngineStats` exports.
    spec: SpecDecodeCounters,
    /// Sequences whose allocations are owed back to the arenas once no step is in flight.
    retiring: Vec<Retired>,
    /// Parked names whose images were dropped unrestored, owed a
    /// [`StepExecutor::discard_parked`] the next time the executor lock is held (ws-k18). Owed
    /// rather than called, because `abort` holds only the core lock.
    discards_owed: Vec<SeqId>,
    /// Whether the executor can branch a sequence group at all, asked once at construction.
    /// `n > 1` is refused at submit when it cannot, rather than quietly serving one choice.
    can_fork: bool,
    /// The widest `prompt_logprobs` the executor can answer, asked once at construction (V7).
    /// `None` refuses every request that asks, at submit, with a 400.
    prompt_logprobs_max: Option<u32>,
    /// **D4's slot law**: the request whose images the tower ran last. The executor has ONE output
    /// slot, which the next tower run overwrites, and a request's rows are read by its prefill
    /// chunks -- so no other image request is encoded while this one is still queued or still
    /// prefilling ([`Core::vision_slot_busy`]). Text requests are never held by it.
    vision_owner: Option<RequestId>,
    /// FN52: the retained prefixes.
    prefix: crate::prefix_cache::PrefixCache,
    /// LR-A: the queue head the last KV refusal was logged for (one line per refused request,
    /// not one per scheduler step: `saber:admission_refused_kv_total` ticks every step the head
    /// waits).
    kv_refused_head: Option<RequestId>,
    /// LR-A: requests (not steps) the KV arena refused at the head of the queue.
    kv_refused_requests: u64,
    /// LR-F: the admit-slack predictor and hint (`AdmitPolicy::slack_ms`).
    slack: SlackState,
    /// LR-F: the head a later request was last admitted past, and how many times
    /// (`AdmitPolicy::skip_kv_blocked`, capped at `AdmitPolicy::SKIP_LIMIT`).
    hol_skip: (Option<RequestId>, u32),
}

/// LR-A: the KV arena and the running set's reservations at one instant (`/metrics` gauges and
/// the once-per-request refusal line). Read-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvAdmissionSnapshot {
    /// Blocks in the KV arena.
    pub total_blocks: u32,
    /// Blocks free.
    pub free_blocks: u32,
    /// Tokens the running sequences reserved (`prompt + max_tokens` each, as resolved at submit).
    pub reserved_tokens: u64,
    /// Tokens the running sequences hold today (`prompt + generated` each).
    pub length_tokens: u64,
    /// The largest single running reservation.
    pub max_reserved_tokens: u64,
}

/// LR-A: the admission gauges on `/metrics` (outside the shutdown census: read-only telemetry).
pub const KV_ADMISSION_GAUGES: &[saber_serve::census::CensusDef] = &[
    saber_serve::census::CensusDef::gauge(
        "saber:kv_blocks_total",
        "Blocks in the scheduler's KV arena (the admission budget).",
        "kvadmit: blocks total {}",
    ),
    saber_serve::census::CensusDef::gauge(
        "saber:kv_blocks_free",
        "Blocks free in the scheduler's KV arena.",
        "kvadmit: free {}",
    ),
    saber_serve::census::CensusDef::gauge(
        "saber:kv_reserved_tokens",
        "Tokens the running sequences reserved (prompt + max_tokens).",
        "kvadmit: reserved tokens {}",
    ),
    saber_serve::census::CensusDef::gauge(
        "saber:kv_length_tokens",
        "Tokens the running sequences hold (prompt + generated).",
        "kvadmit: length tokens {}",
    ),
    saber_serve::census::CensusDef::gauge(
        "saber:kv_max_reserved_tokens",
        "The largest single running reservation in tokens.",
        "kvadmit: max reserved {}",
    ),
    saber_serve::census::CensusDef::counter(
        "saber:admission_refused_kv_requests_total",
        "Queue-head requests the KV arena refused (once per request, not per step).",
        "kvadmit: refused requests {}",
    ),
    saber_serve::census::CensusDef::counter(
        "saber:kv_lazy_grows_total",
        "LR-G (--fn-kv-lazy): KV growths a running sequence took before a step.",
        "kvadmit: lazy grows {}",
    ),
    saber_serve::census::CensusDef::counter(
        "saber:kv_lazy_preemptions_total",
        "LR-G (--fn-kv-lazy): sequences preempted for recompute by a refused growth.",
        "kvadmit: lazy preemptions {}",
    ),
    saber_serve::census::CensusDef::counter(
        "saber:kv_lazy_recompute_tokens_total",
        "LR-G (--fn-kv-lazy): written KV rows the preemptions threw away.",
        "kvadmit: lazy recompute tokens {}",
    ),
    saber_serve::census::CensusDef::counter(
        "saber:kv_lazy_margin_refusals_total",
        "LR-G (--fn-kv-lazy): lazy admissions refused by the growth margin alone (per step).",
        "kvadmit: lazy margin refusals {}",
    ),
];

/// FN52: what admission decided about a request's prefix.
#[derive(Debug, Default)]
struct PrefixPlan {
    /// The cache was consulted.
    looked: bool,
    /// Adopt this hit's blocks and restore its snapshot.
    adopt: Option<crate::prefix_cache::PrefixHit>,
    /// Snapshot at this position.
    snap_at: Option<u32>,
    /// The verify arm's compare point.
    verify_at: Option<(u32, u32, Vec<u32>)>,
}

/// The engine: continuous batching, paged KV plus a separate GDN state arena, dynamic verify
/// width, and session parking, over a [`StepExecutor`].
/// The engine's three locks, and why there are three.
///
/// A step used to hold one lock from admission through to commit — including the executor call,
/// which is the device work and ~22 ms in production. Every question the serving plane asks
/// (`/metrics` → `stats()`, `submit`, `abort`) queued behind it, and `std::sync::Mutex` is not
/// fair, so a step loop that releases and immediately re-acquires could starve a scraper for the
/// length of a whole response: **14.2 s** was measured (`docs/LEDGER.md`, 2026-09-19).
///
/// So:
///
/// * `core` — the scheduler's state. Held for admission, for building the batch, and for the
///   commit. Never across the device call.
/// * `executor` — the device. Held for `execute`, and briefly for `admit` / `release` /
///   `park_state`, which are the executor's other entry points.
/// * `stepping` — one stepper at a time. The step releases `core` in the middle, so two steppers
///   could otherwise interleave into a deadlock (one holding `core` and wanting `executor`, the
///   other the reverse).
///
/// The lock order is **`core` then `executor`**, and it is only ever taken that way inside a step.
/// Everything else — `abort`, `park` — takes `core` and then *tries* the executor without
/// blocking, so nothing outside the step loop can ever wait on the device.
#[derive(Debug)]
pub struct Scheduler<E: StepExecutor> {
    model: ModelInfo,
    core: Mutex<Core>,
    executor: Mutex<E>,
    stepping: Mutex<StepClock>,
    next_request_id: AtomicU64,
    /// The executor's live census (ws-v8), asked for once at construction. Read by `stats()`
    /// without the executor lock; published by the step loop, which already holds it.
    census: Option<Arc<saber_serve::census::CensusMirror>>,
    /// The executor's memory-plane exposition (FN14: Flash-Next's `saber_xtier_*` / `saber_ple_*`),
    /// asked for once at construction; `/metrics` appends it verbatim.
    memory_plane: Option<Arc<Mutex<String>>>,
    /// LR-F: the step clock (`None` = `Instant::now`, the served path). A replay installs a
    /// virtual one ([`Scheduler::with_clock`]) so walls, arrivals and holds run in model time.
    clock: Clock,
}

/// LR-F: an injectable `Instant` source (the replay's virtual time).
#[derive(Clone, Default)]
struct Clock(Option<Arc<dyn Fn() -> std::time::Instant + Send + Sync>>);

impl std::fmt::Debug for Clock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "Clock(virtual)"
        } else {
            "Clock(Instant::now)"
        })
    }
}

impl Clock {
    fn now(&self) -> std::time::Instant {
        match &self.0 {
            Some(f) => f(),
            None => std::time::Instant::now(),
        }
    }
}

/// LR-F: the admission arms' counters (`AdmitPolicy`; all 0 with the arms off).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmitArmCounters {
    /// Holds the step loop ran (`--fn-admit-slack-ms`).
    pub slack_holds: u64,
    /// Holds an arrival ended early.
    pub slack_caught: u64,
    /// Microseconds held, in all.
    pub slack_held_us: u64,
    /// Finishing steps whose hold the gain test declined.
    pub slack_declined: u64,
    /// Times a later request was admitted past a KV-refused head (`skip_kv_blocked`).
    pub hol_skips: u64,
    /// LR-G: lazy growths taken (a sequence's KV grew before a step).
    pub lazy_grows: u64,
    /// LR-G: blocks those growths took.
    pub lazy_grow_blocks: u64,
    /// LR-G: growths the arena refused (each preempts the youngest lazy sequence and retries).
    pub lazy_grow_refused: u64,
    /// LR-G: sequences preempted for recompute.
    pub lazy_preemptions: u64,
    /// LR-G: rows of KV those preemptions threw away (the recompute's prefill, before any prefix
    /// reload the engine finds).
    pub lazy_recompute_tokens: u64,
    /// LR-G: lazy admissions refused by the growth margin alone.
    pub lazy_margin_refusals: u64,
}

/// LR-F: the admit-slack predictor. Fed only while the arm is on; read at a finishing step.
#[derive(Debug, Default)]
struct SlackState {
    /// The end of the last step that finished a sequence, until an arrival answers it or the cap
    /// passes.
    open_finish: Option<std::time::Instant>,
    /// EMA of P(a request arrives within the cap after a finish).
    p_follow: f64,
    /// EMA of the finish -> arrival gap, us, over the arrivals within the cap.
    gap_us: f64,
    /// EMA of the decode step wall, us, by decode rows (0 = never seen).
    dec_wall_us: [f64; 9],
    /// The hold the last step earned, taken by the step loop.
    hint: Option<std::time::Duration>,
    counters: AdmitArmCounters,
}

impl SlackState {
    const ALPHA: f64 = 0.125;

    /// A finish whose cap passed with no arrival is a miss.
    fn close_stale(&mut self, now: std::time::Instant, cap: std::time::Duration) {
        if let Some(f) = self.open_finish {
            if now.saturating_duration_since(f) > cap {
                self.p_follow -= Self::ALPHA * self.p_follow;
                self.open_finish = None;
            }
        }
    }

    fn arrival(&mut self, now: std::time::Instant, cap: std::time::Duration) {
        self.close_stale(now, cap);
        if let Some(f) = self.open_finish.take() {
            let g = now.saturating_duration_since(f).as_secs_f64() * 1e6;
            self.p_follow += Self::ALPHA * (1.0 - self.p_follow);
            self.gap_us = if self.gap_us == 0.0 {
                g
            } else {
                self.gap_us + Self::ALPHA * (g - self.gap_us)
            };
        }
    }

    fn decode_wall(&mut self, rows: usize, us: u64) {
        if let Some(w) = self.dec_wall_us.get_mut(rows) {
            *w = if *w == 0.0 {
                us as f64
            } else {
                *w + Self::ALPHA * (us as f64 - *w)
            };
        }
    }
}

impl<E: StepExecutor> Scheduler<E> {
    /// Builds a scheduler and reserves both arenas on every rank through `ledger`.
    ///
    /// The ledger is moved in: the engine owns one, and the parking tiers bill the same one.
    pub fn new(
        config: SchedulerConfig,
        executor: E,
        model: ModelInfo,
        mut ledger: Ledger,
    ) -> Result<Self, SchedulerError> {
        let arenas = Arenas::reserve(&config, &mut ledger)?;
        let can_fork = executor.can_fork();
        let prompt_logprobs_max = executor.max_prompt_logprobs();
        let census = executor.census_mirror();
        let memory_plane = executor.memory_plane();
        if config.prefix_cache.enabled && !executor.prefix_cache_supported() {
            return Err(SchedulerError::PrefixCacheUnsupported);
        }
        let prefix =
            crate::prefix_cache::PrefixCache::new(config.prefix_cache, config.kv.block_tokens);
        Ok(Self {
            model,
            census,
            memory_plane,
            executor: Mutex::new(executor),
            stepping: Mutex::new(StepClock::default()),
            core: Mutex::new(Core {
                config,
                ledger,
                arenas,
                park_tier: None,
                waiting: VecDeque::new(),
                admit_arrivals: Vec::new(),
                running: Vec::new(),
                parking_states: BTreeMap::new(),
                restoring_images: BTreeMap::new(),
                restore_admitted: BTreeMap::new(),
                parked: Vec::new(),
                next_seq: 1,
                step_index: 0,
                parks_this_step: 0,
                yield_owed: 0,
                auto_parks_in_flight: std::collections::BTreeSet::new(),
                counters: SchedCounters::default(),
                spec: SpecDecodeCounters::default(),
                retiring: Vec::new(),
                discards_owed: Vec::new(),
                can_fork,
                prompt_logprobs_max,
                vision_owner: None,
                prefix,
                kv_refused_head: None,
                kv_refused_requests: 0,
                slack: SlackState::default(),
                hol_skip: (None, 0),
            }),
            next_request_id: AtomicU64::new(1),
            clock: Clock::default(),
        })
    }

    /// LR-F: run the scheduler on `clock` instead of `Instant::now` (the discrete-event replay,
    /// `examples/sched_replay`). Every wall the scheduler measures, every arrival stamp and the
    /// admit-slack predictor read it; nothing else changes.
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Fn() -> std::time::Instant + Send + Sync>) -> Self {
        self.clock = Clock(Some(clock));
        self
    }

    /// LR-F: the hold the last step earned (`AdmitPolicy::slack_ms`), taken once. `None` with the
    /// arm off, and on every step that did not finish a sequence with free seats and an empty
    /// queue. The caller holds at most this long and reports it with [`Self::hold_ended`]. Only
    /// the replay (`examples/sched_replay`) consumes it: the served step loop has no hold (LR-F
    /// NO-GO, the arm was not built).
    pub fn admit_slack_hold(&self) -> Option<std::time::Duration> {
        self.lock().slack.hint.take()
    }

    /// LR-F: a hold ran for `held`; `caught` = an arrival ended it.
    pub fn hold_ended(&self, held: std::time::Duration, caught: bool) {
        let mut core = self.lock();
        let c = &mut core.slack.counters;
        c.slack_holds += 1;
        c.slack_caught += u64::from(caught);
        c.slack_held_us += held.as_micros() as u64;
    }

    /// LR-F: the admission arms' counters.
    pub fn admit_arm_counters(&self) -> AdmitArmCounters {
        self.lock().slack.counters
    }

    /// Attaches a parking tier (RAM first, NVMe under pressure).
    #[must_use]
    pub fn with_park_tier(self, tier: Box<dyn Parkable>) -> Self {
        self.lock().park_tier = Some(tier);
        self
    }

    /// Sets the eviction policy on a scheduler that is already built.
    ///
    /// The policy is a field of [`SchedulerConfig`], which a caller that composes a scheduler out
    /// of a config it did not build (the serve plane: the config comes from the engine's geometry,
    /// the policy from `--park-on-pressure`) has no other way to reach. Before ws-k17
    /// [`crate::attach_park_tier`] attached the tier and DROPPED the policy beside it, so
    /// `saber serve --park-on-pressure` printed `on-pressure true` and ran with the default policy
    /// — every KV refusal a wait, never an eviction (1,320 of them in one run, 0 parks).
    #[must_use]
    pub fn with_park_policy(self, policy: crate::park::ParkPolicy) -> Self {
        self.lock().config.park = policy;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Core> {
        self.core.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_executor(&self) -> std::sync::MutexGuard<'_, E> {
        self.executor.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs exactly one engine step.
    ///
    /// Three phases, and the middle one is the point: **the engine lock is not held across the
    /// executor call.** Admission and batch building happen under it, the device runs without it,
    /// and the commit takes it again — matching results by [`SeqId`], because a sequence can be
    /// aborted while its step is in flight.
    pub fn step(&self) -> StepOutcome {
        let mut clock75 = self.stepping.lock().unwrap_or_else(|e| e.into_inner());
        let t_start = self.clock.now();

        // Phase 1: admit, choose the width and the bucket, build the batch.
        let mut census157 = (0u32, 0u32, [0u32; 8]);
        let ready = {
            let mut core = self.lock();
            let mut exec = self.lock_executor();
            core.drain_retiring(&mut *exec);
            match core.prepare(&mut *exec) {
                Prepared::Idle => {
                    // An idle poll still publishes: an abort's retirement may have moved the
                    // executor's counters, and a first scrape may be waiting on this (ws-v8). A
                    // no-op until a scrape has armed the mirror.
                    exec.publish_census();
                    clock75.idle_polls += 1;
                    // Both guards go before the housekeeping below, which is I/O.
                    drop(exec);
                    drop(core);
                    self.maintain_park_tier();
                    return StepOutcome::Idle;
                }
                Prepared::NoBucket(e) => return StepOutcome::NoBucket(e),
                Prepared::Ready(ready) => {
                    // FN157: the queue left behind and each admitted request's arrival -> entry
                    census157.0 = core.waiting.len() as u32;
                    census157.1 = core.admit_arrivals.len() as u32;
                    for (k, a) in core.admit_arrivals.iter().take(8).enumerate() {
                        census157.2[k] =
                            u32::try_from(t_start.saturating_duration_since(*a).as_micros())
                                .unwrap_or(u32::MAX);
                    }
                    ready
                }
            }
        };

        // Phase 2: the device. No core lock: this is the ~22 ms that `/metrics`, `submit` and
        // `abort` must not queue behind.
        let clock = self.clock.now();
        let (result, chunk_result, more_results) = {
            let mut exec = self.lock_executor();
            let batch = ready.batch();
            match ready.chunk.as_deref() {
                // C10: a packed step's chunks ride in one call (`StepExecutor::execute_packed`).
                Some(chunk) if !ready.more.is_empty() => {
                    let chunks: Vec<StepBatch<'_>> = std::iter::once(chunk)
                        .chain(ready.more.iter())
                        .map(ReadyStep::batch)
                        .collect();
                    let (d, mut cs) = exec.execute_packed(&batch, &chunks);
                    let rest = if cs.len() > 1 {
                        cs.split_off(1)
                    } else {
                        Vec::new()
                    };
                    (d, cs.into_iter().next(), rest)
                }
                // C4: one call runs both halves (`StepExecutor::execute_mixed`).
                Some(chunk) => {
                    let (d, c) = exec.execute_mixed(&batch, &chunk.batch());
                    (d, Some(c), Vec::new())
                }
                None => (exec.execute(&batch), None, Vec::new()),
            }
        };
        let exec_end = self.clock.now();
        let wall_us = (exec_end - clock).as_micros() as u64;

        // Phase 3: commit what came back, against whatever is still running. A mixed step
        // commits its decode half first — its slots' tokens were the step's point — and then its
        // chunk, each under its own contract, as ONE step.
        let mut core = self.lock();
        let mut exec = self.lock_executor();
        let mut summary = core.commit_half(&mut *exec, &ready, result);
        if let (Some(chunk), Some(r)) = (ready.chunk.as_deref(), chunk_result) {
            let c = core.commit_half(&mut *exec, chunk, r);
            summary.committed_tokens += c.committed_tokens;
            summary.finished += c.finished;
        }
        // C10: a packed step's further chunks, in queue order, each under its own contract.
        for (chunk, r) in ready.more.iter().zip(more_results) {
            let c = core.commit_half(&mut *exec, chunk, r);
            summary.committed_tokens += c.committed_tokens;
            summary.finished += c.finished;
        }
        core.step_index += 1;
        core.counters.steps += 1;
        core.counters.record_step_wall(&ready, wall_us);
        // LR-F: the admit-slack predictor and this step's hold (nothing at all with the arm off)
        core.slack.hint = None;
        if core.config.admit.slack_ms > 0 {
            if ready.phase == Phase::Decode && ready.chunk.is_none() {
                core.slack.decode_wall(ready.slots.len(), wall_us);
            }
            core.slack_rule(summary.finished, exec_end);
        }
        // FN75: the served-step ledger's scheduler half
        let end = self.clock.now();
        let chunk_rows = |r: &ReadyStep| -> u32 {
            r.slots
                .iter()
                .filter_map(|x| x.chunk.map(|c| c.prompt_tokens))
                .sum()
        };
        let (decode_rows, mut rows, mut chunks) = if ready.phase == Phase::Prefill {
            (0, chunk_rows(&ready), ready.slots.len() as u32)
        } else {
            (ready.slots.len() as u32, 0, 0)
        };
        for c in ready.chunk.as_deref().into_iter().chain(ready.more.iter()) {
            rows += chunk_rows(c);
            chunks += c.slots.len() as u32;
        }
        exec.step_done(&StepTiming {
            prev_end: clock75.last_end,
            work_at_prev_end: clock75.work_at_end,
            idle_polls: clock75.idle_polls,
            start: t_start,
            exec_start: clock,
            exec_end,
            end,
            phase: ready.phase,
            decode_rows,
            width: ready.width,
            chunk_rows: rows,
            chunks,
            waiting_decoders: ready.waiting_decoders,
            running: ready.running,
            committed_tokens: summary.committed_tokens,
            finished: summary.finished,
            waiting: census157.0,
            admitted: census157.1,
            admit_lag_us: census157.2,
        });
        clock75.last_end = Some(end);
        clock75.work_at_end = !core.running.is_empty() || !core.waiting.is_empty();
        clock75.idle_polls = 0;
        // The step's census, for a scrape that is watching (ws-v8). One relaxed load otherwise.
        exec.publish_census();
        StepOutcome::Ran(summary)
    }

    /// Parking-tier housekeeping, on a step that had nothing to run.
    ///
    /// The only thing it does today is pull a spilled session image back from NVMe into RAM, so
    /// that session's restore is a ~5 ms copy instead of a ~15 ms drive read — and it is **off**
    /// unless the tier was configured to want it, because it is a drive read plus a copy and the
    /// only moment that is free is a step with nothing running.
    ///
    /// Two rules keep it out of the way of the split that `docs/LEDGER.md` (2026-09-19) exists
    /// for. It runs **after** both step guards are dropped, so it never extends a step's critical
    /// section. And it takes the core lock **without blocking**: if the serving plane is asking a
    /// question right now, the promotion waits for the next idle step, one `IDLE_POLL` later,
    /// rather than putting a drive read in front of a `/metrics` scrape. It touches the tier and
    /// the ledger only — never the executor, and therefore never the device.
    fn maintain_park_tier(&self) {
        let Ok(mut core) = self.core.try_lock() else {
            return;
        };
        let Core {
            park_tier, ledger, ..
        } = &mut *core;
        if let Some(tier) = park_tier.as_mut() {
            tier.maintain(ledger);
        }
    }

    /// Runs up to `budget` steps, stopping early when the engine goes idle. Returns the steps run.
    ///
    /// The budget is not a nicety: a mock that never emits a stop token would otherwise spin
    /// forever, and a test that hangs is a test nobody runs.
    pub fn run_steps(&self, budget: usize) -> usize {
        let mut ran = 0;
        for _ in 0..budget {
            // The lock is taken per step, never across the loop: a submit from the HTTP side must
            // not wait on a whole run.
            match self.step() {
                StepOutcome::Idle => break,
                StepOutcome::NoBucket(_) => {
                    ran += 1;
                    break;
                }
                StepOutcome::Ran(_) => ran += 1,
            }
        }
        ran
    }

    /// Whether anything is running or queued.
    pub fn has_work(&self) -> bool {
        let core = self.lock();
        !core.running.is_empty() || !core.waiting.is_empty()
    }

    /// Which-path counters.
    pub fn counters(&self) -> SchedCounters {
        self.lock().counters
    }

    /// The `saber:spec_decode_*` counters in full, including
    /// `num_accepted_tokens_per_pos_total[i]` — the per-position acceptance curve
    /// `eval_drafter.py` reads to see *where* acceptance dies.
    ///
    /// In-process callers take the whole struct here; `/metrics` renders the same snapshot
    /// through [`saber_serve::engine::EngineStats`], which carries the three totals **and** the
    /// histogram (ws-q1, 2026-09-22 — prod exposes `..._per_pos_total` and Saber's exposition
    /// did not, so a harness outside the process could read the rate and never the curve).
    pub fn spec_counters(&self) -> SpecDecodeCounters {
        self.lock().spec.clone()
    }

    /// Admission counters (which allocator refused, and how often).
    pub fn admission_counters(&self) -> crate::alloc::AdmissionCounters {
        self.lock().arenas.counters()
    }

    /// GDN state slots the running set **touches** right now, against the slots it has
    /// **reserved**.
    ///
    /// The two differ while a sequence is prefilling: a prompt chunk carries one recurrent state
    /// (`Bucket::gdn_state_slots` for a prefill bucket is `batch`, not `batch × spec_width`),
    /// where a decode step holds one per draft token. The reservation is the decode-width set
    /// from admission either way — a sequence that could not grow into decode after its prompt
    /// was in would be an OOM at token 4000 wearing a different hat — so this is the accounting,
    /// and the Infinity-Cache budget a prefill step is gated against, not a second allocator.
    pub fn state_slots_active_vs_reserved(&self) -> (u32, u32) {
        let core = self.lock();
        let decode_slots = core.config.gdn_slots_per_seq();
        let active = core
            .running
            .iter()
            .map(|s| s.active_state_slots(decode_slots))
            .sum();
        let reserved = core.running.len() as u32 * decode_slots;
        (active, reserved)
    }

    /// Free KV blocks and free GDN state slot sets, for tests and `/metrics`.
    pub fn arena_free(&self) -> (u32, u32) {
        let core = self.lock();
        (
            core.arenas.kv().free_blocks(),
            core.arenas.gdn().free_sets(),
        )
    }

    /// A copy of the byte ledger.
    pub fn ledger_snapshot(&self) -> LedgerSnapshot {
        self.lock().ledger.snapshot()
    }

    /// The configuration in force.
    pub fn config(&self) -> SchedulerConfig {
        self.lock().config.clone()
    }

    /// Runs `f` against the executor. A step in flight finishes first; the scheduler's own state
    /// is not locked while this is held.
    pub fn with_executor<R>(&self, f: impl FnOnce(&mut E) -> R) -> R {
        f(&mut self.lock_executor())
    }

    /// Parks a running session: its KV and GDN state leave the device as one unit and both
    /// allocations are freed immediately, so the space is available to the next admission.
    /// Parking needs the executor (to copy the state image out) and must not wait on it: a caller
    /// blocking here would be blocking on the device. If a step is in flight the park is refused
    /// with [`ParkError::StepInFlight`] and the caller retries — parking is a session-layer
    /// decision on a per-request time constant, not a step-loop one.
    pub fn park(&self, id: RequestId) -> Result<(), ParkError> {
        let mut core = self.lock();
        let Ok(mut exec) = self.executor.try_lock() else {
            core.counters.park_refusals += 1;
            return Err(ParkError::StepInFlight(id));
        };
        core.park(&mut *exec, id)
    }

    /// Queues a parked session to be restored. It is put at the **head** of the queue, so it is
    /// restored before its next step and before any newly arrived request.
    pub fn unpark(&self, id: RequestId) -> Result<(), ParkError> {
        self.lock().unpark(id)
    }

    /// Requests currently parked.
    pub fn parked_ids(&self) -> Vec<RequestId> {
        self.lock().parked.iter().map(|p| p.request_id).collect()
    }

    /// The parking tier's which-path counters, or zeroes when no tier is attached.
    pub fn park_counters(&self) -> ParkCounters {
        self.lock()
            .park_tier
            .as_ref()
            .map_or_else(ParkCounters::default, |t| t.counters())
    }

    /// FN52: the prefix cache's counters, its entry count and the host bytes it holds per rank.
    pub fn prefix_counters(&self) -> (crate::prefix_cache::PrefixCacheCounters, usize, u64) {
        let core = self.lock();
        (
            core.prefix.counters,
            core.prefix.len(),
            core.prefix.host_bytes(),
        )
    }

    /// FN52: the KV blocks the allocator holds (sequences + retained prefixes) and its total.
    pub fn kv_blocks_held(&self) -> (u32, u32) {
        let core = self.lock();
        let kv = core.arenas.kv();
        (kv.total_blocks() - kv.free_blocks(), kv.total_blocks())
    }

    /// LR-A: the KV arena and the running reservations now, and the requests (not steps) the KV
    /// arena refused at the head of the queue so far.
    pub fn kv_admission(&self) -> (KvAdmissionSnapshot, u64) {
        let core = self.lock();
        (core.kv_admission_snapshot(), core.kv_refused_requests)
    }
}

impl Core {
    fn tokens_to_reserve(&self, prompt_len: usize, max_tokens: usize) -> u64 {
        // Reserve the whole sequence up front, plus one speculative overshoot: a step commits up to
        // `spec_width + 1` tokens and a real executor writes the draft tokens' KV before the
        // verify prunes them. This is what "admission refuses and it never OOMs later" costs.
        (prompt_len + max_tokens) as u64
            + u64::from(self.config.max_spec_width().max(self.config.tree_reserve))
            + 1
    }

    /// LR-G: the lazy reservation -- `prompt + min(max_tokens, headroom)` + the speculative
    /// overshoot ([`Self::tokens_to_reserve`] at the clamped budget).
    fn lazy_reserve(&self, prompt_len: usize, max_tokens: usize) -> u64 {
        let h = self.config.admit.lazy_headroom as usize;
        self.tokens_to_reserve(prompt_len, max_tokens.min(h))
    }

    /// LR-G: the tokens a decoding lazy sequence must have reserved before its next step: every
    /// committed token + the widest step's speculative rows + 1, never past its whole budget.
    fn lazy_need(&self, s: &Seq) -> u64 {
        let spec = u64::from(self.config.max_spec_width().max(self.config.tree_reserve)) + 1;
        let whole = (s.prompt_len + s.max_tokens) as u64 + spec;
        (s.tokens.len() as u64 + spec).min(whole)
    }

    /// **LR-G: lazy growth, before the step is built.** Every decoding lazy sequence whose next
    /// step could write past its blocks takes the missing blocks now, OLDEST first (the
    /// admission order: an older sequence is never preempted for a younger one). A growth the
    /// arena refuses preempts the youngest lazy sequence (never an older one; itself only when it
    /// is the youngest) and retries. The invariant: when this returns, every steppable lazy
    /// sequence's blocks cover [`Self::lazy_need`] -- no step is ever built over a block that
    /// does not exist, so no running sequence meets a full arena mid-step.
    fn grow_lazy<E: StepExecutor>(&mut self, exec: &mut E) {
        if !self.config.admit.lazy() {
            return;
        }
        let mut order: Vec<(u64, u64)> = self
            .running
            .iter()
            .filter(|s| s.lazy)
            .map(|s| (s.admitted_at_step, s.seq_id.0))
            .collect();
        order.sort_unstable();
        for (_, id) in order {
            // (a sequence preempted below as the youngest leaves the running set: done)
            while let Some(i) = self.running.iter().position(|s| s.seq_id.0 == id) {
                let s = &self.running[i];
                if s.prefilling() || !s.steppable() {
                    break; // a prompt is inside its admission reservation
                }
                let need = self.lazy_need(s);
                let have = self.arenas.kv().blocks_for(need);
                let more = have.saturating_sub(s.kv.blocks.len() as u32);
                if more == 0 {
                    break;
                }
                let mut kv = std::mem::replace(
                    &mut self.running[i].kv,
                    KvAllocation {
                        blocks: Vec::new(),
                        shared_prefix: 0,
                        cow_source: None,
                        capacity_tokens: 0,
                    },
                );
                let grown = self.arenas.extend_kv(&mut kv, more);
                self.running[i].kv = kv;
                match grown {
                    Ok(()) => {
                        self.slack.counters.lazy_grows += 1;
                        self.slack.counters.lazy_grow_blocks += u64::from(more);
                        break;
                    }
                    Err(_) => {
                        self.slack.counters.lazy_grow_refused += 1;
                        // the youngest lazy, steppable-or-prefilling sequence (no transfer, no
                        // pending fork); ties by id
                        let victim = self
                            .running
                            .iter()
                            .enumerate()
                            .filter(|(_, v)| {
                                v.lazy && v.transfer.is_none() && v.pending_fork.is_none()
                            })
                            .max_by_key(|(_, v)| (v.admitted_at_step, v.seq_id.0))
                            .map(|(j, _)| j);
                        match victim {
                            Some(j) => self.preempt_lazy(exec, j),
                            None => break,
                        }
                    }
                }
            }
        }
    }

    /// LR-G: preempt `running[idx]` (recompute): its allocations go back NOW (no step is in
    /// flight: `prepare` holds both locks), the executor releases it, and it goes to the FRONT of
    /// the queue with its prompt + committed tokens as the prompt to prefill again. Its stream
    /// continues: same request, same event index, same token budget, same sampling position.
    fn preempt_lazy<E: StepExecutor>(&mut self, exec: &mut E, idx: usize) {
        let seq = self.running.remove(idx);
        let recompute = Recompute {
            prompt_len: seq.prompt_len,
            generated: seq.generated,
            max_tokens: seq.max_tokens,
            step_index: seq.step_index,
            preemptions: seq.preempts + 1,
        };
        self.slack.counters.lazy_preemptions += 1;
        // the rows the re-admission prefills again (its KV was written up to here)
        let written = if seq.prefilling() {
            seq.prefilled
        } else {
            seq.tokens.len().saturating_sub(1)
        };
        self.slack.counters.lazy_recompute_tokens += written as u64;
        tracing::info!(
            request = %seq.request_id, seq = %seq.seq_id, written, generated = seq.generated,
            preemptions = recompute.preemptions,
            "lazy KV growth: preempted for recompute (the youngest sequence)"
        );
        let w = Waiting {
            request_id: seq.request_id,
            params: seq.params.clone(),
            prompt: seq.tokens.clone(),
            prompt_len: seq.tokens.len(),
            max_tokens: seq.max_tokens - seq.generated.min(seq.max_tokens),
            n: 1,
            events: seq.events.clone(),
            resume: None,
            images: None,
            arrived: None,
            recompute: Some(Box::new(recompute)),
        };
        self.retiring.push(Retired {
            seq_id: seq.seq_id,
            kv: seq.kv,
            gdn: seq.gdn,
            admitted: seq.pending_fork.is_none(),
        });
        self.drain_retiring(exec);
        self.waiting.push_front(w);
    }

    fn admit_waiting<E: StepExecutor>(&mut self, exec: &mut E) -> u32 {
        let mut admitted = 0;
        loop {
            if self.waiting.is_empty() {
                break;
            }
            let max_batch = self.config.max_batch();
            let (head_id, prompt_len, max_tokens, n) = {
                let head = self.waiting.front().expect("non-empty");
                (head.request_id, head.prompt_len, head.max_tokens, head.n)
            };
            // The batch cap counts SEQUENCES -- a group of three occupies three slots of the
            // captured shape, and admitting two of its branches would be admitting a request that
            // cannot answer.
            if self.running.len() as u32 + n > max_batch {
                self.arenas.refuse(Refusal::BatchFull {
                    running: self.running.len() as u32,
                    max_batch,
                });
                break;
            }
            // D4: the tower's one output slot. An image request at the head waits (head-of-line,
            // counted) while ANOTHER request's rows there are still owed to its prefill.
            if self.waiting.front().is_some_and(|h| h.images.is_some())
                && self.vision_slot_busy(head_id)
            {
                self.counters.vision_waits += 1;
                break;
            }
            // D3: the vision tower, once per request, before anything is allocated. A tower that
            // refuses (a text-only executor) or fails ends the request with the reason -- never
            // a prefill over image rows nobody computed.
            if let Some(images) = self.waiting.front_mut().and_then(|h| h.images.take()) {
                self.vision_owner = Some(head_id);
                if let Err(why) = exec.encode_images(head_id, &images) {
                    let w = self.waiting.pop_front().expect("non-empty");
                    tracing::error!(request = %head_id, error = %why, "the vision tower refused a request");
                    for choice in 0..w.n {
                        emit_error(&w.events, w.request_id, choice, 0, &why);
                    }
                    continue;
                }
            }
            // LR-G: a plain request reserves lazily (prompt + min(max_tokens, headroom))
            let lazy = self.config.admit.lazy() && {
                let h = self.waiting.front().expect("non-empty");
                h.n == 1
                    && h.resume.is_none()
                    && self.vision_owner != Some(head_id)
                    && h.params.prompt_logprobs.is_none()
            };
            let want = if lazy {
                self.lazy_reserve(prompt_len, max_tokens)
            } else {
                self.tokens_to_reserve(prompt_len, max_tokens)
            };
            // LR-G: a lazy admission leaves the margin free for the running sequences' growth
            if lazy && !self.running.is_empty() && self.config.admit.lazy_margin > 0 {
                let kv = self.arenas.kv();
                let need =
                    kv.blocks_for(want) + kv.blocks_for(u64::from(self.config.admit.lazy_margin));
                if need > kv.free_blocks() {
                    let refusal = Refusal::KvBlocks {
                        want: need,
                        free: kv.free_blocks(),
                    };
                    self.arenas.refuse(refusal);
                    self.slack.counters.lazy_margin_refusals += 1;
                    self.note_kv_refusal(head_id, prompt_len, max_tokens, want, refusal);
                    break;
                }
            }
            // FN52: the longest retained prefix, adopted by reference (no copy).
            let plan = if self.waiting.front().is_some_and(|h| h.recompute.is_some()) {
                PrefixPlan::default()
            } else {
                self.prefix_plan(head_id)
            };
            let taken = match plan.adopt.as_ref() {
                Some(hit) => self
                    .arenas
                    .allocate_prefixed(want, &hit.blocks)
                    .map(|(kv, gdn)| (vec![kv], vec![gdn])),
                None => self.arenas.allocate_group(n, want, prompt_len as u64),
            };
            let (kvs, gdns) = match taken {
                Ok(pair) => pair,
                Err(refusal) => {
                    // FN52: a retained prefix never blocks an admission. Entries go first,
                    // least recently used first (the one this head would adopt last), before
                    // any running session is parked.
                    if matches!(refusal, Refusal::KvBlocks { .. })
                        && self
                            .evict_prefix_for_pressure(exec, plan.adopt.as_ref().map(|h| h.entry))
                    {
                        continue;
                    }
                    // The one place the eviction policy fires: an admission that the *arenas*
                    // refused. A `BatchFull` refusal is not pressure on anything a park would
                    // relieve, so it is left alone.
                    if matches!(
                        refusal,
                        Refusal::KvBlocks { .. } | Refusal::GdnStateSlots { .. }
                    ) && self.park_under_pressure(exec, head_id)
                    {
                        continue;
                    }
                    // Head-of-line: the queue is not scanned past a refusal, so a long request is
                    // never starved by a stream of short ones.
                    tracing::debug!(request = %head_id, n, %refusal, "admission refused");
                    if matches!(refusal, Refusal::KvBlocks { .. }) {
                        self.note_kv_refusal(head_id, prompt_len, max_tokens, want, refusal);
                        // LR-F: an arm (off by default): a later request that fits goes first
                        if self.skip_blocked_head(head_id) {
                            continue;
                        }
                    }
                    break;
                }
            };
            let w = self.waiting.pop_front().expect("non-empty");
            let arrived = w.arrived;
            match self.start_group(exec, w, kvs, gdns, plan, lazy) {
                Ok(()) => {
                    admitted += n;
                    self.admit_arrivals.extend(arrived);
                }
                // The executor's one staging buffer is busy. `start_group` has already put the
                // group back at the FRONT of the queue with its images intact, so this step stops
                // admitting and the next one tries again — without this `break` the loop would pop
                // the same group forever.
                Err(ParkError::TransferBusy(_)) => break,
                Err(_) => {}
            }
        }
        admitted
    }

    /// **LR-F: the admit-slack rule**, at the end of a step that finished `finished` sequences
    /// (`at` = its executor return). Sets `slack.hint` or leaves it `None`.
    ///
    /// Fires only when the step finished a sequence, the queue is empty, a seat is free, nothing
    /// running owes prefill (the next step would be a chunk, which a hold only delays) and `d >= 1`
    /// sequences decode. With `T` the EMA decode wall at `d` rows (no estimate = no hold) the
    /// window is `H = min(slack_ms, T)`. The predictor is the closed-loop `fnadmit` law
    /// leans on: `p` = EMA of P(an arrival within the cap after a finish), `g` = EMA of that
    /// arrival's gap. Row-time ledger of one hold: the newcomer is admitted `T - g` earlier (one
    /// row, with probability `p`); every decoder waits `g` when it comes and `H` when it does not.
    /// Hold iff `p (T - g) > d (p g + (1 - p) H)` (or `slack_force`). It cannot starve: one hold
    /// a finishing step, at most `slack_ms`, never chained (a hold step finishes nothing).
    fn slack_rule(&mut self, finished: u32, at: std::time::Instant) {
        let cap_ms = self.config.admit.slack_ms;
        let cap = std::time::Duration::from_millis(u64::from(cap_ms));
        self.slack.close_stale(at, cap);
        if finished == 0 {
            return;
        }
        self.slack.open_finish = Some(at);
        let d = self
            .running
            .iter()
            .filter(|s| !s.prefilling() && s.steppable())
            .count();
        let owes = self.running.iter().any(|s| s.prefilling());
        if !self.waiting.is_empty()
            || self.running.len() as u32 >= self.config.max_batch()
            || owes
            || d == 0
        {
            return;
        }
        let t = self.slack.dec_wall_us.get(d).copied().unwrap_or(0.0);
        if t <= 0.0 {
            return;
        }
        let h = t.min(f64::from(cap_ms) * 1e3);
        if !self.config.admit.slack_force {
            let p = self.slack.p_follow;
            let g = self.slack.gap_us.min(h);
            let gain = p * (t - g);
            let cost = d as f64 * (p * g + (1.0 - p) * h);
            if gain <= cost {
                self.slack.counters.slack_declined += 1;
                return;
            }
        }
        self.slack.hint = Some(std::time::Duration::from_micros(h as u64));
    }

    /// LR-F (`AdmitPolicy::skip_kv_blocked`): move the first of the next `skip_kv_blocked` queued
    /// requests whose reservation fits the free KV blocks now (a plain one: `n == 1`, no images,
    /// not a parked session) to the front, past the KV-refused `head`. A head is passed at most
    /// `AdmitPolicy::SKIP_LIMIT` times; after that the queue is strict head-of-line until it runs.
    fn skip_blocked_head(&mut self, head: RequestId) -> bool {
        let scan = self.config.admit.skip_kv_blocked as usize;
        if scan == 0 || self.arenas.gdn().free_sets() == 0 {
            return false;
        }
        let passed = if self.hol_skip.0 == Some(head) {
            self.hol_skip.1
        } else {
            0
        };
        if passed >= crate::config::AdmitPolicy::SKIP_LIMIT {
            return false;
        }
        let free = self.arenas.kv().free_blocks();
        let pick = (1..self.waiting.len().min(scan + 1)).find(|&i| {
            let w = &self.waiting[i];
            w.n == 1
                && w.images.is_none()
                && w.resume.is_none()
                && self.arenas.kv().blocks_for(if self.config.admit.lazy() {
                    self.lazy_reserve(w.prompt_len, w.max_tokens)
                } else {
                    self.tokens_to_reserve(w.prompt_len, w.max_tokens)
                }) <= free
        });
        let Some(i) = pick else {
            return false;
        };
        let w = self.waiting.remove(i).expect("in range");
        self.waiting.push_front(w);
        self.hol_skip = (Some(head), passed + 1);
        self.slack.counters.hol_skips += 1;
        true
    }

    /// LR-A: the KV arena and the running set's reservations now.
    fn kv_admission_snapshot(&self) -> KvAdmissionSnapshot {
        let kv = self.arenas.kv();
        let mut s = KvAdmissionSnapshot {
            total_blocks: kv.total_blocks(),
            free_blocks: kv.free_blocks(),
            ..KvAdmissionSnapshot::default()
        };
        for q in &self.running {
            // LR-G: a lazy sequence's reservation is what it holds
            let r = if q.lazy {
                q.kv.capacity_tokens
            } else {
                (q.prompt_len + q.max_tokens) as u64
            };
            s.reserved_tokens += r;
            s.max_reserved_tokens = s.max_reserved_tokens.max(r);
            s.length_tokens += (q.prompt_len + q.generated) as u64;
        }
        s
    }

    /// LR-A: one info line per refused queue head (the head changes only when it is admitted or
    /// aborted, so this is rate-limited by the request rate), and the per-request counter.
    fn note_kv_refusal(
        &mut self,
        head: RequestId,
        prompt_len: usize,
        max_tokens: usize,
        want: u64,
        refusal: Refusal,
    ) {
        if self.kv_refused_head == Some(head) {
            return;
        }
        self.kv_refused_head = Some(head);
        self.kv_refused_requests += 1;
        let s = self.kv_admission_snapshot();
        tracing::info!(
            request = %head,
            prompt_len,
            max_tokens,
            want_tokens = want,
            %refusal,
            running = self.running.len(),
            waiting = self.waiting.len(),
            total_blocks = s.total_blocks,
            free_blocks = s.free_blocks,
            running_reserved_tokens = s.reserved_tokens,
            running_length_tokens = s.length_tokens,
            running_max_reserved_tokens = s.max_reserved_tokens,
            "admission refused for KV (logged once per queue head)"
        );
    }

    /// FN52: what admission does about the queue head's prefix — adopt the longest retained one
    /// (or, in the verify arm, compare against it), and where to snapshot this prompt.
    fn prefix_plan(&mut self, head_id: RequestId) -> PrefixPlan {
        if !self.prefix.enabled() || self.config.prefill.is_empty() {
            return PrefixPlan::default();
        }
        let Some(head) = self.waiting.front() else {
            return PrefixPlan::default();
        };
        // Groups, resumed sessions, image prompts and prompt-logprob requests are never cached:
        // a group's branches share pages already, an image block's key needs the image's hash,
        // and a prompt-logprob request needs every row scored.
        if head.n != 1
            || head.resume.is_some()
            || head.images.is_some()
            || self.vision_owner == Some(head_id)
            || head.params.prompt_logprobs.is_some()
        {
            return PrefixPlan::default();
        }
        let prompt = head.prompt.clone();
        let prompt_len = head.prompt_len;
        let hit = self.prefix.lookup(&prompt);
        let snap_at = self
            .prefix
            .snapshot_point(prompt_len)
            .filter(|a| !self.prefix.holds(&prompt, *a))
            .filter(|a| hit.as_ref().is_none_or(|h| *a > h.tokens));
        if self.prefix.policy().verify {
            PrefixPlan {
                looked: true,
                adopt: None,
                snap_at,
                verify_at: hit.map(|h| (h.entry, h.tokens, h.blocks)),
            }
        } else {
            PrefixPlan {
                looked: true,
                adopt: hit,
                snap_at,
                verify_at: None,
            }
        }
    }

    /// FN52: drop the least recently used retained prefix (other than `keep`, unless it is the
    /// last) so an admission the KV arena refused can be retried. `false` when none is left.
    fn evict_prefix_for_pressure<E: StepExecutor>(
        &mut self,
        exec: &mut E,
        keep: Option<u32>,
    ) -> bool {
        let Some(e) = self
            .prefix
            .evict_lru(keep)
            .or_else(|| self.prefix.evict_lru(None))
        else {
            return false;
        };
        exec.prefix_drop(e.id);
        self.arenas.release_kv(&e.blocks);
        self.prefix.counters.evictions_pressure += 1;
        tracing::info!(
            entry = e.id,
            tokens = e.tokens.len(),
            "prefix cache: entry evicted for KV pressure"
        );
        true
    }

    /// FN52: `seq_id`'s prefill just reached one of its cuts: verify against the hit it was told
    /// to compare with, and/or snapshot and retain its prefix.
    fn prefix_at_cut<E: StepExecutor>(&mut self, exec: &mut E, seq_id: SeqId) {
        let Some(pos) = self.running.iter().position(|s| s.seq_id == seq_id) else {
            return;
        };
        let at = self.running[pos].prefilled as u32;
        if let Some((entry, tokens, blocks)) =
            self.running[pos].verify_at.clone().filter(|v| v.1 == at)
        {
            self.running[pos].verify_at = None;
            if self.prefix.get(entry).is_none() {
                self.prefix.counters.verify_skipped += 1;
            } else {
                match exec.prefix_verify(seq_id, entry, tokens, &blocks) {
                    Ok(true) => {
                        self.prefix.counters.verify_match += 1;
                        tracing::info!(seq = %seq_id, entry, tokens,
                            "prefix cache verify: fresh == entry at {tokens} tokens");
                    }
                    Ok(false) => {
                        self.prefix.counters.verify_mismatch += 1;
                        tracing::error!(seq = %seq_id, entry, tokens,
                            "prefix cache verify: fresh != entry at {tokens} tokens");
                    }
                    Err(why) => {
                        self.prefix.counters.verify_skipped += 1;
                        tracing::warn!(seq = %seq_id, entry, error = %why,
                            "prefix cache verify: not compared");
                    }
                }
            }
        }
        if self.running[pos].snap_at != Some(at) {
            return;
        }
        self.running[pos].snap_at = None;
        let seq = &self.running[pos];
        if self.prefix.holds(&seq.tokens, at) {
            return;
        }
        let nblocks = (at / self.config.kv.block_tokens.max(1)) as usize;
        if nblocks == 0 || nblocks > seq.kv.blocks.len() {
            return;
        }
        let blocks = seq.kv.blocks[..nblocks].to_vec();
        let tokens = seq.tokens[..at as usize].to_vec();
        let id = self.prefix.reserve_id();
        match exec.prefix_save(seq_id, id, at) {
            Ok(bytes) => {
                self.arenas.retain_kv(&blocks);
                let evicted = self.prefix.insert(id, tokens, blocks, bytes);
                tracing::info!(
                    seq = %seq_id, entry = id, tokens = at, host_bytes = bytes,
                    entries = self.prefix.len(), held = self.prefix.host_bytes(),
                    evicted = evicted.len(),
                    "prefix cache: {at} tokens retained"
                );
                for e in evicted {
                    exec.prefix_drop(e.id);
                    self.arenas.release_kv(&e.blocks);
                }
            }
            Err(why) => {
                self.prefix.counters.save_failures += 1;
                tracing::warn!(seq = %seq_id, error = %why, "prefix cache: snapshot refused");
            }
        }
    }

    /// D4: whether the tower's output slot still holds rows another request's prefill has not
    /// consumed. The owner is released when it is neither queued nor prefilling -- its last image
    /// chunk has run, or it finished, was aborted or refused.
    fn vision_slot_busy(&self, head: RequestId) -> bool {
        match self.vision_owner {
            None => false,
            Some(o) if o == head => false,
            Some(o) => {
                self.waiting.iter().any(|w| w.request_id == o)
                    || self
                        .running
                        .iter()
                        // `first_step` too: a plan with no prefill bucket feeds the prompt as
                        // its first step, and that step must not see another request's rows.
                        .any(|s| s.request_id == o && (s.prefilling() || s.first_step))
            }
        }
    }

    /// Park one running session to make room for `waiting_for`, if the policy allows it.
    ///
    /// Returns whether anything was parked — the caller retries the allocation when it was, and
    /// gives up on the head of the queue when it was not. Every refusal is counted: a policy that
    /// silently never fires looks exactly like one that is working.
    ///
    /// The executor is passed in rather than tried for: this runs inside `prepare`, which holds
    /// both locks, so there is no step in flight and [`ParkError::StepInFlight`] cannot arise. A
    /// park decision belongs here and nowhere else — admission is the only caller that knows the
    /// arenas just refused.
    fn park_under_pressure<E: StepExecutor>(
        &mut self,
        exec: &mut E,
        waiting_for: RequestId,
    ) -> bool {
        let policy = self.config.park;
        if !policy.on_pressure || self.park_tier.is_none() {
            return false;
        }
        if self.parks_this_step >= policy.max_parks_per_step {
            return false;
        }
        // A POLICY PARK ALREADY IN FLIGHT is space on its way: wait for it (ws-k17). Choosing again
        // here picked the same victim at every step of its copy, and `park` refused it as
        // `NotRunning` — 25 warnings in one card run — while evicting a SECOND session for the same
        // admission would pay two transfers for one seat.
        // Pruned first: a group aborted while its copy ran leaves the running set without passing
        // through `finish_park`, and a stale entry here would make the policy wait forever.
        let running = &self.running;
        self.auto_parks_in_flight.retain(|id| {
            running
                .iter()
                .any(|s| s.request_id == *id && s.transfer.is_some_and(Transfer::parking))
        });
        if !self.auto_parks_in_flight.is_empty() {
            self.counters.policy_park_waits += 1;
            return false;
        }
        let step = self.step_index;
        // One candidate per **group**, not per branch. A session with three branches would
        // otherwise get three votes, three chances to be chosen, and would make `min_running`
        // read three running sessions where there is one. Aggregated the way an eviction decision
        // actually lands: the group is as old as its oldest branch, as idle as its *most recent*
        // activity, and frees every branch's remaining work when it goes.
        // The running queue's own order is prod's victim order (`VictimOrder::Newest`): admissions
        // and restores are both `extend`-ed onto the back, and a removal keeps the order of the
        // rest, exactly as prod's `self.running.append` / `remove_all` do.
        let mut by_request: BTreeMap<RequestId, ParkCandidate> = BTreeMap::new();
        for (pos, s) in self.running.iter().enumerate() {
            let remaining = (s.max_tokens - s.generated.min(s.max_tokens)) as u64;
            by_request
                .entry(s.request_id)
                .and_modify(|c| {
                    c.admitted_at_step = c.admitted_at_step.min(s.admitted_at_step);
                    c.last_active_step = c.last_active_step.max(s.last_active_step);
                    c.remaining_tokens += remaining;
                    c.queue_pos = c.queue_pos.max(pos as u32);
                })
                .or_insert(ParkCandidate {
                    request_id: s.request_id,
                    admitted_at_step: s.admitted_at_step,
                    last_active_step: s.last_active_step,
                    remaining_tokens: remaining,
                    parks_so_far: s.parks,
                    queue_pos: pos as u32,
                });
        }
        // A group with a transfer in flight (a restore still landing, a park still copying), a
        // branch not forked yet, or a prompt not all in (ws-k17: a park before the prefill was
        // restored as "prefilled" onto an engine holding nothing) cannot be parked — `park`
        // refuses it — so it is not a candidate.
        let busy: Vec<RequestId> = self
            .running
            .iter()
            .filter(|s| s.transfer.is_some() || s.pending_fork.is_some() || s.prefilling())
            .map(|s| s.request_id)
            .collect();
        let candidates: Vec<ParkCandidate> = by_request
            .into_values()
            .filter(|c| !busy.contains(&c.request_id))
            .collect();
        let Some(victim) = choose_victim(&candidates, &policy, step) else {
            self.counters.policy_park_misses += 1;
            return false;
        };
        match self.park(exec, victim) {
            // THE ASYNCHRONOUS PATH: `park` issued the copy and the group is still running with its
            // pages and slots. Nothing is free yet, so the admission waits; the park is the
            // policy's, and `finish_park` marks it so when it lands (ws-k17 — this branch used to
            // count it and mark `parked.last_mut()`, i.e. whichever session had been parked last).
            Ok(())
                if self.running.iter().any(|s| {
                    s.request_id == victim && s.transfer.is_some_and(Transfer::parking)
                }) =>
            {
                self.parks_this_step += 1;
                self.auto_parks_in_flight.insert(victim);
                tracing::debug!(%victim, waiting = %waiting_for, "policy park issued off the step loop");
                false
            }
            Ok(()) => {
                self.parks_this_step += 1;
                self.counters.policy_parks += 1;
                if let Some(p) = self.parked.last_mut() {
                    p.auto = true;
                }
                tracing::debug!(%victim, waiting = %waiting_for, "parked under admission pressure");
                true
            }
            // THE STAGING BUFFER IS BUSY (a restore is copying in): not a miss, the same wait as a
            // park in flight. The first armed card run logged 562 warnings for it (ws-k17).
            Err(ParkError::TransferBusy(_)) => {
                self.counters.policy_park_waits += 1;
                false
            }
            Err(e) => {
                // `park` has already counted the refusal; this counts the *policy's* miss, which
                // is the one that says the policy asked for something it could not have.
                self.counters.policy_park_misses += 1;
                tracing::warn!(%victim, error = %e, "eviction policy could not park its victim");
                false
            }
        }
    }

    /// Queue policy-parked **groups** for restore, oldest park first, into the slots that just
    /// came free.
    ///
    /// Called when sequences finish or a request is aborted, because that is when the space they
    /// were evicted for comes back. Without it an evicted session waits for an explicit `unpark`
    /// that nothing will send.
    ///
    /// `freed_slots` counts **sequences**. The **oldest** evicted group is always requeued when
    /// anything at all came free, and further groups only while the freed slots cover their
    /// branches.
    ///
    /// The asymmetry is deliberate. A group of three would otherwise wait for three sequences to
    /// finish *in one step*, which a stream of single-choice requests never does — it would be
    /// evicted once and never come back. Requeuing it on the first free slot puts it at the head
    /// of the queue, where admission is all-or-nothing: it waits there until the room for all its
    /// branches exists, which is the same head-of-line rule every other request is under, and
    /// `park_under_pressure` can evict again behind it if that takes too long.
    fn requeue_auto_parked(&mut self, freed_slots: u32) {
        if !self.config.park.auto_restore || freed_slots == 0 {
            return;
        }
        let mut budget = freed_slots;
        let mut ids: Vec<RequestId> = Vec::new();
        for p in self.parked.iter().filter(|p| p.auto) {
            let branches = p.branches.len() as u32;
            if ids.is_empty() {
                // The oldest one goes back regardless of its width.
                budget = budget.saturating_sub(branches);
            } else if branches > budget {
                break;
            } else {
                budget -= branches;
            }
            ids.push(p.request_id);
        }
        // Reversed, because `unpark` pushes to the FRONT of the queue: walking backwards leaves
        // the earliest-parked session at the very front, ahead of both later parks and new
        // arrivals.
        for id in ids.into_iter().rev() {
            if self.unpark(id).is_ok() {
                self.counters.policy_restores_queued += 1;
            }
        }
    }

    /// Turns an admitted queue entry into a running sequence. On a failed restore both allocations
    /// go straight back and the session returns to the parked set.
    /// Turns an admitted queue entry into a running **sequence group**: `n` branches over one
    /// prompt, sharing its whole KV blocks and each holding its own GDN state slot set.
    ///
    /// The branches all start at the prompt. What the executor is told — and what a device-side
    /// implementation acts on — is that branch `i > 0` already has the prompt's pages
    /// (`shared_prefix_blocks`) and owes a copy of the partial one (`cow_source`), so the prefill
    /// is computed once and the branches only sample. The host side of "prefill once" is the
    /// pages; the compute side is the executor's, and is stated in [`AdmittedSeq`] rather than
    /// faked here.
    fn start_group<E: StepExecutor>(
        &mut self,
        exec: &mut E,
        w: Waiting,
        kvs: Vec<KvAllocation>,
        gdns: Vec<GdnAllocation>,
        prefix: PrefixPlan,
        lazy: bool,
    ) -> Result<(), ParkError> {
        if prefix.looked {
            self.prefix.counters.lookups += 1;
        }
        if let Some((_, t, _)) = prefix.verify_at.as_ref() {
            self.prefix.counters.hits += 1;
            self.prefix.counters.hit_tokens += u64::from(*t);
        }
        let n = kvs.len();
        debug_assert_eq!(n, gdns.len());
        let mut seqs: Vec<Seq> = Vec::with_capacity(n);
        let mut admitted: Vec<AdmittedSeq> = Vec::with_capacity(n);

        for (choice_index, (kv, gdn)) in kvs.into_iter().zip(gdns).enumerate() {
            let seq_id = SeqId(self.next_seq);
            self.next_seq += 1;
            let seq_admitted = AdmittedSeq {
                request_id: w.request_id,
                choice_index: choice_index as u32,
                seq_id,
                prompt_tokens: w.prompt.clone(),
                kv_blocks: kv.blocks.clone(),
                shared_prefix_blocks: kv.shared_prefix,
                cow_source: kv.cow_source,
                gdn_set: gdn.set,
                gdn_slots: gdn.slots,
            };
            // THE FORK IS DEFERRED, and only for a fresh group's branches. Branch `i > 0` is
            // told to the executor at `StepExecutor::fork`, at the moment the leader's last
            // prefill chunk samples — the only moment the row its first token is drawn from
            // exists. A RESUMED group is different in kind: every branch carries its own image
            // and there is nothing to draw, so each is admitted here and restored below.
            let deferred = choice_index > 0 && w.resume.is_none() && exec.can_fork();
            if deferred {
                self.counters.forks_pending += 1;
            } else {
                exec.admit(&seq_admitted);
            }
            // FN52: the adopted prefix's state, into this sequence's own slots.
            if let Some(hit) = prefix.adopt.as_ref() {
                if let Err(why) = exec.prefix_load(&seq_admitted, hit.entry, hit.tokens) {
                    // Its pages are shared with the entry (and maybe with other sequences that
                    // adopted it): a fresh prefill from 0 would write into them. The entry goes
                    // and the request ends, loudly.
                    self.prefix.counters.load_failures += 1;
                    tracing::error!(request = %w.request_id, entry = hit.entry, error = %why,
                        "prefix cache: the snapshot could not be restored; the request is refused");
                    if let Some(e) = self.prefix.remove(hit.entry) {
                        exec.prefix_drop(e.id);
                        self.arenas.release_kv(&e.blocks);
                    }
                    exec.release(seq_id);
                    self.arenas.free(kv, gdn);
                    emit_error(
                        &w.events,
                        w.request_id,
                        0,
                        0,
                        &format!("prefix cache restore failed: {why}"),
                    );
                    return Ok(());
                }
                self.prefix.counters.hits += 1;
                self.prefix.counters.adopted += 1;
                self.prefix.counters.hit_tokens += u64::from(hit.tokens);
                tracing::info!(
                    request = %w.request_id, seq = %seq_id, entry = hit.entry,
                    adopted = hit.tokens, prompt = w.prompt_len,
                    "prefix cache hit: {} of {} prompt tokens adopted",
                    hit.tokens, w.prompt_len
                );
            }
            seqs.push(Seq {
                request_id: w.request_id,
                choice_index: choice_index as u32,
                seq_id,
                params: w.params.clone(),
                prompt_len: w.prompt_len,
                prefill_len: w.prompt_len,
                lazy: false,
                preempts: 0,
                tokens: w.prompt.clone(),
                generated: 0,
                max_tokens: w.max_tokens,
                kv,
                gdn,
                step_index: 0,
                // The eviction policy's clocks. Every branch of a group carries the same ones:
                // the group was admitted at one step and is a victim as one unit, so a per-branch
                // age would only give the policy `n` ways to say the same thing.
                admitted_at_step: self.step_index,
                last_active_step: self.step_index,
                parks: 0,
                auto_parked: false,
                restore_failures: 0,
                pending_input: w.prompt.clone(),
                first_step: true,
                // With no captured prefill bucket the prompt is fed by one decode-shaped step,
                // which is what this scheduler did before P2. Marking it already prefilled is
                // what routes it down that path instead of waiting for chunks that cannot run.
                prefilled: if self.config.prefill.is_empty() {
                    w.prompt_len
                } else {
                    prefix.adopt.as_ref().map_or(0, |h| h.tokens as usize)
                },
                ema: self.config.width.new_ema(),
                events: w.events.clone(),
                pending_fork: deferred.then(|| Box::new(seq_admitted.clone())),
                // A fresh admission has nothing in flight. A restore sets this before it issues.
                transfer: None,
                // The prompt is the GROUP's, so only the leader scores it; a resumed group
                // emitted its prompt logprobs before it was ever parked.
                prompt_lp: (choice_index == 0
                    && w.resume.is_none()
                    && w.params.prompt_logprobs.is_some())
                .then(|| vec![None]),
                cuts: prefix
                    .snap_at
                    .into_iter()
                    .chain(prefix.verify_at.as_ref().map(|v| v.1))
                    .collect(),
                snap_at: prefix.snap_at,
                verify_at: prefix.verify_at.clone(),
            });
            admitted.push(seq_admitted);
        }

        // LR-G: a lazily reserved sequence, and a recompute's continuation of its stream
        if lazy {
            for seq in &mut seqs {
                seq.lazy = true;
                if let Some(r) = w.recompute.as_deref() {
                    seq.prompt_len = r.prompt_len;
                    seq.generated = r.generated;
                    seq.max_tokens = r.max_tokens;
                    seq.step_index = r.step_index;
                    seq.preempts = r.preemptions;
                }
            }
        }
        if let Some(parked) = w.resume {
            match self.restore_group(exec, &mut seqs, &admitted, *parked) {
                Ok(()) => {}
                Err((e, parked)) => {
                    for seq in seqs {
                        exec.release(seq.seq_id);
                        self.arenas.free(seq.kv, seq.gdn);
                    }
                    let busy = matches!(e, ParkError::TransferBusy(_));
                    if busy {
                        // Not a failure — the executor's one staging buffer is in use. The group
                        // goes back to the FRONT of the queue with its images intact, so the next
                        // step tries again by itself rather than waiting for another `unpark` that
                        // nothing is going to send.
                        self.counters.park_transfer_busy += 1;
                        let n = parked.branches.len() as u32;
                        let prompt = parked
                            .branches
                            .first()
                            .map(|b| b.tokens.clone())
                            .unwrap_or_default();
                        self.waiting.push_front(Waiting {
                            request_id: parked.request_id,
                            params: parked.params.clone(),
                            prompt,
                            prompt_len: parked.prompt_len,
                            max_tokens: parked.max_tokens,
                            n,
                            events: parked.events.clone(),
                            resume: Some(Box::new(parked)),
                            images: None,
                            arrived: None,
                            recompute: None,
                        });
                    } else {
                        self.counters.restore_failures += 1;
                        let mut parked = parked;
                        parked.restore_failures += 1;
                        tracing::error!(
                            request = %parked.request_id,
                            error = %e,
                            attempt = parked.restore_failures,
                            "restore failed; session stays parked"
                        );
                        self.after_restore_failure(parked, &e.to_string());
                    }
                    return Err(e);
                }
            }
        }

        self.running.extend(seqs);
        Ok(())
    }

    /// A parked group whose restore just FAILED (its `restore_failures` already counts this one):
    /// back to the parked set for another attempt, or — at `ParkPolicy::max_restore_attempts` —
    /// ended with a named error (ws-k18).
    ///
    /// A POLICY-parked group is requeued for its next attempt at once: its allocations were handed
    /// back a moment ago, so the space it needs exists now, and waiting for another sequence to
    /// finish is waiting on traffic that may never come — the client would sit out its own timeout
    /// and the server could not stop. An explicit park waits for its operator, as it always has.
    fn after_restore_failure(&mut self, parked: ParkedSeq, why: &str) {
        let limit = self.config.park.max_restore_attempts.max(1);
        if parked.restore_failures >= limit {
            self.give_up_restore(parked, why);
            return;
        }
        let (id, auto) = (parked.request_id, parked.auto);
        self.parked.push(parked);
        if auto && self.config.park.auto_restore && self.unpark(id).is_ok() {
            self.counters.policy_restores_queued += 1;
        }
    }

    /// End a parked request whose restore has failed `max_restore_attempts` times: every branch
    /// gets its `finished` event WITH the reason (`StepEvent::error`, which the serving plane turns
    /// into an OpenAI error body), and the images are dropped from the tier and from the executor.
    fn give_up_restore(&mut self, parked: ParkedSeq, why: &str) {
        let message = format!(
            "the session was parked under memory pressure and could not be restored after {} attempt(s): {why}",
            parked.restore_failures
        );
        tracing::error!(request = %parked.request_id, attempts = parked.restore_failures, error = %why, "giving up on a parked session's restore; the request ends with an error");
        for branch in &parked.branches {
            emit_error(
                &parked.events,
                parked.request_id,
                branch.choice_index,
                branch.step_index,
                &message,
            );
        }
        self.discard_group(&parked);
        self.counters.restores_given_up += 1;
    }

    /// Put one branch's image back on the tier after a restore that took it off failed, and give
    /// the branch the new ticket. A failure here is logged by name: the session then has no image
    /// and every retry reads `NoSuchImage`, which is the state this exists to prevent (ws-k17).
    fn repark_branch(&mut self, parked: &mut ParkedSeq, branch: usize, image: SessionImage) {
        let Some(tier) = self.park_tier.as_mut() else {
            return;
        };
        match tier.park(image, &mut self.ledger) {
            Ok(ticket) => {
                if let Some(b) = parked.branches.get_mut(branch) {
                    b.ticket = ticket;
                }
            }
            Err(e) => tracing::error!(
                request = %parked.request_id,
                branch,
                error = %e,
                "a failed restore's image could not go back on the tier; the session is lost"
            ),
        }
    }

    /// Restores every branch of a parked group, or none of them.
    #[allow(clippy::result_large_err)]
    fn restore_group<E: StepExecutor>(
        &mut self,
        exec: &mut E,
        seqs: &mut [Seq],
        admitted: &[AdmittedSeq],
        mut parked: ParkedSeq,
    ) -> Result<(), (ParkError, ParkedSeq)> {
        if parked.branches.len() != seqs.len() {
            let detail = format!(
                "parked group has {} branches, admission made {}",
                parked.branches.len(),
                seqs.len()
            );
            return Err((
                ParkError::ImageMismatch {
                    request_id: parked.request_id,
                    detail,
                },
                parked,
            ));
        }
        if self.park_tier.is_none() {
            return Err((ParkError::NoTier, parked));
        }
        // Asked BEFORE the tier is touched: a busy answer after the tickets were spent could not be
        // retried, because the images would be off the tier with nothing left to ask for them.
        if exec.parks_async() && exec.transfer_in_flight() {
            return Err((ParkError::TransferBusy(parked.request_id), parked));
        }

        // Every image comes back before anything is written into a sequence: a group that
        // restored three of four branches would be a group that cannot answer.
        let mut images = Vec::with_capacity(parked.branches.len());
        let tickets: Vec<ParkTicket> = parked.branches.iter().map(|b| b.ticket).collect();
        for ticket in &tickets {
            let tier = self.park_tier.as_mut().expect("checked above");
            match tier.restore(ticket, &mut self.ledger) {
                Ok(image) => images.push(image),
                Err(e) => {
                    // Put back what was already taken, so the group stays parked whole — UNDER
                    // THE FRESH TICKETS (ws-k18): a put-back whose ticket was thrown away left the
                    // branch holding a spent one, and every retry read `NoSuchImage`.
                    for (i, image) in images.into_iter().enumerate() {
                        self.repark_branch(&mut parked, i, image);
                    }
                    return Err((e, parked));
                }
            }
        }

        // The scheduler's own half of "resumes byte-identically": the tokens the image carries are
        // the context the KV blocks are supposed to hold.
        let mismatch = parked
            .branches
            .iter()
            .zip(&images)
            .find(|(branch, image)| image.tokens != branch.tokens)
            .map(|(branch, image)| {
                format!(
                    "choice {}: image holds {} tokens, branch had {}",
                    branch.choice_index,
                    image.tokens.len(),
                    branch.tokens.len()
                )
            });
        if let Some(detail) = mismatch {
            // Back on the tier, so the next attempt is a real one and the bound counts attempts
            // rather than a spent ticket (ws-k18).
            for (i, image) in images.into_iter().enumerate() {
                self.repark_branch(&mut parked, i, image);
            }
            return Err((
                ParkError::ImageMismatch {
                    request_id: parked.request_id,
                    detail,
                },
                parked,
            ));
        }
        // The transfer itself. Asynchronous when the executor has that seam: the sequence's
        // host-side bookkeeping below is set straight away, and what waits for the copy is only
        // *stepping* it — `poll_transfers` clears the mark, or hands the allocations back and puts
        // the image on the tier again.
        let async_restore = exec.parks_async();
        // The image's state is **moved** into the executor on the asynchronous path, so what stays
        // here is the rest of the image; a failure gets the bytes back and puts them into this slot
        // before the group is re-parked. That move is what removes the 71.69 ms copy across the
        // engine channel (K9 measured it on a 403 MiB image).
        let mut stateless: Vec<SessionImage> = Vec::with_capacity(images.len());
        // Branches whose restore waits for a sibling's to free the one staging slot (ws-f38).
        let mut queued: Vec<bool> = Vec::with_capacity(images.len());
        for (branch, ((seq, admitted), mut image)) in
            seqs.iter().zip(admitted).zip(images).enumerate()
        {
            if async_restore && queued.last() == Some(&true) {
                // Once one branch queued, the rest queue behind it, image and state kept.
                self.restore_admitted.insert(seq.seq_id, admitted.clone());
                stateless.push(image);
                queued.push(true);
                continue;
            }
            if async_restore {
                let state = std::mem::replace(&mut image.state, SeqState(Vec::new()));
                match exec.restore_begin(admitted, state) {
                    RestoreStart::Started => {
                        stateless.push(image);
                        queued.push(false);
                    }
                    // A SIBLING's restore holds the slot (ws-f38): queue this one rather than
                    // re-parking a group whose first branch is already being written.
                    RestoreStart::Busy(state) if branch > 0 => {
                        image.state = state;
                        self.restore_admitted.insert(seq.seq_id, admitted.clone());
                        stateless.push(image);
                        queued.push(true);
                    }
                    RestoreStart::Busy(state) => {
                        // Not a failure: the executor's one staging slot is in use. The group goes
                        // back whole, images included, and `unpark` is retried next step.
                        //
                        // BACK ON THE TIER, under a fresh ticket (ws-k17): the image was taken off
                        // it above, so returning the group with its OLD ticket made every retry a
                        // `NoSuchImage` — the TP=2 card run looped on "park tier host.pinned has no
                        // image 1" until the client gave up, and held SIGTERM past its deadline.
                        image.state = state;
                        self.repark_branch(&mut parked, branch, image);
                        return Err((ParkError::TransferBusy(parked.request_id), parked));
                    }
                    RestoreStart::Blocking(state) | RestoreStart::Refused(_, Some(state)) => {
                        image.state = state;
                        self.repark_branch(&mut parked, branch, image);
                        return Err((ParkError::ExecutorCannotRestore(seq.seq_id), parked));
                    }
                    RestoreStart::Refused(what, None) => {
                        tracing::error!(seq = %seq.seq_id, error = %what, "a restore was refused and the image was not handed back; the session is lost");
                        return Err((ParkError::ExecutorCannotRestore(seq.seq_id), parked));
                    }
                }
            } else if exec.restore_state(admitted, &image.state) {
                stateless.push(image);
                queued.push(false);
            } else {
                self.repark_branch(&mut parked, branch, image);
                return Err((ParkError::ExecutorCannotRestore(seq.seq_id), parked));
            }
        }
        let images = stateless;

        for (seq, branch) in seqs.iter_mut().zip(parked.branches.iter()) {
            seq.tokens.clone_from(&branch.tokens);
            seq.generated = branch.generated;
            seq.step_index = branch.step_index;
            seq.pending_input.clone_from(&branch.pending_input);
            seq.ema = branch.ema;
            seq.choice_index = branch.choice_index;
            seq.params = parked.params.clone();
            seq.prompt_len = parked.prompt_len;
            seq.max_tokens = parked.max_tokens;
            // A restored session is not prefilling: its KV is whole again, prompt included.
            seq.first_step = false;
            seq.prefill_len = seq.prompt_len;
            seq.lazy = false;
            seq.prefilled = seq.prompt_len;
            // It has been resident for zero steps, which keeps the residency guard from evicting
            // it straight back out: a restore that is immediately re-parked is a thrash loop. The
            // park count is the GROUP's, so every branch carries the same one.
            seq.admitted_at_step = self.step_index;
            seq.last_active_step = self.step_index;
            seq.parks = parked.parks;
            seq.auto_parked = parked.auto;
            seq.restore_failures = parked.restore_failures;
            // Not steppable until the copy lands. On the blocking path it already has.
            seq.transfer = if async_restore {
                Some(Transfer::Restoring)
            } else {
                None
            };
        }
        for (seq, q) in seqs.iter_mut().zip(&queued) {
            if *q {
                seq.transfer = Some(Transfer::RestoreQueued);
            }
        }
        if async_restore {
            // The images are kept so that a failure after this point can put them back: the tickets
            // are already consumed, so without them the session would be lost rather than parked.
            for (seq, image) in seqs.iter().zip(images) {
                self.restoring_images.insert(seq.seq_id, image);
            }
            self.counters.restores_started += 1;
        } else {
            self.counters.restores += 1;
        }
        Ok(())
    }

    /// Advance every in-flight park and restore by one non-blocking poll.
    ///
    /// Runs at the top of `prepare`, which holds both locks with no step in flight, and does no
    /// waiting at all: a transfer that is not finished simply stays in flight and its sequence stays
    /// out of the step. That is the entire mechanism by which a 101 ms park costs the other
    /// sequences nothing but the bandwidth it takes.
    ///
    /// A park's allocations are freed **here**, when its copy is known to be done — never when it was
    /// issued. The byte ledger therefore shows a parked session's bytes as free exactly when they
    /// are free, which is the property `park_under_pressure` depends on to admit the sequence it
    /// evicted for.
    fn poll_transfers<E: StepExecutor>(&mut self, exec: &mut E) {
        if self.running.iter().all(Seq::steppable) {
            return;
        }
        self.counters.steps_during_transfer += 1;

        // ---- parks ---------------------------------------------------------------------------
        let parking: Vec<SeqId> = self
            .running
            .iter()
            .filter(|s| s.transfer == Some(Transfer::Parking))
            .map(|s| s.seq_id)
            .collect();
        let mut failed: Vec<SeqId> = Vec::new();
        for seq_id in parking {
            if self.parking_states.contains_key(&seq_id) {
                continue;
            }
            match exec.park_poll(seq_id) {
                ParkPoll::Pending => {}
                ParkPoll::Done(state) => {
                    self.parking_states.insert(seq_id, state);
                }
                ParkPoll::Failed(e) => {
                    self.counters.park_transfer_failures += 1;
                    tracing::error!(seq = %seq_id, error = %e, "a park in flight failed; the sequence keeps its state and goes back to running");
                    failed.push(seq_id);
                }
            }
        }
        // Branches queued behind a sibling's park (ws-f38), started in running order as the
        // staging slot frees — one poll may finish one copy and start the next.
        let queued: Vec<SeqId> = self
            .running
            .iter()
            .filter(|s| s.transfer == Some(Transfer::ParkQueued))
            .map(|s| s.seq_id)
            .collect();
        for seq_id in queued {
            match exec.park_begin(seq_id) {
                TransferStart::Started => {
                    if let Some(s) = self.running.iter_mut().find(|s| s.seq_id == seq_id) {
                        s.transfer = Some(Transfer::Parking);
                    }
                }
                TransferStart::Busy => break,
                other => {
                    self.counters.park_transfer_failures += 1;
                    tracing::error!(seq = %seq_id, answer = ?other, "a queued branch's park could not start; its group goes back to running");
                    failed.push(seq_id);
                }
            }
        }
        // A failed branch puts its whole group back: a group parks whole or not at all.
        let failed_groups: Vec<RequestId> = self
            .running
            .iter()
            .filter(|s| failed.contains(&s.seq_id))
            .map(|s| s.request_id)
            .collect();
        for id in failed_groups {
            self.auto_parks_in_flight.remove(&id);
            for s in self.running.iter_mut().filter(|s| s.request_id == id) {
                s.transfer = None;
                self.parking_states.remove(&s.seq_id);
            }
            self.counters.park_refusals += 1;
        }

        // Groups whose every parking branch now has a state: complete the park.
        let ready: Vec<RequestId> = {
            let mut ids: Vec<RequestId> = self
                .running
                .iter()
                .filter(|s| s.transfer == Some(Transfer::Parking))
                .map(|s| s.request_id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids.retain(|id| {
                self.running
                    .iter()
                    .filter(|s| s.request_id == *id && s.transfer == Some(Transfer::Parking))
                    .all(|s| self.parking_states.contains_key(&s.seq_id))
                    && !self
                        .running
                        .iter()
                        .any(|s| s.request_id == *id && s.transfer == Some(Transfer::ParkQueued))
            });
            ids
        };
        for id in ready {
            self.finish_park(exec, id);
        }

        // ---- restores ------------------------------------------------------------------------
        let restoring: Vec<SeqId> = self
            .running
            .iter()
            .filter(|s| s.transfer == Some(Transfer::Restoring))
            .map(|s| s.seq_id)
            .collect();
        for seq_id in restoring {
            match exec.restore_poll(seq_id) {
                RestorePoll::Pending => {}
                RestorePoll::Done => {
                    if let Some(s) = self.running.iter_mut().find(|s| s.seq_id == seq_id) {
                        s.transfer = None;
                    }
                    self.restoring_images.remove(&seq_id);
                    self.counters.restores += 1;
                }
                RestorePoll::Failed(e, state) => {
                    self.counters.restore_failures += 1;
                    tracing::error!(seq = %seq_id, error = %e, "a restore in flight failed; the allocations go back and the session stays parked");
                    // The bytes come back with the failure, because the executor owned them. Put
                    // them into the image that was kept for exactly this.
                    if let (Some(state), Some(image)) =
                        (state, self.restoring_images.get_mut(&seq_id))
                    {
                        image.state = state;
                    }
                    self.abandon_restore(exec, seq_id);
                }
            }
        }
        // Restores queued behind a sibling's (ws-f38), started as the slot frees.
        let queued: Vec<SeqId> = self
            .running
            .iter()
            .filter(|s| s.transfer == Some(Transfer::RestoreQueued))
            .map(|s| s.seq_id)
            .collect();
        for seq_id in queued {
            let (Some(admitted), Some(image)) = (
                self.restore_admitted.get(&seq_id).cloned(),
                self.restoring_images.get_mut(&seq_id),
            ) else {
                tracing::error!(seq = %seq_id, "a queued restore has no admission or image kept");
                self.abandon_restore(exec, seq_id);
                continue;
            };
            let state = std::mem::replace(&mut image.state, SeqState(Vec::new()));
            match exec.restore_begin(&admitted, state) {
                RestoreStart::Started => {
                    self.restore_admitted.remove(&seq_id);
                    if let Some(s) = self.running.iter_mut().find(|s| s.seq_id == seq_id) {
                        s.transfer = Some(Transfer::Restoring);
                    }
                }
                RestoreStart::Busy(state) => {
                    if let Some(image) = self.restoring_images.get_mut(&seq_id) {
                        image.state = state;
                    }
                    break;
                }
                RestoreStart::Blocking(state) | RestoreStart::Refused(_, Some(state)) => {
                    if let Some(image) = self.restoring_images.get_mut(&seq_id) {
                        image.state = state;
                    }
                    self.counters.restore_failures += 1;
                    tracing::error!(seq = %seq_id, "a queued restore was refused; the image goes back on the tier");
                    self.abandon_restore(exec, seq_id);
                }
                RestoreStart::Refused(what, None) => {
                    self.counters.restore_failures += 1;
                    tracing::error!(seq = %seq_id, error = %what, "a queued restore was refused and its image was not handed back");
                    self.abandon_restore(exec, seq_id);
                }
            }
        }
    }

    /// Put the images of a group whose restore failed back on the tier, and hand its allocations
    /// back. The session stays parked, which is the same contract the blocking path has.
    fn abandon_restore<E: StepExecutor>(&mut self, exec: &mut E, seq_id: SeqId) {
        let Some(pos) = self.running.iter().position(|s| s.seq_id == seq_id) else {
            return;
        };
        let seq = self.running.remove(pos);
        exec.release(seq.seq_id);
        self.arenas.free(seq.kv, seq.gdn);
        self.restore_admitted.remove(&seq_id);
        let request_id = seq.request_id;
        let image = self.restoring_images.remove(&seq_id);
        let lost = match &image {
            None => true,
            Some(image) => image.state.0.is_empty(),
        };
        if lost {
            // THE REQUEST IS FINISHED, and now it says so (ws-k18): this used to log "the request is
            // finished" and emit nothing, so the client waited out its own timeout.
            tracing::error!(seq = %seq_id, "a failed restore returned no image; the session cannot be re-parked and the request is finished");
            emit_error(
                &seq.events,
                request_id,
                seq.choice_index,
                seq.step_index,
                "the session was parked under memory pressure and its image was lost in a failed \
                 restore",
            );
            self.counters.restores_given_up += 1;
            return;
        }
        if let (Some(image), Some(tier)) = (
            image.filter(|i| !i.state.0.is_empty()),
            self.park_tier.as_mut(),
        ) {
            let parked_as = image.seq_id;
            match tier.park(image, &mut self.ledger) {
                Ok(ticket) => {
                    // Back onto the parked list under a fresh ticket.
                    if let Some(p) = self
                        .parked
                        .iter_mut()
                        .find(|p| p.request_id == seq.request_id)
                    {
                        p.branches.push(ParkedBranch {
                            choice_index: seq.choice_index,
                            tokens: seq.tokens,
                            generated: seq.generated,
                            step_index: seq.step_index,
                            pending_input: seq.pending_input,
                            ema: seq.ema,
                            ticket,
                            parked_as,
                        });
                        p.branches.sort_by_key(|b| b.choice_index);
                    } else {
                        self.parked.push(ParkedSeq {
                            request_id: seq.request_id,
                            params: seq.params,
                            prompt_len: seq.prompt_len,
                            max_tokens: seq.max_tokens,
                            events: seq.events,
                            branches: vec![ParkedBranch {
                                choice_index: seq.choice_index,
                                tokens: seq.tokens,
                                generated: seq.generated,
                                step_index: seq.step_index,
                                pending_input: seq.pending_input,
                                ema: seq.ema,
                                ticket,
                                parked_as,
                            }],
                            // The POLICY's still, when it was (ws-k18): this was `false`, so a
                            // policy-parked session whose restore failed was never asked for again.
                            auto: seq.auto_parked,
                            parks: seq.parks,
                            restore_failures: seq.restore_failures + 1,
                        });
                    }
                }
                Err(e) => {
                    tracing::error!(seq = %seq_id, error = %e, "a failed restore's image could not be put back; the session is lost and the request is finished");
                    emit_error(
                        &seq.events,
                        request_id,
                        seq.choice_index,
                        seq.step_index,
                        &format!(
                            "the session was parked under memory pressure and its image could not \
                             go back on the tier after a failed restore: {e}"
                        ),
                    );
                    self.counters.restores_given_up += 1;
                    return;
                }
            }
        }
        // The whole group is back on the parked list once no branch of it is still restoring:
        // retry it or give up on it, by the bound.
        if !self.running.iter().any(|s| s.request_id == request_id) {
            if let Some(pos) = self.parked.iter().position(|p| p.request_id == request_id) {
                let parked = self.parked.remove(pos);
                self.after_restore_failure(parked, "the restore in flight failed");
            }
        }
    }

    /// Parks a whole **group**: every branch's state image and the KV pages they share, as one
    /// unit.
    ///
    /// All or nothing in both directions. The branches share the prompt's pages, so parking some
    /// of them would leave a prefix pinned by the ones still running and a tier holding images
    /// that cannot be restored on their own; and a tier that refuses halfway puts back what it
    /// took and leaves the group running, untouched.
    fn park<E: StepExecutor>(&mut self, exec: &mut E, id: RequestId) -> Result<(), ParkError> {
        let positions: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.request_id == id)
            .map(|(i, _)| i)
            .collect();
        if positions.is_empty() {
            self.counters.park_refusals += 1;
            return Err(ParkError::NotRunning(id));
        }
        if self.park_tier.is_none() {
            self.counters.park_refusals += 1;
            return Err(ParkError::NoTier);
        }
        // A group whose prompt is not all in: see `ParkError::StillPrefilling` (ws-k17).
        if positions.iter().any(|&p| self.running[p].prefilling()) {
            self.counters.park_refusals += 1;
            return Err(ParkError::StillPrefilling(id));
        }
        // A transfer in flight, or a branch that has not been forked yet: in both cases part of
        // this group's device state is not where a park would read it. A group parks whole or not
        // at all, so this is a refusal rather than a partial image.
        if positions
            .iter()
            .any(|&p| self.running[p].transfer.is_some() || self.running[p].pending_fork.is_some())
        {
            self.counters.park_refusals += 1;
            return Err(ParkError::NotRunning(id));
        }

        // ---- the asynchronous path -----------------------------------------------------------
        // Decided from the executor, once, before anything is issued: a group must not end up with
        // some branches copying and some not, and `parks_async` is the executor stating which of its
        // two seams it implements rather than the scheduler guessing per sequence.
        if exec.parks_async() {
            for (i, &pos) in positions.iter().enumerate() {
                let seq_id = self.running[pos].seq_id;
                match exec.park_begin(seq_id) {
                    TransferStart::Started => {
                        self.running[pos].transfer = Some(Transfer::Parking);
                    }
                    // A SIBLING holds the one staging slot (ws-f38): the group's park has started,
                    // so the rest of it is QUEUED rather than refused. Refusing here left the
                    // first branch's copy in flight on a group marked running again, and made a
                    // group of `n > 1` unparkable on an engine with one staging slot — which is
                    // every serving engine.
                    TransferStart::Busy if i > 0 => {
                        for &rest in &positions[i..] {
                            self.running[rest].transfer = Some(Transfer::ParkQueued);
                        }
                        tracing::debug!(request = %id, queued = positions.len() - i, "a group's park queued its remaining branches behind the first");
                        break;
                    }
                    TransferStart::Busy => {
                        // One staging buffer, one transfer. Ask again next step; the group is
                        // exactly as it was, which is what makes this safe to retry.
                        self.unmark(&positions);
                        self.counters.park_transfer_busy += 1;
                        return Err(ParkError::TransferBusy(id));
                    }
                    TransferStart::CannotPark => {
                        self.unmark(&positions);
                        self.counters.park_refusals += 1;
                        return Err(ParkError::ExecutorCannotPark(seq_id));
                    }
                    TransferStart::Blocking | TransferStart::Failed(_) => {
                        // `parks_async` said otherwise, or the issue failed outright. Either way
                        // nothing is in flight for THIS branch; the ones before it are, so they are
                        // polled to completion by `poll_transfers` and the group is left running.
                        self.unmark(&positions);
                        self.counters.park_refusals += 1;
                        return Err(ParkError::ExecutorCannotPark(seq_id));
                    }
                }
            }
            self.counters.parks_started += 1;
            tracing::debug!(request = %id, branches = positions.len(), "park issued off the step loop");
            return Ok(());
        }

        // ---- the blocking path, unchanged ------------------------------------------------------
        // Every image is taken out of the executor before anything leaves the running set: a
        // branch that cannot be copied means the group does not move at all.
        let mut states = Vec::with_capacity(positions.len());
        for &pos in &positions {
            let seq_id = self.running[pos].seq_id;
            let Some(state) = exec.park_state(seq_id) else {
                self.counters.park_refusals += 1;
                return Err(ParkError::ExecutorCannotPark(seq_id));
            };
            states.push(state);
        }
        self.commit_park(exec, id, &positions, states)
            .inspect_err(|_| {
                self.counters.park_refusals += 1;
            })
    }

    /// Complete a park whose every branch has finished copying.
    fn finish_park<E: StepExecutor>(&mut self, exec: &mut E, id: RequestId) {
        let positions: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.request_id == id && s.transfer == Some(Transfer::Parking))
            .map(|(i, _)| i)
            .collect();
        let mut states = Vec::with_capacity(positions.len());
        for &pos in &positions {
            let seq_id = self.running[pos].seq_id;
            match self.parking_states.remove(&seq_id) {
                Some(state) => states.push(state),
                None => {
                    // Cannot happen: this group was selected because every branch had one. Fail
                    // loud and leave the group running rather than parking half of it.
                    tracing::error!(seq = %seq_id, "a park completed with no image; the group stays running");
                    self.auto_parks_in_flight.remove(&id);
                    self.unmark(&positions);
                    self.counters.park_refusals += 1;
                    return;
                }
            }
        }
        // The marks come off here: `commit_park` releases and frees, and a sequence that is about to
        // leave the running set does not need one.
        self.unmark(&positions);
        let by_policy = self.auto_parks_in_flight.remove(&id);
        match self.commit_park(exec, id, &positions, states) {
            Ok(()) => {
                // `commit_park` pushed this group last. A policy park is marked HERE, where it
                // lands, so `requeue_auto_parked` brings it back when space frees (ws-k17).
                if by_policy {
                    self.counters.policy_parks += 1;
                    if let Some(p) = self.parked.last_mut() {
                        debug_assert_eq!(p.request_id, id);
                        p.auto = true;
                    }
                }
            }
            Err(e) => {
                self.counters.park_refusals += 1;
                tracing::error!(request = %id, error = %e, "a finished park could not be stored; the group stays running");
            }
        }
    }

    /// Clear the parking marks on a group whose park could not be started.
    fn unmark(&mut self, positions: &[usize]) {
        for &pos in positions {
            self.running[pos].transfer = None;
        }
    }

    /// The tail of a park: every branch's image is in hand, so put them on the tier, free the
    /// allocations and move the group to the parked list.
    ///
    /// Shared by both paths, which is the point — the asynchronous one differs only in *when* it
    /// gets here, and the "all or nothing, and the bytes are free exactly when they are free" rule
    /// is written once.
    fn commit_park<E: StepExecutor>(
        &mut self,
        exec: &mut E,
        id: RequestId,
        positions: &[usize],
        states: Vec<SeqState>,
    ) -> Result<(), ParkError> {
        let positions: Vec<usize> = positions.to_vec();
        let mut tickets = Vec::with_capacity(positions.len());
        for (&pos, state) in positions.iter().zip(states) {
            let seq = &self.running[pos];
            let resident_bytes = seq.kv.bytes(&self.config.kv)
                + u64::from(seq.gdn.slots) * self.config.gdn_bytes_per_slot;
            let image = SessionImage {
                request_id: seq.request_id,
                seq_id: seq.seq_id,
                tokens: seq.tokens.clone(),
                kv_blocks: seq.kv.blocks.len() as u32,
                gdn_slots: seq.gdn.slots,
                resident_bytes,
                state,
            };
            let tier = self.park_tier.as_mut().expect("checked above");
            match tier.park(image, &mut self.ledger) {
                Ok(ticket) => tickets.push(ticket),
                Err(e) => {
                    // Refused: hand back the branches already on the tier and leave the group
                    // exactly as it was.
                    for ticket in &tickets {
                        let tier = self.park_tier.as_mut().expect("checked above");
                        tier.discard(ticket, &mut self.ledger);
                    }
                    // Counted by the caller: `park` for the blocking path, `poll_transfers` for the
                    // asynchronous one. Counting here as well would double every refusal.
                    return Err(e);
                }
            }
        }

        // Nothing can fail from here: take the branches out, newest index first so the earlier
        // positions stay valid.
        let mut branches = Vec::with_capacity(positions.len());
        let mut params = None;
        let mut prompt_len = 0;
        let mut max_tokens = 0;
        let mut events = None;
        let mut parks = 0;
        for (&pos, ticket) in positions.iter().zip(tickets).rev() {
            let seq = self.running.remove(pos);
            // KV and state leave together, and only once the image is safely on the tier. No step
            // can be in flight here: the caller holds the executor lock it took without blocking.
            // `release_parked`, not `release`: the image outlives the sequence (ws-k17).
            exec.release_parked(seq.seq_id);
            self.arenas.free(seq.kv, seq.gdn);
            prompt_len = seq.prompt_len;
            max_tokens = seq.max_tokens;
            parks = seq.parks;
            params = Some(seq.params);
            events = Some(seq.events);
            branches.push(ParkedBranch {
                choice_index: seq.choice_index,
                tokens: seq.tokens,
                generated: seq.generated,
                step_index: seq.step_index,
                pending_input: seq.pending_input,
                ema: seq.ema,
                ticket,
                parked_as: seq.seq_id,
            });
        }
        branches.sort_by_key(|b| b.choice_index);
        self.counters.parks += 1;
        if parks > 0 {
            self.counters.reparks += 1;
        }
        self.parked.push(ParkedSeq {
            request_id: id,
            params: params.expect("a group has at least one branch"),
            prompt_len,
            max_tokens,
            events: events.expect("a group has at least one branch"),
            branches,
            // An explicit park is not the policy's; `park_under_pressure` flips this on the group
            // it evicts, which is the only place an automatic park is decided.
            auto: false,
            // The park count is the GROUP's: its branches were evicted together, and counting
            // each one would read `n` times the number the policy's per-session cap compares.
            parks: parks + 1,
            restore_failures: 0,
        });
        Ok(())
    }

    fn unpark(&mut self, id: RequestId) -> Result<(), ParkError> {
        let Some(pos) = self.parked.iter().position(|p| p.request_id == id) else {
            return Err(ParkError::NotParked(id));
        };
        let parked = self.parked.remove(pos);
        let n = parked.branches.len() as u32;
        // The whole group goes back in one queue entry, so admission is all-or-nothing on the way
        // in as it was on the way out. Branch 0's token history is the prompt the group resumes
        // from; each branch's own history is restored from its image.
        let prompt = parked
            .branches
            .first()
            .map(|b| b.tokens.clone())
            .unwrap_or_default();
        self.waiting.push_front(Waiting {
            request_id: parked.request_id,
            params: parked.params.clone(),
            prompt,
            prompt_len: parked.prompt_len,
            max_tokens: parked.max_tokens,
            n,
            events: parked.events.clone(),
            resume: Some(Box::new(parked)),
            images: None,
            arrived: None,
            recompute: None,
        });
        Ok(())
    }

    /// Phase 1: admit, choose the width and the bucket, and build the batch.
    ///
    /// This is also where a park **decision** is made: admission is the only thing that knows the
    /// engine is out of KV blocks or state slot sets, and both locks are held here, so evicting a
    /// session copies its state out through the executor with no step in flight.
    fn prepare<E: StepExecutor>(&mut self, exec: &mut E) -> Prepared {
        // The eviction budget is per step, and `prepare` runs exactly once per step.
        self.parks_this_step = 0;
        self.admit_arrivals.clear();
        // In-flight transfers first: a park that has finished frees its pages and its slot, which is
        // space the admission below can use THIS step rather than next.
        self.poll_transfers(exec);
        // LR-G: the running sequences' growth goes before any admission
        self.grow_lazy(exec);
        let admitted = self.admit_waiting(exec);
        if self.running.is_empty() {
            self.counters.idle_steps += 1;
            return Prepared::Idle;
        }

        // **C4: the mixed step.** With `MixedPrefill` on, a step that has decoding sequences runs
        // them, and ALSO carries the queue head's next prompt chunk when the token budget leaves
        // room — prod's continuous batching (running requests first, then the waiting prompt in
        // what is left of `max_num_batched_tokens`). The chunk is a complete prefill step of its
        // own inside the decode step (`ReadyStep::chunk`), so its bucket, its chunk law and its
        // commit are exactly the unmixed ones. With nothing decoding there is nobody to stall,
        // and the chunk runs alone below.
        if self.config.mixed.enabled {
            match self.prepare_decode(admitted) {
                Some(Prepared::Ready(mut decode)) => {
                    let owes = self.running.iter().any(|s| s.prefilling() && s.steppable());
                    // The serial form's room is prod's law; the pre-C11 control caps it by the
                    // widest rung (C9's cut, a chunk-boundary move).
                    let room = self
                        .config
                        .mixed
                        .room_for(decode.bucket.m(), self.config.prefill.max_chunk());
                    if owes && self.config.mixed.uses_pack_law() {
                        // C10/C11: the packing law, for EVERY fused form — every queued prompt's
                        // unpacked chunk that fits whole, in arrival order, in class
                        // (`mixed_law::pack`; `fused` is the law at one chunk). A chunk is never
                        // cut to fit: a long prompt's 2048-row chunk rides whole and the executor
                        // runs that step in the serial form.
                        match self.prepare_packed(decode.bucket.m()) {
                            Some((first, more)) => {
                                self.counters.mixed_steps += 1;
                                for c in std::iter::once(&first).chain(more.iter()) {
                                    self.counters.mixed_chunk_rows += c
                                        .slots
                                        .first()
                                        .and_then(|s| s.chunk)
                                        .map_or(0, |c| u64::from(c.prompt_tokens));
                                }
                                decode.chunk = Some(Box::new(first));
                                decode.more = more;
                            }
                            None => self.counters.mixed_budget_spent += 1,
                        }
                    } else if owes && room == 0 {
                        // The decode half spent the budget: the chunk waits for the next step.
                        self.counters.mixed_budget_spent += 1;
                    } else if owes {
                        // `None` here is the no-bucket fallback, which `prepare_prefill` counted
                        // and resolved (the prompt goes in by a decode-shaped step): the decode
                        // half runs unmixed.
                        let room = room.min(self.config.prefill.room(true));
                        if let Some(Prepared::Ready(chunk)) = self.prepare_prefill(0, room) {
                            self.counters.mixed_steps += 1;
                            self.counters.mixed_chunk_rows += chunk
                                .slots
                                .first()
                                .and_then(|s| s.chunk)
                                .map_or(0, |c| u64::from(c.prompt_tokens));
                            decode.chunk = Some(Box::new(chunk));
                        }
                    }
                    return Prepared::Ready(decode);
                }
                Some(other) => return other,
                None => {}
            }
        }

        // **Prefill first, and alone** (the unmixed schedule, and the mixed one's step when
        // nothing is decoding). A prefill chunk and a decode step run different ops and budget
        // GDN state differently, so without `MixedPrefill` a step is one phase: when something
        // still owes prefill, this step is its chunk and the decoding sequences wait one step —
        // counted as `stalled_decoder_steps` / `stalled_decoder_us`, C4's stall.
        // FN38: a chunk beside decoding sequences keeps the busy cap (`PrefillChunking::busy_max`)
        let decoding = self
            .running
            .iter()
            .any(|s| !s.prefilling() && s.steppable());
        // LR-B (`PrefillChunking::yield_steps`, off by default): after a chunk ran beside ready
        // decoders, they get this many decode steps before the next chunk, so a decoder waits one
        // chunk rather than the whole prompt. Same cut, same unmixed steps.
        let owes = self.running.iter().any(|s| s.prefilling() && s.steppable());
        if !owes {
            // nothing would wait: the next step is a decode step anyway, not a yield
            self.yield_owed = 0;
        }
        if decoding && self.yield_owed > 0 {
            self.yield_owed -= 1;
            if let Some(prepared) = self.prepare_decode(admitted) {
                self.counters.yield_decode_steps += 1;
                return prepared;
            }
        }
        if let Some(prepared) = self.prepare_prefill(admitted, self.config.prefill.room(decoding)) {
            self.yield_owed = match &prepared {
                Prepared::Ready(r) if decoding && r.phase == Phase::Prefill => {
                    self.config.prefill.yield_steps
                }
                _ => 0,
            };
            return prepared;
        }
        self.yield_owed = 0;
        self.prepare_decode(admitted).unwrap_or_else(|| {
            // Everything running is still prefilling and `prepare_prefill` declined (no captured
            // prefill bucket fits). Nothing to do this step rather than a decode step over rows
            // that have no KV.
            self.counters.idle_steps += 1;
            Prepared::Idle
        })
    }

    /// The decode half of `prepare`: every decoding sequence, in its bucket. `None` when nothing
    /// is decoding.
    fn prepare_decode(&mut self, admitted: u32) -> Option<Prepared> {
        // Decode. Only sequences whose prompt is in take part: a prefilling one holds a slot set
        // and a KV allocation, but it is not a row of this step's shape. Neither is one whose state
        // is in the middle of a transfer.
        let decoding: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.prefilling() && s.steppable())
            .map(|(i, _)| i)
            .collect();
        if decoding.is_empty() {
            return None;
        }
        let running = decoding.len() as u32;

        // RAGGED PER-SLOT WIDTHS (C2), the default whenever the plan captured row bands: every
        // slot at its own width, the step in the smallest band that holds their sum.
        // `--rectangular-widths` turns it off and everything below is the pairing control.
        if self.config.width.ragged.is_some()
            && self
                .config
                .buckets
                .iter()
                .any(|b| b.captured && b.is_band())
        {
            return Some(self.prepare_ragged(&decoding, running, admitted));
        }

        // Width: a sequence on its first step drafts nothing, so it does not widen the bucket.
        // A step where *every* sequence is prefilling verifies nothing at all, and takes the
        // narrowest captured width rather than paying for state slots it will not touch.
        let width = {
            // A SEQUENCE ON ITS FIRST STEP IS NOT PART OF THIS DECISION, and the filter is the
            // whole of that sentence. Its `verify_width` is 0 — it drafts nothing and uses one
            // row of the bucket — so asking what width its (absent) acceptance EMA wants answers
            // the wrong question, and because a cold EMA asks for the ceiling it answers it in
            // the most expensive direction: ONE arriving request would pull a batch of eight
            // settled ones up to the widest captured shape, and at conc-8 over a 48-request
            // corpus a request arrives every time one finishes. prod has no such coupling —
            // `_ref_cap_spec_width` returns early on a request with no `spec_token_ids` and
            // that request contributes ONE row to a ragged `M` — so this filter is what makes
            // the rectangular bucket agree with it (ws-t1).
            let emas: Vec<&AcceptanceEma> = decoding
                .iter()
                .filter(|i| !self.running[**i].first_step)
                .map(|i| &self.running[*i].ema)
                .collect();
            let chosen = if emas.is_empty() {
                // Every sequence is on its first step: nothing drafts, so take the narrowest
                // captured shape rather than paying for state slots no row will touch.
                self.config.width.floor_width()
            } else {
                self.config.width.step_width(running, emas)
            };
            // THE FLOOR (ws-f38): no narrower than what any slot committed last step, because
            // `naccept - 1` indexes this step's own `sidx` block. See `width_floor_raised`.
            let floor = decoding
                .iter()
                .filter(|i| !self.running[**i].first_step)
                .map(|i| self.running[*i].pending_input.len() as u32)
                .max()
                .unwrap_or(1);
            if floor > chosen {
                self.counters.width_floor_raised += 1;
                self.config.width.bucket_width(floor)
            } else {
                chosen
            }
        };

        let ctx_tokens = decoding
            .iter()
            .map(|i| self.running[*i].tokens.len() as u32 + width + 1)
            .max()
            .unwrap_or(1);
        let bucket = match select(
            &self.config.buckets,
            BucketRequest {
                running,
                width,
                ctx_tokens,
            },
        ) {
            Ok(b) => b,
            Err(e) => {
                self.counters.no_bucket_steps += 1;
                tracing::error!(error = %e, "no captured bucket for this step; nothing executed");
                return Some(Prepared::NoBucket(e));
            }
        };

        // The acceptance EMA's cap picked the BUCKET above. Inside it every slot verifies the
        // bucket's width, because the bucket launches `batch × spec_width` rows and the same
        // number of GDN state slots whatever a slot asks for: a slot that verified fewer would
        // pay for the rows anyway and could only accept fewer tokens
        // (`VerifyWidthPolicy::slot_in_bucket` states the law and why prod's is slot-level).
        // The narrowing the rectangular shape absorbed is counted, not hidden.
        let mut absorbed = 0u64;
        let slots: Vec<BatchSlot> = decoding
            .iter()
            .enumerate()
            .map(|(i, seq_index)| {
                let s = &self.running[*seq_index];
                let verify_width = if s.first_step {
                    0
                } else {
                    let w = self
                        .config
                        .width
                        .slot_in_bucket(&s.ema, running, bucket.spec_width);
                    absorbed += u64::from(w.absorbed());
                    w.verify
                };
                decode_batch_slot(s, i as u32, verify_width)
            })
            .collect();
        self.counters.slot_caps_absorbed += absorbed;
        if bucket.spec_width < self.config.width.ceiling_width() {
            self.counters.width_narrowed_steps += 1;
        }

        Some(Prepared::Ready(ReadyStep {
            phase: Phase::Decode,
            bucket,
            slots,
            running,
            width,
            admitted,
            step_index: self.step_index,
            chunk: None,
            more: Vec::new(),
            waiting_decoders: 0,
        }))
    }

    /// **The RAGGED decode step (C2)**: every running slot at its OWN verify width, the step in
    /// the smallest captured ROW BAND that holds `Σ w_i`.
    ///
    /// The per-slot rule is prod's (`VerifyWidthPolicy::ragged_slot`): the acceptance-EMA cap in
    /// draft-token space under a ceiling (the batch table's cell by default, the plan's max width
    /// under `--ragged-ceiling block`), the unverified tail dropped, a first-step slot one row.
    /// Widths are EXACT — a band captures any multiset — so nothing is rounded up to a captured
    /// width and `slot_caps_absorbed` reads 0 here.
    ///
    /// **No floor** (ws-f38's `width_floor_raised` is the rectangle's): a band's `sidx` block is
    /// the MAX width for every live slot with all of its entries filled
    /// (`saber_exec::forward::StepIndex::band_width`), and both GDN decode kernels resume from
    /// `sidx[n * stride + naccept - 1]` and read the conv window at `naccept - 1` — so a slot
    /// that committed 8 last step and verifies 2 now resumes from the right snapshot
    /// (`forward::tests::the_band_resumes_every_sequence_from_what_it_committed`).
    ///
    /// **Uniform widths take the RECTANGLE** when that exact (batch, width) is captured: every
    /// slot asking one width is exactly the rectangular step (below `MIN_BATCH` the cap never
    /// binds, so single-stream and conc-2 always are), and a band would run the same rows at
    /// batch `max_seqs` — the GDN pair route instead of the fused one and a batch-8 attention
    /// grid for one sequence. Counted (`ragged_rect_steps`).
    fn prepare_ragged(&mut self, decoding: &[usize], running: u32, admitted: u32) -> Prepared {
        let widths: Vec<RaggedSlot> = decoding
            .iter()
            .map(|i| {
                let s = &self.running[*i];
                self.config.width.ragged_slot(&s.ema, running, s.first_step)
            })
            .collect();
        let rows: u32 = widths.iter().map(|w| w.rows).sum();
        let ctx_tokens = decoding
            .iter()
            .zip(&widths)
            .map(|(i, w)| self.running[*i].tokens.len() as u32 + w.rows + 1)
            .max()
            .unwrap_or(1);
        let first = widths.first().map_or(1, |w| w.rows);
        let uniform = widths.iter().all(|w| w.rows == first);
        let rect = if uniform {
            select(
                &self.config.buckets,
                BucketRequest {
                    running,
                    width: first,
                    ctx_tokens,
                },
            )
            .ok()
            .filter(|b| b.spec_width == first)
        } else {
            None
        };
        let bucket = match rect {
            Some(b) => b,
            None => match select_band(
                &self.config.buckets,
                BandRequest {
                    rows,
                    running,
                    ctx_tokens,
                },
            ) {
                Ok(b) => b,
                Err(e) => {
                    self.counters.no_bucket_steps += 1;
                    tracing::error!(
                        error = %e,
                        rows,
                        "no captured row band for this ragged step; nothing executed"
                    );
                    return Prepared::NoBucket(e);
                }
            },
        };
        let slots: Vec<BatchSlot> = decoding
            .iter()
            .zip(&widths)
            .enumerate()
            .map(|(i, (seq_index, w))| {
                decode_batch_slot(&self.running[*seq_index], i as u32, w.rows)
            })
            .collect();
        self.counters.ragged_steps += 1;
        self.counters.ragged_rect_steps += u64::from(rect.is_some());
        self.counters.ragged_live_rows += u64::from(rows);
        self.counters.ragged_step_rows += u64::from(bucket.m());
        let narrowed = widths.iter().filter(|w| w.narrowed).count() as u64;
        self.counters.ragged_slots_narrowed += narrowed;
        if narrowed > 0 {
            self.counters.width_narrowed_steps += 1;
        }
        Prepared::Ready(ReadyStep {
            phase: Phase::Decode,
            bucket,
            slots,
            running,
            width: widths.iter().map(|w| w.rows).max().unwrap_or(1),
            admitted,
            step_index: self.step_index,
            chunk: None,
            more: Vec::new(),
            waiting_decoders: 0,
        })
    }
}

/// One running sequence as a DECODE step's batch slot, verifying `verify_width` (rows in a
/// ragged step, the bucket's width in a rectangle, 0 for a first step there).
fn decode_batch_slot(s: &Seq, slot: u32, verify_width: u32) -> BatchSlot {
    BatchSlot {
        request_id: s.request_id,
        seq_id: s.seq_id,
        slot,
        chunk: None,
        // A block-table clone per sequence per step. Host-side only, and the P3
        // executor will take a view instead; the host chain has ~8.5 ms of slack at
        // conc-8 and this is nowhere near it.
        kv_blocks: s.kv.blocks.clone(),
        gdn_set: s.gdn.set,
        verify_width,
        input_tokens: s.pending_input.clone(),
        // **A DECODE STEP FEEDS ONE OF `pending_input`, NOT ALL OF IT**, so the
        // tokens already resident are everything but the last committed one.
        //
        // `pending_input` carries the whole of the previous step's commit because its
        // LENGTH is `naccept` — which of that step's recurrent snapshots this one
        // resumes from — while row 0 feeds only its last entry and rows `1..w` feed
        // the draft. `tokens.len() - pending_input.len()` is the same number only
        // when a step committed exactly ONE token, which every step in this tree did
        // for as long as every drafter in it was a mock the model rejected. The first
        // run with a real proposal the model agreed with put the engine one position
        // ahead of the scheduler on the very step after a two-token commit, and the
        // engine refused by name ("the batch says 236 token(s) are resident and the
        // engine has 237") rather than writing K/V over a position it had already
        // written (ws-s4).
        //
        // `first_step` keeps the old arithmetic, and it is not the same case: there
        // `pending_input` is the whole PROMPT and nothing is resident yet.
        context_len: if s.first_step {
            (s.tokens.len() - s.pending_input.len()) as u32
        } else {
            s.tokens.len().saturating_sub(1) as u32
        },
        first_step: s.first_step,
        sampling: s.sampling(),
    }
}

impl Core {
    /// The prefill half of `prepare`: one sequence's next chunk, in a prefill bucket.
    ///
    /// Returns `None` when nothing is prefilling — and also when something is but the plan
    /// captured no prefill bucket that fits it, in which case the sequence falls through to the
    /// decode path exactly as it did before P2 (one step, whole prompt), because refusing to
    /// serve a prompt over a missing bucket would be a worse answer than serving it the old way.
    ///
    /// `room` caps the chunk (C4's token budget: what a mixed step's decode half left of
    /// `max_batched_tokens`); `u32::MAX` is the unmixed step's "no cap". A `room` of 0 is `None`.
    fn prepare_prefill(&mut self, admitted: u32, room: u32) -> Option<Prepared> {
        if self.config.prefill.is_empty() || room == 0 {
            return None;
        }
        // First in admission order, so a queue of prompts is prefilled in the order it arrived.
        let index = self
            .running
            .iter()
            .position(|s| s.prefilling() && s.steppable())?;
        let remaining = self.running[index].prefill_room() as u32;
        self.chunk_step(index, remaining.min(room), admitted)
    }

    /// C10: a PACKED step's chunks — [`crate::mixed_law::pack`] over the queued prompts' unpacked
    /// chunks, in admission order, beside `decode_rows` decode rows. `None` when nothing rides
    /// (the budget is spent, or no prefill bucket fits the head).
    fn prepare_packed(&mut self, decode_rows: u32) -> Option<(ReadyStep, Vec<ReadyStep>)> {
        if self.config.prefill.is_empty() {
            return None;
        }
        let max_chunks = self.config.mixed.max_chunks;
        let queue: Vec<(usize, u32)> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.prefilling() && s.steppable())
            .take(max_chunks as usize)
            .filter_map(|(i, s)| {
                // FN38: a packed step always carries decode rows -- the busy cap holds
                let remaining = (s.prefill_room() as u32).min(self.config.prefill.room(true));
                self.config
                    .prefill
                    .chunk_for(remaining)
                    .map(|(c, _)| (i, c))
            })
            .collect();
        let unpacked: Vec<u32> = queue.iter().map(|q| q.1).collect();
        let rungs = self.config.prefill.sizes.clone();
        let p = crate::mixed_law::pack(
            &unpacked,
            decode_rows,
            &rungs,
            self.config.mixed.max_batched_tokens,
            max_chunks,
        );
        let mut steps = Vec::with_capacity(p.len());
        for (k, rows) in p.rows.iter().enumerate() {
            // THE CHUNK-SEQUENCE LAW: every packed chunk is the prompt's unpacked chunk (the
            // head's alone may be the operator's budget cut) — `mixed_law::chunk_sequence_holds`
            // is the whole-run statement, tested over every arrival pattern.
            debug_assert!(*rows == unpacked[k] || (k == 0 && p.budget_cut));
            match self.chunk_step(queue[k].0, *rows, 0) {
                Some(Prepared::Ready(s)) => steps.push(s),
                _ => break,
            }
        }
        if steps.is_empty() {
            return None;
        }
        let first = steps.remove(0);
        Some((first, steps))
    }

    /// One sequence's next chunk of at most `feed` rows, as a one-slot prefill step.
    fn chunk_step(&mut self, index: usize, feed: u32, admitted: u32) -> Option<Prepared> {
        let seq = &self.running[index];
        // FN52: a chunk never crosses a cut (the prefix snapshot / verify point).
        let feed = feed.min(seq.prefill_room() as u32);
        let (chunk, bucket_tokens) = self.config.prefill.chunk_for(feed)?;
        let done_before = seq.prefilled as u32;
        let ctx_tokens = done_before + chunk;

        let bucket = match select_prefill(
            &self.config.prefill_buckets,
            PrefillRequest {
                tokens: bucket_tokens,
                ctx_tokens,
            },
        ) {
            Ok(b) => b,
            Err(e) => {
                // Fall through to the decode path, which is what served prompts before P2:
                // the sequence is marked prompt-resident so the decode step feeds the whole
                // thing, rather than waiting for a chunk shape that does not exist. Counted,
                // because a silent fallback looks exactly like success.
                self.counters.prefill_bucket_misses += 1;
                tracing::warn!(error = %e, "no captured prefill bucket fits; feeding the prompt as one decode step");
                let seq = &mut self.running[index];
                seq.prefilled = seq.prefill_len;
                return None;
            }
        };

        let seq = &self.running[index];
        let last = seq.prefilled + chunk as usize >= seq.prefill_len;
        let input: Vec<TokenId> =
            seq.tokens[seq.prefilled..seq.prefilled + chunk as usize].to_vec();
        let slots = vec![BatchSlot {
            request_id: seq.request_id,
            seq_id: seq.seq_id,
            slot: 0,
            kv_blocks: seq.kv.blocks.clone(),
            gdn_set: seq.gdn.set,
            // A prompt has nothing to speculate on.
            verify_width: 0,
            input_tokens: input,
            context_len: done_before,
            first_step: seq.prefilled == 0,
            // The LAST prefill chunk samples the sequence's first token, and it is sampled by the
            // same rule every later step uses. A chunk that is not the last carries it too and
            // nothing reads it, which is cheaper than a special case that could disagree.
            sampling: seq.sampling(),
            chunk: Some(PrefillChunk {
                prompt_tokens: chunk,
                tokens_done_before: done_before,
                last,
                next_prompt_token: if last {
                    None
                } else {
                    seq.tokens.get(seq.prefilled + chunk as usize).copied()
                },
            }),
        }];

        self.counters.prefill_steps += 1;
        // The decoding sequences this chunk's step leaves waiting if it runs alone.
        let waiting_decoders = self
            .running
            .iter()
            .filter(|s| !s.prefilling() && s.steppable())
            .count() as u32;
        Some(Prepared::Ready(ReadyStep {
            phase: Phase::Prefill,
            bucket,
            slots,
            running: 1,
            width: 0,
            admitted,
            step_index: self.step_index,
            chunk: None,
            more: Vec::new(),
            waiting_decoders,
        }))
    }

    /// Phase 3: commit what the executor returned, against whatever is **still** running.
    ///
    /// Results are matched by [`SeqId`], never by slot index. The core lock was released for the
    /// device call, so a sequence in this batch may have been aborted meanwhile: its result is
    /// dropped and counted, and the sequences after it have moved in `running`.
    ///
    /// One HALF of a step: the step's own counters (`steps`, `step_index`) move once per step in
    /// [`Scheduler::step`], after every half has committed, so a mixed step is one step.
    fn commit_half<E: StepExecutor>(
        &mut self,
        exec: &mut E,
        ready: &ReadyStep,
        result: StepResult,
    ) -> StepSummary {
        let mut committed_tokens = 0u32;
        let mut finished = 0u32;
        let mut done: Vec<SeqId> = Vec::new();
        // Leaders whose LAST prefill chunk committed this step, in the order they did. Each one
        // is a group's fork point: the row its first token came from is on the device now and
        // nowhere else, which is why the branches are forked here and not at admission.
        let mut sampled_leaders: Vec<(RequestId, SeqId)> = Vec::new();
        // FN52: sequences whose prefill just reached a cut.
        let mut prefix_owed: Vec<SeqId> = Vec::new();

        // The knobs the EMA is folded with, copied out before the disjoint borrow below.
        let width_policy = self.config.width.prod;
        let step_index = self.step_index;

        {
            // Disjoint field borrows: the sequence being committed, the counters and the spec
            // totals are three different fields of the same struct.
            let Core {
                running,
                counters,
                spec,
                ..
            } = &mut *self;

            for (i, slot) in ready.slots.iter().enumerate() {
                let Some(pos) = running.iter().position(|s| s.seq_id == slot.seq_id) else {
                    // Aborted or parked while the step was in flight. Its allocations are already
                    // in `retiring` and its stream already has its finished event.
                    counters.results_for_gone_sequences += 1;
                    continue;
                };
                let outcome = result.seqs.get(i);
                let seq = &mut running[pos];
                let Some(out) = outcome else {
                    counters.executor_protocol_errors += 1;
                    tracing::error!(seq = %slot.seq_id, "executor returned no result for this slot");
                    emit(seq, Vec::new(), None, Some(FinishReason::Abort));
                    done.push(slot.seq_id);
                    finished += 1;
                    continue;
                };
                // A prefill chunk is its own contract: no token until the last one, and never a
                // draft. The scheduler knows which chunk is last, so the check is exact rather
                // than a range.
                if let Some(chunk) = slot.chunk {
                    let want = usize::from(chunk.last);
                    if out.draft_tokens != 0 || out.accepted_tokens != 0 {
                        counters.executor_protocol_errors += 1;
                        tracing::error!(
                            seq = %slot.seq_id,
                            drafted = out.draft_tokens,
                            accepted = out.accepted_tokens,
                            "a prefill chunk has nothing to speculate on"
                        );
                        emit(seq, Vec::new(), None, Some(FinishReason::Abort));
                        done.push(slot.seq_id);
                        finished += 1;
                        continue;
                    }
                    if out.tokens.len() != want {
                        counters.executor_token_count_errors += 1;
                        tracing::error!(
                            seq = %slot.seq_id,
                            tokens = out.tokens.len(),
                            want,
                            last = chunk.last,
                            "a prefill chunk commits one token if it is the last and none if it \
                             is not"
                        );
                        emit(seq, Vec::new(), None, Some(FinishReason::Abort));
                        done.push(slot.seq_id);
                        finished += 1;
                        continue;
                    }
                    // PROMPT LOGPROBS, before anything else of this chunk is committed (V7): the
                    // rows it scored, appended in order. A request that asked and a chunk that
                    // did not answer, or answered for a different number of rows, is a request
                    // that would otherwise get a list with a hole in it — it ends here, loudly.
                    if let Some(acc) = seq.prompt_lp.as_mut() {
                        let want = chunk.prompt_tokens as usize - usize::from(chunk.last);
                        match &out.prompt_logprobs {
                            Some(rows) if rows.len() == want => acc.extend(rows.iter().cloned()),
                            other => {
                                counters.executor_protocol_errors += 1;
                                let got = other.as_ref().map(Vec::len);
                                tracing::error!(
                                    seq = %slot.seq_id, want, ?got,
                                    "the executor did not score this chunk's prompt rows"
                                );
                                emit_error(
                                    &seq.events,
                                    seq.request_id,
                                    seq.choice_index,
                                    seq.step_index,
                                    &format!(
                                        "prompt_logprobs: the engine scored {got:?} of this \
                                         chunk's {want} prompt row(s); the request is refused \
                                         rather than answered with a partial list"
                                    ),
                                );
                                done.push(slot.seq_id);
                                finished += 1;
                                continue;
                            }
                        }
                    }
                    counters.prefill_computed_tokens += u64::from(chunk.prompt_tokens);
                    seq.prefilled += chunk.prompt_tokens as usize;
                    if seq.cuts.contains(&(seq.prefilled as u32)) {
                        prefix_owed.push(slot.seq_id);
                    }
                    if chunk.last {
                        // The prompt is in. This is the sequence's first token, and the first
                        // thing its client hears: every chunk before it emitted nothing, which
                        // is why TTFT covers the whole prefill.
                        seq.first_step = false;
                        let (commit, finish) = commit_tokens(seq, &out.tokens);
                        committed_tokens += commit.len() as u32;
                        seq.tokens.extend_from_slice(&commit);
                        seq.generated += commit.len();
                        seq.pending_input.clone_from(&commit);
                        seq.last_active_step = ready.step_index;
                        let (lp, short) = logprobs_for(&out.logprobs, commit.len());
                        if short {
                            counters.executor_protocol_errors += 1;
                            tracing::error!(
                                seq = %slot.seq_id,
                                "the executor reported fewer logprobs than the tokens it committed"
                            );
                        }
                        let request_id = seq.request_id;
                        let plp = seq.prompt_lp.take();
                        emit_with_prompt(seq, commit, lp, finish, plp);
                        // The group's fork point, whether or not this token also finished the
                        // leader: the branches draw from the row this chunk produced, and a
                        // leader that stopped on its first token is still the row they draw from.
                        sampled_leaders.push((request_id, slot.seq_id));
                        if let Some(reason) = finish {
                            match reason {
                                FinishReason::Stop => counters.finished_stop += 1,
                                FinishReason::Length => counters.finished_length += 1,
                                _ => {}
                            }
                            done.push(slot.seq_id);
                            finished += 1;
                        }
                    }
                    continue;
                }

                if out.seq_id != slot.seq_id
                    || out.draft_tokens > slot.verify_width
                    || out.accepted_tokens > out.draft_tokens
                {
                    counters.executor_protocol_errors += 1;
                    tracing::error!(
                        seq = %slot.seq_id,
                        got = %out.seq_id,
                        drafted = out.draft_tokens,
                        scheduled = slot.verify_width,
                        accepted = out.accepted_tokens,
                        "executor result does not match the batch"
                    );
                    emit(seq, Vec::new(), None, Some(FinishReason::Abort));
                    done.push(slot.seq_id);
                    finished += 1;
                    continue;
                }
                // A verified step commits the accepted prefix plus exactly one token — the
                // recovery token or the bonus. `_insert_resampled_kernel` increments
                // `num_sampled` unconditionally; there is no path that commits only the prefix
                // and none that commits two extra. An executor whose two numbers disagree would
                // desynchronise the `saber:spec_decode_*` counters from the token stream, silently
                // and forever, so it is a counted protocol error rather than a clamp.
                if out.tokens.len() != out.accepted_tokens as usize + 1 {
                    counters.executor_token_count_errors += 1;
                    tracing::error!(
                        seq = %slot.seq_id,
                        tokens = out.tokens.len(),
                        accepted = out.accepted_tokens,
                        "executor committed {} tokens for {} accepted; the identity is \
                         tokens == accepted + 1",
                        out.tokens.len(),
                        out.accepted_tokens
                    );
                    emit(seq, Vec::new(), None, Some(FinishReason::Abort));
                    done.push(slot.seq_id);
                    finished += 1;
                    continue;
                }

                if out.draft_tokens > 0 {
                    // Folded by `saber-model`'s counters, which are gated against prod's own
                    // definitions -- including `accepted_per_pos`, the per-position curve
                    // `eval_drafter.py` reads to see WHERE acceptance dies.
                    spec.observe(out.draft_tokens, &out.as_verified());
                    // The observation's denominator is the SCHEDULED width, and full acceptance
                    // observes one above it so the cap can climb back out of its own shadow.
                    seq.ema
                        .observe(&width_policy, out.accepted_tokens, out.draft_tokens);
                }

                // A prompt fed by ONE DECODE-SHAPED STEP (no prefill bucket fitted it) was never
                // scored row by row, so a request that asked for its prompt logprobs is refused
                // here rather than answered with a list that does not exist (V7).
                if seq.first_step && seq.prompt_lp.is_some() {
                    counters.executor_protocol_errors += 1;
                    emit_error(
                        &seq.events,
                        seq.request_id,
                        seq.choice_index,
                        seq.step_index,
                        "prompt_logprobs: this prompt was fed by a single decode step because no \
                         prefill bucket fits it, and that step scores no prompt rows",
                    );
                    done.push(slot.seq_id);
                    finished += 1;
                    continue;
                }
                let (commit, finish) = commit_tokens(seq, &out.tokens);
                if !commit.is_empty() {
                    seq.last_active_step = step_index;
                }
                committed_tokens += commit.len() as u32;
                seq.tokens.extend_from_slice(&commit);
                seq.generated += commit.len();
                seq.pending_input.clone_from(&commit);
                // A DECODE step can be the fork point too, and this is the pre-P2 shape: with no
                // captured prefill bucket the prompt is fed by one decode-shaped step, so that
                // step is the one that samples the prompt's first token. `first_step` is what
                // says so, and it is read before it is cleared.
                if seq.first_step {
                    counters.prefill_computed_tokens += slot.input_tokens.len() as u64;
                    sampled_leaders.push((seq.request_id, slot.seq_id));
                }
                seq.first_step = false;
                let (lp, short) = logprobs_for(&out.logprobs, commit.len());
                if short {
                    counters.executor_protocol_errors += 1;
                    tracing::error!(
                        seq = %slot.seq_id,
                        "the executor reported fewer logprobs than the tokens it committed"
                    );
                }
                emit(seq, commit, lp, finish);
                if let Some(reason) = finish {
                    match reason {
                        FinishReason::Stop => counters.finished_stop += 1,
                        FinishReason::Length => counters.finished_length += 1,
                        _ => {}
                    }
                    done.push(slot.seq_id);
                    finished += 1;
                }
            }
        }

        // ---- the forks, at the only step they can happen at -------------------------------------
        //
        // A branch of an `n > 1` request has held its pages and its state slot set since the group
        // was admitted, and the executor has not been told about it (`Seq::pending_fork`). Now the
        // leader's last prefill chunk has sampled, so all three of the branch's starting facts
        // exist: the prompt's pages are written, the leader's GDN state is the prompt's final
        // state, and the row the first token is drawn from is on the device. `StepExecutor::fork`
        // does the copies and returns the branch's OWN draw from that row — a second draw, with
        // this branch's noise key — and that token is this branch's first `StepEvent`.
        for (request_id, leader) in sampled_leaders {
            let branches: Vec<usize> = self
                .running
                .iter()
                .enumerate()
                .filter(|(_, s)| s.request_id == request_id && s.pending_fork.is_some())
                .map(|(i, _)| i)
                .collect();
            for i in branches {
                let admitted = self.running[i]
                    .pending_fork
                    .clone()
                    .expect("filtered on pending_fork");
                let forked = exec.fork(leader, &admitted);
                let step_index = self.step_index;
                let seq = &mut self.running[i];
                let Some(Forked {
                    token,
                    logprobs: first_logprobs,
                }) = forked
                else {
                    tracing::error!(
                        seq = %seq.seq_id, leader = %leader, choice = seq.choice_index,
                        "the executor refused to fork this branch; it is aborted rather than \
                         started from a state nothing copied"
                    );
                    emit(seq, Vec::new(), None, Some(FinishReason::Abort));
                    let seq_id = seq.seq_id;
                    self.counters.fork_failures += 1;
                    done.push(seq_id);
                    finished += 1;
                    continue;
                };
                // It is no longer pending, so it is steppable, releasable and parkable from here.
                seq.pending_fork = None;
                // The prompt is IN — it is the leader's pages, shared — so the branch never
                // prefills and its next step is an ordinary decode step.
                seq.prefilled = seq.prefill_len;
                seq.first_step = false;
                seq.last_active_step = step_index;
                let (commit, finish) = commit_tokens(seq, &[token]);
                committed_tokens += commit.len() as u32;
                seq.tokens.extend_from_slice(&commit);
                seq.generated += commit.len();
                seq.pending_input.clone_from(&commit);
                // THE FIRST TOKEN'S REPORT (ws-f39): the executor's draw over the leader's row
                // reports exactly what the leader's own first token reported — the same row, the
                // same pipeline — so the branch's first choice chunk carries it the way the
                // leader's does. A request that asked and got none is still counted.
                let wants_logprobs = seq.params.logprobs.is_some();
                let lp = if wants_logprobs {
                    first_logprobs.map(|l| vec![l])
                } else {
                    None
                };
                let absent = wants_logprobs && lp.is_none();
                emit(seq, commit, lp, finish);
                let seq_id = seq.seq_id;
                self.counters.forks += 1;
                if absent {
                    self.counters.fork_first_token_logprobs_absent += 1;
                }
                if let Some(reason) = finish {
                    match reason {
                        FinishReason::Stop => self.counters.finished_stop += 1,
                        FinishReason::Length => self.counters.finished_length += 1,
                        _ => {}
                    }
                    done.push(seq_id);
                    finished += 1;
                }
            }
        }
        for seq_id in prefix_owed {
            if !done.contains(&seq_id) {
                self.prefix_at_cut(exec, seq_id);
            }
        }
        // A branch whose leader left WITHOUT sampling — aborted, or a prefill the executor
        // refused — has no row to fork from and never will. A group is one unit, so it goes with
        // it rather than sitting on its arena for ever.
        let orphans: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.pending_fork.is_some())
            .filter(|(_, s)| {
                !self.running.iter().any(|l| {
                    l.request_id == s.request_id && l.choice_index == 0 && !done.contains(&l.seq_id)
                })
            })
            .map(|(i, _)| i)
            .collect();
        for i in orphans {
            let seq = &mut self.running[i];
            tracing::error!(
                seq = %seq.seq_id, choice = seq.choice_index,
                "this branch's leader left before it sampled; there is no row to fork from"
            );
            emit(seq, Vec::new(), None, Some(FinishReason::Abort));
            let seq_id = seq.seq_id;
            self.counters.fork_branches_orphaned += 1;
            done.push(seq_id);
            finished += 1;
        }

        // Finished sequences free both allocations in this step, so the admission at the top of
        // the next one sees the space (continuous batching, not generation-batching). Nothing is
        // in flight here — this runs with both locks — so they go back immediately.
        for seq_id in done {
            if let Some(pos) = self.running.iter().position(|s| s.seq_id == seq_id) {
                let seq = self.running.remove(pos);
                self.retiring.push(Retired {
                    seq_id: seq.seq_id,
                    kv: seq.kv,
                    gdn: seq.gdn,
                    admitted: seq.pending_fork.is_none(),
                });
            }
        }
        self.drain_retiring(exec);
        // ...and the space that just came back is what a policy-parked session was evicted for.
        if finished > 0 {
            self.requeue_auto_parked(finished);
        }

        self.counters.output_committed_tokens += u64::from(committed_tokens);
        StepSummary {
            step_index: ready.step_index,
            bucket: ready.bucket,
            sequences: ready.running,
            padded_slots: ready.bucket.batch - ready.running,
            verify_width: ready.width,
            committed_tokens,
            admitted: ready.admitted,
            finished,
        }
    }

    /// Hands back the allocations of sequences that left the running set while a step might have
    /// been reading their slots. Runs only with both locks held, which is what makes it safe.
    fn drain_retiring<E: StepExecutor>(&mut self, exec: &mut E) {
        // Images dropped unrestored: whatever the executor holds FOR them goes too (ws-k18). At
        // TP=2 that is the follower's half, which nothing else would ever free before exit.
        for seq in std::mem::take(&mut self.discards_owed) {
            exec.discard_parked(seq);
            self.counters.parked_discards += 1;
        }
        for r in std::mem::take(&mut self.retiring) {
            // A branch whose leader never sampled was never admitted to the executor (see
            // `Seq::pending_fork`), so there is nothing there to release. Its arena is still
            // owed: the group took it all at once.
            if r.admitted {
                exec.release(r.seq_id);
            }
            self.arenas.free(r.kv, r.gdn);
        }
    }

    /// Aborts a request: **every** branch of its group, as one unit.
    ///
    /// A group half-aborted would leave branches holding a prompt prefix nobody will read and a
    /// stream nobody is listening to, so the whole group goes. Each branch emits its own final
    /// event (one `finished` per choice) and its allocations are retired rather than freed,
    /// because a step in flight may still be reading its slots.
    fn abort(&mut self, id: RequestId) {
        let positions: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.request_id == id)
            .map(|(i, _)| i)
            .collect();
        if !positions.is_empty() {
            // A running group leaving is the same event as one finishing, as far as a session the
            // policy evicted is concerned: the space it was evicted for is coming back, and a
            // group frees as many slots as it had branches. Only the queue is touched here -- the
            // allocations are owed until `drain_retiring`, and the restore happens in the next
            // `prepare`.
            self.requeue_auto_parked(positions.len() as u32);
            // Highest index first, so the earlier positions stay valid.
            for pos in positions.into_iter().rev() {
                let seq = self.running.remove(pos);
                emit_owned(
                    &seq.events,
                    seq.request_id,
                    seq.choice_index,
                    seq.step_index,
                    Vec::new(),
                    None,
                    Some(FinishReason::Abort),
                );
                self.retiring.push(Retired {
                    seq_id: seq.seq_id,
                    kv: seq.kv,
                    gdn: seq.gdn,
                    admitted: seq.pending_fork.is_none(),
                });
            }
            self.counters.aborts_running += 1;
            return;
        }
        if let Some(pos) = self.waiting.iter().position(|w| w.request_id == id) {
            let w = self.waiting.remove(pos).expect("position is in range");
            for choice in 0..w.n {
                emit_owned(
                    &w.events,
                    w.request_id,
                    choice,
                    0,
                    Vec::new(),
                    None,
                    Some(FinishReason::Abort),
                );
            }
            if let Some(parked) = w.resume {
                self.discard_group(&parked);
                self.counters.aborts_parked += 1;
            } else {
                self.counters.aborts_waiting += 1;
            }
            return;
        }
        if let Some(pos) = self.parked.iter().position(|p| p.request_id == id) {
            let parked = self.parked.remove(pos);
            for branch in &parked.branches {
                emit_owned(
                    &parked.events,
                    parked.request_id,
                    branch.choice_index,
                    branch.step_index,
                    Vec::new(),
                    None,
                    Some(FinishReason::Abort),
                );
            }
            self.discard_group(&parked);
            self.counters.aborts_parked += 1;
        }
    }

    /// FN73: the process is going down (`saber_serve::fatal`): every running branch, every queued
    /// request and every parked group gets its `finished` event WITH the reason (the serving plane
    /// renders it as an error body), and nothing is freed -- a step in flight may still hold the
    /// slots, and the process is exiting.
    fn fail_all(&mut self, reason: &str) -> usize {
        let mut n = 0;
        for seq in std::mem::take(&mut self.running) {
            emit_error(
                &seq.events,
                seq.request_id,
                seq.choice_index,
                seq.step_index,
                reason,
            );
            n += 1;
        }
        for w in std::mem::take(&mut self.waiting) {
            for choice in 0..w.n {
                emit_error(&w.events, w.request_id, choice, 0, reason);
            }
            n += 1;
        }
        for p in std::mem::take(&mut self.parked) {
            for b in &p.branches {
                emit_error(
                    &p.events,
                    p.request_id,
                    b.choice_index,
                    b.step_index,
                    reason,
                );
            }
            n += 1;
        }
        n
    }

    /// Drops every branch of a parked group from the tier, and owes the executor a
    /// [`StepExecutor::discard_parked`] for each: the tier's ticket is released here and the
    /// executor's hold on the image at the next moment the executor lock is held — at TP=2 both
    /// ranks' halves, where before ws-k18 the follower kept its half until it exited.
    fn discard_group(&mut self, parked: &ParkedSeq) {
        for branch in &parked.branches {
            if let Some(tier) = self.park_tier.as_mut() {
                tier.discard(&branch.ticket, &mut self.ledger);
            }
            self.discards_owed.push(branch.parked_as);
        }
    }

    fn stats(&self) -> EngineStats {
        // Requests, not sequences. A group of three branches is one running request: the gates
        // inside the engine count sequences (the batch, the Infinity Cache, MIN_BATCH), and
        // `/metrics` counts what a client would recognise as a request.
        let mut requests: Vec<RequestId> = self.running.iter().map(|s| s.request_id).collect();
        requests.sort_unstable();
        requests.dedup();
        EngineStats {
            num_requests_running: requests.len() as u64,
            num_requests_waiting: self.waiting.len() as u64,
            spec_decode_num_drafts: self.spec.num_drafts,
            spec_decode_num_draft_tokens: self.spec.num_draft_tokens,
            spec_decode_num_accepted_tokens: self.spec.num_accepted_tokens,
            spec_decode_accepted_per_pos: self.spec.accepted_per_pos.clone(),
            census: Vec::new(),
            memory_plane: String::new(),
        }
    }
}

/// Applies the stop conditions to what a step produced.
///
/// Returns the tokens that are actually committed and, if the sequence ends here, why. An EOS token
/// is committed (it counts towards `usage`; the serving plane is what refuses to detokenize it) and
/// everything the executor speculated past it is dropped.
fn commit_tokens(seq: &Seq, produced: &[TokenId]) -> (Vec<TokenId>, Option<FinishReason>) {
    let mut commit = Vec::with_capacity(produced.len());
    for &t in produced {
        commit.push(t);
        let stop = seq.params.stop_token_ids.contains(&t);
        if stop {
            return (commit, Some(FinishReason::Stop));
        }
        if seq.generated + commit.len() >= seq.max_tokens {
            return (commit, Some(FinishReason::Length));
        }
    }
    (commit, None)
}

fn emit(
    seq: &mut Seq,
    tokens: Vec<TokenId>,
    logprobs: Option<Vec<SampledLogprobs>>,
    finished: Option<FinishReason>,
) {
    emit_with_prompt(seq, tokens, logprobs, finished, None);
}

/// [`emit`], carrying the request's `prompt_logprobs` — which the reference server delivers once, with the first
/// token, and which is the only event that ever carries them.
fn emit_with_prompt(
    seq: &mut Seq,
    tokens: Vec<TokenId>,
    logprobs: Option<Vec<SampledLogprobs>>,
    finished: Option<FinishReason>,
    prompt_logprobs: Option<Vec<PromptPositionLogprobs>>,
) {
    let _ = seq.events.send(StepEvent {
        request_id: seq.request_id,
        choice_index: seq.choice_index,
        step_index: seq.step_index,
        new_tokens: tokens,
        finished,
        logprobs,
        prompt_logprobs,
        error: None,
    });
    seq.step_index += 1;
}

/// The logprobs for the tokens this step actually COMMITTED.
///
/// The executor reports one entry per token it produced; the stop rule above may have kept fewer
/// of them, and a reply whose `logprobs.content` is longer than its text is a reply no client can
/// line up. Truncation is therefore the rule, and an executor that returned FEWER entries than
/// tokens is a protocol error rather than a short list — the caller counts it.
fn logprobs_for(
    logprobs: &Option<Vec<SampledLogprobs>>,
    committed: usize,
) -> (Option<Vec<SampledLogprobs>>, bool) {
    match logprobs {
        None => (None, false),
        Some(lp) if lp.len() >= committed => (Some(lp[..committed].to_vec()), false),
        Some(lp) => (Some(lp.clone()), true),
    }
}

/// The `finished` event of a choice the ENGINE ended, with the reason the client is owed
/// (ws-k18). `FinishReason::Abort` plus `StepEvent::error`: the serving plane renders the pair as
/// an OpenAI error body.
fn emit_error(
    events: &EventSink,
    request_id: RequestId,
    choice_index: u32,
    step_index: u32,
    message: &str,
) {
    let _ = events.send(StepEvent {
        request_id,
        choice_index,
        step_index,
        new_tokens: Vec::new(),
        finished: Some(FinishReason::Abort),
        logprobs: None,
        prompt_logprobs: None,
        error: Some(message.to_owned()),
    });
}

fn emit_owned(
    events: &EventSink,
    request_id: RequestId,
    choice_index: u32,
    step_index: u32,
    tokens: Vec<TokenId>,
    logprobs: Option<Vec<SampledLogprobs>>,
    finished: Option<FinishReason>,
) {
    let event = StepEvent {
        request_id,
        choice_index,
        step_index,
        new_tokens: tokens,
        finished,
        // One entry per committed token when the request asked for them, and `None` when it did
        // not. The sampler is what produces these (`saber_serve::sampler::Sampler`); before it
        // was wired into the step loop this was unconditionally `None`.
        logprobs,
        prompt_logprobs: None,
        error: None,
    };
    // The receiver is the client's stream; a dropped stream is not an abort, so a send failure is
    // expected and not an error.
    let _ = events.send(event);
}

impl<E: StepExecutor> Engine for Scheduler<E> {
    fn submit(&self, request: EngineRequest) -> Result<Submission, EngineError> {
        self.submit_inner(request, None)
    }

    /// D3: an image request is queued like a text one, carrying its processed images; the tower
    /// runs when it reaches the head of the queue ([`StepExecutor::encode_images`]). An executor
    /// with no tower is refused HERE, by name, before anything is queued.
    fn submit_with_images(
        &self,
        request: EngineRequest,
        images: saber_serve::engine::ImageBatch,
    ) -> Result<Submission, EngineError> {
        let serves = self
            .executor
            .lock()
            .map(|e| e.serves_images())
            .unwrap_or(false);
        if !serves {
            return Err(EngineError::Unsupported {
                param: "image",
                message: saber_serve::engine::TEXT_ONLY_REFUSAL.to_owned(),
            });
        }
        self.submit_inner(request, Some(images))
    }

    fn abort(&self, id: RequestId) {
        self.abort_inner(id);
    }

    fn stats(&self) -> EngineStats {
        self.stats_inner()
    }

    fn model(&self) -> &ModelInfo {
        self.model_inner()
    }
}

impl<E: StepExecutor> Scheduler<E> {
    fn submit_inner(
        &self,
        request: EngineRequest,
        images: Option<saber_serve::engine::ImageBatch>,
    ) -> Result<Submission, EngineError> {
        let prompt_len = request.prompt_token_ids.len();
        if prompt_len > self.model.max_model_len {
            return Err(EngineError::PromptTooLong {
                got: prompt_len,
                max: self.model.max_model_len,
            });
        }
        let mut core = self.lock();
        // FN73: after a fatal nothing is admitted (under the core lock, so a request queued before
        // the fatal is ended by `fail_all` and one after it is refused: 503)
        if saber_serve::fatal::get().is_some() {
            return Err(EngineError::ShuttingDown);
        }
        if prompt_len >= core.config.max_context_tokens {
            return Err(EngineError::PromptTooLong {
                got: prompt_len,
                max: core.config.max_context_tokens,
            });
        }

        // The reference server's rule: `max_tokens` is clamped to what the context leaves, and the request then
        // finishes with `length` rather than being refused.
        let budget = core.config.max_context_tokens - prompt_len;
        // FN51: a request without `max_tokens` under an implicit context (see the config field)
        let implicit = core.config.implicit_context.map_or(budget, |c| {
            if prompt_len < c {
                c - prompt_len
            } else {
                core.config.implicit_headroom
            }
        });
        let max_tokens = request
            .params
            .max_tokens
            .unwrap_or(implicit)
            .clamp(1, budget);

        // EOS ids are the model's; `ignore_eos` drops them, stop ids from the request are added.
        // Resolving them once, here, is what keeps the step loop free of model lookups.
        let mut params = request.params;
        // The budget as resolved: the executor's drawn walk stops at the last position a request
        // can reach, so a verify step never draws a row nothing will serve (ws-f40).
        params.max_tokens = Some(max_tokens);
        if !params.ignore_eos {
            for eos in &self.model.eos_token_ids {
                if !params.stop_token_ids.contains(eos) {
                    params.stop_token_ids.push(*eos);
                }
            }
        }

        // `n > 1` is a sequence group: one prompt, `n` branches. The executor has to be able to
        // branch one -- a group served by an executor that cannot is `n` requests wearing one
        // request's name -- so it is refused here rather than quietly downgraded.
        let n = u32::try_from(params.n).unwrap_or(u32::MAX).max(1);
        if n > 1 && !core.can_fork {
            // A 400 naming the parameter, not a 500 (V7): the request is well-formed and this
            // engine cannot serve it AS ASKED, and a client that retries gets the same answer.
            // Since ws-f38 the serving engine forks under its drafter; what is left here is an
            // executor with no fork at all, and the in-step proposal order (a pairing control).
            return Err(EngineError::Unsupported {
                param: "n",
                message: format!(
                    "n = {n} is not supported by this engine: a sequence group needs an executor \
                     that can branch one, and this one cannot. Send n = 1 requests instead. \
                     (parameter=n, value={n})"
                ),
            });
        }
        // `prompt_logprobs` (V7): refused by name when this engine cannot produce them or not
        // this many — never served as a `null`. `-1` is the whole vocabulary, which no seam holds.
        if let Some(plp) = params.prompt_logprobs {
            let want = if plp == -1 {
                u32::MAX
            } else {
                plp.max(0) as u32
            };
            let refusal = match core.prompt_logprobs_max {
                None => Some("this engine does not produce prompt logprobs".to_owned()),
                Some(max) if want > max => Some(format!(
                    "this engine reports at most {max} prompt logprobs per position"
                )),
                Some(_) if core.config.prefill.is_empty() => Some(
                    "this engine has no prefill bucket, and prompt logprobs are scored by one"
                        .to_owned(),
                ),
                Some(_) => None,
            };
            if let Some(why) = refusal {
                return Err(EngineError::Unsupported {
                    param: "prompt_logprobs",
                    message: format!(
                        "prompt_logprobs = {plp} is not supported: {why}. \
                         (parameter=prompt_logprobs, value={plp})"
                    ),
                });
            }
        }
        // A group is admitted WHOLE — its branches fork at the leader's last prefill chunk, so
        // they are resident together — and a group wider than the batch could never be. the reference server's
        // own text for an `n` above its limit, with this engine's limit in it: a 400 naming `n`
        // (it was a 500 until ws-f38), because the request is well-formed and cannot be served
        // as asked however often it is retried.
        let max_n = core.config.max_batch();
        if n > max_n {
            return Err(EngineError::Unsupported {
                param: "n",
                message: format!(
                    "n must be at most {max_n}, got {n}. This engine admits a sequence group \
                     whole and serves at most {max_n} sequences at once (--max-seqs). \
                     (parameter=n, value={n})"
                ),
            });
        }

        let id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = unbounded_channel();
        if core.config.admit.slack_ms > 0 {
            // LR-F: the finish -> arrival predictor
            let cap = std::time::Duration::from_millis(u64::from(core.config.admit.slack_ms));
            core.slack.arrival(self.clock.now(), cap);
        }
        core.waiting.push_back(Waiting {
            request_id: id,
            params,
            prompt: request.prompt_token_ids,
            prompt_len,
            max_tokens,
            n,
            events: tx,
            resume: None,
            images,
            arrived: Some(self.clock.now()),
            recompute: None,
        });
        Ok(Submission {
            id,
            events: Box::pin(EventStream { rx }) as StepStream,
        })
    }

    /// FN73: fail every in-flight and queued request with `reason` (the fatal teardown); returns
    /// how many requests it ended. Takes the core lock only (never the executor's: a step may be
    /// stuck in the device).
    pub fn fail_all(&self, reason: &str) -> usize {
        self.lock().fail_all(reason)
    }

    fn abort_inner(&self, id: RequestId) {
        let mut core = self.lock();
        core.abort(id);
        // The allocations go back as soon as no step is using them. `try_lock` *is* the question
        // "is a step in flight": when one is, the sequence waits in `retiring` and the step's own
        // commit hands it back a few milliseconds later.
        if let Ok(mut exec) = self.executor.try_lock() {
            core.drain_retiring(&mut *exec);
            exec.publish_census();
        } else {
            core.counters.retires_deferred += 1;
        }
    }

    /// The engine's gauges and spec counters, and the whole shutdown census (ws-v8).
    ///
    /// The scheduler's and the admission counters are read under the core lock, as the spec
    /// counters always were. The executor's are read from its mirror, which the step loop
    /// publishes and this never waits on — except once: the FIRST scrape arms the mirror and, when
    /// nothing has been published yet, publishes itself if the executor is free or waits (bounded)
    /// for the step in flight to do it.
    fn stats_inner(&self) -> EngineStats {
        let (mut stats, sched) = {
            let core = self.lock();
            let mut sched = saber_serve::census::collect(SchedCounters::CENSUS, |put| {
                core.counters.census_values(put)
            });
            let admission = core.arenas.counters();
            sched.extend(saber_serve::census::collect(
                crate::alloc::AdmissionCounters::CENSUS,
                |put| admission.census_values(put),
            ));
            let k = core.kv_admission_snapshot();
            let refused_requests = core.kv_refused_requests;
            sched.extend(saber_serve::census::collect(KV_ADMISSION_GAUGES, |put| {
                put(u64::from(k.total_blocks));
                put(u64::from(k.free_blocks));
                put(k.reserved_tokens);
                put(k.length_tokens);
                put(k.max_reserved_tokens);
                put(refused_requests);
                let a = core.slack.counters;
                put(a.lazy_grows);
                put(a.lazy_preemptions);
                put(a.lazy_recompute_tokens);
                put(a.lazy_margin_refusals);
            }));
            (core.stats(), sched)
        };
        if let Some(mirror) = &self.census {
            if !mirror.published() {
                mirror.arm();
                match self.executor.try_lock() {
                    Ok(mut exec) => exec.publish_census(),
                    Err(_) => {
                        mirror.wait_first(FIRST_SCRAPE_WAIT);
                    }
                }
            }
            if let Some(samples) = mirror.read() {
                stats.census = samples;
            }
        }
        stats.census.extend(sched);
        if let Some(m) = &self.memory_plane {
            if let Ok(t) = m.lock() {
                stats.memory_plane.clone_from(&t);
            }
        }
        stats
    }

    fn model_inner(&self) -> &ModelInfo {
        &self.model
    }
}

#[cfg(test)]
mod logprob_tests {
    use super::*;

    fn entry(id: TokenId) -> SampledLogprobs {
        SampledLogprobs {
            sampled: saber_serve::engine::TokenLogprob {
                token_id: id,
                logprob: -1.0,
                rank: Some(1),
            },
            top: Vec::new(),
        }
    }

    /// The stop rule can keep fewer tokens than the executor produced — an EOS in the middle of a
    /// verified step, or `max_tokens` landing inside one. The reply lines `logprobs.content` up
    /// with the text, so the list is cut to what committed.
    #[test]
    fn the_distribution_is_cut_to_what_committed() {
        let three = Some(vec![entry(1), entry(2), entry(3)]);
        let (cut, short) = logprobs_for(&three, 2);
        assert!(!short);
        let cut = cut.expect("some");
        assert_eq!(cut.len(), 2);
        assert_eq!(cut[0].sampled.token_id, 1);
        assert_eq!(cut[1].sampled.token_id, 2);
    }

    /// A request that asked for none gets none — and `None` rather than an empty list, because
    /// "the client did not ask" and "the sampler had nothing to say" are different facts.
    #[test]
    fn a_request_that_asked_for_none_gets_none() {
        let (out, short) = logprobs_for(&None, 3);
        assert!(out.is_none());
        assert!(!short);
    }

    /// NEGATIVE CONTROL: an executor that reports FEWER entries than the tokens it committed is a
    /// protocol error, not a short list. The caller counts it; this is the flag it counts on.
    #[test]
    fn fewer_entries_than_tokens_is_reported_as_a_fault() {
        let one = Some(vec![entry(9)]);
        let (out, short) = logprobs_for(&one, 3);
        assert!(short, "the caller has to count this");
        assert_eq!(out.expect("what there was").len(), 1);
    }
}
