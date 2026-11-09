//! The shutdown census, and its live twin on `/metrics` (ws-v8).
//!
//! `saber serve --engine real` prints a block of which-path lines when its listener closes
//! (`docs/SHADOW.md` §2: "no route running off a fallback counter" is read off them). Before ws-v8
//! that block was the ONLY place those counters reached: a window that crashed lost them, and a
//! fallback in the middle of a day could not be dated. Now every counter in it is also a `saber:`
//! family on `/metrics`, and this module is where the two statements are made side by side:
//!
//! * [`shutdown_census`] renders the block (the CLI prints exactly this, and nothing else, after
//!   the listener closes);
//! * [`engine_defs`] lists the executor's `/metrics` twins — [`EngineCounters::CENSUS`], the
//!   [`ENGINE_CLEAN`] gauge and [`DraftCounts::CENSUS`] — each carrying the census line and the text
//!   around the number it mirrors; the scheduler's and the admission counters carry theirs in
//!   `saber-sched` (`SchedCounters::CENSUS`, `AdmissionCounters::CENSUS`);
//! * `tests/census_parity.rs` runs the real scheduler over [`crate::NullForward`] with faults
//!   injected, scrapes `/metrics`, prints this census, and finds every `/metrics` value in the
//!   printed text by its def's pattern — counter for counter.
//!
//! The DFlash2 batched-equivalence arm's own counters (`DraftCounts::batch_*`, ws-s10) are not in
//! the census: they are a gate's diagnostic, zero on every serving boot, and printed by the gate.

use saber_sched::alloc::AdmissionCounters;
use saber_sched::SchedCounters;
use saber_serve::census::CensusDef;

use crate::device::DraftCounts;
use crate::executor::EngineCounters;

saber_serve::census_counters!(
    EngineCounters,
    CENSUS,
    census_values,
    [
        (
            prefill_steps,
            "saber:engine_prefill_steps_total",
            "Prefill chunks run.",
            "counters: prefill {} ("
        ),
        (
            silent_prefill_steps,
            "saber:engine_silent_prefill_steps_total",
            "Prefill chunks that committed nothing (not the last).",
            "counters: ({} silent)"
        ),
        (
            prefill_forward_us,
            "saber:engine_prefill_forward_us_total",
            "Wall of prefill forward calls, microseconds (C4).",
            "stepsplit: prefill forward us {} /"
        ),
        (
            prefill_after_us,
            "saber:engine_prefill_after_us_total",
            "Wall of a prefill chunk's work after its forward, microseconds (C4).",
            "stepsplit: / after us {}, decode"
        ),
        (
            decode_forward_us,
            "saber:engine_decode_forward_us_total",
            "Wall of decode forward calls, microseconds (C4).",
            "stepsplit: decode forward us {} /"
        ),
        (
            decode_after_us,
            "saber:engine_decode_after_us_total",
            "Wall of a decode step's work after its forward, microseconds (C4).",
            "stepsplit: / decode after us {}"
        ),
        (
            prefill_live_rows,
            "saber:engine_prefill_live_rows_total",
            "Prompt rows the prefill chunks carried (C5).",
            "stepsplit: prefill rows live {} /"
        ),
        (
            prefill_plan_rows,
            "saber:engine_prefill_plan_rows_total",
            "Rows the prefill chunks' planned buckets ran, padding included (C5).",
            "stepsplit: / planned {}"
        ),
        (
            mixed_fused_steps,
            "saber:engine_mixed_fused_steps_total",
            "Mixed steps run as one fused forward (C9).",
            "mixedfused: fused {} /"
        ),
        (
            mixed_fused_fallbacks,
            "saber:engine_mixed_fused_fallbacks_total",
            "Mixed steps the fused law refused, run serially (C9).",
            "mixedfused: / serial fallbacks {},"
        ),
        (
            mixed_fused_rung_rows,
            "saber:engine_mixed_fused_rung_rows_total",
            "Prefill-rung rows the fused mixed steps ran, summed (C9).",
            "mixedfused: rung rows {},"
        ),
        (
            mixed_fused_forward_us,
            "saber:engine_mixed_fused_forward_us_total",
            "Wall of the fused mixed forward calls, microseconds (C9).",
            "mixedfused: forward us {},"
        ),
        (
            mixed_packed_steps,
            "saber:engine_mixed_packed_steps_total",
            "Fused mixed steps that carried two or more prompt chunks (C10).",
            "mixedfused: packed {} /"
        ),
        (
            mixed_packed_chunks,
            "saber:engine_mixed_packed_chunks_total",
            "Prompt chunks carried by packed fused steps, summed (C10).",
            "mixedfused: / chunks {}"
        ),
        (
            decode_steps,
            "saber:engine_decode_steps_total",
            "Decode steps run (one per batch).",
            "counters: decode {} ("
        ),
        (
            decode_sequences,
            "saber:engine_decode_sequences_total",
            "Sequences stepped, summed over decode steps.",
            "counters: ({} sequences,"
        ),
        (
            padded_rows,
            "saber:engine_padded_rows_total",
            "Padding rows a captured shape ran.",
            "counters: {} padded rows)"
        ),
        (
            band_steps,
            "saber:engine_band_steps_total",
            "Decode steps run in a row band (C2 ragged widths).",
            "counters: band steps {} ("
        ),
        (
            band_live_rows,
            "saber:engine_band_live_rows_total",
            "Rows the slots fed over the band steps.",
            "counters: (live rows {} /"
        ),
        (
            band_rows,
            "saber:engine_band_rows_total",
            "Rows the band steps ran, padding included.",
            "counters: / band rows {})"
        ),
        (
            graph_steps,
            "saber:engine_graph_steps_total",
            "Steps a captured graph replayed.",
            "counters: path graph {} /"
        ),
        (
            eager_steps,
            "saber:engine_eager_steps_total",
            "Steps the eager op-by-op path ran (the fallback).",
            "counters: eager {},"
        ),
        (
            device_fed_steps,
            "saber:engine_device_fed_steps_total",
            "Steps whose ids the previous step's argmax left on the device.",
            "counters: feed device {} /"
        ),
        (
            host_fed_steps,
            "saber:engine_host_fed_steps_total",
            "Steps whose ids the host uploaded.",
            "counters: host {},"
        ),
        (
            launches,
            "saber:engine_launches_total",
            "Dispatches issued, summed over steps.",
            "counters: launches {},"
        ),
        (
            budget_mismatches,
            "saber:engine_budget_mismatches_total",
            "Steps whose launch count missed the shape's declared budget.",
            "counters: budget mismatches {},"
        ),
        (
            undeclared_shapes,
            "saber:engine_undeclared_shapes_total",
            "Steps at a shape the pass declares no budget for.",
            "counters: undeclared shapes {},"
        ),
        (
            shape_violations,
            "saber:engine_shape_violations_total",
            "Steps whose reported shape contradicted the batch.",
            "counters: shape violations {},"
        ),
        (
            forward_errors,
            "saber:engine_forward_errors_total",
            "Steps the forward refused.",
            "counters: forward errors {},"
        ),
        (
            unknown_sequences,
            "saber:engine_unknown_sequences_total",
            "Steps naming a sequence the engine never admitted.",
            "counters: unknown seqs {},"
        ),
        (
            oversized_step_refusals,
            "saber:engine_oversized_step_refusals_total",
            "Steps refused for carrying more sequences than the engine holds.",
            "counters: oversized refusals {}"
        ),
        (
            greedy_rows,
            "saber:engine_greedy_rows_total",
            "Rows committed straight from the device argmax.",
            "sampler: rows greedy {} /"
        ),
        (
            sampled_rows,
            "saber:engine_sampled_rows_total",
            "Rows the host sampler decided from the seam.",
            "sampler: sampled {} /"
        ),
        (
            dense_rows,
            "saber:engine_dense_rows_total",
            "Rows read back densely and sampled from the whole vocabulary.",
            "sampler: dense {},"
        ),
        (
            seam_refusals,
            "saber:engine_seam_refusals_total",
            "Rows the top-K seam could not decide.",
            "sampler: seam refusals {},"
        ),
        (
            penalty_lists_too_wide,
            "saber:engine_penalty_lists_too_wide_total",
            "Rows whose penalty list was wider than the seam.",
            "sampler: penalty lists too wide {},"
        ),
        (
            sampler_fallbacks_to_argmax,
            "saber:engine_argmax_fallbacks_total",
            "Rows a sampler error left on the argmax (must be 0).",
            "sampler: argmax fallbacks {},"
        ),
        (
            logprobs_skipped_on_verify,
            "saber:engine_logprobs_skipped_on_verify_total",
            "Verify steps whose logprobs could not be reported (must be 0).",
            "sampler: logprobs skipped on a verify step {},"
        ),
        (
            top_ties,
            "saber:engine_top_ties_total",
            "Drawn rows with an exact tie at the top of the logit row (ws-f39).",
            "sampler: ties at the top {}"
        ),
        (
            top_tie_draws,
            "saber:engine_top_tie_draws_total",
            "Draws that committed a tied id other than the argmax's (ws-f39).",
            "sampler: (drawn off the argmax {})"
        ),
        (
            step_keyed_rows,
            "saber:engine_step_keyed_rows_total",
            "Rows drawn under the pre-F40 step-keyed control (must be 0 serving; ws-f40).",
            "sampler: step-keyed control rows {}"
        ),
        (
            pass_proposals,
            "saber:engine_pass_proposals_total",
            "Proposals the pass made and offered.",
            "proposal: pass proposals {} offered"
        ),
        (
            proposals_after_commit,
            "saber:engine_proposals_after_commit_total",
            "Proposals made after the host sampler's commit.",
            "proposal: ({} made after the commit)"
        ),
        (
            pass_proposals_stale,
            "saber:engine_stale_anchor_drops_total",
            "In-step proposals dropped for a stale anchor (a rate above T=0).",
            "proposal: {} dropped with a STALE ANCHOR"
        ),
        (
            stale_anchor_defects,
            "saber:engine_stale_anchor_defects_total",
            "After-commit blocks whose anchor was not the committed token (must be 0).",
            "proposal: {} stale-anchor DEFECT(s)"
        ),
        (
            proposal_collect_failures,
            "saber:engine_proposal_collect_failures_total",
            "Steps whose stashed proposal could not be collected (must be 0).",
            "proposal: {} collection failure(s)"
        ),
        (
            stale_anchor_drafts,
            "saber:engine_stale_anchor_drafts_total",
            "Stale-anchor blocks verified (the control only).",
            "proposal: stale blocks verified {} for"
        ),
        (
            stale_anchor_accepted,
            "saber:engine_stale_anchor_accepted_total",
            "Tokens accepted out of stale-anchor blocks (the control only).",
            "proposal: for {} accepted token(s)"
        ),
        (
            lookup_lookups,
            "saber:engine_lookup_lookups_total",
            "FN69 prompt-lookup queries (one per sequence per drafting point).",
            "lookup: lookups {}, hits"
        ),
        (
            lookup_hits,
            "saber:engine_lookup_hits_total",
            "FN69 prompt-lookup queries that found an earlier occurrence.",
            "lookup: hits {}, used"
        ),
        (
            lookup_used,
            "saber:engine_lookup_used_total",
            "FN69 drafting points the merge law gave to the lookup.",
            "lookup: used {}, lookup"
        ),
        (
            lookup_drafted,
            "saber:engine_lookup_drafted_total",
            "FN69 draft tokens verified from the lookup.",
            "lookup: lookup drafted {} /"
        ),
        (
            lookup_accepted,
            "saber:engine_lookup_accepted_total",
            "FN69 draft tokens accepted from the lookup.",
            "lookup: accepted {} (from the lookup)"
        ),
        (
            lookup_drafter_drafted,
            "saber:engine_lookup_drafter_drafted_total",
            "FN69 draft tokens verified from the drafter (lookup on).",
            "lookup: drafter drafted {} /"
        ),
        (
            lookup_drafter_accepted,
            "saber:engine_lookup_drafter_accepted_total",
            "FN69 draft tokens accepted from the drafter (lookup on).",
            "lookup: accepted {} (from the drafter)"
        ),
        (
            prompt_logprob_chunks,
            "saber:engine_prompt_logprob_chunks_total",
            "Prefill chunks scored for prompt_logprobs.",
            "prompt: chunks {},"
        ),
        (
            prompt_logprob_rows,
            "saber:engine_prompt_logprob_rows_total",
            "Prompt rows scored.",
            "prompt: rows {},"
        ),
        (
            prompt_logprob_refusals,
            "saber:engine_prompt_logprob_refusals_total",
            "Chunks whose prompt rows could not be scored.",
            "prompt: refused {}"
        ),
        (
            draft_tokens,
            "saber:engine_draft_tokens_total",
            "Draft tokens proposed.",
            "spec: draft tokens {},"
        ),
        (
            accepted_tokens,
            "saber:engine_accepted_tokens_total",
            "Draft tokens accepted (no bonus).",
            "spec: accepted {},"
        ),
        (
            committed_tokens,
            "saber:engine_committed_tokens_total",
            "Tokens committed across every slot and step.",
            "spec: committed {},"
        ),
        (
            draft_overruns,
            "saber:engine_draft_overruns_total",
            "Proposals wider than the verify width, truncated.",
            "spec: overruns {}"
        ),
        (
            parks,
            "saber:engine_parks_total",
            "Sessions copied off the device as a park image.",
            "session: parks {},"
        ),
        (
            park_failures,
            "saber:engine_park_failures_total",
            "Park attempts the forward refused (must be 0).",
            "session: park refusals {},"
        ),
        (
            restores,
            "saber:engine_restores_total",
            "Session images put back.",
            "session: restores {},"
        ),
        (
            restore_failures,
            "saber:engine_restore_failures_total",
            "Restore attempts refused (must be 0).",
            "session: restore refusals {}"
        ),
        (
            vision_images,
            "saber:engine_vision_images_total",
            "Images whose tower ran at admission (D3).",
            "vision: images {},"
        ),
        (
            vision_patches,
            "saber:engine_vision_patches_total",
            "Patches the tower ran over.",
            "vision: patches {},"
        ),
        (
            vision_tokens,
            "saber:engine_vision_tokens_total",
            "Image-token rows the tower left in its arena.",
            "vision: tokens {},"
        ),
        (
            vision_runs,
            "saber:engine_vision_runs_total",
            "Tower dispatches (one per image).",
            "vision: tower runs {},"
        ),
        (
            vision_seqs,
            "saber:engine_vision_sequences_total",
            "Image sequences admitted with M-RoPE positions and a scatter (D4).",
            "vision: sequences {},"
        ),
        (
            vision_failures,
            "saber:engine_vision_failures_total",
            "Image requests refused or failed at the tower (must be 0 on a vision boot).",
            "vision: failures {}"
        ),
        (
            parked_discards,
            "saber:engine_parked_discards_total",
            "Unrestored images whose hold on this engine was released.",
            "parking: (engine discards {},"
        ),
        (
            parked_discards_held,
            "saber:engine_parked_discards_held_total",
            "Of those, the ones some rank still held.",
            "parking: held {})"
        ),
        (
            restores_refused_by_injection,
            "saber:engine_restores_refused_by_injection_total",
            "Restores refused by the bounded-restore control (0 when serving).",
            "parking: injected restore refusals {}"
        ),
        (
            forks,
            "saber:engine_forks_total",
            "Branches forked from a leader's sampled row.",
            "forks: forks {} ("
        ),
        (
            fork_refusals,
            "saber:engine_fork_refusals_total",
            "Forks refused (must be 0).",
            "forks: ({} refused)"
        ),
        (
            fork_blocks,
            "saber:engine_fork_first_blocks_total",
            "Forked branches whose first drafter block was proposed from their own token (ws-f38).",
            "forks: first blocks {} ("
        ),
        (
            fork_block_failures,
            "saber:engine_fork_first_block_failures_total",
            "Forked branches under a drafter whose first block could not be made (must be 0).",
            "forks: ({} failed)"
        ),
    ]
);

saber_serve::census_counters!(
    DraftCounts,
    CENSUS,
    census_values,
    [
        (
            proposals,
            "saber:draft_proposals_total",
            "Proposals the engine's drafter ran.",
            "drafted: proposals {} ("
        ),
        (
            refused,
            "saber:draft_refused_total",
            "Proposals the index law refused.",
            "drafted: ({} refused,"
        ),
        (
            out_of_range,
            "saber:draft_out_of_range_total",
            "Proposals dropped for an out-of-vocabulary id.",
            "drafted: {} out of range)"
        ),
        (
            tokens,
            "saber:draft_tokens_total",
            "Draft tokens handed to the mailbox.",
            "drafted: {} draft token(s)"
        ),
        (
            accepted,
            "saber:draft_accepted_total",
            "Draft tokens accepted on steps the drafter proposed for.",
            "drafted: {} accepted of"
        ),
        (
            verified,
            "saber:draft_verified_total",
            "Draft tokens verified on those steps.",
            "drafted: of {} verified"
        ),
        (
            eager_launches,
            "saber:draft_eager_launches_total",
            "Eager context-store dispatches.",
            "drafted: {} eager launch(es)"
        ),
        (
            replayed_nodes,
            "saber:draft_replayed_nodes_total",
            "Captured backbone nodes replayed.",
            "drafted: {} replayed node(s)"
        ),
        (
            pages_cleared,
            "saber:draft_pages_cleared_total",
            "Admissions whose drafter pages were zeroed.",
            "drafted: pages cleared {} /"
        ),
        (
            pages_dirty,
            "saber:draft_pages_dirty_total",
            "Admissions that took their pages as found (the control).",
            "drafted: left dirty {};"
        ),
        (
            ctx_tiles,
            "saber:draft_ctx_tiles_total",
            "Tiles of the prompt's context stored.",
            "drafted: {} tile(s)"
        ),
        (
            ctx_rows,
            "saber:draft_ctx_rows_total",
            "Live prompt rows stored into the drafter's cache.",
            "drafted: {} row(s)"
        ),
        (
            prefill_proposals,
            "saber:draft_prefill_proposals_total",
            "Last prefill chunks that made the first proposal.",
            "drafted: first proposals {} /"
        ),
        (
            prefill_silent,
            "saber:draft_prefill_silent_total",
            "Last prefill chunks that proposed nothing (the control).",
            "drafted: silent last chunks {};"
        ),
        (
            after_commit,
            "saber:draft_after_commit_total",
            "Proposals made after the commit (the default).",
            "drafted: proposed after the commit {} /"
        ),
        (
            in_step,
            "saber:draft_in_step_total",
            "Proposals made inside the step (the control).",
            "drafted: in the step {},"
        ),
        (
            uncollected,
            "saber:draft_uncollected_total",
            "Stashed proposals never collected (must be 0).",
            "drafted: UNCOLLECTED {}"
        ),
    ]
);

/// `EngineCounters::clean()`, as a gauge: 1 while every step the executor ran was clean.
pub const ENGINE_CLEAN: CensusDef = CensusDef::gauge(
    "saber:engine_clean",
    "1 while EngineCounters::clean() holds (no budget, shape, forward, sampler, fork, park or \
     proposal fault), 0 after the first one.",
    "clean: engine clean {}",
);

/// The executor's `/metrics` census, in publish order: the engine's counters, the clean gauge, the
/// drafter's counters. [`crate::RealExecutor`] publishes into a mirror over exactly these.
pub fn engine_defs() -> Vec<CensusDef> {
    let mut defs = EngineCounters::CENSUS.to_vec();
    defs.push(ENGINE_CLEAN);
    defs.extend_from_slice(DraftCounts::CENSUS);
    defs
}

/// Writes the executor's census values in [`engine_defs`] order.
pub fn engine_values(ec: &EngineCounters, dc: &DraftCounts, put: &mut dyn FnMut(u64)) {
    ec.census_values(put);
    put(u64::from(ec.clean()));
    dc.census_values(put);
}

/// The block `saber serve --engine real` prints when its listener closes. Every number in it is
/// on `/metrics` under the name the defs give it; `docs/SHADOW.md` §2 has the table.
///
/// `dc` is the drafter's counters, `None` on a runner that keeps none (the line then reads zeros,
/// as the `/metrics` twins do).
pub fn shutdown_census(
    sc: &SchedCounters,
    ac: &AdmissionCounters,
    ec: &EngineCounters,
    dc: Option<&DraftCounts>,
) -> String {
    let dc = dc.copied().unwrap_or_default();
    let mut out = String::new();
    let mut line = |tag: &str, body: String| {
        out.push_str(&format!("{tag:<15}{body}\n"));
    };
    // V7: zero on a serve nobody asked for prompt logprobs on.
    line("prompt", ec.prompt_line());
    line("counters", ec.line());
    // C4: where a step's wall goes, forward call vs the executor's work after it.
    line(
        "stepsplit",
        format!(
            "prefill forward us {} / after us {}, decode forward us {} / decode after us {}, \
             prefill rows live {} / planned {}",
            ec.prefill_forward_us,
            ec.prefill_after_us,
            ec.decode_forward_us,
            ec.decode_after_us,
            ec.prefill_live_rows,
            ec.prefill_plan_rows
        ),
    );
    // C9: the fused mixed forward.
    line(
        "mixedfused",
        format!(
            "fused {} / serial fallbacks {}, rung rows {}, forward us {}, packed {} / chunks {}",
            ec.mixed_fused_steps,
            ec.mixed_fused_fallbacks,
            ec.mixed_fused_rung_rows,
            ec.mixed_fused_forward_us,
            ec.mixed_packed_steps,
            ec.mixed_packed_chunks
        ),
    );
    // WHICH BRANCH DECIDED THE TOKENS. A serve that sampled every row and one that fell back to the
    // argmax on all of them print the same everything else (`CLAUDE.md` rule 5).
    line("sampler", ec.sampler_line());
    // AND WHICH ROUTE THE PROPOSAL TOOK (ws-f35/f37).
    line("proposal", ec.proposal_line());
    // FN69: the prompt-lookup source (all zero with `--spec-lookup off`).
    line("lookup", ec.lookup_line());
    line(
        "spec",
        format!(
            "draft tokens {}, accepted {}, committed {}, overruns {}",
            ec.draft_tokens, ec.accepted_tokens, ec.committed_tokens, ec.draft_overruns
        ),
    );
    line(
        "session",
        format!(
            "parks {}, park refusals {}, restores {}, restore refusals {}",
            ec.parks, ec.park_failures, ec.restores, ec.restore_failures
        ),
    );
    line("forks", ec.fork_line());
    line(
        "clean",
        format!(
            "engine clean {} (EngineCounters::clean: every census field that must read 0 does)",
            u64::from(ec.clean())
        ),
    );
    line("drafted", dc.line());
    // THE PROMPT'S CONTEXT STORE, wide and narrow, and its check (H35).
    line("store", dc.store_line());
    // AND WHETHER ANYTHING WAS EVER PARKED (ws-k14), and why a park did or did not happen (ws-k17).
    line(
        "parking",
        format!(
            "parks {} (policy {}, refused {}), restores {} (failed {}, queued {}), transfers \
             started {}/{} busy {} failed {}, steps during a transfer {}, sessions parked again \
             {}, restores given up {}, dropped unrestored {} (engine discards {}, held {}), \
             injected restore refusals {}",
            sc.parks,
            sc.policy_parks,
            sc.park_refusals,
            sc.restores,
            sc.restore_failures,
            sc.policy_restores_queued,
            sc.parks_started,
            sc.restores_started,
            sc.park_transfer_busy,
            sc.park_transfer_failures,
            sc.steps_during_transfer,
            sc.reparks,
            sc.restores_given_up,
            sc.parked_discards,
            ec.parked_discards,
            ec.parked_discards_held,
            ec.restores_refused_by_injection,
        ),
    );
    line(
        "vision",
        format!(
            "images {}, patches {}, tokens {}, tower runs {}, sequences {}, failures {}",
            ec.vision_images,
            ec.vision_patches,
            ec.vision_tokens,
            ec.vision_runs,
            ec.vision_seqs,
            ec.vision_failures
        ),
    );
    line(
        "admission",
        format!(
            "admitted {}, refused kv {} / gdn {} / batch {}, policy misses {} (a KV or GDN refusal \
             with no eligible victim), waits on a park in flight {}, kv rollbacks {}",
            ac.admitted,
            ac.refused_kv,
            ac.refused_gdn,
            ac.refused_batch,
            sc.policy_park_misses,
            sc.policy_park_waits,
            ac.kv_rollbacks,
        ),
    );
    line(
        "sched",
        format!(
            "steps {}, idle {}, no bucket {}, protocol errors {}, token count errors {}, aborts \
             running {} / waiting {} / parked {}, finished stop {} / length {}, results for gone \
             seqs {}, retires deferred {}, prefill steps {}, computed prompt tokens {}, \
             committed output tokens {}, prefill bucket misses {}, vision \
             waits {}, forks \
             pending {} / done {} / failed {} / orphaned {}, fork first-token logprobs absent {}, \
             slot caps absorbed {}, width narrowed steps {}, width raised to a previous \
             commit {}, ragged steps {} (in a rectangle {}), live rows {} / run {}, slots \
             narrowed by the cap {}",
            sc.steps,
            sc.idle_steps,
            sc.no_bucket_steps,
            sc.executor_protocol_errors,
            sc.executor_token_count_errors,
            sc.aborts_running,
            sc.aborts_waiting,
            sc.aborts_parked,
            sc.finished_stop,
            sc.finished_length,
            sc.results_for_gone_sequences,
            sc.retires_deferred,
            sc.prefill_steps,
            sc.prefill_computed_tokens,
            sc.output_committed_tokens,
            sc.prefill_bucket_misses,
            sc.vision_waits,
            sc.forks_pending,
            sc.forks,
            sc.fork_failures,
            sc.fork_branches_orphaned,
            sc.fork_first_token_logprobs_absent,
            sc.slot_caps_absorbed,
            sc.width_narrowed_steps,
            sc.width_floor_raised,
            sc.ragged_steps,
            sc.ragged_rect_steps,
            sc.ragged_live_rows,
            sc.ragged_step_rows,
            sc.ragged_slots_narrowed,
        ),
    );
    line(
        "timeline",
        format!(
            "mixed steps {} ({} chunk rows, budget spent {}, wall us {}), prefill alone {} \
             (wall us {}), decode wall us {}, stalled decoder slot-steps {} / slot-us {}, yield decode steps {}",
            sc.mixed_steps,
            sc.mixed_chunk_rows,
            sc.mixed_budget_spent,
            sc.mixed_step_us,
            sc.prefill_alone_steps,
            sc.prefill_alone_us,
            sc.decode_step_us,
            sc.stalled_decoder_steps,
            sc.stalled_decoder_us,
            sc.yield_decode_steps,
        ),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_serve::census::find_in_census;

    /// Every def's pattern finds ITS field in a census where every field holds a distinct value:
    /// the pattern names the right number, not merely a number. A pattern that pointed at a
    /// neighbour would read the neighbour's (different) value and fail here.
    #[test]
    fn every_pattern_reads_its_own_field_from_distinct_values() {
        let mut next = 1000u64;
        let mut bump = || {
            next += 7;
            next
        };
        let mut ec = EngineCounters::default();
        macro_rules! fill {
            ($s:ident, [$($f:ident),*]) => { $( $s.$f = bump(); )* };
        }
        fill!(
            ec,
            [
                prefill_steps,
                silent_prefill_steps,
                prefill_forward_us,
                prefill_after_us,
                decode_forward_us,
                decode_after_us,
                prefill_live_rows,
                prefill_plan_rows,
                mixed_fused_steps,
                mixed_fused_fallbacks,
                mixed_fused_rung_rows,
                mixed_fused_forward_us,
                mixed_packed_steps,
                mixed_packed_chunks,
                decode_steps,
                decode_sequences,
                padded_rows,
                band_steps,
                band_live_rows,
                band_rows,
                graph_steps,
                eager_steps,
                device_fed_steps,
                host_fed_steps,
                launches,
                budget_mismatches,
                undeclared_shapes,
                shape_violations,
                forward_errors,
                unknown_sequences,
                oversized_step_refusals,
                greedy_rows,
                sampled_rows,
                dense_rows,
                seam_refusals,
                penalty_lists_too_wide,
                sampler_fallbacks_to_argmax,
                logprobs_skipped_on_verify,
                top_ties,
                top_tie_draws,
                step_keyed_rows,
                pass_proposals,
                proposals_after_commit,
                pass_proposals_stale,
                stale_anchor_defects,
                proposal_collect_failures,
                stale_anchor_drafts,
                stale_anchor_accepted,
                lookup_lookups,
                lookup_hits,
                lookup_used,
                lookup_drafted,
                lookup_accepted,
                lookup_drafter_drafted,
                lookup_drafter_accepted,
                prompt_logprob_chunks,
                prompt_logprob_rows,
                prompt_logprob_refusals,
                draft_tokens,
                accepted_tokens,
                committed_tokens,
                draft_overruns,
                parks,
                park_failures,
                restores,
                restore_failures,
                parked_discards,
                parked_discards_held,
                restores_refused_by_injection,
                vision_images,
                vision_patches,
                vision_tokens,
                vision_runs,
                vision_seqs,
                vision_failures,
                forks,
                fork_refusals,
                fork_blocks,
                fork_block_failures
            ]
        );
        let mut dc = DraftCounts::default();
        fill!(
            dc,
            [
                proposals,
                refused,
                out_of_range,
                tokens,
                accepted,
                verified,
                eager_launches,
                replayed_nodes,
                pages_cleared,
                pages_dirty,
                ctx_tiles,
                ctx_rows,
                prefill_proposals,
                prefill_silent,
                after_commit,
                in_step,
                uncollected
            ]
        );
        let mut sc = SchedCounters::default();
        fill!(
            sc,
            [
                steps,
                idle_steps,
                no_bucket_steps,
                executor_protocol_errors,
                executor_token_count_errors,
                aborts_running,
                aborts_waiting,
                aborts_parked,
                finished_stop,
                finished_length,
                results_for_gone_sequences,
                retires_deferred,
                prefill_steps,
                prefill_computed_tokens,
                output_committed_tokens,
                prefill_bucket_misses,
                vision_waits,
                forks_pending,
                forks,
                fork_failures,
                fork_branches_orphaned,
                fork_first_token_logprobs_absent,
                slot_caps_absorbed,
                width_narrowed_steps,
                width_floor_raised,
                mixed_steps,
                mixed_chunk_rows,
                mixed_budget_spent,
                prefill_alone_steps,
                stalled_decoder_steps,
                stalled_decoder_us,
                prefill_alone_us,
                decode_step_us,
                mixed_step_us,
                parks,
                policy_parks,
                park_refusals,
                restores,
                restore_failures,
                policy_restores_queued,
                parks_started,
                restores_started,
                park_transfer_busy,
                park_transfer_failures,
                steps_during_transfer,
                reparks,
                restores_given_up,
                parked_discards,
                policy_park_misses,
                policy_park_waits
            ]
        );
        let mut ac = AdmissionCounters::default();
        fill!(
            ac,
            [
                admitted,
                refused_batch,
                refused_kv,
                refused_gdn,
                kv_rollbacks
            ]
        );

        let text = shutdown_census(&sc, &ac, &ec, Some(&dc));
        let mut checked = 0;
        let mut names = std::collections::BTreeSet::new();
        let mut check = |defs: &[CensusDef], values: Vec<u64>| {
            assert_eq!(defs.len(), values.len());
            for (def, want) in defs.iter().zip(values) {
                assert!(
                    names.insert(def.name),
                    "duplicate /metrics name {}",
                    def.name
                );
                assert!(def.name.starts_with("saber:"), "{}", def.name);
                assert_eq!(
                    find_in_census(&text, def),
                    Some(want),
                    "{} ({}) in:\n{text}",
                    def.name,
                    def.census
                );
                checked += 1;
            }
        };
        let mut v = Vec::new();
        engine_values(&ec, &dc, &mut |x| v.push(x));
        check(&engine_defs(), v);
        let mut v = Vec::new();
        sc.census_values(&mut |x| v.push(x));
        check(SchedCounters::CENSUS, v);
        let mut v = Vec::new();
        ac.census_values(&mut |x| v.push(x));
        check(AdmissionCounters::CENSUS, v);
        assert_eq!(
            checked,
            engine_defs().len() + SchedCounters::CENSUS.len() + 5
        );
    }

    /// The clean gauge follows `clean()`, and one fault turns it off.
    #[test]
    fn the_clean_gauge_is_the_structural_gates_question() {
        let text = shutdown_census(
            &SchedCounters::default(),
            &AdmissionCounters::default(),
            &EngineCounters::default(),
            None,
        );
        assert_eq!(find_in_census(&text, &ENGINE_CLEAN), Some(1));
        let ec = EngineCounters {
            sampler_fallbacks_to_argmax: 1,
            ..EngineCounters::default()
        };
        let text = shutdown_census(
            &SchedCounters::default(),
            &AdmissionCounters::default(),
            &ec,
            None,
        );
        assert_eq!(find_in_census(&text, &ENGINE_CLEAN), Some(0));
    }
}
