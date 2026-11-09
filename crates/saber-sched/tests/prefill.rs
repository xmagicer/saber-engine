//! Prefill scheduling: the prompt gets its own bucket class, and its own step.
//!
//! Before P2 a request's first step was a decode step that happened to feed the whole prompt —
//! one row of a decode shape carrying two thousand tokens. The planner now captures a **prefill**
//! bucket class (`Bucket::prefill`, `Phase::Prefill`), which is a different `M`, different ops and
//! a different GDN state budget, and this is the scheduler using it:
//!
//! * the prompt is cut by the rule in [`saber_sched::PrefillChunking`] — as much of what is left
//!   as the smallest fitting bucket will take;
//! * a prefill step runs **one** sequence: the two phases are different captured shapes, so a
//!   batch cannot mix them;
//! * a chunk that is not the last **samples nothing**, so a client hears nothing until the prompt
//!   is in and `TTFT` covers every prefill step;
//! * a prefilling sequence touches **one** GDN state slot, not `spec_width + 1` — the planner's
//!   own split, honoured in the accounting.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{boot_ledger, decode_buckets, plan_with, plan_with_prefill, GIB, MIB};
use saber_plan::Phase;
use saber_sched::{
    ArenaBytes, KvGeometry, PrefillChunking, Scheduler, SchedulerConfig, ScriptExecutor,
};
use saber_serve::detok::{ByteCodec, TextCodec};
use saber_serve::engine::{Engine, EngineRequest, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

const EOS: TokenId = 255;
const TOKENS_PER_STEP: u32 = 5;
/// The prod shapes: 2048 is the measured chunk, 8192 is `--max-num-batched-tokens`.
const PREFILL: [(u32, u32); 2] = [(2048, 65536), (8192, 65536)];

fn answer() -> Vec<TokenId> {
    let mut v = ByteCodec
        .encode("The prompt is in, and this is what comes after it.")
        .expect("encode");
    v.push(EOS);
    v
}

fn roomy() -> ArenaBytes {
    ArenaBytes {
        kv_per_rank: 8 * GIB,
        gdn_state_per_rank: 8 * GIB,
    }
}

/// A scheduler whose plan carries `prefill` buckets (empty = the pre-P2 path).
fn sched_with(prefill: &[(u32, u32)]) -> Arc<Scheduler<ScriptExecutor>> {
    sched_busy(prefill, 0)
}

/// [`sched_with`] with FN38's chunk cap while something decodes (0 = off).
fn sched_busy(prefill: &[(u32, u32)], busy: u32) -> Arc<Scheduler<ScriptExecutor>> {
    sched_yield(prefill, busy, 0)
}

/// [`sched_busy`] with LR-B's decode steps between two chunks (0 = off).
fn sched_yield(prefill: &[(u32, u32)], busy: u32, yields: u32) -> Arc<Scheduler<ScriptExecutor>> {
    // A wide context band, because a chunked prompt is a long one.
    let decode: Vec<(u32, u32, u32)> = decode_buckets()
        .into_iter()
        .filter(|(_, _, band)| *band == 65536)
        .collect();
    let plan = if prefill.is_empty() {
        plan_with(&decode, 2)
    } else {
        plan_with_prefill(&decode, prefill, 2)
    };
    let mut config =
        SchedulerConfig::from_plan(&plan, KvGeometry::PROD_FP8, roomy(), 262_144).expect("config");
    config.prefill = config
        .prefill
        .clone()
        .with_busy_max(busy)
        .with_yield_steps(yields);
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    Arc::new(
        Scheduler::new(
            config,
            ScriptExecutor::new(vec![answer()], TOKENS_PER_STEP, EOS),
            model,
            boot_ledger(),
        )
        .expect("arenas fit"),
    )
}

fn request(prompt_tokens: usize, max_tokens: usize) -> EngineRequest {
    EngineRequest {
        // A prompt of distinguishable tokens, so a chunk that fed the wrong slice is visible.
        prompt_token_ids: (0..prompt_tokens)
            .map(|i| (i % 200) as TokenId + 1)
            .collect(),
        params: SamplingParams {
            max_tokens: Some(max_tokens),
            ..SamplingParams::default()
        },
    }
}

async fn stream(submission: saber_serve::engine::Submission) -> Vec<StepEvent> {
    use futures_util::StreamExt;
    submission.events.collect().await
}

fn tokens_of(events: &[StepEvent]) -> Vec<TokenId> {
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}

// ── the rule itself ─────────────────────────────────────────────────────────────

#[test]
fn the_chunking_rule_is_data() {
    let rule = PrefillChunking::from_sizes(vec![2048, 8192]);
    // As much of what is left as the smallest fitting bucket will take.
    assert_eq!(rule.chunk_for(234), Some((234, 2048)));
    assert_eq!(rule.chunk_for(2048), Some((2048, 2048)));
    assert_eq!(rule.chunk_for(2049), Some((2049, 8192)));
    assert_eq!(
        rule.chunk_for(5000),
        Some((5000, 8192)),
        "a fitting prompt is not cut"
    );
    assert_eq!(
        rule.chunk_for(12000),
        Some((8192, 8192)),
        "over the top bucket, cut there"
    );
    assert_eq!(rule.chunk_for(12000 - 8192), Some((3808, 8192)));
    assert_eq!(rule.steps_for(12000), 2);
    assert_eq!(rule.steps_for(234), 1);
    assert_eq!(PrefillChunking::from_sizes(vec![]).chunk_for(10), None);

    // One size behaves like the reference server's chunked prefill at a fixed `--max-num-batched-tokens`.
    let one = PrefillChunking::from_sizes(vec![2048]);
    assert_eq!(one.chunk_for(5000), Some((2048, 2048)));
    assert_eq!(one.steps_for(5000), 3);
}

/// C7's fill rule: the ladder plus G4's `4096,8192` rungs with those two FILL-ONLY. Every prompt's
/// chunk sequence, against the padded law over the same rungs (G4's +76 % at 4,000 rows) and
/// against the ladder alone (today's default).
#[test]
fn the_fill_rule_cuts_long_prompts_into_filled_rungs() {
    let ladder = vec![96u32, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];
    let mut big = ladder.clone();
    big.extend([4096, 8192]);
    let today = PrefillChunking::from_sizes(ladder.clone());
    let padded = PrefillChunking::from_sizes(big.clone());
    let fill = PrefillChunking::from_sizes(big).with_fill_above(2048);
    assert_eq!(fill.padded_max(), 2048);
    assert_eq!(padded.padded_max(), 8192);

    // Short prompts: all three laws agree (the rungs above 2048 never reach them).
    for n in [1u32, 72, 257, 1000, 2047, 2048] {
        assert_eq!(fill.cut(n), today.cut(n), "{n} rows");
        assert_eq!(fill.cut(n), padded.cut(n), "{n} rows");
    }
    // 4,000 rows: padded = ONE p4096 chunk (at `+8192` alone it was one p8192, G4's +76 %);
    // fill = the ladder's two chunks.
    assert_eq!(padded.cut(4000), vec![(4000, 4096)]);
    assert_eq!(fill.cut(4000), vec![(2048, 2048), (1952, 2048)]);
    assert_eq!(fill.cut(4000), today.cut(4000));
    // 4,096 rows: one filled p4096.
    assert_eq!(fill.cut(4096), vec![(4096, 4096)]);
    assert_eq!(today.cut(4096), vec![(2048, 2048), (2048, 2048)]);
    // 6,000 = 4096 + 1904 (p2048); padded: one 8192; today: 2048 x 2 + 1904.
    assert_eq!(fill.cut(6000), vec![(4096, 4096), (1904, 2048)]);
    assert_eq!(padded.cut(6000), vec![(6000, 8192)]);
    assert_eq!(today.steps_for(6000), 3);
    // 8,192 / 16,384: filled p8192s.
    assert_eq!(fill.cut(8192), vec![(8192, 8192)]);
    assert_eq!(fill.cut(16384), vec![(8192, 8192), (8192, 8192)]);
    // 32,000 = 8192 x 3 + 4096 + 2048 + 1280 (p1536): six chunks where today runs sixteen and
    // the padded law four (the last a 7424-row p8192).
    assert_eq!(
        fill.cut(32000),
        vec![
            (8192, 8192),
            (8192, 8192),
            (8192, 8192),
            (4096, 4096),
            (2048, 2048),
            (1280, 1536)
        ]
    );
    assert_eq!(today.steps_for(32000), 16);
    assert_eq!(padded.cut(32000).last(), Some(&(7424, 8192)));
    // Every filled chunk FILLS its rung, and no chunk above 2048 rows is padded.
    for n in (1..=40_000u32).step_by(97) {
        let cut = fill.cut(n);
        assert_eq!(cut.iter().map(|(c, _)| c).sum::<u32>(), n);
        for (c, b) in cut {
            assert!(b >= c && (b <= 2048 || b == c), "{n}: chunk {c} in p{b}");
        }
    }
    // CONTROL: the fill rule over the ladder alone (top <= 2048) IS today's law, prompt by prompt.
    let ctl = PrefillChunking::from_sizes(ladder).with_fill_above(2048);
    for n in (1..=40_000u32).step_by(89) {
        assert_eq!(ctl.cut(n), today.cut(n), "{n} rows");
    }
    // And the fill rule OFF over the big rungs is the padded law.
    assert_eq!(fill.clone().with_fill_above(0), padded);
}

// ── through the scheduler ───────────────────────────────────────────────────────

#[tokio::test]
async fn a_short_prompt_is_one_prefill_chunk_then_decode() {
    let sched = sched_with(&PREFILL);
    let submission = sched.submit(request(234, 100)).expect("submit");
    sched.step();
    assert_eq!(sched.counters().prefill_computed_tokens, 234);
    assert_eq!(sched.counters().output_committed_tokens, 1);
    sched.run_steps(200);
    let events = stream(submission).await;

    let batches = sched.with_executor(|e| e.batches().to_vec());
    let prefill: Vec<_> = batches
        .iter()
        .filter(|b| b.phase == Phase::Prefill)
        .collect();
    assert_eq!(prefill.len(), 1, "one chunk for a 234-token prompt");
    let first = prefill[0];
    assert_eq!(first.bucket.phase, Phase::Prefill);
    assert_eq!(first.bucket.tokens, 2048, "the smallest fitting bucket");
    assert_eq!(first.bucket.m(), 2048, "a prefill bucket's M is its chunk");
    assert_eq!(first.sequences, 1, "one sequence per prefill step");
    assert_eq!(
        first.verify_widths,
        vec![0],
        "a prompt has nothing to speculate on"
    );
    let chunk = first.chunks[0].expect("a prefill slot carries its chunk");
    assert_eq!(chunk.prompt_tokens, 234);
    assert_eq!(chunk.tokens_done_before, 0);
    assert!(chunk.last);

    // Everything after it is a decode step, and the client heard nothing until the first token.
    assert!(batches[1..].iter().all(|b| b.phase == Phase::Decode));
    assert_eq!(events[0].step_index, 0);
    assert_eq!(
        events[0].new_tokens.len(),
        1,
        "the prefill yields one token"
    );
    assert_eq!(tokens_of(&events), answer());
    assert_eq!(sched.counters().prefill_steps, 1);
    assert_eq!(
        sched.counters().output_committed_tokens,
        tokens_of(&events).len() as u64
    );
    assert_eq!(sched.counters().prefill_bucket_misses, 0);
}

#[tokio::test]
async fn a_5000_token_prompt_is_one_chunk_and_a_12000_token_one_is_two() {
    // 5000 fits the p8192 shape, so it is not cut: prefill is the phase where M is the point.
    let sched = sched_with(&PREFILL);
    let submission = sched.submit(request(5000, 20)).expect("submit");
    sched.run_steps(200);
    let _ = stream(submission).await;
    let batches = sched.with_executor(|e| e.batches().to_vec());
    let prefill: Vec<_> = batches
        .iter()
        .filter(|b| b.phase == Phase::Prefill)
        .collect();
    assert_eq!(prefill.len(), 1);
    assert_eq!(prefill[0].bucket.tokens, 8192);
    assert_eq!(prefill[0].chunks[0].expect("chunk").prompt_tokens, 5000);

    // 12000 does not, so it is 8192 + 3808 with the KV pages and the state carried between them.
    let sched = sched_with(&PREFILL);
    let submission = sched.submit(request(12000, 20)).expect("submit");
    sched.run_steps(400);
    let events = stream(submission).await;
    let batches = sched.with_executor(|e| e.batches().to_vec());
    let prefill: Vec<_> = batches
        .iter()
        .filter(|b| b.phase == Phase::Prefill)
        .collect();
    assert_eq!(prefill.len(), 2, "two chunks");
    let (a, b) = (
        prefill[0].chunks[0].expect("chunk"),
        prefill[1].chunks[0].expect("chunk"),
    );
    assert_eq!(
        (a.prompt_tokens, a.tokens_done_before, a.last),
        (8192, 0, false)
    );
    assert_eq!(
        (b.prompt_tokens, b.tokens_done_before, b.last),
        (3808, 8192, true)
    );
    assert_eq!(
        a.prompt_tokens + b.prompt_tokens,
        12000,
        "every prompt token is fed exactly once"
    );
    // Carried: the same sequence, the same pages, the same state slot set.
    assert_eq!(prefill[0].verify_widths, prefill[1].verify_widths);
    assert_eq!(sched.counters().prefill_steps, 2);
    // And nothing was said until the prompt was in.
    assert_eq!(events[0].step_index, 0);
    assert_eq!(events[0].new_tokens.len(), 1);
}

#[tokio::test]
async fn chunking_does_not_change_the_answer() {
    // The same prompt and the same executor, once through two prefill chunks and once through the
    // pre-P2 path (no prefill bucket, one decode-shaped step feeding everything).
    let chunked = sched_with(&PREFILL);
    let submission = chunked.submit(request(12000, 30)).expect("submit");
    chunked.run_steps(400);
    let with_chunks = tokens_of(&stream(submission).await);

    let whole = sched_with(&[]);
    let submission = whole.submit(request(12000, 30)).expect("submit");
    whole.run_steps(400);
    let in_one_step = tokens_of(&stream(submission).await);

    assert_eq!(with_chunks, in_one_step);
    assert_eq!(
        whole.counters().prefill_steps,
        0,
        "no prefill bucket, no prefill step"
    );
    assert!(chunked.counters().prefill_steps >= 2);
}

#[tokio::test]
async fn a_prefilling_sequence_touches_one_state_slot_not_the_decode_width() {
    let sched = sched_with(&PREFILL);
    let decode_slots = sched.config().gdn_slots_per_seq();
    assert_eq!(decode_slots, 6, "spec width 5 plus the committed state");

    let submission = sched.submit(request(12000, 20)).expect("submit");
    sched.step(); // admit + the first chunk
    let (active, reserved) = sched.state_slots_active_vs_reserved();
    assert_eq!(
        active, 1,
        "a prompt chunk carries one recurrent state, whatever its M"
    );
    assert_eq!(
        reserved, decode_slots,
        "the reservation is the decode-width set: a sequence that could not grow into decode \
         after its prompt was in would be an OOM at token 4000 wearing a different hat"
    );

    // Once the prompt is in, the same sequence touches the decode width.
    sched.run_steps(3);
    let (active, reserved) = sched.state_slots_active_vs_reserved();
    assert_eq!((active, reserved), (decode_slots, decode_slots));
    sched.run_steps(200);
    let _ = stream(submission).await;
}

#[tokio::test]
async fn a_prefill_step_never_shares_a_batch_with_a_decode_step() {
    // Two requests in flight: one prefilling, one decoding. Every batch is one phase.
    let sched = sched_with(&PREFILL);
    let first = sched.submit(request(300, 60)).expect("first");
    sched.run_steps(3);
    let second = sched.submit(request(9000, 20)).expect("second");
    sched.run_steps(400);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    for b in &batches {
        assert_eq!(
            b.phase == Phase::Prefill,
            b.chunks.iter().any(Option::is_some),
            "a batch carries chunks if and only if it is a prefill batch: {b:?}"
        );
        if b.phase == Phase::Prefill {
            assert_eq!(b.sequences, 1, "one sequence per prefill step: {b:?}");
            assert!(b.chunks.iter().all(Option::is_some));
        } else {
            assert!(b.chunks.iter().all(Option::is_none));
        }
    }
    assert!(
        batches.iter().any(|b| b.phase == Phase::Prefill),
        "the long prompt should have been prefilled in chunks"
    );
    let _ = stream(first).await;
    let _ = stream(second).await;
}

#[tokio::test]
async fn five_turns_are_identical_with_prefill_scheduling() {
    // The multi-turn gate again, this time with the prompt going through prefill buckets: a turn's
    // answer must not depend on how its prompt was cut, nor on what ran before it.
    let sched = sched_with(&PREFILL);
    let mut turns: Vec<(usize, Vec<TokenId>)> = Vec::new();
    let mut prompt_len = 300;
    for _ in 0..5 {
        let submission = sched.submit(request(prompt_len, 20)).expect("submit");
        sched.run_steps(400);
        turns.push((prompt_len, tokens_of(&stream(submission).await)));
        prompt_len += 2000;
    }
    for (len, multi) in turns {
        let fresh = sched_with(&PREFILL);
        let submission = fresh.submit(request(len, 20)).expect("submit");
        fresh.run_steps(400);
        let single = tokens_of(&stream(submission).await);
        assert_eq!(
            multi, single,
            "a {len}-token turn differs from its single-turn control"
        );
    }
}

#[tokio::test]
async fn admission_still_refuses_before_a_prompt_is_ever_fed() {
    // The state arena is sized for one sequence; the second request waits, and nothing of its
    // prompt is fed in the meantime — a prefill step is still a step of an admitted sequence.
    let decode: Vec<(u32, u32, u32)> = decode_buckets()
        .into_iter()
        .filter(|(_, _, band)| *band == 65536)
        .collect();
    let plan = plan_with_prefill(&decode, &PREFILL, 2);
    let config = SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 6 * 72 * MIB,
        },
        262_144,
    )
    .expect("config");
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    let sched = Scheduler::new(
        config,
        ScriptExecutor::new(vec![answer()], TOKENS_PER_STEP, EOS),
        model,
        boot_ledger(),
    )
    .expect("arenas fit");
    assert_eq!(sched.arena_free().1, 1, "one slot set");

    // A prompt of two chunks, so there is a step at which it is admitted and still prefilling.
    let _a = sched.submit(request(12000, 20)).expect("a");
    let _b = sched.submit(request(12000, 20)).expect("b");
    sched.run_steps(1);
    assert_eq!(sched.stats().num_requests_running, 1);
    assert_eq!(sched.stats().num_requests_waiting, 1);
    assert!(sched.admission_counters().refused_gdn >= 1);
    // The reservation is the decode width even while the admitted one is only prefilling: that is
    // what makes the refusal honest rather than an over-commit that fails later.
    let (active, reserved) = sched.state_slots_active_vs_reserved();
    assert_eq!((active, reserved), (1, 6));
}

// ── the executor's view ─────────────────────────────────────────────────────────

#[tokio::test]
async fn the_chunk_carries_what_an_executor_needs_to_place_it() {
    let sched = sched_with(&PREFILL);
    let submission = sched.submit(request(12000, 20)).expect("submit");
    sched.run_steps(400);
    let _ = stream(submission).await;

    let batches = sched.with_executor(|e| e.batches().to_vec());
    let mut done = 0u32;
    let mut seen: BTreeMap<u32, u32> = BTreeMap::new();
    for b in batches.iter().filter(|b| b.phase == Phase::Prefill) {
        let chunk = b.chunks[0].expect("chunk");
        assert_eq!(
            chunk.tokens_done_before, done,
            "chunks are contiguous: this one starts where the last ended"
        );
        assert!(
            chunk.prompt_tokens <= b.bucket.tokens,
            "a chunk never exceeds its bucket's rows"
        );
        done += chunk.prompt_tokens;
        *seen.entry(b.bucket.tokens).or_default() += 1;
    }
    assert_eq!(done, 12000);
    assert_eq!(seen, BTreeMap::from([(8192, 2)]));
}

// ── FN38: the wide rung while idle, the narrow one beside decode ─────────────────

/// The prefill chunks `(rows, done before)` of the request whose first chunk starts at 0 and whose
/// prompt is `n` tokens, in feed order.
fn chunks_of(sched: &Scheduler<ScriptExecutor>, n: u32) -> Vec<(u32, u32)> {
    let batches = sched.with_executor(|e| e.batches().to_vec());
    let mut out = Vec::new();
    for b in batches.iter().filter(|b| b.phase == Phase::Prefill) {
        let c = b.chunks[0].expect("chunk");
        let fed: u32 = out.iter().map(|x: &(u32, u32)| x.0).sum();
        if c.tokens_done_before == fed && fed < n && (c.tokens_done_before > 0 || out.is_empty()) {
            out.push((c.prompt_tokens, c.tokens_done_before));
        }
    }
    out
}

#[tokio::test]
async fn fn38_a_wide_rung_is_taken_only_while_nothing_decodes_under_the_busy_cap() {
    // Alone: the busy cap never binds -- 8192 + 3808, the uncapped law.
    let alone = sched_busy(&PREFILL, 2048);
    let s = alone.submit(request(12000, 20)).expect("submit");
    alone.run_steps(400);
    let _ = stream(s).await;
    assert_eq!(chunks_of(&alone, 12000), vec![(8192, 0), (3808, 8192)]);

    // Beside a decoding sequence: every chunk keeps 2,048 rows (the cap), contiguous, whole.
    let busy = sched_busy(&PREFILL, 2048);
    let a = busy.submit(request(100, 200)).expect("submit");
    busy.run_steps(2); // A's prompt + its first decode step: A decodes from here on
    let b = busy.submit(request(12000, 20)).expect("submit");
    busy.run_steps(600);
    let _ = (stream(a).await, stream(b).await);
    let batches = busy.with_executor(|e| e.batches().to_vec());
    let long: Vec<(u32, u32)> = batches
        .iter()
        .filter(|b| b.phase == Phase::Prefill)
        .filter_map(|b| b.chunks[0])
        .filter(|c| c.prompt_tokens + c.tokens_done_before > 100 || c.tokens_done_before > 0)
        .map(|c| (c.prompt_tokens, c.tokens_done_before))
        .collect();
    let mut done = 0;
    for &(rows, before) in &long {
        assert_eq!(before, done, "contiguous");
        assert!(rows <= 2048, "a chunk beside decode keeps the cap: {rows}");
        done += rows;
    }
    assert_eq!(done, 12000, "every prompt token fed once");

    // CONTROL: the same arrivals with the cap off take the wide rung beside decode.
    let ctl = sched_busy(&PREFILL, 0);
    let a = ctl.submit(request(100, 200)).expect("submit");
    ctl.run_steps(2);
    let b = ctl.submit(request(12000, 20)).expect("submit");
    ctl.run_steps(600);
    let _ = (stream(a).await, stream(b).await);
    let batches = ctl.with_executor(|e| e.batches().to_vec());
    assert!(batches
        .iter()
        .filter(|b| b.phase == Phase::Prefill)
        .filter_map(|b| b.chunks[0])
        .any(|c| c.prompt_tokens == 8192));
}

#[test]
fn fn38_the_busy_cap_is_a_room_not_a_rung() {
    let r = PrefillChunking::from_sizes(vec![1024, 2048, 8192]).with_busy_max(2048);
    assert_eq!(r.room(false), u32::MAX);
    assert_eq!(r.room(true), 2048);
    assert_eq!(r.chunk_for(12000u32.min(r.room(true))), Some((2048, 2048)));
    assert_eq!(r.chunk_for(12000u32.min(r.room(false))), Some((8192, 8192)));
    assert_eq!(r.chunk_for(952u32.min(r.room(true))), Some((952, 1024)));
    // off: the pre-FN38 law
    let off = PrefillChunking::from_sizes(vec![1024, 2048, 8192]);
    assert_eq!(off.room(true), u32::MAX);
}

/// LR-B: one long prompt arriving beside a decoding sequence, under `yields` decode steps between
/// chunks. Returns (the long prompt's chunks in order, the decode steps run between two of them,
/// the yield counter).
async fn lrb_run(yields: u32) -> (Vec<(u32, u32)>, Vec<u32>, u64) {
    let s = sched_yield(&PREFILL, 0, yields);
    let a = s.submit(request(100, 400)).expect("submit");
    s.run_steps(2); // A's prompt + its first decode step: A decodes from here on
    let b = s.submit(request(20000, 20)).expect("submit");
    s.run_steps(800);
    let _ = (stream(a).await, stream(b).await);
    let batches = s.with_executor(|e| e.batches().to_vec());
    let mut chunks = Vec::new();
    let mut between = Vec::new();
    let mut run = None;
    for b in &batches {
        match b.phase {
            Phase::Prefill => {
                let c = b.chunks[0].expect("chunk");
                if c.prompt_tokens + c.tokens_done_before <= 100 && c.tokens_done_before == 0 {
                    continue; // A's prompt
                }
                if let Some(n) = run.take() {
                    between.push(n);
                }
                chunks.push((c.prompt_tokens, c.tokens_done_before));
                if c.prompt_tokens + c.tokens_done_before < 20000 {
                    run = Some(0);
                }
            }
            Phase::Decode => {
                if let Some(n) = run.as_mut() {
                    *n += 1;
                }
            }
        }
    }
    (chunks, between, s.counters().yield_decode_steps)
}

#[tokio::test]
async fn lrb_decoders_get_n_steps_between_chunks_and_the_cut_does_not_move() {
    let (c0, b0, y0) = lrb_run(0).await;
    assert_eq!(
        c0,
        vec![(8192, 0), (8192, 8192), (3616, 16384)],
        "the uncapped cut"
    );
    assert_eq!(
        b0,
        vec![0, 0],
        "CONTROL: the chunks run back to back, the decoder waits"
    );
    assert_eq!(y0, 0);
    for n in [1u32, 2, 4] {
        let (c, b, y) = lrb_run(n).await;
        assert_eq!(
            c, c0,
            "yield {n}: the same chunk sequence (no boundary moves)"
        );
        assert_eq!(
            b,
            vec![n, n],
            "yield {n}: n decode steps between two chunks"
        );
        assert_eq!(y, u64::from(2 * n), "yield {n}: counted");
    }
}

#[tokio::test]
async fn lrb_a_prompt_alone_never_yields() {
    // nothing decodes: the chunks run back to back whatever the arm says
    let s = sched_yield(&PREFILL, 0, 4);
    let b = s.submit(request(20000, 20)).expect("submit");
    s.run_steps(400);
    let _ = stream(b).await;
    let batches = s.with_executor(|e| e.batches().to_vec());
    let phases: Vec<Phase> = batches.iter().map(|b| b.phase).take(3).collect();
    assert_eq!(phases, vec![Phase::Prefill; 3]);
    assert_eq!(s.counters().yield_decode_steps, 0);
}
