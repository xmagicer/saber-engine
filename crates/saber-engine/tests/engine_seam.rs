//! The P3 seam, on the host: the real scheduler, the real [`RealExecutor`], a forward pass with the
//! device taken out.
//!
//! These are the same assertions the GPU gate makes (`tools/capture/run_engine.sh` →
//! `saber engine`), run against [`NullForward`] so that the **gate logic** is tested where a card is
//! not needed. What the GPU adds is the arithmetic; what is checked here is everything else:
//!
//! * the same prompt through the pass directly and through the serve path gives the same ids;
//! * **four prompts batched give each sequence the ids it gets alone** — the batch > 1 claim;
//! * a step carries its bucket's padding, and the padding is counted;
//! * one `StepEvent` per engine step, and nothing at all until the prompt is in;
//! * every step's launch count matches the budget declared for the shape it ran — with a **negative
//!   control** that makes the check fire;
//! * a prompt longer than the largest prefill bucket is chunked, and the chunks are silent;
//! * the graph and eager paths produce the same ids and are told apart by a counter;
//! * `n > 1` is refused rather than served as one choice called three.

use std::sync::Arc;

use saber_engine::{
    DecodeSlot, Drafter, FixedDrafter, ForwardRunner, NullForward, RealExecutor, RecordedDrafter,
    RowRequest,
};
use saber_mem::ledger::{Ledger, Pool};
use saber_plan::Bucket;
use saber_sched::{ArenaBytes, KvGeometry, Scheduler, SchedulerConfig, SeqId};
use saber_serve::engine::{
    Engine, EngineRequest, ModelInfo, RequestId, StepEvent, Submission, TokenId,
};
use saber_serve::sampling::SamplingParams;

const GIB: u64 = 1 << 30;
/// The prefill shape ladder: one bucket, as `saber-sched`'s chunking rule would emit it.
const PREFILL: u32 = 2048;
/// The decode batch ladder the engine plans and captures.
const BATCHES: &[u32] = &[1, 2, 4, 8];

fn model_info() -> ModelInfo {
    ModelInfo {
        served_model_name: "Qwen3.8".to_owned(),
        aliases: Vec::new(),
        max_model_len: 65_536,
        root: "/packs/qwen38-27b-nvfp4-devfinal-tp1".to_owned(),
        eos_token_ids: vec![151_645],
    }
}

fn ledger() -> Ledger {
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).expect("rank 0");
    l.declare(Pool::HostPinned, 4 * GIB).expect("pinned");
    l.declare(Pool::HostPageable, 48 * GIB).expect("pageable");
    l
}

/// The scheduler the engine gets: the engine's own decode batches as buckets, the prefill ladder it
/// planned, and **the engine's measured geometry** — `KvGeometry` and the GDN slot size come from the
/// facts, so the scheduler's arena reservations are the engine's real pages and not a second guess at
/// them.
fn scheduler(forward: NullForward) -> Scheduler<RealExecutor<NullForward>> {
    let f = forward.facts();
    let pool_blocks = f.kv_pool_blocks;
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: f.kv_block_bytes / 16,
    };
    let gdn_bytes_per_slot = f.gdn_bytes_per_slot;
    let mut buckets: Vec<Bucket> = BATCHES
        .iter()
        .map(|b| Bucket::decode(*b, 1, 8192, true))
        .collect();
    buckets.push(Bucket::prefill(PREFILL, 8192, true));
    let config = SchedulerConfig::for_mock(
        buckets,
        1,
        gdn_bytes_per_slot,
        kv,
        ArenaBytes {
            kv_per_rank: u64::from(pool_blocks) * kv.block_bytes(),
            gdn_state_per_rank: 16 * gdn_bytes_per_slot,
        },
        65_536,
        "test:engine-seam".to_owned(),
    )
    .expect("config");
    Scheduler::new(config, RealExecutor::new(forward), model_info(), ledger()).expect("arenas fit")
}

fn stub() -> NullForward {
    NullForward::new(&[PREFILL], BATCHES, 8)
}

fn request(prompt: &[TokenId], max_tokens: usize) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt.to_vec(),
        params: SamplingParams {
            max_tokens: Some(max_tokens),
            // TEMPERATURE 0: every test in this file compares IDS, and `SamplingParams::default()`
            // carries the reference server's own 1.0, which since the sampler was wired into the step loop means a
            // draw. A test that sampled would be comparing two draws and calling them a gate.
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

async fn drain(sub: Submission) -> Vec<StepEvent> {
    use futures_util::StreamExt;
    sub.events.collect().await
}

fn tokens_of(events: &[StepEvent]) -> Vec<TokenId> {
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}

/// The pass, driven directly: admit, one prefill chunk, then `gen` decode steps at batch 1. This is
/// what `saber forward` does, one step at a time, and what the serve path must reproduce.
fn drive_directly(prompt: &[TokenId], gen: usize, blocks: &[u32]) -> Vec<TokenId> {
    use saber_engine::DecodeSlot;
    let mut f = stub();
    let seq = SeqId(1);
    f.admit(seq, prompt.len() as u32, blocks).expect("admit");
    let first = f
        .prefill(seq, prompt, 0, true, &RowRequest::greedy())
        .expect("the prompt")
        .token()
        .expect("the last chunk samples");
    let mut out = vec![first];
    for _ in 1..gen {
        let ctx = f.session(seq).expect("live").1;
        let last = *out.last().expect("non-empty");
        let step = f
            .decode(&[DecodeSlot::single(seq, last, ctx)], 1, 1)
            .expect("a decode step");
        out.push(step.commits[0].tokens[0]);
    }
    out
}

#[tokio::test]
async fn the_pass_and_the_serve_path_agree_token_for_token() {
    let prompt: Vec<TokenId> = (1..=234u32).collect();
    let gen = 24usize;

    let sched = Arc::new(scheduler(stub()));
    let sub = sched.submit(request(&prompt, gen)).expect("submit");
    // One step, so the sequence is admitted, and then the pages the scheduler gave it — so the
    // direct drive is the same session and not a similar one.
    let mut steps = sched.run_steps(1);
    let blocks = sched
        .with_executor(|e| e.forward_mut().session(SeqId(1)))
        .expect("the sequence is admitted")
        .2;
    steps += sched.run_steps(1000);
    let events = drain(sub).await;

    let served = tokens_of(&events);
    assert_eq!(served.len(), gen, "one token per step, {gen} of them");
    assert_eq!(events.len(), steps, "one event per engine step");
    assert_eq!(
        drive_directly(&prompt, gen, &blocks),
        served,
        "the same prompt through the pass and through the serve path"
    );
    // CONTROL: the comparison has teeth only if a different prompt gives different ids.
    let other: Vec<TokenId> = (100..=333u32).collect();
    assert_ne!(drive_directly(&other, gen, &blocks), served);

    let counters = sched.with_executor(|e| e.counters());
    assert!(counters.clean(), "a clean run: {}", counters.line());
    assert_eq!(counters.prefill_steps, 1, "one p2048 chunk for 234 tokens");
    assert_eq!(counters.decode_steps, (gen - 1) as u64);
    assert_eq!(counters.graph_steps, (gen - 1) as u64, "decode is captured");
    assert_eq!(counters.eager_steps, 1, "prefill is not");
}

/// **The batch > 1 claim.** Four prompts through one scheduler, which batches them, against the same
/// four prompts run one at a time: every sequence gets the ids it would have got alone. The padding
/// and the batching are the engine's business; the stream is not allowed to notice either.
#[tokio::test]
async fn four_prompts_batched_give_each_sequence_the_ids_it_gets_alone() {
    let prompts: Vec<Vec<TokenId>> = (0..4)
        .map(|i| {
            (1..=(40 + i * 7) as u32)
                .map(|t| t + i as u32 * 1000)
                .collect()
        })
        .collect();
    let gen = 12usize;

    // Alone: one scheduler per prompt, so nothing shares a step.
    let mut alone: Vec<Vec<TokenId>> = Vec::new();
    for p in &prompts {
        let sched = Arc::new(scheduler(stub()));
        let sub = sched.submit(request(p, gen)).expect("submit");
        sched.run_steps(1000);
        alone.push(tokens_of(&drain(sub).await));
    }
    for (i, ids) in alone.iter().enumerate() {
        assert_eq!(ids.len(), gen, "prompt {i} produced {} ids", ids.len());
    }

    // Together: one scheduler, four requests, batched steps.
    let sched = Arc::new(scheduler(stub()));
    let mut subs: Vec<(usize, RequestId, Submission)> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let sub = sched.submit(request(p, gen)).expect("submit");
        subs.push((i, sub.id, sub));
    }
    sched.run_steps(10_000);
    let mut batched: Vec<Vec<TokenId>> = vec![Vec::new(); prompts.len()];
    for (i, _, sub) in subs {
        batched[i] = tokens_of(&drain(sub).await);
    }

    let counters = sched.with_executor(|e| e.counters());
    assert!(counters.clean(), "a clean run: {}", counters.line());
    for (i, (a, b)) in alone.iter().zip(batched.iter()).enumerate() {
        assert_eq!(a, b, "sequence {i}: batched ids differ from solo ids");
    }
    // The point of the exercise: the steps really were batched.
    assert!(
        counters.decode_sequences > counters.decode_steps,
        "some step carried more than one sequence: {}",
        counters.line()
    );
    let widest = counters
        .launches_by_shape
        .keys()
        .filter(|k| k.phase == saber_plan::Phase::Decode)
        .map(|k| k.rows)
        .max()
        .unwrap_or(0);
    assert!(widest >= 2, "the widest decode shape run was {widest} rows");
}

/// A step runs its bucket's shape, padding included, and says how much of it was padding.
#[tokio::test]
async fn a_batched_step_carries_its_buckets_padding_and_counts_it() {
    // Three sequences in a ladder that has a batch-4 bucket: every step of the three together is one
    // padded row.
    let prompts: Vec<Vec<TokenId>> = (0..3).map(|i| vec![7u32 + i as u32; 16]).collect();
    let sched = Arc::new(scheduler(stub()));
    let mut subs = Vec::new();
    for p in &prompts {
        subs.push(sched.submit(request(p, 8)).expect("submit"));
    }
    sched.run_steps(10_000);
    for sub in subs {
        assert_eq!(tokens_of(&drain(sub).await).len(), 8);
    }
    let counters = sched.with_executor(|e| e.counters());
    assert!(counters.clean(), "{}", counters.line());
    assert!(
        counters.padded_rows > 0,
        "three sequences in a 1/2/4/8 ladder must have run padding: {}",
        counters.line()
    );
    // Every shape that ran was a planned one, padding and all.
    for (shape, launches) in &counters.launches_by_shape {
        assert!(
            BATCHES.contains(&shape.rows) || shape.rows == PREFILL,
            "{shape} is not a planned shape ({launches} launches)"
        );
    }
}

/// NEGATIVE CONTROL. Without it, "budget mismatches == 0" is a claim about a check nobody has seen
/// fire; rule 7 is only a gate if one extra dispatch fails.
#[tokio::test]
async fn a_step_that_issues_one_extra_dispatch_is_counted_and_the_request_is_not() {
    let prompt: Vec<TokenId> = (1..=64u32).collect();
    let sched = Arc::new(scheduler(stub().with_extra_launch_on(3)));
    let sub = sched.submit(request(&prompt, 8)).expect("submit");
    sched.run_steps(1000);
    let events = drain(sub).await;

    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(
        counters.budget_mismatches,
        1,
        "exactly the one injected step: {}",
        counters.line()
    );
    assert!(!counters.clean(), "and the run is not clean");
    // The stream is unaffected — the fault is a counted engine fault, not a corrupted response.
    assert_eq!(tokens_of(&events).len(), 8);
}

#[tokio::test]
async fn a_long_prompt_is_chunked_and_the_client_hears_nothing_until_the_last_chunk() {
    let prompt: Vec<TokenId> = (1..=5000u32).collect();
    let sched = Arc::new(scheduler(stub()));
    let sub = sched.submit(request(&prompt, 4)).expect("submit");
    let steps = sched.run_steps(1000);
    let events = drain(sub).await;

    let counters = sched.with_executor(|e| e.counters());
    assert!(counters.clean(), "{}", counters.line());
    assert_eq!(
        counters.prefill_steps, 3,
        "5000 tokens at p2048: 2048 + 2048 + 904"
    );
    assert_eq!(
        counters.silent_prefill_steps, 2,
        "the first two chunks commit nothing"
    );
    assert_eq!(events.len(), steps - 2, "the silent chunks emit no event");
    assert_eq!(tokens_of(&events).len(), 4);
}

#[tokio::test]
async fn a_refused_nonfinal_prefill_chunk_aborts_without_advancing_the_prompt() {
    // The scheduler sends a p2048 chunk; this forward only has a p512 bucket.
    let sched = scheduler(NullForward::new(&[512], BATCHES, 8));
    let prompt: Vec<TokenId> = (0..PREFILL + 1).collect();
    let sub = sched.submit(request(&prompt, 4)).expect("admit");
    assert_eq!(sched.run_steps(10), 1, "do not schedule the next chunk");

    let events = drain(sub).await;
    assert_eq!(events.len(), 1);
    assert!(events[0].new_tokens.is_empty());
    assert_eq!(
        events[0].finished,
        Some(saber_serve::engine::FinishReason::Abort)
    );
    assert_eq!(sched.counters().prefill_computed_tokens, 0);
    assert_eq!(sched.counters().executor_protocol_errors, 1);
    assert_eq!(sched.with_executor(|e| e.counters().forward_errors), 1);
}

/// The graph and the eager path are the same step twice, and the counters are what tell them apart.
#[tokio::test]
async fn the_graph_and_eager_paths_give_the_same_ids() {
    let prompt: Vec<TokenId> = (1..=64u32).collect();
    let gen = 10usize;
    let run = |forward: NullForward| {
        let sched = Arc::new(scheduler(forward));
        let sub = sched.submit(request(&prompt, gen)).expect("submit");
        sched.run_steps(1000);
        (sched, sub)
    };
    let (graph_sched, graph_sub) = run(stub());
    let (eager_sched, eager_sub) = run(stub().without_capture());
    let graph_ids = tokens_of(&drain(graph_sub).await);
    let eager_ids = tokens_of(&drain(eager_sub).await);
    assert_eq!(graph_ids.len(), gen);
    assert_eq!(graph_ids, eager_ids, "the path must not change the ids");

    let g = graph_sched.with_executor(|e| e.counters());
    let e = eager_sched.with_executor(|e| e.counters());
    assert!(g.clean() && e.clean(), "{} / {}", g.line(), e.line());
    assert_eq!(
        g.graph_steps,
        (gen - 1) as u64,
        "every decode step replayed"
    );
    assert_eq!(e.graph_steps, 0, "and none of them did with capture off");
    assert_eq!(e.eager_steps, gen as u64, "prefill + every decode step");
}

/// The token feed: in steady state the device already holds the next step's ids, and the host stops
/// feeding them. The fallback is not an error — it is what a changed slot set costs — and both sides
/// are counted, because a feed that silently disagreed with the scheduler would be a wrong answer
/// with a clean launch count.
#[tokio::test]
async fn the_device_feeds_itself_in_steady_state_and_the_host_feeds_the_rest() {
    let prompt: Vec<TokenId> = (1..=64u32).collect();
    let gen = 10usize;
    let sched = Arc::new(scheduler(stub()));
    let sub = sched.submit(request(&prompt, gen)).expect("submit");
    sched.run_steps(1000);
    let ids = tokens_of(&drain(sub).await);
    let c = sched.with_executor(|e| e.counters());
    assert_eq!(ids.len(), gen);
    assert!(c.clean(), "{}", c.line());
    // One prompt (host) + the first decode step after it (host), then the device feeds itself.
    assert_eq!(c.host_fed_steps, 2, "{}", c.line());
    assert_eq!(c.device_fed_steps, (gen - 2) as u64, "{}", c.line());

    // A second request changes the slot set, so the ids on the device are not the next step's: the
    // host feeds again, once, and the ids are unchanged by any of it.
    let solo = ids;
    let sched = Arc::new(scheduler(stub()));
    let a = sched.submit(request(&prompt, gen)).expect("a");
    sched.run_steps(3);
    let b = sched
        .submit(request(&(100..=163u32).collect::<Vec<_>>(), gen))
        .expect("b");
    sched.run_steps(1000);
    let a_ids = tokens_of(&drain(a).await);
    let _ = drain(b).await;
    let c = sched.with_executor(|e| e.counters());
    assert!(c.clean(), "{}", c.line());
    assert_eq!(
        a_ids, solo,
        "a mid-run arrival must not move another stream's ids"
    );
    assert!(
        c.host_fed_steps > 2,
        "a changed slot set must fall back to a host feed: {}",
        c.line()
    );
    assert!(c.device_fed_steps > 0, "{}", c.line());
}

/// `n > 1` is SERVED since V4 — and still refused by name on an executor that cannot branch,
/// which is the control that keeps the refusal alive (`CLAUDE.md` rule 8).
///
/// The laws of the fork itself are in `tests/fork_seam.rs`; what this asserts is the gate in
/// front of it: a group reaches the engine only when the engine can make one.
#[tokio::test]
async fn n_greater_than_one_is_served_and_refused_by_an_engine_that_cannot_fork() {
    // THE CONTROL: a stub that says it cannot branch. `n > 1` must not reach the arenas at all.
    let cannot = Arc::new(scheduler(stub().without_fork()));
    let mut req = request(&[1, 2, 3], 4);
    req.params.n = 3;
    let err = cannot.submit(req).expect_err("this engine cannot branch");
    let text = err.to_string();
    assert!(
        text.contains('3') || text.to_lowercase().contains("fork") || text.contains('n'),
        "the refusal should say why: {text}"
    );
    let sub = cannot.submit(request(&[1, 2, 3], 4)).expect("n = 1");
    cannot.run_steps(100);
    assert_eq!(tokens_of(&drain(sub).await).len(), 4);

    // THE CASE: the same request on the engine that can. Three choices, one prefill.
    let sched = Arc::new(scheduler(stub()));
    let mut req = request(&[1, 2, 3], 4);
    req.params.n = 3;
    let sub = sched.submit(req).expect("n = 3 is served now");
    sched.run_steps(200);
    let events = drain(sub).await;
    let choices: std::collections::BTreeSet<u32> = events.iter().map(|e| e.choice_index).collect();
    assert_eq!(choices.len(), 3, "three choices came back");
    let c = sched.with_executor(|e| e.counters());
    assert_eq!(
        c.forks,
        2,
        "two branches forked from one leader: {}",
        c.line()
    );
    assert_eq!(c.prefill_steps, 1, "one prefill for three choices");
    assert!(c.clean(), "{}", c.line());
}

/// A SPEC=0 run and a verified run, over the same stub, for one sequence.
///
/// The stub's "model" folds every id it is FED into a running hash and samples from it, so row `i` of
/// a verify step answers a different question from row `i - 1` — which is what makes the acceptance
/// law's prefix comparison mean something here rather than being a formality.
fn run(widths: &[u32], width: u32, dial: u32, steps: usize) -> (Vec<TokenId>, Vec<u32>) {
    let prompt: Vec<TokenId> = vec![7, 8, 9, 10];
    let mut f = NullForward::with_widths(&[16], &[1], widths, 2);
    let seq = SeqId(1);
    f.admit(seq, prompt.len() as u32, &[0, 1, 2, 3])
        .expect("admit");
    let first = f
        .prefill(seq, &prompt, 0, true, &RowRequest::greedy())
        .expect("prefill")
        .token()
        .expect("a token");
    let mut out = vec![first];
    let mut accepted = Vec::new();
    // The oracle's script is the SPEC=0 continuation, which the caller has already measured when
    // `dial > 0`; at width 1 there is nothing to draft.
    let mut drafter = RecordedDrafter::new(width.saturating_sub(1), dial);
    if dial > 0 {
        let (spec0, _) = run(&[1], 1, 0, steps + width as usize + 2);
        drafter.record(seq, spec0[1..].to_vec());
    }
    let mut committed_before = 1u32;
    while out.len() < steps {
        let last = *out.last().expect("a token");
        let draft = drafter.propose(seq, last, width - 1);
        let slot = DecodeSlot {
            seq,
            input: last,
            context_len: prompt.len() as u32 + out.len() as u32 - 1,
            draft,
            committed_before,
            sampling: RowRequest::greedy(),
            decider: None,
        };
        let o = f.decode(&[slot], 1, width).expect("a verify step");
        let c = &o.commits[0];
        accepted.push(c.accepted);
        committed_before = c.tokens.len() as u32;
        drafter.commit(seq, &c.tokens);
        out.extend(&c.tokens);
    }
    out.truncate(steps);
    (out, accepted)
}

/// **The gate, on the host.** A verified run commits the SPEC=0 run's ids — at every acceptance rate
/// from "nothing survives" to "everything does" — because every committed token is the model's own at
/// a position whose whole prefix the model also produced. Speculation buys steps, never tokens.
#[test]
fn a_verified_run_commits_the_spec0_ids_at_every_acceptance_rate() {
    const N: usize = 40;
    let (spec0, zero) = run(&[1], 1, 0, N);
    assert!(zero.iter().all(|a| *a == 0), "SPEC=0 accepts no draft");
    for width in [2u32, 3, 5] {
        for dial in 0..width {
            let (ids, accepted) = run(&[1, width], width, dial, N);
            assert_eq!(ids, spec0, "width {width}, dial {dial}");
            // The dial is an upper bound: a position past it gets a token the model cannot want.
            assert!(
                accepted.iter().all(|a| *a <= dial),
                "width {width} dial {dial} accepted {accepted:?}"
            );
            // And it is reached — a run where nothing was ever accepted at a positive dial would
            // mean the oracle's script had drifted out of step with the model.
            if dial > 0 {
                assert!(
                    accepted.contains(&dial),
                    "width {width} dial {dial} never reached its own acceptance: {accepted:?}"
                );
            }
        }
    }
}

/// Tokens per step is the acceptance plus one, which is the number a real drafter's acceptance rate
/// gets multiplied into. At the full dial a width-5 step commits 5 tokens; at dial 0 it commits 1 and
/// has paid for 5 rows to do it.
#[test]
fn tokens_per_step_is_the_acceptance_plus_one() {
    const N: usize = 40;
    for width in [2u32, 5] {
        for dial in 0..width {
            let (_, accepted) = run(&[1, width], width, dial, N);
            let steps = accepted.len() as f64;
            let committed: u32 = accepted.iter().map(|a| a + 1).sum();
            let per_step = f64::from(committed) / steps;
            assert!(
                (per_step - f64::from(dial + 1)).abs() < 0.25,
                "width {width} dial {dial}: {per_step:.2} tokens/step, expected about {}",
                dial + 1
            );
        }
    }
}

/// The fixed mock: guesses that are wrong, a run that is right. It is the rejection path on every
/// step, and the ids are still the SPEC=0 run's.
#[test]
fn the_fixed_mock_rejects_and_the_ids_are_unchanged() {
    const N: usize = 24;
    let (spec0, _) = run(&[1], 1, 0, N);
    let prompt: Vec<TokenId> = vec![7, 8, 9, 10];
    let mut f = NullForward::with_widths(&[16], &[1], &[1, 4], 2);
    let seq = SeqId(1);
    f.admit(seq, prompt.len() as u32, &[0, 1, 2, 3])
        .expect("admit");
    let mut out = vec![f
        .prefill(seq, &prompt, 0, true, &RowRequest::greedy())
        .expect("prefill")
        .token()
        .expect("a token")];
    let mut drafter = FixedDrafter::new(4321, 3);
    let mut committed_before = 1u32;
    let mut steps = 0usize;
    while out.len() < N {
        let last = *out.last().expect("a token");
        let o = f
            .decode(
                &[DecodeSlot {
                    seq,
                    input: last,
                    context_len: prompt.len() as u32 + out.len() as u32 - 1,
                    draft: drafter.propose(seq, last, 3),
                    committed_before,
                    sampling: RowRequest::greedy(),
                    decider: None,
                }],
                1,
                4,
            )
            .expect("a verify step");
        let c = &o.commits[0];
        committed_before = c.tokens.len() as u32;
        out.extend(&c.tokens);
        steps += 1;
    }
    out.truncate(N);
    assert_eq!(out, spec0);
    // One token per step, from four rows: the cost of a drafter that is always wrong.
    assert_eq!(steps, N - 1);
}

/// NEGATIVE CONTROL: a slot whose rows are not the bucket's width. Padding a sequence's own rows
/// would write K/V at positions the next step has to un-write and would put its snapshots out of
/// step with its `naccept`, so the shape is refused by name.
#[test]
fn a_slot_that_drafted_a_different_width_is_refused() {
    let mut f = NullForward::with_widths(&[16], &[1], &[1, 4], 2);
    let seq = SeqId(1);
    f.admit(seq, 4, &[0, 1, 2, 3]).expect("admit");
    let first = f
        .prefill(seq, &[7, 8, 9, 10], 0, true, &RowRequest::greedy())
        .expect("prefill")
        .token()
        .expect("a token");
    let short = DecodeSlot {
        seq,
        input: first,
        context_len: 4,
        draft: vec![1, 2],
        committed_before: 1,
        sampling: RowRequest::greedy(),
        decider: None,
    };
    let e = f
        .decode(std::slice::from_ref(&short), 1, 4)
        .expect_err("three rows in a width-4 bucket");
    assert!(format!("{e}").contains("bucket's own width"), "{e}");
    // And a width nobody planned is a NoPlan, not a silently narrower step.
    assert!(f.decode(std::slice::from_ref(&short), 1, 3).is_err());
}

/// THE SAME REQUEST TWICE, WITH FOUR OTHERS BETWEEN THEM, MUST GIVE THE SAME IDS.
///
/// Every other identity arm in this tree compares two runs that see the same history: batched
/// against solo, graph against eager, pass against serve. None of them can see a fresh request whose
/// answer depends on what the engine did BEFORE it, because they all ask the question once. This one
/// asks it twice, in the middle of other traffic, and it is the shape of the 2026-09-17 GDN
/// multi-turn corruption — a state slot handed back dirty, a KV page reused before it is overwritten,
/// an arena edge read before it is written, or an index blob still describing the last step's rows.
///
/// The FIRST SAMPLED TOKEN is the assertion that matters. It comes out of prefill, so it is a
/// function of the prompt and of nothing else that the engine is allowed to remember; if token 0
/// moves, the engine is carrying state across a release.
///
/// Found by ws-p5's multi-turn gate against `saber serve --engine real`, where two identical fresh
/// requests answered `['c','tera']` and `['c','ivid']`. This is the host half of that gate: the stub
/// is a true function of the fed history, so a failure HERE is the engine's bookkeeping — slot
/// assignment, release, the index blob — and a pass here with a failure on the card is the device's
/// arena or its kernels.
#[tokio::test]
async fn the_same_request_twice_around_other_traffic_gives_the_same_ids() {
    let gen = 8usize;
    let subject: Vec<TokenId> = (1..=37u32).map(|t| t + 5_000).collect();
    // Four others, each a different length so they occupy different slots and different page counts.
    let others: Vec<Vec<TokenId>> = (0..4)
        .map(|i| {
            (1..=(29 + i * 11) as u32)
                .map(|t| t + i as u32 * 77)
                .collect()
        })
        .collect();

    let sched = Arc::new(scheduler(stub()));

    // First: the subject alone, on a fresh engine.
    let first = {
        let sub = sched.submit(request(&subject, gen)).expect("submit");
        sched.run_steps(10_000);
        tokens_of(&drain(sub).await)
    };
    assert_eq!(first.len(), gen, "the subject produced {} ids", first.len());

    // Then four other requests, run to completion and released.
    for (i, p) in others.iter().enumerate() {
        let sub = sched.submit(request(p, gen)).expect("submit");
        sched.run_steps(10_000);
        let ids = tokens_of(&drain(sub).await);
        assert_eq!(ids.len(), gen, "other {i} produced {} ids", ids.len());
    }

    // And the subject again, byte for byte the same request.
    let again = {
        let sub = sched.submit(request(&subject, gen)).expect("submit");
        sched.run_steps(10_000);
        tokens_of(&drain(sub).await)
    };

    let counters = sched.with_executor(|e| e.counters());
    assert!(counters.clean(), "a clean run: {}", counters.line());
    assert_eq!(
        first.first(),
        again.first(),
        "TOKEN 0 of an identical fresh request moved after other traffic: {:?} then {:?}. The first \
         sampled token is a function of the prompt alone; if it moves, the engine kept something \
         across a release.",
        first.first(),
        again.first()
    );
    assert_eq!(
        first, again,
        "an identical fresh request gave different ids after other traffic"
    );
}

/// The same question with the others still IN FLIGHT rather than finished.
///
/// The control for the test above: there the four others had completed and been released, so a
/// failure would be about what release leaves behind. Here the subject is re-issued while the others
/// are still decoding, so it is about what a step shares between the sequences in it — the same
/// question `four_prompts_batched...` asks, but for a sequence that ARRIVES into the middle of them.
#[tokio::test]
async fn a_request_that_arrives_mid_flight_gets_the_ids_it_gets_alone() {
    let gen = 10usize;
    let subject: Vec<TokenId> = (1..=33u32).map(|t| t + 9_000).collect();

    // Alone, on its own engine.
    let alone = {
        let sched = Arc::new(scheduler(stub()));
        let sub = sched.submit(request(&subject, gen)).expect("submit");
        sched.run_steps(10_000);
        tokens_of(&drain(sub).await)
    };

    // And into the middle of three others that are still decoding.
    let sched = Arc::new(scheduler(stub()));
    let mut held = Vec::new();
    for i in 0..3u32 {
        let p: Vec<TokenId> = (1..=(41 + i * 5)).map(|t| t + i * 131).collect();
        held.push(sched.submit(request(&p, 64)).expect("submit"));
    }
    sched.run_steps(6);
    let sub = sched.submit(request(&subject, gen)).expect("submit");
    sched.run_steps(10_000);
    let mid = tokens_of(&drain(sub).await);
    for h in held {
        let _ = drain(h).await;
    }

    let counters = sched.with_executor(|e| e.counters());
    assert!(counters.clean(), "a clean run: {}", counters.line());
    assert_eq!(
        alone.first(),
        mid.first(),
        "token 0 of a request that arrived mid-flight differs from its solo token 0"
    );
    assert_eq!(alone, mid, "a mid-flight arrival changed a request's ids");
}

/// **H36: the chunk law at 2048 / 4096 / 8192, and a chunk boundary does not change a token.**
///
/// The same 8,000-token prompt (the stub's block table at `[2048]` holds 8,192) served at four
/// prefill bucket sets -- four chunks, two, one, one:
/// the scheduler cuts it by the one law (`PrefillChunking`: as much of what is left as the largest
/// bucket takes, into the smallest bucket that holds it), every chunk builds its index tables from
/// the same laws the device does (`StepIndex::check` refuses an illegal chunk), and the session's
/// resident count and state carry across each boundary. The stub's "model" is a fold over every
/// prompt token in order, so ids that agree across the four arms say the chunk boundaries carried
/// the whole history exactly once; the device half of the claim (the GDN state, the attention over
/// earlier chunks' KV, the M-RoPE positions) is the card gate's.
#[tokio::test]
async fn the_chunk_law_at_2048_4096_and_8192_gives_the_same_ids() {
    fn scheduler_at(prefill: &[u32]) -> Scheduler<RealExecutor<NullForward>> {
        let forward = NullForward::new(prefill, BATCHES, 8);
        let f = forward.facts();
        let kv = KvGeometry {
            block_tokens: 16,
            bytes_per_token_per_rank: f.kv_block_bytes / 16,
        };
        let mut buckets: Vec<Bucket> = BATCHES
            .iter()
            .map(|b| Bucket::decode(*b, 1, 32_768, true))
            .collect();
        for p in prefill {
            buckets.push(Bucket::prefill(*p, 32_768, true));
        }
        let config = SchedulerConfig::for_mock(
            buckets,
            1,
            f.gdn_bytes_per_slot,
            kv,
            ArenaBytes {
                kv_per_rank: u64::from(f.kv_pool_blocks) * kv.block_bytes(),
                gdn_state_per_rank: 16 * f.gdn_bytes_per_slot,
            },
            65_536,
            "test:chunk-law".to_owned(),
        )
        .expect("config");
        Scheduler::new(config, RealExecutor::new(forward), model_info(), ledger())
            .expect("arenas fit")
    }
    let prompt: Vec<TokenId> = (0..8_000u32).map(|i| 1 + (i * 7919) % 150_000).collect();
    let mut runs = Vec::new();
    for (prefill, chunks) in [
        (&[2048u32][..], 4u64),
        (&[2048, 4096][..], 2),
        (&[2048, 8192][..], 1),
        (&[256, 2048, 8192][..], 1),
    ] {
        let sched = Arc::new(scheduler_at(prefill));
        let sub = sched.submit(request(&prompt, 16)).expect("submit");
        sched.run_steps(1000);
        let ids = tokens_of(&drain(sub).await);
        let c = sched.with_executor(|e| e.counters());
        assert!(c.clean(), "{prefill:?}: {}", c.line());
        assert_eq!(c.prefill_steps, chunks, "{prefill:?}: 8000 tokens");
        assert_eq!(ids.len(), 16, "{prefill:?}");
        runs.push((prefill, ids));
    }
    for (prefill, ids) in &runs[1..] {
        assert_eq!(ids, &runs[0].1, "{prefill:?} against the 2048 chunking");
    }
    // The CONTROL: a history that differs in one token of the LAST chunk is a different answer,
    // so the agreement above is not a stub that ignores what the chunks carried.
    let mut other = prompt.clone();
    other[7_999] += 1;
    let sched = Arc::new(scheduler_at(&[2048, 8192]));
    let sub = sched.submit(request(&other, 16)).expect("submit");
    sched.run_steps(1000);
    assert_ne!(tokens_of(&drain(sub).await), runs[0].1);
}
