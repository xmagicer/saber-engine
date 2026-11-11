//! FN52: prefix (conversation) caching through the scheduler, on a HOST MODEL whose every output
//! is a function of the whole history — so an adopted prefix that was wrong, overwritten, or
//! restored at the wrong position shows up as a different token.
//!
//! The model: a recurrent state folded over every fed token (the GDN half) and a KV page entry per
//! position holding the state at that position (the attention half); a sampled token reads the
//! state AND every resident KV entry through the block table. A page that two holders share and
//! one rewrites is counted (`overwrites`), never silently accepted.

mod common;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use common::{boot_ledger, decode_buckets, plan_with_prefill, GIB};
use saber_sched::{
    AdmittedSeq, ArenaBytes, KvGeometry, PrefixCachePolicy, Scheduler, SchedulerConfig,
    SchedulerError, ScriptExecutor, SeqId, SeqStep, StepBatch, StepExecutor, StepResult,
};
use saber_serve::engine::{Engine, EngineRequest, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

const EOS: TokenId = 255;
const BT: usize = 16;
const PREFILL: [(u32, u32); 2] = [(2048, 65536), (8192, 65536)];

fn mix(h: u64, v: u64) -> u64 {
    (h ^ v.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .rotate_left(17)
        .wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

#[derive(Debug, Clone)]
struct Session {
    blocks: Vec<u32>,
    done: u32,
    state: u64,
}

#[derive(Debug, Default)]
struct Toy {
    kv: HashMap<u32, [Option<u64>; BT]>,
    sessions: BTreeMap<SeqId, Session>,
    snaps: HashMap<u32, (u32, u64)>,
    overwrites: u64,
    /// Prompt rows fed, per sequence.
    fed: BTreeMap<SeqId, u32>,
    drops: u64,
}

impl Toy {
    fn feed(&mut self, seq: SeqId, tokens: &[TokenId], at: u32) -> u64 {
        let s = self.sessions.get_mut(&seq).expect("admitted");
        assert_eq!(s.done, at, "{seq}: fed at {at}, resident {}", s.done);
        for (i, t) in tokens.iter().enumerate() {
            s.state = mix(s.state, u64::from(*t));
            let p = at as usize + i;
            let b = s.blocks[p / BT];
            let page = self.kv.entry(b).or_insert([None; BT]);
            if let Some(old) = page[p % BT] {
                if old != s.state {
                    self.overwrites += 1;
                }
            }
            page[p % BT] = Some(s.state);
        }
        s.done += tokens.len() as u32;
        // The sample reads the state and every resident page entry.
        let mut h = s.state;
        for p in 0..s.done as usize {
            let v = self.kv[&s.blocks[p / BT]][p % BT].expect("resident");
            h = mix(h, v);
        }
        h
    }
}

impl StepExecutor for Toy {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let mut out = Vec::new();
        for slot in batch.slots {
            let h = self.feed(slot.seq_id, &slot.input_tokens, slot.context_len);
            if let Some(c) = slot.chunk {
                *self.fed.entry(slot.seq_id).or_default() += c.prompt_tokens;
                if !c.last {
                    out.push(SeqStep {
                        seq_id: slot.seq_id,
                        tokens: Vec::new(),
                        draft_tokens: 0,
                        accepted_tokens: 0,
                        logprobs: None,
                        prompt_logprobs: None,
                    });
                    continue;
                }
            }
            out.push(SeqStep::single(slot.seq_id, (h % 200) as TokenId + 1));
        }
        StepResult::new(out)
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        // A block this sequence owns outright starts empty (it may be one an evicted entry or a
        // finished sequence held); the adopted prefix's blocks keep their pages.
        for b in &seq.kv_blocks[seq.shared_prefix_blocks as usize..] {
            self.kv.remove(b);
        }
        self.sessions.insert(
            seq.seq_id,
            Session {
                blocks: seq.kv_blocks.clone(),
                done: 0,
                state: 0,
            },
        );
    }

    fn release(&mut self, seq: SeqId) {
        self.sessions.remove(&seq);
    }

    fn prefix_cache_supported(&self) -> bool {
        true
    }

    fn prefix_save(&mut self, seq: SeqId, entry: u32, tokens: u32) -> Result<u64, String> {
        let s = &self.sessions[&seq];
        assert_eq!(s.done, tokens, "a snapshot is taken exactly at its cut");
        self.snaps.insert(entry, (tokens, s.state));
        Ok(100)
    }

    fn prefix_load(&mut self, seq: &AdmittedSeq, entry: u32, tokens: u32) -> Result<(), String> {
        let (t, state) = *self.snaps.get(&entry).ok_or("no such entry")?;
        assert_eq!(t, tokens);
        let s = self.sessions.get_mut(&seq.seq_id).ok_or("not admitted")?;
        s.done = tokens;
        s.state = state;
        Ok(())
    }

    fn prefix_drop(&mut self, entry: u32) {
        self.snaps.remove(&entry);
        self.drops += 1;
    }

    fn prefix_verify(
        &mut self,
        seq: SeqId,
        entry: u32,
        tokens: u32,
        entry_blocks: &[u32],
    ) -> Result<bool, String> {
        let (t, state) = *self.snaps.get(&entry).ok_or("no such entry")?;
        let s = &self.sessions[&seq];
        let pages = (tokens as usize) / BT;
        let same_pages =
            (0..pages).all(|i| self.kv.get(&s.blocks[i]) == self.kv.get(&entry_blocks[i]));
        Ok(t == tokens && s.state == state && s.done == tokens && same_pages)
    }
}

fn config(policy: PrefixCachePolicy, kv_bytes: u64) -> SchedulerConfig {
    let decode: Vec<(u32, u32, u32)> = decode_buckets()
        .into_iter()
        .filter(|(_, _, band)| *band == 65536)
        .collect();
    let plan = plan_with_prefill(&decode, &PREFILL, 2);
    let mut c = SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: kv_bytes,
            gdn_state_per_rank: 8 * GIB,
        },
        262_144,
    )
    .expect("config");
    c.prefix_cache = policy;
    c
}

fn sched(policy: PrefixCachePolicy, kv_bytes: u64) -> Arc<Scheduler<Toy>> {
    let config = config(policy, kv_bytes);
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    Arc::new(Scheduler::new(config, Toy::default(), model, boot_ledger()).expect("arenas fit"))
}

fn on() -> PrefixCachePolicy {
    PrefixCachePolicy::on()
}

fn ids(seed: u32, n: usize) -> Vec<TokenId> {
    (0..n as u32)
        .map(|i| (i.wrapping_mul(31) ^ seed) % 200 + 1)
        .collect()
}

async fn turn(s: &Arc<Scheduler<Toy>>, prompt: &[TokenId], max_tokens: usize) -> Vec<TokenId> {
    use futures_util::StreamExt;
    let sub = s
        .submit(EngineRequest {
            prompt_token_ids: prompt.to_vec(),
            params: SamplingParams {
                max_tokens: Some(max_tokens),
                ..SamplingParams::default()
            },
        })
        .expect("submit");
    s.run_steps(4000);
    let events: Vec<StepEvent> = sub.events.collect().await;
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}

/// Two turns of a conversation: turn 2's prompt is turn 1's + its answer + a new message.
async fn conversation(s: &Arc<Scheduler<Toy>>, p1: &[TokenId], extra: &[TokenId]) -> Vec<TokenId> {
    let a1 = turn(s, p1, 24).await;
    let mut p2 = p1.to_vec();
    p2.extend(&a1);
    p2.extend(extra);
    let mut out = a1;
    out.extend(turn(s, &p2, 24).await);
    out
}

#[tokio::test]
async fn an_adopted_prefix_answers_exactly_what_a_fresh_prefill_answers() {
    for len in [700usize, 3000, 9000] {
        let p1 = ids(7, len);
        let extra = ids(9, 333);
        let off = sched(PrefixCachePolicy::OFF, 8 * GIB);
        let want = conversation(&off, &p1, &extra).await;
        let cached = sched(on(), 8 * GIB);
        let got = conversation(&cached, &p1, &extra).await;
        assert_eq!(
            got, want,
            "{len}: the cached turn answers what the fresh one does"
        );
        let (c, entries, _) = cached.prefix_counters();
        let a = ((len - 1) / BT * BT) as u64;
        assert_eq!((c.hits, c.adopted, c.hit_tokens), (1, 1, a), "{len}: {c:?}");
        let uncached = (len * 2 + 24 + extra.len()) as u64;
        assert_eq!(off.counters().prefill_computed_tokens, uncached);
        assert_eq!(cached.counters().prefill_computed_tokens, uncached - a);
        assert_eq!(cached.counters().output_committed_tokens, got.len() as u64);
        assert_eq!(
            entries, 2,
            "turn 1's and turn 2's prefixes are both retained"
        );
        cached.with_executor(|e| {
            assert_eq!(e.overwrites, 0, "an adopted page is never rewritten");
            // Turn 2 fed only what follows the adopted prefix.
            let fed: Vec<u32> = e.fed.values().copied().collect();
            assert_eq!(fed[0] as usize, len, "turn 1 fed its whole prompt");
            assert_eq!(
                fed[1] as u64,
                (len + 24 + 333) as u64 - a,
                "turn 2 fed the tail only"
            );
        });
        off.with_executor(|e| assert_eq!(e.overwrites, 0));
        assert_eq!(off.prefix_counters().0.lookups, 0, "off never looks");
    }
}

#[tokio::test]
async fn a_one_token_difference_in_the_prefix_misses() {
    let s = sched(on(), 8 * GIB);
    let p1 = ids(7, 3000);
    turn(&s, &p1, 8).await;
    let mut p2 = p1.clone();
    p2[5] ^= 1;
    p2.extend(ids(3, 100));
    turn(&s, &p2, 8).await;
    let (c, _, _) = s.prefix_counters();
    assert_eq!((c.lookups, c.hits), (2, 0), "{c:?}");
    // ...and the last token of the cached prefix differing misses too (no partial-block hit).
    let a = (3000 - 1) / BT * BT;
    let mut p3 = p1.clone();
    p3[a - 1] ^= 1;
    p3.extend(ids(4, 100));
    turn(&s, &p3, 8).await;
    let (c, _, _) = s.prefix_counters();
    assert_eq!(c.hits, 0, "{c:?}");
}

#[tokio::test]
async fn retained_prefixes_never_block_an_admission() {
    // A KV arena of 700 blocks (11,200 tokens): two retained 4,000-token prefixes leave too
    // little for a 9,000-token request, which must evict rather than wait.
    let kv = 700 * KvGeometry::PROD_FP8.block_bytes();
    let s = sched(on(), kv);
    turn(&s, &ids(1, 4000), 8).await;
    turn(&s, &ids(2, 4000), 8).await;
    let (c, entries, _) = s.prefix_counters();
    assert_eq!((c.saves, entries), (2, 2));
    let (held, total) = s.kv_blocks_held();
    assert_eq!(total, 700);
    assert_eq!(held, 2 * 249, "only the retained prefixes hold blocks");
    let got = turn(&s, &ids(3, 9000), 8).await;
    assert_eq!(got.len(), 8, "admitted and answered");
    let (c, _, _) = s.prefix_counters();
    assert!(c.evictions_pressure >= 1, "{c:?}");
    s.with_executor(|e| {
        assert_eq!(e.overwrites, 0);
        assert!(e.drops >= 1, "an evicted entry's snapshot is dropped");
    });
}

#[tokio::test]
async fn the_verify_arm_compares_and_adopts_nothing() {
    let s = sched(
        PrefixCachePolicy {
            verify: true,
            ..on()
        },
        8 * GIB,
    );
    let p1 = ids(7, 3000);
    let extra = ids(9, 50);
    let got = conversation(&s, &p1, &extra).await;
    let off = sched(PrefixCachePolicy::OFF, 8 * GIB);
    assert_eq!(got, conversation(&off, &p1, &extra).await);
    let (c, _, _) = s.prefix_counters();
    assert_eq!(
        (c.hits, c.adopted, c.verify_match, c.verify_mismatch),
        (1, 0, 1, 0),
        "{c:?}"
    );
}

#[test]
fn an_executor_that_cannot_snapshot_is_refused_at_boot() {
    let config = config(on(), 8 * GIB);
    let model = saber_sched::mock_model_info("Qwen3.8", "/m", EOS, config.max_context_tokens);
    let r = Scheduler::new(
        config,
        ScriptExecutor::new(vec![vec![1, EOS]], 1, EOS),
        model,
        boot_ledger(),
    );
    assert!(matches!(r, Err(SchedulerError::PrefixCacheUnsupported)));
}
