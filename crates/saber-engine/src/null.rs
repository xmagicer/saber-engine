//! [`NullForward`] — the pass with the device taken out, for the host.
//!
//! It is not a mock of the model: it is a mock of the **device**. Every structural rule the real
//! engine holds is held here too — the prefill shape ladder and its padding, the decode batch ladder
//! and *its* padding, the step index tables (built by [`crate::index_table`], from the same laws, so
//! a wrong one fails on the host), the state-slot free list with slot 0 reserved, the declared launch
//! budget per shape, the graph/eager which-path — and only the arithmetic is fake. That is what lets
//! `saber serve --engine real`'s gate logic be tested without a card: the two implementations differ
//! in where the tokens come from and in nothing else.
//!
//! The launch numbers are deliberately synthetic (10 per body row-group, 4 for the head and its
//! extras). A number that looked like the graph's would invite being read as a measurement.

use std::collections::BTreeMap;

use saber_plan::Phase;
use saber_sched::SeqId;
use saber_serve::engine::TokenId;

use saber_kernels::head::DEVICE_ROW_K;
use saber_serve::sampler::DeviceRow;

use crate::draft::Acceptance;
use crate::forward::{
    DecodeSlot, ForwardError, ForwardFacts, ForwardRunner, RowRequest, SlotCommit, StepOutcome,
    StepPath, StepRows, TokenFeed,
};
use crate::index::{index_table, SeqFacts, StepIndex};

/// Body dispatches a shape declares, in this stub.
const BODY_LAUNCHES: u32 = 10;
/// Head dispatches plus the three the pass adds around it (two row gathers and the argmax).
const HEAD_LAUNCHES: u32 = 4;
/// Bytes this stub says one KV page holds. Prod's is 16 KiB/token/rank x 16 tokens at TP=2; this is
/// a host stub, so it is the smallest size that keeps a page a page and an image several of them.
const NULL_KV_BLOCK_BYTES: u64 = 4096;
/// Bytes this stub says one GDN state slot holds.
const NULL_GDN_SLOT_BYTES: u64 = 1024;
/// The logit every id outside this stub's candidate set holds.
///
/// A CONSTANT background is what makes the stub's seam exact without materialising a
/// 248,320-wide row: every non-candidate id has the same value, so the tie rule (lowest id first)
/// says which of them the top-K holds, and the tail's contribution to a log-sum-exp is one
/// multiplication rather than a sum. It is far enough below the candidates that the whole row's
/// mass is inside the seam, which is the shape a real logit row has.
const NULL_BACKGROUND: f32 = -20.0;
/// Ids this stub gives a distinguished logit to. 32 is under a 64-wide seam on purpose: the seam
/// then holds every candidate AND the lowest-id background tokens, which is where the tie rule
/// gets exercised.
const NULL_CANDIDATES: usize = 32;
/// The best logit. The id that holds it is the one `sample_from` returns, so the stub's greedy
/// token IS its row's argmax and the two paths cannot disagree.
const NULL_PEAK: f32 = 10.0;

/// The index keys this stub builds every step. The real engine asks the plan
/// (`BlockPlan::index_keys()`); this is the same families, one of each, so the laws are exercised.
const KEYS: &[&str] = &[
    "embed_tokens.ids",
    "l0.gdn.conv_prep.cu",
    "l0.gdn.conv_prep.cache_idx",
    "l0.gdn.conv_prep.has_init",
    "l0.gdn.fused.sidx",
    "l0.gdn.fused.naccept",
    "layer.3.attn.prefill.block_table",
    "layer.3.attn.prefill.seqused_k",
];

#[derive(Clone, Debug)]
struct Session {
    /// Running hash of everything fed to this sequence. The "model".
    hash: u64,
    /// Tokens resident.
    done: u32,
    /// GDN state slot, always 1 or more.
    slot: u32,
    /// Tokens the last step committed: which snapshot of the slot run is live (`crate::device`'s
    /// `Session::committed`). There are no slots here, so the stub only has to carry it round the
    /// park so that the field is exercised on the host.
    committed: u32,
    /// Pages the scheduler gave it.
    kv_blocks: Vec<u32>,
}

/// A [`ForwardRunner`] with no device under it.
#[derive(Debug)]
pub struct NullForward {
    facts: ForwardFacts,
    /// Prefill row counts this stub "planned", ascending.
    prefill_shapes: Vec<u32>,
    /// Decode batches this stub "planned", ascending.
    decode_batches: Vec<u32>,
    /// Verify widths this stub "planned", ascending. `[1]` is SPEC=0.
    decode_widths: Vec<u32>,
    /// The acceptance law, so the stub decides the same way the device does.
    law: crate::draft::Greedy,
    declared: BTreeMap<StepRows, u32>,
    live: BTreeMap<SeqId, Session>,
    free_slots: Vec<u32>,
    steps: u64,
    /// The history behind each SAMPLED ROW of the step that just ran, so the dense fallback can
    /// materialise exactly the row the seam refused. Cleared and rebuilt every step, which is the
    /// same lifetime the real engine's head arena has.
    last_rows: BTreeMap<u32, u64>,
    /// Fault injection for the negative control: issue one extra dispatch on this step index.
    extra_launch_on: Option<u64>,
    /// Whether the decode shapes were "captured" — the which-path flag the real engine has.
    capture: bool,
    /// What the "device" ids buffer holds: (batch, slot order, tokens).
    fed: Option<(u32, Vec<SeqId>, Vec<TokenId>)>,
    /// Polls a transfer takes before it reports done. **0 means this stub parks synchronously**,
    /// which is the default and what every gate written before ws-k9 exercises.
    transfer_steps: u32,
    /// The park this stub holds: `(sequence, polls left, the blob it owes)`.
    in_flight: Option<(SeqId, u32, Option<Vec<u8>>)>,
    /// The restore it holds, when there are two staging slots. With one slot a restore uses
    /// `in_flight` and the two take turns, exactly as the engine's single buffer makes them.
    in_flight_restore: Option<(SeqId, u32, Option<Vec<u8>>)>,
    /// Whether this stub can branch a session. True by default; [`NullForward::without_fork`] is
    /// the negative control for `n > 1`'s refusal.
    forks: bool,
    /// The last prefill chunk's sequence and the history behind each of its rows (V7): what
    /// `prompt_rows` scores, with the same lifetime as the real engine's body arena — until the
    /// next step.
    last_prefill: Option<(SeqId, Vec<u64>)>,
    /// Prompt rows whose rank the seam could not decide and a dense read did (V7).
    pub dense_prompt_ranks: u64,
    /// The decode ROW BANDS this stub "planned" (C2), in rows, at the largest batch and the widest
    /// width. Empty = `--rectangular-widths`.
    bands: Vec<u32>,
    /// The id a band's PADDING rows carry (C2's host test b): a poison value a live row must never
    /// read. 0 by default, like the device's zero-filled tail.
    band_pad_id: TokenId,
    /// NEGATIVE CONTROL (C2): read a band's slot `i` at rows `i * max_width` — the rectangle's
    /// layout — instead of by the `cu` prefix, which reads the neighbours' and the padding's rows.
    strided_band_reads: bool,
}

impl NullForward {
    /// A stub that plans `prefill_shapes` (row counts) and `decode_batches` (batch sizes) at verify
    /// width 1, with `max_seqs` sequence state slots.
    pub fn new(prefill_shapes: &[u32], decode_batches: &[u32], max_seqs: u32) -> Self {
        Self::with_widths(prefill_shapes, decode_batches, &[1], max_seqs)
    }

    /// The same, with verify widths: `[1]` is SPEC=0 and `[1, 5]` is SPEC=0 plus `b*s5`.
    ///
    /// A shape is a (batch, width) PAIR, because the plan and the capture are per pair: `b1s5` and
    /// `b4s1` both run 4 rows and are not the same graph.
    pub fn with_widths(
        prefill_shapes: &[u32],
        decode_batches: &[u32],
        decode_widths: &[u32],
        max_seqs: u32,
    ) -> Self {
        let mut prefill: Vec<u32> = prefill_shapes.to_vec();
        prefill.sort_unstable();
        prefill.dedup();
        let mut batches: Vec<u32> = decode_batches.to_vec();
        batches.sort_unstable();
        batches.dedup();
        let mut declared = BTreeMap::new();
        for rows in &prefill {
            for samples in [false, true] {
                declared.insert(
                    StepRows {
                        phase: Phase::Prefill,
                        rows: *rows,
                        width: 1,
                        samples,
                        tree: false,
                        ragged: false,
                    },
                    BODY_LAUNCHES + if samples { HEAD_LAUNCHES } else { 0 },
                );
            }
        }
        let mut widths: Vec<u32> = decode_widths.to_vec();
        widths.sort_unstable();
        widths.dedup();
        if widths.is_empty() {
            widths.push(1);
        }
        for batch in &batches {
            for width in &widths {
                declared.insert(
                    StepRows {
                        phase: Phase::Decode,
                        rows: batch * width,
                        width: *width,
                        samples: true,
                        tree: false,
                        ragged: false,
                    },
                    BODY_LAUNCHES + HEAD_LAUNCHES,
                );
            }
        }
        let max_blocks = prefill.last().copied().unwrap_or(16).div_ceil(16) * 4;
        let captured: Vec<(StepRows, usize)> = declared
            .keys()
            .filter(|k| k.phase == Phase::Decode)
            .map(|k| (*k, (BODY_LAUNCHES + HEAD_LAUNCHES) as usize))
            .collect();
        let facts = ForwardFacts {
            model: "null (no device)".to_owned(),
            tp: 1,
            rank: 0,
            // One card, so no collective and no peer-visible byte. The device pass at TP=2 says 129
            // and a few hundred MiB; this stub says what it is.
            collectives: 0,
            ipc_bytes: 0,
            layers: 64,
            // The model's own interval: layers 3, 7, 11, … are full attention.
            attention_interval: 4,
            vocab: 248_320,
            weights: 0,
            rank_memory_plane: String::new(),
            // The same shape of statement the device pass makes, so the banner path is exercised.
            supplied: vec![("null".to_owned(), 1)],
            device_bytes: 0,
            kv_pool_blocks: max_blocks * max_seqs.max(1),
            max_blocks,
            max_ctx: max_blocks * 16,
            max_seqs,
            kv_block_bytes: 16 * 1024,
            // The stub has one "attention layer", so the per-layer figure IS the arena stride.
            kv_block_stride: 16 * 1024,
            gdn_bytes_per_slot: 144 * 1024 * 1024,
            decode_shapes: batches
                .iter()
                .flat_map(|b| widths.iter().map(move |w| (*b, *w)))
                .collect(),
            max_width: widths.last().copied().unwrap_or(1),
            decode_bands: Vec::new(),
            captured,
            corrected_declarations: 0,
            declared: declared.iter().map(|(k, v)| (*k, *v)).collect(),
            // The stub has no pinned memory and no device; its images are plain Vecs, and
            // this number is what the DEVICE runner bills. Reporting the image's own size keeps
            // `can_park()` honest here without pretending a host buffer was reserved.
            park_staging_bytes: NULL_KV_BLOCK_BYTES * 64 + NULL_GDN_SLOT_BYTES,
            // The stub holds one transfer at a time, like a one-slot engine.
            park_staging_slots: 1,
            // One card: no collective, so no family to resolve and no band to raise.
            ar_families: Vec::new(),
            ar_exact: false,
            // The stub resolves no kernels, so there are no per-shape launch facts to carry and
            // `bucket_op_deltas` over it is empty rather than wrong.
            bucket_ops: Vec::new(),
            draft_tier: None,
            draft_resident: None,
            aux_tap: None,
        };
        Self {
            facts,
            prefill_shapes: prefill,
            decode_batches: batches,
            decode_widths: widths,
            law: crate::draft::Greedy,
            declared,
            live: BTreeMap::new(),
            // Slot 0 is the kernels' null slot on the device; it is reserved here too, so the two
            // implementations allocate the same slots for the same sequences.
            free_slots: (1..=max_seqs).rev().collect(),
            steps: 0,
            last_rows: BTreeMap::new(),
            extra_launch_on: None,
            capture: true,
            fed: None,
            transfer_steps: 0,
            in_flight: None,
            in_flight_restore: None,
            forks: true,
            last_prefill: None,
            dense_prompt_ranks: 0,
            bands: Vec::new(),
            band_pad_id: 0,
            strided_band_reads: false,
        }
    }

    /// Plan decode ROW BANDS of these row counts (C2) at the largest batch and the widest width,
    /// clipped as the engine clips them ([`saber_sched::band_rows_for`]), each declared and
    /// "captured" like a rectangle.
    #[must_use]
    pub fn with_bands(mut self, asked: &[u32]) -> Self {
        let batch = self.decode_batches.last().copied().unwrap_or(1);
        let width = self.decode_widths.last().copied().unwrap_or(1);
        self.bands = saber_sched::band_rows_for(asked, batch, width);
        for rows in &self.bands {
            let key = StepRows {
                phase: Phase::Decode,
                rows: *rows,
                width,
                samples: true,
                tree: false,
                ragged: true,
            };
            self.declared.insert(key, BODY_LAUNCHES + HEAD_LAUNCHES);
            self.facts
                .declared
                .push((key, BODY_LAUNCHES + HEAD_LAUNCHES));
            self.facts
                .captured
                .push((key, (BODY_LAUNCHES + HEAD_LAUNCHES) as usize));
            self.facts.decode_bands.push((batch, width, *rows));
        }
        self
    }

    /// Fill a band's padding rows with `id` (C2's poisoned-pad control's CASE: no live row may
    /// change).
    #[must_use]
    pub fn with_band_pad_id(mut self, id: TokenId) -> Self {
        self.band_pad_id = id;
        self
    }

    /// NEGATIVE CONTROL (C2): read a band's slots at the rectangle's stride instead of by `cu`.
    #[must_use]
    pub fn with_strided_band_reads(mut self) -> Self {
        self.strided_band_reads = true;
        self
    }

    /// Issue one dispatch more than declared on step `n`. The negative control for the launch-budget
    /// assertion: without it, "budget mismatches == 0" is a claim about a check nobody proved fires.
    #[must_use]
    pub fn with_extra_launch_on(mut self, n: u64) -> Self {
        self.extra_launch_on = Some(n);
        self
    }

    /// Park and restore **off the step loop**, taking `polls` polls to finish.
    ///
    /// The stub has no DMA, so what this models is the shape of the seam rather than its cost: the
    /// issue returns at once, the sequence must not be stepped and its allocations must not be
    /// freed until the poll says done, and the poll never blocks. That is exactly the contract the
    /// real engine's asynchronous park has, and it is the one the scheduler can get wrong.
    #[must_use]
    pub fn with_async_park(mut self, polls: u32) -> Self {
        self.transfer_steps = polls.max(1);
        self
    }

    /// A stub that CANNOT fork — the negative control for `n > 1` (`CLAUDE.md` rule 8).
    ///
    /// The scheduler refuses `n > 1` at submit when the executor cannot branch, and that refusal
    /// is the only thing standing between a client asking for three choices and getting one
    /// stream called three. A gate for it needs an executor that says no, and after V4 the real
    /// one says yes.
    #[must_use]
    pub fn without_fork(mut self) -> Self {
        self.forks = false;
        self
    }

    /// Two staging slots, so a park and a restore can be in flight at once.
    #[must_use]
    pub fn with_two_slots(mut self) -> Self {
        self.facts.park_staging_slots = 2;
        self
    }

    /// A small vocabulary (FN69's lookup gate: the stub's streams repeat their n-grams, so the
    /// prompt-lookup source hits). At least 64 ids: the seam's candidate set must fit.
    #[must_use]
    pub fn with_vocab(mut self, vocab: u32) -> Self {
        assert!(
            vocab >= 64,
            "the stub's candidate rows need at least 64 ids"
        );
        self.facts.vocab = vocab;
        self
    }

    /// Run the eager path instead of replaying the "captured" shapes.
    #[must_use]
    pub fn without_capture(mut self) -> Self {
        self.capture = false;
        self.facts.captured.clear();
        self
    }

    /// One sequence's session state, for a caller that needs the allocation the engine saw.
    pub fn session(&self, seq: SeqId) -> Option<(u32, u32, Vec<u32>)> {
        self.live
            .get(&seq)
            .map(|s| (s.slot, s.done, s.kv_blocks.clone()))
    }

    /// The smallest planned prefill shape that holds `rows`.
    fn prefill_shape(&self, rows: u32) -> Result<u32, ForwardError> {
        self.prefill_shapes
            .iter()
            .copied()
            .find(|s| *s >= rows)
            .ok_or_else(|| ForwardError::NoPlan {
                phase: Phase::Prefill,
                rows,
                detail: format!("planned prefill shapes are {:?}", self.prefill_shapes),
            })
    }

    fn path(&self, phase: Phase) -> StepPath {
        if self.capture && phase == Phase::Decode {
            StepPath::Graph
        } else {
            StepPath::Eager
        }
    }

    fn launches(&mut self, base: u32) -> u32 {
        self.steps += 1;
        if self.extra_launch_on == Some(self.steps - 1) {
            base + 1
        } else {
            base
        }
    }

    /// Build every index table for this step, exactly as the device pass does, and fold them into
    /// each live session's hash so that a change in any of them changes the tokens.
    fn run_indices(&mut self, index: &StepIndex<'_>, live: &[SeqId]) -> Result<(), ForwardError> {
        index
            .check()
            .map_err(|e| ForwardError::Device(e.to_string()))?;
        let mut folded = Vec::new();
        for key in KEYS {
            // `has_init` is a prefill-only table, as in the plan.
            if index.phase == Phase::Decode && key.ends_with(".has_init") {
                continue;
            }
            let bytes = index_table(key, index).map_err(|e| ForwardError::Device(e.to_string()))?;
            // Every table is BUILT (so a table with no law, or an illegal step, fails on the host)
            // but the block table is not folded into the "logits", and neither is anything else that
            // is a function of the BATCH rather than of the sequence. On the device a sequence's
            // token does not move because another sequence shares its step — that is the whole claim
            // the batched gate makes — so a stub whose tokens moved with the batch would be more
            // sensitive than the thing it stands in for, and the gate would be testing the stub.
            if key.ends_with(".block_table")
                || key.ends_with(".cu")
                || key.ends_with(".naccept")
                || key.ends_with(".cache_idx")
                || key.ends_with(".sidx")
            {
                continue;
            }
            folded.extend(bytes);
        }
        // Nothing is folded here. Every table is BUILT — which is the point of this function, so an
        // illegal step or a key with no law fails on the host — and the `ids` are folded by the STEP,
        // one row at a time, because at verify width row `i` belongs to sequence `i / width` and its
        // answer has to depend on the draft tokens before it. Folding `ids[i]` into sequence `i` here
        // was right at SPEC=0 and became a different "model" per width, which made a verified run's
        // ids diverge from the SPEC=0 run's at the second step (2026-09-20) — the stub failing the
        // gate rather than the law.
        let _ = (folded, live);
        Ok(())
    }

    /// A decode step at a rectangle (`band` `None`) or in a ROW BAND of `band` rows (C2), with the
    /// device pass's own rules: every table built by the shared laws, each slot's rows read out of
    /// the step's ids buffer by the `cu` prefix, the padding fed nothing a live row reads.
    #[allow(clippy::too_many_lines)]
    fn decode_rows(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
        band: Option<u32>,
    ) -> Result<StepOutcome, ForwardError> {
        // A decode step overwrites the arena the last chunk's rows lived in (V7).
        self.last_prefill = None;
        if slots.is_empty() {
            return Err(ForwardError::Device(
                "a decode step with no slots".to_owned(),
            ));
        }
        if width == 0 {
            return Err(ForwardError::Device(
                "a decode step of verify width 0: every step feeds at least the token the last one \
                 committed"
                    .to_owned(),
            ));
        }
        // One width per bucket. A slot that drafted fewer belongs in a narrower bucket, and padding
        // its own rows would write KV at positions the next step has to un-write. A ROW BAND (C2)
        // takes each slot at its own `1..=width` rows instead, packed by `cu`.
        match band {
            None => {
                if let Some(bad) = slots.iter().find(|s| s.rows() != width) {
                    return Err(ForwardError::Device(format!(
                        "{}: {} row(s) in a width-{width} bucket; every live slot of a verify \
                         step feeds the bucket's own width",
                        bad.seq,
                        bad.rows()
                    )));
                }
            }
            Some(rows) => {
                if let Some(bad) = slots.iter().find(|s| s.rows() > width) {
                    return Err(ForwardError::Device(format!(
                        "{}: {} row(s) in a band of max width {width}",
                        bad.seq,
                        bad.rows()
                    )));
                }
                let fed: u32 = slots.iter().map(DecodeSlot::rows).sum();
                if !self.bands.contains(&rows) || fed > rows {
                    return Err(ForwardError::NoPlan {
                        phase: Phase::Decode,
                        rows,
                        detail: format!(
                            "{fed} row(s) in a {rows}-row band; planned bands are {:?}",
                            self.bands
                        ),
                    });
                }
            }
        }
        if slots.len() as u32 > batch {
            return Err(ForwardError::Device(format!(
                "{} live sequence(s) in a batch-{batch} step",
                slots.len()
            )));
        }
        if !self.decode_batches.contains(&batch) {
            return Err(ForwardError::NoPlan {
                phase: Phase::Decode,
                rows: batch,
                detail: format!("planned decode batches are {:?}", self.decode_batches),
            });
        }
        if !self.decode_widths.contains(&width) {
            return Err(ForwardError::NoPlan {
                phase: Phase::Decode,
                rows: batch * width,
                detail: format!("planned verify widths are {:?}", self.decode_widths),
            });
        }
        // Per-sequence facts, live slots first, then padding on the null slot.
        let mut facts: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(slots.len());
        for s in slots {
            let session = self.live.get(&s.seq).ok_or(ForwardError::Unknown(s.seq))?;
            if session.done != s.context_len {
                return Err(ForwardError::Device(format!(
                    "{}: the batch says {} token(s) are resident and the engine has {}",
                    s.seq, s.context_len, session.done
                )));
            }
            facts.push((session.slot, session.done, session.kv_blocks.clone()));
        }
        let mut seqs: Vec<SeqFacts<'_>> = facts
            .iter()
            .zip(slots)
            .map(|((slot, done, blocks), s)| SeqFacts {
                state_slot: *slot,
                done_before: *done,
                kv_blocks: blocks,
                live_rows: s.rows(),
                rope: None,
            })
            .collect();
        seqs.resize(batch as usize, SeqFacts::padding());
        let rows = band.unwrap_or(batch * width);
        // Row order: the slots' rows back to back — `cu`'s prefix — then the padding. In a
        // rectangle every slot is `width` rows, so this is slot `i` at `i * width` as it always was.
        let mut ids: Vec<TokenId> = slots.iter().flat_map(DecodeSlot::row_ids).collect();
        ids.resize(
            rows as usize,
            if band.is_some() { self.band_pad_id } else { 0 },
        );
        // Where each slot's rows start in `ids`: the `cu` prefix (a rectangle's `i * width`), or,
        // under the negative control, the rectangle's stride in a band.
        let mut starts: Vec<usize> = Vec::with_capacity(slots.len());
        let mut at = 0usize;
        for (i, s) in slots.iter().enumerate() {
            starts.push(if band.is_some() && self.strided_band_reads {
                i * width as usize
            } else {
                at
            });
            at += s.rows() as usize;
        }
        let mut committed_before: Vec<u32> = slots.iter().map(|s| s.committed_before).collect();
        committed_before.resize(batch as usize, 1);
        let index = StepIndex {
            phase: Phase::Decode,
            rows,
            max_blocks: self.facts.max_blocks,
            seqs: &seqs,
            ids: &ids,
            committed_before: &committed_before,
            band_width: band.map(|_| width),
        };
        let live: Vec<SeqId> = slots.iter().map(|s| s.seq).collect();
        // The same rule the device applies, so the host tests exercise the which-path: the ids are
        // already there when the shape, the slot order and the tokens are the ones the last step
        // left behind.
        let inputs_now: Vec<TokenId> = slots.iter().map(|s| s.input).collect();
        let feed = match &self.fed {
            Some((b, seqs, toks))
                if band.is_none() && *b == batch && *seqs == live && *toks == inputs_now =>
            {
                TokenFeed::Device
            }
            _ => TokenFeed::Host,
        };
        self.run_indices(&index, &live)?;
        if band.is_some() {
            self.check_band_padding(&index)?;
        }
        // The stub's "model": fold each fed id into a running hash and sample from it, so row `i`'s
        // answer depends on `draft[..i]`. That is the property the acceptance law is about, which
        // means the host tests exercise a real rejection rather than a fixed answer. The fold is
        // LOCAL until the verdict is in: a rejected row's contribution is dropped exactly as its KV
        // is overwritten, by continuing from what actually committed.
        let mut commits = Vec::with_capacity(slots.len());
        let mut fed_back = Vec::with_capacity(slots.len());
        self.last_rows.clear();
        let mut row_hashes: Vec<Vec<u64>> = Vec::with_capacity(slots.len());
        for (slot_index, s) in slots.iter().enumerate() {
            let before = self
                .live
                .get(&s.seq)
                .ok_or(ForwardError::Unknown(s.seq))?
                .hash;
            let mut h = before;
            let mut rows = Vec::with_capacity(s.rows() as usize);
            let mut hashes = Vec::with_capacity(s.rows() as usize);
            // THE ROWS AS THE DEVICE READS THEM: out of the step's ids buffer at this slot's `cu`
            // offset, not out of the slot's own list — so a row the layout puts somewhere else is
            // a different "model" here exactly as it would be on the card.
            for j in 0..s.rows() as usize {
                let row = starts[slot_index] + j;
                let id = ids.get(row).copied().unwrap_or(0);
                h = h.wrapping_mul(31).wrapping_add(u64::from(id));
                rows.push(Self::sample_from(h, self.facts.vocab));
                hashes.push(h);
                // The row index within the STEP, which is what the dense fallback asks for.
                self.last_rows.insert(row as u32, h);
            }
            // THE LAW, as the real pass applies it (ws-f40): the argmax for a greedy slot, the
            // host sampler's per-position draw for a sampled one (`crate::forward::drawn_rows`).
            let verdict = match &s.decider {
                None => self.law.accept(&s.draft, &rows),
                Some(d) => {
                    let (vocab, request) = (self.facts.vocab, &s.sampling);
                    let targets =
                        crate::forward::drawn_rows(d.0.as_ref(), &s.draft, &rows, &mut |j| {
                            request.wants_row().then(|| {
                                Self::device_row(hashes[j], vocab, request, DEVICE_ROW_K as usize)
                            })
                        });
                    self.law.accept(&s.draft[..targets.len() - 1], &targets)
                }
            }
            .map_err(|e| ForwardError::Device(e.to_string()))?;
            row_hashes.push(hashes);
            // The history the NEXT step continues from: this step's input plus the draft tokens that
            // survived. The last committed token is not folded here — the next step feeds it.
            let mut kept = before.wrapping_mul(31).wrapping_add(u64::from(s.input));
            for t in &verdict.committed[..verdict.accepted as usize] {
                kept = kept.wrapping_mul(31).wrapping_add(u64::from(*t));
            }
            let session = self
                .live
                .get_mut(&s.seq)
                .ok_or(ForwardError::Unknown(s.seq))?;
            session.hash = kept;
            session.done += verdict.committed.len() as u32;
            session.committed = verdict.committed.len() as u32;
            // The ARGMAX of the last committed row is what the device left in its ids buffer (ws-f40:
            // a sampled walk may commit a draw there, which the next step must then upload).
            fed_back.push(rows[verdict.committed.len() - 1]);
            // ONE SEAM PER COMMITTED TOKEN — the mock's rows, in the order they committed, the
            // same law the real pass follows. The LAST is the row the host sampler decides; the
            // ones before it are the accepted prefix, whose tokens the acceptance law chose and
            // whose distributions are reported rather than re-drawn.
            let rows: Vec<Option<DeviceRow>> = (0..verdict.committed.len())
                .map(|j| {
                    s.sampling.wants_row().then(|| {
                        let hash = row_hashes[slot_index].get(j).copied().unwrap_or(kept);
                        Self::device_row(hash, self.facts.vocab, &s.sampling, DEVICE_ROW_K as usize)
                    })
                })
                .collect();
            commits.push(SlotCommit {
                seq: s.seq,
                accepted: verdict.accepted,
                tokens: verdict.committed,
                rows,
            });
        }
        self.fed = Some((batch, live, fed_back));
        Ok(StepOutcome {
            commits,
            rows: StepRows {
                phase: Phase::Decode,
                rows,
                width,
                samples: true,
                tree: false,
                ragged: band.is_some(),
            },
            launches: self.launches(BODY_LAUNCHES + HEAD_LAUNCHES),
            path: self.path(Phase::Decode),
            feed,
            drafts: Vec::new(),
            proposal_pending: Vec::new(),
        })
    }

    /// **A band's padding rows point at nothing** (C2's host test b), asked of the same laws the
    /// device builds its blob from: every row past the live `cu` span has `slot` −1 (`kv_append`
    /// writes nothing) and position 0, every padding SLOT has the null state slot, zero rows and
    /// `seqused_k` 0, and every live slot's `naccept` is at least 1.
    fn check_band_padding(&self, index: &StepIndex<'_>) -> Result<(), ForwardError> {
        let bad = |what: String| Err(ForwardError::Device(format!("band padding: {what}")));
        let i32s = |b: Vec<u8>| -> Vec<i32> {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let i64s = |b: Vec<u8>| -> Vec<i64> {
            b.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        };
        let table =
            |key: &str| index_table(key, index).map_err(|e| ForwardError::Device(e.to_string()));
        let cu = i32s(table("x.cu")?);
        let live = *cu.last().unwrap_or(&0) as usize;
        let slot = i64s(table("x.slot")?);
        let pos = i64s(table("x.pos")?);
        if slot.len() != index.rows as usize {
            return bad(format!(
                "{} slot entries for {} rows",
                slot.len(),
                index.rows
            ));
        }
        for r in live..index.rows as usize {
            if slot[r] != -1 || pos[r] != 0 {
                return bad(format!("row {r} (slot {}, pos {})", slot[r], pos[r]));
            }
        }
        let seqused = i32s(table("x.seqused_k")?);
        let naccept = i32s(table("x.naccept")?);
        let cache = i32s(table("x.cache_idx")?);
        for (n, seq) in index.seqs.iter().enumerate() {
            if !seq.live() && (seqused[n] != 0 || cache[n] != 0 || cu[n + 1] != cu[n]) {
                return bad(format!("padding slot {n} is not inert"));
            }
            if naccept[n] < 1 {
                return bad(format!("slot {n} has naccept {}", naccept[n]));
            }
        }
        Ok(())
    }

    /// The "logits": one token out of the session hash.
    /// The token this sequence's history implies. **Does not advance anything.**
    ///
    /// A function of the fed-id fold and nothing else, which is what makes the stub's SPEC=0 path and
    /// its verify path the same "model": a verify step samples `w` rows out of `w` different
    /// histories and then keeps one of them, so a sampler that advanced hidden state per call would
    /// make the two paths disagree — and the gate that says "a verified run commits the SPEC=0 run's
    /// ids" would be testing the stub's bookkeeping instead of the acceptance law. (Measured
    /// 2026-09-20: it did, and the ids diverged at token 3.)
    fn sample(&self, seq: SeqId) -> Result<TokenId, ForwardError> {
        let session = self.live.get(&seq).ok_or(ForwardError::Unknown(seq))?;
        Ok(Self::sample_from(session.hash, self.facts.vocab))
    }

    /// The stub's token for a given history, without advancing anything.
    ///
    /// A verify step samples `w` rows out of `w` different histories, so the sampler has to be a
    /// function of a hash rather than a method that advances one.
    fn sample_from(hash: u64, vocab: u32) -> TokenId {
        Self::token_of(
            hash.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
            vocab,
        )
    }

    /// Never 0 and never the last id, so a stub token is never mistaken for a padding id or for the
    /// kernel's "every logit was NaN" answer.
    fn token_of(hash: u64, vocab: u32) -> TokenId {
        1 + (hash >> 11) as TokenId % (vocab - 2)
    }

    /// The distinguished ids of a row, best first, and their logits.
    ///
    /// The stub is not a model, and this is not a distribution anybody should read meaning into.
    /// What it IS is a row with the SHAPE of a logit row — one clear winner, a handful of
    /// runners-up, and a flat floor — so that every branch of the sampling path can be exercised
    /// on the host: the seam, its refusal, the dense fallback, the noise, the tie rule.
    fn candidates(hash: u64, vocab: u32) -> Vec<(TokenId, f32)> {
        let mut out: Vec<(TokenId, f32)> = Vec::with_capacity(NULL_CANDIDATES);
        let top = Self::sample_from(hash, vocab);
        out.push((top, NULL_PEAK));
        let mut h = hash;
        while out.len() < NULL_CANDIDATES {
            h = h
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let id = Self::token_of(h, vocab);
            if out.iter().any(|(existing, _)| *existing == id) {
                continue;
            }
            let value = NULL_PEAK - 0.5 * out.len() as f32;
            out.push((id, value));
        }
        out
    }

    /// The seam a head would write for this row: the K largest values and ids in the argmax's own
    /// total order, both log-sum-exps, and the raw logits at the penalty ids.
    ///
    /// Computed in closed form rather than by selecting over a materialised row, because the
    /// background is one constant: the tail contributes `(vocab - candidates) * exp(bg)` to each
    /// sum and the ids below the candidates are simply the lowest ones not already taken.
    fn device_row(hash: u64, vocab: u32, request: &RowRequest, k: usize) -> DeviceRow {
        let mut cands = Self::candidates(hash, vocab);
        cands.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let mut top_values: Vec<f32> = cands.iter().map(|(_, v)| *v).collect();
        let mut top_ids: Vec<TokenId> = cands.iter().map(|(id, _)| *id).collect();
        // The rest of the seam is background-valued, and a tie at the background is broken by the
        // LOWEST id — the same rule `saber_argmax_bf16` reduces with.
        let mut id = 0u32;
        while top_ids.len() < k && id < vocab {
            if !top_ids.contains(&id) {
                top_ids.push(id);
                top_values.push(NULL_BACKGROUND);
            }
            id += 1;
        }
        let tail = f64::from(vocab as i32 - cands.len() as i32).max(0.0);
        let lse = |t: f32| -> f32 {
            let scale = f64::from(t);
            let max = f64::from(NULL_PEAK) / scale;
            let mut sum = 0.0f64;
            for (_, v) in &cands {
                sum += (f64::from(*v) / scale - max).exp();
            }
            sum += tail * (f64::from(NULL_BACKGROUND) / scale - max).exp();
            (max + sum.ln()) as f32
        };
        let penalised = request
            .penalty_ids
            .iter()
            .map(|token| {
                let v = cands
                    .iter()
                    .find(|(id, _)| id == token)
                    .map_or(NULL_BACKGROUND, |(_, v)| *v);
                (*token, v)
            })
            .collect();
        DeviceRow {
            top_values,
            top_ids,
            logsumexp: lse(1.0),
            logsumexp_at_temperature: request.temperature.map(lse),
            penalised,
            vocab_size: vocab as usize,
        }
    }

    /// The whole row, materialised. What `dense_row` hands the sampler when the seam refuses.
    fn dense(hash: u64, vocab: u32) -> Vec<f32> {
        let mut row = vec![NULL_BACKGROUND; vocab as usize];
        for (id, value) in Self::candidates(hash, vocab) {
            row[id as usize] = value;
        }
        row
    }
}

impl ForwardRunner for NullForward {
    fn facts(&self) -> &ForwardFacts {
        &self.facts
    }

    fn admit(
        &mut self,
        seq: SeqId,
        prompt_len: u32,
        kv_blocks: &[u32],
    ) -> Result<(), ForwardError> {
        if self.live.contains_key(&seq) {
            return Err(ForwardError::Device(format!(
                "{seq} is already admitted to the engine"
            )));
        }
        let slot = self.free_slots.pop().ok_or(ForwardError::AtCapacity {
            live: self.live.len() as u32,
            max: self.facts.max_seqs,
        })?;
        if let Some(bad) = kv_blocks.iter().find(|b| **b >= self.facts.kv_pool_blocks) {
            self.free_slots.push(slot);
            return Err(ForwardError::Device(format!(
                "block id {bad} is outside the engine's pool of {} blocks",
                self.facts.kv_pool_blocks
            )));
        }
        self.live.insert(
            seq,
            Session {
                hash: 0xcbf2_9ce4_8422_2325 ^ u64::from(prompt_len),
                done: 0,
                slot,
                committed: 1,
                kv_blocks: kv_blocks.to_vec(),
            },
        );
        Ok(())
    }

    /// The fork's device half, with the device taken out: the branch gets a slot of its own
    /// holding a COPY of the leader's state (this stub's state is one hash, which is the whole
    /// point — a branch that got a fresh one would produce a different stream and the gate would
    /// see it), the leader's residency, and its own block table.
    ///
    /// Every refusal the real one makes is made here, so a host test can exercise them: a branch
    /// that is already admitted, a leader that is not, a page outside the pool, a shared prefix
    /// that is not the leader's own pages, and a copy-on-write source with nowhere to land.
    fn fork(
        &mut self,
        leader: SeqId,
        branch: SeqId,
        kv_blocks: &[u32],
        shared_prefix: u32,
        cow_source: Option<u32>,
    ) -> Result<(), ForwardError> {
        if self.live.contains_key(&branch) {
            return Err(ForwardError::Device(format!(
                "{branch} is already admitted to the engine"
            )));
        }
        let from = self
            .live
            .get(&leader)
            .ok_or(ForwardError::Unknown(leader))?
            .clone();
        if let Some(bad) = kv_blocks.iter().find(|b| **b >= self.facts.kv_pool_blocks) {
            return Err(ForwardError::Device(format!(
                "block id {bad} is outside the engine's pool of {} blocks",
                self.facts.kv_pool_blocks
            )));
        }
        let shared = shared_prefix as usize;
        if shared > kv_blocks.len() || shared > from.kv_blocks.len() {
            return Err(ForwardError::Device(format!(
                "{branch} shares {shared} pages of a {}-page table with a leader holding {}",
                kv_blocks.len(),
                from.kv_blocks.len()
            )));
        }
        if kv_blocks[..shared] != from.kv_blocks[..shared] {
            return Err(ForwardError::Device(format!(
                "{branch}'s shared prefix is not {leader}'s pages"
            )));
        }
        if let Some(src) = cow_source {
            if kv_blocks.len() <= shared {
                return Err(ForwardError::Device(format!(
                    "{branch} owes a copy of block {src} and has no private block to put it in"
                )));
            }
            if from.kv_blocks.get(shared) != Some(&src) {
                return Err(ForwardError::Device(format!(
                    "{branch}'s copy-on-write source {src} is not {leader}'s page at that position"
                )));
            }
        }
        let slot = self.free_slots.pop().ok_or(ForwardError::AtCapacity {
            live: self.live.len() as u32,
            max: self.facts.max_seqs,
        })?;
        self.live.insert(
            branch,
            Session {
                hash: from.hash,
                done: from.done,
                slot,
                committed: from.committed,
                kv_blocks: kv_blocks.to_vec(),
            },
        );
        Ok(())
    }

    fn can_fork(&self) -> bool {
        self.forks
    }

    fn can_park(&self) -> bool {
        true
    }

    fn parks_async(&self) -> bool {
        self.transfer_steps > 0
    }

    /// Issue a park that takes `transfer_steps` polls. Nothing moves; the point is the SHAPE of the
    /// seam — the blob is built now, handed over later, and the caller must not free the session's
    /// pages or its slot in between.
    fn park_begin(&mut self, seq: SeqId) -> Result<(), ForwardError> {
        if self.transfer_steps == 0 {
            return Err(ForwardError::CannotPark);
        }
        if self.in_flight.is_some() {
            return Err(ForwardError::TransferBusy);
        }
        let blob = self.park(seq)?.encode();
        self.in_flight = Some((seq, self.transfer_steps, Some(blob)));
        Ok(())
    }

    fn park_poll(&mut self, seq: SeqId) -> Result<Option<Vec<u8>>, ForwardError> {
        match &mut self.in_flight {
            Some((s, left, blob)) if *s == seq => {
                if *left > 1 {
                    *left -= 1;
                    return Ok(None);
                }
                let blob = blob.take().ok_or(ForwardError::CannotPark)?;
                self.in_flight = None;
                Ok(Some(blob))
            }
            _ => Err(ForwardError::Image(format!("{seq} has no park in flight"))),
        }
    }

    /// The restore's mirror. The bytes go back **now** — a stub has no DMA to wait for — and what
    /// the countdown models is the scheduler holding the sequence out of the step until the copy
    /// lands, which is the property under test.
    fn restore_begin(&mut self, seq: SeqId, blob: Vec<u8>) -> Result<(), ForwardError> {
        if self.transfer_steps == 0 {
            return Err(ForwardError::rejected(
                "this stub parks synchronously".to_owned(),
                blob,
            ));
        }
        let two = self.facts.park_staging_slots >= 2;
        if self.in_flight_restore.is_some() || (!two && self.in_flight.is_some()) {
            return Err(ForwardError::rejected("busy".to_owned(), blob));
        }
        let image = match crate::park::ParkImage::decode(&blob) {
            Ok(i) => i,
            Err(e) => return Err(ForwardError::rejected(e.to_string(), blob)),
        };
        if let Err(e) = self.restore(seq, &image) {
            return Err(ForwardError::rejected(e.to_string(), blob));
        }
        // The stub keeps the blob so a failure can hand it back, exactly as the engine does.
        let slot = if two {
            &mut self.in_flight_restore
        } else {
            &mut self.in_flight
        };
        *slot = Some((seq, self.transfer_steps, Some(blob)));
        Ok(())
    }

    fn restore_poll(&mut self, seq: SeqId) -> Result<bool, ForwardError> {
        let two = self.facts.park_staging_slots >= 2;
        let slot = if two {
            &mut self.in_flight_restore
        } else {
            &mut self.in_flight
        };
        match slot {
            Some((s, left, _)) if *s == seq => {
                if *left > 1 {
                    *left -= 1;
                    return Ok(false);
                }
                *slot = None;
                Ok(true)
            }
            _ => Err(ForwardError::Image(format!(
                "{seq} has no restore in flight"
            ))),
        }
    }

    /// The stub's session state, in the shape the device's is: one region per KV page and one for
    /// the GDN slot.
    ///
    /// There is no device here, so the "contents" of those regions are the running hash and the
    /// token count — the two things that actually carry this sequence forward. They are written
    /// into the GDN region and the KV regions are filled with a per-page pattern derived from them,
    /// so an image of the right shape whose bytes are wrong is still wrong, and the region count and
    /// lengths a restore has to match are the real ones.
    fn park(&mut self, seq: SeqId) -> Result<crate::park::ParkImage, ForwardError> {
        use saber_mem::park::{ImageDigests, RegionKind, RegionSpec};
        let session = self.live.get(&seq).ok_or(ForwardError::Unknown(seq))?;
        let mut layout = Vec::with_capacity(session.kv_blocks.len() + 1);
        let mut bytes = Vec::new();
        for (i, _) in session.kv_blocks.iter().enumerate() {
            layout.push(RegionSpec {
                kind: RegionKind::Kv,
                len: NULL_KV_BLOCK_BYTES,
            });
            let fill = (session.hash ^ (i as u64)).to_le_bytes();
            for k in 0..NULL_KV_BLOCK_BYTES as usize {
                bytes.push(fill[k % 8]);
            }
        }
        layout.push(RegionSpec {
            kind: RegionKind::GdnState,
            len: NULL_GDN_SLOT_BYTES,
        });
        let mut gdn = Vec::with_capacity(NULL_GDN_SLOT_BYTES as usize);
        gdn.extend_from_slice(&session.hash.to_le_bytes());
        gdn.extend_from_slice(&u64::from(session.done).to_le_bytes());
        gdn.resize(NULL_GDN_SLOT_BYTES as usize, 0);
        bytes.extend_from_slice(&gdn);
        let total = bytes.len() as u64;
        let chunks = saber_mem::park::chunk_digests(&bytes, 2);
        Ok(crate::park::ParkImage {
            // The stub keeps its count inside the payload too (the GDN region above), which is
            // exactly why the host gates could not see the device path forgetting it. Set here as
            // well so the two agree and the field is exercised.
            resident: session.done,
            // The stub has no state slots, so this is the round trip's own value and not a slot
            // offset: a restore that dropped it would come back as 1 and the device path's
            // `session.committed` would be wrong on the NEXT park (ws-k14).
            committed: session.committed.max(1),
            digests: ImageDigests {
                layout,
                bytes: total,
                chunk_bytes: saber_mem::park::DIGEST_CHUNK as u32,
                sha256: saber_mem::park::fold_chunk_digests(total as usize, &chunks),
                chunks,
            },
            bytes,
        })
    }

    fn restore(&mut self, seq: SeqId, image: &crate::park::ParkImage) -> Result<(), ForwardError> {
        use saber_mem::park::RegionKind;
        image
            .digests
            .check_self()
            .map_err(|e| ForwardError::Image(e.to_string()))?;
        let session = self.live.get(&seq).ok_or(ForwardError::Unknown(seq))?;
        // The shape has to match what THIS admission was given, which is the check the device half
        // delegates to `publish_from`.
        let want_kv = session.kv_blocks.len();
        let got_kv = image
            .digests
            .layout
            .iter()
            .filter(|r| r.kind == RegionKind::Kv)
            .count();
        if got_kv != want_kv || image.digests.layout.len() != want_kv + 1 {
            return Err(ForwardError::Image(format!(
                "{seq} has {want_kv} page(s) and one slot; the image has {got_kv} page(s) and {}                  region(s) in total",
                image.digests.layout.len()
            )));
        }
        if image.bytes.len() as u64 != image.digests.bytes {
            return Err(ForwardError::Image(format!(
                "{seq}: the image declares {} payload bytes and carries {}",
                image.digests.bytes,
                image.bytes.len()
            )));
        }
        // The stub of the device has to check what the device checks: every chunk against the
        // digest the park took, BEFORE any of it is applied. On the card this is
        // `saber_mem::park::publish_from`, which verifies a chunk and only then issues it.
        if image.digests.verified() {
            let got = saber_mem::park::chunk_digests(&image.bytes, 2);
            if got != image.digests.chunks {
                let bad = got
                    .iter()
                    .zip(&image.digests.chunks)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                return Err(ForwardError::Image(format!(
                    "{seq}: chunk {bad} of the image does not match the digest taken when it was                      parked; nothing was applied"
                )));
            }
        }
        let gdn_at = image.bytes.len() - NULL_GDN_SLOT_BYTES as usize;
        let hash = u64::from_le_bytes(image.bytes[gdn_at..gdn_at + 8].try_into().expect("8 bytes"));
        let done = u64::from_le_bytes(
            image.bytes[gdn_at + 8..gdn_at + 16]
                .try_into()
                .expect("8 bytes"),
        ) as u32;
        let session = self.live.get_mut(&seq).ok_or(ForwardError::Unknown(seq))?;
        session.hash = hash;
        session.done = done;
        // Where the live recurrent snapshot is in the NEW run, from the image and not from this
        // admission (`crate::device::session_regions`). The stub has no slots to publish into, so
        // what this exercises on the host is that the number survives the round trip at all.
        session.committed = image.committed.max(1);
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        if let Some(s) = self.live.remove(&seq) {
            self.free_slots.push(s.slot);
        }
        self.fed = None;
    }

    fn prefill(
        &mut self,
        seq: SeqId,
        chunk: &[TokenId],
        tokens_done_before: u32,
        last: bool,
        sampling: &RowRequest,
    ) -> Result<StepOutcome, ForwardError> {
        let live_rows = u32::try_from(chunk.len())
            .map_err(|_| ForwardError::Device("a chunk longer than u32".to_owned()))?;
        let rows = self.prefill_shape(live_rows)?;
        let (slot, done, blocks) = {
            let session = self.live.get(&seq).ok_or(ForwardError::Unknown(seq))?;
            if session.done != tokens_done_before {
                return Err(ForwardError::Device(format!(
                    "{seq}: the chunk says {tokens_done_before} token(s) are resident and the \
                     engine has {}",
                    session.done
                )));
            }
            (session.slot, session.done, session.kv_blocks.clone())
        };
        let mut ids = chunk.to_vec();
        ids.resize(rows as usize, 0);
        let seqs = [SeqFacts {
            state_slot: slot,
            done_before: done,
            kv_blocks: &blocks,
            live_rows,
            rope: None,
        }];
        let index = StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: self.facts.max_blocks,
            seqs: &seqs,
            ids: &ids,
            // SPEC=0: every
            // slot committed exactly one token last step, which is what an empty table means.
            committed_before: &[],
            band_width: None,
        };
        self.run_indices(&index, &[seq])?;
        let session = self.live.get_mut(&seq).ok_or(ForwardError::Unknown(seq))?;
        session.done += live_rows;
        // The base slot again, exactly as the device path (`crate::device`'s prefill).
        session.committed = 1;
        let mut row_hashes = Vec::with_capacity(chunk.len());
        for t in chunk {
            session.hash = session.hash.wrapping_mul(31).wrapping_add(u64::from(*t));
            row_hashes.push(session.hash);
        }
        self.last_prefill = Some((seq, row_hashes));
        // A prompt chunk rewrote the ids, so the next decode step feeds from the host.
        self.fed = None;
        self.last_rows.clear();
        let commits = if last {
            let hash = self.live.get(&seq).ok_or(ForwardError::Unknown(seq))?.hash;
            self.last_rows.insert(0, hash);
            vec![SlotCommit {
                seq,
                accepted: 0,
                tokens: vec![self.sample(seq)?],
                rows: vec![sampling.wants_row().then(|| {
                    Self::device_row(hash, self.facts.vocab, sampling, DEVICE_ROW_K as usize)
                })],
            }]
        } else {
            Vec::new()
        };
        Ok(StepOutcome {
            commits,
            rows: StepRows {
                phase: Phase::Prefill,
                rows,
                width: 1,
                samples: last,
                tree: false,
                ragged: false,
            },
            launches: self.launches(BODY_LAUNCHES + if last { HEAD_LAUNCHES } else { 0 }),
            path: self.path(Phase::Prefill),
            // A prompt is never on the device already.
            feed: TokenFeed::Host,
            drafts: Vec::new(),
            proposal_pending: Vec::new(),
        })
    }

    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        self.decode_rows(slots, batch, width, None)
    }

    fn decode_band(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        max_width: u32,
        rows: u32,
    ) -> Result<StepOutcome, ForwardError> {
        self.decode_rows(slots, batch, max_width, Some(rows))
    }

    fn prompt_rows(
        &mut self,
        seq: SeqId,
        targets: &[TokenId],
    ) -> Result<Vec<crate::forward::PromptRow>, ForwardError> {
        let Some((last, hashes)) = &self.last_prefill else {
            return Err(ForwardError::Device(
                "no prefill chunk has run, so there are no prompt rows to score".to_owned(),
            ));
        };
        if *last != seq || targets.len() > hashes.len() {
            return Err(ForwardError::Device(format!(
                "prompt rows for {seq} ({} of them) and the last chunk was {last}'s {} row(s)",
                targets.len(),
                hashes.len()
            )));
        }
        let vocab = self.facts.vocab;
        let mut out = Vec::with_capacity(targets.len());
        for (h, t) in hashes.iter().zip(targets) {
            let request = RowRequest {
                temperature: Some(1.0),
                penalty_ids: vec![*t],
            };
            let row = Self::device_row(*h, vocab, &request, DEVICE_ROW_K as usize);
            let x = saber_serve::prompt_logprobs::target_logit(&row, *t).ok_or_else(|| {
                ForwardError::Device("the seam carried no logit for the target".to_owned())
            })?;
            let rank = match saber_serve::prompt_logprobs::rank_from_seam(&row, x) {
                Some(r) => r,
                None => {
                    self.dense_prompt_ranks += 1;
                    saber_serve::prompt_logprobs::rank_from_dense(&Self::dense(*h, vocab), x)
                }
            };
            out.push(crate::forward::PromptRow {
                target: *t,
                row,
                rank,
            });
        }
        Ok(out)
    }

    fn prompt_rows_k(&self) -> Option<u32> {
        Some(DEVICE_ROW_K)
    }

    /// The row the seam refused, materialised. The stub keeps the history behind every sampled
    /// row of the step that just ran, which is the same lifetime the real engine's head arena has.
    fn dense_row(&mut self, row: u32) -> Result<Vec<f32>, ForwardError> {
        match self.last_rows.get(&row) {
            Some(hash) => Ok(Self::dense(*hash, self.facts.vocab)),
            None => Err(ForwardError::Device(format!(
                "no row {row} in the step that just ran"
            ))),
        }
    }

    fn declared_launches(&self, rows: StepRows) -> Option<u32> {
        self.declared.get(&rows).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub() -> NullForward {
        NullForward::new(&[2048], &[1, 2, 4], 4)
    }

    #[test]
    fn a_short_prompt_is_padded_up_into_its_bucket_and_says_so() {
        let mut f = stub();
        f.admit(SeqId(1), 234, &[0, 1, 2]).expect("admit");
        let out = f
            .prefill(SeqId(1), &vec![7u32; 234], 0, true, &RowRequest::greedy())
            .expect("chunk");
        assert_eq!(out.rows.rows, 2048, "a 234-token prompt is one p2048 chunk");
        assert!(out.rows.samples, "the last chunk is the one that samples");
        assert_eq!(f.declared_launches(out.rows), Some(out.launches));
        assert_eq!(out.committed(), 1);
        assert_eq!(out.path, StepPath::Eager, "prefill is not captured");
    }

    #[test]
    fn the_slot_free_list_reserves_the_kernels_null_slot() {
        let mut f = NullForward::new(&[64], &[1], 1);
        f.admit(SeqId(1), 8, &[0]).expect("first");
        let err = f.admit(SeqId(2), 8, &[0]).expect_err("out of slots");
        assert!(matches!(err, ForwardError::AtCapacity { live: 1, max: 1 }));
        f.release(SeqId(1));
        f.admit(SeqId(2), 8, &[0]).expect("the slot came back");
        assert_eq!(f.session(SeqId(2)).expect("live").0, 1, "slot 1, never 0");
    }

    #[test]
    fn a_block_id_outside_the_pool_is_refused_at_admission() {
        let mut f = NullForward::new(&[64], &[1], 1);
        let pool = f.facts().kv_pool_blocks;
        let err = f.admit(SeqId(1), 8, &[pool]).expect_err("outside the pool");
        assert!(
            err.to_string().contains("outside the engine's pool"),
            "{err}"
        );
    }

    /// The claim the batched gate makes, at the stub level: a sequence's tokens do not depend on who
    /// shares its step.
    #[test]
    fn a_sequence_decodes_the_same_ids_alone_and_in_a_batch() {
        let run = |batch: u32, others: usize| {
            let mut f = NullForward::new(&[64], &[1, 2, 4], 4);
            let me = SeqId(1);
            f.admit(me, 4, &[0]).expect("admit");
            let mut all = vec![me];
            for i in 0..others {
                let s = SeqId(10 + i as u64);
                f.admit(s, 4, &[1 + i as u32]).expect("admit other");
                f.prefill(s, &[9, 9, 9, 9], 0, true, &RowRequest::greedy())
                    .expect("prefill other");
                all.push(s);
            }
            let mut out = vec![f
                .prefill(me, &[1, 2, 3, 4], 0, true, &RowRequest::greedy())
                .expect("prefill")
                .token()
                .expect("a token")];
            for _ in 0..6 {
                let slots: Vec<DecodeSlot> = all
                    .iter()
                    .map(|s| {
                        DecodeSlot::single(
                            *s,
                            *out.last().expect("non-empty"),
                            f.session(*s).expect("live").1,
                        )
                    })
                    .collect();
                let step = f.decode(&slots, batch, 1).expect("decode");
                out.push(step.commits[0].tokens[0]);
            }
            out
        };
        assert_eq!(run(1, 0), run(2, 1), "one of two is the same as one alone");
        assert_eq!(run(1, 0), run(4, 3), "one of four, too");
    }

    #[test]
    fn a_decode_step_wider_than_its_bucket_is_refused() {
        let mut f = stub();
        f.admit(SeqId(1), 4, &[0]).expect("admit");
        f.admit(SeqId(2), 4, &[1]).expect("admit");
        f.prefill(SeqId(1), &[1, 2, 3, 4], 0, true, &RowRequest::greedy())
            .expect("p1");
        f.prefill(SeqId(2), &[1, 2, 3, 4], 0, true, &RowRequest::greedy())
            .expect("p2");
        let slots = [
            DecodeSlot::single(SeqId(1), 5, 4),
            DecodeSlot::single(SeqId(2), 5, 4),
        ];
        let err = f.decode(&slots, 1, 1).expect_err("two in a batch of one");
        assert!(err.to_string().contains("batch-1 step"), "{err}");
        // ...and a batch nobody planned is a refusal, not a silent eager fallback.
        let err = f.decode(&slots, 3, 1).expect_err("unplanned batch");
        assert!(matches!(err, ForwardError::NoPlan { rows: 3, .. }));
    }

    #[test]
    fn the_which_path_flag_is_the_only_difference_between_graph_and_eager() {
        let mut graph = stub();
        let mut eager = stub().without_capture();
        for f in [&mut graph, &mut eager] {
            f.admit(SeqId(1), 4, &[0]).expect("admit");
            f.prefill(SeqId(1), &[1, 2, 3, 4], 0, true, &RowRequest::greedy())
                .expect("prefill");
        }
        let slots = [DecodeSlot::single(SeqId(1), 5, 4)];
        let g = graph.decode(&slots, 1, 1).expect("graph");
        let e = eager.decode(&slots, 1, 1).expect("eager");
        assert_eq!(g.path, StepPath::Graph);
        assert_eq!(e.path, StepPath::Eager);
        assert_eq!(g.commits, e.commits, "the path must not change the ids");
        assert!(graph.facts().captured.len() >= 3, "three decode shapes");
        assert!(eager.facts().captured.is_empty());
    }
}
