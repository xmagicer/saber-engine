//! The index tables a step's kernels read — the session bookkeeping the graph does not model.
//!
//! Each one is a fact about the **step**: not about the model, and not about one sequence, because a
//! decode step at batch 8 is eight sessions at once. Every table is therefore `n_seqs` wide, each
//! has its own width and dtype, and a wrong one is a kernel walking off a buffer. They are
//! **The laws live in `saber_exec::forward`** since ws-h15, where the four callers that had each
//! re-derived them share one copy — this module's were the best of the four and are the ones that
//! moved. What stays here is the engine's adapter onto them: its own `Phase` and `TokenId`, its own
//! error type for a step loop to match on, and its tests.
//!
//! An index this module has no law for is a **refusal**, never a zero-filled buffer that a kernel
//! then reads as an offset. The engine builds every planned key once at construction for exactly
//! that reason: an unknown key is a startup error, not a wrong answer at step 400.
//!
//! # The three things the kernels state and nothing above them did
//!
//! 1. **GDN state slot 0 is the kernels' null slot.** `saber_gdn_conv_update` bails on `slot <= 0` and
//!    `saber_gdn_recurrent_update` on `si <= 0` — that is their NULL_BLOCK_ID / padding path. A *live*
//!    sequence pointed at slot 0 therefore runs, issues every dispatch, returns a token, and
//!    silently updates **no** GDN state in any of the 48 layers. So the engine reserves slot 0, hands
//!    live sequences slots from 1, and [`StepIndex::check`] refuses a live sequence at slot 0 — while
//!    *requiring* a padding slot to be exactly 0, because that is what makes the padding harmless.
//! 2. **`naccept` is 1, never 0.** Both decode kernels index `sidx[naccept[n] - 1]`, so a zero there
//!    reads one `i32` *before* the table. At SPEC=0 exactly one token commits per step, so the value
//!    is 1 and the state read is `sidx[0]`, which is the slot the same step writes back.
//! 3. **`seqused_k = 0` is how a padded row is skipped.** Both attention kernels open with
//!    `if (ctx <= 0) return;`, so a padding sequence reads no page and writes none — which is why
//!    its block-table row is never read and needs no page of its own.

use saber_exec::forward;
use saber_plan::Phase;
use saber_serve::engine::TokenId;

/// Why a table could not be built.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IndexError {
    /// An index key with no law here.
    #[error("no law for index table `{0}`: refusing to hand a kernel a zero-filled buffer")]
    Unknown(String),
    /// The session's numbers do not fit the table.
    #[error("index table `{key}` cannot be built: {detail}")]
    Shape {
        /// The key.
        key: String,
        /// What is wrong.
        detail: String,
    },
    /// The step itself is not one a set of tables can be built for.
    #[error("{0}")]
    Step(String),
}

/// One sequence's place in a step — the shared law's own type.
///
/// Re-exported rather than re-stated: a second definition of "padding is the null slot" is a second
/// thing to get wrong, and this crate's tests check the same struct the kernels' laws are written
/// against.
pub use saber_exec::forward::{AttnQLen, SeqFacts};

impl From<forward::IndexError> for IndexError {
    fn from(e: forward::IndexError) -> Self {
        match e {
            forward::IndexError::Unknown(k) => IndexError::Unknown(k),
            forward::IndexError::Shape { key, detail } => IndexError::Shape { key, detail },
            forward::IndexError::Step(d) => IndexError::Step(d),
        }
    }
}

/// Everything the index tables of one step are a projection of.
///
/// The engine's view: its `Phase`, its `TokenId`s, and [`Self::shared`] to hand the laws the form
/// they are written in.
#[derive(Clone, Copy, Debug)]
pub struct StepIndex<'a> {
    /// Which half of the step.
    pub phase: Phase,
    /// Rows the plan runs — the **planned** count, padding included.
    pub rows: u32,
    /// Block-table width the plans were built with.
    pub max_blocks: u32,
    /// One entry per slot of the bucket's batch, live slots first.
    pub seqs: &'a [SeqFacts<'a>],
    /// The step's input ids, `rows` of them (padding included).
    pub ids: &'a [TokenId],
    /// Tokens the **previous** step committed for each slot, one per slot, `1..=q_len`.
    ///
    /// This is `naccept`, and it is the one index law that is about the step BEFORE this one. Both
    /// decode kernels open with
    ///
    /// ```text
    /// // the initial state is the one the last accepted token of the previous step wrote
    /// const int it = naccept ? (naccept[n] - 1) : 0;
    /// const int si = sidx[(size_t)n * sidx_stride + it];
    /// ```
    ///
    /// so `naccept[n] - 1` selects, out of this sequence's `q_len` recurrent snapshots, the one its
    /// **last committed token** wrote — which is where this step's state must continue from. At
    /// SPEC=0 that is always 1 and the choice does not arise; at verify width `w` a step that
    /// committed `a + 1` tokens leaves the right state in snapshot `a`, and a `naccept` of anything
    /// else silently continues from another candidate's state. No launch count can see it and no
    /// shape check can catch it: the step runs, returns a token, and the sequence's recurrence has
    /// forked.
    ///
    /// Empty means "every slot committed one token", which is the SPEC=0 answer and the answer for
    /// a sequence's first decode step after its prompt.
    pub committed_before: &'a [u32],
    /// The shared packed-table width for a row band or single-slot scalar snapshot
    /// transition. Rectangles normally infer it from their live dimensions.
    pub band_width: Option<u32>,
}

impl<'a> StepIndex<'a> {
    /// A single-slot scalar decode may follow a wider commit. Its kernels read
    /// the previous recurrent snapshot at `sidx[naccept - 1]` and write the
    /// new one at `sidx[0]`. The conv kernel opens `cache_idx[0]` at token
    /// offset `naccept - 1`. For slot zero all per-slot stride products are
    /// zero, so the packed single-slot table law can supply a longer `sidx`
    /// without changing the actual one-row query, its KV position, or either
    /// kernel's one-row output geometry. Both ranks consume this staged blob.
    pub(crate) fn with_scalar_snapshot_span(
        mut self,
        state_width: u32,
    ) -> Result<Self, IndexError> {
        if self.phase == Phase::Decode && self.rows == 1 && self.band_width.is_none() {
            if let ([seq], [committed]) = (self.seqs, self.committed_before) {
                if seq.live_rows == 1 && *committed > 1 {
                    if *committed > state_width {
                        return Err(IndexError::Step(format!(
                            "scalar step needs snapshot {committed} but the session owns {state_width}"
                        )));
                    }
                    self.band_width = Some(*committed);
                }
            }
        }
        Ok(self)
    }

    /// Rows the live sequences feed, summed.
    #[must_use]
    pub fn live_rows(&self) -> u32 {
        self.seqs.iter().map(|s| s.live_rows).sum()
    }

    /// Live sequences in this step.
    #[must_use]
    pub fn live_seqs(&self) -> usize {
        self.seqs.iter().filter(|s| s.live()).count()
    }

    /// The ids as the laws want them (`i32`, the dtype the kernels read).
    fn ids_i32(&self) -> Vec<i32> {
        self.ids.iter().map(|t| *t as i32).collect()
    }

    fn shared(&self, ids: &'a [i32]) -> forward::StepIndex<'a> {
        forward::StepIndex {
            ids,
            ..self.shared_without_ids()
        }
    }

    /// The shared view of everything that is not the ids — for the laws that are a function of the
    /// slots alone. One conversion, so the phase mapping is not written twice.
    fn shared_without_ids(&self) -> forward::StepIndex<'a> {
        forward::StepIndex {
            phase: match self.phase {
                Phase::Prefill => saber_model::graph::Phase::Prefill,
                Phase::Decode => saber_model::graph::Phase::Decode,
            },
            rows: self.rows,
            max_blocks: self.max_blocks,
            seqs: self.seqs,
            ids: &[],
            band_width: self.band_width,
        }
    }

    /// The GDN state slot a PREFILL step points `chunk_scan` at, by the shared law
    /// ([`saber_exec::forward::StepIndex::prefill_state_slot`]).
    #[must_use]
    pub fn prefill_state_slot(&self) -> Option<u32> {
        self.shared_without_ids().prefill_state_slot()
    }

    /// Rows each sequence feeds: the verify width, derived from the plan's row count — the
    /// band's MAX width for a row band (C2), by the shared law.
    #[must_use]
    pub fn q_len(&self) -> u32 {
        self.shared_without_ids().q_len()
    }

    /// The previous step's acceptance, as the shared law wants it.
    ///
    /// Empty means "one token each", which is the SPEC=0 answer and the answer for a sequence's
    /// first decode step — and it is [`forward::Accepted::AllRows`] exactly when the step is one row
    /// wide, which is the only shape where the two readings coincide.
    fn accepted(&self) -> forward::Accepted<'_> {
        if self.committed_before.is_empty() {
            forward::Accepted::AllRows
        } else {
            forward::Accepted::PerSeq(self.committed_before)
        }
    }

    /// Everything that must hold before a single table is built — the shared check under this
    /// step's acceptance, plus the one thing the shared law cannot see.
    ///
    /// # Errors
    /// As the shared law: no slots, an id count that is not one per row, a live sequence at the null
    /// slot, a padding slot that is not at it, pages wider than the plans' table, or a
    /// committed-token count that is 0 or past the verify width. And, here: a table that is not one
    /// entry per slot, which the shared law pads with 1 rather than refusing — silently right at the
    /// slots it has and silently wrong at the rest.
    pub fn check(&self) -> Result<(), IndexError> {
        if !self.committed_before.is_empty() && self.committed_before.len() != self.seqs.len() {
            return Err(IndexError::Step(format!(
                "{} slot(s) but {} committed-token count(s): `naccept` is one per slot",
                self.seqs.len(),
                self.committed_before.len()
            )));
        }
        let ids = self.ids_i32();
        self.shared(&ids)
            .check_with(self.accepted())
            .map_err(IndexError::from)
    }
}

/// The shared encoder, for this module's tests.
#[cfg(test)]
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    forward::i32_bytes(v)
}

/// One index table for a step, or a refusal — [`saber_exec::forward::index_table`] for this step.
///
/// # Errors
/// As the shared law: a key with no law here.
pub fn index_table(key: &str, s: &StepIndex<'_>) -> Result<Vec<u8>, IndexError> {
    index_table_for(key, s, AttnQLen::Live)
}

/// [`index_table`] for a step whose prefill plan has been retargeted at the chunk's live rows.
///
/// The engine is the one caller with padded chunks, so it is the one caller that has to say which
/// `q_len` its attention descriptors carry: `AttnQLen::Exact(plan.prefill_q_len())`, the plan's
/// own number, so that the kernarg and `seqused_k` cannot drift apart. Passing the plan's number
/// rather than the step's live rows is what makes the `--no-live-rows` control a single flag: skip
/// the retarget and the plan still says the bucket's row count, so the whole pre-ws-h27 pair comes
/// back together.
///
/// # Errors
/// As the shared law: a key with no law here, or a `q_len` below a live sequence's rows.
pub fn index_table_for(
    key: &str,
    s: &StepIndex<'_>,
    attn: AttnQLen,
) -> Result<Vec<u8>, IndexError> {
    // Every key is the shared law's, including `naccept` — which ws-l14 owned for a day, because
    // `saber_exec::forward` wrote it as the step's own live rows and the kernels want the PREVIOUS
    // step's committed count (`sidx[naccept - 1]` names the snapshot the last committed token
    // wrote). ws-h18 took it as `Accepted::PerSeq`, a parameter rather than a field, so the engine
    // now states the fact and the law applies it.
    let ids = s.ids_i32();
    forward::index_table_for(key, &s.shared(&ids), s.accepted(), attn).map_err(IndexError::from)
}

// =====================================================================================
// The blob: what a STEP rewrites, and what it never does
// =====================================================================================

/// Whether an index table is a constant of the PLAN rather than a fact of the step.
///
/// `cos_sin` is the rope table: a function of `rotary_dim`, `rope_theta` and the context band, none
/// of which change while the engine is open. `k_scale` and `v_scale` are the checkpoint's
/// calibrated 1.0. Everything else — ids, `cu`, `cache_idx`, `block_table`, `seqused_k`, `pos`,
/// `slot`, `sidx`, `naccept` — says where the sequences are, and is rebuilt every step.
#[must_use]
pub fn step_invariant(key: &str) -> bool {
    key.ends_with(".cos_sin") || key.ends_with(".k_scale") || key.ends_with(".v_scale")
}

/// Whether an index table is a function of the sessions' **pages** rather than of their positions.
///
/// One table is: `block_table`, each sequence's page list padded to `max_blocks` with its own last
/// page. It is `max_ctx / 16` entries per slot per attention leg — 2,048 at `max_ctx` 32768 against
/// 256 at 4096 — and it changes only when a sequence gains a page (every 16 tokens) or when the
/// step's slots change. Every other step table changes every step, by construction: they say where
/// the sequences ARE.
#[must_use]
pub fn paged_table(key: &str) -> bool {
    key.ends_with(".block_table")
}

/// Which of the blob's three regions a key belongs to. The order here IS the layout order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    /// A constant of the plan: staged and uploaded once at open.
    Constant,
    /// A fact of the step: rewritten, uploaded and sent every step.
    Step,
    /// A fact of the sessions' pages: rewritten, uploaded and sent when it changes.
    Paged,
}

fn region_of(key: &str) -> Region {
    if step_invariant(key) {
        Region::Constant
    } else if paged_table(key) {
        Region::Paged
    } else {
        Region::Step
    }
}

/// A digest of everything `*.block_table` is a function of: the table width, the slot count, and
/// each slot's page list.
///
/// Equal digests mean a step would write the bytes that are already there, which is what lets the
/// engine skip the copy, the upload and the peer's copy of a 2 MiB table on the fifteen steps out of
/// sixteen where no sequence crossed a page boundary. It is **derived from the same inputs the law
/// reads** (`index_table_for`'s `.block_table` branch), so a law that starts reading something else
/// makes this wrong — which is why it lives here, next to it, and not in the caller.
#[must_use]
pub fn paged_digest(s: &StepIndex<'_>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(u64::from(s.max_blocks));
    eat(s.seqs.len() as u64);
    for seq in s.seqs {
        eat(seq.kv_blocks.len() as u64);
        for b in seq.kv_blocks {
            eat(u64::from(*b));
        }
    }
    h
}

/// CMP27AG `slotdedup`: the group a step-varying key's slot is shared within -- (narrowed by
/// `treeblob`, key side, suffix). The side is a packed chunk's index (`16 + k`) or
/// [`saber_exec::mixed::key_side`]'s (a fused step stages chunk, decode and per-row keys from
/// different steps, so they never share); the narrowed flag keeps `treeblob`'s tree-only keys
/// (written only at a tree step, at a narrower capacity) apart from the keys every step writes.
pub type AliasGroup = (bool, u8, String);

/// The [`AliasGroup`] of `key`.
#[must_use]
pub fn alias_group(key: &str, narrowed: bool) -> AliasGroup {
    if let Some((k, suffix)) = saber_exec::mixed::parse_packed_key(key) {
        let k = u8::try_from(k).unwrap_or(u8::MAX - 16);
        return (narrowed, 16u8.saturating_add(k), suffix.to_owned());
    }
    let suffix = key.rsplit('.').next().unwrap_or(key);
    (
        narrowed,
        saber_exec::mixed::key_side(key) as u8,
        suffix.to_owned(),
    )
}

/// Bytes every slot in the blob is aligned to.
const SLOT_ALIGN: u64 = 256;

/// Where each index table lives in the one device blob, and **which contiguous range of it a step
/// rewrites**.
///
/// The engine stages every table into one pinned buffer and uploads it with one DMA, because one
/// small copy per key would be "16 launches to move 32 bytes" at 16 attention layers. That argument
/// is about the NUMBER of copies and says nothing about their size — and the blob holds two kinds
/// of table:
///
/// * the **constants of the plan** ([`step_invariant`]), staged once at open. `cos_sin` is sized to
///   the whole context band, so at `max_ctx` 32768 it is 4,096 KiB of a 6,540 KiB blob;
/// * the **facts of the step** — where each sequence is, which pages it holds, what it feeds —
///   which really are rewritten every step.
///
/// Laying the constants out FIRST makes the second kind one contiguous range, so a step uploads
/// (and at TP=2 sends its peer) exactly the bytes it rewrote. Uploading the whole blob instead is
/// the same DMA count and 2.7× the bytes at `max_ctx` 32768 — which ws-h28 measured as
/// **+6.6 ms/step** at ctx 234..634 against an identical dispatch sequence with every counter zero.
///
/// A blob is a pure function of the plan's key set and the two SIZING probes, so this is host-only
/// and testable with no card.
#[derive(Debug, Clone)]
pub struct BlobLayout {
    /// key, offset, capacity — the widest bytes any step this engine can run will write there.
    pub table: Vec<(String, u64, usize)>,
    /// The step-invariant tables' own bytes, at their offsets, staged once at open.
    pub constants: Vec<(String, u64, Vec<u8>)>,
    /// Bytes laid out in total.
    pub len: usize,
    /// Where the step-varying region starts. Everything below it is staged and uploaded once.
    pub dynamic_at: usize,
    /// Where the PAGED region starts: `[paged_at, len)` is `block_table` and nothing else, so a
    /// step that no sequence gained a page in sends `[dynamic_at, paged_at)` and stops.
    pub paged_at: usize,
    /// The widest `*.ids` table, which is the token plane's and not part of the blob.
    pub ids_bytes: usize,
}

impl BlobLayout {
    /// Lay out every planned key, sized from the sizing probes.
    ///
    /// `probes` are the widest shapes this engine can run — the widest prefill chunk and the widest
    /// decode batch — because a slot must hold the largest table any step will put in it.
    ///
    /// # Errors
    /// A key with no law, a probe a table cannot be built for, or a layout whose two regions
    /// overlap (which is this function's own bug, and is checked rather than assumed).
    pub fn plan(
        keys: &std::collections::BTreeSet<String>,
        probes: &[StepIndex<'_>],
    ) -> Result<BlobLayout, IndexError> {
        Self::plan_narrowed(keys, probes, &std::collections::BTreeSet::new(), &[])
    }

    /// [`BlobLayout::plan`] with the step-varying keys in `narrow` sized over `narrow_probes`
    /// ONLY (CMP27AF `treeblob`: the b1 tree form's own tables are written only at a tree step,
    /// whose shape is the one narrow probe, so their slots need not hold the 2,048-row prefill's
    /// bytes). A constant is never narrowed. A narrowed key written by a wider step is refused at
    /// staging (`index ... is N bytes and its slot is M`), never truncated.
    ///
    /// # Errors
    /// As [`BlobLayout::plan`], and a `narrow` set with no probe to size it by.
    pub fn plan_narrowed(
        keys: &std::collections::BTreeSet<String>,
        probes: &[StepIndex<'_>],
        narrow: &std::collections::BTreeSet<String>,
        narrow_probes: &[StepIndex<'_>],
    ) -> Result<BlobLayout, IndexError> {
        Self::plan_aliased(keys, probes, narrow, narrow_probes, false)
    }

    /// [`BlobLayout::plan_narrowed`], and with `dedup` (CMP27AG `slotdedup`) every step-varying key
    /// of one [`alias_group`] -- (narrowed or not, key side, suffix) -- laid out at ONE slot (the
    /// group's widest capacity). Every staging writer builds a table from its key's suffix and the
    /// step's [`StepIndex`] for that side and nothing else (ws-p4's suffix law, pinned by
    /// `keys_sharing_a_suffix_are_the_same_table`), so the keys of a group hold the same bytes at
    /// every step: one slot says it once instead of once a layer. The constants already share
    /// slots by content; this does the same for the step and paged regions by law.
    ///
    /// # Errors
    /// As [`BlobLayout::plan_narrowed`].
    pub fn plan_aliased(
        keys: &std::collections::BTreeSet<String>,
        probes: &[StepIndex<'_>],
        narrow: &std::collections::BTreeSet<String>,
        narrow_probes: &[StepIndex<'_>],
        dedup: bool,
    ) -> Result<BlobLayout, IndexError> {
        if !narrow.is_empty() && narrow_probes.is_empty() {
            return Err(IndexError::Step(
                "narrowed index keys and no probe to size them".to_owned(),
            ));
        }
        let mut table: Vec<(String, u64, usize)> = Vec::new();
        let mut constants: Vec<(String, u64, Vec<u8>)> = Vec::new();
        let mut ids_bytes = 0usize;
        let mut off = 0u64;
        let mut dynamic_at = 0u64;
        let mut paged_at = 0u64;
        // THREE PASSES, and the order IS the law: the constants first, so the tables a step
        // rewrites are one range rather than sixty-four islands in a 6.5 MiB blob; the paged table
        // last, so the fifteen steps in sixteen that change no page send a range that stops before
        // it.
        for pass in [Region::Constant, Region::Step, Region::Paged] {
            match pass {
                Region::Constant => {}
                Region::Step => dynamic_at = off,
                Region::Paged => paged_at = off,
            }
            let pass_invariant = pass == Region::Constant;
            // CMP27AG: the aliased groups of this pass, `group -> (offset, capacity)`; the
            // capacities are taken over every member BEFORE any slot is placed.
            let mut groups: std::collections::BTreeMap<AliasGroup, (u64, usize)> =
                std::collections::BTreeMap::new();
            let mut group_cap: std::collections::BTreeMap<AliasGroup, usize> =
                std::collections::BTreeMap::new();
            if dedup && !pass_invariant {
                for key in keys {
                    if region_of(key) != pass || key.ends_with(".ids") {
                        continue;
                    }
                    let sized_by = if narrow.contains(key) {
                        narrow_probes
                    } else {
                        probes
                    };
                    let mut cap = 0usize;
                    for probe in sized_by {
                        cap = cap.max(index_table_for(key, probe, AttnQLen::Live)?.len());
                    }
                    let g = alias_group(key, narrow.contains(key));
                    let e = group_cap.entry(g).or_insert(0);
                    *e = (*e).max(cap);
                }
            }
            for key in keys {
                if region_of(key) != pass {
                    continue;
                }
                let mut cap = 0usize;
                let mut widest: Vec<u8> = Vec::new();
                let sized_by = if !pass_invariant && narrow.contains(key) {
                    narrow_probes
                } else {
                    probes
                };
                for probe in sized_by {
                    // A SIZING probe: `AttnQLen::Live` and `Exact` differ in a table's VALUES and
                    // never in its width, and the only tables kept from this pass are the
                    // step-invariant ones (`cos_sin`, the scales), which read neither.
                    let bytes = index_table_for(key, probe, AttnQLen::Live)?;
                    if bytes.len() > widest.len() {
                        widest = bytes;
                    }
                    cap = cap.max(widest.len());
                }
                // The ids table is the token plane's, not the blob's: at decode the argmax writes
                // it on the device and there is no host copy to lay out.
                if key.ends_with(".ids") {
                    ids_bytes = ids_bytes.max(cap);
                    continue;
                }
                if pass_invariant {
                    // Constants with IDENTICAL contents SHARE one slot. All sixteen attention
                    // layers rope with the same `rotary_dim` and the same `rope_theta`, so
                    // `layer.3.attn.qk_norm_rope.cos_sin` and its fifteen siblings are byte for byte
                    // the same table: laying out sixteen copies costs 64 MiB of blob at max_ctx
                    // 32768 to hold one table sixteen times. Several keys pointing at one address is
                    // exactly what the engine's index map is, and nothing writes these.
                    if let Some(shared) = constants.iter().find(|(_, _, b)| *b == widest) {
                        table.push((key.clone(), shared.1, cap));
                        continue;
                    }
                    constants.push((key.clone(), off, widest));
                }
                if dedup && !pass_invariant {
                    let g = alias_group(key, narrow.contains(key));
                    let gcap = group_cap.get(&g).copied().unwrap_or(cap).max(cap);
                    if let Some((at, c)) = groups.get(&g) {
                        table.push((key.clone(), *at, *c));
                        continue;
                    }
                    groups.insert(g, (off, gcap));
                    table.push((key.clone(), off, gcap));
                    off += (gcap as u64).div_ceil(SLOT_ALIGN) * SLOT_ALIGN;
                    continue;
                }
                table.push((key.clone(), off, cap));
                off += (cap as u64).div_ceil(SLOT_ALIGN) * SLOT_ALIGN;
            }
        }
        let len = usize::try_from(off)
            .map_err(|_| IndexError::Step("the index blob outgrew a usize".to_owned()))?;
        let dynamic_at = usize::try_from(dynamic_at)
            .map_err(|_| IndexError::Step("the index blob outgrew a usize".to_owned()))?;
        let paged_at = usize::try_from(paged_at)
            .map_err(|_| IndexError::Step("the index blob outgrew a usize".to_owned()))?;
        // The property every reader depends on, asserted rather than assumed: each region holds
        // exactly its own kind, so the three boundaries are the whole of the upload law.
        let bounds = |r: Region| -> (usize, usize) {
            match r {
                Region::Constant => (0, dynamic_at),
                Region::Step => (dynamic_at, paged_at),
                Region::Paged => (paged_at, len),
            }
        };
        for (key, offset, cap) in &table {
            let (lo, hi) = bounds(region_of(key));
            let at = *offset as usize;
            if at < lo || at + cap > hi {
                return Err(IndexError::Step(format!(
                    "`{key}` is laid out at {at}..{} and its region is {lo}..{hi}: the step's \
                     upload would carry the wrong bytes",
                    at + cap
                )));
            }
        }
        Ok(BlobLayout {
            table,
            constants,
            len,
            dynamic_at,
            paged_at,
            ids_bytes,
        })
    }

    /// Bytes a step may rewrite, upload and — at TP=2 — put on the wire: both varying regions.
    #[must_use]
    pub fn dynamic_len(&self) -> usize {
        self.len - self.dynamic_at
    }

    /// Bytes EVERY step rewrites, uploads and sends.
    #[must_use]
    pub fn step_len(&self) -> usize {
        self.paged_at - self.dynamic_at
    }

    /// Bytes a step sends only when a sequence's pages changed.
    #[must_use]
    pub fn paged_len(&self) -> usize {
        self.len - self.paged_at
    }

    /// Bytes staged once at open and never uploaded again.
    #[must_use]
    pub fn static_len(&self) -> usize {
        self.dynamic_at
    }
}

// =====================================================================================
// THE STEP FRAME: the bytes a step WROTE, not the region they live in
// =====================================================================================

/// Head of a step frame: the run count, the ids length and the region digest.
pub const STEP_FRAME_HEAD: usize = 16;

/// FNV-1a over a byte range. The frame's optional region digest, and nothing else uses it.
#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// **The bytes a step actually wrote, addressed by offset** — what ws-h29 left on this wire.
///
/// ws-h29 cut the step's peer frame from the whole blob to `[dynamic_at, paged_at)` and then said
/// what was still there: *"The 1,932 KiB a step sends is **not** the bytes a step writes. Every
/// slot is sized for the widest shape the engine can run — the 2,048-row prefill — so `pos` is
/// 3 × 2,048 × 8 B per attention layer (768 KiB), `sidx` 2,048 × 4 per GDN layer (768 KiB) and
/// `slot` 2,048 × 8 (256 KiB), while a batch-8 decode step writes **192, 32 and 64 bytes** of each.
/// A decode step therefore uploads and swaps about 1.9 MiB to say where eight sequences are."*
///
/// This is the per-shape compaction it named. [`super::device`]'s staging loop already knows
/// exactly which byte ranges it wrote — it writes `bytes.len()` at each key's own offset — so the
/// frame carries those RUNS and nothing else, and the follower applies them at the same offsets
/// into its own blob.
///
/// # Why applying runs leaves the two ranks' blobs identical
///
/// Both ranks build their blob from the SAME plan and stage the same constants at open, and the
/// leader's first frame of a link carries the whole `[dynamic_at, paged_at)` region
/// ([`StepFrame::pack_whole`]) rather than assuming anything about what page-locked memory holds.
/// After that the two regions are equal by induction: they were equal before the step, the leader
/// wrote exactly the runs in the frame, and the follower applied exactly those runs. The bytes
/// NEITHER rank writes — a slot's tail past the step's own rows — therefore stay equal too, which
/// matters because a table's padding is read: zero is the GDN kernels' null slot and `-1` is
/// `kv_append`'s early-out (`docs/EXEC.md` §2, and ws-l17's `cache_idx` binding walked off the end
/// of a table into exactly that padding).
///
/// # Why the ids ride in the same frame
///
/// They always have, and they are the other half of `payload_of`: the token plane's staging is
/// sized for the widest shape (2,048 rows = 8 KiB) and a decode step writes `rows × 4` bytes of
/// it. The frame carries the step's own ids and the applier ZERO-FILLS the rest, which is exactly
/// what `stage_ids` does on the leader — "a payload whose unused bytes are whatever was there last
/// is one nobody can compare between two runs".
pub struct StepFrame;

impl StepFrame {
    /// Bytes a frame of these runs and this many ids occupies.
    #[must_use]
    pub fn bytes(runs: &[(u32, u32)], ids_len: usize) -> usize {
        STEP_FRAME_HEAD
            + runs.len() * 8
            + runs.iter().map(|(_, n)| *n as usize).sum::<usize>()
            + ids_len
    }

    /// Pack the runs `host` was written at, plus `ids`, into `out` (which is CLEARED first).
    ///
    /// `out` is the caller's buffer rather than a fresh `Vec` because this is a per-step host path:
    /// the allocation it replaces was 1.9 MiB every step.
    ///
    /// # Errors
    /// A run that runs past `host`.
    pub fn pack(
        host: &[u8],
        runs: &[(u32, u32)],
        ids: &[u8],
        digest: Option<u64>,
        out: &mut Vec<u8>,
    ) -> Result<(), IndexError> {
        out.clear();
        out.reserve(Self::bytes(runs, ids.len()));
        out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        // **THE REGION DIGEST, AND WHY IT IS OPTIONAL.** The frame's whole claim is that applying
        // its runs leaves the follower's `[dynamic_at, paged_at)` equal to the leader's. That is an
        // induction (`StepFrame`), and an induction with no check is an assumption: this is the
        // check, and it costs an FNV over 1.9 MiB on each rank, which is half a millisecond of a
        // 24 ms step. So it is a GATE switch (`saber engine --frame-digest`) and `0` means "not
        // carried" rather than "matched".
        out.extend_from_slice(&digest.unwrap_or(0).to_le_bytes());
        for (off, len) in runs {
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }
        for (off, len) in runs {
            let (a, b) = (*off as usize, *off as usize + *len as usize);
            let slice = host.get(a..b).ok_or_else(|| {
                IndexError::Step(format!(
                    "a step frame run at {a}..{b} of a {}-byte blob",
                    host.len()
                ))
            })?;
            out.extend_from_slice(slice);
        }
        out.extend_from_slice(ids);
        Ok(())
    }

    /// Pack the WHOLE step-varying region as one run: the first frame of a link, which is what
    /// makes the induction above start from something rather than from an assumption.
    ///
    /// # Errors
    /// A region that runs past `host`.
    pub fn pack_whole(
        host: &[u8],
        dynamic_at: usize,
        paged_at: usize,
        ids: &[u8],
        digest: Option<u64>,
        out: &mut Vec<u8>,
    ) -> Result<(), IndexError> {
        if paged_at < dynamic_at || paged_at > host.len() {
            return Err(IndexError::Step(format!(
                "a step region of {dynamic_at}..{paged_at} in a {}-byte blob",
                host.len()
            )));
        }
        let run = [(
            u32::try_from(dynamic_at).unwrap_or(u32::MAX),
            u32::try_from(paged_at - dynamic_at).unwrap_or(u32::MAX),
        )];
        Self::pack(host, &run, ids, digest, out)
    }

    /// Apply a frame to this rank's own blob and ids staging.
    ///
    /// Every run is bounds-checked against `[dynamic_at, paged_at)` and REFUSED by name outside it:
    /// a frame is the peer's bytes, and a peer that named an offset in the constants (staged once
    /// at open) or in the block table (its own frame) would be rewriting a region this rank never
    /// re-uploads.
    ///
    /// # Errors
    /// A truncated frame, a run outside the step-varying region, or more ids than the staging holds.
    pub fn apply(
        frame: &[u8],
        host: &mut [u8],
        dynamic_at: usize,
        paged_at: usize,
        ids: &mut [u8],
    ) -> Result<(), IndexError> {
        if frame.len() < STEP_FRAME_HEAD {
            return Err(IndexError::Step(format!(
                "a step frame of {} byte(s); its head is {STEP_FRAME_HEAD}",
                frame.len()
            )));
        }
        let n = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let ids_len = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
        let digest = u64::from_le_bytes([
            frame[8], frame[9], frame[10], frame[11], frame[12], frame[13], frame[14], frame[15],
        ]);
        let table_end = STEP_FRAME_HEAD + n * 8;
        if frame.len() < table_end {
            return Err(IndexError::Step(format!(
                "a step frame of {} byte(s) declaring {n} run(s)",
                frame.len()
            )));
        }
        let mut runs: Vec<(usize, usize)> = Vec::with_capacity(n);
        let mut total = 0usize;
        for i in 0..n {
            let at = STEP_FRAME_HEAD + i * 8;
            let off = u32::from_le_bytes([frame[at], frame[at + 1], frame[at + 2], frame[at + 3]])
                as usize;
            let len =
                u32::from_le_bytes([frame[at + 4], frame[at + 5], frame[at + 6], frame[at + 7]])
                    as usize;
            let end = off.checked_add(len).ok_or_else(|| {
                IndexError::Step(format!(
                    "a step frame run at {off} of {len} byte(s) overflows"
                ))
            })?;
            if off < dynamic_at || end > paged_at || end > host.len() {
                return Err(IndexError::Step(format!(
                    "a step frame run at {off}..{end} outside this rank's step-varying region                      {dynamic_at}..{paged_at}"
                )));
            }
            runs.push((off, len));
            total += len;
        }
        if frame.len() != table_end + total + ids_len {
            return Err(IndexError::Step(format!(
                "a step frame of {} byte(s) carrying {n} run(s) of {total} byte(s) and {ids_len}                  id byte(s)",
                frame.len()
            )));
        }
        if ids_len > ids.len() {
            return Err(IndexError::Step(format!(
                "a step frame carrying {ids_len} id byte(s); this rank's staging is {}",
                ids.len()
            )));
        }
        let mut at = table_end;
        for (off, len) in runs {
            host[off..off + len].copy_from_slice(&frame[at..at + len]);
            at += len;
        }
        ids[..ids_len].copy_from_slice(&frame[at..at + ids_len]);
        // THE CHECK, when the leader carried one: this rank's region against the leader's, AFTER
        // the runs land. A mismatch is the induction broken and it is named at the step it breaks.
        if digest != 0 {
            let mine = fnv1a(&host[dynamic_at..paged_at]);
            if mine != digest {
                return Err(IndexError::Step(format!(
                    "the step frame's runs left this rank's step-varying region at {mine:#018x} \
                     and the leader's is {digest:#018x}: the two ranks' index blobs have parted"
                )));
            }
        }
        // THE TAIL IS ZEROED, exactly as `stage_ids` zeroes the leader's: the two staging buffers
        // have to be byte-identical, and a tail that is whatever the last step left is a buffer two
        // runs cannot be compared through.
        ids[ids_len..].fill(0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live<'a>(slot: u32, done: u32, blocks: &'a [u32], rows: u32) -> SeqFacts<'a> {
        SeqFacts {
            state_slot: slot,
            done_before: done,
            kv_blocks: blocks,
            live_rows: rows,
            rope: None,
        }
    }

    #[test]
    fn scalar_tail_reads_the_previous_wide_snapshot_and_writes_one_live_row() {
        let pages: Vec<u32> = (0..16).collect();
        let seqs = [live(17, 255, &pages, 1)];
        for count in [2u32, 5, 8, 10, 16] {
            let accepted = [count];
            let index = StepIndex {
                phase: Phase::Decode,
                rows: 1,
                max_blocks: 16,
                seqs: &seqs,
                ids: &[1234],
                committed_before: &accepted,
                band_width: None,
            };
            assert!(
                index.check().is_err(),
                "the narrow sidx cannot hold the previous read"
            );
            let tail = index
                .with_scalar_snapshot_span(16)
                .expect("allocated state span");
            tail.check()
                .expect("scalar step at the last target position");
            let table = |key: &str| index_table_for(key, &tail, AttnQLen::Live).unwrap();
            let sidx = table("l0.gdn.sidx");
            let words: Vec<i32> = sidx
                .chunks_exact(4)
                .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
                .collect();
            assert_eq!(words[0], 17, "current scalar snapshot destination");
            assert_eq!(
                words[count as usize - 1],
                17 + count as i32 - 1,
                "previous accepted snapshot"
            );
            assert_eq!(table("l0.gdn.naccept"), i32_bytes(&[count as i32]));
            assert_eq!(table("l0.gdn.cache_idx"), i32_bytes(&[17]));
            assert_eq!(table("l0.gdn.cu"), i32_bytes(&[0, 1]));
            assert_eq!(table("l3.attn.seqused_k"), i32_bytes(&[256]));
            let ordinary = StepIndex {
                committed_before: &[1],
                ..index
            };
            for key in [
                "embed_tokens.ids",
                "l3.attn.pos",
                "l3.attn.slot",
                "l3.attn.block_table",
                "l3.attn.seqused_k",
                "l0.gdn.cu",
            ] {
                assert_eq!(
                    table(key),
                    index_table_for(key, &ordinary, AttnQLen::Live).unwrap(),
                    "{key}"
                );
            }
            assert!(index.with_scalar_snapshot_span(count - 1).is_err());
        }
        // Multiple slots cannot borrow this layout: their encoded stride is no longer irrelevant.
        let slots = [live(1, 255, &pages, 1), live(17, 255, &pages, 1)];
        let multi = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 16,
            seqs: &slots,
            ids: &[1, 2],
            committed_before: &[5, 1],
            band_width: None,
        };
        assert!(multi
            .with_scalar_snapshot_span(16)
            .unwrap()
            .check()
            .is_err());
    }

    #[test]
    fn the_tables_are_the_ones_a_fresh_prefill_needs() {
        let ids = [11u32, 12, 13, 14];
        let seqs = [live(1, 0, &[0, 1], 4)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 4,
            max_blocks: 4,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("a legal prefill step");
        let t = |k: &str| index_table(k, &s).expect(k);
        assert_eq!(t("embed_tokens.ids"), i32_bytes(&[11, 12, 13, 14]));
        assert_eq!(t("layer.0.gdn.conv_prep.cu"), i32_bytes(&[0, 4]));
        assert_eq!(t("layer.0.gdn.conv_prep.cache_idx"), i32_bytes(&[1]));
        assert_eq!(t("layer.0.gdn.conv_prep.has_init"), vec![0u8]);
        assert_eq!(t("layer.3.attn.prefill.seqused_k"), i32_bytes(&[4]));
        // Padded to the plans' width, with this session's own last page.
        assert_eq!(
            t("layer.3.attn.prefill.block_table"),
            i32_bytes(&[0, 1, 1, 1])
        );
    }

    /// The batched decode step: every table is four sequences wide and each one is that sequence's
    /// own fact. This is the whole of "batch > 1" as the kernels see it.
    #[test]
    fn a_batch_of_four_gives_every_table_four_entries() {
        let ids = [101u32, 202, 303, 404];
        let (b0, b1, b2, b3) = ([0u32, 1], [2u32], [3u32, 4, 5], [6u32]);
        let seqs = [
            live(1, 31, &b0, 1),
            live(2, 15, &b1, 1),
            live(3, 47, &b2, 1),
            live(4, 8, &b3, 1),
        ];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 4,
            max_blocks: 3,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("a legal decode step");
        let t = |k: &str| index_table(k, &s).expect(k);
        assert_eq!(t("embed_tokens.ids"), i32_bytes(&[101, 202, 303, 404]));
        // One row per sequence: the prefix is 0,1,2,3,4.
        assert_eq!(t("l0.gdn.fused.cu"), i32_bytes(&[0, 1, 2, 3, 4]));
        assert_eq!(t("l0.gdn.fused.cache_idx"), i32_bytes(&[1, 2, 3, 4]));
        assert_eq!(t("l0.gdn.fused.sidx"), i32_bytes(&[1, 2, 3, 4]));
        assert_eq!(t("l0.gdn.fused.naccept"), i32_bytes(&[1, 1, 1, 1]));
        // Context after this step, per sequence.
        assert_eq!(
            t("layer.7.attn.decode.seqused_k"),
            i32_bytes(&[32, 16, 48, 9])
        );
        // [n_seqs][max_blocks], each row padded with its own last page.
        assert_eq!(
            t("layer.7.attn.decode.block_table"),
            i32_bytes(&[0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 6])
        );
    }

    /// Padding is not "output nobody reads": it is rows the kernels structurally skip.
    #[test]
    fn padding_slots_span_no_tokens_and_no_state() {
        let ids = [101u32, 202, 0, 0];
        let b0 = [0u32, 1];
        let b1 = [2u32];
        let seqs = [
            live(1, 31, &b0, 1),
            live(2, 15, &b1, 1),
            SeqFacts::padding(),
            SeqFacts::padding(),
        ];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 4,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("two live of four slots");
        let t = |k: &str| index_table(k, &s).expect(k);
        // The GDN kernels see slen = 0 for the padding slots...
        assert_eq!(t("l0.gdn.fused.cu"), i32_bytes(&[0, 1, 2, 2, 2]));
        // ...they are on the null state slot, so conv and recurrent both return early...
        assert_eq!(t("l0.gdn.fused.cache_idx"), i32_bytes(&[1, 2, 0, 0]));
        // ...and the attention returns on `ctx <= 0` before it reads a page.
        assert_eq!(
            t("layer.7.attn.decode.seqused_k"),
            i32_bytes(&[32, 16, 0, 0])
        );
        // `naccept` stays 1 even for padding: it indexes `sidx`, and 0 would read before the table.
        assert_eq!(t("l0.gdn.fused.naccept"), i32_bytes(&[1, 1, 1, 1]));
    }

    /// A padded prefill chunk: `cu` counts the LIVE rows, the embedding gathers one id per PLANNED
    /// row, and `seqused_k` is `done_before` plus whatever the step's attention carries as its
    /// `q_len` — the live rows once the plan is retargeted (ws-h27), the bucket's if it is not
    /// (the pre-h27 pair, which is the control `--no-live-rows` runs). `saber_attn_prefill`'s mask
    /// is right-aligned on that kernarg, so the two are one number and any other key count masks
    /// every live row short by the difference (ws-l15; the law is in `saber_exec::forward`).
    #[test]
    fn a_padded_prefill_chunk_counts_its_live_tokens_and_not_its_padding() {
        let ids = [7u32, 8, 0, 0];
        let b = [2u32];
        let seqs = [live(1, 0, &b, 2)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 4,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("2 live rows in a 4-row bucket");
        assert_eq!(
            index_table("l0.gdn.cu", &s).expect("cu"),
            i32_bytes(&[0, 2])
        );
        // THE LIVE ROWS, because the plan's attention is pointed at them (ws-h27).
        assert_eq!(
            index_table("layer.0.attn.prefill.seqused_k", &s).expect("seqused_k"),
            i32_bytes(&[2])
        );
        assert_eq!(
            index_table_for("layer.0.attn.prefill.seqused_k", &s, AttnQLen::Exact(2))
                .expect("seqused_k"),
            i32_bytes(&[2])
        );
        // THE CONTROL: a plan that was never retargeted still says the bucket's rows, and the key
        // count follows it — the pre-ws-h27 pair, which is what `--no-live-rows` runs.
        assert_eq!(
            index_table_for("layer.0.attn.prefill.seqused_k", &s, AttnQLen::Exact(4))
                .expect("seqused_k"),
            i32_bytes(&[4])
        );
        // ...but the embedding still gathers one id per planned row.
        assert_eq!(
            index_table("embed_tokens.ids", &s).expect("ids"),
            i32_bytes(&[7, 8, 0, 0])
        );
    }

    #[test]
    fn a_later_chunk_continues_the_state_and_counts_every_resident_key() {
        let ids = [21u32, 22];
        let b = [0u32, 1, 2];
        let seqs = [live(1, 8192, &b, 2)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 2,
            max_blocks: 3,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("a continued session");
        assert_eq!(
            index_table("l0.gdn.conv_prep.has_init", &s).expect("has_init"),
            vec![1u8]
        );
        assert_eq!(
            index_table("layer.3.attn.prefill.seqused_k", &s).expect("seqused_k"),
            i32_bytes(&[8194])
        );
    }

    /// NEGATIVE CONTROL: a key with no law is a refusal. A zero-filled buffer would be an offset
    /// table full of zeros, which a kernel reads as "row 0" / "slot 0" and no test would see.
    #[test]
    fn an_index_with_no_law_is_refused() {
        let ids = [1u32];
        let b = [0u32];
        let seqs = [live(1, 0, &b, 1)];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 1,
            max_blocks: 1,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        let err = index_table("layer.0.gdn.decode.rolling_window", &s)
            .expect_err("there is no law for this one");
        assert_eq!(
            err,
            IndexError::Unknown("layer.0.gdn.decode.rolling_window".into())
        );
        assert!(err.to_string().contains("zero-filled"), "{err}");
    }

    /// NEGATIVE CONTROLS for the slot rule, in both directions. A live sequence on slot 0 launches
    /// everything and updates no state; a padding slot on a real one updates a live session's.
    #[test]
    fn the_null_slot_rule_is_enforced_both_ways() {
        let ids = [1u32, 2];
        let b = [0u32];
        let bad_live = [live(0, 0, &b, 1), SeqFacts::padding()];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 1,
            seqs: &bad_live,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        let err = s.check().expect_err("a live sequence on the null slot");
        assert!(err.to_string().contains("null slot"), "{err}");

        let bad_pad = [
            live(1, 0, &b, 1),
            SeqFacts {
                state_slot: 2,
                done_before: 0,
                kv_blocks: &[],
                live_rows: 0,
                rope: None,
            },
        ];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 1,
            seqs: &bad_pad,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        let err = s.check().expect_err("padding on a real slot");
        assert!(err.to_string().contains("padding must use slot 0"), "{err}");
    }

    #[test]
    fn a_step_whose_slots_contradict_it_is_refused() {
        let b = [0u32];
        let ids2 = [1u32, 2];
        // ids that are not the rows
        let seqs = [live(1, 0, &b, 1)];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 4,
            max_blocks: 1,
            seqs: &seqs,
            ids: &ids2,
            committed_before: &[],
            band_width: None,
        };
        assert!(matches!(s.check(), Err(IndexError::Step(_))));

        // a live slot after a padding one
        let seqs = [SeqFacts::padding(), live(1, 0, &b, 1)];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 1,
            seqs: &seqs,
            ids: &ids2,
            committed_before: &[],
            band_width: None,
        };
        let err = s.check().expect_err("live after padding");
        assert!(err.to_string().contains("live slots come first"), "{err}");

        // SEVERAL prompts in one prefill step are legal since ws-n7 (`Bucket::prefill_batch`),
        // and what the shared law refuses is a SHORT chunk beside another live one: slot `i` owns
        // rows `[i*q, (i+1)*q)` while `cu` is the live-row prefix, so a short first chunk would
        // place the second sequence a chunk's padding early. Two FULL chunks are fine.
        let seqs = [live(1, 0, &b, 1), live(2, 0, &b, 1)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 2,
            max_blocks: 1,
            seqs: &seqs,
            ids: &ids2,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("two full chunks in one prefill step");

        // The negative control, which is the half that still refuses: four rows over two slots is
        // a two-row chunk each, and a slot feeding one of its two rows beside another live
        // sequence is the shape `cu` cannot describe.
        let ids4 = [1u32, 2, 3, 4];
        let seqs = [live(1, 0, &b, 1), live(2, 0, &b, 2)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 4,
            max_blocks: 1,
            seqs: &seqs,
            ids: &ids4,
            committed_before: &[],
            band_width: None,
        };
        let err = s
            .check()
            .expect_err("a short chunk beside another live one");
        assert!(
            err.to_string()
                .contains("padded only when it is the step's ONLY one"),
            "{err}"
        );
    }

    /// A verify step: `w` rows per sequence, `w` consecutive snapshots per sequence, and `naccept`
    /// pointing at the snapshot the LAST COMMITTED token of the previous step wrote.
    #[test]
    fn a_verify_step_starts_from_the_snapshot_its_last_committed_token_wrote() {
        let blocks = [0u32, 1];
        // Two sequences at verify width 5, on slot sets 1..6 and 6..11.
        let seqs = [live(1, 40, &blocks, 5), live(6, 17, &blocks, 5)];
        let ids: Vec<TokenId> = (0..10).collect();
        // Last step: sequence 0 committed 3 tokens (2 draft accepted + 1), sequence 1 committed 1
        // (everything rejected, the recovery token only).
        let committed = [3u32, 1];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 10,
            max_blocks: 4,
            seqs: &seqs,
            ids: &ids,
            committed_before: &committed,
            band_width: None,
        };
        s.check().expect("a legal verify step");
        assert_eq!(s.q_len(), 5);
        let t = |k: &str| index_table(k, &s).expect(k);
        // `sidx` is per ROW: five consecutive snapshots from each sequence's own slot.
        assert_eq!(
            t("l0.gdn.recurrent_update.sidx"),
            i32_bytes(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        );
        // And `naccept` picks index 2 of the first sequence's five and index 0 of the second's —
        // `sidx[naccept - 1]`, which is slot 3 and slot 6.
        assert_eq!(t("l0.gdn.recurrent_update.naccept"), i32_bytes(&[3, 1]));
        // Five rows each, so the row prefix runs 0, 5, 10.
        assert_eq!(t("l0.gdn.conv_prep.cu"), i32_bytes(&[0, 5, 10]));
        // One conv slot per SEQUENCE, not per row.
        assert_eq!(t("l0.gdn.conv_prep.cache_idx"), i32_bytes(&[1, 6]));
        // Keys resident after the step: everything the sequence had plus its own rows.
        assert_eq!(t("l3.attn.decode.seqused_k"), i32_bytes(&[45, 22]));
    }

    /// An empty table is the SPEC=0 answer: one token committed per slot, so `naccept` is all ones
    /// and every sequence continues from its own single snapshot.
    #[test]
    fn no_committed_table_means_one_token_each() {
        let blocks = [0u32];
        let seqs = [live(1, 5, &blocks, 1), live(2, 9, &blocks, 1)];
        let ids = [7u32, 8];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        s.check().expect("a legal SPEC=0 step");
        assert_eq!(
            index_table("x.naccept", &s).expect("naccept"),
            i32_bytes(&[1, 1])
        );
    }

    /// Padding reads 1 whatever it is told, because a padding slot is skipped by its NULL STATE
    /// SLOT and not by `naccept` — and a 0 there would read one `i32` before the table for every
    /// sequence in the batch, padding or not.
    #[test]
    fn padding_reads_one_and_is_skipped_by_its_null_slot() {
        let blocks = [0u32];
        let seqs = [live(1, 5, &blocks, 2), SeqFacts::padding()];
        let ids = [7u32, 8, 0, 0];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 4,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[2, 1],
            band_width: None,
        };
        s.check().expect("a legal padded verify step");
        assert_eq!(
            index_table("x.naccept", &s).expect("naccept"),
            i32_bytes(&[2, 1])
        );
        // The padding rows' snapshots are the null slot, which is what makes them harmless.
        assert_eq!(
            index_table("x.sidx", &s).expect("sidx"),
            i32_bytes(&[1, 2, 0, 0])
        );
    }

    /// NEGATIVE CONTROL: a slot that committed nothing.
    ///
    /// Every verified step commits the accepted prefix PLUS one — the recovery token at the
    /// rejected position, or the bonus token when every draft survived — so the count is 1 or more
    /// by construction. A 0 reaching the table makes both decode kernels evaluate
    /// `sidx[naccept - 1]` at index −1, which is a read before the table and not a crash.
    #[test]
    fn a_slot_that_committed_nothing_is_refused() {
        let blocks = [0u32];
        let seqs = [live(1, 5, &blocks, 5)];
        let ids: Vec<TokenId> = (0..5).collect();
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 5,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[0],
            band_width: None,
        };
        let e = s.check().expect_err("a zero naccept");
        // The shared law's words since ws-h18, and the right ones: the harm is the INDEX, not the
        // number. `sidx[naccept - 1]` at 0 reads the `i32` before this sequence's block.
        assert!(format!("{e}").contains("reads before the block"), "{e}");
        // And the TABLE does not refuse — it clamps to 1 (`Accepted::for_seq` is `.max(1)`), which
        // is why `check` is not optional and why the engine calls it on every step before a byte is
        // staged. A caller that skipped the check would get a legal-looking table pointing at the
        // wrong snapshot, which is the original bug wearing a different hat.
        assert_eq!(
            index_table("x.naccept", &s).expect("the law clamps rather than refusing"),
            i32_bytes(&[1])
        );
    }

    /// NEGATIVE CONTROL: a count past the verify width would index past this sequence's snapshots
    /// and into the NEXT sequence's — which is another live session's recurrent state, not padding.
    #[test]
    fn a_count_past_the_width_is_refused() {
        let blocks = [0u32];
        let seqs = [live(1, 5, &blocks, 3), live(4, 5, &blocks, 3)];
        let ids: Vec<TokenId> = (0..6).collect();
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 6,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[4, 1],
            band_width: None,
        };
        let e = s.check().expect_err("naccept past the width");
        assert!(format!("{e}").contains("next sequence's slots"), "{e}");
    }

    /// NEGATIVE CONTROL: a table that is not one entry per slot. Silently short would make the
    /// remaining slots read 1 and continue from the wrong candidate.
    #[test]
    fn a_short_committed_table_is_refused() {
        let blocks = [0u32];
        let seqs = [live(1, 5, &blocks, 2), live(3, 5, &blocks, 2)];
        let ids: Vec<TokenId> = (0..4).collect();
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 4,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[1],
            band_width: None,
        };
        let e = s.check().expect_err("a short naccept table");
        assert!(format!("{e}").contains("one per slot"), "{e}");
    }

    /// Two keys that end the same way are two names for the same bytes.
    ///
    /// This is the property `device::stage_indices` deduplicates on, and it is a property of the
    /// shared law rather than of this crate — `index_table_with` dispatches on the last
    /// dot-component and its branches read only the `StepIndex`. Pinned here because the saving is
    /// large (one 512 KiB rope table per step instead of sixteen, 32.6 ms of host time) and the
    /// cost of the property quietly becoming false is silent wrong indices.
    #[test]
    fn keys_sharing_a_suffix_are_the_same_table() {
        let seqs = [
            SeqFacts {
                state_slot: 1,
                done_before: 40,
                live_rows: 1,
                kv_blocks: &[0, 1, 2],
                rope: None,
            },
            SeqFacts {
                state_slot: 2,
                done_before: 7,
                live_rows: 1,
                kv_blocks: &[3],
                rope: None,
            },
        ];
        let ids = [11u32, 12];
        let step = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 4,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[1, 1],
            band_width: None,
        };
        // Every suffix the law knows, under two different layer prefixes.
        for suffix in [
            "cu",
            "cache_idx",
            "sidx",
            "naccept",
            "has_init",
            "block_table",
            "seqused_k",
            "cos_sin",
            "pos",
            "slot",
        ] {
            let a = index_table(&format!("layer.0.attn.{suffix}"), &step)
                .unwrap_or_else(|e| panic!("layer.0 {suffix}: {e}"));
            let b = index_table(&format!("layer.57.gdn.{suffix}"), &step)
                .unwrap_or_else(|e| panic!("layer.57 {suffix}: {e}"));
            assert_eq!(a, b, "`{suffix}` differs between two layers' keys");
            assert!(!a.is_empty(), "`{suffix}` is empty");
        }
        // The negative control: two DIFFERENT suffixes must not be deduplicated together, or the
        // saving above would be writing one table into another's slot.
        let cu = index_table("layer.0.attn.cu", &step).expect("cu");
        let pos = index_table("layer.0.attn.pos", &step).expect("pos");
        assert_ne!(cu, pos, "two different suffixes gave the same bytes");
    }

    // =================================================================================
    // The blob layout, counted with no card (ws-h29)
    // =================================================================================

    /// The engine's own key set: every index a body plan of either phase binds.
    ///
    /// Built from the real graph and the real planner, because a hand-written key list would be a
    /// second statement of which ops read which tables, and the whole point of this count is that it
    /// is the engine's.
    fn engine_keys() -> std::collections::BTreeSet<String> {
        use saber_exec::layer::{BlockPlan, PlanOptions, RuntimeParams};
        use saber_kernels::rules::LaunchRules;
        use saber_kernels::saber::SaberRules;
        use saber_model::graph::{Bucket, Graph, ModelConfig};

        let graph = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 1).expect("graph");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let params = RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 9,
        };
        let mut keys = std::collections::BTreeSet::new();
        // The shapes `run_engine.sh` boots with: two prefill buckets and the decode batches.
        let buckets = [
            Bucket::Prefill {
                tokens: 256,
                seqs: 1,
            },
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            Bucket::Decode {
                batch: 1,
                verify_width: 1,
            },
            Bucket::Decode {
                batch: 8,
                verify_width: 1,
            },
        ];
        for bucket in buckets {
            let plan = BlockPlan::with_options(
                &graph,
                bucket,
                &layers,
                params,
                &mx,
                &r4,
                PlanOptions::BODY,
            )
            .expect("plan");
            keys.extend(plan.index_keys());
        }
        keys
    }

    /// The layout the engine builds at a given `max_ctx`, from the engine's own sizing probes.
    fn layout_at(max_ctx: u32, max_batch: u32) -> BlobLayout {
        layout_at_with(max_ctx, max_batch, false)
    }

    /// [`layout_at`], with CMP27AG's aliased slots when `dedup`.
    fn layout_at_with(max_ctx: u32, max_batch: u32, dedup: bool) -> BlobLayout {
        let keys = engine_keys();
        let max_blocks = max_ctx.div_ceil(16);
        let all_pages: Vec<u32> = (0..max_blocks).collect();
        let biggest_prefill = 2048u32;
        let ids = vec![0u32; biggest_prefill.max(max_batch) as usize];
        let prefill_seqs = [SeqFacts {
            state_slot: 1,
            done_before: 0,
            kv_blocks: &all_pages,
            live_rows: biggest_prefill,
            rope: None,
        }];
        let decode_seqs: Vec<SeqFacts<'_>> = (0..max_batch)
            .map(|i| SeqFacts {
                state_slot: i + 1,
                done_before: 0,
                kv_blocks: &all_pages,
                live_rows: 1,
                rope: None,
            })
            .collect();
        let probes = [
            StepIndex {
                phase: Phase::Prefill,
                rows: biggest_prefill,
                max_blocks,
                seqs: &prefill_seqs,
                ids: &ids[..biggest_prefill as usize],
                committed_before: &[],
                band_width: None,
            },
            StepIndex {
                phase: Phase::Decode,
                rows: max_batch,
                max_blocks,
                seqs: &decode_seqs,
                ids: &ids[..max_batch as usize],
                committed_before: &[],
                band_width: None,
            },
        ];
        for probe in &probes {
            probe.check().expect("the probe step");
        }
        BlobLayout::plan_aliased(
            &keys,
            &probes,
            &std::collections::BTreeSet::new(),
            &[],
            dedup,
        )
        .expect("the layout")
    }

    /// **CMP27AG `slotdedup`**: the aliased layout lays every (side, suffix) group of the step and
    /// paged regions at ONE slot, each slot holds its widest member, the constants are untouched,
    /// the regions keep their order and bounds -- and the negative controls: two suffixes, two
    /// sides (a chunk's and the decode half's `block_table`) and a narrowed key never share.
    #[test]
    fn cmp27ag_the_aliased_layout_shares_a_slot_per_side_and_suffix() {
        for (max_ctx, max_batch) in [(4096u32, 8u32), (32768, 8)] {
            let plain = layout_at_with(max_ctx, max_batch, false);
            let alias = layout_at_with(max_ctx, max_batch, true);
            assert_eq!(
                plain.static_len(),
                alias.static_len(),
                "the constants moved"
            );
            assert_eq!(plain.table.len(), alias.table.len(), "a key was dropped");
            assert_eq!(plain.ids_bytes, alias.ids_bytes);
            let cap_of = |l: &BlobLayout, k: &str| -> (u64, usize) {
                l.table
                    .iter()
                    .find(|(n, _, _)| n == k)
                    .map(|(_, o, c)| (*o, *c))
                    .expect("key")
            };
            let mut groups: std::collections::BTreeMap<AliasGroup, u64> =
                std::collections::BTreeMap::new();
            for (key, _, cap) in &plain.table {
                let (at, acap) = cap_of(&alias, key);
                if step_invariant(key) {
                    assert_eq!(cap_of(&plain, key).0, at, "a constant moved: `{key}`");
                    continue;
                }
                assert!(
                    acap >= *cap,
                    "`{key}`'s shared slot is narrower than its own"
                );
                let g = alias_group(key, false);
                match groups.get(&g) {
                    Some(o) => assert_eq!(*o, at, "`{key}` left its group's slot"),
                    None => {
                        assert!(
                            !groups.values().any(|o| *o == at),
                            "two groups share a slot at {at} (`{key}`)"
                        );
                        groups.insert(g, at);
                    }
                }
            }
            assert!(
                alias.step_len() * 4 < plain.step_len(),
                "the step region did not shrink: {} -> {} B",
                plain.step_len(),
                alias.step_len()
            );
            assert!(alias.paged_len() < plain.paged_len());
            println!(
                "max_ctx {max_ctx}: step {} -> {} KiB, paged {} -> {} KiB, {} group(s)",
                plain.step_len() >> 10,
                alias.step_len() >> 10,
                plain.paged_len() >> 10,
                alias.paged_len() >> 10,
                groups.len()
            );
        }
        // Negative controls on the grouping itself.
        assert_ne!(
            alias_group("layer.3.attn.decode.block_table", false),
            alias_group("layer.3.attn.prefill.block_table", false),
            "the decode half's and the chunk's block tables share a slot"
        );
        assert_ne!(
            alias_group("layer.2.gdn.conv_prep.cu", false),
            alias_group("layer.2.gdn.conv_prep.sidx", false)
        );
        assert_ne!(
            alias_group("layer.3.attn.tree.anc", true),
            alias_group("layer.7.attn.tree.anc", false),
            "a narrowed key shares an unnarrowed slot"
        );
        assert_ne!(
            alias_group(&saber_exec::mixed::packed_key(1, "cu"), false),
            alias_group(&saber_exec::mixed::packed_key(2, "cu"), false),
            "two packed chunks share a slot"
        );
        assert_eq!(
            alias_group("layer.3.attn.decode.block_table", false),
            alias_group("layer.11.attn.decode.block_table", false)
        );
    }

    /// Bytes per suffix family in a region, for the ledger's table.
    fn by_suffix(l: &BlobLayout, region: Region) -> std::collections::BTreeMap<String, usize> {
        let mut out: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        let mut seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for (key, off, cap) in &l.table {
            if region_of(key) != region {
                continue;
            }
            // A shared slot (the sixteen `cos_sin` keys) is counted once, because it is laid out
            // once.
            if !seen.insert(*off) {
                continue;
            }
            let suffix = key.rsplit('.').next().unwrap_or(key).to_owned();
            *out.entry(suffix).or_default() += cap.div_ceil(256) * 256;
        }
        out
    }

    /// **What `max_ctx` costs a step, in bytes, and where.**
    ///
    /// ws-h28 measured `--max-ctx 32768` at +6.6 ms/step over `--max-ctx 4096` at ctx 234..634,
    /// with an IDENTICAL dispatch sequence and every counter zero — so a per-step HOST cost. This
    /// test is the first half of the attribution and it needs no card: the whole blob was uploaded
    /// every step (and, at TP=2, swapped with the peer), and most of it is a rope table that cannot
    /// change while the engine is open.
    #[test]
    fn the_step_varying_bytes_are_the_blob_minus_the_rope_table() {
        for max_batch in [2u32, 8] {
            let small = layout_at(4096, max_batch);
            let big = layout_at(32768, max_batch);
            println!("max_seqs {max_batch}");
            for (name, l) in [("max_ctx 4096", &small), ("max_ctx 32768", &big)] {
                println!(
                    "  {name}: blob {} KiB = once {} KiB + every step {} KiB + on a page boundary \
                     {} KiB   (once: {:?}, every step: {:?}, paged: {:?})",
                    l.len >> 10,
                    l.static_len() >> 10,
                    l.step_len() >> 10,
                    l.paged_len() >> 10,
                    by_suffix(l, Region::Constant)
                        .iter()
                        .map(|(k, v)| format!("{k} {} KiB", v >> 10))
                        .collect::<Vec<_>>(),
                    by_suffix(l, Region::Step)
                        .iter()
                        .map(|(k, v)| format!("{k} {} KiB", v >> 10))
                        .collect::<Vec<_>>(),
                    by_suffix(l, Region::Paged)
                        .iter()
                        .map(|(k, v)| format!("{k} {} KiB", v >> 10))
                        .collect::<Vec<_>>(),
                );
            }
            // **The number the gate is about**: what a step that changed no page sends is the same
            // at both bands. Every table that is a function of `max_ctx` is now either staged once
            // (`cos_sin`) or in the paged region (`block_table`).
            assert_eq!(
                small.step_len(),
                big.step_len(),
                "a step's own bytes still depend on max_ctx"
            );
            // The rope table is the whole of the growth in the STATIC region: 128 B a position,
            // one shared slot, `max_ctx` positions.
            assert_eq!(
                big.static_len() - small.static_len(),
                (32768 - 4096) * 128,
                "the constants' growth is not the rope table's"
            );
            // And the step-varying region grows by the BLOCK TABLE alone: `max_ctx / 16` entries
            // per slot per attention leg. Nothing else in a step is a function of the band.
            let bt = |l: &BlobLayout| -> usize {
                by_suffix(l, Region::Paged)
                    .get("block_table")
                    .copied()
                    .unwrap_or(0)
            };
            assert_eq!(
                big.dynamic_len() - small.dynamic_len(),
                bt(&big) - bt(&small),
                "something other than `block_table` in the step's own bytes is a function of \
                 max_ctx"
            );
            assert_eq!(bt(&big), big.paged_len(), "the paged region is block_table");
            // The claim the fix rests on: at max_ctx 32768 most of the blob is a constant.
            assert!(
                big.static_len() > big.dynamic_len(),
                "the constants are {} KiB of a {} KiB blob: uploading the whole blob would not be \
                 the cost ws-h28 measured",
                big.static_len() >> 10,
                big.len >> 10
            );
        }
    }

    /// The constants are laid out BELOW the step-varying region, and both regions are contiguous.
    ///
    /// This is the property `flush_indices`, `payload_of` and `apply_payload` all depend on: one
    /// range per step, not sixty-four islands. `BlobLayout::plan` refuses a layout that breaks it,
    /// and the control is that refusal firing on a table built the other way round.
    #[test]
    fn the_two_regions_are_contiguous_and_the_constants_come_first() {
        let l = layout_at(32768, 8);
        assert!(l.dynamic_at > 0, "no constants at all");
        assert!(l.step_len() > 0, "no step-varying tables at all");
        assert!(l.paged_len() > 0, "no paged tables at all");
        for (key, off, cap) in &l.table {
            let at = *off as usize;
            let (lo, hi) = match region_of(key) {
                Region::Constant => (0, l.dynamic_at),
                Region::Step => (l.dynamic_at, l.paged_at),
                Region::Paged => (l.paged_at, l.len),
            };
            assert!(
                at >= lo && at + cap <= hi,
                "`{key}` is at {at}..{} and its region is {lo}..{hi}",
                at + cap
            );
        }
        // Every constant slot is staged, so a step's upload covering `[dynamic_at, len)` carries
        // every table that changes and none that does not.
        for (_, off, bytes) in &l.constants {
            assert!(
                (*off as usize) + bytes.len() <= l.dynamic_at,
                "a staged constant would be overwritten by a step's upload"
            );
        }
    }

    /// The negative control for the region check: a layout that puts a constant in the step's range
    /// is REFUSED, not silently uploaded over.
    #[test]
    fn a_constant_inside_the_step_range_is_refused() {
        let mut keys = std::collections::BTreeSet::new();
        keys.insert("layer.0.attn.qk_norm_rope.cos_sin".to_owned());
        keys.insert("layer.0.attn.cu".to_owned());
        keys.insert("layer.0.attn.ids".to_owned());
        let pages = [0u32, 1];
        let seqs = [live(1, 0, &pages, 1)];
        let ids = [0u32];
        let probe = StepIndex {
            phase: Phase::Decode,
            rows: 1,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            committed_before: &[],
            band_width: None,
        };
        let l = BlobLayout::plan(&keys, &[probe]).expect("layout");
        assert!(l.static_len() > 0 && l.dynamic_len() > 0);
        // The control: a step's range that started at 0 would carry — and so overwrite — the rope
        // table. That is the layout the engine had before ws-h29, and the check above is what makes
        // "the constants are below the line" a fact rather than a convention.
        let staged_inside = l
            .constants
            .iter()
            .any(|(_, off, _)| (*off as usize) >= l.dynamic_at);
        assert!(
            !staged_inside,
            "a constant was laid out in the step's range"
        );
    }

    /// **The paged digest is the page lists and nothing else**, so a step that gains a page is
    /// dirty and a step that does not is not.
    ///
    /// Its negative control is the whole point: a digest that ignored something the law reads would
    /// let a stale `block_table` stand while a sequence's pages moved — a step reading another
    /// session's pages with every dispatch issuing. So each input is moved one at a time and the
    /// digest must move with it.
    #[test]
    fn the_paged_digest_moves_with_every_input_the_block_table_law_reads() {
        let pages = [4u32, 5, 6];
        let grown = [4u32, 5, 6, 7];
        let moved = [4u32, 5, 9];
        let two = [1u32, 2];
        let step = |blocks: &[u32], max_blocks: u32, n: usize| -> u64 {
            let seqs: Vec<SeqFacts<'_>> = (0..n)
                .map(|i| {
                    if i == 0 {
                        live(1, 0, blocks, 1)
                    } else {
                        live(2, 0, &two, 1)
                    }
                })
                .collect();
            let ids = vec![0u32; n];
            paged_digest(&StepIndex {
                phase: Phase::Decode,
                rows: n as u32,
                max_blocks,
                seqs: &seqs,
                ids: &ids,
                committed_before: &[],
                band_width: None,
            })
        };
        let base = step(&pages, 256, 2);
        assert_eq!(base, step(&pages, 256, 2), "the digest is not a function");
        assert_ne!(base, step(&grown, 256, 2), "a page GAINED did not move it");
        assert_ne!(
            base,
            step(&moved, 256, 2),
            "a page REPLACED did not move it"
        );
        assert_ne!(
            base,
            step(&pages, 2048, 2),
            "the table WIDTH did not move it"
        );
        assert_ne!(base, step(&pages, 256, 3), "the slot COUNT did not move it");
        // And the table itself agrees: equal digests, equal bytes.
        let seqs = [live(1, 0, &pages, 1)];
        let ids = [0u32];
        let mk = |max_blocks: u32| {
            index_table(
                "layer.3.attn.block_table",
                &StepIndex {
                    phase: Phase::Decode,
                    rows: 1,
                    max_blocks,
                    seqs: &seqs,
                    ids: &ids,
                    committed_before: &[],
                    band_width: None,
                },
            )
            .expect("block_table")
        };
        assert_eq!(mk(256), mk(256));
        assert_ne!(mk(256), mk(512));
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    const DYN: usize = 16;
    const PAGED: usize = 80;

    /// A blob whose step-varying region holds a recognisable pattern.
    fn blob(seed: u8) -> Vec<u8> {
        (0..96u8)
            .map(|i| i.wrapping_mul(3).wrapping_add(seed))
            .collect()
    }

    /// The law this frame rests on: two blobs equal before a step are equal after it, when the
    /// follower applies exactly the runs the leader wrote.
    #[test]
    fn applying_a_steps_runs_leaves_the_two_ranks_blobs_identical() {
        let mut leader = blob(0);
        let mut follower = blob(0);
        let mut ids_a = vec![0u8; 12];
        let mut ids_b = vec![0xff; 12];
        // The leader writes two keys' worth of bytes, the way `stage_indices` does.
        let runs = [(20u32, 8u32), (48, 4)];
        leader[20..28].copy_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]);
        leader[48..52].copy_from_slice(&[7, 7, 7, 7]);
        ids_a[..6].copy_from_slice(&[1, 0, 2, 0, 3, 0]);
        let mut frame = Vec::new();
        StepFrame::pack(&leader, &runs, &ids_a[..6], None, &mut frame).expect("pack");
        StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids_b).expect("apply");
        assert_eq!(
            leader[DYN..PAGED],
            follower[DYN..PAGED],
            "the step-varying regions must be identical after the frame"
        );
        assert_eq!(ids_a, ids_b, "the ids staging, tail included");
        // And the frame is SMALL: two runs of twelve bytes plus six id bytes, not the region.
        assert_eq!(frame.len(), STEP_FRAME_HEAD + 2 * 8 + 12 + 6);
        assert!(
            frame.len() < (PAGED - DYN),
            "a frame that is not smaller than the region it replaces has bought nothing"
        );
    }

    /// **THE DIGEST TURNS THE INDUCTION INTO AN ASSERTION.** Without it, a frame that omits a run
    /// applies cleanly and the two ranks part silently — which is the whole failure mode a
    /// compacted frame has. With it, the follower refuses AT THE STEP, by name.
    ///
    /// The case is a correct frame, which must pass with the digest on; the CONTROL is the same
    /// frame with one run dropped, which must fail with it on and (the second control) still pass
    /// with it off, because otherwise the test would not be about the digest.
    #[test]
    fn the_region_digest_catches_a_frame_that_left_a_run_out() {
        let mut leader = blob(0);
        leader[20..28].copy_from_slice(&[9; 8]);
        leader[48..52].copy_from_slice(&[7; 4]);
        let digest = Some(fnv1a(&leader[DYN..PAGED]));
        let mut frame = Vec::new();
        let mut ids = vec![0u8; 12];

        // The case: every run declared, and the digest matches.
        let mut follower = blob(0);
        StepFrame::pack(&leader, &[(20, 8), (48, 4)], &[], digest, &mut frame).expect("pack");
        StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids).expect("a correct frame");
        assert_eq!(leader[DYN..PAGED], follower[DYN..PAGED]);

        // CONTROL 1: a run dropped, digest ON — refused by name, at this step.
        let mut follower = blob(0);
        StepFrame::pack(&leader, &[(20, 8)], &[], digest, &mut frame).expect("pack");
        let e = StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids)
            .expect_err("a frame that left a run out was accepted with the digest on");
        assert!(e.to_string().contains("index blobs have parted"), "{e}");

        // CONTROL 2: the same frame with the digest OFF applies cleanly and the two ranks part in
        // silence — which is what the digest exists to stop, and why `0` means "not carried".
        let mut follower = blob(0);
        StepFrame::pack(&leader, &[(20, 8)], &[], None, &mut frame).expect("pack");
        StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids).expect("no digest, no check");
        assert_ne!(leader[DYN..PAGED], follower[DYN..PAGED]);
    }

    /// THE NEGATIVE CONTROL: drop one run and the two blobs must PART. Without it the test above
    /// would pass on a frame that carried nothing at all, because the two blobs start equal.
    #[test]
    fn a_dropped_run_leaves_the_two_blobs_different() {
        let mut leader = blob(0);
        let mut follower = blob(0);
        let mut ids = vec![0u8; 12];
        leader[20..28].copy_from_slice(&[9; 8]);
        leader[48..52].copy_from_slice(&[7; 4]);
        let mut frame = Vec::new();
        // Only the FIRST of the two writes is declared.
        StepFrame::pack(&leader, &[(20, 8)], &[], None, &mut frame).expect("pack");
        StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids).expect("apply");
        assert_ne!(
            leader[DYN..PAGED],
            follower[DYN..PAGED],
            "a frame that omits a run must leave the two ranks different, or the frame is not \
             what makes them the same"
        );
    }

    /// The first frame of a link carries the WHOLE region, so the induction starts from bytes
    /// rather than from an assumption about what page-locked memory holds.
    #[test]
    fn the_whole_region_frame_starts_the_induction() {
        let leader = blob(0);
        // A follower whose pinned buffer holds something else entirely.
        let mut follower = blob(77);
        let mut ids = vec![0u8; 12];
        assert_ne!(leader[DYN..PAGED], follower[DYN..PAGED]);
        let mut frame = Vec::new();
        StepFrame::pack_whole(&leader, DYN, PAGED, &[4, 0, 0, 0], None, &mut frame).expect("pack");
        StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids).expect("apply");
        assert_eq!(leader[DYN..PAGED], follower[DYN..PAGED]);
        assert_eq!(&ids[..4], &[4, 0, 0, 0]);
    }

    /// Every way a frame can be wrong is a refusal that NAMES it, never a silent partial apply.
    #[test]
    fn a_frame_that_is_not_this_ranks_is_refused_by_name() {
        let leader = blob(0);
        let mut follower = blob(0);
        let mut ids = vec![0u8; 12];
        let mut frame = Vec::new();
        // A run in the CONSTANTS, which this rank staged once at open and never re-uploads.
        StepFrame::pack(&leader, &[(4, 4)], &[], None, &mut frame).expect("pack");
        let e = StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids)
            .expect_err("a run below the step-varying region was accepted");
        assert!(
            e.to_string()
                .contains("outside this rank's step-varying region"),
            "{e}"
        );
        // A run in the BLOCK TABLE, which crosses in its own frame.
        StepFrame::pack(&leader, &[(76, 8)], &[], None, &mut frame).expect("pack");
        let e = StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids)
            .expect_err("a run above the step-varying region was accepted");
        assert!(
            e.to_string()
                .contains("outside this rank's step-varying region"),
            "{e}"
        );
        // A TRUNCATED frame.
        StepFrame::pack(&leader, &[(20, 8)], &[], None, &mut frame).expect("pack");
        frame.truncate(frame.len() - 3);
        let e = StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids)
            .expect_err("a truncated frame was accepted");
        assert!(e.to_string().contains("run(s) of"), "{e}");
        // More ids than this rank's staging holds.
        StepFrame::pack(&leader, &[], &[0u8; 20], None, &mut frame).expect("pack");
        let e = StepFrame::apply(&frame, &mut follower, DYN, PAGED, &mut ids)
            .expect_err("an oversized ids payload was accepted");
        assert!(e.to_string().contains("this rank's staging is"), "{e}");
        // And a head that is not even a head.
        let e = StepFrame::apply(&[1, 2, 3], &mut follower, DYN, PAGED, &mut ids)
            .expect_err("a three-byte frame was accepted");
        assert!(e.to_string().contains("its head is"), "{e}");
    }

    /// **THE CEILING AND THE FIRST FRAME ARE ONE ARITHMETIC.** `device::payload_width` - what a
    /// link lays out as its widest frame and what the follower's buffer is sized at - is the
    /// whole-region frame's own width, so the link's first frame fits it exactly. Written twice in
    /// two files is two things to be wrong; this pins them together, and a card would otherwise
    /// find it at the first step of the first prompt.
    #[test]
    fn the_whole_region_frame_is_exactly_the_ceiling_a_link_lays_out() {
        let region = 1932 * 1024usize;
        let ids = 8 * 1024usize;
        assert_eq!(
            StepFrame::bytes(&[(0, region as u32)], ids),
            STEP_FRAME_HEAD + 8 + region + ids,
            "the whole-region frame and `device::payload_width` must be the same number"
        );
    }

    /// The realistic shape, and the number the fix is worth: ws-h29's own census says a batch-8
    /// decode step writes 192 / 32 / 64 bytes of three tables whose SLOTS are 768 / 768 / 256 KiB.
    #[test]
    fn a_decode_steps_frame_is_kilobytes_where_the_region_is_megabytes() {
        // 16 attention layers x (pos 192 B + slot 64 B) + 48 GDN layers x (sidx 32 B), plus the
        // handful of per-step tables every layer shares.
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut at = 4096u32;
        for _ in 0..16 {
            runs.push((at, 192));
            at += 768 * 1024 / 16;
            runs.push((at, 64));
            at += 256 * 1024 / 16;
        }
        for _ in 0..48 {
            runs.push((at, 32));
            at += 768 * 1024 / 48;
        }
        let ids = 8usize * 4;
        let frame = StepFrame::bytes(&runs, ids);
        let region = 1932 * 1024;
        assert!(
            frame * 100 < region,
            "the frame is {frame} B against a {region} B region: under one percent is the claim"
        );
        assert_eq!(runs.len(), 80);
    }
}
