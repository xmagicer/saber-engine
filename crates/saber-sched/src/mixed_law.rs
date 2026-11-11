//! **The fused mixed step's host law** (workstream C8): the row map, the budget, the disjoint
//! ranges and the shared-state isolation of ONE forward over a prompt chunk and a decode half.
//!
//! C4's mixed step (`crate::MixedPrefill`, `StepExecutor::execute_mixed`) runs the chunk's forward
//! and the decode half's forward back to back — the bit-exact SERIAL form. The FUSED form
//! (`docs/EXEC.md` § the mixed step, LEDGER C8) runs one forward over `chunk + decode` rows:
//!
//! * **the row map** ([`FusedRows`]): the chunk's rows FIRST (`0..c`, where the prefill kernels'
//!   row-0 descriptors already point: `saber_attn_prefill`'s right-aligned mask, `conv_prep`,
//!   `chunk_scan`'s slot, the head's `sel` = row `c − 1`), the decode rows at the offset
//!   (`c..c + d`, in the decode bucket's own order), the rung's padding after them;
//! * **the rung**: the smallest planned prefill rung holding `c + d` — the row-local ops (GEMMs,
//!   producers, norms, reduces) run once at that rung's plan over all rows;
//! * **the budget**: prod's law (`room = budget − decode rows`) over WHOLE unpacked chunks
//!   ([`pack`], C10; C11 for the one-chunk form too): a chunk that does not fit a rung in class
//!   beside the decode rows (a long prompt's 2048-row chunk) rides whole and runs the serial
//!   form. C9's cut to `largest − d` ([`fused_room`]) moved the chunk boundary and is the control;
//! * **the two-launch ops** ([`FusedRows::launches`]): attention (`saber_attn_prefill` over the
//!   chunk's range, the paged decode kernel over the decode range), GDN (conv_prep + kkt + scan over
//!   the chunk, conv/recurrent update over the decode rows), the head (the chunk's last row + every
//!   decode row), the drafter (the context store over the chunk, the verify / proposal over the
//!   decode rows) — each over ONE range, and the ranges disjoint and covering the live rows;
//! * **isolation** ([`isolated`]): no KV page and no GDN slot of the chunk's sequence is a decode
//!   row's, and no two decode sequences share one (P12/P13's contamination law: a mixed forward
//!   must not read another row's pages).
//!
//! THE CLASS LAW (the census, `saber-engine/tests/mixed_c8.rs`): the decode rows leave their split
//! decode cells for the fold class and their exact reduce for the lossy one at <= 12 rows, so the
//! fused form is a NUMERICS candidate for every (rung, batch x width) pair; the chunk rows keep
//! their class below the 2048 rung. This module is the host half only; the engine's fused forward
//! is not built (LEDGER C8).

use std::ops::Range;

/// **C9's fused room, now the PRE-C11 CONTROL only** (`MixedPrefill::fused_rung_cut_control`):
/// prod's budget law (what `max_batched_tokens` leaves) capped so that the whole step fits the
/// widest rung. It cuts a long prompt's 2048-row chunk to `2048 − decode rows` — a chunk-boundary
/// move (G4) that [`chunk_sequence_holds`] refuses. Every served fused form takes its chunks from
/// [`pack`] instead (C11). 0 = the step runs its decode half alone.
#[must_use]
pub fn fused_room(max_batched_tokens: u32, decode_rows: u32, largest_rung: u32) -> u32 {
    max_batched_tokens
        .saturating_sub(decode_rows)
        .min(largest_rung.saturating_sub(decode_rows))
}

/// One fused mixed step's rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusedRows {
    /// The chunk's rows, first.
    pub chunk: Range<u32>,
    /// The decode half's rows, at the chunk's end.
    pub decode: Range<u32>,
    /// The prefill rung the row-local ops run at (`>= chunk + decode`).
    pub rung: u32,
}

/// Which side of a fused step an op's launch covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Every row of the rung (the row-local ops: GEMMs, producers, norms, reduces).
    All,
    /// The chunk's rows (prefill attention, the GDN chunk scan, the drafter's context store).
    Chunk,
    /// The decode rows (the paged decode attention, the GDN recurrent update, verify/proposal).
    Decode,
}

impl FusedRows {
    /// The map for a chunk of `chunk_rows` beside `decode_rows` decode rows over the ascending
    /// `rungs`: `None` when no rung holds both or either side is empty (a step with nothing to
    /// fuse is not a fused step).
    #[must_use]
    pub fn of(chunk_rows: u32, decode_rows: u32, rungs: &[u32]) -> Option<Self> {
        if chunk_rows == 0 || decode_rows == 0 {
            return None;
        }
        let m = chunk_rows.checked_add(decode_rows)?;
        let rung = rungs.iter().copied().find(|r| *r >= m)?;
        Some(Self {
            chunk: 0..chunk_rows,
            decode: chunk_rows..m,
            rung,
        })
    }

    /// **C9: the map the engine RUNS**: [`FusedRows::of`], refused (`None`, the step then runs
    /// in the serial form) when the rung is at or above `large_min` (the 512-thread producer
    /// class, `quant_large_min_m` = 2048) while the chunk alone would run below it — the one
    /// place C8's census found the chunk rows' class moving.
    #[must_use]
    pub fn of_in_class(
        chunk_rows: u32,
        decode_rows: u32,
        rungs: &[u32],
        large_min: u32,
    ) -> Option<Self> {
        let f = Self::of(chunk_rows, decode_rows, rungs)?;
        let alone = rungs.iter().copied().find(|r| *r >= chunk_rows)?;
        (f.rung < large_min || alone >= large_min).then_some(f)
    }

    /// Live rows (`chunk + decode`).
    #[must_use]
    pub fn live(&self) -> u32 {
        self.decode.end
    }

    /// The rows one launch of `side` covers.
    #[must_use]
    pub fn launches(&self, side: Side) -> Range<u32> {
        match side {
            Side::All => 0..self.rung,
            Side::Chunk => self.chunk.clone(),
            Side::Decode => self.decode.clone(),
        }
    }

    /// The head's rows: the chunk's last row, then every decode row, in that order.
    #[must_use]
    pub fn head_rows(&self) -> Vec<u32> {
        std::iter::once(self.chunk.end - 1)
            .chain(self.decode.clone())
            .collect()
    }

    /// The law itself: the two sides disjoint, the chunk first, together the live rows, inside
    /// the rung.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.chunk.start == 0
            && self.chunk.end == self.decode.start
            && self.chunk.end > 0
            && self.decode.end > self.decode.start
            && self.decode.end <= self.rung
    }
}

/// The top of the 256-thread producer class (`quant_large_min_m`): a rung at or above it moves
/// a lower rung's chunk (C5/C8's p2048 split). The engine's fused pick reads the same constant.
pub const LARGE_MIN_M: u32 = 2048;

/// Most chunks one packed step carries (prod's corpus: at most 7, C2/C4; one more for the
/// eight-prompt burst BetterBench sends at conc 8).
pub const MAX_PACKED_CHUNKS: u32 = 8;

/// **C10: a PACKED fused step's chunks** — prod's several-prompts-per-step law under this
/// engine's two class laws. Built by [`pack`]; `rows[k]` is the k-th queued prompt's chunk, in
/// arrival order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packing {
    /// Rows each carried chunk feeds, in queue (arrival) order.
    pub rows: Vec<u32>,
    /// The prefill rung the step runs at: the smallest rung holding every chunk plus the decode
    /// rows, or `None` when no rung holds them IN CLASS (the step then carries the head's chunk
    /// alone, in the serial form).
    pub rung: Option<u32>,
    /// The head's chunk was cut below its unpacked size because the OPERATOR's budget
    /// (`max_batched_tokens − decode rows`) is below it — C4's cut, never taken at prod's 8192.
    pub budget_cut: bool,
}

impl Packing {
    /// Chunks carried.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the step carries no chunk (the budget is spent).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Prompt rows carried, summed.
    #[must_use]
    pub fn chunk_rows(&self) -> u32 {
        self.rows.iter().sum()
    }

    /// **The row map**: chunk `k`'s rows, laid out per prompt in queue order from row 0.
    #[must_use]
    pub fn ranges(&self) -> Vec<Range<u32>> {
        let mut at = 0u32;
        self.rows
            .iter()
            .map(|r| {
                let x = at..at + r;
                at += r;
                x
            })
            .collect()
    }

    /// The decode rows' range, after the last chunk.
    #[must_use]
    pub fn decode_range(&self, decode_rows: u32) -> Range<u32> {
        let c = self.chunk_rows();
        c..c + decode_rows
    }
}

/// The rung chunks of `rows` beside `decode_rows` decode rows run at, IN CLASS: every chunk's
/// rows keep the chunk alone's class (C8/C9: every rung below `large_min` is one class for a row;
/// a rung at or above it is the 512-thread producer class, which only a lone chunk that is there
/// alone may take). `None` when no rung holds them in class.
#[must_use]
pub fn packed_in_class(
    rows: &[u32],
    decode_rows: u32,
    rungs: &[u32],
    large_min: u32,
) -> Option<u32> {
    let live: u32 = rows.iter().sum::<u32>().checked_add(decode_rows)?;
    let rung = rungs.iter().copied().find(|r| *r >= live)?;
    if rung < large_min {
        return Some(rung);
    }
    match rows {
        [one] => {
            let alone = rungs.iter().copied().find(|r| *r >= *one)?;
            (alone >= large_min).then_some(rung)
        }
        _ => None,
    }
}

/// **C10: the packing law.** `unpacked[i]` is the i-th queued prompt's NEXT chunk as the unpacked
/// schedule cuts it (`PrefillBuckets::chunk_for(remaining)`), in arrival order. The step takes,
/// from the queue head, every chunk that fits WHOLE:
///
/// * the head's chunk always rides (cut to the operator's budget `max_batched_tokens − decode
///   rows` only when that is below it — prod's law; never at prod's 8192);
/// * a further chunk rides while the step stays under the budget and `max_chunks` and IN CLASS
///   ([`packed_in_class`]: once two chunks share it, the rung holding them + the decode rows is
///   below [`LARGE_MIN_M`]); the first chunk that does not fit stops the packing (arrival order:
///   a later, smaller prompt never jumps a waiting one);
/// * **a chunk is never cut to fit** (prod cuts the last one): a cut moves the prompt's chunk
///   boundary, which is not bit-neutral on the sums (G4), so the chunk waits for the next step
///   whole and every prompt's chunk sequence is the unpacked one ([`chunk_sequence_holds`]).
///
/// `rung` is `None` when even the head alone has no in-class rung beside the decode rows (a
/// 2048-row chunk of a long prompt): the step carries the head's chunk alone at its unpacked
/// size and the executor runs it in the serial form (C4's bit-exact law). Empty when the budget
/// is spent.
#[must_use]
pub fn pack(
    unpacked: &[u32],
    decode_rows: u32,
    rungs: &[u32],
    max_batched_tokens: u32,
    max_chunks: u32,
) -> Packing {
    let room = max_batched_tokens.saturating_sub(decode_rows);
    let empty = Packing {
        rows: Vec::new(),
        rung: None,
        budget_cut: false,
    };
    let Some(&u0) = unpacked.first() else {
        return empty;
    };
    if room == 0 || u0 == 0 {
        return empty;
    }
    let head = u0.min(room);
    let mut rows = vec![head];
    let mut rung = packed_in_class(&rows, decode_rows, rungs, LARGE_MIN_M);
    if rung.is_some() {
        for &u in unpacked.iter().skip(1) {
            if rows.len() as u32 >= max_chunks.max(1) || u == 0 {
                break;
            }
            if rows.iter().sum::<u32>() + u > room {
                break;
            }
            rows.push(u);
            match packed_in_class(&rows, decode_rows, rungs, LARGE_MIN_M) {
                Some(r) => rung = Some(r),
                None => {
                    rows.pop();
                    break;
                }
            }
        }
    }
    Packing {
        rows,
        rung,
        budget_cut: head < u0,
    }
}

/// **The chunk-sequence assertion**: the chunks a prompt was fed, as `(tokens_done_before, rows)`
/// in order, are exactly the unpacked schedule's cut (`cut` = `PrefillBuckets::cut(prompt_len)`'s
/// rows), so packing moved no chunk boundary (G4). `Err` names the first chunk that differs.
///
/// # Errors
/// The first fed chunk that is not the unpacked one, or a count mismatch.
pub fn chunk_sequence_holds(cut: &[u32], fed: &[(u32, u32)]) -> Result<(), String> {
    let mut done = 0u32;
    for (i, want) in cut.iter().enumerate() {
        match fed.get(i) {
            Some(&(d, r)) if d == done && r == *want => done += r,
            Some(&(d, r)) => {
                return Err(format!(
                    "chunk {i}: fed {r} row(s) at {d}, the unpacked schedule cuts {want} at {done}"
                ))
            }
            None => return Err(format!("chunk {i} ({want} rows at {done}) was never fed")),
        }
    }
    if fed.len() != cut.len() {
        return Err(format!(
            "{} chunk(s) fed, the unpacked schedule cuts {}",
            fed.len(),
            cut.len()
        ));
    }
    Ok(())
}

/// P12/P13's contamination law for a PACKED step: every chunk's sequence and every decode
/// sequence own disjoint KV pages and distinct GDN slot sets — chunk against chunk, chunk against
/// decode row. Two DECODE rows may share read-only prompt pages (the branches of an `n > 1`
/// fork), never a GDN set. `Err` names the first shared page or slot.
///
/// # Errors
/// The first KV block or GDN set two sequences share where the law forbids it.
pub fn isolated_packed(chunks: &[RowState<'_>], decode: &[RowState<'_>]) -> Result<(), String> {
    let mut pages: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut sets: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
    for (k, c) in chunks.iter().enumerate() {
        let me = format!("chunk {k}");
        if let Some(prev) = sets.insert(c.gdn_set, me.clone()) {
            return Err(format!("GDN set {} is {prev}'s and {me}'s", c.gdn_set));
        }
        for b in c.kv_blocks {
            if let Some(prev) = pages.insert(*b, k) {
                if prev != k {
                    return Err(format!("KV block {b} is chunk {prev}'s and {me}'s"));
                }
            }
        }
    }
    for (i, d) in decode.iter().enumerate() {
        let me = format!("decode sequence {i}");
        if let Some(prev) = sets.insert(d.gdn_set, me.clone()) {
            return Err(format!("GDN set {} is {prev}'s and {me}'s", d.gdn_set));
        }
        for b in d.kv_blocks {
            if let Some(k) = pages.get(b) {
                return Err(format!("KV block {b} is chunk {k}'s and {me}'s"));
            }
        }
    }
    Ok(())
}

/// A sequence's shared state in a fused step: its KV pages and its GDN slot set.
#[derive(Clone, Debug)]
pub struct RowState<'a> {
    /// The KV blocks its rows read and write.
    pub kv_blocks: &'a [u32],
    /// Its GDN slot set.
    pub gdn_set: u32,
}

/// Check the page tables and resident token counts of every live sequence in a fused step.
/// KV appends start at the resident count. Only whole pages before that cursor can be
/// shared, at the same logical position in every table. GDN sets always have one owner.
///
/// # Errors
/// Invalid page geometry, a shared writable page, a misplaced page or a shared GDN set.
pub fn isolated_at_cursors<'a>(
    rows: impl IntoIterator<Item = (RowState<'a>, u32)>,
    block_tokens: u32,
) -> Result<(), String> {
    use std::collections::{hash_map::Entry, HashMap, HashSet};

    let mut pages = HashMap::new();
    let mut sets = HashSet::new();
    for (row, done_before) in rows {
        let complete = done_before
            .checked_div(block_tokens)
            .ok_or("zero tokens per KV block")? as usize;
        if complete > row.kv_blocks.len() {
            return Err(format!(
                "GDN set {} has fewer KV blocks than its resident prefix",
                row.gdn_set
            ));
        }
        if !sets.insert(row.gdn_set) {
            return Err(format!(
                "GDN set {} belongs to multiple live sequences",
                row.gdn_set
            ));
        }
        for (index, &block) in row.kv_blocks.iter().enumerate() {
            let readonly = index < complete;
            match pages.entry(block) {
                Entry::Vacant(entry) => {
                    entry.insert((row.gdn_set, index, readonly));
                }
                Entry::Occupied(entry) => {
                    let (owner, previous_index, previous_readonly) = *entry.get();
                    if !readonly || !previous_readonly || index != previous_index {
                        return Err(format!(
                            "KV block {block} at logical pages {previous_index}/{index} of GDN sets {owner}/{} is not a shared complete prefix page",
                            row.gdn_set
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// P12/P13's contamination law for a fused step: the chunk's sequence and every decode sequence
/// own disjoint KV pages and distinct GDN slot sets. `Err` names the first shared page or slot.
///
/// # Errors
/// The first KV block or GDN set two sequences share.
pub fn isolated(chunk: &RowState<'_>, decode: &[RowState<'_>]) -> Result<(), String> {
    let mut seen_pages = std::collections::HashMap::new();
    let mut seen_sets = std::collections::HashMap::new();
    for (who, s) in std::iter::once((usize::MAX, chunk)).chain(decode.iter().enumerate()) {
        let name = |i: usize| {
            if i == usize::MAX {
                "the chunk".to_string()
            } else {
                format!("decode sequence {i}")
            }
        };
        if let Some(prev) = seen_sets.insert(s.gdn_set, who) {
            return Err(format!(
                "GDN set {} is {}'s and {}'s",
                s.gdn_set,
                name(prev),
                name(who)
            ));
        }
        for b in s.kv_blocks {
            if let Some(prev) = seen_pages.insert(*b, who) {
                if prev != who {
                    return Err(format!(
                        "KV block {b} is {}'s and {}'s",
                        name(prev),
                        name(who)
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];

    #[test]
    fn the_row_map_puts_the_chunk_first_and_fits_one_rung() {
        // Q3 mean chunk (222 live rows) beside b7s5 (35 rows): 257 rows, so p320.
        let f = FusedRows::of(222, 35, &LADDER).expect("fits");
        assert!(f.holds());
        assert_eq!(
            (f.chunk.clone(), f.decode.clone(), f.rung),
            (0..222, 222..257, 320)
        );
        assert_eq!(f.launches(Side::All), 0..320);
        assert_eq!(f.head_rows().first(), Some(&221));
        assert_eq!(f.head_rows().len(), 36);
        // Exactly a rung: no padding.
        assert_eq!(FusedRows::of(200, 56, &LADDER).expect("fits").rung, 256);
        // Nothing to fuse, or nothing holds it: None (the step is not fused).
        assert_eq!(FusedRows::of(0, 35, &LADDER), None);
        assert_eq!(FusedRows::of(35, 0, &LADDER), None);
        assert_eq!(FusedRows::of(2000, 56, &LADDER), None);
        // Every chunk the fused budget admits fits: the law and the room agree.
        for d in [1u32, 5, 8, 35, 40, 56, 64] {
            let room = fused_room(8192, d, 2048);
            assert_eq!(room, 2048 - d);
            for c in [1u32, 96, 222, room] {
                let f = FusedRows::of(c, d, &LADDER).expect("admitted means planned");
                assert!(f.holds(), "c {c} d {d}");
                let (a, b) = (f.launches(Side::Chunk), f.launches(Side::Decode));
                assert!(a.end <= b.start, "disjoint");
                assert_eq!(b.end - a.start, c + d, "together the live rows");
            }
        }
    }

    #[test]
    fn the_engine_map_refuses_the_one_class_move() {
        // A 1500-row chunk alone is p1536; beside 64 decode rows it would be p2048 (t512
        // producers): refused, the step runs serially.
        assert_eq!(FusedRows::of_in_class(1500, 64, &LADDER, 2048), None);
        // ...but the plain map would have taken it (the control).
        assert_eq!(FusedRows::of(1500, 64, &LADDER).map(|f| f.rung), Some(2048));
        // A chunk that is p2048 alone stays in class at p2048.
        assert_eq!(
            FusedRows::of_in_class(1600, 64, &LADDER, 2048).map(|f| f.rung),
            Some(2048)
        );
        // Below the split every map is in class.
        assert_eq!(
            FusedRows::of_in_class(178, 40, &LADDER, 2048).map(|f| f.rung),
            Some(256)
        );
    }

    #[test]
    fn the_fused_budget_is_prods_law_capped_by_the_widest_rung() {
        // Prod's law binds when the operator lowers the budget below the rung.
        assert_eq!(fused_room(512, 56, 2048), 456);
        // The rung binds at prod's budget: the serial form would take 2048 here.
        assert_eq!(fused_room(8192, 56, 2048), 1992);
        // Spent.
        assert_eq!(fused_room(56, 56, 2048), 0);
        assert_eq!(fused_room(8192, 2048, 2048), 0);
        // The control: the serial form's room (budget − d) is NOT capped, and would overflow.
        let serial = 8192u32 - 56;
        assert!(FusedRows::of(serial.min(2048), 56, &LADDER).is_none());
    }

    #[test]
    fn isolation_refuses_a_shared_page_or_slot() {
        let chunk = RowState {
            kv_blocks: &[10, 11, 12],
            gdn_set: 3,
        };
        let a = RowState {
            kv_blocks: &[20, 21],
            gdn_set: 4,
        };
        let b = RowState {
            kv_blocks: &[30],
            gdn_set: 5,
        };
        assert!(isolated(&chunk, &[a.clone(), b.clone()]).is_ok());
        // Negative controls: P12's stride defect in miniature (a decode row reading the chunk's
        // page), and two sequences on one GDN set.
        let stray = RowState {
            kv_blocks: &[30, 12],
            gdn_set: 5,
        };
        let e = isolated(&chunk, &[a.clone(), stray]).expect_err("shared page");
        assert!(e.contains("KV block 12"), "{e}");
        let twin = RowState {
            kv_blocks: &[40],
            gdn_set: 4,
        };
        let e = isolated(&chunk, &[a, twin]).expect_err("shared slot");
        assert!(e.contains("GDN set 4"), "{e}");
    }

    /// The unpacked cut of a prompt on the default ladder (largest chunk 2048).
    fn cut(mut n: u32) -> Vec<u32> {
        let mut out = Vec::new();
        while n > 0 {
            let c = n.min(2048);
            out.push(c);
            n -= c;
        }
        out
    }

    #[test]
    fn eight_at_once_pack_under_the_class_cap_and_the_one_chunk_law_is_the_control() {
        // BetterBench's burst: eight prompts of the corpus's spread (35..332, mean ~178) beside
        // a b8s1 decode half. Everything fits p1536 (1,424 + 8 rows): ONE step carries all eight.
        let q = [35u32, 332, 178, 150, 210, 96, 240, 183];
        let p = pack(&q, 8, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!(p.rows, q.to_vec());
        assert_eq!(p.rung, Some(1536));
        assert!(!p.budget_cut);
        let r = p.ranges();
        assert_eq!(r[0], 0..35);
        assert_eq!(r[1], 35..367);
        assert_eq!(p.decode_range(8), 1424..1432);
        // THE CONTROL: C9's one-chunk law carries the head only; seven prompts wait a step each.
        let one = pack(&q, 8, &LADDER, 8192, 1);
        assert_eq!(one.rows, vec![35]);
        assert_eq!(one.rung, Some(96));
        // The class cap binds before the budget: 64 decode rows leave 1,472 of p1536, so the
        // sixth chunk (183) waits whole rather than being cut to fit, and the two short ones
        // behind it wait too (arrival order).
        let q = [300u32, 300, 300, 300, 200, 183, 50, 50];
        let p = pack(&q, 64, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!(p.rows, q[..5].to_vec());
        assert_eq!(p.rung, Some(1536));
        assert!(p.chunk_rows() + 64 <= 1536);
    }

    #[test]
    fn staggered_arrivals_pack_what_is_queued() {
        // One queued prompt: the step is C9's (a lone chunk at its own in-class rung).
        let p = pack(&[222], 35, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!((p.rows.clone(), p.rung), (vec![222], Some(320)));
        // Two: both ride.
        let p = pack(&[222, 90], 35, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!((p.rows.clone(), p.rung), (vec![222, 90], Some(384)));
        // Spent budget: nothing rides.
        assert!(pack(&[222], 8192, &LADDER, 8192, MAX_PACKED_CHUNKS).is_empty());
        assert!(pack(&[], 8, &LADDER, 8192, MAX_PACKED_CHUNKS).is_empty());
    }

    #[test]
    fn one_long_and_seven_short_keeps_the_long_prompts_chunks_whole() {
        // A 5000-token prompt at the head: its chunk is 2048 rows, which no rung holds beside the
        // decode rows. It rides ALONE at its unpacked size (the serial form) — never cut to 1992
        // (C9's fused room would have cut it: a G4 boundary move).
        let long = cut(5000);
        assert_eq!(long, vec![2048, 2048, 904]);
        let q = [2048u32, 100, 120, 80, 90, 110, 60, 70];
        let p = pack(&q, 56, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!(p.rows, vec![2048]);
        assert_eq!(p.rung, None, "no in-class rung: the serial form");
        // Its tail (904) packs with the short ones under p1536.
        let q = [904u32, 100, 120, 80, 90, 110, 60, 70];
        let p = pack(&q, 56, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!(p.rows, vec![904, 100, 120, 80, 90, 110, 60]);
        assert_eq!(p.rung, Some(1536));
        // A chunk that is p2048 alone rides alone at 2048 in class, and nothing joins it.
        let p = pack(&[1700, 50], 40, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!((p.rows.clone(), p.rung), (vec![1700], Some(2048)));
        // A short head followed by the long prompt: the long one waits whole.
        let p = pack(&[100, 2048, 50], 40, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!((p.rows.clone(), p.rung), (vec![100], Some(192)));
    }

    #[test]
    fn two_chunks_never_share_the_2048_class() {
        // 1400 + 200 + 8 = 1608 rows: only p2048 holds them, and the 200-row chunk would move to
        // the 512-thread producers there. Refused; the second waits.
        assert_eq!(packed_in_class(&[1400, 200], 8, &LADDER, LARGE_MIN_M), None);
        let p = pack(&[1400, 200], 8, &LADDER, 8192, MAX_PACKED_CHUNKS);
        assert_eq!((p.rows.clone(), p.rung), (vec![1400], Some(1536)));
        // A lone chunk below p2048 alone that needs p2048 with its decode rows: refused (C9's
        // `of_in_class`), the serial form.
        assert_eq!(packed_in_class(&[1500], 64, &LADDER, LARGE_MIN_M), None);
        assert_eq!(pack(&[1500], 64, &LADDER, 8192, 8).rung, None);
    }

    #[test]
    fn the_operators_budget_cuts_the_head_only() {
        // Budget 512 beside 56 decode rows: room 456. The head is cut (prod's law, counted), and
        // nothing else fits.
        let p = pack(&[900, 50], 56, &LADDER, 512, MAX_PACKED_CHUNKS);
        assert_eq!(p.rows, vec![456]);
        assert!(p.budget_cut);
        // A second chunk that would overrun the budget waits whole.
        let p = pack(&[300, 200], 56, &LADDER, 512, MAX_PACKED_CHUNKS);
        assert_eq!(p.rows, vec![300]);
        assert!(!p.budget_cut);
    }

    #[test]
    fn the_chunk_sequence_assertion_holds_and_its_control_fires() {
        let c = cut(5000);
        assert!(chunk_sequence_holds(&c, &[(0, 2048), (2048, 2048), (4096, 904)]).is_ok());
        // C9's fused room (2048 − 56) as the control: the boundary moved.
        let e = chunk_sequence_holds(&c, &[(0, 1992), (1992, 1992), (3984, 1016)])
            .expect_err("a moved boundary");
        assert!(e.contains("chunk 0"), "{e}");
        assert!(chunk_sequence_holds(&c, &[(0, 2048)]).is_err());
    }

    #[test]
    fn isolation_across_chunks() {
        let a = RowState {
            kv_blocks: &[1, 2],
            gdn_set: 3,
        };
        let b = RowState {
            kv_blocks: &[4],
            gdn_set: 5,
        };
        let d = RowState {
            kv_blocks: &[9],
            gdn_set: 7,
        };
        let d2 = RowState {
            kv_blocks: &[9],
            gdn_set: 8,
        };
        // Two decode rows may share read-only prompt pages (n > 1 branches).
        assert!(isolated_packed(&[a.clone(), b.clone()], &[d.clone(), d2]).is_ok());
        // Negative controls: two chunks on one page, a chunk and a decode row on one page, two
        // chunks on one GDN set.
        let b_bad = RowState {
            kv_blocks: &[4, 2],
            gdn_set: 5,
        };
        let e = isolated_packed(&[a.clone(), b_bad], &[]).expect_err("page");
        assert!(e.contains("KV block 2"), "{e}");
        let d_bad = RowState {
            kv_blocks: &[4],
            gdn_set: 7,
        };
        let e = isolated_packed(&[a.clone(), b.clone()], &[d_bad]).expect_err("page");
        assert!(e.contains("KV block 4"), "{e}");
        let twin = RowState {
            kv_blocks: &[6],
            gdn_set: 3,
        };
        let e = isolated_packed(&[a, twin], &[d]).expect_err("set");
        assert!(e.contains("GDN set 3"), "{e}");
    }

    #[test]
    fn cached_pages_must_be_complete_for_every_holder() {
        let row = |blocks, gdn_set| RowState {
            kv_blocks: blocks,
            gdn_set,
        };
        let a = row(&[10, 20, 21], 1);
        let b = row(&[10, 30, 31], 2);
        let c = row(&[10, 40, 41], 3);
        assert!(isolated_at_cursors([(a.clone(), 16), (b.clone(), 17), (c, 32)], 16).is_ok());
        // An append to the partially occupied page must still fail in either ordering.
        for rows in [
            [(a.clone(), 15), (b.clone(), 16)],
            [(b.clone(), 16), (a.clone(), 15)],
        ] {
            assert!(isolated_at_cursors(rows, 16)
                .unwrap_err()
                .contains("KV block 10"));
        }
        // Sharing a future page, a page at a different logical position, or mutable
        // recurrent state remains invalid even when the two also share a valid prefix.
        let future = row(&[10, 35, 21], 4);
        let shifted = row(&[50, 10, 51], 5);
        let same_set = row(&[10, 60], 1);
        for (other, done, what) in [
            (future, 16, "KV block 21"),
            (shifted, 32, "KV block 10"),
            (same_set, 16, "GDN set 1"),
        ] {
            assert!(isolated_at_cursors([(a.clone(), 16), (other, done)], 16)
                .unwrap_err()
                .contains(what));
        }
    }
}
