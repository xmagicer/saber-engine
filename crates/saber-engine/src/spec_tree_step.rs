//! **P27-15c: the tree verify's STEP** -- the arm's boot law (`--engine-spec-tree`: what runs, and
//! the refusal by name when the asked form cannot), the 10-row b1 step TABLES a tree verify binds
//! (ancestor words, rope positions by depth, the KV slot of every row, the conv window's `tsrc` /
//! `hsrc`, the GDN record pair's prologue / parity), the VERDICT in the scheduler's convention (in
//! place of `accept_prefix`), what the verdict moves (the accepted path's KV slots, the drafter's
//! aux rows, the next committed conv window, the pending advance), the record pair as a LEDGER
//! class, and the plans built from the proposal's own lattice read.
//!
//! Host arithmetic only: every table here is what the device form binds, tested against the gated
//! oracles' own table builders (`tools/capture/p27_15b/conv_gate.py`'s `tsrc_of` /
//! `commit_entries` / `kv_moves`, through `tools/capture/p27_15c/step_tables_ref.py`) on P27-10's
//! 2,427 recorded lattices (`tests/p27_15c_step_tables.rs`).
//!
//! # The arm is SAFE at every stop point
//!
//! [`resolve_arm`] is the one place the asked form becomes the run's form. A form this binary
//! cannot run (no drafter, the selector's `explen` chain, TP != 2, or the 10-row tree verify form
//! not built: [`TREE_VERIFY_FORM_BUILT`]) is REFUSED BY NAME and the run falls back to `off`, the
//! chain of 7 -- counted (`spectree ... fallback`), never silent (structural rule 5).

use saber_serve::engine::TokenId;

use crate::draft::{AcceptError, Verdict};
use crate::spec_tree::{
    ConvSrc, LatticeView, PendingAdvance, SpecTreeForm, TreeError, TreePlan, TreeVerdict,
    VerifyRecords, SABER_ROWS_PER_SEQ,
};

/// Rows of the b1 tree verify form (the SABER decode attention tile at GQA 6).
pub const TREE_ROWS: usize = SABER_ROWS_PER_SEQ as usize;

/// The GDN conv's taps (`conv_width`): the window is `taps - 1` committed inputs.
pub const CONV_TAPS: u32 = 4;

/// Committed conv inputs a window holds (`taps - 1`).
pub const CONV_WINDOW: usize = (CONV_TAPS - 1) as usize;

/// **Whether this binary carries the 10-row b1 tree verify FORM** (the graph's tree GDN /
/// attention layers, the captured `b1t10` bucket, the follower's mirror). Until it does, an
/// `ancestor9` boot is refused by name and runs the chain ([`resolve_arm`]).
pub const TREE_VERIFY_FORM_BUILT: bool = true;

/// The ledger class of the GDN tree record pair (P27-7's ledger; `engledger` line).
pub const RECORD_CLASS: &str = "engine.spectree.records";

/// What the boot knows when it resolves the arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmContext {
    /// Tensor-parallel ranks.
    pub tp: u32,
    /// A DFlash2 drafter is booted (`--draft-pack`).
    pub drafter: bool,
    /// Query block used by the native DFlash2 proposal.
    pub draft_block: u32,
    /// The selector re-ranks the chain on the host (`--engine-selector-rank explen`).
    pub selector_reranks: bool,
    /// The 10-row tree verify form exists ([`TREE_VERIFY_FORM_BUILT`]; a test may say otherwise).
    pub form_built: bool,
    /// Sequences the boot admits at once (`--max-seqs`). P27-15e: the tree runs every decode step
    /// of a b1-only boot; above 1 a sequence could take a CHAIN step after a tree step, whose
    /// flush (the pending advance, the conv window, the recurrent slot, the KV moves on both
    /// ranks) is R2's -- refused by name until it is built.
    pub max_seqs: u32,
}

/// The asked form, the form that runs, and why they differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmResolution {
    /// `--engine-spec-tree` as asked.
    pub asked: SpecTreeForm,
    /// The form the verify runs.
    pub runs: SpecTreeForm,
    /// The refusal, by name, when `runs != asked`.
    pub refused: Option<String>,
    /// The proposal builds the tree plans from its lattice read (the arm asked and the lattice
    /// exists) -- counted on the `spectree` line even while the verify runs the chain.
    pub plans: bool,
}

impl ArmResolution {
    /// The boot line (`spectree form ...`).
    #[must_use]
    pub fn line(&self) -> String {
        match &self.refused {
            None => format!(
                "spectree  form {} (asked {}), plans from the lattice {}",
                self.runs.as_str(),
                self.asked.as_str(),
                if self.plans { "on" } else { "off" }
            ),
            Some(why) => format!(
                "spectree  form {} REFUSED -> runs {} (fallback): {why}; plans from the lattice {}",
                self.asked.as_str(),
                self.runs.as_str(),
                if self.plans { "on (counted)" } else { "off" }
            ),
        }
    }

    /// True when the asked form fell back.
    #[must_use]
    pub fn fell_back(&self) -> bool {
        self.refused.is_some()
    }
}

/// **The arm's boot law**: what `--engine-spec-tree` runs in this boot. `off` always runs; an
/// `ancestor9` the boot cannot run is refused BY NAME and falls back to `off`.
#[must_use]
pub fn resolve_arm(asked: SpecTreeForm, ctx: ArmContext) -> ArmResolution {
    let off = |why: String, plans: bool| ArmResolution {
        asked,
        runs: SpecTreeForm::Off,
        refused: Some(why),
        plans,
    };
    match asked {
        SpecTreeForm::Off => ArmResolution {
            asked,
            runs: SpecTreeForm::Off,
            refused: None,
            plans: false,
        },
        SpecTreeForm::Ancestor9 | SpecTreeForm::Ancestor9Pair | SpecTreeForm::Best16 => {
            let supports_wide_draft =
                matches!(asked, SpecTreeForm::Ancestor9 | SpecTreeForm::Best16);
            let wide = supports_wide_draft && ctx.draft_block == 16;
            if !ctx.drafter || (ctx.draft_block != 8 && !wide) {
                return off(
                    format!(
                        "{} requires the DFlash2 drafter at block 8{} (drafter {}, block {})",
                        asked.as_str(),
                        if supports_wide_draft { " or 16" } else { "" },
                        if ctx.drafter { "on" } else { "off" },
                        ctx.draft_block
                    ),
                    false,
                );
            }
            if wide && ctx.max_seqs != 1 {
                return off(
                    "the block-16 native tree requires one active sequence".to_owned(),
                    false,
                );
            }
            if ctx.selector_reranks {
                return off(
                    "`--engine-selector-rank explen` re-ranks the CHAIN the tree replaces: pick \
                     one"
                    .to_owned(),
                    false,
                );
            }
            if ctx.tp != 2 {
                return off(
                    format!(
                        "the tree verify form is planned at TP=2 (the 27B's served pair); this \
                         boot is TP={}",
                        ctx.tp
                    ),
                    true,
                );
            }
            if !ctx.form_built {
                return off(
                    "the 10-row b1 tree verify form (tree GDN / anc attention layers, the \
                     captured b1t10 bucket, the follower's mirror) is not built in this binary \
                     (P27-15c: flag, tables, verdict, ledger class and lattice plans wired; \
                     P27-15d: the form's PLAN -- `Bucket::DecodeTree`, the tree ops, descriptors, \
                     bindings and blob tables -- host-tested; the engine's tree bucket, capture, \
                     dispatch, commit and follower are the next round)"
                        .to_owned(),
                    true,
                );
            }
            // P27-W2-4 (R2): `ancestor9` flushes every tree step to the chain's form (K/V moves,
            // conv window, recurrent advance into the chain's slot, both ranks), so a sequence may
            // take a chain step -- a batched decode, a mixed half -- after any tree step, and a
            // served multi-sequence boot runs the tree at b1 and the chain above. The pair control
            // and the 16-row `best16` form have no flush and stay b1-only.
            if ctx.max_seqs > 1 && asked != SpecTreeForm::Ancestor9 {
                return off(
                    format!(
                        "the `{}` form has no chain-after-tree flush (R2 is `ancestor9`'s): at \
                         --max-seqs {} a sequence could take a chain step after a tree step; boot \
                         with --max-seqs 1",
                        asked.as_str(),
                        ctx.max_seqs
                    ),
                    true,
                );
            }
            ArmResolution {
                asked,
                runs: asked,
                refused: None,
                plans: true,
            }
        }
    }
}

/// The record pair's ledger bytes a rank: ONE pair (the arm is b1-only: any other slot's pending
/// advance is flushed before a tree verify, [`flush_foreign`]), 0 when the arm does not run.
#[must_use]
pub fn record_ledger_bytes(runs: SpecTreeForm, gdn_layers: u32, v_heads_per_rank: u32) -> u64 {
    match runs {
        SpecTreeForm::Off => 0,
        SpecTreeForm::Ancestor9 | SpecTreeForm::Ancestor9Pair => {
            crate::spec_tree::tree_record_pair_bytes(gdn_layers, v_heads_per_rank)
        }
        // E4-3: the 16-row record (U / k^ at 16 rows), the same pair law
        SpecTreeForm::Best16 => {
            2 * u64::from(gdn_layers)
                * u64::from(v_heads_per_rank)
                * crate::spec_tree::tree_record_bytes(runs.rows())
        }
    }
}

/// **One pair a rank** (the b1-only arm): before slot `slot`'s tree verify, every OTHER slot's
/// pending advance is flushed (its own launch), so the only live record is `slot`'s own and the
/// verify writes the other parity. Returns the flushes to issue, in slot order, then the verify's
/// records.
pub fn flush_foreign(
    law: &mut crate::spec_tree::AdvanceLaw,
    slots: usize,
    slot: usize,
) -> (Vec<(usize, PendingAdvance)>, VerifyRecords) {
    let mut flushed = Vec::new();
    for s in 0..slots {
        if s != slot {
            if let Some(p) = law.flush(s) {
                flushed.push((s, p));
            }
        }
    }
    (flushed, law.verify(slot))
}

/// The chain law's committed conv window for `naccept` (`saber_gdn_conv_update` reads entries
/// `naccept - 1 ..`): the window a tree step starts from after a chain step or a prefill.
#[must_use]
pub fn chain_window(naccept: u32) -> [i32; CONV_WINDOW] {
    let o = naccept.max(1) as i32 - 1;
    [o, o + 1, o + 2]
}

/// The KV cache's token slot of position `pos` (`block_table[pos / bs] * bs + pos % bs`).
#[must_use]
pub fn kv_slot(kv_blocks: &[u32], block_size: u32, pos: u32) -> Option<i64> {
    let bs = block_size.max(1);
    kv_blocks
        .get((pos / bs) as usize)
        .map(|b| i64::from(*b) * i64::from(bs) + i64::from(pos % bs))
}

/// **The b1 tree verify's step tables** for one sequence: what the device form binds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeStepTables {
    /// Rows the plan verifies (1 + nodes); rows past it are the form's padding.
    pub live_rows: usize,
    /// The token each row embeds ([`TREE_ROWS`]; padding rows = the anchor).
    pub ids: Vec<i64>,
    /// One u16 ancestor word a row (`anc`: the tree attention's mask and the GDN tree's Gram);
    /// a padding row sees the anchor and itself.
    pub words: Vec<u16>,
    /// **Rope by depth**: row `r`'s position = the anchor's + its depth (padding: depth 1).
    pub positions: Vec<i32>,
    /// The KV slot every row's K/V is WRITTEN at: its ROW slot (`anchor + r`), compacted to the
    /// depth slots after the verdict.
    pub slots: Vec<i64>,
    /// `seqused_k`: the anchor's position + the form's rows (every row slot is resident).
    pub seq_len: i32,
    /// The conv window's older taps a row (`p27_gdn_conv_tree`'s `tsrc` `[rows][4]`: 0..2 = the
    /// committed window, `3 + r` = verify row `r`; the 4th entry is 0).
    pub tsrc: Vec<[i32; 4]>,
    /// The committed window's cache entries (`hsrc` `[4]`, the 4th 0).
    pub hsrc: [i32; 4],
    /// The GDN tree's `acc`: the prologue's accepted row, -1 = none.
    pub acc: i32,
    /// The record parity the prologue reads (`ts_in`), meaningful when `acc >= 0`.
    pub ts_in: u8,
    /// The record parity this verify writes (`ts_out`).
    pub ts_out: u8,
}

impl TreeStepTables {
    /// The tables of `plan` verified with row 0 at position `anchor_pos` (the sequence's resident
    /// tokens before the step), over the sequence's block table, from the committed conv window
    /// `window` and the records the advance law gave this verify.
    ///
    /// # Errors
    /// A plan wider than the form, or a block table without a page for the form's last row.
    pub fn new(
        plan: &TreePlan,
        anchor_pos: u32,
        kv_blocks: &[u32],
        block_size: u32,
        window: [i32; CONV_WINDOW],
        rec: VerifyRecords,
    ) -> Result<TreeStepTables, TreeError> {
        Self::new_rows(
            plan, TREE_ROWS, anchor_pos, kv_blocks, block_size, window, rec,
        )
    }

    /// [`TreeStepTables::new`] for a form of `form_rows` rows (E4-3: 16 under `best16`).
    ///
    /// # Errors
    /// As [`TreeStepTables::new`].
    pub fn new_rows(
        plan: &TreePlan,
        form_rows: usize,
        anchor_pos: u32,
        kv_blocks: &[u32],
        block_size: u32,
        window: [i32; CONV_WINDOW],
        rec: VerifyRecords,
    ) -> Result<TreeStepTables, TreeError> {
        let live = plan.rows();
        if live > form_rows || live == 0 || form_rows > crate::spec_tree::TREE_MAX_ROWS as usize {
            return Err(TreeError::TooWide { rows: live as u32 });
        }
        let mut ids = plan.tokens().to_vec();
        let mut words = plan.words().to_vec();
        let mut positions: Vec<i32> = plan
            .positions(anchor_pos)
            .into_iter()
            .map(|p| p as i32)
            .collect();
        let mut tsrc: Vec<[i32; 4]> = (0..live)
            .map(|r| {
                let mut t = [0i32; 4];
                for (j, s) in plan.conv_sources(r, CONV_TAPS)[..CONV_WINDOW]
                    .iter()
                    .enumerate()
                {
                    t[j] = match *s {
                        ConvSrc::Tail(e) => e as i32,
                        ConvSrc::Row(x) => CONV_WINDOW as i32 + x as i32,
                    };
                }
                t
            })
            .collect();
        for r in live..form_rows {
            ids.push(plan.tokens()[0]);
            words.push(1 | (1u16 << r));
            positions.push(anchor_pos as i32 + 1);
            // a depth-1 padding row: the window's last two + the anchor's row
            tsrc.push([1, 2, CONV_WINDOW as i32, 0]);
        }
        let mut slots = Vec::with_capacity(form_rows);
        for r in 0..form_rows as u32 {
            slots.push(
                kv_slot(kv_blocks, block_size, anchor_pos + r).ok_or_else(|| {
                    TreeError::Malformed(format!(
                        "no KV page for row {r} at position {} ({} page(s) of {block_size})",
                        anchor_pos + r,
                        kv_blocks.len()
                    ))
                })?,
            );
        }
        Ok(TreeStepTables {
            live_rows: live,
            ids,
            words,
            positions,
            slots,
            seq_len: (anchor_pos as usize + form_rows) as i32,
            tsrc,
            hsrc: [window[0], window[1], window[2], 0],
            acc: rec.prologue.map_or(-1, |p| i32::from(p.row)),
            ts_in: rec.prologue.map_or(0, |p| p.parity),
            ts_out: rec.out_parity,
        })
    }
}

/// The index-table suffixes a tree step OVERRIDES in its blob (P27-15d): the tree's own tables and
/// the two row tables whose law is the chain's (`pos` -- rope BY DEPTH -- and `slot`, `seqused_k`,
/// equal to the law's by construction and written from the plan anyway). Every other key of the
/// tree plan is the shared law's (`saber_exec::forward::index_table_for`).
pub const TREE_TABLE_SUFFIXES: [&str; 8] = [
    "anc",
    "tsrc",
    "hsrc",
    "acc",
    "tslot",
    "pos",
    "slot",
    "seqused_k",
];

/// **P27-15d: one of a tree step's own index tables**, in the bytes the device form reads, or
/// `None` for a suffix the shared law owns. `tslot` is the sequence's ONE committed recurrent
/// slot (after a chain step or a prefill: `state_slot + naccept - 1`; after a tree step: the same
/// slot, advanced in place). Text-only rope (`pos` is `[3, rows]` i64 of the depth positions).
#[must_use]
pub fn tree_index_table(suffix: &str, t: &TreeStepTables, tslot: u32) -> Option<Vec<u8>> {
    let i32s = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let i64s = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    Some(match suffix {
        "anc" => t.words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        "tsrc" => i32s(&t.tsrc.concat()),
        "hsrc" => i32s(&t.hsrc),
        "acc" => i32s(&[t.acc]),
        "tslot" => i32s(&[i32::try_from(tslot).unwrap_or(0)]),
        "pos" => {
            let p: Vec<i64> = t.positions.iter().map(|x| i64::from(*x)).collect();
            i64s(&[p.clone(), p.clone(), p].concat())
        }
        "slot" => i64s(&t.slots),
        "seqused_k" => i32s(&[t.seq_len]),
        _ => return None,
    })
}

/// What a tree verify's verdict moves, for the step after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeCommit {
    /// The verdict (accepted path rows, committed tokens, accepted row).
    pub verdict: TreeVerdict,
    /// `p27_tree_kv_compact`'s `moves` `[n][2]`: (row slot, depth slot), absolute, in order.
    pub kv_moves: Vec<[i32; 2]>,
    /// `saber_row_gather_bf16`'s rows: the drafter's `aux.cat` rows in position order.
    pub aux_rows: Vec<i32>,
    /// The NEXT step's committed conv window (cache entries after this step's rewrite: the old
    /// window's element `j` -> entry `j - 1`, verify row `r` -> entry `2 + r`).
    pub next_window: [i32; CONV_WINDOW],
    /// The advance the next verify's prologue (or a flush) applies.
    pub advance: PendingAdvance,
    /// Resident tokens after the commit (`anchor + 1 + accepted depth`; the correction token is
    /// the next step's anchor and is not yet resident).
    pub resident_after: u32,
}

/// The commit of `plan`'s verdict `v` (the verify ran with `tables` at `anchor_pos`, writing the
/// record parity `ts_out`).
///
/// # Errors
/// A move outside the block table.
pub fn commit(
    v: &TreeVerdict,
    tables: &TreeStepTables,
    anchor_pos: u32,
    kv_blocks: &[u32],
    block_size: u32,
) -> Result<TreeCommit, TreeError> {
    let mut kv_moves = Vec::new();
    for (from, to) in v.kv_moves() {
        let s = |row: usize| {
            kv_slot(kv_blocks, block_size, anchor_pos + row as u32)
                .and_then(|x| i32::try_from(x).ok())
                .ok_or_else(|| TreeError::Malformed(format!("no KV slot for row {row}")))
        };
        kv_moves.push([s(from)?, s(to)?]);
    }
    let mut next_window = [0i32; CONV_WINDOW];
    for (j, s) in v.conv_commit(CONV_TAPS).iter().enumerate() {
        next_window[j] = match *s {
            ConvSrc::Tail(e) => e as i32 - 1,
            ConvSrc::Row(r) => CONV_WINDOW as i32 - 1 + r as i32,
        };
    }
    Ok(TreeCommit {
        kv_moves,
        aux_rows: v.aux_rows().iter().map(|r| *r as i32).collect(),
        next_window,
        advance: PendingAdvance {
            parity: tables.ts_out,
            row: v.accepted_row as u8,
        },
        resident_after: anchor_pos + 1 + v.depth() as u32,
        verdict: v.clone(),
    })
}

/// **THE VERDICT IN PLACE OF `accept_prefix`**, in the scheduler's convention
/// ([`crate::draft::Greedy`]'s: `accepted` = drafted tokens accepted, `committed` = the NEW
/// tokens, `accepted + 1` of them). `rows` = the target's token at every verify row, row order.
///
/// # Errors
/// Fewer rows than the plan verifies, or no rows.
pub fn tree_accept(
    plan: &TreePlan,
    rows: &[TokenId],
) -> Result<(Verdict, TreeVerdict), AcceptError> {
    if rows.is_empty() {
        return Err(AcceptError::Empty);
    }
    if rows.len() < plan.rows() {
        return Err(AcceptError::Rows {
            draft: plan.rows() - 1,
            want: plan.rows(),
            rows: rows.len(),
        });
    }
    let v = plan.verdict(|r| i64::from(rows[r]));
    let committed: Vec<TokenId> = v.committed.iter().map(|t| *t as TokenId).collect();
    if committed.len() != v.depth() + 1 {
        return Err(AcceptError::Law(format!(
            "the tree verdict committed {} token(s) at depth {}",
            committed.len(),
            v.depth()
        )));
    }
    Ok((
        Verdict {
            accepted: v.depth() as u32,
            committed,
        },
        v,
    ))
}

/// **The plans from the proposal's lattice read** (the P27-12 batch: `sel.scores` f32
/// `[live][steps][k][k]` then `sel.cand64` i64 `[live][steps][k]`, little-endian), one a live
/// request, each under its anchor.
#[must_use]
pub fn plans_from_read(
    form: SpecTreeForm,
    anchors: &[i64],
    scores: &[u8],
    cand: &[u8],
    steps: usize,
    k: usize,
) -> Vec<Result<TreePlan, TreeError>> {
    let (ls, lc) = (steps * k * k * 4, steps * k * 8);
    anchors
        .iter()
        .enumerate()
        .map(|(r, &anchor)| {
            let (Some(sb), Some(cb)) = (
                scores.get(r * ls..(r + 1) * ls),
                cand.get(r * lc..(r + 1) * lc),
            ) else {
                return Err(TreeError::Malformed(format!(
                    "request {r}: the lattice read holds {} / {} bytes",
                    scores.len(),
                    cand.len()
                )));
            };
            let s: Vec<f32> = sb
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let c: Vec<i64> = cb
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap_or_default()))
                .collect();
            TreePlan::from_lattice(
                form,
                anchor,
                &LatticeView {
                    cand: &c,
                    scores: &s,
                    steps,
                    k,
                },
            )
        })
        .collect()
}

/// Native lattice retained from the most recent proposal. In the one-sequence B16
/// profile the scheduler may ask for exactly the first nine of the fifteen tokens.
/// Bind that shorter view to the sequence, absolute cursor and anchor that generated it.
#[derive(Debug)]
pub(crate) struct NativeTreeOffer {
    pub(crate) owner: Option<(saber_sched::SeqId, u32, TokenId)>,
    pub(crate) chain: Vec<TokenId>,
    pub(crate) plan: TreePlan,
}

impl NativeTreeOffer {
    pub(crate) fn for_step(
        self,
        seq: saber_sched::SeqId,
        position: u32,
        anchor: TokenId,
        scheduled: &[TokenId],
        block16: bool,
    ) -> Option<TreePlan> {
        let bound = if block16 {
            self.owner == Some((seq, position, anchor))
                && self.chain.len() == 15
                && scheduled.len() + 1 == self.plan.rows()
                && self.chain.starts_with(scheduled)
        } else {
            self.chain == scheduled
        };
        bound.then(|| self.plan.with_anchor(i64::from(anchor)))
    }
}

/// The lattice plans' counters (`spectree plans ...`): plans built, nodes, the deepest node a
/// plan, the chain's walk found as a root path (its depth), refusals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCounts {
    /// Plans built.
    pub plans: u64,
    /// Nodes over them.
    pub nodes: u64,
    /// Sum of each plan's deepest node.
    pub max_depth: u64,
    /// Sum over plans of how deep the walk's chain runs inside the tree (its prefix that is a
    /// root path of the tree).
    pub chain_in_tree: u64,
    /// Lattices the plan refused.
    pub errors: u64,
}

impl PlanCounts {
    /// Count one plan against the chain the walk proposed (`block` = its 7 tokens).
    pub fn record(&mut self, plan: &TreePlan, block: &[TokenId]) {
        self.plans += 1;
        self.nodes += (plan.rows() - 1) as u64;
        self.max_depth += (1..plan.rows())
            .map(|r| u64::from(plan.depth(r)))
            .max()
            .unwrap_or(0);
        // walk the chain down the tree: at each depth the child of the current row with its token
        let mut at = 0usize;
        let mut d = 0u64;
        for t in block {
            let next = (1..plan.rows())
                .find(|&r| plan.parent_row(r) == Some(at) && plan.tokens()[r] == i64::from(*t));
            match next {
                Some(r) => {
                    at = r;
                    d += 1;
                }
                None => break,
            }
        }
        self.chain_in_tree += d;
    }

    /// The counts since `earlier` (a report's delta).
    #[must_use]
    pub fn since(&self, earlier: &PlanCounts) -> PlanCounts {
        PlanCounts {
            plans: self.plans - earlier.plans,
            nodes: self.nodes - earlier.nodes,
            max_depth: self.max_depth - earlier.max_depth,
            chain_in_tree: self.chain_in_tree - earlier.chain_in_tree,
            errors: self.errors - earlier.errors,
        }
    }

    /// The end-of-run line.
    #[must_use]
    pub fn line(&self) -> String {
        let per = |x: u64| {
            if self.plans == 0 {
                0.0
            } else {
                x as f64 / self.plans as f64
            }
        };
        format!(
            "spectree  plans {} nodes/plan {:.2} max depth/plan {:.2} walk's chain inside the \
             tree {:.2} deep, refused {}",
            self.plans,
            per(self.nodes),
            per(self.max_depth),
            per(self.chain_in_tree),
            self.errors
        )
    }
}

/// **P27-15e: a CHAIN as a tree plan** -- the scheduler's own draft under `anchor` (row `r` =
/// depth `r`), for a b1 step the lattice gave no plan (or whose draft is not the walk's 7, or a
/// sampled slot): the tree form verifies it with chain words, so its K/V needs no compaction and
/// its aux rows are already contiguous.
///
/// # Errors
/// A draft longer than the drafter's positions.
pub fn draft_chain_plan(anchor: TokenId, draft: &[TokenId]) -> Result<TreePlan, TreeError> {
    draft_chain_plan_limited(anchor, draft, crate::spec_tree::DRAFT_POSITIONS)
}

/// A chain fallback bounded by both the active drafter and the verify tile.
pub(crate) fn draft_chain_plan_limited(
    anchor: TokenId,
    draft: &[TokenId],
    max_depth: u32,
) -> Result<TreePlan, TreeError> {
    let cands: Vec<u32> = (0..draft.len() as u32).collect();
    let tree = crate::spec_tree::SpecTree::chain_limited(&cands, max_depth)?;
    TreePlan::new(tree, i64::from(anchor), |n| {
        draft.get(n.depth as usize - 1).map_or(0, |t| i64::from(*t))
    })
}

/// **P27-15e: the accepted path from the step's own TABLES** (both ranks: the ancestor words and
/// the row ids the blob carried, the merged argmax of every live row) -- the follower's half of
/// the verdict, since rank 1 holds no plan. Row `r`'s parent is the highest proper ancestor bit of
/// its word; the walk takes, from the current row, the child whose token is the current row's
/// argmax (best-first never gives one parent two children of one token). Returns the path rows,
/// the anchor first; == `TreePlan::verdict`'s `path_rows` (tested).
#[must_use]
pub fn path_from_tables(words: &[u16], ids: &[i64], argmax: &[i64]) -> Vec<usize> {
    let live = words.len().min(ids.len()).min(argmax.len());
    let parent = |r: usize| -> Option<usize> {
        let w = u32::from(words[r]) & !(1u32 << r);
        (w != 0).then(|| 31 - w.leading_zeros() as usize)
    };
    let mut path = vec![0usize];
    let mut at = 0usize;
    while let Some(c) = (1..live).find(|&c| parent(c) == Some(at) && ids[c] == argmax[at]) {
        path.push(c);
        at = c;
    }
    path
}

/// The fixed move count of the compaction launch (`n_moves`): at most the drafter's 7 depths,
/// padded with self-moves to the form's 9 nodes.
pub const COMPACT_MOVES: usize = TREE_ROWS - 1;

/// **P27-15e: the K/V compaction of a tree step's verdict**, derived from that step's TABLES
/// (`words`, the row `slots`) and the accepted row `acc` the NEXT tree step's `acc` table names
/// (both ranks hold all three): the accepted path's row slots -> its depth slots, in order
/// (`TreeVerdict::kv_moves` in absolute slots), padded with self-moves on the anchor's slot to
/// [`COMPACT_MOVES`]. Returns the moves and how many are real.
#[must_use]
pub fn compaction_moves(words: &[u16], slots: &[i64], acc: usize) -> (Vec<[i32; 2]>, usize) {
    compaction_moves_padded(words, slots, acc, COMPACT_MOVES)
}

/// [`compaction_moves`] padded to `pad` moves (E4-3: `rows - 1` = 15 under `best16`).
#[must_use]
pub fn compaction_moves_padded(
    words: &[u16],
    slots: &[i64],
    acc: usize,
    pad: usize,
) -> (Vec<[i32; 2]>, usize) {
    let s = |r: usize| {
        slots
            .get(r)
            .and_then(|x| i32::try_from(*x).ok())
            .unwrap_or(0)
    };
    let path: Vec<usize> = match words.get(acc) {
        Some(w) => (0..16).filter(|b| (u32::from(*w) >> b) & 1 == 1).collect(),
        None => vec![0],
    };
    let mut moves: Vec<[i32; 2]> = path
        .iter()
        .enumerate()
        .filter(|(d, r)| *d != **r)
        .map(|(d, r)| [s(*r), s(d)])
        .collect();
    let real = moves.len();
    moves.resize(pad.max(real), [s(0), s(0)]);
    (moves, real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::{Acceptance, Greedy};
    use crate::spec_tree::{AdvanceLaw, Node, SpecTree};

    fn ctx(form_built: bool) -> ArmContext {
        ArmContext {
            tp: 2,
            drafter: true,
            draft_block: 8,
            selector_reranks: false,
            form_built,
            max_seqs: 1,
        }
    }

    #[test]
    fn off_runs_off_and_ancestor9_falls_back_by_name_until_the_form_is_built() {
        let r = resolve_arm(SpecTreeForm::Off, ctx(false));
        assert_eq!(r.runs, SpecTreeForm::Off);
        assert!(!r.fell_back() && !r.plans);
        let r = resolve_arm(SpecTreeForm::Ancestor9, ctx(false));
        assert_eq!(
            r.runs,
            SpecTreeForm::Off,
            "a binary with no tree verify form"
        );
        assert!(r.fell_back() && r.plans);
        assert!(r.line().contains("REFUSED") && r.line().contains("not built"));
        const { assert!(TREE_VERIFY_FORM_BUILT, "P27-15e: the form is built") };
        // P27-W2-4 (R2): a boot that admits more than one sequence runs `ancestor9` (the flush);
        // the forms without one stay on the chain, by name
        let mut c = ctx(true);
        c.max_seqs = 8;
        let r = resolve_arm(SpecTreeForm::Ancestor9, c);
        assert_eq!(r.runs, SpecTreeForm::Ancestor9);
        let r = resolve_arm(SpecTreeForm::Best16, c);
        assert_eq!(r.runs, SpecTreeForm::Off);
        assert!(r.plans && r.refused.expect("refused").contains("--max-seqs 8"));
        let r = resolve_arm(SpecTreeForm::Ancestor9, ctx(true));
        assert_eq!(r.runs, SpecTreeForm::Ancestor9);
        assert!(!r.fell_back());
        // P27-15f: the two-record control runs as itself, under the same boot law
        let r = resolve_arm(SpecTreeForm::Ancestor9Pair, ctx(true));
        assert_eq!(r.runs, SpecTreeForm::Ancestor9Pair);
        assert!(r.runs.is_tree() && !SpecTreeForm::Off.is_tree());
        assert_eq!(
            SpecTreeForm::parse(SpecTreeForm::Ancestor9Pair.as_str()),
            Ok(SpecTreeForm::Ancestor9Pair)
        );
        let mut c = ctx(true);
        c.max_seqs = 8;
        assert_eq!(
            resolve_arm(SpecTreeForm::Ancestor9Pair, c).runs,
            SpecTreeForm::Off
        );
        assert_eq!(
            record_ledger_bytes(SpecTreeForm::Ancestor9Pair, 48, 24),
            record_ledger_bytes(SpecTreeForm::Ancestor9, 48, 24)
        );
        // the refusals by name
        let mut c = ctx(true);
        c.drafter = false;
        assert!(resolve_arm(SpecTreeForm::Ancestor9, c)
            .refused
            .expect("refused")
            .contains("drafter"));
        let mut c = ctx(true);
        c.selector_reranks = true;
        assert!(resolve_arm(SpecTreeForm::Ancestor9, c)
            .refused
            .expect("refused")
            .contains("explen"));
        let mut c = ctx(true);
        c.tp = 1;
        assert!(resolve_arm(SpecTreeForm::Ancestor9, c)
            .refused
            .expect("refused")
            .contains("TP=1"));
    }

    #[test]
    fn native_block16_uses_all_fifteen_selector_positions_on_best16() {
        let mut wide = ctx(true);
        wide.draft_block = 16;
        let arm = resolve_arm(SpecTreeForm::Best16, wide);
        assert_eq!(arm.runs, SpecTreeForm::Best16);
        assert!(arm.plans);
        assert_eq!(
            resolve_arm(SpecTreeForm::Ancestor9, wide).runs,
            SpecTreeForm::Ancestor9
        );
        assert!(resolve_arm(SpecTreeForm::Ancestor9Pair, wide).fell_back());
        wide.max_seqs = 2;
        assert!(resolve_arm(SpecTreeForm::Best16, wide).fell_back());

        let (steps, k) = (15usize, 16usize);
        let mut scores = vec![-64.0f32; steps * k * k];
        let mut expected = vec![900i64];
        let mut lane = 0;
        for position in 0..steps {
            for prev in 0..k {
                let best = (prev + position + 1) % k;
                scores[(position * k + prev) * k + best] = 0.0;
            }
            lane = (lane + position + 1) % k;
            expected.push((position * k + lane) as i64);
        }
        let scores: Vec<u8> = scores.iter().flat_map(|x| x.to_le_bytes()).collect();
        let candidates: Vec<u8> = (0..(steps * k) as i64).flat_map(i64::to_le_bytes).collect();
        let plans = plans_from_read(SpecTreeForm::Best16, &[900], &scores, &candidates, steps, k);
        let plan = plans[0].as_ref().expect("a full native selector lattice");
        assert_eq!(plan.tokens(), &expected);
        assert_eq!(plan.depth(15), 15);
        assert_eq!(plan.words()[15], u16::MAX);

        let draft: Vec<u32> = expected[1..].iter().map(|id| *id as u32).collect();
        assert_eq!(
            draft_chain_plan_limited(900, &draft, 15).unwrap().words(),
            plan.words()
        );
        assert!(draft_chain_plan(900, &draft).is_err());
    }

    #[test]
    fn block16_ancestor9_binds_only_its_own_truncated_mailbox_chain() {
        use crate::draft::{Drafter, EngineDrafter};
        use saber_sched::SeqId;
        let seq = SeqId(27);
        let (k, steps) = (16usize, 15usize);
        let mut scores = vec![-64f32; steps * k * k];
        for row in scores.chunks_exact_mut(k) {
            row[0] = 0.;
        }
        let scores: Vec<u8> = scores.iter().flat_map(|x| x.to_le_bytes()).collect();
        let candidates: Vec<u8> = (0..(k * steps) as i64).flat_map(i64::to_le_bytes).collect();
        let plan = plans_from_read(
            SpecTreeForm::Ancestor9,
            &[0],
            &scores,
            &candidates,
            steps,
            k,
        )
        .pop()
        .unwrap()
        .unwrap();
        assert_eq!(plan.rows(), 10);
        assert_eq!(plan.words()[9], 0x03ff);
        let chain: Vec<TokenId> = (0..15).map(|i| i * 16).collect();
        let offer = || NativeTreeOffer {
            owner: Some((seq, 128, 41)),
            chain: chain.clone(),
            plan: plan.clone(),
        };
        let mut mailbox = EngineDrafter::new(9);
        mailbox.offer(seq, &chain);
        let scheduled = mailbox.propose(seq, 41, 9);
        let tree = offer().for_step(seq, 128, 41, &scheduled, true).unwrap();
        assert_eq!(tree.tokens()[0], 41);
        assert_eq!(
            &tree.tokens()[1..],
            &scheduled.iter().map(|t| *t as i64).collect::<Vec<_>>()
        );
        assert!(offer()
            .for_step(SeqId(28), 128, 41, &scheduled, true)
            .is_none());
        assert!(offer().for_step(seq, 129, 41, &scheduled, true).is_none());
        assert!(offer().for_step(seq, 128, 42, &scheduled, true).is_none());
        assert!(offer()
            .for_step(seq, 128, 41, &scheduled[..8], true)
            .is_none());
        let mut substituted = scheduled.clone();
        substituted[8] += 1;
        assert!(offer().for_step(seq, 128, 41, &substituted, true).is_none());
        assert!(draft_chain_plan_limited(41, &chain[..10], 9).is_err());
    }

    #[test]
    fn the_record_pair_is_one_pair_a_rank_and_nothing_when_off() {
        // the 27B at TP=2: 48 GDN layers x 24 v heads a rank
        assert_eq!(
            record_ledger_bytes(SpecTreeForm::Ancestor9, 48, 24),
            23_887_872
        );
        assert_eq!(record_ledger_bytes(SpecTreeForm::Off, 48, 24), 0);
        // vs P27-10's feared 576 MiB of snapshot slots
        assert!(record_ledger_bytes(SpecTreeForm::Ancestor9, 48, 24) < 576 << 20);
    }

    #[test]
    fn a_foreign_pending_advance_is_flushed_before_a_tree_verify() {
        let mut law = AdvanceLaw::new(4);
        let (f, r0) = flush_foreign(&mut law, 4, 1);
        assert!(f.is_empty() && r0.prologue.is_none());
        law.settle(1, r0, 3);
        // slot 1 again: its own prologue, the other parity
        let (f, r1) = flush_foreign(&mut law, 4, 1);
        assert!(f.is_empty());
        assert_eq!(
            r1.prologue.map(|p| (p.parity, p.row)),
            Some((r0.out_parity, 3))
        );
        assert_eq!(r1.out_parity, r0.out_parity ^ 1);
        law.settle(1, r1, 2);
        // slot 2 now: slot 1's pending is flushed first, the one pair is free
        let (f, r2) = flush_foreign(&mut law, 4, 2);
        assert_eq!(
            f,
            vec![(
                1,
                PendingAdvance {
                    parity: r1.out_parity,
                    row: 2
                }
            )]
        );
        assert!(r2.prologue.is_none());
        assert_eq!(law.flushes, 1);
        assert_eq!(law.prologues, 1);
    }

    fn chain_plan(k: usize) -> TreePlan {
        let t = SpecTree::chain(&vec![0; k]).expect("chain");
        TreePlan::new(t, 7, |n| 100 + i64::from(n.depth)).expect("plan")
    }

    #[test]
    fn a_chains_tables_are_the_shipped_chain_law() {
        // the chain of 7 at 8 rows: causal words, positions anchor + r, the sliding conv window
        // from `naccept - 1` (conv_gate.py's CHAIN law: `tsrc_of(.., sliding=True)`).
        let plan = chain_plan(7);
        let blocks = [5u32, 9, 2];
        let rec = AdvanceLaw::new(1).verify(0);
        let t = TreeStepTables::new(&plan, 13, &blocks, 16, chain_window(3), rec).expect("tables");
        assert_eq!(t.live_rows, 8);
        for r in 0..8 {
            assert_eq!(t.words[r], ((2u32 << r) - 1) as u16);
            assert_eq!(t.positions[r], 13 + r as i32);
            let seq: Vec<i32> = [0, 1, 2]
                .into_iter()
                .chain((0..=r as i32).map(|c| 3 + c))
                .collect();
            assert_eq!(
                &t.tsrc[r][..3],
                &seq[seq.len() - 4..seq.len() - 1],
                "row {r}"
            );
        }
        assert_eq!(t.hsrc, [2, 3, 4, 0]);
        // row slots across the page boundary at position 16
        assert_eq!(t.slots[0], 5 * 16 + 13);
        assert_eq!(t.slots[3], 9 * 16);
        assert_eq!(t.seq_len, 23);
        assert_eq!(t.acc, -1);
        // padding rows 8, 9: the anchor's token, a depth-1 word, a free slot
        assert_eq!(t.ids[8], 7);
        assert_eq!(t.words[9], 1 | (1 << 9));
        // a chain's verdict never moves KV and its next window is the chain law's
        let rows: Vec<TokenId> = vec![101, 102, 103, 999, 0, 0, 0, 0, 0, 0];
        let (sv, tv) = tree_accept(&plan, &rows).expect("verdict");
        assert_eq!(sv.accepted, 3);
        let c = commit(&tv, &t, 13, &blocks, 16).expect("commit");
        assert!(c.kv_moves.is_empty());
        assert_eq!(c.next_window, chain_window(4));
        assert_eq!(c.aux_rows, vec![0, 1, 2, 3]);
        assert_eq!(c.resident_after, 17);
    }

    /// P27-15d: the tree step's OWN tables for the deepest CHAIN plan (7 nodes, the lattice's
    /// depth: 8 live rows of the form's 10) are byte for byte the shared law's chain tables for a
    /// b1 10-row step at the same anchor, pages and acceptance -- every overridden key (`anc`,
    /// `tsrc`, `pos` on the 8 live rows; `hsrc`, `acc`, `tslot`, `slot`, `seqused_k` whole; the
    /// two padding rows are depth-1 anchor children by the form's law); and a BRANCHED plan differs from the law exactly where a tree must
    /// (the words, the taps, the depth positions), never in its row slots.
    #[test]
    fn p27_15d_a_chain_trees_tables_are_the_shared_laws() {
        use saber_exec::forward::{index_table_for, Accepted, AttnQLen, SeqFacts, StepIndex};
        let pages = [5u32, 9, 2];
        let (anchor, naccept, base) = (13u32, 4u32, 9u32);
        let seqs = [SeqFacts {
            state_slot: base,
            done_before: anchor,
            kv_blocks: &pages,
            live_rows: TREE_ROWS as u32,
            rope: None,
        }];
        let ids = vec![0i32; TREE_ROWS];
        let law = StepIndex {
            phase: saber_model::graph::Phase::Decode,
            rows: TREE_ROWS as u32,
            max_blocks: 4,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        let acc = [naccept];
        let table = |sfx: &str| {
            index_table_for(
                &format!("x.{sfx}"),
                &law,
                Accepted::PerSeq(&acc),
                AttnQLen::Live,
            )
            .expect(sfx)
        };
        let plan = chain_plan(7);
        let rec = AdvanceLaw::new(1).verify(0);
        let t = TreeStepTables::new(&plan, anchor, &pages, 16, chain_window(naccept), rec)
            .expect("tables");
        assert_eq!(t.live_rows, 8);
        let tslot = base + naccept - 1;
        // the live rows' prefix of a row table (`pos` is [3, rows]: each axis's prefix)
        let live = |sfx: &str, b: &[u8]| -> Vec<u8> {
            match sfx {
                "anc" => b[..8 * 2].to_vec(),
                "tsrc" => b[..8 * 16].to_vec(),
                "pos" => (0..3)
                    .flat_map(|a| b[a * TREE_ROWS * 8..a * TREE_ROWS * 8 + 8 * 8].to_vec())
                    .collect(),
                _ => b.to_vec(),
            }
        };
        for sfx in TREE_TABLE_SUFFIXES {
            let ours = tree_index_table(sfx, &t, tslot).expect(sfx);
            let law = table(sfx);
            assert_eq!(ours.len(), law.len(), "`{sfx}`: the blob slot's width");
            assert_eq!(
                live(sfx, &ours),
                live(sfx, &law),
                "`{sfx}`: the chain tree's table is the law's"
            );
        }
        assert_eq!(tree_index_table("cu", &t, tslot), None, "the law owns `cu`");
        // A branched plan: root -> a(d1) -> b(d2); root -> c(d1) -- rows 0 a b c.
        let tree = SpecTree::new(vec![
            Node {
                depth: 1,
                cand: 0,
                parent: None,
            },
            Node {
                depth: 2,
                cand: 0,
                parent: Some(0),
            },
            Node {
                depth: 1,
                cand: 1,
                parent: None,
            },
        ])
        .expect("tree");
        let plan = TreePlan::new(tree, 7, |n| 100 + i64::from(n.cand)).expect("plan");
        let b = TreeStepTables::new(&plan, anchor, &pages, 16, chain_window(naccept), rec)
            .expect("tables");
        assert_ne!(tree_index_table("anc", &b, tslot), Some(table("anc")));
        assert_ne!(tree_index_table("tsrc", &b, tslot), Some(table("tsrc")));
        assert_ne!(tree_index_table("pos", &b, tslot), Some(table("pos")));
        for sfx in ["slot", "seqused_k", "hsrc", "acc", "tslot"] {
            assert_eq!(
                tree_index_table(sfx, &b, tslot),
                Some(table(sfx)),
                "`{sfx}`"
            );
        }
        // row 3 (c, depth 1) sits at the anchor + 1 and reads the window's last two + the anchor
        assert_eq!(b.positions[3], anchor as i32 + 1);
        assert_eq!(b.tsrc[3], [1, 2, 3, 0]);
        assert_eq!(b.words[3], 0b1001);
    }

    #[test]
    fn the_tree_verdict_is_accept_prefix_on_every_chain() {
        // exhaustive over a small alphabet: any chain plan + any argmax rows -> Greedy's verdict
        for k in 1..=4usize {
            let plan = chain_plan(k);
            let draft: Vec<TokenId> = (1..=k as u32).map(|d| 100 + d).collect();
            let n = k + 1;
            let alphabet = [101u32, 102, 103, 104, 7];
            let total = alphabet.len().pow(n as u32);
            for code in 0..total {
                let mut c = code;
                let rows: Vec<TokenId> = (0..n)
                    .map(|_| {
                        let t = alphabet[c % alphabet.len()];
                        c /= alphabet.len();
                        t
                    })
                    .collect();
                let want = Greedy.accept(&draft, &rows).expect("greedy");
                let (got, _) = tree_accept(&plan, &rows).expect("tree");
                assert_eq!(got, want, "k {k} rows {rows:?}");
            }
        }
    }

    #[test]
    fn a_branched_verdict_moves_its_path_and_names_the_next_window() {
        // root -> a(d1) -> b(d2); root -> c(d1) -> d(d2): rows 0 a b c d
        let t = SpecTree::new(vec![
            Node {
                depth: 1,
                cand: 0,
                parent: None,
            },
            Node {
                depth: 2,
                cand: 0,
                parent: Some(0),
            },
            Node {
                depth: 1,
                cand: 1,
                parent: None,
            },
            Node {
                depth: 2,
                cand: 1,
                parent: Some(2),
            },
        ])
        .expect("tree");
        let toks = [0i64, 10, 11, 20, 21];
        let plan = TreePlan::new(t, 5, |n| match (n.depth, n.cand) {
            (1, 0) => 10,
            (2, 0) => 11,
            (1, 1) => 20,
            _ => 21,
        })
        .expect("plan");
        assert_eq!(plan.tokens()[1..], toks[1..]);
        let rec = AdvanceLaw::new(1).verify(0);
        let blocks = [3u32, 4];
        let tb = TreeStepTables::new(&plan, 14, &blocks, 16, [0, 1, 2], rec).expect("tables");
        // rope by depth: siblings share a position
        assert_eq!(&tb.positions[..5], &[14, 15, 16, 15, 16]);
        // row 4's window: tail 2, the anchor's row, row 3
        assert_eq!(tb.tsrc[4], [2, 3, 6, 0]);
        // the target says c then 99: path rows 0, 3, 4 accepted to depth 2
        let rows: Vec<TokenId> = vec![20, 0, 0, 21, 99, 0, 0, 0, 0, 0];
        let (sv, tv) = tree_accept(&plan, &rows).expect("verdict");
        assert_eq!(sv.committed, vec![20, 21, 99]);
        assert_eq!(tv.accepted_row, 4);
        let c = commit(&tv, &tb, 14, &blocks, 16).expect("commit");
        // rows 3, 4 -> depths 1, 2; positions 17, 18 (page 4) -> 15 (page 3), 16 (page 4)
        assert_eq!(
            c.kv_moves,
            vec![[4 * 16 + 1, 3 * 16 + 15], [4 * 16 + 2, 4 * 16]]
        );
        // next window: rows 0, 3, 4 -> entries 2, 5, 6
        assert_eq!(c.next_window, [2, 5, 6]);
        assert_eq!(
            c.advance,
            PendingAdvance {
                parity: rec.out_parity,
                row: 4
            }
        );
    }

    #[test]
    fn plans_from_the_read_are_the_plans_from_the_view() {
        let (steps, k) = (7usize, 16usize);
        let mut scores = Vec::new();
        let mut cand = Vec::new();
        for r in 0..2u32 {
            for i in 0..steps * k * k {
                let x = ((i as u32)
                    .wrapping_mul(2_654_435_761u32)
                    .wrapping_add(r * 97)
                    % 1000) as f32
                    / 97.0;
                scores.extend_from_slice(&x.to_le_bytes());
            }
            for i in 0..steps * k {
                cand.extend_from_slice(&(1000 * i64::from(r) + i as i64).to_le_bytes());
            }
        }
        let got = plans_from_read(SpecTreeForm::Ancestor9, &[5, 6], &scores, &cand, steps, k);
        assert_eq!(got.len(), 2);
        for (r, p) in got.iter().enumerate() {
            let p = p.as_ref().expect("plan");
            assert_eq!(p.rows(), TREE_ROWS);
            assert_eq!(p.tokens()[0], 5 + r as i64);
            let mut pc = PlanCounts::default();
            pc.record(p, &[]);
            assert_eq!((pc.plans, pc.nodes), (1, 9));
        }
        // short reads are refused by name, never read past
        let short = plans_from_read(
            SpecTreeForm::Ancestor9,
            &[5, 6, 7],
            &scores,
            &cand,
            steps,
            k,
        );
        assert!(short[2].is_err());
    }

    #[test]
    fn p27_15e_the_tables_verdict_and_moves_are_the_plans() {
        // a branched plan: two children of the anchor, a chain of two under the second, and a
        // child of the first
        let nodes = vec![
            Node {
                depth: 1,
                cand: 0,
                parent: None,
            },
            Node {
                depth: 1,
                cand: 1,
                parent: None,
            },
            Node {
                depth: 2,
                cand: 0,
                parent: Some(1),
            },
            Node {
                depth: 3,
                cand: 0,
                parent: Some(2),
            },
            Node {
                depth: 2,
                cand: 1,
                parent: Some(0),
            },
        ];
        let tree = SpecTree::new(nodes).expect("tree");
        let plan = TreePlan::new(tree, 7, |n| match (n.depth, n.cand) {
            (1, 0) => 10,
            (1, 1) => 20,
            (2, 0) => 30,
            (3, 0) => 40,
            _ => 50,
        })
        .expect("plan");
        let slots: Vec<i64> = (0..TREE_ROWS as i64).map(|r| 1000 + r).collect();
        // every argmax pattern over a small alphabet: the tables' walk == the plan's verdict
        let alphabet = [10i64, 20, 30, 40, 50, 99];
        let rows = plan.rows();
        let mut n = 0usize;
        let mut idx = vec![0usize; rows];
        let mut branched = 0usize;
        'all: loop {
            let am: Vec<i64> = idx.iter().map(|i| alphabet[*i]).collect();
            let v = plan.verdict(|r| am[r]);
            let p = path_from_tables(plan.words(), plan.tokens(), &am);
            assert_eq!(p, v.path_rows, "argmax {am:?}");
            let (moves, real) = compaction_moves(plan.words(), &slots, v.accepted_row);
            let want: Vec<[i32; 2]> = v
                .kv_moves()
                .iter()
                .map(|(f, t)| [1000 + *f as i32, 1000 + *t as i32])
                .collect();
            assert_eq!(&moves[..real], &want[..]);
            branched += usize::from(real > 0);
            assert_eq!(moves.len(), COMPACT_MOVES);
            assert!(moves[real..].iter().all(|m| *m == [1000, 1000]));
            n += 1;
            for i in idx.iter_mut() {
                *i += 1;
                if *i < alphabet.len() {
                    continue 'all;
                }
                *i = 0;
            }
            break;
        }
        assert_eq!(n, alphabet.len().pow(rows as u32));
        assert!(branched > 0, "some verdicts move K/V");
        // a chain plan of the scheduler's draft: chain words, no moves, the Greedy law
        let c = draft_chain_plan(5, &[6, 7, 8]).expect("chain");
        assert_eq!(c.words(), &[1, 3, 7, 15]);
        assert_eq!(c.tokens(), &[5, 6, 7, 8]);
        let v = c.verdict(|r| [6i64, 7, 9, 0][r]);
        assert_eq!(v.path_rows, vec![0, 1, 2]);
        assert_eq!(compaction_moves(c.words(), &slots, v.accepted_row).1, 0);
        // control: the row order read as a chain misses a branched verdict
        let am = [20i64, 99, 30, 40, 99, 99];
        assert_ne!(
            path_from_tables(&[1, 3, 7, 15, 31, 63], plan.tokens(), &am),
            plan.verdict(|r| am[r]).path_rows
        );
    }
}
