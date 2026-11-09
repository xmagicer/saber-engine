//! **C9/C10: the FUSED mixed step** — prompt chunks and a decode half as ONE forward
//! (`--engine-mixed-prefill fused` carries one chunk, `packed` several; the law is
//! `saber_exec::mixed`, the host maps `saber_sched::mixed_law`, the design `docs/EXEC.md` § the
//! mixed step).
//!
//! The chunks' rows are laid out per prompt in queue order from row 0 of the prefill rung `R`
//! (chunk `k` at rows `r_k..r_k + c_k`, `r_0 = 0`); the decode bucket's `D` rows follow the last
//! chunk (`C..C + D`, `C = Σ c_k`). `R` is the smallest planned rung holding `C + D` in every
//! chunk's own class (`FusedRows::of_in_class` for one chunk, `mixed_law::packed_in_class` for
//! several). The row-local ops run once at `R`; rope + KV append run once over every row with a
//! combined `pos` / `slot` table. The prefill plan's attention / GDN chunk path runs over each
//! chunk in turn — chunk 0 as the plan is bound (retargeted at its rows and slot), each further
//! chunk as the same steps with their row-carrying arena bindings SHIFTED to its rows
//! (`saber_exec::mixed::shift_rows`; the chunk path's private scratch is reused at the plan's
//! offsets, the groups run one after another), its attention retargeted at its rows, its GDN
//! state at its slot and its index tables from its own blob set (`saber_exec::mixed::packed_key`)
//! — and then the decode plan's attention / GDN update over the decode rows (bindings rebound
//! onto the prefill arena at row `C`). Each sampling chunk's head samples its last row with the
//! one-row prefill head (the first inside the body's submission, the others each a head of its
//! own after it; with two or more, each chunk's dense logit row is kept on the host for the
//! executor's dense fallback); the decode head samples rows `C..C + D`; the aux tap copies all
//! `R` rows, so each chunk's context store reads `aux.cat` rows `r_k..r_k + c_k` and the decode
//! proposal rows `C..`.
//!
//! **What stays eager, and why**: the whole fused step. A prefill chunk is never captured (its
//! attention and GDN bindings are retargeted per chunk — `q_len`, the state slot), and a fused
//! step is a chunk; a captured form would need rung x decode-bucket graphs with per-step
//! retargets inside them (LEDGER C10 prices it). The decode half's own graphs are untouched: a
//! decode step without a chunk replays them exactly as before.
//!
//! **When it does not fuse** (one chunk: the serial form runs in the same call, chunk first,
//! counted as a fallback; several: the executor asks [`fused_pick`] first and runs them serially
//! itself): no rung holds the rows in class, a ROW BAND decode half, the in-step proposal order
//! (the fused step only defers), the store check or the drafter probe, a chunk with image rows,
//! the `--no-live-rows` control, or an `aux.cat` too short for the proposal's window at row `C`.

use super::*;

use crate::forward::MixedChunk;
use saber_exec::layer::{Source, StateAt};
use saber_exec::mixed::{
    is_chunk_side, layer_of, merge_rows, packed_key, parse_packed_key, shift_rows, FusedBinding,
    FusedOp, KeySide,
};
use saber_exec::Route;
use saber_model::graph::KernelOp;

/// The top of the 256-thread producer class: a rung at or above it moves a lower rung's chunk
/// (`quant_large_min_m`, C5/C8's p2048 split).
const LARGE_MIN_M: u32 = saber_sched::mixed_law::LARGE_MIN_M;

/// Most chunks one packed step carries (the blob holds this many chunk-side table sets).
pub(super) const MAX_CHUNKS: usize = saber_sched::mixed_law::MAX_PACKED_CHUNKS as usize;

/// Which plans a fused step runs: `(prefill rung plan, decode plan, rung rows)`, or `None` for
/// the serial form.
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_pick(
    plans: &[Planned<'_>],
    sessions: &BTreeMap<SeqId, Session>,
    chunks: &[MixedChunk],
    batch: u32,
    width: u32,
    defer: bool,
    live_row_attn: bool,
    tap: Option<&AuxTap>,
    drafter: Option<&DraftResident>,
    draft_counts: &DraftCounters,
) -> Option<(usize, usize, u32)> {
    if !defer
        || !live_row_attn
        || draft_counts.store_check.load(Ordering::Relaxed)
        || draft_counts.probe.load(Ordering::Relaxed)
        || tap.is_some() != drafter.is_some()
        || chunks.is_empty()
        || chunks.len() > MAX_CHUNKS
    {
        return None;
    }
    let mut rows = Vec::with_capacity(chunks.len());
    for ch in chunks {
        let session = sessions.get(&ch.seq)?;
        if session.images.is_some() {
            return None;
        }
        let c = u32::try_from(ch.tokens.len()).ok()?;
        if c == 0 {
            return None;
        }
        rows.push(c);
    }
    let c: u32 = rows.iter().sum();
    let d_rows = batch * width;
    let dec = plans.iter().position(|p| {
        p.phase == Phase::Decode && !p.ragged && p.rows == d_rows && p.width == width
    })?;
    let mut rungs: Vec<u32> = plans
        .iter()
        .filter(|p| p.phase == Phase::Prefill)
        .map(|p| p.rows)
        .collect();
    rungs.sort_unstable();
    let rung = if rows.len() == 1 {
        let f = saber_sched::mixed_law::FusedRows::of_in_class(c, d_rows, &rungs, LARGE_MIN_M)?;
        if !f.holds() {
            return None;
        }
        f.rung
    } else {
        saber_sched::mixed_law::packed_in_class(&rows, d_rows, &rungs, LARGE_MIN_M)?
    };
    if let (Some(t), Some(res)) = (tap, drafter) {
        let widest = res
            .shapes
            .iter()
            .map(|s| s.plan.shape.ctx_rows)
            .max()
            .unwrap_or(0);
        if c + widest.max(d_rows) > t.rows || !t.descs.contains_key(&rung) {
            return None;
        }
    }
    let pre = plans
        .iter()
        .position(|p| p.phase == Phase::Prefill && p.rows == rung)?;
    Some((pre, dec, rung))
}

/// One chunk's index view in a fused step: its step (chunk 0's at the rung's rows over the
/// combined ids, a further chunk's at its own rows), the `q_len` its attention carries, and its
/// first row in the rung.
pub(super) struct ChunkIndex<'a> {
    pub idx: StepIndex<'a>,
    pub attn: AttnQLen,
    pub row0: u32,
}

/// The fused step's index tables, into the blob: each key from the step it belongs to
/// (`saber_exec::mixed::key_side`) — chunk 0's for the prefill plan's per-sequence ops and every
/// row-local table (the ids are the combined rows), a further chunk `k`'s for its own set
/// (`packed_key(k, ..)`), the decode half's for the decode-side ops, and the shared rope /
/// append's `pos` / `slot` merged row by row (every chunk at its rows, the decode rows at
/// `decode_row0`). Every block table is rewritten, so the paged region always moves and the next
/// step re-stages its own.
pub(super) fn stage_indices_fused(
    indices: &mut Indices,
    chunks: &[ChunkIndex<'_>],
    decode: &StepIndex<'_>,
    decode_row0: u32,
) -> Result<(), ForwardError> {
    let Some(first) = chunks.first() else {
        return Err(ForwardError::Device(
            "a fused step with no chunk".to_owned(),
        ));
    };
    for (k, c) in chunks.iter().enumerate() {
        c.idx
            .check()
            .map_err(|e| ForwardError::Device(format!("the fused step's chunk {k}: {e}")))?;
    }
    decode
        .check()
        .map_err(|e| ForwardError::Device(format!("the fused step's decode half: {e}")))?;
    indices.paged_digest = None;
    let rung_rows = first.idx.rows;
    if let Some(c) = indices.dedup_check.as_mut() {
        c.begin_step();
    }
    let Indices {
        table,
        host,
        runs,
        paged_recipe,
        record_recipe,
        tree_only,
        tree_skipped,
        dedup,
        dedup_check,
        ..
    } = indices;
    // CMP27AG `slotdedup`: each shared slot written once (its keys are one (side, suffix) group).
    let mut wrote: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    let mut built: BTreeMap<(u8, &str), Vec<u8>> = BTreeMap::new();
    // P27-14 (`table`): this staging's paged writes (a fused step always writes the table).
    let mut recipe = record_recipe.then(crate::host_levers::PagedRecipe::default);
    let mut recipe_ix: BTreeMap<(u8, String), u32> = BTreeMap::new();
    runs.clear();
    let err = |e: index::IndexError| ForwardError::Device(e.to_string());
    let merr = |e: saber_exec::Error| ForwardError::Device(e.to_string());
    for (i, (key, offset, cap)) in table.iter().enumerate() {
        if index::step_invariant(key) {
            continue;
        }
        // CMP27AF `treeblob`: a fused step is never a tree step; the tree-only keys (and their
        // own page digest) are left as the last tree step wrote them.
        if tree_only.get(i).copied().unwrap_or(false) {
            *tree_skipped += 1;
            continue;
        }
        let (tag, suffix) = if let Some((k, suffix)) = parse_packed_key(key) {
            // A further chunk's set: written only when the step carries chunk `k`.
            if k == 0 || k as usize >= chunks.len() {
                continue;
            }
            (16 + k as u8, suffix)
        } else {
            let suffix = key.rsplit('.').next().unwrap_or(key.as_str());
            (saber_exec::mixed::key_side(key) as u8, suffix)
        };
        if let std::collections::btree_map::Entry::Vacant(slot) = built.entry((tag, suffix)) {
            let bytes = if tag >= 16 {
                let c = &chunks[usize::from(tag - 16)];
                index_table_for(key, &c.idx, c.attn).map_err(err)?
            } else {
                match saber_exec::mixed::key_side(key) {
                    KeySide::Chunk => index_table_for(key, &first.idx, first.attn).map_err(err)?,
                    KeySide::Decode => index_table_for(key, decode, AttnQLen::Live).map_err(err)?,
                    KeySide::PerRow => {
                        let mut t = index_table_for(key, &first.idx, first.attn).map_err(err)?;
                        for c in &chunks[1..] {
                            let tk = index_table_for(key, &c.idx, c.attn).map_err(err)?;
                            t = merge_rows(&t, rung_rows, &tk, c.idx.rows, c.row0, 8)
                                .map_err(merr)?;
                        }
                        let d = index_table_for(key, decode, AttnQLen::Live).map_err(err)?;
                        merge_rows(&t, rung_rows, &d, decode.rows, decode_row0, 8).map_err(merr)?
                    }
                }
            };
            slot.insert(bytes);
        }
        let bytes = &built[&(tag, suffix)];
        if bytes.len() > *cap {
            return Err(ForwardError::Device(format!(
                "index `{key}` is {} bytes and its slot is {cap}",
                bytes.len()
            )));
        }
        if let Some(c) = dedup_check.as_mut().filter(|c| c.active) {
            // The bytes this key's private slot would hold: the (side, suffix) build is the
            // control's own law here (a per-row merge has no per-key form), noted per key.
            c.note(i, bytes);
        }
        if *dedup && !wrote.insert(*offset) {
            continue;
        }
        let at = *offset as usize;
        host.as_mut_slice()[at..at + bytes.len()].copy_from_slice(bytes);
        if let (Some(r), true) = (recipe.as_mut(), index::paged_table(key)) {
            let ix = *recipe_ix
                .entry((tag, suffix.to_owned()))
                .or_insert_with(|| {
                    r.tables.push(bytes.clone());
                    u32::try_from(r.tables.len() - 1).unwrap_or(u32::MAX)
                });
            r.spans.push((u32::try_from(at).unwrap_or(u32::MAX), ix));
        }
        if !index::paged_table(key) {
            runs.push((
                u32::try_from(at).unwrap_or(u32::MAX),
                u32::try_from(bytes.len()).unwrap_or(u32::MAX),
            ));
        }
    }
    *paged_recipe = recipe;
    if let Some(c) = dedup_check.as_mut() {
        c.verify("stage_indices_fused", table, host.as_slice());
    }
    Ok(())
}

/// The fused binding plans, built once per (prefill rung, decode bucket) pair and reused.
pub(super) type FusedCache = BTreeMap<(usize, usize), FusedBinding>;

/// The pair's [`FusedBinding`], built on first use (its ranges checked at the largest chunk the
/// pair can carry: the rung minus the decode rows).
pub(super) fn binding_for<'c>(
    cache: &'c mut FusedCache,
    plans: &[Planned<'_>],
    pre: usize,
    dec: usize,
) -> Result<&'c FusedBinding, ForwardError> {
    if let std::collections::btree_map::Entry::Vacant(v) = cache.entry((pre, dec)) {
        let max_row0 = u64::from(plans[pre].rows.saturating_sub(plans[dec].rows));
        v.insert(
            FusedBinding::new(&plans[pre].plan, &plans[dec].plan, max_row0)
                .map_err(device_err("the fused binding plan"))?,
        );
    }
    Ok(&cache[&(pre, dec)])
}

/// **C10: a packed step's further chunk** (`k >= 1`): its rows, its GDN slot, and its own
/// attention descriptors (the prefill plan's, retargeted at its live rows), by prefill step.
pub(super) struct ExtraChunk {
    pub k: u32,
    pub row0: u32,
    pub rows: u32,
    pub slot: u32,
    pub attn: BTreeMap<usize, Vec<LaunchDescriptor>>,
}

/// The further chunks `(row0, rows, slot)` of a packed step, with their attention descriptors.
pub(super) fn extra_chunks(
    pre: &Planned<'_>,
    list: &[(u32, u32, u32)],
    saber: &SaberRules,
) -> Result<Vec<ExtraChunk>, ForwardError> {
    let q = pre.plan.bucket.q_len();
    let mut out = Vec::with_capacity(list.len());
    for (i, &(row0, rows, slot)) in list.iter().enumerate() {
        if slot == 0 {
            return Err(ForwardError::Device(format!(
                "packed chunk {}: GDN slot 0 is the kernels' null slot",
                i + 1
            )));
        }
        let mut attn = BTreeMap::new();
        for (j, s) in pre.plan.steps.iter().enumerate() {
            if s.op.kernel != KernelOp::AttnPrefillPaged {
                continue;
            }
            let Route::Kernel { descs, .. } = &s.route else {
                continue;
            };
            let mut d = descs.clone();
            for x in &mut d {
                saber_kernels::saber::retarget_attn_prefill(x, rows, q, saber)
                    .map_err(|e| ForwardError::Device(format!("{}: {e}", s.name())))?;
            }
            attn.insert(j, d);
        }
        out.push(ExtraChunk {
            k: i as u32 + 1,
            row0,
            rows,
            slot,
            attn,
        });
    }
    Ok(out)
}

/// One chunk-side prefill step bound for a further chunk: row-carrying arena bindings shifted to
/// its rows, the chunk path's private scratch at the plan's offsets, its index tables from its
/// blob set (a step-invariant table stays chunk 0's), its GDN state at its slot.
fn bind_extra<'a>(
    pre: &'a Planned<'_>,
    j: usize,
    e: &'a ExtraChunk,
    fb: &FusedBinding,
    inputs: &BlockInputs,
) -> Result<BoundStep<'a>, ForwardError> {
    let step = &pre.plan.steps[j];
    let Route::Kernel { descs, shape } = &step.route else {
        return Err(ForwardError::Device(format!(
            "the chunk-side `{}` launches nothing",
            step.name()
        )));
    };
    let descs: &'a [LaunchDescriptor] = e.attn.get(&j).map_or(descs.as_slice(), Vec::as_slice);
    let mut args = ArgBindings::new(*shape);
    let bind_err = device_err("binding a packed chunk");
    for (name, src) in &step.bindings {
        let addr = match src {
            Source::Arena {
                buffer,
                offset,
                bytes,
            } if !fb.chunk_private.contains(buffer) => {
                let at = shift_rows(
                    &pre.plan,
                    buffer,
                    *offset,
                    u64::from(e.row0),
                    u64::from(e.rows),
                )
                .map_err(&bind_err)?;
                pre.exec
                    .address(
                        &Source::Arena {
                            buffer: buffer.clone(),
                            offset: at,
                            bytes: *bytes,
                        },
                        inputs,
                    )
                    .map_err(&bind_err)?
            }
            Source::Index(key) if !index::step_invariant(key) => {
                let suffix = key.rsplit('.').next().unwrap_or(key.as_str());
                let pk = packed_key(e.k, suffix);
                *inputs.indices.get(&pk).ok_or_else(|| {
                    ForwardError::Device(format!("no packed index table `{pk}` in the blob"))
                })?
            }
            Source::State {
                key,
                at: StateAt::Slot { slot, offset },
            } => {
                let per = offset / u64::from((*slot).max(1));
                pre.exec
                    .address(
                        &Source::State {
                            key: key.clone(),
                            at: StateAt::Slot {
                                slot: e.slot,
                                offset: u64::from(e.slot) * per,
                            },
                        },
                        inputs,
                    )
                    .map_err(&bind_err)?
            }
            other => pre.exec.address(other, inputs).map_err(&bind_err)?,
        };
        args.bind(name, addr);
    }
    Ok(BoundStep {
        name: step.name(),
        descs,
        args,
    })
}

/// The fused body's bound launches, in the pair's [`FusedBinding`] order: the prefill plan's
/// steps on its own arena, each layer's chunk group followed by the further chunks' groups
/// (`extras`, in order), the decode-side steps rebound onto it at row `row0` (a decode-private
/// scratch buffer stays on the decode arena), with the aux tap's copies spliced in after the ops
/// they follow.
pub(super) fn bind_fused<'a>(
    pre: &'a Planned<'_>,
    dec: &'a Planned<'_>,
    fb: &FusedBinding,
    inputs: &BlockInputs,
    row0: u32,
    tap: Option<&'a AuxTap>,
    extras: &'a [ExtraChunk],
) -> Result<Vec<BoundStep<'a>>, ForwardError> {
    if u64::from(row0) > fb.max_row0 {
        return Err(ForwardError::Device(format!(
            "fused chunks of {row0} row(s) past the pair's checked {}",
            fb.max_row0
        )));
    }
    let pbase = pre.exec.arena_base();
    let dbase = dec.exec.arena_base();
    let mut out: Vec<BoundStep<'a>> =
        Vec::with_capacity(fb.ops.len() * (1 + usize::from(!extras.is_empty())));
    let mut group: Vec<usize> = Vec::new();
    for (pos, op) in fb.ops.iter().enumerate() {
        let (step, rebound) = match *op {
            FusedOp::Chunk(i) => (&pre.plan.steps[i], None),
            FusedOp::Decode(i) => (&dec.plan.steps[i], fb.rebound.get(&i)),
        };
        if let Route::Kernel { descs, shape } = &step.route {
            let mut args = ArgBindings::new(*shape);
            for (j, (name, src)) in step.bindings.iter().enumerate() {
                let addr = match rebound {
                    None => pre
                        .exec
                        .address(src, inputs)
                        .map_err(device_err("binding the fused body"))?,
                    Some(r) => match r.get(j).copied().flatten() {
                        Some((on_prefill, at0, per)) => {
                            let base = if on_prefill { pbase } else { dbase };
                            base.offset(at0 + u64::from(row0) * per).ok_or_else(|| {
                                ForwardError::Device(format!("`{name}` rebound past its arena"))
                            })?
                        }
                        None => dec
                            .exec
                            .address(src, inputs)
                            .map_err(device_err("binding a decode-side op"))?,
                    },
                };
                args.bind(name, addr);
            }
            out.push(BoundStep {
                name: step.name(),
                descs,
                args,
            });
        }
        // C10: after the LAST step of a layer's chunk group, the further chunks' groups.
        if let FusedOp::Chunk(i) = *op {
            let name = pre.plan.steps[i].name();
            if !extras.is_empty() && is_chunk_side(name) {
                group.push(i);
                let closes = match fb.ops.get(pos + 1) {
                    Some(FusedOp::Chunk(n)) => {
                        let nx = pre.plan.steps[*n].name();
                        !(is_chunk_side(nx) && layer_of(nx) == layer_of(name))
                    }
                    _ => true,
                };
                if closes {
                    for e in extras {
                        for &j in &group {
                            out.push(bind_extra(pre, j, e, fb, inputs)?);
                        }
                    }
                    group.clear();
                }
            }
        }
    }
    match tap {
        Some(t) => insert_aux_taps(out, pre, t, crate::aux_tap::TapAt::Point),
        None => Ok(out),
    }
}

/// Dispatches the prefill plan's chunk-side steps issue for ONE chunk (what each further chunk
/// adds to a packed step's budget).
fn chunk_side_launches(plan: &BlockPlan<'_>) -> u32 {
    plan.steps
        .iter()
        .filter(|s| is_chunk_side(s.name()))
        .map(|s| s.route.launches())
        .sum()
}

/// The per-slot facts, ids and `cu` starts of a decode half, checked as `decode_step` checks them.
type DecodeFacts = (
    Vec<(
        u32,
        u32,
        Vec<u32>,
        Option<std::sync::Arc<saber_exec::mrope::MropePositions>>,
    )>,
    Vec<TokenId>,
    Vec<usize>,
);

fn decode_facts(
    slots: &[DecodeSlot],
    sessions: &BTreeMap<SeqId, Session>,
    width: u32,
    rows: u32,
    geom: Geom,
) -> Result<DecodeFacts, ForwardError> {
    let mut facts = Vec::with_capacity(slots.len());
    for s in slots {
        if s.rows() != width {
            return Err(ForwardError::Device(format!(
                "{}: {} row(s) in a width-{width} bucket",
                s.seq,
                s.rows()
            )));
        }
        let session = sessions.get(&s.seq).ok_or(ForwardError::Unknown(s.seq))?;
        if session.done != s.context_len {
            return Err(ForwardError::Device(format!(
                "{}: the batch says {} token(s) are resident and the engine has {}",
                s.seq, s.context_len, session.done
            )));
        }
        check_room(
            s.seq,
            session.done + s.rows().max(1) - 1,
            geom.max_ctx,
            geom.max_blocks,
            geom.block_size,
        )?;
        if let Some(im) = &session.images {
            if session.done < im.rope.image_end() {
                return Err(ForwardError::Device(format!(
                    "{}: a decode row inside the prompt's image rows",
                    s.seq
                )));
            }
        }
        facts.push((
            session.slot,
            session.done,
            session.kv_blocks.clone(),
            session.images.as_ref().map(|i| i.rope.clone()),
        ));
    }
    let mut ids: Vec<TokenId> = slots.iter().flat_map(DecodeSlot::row_ids).collect();
    ids.resize(rows as usize, 0);
    let starts: Vec<usize> = slots
        .iter()
        .scan(0usize, |at, s| {
            let here = *at;
            *at += s.rows() as usize;
            Some(here)
        })
        .collect();
    Ok((facts, ids, starts))
}

/// A chunk's context store at the fused step (leader), prod's deferred order only: the tiles
/// over `aux.cat` rows `aux0..aux0 + c`, their frames, the `End` marker, and the first proposal's
/// inputs when this is the prompt's last chunk (`None` otherwise) — `prefill_step`'s law.
#[allow(clippy::too_many_arguments)]
fn chunk_store(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    res: &DraftResident,
    t: &AuxTap,
    tier: &crate::draft_tier::DraftTier,
    link: &mut Option<&mut RankLink>,
    seq: SeqId,
    done: u32,
    live_rows: u32,
    aux0: u32,
    anchor: Option<TokenId>,
    last: bool,
    max_ctx: u32,
    vocab_total: u32,
    draft_counts: &DraftCounters,
    trace: &HostTrace,
) -> Result<Option<PendingProposal>, ForwardError> {
    use crate::draft_inputs::CtxFrame;
    let block = res.cfg.block_size;
    let wire = res.ctx_wire();
    let geom_d = crate::draft_inputs::DraftGeometry {
        block,
        page: res.kv_block,
        max_reqs: res.shape(res.store)?.plan.shape.reqs,
        max_pos: max_ctx,
        mask_token: crate::draft_inputs::MASK_TOKEN,
    };
    let narrow = draft_counts.store_narrow.load(Ordering::Relaxed);
    let natural_off = draft_counts.store_natural_off.load(Ordering::Relaxed);
    let t_sched = trace.start();
    let sched = match tier.session(seq) {
        None => Err(ForwardError::Device(format!(
            "{seq}: a prefilling request with no drafter session"
        ))),
        Some(s) => crate::draft_inputs::DraftCtxTile::schedule(
            done,
            live_rows,
            aux0,
            &s.kv_blocks,
            geom_d,
            s.rope.as_deref(),
            anchor.is_some(),
            res.tiling(t.rows, narrow, natural_off),
        ),
    };
    trace.stop(HostPhase::Store, t_sched);
    let end = crate::draft_inputs::DraftCtxTile::marker(CtxFrame::End, block, wire);
    let sched = match sched {
        Ok(s) => s,
        Err(e) => {
            draft_counts.refused.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(seq = %seq, error = %e, "the fused chunk's context tables were refused");
            if let Some(l) = link.as_mut() {
                l.swap_draft(&end)?;
            }
            return Ok(None);
        }
    };
    let mut stage_at = 0usize;
    for (i, tile) in sched.stored.iter().enumerate() {
        let kind = match (tile.width() == block, sched.chunk_class, sched.natural_wide) {
            (true, true, _) => CtxFrame::TileChunkClass,
            (false, _, true) => CtxFrame::TileWideNatural,
            _ => CtxFrame::Tile,
        };
        let tf = trace.start();
        if let Some(l) = link.as_mut() {
            l.swap_draft(&tile.pack_wire(kind, wire)?)?;
        }
        trace.stop(HostPhase::StoreFrame, tf);
        let ts = trace.start();
        store_ctx_tile(
            hip,
            ctx,
            launcher,
            stream,
            res,
            t,
            tile,
            draft_counts,
            kind,
            &mut stage_at,
            trace.race(),
        )
        .map_err(|e| {
            let _ = wait_bounded_local(hip, stream, "a failed context store");
            ForwardError::Device(format!("the fused chunk's context tile {i}: {e}"))
        })?;
        trace.stop(HostPhase::Store, ts);
    }
    if !sched.stored.is_empty() {
        finish_ctx_store(hip, stream, res, trace, &mut stage_at)?;
    }
    if let Some(l) = link.as_mut() {
        l.swap_draft(&end)?;
    }
    match (anchor, sched.held.as_ref()) {
        // The held tile's `row0` is its `aux.cat` row (`aux0` + its row in the chunk); its first
        // position is the chunk's `done` + its row in the chunk.
        (Some(_), Some(tail)) => Ok(Some(PendingProposal::Prefill {
            seq,
            done_before: done + (tail.row0 - aux0),
            committed: tail.live,
            row0: tail.row0,
            vocab: vocab_total,
            // P27-3: a fused mixed step's chunk stores on path (the arm is the unmixed chunk's).
            store: None,
        })),
        _ => {
            if last {
                draft_counts.prefill_silent.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None)
        }
    }
}

/// What a fused step hands back: the decode half's outcome, each chunk's (in order), and — for a
/// packed step whose heads sampled two or more chunks — each sampled chunk's dense logit shard,
/// for the executor's dense fallback (the prefill head's arena holds only the last one's).
pub(super) struct FusedOut {
    pub decode: StepOutcome,
    pub chunks: Vec<StepOutcome>,
    pub stash: Vec<Option<Vec<f32>>>,
}

/// Whether chunk `k`'s head runs the sampler's seam, and with penalties.
fn chunk_seam(ch: &MixedChunk, head: &Head<'_>) -> (bool, bool) {
    let seam = ch.last && ch.sampling.wants_row() && head.seam.is_some();
    (seam, seam && !ch.sampling.penalty_ids.is_empty())
}

/// **The fused mixed step, on the leader.** See the module doc; `pick` is [`fused_pick`]'s.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn fused_step(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    plans: &mut [Planned<'_>],
    prefill_head: &mut Head<'_>,
    heads: &[Head<'_>],
    indices: &mut Indices,
    token_plane: &mut Tokens,
    inputs: &BlockInputs,
    sessions: &mut BTreeMap<SeqId, Session>,
    geom: Geom,
    link: Option<&mut RankLink>,
    gathers: &AtomicU64,
    law: &dyn Acceptance,
    trace: &HostTrace,
    per_row: bool,
    retargets: &AtomicU64,
    saber: &SaberRules,
    pick: (usize, usize, u32),
    cache: &mut FusedCache,
    chunks: &[MixedChunk],
    slots: &[DecodeSlot],
    batch: u32,
    width: u32,
    tap: Option<&AuxTap>,
    drafter: Option<(&DraftResident, &crate::draft_tier::DraftTier)>,
    prefill_proposal: bool,
    draft_counts: &DraftCounters,
    slot_counts: &SlotCounters,
    stash: &mut Option<PendingProposal>,
) -> Result<FusedOut, ForwardError> {
    let (pre_ix, dec_ix, rung) = pick;
    let kn = chunks.len();
    let rows: Vec<u32> = chunks.iter().map(|c| c.tokens.len() as u32).collect();
    let row0s: Vec<u32> = rows
        .iter()
        .scan(0u32, |at, r| {
            let here = *at;
            *at += r;
            Some(here)
        })
        .collect();
    let c: u32 = rows.iter().sum();
    let d_rows = batch * width;
    if slots.is_empty()
        || slots.len() as u32 > batch
        || kn == 0
        || kn > MAX_CHUNKS
        || rows.contains(&0)
        || c + d_rows > rung
    {
        return Err(ForwardError::Device(format!(
            "a fused step of {kn} chunk(s) of {rows:?} row(s) and {} slot(s) in b{batch}s{width} \
             at p{rung}",
            slots.len()
        )));
    }
    let t_plan = trace.start();
    // ---- each chunk's facts (prefill_step's) ------------------------------------------------
    let mut cfacts: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(kn);
    for (ch, r) in chunks.iter().zip(&rows) {
        let s = sessions.get(&ch.seq).ok_or(ForwardError::Unknown(ch.seq))?;
        super::replay_prefill_ok(indices, ch.seq, s)?;
        if s.done != ch.done_before {
            return Err(ForwardError::Device(format!(
                "{}: the chunk says {} token(s) are resident and the engine has {}",
                ch.seq, ch.done_before, s.done
            )));
        }
        check_room(
            ch.seq,
            s.done + r - 1,
            geom.max_ctx,
            geom.max_blocks,
            geom.block_size,
        )?;
        cfacts.push((s.slot, s.done, s.kv_blocks.clone()));
    }
    // ---- the decode half's facts (decode_step's) -------------------------------------------
    let (facts, dids, starts) = decode_facts(slots, sessions, width, d_rows, geom)?;
    let mut dseqs: Vec<SeqFacts<'_>> = facts
        .iter()
        .zip(slots)
        .map(|((slot, done, blocks, rope), s)| SeqFacts {
            state_slot: *slot,
            done_before: *done,
            kv_blocks: blocks,
            live_rows: s.rows(),
            rope: rope.as_deref(),
        })
        .collect();
    dseqs.resize(batch as usize, SeqFacts::padding());
    let mut committed_before: Vec<u32> = slots.iter().map(|s| s.committed_before).collect();
    committed_before.resize(batch as usize, 1);
    let dindex = StepIndex {
        phase: Phase::Decode,
        rows: d_rows,
        max_blocks: geom.max_blocks,
        seqs: &dseqs,
        ids: &dids,
        committed_before: &committed_before,
        band_width: None,
    };
    let live_rows_total: usize = slots.iter().map(|s| s.rows() as usize).sum();
    // ---- the combined ids: every chunk's rows, then the decode rows, then zeros ----------------
    let mut ids: Vec<TokenId> = chunks
        .iter()
        .flat_map(|ch| ch.tokens.iter().copied())
        .collect();
    ids.extend_from_slice(&dids);
    ids.resize(rung as usize, 0);
    let cseqs: Vec<[SeqFacts<'_>; 1]> = cfacts
        .iter()
        .zip(&rows)
        .map(|((slot, done, blocks), r)| {
            [SeqFacts {
                state_slot: *slot,
                done_before: *done,
                kv_blocks: blocks,
                live_rows: *r,
                rope: None,
            }]
        })
        .collect();
    // Check the actual cursors and tables used by the paged append, before either
    // rank submits GPU work. Padding rows have no owning sequence or KV pages.
    let live_seqs = cseqs.iter().map(|s| &s[0]).chain(&dseqs[..slots.len()]);
    let accesses = live_seqs.map(|s| {
        (
            saber_sched::mixed_law::RowState {
                kv_blocks: s.kv_blocks,
                gdn_set: s.state_slot,
            },
            s.done_before,
        )
    });
    saber_sched::mixed_law::isolated_at_cursors(accesses, saber_exec::forward::KV_BLOCK)
        .map_err(|e| ForwardError::Device(format!("the fused step is not isolated: {e}")))?;
    // ---- the prefill rung, retargeted at chunk 0 (prefill_step's) ----------------------------
    let moved = plans[pre_ix]
        .plan
        .retarget_prefill_rows(rows[0], saber)
        .map_err(|e| ForwardError::Device(e.to_string()))?;
    if moved > 0 && rows[0] < rung {
        retargets.fetch_add(1, Ordering::Relaxed);
    }
    let state_slot = cfacts[0].0;
    if state_slot == 0 {
        return Err(ForwardError::Device(format!(
            "{}: a fused chunk with no live sequence",
            chunks[0].seq
        )));
    }
    plans[pre_ix]
        .plan
        .retarget_prefill_state(state_slot)
        .map_err(|e| ForwardError::Device(e.to_string()))?;
    plans[pre_ix]
        .exec
        .zero_constants(hip, &plans[pre_ix].plan, stream)
        .map_err(device_err("re-seeding the shared prefill arena's zeros"))?;
    let extras = extra_chunks(
        &plans[pre_ix],
        &(1..kn)
            .map(|k| (row0s[k], rows[k], cfacts[k].0))
            .collect::<Vec<_>>(),
        saber,
    )?;
    trace.stop(HostPhase::ChunkPlan, t_plan);
    let t_index = trace.start();
    let cidx: Vec<ChunkIndex<'_>> = (0..kn)
        .map(|k| ChunkIndex {
            idx: StepIndex {
                phase: Phase::Prefill,
                rows: if k == 0 { rung } else { rows[k] },
                max_blocks: geom.max_blocks,
                seqs: &cseqs[k],
                ids: if k == 0 { &ids } else { &chunks[k].tokens },
                committed_before: &[],
                band_width: None,
            },
            attn: if k == 0 {
                AttnQLen::Exact(plans[pre_ix].plan.prefill_q_len())
            } else {
                AttnQLen::Exact(rows[k])
            },
            row0: row0s[k],
        })
        .collect();
    stage_indices_fused(indices, &cidx, &dindex, c)?;
    // P27-W2-5: the decode half's replay table (arm only).
    super::stage_replay(indices, sessions, slots, batch)?;
    stage_ids(token_plane, &ids)?;
    trace.stop(HostPhase::Index, t_index);
    let head_at = heads
        .iter()
        .position(|h| h.rows == d_rows)
        .ok_or(ForwardError::NoPlan {
            phase: Phase::Decode,
            rows: d_rows,
            detail: "no decode head for the fused step's decode half".to_owned(),
        })?;
    let dhead = &heads[head_at];
    // ---- the heads' inputs: the sampling chunks' rows and seams, the decode seam --------------
    let sampled: Vec<usize> = (0..kn).filter(|k| chunks[*k].last).collect();
    let seams: Vec<(bool, bool)> = chunks
        .iter()
        .map(|ch| chunk_seam(ch, prefill_head))
        .collect();
    let seam_in: Vec<SeamInputs> = chunks
        .iter()
        .map(|ch| {
            if ch.last && prefill_head.seam.is_some() {
                let (temps, pen) =
                    seam_inputs(prefill_head.rows, std::slice::from_ref(&ch.sampling));
                SeamInputs {
                    temps,
                    pen_ids: pen,
                }
            } else {
                SeamInputs::default()
            }
        })
        .collect();
    let write_chunk_head = |k: usize| -> Result<(), ForwardError> {
        let sel = ((row0s[k] + rows[k]) as i32 - 1).to_le_bytes();
        hip.memcpy_h2d_blocking(&prefill_head.sel, &sel, stream)
            .map_err(device_err("the row index"))?;
        if let Some(seam) = &prefill_head.seam {
            seam.write(hip, stream, &seam_in[k].temps, &seam_in[k].pen_ids, None)?;
        }
        Ok(())
    };
    if let Some(&k0) = sampled.first() {
        write_chunk_head(k0)?;
    }
    let dec_seam = dhead.seam.is_some() && slots.iter().any(|s| s.sampling.wants_row());
    let dec_pen = dec_seam && slots.iter().any(|s| !s.sampling.penalty_ids.is_empty());
    let mut seam_d = SeamInputs::default();
    if let Some(seam) = &dhead.seam {
        let per_row: Vec<RowRequest> = slots
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.sampling.clone(), s.rows() as usize))
            .collect();
        let (temps, pen) = seam_inputs(dhead.rows, &per_row);
        seam.write(hip, stream, &temps, &pen, None)?;
        seam_d = SeamInputs {
            temps,
            pen_ids: pen,
        };
    }
    // ---- the frames (the follower mirrors this order) ------------------------------------------
    let mask = |f: &dyn Fn(usize) -> bool| -> u8 {
        (0..kn).filter(|k| f(*k)).fold(0u8, |m, k| m | (1 << k))
    };
    let packed = if kn == 1 {
        0
    } else {
        crate::tp2::packed_word(
            kn as u32 - 1,
            mask(&|k| chunks[k].last),
            mask(&|k| seams[k].0),
            mask(&|k| seams[k].1),
        )
    };
    let mut link = link;
    if let Some(l) = link.as_deref_mut() {
        let t_lead = trace.start();
        indices.pack_frame(token_plane, rung as usize)?;
        l.lead(
            Command::Mixed {
                rung,
                chunk_live: rows[0],
                chunk_samples: chunks[0].last,
                slot: state_slot,
                decode_rows: d_rows,
                width,
                decode_live: live_rows_total as u32,
                paged: true,
                chunk_seam: seams[0].0,
                chunk_seam_penalties: seams[0].1,
                decode_seam: dec_seam,
                decode_seam_penalties: dec_pen,
                defer_proposals: true,
                packed,
            },
            indices.frame(),
        )?;
        trace.stop(HostPhase::Lead, t_lead);
        let t = trace.start();
        super::send_paged(l, indices)?;
        trace.stop(HostPhase::Paged, t);
        for &k in &sampled {
            if seams[k].0 {
                l.lead_seam(prefill_head.rows, &seam_in[k])?;
            }
        }
        if dec_seam {
            l.lead_seam(dhead.rows, &seam_d)?;
        }
    }
    // ---- the forward: body (+ the first sampling chunk's head), then each further sampling
    // chunk's head, then the decode head ----------------------------------------------------------
    let fb = binding_for(cache, plans, pre_ix, dec_ix)?;
    let first_seam = sampled.first().map_or((false, false), |k| seams[*k]);
    let (mut launches, out0, seam0, peer0) = run_fused_body(
        hip,
        ctx,
        launcher,
        stream,
        plans,
        (pre_ix, dec_ix),
        fb,
        prefill_head,
        indices,
        token_plane,
        inputs,
        c,
        rung,
        !sampled.is_empty(),
        tap,
        link.as_deref_mut(),
        gathers,
        trace,
        per_row,
        first_seam.0,
        first_seam.1,
        &extras,
    )?;
    type HeadOut = (Vec<TokenId>, Option<SeamOut>, Option<SeamOut>);
    let mut outs: Vec<HeadOut> = (0..kn).map(|_| (Vec::new(), None, None)).collect();
    let keep = sampled.len() >= 2;
    let mut stash_rows: Vec<Option<Vec<f32>>> = vec![None; kn];
    let mut first_out = Some((out0, seam0, peer0));
    for (n, &k) in sampled.iter().enumerate() {
        if n == 0 {
            if let Some(o) = first_out.take() {
                outs[k] = o;
            }
        } else {
            write_chunk_head(k)?;
            let (l, o, s, p) = run_chunk_head(
                hip,
                ctx,
                launcher,
                stream,
                &plans[pre_ix],
                prefill_head,
                token_plane,
                inputs,
                link.as_deref_mut(),
                gathers,
                trace,
                per_row,
                seams[k].0,
                seams[k].1,
            )?;
            launches += l;
            outs[k] = (o, s, p);
        }
        if keep {
            stash_rows[k] = Some(dense_shard_of(hip, stream, prefill_head, 0)?);
        }
    }
    let (dl, sampled_d, seam_out_d, peer_d) = run_decode_head(
        hip,
        ctx,
        launcher,
        stream,
        &plans[pre_ix],
        dhead,
        token_plane,
        inputs,
        c,
        live_rows_total,
        link.as_deref_mut(),
        gathers,
        trace,
        per_row,
        dec_seam,
        dec_pen,
    )?;
    let launches = launches + dl;
    // RULE 7 for a shape no plan declares: the prefill rung's body, each further chunk's
    // chunk-side steps, the decode-side ops, the aux tap's copies, every chunk head and the
    // decode head, counted from the plans, against what was issued.
    let decode_side: u32 = plans[dec_ix]
        .plan
        .steps
        .iter()
        .filter(|s| saber_exec::mixed::is_decode_side(s.name()))
        .map(|s| s.route.launches())
        .sum();
    let expected = plans[pre_ix].plan.expected_launches()
        + (kn as u32 - 1) * chunk_side_launches(&plans[pre_ix].plan)
        + decode_side
        + if tap.is_some() {
            crate::draft_inputs::AUX_TAPS.len() as u32
        } else {
            0
        }
        + sampled.len() as u32 * prefill_head.declared_launches()
        + dhead.declared_launches();
    if launches != expected {
        return Err(ForwardError::Device(format!(
            "the fused step p{rung} ({kn} chunk(s)) + b{batch}s{width} issued {launches} \
             dispatches and its plans declare {expected}"
        )));
    }
    // ---- the decode half's acceptance (decode_step's) ------------------------------------------
    let t_accept = trace.start();
    let (commits, _fed_back, proposals) = accept_slots(
        slots,
        &starts,
        &sampled_d,
        dhead,
        seam_out_d.as_ref(),
        peer_d.as_ref(),
        law,
        sessions,
        drafter.is_some(),
        draft_counts,
        slot_counts,
    )?;
    trace.stop(HostPhase::Accept, t_accept);
    let (peer_base, vocab_total) = prefill_head.shards();
    // ---- each chunk's context store, then every proposal stashed, the decode half's first ------
    let mut chunk_pending: Vec<Option<PendingProposal>> = (0..kn).map(|_| None).collect();
    if let (Some((res, tier)), Some(t)) = (drafter, tap) {
        for k in 0..kn {
            let anchor = if chunks[k].last && prefill_proposal {
                outs[k].0.first().copied()
            } else {
                None
            };
            chunk_pending[k] = chunk_store(
                hip,
                ctx,
                launcher,
                stream,
                res,
                t,
                tier,
                &mut link,
                chunks[k].seq,
                cfacts[k].1,
                rows[k],
                row0s[k],
                anchor,
                chunks[k].last,
                geom.max_ctx,
                vocab_total,
                draft_counts,
                trace,
            )?;
        }
    }
    let mut decode_pending: Vec<SeqId> = Vec::new();
    if drafter.is_some() && tap.is_some() {
        decode_pending = proposals.iter().map(|p| p.0).collect();
        *stash = Some(PendingProposal::Decode {
            reqs: proposals,
            width,
            ragged_rows: None,
            vocab: vocab_total,
            row0: c,
        });
    }
    let mut chunk_outs = Vec::with_capacity(kn);
    for (k, (ch, pending)) in chunks.iter().zip(chunk_pending).enumerate() {
        let pending_seqs = match pending {
            Some(p) => {
                push_pending(stash, p);
                vec![ch.seq]
            }
            None => Vec::new(),
        };
        // ---- the chunk's session (prefill_step's) ----------------------------------------------
        let session = sessions
            .get_mut(&ch.seq)
            .ok_or(ForwardError::Unknown(ch.seq))?;
        session.done += rows[k];
        session.committed = 1;
        let (out_k, seam_k, peer_k) = std::mem::take(&mut outs[k]);
        chunk_outs.push(StepOutcome {
            commits: out_k
                .into_iter()
                .map(|t| SlotCommit {
                    seq: ch.seq,
                    accepted: 0,
                    tokens: vec![t],
                    rows: vec![seam_k.as_ref().and_then(|o| {
                        prefill_head.seam.as_ref().and_then(|s| {
                            merge_row(
                                o,
                                peer_k.as_ref(),
                                0,
                                s,
                                peer_base,
                                vocab_total,
                                &ch.sampling,
                            )
                        })
                    })],
                })
                .collect(),
            rows: StepRows {
                phase: Phase::Prefill,
                rows: rung,
                width: 1,
                samples: ch.last,
                tree: false,
                ragged: false,
            },
            launches: 0,
            path: StepPath::Fused,
            feed: TokenFeed::Host,
            drafts: Vec::new(),
            proposal_pending: pending_seqs,
        });
    }
    let decode_out = StepOutcome {
        commits,
        rows: StepRows {
            phase: Phase::Decode,
            rows: d_rows,
            width,
            samples: true,
            tree: false,
            ragged: false,
        },
        launches,
        path: StepPath::Fused,
        feed: TokenFeed::Host,
        drafts: Vec::new(),
        proposal_pending: decode_pending,
    };
    Ok(FusedOut {
        decode: decode_out,
        chunks: chunk_outs,
        stash: stash_rows,
    })
}

/// A further pending proposal behind the ones already stashed, in the executor's commit order
/// (the decode half's, then each chunk's): the queue is right-nested
/// ([`PendingProposal::Queued`]'s `then`), so the collector takes `first` and the rest becomes
/// pending; a lone proposal is the stash itself.
pub(super) fn push_pending(stash: &mut Option<PendingProposal>, p: PendingProposal) {
    *stash = Some(match stash.take() {
        None => p,
        Some(PendingProposal::Queued { first, then }) => {
            let mut rest = Some(*then);
            push_pending(&mut rest, p);
            PendingProposal::Queued {
                first,
                then: Box::new(rest.expect("push_pending always leaves a stash")),
            }
        }
        Some(first) => PendingProposal::Queued {
            first: Box::new(first),
            then: Box::new(p),
        },
    });
}

/// The fused body's eager run (both ranks): the uploads, the body and — when a chunk samples —
/// the first sampling chunk's head (its row already in the head's `sel`) and its pick, and its
/// seam. Returns `(launches, the chunk's token(s), its seam, the peer's seam)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn run_fused_body(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    plans: &[Planned<'_>],
    (pre_ix, dec_ix): (usize, usize),
    fb: &FusedBinding,
    prefill_head: &Head<'_>,
    indices: &Indices,
    token_plane: &mut Tokens,
    inputs: &BlockInputs,
    c: u32,
    rung: u32,
    samples: bool,
    tap: Option<&AuxTap>,
    mut link: Option<&mut RankLink>,
    gathers: &AtomicU64,
    trace: &HostTrace,
    per_row: bool,
    seam_here: bool,
    seam_penalties: bool,
    extras: &[ExtraChunk],
) -> Result<(u32, Vec<TokenId>, Option<SeamOut>, Option<SeamOut>), ForwardError> {
    let t_blob = trace.start();
    let uploads = vec![
        index_upload(stream, indices, true)?,
        ids_upload(stream, token_plane, rung as usize * 4)?,
    ];
    hip.run_batch(&uploads)
        .map_err(device_err("uploading the fused step's tables and ids"))?;
    trace.stop(HostPhase::Blob, t_blob);
    let t_bind = trace.start();
    let pre = &plans[pre_ix];
    let mut steps = bind_fused(pre, &plans[dec_ix], fb, inputs, c, tap, extras)?;
    if samples {
        steps.extend(head_ops(
            &pre.plan,
            pre.exec.arena_base(),
            0,
            prefill_head,
            inputs,
            prefill_head.sample_to(token_plane),
        )?);
    }
    trace.stop(HostPhase::Bind, t_bind);
    {
        let mut clock = prefill_head.clock.borrow_mut();
        clock.key = None;
        clock.armed = None;
    }
    let t_submit = trace.start();
    let launches = run_bound(ctx, launcher, stream, &steps)?;
    trace.stop(HostPhase::Submit, t_submit);
    let mut toks = Vec::new();
    let mut mine = None;
    let mut peer = None;
    if samples {
        toks = sample(
            hip,
            stream,
            prefill_head,
            token_plane,
            1,
            link.as_deref_mut(),
            gathers,
            trace,
            per_row,
        )?;
        if seam_here {
            if let Some(seam) = &prefill_head.seam {
                mine = Some(seam.read(
                    hip,
                    stream,
                    prefill_head.rows as usize,
                    DEVICE_ROW_K as usize,
                    seam_penalties,
                )?);
                if let Some(l) = link.as_mut() {
                    peer = Some(SeamOut::from_frame(
                        l.recv_seam(prefill_head.rows, seam_penalties)?,
                        DEVICE_ROW_K as usize,
                    ));
                }
            }
        }
    }
    Ok((launches, toks, mine, peer))
}

/// C10: a further sampling chunk's head (leader), its row already in the prefill head's `sel`
/// and its seam inputs written: the head, the pick, the seam.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_chunk_head(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    pre: &Planned<'_>,
    prefill_head: &Head<'_>,
    token_plane: &mut Tokens,
    inputs: &BlockInputs,
    mut link: Option<&mut RankLink>,
    gathers: &AtomicU64,
    trace: &HostTrace,
    per_row: bool,
    seam_here: bool,
    seam_penalties: bool,
) -> Result<(u32, Vec<TokenId>, Option<SeamOut>, Option<SeamOut>), ForwardError> {
    let steps = head_ops(
        &pre.plan,
        pre.exec.arena_base(),
        0,
        prefill_head,
        inputs,
        prefill_head.sample_to(token_plane),
    )?;
    {
        let mut clock = prefill_head.clock.borrow_mut();
        clock.key = None;
        clock.armed = None;
    }
    let launches = run_bound(ctx, launcher, stream, &steps)?;
    let toks = sample(
        hip,
        stream,
        prefill_head,
        token_plane,
        1,
        link.as_deref_mut(),
        gathers,
        trace,
        per_row,
    )?;
    let mut mine = None;
    let mut peer = None;
    if seam_here {
        if let Some(seam) = &prefill_head.seam {
            mine = Some(seam.read(
                hip,
                stream,
                prefill_head.rows as usize,
                DEVICE_ROW_K as usize,
                seam_penalties,
            )?);
            if let Some(l) = link.as_mut() {
                peer = Some(SeamOut::from_frame(
                    l.recv_seam(prefill_head.rows, seam_penalties)?,
                    DEVICE_ROW_K as usize,
                ));
            }
        }
    }
    Ok((launches, toks, mine, peer))
}

/// The fused step's decode head (leader): the gathers from rows `c..c + D` of the prefill rung's
/// arena, the head, the argmax and the seam; then every live decode row's pick and the seams.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_decode_head(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    pre: &Planned<'_>,
    dhead: &Head<'_>,
    token_plane: &mut Tokens,
    inputs: &BlockInputs,
    c: u32,
    live: usize,
    mut link: Option<&mut RankLink>,
    gathers: &AtomicU64,
    trace: &HostTrace,
    per_row: bool,
    seam_here: bool,
    seam_penalties: bool,
) -> Result<(u32, Vec<TokenId>, Option<SeamOut>, Option<SeamOut>), ForwardError> {
    let steps = head_ops(
        &pre.plan,
        pre.exec.arena_base(),
        c,
        dhead,
        inputs,
        dhead.sample_to(token_plane),
    )?;
    {
        // An eager submit: the aimed wait's key is a captured shape's, never this one.
        let mut clock = dhead.clock.borrow_mut();
        clock.key = None;
        clock.armed = None;
    }
    let launches = run_bound(ctx, launcher, stream, &steps)?;
    let sampled = sample(
        hip,
        stream,
        dhead,
        token_plane,
        live,
        link.as_deref_mut(),
        gathers,
        trace,
        per_row,
    )?;
    let mut mine = None;
    let mut peer = None;
    if seam_here {
        if let Some(seam) = &dhead.seam {
            mine = Some(seam.read(
                hip,
                stream,
                dhead.rows as usize,
                DEVICE_ROW_K as usize,
                seam_penalties,
            )?);
            if let Some(l) = link.as_mut() {
                peer = Some(SeamOut::from_frame(
                    l.recv_seam(dhead.rows, seam_penalties)?,
                    DEVICE_ROW_K as usize,
                ));
            }
        }
    }
    Ok((launches, sampled, mine, peer))
}

/// A further chunk's `(rows, GDN slot)` as the leader wrote them into the step's blob (its
/// packed `cu` = `[0, rows]` and `cache_idx` = `[slot]`), read on the follower, which keeps no
/// sessions.
fn packed_chunk_facts(indices: &Indices, k: u32) -> Result<(u32, u32), ForwardError> {
    let read = |suffix: &str, at: usize| -> Result<i32, ForwardError> {
        let key = packed_key(k, suffix);
        let (_, offset, _) = indices
            .table
            .iter()
            .find(|(n, _, _)| *n == key)
            .ok_or_else(|| ForwardError::Peer(format!("no packed table `{key}` in the blob")))?;
        let o = *offset as usize + at * 4;
        let b = indices
            .host
            .as_slice()
            .get(o..o + 4)
            .ok_or_else(|| ForwardError::Peer(format!("`{key}` past the blob")))?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let rows = read("cu", 1)?;
    let slot = read("cache_idx", 0)?;
    if rows <= 0 || slot <= 0 {
        return Err(ForwardError::Peer(format!(
            "packed chunk {k}: {rows} row(s) at GDN slot {slot}"
        )));
    }
    Ok((rows as u32, slot as u32))
}

/// **The fused mixed step, on the follower**: the leader's [`Command::Mixed`] mirrored — its
/// blob and block tables, chunk 0's retargets and the further chunks' rows and slots (read from
/// the blob), every seam's inputs, the same launch list from this rank's own plans, each sampling
/// chunk's head, pick and seam, the decode head's picks and seam, and each chunk's context-store
/// frames. Every proposal comes later, as `Propose`s. Returns each sampled chunk's dense logit
/// shard when two or more sampled (the leader's dense fallback asks for them by chunk).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn follow_mixed(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    plans: &mut [Planned<'_>],
    prefill_head: &mut Head<'_>,
    decode_heads: &[Head<'_>],
    indices: &mut Indices,
    tokens: &mut Tokens,
    inputs: &BlockInputs,
    link: &mut RankLink,
    gathers: &AtomicU64,
    trace: &HostTrace,
    per_row: bool,
    retargets: &AtomicU64,
    saber: &SaberRules,
    cmd: &Command,
    cache: &mut FusedCache,
    blob: &[u8],
    tap: Option<&AuxTap>,
    drafter: Option<&DraftResident>,
    draft_counts: &DraftCounters,
) -> Result<Vec<Option<Vec<f32>>>, ForwardError> {
    let Command::Mixed {
        rung,
        chunk_live,
        chunk_samples,
        slot,
        decode_rows,
        width,
        decode_live,
        paged,
        chunk_seam,
        chunk_seam_penalties,
        decode_seam,
        decode_seam_penalties,
        defer_proposals,
        packed,
    } = *cmd
    else {
        return Err(ForwardError::Peer(
            "follow_mixed on another command".to_owned(),
        ));
    };
    if !defer_proposals {
        return Err(ForwardError::Peer(
            "a fused step without deferred proposals: this build only defers".to_owned(),
        ));
    }
    let (more, samples, seam_mask, pen_mask) = if packed == 0 {
        (
            0,
            u8::from(chunk_samples),
            u8::from(chunk_seam),
            u8::from(chunk_seam_penalties),
        )
    } else {
        crate::tp2::packed_fields(packed)
    };
    let kn = 1 + more as usize;
    if kn > MAX_CHUNKS {
        return Err(ForwardError::Peer(format!(
            "a packed step of {kn} chunks; this build carries {MAX_CHUNKS}"
        )));
    }
    let bit = |m: u8, k: usize| m & (1 << k) != 0;
    let t_step = trace.start();
    apply_payload(indices, tokens, blob)?;
    if paged {
        let got = link.follow_paged()?;
        super::apply_paged_any(indices, &got, link)?;
    }
    let pre_ix = plans
        .iter()
        .position(|p| p.phase == Phase::Prefill && p.rows == rung)
        .ok_or(ForwardError::NoPlan {
            phase: Phase::Prefill,
            rows: rung,
            detail: "the leader's fused rung is not a prefill plan of this rank".to_owned(),
        })?;
    let dec_ix = plans
        .iter()
        .position(|p| {
            p.phase == Phase::Decode && !p.ragged && p.rows == decode_rows && p.width == width
        })
        .ok_or(ForwardError::NoPlan {
            phase: Phase::Decode,
            rows: decode_rows,
            detail: "the leader's fused decode half is not a plan of this rank".to_owned(),
        })?;
    // Every chunk's rows and slot: chunk 0's from the command, the others' from the blob.
    let mut facts: Vec<(u32, u32)> = vec![(chunk_live, slot)];
    for k in 1..kn {
        facts.push(packed_chunk_facts(indices, k as u32)?);
    }
    let row0s: Vec<u32> = facts
        .iter()
        .scan(0u32, |at, (r, _)| {
            let here = *at;
            *at += r;
            Some(here)
        })
        .collect();
    let c: u32 = facts.iter().map(|f| f.0).sum();
    let sampled: Vec<usize> = (0..kn).filter(|k| bit(samples, *k)).collect();
    let set_sel = |k: usize| -> Result<(), ForwardError> {
        let sel = ((row0s[k] + facts[k].0) as i32 - 1).to_le_bytes();
        hip.memcpy_h2d_blocking(&prefill_head.sel, &sel, stream)
            .map_err(device_err("the row index"))
    };
    if let Some(&k0) = sampled.first() {
        set_sel(k0)?;
    }
    let moved = plans[pre_ix]
        .plan
        .retarget_prefill_rows(chunk_live, saber)
        .map_err(|e| ForwardError::Device(e.to_string()))?;
    if moved > 0 && chunk_live < rung {
        retargets.fetch_add(1, Ordering::Relaxed);
    }
    plans[pre_ix]
        .plan
        .retarget_prefill_state(slot)
        .map_err(|e| ForwardError::Device(e.to_string()))?;
    plans[pre_ix]
        .exec
        .zero_constants(hip, &plans[pre_ix].plan, stream)
        .map_err(device_err("re-seeding the shared prefill arena's zeros"))?;
    let extras = extra_chunks(
        &plans[pre_ix],
        &(1..kn)
            .map(|k| (row0s[k], facts[k].0, facts[k].1))
            .collect::<Vec<_>>(),
        saber,
    )?;
    let dhead =
        decode_heads
            .iter()
            .find(|h| h.rows == decode_rows)
            .ok_or(ForwardError::NoPlan {
                phase: Phase::Decode,
                rows: decode_rows,
                detail: "no decode head for the fused step".to_owned(),
            })?;
    // Every chunk seam's inputs, in the leader's order, then the decode seam's.
    let mut got_chunk: Vec<Option<SeamInputs>> = (0..kn).map(|_| None).collect();
    for &k in &sampled {
        if bit(seam_mask, k) {
            let got = link.follow_seam(prefill_head.rows, bit(pen_mask, k))?;
            got_chunk[k] = Some(SeamInputs {
                temps: got.temps,
                pen_ids: got.pen_ids,
            });
        }
    }
    let write_seam = |k: usize| -> Result<(), ForwardError> {
        if let Some(g) = &got_chunk[k] {
            let seam = prefill_head.seam.as_ref().ok_or_else(|| {
                ForwardError::Peer(
                    "the leader ran the chunk's seam and this rank has none".to_owned(),
                )
            })?;
            seam.write(hip, stream, &g.temps, &g.pen_ids, None)?;
        }
        Ok(())
    };
    if let Some(&k0) = sampled.first() {
        write_seam(k0)?;
    }
    if decode_seam {
        let got = link.follow_seam(dhead.rows, decode_seam_penalties)?;
        let seam = dhead.seam.as_ref().ok_or_else(|| {
            ForwardError::Peer("the leader ran the decode seam and this rank has none".to_owned())
        })?;
        seam.write(hip, stream, &got.temps, &got.pen_ids, None)?;
    }
    // The body and the first sampling chunk's head, then its pick.
    let t_blob = trace.start();
    let uploads = vec![
        index_upload(stream, indices, true)?,
        ids_upload(stream, tokens, rung as usize * 4)?,
    ];
    hip.run_batch(&uploads)
        .map_err(device_err("uploading the fused step's tables and ids"))?;
    trace.stop(HostPhase::Blob, t_blob);
    {
        let fb = binding_for(cache, plans, pre_ix, dec_ix)?;
        let pre = &plans[pre_ix];
        let mut steps = bind_fused(pre, &plans[dec_ix], fb, inputs, c, tap, &extras)?;
        if !sampled.is_empty() {
            steps.extend(head_ops(
                &pre.plan,
                pre.exec.arena_base(),
                0,
                prefill_head,
                inputs,
                prefill_head.sample_to(tokens),
            )?);
        }
        {
            let mut clock = prefill_head.clock.borrow_mut();
            clock.key = None;
            clock.armed = None;
        }
        run_bound(ctx, launcher, stream, &steps)?;
    }
    let keep = sampled.len() >= 2;
    let mut stash: Vec<Option<Vec<f32>>> = vec![None; kn];
    for (n, &k) in sampled.iter().enumerate() {
        if n > 0 {
            set_sel(k)?;
            write_seam(k)?;
            let pre = &plans[pre_ix];
            let steps = head_ops(
                &pre.plan,
                pre.exec.arena_base(),
                0,
                prefill_head,
                inputs,
                prefill_head.sample_to(tokens),
            )?;
            {
                let mut clock = prefill_head.clock.borrow_mut();
                clock.key = None;
                clock.armed = None;
            }
            run_bound(ctx, launcher, stream, &steps)?;
        }
        follow_picks(hip, stream, prefill_head, 1, link, gathers, trace, per_row)?;
        if bit(seam_mask, k) {
            send_seam_half(hip, stream, prefill_head, link, bit(pen_mask, k))?;
        }
        if keep {
            stash[k] = Some(dense_shard_of(hip, stream, prefill_head, 0)?);
        }
    }
    // The decode head over rows `c..`.
    {
        let pre = &plans[pre_ix];
        let steps = head_ops(
            &pre.plan,
            pre.exec.arena_base(),
            c,
            dhead,
            inputs,
            dhead.sample_to(tokens),
        )?;
        {
            let mut clock = dhead.clock.borrow_mut();
            clock.key = None;
            clock.armed = None;
        }
        run_bound(ctx, launcher, stream, &steps)?;
    }
    follow_picks(
        hip,
        stream,
        dhead,
        decode_live.max(1) as usize,
        link,
        gathers,
        trace,
        per_row,
    )?;
    if decode_seam {
        send_seam_half(hip, stream, dhead, link, decode_seam_penalties)?;
    }
    if let (Some(res), Some(t)) = (drafter, tap) {
        for _ in 0..kn {
            follow_ctx_store(
                hip,
                ctx,
                launcher,
                stream,
                res,
                t,
                link,
                draft_counts,
                trace,
            )?;
        }
    }
    if let Some(t) = t_step {
        trace.chunk(rung, c, t.elapsed());
    }
    trace.stop(HostPhase::Step, t_step);
    Ok(stash)
}

/// The follower's half of a head's picks: post, wait, read and swap (follow_step's).
#[allow(clippy::too_many_arguments)]
fn follow_picks(
    hip: &HipThread,
    stream: &Stream,
    head: &Head<'_>,
    sampled: usize,
    link: &mut RankLink,
    gathers: &AtomicU64,
    trace: &HostTrace,
    per_row: bool,
) -> Result<(), ForwardError> {
    let t = trace.start();
    post_pick_table(hip, stream, head, sampled)?;
    wait_step(hip, stream, head, trace)?;
    trace.stop(HostPhase::Wait, t);
    let t = trace.start();
    let mine = read_picks(hip, stream, head, sampled, gathers, per_row)?;
    trace.stop(HostPhase::Picks, t);
    let t = trace.start();
    let _ = link.swap_picks(&mine)?;
    trace.stop(HostPhase::SwapPicks, t);
    Ok(())
}

/// The follower's half of a head's seam, in the frame the leader waits for.
fn send_seam_half(
    hip: &HipThread,
    stream: &Stream,
    head: &Head<'_>,
    link: &mut RankLink,
    penalties: bool,
) -> Result<(), ForwardError> {
    if let Some(seam) = &head.seam {
        let out = seam.read(
            hip,
            stream,
            head.rows as usize,
            DEVICE_ROW_K as usize,
            penalties,
        )?;
        link.send_seam(head.rows, &out.frame(head.rows as usize))?;
    }
    Ok(())
}
