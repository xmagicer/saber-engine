//! **C10: the packed fused step's host laws**, over the engine's own plans and with no card.
//!
//! A packed step lays K chunks out per prompt in queue order (`saber_sched::mixed_law::pack`),
//! runs chunk 0's chunk group as the prefill plan is bound and each further chunk's as the same
//! steps with their row-carrying arena bindings SHIFTED to its rows
//! (`saber_exec::mixed::shift_rows`) and the chunk path's private scratch reused at the plan's
//! offsets (`chunk_private_buffers`). These tests pin, at every rung a packed step can take:
//!
//! * **the chunk-private set** is exactly the buffers no row-local op binds (so reusing them per
//!   chunk cannot hide a row another op reads), and it is not empty (the GDN chunk path's split);
//!   the attention's `q` / `out` edges are NOT in it (they carry rows) — the control;
//! * **the row map**: every shifted binding of chunk `k` lands at rows `r_k..r_k + c_k` of the
//!   same buffer, the chunks' ranges are disjoint, in order, and end before the decode rows at
//!   `C`; a chunk that would run past the buffer is refused (the negative control);
//! * **the blob law**: every step-varying chunk-side table has one suffix every op shares, so a
//!   packed chunk's one table set (`packed_key(k, suffix)`) serves all of its layers.

use std::collections::{BTreeMap, BTreeSet};

use saber_engine::small_chunk::{rules_for_rows, small_chunk_rules, SmallChunkClass};
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, RuntimeParams, Source};
use saber_exec::mixed::{
    chunk_private_buffers, is_chunk_side, packed_key, parse_packed_key, shift_rows,
};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};
use saber_sched::mixed_law::{pack, MAX_PACKED_CHUNKS};

const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];

fn graph() -> Graph {
    Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph")
}

fn body() -> PlanOptions {
    PlanOptions::BODY
        .rank(0)
        .with_collectives(Collectives::Real)
        .with_batch_invariant_attn()
        .with_table_blocks(257)
}

fn prefill<'g>(g: &'g Graph, rung: u32) -> BlockPlan<'g> {
    let mut base = LaunchRules::prod_mxfp4().expect("rules");
    base.knobs.folded_mt = true;
    base.knobs.atiled_mid = true;
    base.knobs.skinny = true;
    let small = small_chunk_rules(&base, 128, SmallChunkClass::Exact).expect("small");
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    BlockPlan::with_options(
        g,
        Bucket::Prefill {
            tokens: rung,
            seqs: 1,
        },
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        rules_for_rows(&base, Some(&(128, small)), rung),
        &SaberRules::prod_saber().expect("saber"),
        body(),
    )
    .expect("a plan")
}

/// Every arena binding of a chunk-side step: `(step, binding name, buffer, offset, bytes)`.
fn chunk_side_arena(p: &BlockPlan<'_>) -> Vec<(String, String, String, u64, u64)> {
    let mut out = Vec::new();
    for s in &p.steps {
        if !is_chunk_side(s.name()) {
            continue;
        }
        for (name, src) in &s.bindings {
            if let Source::Arena {
                buffer,
                offset,
                bytes,
            } = src
            {
                out.push((
                    s.name().to_owned(),
                    name.clone(),
                    buffer.clone(),
                    *offset,
                    *bytes,
                ));
            }
        }
    }
    out
}

#[test]
fn the_chunk_private_set_is_the_chunk_paths_own_scratch_and_the_row_edges_are_shifted() {
    let g = graph();
    for rung in [96u32, 192, 384, 1536] {
        let p = prefill(&g, rung);
        let private = chunk_private_buffers(&p);
        assert!(
            !private.is_empty(),
            "p{rung}: the GDN chunk path has scratch"
        );
        // No row-local step binds a private buffer.
        for s in &p.steps {
            if is_chunk_side(s.name()) {
                continue;
            }
            for (_, src) in &s.bindings {
                if let Source::Arena { buffer, .. } = src {
                    assert!(
                        !private.contains(buffer),
                        "p{rung}: `{}` binds the chunk-private `{buffer}`",
                        s.name()
                    );
                }
            }
        }
        // THE CONTROL: the attention's row edges carry rows a row-local op wrote or reads, so
        // they are shifted, never reused.
        let attn: BTreeSet<String> = chunk_side_arena(&p)
            .into_iter()
            .filter(|(op, ..)| op.ends_with(".attn.prefill"))
            .map(|(_, _, b, ..)| b)
            .collect();
        assert!(!attn.is_empty());
        assert!(
            attn.iter().all(|b| !private.contains(b)),
            "p{rung}: an attention edge is marked private: {attn:?}"
        );
    }
}

#[test]
fn every_packed_chunk_lands_on_its_own_rows_in_order_before_the_decode_rows() {
    let g = graph();
    // BetterBench's burst beside a b8s1 decode half, and a mid-size pack beside b8s5.
    for (queue, d_rows) in [
        (vec![35u32, 332, 178, 150, 210, 96, 240, 183], 8u32),
        (vec![120, 90, 60], 40),
        (vec![40, 30], 8),
    ] {
        let pk = pack(&queue, d_rows, &LADDER, 8192, MAX_PACKED_CHUNKS);
        let rung = pk.rung.expect("in class");
        assert!(pk.len() >= 2);
        let p = prefill(&g, rung);
        let private = chunk_private_buffers(&p);
        let ranges = pk.ranges();
        let c = pk.chunk_rows();
        assert!(c + d_rows <= rung);
        // Per buffer, the byte ranges every chunk's shifted bindings cover.
        let mut spans: BTreeMap<(String, String), Vec<(u64, u64)>> = BTreeMap::new();
        for (op, name, buffer, offset, bytes) in chunk_side_arena(&p) {
            if private.contains(&buffer) {
                continue;
            }
            let b = p.buffer(&buffer).expect("planned");
            let per = b.rows.rows(b.bytes).max(1);
            let per = b.bytes / per;
            for r in &ranges {
                let at = shift_rows(
                    &p,
                    &buffer,
                    offset,
                    u64::from(r.start),
                    u64::from(r.end - r.start),
                )
                .unwrap_or_else(|e| panic!("p{rung} {op}.{name}: {e}"));
                assert_eq!(at, offset + u64::from(r.start) * per, "{op}.{name}");
                // The chunk's own rows (the kernels read `cu`'s live rows, not the binding's
                // planned extent `bytes`) stay inside the buffer.
                let _ = bytes;
                assert!(
                    at + u64::from(r.end - r.start) * per <= b.offset + b.bytes,
                    "{op}.{name} inside `{buffer}`"
                );
                spans
                    .entry((op.clone(), name.clone()))
                    .or_default()
                    .push((at, u64::from(r.end - r.start) * per));
            }
            // The decode rows start after the last chunk: a range at `c` still fits the rung.
            assert!(shift_rows(&p, &buffer, offset, u64::from(c), u64::from(d_rows)).is_ok());
            // NEGATIVE CONTROL: a chunk running past the rung is refused.
            assert!(
                shift_rows(&p, &buffer, offset, u64::from(rung - 10), 20).is_err(),
                "{op}.{name}: past the rung must be refused"
            );
        }
        // The chunks' row spans of one binding are in order and disjoint.
        for ((op, name), s) in spans {
            for w in s.windows(2) {
                assert!(
                    w[0].0 + w[0].1 <= w[1].0,
                    "p{rung} {op}.{name}: chunk spans overlap"
                );
            }
        }
    }
}

#[test]
fn a_packed_chunks_tables_are_one_set_by_suffix() {
    let g = graph();
    let p = prefill(&g, 384);
    let keys: BTreeSet<String> = p.index_keys().into_iter().collect();
    let suffixes: BTreeSet<&str> = keys
        .iter()
        .filter_map(|k| k.rsplit_once('.'))
        .filter(|(op, _)| is_chunk_side(op))
        .map(|(_, s)| s)
        .collect();
    for want in ["cu", "cache_idx", "block_table", "seqused_k"] {
        assert!(suffixes.contains(want), "chunk-side `{want}`: {suffixes:?}");
    }
    for s in &suffixes {
        let k = packed_key(3, s);
        assert_eq!(parse_packed_key(&k), Some((3, *s)));
        assert!(!keys.contains(&k), "a packed key collides with a plan key");
    }
}

#[test]
fn cached_prefixes_survive_chunk_chunk_and_chunk_decode_appends() {
    use saber_engine::{index_table, SeqFacts, StepIndex};
    use saber_exec::forward::KV_BLOCK;
    use saber_exec::mixed::merge_rows;
    use saber_plan::Phase;
    use saber_sched::mixed_law::{isolated_at_cursors, RowState};

    // Two chunks and one decoder share both complete prefix pages. Their next
    // two tokens start at offsets 0, 1, and 2 in three distinct append pages.
    let tables = [[7, 8, 11], [7, 8, 12], [7, 8, 13]];
    let seqs: Vec<_> = tables
        .iter()
        .enumerate()
        .map(|(i, pages)| SeqFacts {
            state_slot: i as u32 + 1,
            done_before: 2 * KV_BLOCK + i as u32,
            kv_blocks: pages,
            live_rows: 2,
            rope: None,
        })
        .collect();
    isolated_at_cursors(
        seqs.iter().map(|s| {
            (
                RowState {
                    kv_blocks: s.kv_blocks,
                    gdn_set: s.state_slot,
                },
                s.done_before,
            )
        }),
        KV_BLOCK,
    )
    .expect("complete shared pages");

    let append = |idx, phase, rows| {
        let ids = vec![0; rows as usize];
        let step = StepIndex {
            phase,
            rows,
            max_blocks: 3,
            seqs: &seqs[idx..=idx],
            ids: &ids,
            committed_before: &[1],
            band_width: None,
        };
        index_table("l3.attn.kv_append.slot", &step).expect("append slots")
    };
    let first = append(0, Phase::Prefill, 8);
    let second = append(1, Phase::Prefill, 2);
    let decode = append(2, Phase::Decode, 2);
    let chunks = merge_rows(&first, 8, &second, 2, 2, 8).unwrap();
    let all = merge_rows(&chunks, 8, &decode, 2, 4, 8).unwrap();
    let slots: Vec<_> = all
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let page = i64::from(KV_BLOCK);
    assert_eq!(
        slots,
        [
            11 * page,
            11 * page + 1,
            12 * page + 1,
            12 * page + 2,
            13 * page + 2,
            13 * page + 3,
            -1,
            -1
        ]
    );
}
