//! The planner: resolution, binding, and the four refusals that are structural.

mod common;

use common::{tiny_manifest, tiny_model, tuned_cache, tuned_cache_with};
use saber_plan::{
    plan, reference, resolve, AutotuneCache, BucketOps, Constraint, DeviceProfile, Implementation,
    LaunchModel, Manifest, ModelSpec, Phase, PlanError, Provenance, Registry, WorkspaceKind,
};

const MIB: u64 = 1024 * 1024;

fn qwen_setup() -> (Manifest, DeviceProfile, ModelSpec, Registry) {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let registry = reference::reference_registry();
    (manifest, device, model, registry)
}

#[test]
fn qwen38_dense_shapes_are_sharded_by_tp() {
    let tp1 = ModelSpec::qwen38_27b_dense(1).unwrap();
    let tp2 = ModelSpec::qwen38_27b_dense(2).unwrap();
    // 8 merged linear rows, paged attention decode, the THREE attention-glue rows (q/k norm
    // + rope, the output gate's quant, the paged K/V write — all in both phases), the two GDN
    // decode routes, and the five prefill-only rows (query-tiled attention, the GDN chunk
    // trio, the gated norm). The all-reduce exists only when the model is sharded, so TP=1 has
    // 19 and TP=2 has 21.
    assert_eq!(tp1.ops.len(), 19);
    // TP=2 adds BOTH all-reduce rows; exactly one of them applies in any given bucket.
    assert_eq!(tp2.ops.len(), 21);
    assert!(
        tp1.op("layer.allreduce").is_none(),
        "nothing to reduce at TP=1"
    );
    // The all-reduce is two rows banded on the MESSAGE SIZE: the exact family below
    // `quant_min_bytes`, the WHT one at or above it. At hidden 5120 x bf16 that is 10240 B per
    // row, so m <= 12 is exact — b1s5 (m=5) and b2s5 (m=10) — and m >= 13 is quantised.
    let ar_exact = tp2.op("layer.allreduce.exact").unwrap();
    let ar_wht = tp2.op("layer.allreduce.wht").unwrap();
    assert_eq!((ar_exact.count, ar_wht.count), (128, 128));
    assert_eq!(
        (
            ar_exact.calls_in(Phase::Decode, 1, 5),
            ar_wht.calls_in(Phase::Decode, 1, 5)
        ),
        (128, 0)
    );
    assert_eq!(
        (
            ar_exact.calls_in(Phase::Decode, 2, 10),
            ar_wht.calls_in(Phase::Decode, 2, 10)
        ),
        (128, 0)
    );
    assert_eq!(
        (
            ar_exact.calls_in(Phase::Decode, 4, 20),
            ar_wht.calls_in(Phase::Decode, 4, 20)
        ),
        (0, 128)
    );
    assert_eq!(
        (
            ar_exact.calls_in(Phase::Decode, 8, 40),
            ar_wht.calls_in(Phase::Decode, 8, 40)
        ),
        (0, 128)
    );
    // The band edge itself, derived from hidden and the rules' threshold: 12 x 10240 = 122880 B
    // is under 128 KiB, 13 x 10240 = 133120 B is over.
    assert_eq!(
        (
            ar_exact.calls_in(Phase::Decode, 1, 12),
            ar_wht.calls_in(Phase::Decode, 1, 12)
        ),
        (128, 0)
    );
    assert_eq!(
        (
            ar_exact.calls_in(Phase::Decode, 1, 13),
            ar_wht.calls_in(Phase::Decode, 1, 13)
        ),
        (0, 128)
    );
    assert_eq!(tp1.layers, 64);
    assert_eq!(tp1.gdn_layers + tp1.attn_layers, tp1.layers);

    // Column-parallel: N divides. Row-parallel: K divides.
    // The MERGED GDN in_proj: concat(q, k, v, z) = 10240 + 6144. At TP=2 that is the `n8192`
    // shape the tuned decode band is written against.
    let qkv1 = tp1.op("layer.gdn.in_proj_qkvz").unwrap();
    let qkv2 = tp2.op("layer.gdn.in_proj_qkvz").unwrap();
    assert_eq!((qkv1.n, qkv1.k), (16384, 5120));
    assert_eq!((qkv2.n, qkv2.k), (8192, 5120));
    let down1 = tp1.op("layer.mlp.down_proj").unwrap();
    let down2 = tp2.op("layer.mlp.down_proj").unwrap();
    assert_eq!((down1.n, down1.k), (5120, 17408));
    assert_eq!((down2.n, down2.k), (5120, 8704));
    assert_eq!(tp2.op("lm_head").unwrap().n, 124_160);
    // The merged attention QKV: q+gate 12288, k 1024, v 1024 -> `n7168` at TP=2. And the
    // merged in_proj_ba, two rows per value head -> the `gate48` shape.
    assert_eq!(tp2.op("layer.attn.qkv").unwrap().n, 7168);
    assert_eq!(tp2.op("layer.gdn.in_proj_ba").unwrap().n, 48);
    assert_eq!(tp2.op("layer.mlp.gate_up").unwrap().n, 17408);

    // 305 GEMM calls per step, MERGED as prod launches them: 48 GDN x3 (qkvz, ba, out_proj),
    // 16 attention x2 (qkv, o_proj), 64 MLP x2 (gate_up, down), one head. Prod's own census
    // says `linear layers: 304/304` -- this minus lm_head, which its requant patch handles
    // separately.
    let calls: u32 = tp2
        .ops
        .iter()
        .filter(|o| o.op == "gemm_w4a8")
        .map(|o| o.count)
        .sum();
    assert_eq!(calls, 305);
    // And the saber half: 16 attention layers, 48 GDN layers, 2 all-reduces per layer.
    assert_eq!(tp2.op("layer.attn.paged_decode").unwrap().count, 16);
    // The GDN decode step is two routes banded on the bucket's batch: at TP=2, H = 24 against
    // FUSED_MAX_ITEMS 32, so batch 1 is the fused kernel and batch 2 and up is the pair.
    let fused = tp2.op("layer.gdn.decode_step.fused").unwrap();
    let pair = tp2.op("layer.gdn.decode_step.pair").unwrap();
    // (batch, m) — the GDN band reads the batch, the all-reduce band reads m.
    assert_eq!(
        (
            fused.calls_in(Phase::Decode, 1, 5),
            fused.calls_in(Phase::Decode, 8, 40)
        ),
        (48, 0)
    );
    assert_eq!(
        (
            pair.calls_in(Phase::Decode, 1, 5),
            pair.calls_in(Phase::Decode, 8, 40)
        ),
        (0, 48)
    );
    // At TP=1 a rank owns all 48 value heads, so even batch 1 is 48 items: the pair always.
    let fused1 = tp1.op("layer.gdn.decode_step.fused").unwrap();
    assert_eq!(
        fused1.calls_in(Phase::Decode, 1, 5),
        0,
        "48 items exceeds FUSED_MAX_ITEMS"
    );
}

#[test]
fn tp3_needs_a_padded_checkpoint_and_says_so() {
    let err = ModelSpec::qwen38_27b_dense(3).unwrap_err();
    assert!(matches!(err, PlanError::Model(_)), "{err}");
    assert!(err.to_string().contains("does not divide by tp=3"), "{err}");
    assert!(err.to_string().contains("padded head layout"), "{err}");
}

#[test]
fn plans_the_reference_configuration() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let plan = plan(&manifest, &device, &model, &registry, &cache).unwrap();

    assert_eq!(plan.body.tier_id, manifest.model.tier_id);
    assert_eq!(plan.body.tp, 2);
    assert_eq!(plan.body.power_cap_w, 210);
    assert_eq!(plan.body.gfx_target, "gfx1201");
    // Three: the MXFP4 family, saber, and the attention glue.
    assert_eq!(plan.body.plugins.len(), 3);
    assert_eq!(plan.plan_hash().len(), 64);
    assert!(plan.verify_hash());
    assert_eq!(plan.buckets().len(), 4);

    for bucket in plan
        .buckets()
        .iter()
        .filter(|b| b.bucket.phase == Phase::Decode)
    {
        // 8 merged GEMM rows + attention + the THREE attention-glue routes + the all-reduce
        // + exactly ONE of the two GDN decode routes.
        assert_eq!(bucket.ops.len(), 14, "one row per distinct (op, shape)");
        let ops = bucket.ops.ops();
        // Every bound route came from a measurement, so every one is at least `measured`.
        assert!(ops.iter().all(|o| o.promotion.provenance.is_some()));
        assert!(ops.iter().all(|o| o.graph_safe && !o.may_sync));
        // The decode band wins at both bucket M values, over the WPERM candidate.
        let gemms: Vec<_> = ops.iter().filter(|o| o.op == "gemm_w4a8").collect();
        // Eight merged rows, not eleven tensors: qkvz, ba, out_proj, qkv, o_proj, gate_up,
        // down_proj, lm_head.
        assert_eq!(gemms.len(), 8);
        assert!(
            gemms
                .iter()
                .all(|o| o.impl_name == "gemm_w4a8_mxfp4_decode"),
            "{:?}",
            gemms[0]
        );
        assert!(gemms.iter().all(|o| o.plugin == "mxfp4"));
        // The saber half binds to the second plugin.
        let saber: Vec<_> = ops.iter().filter(|o| o.plugin == "saber").collect();
        assert_eq!(
            saber.len(),
            3,
            "attention, the fused GDN update, the all-reduce"
        );
    }
}

#[test]
fn launch_count_is_declared_statically() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let plan = resolve(&manifest, &device, &model, &registry)
        .unwrap()
        .bind(&tuned_cache(&resolved))
        .unwrap();

    // Every fixture record splits K 4 ways, so each GEMM call is a GEMM plus its reduce: 2 x 401
    // = 802. saber adds 16 attention decodes at TWO launches (the split-KV combine is the
    // second), 128 all-reduces at one, and the GDN decode step at 48 x 1 (fused, batch 1) or
    // 48 x 2 (the pair, batch 8). Prod's traced figure for the same step is 1084.
    //
    // The two buckets DIFFER, and that is the result: a flat per-step count could not say it.
    for bucket in plan.buckets() {
        let want = match (bucket.bucket.phase, bucket.bucket.batch) {
            // Prefill: 354 GEMM + 16 query-tiled attention + 144 for the GDN chunk trio + 48
            // gated norms + 128 all-reduces. A different op set from either decode bucket, not
            // a scaled one. The GEMM 354 is 256 A-tiled calls at one launch each, plus 49
            // row-major rows at two launches (the split-K reduce): `gdn.out_proj`'s 48 calls,
            // and lm_head's one -- which runs at M=1 here, on the sampled position, so it is
            // row-major like every decode GEMM rather than A-tiled like the rest of the chunk.
            //
            // **It was 322 until I20**, with the 16 `attn.o_proj` calls row-major and the 48
            // `gdn.out_proj` ones A-tiled: `ModelSpec` carried `.a_row_major()` on the wrong
            // row. The engine never read that declaration (its kernels come from
            // `saber_kernels::rules`, and `saber_exec::layer` reads the A EDGE since P6), so no
            // token depended on it -- but the plan plane's launch budget did, and it was a
            // budget for a routing nothing performs. 32 more launches is 48 row-major GEMMs
            // gaining a split-K reduce each and 16 gaining the A-tiled kernel's single dispatch:
            // +96 - 64 = +32.
            //
            // Every bucket also carries the attention glue's 48 — 16 layers x
            // (qk_norm_rope, kv_append, gate_quant), one launch each, in BOTH phases. They
            // took the declared budget from 658 / 818 / 866 to 706 / 866 / 914, and I20's swap
            // takes the prefill half of that from 706 to 738.
            //
            // G2 (2026-09-23): with the tiled GDN norm (`gdn_norm_tiled`) `gdn.out_proj`'s 48
            // calls are A-tiled too, one launch each instead of two: 354 -> 306, 738 -> 690.
            // lm_head stays the one row-major row.
            (Phase::Prefill, _) => 306 + 16 + 144 + 48 + 128 + 48,
            // Decode: 305 merged GEMMs at two launches each (the split-K reduce).
            (Phase::Decode, 1) => 610 + 32 + 48 + 128 + 48,
            (Phase::Decode, _) => 610 + 32 + 96 + 128 + 48,
        };
        assert_eq!(bucket.n_launches, want, "{}", bucket.bucket.label());
        let summed: u32 = bucket.ops.ops().iter().map(|o| o.launches_total).sum();
        assert_eq!(summed, bucket.n_launches);
        let gdn = bucket
            .ops
            .ops()
            .into_iter()
            .find(|o| o.op_name == "layer.gdn.in_proj_qkvz")
            .unwrap();
        assert_eq!(
            gdn.launches_each,
            if bucket.bucket.phase == Phase::Prefill {
                1
            } else {
                2
            }
        );
        assert_eq!(gdn.occurrences, 48);
        assert_eq!(
            gdn.launches_total,
            if bucket.bucket.phase == Phase::Prefill {
                48
            } else {
                96
            }
        );
    }
    // Each of the four buckets gained the attention glue's 48 (16 layers x 3 ops, one launch
    // each, in both phases): 818 -> 866, 866 -> 914, 658 -> 706. I20's producer swap then took
    // the two prefill buckets 706 -> 738 and left both decode buckets alone, which is what a
    // change to the A-LAYOUT alone must do: at M = 5 and M = 40 every edge is row-major
    // whichever producer wrote it, so nothing below the band edge can move. G2's tiled GDN norm
    // takes the prefill buckets 738 -> 690 (48 out_proj calls lose their split-K reduce) and,
    // for the same reason, leaves both decode buckets alone.
    assert_eq!(plan.total_launches(), 866 + 914 + 690 + 690);
}

#[test]
fn gdn_state_is_gated_against_the_infinity_cache() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let plan = resolve(&manifest, &device, &model, &registry)
        .unwrap()
        .bind(&tuned_cache(&resolved))
        .unwrap();

    let conc8 = plan.buckets().iter().find(|b| b.bucket.batch == 8).unwrap();
    // The 09-14 arithmetic: 8 x 5 x 3.0 MiB / tp 2 = 60 MiB = 62.9 MB, 98% of the 64 MB IC.
    assert_eq!(conc8.gdn_state.slots, 40);
    assert_eq!(conc8.gdn_state.bytes_per_slot_per_layer, 3 * MIB / 2);
    assert_eq!(conc8.gdn_state.working_set_bytes, 60 * MIB);
    assert_eq!(conc8.gdn_state.resident_bytes, 60 * MIB * 48);
    assert!((conc8.gdn_state.ic_fraction() - 0.9375).abs() < 1e-9);

    let single = plan.buckets().iter().find(|b| b.bucket.batch == 1).unwrap();
    assert_eq!(single.gdn_state.working_set_bytes, 7 * MIB + MIB / 2);
}

#[test]
fn an_ic_overflowing_bucket_is_refused_with_the_arithmetic() {
    let (mut manifest, device, model, registry) = qwen_setup();
    // `max_num_seqs` 8 x verify width 6 — the configuration measured at +2.2 ms/step on 09-14.
    // (The ledger writes that cell `8:6`; it is NOT a `SABER_DRAFT_SCHEDULE` entry, which is a
    // batch-size ceiling on serial MTP forwards and is inert under dflash — ws-t1.)
    manifest.buckets[1].spec_width = 6;
    let err = resolve(&manifest, &device, &model, &registry).unwrap_err();
    let PlanError::InfinityCache(overflow) = err else {
        panic!("expected an IC refusal")
    };
    assert_eq!(overflow.bucket, "b8s6c8192");
    assert_eq!(overflow.working_set_bytes, 72 * MIB);
    assert_eq!(overflow.ic_bytes, 64 * MIB);
    let rendered = overflow.to_string();
    assert!(rendered.contains("72.00 MiB > 64.00 MiB"), "{rendered}");
    assert!(
        rendered.contains("batch 8 x spec_width 6 x 3.00 MiB per GDN layer / tp 2"),
        "{rendered}"
    );

    // max_num_seqs = 16 at width 5 is further over the edge (+5.5 ms/step, same finding).
    let mut manifest16 = manifest.clone();
    manifest16.buckets[1].spec_width = 5;
    manifest16.buckets[1].batch = 16;
    assert!(matches!(
        resolve(&manifest16, &device, &model, &registry),
        Err(PlanError::InfinityCache(_))
    ));

    // AND SO IS THE SHAPE SABER ACTUALLY SERVES AT conc-8 (ws-t1): batch 8 x the DFlash2 block's
    // own width 8 is 96 MiB, half as much again as the cache. The engine's own bucket builder
    // does not consult this budget — only the manifest planner does — so this assertion is what
    // says the shape `--engine-decode-widths 1,8 --engine-max-seqs 8` runs is one a PLAN would
    // have refused, which is the reason the ladder exists.
    let mut manifest88 = manifest.clone();
    manifest88.buckets[1].spec_width = 8;
    let err = resolve(&manifest88, &device, &model, &registry).unwrap_err();
    let PlanError::InfinityCache(overflow) = err else {
        panic!("expected an IC refusal at b8s8")
    };
    assert_eq!(overflow.bucket, "b8s8c8192");
    assert_eq!(overflow.working_set_bytes, 96 * MIB);
}

#[test]
fn an_untuned_route_is_a_hard_error() {
    let (manifest, device, model, registry) = qwen_setup();
    let err = plan(&manifest, &device, &model, &registry, &AutotuneCache::new()).unwrap_err();
    let PlanError::NotTuned(not_tuned) = err else {
        panic!("expected NotTuned")
    };
    assert_eq!(not_tuned.key.op, "gemm_w4a8");
    assert_eq!(not_tuned.key.power_cap_w, 210);
    assert_eq!(
        not_tuned.candidates,
        vec![
            "gemm_w4a8_mxfp4_decode".to_string(),
            "gemm_w4a8_mxfp4_wperm".to_string(),
            "gemm_w4a8_mxfp4_folded".to_string(),
        ]
    );
}

#[test]
fn requirements_enumerate_the_tuning_campaign() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let requirements = resolved.requirements();
    // 14 rows x 2 decode buckets + 17 x 2 prefill ones: the attention glue added three
    // routes to every bucket, and every route is a measurement the campaign owes.
    assert_eq!(requirements.len(), 62);
    assert_eq!(resolved.missing(&AutotuneCache::new()).len(), 62);

    // Every key carries M, and the two buckets ask for different M.
    let ms: Vec<i64> = requirements
        .iter()
        .filter_map(|r| r.key.shape.m())
        .collect();
    // Two decode buckets and two prefill ones -- plus M=1, which is `lm_head` at a prefill
    // bucket: logits are computed for the SAMPLED positions, one per sequence, not for every
    // token of the chunk.
    assert!(
        ms.iter().all(|m| matches!(m, 1 | 5 | 40 | 2048 | 8192)),
        "{ms:?}"
    );
    assert_eq!(
        ms.iter().filter(|m| **m == 1).count(),
        2,
        "one lm_head per prefill bucket"
    );

    let cache = tuned_cache(&resolved);
    assert!(resolved.missing(&cache).is_empty());

    // Drop one record and the routes that KEY on it reappear as work. Usually that is one
    // route; it is more when several routes share an autotune key, which is the collapsing the
    // work list exists to do — `attn_gate_quant` and `kv_append` run at the same bucket shape
    // in both phases, so a prefill and a decode bucket can share a measurement.
    let mut partial = AutotuneCache::new();
    let dropped = cache
        .entries()
        .next()
        .expect("a non-empty cache")
        .key
        .clone();
    for entry in cache.entries().skip(1) {
        partial.insert(entry.key.clone(), entry.record.clone());
    }
    let missing = resolved.missing(&partial);
    assert!(!missing.is_empty(), "dropping a record must create work");
    assert!(
        missing.iter().all(|r| r.key == dropped),
        "only the dropped key may be missing: {missing:?}"
    );
}

#[test]
fn plan_hash_is_stable_across_runs() {
    let (manifest, device, model, registry) = qwen_setup();
    let first = {
        let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
        let cache = tuned_cache(&resolved);
        resolved.bind(&cache).unwrap()
    };
    let second = {
        let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
        let cache = tuned_cache(&resolved);
        resolved.bind(&cache).unwrap()
    };
    assert_eq!(first.plan_hash(), second.plan_hash());
    assert_eq!(first.body.plan_input_hash, second.body.plan_input_hash);
}

#[test]
fn plan_hash_changes_when_a_binding_changes() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let base = resolved.bind(&tuned_cache(&resolved)).unwrap();

    // Same inputs, different binding: the WPERM candidate wins the sweep instead. Only the GEMM
    // rows have a second candidate, so every other op keeps the one it had -- which is enough to
    // move the plan hash, and is the honest thing to write: a record naming an implementation that
    // does not cover the op is a StaleRecord, not a rebinding.
    let other = tuned_cache_with(&resolved, |r| {
        if r.candidates.contains(&"gemm_w4a8_mxfp4_wperm".to_string()) {
            "gemm_w4a8_mxfp4_wperm".to_string()
        } else {
            r.candidates[0].clone()
        }
    });
    let rebound = resolved.bind(&other).unwrap();
    assert_eq!(base.body.plan_input_hash, rebound.body.plan_input_hash);
    assert_ne!(base.plan_hash(), rebound.plan_hash());
    // A different winner is also a different measurement; both hashes move.
    assert_ne!(base.measurement_hash(), rebound.measurement_hash());
}

/// The plan hash is over the **decisions**. Re-measuring the same winners must not move it.
///
/// Three identical autotune sweeps on this box produced identical bindings and three different
/// plan hashes while the microseconds were inside the hashed body (`LEDGER.md`, *Autotune sweep*).
/// Anything keyed on the plan hash — `saber_mem::pack`'s `check_plan_hash` refuses a pack outright
/// on a mismatch — would have been invalidated by re-running a benchmark that changed nothing.
#[test]
fn plan_hash_ignores_the_microseconds_but_the_measurement_hash_does_not() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let base = resolved.bind(&cache).unwrap();

    // Same winners and the same variants; different numbers, a different day, a different harness.
    let mut remeasured = AutotuneCache::new();
    for entry in cache.entries() {
        let mut record = entry.record.clone();
        record.measured_us += 0.37;
        record.samples_kept += 1;
        if let Some(runner_up) = record.runner_up.as_mut() {
            runner_up.measured_us += 0.11;
        }
        record.provenance = Provenance::new("someone-else", "2026-09-19", "a different harness");
        remeasured.insert(entry.key.clone(), record);
    }
    let again = resolved.bind(&remeasured).unwrap();

    assert_eq!(
        base.plan_hash(),
        again.plan_hash(),
        "same bindings, so the same plan"
    );
    assert_ne!(
        base.measurement_hash(),
        again.measurement_hash(),
        "different numbers, so a different measurement"
    );
    assert!(base.verify_hash() && again.verify_hash());
    // The numbers are still in the plan; they are simply outside the plan hash. (The promotion
    // record can outrank the sweep's: a row the registry already ships as `default` keeps its own
    // provenance, which is why this asserts on the microseconds and not on `who`.)
    let first = again.buckets()[0].ops.ops()[0];
    assert!((first.measured_us - 31.57).abs() < 1e-9);
    assert!(first.promotion.state >= saber_plan::PromotionState::Measured);
}

#[test]
fn plan_hash_changes_when_any_input_changes() {
    let (manifest, device, model, registry) = qwen_setup();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let base = resolved.bind(&tuned_cache(&resolved)).unwrap();

    let rebuild = |m: &Manifest, d: &DeviceProfile, mo: &ModelSpec, r: &Registry| {
        let resolved = resolve(m, d, mo, r).unwrap();
        let cache = tuned_cache(&resolved);
        resolved.bind(&cache).unwrap()
    };

    // A different tier.
    let mut other_manifest = manifest.clone();
    other_manifest.model.tier_id = "tier:other".to_string();
    let by_manifest = rebuild(&other_manifest, &device, &model, &registry);
    assert_ne!(base.body.plan_input_hash, by_manifest.body.plan_input_hash);
    assert_ne!(base.plan_hash(), by_manifest.plan_hash());

    // A different device profile.
    let mut other_device = device.clone();
    other_device.vram_bytes += 1;
    let by_device = rebuild(&manifest, &other_device, &model, &registry);
    assert_ne!(base.plan_hash(), by_device.plan_hash());

    // A different registry.
    let mut other_registry = registry.clone();
    other_registry.insert(
        Implementation::new("gemm_w4a8_experimental", "mxfp4", "gemm_w4a8")
            .constrain(Constraint::eq("m", -1)),
    );
    let by_registry = rebuild(&manifest, &device, &model, &other_registry);
    assert_ne!(base.plan_hash(), by_registry.plan_hash());

    // A different plugin build.
    let mut other_plugin = manifest.clone();
    other_plugin.plugins[0].build_hash = "deadbeef".to_string();
    let by_plugin = rebuild(&other_plugin, &device, &model, &registry);
    assert_ne!(base.plan_hash(), by_plugin.plan_hash());
}

/// A registry whose only `gemm_w4a8` route declares `may_sync`.
fn unsafe_registry() -> Registry {
    Registry::new().with(
        Implementation::new("gemm_w4a8_shadow_compare", "mxfp4", "gemm_w4a8")
            .describe(
                "the GEMM with a host-visible shadow compare",
                "any",
                "w=mxfp4 a=int8",
            )
            .constrain(Constraint::le("m", 4096))
            .capture(true, true)
            .code_object(32, 1024)
            .launches(LaunchModel::Fixed(1)),
    )
}

#[test]
fn a_may_sync_route_cannot_enter_a_captured_bucket() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = unsafe_registry();

    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let err = resolved.bind(&cache).unwrap_err();
    let PlanError::NotGraphSafe(refusal) = err else {
        panic!("expected a capture refusal")
    };
    assert_eq!(refusal.impl_name, "gemm_w4a8_shadow_compare");
    assert_eq!(refusal.reason, "declared may_sync");
    assert!(
        refusal.to_string().contains("mark the bucket eager"),
        "{refusal}"
    );
}

#[test]
fn the_same_route_is_legal_in_an_eager_bucket() {
    let mut manifest = tiny_manifest();
    manifest.plan.capture = false;
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = unsafe_registry();

    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let plan = resolved.bind(&cache).unwrap();
    assert!(matches!(plan.buckets()[0].ops, BucketOps::Eager(_)));
    assert!(!plan.buckets()[0].bucket.captured);
    assert_eq!(
        plan.buckets()[0].ops.ops()[0].impl_name,
        "gemm_w4a8_shadow_compare"
    );
}

#[test]
fn a_captured_bucket_holds_proved_ops() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let plan = resolved.bind(&tuned_cache(&resolved)).unwrap();
    let BucketOps::Captured(ops) = &plan.buckets()[0].ops else {
        panic!("expected capture")
    };
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].op().op_name, "layer.mlp.down_proj");
    assert!(ops[0].op().graph_safe);
}

#[test]
fn workspaces_are_budgeted_by_kind() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let plan = resolved.bind(&tuned_cache(&resolved)).unwrap();
    let bucket = &plan.buckets()[0];

    // M = batch 2 x spec 2 = 4; split_k 4; partials = split_k * M * N * 4 B.
    let down = bucket
        .ops
        .ops()
        .into_iter()
        .find(|o| o.op_name == "layer.mlp.down_proj")
        .unwrap();
    assert_eq!(down.workspaces[0].kind, WorkspaceKind::Scratch);
    assert_eq!(down.workspaces[0].bytes, 4 * 4 * 256 * 4);
    assert_eq!(down.workspaces[1].kind, WorkspaceKind::Persistent);
    assert_eq!(down.workspaces[1].bytes, 4096);

    let scratch = bucket
        .workspaces
        .iter()
        .find(|w| w.kind == WorkspaceKind::Scratch)
        .unwrap();
    let persistent = bucket
        .workspaces
        .iter()
        .find(|w| w.kind == WorkspaceKind::Persistent)
        .unwrap();
    // Scratch is aliasable, so the bucket needs the largest single request (lm_head, N = 1024).
    assert_eq!(scratch.bytes, 4 * 4 * 1024 * 4);
    assert_eq!(scratch.contributors, 2);
    // Persistent survives replay and is summed.
    assert_eq!(persistent.bytes, 2 * 4096);
}

#[test]
fn a_device_mismatch_is_refused() {
    let (mut manifest, device, model, registry) = qwen_setup();
    manifest.device.target_gfx = "gfx1200".to_string();
    let err = resolve(&manifest, &device, &model, &registry).unwrap_err();
    assert!(matches!(err, PlanError::DeviceMismatch { .. }), "{err}");
}

#[test]
fn a_tp_mismatch_between_manifest_and_model_is_refused() {
    let (manifest, device, _, registry) = qwen_setup();
    let model = ModelSpec::qwen38_27b_dense(1).unwrap();
    let err = resolve(&manifest, &device, &model, &registry).unwrap_err();
    assert!(err.to_string().contains("sharded for tp=1"), "{err}");
}

#[test]
fn a_drafter_from_another_tier_is_refused() {
    let (mut manifest, device, model, registry) = qwen_setup();
    manifest.model.drafter_tier_id = Some("tier:some-other-quantization".to_string());
    let err = resolve(&manifest, &device, &model, &registry).unwrap_err();
    assert!(matches!(err, PlanError::TierMismatch { .. }), "{err}");
    assert!(err.to_string().contains("3.67 tok/draft"), "{err}");

    manifest.model.drafter_tier_id = Some(manifest.model.tier_id.clone());
    assert!(resolve(&manifest, &device, &model, &registry).is_ok());
}

#[test]
fn an_implementation_from_an_undeclared_plugin_is_refused() {
    let mut manifest = tiny_manifest();
    manifest.plugins[0].name = "not-mxfp4".to_string();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let err = resolved.bind(&cache).unwrap_err();
    assert!(matches!(err, PlanError::UnknownPlugin { .. }), "{err}");
}

#[test]
fn an_implementation_the_device_cannot_run_is_refused() {
    let manifest = tiny_manifest();
    let mut device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();

    // A target with no wave32 support cannot run a wave32 code object.
    device.wave_sizes = vec![64];
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let err = resolved.bind(&cache).unwrap_err();
    assert!(err.to_string().contains("compiled for wave32"), "{err}");

    // And an implementation asking for more LDS than the target has.
    let mut small_lds = DeviceProfile::gfx1201_reference();
    small_lds.lds_bytes = 16 * 1024;
    let resolved = resolve(&manifest, &small_lds, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let err = resolved.bind(&cache).unwrap_err();
    assert!(err.to_string().contains("of LDS per workgroup"), "{err}");
}

#[test]
fn overrides_pin_a_route_and_assert_its_variant() {
    let mut manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();

    // Pinning the route the measurement already chose, and asserting its variant, is fine.
    manifest.plan.overrides.insert(
        "lm_head".to_string(),
        saber_plan::PlanOverride {
            impl_name: Some("gemm_w4a8_mxfp4_decode".to_string()),
            variant_id: Some(12),
        },
    );
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    assert!(resolved.bind(&cache).is_ok());

    // An override is part of the manifest, so changing one changes the plan input hash and needs
    // its own records — each case below re-resolves and re-tunes.
    let bind_err = |m: &Manifest| -> PlanError {
        let resolved = resolve(m, &device, &model, &registry).unwrap();
        let cache = tuned_cache(&resolved);
        resolved.bind(&cache).unwrap_err()
    };

    // Asserting a variant the record does not have is an error, not a silent re-pick.
    let mut wrong_variant = manifest.clone();
    wrong_variant
        .plan
        .overrides
        .get_mut("lm_head")
        .unwrap()
        .variant_id = Some(3);
    let err = bind_err(&wrong_variant);
    assert!(err.to_string().contains("is variant 12, not 3"), "{err}");

    // Pinning a candidate that no measurement covers is an error: an override cannot conjure an
    // unmeasured route.
    let mut unmeasured = manifest.clone();
    unmeasured
        .plan
        .overrides
        .get_mut("lm_head")
        .unwrap()
        .impl_name = Some("gemm_w4a8_mxfp4_wperm".to_string());
    unmeasured
        .plan
        .overrides
        .get_mut("lm_head")
        .unwrap()
        .variant_id = None;
    let err = bind_err(&unmeasured);
    assert!(
        err.to_string()
            .contains("re-tune with `gemm_w4a8_mxfp4_wperm`"),
        "{err}"
    );

    // Pinning a route the constraints refuse names the constraint that refused it.
    let mut refused = manifest.clone();
    refused.plan.overrides.get_mut("lm_head").unwrap().impl_name =
        Some("gemm_w4a8_mxfp4_atiled".to_string());
    let err = bind_err(&refused);
    assert!(err.to_string().contains("m=4, needs >= 2048"), "{err}");

    // An override naming an op the model does not have is refused at resolution.
    let mut unknown = tiny_manifest();
    unknown.plan.overrides.insert(
        "layer.mlp.gate_proj".to_string(), // merged away: the row is `layer.mlp.gate_up`
        saber_plan::PlanOverride {
            impl_name: None,
            variant_id: Some(1),
        },
    );
    let err = resolve(&unknown, &device, &model, &registry).unwrap_err();
    assert!(err.to_string().contains("has no op of that name"), "{err}");
}

#[test]
fn a_record_naming_an_implementation_that_no_longer_matches_is_stale() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    // A-tiled does not cover M=4, so a record naming it is stale rather than usable.
    let cache = tuned_cache_with(&resolved, |_| "gemm_w4a8_mxfp4_atiled".to_string());
    let err = resolved.bind(&cache).unwrap_err();
    assert!(matches!(err, PlanError::StaleRecord { .. }), "{err}");

    // As is a record naming an implementation that has left the registry entirely.
    let cache = tuned_cache_with(&resolved, |_| "gemm_w4a8_deleted".to_string());
    let err = resolved.bind(&cache).unwrap_err();
    assert!(matches!(err, PlanError::StaleRecord { .. }), "{err}");
}

#[test]
fn a_promotion_without_provenance_is_refused() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = Registry::new().with(
        Implementation::new("gemm_w4a8_claimed_default", "mxfp4", "gemm_w4a8")
            .constrain(Constraint::le("m", 4096))
            .promoted(saber_plan::Promotion {
                state: saber_plan::PromotionState::Default,
                provenance: None,
            }),
    );
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let err = resolved.bind(&cache).unwrap_err();
    assert!(
        err.to_string().contains("carries no provenance record"),
        "{err}"
    );
}

#[test]
fn gdn_fused_max_items_agrees_with_the_launch_rules() {
    // `GDN_FUSED_MAX_ITEMS` is a saber knob restated in the model, because the model is what knows
    // how many value heads a rank owns. Two files holding one number is the failure this project
    // exists to delete, so the copy is asserted against the source rather than trusted.
    let rules = saber_kernels::saber::SaberRules::prod_saber().expect("prod saber rules parse");
    assert_eq!(
        rules.gdn.fused_max_items,
        saber_plan::GDN_FUSED_MAX_ITEMS,
        "rules/saber_gfx1201.toml [gdn] fused_max_items moved; ModelSpec routes the decode step on it"
    );
}

#[test]
fn ar_quant_min_bytes_agrees_with_the_launch_rules() {
    // Same contract as `gdn_fused_max_items_agrees_with_the_launch_rules`: the threshold lives in
    // the rules file and is restated in the model, because the model is what knows the message
    // size. Asserting the copy is what makes "a rules edit moves the band" true.
    let rules = saber_kernels::saber::SaberRules::prod_saber().expect("prod saber rules parse");
    assert_eq!(
        rules.ar.quant_min_bytes,
        saber_plan::AR_QUANT_MIN_BYTES,
        "rules/saber_gfx1201.toml [ar] quant_min_bytes moved; ModelSpec bands the all-reduce on it"
    );
    // And the band it produces for this checkpoint, spelled out once so the arithmetic is visible:
    // 5120 hidden x 2 B = 10240 B per row, 131072 / 10240 -> m <= 12 exact, m >= 13 quantised.
    assert_eq!(saber_plan::AR_PAYLOAD_BYTES, 2);
    let per_row = 5120 * saber_plan::AR_PAYLOAD_BYTES;
    assert_eq!((saber_plan::AR_QUANT_MIN_BYTES - 1) / per_row, 12);
}

/// `q_len` and `max_ctx` come from the bucket, because an autotune record is a measurement of the
/// geometry its key names.
#[test]
fn the_attention_key_carries_the_bucket_not_the_worst_case() {
    let model = ModelSpec::qwen38_27b_dense(2).unwrap();
    let attn = model
        .op("layer.attn.paged_decode")
        .expect("the paged attention decode row");

    // The op itself declares neither: they are not properties of the op.
    assert!(!attn.extra.contains_key("q_len"));
    assert!(!attn.extra.contains_key("max_ctx"));

    // b1s5 at an 8k band.
    let keys = attn.shape_keys(Phase::Decode, 5, 1, 5, 8192);
    assert_eq!(keys.num("q_len"), Some(5), "the width actually verified");
    assert_eq!(keys.num("max_ctx"), Some(8192), "the bucket's context band");

    // A wider verify window is a different measurement...
    let wider = attn.shape_keys(Phase::Decode, 8, 1, 8, 8192);
    assert_eq!(wider.num("q_len"), Some(8));
    assert_ne!(keys, wider);

    // ...and so is a different context band, which sets the split count and therefore the cost.
    let deeper = attn.shape_keys(Phase::Decode, 5, 1, 5, 32768);
    assert_eq!(deeper.num("max_ctx"), Some(32768));
    assert_ne!(keys, deeper);

    // The registry's necessary condition is untouched: q_len <= 10 still admits both.
    let registry = saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
    for k in [&keys, &wider] {
        assert!(
            registry.select("attn_decode_paged", k).is_ok(),
            "q_len <= 10 is a necessary condition on the kernel, not the served width"
        );
    }

    // Moving a bucket's band moves that bucket's attention key, and only that one: two bands are
    // two measurements. Before this change the band was nowhere in the key and one record would
    // have covered both.
    let (manifest, device, _, reg) = qwen_setup();
    // (The key also carries the plan INPUT hash, which covers the manifest, so editing the
    // manifest moves every key. The geometry is what is asserted here.)
    let attn_shapes = |m: &Manifest| -> Vec<saber_plan::ShapeKeys> {
        let resolved = resolve(m, &device, &model, &reg).unwrap();
        resolved
            .requirements()
            .iter()
            .filter(|r| r.key.op == "attn_decode_paged")
            .map(|r| r.key.shape.clone())
            .collect()
    };
    let before = attn_shapes(&manifest);
    let mut two_bands = manifest.clone();
    two_bands.buckets[1].ctx_band = 32768;
    let after = attn_shapes(&two_bands);
    assert_eq!(before.len(), after.len());
    assert_eq!(
        before[0], after[0],
        "the bucket that did not move keeps its geometry"
    );
    assert_ne!(
        before[1], after[1],
        "the one that moved bands is a new measurement"
    );
    assert_eq!(after[1].num("max_ctx"), Some(32768));
}

/// The split-KV scratch allocates its capacity and reports the law beside it. Changing which of
/// the two `bytes()` returns would be a policy change; this pins that it is the capacity.
#[test]
fn the_split_kv_scratch_allocates_the_capacity_and_reports_the_law() {
    use saber_plan::{ShapeKeys, WorkspaceSizing};
    let sizing = WorkspaceSizing::SplitKvPartials {
        capacity: 63_897_600,
        bytes_per_split: 520,
    };
    // b1s5: 5 tokens x 12 q heads x 64 splits x 520 B.
    let shape = ShapeKeys::new().with("m", 5i64).with("n", 12i64);
    assert_eq!(
        sizing.bytes(&shape, 1),
        63_897_600,
        "the allocation is the capacity"
    );
    assert_eq!(sizing.used_bytes(&shape, Some(64)), Some(5 * 12 * 64 * 520));
    // Half the splits at twice the batch is the same working set, which is why the ratio is flat
    // across this bucket set.
    let b8 = ShapeKeys::new().with("m", 40i64).with("n", 12i64);
    assert_eq!(sizing.used_bytes(&b8, Some(16)), Some(40 * 12 * 16 * 520));
    // A variant with no split count says nothing rather than guessing.
    assert_eq!(sizing.used_bytes(&shape, None), None);
    // And a sizing with no per-bucket form has no law to report.
    assert_eq!(WorkspaceSizing::Fixed(4).used_bytes(&shape, Some(64)), None);
}

/// The plan input hash keys on the rows a plan can BIND, not on the registry.
///
/// Every autotune key carries the input hash, so anything inside it invalidates every record when
/// it changes. On 2026-09-19 eight Paro rows — a plugin for a different tier, which no op of
/// an NVFP4 model can select — moved every key and made a freshly measured cache stale: a
/// 44-shape re-sweep to describe a change that could not reach any of them.
#[test]
fn a_row_no_model_op_can_bind_does_not_move_the_input_hash() {
    let (manifest, device, model, registry) = qwen_setup();
    let base = resolve(&manifest, &device, &model, &registry)
        .unwrap()
        .input_hash()
        .to_string();

    // A row for another tier: a different op name, so no model op ever selects it.
    let mut other_tier = registry.clone();
    other_tier.insert(
        Implementation::new("paro_rotate_quant", "paro", "rotate_quant")
            .constrain(Constraint::ge("m", 1))
            .constrain(Constraint::div("k", 128)),
    );
    let with_row = resolve(&manifest, &device, &model, &other_tier).unwrap();
    assert_eq!(
        with_row.input_hash(),
        base,
        "a row nothing can bind cannot change what any record means"
    );
    // It is in the registry, and `explain` still prints it: hash-neutral is not invisible.
    assert_ne!(
        other_tier.fingerprint(),
        registry.fingerprint(),
        "the registry itself did change"
    );

    // A row for an op the model DOES have, but at a geometry it never resolves at, is equally
    // unbindable and equally neutral.
    let mut unreachable = registry.clone();
    unreachable.insert(
        Implementation::new("gemm_w4a8_impossible", "mxfp4", "gemm_w4a8")
            .constrain(Constraint::eq("m", -1)),
    );
    assert_eq!(
        resolve(&manifest, &device, &model, &unreachable)
            .unwrap()
            .input_hash(),
        base,
        "a candidate set that never contains it is a row that cannot bind"
    );
}

/// The tier's launch POLICY is an input to the plan, so editing it invalidates the cache.
///
/// A registry row carries the band EDGES it reads out of the rules (`m <= decode_max_m`,
/// `k % 64`), never the bands themselves — so before 2026-09-19 a rules edit that changed which
/// kernel a shape takes moved nothing the autotune key could see. That is not academic: I12 made
/// `lm_head`'s decode tile legal and the committed cache went on binding the folded record it had
/// measured when the decode tile was not, and only a coincidence (the same knob is also a registry
/// constraint) made the guard notice. The fifth component of the input hash is the digest of the
/// committed rule tables.
/// Every route's key digest, by op name, for the policy tests below.
fn key_digests(
    manifest: &Manifest,
    device: &DeviceProfile,
    model: &ModelSpec,
    registry: &Registry,
) -> std::collections::BTreeMap<String, String> {
    resolve(manifest, device, model, registry)
        .unwrap()
        .requirements()
        .into_iter()
        .map(|r| {
            (
                format!("{} {}", r.bucket.label(), r.op_name),
                r.key.digest(),
            )
        })
        .collect()
}

#[test]
fn editing_the_launch_policy_invalidates_every_record_it_can_reach() {
    let (manifest, device, model, registry) = qwen_setup();
    let base = key_digests(&manifest, &device, &model, &registry);

    // The same registry, cut from a policy one byte different. Nothing else moves: the rows, the
    // constraints, the workspaces and the model are identical. This registry is hand-built and
    // carries no per-table policy, so every route falls back to the whole-tier digest -- which is
    // the conservative direction and is what makes this the "reaches everything" case.
    let mut edited = registry.clone();
    edited.set_policy_digest("a-policy-with-one-band-moved");
    let after = key_digests(&manifest, &device, &model, &edited);
    assert_eq!(base.len(), after.len());
    assert!(
        base.iter().all(|(k, v)| after[k] != *v),
        "a policy edit must move the keys it can reach, or the records outlive the question \
         they answered"
    );

    // ... and the same policy hashes the same, so a re-run of the same tree is still cache-warm.
    let mut same = registry.clone();
    same.set_policy_digest(registry.policy_digest());
    assert_eq!(key_digests(&manifest, &device, &model, &same), base);

    // The plan INPUT hash no longer carries the policy at all (I18): it is per key, so that an
    // edit to one table does not re-key the routes no row of it could bind. This is the
    // difference stated as an assertion rather than left to the comment above.
    assert_eq!(
        resolve(&manifest, &device, &model, &edited)
            .unwrap()
            .input_hash(),
        resolve(&manifest, &device, &model, &registry)
            .unwrap()
            .input_hash(),
        "the policy belongs to the KEY now, not to the hash every key shares"
    );
}

/// NEGATIVE CONTROL for the per-plugin digest: editing `attn_glue`'s table re-keys the glue
/// routes and **nothing else**.
///
/// With one digest over all seven tables this test fails on its second assertion: every route's
/// key moves, which is why a one-line change to any table used to cost a full 85-key sweep. The
/// registry has to be the REAL one — the reference registry's plugins are stand-in names
/// (`mxfp4`, `saber`) that no committed table claims, and those fall back to the whole-tier
/// digest by design.
#[test]
fn a_policy_edit_re_keys_only_the_routes_that_table_can_reach() {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let registry =
        saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir()).expect("registry");
    let base = key_digests(&manifest, &device, &model, &registry);

    // One byte into `attn_glue_gfx1201.toml`, nothing else touched.
    let owned: Vec<(String, String)> = saber_kernels::committed_rule_tables()
        .iter()
        .map(|(n, t)| {
            let text = if *n == "attn_glue_gfx1201.toml" {
                format!("{t}\n# one byte\n")
            } else {
                (*t).to_owned()
            };
            ((*n).to_owned(), text)
        })
        .collect();
    let borrowed: Vec<(&str, &str)> = owned
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let mut edited = registry.clone();
    edited.set_policy_tables(&borrowed);
    let after = key_digests(&manifest, &device, &model, &edited);

    // The MODEL op names, which is what `requirements()` reports: `layer.attn.gate_quant`,
    // not the kernel-plane `attn_gate_quant`.
    let glue = ["qk_norm_rope", "gate_quant", "kv_append"];
    let (mut moved, mut still) = (0usize, 0usize);
    for (route, digest) in &base {
        let is_glue = glue.iter().any(|g| route.contains(g));
        if is_glue {
            assert_ne!(after[route], *digest, "{route} must be re-keyed");
            moved += 1;
        } else {
            assert_eq!(
                after[route], *digest,
                "{route} cannot bind an attn_glue row and must NOT be re-keyed"
            );
            still += 1;
        }
    }
    // Both sides have to be non-trivial or the test proves nothing: all-moved and none-moved
    // are exactly the two failures it exists to tell apart. (Three glue ops per bucket; the
    // example manifest's bucket list decides how many that is.)
    assert!(moved >= 12, "only {moved} glue route(s) were re-keyed");
    assert_eq!(moved % 3, 0, "three glue ops per bucket, got {moved}");
    assert!(still > 40, "only {still} route(s) held still");
}

/// The digest is the committed tables, not a constant: a registry derived from the shipped rules
/// carries one, and a registry built by hand in a test carries none (and says so in `explain`).
#[test]
fn the_policy_digest_is_the_committed_rule_tables() {
    let derived =
        saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir()).expect("registry");
    assert_eq!(
        derived.policy_digest(),
        saber_plan::hash::digest_of(&saber_kernels::committed_rule_tables()),
        "the digest must be OF the tables, so editing one is what changes it"
    );
    assert!(!derived.policy_digest().is_empty());
    assert!(
        saber_plan::Registry::new().policy_digest().is_empty(),
        "a hand-built registry has no policy behind it and must not pretend to"
    );
}

/// One byte of ANY committed rules table moves the digest — asserted over every table, not over
/// the one somebody remembered to check.
///
/// `saber-kernels` holds the list against its own `rules/` directory; this is the other half of
/// the same claim, and it has to live here because `digest_of` is sha256 over canonical JSON and
/// a hand-rolled copy of it in another crate would be testing the copy.
#[test]
fn one_byte_of_any_rules_table_moves_the_policy_digest() {
    use saber_plan::hash::digest_of;
    let tables = saber_kernels::committed_rule_tables();
    let base = digest_of(&tables);
    for i in 0..tables.len() {
        let mut edited: Vec<(String, String)> = tables
            .iter()
            .map(|(n, t)| ((*n).to_owned(), (*t).to_owned()))
            .collect();
        edited[i].1.push_str("\n# one byte\n");
        assert_ne!(
            digest_of(&edited),
            base,
            "editing {} moves no hash, so every record measured under the old {} survives it",
            tables[i].0,
            tables[i].0
        );
    }
    // And the digest is over the NAMES too: two tables that swapped contents are a different
    // policy, not the same one.
    let mut swapped: Vec<(String, String)> = tables
        .iter()
        .map(|(n, t)| ((*n).to_owned(), (*t).to_owned()))
        .collect();
    let first = swapped[0].1.clone();
    swapped[0].1 = swapped[1].1.clone();
    swapped[1].1 = first;
    assert_ne!(digest_of(&swapped), base);
}

/// The incremental re-sweep, measured where it is spent: `Resolved::missing` after a one-table
/// edit is the work list `saber tune --autotune-in` will run.
///
/// This is the same claim as `a_policy_edit_re_keys_only_the_routes_that_table_can_reach`, stated
/// against the thing the GPU time is proportional to. With one digest for the whole tier the
/// answer is "all of them" for any edit to any table, which is what made a one-line change to
/// `dflash_gfx1201.toml` -- whose rows no bound route can even reach today -- cost a full sweep
/// the day I17 landed.
#[test]
fn a_one_table_edit_leaves_only_that_tables_routes_to_re_measure() {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let registry =
        saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir()).expect("registry");

    // A cache that answers every route, as a finished sweep leaves it.
    let resolved = resolve(&manifest, &device, &model, &registry).expect("resolve");
    let mut cache = saber_plan::AutotuneCache::new();
    for r in resolved.requirements() {
        cache.insert(
            r.key.clone(),
            saber_plan::AutotuneRecord::new(
                &r.candidates[0],
                saber_plan::VariantParams::new(1),
                12.5,
                saber_plan::Provenance::new("fixture", "2026-09-20", "none"),
            ),
        );
    }
    assert!(resolved.missing(&cache).is_empty(), "the cache is warm");

    let edit = |file: &str| {
        let owned: Vec<(String, String)> = saber_kernels::committed_rule_tables()
            .iter()
            .map(|(n, t)| {
                let text = if *n == file {
                    format!("{t}\n# one byte\n")
                } else {
                    (*t).to_owned()
                };
                ((*n).to_owned(), text)
            })
            .collect();
        let borrowed: Vec<(&str, &str)> = owned
            .iter()
            .map(|(n, t)| (n.as_str(), t.as_str()))
            .collect();
        let mut reg = registry.clone();
        reg.set_policy_tables(&borrowed);
        resolve(&manifest, &device, &model, &reg)
            .expect("resolve")
            .missing(&cache)
            .into_iter()
            .map(|r| r.op_name)
            .collect::<Vec<_>>()
    };

    // `dflash`: the drafter tier binds nothing yet, so an edit to it costs NOTHING to re-measure.
    // This is the case that cost a full 85-key sweep on main, and it is the sharpest statement of
    // what the per-route digest buys.
    assert!(
        edit("dflash_gfx1201.toml").is_empty(),
        "a dflash edit re-sweeps dflash routes, and no route binds one"
    );

    // `attn_glue`: its three ops and nothing else.
    let glue = edit("attn_glue_gfx1201.toml");
    assert!(!glue.is_empty());
    for op in &glue {
        assert!(
            op.contains("qk_norm_rope") || op.contains("gate_quant") || op.contains("kv_append"),
            "{op} is not a glue route"
        );
    }
    // And the linears are untouched, which is the whole saving.
    let all = resolved.requirements().len();
    assert!(
        glue.len() * 3 < all,
        "{} of {all} route(s) to re-measure is not much of a saving",
        glue.len()
    );

    // `mxfp4`: the linears, and not the glue.
    let mx = edit("mxfp4_gfx1201.toml");
    assert!(mx.len() > glue.len());
    assert!(
        !mx.iter().any(|op| op.contains("qk_norm_rope")),
        "an mxfp4 edit cannot reach the rope"
    );
}

/// The other half: a row that IS a candidate is part of what a record means, so changing it moves
/// the hash and the records go with it.
#[test]
fn changing_a_candidate_row_moves_the_input_hash() {
    let (manifest, device, model, registry) = qwen_setup();
    let base = resolve(&manifest, &device, &model, &registry)
        .unwrap()
        .input_hash()
        .to_string();

    // Rebuild the registry with one candidate row edited in place -- `insert` appends, so a
    // second row of the same name would never be the one `select` returns.
    let edited = |extra: Constraint| -> Registry {
        let mut out = Registry::new();
        for imp in registry.implementations() {
            if imp.name == "gemm_w4a8_mxfp4_decode" {
                out.insert(imp.clone().constrain(extra.clone()));
            } else {
                out.insert(imp.clone());
            }
        }
        out
    };

    // Tighten a constraint on a row the model binds. It stays a candidate here, but what it
    // claims about the geometry changed.
    let tightened = edited(Constraint::div("n", 32));
    assert_ne!(
        resolve(&manifest, &device, &model, &tightened)
            .unwrap()
            .input_hash(),
        base,
        "a candidate's constraints are part of what its record means"
    );

    // And a constraint that removes it from the candidate set entirely: the SET changed.
    let removed = edited(Constraint::le("m", 0));
    assert_ne!(
        resolve(&manifest, &device, &model, &removed)
            .unwrap()
            .input_hash(),
        base,
        "a row that stops being a candidate changes the set it was in"
    );
}

/// A row that only becomes reachable later still moves the hash the day it does.
#[test]
fn a_row_that_becomes_a_candidate_moves_the_hash_then() {
    let (manifest, device, model, registry) = qwen_setup();
    let base = resolve(&manifest, &device, &model, &registry)
        .unwrap()
        .input_hash()
        .to_string();

    // Unreachable at first: no bucket has M above 10_000.
    let row = Implementation::new("gemm_w4a8_huge_m", "mxfp4", "gemm_w4a8")
        .constrain(Constraint::ge("m", 10_000));
    let mut later = registry.clone();
    later.insert(row.clone());
    assert_eq!(
        resolve(&manifest, &device, &model, &later)
            .unwrap()
            .input_hash(),
        base
    );

    // Widen it until a bucket's M selects it, and the hash moves.
    let mut now = registry.clone();
    now.insert(
        Implementation::new("gemm_w4a8_reachable", "mxfp4", "gemm_w4a8")
            .constrain(Constraint::ge("m", 1)),
    );
    assert_ne!(
        resolve(&manifest, &device, &model, &now)
            .unwrap()
            .input_hash(),
        base
    );
}

// ---------------------------------------------------------------- the incremental merge (I19)

/// A registry cut from the committed tables with ONE table's text one byte different — a rules
/// edit, exactly as `a_policy_edit_re_keys_only_the_routes_that_table_can_reach` makes one.
fn registry_with_table_edited(base: &Registry, file: &str) -> Registry {
    let owned: Vec<(String, String)> = saber_kernels::committed_rule_tables()
        .iter()
        .map(|(n, t)| {
            let text = if *n == file {
                format!("{t}\n# one row moved\n")
            } else {
                (*t).to_owned()
            };
            ((*n).to_owned(), text)
        })
        .collect();
    let borrowed: Vec<(&str, &str)> = owned
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let mut edited = base.clone();
    edited.set_policy_tables(&borrowed);
    edited
}

/// Every route answered, as a finished sweep leaves the file. One op's cells carry an `unstable`
/// pair, because that is a thing a merge can quietly lose.
fn swept_cache(resolved: &saber_plan::Resolved<'_>, unstable_op: &str) -> AutotuneCache {
    let mut cache = AutotuneCache::new();
    for (i, r) in resolved.requirements().iter().enumerate() {
        let mut record = saber_plan::AutotuneRecord::new(
            &r.candidates[0],
            saber_plan::VariantParams::new(1),
            12.5 + i as f64,
            Provenance::new("saber tune", "2026-09-20", "fixture"),
        );
        if r.key.op == unstable_op {
            record.unstable = Some(saber_plan::Unstable {
                first: saber_plan::RunnerUp {
                    impl_name: r.candidates[0].clone(),
                    variant_id: 1,
                    measured_us: 81.32,
                },
                second: saber_plan::RunnerUp {
                    impl_name: r.candidates[0].clone(),
                    variant_id: 2,
                    measured_us: 73.96,
                },
                worst_swing_pct: 9.0,
            });
        }
        cache.insert(r.key.clone(), record);
    }
    cache
}

#[test]
fn mixed_widths_keep_separate_policy_families_but_reject_stale_wide_records() {
    let mut manifest = Manifest::from_str(
        include_str!("../../../examples/qwen38-27b-nvfp4-tp2-w16.toml"),
        "mixed-widths",
    )
    .unwrap();
    let mut narrow = manifest.buckets[0].clone();
    narrow.spec_width = 5;
    manifest.buckets.push(narrow);
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(2).unwrap();
    let registry = saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let mut fixture = swept_cache(&resolved, "");
    fixture.one_policy_digest_per_op().unwrap();
    let policies = fixture.policy_digests_by_op();
    assert_ne!(
        policies["attn_decode_paged"],
        policies["attn_decode_paged[uniform16]"]
    );
    let entry = fixture
        .entries()
        .find(|e| e.record.impl_name.starts_with("attn_decode_uniform16"))
        .unwrap()
        .clone();
    let mut stale = entry.key;
    stale.policy_digest.push_str("-previous");
    fixture.insert(stale, entry.record);
    assert_eq!(
        fixture.one_policy_digest_per_op().unwrap_err().op,
        "attn_decode_paged[uniform16]"
    );
}

/// Records per op, which is what a census over the cache reads and what a merge can silently
/// double.
fn counts_by_op(cache: &AutotuneCache) -> std::collections::BTreeMap<String, usize> {
    let mut out: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for e in cache.entries() {
        *out.entry(e.key.op.clone()).or_default() += 1;
    }
    out
}

/// **The bug, and the fix.** After a one-table edit the incremental sweep must land on exactly the
/// file a full sweep would write: the same keys, one policy digest per op, the same per-op counts.
///
/// I18 made a rules edit cost only the routes that table can reach, and `--autotune-in` the
/// default. What it left open was the merge: the old records stayed in the output under the OLD
/// key, so the op carried a record under each policy. On 2026-09-21 the CI guard caught it as
/// `attn_gate_quant carries 2 policy digests` after F's rope row re-keyed the glue routes, and the
/// workaround was the full sweep the incremental path exists to avoid.
///
/// The control is the first half of this test: the union — what the sweep wrote before
/// [`saber_plan::Resolved::carry_forward`] existed — is built here and asserted to FAIL the very
/// same guard, by name. Without it the test would pass against a merge that dropped nothing,
/// because a correct merge and a broken one differ only in what is left behind.
#[test]
fn an_incremental_merge_lands_where_a_full_sweep_would() {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let before = saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir())
        .expect("the committed registry");

    // Digest A: a warm cache over the committed tables, with one op's cells not reproducible.
    let resolved_a = resolve(&manifest, &device, &model, &before).expect("resolve");
    let cache_a = swept_cache(&resolved_a, "attn_decode_paged");
    assert!(resolved_a.missing(&cache_a).is_empty(), "the cache is warm");
    cache_a
        .one_policy_digest_per_op()
        .unwrap_or_else(|split| panic!("the fixture is already split: {split}"));

    // Digest B: one row of `attn_glue_gfx1201.toml` moved, which is F's rope row.
    let after = registry_with_table_edited(&before, "attn_glue_gfx1201.toml");
    let resolved_b = resolve(&manifest, &device, &model, &after).expect("resolve");
    let re_keyed = resolved_b.missing(&cache_a);
    assert!(
        !re_keyed.is_empty(),
        "an attn_glue edit must re-key the glue routes, or this test measures nothing"
    );

    // ---- the control: the union, which is what `saber tune` wrote before this fix ------------
    let measured = |r: &saber_plan::Requirement| {
        saber_plan::AutotuneRecord::new(
            &r.candidates[0],
            saber_plan::VariantParams::new(1),
            11.0,
            Provenance::new("saber tune", "2026-09-21", "fixture"),
        )
    };
    let mut union = cache_a.clone();
    for r in &re_keyed {
        union.insert(r.key.clone(), measured(r));
    }
    let split = union
        .one_policy_digest_per_op()
        .expect_err("the union MUST fail the guard — that is the bug this test is about");
    assert_eq!(split.digests.len(), 2, "{split}");
    assert!(
        ["qk_norm_rope", "attn_gate_quant", "kv_append"].contains(&split.op.as_str()),
        "the split op must be one an attn_glue edit reaches, got {}",
        split.op
    );

    // ---- the fix: carry forward onto THIS resolution's keys, then measure what is missing ----
    let (mut merged, stale) = resolved_b.carry_forward(&cache_a);
    assert_eq!(
        stale.len(),
        re_keyed.len(),
        "exactly the re-keyed routes' old records are dropped"
    );
    assert!(
        stale
            .iter()
            .all(|s| matches!(s.why, saber_plan::StaleWhy::Superseded(_))),
        "a re-keyed route is superseded, not absent: {stale:?}"
    );
    for r in &re_keyed {
        merged.insert(r.key.clone(), measured(r));
    }

    // THE guard, the same function the CI check and `saber tune` itself call.
    merged
        .one_policy_digest_per_op()
        .unwrap_or_else(|split| panic!("{split}"));

    // ... and the stronger claim the guard cannot make on its own: the incremental file IS the
    // full-sweep file. A full sweep starts from nothing, so this is what it would have written.
    let full = swept_cache(&resolved_b, "attn_decode_paged");
    let keys = |c: &AutotuneCache| -> Vec<String> {
        let mut v: Vec<String> = c.entries().map(|e| e.key.digest()).collect();
        v.sort();
        v
    };
    assert_eq!(keys(&merged), keys(&full), "same key set as `--full`");
    assert_eq!(merged.len(), full.len(), "same record count as `--full`");
    assert_eq!(
        counts_by_op(&merged),
        counts_by_op(&full),
        "same per-op record counts as `--full`"
    );
    let digests = |c: &AutotuneCache| -> std::collections::BTreeSet<String> {
        c.entries().map(|e| e.key.policy_digest.clone()).collect()
    };
    assert_eq!(digests(&merged), digests(&full), "same policy digest set");
    assert!(
        digests(&merged).len() >= 3,
        "the per-plugin split is what makes the incremental path worth having"
    );

    // The carried half is not re-measured and not rewritten: the `unstable` pair, the median and
    // the provenance of a route the edit did not reach are the ones the old sweep wrote.
    let unstable: Vec<&saber_plan::CacheEntry> = merged
        .entries()
        .filter(|e| e.record.unstable.is_some())
        .collect();
    assert!(
        !unstable.is_empty(),
        "the `unstable` cells did not survive the merge"
    );
    for e in &unstable {
        assert_eq!(e.key.op, "attn_decode_paged");
        assert_eq!(
            e.record.provenance.when, "2026-09-20",
            "carried, not re-run"
        );
        let u = e.record.unstable.as_ref().expect("filtered");
        assert!((u.worst_swing_pct - 9.0).abs() < f64::EPSILON);
    }
    // And it answers every route of the plan it was carried onto, which is the only thing a cache
    // is for. (`bind` itself needs the manifest's declared plugin set, which the example manifest
    // does not share with the prebuilt registry; `the_committed_cache_binds_the_committed_manifest`
    // in `saber-cli` is where binding is asserted, over the committed artifacts.)
    assert!(
        resolved_b.missing(&merged).is_empty(),
        "the merged cache leaves routes untuned"
    );
}

/// A record for a route this plan does not have is dropped too, and says so with a different
/// reason — a full sweep would not have written it either.
///
/// This is the other half of "the same file a full sweep would write". A cache fed in from a
/// different manifest (or from before a route was removed) is not an error, but keeping its
/// records means the file states two plans' worth of measurements and every census over it counts
/// both.
#[test]
fn a_record_no_route_asks_for_is_dropped_as_not_required() {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let registry = saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir())
        .expect("the committed registry");
    let resolved = resolve(&manifest, &device, &model, &registry).expect("resolve");

    let mut cache = swept_cache(&resolved, "attn_decode_paged");
    let warm = cache.len();
    // A shape no bucket asks for, on an op the model does name.
    let orphan = saber_plan::AutotuneKey::new(
        "gemm_w4a8",
        saber_plan::ShapeKeys::new()
            .with("m", saber_plan::ShapeValue::Num(4097))
            .with("n", saber_plan::ShapeValue::Num(11))
            .with("k", saber_plan::ShapeValue::Num(13)),
        210,
        "a-plan-that-was",
        "plugins-that-were",
        "a-policy-that-was",
    )
    .expect("key");
    cache.insert(
        orphan.clone(),
        saber_plan::AutotuneRecord::new(
            "gemm_w4a8_mxfp4_decode",
            saber_plan::VariantParams::new(1),
            99.0,
            Provenance::new("saber tune", "2026-09-01", "fixture"),
        ),
    );
    assert_eq!(cache.len(), warm + 1);

    let (merged, stale) = resolved.carry_forward(&cache);
    assert_eq!(merged.len(), warm, "the orphan is gone, the rest is kept");
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].key, orphan);
    assert!(matches!(stale[0].why, saber_plan::StaleWhy::NotRequired));
    assert!(
        format!("{}", stale[0]).contains("not a route in this plan"),
        "{}",
        stale[0]
    );
    // Nothing else moved: a merge that drops one record must drop exactly one.
    assert!(resolved.missing(&merged).is_empty());
}

/// Carrying a cache forward onto the resolution that wrote it is the identity — the warm case,
/// which is the one that must not cost a re-sweep.
#[test]
fn carrying_a_cache_onto_its_own_resolution_changes_nothing() {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let registry = saber_plan::registry_from_prebuilt(&saber_kernels::prebuilt::dir())
        .expect("the committed registry");
    let resolved = resolve(&manifest, &device, &model, &registry).expect("resolve");
    let cache = swept_cache(&resolved, "attn_decode_paged");

    let (merged, stale) = resolved.carry_forward(&cache);
    assert!(stale.is_empty(), "{stale:?}");
    assert_eq!(merged, cache, "record for record the same cache");
    assert!(
        resolved.missing(&merged).is_empty(),
        "nothing to re-measure"
    );
}
