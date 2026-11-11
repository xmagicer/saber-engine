//! `explain`: the plan dump `saber explain` prints, held to a snapshot.

mod common;

use common::{tiny_manifest, tiny_model, tuned_cache};
use saber_plan::{
    explain, explain_missing, human_bytes, plan, reference, resolve, AutotuneCache, Constraint,
    DeviceProfile, Implementation, Manifest, ModelSpec, Registry, TieBreak,
};

/// The snapshot. Regenerate by running this test: on a mismatch it writes
/// `tests/data/explain_small.actual.txt`, which you diff and then move over the fixture.
const SNAPSHOT: &str = include_str!("data/explain_small.txt");

fn tiny_plan() -> saber_plan::ExecutionPlan {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    resolved.bind(&cache).unwrap()
}

#[test]
fn explain_matches_the_snapshot() {
    let actual = explain(&tiny_plan());
    if actual != SNAPSHOT {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/explain_small.actual.txt"
        );
        std::fs::write(path, &actual).expect("writing the actual output");
        panic!("explain output changed; diff {path} against tests/data/explain_small.txt");
    }
}

#[test]
fn explain_is_deterministic() {
    assert_eq!(explain(&tiny_plan()), explain(&tiny_plan()));
}

#[test]
fn explain_prints_everything_a_reader_needs() {
    let manifest = Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").unwrap();
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(2).unwrap();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let plan = plan(&manifest, &device, &model, &registry, &cache).unwrap();
    let text = explain(&plan);

    // Identity.
    assert!(
        text.contains("tier:qwen38-27b-nvfp4-mxfp4-dflash2-v22.3"),
        "{text}"
    );
    assert!(text.contains(plan.plan_hash()), "the plan hash");
    assert!(text.contains(&plan.body.plan_input_hash), "the input hash");
    assert!(text.contains("qwen3.8-27b-dense-tp2  tp=2"), "{text}");
    assert!(text.contains("gfx1201  power cap 210 W"), "{text}");
    assert!(text.contains("plugin         mxfp4 @ plugins/gfx1201/mxfp4.hsaco (build b7f3a1c0)"));

    // Per bucket: capture state, static launch count, the IC budget.
    assert!(
        text.contains("bucket b8s5c8192  decode  batch 8  spec 5  ctx_band 8192  M=40  CAPTURED")
    );
    // 610 GEMM launches (305 merged GEMMs, two each) + the reference's 32 attention (two each)
    // + 48 fused GDN + 128 all-reduce + the attention glue's 48 (3 ops x 16 layers), which
    // took the declared budget from 818 to 866 when the `attn_glue` plugin landed.
    assert!(
        text.contains("866 per step (declared statically, asserted at capture)"),
        "{text}"
    );
    assert!(text.contains("60.00 MiB touched per layer = 93.8% of the 64.00 MiB Infinity Cache"));
    assert!(text.contains("2.81 GiB resident"), "{text}");

    // Per op: impl, variant, promotion state with provenance, capture bits, workspaces, and why.
    assert!(
        text.contains("layer.mlp.down_proj  x64  gemm_w4a8"),
        "{text}"
    );
    assert!(text.contains("impl       gemm_w4a8_mxfp4_decode (plugin mxfp4, wave32)"));
    assert!(
        text.contains("variant 12 (split_k=4, tm=4)   31.20 us   margin +4.2% over the runner-up")
    );
    assert!(text
        .contains("state      default (who=maintainer when=2026-09-15 harness=bench_decode_ctx)"));
    assert!(text.contains("capture    graph_safe=yes may_sync=no   launches 2 each, 128 per step"));
    assert!(
        text.contains("workspace  scratch              3.12 MiB  \"split-K partials\""),
        "{text}"
    );
    assert!(
        text.contains("workspace  persistent           4.00 KiB  \"split-K block counter\""),
        "{text}"
    );
    assert!(text.contains("+ gemm_w4a8_mxfp4_decode"), "{text}");
    assert!(
        text.contains("- gemm_w4a8_mxfp4_atiled           m=40, needs >= 2048"),
        "{text}"
    );
    // A decode GEMM: decode, wperm and folded all match a row-major M=40 edge. (This read "2
    // candidates" until G2: that line was `gdn.out_proj` at p2048, row-major under the
    // reference's law; with the tiled GDN norm it is A-tiled and has one candidate.)
    assert!(
        text.contains(
            "3 candidates matched; the tie was broken by measurement, not by a cost model"
        ),
        "{text}"
    );
    assert!(
        text.contains("one candidate matched; the variant came from the autotune cache"),
        "{text}"
    );
}

/// A tie says the margin decided nothing and names what did. Without it a reader sees `+0.4%` and
/// concludes the sweep measured a difference that it did not.
#[test]
fn explain_names_what_broke_a_below_floor_tie() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();

    let mut cache = tuned_cache(&resolved);
    for r in resolved.requirements() {
        let mut record = cache.lookup(&r.key).unwrap().clone();
        // The runner-up measured FASTER, by less than the floor: the tie went the other way.
        record.runner_up.as_mut().unwrap().measured_us = 31.0;
        record.provenance = record
            .provenance
            .clone()
            .with_tie_break(Some(TieBreak::ReferenceLauncher));
        cache.insert(r.key.clone(), record);
    }
    let plan = resolved.bind(&cache).unwrap();
    let text = explain(&plan);
    assert!(
        text.contains("tie broken by the reference launcher"),
        "{text}"
    );
    assert!(text.contains("noise floor"), "{text}");
    // And the hash the plan is keyed on is printed next to the one the numbers are keyed on.
    assert!(text.contains(plan.plan_hash()), "{text}");
    assert!(text.contains(plan.measurement_hash()), "{text}");
    assert!(
        text.contains("NOT part of the plan hash"),
        "the reader is told which is which: {text}"
    );
}

#[test]
fn explain_marks_a_margin_below_the_noise_floor() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();

    let mut cache = tuned_cache(&resolved);
    for r in resolved.requirements() {
        let mut record = cache.lookup(&r.key).unwrap().clone();
        record.runner_up.as_mut().unwrap().measured_us = 31.4; // +0.6%, under the floor
        cache.insert(r.key.clone(), record);
    }
    let text = explain(&resolved.bind(&cache).unwrap());
    assert!(
        text.contains("margin +0.6% over the runner-up (below the 2.2% noise floor)"),
        "{text}"
    );
}

#[test]
fn explain_says_what_forced_a_bucket_eager() {
    let mut manifest = tiny_manifest();
    manifest.plan.capture = false;
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();

    // A route that may synchronise is legal eager and illegal under capture; `explain` has to say
    // which route it was, not merely that the bucket is eager.
    let unsafe_registry = Registry::new().with(
        Implementation::new("gemm_w4a8_shadow_compare", "mxfp4", "gemm_w4a8")
            .constrain(Constraint::le("m", 4096))
            .capture(true, true),
    );
    let resolved = resolve(&manifest, &device, &model, &unsafe_registry).unwrap();
    let cache = tuned_cache(&resolved);
    let text = explain(&resolved.bind(&cache).unwrap());
    assert!(text.contains("M=4  EAGER"), "{text}");
    assert!(
        text.contains(
            "eager because  these routes cannot be captured: gemm_w4a8_shadow_compare (may_sync)"
        ),
        "{text}"
    );

    // When nothing blocks capture, the manifest is what forced it.
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let cache = tuned_cache(&resolved);
    let text = explain(&resolved.bind(&cache).unwrap());
    assert!(
        text.contains("eager because  the manifest asks for it; every route here is capturable"),
        "{text}"
    );
}

#[test]
fn explain_missing_lists_the_tuning_work() {
    let manifest = tiny_manifest();
    let device = DeviceProfile::gfx1201_reference();
    let model = tiny_model();
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
    let text = explain_missing(&resolved, &AutotuneCache::new());
    assert!(
        text.starts_with("2 untuned route(s) at plan input hash "),
        "{text}"
    );
    assert!(
        text.contains(
            "b2s2c4096  layer.mlp.down_proj  \
             [a_dtype=int8 a_layout=row_major batch=2 k=512 m=4 n=256 spec_width=2 w_dtype=mxfp4]"
        ),
        "{text}"
    );
    assert!(
        text.contains("candidates: gemm_w4a8_mxfp4_decode, gemm_w4a8_mxfp4_wperm"),
        "{text}"
    );

    let cache = tuned_cache(&resolved);
    assert_eq!(explain_missing(&resolved, &cache).lines().count(), 1);
}

#[test]
fn bytes_are_printed_in_binary_units() {
    assert_eq!(human_bytes(0), "0 B");
    assert_eq!(human_bytes(512), "512 B");
    assert_eq!(human_bytes(4096), "4.00 KiB");
    assert_eq!(human_bytes(60 * 1024 * 1024), "60.00 MiB");
    assert_eq!(human_bytes(60 * 1024 * 1024 * 48), "2.81 GiB");
}
