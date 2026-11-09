//! The saber plugin's launch rules, its constraint registry, and their negative controls.
//!
//! Three things are being checked, and the third is the one that needs a known-failing case:
//!
//! 1. `resolve` answers what the reference's own launcher would have launched, on the grid of shapes the
//!    `qwen_prod_nvfp4` unit actually schedules (TP=2, spec width up to 8, 4k and 32k KV).
//! 2. The registry answers `select()`'s question with `select()`'s wording — an unsupported
//!    head_dim is refused with the reference's exact sentence, not a paraphrase of it.
//! 3. A kernarg layout the host and the kernel disagree about is refused (see
//!    `tests/manifest_validate.rs` for the same control on the MXFP4 plugin).

use saber_kernels::abi;
use saber_kernels::manifest::Manifest;
use saber_kernels::rules::{ArgValue, WorkspaceKind};
use saber_kernels::saber::{
    self as saber, gdn_decode_pair_recurrent, resolve, resolve_with, ArDtype, ArNqShape, ArPayload,
    ArShape, AttnShape, GdnPrefillShape, GdnShape, Geometry, KvDtype, NormGate, NormShape, Op,
    Registry, SaberRules, Shape,
};
use saber_kernels::{prebuilt, Error};

fn rules() -> SaberRules {
    SaberRules::prod_saber().expect("prod saber rules parse")
}

/// The model at TP=2: 24 q heads / 4 kv heads shard to 12 / 2, head_dim 256, paged block 16.
fn attn(num_seqs: u32, q_len: u32, max_ctx: u32) -> AttnShape {
    AttnShape {
        num_seqs,
        q_len,
        live_rows: q_len,
        q_heads: 12,
        kv_heads: 2,
        head_dim: 256,
        block_size: 16,
        max_blocks: max_ctx.div_ceil(16),
        max_ctx,
        splits: 0,
        kv_dtype: KvDtype::Fp8E4m3,
        // Per-call, from the cache tensor: kv_heads * block_size * 2 * head_dim elements between
        // blocks, block_size * 2 * head_dim between heads inside one.
        kv_block_stride: 2 * 16 * 512,
        kv_head_stride: 16 * 512,
        scale_bits: 0.0625f32.to_bits(), // head_dim ** -0.5
        has_descales: false,
        ragged_rows: 0,
    }
}

// ---------------------------------------------------------------------------- rules file

#[test]
fn prod_rules_load_and_check() {
    let r = rules();
    assert_eq!(r.name, "saber-gfx1201-rx6");
    assert_eq!(r.knobs.attn_max_decode_rows, 64);
    assert_eq!(r.gdn.fused_max_items, 32);
    assert_eq!(r.ar.max_bytes, 86016 * 1024);
    // Every symbol a rule row can name has to exist in a committed code object.
    let mut named: Vec<&str> = Vec::new();
    named.extend(r.attn_prefill.iter().map(|x| x.symbol.as_str()));
    named.extend(r.attn_decode.iter().map(|x| x.symbol.as_str()));
    named.extend(r.attn_combine.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_fused_update.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_conv_prep.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_conv_update.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_recurrent_update.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_kkt_solve.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_chunk_scan.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_norm.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ar_wht.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ar_exact.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ar_exact_nq.iter().map(|x| x.symbol.as_str()));
    // 54 + C1's f16 stored-state rows (8 recurrent, 2 fused, 1 scan) + C2's ragged attention
    // (8 decode, 2 combine) + 2 uniform16 decode rows.
    assert_eq!(named.len(), 77);
    let dir = prebuilt::dir();
    let mut have: Vec<String> = Vec::new();
    // Every committed saber plugin, taken from the manifest list rather than a second copy of it.
    for (blob, _) in prebuilt::ALL
        .iter()
        .filter(|(b, _)| b.starts_with("saber_"))
    {
        let stem = blob.trim_end_matches(".hsaco");
        let m = Manifest::load(dir.join(format!("{stem}.manifest.toml"))).expect("manifest");
        have.extend(m.kernel.iter().map(|k| k.name.clone()));
    }
    for s in named {
        assert!(have.iter().any(|h| h == s), "no committed kernel `{s}`");
    }
}

/// NEGATIVE-CONTROL-SHAPED: a symbol may exist in SOME committed blob and still be looked up in
/// the wrong one. `saber_ar_oneshot_2rank_exact_bf16` also starts with `saber_ar_oneshot_`, and a
/// prefix table that tested the lossy family first sent it to `saber_ar_wht.hsaco`, where
/// `hipModuleGetFunction` reported "named symbol not found" — at launch, in a two-rank harness,
/// not at build. This checks the MAPPING, not just the existence.
#[test]
fn every_rules_symbol_lives_in_the_plugin_this_maps_it_to() {
    use saber_kernels::saber::plugin_for;
    let r = rules();
    let dir = prebuilt::dir();
    let mut named: Vec<&str> = Vec::new();
    named.extend(r.attn_prefill.iter().map(|x| x.symbol.as_str()));
    named.extend(r.attn_decode.iter().map(|x| x.symbol.as_str()));
    named.extend(r.attn_combine.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_fused_update.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_conv_prep.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_conv_update.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_recurrent_update.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_kkt_solve.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_chunk_scan.iter().map(|x| x.symbol.as_str()));
    named.extend(r.gdn_norm.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ar_wht.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ar_exact.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ar_exact_nq.iter().map(|x| x.symbol.as_str()));
    for sym in named {
        let plugin = plugin_for(sym).unwrap_or_else(|e| panic!("{sym}: {e}"));
        let m = Manifest::load(dir.join(format!("{plugin}.manifest.toml")))
            .unwrap_or_else(|e| panic!("{plugin}: {e}"));
        assert!(
            m.kernel.iter().any(|k| k.name == sym),
            "`{sym}` is mapped to {plugin}.hsaco, which does not contain it"
        );
    }
}

/// NEGATIVE CONTROL: the rules file states the same geometry twice (the decode row count and the
/// warp ladder that has to cover it). A file where they disagree must be refused at load, because
/// the mismatch would only show up as a missing symbol at the first 9-token verify step.
#[test]
fn rules_that_promise_a_query_tile_with_no_instantiation_are_refused() {
    let text = include_str!("../rules/saber_gfx1201.toml");
    let bad = text.replace("decode_max_nw = 4", "decode_max_nw = 3");
    match SaberRules::from_toml(&bad) {
        Err(Error::Rule(m)) => assert!(m.contains("attn_max_decode_rows"), "{m}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    let bad = text.replace("max_bytes = 88080384", "max_bytes = 88080380");
    assert!(SaberRules::from_toml(&bad).is_err(), "16 B alignment");
}

// ---------------------------------------------------------------------------- the registry

#[test]
fn the_registry_selects_what_prod_binds() {
    let r = rules();
    let reg = Registry::new(&r);
    let g = Geometry::new()
        .num("head_dim", 256)
        .num("gqa", 6)
        .num("block_size", 16)
        .num("causal", 1)
        .dtype("q_dtype", "torch.bfloat16")
        .dtype("kv_dtype", "fp8"); // the reference server's spelling; norm_dtype folds it to fp8_e4m3
    assert_eq!(
        reg.select("attn_prefill_paged", &g).unwrap().entry,
        "attn_prefill_h256_gqa6_fp8kv"
    );
    // The decode row adds q_len <= 10; without q_len at all it is "not given", as in the reference.
    assert_eq!(
        reg.select("attn_decode_paged", &g).unwrap_err(),
        "q_len not given"
    );
    let g8 = g.clone().num("q_len", 8);
    assert_eq!(
        reg.select("attn_decode_paged", &g8).unwrap().entry,
        "attn_decode_h256_gqa6_fp8kv"
    );
    let g11 = g.num("q_len", 11);
    assert_eq!(
        reg.select("attn_decode_paged", &g11).unwrap_err(),
        "q_len=11, needs <= 10; q_len=11, needs == 16"
    );
}

/// NEGATIVE CONTROL, and the wording is the assertion: the reference's `matches()` names the constraint
/// that failed and what it wanted, because "no kernel" on its own is not something a reader can
/// act on. A paraphrase here would be a second copy of the constraints, which is the drift the
/// registry exists to prevent.
#[test]
fn an_unsupported_head_dim_is_refused_in_sabers_own_words() {
    let r = rules();
    let reg = Registry::new(&r);
    let g = Geometry::new()
        .num("head_dim", 128)
        .num("gqa", 6)
        .num("block_size", 16)
        .num("causal", 1)
        .dtype("q_dtype", "bf16")
        .dtype("kv_dtype", "fp8_e4m3");
    assert_eq!(
        reg.select("attn_prefill_paged", &g).unwrap_err(),
        "head_dim=128, needs == 256"
    );
    // explain() reports every candidate, matched or not.
    let rows = reg.explain("attn_prefill_paged", &g);
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|v| !v.ok));
    assert!(rows
        .iter()
        .all(|v| v.reason == "head_dim=128, needs == 256"));

    // A dtype the build has no kernel for, and an op it does not implement at all.
    let g = Geometry::new()
        .num("head_dim", 256)
        .num("gqa", 6)
        .num("block_size", 16)
        .num("causal", 1)
        .dtype("q_dtype", "bf16")
        .dtype("kv_dtype", "fp32");
    assert_eq!(
        reg.select("attn_prefill_paged", &g).unwrap_err(),
        "kv_dtype=fp32, needs one of: fp8_e4m3; kv_dtype=fp32, needs one of: bf16"
    );
    // An op this build has no rows for at all -- the reference's registry carries the vision-tower
    // attention, and Saber's does not because the plugin was not ported.
    assert_eq!(
        reg.select("attn_vit", &Geometry::new()).unwrap_err(),
        "no kernel in this build has op 'attn_vit'"
    );
    // The lossy all-reduce refuses a message that does not tile a rotation group.
    let g = Geometry::new()
        .num("world_size", 2)
        .num("exact", 0)
        .num("bits", 6)
        .dtype("dtype", "bf16")
        .num("numel", 100);
    // Every distinct miss across the rows of that op, deduped and joined, exactly as
    // select_kernel() builds its reason: the exact row fails on `exact`, the wht6 row on numel,
    // the wht4 row on bits.
    assert_eq!(
        reg.select("allreduce", &g).unwrap_err(),
        "exact=0, needs == 1; numel=100, needs divisible by 64; bits=6, needs == 4"
    );
}

// ---------------------------------------------------------------------------- attention

#[test]
fn prefill_tiles_the_query_at_sixty_four_rows_a_block() {
    let r = rules();
    for (q_len, blocks) in [(128u32, 2u32), (2048, 32), (8192, 128)] {
        let d = resolve(
            Op::AttnPrefill { fp8_mode: 3 },
            Shape::Attn(attn(1, q_len, 262144)),
            &r,
        )
        .expect("resolve");
        assert_eq!(d.kernel, "saber_attn_prefill_h256_gqa6_fp8kv_qk8pv8");
        assert_eq!(d.grid, [blocks, 2, 1]);
        assert_eq!(d.block, [768, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert!(d.workspaces.is_empty());
        assert_eq!(d.kernarg_bytes, 136);
    }
    // SABER_ATTN_FP8 is a row, not a getenv: each value names a different symbol.
    for (mode, sym) in [
        (0u32, "saber_attn_prefill_h256_gqa6_fp8kv"),
        (1, "saber_attn_prefill_h256_gqa6_fp8kv_qk8"),
        (2, "saber_attn_prefill_h256_gqa6_fp8kv_pv8"),
    ] {
        let d = resolve(
            Op::AttnPrefill { fp8_mode: mode },
            Shape::Attn(attn(1, 2048, 262144)),
            &r,
        )
        .unwrap();
        assert_eq!(d.kernel, sym);
    }
    // ... and the bf16-KV kernel has no 8-bit leg to name.
    let mut s = attn(1, 2048, 262144);
    s.kv_dtype = KvDtype::Bf16;
    assert_eq!(
        resolve(Op::AttnPrefill { fp8_mode: 0 }, Shape::Attn(s), &r)
            .unwrap()
            .kernel,
        "saber_attn_prefill_h256_gqa6_bf16kv"
    );
    assert!(resolve(Op::AttnPrefill { fp8_mode: 3 }, Shape::Attn(s), &r).is_err());
}

/// ws-h27: the `q_len` kernarg is the CHUNK's live rows, the grid covers only those rows, and a
/// retarget is a re-resolve.
///
/// The third claim is what makes `retarget_attn_prefill` an optimisation of `resolve` rather than
/// a second law: the prefill leg's registry geometry does not carry `q_len`, so the kernarg and
/// the grid are the whole difference between two resolves.
#[test]
fn a_padded_chunk_resolves_at_its_live_rows_and_a_retarget_is_a_re_resolve() {
    let r = rules();
    let bucket = 2048u32;
    for live in [1u32, 63, 64, 65, 134, 234, 1024, 2048] {
        let mut s = attn(1, bucket, 262144);
        s.live_rows = live;
        let at_live =
            resolve(Op::AttnPrefill { fp8_mode: 3 }, Shape::Attn(s), &r).expect("resolve");
        // The grid covers the LIVE rows: a 134-row chunk in a 2048-row bucket is 3 CTAs, not 32.
        assert_eq!(at_live.grid, [live.div_ceil(64), 2, 1], "live={live}");
        assert_eq!(
            at_live.arg("q_len").map(|a| a.value.clone()),
            Some(ArgValue::I32 {
                ival: i64::from(live)
            }),
            "live={live}"
        );
        // ...and a descriptor resolved at the bucket and then retargeted is byte-for-byte that one.
        let mut retargeted = resolve(
            Op::AttnPrefill { fp8_mode: 3 },
            Shape::Attn(attn(1, bucket, 262144)),
            &r,
        )
        .expect("resolve at the bucket");
        saber_kernels::saber::retarget_attn_prefill(&mut retargeted, live, bucket, &r)
            .expect("retarget");
        assert_eq!(retargeted, at_live, "live={live}");
        // Idempotent and re-entrant: the BUCKET is the authority, never the last chunk.
        saber_kernels::saber::retarget_attn_prefill(&mut retargeted, bucket, bucket, &r)
            .expect("retarget back");
        saber_kernels::saber::retarget_attn_prefill(&mut retargeted, live, bucket, &r)
            .expect("retarget again");
        assert_eq!(retargeted, at_live, "live={live}");
    }
    // NEGATIVE CONTROLS. A chunk is padded UP into its bucket, never truncated; a decode step's
    // `q_len` is also its `sidx` block width and its row stride, so it has no live/bucket split;
    // and a retarget of something that is not a prefill attention descriptor is a refusal rather
    // than a scalar written into whichever kernarg happens to be called `q_len`.
    let mut over = attn(1, 256, 262144);
    over.live_rows = 257;
    assert!(resolve(Op::AttnPrefill { fp8_mode: 3 }, Shape::Attn(over), &r).is_err());
    over.live_rows = 0;
    assert!(resolve(Op::AttnPrefill { fp8_mode: 3 }, Shape::Attn(over), &r).is_err());
    let mut short_decode = attn(1, 5, 4096);
    short_decode.live_rows = 3;
    let e = resolve(Op::AttnDecode, Shape::Attn(short_decode), &r).unwrap_err();
    assert!(format!("{e}").contains("sidx"), "{e}");
    let mut dec = resolve(Op::AttnDecode, Shape::Attn(attn(1, 5, 4096)), &r).unwrap();
    assert!(saber_kernels::saber::retarget_attn_prefill(&mut dec, 3, 5, &r).is_err());
    let mut pre = resolve(
        Op::AttnPrefill { fp8_mode: 3 },
        Shape::Attn(attn(1, 256, 262144)),
        &r,
    )
    .unwrap();
    assert!(saber_kernels::saber::retarget_attn_prefill(&mut pre, 0, 256, &r).is_err());
    assert!(saber_kernels::saber::retarget_attn_prefill(&mut pre, 257, 256, &r).is_err());
}

#[test]
fn decode_picks_the_warp_count_from_the_spec_width_and_the_splits_from_the_law() {
    let r = rules();
    // (spec width, batch, KV) -> (nw, splits). gqa 6: 1-2 tokens -> 1 warp, 3-5 -> 2, 6-8 -> 3.
    let cases = [
        (1u32, 1u32, 4096u32, 1u32, 64u32),
        (5, 1, 4096, 2, 64),
        (8, 1, 4096, 3, 64),
        (1, 8, 4096, 1, 16),
        (5, 8, 4096, 2, 16),
        (8, 8, 4096, 3, 16),
        (8, 1, 32768, 3, 64),
        (8, 8, 32768, 3, 16),
    ];
    for (q_len, batch, ctx, nw, splits) in cases {
        let s = attn(batch, q_len, ctx);
        let d = resolve(Op::AttnDecode, Shape::Attn(s), &r).expect("resolve");
        assert_eq!(
            d.kernel,
            format!("saber_attn_decode_h256_gqa6_fp8kv_nw{nw}"),
            "q_len={q_len} batch={batch}"
        );
        assert_eq!(d.grid, [splits, 2, batch], "q_len={q_len} batch={batch}");
        assert_eq!(d.block, [nw * 32, 1, 1]);
        // The split-KV combine is a second dispatch, declared statically (invariant 7).
        assert_eq!(d.n_launches, 2);
        assert_eq!(d.kernarg_bytes, 140);
        let ws = &d.workspaces[0];
        assert_eq!(ws.name, "attn_decode_scratch");
        assert_eq!(ws.kind, WorkspaceKind::Scratch);
        assert_eq!(
            ws.bytes,
            batch as u64 * q_len as u64 * 12 * splits as u64 * 520
        );
        let c = saber_kernels::saber::attn_decode_combine(s, &r).expect("combine");
        assert_eq!(c.kernel, "saber_attn_splitkv_combine_h256_pf1");
        assert_eq!(c.grid, [batch * q_len * 12, 1, 1]);
        assert_eq!(c.block, [256, 1, 1]);
        assert_eq!(c.shared_bytes, splits * 4);
        assert_eq!(c.kernarg_bytes, 144);
    }
}

#[test]
fn uniform16_uses_six_waves_and_the_real_partial_layout() {
    let r = rules();
    let manifest = Manifest::load(prebuilt::dir().join("saber_attn16.manifest.toml")).unwrap();
    for (dtype, suffix) in [(KvDtype::Fp8E4m3, "fp8"), (KvDtype::Bf16, "bf16")] {
        let mut s = attn(2, 16, 8192);
        s.kv_dtype = dtype;
        let head = resolve(Op::AttnDecode, Shape::Attn(s), &r).unwrap();
        assert_eq!(
            head.kernel,
            format!("saber_attn_decode_uniform16_h256_gqa6_{suffix}kv_nw6")
        );
        assert_eq!(head.plugin, "saber_attn16");
        assert_eq!(head.block, [192, 1, 1]);
        let splits = r.decode_splits(s.max_ctx, s.kv_heads, s.num_seqs, s.splits);
        let bytes = u64::from(2 * 16 * 12 * splits) * 520;
        assert_eq!(head.grid, [splits, 2, 2]);
        assert_eq!(head.workspaces[0].bytes, bytes);
        assert_eq!(head.kernarg_bytes, 140);
        manifest
            .check_signature(&head.kernel, abi::SIG_SABER_ATTN_DECODE)
            .unwrap();

        let combine = saber::attn_decode_combine(s, &r).unwrap();
        assert_eq!(combine.kernel, "saber_attn_splitkv_combine_h256_pf1");
        assert_eq!(combine.plugin, "saber_attn");
        assert_eq!(combine.grid, [2 * 16 * 12, 1, 1]);
        assert_eq!(combine.workspaces[0].bytes, bytes);
        assert_eq!(combine.kernarg_bytes, 144);
        assert!(saber::attn_decode_declc(s, &r).is_err());

        // Every refusal below would otherwise give a plausible but incorrect descriptor.
        for bad in [
            AttnShape { live_rows: 15, ..s },
            AttnShape {
                ragged_rows: 30,
                ..s
            },
            AttnShape { q_heads: 10, ..s },
            AttnShape { head_dim: 128, ..s },
            AttnShape {
                block_size: 32,
                ..s
            },
            attn(2, 15, 8192),
            attn(2, 17, 8192),
        ] {
            assert!(resolve(Op::AttnDecode, Shape::Attn(bad), &r).is_err());
        }
        let mut small = r.clone();
        small
            .workspace
            .get_mut("attn_decode_scratch")
            .unwrap()
            .capacity_bytes = bytes - 1;
        assert!(resolve(Op::AttnDecode, Shape::Attn(s), &small).is_err());
        assert!(saber::attn_decode_combine(s, &small).is_err());
        let mut pf32 = r.clone();
        pf32.knobs.combine_partial_f16 = false;
        assert!(resolve(Op::AttnDecode, Shape::Attn(s), &pf32).is_err());
        assert!(saber::attn_decode_combine(s, &pf32).is_err());

        // An incorrect table pointing W16 back to a narrow symbol still has its 64-row guard.
        let mut narrow = r.clone();
        for row in &mut narrow.attn_decode {
            row.symbol = row.symbol.replace("_uniform16", "").replace("_nw6", "_nw4");
        }
        assert!(resolve(Op::AttnDecode, Shape::Attn(s), &narrow)
            .unwrap_err()
            .to_string()
            .contains("exceeds the 64"));
    }
}

/// **The RAGGED decode pair (C2).** A row band's shape resolves to the `saber_attn_ragged` plugin:
/// NW by the uniform law at the plan's MAX width, the same split law and scratch, `cu_q` as the
/// last kernarg (8-aligned at 144), and a combine over the BAND's rows — never `num_seqs * q_len`,
/// which would write past a band's arena.
#[test]
fn a_row_band_resolves_the_ragged_pair_and_the_combine_covers_the_band_rows() {
    let r = rules();
    for (batch, max_w, rows, nw) in [(8u32, 8u32, 40u32, 3u32), (8, 5, 24, 2), (4, 8, 16, 3)] {
        let mut s = attn(batch, max_w, 4096);
        s.ragged_rows = rows;
        let uniform = resolve(Op::AttnDecode, Shape::Attn(attn(batch, max_w, 4096)), &r).unwrap();
        let d = resolve(Op::AttnDecode, Shape::Attn(s), &r).expect("ragged resolves");
        assert_eq!(
            d.kernel,
            format!("saber_attn_decode_ragged_h256_gqa6_fp8kv_nw{nw}")
        );
        assert_eq!(d.plugin, "saber_attn_ragged");
        // Everything but the symbol and the extra kernarg is the uniform descriptor's.
        assert_eq!(d.grid, uniform.grid);
        assert_eq!(d.block, uniform.block);
        assert_eq!(d.workspaces, uniform.workspaces);
        assert_eq!(d.n_launches, 2);
        assert_eq!(d.kernarg_bytes, 152);
        let cu = d.arg("cu_q").expect("cu_q");
        assert_eq!((cu.offset, cu.size), (144, 8));
        assert_eq!(
            cu.value,
            ArgValue::Buffer {
                buffer: "cu_q".into()
            }
        );
        // q_len stays the plan's MAX width: it is the partials' row stride.
        assert_eq!(
            d.arg("q_len").map(|a| a.value.clone()),
            Some(ArgValue::I32 {
                ival: i64::from(max_w)
            })
        );
        let c = saber_kernels::saber::attn_decode_combine(s, &r).expect("combine");
        assert_eq!(c.kernel, "saber_attn_splitkv_combine_ragged_h256_pf1");
        assert_eq!(c.plugin, "saber_attn_ragged");
        assert_eq!(c.grid, [rows * 12, 1, 1], "one job per band row per q head");
        assert_eq!(c.kernarg_bytes, 152);
        assert_eq!(c.arg("cu_q").map(|a| a.offset), Some(144));
        assert_eq!(c.arg("tile").map(|a| a.offset), Some(140));
    }
    // And the uniform shape is untouched by the new rows: same symbol family, no `cu_q`.
    let u = resolve(Op::AttnDecode, Shape::Attn(attn(8, 5, 4096)), &r).unwrap();
    assert_eq!(u.kernel, "saber_attn_decode_h256_gqa6_fp8kv_nw2");
    assert!(u.arg("cu_q").is_none());
    assert_eq!(u.kernarg_bytes, 140);
}

/// NEGATIVE CONTROLS for the ragged route. (1) The rx6 table ALONE — no ragged table merged —
/// has no `attn_decode_paged_ragged` op, so a band is refused by name and never falls through to
/// the uniform kernel (which would right-align every sequence on the max width). (2) A ragged
/// table whose entry is NAMED like a uniform one resolves under the ragged op to a uniform
/// entry, and `resolve` refuses the mix by name. (3) A band wider than `num_seqs x q_len` is
/// refused: the partials would not hold it. (4) A ragged table that reaches another op is refused
/// at merge.
#[test]
fn a_ragged_shape_is_never_resolved_to_the_uniform_kernel() {
    let mut s = attn(8, 8, 4096);
    s.ragged_rows = 40;
    let rx6 = SaberRules::from_toml(saber::PROD_RULES_TEXT).expect("parses");
    let e = resolve(Op::AttnDecode, Shape::Attn(s), &rx6).unwrap_err();
    assert!(format!("{e}").contains("attn_decode_paged_ragged"), "{e}");
    // The uniform shape is served by the rx6 table alone, as before C2.
    assert!(resolve(Op::AttnDecode, Shape::Attn(attn(8, 8, 4096)), &rx6).is_ok());

    let bad = saber::RAGGED_RULES_TEXT.replace(
        "attn_decode_h256_gqa6_fp8kv_ragged",
        "attn_decode_h256_gqa6_fp8kv",
    );
    assert_ne!(
        bad,
        saber::RAGGED_RULES_TEXT,
        "the control must edit the table"
    );
    let rb = SaberRules::from_toml(saber::PROD_RULES_TEXT)
        .and_then(|r| r.with_ragged_toml(&bad))
        .expect("parses");
    let e = resolve(Op::AttnDecode, Shape::Attn(s), &rb).unwrap_err();
    assert!(format!("{e}").contains("RAGGED (40-row band)"), "{e}");
    let e = saber_kernels::saber::attn_decode_combine(s, &rb).unwrap_err();
    assert!(format!("{e}").contains("never substitutes"), "{e}");

    // The real rules resolve the same shape.
    let r = rules();
    assert!(resolve(Op::AttnDecode, Shape::Attn(s), &r).is_ok());
    // A band past batch x max width.
    s.ragged_rows = 65;
    let e = resolve(Op::AttnDecode, Shape::Attn(s), &r).unwrap_err();
    assert!(
        format!("{e}").contains("clipped to batch x max width"),
        "{e}"
    );
    // A ragged table naming another op cannot be merged.
    let other = saber::RAGGED_RULES_TEXT.replace(
        "op = \"attn_decode_paged_ragged\"",
        "op = \"attn_decode_paged\"",
    );
    assert!(SaberRules::from_toml(saber::PROD_RULES_TEXT)
        .and_then(|r| r.with_ragged_toml(&other))
        .is_err());
}

#[test]
fn decode_refuses_a_verify_width_past_the_query_tile() {
    let r = rules();
    // 11 tokens x gqa 6 = 66 rows; the workgroup holds 64. The reference returns -4.
    let e = resolve(Op::AttnDecode, Shape::Attn(attn(1, 11, 4096)), &r).unwrap_err();
    assert!(format!("{e}").contains("q_len=11, needs <= 10"), "{e}");
    // A batch whose q_heads are not a whole multiple of kv_heads is a per-call refusal.
    let mut s = attn(1, 1, 4096);
    s.kv_heads = 5;
    assert!(resolve(Op::AttnDecode, Shape::Attn(s), &r).is_err());
}

#[test]
fn every_attention_argument_lands_where_the_by_value_struct_puts_it() {
    let r = rules();
    let d = resolve(Op::AttnDecode, Shape::Attn(attn(8, 8, 32768)), &r).unwrap();
    assert_eq!(d.arg("q").map(|s| s.offset), Some(0));
    // A slot is named for the STRUCT FIELD; the buffer it binds is named for the workspace, so
    // the descriptor says both which pointer this is and where its memory comes from.
    assert_eq!(
        d.arg("scratch").map(|s| s.offset),
        Some(64),
        "the ninth pointer of SABERArgs"
    );
    assert_eq!(
        d.arg("scratch").map(|s| s.value.clone()),
        Some(ArgValue::Buffer {
            buffer: "attn_decode_scratch".into()
        })
    );
    assert_eq!(d.arg("num_seqs").map(|s| s.offset), Some(72));
    assert_eq!(d.arg("kv_block_stride").map(|s| s.offset), Some(104));
    assert_eq!(d.arg("max_ctx").map(|s| s.offset), Some(128));
    assert_eq!(d.arg("splits").map(|s| s.offset), Some(136));
    // The query descale is documented unused: bound null, and named so the descriptor says which
    // pointer is absent rather than carrying an unexplained zero.
    assert!(matches!(
        d.arg("q_descale").map(|s| s.value.clone()),
        Some(ArgValue::NullBuffer { .. })
    ));
    // Prefill dereferences no scratch at all.
    let p = resolve(
        Op::AttnPrefill { fp8_mode: 3 },
        Shape::Attn(attn(1, 2048, 262144)),
        &r,
    )
    .unwrap();
    assert!(matches!(
        p.arg("scratch").map(|s| s.value.clone()),
        Some(ArgValue::NullBuffer { .. })
    ));
}

// ---------------------------------------------------------------------------- GDN

fn gdn(n_seqs: u32, maxq: u32) -> GdnShape {
    GdnShape {
        n_seqs,
        h: 24, // 48 v heads at TP=2
        hg: 8, // 16 k heads at TP=2
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        maxq,
        o_rows: n_seqs * maxq,
        xpitch: 5120, // the 40-unit qkvz split
        cs_seq: 5120 * (3 + maxq as i64 - 1),
        cs_dim: 3 + maxq as i64 - 1,
        cs_tok: 1,
        ci_stride: maxq as i64,
        ab_stride: 24,
        st_slot: 24 * 128 * 128,
        st_head: 128 * 128,
        sidx_stride: maxq as i64,
        scale_bits: 0.088_388_35f32.to_bits(), // head_k ** -0.5
        softplus_thr_bits: 20.0f32.to_bits(),
        has_naccept: true,
        has_bias: true,
        state_f16: false,
    }
}

#[test]
fn the_fused_gdn_update_is_one_launch_and_owns_a_persistent_barrier() {
    let r = rules();
    for maxq in [1u32, 5, 8] {
        let d = resolve(
            Op::GdnFusedUpdate { ab_bf16: true },
            Shape::Gdn(gdn(1, maxq)),
            &r,
        )
        .expect("resolve");
        assert_eq!(d.kernel, "saber_gdn_fused_update_w4k128v128_ab1");
        // numWG = min(N*H, FU_MAXWG) = min(24, 32).
        assert_eq!(d.grid, [24, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        // conv -> grid barrier -> recurrent, ONE dispatch. This is the whole point of the kernel.
        assert_eq!(d.n_launches, 1);
        assert_eq!(d.kernarg_bytes, 256);
        assert_eq!(
            d.arg("num_wg").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 24 })
        );
        assert_eq!(
            d.arg("state_len_max").map(|s| s.value.clone()),
            Some(ArgValue::I32 {
                ival: (3 + maxq - 1) as i64
            })
        );
        let b = &d.workspaces[0];
        assert_eq!(b.name, "gdn_fused_barrier");
        assert_eq!(b.kind, WorkspaceKind::Persistent);
        assert!(b.zero_once);
        assert_eq!(b.bytes, 4);
        // The GDN state is CALLER-OWNED: it is the sequence's memory, and it must never appear
        // as a workspace (docs/ARCHITECTURE.md, "What exploration corrected" item 3).
        assert!(d.workspaces.iter().all(|w| w.name != "state"));
        assert!(matches!(
            d.arg("state").map(|s| s.value.clone()),
            Some(ArgValue::Buffer { .. })
        ));
    }
    assert_eq!(
        resolve(
            Op::GdnFusedUpdate { ab_bf16: false },
            Shape::Gdn(gdn(1, 8)),
            &r
        )
        .unwrap()
        .kernel,
        "saber_gdn_fused_update_w4k128v128_ab0"
    );
}

/// NEGATIVE CONTROL: above `fused_max_items` prod runs the conv+recurrent PAIR, which this plugin
/// does not carry. Silently launching the fused kernel anyway would be correct-but-slow at best
/// (its work loop would serialise the overflow onto 32 workgroups) — so it refuses and names the
/// route. At TP=2's H=24 that means batch 1 and nothing else takes the fused path, which is
/// exactly what the 2026-09-02 serving A/B measured.
#[test]
fn the_fused_gdn_update_refuses_the_batch_the_pair_owns() {
    let r = rules();
    let e = resolve(
        Op::GdnFusedUpdate { ab_bf16: true },
        Shape::Gdn(gdn(2, 8)),
        &r,
    )
    .unwrap_err();
    let m = format!("{e}");
    assert!(m.contains("48 items exceeds fused_max_items 32"), "{m}");
    assert!(m.contains("recurrent_update pair"), "{m}");
    // Geometry the library has no kernel for is the registry's refusal, in its own words.
    let mut s = gdn(1, 8);
    s.head_v = 64;
    let e = resolve(Op::GdnFusedUpdate { ab_bf16: true }, Shape::Gdn(s), &r).unwrap_err();
    assert!(format!("{e}").contains("head_v=64, needs == 128"), "{e}");
}

#[test]
fn the_gated_rmsnorm_is_one_wave_a_row() {
    let r = rules();
    let s = NormShape {
        rows: 8 * 24,
        channels: 128,
        xrow: 128,
        zrow: 128,
        orow: 128,
        eps_bits: 1e-6f32.to_bits(),
    };
    let d = resolve(Op::GdnGatedRmsNorm { act: 0 }, Shape::Norm(s), &r).expect("resolve");
    assert_eq!(d.kernel, "saber_gdn_gated_rmsnorm_h128_silu");
    assert_eq!(d.grid, [24, 1, 1]); // 192 rows / 8 rows per block
    assert_eq!(d.block, [256, 1, 1]);
    assert_eq!(d.n_launches, 1);
    assert_eq!(d.kernarg_bytes, 68);
    assert_eq!(
        resolve(Op::GdnGatedRmsNorm { act: 1 }, Shape::Norm(s), &r)
            .unwrap()
            .kernel,
        "saber_gdn_gated_rmsnorm_h128_sigmoid"
    );
    // A row stride that breaks the 8-byte-per-lane access is a per-call refusal.
    let mut bad = s;
    bad.zrow = 130;
    assert!(resolve(Op::GdnGatedRmsNorm { act: 0 }, Shape::Norm(bad), &r).is_err());
    // A channel count the library has no kernel for is the registry's.
    let mut bad = s;
    bad.channels = 64;
    let e = resolve(Op::GdnGatedRmsNorm { act: 0 }, Shape::Norm(bad), &r).unwrap_err();
    assert!(format!("{e}").contains("channels=64, needs == 128"), "{e}");
}

fn gdn_prefill(t: u32) -> GdnPrefillShape {
    GdnPrefillShape {
        n_seqs: 1,
        t,
        h: 24,
        hg: 8,
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        chunk: 64,
        xpitch: 5120,
        cs_seq: 5120 * 3,
        cs_dim: 3,
        cs_tok: 1,
        ci_stride: 1,
        ab_stride: 24,
        scale_bits: 0.088_388_35f32.to_bits(),
        softplus_thr_bits: 20.0f32.to_bits(),
        has_bias: true,
        has_init: true,
        state_f16: false,
    }
}

/// The batch >= 2 decode path, which at TP=2 is every batch but one: TWO launches, and the second
/// one's v-row split falls out of how full the (N, H) grid already is.
#[test]
fn the_decode_pair_is_two_launches_and_splits_v_rows_to_fill_the_grid() {
    let r = rules();
    // (batch, expected vs, expected recurrent block). H = 24, so N*H is 48 at batch 2 and 192 at
    // batch 8; the fill target is 64 workgroups.
    for (batch, vs, threads) in [(2u32, 2u32, 128u32), (8, 1, 256)] {
        for maxq in [1u32, 5, 8] {
            let s = GdnShape {
                n_seqs: batch,
                ..gdn(batch, maxq)
            };
            let conv = resolve(Op::GdnDecodePair { ab_bf16: true }, Shape::Gdn(s), &r)
                .expect("conv_update");
            // hpb halves until N * (units / hpb) >= 256; at TP=2 that bottoms out at 1.
            assert_eq!(conv.kernel, "saber_gdn_conv_update_w4_h128_hpb1");
            assert_eq!(conv.grid, [batch, 40, 1], "units = 2*Hg + H = 40");
            assert_eq!(conv.block, [32, 1, 1]);
            assert_eq!(conv.kernarg_bytes, 140);
            // The pair is TWO launches, declared on the op that names it (invariant 7).
            assert_eq!(conv.n_launches, 2);
            assert!(conv.workspaces.is_empty());
            assert_eq!(
                conv.arg("state_len_max").map(|x| x.value.clone()),
                Some(ArgValue::I32 {
                    ival: (3 + maxq - 1) as i64
                })
            );

            let rec = gdn_decode_pair_recurrent(s, true, None, &r).expect("recurrent_update");
            assert_eq!(
                rec.kernel,
                format!("saber_gdn_recurrent_update_k128_v128_ab1_norm0_vs{vs}")
            );
            assert_eq!(rec.grid, [batch, 24, vs]);
            assert_eq!(rec.block, [threads, 1, 1]);
            assert_eq!(rec.kernarg_bytes, 168);
            assert_eq!(rec.n_launches, 1);
            // The paged fp32 state is CALLER-OWNED, exactly as in the fused kernel.
            assert!(rec.workspaces.is_empty());
            assert!(matches!(
                rec.arg("state").map(|x| x.value.clone()),
                Some(ArgValue::Buffer { .. })
            ));
            // Prod passes no z gate and no norm weight: both pointers are named and null.
            assert!(matches!(
                rec.arg("zg").map(|x| x.value.clone()),
                Some(ArgValue::NullBuffer { .. })
            ));
        }
    }
}

/// The fused row norm needs the whole 128-wide row, so asking for it forces the unsplit kernel —
/// which is a `static_assert` in the reference and a missing row here.
#[test]
fn the_fused_row_norm_forces_the_unsplit_recurrent_kernel() {
    let r = rules();
    let norm = Some(NormGate {
        eps_bits: 1e-6f32.to_bits(),
        act: 0,
    });
    let d = gdn_decode_pair_recurrent(gdn(2, 8), true, norm, &r).expect("norm at vs=1");
    assert_eq!(
        d.kernel,
        "saber_gdn_recurrent_update_k128_v128_ab1_norm1_vs1"
    );
    assert_eq!(d.grid, [2, 24, 1]);
    assert_eq!(d.block, [256, 1, 1]);
    assert!(matches!(
        d.arg("nw").map(|x| x.value.clone()),
        Some(ArgValue::Buffer { .. })
    ));
    assert_eq!(
        d.arg("nact").map(|x| x.value.clone()),
        Some(ArgValue::I32 { ival: 0 })
    );
}

#[test]
fn the_prefill_trio_walks_an_upper_bound_on_the_chunk_count() {
    let r = rules();
    for (t, chunks) in [(128u32, 3u32), (2048, 33), (8192, 129)] {
        let s = gdn_prefill(t);
        let prep = resolve(Op::GdnConvPrep { ab_bf16: true }, Shape::GdnPrefill(s), &r)
            .expect("conv_prep");
        // hpb 8 divides both Hg=8 and H=24; the prefill entry does NOT narrow further.
        assert_eq!(prep.kernel, "saber_gdn_conv_prep_w4_h128_hpb8_ab1");
        assert_eq!(prep.grid, [chunks, 5, 1], "units 40 / hpb 8");
        assert_eq!(prep.block, [256, 1, 1]);
        assert_eq!(prep.kernarg_bytes, 192);
        assert_eq!(prep.n_launches, 1);
        assert!(prep.workspaces.is_empty());

        let kkt = resolve(Op::GdnKktSolve, Shape::GdnPrefill(s), &r).expect("kkt_solve");
        assert_eq!(kkt.kernel, "saber_gdn_kkt_solve_k128_c64_r3", "H/Hg = 24/8");
        assert_eq!(kkt.grid, [chunks, 8, 1], "one gram per k head");
        assert_eq!(kkt.block, [128, 1, 1]);
        assert_eq!(kkt.kernarg_bytes, 52);

        let scan = resolve(Op::GdnChunkScan, Shape::GdnPrefill(s), &r).expect("chunk_scan");
        assert_eq!(scan.kernel, "saber_gdn_chunk_scan_k128_v128_c64");
        assert_eq!(
            scan.grid,
            [2, 24, 1],
            "head_v 128 / BV 64, per v head, per sequence"
        );
        assert_eq!(scan.block, [256, 1, 1]);
        assert_eq!(scan.kernarg_bytes, 92);
        assert_eq!(scan.n_launches, 1);
    }
    // ab_bf16 is a row on the preamble (it reads the gates) and nowhere else.
    assert_eq!(
        resolve(
            Op::GdnConvPrep { ab_bf16: false },
            Shape::GdnPrefill(gdn_prefill(2048)),
            &r
        )
        .unwrap()
        .kernel,
        "saber_gdn_conv_prep_w4_h128_hpb8_ab0"
    );
}

/// NEGATIVE CONTROLS for the prefill trio. `H % Hg` relates two parameters, so it is a per-call
/// refusal and not a registry predicate; `H / Hg` picks a kkt instantiation and the reference
/// returns -3 for a ratio it was not compiled for.
#[test]
fn the_prefill_trio_refuses_head_counts_it_has_no_instantiation_for() {
    let r = rules();
    let mut s = gdn_prefill(2048);
    s.hg = 7;
    let e = resolve(Op::GdnKktSolve, Shape::GdnPrefill(s), &r).unwrap_err();
    assert!(format!("{e}").contains("not a whole multiple"), "{e}");
    // 24 / 5 is not whole either; make one that IS whole but has no arm: H/Hg = 5.
    let mut s = gdn_prefill(2048);
    s.h = 40;
    s.hg = 8;
    let e = resolve(Op::GdnKktSolve, Shape::GdnPrefill(s), &r).unwrap_err();
    assert!(
        format!("{e}").contains("H/Hg = 5 has no instantiation"),
        "{e}"
    );
    // A chunk size the library has no kernel for is the registry's refusal, in its own words.
    let mut s = gdn_prefill(2048);
    s.chunk = 32;
    let e = resolve(Op::GdnChunkScan, Shape::GdnPrefill(s), &r).unwrap_err();
    assert!(format!("{e}").contains("chunk=32, needs == 64"), "{e}");
    // ... and a v head the 64-wide tile does not divide is a per-call one.
    let mut s = gdn_prefill(2048);
    s.head_v = 96;
    let e = resolve(Op::GdnChunkScan, Shape::GdnPrefill(s), &r).unwrap_err();
    assert!(format!("{e}").contains("head_v=96, needs == 128"), "{e}");
    // An op handed the wrong KIND of shape is refused rather than silently reinterpreted.
    assert!(resolve(Op::GdnKktSolve, Shape::Gdn(gdn(1, 8)), &r).is_err());
}

/// The selection log: every distinct question this registry was asked, repeats folded, refusals
/// included. `explain` asks nothing, so it records nothing.
#[test]
fn the_registry_logs_every_question_and_folds_the_repeats() {
    let r = rules();
    let reg = Registry::new(&r);
    // A 64-layer model asks the same question once per instantiation; "asked N times" is how much
    // of the model resolved to that kernel.
    for _ in 0..16 {
        resolve_with(
            Op::AttnPrefill { fp8_mode: 3 },
            Shape::Attn(attn(1, 2048, 262144)),
            &r,
            &reg,
        )
        .unwrap();
    }
    for _ in 0..48 {
        resolve_with(
            Op::GdnDecodePair { ab_bf16: true },
            Shape::Gdn(gdn(8, 8)),
            &r,
            &reg,
        )
        .unwrap();
    }
    // A refusal is recorded too: it is the reason a fallback is running.
    let mut bad = gdn_prefill(2048);
    bad.chunk = 32;
    assert!(resolve_with(Op::GdnChunkScan, Shape::GdnPrefill(bad), &r, &reg).is_err());

    let log = reg.selections();
    assert_eq!(log.len(), 3, "three distinct questions: {log:#?}");
    assert_eq!(log[0].op, "attn_prefill_paged");
    assert_eq!(log[0].kernel, "attn_prefill_h256_gqa6_fp8kv");
    assert_eq!(log[0].count, 16);
    assert!(log[0].reason.is_empty());
    // The query is recorded normalised, the way the matcher compared it.
    assert!(log[0]
        .query
        .contains(&("kv_dtype".to_owned(), "fp8_e4m3".to_owned())));
    assert_eq!(log[1].op, "gdn_conv_update");
    assert_eq!(log[1].count, 48);
    assert_eq!(log[2].op, "gdn_chunk_scan");
    assert!(log[2].kernel.is_empty());
    assert_eq!(log[2].reason, "chunk=32, needs == 64");
    assert_eq!(log[2].count, 1);

    // explain() re-runs a question already recorded; it must not add to the log.
    let before = reg.selections().len();
    reg.explain("attn_prefill_paged", &Geometry::new().num("head_dim", 128));
    assert_eq!(reg.selections().len(), before);
}

// ---------------------------------------------------------------------------- all-reduce

fn ar(n_elem: u64) -> ArShape {
    ArShape {
        n_elem,
        elem_bytes: 2,
        world_size: 2,
    }
}

#[test]
fn the_quantised_all_reduce_carries_peer_visible_workspaces() {
    let r = rules();
    // The prefill case: one 8192-token chunk of hidden 5120 in bf16 = 80 MiB.
    let d = resolve(
        Op::AllReduceWht {
            bits: 6,
            dtype: ArDtype::Bf16,
        },
        Shape::Ar(ar(8192 * 5120)),
        &r,
    )
    .expect("resolve");
    assert_eq!(d.kernel, "saber_ar_oneshot_2rank_wht6_bf16");
    assert_eq!(d.grid, [96, 1, 1]); // the block heuristic saturates at SABER_AR_QNB
    assert_eq!(d.block, [1024, 1, 1]);
    assert_eq!(d.n_launches, 1);
    assert_eq!(d.kernarg_bytes, 96);
    assert_eq!(
        d.arg("n_groups").map(|s| s.value.clone()),
        Some(ArgValue::I32 { ival: 655360 })
    );
    assert_eq!(
        d.arg("n_chunks").map(|s| s.value.clone()),
        Some(ArgValue::I32 { ival: 20480 })
    );
    let kinds: Vec<_> = d
        .workspaces
        .iter()
        .map(|w| (w.name.as_str(), w.kind))
        .collect();
    assert_eq!(
        kinds,
        [
            ("ar_scratch", WorkspaceKind::IpcPeerVisible),
            ("ar_peer_scratch", WorkspaceKind::IpcPeerVisible),
            ("ar_flags", WorkspaceKind::IpcPeerVisible),
            ("ar_peer_flags", WorkspaceKind::IpcPeerVisible),
            // Allocated locally, but its VALUE is cross-rank protocol state: slot = seq&1 picks
            // which half of the double-buffered scratch BOTH ranks use.
            ("ar_seq", WorkspaceKind::IpcPeerVisible),
            ("ar_loc_pack", WorkspaceKind::Scratch),
        ]
    );

    // The decode case: hidden 5120 x (batch 8 x spec 8) tokens.
    let d = resolve(
        Op::AllReduceWht {
            bits: 6,
            dtype: ArDtype::Bf16,
        },
        Shape::Ar(ar(64 * 5120)),
        &r,
    )
    .expect("resolve");
    assert_eq!(d.grid, [29, 1, 1]);
    // wht4 is a symbol, not a recompile.
    let d = resolve(
        Op::AllReduceWht {
            bits: 4,
            dtype: ArDtype::Bf16,
        },
        Shape::Ar(ar(64 * 5120)),
        &r,
    )
    .expect("resolve");
    assert_eq!(d.kernel, "saber_ar_oneshot_2rank_wht4_bf16");
    let d = resolve(
        Op::AllReduceWht {
            bits: 6,
            dtype: ArDtype::Fp16,
        },
        Shape::Ar(ar(64 * 5120)),
        &r,
    )
    .expect("resolve");
    assert_eq!(d.kernel, "saber_ar_oneshot_2rank_wht6_fp16");
}

/// The exact reduce is a serve-path kernel, not a fallback: prod runs it for every message under
/// the 128 KiB threshold, and a batch-1 step at spec 8 is 8 x 5120 x bf16 = 80 KiB.
#[test]
fn the_exact_all_reduce_covers_the_messages_below_the_threshold() {
    let r = rules();
    // (elements, element bytes, payload, expected blocks). The block target is the same
    // nbytes/22400 as the lossy path but counted in 16 B words, and its cap is 24, not 96.
    let cases = [
        (8192u64, 2u64, ArPayload::Bf16, 4u32), // 16 KiB
        (40960, 2, ArPayload::Bf16, 4),         // 80 KiB -- the batch-1 spec-8 serve case
        (65528, 2, ArPayload::Bf16, 5),         // 128 KiB - 16 B, the last exact message
        (8192, 2, ArPayload::Fp16, 4),
        (4096, 4, ArPayload::Fp32, 4),
    ];
    for (n_elem, elem_bytes, dtype, nb) in cases {
        let s = ArShape {
            n_elem,
            elem_bytes,
            world_size: 2,
        };
        let d = resolve(Op::AllReduceExact { dtype }, Shape::Ar(s), &r).expect("resolve");
        let tag = match dtype {
            ArPayload::Bf16 => "bf16",
            ArPayload::Fp16 => "fp16",
            ArPayload::Fp32 => "fp32",
        };
        assert_eq!(d.kernel, format!("saber_ar_oneshot_2rank_exact_{tag}"));
        assert_eq!(d.grid, [nb, 1, 1], "{n_elem} x {elem_bytes} B");
        assert_eq!(d.block, [1024, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert_eq!(d.kernarg_bytes, 72);
        assert_eq!(
            d.arg("n16").map(|x| x.value.clone()),
            Some(ArgValue::I32 {
                ival: (n_elem * elem_bytes / 16) as i64
            })
        );
        // The slot stride is in 16-byte WORDS here and in BYTES in the lossy kernel.
        assert_eq!(
            d.arg("slot_stride16").map(|x| x.value.clone()),
            Some(ArgValue::I32 {
                ival: (r.ar.max_bytes / 16) as i64
            })
        );
        // Same peer-visible workspaces as the lossy path, minus the local packed half: the exact
        // kernel pushes raw payload and folds the peer's copy in place.
        let kinds: Vec<_> = d
            .workspaces
            .iter()
            .map(|w| (w.name.as_str(), w.kind))
            .collect();
        assert_eq!(
            kinds,
            [
                ("ar_scratch", WorkspaceKind::IpcPeerVisible),
                ("ar_peer_scratch", WorkspaceKind::IpcPeerVisible),
                ("ar_flags", WorkspaceKind::IpcPeerVisible),
                ("ar_peer_flags", WorkspaceKind::IpcPeerVisible),
                ("ar_seq", WorkspaceKind::IpcPeerVisible),
            ]
        );
    }
}

/// The policy op: the SIZE decides the `exact` key and the registry decides the kernel, which is
/// the two-step rule `ref_allreduce.py` applies per call.
#[test]
fn the_message_size_picks_the_all_reduce_family() {
    let r = rules();
    let pick = |n_elem: u64, elem_bytes: u64, dtype: ArPayload| {
        resolve(
            Op::AllReduce { dtype },
            Shape::Ar(ArShape {
                n_elem,
                elem_bytes,
                world_size: 2,
            }),
            &r,
        )
        .expect("resolve")
        .kernel
    };
    // Below 128 KiB: exact. 80 KiB is a real decode step, not a corner case.
    assert_eq!(
        pick(8192, 2, ArPayload::Bf16),
        "saber_ar_oneshot_2rank_exact_bf16"
    );
    assert_eq!(
        pick(40960, 2, ArPayload::Bf16),
        "saber_ar_oneshot_2rank_exact_bf16"
    );
    assert_eq!(
        pick(65528, 2, ArPayload::Bf16),
        "saber_ar_oneshot_2rank_exact_bf16"
    );
    // At and above it, the rotated payload at the configured width.
    assert_eq!(r.ar.qbits, 6);
    assert_eq!(
        pick(327680, 2, ArPayload::Bf16),
        "saber_ar_oneshot_2rank_wht6_bf16"
    );
    assert_eq!(
        pick(41943040, 2, ArPayload::Bf16),
        "saber_ar_oneshot_2rank_wht6_bf16"
    );
    assert_eq!(
        pick(327680, 2, ArPayload::Fp16),
        "saber_ar_oneshot_2rank_wht6_fp16"
    );
    // Big enough, but it does not tile a rotation group: the lossy kernel cannot take it.
    assert_eq!(
        pick(327688, 2, ArPayload::Bf16),
        "saber_ar_oneshot_2rank_exact_bf16"
    );
    // fp32 has no lossy arm at all, at any size.
    assert_eq!(
        pick(1048576, 4, ArPayload::Fp32),
        "saber_ar_oneshot_2rank_exact_fp32"
    );
}

/// The fused post-AR epilogue: one workgroup per row, because an RMS needs the whole row.
#[test]
fn the_nq_epilogue_is_one_block_per_row() {
    let r = rules();
    let nq = |m_rows: u32, k_cols: u32| ArNqShape {
        m_rows,
        k_cols,
        eps_bits: 1e-6f32.to_bits(),
        world_size: 2,
    };
    for (m, k) in [(2u32, 4096u32), (8, 5120), (8, 8184)] {
        let d = resolve(Op::AllReduceExactNq, Shape::ArNq(nq(m, k)), &r).expect("resolve");
        assert_eq!(d.kernel, "saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16");
        assert_eq!(d.grid, [m, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert_eq!(d.kernarg_bytes, 108);
        assert_eq!(
            d.arg("k16").map(|x| x.value.clone()),
            Some(ArgValue::I32 {
                ival: (k / 8) as i64
            })
        );
        assert_eq!(d.workspaces.len(), 5);
    }
    // The three conditions the reference entry point throws on, in its order.
    let e = resolve(Op::AllReduceExactNq, Shape::ArNq(nq(8, 4095)), &r).unwrap_err();
    assert!(format!("{e}").contains("K must be a multiple of 8"), "{e}");
    let e = resolve(Op::AllReduceExactNq, Shape::ArNq(nq(8, 10248)), &r).unwrap_err();
    assert!(format!("{e}").contains("register budget"), "{e}");
    let e = resolve(Op::AllReduceExactNq, Shape::ArNq(nq(513, 512)), &r).unwrap_err();
    assert!(format!("{e}").contains("too many rows"), "{e}");
}

/// NEGATIVE CONTROL: `exact` is the key that separates the two families, and saber refuses a
/// request that does not declare it rather than guessing.
#[test]
fn an_all_reduce_request_that_does_not_declare_exact_gets_no_kernel() {
    let r = rules();
    let reg = Registry::new(&r);
    let g = Geometry::new().num("world_size", 2).dtype("dtype", "bf16");
    assert_eq!(reg.select("allreduce", &g).unwrap_err(), "exact not given");
    // Declared, and the exact row is first in preference order.
    let g = g.num("exact", 1);
    assert_eq!(
        reg.select("allreduce", &g).unwrap().entry,
        "ar_oneshot_2rank_exact"
    );
    // The exact family takes fp32; the lossy one does not.
    let g32 = Geometry::new()
        .num("world_size", 2)
        .num("exact", 1)
        .dtype("dtype", "fp32");
    assert!(reg.select("allreduce", &g32).is_ok());
    let e = resolve(
        Op::AllReduceWht {
            bits: 6,
            dtype: ArDtype::Bf16,
        },
        Shape::Ar(ArShape {
            n_elem: 327680,
            elem_bytes: 4,
            world_size: 2,
        }),
        &r,
    );
    assert!(e.is_err(), "a 4-byte payload does not pack 8 to an int4");
}

/// NEGATIVE CONTROLS for the message bounds the caller enforces per call: below
/// `quant_min_bytes` the exact kernel wins, above `max_bytes` the message goes to RCCL, a width
/// this build does not carry is a refusal rather than a silent 6, and three ranks is a different
/// kernel entirely.
#[test]
fn all_reduce_message_bounds_are_refused_not_clamped() {
    let r = rules();
    let op = Op::AllReduceWht {
        bits: 6,
        dtype: ArDtype::Bf16,
    };
    // 8 tokens x 5120 x bf16 = 80 KiB, below the 128 KiB compression threshold.
    let e = resolve(op, Shape::Ar(ar(8 * 5120)), &r).unwrap_err();
    assert!(format!("{e}").contains("below quant_min_bytes"), "{e}");
    // 84 MiB is the slot; 96 MiB is RCCL's.
    let e = resolve(op, Shape::Ar(ar(48 * 1024 * 1024)), &r).unwrap_err();
    assert!(format!("{e}").contains("exceeds the"), "{e}");
    // A message that does not tile a rotation group.
    assert!(resolve(op, Shape::Ar(ar(5120 * 64 + 8)), &r).is_err());
    let e = resolve(
        Op::AllReduceWht {
            bits: 5,
            dtype: ArDtype::Bf16,
        },
        Shape::Ar(ar(64 * 5120)),
        &r,
    )
    .unwrap_err();
    assert!(format!("{e}").contains("bits=5"), "{e}");
    let mut three = ar(64 * 5120);
    three.world_size = 3;
    let e = resolve(op, Shape::Ar(three), &r).unwrap_err();
    assert!(format!("{e}").contains("world_size=3, needs == 2"), "{e}");
}

// ---------------------------------------------------------------------------- kernarg drift

/// NEGATIVE CONTROL: the 136-byte by-value `SABERArgs` is the one place a drifted offset would be
/// invisible — the code object reports a single opaque blob, so nothing but this check ties the
/// host's field layout to the kernel's. Move one field and the manifest must refuse.
#[test]
fn a_drifted_saber_kernarg_layout_is_refused() {
    let dir = prebuilt::dir();
    let mut m = Manifest::load(dir.join("saber_attn.manifest.toml")).expect("load");
    let bytes = std::fs::read(m.hsaco_path(&dir)).expect("blob");
    m.validate_against_hsaco(&bytes)
        .expect("the committed pair agrees");

    // The struct is one argument; grow it by the 4 bytes of trailing padding and the blob says no.
    let k = m
        .kernel
        .iter_mut()
        .find(|k| k.name == "saber_attn_decode_h256_gqa6_fp8kv_nw3")
        .expect("kernel");
    assert_eq!(k.arg[0].size, abi::SABER_ARGS_BYTES);
    k.arg[0].size += 4;
    match m.validate_against_hsaco(&bytes) {
        Err(Error::Validate(v)) => assert!(v.iter().any(|s| s.contains("arg 0")), "{v:?}"),
        other => panic!("a drifted by-value struct must be refused, got {other:?}"),
    }

    // And a signature whose fields pack differently is refused against the manifest.
    let m = Manifest::load(dir.join("saber_gdn_fused_update.manifest.toml")).expect("load");
    m.check_signature(
        "saber_gdn_fused_update_w4k128v128_ab1",
        abi::SIG_SABER_GDN_FUSED_UPDATE,
    )
    .expect("the declared signature packs the way the blob does");
    match m.check_signature(
        "saber_gdn_fused_update_w4k128v128_ab1",
        abi::SIG_SABER_AR_WHT, // a different kernel's layout
    ) {
        Err(Error::Validate(_)) => {}
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ------------------------------------------------------- the row bands (workstream H28)

/// **The chunking law, at the long-context gate's own bins.**
///
/// The slot is 88,080,384 B and a row at hidden 5120 in bf16 is 10,240 B, so 8,601 rows is the
/// largest message that fits whole — which is exactly the length `saber forward --tp 2` refused
/// above until H28. Everything at or below it must still be ONE band, or a decode step and every
/// prefill chunk the engine runs would get a descriptor they have never had.
#[test]
fn the_all_reduce_bands_are_the_message_cut_to_the_slot() {
    let r = rules();
    let hidden = 5120u64;
    let slot = r.ar.max_bytes;
    assert_eq!(
        slot / (hidden * 2),
        8601,
        "the row that sets every count below"
    );

    for (rows, want) in [
        (1u64, 1usize), // a decode step
        (2048, 1),      // the engine's prefill chunk
        (8601, 1),      // the largest message that fits the slot whole
        (8602, 2),      // one row past it
        (4195, 1),      // the gate's 4k bin
        (8153, 1),      // 8k
        (14301, 2),     // 14k -- the first bin the old rule refused
        (22486, 3),     // 22k
        (30740, 4),     // 30k
        (32768, 4),     // the engine's max_ctx
    ] {
        let bands = saber::ar_bands(rows, hidden, 2, slot).expect("banded");
        assert_eq!(bands.len(), want, "{rows} rows");
        // Every band fits the slot, the bands tile the message exactly once, in order, and
        // none of them is empty.
        let mut next = 0u64;
        for b in &bands {
            assert_eq!(b.first_row, next, "{rows} rows: a gap or an overlap");
            assert!(b.rows > 0, "{rows} rows: an empty band");
            assert!(
                b.rows * hidden * 2 <= slot,
                "{rows} rows: a {}-row band is {} B, over the {slot} B slot",
                b.rows,
                b.rows * hidden * 2
            );
            next += b.rows;
        }
        assert_eq!(next, rows, "the bands must cover the message");
        // BALANCED: no two bands differ by more than one row, so no band can fall into the
        // other all-reduce family while its siblings are in this one.
        let (lo, hi) = (
            bands.iter().map(|b| b.rows).min().expect("a band"),
            bands.iter().map(|b| b.rows).max().expect("a band"),
        );
        assert!(
            hi - lo <= 1,
            "{rows} rows: bands {lo}..{hi} are not balanced"
        );
    }
}

/// A row wider than the slot is the message the band law cannot cut, and it refuses by name.
/// This is the control for the test above: the law must still have a case it says no to.
#[test]
fn a_row_wider_than_the_ipc_slot_is_refused_rather_than_split() {
    let r = rules();
    let too_wide = r.ar.max_bytes / 2 + 8; // elements, so bytes are 16 over the slot
    let e = saber::ar_bands(1, too_wide, 2, r.ar.max_bytes).expect_err("one row over the slot");
    assert!(format!("{e}").contains("falls back to RCCL"), "{e}");
    // And a zero-row message is not a message.
    assert!(saber::ar_bands(0, 5120, 2, r.ar.max_bytes).is_err());
}

/// A one-band message resolves to the descriptor it always had, and a banded one declares its
/// band count on the HEAD (structural rule 7) with one dispatch per band.
#[test]
fn a_banded_all_reduce_declares_its_bands_and_names_each_ones_buffers() {
    let r = rules();
    let hidden = 5120u64;

    // One band: byte-identical to `Op::AllReduce` over the whole message.
    let one = saber::resolve_ar_banded(2048, hidden, 2, ArPayload::Bf16, 2, &r, &Registry::new(&r))
        .expect("one band");
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].n_launches, 1);
    let plain = resolve(
        Op::AllReduce {
            dtype: ArPayload::Bf16,
        },
        Shape::Ar(ar(2048 * hidden)),
        &r,
    )
    .expect("the unbanded descriptor");
    assert_eq!(one[0].kernel, plain.kernel);
    assert_eq!(one[0].grid, plain.grid);
    assert_eq!(one[0].block, plain.block);
    assert_eq!(one[0].args, plain.args, "band 0 keeps `in` and `out`");

    // 14,301 rows: two bands, two dispatches, and the head carries the count.
    let two =
        saber::resolve_ar_banded(14301, hidden, 2, ArPayload::Bf16, 2, &r, &Registry::new(&r))
            .expect("two bands");
    assert_eq!(two.len(), 2);
    assert_eq!(two[0].n_launches, 2, "the head declares the whole op");
    assert_eq!(two[1].n_launches, 1);
    // Same family on both, or the message would be reduced two different ways.
    assert_eq!(two[0].op, two[1].op);
    let names = |d: &saber_kernels::rules::LaunchDescriptor| -> Vec<String> {
        d.args
            .iter()
            .filter_map(|s| match &s.value {
                saber_kernels::rules::ArgValue::Buffer { buffer } => Some(buffer.clone()),
                _ => None,
            })
            .collect()
    };
    assert!(names(&two[0]).contains(&"in".to_owned()));
    assert!(names(&two[0]).contains(&"out".to_owned()));
    assert!(names(&two[1]).contains(&"in.b1".to_owned()));
    assert!(names(&two[1]).contains(&"out.b1".to_owned()));
    assert!(
        !names(&two[1]).contains(&"in".to_owned()),
        "band 1 must not bind band 0's buffer"
    );
}

// ---------------------------------------------------------------- ws-h38: the last-arriver merge

/// The last-arriver decode IS the unfused decode's descriptor under another symbol: the same
/// grid (so the same split count), block, SABERArgs blob and partial scratch, plus the arrival
/// counter -- a Persistent, zeroed-once workspace at a FIXED size (the region is keyed per
/// plugin and op, so it must not change size between buckets). One launch where the pair was two.
/// At the gate's ctx 4096 the split law answers 64 at b1s8 and 16 at b4s8 (the batch-dependent
/// law; the engine's batch-invariant plan pins 64): both > 1, both fused.
#[test]
fn the_last_arriver_decode_is_the_decode_descriptor_plus_a_counter() {
    let r = rules();
    for (seqs, q_len, splits, nw) in [
        (1u32, 8u32, 64u32, 3u32),
        (4, 8, 16, 3),
        (1, 5, 64, 2),
        (1, 1, 64, 1),
    ] {
        let s = attn(seqs, q_len, 4096);
        let head = resolve(Op::AttnDecode, Shape::Attn(s), &r).expect("the decode");
        let f = saber::attn_decode_declc(s, &r).expect("the fused decode");
        assert_eq!(f.op, "attn_decode_declc");
        assert_eq!(f.kernel, format!("saber_attn_declc_h256_gqa6_fp8kv_nw{nw}"));
        assert_eq!(
            head.kernel,
            format!("saber_attn_decode_h256_gqa6_fp8kv_nw{nw}")
        );
        assert_eq!((f.grid, f.block), (head.grid, head.block));
        assert_eq!(f.grid[0], splits, "b{seqs}s{q_len}");
        assert_eq!(f.n_launches, 1);
        assert_eq!(f.kernarg_bytes, 152);
        assert_eq!(
            &f.args[..head.args.len()],
            &head.args[..],
            "the decode's own kernargs"
        );
        let c = f.args.last().expect("the counter");
        assert_eq!(
            (c.name.as_str(), c.offset, c.size),
            ("attn_decode_counter", 144, 8)
        );
        let w = f
            .workspaces
            .iter()
            .find(|w| w.name == "attn_decode_counter")
            .expect("the counter workspace");
        assert_eq!(w.kind, WorkspaceKind::Persistent);
        assert!(w.zero_once);
        assert_eq!(w.bytes, 1024);
        assert!(f.workspaces.iter().any(|w| w.name == "attn_decode_scratch"));
        assert_eq!(saber::plugin_for(&f.kernel).expect("mapped"), "saber_attn");
        assert_eq!(
            abi::signature_for(&f.kernel),
            Some(abi::SIG_SABER_ATTN_DECLC)
        );
    }
}

/// NEGATIVE CONTROLS: one split has nothing to merge (the pair stays), the bf16 cache has no
/// instantiation, and a counter table too small for the batch's items is refused rather than
/// letting two items share a counter.
#[test]
fn a_last_arriver_decode_that_cannot_be_served_is_refused() {
    let r = rules();
    assert!(
        saber::attn_decode_declc(attn(1, 8, 4096), &r).is_ok(),
        "the control"
    );
    let e = saber::attn_decode_declc(attn(1, 8, 16), &r).unwrap_err();
    assert!(e.to_string().contains("nothing to merge"), "{e}");
    let mut bf = attn(1, 8, 4096);
    bf.kv_dtype = KvDtype::Bf16;
    bf.kv_block_stride *= 2;
    bf.kv_head_stride *= 2;
    assert!(saber::attn_decode_declc(bf, &r).is_err(), "bf16 KV");
    let text = include_str!("../rules/saber_gfx1201.toml");
    let tiny = text.replace("capacity_bytes = 1024\n", "capacity_bytes = 4\n");
    assert_ne!(tiny, text, "the edit must land");
    let tiny = SaberRules::from_toml(&tiny).expect("parses");
    let e = saber::attn_decode_declc(attn(1, 8, 4096), &tiny).unwrap_err();
    assert!(e.to_string().contains("counters"), "{e}");
}

/// Every `[[attn_decode_declc]]` symbol and control is in the committed `saber_attn` code object,
/// with the declared kernarg size, and the controls are exactly the symbols' `_drop1` twins.
#[test]
fn every_last_arriver_symbol_and_control_is_committed() {
    let r = rules();
    assert_eq!(r.attn_decode_declc.len(), 4, "nw 1..4");
    let m = Manifest::load(prebuilt::dir().join("saber_attn.manifest.toml")).expect("manifest");
    for row in &r.attn_decode_declc {
        assert_eq!(row.control, format!("{}_drop1", row.symbol));
        assert_eq!(row.max_splits, r.decode_splits.ceiling);
        for s in [&row.symbol, &row.control] {
            let k = m
                .kernel
                .iter()
                .find(|k| &k.name == s)
                .unwrap_or_else(|| panic!("no committed kernel `{s}`"));
            assert_eq!(k.kernarg_segment_size, 152, "{s}");
            assert_eq!(k.max_flat_workgroup_size, row.nw * 32, "{s}");
            assert_eq!(saber::plugin_for(s).expect("mapped"), "saber_attn");
        }
    }
}
