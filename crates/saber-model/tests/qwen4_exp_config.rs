//! `Qwen4ExpConfig` against the real Flash-Next `config.json` files, and doctored copies as the
//! negative controls: every doctored file must be REFUSED, and the refusal must name the field.
//!
//! The two fixtures under `tests/data/` are byte copies of the checkpoints' own `config.json`
//! (`/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4` and `-NVFP4`); when the checkpoint is on
//! this box, `fixture_is_the_checkpoints_config` holds them to it. `qwen4_exp_quark_20261026`
//! is the hub's later revision of the Quark file (see [`QUARK_1026`]); the box's checkpoint is
//! the original, so it is held to the hub copy only.

use std::path::Path;

use saber_model::qwen4_exp::config::glob_match;
use saber_model::qwen4_exp::config::{nth_prime_after, DECLARED_DEFAULTS};
use saber_model::qwen4_exp::{LayerKind, ModeloptAlgo, QuantConfig, Qwen4ExpConfig, ScaleEncoding};
use serde_json::Value;

const QUARK: &str = include_str!("data/qwen4_exp_quark.config.json");
const NVFP4: &str = include_str!("data/qwen4_exp_nvfp4.config.json");
const FP4LOCAL: &str = include_str!("data/qwen4_exp_fp4local.config.json");
/// AMD's 2026-10-26 revision of the Quark repo's `config.json` ("expand exclude for MTP experts +
/// embed"; hub commit b71e28afcd4b, blob 721191e7b4a5): a byte copy of
/// `https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4/raw/main/config.json`. The shards
/// did not change; the revision adds `zero_point_type: int32` to both MX specs, Quark version
/// 0.14, and 1,667 exclude entries (2,462 total, none removed).
const QUARK_1026: &str = include_str!("data/qwen4_exp_quark_20261026.config.json");

fn parse(s: &str) -> Qwen4ExpConfig {
    Qwen4ExpConfig::from_config_json(s).expect("the real config parses")
}

fn refused(v: &Value) -> String {
    match Qwen4ExpConfig::from_config_json(&v.to_string()) {
        Ok(_) => panic!("a doctored config was ACCEPTED"),
        Err(e) => e.to_string(),
    }
}

fn doctor(f: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(QUARK).unwrap();
    f(&mut v);
    refused(&v)
}

#[test]
fn fixture_is_the_checkpoints_config() {
    for (fixture, dir) in [
        (QUARK, "/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4"),
        (NVFP4, "/mnt/nvme/models/Qwen3.8-Flash-Next-NVFP4"),
        (FP4LOCAL, "/mnt/nvme/models/Qwen3.8-Flash-Next-FP4-Local"),
    ] {
        let p = Path::new(dir).join("config.json");
        if !p.exists() {
            eprintln!("SKIP: {} absent; the fixture stands alone", p.display());
            continue;
        }
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            fixture,
            "{}",
            p.display()
        );
    }
}

#[test]
fn quark_config_names_every_forward_fact() {
    let c = parse(QUARK);
    assert_eq!(c.hidden_size, 2560);
    assert_eq!(c.num_layers(), 48);
    let qsa: Vec<u32> = (0..48)
        .filter(|&i| c.layers[i as usize] == LayerKind::Qsa)
        .collect();
    assert_eq!(qsa, (0..12).map(|i| 4 * i + 3).collect::<Vec<_>>());
    assert_eq!((c.hc.count, c.hc.lowrank, c.hc_width()), (4, 320, 10240));
    assert_eq!(
        (
            c.attention.num_heads,
            c.attention.num_kv_heads,
            c.attention.head_dim
        ),
        (24, 2, 256)
    );
    assert_eq!(c.attention.num_heads / c.attention.num_kv_heads, 12);
    assert!(c.attention.output_gate_sigmoid);
    assert_eq!(c.q_proj_rows(), 12288);
    assert_eq!(c.rope.rotary_dim(256), 64);
    assert_eq!(c.rope.mrope_section, vec![11, 11, 10]);
    assert!(c.rope.mrope_interleaved);
    assert_eq!(c.rope.theta, 1e7);
    assert_eq!((c.qsa.n_heads, c.qsa.kv_heads, c.qsa.head_dim), (4, 1, 128));
    assert_eq!(
        (c.qsa.budget, c.qsa.block_topk(), c.qsa.qk_proj_rows()),
        (2048, 512, 640)
    );
    assert_eq!(
        (
            c.gdn.qkv_rows(),
            c.gdn.value_dim(),
            c.gdn.key_dim(),
            c.gdn.conv_kernel
        ),
        (10240, 6144, 2048, 4)
    );
    assert_eq!(
        (
            c.moe.num_experts,
            c.moe.top_k,
            c.moe.intermediate,
            c.moe.shared_intermediate
        ),
        (512, 10, 640, 640)
    );
    assert!(c.moe.norm_topk_prob);
    assert_eq!(c.ple.layer_indices(), vec![1]);
    assert_eq!((c.ple.ngram_heads(), c.ple.row_dim()), (16, 160));
    assert_eq!((c.ple.conv_dilation(), c.ple.conv_state_rows()), (3, 9));
    assert_eq!(c.ple.split_parts, 128);
    assert_eq!(c.ple.hash_seed, 1234);
    // 16 next-primes above 20 M, padded to 128, split 128 ways: the checkpoint's shards are
    // [2,500,012, 160] each (FN1 census).
    assert_eq!(c.ple_rows_per_shard(), 2_500_012);
    assert!(c.ple_table_rows() <= 128 * 2_500_012);
    assert_eq!(c.mtp.num_layers, 1);
    assert_eq!(c.mtp.layer_kinds, vec![LayerKind::Qsa]);
    assert!(!c.mtp.dedicated_embeddings);
    assert_eq!(c.vocab_size, 248_320);
    assert!(!c.tie_word_embeddings);
    assert_eq!((c.bos_token_id, c.eos_token_id), (248_044, 248_044));
    assert_eq!(c.vision.depth, 27);
    assert_eq!(c.vision.out_hidden_size, 2560);
    match &c.quant {
        QuantConfig::QuarkMxfp4 {
            exclude,
            weight,
            input,
            ..
        } => {
            assert_eq!(exclude.len(), 795);
            assert!(!weight.dynamic && input.dynamic);
        }
        q => panic!("expected Quark, got {q:?}"),
    }
    assert_eq!(c.declared_defaults().len(), DECLARED_DEFAULTS.len());
}

#[test]
fn nvfp4_config_parses_as_modelopt_with_the_same_text_facts() {
    let q = parse(QUARK);
    let n = parse(NVFP4);
    match &n.quant {
        QuantConfig::ModelOpt { layers, .. } => {
            assert_eq!(
                layers
                    .iter()
                    .filter(|l| l.algo == ModeloptAlgo::Nvfp4)
                    .count(),
                48
            );
            assert!(layers
                .iter()
                .any(|l| l.module.ends_with("ngram_embedding") && l.algo == ModeloptAlgo::Fp8));
            assert!(layers.iter().all(|l| l.encoding == ScaleEncoding::Plain));
        }
        other => panic!("expected ModelOpt, got {other:?}"),
    }
    let mut n2 = n.clone();
    n2.quant = q.quant.clone();
    assert_eq!(
        n2, q,
        "the two checkpoints' text/vision facts must be identical"
    );
}

#[test]
fn prime_helper_matches_upstream_semantics() {
    assert_eq!(nth_prime_after(10, 1), 11);
    assert_eq!(nth_prime_after(10, 2), 13);
    assert_eq!(nth_prime_after(19_999_999, 1), 20_000_003);
}

// ---- negative controls: each doctored file is refused, and the refusal names the field -----

#[test]
fn refuses_an_unknown_text_field_by_name() {
    let e = doctor(|v| {
        v["text_config"]["hc_dropout"] = Value::from(0.1);
    });
    assert!(e.contains("text_config.hc_dropout"), "{e}");
    assert!(e.contains("unknown field"), "{e}");
}

#[test]
fn refuses_an_unknown_top_level_field_by_name() {
    let e = doctor(|v| {
        v["rope_scaling"] = Value::Null;
    });
    assert!(
        e.contains("rope_scaling") && e.contains("unknown field"),
        "{e}"
    );
}

#[test]
fn refuses_unknown_nested_fields_by_path() {
    let e = doctor(|v| {
        v["text_config"]["rope_parameters"]["factor"] = Value::from(2.0);
    });
    assert!(e.contains("text_config.rope_parameters.factor"), "{e}");
    let e = doctor(|v| {
        v["text_config"]["mtp"]["num_nextn_predict_layers"] = Value::from(2);
    });
    assert!(
        e.contains("text_config.mtp.num_nextn_predict_layers"),
        "{e}"
    );
    let e = doctor(|v| {
        v["vision_config"]["window_size"] = Value::from(112);
    });
    assert!(e.contains("vision_config.window_size"), "{e}");
    let e = doctor(|v| {
        v["quantization_config"]["global_quant_config"]["weight"]["zero_point"] = Value::from(0);
    });
    assert!(
        e.contains("quantization_config.global_quant_config.weight.zero_point"),
        "{e}"
    );
}

#[test]
fn refuses_a_missing_required_field_by_name() {
    for key in [
        "hc_count",
        "hc_lowrank",
        "ple_layer_ids",
        "split_ngram_parts",
        "indexer_budget",
        "num_experts_per_tok",
        "shared_expert_intermediate_size",
        "linear_num_value_heads",
        "vocab_size",
        "mtp_num_hidden_layers",
    ] {
        let e = doctor(|v| {
            v["text_config"].as_object_mut().unwrap().remove(key);
        });
        assert!(
            e.contains(&format!("text_config.{key}")) && e.contains("missing"),
            "{key}: {e}"
        );
    }
    let e = doctor(|v| {
        v["vision_config"].as_object_mut().unwrap().remove("depth");
    });
    assert!(
        e.contains("vision_config.depth") && e.contains("missing"),
        "{e}"
    );
}

#[test]
fn refuses_values_the_forward_cannot_run() {
    // A layer schedule that disagrees with full_attention_interval.
    let e = doctor(|v| {
        v["text_config"]["layer_types"][3] = Value::from("linear_attention");
    });
    assert!(e.contains("layer_types[3]"), "{e}");
    // A PLE layer out of range (1-based).
    let e = doctor(|v| {
        v["text_config"]["ple_layer_ids"] = serde_json::json!([49]);
    });
    assert!(e.contains("ple_layer_ids"), "{e}");
    // An indexer budget upstream refuses.
    let e = doctor(|v| {
        v["text_config"]["indexer_budget"] = Value::from(1000);
    });
    assert!(e.contains("indexer_budget"), "{e}");
    // An MX scale format that is not e8m0.
    let e = doctor(|v| {
        v["quantization_config"]["global_quant_config"]["weight"]["scale_format"] =
            Value::from("e4m3");
    });
    assert!(e.contains("weight.scale_format"), "{e}");
    // A declared default stated with an unsupported value.
    let e = doctor(|v| {
        v["text_config"]["decoder_sparse_step"] = Value::from(2);
    });
    assert!(e.contains("decoder_sparse_step"), "{e}");
    // Tied embeddings disagreeing between the top level and text_config.
    let e = doctor(|v| {
        v["tie_word_embeddings"] = Value::Bool(true);
    });
    assert!(e.contains("tie_word_embeddings"), "{e}");
    // mrope sections that do not cover the rotary dims.
    let e = doctor(|v| {
        v["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([11, 11, 11]);
    });
    assert!(e.contains("mrope_section"), "{e}");
}

#[test]
fn refuses_a_quark_exclude_that_does_not_match_the_model() {
    // An entry naming no module of this model.
    let e = doctor(|v| {
        v["quantization_config"]["exclude"]
            .as_array_mut()
            .unwrap()
            .push(Value::from(
                "model.language_model.layers.3.linear_attn.in_proj_qkv",
            ));
    });
    assert!(
        e.contains("layers.3.linear_attn.in_proj_qkv") && e.contains("not a linear"),
        "{e}"
    );
    // A dense linear dropped from the exclude list = Quark would have quantized it.
    let e = doctor(|v| {
        let ex = v["quantization_config"]["exclude"].as_array_mut().unwrap();
        ex.retain(|x| x.as_str() != Some("model.language_model.layers.7.self_attn.o_proj"));
    });
    assert!(
        e.contains("layers.7.self_attn.o_proj") && e.contains("not excluded"),
        "{e}"
    );
}

fn doctor_1026(f: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(QUARK_1026).unwrap();
    f(&mut v);
    refused(&v)
}

/// The 2026-10-26 Quark config parses to the same model as the original: identical text /
/// vision facts and MX specs; only the exclude list (a superset) and the version differ.
#[test]
fn quark_1026_config_parses_to_the_same_model() {
    let old = parse(QUARK);
    let new = parse(QUARK_1026);
    let (
        QuantConfig::QuarkMxfp4 {
            weight: ow,
            input: oi,
            exclude: oe,
            version: ov,
        },
        QuantConfig::QuarkMxfp4 {
            weight: nw,
            input: ni,
            exclude: ne,
            version: nv,
        },
    ) = (&old.quant, &new.quant)
    else {
        panic!("both must be Quark");
    };
    assert_eq!(
        (ow, oi),
        (nw, ni),
        "the weight / input MX specs must not change"
    );
    assert_eq!(
        (ov.as_str(), nv.as_str()),
        ("0.13+c3526c2f56c", "0.14+67d3ddd8fc")
    );
    assert_eq!((oe.len(), ne.len()), (795, 2462));
    assert!(oe.iter().all(|e| ne.contains(e)), "nothing was removed");
    let added: Vec<&String> = ne.iter().filter(|e| !oe.contains(e)).collect();
    let count = |f: &dyn Fn(&str) -> bool| added.iter().filter(|e| f(e)).count();
    assert_eq!(count(&|e| e.starts_with("mtp.layers.0.mlp.experts.")), 1536);
    assert_eq!(
        count(&|e| e.contains(".ple.ple_embedding.ngram_embedding.shard_")),
        128
    );
    for m in [
        "mtp.fc_embedding",
        "model.visual.patch_embed.proj",
        "model.visual.pos_embed",
    ] {
        assert!(added.iter().any(|e| e.as_str() == m), "{m}");
    }
    assert_eq!(added.len(), 1536 + 128 + 3);
    let mut same = new.clone();
    same.quant = old.quant.clone();
    assert_eq!(
        same, old,
        "the two revisions' model facts must be identical"
    );
}

#[test]
fn quark_1026_refuses_a_zero_point_type_other_than_int32() {
    for spec in ["weight", "input_tensors"] {
        let e = doctor_1026(|v| {
            v["quantization_config"]["global_quant_config"][spec]["zero_point_type"] =
                Value::from("int8");
        });
        assert!(
            e.contains(&format!(
                "quantization_config.global_quant_config.{spec}.zero_point_type"
            )) && e.contains("int8"),
            "{e}"
        );
    }
    // Absent is the 0.13 file (parsed above); `int32` is the 0.14 file. Both stand.
}

#[test]
fn quark_1026_refuses_exclude_entries_that_name_nothing() {
    for bad in [
        // Expert index past num_experts (512).
        "mtp.layers.0.mlp.experts.512.gate_proj",
        // MTP layer index past mtp_num_hidden_layers (1).
        "mtp.layers.1.mlp.experts.0.up_proj",
        // PLE shard past split_ngram_parts (128).
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_128",
        // PLE shards at a layer the PLE does not run at.
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0",
        // A typo of an allowed non-linear.
        "model.visual.pos_embeds",
    ] {
        let e = doctor_1026(|v| {
            v["quantization_config"]["exclude"]
                .as_array_mut()
                .unwrap()
                .push(Value::from(bad));
        });
        assert!(e.contains(bad) && e.contains("not a linear"), "{bad}: {e}");
    }
}

#[test]
fn quark_1026_refuses_a_main_dense_linear_left_out_of_exclude() {
    for m in [
        "model.language_model.layers.7.self_attn.o_proj",
        "model.language_model.layers.1.ple.key_proj",
        "lm_head",
    ] {
        let e = doctor_1026(|v| {
            let ex = v["quantization_config"]["exclude"].as_array_mut().unwrap();
            ex.retain(|x| x.as_str() != Some(m));
        });
        assert!(e.contains(m) && e.contains("not excluded"), "{m}: {e}");
    }
}

/// FN144: the FP4-Local config -- the same text / vision facts as Quark; 545 modules: 48 CSF
/// NVFP4 routed-expert blocks, the PLE table NVFP4 plain, 438 MXFP8, 58 W4A16 NVFP4; ten
/// `ignore` globs that keep the MTP attention / fc / shared expert bf16.
#[test]
fn fp4local_config_parses_every_module_algo_and_encoding() {
    let q = parse(QUARK);
    let f = parse(FP4LOCAL);
    assert!(f.quant.is_fp4_local() && !parse(NVFP4).quant.is_fp4_local());
    let QuantConfig::ModelOpt { layers, ignore, .. } = &f.quant else {
        panic!("expected ModelOpt");
    };
    assert_eq!(layers.len(), 545);
    let count = |a: ModeloptAlgo, e: ScaleEncoding| {
        layers
            .iter()
            .filter(|l| l.algo == a && l.encoding == e)
            .count()
    };
    assert_eq!(count(ModeloptAlgo::Nvfp4, ScaleEncoding::Csf), 48);
    assert_eq!(count(ModeloptAlgo::Nvfp4, ScaleEncoding::Plain), 1);
    assert_eq!(count(ModeloptAlgo::Mxfp8, ScaleEncoding::Plain), 438);
    assert_eq!(count(ModeloptAlgo::W4a16Nvfp4, ScaleEncoding::Plain), 58);
    assert_eq!(ignore.len(), 10);
    let lm = "model.language_model.layers";
    for (m, a) in [
        (format!("{lm}.0.mlp.experts"), ModeloptAlgo::Nvfp4),
        (
            format!("{lm}.0.linear_attn.in_proj_qkv"),
            ModeloptAlgo::Mxfp8,
        ),
        (
            format!("{lm}.3.self_attn.indexer.index_qk_proj"),
            ModeloptAlgo::Mxfp8,
        ),
        (
            format!("{lm}.5.mlp.shared_expert.down_proj"),
            ModeloptAlgo::Mxfp8,
        ),
        (
            "mtp.layers.0.mlp.experts".to_string(),
            ModeloptAlgo::W4a16Nvfp4,
        ),
        (
            "model.visual.merger.linear_fc1".to_string(),
            ModeloptAlgo::W4a16Nvfp4,
        ),
    ] {
        assert_eq!(f.quant.modelopt_layer(&m).map(|l| l.algo), Some(a), "{m}");
    }
    for m in [
        "lm_head",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.mlp.shared_expert.up_proj",
        "mtp.pre_fc_norm_hidden",
    ] {
        assert!(f.quant.modelopt_ignored(m), "{m} must be ignored");
    }
    assert!(!f.quant.modelopt_ignored("mtp.layers.0.mlp.experts"));
    let mut f2 = f.clone();
    f2.quant = q.quant.clone();
    assert_eq!(f2, q, "FP4-Local's text/vision facts must equal Quark's");
}

#[test]
fn fp4local_doctored_quantization_is_refused_by_name() {
    let base: Value = serde_json::from_str(FP4LOCAL).unwrap();
    const QL: &str = "model.language_model.layers.0.mlp.experts";
    type Doctor = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(Doctor, &str)> = vec![
        (
            Box::new(|v| {
                v["quantization_config"]["quantized_layers"][QL]["weight_scale_encoding"] =
                    "rle".into()
            }),
            "weight_scale_encoding",
        ),
        (
            Box::new(|v| {
                v["quantization_config"]["quantized_layers"]["mtp.layers.0.mlp.experts"]
                    ["weight_scale_encoding"] = "csf".into()
            }),
            "weight_scale_encoding",
        ),
        (
            Box::new(|v| {
                v["quantization_config"]["quantized_layers"]
                    ["model.language_model.layers.0.linear_attn.in_proj_qkv"]["group_size"] =
                    16.into()
            }),
            "group_size",
        ),
        (
            Box::new(|v| {
                v["quantization_config"]["quantized_layers"][QL]["quant_algo"] = "MXFP6".into()
            }),
            "quant_algo",
        ),
        (
            Box::new(|v| v["quantization_config"]["kv_cache_quant_algo"] = "FP8".into()),
            "kv_cache_quant_algo",
        ),
        (
            Box::new(|v| v["quantization_config"]["quantized_layers"][QL]["extra"] = 1.into()),
            "extra",
        ),
    ];
    for (f, field) in cases {
        let mut v = base.clone();
        f(&mut v);
        let e = refused(&v);
        assert!(e.contains(field), "refusal `{e}` must name `{field}`");
    }
}

#[test]
fn ignore_globs_match_like_fnmatch_star() {
    assert!(glob_match(
        "mtp.layers.0.self_attn*",
        "mtp.layers.0.self_attn.q_proj"
    ));
    assert!(glob_match("mtp.pre_fc_norm_*", "mtp.pre_fc_norm_embedding"));
    assert!(glob_match("lm_head", "lm_head"));
    assert!(!glob_match("lm_head", "lm_head.x"));
    assert!(glob_match("a*b*c", "aXXbYYc"));
    assert!(!glob_match("a*b*c", "aXXcYYb"));
    assert!(!glob_match("ab*ba", "aba"));
}
