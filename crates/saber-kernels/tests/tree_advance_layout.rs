use saber_kernels::rules::ArgValue;
use saber_kernels::saber::{self, GdnShape, SaberRules};

fn shape(rows: u32) -> GdnShape {
    GdnShape {
        n_seqs: 1,
        h: 24,
        hg: 8,
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        maxq: rows,
        o_rows: rows,
        xpitch: 8192,
        cs_seq: 92160,
        cs_dim: 18,
        cs_tok: 1,
        ci_stride: 4,
        ab_stride: 48,
        st_slot: 24 * 128 * 128,
        st_head: 128 * 128,
        sidx_stride: rows as i64,
        scale_bits: (1.0_f32 / 128.0_f32.sqrt()).to_bits(),
        softplus_thr_bits: 20.0_f32.to_bits(),
        has_naccept: true,
        has_bias: true,
        state_f16: true,
    }
}

#[test]
fn state_advance_keeps_the_record_layout_of_both_tree_forms() {
    let rules = SaberRules::prod_saber().unwrap();
    for (rows, plugin, kernel, floats) in [
        (
            10,
            "p27_tree_gdn",
            "p27_gdn_tree_advance_f16s_k128_v128",
            10 * 256 + 32,
        ),
        (
            16,
            "p27_tree_gdn16",
            "p27_gdn_tree16_advance_f16s_k128_v128",
            16 * 256 + 32,
        ),
    ] {
        let update = saber::tree_gdn_update(shape(rows), &rules).unwrap();
        let from_update = saber::tree_gdn_advance_like(&update).unwrap();
        let direct = saber::tree_gdn_advance(shape(rows)).unwrap();
        for advance in [&from_update, &direct] {
            assert_eq!(advance.plugin, plugin);
            assert_eq!(advance.kernel, kernel);
            assert_eq!(
                advance.arg("ts_head").unwrap().value,
                ArgValue::I64 { ival: floats }
            );
            assert_eq!(
                advance.arg("ts_seq").unwrap().value,
                ArgValue::I64 { ival: 24 * floats }
            );
            assert_eq!(advance.grid, [1, 24, 1]);
        }
        for name in ["st_slot", "st_head", "ts_seq", "ts_head"] {
            assert_eq!(
                from_update.arg(name).unwrap().value,
                update.arg(name).unwrap().value
            );
        }
        let batched = saber::tree_gdn_advance_layers(&from_update, 48).unwrap();
        assert_eq!(batched.grid, [48, 24, 1]);
        let expected = if rows == 16 {
            (
                "p27_tree_advl16",
                "p27_gdn_tree_advance_layers_t16_f16s_k128_v128",
            )
        } else {
            (
                "p27_tree_advl",
                "p27_gdn_tree_advance_layers_f16s_k128_v128",
            )
        };
        assert_eq!((batched.plugin.as_str(), batched.kernel.as_str()), expected);
        for name in ["st_slot", "st_head", "ts_seq", "ts_head"] {
            assert_eq!(
                batched.arg(name).unwrap().value,
                from_update.arg(name).unwrap().value
            );
        }
        let mut wrong_record = from_update.clone();
        wrong_record
            .args
            .iter_mut()
            .find(|s| s.name == "ts_head")
            .unwrap()
            .value = ArgValue::I64 {
            ival: if rows == 16 {
                10 * 256 + 32
            } else {
                16 * 256 + 32
            },
        };
        assert!(saber::tree_gdn_advance_layers(&wrong_record, 48).is_err());
    }
    assert!(saber::tree_gdn_advance(shape(17)).is_err());
}
