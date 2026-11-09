use saber_kernels::saber::{self, ArNqShape, ArPayload, ArShape, Op, SaberRules, Shape};

#[test]
fn host_transport_selects_bounded_exact_objects_and_keeps_native_default() {
    let mut rules = SaberRules::prod_saber().unwrap();
    let shape = Shape::Ar(ArShape {
        n_elem: 5120,
        elem_bytes: 2,
        world_size: 2,
    });
    let op = Op::AllReduceExact {
        dtype: ArPayload::Bf16,
    };
    let native = saber::resolve(op, shape, &rules).unwrap();
    assert_eq!(native.plugin, "saber_ar_exact");
    rules.ar.host_mapped = true;
    let host = saber::resolve(op, shape, &rules).unwrap();
    assert_eq!(host.plugin, "saber_ar_host_exact");
    assert_eq!(host.kernel, native.kernel);
    assert_eq!(host.kernarg_bytes, native.kernarg_bytes);
    assert_eq!(host.grid, native.grid);
    assert_eq!(host.workspaces.len(), native.workspaces.len());
    rules.ar.race_check = true;
    let checked = saber::resolve(op, shape, &rules).unwrap();
    assert_eq!(checked.plugin, "saber_ar_host_exact_chk");
    assert!(checked.kernel.ends_with("_chk"));
}

#[test]
fn host_norm_epilogue_uses_its_bounded_checked_twin() {
    let mut rules = SaberRules::prod_saber().unwrap();
    rules.ar.host_mapped = true;
    let shape = Shape::ArNq(ArNqShape {
        m_rows: 8,
        k_cols: 5120,
        eps_bits: 1e-6_f32.to_bits(),
        world_size: 2,
    });
    for checked in [false, true] {
        rules.ar.race_check = checked;
        let desc = saber::resolve(Op::AllReduceExactNq, shape, &rules).unwrap();
        assert_eq!(
            desc.plugin,
            if checked {
                "saber_ar_host_exact_chk"
            } else {
                "saber_ar_host_exact"
            }
        );
        assert!(desc.kernel.contains("exact_nq"));
        assert_eq!(desc.kernel.ends_with("_chk"), checked);
    }
}

#[test]
fn banded_entrypoint_also_selects_the_bounded_host_object() {
    let mut rules = SaberRules::prod_saber().unwrap();
    rules.ar.host_mapped = true;
    rules.ar.race_check = true;
    let reg = saber::Registry::new(&rules);
    let descs = saber::resolve_ar_banded(8, 5120, 2, ArPayload::Bf16, 2, &rules, &reg).unwrap();
    assert_eq!(descs.len(), 1);
    assert_eq!(descs[0].plugin, "saber_ar_host_exact_chk");
}

#[test]
fn hybrid_profile_keeps_small_messages_native_and_bulk_messages_external() {
    let mut rules = SaberRules::prod_saber().unwrap();
    rules.ar.host_mapped = true;
    rules.ar.external_rccl = true;
    rules.ar.external_rccl_min_bytes = 65 * 5120 * 2;
    for rows in [1, 8, 10, 16, 64, 65, 96, 2048] {
        let registry = saber::Registry::new(&rules);
        let descs =
            saber::resolve_ar_banded(rows, 5120, 2, ArPayload::Bf16, 2, &rules, &registry).unwrap();
        let expected = if rows >= 65 {
            saber::RCCL_PLUGIN
        } else {
            "saber_ar_host_exact"
        };
        assert!(descs.iter().all(|d| d.plugin == expected), "rows={rows}");
        if rows >= 65 {
            assert!(descs.iter().all(|d| d.workspaces.is_empty()));
        }
    }
    rules.ar.external_rccl_min_bytes = 0;
    let desc = saber::resolve(
        Op::AllReduceExact {
            dtype: ArPayload::Bf16,
        },
        Shape::Ar(ArShape {
            n_elem: 5120,
            elem_bytes: 2,
            world_size: 2,
        }),
        &rules,
    )
    .unwrap();
    assert_eq!(
        desc.plugin,
        saber::RCCL_PLUGIN,
        "all-RCCL profile unchanged"
    );
}
