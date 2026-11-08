use saber_kernels::rules::ArgValue;
use saber_kernels::saber::{
    resolve, ArPayload, ArShape, Op, SaberRules, Shape, RCCL_BF16_KERNEL, RCCL_PLUGIN,
};

#[test]
fn rccl_policy_has_an_external_signature_and_no_peer_memory() {
    let mut rules = SaberRules::prod_saber().unwrap();
    rules.ar.external_rccl = true;
    for count in [5120u64, 327680, 10485760] {
        let desc = resolve(
            Op::AllReduce {
                dtype: ArPayload::Bf16,
            },
            Shape::Ar(ArShape {
                n_elem: count,
                elem_bytes: 2,
                world_size: 2,
            }),
            &rules,
        )
        .unwrap();
        assert_eq!(desc.plugin, RCCL_PLUGIN);
        assert_eq!(desc.kernel, RCCL_BF16_KERNEL);
        assert!(desc.workspaces.is_empty());
        assert_eq!(desc.args.len(), 3);
        assert_eq!(
            desc.arg("count").unwrap().value,
            ArgValue::I32 { ival: count as i64 }
        );
        assert_eq!(desc.n_launches, 1);
    }
}

#[test]
fn rccl_refuses_unsupported_data_layouts_instead_of_reinterpreting_memory() {
    let mut rules = SaberRules::prod_saber().unwrap();
    rules.ar.external_rccl = true;
    for (dtype, bytes, world) in [
        (ArPayload::Bf16, 4, 2),
        (ArPayload::Fp32, 4, 2),
        (ArPayload::Fp16, 2, 2),
        (ArPayload::Bf16, 2, 3),
    ] {
        assert!(resolve(
            Op::AllReduceExact { dtype },
            Shape::Ar(ArShape {
                n_elem: 5120,
                elem_bytes: bytes,
                world_size: world
            }),
            &rules
        )
        .is_err());
    }
}

#[test]
fn the_native_backend_remains_the_default() {
    let rules = SaberRules::prod_saber().unwrap();
    assert!(!rules.ar.external_rccl);
    let desc = resolve(
        Op::AllReduce {
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
    assert_ne!(desc.plugin, RCCL_PLUGIN);
    assert!(!desc.workspaces.is_empty());
}
