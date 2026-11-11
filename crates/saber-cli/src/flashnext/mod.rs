//! The Flash-Next engine path (FN12, plan §8 F3): `saber engine --pack <flash-next pack>` boots
//! this module instead of the 27B's path. The pack's `manifest.json` names the model
//! (`"model": "qwen4_exp"`) and the checkpoint's `config.json` must parse as
//! `saber_model::qwen4_exp::Qwen4ExpConfig`; nothing of the 27B's boot runs on a Flash-Next pack and
//! nothing here runs on a 27B pack (their manifests carry no `model` key).
//!
//! * [`kern`] -- kernel declarations, kernarg images, the arena and its seed law;
//! * [`exec`] -- the layer executor (`layer_graph`'s ops as launches);
//! * [`plan`] -- host emission: the arena sized by the widest chunk, launch counts vs the graph;
//! * [`store`] / [`model`] -- weight bytes (checkpoint or sha256-checked pack records) on the card;
//! * [`dev`] -- one rank on its card, the eager launch, the two-rank reduce;
//! * [`run`] -- the drivers: gate A (TP=1, layer-major, vs the Python harness) and the engine
//!   (TP=2 / EP=2 with the tiers).

pub mod bufs;
pub mod couple;
pub mod depth;
pub mod dev;
pub mod dstep;
pub mod exec;
pub mod fn150;
pub mod fp4local;
pub mod hand;
mod host_ar;
pub mod hc8;
pub mod keepalive;
pub mod kern;
pub mod kvscale;
pub mod kvwire;
pub mod lazyset;
pub mod model;
pub mod mtp8;
pub mod mtpx4;
pub mod overlap;
pub mod plan;
pub mod ple121;
pub mod run;
pub mod seqpool;
pub mod serve;
pub mod shx114;
pub mod stepledger;
pub mod store;
pub mod trunk4;
pub mod trunk8;
pub mod vision;
pub mod worker;
pub mod xsplit;
pub mod xtp2;

use std::path::Path;

use anyhow::Context;

/// The pack's model, from its `manifest.json` (`None` = no `model` key: a 27B pack).
pub fn pack_model(pack: &Path) -> anyhow::Result<Option<String>> {
    let m = pack.join("manifest.json");
    if !pack.is_dir() || !m.is_file() {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&m).with_context(|| m.display().to_string())?,
    )
    .with_context(|| format!("{}: not JSON", m.display()))?;
    Ok(v.get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string))
}

/// Whether `pack` is a Flash-Next pack (the model selection law).
pub fn is_flashnext_pack(pack: &Path) -> anyhow::Result<bool> {
    Ok(pack_model(pack)?.as_deref() == Some("qwen4_exp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_selection_reads_the_manifest_and_nothing_else() {
        // Under `target/`, removed on drop -- never `/tmp` (FN51h).
        let d = saber_test_scratch::ScratchDir::new("fn12-sel");
        assert!(
            !is_flashnext_pack(&d).unwrap(),
            "no manifest: not Flash-Next"
        );
        std::fs::write(
            d.join("manifest.json"),
            r#"{"magic":"SBRPACK1","tp_size":2}"#,
        )
        .unwrap();
        assert!(
            !is_flashnext_pack(&d).unwrap(),
            "a 27B manifest has no model key"
        );
        std::fs::write(
            d.join("manifest.json"),
            r#"{"pack":"x","model":"qwen4_exp"}"#,
        )
        .unwrap();
        assert!(is_flashnext_pack(&d).unwrap());
        std::fs::write(d.join("manifest.json"), r#"{"model":"qwen3_next"}"#).unwrap();
        assert!(!is_flashnext_pack(&d).unwrap());
        assert!(
            !is_flashnext_pack(&d.join("manifest.json")).unwrap(),
            "a file path is a 27B rank pack"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
