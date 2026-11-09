//! Build the constraint registry from the shipped artifacts, instead of transcribing it by hand.
//!
//! [`crate::reference::reference_registry`] is a transcription: someone read the production
//! launcher and typed out what it does. That is fine as a fixture and fatal as a source of truth —
//! it is the same failure mode as the 137 `SABER_*` variables, "two independent readers deriving
//! behaviour from the same name", one of which drifts.
//!
//! This module reads the two artifacts that already exist and cannot drift from the binary:
//!
//! * `prebuilt/gfx1201/*.manifest.toml` — the code objects' own metadata, validated against the
//!   `.hsaco` at load ([`saber_kernels::Manifest::validate_against_hsaco`]). It supplies **wave
//!   size** (a translation-unit property) and **LDS bytes**, which the registry must not guess.
//! * `crates/saber-kernels/rules/*.toml` — the launch rules, already data. They supply the
//!   **constraints**: the geometry bands each plugin covers, the workspace kinds and their size
//!   laws, and the launch count.
//!
//! # One row per plugin, not per instantiation
//!
//! `mxfp4_gemm_decode.hsaco` holds ~1,400 kernel symbols — the `bk × ks × tm × wperm × nt` cross
//! product. Emitting one [`Implementation`] per symbol would reproduce the macro explosion the
//! kernel plane exists to prevent:
//!
//! > Keep the reference's split exactly — **constraints are NECESSARY conditions that discriminate *ops*,
//! > never tiles**. Tile choice is an opaque `variant_id` + parameter blob from the autotune cache.
//! >
//! > — ARCHITECTURE.md, *Kernel plane*
//!
//! So a plugin is an implementation, and the instantiation is the variant. The constraint rows
//! below are the *necessary conditions* [`saber_kernels::rules::resolve`] itself enforces; which
//! symbol inside the plugin runs is `resolve`'s answer at launch time, not the planner's.
//!
//! # What is deliberately not here
//!
//! No measurement. Every row lands at [`PromotionState::Implemented`] — it exists, it loads, it
//! produces output. Only a real autotune record says it is fast, and only that record can bind it.

use std::path::{Path, PathBuf};

use saber_kernels::dflash::DflashRules;
use saber_kernels::glue::GlueRules;
use saber_kernels::head::HeadRules;
use saber_kernels::noop::NoopRules;
use saber_kernels::paro::ParoRules;
use saber_kernels::rules::{LaunchRules, WorkspaceKind as RuleWsKind};
use saber_kernels::saber::{
    plugin_for, ConstraintOp as SaberConstraintOp, RegistryRow, SaberRules,
};
use saber_kernels::Manifest as KernelManifest;

use crate::error::{PlanError, Result};
use crate::manifest::Manifest as PlanManifest;
use crate::provenance::Promotion;
use crate::registry::{
    Constraint, Implementation, LaunchModel, Registry, WorkspaceKind, WorkspaceReq, WorkspaceSizing,
};

/// One plugin's artifacts on disk: its manifest, and where the manifest came from.
#[derive(Clone, Debug)]
pub struct PluginArtifact {
    /// The manifest as committed next to the blob.
    pub manifest: KernelManifest,
    /// The `*.manifest.toml` this was read from.
    pub path: PathBuf,
}

impl PluginArtifact {
    /// Kernel-set stem, e.g. `mxfp4_gemm_decode`.
    #[must_use]
    pub fn plugin(&self) -> &str {
        &self.manifest.plugin
    }

    /// The single wave size of the translation unit, or an error if the manifest disagrees with
    /// itself. `-mwavefrontsize64` is TU-wide and the per-function attribute is silently ignored,
    /// so "the plugin's wave size" is a real, single number or the artifact is malformed.
    pub fn wave_size(&self) -> Result<u32> {
        let mut it = self.manifest.kernel.iter();
        let first = it
            .next()
            .ok_or_else(|| {
                PlanError::Model(format!("{}: manifest declares no kernels", self.plugin()))
            })?
            .wavefront_size;
        for k in it {
            if k.wavefront_size != first {
                return Err(PlanError::Model(format!(
                    "{}: kernel `{}` is wave{} but the plugin's first kernel is wave{first}; \
                     -mwavefrontsize64 is translation-unit wide",
                    self.plugin(),
                    k.name,
                    k.wavefront_size
                )));
            }
        }
        Ok(first)
    }

    /// The largest static LDS any kernel matching `prefix` declares.
    #[must_use]
    pub fn max_lds(&self, prefix: &str) -> u32 {
        self.manifest
            .kernel
            .iter()
            .filter(|k| k.name.starts_with(prefix))
            .map(|k| k.group_segment_fixed_size)
            .max()
            .unwrap_or(0)
    }

    /// How many instantiations match `prefix` — the size of the variant space behind one row.
    #[must_use]
    pub fn variants(&self, prefix: &str) -> usize {
        self.manifest
            .kernel
            .iter()
            .filter(|k| k.name.starts_with(prefix))
            .count()
    }
}

/// Read every `*.manifest.toml` in a prebuilt directory, sorted by file name.
///
/// The blobs are **not** read here and nothing is validated against them: that is
/// `saber-exec`'s loader, which does it on the HIP thread immediately before
/// `hipModuleLoadData`. This crate touches no code object, by design.
pub fn load_artifacts(dir: &Path) -> Result<Vec<PluginArtifact>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| PlanError::Model(format!("{}: {e}", dir.display())))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".manifest.toml"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(PlanError::Model(format!(
            "{}: no *.manifest.toml — is this the prebuilt directory?",
            dir.display()
        )));
    }
    paths
        .into_iter()
        .map(|path| {
            let manifest = KernelManifest::load(&path)
                .map_err(|e| PlanError::Model(format!("{}: {e}", path.display())))?;
            Ok(PluginArtifact { manifest, path })
        })
        .collect()
}

/// The dtypes every W4A8 route accepts, as selection keys rather than prose.
const W_DTYPES: &str = "mxfp4 nvfp4";
const A_DTYPES: &str = "int8 fp8_e4m3";

/// Every committed rule file, loaded once.
///
/// There is one per plugin family and they are not interchangeable: the MXFP4 GEMM family's bands
/// come out of `rules/mxfp4_gfx1201.toml`, the reference's out of `rules/saber_gfx1201.toml`, Paro's
/// out of `rules/paro_gfx1201.toml`, and the artifact's identity picks between them. Handing the
/// GEMM rules to a saber artifact is how the `saber_*` plugins silently produced no registry rows
/// at all.
#[derive(Debug, Clone)]
pub struct RuleSets {
    /// `rules/mxfp4_gfx1201.toml` — the W4A8 GEMM family and its fused producers.
    pub mxfp4: LaunchRules,
    /// `rules/saber_gfx1201.toml` — attention, the fused GDN update, the gated norm, the all-reduce.
    pub saber: SaberRules,
    /// `rules/paro_gfx1201.toml` — the Paro rotation producers, per-group and per-token.
    ///
    /// The PARO tier is prod's **rollback** unit, not the served model, so these rows appear in the
    /// registry and bind to nothing: `ModelSpec::qwen38_27b_dense` describes the NVFP4 tier and
    /// names no `paro_*` op. A row with no bucket is not a gap — it is a route the engine can
    /// launch as soon as a PARO model spec asks for it, and `saber explain` listing it is how a
    /// reader sees that the plugin is described rather than merely present.
    pub paro: ParoRules,
    /// `rules/noop_gfx1201.toml` — the null plugin, an instrument rather than a serve path.
    ///
    /// Its two rows carry op names (`noop`, `noop_touch`) that no [`crate::ModelSpec`] contains,
    /// so no bucket can ever select them: they are in the registry and in `saber explain`, and
    /// they are not in any autotune key. That is the same hash-neutrality the Paro rows
    /// have, reached the same way — by being unbindable, not by being hidden.
    pub noop: NoopRules,
    /// `rules/dflash_gfx1201.toml` — the DFlash2 drafter's fp8 W8A8 blockscale GEMM.
    ///
    /// Same standing again, and for a third reason: this row IS prod's serve path — the drafter
    /// launches it 21 times per decode step — but it is the SPECULATIVE path, and
    /// [`crate::ModelSpec`] describes the target model. The spec grows a drafter the day the
    /// rest of the drafter (selector, attention, the two draft heads) is code objects too; until
    /// then the row is in the registry and in `saber explain`, bound to nothing.
    pub dflash: DflashRules,
    /// `rules/head_gfx1201.toml` — the embedding lookup and the two head kernels.
    ///
    /// Same standing as the two sets above, and for a different reason: these rows ARE a serve
    /// path (every forward pass runs all three), but [`crate::ModelSpec`] describes the model as
    /// its linear/attention/GDN ops and names no `embedding`, `row_gather` or `argmax` op, so
    /// nothing selects them yet. They are here so that a plugin which ships is a plugin the
    /// registry describes — the invariant `every_registry_row_is_attributed_to_exactly_one_plugin`
    /// exists to enforce — and they bind the day the model spec grows those buckets.
    pub head: HeadRules,
    /// `rules/attn_glue_gfx1201.toml` — the attention layer's glue: `qk_norm_rope`,
    /// `attn_gate_quant` and the paged K/V append.
    ///
    /// Unlike the head's rows these ARE selected: [`crate::ModelSpec`] names all three, 16 of
    /// each per step, in both phases. They were the last ops in a forward pass that prod runs
    /// and Saber could only supply.
    pub glue: GlueRules,
}

impl RuleSets {
    /// The rules the `qwen_prod_nvfp4` unit runs today.
    pub fn prod() -> Result<Self> {
        Ok(RuleSets {
            mxfp4: LaunchRules::prod_mxfp4()
                .map_err(|e| PlanError::Model(format!("mxfp4 launch rules: {e}")))?,
            saber: SaberRules::prod_saber()
                .map_err(|e| PlanError::Model(format!("saber launch rules: {e}")))?,
            paro: ParoRules::prod_paro()
                .map_err(|e| PlanError::Model(format!("paro launch rules: {e}")))?,
            noop: NoopRules::prod_noop()
                .map_err(|e| PlanError::Model(format!("noop launch rules: {e}")))?,
            head: HeadRules::prod_head()
                .map_err(|e| PlanError::Model(format!("head launch rules: {e}")))?,
            glue: GlueRules::prod()
                .map_err(|e| PlanError::Model(format!("attn_glue launch rules: {e}")))?,
            dflash: DflashRules::prod_dflash()
                .map_err(|e| PlanError::Model(format!("dflash launch rules: {e}")))?,
        })
    }
}

/// Build the registry from artifacts plus launch rules.
///
/// Plugins the rules do not describe are skipped with a note rather than guessed at: a row whose
/// constraints were invented is worse than no row, because `select` would bind it.
pub fn registry_from_artifacts(artifacts: &[PluginArtifact], rules: &RuleSets) -> Result<Registry> {
    let mut registry = Registry::new();
    // The policy these rows were derived from, verbatim. A registry row only carries the band
    // edges it reads out of `rules.knobs`; the bands, tiles and splits themselves are not in any
    // row, so without this an edit to them is invisible to the plan input hash and every autotune
    // record goes on binding the variant the old policy chose (see `Registry::policy_digest`).
    // The TABLES, not just their joint digest: a route is keyed on the ones its own candidates
    // could bind (`Registry::policy_digest_for`), and the joint digest is set from them here.
    registry.set_policy_tables(&saber_kernels::committed_rule_tables());
    for art in artifacts {
        for imp in implementations_for(art, rules)? {
            registry.insert(imp);
        }
    }
    if registry.implementations().is_empty() {
        return Err(PlanError::Model(
            "no implementation rows could be derived from these artifacts".to_string(),
        ));
    }
    Ok(registry)
}

/// [`registry_from_artifacts`] over a prebuilt directory, with the production launch rules.
pub fn registry_from_prebuilt(dir: &Path) -> Result<Registry> {
    registry_from_artifacts(&load_artifacts(dir)?, &RuleSets::prod()?)
}

/// Plugins that SHIP, were MEASURED, and contribute no registry row on purpose.
///
/// Not "undescribed" and not "unsupported". Each of these was built, gated bit-exact and timed,
/// and the measurement said no. The code object stays in `prebuilt/` so the number can be
/// re-taken without rebuilding the experiment, and this table is what keeps
/// `every_plugin_the_rules_describe_yields_a_row_and_no_other_plugin_does` a real invariant
/// rather than a test with a hole in it: the exception is NAMED, dated and carries its reason,
/// so a plugin that silently stopped producing rows still fails.
///
/// A row here is a decision, not a convenience. Deleting one is how the plugin comes back.
pub const REJECTED_PLUGINS: &[(&str, &str)] = &[(
    "mxfp4_gemm_decode_silu",
    "measured and rejected 2026-09-19 (workstream F, the fusion lever's second measurement): \
     `silu_mul_quant` folded into `down_proj`'s A staging is BIT-EXACT at every M and a LOSS of \
     8x to 26x per node — net -23.8 us at M=1 rising to -283.6 us at M=40, because the fused \
     GEMM costs far more than the producer it removes. `docs/LEDGER.md` carries the three-chain \
     measurement (producer alone, fused, and the GEMM with no producer at all)",
)];

/// Plugins whose launches are planned OUTSIDE the text model's `ExecutionPlan`, with the planner
/// that owns them. They contribute no route rows here because no text op can bind them -- a row
/// would make them `select` candidates for ops they do not implement -- and they are named rather
/// than skipped so the "every plugin yields a row or says why" invariant still holds.
///
/// `vision` (workstream D1): the image tower, planned per image grid by
/// `saber_kernels::vision::plan_tower` over the pack's 333 `model.visual.*` records and gated by
/// `the internal reference harness`; it runs before the text prefill, not inside a text route.
///
/// `mxfp4_gemm_decode_arnq` (ws-h38): fusion-map row 3, one dispatch for a decode GEMM, the exact
/// reduce after it and the norm after that. It replaces three graph ops at once, so no one op's
/// route row can describe it; `saber_exec::layer`'s block planner binds it when its fold fires.
pub const SEPARATELY_PLANNED: &[(&str, &str)] = &[
    (
        "saber_ar_host_exact",
        "explicit registered-host transport: selected by saber::Ar::host_mapped, never a generic \
         native-P2P autotune candidate because its buffers and bounded protocol belong to the \
         host transport profile",
    ),
    (
        "saber_ar_host_exact_chk",
        "checked registered-host transport: selected only by the host profile's race diagnostic, \
         never a generic serving/autotune candidate",
    ),
    (
        "mxfp4_gemm_decode_host_arnq",
        "registered-host GEMM/reduce/norm fusion: selected explicitly by the block planner for \
         the host transport, not an implementation of a single generic model op",
    ),
    (
        "mxfp4_gemm_decode_host_arnq_chk",
        "checked registered-host GEMM/reduce/norm fusion: selected only by the host profile's \
         race diagnostic, not an implementation of a single generic model op",
    ),
    (
        "flashnext_attn",
        "Flash-Next (FN6): the reference's attention bodies at GQA 12 over bf16 K/V pages -- planned \
         by `saber_kernels::fn6` for the qwen4_exp QSA layers (F2), never a row of the 27B's \
         text plan (a row would let `select` bind a GQA-12 body to a GQA-6 cache)",
    ),
    (
        "flashnext_glue",
        "Flash-Next (FN6): the bf16 KV append + indexer ring, the indexer store, the PLE gate \
         and conv, the MTP norms / add, the bf16 SiluAndMul -- ops no 27B graph has; planned \
         by `saber_kernels::fn6` (F2)",
    ),
    (
        "flashnext_gdn",
        "Flash-Next (FN9b): the GDN chunk scan with exact gates -- named by the qwen4_exp layer \
         graph (F2), never a row of the 27B's text plan (the 27B keeps the certified scan)",
    ),
    (
        "flashnext_gdn_fc",
        "Flash-Next (FN24): the f16-carry GDN decode updates of a width-1-class MTP verify window \
         -- named by the qwen4_exp executor, never a row of the 27B's text plan",
    ),
    (
        "flashnext_moe",
        "Flash-Next (FN6): the grouped MTP-expert GEMMs (bf16 default, W8A8 gated) over FN4's \
         tile tables; planned by `saber_kernels::fn6` / `moe` for the MTP layer (F4)",
    ),
    (
        "flashnext_qsa",
        "Flash-Next (FN19): the QSA selection -- block scores, the exact top-512, the per-CTA \
         union key list -- ops no 27B graph has; planned by `saber_kernels::fn19` for the \
         qwen4_exp QSA layers above 2,048 tokens",
    ),
    (
        "flashnext_draft",
        "Flash-Next (FN22): the MTP drafter's device-side chain -- argmax over a vocabulary half, \
         the exchange payload, the pair's pick -> the next pass's embedding row; planned by \
         `saber_kernels::fn22` for the drafter's passes, never a row of the 27B's text plan",
    ),
    (
        "flashnext_xmiss",
        "Flash-Next (FN53c): the pressure valve's device-side miss flag -- one launch a layer \
         between the route and the experts while the expert tier is incomplete -- named by the \
         qwen4_exp executor under `--fn-mem-valve on`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_ar",
        "Flash-Next (FN114b): the TP = 2 exact reduce with the flag carried in the data words \
         (the one-shot's vadd on the same words, byte-equal to it by the two-rank oracle) -- named \
         by the qwen4_exp reduce under `--fn-ar ll`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_ar_gp",
        "Flash-Next (FN114d): the shipped exact / wht6 one-shot reduces with the workspace slot \
         chosen by the call's global parity (the shipped bodies, byte-equal by the two-rank \
         oracle) -- named by the qwen4_exp reduce under `--fn-ar gp`, never a row of the 27B's \
         text plan",
    ),
    (
        "flashnext_ar_host",
        "Flash-Next TP2 exact BF16 over registered host memory, selected only by --fn-ar host",
    ),
    (
        "flashnext_couple",
        "Flash-Next (LR-N): coupled drafting -- the MTP chain tail drafts a sampled row with the \
         target's own noise at the draft's verify position (the target's tokens unchanged) -- \
         named by the qwen4_exp drafter under `--spec-couple on`, never a row of the 27B's text \
         plan",
    ),
    (
        "flashnext_ahead3",
        "Flash-Next (FN119d): the next verify's input embedding rows gathered on the card under \
         the drafter round -- named by the qwen4_exp serve plane under `--fn-device-step next`, \
         never a row of the 27B's text plan",
    ),
    (
        "flashnext_ahead2",
        "Flash-Next (FN119c): the b > 1 issue-ahead's chain inputs composed per sequence on the \
         card (a copy + word runs) -- named by the qwen4_exp serve plane under \
         `--fn-device-step ahead`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_df120",
        "Flash-Next (FN120, the megakernel program's Phase A): the persistent dataflow primitive \
         (residency sweep, the atomic-ticket runtime, the family-1 QSA pair as dataflow tasks) -- \
         a measurement object launched only by tools/capture/fn120/fn120_probe.py, never a row \
         of the 27B's text plan",
    ),
    (
        "flashnext_df120b",
        "Flash-Next (FN120c, the megakernel program's Phase B): the b1 verify's GDN mixer tail as \
         one persistent dataflow launch (`--fn-dataflow gdn`; the executor names it), never a row \
         of the 27B's text plan",
    ),
    (
        "flashnext_ahead",
        "Flash-Next (FN119 round 2): the issue-ahead's device half -- the sampler + walk twin on \
         both ranks over a P2P window and the drafter round's acc-dependent inputs -- named by \
         the qwen4_exp serve plane under `--fn-device-step ahead`, never a row of the 27B's text \
         plan",
    ),
    (
        "flashnext_dstep",
        "Flash-Next (FN119): the device-resident step's sampler + acceptance twin -- the host \
         sampler's draw and the drawn-row walk on the card, op for op -- named by the qwen4_exp \
         serve plane under `--fn-device-step check`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_pick",
        "Flash-Next (FN57): the step's seam pick on the card -- each logits row's top-64 over the \
         rank's vocabulary half by the bf16 key law (the host seam's order) -- named by the \
         qwen4_exp executor under `--fn-pick device`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_trunk8",
        "Flash-Next (FN61c): the W8 dense trunk's output-stride cells (dflash_gemm's W8A8 body and \
         FN5's one-split bf16 band with an ldc) -- named by the qwen4_exp executor under \
         `--fn-trunk w8`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_trunk8p",
        "Flash-Next (FN61e): the W8 dense trunk's multi-row tiles (decode / verify 17..64 rows, \
         prefill 64-row tiles, the bf16 suffix at 64 rows; each element the FN61c parent's \
         arithmetic) -- named by the qwen4_exp executor under `--fn-trunk w8`, never a row of \
         the 27B's text plan",
    ),
    (
        "flashnext_hc8",
        "Flash-Next (FN108): the HC mix with the mixing matrices at fp8 (`--fn-trunk w8-hc8`; \
         each `_h8` symbol = its flashnext_hc parent on bf16(code x scale)) -- named by the \
         qwen4_exp executor, never a row of the 27B's text plan",
    ),
    (
        "flashnext_hc141",
        "Flash-Next (FN141): the drained-ring twins (`--fn-k141`: the fp8-mixer decode up, the \
         router ring; each = its shipped parent byte for byte) -- named by the qwen4_exp executor, \
         never a row of the 27B's text plan",
    ),
    (
        "flashnext_t8141",
        "Flash-Next (FN141): the out / o W8 GEMM as a register ring (`--fn-k141 t8`; = the shipped \
         fn111 / t8p tiles byte for byte) -- named by the qwen4_exp executor, never a row of the \
         27B's text plan",
    ),
    (
        "flashnext_t8far",
        "Flash-Next (FN164): the out / o W8 GEMM with the mixer-out all-reduce fused into its \
         epilogue (`--fn-mixer-ar-push`; the shipped fn111 / t8141 / t8p bodies + the exact \
         one-shot's vadd, byte-equal by the two-rank oracle) -- named by the qwen4_exp executor, \
         never a row of the 27B's text plan",
    ),
    (
        "moe_route163",
        "Flash-Next (FN163): the b8 route at 32 waves + the b1 front's register-rank top-10 \
         (`--fn-route163`; each symbol = its `moe_route` / `moe_front_e512` parent byte for \
         byte) -- named by the qwen4_exp executor, never a row of the 27B's text plan",
    ),
    (
        "flashnext_hc131",
        "Flash-Next (FN131): the fp8-mixer HC down at the decode forms re-tiled by phase \
         (`--fn-hc-down split`; each symbol = flashnext_hc8's decode down of the same cfg, byte \
         for byte) -- named by the qwen4_exp executor, never a row of the 27B's text plan",
    ),
    (
        "flashnext_hc8p",
        "Flash-Next (FN122b): the fp8-mixer HC down's PREFILL tiles (`--fn-prefill-tiles hc`; \
         each symbol = flashnext_hc8's m2w21t1_h8 down at another row-block / wave / slab \
         geometry, byte for byte) -- named by the qwen4_exp executor, never a row of the 27B's \
         text plan",
    ),
    (
        "flashnext_trunk8r",
        "Flash-Next (FN61g): the drafter's exact rerank scored from the preshuffled W8 head \
         (`--fn-trunk w8-head`; each candidate column the W8 head GEMM's logit) -- named by the \
         qwen4_exp executor, never a row of the 27B's text plan",
    ),
    (
        "flashnext_mtp8",
        "Flash-Next (FN50): the MTP experts' mixed-table grouped GEMMs (Qwen's fp8 routed \
         experts on FN6's W8A8 body, the bf16 shared expert on its bf16 band; a tile's scale \
         pointer picks the launch) -- planned by the qwen4_exp executor, never a 27B row",
    ),
    (
        "flashnext_qsa_attn",
        "Flash-Next (FN19): the reference's attention bodies with the key axis re-pointed through a \
         union key list (GQA 12, bf16 pages) -- planned by `saber_kernels::fn19`, never a row \
         of the 27B's text plan",
    ),
    (
        "p27_tree_gdn",
        "the 27B's tree verify (P27-15): the WY-form GDN tree update + the accepted-path advance \
         -- named by the `--engine-spec-tree ancestor9` arm, never a row of the chain plan",
    ),
    (
        "p27w2_gdn_replay",
        "the 27B's GDN decode update as base + stash + replay (P27-W2-5): swapped in for the \
         planned recurrent update by `saber_engine::gdn_replay::rewrite_plan` under the \
         `--engine-gdn-replay on` arm, never a row of the text plan",
    ),
    (
        "p27w2_gdn_replay_lds",
        "CMP27T: the replay update's LDS-staged twin (the shipped replay's bytes), swapped in by \
         `saber_engine::gdn_replay::rewrite_plan_as` under `--engine-c4-arms replay`, never a row \
         of the text plan",
    ),
    (
        "saber_device_row_ilp",
        "CMP27X: the sampler seam's row kernel with a thread's loads issued ahead (the bytes of \
         `saber_device_row_bf16`), named by the head under `--engine-step-arms devrow`, never a \
         head rules row",
    ),
    (
        "saber_device_row_split",
        "CMP27Y: the sampler seam's row over G workgroups with a per-row sync slab (the bytes of \
         `saber_device_row_bf16`), named by the head under `--engine-step-arms rowsplit`, never a \
         head rules row",
    ),
    (
        "saber_device_row_stage",
        "CMP27AJ: the sampler seam's row in the staged-term form, 1,024 threads a row (the bytes of \
         `saber_device_row_bf16`), named by the head under `--engine-step-arms rowstage`, never a \
         head rules row",
    ),
    (
        "p27w2_gdn_cuf",
        "CMP27AH: the conv update fused into the replay update (the pair's bytes), swapped in by \
         `saber_engine::gdn_cuf::fuse_plan` under `--engine-step-arms cuf`, never a row of the text \
         plan",
    ),
    (
        "saber_argmax_pv",
        "CMP27T: the TP > 1 local argmax that also writes its winner's logit bits (the index is \
         `saber_argmax_bf16`'s), named by the head under `--engine-c4-arms picks`, never a head \
         rules row",
    ),
    (
        "p27_tree_gdn16",
        "the 27B's 16-row tree verify (E4-3, the 15-node best-first tree): the WY GDN tree update + advance \
         at 16 rows -- named by the `--engine-spec-tree best16` arm, never a row of the chain plan",
    ),
    (
        "p27_tree_attn16",
        "the 27B's 16-row tree verify (E4-3, the 15-node best-first tree): the ancestor-word decode at 6 \
         warps -- named by the `--engine-spec-tree best16` arm, never a row of the chain plan",
    ),
    (
        "p27_tree_flush",
        "the 27B's tree verify (P27-W2-4, R2): the accepted conv window moved to the chain's \
         entries after every tree step -- named by the tree arm's flush, never a row of the chain \
         plan",
    ),
    (
        "p27_tree_kv",
        "the 27B's tree verify (P27-15b): the accepted path's K/V rows moved to their depth slots \
         -- named by the `--engine-spec-tree ancestor9` arm, never a row of the chain plan",
    ),
    (
        "p27_tree_conv",
        "the 27B's tree verify (P27-15b): the decode conv update over each row's parent path -- \
         named by the `--engine-spec-tree ancestor9` arm, never a row of the chain plan",
    ),
    (
        "p27_tree_rows",
        "CMP27L: the tree conv with its rows in parallel (bit-exact vs `p27_tree_conv`) -- \
         swapped in by the `--engine-b1-arms conv` arm, never a row of the chain plan",
    ),
    (
        "p27_tree_advl16",
        "The 16-row tree flush batches the GDN state advances across layers.",
    ),
    (
        "p27_tree_advl",
        "CMP27L: the R2 flush's GDN advances for every layer in ONE launch (bit-exact vs \
         `p27_tree_gdn`'s advance) -- named by the `--engine-b1-arms adv` arm, never a row of \
         the chain plan",
    ),
    (
        "p27_tree_attn",
        "the 27B's tree verify (P27-15): the ragged decode attention with per-row ancestor words \
         -- named by the `--engine-spec-tree ancestor9` arm, never a row of the chain plan",
    ),
    (
        "flashnext_attn_kv8",
        "Flash-Next (FN68): FN6's attention bodies over FP8 (e4m3) K/V pages -- named by the \
         qwen4_exp executor under `--fn-kv-dtype fp8`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_qsa_attn_kv8",
        "Flash-Next (FN68): FN19's gather attention over FP8 (e4m3) K/V pages -- named by the \
         qwen4_exp executor under `--fn-kv-dtype fp8`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_qsa128",
        "Flash-Next (FN128): the QSA selection at the decode / verify forms (top-k in registers, \
         the union by rank, the score at one thread a group) -- named by the qwen4_exp executor \
         under `--fn-sel128`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_attnfuse",
        "Flash-Next (ATTN-OPT-3): the QSA front and the combine + gate + quant as one launch each \
         -- named by the qwen4_exp executor under `--fn-fuse f6` / `f7`, never a row of the 27B's \
         text plan",
    ),
    (
        "flashnext_qsa124",
        "Flash-Next (FN124): the fp8-KV QSA gather prefill with the masked sub-tile skip -- named \
         by the qwen4_exp executor under `--fn-prefill-tiles qsa`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_glue_kv8",
        "Flash-Next (FN68): the e4m3 KV append + the indexer ring -- named by the qwen4_exp \
         executor under `--fn-kv-dtype fp8`, never a row of the 27B's text plan",
    ),
    (
        "flashnext_hc",
        "Flash-Next (FN5): the hyper-connection mix / combine kernels, the bf16 split-K band and \
         LDS tile, the bf16 norm twins -- planned by `saber_kernels::flashnext` for the qwen4_exp \
         layers (F2), never a row of the 27B's text plan",
    ),
    (
        "vision",
        "the vision tower: planned per image grid by `saber_kernels::vision::plan_tower`, not a \
         text route (docs/VISION.md)",
    ),
    (
        "mxfp4_gemm_decode_arnq",
        "fusion-map row 3 (ws-h38): the split decode GEMM + exact all-reduce + `_nq` epilogue in \
         one dispatch, bound by the block planner's fold (`Planner::gemm_ar_fusion` -> \
         `saber_kernels::rules::gemm_decode_ar_nq`) over THREE graph ops, never a route row for \
         one op -- a `gemm_w4a8` row would let `select` bind a GEMM that also reduces",
    ),
    (
        "mxfp4_gemm_decode_smq",
        "fusion-map row 4 (ws-h40): the gate_up decode GEMM + `silu_mul_quant` in one dispatch, \
         bound by the block planner's fold (`Planner::smq_fusion` -> \
         `saber_kernels::rules::gemm_decode_smq`) over TWO graph ops, never a route row for one \
         op -- a `gemm_w4a8` row would let `select` bind a GEMM that also quantises its silu",
    ),
    (
        "mxfp4_gemm_folded_mt",
        "G5: the folded prefill tile at mid-M heights, bound by `saber_kernels::rules::resolve` \
         under the `folded_mt` knob (`[[folded_mt]]`, `--engine-mid-chunk-gemm`) in the block \
         planner, never a registry row -- a `gemm_w4a8` row would re-key every autotune record \
         and let `select` bind a tile the committed table leaves off",
    ),
    (
        "dflash_gemm_m64",
        "CMP27K: the drafter gate_up's one-weight-pass 64-row tile (bit-exact vs the m-tiled \
         blockscale GEMM), bound by `saber_kernels::dflash::resolve_blockscale_gateup` under the \
         `gateup_m64` knob (`--engine-drafter-gateup-m64`), never a dflash rules row -- the \
         blockscale GEMM's committed rows stay the rules' own",
    ),
    (
        "dflash_head_m64",
        "CMP27Z: the drafter int2 head's one-weight-pass 64-row twin (bit-exact vs the 32-row \
         `_lx` band), bound by `saber_kernels::dflash::resolve_draft_head_int2` under the head's \
         `m64` knob (`--engine-step-arms dh64`), never a dflash rules row",
    ),
    (
        "saber_head_w8_m64",
        "CMP27F: the fp8 head's one-pass 64-row tile (bit-exact vs the 32-row band), bound by \
         `saber_kernels::dflash::resolve_head_w8_m64` under the `head_m64` knob \
         (`--engine-head-m64`), never a dflash rules row -- the head's committed bands stay the \
         rules' own",
    ),
    (
        "mxfp4_gemm_decode_pf",
        "CMP27E: the b8 decode GEMMs' register-prefetch twins (bit-exact), bound by \
         `saber_kernels::rules::resolve` / `gemm_decode_pair` under the `decode_pf` knob \
         (`--engine-decode-pf`) in the block planner, never a registry row -- a `gemm_w4a8` row \
         would re-key every autotune record and let `select` bind a body the committed table \
         leaves off",
    ),
    (
        "mxfp4_gemm_mid",
        "G6: the A-tiled tile at mid-M heights and the skinny-N gate body, bound by \
         `saber_kernels::rules::resolve` under the `atiled_mid` / `skinny` knobs (`[[atiled_mt]]`, \
         `[[skinny]]`, `--engine-mid-chunk-atiled` / `--engine-skinny-gate`) in the block planner, \
         never a registry row -- a `gemm_w4a8` row would re-key every autotune record and let \
         `select` bind a body the committed table leaves off",
    ),
    (
        "saber_gdn_fnq",
        "fusion-map row 5 (ws-h42): the GDN fused update + `gdn_norm_quant` in one dispatch, \
         bound by the block planner's fold (`Planner::gdn_nq_fusion` -> \
         `saber_kernels::saber::gdn_fused_update_nq`) over TWO graph ops, never a route row for \
         one op -- a `gdn_fused_update` row would let `select` bind an update that also \
         quantises its norm",
    ),
    (
        "saber_ar_exact_chk",
        "T4's RACE-CHECK twins of the exact all-reduce and its `_nq` epilogue (`_chk`: the served \
         kernels with `hip/ar_race_check.h`'s checks inserted), bound only when \
         `--engine-race-check` sets `saber::Ar::race_check`, by `saber::ar_kernel` inside the served \
         resolve -- a registry row would let `select` bind a diagnostic on a serving plan",
    ),
    (
        "saber_ar_wht_split",
        "P27-1d's SPLIT rotated all-reduce (the one-shot's push and reduce halves + the \
         fewer-blocks push, joined by a command-processor wait), derived from the one-shot slice \
         by `saber::ar_wht_split` under `--ar-form split` inside the prefill overlap -- never a \
         registry row, so `select` cannot bind it on a plan that did not ask for it",
    ),
    (
        "saber_ar_wht_chk",
        "T4's RACE-CHECK twins of the rotated all-reduce (`_chk`), bound only under \
         `--engine-race-check` by `saber::ar_kernel` -- never a registry row, for the same reason",
    ),
    (
        "saber_ar_wht_wpar",
        "CMP27S: the WAVE-PARALLEL body of the served wht6 bf16 one-shot (the same bytes, grid \
         and kernargs; one (chunk, iteration) item a wave), bound by `saber::ar_wpar_route` inside \
         the served resolve under `--engine-ar-wpar` -- never a registry row, so the registry's \
         rotated row keeps its symbol and every autotune key",
    ),
    (
        "saber_ar_wht_wpar_chk",
        "CMP27S: the wave-parallel body's RACE-CHECK twin (`_chk`), bound only under \
         `--engine-race-check` with `--engine-ar-wpar` by `saber::ar_kernel`",
    ),
    (
        "p27_dstep",
        "P27-W2-2's DEVICE CANDIDATE SEAM (`--engine-device-step cand`): the 27B drafter's \
         backbone -> tail top-K merge across the two ranks, launched raw by the drafter resident \
         (`saber_kernels::p27d`) between its two graphs -- an exact transport of the host merge, \
         so a registry row would only let `select` bind it",
    ),
    (
        "race_digest",
        "T6's SHADOW DIGEST (`--engine-race-check shadow`): two instrument kernels the engine \
         enqueues after a proposal's last kernel (`saber_kernels::race_digest::resolve_regions` / \
         `resolve_kv`, from the drafter resident's own digest tables) -- a diagnostic that \
         computes nothing a model op does, so a registry row would only let `select` bind it",
    ),
    (
        "moe_grouped",
        "FN4: Flash-Next's grouped MXFP4 W4A8 GEMM (derived from the certified decode body) and the \
         expert-major silu-quant -- the MoE block of a model no 27B spec describes; bound by the \
         Flash-Next layer planner (F2) through `saber_kernels::moe`, never a 27B registry row -- a \
         `gemm_w4a8` row would let `select` bind a GEMM that reads its weights through a tile table",
    ),
    (
        "moe_grouped_tp2",
        "FN80b-1: the TP=2 half-width expert cells (the down half at DBK 64, K 320; the half-width \
         fused back half) -- FN4 / FN26's derived bodies at another slab; named by \
         `saber_kernels::moe::ExpertShape::TpHalf`, never a 27B registry row",
    ),
    (
        "moe_fn148",
        "FN148: the NVFP4 (FP4-Local) expert bodies over the xtier v2 record (forms b / c of the \
         FN4 decode body + the exact-dequant oracle) -- no route yet (phase 3 wires the engine), \
         never a 27B registry row",
    ),
    (
        "moe_fn153",
        "FN153: FN148's NVFP4 form (b) in the grouped prefill tile (`pf_tm*`) -- byte-equal to the \
         fn148_w4a8b ks1 cells; no engine route, never a 27B registry row",
    ),
    (
        "moe_fn151",
        "FN151: FN148's NVFP4 form (b) in the served `_p1` (PFD 1) class -- byte-equal twins of \
         the fn148_w4a8b cells; no engine route, never a 27B registry row",
    ),
    (
        "moe_fn152",
        "FN152: FN151's NVFP4 `_p1` body in the fused back halves (EP whole, TP half, FN137's far) -- \
         named by `saber_kernels::moe::nvfp4_back_symbol` under --fn-experts-fmt nvfp4, never a 27B \
         registry row",
    ),
    (
        "moe_grouped_ss",
        "FN114c: the fused back half with the shared expert split across the EP ranks (the shared \
         tile at FN80b-1's half shape, `--fn-shared-expert split`) -- FN26's derived body; named \
         by the Flash-Next emitter, never a 27B registry row",
    ),
    (
        "moe_grouped_far",
        "FN137: FN80b-1's half-width fused back half with the MoE-out all-reduce fused into its \
         combine slices (`--fn-moe-ar-push on`; the shipped body + the exact one-shot's vadd, \
         byte-equal by the two-rank oracle) -- named by the Flash-Next emitter, never a 27B \
         registry row",
    ),
    (
        "moe_route",
        "FN4: Flash-Next's router top-10 + expert-major sort, the weighted combine and the A16 \
         control arm -- ops no 27B graph has; bound by the Flash-Next layer planner (F2) through \
         `saber_kernels::moe`",
    ),
    (
        "mxfp4_gemm_decode_arw",
        "CMP27AE: the N 5120 dwn4 decode GEMM + the wht6 two-rank all-reduce in one dispatch, \
         bound by the block planner's fold (`Planner::gemm_wht_fusion` -> \
         `saber_kernels::rules::gemm_decode_arw`) over TWO graph ops under `--engine-ar-fuse`, \
         never a route row for one op -- a `gemm_w4a8` row would let `select` bind a GEMM that \
         also reduces",
    ),
    (
        "mxfp4_gemm_decode_arnq_chk",
        "T4's RACE-CHECK twin of fusion-map row 3, bound by `rules::gemm_decode_ar_nq` only when \
         the `_nq` it folds is the checked one (`--engine-race-check`) -- row 3's own reason, and \
         a diagnostic besides",
    ),
];

/// Why a plugin contributes no route rows because another planner owns it.
#[must_use]
pub fn separately_planned(plugin: &str) -> Option<&'static str> {
    SEPARATELY_PLANNED
        .iter()
        .find(|(p, _)| *p == plugin)
        .map(|(_, why)| *why)
}

/// Why a plugin contributes no rows, when it is one of the measured-and-rejected ones.
///
/// # Errors
/// None — a plugin that is not in the table simply answers `None`, which is the normal case.
#[must_use]
pub fn rejected_reason(plugin: &str) -> Option<&'static str> {
    REJECTED_PLUGINS
        .iter()
        .find(|(p, _)| *p == plugin)
        .map(|(_, why)| *why)
}

/// Every implementation row one plugin contributes.
///
/// The `match` is a dispatch on the artifact's *identity*, not a measured ladder: each arm states
/// which op the plugin implements and reads its band edges out of `rules.knobs`. Nothing here is a
/// number that was typed twice.
pub fn implementations_for(art: &PluginArtifact, rules: &RuleSets) -> Result<Vec<Implementation>> {
    // Measured, rejected, and deliberately unbound. Returning an empty row set here rather than
    // falling through to the MXFP4 arm is the point: the fused instantiation would otherwise be
    // described as a `gemm_w4a8` candidate and `select` could bind an experiment the ledger
    // says loses by 8x to 26x.
    if rejected_reason(art.plugin()).is_some() || separately_planned(art.plugin()).is_some() {
        return Ok(Vec::new());
    }
    if art.plugin().starts_with("saber_") {
        return saber_implementations_for(art, &rules.saber);
    }
    // Paro is matched through its rules' OWN symbol map, not through the stem: the rules say
    // which code object their 36 symbols live in, and this asks them. A prefix test would be a
    // second, drifting opinion about the same fact.
    let paro_owner = rules
        .paro
        .owner()
        .map_err(|e| PlanError::Model(format!("paro launch rules: {e}")))?;
    if art.plugin() == paro_owner {
        return paro_implementations_for(art, &rules.paro);
    }
    if art.plugin() == rules.noop.owner() {
        return noop_implementations_for(art, &rules.noop);
    }
    if art.plugin() == rules.head.owner() {
        return head_implementations_for(art, &rules.head);
    }
    // Asked of the glue rules' OWN symbol map, like Paro above and for the same reason: the
    // rules say which code object their symbols live in, and a prefix test on `attn_` would be a
    // second, drifting opinion about the same fact.
    let glue_owner = rules
        .glue
        .owner()
        .map_err(|e| PlanError::Model(format!("attn_glue launch rules: {e}")))?;
    if art.plugin() == glue_owner {
        return glue_implementations_for(art, &rules.glue);
    }
    if rules.dflash.owns(art.plugin()) {
        return dflash_implementations_for(art, &rules.dflash);
    }
    let rules = &rules.mxfp4;
    let kn = &rules.knobs;
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let gemm_dtypes = |i: Implementation| {
        i.constrain(Constraint::is_in("w_dtype", W_DTYPES))
            .constrain(Constraint::is_in("a_dtype", A_DTYPES))
    };

    let rows = match plugin.as_str() {
        "mxfp4_gemm_decode" => {
            let partial = &rules.workspace.decode_partial;
            let counter = &rules.workspace.decode_counter;
            vec![gemm_dtypes(
                Implementation::new("gemm_w4a8_mxfp4_decode", &plugin, "gemm_w4a8")
                    .constrain(Constraint::is_in("a_layout", "row_major"))
                    .describe(
                        "C[M,N] = A[M,K] @ dequant(W)[N,K]^T with the split-K reduce FUSED into \
                         the same kernel",
                        &format!(
                            "1 <= M <= {}, N % 16, K % {} ({} instantiations: bk x ks x tm x \
                             wperm x nt). N is NOT bounded: `dec_max_n` ({}) is the width of the \
                             SPLIT-K scratch, and a shape wider than it takes no split",
                            kn.decode_max_m,
                            kn.k_multiple,
                            art.variants("mxfp4_gemm_decode"),
                            kn.dec_max_n,
                        ),
                        "w=mxfp4 a=int8/fp8_e4m3 c=bf16",
                    )
                    .constrain(Constraint::ge("m", 1))
                    .constrain(Constraint::le("m", i64::from(kn.decode_max_m)))
                    // No `n <= dec_max_n`. That number is the split-K scratch's width (the partial
                    // slab and the per-n-block counter), which is a constraint on the SPLIT arm --
                    // and on the split arm it is vacuous, provably: `dec_max_n / 128 = 288`
                    // n-blocks already clears the largest fill target in `split_k_fill` (110), so
                    // any N above the cap answers `ks = 1`, where the kernel writes C directly and
                    // this plan binds both scratch slots null. Keeping the cap here cost the one
                    // shape wider than it -- `lm_head` -- the decode tile at a decode M, and with
                    // it 2.2 ms of a 33.8 ms step (docs/LEDGER.md, 2026-09-19). The kernel-plane
                    // half of the same statement is in `rules.rs` at the band test; the two planes
                    // must agree or the planner offers a candidate the launcher will not produce.
                    .constrain(Constraint::div("n", 16))
                    .constrain(Constraint::div("k", i64::from(kn.k_multiple)))
                    .capture(true, false)
                    .code_object(wave, art.max_lds("mxfp4_gemm_decode"))
                    .workspace(WorkspaceReq::new(
                        "decode_partial",
                        ws_kind(partial.kind),
                        WorkspaceSizing::SplitKPartials { elem_bytes: 4 },
                    ))
                    .workspace(WorkspaceReq::new(
                        "decode_counter",
                        ws_kind(counter.kind),
                        WorkspaceSizing::Fixed(counter.capacity_bytes),
                    ))
                    // The split-K reduce is fused, so a split is still ONE launch. This is the
                    // single place that fact is recorded for the launch budget.
                    .launches(LaunchModel::Fixed(1))
                    .promoted(Promotion::implemented()),
            )]
        }
        "mxfp4_gemm_folded" => vec![gemm_dtypes(
            Implementation::new("gemm_w4a8_mxfp4_folded", &plugin, "gemm_w4a8")
                .constrain(Constraint::is_in("a_layout", "row_major"))
                .describe(
                    "the folded prefill tile: per-row reference exponent folded into the epilogue",
                    &format!(
                        "any M (the general path), N % 16, K % {} ({} instantiations: tn x wperm \
                         x epifast)",
                        kn.k_multiple,
                        art.variants("mxfp4_gemm_folded")
                    ),
                    "w=mxfp4 a=int8/fp8_e4m3 c=bf16",
                )
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::div("n", 16))
                .constrain(Constraint::div("k", i64::from(kn.k_multiple)))
                .capture(true, false)
                .code_object(wave, art.max_lds("mxfp4_gemm_folded"))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        )],
        "mxfp4_gemm_atiled" => vec![gemm_dtypes(
            Implementation::new("gemm_w4a8_mxfp4_atiled", &plugin, "gemm_w4a8")
                .describe(
                    "prefill GEMM with activations already in fragment-tiled layout, A straight to \
                     WMMA",
                    &format!(
                        "M >= {}, N % 16, K % {} ({} instantiations: tn x lbk x wperm)",
                        kn.a_tiled_min_m,
                        kn.atiled_k_multiple,
                        art.variants("mxfp4_gemm_atiled")
                    ),
                    "w=mxfp4 a=int8/fp8_e4m3 c=bf16",
                )
                // The same threshold the PRODUCER switches layout at. It is one number in one
                // file precisely because in prod it lives in two and must agree by convention.
                .constrain(Constraint::ge("m", i64::from(kn.a_tiled_min_m)))
                // This kernel reads A ALREADY in WMMA fragment order, so the M threshold is a
                // necessary condition and not a sufficient one: an edge written by a producer
                // that is not fused is row-major at every M, and reading it here is garbage
                // rather than a regression (IMPLEMENTATION-PLAN.md P0 follow-up (a)).
                .constrain(Constraint::is_in("a_layout", "a_tiled"))
                .constrain(Constraint::div("n", 16))
                .constrain(Constraint::div("k", i64::from(kn.atiled_k_multiple)))
                .capture(true, false)
                .code_object(wave, art.max_lds("mxfp4_gemm_atiled"))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        )],
        "mxfp4_gemm_base" => vec![gemm_dtypes(
            Implementation::new("gemm_w4a8_mxfp4_unfolded", &plugin, "gemm_w4a8")
                .constrain(Constraint::is_in("a_layout", "row_major"))
                .describe(
                    "the unfolded path: no per-row reference exponent, per-32-block rescale in the \
                     inner loop, checkpoint weight layout",
                    &format!("any M, K % {}, and a checkpoint with NO row reference", kn.k_multiple),
                    "w=mxfp4 (no wref) a=int8/fp8_e4m3 c=bf16",
                )
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::div("k", i64::from(kn.k_multiple)))
                // A necessary condition the dense NVFP4 tier does not meet, and says so: the model
                // spec never sets `w_ref`, so this row is refused with "w_ref not given" rather
                // than quietly never being chosen.
                .constrain(Constraint::is_in("w_ref", "absent"))
                .capture(true, false)
                .code_object(wave, art.max_lds("mxfp4_gemm_base"))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        )],
        "mxfp4_quant" => vec![
            Implementation::new("add_rms_quant_fused", &plugin, "add_rms_quant")
                .describe(
                    "residual add + Gemma RMSNorm + per-token e4m3 quant, one workgroup per row",
                    &format!("K % 8, K <= {}", kn.quant_max_k),
                    "in=bf16 out=e4m3 + fp32 row scale",
                )
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::div("k", 8))
                .constrain(Constraint::le("k", i64::from(kn.quant_max_k)))
                .capture(true, false)
                .code_object(wave, art.max_lds("add_rms_quant"))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
            Implementation::new("silu_mul_quant_fused", &plugin, "silu_mul_quant")
                .describe(
                    "silu(gate) * up + per-token e4m3 quant",
                    &format!("N % 8, N <= {}", kn.silu_max_n),
                    "in=bf16 out=e4m3 + fp32 row scale",
                )
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::div("n", 8))
                .constrain(Constraint::le("n", i64::from(kn.silu_max_n)))
                .capture(true, false)
                .code_object(wave, art.max_lds("silu_mul_quant"))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
            Implementation::new("gdn_norm_quant_fused", &plugin, "gdn_norm_quant")
                .describe(
                    "per-head gated RMSNorm + per-token e4m3 quant",
                    &format!("N % {}, N <= {}", kn.gdn_n_multiple, kn.gdn_max_n),
                    "in=bf16 out=e4m3 + fp32 row scale",
                )
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::div("n", i64::from(kn.gdn_n_multiple)))
                .constrain(Constraint::le("n", i64::from(kn.gdn_max_n)))
                .capture(true, false)
                .code_object(wave, art.max_lds("gdn_norm_quant"))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        ],
        other => {
            // Not a fallback route: nothing is bound, nothing degrades. The artifact is simply not
            // described by these rules yet, and saying so beats inventing constraints for it.
            tracing_note(other);
            Vec::new()
        }
    };
    Ok(rows)
}

// ---------------------------------------------------------------------------- Paro

/// One band of one producer family: the rows are `(family, band)` pairs because the two bands
/// compute different things, not because one is faster.
struct ParoBand {
    /// Registry row name.
    name: &'static str,
    /// Op the row implements — the same string `paro::resolve` puts in its descriptors, so the
    /// registry and the launch path name the route identically.
    op: &'static str,
    computes: &'static str,
    /// The symbols this band can reach, given the tier's knobs.
    symbols: Vec<String>,
    /// The necessary conditions `paro::resolve` itself enforces for this band.
    constraints: Vec<Constraint>,
    /// Human form of the same conditions.
    shape: String,
    dtypes: &'static str,
}

/// The distinct `mode` values a table carries, as a selection set (`0 1 2` today).
///
/// Read off the rules rather than typed: a fourth mode would appear here the day it appears in the
/// table, and `resolve` refuses a mode with no instantiation, which is what makes this a necessary
/// condition rather than documentation.
fn paro_mode_set(modes: &mut Vec<u32>) -> String {
    modes.sort_unstable();
    modes.dedup();
    modes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Every implementation row the Paro producer plugin contributes.
///
/// Four families × two M bands. The bands are the launcher's own two decisions, and both are read
/// out of `knobs` rather than typed here:
///
/// * per-GROUP (`add_rms_rot` / `ew_rot`): `ROT` is `M <= decode_max_m` — at and below it the
///   producer rotates and per-group quantises, above it it writes `hs` only;
/// * per-TOKEN (`add_rms_rot_tok` / `ew_rot_tok`): `TILED` is `M >= a_tiled_min_m`, **the same
///   number the MXFP4 A-tiled row constrains on**, because the producer writes the fragment layout
///   that GEMM reads.
///
/// `i8` is a tier knob, not a shape: with `PQ_I8` off (the paromx unit), the `_i81` instantiations
/// are unreachable and are not counted as variants of any row. Workspaces: none — every buffer
/// these producers touch is the layer's own tensor, which the rules file states and this asserts by
/// declaring no workspace at all.
/// The null plugin's rows: one per null op, straight off the rules table.
///
/// These rows exist to be *described*, not selected. Their op names appear in no model spec, so
/// the constraint match never returns them and they stay outside every autotune key
/// (`Registry::bindable_fingerprint`); `saber chain-bench` reaches the kernels through
/// `saber_kernels::noop::resolve`, not through a plan. They carry no constraints because there is
/// no shape to constrain: the geometry is one workgroup of one thread, fixed by the rules file
/// and checked when it loads.
fn noop_implementations_for(
    art: &PluginArtifact,
    rules: &NoopRules,
) -> Result<Vec<Implementation>> {
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let mut rows = Vec::with_capacity(rules.kernel.len());
    for row in &rules.kernel {
        let computes = if row.op == "noop" {
            "nothing at all: no arguments, no memory traffic — the cheapest dispatch this \
             hardware can be asked to run"
        } else {
            "one byte, read-modify-written — the smallest RAW dependency a chain can have"
        };
        rows.push(
            Implementation::new(&format!("bench_{}", row.op), &plugin, &row.op)
                .describe(
                    computes,
                    "grid 1 x block 1 (no shape: widening it would measure occupancy, not \
                     dispatch)",
                    "none",
                )
                .capture(true, false)
                .code_object(wave, art.max_lds(&row.symbol))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        );
    }
    Ok(rows)
}

/// The head plugin's rows: one per head op, straight off the rules table.
///
/// The constraints are the ones `saber_kernels::head::resolve` itself enforces, and no others:
/// a positive row count no larger than the rules' `max_rows`, and a positive width. There is no
/// tile ladder to state because there is no tile — the geometry is one workgroup per output row,
/// and the workgroup's width is a knob these three kernels are COMPILED for (the argmax reduces
/// in LDS over exactly `block` entries), which is why it is checked when the rules load rather
/// than constrained here.
/// The attention glue's rows: one per op, with the constraints `glue::resolve` itself enforces.
///
/// `head_dim` and `rotary_dim` are COMPILED INTO the symbols — the rope pair count and the LDS
/// staging are sized by them — so they are equality constraints rather than bands: a shape off
/// them is a refusal, not a slower path. Everything else the rules carry (the reduction order,
/// the A-tiled threshold, which of the two mrope layouts runs) is a variant choice inside
/// `resolve`, and stating it here would be the tile ladder the kernel plane exists to prevent.
fn glue_implementations_for(
    art: &PluginArtifact,
    rules: &GlueRules,
) -> Result<Vec<Implementation>> {
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let m = &rules.model;
    let lds = |symbol: &str| art.max_lds(symbol);
    // THIS implementation's band rows: the ones compiled for the model's own head width and
    // flavour. The rules file may hold a second ladder (the drafter's 128-wide plain-rope rows),
    // and a row of that ladder must not set this row's LDS bound or its described band edge --
    // it is a different implementation, and it gets its own registry row when it exists.
    let qk_rows: Vec<&saber_kernels::glue::QkBand> = rules
        .qk_norm_rope
        .band
        .iter()
        .filter(|b| {
            b.head_dim == m.head_dim && b.flavour() == saber_kernels::glue::QkFlavour::TARGET
        })
        .collect();
    Ok(vec![
        Implementation::new("qk_norm_rope", &plugin, "qk_norm_rope")
            .describe(
                "per-head q/k RMS norm and the partial mrope, ONE launch over prod's five (two \
                 ATen norms, the cos_sin_cache[positions] gather, two chunks, the Triton mrope)",
                &format!(
                    "head_dim == {}, rotary_dim == {}, 1 <= m; TWO decompositions, banded on \
                     the (token, head) pair count — one {}-thread workgroup per pair up to {} \
                     pairs, one WAVE per pair above it",
                    m.head_dim,
                    m.rotary_dim,
                    rules.qk_norm_rope.threads,
                    qk_rows.first().and_then(|b| b.pairs_max).unwrap_or(0),
                ),
                "bf16 projection in, bf16 q/k out, fp32 cos/sin cache, i32 mrope positions",
            )
            .constrain(Constraint::eq("head_dim", i64::from(m.head_dim)))
            .constrain(Constraint::eq("rotary_dim", i64::from(m.rotary_dim)))
            .constrain(Constraint::ge("m", 1))
            .capture(true, false)
            // The LDS of the widest band: the per-head decomposition stages its reduction
            // through LDS and the wave one needs none, so the bound is the first row's.
            .code_object(
                wave,
                qk_rows
                    .iter()
                    .map(|b| lds(&b.interleaved).max(lds(&b.chunked)))
                    .max()
                    .unwrap_or(0),
            )
            .launches(LaunchModel::Fixed(rules.qk_norm_rope.n_launches))
            .promoted(Promotion::implemented()),
        Implementation::new("attn_gate_quant", &plugin, "attn_gate_quant")
            .describe(
                "core * sigmoid(gate) with the per-token e4m3 quant o_proj reads, with the gate \
                 taken IN PLACE out of the projection's [q|gate] block",
                &format!(
                    "head_dim == {}, 1 <= m; the A side is row-major below {} rows and \
                     fragment-tiled at or above it, the same edge add_rms_quant takes",
                    m.head_dim,
                    rules.attn_gate_quant.a_tiled_min_m()
                ),
                "bf16 core and gate in, fp8_e4m3 + per-token fp32 scale out",
            )
            .constrain(Constraint::eq("head_dim", i64::from(m.head_dim)))
            .constrain(Constraint::ge("m", 1))
            .capture(true, false)
            .code_object(
                wave,
                lds(&rules.attn_gate_quant.tiled0).max(lds(&rules.attn_gate_quant.tiled1)),
            )
            .launches(LaunchModel::Fixed(rules.attn_gate_quant.n_launches))
            .promoted(Promotion::implemented()),
        Implementation::new("kv_append", &plugin, "kv_append")
            .describe(
                "the paged fp8 K/V write, BOTH tensors in one launch as \
                 triton_reshape_and_cache_flash does: the V base is the K base plus head_size \
                 elements of the same paged tensor",
                &format!(
                    "head_dim == {}, block_size == {}, 1 <= m; {} threads per (token, head), \
                     each writing one packed e4m3 pair of K and one of V",
                    m.head_dim, m.block_size, rules.kv_append.threads
                ),
                "bf16 key/value in, fp8_e4m3 paged cache out",
            )
            .constrain(Constraint::eq("head_dim", i64::from(m.head_dim)))
            .constrain(Constraint::eq("block_size", i64::from(m.block_size)))
            .constrain(Constraint::is_in("kv_dtype", "fp8_e4m3"))
            // `m` and not `tokens`: the row count is `m` everywhere else in this registry, and
            // a second name for it would be a key the model spec has to remember to spell
            // differently for one op. `glue::KvShape` calls the same number `tokens`.
            .constrain(Constraint::ge("m", 1))
            .capture(true, false)
            .code_object(wave, lds(&rules.kv_append.kernel))
            .launches(LaunchModel::Fixed(rules.kv_append.n_launches))
            .promoted(Promotion::implemented()),
    ])
}

fn head_implementations_for(
    art: &PluginArtifact,
    rules: &HeadRules,
) -> Result<Vec<Implementation>> {
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let max_rows = i64::from(rules.knobs.max_rows);
    let mut rows = Vec::with_capacity(rules.kernel.len());
    for row in &rules.kernel {
        let (computes, dtypes) = match row.op.as_str() {
            "embedding" => (
                "out[t] = table[ids[t] - vocab_begin], or a ZERO row when this rank does not own \
                 the id — the vocab-parallel contract the all-reduce after it sums",
                "bf16 table, i32 ids, bf16 out",
            ),
            "row_gather" => (
                "out[r] = src[idx[r]]: the row selection before `lm_head`, which is what makes a \
                 2048-token prefill's head ONE [1, hidden] GEMM instead of 2048",
                "bf16 in, i32 index, bf16 out",
            ),
            "argmax" => (
                "the index of the largest logit in each row, ties to the LOWEST index (torch's \
                 rule); an all-NaN row yields `cols`, which is out of range on purpose",
                "bf16 logits, i32 token id",
            ),
            "device_row" => (
                "THE SAMPLER'S SEAM: the K largest logits and their ids in the argmax's own total \
                 order (value descending, id ascending), the row's log-sum-exp at T = 1 and at its \
                 own temperature, and the raw logits at the ids the penalties can move. A radix \
                 select on the bf16 bit pattern, so the threshold is an exact value and the answer \
                 does not depend on how the work was split; a row whose temperature is not \
                 positive is SKIPPED, which is what keeps a greedy step the step it was",
                "bf16 logits, f32 temperatures, i32 penalty ids, f32 values, i32 ids, f32 \
                 log-sum-exps, f32 penalty logits",
            ),
            "row_copy" => (
                "THE DFLASH2 AUX TAP: `rows x cols` bf16 copied into a COLUMN BLOCK of a wider \
                 destination, source and destination row strides independent. It is not \
                 `row_gather` with a stride — that one packs SELECTED rows together, this one \
                 copies every row into a slice of a `rows x 5 x hidden` concatenation, which is \
                 what lets the five target-layer taps land in one buffer with no sixth launch",
                "bf16 in, bf16 out",
            ),
            "widen_i64" => (
                "an i32 table sign-extended to i64, rows `src_stride` / `dst_stride` apart: the \
                 one launch of the drafter's captured proposal prod does not make, bridging the \
                 i32 `dflash_topk_bf16_k16` writes to the `const long *` the selector's score and \
                 walk read",
                "i32 in, i64 out",
            ),
            "add_rms_bf16" => (
                "THE bf16 HEAD'S FINAL NORM (N2): the residual add and Gemma RMS norm of \
                 `add_rms_quant` with the bf16 row as the output, for a `lm_head` that is a bf16 \
                 GEMM (`Graph::with_bf16_head`); cols is the hidden size, at most 4 x 8 x the block",
                "bf16 hidden and residual in, bf16 normalized row and bf16 residual out",
            ),
            other => {
                return Err(PlanError::Model(format!(
                    "the head rules name op `{other}`, which this planner cannot describe"
                )))
            }
        };
        rows.push(
            Implementation::new(&format!("head_{}", row.op), &plugin, &row.op)
                .describe(
                    computes,
                    &format!(
                        "1 <= rows <= {max_rows}, cols >= 1 (one workgroup of {} per output row)",
                        rules.knobs.block
                    ),
                    dtypes,
                )
                .constrain(Constraint::ge("rows", 1))
                .constrain(Constraint::le("rows", max_rows))
                .constrain(Constraint::ge("cols", 1))
                .capture(true, false)
                .code_object(wave, art.max_lds(&row.symbol))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        );
    }
    Ok(rows)
}

fn paro_implementations_for(
    art: &PluginArtifact,
    rules: &ParoRules,
) -> Result<Vec<Implementation>> {
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let kn = &rules.knobs;
    let group = i64::from(kn.group);
    let krot_max = i64::from(kn.krot_max);
    let rot_max_m = i64::from(kn.decode_max_m);
    // The ONE band edge, read rather than restated (`saber_kernels::rules::a_tiled_min_m`).
    let tiled_min_m = i64::from(saber_kernels::rules::a_tiled_min_m());
    // The per-token producers stage the whole rotated row in dynamic LDS: `width * 2` bytes, which
    // bounds the width they can take at all. bf16, hence the 2.
    let max_tok_width = i64::try_from(kn.lds_max_bytes / 2).unwrap_or(i64::MAX);

    let sym =
        |it: &mut dyn Iterator<Item = &String>| -> Vec<String> { it.map(String::clone).collect() };

    let mut bands: Vec<ParoBand> = Vec::new();

    // ---- per-GROUP add_rms_rot ---------------------------------------------------------------
    for rot in [true, false] {
        let i8 = kn.i8 && rot;
        let symbols = sym(&mut rules
            .add_rms_rot
            .iter()
            .filter(|r| r.rot == rot && r.i8 == i8)
            .map(|r| &r.symbol));
        let mut constraints = vec![
            Constraint::div("k", group),
            Constraint::ge("krot", 1),
            Constraint::le("krot", krot_max),
            Constraint::ge("p", 1),
        ];
        let shape = if rot {
            constraints.insert(0, Constraint::ge("m", 1));
            constraints.insert(1, Constraint::le("m", rot_max_m));
            format!("1 <= M <= {rot_max_m}, K % {group}, 1 <= krot <= {krot_max}, P >= 1")
        } else {
            constraints.insert(0, Constraint::ge("m", rot_max_m + 1));
            format!(
                "M >= {}, K % {group}, 1 <= krot <= {krot_max}, P >= 1",
                rot_max_m + 1
            )
        };
        bands.push(ParoBand {
            name: if rot {
                "paro_add_rms_rot_fused"
            } else {
                "paro_add_rms_rot_plain"
            },
            op: "paro_add_rms_rot",
            computes: if rot {
                "residual add + RMSNorm with the rotation and the per-GROUP fp8 quant FUSED into \
                 the producer (the launcher's `fused` arm)"
            } else {
                "residual add + RMSNorm writing the rotated `hs` only; above the decode band the \
                 prefill path quantises separately"
            },
            symbols,
            constraints,
            shape,
            dtypes: "in=bf16 res=bf16 -> hs=bf16 (+ e4m3 activations and per-group scales on the \
                     fused arm; int8 when PQ_I8 is on, which this tier does not set)",
        });
    }

    // ---- per-GROUP ew_rot --------------------------------------------------------------------
    let mut ew_modes: Vec<u32> = rules.ew_rot.iter().map(|r| r.mode).collect();
    let ew_modes = paro_mode_set(&mut ew_modes);
    for rot in [true, false] {
        let i8 = kn.i8 && rot;
        let symbols = sym(&mut rules
            .ew_rot
            .iter()
            .filter(|r| r.rot == rot && r.i8 == i8)
            .map(|r| &r.symbol));
        let mut constraints = vec![
            Constraint::div("n", group),
            Constraint::ge("krot", 1),
            Constraint::le("krot", krot_max),
            Constraint::is_in("mode", &ew_modes),
        ];
        let shape = if rot {
            constraints.insert(0, Constraint::ge("m", 1));
            constraints.insert(1, Constraint::le("m", rot_max_m));
            format!("1 <= M <= {rot_max_m}, N % {group}, 1 <= krot <= {krot_max}, mode in {{{ew_modes}}}")
        } else {
            constraints.insert(0, Constraint::ge("m", rot_max_m + 1));
            format!(
                "M >= {}, N % {group}, 1 <= krot <= {krot_max}, mode in {{{ew_modes}}}",
                rot_max_m + 1
            )
        };
        bands.push(ParoBand {
            name: if rot {
                "paro_ew_rot_fused"
            } else {
                "paro_ew_rot_plain"
            },
            op: "paro_ew_rot",
            computes: if rot {
                "elementwise producer -- mode 0 silu-mul, 1 attention gate, 2 GDN gated RMSNorm -- \
                 with rotation and per-GROUP quant fused in"
            } else {
                "the same elementwise producer writing rotated `hs` only, one workgroup per \
                 group-block (the reference forces gpw = 1 on this arm)"
            },
            symbols,
            constraints,
            shape,
            dtypes: "in=bf16 (+ gate/z operand) -> hs=bf16 (+ e4m3 and per-group scales on the \
                     fused arm)",
        });
    }

    // ---- per-TOKEN add_rms_rot_tok ------------------------------------------------------------
    for tiled in [false, true] {
        let symbols = sym(&mut rules
            .add_rms_rot_tok
            .iter()
            .filter(|r| r.tiled == tiled)
            .map(|r| &r.symbol));
        let mut constraints = vec![
            Constraint::div("k", group),
            Constraint::le("k", max_tok_width),
            Constraint::ge("krot", 1),
            Constraint::le("krot", krot_max),
            Constraint::ge("p", 1),
        ];
        let shape = if tiled {
            constraints.insert(0, Constraint::ge("m", tiled_min_m));
            format!(
                "M >= {tiled_min_m}, K % {group}, K <= {max_tok_width}, 1 <= krot <= {krot_max}, \
                 P >= 1"
            )
        } else {
            constraints.insert(0, Constraint::ge("m", 1));
            constraints.insert(1, Constraint::le("m", tiled_min_m - 1));
            format!(
                "1 <= M <= {}, K % {group}, K <= {max_tok_width}, 1 <= krot <= {krot_max}, P >= 1",
                tiled_min_m - 1
            )
        };
        bands.push(ParoBand {
            name: if tiled {
                "paro_add_rms_rot_tok_tiled"
            } else {
                "paro_add_rms_rot_tok_rowwise"
            },
            op: "paro_add_rms_rot_tok",
            computes: if tiled {
                "per-TOKEN residual add + RMSNorm writing the FRAGMENT layout the A-tiled GEMM \
                 reads -- the same M threshold that GEMM's row constrains on"
            } else {
                "per-TOKEN residual add + RMSNorm, row-major output; the whole rotated row is \
                 staged in DYNAMIC LDS, which is why K is bounded"
            },
            symbols,
            constraints,
            shape,
            dtypes: "in=bf16 res=bf16 -> hs=bf16 + e4m3 activations with ONE scale per token",
        });
    }

    // ---- per-TOKEN ew_rot_tok -----------------------------------------------------------------
    let mut tok_modes: Vec<u32> = rules.ew_rot_tok.iter().map(|r| r.mode).collect();
    let tok_modes = paro_mode_set(&mut tok_modes);
    for tiled in [false, true] {
        let symbols = sym(&mut rules
            .ew_rot_tok
            .iter()
            .filter(|r| r.tiled == tiled)
            .map(|r| &r.symbol));
        let mut constraints = vec![
            Constraint::div("n", group),
            Constraint::le("n", max_tok_width),
            Constraint::ge("krot", 1),
            Constraint::le("krot", krot_max),
            Constraint::is_in("mode", &tok_modes),
        ];
        let shape = if tiled {
            constraints.insert(0, Constraint::ge("m", tiled_min_m));
            format!(
                "M >= {tiled_min_m}, N % {group}, N <= {max_tok_width}, 1 <= krot <= {krot_max}, \
                 mode in {{{tok_modes}}}"
            )
        } else {
            constraints.insert(0, Constraint::ge("m", 1));
            constraints.insert(1, Constraint::le("m", tiled_min_m - 1));
            format!(
                "1 <= M <= {}, N % {group}, N <= {max_tok_width}, 1 <= krot <= {krot_max}, \
                 mode in {{{tok_modes}}}",
                tiled_min_m - 1
            )
        };
        bands.push(ParoBand {
            name: if tiled {
                "paro_ew_rot_tok_tiled"
            } else {
                "paro_ew_rot_tok_rowwise"
            },
            op: "paro_ew_rot_tok",
            computes: if tiled {
                "per-TOKEN elementwise producer in the A-tiled fragment layout"
            } else {
                "per-TOKEN elementwise producer, row-major output, row staged in dynamic LDS"
            },
            symbols,
            constraints,
            shape,
            dtypes: "in=bf16 (+ gate/z operand) -> hs=bf16 + e4m3 with ONE scale per token",
        });
    }

    let mut rows = Vec::with_capacity(bands.len());
    for band in bands {
        if band.symbols.is_empty() {
            return Err(PlanError::Model(format!(
                "{plugin}: band `{}` has no reachable instantiation in the rules",
                band.name
            )));
        }
        // Attribution, per symbol, through the rules' own map — the same discipline the saber
        // rows use. A band whose symbols claimed two plugins would be a route split across code
        // objects, which nothing can launch.
        for symbol in &band.symbols {
            let owner = rules
                .plugin_for(symbol)
                .map_err(|e| PlanError::Model(format!("{plugin}: band `{}`: {e}", band.name)))?;
            if owner != plugin {
                return Err(PlanError::Model(format!(
                    "{plugin}: band `{}` names `{symbol}`, which the rules place in `{owner}`",
                    band.name
                )));
            }
        }
        // ...and the blob has to carry them. A rules table naming a symbol no code object holds is
        // the failure the saber rows already met once: the row silently vanishes.
        let mine: Vec<&String> = band
            .symbols
            .iter()
            .filter(|sym| art.manifest.kernel(sym).is_some())
            .collect();
        if mine.is_empty() {
            return Err(PlanError::Model(format!(
                "{plugin}: band `{}` names {} symbol(s), none of which this code object carries",
                band.name,
                band.symbols.len()
            )));
        }
        let lds = mine
            .iter()
            .filter_map(|sym| art.manifest.kernel(sym))
            .map(|k| k.group_segment_fixed_size)
            .max()
            .unwrap_or(0);
        let mut imp = Implementation::new(band.name, &plugin, band.op)
            .describe(
                band.computes,
                &format!("{} ({} instantiations)", band.shape, mine.len()),
                band.dtypes,
            )
            // Plain kernels on the engine's stream: graph-safe, and nothing here syncs.
            .capture(true, false)
            .code_object(wave, lds)
            // One dispatch. There is no second pass and no reduce to fuse: the producer writes
            // every output it has in one launch.
            .launches(LaunchModel::Fixed(1))
            .promoted(Promotion::implemented());
        for c in band.constraints {
            imp = imp.constrain(c);
        }
        rows.push(imp);
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------- saber

/// What one saber op needs beyond its `[[registry]]` row: the workspaces it takes, how many
/// dispatches one call is, and a sentence a reader can act on.
///
/// A table, not a ladder. The launch counts in particular are the numbers `CLAUDE.md` rule 7 is
/// about: attention decode is **2** because the split-KV combine is a second dispatch over the
/// partials the first one wrote, and everything else is 1.
struct SaberOpFacts {
    /// `(workspace name, rules-file key)`. The two differ for the all-reduce: a rank allocates its
    /// own scratch and flags AND maps the peer's, so two distinct workspaces are sized by one rule.
    workspaces: &'static [(&'static str, &'static str)],
    launches: u32,
    computes: &'static str,
}

/// The all-reduce workspaces, exactly as `saber::ar_workspaces` builds them.
///
/// Five for the exact family; the quantised one adds `ar_loc_pack`, this rank's own packed half,
/// because the reduce must fold exactly the bytes this rank sent and they cannot be recomputed.
const AR_WS_EXACT: &[(&str, &str)] = &[
    ("ar_scratch", "ar_scratch"),
    ("ar_peer_scratch", "ar_scratch"),
    ("ar_flags", "ar_flags"),
    ("ar_peer_flags", "ar_flags"),
    ("ar_seq", "ar_seq"),
];
const AR_WS_QUANT: &[(&str, &str)] = &[
    ("ar_scratch", "ar_scratch"),
    ("ar_peer_scratch", "ar_scratch"),
    ("ar_flags", "ar_flags"),
    ("ar_peer_flags", "ar_flags"),
    ("ar_seq", "ar_seq"),
    ("ar_loc_pack", "ar_loc_pack"),
];

/// Is this all-reduce row the exact family? `exact` is the predicate the reference's own table uses to
/// separate them, and the entry name follows it.
fn is_exact_ar(row: &RegistryRow) -> bool {
    row.entry.starts_with("ar_oneshot_2rank_exact")
}

/// What one saber row needs beyond its predicates. Keyed on the ROW, not on the op alone: the two
/// all-reduce families share op `allreduce` and do not share a workspace set.
fn saber_op_facts(row: &RegistryRow) -> Option<SaberOpFacts> {
    Some(match row.op.as_str() {
        "attn_prefill_paged" => SaberOpFacts {
            workspaces: &[],
            launches: 1,
            computes: "paged causal attention, query-tiled; one launch per run of requests with \
                       equal query length",
        },
        "attn_decode_paged" | "attn_decode_paged_ragged" => SaberOpFacts {
            workspaces: &[("attn_decode_scratch", "attn_decode_scratch")],
            launches: 2,
            computes: "paged causal attention, split-KV; the split-KV combine is a SECOND \
                       dispatch, because a token's partials are spread across `splits` workgroups \
                       of the first",
        },
        "gdn_fused_update" => SaberOpFacts {
            workspaces: &[("gdn_fused_barrier", "gdn_fused_barrier")],
            launches: 1,
            computes: "the GDN decode step -- conv update, grid-wide barrier, recurrent \
                       delta-rule update -- in ONE launch",
        },
        "gdn_conv_update" => SaberOpFacts {
            workspaces: &[],
            // The PAIR. Above `FUSED_MAX_ITEMS` (batch >= 2 at TP=2, where H = 24) the decode step
            // is the conv update followed by the recurrent update: two dispatches, because a
            // recurrent head consumes q/k written by DIFFERENT conv units, so the handoff crosses
            // workgroups and cannot be block-fused.
            launches: 2,
            computes: "the GDN decode conv update; the recurrent update is its SECOND dispatch",
        },
        "gdn_recurrent_update" => SaberOpFacts {
            workspaces: &[],
            launches: 1,
            computes: "the recurrent delta-rule update -- the second half of the decode pair, \
                       counted by `gdn_conv_update`",
        },
        "gdn_conv_prep" => SaberOpFacts {
            workspaces: &[],
            launches: 1,
            computes: "prefill causal depthwise conv over the whole chunk",
        },
        "gdn_kkt_solve" => SaberOpFacts {
            workspaces: &[],
            launches: 1,
            computes: "prefill: the chunked KKT / UT-transform solve",
        },
        "gdn_chunk_scan" => SaberOpFacts {
            workspaces: &[],
            launches: 1,
            computes: "prefill: the chunked delta-rule scan",
        },
        "gdn_gated_rmsnorm" => SaberOpFacts {
            workspaces: &[],
            launches: 1,
            computes: "per-head gated RMS norm over the linear-attention output",
        },
        // Two families under one op, separated by the `exact` predicate exactly as the reference
        // separates them. Prod runs BOTH every step: the exact kernel below the 128 KiB
        // quantisation threshold and the WHT-rotated one above it.
        "allreduce" if is_exact_ar(row) => SaberOpFacts {
            workspaces: AR_WS_EXACT,
            launches: 1,
            computes: "one-shot 2-rank push all-reduce over P2P, fp32 accumulate, no quantisation",
        },
        "allreduce" => SaberOpFacts {
            workspaces: AR_WS_QUANT,
            launches: 1,
            computes: "one-shot 2-rank push all-reduce over P2P, Walsh-Hadamard rotated and \
                       quantised to BITS bits per symbol with a bf16 scale per group of 64",
        },
        // The reference's own registry carries no row for `_nq`; this one is Saber's, and it is its own
        // op because it computes something different — the all-reduce with the post-reduce
        // epilogue fused into it.
        "allreduce_nq" => SaberOpFacts {
            workspaces: AR_WS_EXACT,
            launches: 1,
            computes: "the exact 2-rank all-reduce with the post-reduce epilogue fused in",
        },
        _ => return None,
    })
}

/// The symbols of the op's FIRST dispatch — the ones that decide which plugin owns the row.
///
/// Attribution has to use the head alone. `gdn_conv_update` dispatches the recurrent update second,
/// and that kernel lives in a *different* code object; attributing on the full set would put one
/// row in two plugins under one name.
fn head_symbols_for(row: &RegistryRow, rules: &SaberRules) -> Vec<String> {
    match row.op.as_str() {
        "attn_decode_paged" | "attn_decode_paged_ragged" => rules
            .attn_decode
            .iter()
            .filter(|r| r.entry == row.entry)
            .map(|r| r.symbol.clone())
            .collect(),
        "gdn_conv_update" => rules
            .gdn_conv_update
            .iter()
            .map(|r| r.symbol.clone())
            .collect(),
        _ => symbols_for(row, rules),
    }
}

/// Every symbol the instantiation tables map this registry row's entry to, across ALL of the op's
/// dispatches.
///
/// The `[[registry]]` row says a kernel EXISTS for a geometry; the instantiation tables say which
/// symbol. This set is what the row's LDS figure is taken over, so that a two-dispatch op is priced
/// at the worse of its two kernels — but only over the symbols the artifact in hand actually
/// carries, which is what keeps a cross-plugin second dispatch out of the wrong row.
fn symbols_for(row: &RegistryRow, rules: &SaberRules) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // C1: a `_f16state` entry owns the instantiation rows that store the GDN state as f16, and
    // every other entry owns the fp32 ones -- two registry rows, two plugins, never one row split.
    let f16 = row.entry.ends_with("_f16state");
    match row.op.as_str() {
        "attn_prefill_paged" => out.extend(
            rules
                .attn_prefill
                .iter()
                .filter(|r| r.entry == row.entry)
                .map(|r| r.symbol.clone()),
        ),
        "attn_decode_paged" | "attn_decode_paged_ragged" => {
            out.extend(
                rules
                    .attn_decode
                    .iter()
                    .filter(|r| r.entry == row.entry)
                    .map(|r| r.symbol.clone()),
            );
            // The combine is the decode op's second launch, not an op of its own, so its symbols
            // belong to this row: its LDS and its wave size are part of what this route costs.
            // C2: a `_ragged` entry owns the ragged combine (its own plugin, `saber_attn_ragged`),
            // every other entry the uniform one -- one plugin per row, as C1's state rows.
            let ragged = row.entry.ends_with("_ragged");
            out.extend(
                rules
                    .attn_combine
                    .iter()
                    .filter(|r| r.ragged == ragged)
                    .map(|r| r.symbol.clone()),
            );
        }
        "gdn_fused_update" => out.extend(
            rules
                .gdn_fused_update
                .iter()
                .filter(|r| r.state_f16 == f16)
                .map(|r| r.symbol.clone()),
        ),
        "gdn_conv_prep" => out.extend(rules.gdn_conv_prep.iter().map(|r| r.symbol.clone())),
        "gdn_conv_update" => {
            out.extend(rules.gdn_conv_update.iter().map(|r| r.symbol.clone()));
            // The recurrent update is this op's second dispatch, so its symbols belong to this
            // row: what the pair costs in LDS is the worst of the two. The fp32 ones: the f16
            // stored-state update (C1) lives in its own plugin under its own registry row.
            out.extend(
                rules
                    .gdn_recurrent_update
                    .iter()
                    .filter(|r| !r.state_f16)
                    .map(|r| r.symbol.clone()),
            );
        }
        "gdn_recurrent_update" => out.extend(
            rules
                .gdn_recurrent_update
                .iter()
                .filter(|r| r.state_f16 == f16)
                .map(|r| r.symbol.clone()),
        ),
        "gdn_kkt_solve" => out.extend(rules.gdn_kkt_solve.iter().map(|r| r.symbol.clone())),
        "gdn_chunk_scan" => out.extend(
            rules
                .gdn_chunk_scan
                .iter()
                .filter(|r| r.state_f16 == f16)
                .map(|r| r.symbol.clone()),
        ),
        "gdn_gated_rmsnorm" => out.extend(rules.gdn_norm.iter().map(|r| r.symbol.clone())),
        "allreduce" if is_exact_ar(row) => {
            out.extend(rules.ar_exact.iter().map(|r| r.symbol.clone()))
        }
        "allreduce" => out.extend(
            rules
                .ar_wht
                .iter()
                .filter(|r| r.entry == row.entry)
                .map(|r| r.symbol.clone()),
        ),
        "allreduce_nq" => out.extend(rules.ar_exact_nq.iter().map(|r| r.symbol.clone())),
        _ => {}
    }
    out
}

fn plugin_row_constraint(c: &saber_kernels::saber::Constraint) -> Constraint {
    // The reference's five predicates and the planner's are the same five, with the same semantics and
    // the same refusal wording; only the spelling of one operator differs.
    match c.op {
        SaberConstraintOp::Eq => Constraint::eq(&c.key, c.ival),
        SaberConstraintOp::Le => Constraint::le(&c.key, c.ival),
        SaberConstraintOp::Ge => Constraint::ge(&c.key, c.ival),
        SaberConstraintOp::DivisibleBy => Constraint::div(&c.key, c.ival),
        SaberConstraintOp::In => Constraint::is_in(&c.key, &c.sval),
    }
}

/// The one plugin that owns a registry row, via the authoritative symbol map.
///
/// The drafter plugin's rows: one per drafter op, straight off the rules table.
///
/// The constraints are the ones `saber_kernels::dflash::resolve` itself enforces and no others:
/// the tile's 16 rows, a whole number of 128-wide workgroups down N, and a whole number of
/// 128-wide scale groups down K. The tile itself is not constrained here because it is not a
/// choice — the rules loader refuses a file that names a tile the code object was not compiled
/// for, which is a stronger statement than a band.
fn dflash_implementations_for(
    art: &PluginArtifact,
    rules: &DflashRules,
) -> Result<Vec<Implementation>> {
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let kn = &rules.knobs;
    let hd = &rules.head;
    let bg = &rules.bgemm;
    let sl = &rules.selector;
    let cv = &rules.conv;
    let at = &rules.attn;
    let kv = &rules.kv;
    let ml = &rules.mlp;
    let cx = &rules.ctx;
    let mut rows = Vec::new();
    for row in rules.rows_for(&plugin) {
        let (computes, envelope, dtypes) = match row.op.as_str() {
            "gemm_a8w8_blockscale" => (
                "c[m][n] = sum over k tiles of dot_fp32(a_tile, b_tile) * (a_scale[m] * b_scale), \
                 the fp8 W8A8 GEMM the DFlash2 drafter runs 21 times per decode step. B is \
                 PRESHUFFLED to [N/16, K*16] — the drafter's checkpoint bytes — and the 256-byte \
                 block that layout produces IS a w32 WMMA fragment, so the weight stream goes from \
                 global straight into the operand register",
                format!(
                    "1 <= M <= {}, N a multiple of {} and <= {}, K a multiple of {} (one \
                     workgroup of {} per {} of N)",
                    kn.bm, kn.bn, kn.max_n, kn.group, kn.block, kn.bn
                ),
                "e4m3 a, e4m3 preshuffled w, fp32 128x128 block scales, bf16 out",
            ),
            "gemm_a8w8_blockscale_mt" => (
                "the same fp8 W8A8 GEMM M-TILED: `blockIdx.y` picks which 16 rows a workgroup \
                 owns and everything inside it is the tile above -- the same k tiles in the same \
                 order, the same scale applied to the same tile dot product, the same single \
                 round to bf16 -- so every output row's arithmetic is unchanged. It exists \
                 because a BATCHED proposal runs 21 of these launches at `reqs * block` rows and \
                 the untiled symbol stops at 16",
                format!(
                    "{} < M <= {}, N a multiple of {} and <= {}, K a multiple of {} (a grid of \
                     ceil(M / {}) on the y axis, one workgroup of {} per {} of N)",
                    kn.bm, kn.max_m, kn.bn, kn.max_n, kn.group, kn.bm, kn.block, kn.bn
                ),
                "e4m3 a, e4m3 preshuffled w, fp32 128x128 block scales, bf16 out",
            ),
            "draft_head_int2" => (
                "the drafter's logits against a 2-bit vocabulary-parallel head — 0.167 GiB/rank \
                 against the bf16 head's 1.18 — plus the top KCAND of each 64-column block, which \
                 is where the rerank's candidates come from. Asymmetric int2 at one scale per \
                 (row, group of 128), quarter-split packed so one byte load serves four contiguous \
                 dots, and dequantised as a BIT PATTERN (`0x3F80 | ((b << (5-2q)) & 0x60)` reads \
                 as 1 + v/4) whose 1.0 bias divides out through the group's sum of x",
                format!(
                    "M == {} exactly (prod pads the draft rows to it and the kernel writes all of \
                     them), N a multiple of {}, K a multiple of {} (one workgroup of {} per {} of \
                     N, {} candidates emitted per block)",
                    hd.bm,
                    hd.bn,
                    4 * hd.group,
                    hd.block,
                    hd.bn,
                    hd.kcand
                ),
                "bf16 x, int2 packed w, bf16 premultiplied scale/zero, fp32 group sums, bf16 out",
            ),
            "draft_head_int2_mt" => (
                "the same int2 head M-TILED: `blockIdx.y` picks which 16 rows a workgroup owns \
                 and everything inside is the tile above -- the same ten groups of each of the \
                 same four quarters in the same order, the same two accumulator ops per group, \
                 the same emit over the same 64 columns. Prod carries M = 32 and M = 64 as \
                 different instantiations, but that is `_cfg_for`'s num_warps and not arithmetic: \
                 per output element the reduction does not depend on BLOCK_M at all. It exists \
                 because a BATCHED proposal's head scores `_pow2_at_least(reqs * (block - 1))` \
                 sampled rows and the untiled symbol takes exactly 16",
                format!(
                    "{} < M <= {} and a whole number of {}-row tiles (the emit writes all {} rows \
                     of its own and prod allocates y/bmax/bidx at the padded height), N a \
                     multiple of {}, K a multiple of {}",
                    hd.bm,
                    hd.max_m,
                    hd.bm,
                    hd.bm,
                    hd.bn,
                    4 * hd.group
                ),
                "bf16 x, int2 packed w, bf16 premultiplied scale/zero, fp32 group sums, bf16 out",
            ),
            "draft_head_int2_m32" => (
                "the same int2 head over a 32-ROW BAND: one workgroup holds a [32, 64] accumulator \
                 and reads each of the head's 159 MiB of int2 weight bytes ONCE for thirty-two \
                 rows, where the row slice above is two workgroups reading it twice. Per output \
                 element the reduction is unchanged -- the same ten groups of each of the same \
                 four quarters in the same order, the same two accumulator ops per group -- so it \
                 is bit-identical to the 16-row symbol per 16-row slice; what moves is the weight \
                 stream, which is what the launch costs (prod's own BLOCK_M = 32 band is 592.64 us \
                 against 868.54 for the row slice and 743.20 for two 16-row launches)",
                format!(
                    "{} < M <= {} and a whole number of {}-row bands (an M of whole {}-row tiles \
                     and not whole bands is `draft_head_int2_mt`), N a multiple of {}, K a \
                     multiple of {}",
                    hd.bm,
                    hd.max_m,
                    hd.bm2,
                    hd.bm,
                    hd.bn,
                    4 * hd.group
                ),
                "bf16 x, int2 packed w, bf16 premultiplied scale/zero, fp32 group sums, bf16 out",
            ),
            "bgemm_bf16" => (
                "c[m][n] = sum over k of a[m][k] * w[n][k] in bf16 with an fp32 accumulator -- the \
                 drafter's two UNQUANTISED Linear layers, `kernel_projection` (twice per layer) \
                 and the selector's `hidden_projection`, which its checkpoint lists in \
                 modules_to_not_convert. One wave per 16x16 output tile walking all of K, which is \
                 the reference's own parallelism and the only one that can be bit-exact",
                format!(
                    "1 <= M <= {}, N a multiple of {} and <= {}, K a multiple of {}",
                    bg.m, bg.n, bg.max_n, bg.k_multiple
                ),
                "bf16 a, bf16 w [N, K] row-major, bf16 out",
            ),
            "bgemm_w8_head" | "bgemm_w8_head_m32" | "bgemm_w8_head_wide" => (
                "THE TARGET'S fp8-WEIGHT lm_head (N4, `Graph::with_fp8_head`): the checkpoint's \
                 per-channel e4m3 matrix and its f32 scale expanded in register to \
                 bf16(f32(w8) * scale) -- exactly the bf16 head's weight bytes -- then the bf16 \
                 bands' WMMA sequence unchanged (the 16-row tile, the 32-row band above it, or \
                 the one-pass wide band above that, H33), so \
                 its output is the bf16 head's element for element at half the weight bytes",
                format!(
                    "1 <= M <= {} (the band above {}), N a multiple of {} and <= {} (the head's \
                     own envelope), K a multiple of {}",
                    bg.max_m, bg.m, bg.n, bg.head_max_n, bg.k_multiple
                ),
                "bf16 a, e4m3 w [N, K] row-major + f32 per-row scale, bf16 out",
            ),
            "bgemm_bf16_mt" => (
                "the same projection M-TILED: `blockIdx.y` picks which 16 rows a workgroup owns \
                 and everything inside it is the tile above, so every output row's arithmetic is \
                 unchanged. It exists because a BATCHED proposal's context store runs at \
                 `reqs * block` rows and the untiled symbol stops at 16",
                format!(
                    "{} < M <= {}, N a multiple of {} and <= {}, K a multiple of {}",
                    bg.m, bg.max_m, bg.n, bg.max_n, bg.k_multiple
                ),
                "bf16 a, bf16 w [N, K] row-major, bf16 out",
            ),
            "bgemm_bf16_m32" => (
                "the same projection over a 32-ROW BAND: two WMMA m-fragments per wave against ONE \
                 weight fragment, so the weight stream is read once for both. Every output \
                 element's k order is the tile's, so the arithmetic is unchanged; what moves is \
                 bandwidth, and this launch is weight-bound -- N * K * 2 bytes, 104.9 MB at the \
                 context store's N = 10240, a 165 us floor at 635 GB/s where a row slice at M = 32 \
                 reads 209.7 MB",
                format!(
                    "{} < M <= {} (a grid of ceil(M / {}) on the y axis), N a multiple of {} and \
                     <= {}, K a multiple of {}",
                    bg.m, bg.max_m, bg.m2, bg.n, bg.max_n, bg.k_multiple
                ),
                "bf16 a, bf16 w [N, K] row-major, bf16 out",
            ),
            "selector_score" => (
                "out[p][c] = bf16(einsum result) + unary[c], where the einsum contracts \
                 predecessor_codebook[pred_id[p]] * hidden over the rank against \
                 successor_codebook[cand[c]] -- the whole of `_score_edges`, gathers included, in \
                 one wave per block position. The codebooks are indexed rather than gathered, so \
                 the two [vocab, rank] tables are touched for 32 rows per position",
                format!(
                    "rank == {} and top_k == {} exactly (the rank is the kernel's WMMA k-step \
                     count and the top-k is its fragment); positions a whole number of blocks",
                    sl.rank, sl.top_k
                ),
                "bf16 codebooks, i64 candidates, i32 anchor, f32 unary, bf16 hidden, f32 out",
            ),
            "grouped_conv" | "grouped_conv_quant" => (
                "out[t][h] = sum over taps of (base[tap][h] + delta[t][tap][h/group_size]) * \
                 x[t-tap][h], the tap > 0 terms gated by (t & (block_size-1)) >= tap so the \
                 convolution restarts each block -- and, in the `_quant` variant, the per-128-group \
                 e4m3 quantisation of that output in the SAME pass, which is the fusion inductor \
                 could not finish because it computes the scale in one kernel and applies it in \
                 another",
                format!(
                    "channels a whole number of {}-wide conv groups (and of {}-wide quant groups \
                     for the quantised variant), block_size a power of two, side in 0..=1",
                    cv.group_size, cv.quant_group
                ),
                "bf16 x/coefficients/base, bf16 out or e4m3 out with fp32 per-group scales",
            ),
            "attn_bf16kv" | "attn_fp8kv" => (
                "the drafter's attention: a running softmax over a SYMMETRIC 2048 sliding window \
                 (the drafter is non-causal, so the window bounds keys on both sides of the query) \
                 against the reference server's FLASH KV layout -- separate [blocks, block_size, kv_heads, \
                 head_size] caches. One wave per (q block, kv head), because every cross-wave split \
                 of the loop would re-associate either the head reduction or the running softmax. \
                 ONE SEQUENCE: the program id IS the local q-block index and the length is \
                 `seq_lens[0]`",
                format!(
                    "head_dim == {}, GQA == {}, window {}, KV block size {} (one workgroup of {} \
                     per q block of {})",
                    at.head, at.gqa, at.window, at.block_size, at.block, at.bq
                ),
                "bf16 q, bf16 or e4m3 KV caches with an fp32 per-tensor scale, bf16 out",
            ),
            "attn_bf16kv_ns" | "attn_fp8kv_ns" => (
                "the same attention over a BATCH of sequences in one launch (ws-s11): the walk \
                 `seq = qblock * BLOCK_Q / q_per_seq` in place of the reference's binary search \
                 over `cu_seqlens_q`, which is exact rather than approximate because a proposal's \
                 batch is RECTANGULAR -- every request contributes the drafter's own block of \
                 rows. Each sequence reads its own `seq_lens[i]` and its own row of the block \
                 table, so a batched proposal is one launch where it used to be one per request",
                format!(
                    "head_dim == {}, GQA == {}, window {}, KV block size {}; rows a whole number \
                     of sequences and each sequence a whole number of {}-row q blocks",
                    at.head, at.gqa, at.window, at.block_size, at.bq
                ),
                "bf16 q, bf16 or e4m3 KV caches with an fp32 per-tensor scale, bf16 out",
            ),
            "kv_append_bf16" | "kv_append_fp8" => (
                "`reshape_and_cache_flash` at the drafter's head_dim: both caches written in one \
                 launch, indexed by a slot mapping that is a BLOCK TABLE WALK rather than the token \
                 index, with negative slots skipped as the reference server's padding marker",
                format!("head_dim == {}, one workgroup of {} per token row", at.head, kv.block),
                "bf16 key/value in, bf16 or e4m3 caches out",
            ),
            "silu_quant" => (
                "silu(gate) * up followed by the per-128-group e4m3 quant its consumer needs -- \
                 the drafter's MLP activation, which the target's per-TOKEN silu_mul_quant cannot \
                 stand in for because a blockscale GEMM reads group scales",
                format!("channels a whole number of {}-wide groups", ml.quant_group),
                "bf16 gate_up in, e4m3 out with fp32 per-group scales",
            ),
            "conv_add_rms_open" | "conv_add_rms_close" => (
                "the convolution's `finish` half, the residual add and the RMS norm in ONE kernel, \
                 as inductor already fuses them: the census's `triton_red_fused_add_arange_*` \
                 carries the conv's ops AND the reduction that is this norm's variance. One \
                 workgroup owns a row, because nothing smaller owns the variance. The conv's \
                 output is NOT rounded to bf16 on the way into the residual and the norm rounds \
                 ONCE, at the store: inductor keeps a fused region in fp32, and the two roundings \
                 the Python and the C++ kernel both have cost 47.5% of `chain.l0.v`'s bytes",
                format!(
                    "channels a whole number of {}-wide conv groups, block_size a power of two",
                    cv.group_size
                ),
                if row.op == "conv_add_rms_open" {
                    "bf16 x, coefficients, base, RESIDUAL and weight in; bf16 normed output and an \
                     FP32 residual out -- prod's compiled region opens the layer's residual as \
                     fp32 here"
                } else {
                    "bf16 x, coefficients, base and weight in with an FP32 RESIDUAL; bf16 normed \
                     output and a bf16 residual out -- the region ends here and the next \
                     attention reads bf16"
                },
            ),
            "rms_norm" => (
                "the RMS norm on its own -- `ir.ops.rms_norm`, which is `fused_add_rms_norm` \
                 without the add and without the residual store. The drafter runs it exactly \
                 once: layer 0 is the only layer whose residual is None, so it is the only norm \
                 with no convolution in front of it to fuse into. Its input is an unmodified bf16 \
                 tensor, so unlike the fused form it re-reads the row instead of holding the fp32 \
                 values between its two passes",
                "channels a whole number of bf16x4 loads".to_string(),
                "bf16 x and weight in, bf16 normed output out",
            ),
            "selector_walk" => (
                "the seven-step walk from the selector's edge scores to the draft block: each step \
                 is an argmax over the top-k candidates of the row the PREVIOUS step's winner \
                 names, so the steps are sequential by construction. One program per request and \
                 one wave, as the reference launches it; at prod's greedy sampling the gumbel path \
                 falls away and the argmax's ties go to the lowest index",
                format!("top_k == {} exactly (one wave's argmax)", sl.top_k),
                "fp32 scores, i32 candidates and request states in; the draft block and the \
                 realized scores out",
            ),
            "selector_topk" => (
                "the selector's `torch.topk(logits, 16)` over this rank's vocabulary shard: one \
                 workgroup per row keeps a per-thread top-k in LDS through a single streaming pass \
                 and then merges the 256 sorted lists. Exact, not approximate -- the per-thread \
                 capacity IS k, which is the condition for exactness, since one thread's slice may \
                 contain every winner. Equal values go to the LOWER index, which is ATen's order at \
                 this shape and NOT at every shape",
                format!(
                    "k == {} exactly (it is the per-thread capacity), n >= k, one workgroup of {} \
                     per row",
                    kv.topk_k, kv.topk_block
                ),
                "bf16 logits in, fp32 values and i32 indices out",
            ),
            "rerank_exact" => (
                "the exact fp32 logit for each of R candidate columns per draft row, straight off \
                 the head's own e4m3 weight decoded by bit arithmetic — a few hundred KB against \
                 the coarse pass's 0.167 GiB, and what makes a 2-bit head match the bf16 argmax",
                format!(
                    "1 <= R <= {}, K a multiple of {} (one workgroup of {} per (row, candidate))",
                    hd.max_r, hd.rerank_block_k, hd.rerank_block
                ),
                "bf16 x, e4m3 w as raw bytes, fp32 per-row scale, i32 candidates, fp32 out",
            ),
            "rerank_exact_bf16" => (
                "the same exact rerank against a bf16 head — `_rerank_exact`'s FP8=False arm, and \
                 the one the DFlash2 serve takes, because `SABER_NVFP4_LMHEAD=bf16` has \
                 already replaced the checkpoint's e4m3 `lm_head` with `bf16(f32(w8) * \
                 f32(scale))` before `ref_drafthead._head_matrix` looks at it. No per-row \
                 scale in the signature at all: the reference passes `x` for that argument and \
                 never loads it",
                format!(
                    "1 <= R <= {}, K a multiple of {} (one workgroup of {} per (row, candidate))",
                    hd.max_r, hd.rerank_block_k, hd.rerank_block
                ),
                "bf16 x, bf16 w, i32 candidates, fp32 out",
            ),
            "rerank_exact_w8" => (
                "the bf16 arm's exact rerank over the CHECKPOINT's own fp8 head (workstream N6): \
                 `head.rerank.w8` + `head.rerank.ws`, the same bytes as the target's `lm_head.w8` \
                 / `lm_head.ws` so one record per rank serves both, expanded in register to \
                 `bf16(f32(w8) * f32(scale))` -- the bf16 record's element bit for bit -- and then \
                 the bf16 arm's dot and reduction unchanged",
                format!(
                    "1 <= R <= {}, K a multiple of {} (one workgroup of {} per (row, candidate))",
                    hd.max_r, hd.rerank_block_k, hd.rerank_block
                ),
                "bf16 x, e4m3 w8 as raw bytes, fp32 per-row scale, i32 candidates, fp32 out",
            ),
            "ctx_rms_norm" => (
                "the RMS norm of the drafter's CONTEXT-KV store, which is the C++ \
                 `upstream::rms_norm_kernel` and NOT `ir.ops.rms_norm`: \
                 `precompute_and_store_context_kv` calls `torch.ops._C` directly and so bypasses \
                 the serve's `custom_ops: [..., \"none\"]` toggle. Two differences carry the whole \
                 gate -- a CUB block reduction (a shfl_DOWN tree at offsets 1, 2, 4, 8, 16 with a \
                 validity guard, not the shfl_XOR ladder) and an epilogue that rounds ONCE, \
                 `bf16(x * rsqrt * weight)` rather than `bf16(x * rsqrt) * weight`. The weight is \
                 [groups, h] and a row takes group `row / rows_per_group`, which is how the five \
                 per-layer K norms are ONE launch",
                format!(
                    "channels a whole number of {}-element vectors, rows a whole number of weight \
                     groups; one workgroup per row of min(h/{}, {}) threads and {} once there are \
                     {} rows -- the launcher's own rule, traced, and arithmetic rather than \
                     occupancy because the narrow block chains vectors into one accumulator",
                    cx.vec, cx.vec, cx.wide_block, cx.narrow_block, cx.narrow_rows
                ),
                "bf16 x and [groups, h] weight in, bf16 out; fp32 accumulation and an fp32 \
                 epsilon",
            ),
            "ctx_rope" => (
                "the context store's ONE batched rope: K only, IN PLACE, over L x num_ctx rows \
                 with the positions repeated per layer and `key = nullptr` so K rides in the \
                 reference's query slot. NEOX pairing -- (i, i + rot_dim/2) sharing cos[i]/sin[i] \
                 -- and rotary_dim == head_dim, a FULL plain rope. The arithmetic is the C++ \
                 kernel's, contracted: `t = sin*y; out_x = fma(x, cos, -t)` and `t = cos*y; \
                 out_y = fma(sin, x, t)`, one rounding per output. This is NOT an `attn_glue` \
                 instantiation: that plugin is compiled -ffp-contract=off for a TRITON reference \
                 that does not contract, and its `qk_norm_rope` norms first and writes q and k to \
                 separate outputs",
                format!(
                    "rot_dim even and at most head_size (one workgroup of min(heads x rot_dim/2, \
                     {}) per row)",
                    cx.rope_max_block
                ),
                "i64 positions, bf16 K rotated in place, bf16 [max_position, rot_dim] cos/sin \
                 cache",
            ),
            "act_quant" | "act_quant_recip" => (
                "the blockscale GEMM's OWN activation quant: bf16 -> e4m3 with per-128-group fp32 \
                 scales. Prod runs it inside the layer's quant method, which is why a launch \
                 census that counts GEMMs never sees it and why the drafter's plan declared it \
                 absent for four rounds -- a staged replay seeds past a missing producer and only \
                 a CHAIN notices. The reference is `QuantFP8.forward_native`, not the reference's \
                 per-group quant: the reference server's inductor-piecewise compilation leaves the custom ops off, \
                 so both the eager site (`model.fc`) and the compiled one (`o_proj`) take the \
                 native path. TWO PROGRAMS AT ONE SOURCE LINE, which is why there are two ops: \
                 inductor strength-reduces `absmax / 448` into `absmax * (1/448)` INSIDE the \
                 region (`act_quant_recip`, the five `o_proj` sites) and ATen's `div` by a python \
                 float does not (`act_quant`, `model.fc`). The scale is applied as a real division \
                 by a TENSOR in both. The input IS materialised bf16, so the amax sees bf16 and \
                 not fp32",
                format!(
                    "columns a whole number of {}-wide groups, and N = K (one axis, the GEMM's \
                     reduction)",
                    ml.quant_group
                ),
                "bf16 x in, e4m3 out with fp32 per-group scales",
            ),
            "head_topk_thresh" => (
                "pass one of the int2 draft head's own selection, `bm.topk(RERANK)` over the block \
                 maxima: a radix select for the row's k-th largest (value, ~index) composite. The \
                 composite is what makes ATen's tie order a non-question -- it is a TOTAL order \
                 (value descending, index ascending, which is `sbtopk`'s own at this shape), so \
                 the threshold is a single element and the gather needs no tie-break and no \
                 capacity guess",
                format!(
                    "k = {} exactly (the serve's SABER_DRAFT_RERANK), n >= k; one workgroup of \
                     {} per row",
                    hd.topk_k, hd.topk_block
                ),
                "fp32 block maxima in, one u64 composite per row out",
            ),
            "head_topk_gather" => (
                "... and pass two: every element at or above the threshold composite is in the \
                 answer, and there are exactly k of them because the order is total. The rank is \
                 k x k comparisons, one thread per entry. TWO launches is also ATen's own count \
                 for this selection (`sbtopk::gatherTopK`, 98.1 us), which is why covering it did \
                 not move the drafter's declared launch total",
                format!(
                    "k = {} exactly, n >= k; one workgroup of {} per row",
                    hd.topk_k, hd.topk_block
                ),
                "fp32 block maxima and the row's u64 threshold in, fp32 values and i32 indices \
                 out (prod's are i64 and the caller widens)",
            ),
            "row_gather" => (
                "`x[idx]` over whole rows: `last_hidden[sample_indices]` and \
                 `input_ids[_anchor_indices]`, the two ATen row gathers `_generate_draft` makes \
                 between the drafter's forward and its head. One op at two shapes -- a gather has \
                 a row WIDTH in bytes and no dtype -- and both were invisible to a staged replay, \
                 which starts from `head.hidden` and is handed the answer",
                format!(
                    "rows, row_bytes and src_rows all non-zero; workgroups of {}",
                    hd.pad_block
                ),
                "bytes in, bytes out, i64 indices -- a gather, so no rounding at all",
            ),
            "head_cand_gather" => (
                "`bi.gather(1, bm.topk(RERANK).indices)`: the int2 head's block-maxima POSITIONS \
                 mapped to the vocabulary COLUMNS the exact rerank scores. It is the third launch \
                 of that selection and no plan row named it until a chain asked what feeds the \
                 rerank's `idx` -- a staged replay seeds that slot from prod's own tensor and \
                 never notices",
                format!(
                    "k = {} exactly, n >= k; workgroups of {} over rows x k",
                    hd.topk_k, hd.pad_block
                ),
                "i32 in, i32 out -- a gather, so no rounding at all",
            ),
            "head_group_sum" => (
                "the int2 draft head's `xs`, `x.reshape(M, ng, GROUP).float().sum(-1)` -- the \
                 per-128-group fp32 column sum the head reads as a kernarg and that the plan \
                 treated as an INPUT until a chain asked for its producer. The ORDER is ATen's \
                 own, read out of `ATen/native/hip/Reduce.cuh` in the serving image: one float4 \
                 per lane summed left to right, then `block_x_reduce`'s ROCm fold, which is \
                 ascending (a balanced 32-lane tree)",
                format!(
                    "group {} exactly, n == k and a whole number of groups; one WAVE per (row, \
                     group), {} of them per workgroup",
                    hd.group, hd.group_sum_waves
                ),
                "bf16 x in, fp32 group sums out (prod materialises the fp32 cast first; this \
                 kernel absorbs it)",
            ),
            "head_row_pad" => (
                "`_pow2_at_least`'s row pad, `torch.cat([x, x.new_zeros(M - m, k)])`: the head's \
                 input padded up to `tl.dot`'s 16 rows. Prod issues the fill and the \
                 concatenation as two ATen launches and this kernel writes both halves, the same \
                 absorption the head's own `_topk` variant makes of prod's `y.fill_(-inf)`. A \
                 copy computes nothing, so the only way it can be wrong is the rows it writes",
                format!(
                    "m a power of two at or above {}, 0 < live rows <= m, n == k; workgroups of \
                     {}",
                    hd.bm, hd.pad_block
                ),
                "bf16 in, bf16 out -- a copy and a fill, so no rounding at all",
            ),
            "ctx_kv_layout" => (
                "the context store's `all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, \
                 3, 4).contiguous()` -- an ATen copy that computes NOTHING and is still a launch \
                 prod makes. The plan called it a fusion to take rather than a kernel to port, \
                 which is true of a Saber store that has the projection write layer-major and \
                 false of a CHAIN: nothing else stands between `bgemm_bf16` and the grouped K \
                 norm, and an absent op in a chain contributes zeros. The innermost nkv x head_dim \
                 run is contiguous on both sides, so one workgroup copies one run",
                format!(
                    "head_dim > 0; the grid is 2 x layers x num_ctx workgroups of {}",
                    cx.layout_block
                ),
                "bf16 in, bf16 out -- a permutation, so no rounding at all",
            ),
            "qk_norm_rope_draft" => (
                "the drafter's QUERY q/k norm + rope, one launch a layer: a per-head plain \
                 `RMSNorm` over each 128-wide head of the ungated `[q | k | v]` projection, then \
                 a full NEOX rope, q and k to separate outputs. NOT an `attn_glue` \
                 instantiation, and not for a shape reason: that plugin implements the TARGET's \
                 GemmaRMSNorm (`x * rsqrt * (1 + w)`) over ATen's reduction with a rope that does \
                 not contract, and is compiled `-ffp-contract=off` for it. The drafter's norm has \
                 no `1 +`, its variance is Triton's `tl.sum` (a 32-lane DPP ladder), its normed \
                 value is never rounded to bf16 before the rope reads it -- there is not one \
                 `tl.bfloat16` in the generated module -- and its rope contracts, one product \
                 rounded and the other fused, with the two outputs rounding DIFFERENT products",
                "head_dim 128 exactly and both head counts non-zero; one WAVE per (token, head), \
                 four contiguous elements a lane"
                    .to_owned(),
                "bf16 projection and bf16 [128] norm weights in, i64 positions, a bf16 \
                 [max_position, 128] cos/sin cache, bf16 q and k out; fp32 throughout with one \
                 rounding at the store",
            ),
            other => {
                return Err(PlanError::Model(format!(
                    "the dflash rules name op `{other}`, which this planner cannot describe"
                )))
            }
        };
        let mut imp = Implementation::new(&format!("dflash_{}", row.op), &plugin, &row.op)
            .describe(computes, &envelope, dtypes)
            .capture(true, false)
            .code_object(wave, art.max_lds(&row.symbol))
            .launches(LaunchModel::Fixed(1))
            .promoted(Promotion::implemented());
        // The constraints are the ones `saber_kernels::dflash::resolve` itself enforces and no
        // others. The TILE is not among them because it is not a choice: the rules loader refuses
        // a file naming a tile the code object was not compiled for, which is stronger than a band.
        imp = match row.op.as_str() {
            "gemm_a8w8_blockscale" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::le("m", i64::from(kn.bm)))
                .constrain(Constraint::ge("n", i64::from(kn.bn)))
                .constrain(Constraint::le("n", i64::from(kn.max_n)))
                .constrain(Constraint::ge("k", i64::from(kn.group))),
            "draft_head_int2" => imp
                .constrain(Constraint::ge("m", i64::from(hd.bm)))
                .constrain(Constraint::le("m", i64::from(hd.bm)))
                .constrain(Constraint::ge("n", i64::from(hd.bn)))
                .constrain(Constraint::ge("k", i64::from(4 * hd.group))),
            // The M-tiled arms' envelopes start where their tiles' stop: each pair is one program
            // over two disjoint M bands, and the route picks by M.
            "gemm_a8w8_blockscale_mt" => imp
                .constrain(Constraint::ge("m", i64::from(kn.bm) + 1))
                .constrain(Constraint::le("m", i64::from(kn.max_m)))
                .constrain(Constraint::ge("n", i64::from(kn.bn)))
                .constrain(Constraint::le("n", i64::from(kn.max_n)))
                .constrain(Constraint::ge("k", i64::from(kn.group))),
            // The BAND and the row slice share an M envelope and are picked apart by whether M is
            // a whole number of bands, which is a `resolve` refusal rather than a constraint the
            // planner can carry: an envelope is a range and "a multiple of 32" is not one.
            "draft_head_int2_mt" | "draft_head_int2_m32" => imp
                .constrain(Constraint::ge("m", i64::from(hd.bm) + 1))
                .constrain(Constraint::le("m", i64::from(hd.max_m)))
                .constrain(Constraint::ge("n", i64::from(hd.bn)))
                .constrain(Constraint::ge("k", i64::from(4 * hd.group))),
            "bgemm_bf16" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::le("m", i64::from(bg.m)))
                .constrain(Constraint::ge("n", i64::from(bg.n)))
                .constrain(Constraint::le("n", i64::from(bg.max_n)))
                .constrain(Constraint::ge("k", i64::from(bg.k_multiple))),
            // The fp8-weight head (N4): the head's own N envelope, the tile then the band by M.
            "bgemm_w8_head" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::le("m", i64::from(bg.m)))
                .constrain(Constraint::ge("n", i64::from(bg.n)))
                .constrain(Constraint::le("n", i64::from(bg.head_max_n)))
                .constrain(Constraint::ge("k", i64::from(bg.k_multiple))),
            "bgemm_w8_head_m32" => imp
                .constrain(Constraint::ge("m", i64::from(bg.m) + 1))
                .constrain(Constraint::le("m", i64::from(bg.max_m)))
                .constrain(Constraint::ge("n", i64::from(bg.n)))
                .constrain(Constraint::le("n", i64::from(bg.head_max_n)))
                .constrain(Constraint::ge("k", i64::from(bg.k_multiple))),
            // H33: the wide band starts where the 32-row band's route stops.
            "bgemm_w8_head_wide" => imp
                .constrain(Constraint::ge("m", i64::from(bg.m2) + 1))
                .constrain(Constraint::le("m", i64::from(bg.max_m)))
                .constrain(Constraint::ge("n", i64::from(bg.n)))
                .constrain(Constraint::le("n", i64::from(bg.head_max_n)))
                .constrain(Constraint::ge("k", i64::from(bg.k_multiple))),
            // The M-tiled arm's envelope starts where the tile's stops: the two are one program
            // over two disjoint M bands, and the route picks by M.
            "bgemm_bf16_mt" | "bgemm_bf16_m32" => imp
                .constrain(Constraint::ge("m", i64::from(bg.m) + 1))
                .constrain(Constraint::le("m", i64::from(bg.max_m)))
                .constrain(Constraint::ge("n", i64::from(bg.n)))
                .constrain(Constraint::le("n", i64::from(bg.max_n)))
                .constrain(Constraint::ge("k", i64::from(bg.k_multiple))),
            "selector_score" => imp
                .constrain(Constraint::ge("k", i64::from(sl.rank)))
                .constrain(Constraint::le("k", i64::from(sl.rank)))
                .constrain(Constraint::ge("r", i64::from(sl.top_k)))
                .constrain(Constraint::le("r", i64::from(sl.top_k))),
            "grouped_conv" | "grouped_conv_quant" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", i64::from(cv.quant_group)))
                .constrain(Constraint::ge("k", 1)),
            "attn_bf16kv" | "attn_fp8kv" | "attn_bf16kv_ns" | "attn_fp8kv_ns" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", i64::from(at.gqa)))
                .constrain(Constraint::ge("k", 1)),
            "kv_append_bf16" | "kv_append_fp8" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("k", i64::from(at.head)))
                .constrain(Constraint::le("k", i64::from(at.head))),
            "silu_quant" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", i64::from(ml.quant_group))),
            "conv_add_rms_open" | "conv_add_rms_close" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", i64::from(cv.group_size)))
                .constrain(Constraint::ge("k", 1)),
            "selector_walk" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", 1))
                .constrain(Constraint::ge("r", i64::from(sl.top_k)))
                .constrain(Constraint::le("r", i64::from(sl.top_k))),
            "selector_topk" => imp
                .constrain(Constraint::ge("r", i64::from(kv.topk_k)))
                .constrain(Constraint::le("r", i64::from(kv.topk_k)))
                .constrain(Constraint::ge("n", i64::from(kv.topk_k))),
            "head_topk_thresh" | "head_topk_gather" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("r", i64::from(hd.topk_k)))
                .constrain(Constraint::le("r", i64::from(hd.topk_k)))
                .constrain(Constraint::ge("n", i64::from(hd.topk_k))),
            "act_quant" | "act_quant_recip" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", i64::from(ml.quant_group)))
                .constrain(Constraint::ge("k", i64::from(ml.quant_group))),
            "ctx_kv_layout" => imp
                .constrain(Constraint::ge("m", 1))
                .constrain(Constraint::ge("n", 1))
                .constrain(Constraint::ge("k", 1))
                .constrain(Constraint::ge("r", 1)),
            _ => imp
                .constrain(Constraint::ge("r", 1))
                .constrain(Constraint::le("r", i64::from(hd.max_r)))
                .constrain(Constraint::ge("k", i64::from(hd.rerank_block_k))),
        };
        rows.push(imp);
    }
    Ok(rows)
}

/// Every head symbol of a row must land in the same plugin. Two different answers would mean the
/// row describes a route split across code objects, which nothing in the kernel plane can launch —
/// so it is an error here rather than a row that quietly attaches to whichever artifact is
/// enumerated first.
pub fn row_owner(row: &RegistryRow, rules: &SaberRules) -> Result<String> {
    let heads = head_symbols_for(row, rules);
    if heads.is_empty() {
        return Err(PlanError::Model(format!(
            "registry row `{}` (op `{}`) maps to no symbol in any instantiation table",
            row.entry, row.op
        )));
    }
    let mut owner: Option<(String, String)> = None;
    for sym in &heads {
        let p = plugin_for(sym)
            .map_err(|e| PlanError::Model(format!("registry row `{}`: {e}", row.entry)))?;
        match &owner {
            None => owner = Some((p.to_string(), sym.clone())),
            Some((have, first)) if have != p => {
                return Err(PlanError::Model(format!(
                    "registry row `{}` is split across plugins: `{first}` is in `{have}` but \
                     `{sym}` is in `{p}`",
                    row.entry
                )));
            }
            Some(_) => {}
        }
    }
    Ok(owner.expect("checked non-empty").0)
}

/// One [`Implementation`] per `[[registry]]` row whose symbols this artifact carries.
///
/// The row IS the implementation: the reference's registry is the model the kernel plane was built on, and
/// re-deriving its predicates here would be the second transcription the whole design exists to
/// avoid. What this function adds is what the row cannot carry — the wave size and LDS read out of
/// the code object, the workspace kinds, and the static launch count.
///
/// Per-call conditions are deliberately **not** turned into constraints. `nseq * H <=
/// FUSED_MAX_ITEMS` for the fused GDN update, `q_len * gqa <= 64` for attention decode, and the
/// split-KV scratch capacity are all checked by `saber::resolve` at the entry point, exactly as
/// the reference checks them: "constraints are NECESSARY conditions that discriminate ops, never tiles;
/// strides, alignment and buffer sizes stay per-call at the entry point."
fn saber_implementations_for(
    art: &PluginArtifact,
    rules: &SaberRules,
) -> Result<Vec<Implementation>> {
    let wave = art.wave_size()?;
    let plugin = art.plugin().to_string();
    let mut rows = Vec::new();
    for row in &rules.registry {
        // Ownership is decided by the head dispatch, through `saber_kernels::saber::plugin_for` —
        // the AUTHORITATIVE symbol -> plugin map, and prefix-ordered, so
        // `saber_ar_oneshot_2rank_exact_*` reaches `saber_ar_exact` before the generic
        // `saber_ar_oneshot_` arm claims it for `saber_ar_wht`.
        //
        // Asking the artifact's own manifest instead ("does this blob contain the symbol?") looks
        // equivalent and is not: a row whose symbols are in no committed blob is attributed to
        // NOTHING and vanishes, which is exactly how `saber_ar_exact` ended up with zero rows. Going
        // through the mapping makes that a refusal.
        let owner = row_owner(row, rules)?;
        if owner != art.plugin() {
            continue;
        }
        let symbols = symbols_for(row, rules);
        let mine: Vec<&String> = symbols
            .iter()
            .filter(|sym| art.manifest.kernel(sym).is_some())
            .collect();
        if mine.is_empty() {
            continue;
        }
        let Some(facts) = saber_op_facts(row) else {
            return Err(PlanError::Model(format!(
                "{plugin}: registry row `{}` has op `{}`, which this build has no workspace or \
                 launch-count facts for",
                row.entry, row.op
            )));
        };
        let lds = mine
            .iter()
            .filter_map(|sym| art.manifest.kernel(sym))
            .map(|k| k.group_segment_fixed_size)
            .max()
            .unwrap_or(0);
        let shape = row
            .constraints
            .iter()
            .map(|c| plugin_row_constraint(c).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut imp = Implementation::new(&row.entry, &plugin, &row.op)
            .describe(
                facts.computes,
                &format!("{shape} ({} instantiations)", mine.len()),
                "bf16 activations; see the constraint keys for the KV and payload dtypes",
            )
            // Every saber kernel prod runs is recorded inside a captured bucket today. The
            // all-reduce spins on a PEER's flag, which is device-side and legal under capture --
            // it is not `may_sync`, whose meaning is a device-wide barrier or a host flag -- and
            // its per-block sequence counter is a peer-visible workspace precisely so that a
            // replay does not reset protocol state the other rank is still reading.
            .capture(true, false)
            .code_object(wave, lds)
            .launches(LaunchModel::Fixed(facts.launches))
            .promoted(Promotion::implemented());
        for c in &row.constraints {
            imp = imp.constrain(plugin_row_constraint(c));
        }
        for (name, rule) in facts.workspaces {
            // Both rule files spell workspace kinds with the same type; the saber table is a map
            // keyed on the RULE, which is not always the workspace's name: a rank allocates its own
            // all-reduce scratch and maps the peer's, and one rule sizes both.
            let w = rules.workspace.get(*rule).ok_or_else(|| {
                PlanError::Model(format!(
                    "{plugin}: op `{}` needs workspace `{name}` (rule `{rule}`), which the rules \
                     file does not declare",
                    row.op
                ))
            })?;
            // Sized once at construction for the worst shape the engine can schedule:
            // allocating during a graph capture is not allowed and the address has to survive
            // replay, so the capacity IS the size law. The split-KV partials carry the law
            // alongside it so `explain` can say what the bucket actually uses.
            let sizing = if *rule == "attn_decode_scratch" {
                WorkspaceSizing::SplitKvPartials {
                    capacity: w.capacity_bytes,
                    bytes_per_split: rules.knobs.combine_partial_bytes_per_split,
                }
            } else {
                WorkspaceSizing::Fixed(w.capacity_bytes)
            };
            imp = imp.workspace(WorkspaceReq::new(name, ws_kind(w.kind), sizing));
        }
        rows.push(imp);
    }
    if rows.is_empty() {
        return Err(PlanError::Model(format!(
            "{plugin}: no registry row's symbols are in this code object"
        )));
    }
    Ok(rows)
}

fn ws_kind(k: RuleWsKind) -> WorkspaceKind {
    match k {
        RuleWsKind::Scratch => WorkspaceKind::Scratch,
        RuleWsKind::Persistent => WorkspaceKind::Persistent,
        RuleWsKind::IpcPeerVisible => WorkspaceKind::IpcPeerVisible,
    }
}

/// `saber-plan` has no logging dependency; the note goes to stderr, once per unknown plugin, so a
/// registry that silently covers less than the directory does is visible.
fn tracing_note(plugin: &str) {
    eprintln!(
        "saber-plan: prebuilt plugin `{plugin}` has no launch-rule description; \
         no registry row was derived for it"
    );
}

/// Refuse a registry whose **bindable** implementations name a plugin the manifest does not declare.
///
/// > The plugin that ships it; must be declared in the manifest's `[[plugins]]`.
/// > — [`Implementation::plugin`]
///
/// Checked rather than assumed, because the failure is a plan that resolves and then cannot load.
///
/// "Bindable" is the word that earns its keep. The registry is derived from the whole prebuilt
/// directory — everything that *ships* — while a manifest declares what this deployment *loads*,
/// and since the Paro producers landed those two sets differ: `paro_producers` is the
/// rollback tier's plugin, its rows implement ops (`paro_add_rms_rot`, …) that
/// `ModelSpec::qwen38_27b_dense` never names, and `resolve` therefore cannot bind one. Refusing on
/// its account would make `saber explain` fail for a row it is only *listing*. So the check is over
/// the rows whose op this model actually selects on; a row for an op the model does not have
/// cannot cause a load failure, and declaring its plugin anyway would be worse than pointless —
/// `Manifest::plugin_build_hash` covers every declared plugin, so it would invalidate every
/// autotune record in the cache.
pub fn check_plugins_declared(
    registry: &Registry,
    manifest: &PlanManifest,
    model: &crate::ModelSpec,
) -> Result<()> {
    // Use the same bucket geometry and call count as resolve(). The narrow and
    // 16-query uniform heads share an op and keys but have disjoint widths.
    let bindable = |i: &crate::registry::Implementation| {
        model.ops.iter().filter(|o| o.op == i.op).any(|o| {
            manifest.buckets.iter().any(|section| {
                let b = manifest.bucket(section);
                o.calls_in(b.phase, b.batch, b.m()) != 0
                    && i.matches(&o.shape_keys(b.phase, b.m(), b.batch, b.spec_width, b.ctx_band))
                        .is_ok()
            })
        })
    };
    let mut missing: Vec<&str> = registry
        .implementations()
        .iter()
        .filter(|i| bindable(i))
        .flat_map(|i| {
            std::iter::once(i.plugin.as_str())
                .chain(saber_kernels::companion_plugins(&i.plugin).iter().copied())
        })
        .filter(|p| manifest.plugin(p).is_none())
        .collect();
    missing.sort_unstable();
    missing.dedup();
    if missing.is_empty() {
        return Ok(());
    }
    Err(PlanError::Model(format!(
        "the registry binds plugin(s) the manifest does not declare in [[plugins]]: {}",
        missing.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ShapeKeys;
    use std::collections::{BTreeMap, BTreeSet};

    fn artifacts() -> Vec<PluginArtifact> {
        load_artifacts(&saber_kernels::prebuilt::dir()).expect("the prebuilt plugins are committed")
    }

    fn rules() -> RuleSets {
        RuleSets::prod().expect("both prod rule files parse")
    }

    #[test]
    fn every_plugin_the_rules_describe_yields_a_row_and_no_other_plugin_does() {
        let arts = artifacts();
        assert!(arts.len() >= saber_kernels::prebuilt::ALL.len());
        // The saber rules landed (workstreams F + H2/H3), so the test the comment below once
        // described has tightened: EVERY committed plugin is described by one of the two rules
        // files and yields at least one row. An undescribed plugin would be a silent gap.
        let rules = rules();
        let paro_owner = rules
            .paro
            .owner()
            .expect("the paro rules name their plugin");
        let glue_owner = rules
            .glue
            .owner()
            .expect("the attn_glue rules name their plugin");
        let mut mxfp4 = 0usize;
        let mut saber = 0usize;
        let mut paro = 0usize;
        let mut paro_rows = 0usize;
        let mut noop = 0usize;
        let mut noop_rows = 0usize;
        let mut head = 0usize;
        let mut head_rows = 0usize;
        let mut glue = 0usize;
        let mut glue_rows = 0usize;
        let mut rejected = 0usize;
        let mut elsewhere = 0usize;
        let mut dflash = 0usize;
        let mut dflash_rows = 0usize;
        for art in &arts {
            let rows = implementations_for(art, &rules).expect("rows");
            // A plugin produces rows unless it is a NAMED measured-and-rejected one, and then it
            // produces none. Both directions are asserted, so neither a new undescribed plugin
            // nor a rejected plugin that quietly grew rows can pass.
            if let Some(why) = separately_planned(art.plugin()) {
                assert!(
                    rows.is_empty(),
                    "{} is planned outside the text plan and must contribute no rows ({why})",
                    art.plugin()
                );
                elsewhere += 1;
                continue;
            }
            match rejected_reason(art.plugin()) {
                Some(why) => {
                    assert!(
                        rows.is_empty(),
                        "{} is measured-and-rejected and must contribute no rows ({why})",
                        art.plugin()
                    );
                    rejected += 1;
                    continue;
                }
                None => assert!(!rows.is_empty(), "{} produced no rows", art.plugin()),
            }
            if art.plugin() == rules.noop.owner() {
                noop += 1;
                noop_rows += rows.len();
            } else if art.plugin() == rules.head.owner() {
                head += 1;
                head_rows += rows.len();
            } else if art.plugin() == glue_owner {
                glue += 1;
                glue_rows += rows.len();
            } else if rules.dflash.owns(art.plugin()) {
                dflash += 1;
                dflash_rows += rows.len();
            } else if art.plugin() == paro_owner {
                // Asked of the rules' own map, not of the stem: `paro_producers` is a name in one
                // place (`rules/paro_gfx1201.toml` via `ParoRules::owner`), and this is that place.
                paro += 1;
                paro_rows += rows.len();
            } else if art.plugin().starts_with("mxfp4_") {
                mxfp4 += 1;
            } else if art.plugin().starts_with("saber_") {
                saber += 1;
            } else {
                panic!("{}: a plugin no rules file describes", art.plugin());
            }
        }
        assert_eq!(mxfp4, 5, "the MXFP4 W4A8 family");
        assert_eq!(
            elsewhere,
            SEPARATELY_PLANNED.len(),
            "the vision tower, planned per grid, and row 3, planned by the block planner's fold"
        );
        assert_eq!(paro, 1, "the Paro producers are ONE code object");
        // Four producer families x two M bands each: ROT / plain for the per-group pair, TILED /
        // row-wise for the per-token pair. The bands are the launcher's own decisions, so a row
        // appearing or vanishing here means the rules changed a decision, not a count.
        assert_eq!(paro_rows, 8, "four families x two bands");
        // The null plugin is described like any other: one code object, one row per null op.
        assert_eq!(noop, 1, "the null plugin is ONE code object");
        assert_eq!(noop_rows, 2, "noop and noop_touch");
        // The head plugin the same way: one code object, one row per head op. Three of the six
        // are the places where prod reaches for torch instead of a kernel -- the embedding
        // lookup, the row selection before `lm_head`, and the argmax; the fourth is the sampler's
        // seam, which prod does not have at all because prod reads the dense row back. The last
        // two are ws-s3's: the DFlash2 aux tap (a strided copy into a column block of the
        // drafter's `aux.cat`) and the i32 -> i64 widen that bridges the selector's candidate ids.
        assert_eq!(head, 1, "the head plugin is ONE code object");
        assert_eq!(
            head_rows, 7,
            "embedding, row_gather, argmax, the sampler's seam, the aux tap, the widen and N2's bf16 head norm"
        );
        // The attention glue the same way: one code object, one row per op. Three ops, because
        // an attention layer has exactly three places where prod ran Triton or ATen and the
        // P1 block had to supply the answer -- the q/k norm + rope, the output gate's quant,
        // and the paged K/V write.
        assert_eq!(glue, 1, "the attention glue is ONE code object");
        assert_eq!(glue_rows, 3, "qk_norm_rope, attn_gate_quant and kv_append");
        // The drafter's code objects, and rows grow as the rest of the DFlash2 path lands
        // (the census lists what is left). Each one has to be described HERE before it can ship --
        // the invariant that caught that plugin on the day it was built, and the one the attention
        // glue's rows exist to satisfy too. The twelfth is `dflash_qk`, the drafter's own q/k
        // norm + rope: a THIRD norm-and-rope code object, because there are three references --
        // `attn_glue` is the TARGET's (GemmaRMSNorm, ATen's reduction, no contraction),
        // `dflash_ctx` is the C++ `upstream::` pair the context store calls directly, and this one is
        // what inductor compiles for the drafter's own query rows.
        assert_eq!(dflash, 12, "the drafter's twelve code objects");
        assert_eq!(
            dflash_rows, 41,
            "the blockscale GEMM, the int2 head + THREE rerank arms (e4m3-plus-scale, the bf16 \
             one, and N6's bf16 arithmetic over the fp8 record the default serve runs), the bf16 projection (and N4's fp8-weight head at its two bands and H33's wide band), the selector, the \
             convolution with and without its fused quant, the fused conv+add+norm at BOTH ends \
             of prod's fp32 residual, the attention and the KV append at both \
             KV dtypes, the selector's top-k and its path walk, the three MLP producers -- the \
             activation, the fused conv+add+norm, and layer 0's standalone norm -- the context \
             store's two norms-and-rope ops, whose reference is the C++ `upstream::` op rather than \
             inductor's Triton, F26's four (the blockscale GEMM's own activation quant, the int2 \
             head's two-pass top-80, and the context store's layout copy) and F27's four, which the \
             backbone's chain found between the drafter's forward and its head: \
             `_pow2_at_least`'s row pad, `xs`, the `bi` gather that turns the top-80's positions \
             into vocabulary columns, and the row gather that is two of them -- and F31's one, \
             which is F26's activation quant reached the other way: inductor strength-reduces \
             `absmax / 448` INSIDE the drafter's region and ATen does not outside it, so \
             `act_quant` (`model.fc`) and `act_quant_recip` (the five `o_proj`s) are two ops, and the THREE M-TILED instantiations -- S6's bf16 projection and F33's blockscale GEMM and int2 head -- each a SECOND symbol over the M band above its own tile, because a BATCHED proposal runs at `reqs * block` rows and all three tiles the proposal was gated at stop at 16 -- and F34's TWO 32-ROW BANDS beside two of those three row slices (the int2 head and the bf16 projection), which are what the rules pick above the tile because a row slice reads the weight once per 16-row TILE and a band reads it once for thirty-two rows -- and S11's TWO, the attention's `num_seqs` walk at both KV dtypes, which is the nineteenth site of nineteen entering the grid: a batched proposal was `reqs` launches of the one-sequence kernel at static offsets and is ONE launch now"
        );
        // And exactly one plugin ships without rows on purpose.
        assert_eq!(rejected, 1, "mxfp4_gemm_decode_silu");
        assert_eq!(
            mxfp4 + saber + paro + noop + head + glue + dflash + rejected + elsewhere,
            saber_kernels::prebuilt::ALL.len()
        );
    }

    /// A plugin can be absent from the registry for exactly one reason, and the reason is
    /// written down.
    ///
    /// The alternative — letting `implementations_for` fall through and describe the fused
    /// instantiation as another `gemm_w4a8` candidate — is the failure this guards: `select`
    /// would be free to bind an experiment the ledger measured at 8x to 26x slower, and nothing
    /// in the plan would say so.
    #[test]
    fn a_measured_and_rejected_plugin_contributes_no_rows_and_says_why() {
        let rules = rules();
        let arts = artifacts();
        for (plugin, why) in REJECTED_PLUGINS {
            // It SHIPS. A rejected plugin that stopped shipping should lose its entry, not keep
            // a note about a code object nobody can measure again.
            let art = arts
                .iter()
                .find(|a| a.plugin() == *plugin)
                .unwrap_or_else(|| panic!("{plugin} is not a committed code object"));
            assert!(
                implementations_for(art, &rules).expect("rows").is_empty(),
                "{plugin} contributed rows"
            );
            // The reason says what happened and when, so a reader does not have to guess
            // whether it was never tried or tried and lost.
            assert!(why.contains("measured and rejected"), "{plugin}: {why}");
            assert!(
                why.contains("2026-"),
                "{plugin}: the reason carries no date"
            );
        }
        // And the registry that results names no op of that plugin.
        let registry = registry_from_artifacts(&arts, &rules).expect("registry");
        assert!(
            registry
                .implementations()
                .iter()
                .all(|i| rejected_reason(&i.plugin).is_none()),
            "a rejected plugin reached the registry"
        );
    }

    /// The bug class this file has now produced twice: a saber registry row whose symbols no
    /// artifact claimed, so the row silently vanished and its plugin came out empty. Counting
    /// rows per plugin cannot catch it — the row is simply absent — so the invariant is stated
    /// directly: **every row is owned by exactly one plugin, and every plugin owns at least one
    /// row.**
    #[test]
    fn every_registry_row_is_attributed_to_exactly_one_plugin() {
        let rules = rules();
        let arts = artifacts();
        let mut owners: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for row in &rules.saber.registry {
            let owner = row_owner(row, &rules.saber)
                .unwrap_or_else(|e| panic!("row `{}` has no owner: {e}", row.entry));
            // The owner must be a plugin that actually ships, not just a name the map knows.
            assert!(
                arts.iter().any(|a| a.plugin() == owner),
                "row `{}` is owned by `{owner}`, which is not a committed plugin",
                row.entry
            );
            owners.entry(row.entry.as_str()).or_default().push("x");
            // And the artifact-driven pass must agree with the map, for this row and no other.
            let claimed: Vec<&str> = arts
                .iter()
                .filter(|a| {
                    implementations_for(a, &rules)
                        .expect("rows")
                        .iter()
                        .any(|i| i.name == row.entry)
                })
                .map(PluginArtifact::plugin)
                .collect();
            assert_eq!(
                claimed,
                vec![owner.as_str()],
                "row `{}` was claimed by {claimed:?}",
                row.entry
            );
        }

        // Every saber plugin owns at least one row. This is the half that fails loudly when a
        // new plugin lands and nothing here has been taught about it.
        let mut owned: BTreeSet<String> = BTreeSet::new();
        for row in &rules.saber.registry {
            owned.insert(row_owner(row, &rules.saber).expect("owner"));
        }
        // ws-h42: `saber_gdn_fnq` is `saber_`-named for the reference's build flag, but it is a fold-only
        // plugin (listed with its reason in the fold table), which by design owns no row.
        for art in arts.iter().filter(|a| {
            a.plugin().starts_with("saber_") && separately_planned(a.plugin()).is_none()
        }) {
            assert!(
                owned.contains(art.plugin()),
                "`{}` ships but owns no registry row",
                art.plugin()
            );
        }

        // The two all-reduce families are the case that broke: same op, different plugins.
        assert_eq!(
            row_owner(
                rules
                    .saber
                    .registry
                    .iter()
                    .find(|r| r.entry == "ar_oneshot_2rank_exact")
                    .expect("the exact row"),
                &rules.saber
            )
            .unwrap(),
            "saber_ar_exact"
        );
        assert_eq!(
            row_owner(
                rules
                    .saber
                    .registry
                    .iter()
                    .find(|r| r.entry == "ar_oneshot_2rank_wht6")
                    .expect("the wht6 row"),
                &rules.saber
            )
            .unwrap(),
            "saber_ar_wht"
        );
    }

    /// The Paro rows' bands have to be the ones `paro::resolve` actually picks, or the
    /// registry is describing a route the launcher does not take. Both edges, both families.
    #[test]
    fn the_paro_bands_are_the_ones_paro_resolve_picks() {
        use saber_kernels::paro::{self, Op, ParoEwShape, ParoShape, Shape};

        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).expect("registry");
        let paro_rules = rules().paro;
        let kn = paro_rules.knobs.clone();
        // The band edge is NOT a paro knob: one number, read from the MXFP4 rules (I20).
        let band = saber_kernels::rules::a_tiled_min_m();
        let eps = 1e-6f32.to_bits();
        let names = |op: &str, keys: &ShapeKeys| -> Vec<String> {
            reg.select(op, keys)
                .unwrap_or_default()
                .iter()
                .map(|id| reg.get(*id).unwrap().name.clone())
                .collect()
        };
        let rms = |m: u32| {
            ShapeKeys::new()
                .with("m", i64::from(m))
                .with("k", 5120i64)
                .with("krot", 4i64)
                .with("p", 1i64)
        };
        let ew = |m: u32| {
            ShapeKeys::new()
                .with("m", i64::from(m))
                .with("n", 5120i64)
                .with("krot", 4i64)
                .with("mode", "0")
        };
        let rms_shape = |m: u32| {
            Shape::Rms(ParoShape {
                m,
                k: 5120,
                p: 1,
                krot: 4,
                eps_bits: eps,
            })
        };
        let ew_shape = |m: u32| {
            Shape::Ew(ParoEwShape {
                m,
                n: 5120,
                krot: 4,
                ys: 0,
                eps_bits: eps,
            })
        };

        // ---- per-GROUP: the ROT edge is `decode_max_m`, inclusive.
        assert_eq!(
            names("paro_add_rms_rot", &rms(kn.decode_max_m)),
            vec!["paro_add_rms_rot_fused"]
        );
        let d = paro::resolve(Op::AddRmsRot, rms_shape(kn.decode_max_m), &paro_rules).unwrap();
        assert!(d.kernel.contains("rot1"), "{}", d.kernel);
        assert_eq!(
            names("paro_add_rms_rot", &rms(kn.decode_max_m + 1)),
            vec!["paro_add_rms_rot_plain"]
        );
        let d = paro::resolve(Op::AddRmsRot, rms_shape(kn.decode_max_m + 1), &paro_rules).unwrap();
        assert!(d.kernel.contains("rot0"), "{}", d.kernel);
        assert_eq!(d.n_launches, 1, "a producer is one dispatch");
        assert!(d.workspaces.is_empty(), "these producers take no workspace");

        assert_eq!(
            names("paro_ew_rot", &ew(kn.decode_max_m)),
            vec!["paro_ew_rot_fused"]
        );
        let d = paro::resolve(
            Op::EwRot { mode: 0 },
            ew_shape(kn.decode_max_m),
            &paro_rules,
        )
        .unwrap();
        assert!(d.kernel.contains("m0_rot1"), "{}", d.kernel);

        // ---- per-TOKEN: the TILED edge is `a_tiled_min_m`, and it is the SAME number the A-tiled
        // GEMM row constrains on. Two rule files, one decision: the producer writes the layout
        // that GEMM reads, and a disagreement would be silent garbage.
        assert_eq!(
            names("paro_add_rms_rot_tok", &rms(band - 1)),
            vec!["paro_add_rms_rot_tok_rowwise"]
        );
        let d = paro::resolve(Op::AddRmsRotTok, rms_shape(band - 1), &paro_rules).unwrap();
        assert!(d.kernel.ends_with("tiled0"), "{}", d.kernel);
        assert_eq!(
            names("paro_add_rms_rot_tok", &rms(band)),
            vec!["paro_add_rms_rot_tok_tiled"]
        );
        let d = paro::resolve(Op::AddRmsRotTok, rms_shape(band), &paro_rules).unwrap();
        assert!(d.kernel.ends_with("tiled1"), "{}", d.kernel);
        assert_eq!(
            names("paro_ew_rot_tok", &ew(band)),
            vec!["paro_ew_rot_tok_tiled"]
        );

        let (_, atiled) = reg
            .by_name("gemm_w4a8_mxfp4_atiled")
            .expect("the A-tiled row");
        let gemm_edge = atiled
            .constraints
            .iter()
            .find(|c| c.key == "m")
            .expect("the A-tiled row bands on M")
            .ival;
        assert_eq!(
            gemm_edge,
            i64::from(band),
            "the producer and its consumer must switch layout at the same M"
        );

        // ---- and nothing binds them: the PARO tier is the rollback, so the served model names no
        // `paro_*` op and these rows appear in the registry table only.
        let model = crate::ModelSpec::qwen38_27b_dense(2).expect("model");
        assert!(
            !model.ops.iter().any(|o| o.op.starts_with("paro_")),
            "the dense NVFP4 model must not name a Paro op"
        );
    }

    #[test]
    fn wave_size_and_lds_come_from_the_code_object_not_from_a_constant() {
        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).expect("registry");
        let (_, decode) = reg
            .by_name("gemm_w4a8_mxfp4_decode")
            .expect("the decode row exists");
        // gfx1201 kernels are built wave32; the point is that the number was READ, so assert it
        // against the manifest rather than against a literal typed here.
        let art = artifacts()
            .into_iter()
            .find(|a| a.plugin() == "mxfp4_gemm_decode")
            .unwrap();
        assert_eq!(decode.wave_size, art.wave_size().unwrap());
        assert_eq!(decode.lds_bytes, art.max_lds("mxfp4_gemm_decode"));
        assert!(decode.lds_bytes > 0, "the decode kernel stages through LDS");
    }

    #[test]
    fn the_decode_row_declares_both_workspace_kinds() {
        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
        let (_, decode) = reg.by_name("gemm_w4a8_mxfp4_decode").unwrap();
        let kinds: Vec<_> = decode.workspaces.iter().map(|w| w.kind).collect();
        assert_eq!(
            kinds,
            vec![WorkspaceKind::Scratch, WorkspaceKind::Persistent],
            "the split-K partials alias; the block counter must not"
        );
        // A split is still one launch: the reduce is fused into the decode kernel.
        assert_eq!(decode.launches.n_launches(4), 1);
    }

    #[test]
    fn the_bands_agree_with_what_resolve_actually_picks() {
        // The registry is only useful if its necessary conditions are the ones `resolve` enforces.
        // Three cells, each chosen because it separates two rows.
        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
        let shape = |m: i64, n: i64, k: i64| {
            ShapeKeys::new()
                .with("m", m)
                .with("n", n)
                .with("k", k)
                .with("w_dtype", "mxfp4")
                .with("a_dtype", "int8")
                // Required on every GEMM row: a query that does not say what its A-side edge
                // carries gets NO kernel rather than whichever row is listed first. What a
                // fused producer writes at this M.
                .with("a_layout", if m >= 513 { "a_tiled" } else { "row_major" })
        };

        // Decode band: gate_up at M=8 — decode and folded both cover it, and the tie is a
        // measurement, never registry order.
        let hits = reg.select("gemm_w4a8", &shape(8, 8704, 5120)).unwrap();
        let names: Vec<&str> = hits
            .iter()
            .map(|id| reg.get(*id).unwrap().name.as_str())
            .collect();
        assert!(names.contains(&"gemm_w4a8_mxfp4_decode"), "{names:?}");
        assert!(names.contains(&"gemm_w4a8_mxfp4_folded"), "{names:?}");
        assert!(!names.contains(&"gemm_w4a8_mxfp4_atiled"), "{names:?}");

        // lm_head at M=8: BOTH tiles cover it since 2026-09-19. `dec_max_n` is the split-K
        // scratch's width and the vocabulary width takes no split (nblk 970 clears every fill
        // target), so the decode row no longer excludes it -- and `resolve` answers decode here,
        // which is the point: the registry's necessary conditions must not refuse what the
        // launcher produces. The measurement, not registry order, picks between the two.
        let names: Vec<String> = reg
            .select("gemm_w4a8", &shape(8, 124_160, 5120))
            .unwrap()
            .iter()
            .map(|id| reg.get(*id).unwrap().name.clone())
            .collect();
        assert_eq!(
            names,
            vec![
                "gemm_w4a8_mxfp4_decode".to_string(),
                "gemm_w4a8_mxfp4_folded".to_string()
            ]
        );
        assert_eq!(
            saber_kernels::rules::resolve(
                saber_kernels::rules::Op::gemm(),
                saber_kernels::rules::Shape::new(8, 124_160, 5120),
                &saber_kernels::rules::LaunchRules::prod_mxfp4().unwrap(),
            )
            .unwrap()
            .plugin,
            "mxfp4_gemm_decode",
            "the registry admits the decode row because the launcher answers it here"
        );

        // Prefill: at M=2048 the A-tiled kernel joins.
        let names: Vec<String> = reg
            .select("gemm_w4a8", &shape(2048, 8704, 5120))
            .unwrap()
            .iter()
            .map(|id| reg.get(*id).unwrap().name.clone())
            .collect();
        assert!(
            names.contains(&"gemm_w4a8_mxfp4_atiled".to_string()),
            "{names:?}"
        );
    }

    #[test]
    fn the_unfolded_row_refuses_a_checkpoint_that_has_a_row_reference() {
        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
        let trace = reg.explain(
            "gemm_w4a8",
            &ShapeKeys::new()
                .with("m", 8i64)
                .with("n", 8704i64)
                .with("k", 5120i64)
                .with("w_dtype", "mxfp4")
                .with("a_dtype", "int8")
                .with("a_layout", "row_major"),
        );
        let unfolded = trace
            .candidates
            .iter()
            .find(|c| c.impl_name == "gemm_w4a8_mxfp4_unfolded")
            .expect("the unfolded row is a candidate, and is refused");
        assert!(!unfolded.ok);
        assert!(unfolded.reason.contains("w_ref"), "{}", unfolded.reason);
    }

    /// The shipped example manifest, which is the one `saber explain` is documented against.
    fn prod_manifest() -> PlanManifest {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("qwen38-27b-nvfp4-tp2.toml");
        PlanManifest::from_path(&path).expect("the shipped example manifest parses")
    }

    /// The all-reduce family is picked by the MESSAGE SIZE, and the plan says so per bucket.
    ///
    /// `_quant_ok` takes the rotated kernel at or above `quant_min_bytes` and the exact one below
    /// it. At hidden 5120 in bf16 that is 10,240 B per row, so the 128 KiB threshold falls between
    /// m = 12 (122,880 B) and m = 13 (133,120 B): b1s5 and b2s5 are exact, b4s5 and b8s5 are wht6.
    /// Prod runs both every step, and a plan that bound one of them everywhere would launch the
    /// wrong kernel for half the bucket set.
    #[test]
    fn the_small_buckets_bind_the_exact_all_reduce_and_the_large_ones_bind_wht6() {
        let manifest = prod_manifest();
        let registry = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).expect("registry");
        let device = crate::DeviceProfile::gfx1201_reference();
        let model = crate::ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
        check_plugins_declared(&registry, &manifest, &model).expect("every plugin is declared");
        let resolved =
            crate::resolve(&manifest, &device, &model, &registry).expect("the plan resolves");

        let mut by_bucket: BTreeMap<String, Vec<(String, Vec<String>)>> = BTreeMap::new();
        for r in resolved.requirements() {
            if r.op_name.starts_with("layer.allreduce") {
                by_bucket
                    .entry(r.bucket.label())
                    .or_default()
                    .push((r.op_name.clone(), r.candidates.clone()));
            }
        }

        for (bucket, want) in [
            ("b1s5c8192", "ar_oneshot_2rank_exact"),
            ("b2s5c8192", "ar_oneshot_2rank_exact"),
            ("b4s5c8192", "ar_oneshot_2rank_wht6"),
            ("b8s5c8192", "ar_oneshot_2rank_wht6"),
        ] {
            let rows = by_bucket
                .get(bucket)
                .unwrap_or_else(|| panic!("{bucket} resolves no all-reduce at all"));
            // EXACTLY one all-reduce row per bucket: both would double the message, neither would
            // drop it.
            assert_eq!(rows.len(), 1, "{bucket}: {rows:?}");
            let (op_name, candidates) = &rows[0];
            assert_eq!(
                candidates,
                &vec![want.to_string()],
                "{bucket} ({op_name}) should bind `{want}`"
            );
        }

        // And the count is untouched: 2 per layer x 64 layers, whichever family runs.
        for rows in by_bucket.values() {
            let op = model.op(&rows[0].0).expect("model op");
            assert_eq!(op.count, 128);
        }
    }

    #[test]
    fn uniform16_manifest_declares_both_dispatches_and_keys_both_policies() {
        let source = include_str!("../../../examples/qwen38-27b-nvfp4-tp2-w16.toml");
        let manifest = PlanManifest::from_str(source, "uniform16").expect("wide plan");
        let model = crate::ModelSpec::qwen38_27b_dense(manifest.device.tp).unwrap();
        let device = crate::DeviceProfile::gfx1201_reference();
        let registry = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
        check_plugins_declared(&registry, &manifest, &model).unwrap();
        let resolved =
            crate::resolve(&manifest, &device, &model, &registry).expect("wide resolution");
        let requirements = resolved.requirements();
        let attention = requirements
            .iter()
            .find(|r| r.key.op == "attn_decode_paged")
            .unwrap();
        assert_eq!(
            attention.candidates,
            ["attn_decode_uniform16_h256_gqa6_fp8kv"]
        );
        let tables: BTreeMap<_, _> = saber_kernels::committed_rule_tables()
            .into_iter()
            .filter(|(file, _)| {
                matches!(*file, "saber_gfx1201.toml" | "saber_uniform16_gfx1201.toml")
            })
            .map(|(file, text)| (file, crate::hash::digest_of(&text)))
            .collect();
        assert_eq!(tables.len(), 2);
        assert_eq!(attention.key.policy_digest, crate::hash::digest_of(&tables));
        for required in ["saber_attn", "saber_attn16"] {
            let mut missing = manifest.clone();
            missing.plugins.retain(|plugin| plugin.name != required);
            let error = check_plugins_declared(&registry, &missing, &model).unwrap_err();
            assert!(error.to_string().contains(required), "{error}");
        }
    }

    #[test]
    fn negative_control_a_registry_naming_an_undeclared_plugin_is_refused() {
        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
        let manifest =
            PlanManifest::from_str(crate::reference::EXAMPLE_MANIFEST_TOML, "example").unwrap();
        // The example manifest declares `mxfp4` and `saber`, not the prebuilt stems, and the
        // MXFP4 rows implement `gemm_w4a8`, which this model DOES name — so they are bindable and
        // the refusal stands.
        let model = crate::ModelSpec::qwen38_27b_dense(2).expect("model");
        assert!(check_plugins_declared(&reg, &manifest, &model).is_err());
    }

    #[test]
    fn every_derived_row_is_implemented_and_nothing_claims_a_measurement() {
        let reg = registry_from_prebuilt(&saber_kernels::prebuilt::dir()).unwrap();
        for imp in reg.implementations() {
            assert_eq!(
                imp.promotion.state,
                crate::provenance::PromotionState::Implemented,
                "{} must not claim a measurement it does not have",
                imp.name
            );
        }
    }
}
