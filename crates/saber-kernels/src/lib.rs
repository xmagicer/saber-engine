//! saber-kernels — the kernel plane's *artifacts*: HIP sources under `hip/`, the `.hsaco` code
//! objects built from them, the manifest that declares each kernel's kernarg ABI, and the launch
//! rules that used to be `getenv` ladders inside the prod launcher.
//!
//! See docs/ARCHITECTURE.md ("Kernel plane", "Packaging") and docs/IMPLEMENTATION-PLAN.md §2 / P0
//! items 4-6. Three things are structural here:
//!
//! 1. **No host code in the artifact.** `hip/*.hip` is device code only: every tile parameter the
//!    prod launcher chose from an environment variable is an explicit `extern "C" __global__`
//!    symbol (`mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1`, ...). There is no `getenv` to reach.
//! 2. **The kernarg layout is declared and validated.** [`Manifest::validate_against_hsaco`]
//!    refuses on any disagreement between the manifest and the code object's own metadata. A
//!    kernarg layout the host and the kernel disagree about is memory corruption, not an error.
//! 3. **The launch decision is data.** [`rules::LaunchRules`] is a TOML table; [`rules::resolve`]
//!    is a pure function of (op, shape, rules) with no branch that is not a table row.
//!
//! Nothing in this crate links HIP or talks to a GPU: it runs on the host, in unit tests, without
//! a container. Loading and launching belong to `saber-hip` / `saber-exec`.

pub mod abi;
pub mod codeobj;
pub mod dflash;
pub mod flashnext;
pub mod fn19;
pub mod fn22;
pub mod fn33;
pub mod fn50;
pub mod fn6;
pub mod glue;
pub mod head;
pub mod ladder;
pub mod manifest;
pub mod moe;
pub mod msgpack;
pub mod noop;
pub mod p27d;
pub mod paro;
pub mod producer;
pub mod race_digest;
pub mod rules;
pub mod saber;
pub mod shapes;
pub mod vision;

pub use manifest::{ArgDecl, KernelDecl, Manifest};
pub use producer::AProducer;
pub use rules::{resolve, resolve_with_layout, ALayout, LaunchDescriptor, LaunchRules, Op, Shape};

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
    #[error("manifest does not match the code object:\n  {}", .0.join("\n  "))]
    Validate(Vec<String>),
    #[error("launch rule: {0}")]
    Rule(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Where the committed, hash-matched code objects live. One plugin = one `.hsaco` + one
/// `.manifest.toml`, both named after the kernel-set stem.
/// The committed launch-rule tables, verbatim, in a fixed order.
///
/// These are the tier's POLICY: which kernel, which tile, which split a shape takes. An autotune
/// record is a measurement made over the variant set the policy allowed on the day it ran, so a
/// plan whose policy has changed since is a plan whose records may be answering a question that is
/// no longer being asked -- which is why `saber-plan` folds this into the plan input hash rather
/// than trusting a record to still be about the same candidate set (docs/LEDGER.md, 2026-09-19).
///
/// **Every** table in `rules/`, and `the_policy_digest_covers_every_committed_rules_table` holds
/// this list against that directory. It had five of seven for a week: `attn_glue` and `dflash`
/// landed after the digest did and nobody added them, so an edit to either -- the rope band that
/// picks between two decompositions, 18 of the 36 non-GEMM routes the plan binds -- was invisible
/// to the plan input hash, and every record went on binding the variant the old policy chose. The
/// list is now derived from the directory by a test rather than by memory (docs/LEDGER.md, I17).
#[must_use]
pub fn committed_rule_tables() -> [(&'static str, &'static str); 12] {
    [
        ("mxfp4_gfx1201.toml", rules::PROD_RULES_TEXT),
        ("saber_gfx1201.toml", saber::PROD_RULES_TEXT),
        // C2's ragged decode attention, its own table so its edits re-key only its own route.
        ("saber_ragged_gfx1201.toml", saber::RAGGED_RULES_TEXT),
        ("saber_uniform16_gfx1201.toml", saber::UNIFORM16_RULES_TEXT),
        ("paro_gfx1201.toml", paro::PROD_RULES_TEXT),
        ("head_gfx1201.toml", head::PROD_RULES_TEXT),
        ("noop_gfx1201.toml", noop::PROD_RULES_TEXT),
        ("attn_glue_gfx1201.toml", glue::PROD_RULES_TEXT),
        ("dflash_gfx1201.toml", dflash::PROD_RULES_TEXT),
        ("vision_gfx1201.toml", vision::PROD_RULES_TEXT),
        // FN5: Flash-Next's HC kernels and bf16 GEMM routes, keyed only to `flashnext_hc`.
        ("flashnext_gfx1201.toml", flashnext::PROD_RULES_TEXT),
        // FN10: Flash-Next's attention split law + route forms and the grouped MTP-expert split
        // rows, keyed only to `flashnext_attn` / `flashnext_moe` (no 27B route re-keys).
        ("flashnext_attn_gfx1201.toml", fn6::PROD_ATTN_RULES_TEXT),
    ]
}

/// Which committed table carries **this plugin's** policy.
///
/// The policy digest exists so that editing a band re-keys the records measured under the old one.
/// Folding all seven tables into one digest makes that true and expensive: every record in the
/// cache is re-keyed by any edit, so a one-line change to `dflash` costs a full sweep of the 85
/// keys, 84 of which cannot have been affected by it. A record is re-keyed by the tables its own
/// route could bind, and this is the map that says which those are.
///
/// It is a prefix map, which is the shape of the thing being described: a code object's name
/// carries the reference translation unit it was sliced from, and the tables are named for the
/// same units. `plugin_table_map_covers_every_committed_plugin` holds it against
/// [`prebuilt::ALL`] and against each module's own `symbols()`/`plugin_for`, so a plugin that is
/// added without a table -- or moved between tables -- fails a host test rather than quietly
/// getting an empty policy.
#[must_use]
pub fn rule_table_for_plugin(plugin: &str) -> Option<&'static str> {
    // Longest-prefix first where two would match; none do today, and the test says so.
    Some(match plugin {
        "attn_glue" => "attn_glue_gfx1201.toml",
        "head" => "head_gfx1201.toml",
        "noop" => "noop_gfx1201.toml",
        // T6's shadow digest: an instrument with no launch policy (its geometry is the digest's
        // own definition, `race_digest::{BLOCK, PARTS, KV_PARTS}`), keyed to the null plugin's
        // instrument table so it re-keys nothing and no table was added for it.
        "race_digest" => "noop_gfx1201.toml",
        // P27-W2-2: the 27B drafter's device candidate seam (`--engine-device-step cand`): an
        // exact merge with no launch policy (one row a workgroup, `crate::p27d`), keyed to the
        // instrument table so adding it re-keys nothing.
        "p27_dstep" => "noop_gfx1201.toml",
        // Flash-Next's MoE family (FN4): no launch policy is BOUND this round -- no Flash-Next
        // planner exists yet, and the geometry is `saber_kernels::moe`'s declared law. Keyed to the
        // instrument table like the shadow digest, so adding it re-keys nothing of the 27B's; a
        // `moe_gfx1201.toml` arrives with the F2 planner that binds these symbols (adding a table
        // now would move the plan input hash of every 27B plan).
        "moe_grouped" | "moe_grouped_tp2" | "moe_grouped_ss" | "moe_grouped_far" | "moe_route" => {
            "noop_gfx1201.toml"
        }
        "paro_producers" => "paro_gfx1201.toml",
        "vision" => "vision_gfx1201.toml",
        "flashnext_hc" => "flashnext_gfx1201.toml",
        // FN6: the indexer / PLE / MTP glue. As FN4's MoE family: no Flash-Next planner binds it
        // yet (F2), its geometry is `saber_kernels::fn6`'s declared law, and keying it to the
        // instrument table re-keys nothing of the 27B's.
        "flashnext_glue" => "noop_gfx1201.toml",
        // FN9b: the exact-gate GDN scan. No policy row binds it this round (the Flash-Next layer
        // graph names it); keyed to the instrument table so it re-keys nothing of the 27B's.
        "flashnext_gdn" => "noop_gfx1201.toml",
        // FN24: the f16-carry GDN decode updates (the width-1-class verify window). No policy row
        // binds them: the Flash-Next executor names them by the verify class.
        "flashnext_gdn_fc" => "noop_gfx1201.toml",
        // FN19: the QSA selection kernels and the gather attention. No policy row binds them: their
        // geometry is `saber_kernels::fn19`'s declared law (the decode form reuses FN10's split rows
        // at the sparse context band), keyed to the instrument table so nothing is re-keyed.
        "flashnext_qsa" | "flashnext_qsa_attn" => "noop_gfx1201.toml",
        // FN22: the MTP drafter's device-side chain (argmax -> the next token's embedding row). No
        // policy row binds it: its geometry is `saber_kernels::fn22`'s declared law.
        "flashnext_draft" => "noop_gfx1201.toml",
        // FN57: the step's seam pick on the card (top-64 a row by the bf16 key law). No policy row
        // binds it: one workgroup a row is its declared geometry.
        "flashnext_pick" => "noop_gfx1201.toml",
        // FN119: the device-resident step's sampler + acceptance twin (`--fn-device-step check`).
        // No policy row binds it: one workgroup a sequence, one wave a row.
        "flashnext_dstep" => "noop_gfx1201.toml",
        // FN120: the persistent dataflow primitive (residency sweep, the ticket runtime, the
        // family-1 QSA body pair as dataflow tasks). A measurement object; no policy row binds it.
        "flashnext_df120" => "noop_gfx1201.toml",
        // FN120c: the GDN mixer tail as one persistent dataflow launch (`--fn-dataflow gdn`); the
        // executor names it, no policy row binds it.
        "flashnext_df120b" => "noop_gfx1201.toml",
        // FN119 round 2: the issue-ahead's device half (`--fn-device-step ahead`); the serve plane
        // names it, no policy row binds it.
        "flashnext_ahead" => "noop_gfx1201.toml",
        // FN119c: the b > 1 composition (the serve plane names it).
        "flashnext_ahead2" => "noop_gfx1201.toml",
        // FN119d: the next verify's embedding rows (the serve plane names it).
        "flashnext_ahead3" => "noop_gfx1201.toml",
        // LR-N: coupled drafting's chain tail (`--spec-couple on`; the serve plane names it).
        "flashnext_couple" => "noop_gfx1201.toml",
        // FN53c: the valve's device-side miss flag (one workgroup a layer, a host handshake). No
        // policy row binds it.
        "flashnext_xmiss" => "noop_gfx1201.toml",
        // FN114b: the flag-in-data exact reduce (`--fn-ar ll`); the executor names it.
        "flashnext_ar" => "noop_gfx1201.toml",
        // FN114d: the global-parity one-shot reduces (`--fn-ar gp`); the executor names them.
        "flashnext_ar_gp" => "noop_gfx1201.toml",
        // FN50: the MTP experts' mixed-table GEMMs (FN6's bodies, one skip line each). No policy
        // row binds them: the fp8 arm's geometry is FN6's (16-row tiles, the bf16 split rows).
        "flashnext_mtp8" => "noop_gfx1201.toml",
        // FN61c: the W8 trunk's output-stride cells (dflash_gemm's / FN5's bodies + an ldc). No
        // policy row binds them: the executor names them under `--fn-trunk w8`.
        "flashnext_trunk8" | "flashnext_trunk8p" | "flashnext_trunk8r" => "noop_gfx1201.toml",
        // FN108: the HC mix with the mixing matrices at fp8 (`--fn-trunk w8-hc8[-head]`): the
        // executor renames the planned v2 symbol `<sym>_h8`; no policy row binds them.
        "flashnext_hc8" => "noop_gfx1201.toml",
        // FN122b: the fp8-mixer HC down's PREFILL tiles (the shipped body at other row-block /
        // wave / slab geometry); the executor names them under `--fn-prefill-tiles hc`.
        "flashnext_hc8p" => "noop_gfx1201.toml",
        // FN131: the fp8-mixer HC down at the decode forms re-tiled by phase; the executor names
        // them under `--fn-hc-down split`.
        "flashnext_hc131" => "noop_gfx1201.toml",
        // FN141: the drained-ring twins; the executor names them under `--fn-k141`.
        "flashnext_hc141" | "flashnext_t8141" => "noop_gfx1201.toml",
        // FN164: the out / o W8 GEMM with the mixer-out all-reduce fused into its epilogue; the
        // executor names them under `--fn-mixer-ar-push`.
        "flashnext_t8far" => "noop_gfx1201.toml",
        // FN163: the parallel route / re-tiled front twins; the executor names them under
        // `--fn-route163`.
        "moe_route163" => "noop_gfx1201.toml",
        // FN148: the NVFP4 (FP4-Local) expert bodies + their oracle; no engine route yet (phase 3).
        "moe_fn148" => "noop_gfx1201.toml",
        // FN151: form (b) in the `_p1` class (tools/gen_fn151_nvfp4p1.py); no engine route yet.
        "moe_fn151" => "noop_gfx1201.toml",
        // FN152: FN151's body in the fused back halves (tools/gen_fn152_back.py).
        "moe_fn152" => "noop_gfx1201.toml",
        // FN153: form (b) in the grouped prefill tile (tools/gen_fn153_nvfp4pf.py); no engine route yet.
        "moe_fn153" => "noop_gfx1201.toml",
        // FN124: the fp8-KV QSA gather prefill with the wave-uniform masked sub-tile skip (+ the
        // twin's control and comparator); the executor names it under `--fn-prefill-tiles qsa`.
        "flashnext_qsa124" => "noop_gfx1201.toml",
        // FN128: the QSA selection re-tiled for the decode / verify forms (top-k, union, score);
        // the executor names them under `--fn-sel128`.
        "flashnext_qsa128" => "noop_gfx1201.toml",
        // ATTN-OPT-3: the attention-neighbour launch fusions (the QSA front, the combine + gate +
        // quant); the executor names them under `--fn-fuse f6` / `f7`.
        "flashnext_attnfuse" => "noop_gfx1201.toml",
        // FN68: the fp8-KV attention / gather / append objects (`--fn-kv-dtype fp8`). No policy
        // row binds them: the executor swaps the bf16kv symbol for its fp8kv twin by name.
        "flashnext_attn_kv8" | "flashnext_qsa_attn_kv8" | "flashnext_glue_kv8" => {
            "noop_gfx1201.toml"
        }
        // FN10: the attention's split law / route forms and the grouped MTP-expert split rows are
        // DATA in their own table, keyed to these two plugins only.
        "flashnext_attn" | "flashnext_moe" => "flashnext_attn_gfx1201.toml",
        // P27-15: the 27B's ancestor-mask tree verify (the WY GDN tree update + advance, the
        // ancestor-word decode attention). No policy row binds them: the tree arm names them.
        "p27_tree_gdn" | "p27_tree_attn" | "p27_tree_conv" | "p27_tree_kv" | "p27_tree_flush"
        | "p27_tree_gdn16" | "p27_tree_attn16" | "p27_tree_rows" | "p27_tree_advl"
        | "p27_tree_advl16" => "noop_gfx1201.toml",
        // P27-W2-5: the GDN decode update as base + stash + replay. No policy row binds it: the
        // `--engine-gdn-replay` arm names it.
        "p27w2_gdn_replay" => "noop_gfx1201.toml",
        // CMP27T: the replay's LDS-staged twin (`--engine-c4-arms replay`) and the TP > 1 argmax
        // that also writes its winner's logit (`--engine-c4-arms picks`). No policy row binds
        // either: the arm names them (the argmax keeps the head rules' geometry).
        "p27w2_gdn_replay_lds" | "saber_argmax_pv" => "noop_gfx1201.toml",
        // CMP27AH: the conv update fused into the replay update (`--engine-step-arms cuf`). No
        // policy row binds it: the arm names it.
        "p27w2_gdn_cuf" => "noop_gfx1201.toml",
        // CMP27X: the seam row's load-ahead twin (`--engine-step-arms devrow`); the head names it.
        "saber_device_row_ilp" => "noop_gfx1201.toml",
        // CMP27Y: the seam row split over workgroups (`--engine-step-arms rowsplit`).
        "saber_device_row_split" => "noop_gfx1201.toml",
        // CMP27AJ: the seam row's staged-term form (`--engine-step-arms rowstage`).
        "saber_device_row_stage" => "noop_gfx1201.toml",
        p if p.starts_with("mxfp4_") => "mxfp4_gfx1201.toml",
        "saber_attn_ragged" => "saber_ragged_gfx1201.toml",
        "saber_attn16" => "saber_uniform16_gfx1201.toml",
        p if p.starts_with("saber_") => "saber_gfx1201.toml",
        p if p.starts_with("dflash_") => "dflash_gfx1201.toml",
        // CMP27F: the fp8 head's one-pass 64-row tile is a route of the dflash head rules
        // (`dflash::resolve_head_w8_m64` reshapes the 32-row band's descriptor).
        "saber_head_w8_m64" => "dflash_gfx1201.toml",
        _ => return None,
    })
}

/// Additional executable objects needed by a registry entry's head object.
pub fn companion_plugins(plugin: &str) -> &'static [&'static str] {
    match plugin {
        // The W16 head writes partials for the uniform combine in saber_attn.
        "saber_attn16" => &["saber_attn"],
        _ => &[],
    }
}

pub mod prebuilt {
    use super::PathBuf;

    /// Every committed plugin, with one kernel name per plugin used as a smoke sample in tests.
    pub const ALL: &[(&str, &str)] = &[
        // Bounded exact shared-host transports; native P2P objects retain their own entries.
        (
            "saber_ar_host_exact.hsaco",
            "saber_ar_oneshot_2rank_exact_bf16",
        ),
        (
            "saber_ar_host_exact_chk.hsaco",
            "saber_ar_oneshot_2rank_exact_bf16_chk",
        ),
        (
            "mxfp4_gemm_decode_host_arnq.hsaco",
            "mxfp4_gemm_decode_arnq_bk128_ks4_tm1_wperm1_nt1",
        ),
        (
            "mxfp4_gemm_decode_host_arnq_chk.hsaco",
            "mxfp4_gemm_decode_arnq_bk128_ks4_tm1_wperm1_nt1_chk",
        ),
        (
            "mxfp4_gemm_decode.hsaco",
            "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1",
        ),
        (
            "mxfp4_gemm_folded.hsaco",
            "mxfp4_gemm_folded_tn4_wperm1_epifast1",
        ),
        (
            "mxfp4_gemm_atiled.hsaco",
            "mxfp4_gemm_atiled_tn4_lbk64_wperm1",
        ),
        ("mxfp4_gemm_base.hsaco", "mxfp4_gemm_base"),
        ("mxfp4_quant.hsaco", "add_rms_quant_t256_tiled0"),
        // saber (workstream F). One code object per reference translation unit; the two
        // attention kernels share one because the decode kernel calls device helpers the prefill
        // file defines, exactly as ref_attn_paged_h256_gqa6.hip does.
        (
            "saber_attn.hsaco",
            "saber_attn_prefill_h256_gqa6_fp8kv_qk8pv8",
        ),
        (
            "saber_attn16.hsaco",
            "saber_attn_decode_uniform16_h256_gqa6_fp8kv_nw6",
        ),
        // The RAGGED decode attention (workstream C2): rx6's decode kernel and combine with each
        // sequence's query length read from `cu_q` (the retired source generator RAGGED_SUBS). Its
        // own code object, so `saber_attn` keeps rx6's bytes.
        (
            "saber_attn_ragged.hsaco",
            "saber_attn_decode_ragged_h256_gqa6_fp8kv_nw3",
        ),
        (
            "saber_gdn_fused_update.hsaco",
            "saber_gdn_fused_update_w4k128v128_ab1",
        ),
        ("saber_gdn_norm.hsaco", "saber_gdn_gated_rmsnorm_h128_silu"),
        ("saber_ar_wht.hsaco", "saber_ar_oneshot_2rank_wht6_bf16"),
        ("saber_ar_exact.hsaco", "saber_ar_oneshot_2rank_exact_bf16"),
        // Paro rotation producers (workstream F, plugin 3). The PARO tier is prod's
        // rollback unit, so these are a serve path.
        ("paro_producers.hsaco", "paro_add_rms_rot_rot1_i80"),
        ("saber_gdn_conv.hsaco", "saber_gdn_conv_update_w4_h128_hpb1"),
        (
            "saber_gdn_recurrent_update.hsaco",
            "saber_gdn_recurrent_update_k128_v128_ab1_norm0_vs1",
        ),
        (
            "saber_gdn_kkt_solve.hsaco",
            "saber_gdn_kkt_solve_k128_c64_r3",
        ),
        (
            "saber_gdn_chunk_scan.hsaco",
            "saber_gdn_chunk_scan_k128_v128_c64",
        ),
        // The f16 STORED-state trio (workstream C1, `--gdn-state-dtype f16`): rx7's templated
        // decode updates at SABER_ST_F16 and the scan with its h0/ht accesses narrowed the same way.
        // Separate code objects, so the fp32 plugins above keep rx6's bytes.
        (
            "saber_gdn_recurrent_update_f16s.hsaco",
            "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1",
        ),
        (
            "saber_gdn_fused_update_f16s.hsaco",
            "saber_gdn_fused_update_f16s_w4k128v128_ab1",
        ),
        (
            "saber_gdn_chunk_scan_f16s.hsaco",
            "saber_gdn_chunk_scan_f16s_k128_v128_c64",
        ),
        // The null plugin (workstream H, part 5). An INSTRUMENT, not a serve path: two kernels
        // that do nothing, so `saber chain-bench` can price a dispatch with no kernel work and no
        // memory traffic against one that has both. It is in no model spec and no served
        // manifest, so it binds nothing and moves no autotune key.
        ("noop.hsaco", "saber_noop"),
        // T6's SHADOW DIGEST (`--engine-race-check shadow`): two instrument kernels the engine
        // enqueues after a proposal's last kernel to digest its stages on the card, the record
        // riding the proposal's own pinned read. A diagnostic, never a serving mode; in no model
        // spec and no registry row (`saber_plan::SEPARATELY_PLANNED`).
        ("race_digest.hsaco", "saber_race_digest_regions"),
        // P27-W2-2 (`--engine-device-step cand`): the 27B drafter's candidate seam merged on the
        // card through a P2P window (`crate::p27d`); an ARM, in no model spec and no registry row.
        ("p27_dstep.hsaco", "p27d_cand_seam"),
        // The vision tower (workstream D1, P5's parity milestone). Prod runs the tower in ATen
        // (hipBLASLt GEMMs, torch SDPA, LayerNorm), so there is no source to slice: the
        // elementwise ops are written to ATen's rounding points (bit-exact, pinned on the host by
        // `vision_preflight.py`) and the reductions are gated by accumulation class.
        ("vision.hsaco", "vis_gemm_bf16_bias"),
        // The head plugin (workstream P, phase P3). The two ends of a forward pass that prod has
        // no kernels for at all -- the reference server does them with `index_select` and `torch.argmax` -- plus
        // the row selection that keeps `lm_head` one GEMM wide instead of one per prompt token.
        ("head.hsaco", "saber_embed_bf16"),
        // The attention layer's glue (workstream F, plugin 4). NOT sliced from a reference .hip:
        // prod runs Triton and ATen for all three of these, so the device code is written from
        // The reference server's reference semantics and gated against what prod actually executes.
        ("attn_glue.hsaco", "qk_norm_rope_h256_rot64_mi1_red1"),
        // The fusion lever's second measurement (workstream F): silu_mul_quant folded into
        // down_proj's A staging. An EXPERIMENT -- see the ledger before binding it anywhere.
        (
            "mxfp4_gemm_decode_silu.hsaco",
            "mxfp4_gemm_decode_fused_silu_bk128_ks4_tm1_wperm1_nt1",
        ),
        // Fusion-map row 3 (ws-h38): the split decode GEMM whose last blocks also run the exact
        // two-rank all-reduce and the `_nq` epilogue. Its own code object so that the declared
        // `mxfp4_gemm_decode` keeps its bytes (and every autotune key its digest); bound by the
        // block planner's fold (`rules::gemm_decode_ar_nq`), never by a registry row.
        (
            "mxfp4_gemm_decode_arnq.hsaco",
            "mxfp4_gemm_decode_arnq_bk128_ks4_tm1_wperm1_nt1",
        ),
        // T4's RACE-CHECK twins: the served all-reduces (row 3 included) with
        // `hip/ar_race_check.h`'s payload / flag / counter checks inserted and nothing else
        // changed. Their own code objects, so every served object keeps its bytes; selected only
        // by `--engine-race-check` (`saber::Ar::race_check`), a diagnostic, never a serving mode.
        (
            "mxfp4_gemm_decode_arnq_chk.hsaco",
            "mxfp4_gemm_decode_arnq_bk128_ks4_tm1_wperm1_nt1_chk",
        ),
        (
            "saber_ar_exact_chk.hsaco",
            "saber_ar_oneshot_2rank_exact_bf16_chk",
        ),
        (
            "saber_ar_wht_chk.hsaco",
            "saber_ar_oneshot_2rank_wht6_bf16_chk",
        ),
        // CMP27S: the wave-parallel wht6 one-shot body and its race-check twin. Their own code
        // objects (`saber_ar_wht` keeps its bytes); bound by `saber::ar_wpar_route` under the `wpar`
        // knob (`--engine-ar-wpar`), never by a registry row.
        ("saber_ar_wht_wpar.hsaco", "saber_ar_wpar_2rank_wht6_bf16"),
        (
            "saber_ar_wht_wpar_chk.hsaco",
            "saber_ar_wpar_2rank_wht6_bf16_chk",
        ),
        // CMP27AE: the N 5120 dwn4 decode GEMM with the wht6 two-rank all-reduce as its tile
        // epilogue (CMP27AB's probe body). Its own code object (`mxfp4_gemm_decode_pf` keeps its
        // bytes); bound by the block planner's fold (`rules::gemm_decode_arw`) under
        // `--engine-ar-fuse`, never by a registry row. No race-check twin: under
        // `--engine-race-check` the fold stands down to the two launches.
        (
            "mxfp4_gemm_decode_arw.hsaco",
            "mxfp4_gemm_decode_arw_bk128_ks2_tm4_dwn4",
        ),
        // P27-1d: the SPLIT rotated all-reduce -- P27-1b's push / reduce halves and P27-1c's
        // fewer-blocks push (`_push_nb`), handshaken by a command-processor wait instead of a
        // spinning CU. No registry row binds it: `--ar-form split` (`saber_exec::overlap`) derives
        // it from the one-shot slice the rules resolved (`saber::ar_wht_split`). The shipped one-shot
        // object is untouched.
        (
            "saber_ar_wht_split.hsaco",
            "saber_ar_split_2rank_wht6_bf16_push_nb",
        ),
        // G5: the folded prefill tile at mid-M heights (320 / 160 / 192 rows). Its own code object
        // for row 3's reason: `mxfp4_gemm_folded` keeps its bytes, and no committed registry row
        // binds these symbols -- `[[folded_mt]]` does, under the `folded_mt` knob.
        (
            "mxfp4_gemm_folded_mt.hsaco",
            "mxfp4_gemm_folded_mt_tn2_wperm1_t5w4",
        ),
        // G6: the A-tiled tile at mid-M heights and the skinny-N body for the 48-wide gate. Its own
        // code object for row 3's reason: `mxfp4_gemm_atiled` keeps its bytes, and no committed
        // registry row binds these symbols -- `[[atiled_mt]]` / `[[skinny]]` do, under the
        // `atiled_mid` / `skinny` knobs.
        ("mxfp4_gemm_mid.hsaco", "mxfp4_gemm_skinny_n1t_w4_u8_wperm1"),
        // CMP27E: the b8 decode GEMMs' register-prefetch twins. Their own code object for row 3's
        // reason: `mxfp4_gemm_decode` keeps its bytes; bound by `rules::resolve` /
        // `rules::gemm_decode_pair` under the `decode_pf` knob, never by a registry row.
        (
            "mxfp4_gemm_decode_pf.hsaco",
            "mxfp4_gemm_decode_pf_bk128_ks2_tm4_wperm1_nt1",
        ),
        // CMP27F: the fp8 head's one-pass 64-row tile. Its own code object (`dflash_bgemm` keeps
        // its bytes); bound by `dflash::resolve_head_w8_m64` under the `head_m64` knob, never by a
        // rules row.
        ("saber_head_w8_m64.hsaco", "saber_head_w8_m64_lds_nw4_kc128"),
        // CMP27K: the drafter gate_up's one-weight-pass 64-row tile. Its own code object
        // (`dflash_gemm` keeps its bytes); bound by `dflash::resolve_blockscale_gateup` under the
        // `gateup_m64` knob, never by a rules row.
        (
            "dflash_gemm_m64.hsaco",
            "dflash_gemm_a8w8_bs128_m64n128k128_nw8",
        ),
        // CMP27Z: the drafter int2 head's one-weight-pass 64-row twin. Its own code object
        // (`dflash_head` keeps its bytes); bound by `dflash::resolve_draft_head_int2` under the
        // head's `m64` knob, never by a rules row.
        (
            "dflash_head_m64.hsaco",
            "dflash_draft_head_int2_m64n64x2g128_lx",
        ),
        // Fusion-map row 4 (ws-h40): the gate_up decode GEMM whose last M arriving tiles run
        // `silu_mul_quant` on one row each. Its own code object for row 3's reason; bound by the
        // block planner's fold (`rules::gemm_decode_smq`), never by a registry row.
        (
            "mxfp4_gemm_decode_smq.hsaco",
            "mxfp4_gemm_decode_smq_bk128_ks1_tm1_wperm1_nt1",
        ),
        // Fusion-map row 5 (ws-h42): the GDN fused update whose last M arriving workgroups run
        // `gdn_norm_quant` on one row each. Its own code object; bound by the block planner's
        // fold (`saber::gdn_fused_update_nq`), never by a registry row.
        ("saber_gdn_fnq.hsaco", "saber_gdn_fnq_f16s_w4k128v128_ab1"),
        // The DFlash2 drafter (workstream F, plugin 5). BOTH references are Triton -- the reference's
        // blockscale GEMM and the reference's int2 draft head -- so neither is a port: there is no HIP
        // source to slice, and each is written from the reference's semantics and gated bit-exact
        // against its output. The head and its rerank share one code object because they share the
        // weight format and are launched as a pair.
        ("dflash_gemm.hsaco", "dflash_gemm_a8w8_bs128_m16n128k128"),
        ("dflash_head.hsaco", "dflash_draft_head_int2_m16n64g128"),
        ("dflash_bgemm.hsaco", "dflash_bgemm_bf16_m16n16"),
        ("dflash_selector.hsaco", "dflash_selector_score_r256_k16"),
        ("dflash_conv.hsaco", "dflash_grouped_conv_t2g16"),
        ("dflash_attn.hsaco", "dflash_attn_h128_gqa4_bf16kv_t32"),
        ("dflash_kv.hsaco", "dflash_kv_append_flash_bf16"),
        ("dflash_topk.hsaco", "dflash_topk_bf16_k16"),
        ("dflash_mlp.hsaco", "dflash_silu_quant_g128"),
        ("dflash_walk.hsaco", "dflash_selector_walk_greedy"),
        // The drafter's CONTEXT-KV store. A separate code object from `dflash_mlp` because its
        // reference is a separate PROGRAM: `precompute_and_store_context_kv` invokes
        // `torch.ops._C.rms_norm` and `torch.ops._C.rotary_embedding` directly, so these two are
        // the C++ `upstream::` kernels while every other norm and rope on the drafter's path is
        // inductor-compiled Triton. The roundings differ, and so would the build flag: `attn_glue`
        // carries `-ffp-contract=off` for its Triton rope and the C++ rope contracts.
        ("dflash_ctx.hsaco", "dflash_rope_neox_k_inplace"),
        // The drafter's QUERY q/k norm + rope. A third code object for the norm-and-rope seam,
        // because there are three references: `attn_glue` is the TARGET's (GemmaRMSNorm, ATen's
        // reduction, a rope that does not contract, `-ffp-contract=off`), `dflash_ctx` is the
        // C++ `upstream::` pair the context store calls directly, and this one is what inductor
        // compiles for the drafter's own query rows -- a plain `RMSNorm`, Triton's reduction, no
        // bf16 round before the rope, and a rope that DOES contract.
        ("dflash_qk.hsaco", "dflash_qk_norm_rope_h128"),
        // Flash-Next's MoE family (workstream FN4, `saber_kernels::moe`). `moe_grouped` is DERIVED
        // from the certified decode GEMM and `silu_mul_quant` (tools/gen_moe_grouped.py) -- its own
        // code object for row 3's reason, so `mxfp4_gemm_decode` and `mxfp4_quant` keep their
        // bytes; `moe_route` is the router top-10 + expert-major sort, the weighted combine and the
        // A16 control arm, Saber's own arithmetic built `-ffp-contract=off`. Neither is in any 27B
        // spec or registry row (`saber_plan::SEPARATELY_PLANNED`): the Flash-Next layer planner
        // (F2) binds them.
        ("moe_grouped.hsaco", "moe_grouped_w4a8_bk128_ks1_tm1_g1"),
        // FN80b-1: the TP=2 half-width cells (the down half at DBK 64 for K 320, the half-width
        // fused back half) -- the same derived bodies at another slab, its own object so
        // `moe_grouped` keeps its bytes; named by `moe::ExpertShape::TpHalf` (nothing routes it).
        (
            "moe_grouped_tp2.hsaco",
            "moe_grouped_w4a8_bk64_ks1_tm1_g0_p1",
        ),
        // FN114c: the fused back half with the shared expert split across the EP ranks
        // (`--fn-shared-expert split`) -- FN26's back half with the shared tile at FN80b-1's half
        // shape, its own object so `moe_grouped` / `moe_grouped_tp2` keep their bytes.
        ("moe_grouped_ss.hsaco", "moe_back_ss_gu1_dn1_p1"),
        // FN137 Phase B (`--fn-moe-ar-push on`): FN80b-1's half-width back half with the MoE-out
        // all-reduce fused into its combine slices (generated by tools/gen_fn137b_far.py from the
        // committed `moe_grouped_tp2.hip`); its own object so `moe_grouped_tp2` keeps its bytes;
        // `_ctl` = the no-spin control, `_chk` = the race checker.
        ("moe_grouped_far.hsaco", "fn137_moe_back_h2_far_gu1_dn1_p1"),
        ("moe_route.hsaco", "moe_route_topk10_sort_e512"),
        // Flash-Next (FN5): the hyper-connection mix / combine kernels, the bf16 split-K band and
        // the no-bias LDS tile, the bf16-output norm twins. No 27B plan binds any of them.
        ("flashnext_hc.hsaco", "fn_hc_down_combine_mf1"),
        // Flash-Next (FN6, `saber_kernels::fn6`): the certified saber attention bodies at GQA 12
        // over bf16 K/V pages (generated by the retired source generator, its own code object so
        // `saber_attn` keeps its bytes); the bf16 KV append + indexer ring, the indexer's
        // compressed-key store, the PLE gate / dilated conv, the MTP norms and add, the bf16
        // SiluAndMul; the grouped MTP-expert GEMMs derived from FN5's split-K band and the
        // drafter's W8A8 body (tools/gen_fn6_grouped.py). No 27B plan binds any of them.
        ("flashnext_attn.hsaco", "fn_attn_prefill_h256_gqa12_bf16kv"),
        ("flashnext_glue.hsaco", "fn_idx_q_compress_h128"),
        ("flashnext_moe.hsaco", "fn_moe_bgemm_sk_m16_g1"),
        // Flash-Next (FN9b): the GDN chunk scan with EXACT gates (every decay weight one exponent
        // <= 0; generated by the retired source generator `SCAN_XG_SUBS` from the certified scan, whose
        // split-and-clamped midpoint form attenuates Flash-Next's > 160-nat chunks), f16 and fp32
        // stored state. Its own code object so both 27B scans keep their bytes; no 27B plan binds it.
        (
            "flashnext_gdn.hsaco",
            "fn_gdn_chunk_scan_xg_f16s_k128_v128_c64",
        ),
        // Flash-Next (FN24): rx7's f16-stored GDN decode updates with the carried state rounded to
        // f16 after each row of a verify window (generated by the retired source generator
        // `GDN_FC_*_SUBS`): the window == SPEC 0's store + reload. Its own code object so both 27B
        // f16s objects keep their bytes; no 27B plan binds it.
        (
            "flashnext_gdn_fc.hsaco",
            "fn_gdn_fused_update_fc_f16s_w4k128v128_ab1",
        ),
        // Flash-Next (FN19): the QSA selection (block scores, the exact top-512, the per-CTA union
        // key list; Saber's own arithmetic) and the gather attention (the reference's bodies with the key
        // axis re-pointed, generated by the retired source generator QSA_*_SUBS). No 27B plan binds them.
        ("flashnext_qsa.hsaco", "fn_qsa_topk512"),
        (
            "flashnext_qsa_attn.hsaco",
            "fn_qsa_attn_prefill_h256_gqa12_bf16kv",
        ),
        // Flash-Next (FN22): the MTP drafter's device-side chain -- the argmax over the rank's
        // vocabulary half, the cross-rank exchange payload (the candidate's embedding row + its key
        // and id as exact bf16 lanes, summed with -0.0 by the certified exact all-reduce) and the
        // pair's pick. Saber's own integer law; no 27B plan binds it.
        ("flashnext_draft.hsaco", "fn_draft_select"),
        // Flash-Next (FN50): the MTP experts' MIXED table (Qwen's first-party fp8 routed experts +
        // the bf16 shared expert) -- FN6's W8A8 and bf16 grouped bodies with one tile-skip line
        // each (tools/gen_fn50_mtp8.py); its own object so `flashnext_moe` keeps its bytes.
        ("flashnext_mtp8.hsaco", "fn_moe_a8w8_bs128_mx_g1"),
        // Flash-Next (FN57): the step's seam pick on the card -- each logits row's top-64 over the
        // rank's vocabulary half in the host seam's total order (the bf16 key law, lower index on
        // a tie). Saber's own integer law; no 27B plan binds it.
        ("flashnext_pick.hsaco", "fn_pick_top64_bf16"),
        // Flash-Next (FN119, `--fn-device-step check`): the host sampler's draw + F40's walk on the
        // card, op for op (glibc's expf ported, the noise's f32 rounding guarded); its own object.
        ("flashnext_dstep.hsaco", "fn119_draw_walk"),
        // Flash-Next (FN120, Phase A of the megakernel program): the persistent dataflow
        // primitive -- co-residency proof, atomic-ticket worker loop, dependency counters -- and
        // the QSA family-1 pair (attention gate + q8 -> the W8 o_proj tiles) as dataflow tasks over
        // the certified sources included verbatim. A measurement object; nothing launches it.
        ("flashnext_df120.hsaco", "fn120_df_run"),
        // Flash-Next (FN120c, Phase B, `--fn-dataflow gdn`): the b1 verify's GDN mixer tail --
        // merged projection, fused update + norm + quant, out projection -- as tasks of ONE
        // persistent launch over the certified sources included verbatim; its own object.
        ("flashnext_df120b.hsaco", "fn120_df_gdn_b1"),
        // Flash-Next (FN119 round 2, `--fn-device-step ahead`): the walk on both ranks over the P2P
        // window + the drafter round's acc-dependent inputs on the card; its own object.
        ("flashnext_ahead.hsaco", "fn119a_walk"),
        // Flash-Next (FN119c): the b > 1 issue-ahead's chain inputs composed per sequence from the
        // uniform variants (a copy + per-sequence word runs); its own object.
        ("flashnext_ahead2.hsaco", "fn119b_compose"),
        // Flash-Next (FN119d, `--fn-device-step next`): the next verify's input embedding rows
        // from the walk's result block + the chain's ids, gathered under the round; its own object.
        ("flashnext_ahead3.hsaco", "fn119d_vemb"),
        // Flash-Next (LR-N, `--spec-couple on`): coupled drafting -- the chain tail drafts a sampled
        // row with the target's own noise at the draft's verify position; its own object.
        ("flashnext_couple.hsaco", "fn_couple_acp"),
        // Flash-Next (FN53c): the valve's device-side miss flag -- a routed record that is not
        // resident raises the host mailbox and the kernel patches the layer's xptr tables with
        // the host's answer (the drive -> bounce reads). No 27B plan binds it. FN98 added
        // `fn_gate98` (the in-graph layer gate of `--fn-graph-form one`) to the same object.
        ("flashnext_xmiss.hsaco", "fn_xmiss_e256"),
        // Flash-Next (FN114b, `--fn-ar ll`): the TP = 2 exact reduce with the flag carried in the
        // data words (each word stored inverted, an all-zero dword = not landed, relaxed polls of
        // the data); the one-shot's vadd verbatim. Its own object so `saber_ar_exact` keeps its
        // bytes; `_chk` / `_chkctl` are its race checker and the checker's control.
        ("flashnext_ar.hsaco", "fn_ar_ll_exact_bf16"),
        // Flash-Next (FN114d, `--fn-ar gp`): the exact / wht6 one-shot bodies (an instantiation
        // table of the same headers as `saber_ar_exact` / `saber_ar_wht`) with the workspace slot
        // chosen by the call's global parity; its own object so those keep their bytes; `_chk` =
        // the checked bodies + the two ranks' parity assertion.
        (
            "flashnext_ar_gp.hsaco",
            "saber_ar_oneshot_2rank_exact_bf16_gp",
        ),
        // Flash-Next (FN61c): the W8 dense trunk's output-stride cells -- dflash_gemm's W8A8 body
        // and FN5's one-split bf16 band, each with the store's row stride an argument; its own
        // object so `dflash_gemm` / `flashnext_hc` keep their bytes.
        ("flashnext_trunk8.hsaco", "fn_t8_a8w8_bs128_ldc"),
        // Flash-Next (FN61e): the W8 trunk's multi-row tiles (decode / verify 17..64 rows,
        // prefill 64-row tiles, the bf16 suffix at 64 rows) -- each element the FN61c parent's
        // arithmetic; its own object so `dflash_gemm` / `flashnext_trunk8` keep their bytes.
        ("flashnext_trunk8p.hsaco", "fn_t8p_a8w8_p4w8_ldc"),
        // Flash-Next (FN68, `--fn-kv-dtype fp8`): FN6's / FN19's attention bodies over FP8 (e4m3)
        // K/V pages (KVP 0, generated by the retired source generator) and the e4m3 append; their own
        // objects so `flashnext_attn` / `flashnext_qsa_attn` / `flashnext_glue` keep their bytes.
        (
            "flashnext_attn_kv8.hsaco",
            "fn_attn_prefill_h256_gqa12_fp8kv",
        ),
        (
            "flashnext_qsa_attn_kv8.hsaco",
            "fn_qsa_attn_prefill_h256_gqa12_fp8kv",
        ),
        ("flashnext_glue_kv8.hsaco", "fn_kv_append_ring_h256_fp8"),
        // Flash-Next (FN124, `--fn-prefill-tiles qsa`): the fp8-KV QSA gather prefill body with a
        // wave-uniform skip of fully masked 16-key sub-tiles (the identity; the shared prefill template's
        // `Skip::Wave` policy, sattn_prefill.h);
        // its own object so `flashnext_qsa_attn_kv8` keeps its bytes.
        (
            "flashnext_qsa124.hsaco",
            "fn124_qsa_attn_prefill_h256_gqa12_fp8kv_skip",
        ),
        // Flash-Next (FN128, `--fn-sel128`): the QSA selection at the decode / verify forms -- the
        // exact top-512 with the keys in registers, the union by rank, the score at one thread a
        // group; each byte-equal to its `flashnext_qsa` parent. Its own object so the parent keeps
        // its bytes.
        ("flashnext_qsa128.hsaco", "fn128_qsa_topk512_k16"),
        // Flash-Next (ATTN-OPT-3, `--fn-fuse f6` / `f7`): the QSA layer's front (q/k norm + rope,
        // the indexer, the KV / ring append) as one launch and the split-KV combine + family 1's
        // gate + act quant as one launch; each byte-equal to its chain. Its own object so the
        // parents keep their bytes.
        ("flashnext_attnfuse.hsaco", "fn_attnfuse_front_h256_fp8"),
        // Flash-Next (FN61g): the drafter's exact rerank scored from the preshuffled W8 head
        // (each candidate column the W8 head GEMM's logit bit for bit); its own object so the W8
        // GEMMs keep their bytes.
        ("flashnext_trunk8r.hsaco", "fn_t8r_a8w8_gather_m16n64"),
        // Flash-Next (FN108): FN11 / FN21's HC mix bodies with the mixing matrices read as e4m3 +
        // f32 row-group scales (the reference's hc8m), dequantized to bf16 in the register (generated by
        // tools/gen_fn108_hc8.py; its own object so `flashnext_hc` keeps its bytes).
        ("flashnext_hc8.hsaco", "fn_hc_down_combine_v2_m1w3t1d_h8"),
        // Flash-Next (FN122b): the fp8-mixer HC down's PREFILL tiles -- FN108's down body at other
        // (MF, NW, T, KC, PF), byte-equal to the shipped m2w21t1_h8 (tools/gen_fn122_hc8p.py; its
        // own object so `flashnext_hc8` keeps its bytes).
        (
            "flashnext_hc8p.hsaco",
            "fn_hc_down_combine_v2_m8w21t1k64_h8",
        ),
        // Flash-Next (FN131): the fp8-mixer HC down at the decode forms re-tiled by phase -- tile
        // waves stream the weight ring while phase waves run the norm, byte-equal to the parent
        // (tools/gen_fn131_hc131.py; its own object so `flashnext_hc8` keeps its bytes).
        ("flashnext_hc131.hsaco", "fn131_down_combine_v2_m1w3t1d_h8"),
        // Flash-Next (FN141): the drained-ring twins (tools/gen_fn141_{hc141,t8141}.py; their own
        // objects so `flashnext_hc8` / `flashnext_trunk8` / `flashnext_hc` keep their bytes).
        ("flashnext_hc141.hsaco", "fn141_up_q8_v2_s1d_h8"),
        ("flashnext_t8141.hsaco", "fn141_t8_a8w8_n64_pf4_ldc"),
        // Flash-Next (FN164): the out / o W8 GEMM (fn111 n64 / FN141 ring / t8p d2w4 / d4w4) with
        // the mixer-out all-reduce fused into its epilogue (tools/gen_fn164_t8far.py from the
        // committed t8141 / trunk8p text; its own object so every parent keeps its bytes;
        // `_ctl` = the no-spin control, `_chk` = the race checker).
        (
            "flashnext_t8far.hsaco",
            "fn164_far_fn141_t8_a8w8_n64_pf4_ldc",
        ),
        // Flash-Next (FN163): the b8 route at 32 waves + the b1 front's register-rank top-10
        // (tools/capture/fn163/fn163_probe.py; its own object so `moe_route` keeps its bytes).
        ("moe_route163.hsaco", "fn163_route_nt1024"),
        // FN148: the NVFP4 expert bodies (tools/gen_fn148_nvfp4.py from `moe_grouped_tp2`'s decode
        // body; its own object so `moe_grouped*` keep their bytes) + the exact-dequant oracle.
        ("moe_fn148.hsaco", "fn148_a16_nvfp4_ref_g1"),
        // FN151: FN148 form (b) in the served `_p1` (PFD 1) class (tools/gen_fn151_nvfp4p1.py).
        ("moe_fn151.hsaco", "fn151_w4a8b_bk128_ks1_tm1_g1"),
        // FN152: FN151's NVFP4 `_p1` body in the fused back halves (tools/gen_fn152_back.py).
        ("moe_fn152.hsaco", "fn152_moe_back_h2_far_gu4_dn2"),
        // FN153: FN148 form (b) in the grouped prefill tile `pf_tm*` (tools/gen_fn153_nvfp4pf.py).
        ("moe_fn153.hsaco", "fn153_w4a8b_pf_tm4_g1"),
        // P27-15: the 27B's ancestor-mask tree verify -- the WY-form GDN tree update over a
        // 10-row tile + the accepted-path advance (Saber's own arithmetic), and the ragged decode
        // attention with a per-row ancestor word on the q_len tail (generated by
        // the retired source generator ANC_SUBS). Their own objects so every saber object keeps its
        // bytes; no plan binds them (the `--engine-spec-tree ancestor9` arm names them).
        (
            "p27_tree_gdn.hsaco",
            "p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs1_adv0",
        ),
        (
            "p27_tree_attn.hsaco",
            "p27_attn_decode_anc_h256_gqa6_fp8kv_nw4",
        ),
        // P27-15b: the tree conv window (the decode conv update gathering each row's older taps
        // along its parent path; the retired source generator TREE_CONV_SUBS).
        ("p27_tree_conv.hsaco", "p27_gdn_conv_tree_w4_h128_hpb8"),
        // P27-15b: the accepted path's K/V rows compacted to their depth slots (Saber's own HIP).
        ("p27_tree_kv.hsaco", "p27_tree_kv_compact"),
        // P27-W2-4 (R2): the accepted conv window moved to the chain's entries (Saber's own HIP).
        ("p27_tree_flush.hsaco", "p27_tree_conv_commit"),
        // CMP27L (Saber's own HIP, arms of `--engine-b1-arms`): the tree conv with its rows in
        // parallel, and the flush's advance for every GDN layer in one launch. Their own objects so
        // `p27_tree_conv` / `p27_tree_gdn` keep their bytes.
        ("p27_tree_rows.hsaco", "p27_gdn_conv_tree_rows_w4_h128_t10"),
        (
            "p27_tree_advl.hsaco",
            "p27_gdn_tree_advance_layers_f16s_k128_v128",
        ),
        (
            "p27_tree_advl16.hsaco",
            "p27_gdn_tree_advance_layers_t16_f16s_k128_v128",
        ),
        // E4-3 (the 16-row tree form): the 16-row tree objects -- the WY GDN tree at TG_TM 16 and
        // the anc decode body at 6 warps (96 rows = 16 x GQA 6). Their own objects: the 10-row
        // ones keep their bytes; the `--engine-spec-tree best16` arm names them.
        (
            "p27_tree_gdn16.hsaco",
            "p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs1_adv1",
        ),
        (
            "p27_tree_attn16.hsaco",
            "p27_attn_decode_anc16_h256_gqa6_fp8kv_nw6",
        ),
        // P27-W2-5: the GDN decode recurrent update with no per-candidate snapshot -- one base
        // state a (sequence, head) + a stash of the step's k / v / a / b, replayed by the next
        // step (Saber's own HIP over rx7's arithmetic). Its own object: every saber object keeps
        // its bytes.
        (
            "p27w2_gdn_replay.hsaco",
            "p27w2_gdn_replay_f16s_k128_v128_ab1_vs1",
        ),
        // CMP27T: the replay with the shared q / k rows, gates and v column staged once a
        // workgroup in LDS (`--engine-c4-arms replay`; the shipped replay's bytes). Its own
        // object: `p27w2_gdn_replay` keeps its bytes.
        (
            "p27w2_gdn_replay_lds.hsaco",
            "p27w2_gdn_replay_lds_f16s_k128_v128_ab1_vs1",
        ),
        // CMP27T: the TP > 1 local argmax + its winner's logit bits (`--engine-c4-arms picks`).
        // Its own object: `head` keeps its bytes.
        ("saber_argmax_pv.hsaco", "saber_argmax_pv_bf16"),
        // CMP27X: the sampler seam's row with each thread's loads issued ahead
        // (`--engine-step-arms devrow`). Its own object: `head` keeps its bytes.
        ("saber_device_row_ilp.hsaco", "saber_device_row_ilp_bf16"),
        // CMP27Y: the seam row over G workgroups (`--engine-step-arms rowsplit`; g4 / g8 in the
        // same object, unrouted).
        (
            "saber_device_row_split.hsaco",
            "saber_device_row_split_g2_bf16",
        ),
        // CMP27AJ: the seam row's staged-term form, 1,024 threads a row (`--engine-step-arms
        // rowstage`; t512 in the same object, unrouted).
        (
            "saber_device_row_stage.hsaco",
            "saber_device_row_stage_t1024_bf16",
        ),
        // CMP27AH: the batched GDN decode pair as one launch, race-free by a last-arriver counter
        // (`--engine-step-arms cuf`; the pair's bytes). Its own object: the conv and both replay
        // objects keep their bytes.
        (
            "p27w2_gdn_cuf.hsaco",
            "p27w2_gdn_cuf_f16s_k128_v128_ab1_vs1",
        ),
    ];

    pub const ARCH: &str = "gfx1201";

    /// `<repo>/prebuilt/gfx1201`.
    pub fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("prebuilt")
            .join(ARCH)
    }

    /// `<repo>/crates/saber-kernels/hip`.
    pub fn hip_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("hip")
    }
}
