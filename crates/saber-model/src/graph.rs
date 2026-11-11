//! The layer graph — one Qwen3.8-27B forward pass as **data**.
//!
//! This module is a description, not an executor. It answers three questions that the
//! planner, the loader and the launch-budget CI job each ask in a different form:
//!
//! 1. **What runs, in what order, at a given bucket?** [`Graph::ops_for_bucket`] returns the
//!    ordered op list for `(M, phase)` with the shape keys each op is selected on, in the
//!    same vocabulary the reference's `select(op, **geometry)` uses
//!    (the kernel inventory doc §1) and the same one `saber_plan::shape::ShapeKeys`
//!    carries.
//! 2. **Which bytes does each op read?** Every weight is bound by its **exact tier-pack
//!    record name**, with the rank-local shape, dtype and shard kind the pack actually holds
//!    (`docs/abi/PACK.md`). Nothing here is a guess: the unit tests at the bottom open the
//!    real `manifest.json` of `/mnt/nvme/saber-packs/qwen38-27b-nvfp4` and assert every bound
//!    name exists on rank 0 **and** rank 1 with the expected shape and dtype, and that every
//!    non-vision record is either bound or listed in [`UNUSED_RECORDS`].
//! 3. **How many launches is that?** [`Graph::launch_count_lower_bound`] sums the declared
//!    `n_launches` per op (structural rule 7: launch count per bucket is declared statically
//!    and asserted at capture). It is a *lower* bound by construction — ops no kernel plane
//!    holds declare no launch count and are reported separately.
//! 4. **Does a kernel for it exist yet?** [`ResolvedOp::ported`] and [`ResolvedOp::entry`] are
//!    **derived**, not listed: a saber op is put to `saber::Registry::select` over the
//!    committed `[[registry]]` rows, an MXFP4 op to `rules::resolve` over the committed bands,
//!    and the symbol either belongs to a plugin in `saber_kernels::prebuilt::ALL` or it does
//!    not. Porting a kernel flips the answer here with no edit to this file. That is not a
//!    hypothetical: saber parts 2 and 3 landed the GDN decode pair, the GDN prefill triple
//!    and the exact all-reduce, and this module reported them as ported without changing a
//!    line of its routing.
//!
//! # The thing this module exists to make expressible
//!
//! `docs/IMPLEMENTATION-PLAN.md`, P0 follow-up (a): *"Activation layout is a per-edge
//! property. Prod decides the A-tiled GEMM path by a runtime handshake between producer and
//! consumer keyed on `data_ptr`; the port collapsed it to one M threshold, which is right for
//! the fused producers but cannot say 'this edge is row-major because a different producer
//! wrote it.'"*
//!
//! So every edge between two ops carries a [`LayoutRule`], and every consumer declares the
//! set of [`Layout`]s it accepts. The fused producers ([`KernelOp::AddRmsQuant`],
//! [`KernelOp::SiluMulQuant`], [`KernelOp::GdnNormQuant`]) declare
//! [`LayoutRule::FusedActivationFp8`], which resolves to [`Layout::ATiledFp8`] at
//! `M >= `[`a_tiled_min_m`] and [`Layout::RowMajorFp8`] below it — and so does
//! [`KernelOp::AttnGateQuant`], which grew a `tiled0`/`tiled1` pair in the `attn_glue` plugin.
//! [`KernelOp::GdnNormQuant`] does **not**: it has exactly one instantiation, whose store is
//! `q += row * N`, so its edge is [`LayoutRule::Fixed`]`(`[`Layout::RowMajorFp8`]`)` at every M
//! and `gdn.out_proj` takes the row-major GEMM family at a prefill M where every other GEMM in
//! the same layer takes the A-tiled one. **This paragraph named `o_proj` for two weeks and was
//! wrong** — nothing read either declaration, so neither was wrong out loud, and at
//! `M >= 513` all 48 GDN layers read row-major bytes as WMMA fragments (P6).
//! [`Graph::check_layouts`] is what makes the edge checkable rather than hoped for,
//! [`ResolvedOp::gemm_family`] reads the answer off the **edge**, never off `M`, and
//! `saber_kernels::AProducer` is what both planes ask what a producer writes.
//!
//! # What this module is not
//!
//! No kernel selection, no tile or split-K choice, no autotune, no allocation. Where a launch
//! rule would have to be restated here it is named and delegated, because a second copy of a
//! rule is the bug class this whole engine exists to delete — and the few constants this
//! module does restate ([`AR_BITS`], [`AR_QUANT_MIN_BYTES`],
//! [`GDN_FUSED_MAX_ITEMS`], [`MAX_DECODE_QLEN`]) are pinned against the committed rule tables
//! by `the_saber_constants_match_the_committed_rules`.
//!
//! The graph also does not invent **per-call** facts. The reference's split is that a constraint is a
//! necessary condition on *geometry*, while strides, pitches, contiguity and buffer sizes stay
//! with the entry point because they come from the caller's tensors. So this module asks the
//! registry, which wants only geometry, and never `saber::resolve`, which wants strides it has
//! no business knowing.

use std::collections::BTreeMap;
use std::fmt;

use saber_kernels::glue::{GateShape, GlueRules, KvShape, QkShape};
use saber_kernels::rules::{LaunchRules, Partitions, NO_PARTITIONS};
use saber_kernels::saber::{self, Registry, SaberRules};
use saber_kernels::{prebuilt, Shape};

use crate::ModelError;

// =====================================================================================
// Model configuration
// =====================================================================================

/// The subset of `config.json` the forward pass is shaped by.
///
/// Values come from `/mnt/nvme/models/Qwen3.8-27B-NVFP4/config.json` (`text_config`);
/// [`ModelConfig::from_config_json`] parses that file and a unit test checks the constants in
/// [`ModelConfig::qwen38_27b`] against it whenever the checkpoint is present.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// `hidden_size`, 5120.
    pub hidden_size: u32,
    /// `num_hidden_layers`, 64.
    pub num_hidden_layers: u32,
    /// `full_attention_interval`, 4: every 4th layer is full attention, the rest are gated
    /// delta net. Layer `i` is attention iff `(i + 1) % interval == 0`, which puts them at
    /// 3, 7, … 63 — exactly what the checkpoint's `layer_types` array says.
    pub full_attention_interval: u32,
    /// `num_attention_heads`, 24.
    pub num_attention_heads: u32,
    /// `num_key_value_heads`, 4. GQA is 24/4 = 6.
    pub num_key_value_heads: u32,
    /// `head_dim`, 256.
    pub head_dim: u32,
    /// `attn_output_gate`, true: `q_proj` emits `2 * head_dim` per head — the query and a
    /// sigmoid gate applied to the attention output. It is why `q_proj` is `[6144, 5120]` at
    /// TP=2 rather than `[3072, 5120]`.
    pub attn_output_gate: bool,
    /// `intermediate_size`, 17408.
    pub intermediate_size: u32,
    /// `linear_num_key_heads`, 16.
    pub linear_num_key_heads: u32,
    /// `linear_num_value_heads`, 48.
    pub linear_num_value_heads: u32,
    /// `linear_key_head_dim`, 128.
    pub linear_key_head_dim: u32,
    /// `linear_value_head_dim`, 128.
    pub linear_value_head_dim: u32,
    /// `linear_conv_kernel_dim`, 4.
    pub linear_conv_kernel_dim: u32,
    /// `vocab_size`, 248320. The *embedding* size; the tokenizer's own vocabulary is smaller.
    pub vocab_size: u32,
    /// `rms_norm_eps`, 1e-6.
    pub rms_norm_eps: f32,
    /// `mtp_num_hidden_layers`, 1 — the MTP/drafter head is one decoder layer plus `fc`.
    pub mtp_num_hidden_layers: u32,
    /// Layers whose MLP is fp8 channel-quantized rather than NVFP4 block-quantized.
    ///
    /// `quantization_config.config_groups.group_0.targets` carries
    /// `re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$`, so the last eight
    /// layers' MLPs are `weight` + `weight_scale` while layers 0..=55 are `weight_packed` +
    /// `weight_scale` + `weight_global_scale` + `input_global_scale`. The record *names*
    /// differ, so this is not cosmetic: a graph that binds the wrong pair fails to load.
    pub fp8_mlp_layers: Vec<u32>,
    /// `--kv-cache-dtype fp8` on the prod unit; picks the `*_fp8kv` attention rows.
    pub kv_cache_fp8: bool,
}

impl ModelConfig {
    /// Qwen3.8-27B (`qwen3_5_text`), the tier `qwen_prod_nvfp4` serves.
    #[must_use]
    pub fn qwen38_27b() -> Self {
        Self {
            hidden_size: 5120,
            num_hidden_layers: 64,
            full_attention_interval: 4,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            attn_output_gate: true,
            intermediate_size: 17408,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            mtp_num_hidden_layers: 1,
            fp8_mlp_layers: (56..64).collect(),
            kv_cache_fp8: true,
        }
    }

    /// Parse a checkpoint `config.json`.
    ///
    /// Reads `text_config` for the geometry and `quantization_config` for the fp8 MLP layer
    /// set. Cross-checks `layer_types` against `full_attention_interval` and refuses a
    /// checkpoint where the two disagree — the layer kind decides which weights exist, so a
    /// silent disagreement is a load error waiting to happen.
    pub fn from_config_json(json: &str) -> Result<Self, ModelError> {
        let root: serde_json::Value = serde_json::from_str(json)?;
        let text = root
            .get("text_config")
            .ok_or_else(|| ModelError::Config("config.json has no text_config".into()))?;

        let u = |v: &serde_json::Value, key: &str| -> Result<u32, ModelError> {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|x| u32::try_from(x).ok())
                .ok_or_else(|| ModelError::Config(format!("text_config.{key} missing or not u32")))
        };

        let mut cfg = Self {
            hidden_size: u(text, "hidden_size")?,
            num_hidden_layers: u(text, "num_hidden_layers")?,
            full_attention_interval: u(text, "full_attention_interval")?,
            num_attention_heads: u(text, "num_attention_heads")?,
            num_key_value_heads: u(text, "num_key_value_heads")?,
            head_dim: u(text, "head_dim")?,
            attn_output_gate: text
                .get("attn_output_gate")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            intermediate_size: u(text, "intermediate_size")?,
            linear_num_key_heads: u(text, "linear_num_key_heads")?,
            linear_num_value_heads: u(text, "linear_num_value_heads")?,
            linear_key_head_dim: u(text, "linear_key_head_dim")?,
            linear_value_head_dim: u(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: u(text, "linear_conv_kernel_dim")?,
            vocab_size: u(text, "vocab_size")?,
            rms_norm_eps: text
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6) as f32,
            mtp_num_hidden_layers: u(text, "mtp_num_hidden_layers").unwrap_or(1),
            fp8_mlp_layers: fp8_mlp_layers_from_quant_config(&root),
            kv_cache_fp8: root
                .pointer("/quantization_config/kv_cache_scheme")
                .is_some(),
        };
        cfg.fp8_mlp_layers.sort_unstable();
        cfg.fp8_mlp_layers.dedup();

        if let Some(kinds) = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
        {
            if kinds.len() != cfg.num_hidden_layers as usize {
                return Err(ModelError::Config(format!(
                    "layer_types has {} entries, num_hidden_layers is {}",
                    kinds.len(),
                    cfg.num_hidden_layers
                )));
            }
            for (i, kind) in kinds.iter().enumerate() {
                let declared = kind.as_str().unwrap_or("");
                let derived = match cfg.layer_kind(i as u32) {
                    LayerKind::Attention => "full_attention",
                    LayerKind::Gdn => "linear_attention",
                };
                if declared != derived {
                    return Err(ModelError::Config(format!(
                        "layer {i}: layer_types says {declared:?} but full_attention_interval \
                         {} implies {derived:?}",
                        cfg.full_attention_interval
                    )));
                }
            }
        }
        Ok(cfg)
    }

    /// Whether layer `i` is full attention or gated delta net.
    #[must_use]
    pub fn layer_kind(&self, layer: u32) -> LayerKind {
        if self.full_attention_interval != 0 && (layer + 1) % self.full_attention_interval == 0 {
            LayerKind::Attention
        } else {
            LayerKind::Gdn
        }
    }

    /// Whether layer `i`'s MLP is stored fp8 (`weight`) rather than NVFP4 (`weight_packed`).
    #[must_use]
    pub fn mlp_is_fp8(&self, layer: u32) -> bool {
        self.fp8_mlp_layers.contains(&layer)
    }

    /// Full-attention layer count, 16.
    #[must_use]
    pub fn attention_layers(&self) -> u32 {
        (0..self.num_hidden_layers)
            .filter(|&i| self.layer_kind(i) == LayerKind::Attention)
            .count() as u32
    }

    /// Gated-delta-net layer count, 48.
    #[must_use]
    pub fn gdn_layers(&self) -> u32 {
        self.num_hidden_layers - self.attention_layers()
    }

    /// Grouped-query ratio, 6. Rank-invariant: both head counts divide by `tp`.
    #[must_use]
    pub fn gqa(&self) -> u32 {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Pull the fp8-MLP layer indices out of `quantization_config`.
///
/// The targets are compressed-tensors regexes such as
/// `re:.*layers\.(56|57|58)\.mlp\.(gate|up|down)_proj$`. Rather than depending on a regex
/// crate to parse a regex, this reads the alternation between `layers\.(` and `)` — which is
/// the only shape compressed-tensors emits for a layer list. An unrecognised target yields no
/// layers, and the manifest-binding test is what catches that.
fn fp8_mlp_layers_from_quant_config(root: &serde_json::Value) -> Vec<u32> {
    let mut out = Vec::new();
    let Some(groups) = root
        .pointer("/quantization_config/config_groups")
        .and_then(serde_json::Value::as_object)
    else {
        return out;
    };
    for group in groups.values() {
        if group.get("format").and_then(serde_json::Value::as_str) != Some("float-quantized") {
            continue;
        }
        let Some(targets) = group.get("targets").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for target in targets.iter().filter_map(serde_json::Value::as_str) {
            if !target.contains(".mlp.") && !target.contains("mlp\\.") {
                continue;
            }
            let Some(open) = target.find("layers\\.(") else {
                continue;
            };
            let rest = &target[open + "layers\\.(".len()..];
            let Some(close) = rest.find(')') else {
                continue;
            };
            for piece in rest[..close].split('|') {
                if let Ok(n) = piece.trim().parse::<u32>() {
                    out.push(n);
                }
            }
        }
    }
    out
}

/// Which attention a decoder layer uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated delta net (`linear_attn.*`), 48 of 64.
    Gdn,
    /// Full paged attention (`self_attn.*`), 16 of 64.
    Attention,
}

// =====================================================================================
// Tensor facts: dtypes, shard kinds, layouts
// =====================================================================================

/// Pack record dtype. Tags and names mirror `docs/abi/PACK.md` § Dtype tags and the
/// `DTYPES` table in `the internal reference harness`; [`Dtype::manifest_name`] is what
/// `manifest.json` prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    /// Unsigned 8-bit; the dtype of NVFP4 `weight_packed` at `layout_id 0`.
    U8,
    /// Signed 8-bit.
    I8,
    /// IEEE binary16.
    F16,
    /// bfloat16.
    Bf16,
    /// IEEE binary32.
    F32,
    /// OCP `e4m3` 8-bit float.
    F8E4M3,
    /// MX block-scale exponent.
    E8M0,
    /// Two `e2m1` values per byte, low nibble = even element. The device-final MXFP4 weight
    /// layouts carry their LOGICAL `[N, K]` shape and half that many bytes.
    Fp4E2m1X2,
}

impl Dtype {
    /// The string `manifest.json` uses.
    #[must_use]
    pub fn manifest_name(self) -> &'static str {
        match self {
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::F16 => "F16",
            Dtype::Bf16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::E8M0 => "E8M0",
            Dtype::Fp4E2m1X2 => "FP4_E2M1_X2",
        }
    }

    /// Bits per logical element.
    #[must_use]
    pub fn bits(self) -> u32 {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::E8M0 => 8,
            Dtype::F16 | Dtype::Bf16 => 16,
            Dtype::F32 => 32,
            Dtype::Fp4E2m1X2 => 4,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.manifest_name())
    }
}

/// How a record was split across ranks (`docs/abi/PACK.md` § Shard kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardKind {
    /// Identical bytes on every rank.
    Replicated,
    /// Column-parallel: split along dim 0.
    Dim0,
    /// Row-parallel: split along dim 1.
    Dim1,
    /// Dim 0 is a concatenation of segments, each split independently.
    Dim0Segments,
}

impl ShardKind {
    /// The string `manifest.json` uses.
    #[must_use]
    pub fn manifest_name(self) -> &'static str {
        match self {
            ShardKind::Replicated => "Replicated",
            ShardKind::Dim0 => "Dim0",
            ShardKind::Dim1 => "Dim1",
            ShardKind::Dim0Segments => "Dim0Segments",
        }
    }
}

impl fmt::Display for ShardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.manifest_name())
    }
}

/// The in-memory layout of a tensor **on one edge of the graph**.
///
/// This is the per-edge property P0 follow-up (a) requires. It is a property of the edge —
/// of what the producer actually wrote — and never re-derived by the consumer from `M`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layout {
    /// Per-token `e4m3` activations in ordinary row-major order, with the per-token scale
    /// vector beside them. What every quant producer writes below [`a_tiled_min_m`], and what a
    /// non-fused producer (`scaled_fp8_quant`, the traced quant path) writes at **every** M.
    RowMajorFp8,
    /// Per-token `e4m3` activations already in WMMA fragment-tile order, so the A-tiled
    /// prefill GEMM reads A straight into fragments and only the weights go through LDS
    /// (`docs/LEDGER.md`, "A-tiled prefill kernel": −12..−16% prefill GEMM at M ≥ 2048,
    /// bit-identical).
    ATiledFp8,
    /// Per-token int8 activations. The Paro/int4 W4A8 tier's A side; the NVFP4 tier does
    /// not use it, and it is named here so a plan that mixes tiers cannot express the mix by
    /// accident.
    Int8,
    /// `bfloat16`, row-major. Residual stream, GEMM outputs, GDN conv state.
    Bf16,
    /// fp32 recurrent state, caller-owned and paged: 48 heads × 128 × 128 × 4 B =
    /// 3.0 MiB per layer per sequence slot before the TP divide. Never a workspace, never
    /// coupled to the KV block size (`ARCHITECTURE.md`, "What exploration corrected" item 3).
    Fp32State,
    /// The SAME recurrent state STORED as IEEE f16 (workstream C1, `--gdn-state-dtype f16`,
    /// [`Graph::with_gdn_state_f16`]): half the bytes per slot, the same `[heads, v, k]` layout and
    /// the same ELEMENT strides. Every kernel that touches it accumulates in fp32 and narrows only
    /// what it stores (round-to-nearest-even; widening is exact), which is prod's
    /// `--mamba-ssm-cache-dtype float16` under reference rx7.
    F16State,
}

impl Layout {
    /// Short name for `explain`-style output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::RowMajorFp8 => "row_major_fp8",
            Layout::ATiledFp8 => "a_tiled_fp8",
            Layout::Int8 => "int8",
            Layout::Bf16 => "bf16",
            Layout::Fp32State => "fp32_state",
            Layout::F16State => "f16_state",
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `SABER_MXFP4_A_TILED_MIN_M`, from `crates/saber-kernels/rules/mxfp4_gfx1201.toml`
/// (`a_tiled_min_m = 513`): at and above this M the **fused** producers emit fragment-tiled
/// activations and the GEMM takes the A-direct prefill kernel.
///
/// **Read, not repeated.** This was a `const A_TILED_MIN_M = 513` here — one of four copies of
/// the number, each pinned to the rules file by a host test — until P6 showed what a copy costs:
/// L15's "raise `a_tiled_min_m` to 99999" control moved ONE of them, left the rest at 513, and so
/// swapped one layout mismatch for another instead of removing it. `saber_kernels::rules` owns
/// the number; this module owns what the bytes look like at it.
#[must_use]
pub fn a_tiled_min_m() -> u32 {
    saber_kernels::rules::a_tiled_min_m()
}

/// The layout rule of the `gdn.norm` -> `gdn.out_proj` edge: what `gdn_norm_quant` writes under
/// the committed mxfp4 rules. Row-major at every M is the reference's law (one instantiation);
/// with `gdn_norm_tiled` on, the producer switches at [`a_tiled_min_m`] like the other two (G2).
#[must_use]
pub fn gdn_norm_layout_rule() -> LayoutRule {
    match saber_kernels::producer::AProducer::GdnNormQuant
        .a_layout()
        .expect("the committed mxfp4 and glue rules are compiled in and parse")
    {
        saber_kernels::rules::ALayout::FollowsM => LayoutRule::FusedActivationFp8,
        saber_kernels::rules::ALayout::RowMajor => LayoutRule::Fixed(Layout::RowMajorFp8),
    }
}

/// How an edge's [`Layout`] is decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutRule {
    /// One layout at every M. A non-fused producer writes row-major whatever M is, which is
    /// the case the single-threshold model could not express.
    Fixed(Layout),
    /// A fused quant producer: [`Layout::ATiledFp8`] at `M >= `[`a_tiled_min_m`], else
    /// [`Layout::RowMajorFp8`].
    FusedActivationFp8,
}

impl LayoutRule {
    /// The layout this edge carries at row count `m`, under the COMMITTED layout law.
    #[must_use]
    pub fn resolve(self, m: u32) -> Layout {
        self.resolve_under(m, saber_kernels::rules::ATiledLaw::committed())
    }

    /// The layout this edge carries at row count `m` under `law` — the layout law of the mxfp4
    /// rules a plan is built with (G6: the mid band below `a_tiled_min_m` when `atiled_mid` is on).
    /// The planner asks this, with the law of the SAME rules its producers and GEMMs resolve
    /// against, so the edge, the producer and the consumer cannot disagree.
    #[must_use]
    pub fn resolve_under(self, m: u32, law: saber_kernels::rules::ATiledLaw) -> Layout {
        match self {
            LayoutRule::Fixed(l) => l,
            LayoutRule::FusedActivationFp8 => {
                if law.tiled(m) {
                    Layout::ATiledFp8
                } else {
                    Layout::RowMajorFp8
                }
            }
        }
    }
}

/// Which GEMM family an edge's layout forces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmFamily {
    /// The A-direct prefill kernel (`mxfp4_gemm_atiled`).
    ATiled,
    /// The decode / base / folded kernels, all of which read A row-major. Which of the three
    /// runs is an M-band and split-K decision owned by `saber_kernels::rules`.
    RowMajor,
}

impl GemmFamily {
    /// Short name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            GemmFamily::ATiled => "a_tiled",
            GemmFamily::RowMajor => "row_major",
        }
    }
}

// =====================================================================================
// Shape keys
// =====================================================================================

/// One geometry value. Mirrors `saber_plan::shape::ShapeValue`; every string this module
/// produces is a literal, so the borrowed form is enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShapeValue {
    /// A numeric parameter (`m`, `n`, `k`, `q_len`, `causal` as 0/1).
    Num(i64),
    /// A named parameter (`w_dtype = "mxfp4"`).
    Str(&'static str),
}

impl fmt::Display for ShapeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeValue::Num(v) => write!(f, "{v}"),
            ShapeValue::Str(s) => f.write_str(s),
        }
    }
}

impl From<u32> for ShapeValue {
    fn from(v: u32) -> Self {
        ShapeValue::Num(i64::from(v))
    }
}

impl From<i64> for ShapeValue {
    fn from(v: i64) -> Self {
        ShapeValue::Num(v)
    }
}

impl From<bool> for ShapeValue {
    fn from(v: bool) -> Self {
        ShapeValue::Num(i64::from(v))
    }
}

impl From<&'static str> for ShapeValue {
    fn from(v: &'static str) -> Self {
        ShapeValue::Str(v)
    }
}

/// An ordered geometry key set — what a constraint row is tested against and what an autotune
/// record is keyed on. Sorted, so every rendering of it is canonical.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShapeKeys(BTreeMap<&'static str, ShapeValue>);

impl ShapeKeys {
    /// An empty key set.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Builder-style insert.
    #[must_use]
    pub fn with(mut self, key: &'static str, value: impl Into<ShapeValue>) -> Self {
        self.0.insert(key, value.into());
        self
    }

    /// [`Self::with`] only when `cond` holds: a key that exists for some buckets and not others
    /// (C2's `ragged`, which only a row band states).
    #[must_use]
    pub fn with_if(self, cond: bool, key: &'static str, value: impl Into<ShapeValue>) -> Self {
        if cond {
            self.with(key, value)
        } else {
            self
        }
    }

    /// The value under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ShapeValue> {
        self.0.get(key).copied()
    }

    /// The numeric value under `key`.
    #[must_use]
    pub fn num(&self, key: &str) -> Option<i64> {
        match self.0.get(key) {
            Some(ShapeValue::Num(v)) => Some(*v),
            _ => None,
        }
    }

    /// Iterate the keys in canonical order.
    pub fn iter(&self) -> impl Iterator<Item = (&&'static str, &ShapeValue)> {
        self.0.iter()
    }

    /// Number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the key set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for ShapeKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (k, v) in &self.0 {
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            write!(f, "{k}={v}")?;
        }
        Ok(())
    }
}

// =====================================================================================
// Kernel-plane op ids
// =====================================================================================

/// Which plane **owns** an op — that is, which crate would be asked whether a kernel exists.
///
/// Ownership, not availability. An op's plane does not change when a kernel is ported; whether
/// a committed code object implements it is [`ResolvedOp::ported`], derived per bucket from
/// `saber_kernels`. Keeping the two apart is what stopped this enum from being a second,
/// hand-kept copy of the artifact list: saber parts 2 and 3 flipped five ops to `ported` and
/// this table did not move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    /// The MXFP4 W4A8 GEMM family and its fused quant producers, resolved by
    /// `saber_kernels::rules::resolve` (`prebuilt/gfx1201/mxfp4_*`).
    Mxfp4,
    /// saber: attention, GDN, all-reduce, selected by `saber_kernels::saber::Registry::select`
    /// (`prebuilt/gfx1201/saber_*`).
    Saber,
    /// The attention layer's glue: `qk_norm_rope`, `attn_gate_quant` and the paged K/V write,
    /// resolved by `saber_kernels::glue::resolve` (`prebuilt/gfx1201/attn_glue.hsaco`).
    Glue,
    /// The bf16 head (workstream N2): its norm in the head plugin (`saber_add_rms_bf16`) and its
    /// GEMM on the drafter's `dflash_bgemm` symbols (`saber_kernels::dflash::resolve_head_bgemm`).
    Head,
    /// No plane holds this op at all — there is no registry row and no launch rule to ask, so
    /// nothing can be ported. These are the elementwise ops and the bf16 drafter GEMM.
    None,
}

/// A kernel-plane op id — the key `select(op, **geometry)` is asked with.
///
/// The `_paged`, `gdn_*` and `allreduce` spellings are the reference's registry keys verbatim
/// (`crates/saber-kernels/rules/saber_gfx1201.toml`); `gemm_w4a8` and the three quant producers
/// are the names `saber_plan::model::ModelOp::op` and `saber_kernels::rules` use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelOp {
    /// W4A8 linear. One launch: the split-K reduce is fused into the kernel.
    GemmW4A8,
    /// A bf16 linear. The MTP head ships bf16 in this tier and there is no committed bf16
    /// GEMM (`gemm_bf16_nt_m16` exists in the reference with no caller and was not ported).
    GemmBf16,
    /// Fused residual-add + RMS norm + per-token `e4m3` quant.
    AddRmsQuant,
    /// Fused `silu(gate) * up` + per-token `e4m3` quant.
    SiluMulQuant,
    /// Per-head gated RMS norm + per-token `e4m3` quant, over the GDN core output gated by
    /// the `z` slice of the merged `in_proj` output.
    GdnNormQuant,
    /// Paged attention, prefill. One launch per run of equal-`q_len` requests.
    AttnPrefillPaged,
    /// Paged attention, decode. **Two** launches: the split-KV kernel and the combine.
    AttnDecodePaged,
    /// Conv update + grid-wide barrier + recurrent delta-rule update in one launch. Selected
    /// only when `num_seqs * value_heads_per_rank <= GDN_FUSED_MAX_ITEMS`.
    GdnFusedUpdate,
    /// Decode conv update (the first half of the pair that serves `num_seqs >= 2`).
    GdnConvUpdate,
    /// Decode recurrent update (the second half of that pair).
    GdnRecurrentUpdate,
    /// Prefill conv preparation.
    GdnConvPrep,
    /// Prefill KKT solve.
    GdnKktSolve,
    /// Prefill chunked scan.
    GdnChunkScan,
    /// **P27-15d: the tree verify's conv** (`p27_gdn_conv_tree_*`): the decode conv update with each
    /// row's older taps GATHERED along its parent path (`tsrc`) from the committed window (`hsrc`).
    GdnConvTree,
    /// **P27-15d: the tree verify's GDN update** (`p27_gdn_tree_*_adv1`): the WY-form ancestor-masked
    /// update from the ONE committed slot, the previous tree's accepted-row advance as its
    /// prologue, the record left for the next.
    GdnTreeUpdate,
    /// **P27-15d: the tree verify's attention** (`p27_attn_decode_anc_*` + the ragged combine): the
    /// ragged decode body with the q_len tail's causal test replaced by the row's ancestor word.
    AttnDecodeTree,
    /// Two-rank one-shot all-reduce.
    AllReduce,
    /// Vocab-parallel embedding lookup.
    Embedding,
    /// Per-head q/k RMS norm plus interleaved mrope, and — on the attention path — nothing
    /// else: the output gate is a separate producer.
    QkNormRope,
    /// The attention output gate applied to the core output, with the `e4m3` quant that feeds
    /// `o_proj`. Prod fuses this into a rotate+quant producer on the Paro tier; on the
    /// NVFP4 tier it is a non-fused producer, which is why its edge is row-major at every M.
    AttnGateQuant,
    /// Gather of the vocab-parallel logit shards.
    LogitsGather,
    /// The paged K/V write: `triton_reshape_and_cache_flash`'s job, one launch for both
    /// tensors. It is a separate op because it is a separate LAUNCH — prod's attention backend
    /// calls it before the attention kernel, and an engine that folded it into the attention op
    /// would be declaring a launch budget it does not submit.
    KvAppend,
    /// The bf16 head's final norm (workstream N2, [`Graph::with_bf16_head`]): residual add + Gemma
    /// RMS norm with a **bf16** output, the activation prod's bf16 `lm_head` reads.
    AddRmsBf16,
    /// The target's `lm_head` as a bf16 x bf16 GEMM over the checkpoint's fp8 head dequantized to
    /// bf16 at pack build (`SABER_NVFP4_LMHEAD=bf16`'s matrix), on the drafter's bf16 GEMM.
    GemmBf16Head,
    /// The target's `lm_head` as **W8A16** (workstream N4, [`Graph::with_fp8_head`]): the
    /// checkpoint's own e4m3 matrix and its per-channel f32 scale, expanded in register to
    /// [`KernelOp::GemmBf16Head`]'s bf16 weight bytes (`bf16(f32(w8) * scale)`, prod's loader
    /// arithmetic) before the same bf16 WMMA sequence -- half the head's DRAM bytes.
    GemmFp8Head,
}

impl KernelOp {
    /// The op id `select()` is asked with.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            KernelOp::GemmW4A8 => "gemm_w4a8",
            KernelOp::GemmBf16 => "gemm_bf16",
            KernelOp::AddRmsQuant => "add_rms_quant",
            KernelOp::SiluMulQuant => "silu_mul_quant",
            KernelOp::GdnNormQuant => "gdn_norm_quant",
            KernelOp::AttnPrefillPaged => "attn_prefill_paged",
            KernelOp::AttnDecodePaged => "attn_decode_paged",
            KernelOp::GdnFusedUpdate => "gdn_fused_update",
            KernelOp::GdnConvUpdate => "gdn_conv_update",
            KernelOp::GdnRecurrentUpdate => "gdn_recurrent_update",
            KernelOp::GdnConvPrep => "gdn_conv_prep",
            KernelOp::GdnKktSolve => "gdn_kkt_solve",
            KernelOp::GdnChunkScan => "gdn_chunk_scan",
            KernelOp::GdnConvTree => "gdn_conv_tree",
            KernelOp::GdnTreeUpdate => "gdn_tree_update",
            KernelOp::AttnDecodeTree => "attn_decode_tree",
            KernelOp::AllReduce => "allreduce",
            KernelOp::Embedding => "embedding",
            KernelOp::QkNormRope => "qk_norm_rope",
            KernelOp::AttnGateQuant => "attn_gate_quant",
            KernelOp::LogitsGather => "logits_gather",
            KernelOp::KvAppend => "kv_append",
            KernelOp::AddRmsBf16 => "add_rms_bf16",
            KernelOp::GemmBf16Head => "gemm_bf16_head",
            KernelOp::GemmFp8Head => "gemm_fp8_head",
        }
    }

    /// Which plane holds the implementation.
    #[must_use]
    pub fn plane(self) -> Plane {
        match self {
            KernelOp::GemmW4A8
            | KernelOp::AddRmsQuant
            | KernelOp::SiluMulQuant
            | KernelOp::GdnNormQuant => Plane::Mxfp4,
            KernelOp::AttnPrefillPaged
            | KernelOp::AttnDecodePaged
            | KernelOp::GdnFusedUpdate
            | KernelOp::GdnConvUpdate
            | KernelOp::GdnRecurrentUpdate
            | KernelOp::GdnConvPrep
            | KernelOp::GdnKktSolve
            | KernelOp::GdnChunkScan
            | KernelOp::GdnConvTree
            | KernelOp::GdnTreeUpdate
            | KernelOp::AttnDecodeTree
            | KernelOp::AllReduce => Plane::Saber,
            KernelOp::QkNormRope | KernelOp::AttnGateQuant | KernelOp::KvAppend => Plane::Glue,
            KernelOp::AddRmsBf16 | KernelOp::GemmBf16Head | KernelOp::GemmFp8Head => Plane::Head,
            KernelOp::GemmBf16 | KernelOp::Embedding | KernelOp::LogitsGather => Plane::None,
        }
    }

    /// Launches per invocation, when the op declares one.
    ///
    /// The numbers and their provenance:
    ///
    /// | op | launches | why |
    /// |---|---:|---|
    /// | every GEMM | 1 | the split-K reduce is fused, so a split is still one launch (`docs/LEDGER.md`, "the registry is derived from the artifacts") |
    /// | `add_rms_quant`, `silu_mul_quant`, `gdn_norm_quant` | 1 | one workgroup per row, one launch |
    /// | `attn_prefill_paged` | 1 | the kernel inventory doc § "`n_launches` per op" |
    /// | `attn_decode_paged` | **2** | the split-KV kernel plus the combine; the op declares the pair |
    /// | `gdn_fused_update` | 1 | conv + barrier + recurrent in one launch (§3.2) |
    /// | `gdn_conv_update` / `gdn_recurrent_update` | 1 each | the pair that serves `num_seqs >= 2`, so 2 per layer — `saber_kernels::saber::Op::GdnDecodePair` declares the same 2 for the pair as a whole |
    /// | `gdn_conv_prep` / `gdn_kkt_solve` / `gdn_chunk_scan` | 1 each | the prefill triple, 3 per layer |
    /// | `allreduce` | 1 | one one-shot kernel per message, exact or lossy |
    ///
    /// `declared_launch_counts_agree_with_the_kernel_plane` holds these against what
    /// `saber_kernels` declares, so the table cannot drift away from the descriptors.
    ///
    /// `None` means **no kernel plane holds the op** — there is no registry row and no launch
    /// rule to ask — so it declares no count rather than a guessed one. Those ops
    /// are what separates this bound from a prediction; [`LaunchBudget::unmodelled`] names
    /// them rather than guessing.
    #[must_use]
    pub fn n_launches(self) -> Option<u32> {
        match self {
            KernelOp::GemmW4A8
            | KernelOp::AddRmsQuant
            | KernelOp::SiluMulQuant
            | KernelOp::GdnNormQuant
            | KernelOp::AttnPrefillPaged
            | KernelOp::GdnFusedUpdate
            | KernelOp::GdnConvUpdate
            | KernelOp::GdnRecurrentUpdate
            | KernelOp::GdnConvPrep
            | KernelOp::GdnKktSolve
            | KernelOp::GdnChunkScan
            | KernelOp::GdnConvTree
            | KernelOp::GdnTreeUpdate
            | KernelOp::AllReduce
            // The vocab-parallel lookup became ONE dispatch when the head plugin landed
            // (workstream P, phase P3): `saber_embed_bf16`, one workgroup per token. It read
            // `None` here while it had no code object, and a forward pass then issued one more
            // dispatch than the graph declared -- which is exactly what structural rule 7 is
            // for, and how this was found.
            | KernelOp::Embedding
            // The attention layer's glue joined the count when the `attn_glue` plugin landed
            // (workstream F, plugin 4). All three are ONE launch each, which is the whole point
            // of the port: prod spends five launches on `qk_norm_rope` alone (two ATen norms,
            // the cos/sin gather, two rope chunks, the Triton mrope).
            | KernelOp::QkNormRope
            | KernelOp::AttnGateQuant
            | KernelOp::KvAppend
            // The bf16 head (N2): one norm launch and one GEMM launch, exactly the MXFP4 head's
            // two, so a bf16-head plan declares the same count per bucket.
            | KernelOp::AddRmsBf16
            | KernelOp::GemmBf16Head
            | KernelOp::GemmFp8Head => Some(1),
            KernelOp::AttnDecodePaged | KernelOp::AttnDecodeTree => Some(2),
            KernelOp::GemmBf16 | KernelOp::LogitsGather => None,
        }
    }
}

impl fmt::Display for KernelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

// =====================================================================================
// Weight bindings
// =====================================================================================

/// What a bound record is to the op that names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRole {
    /// An unpacked weight matrix (fp8 or bf16).
    Weight,
    /// NVFP4 `weight_packed`: two `e2m1` per byte.
    WeightPacked,
    /// Per-row (fp8 channel) or per-block (NVFP4) weight scale.
    WeightScale,
    /// NVFP4 per-tensor weight scale.
    WeightGlobalScale,
    /// NVFP4 static activation scale.
    InputGlobalScale,
    /// An RMS-norm gain vector.
    NormWeight,
    /// The GDN depthwise conv filter.
    ConvWeight,
    /// GDN `A_log`, one per value head.
    ALog,
    /// GDN `dt_bias`, one per value head.
    DtBias,
    /// The static fp8 KV-cache scale.
    KvScale,
    /// The embedding table.
    Embedding,
}

impl WeightRole {
    /// Short name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            WeightRole::Weight => "weight",
            WeightRole::WeightPacked => "weight_packed",
            WeightRole::WeightScale => "weight_scale",
            WeightRole::WeightGlobalScale => "weight_global_scale",
            WeightRole::InputGlobalScale => "input_global_scale",
            WeightRole::NormWeight => "norm_weight",
            WeightRole::ConvWeight => "conv_weight",
            WeightRole::ALog => "a_log",
            WeightRole::DtBias => "dt_bias",
            WeightRole::KvScale => "kv_scale",
            WeightRole::Embedding => "embedding",
        }
    }
}

/// Who reads a bound record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consumed {
    /// A kernel reads it on the serve path.
    Kernel,
    /// Only `tools/pack` reads it, when it builds the device-final layout. It is bound here
    /// so the pack is fully accounted for, and marked so that nobody looks for it in a
    /// kernarg list.
    PackBuild,
}

/// One weight, bound by its exact tier-pack record name.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// The record name, verbatim, as it appears in `manifest.json` and in the pack's name
    /// blob. Sharded records keep the same name on both ranks: the rank is the
    /// discriminator, not the name (`docs/abi/PACK.md` § Name blob).
    pub record: String,
    /// What this record is to the op.
    pub role: WeightRole,
    /// The dtype the pack holds.
    pub dtype: Dtype,
    /// The **rank-local** shape the pack holds.
    pub shape: Vec<u64>,
    /// How the record was split.
    pub shard: ShardKind,
    /// Who reads it.
    pub consumed: Consumed,
}

impl Binding {
    fn new(
        record: String,
        role: WeightRole,
        dtype: Dtype,
        shape: Vec<u64>,
        shard: ShardKind,
    ) -> Self {
        Self {
            record,
            role,
            dtype,
            shape,
            shard,
            consumed: Consumed::Kernel,
        }
    }

    fn pack_build(mut self) -> Self {
        self.consumed = Consumed::PackBuild;
        self
    }
}

/// A class of records the text-only forward pass does not read.
///
/// The partition is exact and a test enforces it: every record in the pack is either bound by
/// an op or matched here. "Unused" is a claim with a reason attached, never a silence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnusedRecords {
    /// Name prefix that identifies the class.
    pub prefix: &'static str,
    /// Why it is not read.
    pub reason: &'static str,
}

/// Every record class the text-only graph deliberately leaves unbound.
pub const UNUSED_RECORDS: &[UnusedRecords] = &[UnusedRecords {
    prefix: "model.visual.",
    reason: "vision tower (27 blocks + merger + patch embed + pos embed, 333 records, ~0.9 GB \
             per rank, Replicated in v0): the serve path is text-only until P5, where DeepStack \
             vision is a parity-milestone item. config.json sets language_model_only=false and \
             deepstack_visual_indexes=[], and the whole tower is in quantization_config.ignore, \
             so it is bf16 and untouched by the requant.",
}];

// =====================================================================================
// Edges
// =====================================================================================

/// A tensor produced by one op and read by others.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    /// Unique name inside the graph, e.g. `l12.gdn.out_q8`.
    pub name: String,
    /// How this edge's layout is decided. **This** is the per-edge property.
    pub rule: LayoutRule,
    /// One line on what the tensor is.
    pub note: &'static str,
}

impl Edge {
    fn new(name: impl Into<String>, rule: LayoutRule, note: &'static str) -> Self {
        Self {
            name: name.into(),
            rule,
            note,
        }
    }

    fn bf16(name: impl Into<String>, note: &'static str) -> Self {
        Edge::new(name, LayoutRule::Fixed(Layout::Bf16), note)
    }

    fn fused_q8(name: impl Into<String>, note: &'static str) -> Self {
        Edge::new(name, LayoutRule::FusedActivationFp8, note)
    }
}

/// A consumer's port: the edge it reads and the layouts it can read.
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    /// The edge name.
    pub edge: String,
    /// Layouts this consumer accepts. A resolved layout outside this set is a plan error, not
    /// a slow path — reading A-tiled bytes with the row-major kernel is garbage, not a
    /// regression.
    pub accepts: &'static [Layout],
}

/// Either fp8 activation layout: what every W4A8 GEMM can read, choosing its family from
/// which one the edge actually carries.
pub const ACCEPTS_FP8_ACTIVATION: &[Layout] = &[Layout::RowMajorFp8, Layout::ATiledFp8];
/// bf16 only.
pub const ACCEPTS_BF16: &[Layout] = &[Layout::Bf16];

impl Input {
    fn new(edge: impl Into<String>, accepts: &'static [Layout]) -> Self {
        Self {
            edge: edge.into(),
            accepts,
        }
    }
}

/// Which index table names the slot (or page) a state is read at, and therefore **which of a
/// session's state slots holds the value the NEXT step reads**.
///
/// A sequence owns `q` CONSECUTIVE state slots from its base (`docs/EXEC.md` § 2), and the two
/// slotted states in this graph do not live in the same one:
///
/// * the conv cache is addressed by `cache_idx`, which a decode step binds to the `sidx` tensor at
///   `ci_stride = q_len` — so `cache_idx[n * ci_stride]` is entry 0 of the sequence's block, i.e.
///   slot `base + 0`, whatever the step accepted;
/// * the recurrent state is addressed by `sidx` per ROW: the decode kernels write row `t`'s
///   snapshot into `sidx[n * stride + t]` and open the next step at
///   `sidx[n * stride + naccept - 1]` — so the live one is slot `base + naccept - 1`.
///
/// It is stated HERE, next to `bytes_per_slot`, because the two are one fact about a slot run and
/// a second derivation of either is how they drift. The park image is the first reader that needs
/// it: a session that moves has to publish each state into the slot the next step will read, and
/// before ws-k14 the image carried slot `base` for both (LEDGER, T2 item 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateIndexedBy {
    /// `cache_idx` — ONE slot per SEQUENCE: the run's base.
    CacheIdx,
    /// `sidx` — one snapshot per ROW: the live one is `base + naccept - 1`.
    Sidx,
    /// `block_table` / `slot` — the paged K/V cache. Pages from an allocator, not a slot run.
    BlockTable,
}

/// Caller-owned state an op reads and rewrites in place: KV pages, the GDN conv cache, the
/// GDN recurrent state. Not an edge — it has no producer inside one forward pass, it outlives
/// every launch, and it is paged by its own allocator.
#[derive(Clone, Debug, PartialEq)]
pub struct StateRef {
    /// Name, e.g. `l12.gdn.recurrent_state`.
    pub name: String,
    /// The layout of the stored bytes.
    pub layout: Layout,
    /// Bytes per sequence slot per rank, when the graph can state it.
    pub bytes_per_slot: Option<u64>,
    /// Which index table names this state's slot — and so which slot of a session's run holds the
    /// value the next step reads ([`StateIndexedBy`]).
    ///
    /// Paired with `bytes_per_slot`: `BlockTable` iff the graph states no slot size, which
    /// `every_state_pairs_its_slot_size_with_its_index_law` asserts over the whole graph.
    pub indexed_by: StateIndexedBy,
    /// One line on what it is.
    pub note: &'static str,
}

// =====================================================================================
// Geometry
// =====================================================================================

/// The per-op geometry that [`ShapeKeys`] are built from.
#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    /// A linear. `n` and `k` are rank-local.
    Gemm {
        /// Output features after sharding.
        n: u32,
        /// Input features after sharding.
        k: u32,
        /// Weight format the kernel sees.
        w_dtype: &'static str,
        /// Activation format the kernel sees.
        a_dtype: &'static str,
    },
    /// `add_rms_quant`: `m` rows of `k` columns.
    Rms {
        /// Columns.
        k: u32,
    },
    /// `silu_mul_quant`: `n` is **half** the gate_up width.
    SiluMul {
        /// Half width.
        n: u32,
    },
    /// `gdn_norm_quant`: `n` columns, `zs` the `z` row stride in elements, `channels` the
    /// head width the norm runs over.
    GdnNorm {
        /// Columns.
        n: u32,
        /// `z` row stride in elements — the merged `in_proj` changes it, which is why it is a
        /// selection key and not a constant.
        zs: u32,
        /// Head width (128).
        channels: u32,
    },
    /// Paged attention.
    Attn {
        /// 256.
        head_dim: u32,
        /// 6.
        gqa: u32,
        /// Paged block size, 16.
        block_size: u32,
        /// Query heads per rank.
        q_heads: u32,
        /// KV heads per rank.
        kv_heads: u32,
        /// `bf16` or `fp8_e4m3`.
        kv_dtype: &'static str,
    },
    /// The GDN core.
    GdnCore {
        /// Conv kernel width, 4.
        conv_width: u32,
        /// Key head dim, 128.
        head_k: u32,
        /// Value head dim, 128.
        head_v: u32,
        /// Value heads per rank.
        heads: u32,
        /// Key heads per rank.
        key_heads: u32,
        /// The prefill token block, 64. A selection key, not decoration: the `conv_prep`,
        /// `kkt_solve` and `chunk_scan` registry rows all carry `chunk == 64`, so a query
        /// without it is refused ("chunk not given") rather than defaulted.
        chunk: u32,
    },
    /// A tensor-parallel all-reduce over `elems_per_token * m` bf16 elements.
    AllReduce {
        /// Ranks.
        world_size: u32,
        /// Elements per row (the hidden size).
        elems_per_token: u32,
    },
    /// The vocab-parallel embedding lookup.
    Embedding {
        /// Rank-local vocabulary.
        vocab: u32,
        /// Hidden size.
        hidden: u32,
    },
    /// q/k norm plus mrope.
    QkNorm {
        /// 256.
        head_dim: u32,
        /// Query heads per rank.
        q_heads: u32,
        /// KV heads per rank.
        kv_heads: u32,
        /// Rotated dims per head: `partial_rotary_factor 0.25` of 256 = 64.
        rotary_dim: u32,
    },
    /// The attention output gate plus the quant that feeds `o_proj`.
    GateQuant {
        /// Gated width per rank.
        n: u32,
        /// Head width, 256 on this checkpoint. The kernel is compiled for it (the gate is
        /// read IN PLACE out of the projection's `[q|gate]` block at a `+head_dim` offset),
        /// so it is a selection key rather than a loop bound.
        head_dim: u32,
        /// Element stride between rows of the projection the gate is read out of, i.e. the
        /// merged `qkv` width. The gate is never materialised as a tensor of its own.
        gate_stride: u32,
    },
    /// A gather with no geometry worth a key beyond `m`.
    Gather {
        /// Rank-local width.
        n: u32,
    },
}

/// `SABER_AR_QUANT_MIN_KB`, 128 KiB: below it the exact all-reduce runs, at or above it
/// the lossy WHT one (the kernel inventory doc §3.4).
pub const AR_QUANT_MIN_BYTES: u64 = 128 * 1024;

/// `[ar] max_bytes` from `rules/saber_gfx1201.toml` (`SABER_AR_MAX_KB = 86016`), restated for the
/// same reason as [`AR_QUANT_MIN_BYTES`]: the MODEL is what knows the message size, the rules file
/// is what knows the slot. `the_saber_constants_match_the_committed_rules` holds the copy against its
/// source.
///
/// It is not a refusal any more, it is a CHUNK SIZE. A `tokens x hidden` reduce above it is cut
/// into row bands that each fit (`saber_kernels::saber::ar_bands`), which is why an all-reduce op's
/// declared launch count is a function of the bucket's rows and not the constant 1 it was until
/// H28.
pub const AR_MAX_BYTES: u64 = 88_080_384;
/// `SABER_AR_QBITS`, 6.
pub const AR_BITS: u32 = 6;
/// `SABER_GDN_FUSED_MAX_ITEMS`, 32: the fused GDN decode kernel runs only while
/// `num_seqs * value_heads_per_rank` is at or below it. At TP=2 that is `num_seqs == 1`.
pub const GDN_FUSED_MAX_ITEMS: u32 = 32;
/// `MAX_DECODE_QLEN = ATTN_MAX_DECODE_ROWS / GQA = 64 / 6 = 10`: above it the prefill
/// attention kernel runs even on a decode step.
pub const MAX_DECODE_QLEN: u32 = 10;

// =====================================================================================
// Buckets and phases
// =====================================================================================

/// Which half of the step an op list is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Chunked prefill.
    Prefill,
    /// Decode, with or without speculative width.
    Decode,
}

/// A capture bucket: the shape the whole step is planned and captured at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bucket {
    /// A prefill chunk.
    Prefill {
        /// Tokens in the chunk.
        tokens: u32,
        /// Sequences the chunk covers (1 for the P2 single-sequence path).
        seqs: u32,
    },
    /// A decode step.
    Decode {
        /// Sequences in flight.
        batch: u32,
        /// Query rows per sequence — `num_speculative_tokens + 1` under spec decode, 1 at
        /// SPEC=0.
        verify_width: u32,
    },
    /// **A decode ROW BAND (workstream C2): `batch` sequences at their OWN verify widths, packed
    /// into `rows` rows.**
    ///
    /// Sequence `i` feeds `w_i` rows (`1..=verify_width`, 0 for a padding slot) laid out back to
    /// back by the step's `cu` prefix, and the step runs at the smallest captured band `rows >=
    /// Σ w_i`; the rows past `cu[batch]` are the band's padding (slot −1, position 0, never a
    /// sequence's row). What a band shares with the rectangle `Decode { batch, verify_width }`:
    /// the sequence count, the MAX width (`q_len`, which is the `sidx` / conv-cache stride, the
    /// attention scratch stride and every GDN kernel's `maxq`) and therefore the GDN state
    /// arena. What differs: the row count the GEMMs, the head and the arena run at (`m`), and
    /// the attention route, which is the ragged pair (`saber_attn_ragged`) reading each sequence's
    /// length from `cu`. The op list does not depend on the width multiset — one captured graph
    /// per band, whatever the slots ask (`CLAUDE.md` rule 7).
    DecodeBand {
        /// Sequences in flight (the plan's batch; padding slots included).
        batch: u32,
        /// The MAX query rows per sequence: the plan's widest verify width.
        verify_width: u32,
        /// Rows the band runs, `<= batch * verify_width`.
        rows: u32,
    },
    /// **P27-15d: the b1 TREE VERIFY (`--engine-spec-tree ancestor9`)**: one sequence's `rows`
    /// verify rows (the anchor + the best-first tree's nodes, padded to the form's 10) related by
    /// ancestor words rather than by row order. Its GEMMs, head and arena run at `rows`; its GDN
    /// layers run the tree conv -> the WY tree update -> `gdn_norm_quant` and its attention the
    /// ancestor-word ragged pair ([`Selector::DecodeTree`]); every chain-only decode route
    /// ([`Selector::DecodeChain`], the fused / pair GDN updates) stays out. `band_rows()` is
    /// `Some(rows)`: the attention is the ragged form (one sequence: a band of one width).
    DecodeTree {
        /// Verify rows (the form's 10).
        rows: u32,
    },
}

impl Bucket {
    /// Total rows the GEMMs see.
    #[must_use]
    pub fn m(&self) -> u32 {
        match *self {
            Bucket::Prefill { tokens, .. } => tokens,
            Bucket::Decode {
                batch,
                verify_width,
            } => batch * verify_width,
            Bucket::DecodeBand { rows, .. } | Bucket::DecodeTree { rows } => rows,
        }
    }

    /// Whether this is the P27-15d tree verify bucket.
    #[must_use]
    pub fn is_tree(&self) -> bool {
        matches!(self, Bucket::DecodeTree { .. })
    }

    /// The band's row count when this is a row band (C2), `None` for a rectangle or a chunk.
    #[must_use]
    pub fn band_rows(&self) -> Option<u32> {
        match *self {
            Bucket::DecodeBand { rows, .. } | Bucket::DecodeTree { rows } => Some(rows),
            _ => None,
        }
    }

    /// Phase.
    #[must_use]
    pub fn phase(&self) -> Phase {
        match self {
            Bucket::Prefill { .. } => Phase::Prefill,
            Bucket::Decode { .. } | Bucket::DecodeBand { .. } | Bucket::DecodeTree { .. } => {
                Phase::Decode
            }
        }
    }

    /// Sequences in the bucket.
    #[must_use]
    pub fn num_seqs(&self) -> u32 {
        match *self {
            Bucket::Prefill { seqs, .. } => seqs.max(1),
            Bucket::Decode { batch, .. } | Bucket::DecodeBand { batch, .. } => batch,
            Bucket::DecodeTree { .. } => 1,
        }
    }

    /// Query rows per sequence.
    #[must_use]
    pub fn q_len(&self) -> u32 {
        match *self {
            Bucket::Prefill { tokens, seqs } => tokens / seqs.max(1),
            // A band's is its MAX width: the stride every per-sequence table is laid out at.
            Bucket::Decode { verify_width, .. } | Bucket::DecodeBand { verify_width, .. } => {
                verify_width
            }
            Bucket::DecodeTree { rows } => rows,
        }
    }

    /// A prefill bucket carrying SEVERAL sequences' chunks in one step.
    ///
    /// The L9 rule picks a bucket by TOTAL rows, so `n` sequences of `per_seq` tokens is one
    /// `n * per_seq`-row prefill step, not `n` steps. The `cu` prefix over the chunks is what
    /// tells every kernel where one sequence's rows end and the next begin, and the head runs on
    /// the LAST row of each chunk -- `sampled_rows` is `n`, not 1 and not `n * per_seq`.
    ///
    /// Equal chunks, because the row count has to divide: `q_len` is `tokens / seqs`, and every
    /// index law derives a sequence's span from it. A step mixing a 1024-token chunk with a
    /// 512-token one is a different shape and is not this one.
    #[must_use]
    pub fn prefill_batch(seqs: u32, per_seq: u32) -> Bucket {
        Bucket::Prefill {
            tokens: seqs.max(1) * per_seq,
            seqs: seqs.max(1),
        }
    }

    /// Whether this bucket's chunks divide evenly, which every index law assumes.
    ///
    /// A `Prefill { tokens: 1500, seqs: 2 }` answers `q_len` 750 and covers 1500 rows, but a
    /// caller that built it from two unequal chunks has a step whose `cu` prefix and whose row
    /// order disagree. Ask before planning rather than discovering it in a wrong token.
    #[must_use]
    pub fn chunks_divide(&self) -> bool {
        match *self {
            Bucket::Prefill { tokens, seqs } => {
                let n = seqs.max(1);
                tokens % n == 0 && tokens >= n
            }
            Bucket::Decode { .. } | Bucket::DecodeBand { .. } | Bucket::DecodeTree { .. } => true,
        }
    }

    /// Rows the **head** runs on: the positions a sampler will actually read logits for.
    ///
    /// This is not [`Self::m`] at prefill, and the difference is the whole point. A prefill chunk
    /// pushes `tokens` rows through every layer, but the sampler reads **one row per sequence** —
    /// the last one — so `lm_head` has no reason to compute the rest. At 2048 tokens the full
    /// product is 2048 x 248320 logits, 970 MiB, of which one row is read; prod does not compute
    /// it either, because `LogitsProcessor` prunes the hidden states to the sampled positions
    /// before the head GEMM.
    ///
    /// At decode every row IS a sampled position — under speculative decode each draft position
    /// needs its own logits to be verified against — so there the two agree and this returns `m`.
    ///
    /// The same rule is spelled `Rows::Sampled` in `saber_plan::ModelSpec`; the two are held
    /// against each other by `the_head_runs_on_the_sampled_rows_only`.
    #[must_use]
    pub fn sampled_rows(&self) -> u32 {
        match *self {
            Bucket::Prefill { seqs, .. } => seqs.max(1),
            Bucket::Decode {
                batch,
                verify_width,
            } => batch * verify_width,
            // Every row of a band is sampled: the padding rows' logits are computed and never
            // read, exactly like a rectangle's padding slots'.
            Bucket::DecodeBand { rows, .. } | Bucket::DecodeTree { rows } => rows,
        }
    }
}

/// When an op is in the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selector {
    /// Every bucket.
    Always,
    /// Prefill buckets only.
    Prefill,
    /// Decode buckets only.
    Decode,
    /// Decode, and `num_seqs * value_heads_per_rank <= GDN_FUSED_MAX_ITEMS`.
    DecodeGdnFused,
    /// Decode, and the fused route does not apply — the conv + recurrent pair.
    DecodeGdnPair,
    /// **P27-15d**: a CHAIN decode bucket only (every decode bucket but the tree's): the causal
    /// decode attention.
    DecodeChain,
    /// **P27-15d**: the tree verify bucket only ([`Bucket::DecodeTree`]).
    DecodeTree,
}

// =====================================================================================
// Ops
// =====================================================================================

/// Where in the model an op sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// The input embedding.
    Embedding,
    /// Inside decoder layer `n`.
    Layer(u32),
    /// The final norm and the language-model head.
    Head,
    /// The MTP / DFlash2 drafter head.
    Drafter,
}

/// One op in the forward pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    /// Unique name, e.g. `layer.12.gdn.in_proj_qkvz`. A plan override keys on it.
    pub name: String,
    /// Where it sits.
    pub site: Site,
    /// The kernel-plane op id it selects on.
    pub kernel: KernelOp,
    /// When it runs.
    pub selector: Selector,
    /// Edges it reads, with the layouts it accepts.
    pub inputs: Vec<Input>,
    /// Edges it writes.
    pub outputs: Vec<Edge>,
    /// Caller-owned state it reads and rewrites.
    pub state: Vec<StateRef>,
    /// Weights, by exact pack record name.
    pub weights: Vec<Binding>,
    /// The geometry its shape keys are built from.
    pub geometry: Geometry,
    /// One line of why it looks the way it does.
    pub note: &'static str,
}

impl Op {
    /// The launch count this op declares AT A ROW COUNT (structural rule 7).
    ///
    /// Every op but one declares a count that is a property of the kernel alone
    /// ([`KernelOp::n_launches`]). The all-reduce is the exception: its message is
    /// `rows x elems_per_token x 2` bytes and anything above the peer-visible IPC scratch slot is
    /// reduced as a SEQUENCE of row bands, one dispatch each — so the count is
    /// `ar_bands(...).len()`, known at plan time from the bucket and therefore still static in
    /// the sense rule 7 means. Below 8,601 rows at hidden 5120 it is 1, which is every decode
    /// step and every prefill chunk the engine runs.
    #[must_use]
    pub fn n_launches_at(&self, m: u32) -> Option<u32> {
        match (self.kernel, &self.geometry) {
            (
                KernelOp::AllReduce,
                &Geometry::AllReduce {
                    elems_per_token, ..
                },
            ) => saber_kernels::saber::ar_band_count(
                u64::from(m),
                u64::from(elems_per_token),
                // bf16: the same 2 `Geometry::AllReduce`'s own route key multiplies by.
                2,
                AR_MAX_BYTES,
            )
            .ok(),
            _ => self.kernel.n_launches(),
        }
    }

    /// Shape keys for this op at `bucket`, in the reference's `select()` vocabulary.
    #[must_use]
    pub fn shape_keys(&self, bucket: Bucket) -> ShapeKeys {
        let m = bucket.m();
        let keys = ShapeKeys::new();
        match self.geometry {
            Geometry::Gemm {
                n,
                k,
                w_dtype,
                a_dtype,
            } => keys
                .with("m", m)
                .with("n", n)
                .with("k", k)
                .with("w_dtype", w_dtype)
                .with("a_dtype", a_dtype),
            Geometry::Rms { k } => keys.with("m", m).with("k", k),
            Geometry::SiluMul { n } => keys.with("m", m).with("n", n),
            Geometry::GdnNorm { n, zs, channels } => keys
                .with("m", m)
                .with("n", n)
                .with("zs", zs)
                .with("channels", channels),
            Geometry::Attn {
                head_dim,
                gqa,
                block_size,
                q_heads,
                kv_heads,
                kv_dtype,
            } => keys
                .with("head_dim", head_dim)
                .with("gqa", gqa)
                .with("block_size", block_size)
                .with("causal", true)
                .with("q_dtype", "bf16")
                .with("kv_dtype", kv_dtype)
                .with("q_len", bucket.q_len())
                .with("num_seqs", bucket.num_seqs())
                .with("q_heads", q_heads)
                .with("kv_heads", kv_heads)
                // A row band's attention is the RAGGED pair (C2): the key its registry rows test
                // and no rectangle states.
                .with_if(bucket.band_rows().is_some(), "ragged", 1u32),
            Geometry::GdnCore {
                conv_width,
                head_k,
                head_v,
                heads,
                key_heads,
                chunk,
            } => keys
                .with("conv_width", conv_width)
                .with("head_k", head_k)
                .with("head_v", head_v)
                .with("heads", heads)
                .with("key_heads", key_heads)
                .with("chunk", chunk)
                .with("num_seqs", bucket.num_seqs())
                .with("q_len", bucket.q_len())
                // saber_gdn_fused_update sizes its conv window from both:
                // state_len_max = (width - 1) + (max_query_len - 1).
                .with(
                    "state_len_max",
                    conv_width - 1 + bucket.q_len().saturating_sub(1),
                )
                .with("a_dtype", "bf16"),
            Geometry::AllReduce {
                world_size,
                elems_per_token,
            } => {
                let numel = u64::from(m) * u64::from(elems_per_token);
                let bytes = numel * 2;
                let exact = bytes < AR_QUANT_MIN_BYTES;
                let keys = keys
                    .with("world_size", world_size)
                    .with("dtype", "bf16")
                    .with("numel", i64::try_from(numel).unwrap_or(i64::MAX))
                    .with("exact", exact);
                if exact {
                    keys
                } else {
                    keys.with("bits", AR_BITS)
                }
            }
            Geometry::Embedding { vocab, hidden } => keys
                .with("m", m)
                .with("vocab", vocab)
                .with("hidden", hidden),
            Geometry::QkNorm {
                head_dim,
                q_heads,
                kv_heads,
                rotary_dim,
            } => keys
                .with("m", m)
                .with("head_dim", head_dim)
                .with("q_heads", q_heads)
                .with("kv_heads", kv_heads)
                .with("rotary_dim", rotary_dim),
            Geometry::GateQuant {
                n,
                head_dim,
                gate_stride,
            } => keys
                .with("m", m)
                .with("n", n)
                .with("head_dim", head_dim)
                .with("gate_stride", gate_stride),
            Geometry::Gather { n } => keys.with("m", m).with("n", n),
        }
    }

    /// Whether this op runs in `bucket`.
    #[must_use]
    pub fn runs_in(&self, bucket: Bucket, gdn_value_heads_per_rank: u32) -> bool {
        self.runs_in_at(bucket, gdn_value_heads_per_rank, GDN_FUSED_MAX_ITEMS)
    }

    /// [`Op::runs_in`] at an explicit fused-update threshold (the graph's own).
    #[must_use]
    pub fn runs_in_at(
        &self,
        bucket: Bucket,
        gdn_value_heads_per_rank: u32,
        fused_max: u32,
    ) -> bool {
        let fused = bucket.num_seqs() * gdn_value_heads_per_rank <= fused_max;
        let tree = bucket.is_tree();
        match self.selector {
            Selector::Always => true,
            Selector::Prefill => bucket.phase() == Phase::Prefill,
            Selector::Decode => bucket.phase() == Phase::Decode,
            Selector::DecodeGdnFused => bucket.phase() == Phase::Decode && !tree && fused,
            Selector::DecodeGdnPair => bucket.phase() == Phase::Decode && !tree && !fused,
            Selector::DecodeChain => bucket.phase() == Phase::Decode && !tree,
            Selector::DecodeTree => tree,
        }
    }
}

/// An op with everything the bucket decided filled in.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedOp<'g> {
    /// The op.
    pub op: &'g Op,
    /// Its shape keys at this bucket.
    pub shape_keys: ShapeKeys,
    /// The resolved layout of every input edge, in `op.inputs` order.
    pub input_layouts: Vec<Layout>,
    /// The resolved layout of every output edge, in `op.outputs` order.
    pub output_layouts: Vec<Layout>,
    /// For a GEMM: which family the **input edge's layout** forces. `None` for anything else.
    ///
    /// Read off the EDGE, never off `M`. Where it disagrees with [`Self::entry`] — which the
    /// kernel plane names from `(m, n, k)` alone — the edge is right and the shape-only answer
    /// is the defect P0 follow-up (a) describes. [`Self::shape_only_family_disagrees`] says so.
    pub gemm_family: Option<GemmFamily>,
    /// What the **kernel plane** named for this geometry: the `[[registry]]` entry for a saber
    /// op, the kernel symbol for an MXFP4 one. `None` when no route exists, and then
    /// [`Self::refusal`] carries the kernel plane's own words.
    pub entry: Option<String>,
    /// The committed plugin that holds [`Self::entry`], from `saber_kernels::prebuilt::ALL`.
    pub plugin: Option<&'static str>,
    /// Why the kernel plane named nothing, in its own wording. `None` when a route was found
    /// **and** when the op never had one to look for (the elementwise ops).
    pub refusal: Option<String>,
    /// Whether a committed code object implements this route. Derived: there is a route
    /// through `saber_kernels` **and** the plugin it names is in `prebuilt::ALL`.
    pub ported: bool,
    /// Declared launches, or `None` when the op has no ported kernel.
    pub n_launches: Option<u32>,
}

impl ResolvedOp<'_> {
    /// The op's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.op.name
    }

    /// True when the kernel plane's shape-only answer names an A-tiled kernel for an edge that
    /// is row-major, or the other way round.
    ///
    /// This is not hypothetical and it is not a bug in `saber_kernels`: `rules::resolve` is
    /// given `(m, n, k)` and nothing else, so at a prefill `M` it must assume the fused
    /// producers wrote A-tiled bytes. For `gdn.out_proj`, whose producer is not fused, that
    /// assumption is wrong — and reading A-tiled bytes with the row-major kernel (or the
    /// reverse) is garbage, not a regression. The planner has to pass the edge's layout down;
    /// this flag is the graph saying where.
    #[must_use]
    pub fn shape_only_family_disagrees(&self) -> bool {
        let (Some(family), Some(entry)) = (self.gemm_family, self.entry.as_deref()) else {
            return false;
        };
        let named_atiled = entry.contains("atiled");
        named_atiled != (family == GemmFamily::ATiled)
    }
}

/// What the kernel plane answered for one op at one bucket.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct Route {
    entry: Option<String>,
    plugin: Option<&'static str>,
    refusal: Option<String>,
}

impl Route {
    /// An op no kernel plane holds. Not a refusal: nothing was asked.
    fn none() -> Self {
        Self::default()
    }

    /// Ask the reference's registry, exactly as `select(op, **geometry)` does, then turn the entry it
    /// names into the committed plugin that holds it.
    fn from_registry(op_id: &str, keys: &ShapeKeys, reg: &Registry<'_>) -> Self {
        let mut g = saber::Geometry::new();
        for (key, value) in keys.iter() {
            g = match value {
                ShapeValue::Num(v) => g.num(key, *v),
                ShapeValue::Str(v) => g.dtype(key, v),
            };
        }
        match reg.select(op_id, &g) {
            // The reference's table stores the entry WITHOUT the `saber_` prefix; the symbol carries it.
            Ok(row) => Self::from_symbol(row.entry.clone(), &format!("saber_{}", row.entry)),
            Err(reason) => Self {
                entry: None,
                plugin: None,
                refusal: Some(reason),
            },
        }
    }

    fn from_symbol(entry: String, symbol: &str) -> Self {
        match saber::plugin_for(symbol) {
            Ok(stem) => Self::from_plugin_name(entry, stem),
            Err(e) => Self {
                entry: Some(entry),
                plugin: None,
                refusal: Some(e.to_string()),
            },
        }
    }

    /// A plugin name is only a route if a code object for it is actually committed.
    fn from_plugin_name(entry: String, stem: &str) -> Self {
        let committed = prebuilt::ALL
            .iter()
            .find(|(hsaco, _)| hsaco.strip_suffix(".hsaco") == Some(stem));
        match committed {
            Some((hsaco, _)) => Self {
                entry: Some(entry),
                plugin: hsaco.strip_suffix(".hsaco"),
                refusal: None,
            },
            None => Self {
                entry: Some(entry),
                plugin: None,
                refusal: Some(format!(
                    "plugin `{stem}` is not in saber_kernels::prebuilt::ALL"
                )),
            },
        }
    }
}

// =====================================================================================
// Launch budget
// =====================================================================================

/// The statically declared launch count for a bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchBudget {
    /// The bucket it was computed for.
    pub bucket: Bucket,
    /// Sum of every declared `n_launches`. This is the number structural rule 7 asserts at
    /// capture, and a **lower** bound on what a step will actually submit.
    pub total: u32,
    /// Launches per kernel-plane op id.
    pub by_kernel: BTreeMap<&'static str, u32>,
    /// Ops with no declared launch count, by op id and how many of them there are. Every one
    /// of these is work the step still has to do; the difference between [`Self::total`] and a
    /// measured launch census lives here and in the drafter.
    pub unmodelled: BTreeMap<&'static str, u32>,
}

impl fmt::Display for LaunchBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "bucket {:?} (M={}): {} declared launches",
            self.bucket,
            self.bucket.m(),
            self.total
        )?;
        for (op, n) in &self.by_kernel {
            writeln!(f, "  {op:<22} {n:>5}")?;
        }
        if !self.unmodelled.is_empty() {
            writeln!(f, "  not modelled (no kernel plane holds these):")?;
            for (op, n) in &self.unmodelled {
                writeln!(f, "  {op:<22} {n:>5} op(s)")?;
            }
        }
        Ok(())
    }
}

// =====================================================================================
// Layout faults
// =====================================================================================

/// A producer/consumer layout disagreement, or an edge with no producer.
#[derive(Clone, Debug, PartialEq)]
pub enum LayoutFault {
    /// A consumer reads an edge nothing in this bucket writes.
    Unproduced {
        /// The consuming op.
        op: String,
        /// The edge it wanted.
        edge: String,
    },
    /// The edge's resolved layout is not one the consumer accepts.
    Mismatch {
        /// The consuming op.
        op: String,
        /// The edge.
        edge: String,
        /// Who wrote it.
        produced_by: String,
        /// What the producer's rule resolved to at this M.
        found: Layout,
        /// What the consumer can read.
        accepts: &'static [Layout],
    },
}

impl fmt::Display for LayoutFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutFault::Unproduced { op, edge } => {
                write!(
                    f,
                    "{op}: reads edge `{edge}`, which nothing produces in this bucket"
                )
            }
            LayoutFault::Mismatch {
                op,
                edge,
                produced_by,
                found,
                accepts,
            } => {
                let list: Vec<&str> = accepts.iter().map(|l| l.as_str()).collect();
                write!(
                    f,
                    "{op}: edge `{edge}` is {found} (written by {produced_by}), but this op \
                     accepts only [{}]",
                    list.join(", ")
                )
            }
        }
    }
}

// =====================================================================================
// The graph
// =====================================================================================

/// One Qwen3.8-27B forward pass, as data.
///
/// `PartialEq` is deliberately absent: the graph carries the kernel plane's parsed rule tables
/// so that [`Graph::ops_for`] can ask them whether a route exists, and those are data files,
/// not values worth comparing two graphs by.
#[derive(Clone, Debug)]
pub struct Graph {
    /// The config it was built from.
    pub config: ModelConfig,
    /// Tensor-parallel ranks the shapes are sharded for.
    pub tp: u32,
    /// The verify width [`Graph::ops_for_bucket`] assumes when it has only `M` to go on, and the
    /// width the GDN conv cache's slot size and the planner's strides are both derived from.
    /// [`VERIFY_WIDTH`]'s 5 is the bucket set in `examples/qwen38-27b-nvfp4-tp2.toml`;
    /// [`Graph::qwen38_27b_at_width`] builds at another (SPEC=7 is width **8**, not 7).
    ///
    /// **It is NOT `SABER_DRAFT_SCHEDULE`'s `8:5`**, which this comment said until ws-t1 and
    /// which is a different knob entirely: that schedule is a `batch:max_serial_MTP_forwards`
    /// carry-forward table for the **V1** proposer (`ref_draft.py::_batch_ceil`,
    /// [`crate::draft::DraftSchedule`]), and under `SPEC_METHOD=dflash` — the unit prod serves —
    /// nothing reads it at all. prod's verify width at concurrency comes from the scheduler-side
    /// acceptance-EMA cap (`patch_dynwidth.py`), which lands near 5 rather than being set to it.
    pub verify_width: u32,
    /// The target model's ops, in forward order.
    pub ops: Vec<Op>,
    /// The MTP / drafter head's ops, in forward order. Spec decode is P4; these are here so
    /// that the tier's `mtp.*` records are accounted for and the drafter/target pairing has
    /// somewhere to be checked.
    pub drafter: Vec<Op>,
    /// The committed MXFP4 launch bands (`crates/saber-kernels/rules/mxfp4_gfx1201.toml`).
    mxfp4: LaunchRules,
    /// The committed saber rules and `[[registry]]` rows
    /// (`crates/saber-kernels/rules/saber_gfx1201.toml`).
    saber: SaberRules,
    /// The committed attention-glue rules
    /// (`crates/saber-kernels/rules/attn_glue_gfx1201.toml`).
    glue: GlueRules,
    /// The fused-vs-pair GDN decode threshold this graph routes by (`nseq * H <= this` runs the
    /// fused update). [`GDN_FUSED_MAX_ITEMS`] by default; 0 under P27-W2-5's replay arm
    /// ([`Graph::with_gdn_pair_always`]), so every decode bucket runs the PAIR whose recurrent half
    /// has a replay form (the pair and the fused update are byte-identical: LEDGER 2026-09-19).
    gdn_fused_max_items: u32,
}

/// The glue shape an attention-glue op runs at, from the op's own geometry.
///
/// Written once, here, and called by BOTH readers — `Graph::route`, which reports what the plan
/// would launch, and `saber_exec::layer`, which launches it. A divergence between the two would
/// be a plan that resolves one kernel and a run that launches another. The paged strides are the cache layout the graph already states —
/// `[blocks, kv_heads, block_size, 2 * head_dim]` — and nothing else.
#[must_use]
pub fn glue_shape(
    op: &Op,
    m: u32,
) -> Option<(saber_kernels::glue::Op, saber_kernels::glue::Shape)> {
    use saber_kernels::glue::{Op as GOp, Shape as GShape};
    match (op.kernel, &op.geometry) {
        (
            KernelOp::QkNormRope,
            Geometry::QkNorm {
                head_dim,
                q_heads,
                kv_heads,
                ..
            },
        ) => Some((
            // The CONFIG's `mrope_interleaved`, which for this checkpoint is true. It is a
            // table value, not a shape: for a text-only request the three position rows are
            // equal and the choice cannot be observed in the output.
            GOp::QkNormRope {
                interleaved: true,
                flavour: saber_kernels::glue::QkFlavour::TARGET,
            },
            GShape::Qk(QkShape {
                m,
                nq: *q_heads,
                nkv: *kv_heads,
                head_dim: *head_dim,
                // `attn_output_gate` doubles the q block, then k, then v.
                qkv_n: 2 * q_heads * head_dim + 2 * kv_heads * head_dim,
            }),
        )),
        (
            KernelOp::AttnGateQuant,
            Geometry::GateQuant {
                n,
                head_dim,
                gate_stride,
            },
        ) => Some((
            GOp::AttnGateQuant,
            GShape::Gate(GateShape {
                m,
                n: *n,
                head_dim: *head_dim,
                gs: i64::from(*gate_stride),
            }),
        )),
        (
            KernelOp::KvAppend,
            Geometry::Attn {
                head_dim,
                block_size,
                kv_heads,
                q_heads,
                ..
            },
        ) => {
            let hd = i64::from(*head_dim);
            let nkv = i64::from(*kv_heads);
            let bs = i64::from(*block_size);
            Some((
                GOp::KvAppend,
                GShape::Kv(KvShape {
                    tokens: m,
                    nkv: *kv_heads,
                    head_dim: *head_dim,
                    block_size: *block_size,
                    // K is the COMPACT `[T, kv_heads, head_dim]` tensor `qk_norm_rope` wrote.
                    key_stride: nkv * hd,
                    // V is read IN PLACE out of the `[T, qkv_n]` projection, so its row stride
                    // is the projection's width, not the compact one. Passing the compact
                    // stride here would read the wrong rows and every address would still be
                    // inside the tensor.
                    value_stride: (2 * i64::from(*q_heads) + 2 * nkv) * hd,
                    // The paged cache is `[blocks, kv_heads, block_size, 2 * head_dim]`.
                    block_stride: nkv * bs * 2 * hd,
                    page_stride: 2 * hd,
                    head_stride: bs * 2 * hd,
                }),
            ))
        }
        _ => None,
    }
}

/// The widest speculative step this model plans for: `k` draft tokens plus the one the previous
/// step committed.
///
/// A MODEL constant rather than a per-bucket one, because the GDN conv cache's LAYOUT depends on it:
/// the cache is `[slot][qkv_n][conv_width - 1 + q - 1]`, one allocation serves every bucket, and a
/// buffer laid out for one width and indexed at another collides its own dimensions — silently,
/// since every dispatch still issues. So the graph publishes one width, `Dims` carries it, and both
/// the slot size here and the planner's strides (`saber_exec::layer::gdn_shape`) are derived from
/// it rather than from whichever bucket happens to be planning.
pub const VERIFY_WIDTH: u32 = 5;

/// Prod's declared decode bucket set, from `examples/qwen38-27b-nvfp4-tp2.toml`.
pub const PROD_DECODE_BUCKETS: &[Bucket] = &[
    Bucket::Decode {
        batch: 1,
        verify_width: 5,
    },
    Bucket::Decode {
        batch: 2,
        verify_width: 5,
    },
    Bucket::Decode {
        batch: 4,
        verify_width: 5,
    },
    Bucket::Decode {
        batch: 8,
        verify_width: 5,
    },
];

/// Prod's measured launch count per decode step at concurrency 8, for the comparison in
/// [`Graph::launch_count_lower_bound`]'s docs (`docs/LEDGER.md`, P0.5 step loop).
pub const PROD_LAUNCHES_PER_STEP_CONC8: u32 = 1084;

impl Graph {
    /// Build the graph for `config` sharded over `tp` ranks, at the default verify width.
    ///
    /// Errors when a dimension does not divide by `tp`. TP=3 needs the padded head layout
    /// (36/6/18/54 with the drafter at 48/12), which is a checkpoint-preparation decision:
    /// the graph will not invent it.
    pub fn qwen38_27b(config: ModelConfig, tp: u32) -> Result<Self, ModelError> {
        Self::qwen38_27b_at_width(config, tp, VERIFY_WIDTH)
    }

    /// The same graph at an explicit verify width — **the widest speculative step this model will
    /// be asked to run**, which is `1 + SPEC` and not `SPEC`.
    ///
    /// It is a build parameter rather than a constant because prod's DFlash2 block is
    /// `1 + num_speculative_tokens` rows: SPEC=5 is six rows and SPEC=7 is eight, and
    /// [`VERIFY_WIDTH`]'s 5 can express neither. It stays ONE number per graph for the reason the
    /// constant existed — the GDN conv cache is `[slot][qkv_n][conv_width - 1 + q - 1]`, one
    /// allocation serves every bucket of a graph, and the planner's strides
    /// (`saber_exec::layer::gdn_shape`) read this same field — so a caller that wants two widths
    /// asks for the WIDER one and plans both buckets under it, exactly as `1,5` does today.
    ///
    /// # Errors
    /// A zero width, or a width beyond the narrow workgroup without a matching
    /// uniform decode row in the kernel registry.
    pub fn qwen38_27b_at_width(
        config: ModelConfig,
        tp: u32,
        verify_width: u32,
    ) -> Result<Self, ModelError> {
        Self::build_at_width(config, tp, verify_width)
    }

    /// The same graph with prod's **bf16 `lm_head`** instead of Saber's MXFP4 one (workstream N2).
    ///
    /// Prod dequantizes the checkpoint's per-channel fp8 `lm_head` to bf16 at load
    /// (`SABER_NVFP4_LMHEAD=bf16`) and runs a bf16 x bf16 GEMM over the final norm's bf16 row;
    /// the default graph quantizes both sides (e4m3 activation, MXFP4 weight). This rewrites the
    /// two head ops and nothing else: `model.norm` becomes [`KernelOp::AddRmsBf16`] writing a
    /// bf16 `head.in`, and `lm_head` becomes [`KernelOp::GemmBf16Head`] reading it, whose one
    /// record `lm_head.w` is [`LAYOUT_BF16_FROM_FP8_CHANNEL`] — the drafter pack's
    /// `head.rerank.w` bytes, sharded the same way. A pack built from the default graph has no such
    /// record (its `lm_head.w` is MXFP4) and is refused by layout at bind.
    ///
    /// # Errors
    /// A graph without the two head ops.
    pub fn with_bf16_head(mut self) -> Result<Self, ModelError> {
        let norm = self
            .ops
            .iter_mut()
            .find(|o| o.name == "model.norm")
            .ok_or_else(|| ModelError::Config("the graph has no `model.norm`".into()))?;
        if norm.kernel != KernelOp::AddRmsQuant || norm.outputs.len() != 2 {
            return Err(ModelError::Config(
                "`model.norm` is not the fused add+norm+quant the bf16 head replaces".into(),
            ));
        }
        norm.kernel = KernelOp::AddRmsBf16;
        norm.outputs[0] = Edge::bf16("head.in", "bf16 input to the bf16 lm_head");
        norm.note = "final RMS norm, fused with the last residual add; bf16 out (the bf16 head)";
        let head = self
            .ops
            .iter_mut()
            .find(|o| o.name == "lm_head")
            .ok_or_else(|| ModelError::Config("the graph has no `lm_head`".into()))?;
        let Geometry::Gemm {
            w_dtype, a_dtype, ..
        } = &mut head.geometry
        else {
            return Err(ModelError::Config("`lm_head` is not a GEMM".into()));
        };
        *w_dtype = "bf16";
        *a_dtype = "bf16";
        head.kernel = KernelOp::GemmBf16Head;
        head.inputs = vec![Input::new("head.in", ACCEPTS_BF16)];
        head.note =
            "bf16 x bf16 over the checkpoint's fp8 head dequantized to bf16 (prod's matrix)";
        Ok(self)
    }

    /// The same graph with the **fp8-weight `lm_head`** (workstream N4, `W8A16`): the bf16 head's
    /// norm ([`KernelOp::AddRmsBf16`] writing a bf16 `head.in`) and [`KernelOp::GemmFp8Head`],
    /// which reads the checkpoint's e4m3 matrix RAW as `lm_head.w8` ([`LAYOUT_FP8_CHANNEL`]) and
    /// its per-channel scale as `lm_head.ws` ([`LAYOUT_F32_CHANNEL_SCALE`], f32 as prod's parameter
    /// holds it) and expands them in register to exactly the bf16 head's weight bytes. 0.636 GB
    /// per rank at TP=2 against the bf16 head's 1.271 (and MXFP4's 0.338); the drafter pack's
    /// `head.rerank.w` stays bf16 and is not shared with it.
    ///
    /// # Errors
    /// A graph without the two head ops.
    pub fn with_fp8_head(self) -> Result<Self, ModelError> {
        let mut g = self.with_bf16_head()?;
        let head = g
            .ops
            .iter_mut()
            .find(|o| o.name == "lm_head")
            .ok_or_else(|| ModelError::Config("the graph has no `lm_head`".into()))?;
        let Geometry::Gemm { w_dtype, .. } = &mut head.geometry else {
            return Err(ModelError::Config("`lm_head` is not a GEMM".into()));
        };
        *w_dtype = "e4m3";
        head.kernel = KernelOp::GemmFp8Head;
        head.note =
            "W8A16: the checkpoint's per-channel e4m3 head expanded in register to the bf16 \
                     head's bytes (prod's matrix), bf16 activation";
        Ok(g)
    }

    /// The same graph with the GDN recurrent state STORED as f16 (workstream C1).
    ///
    /// Rewrites every [`Layout::Fp32State`] [`StateRef`] — the `l<n>.gdn.recurrent_state` each of
    /// the decode updates and the prefill scan carries — to [`Layout::F16State`] at half the
    /// `bytes_per_slot`, and nothing else. The ops keep their kernels and their launch counts; the
    /// route reads the stored dtype off the op's own state (`saber_exec::layer`'s `gdn_shape`) and
    /// resolves the rules' `state_f16` rows, so the dtype has ONE source, here. Idempotent.
    ///
    /// The arena, the park image and the fork copy all size from `bytes_per_slot`, so each of them
    /// halves with it without a second statement of the width.
    ///
    /// # Errors
    /// A graph whose GDN layers carry no recurrent state (nothing to narrow is a refusal, not a
    /// silent no-op).
    pub fn with_gdn_state_f16(mut self) -> Result<Self, ModelError> {
        let mut n = 0usize;
        for st in self.ops.iter_mut().flat_map(|o| o.state.iter_mut()) {
            match st.layout {
                Layout::Fp32State => {
                    st.layout = Layout::F16State;
                    st.bytes_per_slot = st.bytes_per_slot.map(|b| b / 2);
                    st.note = "the delta-rule state STORED as f16 (C1): 0.75 MiB per layer per \
                               slot at TP=2; fp32 accumulation, only the store narrows";
                    n += 1;
                }
                Layout::F16State => n += 1,
                _ => {}
            }
        }
        if n == 0 {
            return Err(ModelError::Config(
                "--gdn-state-dtype f16 on a graph with no GDN recurrent state".into(),
            ));
        }
        Ok(self)
    }

    /// **P27-W2-5**: route every decode bucket's GDN update to the PAIR (conv update + recurrent
    /// update), the b1 bucket included, so the replay arm's recurrent half covers every decode form.
    #[must_use]
    pub fn with_gdn_pair_always(mut self) -> Self {
        self.gdn_fused_max_items = 0;
        self
    }

    /// Whether this graph routes every decode bucket's GDN update to the pair.
    #[must_use]
    pub fn gdn_pair_always(&self) -> bool {
        self.gdn_fused_max_items == 0
    }

    /// **P27-15d: the GDN conv cache laid out for a `rows`-row verify** (the tree form's 10) when
    /// that is wider than the graph's own verify width: every `*.gdn.conv_state` slot becomes
    /// `qkv_n x (conv_width - 1 + rows - 1)` bf16. The recurrent snapshots (`sidx`, one per chain
    /// row) are untouched -- the tree uses ONE committed slot -- so only the conv cache widens;
    /// [`Graph::conv_layout_rows`] is what every conv kernel's stride is derived from.
    ///
    /// # Errors
    /// A graph with no GDN conv cache, or a slot size that is not the layout law's.
    pub fn with_tree_conv_rows(mut self, rows: u32) -> Result<Self, ModelError> {
        let cw = self.config.linear_conv_kernel_dim;
        let old_len = u64::from(cw - 1 + self.verify_width.saturating_sub(1));
        let new_len = u64::from(cw - 1 + rows.max(self.verify_width).saturating_sub(1));
        let mut n = 0usize;
        for st in self.ops.iter_mut().flat_map(|o| o.state.iter_mut()) {
            if !st.name.ends_with(".gdn.conv_state") {
                continue;
            }
            let Some(b) = st.bytes_per_slot else {
                continue;
            };
            if b % (old_len * 2) != 0 {
                return Err(ModelError::Config(format!(
                    "{}: {b} B a slot is not qkv_n x {old_len} bf16",
                    st.name
                )));
            }
            st.bytes_per_slot = Some(b / old_len * new_len);
            n += 1;
        }
        if n == 0 {
            return Err(ModelError::Config(
                "the tree conv rows on a graph with no GDN conv cache".into(),
            ));
        }
        Ok(self)
    }

    /// The rows the GDN conv cache is laid out for: the graph's verify width, or the tree form's
    /// wider window ([`Graph::with_tree_conv_rows`]) -- read back from the cache's own slot size,
    /// so the layout has one source. `qkv_n` is the conv's channel count a rank.
    #[must_use]
    pub fn conv_layout_rows(&self, qkv_n: u32) -> u32 {
        let cw = u64::from(self.config.linear_conv_kernel_dim);
        self.ops
            .iter()
            .flat_map(|o| o.state.iter())
            .find(|st| st.name.ends_with(".gdn.conv_state"))
            .and_then(|st| st.bytes_per_slot)
            .map_or(self.verify_width, |b| {
                let len = b / (u64::from(qkv_n).max(1) * 2);
                u32::try_from((len + 2).saturating_sub(cw)).unwrap_or(u32::MAX)
            })
            .max(self.verify_width)
    }

    /// Whether this graph stores its GDN recurrent state as f16 ([`Graph::with_gdn_state_f16`]).
    #[must_use]
    pub fn gdn_state_f16(&self) -> bool {
        self.ops
            .iter()
            .flat_map(|o| o.state.iter())
            .any(|s| s.layout == Layout::F16State)
    }

    fn build_at_width(config: ModelConfig, tp: u32, verify_width: u32) -> Result<Self, ModelError> {
        if tp == 0 {
            return Err(ModelError::Config("tp must be >= 1".into()));
        }
        if verify_width == 0 {
            return Err(ModelError::Config(
                "a verify width of 0: every decode step feeds at least the token the last one \
                 committed, so the narrowest graph is width 1 (SPEC=0)"
                    .into(),
            ));
        }
        // The kernel plane's committed rule tables. They are `include_str!`ed TOML inside
        // saber-kernels, so this reads no files and can only fail if a rules file in the repo
        // stops parsing -- in which case failing here is exactly right.
        let mxfp4 = LaunchRules::prod_mxfp4()
            .map_err(|e| ModelError::Config(format!("mxfp4 launch rules: {e}")))?;
        let saber = SaberRules::prod_saber()
            .map_err(|e| ModelError::Config(format!("saber rules: {e}")))?;
        let glue =
            GlueRules::prod().map_err(|e| ModelError::Config(format!("attn_glue rules: {e}")))?;
        if config.num_key_value_heads == 0
            || config.num_attention_heads % config.num_key_value_heads != 0
        {
            return Err(ModelError::Config(
                "attention heads must be a whole multiple of nonzero KV heads".into(),
            ));
        }
        let gqa = config.gqa();
        let narrow_width = saber.knobs.attn_max_decode_rows / gqa.max(1);
        if verify_width > narrow_width || verify_width > MAX_DECODE_QLEN {
            // Admit a wider graph only when the kernel registry actually has a uniform
            // causal entry for its width, dtype and head geometry. Strides and scratch
            // remain per-call checks in the launch descriptor.
            let geometry = saber::Geometry::new()
                .num("q_len", i64::from(verify_width))
                .num("head_dim", i64::from(config.head_dim))
                .num("gqa", i64::from(gqa))
                .num("block_size", 16)
                .num("causal", 1)
                .dtype("q_dtype", "bf16")
                .dtype(
                    "kv_dtype",
                    if config.kv_cache_fp8 {
                        "fp8_e4m3"
                    } else {
                        "bf16"
                    },
                );
            Registry::new(&saber).select(KernelOp::AttnDecodePaged.id(), &geometry)
                .map_err(|why| ModelError::Config(format!(
                    "verify width {verify_width} (SPEC={}) exceeds the narrow attn_max_decode_rows = {} at GQA {gqa}; no uniform decode kernel matches: {why}",
                    verify_width - 1, saber.knobs.attn_max_decode_rows,
                )))?;
        }
        let d = Dims::new(&config, tp, &saber, verify_width)?;
        let mut ops = Vec::new();

        // ---- embedding -----------------------------------------------------------------
        // VocabParallelEmbedding: each rank holds vocab/tp rows, looks up the ids it owns,
        // zeroes the rest and all-reduces. That is one all-reduce per step that has nothing
        // to do with a layer, and it is easy to forget in a launch budget.
        ops.push(Op {
            name: "embed_tokens".into(),
            site: Site::Embedding,
            kernel: KernelOp::Embedding,
            selector: Selector::Always,
            inputs: vec![],
            outputs: vec![Edge::bf16(
                if tp > 1 { "embed.partial" } else { "hidden.0" },
                "this rank's slice of the embedding",
            )],
            state: vec![],
            weights: vec![Binding::new(
                "model.language_model.embed_tokens.weight".into(),
                WeightRole::Embedding,
                Dtype::Bf16,
                vec![u64::from(d.vocab_local), u64::from(d.hidden)],
                ShardKind::Dim0,
            )],
            geometry: Geometry::Embedding {
                vocab: d.vocab_local,
                hidden: d.hidden,
            },
            note: "vocab-parallel lookup; rows this rank does not own read as zero",
        });
        if tp > 1 {
            ops.push(all_reduce_op(
                "embed_tokens.all_reduce",
                Site::Embedding,
                Selector::Always,
                "embed.partial",
                Edge::bf16("hidden.0", "the residual stream entering layer 0"),
                &d,
                "VocabParallelEmbedding's reduce: without it every rank sees only its own vocab \
                 slice",
            ));
        }

        // ---- decoder layers ------------------------------------------------------------
        let mut hidden = "hidden.0".to_string();
        let mut residual: Option<String> = None;
        for layer in 0..config.num_hidden_layers {
            let (next_hidden, next_residual) =
                push_layer(&mut ops, &config, &d, layer, &hidden, residual.as_deref());
            hidden = next_hidden;
            residual = Some(next_residual);
        }

        // ---- head ----------------------------------------------------------------------
        let resid = residual.expect("at least one layer");
        ops.push(Op {
            name: "model.norm".into(),
            site: Site::Head,
            kernel: KernelOp::AddRmsQuant,
            selector: Selector::Always,
            inputs: vec![
                Input::new(hidden.clone(), ACCEPTS_BF16),
                Input::new(resid.clone(), ACCEPTS_BF16),
            ],
            outputs: vec![
                Edge::fused_q8("head.in_q8", "quantized input to lm_head"),
                Edge::bf16(
                    "head.resid",
                    "residual, consumed by nothing (kept for symmetry)",
                ),
            ],
            state: vec![],
            weights: vec![Binding::new(
                "model.language_model.norm.weight".into(),
                WeightRole::NormWeight,
                Dtype::Bf16,
                vec![u64::from(d.hidden)],
                ShardKind::Replicated,
            )],
            geometry: Geometry::Rms { k: d.hidden },
            note: "final RMS norm, fused with the last residual add and the lm_head quant",
        });
        ops.push(Op {
            name: "lm_head".into(),
            site: Site::Head,
            kernel: KernelOp::GemmW4A8,
            selector: Selector::Always,
            inputs: vec![Input::new("head.in_q8", ACCEPTS_FP8_ACTIVATION)],
            outputs: vec![Edge::bf16(
                "logits.partial",
                "this rank's vocab slice of the logits",
            )],
            state: vec![],
            weights: vec![
                Binding::new(
                    "lm_head.weight".into(),
                    WeightRole::Weight,
                    Dtype::F8E4M3,
                    vec![u64::from(d.vocab_local), u64::from(d.hidden)],
                    ShardKind::Dim0,
                ),
                Binding::new(
                    "lm_head.weight_scale".into(),
                    WeightRole::WeightScale,
                    Dtype::Bf16,
                    vec![u64::from(d.vocab_local), 1],
                    ShardKind::Dim0,
                ),
            ],
            geometry: Geometry::Gemm {
                n: d.vocab_local,
                k: d.hidden,
                w_dtype: "mxfp4",
                a_dtype: "fp8_e4m3",
            },
            note: "N = 124160 at TP=2 exceeds DEC_MAX_N 36864, so this shape can never take \
                   the decode kernel however small M is (docs/LEDGER.md, the pinned band cells)",
        });
        if tp > 1 {
            ops.push(Op {
                name: "logits.gather".into(),
                site: Site::Head,
                kernel: KernelOp::LogitsGather,
                selector: Selector::Always,
                inputs: vec![Input::new("logits.partial", ACCEPTS_BF16)],
                outputs: vec![Edge::bf16("logits", "the full-vocabulary logits")],
                state: vec![],
                weights: vec![],
                geometry: Geometry::Gather { n: d.vocab_local },
                note: "lm_head is column-parallel, so the shards are GATHERED, not reduced",
            });
        }

        let drafter = drafter_ops(&config, &d);

        Ok(Self {
            config,
            tp,
            verify_width,
            ops,
            drafter,
            mxfp4,
            saber,
            glue,
            gdn_fused_max_items: GDN_FUSED_MAX_ITEMS,
        })
    }

    /// The ordered op list for `(m, phase)`, with shape keys and resolved layouts.
    ///
    /// `m` is the total row count the GEMMs see. Where an op's selection needs the
    /// `(batch, q_len)` split that `m` alone cannot recover — attention's `q_len <= 10` rule,
    /// the GDN fused/pair routing, the all-reduce size threshold — this assumes the graph's
    /// [`Graph::verify_width`]: `batch = m / verify_width` when it divides, otherwise
    /// `q_len = 1` and `batch = m`, which is the SPEC=0 reading. Call [`Graph::ops_for`] with
    /// an explicit [`Bucket`] when the split matters.
    #[must_use]
    pub fn ops_for_bucket(&self, m: u32, phase: Phase) -> Vec<ResolvedOp<'_>> {
        self.ops_for(self.bucket_for(m, phase))
    }

    /// The bucket [`Graph::ops_for_bucket`] infers from `(m, phase)`.
    #[must_use]
    pub fn bucket_for(&self, m: u32, phase: Phase) -> Bucket {
        match phase {
            Phase::Prefill => Bucket::Prefill { tokens: m, seqs: 1 },
            Phase::Decode => {
                if self.verify_width > 1 && m % self.verify_width == 0 {
                    Bucket::Decode {
                        batch: m / self.verify_width,
                        verify_width: self.verify_width,
                    }
                } else {
                    Bucket::Decode {
                        batch: m,
                        verify_width: 1,
                    }
                }
            }
        }
    }

    /// The bucket the **head site** is planned at, for a step whose body runs at `bucket`.
    ///
    /// One row per sampled position ([`Bucket::sampled_rows`]), always as a decode bucket of
    /// width 1: the head's geometry is a row count and nothing else, and calling it a prefill
    /// bucket of `tokens` rows is exactly the mistake this exists to stop. A body plan and a head
    /// plan are therefore two plans, joined by the row selection the forward library launches
    /// between them.
    ///
    /// A caller that plans the head in the SAME plan as the body gets the body's row count for
    /// both, which is correct only where `sampled_rows == m` — every decode bucket, and no
    /// prefill chunk longer than its sequence count.
    #[must_use]
    pub fn head_bucket(&self, bucket: Bucket) -> Bucket {
        Bucket::Decode {
            batch: bucket.sampled_rows(),
            verify_width: 1,
        }
    }

    /// The ordered op list for an explicit bucket.
    #[must_use]
    pub fn ops_for(&self, bucket: Bucket) -> Vec<ResolvedOp<'_>> {
        let producers = self.producers(bucket);
        // One registry for the whole bucket, so `Registry::selections()` ends up holding every
        // question the pass asked -- the reference's own reason for threading it through.
        let reg = Registry::new(&self.saber);
        self.ops
            .iter()
            .filter(|op| {
                op.runs_in_at(
                    bucket,
                    self.gdn_value_heads_per_rank(),
                    self.gdn_fused_max_items,
                )
            })
            .map(|op| self.resolve(op, bucket, &producers, &reg))
            .collect()
    }

    /// The drafter head's ops at a bucket. Separate from [`Graph::ops_for`] because spec
    /// decode is P4 and the drafter is a second model in the tier, not a stage of this one.
    #[must_use]
    pub fn drafter_ops_for(&self, bucket: Bucket) -> Vec<ResolvedOp<'_>> {
        let producers = self.producers(bucket);
        let reg = Registry::new(&self.saber);
        self.drafter
            .iter()
            .filter(|op| {
                op.runs_in_at(
                    bucket,
                    self.gdn_value_heads_per_rank(),
                    self.gdn_fused_max_items,
                )
            })
            .map(|op| self.resolve(op, bucket, &producers, &reg))
            .collect()
    }

    fn resolve<'a>(
        &'a self,
        op: &'a Op,
        bucket: Bucket,
        producers: &BTreeMap<&str, (&str, LayoutRule)>,
        reg: &Registry<'_>,
    ) -> ResolvedOp<'a> {
        let m = bucket.m();
        let input_layouts: Vec<Layout> = op
            .inputs
            .iter()
            .map(|i| {
                producers
                    .get(i.edge.as_str())
                    .map_or(Layout::Bf16, |(_, rule)| rule.resolve(m))
            })
            .collect();
        let gemm_family = match op.kernel {
            KernelOp::GemmW4A8 | KernelOp::GemmBf16 => {
                Some(if input_layouts.first() == Some(&Layout::ATiledFp8) {
                    GemmFamily::ATiled
                } else {
                    GemmFamily::RowMajor
                })
            }
            _ => None,
        };
        let shape_keys = op.shape_keys(bucket);
        let route = self.route(op, &shape_keys, reg);
        ResolvedOp {
            op,
            input_layouts,
            output_layouts: op.outputs.iter().map(|e| e.rule.resolve(m)).collect(),
            gemm_family,
            entry: route.entry,
            plugin: route.plugin,
            refusal: route.refusal,
            ported: route.plugin.is_some(),
            n_launches: op.n_launches_at(m),
            shape_keys,
        }
    }

    /// Ask the **kernel plane** what runs, and derive `ported` from its answer.
    ///
    /// Nothing here is a list of what Saber has ported. There are two questions and each goes
    /// to the crate that owns it:
    ///
    /// * **saber ops** go to `saber::Registry::select(op_id, geometry)` over the committed
    ///   `[[registry]]` rows — the same necessary-condition question prod asks
    ///   (`name = saber.select(op, **geometry)`), answered from the geometry the *graph* legitimately
    ///   owns. Strides, pitches and buffer sizes are per-call and deliberately absent: they come
    ///   from the caller's tensors, not from the model, so the graph does not invent them and
    ///   does not call the full `saber::resolve`.
    /// * **MXFP4 ops** go to `rules::resolve(op, Shape { m, n, k }, rules)`, which needs nothing
    ///   the graph does not have.
    ///
    /// Either way the answer names a symbol; `saber::plugin_for` and `prebuilt::ALL` turn that into
    /// "a committed code object holds it, or nothing does". Adding a plugin to `prebuilt::ALL`
    /// flips `ported` here with no edit to this file, which is the point.
    fn route(&self, op: &Op, keys: &ShapeKeys, reg: &Registry<'_>) -> Route {
        match op.kernel {
            // ---- saber: the registry answers ------------------------------------------
            KernelOp::AttnPrefillPaged
            | KernelOp::AttnDecodePaged
            | KernelOp::GdnFusedUpdate
            | KernelOp::GdnConvUpdate
            | KernelOp::GdnRecurrentUpdate
            | KernelOp::GdnConvPrep
            | KernelOp::GdnKktSolve
            | KernelOp::GdnChunkScan
            | KernelOp::AllReduce => {
                let op_id =
                    if op.kernel == KernelOp::AttnDecodePaged && keys.num("ragged") == Some(1) {
                        saber::ATTN_DECODE_RAGGED_OP
                    } else if op.kernel == KernelOp::AttnDecodePaged
                        && keys.num("q_len").unwrap_or(1) > i64::from(MAX_DECODE_QLEN)
                    {
                        // A separately compiled wider decode tile takes priority over the
                        // inventory's prefill fallback for otherwise unsupported rectangles.
                        let decode = Route::from_registry(op.kernel.id(), keys, reg);
                        if decode.refusal.is_none() {
                            return decode;
                        }
                        KernelOp::AttnPrefillPaged.id()
                    } else {
                        op.kernel.id()
                    };
                Route::from_registry(op_id, keys, reg)
            }
            // ---- P27-15d: the tree form's twins are asked as their PARENTS (the registry holds
            // no tree rows: `noop_gfx1201.toml`); the executor names the p27 symbols itself.
            KernelOp::GdnConvTree => Route::from_registry(KernelOp::GdnConvUpdate.id(), keys, reg),
            KernelOp::GdnTreeUpdate => {
                Route::from_registry(KernelOp::GdnRecurrentUpdate.id(), keys, reg)
            }
            KernelOp::AttnDecodeTree => {
                Route::from_registry(saber::ATTN_DECODE_RAGGED_OP, keys, reg)
            }

            // ---- MXFP4: the launch bands answer ----------------------------------------
            KernelOp::GemmW4A8 => {
                let (n, k) = match op.geometry {
                    Geometry::Gemm { n, k, .. } => (n, k),
                    _ => return Route::none(),
                };
                let m = u32::try_from(keys.num("m").unwrap_or(1)).unwrap_or(1);
                self.mxfp4_route(
                    saber_kernels::Op::Gemm {
                        has_wref: true,
                        partitions: self.partitions_for(op),
                    },
                    Shape::new(m, n, k),
                )
            }
            KernelOp::AddRmsQuant => {
                let k = match op.geometry {
                    Geometry::Rms { k } => k,
                    _ => return Route::none(),
                };
                let m = u32::try_from(keys.num("m").unwrap_or(1)).unwrap_or(1);
                self.mxfp4_route(
                    saber_kernels::Op::AddRmsQuant {
                        eps: self.config.rms_norm_eps,
                    },
                    Shape::new(m, 0, k),
                )
            }
            KernelOp::SiluMulQuant => {
                let n = match op.geometry {
                    Geometry::SiluMul { n } => n,
                    _ => return Route::none(),
                };
                let m = u32::try_from(keys.num("m").unwrap_or(1)).unwrap_or(1);
                self.mxfp4_route(saber_kernels::Op::SiluMulQuant, Shape::new(m, n, 0))
            }
            KernelOp::GdnNormQuant => {
                let (n, zs) = match op.geometry {
                    Geometry::GdnNorm { n, zs, .. } => (n, zs),
                    _ => return Route::none(),
                };
                let m = u32::try_from(keys.num("m").unwrap_or(1)).unwrap_or(1);
                self.mxfp4_route(
                    saber_kernels::Op::GdnNormQuant {
                        zs: i64::from(zs),
                        eps: self.config.rms_norm_eps,
                    },
                    Shape::new(m, n, 0),
                )
            }

            // ---- the attention glue: its own rules answer -------------------------------
            //
            // The shapes are derived from the same geometry the executor derives them from, so
            // a refusal here is a refusal there. The paged strides `kv_append` takes are a
            // property of the cache LAYOUT, which the graph states once
            // (`[blocks, kv_heads, block_size, 2 * head_dim]`) and both readers compute from.
            KernelOp::QkNormRope | KernelOp::AttnGateQuant | KernelOp::KvAppend => {
                let m = u32::try_from(keys.num("m").unwrap_or(1)).unwrap_or(1);
                match glue_shape(op, m) {
                    Some((gop, shape)) => {
                        match saber_kernels::glue::resolve(gop, shape, &self.glue) {
                            Ok(desc) => Route::from_plugin_name(desc.kernel, &desc.plugin),
                            Err(e) => Route {
                                entry: None,
                                plugin: None,
                                refusal: Some(e.to_string()),
                            },
                        }
                    }
                    None => Route::none(),
                }
            }

            // ---- the bf16 head (N2): the head plugin's norm, the drafter's bf16 GEMM --------
            KernelOp::AddRmsBf16 | KernelOp::GemmBf16Head | KernelOp::GemmFp8Head => {
                let m = u32::try_from(keys.num("m").unwrap_or(1)).unwrap_or(1);
                let desc = match (op.kernel, op.geometry.clone()) {
                    (KernelOp::AddRmsBf16, Geometry::Rms { k }) => {
                        saber_kernels::head::HeadRules::prod_head().and_then(|r| {
                            saber_kernels::head::resolve(
                                saber_kernels::head::HeadOp::AddRmsBf16 {
                                    eps_bits: self.config.rms_norm_eps.to_bits(),
                                },
                                saber_kernels::head::HeadShape::packed(m, k),
                                &r,
                            )
                        })
                    }
                    (KernelOp::GemmBf16Head, Geometry::Gemm { n, k, .. }) => {
                        saber_kernels::dflash::DflashRules::prod_dflash()
                            .and_then(|r| saber_kernels::dflash::resolve_head_bgemm(m, n, k, &r))
                    }
                    (KernelOp::GemmFp8Head, Geometry::Gemm { n, k, .. }) => {
                        saber_kernels::dflash::DflashRules::prod_dflash()
                            .and_then(|r| saber_kernels::dflash::resolve_head_w8(m, n, k, &r))
                    }
                    _ => return Route::none(),
                };
                match desc {
                    Ok(desc) => Route::from_plugin_name(desc.kernel, &desc.plugin),
                    Err(e) => Route {
                        entry: None,
                        plugin: None,
                        refusal: Some(e.to_string()),
                    },
                }
            }

            // ---- no kernel plane holds these -------------------------------------------
            KernelOp::GemmBf16 | KernelOp::Embedding | KernelOp::LogitsGather => Route::none(),
        }
    }

    /// Merged-linear partition select. Every merged GEMM in this tier reads ONE copy of A — the
    /// partitions exist for the Paro rotation streams, where each merged segment reads a
    /// differently rotated A — so this is the stock `NO_PARTITIONS` today. It is a method rather
    /// than a constant so that a rotated tier has one place to override.
    fn partitions_for(&self, _op: &Op) -> Partitions {
        NO_PARTITIONS
    }

    fn mxfp4_route(&self, op: saber_kernels::Op, shape: Shape) -> Route {
        match saber_kernels::rules::resolve(op, shape, &self.mxfp4) {
            Ok(desc) => Route::from_plugin_name(desc.kernel, &desc.plugin),
            Err(e) => Route {
                entry: None,
                plugin: None,
                refusal: Some(e.to_string()),
            },
        }
    }

    /// Edge name -> (producing op name, the edge's layout rule), for the ops in `bucket`.
    fn producers(&self, bucket: Bucket) -> BTreeMap<&str, (&str, LayoutRule)> {
        let mut map = BTreeMap::new();
        for op in self.ops.iter().chain(self.drafter.iter()).filter(|op| {
            op.runs_in_at(
                bucket,
                self.gdn_value_heads_per_rank(),
                self.gdn_fused_max_items,
            )
        }) {
            for edge in &op.outputs {
                map.insert(edge.name.as_str(), (op.name.as_str(), edge.rule));
            }
        }
        map
    }

    /// Check that every consumer can read what its producer wrote, at this bucket.
    ///
    /// This is the check P0 follow-up (a) asks the planner to run. It is not a formality: at
    /// `M >= `[`a_tiled_min_m`] the fused producers switch to [`Layout::ATiledFp8`] while
    /// `gdn_norm_quant` — which has one instantiation and writes row-major at every M, exactly
    /// as the reference does — stays row-major, so `gdn.out_proj` is the one GEMM in a GDN
    /// layer whose family does not follow M.
    ///
    /// **What it cannot catch**, and did not: a W4A8 GEMM `accepts` BOTH fp8 layouts, so a
    /// consumer that resolves its family against M rather than against this edge passes this
    /// check and still reads the wrong bytes. The planner asks the rules what the producer
    /// WRITES (`saber_kernels::rules::producer_writes_tiled`) and refuses a disagreement;
    /// `saber_exec::layer` is where that lives, because only the kernel plane knows.
    pub fn check_layouts(&self, bucket: Bucket) -> Result<(), Vec<LayoutFault>> {
        self.check_layouts_under(bucket, saber_kernels::rules::ATiledLaw::committed())
    }

    /// [`Graph::check_layouts`] under a plan's own layout law (G6's mid band).
    ///
    /// # Errors
    /// Every consumer that cannot read what its producer writes under `law`, at this bucket.
    pub fn check_layouts_under(
        &self,
        bucket: Bucket,
        law: saber_kernels::rules::ATiledLaw,
    ) -> Result<(), Vec<LayoutFault>> {
        let m = bucket.m();
        let producers = self.producers(bucket);
        let mut faults = Vec::new();
        for op in self.ops.iter().chain(self.drafter.iter()).filter(|op| {
            op.runs_in_at(
                bucket,
                self.gdn_value_heads_per_rank(),
                self.gdn_fused_max_items,
            )
        }) {
            for input in &op.inputs {
                let Some((producer, rule)) = producers.get(input.edge.as_str()) else {
                    faults.push(LayoutFault::Unproduced {
                        op: op.name.clone(),
                        edge: input.edge.clone(),
                    });
                    continue;
                };
                let found = rule.resolve_under(m, law);
                if !input.accepts.contains(&found) {
                    faults.push(LayoutFault::Mismatch {
                        op: op.name.clone(),
                        edge: input.edge.clone(),
                        produced_by: (*producer).to_string(),
                        found,
                        accepts: input.accepts,
                    });
                }
            }
        }
        if faults.is_empty() {
            Ok(())
        } else {
            Err(faults)
        }
    }

    /// The statically declared launch count for `bucket`, target model only.
    ///
    /// Compare against prod's measured **1084 launches per decode step at concurrency 8**
    /// ([`PROD_LAUNCHES_PER_STEP_CONC8`], `docs/LEDGER.md`). This bound is smaller, and every
    /// part of the difference is nameable:
    ///
    /// * **The drafter.** DFlash2 block spec decode runs a second model — one attention layer
    ///   plus `fc` — once per draft position. [`Graph::drafter_ops_for`] describes it; it is
    ///   not in this total because spec decode is P4 and the drafter is a separate tier
    ///   member. At `verify_width 5` it is several hundred launches on its own.
    /// * **Sampling.** Argmax, top-k/top-p, the rejection-sampling verify step, the
    ///   acceptance bookkeeping and the per-step metrics all launch kernels.
    /// * **Elementwise work no plane holds.** The embedding lookup, q/k norm, mrope, the
    ///   attention output gate, KV `reshape_and_cache`, the GDN per-candidate state snapshots,
    ///   and the logits gather are [`LaunchBudget::unmodelled`] here: they have no registry row
    ///   and no launch rule to ask, so they declare no count rather than a guessed one.
    /// * **Batch-cut multiplicity.** Attention launches once per *run of equal `q_len`*, so a
    ///   mixed batch launches it more than once per layer; this bound assumes one run.
    #[must_use]
    pub fn launch_count_lower_bound(&self, bucket: Bucket) -> LaunchBudget {
        let mut by_kernel: BTreeMap<&'static str, u32> = BTreeMap::new();
        let mut unmodelled: BTreeMap<&'static str, u32> = BTreeMap::new();
        let mut total = 0u32;
        for op in self.ops.iter().filter(|op| {
            op.runs_in_at(
                bucket,
                self.gdn_value_heads_per_rank(),
                self.gdn_fused_max_items,
            )
        }) {
            match op.n_launches_at(bucket.m()) {
                Some(n) => {
                    *by_kernel.entry(op.kernel.id()).or_default() += n;
                    total += n;
                }
                None => *unmodelled.entry(op.kernel.id()).or_default() += 1,
            }
        }
        LaunchBudget {
            bucket,
            total,
            by_kernel,
            unmodelled,
        }
    }

    /// Value heads per rank, 24 at TP=2. The GDN fused/pair routing keys on it.
    #[must_use]
    pub fn gdn_value_heads_per_rank(&self) -> u32 {
        self.config.linear_num_value_heads / self.tp
    }

    /// Every distinct bound record, target and drafter, keyed by record name.
    ///
    /// Panics if two ops bind the same record with different shapes or dtypes — that would
    /// mean the graph disagrees with itself about the bytes, which is the failure the pack's
    /// per-record hash exists to make loud at load time and which should be impossible here.
    #[must_use]
    pub fn bound_records(&self) -> BTreeMap<&str, &Binding> {
        let mut map: BTreeMap<&str, &Binding> = BTreeMap::new();
        for op in self.ops.iter().chain(self.drafter.iter()) {
            for binding in &op.weights {
                if let Some(existing) = map.get(binding.record.as_str()) {
                    assert_eq!(
                        (existing.dtype, &existing.shape),
                        (binding.dtype, &binding.shape),
                        "record {} is bound twice with different bytes",
                        binding.record
                    );
                } else {
                    map.insert(binding.record.as_str(), binding);
                }
            }
        }
        map
    }

    /// The record classes the text-only graph deliberately does not bind.
    #[must_use]
    pub fn unused_records(&self) -> &'static [UnusedRecords] {
        UNUSED_RECORDS
    }

    /// Whether `record` is covered by an [`UNUSED_RECORDS`] rule.
    #[must_use]
    pub fn is_unused(&self, record: &str) -> bool {
        UNUSED_RECORDS.iter().any(|r| record.starts_with(r.prefix))
    }

    /// The record set a **device-final** pack holds, in forward order.
    ///
    /// A v0 pack is the checkpoint's own bytes with the TP shard applied, so the engine pays the
    /// NVFP4 → MXFP4 requant, the WPERM fragment order, the merges and two dtype casts at load —
    /// which is exactly what the tier pack exists to delete (`ARCHITECTURE.md` § "The model
    /// tier"). This function is the model's half of that: it says **which records must exist,
    /// what layout each one is, and which checkpoint tensors produced it**. `tools/pack` does the
    /// numerics; `docs/abi/PACK.md` § Layout ids defines the layouts; `saber-mem`'s
    /// `pack::LayoutId` is the same table in code.
    ///
    /// Three rules make the set:
    ///
    /// 1. **Every `gemm_w4a8` op becomes three records** — `<op>.w`, `<op>.ws`, `<op>.wref` —
    ///    built from the op's weight bindings in order. The op's name is the record's name,
    ///    because a merged weight is no longer any one checkpoint tensor.
    /// 2. **A `gemm_bf16` op with several sources becomes one merged `bf16` record**; with one
    ///    source it stays that source's raw record, since a rename would buy nothing.
    /// 3. **Everything else a kernel reads passes through raw**, except `A_log` and `dt_bias`,
    ///    which the GDN kernels take as `f32` and the checkpoint stores as `bf16`.
    ///
    /// Bindings marked [`Consumed::PackBuild`] — `weight_global_scale`, `input_global_scale` —
    /// are **inputs to the transform and are not emitted**: a device-final pack that carried them
    /// would be inviting a second application of the same dequantization.
    ///
    /// The order is the op order, so a device-final pack's records are laid out layer by layer and
    /// a loader that wants the first N layers can take a contiguous window (`docs/LEDGER.md`,
    /// workstream K: "the record order in a pack is a load-time property").
    ///
    /// # Errors
    /// An op whose weight list this rule set cannot group — a GEMM with no weight at all, or a
    /// merged weight whose sources disagree about their quantization.
    pub fn device_final_records(&self) -> Result<Vec<DeviceFinalRecord>, ModelError> {
        let mut out: Vec<DeviceFinalRecord> = Vec::new();
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        for op in self.ops.iter().chain(self.drafter.iter()) {
            let groups = source_groups(op)?;
            let matrix = matches!(
                op.kernel,
                KernelOp::GemmW4A8
                    | KernelOp::GemmBf16
                    | KernelOp::GemmBf16Head
                    | KernelOp::GemmFp8Head
            );
            if matrix && groups.is_empty() {
                return Err(ModelError::Config(format!(
                    "{}: a {} op with no weight binding",
                    op.name, op.kernel
                )));
            }
            if matrix {
                for rec in gemm_records(op, &groups)? {
                    push_unique(&mut out, &mut seen, rec)?;
                }
            }
            // Everything the op binds that the matrix records did not consume.
            for b in &op.weights {
                if b.consumed == Consumed::PackBuild {
                    continue;
                }
                if matrix && groups.iter().any(|g| g.covers(&b.record)) {
                    continue;
                }
                let layout = match b.role {
                    WeightRole::ALog | WeightRole::DtBias => LAYOUT_F32_FROM_BF16,
                    _ => LAYOUT_RAW,
                };
                push_unique(
                    &mut out,
                    &mut seen,
                    DeviceFinalRecord {
                        name: b.record.clone(),
                        layout_id: layout,
                        role: DeviceFinalRole::Passthrough,
                        op: Some(op.name.clone()),
                        dtype: if layout == LAYOUT_F32_FROM_BF16 {
                            Dtype::F32
                        } else {
                            b.dtype
                        },
                        shard: b.shard,
                        shape: b.shape.clone(),
                        sources: vec![SourceTensor {
                            weight: b.record.clone(),
                            quant: SourceQuant::AsStored,
                            scale: None,
                            global_scale: None,
                            input_scale: None,
                        }],
                    },
                )?;
            }
        }
        Ok(out)
    }
}

// =====================================================================================
// Device-final records
// =====================================================================================

/// `layout_id 0`, `docs/abi/PACK.md`: raw checkpoint bytes.
pub const LAYOUT_RAW: u32 = 0;
/// `layout_id 1`: MXFP4 W4A8 weight in WMMA fragment order.
pub const LAYOUT_MXFP4_WPERM: u32 = 1;
/// `layout_id 2`: its `e8m0` block scales, transposed to `[K/32, N]`.
pub const LAYOUT_MXFP4_BLOCK_SCALE: u32 = 2;
/// `layout_id 3`: its per-row reference exponent, `[N]`.
pub const LAYOUT_MXFP4_ROW_REF: u32 = 3;
/// `layout_id 4`: [`LAYOUT_MXFP4_WPERM`] over a dim-0 concatenation of several tensors.
pub const LAYOUT_MXFP4_WPERM_MERGED: u32 = 4;
/// `layout_id 5`: an `f32` copy of a `bf16` tensor (`A_log`, `dt_bias`).
pub const LAYOUT_F32_FROM_BF16: u32 = 5;
/// `layout_id 6`: a `bf16` dim-0 concatenation with no requantization (the drafter head).
pub const LAYOUT_BF16_MERGED: u32 = 6;
/// `layout_id 7`: the DFlash2 drafter's fp8 W8A8 B operand, preshuffled to `[N/16, K*16]`.
pub const LAYOUT_FP8_PRESHUFFLE: u32 = 7;
/// `layout_id 8`: its `[N/128, K/128]` `f32` block scales.
pub const LAYOUT_FP8_BLOCK_SCALE: u32 = 8;
/// `layout_id 9`: the reference's int2 draft head, quarter-split packed to `[N, K/4]`.
pub const LAYOUT_INT2_HEAD: u32 = 9;
/// `layout_id 10`: its premultiplied `bf16` group scale, `[N, K/128]`.
pub const LAYOUT_INT2_HEAD_SCALE: u32 = 10;
/// `layout_id 11`: its premultiplied `bf16` group zero, `[N, K/128]`.
pub const LAYOUT_INT2_HEAD_ZERO: u32 = 11;
/// `layout_id 12`: a `bf16` matrix dequantized from a block-fp8 source, dim-0 concatenated.
pub const LAYOUT_BF16_FROM_FP8_BLOCK: u32 = 12;
/// `layout_id 13`: a rope `cos|sin` table computed from the checkpoint's rope parameters.
pub const LAYOUT_ROPE_COS_SIN: u32 = 13;
/// `layout_id 14`: a `bf16` matrix dequantized from a **per-CHANNEL** fp8 source — the target's
/// `lm_head`, as `ref_nvfp4.RefFp8ChannelToBf16` leaves it at load.
pub const LAYOUT_BF16_FROM_FP8_CHANNEL: u32 = 14;
/// `layout_id 15`: a **per-channel** e4m3 matrix carried RAW, `[N, K]` in the checkpoint's own row
/// order and dim-0 sharded -- the fp8-weight head's `lm_head.w8` (workstream N4), which the kernel
/// expands in register to exactly [`LAYOUT_BF16_FROM_FP8_CHANNEL`]'s bytes.
pub const LAYOUT_FP8_CHANNEL: u32 = 15;
/// `layout_id 16`: its per-output-row scale as `f32` `[N]`, an exact widening of the checkpoint's
/// `bf16` `weight_scale` (prod's `ChannelQuantScaleParameter` is `f32`).
pub const LAYOUT_F32_CHANNEL_SCALE: u32 = 16;

/// What a device-final record is to the op that reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceFinalRole {
    /// The weight matrix itself (`w`).
    Weight,
    /// Its block scales (`ws`).
    BlockScale,
    /// Its row reference exponent (`wref`).
    RowRef,
    /// A record that is carried across unchanged, or cast without changing what it means.
    Passthrough,
}

impl DeviceFinalRole {
    /// The kernarg-role suffix a matrix record's name carries.
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            DeviceFinalRole::Weight => "w",
            DeviceFinalRole::BlockScale => "ws",
            DeviceFinalRole::RowRef => "wref",
            DeviceFinalRole::Passthrough => "",
        }
    }

    /// Short name for a manifest or a table.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceFinalRole::Weight => "weight",
            DeviceFinalRole::BlockScale => "block_scale",
            DeviceFinalRole::RowRef => "row_ref",
            DeviceFinalRole::Passthrough => "passthrough",
        }
    }
}

/// How a source tensor is quantized in the checkpoint — what the builder has to undo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceQuant {
    /// NVFP4: `weight_packed` (two `e2m1` per byte) × an `e4m3` scale per 16 values, ÷ a per-tensor
    /// `weight_global_scale` **divisor** (compressed-tensors stores `(6 * 448) / amax`).
    Nvfp4,
    /// fp8 per output channel: `weight` (`e4m3`) × `weight_scale` `[N, 1]`.
    Fp8Channel,
    /// `bf16`, nothing to undo.
    Bf16,
    /// Not a matrix: carried as stored.
    AsStored,
}

impl SourceQuant {
    /// Short name for the plan JSON.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            SourceQuant::Nvfp4 => "nvfp4",
            SourceQuant::Fp8Channel => "fp8_channel",
            SourceQuant::Bf16 => "bf16",
            SourceQuant::AsStored => "as_stored",
        }
    }
}

/// One checkpoint tensor that feeds a device-final record, with the companions its dequantization
/// needs.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceTensor {
    /// The weight record's name.
    pub weight: String,
    /// What has to be undone to get fp32.
    pub quant: SourceQuant,
    /// `weight_scale`, when there is one.
    pub scale: Option<String>,
    /// `weight_global_scale` (NVFP4 only).
    pub global_scale: Option<String>,
    /// `input_global_scale` (NVFP4 only; the activation scale, which this tier's kernel does not
    /// read — it is named so the builder can account for every record it consumed).
    pub input_scale: Option<String>,
}

/// One record a device-final pack holds.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceFinalRecord {
    /// The record's name in the pack: the checkpoint's for a passthrough, `<op>.<role>` for a
    /// matrix.
    pub name: String,
    /// `docs/abi/PACK.md` § Layout ids.
    pub layout_id: u32,
    /// What it is to its consumer.
    pub role: DeviceFinalRole,
    /// The op that reads it.
    pub op: Option<String>,
    /// The dtype the pack holds.
    pub dtype: Dtype,
    /// How the record is sharded.
    pub shard: ShardKind,
    /// The **rank-local** logical shape.
    pub shape: Vec<u64>,
    /// The checkpoint tensors that produced it, in concatenation order.
    pub sources: Vec<SourceTensor>,
}

impl DeviceFinalRecord {
    /// Whether this record is one the builder has to compute rather than copy.
    #[must_use]
    pub fn is_transformed(&self) -> bool {
        self.layout_id != LAYOUT_RAW
    }
}

/// One source group of a GEMM op: its weight binding and the companions that go with it.
struct Group<'a> {
    weight: &'a Binding,
    scale: Option<&'a Binding>,
    global_scale: Option<&'a Binding>,
    input_scale: Option<&'a Binding>,
}

impl Group<'_> {
    fn covers(&self, record: &str) -> bool {
        [
            Some(self.weight),
            self.scale,
            self.global_scale,
            self.input_scale,
        ]
        .into_iter()
        .flatten()
        .any(|b| b.record == record)
    }

    fn quant(&self) -> SourceQuant {
        match (self.weight.role, self.weight.dtype) {
            (WeightRole::WeightPacked, _) => SourceQuant::Nvfp4,
            (WeightRole::Weight, Dtype::F8E4M3) => SourceQuant::Fp8Channel,
            _ => SourceQuant::Bf16,
        }
    }

    fn source(&self) -> SourceTensor {
        SourceTensor {
            weight: self.weight.record.clone(),
            quant: self.quant(),
            scale: self.scale.map(|b| b.record.clone()),
            global_scale: self.global_scale.map(|b| b.record.clone()),
            input_scale: self.input_scale.map(|b| b.record.clone()),
        }
    }
}

/// Split an op's weight list into source groups: a `Weight` / `WeightPacked` binding opens a
/// group and the scale bindings that follow it belong to it.
///
/// This is the op's own order, which is the concatenation order of the merge — `gate_proj` before
/// `up_proj`, `in_proj_qkv` before `in_proj_z`, `b` before `a`. Getting it from the binding list
/// rather than from a table here is the point: the graph already had to say it, once.
fn source_groups(op: &Op) -> Result<Vec<Group<'_>>, ModelError> {
    let mut groups: Vec<Group<'_>> = Vec::new();
    for b in &op.weights {
        match b.role {
            WeightRole::Weight | WeightRole::WeightPacked => groups.push(Group {
                weight: b,
                scale: None,
                global_scale: None,
                input_scale: None,
            }),
            WeightRole::WeightScale
            | WeightRole::WeightGlobalScale
            | WeightRole::InputGlobalScale => {
                let Some(g) = groups.last_mut() else {
                    return Err(ModelError::Config(format!(
                        "{}: `{}` is a scale with no weight before it",
                        op.name, b.record
                    )));
                };
                let slot = match b.role {
                    WeightRole::WeightScale => &mut g.scale,
                    WeightRole::WeightGlobalScale => &mut g.global_scale,
                    _ => &mut g.input_scale,
                };
                if slot.is_some() {
                    return Err(ModelError::Config(format!(
                        "{}: two {:?} bindings for `{}`",
                        op.name, b.role, g.weight.record
                    )));
                }
                *slot = Some(b);
            }
            _ => {}
        }
    }
    Ok(groups)
}

/// The records one GEMM op contributes.
fn gemm_records(op: &Op, groups: &[Group<'_>]) -> Result<Vec<DeviceFinalRecord>, ModelError> {
    let Geometry::Gemm { n, k, .. } = op.geometry else {
        return Err(ModelError::Config(format!(
            "{}: a GEMM op without a Gemm geometry",
            op.name
        )));
    };
    let merged = groups.len() > 1;
    let sources: Vec<SourceTensor> = groups.iter().map(Group::source).collect();
    // The rank-local rows of a merged weight are the sum of its sources' rank-local rows, and the
    // geometry already states it; disagreeing with the geometry means one of the two is wrong.
    let rows: u64 = groups.iter().map(|g| g.weight.shape[0]).sum();
    if rows != u64::from(n) {
        return Err(ModelError::Config(format!(
            "{}: the weight bindings give {rows} rows, the geometry says N = {n}",
            op.name
        )));
    }
    let shard = if merged {
        ShardKind::Dim0Segments
    } else {
        groups[0].weight.shard
    };
    if op.kernel == KernelOp::GemmFp8Head {
        // The fp8-weight head (N4): TWO records, both raw bytes of the checkpoint's per-channel
        // fp8 source -- the e4m3 matrix and its scale widened to f32 -- and nothing requantized.
        let g = match groups {
            [g] if g.scale.is_some() => g,
            _ => {
                return Err(ModelError::Config(format!(
                    "{}: the fp8 head wants one per-channel fp8 weight and its scale",
                    op.name
                )))
            }
        };
        return Ok(vec![
            DeviceFinalRecord {
                name: format!("{}.w8", op.name),
                layout_id: LAYOUT_FP8_CHANNEL,
                role: DeviceFinalRole::Weight,
                op: Some(op.name.clone()),
                dtype: Dtype::F8E4M3,
                shard: g.weight.shard,
                shape: vec![u64::from(n), u64::from(k)],
                sources: sources.clone(),
            },
            DeviceFinalRecord {
                name: format!("{}.ws", op.name),
                layout_id: LAYOUT_F32_CHANNEL_SCALE,
                role: DeviceFinalRole::BlockScale,
                op: Some(op.name.clone()),
                dtype: Dtype::F32,
                shard: g.weight.shard,
                shape: vec![u64::from(n)],
                sources,
            },
        ]);
    }
    if op.kernel == KernelOp::GemmBf16Head {
        // The target's bf16 `lm_head` (N2): ONE record, the per-channel fp8 source dequantized to
        // bf16 exactly as prod's loader does (`bf16(f32(w8) * f32(scale))`), the builder's
        // `bf16_head` — the same bytes the drafter pack carries as `head.rerank.w`.
        let g = match groups {
            [g] if g.scale.is_some() => g,
            _ => {
                return Err(ModelError::Config(format!(
                    "{}: the bf16 head wants one per-channel fp8 weight and its scale",
                    op.name
                )))
            }
        };
        return Ok(vec![DeviceFinalRecord {
            name: format!("{}.w", op.name),
            layout_id: LAYOUT_BF16_FROM_FP8_CHANNEL,
            role: DeviceFinalRole::Weight,
            op: Some(op.name.clone()),
            dtype: Dtype::Bf16,
            shard: g.weight.shard,
            shape: vec![u64::from(n), u64::from(k)],
            sources,
        }]);
    }
    if op.kernel == KernelOp::GemmBf16 {
        // The drafter head. One source stays its own raw record; several merge, and nothing is
        // requantized either way.
        if !merged {
            return Ok(vec![DeviceFinalRecord {
                name: groups[0].weight.record.clone(),
                layout_id: LAYOUT_RAW,
                role: DeviceFinalRole::Passthrough,
                op: Some(op.name.clone()),
                dtype: groups[0].weight.dtype,
                shard,
                shape: groups[0].weight.shape.clone(),
                sources,
            }]);
        }
        return Ok(vec![DeviceFinalRecord {
            name: format!("{}.w", op.name),
            layout_id: LAYOUT_BF16_MERGED,
            role: DeviceFinalRole::Weight,
            op: Some(op.name.clone()),
            dtype: Dtype::Bf16,
            shard,
            shape: vec![u64::from(n), u64::from(k)],
            sources,
        }]);
    }
    let w_layout = if merged {
        LAYOUT_MXFP4_WPERM_MERGED
    } else {
        LAYOUT_MXFP4_WPERM
    };
    Ok(vec![
        DeviceFinalRecord {
            name: format!("{}.w", op.name),
            layout_id: w_layout,
            role: DeviceFinalRole::Weight,
            op: Some(op.name.clone()),
            dtype: Dtype::Fp4E2m1X2,
            shard,
            shape: vec![u64::from(n), u64::from(k)],
            sources: sources.clone(),
        },
        DeviceFinalRecord {
            name: format!("{}.ws", op.name),
            layout_id: LAYOUT_MXFP4_BLOCK_SCALE,
            role: DeviceFinalRole::BlockScale,
            op: Some(op.name.clone()),
            dtype: Dtype::E8M0,
            shard,
            // Transposed: the kernel indexes [K/32, N].
            shape: vec![u64::from(k) / 32, u64::from(n)],
            sources: sources.clone(),
        },
        DeviceFinalRecord {
            name: format!("{}.wref", op.name),
            layout_id: LAYOUT_MXFP4_ROW_REF,
            role: DeviceFinalRole::RowRef,
            op: Some(op.name.clone()),
            dtype: Dtype::E8M0,
            shard,
            shape: vec![u64::from(n)],
            sources,
        },
    ])
}

/// Add a record unless an identical one is already there; a record bound by two ops (the GDN conv
/// filter, which the fused kernel, the decode pair and the prefill trio all read) is emitted once.
fn push_unique(
    out: &mut Vec<DeviceFinalRecord>,
    seen: &mut BTreeMap<String, usize>,
    rec: DeviceFinalRecord,
) -> Result<(), ModelError> {
    if let Some(i) = seen.get(&rec.name) {
        let old = &out[*i];
        if old.layout_id != rec.layout_id || old.dtype != rec.dtype || old.shape != rec.shape {
            return Err(ModelError::Config(format!(
                "record `{}` is claimed twice with different bytes: {:?}/{} vs {:?}/{}",
                rec.name, old.dtype, old.layout_id, rec.dtype, rec.layout_id
            )));
        }
        return Ok(());
    }
    seen.insert(rec.name.clone(), out.len());
    out.push(rec);
    Ok(())
}

// =====================================================================================
// Construction helpers
// =====================================================================================

/// Rank-local dimensions, derived once.
#[derive(Clone, Copy, Debug)]
struct Dims {
    tp: u32,
    hidden: u32,
    /// Query heads per rank.
    q_heads: u32,
    /// KV heads per rank.
    kv_heads: u32,
    head_dim: u32,
    gqa: u32,
    /// `q_proj` output width per rank: `q_heads * head_dim * 2` with the output gate.
    q_out: u32,
    /// One of `k_proj` / `v_proj`'s output width per rank.
    kv_out: u32,
    /// Merged QKV width per rank.
    qkv_n: u32,
    /// `o_proj`'s K per rank.
    o_k: u32,
    /// GDN key heads per rank.
    gdn_key_heads: u32,
    /// GDN value heads per rank.
    gdn_value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    conv_width: u32,
    /// The prefill token block the GDN triple is templated on, read from the committed saber
    /// rules (`[gdn] chunk`) rather than restated here.
    gdn_chunk: u32,
    /// The graph's widest speculative step ([`Graph::qwen38_27b_at_width`], [`VERIFY_WIDTH`] by
    /// default), carried here because the conv cache's slot size is a function of it.
    verify_width: u32,
    /// `in_proj_qkv` width per rank: `2 * key_heads * key_dim + value_heads * value_dim`.
    gdn_qkv_n: u32,
    /// `in_proj_z` width per rank.
    gdn_z_n: u32,
    /// Merged `in_proj_qkvz` width per rank.
    gdn_qkvz_n: u32,
    /// Merged `in_proj_ba` width per rank: one row each for `a` and `b` per value head.
    gdn_ba_n: u32,
    /// GDN `out_proj`'s K per rank.
    gdn_out_k: u32,
    /// MLP intermediate per rank.
    inter: u32,
    /// Merged gate+up width per rank.
    gate_up_n: u32,
    /// Rank-local vocabulary.
    vocab_local: u32,
    /// 64 = `partial_rotary_factor 0.25` x `head_dim`.
    rotary_dim: u32,
    kv_dtype: &'static str,
}

impl Dims {
    fn new(
        cfg: &ModelConfig,
        tp: u32,
        saber: &SaberRules,
        verify_width: u32,
    ) -> Result<Self, ModelError> {
        let div = |what: &str, value: u32| -> Result<u32, ModelError> {
            if value % tp != 0 {
                return Err(ModelError::Config(format!(
                    "{what}={value} does not divide by tp={tp}; TP={tp} needs the padded head \
                     layout in the checkpoint (36/6/18/54 with the drafter at 48/12), which the \
                     graph will not invent"
                )));
            }
            Ok(value / tp)
        };
        let q_heads = div("num_attention_heads", cfg.num_attention_heads)?;
        let kv_heads = div("num_key_value_heads", cfg.num_key_value_heads)?;
        let gdn_key_heads = div("linear_num_key_heads", cfg.linear_num_key_heads)?;
        let gdn_value_heads = div("linear_num_value_heads", cfg.linear_num_value_heads)?;
        let inter = div("intermediate_size", cfg.intermediate_size)?;
        let vocab_local = div("vocab_size", cfg.vocab_size)?;

        let q_out = q_heads * cfg.head_dim * if cfg.attn_output_gate { 2 } else { 1 };
        let kv_out = kv_heads * cfg.head_dim;
        let gdn_qkv_n = 2 * gdn_key_heads * cfg.linear_key_head_dim
            + gdn_value_heads * cfg.linear_value_head_dim;
        let gdn_z_n = gdn_value_heads * cfg.linear_value_head_dim;
        Ok(Self {
            tp,
            hidden: cfg.hidden_size,
            q_heads,
            kv_heads,
            head_dim: cfg.head_dim,
            gqa: cfg.gqa(),
            q_out,
            kv_out,
            qkv_n: q_out + 2 * kv_out,
            o_k: q_heads * cfg.head_dim,
            gdn_key_heads,
            gdn_value_heads,
            key_head_dim: cfg.linear_key_head_dim,
            value_head_dim: cfg.linear_value_head_dim,
            conv_width: cfg.linear_conv_kernel_dim,
            gdn_chunk: saber.gdn.chunk,
            verify_width,
            gdn_qkv_n,
            gdn_z_n,
            gdn_qkvz_n: gdn_qkv_n + gdn_z_n,
            gdn_ba_n: 2 * gdn_value_heads,
            gdn_out_k: gdn_z_n,
            inter,
            gate_up_n: 2 * inter,
            vocab_local,
            rotary_dim: cfg.head_dim / 4,
            kv_dtype: if cfg.kv_cache_fp8 { "fp8_e4m3" } else { "bf16" },
        })
    }
}

fn all_reduce_op(
    name: &str,
    site: Site,
    selector: Selector,
    input: &str,
    output: Edge,
    d: &Dims,
    note: &'static str,
) -> Op {
    Op {
        name: name.into(),
        site,
        kernel: KernelOp::AllReduce,
        selector,
        inputs: vec![Input::new(input, ACCEPTS_BF16)],
        outputs: vec![output],
        state: vec![],
        weights: vec![],
        geometry: Geometry::AllReduce {
            world_size: d.tp,
            elems_per_token: d.hidden,
        },
        note,
    }
}

/// The four edge names an [`add_rms_quant_op`] wires, grouped so the builder stays under the
/// argument limit and so the two reads and the two writes are visibly paired.
struct NormEdges<'a> {
    /// The activations to normalise.
    hidden_in: &'a str,
    /// The residual to add first — `None` at layer 0 and on the drafter's first norm.
    residual_in: Option<&'a str>,
    /// The quantized output.
    out_q8: &'a str,
    /// The updated residual.
    out_resid: &'a str,
}

/// The fused residual-add + RMS norm + quant that opens a block.
fn add_rms_quant_op(
    name: String,
    site: Site,
    edges: NormEdges<'_>,
    weight_record: String,
    d: &Dims,
    note: &'static str,
) -> Op {
    let NormEdges {
        hidden_in,
        residual_in,
        out_q8,
        out_resid,
    } = edges;
    let mut inputs = vec![Input::new(hidden_in, ACCEPTS_BF16)];
    if let Some(res) = residual_in {
        inputs.push(Input::new(res, ACCEPTS_BF16));
    }
    Op {
        name,
        site,
        kernel: KernelOp::AddRmsQuant,
        selector: Selector::Always,
        inputs,
        outputs: vec![
            Edge::fused_q8(out_q8, "per-token e4m3 activations for the next GEMM"),
            Edge::bf16(
                out_resid,
                "the residual stream, updated in place by the same kernel",
            ),
        ],
        state: vec![],
        weights: vec![Binding::new(
            weight_record,
            WeightRole::NormWeight,
            Dtype::Bf16,
            vec![u64::from(d.hidden)],
            ShardKind::Replicated,
        )],
        geometry: Geometry::Rms { k: d.hidden },
        note,
    }
}

/// The MLP block: merged gate_up GEMM, `silu_mul_quant`, `down_proj`, all-reduce.
///
/// `gate` and `up` are one GEMM. The reference server merges them into a `MergedColumnParallelLinear` and so
/// does prod: `n8192`/`n7168`/`gate_up`/`gate48` are the shape names the tuned bands in
/// `crates/saber-kernels/rules/mxfp4_gfx1201.toml` are written against, and `gate_up` there is
/// the merged 17408-wide shape at TP=2, not two 8704-wide ones.
fn push_mlp(
    ops: &mut Vec<Op>,
    cfg: &ModelConfig,
    d: &Dims,
    layer: u32,
    in_q8: &str,
    out_edge: &str,
) {
    let site = Site::Layer(layer);
    let p = format!("model.language_model.layers.{layer}.mlp");
    let fp8 = cfg.mlp_is_fp8(layer);
    // At TP=1 there is no reduce, so down_proj writes the block output directly rather than a
    // partial that an identity op would have to copy.
    let down_out = if d.tp > 1 {
        format!("l{layer}.mlp.partial")
    } else {
        out_edge.to_string()
    };

    let mut gate_up_weights = Vec::new();
    for proj in ["gate_proj", "up_proj"] {
        gate_up_weights.extend(linear_bindings(
            &format!("{p}.{proj}"),
            fp8,
            d.inter,
            d.hidden,
            ShardKind::Dim0,
            ShardKind::Dim0,
        ));
    }
    ops.push(Op {
        name: format!("layer.{layer}.mlp.gate_up"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(in_q8, ACCEPTS_FP8_ACTIVATION)],
        outputs: vec![Edge::bf16(
            format!("l{layer}.mlp.gate_up"),
            "concat(gate, up), 17408 wide per rank at TP=2",
        )],
        state: vec![],
        weights: gate_up_weights,
        geometry: Geometry::Gemm {
            n: d.gate_up_n,
            k: d.hidden,
            w_dtype: if fp8 { "fp8_channel" } else { "mxfp4" },
            a_dtype: "fp8_e4m3",
        },
        note: "gate and up are ONE launch (merged column-parallel); the partition boundary is \
               128-aligned, which partition_align requires",
    });
    ops.push(Op {
        name: format!("layer.{layer}.mlp.act"),
        site,
        kernel: KernelOp::SiluMulQuant,
        selector: Selector::Always,
        inputs: vec![Input::new(format!("l{layer}.mlp.gate_up"), ACCEPTS_BF16)],
        outputs: vec![Edge::fused_q8(
            format!("l{layer}.mlp.act_q8"),
            "silu(gate) * up, quantized per token",
        )],
        state: vec![],
        weights: vec![],
        geometry: Geometry::SiluMul { n: d.inter },
        note: "N = 8704 <= silu_large_n 10240, so the 256-thread instantiation",
    });
    ops.push(Op {
        name: format!("layer.{layer}.mlp.down_proj"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(
            format!("l{layer}.mlp.act_q8"),
            ACCEPTS_FP8_ACTIVATION,
        )],
        outputs: vec![Edge::bf16(down_out.clone(), "row-parallel partial sum")],
        state: vec![],
        weights: linear_bindings(
            &format!("{p}.down_proj"),
            fp8,
            d.hidden,
            d.inter,
            ShardKind::Dim1,
            // The fp8 path's weight_scale is per OUTPUT row, so a row-parallel split leaves it
            // replicated; the NVFP4 path's is per 16-value block along K, so it splits Dim1.
            if fp8 {
                ShardKind::Replicated
            } else {
                ShardKind::Dim1
            },
        ),
        geometry: Geometry::Gemm {
            n: d.hidden,
            k: d.inter,
            w_dtype: if fp8 { "fp8_channel" } else { "mxfp4" },
            a_dtype: "fp8_e4m3",
        },
        note: "K = 8704 is 68 k-slabs of 128 -- the `down` row the split-K band is written for",
    });
    if d.tp > 1 {
        ops.push(all_reduce_op(
            &format!("layer.{layer}.mlp.all_reduce"),
            site,
            Selector::Always,
            &down_out,
            Edge::bf16(out_edge, "the block output entering the next layer"),
            d,
            "row-parallel reduce; 80 KiB at batch 1 spec 5 selects the exact kernel, 400 KiB at \
             batch 8 the lossy wht6 -- both committed since saber part 3",
        ));
    }
}

/// The four records an NVFP4 linear needs, or the two an fp8 one does.
fn linear_bindings(
    prefix: &str,
    fp8: bool,
    n: u32,
    k: u32,
    weight_shard: ShardKind,
    scale_shard: ShardKind,
) -> Vec<Binding> {
    if fp8 {
        vec![
            Binding::new(
                format!("{prefix}.weight"),
                WeightRole::Weight,
                Dtype::F8E4M3,
                vec![u64::from(n), u64::from(k)],
                weight_shard,
            ),
            Binding::new(
                format!("{prefix}.weight_scale"),
                WeightRole::WeightScale,
                Dtype::Bf16,
                vec![u64::from(n), 1],
                scale_shard,
            ),
        ]
    } else {
        vec![
            Binding::new(
                format!("{prefix}.weight_packed"),
                WeightRole::WeightPacked,
                Dtype::U8,
                vec![u64::from(n), u64::from(k) / 2],
                weight_shard,
            ),
            Binding::new(
                format!("{prefix}.weight_scale"),
                WeightRole::WeightScale,
                Dtype::F8E4M3,
                vec![u64::from(n), u64::from(k) / 16],
                scale_shard,
            ),
            Binding::new(
                format!("{prefix}.weight_global_scale"),
                WeightRole::WeightGlobalScale,
                Dtype::F32,
                vec![1],
                ShardKind::Replicated,
            )
            .pack_build(),
            Binding::new(
                format!("{prefix}.input_global_scale"),
                WeightRole::InputGlobalScale,
                Dtype::F32,
                vec![1],
                ShardKind::Replicated,
            )
            .pack_build(),
        ]
    }
}

/// One decoder layer. Returns `(hidden_out_edge, residual_out_edge)`.
fn push_layer(
    ops: &mut Vec<Op>,
    cfg: &ModelConfig,
    d: &Dims,
    layer: u32,
    hidden_in: &str,
    residual_in: Option<&str>,
) -> (String, String) {
    let site = Site::Layer(layer);
    let lp = format!("model.language_model.layers.{layer}");
    let resid_a = format!("l{layer}.resid.a");
    let resid_b = format!("l{layer}.resid.b");
    let in_q8 = format!("l{layer}.in_q8");
    let mixer_out = format!("l{layer}.mixer.out");
    let mlp_in_q8 = format!("l{layer}.mlp.in_q8");
    let block_out = format!("l{layer}.out");

    ops.push(add_rms_quant_op(
        format!("layer.{layer}.input_layernorm"),
        site,
        NormEdges {
            hidden_in,
            residual_in,
            out_q8: &in_q8,
            out_resid: &resid_a,
        },
        format!("{lp}.input_layernorm.weight"),
        d,
        "layer 0 has no residual to add: the same op id, a different instantiation, and a \
         which_path counter is owed at construction",
    ));

    match cfg.layer_kind(layer) {
        LayerKind::Gdn => push_gdn(ops, d, layer, &lp, &in_q8, &mixer_out),
        LayerKind::Attention => push_attention(ops, d, layer, &lp, &in_q8, &mixer_out),
    }

    ops.push(add_rms_quant_op(
        format!("layer.{layer}.post_attention_layernorm"),
        site,
        NormEdges {
            hidden_in: &mixer_out,
            residual_in: Some(&resid_a),
            out_q8: &mlp_in_q8,
            out_resid: &resid_b,
        },
        format!("{lp}.post_attention_layernorm.weight"),
        d,
        "second residual add of the block",
    ));
    push_mlp(ops, cfg, d, layer, &mlp_in_q8, &block_out);
    (block_out, resid_b)
}

/// The gated-delta-net mixer.
fn push_gdn(ops: &mut Vec<Op>, d: &Dims, layer: u32, lp: &str, in_q8: &str, out_edge: &str) {
    let site = Site::Layer(layer);
    let la = format!("{lp}.linear_attn");
    let qkvz = format!("l{layer}.gdn.qkvz");
    let ba = format!("l{layer}.gdn.ba");
    let conv = format!("l{layer}.gdn.conv");
    let kkt = format!("l{layer}.gdn.kkt");
    let core = format!("l{layer}.gdn.core");
    let out_q8 = format!("l{layer}.gdn.out_q8");
    let partial = if d.tp > 1 {
        format!("l{layer}.gdn.partial")
    } else {
        out_edge.to_string()
    };

    let conv_state = StateRef {
        name: format!("l{layer}.gdn.conv_state"),
        layout: Layout::Bf16,
        // `conv_width - 1 + verify_width - 1`, not `conv_width - 1`. The cache is
        // `[slot][qkv_n][state_len]` and the kernels walk `state_len = conv_width - 1 + q - 1`, so a
        // slot sized for `q = 1` is overrun by the first verify step — measured 2026-09-20 as
        // `Memory Fault Error … saber_gdn_conv_update_w4_h128_hpb1`. It is sized for the GRAPH's
        // verify width rather than for a bucket's, because one allocation serves every bucket and
        // the layout has to be the same in all of them (`saber_exec::layer::gdn_shape`).
        bytes_per_slot: Some(
            u64::from(d.gdn_qkv_n)
                * u64::from(d.conv_width - 1 + d.verify_width.saturating_sub(1))
                * 2,
        ),
        // `cache_idx`, which a DECODE step binds to `sidx` at `ci_stride = q_len`: every sequence's
        // conv state is entry 0 of its block, i.e. the run's BASE, whatever the step accepted.
        indexed_by: StateIndexedBy::CacheIdx,
        note: "depthwise conv cache, rewritten in place; one row per qkv dim, the graph's widest                speculative window wide",
    };
    let rec_state = StateRef {
        name: format!("l{layer}.gdn.recurrent_state"),
        layout: Layout::Fp32State,
        bytes_per_slot: Some(
            u64::from(d.gdn_value_heads)
                * u64::from(d.key_head_dim)
                * u64::from(d.value_head_dim)
                * 4,
        ),
        // `sidx`, per ROW: row `t` writes `sidx[n*stride + t]` and the next step opens at
        // `sidx[n*stride + naccept - 1]`, so the LIVE snapshot is `base + naccept - 1`.
        indexed_by: StateIndexedBy::Sidx,
        note: "the delta-rule state: 1.5 MiB per layer per slot at TP=2, fp32. The total across \
               layers x slots is what the 64 MB Infinity-Cache cliff is measured against",
    };
    let gdn_geom = Geometry::GdnCore {
        conv_width: d.conv_width,
        head_k: d.key_head_dim,
        head_v: d.value_head_dim,
        heads: d.gdn_value_heads,
        key_heads: d.gdn_key_heads,
        chunk: d.gdn_chunk,
    };

    // in_proj_qkv and in_proj_z are one GEMM: SABER_GDN_MERGE_INPROJ=1 on the prod unit,
    // and 5120 + 3072 = 8192 is the `n8192` shape the decode band was measured on. The merge
    // is also why gdn_norm_quant needs `zs` as a key: it changes the z view's pitch.
    ops.push(Op {
        name: format!("layer.{layer}.gdn.in_proj_qkvz"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(in_q8, ACCEPTS_FP8_ACTIVATION)],
        outputs: vec![Edge::bf16(
            qkvz.clone(),
            "concat(q, k, v, z): 2048 + 3072 + 3072 per rank at TP=2",
        )],
        state: vec![],
        weights: vec![
            Binding::new(
                format!("{la}.in_proj_qkv.weight"),
                WeightRole::Weight,
                Dtype::F8E4M3,
                vec![u64::from(d.gdn_qkv_n), u64::from(d.hidden)],
                ShardKind::Dim0Segments,
            ),
            Binding::new(
                format!("{la}.in_proj_qkv.weight_scale"),
                WeightRole::WeightScale,
                Dtype::Bf16,
                vec![u64::from(d.gdn_qkv_n), 1],
                ShardKind::Dim0Segments,
            ),
            Binding::new(
                format!("{la}.in_proj_z.weight"),
                WeightRole::Weight,
                Dtype::F8E4M3,
                vec![u64::from(d.gdn_z_n), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
            Binding::new(
                format!("{la}.in_proj_z.weight_scale"),
                WeightRole::WeightScale,
                Dtype::Bf16,
                vec![u64::from(d.gdn_z_n), 1],
                ShardKind::Dim0,
            ),
        ],
        geometry: Geometry::Gemm {
            n: d.gdn_qkvz_n,
            k: d.hidden,
            w_dtype: "mxfp4",
            a_dtype: "fp8_e4m3",
        },
        note: "in_proj_qkv is Dim0Segments: dim 0 is concat(q, k, v) and a contiguous half-split \
               would be WRONG",
    });

    // a and b are the other half of the merge, and they are bf16: config.json's
    // quantization_config.ignore lists every layer's in_proj_a / in_proj_b, so the requant
    // never touched them. N = 48 is the `gate48` shape in the decode band.
    ops.push(Op {
        name: format!("layer.{layer}.gdn.in_proj_ba"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(in_q8, ACCEPTS_FP8_ACTIVATION)],
        outputs: vec![Edge::bf16(
            ba.clone(),
            "concat(b, a): one row each per value head",
        )],
        state: vec![],
        weights: vec![
            Binding::new(
                format!("{la}.in_proj_b.weight"),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.gdn_value_heads), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
            Binding::new(
                format!("{la}.in_proj_a.weight"),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.gdn_value_heads), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
        ],
        geometry: Geometry::Gemm {
            n: d.gdn_ba_n,
            k: d.hidden,
            w_dtype: "mxfp4",
            a_dtype: "fp8_e4m3",
        },
        note: "the pack holds these bf16 (quantization_config.ignore); prod requantizes them at \
               load, so the pack builder owes this shape a device-final layout id",
    });

    // ---- the core, three routes ------------------------------------------------------
    ops.push(Op {
        name: format!("layer.{layer}.gdn.fused_update"),
        site,
        kernel: KernelOp::GdnFusedUpdate,
        selector: Selector::DecodeGdnFused,
        inputs: vec![
            Input::new(qkvz.clone(), ACCEPTS_BF16),
            Input::new(ba.clone(), ACCEPTS_BF16),
        ],
        outputs: vec![Edge::bf16(
            core.clone(),
            "the delta-rule output, per value head",
        )],
        state: vec![conv_state.clone(), rec_state.clone()],
        weights: gdn_core_weights(&la, d),
        geometry: gdn_geom.clone(),
        note: "conv update + grid-wide barrier + recurrent update in ONE launch. At TP=2 the \
               value heads are 24, so nseq*H <= FUSED_MAX_ITEMS 32 holds only at nseq = 1",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.conv_update"),
        site,
        kernel: KernelOp::GdnConvUpdate,
        selector: Selector::DecodeGdnPair,
        inputs: vec![Input::new(qkvz.clone(), ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(conv.clone(), "convolved q/k/v")],
        state: vec![conv_state.clone()],
        weights: vec![gdn_conv_binding(&la, d)],
        geometry: gdn_geom.clone(),
        note: "the pair that serves nseq >= 2, which at TP=2 is MOST of prod's GDN work. \
               saber declares n_launches = 2 for the pair as a whole; the graph draws two ops \
               because the conv's q/k/v is a real edge the recurrent update consumes",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.recurrent_update"),
        site,
        kernel: KernelOp::GdnRecurrentUpdate,
        selector: Selector::DecodeGdnPair,
        inputs: vec![
            Input::new(conv.clone(), ACCEPTS_BF16),
            Input::new(ba.clone(), ACCEPTS_BF16),
        ],
        outputs: vec![Edge::bf16(
            core.clone(),
            "the delta-rule output, per value head",
        )],
        state: vec![rec_state.clone()],
        weights: gdn_rate_bindings(&la, d),
        geometry: gdn_geom.clone(),
        note: "second half of the decode pair",
    });
    // ---- P27-15d: the TREE verify's core (the b1 `DecodeTree` bucket only) ---------------
    ops.push(Op {
        name: format!("layer.{layer}.gdn.conv_tree"),
        site,
        kernel: KernelOp::GdnConvTree,
        selector: Selector::DecodeTree,
        inputs: vec![Input::new(qkvz.clone(), ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(
            conv.clone(),
            "convolved q/k/v, one row per tree node",
        )],
        state: vec![conv_state.clone()],
        weights: vec![gdn_conv_binding(&la, d)],
        geometry: gdn_geom.clone(),
        note: "P27-15d: the decode conv with each row's three older taps GATHERED along its \
               parent path (`tsrc`: 0..2 = the committed window, 3 + r = row r) from the window \
               named by cache entries (`hsrc`); the cache rewrite is the shipped one",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.tree_update"),
        site,
        kernel: KernelOp::GdnTreeUpdate,
        selector: Selector::DecodeTree,
        inputs: vec![
            Input::new(conv.clone(), ACCEPTS_BF16),
            Input::new(ba.clone(), ACCEPTS_BF16),
        ],
        outputs: vec![Edge::bf16(
            core.clone(),
            "the delta-rule output, per value head",
        )],
        state: vec![rec_state.clone()],
        weights: gdn_rate_bindings(&la, d),
        geometry: gdn_geom.clone(),
        note: "P27-15d: the WY-form tree update from the ONE committed f16 slot (`tslot`), the \
               previous tree's accepted row (`acc`) advanced as the prologue, the record for the \
               next left in the layer's tree record; then the shipped `gdn_norm_quant`",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.conv_prep"),
        site,
        kernel: KernelOp::GdnConvPrep,
        selector: Selector::Prefill,
        inputs: vec![Input::new(qkvz.clone(), ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(conv.clone(), "convolved q/k/v for the chunk")],
        state: vec![conv_state],
        weights: vec![gdn_conv_binding(&la, d)],
        geometry: gdn_geom.clone(),
        note: "prefill GDN triple, step 1 of 3",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.kkt_solve"),
        site,
        kernel: KernelOp::GdnKktSolve,
        selector: Selector::Prefill,
        inputs: vec![
            Input::new(conv.clone(), ACCEPTS_BF16),
            Input::new(ba.clone(), ACCEPTS_BF16),
        ],
        outputs: vec![Edge::bf16(kkt.clone(), "the chunked inverse factor")],
        state: vec![],
        weights: vec![],
        geometry: gdn_geom.clone(),
        note: "prefill GDN triple, step 2 of 3",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.chunk_scan"),
        site,
        kernel: KernelOp::GdnChunkScan,
        selector: Selector::Prefill,
        inputs: vec![
            Input::new(conv, ACCEPTS_BF16),
            Input::new(kkt, ACCEPTS_BF16),
            Input::new(ba, ACCEPTS_BF16),
        ],
        outputs: vec![Edge::bf16(
            core.clone(),
            "the delta-rule output, per value head",
        )],
        state: vec![rec_state],
        weights: gdn_rate_bindings(&la, d),
        geometry: gdn_geom,
        note: "prefill GDN triple, step 3 of 3; writes the recurrent state the decode path \
               continues from",
    });

    // ---- gated norm + out_proj + all-reduce -------------------------------------------
    ops.push(Op {
        name: format!("layer.{layer}.gdn.norm"),
        site,
        kernel: KernelOp::GdnNormQuant,
        selector: Selector::Always,
        inputs: vec![
            Input::new(core, ACCEPTS_BF16),
            // z is a SLICE of the merged in_proj output, which is why the op needs the row
            // stride as a selection key and not a constant.
            Input::new(qkvz, ACCEPTS_BF16),
        ],
        // ROW-MAJOR AT EVERY M, and this is the fact the 2026-09-21 prefill garbage was:
        // `gdn_norm_quant` has exactly ONE instantiation (`gdn_norm_quant_t{block}`,
        // `hip/mxfp4_quant.hip`), whose store is `q += row * N`. The reference has one too --
        // `saber_gdn_norm_quant` is templated on the thread count alone -- and prod's own law
        // says so from the other end: `ref_arnq.py` calls `a_tiled_register` after
        // `add_rms_quant` and after `silu_mul_quant` and NOT after `gdn_norm_quant`, so
        // `ref_mxfp4.a_tiled_take` hands `out_proj` the row-major family however large M is.
        // Declaring this edge `fused_q8` made `gdn.out_proj` take `mxfp4_gemm_atiled` at
        // M >= a_tiled_min_m() and read row-major bytes as WMMA fragments -- layer 0 diverging at a
        // 1024- or 2048-row prefill chunk while 256 and 512 were bit-identical (ws-p6).
        //
        // G2 (2026-09-23): Saber added the tiled store (`gdn_norm_quant_t256_tiled1`) behind the
        // mxfp4 rules' `gdn_norm_tiled` knob. The edge therefore asks the PRODUCER what it writes
        // (`AProducer::a_layout`, the same answer the rules give the kernel), never a constant:
        // with the knob off it is row-major at every M exactly as above.
        outputs: vec![Edge::new(
            out_q8.clone(),
            gdn_norm_layout_rule(),
            "per-head gated RMS norm output, quantized per token -- the layout its producer writes",
        )],
        state: vec![],
        weights: vec![Binding::new(
            format!("{la}.norm.weight"),
            WeightRole::NormWeight,
            Dtype::Bf16,
            vec![u64::from(d.value_head_dim)],
            ShardKind::Replicated,
        )],
        geometry: Geometry::GdnNorm {
            n: d.gdn_z_n,
            zs: d.gdn_qkvz_n,
            channels: d.value_head_dim,
        },
        note: "saber also has gdn_gated_rmsnorm, which prod BINDS and never calls: the decode \
               kernel folds the same arithmetic into its epilogue and the fused producer here \
               does the quant in the same launch",
    });
    ops.push(Op {
        name: format!("layer.{layer}.gdn.out_proj"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(out_q8, ACCEPTS_FP8_ACTIVATION)],
        outputs: vec![Edge::bf16(partial.clone(), "row-parallel partial sum")],
        state: vec![],
        weights: vec![
            Binding::new(
                format!("{la}.out_proj.weight"),
                WeightRole::Weight,
                Dtype::F8E4M3,
                vec![u64::from(d.hidden), u64::from(d.gdn_out_k)],
                ShardKind::Dim1,
            ),
            Binding::new(
                format!("{la}.out_proj.weight_scale"),
                WeightRole::WeightScale,
                Dtype::Bf16,
                vec![u64::from(d.hidden), 1],
                ShardKind::Replicated,
            ),
        ],
        geometry: Geometry::Gemm {
            n: d.hidden,
            k: d.gdn_out_k,
            w_dtype: "mxfp4",
            a_dtype: "fp8_e4m3",
        },
        note: "row-parallel: weight splits Dim1, the per-output-row scale stays replicated. Its A \
               side is `gdn.norm`'s ROW-MAJOR edge, so this is the one GEMM in a GDN layer whose \
               family does not follow M",
    });
    if d.tp > 1 {
        ops.push(all_reduce_op(
            &format!("layer.{layer}.gdn.all_reduce"),
            site,
            Selector::Always,
            &partial,
            Edge::bf16(out_edge.to_string(), "the GDN mixer output"),
            d,
            "one of two all-reduces per layer; 128 per forward at 64 layers",
        ));
    }
}

fn gdn_conv_binding(la: &str, d: &Dims) -> Binding {
    Binding::new(
        format!("{la}.conv1d.weight"),
        WeightRole::ConvWeight,
        Dtype::Bf16,
        vec![u64::from(d.gdn_qkv_n), 1, u64::from(d.conv_width)],
        ShardKind::Dim0Segments,
    )
}

fn gdn_rate_bindings(la: &str, d: &Dims) -> Vec<Binding> {
    vec![
        Binding::new(
            format!("{la}.A_log"),
            WeightRole::ALog,
            Dtype::Bf16,
            vec![u64::from(d.gdn_value_heads)],
            ShardKind::Dim0,
        ),
        Binding::new(
            format!("{la}.dt_bias"),
            WeightRole::DtBias,
            Dtype::Bf16,
            vec![u64::from(d.gdn_value_heads)],
            ShardKind::Dim0,
        ),
    ]
}

fn gdn_core_weights(la: &str, d: &Dims) -> Vec<Binding> {
    let mut out = vec![gdn_conv_binding(la, d)];
    out.extend(gdn_rate_bindings(la, d));
    out
}

/// The full-attention mixer.
fn push_attention(ops: &mut Vec<Op>, d: &Dims, layer: u32, lp: &str, in_q8: &str, out_edge: &str) {
    let site = Site::Layer(layer);
    let sa = format!("{lp}.self_attn");
    let qkv = format!("l{layer}.attn.qkv");
    let roped = format!("l{layer}.attn.roped");
    let core = format!("l{layer}.attn.core");
    let out_q8 = format!("l{layer}.attn.out_q8");
    let partial = if d.tp > 1 {
        format!("l{layer}.attn.partial")
    } else {
        out_edge.to_string()
    };

    let mut qkv_weights = Vec::new();
    for (proj, n) in [
        ("q_proj", d.q_out),
        ("k_proj", d.kv_out),
        ("v_proj", d.kv_out),
    ] {
        qkv_weights.push(Binding::new(
            format!("{sa}.{proj}.weight"),
            WeightRole::Weight,
            Dtype::F8E4M3,
            vec![u64::from(n), u64::from(d.hidden)],
            ShardKind::Dim0,
        ));
        qkv_weights.push(Binding::new(
            format!("{sa}.{proj}.weight_scale"),
            WeightRole::WeightScale,
            Dtype::Bf16,
            vec![u64::from(n), 1],
            ShardKind::Dim0,
        ));
    }
    ops.push(Op {
        name: format!("layer.{layer}.attn.qkv"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(in_q8, ACCEPTS_FP8_ACTIVATION)],
        outputs: vec![Edge::bf16(
            qkv.clone(),
            "concat(q+gate, k, v): 6144 + 512 + 512 per rank at TP=2",
        )],
        state: vec![],
        weights: qkv_weights,
        geometry: Geometry::Gemm {
            n: d.qkv_n,
            k: d.hidden,
            w_dtype: "mxfp4",
            a_dtype: "fp8_e4m3",
        },
        note: "q, k and v are ONE launch; N = 7168 is the `n7168` row the decode band was \
               measured on. attn_output_gate doubles q_proj's width",
    });
    ops.push(Op {
        name: format!("layer.{layer}.attn.qk_norm_rope"),
        site,
        kernel: KernelOp::QkNormRope,
        selector: Selector::Always,
        inputs: vec![Input::new(qkv.clone(), ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(roped.clone(), "q and k, normed and rotated")],
        state: vec![],
        weights: vec![
            Binding::new(
                format!("{sa}.q_norm.weight"),
                WeightRole::NormWeight,
                Dtype::Bf16,
                vec![u64::from(d.head_dim)],
                ShardKind::Replicated,
            ),
            Binding::new(
                format!("{sa}.k_norm.weight"),
                WeightRole::NormWeight,
                Dtype::Bf16,
                vec![u64::from(d.head_dim)],
                ShardKind::Replicated,
            ),
        ],
        geometry: Geometry::QkNorm {
            head_dim: d.head_dim,
            q_heads: d.q_heads,
            kv_heads: d.kv_heads,
            rotary_dim: d.rotary_dim,
        },
        note: "interleaved mrope over 64 of 256 dims (partial_rotary_factor 0.25, mrope_section \
               [11, 11, 10]); ONE launch in the attn_glue plugin against prod's five",
    });

    let kv_cache = StateRef {
        name: format!("l{layer}.attn.kv_cache"),
        layout: Layout::RowMajorFp8,
        bytes_per_slot: None,
        indexed_by: StateIndexedBy::BlockTable,
        note: "paged fp8 KV, block size 16, 16 KiB per token per rank across all 16 layers",
    };
    let attn_geom = Geometry::Attn {
        head_dim: d.head_dim,
        gqa: d.gqa,
        block_size: 16,
        q_heads: d.q_heads,
        kv_heads: d.kv_heads,
        kv_dtype: d.kv_dtype,
    };
    let kv_scales = vec![
        Binding::new(
            format!("{sa}.k_scale"),
            WeightRole::KvScale,
            Dtype::Bf16,
            vec![1],
            ShardKind::Replicated,
        ),
        Binding::new(
            format!("{sa}.v_scale"),
            WeightRole::KvScale,
            Dtype::Bf16,
            vec![1],
            ShardKind::Replicated,
        ),
    ];
    // The paged K/V write. The cache is STATE, and until this op existed nothing in the graph
    // wrote it: the two attention ops listed it and read it, and a forward pass that ran them
    // read whatever the allocator handed back. prod runs the same write as its own launch
    // (`TritonAttentionImpl.do_kv_cache_update`, which `SABERAttentionImpl` inherits), before the
    // attention kernel and after the rope.
    ops.push(Op {
        name: format!("layer.{layer}.attn.kv_append"),
        site,
        kernel: KernelOp::KvAppend,
        selector: Selector::Always,
        // The roped K comes from `qk_norm_rope`; V is read in place out of the projection, which
        // is why both edges are inputs and neither is copied first.
        inputs: vec![
            Input::new(roped.clone(), ACCEPTS_BF16),
            Input::new(qkv.clone(), ACCEPTS_BF16),
        ],
        outputs: vec![],
        state: vec![kv_cache.clone()],
        weights: kv_scales.clone(),
        geometry: attn_geom.clone(),
        note: "ONE launch writes both K and V, as triton_reshape_and_cache_flash does: the V                base is the K base plus head_size elements of the same paged tensor",
    });
    ops.push(Op {
        name: format!("layer.{layer}.attn.prefill"),
        site,
        kernel: KernelOp::AttnPrefillPaged,
        selector: Selector::Prefill,
        inputs: vec![Input::new(roped.clone(), ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(core.clone(), "attention output, per query head")],
        state: vec![kv_cache.clone()],
        weights: kv_scales.clone(),
        geometry: attn_geom.clone(),
        note: "one launch per run of equal q_len; BQ = 64 query rows per block, grid \
               (ceil(q_len/64), kv_heads, num_seqs). It READS the paged cache; \
               `attn.kv_append` is what wrote this chunk's rows into it",
    });
    ops.push(Op {
        name: format!("layer.{layer}.attn.decode"),
        site,
        kernel: KernelOp::AttnDecodePaged,
        selector: Selector::DecodeChain,
        inputs: vec![Input::new(roped.clone(), ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(core.clone(), "attention output, per query head")],
        state: vec![kv_cache.clone()],
        weights: kv_scales.clone(),
        geometry: attn_geom.clone(),
        note: "TWO launches: the split-KV kernel and the combine. 64 splits at batch 1 and 16 \
               at batch 8, at both 4k and 32k KV. It READS the paged cache; `attn.kv_append` \
               is what wrote this step's rows into it",
    });
    ops.push(Op {
        name: format!("layer.{layer}.attn.tree"),
        site,
        kernel: KernelOp::AttnDecodeTree,
        selector: Selector::DecodeTree,
        inputs: vec![Input::new(roped, ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(core.clone(), "attention output, per query head")],
        state: vec![kv_cache],
        weights: kv_scales,
        geometry: attn_geom,
        note: "P27-15d: TWO launches -- `p27_attn_decode_anc_*` (the ragged decode body, the \
               q_len tail masked by each row's ancestor word `anc`) and the ragged combine. Every \
               row's K/V sits at its ROW slot (rope by depth): siblings never see each other",
    });

    // This edge used to be declared row-major at every M, on the grounds that there was no fused
    // gate+quant producer on this tier. There is one now (`attn_gate_quant_t256_tiled{0,1}`), and
    // `rules/attn_glue_gfx1201.toml` keys it on the same band edge as `add_rms_quant`. The edge
    // that IS row-major at every M is `gdn.norm`'s, above -- which is the edge P0 follow-up (a)
    // is really about, and the one a consumer resolving against M read as fragment-tiled garbage.
    ops.push(Op {
        name: format!("layer.{layer}.attn.gate_quant"),
        site,
        kernel: KernelOp::AttnGateQuant,
        selector: Selector::Always,
        inputs: vec![
            Input::new(core, ACCEPTS_BF16),
            Input::new(qkv, ACCEPTS_BF16),
        ],
        // A FUSED producer, at the same band edge as `add_rms_quant`: `attn_gate_quant` carries
        // `tiled0`/`tiled1` instantiations (`hip/attn_glue.hip`) and
        // `rules/attn_glue_gfx1201.toml` keys them on its own `a_tiled_min_m`, pinned to
        // the ONE band edge (`saber_kernels::rules::a_tiled_min_m`). This edge was declared
        // row-major before the glue
        // plugin grew the tiled arm and the declaration was never corrected; nothing read it,
        // because the planner resolved every GEMM against M instead of against its A edge.
        outputs: vec![Edge::fused_q8(
            out_q8.clone(),
            "gated attention output, quantized per token",
        )],
        state: vec![],
        weights: vec![],
        geometry: Geometry::GateQuant {
            n: d.o_k,
            head_dim: d.head_dim,
            gate_stride: d.qkv_n,
        },
        note: "the fused gate+quant producer: it writes the fragment-tiled A layout at \
               M >= a_tiled_min_m(), so o_proj follows it into the A-tiled family",
    });
    ops.push(Op {
        name: format!("layer.{layer}.attn.o_proj"),
        site,
        kernel: KernelOp::GemmW4A8,
        selector: Selector::Always,
        inputs: vec![Input::new(out_q8, ACCEPTS_FP8_ACTIVATION)],
        outputs: vec![Edge::bf16(partial.clone(), "row-parallel partial sum")],
        state: vec![],
        weights: vec![
            Binding::new(
                format!("{sa}.o_proj.weight"),
                WeightRole::Weight,
                Dtype::F8E4M3,
                vec![u64::from(d.hidden), u64::from(d.o_k)],
                ShardKind::Dim1,
            ),
            Binding::new(
                format!("{sa}.o_proj.weight_scale"),
                WeightRole::WeightScale,
                Dtype::Bf16,
                vec![u64::from(d.hidden), 1],
                ShardKind::Replicated,
            ),
        ],
        geometry: Geometry::Gemm {
            n: d.hidden,
            k: d.o_k,
            w_dtype: "mxfp4",
            a_dtype: "fp8_e4m3",
        },
        note: "its input edge, not M, decides the GEMM family here -- which for this edge is the \
               same answer, because its producer is fused",
    });
    if d.tp > 1 {
        ops.push(all_reduce_op(
            &format!("layer.{layer}.attn.all_reduce"),
            site,
            Selector::Always,
            &partial,
            Edge::bf16(out_edge.to_string(), "the attention mixer output"),
            d,
            "row-parallel reduce",
        ));
    }
}

/// The MTP / DFlash2 drafter head: one full-attention decoder layer plus the `fc` that folds
/// the embedding of the next token into the target's hidden state.
///
/// Everything here is **bf16** in this tier: `quantization_config.ignore` ends with
/// `re:^mtp.*`, so the requant never touched the head. There is no committed bf16 GEMM
/// (the reference's `gemm_bf16_nt_m16` has no caller and was not ported), so every linear below is
/// [`KernelOp::GemmBf16`] with no declared launch count.
///
/// The drafter also has **no `lm_head` of its own**: `mtp_use_dedicated_embeddings` is false
/// and there is no `mtp.lm_head.*` record, so it shares the target's embedding and head. That
/// sharing is exactly why the tier binds them together and why the planner refuses to bind a
/// drafter whose tier id does not match the target's: the same DFlash2 drafter accepts
/// 3.67 tok/draft on MXFP4 and 2.72 on int4.
fn drafter_ops(cfg: &ModelConfig, d: &Dims) -> Vec<Op> {
    let mut ops = Vec::new();
    let site = Site::Drafter;
    let norm = |record: &str, width: u32| {
        Binding::new(
            record.to_string(),
            WeightRole::NormWeight,
            Dtype::Bf16,
            vec![u64::from(width)],
            ShardKind::Replicated,
        )
    };

    ops.push(Op {
        name: "mtp.pre_fc_norm".into(),
        site,
        kernel: KernelOp::AddRmsQuant,
        selector: Selector::Decode,
        inputs: vec![Input::new("hidden.0", ACCEPTS_BF16)],
        outputs: vec![
            Edge::bf16("mtp.fc_in", "concat(norm(embedding), norm(hidden))"),
            Edge::bf16(
                "mtp.fc_in.resid",
                "unused second output, kept for op symmetry",
            ),
        ],
        state: vec![],
        weights: vec![
            norm("mtp.pre_fc_norm_embedding.weight", cfg.hidden_size),
            norm("mtp.pre_fc_norm_hidden.weight", cfg.hidden_size),
        ],
        geometry: Geometry::Rms { k: d.hidden },
        note: "two RMS norms, one over the next token's embedding and one over the target's \
               hidden state, concatenated for fc",
    });
    ops.push(Op {
        name: "mtp.fc".into(),
        site,
        kernel: KernelOp::GemmBf16,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.fc_in", ACCEPTS_BF16)],
        outputs: vec![Edge::bf16("mtp.hidden", "the drafter layer's input")],
        state: vec![],
        weights: vec![Binding::new(
            "mtp.fc.weight".into(),
            WeightRole::Weight,
            Dtype::Bf16,
            vec![u64::from(d.hidden), u64::from(2 * d.hidden)],
            ShardKind::Replicated,
        )],
        geometry: Geometry::Gemm {
            n: d.hidden,
            k: 2 * d.hidden,
            w_dtype: "bf16",
            a_dtype: "bf16",
        },
        note: "REPLICATED, not sharded: every rank holds the full [5120, 10240] and computes \
               the same result from the already-reduced hidden state",
    });
    ops.push(Op {
        name: "mtp.layer.0.input_layernorm".into(),
        site,
        kernel: KernelOp::AddRmsQuant,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.hidden", ACCEPTS_BF16)],
        outputs: vec![
            Edge::bf16("mtp.l0.in", "normed drafter hidden"),
            Edge::bf16("mtp.l0.resid.a", "drafter residual"),
        ],
        state: vec![],
        weights: vec![norm("mtp.layers.0.input_layernorm.weight", cfg.hidden_size)],
        geometry: Geometry::Rms { k: d.hidden },
        note: "bf16 in, bf16 out: nothing quantizes on the drafter path in this tier",
    });
    ops.push(Op {
        name: "mtp.layer.0.attn.qkv".into(),
        site,
        kernel: KernelOp::GemmBf16,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.l0.in", ACCEPTS_BF16)],
        outputs: vec![Edge::bf16("mtp.l0.qkv", "concat(q+gate, k, v)")],
        state: vec![],
        weights: vec![
            Binding::new(
                "mtp.layers.0.self_attn.q_proj.weight".into(),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.q_out), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
            Binding::new(
                "mtp.layers.0.self_attn.k_proj.weight".into(),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.kv_out), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
            Binding::new(
                "mtp.layers.0.self_attn.v_proj.weight".into(),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.kv_out), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
            Binding::new(
                "mtp.layers.0.self_attn.q_norm.weight".into(),
                WeightRole::NormWeight,
                Dtype::Bf16,
                vec![u64::from(d.head_dim)],
                ShardKind::Replicated,
            ),
            Binding::new(
                "mtp.layers.0.self_attn.k_norm.weight".into(),
                WeightRole::NormWeight,
                Dtype::Bf16,
                vec![u64::from(d.head_dim)],
                ShardKind::Replicated,
            ),
        ],
        geometry: Geometry::Gemm {
            n: d.qkv_n,
            k: d.hidden,
            w_dtype: "bf16",
            a_dtype: "bf16",
        },
        note: "the drafter's q/k norms ride with the qkv GEMM here because the drafter has no \
               ported kernel to split them across",
    });
    ops.push(Op {
        name: "mtp.layer.0.attn.decode".into(),
        site,
        kernel: KernelOp::AttnDecodePaged,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.l0.qkv", ACCEPTS_BF16)],
        outputs: vec![Edge::bf16("mtp.l0.attn", "drafter attention output")],
        state: vec![StateRef {
            name: "mtp.l0.kv_cache".into(),
            layout: Layout::RowMajorFp8,
            bytes_per_slot: None,
            indexed_by: StateIndexedBy::BlockTable,
            note: "the drafter keeps its own KV pages",
        }],
        weights: vec![],
        geometry: Geometry::Attn {
            head_dim: d.head_dim,
            gqa: d.gqa,
            block_size: 16,
            q_heads: d.q_heads,
            kv_heads: d.kv_heads,
            kv_dtype: d.kv_dtype,
        },
        note: "same paged-attention op as the target's layers",
    });
    ops.push(Op {
        name: "mtp.layer.0.attn.o_proj".into(),
        site,
        kernel: KernelOp::GemmBf16,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.l0.attn", ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(
            if d.tp > 1 {
                "mtp.l0.attn.partial"
            } else {
                "mtp.l0.attn.out"
            },
            "row-parallel partial sum",
        )],
        state: vec![],
        weights: vec![Binding::new(
            "mtp.layers.0.self_attn.o_proj.weight".into(),
            WeightRole::Weight,
            Dtype::Bf16,
            vec![u64::from(d.hidden), u64::from(d.o_k)],
            ShardKind::Dim1,
        )],
        geometry: Geometry::Gemm {
            n: d.hidden,
            k: d.o_k,
            w_dtype: "bf16",
            a_dtype: "bf16",
        },
        note: "row-parallel, so it needs its own all-reduce",
    });
    if d.tp > 1 {
        ops.push(all_reduce_op(
            "mtp.layer.0.attn.all_reduce",
            site,
            Selector::Decode,
            "mtp.l0.attn.partial",
            Edge::bf16("mtp.l0.attn.out", "drafter attention output"),
            d,
            "the drafter is sharded too: two more all-reduces per drafter step",
        ));
    }
    ops.push(Op {
        name: "mtp.layer.0.post_attention_layernorm".into(),
        site,
        kernel: KernelOp::AddRmsQuant,
        selector: Selector::Decode,
        inputs: vec![
            Input::new("mtp.l0.attn.out", ACCEPTS_BF16),
            Input::new("mtp.l0.resid.a", ACCEPTS_BF16),
        ],
        outputs: vec![
            Edge::bf16("mtp.l0.mlp.in", "normed drafter hidden"),
            Edge::bf16("mtp.l0.resid.b", "drafter residual"),
        ],
        state: vec![],
        weights: vec![norm(
            "mtp.layers.0.post_attention_layernorm.weight",
            cfg.hidden_size,
        )],
        geometry: Geometry::Rms { k: d.hidden },
        note: "second residual add of the drafter block",
    });
    ops.push(Op {
        name: "mtp.layer.0.mlp.gate_up".into(),
        site,
        kernel: KernelOp::GemmBf16,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.l0.mlp.in", ACCEPTS_BF16)],
        outputs: vec![Edge::bf16("mtp.l0.mlp.gate_up", "concat(gate, up)")],
        state: vec![],
        weights: vec![
            Binding::new(
                "mtp.layers.0.mlp.gate_proj.weight".into(),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.inter), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
            Binding::new(
                "mtp.layers.0.mlp.up_proj.weight".into(),
                WeightRole::Weight,
                Dtype::Bf16,
                vec![u64::from(d.inter), u64::from(d.hidden)],
                ShardKind::Dim0,
            ),
        ],
        geometry: Geometry::Gemm {
            n: d.gate_up_n,
            k: d.hidden,
            w_dtype: "bf16",
            a_dtype: "bf16",
        },
        note: "merged, as on the target path",
    });
    ops.push(Op {
        name: "mtp.layer.0.mlp.down_proj".into(),
        site,
        kernel: KernelOp::GemmBf16,
        selector: Selector::Decode,
        inputs: vec![Input::new("mtp.l0.mlp.gate_up", ACCEPTS_BF16)],
        outputs: vec![Edge::bf16(
            if d.tp > 1 {
                "mtp.l0.mlp.partial"
            } else {
                "mtp.l0.out"
            },
            "row-parallel partial sum",
        )],
        state: vec![],
        weights: vec![Binding::new(
            "mtp.layers.0.mlp.down_proj.weight".into(),
            WeightRole::Weight,
            Dtype::Bf16,
            vec![u64::from(d.hidden), u64::from(d.inter)],
            ShardKind::Dim1,
        )],
        geometry: Geometry::Gemm {
            n: d.hidden,
            k: d.inter,
            w_dtype: "bf16",
            a_dtype: "bf16",
        },
        note: "the silu+mul between gate_up and down has no bf16 producer either",
    });
    if d.tp > 1 {
        ops.push(all_reduce_op(
            "mtp.layer.0.mlp.all_reduce",
            site,
            Selector::Decode,
            "mtp.l0.mlp.partial",
            Edge::bf16("mtp.l0.out", "the drafter block output"),
            d,
            "second drafter all-reduce",
        ));
    }
    ops.push(Op {
        name: "mtp.norm".into(),
        site,
        kernel: KernelOp::AddRmsQuant,
        selector: Selector::Decode,
        inputs: vec![
            Input::new("mtp.l0.out", ACCEPTS_BF16),
            Input::new("mtp.l0.resid.b", ACCEPTS_BF16),
        ],
        outputs: vec![
            Edge::bf16("mtp.out", "drafter hidden, ready for the SHARED lm_head"),
            Edge::bf16("mtp.out.resid", "unused"),
        ],
        state: vec![],
        weights: vec![norm("mtp.norm.weight", cfg.hidden_size)],
        geometry: Geometry::Rms { k: d.hidden },
        note: "the drafter has no lm_head record of its own; it reuses the target's",
    });
    ops
}

// =====================================================================================
// Tests
// =====================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// The fp8-weight head (N4) is the bf16 head's norm and input with a `GemmFp8Head`, and its
    /// records are the per-channel fp8 source carried raw: `lm_head.w8` (layout 15, e4m3) and
    /// `lm_head.ws` (layout 16, f32) in place of the MXFP4 trio, every other record unchanged.
    /// Control: a graph without a `model.norm` is refused.
    #[test]
    fn the_fp8_head_rewrites_the_two_head_ops_and_carries_the_fp8_source_raw() {
        let base = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8).expect("graph");
        let f8 = base.clone().with_fp8_head().expect("fp8 head");
        let bf = base.clone().with_bf16_head().expect("bf16 head");
        let op = |g: &Graph, n: &str| g.ops.iter().find(|o| o.name == n).cloned().expect("op");
        assert_eq!(op(&f8, "lm_head").kernel, KernelOp::GemmFp8Head);
        // The norm and the head's input are the bf16 head's, exactly.
        assert_eq!(op(&f8, "model.norm"), op(&bf, "model.norm"));
        assert_eq!(op(&f8, "lm_head").inputs, op(&bf, "lm_head").inputs);
        assert_eq!(f8.ops.len(), base.ops.len());
        let recs = |g: &Graph| {
            g.device_final_records()
                .expect("records")
                .into_iter()
                .map(|r| (r.name.clone(), (r.layout_id, r.dtype, r.shape.clone())))
                .collect::<BTreeMap<_, _>>()
        };
        let (a, b) = (recs(&base), recs(&f8));
        assert_eq!(
            b["lm_head.w8"],
            (LAYOUT_FP8_CHANNEL, Dtype::F8E4M3, vec![124_160, 5120])
        );
        assert_eq!(
            b["lm_head.ws"],
            (LAYOUT_F32_CHANNEL_SCALE, Dtype::F32, vec![124_160])
        );
        assert!(!b.contains_key("lm_head.w") && !b.contains_key("lm_head.wref"));
        for (name, v) in &a {
            if !name.starts_with("lm_head.") {
                assert_eq!(b.get(name), Some(v), "{name} moved");
            }
        }
        // MXFP4 binds w/ws/wref; the fp8 head w8/ws.
        assert_eq!(a.len(), b.len() + 1);
        // Both of its records come from the ONE per-channel fp8 source and its scale.
        let r8 = f8.device_final_records().expect("records");
        for n in ["lm_head.w8", "lm_head.ws"] {
            let r = r8.iter().find(|r| r.name == n).expect("record");
            assert_eq!(r.sources.len(), 1, "{n}");
            assert_eq!(r.sources[0].weight, "lm_head.weight", "{n}");
            assert_eq!(
                r.sources[0].scale.as_deref(),
                Some("lm_head.weight_scale"),
                "{n}"
            );
        }
        let mut bare = base.clone();
        bare.ops.retain(|o| o.name != "model.norm");
        assert!(bare.with_fp8_head().is_err());
    }

    /// The bf16 head (N2) rewrites exactly the two head ops: the norm writes a bf16 `head.in`, the
    /// head reads it, the record set loses the MXFP4 trio for ONE layout-14 bf16 record of the
    /// same rows, and every other record is unchanged. Control: the default graph's head is the
    /// W4A8 trio, and a graph without a `model.norm` is refused.
    #[test]
    fn the_bf16_head_rewrites_the_two_head_ops_and_one_record() {
        let base = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8).expect("graph");
        let bf = base.clone().with_bf16_head().expect("bf16 head");
        let op = |g: &Graph, n: &str| g.ops.iter().find(|o| o.name == n).cloned().expect("op");
        assert_eq!(op(&base, "lm_head").kernel, KernelOp::GemmW4A8);
        assert_eq!(op(&bf, "lm_head").kernel, KernelOp::GemmBf16Head);
        assert_eq!(op(&bf, "model.norm").kernel, KernelOp::AddRmsBf16);
        assert_eq!(op(&bf, "model.norm").outputs[0].name, "head.in");
        assert_eq!(op(&bf, "lm_head").inputs[0].edge, "head.in");
        assert_eq!(bf.ops.len(), base.ops.len());
        let recs = |g: &Graph| {
            g.device_final_records()
                .expect("records")
                .into_iter()
                .map(|r| (r.name.clone(), (r.layout_id, r.shape.clone())))
                .collect::<BTreeMap<_, _>>()
        };
        let (a, b) = (recs(&base), recs(&bf));
        assert_eq!(a["lm_head.w"].0, LAYOUT_MXFP4_WPERM);
        assert_eq!(
            b["lm_head.w"],
            (LAYOUT_BF16_FROM_FP8_CHANNEL, vec![124_160, 5120])
        );
        assert!(!b.contains_key("lm_head.ws") && !b.contains_key("lm_head.wref"));
        for (name, v) in &a {
            if !name.starts_with("lm_head.") {
                assert_eq!(b.get(name), Some(v), "{name} moved");
            }
        }
        assert_eq!(a.len(), b.len() + 2);
        let mut bare = base.clone();
        bare.ops.retain(|o| o.name != "model.norm");
        assert!(bare.with_bf16_head().is_err());
    }

    /// The real TP=2 pack built by `the internal reference harness` on 2026-09-18.
    const PACK_MANIFEST: &str = "/mnt/nvme/saber-packs/qwen38-27b-nvfp4/manifest.json";
    const CHECKPOINT_CONFIG: &str = "/mnt/nvme/models/Qwen3.8-27B-NVFP4/config.json";

    #[derive(Debug, serde::Deserialize)]
    struct ManifestRecord {
        name: String,
        dtype: String,
        shape: Vec<u64>,
        shard_kind: String,
        layout_id: u32,
    }

    #[derive(Debug, serde::Deserialize)]
    struct ManifestRank {
        rank: u32,
        records: Vec<ManifestRecord>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct Manifest {
        tp_size: u32,
        ranks: Vec<ManifestRank>,
    }

    /// `None` means "pack absent" — the caller prints the loud skip and returns.
    fn load_manifest() -> Option<Manifest> {
        let path = Path::new(PACK_MANIFEST);
        if !path.exists() {
            eprintln!(
                "\n=====================================================================\n\
                 SKIPPING THE GRAPH/PACK BINDING GATE: {PACK_MANIFEST} is absent.\n\
                 This gate is what proves every weight name in saber-model::graph is a name\n\
                 the tier pack actually holds, at the right rank-local shape and dtype.\n\
                 Rebuild the pack (docs/abi/PACK.md, \"Building a pack\") to run it.\n\
                 =====================================================================\n"
            );
            return None;
        }
        let text = std::fs::read_to_string(path).expect("read manifest");
        Some(serde_json::from_str(&text).expect("parse manifest"))
    }

    fn graph_tp2() -> Graph {
        Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("build graph")
    }

    // ---------------------------------------------------------------------------------
    // Geometry
    // ---------------------------------------------------------------------------------

    #[test]
    fn layer_kinds_match_full_attention_interval() {
        let cfg = ModelConfig::qwen38_27b();
        assert_eq!(cfg.attention_layers(), 16);
        assert_eq!(cfg.gdn_layers(), 48);
        let attn: Vec<u32> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.layer_kind(i) == LayerKind::Attention)
            .collect();
        assert_eq!(
            attn,
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
        );
        assert_eq!(cfg.gqa(), 6);
    }

    #[test]
    fn tp3_is_refused_rather_than_invented() {
        let err = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 3).unwrap_err();
        let text = format!("{err}");
        assert!(text.contains("num_key_value_heads=4"), "{text}");
        assert!(text.contains("padded head layout"), "{text}");
    }

    #[test]
    fn merged_shapes_are_the_ones_the_tuned_bands_name() {
        let g = graph_tp2();
        let by_name: BTreeMap<&str, &Op> = g.ops.iter().map(|o| (o.name.as_str(), o)).collect();
        let n_of = |name: &str| match by_name[name].geometry {
            Geometry::Gemm { n, k, .. } => (n, k),
            _ => panic!("{name} is not a GEMM"),
        };
        // These four numbers appear by name in crates/saber-kernels/rules/mxfp4_gfx1201.toml's
        // measured decode band: n8192, n7168, gate_up, gate48, and `down` at 68 k-slabs.
        assert_eq!(n_of("layer.0.gdn.in_proj_qkvz"), (8192, 5120));
        assert_eq!(n_of("layer.3.attn.qkv"), (7168, 5120));
        assert_eq!(n_of("layer.0.mlp.gate_up"), (17408, 5120));
        assert_eq!(n_of("layer.0.gdn.in_proj_ba"), (48, 5120));
        assert_eq!(n_of("layer.0.mlp.down_proj"), (5120, 8704));
        assert_eq!(n_of("layer.3.attn.o_proj"), (5120, 3072));
        assert_eq!(n_of("layer.0.gdn.out_proj"), (5120, 3072));
        assert_eq!(n_of("lm_head"), (124_160, 5120));
    }

    #[test]
    fn gdn_norm_takes_the_merged_row_stride_as_a_key() {
        let g = graph_tp2();
        let op = g.ops.iter().find(|o| o.name == "layer.0.gdn.norm").unwrap();
        let keys = op.shape_keys(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        assert_eq!(keys.num("n"), Some(3072));
        // 8192, not 3072: the merged in_proj is what makes this a key rather than a constant.
        assert_eq!(keys.num("zs"), Some(8192));
        assert_eq!(keys.num("channels"), Some(128));
        assert_eq!(
            keys.num("zs").unwrap() % 8,
            0,
            "z row stride must be 8-aligned"
        );
        assert_eq!(keys.num("n").unwrap() % 128, 0, "gdn_n_multiple is 128");
    }

    // ---------------------------------------------------------------------------------
    // The per-edge layout property
    // ---------------------------------------------------------------------------------

    /// The committed mxfp4 table's G2 knob: does `gdn_norm_quant` write the tiled layout?
    fn gdn_norm_tiled() -> bool {
        LaunchRules::prod_mxfp4()
            .expect("committed rules")
            .knobs
            .gdn_norm_tiled
    }

    #[test]
    fn fused_producers_switch_to_a_tiled_and_the_gdn_norm_follows_its_knob() {
        let g = graph_tp2();
        let tiled_gdn = gdn_norm_tiled();
        for (m, phase, fused_expect) in [
            (5u32, Phase::Decode, Layout::RowMajorFp8),
            (2048, Phase::Prefill, Layout::ATiledFp8),
        ] {
            let ops = g.ops_for_bucket(m, phase);
            let gate = ops
                .iter()
                .find(|r| r.name() == "layer.3.attn.gate_quant")
                .unwrap();
            let norm = ops
                .iter()
                .find(|r| r.name() == "layer.3.input_layernorm")
                .unwrap();
            assert_eq!(norm.output_layouts[0], fused_expect, "M={m}");
            // `attn_gate_quant` IS fused -- it carries `tiled0`/`tiled1` at the same band edge.
            assert_eq!(gate.output_layouts[0], fused_expect, "M={m}");
            // `gdn_norm_quant` has one instantiation in the reference: row-major at BOTH M, and
            // declaring otherwise is the 2026-09-21 prefill garbage (ws-p6). G2's tiled store
            // makes it switch like the others, and the edge follows the PRODUCER's answer.
            let gdn_norm = ops.iter().find(|r| r.name() == "layer.0.gdn.norm").unwrap();
            let gdn_expect = if tiled_gdn {
                fused_expect
            } else {
                Layout::RowMajorFp8
            };
            assert_eq!(gdn_norm.output_layouts[0], gdn_expect, "M={m}");

            let o_proj = ops
                .iter()
                .find(|r| r.name() == "layer.3.attn.o_proj")
                .unwrap();
            let gdn_out = ops
                .iter()
                .find(|r| r.name() == "layer.0.gdn.out_proj")
                .unwrap();
            let qkv = ops.iter().find(|r| r.name() == "layer.3.attn.qkv").unwrap();
            let follows_m = Some(if m >= a_tiled_min_m() {
                GemmFamily::ATiled
            } else {
                GemmFamily::RowMajor
            });
            assert_eq!(o_proj.gemm_family, follows_m, "M={m}");
            assert_eq!(qkv.gemm_family, follows_m, "M={m}");
            assert_eq!(
                gdn_out.gemm_family,
                if tiled_gdn {
                    follows_m
                } else {
                    Some(GemmFamily::RowMajor)
                },
                "the GDN out_proj follows its edge, not M (M={m})"
            );
        }
    }

    #[test]
    fn every_edge_is_produced_and_every_consumer_can_read_it() {
        let g = graph_tp2();
        for bucket in [
            Bucket::Decode {
                batch: 1,
                verify_width: 5,
            },
            Bucket::Decode {
                batch: 8,
                verify_width: 5,
            },
            Bucket::Decode {
                batch: 1,
                verify_width: 1,
            },
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            Bucket::Prefill {
                tokens: 8192,
                seqs: 1,
            },
        ] {
            if let Err(faults) = g.check_layouts(bucket) {
                let text: Vec<String> = faults.iter().map(ToString::to_string).collect();
                panic!("{bucket:?}:\n  {}", text.join("\n  "));
            }
        }
    }

    /// The negative control structural rule 8 asks for: a checker that cannot fail proves
    /// nothing. Narrow one consumer to the A-tiled layout only and the row-major edge that
    /// feeds it must be reported, by name, at a decode M.
    #[test]
    fn negative_control_the_layout_checker_reports_a_mismatch() {
        let mut g = graph_tp2();
        const ONLY_A_TILED: &[Layout] = &[Layout::ATiledFp8];
        let op = g
            .ops
            .iter_mut()
            .find(|o| o.name == "layer.3.attn.o_proj")
            .unwrap();
        op.inputs[0].accepts = ONLY_A_TILED;

        let faults = g
            .check_layouts(Bucket::Decode {
                batch: 1,
                verify_width: 5,
            })
            .expect_err("the checker must report the mismatch it was given");
        assert_eq!(faults.len(), 1, "{faults:?}");
        match &faults[0] {
            LayoutFault::Mismatch {
                op,
                edge,
                produced_by,
                found,
                ..
            } => {
                assert_eq!(op, "layer.3.attn.o_proj");
                assert_eq!(edge, "l3.attn.out_q8");
                assert_eq!(produced_by, "layer.3.attn.gate_quant");
                assert_eq!(*found, Layout::RowMajorFp8);
            }
            other => panic!("wrong fault: {other:?}"),
        }
    }

    // ---------------------------------------------------------------------------------
    // Routing and launch budget
    // ---------------------------------------------------------------------------------

    #[test]
    fn gdn_routes_fused_at_batch_one_and_the_pair_above_it() {
        let g = graph_tp2();
        assert_eq!(g.gdn_value_heads_per_rank(), 24);

        let one = g.ops_for(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        let fused = one
            .iter()
            .filter(|r| r.op.kernel == KernelOp::GdnFusedUpdate)
            .count();
        assert_eq!(
            fused, 48,
            "batch 1: 24 items <= 32, so every GDN layer fuses"
        );
        assert_eq!(
            one.iter()
                .filter(|r| r.op.kernel == KernelOp::GdnConvUpdate)
                .count(),
            0
        );

        let two = g.ops_for(Bucket::Decode {
            batch: 2,
            verify_width: 5,
        });
        assert_eq!(
            two.iter()
                .filter(|r| r.op.kernel == KernelOp::GdnFusedUpdate)
                .count(),
            0,
            "batch 2: 48 items > 32, so the pair runs"
        );
        assert_eq!(
            two.iter()
                .filter(|r| r.op.kernel == KernelOp::GdnConvUpdate)
                .count(),
            48
        );
    }

    #[test]
    fn all_reduce_route_follows_the_message_size() {
        let g = graph_tp2();
        // batch 1 x spec 5 = 5 rows x 5120 x 2 B = 50 KiB, under the 128 KiB threshold.
        let small = g.ops_for(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        let ar = small
            .iter()
            .find(|r| r.op.kernel == KernelOp::AllReduce)
            .unwrap();
        assert_eq!(ar.shape_keys.num("exact"), Some(1));
        assert_eq!(ar.entry.as_deref(), Some("ar_oneshot_2rank_exact"));
        assert_eq!(ar.plugin, Some("saber_ar_exact"));
        assert!(ar.ported, "saber part 3 ported the exact all-reduce");
        assert_eq!(ar.refusal, None);

        // batch 8 x spec 5 = 40 rows = 400 KiB, over the threshold.
        let big = g.ops_for(Bucket::Decode {
            batch: 8,
            verify_width: 5,
        });
        let ar = big
            .iter()
            .find(|r| r.op.kernel == KernelOp::AllReduce)
            .unwrap();
        assert_eq!(ar.shape_keys.num("exact"), Some(0));
        assert_eq!(ar.shape_keys.num("bits"), Some(6));
        assert_eq!(ar.entry.as_deref(), Some("ar_oneshot_2rank_wht6"));
        assert_eq!(ar.plugin, Some("saber_ar_wht"));
        assert!(ar.ported);
        assert_eq!(ar.shape_keys.num("numel"), Some(40 * 5120));
        assert_eq!(ar.shape_keys.num("numel").unwrap() % 64, 0);
    }

    /// The gate on `ResolvedOp::ported`: it is derived, and here is the derivation checked from
    /// the other end.
    ///
    /// For every op in every bucket: if the graph says `ported`, the committed kernel plane must
    /// actually produce a route for that geometry — `saber::Registry::select` over the
    /// `[[registry]]` rows for a saber op, `rules::resolve` over the MXFP4 bands for an MXFP4
    /// one — and the plugin it names must be a committed code object. If the graph says
    /// unported, there must be **no** such route: no registry row carries that op id at all.
    #[test]
    fn every_ported_op_resolves_and_every_unported_op_has_no_route() {
        use saber_kernels::rules::LaunchRules;
        use saber_kernels::saber::{Registry, SaberRules};

        let g = graph_tp2();
        let saber_rules = SaberRules::prod_saber().unwrap();
        let mxfp4 = LaunchRules::prod_mxfp4().unwrap();
        let glue = GlueRules::prod().unwrap();
        let reg = Registry::new(&saber_rules);

        let mut ported_ids: BTreeSet<&str> = BTreeSet::new();
        let mut unported_ids: BTreeSet<&str> = BTreeSet::new();

        for bucket in [
            Bucket::Decode {
                batch: 1,
                verify_width: 5,
            },
            Bucket::Decode {
                batch: 8,
                verify_width: 5,
            },
            Bucket::Decode {
                batch: 1,
                verify_width: 11,
            },
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
        ] {
            for r in g
                .ops_for(bucket)
                .iter()
                .chain(g.drafter_ops_for(bucket).iter())
            {
                if r.ported {
                    ported_ids.insert(r.op.kernel.id());
                    let entry = r
                        .entry
                        .as_deref()
                        .unwrap_or_else(|| panic!("{} is ported but names no entry", r.name()));
                    let plugin = r
                        .plugin
                        .unwrap_or_else(|| panic!("{} is ported but names no plugin", r.name()));
                    assert_eq!(r.refusal, None, "{} is ported and yet refused", r.name());
                    assert!(
                        prebuilt::ALL
                            .iter()
                            .any(|(h, _)| h.strip_suffix(".hsaco") == Some(plugin)),
                        "{}: plugin {plugin} is not a committed code object",
                        r.name()
                    );

                    // And the plane really does answer, asked independently of the graph.
                    match r.op.kernel.plane() {
                        Plane::Saber => {
                            let mut geom = saber_kernels::saber::Geometry::new();
                            for (k, v) in r.shape_keys.iter() {
                                geom = match v {
                                    ShapeValue::Num(n) => geom.num(k, *n),
                                    ShapeValue::Str(s) => geom.dtype(k, s),
                                };
                            }
                            // The one op id the graph rewrites: a verify width over
                            // MAX_DECODE_QLEN goes to the prefill kernel, so ask for what it
                            // actually named rather than for the op's own id.
                            let op_id = if entry.starts_with("attn_prefill") {
                                "attn_prefill_paged"
                            } else {
                                r.op.kernel.id()
                            };
                            let row = reg.select(op_id, &geom).unwrap_or_else(|why| {
                                panic!("{}: registry refused `{op_id}`: {why}", r.name())
                            });
                            assert_eq!(row.entry, entry, "{}", r.name());
                        }
                        Plane::Mxfp4 => {
                            // resolve() is what route() called; re-deriving the op here would
                            // duplicate the mapping, so assert the weaker independent fact: the
                            // kernel the graph reports is a real symbol of that plugin's rules.
                            assert!(
                                !entry.is_empty(),
                                "{}: MXFP4 route named an empty kernel",
                                r.name()
                            );
                            assert!(
                                entry.starts_with("mxfp4_")
                                    || entry.starts_with("add_rms_quant")
                                    || entry.starts_with("silu_mul_quant")
                                    || entry.starts_with("gdn_norm_quant"),
                                "{}: unexpected MXFP4 kernel {entry}",
                                r.name()
                            );
                            let _ = &mxfp4;
                        }
                        Plane::Glue => {
                            // Same weaker independent fact as the MXFP4 arm: the symbol the
                            // graph reports is one the glue rules' OWN map claims, asked of
                            // `plugin_for` rather than of a prefix.
                            let owner = glue
                                .plugin_for(entry)
                                .unwrap_or_else(|why| panic!("{}: {why}", r.name()));
                            assert_eq!(owner, plugin, "{}", r.name());
                        }
                        Plane::Head => {
                            // The bf16 head (N2): its norm is the head plugin's, its GEMM the
                            // drafter's bf16 GEMM.
                            assert!(
                                entry == "saber_add_rms_bf16" || entry.starts_with("dflash_bgemm"),
                                "{}: unexpected bf16-head kernel {entry}",
                                r.name()
                            );
                        }
                        Plane::None => panic!("{} is ported but claims no plane", r.name()),
                    }
                } else {
                    unported_ids.insert(r.op.kernel.id());
                    assert_eq!(
                        r.plugin,
                        None,
                        "{} is unported but names a plugin",
                        r.name()
                    );
                    assert_eq!(r.entry, None, "{} is unported but names an entry", r.name());
                    assert_eq!(
                        r.op.kernel.plane(),
                        Plane::None,
                        "{} is unported but claims a plane",
                        r.name()
                    );
                    // The other half of the claim: the kernel plane has nothing for it.
                    assert!(
                        !saber_rules
                            .registry
                            .iter()
                            .any(|row| row.op == r.op.kernel.id()),
                        "{}: the registry DOES carry op `{}`, so `ported` is wrong",
                        r.name(),
                        r.op.kernel.id()
                    );
                }
            }
        }

        // The exact partition, so that an op quietly changing sides fails here.
        assert_eq!(
            ported_ids.iter().copied().collect::<Vec<_>>(),
            vec![
                "add_rms_quant",
                "allreduce",
                "attn_decode_paged",
                "attn_gate_quant",
                "attn_prefill_paged",
                "gdn_chunk_scan",
                "gdn_conv_prep",
                "gdn_conv_update",
                "gdn_fused_update",
                "gdn_kkt_solve",
                "gdn_norm_quant",
                "gdn_recurrent_update",
                "gemm_w4a8",
                "kv_append",
                "qk_norm_rope",
                "silu_mul_quant",
            ]
        );
        // What is left unported is ONE op of the target model — the TP > 1 vocabulary
        // all-gather — plus the drafter's bf16 GEMM and the embedding, whose kernel lives in
        // the head plugin and which this graph therefore reports as having no saber/MXFP4
        // route. `attn_gate_quant` and `qk_norm_rope` moved to the ported side when the
        // `attn_glue` plugin landed, and `kv_append` arrived already ported.
        assert_eq!(
            unported_ids.iter().copied().collect::<Vec<_>>(),
            vec!["embedding", "gemm_bf16", "logits_gather"]
        );
    }

    /// The shape-only answer is wrong for exactly one edge in the model under the reference's
    /// law, and the graph says so. With G2's tiled GDN norm (`gdn_norm_tiled`) there is no
    /// unfused edge left and the shape-only answer is right everywhere; the committed knob picks
    /// which half of this test runs, and the other half is the branch below.
    #[test]
    fn the_kernel_plane_names_the_wrong_gemm_family_for_the_unfused_edge() {
        let g = graph_tp2();
        if gdn_norm_tiled() {
            let prefill = g.ops_for(Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            });
            assert!(prefill.iter().all(|r| !r.shape_only_family_disagrees()));
            let out_proj = prefill
                .iter()
                .find(|r| r.name() == "layer.0.gdn.out_proj")
                .unwrap();
            assert_eq!(out_proj.gemm_family, Some(GemmFamily::ATiled));
            return;
        }

        // At a decode M nothing is A-tiled, so nothing disagrees.
        let decode = g.ops_for(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        assert!(decode.iter().all(|r| !r.shape_only_family_disagrees()));

        // At a prefill M the fused producers are A-tiled and the kernel plane agrees with them;
        // `gdn_norm_quant` is not fused (one instantiation, row-major at every M), so
        // `rules::resolve` names the A-tiled kernel for an edge that is row-major. Forty-eight of
        // them, one per GDN layer, and nothing else. Taking that shape-only answer is what made a
        // 1024- or 2048-row prefill chunk diverge at layer 0 (ws-p6).
        let prefill = g.ops_for(Bucket::Prefill {
            tokens: 2048,
            seqs: 1,
        });
        let disagree: Vec<&str> = prefill
            .iter()
            .filter(|r| r.shape_only_family_disagrees())
            .map(ResolvedOp::name)
            .collect();
        assert_eq!(disagree.len(), 48, "{disagree:?}");
        assert!(disagree.iter().all(|n| n.ends_with(".gdn.out_proj")));

        let out_proj = prefill
            .iter()
            .find(|r| r.name() == "layer.0.gdn.out_proj")
            .unwrap();
        assert_eq!(out_proj.gemm_family, Some(GemmFamily::RowMajor));
        assert!(out_proj.entry.as_deref().unwrap().contains("atiled"));
        assert!(
            out_proj.ported,
            "the kernel exists; the LAYOUT is the problem"
        );
    }

    /// The graph's declared launch counts and the kernel plane's must not drift apart.
    #[test]
    fn declared_launch_counts_agree_with_the_kernel_plane() {
        let g = graph_tp2();

        // MXFP4: every descriptor the plane returns declares 1 (the split-K reduce is fused).
        for bucket in [
            Bucket::Decode {
                batch: 1,
                verify_width: 5,
            },
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
        ] {
            for r in g.ops_for(bucket) {
                if r.op.kernel.plane() == Plane::Mxfp4 {
                    assert_eq!(r.n_launches, Some(1), "{}", r.name());
                }
            }
        }

        // saber declares `n_launches = 2` for attention decode and 2 for the GDN decode PAIR
        // as a whole (`saber::Op::GdnDecodePair`). The graph draws the pair as two ops because
        // the conv's q/k/v is a real edge the recurrent update consumes, so it splits the same
        // two launches across them -- and the totals have to match.
        let pair = g.launch_count_lower_bound(Bucket::Decode {
            batch: 8,
            verify_width: 5,
        });
        assert_eq!(pair.by_kernel["gdn_conv_update"], 48);
        assert_eq!(pair.by_kernel["gdn_recurrent_update"], 48);
        assert_eq!(
            pair.by_kernel["gdn_conv_update"] + pair.by_kernel["gdn_recurrent_update"],
            2 * 48
        );
        assert_eq!(pair.by_kernel["attn_decode_paged"], 2 * 16);
    }

    /// The knobs the graph restates are pinned against the committed rule tables.
    #[test]
    fn the_saber_constants_match_the_committed_rules() {
        use saber_kernels::rules::LaunchRules;
        use saber_kernels::saber::SaberRules;

        let r = SaberRules::prod_saber().unwrap();
        assert_eq!(GDN_FUSED_MAX_ITEMS, r.gdn.fused_max_items);
        assert_eq!(AR_BITS, r.ar.qbits);
        assert_eq!(AR_QUANT_MIN_BYTES, r.ar.quant_min_bytes);
        assert_eq!(AR_MAX_BYTES, r.ar.max_bytes);
        // MAX_DECODE_QLEN is the prose's `q_len * gqa <= ATTN_MAX_DECODE_ROWS`, evaluated.
        assert_eq!(
            MAX_DECODE_QLEN,
            r.knobs.attn_max_decode_rows / r.knobs.attn_gqa
        );
        // `a_tiled_min_m` is no longer restated here: the accessor IS the rules' knob, so this
        // is an identity rather than a drift test -- and that is the point. The guard that used
        // to live here is `a_tiled_min_m_is_written_down_once` in `saber-kernels`, which fails
        // when a SECOND definition appears rather than when two copies disagree.
        let m = LaunchRules::prod_mxfp4().unwrap();
        assert_eq!(a_tiled_min_m(), m.knobs.a_tiled_min_m);
        assert_eq!(
            LayoutRule::FusedActivationFp8.resolve(m.knobs.a_tiled_min_m),
            Layout::ATiledFp8
        );
        assert_eq!(
            LayoutRule::FusedActivationFp8.resolve(m.knobs.a_tiled_min_m - 1),
            Layout::RowMajorFp8
        );
    }

    #[test]
    fn attention_decode_declares_two_launches_and_prefill_one() {
        let g = graph_tp2();
        let decode = g.launch_count_lower_bound(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        assert_eq!(decode.by_kernel["attn_decode_paged"], 32, "16 layers x 2");
        let prefill = g.launch_count_lower_bound(Bucket::Prefill {
            tokens: 2048,
            seqs: 1,
        });
        assert_eq!(prefill.by_kernel["attn_prefill_paged"], 16);
    }

    /// Several sequences' chunks in ONE prefill step: the L9 shape, by total rows.
    ///
    /// The engine runs prefill one chunk per step today, so this is the shape the planner has to
    /// describe before the engine can batch them. Everything a caller needs is derived from the
    /// pair `(tokens, seqs)`: the bucket's M is the TOTAL, `q_len` is one chunk, and the head
    /// runs on one row per sequence -- the last of each chunk -- which is neither 1 nor the total.
    #[test]
    fn several_chunks_in_one_prefill_step() {
        // Two sequences of 1024, which is the gate's shape.
        let b = Bucket::prefill_batch(2, 1024);
        assert_eq!(b.m(), 2048, "the step is sized by TOTAL rows (the L9 rule)");
        assert_eq!(b.q_len(), 1024, "each sequence feeds its own chunk");
        assert_eq!(b.num_seqs(), 2);
        assert_eq!(
            b.sampled_rows(),
            2,
            "one row per sequence: the last of each chunk, not 1 and not 2048"
        );
        assert_eq!(b.phase(), Phase::Prefill);
        assert!(b.chunks_divide());

        // It degenerates to the single-sequence bucket, so the existing shape is a special case
        // rather than a separate one.
        let one = Bucket::prefill_batch(1, 2048);
        assert_eq!(
            one,
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1
            }
        );
        assert_eq!(one.sampled_rows(), 1);
        assert_eq!(one.q_len(), 2048);

        // `seqs: 0` is not a divide-by-zero and not a zero-row step.
        let z = Bucket::prefill_batch(0, 64);
        assert_eq!(z.num_seqs(), 1);
        assert_eq!(z.m(), 64);

        // The head's bucket is `seqs` rows wide, so a multi-sequence chunk plans a 2-row head
        // and a single-sequence one plans a 1-row head, from the same rule.
        let g = graph_tp2();
        assert_eq!(g.head_bucket(b).m(), 2);
        assert_eq!(g.head_bucket(one).m(), 1);
        assert!(g.head_bucket(b).m() < b.m(), "the head is not the body");

        // Unequal chunks are a different shape and this says so rather than planning them.
        // (1500 over 4 divides exactly, which is why the example is 1000 over 3.)
        assert!(!Bucket::Prefill {
            tokens: 1000,
            seqs: 3
        }
        .chunks_divide());
        assert!(Bucket::Prefill {
            tokens: 1500,
            seqs: 4
        }
        .chunks_divide());
        // Fewer rows than sequences is not a step either.
        assert!(!Bucket::Prefill { tokens: 2, seqs: 4 }.chunks_divide());
    }

    /// The head runs on the sampled positions, and `sampled_rows` is where that is written down.
    ///
    /// At prefill it is one row per sequence however long the chunk is; at decode it is every row,
    /// because under speculative decode each draft position needs its own logits to be verified.
    /// The 2026-09-20 I15 measurement found the prefill step computing logits for all 2048 —
    /// 27.6 ms of a 637 ms step, 970 MiB of a GEMM of which one row is read.
    #[test]
    fn the_head_runs_on_the_sampled_rows_only() {
        // A prefill chunk: `m` is the tokens, the head's rows are the sequences.
        for tokens in [1u32, 64, 2048, 8192] {
            let b = Bucket::Prefill { tokens, seqs: 1 };
            assert_eq!(b.m(), tokens);
            assert_eq!(
                b.sampled_rows(),
                1,
                "one sequence samples once, at {tokens}"
            );
        }
        // Two sequences in one chunk sample twice, still not once per token.
        let two = Bucket::Prefill {
            tokens: 2048,
            seqs: 2,
        };
        assert_eq!(two.sampled_rows(), 2);
        // `seqs: 0` is not a divide-by-zero and not zero rows.
        assert_eq!(Bucket::Prefill { tokens: 8, seqs: 0 }.sampled_rows(), 1);

        // At decode every row is a sampled position, so the two agree and nothing changes there.
        // This is what makes the planner's guard silent on every decode plan.
        for (batch, width) in [(1u32, 1u32), (1, 5), (8, 5), (4, 1)] {
            let b = Bucket::Decode {
                batch,
                verify_width: width,
            };
            assert_eq!(
                b.sampled_rows(),
                b.m(),
                "decode b{batch}s{width}: every row is verified"
            );
        }

        // `head_bucket` turns that into the bucket the head site is planned at: a row count and
        // nothing else, never a prefill bucket carrying the chunk's tokens.
        let g = graph_tp2();
        let head = g.head_bucket(Bucket::Prefill {
            tokens: 8192,
            seqs: 1,
        });
        assert_eq!(
            head,
            Bucket::Decode {
                batch: 1,
                verify_width: 1
            }
        );
        assert_eq!(head.m(), 1, "the head GEMM sees ONE row, not 8192");
        // And it is idempotent: planning the head's own bucket gives the same bucket back.
        assert_eq!(g.head_bucket(head), head);

        // The head's rows never exceed the body's, at any bucket.
        for b in [
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            Bucket::Decode {
                batch: 8,
                verify_width: 5,
            },
        ] {
            assert!(g.head_bucket(b).m() <= b.m(), "{b:?}");
        }
    }

    #[test]
    fn launch_budget_is_a_lower_bound_under_prods_measured_count() {
        let g = graph_tp2();
        let b1 = g.launch_count_lower_bound(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        let b8 = g.launch_count_lower_bound(Bucket::Decode {
            batch: 8,
            verify_width: 5,
        });
        let pre = g.launch_count_lower_bound(Bucket::Prefill {
            tokens: 2048,
            seqs: 1,
        });

        // Per GDN layer at batch 1: norm, qkvz, ba, fused, gdn_norm, out_proj, AR,
        // post-norm, gate_up, silu, down, AR = 12.  Per attention layer: norm, qkv,
        // QK_NORM_ROPE, KV_APPEND, attn(2), GATE_QUANT, o_proj, AR, post-norm, gate_up, silu,
        // down, AR = 14.  Ends: the embedding LOOKUP and its AR, the final norm, lm_head = 4.
        //
        // The three capitalised ones joined the count when the `attn_glue` plugin landed
        // (workstream F, plugin 4) — the budget went 756 -> 804 at batch 1, 804 -> 852 at
        // batch 8 and 836 -> 884 at prefill, 48 launches each time, which is 3 x 16 attention
        // layers. `logits.gather` still has no code object, so the TP > 1 vocabulary
        // all-gather is still not in this number.
        assert_eq!(b1.total, 48 * 12 + 16 * 14 + 4);
        assert_eq!(b1.total, 804);
        // batch 8 replaces one fused launch per GDN layer with the two-kernel pair.
        assert_eq!(b8.total, b1.total + 48);
        assert_eq!(b8.total, 852);
        // Prefill: the GDN triple replaces the fused kernel (+2 per GDN layer) and attention
        // is one launch instead of two (-1 per attention layer).
        assert_eq!(pre.total, b1.total + 48 * 2 - 16);
        assert_eq!(pre.total, 884);

        assert!(
            b8.total < PROD_LAUNCHES_PER_STEP_CONC8,
            "a lower bound that exceeds prod's measured 1084 is not a lower bound"
        );
        // The gap is named, not hidden: every unmodelled op id is one of these. `embedding`
        // left when the head plugin landed; `qk_norm_rope` and `attn_gate_quant` left when the
        // attention glue did. What remains is ONE op — the TP > 1 vocabulary all-gather — and
        // it does not run at TP=1 at all.
        let unmodelled: Vec<&str> = b8.unmodelled.keys().copied().collect();
        assert_eq!(unmodelled, vec!["logits_gather"]);
        assert_eq!(b8.unmodelled["logits_gather"], 1);
    }

    /// An independent corroboration of the merge model, and the reason it is worth a test of
    /// its own.
    ///
    /// The prod NVFP4 unit logs `linear layers: 304/304` when the load-time NVFP4 -> MXFP4
    /// requant has covered every linear — and "any layer left on hipBLASLt fp8 wedges the GPU
    /// at conc-8", so that count is checked on every boot. This graph declares **305** W4A8
    /// GEMMs per forward, and the one extra is `lm_head`, which the requant patch handles
    /// separately from the decoder's linear layers. If the merges here were wrong — gate and
    /// up split, or `in_proj_qkv`/`in_proj_z` not merged — the count would be 369 or 353, not
    /// 305. Nothing else in this module was fitted to that number.
    #[test]
    fn the_gemm_count_matches_prods_linear_layer_census() {
        let g = graph_tp2();
        let budget = g.launch_count_lower_bound(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        assert_eq!(budget.by_kernel["gemm_w4a8"], 305);
        // 48 GDN x (qkvz + ba + out_proj) + 16 attn x (qkv + o_proj) + 64 x (gate_up + down)
        assert_eq!(48 * 3 + 16 * 2 + 64 * 2 + 1, 305);
    }

    #[test]
    fn ops_for_bucket_infers_the_prod_split_from_m() {
        let g = graph_tp2();
        assert_eq!(
            g.bucket_for(40, Phase::Decode),
            Bucket::Decode {
                batch: 8,
                verify_width: 5
            }
        );
        // Not a multiple of the verify width: read as SPEC=0 decode, not silently rounded.
        assert_eq!(
            g.bucket_for(3, Phase::Decode),
            Bucket::Decode {
                batch: 3,
                verify_width: 1
            }
        );
        assert_eq!(
            g.bucket_for(8192, Phase::Prefill),
            Bucket::Prefill {
                tokens: 8192,
                seqs: 1
            }
        );
        for bucket in PROD_DECODE_BUCKETS {
            let ops = g.ops_for(*bucket);
            assert!(!ops.is_empty());
            assert_eq!(g.bucket_for(bucket.m(), Phase::Decode), *bucket);
        }
    }

    #[test]
    fn tp1_has_no_all_reduce_at_all() {
        // At TP=1 the reduce is not a no-op to skip at runtime, it does not exist: the
        // row-parallel GEMMs write the block output directly. The TP=1 single-card pass is a
        // real configuration (docs/LEDGER.md), so the graph has to be valid there.
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("tp=1 graph");
        assert!(g.ops.iter().all(|o| o.kernel != KernelOp::AllReduce));
        assert!(g.ops.iter().all(|o| o.kernel != KernelOp::LogitsGather));
        g.check_layouts(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        })
        .expect("tp=1 edges resolve");
        g.check_layouts(Bucket::Prefill {
            tokens: 2048,
            seqs: 1,
        })
        .expect("tp=1 prefill edges resolve");

        // 129 all-reduces vanish: two per layer plus the embedding's.
        let tp2 = graph_tp2().launch_count_lower_bound(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        let tp1 = g.launch_count_lower_bound(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        assert_eq!(tp2.by_kernel["allreduce"], 129);
        assert!(!tp1.by_kernel.contains_key("allreduce"));
        // …and 48 launches come back, because at TP=1 the value heads are 48 and
        // nseq * H = 48 > FUSED_MAX_ITEMS 32 even at batch 1: the fused GDN decode kernel
        // NEVER runs at TP=1. That is the routing rule reading out a consequence, not a
        // separate rule.
        assert!(!tp1.by_kernel.contains_key("gdn_fused_update"));
        assert_eq!(tp1.by_kernel["gdn_conv_update"], 48);
        assert_eq!(tp1.by_kernel["gdn_recurrent_update"], 48);
        assert_eq!(tp1.total, tp2.total - 129 + 48);

        // And the shapes double, because nothing is sharded.
        let by_name: BTreeMap<&str, &Op> = g.ops.iter().map(|o| (o.name.as_str(), o)).collect();
        match by_name["layer.0.gdn.in_proj_qkvz"].geometry {
            Geometry::Gemm { n, k, .. } => assert_eq!((n, k), (16384, 5120)),
            _ => panic!("not a GEMM"),
        }
        assert_eq!(g.gdn_value_heads_per_rank(), 48);
    }

    #[test]
    fn a_long_verify_width_falls_back_to_the_prefill_attention_kernel() {
        let g = graph_tp2();
        // q_len 11 > MAX_DECODE_QLEN 10: the decode kernel cannot take it and the backend
        // sends the run to the prefill kernel instead.
        let ops = g.ops_for(Bucket::Decode {
            batch: 1,
            verify_width: 11,
        });
        let attn = ops
            .iter()
            .find(|r| r.op.kernel == KernelOp::AttnDecodePaged)
            .unwrap();
        assert_eq!(attn.entry.as_deref(), Some("attn_prefill_h256_gqa6_fp8kv"));
    }

    /// The human-readable form of [`Graph::launch_count_lower_bound`]. Run it with
    /// `cargo test -p saber-model -- --nocapture budget_table`; the numbers themselves are
    /// asserted in `launch_budget_is_a_lower_bound_under_prods_measured_count`.
    #[test]
    fn budget_table() {
        let g = graph_tp2();
        let buckets = PROD_DECODE_BUCKETS.iter().copied().chain([
            Bucket::Decode {
                batch: 1,
                verify_width: 1,
            },
            Bucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            Bucket::Prefill {
                tokens: 8192,
                seqs: 1,
            },
        ]);
        for bucket in buckets {
            let budget = g.launch_count_lower_bound(bucket);
            print!("{budget}");
            assert!(budget.total > 0);
        }
        println!("prod measured, conc-8: {PROD_LAUNCHES_PER_STEP_CONC8} launches/step");
    }

    #[test]
    fn op_order_is_a_forward_pass() {
        let g = graph_tp2();
        let names: Vec<&str> = g.ops.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names[0], "embed_tokens");
        assert_eq!(names[1], "embed_tokens.all_reduce");
        assert_eq!(names[2], "layer.0.input_layernorm");
        assert_eq!(names[names.len() - 3], "model.norm");
        assert_eq!(names[names.len() - 2], "lm_head");
        assert_eq!(names[names.len() - 1], "logits.gather");
        // Layers appear in order and never interleave.
        let mut last = 0u32;
        for op in &g.ops {
            if let Site::Layer(l) = op.site {
                assert!(l >= last, "layer {l} after {last}");
                last = l;
            }
        }
        assert_eq!(last, 63);
    }

    #[test]
    fn the_static_op_list_is_the_union_of_every_route() {
        let g = graph_tp2();
        // The graph holds every route; a bucket selects a subset. Per GDN layer: norm, qkvz,
        // ba, {fused | conv+recurrent | prep+kkt+scan} = 6 core rows, gdn norm, out_proj, AR,
        // post-norm, gate_up, silu, down, AR = 17. Per attention layer: norm, qkv, qk_norm,
        // KV_APPEND, {prefill | decode} = 2 rows, gate_quant, o_proj, AR, post-norm, gate_up,
        // silu, down, AR = 14. Head 3, embedding 2.
        //
        // `kv_append` is the row that arrived with the `attn_glue` plugin: the paged cache was
        // state that every attention op READ and no op wrote, so a forward pass read whatever
        // the allocator handed back.
        //
        // P27-15d added the TREE verify's rows: per GDN layer `conv_tree` + `tree_update` (19),
        // per attention layer `attn.tree` (15).
        assert_eq!(g.ops.len(), 2 + 48 * 19 + 16 * 15 + 3);
        assert_eq!(g.ops.len(), 1157);
        assert_eq!(g.drafter.len(), 12);
        // And a decode bucket selects strictly fewer, because the prefill-only GDN triple and
        // the prefill attention row drop out.
        let decode = g.ops_for(Bucket::Decode {
            batch: 1,
            verify_width: 5,
        });
        assert_eq!(decode.len(), g.ops.len() - 48 * 7 - 32);
        // The tree bucket: no prefill triple, no fused / pair update, no prefill or causal
        // decode attention -- its own three rows instead.
        let tree = g.ops_for(Bucket::DecodeTree { rows: 10 });
        assert_eq!(tree.len(), g.ops.len() - 48 * 6 - 32);
        assert_eq!(decode.len() + 48, tree.len(), "+1 op a GDN layer");
    }

    #[test]
    fn layer_zero_has_no_residual_input() {
        let g = graph_tp2();
        let l0 = g
            .ops
            .iter()
            .find(|o| o.name == "layer.0.input_layernorm")
            .unwrap();
        assert_eq!(l0.inputs.len(), 1, "layer 0 adds no residual");
        let l1 = g
            .ops
            .iter()
            .find(|o| o.name == "layer.1.input_layernorm")
            .unwrap();
        assert_eq!(l1.inputs.len(), 2);
    }

    /// **The slot law is DATA, and it is paired with the slot size** (ws-k14).
    ///
    /// A session owns `q` consecutive state slots and the two slotted states do not live in the
    /// same one: the conv cache is `cache_idx`'s, which a decode step binds to `sidx` at
    /// `ci_stride = q_len`, so it is entry 0 of the sequence's block — the run's BASE — whatever
    /// the step accepted; the recurrent state is `sidx`'s, per ROW, so the live snapshot is
    /// `base + naccept - 1`. A park image that carried the base for both is sound only when the
    /// restored session lands on the slots it already held (LEDGER, T2 item 2), which a
    /// serve-plane park never does.
    ///
    /// The pairing is the half that cannot rot: a state that states a slot SIZE and no slot law
    /// (or the other way round) would be a slot run the park could not place.
    #[test]
    fn every_state_pairs_its_slot_size_with_its_index_law() {
        for g in [
            graph_tp2(),
            Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("build graph"),
        ] {
            let mut conv = 0usize;
            let mut rec = 0usize;
            let mut paged = 0usize;
            for op in g.ops.iter().chain(g.drafter.iter()) {
                for st in &op.state {
                    match st.indexed_by {
                        StateIndexedBy::BlockTable => {
                            assert_eq!(
                                st.bytes_per_slot, None,
                                "{} is paged and states a slot size",
                                st.name
                            );
                            paged += 1;
                        }
                        StateIndexedBy::CacheIdx => {
                            assert!(
                                st.bytes_per_slot.is_some(),
                                "{} has a slot law and no slot size",
                                st.name
                            );
                            assert!(st.name.ends_with("conv_state"), "{}", st.name);
                            conv += 1;
                        }
                        StateIndexedBy::Sidx => {
                            assert!(
                                st.bytes_per_slot.is_some(),
                                "{} has a slot law and no slot size",
                                st.name
                            );
                            assert!(st.name.ends_with("recurrent_state"), "{}", st.name);
                            rec += 1;
                        }
                    }
                }
            }
            // The CONTROL for "the walk found anything at all": all three kinds exist in this
            // graph, so a loop that silently matched none of them cannot pass.
            assert!(conv > 0 && rec > 0 && paged > 0, "{conv} {rec} {paged}");
        }
    }

    #[test]
    fn gdn_state_is_fp32_and_sized_for_the_infinity_cache_budget() {
        let g = graph_tp2();
        let op = g
            .ops
            .iter()
            .find(|o| o.name == "layer.0.gdn.fused_update")
            .unwrap();
        let state = op
            .state
            .iter()
            .find(|s| s.name.ends_with("recurrent_state"))
            .unwrap();
        assert_eq!(state.layout, Layout::Fp32State);
        // 24 heads x 128 x 128 x 4 B = 1.5 MiB per layer per slot at TP=2, i.e. 72 MiB over
        // 48 layers -- the figure ARCHITECTURE.md item 3 and the kernel inventory doc §2 quote.
        assert_eq!(state.bytes_per_slot, Some(1536 * 1024));
        let total = state.bytes_per_slot.unwrap() * u64::from(g.config.gdn_layers());
        assert_eq!(total, 72 * 1024 * 1024);
    }

    // ---------------------------------------------------------------------------------
    // The manifest binding gate
    // ---------------------------------------------------------------------------------

    #[test]
    fn every_bound_name_exists_in_the_pack_with_the_right_shape_and_dtype() {
        let Some(manifest) = load_manifest() else {
            return;
        };
        assert_eq!(
            manifest.tp_size, 2,
            "this gate is written for the TP=2 pack"
        );
        let g = graph_tp2();
        let bound = g.bound_records();

        for rank in &manifest.ranks {
            let have: BTreeMap<&str, &ManifestRecord> =
                rank.records.iter().map(|r| (r.name.as_str(), r)).collect();
            let mut missing = Vec::new();
            let mut wrong = Vec::new();
            for (name, binding) in &bound {
                let Some(record) = have.get(name) else {
                    missing.push(*name);
                    continue;
                };
                if record.dtype != binding.dtype.manifest_name()
                    || record.shape != binding.shape
                    || record.shard_kind != binding.shard.manifest_name()
                {
                    wrong.push(format!(
                        "{name}: graph says {} {:?} {}, pack rank {} has {} {:?} {}",
                        binding.dtype,
                        binding.shape,
                        binding.shard,
                        rank.rank,
                        record.dtype,
                        record.shape,
                        record.shard_kind
                    ));
                }
                assert_eq!(
                    record.layout_id, 0,
                    "{name}: this graph describes a layout_id 0 (raw) pack; a device-final \
                     layout is an ABI change and must be taught here too"
                );
            }
            assert!(
                missing.is_empty(),
                "rank {}: {} bound name(s) are not in the pack:\n  {}",
                rank.rank,
                missing.len(),
                missing.join("\n  ")
            );
            assert!(
                wrong.is_empty(),
                "rank {}: {} binding(s) disagree with the pack:\n  {}",
                rank.rank,
                wrong.len(),
                wrong.join("\n  ")
            );
        }
    }

    #[test]
    fn every_pack_record_is_bound_or_explicitly_unused() {
        let Some(manifest) = load_manifest() else {
            return;
        };
        let g = graph_tp2();
        let bound = g.bound_records();

        for rank in &manifest.ranks {
            let mut orphans = Vec::new();
            let mut both = Vec::new();
            for record in &rank.records {
                let is_bound = bound.contains_key(record.name.as_str());
                let is_unused = g.is_unused(&record.name);
                if is_bound && is_unused {
                    both.push(record.name.as_str());
                } else if !is_bound && !is_unused {
                    orphans.push(record.name.as_str());
                }
            }
            assert!(
                both.is_empty(),
                "rank {}: bound AND listed unused: {both:?}",
                rank.rank
            );
            assert!(
                orphans.is_empty(),
                "rank {}: {} record(s) neither bound nor listed in UNUSED_RECORDS. Every one \
                 is either a weight this graph forgot or a claim that needs a reason:\n  {}",
                rank.rank,
                orphans.len(),
                orphans.join("\n  ")
            );

            // And the counts, so a silently emptied graph cannot pass the two checks above.
            let vision = rank.records.iter().filter(|r| g.is_unused(&r.name)).count();
            assert_eq!(vision, 333, "rank {}: vision tower record count", rank.rank);
            assert_eq!(
                bound.len() + vision,
                rank.records.len(),
                "rank {}: {} bound + {vision} unused != {} records",
                rank.rank,
                bound.len(),
                rank.records.len()
            );
        }
    }

    #[test]
    fn the_drafter_head_is_bound_in_full() {
        let Some(manifest) = load_manifest() else {
            return;
        };
        let g = graph_tp2();
        let bound = g.bound_records();
        let mtp: BTreeSet<&str> = manifest.ranks[0]
            .records
            .iter()
            .map(|r| r.name.as_str())
            .filter(|n| n.starts_with("mtp."))
            .collect();
        assert_eq!(mtp.len(), 15, "the tier's MTP head is 15 records");
        for name in &mtp {
            assert!(bound.contains_key(name), "drafter record {name} is unbound");
        }
        // And nothing else claims them.
        let from_drafter: BTreeSet<&str> = g
            .drafter
            .iter()
            .flat_map(|o| o.weights.iter())
            .map(|b| b.record.as_str())
            .collect();
        assert_eq!(from_drafter, mtp);
    }

    #[test]
    fn the_fp8_mlp_layers_are_the_ones_the_pack_stores_unpacked() {
        let Some(manifest) = load_manifest() else {
            return;
        };
        let cfg = ModelConfig::qwen38_27b();
        let mut fp8_in_pack: Vec<u32> = manifest.ranks[0]
            .records
            .iter()
            .filter_map(|r| {
                r.name
                    .strip_prefix("model.language_model.layers.")
                    .and_then(|rest| rest.strip_suffix(".mlp.gate_proj.weight"))
                    .and_then(|n| n.parse().ok())
            })
            .collect();
        fp8_in_pack.sort_unstable();
        assert_eq!(fp8_in_pack, cfg.fp8_mlp_layers);
    }

    // ---------------------------------------------------------------------------------
    // Cross-checks against the files the constants came from
    // ---------------------------------------------------------------------------------

    #[test]
    fn the_constants_match_the_checkpoints_config_json() {
        let path = Path::new(CHECKPOINT_CONFIG);
        if !path.exists() {
            eprintln!("SKIP: {CHECKPOINT_CONFIG} is absent; ModelConfig constants unchecked");
            return;
        }
        let text = std::fs::read_to_string(path).expect("read config.json");
        let parsed = ModelConfig::from_config_json(&text).expect("parse config.json");
        assert_eq!(parsed, ModelConfig::qwen38_27b());
    }

    #[test]
    fn config_json_layer_types_disagreement_is_refused() {
        // Negative control for from_config_json: a checkpoint whose layer_types and
        // full_attention_interval disagree must be an error, not a silent reinterpretation.
        let json = r#"{
            "text_config": {
                "hidden_size": 8, "num_hidden_layers": 2, "full_attention_interval": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "intermediate_size": 8, "linear_num_key_heads": 1, "linear_num_value_heads": 1,
                "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
                "vocab_size": 16, "mtp_num_hidden_layers": 1,
                "layer_types": ["full_attention", "linear_attention"]
            }
        }"#;
        let err = ModelConfig::from_config_json(json).unwrap_err();
        assert!(format!("{err}").contains("layer_types"), "{err}");
    }

    #[test]
    fn shape_keys_render_canonically() {
        let g = graph_tp2();
        let op = g
            .ops
            .iter()
            .find(|o| o.name == "layer.0.mlp.gate_up")
            .unwrap();
        let keys = op.shape_keys(Bucket::Decode {
            batch: 8,
            verify_width: 5,
        });
        assert_eq!(
            keys.to_string(),
            "a_dtype=fp8_e4m3 k=5120 m=40 n=17408 w_dtype=mxfp4"
        );
        assert_eq!(keys.len(), 5);
        assert!(!keys.is_empty());
    }

    // =================================================================================
    // Device-final records (P2)
    // =================================================================================

    fn devfinal() -> Vec<DeviceFinalRecord> {
        Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2)
            .expect("graph")
            .device_final_records()
            .expect("device-final plan")
    }

    #[test]
    fn every_w4a8_gemm_becomes_exactly_three_records() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let gemms = g
            .ops
            .iter()
            .filter(|o| o.kernel == KernelOp::GemmW4A8)
            .count();
        let recs = devfinal();
        let w = recs
            .iter()
            .filter(|r| r.role == DeviceFinalRole::Weight && r.dtype == Dtype::Fp4E2m1X2)
            .count();
        let ws = recs
            .iter()
            .filter(|r| r.layout_id == LAYOUT_MXFP4_BLOCK_SCALE)
            .count();
        let wref = recs
            .iter()
            .filter(|r| r.layout_id == LAYOUT_MXFP4_ROW_REF)
            .count();
        // 305: 48 x (in_proj_qkvz, in_proj_ba, out_proj) + 16 x (qkv, o_proj) + 64 x (gate_up,
        // down) + lm_head -- the same 305 the launch budget counts and prod logs as
        // "linear layers: 304/304" plus the lm_head its requant patch handles separately.
        assert_eq!((gemms, w, ws, wref), (305, 305, 305, 305));
        let merged = recs
            .iter()
            .filter(|r| r.layout_id == LAYOUT_MXFP4_WPERM_MERGED)
            .count();
        let single = recs
            .iter()
            .filter(|r| r.layout_id == LAYOUT_MXFP4_WPERM)
            .count();
        // Merged: 48 x (qkvz, ba) + 16 x qkv + 64 x gate_up. Single: the rest.
        assert_eq!(
            (merged, single),
            (48 * 2 + 16 + 64, 305 - (48 * 2 + 16 + 64))
        );
    }

    #[test]
    fn a_merged_record_lists_its_sources_in_concatenation_order() {
        let recs = devfinal();
        let qkvz = recs
            .iter()
            .find(|r| r.name == "layer.0.gdn.in_proj_qkvz.w")
            .expect("the merged GDN in_proj");
        assert_eq!(qkvz.layout_id, LAYOUT_MXFP4_WPERM_MERGED);
        assert_eq!(qkvz.shape, vec![8192, 5120], "rank-local qkv + z at TP=2");
        let names: Vec<&str> = qkvz.sources.iter().map(|s| s.weight.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            ],
            "qkv before z: the order is the binding order, which is the concat order"
        );
        assert!(qkvz
            .sources
            .iter()
            .all(|s| s.quant == SourceQuant::Fp8Channel));
        // gate_up is the NVFP4 pair, and its global scales are named so the builder can account
        // for every record it consumed.
        let gate_up = recs
            .iter()
            .find(|r| r.name == "layer.0.mlp.gate_up.w")
            .expect("the merged MLP");
        assert_eq!(gate_up.sources.len(), 2);
        assert!(gate_up
            .sources
            .iter()
            .all(|s| s.quant == SourceQuant::Nvfp4));
        assert!(gate_up.sources[0].global_scale.is_some());
        assert!(gate_up.sources[0].scale.is_some());
        // in_proj_ba is bf16 in the checkpoint and MXFP4 in the pack: prod requantizes it too
        // (SABER_NVFP4_BF16_LAYERS=in_proj_ba), which is what lets the GDN in_proj merge fire.
        let ba = recs
            .iter()
            .find(|r| r.name == "layer.0.gdn.in_proj_ba.w")
            .expect("the merged gates");
        assert!(ba.sources.iter().all(|s| s.quant == SourceQuant::Bf16));
        assert_eq!(ba.shape, vec![48, 5120]);
    }

    #[test]
    fn the_transform_inputs_are_consumed_not_emitted() {
        let recs = devfinal();
        for r in &recs {
            assert!(
                !r.name.ends_with("weight_global_scale") && !r.name.ends_with("input_global_scale"),
                "{} is an input to the requant; a device-final pack that carried it would be \
                 inviting a second application",
                r.name
            );
            assert!(
                !r.name.ends_with(".weight_packed"),
                "{} is the checkpoint's NVFP4, which the requant replaced",
                r.name
            );
        }
        // And the scales that WERE consumed are still named, so nothing is lost silently.
        let gate_up = recs
            .iter()
            .find(|r| r.name == "layer.0.mlp.gate_up.ws")
            .unwrap();
        assert_eq!(gate_up.sources.len(), 2);
    }

    #[test]
    fn the_gdn_rate_tensors_are_cast_to_f32_under_their_own_names() {
        let recs = devfinal();
        for role in ["A_log", "dt_bias"] {
            let name = format!("model.language_model.layers.0.linear_attn.{role}");
            let r = recs.iter().find(|r| r.name == name).expect(&name);
            assert_eq!(r.layout_id, LAYOUT_F32_FROM_BF16);
            assert_eq!(r.dtype, Dtype::F32, "the GDN kernels take const float*");
            assert_eq!(r.shape, vec![24], "value heads per rank at TP=2");
        }
        assert_eq!(
            recs.iter()
                .filter(|r| r.layout_id == LAYOUT_F32_FROM_BF16)
                .count(),
            48 * 2,
            "one A_log and one dt_bias per GDN layer"
        );
    }

    #[test]
    fn the_drafter_head_merges_but_never_requantizes() {
        let recs = devfinal();
        let merged: Vec<&DeviceFinalRecord> = recs
            .iter()
            .filter(|r| r.layout_id == LAYOUT_BF16_MERGED)
            .collect();
        // The MTP head is bf16 in this tier and an MXFP4 drafter is a measured negative result,
        // so the only transform it gets is the merge: qkv and gate_up.
        assert_eq!(
            merged.len(),
            2,
            "{:?}",
            merged.iter().map(|r| &r.name).collect::<Vec<_>>()
        );
        assert!(merged.iter().all(|r| r.dtype == Dtype::Bf16));
        assert!(merged.iter().all(|r| r.name.starts_with("mtp.")));
        // mtp.fc is one source, so it keeps its checkpoint name rather than being renamed.
        assert!(recs
            .iter()
            .any(|r| r.name == "mtp.fc.weight" && r.layout_id == LAYOUT_RAW));
    }

    #[test]
    fn records_are_unique_and_every_source_is_a_record_the_graph_binds() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let bound = g.bound_records();
        let recs = g.device_final_records().expect("plan");
        let mut names = BTreeSet::new();
        for r in &recs {
            assert!(names.insert(r.name.clone()), "duplicate record {}", r.name);
            for s in &r.sources {
                for n in [
                    Some(&s.weight),
                    s.scale.as_ref(),
                    s.global_scale.as_ref(),
                    s.input_scale.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    assert!(bound.contains_key(n.as_str()), "{n} is not a bound record");
                }
            }
        }
    }

    #[test]
    fn the_record_order_is_the_forward_order_so_a_window_is_a_prefix_of_layers() {
        // A loader that wants layers 0..N takes a contiguous window, which is what
        // `saber layer --pack` does and what workstream K's "record order is a load-time
        // property" asked for.
        let recs = devfinal();
        let mut last = -1i64;
        for r in &recs {
            let Some(op) = &r.op else { continue };
            let Some(rest) = op.strip_prefix("layer.") else {
                continue;
            };
            let Some(n) = rest.split('.').next().and_then(|d| d.parse::<i64>().ok()) else {
                continue;
            };
            assert!(n >= last, "layer {n} appears after layer {last}");
            last = n;
        }
        assert_eq!(last, 63, "the last layer's records are the last layer's");
    }

    /// The conv cache's slot size follows the graph's verify width, and SPEC=7 is width **8**.
    ///
    /// `VERIFY_WIDTH`'s 5 is SPEC=4: prod's DFlash2 block is `1 + num_speculative_tokens` rows, so
    /// a tree that could only build at 5 could not run prod's own bucket at all. The slot is
    /// `qkv_n * (conv_width - 1 + q - 1) * 2` bytes, so widening the graph by three widens every
    /// slot by `3 * qkv_n * 2` and by nothing else — which is what this pins, against the two
    /// readers that derive from the same field (`StateRef::bytes_per_slot` here and
    /// `saber_exec::layer::gdn_shape`'s `cs_dim`).
    #[test]
    fn the_conv_slot_is_sized_from_the_graphs_own_verify_width() {
        let slot = |w: u32| -> u64 {
            let g =
                Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 1, w).expect("a legal width");
            assert_eq!(g.verify_width, w);
            g.ops
                .iter()
                .flat_map(|o| &o.state)
                .find(|s| s.name == "l0.gdn.conv_state")
                .and_then(|s| s.bytes_per_slot)
                .expect("the conv cache declares its slot size")
        };
        let at5 = slot(5);
        let at8 = slot(8);
        assert_eq!(slot(VERIFY_WIDTH), at5, "the default is still 5");
        // One qkv row per dim per extra window entry, bf16.
        let qkv_n = (at5 / 2) / u64::from(ModelConfig::qwen38_27b().linear_conv_kernel_dim - 1 + 4);
        assert_eq!(at8 - at5, 3 * qkv_n * 2, "three more entries per dim");
        assert!(at8 > at5);
        // Width 1 is the SPEC=0 layout: `conv_width - 1` entries and no speculative window.
        assert_eq!(
            slot(1),
            qkv_n * u64::from(ModelConfig::qwen38_27b().linear_conv_kernel_dim - 1) * 2
        );
    }

    /// Narrow and dedicated wide decode tiles have separate width ranges.
    #[test]
    fn a_verify_width_the_attention_cannot_hold_is_refused() {
        let cfg = ModelConfig::qwen38_27b();
        assert!(Graph::qwen38_27b_at_width(cfg.clone(), 1, 0).is_err());
        // The widest narrow tile builds. Its immediate successor has no registered tile.
        let cap = SaberRules::prod_saber()
            .expect("rules")
            .knobs
            .attn_max_decode_rows
            / cfg.gqa();
        assert_eq!(cap, 10, "the committed rules' 64 rows at GQA 6");
        assert!(Graph::qwen38_27b_at_width(cfg.clone(), 1, cap).is_ok());
        let err = Graph::qwen38_27b_at_width(cfg.clone(), 1, cap + 1)
            .expect_err("a width past the kernel's workgroup");
        let text = err.to_string();
        assert!(text.contains("attn_max_decode_rows"), "{text}");
        assert!(
            text.contains("SPEC="),
            "the message names the SPEC it was asked for: {text}"
        );
        for fp8 in [true, false] {
            let mut wide = cfg.clone();
            wide.kv_cache_fp8 = fp8;
            let graph = Graph::qwen38_27b_at_width(wide, 2, 16).expect("uniform 16-query kernel");
            let decode = graph.ops_for(Bucket::Decode {
                batch: 1,
                verify_width: 16,
            });
            let attn = decode
                .iter()
                .find(|r| r.op.kernel == KernelOp::AttnDecodePaged)
                .unwrap();
            assert_eq!(attn.plugin, Some("saber_attn16"));
        }
        for width in [12, 15, 17] {
            assert!(Graph::qwen38_27b_at_width(cfg.clone(), 2, width).is_err());
        }
        let mut wrong = cfg.clone();
        wrong.head_dim = 128;
        assert!(Graph::qwen38_27b_at_width(wrong, 2, 16).is_err());
        // And SPEC=7 — prod's own — is inside it.
        assert!(Graph::qwen38_27b_at_width(cfg, 1, 8).is_ok());
    }
}
