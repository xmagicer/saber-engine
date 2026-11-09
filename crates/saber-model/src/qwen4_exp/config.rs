//! `Qwen4ExpConfig`: Qwen3.8 Flash-Next's `config.json` (`model_type qwen4_exp`) as typed data.
//!
//! Every field the forward needs is named here — the hyper-connection streams and rank, the PLE
//! n-gram table (rows, width, hash heads, conv), the QSA indexer, the MoE (experts, top-k,
//! intermediate, shared expert, router activation and renormalisation), the GDN geometry, the
//! attention (GQA 12 × 256 with the sigmoid output gate), the MTP layer, vocab and head tying,
//! RoPE / M-RoPE, the vision tower and the quantization block. The reader REFUSES:
//!
//! * a key it does not know, anywhere (top level, `text_config`, `rope_parameters`, `mtp`,
//!   `vision_config`, `quantization_config` and its nested specs) — by its dotted path;
//! * a required key that is absent — by its dotted path;
//! * a value the forward cannot run (a layer schedule that disagrees with
//!   `full_attention_interval`, a PLE layer out of range, an indexer budget upstream refuses, an
//!   MX spec that is not e2m1 + e8m0 per 32, a Quark `exclude` entry that names no module of this
//!   model, a dense linear that Quark would quantize).
//!
//! Values the checkpoint omits and upstream fills in are not silent defaults: each is a
//! [`DeclaredDefault`] with the upstream file that sets it, listed in
//! [`Qwen4ExpConfig::declared_defaults`] and printed by the pack plan.
//!
//! Sources: the checkpoint's `config.json`; upstream 0.29 (`localhost/saber-base:0.29.0-test2`)
//! `upstream/models/qwen4_exp/{config.py, amd/model.py, amd/ple_layer.py}` and
//! `upstream/transformers_utils/configs/qwen3_next.py`; `docs/FLASHNEXT-PLAN.md` §2.

use serde_json::Value;

use super::fields::Fields;
use crate::ModelError;

fn err(msg: impl Into<String>) -> ModelError {
    ModelError::Config(msg.into())
}

/// The kind of one of the 48 decoder layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated delta net (`linear_attention`).
    Gdn,
    /// Qwen Sparse Attention (`full_attention`): dense attention + the top-k block indexer.
    Qsa,
}

/// A value `config.json` does not state and upstream supplies. Named, with its provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct DeclaredDefault {
    /// The config key upstream reads.
    pub key: &'static str,
    /// The value upstream uses when the key is absent.
    pub value: &'static str,
    /// Where upstream sets it.
    pub source: &'static str,
}

/// The upstream defaults this config relies on (each checked ABSENT from the file; a file that
/// states one of them is parsed for it instead).
pub const DECLARED_DEFAULTS: [DeclaredDefault; 4] = [
    DeclaredDefault {
        key: "text_config.norm_topk_prob",
        value: "true",
        source: "upstream/transformers_utils/configs/qwen3_next.py Qwen3NextConfig(norm_topk_prob=True)",
    },
    DeclaredDefault {
        key: "text_config.decoder_sparse_step",
        value: "1",
        source: "upstream/transformers_utils/configs/qwen3_next.py Qwen3NextConfig(decoder_sparse_step=1)",
    },
    DeclaredDefault {
        key: "text_config.mlp_only_layers",
        value: "[]",
        source: "upstream/transformers_utils/configs/qwen3_next.py Qwen3NextConfig(mlp_only_layers=None -> [])",
    },
    DeclaredDefault {
        key: "text_config.seed",
        value: "1234",
        source: "upstream/models/qwen4_exp/amd/ple_layer.py getattr(config, \"seed\", 1234) (PLE hash multipliers)",
    },
];

/// Standard attention (QSA dense part).
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionConfig {
    /// `num_attention_heads`, 24.
    pub num_heads: u32,
    /// `num_key_value_heads`, 2 (GQA 12).
    pub num_kv_heads: u32,
    /// `head_dim`, 256.
    pub head_dim: u32,
    /// `output_gate_type`, `sigmoid`: `q_proj` emits q and a gate per head (`[12288, 2560]`).
    pub output_gate_sigmoid: bool,
    /// `attention_bias`, false.
    pub bias: bool,
    /// `max_position_embeddings`, 262144.
    pub max_position_embeddings: u32,
}

/// RoPE / interleaved M-RoPE (`text_config.rope_parameters`).
#[derive(Clone, Debug, PartialEq)]
pub struct RopeConfig {
    /// `rope_theta`, 1e7.
    pub theta: f64,
    /// `partial_rotary_factor`, 0.25 → 64 rotary dims of 256.
    pub partial_rotary_factor: f64,
    /// `mrope_section`, [11, 11, 10] (sums to rotary_dim / 2).
    pub mrope_section: Vec<u32>,
    /// `mrope_interleaved`, true.
    pub mrope_interleaved: bool,
}

impl RopeConfig {
    /// Rotary dimensions per head.
    #[must_use]
    pub fn rotary_dim(&self, head_dim: u32) -> u32 {
        (f64::from(head_dim) * self.partial_rotary_factor).round() as u32
    }
}

/// Qwen Sparse Attention's indexer.
#[derive(Clone, Debug, PartialEq)]
pub struct QsaConfig {
    /// `indexer_n_heads`, 4.
    pub n_heads: u32,
    /// `indexer_kv_heads`, 1 (upstream's MQA operators require 1).
    pub kv_heads: u32,
    /// `indexer_head_dim`, 128.
    pub head_dim: u32,
    /// `indexer_budget`, 2048 tokens selected per query row.
    pub budget: u32,
    /// `indexer_compress_ratio`, 4: keys mean-pooled in groups of 4.
    pub compress_ratio: u32,
}

impl QsaConfig {
    /// Selected blocks per row: budget / compress ratio = 512.
    #[must_use]
    pub fn block_topk(&self) -> u32 {
        self.budget / self.compress_ratio
    }
    /// `index_qk_proj` rows: (n_heads + kv_heads) × head_dim = 640.
    #[must_use]
    pub fn qk_proj_rows(&self) -> u32 {
        (self.n_heads + self.kv_heads) * self.head_dim
    }
}

/// The gated delta net geometry (the 27B's exactly).
#[derive(Clone, Debug, PartialEq)]
pub struct GdnConfig {
    /// `linear_num_key_heads`, 16.
    pub num_key_heads: u32,
    /// `linear_num_value_heads`, 48.
    pub num_value_heads: u32,
    /// `linear_key_head_dim`, 128.
    pub key_head_dim: u32,
    /// `linear_value_head_dim`, 128.
    pub value_head_dim: u32,
    /// `linear_conv_kernel_dim`, 4.
    pub conv_kernel: u32,
    /// `mamba_ssm_dtype`, `float32` in the checkpoint (Saber's served state dtype is a plan choice).
    pub ssm_dtype_f32: bool,
}

impl GdnConfig {
    /// `in_proj_qkv` rows: 2 × 16 × 128 + 48 × 128 = 10240 (also the conv dim).
    #[must_use]
    pub fn qkv_rows(&self) -> u32 {
        2 * self.num_key_heads * self.key_head_dim + self.num_value_heads * self.value_head_dim
    }
    /// `in_proj_z` rows and `out_proj` K: 48 × 128 = 6144.
    #[must_use]
    pub fn value_dim(&self) -> u32 {
        self.num_value_heads * self.value_head_dim
    }
    /// `q` (= `k`) rows of `in_proj_qkv`: 2048.
    #[must_use]
    pub fn key_dim(&self) -> u32 {
        self.num_key_heads * self.key_head_dim
    }
}

/// The router's activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterActivation {
    /// Softmax over all experts, then top-k (Qwen3-Next's `SparseMoeBlock`).
    Softmax,
}

/// The mixture of experts on every layer.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    /// `num_experts`, 512.
    pub num_experts: u32,
    /// `num_experts_per_tok`, 10.
    pub top_k: u32,
    /// `moe_intermediate_size`, 640.
    pub intermediate: u32,
    /// `shared_expert_intermediate_size`, 640 (+ a `[1, hidden]` sigmoid gate).
    pub shared_intermediate: u32,
    /// Softmax (Qwen3-Next's block; no `scoring_func` field exists in this config family).
    pub router: RouterActivation,
    /// Renormalise the top-k weights to sum to 1 (declared default: true).
    pub norm_topk_prob: bool,
    /// `hidden_act`, `silu` (gate · silu, the SwiGLU of every expert).
    pub act_silu: bool,
    /// `decoder_sparse_step` (declared default 1): every layer is MoE.
    pub decoder_sparse_step: u32,
    /// `mlp_only_layers` (declared default []): no dense-MLP layers.
    pub mlp_only_layers: Vec<u32>,
}

/// Hyper-connections: the multi-stream residual.
#[derive(Clone, Debug, PartialEq)]
pub struct HcConfig {
    /// `hc_count`, 4 streams → a 10,240-wide residual.
    pub count: u32,
    /// `hc_lowrank`, 320: the mix's low-rank bottleneck.
    pub lowrank: u32,
}

/// The per-layer n-gram embedding (PLE).
#[derive(Clone, Debug, PartialEq)]
pub struct PleConfig {
    /// `ple_layer_ids` as the file states them (1-based): [2].
    pub layer_ids_one_based: Vec<u32>,
    /// `ple_embed_dim`, 2560.
    pub embed_dim: u32,
    /// `ple_conv_kernel_size`, 4.
    pub conv_kernel: u32,
    /// `ngram_size`, 3: 2- and 3-grams.
    pub ngram_size: u32,
    /// `heads_per_ngram`, 8.
    pub heads_per_ngram: u32,
    /// `ngram_vocab_size_base`, 20,000,000: each head's table is the next prime above it.
    pub ngram_vocab_size_base: u64,
    /// `make_ngram_vocab_size_divisible_by`, 128.
    pub vocab_divisible_by: u64,
    /// `split_ngram_parts`, 128: the checkpoint's shard count of the table.
    pub split_parts: u32,
    /// The hash seed (declared default 1234; the multipliers are ALSO stored in the checkpoint as
    /// `ple_embedding.layer_multipliers`, and the pack builder checks the two agree).
    pub hash_seed: u64,
}

impl PleConfig {
    /// 0-based layer indices the PLE runs at: [1].
    #[must_use]
    pub fn layer_indices(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.layer_ids_one_based.iter().map(|x| x - 1).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    /// Hash heads: (ngram_size − 1) × heads_per_ngram = 16 rows looked up per token.
    #[must_use]
    pub fn ngram_heads(&self) -> u32 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
    /// One row's width: embed_dim / ngram_heads = 160.
    #[must_use]
    pub fn row_dim(&self) -> u32 {
        self.embed_dim / self.ngram_heads()
    }
    /// The short conv's dilation: upstream passes `ngram_size` (amd/model.py).
    #[must_use]
    pub fn conv_dilation(&self) -> u32 {
        self.ngram_size
    }
    /// Rows of short-conv state per sequence: (kernel − 1) × dilation = 9.
    #[must_use]
    pub fn conv_state_rows(&self) -> u32 {
        (self.conv_kernel - 1) * self.conv_dilation()
    }
}

/// The MTP drafter layer.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpConfig {
    /// `mtp_num_hidden_layers` (= `mtp.num_hidden_layers`), 1.
    pub num_layers: u32,
    /// `mtp.layer_types`: every MTP layer is `full_attention` (QSA + MoE + HC).
    pub layer_kinds: Vec<LayerKind>,
    /// `mtp.hybrid`, true.
    pub hybrid: bool,
    /// `mtp.rope_theta`, 1e7.
    pub rope_theta: f64,
    /// `mtp_use_dedicated_embeddings`, false: the MTP layer shares `embed_tokens` and `lm_head`.
    pub dedicated_embeddings: bool,
}

/// The vision tower (the 27B's).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    /// `depth`, 27.
    pub depth: u32,
    /// `hidden_size`, 1152.
    pub hidden_size: u32,
    /// `num_heads`, 16.
    pub num_heads: u32,
    /// `intermediate_size`, 4304.
    pub intermediate_size: u32,
    /// `out_hidden_size`, 2560 (= the text hidden).
    pub out_hidden_size: u32,
    /// `patch_size`, 16.
    pub patch_size: u32,
    /// `temporal_patch_size`, 2.
    pub temporal_patch_size: u32,
    /// `spatial_merge_size`, 2.
    pub spatial_merge_size: u32,
    /// `in_channels`, 3.
    pub in_channels: u32,
    /// `num_position_embeddings`, 2304.
    pub num_position_embeddings: u32,
    /// `deepstack_visual_indexes`, [].
    pub deepstack_visual_indexes: Vec<u32>,
    /// `hidden_act`, `gelu_pytorch_tanh`.
    pub act_gelu_tanh: bool,
}

/// One MX quantization spec (Quark's `weight` / `input_tensors`).
#[derive(Clone, Debug, PartialEq)]
pub struct MxSpec {
    /// `is_dynamic`: false for weights, true for activations.
    pub dynamic: bool,
}

/// How the checkpoint is quantized.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantConfig {
    /// AMD Quark MXFP4: e2m1 + e8m0 per 32 along K (`scale_calculation_mode even`), on the routed
    /// and shared experts of the 48 main layers; everything in `exclude` stays bf16.
    QuarkMxfp4 {
        /// The weight spec (static).
        weight: MxSpec,
        /// The activation spec (dynamic fp4 per 32 — the card's A4 intent; Saber runs A8).
        input: MxSpec,
        /// Module paths left unquantized (each checked to be a module of this model).
        exclude: Vec<String>,
        /// Quark's `version`.
        version: String,
    },
    /// NVIDIA ModelOpt mixed precision. Two checkpoints: `Qwen3.8-Flash-Next-NVFP4` (PTQ: NVFP4
    /// routed experts, FP8 block-128 MTP experts, FP8 per-tensor PLE table) and FN144's
    /// `Qwen3.8-Flash-Next-FP4-Local` (QAD: NVFP4 routed experts with CSF-coded scales, MXFP8 trunk
    /// and shared experts, W4A16 NVFP4 MTP experts and vision MLPs, NVFP4 PLE rows).
    ModelOpt {
        /// `quant_algo`, `MIXED_PRECISION`.
        algo: String,
        /// The `ignore` globs (`*` = any run of characters), module paths left bf16.
        ignore: Vec<String>,
        /// `quantized_layers`, in file order.
        layers: Vec<ModeloptLayer>,
    },
}

/// One ModelOpt `quantized_layers` algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeloptAlgo {
    /// NVFP4: e2m1 + E4M3 per 16 along K + an f32 global (`weight_scale_2`), W4A4 by the config.
    Nvfp4,
    /// W4A16 NVFP4: the same weight bytes, activations kept 16-bit.
    W4a16Nvfp4,
    /// MXFP8: e4m3 + UE8M0 per 32 along K.
    Mxfp8,
    /// FP8 per tensor.
    Fp8,
    /// FP8 per block of 128, weight only.
    Fp8PbWo,
}

impl ModeloptAlgo {
    /// The config spelling.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            ModeloptAlgo::Nvfp4 => "NVFP4",
            ModeloptAlgo::W4a16Nvfp4 => "W4A16_NVFP4",
            ModeloptAlgo::Mxfp8 => "MXFP8",
            ModeloptAlgo::Fp8 => "FP8",
            ModeloptAlgo::Fp8PbWo => "FP8_PB_WO",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        [
            ModeloptAlgo::Nvfp4,
            ModeloptAlgo::W4a16Nvfp4,
            ModeloptAlgo::Mxfp8,
            ModeloptAlgo::Fp8,
            ModeloptAlgo::Fp8PbWo,
        ]
        .into_iter()
        .find(|a| a.key() == s)
    }
    /// The group sizes this algo is defined at (`None` = no `group_size` key, per tensor).
    #[must_use]
    pub fn group_size(self) -> Option<u32> {
        match self {
            ModeloptAlgo::Nvfp4 | ModeloptAlgo::W4a16Nvfp4 => Some(16),
            ModeloptAlgo::Mxfp8 => Some(32),
            ModeloptAlgo::Fp8PbWo => Some(128),
            ModeloptAlgo::Fp8 => None,
        }
    }
}

/// How a ModelOpt NVFP4 module's E4M3 block scales are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleEncoding {
    /// Plain `weight_scale` E4M3 `[N, K/16]`.
    Plain,
    /// `weight_scale_encoding: csf` -- `weight_scale.nvfp4_csf_fixed` + `..._exceptions`
    /// (`nvfp4_csf`, lossless).
    Csf,
}

/// One `quantized_layers` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeloptLayer {
    /// Module path.
    pub module: String,
    /// Its algorithm.
    pub algo: ModeloptAlgo,
    /// Its scale encoding (only NVFP4 may be `Csf`).
    pub encoding: ScaleEncoding,
}

/// `*`-glob match (ModelOpt's `ignore` uses fnmatch; these checkpoints use `*` only).
#[must_use]
pub fn glob_match(pat: &str, s: &str) -> bool {
    let parts: Vec<&str> = pat.split('*').collect();
    if parts.len() == 1 {
        return pat == s;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if s.len() < first.len() + last.len() || !s.starts_with(first) || !s.ends_with(last) {
        return false;
    }
    let mut at = first.len();
    let end = s.len() - last.len();
    for mid in &parts[1..parts.len() - 1] {
        match s[at..end].find(mid) {
            Some(i) => at += i + mid.len(),
            None => return false,
        }
    }
    true
}

impl QuantConfig {
    /// The ModelOpt entry for a module path, `None` for Quark or an unlisted module.
    #[must_use]
    pub fn modelopt_layer(&self, module: &str) -> Option<&ModeloptLayer> {
        match self {
            QuantConfig::ModelOpt { layers, .. } => layers.iter().find(|l| l.module == module),
            QuantConfig::QuarkMxfp4 { .. } => None,
        }
    }
    /// Whether a ModelOpt `ignore` glob covers a module path.
    #[must_use]
    pub fn modelopt_ignored(&self, module: &str) -> bool {
        match self {
            QuantConfig::ModelOpt { ignore, .. } => ignore.iter().any(|g| glob_match(g, module)),
            QuantConfig::QuarkMxfp4 { .. } => false,
        }
    }
    /// FN144: the FP4-Local checkpoint (ModelOpt with CSF-coded NVFP4 routed experts).
    #[must_use]
    pub fn is_fp4_local(&self) -> bool {
        match self {
            QuantConfig::ModelOpt { layers, .. } => layers
                .iter()
                .any(|l| l.encoding == ScaleEncoding::Csf && l.module.ends_with(".mlp.experts")),
            QuantConfig::QuarkMxfp4 { .. } => false,
        }
    }
}

/// Qwen3.8 Flash-Next's model facts.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpConfig {
    /// `hidden_size`, 2560.
    pub hidden_size: u32,
    /// The 48 layer kinds, in order (`layer_types`, checked against `full_attention_interval`).
    pub layers: Vec<LayerKind>,
    /// `full_attention_interval`, 4.
    pub full_attention_interval: u32,
    /// `vocab_size`, 248,320 (embedding rows).
    pub vocab_size: u32,
    /// `tie_word_embeddings`: false (checked equal at the top level and in `text_config`).
    pub tie_word_embeddings: bool,
    /// `rms_norm_eps`, 1e-6.
    pub rms_norm_eps: f64,
    /// `bos_token_id`, 248044.
    pub bos_token_id: u32,
    /// `eos_token_id`, 248044.
    pub eos_token_id: u32,
    /// Attention.
    pub attention: AttentionConfig,
    /// RoPE.
    pub rope: RopeConfig,
    /// QSA indexer.
    pub qsa: QsaConfig,
    /// GDN.
    pub gdn: GdnConfig,
    /// MoE.
    pub moe: MoeConfig,
    /// Hyper-connections.
    pub hc: HcConfig,
    /// PLE.
    pub ple: PleConfig,
    /// MTP.
    pub mtp: MtpConfig,
    /// Vision tower.
    pub vision: VisionConfig,
    /// `image_token_id`, `video_token_id`, `vision_start_token_id`, `vision_end_token_id`.
    pub vision_tokens: [u32; 4],
    /// Quantization.
    pub quant: QuantConfig,
}

fn layer_kind(s: &str, path: &str) -> Result<LayerKind, ModelError> {
    match s {
        "linear_attention" => Ok(LayerKind::Gdn),
        "full_attention" => Ok(LayerKind::Qsa),
        other => Err(err(format!("{path}: unknown layer type `{other}`"))),
    }
}

/// A key that must be ABSENT for its declared default to apply; if the file states it, it is
/// parsed and must be a value the forward supports.
fn declared_bool(t: &mut Fields<'_>, key: &str, default: bool) -> Result<bool, ModelError> {
    match t.optional(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| err(format!("text_config.{key}: expected a boolean, got {v}"))),
    }
}

impl Qwen4ExpConfig {
    /// Parse `config.json`. Refuses unknown and missing fields by name.
    ///
    /// # Errors
    /// [`ModelError::Config`] naming the offending field, or a JSON parse error.
    pub fn from_config_json(json: &str) -> Result<Self, ModelError> {
        let root: Value = serde_json::from_str(json)?;
        let mut top = Fields::new("", &root)?;

        let arch = top.array("architectures")?;
        if arch.len() != 1 || arch[0].as_str() != Some("Qwen4ExpForConditionalGeneration") {
            return Err(err(format!(
                "architectures: expected [\"Qwen4ExpForConditionalGeneration\"], got {}",
                Value::Array(arch.clone())
            )));
        }
        top.expect("model_type", &Value::from("qwen4_exp"))?;
        top.ack("transformers_version");
        // The NVFP4 repo states the dtype at the top level too; the Quark repo does not.
        if let Some(d) = top.optional("dtype") {
            if d != "bfloat16" {
                return Err(err(format!("dtype: expected \"bfloat16\", got {d}")));
            }
        }
        top.expect("language_model_only", &Value::Bool(false))?;
        let vision_tokens = [
            top.u32("image_token_id")?,
            top.u32("video_token_id")?,
            top.u32("vision_start_token_id")?,
            top.u32("vision_end_token_id")?,
        ];
        let tie_top = top.bool("tie_word_embeddings")?;

        // ---- text_config ------------------------------------------------------------------
        let mut t = top.object("text_config")?;
        t.expect("model_type", &Value::from("qwen4_exp_text"))?;
        t.expect("dtype", &Value::from("bfloat16"))?;
        for k in [
            "attention_dropout",
            "initializer_range",
            "output_router_logits",
            "router_aux_loss_coef",
            "use_cache",
        ] {
            // Training-time or HF-generation fields; the forward never reads them.
            t.ack(k);
        }
        match t.value("pad_token_id")? {
            Value::Null => {}
            v => {
                return Err(err(format!(
                    "text_config.pad_token_id: expected null, got {v}"
                )))
            }
        }
        let hidden_size = t.u32("hidden_size")?;
        let num_layers = t.u32("num_hidden_layers")?;
        let interval = t.u32("full_attention_interval")?;
        let types = t.array("layer_types")?;
        if types.len() != num_layers as usize {
            return Err(err(format!(
                "text_config.layer_types: {} entries, num_hidden_layers is {num_layers}",
                types.len()
            )));
        }
        let mut layers = Vec::with_capacity(types.len());
        for (i, v) in types.iter().enumerate() {
            let path = format!("text_config.layer_types[{i}]");
            let kind = layer_kind(v.as_str().unwrap_or(""), &path)?;
            let want = if (i as u32 + 1) % interval == 0 {
                LayerKind::Qsa
            } else {
                LayerKind::Gdn
            };
            if kind != want {
                return Err(err(format!(
                    "{path}: {kind:?}, but full_attention_interval {interval} puts {want:?} there"
                )));
            }
            layers.push(kind);
        }
        let vocab_size = t.u32("vocab_size")?;
        let tie_text = t.bool("tie_word_embeddings")?;
        if tie_text != tie_top {
            return Err(err(format!(
                "tie_word_embeddings: top level {tie_top}, text_config {tie_text}"
            )));
        }
        let rms_norm_eps = t.f64("rms_norm_eps")?;
        let bos_token_id = t.u32("bos_token_id")?;
        let eos_token_id = t.u32("eos_token_id")?;
        if t.str("hidden_act")? != "silu" {
            return Err(err("text_config.hidden_act: only `silu` is supported"));
        }

        let output_gate = t.str("output_gate_type")?;
        if output_gate != "sigmoid" {
            return Err(err(format!(
                "text_config.output_gate_type: `{output_gate}` (only `sigmoid` is supported)"
            )));
        }
        let attention = AttentionConfig {
            num_heads: t.u32("num_attention_heads")?,
            num_kv_heads: t.u32("num_key_value_heads")?,
            head_dim: t.u32("head_dim")?,
            output_gate_sigmoid: true,
            bias: t.bool("attention_bias")?,
            max_position_embeddings: t.u32("max_position_embeddings")?,
        };
        if attention.bias {
            return Err(err("text_config.attention_bias: true is not supported"));
        }
        if attention.num_kv_heads == 0 || attention.num_heads % attention.num_kv_heads != 0 {
            return Err(err(format!(
                "text_config.num_key_value_heads: {} does not divide num_attention_heads {}",
                attention.num_kv_heads, attention.num_heads
            )));
        }

        let prf = t.f64("partial_rotary_factor")?;
        let mut rp = t.object("rope_parameters")?;
        let rope = RopeConfig {
            theta: rp.f64("rope_theta")?,
            partial_rotary_factor: rp.f64("partial_rotary_factor")?,
            mrope_section: rp.u32_array("mrope_section")?,
            mrope_interleaved: rp.bool("mrope_interleaved")?,
        };
        if rp.str("rope_type")? != "default" {
            return Err(err("text_config.rope_parameters.rope_type: only `default`"));
        }
        rp.finish()?;
        if (rope.partial_rotary_factor - prf).abs() > 0.0 {
            return Err(err(format!(
                "partial_rotary_factor: text_config {prf}, rope_parameters {}",
                rope.partial_rotary_factor
            )));
        }
        let half = rope.rotary_dim(attention.head_dim) / 2;
        if rope.mrope_section.iter().sum::<u32>() != half {
            return Err(err(format!(
                "text_config.rope_parameters.mrope_section: {:?} sums to {}, rotary_dim/2 is {half}",
                rope.mrope_section,
                rope.mrope_section.iter().sum::<u32>()
            )));
        }

        let qsa = QsaConfig {
            n_heads: t.u32("indexer_n_heads")?,
            kv_heads: t.u32("indexer_kv_heads")?,
            head_dim: t.u32("indexer_head_dim")?,
            budget: t.u32("indexer_budget")?,
            compress_ratio: t.u32("indexer_compress_ratio")?,
        };
        // upstream config.py _validate_qsa_config
        if qsa.kv_heads != 1 {
            return Err(err(
                "text_config.indexer_kv_heads: upstream's QSA requires 1",
            ));
        }
        if qsa.compress_ratio == 0
            || qsa.budget % qsa.compress_ratio != 0
            || !matches!(qsa.block_topk(), 512 | 2048)
        {
            return Err(err(format!(
                "text_config.indexer_budget / indexer_compress_ratio: {} / {} (upstream needs a \
                 block top-k of 512 or 2048)",
                qsa.budget, qsa.compress_ratio
            )));
        }
        if rope.rotary_dim(attention.head_dim) > qsa.head_dim {
            return Err(err(
                "text_config.indexer_head_dim: smaller than the rotary dim",
            ));
        }

        let gdn = GdnConfig {
            num_key_heads: t.u32("linear_num_key_heads")?,
            num_value_heads: t.u32("linear_num_value_heads")?,
            key_head_dim: t.u32("linear_key_head_dim")?,
            value_head_dim: t.u32("linear_value_head_dim")?,
            conv_kernel: t.u32("linear_conv_kernel_dim")?,
            ssm_dtype_f32: match t.str("mamba_ssm_dtype")? {
                "float32" => true,
                other => {
                    return Err(err(format!(
                        "text_config.mamba_ssm_dtype: `{other}` (the checkpoint's state is float32)"
                    )))
                }
            },
        };

        let moe = MoeConfig {
            num_experts: t.u32("num_experts")?,
            top_k: t.u32("num_experts_per_tok")?,
            intermediate: t.u32("moe_intermediate_size")?,
            shared_intermediate: t.u32("shared_expert_intermediate_size")?,
            router: RouterActivation::Softmax,
            norm_topk_prob: declared_bool(&mut t, "norm_topk_prob", true)?,
            act_silu: true,
            decoder_sparse_step: match t.optional("decoder_sparse_step") {
                None => 1,
                Some(v) => v
                    .as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or_else(|| {
                        err(format!(
                            "text_config.decoder_sparse_step: expected an integer, got {v}"
                        ))
                    })?,
            },
            mlp_only_layers: match t.optional("mlp_only_layers") {
                None | Some(Value::Null) => Vec::new(),
                Some(_) => t.u32_array("mlp_only_layers")?,
            },
        };
        if moe.decoder_sparse_step != 1 || !moe.mlp_only_layers.is_empty() {
            return Err(err(
                "text_config.decoder_sparse_step / mlp_only_layers: only an all-MoE stack is supported",
            ));
        }
        if moe.top_k == 0 || moe.top_k > moe.num_experts {
            return Err(err("text_config.num_experts_per_tok: out of range"));
        }

        let hc = HcConfig {
            count: t.u32("hc_count")?,
            lowrank: t.u32("hc_lowrank")?,
        };
        if hc.count <= 1 || hc.lowrank == 0 {
            return Err(err(format!(
                "text_config.hc_count / hc_lowrank: {} / {} (upstream requires count > 1)",
                hc.count, hc.lowrank
            )));
        }

        // FN144: FP4-Local names its PLE table's storage (`nvfp4`); cross-checked against the
        // quantization map below (the table's module must be NVFP4 there).
        let ple_nvfp4 = match t.optional("ple_embedding_dtype") {
            None => false,
            Some(v) if v.as_str() == Some("nvfp4") => true,
            Some(v) => {
                return Err(err(format!(
                    "text_config.ple_embedding_dtype: `{v}` (only nvfp4)"
                )))
            }
        };
        let ple = PleConfig {
            layer_ids_one_based: t.u32_array("ple_layer_ids")?,
            embed_dim: t.u32("ple_embed_dim")?,
            conv_kernel: t.u32("ple_conv_kernel_size")?,
            ngram_size: t.u32("ngram_size")?,
            heads_per_ngram: t.u32("heads_per_ngram")?,
            ngram_vocab_size_base: t.u64("ngram_vocab_size_base")?,
            vocab_divisible_by: t.u64("make_ngram_vocab_size_divisible_by")?,
            split_parts: t.u32("split_ngram_parts")?,
            hash_seed: match t.optional("seed") {
                None => 1234,
                Some(v) => v.as_u64().ok_or_else(|| {
                    err(format!("text_config.seed: expected an integer, got {v}"))
                })?,
            },
        };
        if ple.layer_ids_one_based.is_empty()
            || ple
                .layer_ids_one_based
                .iter()
                .any(|&l| l == 0 || l > num_layers)
        {
            return Err(err(format!(
                "text_config.ple_layer_ids: {:?} (1-based, 1..={num_layers}, at least one)",
                ple.layer_ids_one_based
            )));
        }
        if ple.ngram_size < 2
            || ple.heads_per_ngram == 0
            || ple.embed_dim % ple.ngram_heads() != 0
            || ple.conv_kernel == 0
            || ple.split_parts == 0
            || ple.vocab_divisible_by == 0
        {
            return Err(err(
                "text_config PLE fields: upstream's _validate_ple_config refuses them",
            ));
        }
        if ple.embed_dim != hidden_size {
            return Err(err(format!(
                "text_config.ple_embed_dim: {} (the PLE adds into the hidden, {hidden_size})",
                ple.embed_dim
            )));
        }

        // NVIDIA's repo states `number_of_conv_states` (= ngram_size, the PLE short conv's
        // dilation); upstream 0.29 never reads it. Held equal to what the forward uses.
        if let Some(n) = t.optional("number_of_conv_states") {
            if n.as_u64() != Some(u64::from(ple.conv_dilation())) {
                return Err(err(format!(
                    "text_config.number_of_conv_states: {n}, the PLE conv dilation is {}",
                    ple.conv_dilation()
                )));
            }
        }
        let mtp_layers = t.u32("mtp_num_hidden_layers")?;
        let mut m = t.object("mtp")?;
        let m_layers = m.u32("num_hidden_layers")?;
        if m_layers != mtp_layers {
            return Err(err(format!(
                "text_config.mtp.num_hidden_layers {m_layers} != mtp_num_hidden_layers {mtp_layers}"
            )));
        }
        let mut layer_kinds = Vec::new();
        for (i, v) in m.array("layer_types")?.iter().enumerate() {
            layer_kinds.push(layer_kind(
                v.as_str().unwrap_or(""),
                &format!("text_config.mtp.layer_types[{i}]"),
            )?);
        }
        if layer_kinds.len() != mtp_layers as usize
            || layer_kinds.iter().any(|k| *k != LayerKind::Qsa)
        {
            return Err(err(
                "text_config.mtp.layer_types: one `full_attention` entry per MTP layer",
            ));
        }
        match m.value("mtp_use_hidden_state_from_layer")? {
            Value::Null => {}
            v => {
                return Err(err(format!(
                    "text_config.mtp.mtp_use_hidden_state_from_layer: {v} (only null: the \
                     pre-final-mixer stream)"
                )))
            }
        }
        let mtp = MtpConfig {
            num_layers: mtp_layers,
            layer_kinds,
            hybrid: m.bool("hybrid")?,
            rope_theta: m.f64("rope_theta")?,
            dedicated_embeddings: t.bool("mtp_use_dedicated_embeddings")?,
        };
        m.finish()?;
        if mtp.dedicated_embeddings {
            return Err(err(
                "text_config.mtp_use_dedicated_embeddings: true (the pack shares embed/head)",
            ));
        }
        t.finish()?;

        // ---- vision_config -----------------------------------------------------------------
        let mut v = top.object("vision_config")?;
        match v.str("model_type")? {
            // Quark's repo says `qwen4_exp`, NVIDIA's `qwen4_exp_vision`: the same tower.
            "qwen4_exp" | "qwen4_exp_vision" => {}
            other => return Err(err(format!("vision_config.model_type: `{other}`"))),
        }
        v.ack("initializer_range");
        if let Some(d) = v.optional("dtype") {
            if d != "bfloat16" {
                return Err(err(format!(
                    "vision_config.dtype: expected \"bfloat16\", got {d}"
                )));
            }
        }
        let vision = VisionConfig {
            depth: v.u32("depth")?,
            hidden_size: v.u32("hidden_size")?,
            num_heads: v.u32("num_heads")?,
            intermediate_size: v.u32("intermediate_size")?,
            out_hidden_size: v.u32("out_hidden_size")?,
            patch_size: v.u32("patch_size")?,
            temporal_patch_size: v.u32("temporal_patch_size")?,
            spatial_merge_size: v.u32("spatial_merge_size")?,
            in_channels: v.u32("in_channels")?,
            num_position_embeddings: v.u32("num_position_embeddings")?,
            deepstack_visual_indexes: v.u32_array("deepstack_visual_indexes")?,
            act_gelu_tanh: v.str("hidden_act")? == "gelu_pytorch_tanh",
        };
        if !vision.act_gelu_tanh {
            return Err(err("vision_config.hidden_act: only `gelu_pytorch_tanh`"));
        }
        if vision.out_hidden_size != hidden_size {
            return Err(err(
                "vision_config.out_hidden_size: must equal the text hidden",
            ));
        }
        v.finish()?;

        let mut cfg = Self {
            hidden_size,
            layers,
            full_attention_interval: interval,
            vocab_size,
            tie_word_embeddings: tie_top,
            rms_norm_eps,
            bos_token_id,
            eos_token_id,
            attention,
            rope,
            qsa,
            gdn,
            moe,
            hc,
            ple,
            mtp,
            vision,
            vision_tokens,
            quant: QuantConfig::ModelOpt {
                algo: String::new(),
                ignore: Vec::new(),
                layers: Vec::new(),
            },
        };

        // ---- quantization_config ------------------------------------------------------------
        let mut q = top.object("quantization_config")?;
        cfg.quant = match q.str("quant_method")? {
            "quark" => cfg.parse_quark(&mut q)?,
            "modelopt" => parse_modelopt(&mut q)?,
            other => {
                return Err(err(format!(
                    "quantization_config.quant_method: `{other}` (quark or modelopt)"
                )))
            }
        };
        q.finish()?;
        top.finish()?;
        if ple_nvfp4 {
            let m = format!(
                "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding",
                cfg.ple.layer_indices().first().copied().unwrap_or(0)
            );
            if cfg.quant.modelopt_layer(&m).map(|l| l.algo) != Some(ModeloptAlgo::Nvfp4) {
                return Err(err(format!(
                    "text_config.ple_embedding_dtype nvfp4: `{m}` is not NVFP4 in quantized_layers"
                )));
            }
        }
        Ok(cfg)
    }

    fn parse_quark(&self, q: &mut Fields<'_>) -> Result<QuantConfig, ModelError> {
        for k in ["algo_config", "softmax_quant_spec"] {
            match q.value(k)? {
                Value::Null => {}
                v => {
                    return Err(err(format!(
                        "quantization_config.{k}: expected null, got {v}"
                    )))
                }
            }
        }
        for k in [
            "layer_type_quant_config",
            "layer_quant_config",
            "kv_cache_quant_config",
        ] {
            match q.value(k)? {
                Value::Object(o) if o.is_empty() => {}
                v => {
                    return Err(err(format!(
                    "quantization_config.{k}: expected {{}} (per-layer overrides and a quantized \
                         KV cache are not supported), got {v}"
                )))
                }
            }
        }
        q.expect("kv_cache_post_rope", &Value::Bool(false))?;
        q.expect("quant_mode", &Value::from("eager_mode"))?;
        let version = q.str("version")?.to_string();
        let mut ex = q.object("export")?;
        ex.expect("kv_cache_group", &Value::Array(Vec::new()))?;
        ex.ack("min_kv_scale");
        ex.expect("pack_method", &Value::from("reorder"))?;
        ex.expect("weight_format", &Value::from("real_quantized"))?;
        ex.expect("weight_merge_groups", &Value::Null)?;
        ex.finish()?;

        let mut g = q.object("global_quant_config")?;
        let weight = mx_spec(g.object("weight")?, false)?;
        let input = mx_spec(g.object("input_tensors")?, true)?;
        for k in ["output_tensors", "bias", "target_device"] {
            g.expect(k, &Value::Null)?;
        }
        g.finish()?;

        let exclude: Vec<String> = q
            .array("exclude")?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| err(format!("quantization_config.exclude: non-string {x}")))
            })
            .collect::<Result<_, _>>()?;
        let mut known: std::collections::HashSet<_> = self.linear_modules().into_iter().collect();
        // Quark 0.14 also lists unquantized embedding and patch-projection modules.
        // A PLE shard must belong to a configured PLE layer and be within its table.
        for layer in self.ple.layer_indices() {
            for shard in 0..self.ple.split_parts {
                known.insert(format!(
                    "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{shard}"
                ));
            }
        }
        known.insert("model.visual.pos_embed".into());
        known.insert("model.visual.patch_embed.proj".into());
        for e in &exclude {
            if !known.contains(e) {
                return Err(err(format!(
                    "quantization_config.exclude: `{e}` is not a linear module or a recognized embedding/patch module of this model"
                )));
            }
        }
        // Every MAIN-model dense linear must be excluded (Quark quantizes only the experts).
        for m in self.main_dense_linears() {
            if !exclude.contains(&m) {
                return Err(err(format!(
                    "quantization_config.exclude: dense linear `{m}` is not excluded, so Quark \
                     quantized it; the pack has no source kind for an MXFP4 dense layer"
                )));
            }
        }
        Ok(QuantConfig::QuarkMxfp4 {
            weight,
            input,
            exclude,
            version,
        })
    }

    /// The declared upstream defaults this parse relied on (the keys absent from the file).
    #[must_use]
    pub fn declared_defaults(&self) -> &'static [DeclaredDefault] {
        &DECLARED_DEFAULTS
    }

    /// Number of decoder layers.
    #[must_use]
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// The HC stream width: hidden × hc_count = 10,240.
    #[must_use]
    pub fn hc_width(&self) -> u32 {
        self.hidden_size * self.hc.count
    }

    /// `q_proj` rows: heads × head_dim × 2 (q and the sigmoid gate) = 12,288.
    #[must_use]
    pub fn q_proj_rows(&self) -> u32 {
        self.attention.num_heads * self.attention.head_dim * 2
    }

    /// The PLE table's total row count: the next-prime head sizes padded to the divisor.
    #[must_use]
    pub fn ple_table_rows(&self) -> u64 {
        // `ple_dense_layer_id` is the index AMONG the PLE layers (upstream amd/model.py), 0 here.
        let dense_id = 0u64;
        let heads = u64::from(self.ple.ngram_heads());
        let mut total = 0u64;
        for local in 0..heads {
            let global = dense_id * heads + local;
            total += nth_prime_after(self.ple.ngram_vocab_size_base - 1, global + 1);
        }
        total.div_ceil(self.ple.vocab_divisible_by) * self.ple.vocab_divisible_by
    }

    /// Rows per checkpoint shard of the PLE table: ceil(rows / split_parts) (upstream
    /// `load_weights`).
    #[must_use]
    pub fn ple_rows_per_shard(&self) -> u64 {
        self.ple_table_rows()
            .div_ceil(u64::from(self.ple.split_parts))
    }

    /// Every linear module path of the model (target, MTP, vision), as Quark names them.
    #[must_use]
    pub fn linear_modules(&self) -> Vec<String> {
        let mut out = self.main_dense_linears();
        for (l, _) in self.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{l}.mlp");
            for e in 0..self.moe.num_experts {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    out.push(format!("{p}.experts.{e}.{proj}"));
                }
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                out.push(format!("{p}.shared_expert.{proj}"));
            }
        }
        out.push("mtp.fc_embedding".into());
        out.push("mtp.fc_hidden".into());
        for w in ["input_mix_weight_down", "input_mix_weight_up"] {
            out.push(format!("mtp.hyper_connection_mixer.{w}"));
        }
        for m in 0..self.mtp.num_layers {
            let p = format!("mtp.layers.{m}");
            push_layer_common(&mut out, &p);
            push_qsa(&mut out, &p);
            for e in 0..self.moe.num_experts {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    out.push(format!("{p}.mlp.experts.{e}.{proj}"));
                }
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                out.push(format!("{p}.mlp.shared_expert.{proj}"));
            }
        }
        for b in 0..self.vision.depth {
            for s in ["attn.proj", "attn.qkv", "mlp.linear_fc1", "mlp.linear_fc2"] {
                out.push(format!("model.visual.blocks.{b}.{s}"));
            }
        }
        out.push("model.visual.merger.linear_fc1".into());
        out.push("model.visual.merger.linear_fc2".into());
        out
    }

    /// The main model's dense (non-expert) linears: the set Quark must leave in bf16.
    #[must_use]
    pub fn main_dense_linears(&self) -> Vec<String> {
        let mut out = vec!["lm_head".to_string()];
        for w in ["input_mix_weight_down", "input_mix_weight_up"] {
            out.push(format!("model.language_model.hyper_connection_mixer.{w}"));
        }
        let ple_layers = self.ple.layer_indices();
        for (l, kind) in self.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{l}");
            push_layer_common(&mut out, &p);
            match kind {
                LayerKind::Gdn => {
                    for s in [
                        "conv1d",
                        "in_proj_a",
                        "in_proj_b",
                        "in_proj_qkv",
                        "in_proj_z",
                        "out_proj",
                    ] {
                        out.push(format!("{p}.linear_attn.{s}"));
                    }
                }
                LayerKind::Qsa => push_qsa(&mut out, &p),
            }
            if ple_layers.contains(&(l as u32)) {
                for s in ["conv1d", "key_proj", "value_proj"] {
                    out.push(format!("{p}.ple.{s}"));
                }
            }
        }
        out
    }
}

fn push_layer_common(out: &mut Vec<String>, p: &str) {
    for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
        for w in [
            "block_inject_weight",
            "input_mix_weight_down",
            "input_mix_weight_up",
        ] {
            out.push(format!("{p}.{hc}.{w}"));
        }
    }
    out.push(format!("{p}.mlp.gate"));
    out.push(format!("{p}.mlp.shared_expert_gate"));
}

fn push_qsa(out: &mut Vec<String>, p: &str) {
    for s in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "indexer.index_qk_proj",
    ] {
        out.push(format!("{p}.self_attn.{s}"));
    }
}

fn mx_spec(mut s: Fields<'_>, dynamic: bool) -> Result<MxSpec, ModelError> {
    s.expect("dtype", &Value::from("fp4"))?;
    s.expect("is_dynamic", &Value::Bool(dynamic))?;
    s.expect("qscheme", &Value::from("per_group"))?;
    s.expect("ch_axis", &Value::from(-1))?;
    s.expect("group_size", &Value::from(32))?;
    s.expect("block_size", &Value::Null)?;
    s.expect("symmetric", &Value::Null)?;
    s.expect("round_method", &Value::from("half_even"))?;
    s.expect("scale_type", &Value::from("float"))?;
    // Quark 0.14 serializes the zero-point storage type even for this MX format,
    // which has no additive zero point. Older exports omit the annotation.
    if s.optional("zero_point_type").is_some() {
        s.expect("zero_point_type", &Value::from("int32"))?;
    }
    s.expect("scale_format", &Value::from("e8m0"))?;
    s.expect("scale_calculation_mode", &Value::from("even"))?;
    s.expect("mx_element_dtype", &Value::Null)?;
    s.expect("observer_cls", &Value::from("PerBlockMXObserver"))?;
    s.expect("is_scale_quant", &Value::Bool(false))?;
    s.ack("enable_buffer_reuse");
    s.ack("max_input_numel");
    s.finish()?;
    Ok(MxSpec { dynamic })
}

fn parse_modelopt(q: &mut Fields<'_>) -> Result<QuantConfig, ModelError> {
    let algo = q.str("quant_algo")?.to_string();
    if algo != "MIXED_PRECISION" {
        return Err(err(format!("quantization_config.quant_algo: `{algo}`")));
    }
    q.ack("producer");
    // The groups are validated through `quantized_layers`, which names every module's algo.
    q.ack("config_groups");
    // FP4-Local states `kv_cache_quant_algo: null`; a KV recipe is refused (ours is unchanged).
    if let Some(v) = q.optional("kv_cache_quant_algo") {
        if !v.is_null() {
            return Err(err(format!(
                "quantization_config.kv_cache_quant_algo: `{v}` (only null)"
            )));
        }
    }
    let mut ignore = Vec::new();
    for (i, g) in q.array("ignore")?.iter().enumerate() {
        match g.as_str() {
            Some(s) if !s.is_empty() && !s.contains(['?', '[']) => ignore.push(s.to_string()),
            _ => {
                return Err(err(format!(
                    "quantization_config.ignore[{i}]: `{g}` (a `*` glob)"
                )))
            }
        }
    }
    let mut layers = Vec::new();
    let mut ql = q.object("quantized_layers")?;
    let keys: Vec<String> = match q.optional("quantized_layers") {
        Some(Value::Object(o)) => o.keys().cloned().collect(),
        _ => Vec::new(),
    };
    for k in keys {
        let path = format!("quantization_config.quantized_layers.{k}");
        let mut e = ql.object(&k)?;
        let a = e.str("quant_algo")?.to_string();
        let Some(al) = ModeloptAlgo::parse(&a) else {
            return Err(err(format!("{path}.quant_algo: `{a}`")));
        };
        let gs = e.optional("group_size").map(Value::as_u64);
        if gs != al.group_size().map(|g| Some(u64::from(g))) {
            return Err(err(format!(
                "{path}.group_size: {gs:?} (`{a}` is defined at {:?})",
                al.group_size()
            )));
        }
        let encoding = match e.optional("weight_scale_encoding") {
            None => ScaleEncoding::Plain,
            Some(v) if v.as_str() == Some("csf") && al == ModeloptAlgo::Nvfp4 => ScaleEncoding::Csf,
            Some(v) => {
                return Err(err(format!(
                    "{path}.weight_scale_encoding: `{v}` (only `csf`, only on NVFP4)"
                )))
            }
        };
        e.finish()?;
        layers.push(ModeloptLayer {
            module: k,
            algo: al,
            encoding,
        });
    }
    ql.finish()?;
    Ok(QuantConfig::ModelOpt {
        algo,
        ignore,
        layers,
    })
}

fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    // Deterministic Miller-Rabin for 64-bit (upstream `_is_prime_64` uses the same bases).
    let mut d = n - 1;
    let mut s = 0;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    let mulmod = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) % u128::from(n)) as u64;
    let powmod = |mut b: u64, mut e: u64| {
        let mut r = 1u64;
        b %= n;
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        r
    };
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// The `count`-th prime strictly greater than `start` (upstream `_nth_prime_after`).
#[must_use]
pub fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut p = start;
    for _ in 0..count {
        let mut c = p + 1;
        while !is_prime_u64(c) {
            c += 1;
        }
        p = c;
    }
    p
}
