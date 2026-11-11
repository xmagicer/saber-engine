//! The autotune sweep: enumerate a shape's legal variants, then time them **interleaved**.
//!
//! > Autotune cache keys must include **M** [...] The protocol must discard warm-ups **per shape**
//! > — the first-measured variant per shape reads ~17% inflated, so a naive measure-in-order
//! > autotuner systematically rejects whatever it measured first, silently and forever.
//! >
//! > — ARCHITECTURE.md, *Kernel plane*
//!
//! This module is the measurement half of `saber tune`. It owns no policy: the key, the warm-up
//! discard and the record belong to `saber_plan::Autotuner`, and *which* kernel a knob setting
//! selects belongs to `saber_kernels::rules::resolve`. What is here is (a) enumerating the legal
//! knob settings out of the rules table, (b) allocating random-but-valid inputs, and (c) timing.
//!
//! # Variants are enumerated from the rules, never listed here
//!
//! One [`saber_plan::Implementation`] row covers many kernel symbols — the decode plugin alone
//! holds 121. The variant set for a shape is therefore *what `resolve` can produce for that shape
//! under different knob values*, and the knob values come out of the rules' own tables:
//!
//! | axis | where its legal values come from |
//! |---|---|
//! | split-K | `split_k_ladder` |
//! | decode `bk` | every `bk` in `decode_bk`, `decode_band` and `k_align_override` |
//! | folded / A-tiled `tn` | every `tn` in `folded_tn` |
//! | A-tiled `lbk` | every `lbk` in `atiled_lbk` |
//! | quant block width | the two `quant_*_block` widths the threshold knob selects between |
//!
//! A setting is offered to `resolve` by *overlaying* it on a clone of the rules and asking what
//! comes out — so a variant's descriptor is built by the same function the serve path uses, and
//! nothing here re-derives a grid, a tile or a kernarg offset. Settings the rules refuse, and
//! settings whose symbol is not in the code object (there is no `bk64_ks2`, no `tn2_lbk64`), are
//! reported as refusals rather than silently dropped.
//!
//! `wperm` and `decode_nt` are **not** enumerated: they name the weight layout the checkpoint was
//! packed in, so they are a property of the tier and not a tile choice. `epifast` is not
//! enumerated either — the rules carry it as one knob with no alternatives table, and it is
//! bit-identical by construction.
//!
//! # The A-tiled threshold is a layout, not a tile
//!
//! `a_tiled(m, rules)` is read by the producer *and* the consumer, so at `M >= a_tiled_min_m` the
//! activations physically are in fragment-tiled order and only the A-tiled kernel can read them.
//! Above the threshold this module therefore offers A-tiled variants only, and below it offers
//! decode and folded. Timing a folded tile against an A-tiled one at the same M would be timing
//! two different edges of the graph.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use saber_hip::{DevicePtr, EventFlags, Stream};
use saber_kernels::glue::{self, GateShape, GlueRules, KvShape, QkShape};
use saber_kernels::rules::{
    a_tiled, resolve, BandRow, LaunchRules, LbkRow, Op, Shape, TnRow, WmRow,
};
use saber_kernels::saber::{
    attn_decode_combine, gdn_decode_pair_recurrent, AttnShape, GdnPrefillShape, GdnShape,
    NormShape, SaberRules,
};
use saber_kernels::{saber, LaunchDescriptor, Manifest};
use saber_model::graph::Phase;

use crate::forward::{
    i64_bytes, rope_cache, SeqFacts, StepIndex, FIRST_STATE_SLOT, KV_BLOCK, ROTARY_DIM,
};

use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};

/// Which kernel symbols the committed code objects actually contain.
///
/// Read from the `*.manifest.toml` beside each blob rather than from a loaded module: the loader
/// has already proved the manifest matches the `.hsaco` symbol for symbol
/// (`Manifest::validate_against_hsaco`), so the manifest is a faithful index and enumeration stays
/// testable on a host with no ROCm. A symbol that slipped through anyway is still refused at
/// launch, by name, in [`Launcher::launch`].
#[derive(Debug, Clone, Default)]
pub struct SymbolIndex {
    have: BTreeSet<(String, String)>,
}

impl SymbolIndex {
    /// Every `(plugin, kernel)` declared by the manifests in a prebuilt directory.
    pub fn from_manifest_dir(dir: &Path) -> Result<Self> {
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error::io(dir, e))?
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(".manifest.toml"))
            .collect();
        paths.sort();
        let mut have = BTreeSet::new();
        for path in paths {
            let m = Manifest::load(&path)?;
            for k in &m.kernel {
                have.insert((m.plugin.clone(), k.name.clone()));
            }
        }
        Ok(SymbolIndex { have })
    }

    /// Whether this plugin ships that symbol.
    #[must_use]
    pub fn has(&self, plugin: &str, kernel: &str) -> bool {
        self.have.contains(&(plugin.to_owned(), kernel.to_owned()))
    }

    /// How many symbols are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.have.len()
    }

    /// Whether nothing is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.have.is_empty()
    }
}

/// One legal (implementation, knob setting) pair for a shape, with what runs it.
///
/// **One or more descriptors.** A descriptor describes one dispatch, and three of the routes a
/// plan binds are not one dispatch: paged attention decode is the split-KV kernel plus its
/// combine, and the GDN decode pair is `conv_update` then `recurrent_update`. Those are one
/// *route* with one autotune key and `n_launches = 2`, so the sweep times both launches inside
/// one event bracket — timing half of a route would be timing something the engine never runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// The plugin the first symbol lives in, e.g. `mxfp4_gemm_decode`.
    pub plugin: String,
    /// The kernel symbol that names this variant. For a multi-launch route it is the first one;
    /// the rest are in [`Variant::descs`].
    pub kernel: String,
    /// The knobs, parsed back out of the symbols so that what is recorded is what ran.
    pub params: BTreeMap<String, i64>,
    /// What to launch, in order.
    pub descs: Vec<LaunchDescriptor>,
}

impl Variant {
    /// A stable id for this configuration: FNV-1a/32 of the kernel symbol.
    ///
    /// Stable across runs and across machines, and reconstructible — the symbol name is a pure
    /// function of the parameters recorded beside it, so a reader can check the id.
    #[must_use]
    pub fn variant_id(&self) -> u32 {
        fnv1a32(&self.kernel)
    }

    /// Dispatches this variant issues, summed over its descriptors.
    #[must_use]
    pub fn n_launches(&self) -> u32 {
        self.descs.iter().map(|d| d.n_launches).sum()
    }

    /// The first descriptor. Every variant has at least one.
    #[must_use]
    pub fn desc(&self) -> &LaunchDescriptor {
        &self.descs[0]
    }

    /// `split_k=1 bk=128 tm=1` — the parameters, in canonical order.
    #[must_use]
    pub fn label(&self) -> String {
        self.params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// The variant id a kernel symbol carries, for a route bound without a sweep.
#[must_use]
pub fn variant_id_of(kernel: &str) -> u32 {
    fnv1a32(kernel)
}

/// FNV-1a, 32-bit. Deterministic and dependency-free.
fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.as_bytes() {
        h ^= u32::from(*b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// `qk_norm_rope`'s swept cell: the call the kernel sees, plus the session facts its `cos_sin`
/// and `pos` operands are a function of.
///
/// **Why the band is here rather than on [`QkShape`].** I19 left this open and named the fix as "a
/// context band on `QkShape`". `QkShape` is the KERNEL PLANE's shape — the thing
/// `glue::resolve` takes and `the internal reference harness`'s grammar spells — and neither the
/// kernel nor the capture line carries a band: `qk_norm_rope` takes no `max_pos` kernarg, and
/// `saber_model::graph::Bucket` (the other constructor of a `QkShape`) has no context band at all.
/// A field two of its three constructors would have to invent is worse than no field, and nothing
/// would ever read a wrong value from them. So the band lives on the CELL, which is the thing that
/// binds the buffers — and it reaches the cell through the autotune KEY (`BucketKey::MaxCtx` on
/// `layer.attn.qk_norm_rope`), which is where I19 said the divergence was: *"OPEN, and open in the
/// KEY rather than in the fill."*
///
/// The numbers this closes, at `b1s5c8192`: a rope table of **640 bytes** where the step
/// binds **1 MiB** (`max_pos * ROTARY_DIM * 2`, 5 x 64 x 2 against 8192 x 64 x 2), and
/// positions `0..5` where the step feeds `8187..8191`. I19's entry says 1.2 KiB against
/// 2 MiB; those are the sizes measured here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkCell {
    /// What the kernel is handed.
    pub shape: QkShape,
    /// Sequences the step's rows are divided between — the bucket's batch.
    pub num_seqs: u32,
    /// Rows each sequence feeds: the bucket's verify width at decode, the whole chunk at prefill.
    /// `num_seqs * q_len` is [`QkShape::m`].
    pub q_len: u32,
    /// The bucket's context band, from the key's `max_ctx`. Sizes the rope table and places the
    /// step's positions.
    pub max_ctx: u32,
    /// Which half of the step. A decode bucket's sequences sit at the TOP of the band (the same
    /// statement `TuneOp::AttnDecode`'s cell makes); a prefill chunk is the first thing in its
    /// sequence.
    pub phase: Phase,
}

impl QkCell {
    /// A cell with no history: one sequence of `m` rows at position 0, a band no wider than the
    /// chunk itself.
    ///
    /// What a CAPTURE CASE states, because `capture_glue.py` ran each kernel standalone over
    /// positions `0..m` — and it is a true statement about that call, not a default: the step's
    /// own law degenerates to exactly this when the band does not exceed the chunk.
    #[must_use]
    pub fn fresh_chunk(shape: QkShape) -> Self {
        QkCell {
            shape,
            num_seqs: 1,
            q_len: shape.m,
            max_ctx: shape.m,
            phase: Phase::Prefill,
        }
    }

    /// Pages the block table spans at this band — the step's `max_blocks`.
    #[must_use]
    pub fn max_blocks(&self) -> u32 {
        self.max_ctx.div_ceil(KV_BLOCK)
    }

    /// Where each sequence already is when the step runs.
    #[must_use]
    pub fn done_before(&self) -> u32 {
        match self.phase {
            Phase::Decode => self.max_ctx.saturating_sub(self.q_len),
            Phase::Prefill => 0,
        }
    }

    /// The rope table's extent, as `forward::index_table` sizes it:
    /// `(max_blocks * KV_BLOCK).max(rows)`.
    #[must_use]
    pub fn max_pos(&self) -> u32 {
        (self.max_blocks() * KV_BLOCK).max(self.shape.m).max(1)
    }
}

/// What a sweep can time: the MXFP4 W4A8 family, or one of the saber routes a plan binds.
///
/// The all-reduce is deliberately absent. It is a two-rank op — every one of its workspaces is
/// `IpcPeerVisible` and the kernel spins on a peer's flags — so a single-process sweeper cannot
/// launch it at all, and the launcher refuses the workspace kind rather than pretending. That
/// route is bound by `Autotuner::record_unmeasured`, never timed here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TuneOp {
    /// The MXFP4 GEMM family and its fused producers.
    Mxfp4 {
        /// Which op.
        op: Op,
        /// Its geometry.
        shape: Shape,
    },
    /// Paged attention, decode: the split-KV kernel plus its combine.
    AttnDecode(AttnShape),
    /// The GDN decode step as one launch (conv, grid barrier, recurrent).
    GdnFused(GdnShape),
    /// The GDN decode step as the `conv_update` + `recurrent_update` pair.
    GdnPair(GdnShape),
    /// Paged attention, PREFILL: one launch, query-tiled, no split-KV.
    AttnPrefill(AttnShape),
    /// The GDN prefill preamble: causal conv, qkv split, qk l2 norm, gating, per-chunk cumsum.
    GdnConvPrep(GdnPrefillShape),
    /// `A = (I + strict_lower(diag(beta) K K^T e^dg))^-1` per chunk.
    GdnKktSolve(GdnPrefillShape),
    /// The chunked scan: WY recompute, state recurrence and output in one kernel.
    GdnChunkScan(GdnPrefillShape),
    /// The prefill path's gated RMS norm, one row per (token, head).
    GdnNorm(NormShape),
    /// The attention layer's q/k norm + partial mrope, one launch.
    GlueQkNormRope(QkCell),
    /// `core * sigmoid(gate)` with the per-token e4m3 quant `o_proj` reads.
    GlueGateQuant(GateShape),
    /// The paged fp8 K/V write, both tensors in one launch.
    GlueKvAppend(KvShape),
}

impl TuneOp {
    /// The geometry a variant choice depends on, as a string, for matching a CAPTURE CASE to a
    /// BOUND ROUTE.
    ///
    /// Deliberately not the whole shape: a case and a route can describe the same kernel choice
    /// and still differ in the fields that are pointers-in-disguise -- the KV block and head
    /// strides, the scale bits, the split hint. Those go in the kernarg and must come from the
    /// case (whose buffers they describe); the choice of INSTANTIATION comes from the fields
    /// below, which is what the autotune record is about. Two ops with the same signature are the
    /// same tuning question asked of the same geometry.
    #[must_use]
    pub fn tuning_signature(&self) -> String {
        match self {
            TuneOp::Mxfp4 { shape, .. } => {
                format!("gemm/{}/{}/{}", shape.m, shape.n, shape.k)
            }
            TuneOp::AttnDecode(s) => format!(
                "attn_decode/{}/{}/{}/{}/{}/{}/{}/{:?}",
                s.num_seqs,
                s.q_len,
                s.q_heads,
                s.kv_heads,
                s.head_dim,
                s.block_size,
                s.max_ctx,
                s.kv_dtype
            ),
            TuneOp::AttnPrefill(s) => format!(
                "attn_prefill/{}/{}/{}/{}/{}/{}/{}/{:?}",
                s.num_seqs,
                s.q_len,
                s.q_heads,
                s.kv_heads,
                s.head_dim,
                s.block_size,
                s.max_ctx,
                s.kv_dtype
            ),
            TuneOp::GdnFused(s) => format!(
                "gdn_fused/{}/{}/{}/{}/{}/{}/{}",
                s.n_seqs, s.h, s.hg, s.maxq, s.head_k, s.head_v, s.conv_width
            ),
            TuneOp::GdnPair(s) => format!(
                "gdn_pair/{}/{}/{}/{}/{}/{}/{}",
                s.n_seqs, s.h, s.hg, s.maxq, s.head_k, s.head_v, s.conv_width
            ),
            TuneOp::GdnConvPrep(s) => format!(
                "gdn_prep/{}/{}/{}/{}/{}/{}/{}/{}",
                s.n_seqs, s.t, s.h, s.hg, s.head_k, s.head_v, s.conv_width, s.chunk
            ),
            TuneOp::GdnKktSolve(s) => format!(
                "gdn_kkt/{}/{}/{}/{}/{}/{}/{}/{}",
                s.n_seqs, s.t, s.h, s.hg, s.head_k, s.head_v, s.conv_width, s.chunk
            ),
            TuneOp::GdnChunkScan(s) => format!(
                "gdn_scan/{}/{}/{}/{}/{}/{}/{}/{}",
                s.n_seqs, s.t, s.h, s.hg, s.head_k, s.head_v, s.conv_width, s.chunk
            ),
            TuneOp::GdnNorm(s) => format!("gdn_norm/{}/{}", s.rows, s.channels),
            // The attn_glue trio (ws-h19). Same rule as the saber ops: the geometry, and not the
            // strides and pitches, which are facts about the CALLER's buffers -- `qkv_n` is the
            // merged projection's row pitch, `gs` the gate's, and the five `kv_append` strides
            // describe the source tensors and the paged view. Each of these ops has exactly one
            // legal instantiation, so the signature's whole job here is to say WHICH bound route a
            // case is, never which kernel.
            // The band is NOT in the signature, and that is deliberate: this string matches a
            // CAPTURE CASE to a bound route, and `capture_glue.py`'s grammar has no band to
            // state. It is in the KEY, which is what a record is a measurement of.
            TuneOp::GlueQkNormRope(c) => format!(
                "qk_norm_rope/{}/{}/{}/{}",
                c.shape.m, c.shape.nq, c.shape.nkv, c.shape.head_dim
            ),
            TuneOp::GlueGateQuant(s) => {
                format!("attn_gate_quant/{}/{}/{}", s.m, s.n, s.head_dim)
            }
            TuneOp::GlueKvAppend(s) => format!(
                "kv_append/{}/{}/{}/{}",
                s.tokens, s.nkv, s.head_dim, s.block_size
            ),
        }
    }

    /// The `(m, n, k)` triple the launcher keys a Persistent workspace on.
    ///
    /// `ArgBindings` carries an MXFP4 `Shape` because that is what the GEMM family's workspaces
    /// are keyed on. A saber route's geometry is not (m, n, k), so it is projected onto one that
    /// is unique per route and per shape — the only property the key needs.
    #[must_use]
    pub fn workspace_key(&self) -> Shape {
        match *self {
            TuneOp::Mxfp4 { shape, .. } => shape,
            TuneOp::AttnDecode(s) => Shape::new(s.num_seqs, s.q_len, s.max_blocks),
            TuneOp::GdnFused(s) => Shape::new(s.n_seqs, s.maxq, s.h),
            // Distinct from the fused route at the same geometry: different plugin, different
            // workspace set, and a shared key would make the launcher's refusal unreadable.
            TuneOp::GdnPair(s) => Shape::new(s.n_seqs, s.maxq, s.h + 1),
            TuneOp::AttnPrefill(s) => Shape::new(s.num_seqs, s.q_len, s.max_blocks),
            // The prefill trio share a geometry and are three different ops, so the third
            // component separates them: a Persistent workspace must never be shared by two.
            TuneOp::GlueQkNormRope(c) => Shape::new(c.shape.m, c.shape.nq, c.shape.head_dim),
            TuneOp::GlueGateQuant(s) => Shape::new(s.m, s.n, s.head_dim),
            TuneOp::GlueKvAppend(s) => Shape::new(s.tokens, s.nkv, s.head_dim),
            TuneOp::GdnConvPrep(s) => Shape::new(s.n_seqs, s.t, 1),
            TuneOp::GdnKktSolve(s) => Shape::new(s.n_seqs, s.t, 2),
            TuneOp::GdnChunkScan(s) => Shape::new(s.n_seqs, s.t, 3),
            TuneOp::GdnNorm(s) => {
                Shape::new(u32::try_from(s.rows).unwrap_or(u32::MAX), s.channels, 4)
            }
        }
    }

    /// A label for the sweep table.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            TuneOp::Mxfp4 { .. } => "mxfp4",
            TuneOp::AttnDecode(_) => "attn_decode",
            TuneOp::GdnFused(_) => "gdn_fused",
            TuneOp::GdnPair(_) => "gdn_pair",
            TuneOp::AttnPrefill(_) => "attn_prefill",
            TuneOp::GlueQkNormRope(_) => "qk_norm_rope",
            TuneOp::GlueGateQuant(_) => "attn_gate_quant",
            TuneOp::GlueKvAppend(_) => "kv_append",
            TuneOp::GdnConvPrep(_) => "gdn_conv_prep",
            TuneOp::GdnKktSolve(_) => "gdn_kkt_solve",
            TuneOp::GdnChunkScan(_) => "gdn_chunk_scan",
            TuneOp::GdnNorm(_) => "gdn_norm",
        }
    }
}

/// The result of enumerating one shape: what can run, and why everything else cannot.
#[derive(Debug, Clone, Default)]
pub struct VariantSet {
    /// The variants to measure, in a deterministic order.
    pub variants: Vec<Variant>,
    /// Settings the rules or the code object refused, each with its reason. Printed, not dropped:
    /// "there is no `bk64_ks2`" is a fact about the build, and a silent gap in a sweep is how a
    /// cache ends up covering less than it claims.
    pub refused: Vec<String>,
}

/// Knob axes, read out of the rules tables.
fn ladder_ks(rules: &LaunchRules) -> Vec<u32> {
    rules.split_k_ladder.clone()
}

fn table_bk(rules: &LaunchRules) -> Vec<u32> {
    let mut v: BTreeSet<u32> = rules.decode_bk.iter().map(|r| r.bk).collect();
    v.extend(rules.decode_band.iter().filter_map(|r| r.bk));
    v.insert(rules.k_align_override.bk);
    v.into_iter().collect()
}

fn table_tn(rules: &LaunchRules) -> Vec<u32> {
    rules
        .folded_tn
        .iter()
        .map(|r| r.tn)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn table_lbk(rules: &LaunchRules) -> Vec<u32> {
    rules
        .atiled_lbk
        .iter()
        .map(|r| r.lbk)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The A-tiled tile bodies: the reference's height 4, plus every (height, body) `[[atiled_wm]]`
/// compiles (G1's `wm = 8`, G3's `variant = "s"` / `"sp"` of it).
fn table_wm(rules: &LaunchRules) -> Vec<(u32, Option<String>)> {
    std::iter::once((4, None))
        .chain(rules.atiled_wm.iter().map(|r| (r.wm, r.variant.clone())))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Parse the knobs back out of a kernel symbol.
///
/// The symbol is what will actually run, so reading the parameters off it cannot drift from the
/// launch the way a second derivation from the rules could. `ks` is recorded as `split_k` because
/// that is the name the planner sizes the split-K workspace from.
fn params_from_symbol(kernel: &str) -> BTreeMap<String, i64> {
    const AXES: &[(&str, &str)] = &[
        ("lbk", "lbk"),
        ("bk", "bk"),
        ("ks", "split_k"),
        ("tm", "tm"),
        ("tn", "tn"),
        ("wperm", "wperm"),
        ("nt", "nt"),
        ("epifast", "epifast"),
        ("tiled", "tiled"),
        ("wm", "wm"),
        ("t", "block"),
    ];
    let mut out = BTreeMap::new();
    for token in kernel.split('_') {
        // G3's tile bodies: `wm8s` = SADDR addressing, `wm8sp` = + the W double buffer.
        if let Some(rest) = token.strip_prefix("wm") {
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            let body = &rest[digits..];
            if digits > 0 && (body == "s" || body == "sp") {
                if let Ok(v) = rest[..digits].parse::<i64>() {
                    out.insert("wm".to_string(), v);
                    out.insert("saddr".to_string(), 1);
                    out.insert("wpipe".to_string(), i64::from(body == "sp"));
                }
                continue;
            }
        }
        for (prefix, name) in AXES {
            let Some(rest) = token.strip_prefix(prefix) else {
                continue;
            };
            if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if let Ok(v) = rest.parse::<i64>() {
                out.insert((*name).to_string(), v);
            }
            break;
        }
    }
    out
}

/// Every variant of `op` at `shape` that the rules permit and the code objects contain.
///
/// Deterministic: the order is (implementation, then the knob axes in table order), which is the
/// order the interleaved sweep measures in and therefore the order the warm-up discard applies to.
#[must_use]
fn enumerate_mxfp4(op: Op, shape: Shape, rules: &LaunchRules, symbols: &SymbolIndex) -> VariantSet {
    let mut set = VariantSet::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    let mut offer = |setting: String, r: &LaunchRules, want_op: &str, set: &mut VariantSet| {
        match resolve(op, shape, r) {
            Ok(desc) => {
                if desc.op != want_op {
                    set.refused
                        .push(format!("{setting}: the rules answer `{}` here", desc.op));
                    return;
                }
                if !symbols.has(&desc.plugin, &desc.kernel) {
                    set.refused.push(format!(
                        "{setting}: `{}` is not a symbol in `{}`",
                        desc.kernel, desc.plugin
                    ));
                    return;
                }
                if !seen.insert(desc.kernel.clone()) {
                    return; // another setting already resolved to this symbol
                }
                set.variants.push(Variant {
                    plugin: desc.plugin.clone(),
                    kernel: desc.kernel.clone(),
                    params: params_from_symbol(&desc.kernel),
                    descs: vec![desc],
                });
            }
            Err(e) => set.refused.push(format!("{setting}: {e}")),
        }
    };

    match op {
        Op::Gemm { has_wref, .. } => {
            if !has_wref {
                // The unfolded kernel is one instantiation; there is no tile to choose.
                offer("unfolded".to_string(), rules, "gemm_base", &mut set);
                return set;
            }
            if a_tiled(shape.m, rules) {
                // Above the threshold the activations ARE fragment-tiled; only A-tiled can read
                // them, so the decode and folded rows are not alternatives at this M.
                for tn in table_tn(rules) {
                    for lbk in table_lbk(rules) {
                        let mut r = rules.clone();
                        r.folded_tn = vec![TnRow {
                            m_min: None,
                            n_min: None,
                            a_tiled: None,
                            tn,
                        }];
                        r.atiled_lbk = vec![LbkRow { tn, lbk }];
                        // The tile height: the reference's 4 (no row) and every height the table
                        // compiles (G1's 8), each offered at every M so the sweep can say where
                        // the tall tile stops paying.
                        for (wm, variant) in table_wm(rules) {
                            let mut r = r.clone();
                            r.atiled_wm = rules
                                .atiled_wm
                                .iter()
                                .filter(|row| {
                                    row.wm == wm && row.variant == variant && row.tn == tn
                                })
                                .map(|row| WmRow {
                                    m_min: None,
                                    m_max: None,
                                    ..row.clone()
                                })
                                .collect();
                            offer(
                                format!(
                                    "atiled tn={tn} lbk={lbk} wm={wm}{}",
                                    variant.as_deref().unwrap_or("")
                                ),
                                &r,
                                "gemm_atiled",
                                &mut set,
                            );
                        }
                    }
                }
                return set;
            }
            for ks in ladder_ks(rules) {
                for bk in table_bk(rules) {
                    let mut r = rules.clone();
                    r.knobs.a_tiled_min_m = 0;
                    r.decode_band = vec![BandRow {
                        ks: Some(ks),
                        bk: Some(bk),
                        ..BandRow::default()
                    }];
                    offer(
                        format!("decode split_k={ks} bk={bk}"),
                        &r,
                        "gemm_decode",
                        &mut set,
                    );
                }
            }
            for tn in table_tn(rules) {
                let mut r = rules.clone();
                // Take the decode band and the A-tiled layout out of the way so the folded tile
                // is reachable at this M; neither edit changes the tile being offered.
                r.knobs.decode_max_m = 0;
                r.knobs.a_tiled_min_m = 0;
                r.folded_tn = vec![TnRow {
                    m_min: None,
                    n_min: None,
                    a_tiled: None,
                    tn,
                }];
                offer(format!("folded tn={tn}"), &r, "gemm_folded", &mut set);
            }
        }
        Op::AddRmsQuant { .. } | Op::SiluMulQuant => {
            let want = if matches!(op, Op::SiluMulQuant) {
                "silu_mul_quant"
            } else {
                "add_rms_quant"
            };
            // The block width is a threshold knob with exactly two answers; both symbols ship.
            for (label, min_m, large_n) in
                [("block=small", u32::MAX, u32::MAX), ("block=large", 0, 0)]
            {
                let mut r = rules.clone();
                r.knobs.quant_large_min_m = min_m;
                r.knobs.silu_large_n = large_n;
                offer(label.to_string(), &r, want, &mut set);
            }
        }
        Op::GdnNormQuant { .. } => {
            offer("single".to_string(), rules, "gdn_norm_quant", &mut set);
        }
    }
    set
}

/// Every variant of a saber route that the rules permit and the code objects contain.
///
/// The knob ladders are the rules' own instantiation tables, exactly as for MXFP4:
///
/// | route | axis | where the values come from |
/// |---|---|---|
/// | attention decode | `nw` (warps per workgroup) | the `[[attn_decode]]` rows for this entry |
/// | GDN decode pair | `hpb` (heads per block) | the `[[gdn_conv_update]]` rows |
/// | GDN decode pair | `vs` (v-row split) | the `[[gdn_recurrent_update]]` rows |
/// | GDN fused | — | one instantiation per `ab_bf16`, and `ab_bf16` is the gate dtype the producer wrote, not a tile |
///
/// A setting is offered by overlaying it on a clone of the rules and asking `saber::resolve` what
/// comes out, so the descriptor is built by the function the serve path calls. Legality is the
/// rules' own: `nw` must cover `q_len * gqa` rows at the *real* `decode_rows_per_warp`, and `hpb`
/// must divide both head counts.
fn enumerate_saber(op: &TuneOp, rules: &SaberRules, symbols: &SymbolIndex) -> VariantSet {
    let mut set = VariantSet::default();
    match *op {
        TuneOp::Mxfp4 { .. } => {}
        // The attention glue offers ONE variant per shape and it is not a tile: `glue::resolve`
        // picks the qk band from the (token, head) pair count and the gate's A-side layout from
        // the same threshold `add_rms_quant` takes, both from the rules. There is nothing to
        // sweep and everything to MEASURE, which is the difference between a variant ladder and
        // a number.
        TuneOp::GlueQkNormRope(_) | TuneOp::GlueGateQuant(_) | TuneOp::GlueKvAppend(_) => {
            let glue = match GlueRules::prod() {
                Ok(g) => g,
                Err(e) => {
                    set.refused.push(format!("attn_glue rules: {e}"));
                    return set;
                }
            };
            let (gop, shape) = match *op {
                TuneOp::GlueQkNormRope(c) => (
                    // The CONFIG's `mrope_interleaved`, not a shape: for a text-only request the
                    // three position rows are equal and the choice cannot be observed.
                    glue::Op::QkNormRope {
                        interleaved: true,
                        // The TARGET's rope: the sweeper measures the routes the plan binds.
                        flavour: glue::QkFlavour::TARGET,
                    },
                    glue::Shape::Qk(c.shape),
                ),
                TuneOp::GlueGateQuant(s) => (glue::Op::AttnGateQuant, glue::Shape::Gate(s)),
                TuneOp::GlueKvAppend(s) => (glue::Op::KvAppend, glue::Shape::Kv(s)),
                _ => unreachable!("matched above"),
            };
            match glue::resolve(gop, shape, &glue) {
                Ok(desc) => {
                    if symbols.has(&desc.plugin, &desc.kernel) {
                        set.variants.push(Variant {
                            plugin: desc.plugin.clone(),
                            kernel: desc.kernel.clone(),
                            params: params_from_symbol(&desc.kernel),
                            descs: vec![desc],
                        });
                    } else {
                        set.refused.push(format!(
                            "`{}` is not a symbol in `{}`",
                            desc.kernel, desc.plugin
                        ));
                    }
                }
                Err(e) => set.refused.push(e.to_string()),
            }
        }
        TuneOp::AttnDecode(s) => {
            let kn = &rules.knobs;
            let gqa = s.q_heads / s.kv_heads.max(1);
            let rows = s.q_len * gqa;
            // A warp covers `decode_rows_per_warp` rows. Fewer warps than that needs would drop
            // rows, so the ladder starts at the smallest that covers them; more is legal and is
            // exactly the question worth measuring.
            let need = rows.div_ceil(kn.decode_rows_per_warp).max(1);
            // Only a head explicitly resolved from the wider object may tune above
            // the four-wave ladder. The original objects keep their existing limit.
            let max_nw = if need > kn.decode_max_nw {
                saber::resolve(saber::Op::AttnDecode, saber::Shape::Attn(s), rules)
                    .ok()
                    .filter(|head| head.plugin == "saber_attn16")
                    .map_or(kn.decode_max_nw, |head| head.block[0] / 32)
            } else {
                kn.decode_max_nw
            };
            let ladder: BTreeSet<u32> = rules.attn_decode.iter().map(|r| r.nw).collect();
            for nw in ladder {
                if nw < need {
                    set.refused.push(format!(
                        "nw={nw}: {rows} rows need at least {need} warps at {} rows per warp",
                        kn.decode_rows_per_warp
                    ));
                    continue;
                }
                if nw > max_nw {
                    set.refused.push(format!(
                        "nw={nw}: past the {} the entry point takes",
                        max_nw
                    ));
                    continue;
                }
                let mut r = rules.clone();
                // `nw = ceil(rows / decode_rows_per_warp)`: to offer a given nw, say how many
                // rows a warp is being asked to take. The kernel's own tile is unchanged.
                r.knobs.decode_rows_per_warp = rows.div_ceil(nw).max(1);
                match saber::resolve(saber::Op::AttnDecode, saber::Shape::Attn(s), &r) {
                    Ok(decode) => {
                        if !decode.kernel.ends_with(&format!("_nw{nw}")) {
                            set.refused.push(format!(
                                "nw={nw}: the rules answer `{}` at this geometry",
                                decode.kernel
                            ));
                            continue;
                        }
                        match attn_decode_combine(s, &r) {
                            Ok(combine) => {
                                if !symbols.has(&decode.plugin, &decode.kernel)
                                    || !symbols.has(&combine.plugin, &combine.kernel)
                                {
                                    set.refused.push(format!(
                                        "nw={nw}: `{}` or `{}` is not in the code object",
                                        decode.kernel, combine.kernel
                                    ));
                                    continue;
                                }
                                let mut params = params_from_symbol(&decode.kernel);
                                params.insert("nw".to_string(), i64::from(nw));
                                params.insert("splits".to_string(), i64::from(decode.grid[0]));
                                set.variants.push(Variant {
                                    plugin: decode.plugin.clone(),
                                    kernel: decode.kernel.clone(),
                                    params,
                                    descs: vec![decode, combine],
                                });
                            }
                            Err(e) => set.refused.push(format!("nw={nw}: combine: {e}")),
                        }
                    }
                    Err(e) => set.refused.push(format!("nw={nw}: {e}")),
                }
            }
        }
        TuneOp::GdnFused(s) => {
            // One symbol per `ab_bf16`, and `ab_bf16` names the dtype the producer wrote the
            // gates in -- a property of the edge, not a tile. So: one variant, no ladder.
            match saber::resolve(
                saber::Op::GdnFusedUpdate { ab_bf16: true },
                saber::Shape::Gdn(s),
                rules,
            ) {
                Ok(desc) => {
                    if symbols.has(&desc.plugin, &desc.kernel) {
                        let params = params_from_symbol(&desc.kernel);
                        set.variants.push(Variant {
                            plugin: desc.plugin.clone(),
                            kernel: desc.kernel.clone(),
                            params,
                            descs: vec![desc],
                        });
                    } else {
                        set.refused
                            .push(format!("`{}` is not in the code object", desc.kernel));
                    }
                }
                Err(e) => set.refused.push(format!("fused: {e}")),
            }
        }
        TuneOp::GdnPair(s) => {
            let hpbs: BTreeSet<u32> = rules.gdn_conv_update.iter().map(|r| r.hpb).collect();
            let vss: BTreeSet<u32> = rules
                .gdn_recurrent_update
                .iter()
                .filter(|r| r.ab_bf16 && !r.norm)
                .map(|r| r.vs)
                .collect();
            for hpb in &hpbs {
                if s.hg % hpb != 0 || s.h % hpb != 0 {
                    set.refused.push(format!(
                        "hpb={hpb}: does not divide both head counts ({} k, {} v)",
                        s.hg, s.h
                    ));
                    continue;
                }
                for vs in &vss {
                    let mut r = rules.clone();
                    // Force the conv grouping: start the search at this hpb and take the floor
                    // out of the way so the grid-fill halving cannot walk past it.
                    r.gdn.hpb_start = *hpb;
                    r.gdn.hpb_min_workgroups = 0;
                    // Force the v-row split: ask for the widest, then cap it at the target.
                    r.gdn.vs_fill_workgroups = 4 * (s.n_seqs * s.h).max(1);
                    r.gdn.vs_max = *vs;
                    let setting = format!("hpb={hpb} vs={vs}");
                    let conv = match saber::resolve(
                        saber::Op::GdnDecodePair { ab_bf16: true },
                        saber::Shape::Gdn(s),
                        &r,
                    ) {
                        Ok(d) => d,
                        Err(e) => {
                            set.refused.push(format!("{setting}: conv: {e}"));
                            continue;
                        }
                    };
                    let rec = match gdn_decode_pair_recurrent(s, true, None, &r) {
                        Ok(d) => d,
                        Err(e) => {
                            set.refused.push(format!("{setting}: recurrent: {e}"));
                            continue;
                        }
                    };
                    if !symbols.has(&conv.plugin, &conv.kernel)
                        || !symbols.has(&rec.plugin, &rec.kernel)
                    {
                        set.refused.push(format!(
                            "{setting}: `{}` or `{}` is not in the code object",
                            conv.kernel, rec.kernel
                        ));
                        continue;
                    }
                    let mut params = BTreeMap::new();
                    params.insert("hpb".to_string(), i64::from(*hpb));
                    params.insert("vs".to_string(), i64::from(*vs));
                    set.variants.push(Variant {
                        plugin: conv.plugin.clone(),
                        kernel: format!("{} + {}", conv.kernel, rec.kernel),
                        params,
                        descs: vec![conv, rec],
                    });
                }
            }
        }
        TuneOp::AttnPrefill(a) => {
            // The `fp8_mode` ladder, from the `[[attn_prefill]]` rows for this entry. It is an
            // OPT mode -- which of the QK and PV legs run in 8 bits -- so it is a **numerics**
            // axis, not a tile: mode 0 computes something slightly different from mode 3. They
            // are timed together because the question "what does the 8-bit leg buy" is worth an
            // answer, and the caller says which one the tier actually binds.
            let modes: BTreeSet<u32> = rules.attn_prefill.iter().map(|r| r.fp8_mode).collect();
            for mode in modes {
                let setting = format!("fp8_mode={mode}");
                match saber::resolve(
                    saber::Op::AttnPrefill { fp8_mode: mode },
                    saber::Shape::Attn(a),
                    rules,
                ) {
                    Ok(desc) => {
                        if !symbols.has(&desc.plugin, &desc.kernel) {
                            set.refused.push(format!(
                                "{setting}: `{}` is not in the code object",
                                desc.kernel
                            ));
                            continue;
                        }
                        let mut params = params_from_symbol(&desc.kernel);
                        params.insert("fp8_mode".to_string(), i64::from(mode));
                        set.variants.push(Variant {
                            plugin: desc.plugin.clone(),
                            kernel: desc.kernel.clone(),
                            params,
                            descs: vec![desc],
                        });
                    }
                    Err(e) => set.refused.push(format!("{setting}: {e}")),
                }
            }
        }
        TuneOp::GdnConvPrep(g) => {
            // `hpb` is the conv kernel's channel grouping, and the prefill entry does NOT halve
            // it for grid fill (that is the decode entry's rule), so every row the code object
            // has and the head counts divide is legal here.
            let hpbs: BTreeSet<u32> = rules.gdn_conv_prep.iter().map(|r| r.hpb).collect();
            for hpb in hpbs {
                if g.hg % hpb != 0 || g.h % hpb != 0 {
                    set.refused.push(format!(
                        "hpb={hpb}: does not divide both head counts ({} k, {} v)",
                        g.hg, g.h
                    ));
                    continue;
                }
                let mut r = rules.clone();
                r.gdn.hpb_start = hpb;
                r.gdn.hpb_min_workgroups = 0;
                offer_single(
                    &mut set,
                    symbols,
                    format!("hpb={hpb}"),
                    saber::Op::GdnConvPrep { ab_bf16: true },
                    saber::Shape::GdnPrefill(g),
                    &r,
                    Some(("hpb", i64::from(hpb))),
                );
            }
        }
        TuneOp::GdnKktSolve(g) => {
            // `ratio` is H/Hg, which the geometry fixes at 3. There is no ladder: the other five
            // instantiations exist for head counts this model does not have.
            offer_single(
                &mut set,
                symbols,
                "ratio from H/Hg".to_string(),
                saber::Op::GdnKktSolve,
                saber::Shape::GdnPrefill(g),
                rules,
                None,
            );
        }
        TuneOp::GdnChunkScan(g) => {
            // One instantiation, no template parameter at all -- the rules file says so in as
            // many words. Nothing to choose, and the record says what it measured.
            offer_single(
                &mut set,
                symbols,
                "single instantiation".to_string(),
                saber::Op::GdnChunkScan,
                saber::Shape::GdnPrefill(g),
                rules,
                None,
            );
        }
        TuneOp::GdnNorm(n) => {
            // `act` is NOT a variant: 0 is silu/swish and 1 is sigmoid, and which one the layer
            // wants is `output_gate_type` in the checkpoint's config, not something a stopwatch
            // may decide. This model is swish. (The key carries no `act`, which is why the choice
            // cannot be made here -- see the ledger.)
            offer_single(
                &mut set,
                symbols,
                "act=0 (swish, this checkpoint's output_gate_type)".to_string(),
                saber::Op::GdnGatedRmsNorm { act: 0 },
                saber::Shape::Norm(n),
                rules,
                Some(("act", 0)),
            );
        }
    }
    set
}

/// Offer one single-launch saber setting: resolve it, check the symbol, keep it.
fn offer_single(
    set: &mut VariantSet,
    symbols: &SymbolIndex,
    setting: String,
    op: saber::Op,
    shape: saber::Shape,
    rules: &SaberRules,
    extra: Option<(&str, i64)>,
) {
    match saber::resolve(op, shape, rules) {
        Ok(desc) => {
            if !symbols.has(&desc.plugin, &desc.kernel) {
                set.refused.push(format!(
                    "{setting}: `{}` is not a symbol in `{}`",
                    desc.kernel, desc.plugin
                ));
                return;
            }
            if set.variants.iter().any(|v| v.kernel == desc.kernel) {
                return; // another setting already resolved to this symbol
            }
            let mut params = params_from_symbol(&desc.kernel);
            if let Some((k, v)) = extra {
                params.insert(k.to_string(), v);
            }
            set.variants.push(Variant {
                plugin: desc.plugin.clone(),
                kernel: desc.kernel.clone(),
                params,
                descs: vec![desc],
            });
        }
        Err(e) => set.refused.push(format!("{setting}: {e}")),
    }
}

/// Every variant of `op` at its shape that the rules permit and the code objects contain.
///
/// Deterministic: the order is (implementation, then the knob axes in table order), which is the
/// order the interleaved sweep measures in and therefore the order the warm-up discard applies to.
#[must_use]
pub fn enumerate_variants(
    op: &TuneOp,
    rules: &LaunchRules,
    saber_rules: &SaberRules,
    symbols: &SymbolIndex,
) -> VariantSet {
    match *op {
        TuneOp::Mxfp4 { op, shape } => enumerate_mxfp4(op, shape, rules, symbols),
        _ => enumerate_saber(op, saber_rules, symbols),
    }
}

// ------------------------------------------------------------------------------- inputs

/// How a buffer is filled. Random, but inside the value range the kernel is fed in production:
/// an activation byte that decodes to e4m3 NaN, or a block scale of 2^127, is not a tile choice —
/// it is a different numerical workload, and it would be measured as one.
// No `Eq`: `ConstF32` carries a float, and `Fill` is only ever compared for equality with
// `Fill::Zero` (the "skip the RNG" fast path), which `PartialEq` answers.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fill {
    /// Packed e2m1 nibble pairs. Every byte is a legal pair, so: uniform.
    Nibble,
    /// e4m3 activations, exponents held around 1.0.
    E4m3,
    /// e8m0 block scales, 118..=134 with ~1% zero blocks, as the capture generates them.
    E8m0,
    /// bf16 values around 1.0.
    Bf16,
    /// fp32 per-row scales in [0.001, 0.021).
    Scale,
    /// An output. Zeroed.
    Zero,
    /// fp32 recurrent state: small values, because a delta-net state that starts at N(0, 1) is
    /// not the workload the kernel sees after a few hundred steps.
    F32State,
    /// `1, 2, 3, …` as `i32`: the per-token state slot indices. Slot 0 is never handed out — the
    /// kernel skips a sequence whose slot is <= 0, which is how prod's allocator behaves.
    SlotIndex,
    /// `0, step, 2*step, …` as `i32`: the cumulative token offsets (`cu`).
    CuSeq {
        /// Tokens per sequence.
        step: u32,
    },
    /// `0, 1, 2, …` as `i32`, repeated per sequence: one shared block pool, which is what the
    /// capture does (a per-sequence pool at 32k context would be 268 MiB of a ~1 GiB budget).
    BlockTable {
        /// Blocks per sequence.
        blocks: u32,
    },
    /// The same `i32` in every slot.
    ConstI32(i32),
    /// The rotary cos/sin cache as bf16 `[max_pos, rotary_dim]` rows of `[cos | sin]`, exactly as
    /// `saber_exec::forward::rope_cache` builds it — the same table a forward pass binds, so the
    /// sweep times the kernel over the values it will read in service.
    RopeCache {
        /// Positions the cache addresses.
        max_pos: u32,
        /// Rotated dims per head.
        rotary_dim: u32,
    },
    /// mrope positions, `[3, rows]` **int64**, all three sections equal: a text-only request,
    /// which is what this tier serves. Row `t` of a sequence sits at `done_before + t`, which is
    /// `forward::index_table`'s law — ascending from 0 only for a chunk at the start of its
    /// sequence.
    MropePos {
        /// Sequences the rows are divided between.
        seqs: u32,
        /// Rows each sequence feeds.
        q_len: u32,
        /// Where each sequence already is.
        done_before: u32,
    },
    /// The paged slot map, `[rows]` int64, ascending from 0. No negative entry: a padding row is
    /// an early-out and timing the early-out would measure less work than the step does.
    SlotMap {
        /// Rows in the step.
        rows: u32,
    },
    /// The same `f32` in every slot: the K/V quant scales, which the checkpoint calibrates to 1.0.
    ConstF32(f32),
}

/// One device buffer a case needs.
#[derive(Clone, Debug)]
struct BufSpec {
    /// The name the descriptor binds it under.
    name: &'static str,
    /// Pool key. Weights are keyed on `(n, k)` only, so consecutive M at one shape reuse them.
    key: String,
    bytes: u64,
    fill: Fill,
}

fn spec(name: &'static str, key: String, bytes: u64, fill: Fill) -> BufSpec {
    BufSpec {
        name,
        key,
        bytes,
        fill,
    }
}

/// Rows of activation storage: whole 16-row tiles once the layout is fragment-tiled.
fn a_rows(m: u32, tiled: bool) -> u64 {
    if tiled {
        u64::from(m.div_ceil(16)) * 16
    } else {
        u64::from(m)
    }
}

/// Every buffer the descriptor for this op names, with its size law and its filling.
///
/// The sizes are the reference capture's (`the internal reference harness`), which is the same
/// generator the bit-exact gate replays — so a sweep runs the kernels over the byte layout they
/// were proved correct on.
fn buffers_for(op: &TuneOp, tiled: bool) -> Vec<BufSpec> {
    match *op {
        TuneOp::Mxfp4 { op, shape } => mxfp4_buffers(op, shape, tiled),
        TuneOp::AttnDecode(s) => attn_buffers(&s, s.max_ctx),
        // A prefill chunk is the FIRST thing in its sequence: `forward::index_table` gives it
        // `done_before = 0`, so `seqused_k` is the chunk's own length, not the context band.
        // Filling it with `max_ctx` measured 2048 queries against 8192 keys -- a cell the plan
        // never runs and 3.2x the cost of the one it does (I18, against F's census).
        TuneOp::AttnPrefill(s) => attn_buffers(&s, s.q_len),
        TuneOp::GdnFused(s) | TuneOp::GdnPair(s) => gdn_decode_buffers(&s),
        TuneOp::GdnConvPrep(g) => gdn_prefill_buffers(&g, Stage::ConvPrep),
        TuneOp::GdnKktSolve(g) => gdn_prefill_buffers(&g, Stage::KktSolve),
        TuneOp::GdnChunkScan(g) => gdn_prefill_buffers(&g, Stage::ChunkScan),
        TuneOp::GdnNorm(n) => gdn_norm_buffers(&n),
        TuneOp::GlueQkNormRope(c) => glue_qk_buffers(&c),
        TuneOp::GlueGateQuant(s) => glue_gate_buffers(&s),
        TuneOp::GlueKvAppend(s) => glue_kv_buffers(&s),
    }
}

// ----------------------------------- the sweep's inputs against the step's own index laws (I19)

/// How a swept case's buffer relates to the law [`crate::forward::index_table`] states for a
/// buffer of that name.
///
/// # Why this exists
///
/// The `attn_prefill_paged` records in every cache up to and including I17 were measurements of a
/// cell the engine does not run: the sweep filled `seqused_k` with `max_ctx` for every attention
/// shape, and a prefill chunk gets `done_before = 0` from `forward::index_table`, so its
/// `seqused_k` is its own length. 2048 queries were timed against 8192 keys — 3.2x the cost of the
/// cell the plan runs — and **nothing in the key or the provenance could have said so**: the
/// record is keyed on the SHAPE, the shape was right, and it was the INPUT that was wrong (I18,
/// against F's census). No digest, no fingerprint and no guard in this repo covered that class.
///
/// This is the guard for it. Every buffer a swept case declares whose name has a law in
/// `forward::index_table` is either **derived by that law** from the session facts the cell
/// states, or **declared** to differ with the reason written down. A buffer that is neither is a
/// refusal at sweep time naming the op, which is the one outcome the `seqused_k` mistake could
/// not produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LawOf {
    /// The sweep's bytes ARE the step's law over this cell's facts, byte for byte.
    Derived,
    /// One allocation bound as a strided view of another, which is what a descriptor's
    /// `ci_stride` means: the law's `n` entries are this buffer's `i32`s at `0, s, 2s, …`.
    Strided(u32),
    /// A legal value of the same law that the cell states differently, or a step fact the cell
    /// does not carry at all. Declared with the reason, printed by the sweep and reviewed — the
    /// one thing it may never be is silent.
    Declared(&'static str),
}

/// The buffer names [`crate::forward::index_table`] has a law for. A case's buffer whose name is
/// not one of these is data (weights, activations, outputs) and is nothing to do with the step's
/// session bookkeeping.
const INDEX_LAW_BUFFERS: &[&str] = &[
    "ids",
    "cu",
    "cache_idx",
    "sidx",
    "naccept",
    "has_init",
    "block_table",
    "seqused_k",
    "cos_sin",
    "pos",
    "slot",
    "k_scale",
    "v_scale",
];

/// The session facts a swept cell states, in the step's own terms — the input to
/// [`crate::forward::index_table`], built from the shape and nothing else.
struct CellFacts {
    phase: Phase,
    rows: u32,
    max_blocks: u32,
    /// `(state_slot, done_before, live_rows)` per slot of the cell's batch.
    seqs: Vec<(u32, u32, u32)>,
    /// The page pool each sequence's block table indexes. The sweep gives every sequence the
    /// same pool, as the reference capture script (`the internal reference harness`) does — a per-sequence pool at a 32k context
    /// would be 268 MiB of a ~1 GiB budget — and that is a statement about the CELL, so it is
    /// what the facts say and what the law is then asked over.
    pages: Vec<u32>,
}

/// A buffer whose values the sweep states differently from the step, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexLawNote {
    /// The buffer name in the case's descriptor.
    pub buffer: &'static str,
    /// The reason, as declared at the case's law map.
    pub why: &'static str,
}

/// The facts a cell states and what each of its index-law buffers is, per op.
///
/// `None` facts means the cell states none: every index-law buffer it declares must then be
/// `Declared`, because there is nothing to derive a law from.
fn index_law_plan(op: &TuneOp) -> (Option<CellFacts>, BTreeMap<&'static str, LawOf>) {
    let mut laws: BTreeMap<&'static str, LawOf> = BTreeMap::new();
    match *op {
        // Weights, activations and outputs only.
        TuneOp::Mxfp4 { .. } | TuneOp::GdnNorm(_) | TuneOp::GlueGateQuant(_) => (None, laws),

        // A decode step: every sequence sits at the bucket's context band and feeds `q_len` rows
        // of it, so `done_before + live_rows` IS `max_ctx` — which is what the sweep fills
        // `seqused_k` with, and now derives rather than asserts.
        TuneOp::AttnDecode(s) => {
            laws.insert("seqused_k", LawOf::Derived);
            laws.insert("block_table", LawOf::Derived);
            let q = s.q_len;
            (
                Some(CellFacts {
                    phase: Phase::Decode,
                    rows: s.num_seqs * q,
                    max_blocks: s.max_blocks,
                    seqs: (0..s.num_seqs)
                        .map(|i| (FIRST_STATE_SLOT + i * q, s.max_ctx.saturating_sub(q), q))
                        .collect(),
                    pages: (0..s.max_blocks).collect(),
                }),
                laws,
            )
        }

        // A prefill chunk is the FIRST thing in its sequence (`done_before = 0`), so `seqused_k`
        // is the chunk's own length. THIS is the cell I18 got wrong for four caches.
        TuneOp::AttnPrefill(s) => {
            laws.insert("seqused_k", LawOf::Derived);
            laws.insert("block_table", LawOf::Derived);
            let q = s.q_len;
            (
                Some(CellFacts {
                    phase: Phase::Prefill,
                    rows: s.num_seqs * q,
                    max_blocks: s.max_blocks,
                    seqs: (0..s.num_seqs)
                        .map(|i| (FIRST_STATE_SLOT + i * q, 0, q))
                        .collect(),
                    pages: (0..s.max_blocks).collect(),
                }),
                laws,
            )
        }

        TuneOp::GdnFused(s) | TuneOp::GdnPair(s) => {
            laws.insert("sidx", LawOf::Derived);
            laws.insert("cu", LawOf::Derived);
            // `cache_idx` is the SAME allocation as `sidx`, read at stride `ci_stride = maxq`;
            // the strided read is the step's per-sequence `cache_idx` table, which is what this
            // checks rather than taking the descriptor's word for it.
            laws.insert("cache_idx", LawOf::Strided(s.maxq));
            laws.insert(
                "naccept",
                LawOf::Declared(
                    "the sweep times a steady state in which the previous step accepted ONE \
                     token; the step's table carries that step's real acceptance. Both are legal \
                     values of one law — the kernel reads `sidx[naccept - 1]`, one i32 either \
                     way — and the cell carries no acceptance history to derive it from.",
                ),
            );
            let q = s.maxq;
            (
                Some(CellFacts {
                    phase: Phase::Decode,
                    rows: s.n_seqs * q,
                    max_blocks: 0,
                    seqs: (0..s.n_seqs)
                        .map(|i| (FIRST_STATE_SLOT + i * q, 0, q))
                        .collect(),
                    pages: Vec::new(),
                }),
                laws,
            )
        }

        TuneOp::GdnConvPrep(s) | TuneOp::GdnKktSolve(s) | TuneOp::GdnChunkScan(s) => {
            laws.insert("cu", LawOf::Derived);
            laws.insert("cache_idx", LawOf::Derived);
            laws.insert("has_init", LawOf::Derived);
            // `t` is the batch's TOTAL tokens; each sequence feeds its share. At one sequence the
            // two coincide, which is every prefill bucket today — and the day a batched prefill
            // bucket exists (N7), a `cu` of `0, t, 2t` is refused here rather than measured.
            let per_seq = s.t / s.n_seqs.max(1);
            (
                Some(CellFacts {
                    phase: Phase::Prefill,
                    rows: s.t,
                    max_blocks: 0,
                    seqs: (0..s.n_seqs)
                        .map(|i| (FIRST_STATE_SLOT + i, 0, per_seq))
                        .collect(),
                    pages: Vec::new(),
                }),
                laws,
            )
        }

        // The paged K/V write: one sequence starting at position 0, one row per slot.
        TuneOp::GlueKvAppend(s) => {
            laws.insert("slot", LawOf::Derived);
            laws.insert("k_scale", LawOf::Derived);
            laws.insert("v_scale", LawOf::Derived);
            let blocks = s.tokens.div_ceil(s.block_size.max(1));
            (
                Some(CellFacts {
                    phase: Phase::Prefill,
                    rows: s.tokens,
                    max_blocks: blocks,
                    seqs: vec![(FIRST_STATE_SLOT, 0, s.tokens)],
                    pages: (0..blocks).collect(),
                }),
                laws,
            )
        }

        // CLOSED by I20, and it was open in the KEY rather than in the fill. `QkShape` carries
        // `m`, the head counts and `head_dim` and no context band, so the cell could not state
        // `done_before` or the extent the step sizes `cos_sin` by: it measured a 640-byte rope
        // table against the step's 1 MiB one at `b1s5c8192` and filled `pos` with `0..5` where
        // the step feeds `8187..8191`. `layer.attn.qk_norm_rope` carries `BucketKey::MaxCtx` now,
        // so the band is in the key the record is a measurement of, and [`QkCell`] carries it to
        // the fill. Both buffers are DERIVED from the step's own law.
        TuneOp::GlueQkNormRope(c) => {
            laws.insert("cos_sin", LawOf::Derived);
            laws.insert("pos", LawOf::Derived);
            let q = c.q_len;
            let done = c.done_before();
            let blocks = c.max_blocks();
            (
                Some(CellFacts {
                    phase: c.phase,
                    rows: c.shape.m,
                    max_blocks: blocks,
                    seqs: (0..c.num_seqs)
                        .map(|i| (FIRST_STATE_SLOT + i * q, done, q))
                        .collect(),
                    pages: (0..blocks).collect(),
                }),
                laws,
            )
        }
    }
}

/// The sweep's index-law buffers against the step's laws, for one case.
///
/// Returns the buffers that are declared to differ, so the sweep can print them; `Err` is a
/// refusal naming the op, the buffer and the first element that disagrees.
fn index_law_report(
    op: &TuneOp,
    specs: &[BufSpec],
) -> std::result::Result<Vec<IndexLawNote>, String> {
    let (facts, laws) = index_law_plan(op);
    let sig = op.tuning_signature();
    let mut notes = Vec::new();

    // `pages` outlives the `SeqFacts` that borrow it, which is the whole reason these three are
    // separate bindings rather than one struct.
    let pages: Vec<u32> = facts.as_ref().map(|f| f.pages.clone()).unwrap_or_default();
    let ids: Vec<i32> = facts
        .as_ref()
        .map(|f| vec![0i32; f.rows as usize])
        .unwrap_or_default();
    let seq_facts: Vec<SeqFacts<'_>> = facts.as_ref().map_or_else(Vec::new, |f| {
        f.seqs
            .iter()
            .map(|&(slot, done, live)| SeqFacts {
                state_slot: slot,
                done_before: done,
                kv_blocks: &pages,
                live_rows: live,
                rope: None,
            })
            .collect()
    });
    let step = facts.as_ref().map(|f| StepIndex {
        phase: f.phase,
        rows: f.rows,
        max_blocks: f.max_blocks,
        seqs: &seq_facts,
        ids: &ids,
        band_width: None,
    });

    for spec in specs {
        if !INDEX_LAW_BUFFERS.contains(&spec.name) {
            continue;
        }
        let Some(law) = laws.get(spec.name) else {
            return Err(format!(
                "{sig}: the sweep builds `{}`, which `forward::index_table` has a law for, and \
                 the case says nothing about how the two relate. Derive it from the step's law \
                 or declare why the cell cannot — a fill that silently disagrees with the step \
                 is how `attn_prefill_paged` was timed at 4x its context for four caches (I18).",
                spec.name
            ));
        };
        if let LawOf::Declared(why) = law {
            notes.push(IndexLawNote {
                buffer: spec.name,
                why,
            });
            continue;
        }
        let Some(step) = &step else {
            return Err(format!(
                "{sig}: `{}` is to be derived from the step's law, but the case states no \
                 session facts to derive it from",
                spec.name
            ));
        };
        let want = crate::forward::index_table(&format!("tune.{}", spec.name), step)
            .map_err(|e| format!("{sig}: `{}`: {e}", spec.name))?;
        let got = fill_bytes(spec.fill, spec.bytes, 1);
        let (want, got) = match *law {
            LawOf::Strided(stride) => (want.clone(), stride_i32(&got, stride, want.len() / 4)),
            _ => (want, got),
        };
        if want != got {
            let at = want
                .iter()
                .zip(&got)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| want.len().min(got.len()));
            return Err(format!(
                "{sig}: the sweep's `{}` is not what the step feeds. The step's law gives {} \
                 byte(s), the sweep {}; they first differ at byte {at} (step {:?}, sweep {:?}). \
                 A record measured over it is a measurement of a cell the engine does not run — \
                 which is what `seqused_k = max_ctx` at prefill was (I18).",
                spec.name,
                want.len(),
                got.len(),
                want.get(at),
                got.get(at),
            ));
        }
    }
    Ok(notes)
}

/// Every `i32` at `0, stride, 2*stride, …`, `n` of them, back as bytes.
fn stride_i32(bytes: &[u8], stride: u32, n: usize) -> Vec<u8> {
    let stride = stride.max(1) as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let at = i * stride * 4;
        match bytes.get(at..at + 4) {
            Some(w) => out.extend_from_slice(w),
            None => return out,
        }
    }
    out
}

/// **The sweep's inputs must be the step's.** Checks one case's index-law buffers against
/// [`crate::forward::index_table`] and returns the declared divergences.
///
/// See [`LawOf`] for what this is and the mistake it exists to catch. `Sweeper::sweep` runs it on
/// every case before it launches anything, so a cell whose inputs disagree with the step is
/// refused rather than measured.
///
/// # Errors
/// A buffer that is neither derived from the step's law nor declared to differ from it, naming
/// the op and the buffer.
pub fn check_index_laws(op: &TuneOp) -> Result<Vec<IndexLawNote>> {
    index_law_report(op, &buffers_for(op, false)).map_err(Error::Refused)
}

/// `qk_norm_rope`'s buffers, at `the internal reference harness`'s geometry: one fused projection
/// of `m` rows, two per-head norm weights, the rope cache and this step's mrope positions.
///
/// The last two are the step's, not the cell's own invention: `cos_sin` is sized by the CONTEXT
/// BAND (`(max_blocks * KV_BLOCK).max(rows)`) and `pos` carries `done_before + t`, which is what
/// [`crate::forward::index_table`] gives and what [`check_index_laws`] now derives rather than
/// accepting a declaration for (I19 -> I20).
fn glue_qk_buffers(c: &QkCell) -> Vec<BufSpec> {
    let s = &c.shape;
    let m = u64::from(s.m);
    let hd = u64::from(s.head_dim);
    let key = format!("{}:{}:{}", s.m, s.nq, s.nkv);
    let max_pos = c.max_pos();
    let done = c.done_before();
    vec![
        spec(
            "qkv",
            format!("qkv:{key}"),
            m * u64::from(s.qkv_n) * 2,
            Fill::Bf16,
        ),
        spec("qw", format!("qw:{}", s.head_dim), hd * 2, Fill::Bf16),
        spec("kw", format!("kw:{}", s.head_dim), hd * 2, Fill::Bf16),
        spec(
            "cos_sin",
            format!("cos_sin:{max_pos}"),
            u64::from(max_pos) * u64::from(ROTARY_DIM) * 2,
            Fill::RopeCache {
                max_pos,
                rotary_dim: ROTARY_DIM,
            },
        ),
        spec(
            "pos",
            format!("pos:{}:{}:{done}", c.num_seqs, c.q_len),
            m * 3 * 8,
            Fill::MropePos {
                seqs: c.num_seqs,
                q_len: c.q_len,
                done_before: done,
            },
        ),
        spec(
            "q_out",
            format!("qout:{key}"),
            m * u64::from(s.nq) * hd * 2,
            Fill::Zero,
        ),
        spec(
            "k_out",
            format!("kout:{key}"),
            m * u64::from(s.nkv) * hd * 2,
            Fill::Zero,
        ),
    ]
}

/// `attn_gate_quant`'s buffers. `gate` is the PROJECTION the kernel reads the gate out of in
/// place, so it is `gs` wide and not `n` — passing a contiguous copy would time a kernel this
/// instantiation does not serve.
fn glue_gate_buffers(s: &GateShape) -> Vec<BufSpec> {
    let m = u64::from(s.m);
    let n = u64::from(s.n);
    let key = format!("{}:{}", s.m, s.n);
    vec![
        spec("core", format!("core:{key}"), m * n * 2, Fill::Bf16),
        spec(
            "gate",
            format!("gate:{}:{}", s.m, s.gs),
            m * u64::try_from(s.gs).unwrap_or(0) * 2,
            Fill::Bf16,
        ),
        spec("q", format!("gq:{key}"), m * n, Fill::Zero),
        spec("scale", format!("gsc:{}", s.m), m * 4, Fill::Zero),
    ]
}

/// `kv_append`'s buffers: the roped K, the V read in place out of the projection, and a SEEDED
/// paged cache — a zeroed one would let a kernel that wrote whole pages look correct, which is
/// the same reason the bit-exact capture seeds it.
fn glue_kv_buffers(s: &KvShape) -> Vec<BufSpec> {
    let t = u64::from(s.tokens);
    let nkv = u64::from(s.nkv);
    let hd = u64::from(s.head_dim);
    let bs = u64::from(s.block_size);
    // Enough pages to hold every row this step writes, one row per slot.
    let blocks = t.div_ceil(bs);
    let key = format!("{}:{}", s.tokens, s.nkv);
    vec![
        spec("key", format!("kvk:{key}"), t * nkv * hd * 2, Fill::Bf16),
        spec(
            "value",
            format!("kvv:{}:{}", s.tokens, s.value_stride),
            t * u64::try_from(s.value_stride).unwrap_or(0) * 2,
            Fill::Bf16,
        ),
        spec(
            "kv",
            format!("kvpage:{blocks}:{}", s.nkv),
            blocks.max(1) * nkv * bs * 2 * hd,
            Fill::Nibble,
        ),
        spec(
            "slot",
            format!("kvslot:{}", s.tokens),
            t * 8,
            Fill::SlotMap { rows: s.tokens },
        ),
        spec("k_scale", "kvscale".into(), 4, Fill::ConstF32(1.0)),
        spec("v_scale", "kvscale".into(), 4, Fill::ConstF32(1.0)),
    ]
}

/// Which of the prefill trio is being built for. They share a geometry and read overlapping
/// buffers -- the scan consumes what the prep and the solve wrote -- so the pool keys are shared
/// on purpose: three consecutive work items at one T upload the big ones once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    ConvPrep,
    KktSolve,
    ChunkScan,
}

/// The GDN PREFILL trio's buffers, at the reference capture script (`the internal reference harness`)'s geometry: one sequence of
/// `t` tokens, a conv state of `conv_width - 1` (a prefill has no speculative window to carry),
/// and `cu = [0, t]`.
fn gdn_prefill_buffers(s: &GdnPrefillShape, stage: Stage) -> Vec<BufSpec> {
    let (h, hg) = (u64::from(s.h), u64::from(s.hg));
    let (hk, hv) = (u64::from(s.head_k), u64::from(s.head_v));
    let t = u64::from(s.t);
    let chunk = u64::from(s.chunk);
    let n_seqs = u64::from(s.n_seqs);
    let units = 2 * hg + h;
    let dim = units * hk;
    let slots = n_seqs + 1;
    let state_len = u64::from(s.conv_width - 1);
    let key = format!("{}:{}", s.n_seqs, s.t);

    // q/k/v/g/beta are the prep's OUTPUTS and the solve's and the scan's inputs, so they carry
    // one pool key across the three stages. The fill differs with the stage for the same reason:
    // a kernel that writes them wants them zeroed, one that reads them wants numbers.
    let produced = stage == Stage::ConvPrep;
    let num = |zeroed: bool| if zeroed { Fill::Zero } else { Fill::Bf16 };
    let numf = |zeroed: bool| if zeroed { Fill::Zero } else { Fill::F32State };

    let mut v = vec![
        spec(
            "cu",
            format!("pcu:{key}"),
            (n_seqs + 1) * 4,
            Fill::CuSeq { step: s.t },
        ),
        spec("q", format!("pq:{key}"), t * hg * hk * 2, num(produced)),
        spec("k", format!("pk:{key}"), t * hg * hk * 2, num(produced)),
        spec("v", format!("pv:{key}"), t * h * hv * 2, num(produced)),
        // The prep writes the gate and the decay as fp32; both later stages read them.
        spec("g", format!("pg:{key}"), t * h * 4, numf(produced)),
        spec("gcs", format!("pg:{key}"), t * h * 4, numf(produced)),
        spec("beta", format!("pbeta:{key}"), t * h * 4, numf(produced)),
    ];
    match stage {
        Stage::ConvPrep => v.extend([
            spec("x", format!("px:{key}"), t * dim * 2, Fill::Bf16),
            spec(
                "wgt",
                format!("pw:{}", s.conv_width),
                dim * u64::from(s.conv_width) * 2,
                Fill::Bf16,
            ),
            spec("bias", "pb".to_string(), dim * 2, Fill::Bf16),
            spec(
                "cstate",
                format!("pcs:{key}"),
                slots * dim * state_len * 2,
                Fill::Bf16,
            ),
            spec(
                "cache_idx",
                format!("pci:{}", s.n_seqs),
                n_seqs * 4,
                Fill::ConstI32(1),
            ),
            // A fresh prefill carries no conv state in: one byte per sequence, all zero.
            spec("has_init", format!("phi:{}", s.n_seqs), n_seqs, Fill::Zero),
            spec("av", format!("pav:{key}"), t * h * 2, Fill::Bf16),
            spec("bv", format!("pbv:{key}"), t * h * 2, Fill::Bf16),
            spec("a_log", format!("pal:{}", s.h), h * 4, Fill::F32State),
            spec("dt_bias", format!("pdt:{}", s.h), h * 4, Fill::F32State),
        ]),
        Stage::KktSolve => v.push(spec(
            "a_out",
            format!("pa:{key}"),
            t * h * chunk * 2,
            Fill::Zero,
        )),
        Stage::ChunkScan => v.extend([
            // The solve's output is the scan's input.
            spec("a_mat", format!("pa:{key}"), t * h * chunk * 2, Fill::Bf16),
            spec(
                "h0",
                format!("ph0:{}", s.n_seqs),
                n_seqs * h * hv * hk * 4,
                Fill::F32State,
            ),
            spec(
                "ht",
                format!("pht:{}", s.n_seqs),
                n_seqs * h * hv * hk * 4,
                Fill::Zero,
            ),
            spec("o", format!("po:{key}"), t * h * hv * 2, Fill::Zero),
        ]),
    }
    v
}

/// The gated RMS norm's buffers: one row per (token, head), `channels` wide. `w` is fp32.
fn gdn_norm_buffers(s: &NormShape) -> Vec<BufSpec> {
    let rows = s.rows;
    let ch = u64::from(s.channels);
    let key = format!("{rows}:{}", s.channels);
    vec![
        spec("x", format!("nx:{key}"), rows * ch * 2, Fill::Bf16),
        spec("z", format!("nz:{key}"), rows * ch * 2, Fill::Bf16),
        spec("w", format!("nw:{}", s.channels), ch * 4, Fill::F32State),
        spec("o", format!("no:{key}"), rows * ch * 2, Fill::Zero),
    ]
}

/// The paged attention decode route's buffers, at the reference capture script (`the internal reference harness`)'s shapes.
///
/// One block pool shared by every sequence's block table, as the capture does: the kernels only
/// read it, and a per-sequence pool at a 32k context would be 268 MiB of a ~1 GiB budget.
fn attn_buffers(s: &AttnShape, seqused_k: u32) -> Vec<BufSpec> {
    let rows = u64::from(s.num_seqs) * u64::from(s.q_len);
    let qh = u64::from(s.q_heads);
    let d = u64::from(s.head_dim);
    let blocks = u64::from(s.max_blocks);
    let kv_elems = blocks * u64::from(s.kv_heads) * u64::from(s.block_size) * 2 * d;
    // fp8 KV is one byte per element, bf16 two.
    let kv_bytes = kv_elems
        * if s.kv_dtype == saber::KvDtype::Bf16 {
            2
        } else {
            1
        };
    let key = format!("{}:{}:{}", s.num_seqs, s.q_len, s.max_blocks);
    vec![
        spec("q", format!("aq:{key}"), rows * qh * d * 2, Fill::Bf16),
        spec(
            "kv",
            format!(
                "akv:{}:{}",
                s.max_blocks,
                s.kv_dtype == saber::KvDtype::Bf16
            ),
            kv_bytes,
            // The cache is fp8 e4m3 in prod; the same byte range is valid bf16 nonsense at bf16.
            if s.kv_dtype == saber::KvDtype::Bf16 {
                Fill::Bf16
            } else {
                Fill::E4m3
            },
        ),
        spec(
            "block_table",
            format!("abt:{}:{}", s.num_seqs, s.max_blocks),
            u64::from(s.num_seqs) * blocks * 4,
            Fill::BlockTable {
                blocks: s.max_blocks,
            },
        ),
        spec(
            "seqused_k",
            format!("asl:{}:{seqused_k}", s.num_seqs),
            u64::from(s.num_seqs) * 4,
            // How many keys each sequence really has. At DECODE that is the bucket's context
            // band -- the split law already read `max_ctx`, and a shorter value would time a
            // shorter kernel than the plan budgeted. At PREFILL it is the chunk's own length.
            // The caller says which, because only the caller knows the phase.
            Fill::ConstI32(seqused_k as i32),
        ),
        spec("out", format!("ao:{key}"), rows * qh * d * 2, Fill::Zero),
    ]
}

/// The GDN decode step's buffers, at the reference capture script (`the internal reference harness`)'s shapes. The fused kernel and
/// the pair read exactly the same set; the pair simply splits the work across two launches.
fn gdn_decode_buffers(s: &GdnShape) -> Vec<BufSpec> {
    let (h, hg) = (u64::from(s.h), u64::from(s.hg));
    let (hk, hv) = (u64::from(s.head_k), u64::from(s.head_v));
    let t = u64::from(s.n_seqs) * u64::from(s.maxq);
    // One state slot per candidate token, plus the unused slot 0 and one spare.
    let slots = t + 2;
    let state_len = u64::from(s.conv_width - 1 + s.maxq - 1);
    let units = 2 * hg + h;
    let dim = units * hk;
    let key = format!("{}:{}", s.n_seqs, s.maxq);
    vec![
        spec("x", format!("gx:{key}"), t * dim * 2, Fill::Bf16),
        spec(
            "wgt",
            format!("gw:{}", s.conv_width),
            dim * u64::from(s.conv_width) * 2,
            Fill::Bf16,
        ),
        spec("bias", "gb".to_string(), dim * 2, Fill::Bf16),
        spec(
            "cstate",
            format!("gcs:{key}"),
            slots * dim * state_len * 2,
            Fill::Bf16,
        ),
        // `cache_idx` is a strided view of `sidx`: the same allocation, read with stride `maxq`.
        // Binding them to one buffer is not a shortcut, it is what the descriptor's `ci_stride`
        // means -- a separate allocation would walk off the end for every sequence but the first.
        spec("sidx", format!("gsi:{key}"), t * 4, Fill::SlotIndex),
        spec("cache_idx", format!("gsi:{key}"), t * 4, Fill::SlotIndex),
        spec(
            "naccept",
            format!("gna:{}", s.n_seqs),
            u64::from(s.n_seqs) * 4,
            Fill::ConstI32(1),
        ),
        spec(
            "cu",
            format!("gcu:{key}"),
            (u64::from(s.n_seqs) + 1) * 4,
            Fill::CuSeq { step: s.maxq },
        ),
        spec("q", format!("gq:{key}"), t * hg * hk * 2, Fill::Zero),
        spec("k", format!("gk:{key}"), t * hg * hk * 2, Fill::Zero),
        spec("v", format!("gv:{key}"), t * h * hv * 2, Fill::Zero),
        spec("av", format!("gav:{key}"), t * h * 2, Fill::Bf16),
        spec("bv", format!("gbv:{key}"), t * h * 2, Fill::Bf16),
        spec("a_log", format!("gal:{}", s.h), h * 4, Fill::F32State),
        spec("dt_bias", format!("gdt:{}", s.h), h * 4, Fill::F32State),
        spec(
            "state",
            format!("gst:{key}"),
            slots * h * hv * hk * 4,
            Fill::F32State,
        ),
        spec(
            "o",
            format!("go:{key}"),
            u64::from(s.o_rows) * h * hv * 2,
            Fill::Zero,
        ),
    ]
}

/// The MXFP4 family's buffers.
fn mxfp4_buffers(op: Op, shape: Shape, tiled: bool) -> Vec<BufSpec> {
    let Shape { m, n, k } = shape;
    let (m64, n64, k64) = (u64::from(m), u64::from(n), u64::from(k));
    let rows = a_rows(m, tiled);
    match op {
        Op::Gemm { has_wref, .. } => {
            let mut v = vec![
                spec("a", format!("a:{rows}:{k}"), rows * k64, Fill::E4m3),
                spec("w", format!("w:{n}:{k}"), n64 * (k64 / 2), Fill::Nibble),
                spec("ws", format!("ws:{n}:{k}"), (k64 / 32) * n64, Fill::E8m0),
                spec("as", format!("as:{m}"), m64 * 4, Fill::Scale),
                spec("c", format!("c:{m}:{n}"), m64 * n64 * 2, Fill::Zero),
            ];
            if has_wref {
                v.push(spec("wref", format!("wref:{n}"), n64, Fill::E8m0));
            }
            v
        }
        Op::AddRmsQuant { .. } => vec![
            spec("y", format!("y:{m}:{k}"), m64 * k64 * 2, Fill::Bf16),
            spec("res", format!("res:{m}:{k}"), m64 * k64 * 2, Fill::Bf16),
            spec("w", format!("rmsw:{k}"), k64 * 2, Fill::Bf16),
            spec("q", format!("q:{rows}:{k}"), rows * k64, Fill::Zero),
            spec("scale", format!("scale:{m}"), m64 * 4, Fill::Zero),
            spec(
                "res_out",
                format!("resout:{m}:{k}"),
                m64 * k64 * 2,
                Fill::Zero,
            ),
        ],
        Op::SiluMulQuant => vec![
            spec("gu", format!("gu:{m}:{n}"), m64 * n64 * 4, Fill::Bf16),
            spec("q", format!("q:{rows}:{n}"), rows * n64, Fill::Zero),
            spec("scale", format!("scale:{m}"), m64 * 4, Fill::Zero),
        ],
        Op::GdnNormQuant { zs, .. } => vec![
            spec("x", format!("x:{m}:{n}"), m64 * n64 * 2, Fill::Bf16),
            spec(
                "z",
                format!("z:{m}:{zs}"),
                m64 * (zs.unsigned_abs()) * 2,
                Fill::Bf16,
            ),
            spec("w", format!("gdnw:{n}"), n64 * 2, Fill::Bf16),
            spec("q", format!("q:{m}:{n}"), m64 * n64, Fill::Zero),
            spec("scale", format!("scale:{m}"), m64 * 4, Fill::Zero),
        ],
    }
}

/// xorshift64*, seeded per buffer key so a re-run fills the same bytes.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Mix first: two adjacent seeds must not collapse to the same state.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn fill_bytes(fill: Fill, bytes: u64, seed: u64) -> Vec<u8> {
    let n = bytes as usize;
    let mut out = vec![0u8; n];
    if fill == Fill::Zero {
        return out;
    }
    let mut rng = Rng::new(seed);
    match fill {
        Fill::Nibble => {
            for b in out.iter_mut() {
                *b = (rng.next() >> 24) as u8;
            }
        }
        Fill::E4m3 => {
            // sign | exp in 4..12 | mantissa: magnitudes 2^-3 .. 2^4, never NaN (exp 15, mant 7).
            for b in out.iter_mut() {
                let r = rng.next();
                let sign = ((r >> 3) & 1) as u8;
                let exp = 4 + ((r >> 8) % 8) as u8;
                let mant = ((r >> 16) & 7) as u8;
                *b = (sign << 7) | (exp << 3) | mant;
            }
        }
        Fill::E8m0 => {
            for b in out.iter_mut() {
                let r = rng.next();
                *b = if r % 100 == 0 {
                    0 // a zero block: 2^-127 reads as 0.0
                } else {
                    (118 + (r >> 8) % 17) as u8
                };
            }
        }
        Fill::Bf16 => {
            for pair in out.chunks_mut(2) {
                let r = rng.next();
                let sign = ((r >> 3) & 1) as u16;
                let exp = (124 + (r >> 8) % 7) as u16; // 2^-3 .. 2^3
                let mant = ((r >> 16) & 0x7f) as u16;
                let v = (sign << 15) | (exp << 7) | mant;
                let le = v.to_le_bytes();
                for (i, dst) in pair.iter_mut().enumerate() {
                    *dst = le[i];
                }
            }
        }
        Fill::Scale => {
            for word in out.chunks_mut(4) {
                let r = rng.next();
                let v = 0.001f32 + (r % 1000) as f32 * 2.0e-5;
                let le = v.to_le_bytes();
                for (i, dst) in word.iter_mut().enumerate() {
                    *dst = le[i];
                }
            }
        }
        Fill::F32State => {
            for word in out.chunks_mut(4) {
                let r = rng.next();
                let v = ((r >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 * 0.1;
                let le = v.to_le_bytes();
                for (i, dst) in word.iter_mut().enumerate() {
                    *dst = le[i];
                }
            }
        }
        Fill::SlotIndex => {
            for (i, word) in out.chunks_mut(4).enumerate() {
                let le = (i as i32 + 1).to_le_bytes();
                for (j, dst) in word.iter_mut().enumerate() {
                    *dst = le[j];
                }
            }
        }
        Fill::CuSeq { step } => {
            for (i, word) in out.chunks_mut(4).enumerate() {
                let le = (i as i32 * step as i32).to_le_bytes();
                for (j, dst) in word.iter_mut().enumerate() {
                    *dst = le[j];
                }
            }
        }
        Fill::BlockTable { blocks } => {
            for (i, word) in out.chunks_mut(4).enumerate() {
                let le = ((i as u32 % blocks.max(1)) as i32).to_le_bytes();
                for (j, dst) in word.iter_mut().enumerate() {
                    *dst = le[j];
                }
            }
        }
        Fill::RopeCache {
            max_pos,
            rotary_dim,
        } => {
            // The real table, from the forward library, so the sweep reads the values service
            // reads. `out` was sized from the same two numbers, so this fills it exactly.
            let table = rope_cache(max_pos, rotary_dim, crate::forward::ROPE_THETA);
            let n = table.len().min(out.len());
            out[..n].copy_from_slice(&table[..n]);
        }
        Fill::MropePos {
            seqs,
            q_len,
            done_before,
        } => {
            let mut base: Vec<i64> = Vec::with_capacity((seqs as usize) * (q_len as usize));
            for _ in 0..seqs {
                for t in 0..q_len {
                    base.push(i64::from(done_before + t));
                }
            }
            let mut v = Vec::with_capacity(base.len() * 3);
            for _ in 0..3 {
                v.extend_from_slice(&base);
            }
            let bytes = i64_bytes(&v);
            let n = bytes.len().min(out.len());
            out[..n].copy_from_slice(&bytes[..n]);
        }
        Fill::SlotMap { rows } => {
            let bytes = i64_bytes(&(0..i64::from(rows)).collect::<Vec<_>>());
            let n = bytes.len().min(out.len());
            out[..n].copy_from_slice(&bytes[..n]);
        }
        Fill::ConstF32(v) => {
            for c in out.chunks_exact_mut(4) {
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        Fill::ConstI32(v) => {
            for word in out.chunks_mut(4) {
                let le = v.to_le_bytes();
                for (j, dst) in word.iter_mut().enumerate() {
                    *dst = le[j];
                }
            }
        }
        Fill::Zero => unreachable!("handled above"),
    }
    out
}

// ------------------------------------------------------------------------------- sweep

/// How many samples, and how much VRAM a shape may claim.
#[derive(Clone, Copy, Debug)]
pub struct SweepConfig {
    /// Untimed rounds. Each round launches every variant once.
    pub warmup: usize,
    /// Timed rounds. Each round contributes one sample to every variant.
    pub samples: usize,
    /// A shape whose buffers exceed this is skipped and reported. Prod owns the card.
    pub budget_bytes: u64,
    /// Rotate every streamed buffer through enough copies that no sample reads a warm cache line
    /// (see [`IC_BYTES`]). Default on: the step is cold, so the measurement is cold. Off is for
    /// the A/B that shows what the old, hot number was.
    pub ic_cold: bool,
}

impl Default for SweepConfig {
    fn default() -> Self {
        SweepConfig {
            warmup: 5,
            samples: 20,
            budget_bytes: 520 << 20,
            ic_cold: true,
        }
    }
}

/// Bind round `round` of the rotation: copy `round % copies` of every buffer that has more than
/// one. A buffer with a single copy is bound to it every round, which is what a shape at or above
/// the cache -- or below the rotation floor -- wants.
fn rotate(
    pool: &BTreeMap<String, DevicePtr>,
    specs: &[BufSpec],
    copies: &[u64],
    round: u64,
    bindings: &mut ArgBindings,
) {
    for (s, n) in specs.iter().zip(copies) {
        let key = pool_key(&s.key, if *n == 1 { 0 } else { round % *n });
        bindings.bind(s.name, pool[&key].addr());
    }
}

/// The device's Infinity Cache, and the reason a solo median can be a fiction.
///
/// A sweep times one variant over and over on ONE set of buffers. Anything that fits the IC is
/// therefore served from it on every sample after the first, while the step it is standing in for
/// streams 15.4 GB of weights per token and never hits a warm line. H12 priced the gap at 2x on
/// the GEMM sums. Every weight in this model below 64 MB was measured hot: `gate_up` at TP=2 is
/// 47.4 MB, `down` 47.4, `in_proj_qkvz` 21, `qkv` 18.4, `o_proj` 7.9.
const IC_BYTES: u64 = 64 << 20;

/// What the scrub writes between two timed samples: twice the cache, so nothing a sample read
/// can still be resident when the next one starts.
///
/// The rotation alone was NOT enough, and the roofline said so. At `gate_up` (N=17408 at TP=2,
/// 47.4 MB of weights, a 74.7 µs floor at 635 GB/s) rotating three copies left
/// `bk128_ks2_tm1` reading at **925 GB/s** -- 1.46x DRAM peak, which is not a number DRAM can
/// produce. Three copies cover the cache by the arithmetic and still do not evict it in practice:
/// a memory-side cache is not an LRU queue. A write of 2x the cache is.
const SCRUB_BYTES: u64 = 128 << 20;

/// Buffers smaller than this are not rotated. Below a MiB the copies would be hundreds of
/// allocations for a tensor whose residency is not what the step's cost is made of -- and the
/// activations, which ARE hot in the step (a producer just wrote them), live down here.
const ROTATE_MIN_BYTES: u64 = 1 << 20;

/// How many copies of one buffer the rotation needs so that returning to a copy finds it evicted.
///
/// `(copies - 1) * bytes >= IC_BYTES`: by the time the cycle comes back, at least a cache's worth
/// of other bytes has been streamed through. A buffer at or above the cache is already self-cold
/// -- its own tail evicts its head -- so it stays at one copy, which is what keeps `lm_head`'s
/// 338 MB from asking for a second.
fn rotation_copies(bytes: u64, ic_cold: bool) -> u64 {
    if !ic_cold || !(ROTATE_MIN_BYTES..IC_BYTES).contains(&bytes) {
        1
    } else {
        1 + IC_BYTES.div_ceil(bytes)
    }
}

/// The pool key of one copy.
fn pool_key(key: &str, copy: u64) -> String {
    if copy == 0 {
        key.to_owned()
    } else {
        format!("{key}#{copy}")
    }
}

/// One variant's raw samples, in the order they were measured.
#[derive(Clone, Debug, PartialEq)]
pub struct VariantSamples {
    /// The plugin the symbol lives in.
    pub plugin: String,
    /// The symbol that ran.
    pub kernel: String,
    /// The knobs that produced it.
    pub params: BTreeMap<String, i64>,
    /// FNV-1a/32 of the symbol.
    pub variant_id: u32,
    /// Microseconds, one per timed round, in measurement order.
    pub samples: Vec<f64>,
}

/// What a shape's sweep produced.
#[derive(Clone, Debug)]
pub enum ShapeOutcome {
    /// Measured. One entry per variant, in enumeration order.
    Swept(Vec<VariantSamples>),
    /// Not measured, and why. A skip is a reportable outcome, never a silent gap.
    Skipped {
        /// Human-readable reason.
        reason: String,
        /// Device bytes the shape would have needed.
        bytes: u64,
    },
}

/// The flat measurement order for `variants` over `rounds`: `A B C A B C …`.
///
/// Round-robin rather than blocked, so that clock and power drift over the sweep lands on every
/// variant equally instead of on whichever one was measured last. The paired-A/B method in
/// `LEDGER.md` is the same idea with two arms.
#[must_use]
pub fn interleaved_order(variants: usize, rounds: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(variants * rounds);
    for _ in 0..rounds {
        for v in 0..variants {
            out.push(v);
        }
    }
    out
}

/// Times variants over one stream, reusing device buffers between shapes that share them.
pub struct Sweeper<'a, 'f> {
    ctx: &'a LaunchCtx<'f>,
    stream: &'a Stream,
    launcher: Launcher,
    pool: BTreeMap<String, DevicePtr>,
    /// The scrub buffer: [`SCRUB_BYTES`] that hold nothing, written between timed samples so no
    /// sample can read a line the previous one warmed. Allocated on first use under `ic_cold`.
    scrub: Option<DevicePtr>,
}

impl<'a, 'f> Sweeper<'a, 'f> {
    /// A sweeper over one launch context and one stream.
    pub fn new(ctx: &'a LaunchCtx<'f>, stream: &'a Stream, launcher: Launcher) -> Self {
        Sweeper {
            ctx,
            stream,
            launcher,
            pool: BTreeMap::new(),
            scrub: None,
        }
    }

    /// Device bytes the input pool currently holds.
    #[must_use]
    pub fn pool_bytes(&self) -> u64 {
        self.pool.values().map(DevicePtr::len).sum()
    }

    /// The launcher, for its arena figures and counters.
    #[must_use]
    pub fn launcher(&self) -> &Launcher {
        &self.launcher
    }

    /// Drops every pooled input buffer. The workspace arenas belong to the launcher and stay.
    pub fn release_inputs(&mut self) {
        self.pool.clear();
    }

    /// Warm up, then time every variant round-robin.
    ///
    /// Samples come back in measurement order and the first variant's first sample is the shape's
    /// warm-up as far as `Autotuner::record_measurements` is concerned — this harness discards
    /// nothing itself, so the protocol lives in exactly one place.
    pub fn sweep(
        &mut self,
        op: &TuneOp,
        variants: &[Variant],
        cfg: &SweepConfig,
    ) -> Result<ShapeOutcome> {
        if variants.is_empty() {
            return Ok(ShapeOutcome::Skipped {
                reason: "no legal variant for this shape".to_string(),
                bytes: 0,
            });
        }
        if cfg.samples < 2 {
            return Err(Error::Refused(
                "a sweep needs at least 2 timed rounds: the first sample of the first variant is \
                 the shape's warm-up and is discarded"
                    .to_string(),
            ));
        }
        let tiled = variants.iter().any(|v| v.desc().op == "gemm_atiled");
        let specs = buffers_for(op, tiled);
        // Before a single launch: the inputs this case is about to time must be the ones the step
        // feeds, derived by the step's own law. `seqused_k = max_ctx` at prefill passed every gate
        // in this repo for four caches because the SHAPE was right and only the INPUT was wrong
        // (I18); this is the check that refuses it, and it names the op.
        index_law_report(op, &specs).map_err(Error::Refused)?;
        // A shape whose OWN inputs already exceed the cache is self-cold: one pass over them
        // evicts the start of the pass. Rotating it would multiply a 285 MB activation for
        // nothing and push the shape over the budget -- which is how `p8192 gate_up` and
        // `p8192 chunk_scan` went untuned on the first cold sweep. The scrub between samples is
        // what makes the eviction certain either way; the rotation is only worth its memory
        // where the inputs are small enough to stay resident.
        let resident = specs.iter().map(|s| s.bytes).sum::<u64>() < IC_BYTES;
        let copies: Vec<u64> = specs
            .iter()
            .map(|s| rotation_copies(s.bytes, cfg.ic_cold && resident))
            .collect();
        let total: u64 = specs.iter().zip(&copies).map(|(s, c)| s.bytes * c).sum();
        if total > cfg.budget_bytes {
            return Ok(ShapeOutcome::Skipped {
                reason: format!(
                    "{} MiB of inputs exceeds the {} MiB budget",
                    total >> 20,
                    cfg.budget_bytes >> 20
                ),
                bytes: total,
            });
        }

        // Keep only what this shape needs, then allocate and fill the rest. Weights survive a
        // change of M and are dropped when (n, k) moves on.
        let wanted: BTreeSet<String> = specs
            .iter()
            .zip(&copies)
            .flat_map(|(s, c)| (0..*c).map(|i| pool_key(&s.key, i)))
            .collect();
        self.pool.retain(|k, _| wanted.contains(k));

        // The shape the launcher keys a Persistent workspace on. For a saber route the geometry
        // is not (m, n, k), so it is projected onto one: what matters is only that two different
        // routes never collide on it, which `TuneOp::workspace_key` guarantees.
        let mut bindings = ArgBindings::new(op.workspace_key());
        for (s, n) in specs.iter().zip(&copies) {
            // Every copy holds the SAME bytes: the rotation moves the ADDRESS, never the data, so
            // each sample does identical work on a cache line nobody has touched recently.
            let mut host: Option<Vec<u8>> = None;
            for i in 0..*n {
                let key = pool_key(&s.key, i);
                if self.pool.contains_key(&key) {
                    continue;
                }
                let ptr = self.ctx.hip.malloc(s.bytes.max(1))?;
                let bytes =
                    host.get_or_insert_with(|| fill_bytes(s.fill, s.bytes, fnv1a32(&s.key).into()));
                self.ctx.hip.memcpy_h2d_blocking(&ptr, bytes, self.stream)?;
                self.pool.insert(key, ptr);
            }
        }
        self.ctx.hip.sync_stream(self.stream)?;

        // Warm-up rounds are untimed and round-robin too: they warm every variant's code path,
        // its workspace allocation and (for a split) the one-off counter memset.
        // The rotation advances per LAUNCH, warm-ups included: two consecutive launches never
        // read the same copy, so the interleave's other variants cannot keep a copy warm either.
        let mut round = 0u64;
        for _ in 0..cfg.warmup {
            for v in variants {
                rotate(&self.pool, &specs, &copies, round, &mut bindings);
                round += 1;
                let descs: Vec<&LaunchDescriptor> = v.descs.iter().collect();
                self.launcher
                    .launch_op(self.ctx, &descs, &bindings, self.stream)?;
            }
        }
        // Every copy must have been touched before the first timed sample, or the first sample of
        // a long rotation pays a first-touch the others do not.
        let widest = copies.iter().copied().max().unwrap_or(1);
        while round < widest {
            rotate(&self.pool, &specs, &copies, round, &mut bindings);
            round += 1;
            let descs: Vec<&LaunchDescriptor> = variants[0].descs.iter().collect();
            self.launcher
                .launch_op(self.ctx, &descs, &bindings, self.stream)?;
        }
        self.ctx.hip.sync_stream(self.stream)?;

        // One scrub buffer for the whole sweep, allocated on first use and kept: it is 128 MB
        // that never holds anything, and re-allocating it per shape would be 67 allocations of it.
        if cfg.ic_cold && self.scrub.is_none() {
            self.scrub = Some(self.ctx.hip.malloc(SCRUB_BYTES)?);
        }

        let start = self.ctx.hip.create_event(EventFlags::Default)?;
        let stop = self.ctx.hip.create_event(EventFlags::Default)?;
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(cfg.samples); variants.len()];
        for i in interleaved_order(variants.len(), cfg.samples) {
            let v = &variants[i];
            rotate(&self.pool, &specs, &copies, round, &mut bindings);
            // Scrub BEFORE the bracket: same stream, so it is ordered ahead of the launch and the
            // start event is recorded after it. The sample pays no part of the scrub's own time.
            if let Some(scrub) = &self.scrub {
                self.ctx
                    .hip
                    .memset(scrub, (round & 0xff) as u8, self.stream)?;
            }
            round += 1;
            self.ctx.hip.record_event(&start, self.stream)?;
            // Every launch of the route, inside ONE bracket and under ONE launch budget: a
            // split-KV decode without its combine, or a conv_update without its
            // recurrent_update, is not a thing the engine ever runs, so timing one of them would
            // answer a question nobody asked.
            let descs: Vec<&LaunchDescriptor> = v.descs.iter().collect();
            self.launcher
                .launch_op(self.ctx, &descs, &bindings, self.stream)?;
            self.ctx.hip.record_event(&stop, self.stream)?;
            self.ctx.hip.sync_event(&stop)?;
            let us = f64::from(self.ctx.hip.elapsed_ms(&start, &stop)?) * 1000.0;
            samples[i].push(us);
        }

        Ok(ShapeOutcome::Swept(
            variants
                .iter()
                .zip(samples)
                .map(|(v, s)| VariantSamples {
                    plugin: v.plugin.clone(),
                    kernel: v.kernel.clone(),
                    params: v.params.clone(),
                    variant_id: v.variant_id(),
                    samples: s,
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::NO_PARTITIONS;

    fn rules() -> LaunchRules {
        LaunchRules::prod_mxfp4().expect("prod rules parse")
    }

    fn saber_rules() -> SaberRules {
        SaberRules::prod_saber().expect("prod saber rules parse")
    }

    fn symbols() -> SymbolIndex {
        SymbolIndex::from_manifest_dir(&saber_kernels::prebuilt::dir())
            .expect("the prebuilt manifests are committed")
    }

    fn kernels(set: &VariantSet) -> Vec<&str> {
        set.variants.iter().map(|v| v.kernel.as_str()).collect()
    }

    /// The rotation is sized so that coming back to a copy finds it evicted, and it only applies
    /// where residency is a lie about the step.
    ///
    /// Every weight in this model under the 64 MB cache was being measured HOT: `gate_up` at TP=2
    /// is 47.4 MB, `down` 47.4, `in_proj_qkvz` 21, `qkv` 18.4, `o_proj` 7.9. The step streams
    /// 15.4 GB of weights per token, so none of them is warm when it runs.
    #[test]
    fn the_rotation_covers_the_cache_for_everything_that_fits_it() {
        for mib in [1u64, 2, 8, 18, 21, 47, 63] {
            let bytes = mib << 20;
            let n = rotation_copies(bytes, true);
            assert!(
                (n - 1) * bytes >= IC_BYTES,
                "{mib} MiB rotated {n} ways covers only {} MiB",
                ((n - 1) * bytes) >> 20
            );
            // ... and not wastefully: one copy fewer would not cover it.
            assert!(
                (n - 2) * bytes < IC_BYTES,
                "{mib} MiB: {n} copies is more than the cache needs"
            );
            assert!(
                n * bytes <= 2 * IC_BYTES + bytes,
                "{mib} MiB: {n} copies is unbounded"
            );
        }
        // At or above the cache a buffer is self-cold: its own tail evicts its head. `lm_head`'s
        // 338 MB must not ask for a second copy -- there is no budget for one.
        assert_eq!(rotation_copies(IC_BYTES, true), 1);
        assert_eq!(rotation_copies(338 << 20, true), 1);
        // Below the floor the copies would be hundreds of allocations for a tensor that is warm
        // in the step anyway (a producer just wrote it).
        assert_eq!(rotation_copies(ROTATE_MIN_BYTES - 1, true), 1);
    }

    /// The negative control for the row above: with the rotation OFF -- the pre-2026-09-19
    /// sweeper, and the `--ic-hot` arm -- every buffer has exactly one copy, which is the
    /// configuration that measured `gate_up` out of the Infinity Cache. If this test passes while
    /// the one above also passes, the flag is doing something.
    #[test]
    fn with_the_rotation_off_every_buffer_is_read_from_one_place() {
        for mib in [1u64, 8, 21, 47, 63, 338] {
            assert_eq!(rotation_copies(mib << 20, false), 1, "{mib} MiB");
        }
    }

    /// The pool key names copy 0 exactly as it always did, so a shape that is not rotated keeps
    /// its weights across a change of M -- which is what makes one sweep of 67 keys affordable.
    #[test]
    fn the_first_copy_keeps_the_plain_pool_key() {
        assert_eq!(pool_key("w_n17408_k5120", 0), "w_n17408_k5120");
        assert_eq!(pool_key("w_n17408_k5120", 3), "w_n17408_k5120#3");
    }

    #[test]
    fn the_symbol_index_is_the_committed_code_objects() {
        let s = symbols();
        // 121 decode + 8 folded + 4 A-tiled + 1 base + 9 quant = 143 MXFP4 symbols; the saber
        // plugins add more, so the index must hold at least the MXFP4 family.
        assert!(s.len() >= 143, "symbol index holds {} symbols", s.len());
        assert!(s.has(
            "mxfp4_gemm_decode",
            "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1"
        ));
        // The cell the decode ladder cannot reach: BK=64 exists for splits 1 and 4 only.
        assert!(!s.has(
            "mxfp4_gemm_decode",
            "mxfp4_gemm_decode_bk64_ks2_tm1_wperm1_nt1"
        ));
        assert!(!s.has("mxfp4_gemm_atiled", "mxfp4_gemm_atiled_tn2_lbk64_wperm1"));
    }

    #[test]
    fn a_decode_shape_enumerates_five_splits_and_two_folded_tiles() {
        // gate_up at the 8x5 bucket: M=40, N=8704 (TP=2), K=5120.
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(40, 8704, 5120),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        assert_eq!(
            kernels(&set),
            vec![
                "mxfp4_gemm_decode_bk64_ks1_tm3_wperm1_nt1",
                "mxfp4_gemm_decode_bk128_ks1_tm3_wperm1_nt1",
                "mxfp4_gemm_decode_bk128_ks2_tm3_wperm1_nt1",
                "mxfp4_gemm_decode_bk64_ks4_tm3_wperm1_nt1",
                "mxfp4_gemm_decode_bk128_ks4_tm3_wperm1_nt1",
                "mxfp4_gemm_folded_tn2_wperm1_epifast1",
                "mxfp4_gemm_folded_tn4_wperm1_epifast1",
            ],
            "5 decode cells (bk64 has no ks2) x 2 folded tiles"
        );
        // The refusal is recorded rather than dropped.
        assert!(
            set.refused
                .iter()
                .any(|r| r.contains("bk64_ks2") && r.contains("not a symbol")),
            "{:?}",
            set.refused
        );
    }

    #[test]
    fn every_variant_carries_the_knobs_the_planner_sizes_from() {
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(40, 8704, 5120),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        for v in &set.variants {
            assert!(v.params.contains_key("wperm"), "{}", v.kernel);
            if v.desc().op == "gemm_decode" {
                let split = v.params["split_k"];
                assert!(matches!(split, 1 | 2 | 4), "{}", v.kernel);
                // The workspace the planner will size is exactly the one the descriptor asks for.
                assert_eq!(v.desc().workspaces.len(), usize::from(split > 1) * 2);
            }
        }
    }

    /// The vocabulary GEMM offers BOTH families, and that is the change of 2026-09-19: `dec_max_n`
    /// is the split-K scratch's width, so it stopped gating the direct arm and the decode tile
    /// became legal at `N = 124160`. Before, this shape enumerated the folded tile alone and the
    /// sweep could only choose between two prefill tiles for a five-row GEMM.
    ///
    /// Only the `ks = 1` decode cells are offered -- the fill rule answers 1 at nblk 970, and the
    /// enumerator's split/bk sweep refuses the rest -- which is exactly the arm that needs no
    /// scratch.
    #[test]
    fn the_vocabulary_gemm_offers_the_decode_tile_as_well_as_the_folded_one() {
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(40, 124_160, 5120),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        assert_eq!(
            kernels(&set),
            vec![
                "mxfp4_gemm_decode_bk64_ks1_tm3_wperm1_nt1",
                "mxfp4_gemm_decode_bk128_ks1_tm3_wperm1_nt1",
                "mxfp4_gemm_folded_tn2_wperm1_epifast1",
                "mxfp4_gemm_folded_tn4_wperm1_epifast1",
            ]
        );
        // Neither decode variant asks for a workspace: the direct arm writes C and binds the
        // partial slab and the counter null, which is why the cap did not apply to it.
        for v in &set.variants {
            if v.desc().op == "gemm_decode" {
                assert_eq!(v.params["split_k"], 1, "{}", v.kernel);
                assert!(v.desc().workspaces.is_empty(), "{}", v.kernel);
            }
        }
        assert_eq!(
            set.refused.len(),
            4,
            "the split cells are still refused, by the rules, with a reason: {:?}",
            set.refused
        );
    }

    #[test]
    fn above_the_a_tiled_threshold_only_the_a_tiled_layout_is_offered() {
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(2048, 8704, 5120),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        assert_eq!(
            kernels(&set),
            vec![
                "mxfp4_gemm_atiled_tn2_lbk128_wperm1",
                "mxfp4_gemm_atiled_tn4_lbk64_wperm1",
                // The tuner offers every tall body a `[[atiled_wm]]` row names; since G4 both
                // rows name G3's `_wm8sp` (G1's `_wm8` stays compiled, bound nowhere).
                "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp"
            ],
            "the producer writes fragment-tiled activations at this M; folded cannot read them"
        );
        // tn2/lbk64 and tn4/lbk128 are legal settings with no code object behind them, at every
        // tile body (wm=4, G3 wm=8sp; the tall one is compiled for tn4/lbk64 only).
        assert_eq!(set.refused.len(), 4, "{:?}", set.refused);
    }

    #[test]
    fn the_unfolded_path_has_exactly_one_instantiation() {
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::Gemm {
                    has_wref: false,
                    partitions: NO_PARTITIONS,
                },
                shape: Shape::new(8, 10240, 5120),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        assert_eq!(kernels(&set), vec!["mxfp4_gemm_base"]);
    }

    #[test]
    fn the_producers_offer_their_two_block_widths() {
        let s = symbols();
        let r = rules();
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::AddRmsQuant { eps: 1e-6 },
                shape: Shape::new(8, 0, 5120),
            },
            &r,
            &saber_rules(),
            &s,
        );
        assert_eq!(
            kernels(&set),
            vec!["add_rms_quant_t256_tiled0", "add_rms_quant_t512_tiled0"]
        );
        assert_eq!(set.variants[1].params["block"], 512);
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::SiluMulQuant,
                shape: Shape::new(8, 5120, 0),
            },
            &r,
            &saber_rules(),
            &s,
        );
        assert_eq!(
            kernels(&set),
            vec!["silu_mul_quant_t256_tiled0", "silu_mul_quant_t512_tiled0"]
        );
        // One instantiation, so one variant and no tie to break.
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::GdnNormQuant {
                    zs: 3072,
                    eps: 1e-6,
                },
                shape: Shape::new(8, 3072, 0),
            },
            &r,
            &saber_rules(),
            &s,
        );
        assert_eq!(kernels(&set), vec!["gdn_norm_quant_t256"]);
    }

    #[test]
    fn an_illegal_geometry_enumerates_nothing_and_says_why() {
        // K not a multiple of the GEMM's K slab: refused, never rounded.
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(8, 5120, 5100),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        assert!(set.variants.is_empty());
        assert!(
            set.refused.iter().all(|r| r.contains("multiple of 64")),
            "{:?}",
            set.refused
        );
    }

    #[test]
    fn measurement_order_is_round_robin_not_blocked() {
        assert_eq!(
            interleaved_order(3, 4),
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        );
        // The negative control: blocking would put every sample of variant 0 first, so drift over
        // the sweep would land on variant 2 alone.
        let blocked: Vec<usize> = (0..3).flat_map(|v| std::iter::repeat_n(v, 4)).collect();
        assert_ne!(interleaved_order(3, 4), blocked);
        assert_eq!(interleaved_order(0, 4), Vec::<usize>::new());
        assert_eq!(interleaved_order(3, 0), Vec::<usize>::new());
    }

    #[test]
    fn the_variant_id_is_a_stable_function_of_the_symbol() {
        let set = enumerate_variants(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(40, 8704, 5120),
            },
            &rules(),
            &saber_rules(),
            &symbols(),
        );
        let ids: BTreeSet<u32> = set.variants.iter().map(Variant::variant_id).collect();
        assert_eq!(ids.len(), set.variants.len(), "no id collides in a sweep");
        assert_eq!(
            fnv1a32("mxfp4_gemm_folded_tn2_wperm1_epifast1"),
            1_704_097_558
        );
    }

    #[test]
    fn parsed_knobs_come_from_the_symbol_that_will_run() {
        let p = params_from_symbol("mxfp4_gemm_decode_bk64_ks4_tm3_wperm1_nt1");
        assert_eq!(p["bk"], 64);
        assert_eq!(p["split_k"], 4);
        assert_eq!(p["tm"], 3);
        assert_eq!(p["wperm"], 1);
        assert_eq!(p["nt"], 1);
        let p = params_from_symbol("mxfp4_gemm_atiled_tn4_lbk64_wperm1");
        assert_eq!(p["tn"], 4);
        assert_eq!(p["lbk"], 64);
        assert!(!p.contains_key("bk"), "`lbk64` is not a `bk`");
        let p = params_from_symbol("add_rms_quant_t512_tiled1");
        assert_eq!(p["block"], 512);
        assert_eq!(p["tiled"], 1);
        assert!(params_from_symbol("mxfp4_gemm_base").is_empty());
        // G3's tile bodies carry their height and their body on one token.
        let p = params_from_symbol("mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp");
        assert_eq!((p["wm"], p["saddr"], p["wpipe"], p["tn"]), (8, 1, 1, 4));
        let p = params_from_symbol("mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8s");
        assert_eq!((p["wm"], p["saddr"], p["wpipe"]), (8, 1, 0));
        let p = params_from_symbol("mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8");
        assert_eq!(p["wm"], 8);
        assert!(!p.contains_key("saddr"), "G1's body is not the SADDR one");
        // negative control: an unknown body letter is not silently read as a height
        assert!(!params_from_symbol("mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8x").contains_key("wm"));
    }

    #[test]
    fn inputs_are_valid_numbers_not_uniform_bytes() {
        // e4m3: 0x7f and 0xff are NaN. A NaN activation is a different workload, not a tile.
        let a = fill_bytes(Fill::E4m3, 4096, 7);
        assert!(a.iter().all(|b| b & 0x7f != 0x7f));
        assert!(a.iter().any(|b| *b != a[0]), "and it is not a constant");
        // e8m0 scales: the capture's 118..=134 band, with a few zero blocks.
        let ws = fill_bytes(Fill::E8m0, 4096, 9);
        assert!(ws.iter().all(|b| *b == 0 || (118..=134).contains(b)));
        assert!(ws.contains(&0));
        // bf16 around 1.0: never inf, never NaN.
        let bf = fill_bytes(Fill::Bf16, 4096, 11);
        for pair in bf.chunks(2) {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            assert!(v.is_finite() && v.abs() > 0.0 && v.abs() < 16.0, "{v}");
        }
        assert!(fill_bytes(Fill::Zero, 64, 3).iter().all(|b| *b == 0));
    }

    #[test]
    fn filling_is_deterministic_so_a_re_run_measures_the_same_bytes() {
        assert_eq!(
            fill_bytes(Fill::Bf16, 512, 42),
            fill_bytes(Fill::Bf16, 512, 42)
        );
        assert_ne!(
            fill_bytes(Fill::Bf16, 512, 42),
            fill_bytes(Fill::Bf16, 512, 43)
        );
    }

    #[test]
    fn weights_are_pooled_per_shape_and_activations_per_m() {
        let at5 = buffers_for(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(5, 8704, 5120),
            },
            false,
        );
        let at40 = buffers_for(
            &TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape: Shape::new(40, 8704, 5120),
            },
            false,
        );
        let key = |v: &[BufSpec], n: &str| {
            v.iter()
                .find(|s| s.name == n)
                .expect("named buffer")
                .key
                .clone()
        };
        assert_eq!(
            key(&at5, "w"),
            key(&at40, "w"),
            "weights survive a change of M"
        );
        assert_eq!(key(&at5, "ws"), key(&at40, "ws"));
        assert_ne!(key(&at5, "a"), key(&at40, "a"));
        // The A-tiled layout pads to whole 16-row tiles; the row-major one does not.
        assert_eq!(a_rows(40, false), 40);
        assert_eq!(a_rows(40, true), 48);
    }

    /// The TP=2 prefill attention cell, as the sweep resolves it: 2048 queries of one sequence
    /// against a 8192-token context band.
    fn prefill_cell() -> AttnShape {
        AttnShape {
            num_seqs: 1,
            q_len: 2048,
            live_rows: 2048,
            q_heads: 12,
            kv_heads: 2,
            head_dim: 256,
            block_size: 16,
            max_blocks: 8192 / 16,
            max_ctx: 8192,
            splits: 0,
            kv_dtype: saber::KvDtype::Fp8E4m3,
            kv_block_stride: 2 * 16 * 2 * 256,
            kv_head_stride: 16 * 2 * 256,
            scale_bits: (f64::from(256u32).powf(-0.5) as f32).to_bits(),
            has_descales: false,
            ragged_rows: 0,
        }
    }

    #[test]
    fn uniform16_tunes_the_six_wave_head_and_its_actual_combine() {
        let s = AttnShape {
            q_len: 16,
            live_rows: 16,
            ..prefill_cell()
        };
        let candidates =
            enumerate_variants(&TuneOp::AttnDecode(s), &rules(), &saber_rules(), &symbols());
        assert_eq!(
            kernels(&candidates),
            ["saber_attn_decode_uniform16_h256_gqa6_fp8kv_nw6"]
        );
        assert_eq!(candidates.variants[0].descs[1].plugin, "saber_attn");
    }

    /// **NEGATIVE CONTROL for the index-law check**: the I18 bug, rebuilt.
    ///
    /// `saber tune` filled `seqused_k` with `max_ctx` for every attention shape through one buffer
    /// builder. Right at decode, where a sequence really does have a context band of keys
    /// resident; wrong at prefill, where `forward::index_table` gives a chunk `done_before = 0`
    /// and `seqused_k` is the chunk's own length. It timed 2048 queries against 8192 keys — 3.2x
    /// the cell the plan runs — and every gate in this repo passed it, because the record is keyed
    /// on the SHAPE, the shape was right, and it was the INPUT that was wrong.
    ///
    /// The control is the pair: the same builder at the same shape with the right `seqused_k` must
    /// PASS, and with `max_ctx` must be refused **naming the buffer**. One without the other
    /// proves nothing — a check that refuses everything would satisfy the second half alone.
    #[test]
    fn the_prefill_seqused_k_bug_is_refused_and_the_fix_is_not() {
        let s = prefill_cell();
        let op = TuneOp::AttnPrefill(s);

        // The cell the engine runs: `seqused_k` is the chunk's own length.
        index_law_report(&op, &attn_buffers(&s, s.q_len)).expect("the shipped inputs are the law");

        // The cell four caches measured.
        let err = index_law_report(&op, &attn_buffers(&s, s.max_ctx))
            .expect_err("`seqused_k = max_ctx` at prefill must be refused");
        assert!(err.contains("seqused_k"), "{err}");
        assert!(err.contains("attn_prefill"), "{err}");

        // And the decode cell, where `max_ctx` is the right answer and the chunk length is not:
        // the same law, the other way round, which is what makes this a law and not a constant.
        let d = AttnShape {
            q_len: 5,
            live_rows: 5,
            ..prefill_cell()
        };
        let dop = TuneOp::AttnDecode(d);
        index_law_report(&dop, &attn_buffers(&d, d.max_ctx)).expect("decode reads the whole band");
        index_law_report(&dop, &attn_buffers(&d, d.q_len))
            .expect_err("a decode step that declared only its own rows resident is not the step");
    }

    /// A buffer the step has a law for that the case says nothing about is refused, rather than
    /// being filled with whatever the sweep felt like.
    ///
    /// This is the half that makes the check a guard on FUTURE cases: the `seqused_k` mistake was
    /// made once, and the next one will be made in a buffer nobody is looking at.
    #[test]
    fn an_undeclared_index_law_buffer_is_refused() {
        let s = prefill_cell();
        let op = TuneOp::AttnPrefill(s);
        let mut specs = attn_buffers(&s, s.q_len);
        // `naccept` has a law in `forward::index_table`; the attention case declares none.
        specs.push(spec("naccept", "x".to_string(), 4, Fill::ConstI32(1)));
        let err = index_law_report(&op, &specs).expect_err("an undeclared law buffer is refused");
        assert!(err.contains("naccept"), "{err}");
        assert!(err.contains("declare"), "{err}");
    }

    /// The GDN decode case binds `cache_idx` to the SAME allocation as `sidx`, read at stride
    /// `ci_stride = maxq`. The strided read has to be the step's own per-sequence `cache_idx`
    /// table, and this is where that is checked rather than taken on trust — a separate
    /// allocation of `1, 2, 3, …` would give sequence 1 slot 2 where the step gives it slot
    /// `1 + maxq`, and nothing about the timing would look wrong.
    #[test]
    fn the_strided_cache_idx_view_is_the_steps_own_table() {
        let maxq = 5u32;
        let s = GdnShape {
            n_seqs: 4,
            h: 24,
            hg: 8,
            head_k: 128,
            head_v: 128,
            conv_width: 4,
            maxq,
            o_rows: 4 * maxq,
            xpitch: 1,
            cs_seq: 1,
            cs_dim: 1,
            cs_tok: 1,
            ci_stride: i64::from(maxq),
            ab_stride: 1,
            st_slot: 1,
            st_head: 1,
            sidx_stride: i64::from(maxq),
            scale_bits: 0,
            softplus_thr_bits: 0,
            has_naccept: true,
            has_bias: true,
            state_f16: false,
        };
        let op = TuneOp::GdnFused(s);
        let notes = check_index_laws(&op).expect("the GDN decode cell states the step's facts");
        // `naccept` is the one declared divergence here, and it says why.
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].buffer, "naccept");
        assert!(notes[0].why.contains("naccept"), "{}", notes[0].why);

        // The control on the instrument: a SEPARATE `cache_idx` table of `1, 2, 3, 4` — what a
        // reader who did not know about `ci_stride` would write — gives sequence 1 slot 2 where
        // the step gives it slot `1 + maxq`, and must be refused.
        let mut specs = gdn_decode_buffers(&s);
        for sp in &mut specs {
            if sp.name == "cache_idx" {
                *sp = spec(
                    "cache_idx",
                    "ci".to_string(),
                    u64::from(s.n_seqs) * 4,
                    Fill::SlotIndex,
                );
            }
        }
        let err = index_law_report(&op, &specs)
            .expect_err("a per-sequence `cache_idx` is not the strided view the descriptor binds");
        assert!(err.contains("cache_idx"), "{err}");
    }

    /// **The `qk_norm_rope` cell binds the STEP's rope table and the STEP's positions**, and the
    /// cell it replaced is the control.
    ///
    /// I19 left `cos_sin` and `pos` declared rather than derived, because `QkShape` carries no
    /// context band: the cell could only say "a table of `rows` rows, positions `0..rows`" where
    /// a `b1s5c8192` step binds 8192 rope rows and feeds positions `8187..8191`. The band is in
    /// the key now (`BucketKey::MaxCtx`) and in [`QkCell`], and both buffers are derived.
    ///
    /// The control is the old cell, and it is the right control: it differs from the case in the
    /// way the case differed from the truth (a band of `m`, positions from 0), and it must be
    /// REFUSED. Without it this test would pass against a checker that derives nothing.
    #[test]
    fn the_qk_cell_binds_the_bands_rope_table_and_the_steps_positions() {
        let shape = QkShape {
            m: 5,
            nq: 12,
            nkv: 2,
            head_dim: 256,
            qkv_n: 2 * 12 * 256 + 2 * 2 * 256,
        };
        let cell = QkCell {
            shape,
            num_seqs: 1,
            q_len: 5,
            max_ctx: 8192,
            phase: Phase::Decode,
        };
        assert_eq!(cell.max_pos(), 8192);
        assert_eq!(cell.done_before(), 8187);

        let op = TuneOp::GlueQkNormRope(cell);
        let notes = check_index_laws(&op).expect("the qk cell states the step's facts");
        assert!(
            notes.is_empty(),
            "`cos_sin` and `pos` are DERIVED now, not declared: {notes:?}"
        );

        // The table is the band's, in bytes, not the chunk's: 8192 positions x 64 rotary dims
        // in bf16 = 1 MiB, against the 640 B the pre-I20 cell sized from `m` alone.
        let specs = buffers_for(&op, false);
        let cos_sin = specs
            .iter()
            .find(|b| b.name == "cos_sin")
            .expect("the rope table");
        assert_eq!(
            cos_sin.bytes,
            8192 * u64::from(ROTARY_DIM) * 2,
            "the rope table is sized by the BAND"
        );
        assert_eq!(cos_sin.bytes, 1024 * 1024);

        // CONTROL: the cell I19 declared, reproduced exactly. `max_ctx: 0` is a cell with no
        // band -- which is all a shape without the field could say -- so `max_pos` falls
        // through to `.max(rows)` and the buffers are byte for byte the pre-I20 ones:
        // `cos_sin` sized `m * ROTARY_DIM * 2` and `pos` ascending from 0.
        let old = QkCell {
            shape,
            num_seqs: 1,
            q_len: 5,
            max_ctx: 0,
            phase: Phase::Prefill,
        };
        assert_eq!(old.max_pos(), shape.m);
        let old_specs = buffers_for(&TuneOp::GlueQkNormRope(old), false);
        assert_eq!(
            old_specs
                .iter()
                .find(|b| b.name == "cos_sin")
                .expect("the rope table")
                .bytes,
            640,
            "the pre-I20 cell sized the table from `m` alone: 5 x 64 x 2"
        );
        // Those buffers against THIS step's law: both must be refused, one at a time so the
        // failure names the buffer rather than whichever comes first.
        for name in ["cos_sin", "pos"] {
            let mut specs = buffers_for(&op, false);
            for sp in &mut specs {
                if sp.name == name {
                    *sp = old_specs
                        .iter()
                        .find(|b| b.name == name)
                        .expect("the old spec")
                        .clone();
                }
            }
            let err = index_law_report(&op, &specs).expect_err(
                "the pre-I20 cell's buffers are not what the step feeds and must be refused",
            );
            assert!(err.contains(name), "{err}");
        }
    }

    /// A prefill chunk is the first thing in its sequence, so its positions ascend from 0 and its
    /// rope table is still the BAND's — the two halves that make this a law rather than a pair of
    /// constants. Same shape of claim as `the_prefill_seqused_k_bug_is_refused_and_the_fix_is_not`.
    #[test]
    fn a_prefill_qk_cell_starts_at_zero_and_still_gets_the_bands_table() {
        let shape = QkShape {
            m: 2048,
            nq: 12,
            nkv: 2,
            head_dim: 256,
            qkv_n: 2 * 12 * 256 + 2 * 2 * 256,
        };
        let cell = QkCell {
            shape,
            num_seqs: 1,
            q_len: 2048,
            max_ctx: 8192,
            phase: Phase::Prefill,
        };
        assert_eq!(cell.done_before(), 0);
        assert_eq!(cell.max_pos(), 8192);
        let op = TuneOp::GlueQkNormRope(cell);
        assert!(check_index_laws(&op)
            .expect("the prefill qk cell states the step's facts")
            .is_empty());

        // And a DECODE cell of the same rows would place them at the top of the band, which is
        // the difference the phase makes.
        let decode = QkCell {
            phase: Phase::Decode,
            num_seqs: 1,
            q_len: 2048,
            ..cell
        };
        assert_eq!(decode.done_before(), 6144);
    }

    /// The prefill GDN trio states one sequence of `t` tokens, and the day a bucket batches two
    /// chunks into one step (N7) its `cu` of `0, t, 2t` stops being the step's prefix and is
    /// refused here instead of measured.
    #[test]
    fn a_batched_prefill_chunk_is_refused_until_its_cu_is_the_steps() {
        let mk = |n_seqs: u32| GdnPrefillShape {
            n_seqs,
            t: 2048,
            h: 24,
            hg: 8,
            head_k: 128,
            head_v: 128,
            conv_width: 4,
            chunk: 64,
            xpitch: 1,
            cs_seq: 1,
            cs_dim: 1,
            cs_tok: 1,
            ci_stride: 1,
            ab_stride: 1,
            scale_bits: 0,
            softplus_thr_bits: 0,
            has_bias: true,
            has_init: false,
            state_f16: false,
        };
        // Today's prefill buckets: one chunk, one sequence.
        check_index_laws(&TuneOp::GdnConvPrep(mk(1))).expect("one sequence is the step's prefix");
        check_index_laws(&TuneOp::GdnChunkScan(mk(1))).expect("and the scan reads the same tables");
        // Two, where `Fill::CuSeq { step: t }` would declare 2048 tokens for EACH of them.
        let err = check_index_laws(&TuneOp::GdnConvPrep(mk(2)))
            .expect_err("a two-sequence prefill chunk's `cu` is not `0, t, 2t`");
        assert!(format!("{err}").contains("cu"), "{err}");
    }
}
