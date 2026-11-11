//! The DFlash2 drafter plugin: launch rules for the speculative-decoding path.
//!
//! Prod's drafter is a five-layer block-diffusion model whose linear algebra is ONE kernel shape
//! launched 21 times per decode step — the reference's Triton `_gemm_a8w8_blockscale_preshuffle_kernel`,
//! 1567 µs, 7.9% of the step (the census in `docs/LEDGER.md`, workstream F). Saber had no kernel
//! for it at all, so `hip/dflash_gemm.hip` is not a port: there is no HIP source to slice, and it
//! is written from the Triton kernel's semantics and gated bit-exact against its output, the way
//! `attn_glue` was written from prod's Triton and ATen.
//!
//! What is stated here is the shape envelope, out of `rules/dflash_gfx1201.toml`: the tile the
//! code object is compiled for, and the refusal for anything outside it. The kernel handles every
//! M in 1..=`bm` with one grid — that is why the trace shows a single `GRID_MN` per shape — but a
//! larger M, a partial N tile or a K that is not a whole number of scale groups is a different
//! kernel, not a slower launch, so `resolve` says so by name.

use serde::Deserialize;

use crate::abi;
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

const PROD_RULES: &str = include_str!("../rules/dflash_gfx1201.toml");

/// The committed table, verbatim, for the plan's policy digest.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

/// The two code objects these symbols live in. Ownership is stated PER ROW in the rules rather
/// than derived from a file stem, because the drafter's kernels do not share one: the GEMM is its
/// own object and the head plus its rerank are another, and a stem test would be a second, drifting
/// opinion about the same fact.
const PLUGINS: [&str; 12] = [
    "dflash_qk",
    "dflash_gemm",
    "dflash_head",
    "dflash_bgemm",
    "dflash_selector",
    "dflash_conv",
    "dflash_attn",
    "dflash_kv",
    "dflash_topk",
    "dflash_mlp",
    "dflash_walk",
    "dflash_ctx",
];

/// Geometry knobs (`[knobs]`), every one of them read off prod's own Triton specialisation.
#[derive(Debug, Clone, Deserialize)]
pub struct Knobs {
    /// `BLOCK_SIZE_M`: rows per tile, and the largest M one launch can serve.
    pub bm: u32,
    /// `BLOCK_SIZE_N`: columns per workgroup. The grid is `n / bn`.
    pub bn: u32,
    /// `BLOCK_SIZE_K`: one k tile.
    pub bk: u32,
    /// Threads per workgroup: four waves, and the wave index picks a wave's n-fragments.
    pub block: u32,
    /// `GROUP_N` == `GROUP_K`, the blockscale group.
    pub group: u32,
    /// The largest N a single launch may be asked for.
    pub max_n: u32,
    /// The largest M the M-TILED instantiation serves. The one above it is the WMMA tile.
    pub max_m: u32,
    /// SABER-AUTHORED (CMP27K, an ARM): the drafter's `mlp.gate_up` at
    /// `gateup_m64..=64` rows (0 = off; the CLI's bare flag = `GATEUP_M64_MIN_ROWS`, the floor
    /// `GATEUP_M64_FLOOR_ROWS`) takes the one-weight-pass 64-row tile
    /// (`hip/dflash_gemm_m64.hip`: four m-fragments on each B fragment, the same k order and the same
    /// scale-then-accumulate, 0 bytes differing) instead of four 16-row workgroups each reading the
    /// whole slab. Never in the committed table (`serde(default)` = off): `saber serve
    /// --engine-drafter-gateup-m64` sets it on the loaded rules.
    #[serde(default)]
    pub gateup_m64: u32,
}

/// The draft head's geometry (`[head]`), every number read off prod's own module and traced launch.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadKnobs {
    /// Engine-selected full W8 scoring on a one-request proposal.
    #[serde(skip)]
    pub full_vocab_b1: bool,
    /// `BLOCK_M`: the tile height, and the row count of every buffer the head writes.
    pub bm: u32,
    /// `BLOCK_N`: columns per workgroup, and the width of one block-max entry. The grid is `n/bn`.
    pub bn: u32,
    /// The weight-quantisation group along K, and the kernel's k tile.
    pub group: u32,
    /// `KCAND`: candidates emitted per block.
    pub kcand: u32,
    /// Threads per workgroup: two waves.
    pub block: u32,
    /// `BLOCK_K` in the rerank.
    pub rerank_block_k: u32,
    /// Threads per workgroup in the rerank: four waves.
    pub rerank_block: u32,
    /// The largest candidate count a single rerank launch may be asked for.
    pub max_r: u32,
    /// The SECOND band's `BLOCK_M`: how many rows one workgroup's accumulator covers when the
    /// caller has more than `bm` of them. Prod carries 16 / 32 / 64 as different instantiations and
    /// this is the one Saber compiles beside the traced band, because it is where the weight stream
    /// stops being read twice: a 32-row band reads each of the head's 159 MiB of int2 weight bytes
    /// once for thirty-two rows, where a y-axis row slice reads it once per 16-row tile.
    pub bm2: u32,
    /// The largest M the M-TILED head serves: `bm` above is the WMMA tile and this is how many of
    /// them one launch may put on the grid's y axis, times the tile.
    pub max_m: u32,
    /// `SABER_DRAFT_RERANK`, and therefore the `k` of `bm.topk(RERANK)` — the selection that
    /// picks which block maxima the exact rerank scores. It is a COMPILED-IN constant of the
    /// gather pass (its LDS array is `k` entries), so a different rerank width is a different
    /// instantiation and `resolve` says so rather than launching this one wider.
    pub topk_k: u32,
    /// Threads per workgroup in both passes of that selection; one workgroup owns one row.
    pub topk_block: u32,
    /// Waves per workgroup in `xs`, the head's per-128-group column sum. ATen gives one WAVE the
    /// whole group (`block_width = 32`, `block_height = 16`, no cross-wave stage), so this is how
    /// many of those waves Saber packs into one workgroup and it changes no rounding.
    pub group_sum_waves: u32,
    /// Threads per workgroup in the row pad. A copy, so occupancy and only occupancy.
    pub pad_block: u32,
    /// 64-column blocks per workgroup of the traced head and its 32-row band (ws-h43). `1` is the
    /// shipped band (two waves, every wave reading `x` for itself); `2` is the `_lx` instantiation,
    /// four waves over two adjacent blocks with each quarter's `x` slice staged ONCE in LDS -- the
    /// same per-column arithmetic, gated byte-identical, 0.81x at M = 32 / 64 and 0.95x at M = 16.
    /// The row slice (`draft_head_int2_mt`) stays one block a workgroup whatever this says.
    #[serde(default = "one_block")]
    pub wg_blocks: u32,
    /// SABER-AUTHORED (CMP27Z, an ARM): the int2 coarse head at a whole number of 64-row tiles
    /// (the b5..b8 drafter round's `m_pad` 64) takes the one-weight-pass 64-row twin
    /// (`hip/dflash_head_m64.hip`: four m-fragments on each decoded B fragment, the band's
    /// k order and accumulator expressions, 0 bytes differing in y / bmax / bidx) instead of the
    /// 32-row band on a y extent of `m / 32`. Never in the committed table (`serde(default)` =
    /// off): `--engine-step-arms dh64` sets it on the loaded rules.
    #[serde(default)]
    pub m64: bool,
}

fn one_block() -> u32 {
    1
}

/// The bf16 projections' geometry (`[bgemm]`), which is the REFERENCE's: one wave per 16x16 tile.
#[derive(Debug, Clone, Deserialize)]
pub struct BgemmKnobs {
    /// The WMMA M tile, and the largest M one launch serves.
    pub m: u32,
    /// The SECOND band's rows per wave: two WMMA m-fragments against ONE weight fragment, which is
    /// the tune for the wide N. The projection is weight-bandwidth-bound (104.9 MB at N = 10240,
    /// K = 5120), so what a band buys is one weight pass instead of one per 16-row tile.
    pub m2: u32,
    /// One wave's output columns.
    pub n: u32,
    /// Threads per workgroup.
    pub block: u32,
    /// K must be a whole number of WMMA k-steps.
    pub k_multiple: u32,
    /// The largest N a single launch may be asked for.
    pub max_n: u32,
    /// The largest M the M-TILED instantiation serves. The one above it is the WMMA tile.
    pub max_m: u32,
    /// The largest N the TARGET's bf16 `lm_head` may ask for ([`resolve_head_bgemm`], workstream
    /// N2): its own envelope, because `max_n` above is where a bit-exactness gate against
    /// hipBLASLt stopped and the head is not claimed bit-exact against anything (every output
    /// column is an independent K-ascending dot product; the accumulation-order class is stated,
    /// not gated).
    pub head_max_n: u32,
    /// The fp8 head's WIDE band (workstream H33): rows per wave of the one-pass band the head
    /// takes for `m2 < M <= head_band`, i.e. how many WMMA m-fragments one wave holds against ONE
    /// expanded weight fragment. The head is 0.636 GB per rank and the 32-row band reads it once
    /// per 32 rows, so b7s5 / b8s5 (35 / 40 rows) read it twice; this band reads it once. Above
    /// `head_band` the head keeps the 32-row band (two 48-row passes cost more than two 32-row
    /// ones). A COMPILED-IN row count: the rules' `bgemm_w8_head_wide` symbol carries it
    /// (`_m{head_band}n16_`) and the check refuses a mismatch.
    pub head_band: u32,
}

/// The target's bf16 `lm_head` on the drafter's bf16 GEMM (workstream N2): the same three symbols
/// [`DflashOp::bgemm_for`] picks between, at the head's own N envelope (`[bgemm] head_max_n`)
/// instead of the drafter projections' gated `max_n`.
///
/// # Errors
/// Everything [`resolve`] refuses, with `head_max_n` as the N bound.
pub fn resolve_head_bgemm(m: u32, n: u32, k: u32, rules: &DflashRules) -> Result<LaunchDescriptor> {
    if n > rules.bgemm.head_max_n {
        return Err(Error::Rule(format!(
            "bf16 lm_head: N={n} exceeds [bgemm] head_max_n {}",
            rules.bgemm.head_max_n
        )));
    }
    let mut widened = rules.clone();
    widened.bgemm.max_n = rules.bgemm.head_max_n;
    resolve(
        DflashOp::bgemm_for(m, rules),
        DflashShape::new(m, n, k),
        &widened,
    )
}

/// The target's **fp8-WEIGHT** `lm_head` (workstream N4, `W8A16`): the checkpoint's own e4m3 matrix
/// and its per-channel f32 scale, expanded in register to the bf16 head's bytes and then the bf16
/// bands' arithmetic unchanged: the 16-row symbol at `M <= [bgemm] m`, the 32-row band to
/// `[bgemm] m2`, the one-pass WIDE band of `[bgemm] head_band` rows above that (H33: one weight
/// read where the 32-row band reads it twice), and the 32-row band again past `head_band`; the same
/// `head_max_n` envelope, and ONE more kernarg (`ws`) between the weight and the output.
///
/// # Errors
/// Everything [`resolve`] refuses, with `head_max_n` as the N bound.
pub fn resolve_head_w8(m: u32, n: u32, k: u32, rules: &DflashRules) -> Result<LaunchDescriptor> {
    if n > rules.bgemm.head_max_n {
        return Err(Error::Rule(format!(
            "fp8 lm_head: N={n} exceeds [bgemm] head_max_n {}",
            rules.bgemm.head_max_n
        )));
    }
    let mut widened = rules.clone();
    widened.bgemm.max_n = rules.bgemm.head_max_n;
    let op = if m <= rules.bgemm.m {
        DflashOp::BgemmW8Head
    } else if m <= rules.bgemm.m2 || m > rules.bgemm.head_band {
        DflashOp::BgemmW8HeadM32
    } else {
        DflashOp::BgemmW8HeadWide
    };
    resolve(op, DflashShape::new(m, n, k), &widened)
}

/// **CMP27L ARM**: [`resolve_head_w8`] with the 32-row band (`BgemmW8HeadM32`: the 16-byte
/// nontemporal body, one row tile) for `M <= [bgemm] m` (the 16-row band's rows); every other M is
/// exactly [`resolve_head_w8`]. Same kernargs; 0 bytes differing (CMP27L probe, both ranks).
///
/// # Errors
/// Everything [`resolve_head_w8`] refuses.
pub fn resolve_head_w8_b1(m: u32, n: u32, k: u32, rules: &DflashRules) -> Result<LaunchDescriptor> {
    if m > rules.bgemm.m {
        return resolve_head_w8(m, n, k, rules);
    }
    if n > rules.bgemm.head_max_n {
        return Err(Error::Rule(format!(
            "fp8 lm_head: N={n} exceeds [bgemm] head_max_n {}",
            rules.bgemm.head_max_n
        )));
    }
    let mut widened = rules.clone();
    widened.bgemm.max_n = rules.bgemm.head_max_n;
    resolve(
        DflashOp::BgemmW8HeadM32,
        DflashShape::new(m, n, k),
        &widened,
    )
}

/// CMP27F: the one-pass 64-row fp8 head's code object and routed instantiation (4 waves a
/// workgroup, 64 output columns, A staged in LDS 128 k at a time).
pub const HEAD_M64_PLUGIN: &str = "saber_head_w8_m64";
/// The routed symbol (the 4-wave, 128-k-chunk instantiation: CMP27F h4, 1,630 vs 2,252 us at 64 rows).
pub const HEAD_M64_SYMBOL: &str = "saber_head_w8_m64_lds_nw4_kc128";
/// The smallest M the one-pass tile takes: above the wide band (`head_band` 48), whose one
/// pass is faster at <= 48 rows.
pub const HEAD_M64_MIN_ROWS: u32 = 49;

/// [`resolve_head_w8`] with the CMP27F one-pass 64-row tile for `HEAD_M64_MIN_ROWS <= M <= 64`
/// (K a multiple of its 128-k chunk): the same kernargs as the band it replaces (the 32-row band's
/// signature, `abi::SIG_DFLASH_BGEMM_W8`), a 4-wave workgroup per 64 output columns, one row
/// tile. Every other M is exactly [`resolve_head_w8`].
///
/// # Errors
/// Everything [`resolve_head_w8`] refuses.
pub fn resolve_head_w8_m64(
    m: u32,
    n: u32,
    k: u32,
    rules: &DflashRules,
) -> Result<LaunchDescriptor> {
    let mut d = resolve_head_w8(m, n, k, rules)?;
    if (HEAD_M64_MIN_ROWS..=64).contains(&m)
        && k % 128 == 0
        && n % 16 == 0
        && m > rules.bgemm.head_band
    {
        d.op = "bgemm_w8_head_m64".into();
        d.plugin = HEAD_M64_PLUGIN.into();
        d.kernel = HEAD_M64_SYMBOL.into();
        d.grid = [(n / 16).div_ceil(4), 1, 1];
        d.block = [128, 1, 1];
    }
    Ok(d)
}

/// CMP27K: the drafter gate_up's one-weight-pass 64-row tile's code object and routed symbol
/// (8 waves, one n-fragment x four m-fragments a wave; 224 VGPRs, no spill).
pub const GATEUP_M64_PLUGIN: &str = "dflash_gemm_m64";
/// The routed symbol (CMP27K p1: 311.7 -> 159.7 us a layer at M 64 on card 1, IC-cold, bit-exact).
pub const GATEUP_M64_SYMBOL: &str = "dflash_gemm_a8w8_bs128_m64n128k128_nw8";
/// The smallest M the one-pass tile takes: above two 16-row tiles (at 17..32 rows the m-tiled
/// kernel's two workgroups already fill the card; the brief's cell is the b8 33..64).
pub const GATEUP_M64_MIN_ROWS: u32 = 33;
/// The smallest M the arm may be asked to take (`--engine-drafter-gateup-m64 17`): at 17..32 rows
/// the one pass is also faster (CMP27K p1: 176.7 -> 151.9 us at M 32, 0 bytes at M 24 / 32); at
/// M <= 16 it is not (147.6 vs 149.2 at M 8), so a smaller value is clamped up to this.
pub const GATEUP_M64_FLOOR_ROWS: u32 = 17;

impl Knobs {
    /// The smallest M the one-pass gate_up tile takes under this table (`u32::MAX` = off).
    #[must_use]
    pub fn gateup_m64_min(&self) -> u32 {
        if self.gateup_m64 == 0 {
            u32::MAX
        } else {
            self.gateup_m64.max(GATEUP_M64_FLOOR_ROWS)
        }
    }
}
/// Threads a workgroup of the routed symbol (`__launch_bounds__(256)`).
pub const GATEUP_M64_BLOCK: u32 = 256;

/// The blockscale GEMM [`resolve`] would bind at `m` ([`DflashOp::blockscale_for`]), with the
/// CMP27K one-pass 64-row tile for `GATEUP_M64_MIN_ROWS <= M <= 64` when the rules carry the
/// `gateup_m64` knob: the same kernargs (`abi::SIG_DFLASH_GEMM`'s eight), `n / bn` workgroups of
/// 256 threads, one row tile. Every other M, and every M without the knob, is exactly [`resolve`].
/// The drafter's plan asks this for `mlp.gate_up` only (o_proj / down / qkv stay on the m-tiled
/// kernel: CMP27J, their four m-tiles are what fills the card).
///
/// # Errors
/// Everything [`resolve`] refuses.
pub fn resolve_blockscale_gateup(
    m: u32,
    n: u32,
    k: u32,
    rules: &DflashRules,
) -> Result<LaunchDescriptor> {
    let mut d = resolve(
        DflashOp::blockscale_for(m, rules),
        DflashShape::new(m, n, k),
        rules,
    )?;
    if (rules.knobs.gateup_m64_min()..=64).contains(&m) {
        d.op = "gemm_a8w8_blockscale_m64".into();
        d.plugin = GATEUP_M64_PLUGIN.into();
        d.kernel = GATEUP_M64_SYMBOL.into();
        d.grid = [n / rules.knobs.bn, 1, 1];
        d.block = [GATEUP_M64_BLOCK, 1, 1];
    }
    Ok(d)
}

/// CMP27Z: the int2 head's one-pass 64-row twin -- its own code object, so `dflash_head` keeps its
/// bytes.
pub const HEAD_INT2_M64_PLUGIN: &str = "dflash_head_m64";
/// CMP27Z: the one-pass 64-row twin's symbol (`__launch_bounds__(128)`, the `_lx` tile's).
pub const HEAD_INT2_M64_SYMBOL: &str = "dflash_draft_head_int2_m64n64x2g128_lx";
/// CMP27Z: rows one workgroup of the twin owns.
pub const HEAD_INT2_M64_ROWS: u32 = 64;

/// The int2 head [`resolve`] would bind at `m` ([`DflashOp::draft_head_for`]), with the CMP27Z
/// one-pass 64-row twin when the rules carry the head's `m64` knob and the route is the 32-row
/// `_lx` band at a whole number of 64-row tiles: the same kernargs, the same x extent and block,
/// the y extent `m / 64`. Every other M, and every M without the knob, is exactly [`resolve`].
///
/// # Errors
/// Everything [`resolve`] refuses.
pub fn resolve_draft_head_int2(
    m: u32,
    n: u32,
    k: u32,
    rules: &DflashRules,
) -> Result<LaunchDescriptor> {
    let op = DflashOp::draft_head_for(m, rules);
    let mut d = resolve(op, DflashShape::new(m, n, k), rules)?;
    if rules.head.m64
        && op == DflashOp::DraftHeadInt2M32
        && d.kernel.ends_with("_lx")
        && m % HEAD_INT2_M64_ROWS == 0
    {
        d.op = "draft_head_int2_m64".into();
        d.plugin = HEAD_INT2_M64_PLUGIN.into();
        d.kernel = HEAD_INT2_M64_SYMBOL.into();
        d.grid[1] = m / HEAD_INT2_M64_ROWS;
    }
    Ok(d)
}

/// The selector's geometry (`[selector]`). Both numbers are the drafter's own config and the
/// kernel is compiled for them.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectorKnobs {
    /// `selector_rank`: 16 WMMA k-steps.
    pub rank: u32,
    /// `selector_top_k`: exactly one 16x16 fragment.
    pub top_k: u32,
    /// Threads per workgroup: one wave scores one block position.
    pub block: u32,
}

/// The grouped convolution's geometry (`[conv]`).
#[derive(Debug, Clone, Deserialize)]
pub struct ConvKnobs {
    /// `conv_kernel_size`.
    pub taps: u32,
    /// `conv_group_size`: the width one coefficient covers.
    pub group_size: u32,
    /// The activation-quant group, `GroupShape(1, 128)`.
    pub quant_group: u32,
    /// bf16 elements per thread; 32 lanes cover one quant group.
    pub vec: u32,
    /// Threads per workgroup.
    pub block: u32,
}

/// `TILE_SIZE_PREFILL`, which is the tile the **2D** path takes -- and the drafter takes the 2D
/// path at every step, because `use_3d` is false whenever `max_seqlen_q > 1` and the drafter's
/// block is 8 query rows.
///
/// It is 32 at BOTH KV widths: `_get_tile_size` returns 32 for `is_prefill=True` before it looks
/// at `element_size`, and the `16 if element_size >= 2 else 32` line is the decode arm
/// (`triton_unified_attention.py:784-799`). Reading that arm is how `attn_bf16kv` came to bind a
/// `_t16` instantiation; see [`DflashRules::validate`].
pub const TILE_2D: u32 = 32;

/// The head width `dflash_qk_norm_rope_h128` is compiled for, and the wave that reduces it.
///
/// Not a knob and not a loop bound: the reference's reduction hands each of 32 lanes four
/// CONTIGUOUS elements and folds them with a DPP ladder, so 32 x 4 = 128 is the kernel. A
/// different head width is a different reduction shape and would be a different symbol.
pub const QK_HEAD_DIM: u32 = 128;
/// One wave per (token, head) -- see [`QK_HEAD_DIM`].
pub const QK_WAVE: u32 = 32;

/// The attention's geometry (`[attn]`), all of it prod's.
#[derive(Debug, Clone, Deserialize)]
pub struct AttnKnobs {
    /// `BLOCK_M`.
    pub bm: u32,
    /// `BLOCK_Q` = `bm / gqa`.
    pub bq: u32,
    /// `num_queries_per_kv`.
    pub gqa: u32,
    /// `HEAD_SIZE`.
    pub head: u32,
    /// `TILE_SIZE` at two bytes per KV element. [`TILE_2D`], like `tile_fp8`: the width does not
    /// choose the tile on the 2D path, which is the only path the drafter takes.
    pub tile_bf16: u32,
    /// ... and at one byte, where the reference requires at least 32 even on the decode arm.
    pub tile_fp8: u32,
    /// The sliding window, applied on BOTH sides because the drafter is non-causal.
    pub window: u32,
    /// Threads per workgroup; wave `w` owns kv head `w`.
    pub block: u32,
    /// The KV cache's block size.
    pub block_size: u32,
}

/// The KV append's and the selector top-k's geometry (`[kv]`).
#[derive(Debug, Clone, Deserialize)]
pub struct KvKnobs {
    /// Threads per workgroup in the append; one workgroup per token row.
    pub block: u32,
    /// Threads per workgroup in the top-k; one workgroup per row.
    pub topk_block: u32,
    /// The top-k's `k`, which is ALSO the per-thread capacity — that equality is what makes one
    /// pass exact, and it is why `k` is a compiled-in constant rather than an argument.
    pub topk_k: u32,
}

/// The MLP producers' geometry (`[mlp]`).
#[derive(Debug, Clone, Deserialize)]
pub struct MlpKnobs {
    /// The activation-quant group.
    pub quant_group: u32,
    /// bf16 elements per thread in the quant.
    pub vec: u32,
    /// Threads per workgroup in the quant.
    pub block: u32,
    /// Threads per workgroup in the fused conv + add + RMS; one workgroup owns a whole row.
    pub row_block: u32,
}

/// The path walk's geometry (`[walk]`).
#[derive(Debug, Clone, Deserialize)]
pub struct WalkKnobs {
    /// Threads per workgroup: one wave, as the reference launches it.
    pub block: u32,
}

/// The context-KV store's geometry (`[ctx]`) — the two ops the drafter calls as `upstream::` C++
/// kernels rather than through a module.
#[derive(Debug, Clone, Deserialize)]
pub struct CtxKnobs {
    /// `vectorize_read_with_alignment<W>`'s width in the norm the reference launches: how many
    /// contiguous elements one thread squares into its accumulator before the block reduction
    /// sees it. A MEASURED number — the code object carries instantiations at 1, 2, 4, 8 and 16
    /// and only the launched one reproduces prod's bytes, so this knob is the gate's answer and
    /// the symbol below has to spell it.
    pub vec: u32,
    /// The norm block below [`CtxKnobs::narrow_rows`] rows: the launcher gives a row all the
    /// threads its vectors can use, up to this.
    pub wide_block: u32,
    /// … and at or above it, where the grid is wide enough to fill the device on its own. This
    /// is not only an occupancy choice: `narrow_block * vec` is 2048 against a 5120-wide row, so
    /// at this block a thread CHAINS three vectors into one accumulator and the sum associates
    /// differently. Measured off prod's own launches, not chosen.
    pub narrow_block: u32,
    /// The row count at which the launcher switches between the two. Probed at h = 5120 over
    /// rows 1..16384: 128 rows take the wide block and 256 take the narrow one.
    pub narrow_rows: u32,
    /// `dim3 block(std::min(num_heads * rot_dim / 2, 512))`, the reference's rope launch.
    pub rope_max_block: u32,
    /// Threads per workgroup in the layout copy. It computes nothing, so this is occupancy and
    /// only occupancy — the one knob in this table that no rounding depends on.
    pub layout_block: u32,
    /// The drafter's `rms_norm_eps`, from its `config.json`. One number for the whole model:
    /// `_build_fused_kv_buffers` asserts every layer's `q_norm.variance_epsilon` is equal before
    /// it fuses them, so there is exactly one and it belongs in the table rather than in a
    /// caller.
    pub eps: f32,
}

/// One row: an op name and the symbol it launches.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRow {
    /// The op id `resolve` is asked with.
    pub op: String,
    /// The code object this symbol lives in.
    pub plugin: String,
    /// The symbol in the code object.
    pub symbol: String,
}

/// The drafter plugin's rules.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRules {
    /// Schema version; 1 for this build.
    pub schema_version: u32,
    /// Human name.
    pub name: String,
    /// The GEMM's geometry.
    pub knobs: Knobs,
    /// The draft head's geometry.
    pub head: HeadKnobs,
    /// The bf16 projections' geometry.
    pub bgemm: BgemmKnobs,
    /// The selector's geometry.
    pub selector: SelectorKnobs,
    /// The grouped convolution's geometry.
    pub conv: ConvKnobs,
    /// The attention's geometry.
    pub attn: AttnKnobs,
    /// The KV append's and top-k's geometry.
    pub kv: KvKnobs,
    /// The MLP producers' geometry.
    pub mlp: MlpKnobs,
    /// The path walk's geometry.
    pub walk: WalkKnobs,
    /// The context-KV store's geometry.
    pub ctx: CtxKnobs,
    /// The op → symbol table.
    #[serde(default)]
    pub kernel: Vec<DflashRow>,
}

impl DflashRules {
    /// The committed rules.
    ///
    /// # Errors
    /// The committed file failing to parse or to pass [`DflashRules::check`].
    pub fn prod_dflash() -> Result<Self> {
        Self::from_toml(PROD_RULES)
    }

    /// Parse and check.
    ///
    /// # Errors
    /// A schema this build does not speak, a knob the code object contradicts, a duplicated op,
    /// or a symbol with no declared signature in [`crate::abi`].
    pub fn from_toml(text: &str) -> Result<Self> {
        let r: DflashRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "dflash rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        let kn = &self.knobs;
        // The tile the ONE committed instantiation is compiled for. These are not preferences: the
        // symbol name carries them (`_bs128_m16n128k128`), the fragment decomposition is written
        // for `bn / 16` n-fragments across `block / 32` waves, and the scalar-b_scale arm is only
        // correct while `group >= bn` and `group >= bk`.
        if (kn.bm, kn.bn, kn.bk, kn.block, kn.group) != (16, 128, 128, 128, 128) {
            return Err(Error::Rule(format!(
                "dflash rules state tile (bm {}, bn {}, bk {}, block {}, group {}) but the only \
                 instantiation is (16, 128, 128, 128, 128)",
                kn.bm, kn.bn, kn.bk, kn.block, kn.group
            )));
        }
        if kn.max_n == 0 {
            return Err(Error::Rule("dflash knob `max_n` must be > 0".into()));
        }
        if kn.max_m < kn.bm {
            return Err(Error::Rule(format!(
                "dflash knobs state max_m {} below the tile's {} rows: the M-tiled instantiation \
                 serves a SUPERSET of what one tile does",
                kn.max_m, kn.bm
            )));
        }
        let hd = &self.head;
        // The head's one committed instantiation, the same way: the symbol name carries the tile
        // (`_m16n64g128`), the emit is written for two 32-lane halves of a 64-thread workgroup, and
        // the rerank's reduction tree is written for four waves over a 512-wide partial vector.
        if (hd.bm, hd.bn, hd.group, hd.kcand, hd.block) != (16, 64, 128, 8, 64)
            || (hd.rerank_block_k, hd.rerank_block) != (512, 128)
        {
            return Err(Error::Rule(format!(
                "dflash head rules state (bm {}, bn {}, group {}, kcand {}, block {}, rerank {}x{}) \
                 but the only instantiations are (16, 64, 128, 8, 64) and (512, 128)",
                hd.bm, hd.bn, hd.group, hd.kcand, hd.block, hd.rerank_block_k, hd.rerank_block
            )));
        }
        if hd.max_r == 0 {
            return Err(Error::Rule("dflash head knob `max_r` must be > 0".into()));
        }
        // The `_lx` instantiation is compiled for exactly two blocks a workgroup (`DH_LXB`), and
        // its symbols say so; one block is the shipped band. Nothing else exists.
        if hd.wg_blocks != 1 && hd.wg_blocks != 2 {
            return Err(Error::Rule(format!(
                "dflash head knob wg_blocks {} but the compiled instantiations are 1 (the shipped \
                 band) and 2 (`_n64x2g128_lx`, x staged once in LDS)",
                hd.wg_blocks
            )));
        }
        // The second band is a whole number of tiles, `max_m` is a whole number of BANDS, and the
        // compiled instantiation is 32: a different band is a different kernel, not an argument.
        if hd.bm2 != 2 * hd.bm {
            return Err(Error::Rule(format!(
                "dflash head rules state bm2 {} against a tile of {}: the only second band compiled                  is {} rows (two WMMA m-fragments in one workgroup's accumulator)",
                hd.bm2,
                hd.bm,
                2 * hd.bm
            )));
        }
        if hd.max_m % hd.bm2 != 0 {
            return Err(Error::Rule(format!(
                "dflash head rules state max_m {} against a band of {}: the widest M a plan may ask                  for has to be a whole number of bands as well as of tiles",
                hd.max_m, hd.bm2
            )));
        }
        // The grid follows `wg_blocks` and the symbol has to be the instantiation compiled for it:
        // a 64-thread band launched on a two-block grid would score half the vocabulary.
        for op in [DflashOp::DraftHeadInt2, DflashOp::DraftHeadInt2M32] {
            if let Ok(row) = self.row(op) {
                if row.symbol.ends_with("_lx") != (hd.wg_blocks == 2) {
                    return Err(Error::Rule(format!(
                        "dflash head: `{}` is bound to `{}` but wg_blocks is {}; the two-block \
                         grid needs the `_lx` instantiation and the one-block grid the shipped band",
                        op.as_str(),
                        row.symbol,
                        hd.wg_blocks
                    )));
                }
            }
        }
        if hd.max_m < hd.bm || hd.max_m % hd.bm != 0 {
            return Err(Error::Rule(format!(
                "dflash head rules state max_m {} against a tile of {}: the M-tiled head serves a \
                 whole number of tiles and nothing else -- the emit writes all {} rows of its own",
                hd.max_m, hd.bm, hd.bm
            )));
        }
        let bg = &self.bgemm;
        if bg.max_m < bg.m {
            return Err(Error::Rule(format!(
                "dflash bgemm rules state max_m {} below the tile's {} rows: the M-tiled \
                 instantiation serves a SUPERSET of what one tile does",
                bg.max_m, bg.m
            )));
        }
        if bg.m2 != 2 * bg.m {
            return Err(Error::Rule(format!(
                "dflash bgemm rules state m2 {} against a tile of {}: the only second band compiled                  is {} rows (two WMMA m-fragments per wave against one weight fragment)",
                bg.m2,
                bg.m,
                2 * bg.m
            )));
        }
        if bg.head_band <= bg.m2 || bg.head_band % bg.m != 0 || bg.head_band > 4 * bg.m {
            return Err(Error::Rule(format!(
                "dflash bgemm rules state head_band {} against a tile of {} and a band of {}: the \
                 wide head band is a whole number of tiles above the 32-row band and at most four \
                 m-fragments in one wave",
                bg.head_band, bg.m, bg.m2
            )));
        }
        let tag = format!("_m{}n16_", bg.head_band);
        if let Some(row) = self
            .kernel
            .iter()
            .find(|r| r.op == DflashOp::BgemmW8HeadWide.as_str())
        {
            if !row.symbol.contains(&tag) {
                return Err(Error::Rule(format!(
                    "dflash rules route `bgemm_w8_head_wide` to `{}` but state head_band {}: the \
                     band's row count is compiled into the symbol (`{tag}`)",
                    row.symbol, bg.head_band
                )));
            }
        }
        if (bg.m, bg.n, bg.block, bg.k_multiple) != (16, 16, 128, 16) || bg.max_n == 0 {
            return Err(Error::Rule(format!(
                "dflash bgemm rules state (m {}, n {}, block {}, k_multiple {}) but the only \
                 instantiation is (16, 16, 128, 16)",
                bg.m, bg.n, bg.block, bg.k_multiple
            )));
        }
        let sel = &self.selector;
        if (sel.rank, sel.top_k, sel.block) != (256, 16, 32) {
            return Err(Error::Rule(format!(
                "dflash selector rules state (rank {}, top_k {}, block {}) but the only \
                 instantiation is (256, 16, 32)",
                sel.rank, sel.top_k, sel.block
            )));
        }
        let at = &self.attn;
        if (at.bm, at.bq, at.gqa, at.head, at.block, at.block_size) != (16, 4, 4, 128, 128, 16)
            || (at.tile_bf16, at.tile_fp8) != (TILE_2D, TILE_2D)
            || at.window == 0
        {
            return Err(Error::Rule(format!(
                "dflash attn rules state (bm {}, bq {}, gqa {}, head {}, block {}, block_size {}, \
                 tiles {}/{}) but the only instantiations are (16, 4, 4, 128, 128, 16) and \
                 {TILE_2D}/{TILE_2D} -- the 2D path takes TILE_SIZE_PREFILL at BOTH KV widths",
                at.bm, at.bq, at.gqa, at.head, at.block, at.block_size, at.tile_bf16, at.tile_fp8
            )));
        }
        // THE TILE AND THE SYMBOL ARE ONE FACT STATED TWICE, so they are checked against each
        // other -- the same check the ctx norm's `vec` gets below, and for a sharper reason. The
        // attention's KV width picks the instantiation, and until 2026-09-21 the bf16 route bound
        // `..._bf16kv_t16` because the tile had been read off `_get_tile_size(..., is_prefill=
        // False)`. Nothing runs the bf16 route today (the drafter's cache is e4m3), so no gate
        // could have caught it; F24 measured the same substitution on the fp8 side at 14,306
        // differing bytes of 983,040, which is what the wrong tile costs. `_t16` stays COMPILED as
        // that control and is refused here BY NAME, so a future bf16 drafter cache cannot bind it.
        for op in [
            DflashOp::AttnBf16Kv,
            DflashOp::AttnFp8Kv,
            DflashOp::AttnBf16KvNs,
            DflashOp::AttnFp8KvNs,
        ] {
            let row = self.row(op)?;
            let want = format!("_t{}", at.tile_bf16);
            // The batched walk is the same tile with one more kernarg, so its symbol ends in the
            // tile plus `_ns` and nothing else -- a `_t16_ns` would be refused here exactly as a
            // `_t16` is (ws-s11).
            let ns = format!("{want}_ns");
            let ok = if matches!(op, DflashOp::AttnBf16KvNs | DflashOp::AttnFp8KvNs) {
                row.symbol.ends_with(&ns)
            } else {
                row.symbol.ends_with(&want)
            };
            if !ok {
                return Err(Error::Rule(format!(
                    "dflash attn rules bind `{}` to `{}`, whose tile is not the 2D path's \
                     TILE_SIZE_PREFILL ({}): the symbol must end in `{want}`. The 2D path is the \
                     only one the drafter takes (`max_seqlen_q > 1` at every step), and a tile of \
                     16 there is a different online softmax, not a slower one",
                    row.op, row.symbol, at.tile_bf16
                )));
            }
        }
        let kv = &self.kv;
        if (kv.block, kv.topk_block, kv.topk_k) != (256, 256, 16) {
            return Err(Error::Rule(format!(
                "dflash kv rules state (block {}, topk_block {}, topk_k {}) but the only \
                 instantiations are (256, 256, 16)",
                kv.block, kv.topk_block, kv.topk_k
            )));
        }
        let ml = &self.mlp;
        if (ml.quant_group, ml.vec, ml.block, ml.row_block) != (128, 4, 128, 256) {
            return Err(Error::Rule(format!(
                "dflash mlp rules state (quant_group {}, vec {}, block {}, row_block {}) but the \
                 only instantiations are (128, 4, 128, 256)",
                ml.quant_group, ml.vec, ml.block, ml.row_block
            )));
        }
        if self.walk.block != 32 {
            return Err(Error::Rule(format!(
                "dflash walk rules state block {} but the reference launches ONE wave",
                self.walk.block
            )));
        }
        let cx = &self.ctx;
        if !matches!(cx.vec, 1 | 2 | 4 | 8 | 16) {
            return Err(Error::Rule(format!(
                "dflash ctx rules state vec {} but the reference instantiates \
                 vectorize_read_with_alignment at 1, 2, 4, 8 and 16",
                cx.vec
            )));
        }
        for (name, b) in [
            ("wide_block", cx.wide_block),
            ("narrow_block", cx.narrow_block),
        ] {
            if b == 0 || b > 1024 || b % 32 != 0 {
                return Err(Error::Rule(format!(
                    "dflash ctx rules state {name} {b} but the norm's block reduction is written \
                     for whole waves up to 1024 threads"
                )));
            }
        }
        if cx.narrow_block > cx.wide_block || cx.narrow_rows == 0 {
            return Err(Error::Rule(format!(
                "dflash ctx rules state narrow_block {} / wide_block {} / narrow_rows {}, which \
                 is not a launcher that NARROWS as the grid grows",
                cx.narrow_block, cx.wide_block, cx.narrow_rows
            )));
        }
        if cx.rope_max_block == 0 || cx.rope_max_block > 512 {
            return Err(Error::Rule(format!(
                "dflash ctx rules state rope_max_block {} but the reference caps its rope launch \
                 at 512",
                cx.rope_max_block
            )));
        }
        if !(cx.eps > 0.0 && cx.eps < 1.0) {
            return Err(Error::Rule(format!(
                "dflash ctx rules state eps {} which is not a norm epsilon",
                cx.eps
            )));
        }
        // The measured width and the symbol are ONE fact stated twice, so they are checked
        // against each other: a `vec` edited without the row (or the reverse) would launch an
        // instantiation whose accumulation order is not the one the gate measured, and nothing
        // downstream would notice.
        if let Some(row) = self
            .kernel
            .iter()
            .find(|r| r.op == DflashOp::CtxRmsNorm.as_str())
        {
            let want = format!("dflash_rms_norm_grouped_w{}", cx.vec);
            if row.symbol != want {
                return Err(Error::Rule(format!(
                    "dflash ctx rules state vec {} but `{}` names `{}` rather than `{want}`",
                    cx.vec, row.op, row.symbol
                )));
            }
        }
        let cv = &self.conv;
        if (cv.taps, cv.group_size, cv.quant_group, cv.vec, cv.block) != (2, 16, 128, 4, 128) {
            return Err(Error::Rule(format!(
                "dflash conv rules state (taps {}, group_size {}, quant_group {}, vec {}, \
                 block {}) but the only instantiation is (2, 16, 128, 4, 128)",
                cv.taps, cv.group_size, cv.quant_group, cv.vec, cv.block
            )));
        }
        for (i, row) in self.kernel.iter().enumerate() {
            if self.kernel[..i].iter().any(|r| r.op == row.op) {
                return Err(Error::Rule(format!(
                    "dflash rules state op `{}` twice",
                    row.op
                )));
            }
            if !PLUGINS.contains(&row.plugin.as_str()) {
                return Err(Error::Rule(format!(
                    "dflash rules put op `{}` in plugin `{}`, which is not one of {PLUGINS:?}",
                    row.op, row.plugin
                )));
            }
            abi::signature_for(&row.symbol)
                .ok_or_else(|| Error::Rule(format!("{}: no declared signature", row.symbol)))?;
        }
        for op in [
            DflashOp::GemmBlockscale,
            DflashOp::GemmBlockscaleMt,
            DflashOp::DraftHeadInt2,
            DflashOp::DraftHeadInt2Mt,
            DflashOp::DraftHeadInt2M32,
            DflashOp::RerankExact,
            DflashOp::RerankExactBf16,
            DflashOp::RerankExactW8,
            DflashOp::BgemmBf16,
            DflashOp::BgemmBf16Mt,
            DflashOp::BgemmBf16M32,
            DflashOp::BgemmW8Head,
            DflashOp::BgemmW8HeadM32,
            DflashOp::BgemmW8HeadWide,
            DflashOp::SelectorScore,
            DflashOp::GroupedConv,
            DflashOp::GroupedConvQuant,
            DflashOp::AttnBf16Kv,
            DflashOp::AttnFp8Kv,
            DflashOp::AttnBf16KvNs,
            DflashOp::AttnFp8KvNs,
            DflashOp::KvAppendBf16,
            DflashOp::KvAppendFp8,
            DflashOp::SelectorTopk,
            DflashOp::SiluQuant,
            DflashOp::ConvAddRmsOpen,
            DflashOp::ConvAddRmsClose,
            DflashOp::RmsNorm,
            DflashOp::SelectorWalk,
            DflashOp::CtxRmsNorm,
            DflashOp::CtxRope,
            DflashOp::CtxKvLayout,
            DflashOp::HeadTopkThresh,
            DflashOp::HeadTopkGather,
            DflashOp::ActQuant,
            DflashOp::ActQuantRecip,
            DflashOp::HeadGroupSum,
            DflashOp::HeadRowPad,
            DflashOp::HeadCandGather,
            DflashOp::RowGather,
        ] {
            if !self.kernel.iter().any(|r| r.op == op.as_str()) {
                return Err(Error::Rule(format!(
                    "dflash rules describe no op `{}`; the drafter has no other way to do it",
                    op.as_str()
                )));
            }
        }
        Ok(())
    }

    /// Every symbol these rules name.
    #[must_use]
    pub fn symbols(&self) -> Vec<&str> {
        self.kernel.iter().map(|r| r.symbol.as_str()).collect()
    }

    /// Does this code object belong to the drafter plugin family?
    #[must_use]
    pub fn owns(&self, plugin: &str) -> bool {
        self.kernel.iter().any(|r| r.plugin == plugin)
    }

    /// The rows one code object contributes, in rules order.
    #[must_use]
    pub fn rows_for(&self, plugin: &str) -> Vec<&DflashRow> {
        self.kernel.iter().filter(|r| r.plugin == plugin).collect()
    }

    /// The symbol these rules bind an op to. The instantiation is what carries the element
    /// WIDTHS (`tests/capture_widths.rs`) as well as the tile, so a caller that wants to ask a
    /// question about the kernel and not about a shape needs the name without a `resolve`.
    pub fn symbol_for(&self, op: DflashOp) -> Result<&str> {
        Ok(self.row(op)?.symbol.as_str())
    }

    fn row(&self, op: DflashOp) -> Result<&DflashRow> {
        self.kernel
            .iter()
            .find(|r| r.op == op.as_str())
            .ok_or_else(|| {
                Error::Rule(format!("the dflash rules describe no op `{}`", op.as_str()))
            })
    }
}

/// The drafter ops Saber implements as code objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashOp {
    /// The fp8 W8A8 GEMM with 128x128 block scales and a preshuffled B.
    GemmBlockscale,
    /// The same GEMM **M-tiled**: one workgroup per 16 rows on the grid's y axis.
    ///
    /// A second op rather than a knob on the first, for the reason every second symbol in this
    /// file is one: the shape the whole proposal was gated at keeps the code object it was gated
    /// with. [`DflashOp::GemmBlockscale`] refuses an M above the tile BY NAME and names this one.
    /// The arithmetic per output element is identical — the same k tiles in the same order, the
    /// same scale applied to the same tile dot product, the same single round to bf16 — so the two
    /// are bit-identical wherever both are legal, which is the gate. **21 of the proposal's
    /// launches** are this op.
    GemmBlockscaleMt,
    /// the reference's int2 draft head: coarse logits plus the top `KCAND` of each 64-column block.
    DraftHeadInt2,
    /// The same head **M-tiled**: one workgroup per 16 rows on the grid's y axis.
    ///
    /// Prod carries M = 32 and M = 64 as different instantiations, and that is a LAUNCH CONFIG
    /// fact (`_cfg_for` picks num_warps 2 / 4 / 8 by band) rather than an arithmetic one: per
    /// output element the reduction is the same ten groups of each of the same four quarters, in
    /// the same order, whatever `BLOCK_M` is. So a row slice computes prod's own numbers at every
    /// padded M — which is the gate, from both ends: bit-identical to the m16 symbol on each
    /// 16-row slice, and against prod's Triton head at BLOCK_M 32 and 64.
    ///
    /// An M that is not a whole number of tiles is refused rather than padded: the emit writes all
    /// sixteen rows of its tile, and prod allocates `y`/`bmax`/`bidx` at the padded height that
    /// `_pow2_at_least` gives, which is always a multiple of 16.
    DraftHeadInt2Mt,
    /// The same head at a **32-row band**: one workgroup per 32 rows, and the one the rules pick
    /// above the traced tile.
    ///
    /// The row slice above makes a batched shape LEGAL and is the wrong launch for it: at a y
    /// extent of 2 the two m tiles of one 64-column block are two different workgroups, so the
    /// head's 159 MiB int2 weight is read once per tile exactly as two separate launches read it —
    /// **868.54 µs against 743.20 for two m16 launches** (F33's per-launch table), where prod's own
    /// Triton at `BLOCK_M = 32` is 592.64 because ONE workgroup holds a `[32, 64]` accumulator and
    /// reads each weight byte once for thirty-two rows. This op is that band.
    ///
    /// It is the same arithmetic: `acc[mf][f][e]` accumulates `dot * S - XS * ZS` over the same ten
    /// groups of each of the same four quarters in the same order, and `tl.dot` inside a group is
    /// the same eight ascending WMMA k-steps — the second m-fragment only rides the same decoded
    /// weight. So it is bit-identical to the 16-row symbol per 16-row slice, which is the gate.
    ///
    /// It serves `M / 32` bands, i.e. 32 / 64 / 128 / 256 — every M `_pow2_at_least` produces above
    /// the traced one. An M that is a whole number of 16-row tiles but NOT of 32 is
    /// [`DflashOp::DraftHeadInt2Mt`]'s, and the refusal names it as exactly that fallback.
    DraftHeadInt2M32,
    /// The exact fp8 logit for each of `r` candidates per draft row, which is what makes the
    /// 2-bit head exact. `_rerank_exact`'s `FP8=True` arm: an e4m3 head plus a per-row scale.
    RerankExact,
    /// The same rerank against a **bf16** head — `_rerank_exact`'s `FP8=False` arm, and the one
    /// the DFlash2 serve runs.
    ///
    /// Which arm a serve takes is decided at LOAD, by what `ref_drafthead._head_matrix`
    /// finds bound: `SABER_NVFP4_LMHEAD=bf16` (the unit's setting and the module's own
    /// default) has `ref_nvfp4.RefFp8ChannelToBf16` replace the checkpoint's e4m3
    /// `lm_head` with `bf16(f32(w8) * f32(scale))` before the drafter ever sees it, so the head
    /// has no scale and this arm is selected. It is a separate op rather than a knob because the
    /// two arms read DIFFERENT BYTES — the pack's `head.rerank.w` is one dtype or the other, and
    /// the int2 head is quantised from whichever one is live — so it is a plan fact.
    RerankExactBf16,
    /// The bf16 arm's rerank over the checkpoint's OWN fp8 head (workstream N6): `head.rerank.w8`
    /// (e4m3, the target's `lm_head.w8` bytes) and `head.rerank.ws` (f32 per-row scale) expanded
    /// in register to `bf16(f32(w8) * f32(scale))` — N4's argument: the multiply is exact and the
    /// bf16 round-to-nearest-even is the only rounding, so every weight the dot reads is the bf16
    /// record's element and the output is [`DflashOp::RerankExactBf16`]'s bit for bit. Which of
    /// the two a drafter runs is the PACK's fact (the record kind it carries), like the target's
    /// head: one head record per rank, shared by the target and the drafter.
    RerankExactW8,
    /// The drafter's bf16 projections: `kernel_projection` and `hidden_projection`.
    BgemmBf16,
    /// The same projection **M-tiled**: one workgroup per 16 rows on the grid's y axis.
    ///
    /// A second op rather than a knob on the first, for the reason every second symbol in this
    /// file is one: the shape the whole proposal was gated at keeps the code object it was gated
    /// with. [`DflashOp::BgemmBf16`] refuses an M above the tile BY NAME and names this one; the
    /// caller that has more rows than that asks for this. The arithmetic per output row is
    /// identical — same tile, same k-steps, same single fp32 accumulator — so the two are
    /// bit-identical wherever both are legal, which is the gate.
    BgemmBf16Mt,
    /// The same projection at a **32-row band**: two WMMA m-fragments per wave against ONE weight
    /// fragment, and the one the rules pick above the tile.
    ///
    /// The tune for the wide N, and it is a bandwidth statement rather than a tile preference. One
    /// wave owns one 16x16 output tile and walks all of K (the reference's own parallelism, and the
    /// only one that can be bit-exact), so a launch reads `N * K * 2` bytes of weight: 104.9 MB at
    /// N = 10240, K = 5120, whose floor is 165 µs at the card's measured 635 GB/s. hipBLASLt is AT
    /// that floor (184.56 µs); the y-axis row slice reads the weight ONCE PER 16-ROW TILE, so at
    /// M = 32 its own floor is 330 µs and it measured 334.38 — on the wrong roofline rather than
    /// off its own. This band puts the second m tile in the same wave, so the weight is read once.
    ///
    /// Every output element's arithmetic is the 16-row symbol's — the same k-steps in the same
    /// ascending order into the same single fp32 accumulator — so the declared residual class
    /// against `F.linear` (hipBLASLt's algorithm choice per shape, 3 elements per row at
    /// N = 10240, linear in M with the same constant) is unchanged, and the gate checks it per
    /// 16-row slice. An M that is a whole number of 16-row tiles but not of 32 is
    /// [`DflashOp::BgemmBf16Mt`]'s.
    BgemmBf16M32,
    /// The TARGET's fp8-weight `lm_head` at up to one WMMA tile of rows (workstream N4): the
    /// checkpoint's e4m3 matrix + per-channel f32 scale, expanded in register to the bf16 head's
    /// bytes (`bf16(f32(w8) * scale)`, prod's loader arithmetic) before [`DflashOp::BgemmBf16`]'s
    /// WMMA sequence. Only [`resolve_head_w8`] asks for it.
    BgemmW8Head,
    /// ... and at the 32-row band, [`DflashOp::BgemmBf16M32`]'s arithmetic over the same expansion.
    BgemmW8HeadM32,
    /// ... and at the WIDE band (workstream H33): `[bgemm] head_band` rows per wave, three or four
    /// WMMA m-fragments against ONE expanded weight fragment, so the head's 0.636 GB is read once
    /// for up to sixty-four rows where the 32-row band reads it once per 32. Every output
    /// element's arithmetic is the 16-row symbol's (the same expanded fragment, the same k order,
    /// one fp32 accumulator, one bf16 rounding), gated per 16-row slice.
    BgemmW8HeadWide,
    /// The candidate selector's edge scoring, gathers and all.
    SelectorScore,
    /// The grouped convolution, bf16 in and out.
    GroupedConv,
    /// The same convolution with the per-group e4m3 quant fused into it.
    GroupedConvQuant,
    /// The drafter's attention over a bf16 KV cache.
    AttnBf16Kv,
    /// ... and over an e4m3 one, where the reference's tile is twice as wide.
    AttnFp8Kv,
    /// The BATCHED walk over a bf16 cache: `num_seqs` sequences in one launch (ws-s11).
    ///
    /// A separate op and therefore a separate rules row, because it is a separate SYMBOL with a
    /// separate kernarg list — and because the one-sequence kernel handed a shape with more than
    /// one sequence would serve request 0's attention `num_seqs` times and report success. That
    /// is refused by name in [`resolve`], which is what this op exists to make possible.
    AttnBf16KvNs,
    /// ... and the batched walk over an e4m3 cache, which is the one the drafter runs.
    AttnFp8KvNs,
    /// The KV append into the reference server's flash layout, bf16 cache.
    KvAppendBf16,
    /// ... and e4m3 cache.
    KvAppendFp8,
    /// The selector's candidate top-k over a vocabulary shard.
    SelectorTopk,
    /// The int2 draft head's own selection, `bm.topk(RERANK)`: pass one, the radix select that
    /// finds the row's k-th largest `(value, ~index)` composite.
    HeadTopkThresh,
    /// ... and pass two, which gathers the k elements at or above that composite and ranks them.
    HeadTopkGather,
    /// The blockscale GEMM's own activation quant: bf16 -> e4m3 with per-128-group f32 scales.
    /// The EAGER one — `absmax / 448` as a real ATen division. One launch a step, in front of
    /// `aux_hidden_projection`, because `combine_hidden_states` calls `model.fc` OUTSIDE the
    /// drafter's compiled region.
    ActQuant,
    /// ... and the COMPILED one, `absmax * (1/448)`, in front of each layer's `attn.o_proj`.
    /// Five launches a step. Two ops and not one, because the drafter's six `act_quant` launches
    /// are TWO PROGRAMS at one source line and `DflashRules` binds one symbol per op: inductor
    /// strength-reduces the constant divisor inside the region (`tmp8 = tmp6 *
    /// 0.002232142857142857` in `_0`) and ATen's `div` by a python float does not. F26 measured
    /// the division against the EAGER `quant.*.q/s` and bound it everywhere; F30 read the
    /// generated code and found the other half. The two differ in 35 of 32,768 quantised bytes
    /// and 130 of 1,024 scales at the o_proj shape, and the CHAIN is what gates this one — the
    /// capture carries no compiled-region quant to compare it against.
    ActQuantRecip,
    /// `silu(gate) * up` and the per-group e4m3 quant its consumer needs.
    SiluQuant,
    /// The convolution's `finish` half, the residual add and the RMS norm, in one kernel — as
    /// inductor already fuses them — at the ATTENTION half, which OPENS the fp32 residual prod
    /// carries across a layer. `resid` is bf16 (the previous layer's, through the eager attention
    /// seam) and `res_out` is fp32.
    ConvAddRmsOpen,
    /// The same kernel at the MLP half, which CLOSES that residual: fp32 in, bf16 out, because
    /// prod's compiled region ends there and hands the next attention a bf16 tensor. Two ops and
    /// not one shape field, because the pair differs in two ELEMENT WIDTHS and `abi::SigArg::elem`
    /// is where an element width is declared.
    ConvAddRmsClose,
    /// The RMS norm on its own: layer 0's `input_layernorm`, the only one in the drafter with no
    /// residual to add and no convolution in front of it to fuse into.
    RmsNorm,
    /// The selector's path walk: the draft block, one request per program.
    SelectorWalk,
    /// The context-KV store's RMS norm: the C++ `upstream::rms_norm_kernel`, with the weight selected
    /// per group of rows. Two launches a step — the hidden norm at one group and the stacked
    /// per-layer K norm at `num_layers`.
    CtxRmsNorm,
    /// The context-KV store's rope: one batched, in-place, K-only launch over every layer's
    /// context rows at once.
    CtxRope,
    /// The context-KV store's layout copy: `permute(2, 1, 0, 3, 4).contiguous()`, K and V
    /// separated and transposed to layer-major. A permutation, not arithmetic.
    CtxKvLayout,
    /// The int2 head's `xs`: `x.reshape(M, ng, GROUP).float().sum(-1)`, the per-128-group fp32
    /// column sum the head reads as a kernarg. A launch prod makes that the plan treated as an
    /// INPUT until F27 chained the backbone.
    HeadGroupSum,
    /// `_pow2_at_least`'s row pad: `torch.cat([x, x.new_zeros(M - m, k)])`, the head's input
    /// padded up to `tl.dot`'s 16 rows. The second of that pair.
    HeadRowPad,
    /// `bi.gather(1, bm.topk(RERANK).indices)`: the map from the top-80's block-maxima POSITIONS
    /// to the vocabulary COLUMNS the exact rerank scores. The third launch of the head's
    /// selection, and the one F26 did not see because a staged replay seeds the rerank's `idx`
    /// from prod's own.
    HeadCandGather,
    /// `x[idx]` over whole rows: `last_hidden[sample_indices]` and
    /// `input_ids[_anchor_indices]`, the two ATen row gathers `_generate_draft` makes. One op at
    /// two shapes -- a gather has a row WIDTH and no dtype.
    RowGather,
    /// The drafter's QUERY q/k norm + rope, one launch per layer.
    ///
    /// It is a `DflashOp` and not an `attn_glue` band row, and the reason is arithmetic rather
    /// than shape: `attn_glue::qk_norm_rope` implements the TARGET's **GemmaRMSNorm** (`1 + w`)
    /// over ATen's reduction with a rope that does not contract, and the drafter's `q_norm` /
    /// `k_norm` are the reference server's PLAIN `RMSNorm` over a Triton reduction with a rope that does. The
    /// `1 + w` alone is a ~2x error on every element of q and k, which is what F28 found under
    /// `--no-seed`'s red `chain.attn.l0.q`; the contraction alone would make an `attn_glue`
    /// instantiation impossible anyway, because that plugin is compiled `-ffp-contract=off`.
    QkNormRope,
}

impl DflashOp {
    /// **Which bgemm instantiation serves `m` rows**, and it is the rules' own knobs that say.
    ///
    /// F34 measured the three: the 16-row tile is what a one-request proposal asks for, a y-axis
    /// ROW SLICE of it reads the whole weight once per tile (334.40 µs at M = 32, two weight
    /// passes at the card's peak), and the 32-row BAND puts the second m-fragment in the same wave
    /// against the same weight fragment (197.64 µs, one pass, 1.071x hipBLASLt). So there is no M
    /// at which the row slice is the answer for THIS kernel — it clamps a partial last band rather
    /// than predicating it, exactly as the 16-row tile does — and the selection is two cases.
    #[must_use]
    pub fn bgemm_for(m: u32, rules: &DflashRules) -> DflashOp {
        if m <= rules.bgemm.m {
            DflashOp::BgemmBf16
        } else {
            DflashOp::BgemmBf16M32
        }
    }

    /// The same for the blockscale GEMM, which has ONE M-tiled symbol and no band.
    ///
    /// Its A rows are CLAMPED rather than predicated (a masked load on gfx1201 is a full memory
    /// round trip), so a partial last tile is legal and the rows it produces are not stored.
    #[must_use]
    pub fn blockscale_for(m: u32, rules: &DflashRules) -> DflashOp {
        if m <= rules.knobs.bm {
            DflashOp::GemmBlockscale
        } else {
            DflashOp::GemmBlockscaleMt
        }
    }

    /// The attention's symbol for a proposal of `seqs` requests, over an e4m3 cache when
    /// `fp8` (which is what the drafter's own cache is).
    ///
    /// ONE request takes the one-sequence kernel — the program every gate in this tree
    /// byte-compared against prod's own live call — and a batch takes the `num_seqs` walk. The
    /// choice is here rather than at the plan's site for the same reason `bgemm_for` is: a caller
    /// that picked a symbol would be a caller that had learned the rules.
    #[must_use]
    pub fn attn_for(seqs: u32, fp8: bool) -> DflashOp {
        match (seqs > 1, fp8) {
            (false, false) => DflashOp::AttnBf16Kv,
            (false, true) => DflashOp::AttnFp8Kv,
            (true, false) => DflashOp::AttnBf16KvNs,
            (true, true) => DflashOp::AttnFp8KvNs,
        }
    }

    /// The same for the int2 head, which is the one op of the three with THREE bands and a
    /// refusal between them.
    ///
    /// The emit writes all `bm` rows of its own tile, so a partial tile would run past `y` /
    /// `bmax` / `bidx`; the band's is `bm2` rows. Every M this plan asks for is
    /// `_pow2_at_least(max(steps, 16))`, which is 16 or a multiple of 32 and never 48 — so the row
    /// slice is named here for the M that is whole tiles and not whole bands and is unreachable
    /// from the engine's own padding, which is a fact worth having in the selection rather than
    /// in a comment on the caller.
    #[must_use]
    pub fn draft_head_for(m: u32, rules: &DflashRules) -> DflashOp {
        if m <= rules.head.bm {
            DflashOp::DraftHeadInt2
        } else if rules.head.bm2 > 0 && m % rules.head.bm2 == 0 {
            DflashOp::DraftHeadInt2M32
        } else {
            DflashOp::DraftHeadInt2Mt
        }
    }

    /// The op id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            DflashOp::GemmBlockscale => "gemm_a8w8_blockscale",
            DflashOp::GemmBlockscaleMt => "gemm_a8w8_blockscale_mt",
            DflashOp::DraftHeadInt2 => "draft_head_int2",
            DflashOp::DraftHeadInt2Mt => "draft_head_int2_mt",
            DflashOp::DraftHeadInt2M32 => "draft_head_int2_m32",
            DflashOp::RerankExact => "rerank_exact",
            DflashOp::RerankExactBf16 => "rerank_exact_bf16",
            DflashOp::RerankExactW8 => "rerank_exact_w8",
            DflashOp::BgemmBf16 => "bgemm_bf16",
            DflashOp::BgemmBf16Mt => "bgemm_bf16_mt",
            DflashOp::BgemmBf16M32 => "bgemm_bf16_m32",
            DflashOp::BgemmW8Head => "bgemm_w8_head",
            DflashOp::BgemmW8HeadM32 => "bgemm_w8_head_m32",
            DflashOp::BgemmW8HeadWide => "bgemm_w8_head_wide",
            DflashOp::SelectorScore => "selector_score",
            DflashOp::GroupedConv => "grouped_conv",
            DflashOp::GroupedConvQuant => "grouped_conv_quant",
            DflashOp::AttnBf16Kv => "attn_bf16kv",
            DflashOp::AttnFp8Kv => "attn_fp8kv",
            DflashOp::AttnBf16KvNs => "attn_bf16kv_ns",
            DflashOp::AttnFp8KvNs => "attn_fp8kv_ns",
            DflashOp::KvAppendBf16 => "kv_append_bf16",
            DflashOp::KvAppendFp8 => "kv_append_fp8",
            DflashOp::SelectorTopk => "selector_topk",
            DflashOp::HeadTopkThresh => "head_topk_thresh",
            DflashOp::HeadTopkGather => "head_topk_gather",
            DflashOp::ActQuant => "act_quant",
            DflashOp::ActQuantRecip => "act_quant_recip",
            DflashOp::SiluQuant => "silu_quant",
            DflashOp::ConvAddRmsOpen => "conv_add_rms_open",
            DflashOp::ConvAddRmsClose => "conv_add_rms_close",
            DflashOp::RmsNorm => "rms_norm",
            DflashOp::SelectorWalk => "selector_walk",
            DflashOp::CtxRmsNorm => "ctx_rms_norm",
            DflashOp::CtxRope => "ctx_rope",
            DflashOp::CtxKvLayout => "ctx_kv_layout",
            DflashOp::HeadGroupSum => "head_group_sum",
            DflashOp::HeadRowPad => "head_row_pad",
            DflashOp::HeadCandGather => "head_cand_gather",
            DflashOp::RowGather => "row_gather",
            DflashOp::QkNormRope => "qk_norm_rope_draft",
        }
    }
}

/// What the drafter's kernels are asked for: `m` rows against an `n`-wide weight over a `k`-long
/// reduction, plus `r` candidates where the op has candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashShape {
    /// Rows. For the GEMM, 1 + `num_speculative_tokens` per sequence (8 in the traced step); for
    /// the head, the draft rows padded up to the tile (16).
    pub m: u32,
    /// Output columns: the GEMM's N, or the head's vocabulary shard.
    pub n: u32,
    /// The reduction axis.
    pub k: u32,
    /// Candidates per row for the rerank, the top-k for the selector, the block size for the
    /// convolution; 0 where the op has no such axis.
    pub r: u32,
    /// Which half of the convolution's coefficient buffer to read; 0 elsewhere.
    pub side: u32,
    /// The KV cache's block size, for the attention and the KV append. A RUNTIME value, not a
    /// knob: prod's drafter cache is `[1212, 1648, 4, 128]` -- 1648 tokens per block, because
    /// `--mamba-cache-mode align` sizes the attention page to the GDN one. The first draft put the
    /// rules' `block_size = 16` in the descriptor and would have indexed the block table wrongly
    /// against prod's own cache.
    pub block_size: u32,
    /// The BLOCK TABLE's row stride, in entries: the second dimension of the
    /// `[num_seqs, max_blocks]` table the scheduler wrote. For the attention, and zero elsewhere.
    ///
    /// It is here for the same reason `block_size` is and for a sharper one. It is a runtime fact
    /// -- the scheduler pads every sequence's page list to the widest one in the step -- so there
    /// is no rules knob it could come from and no derivation from `(m, n, k, r)` that is not a
    /// guess. Until 2026-09-21 the shape table simply did not name it, and `resolve`'s
    /// `None => ArgValue::Buffer` arm then handed the kernel an eight-byte ADDRESS in a four-byte
    /// `I32` slot, which shifts every kernarg after it. `saber draft-step --proposal` refused the
    /// five attention launches by name rather than dispatch that; the field is what un-refuses
    /// them, and `resolve` refuses a zero instead, because a block table with no columns is a
    /// caller that has not been told the stride rather than a legal launch.
    pub bt_stride: u32,
    /// The KV append's SOURCE row strides, in elements: `key` first, then `value`. Zero means
    /// "contiguous", which `resolve` turns into `nkv * head_dim`.
    ///
    /// They are a property of the BINDING and not of `(m, n, k)`, which is exactly why they have
    /// to be named. The reference server's own `reshape_and_cache_flash` takes both, because the caller hands it
    /// views: the drafter's k is the rope's contiguous output and its **v is a view of the
    /// `[q | k | v]` projection**, stride 6144 where the block is 1024. A kernel that assumed
    /// contiguity read v row 0 correctly and then walked into the next row's q -- every appended
    /// v byte wrong, and invisible to a staged replay, which seeds the attention from prod's own
    /// cache (F29).
    pub kv_src_stride: (u32, u32),
    /// SEQUENCES in this launch, for the attention's batched walk; 0 or 1 is one sequence.
    ///
    /// It is a shape field and not a knob for the same reason `bt_stride` is — it is a property of
    /// the STEP — and it is separate from `m` because `m` is the launch's total rows: the walk is
    /// `seq = qblock * BLOCK_Q / (m / seqs)`, which is exact only because every request in a
    /// proposal contributes the same `block` rows. A shape whose rows do not divide by its
    /// sequences is refused rather than rounded (ws-s11).
    pub seqs: u32,
}

impl DflashShape {
    /// A GEMM or head shape, which has no candidate count.
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        DflashShape {
            m,
            n,
            k,
            r: 0,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// An attention or KV-append shape: `m` query rows over `n` query heads and `r` kv heads, a
    /// sequence of `k`, in a cache of `block_size`-token blocks whose block table carries
    /// `bt_stride` entries per sequence.
    ///
    /// `bt_stride` has no default and no knob: see the field. A caller that does not know the
    /// step's block table cannot launch this kernel, and saying so here is cheaper than a shifted
    /// kernarg.
    #[must_use]
    pub const fn attn(
        m: u32,
        q_heads: u32,
        seq_len: u32,
        kv_heads: u32,
        block_size: u32,
        bt_stride: u32,
    ) -> Self {
        Self::attn_seqs(m, q_heads, seq_len, kv_heads, block_size, bt_stride, 1)
    }

    /// The same over a BATCH: `m` is every sequence's rows together and `seqs` is how many
    /// sequences they are (ws-s11). `seqs = 1` is [`DflashShape::attn`].
    ///
    /// The per-sequence row count is `m / seqs` and it is what the kernel's walk divides by, so
    /// `m % seqs != 0` is a refusal in [`resolve`] rather than a rounding: a proposal whose
    /// requests did not all contribute the same rows is a RAGGED batch, which this kernel's walk
    /// is exact only because it is not.
    #[must_use]
    pub const fn attn_seqs(
        m: u32,
        q_heads: u32,
        seq_len: u32,
        kv_heads: u32,
        block_size: u32,
        bt_stride: u32,
        seqs: u32,
    ) -> Self {
        DflashShape {
            m,
            n: q_heads,
            k: seq_len,
            r: kv_heads,
            side: 0,
            block_size,
            bt_stride,
            kv_src_stride: (0, 0),
            seqs,
        }
    }

    /// A rerank shape: `r` candidates per row, scored against `n` rows of exact weight.
    #[must_use]
    pub const fn rerank(m: u32, n: u32, k: u32, r: u32) -> Self {
        DflashShape {
            m,
            n,
            k,
            r,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// A selector shape. The fields carry the selector's own names: `m` is the number of block
    /// POSITIONS (batch x block length), `n` the block length (which is what makes position 0 of
    /// each sequence take the anchor rather than the previous candidates), `k` the rank and `r`
    /// the top-k.
    #[must_use]
    pub const fn selector(positions: u32, l_len: u32, rank: u32, top_k: u32) -> Self {
        DflashShape {
            m: positions,
            n: l_len,
            k: rank,
            r: top_k,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// A context-store norm: `rows` rows of `h` channels, with the weight row advancing once
    /// every `rows_per_group` rows. The stacked K norm is `[L, num_ctx, nkv, head_dim]` against a
    /// `[L, head_dim]` weight, so its group covers `num_ctx * nkv` rows; the hidden norm is one
    /// group over every row.
    #[must_use]
    pub const fn ctx_norm(rows: u32, h: u32, rows_per_group: u32) -> Self {
        DflashShape {
            m: rows,
            // The reduction IS the row, so `k` is `h` rather than a second axis.
            n: h,
            k: h,
            r: rows_per_group,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// The drafter's QUERY q/k norm + rope: `rows` rows of `nq` q heads and `nkv` k heads, each
    /// `head_dim` wide and FULLY rotated (the drafter's config has no `partial_rotary_factor`).
    ///
    /// The projection's row stride is DERIVED and not passed, because the drafter's qkv is a
    /// plain `[q | k | v]` with no output gate: `(nq + 2 * nkv) * head_dim` is the whole tensor.
    /// That is a law of this checkpoint (`DFlashQwen3Attention.forward` splits exactly
    /// `[q_size, kv_size, kv_size]`), so a caller that has to pass a stride is a caller that has
    /// the wrong tensor.
    #[must_use]
    pub const fn qk_rope(rows: u32, nq: u32, nkv: u32, head_dim: u32) -> Self {
        DflashShape {
            m: rows,
            n: nq,
            k: nkv,
            r: head_dim,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// A context-store rope: `rows` rows of `heads` heads of `head_size`, of which the first
    /// `rot_dim` elements rotate. The row stride is `heads * head_size` and is DERIVED rather
    /// than passed, because this call site's tensor is `all_k_normed.view(L * num_ctx, kv)` — a
    /// view of a contiguous buffer, so the reference's `query_stride` is exactly that product.
    #[must_use]
    pub const fn ctx_rope(rows: u32, heads: u32, head_size: u32, rot_dim: u32) -> Self {
        DflashShape {
            m: rows,
            n: heads,
            k: head_size,
            r: rot_dim,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// The context store's layout copy: `[num_ctx, L, 2, nkv, hd]` in, `[2, L, num_ctx, nkv, hd]`
    /// out. Four extents and no reduction, because the op computes nothing — which is why they
    /// ride in `m`/`n`/`k`/`r` in the order the reference's own `view(...)` writes them rather
    /// than in a shape that pretends one of them is a K.
    #[must_use]
    pub const fn ctx_layout(num_ctx: u32, layers: u32, kv_heads: u32, head_dim: u32) -> Self {
        DflashShape {
            m: num_ctx,
            n: layers,
            k: kv_heads,
            r: head_dim,
            side: 0,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }

    /// A convolution shape: `m` rows of `n` channels in `k` groups, restarting every `r` rows,
    /// reading the `side` half of the coefficient buffer (0 for `prepare`, 1 for `finish`).
    #[must_use]
    pub const fn conv(t_rows: u32, h: u32, groups: u32, block_size: u32, side: u32) -> Self {
        DflashShape {
            m: t_rows,
            n: h,
            k: groups,
            // The CONV's block size is its restart period, not a cache page -- a different number
            // that happens to share a name, which is why it rides in `r` here and `block_size`
            // stays zero.
            r: block_size,
            side,
            block_size: 0,
            bt_stride: 0,
            kv_src_stride: (0, 0),
            seqs: 0,
        }
    }
}

/// The one decision function: (op, shape, rules) in, descriptor out. Pure, no environment.
///
/// # Errors
/// A shape the committed instantiations cannot serve. Every refusal names the knob it failed, and
/// none of them is a fallback: the drafter's kernels are single-tile, so a shape outside the tile
/// is a caller bug and a wider launch would be silently wrong rather than slow.
pub fn resolve(op: DflashOp, shape: DflashShape, rules: &DflashRules) -> Result<LaunchDescriptor> {
    let row = rules.row(op)?;
    let sig = abi::signature_for(&row.symbol)
        .ok_or_else(|| Error::Rule(format!("{}: no declared signature", row.symbol)))?;
    let DflashShape { m, n, k, r, .. } = shape;
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Rule(format!(
            "{}: {m} x {n} x {k} has nothing to do; an empty launch is a caller bug, not a no-op",
            op.as_str()
        )));
    }
    let (grid, block, values): (_, _, Vec<(&str, i64)>) = match op {
        DflashOp::GemmBlockscale | DflashOp::GemmBlockscaleMt => {
            let kn = &rules.knobs;
            let tiled = op == DflashOp::GemmBlockscaleMt;
            if !tiled && m > kn.bm {
                return Err(Error::Rule(format!(
                    "{}: M={m} exceeds the tile's {} rows. The drafter decodes at M = 1 + \
                     num_speculative_tokens; a BATCHED proposal is `max_seqs * block` rows and \
                     asks for `gemm_a8w8_blockscale_mt`, the M-tiled instantiation",
                    op.as_str(),
                    kn.bm
                )));
            }
            if tiled && m > kn.max_m {
                return Err(Error::Rule(format!(
                    "{}: M={m} exceeds max_m {}; a batched proposal is `max_seqs * block` rows and \
                     this is the widest one this plan may ask for",
                    op.as_str(),
                    kn.max_m
                )));
            }
            if n % kn.bn != 0 {
                return Err(Error::Rule(format!(
                    "{}: N={n} is not a multiple of {}. B is preshuffled to [N/16, K*16] and a \
                     partial workgroup would read n-groups past the end of it",
                    op.as_str(),
                    kn.bn
                )));
            }
            if n > kn.max_n {
                return Err(Error::Rule(format!(
                    "{}: N={n} exceeds max_n {}",
                    op.as_str(),
                    kn.max_n
                )));
            }
            if k % kn.group != 0 {
                return Err(Error::Rule(format!(
                    "{}: K={k} is not a whole number of {}-wide scale groups; the k loop applies \
                     one a_scale/b_scale pair per tile and a partial group has no scale to apply",
                    op.as_str(),
                    kn.group
                )));
            }
            // The y extent is the M TILES, and it is 1 at every shape the untiled symbol serves
            // -- which is what makes the two kernels the same launch at M <= 16.
            let rows = if tiled { m.div_ceil(kn.bm) } else { 1 };
            (
                [n / kn.bn, rows, 1],
                [kn.block, 1, 1],
                vec![
                    ("m", i64::from(m)),
                    ("n", i64::from(n)),
                    ("k", i64::from(k)),
                ],
            )
        }
        DflashOp::DraftHeadInt2 | DflashOp::DraftHeadInt2Mt | DflashOp::DraftHeadInt2M32 => {
            let hd = &rules.head;
            let band = op == DflashOp::DraftHeadInt2M32;
            let tiled = band || op == DflashOp::DraftHeadInt2Mt;
            // How many rows ONE workgroup owns: the traced tile for the row slice, the second band
            // for the op above it. The grid's y extent is the shape divided by this, and nothing
            // else about the launch moves.
            let rows_per_wg = if band { hd.bm2 } else { hd.bm };
            // Not `m <= bm`: prod pads the rows up to the tile and allocates y/bmax/bidx at that
            // height, and the kernel writes all of them. A caller that passes a shorter m has
            // buffers the kernel will run past.
            if !tiled && m != hd.bm {
                return Err(Error::Rule(format!(
                    "{}: M={m} but this instantiation writes exactly {} rows. Prod pads the draft \
                     rows to a power of two >= 16 (`_pow2_at_least`) and allocates the outputs at \
                     that height; a taller M is `draft_head_int2_m32`, the {}-row band (one \
                     workgroup, one weight pass for both its m tiles), with `draft_head_int2_mt` \
                     the row-slice fallback for an M that is whole 16-row tiles but not whole bands",
                    op.as_str(),
                    hd.bm,
                    hd.bm2
                )));
            }
            // The M-tiled head takes WHOLE TILES and nothing else, for the same reason the untiled
            // one takes exactly 16: each workgroup's emit writes all sixteen rows of its own tile,
            // so a partial tile would run past the buffers. `_pow2_at_least` never produces one.
            // The BAND says the same thing about its own 32 rows, and names the row slice as what
            // serves the M it refuses.
            if tiled && m % rows_per_wg != 0 {
                return Err(Error::Rule(format!(
                    "{}: M={m} is not a whole number of {}-row tiles; the emit writes all {} rows \
                     of its tile and prod allocates y/bmax/bidx at `_pow2_at_least(m)`, which is \
                     always a multiple of {}{}",
                    op.as_str(),
                    rows_per_wg,
                    rows_per_wg,
                    hd.bm,
                    if band {
                        " -- an M of whole 16-row tiles and not whole bands is `draft_head_int2_mt`"
                    } else {
                        ""
                    }
                )));
            }
            if tiled && m > hd.max_m {
                return Err(Error::Rule(format!(
                    "{}: M={m} exceeds max_m {}; the head scores `max_seqs * (block - 1)` sampled \
                     rows padded up, and this is the widest one this plan may ask for",
                    op.as_str(),
                    hd.max_m
                )));
            }
            if n % hd.bn != 0 {
                return Err(Error::Rule(format!(
                    "{}: N={n} is not a multiple of {}; one workgroup owns one {}-column block and \
                     the emit reduces over exactly that many",
                    op.as_str(),
                    hd.bn,
                    hd.bn
                )));
            }
            // The packing splits K into quarters and quantises in groups, so a quarter has to be a
            // whole number of groups -- prod asserts the same thing as `k % (2 * GROUP)`.
            if k % (4 * hd.group) != 0 {
                return Err(Error::Rule(format!(
                    "{}: K={k} is not a multiple of {}; the quarter-split packing puts K/4 \
                     elements of each plane in a row and each quarter must be whole groups of {}",
                    op.as_str(),
                    4 * hd.group,
                    hd.group
                )));
            }
            // `wg_blocks` 64-column blocks a workgroup (ws-h43) on the traced tile and the band;
            // the kernel stores nothing for a second block past `nblk`, so the x extent rounds up.
            let wgb = if op == DflashOp::DraftHeadInt2Mt {
                1
            } else {
                hd.wg_blocks
            };
            (
                [
                    (n / hd.bn).div_ceil(wgb),
                    if tiled { m / rows_per_wg } else { 1 },
                    1,
                ],
                [hd.block * wgb, 1, 1],
                vec![
                    ("m", i64::from(m)),
                    ("n", i64::from(n)),
                    ("k", i64::from(k)),
                    // Every stride is derived from the shape, never passed in: the packed row is
                    // K/4 bytes, both scale rows are K/group, and the block count is N/bn.
                    ("stride_wq", i64::from(k / 4)),
                    ("stride_s", i64::from(k / hd.group)),
                    ("stride_xs", i64::from(k / hd.group)),
                    ("nblk", i64::from(n / hd.bn)),
                ],
            )
        }
        DflashOp::BgemmBf16
        | DflashOp::BgemmBf16Mt
        | DflashOp::BgemmBf16M32
        | DflashOp::BgemmW8Head
        | DflashOp::BgemmW8HeadM32
        | DflashOp::BgemmW8HeadWide => {
            let bg = &rules.bgemm;
            let band = matches!(op, DflashOp::BgemmBf16M32 | DflashOp::BgemmW8HeadM32);
            let wide = op == DflashOp::BgemmW8HeadWide;
            let tiled = band || wide || op == DflashOp::BgemmBf16Mt;
            // Rows one workgroup's waves own: the WMMA tile for the row slice, two of them for the
            // band, `head_band` for the fp8 head's wide band -- the only difference between the
            // symbols.
            let rows_per_wg = if wide {
                bg.head_band
            } else if band {
                bg.m2
            } else {
                bg.m
            };
            if !tiled && m > bg.m {
                return Err(Error::Rule(format!(
                    "{}: M={m} exceeds the tile's {} rows; the drafter's projections run at its \
                     token count, and a taller M is `bgemm_bf16_m32`, the {}-row band (one weight \
                     pass for two m tiles), with `bgemm_bf16_mt` the row-slice fallback",
                    op.as_str(),
                    bg.m,
                    bg.m2
                )));
            }
            if tiled && m > bg.max_m {
                return Err(Error::Rule(format!(
                    "{}: M={m} exceeds max_m {}; a batched proposal is `max_seqs * block` context \
                     rows and this is the widest one this plan may ask for",
                    op.as_str(),
                    bg.max_m
                )));
            }
            if n % bg.n != 0 || n > bg.max_n {
                return Err(Error::Rule(format!(
                    "{}: N={n} must be a multiple of {} and at most {}; one wave owns one {}-wide \
                     output tile",
                    op.as_str(),
                    bg.n,
                    bg.max_n,
                    bg.n
                )));
            }
            if k % bg.k_multiple != 0 {
                return Err(Error::Rule(format!(
                    "{}: K={k} is not a whole number of {}-element WMMA k-steps",
                    op.as_str(),
                    bg.k_multiple
                )));
            }
            let waves = bg.block / 32;
            // The y extent is the M TILES, and it is 1 at every shape the untiled symbol serves —
            // which is what makes the two kernels the same launch at M <= 16.
            let rows = if tiled { m.div_ceil(rows_per_wg) } else { 1 };
            (
                [(n / bg.n).div_ceil(waves), rows, 1],
                [bg.block, 1, 1],
                vec![
                    ("m", i64::from(m)),
                    ("n", i64::from(n)),
                    ("k", i64::from(k)),
                ],
            )
        }
        DflashOp::SelectorScore => {
            let sl = &rules.selector;
            if k != sl.rank || r != sl.top_k {
                return Err(Error::Rule(format!(
                    "{}: rank {k} x top_k {r}, but this instantiation is compiled for {} x {} \
                     (the rank is its WMMA k-step count and the top-k is its fragment)",
                    op.as_str(),
                    sl.rank,
                    sl.top_k
                )));
            }
            if n == 0 || m % n != 0 {
                return Err(Error::Rule(format!(
                    "{}: {m} positions is not a whole number of blocks of {n}; the anchor is taken \
                     at position 0 of each block and the previous candidates elsewhere",
                    op.as_str()
                )));
            }
            (
                [m, 1, 1],
                [sl.block, 1, 1],
                vec![
                    ("bl", i64::from(m)),
                    ("l_len", i64::from(n)),
                    ("r", i64::from(k)),
                    ("k_top", i64::from(r)),
                ],
            )
        }
        DflashOp::GroupedConv | DflashOp::GroupedConvQuant => {
            let cv = &rules.conv;
            let quant = op == DflashOp::GroupedConvQuant;
            if n % (k * cv.group_size) != 0 || n / cv.group_size != k {
                return Err(Error::Rule(format!(
                    "{}: {n} channels in {k} groups contradicts a group size of {}",
                    op.as_str(),
                    cv.group_size
                )));
            }
            if r == 0 || (r & (r - 1)) != 0 {
                return Err(Error::Rule(format!(
                    "{}: block_size {r} must be a power of two -- prod's `_grouped_conv` takes the \
                     `position & (block_size - 1)` branch, and the modulo branch is a different \
                     kernel",
                    op.as_str()
                )));
            }
            if shape.side > 1 {
                return Err(Error::Rule(format!(
                    "{}: side {} is neither `prepare` (0) nor `finish` (1)",
                    op.as_str(),
                    shape.side
                )));
            }
            if quant && n % cv.quant_group != 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} channels is not a whole number of {}-wide quant groups",
                    op.as_str(),
                    cv.quant_group
                )));
            }
            let waves = cv.block / 32;
            let grid = if quant {
                (m * (n / cv.quant_group)).div_ceil(waves)
            } else {
                (m * n / cv.vec).div_ceil(cv.block)
            };
            (
                [grid, 1, 1],
                [cv.block, 1, 1],
                vec![
                    ("t_rows", i64::from(m)),
                    ("h", i64::from(n)),
                    ("groups", i64::from(k)),
                    ("block_size", i64::from(r)),
                    ("side", i64::from(shape.side)),
                ],
            )
        }
        DflashOp::AttnBf16Kv
        | DflashOp::AttnFp8Kv
        | DflashOp::AttnBf16KvNs
        | DflashOp::AttnFp8KvNs => {
            let at = &rules.attn;
            let batched = matches!(op, DflashOp::AttnBf16KvNs | DflashOp::AttnFp8KvNs);
            let seqs = shape.seqs.max(1);
            // THE ONE-SEQUENCE KERNEL IS REFUSED A BATCH BY NAME (ws-s11). It reads `seq_lens[0]`
            // and indexes the block table with no row stride, so handed a shape of four it would
            // serve REQUEST 0's attention four times -- every dispatch issuing, every launch
            // counted, and three requests drafting a continuation of somebody else's prompt. The
            // kernel's own header has said "this kernel must not be handed it" since F22; this is
            // that sentence as a refusal.
            if !batched && seqs > 1 {
                return Err(Error::Rule(format!(
                    "{}: a shape of {seqs} sequence(s). This instantiation serves ONE -- it reads \
                     `seq_lens[0]` and indexes `block_table` with no row stride -- so it would \
                     compute request 0's attention {seqs} times and report success. The batched \
                     walk is `{}`",
                    op.as_str(),
                    DflashOp::attn_for(seqs, matches!(op, DflashOp::AttnFp8Kv)).as_str(),
                )));
            }
            // ... and the batched walk's own two laws, which are what make the division exact. A
            // RAGGED batch has neither, and would need the reference's binary search over
            // `cu_seqlens_q` -- a different kernel, not a different argument.
            if batched && (m % seqs != 0 || (m / seqs) % at.bq != 0) {
                return Err(Error::Rule(format!(
                    "{}: {m} row(s) over {seqs} sequence(s), and the walk \
                     `seq = qblock * {} / q_per_seq` is exact only when every sequence \
                     contributes the SAME whole number of BLOCK_Q groups",
                    op.as_str(),
                    at.bq
                )));
            }
            // `m` is the query row count, `n` the query heads, `k` the sequence length, and
            // `r` the kv head count -- the attention's own axes.
            if n % at.gqa != 0 || r * at.gqa != n {
                return Err(Error::Rule(format!(
                    "{}: {n} query heads over {r} kv heads is not a GQA of {}",
                    op.as_str(),
                    at.gqa
                )));
            }
            // PER SEQUENCE, because `m` is the whole batch's rows: the bound is what ONE
            // sequence's attention may read up to.
            if k < m / seqs {
                return Err(Error::Rule(format!(
                    "{}: a sequence of {k} cannot hold {} query rows",
                    op.as_str(),
                    m / seqs
                )));
            }
            // THE BLOCK TABLE'S ROW STRIDE IS A RUNTIME FACT AND THERE IS NO KNOB TO FALL BACK ON.
            // Zero is the value a caller that never set the field has, and the cost of accepting
            // it is not a wrong stride -- it is the `None => ArgValue::Buffer` arm below writing
            // an eight-byte address into this four-byte slot and shifting `window` and `scale`
            // with it. Refuse by name instead (workstream F, round 22).
            if shape.bt_stride == 0 {
                return Err(Error::Rule(format!(
                    "{}: bt_stride is 0. It is the block table's row stride -- the second \
                     dimension of the `[num_seqs, max_blocks]` table THIS step wrote -- and it \
                     has no rules knob and no derivation from the shape; a caller that does not \
                     carry it cannot launch this kernel",
                    op.as_str()
                )));
            }
            // One workgroup per q block, four waves inside it taking four kv heads, and a SECOND
            // grid dimension for the kv heads past the fourth. The reference's x extent is
            // `q.shape[0] // BLOCK_Q + num_seqs` and the `+ num_seqs` block early-returns -- kept
            // so the launch matches it exactly.
            //
            // The y extent is 1 at the drafter's TP=2 shape (4 kv heads over 4 waves), which is
            // every launch any gate has taken, and 2 at TP=1, where the checkpoint's 8 kv heads
            // meet the same four waves. Before 2026-09-21 there was no y extent and TP=1 computed
            // heads 0..3 and left 4..7 holding whatever was in the output buffer -- a silent half
            // answer that the GQA check (`r * gqa == n`, which 8 x 4 = 32 satisfies) cannot see.
            let waves = at.block / 32;
            // THE x EXTENT CARRIES `num_seqs` AND NO KERNARG DOES: the reference's own extent is
            // `q.shape[0] // BLOCK_Q + num_seqs`, its trailing `num_seqs` blocks early-return, and
            // the walk divides by `q_per_seq` -- so a frame that disagreed with the grid about the
            // batch would compute a sequence nobody launched. One number, in one place.
            let mut args = vec![
                ("rows", i64::from(m)),
                ("num_q_heads", i64::from(n)),
                ("num_kv_heads", i64::from(r)),
                // The cache's block size is a RUNTIME value the shape carries; the rules'
                // knob is only the default a caller may leave unset.
                (
                    "block_size",
                    i64::from(if shape.block_size > 0 {
                        shape.block_size
                    } else {
                        at.block_size
                    }),
                ),
                // The block table's row stride, from the SHAPE for the same reason
                // `block_size` is: the scheduler pads every sequence's page list to the
                // widest in the step, so it is a property of the step and not of the model.
                ("bt_stride", i64::from(shape.bt_stride)),
                ("window", i64::from(at.window)),
                // The softmax scale, DERIVED from the head size rather than carried as a
                // knob: `Qwen3Attention` sets `self.scaling = head_dim ** -0.5` and
                // `_build_fused_kv_buffers` has already asserted every layer's attention
                // config is equal. It was missing until 2026-09-21 and, being an `F32` slot
                // with no value, was taking a POINTER (see `conv_add_rms`'s note). A
                // checkpoint with a trained `scaling` would need a rules row here, and the
                // fact that this one does not is a property of the model, not of the kernel.
                (
                    "scale",
                    i64::from((1.0f32 / (at.head as f32).sqrt()).to_bits()),
                ),
            ];
            if batched {
                // The drafter's own block, which is every request's row count. `num_seqs` is
                // `rows / q_per_seq` and is the grid's, above.
                args.push(("q_per_seq", i64::from(m / seqs)));
            }
            (
                [m / at.bq + seqs, r.div_ceil(waves.max(1)), 1],
                [at.block, 1, 1],
                args,
            )
        }
        DflashOp::KvAppendBf16 | DflashOp::KvAppendFp8 => {
            // `m` rows of `n` kv heads x `k` head dims.
            if k != rules.attn.head {
                return Err(Error::Rule(format!(
                    "{}: head_dim {k}, but this instantiation is the drafter's {}",
                    op.as_str(),
                    rules.attn.head
                )));
            }
            let contiguous = i64::from(n) * i64::from(k);
            let stride = |s: u32| if s == 0 { contiguous } else { i64::from(s) };
            (
                [m, 1, 1],
                [rules.kv.block, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("nkv", i64::from(n)),
                    ("head", i64::from(k)),
                    (
                        "block_size",
                        i64::from(if shape.block_size > 0 {
                            shape.block_size
                        } else {
                            rules.attn.block_size
                        }),
                    ),
                    ("key_stride", stride(shape.kv_src_stride.0)),
                    ("value_stride", stride(shape.kv_src_stride.1)),
                ],
            )
        }
        DflashOp::SelectorTopk => {
            // `m` rows of `n` columns, `r` = k.
            if r != rules.kv.topk_k {
                return Err(Error::Rule(format!(
                    "{}: k={r}, but the per-thread capacity this kernel is compiled for is {} -- \
                     and that equality is what makes one pass EXACT, so a larger k is a different \
                     kernel (a radix select), not a wider launch",
                    op.as_str(),
                    rules.kv.topk_k
                )));
            }
            if n < r {
                return Err(Error::Rule(format!(
                    "{}: a row of {n} cannot yield {r} entries",
                    op.as_str()
                )));
            }
            (
                [m, 1, 1],
                [rules.kv.topk_block, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("n", i64::from(n)),
                    ("k", i64::from(r)),
                ],
            )
        }
        DflashOp::HeadTopkThresh | DflashOp::HeadTopkGather => {
            // `m` rows of `n` columns, `r` = k. The two passes take the same shape because they
            // are two passes over the same row: separating them would let a caller select at one
            // width and gather at another, which is the one way this pair can be silently wrong.
            let hd = &rules.head;
            if r != hd.topk_k {
                return Err(Error::Rule(format!(
                    "{}: k={r}, but the gather's LDS array is compiled for {} entries. The rerank                      width is a SERVE setting (SABER_DRAFT_RERANK), so a different one is a                      different instantiation and not a wider launch",
                    op.as_str(),
                    hd.topk_k
                )));
            }
            if n < r {
                return Err(Error::Rule(format!(
                    "{}: a row of {n} cannot yield {r} entries",
                    op.as_str()
                )));
            }
            (
                [m, 1, 1],
                [hd.topk_block, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("n", i64::from(n)),
                    ("k", i64::from(r)),
                ],
            )
        }
        DflashOp::ActQuant | DflashOp::ActQuantRecip => {
            // `m` rows of `n` columns, and the input is the same `[m, n]` rather than the
            // `[m, 2n]` the silu variant reads — which is the whole difference between them.
            let ml = &rules.mlp;
            if n % ml.quant_group != 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} columns is not a whole number of {}-wide quant groups; the GEMM's                      `a_scale` is [M, K/{}] and a partial group has no scale",
                    op.as_str(),
                    ml.quant_group,
                    ml.quant_group
                )));
            }
            if n != k {
                return Err(Error::Rule(format!(
                    "{}: the quant's N={n} and K={k} are the SAME axis (the GEMM's reduction), so                      a shape that states two different numbers has not decided which is the width",
                    op.as_str()
                )));
            }
            let waves = ml.block / 32;
            (
                [(m * (n / ml.quant_group)).div_ceil(waves), 1, 1],
                [ml.block, 1, 1],
                vec![("t_rows", i64::from(m)), ("n", i64::from(n))],
            )
        }
        DflashOp::CtxKvLayout => {
            // `m` context rows, `n` layers, `k` kv heads, `r` head dim — the four extents of the
            // permutation, in the order the reference's `view(...)` writes them.
            let cx = &rules.ctx;
            if r == 0 {
                return Err(Error::Rule(format!(
                    "{}: head_dim 0 is not a layout",
                    op.as_str()
                )));
            }
            (
                [2 * n * m, 1, 1],
                [cx.layout_block, 1, 1],
                vec![
                    ("num_ctx", i64::from(m)),
                    ("layers", i64::from(n)),
                    ("nkv", i64::from(k)),
                    ("hd", i64::from(r)),
                ],
            )
        }
        DflashOp::HeadGroupSum => {
            // `m` rows of `n` columns reduced in groups of `head.group` -- and N and K are the
            // SAME axis here, exactly as they are for `act_quant`, because a reduction has one
            // width and not two.
            let hd = &rules.head;
            if n != k {
                return Err(Error::Rule(format!(
                    "{}: the group sum's N={n} and K={k} are the same axis (the head's \
                     reduction), so a shape that states two different numbers has not decided \
                     which is the width",
                    op.as_str()
                )));
            }
            if hd.group != 128 {
                return Err(Error::Rule(format!(
                    "{}: the kernel gives one WAVE a whole group as 32 lanes x 4 elements, which \
                     is ATen's own decomposition at group 128; group {} is a different \
                     instantiation and not a wider launch",
                    op.as_str(),
                    hd.group
                )));
            }
            if n % hd.group != 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} columns is not a whole number of {}-wide groups; the head reads one \
                     `xs` entry per group and a partial group has none",
                    op.as_str(),
                    hd.group
                )));
            }
            let ng = n / hd.group;
            let waves = hd.group_sum_waves.max(1);
            (
                [(m * ng).div_ceil(waves), 1, 1],
                [waves * 32, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("k", i64::from(n)),
                    ("stride_xs", i64::from(ng)),
                ],
            )
        }
        DflashOp::HeadRowPad => {
            // `m` PADDED rows of `n` columns, of which `r` carry the head's live draft rows.
            let hd = &rules.head;
            if n != k {
                return Err(Error::Rule(format!(
                    "{}: a copy has one width; N={n} and K={k} disagree about it",
                    op.as_str()
                )));
            }
            if r == 0 || r > m {
                return Err(Error::Rule(format!(
                    "{}: {r} live rows in a {m}-row pad is not a pad -- `_pow2_at_least` never \
                     shrinks, and zero live rows would make the whole output the fill",
                    op.as_str()
                )));
            }
            if !m.is_power_of_two() || m < hd.bm {
                return Err(Error::Rule(format!(
                    "{}: {m} is not a power of two at or above the head's {} rows, which is what \
                     `_pow2_at_least` returns",
                    op.as_str(),
                    hd.bm
                )));
            }
            let block = hd.pad_block.max(1);
            (
                [(m * n).div_ceil(8 * block).max(1), 1, 1],
                [block, 1, 1],
                vec![
                    ("m", i64::from(r)),
                    ("rows", i64::from(m)),
                    ("k", i64::from(n)),
                ],
            )
        }
        DflashOp::RowGather => {
            // `m` output rows of `n` BYTES each, gathered from `k` source rows.
            let hd = &rules.head;
            if m == 0 || n == 0 || k == 0 {
                return Err(Error::Rule(format!(
                    "{}: {m} rows x {n} bytes out of {k} is not a gather",
                    op.as_str()
                )));
            }
            let block = hd.pad_block.max(1);
            (
                [(m * n).div_ceil(8 * block).max(1), 1, 1],
                [block, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("row_bytes", i64::from(n)),
                    ("src_rows", i64::from(k)),
                ],
            )
        }
        DflashOp::HeadCandGather => {
            // `m` rows, `n` block-maxima entries per row, `r` selected positions per row.
            let hd = &rules.head;
            if r != hd.topk_k {
                return Err(Error::Rule(format!(
                    "{}: k={r} against the serve's rerank width of {}. The gather this maps is \
                     `bm.topk(RERANK)`'s, so a different k is a different selection and not a \
                     wider launch",
                    op.as_str(),
                    hd.topk_k
                )));
            }
            if n < r {
                return Err(Error::Rule(format!(
                    "{}: a row of {n} block maxima cannot have {r} positions selected out of it",
                    op.as_str()
                )));
            }
            let block = hd.pad_block.max(1);
            (
                [(m * r).div_ceil(block).max(1), 1, 1],
                [block, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("n", i64::from(n)),
                    ("k", i64::from(r)),
                ],
            )
        }
        DflashOp::SiluQuant => {
            // `m` rows of `n` output channels; the input is [m, 2n].
            let ml = &rules.mlp;
            if n % ml.quant_group != 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} channels is not a whole number of {}-wide quant groups",
                    op.as_str(),
                    ml.quant_group
                )));
            }
            let waves = ml.block / 32;
            (
                [(m * (n / ml.quant_group)).div_ceil(waves), 1, 1],
                [ml.block, 1, 1],
                vec![("t_rows", i64::from(m)), ("n", i64::from(n))],
            )
        }
        DflashOp::ConvAddRmsOpen | DflashOp::ConvAddRmsClose => {
            // `m` rows of `n` channels in `k` conv groups, restarting every `r` rows.
            let cv = &rules.conv;
            if n / cv.group_size != k {
                return Err(Error::Rule(format!(
                    "{}: {n} channels in {k} groups contradicts a group size of {}",
                    op.as_str(),
                    cv.group_size
                )));
            }
            if r == 0 || (r & (r - 1)) != 0 {
                return Err(Error::Rule(format!(
                    "{}: block_size {r} must be a power of two",
                    op.as_str()
                )));
            }
            // The kernel holds the row's fp32 values in REGISTERS between its two passes, because
            // the norm scales the fp32 sum and not the bf16 residual it stores. That array is 32
            // floats per thread, so a row wider than 8192 would need a re-read (and an extra
            // rounding in front of the weight multiply) rather than a wider launch.
            if n > rules.mlp.row_block * 4 * 8 {
                return Err(Error::Rule(format!(
                    "{}: {n} channels exceeds the {} this kernel keeps in registers between its \
                     two passes",
                    op.as_str(),
                    rules.mlp.row_block * 4 * 8
                )));
            }
            (
                [m, 1, 1],
                [rules.mlp.row_block, 1, 1],
                vec![
                    ("t_rows", i64::from(m)),
                    ("h", i64::from(n)),
                    ("groups", i64::from(k)),
                    ("block_size", i64::from(r)),
                    ("side", i64::from(shape.side)),
                    // The norm's epsilon. It was MISSING until 2026-09-21 (workstream F round
                    // 21), and the failure mode is the one the note at the bottom of this
                    // function describes: an F32 slot with no value in the table takes the
                    // `Buffer` arm and gets an eight-byte pointer written into a four-byte
                    // kernarg. Every gate this kernel had launched it from Python with
                    // hand-packed kernargs, so none of them went through this packing; it showed
                    // the first time a proposal ran the op through the real launcher.
                    ("eps", i64::from(rules.ctx.eps.to_bits())),
                ],
            )
        }
        DflashOp::RmsNorm => {
            // `m` rows of `n` channels, and nothing else: no conv, no residual, no second output.
            //
            // There is no register cap here, unlike `conv_add_rms`: that kernel has to keep the
            // fp32 row between its passes because its fp32 sum is not recoverable from the bf16
            // residual it stores, while this one's input is an unmodified bf16 tensor and
            // bf16 -> fp32 is exact, so pass 2 re-reads it. The only width this refuses is one
            // the vectorised load cannot cover.
            if n % 4 != 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} channels is not a whole number of bf16x4 loads",
                    op.as_str()
                )));
            }
            (
                [m, 1, 1],
                [rules.mlp.row_block, 1, 1],
                vec![
                    ("t_rows", i64::from(m)),
                    ("h", i64::from(n)),
                    // Same omission, same cause, found in the same launch (see `conv_add_rms`).
                    ("eps", i64::from(rules.ctx.eps.to_bits())),
                ],
            )
        }
        DflashOp::SelectorWalk => {
            // `m` requests, `n` steps, `r` candidates per step.
            if r != rules.selector.top_k {
                return Err(Error::Rule(format!(
                    "{}: top_k {r}, but the walk's argmax is compiled for {} -- and it is one wave, \
                     so a wider top-k is a different kernel",
                    op.as_str(),
                    rules.selector.top_k
                )));
            }
            if m == 0 || n == 0 {
                return Err(Error::Rule(format!(
                    "{}: {m} requests x {n} steps has nothing to walk",
                    op.as_str()
                )));
            }
            (
                [m, 1, 1],
                [rules.walk.block, 1, 1],
                vec![
                    ("reqs", i64::from(m)),
                    ("num_steps", i64::from(n)),
                    ("top_k", i64::from(r)),
                ],
            )
        }
        DflashOp::CtxRmsNorm => {
            // `m` rows of `n` channels, weight row every `r` rows.
            let cx = &rules.ctx;
            if n % cx.vec != 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} channels is not a whole number of {}-element vectors. The reference's\
                     helper would take its SCALAR prefix and tail loops for the remainder, which \
                     accumulate in a different order -- a different answer, not a slower one",
                    op.as_str(),
                    cx.vec
                )));
            }
            if r == 0 || m % r != 0 {
                return Err(Error::Rule(format!(
                    "{}: {m} rows is not a whole number of {r}-row weight groups",
                    op.as_str()
                )));
            }
            // THE LAUNCHER'S OWN RULE, measured (see `[ctx]`). Not an occupancy preference: the
            // narrow block puts three vectors in one accumulator where the wide one puts one, and
            // those are different sums. A block derived any other way is a different answer at
            // every shape where `block * vec < n`.
            let cap = if m >= cx.narrow_rows {
                cx.narrow_block
            } else {
                cx.wide_block
            };
            let block = (n / cx.vec).min(cap).max(1);
            (
                [m, 1, 1],
                [block, 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("h", i64::from(n)),
                    ("rows_per_group", i64::from(r)),
                    ("eps", i64::from(cx.eps.to_bits())),
                ],
            )
        }
        DflashOp::QkNormRope => {
            // `m` rows, `n` q heads, `k` kv heads, `r`-wide heads. One WAVE per (token, head):
            // the reference's reduction is a 32-lane DPP ladder over four elements per lane, so
            // the block is the wave and the head width it is compiled for is fixed.
            if r != QK_HEAD_DIM {
                return Err(Error::Rule(format!(
                    "{}: head_dim {r} -- the drafter's q/k norm + rope is compiled for {} and its \
                     reduction (four contiguous elements per lane over one wave) is the head \
                     width, not a loop bound",
                    op.as_str(),
                    QK_HEAD_DIM
                )));
            }
            if n == 0 || k == 0 {
                return Err(Error::Rule(format!(
                    "{}: {n} q heads and {k} kv heads; one launch does both blocks and neither \
                     may be empty",
                    op.as_str()
                )));
            }
            let qkv_n = (n + 2 * k) * r;
            (
                [m, n + k, 1],
                [QK_WAVE, 1, 1],
                vec![
                    ("m", i64::from(m)),
                    ("nq", i64::from(n)),
                    ("nkv", i64::from(k)),
                    // Derived, not passed: see `DflashShape::qk_rope`.
                    ("qkv_n", i64::from(qkv_n)),
                    // The drafter has ONE `rms_norm_eps` and `_build_fused_kv_buffers` asserts
                    // every layer's `q_norm.variance_epsilon` is it, which is the same value the
                    // context store's norms take.
                    ("eps", i64::from(rules.ctx.eps.to_bits())),
                ],
            )
        }
        DflashOp::CtxRope => {
            // `m` rows of `n` heads of `k` head_size, `r` of them rotated.
            let cx = &rules.ctx;
            if r == 0 || r % 2 != 0 || r > k {
                return Err(Error::Rule(format!(
                    "{}: rot_dim {r} must be even and at most head_size {k}",
                    op.as_str()
                )));
            }
            let pairs = n * (r / 2);
            (
                [m, 1, 1],
                [pairs.min(cx.rope_max_block), 1, 1],
                vec![
                    ("rows", i64::from(m)),
                    ("heads", i64::from(n)),
                    ("head_size", i64::from(k)),
                    ("rot_dim", i64::from(r)),
                    // Derived, not passed: see `DflashShape::ctx_rope`.
                    ("stride", i64::from(n * k)),
                ],
            )
        }
        DflashOp::RerankExact | DflashOp::RerankExactBf16 | DflashOp::RerankExactW8 => {
            let hd = &rules.head;
            if r == 0 || r > hd.max_r {
                return Err(Error::Rule(format!(
                    "{}: R={r} outside 1..={} (one workgroup per (row, candidate))",
                    op.as_str(),
                    hd.max_r
                )));
            }
            if k % hd.rerank_block_k != 0 {
                return Err(Error::Rule(format!(
                    "{}: K={k} is not a multiple of {}; the k loop is a whole number of \
                     BLOCK_K steps and the reduction tree is written for that width",
                    op.as_str(),
                    hd.rerank_block_k
                )));
            }
            (
                [m, r, 1],
                [hd.rerank_block, 1, 1],
                vec![
                    ("k", i64::from(k)),
                    ("stride_w", i64::from(k)),
                    ("r", i64::from(r)),
                    // The vocabulary width the `_scatter` variant writes into. `n` is the exact
                    // weight's row count, which IS that width: the rerank scores candidate columns
                    // of the same head the coarse pass wrote.
                    ("n", i64::from(n)),
                ],
            )
        }
    };
    let (packed, bytes) = abi::pack(sig);
    let mut args = Vec::with_capacity(sig.len());
    for (a, (offset, size)) in sig.iter().zip(packed) {
        let value = match values.iter().find(|(nm, _)| *nm == a.name) {
            // A scalar slot whose declared kind is F32 carries the BITS of the float in the
            // table, because the table is one integer type. Without this a four-byte `eps` slot
            // takes the `Buffer` arm below and gets an eight-byte pointer written into it --
            // silent kernarg corruption, and the reason the context store's epsilon is a rules
            // knob rather than a caller's argument.
            Some((_, v)) if a.kind == abi::ArgKind::F32 => ArgValue::F32 {
                #[allow(clippy::cast_possible_truncation)]
                fval: f32::from_bits(*v as u32),
            },
            Some((_, v)) => ArgValue::I32 { ival: *v },
            None => ArgValue::Buffer {
                buffer: a.name.to_owned(),
            },
        };
        args.push(Slot {
            name: a.name.to_owned(),
            offset,
            size,
            value,
        });
    }
    Ok(LaunchDescriptor {
        op: row.op.clone(),
        plugin: row.plugin.clone(),
        kernel: row.symbol.clone(),
        grid,
        block,
        // The GEMM takes B straight into the WMMA operand registers and the head takes its weight
        // the same way; the only LDS in either is declared statically by the code object.
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> DflashRules {
        DflashRules::prod_dflash().expect("the dflash rules parse")
    }

    /// The target's bf16 `lm_head` (N2) takes the drafter's bf16 GEMM at the vocabulary's N on its
    /// OWN envelope, while the drafter's projections keep the gated `max_n`: the control is the
    /// same N through plain `resolve`, which must still refuse.
    #[test]
    fn the_bf16_head_resolves_at_the_vocabulary_and_the_projection_envelope_does_not_move() {
        let r = rules();
        for (m, sym, y) in [
            (1, "dflash_bgemm_bf16_m16n16", 1),
            (8, "dflash_bgemm_bf16_m16n16", 1),
            (32, "dflash_bgemm_bf16_m32n16", 1),
            (64, "dflash_bgemm_bf16_m32n16", 2),
        ] {
            let d = resolve_head_bgemm(m, 124_160, 5120, &r).expect("the TP=2 head");
            assert_eq!(d.kernel, sym, "M={m}");
            assert_eq!(d.grid, [124_160 / 16 / 4, y, 1], "M={m}");
        }
        resolve_head_bgemm(1, 248_320, 5120, &r).expect("the TP=1 head");
        let e = resolve_head_bgemm(1, 248_336, 5120, &r).expect_err("past head_max_n");
        assert!(e.to_string().contains("head_max_n"), "{e}");
        let e = resolve(DflashOp::BgemmBf16, DflashShape::new(8, 124_160, 5120), &r)
            .expect_err("the projections' envelope is unchanged");
        assert!(e.to_string().contains("at most 10240"), "{e}");
    }

    /// The fp8-weight head (N4) takes the same bands as the bf16 head at the same grids, one more
    /// kernarg (`ws`, a buffer between `w` and `c`), and the same `head_max_n` refusal. Control:
    /// the bf16 head's descriptor at the same shape has no `ws` and names the bf16 symbols.
    #[test]
    fn the_fp8_head_takes_the_bf16_heads_bands_with_one_more_kernarg() {
        let r = rules();
        for (m, sym, y) in [
            (1, "dflash_bgemm_w8_m16n16", 1),
            (8, "dflash_bgemm_w8_m16n16", 1),
            (16, "dflash_bgemm_w8_m16n16", 1),
            (32, "dflash_bgemm_w8_m32n16", 1),
        ] {
            let d = resolve_head_w8(m, 124_160, 5120, &r).expect("the TP=2 fp8 head");
            let bf = resolve_head_bgemm(m, 124_160, 5120, &r).expect("the bf16 head");
            assert!(d.kernel.starts_with(sym), "M={m}: {}", d.kernel);
            assert_eq!((d.grid, d.block), (bf.grid, bf.block), "M={m}");
            assert_eq!(d.grid, [124_160 / 16 / 4, y, 1], "M={m}");
            let names: Vec<&str> = d.args.iter().map(|a| a.name.as_str()).collect();
            assert_eq!(names, ["a", "w", "ws", "c", "m", "n", "k"], "M={m}");
            assert!(bf.arg("ws").is_none(), "the bf16 head has no scale");
            assert!(bf.kernel.starts_with("dflash_bgemm_bf16"), "{}", bf.kernel);
        }
        resolve_head_w8(1, 248_320, 5120, &r).expect("the TP=1 head");
        let e = resolve_head_w8(1, 248_336, 5120, &r).expect_err("past head_max_n");
        assert!(e.to_string().contains("head_max_n"), "{e}");
    }

    /// CMP27F: under the `head_m64` knob ONLY 49..64 rows move to the one-pass tile (one row tile,
    /// 64 columns a 4-wave workgroup, the band's kernargs); every other M is the committed route,
    /// and without the knob 64 rows stay on the 32-row band (the control).
    #[test]
    fn head_m64_knob_routes_only_the_49_to_64_row_head() {
        let r = DflashRules::prod_dflash().unwrap();
        for m in [1u32, 8, 16, 32, 33, 40, 48, 49, 57, 64] {
            let base = resolve_head_w8(m, 124_160, 5120, &r).unwrap();
            let d = resolve_head_w8_m64(m, 124_160, 5120, &r).unwrap();
            if m >= HEAD_M64_MIN_ROWS {
                assert_eq!(d.kernel, HEAD_M64_SYMBOL);
                assert_eq!(d.plugin, HEAD_M64_PLUGIN);
                assert_eq!(d.grid, [124_160 / 64, 1, 1]);
                assert_eq!(d.block, [128, 1, 1]);
                assert_eq!(d.kernarg_bytes, base.kernarg_bytes);
                assert!(
                    base.kernel.contains("m32n16"),
                    "{m}: control {}",
                    base.kernel
                );
            } else {
                assert_eq!(d.kernel, base.kernel, "{m}");
                assert_eq!(d.grid, base.grid, "{m}");
            }
        }
        assert!(crate::abi::signature_for(HEAD_M64_SYMBOL).is_some());
    }

    /// CMP27K: under the `gateup_m64` knob ONLY 33..64 rows move to the one-pass tile (one row
    /// tile, 136 workgroups of 256 threads at gate_up's N, the m-tiled kernel's kernargs); without
    /// the knob 64 rows stay on four 16-row tiles (the control), and M <= 32 never moves.
    #[test]
    fn gateup_m64_knob_routes_only_the_33_to_64_row_gemm() {
        let off = DflashRules::prod_dflash().unwrap();
        assert_eq!(off.knobs.gateup_m64, 0, "never in the committed table");
        let mut on = off.clone();
        on.knobs.gateup_m64 = GATEUP_M64_MIN_ROWS;
        let mut wide = off.clone();
        wide.knobs.gateup_m64 = 1; // clamped up to the floor
        for m in [16u32, 17, 24, 32] {
            let d = resolve_blockscale_gateup(m, 17_408, 5120, &wide).unwrap();
            assert_eq!(
                d.kernel == GATEUP_M64_SYMBOL,
                m >= GATEUP_M64_FLOOR_ROWS,
                "{m}"
            );
        }
        for m in [1u32, 8, 16, 17, 32, 33, 40, 48, 56, 64, 65, 128] {
            let base = resolve(
                DflashOp::blockscale_for(m, &off),
                DflashShape::new(m, 17_408, 5120),
                &off,
            )
            .unwrap();
            let ctl = resolve_blockscale_gateup(m, 17_408, 5120, &off).unwrap();
            assert_eq!(
                ctl.kernel, base.kernel,
                "{m}: knob off is the committed route"
            );
            assert_eq!(ctl.grid, base.grid, "{m}");
            let d = resolve_blockscale_gateup(m, 17_408, 5120, &on).unwrap();
            if (GATEUP_M64_MIN_ROWS..=64).contains(&m) {
                assert_eq!(d.kernel, GATEUP_M64_SYMBOL);
                assert_eq!(d.plugin, GATEUP_M64_PLUGIN);
                assert_eq!(d.grid, [136, 1, 1]);
                assert_eq!(d.block, [256, 1, 1]);
                assert_eq!(d.kernarg_bytes, base.kernarg_bytes);
                assert_eq!(
                    base.kernel, "dflash_gemm_a8w8_bs128_mtn128k128",
                    "{m}: control"
                );
                assert_eq!(base.grid, [136, m.div_ceil(16), 1]);
            } else {
                assert_eq!(d.kernel, base.kernel, "{m}");
                assert_eq!(d.grid, base.grid, "{m}");
            }
        }
        assert!(crate::abi::signature_for(GATEUP_M64_SYMBOL).is_some());
    }

    /// CMP27Z: under the head's `m64` knob ONLY whole 64-row tiles of the 32-row `_lx` band move
    /// to the one-pass twin (the same kernargs, x extent and block; y extent m / 64); without the
    /// knob 64 rows stay on the band's two y tiles (the control), and 16 / 32 / 48 never move.
    #[test]
    fn cmp27z_head_m64_knob_routes_only_whole_64_row_tiles() {
        let off = DflashRules::prod_dflash().unwrap();
        assert!(!off.head.m64, "never in the committed table");
        let mut on = off.clone();
        on.head.m64 = true;
        for m in [16u32, 32, 48, 64, 128] {
            let base = resolve(
                DflashOp::draft_head_for(m, &off),
                DflashShape::new(m, 124_160, 5120),
                &off,
            )
            .unwrap();
            let ctl = resolve_draft_head_int2(m, 124_160, 5120, &off).unwrap();
            assert_eq!(
                ctl.kernel, base.kernel,
                "{m}: knob off is the committed route"
            );
            assert_eq!(ctl.grid, base.grid, "{m}");
            assert_eq!(ctl.args, base.args, "{m}");
            let d = resolve_draft_head_int2(m, 124_160, 5120, &on).unwrap();
            if m % 64 == 0 && m <= off.head.max_m {
                assert_eq!(
                    base.kernel, "dflash_draft_head_int2_m32n64x2g128_lx",
                    "{m}: control"
                );
                assert_eq!(d.kernel, HEAD_INT2_M64_SYMBOL);
                assert_eq!(d.plugin, HEAD_INT2_M64_PLUGIN);
                assert_eq!(d.grid, [base.grid[0], m / 64, 1]);
                assert_eq!(base.grid[1], m / 32);
                assert_eq!(d.block, base.block);
                assert_eq!(d.args, base.args);
                assert_eq!(d.kernarg_bytes, base.kernarg_bytes);
            } else {
                assert_eq!(d.kernel, base.kernel, "{m}");
                assert_eq!(d.grid, base.grid, "{m}");
            }
        }
        assert!(crate::abi::signature_for(HEAD_INT2_M64_SYMBOL).is_some());
    }

    /// H33: between the 32-row band and `head_band` the fp8 head takes the ONE-PASS wide band,
    /// one y-band where the 32-row band would put two on the grid and read the weight twice
    /// (b7s5 = 35, b8s5 = 40); past `head_band` it keeps the 32-row band. Control: the bf16 head at
    /// the same M still takes the 32-row band's two passes.
    /// CMP27L: the b1 head arm moves ONLY 1..16 rows, onto the 32-row band (one row tile, the
    /// same kernargs); 17+ rows are exactly the committed route. Control: the committed route at
    /// the same M is the 16-row band.
    #[test]
    fn cmp27l_the_b1_head_arm_moves_only_the_16_row_band() {
        let r = rules();
        for m in [1, 4, 8, 10, 16] {
            let d = resolve_head_w8_b1(m, 124_160, 5120, &r).expect("the arm");
            let c = resolve_head_w8(m, 124_160, 5120, &r).expect("the committed route");
            assert!(c.kernel.contains("_m16n16_"), "M={m}: {}", c.kernel);
            assert_eq!(d.kernel, "dflash_bgemm_w8_m32n16_w128_u16_nt", "M={m}");
            assert_eq!(d.grid, [124_160 / 16 / 4, 1, 1], "M={m}");
            assert_eq!(d.args, c.args, "M={m}: the kernargs");
        }
        for m in [17, 32, 40, 64] {
            assert_eq!(
                resolve_head_w8_b1(m, 124_160, 5120, &r).expect("arm"),
                resolve_head_w8(m, 124_160, 5120, &r).expect("route"),
                "M={m}"
            );
        }
    }

    #[test]
    fn the_fp8_head_reads_the_weight_once_up_to_the_wide_band() {
        let r = rules();
        let hb = r.bgemm.head_band;
        assert_eq!(hb, 48, "the committed wide band");
        let tag = format!("_m{hb}n16_");
        for m in [33, 35, 40, hb] {
            let d = resolve_head_w8(m, 124_160, 5120, &r).expect("the wide band");
            assert!(d.kernel.contains(&tag), "M={m}: {}", d.kernel);
            assert_eq!(d.grid, [124_160 / 16 / 4, 1, 1], "M={m}");
            let names: Vec<&str> = d.args.iter().map(|a| a.name.as_str()).collect();
            assert_eq!(names, ["a", "w", "ws", "c", "m", "n", "k"], "M={m}");
        }
        for m in [32, hb + 1, 64, 256] {
            let d = resolve_head_w8(m, 124_160, 5120, &r).expect("the 32-row band");
            assert!(d.kernel.contains("_m32n16_"), "M={m}: {}", d.kernel);
            assert_eq!(d.grid, [124_160 / 16 / 4, m.div_ceil(32), 1], "M={m}");
        }
        let bf = resolve_head_bgemm(40, 124_160, 5120, &r).expect("the bf16 head");
        assert_eq!(
            bf.grid[1], 2,
            "the control: the bf16 head's 32-row band reads twice at 40"
        );
    }

    /// H41: every fp8-head band routes the 16-byte weight load (`_w128_`), whose output is the
    /// 8-byte symbol's element for element (gated on the card); the 8-byte symbols stay compiled
    /// as the gate's reference. Control: the pre-H41 rules text (the 8-byte rows) resolves the same
    /// bands, grids and kernargs to symbols WITHOUT `_w128_`, so the check can fail.
    #[test]
    fn the_fp8_head_routes_the_sixteen_byte_load_at_every_band() {
        let r = rules();
        let text = include_str!("../rules/dflash_gfx1201.toml");
        let old = text
            .replacen(
                "symbol = \"dflash_bgemm_w8_m16n16_w128_u16\"",
                "symbol = \"dflash_bgemm_w8_m16n16_rne_u16\"",
                1,
            )
            .replacen(
                "symbol = \"dflash_bgemm_w8_m32n16_w128_u16_nt\"",
                "symbol = \"dflash_bgemm_w8_m32n16_rne\"",
                1,
            )
            .replacen(
                "symbol = \"dflash_bgemm_w8_m48n16_w128_u8\"",
                "symbol = \"dflash_bgemm_w8_m48n16_rne\"",
                1,
            );
        let ctl = DflashRules::from_toml(&old).expect("the 8-byte rows");
        for (m, band) in [
            (1, "_m16n16_"),
            (8, "_m16n16_"),
            (16, "_m16n16_"),
            (17, "_m32n16_"),
            (32, "_m32n16_"),
            (33, "_m48n16_"),
            (40, "_m48n16_"),
            (48, "_m48n16_"),
            (64, "_m32n16_"),
        ] {
            let d = resolve_head_w8(m, 124_160, 5120, &r).expect("the routed head");
            let c = resolve_head_w8(m, 124_160, 5120, &ctl).expect("the control");
            assert!(
                d.kernel.contains("_w128_") && d.kernel.contains(band),
                "M={m}: {}",
                d.kernel
            );
            assert!(
                !c.kernel.contains("_w128_") && c.kernel.contains(band),
                "M={m}: {}",
                c.kernel
            );
            assert_eq!((d.grid, d.block), (c.grid, c.block), "M={m}");
            let dn: Vec<&str> = d.args.iter().map(|a| a.name.as_str()).collect();
            let cn: Vec<&str> = c.args.iter().map(|a| a.name.as_str()).collect();
            assert_eq!(dn, cn, "M={m}");
        }
    }

    /// The wide band's row count is compiled into its symbol, so a rules file that states another
    /// `head_band` than the routed symbol carries is refused by name (and one that states a band
    /// that is not whole tiles above the 32-row band, or more than four fragments, too).
    #[test]
    fn a_head_band_the_symbol_does_not_carry_is_refused() {
        let text = include_str!("../rules/dflash_gfx1201.toml");
        for (bad, what) in [
            ("head_band = 64", "`bgemm_w8_head_wide`"),
            ("head_band = 32", "head_band 32"),
            ("head_band = 72", "head_band 72"),
            ("head_band = 80", "head_band 80"),
        ] {
            let t = text.replacen("head_band = 48", bad, 1);
            let e = DflashRules::from_toml(&t).expect_err(bad);
            assert!(e.to_string().contains(what), "{bad}: {e}");
        }
        DflashRules::from_toml(text).expect("the committed rules");
    }

    /// The five shapes the census read out of prod's Triton specialisation names.
    const CENSUS: [(&str, u32, u32); 5] = [
        ("qkv_proj", 3072, 5120),
        ("o_proj", 5120, 2048),
        ("gate_up", 17408, 5120),
        ("down_proj", 5120, 8704),
        ("aux_fc", 5120, 25600),
    ];

    #[test]
    fn the_committed_rules_resolve_every_census_shape() {
        let r = rules();
        for p in [
            "dflash_gemm",
            "dflash_head",
            "dflash_bgemm",
            "dflash_selector",
            "dflash_conv",
        ] {
            assert!(r.owns(p), "{p}");
        }
        // the blockscale GEMM and its M-TILED instantiation, which is the same tile on the grid's
        // y axis and the op a batched proposal's 21 launches ask for.
        assert_eq!(r.rows_for("dflash_gemm").len(), 2);
        // the int2 head, its M-tiled row slice and its 32-ROW BAND (the one the rules pick above
        // the traced tile), plus the THREE rerank arms (an e4m3 head with a per-row scale, a bf16
        // one and N6's fp8 record expanded to bf16), plus the FOUR undeclared launches a chain of
        // the backbone found in this seam: `_pow2_at_least`'s row pad, `xs`, the `bi` gather that
        // turns the top-80's positions into vocabulary columns, and the row gather that is two of
        // `_generate_draft`'s.
        assert_eq!(r.rows_for("dflash_head").len(), 10);
        assert_eq!(r.rows_for("dflash_conv").len(), 2);
        for (name, n, k) in CENSUS {
            let d = resolve(DflashOp::GemmBlockscale, DflashShape::new(8, n, k), &r)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(d.plugin, "dflash_gemm");
            assert_eq!(d.kernel, "dflash_gemm_a8w8_bs128_m16n128k128");
            assert_eq!(
                d.grid,
                [n / 128, 1, 1],
                "{name}: one workgroup per 128 of N"
            );
            assert_eq!(d.block, [128, 1, 1]);
            assert_eq!(d.kernarg_bytes, 52);
            assert_eq!(d.n_launches, 1);
            assert!(d.workspaces.is_empty(), "{name}: no split-K slab");
        }
    }

    /// The two bf16 projections the census traced, at their own shapes: `kernel_projection`
    /// (5120 -> 2 * taps * groups = 1280, twice per layer) and the selector's `hidden_projection`
    /// (5120 -> the selector rank).
    #[test]
    fn the_bf16_projections_resolve_at_the_shapes_the_census_traced() {
        let r = rules();
        for (name, n, waves) in [
            ("kernel_projection", 1280u32, 80u32),
            ("hidden_projection", 256, 16),
        ] {
            let d = resolve(DflashOp::BgemmBf16, DflashShape::new(8, n, 5120), &r)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(d.plugin, "dflash_bgemm");
            assert_eq!(d.block, [128, 1, 1]);
            // One wave per 16-wide output tile, four waves to a workgroup.
            assert_eq!(d.grid, [waves / 4, 1, 1], "{name}");
            assert_eq!(d.kernarg_bytes, 36);
        }
        let e = resolve(DflashOp::BgemmBf16, DflashShape::new(17, 256, 5120), &r)
            .expect_err("M past the tile");
        assert!(format!("{e}").contains("exceeds the tile's 16 rows"), "{e}");
    }

    /// The bgemm's envelope is where the GATE stopped, not where the kernel stops running.
    ///
    /// The reference is `F.linear`, and hipBLASLt picks an algorithm per shape: at M = 8, K = 5120
    /// it is byte-identical at N = 1280 and N = 2560 and differs in 19 of 40960 at N = 5120
    /// (2026-09-21). The envelope was 2560 for a round because of that, and the context store's
    /// fused KV projection — N = `L * 2 * kv_size`, 10240 on one rank — sat outside it.
    ///
    /// F26 took the shape back, and the number this test pins moved with the evidence rather than
    /// with a preference. What it still pins is the SHAPE of the claim: the widest N the rules
    /// will hand this kernel is exactly the proposal's widest launch, every N up to it is a whole
    /// number of 16-wide tiles, and one past it is refused BY NAME. A `max_n` edited without a
    /// gate still fails here, because the number is stated twice — here and in the plan's own
    /// `context_kv_n()` — and the two have to agree.
    #[test]
    fn the_bgemm_envelope_is_the_widest_launch_the_proposal_makes() {
        let r = rules();
        assert_eq!(
            r.bgemm.max_n, 10240,
            "the measured band, not a hardware limit"
        );
        // Every shape the drafter's projections and the context store ask for.
        for n in [256, 1280, 2560, 5120, 10240] {
            resolve(DflashOp::BgemmBf16, DflashShape::new(8, n, 5120), &r)
                .unwrap_or_else(|e| panic!("N={n}: {e}"));
        }
        let e = resolve(DflashOp::BgemmBf16, DflashShape::new(8, 10496, 5120), &r)
            .expect_err("one tile past the envelope");
        assert!(format!("{e}").contains("at most 10240"), "{e}");
    }

    /// The four launches F26 covered, at the shapes the r23 capture actually ran them at.
    ///
    /// Each one was `Coverage::Absent` in the plan, and each refusal named a different kind of
    /// gap: a producer with no kernel (the activation quant), an ATen selection whose tie order
    /// could not be pinned (the two top-k passes), and a copy that computes nothing (the layout).
    /// This asserts the grid and the block of each, so a rules edit that moves one is a failing
    /// test rather than a launch that runs at the wrong width.
    #[test]
    fn the_sites_that_kept_the_drafter_out_of_the_engine_resolve() {
        let r = rules();
        // The activation quant at BOTH its shapes: the aux projection's [8, 25600] and a layer's
        // [8, 4096]. One wave owns one 128-wide quant group, four waves to a workgroup.
        for (m, n) in [(8u32, 25600u32), (8, 4096)] {
            let d = resolve(DflashOp::ActQuant, DflashShape::new(m, n, n), &r)
                .unwrap_or_else(|e| panic!("{m}x{n}: {e}"));
            assert_eq!(d.plugin, "dflash_mlp");
            assert_eq!(d.block, [128, 1, 1]);
            assert_eq!(d.grid, [(m * (n / 128)).div_ceil(4), 1, 1]);
        }
        // A width that is not a whole number of groups has no scale for its tail.
        let e = resolve(DflashOp::ActQuant, DflashShape::new(8, 4100, 4100), &r)
            .expect_err("a partial quant group");
        assert!(format!("{e}").contains("whole number of 128-wide"), "{e}");
        // ... and a shape whose N and K disagree has not decided which axis it means.
        let e = resolve(DflashOp::ActQuant, DflashShape::new(8, 4096, 5120), &r)
            .expect_err("two widths for one axis");
        assert!(format!("{e}").contains("SAME axis"), "{e}");

        // The int2 head's own selection, both passes, at TP=1's 31040 block maxima.
        for op in [DflashOp::HeadTopkThresh, DflashOp::HeadTopkGather] {
            let d = resolve(op, DflashShape::rerank(16, 31040, 5120, 80), &r)
                .unwrap_or_else(|e| panic!("{}: {e}", op.as_str()));
            assert_eq!(d.plugin, "dflash_topk");
            assert_eq!(d.grid, [16, 1, 1], "one workgroup per row");
            // 1024 since F30, and the literal is deliberate: `head.topk_block` in
            // `rules/dflash_gfx1201.toml` is the number, the code object's
            // `max_flat_workgroup_size` is what refuses a rules file the kernel was not compiled
            // for, and this line is what notices that the two moved apart.
            assert_eq!(d.block, [1024, 1, 1]);
        }
        // The rerank width is a SERVE setting and the gather's array is compiled for it.
        let e = resolve(
            DflashOp::HeadTopkGather,
            DflashShape::rerank(16, 31040, 5120, 32),
            &r,
        )
        .expect_err("the module default rather than the serve's");
        assert!(format!("{e}").contains("compiled for 80 entries"), "{e}");

        // The layout copy: 2 x L x num_ctx runs of nkv x head_dim.
        let d = resolve(
            DflashOp::CtxKvLayout,
            DflashShape::ctx_layout(8, 5, 8, 128),
            &r,
        )
        .expect("the context store's permutation");
        assert_eq!(d.plugin, "dflash_ctx");
        assert_eq!(d.grid, [2 * 5 * 8, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
    }

    /// The context store's two ops, at the four launches a chrome trace of prod recorded.
    ///
    /// The block is the whole point. `upstream::rms_norm_kernel`'s launcher NARROWS as the grid grows
    /// — the same kernel at the same hidden size took 640 threads over 8 rows and 256 over 2048 —
    /// and that is arithmetic and not occupancy, because 256 x 8 is 2048 against a 5120-wide row
    /// and a thread that chains three vectors does not associate the sum the way one that chains
    /// one does. Every number below is a traced launch, so a rule that drifts fails here rather
    /// than in a gate round.
    #[test]
    fn the_context_store_reproduces_prods_traced_launches() {
        let r = rules();
        // (rows, hidden, rows_per_group) -> prod's grid and block.
        for (rows, h, rpg, block) in [
            // the hidden norm at a steady-state decode block, and after a prefill
            (8u32, 5120u32, 8u32, 640u32),
            (2048, 5120, 2048, 256),
            // the stacked K norm at both, which is 16 vectors and never reaches either cap
            (160, 128, 32, 16),
            (40960, 128, 8192, 16),
        ] {
            let d = resolve(
                DflashOp::CtxRmsNorm,
                DflashShape::ctx_norm(rows, h, rpg),
                &r,
            )
            .unwrap_or_else(|e| panic!("rows {rows} h {h}: {e}"));
            assert_eq!(d.plugin, "dflash_ctx");
            assert_eq!(d.kernel, "dflash_rms_norm_grouped_w8");
            assert_eq!(d.grid, [rows, 1, 1], "one workgroup per row");
            assert_eq!(d.block, [block, 1, 1], "rows {rows}, h {h}");
        }
        // The rope: one workgroup per row, `min(heads * rot_dim / 2, 512)` threads — traced at
        // 256 for the drafter's four kv heads.
        for rows in [40u32, 10240] {
            let d = resolve(
                DflashOp::CtxRope,
                DflashShape::ctx_rope(rows, 4, 128, 128),
                &r,
            )
            .expect("the batched context rope");
            assert_eq!(d.kernel, "dflash_rope_neox_k_inplace");
            assert_eq!(d.grid, [rows, 1, 1]);
            assert_eq!(d.block, [256, 1, 1]);
        }
        // A width the reference's helper would have to split into a scalar prefix and a vector
        // run is refused rather than launched: the two loops accumulate in a different order.
        let e = resolve(DflashOp::CtxRmsNorm, DflashShape::ctx_norm(8, 5124, 8), &r)
            .expect_err("5124 is not a whole number of 8-element vectors");
        assert!(
            format!("{e}").contains("whole number of 8-element vectors"),
            "{e}"
        );
    }

    /// One workgroup per block position, and the block length is what makes position 0 take the
    /// anchor instead of the previous position's candidates.
    #[test]
    fn the_selector_resolves_one_workgroup_per_position() {
        let r = rules();
        let d = resolve(
            DflashOp::SelectorScore,
            DflashShape::selector(8, 8, 256, 16),
            &r,
        )
        .expect("the drafter's block");
        assert_eq!(d.plugin, "dflash_selector");
        assert_eq!(d.grid, [8, 1, 1]);
        assert_eq!(d.block, [32, 1, 1]);
        assert_eq!(d.kernarg_bytes, 72);
        let e = resolve(
            DflashOp::SelectorScore,
            DflashShape::selector(8, 8, 128, 16),
            &r,
        )
        .expect_err("a rank the kernel is not compiled for");
        assert!(format!("{e}").contains("compiled for 256 x 16"), "{e}");
    }

    /// The convolution at the drafter's own geometry: 8 rows of 5120 in 320 groups of 16,
    /// restarting every 8 rows. The quant variant's grid is in WAVES, one per 128-wide group.
    #[test]
    fn the_convolution_resolves_both_variants() {
        let r = rules();
        let plain = resolve(
            DflashOp::GroupedConv,
            DflashShape::conv(8, 5120, 320, 8, 1),
            &r,
        )
        .expect("finish");
        assert_eq!(plain.plugin, "dflash_conv");
        assert_eq!(plain.kernel, "dflash_grouped_conv_t2g16");
        assert_eq!(plain.grid, [(8 * 5120 / 4) / 128, 1, 1]);
        assert_eq!(plain.block, [128, 1, 1]);
        assert_eq!(plain.kernarg_bytes, 52);
        let q = resolve(
            DflashOp::GroupedConvQuant,
            DflashShape::conv(8, 5120, 320, 8, 0),
            &r,
        )
        .expect("prepare");
        assert_eq!(q.kernel, "dflash_grouped_conv_quant_t2g16");
        assert_eq!(
            q.grid,
            [(8 * 40) / 4, 1, 1],
            "one wave per quant group, four to a workgroup"
        );
        assert_eq!(q.kernarg_bytes, 60);
        // A block size that is not a power of two takes prod's OTHER branch (`position % bs`),
        // which is a different kernel, not a slower one.
        let e = resolve(
            DflashOp::GroupedConv,
            DflashShape::conv(8, 5120, 320, 6, 0),
            &r,
        )
        .expect_err("modulo branch");
        assert!(format!("{e}").contains("power of two"), "{e}");
        let e = resolve(
            DflashOp::GroupedConv,
            DflashShape::conv(8, 5120, 320, 8, 2),
            &r,
        )
        .expect_err("no such side");
        assert!(format!("{e}").contains("neither `prepare`"), "{e}");
    }

    /// Every M in 1..=16 is the SAME launch. That is not an optimisation, it is why the traced
    /// step shows one grid per shape however many tokens the drafter proposed.
    #[test]
    fn every_m_up_to_the_tile_is_one_grid() {
        let r = rules();
        let grid = |m| {
            resolve(
                DflashOp::GemmBlockscale,
                DflashShape::new(m, 5120, 2048),
                &r,
            )
            .map(|d| d.grid)
        };
        let want = grid(8).expect("M=8 resolves");
        for m in 1..=16 {
            assert_eq!(grid(m).expect("M in 1..=16 resolves"), want, "M={m}");
        }
        let e = grid(17).expect_err("M past the tile is refused");
        assert!(format!("{e}").contains("exceeds the tile's 16 rows"), "{e}");
    }

    /// The traced head launch, argument for argument: grid 124160x1x1 at 64 threads is 1940
    /// workgroups, and every stride is DERIVED from the shape rather than passed in.
    #[test]
    fn the_head_resolves_the_launch_the_census_traced() {
        let r = rules();
        let d = resolve(
            DflashOp::DraftHeadInt2,
            DflashShape::new(16, 124_160, 5120),
            &r,
        )
        .expect("prod's head shape");
        assert_eq!(d.plugin, "dflash_head");
        // ws-h43: two 64-column blocks a workgroup (`wg_blocks = 2`), x staged once in LDS -- 970
        // workgroups of 128 threads over the same 1940 blocks (`nblk` below is unchanged).
        assert_eq!(d.kernel, "dflash_draft_head_int2_m16n64x2g128_lx");
        assert_eq!(d.grid, [970, 1, 1]);
        assert_eq!(d.block, [128, 1, 1]);
        assert_eq!(d.kernarg_bytes, 92);
        let val = |nm: &str| match d.arg(nm).expect(nm).value {
            ArgValue::I32 { ival } => ival,
            _ => panic!("{nm} is not an immediate"),
        };
        assert_eq!(val("stride_wq"), 1280, "the packed row is K/4 bytes");
        assert_eq!(val("stride_s"), 40, "one scale per 128-wide group");
        assert_eq!(val("stride_xs"), 40);
        assert_eq!(val("nblk"), 1940);
    }

    /// NEGATIVE CONTROL for `wg_blocks`: a two-block grid on the shipped 64-thread band (or the
    /// `_lx` symbol on a one-block grid) would score half the vocabulary, so the rules refuse the
    /// pairing by name; the shipped band at one block is the control that still resolves.
    #[test]
    fn the_head_grid_and_its_symbol_agree_on_blocks_per_workgroup() {
        let mut r = rules();
        r.head.wg_blocks = 1;
        let e = r.check().expect_err("the _lx symbol on a one-block grid");
        assert!(format!("{e}").contains("wg_blocks"), "{e}");
        for row in &mut r.kernel {
            if row.op == "draft_head_int2" {
                row.symbol = "dflash_draft_head_int2_m16n64g128".into();
            }
            if row.op == "draft_head_int2_m32" {
                row.symbol = "dflash_draft_head_int2_m32n64g128".into();
            }
        }
        r.check().expect("the shipped band at one block");
        let d = resolve(
            DflashOp::DraftHeadInt2,
            DflashShape::new(16, 124_160, 5120),
            &r,
        )
        .expect("the control resolves");
        assert_eq!((d.grid, d.block), ([1940, 1, 1], [64, 1, 1]));
        r.head.wg_blocks = 2;
        assert!(r.check().is_err(), "the band's symbol on a two-block grid");
        r.head.wg_blocks = 3;
        assert!(r.check().is_err(), "no three-block instantiation exists");
    }

    /// The rerank's grid IS the pair (rows, candidates), which is how the census read
    /// SABER_DRAFT_RERANK=80 off a trace that never mentions the variable.
    #[test]
    fn the_rerank_grid_is_rows_by_candidates() {
        let r = rules();
        let d = resolve(
            DflashOp::RerankExact,
            DflashShape::rerank(16, 4096, 5120, 80),
            &r,
        )
        .expect("prod's rerank shape");
        assert_eq!(d.plugin, "dflash_head");
        assert_eq!(d.grid, [16, 80, 1]);
        assert_eq!(d.block, [128, 1, 1]);
        // 5 buffers + k, stride_w, r, n -- `n` is the vocabulary width, which only the `_scatter`
        // variant reads, and it is in the signature of both so one descriptor serves both.
        assert_eq!(d.kernarg_bytes, 56);
    }

    /// N6: the fp8-record rerank resolves at the serve's shape to its own symbol, on the bf16
    /// arm's grid, with `w8` + `ws` where the bf16 arm has `w` — and the bf16 arm (the CONTROL)
    /// still binds no scale, so a plan cannot confuse the two record kinds.
    #[test]
    fn the_w8_rerank_is_the_bf16_arms_grid_over_the_fp8_record() {
        let r = rules();
        let shape = DflashShape::rerank(16, 124_160, 5120, 80);
        let w8 = resolve(DflashOp::RerankExactW8, shape, &r).expect("the w8 rerank");
        let bf = resolve(DflashOp::RerankExactBf16, shape, &r).expect("the bf16 rerank");
        assert_eq!(w8.kernel, "dflash_rerank_exact_w8_k512");
        assert_eq!(w8.plugin, "dflash_head");
        assert_eq!((w8.grid, w8.block), (bf.grid, bf.block));
        let names = |d: &crate::rules::LaunchDescriptor| {
            d.args.iter().map(|a| a.name.clone()).collect::<Vec<_>>()
        };
        assert_eq!(
            names(&w8),
            ["x", "w8", "ws", "idx", "out", "k", "stride_w", "r", "n"]
        );
        assert!(bf.arg("ws").is_none() && bf.arg("w8").is_none());
        assert!(w8.arg("w").is_none());
        assert_eq!(w8.kernarg_bytes, 56);
    }

    /// EVERY SCALAR slot a `dflash` signature declares carries a VALUE, never a buffer — `I32`
    /// as well as `F32`.
    ///
    /// `resolve` fills a slot the shape table does not name with `ArgValue::Buffer`, which is
    /// right for a pointer and silent corruption for a four-byte scalar: the launcher writes
    /// eight bytes of address into it and every kernarg after it moves. `rms_norm` and
    /// `conv_add_rms` both shipped without their `eps` and the attention without its `scale`, and
    /// no gate could see any of them, because every gate those kernels had launched them from
    /// Python with hand-packed kernargs — the packing under test here is the one only the real
    /// launcher uses (workstream F round 21, found by running a proposal through it).
    ///
    /// Round 22 widened it from `F32` to every scalar kind, because the fourth one found was an
    /// `I32`: the attention's `bt_stride`, which is a runtime fact of the block table and so had
    /// no knob to come from. A name list would have had to be extended each time; asking the
    /// SIGNATURE which slots are scalars cannot go stale.
    ///
    /// The CONTROL is the count: if this test could pass by finding no scalar slots at all it
    /// would be worthless, so it also insists there are some of each kind.
    #[test]
    fn every_f32_kernarg_is_a_value_and_not_a_pointer() {
        let r = rules();
        let cases: &[(DflashOp, DflashShape)] = &[
            (DflashOp::RmsNorm, DflashShape::new(8, 5120, 5120)),
            (
                DflashOp::ConvAddRmsOpen,
                DflashShape::conv(8, 5120, 320, 8, 1),
            ),
            (
                DflashOp::ConvAddRmsClose,
                DflashShape::conv(8, 5120, 320, 8, 1),
            ),
            (DflashOp::CtxRmsNorm, DflashShape::ctx_norm(8, 5120, 8)),
            (DflashOp::CtxRope, DflashShape::ctx_rope(40, 4, 128, 128)),
            (
                DflashOp::AttnFp8Kv,
                DflashShape::attn(8, 16, 4096, 4, 1648, 8),
            ),
            (
                DflashOp::AttnBf16Kv,
                DflashShape::attn(8, 16, 4096, 4, 16, 64),
            ),
            (DflashOp::KvAppendFp8, DflashShape::new(8, 4, 128)),
            (DflashOp::GemmBlockscale, DflashShape::new(8, 3072, 5120)),
            (DflashOp::SelectorWalk, DflashShape::rerank(1, 7, 16, 16)),
            (
                DflashOp::SelectorScore,
                DflashShape::selector(7, 7, 256, 16),
            ),
            (
                DflashOp::RerankExact,
                DflashShape::rerank(16, 4096, 5120, 80),
            ),
            (DflashOp::DraftHeadInt2, DflashShape::new(16, 124_160, 5120)),
            (DflashOp::BgemmBf16, DflashShape::new(8, 1280, 5120)),
            (DflashOp::SiluQuant, DflashShape::new(8, 8704, 17408)),
            (
                DflashOp::GroupedConvQuant,
                DflashShape::conv(8, 5120, 320, 8, 0),
            ),
            (
                DflashOp::SelectorTopk,
                DflashShape::rerank(7, 124_160, 1, 16),
            ),
        ];
        let (mut f32_slots, mut i32_slots) = (0usize, 0usize);
        for (op, shape) in cases {
            let d = resolve(*op, *shape, &r).expect("a shape the plan runs");
            let sig = abi::signature_for(&d.kernel).expect("a declared signature");
            for slot in &d.args {
                let kind = sig
                    .iter()
                    .find(|a| a.name == slot.name)
                    .map(|a| a.kind)
                    .expect("every slot comes from the signature");
                if kind == abi::ArgKind::Buffer {
                    continue;
                }
                assert!(
                    !matches!(slot.value, crate::rules::ArgValue::Buffer { .. }),
                    "{}: `{}` is a {kind:?} kernarg and resolve left it a POINTER",
                    op.as_str(),
                    slot.name
                );
                match kind {
                    abi::ArgKind::F32 => f32_slots += 1,
                    abi::ArgKind::I32 => i32_slots += 1,
                    _ => {}
                }
            }
        }
        assert!(
            f32_slots >= 4 && i32_slots >= 40,
            "the case list must actually contain scalars of both kinds; found {f32_slots} F32 \
             and {i32_slots} I32, so this test could otherwise pass on a signature set with none"
        );
    }

    /// The `bt_stride` refusal has teeth: the attention will not resolve without it, which is
    /// what keeps a caller from launching with an address in a four-byte slot.
    #[test]
    fn the_attention_refuses_a_block_table_stride_of_zero() {
        let r = rules();
        for op in [DflashOp::AttnBf16Kv, DflashOp::AttnFp8Kv] {
            let e = resolve(op, DflashShape::attn(8, 16, 608, 4, 1648, 0), &r)
                .expect_err("a zero stride is refused");
            assert!(e.to_string().contains("bt_stride is 0"), "{e}");
            resolve(op, DflashShape::attn(8, 16, 608, 4, 1648, 8), &r)
                .expect("the same shape WITH a stride resolves");
        }
    }

    #[test]
    fn the_shapes_the_instantiations_cannot_serve_are_refused_by_name() {
        let r = rules();
        let bad = |op, sh: DflashShape| resolve(op, sh, &r).expect_err("refused").to_string();
        let gemm = DflashOp::GemmBlockscale;
        assert!(bad(gemm, DflashShape::new(0, 5120, 2048)).contains("nothing to do"));
        assert!(bad(gemm, DflashShape::new(8, 5120, 0)).contains("nothing to do"));
        assert!(bad(gemm, DflashShape::new(8, 5184, 2048)).contains("not a multiple of 128"));
        assert!(bad(gemm, DflashShape::new(8, 131_072, 2048)).contains("exceeds max_n"));
        assert!(bad(gemm, DflashShape::new(8, 5120, 2112)).contains("scale groups"));
        let head = DflashOp::DraftHeadInt2;
        // A SHORTER m is refused too, not padded: the kernel writes all 16 rows.
        assert!(bad(head, DflashShape::new(8, 124_160, 5120)).contains("writes exactly 16 rows"));
        assert!(bad(head, DflashShape::new(16, 124_128, 5120)).contains("not a multiple of 64"));
        assert!(bad(head, DflashShape::new(16, 124_160, 5248)).contains("not a multiple of 512"));
        let rr = DflashOp::RerankExact;
        assert!(bad(rr, DflashShape::rerank(16, 4096, 5120, 0)).contains("outside 1..="));
        assert!(bad(rr, DflashShape::rerank(16, 4096, 5120, 2048)).contains("outside 1..="));
        assert!(bad(rr, DflashShape::rerank(16, 4096, 5248, 80)).contains("multiple of 512"));
    }

    /// The attention's launch IS the reference's: `q.shape[0] // BLOCK_Q + num_seqs` workgroups of
    /// four waves, one wave per kv head. The census traced 384x4x1 at 128 threads = 3 x 4.
    #[test]
    fn the_attention_resolves_the_grid_the_census_traced() {
        let r = rules();
        for op in [DflashOp::AttnBf16Kv, DflashOp::AttnFp8Kv] {
            let d =
                resolve(op, DflashShape::attn(8, 16, 608, 4, 16, 38), &r).expect("prod's shape");
            assert_eq!(d.plugin, "dflash_attn");
            assert_eq!(
                d.grid,
                [3, 1, 1],
                "8 rows / BLOCK_Q 4, plus the one that early-returns; 4 kv heads = 1 wave group"
            );
            assert_eq!(d.block, [128, 1, 1]);
            // 8 buffers (k_scale and v_scale separately, as prod passes them) + 7 scalars.
            assert_eq!(d.kernarg_bytes, 92);
            // TP=1 is 32 q heads over EIGHT kv heads, which is two wave groups. The y extent is
            // the only thing that changes, and at 4 kv heads it is 1 — so the launch every gate
            // took is byte for byte the same one.
            let tp1 = resolve(op, DflashShape::attn(8, 32, 608, 8, 1648, 8), &r)
                .expect("the drafter's unsharded shape");
            assert_eq!(
                tp1.grid,
                [3, 2, 1],
                "8 kv heads over 4 waves is two grid rows"
            );
            assert_eq!(tp1.block, [128, 1, 1]);
        }
        // A head count that is not GQA 4 over the kv heads is a different model, not a wider launch.
        let e = resolve(
            DflashOp::AttnBf16Kv,
            DflashShape::attn(8, 16, 608, 8, 16, 38),
            &r,
        )
        .expect_err("GQA 2");
        assert!(format!("{e}").contains("is not a GQA of 4"), "{e}");
    }

    /// The KV append is compiled for the drafter's head_dim; the target's 256 is a different
    /// instantiation and is refused rather than launched half-width.
    /// Prod's drafter KV cache is `[1212, 1648, 4, 128]` -- 1648 tokens per block, because
    /// `--mamba-cache-mode align` sizes the attention page to the GDN one. So the block size is a
    /// RUNTIME value the shape carries, and the rules' knob is only a default; putting the knob in
    /// the descriptor would index prod's block table wrongly.
    #[test]
    fn the_block_size_comes_from_the_shape_not_the_knob() {
        let r = rules();
        let val = |d: &LaunchDescriptor| match d.arg("block_size").expect("block_size").value {
            ArgValue::I32 { ival } => ival,
            _ => panic!("not an immediate"),
        };
        let d = resolve(
            DflashOp::AttnFp8Kv,
            DflashShape::attn(8, 16, 4096, 4, 1648, 8),
            &r,
        )
        .expect("prod's cache geometry");
        assert_eq!(
            val(&d),
            1648,
            "prod's mamba-aligned page, not the knob's 16"
        );
        // Left unset, the knob is the default -- which is what a synthetic gate uses. `bt_stride`
        // has no such default and must still be given.
        let d = resolve(
            DflashOp::AttnFp8Kv,
            DflashShape::attn(8, 16, 4096, 4, 0, 8),
            &r,
        )
        .expect("the default");
        assert_eq!(val(&d), 16);
    }

    #[test]
    fn the_kv_append_is_bound_to_the_drafters_head_dim() {
        let r = rules();
        let d = resolve(DflashOp::KvAppendFp8, DflashShape::new(8, 4, 128), &r).expect("h128");
        assert_eq!(d.plugin, "dflash_kv");
        assert_eq!(d.grid, [8, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        let e = resolve(DflashOp::KvAppendBf16, DflashShape::new(8, 4, 256), &r)
            .expect_err("the target's head_dim");
        assert!(format!("{e}").contains("head_dim 256"), "{e}");
    }

    /// `k` is the per-thread capacity, and that equality is what makes one pass exact -- so a larger
    /// k is a DIFFERENT ALGORITHM (a radix select), which the refusal says in as many words.
    #[test]
    fn the_topk_refuses_a_k_it_cannot_be_exact_for() {
        let r = rules();
        let d = resolve(DflashOp::SelectorTopk, DflashShape::rerank(7, 0, 0, 16), &r);
        assert!(d.is_err(), "n = 0 has nothing to select from");
        let d = resolve(
            DflashOp::SelectorTopk,
            DflashShape::rerank(7, 124_160, 124_160, 16),
            &r,
        )
        .expect("the selector's own shape");
        assert_eq!(d.plugin, "dflash_topk");
        assert_eq!(d.grid, [7, 1, 1], "one workgroup per mask position");
        assert_eq!(d.block, [256, 1, 1]);
        let e = resolve(
            DflashOp::SelectorTopk,
            DflashShape::rerank(16, 15520, 15520, 80),
            &r,
        )
        .expect_err("the draft head's k");
        assert!(format!("{e}").contains("radix select"), "{e}");
    }

    #[test]
    fn a_rules_file_that_contradicts_a_code_object_is_refused() {
        for (from, to, want) in [
            ("bn = 128", "bn = 256", "the only instantiation is"),
            ("bn = 64", "bn = 32", "the only instantiations are"),
            ("kcand = 8", "kcand = 4", "the only instantiations are"),
            (
                "rank = 256",
                "rank = 128",
                "the only instantiation is (256, 16, 32)",
            ),
            (
                "taps = 2",
                "taps = 3",
                "the only instantiation is (2, 16, 128, 4, 128)",
            ),
            (
                "k_multiple = 16",
                "k_multiple = 8",
                "the only instantiation is (16, 16, 128, 16)",
            ),
        ] {
            let e = DflashRules::from_toml(&PROD_RULES.replace(from, to))
                .expect_err("a tile the kernels are not compiled for");
            assert!(format!("{e}").contains(want), "{from} -> {to}: {e}");
        }
    }

    /// THE BF16 ROUTE CANNOT BIND THE TILE-16 INSTANTIATION, and the shipped rules bind the
    /// tile-32 one.
    ///
    /// `_get_tile_size(..., is_prefill=True)` is 32 at every KV width, and the 2D path -- the only
    /// one a drafter with 8 query rows takes -- is the only reader of it. The bf16 row bound
    /// `..._t16` from the day it was written and no gate could see it, because the drafter's cache
    /// is e4m3 and nothing runs that route; the same substitution on the fp8 side is worth 14,306
    /// differing bytes of 983,040 (F24). So the refusal is by NAME, on the rules, off the card.
    ///
    /// The CONTROL is the second half: a rules file that binds `_t16` must be refused, and the
    /// message must name the op and the symbol.
    #[test]
    fn the_bf16_attention_route_cannot_bind_the_tile_16_instantiation() {
        let r = rules();
        assert_eq!(
            r.row(DflashOp::AttnBf16Kv)
                .expect("a bf16 attention row")
                .symbol,
            "dflash_attn_h128_gqa4_bf16kv_t32"
        );
        assert_eq!(
            r.row(DflashOp::AttnFp8Kv)
                .expect("an fp8 attention row")
                .symbol,
            "dflash_attn_h128_gqa4_fp8kv_t32"
        );
        assert_eq!((r.attn.tile_bf16, r.attn.tile_fp8), (TILE_2D, TILE_2D));
        let e = DflashRules::from_toml(&PROD_RULES.replace(
            "dflash_attn_h128_gqa4_bf16kv_t32",
            "dflash_attn_h128_gqa4_bf16kv_t16",
        ))
        .expect_err("the 3D path's tile on the 2D path");
        let e = format!("{e}");
        assert!(e.contains("attn_bf16kv"), "{e}");
        assert!(e.contains("dflash_attn_h128_gqa4_bf16kv_t16"), "{e}");
        assert!(e.contains("TILE_SIZE_PREFILL"), "{e}");
        // ... and the knob alone cannot move it either: 16 is refused before the symbols are read.
        let e = DflashRules::from_toml(&PROD_RULES.replace("tile_bf16 = 32", "tile_bf16 = 16"))
            .expect_err("a tile the 2D path does not take");
        assert!(
            format!("{e}").contains("TILE_SIZE_PREFILL at BOTH KV widths"),
            "{e}"
        );
    }

    /// A row may only claim a code object the crate actually ships; a typo here would otherwise
    /// produce a registry row no artifact owns, which is the bug class
    /// `every_registry_row_is_attributed_to_exactly_one_plugin` exists for.
    #[test]
    fn a_row_in_an_unknown_plugin_is_refused() {
        let e = DflashRules::from_toml(
            &PROD_RULES.replace(r#"plugin = "dflash_head""#, r#"plugin = "dflash_heads""#),
        )
        .expect_err("an unknown code object");
        assert!(format!("{e}").contains("which is not one of"), "{e}");
    }
}
