//! Launch logic as data.
//!
//! `ref/ref_mxfp4_fp8.hip` decides, in C++ and from `getenv`, which of ~140
//! kernel instantiations runs for a given (M, N, K): `split_k_for` (:1097), `decode_bk64` (:1089),
//! the M>64 band ladder and the TN/LBK choice inside `launch_impl` (:1136) / `launch_at_impl`
//! (:1660). Here those are rows in [`LaunchRules`] (see `rules/mxfp4_gfx1201.toml`) and [`resolve`]
//! is a pure function over them: no branch in this file exists that is not a table row, which is
//! invariant 6 ("anything measured is data, never a control-flow ladder").
//!
//! [`resolve`] returns a [`LaunchDescriptor`] — kernel symbol, grid, block, the kernarg byte
//! layout with its values bound, the workspaces the launch needs and their kinds, and the launch
//! count. It never touches HIP; `saber-exec` turns a descriptor into a dispatch.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::abi::{self, ArgKind, SigArg};
use crate::{Error, Result};

// ---------------------------------------------------------------------------- rules (data)

#[derive(Debug, Clone, Deserialize)]
pub struct LaunchRules {
    pub schema_version: u32,
    pub name: String,
    pub knobs: Knobs,
    pub split_k_ladder: Vec<u32>,
    pub split_k_fill: Vec<FillRow>,
    pub decode_bk: Vec<BkRow>,
    pub decode_band: Vec<BandRow>,
    pub k_align_override: KAlign,
    pub folded_tn: Vec<TnRow>,
    pub atiled_lbk: Vec<LbkRow>,
    /// The A-tiled tile's height in waves (G1). No matching row = the reference's WM = 4.
    #[serde(default)]
    pub atiled_wm: Vec<WmRow>,
    /// The folded tile's mid-M heights (G5), bound only while [`Knobs::folded_mt`] is on. No
    /// matching row = the reference's 256-row tile.
    #[serde(default)]
    pub folded_mt: Vec<MtRow>,
    /// The A-tiled tile's mid-M heights (G6), bound only while [`Knobs::atiled_mid`] is on, below
    /// `a_tiled_min_m`. No matching row = the reference's A-tiled tile.
    #[serde(default)]
    pub atiled_mt: Vec<AtiledMtRow>,
    /// The skinny-N body (G6), bound only while [`Knobs::skinny`] is on. No matching row = the
    /// family the layout and the band pick.
    #[serde(default)]
    pub skinny: Vec<SkinnyRow>,
    pub decode_direct_control: DirectControl,
    #[serde(default)]
    pub decode_fused_silu: Vec<FusedSiluRow>,
    pub workspace: Workspaces,
}

/// The one decode instantiation the prod launcher cannot reach: `DIRECT=false` at `DKS==1`, kept
/// so the "writes C directly" fast path can be TESTED against the reduce path rather than
/// asserted. [`gemm_decode_indirect`] resolves it and refuses any shape off its exact cell.
/// One compiled cell of the fused silu route. See [`gemm_decode_fused_silu`].
#[derive(Debug, Clone, Deserialize)]
pub struct FusedSiluRow {
    pub bk: u32,
    pub ks: u32,
    pub tm: u32,
    pub kernel: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DirectControl {
    pub kernel: String,
    pub bk: u32,
    pub ks: u32,
    pub tm: u32,
    pub wperm_required: bool,
    pub nt_required: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Knobs {
    pub decode_max_m: u32,
    pub a_tiled_min_m: u32,
    pub dec_max_n: u32,
    pub dec_max_tm: u32,
    pub dec_mtile: u32,
    pub dec_dwn: u32,
    pub dec_ks: u32,
    pub wperm: bool,
    pub decode_nt: bool,
    pub epifast: bool,
    pub k_multiple: u32,
    pub atiled_k_multiple: u32,
    pub partition_align: u32,
    pub base_bm: u32,
    pub base_bn: u32,
    pub base_block: u32,
    pub folded_bmf: u32,
    pub folded_wn: u32,
    pub folded_block: u32,
    pub quant_large_min_m: u32,
    pub quant_small_block: u32,
    pub quant_large_block: u32,
    pub quant_max_k: u32,
    pub silu_max_n: u32,
    pub silu_large_n: u32,
    pub gdn_max_n: u32,
    pub gdn_n_multiple: u32,
    /// SABER-AUTHORED (G2): `gdn_norm_quant` writes the fragment-tiled layout at
    /// [`a_tiled`] M (`gdn_norm_quant_t{block}_tiled1`, `hip/mxfp4_gdn_norm_tiled.h`), so the GDN
    /// `out_proj` takes the A-tiled prefill GEMM like every other prefill linear. Absent or false
    /// is the reference's law: one row-major instantiation, the folded GEMM at every M.
    #[serde(default)]
    pub gdn_norm_tiled: bool,
    /// SABER-AUTHORED (G5): the folded prefill GEMM takes the mid-M tile heights of
    /// `[[folded_mt]]` (`hip/mxfp4_gemm_folded_mt.hip`, one summation class with the reference's
    /// tile). Absent or false is the reference's 256-row tile at every M. OFF in the committed
    /// table: `saber serve --engine-mid-chunk-gemm` turns it on for a plan's rules.
    #[serde(default)]
    pub folded_mt: bool,
    /// SABER-AUTHORED (G6): **the layout law below `a_tiled_min_m`.** With this on, every fused
    /// quant producer writes the fragment-tiled A layout from [`Knobs::atiled_mid_min_m`] rows up
    /// (not only from `a_tiled_min_m`), and the GEMMs reading those edges take the A-tiled family
    /// (`[[atiled_mt]]`'s mid-M heights, else the reference's A-tiled tile). One summation class
    /// with the folded tile it replaces (G2). OFF in the committed table: `saber serve
    /// --engine-mid-chunk-atiled` turns it on for a plan's rules; [`ATiledLaw`] is how every
    /// reader asks.
    #[serde(default)]
    pub atiled_mid: bool,
    /// The mid band's edge (G6): the lowest M whose producers write the tiled layout while
    /// [`Knobs::atiled_mid`] is on. Must exceed `decode_max_m` (the decode epilogues write
    /// row-major) and sit below `a_tiled_min_m`. 0 = no mid band.
    #[serde(default)]
    pub atiled_mid_min_m: u32,
    /// SABER-AUTHORED (G6): the skinny-N body for the 48-wide GDN gate, `[[skinny]]` (one wave per
    /// 16-row fragment walking all of K in the reference's order; one summation class). OFF in the
    /// committed table: `saber serve --engine-skinny-gate` turns it on.
    #[serde(default)]
    pub skinny: bool,
    /// SABER-AUTHORED (CMP27E, an ARM): the b8 decode cells in [`DECODE_PF_CELLS`] and the
    /// `bk128 tm4` decode pair take the register-prefetch twins (`hip/mxfp4_gemm_decode_pf.hip`:
    /// same LDS bytes, same WMMA order, the pair's short half dispatched first; 0 bytes differing).
    /// OFF in the committed table: `saber serve --engine-decode-pf` turns it on.
    #[serde(default)]
    pub decode_pf: bool,
    /// SABER-AUTHORED (CMP27T, an ARM): the conc-4 decode cells in [`DECODE_PF_TM2_CELLS`] and the
    /// `bk128 tm2` decode pair take the same register-prefetch body at 32-row tiles (same LDS
    /// bytes, same WMMA order, the pair's short half first; 0 bytes differing). OFF in the
    /// committed table: `saber serve --engine-c4-arms pf2`.
    #[serde(default)]
    pub decode_pf_tm2: bool,
    /// SABER-AUTHORED (CMP27X, an ARM): the prefetch cells in [`DECODE_PF_DWN4_CELLS`] at the
    /// N in [`DECODE_PF_DWN4_N`] launch the `_dwn4` twin (4 waves = 64 columns a workgroup, 2x the
    /// workgroups; each wave's K walk and the split's fold unchanged: 0 bytes differing). The
    /// split `ks` is still the DWN 8 grid's. OFF in the committed table: `--engine-step-arms dwn4`.
    #[serde(default)]
    pub decode_pf_dwn4: bool,
    /// SABER-AUTHORED (CMP27F, an ARM): the target's fp8 head at `HEAD_M64_MIN_ROWS..=64` rows
    /// takes the one-pass 64-row tile (`hip/saber_head_w8_m64.hip`: A staged once a workgroup in
    /// LDS; the same fragments in the same k order as the 32-row band, 0 bytes differing) instead
    /// of two 32-row passes. OFF in the committed table: `saber serve --engine-head-m64`.
    #[serde(default)]
    pub head_m64: bool,
    /// SABER-AUTHORED (CMP27L, an ARM): the b1 tree verify's conv takes the rows-parallel twin
    /// (`p27_gdn_conv_tree_rows_w4_h128_t10`: one wave a tree row, the cache rewrite after a
    /// workgroup barrier; q / k / v and the cache are the serial kernel's bytes). OFF in the
    /// committed table: `saber serve --engine-b1-arms conv`.
    #[serde(default)]
    pub gdn_conv_rows: bool,
    /// SABER-AUTHORED (CMP27L, an ARM): the target's fp8 head at 1..16 rows (the b1 verify)
    /// takes the 32-row band's nontemporal 16-byte body (`dflash_bgemm_w8_m32n16_w128_u16_nt`,
    /// one row tile) instead of the 16-row band; 0 bytes differing at M 1/4/8/10/16 on both
    /// ranks' heads (`tools/capture/cmp27l/probe.py`). OFF in the committed table: `saber serve
    /// --engine-b1-arms head`.
    #[serde(default)]
    pub head_b1_nt: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FillRow {
    pub m_max: Option<u32>,
    pub fill: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BkRow {
    pub tm: Option<u32>,
    pub ks: Option<u32>,
    pub bk: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BandRow {
    pub m_min: Option<u32>,
    pub m_max: Option<u32>,
    pub nblk_min: Option<u32>,
    pub nblk_max: Option<u32>,
    pub k_min: Option<u32>,
    pub k_max: Option<u32>,
    pub ks: Option<u32>,
    pub bk: Option<u32>,
    #[serde(default)]
    pub ks_from_fill: bool,
    #[serde(default)]
    pub bk_from_tm: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KAlign {
    pub k_modulus: u32,
    pub bk: u32,
    pub ks_from: u32,
    pub ks_to: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TnRow {
    pub m_min: Option<u32>,
    pub n_min: Option<u32>,
    /// Which GEMM family the row is for: `true` the A-tiled prefill kernel, `false` the folded
    /// one, absent both. The two stage A differently and do NOT share a TN optimum -- at one and
    /// the same (M, N, K) the measurement picks opposite tiles (see `rules/mxfp4_gfx1201.toml`),
    /// so a row that does not say which kernel it describes cannot be right for both.
    pub a_tiled: Option<bool>,
    pub tn: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LbkRow {
    pub tn: u32,
    pub lbk: u32,
}

/// One compiled cell of the Saber-authored A-tiled tile (`mxfp4_atiled_wm.h`): at this TN, this
/// M band and the fragment-order W layout, the tile is `wm` waves tall instead of the reference's
/// four. First match wins; no match is WM = 4, the reference's own kernel.
#[derive(Debug, Clone, Deserialize)]
pub struct WmRow {
    pub tn: u32,
    pub m_min: Option<u32>,
    pub m_max: Option<u32>,
    /// The weight layout the row's symbol is compiled for (`true` = fragment order, the loader's
    /// `wperm`); a row never binds a layout it has no symbol for. Named apart from the knob so a
    /// blanket `wperm = true -> false` edit of the table moves the knob and not this.
    pub w_fragment_order: bool,
    pub wm: u32,
    /// Which compiled body of that height (G3): absent is G1's `_wm{wm}`; `"s"` is `_wm{wm}s`
    /// (SADDR W / Ws addressing, no VCC carry chain in the loop), `"sp"` is `_wm{wm}sp` (the same
    /// plus the W slab double-buffered in LDS). All three are one summation (`summation_class`).
    #[serde(default)]
    pub variant: Option<String>,
}

impl WmRow {
    /// The symbol suffix this row binds: `_wm8`, `_wm8s`, `_wm8sp`.
    #[must_use]
    pub fn suffix(&self) -> String {
        format!("_wm{}{}", self.wm, self.variant.as_deref().unwrap_or(""))
    }
}

/// One compiled cell of the Saber-authored folded tile at a mid-M height (G5,
/// `hip/mxfp4_gemm_folded_mt.hip`): in this M band and N range, on the fragment-order W layout at
/// TN = 2, the tile is `tm` 16-row fragments per wave times `wm` waves. First match wins; no match
/// is the reference's 256-row tile.
#[derive(Debug, Clone, Deserialize)]
pub struct MtRow {
    pub m_min: Option<u32>,
    pub m_max: Option<u32>,
    pub n_min: Option<u32>,
    pub n_max: Option<u32>,
    pub tm: u32,
    pub wm: u32,
}

impl MtRow {
    fn matches(&self, m: u32, n: u32) -> bool {
        self.m_min.is_none_or(|x| m >= x)
            && self.m_max.is_none_or(|x| m <= x)
            && self.n_min.is_none_or(|x| n >= x)
            && self.n_max.is_none_or(|x| n <= x)
    }

    /// The tile's rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.tm * self.wm * 16
    }
}

/// The plugin the mid-M folded tiles live in (G5): their own code object, so the declared
/// `mxfp4_gemm_folded` keeps its bytes.
pub const FOLDED_MT_PLUGIN: &str = "mxfp4_gemm_folded_mt";

/// CMP27E: the register-prefetch twins' code object (the `decode_pf` knob), its own so the
/// shipped `mxfp4_gemm_decode` keeps its bytes.
pub const DECODE_PF_PLUGIN: &str = "mxfp4_gemm_decode_pf";

/// CMP27E: the `(bk, ks, tm)` decode cells (at `wperm1_nt1`) the prefetch twins are compiled for --
/// the b8 verify's cells (out_proj / o_proj / qkv / down at `bk128 ks2 tm4`, gate_up at
/// `bk64 ks1 tm4`). The pair twin is `bk128 tm4` (either split pair).
pub const DECODE_PF_CELLS: [(u32, u32, u32); 2] = [(128, 2, 4), (64, 1, 4)];

/// The pair twin's symbol (CMP27E).
pub const DECODE_PF_PAIR: &str = "mxfp4_gemm_decode_pair_pf_bk128_tm4_wperm1_nt1";

/// CMP27T: the `(bk, ks, tm)` conc-4 decode cell (at `wperm1_nt1`) the prefetch body is compiled
/// for under the `decode_pf_tm2` knob -- out_proj / o_proj / qkv / down at `bk128 ks2 tm2` (25..32
/// rows). gate_up's `bk128 ks1 tm2` and the tm2 pair measured slower and have no twin.
pub const DECODE_PF_TM2_CELLS: [(u32, u32, u32); 1] = [(128, 2, 2)];

/// CMP27X: the prefetch cells whose `_dwn4` twin is compiled (`hip/mxfp4_gemm_decode_pf.hip`),
/// taken under the `decode_pf_dwn4` knob at the N in [`DECODE_PF_DWN4_N`] only.
pub const DECODE_PF_DWN4_CELLS: [(u32, u32, u32); 2] = [(128, 2, 2), (128, 2, 4)];

/// CMP27X: the output widths the `_dwn4` twin is routed at (the TP 2 rank's out_proj / o_proj /
/// down: 40 DWN 8 workgroups a split, parallelism-starved; card-1 probe `tools/capture/cmp27x`).
pub const DECODE_PF_DWN4_N: [u32; 1] = [5120];

/// One compiled cell of the A-tiled tile at a mid-M height (G6, `hip/mxfp4_gemm_mid.hip`,
/// `hip/mxfp4_atiled_mt.h`): in this M band and N range, on the fragment-order W layout, the
/// A-tiled tile is `tm` 16-row fragments per wave times `wm` waves, TN = `tn`, K slab `lbk`,
/// `pipe` = the W slab's global loads double-buffered. First match wins; no match is the
/// reference's A-tiled tile.
#[derive(Debug, Clone, Deserialize)]
pub struct AtiledMtRow {
    pub m_min: Option<u32>,
    pub m_max: Option<u32>,
    pub n_min: Option<u32>,
    pub n_max: Option<u32>,
    pub tn: u32,
    pub lbk: u32,
    pub tm: u32,
    pub wm: u32,
    #[serde(default)]
    pub pipe: bool,
}

impl AtiledMtRow {
    fn matches(&self, m: u32, n: u32) -> bool {
        self.m_min.is_none_or(|x| m >= x)
            && self.m_max.is_none_or(|x| m <= x)
            && self.n_min.is_none_or(|x| n >= x)
            && self.n_max.is_none_or(|x| n <= x)
    }

    /// The tile's rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.tm * self.wm * 16
    }

    /// The row's symbol in [`MID_PLUGIN`].
    #[must_use]
    pub fn kernel(&self) -> String {
        format!(
            "mxfp4_gemm_atiled_mt_tn{}_lbk{}_wperm1_t{}w{}{}",
            self.tn,
            self.lbk,
            self.tm,
            self.wm,
            if self.pipe { "p" } else { "" }
        )
    }
}

/// One compiled cell of the skinny-N body (G6, `hip/mxfp4_skinny.h`): at N <= `n_max` in this M
/// band and on this A layout, one wave per 16-row fragment and `tns` 16-column tiles, `wpb` waves
/// a workgroup, `unr` k-steps a load batch. First match wins.
#[derive(Debug, Clone, Deserialize)]
pub struct SkinnyRow {
    pub m_min: Option<u32>,
    pub m_max: Option<u32>,
    pub n_max: u32,
    /// The A layout the row's symbol reads: `true` fragment order, `false` row-major.
    pub a_tiled: bool,
    pub tns: u32,
    pub wpb: u32,
    pub unr: u32,
}

impl SkinnyRow {
    fn matches(&self, m: u32, n: u32, tiled: bool) -> bool {
        self.m_min.is_none_or(|x| m >= x)
            && self.m_max.is_none_or(|x| m <= x)
            && n <= self.n_max
            && self.a_tiled == tiled
    }

    /// The row's symbol in [`MID_PLUGIN`].
    #[must_use]
    pub fn kernel(&self) -> String {
        format!(
            "mxfp4_gemm_skinny_n{}{}_w{}_u{}_wperm1",
            self.tns,
            if self.a_tiled { "t" } else { "r" },
            self.wpb,
            self.unr
        )
    }
}

/// The plugin the G6 bodies live in: the A-tiled tile at mid-M heights and the skinny-N body. Its
/// own code object, so `mxfp4_gemm_atiled` keeps its bytes and every autotune key its digest.
pub const MID_PLUGIN: &str = "mxfp4_gemm_mid";

/// **The layout law**: at which M the fused quant producers write the fragment-tiled A layout
/// (and so at which M a GEMM reading a fused producer's edge takes the A-tiled family).
///
/// Until G6 this was one number, `a_tiled_min_m`. G6 adds a mid band below it, on under the
/// `atiled_mid` knob: from `atiled_mid_min_m` rows up. Every reader asks THIS, over the rules the
/// plan was built with — the producers (`producer_writes_tiled`), the GEMM (`gemm`'s A-tiled
/// branch), the attention glue's gate producer (`glue::resolve_under`), and the model graph's edge
/// layouts (`LayoutRule::resolve_under`) — so a knob flipped on one table moves every reader at
/// once (the P6 lesson: a threshold with copies is a layout mismatch waiting to happen).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ATiledLaw {
    /// `a_tiled_min_m`; 0 = row-major at every M (the sweepers' `ALayout::RowMajor`).
    pub min_m: u32,
    /// The mid band's edge when the `atiled_mid` knob is on.
    pub mid_min_m: Option<u32>,
}

impl ATiledLaw {
    /// The law a rules table states.
    #[must_use]
    pub fn of(rules: &LaunchRules) -> Self {
        let k = &rules.knobs;
        ATiledLaw {
            min_m: k.a_tiled_min_m,
            mid_min_m: (k.atiled_mid && k.atiled_mid_min_m > 0).then_some(k.atiled_mid_min_m),
        }
    }

    /// The committed table's law: [`a_tiled_min_m`] and no mid band.
    #[must_use]
    pub fn committed() -> Self {
        static LAW: LazyLock<ATiledLaw> = LazyLock::new(|| {
            ATiledLaw::of(
                &LaunchRules::prod_mxfp4().expect("the committed mxfp4 rules are compiled in"),
            )
        });
        *LAW
    }

    /// Does a fused producer write the fragment-tiled layout at `m` under this law?
    #[must_use]
    pub fn tiled(self, m: u32) -> bool {
        self.min_m > 0 && (m >= self.min_m || self.mid_min_m.is_some_and(|x| m >= x))
    }

    /// The lowest M that is tiled under this law (0 = none).
    #[must_use]
    pub fn edge(self) -> u32 {
        if self.min_m == 0 {
            0
        } else {
            self.mid_min_m.map_or(self.min_m, |x| x.min(self.min_m))
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Workspaces {
    pub decode_partial: WorkspaceRule,
    pub decode_counter: WorkspaceRule,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceRule {
    pub kind: WorkspaceKind,
    pub capacity_bytes: u64,
}

/// Workspace kinds, as the plugin ABI defines them. "N bytes" cannot express the difference:
/// the split-K partial buffer is aliasable between launches, the block counter must be zeroed
/// exactly once and survive graph replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceKind {
    Scratch,
    Persistent,
    IpcPeerVisible,
}

const PROD_RULES: &str = include_str!("../rules/mxfp4_gfx1201.toml");

/// The same text, for the plan input hash: the policy this module enforces.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

impl LaunchRules {
    /// The rules prod runs today (`serve-mxfp4.sh` defaults, TP=2, MAXSEQS=8).
    pub fn prod_mxfp4() -> Result<Self> {
        Self::from_toml(PROD_RULES)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: LaunchRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    /// Invariants the prod launcher asserts at import; a rules file that breaks one is refused
    /// here rather than mis-routing a shape at run time.
    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        let k = &self.knobs;
        if k.a_tiled_min_m != 0 && k.a_tiled_min_m <= 512.max(k.decode_max_m) {
            return Err(Error::Rule(
                "a_tiled_min_m must exceed both 512 and decode_max_m (the decode epilogue and the \
                 exact_nq epilogue both write row-major)"
                    .into(),
            ));
        }
        // G6: the mid band sits strictly between the decode band and `a_tiled_min_m`. Below it
        // the decode kernel (and the decode plans' `_nq` epilogues, which write row-major) own
        // the rows; the prefill rungs in it never take an `_nq` epilogue (prefill plans do not
        // fuse), so the reason `a_tiled_min_m` must exceed 512 does not reach them.
        if k.atiled_mid_min_m != 0
            && (k.atiled_mid_min_m <= k.decode_max_m
                || (k.a_tiled_min_m != 0 && k.atiled_mid_min_m >= k.a_tiled_min_m))
        {
            return Err(Error::Rule(format!(
                "atiled_mid_min_m {} must exceed decode_max_m {} and sit below a_tiled_min_m {}",
                k.atiled_mid_min_m, k.decode_max_m, k.a_tiled_min_m
            )));
        }
        if k.atiled_mid && k.atiled_mid_min_m == 0 {
            return Err(Error::Rule(
                "atiled_mid is on with no atiled_mid_min_m: the mid band has no edge".into(),
            ));
        }
        if k.decode_max_m > k.dec_mtile * k.dec_max_tm {
            return Err(Error::Rule(format!(
                "decode_max_m {} exceeds the widest decode tile {}",
                k.decode_max_m,
                k.dec_mtile * k.dec_max_tm
            )));
        }
        if self.split_k_ladder.is_empty() {
            return Err(Error::Rule("split_k_ladder is empty".into()));
        }
        if self.decode_band.is_empty() || self.folded_tn.is_empty() {
            return Err(Error::Rule(
                "decode_band and folded_tn must have a default row".into(),
            ));
        }
        Ok(())
    }

    /// `split_k_for(nblk, M)`: the smallest split in the ladder whose grid clears the fill target.
    pub fn split_k(&self, nblk: u32, m: u32) -> u32 {
        let fill = self
            .split_k_fill
            .iter()
            .find(|r| r.m_max.is_none_or(|x| m <= x))
            .map(|r| r.fill)
            .unwrap_or(0);
        let last = *self.split_k_ladder.last().expect("checked non-empty");
        for &ks in &self.split_k_ladder {
            if ks == last {
                break;
            }
            if nblk * ks >= fill {
                return ks;
            }
        }
        last
    }

    /// `decode_bk64(tm, ks)` as a table: first row whose stated fields match.
    pub fn decode_bk(&self, tm: u32, ks: u32) -> u32 {
        self.decode_bk
            .iter()
            .find(|r| r.tm.is_none_or(|x| x == tm) && r.ks.is_none_or(|x| x == ks))
            .map(|r| r.bk)
            .unwrap_or(128)
    }

    /// The (ks, bk) the band table gives for this cell, before the K-alignment override.
    pub fn decode_split(&self, m: u32, nblk: u32, k: u32) -> Result<(u32, u32)> {
        let row = self
            .decode_band
            .iter()
            .find(|r| {
                r.m_min.is_none_or(|x| m >= x)
                    && r.m_max.is_none_or(|x| m <= x)
                    && r.nblk_min.is_none_or(|x| nblk >= x)
                    && r.nblk_max.is_none_or(|x| nblk <= x)
                    && r.k_min.is_none_or(|x| k >= x)
                    && r.k_max.is_none_or(|x| k <= x)
            })
            .ok_or_else(|| {
                Error::Rule(format!(
                    "no decode_band row matches M={m} nblk={nblk} K={k}"
                ))
            })?;
        let ks = if row.ks_from_fill {
            self.split_k(nblk, m)
        } else {
            row.ks.ok_or_else(|| {
                Error::Rule("decode_band row has neither ks nor ks_from_fill".into())
            })?
        };
        let tm = m.div_ceil(self.knobs.dec_mtile);
        let bk = if row.bk_from_tm {
            self.decode_bk(tm, ks)
        } else {
            row.bk.ok_or_else(|| {
                Error::Rule("decode_band row has neither bk nor bk_from_tm".into())
            })?
        };
        Ok((ks, bk))
    }

    /// The N tile, for the family that is asking. `a_tiled` is the kernel, not the edge: the
    /// A-tiled kernel reads A straight into the WMMA fragments, the folded one stages it through
    /// LDS, and that is the axis the measurement splits on.
    fn tn_for(&self, m: u32, n: u32, a_tiled: bool) -> u32 {
        self.folded_tn
            .iter()
            .find(|r| {
                r.m_min.is_none_or(|x| m >= x)
                    && r.n_min.is_none_or(|x| n >= x)
                    && r.a_tiled.is_none_or(|x| x == a_tiled)
            })
            .map(|r| r.tn)
            .unwrap_or(2)
    }

    /// The A-tiled tile height in waves: the first `[[atiled_wm]]` row that matches, else the
    /// reference's 4.
    fn wm_for(&self, m: u32, tn: u32, wperm: bool) -> Option<&WmRow> {
        self.atiled_wm.iter().find(|r| {
            r.tn == tn
                && r.w_fragment_order == wperm
                && r.m_min.is_none_or(|x| m >= x)
                && r.m_max.is_none_or(|x| m <= x)
        })
    }

    fn lbk_for(&self, tn: u32) -> Result<u32> {
        self.atiled_lbk
            .iter()
            .find(|r| r.tn == tn)
            .map(|r| r.lbk)
            .ok_or_else(|| Error::Rule(format!("no atiled_lbk row for TN={tn}")))
    }
}

/// The symbol a kernel's ARITHMETIC is named by: two symbols with one class sum every output in
/// the same order and write the same bytes. Today one pair is known and gated (G1, 2026-09-23):
/// `..._wm8` is the reference's A-tiled kernel re-tiled 512 rows tall (`hip/mxfp4_atiled_wm.h`),
/// 0 of 13 shapes differing against it (`tools/capture/g1/lab_bench.cpp`). A caller that asks
/// "do these two plans sum the same way" compares classes, not symbols.
#[must_use]
pub fn summation_class(kernel: &str) -> &str {
    // G3 adds two bodies of the same tile: `_wm8s` (SADDR W / Ws addressing) and `_wm8sp` (the
    // same plus the W double buffer), each gated 0 of 13 shapes differing (`tools/capture/g3`).
    for tail in ["_wm8sp", "_wm8s", "_wm8"] {
        if let Some(base) = kernel.strip_suffix(tail) {
            return base;
        }
    }
    // G5: the folded tile at a mid-M height sums every output in the reference folded tile's
    // order (0 bytes differing, `tools/capture/g5`), so it is named by that tile.
    if kernel.starts_with("mxfp4_gemm_folded_mt_tn2_wperm1_t") {
        return "mxfp4_gemm_folded_tn2_wperm1_epifast1";
    }
    // G6: the A-tiled tile at a mid-M height (any TM x WM, piped or not) sums every output in the
    // reference A-tiled tile's order at its TN (0 bytes differing, `tools/capture/g6`).
    if kernel.starts_with("mxfp4_gemm_atiled_mt_tn2_") {
        return "mxfp4_gemm_atiled_tn2_lbk128_wperm1";
    }
    if kernel.starts_with("mxfp4_gemm_atiled_mt_tn4_") {
        return "mxfp4_gemm_atiled_tn4_lbk64_wperm1";
    }
    // G6: the skinny-N body sums every output as the folded tile does (0 bytes differing).
    if kernel.starts_with("mxfp4_gemm_skinny_n") {
        return "mxfp4_gemm_folded_tn2_wperm1_epifast1";
    }
    // CMP27E: the register-prefetch twins write their shipped cells' bytes (0 differing).
    match kernel {
        "mxfp4_gemm_decode_pf_bk128_ks2_tm4_wperm1_nt1" => {
            return "mxfp4_gemm_decode_bk128_ks2_tm4_wperm1_nt1";
        }
        "mxfp4_gemm_decode_pf_bk64_ks1_tm4_wperm1_nt1" => {
            return "mxfp4_gemm_decode_bk64_ks1_tm4_wperm1_nt1";
        }
        DECODE_PF_PAIR => return "mxfp4_gemm_decode_pair_bk128_tm4_wperm1_nt1",
        // CMP27T: the conc-4 (tm2) twins of the same body.
        "mxfp4_gemm_decode_pf_bk128_ks2_tm2_wperm1_nt1" => {
            return "mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1";
        }
        // CMP27X: the 4-wave twins of the same body.
        "mxfp4_gemm_decode_pf_bk128_ks2_tm2_wperm1_nt1_dwn4" => {
            return "mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1";
        }
        "mxfp4_gemm_decode_pf_bk128_ks2_tm4_wperm1_nt1_dwn4" => {
            return "mxfp4_gemm_decode_bk128_ks2_tm4_wperm1_nt1";
        }
        _ => {}
    }
    kernel
}

/// The ONE summation the served MXFP4 GEMM families share, across A layouts and kernels: every
/// output an fp32 WMMA chain over k ascending in 16-deep steps from zero, on the kMag-folded W, with
/// the EPIFAST epilogue. `Some("mxfp4-fold-kasc")` for a symbol in it, `None` otherwise.
///
/// The evidence, gate by gate: the folded tile and the A-tiled tiles at every TN, LBK and height
/// write the same bytes over the same A (G2, 0 bytes on 11 shapes, 44 arms); the decode kernel at
/// ONE split writes the folded tile's bytes (the `lm_head` cell, C6's small chunk); the G1/G3 tall
/// bodies, G5's mid-M folded heights and G6's mid-M A-tiled heights and skinny body join through
/// [`summation_class`]. A split decode cell (`_ks2`, `_ks4`) is a DIFFERENT summation (the partial
/// sums reassociate) and is not in the class -- the negative control of the test below.
#[must_use]
pub fn fold_class(kernel: &str) -> Option<&'static str> {
    const FOLD: &str = "mxfp4-fold-kasc";
    let base = summation_class(kernel);
    let decode_one_split = base.starts_with("mxfp4_gemm_decode_bk")
        && base.contains("_ks1_")
        && base.ends_with("_wperm1_nt1");
    matches!(
        base,
        "mxfp4_gemm_folded_tn2_wperm1_epifast1"
            | "mxfp4_gemm_atiled_tn2_lbk128_wperm1"
            | "mxfp4_gemm_atiled_tn4_lbk64_wperm1"
    )
    .then_some(FOLD)
    .or(decode_one_split.then_some(FOLD))
}

// ---------------------------------------------------------------------------- descriptor

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Shape {
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Shape { m, n, k }
    }
}

/// Merged-linear partition select (`launch_p`): n-blocks in [0, pb1) read the first rotated copy
/// of A, [pb1, pb2) the second, [pb2, N) the third. The default is one partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partitions {
    pub pb1: i32,
    pub pb2: i32,
    pub astride: i64,
}

/// `pb1 = pb2 = 1<<30` is what the stock entry passes.
pub const NO_PARTITIONS: Partitions = Partitions {
    pb1: 1 << 30,
    pb2: 1 << 30,
    astride: 0,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    /// W4A8 GEMM. `has_wref` is false only for a checkpoint with no per-row reference exponent,
    /// which takes the unfolded kernel and cannot be partitioned.
    Gemm {
        has_wref: bool,
        partitions: Partitions,
    },
    /// Fused residual-add + Gemma RMSNorm + per-token e4m3 quant. Shape: m rows, k columns.
    AddRmsQuant { eps: f32 },
    /// Fused silu(gate)*up + per-token e4m3 quant. Shape: m rows, n = half the gate_up width.
    SiluMulQuant,
    /// Per-head gated RMSNorm + per-token e4m3 quant. Shape: m rows, n columns; `zs` is the z row
    /// stride in elements.
    GdnNormQuant { zs: i64, eps: f32 },
}

impl Op {
    pub fn gemm() -> Self {
        Op::Gemm {
            has_wref: true,
            partitions: NO_PARTITIONS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArgValue {
    Buffer {
        buffer: String,
    },
    /// A pointer slot the chosen instantiation provably never dereferences -- the split-K partial
    /// buffer and the block counter at DKS==1, where the accumulator already holds the whole K
    /// range and the kernel writes C directly. Named, so the descriptor says WHICH buffer is
    /// absent rather than carrying an unexplained zero.
    NullBuffer {
        buffer: String,
    },
    I32 {
        ival: i64,
    },
    I64 {
        ival: i64,
    },
    F32 {
        fval: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Slot {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    #[serde(flatten)]
    pub value: ArgValue,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WorkspaceReq {
    pub name: String,
    pub kind: WorkspaceKind,
    pub bytes: u64,
    /// Persistent workspaces are zeroed once at construction and never again; a Scratch one is
    /// whatever the last launch left behind.
    pub zero_once: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LaunchDescriptor {
    pub op: String,
    pub plugin: String,
    pub kernel: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic LDS. Always 0 here: every kernel declares its LDS statically (the manifest carries
    /// `group_segment_fixed_size`, which is what the loader checks against the device limit).
    pub shared_bytes: u32,
    pub kernarg_bytes: u32,
    pub args: Vec<Slot>,
    pub workspaces: Vec<WorkspaceReq>,
    /// Launch count per call, declared statically so it can be budgeted before running (I7). The
    /// split-K reduce is FUSED into the decode kernel, so a split is still one launch.
    pub n_launches: u32,
}

impl LaunchDescriptor {
    pub fn arg(&self, name: &str) -> Option<&Slot> {
        self.args.iter().find(|s| s.name == name)
    }

    /// A short, stable line for goldens and logs.
    pub fn summary(&self) -> String {
        format!(
            "{} grid={}x{}x{} block={} n_launches={}",
            self.kernel, self.grid[0], self.grid[1], self.grid[2], self.block[0], self.n_launches
        )
    }
}

struct Builder {
    sig: &'static [SigArg],
    values: Vec<ArgValue>,
}

impl Builder {
    fn new(sig: &'static [SigArg]) -> Self {
        Builder {
            sig,
            values: Vec::with_capacity(sig.len()),
        }
    }

    fn buf(mut self, name: &str) -> Self {
        self.values.push(ArgValue::Buffer {
            buffer: name.to_owned(),
        });
        self
    }

    /// A pointer the chosen instantiation never dereferences (see [`ArgValue::NullBuffer`]).
    fn null_buf(mut self, name: &str) -> Self {
        self.values.push(ArgValue::NullBuffer {
            buffer: name.to_owned(),
        });
        self
    }

    fn i32(mut self, v: i64) -> Self {
        self.values.push(ArgValue::I32 { ival: v });
        self
    }

    fn i64(mut self, v: i64) -> Self {
        self.values.push(ArgValue::I64 { ival: v });
        self
    }

    fn f32(mut self, v: f32) -> Self {
        self.values.push(ArgValue::F32 { fval: v });
        self
    }

    fn finish(self) -> Result<(Vec<Slot>, u32)> {
        if self.values.len() != self.sig.len() {
            return Err(Error::Rule(format!(
                "kernarg builder: {} values for a {}-argument signature",
                self.values.len(),
                self.sig.len()
            )));
        }
        let (slots, size) = abi::pack(self.sig);
        let mut out = Vec::with_capacity(self.sig.len());
        for ((s, (off, sz)), v) in self.sig.iter().zip(slots).zip(self.values) {
            let want = match v {
                ArgValue::Buffer { .. } | ArgValue::NullBuffer { .. } => ArgKind::Buffer,
                ArgValue::I32 { .. } => ArgKind::I32,
                ArgValue::I64 { .. } => ArgKind::I64,
                ArgValue::F32 { .. } => ArgKind::F32,
            };
            if want != s.kind {
                return Err(Error::Rule(format!(
                    "kernarg builder: `{}` is {:?} in the signature, bound as {want:?}",
                    s.name, s.kind
                )));
            }
            out.push(Slot {
                name: s.name.to_owned(),
                offset: off,
                size: sz,
                value: v,
            });
        }
        Ok((out, size))
    }
}

// ---------------------------------------------------------------------------- resolve

/// The one decision function. Pure: (op, shape, rules) in, descriptor out, no environment.
pub fn resolve(op: Op, shape: Shape, rules: &LaunchRules) -> Result<LaunchDescriptor> {
    resolve_with_layout(op, shape, rules, ALayout::FollowsM)
}

/// Where a GEMM's A-side activation layout came from.
///
/// [`a_tiled`] answers "does this M take the fragment-tiled layout", which is the right
/// question for a **producer**: it is about to write the bytes, and the threshold is what
/// decides which way. It is the wrong question for a **consumer**, because a consumer reads
/// whatever its producer actually wrote — and `IMPLEMENTATION-PLAN.md` P0 follow-up (a) is
/// precisely that the port collapsed the two: *"right for the fused producers but cannot say
/// 'this edge is row-major because a different producer wrote it.'"*
///
/// `saber_model::graph` finds one such edge in Qwen3.8-27B: `o_proj`'s A side, written by the
/// attention output-gate producer, which has no fused implementation on this tier and therefore
/// writes row-major bytes at every M. Reading them with the A-tiled kernel is garbage, not a
/// regression.
///
/// This is the smallest additive way to let a caller say so: [`resolve`] keeps its signature and
/// its meaning ("assume a fused producer"), and a caller that knows better calls
/// [`resolve_with_layout`]. No `Op` variant grew a field, so no construction site had to change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ALayout {
    /// A fused quant producer wrote the A side, so the layout follows M against
    /// `a_tiled_min_m`. What [`resolve`] assumes.
    #[default]
    FollowsM,
    /// A non-fused producer wrote it: row-major at every M, and the A-tiled family must not be
    /// selected however large M is.
    RowMajor,
}

/// [`resolve`], told what layout the A-side edge actually carries.
///
/// Only a GEMM reads it; the three quant producers are what *writes* a layout, so their
/// decision stays [`a_tiled`] against M and this argument does not reach them.
pub fn resolve_with_layout(
    op: Op,
    shape: Shape,
    rules: &LaunchRules,
    a_layout: ALayout,
) -> Result<LaunchDescriptor> {
    match op {
        Op::Gemm {
            has_wref,
            partitions,
        } => gemm(shape, has_wref, partitions, rules, a_layout),
        Op::AddRmsQuant { eps } => add_rms_quant(shape, eps, rules),
        Op::SiluMulQuant => silu_mul_quant(shape, rules),
        Op::GdnNormQuant { zs, eps } => gdn_norm_quant(shape, zs, eps, rules),
    }
}

/// The decode GEMM with `silu_mul_quant` folded into its A staging: it reads the bf16 gate_up
/// output directly and computes the per-token scale itself, so the producer's launch disappears.
///
/// Deliberately NOT an arm of [`gemm`] and not an [`Op`]. Whether this route is legal is a
/// property of the GRAPH EDGE -- "the A operand of this GEMM is what silu_mul would have
/// produced" -- and no shape carries that. A caller that knows its edge calls this by name; every
/// other caller resolves the same shape to the plain kernel and cannot reach the fused symbol by
/// accident.
///
/// It refuses any cell the fused family was not compiled for, which is deliberate: the fused
/// kernel keeps the band's own split-K, and a cell it does not have is a shape whose fused cost
/// has never been measured.
pub fn gemm_decode_fused_silu(shape: Shape, rules: &LaunchRules) -> Result<LaunchDescriptor> {
    let Shape { m, n, k } = shape;
    let kn = &rules.knobs;
    if k % kn.k_multiple != 0 {
        return Err(Error::Rule(format!(
            "K={k} must be a multiple of {}",
            kn.k_multiple
        )));
    }
    if m > kn.decode_max_m {
        return Err(Error::Rule(format!(
            "fused silu: M={m} is past the decode band ({}); the fused route exists only there",
            kn.decode_max_m
        )));
    }
    let bnd = kn.dec_dwn * 16;
    let nblk = n.div_ceil(bnd);
    let (ks, bk) = rules.decode_split(m, nblk, k)?;
    let tm = m.div_ceil(kn.dec_mtile);
    let row = rules
        .decode_fused_silu
        .iter()
        .find(|r| r.bk == bk && r.ks == ks && r.tm == tm)
        .ok_or_else(|| {
            Error::Rule(format!(
                "fused silu: M={m} N={n} K={k} resolves to (bk {bk}, ks {ks}, tm {tm}) and no \
                 fused instantiation covers that cell"
            ))
        })?;
    let (args, bytes) = Builder::new(abi::SIG_GEMM_DECODE_SILU)
        .buf("gu")
        .buf("w")
        .buf("ws")
        .buf("wref")
        .buf("decode_partial")
        .buf("decode_counter")
        .buf("c")
        .i32(m as i64)
        .i32(n as i64)
        .i32(k as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gemm_decode_fused_silu".into(),
        plugin: "mxfp4_gemm_decode_silu".into(),
        kernel: row.kernel.clone(),
        grid: [nblk, 1, ks],
        block: [kn.dec_dwn * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![
            WorkspaceReq {
                name: "decode_partial".into(),
                kind: rules.workspace.decode_partial.kind,
                bytes: ks as u64 * m as u64 * n as u64 * 4,
                zero_once: false,
            },
            WorkspaceReq {
                name: "decode_counter".into(),
                kind: rules.workspace.decode_counter.kind,
                bytes: nblk as u64 * 4,
                zero_once: true,
            },
        ],
        // One launch where the plain route spends two. That is the whole claim, and the ledger
        // says what it costs.
        n_launches: 1,
    })
}

/// The decode PAIR (ws-h37): two decode GEMMs that read ONE activation, in one dispatch.
///
/// Each half is exactly the descriptor [`resolve`] gives it alone -- its own split count, its own
/// partial slab and block counter -- so the pair is not a new tiling and not a new choice: it is
/// the two unpaired grids laid end to end (`nb0 * ks0 + nb1 * ks1` blocks, problem 0 first) under
/// one kernel whose block index and split count are arguments. That is why it can be gated
/// BIT-EXACT against the two dispatches it replaces (`step-tp2`'s fusion gate), and why it refuses
/// anything else: both halves must be in the decode band at the same `bk`, `tm`, `wperm1_nt1`
/// (the only instantiations compiled), with the same `M` and `K`.
pub fn gemm_decode_pair(
    first: (&LaunchDescriptor, Shape),
    second: (&LaunchDescriptor, Shape),
    rules: &LaunchRules,
) -> Result<LaunchDescriptor> {
    let ((d0, s0), (d1, s1)) = (first, second);
    let (bk0, ks0, tm0) = decode_cell(d0)?;
    let (bk1, ks1, tm1) = decode_cell(d1)?;
    if (bk0, tm0) != (bk1, tm1) || s0.m != s1.m || s0.k != s1.k {
        return Err(Error::Rule(format!(
            "decode pair: halves differ in (bk, tm, M, K): ({bk0}, {tm0}, {}, {}) vs ({bk1}, \
             {tm1}, {}, {})",
            s0.m, s0.k, s1.m, s1.k
        )));
    }
    let kn = &rules.knobs;
    let (nb0, nb1) = (d0.grid[0], d1.grid[0]);
    let mut b = Builder::new(abi::SIG_GEMM_DECODE_PAIR)
        .buf("a")
        .buf("as")
        .buf("w")
        .buf("ws")
        .buf("wref");
    b = if ks0 > 1 {
        b.buf("decode_partial").buf("decode_counter")
    } else {
        b.null_buf("decode_partial").null_buf("decode_counter")
    };
    b = b.buf("c").buf("w_b").buf("ws_b").buf("wref_b");
    b = if ks1 > 1 {
        b.buf("decode_partial_b").buf("decode_counter_b")
    } else {
        b.null_buf("decode_partial_b").null_buf("decode_counter_b")
    };
    let (args, bytes) = b
        .buf("c_b")
        .i32(i64::from(s0.m))
        .i32(i64::from(s0.k))
        .i32(i64::from(s0.n))
        .i32(i64::from(ks0))
        .i32(i64::from(nb0))
        .i32(i64::from(s1.n))
        .i32(i64::from(ks1))
        .i32(i64::from(nb1))
        .finish()?;
    let mut workspaces = Vec::new();
    for (ks, shape, nblk, sfx) in [(ks0, s0, nb0, ""), (ks1, s1, nb1, "_b")] {
        if ks > 1 {
            workspaces.push(WorkspaceReq {
                name: format!("decode_partial{sfx}"),
                kind: rules.workspace.decode_partial.kind,
                bytes: u64::from(ks) * u64::from(shape.m) * u64::from(shape.n) * 4,
                zero_once: false,
            });
            workspaces.push(WorkspaceReq {
                name: format!("decode_counter{sfx}"),
                kind: rules.workspace.decode_counter.kind,
                bytes: u64::from(nblk) * 4,
                zero_once: true,
            });
        }
    }
    // CMP27E: the prefetch twin (problem 1 dispatched first; the same blocks, the same bytes).
    let pf = kn.decode_pf && bk0 == 128 && tm0 == 4;
    Ok(LaunchDescriptor {
        op: "gemm_decode_pair".into(),
        plugin: if pf {
            DECODE_PF_PLUGIN
        } else {
            "mxfp4_gemm_decode"
        }
        .into(),
        kernel: if pf {
            DECODE_PF_PAIR.to_owned()
        } else {
            format!("mxfp4_gemm_decode_pair_bk{bk0}_tm{tm0}_wperm1_nt1")
        },
        grid: [nb0 * ks0 + nb1 * ks1, 1, 1],
        block: [kn.dec_dwn * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        n_launches: 1,
    })
}

/// `(bk, ks, tm)` of a decode-band descriptor the pair can carry, or a refusal naming it.
fn decode_cell(d: &LaunchDescriptor) -> Result<(u32, u32, u32)> {
    let bad = || {
        Error::Rule(format!(
            "decode pair: `{}` is not a paired decode cell",
            d.kernel
        ))
    };
    if d.op != "gemm_decode" {
        return Err(bad());
    }
    let rest = d
        .kernel
        .strip_prefix("mxfp4_gemm_decode_bk")
        .or_else(|| d.kernel.strip_prefix("mxfp4_gemm_decode_pf_bk"))
        .ok_or_else(bad)?;
    let parts: Vec<&str> = rest.split('_').collect();
    let [bk, ks, tm, wp, nt] = parts[..] else {
        return Err(bad());
    };
    if wp != "wperm1" || nt != "nt1" {
        return Err(bad());
    }
    let num = |t: &str, p: &str| t.strip_prefix(p).and_then(|v| v.parse::<u32>().ok());
    match (num(bk, ""), num(ks, "ks"), num(tm, "tm")) {
        (Some(b), Some(k), Some(t)) if d.grid[2] == k => Ok((b, k, t)),
        _ => Err(bad()),
    }
}

/// The split cells fusion-map row 3 is compiled for (`the internal reference harness`):
/// `(bk, ks)` at `tm1`, `wperm1_nt1`. The exact reduce only fires at M <= 12, so `tm1` is all it
/// can reach, and a split (`ks > 1`) is what gives the kernel its last-arriving block.
pub const GEMM_AR_NQ_CELLS: &[(u32, u32)] = &[(64, 4), (128, 4), (128, 2)];

/// The one Persistent workspace fusion-map row 3 adds: a MONOTONIC 64-bit arrival counter over
/// the GEMM's n-tiles (never reset; call `c` holds tickets `[c*nt, (c+1)*nt)`).
pub const GEMM_AR_NQ_TILES: &str = "ar_tiles";

/// The code object row 3 lives in. NOT `mxfp4_gemm_decode`: every autotune key carries a digest
/// of every declared plugin's build hash, so appending these symbols there would re-key the whole
/// committed cache; this plugin binds no registry row (the block planner's fold binds it), so it
/// is not declared and moves no key.
pub const GEMM_AR_NQ_PLUGIN: &str = "mxfp4_gemm_decode_arnq";

/// Row 3's race-check twin (T4, `the internal reference harness`): the served row 3
/// with `hip/ar_race_check.h`'s checks inserted, selected when the `_nq` it folds is the checked
/// one (`--engine-race-check`).
pub const GEMM_AR_NQ_CHK_PLUGIN: &str = "mxfp4_gemm_decode_arnq_chk";

/// Fusion-map ROW 3 (ws-h38): a row-parallel decode GEMM whose split-K last block also runs the
/// EXACT two-rank all-reduce and the `_nq` epilogue (residual add, Gemma RMS norm, per-token e4m3
/// quant). One dispatch for `GEMM -> allreduce -> add_rms_quant`.
///
/// Built from the two descriptors it replaces, and refused unless both are exactly what the
/// kernel was compiled for:
///
/// * the GEMM is a decode cell with a split (`ks > 1`: the kernel's tail runs in the split-K
///   reduce's last-arriving block) at `tm1`, `wperm1_nt1`, one of [`GEMM_AR_NQ_CELLS`], with no
///   merged-linear partitions, and `N` a whole number of 128-column n-tiles;
/// * `nq` is the rules' `allreduce_exact_nq` descriptor for the GEMM's own output: `k16 * 8 == N`
///   and one block per row, `M` rows -- so the exact family, never the rotated one (a banded or
///   rotated reduce has no `_nq` arm to fold);
/// * the row owners are the last `M` of the `N / 128` tile arrivals, so `N / 128 >= M`, and each
///   tile pushes `M * 16` words, one per thread of a `dec_dwn * 32 = SABER_ARNQ_T` block.
///
/// The kernarg image is the GEMM's fourteen arguments unchanged, then the epilogue's own values
/// taken from `nq` (`k16`, `slot_stride16`, `eps`, `drain`, `acq`), then `ep_rows = M`. The grid is
/// the GEMM's. The workspaces are the GEMM's (the split-K partial slab and block counter), the
/// reduce's five (`ar_scratch`, `ar_peer_scratch`, `ar_flags`, `ar_peer_flags`, `ar_seq`), and
/// [`GEMM_AR_NQ_TILES`]. The launcher keys peer-visible and Persistent regions by `(plugin, op,
/// shape)`, so this op owns its OWN scratch / flag / sequence set, in lockstep across the ranks
/// because both issue the identical sequence.
pub fn gemm_decode_ar_nq(
    gemm: (&LaunchDescriptor, Shape),
    nq: &LaunchDescriptor,
    rules: &LaunchRules,
) -> Result<LaunchDescriptor> {
    let (g, s) = gemm;
    let refuse = |why: String| Error::Rule(format!("gemm_decode_ar_nq: {why}"));
    let (bk, ks, tm) = decode_cell(g)?;
    if tm != 1 || !GEMM_AR_NQ_CELLS.contains(&(bk, ks)) {
        return Err(refuse(format!(
            "`{}` is not a compiled row-3 cell (tm1 and (bk, ks) in {GEMM_AR_NQ_CELLS:?})",
            g.kernel
        )));
    }
    let kn = &rules.knobs;
    let bnd = kn.dec_dwn * 16;
    let block = kn.dec_dwn * 32;
    let ntiles = g.grid[0];
    let int = |d: &LaunchDescriptor, name: &str| match d.arg(name).map(|x| &x.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => Ok(*ival),
        _ => Err(refuse(format!("`{}` has no integer `{name}`", d.kernel))),
    };
    if int(g, "pb1")? < i64::from(s.n) || int(g, "pb2")? < i64::from(s.n) {
        return Err(refuse("a partitioned (merged-linear) GEMM".into()));
    }
    if s.n % bnd != 0 || ntiles * bnd != s.n {
        return Err(refuse(format!(
            "N = {} is not a whole number of {bnd}-column n-tiles",
            s.n
        )));
    }
    if nq.op != "allreduce_exact_nq" {
        return Err(refuse(format!(
            "the reduce is `{}`, not the exact `_nq` epilogue",
            nq.op
        )));
    }
    let k16 = int(nq, "k16")?;
    if k16 * 8 != i64::from(s.n) || nq.grid[0] != s.m {
        return Err(refuse(format!(
            "the epilogue is {} x {} and the GEMM writes {} x {}",
            nq.grid[0],
            k16 * 8,
            s.m,
            s.n
        )));
    }
    if nq.block[0] != block {
        return Err(refuse(format!(
            "the epilogue's block is {} threads and the decode block is {block}",
            nq.block[0]
        )));
    }
    if s.m == 0 || ntiles < s.m || s.m * 16 > block {
        return Err(refuse(format!(
            "M = {} rows need M row owners among {ntiles} n-tiles and M * 16 push lanes of {block}",
            s.m
        )));
    }
    let eps = match nq.arg("eps").map(|x| &x.value) {
        Some(ArgValue::F32 { fval }) => *fval,
        _ => return Err(refuse("the epilogue has no `eps`".into())),
    };
    let (args, bytes) = Builder::new(abi::SIG_GEMM_DECODE_AR_NQ)
        .buf("a")
        .buf("w")
        .buf("ws")
        .buf("wref")
        .buf("as")
        .buf("decode_partial")
        .buf("decode_counter")
        .buf("c")
        .i32(i64::from(s.m))
        .i32(i64::from(s.n))
        .i32(i64::from(s.k))
        .i32(int(g, "pb1")?)
        .i32(int(g, "pb2")?)
        .i64(int(g, "astride")?)
        .buf("ar_peer_scratch")
        .buf("ar_scratch")
        .buf("residual")
        .buf("norm_w")
        .buf("q_out")
        .buf("scale_out")
        .buf("res_out")
        .buf("ar_peer_flags")
        .buf("ar_flags")
        .buf("ar_seq")
        .buf(GEMM_AR_NQ_TILES)
        .i32(k16)
        .i32(int(nq, "slot_stride16")?)
        .f32(eps)
        .i32(int(nq, "drain")?)
        .i32(int(nq, "acq")?)
        .i32(i64::from(s.m))
        .finish()?;
    let mut workspaces: Vec<WorkspaceReq> = g.workspaces.clone();
    if !["decode_partial", "decode_counter"]
        .iter()
        .all(|n| workspaces.iter().any(|w| w.name == *n))
    {
        return Err(refuse(format!(
            "`{}` does not declare the split-K workspaces",
            g.kernel
        )));
    }
    for w in &nq.workspaces {
        if !w.name.starts_with("ar_") {
            return Err(refuse(format!(
                "the epilogue declares `{}`, which row 3 does not carry",
                w.name
            )));
        }
        workspaces.push(w.clone());
    }
    workspaces.push(WorkspaceReq {
        name: GEMM_AR_NQ_TILES.into(),
        kind: WorkspaceKind::Persistent,
        bytes: 8,
        zero_once: true,
    });
    // T4: an `_nq` resolved under the race check folds into row 3's race-check twin (its own code
    // object, `decode_arnq_chk`), so a checked run never folds a checked reduce into an unchecked
    // GEMM tail.
    let checked = nq.kernel.ends_with(crate::saber::AR_CHECK_SUFFIX);
    let host = matches!(
        nq.plugin.as_str(),
        "saber_ar_host_exact" | "saber_ar_host_exact_chk"
    );
    let (plugin, suffix) = match (host, checked) {
        (true, true) => (
            "mxfp4_gemm_decode_host_arnq_chk",
            crate::saber::AR_CHECK_SUFFIX,
        ),
        (true, false) => ("mxfp4_gemm_decode_host_arnq", ""),
        (false, true) => (GEMM_AR_NQ_CHK_PLUGIN, crate::saber::AR_CHECK_SUFFIX),
        (false, false) => (GEMM_AR_NQ_PLUGIN, ""),
    };
    Ok(LaunchDescriptor {
        op: "gemm_decode_ar_nq".into(),
        // Its own code object (`prebuilt::ALL`): see `the internal reference harness` for why.
        plugin: plugin.into(),
        kernel: format!("mxfp4_gemm_decode_arnq_bk{bk}_ks{ks}_tm1_wperm1_nt1{suffix}"),
        grid: g.grid,
        block: g.block,
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        n_launches: 1,
    })
}

/// CMP27AE: the GEMM + wht6 fold's code object (`hip/mxfp4_gemm_decode_arw.hip`).
pub const GEMM_ARW_PLUGIN: &str = "mxfp4_gemm_decode_arw";

/// CMP27AE: the dwn4 prefetch cells the fold is compiled for: `(bk, ks, tm)`.
pub const GEMM_ARW_CELLS: &[(u32, u32, u32)] = &[(128, 2, 2), (128, 2, 4)];

/// **CMP27AE: the N 5120 dwn4 decode GEMM and the rotated (wht6) all-reduce of its output as ONE
/// dispatch** (`mxfp4_gemm_decode_arw_*`, CMP27AB's probe body): the last split-K arriver of each
/// 64-column tile folds the partials with the shipped fold, quantizes the tile's rotation groups
/// (one a row at DWN 4) with the shipped wht6 arithmetic, pushes them to the peer, waits for the
/// peer's tile and stores the reduced rows to `out` -- the GEMM's `c`, the reduce being in place.
///
/// Built from the two descriptors it replaces and refused unless both are exactly what the kernel
/// was compiled for: the GEMM a `_dwn4` prefetch cell of [`GEMM_ARW_CELLS`] at N 5120 (80 tiles,
/// unpartitioned), the reduce the served wht6 bf16 one-shot (or its wave-parallel body) over the
/// same `M x N` message in one band, never a race-check twin (the fold has none: under
/// `--engine-race-check` it stands down to the two launches). The kernarg image is the GEMM's
/// fourteen arguments, then the reduce's regions, `out` = `c`, the slot stride, the family's
/// offset in a slot ([`crate::saber::ar_fuse_off`]), drain and acquire. The workspaces are the
/// GEMM's (partial slab, tile counter) and the reduce's five protocol regions (no `ar_loc_pack`:
/// the tile keeps its own half in registers); the launcher keys the latter onto the one-shot's
/// regions (`saber_exec::launch::protocol_family` / `protocol_op`).
///
/// # Errors
/// Any mismatch above, by name.
pub fn gemm_decode_arw(
    gemm: (&LaunchDescriptor, Shape),
    ar: &LaunchDescriptor,
    saber: &crate::saber::SaberRules,
) -> Result<LaunchDescriptor> {
    let (g, s) = gemm;
    let refuse = |why: String| Error::Rule(format!("gemm_decode_arw: {why}"));
    let rest = g
        .kernel
        .strip_prefix("mxfp4_gemm_decode_pf_bk")
        .and_then(|r| r.strip_suffix("_wperm1_nt1_dwn4"))
        .ok_or_else(|| refuse(format!("`{}` is not a dwn4 prefetch cell", g.kernel)))?;
    let cell: Vec<u32> = rest
        .split('_')
        .enumerate()
        .filter_map(|(i, t)| {
            let t = match i {
                0 => Some(t),
                1 => t.strip_prefix("ks"),
                2 => t.strip_prefix("tm"),
                _ => None,
            }?;
            t.parse().ok()
        })
        .collect();
    let [bk, ks, tm] = cell[..] else {
        return Err(refuse(format!("`{}`: no (bk, ks, tm)", g.kernel)));
    };
    if !GEMM_ARW_CELLS.contains(&(bk, ks, tm)) || g.grid[2] != ks {
        return Err(refuse(format!(
            "`{}` is not a compiled cell ({GEMM_ARW_CELLS:?})",
            g.kernel
        )));
    }
    let tiles = crate::saber::AR_FUSE_TILES as u32;
    if s.n != tiles * 64 || g.grid[0] != tiles || g.block[0] != 128 {
        return Err(refuse(format!(
            "N = {} on a {}-tile grid of {} threads; the fold is compiled for {tiles} tiles of 64 \
             columns on 4 waves",
            s.n, g.grid[0], g.block[0]
        )));
    }
    if s.m == 0 || s.m > 16 * tm {
        return Err(refuse(format!("M = {} rows on a tm{tm} tile", s.m)));
    }
    let int = |d: &LaunchDescriptor, name: &str| match d.arg(name).map(|x| &x.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => Ok(*ival),
        _ => Err(refuse(format!("`{}` has no integer `{name}`", d.kernel))),
    };
    if int(g, "pb1")? < i64::from(s.n) || int(g, "pb2")? < i64::from(s.n) {
        return Err(refuse("a partitioned (merged-linear) GEMM".into()));
    }
    let body = ar.kernel.as_str();
    if ar.op != "allreduce_wht"
        || !(body == crate::saber::AR_WPAR_SOURCE || body == crate::saber::AR_WPAR_SYMBOL)
    {
        return Err(refuse(format!(
            "the reduce is `{}` (`{}`), not the served wht6 bf16 one-shot (a race-check twin \
             stands the fold down)",
            ar.kernel, ar.op
        )));
    }
    let groups = u64::from(s.m) * u64::from(s.n) / 64;
    if int(ar, "n_groups")? != groups as i64 {
        return Err(refuse(format!(
            "the reduce carries {} groups and the GEMM writes {} x {} = {groups}",
            int(ar, "n_groups")?,
            s.m,
            s.n
        )));
    }
    let fuse_off = crate::saber::ar_fuse_off(saber).map_err(|e| refuse(e.to_string()))?;
    let (args, bytes) = Builder::new(abi::SIG_GEMM_DECODE_ARW)
        .buf("a")
        .buf("w")
        .buf("ws")
        .buf("wref")
        .buf("as")
        .buf("decode_partial")
        .buf("decode_counter")
        .buf("c")
        .i32(i64::from(s.m))
        .i32(i64::from(s.n))
        .i32(i64::from(s.k))
        .i32(int(g, "pb1")?)
        .i32(int(g, "pb2")?)
        .i64(int(g, "astride")?)
        .buf("ar_peer_scratch")
        .buf("ar_scratch")
        .buf("c")
        .buf("ar_peer_flags")
        .buf("ar_flags")
        .buf("ar_seq")
        .i64(int(ar, "slot_stride_bytes")?)
        .i64(fuse_off as i64)
        .i32(int(ar, "drain")?)
        .i32(int(ar, "acq")?)
        .finish()?;
    let mut workspaces: Vec<WorkspaceReq> = g.workspaces.clone();
    if !["decode_partial", "decode_counter"]
        .iter()
        .all(|n| workspaces.iter().any(|w| w.name == *n))
    {
        return Err(refuse(format!(
            "`{}` does not declare the split-K workspaces",
            g.kernel
        )));
    }
    for name in [
        "ar_scratch",
        "ar_peer_scratch",
        "ar_flags",
        "ar_peer_flags",
        "ar_seq",
    ] {
        let w = ar
            .workspaces
            .iter()
            .find(|w| w.name == name)
            .ok_or_else(|| refuse(format!("the reduce declares no `{name}`")))?;
        workspaces.push(w.clone());
    }
    Ok(LaunchDescriptor {
        op: "gemm_decode_arw".into(),
        plugin: GEMM_ARW_PLUGIN.into(),
        kernel: format!("mxfp4_gemm_decode_arw_bk{bk}_ks{ks}_tm{tm}_dwn4"),
        grid: g.grid,
        block: g.block,
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        n_launches: 1,
    })
}

/// The DIRECT cells fusion-map row 4 is compiled for (`the internal reference harness`):
/// `(bk, tm)` at `ks1`, `wperm1_nt1` -- every cell the decode band gives gate_up (N = 17,408 at
/// TP=2 fills the machine unsplit at every decode M).
pub const GEMM_SMQ_CELLS: &[(u32, u32)] = &[(128, 1), (128, 2), (128, 3), (128, 4), (64, 4)];

/// Row 4's Persistent arrival counter: MONOTONIC and 64-bit, like row 3's `ar_tiles`, and its
/// own name so a layer's two regions never share one.
pub const GEMM_SMQ_TILES: &str = "smq_tiles";

/// The code object row 4 lives in (its own, for row 3's autotune-key reason).
pub const GEMM_SMQ_PLUGIN: &str = "mxfp4_gemm_decode_smq";

/// Fusion-map ROW 4 (ws-h40): the gate_up decode GEMM `gemm` at `shape` and the
/// `silu_mul_quant` after it as ONE dispatch. The GEMM's DIRECT path writes its tile of the
/// gate_up row exactly as alone; then every tile takes a ticket on a monotonic counter and the
/// last `m` arrivals each run the producer's body on one row (the generated slice, its row an
/// argument). Refuses anything but a compiled `ks1` DIRECT cell, a `[M, 2N]` gate_up feeding a
/// `[M, N]` untiled 256-thread producer, and a grid with fewer n-tiles than rows.
///
/// `ep_rows` is `m` in production; the gate's `smq-rows` control passes `m - 1`.
///
/// # Errors
/// [`Error::Rule`] naming the first condition that fails.
pub fn gemm_decode_smq(
    gemm: (&LaunchDescriptor, Shape),
    silu: &LaunchDescriptor,
    rules: &LaunchRules,
) -> Result<LaunchDescriptor> {
    let (g, s) = gemm;
    let refuse = |why: String| Error::Rule(format!("gemm_decode_smq: {why}"));
    let (bk, ks, tm) = decode_cell(g)?;
    if ks != 1 || !GEMM_SMQ_CELLS.contains(&(bk, tm)) {
        return Err(refuse(format!(
            "`{}` is not a compiled row-4 cell (ks1 and (bk, tm) in {GEMM_SMQ_CELLS:?})",
            g.kernel
        )));
    }
    let kn = &rules.knobs;
    let bnd = kn.dec_dwn * 16;
    let block = kn.dec_dwn * 32;
    let ntiles = g.grid[0];
    let int = |d: &LaunchDescriptor, name: &str| match d.arg(name).map(|x| &x.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => Ok(*ival),
        _ => Err(refuse(format!("`{}` has no integer `{name}`", d.kernel))),
    };
    if s.n % bnd != 0 || ntiles * bnd != s.n {
        return Err(refuse(format!(
            "N = {} is not a whole number of {bnd}-column n-tiles",
            s.n
        )));
    }
    if silu.op != "silu_mul_quant" || silu.kernel != format!("silu_mul_quant_t{block}_tiled0") {
        return Err(refuse(format!(
            "the producer is `{}`, not the untiled {block}-thread `silu_mul_quant`",
            silu.kernel
        )));
    }
    if int(silu, "n")? * 2 != i64::from(s.n) || silu.grid[0] != s.m {
        return Err(refuse(format!(
            "the producer is {} x {} and the GEMM writes {} x {} (gate | up)",
            silu.grid[0],
            int(silu, "n")?,
            s.m,
            s.n
        )));
    }
    if s.m == 0 || ntiles < s.m {
        return Err(refuse(format!(
            "M = {} rows need M row owners among {ntiles} n-tiles",
            s.m
        )));
    }
    let (args, bytes) = Builder::new(abi::SIG_GEMM_DECODE_SMQ)
        .buf("a")
        .buf("w")
        .buf("ws")
        .buf("wref")
        .buf("as")
        // The DIRECT path (DKS == 1) never touches the split-K partial slab or its counter.
        .null_buf("decode_partial")
        .null_buf("decode_counter")
        .buf("c")
        .i32(i64::from(s.m))
        .i32(i64::from(s.n))
        .i32(i64::from(s.k))
        .i32(int(g, "pb1")?)
        .i32(int(g, "pb2")?)
        .i64(int(g, "astride")?)
        .buf("q")
        .buf("scale")
        .buf(GEMM_SMQ_TILES)
        .i32(i64::from(s.m))
        .finish()?;
    let mut workspaces: Vec<WorkspaceReq> = g.workspaces.clone();
    workspaces.push(WorkspaceReq {
        name: GEMM_SMQ_TILES.into(),
        kind: WorkspaceKind::Persistent,
        bytes: 8,
        zero_once: true,
    });
    Ok(LaunchDescriptor {
        op: "gemm_decode_smq".into(),
        plugin: GEMM_SMQ_PLUGIN.into(),
        kernel: format!("mxfp4_gemm_decode_smq_bk{bk}_ks1_tm{tm}_wperm1_nt1"),
        grid: g.grid,
        block: g.block,
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        n_launches: 1,
    })
}

/// The `DIRECT=false` decode CONTROL. Same inputs as [`Op::Gemm`] and the same output, but it
/// takes the split-K partial slab and the block counter at `DKS == 1` instead of writing C
/// directly -- so a byte compare against the direct arm proves the fast path, rather than the
/// source comment asserting it. Refuses any shape that does not resolve to the one cell the
/// instantiation was compiled for.
///
/// Deliberately NOT an [`Op`]: nothing in the engine routes to it, and a variant no production
/// path can select would only force a dead arm into every downstream match on `Op`. The gate and
/// `saber-kplan` call it by name.
pub fn gemm_decode_indirect(shape: Shape, rules: &LaunchRules) -> Result<LaunchDescriptor> {
    let Shape { m, n, k } = shape;
    let kn = &rules.knobs;
    let c = &rules.decode_direct_control;
    if k % kn.k_multiple != 0 {
        return Err(Error::Rule(format!(
            "K={k} must be a multiple of {}",
            kn.k_multiple
        )));
    }
    let bnd = kn.dec_dwn * 16;
    let nblk = n.div_ceil(bnd);
    let (ks, bk) = rules.decode_split(m, nblk, k)?;
    let tm = m.div_ceil(kn.dec_mtile);
    // One cell, and only one: the reference compiles this instantiation and no other.
    if (bk, ks, tm, kn.wperm, kn.decode_nt) != (c.bk, c.ks, c.tm, c.wperm_required, c.nt_required) {
        return Err(Error::Rule(format!(
            "decode DIRECT control: M={m} N={n} K={k} resolves to (bk {bk}, ks {ks}, tm {tm},              wperm {}, nt {}) but the only instantiation is (bk {}, ks {}, tm {}, wperm {},              nt {})",
            kn.wperm, kn.decode_nt, c.bk, c.ks, c.tm, c.wperm_required, c.nt_required
        )));
    }
    // The point of the control: bind the slab and the counter that the DIRECT arm never touches.
    let (args, bytes) = Builder::new(abi::SIG_GEMM_DECODE)
        .buf("a")
        .buf("w")
        .buf("ws")
        .buf("wref")
        .buf("as")
        .buf("decode_partial")
        .buf("decode_counter")
        .buf("c")
        .i32(m as i64)
        .i32(n as i64)
        .i32(k as i64)
        .i32(NO_PARTITIONS.pb1 as i64)
        .i32(NO_PARTITIONS.pb2 as i64)
        .i64(NO_PARTITIONS.astride)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gemm_decode_indirect".into(),
        plugin: "mxfp4_gemm_decode".into(),
        kernel: c.kernel.clone(),
        grid: [nblk, 1, ks],
        block: [kn.dec_dwn * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![
            WorkspaceReq {
                name: "decode_partial".into(),
                kind: rules.workspace.decode_partial.kind,
                bytes: ks as u64 * m as u64 * n as u64 * 4,
                zero_once: false,
            },
            WorkspaceReq {
                name: "decode_counter".into(),
                kind: rules.workspace.decode_counter.kind,
                bytes: nblk as u64 * 4,
                zero_once: true,
            },
        ],
        n_launches: 1,
    })
}

/// Does this M take the fragment-tiled activation layout? One threshold, used by BOTH the
/// producer (which writes the layout) and the consumer (which reads it) — in prod those live in
/// two files and must agree by convention.
pub fn a_tiled(m: u32, rules: &LaunchRules) -> bool {
    ATiledLaw::of(rules).tiled(m)
}

/// **`a_tiled_min_m`, and the one place it is written down.**
///
/// The M at and above which a fused quant producer emits fragment-tiled activations and the GEMM
/// that reads them takes the A-direct kernel. It used to be **four** numbers — this table's
/// `[knobs] a_tiled_min_m`, `rules/attn_glue_gfx1201.toml`'s `[attn_gate_quant]` copy,
/// `rules/paro_gfx1201.toml`'s `[knobs]` copy, and a `const A_TILED_MIN_M` in each of
/// `saber-model` and `saber-plan` — with a host test pinning each copy to this one.
///
/// A drift test is not a source of truth, and the difference cost L15 an experiment. Raising one
/// copy to 99999 to ask "is the A-tiled kernel reached?" left `attn_gate_quant` writing the
/// fragment layout at `m >= 513` while the GEMM reading it went row-major: the run swapped one
/// layout mismatch for a different one and measured the garbage that produced (P6). **A control
/// must differ from the case in the way the case differs from the truth**, and a one-line knob
/// edit is not a one-line experiment when the knob has four copies.
///
/// So the copies are gone. This function is the number, the two other rule tables no longer carry
/// the row (and are refused if they grow it back), and the two Rust constants are deleted:
/// `saber_kernels::glue::GateRow::a_tiled_min_m`, `saber_kernels::paro::ParoRules::tiled`,
/// `saber_model::graph::a_tiled_min_m` and `saber_plan::a_layout_key` all read it.
/// `a_tiled_min_m_is_written_down_once` fails if a second definition appears anywhere.
///
/// Note what this is NOT: [`a_tiled`] still takes a `&LaunchRules`, because a caller may be
/// enumerating over an EDITED policy (`ALayout::RowMajor` is expressed as `a_tiled_min_m = 0` on
/// a clone, in both sweepers). The committed number and a live table are different questions and
/// only the first one has one answer.
#[must_use]
pub fn a_tiled_min_m() -> u32 {
    static VALUE: LazyLock<u32> = LazyLock::new(|| {
        LaunchRules::prod_mxfp4()
            .expect("the committed mxfp4 rules are compiled in and parse")
            .knobs
            .a_tiled_min_m
    });
    *VALUE
}

/// Which A-side layout a **producer** writes at this M — `Some(true)` fragment-tiled,
/// `Some(false)` row-major, `None` for an op that produces no fp8 activation edge.
///
/// This is the half of the question [`a_tiled`] cannot answer, and prod does not answer it with a
/// threshold at all: `ref_arnq.py` calls `ref_mxfp4.a_tiled_register(q, M, K)` after
/// `add_rms_quant` and after `silu_mul_quant` and **not** after `gdn_norm_quant`, and the GEMM's
/// `a_tiled_take(q, M, K)` pops that registration — a buffer nobody registered takes the
/// row-major family however large M is. So the threshold is a property of the PRODUCER, and a
/// consumer that re-derives it from M is guessing about a buffer it did not write.
///
/// `gdn_norm_quant` has exactly one instantiation in the reference (`saber_gdn_norm_quant`,
/// templated on the thread count alone, store `q += row * N`), so under the reference's law it
/// answers `false` at every M. Saber adds the tiled store (`gdn_norm_quant_t{block}_tiled1`,
/// `hip/mxfp4_gdn_norm_tiled.h`, G2) behind the `gdn_norm_tiled` knob: with it on, this producer
/// switches at [`a_tiled`] exactly like the other two, and the GDN `out_proj` follows it.
#[must_use]
pub fn producer_writes_tiled(op: Op, m: u32, rules: &LaunchRules) -> Option<bool> {
    match op {
        Op::AddRmsQuant { .. } | Op::SiluMulQuant => Some(a_tiled(m, rules)),
        Op::GdnNormQuant { .. } => Some(rules.knobs.gdn_norm_tiled && a_tiled(m, rules)),
        Op::Gemm { .. } => None,
    }
}

fn check_partitions(n: u32, p: Partitions, rules: &LaunchRules) -> Result<()> {
    let align = rules.knobs.partition_align as i32;
    for (name, b) in [("pb1", p.pb1), ("pb2", p.pb2)] {
        if b < n as i32 && b % align != 0 {
            return Err(Error::Rule(format!(
                "partition boundary {name}={b} must be a multiple of {align}"
            )));
        }
    }
    Ok(())
}

fn gemm(
    shape: Shape,
    has_wref: bool,
    p: Partitions,
    rules: &LaunchRules,
    a_layout: ALayout,
) -> Result<LaunchDescriptor> {
    let Shape { m, n, k } = shape;
    let kn = &rules.knobs;
    if k % kn.k_multiple != 0 {
        return Err(Error::Rule(format!(
            "K={k} must be a multiple of {}",
            kn.k_multiple
        )));
    }
    check_partitions(n, p, rules)?;
    if !has_wref && p.pb1 < n as i32 {
        return Err(Error::Rule(
            "partitions need the folded path (no wref)".into(),
        ));
    }

    if !has_wref {
        // The unfolded kernel: no per-row reference exponent, per-32-block rescale in the loop.
        let (args, bytes) = Builder::new(abi::SIG_GEMM_BASE)
            .buf("a")
            .buf("w")
            .buf("ws")
            .buf("as")
            .buf("c")
            .i32(m as i64)
            .i32(n as i64)
            .i32(k as i64)
            .finish()?;
        return Ok(LaunchDescriptor {
            op: "gemm_base".into(),
            plugin: "mxfp4_gemm_base".into(),
            kernel: "mxfp4_gemm_base".into(),
            grid: [n.div_ceil(kn.base_bn), m.div_ceil(kn.base_bm), 1],
            block: [kn.base_block, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: bytes,
            args,
            workspaces: vec![],
            n_launches: 1,
        });
    }

    let wp = u8::from(kn.wperm);
    let tiled = a_layout == ALayout::FollowsM && a_tiled(m, rules);
    // G6: the skinny-N body, over whichever A layout the edge carries. It sums every output in
    // the folded / A-tiled / one-split decode order, so it may replace any of those cells -- but
    // never a SPLIT decode cell (a different summation), and only for the served layout.
    if kn.skinny && kn.wperm && kn.epifast && p.pb1 >= n as i32 {
        if let Some(row) = rules.skinny.iter().find(|r| r.matches(m, n, tiled)) {
            let one_split = m > kn.decode_max_m || {
                let nblk = n.div_ceil(kn.dec_dwn * 16);
                let (ks, _) = rules.decode_split(m, nblk, k)?;
                ks == 1
            };
            if one_split && n % 16 == 0 && k % (16 * row.unr) == 0 {
                let (args, bytes) = gemm_folded_args(shape, p)?;
                return Ok(LaunchDescriptor {
                    op: "gemm_skinny".into(),
                    plugin: MID_PLUGIN.into(),
                    kernel: row.kernel(),
                    grid: [
                        n.div_ceil(16 * row.tns),
                        m.div_ceil(16).div_ceil(row.wpb),
                        1,
                    ],
                    block: [row.wpb * 32, 1, 1],
                    shared_bytes: 0,
                    kernarg_bytes: bytes,
                    args,
                    workspaces: vec![],
                    n_launches: 1,
                });
            }
        }
    }
    // The A-tiled kernel reads A already in WMMA fragment order. A row-major edge falls through
    // to the folded tile below, which is the same arithmetic over row-major A -- one kernel
    // slower, and correct.
    if tiled {
        if k % kn.atiled_k_multiple != 0 {
            return Err(Error::Rule(format!(
                "tiled A needs K % {} == 0 (K={k})",
                kn.atiled_k_multiple
            )));
        }
        // G6: below `a_tiled_min_m` (the mid band of the layout law), a mid-M height of the same
        // A-tiled tile where a row names one. Compiled for the served layout only.
        if m < kn.a_tiled_min_m && kn.wperm && kn.epifast && kn.folded_wn == 2 {
            if let Some(row) = rules.atiled_mt.iter().find(|r| r.matches(m, n)) {
                let (args, bytes) = gemm_folded_args(shape, p)?;
                return Ok(LaunchDescriptor {
                    op: "gemm_atiled".into(),
                    plugin: MID_PLUGIN.into(),
                    kernel: row.kernel(),
                    grid: [
                        n.div_ceil(kn.folded_wn * row.tn * 16),
                        m.div_ceil(row.rows()),
                        1,
                    ],
                    block: [row.wm * 64, 1, 1],
                    shared_bytes: 0,
                    kernarg_bytes: bytes,
                    args,
                    workspaces: vec![],
                    n_launches: 1,
                });
            }
        }
        let tn = rules.tn_for(m, n, true);
        let lbk = rules.lbk_for(tn)?;
        let bnf = kn.folded_wn * tn * 16;
        // The tile height: the reference's four waves (`folded_bmf` rows, `folded_block`
        // threads), or a Saber-authored taller tile that scales both with it (G1).
        let (bmf, block, kernel) = match rules.wm_for(m, tn, kn.wperm) {
            Some(row) if row.wm != 4 || row.variant.is_some() => (
                kn.folded_bmf * row.wm / 4,
                kn.folded_block * row.wm / 4,
                format!(
                    "mxfp4_gemm_atiled_tn{tn}_lbk{lbk}_wperm{wp}{}",
                    row.suffix()
                ),
            ),
            _ => (
                kn.folded_bmf,
                kn.folded_block,
                format!("mxfp4_gemm_atiled_tn{tn}_lbk{lbk}_wperm{wp}"),
            ),
        };
        let (args, bytes) = gemm_folded_args(shape, p)?;
        return Ok(LaunchDescriptor {
            op: "gemm_atiled".into(),
            plugin: "mxfp4_gemm_atiled".into(),
            kernel,
            grid: [n.div_ceil(bnf), m.div_ceil(bmf), 1],
            block: [block, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: bytes,
            args,
            workspaces: vec![],
            n_launches: 1,
        });
    }

    // ---- decode band -------------------------------------------------------------------
    let bnd = kn.dec_dwn * 16;
    let nblk = n.div_ceil(bnd);
    let (mut ks, mut bk) = rules.decode_split(m, nblk, k)?;
    if k % rules.k_align_override.k_modulus != 0 {
        bk = rules.k_align_override.bk;
        if ks == rules.k_align_override.ks_from {
            ks = rules.k_align_override.ks_to;
        }
    }
    let partial_bytes = ks as u64 * m as u64 * n as u64 * 4;
    let scratch_ok = partial_bytes <= rules.workspace.decode_partial.capacity_bytes;
    // `dec_max_n` guards the SPLIT arm only, and this is where Saber finishes a sentence the
    // reference starts. ref_mxfp4_fp8.hip:1215 says it itself -- "DKS==1 writes C directly and
    // needs neither the partial buffer nor the block counter [...] Sizing the guard by the CHOSEN
    // ks (not the widest) also means a shape that would have overflowed the scratch at DKS=4 can
    // still take the kernel at DKS=1" -- and then ANDs `N <= DEC_MAX_N` in next to that guard
    // anyway. Both of the kernel's N-proportional allocations are the split's: the partial slab
    // (`DKS*M*N` floats) and the per-n-block counter (`DEC_MAX_N/128` ints). At DKS=1 this
    // descriptor binds both slots NULL, so the cap is guarding an allocation that is not there.
    //
    // What it costs to leave it: `lm_head` (N = 124160 at TP=2, 248320 at TP=1) is the one shape in
    // the model wider than 36864, so at a decode M it fell through to the PREFILL folded tile,
    // which stages BMF = 256 rows of A for the 5 the bucket has. Measured at the TP=2 shard,
    // paired A/B on one stream against the reference's own launch (`the internal reference harness`):
    //
    //     M      folded (ref)   decode (ours)   ratio
    //     1        1410.8 us        511.6 us    0.363
    //     5        1425.9           518.6       0.364
    //    10        1534.6           535.0       0.349
    //    20        1656.1           580.3       0.350
    //    40        1710.1           615.8       0.360
    //   128        1818.9          1814.1       0.997   <- control: above decode_max_m, unchanged
    //
    // 338 MB of weights at 635 GB/s is a 532 us floor, so the decode arm runs this GEMM AT the
    // DRAM roofline and the folded arm at 38% of it. Bit-identical at every cell (the gate's
    // `gemm {1,5,10,20,40,64} 124160 5120`), which is what the ks=1 arm's ascending-k fp32
    // accumulation buys: no reassociation, unlike a split (see the I11 entry).
    let in_band = m > 0
        && m <= kn.decode_max_m
        && m <= kn.dec_mtile * kn.dec_max_tm
        && (ks == 1 || (n <= kn.dec_max_n && scratch_ok));
    if in_band {
        let tm = m.div_ceil(kn.dec_mtile);
        let nt = u8::from(kn.decode_nt && kn.wperm);
        // At DKS==1 the `DIRECT` instantiation returns before it ever touches the partial slab or
        // the counter, so both slots are bound null rather than to a workspace nobody allocates.
        let split = ks > 1;
        let mut b = Builder::new(abi::SIG_GEMM_DECODE)
            .buf("a")
            .buf("w")
            .buf("ws")
            .buf("wref")
            .buf("as");
        b = if split {
            b.buf("decode_partial").buf("decode_counter")
        } else {
            b.null_buf("decode_partial").null_buf("decode_counter")
        };
        let (args, bytes) = b
            .buf("c")
            .i32(m as i64)
            .i32(n as i64)
            .i32(k as i64)
            .i32(p.pb1 as i64)
            .i32(p.pb2 as i64)
            .i64(p.astride)
            .finish()?;
        let mut workspaces = Vec::new();
        if split {
            workspaces.push(WorkspaceReq {
                name: "decode_partial".into(),
                kind: rules.workspace.decode_partial.kind,
                bytes: partial_bytes,
                zero_once: false,
            });
            workspaces.push(WorkspaceReq {
                name: "decode_counter".into(),
                kind: rules.workspace.decode_counter.kind,
                bytes: nblk as u64 * 4,
                zero_once: true,
            });
        }
        // CMP27E: the register-prefetch twin of a compiled cell (bit-exact), under the knob.
        let pf = wp == 1
            && nt == 1
            && ((kn.decode_pf && DECODE_PF_CELLS.contains(&(bk, ks, tm)))
                || (kn.decode_pf_tm2 && DECODE_PF_TM2_CELLS.contains(&(bk, ks, tm))));
        // CMP27X: the prefetch twin at 4 waves a workgroup (same `ks`, 2x the workgroups).
        let dwn4 = pf
            && kn.decode_pf_dwn4
            && kn.dec_dwn == 8
            && DECODE_PF_DWN4_CELLS.contains(&(bk, ks, tm))
            && DECODE_PF_DWN4_N.contains(&n);
        let (nblk, dwn) = if dwn4 {
            (n.div_ceil(64), 4)
        } else {
            (nblk, kn.dec_dwn)
        };
        if dwn4 && split {
            if let Some(c) = workspaces.iter_mut().find(|w| w.name == "decode_counter") {
                c.bytes = nblk as u64 * 4;
            }
        }
        return Ok(LaunchDescriptor {
            op: "gemm_decode".into(),
            plugin: if pf {
                DECODE_PF_PLUGIN
            } else {
                "mxfp4_gemm_decode"
            }
            .into(),
            kernel: if dwn4 {
                format!("mxfp4_gemm_decode_pf_bk{bk}_ks{ks}_tm{tm}_wperm1_nt1_dwn4")
            } else if pf {
                format!("mxfp4_gemm_decode_pf_bk{bk}_ks{ks}_tm{tm}_wperm1_nt1")
            } else {
                format!("mxfp4_gemm_decode_bk{bk}_ks{ks}_tm{tm}_wperm{wp}_nt{nt}")
            },
            grid: [nblk, 1, ks],
            block: [dwn * 32, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: bytes,
            args,
            workspaces,
            n_launches: 1,
        });
    }

    // ---- folded prefill tile ------------------------------------------------------------
    let tn = rules.tn_for(m, n, false);
    let bnf = kn.folded_wn * tn * 16;
    let ef = u8::from(kn.epifast);
    let (args, bytes) = gemm_folded_args(shape, p)?;
    // G5: a mid-M height of the same tile, where a row names one and the knob is on. Compiled for
    // the served layout only (TN 2, fragment-order W, EPIFAST): any other cell keeps the reference.
    if kn.folded_mt && tn == 2 && kn.wperm && kn.epifast && kn.folded_wn == 2 {
        if let Some(row) = rules.folded_mt.iter().find(|r| r.matches(m, n)) {
            return Ok(LaunchDescriptor {
                op: "gemm_folded".into(),
                plugin: FOLDED_MT_PLUGIN.into(),
                kernel: format!("mxfp4_gemm_folded_mt_tn2_wperm1_t{}w{}", row.tm, row.wm),
                grid: [n.div_ceil(bnf), m.div_ceil(row.rows()), 1],
                block: [row.wm * 64, 1, 1],
                shared_bytes: 0,
                kernarg_bytes: bytes,
                args,
                workspaces: vec![],
                n_launches: 1,
            });
        }
    }
    Ok(LaunchDescriptor {
        op: "gemm_folded".into(),
        plugin: "mxfp4_gemm_folded".into(),
        kernel: format!("mxfp4_gemm_folded_tn{tn}_wperm{wp}_epifast{ef}"),
        grid: [n.div_ceil(bnf), m.div_ceil(kn.folded_bmf), 1],
        block: [kn.folded_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn gemm_folded_args(shape: Shape, p: Partitions) -> Result<(Vec<Slot>, u32)> {
    Builder::new(abi::SIG_GEMM_FOLDED)
        .buf("a")
        .buf("w")
        .buf("ws")
        .buf("wref")
        .buf("as")
        .buf("c")
        .i32(shape.m as i64)
        .i32(shape.n as i64)
        .i32(shape.k as i64)
        .i32(p.pb1 as i64)
        .i32(p.pb2 as i64)
        .i64(p.astride)
        .finish()
}

fn add_rms_quant(shape: Shape, eps: f32, rules: &LaunchRules) -> Result<LaunchDescriptor> {
    let kn = &rules.knobs;
    let (m, k) = (shape.m, shape.k);
    if k > kn.quant_max_k || k % 8 != 0 {
        return Err(Error::Rule(format!(
            "add_rms_quant: K={k} must be 8-aligned and <= {}",
            kn.quant_max_k
        )));
    }
    let tiled = a_tiled(m, rules);
    if tiled && k % 16 != 0 {
        return Err(Error::Rule(format!(
            "add_rms_quant: tiled needs K % 16 == 0 (K={k})"
        )));
    }
    let t = if m >= kn.quant_large_min_m {
        kn.quant_large_block
    } else {
        kn.quant_small_block
    };
    let (args, bytes) = Builder::new(abi::SIG_ADD_RMS_QUANT)
        .buf("y")
        .buf("res")
        .buf("w")
        .buf("q")
        .buf("scale")
        .buf("res_out")
        .i32(k as i64)
        .f32(eps)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "add_rms_quant".into(),
        plugin: "mxfp4_quant".into(),
        kernel: format!("add_rms_quant_t{t}_tiled{}", u8::from(tiled)),
        grid: [m, 1, 1],
        block: [t, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn silu_mul_quant(shape: Shape, rules: &LaunchRules) -> Result<LaunchDescriptor> {
    let kn = &rules.knobs;
    let (m, n) = (shape.m, shape.n);
    if n > kn.silu_max_n || n % 8 != 0 {
        return Err(Error::Rule(format!(
            "silu_mul_quant: N={n} must be 8-aligned and <= {}",
            kn.silu_max_n
        )));
    }
    let tiled = a_tiled(m, rules);
    if tiled && n % 16 != 0 {
        return Err(Error::Rule(format!(
            "silu_mul_quant: tiled needs N % 16 == 0 (N={n})"
        )));
    }
    // Above the 256-thread block's register budget the 512-thread instantiation is the only one
    // that covers N; below it, 512 threads are for prefill-class M only.
    let t = if n > kn.silu_large_n || m >= kn.quant_large_min_m {
        kn.quant_large_block
    } else {
        kn.quant_small_block
    };
    let (args, bytes) = Builder::new(abi::SIG_SILU_MUL_QUANT)
        .buf("gu")
        .buf("q")
        .buf("scale")
        .i32(n as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "silu_mul_quant".into(),
        plugin: "mxfp4_quant".into(),
        kernel: format!("silu_mul_quant_t{t}_tiled{}", u8::from(tiled)),
        grid: [m, 1, 1],
        block: [t, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn gdn_norm_quant(
    shape: Shape,
    zs: i64,
    eps: f32,
    rules: &LaunchRules,
) -> Result<LaunchDescriptor> {
    let kn = &rules.knobs;
    let (m, n) = (shape.m, shape.n);
    if n > kn.gdn_max_n || n % kn.gdn_n_multiple != 0 {
        return Err(Error::Rule(format!(
            "gdn_norm_quant: N={n} must be a multiple of {} and <= {}",
            kn.gdn_n_multiple, kn.gdn_max_n
        )));
    }
    if zs % 8 != 0 {
        return Err(Error::Rule(format!(
            "gdn_norm_quant: z row stride {zs} must be 8-aligned"
        )));
    }
    // The layout is the producer's own answer, the one `producer_writes_tiled` gives the planner.
    let tiled = producer_writes_tiled(Op::GdnNormQuant { zs, eps }, m, rules) == Some(true);
    if tiled && n % 16 != 0 {
        return Err(Error::Rule(format!(
            "gdn_norm_quant: tiled needs N % 16 == 0 (N={n})"
        )));
    }
    let (args, bytes) = Builder::new(abi::SIG_GDN_NORM_QUANT)
        .buf("x")
        .buf("z")
        .i64(zs)
        .buf("w")
        .buf("q")
        .buf("scale")
        .i32(n as i64)
        .f32(eps)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_norm_quant".into(),
        plugin: "mxfp4_quant".into(),
        kernel: if tiled {
            format!("gdn_norm_quant_t{}_tiled1", kn.quant_small_block)
        } else {
            format!("gdn_norm_quant_t{}", kn.quant_small_block)
        },
        grid: [m, 1, 1],
        block: [kn.quant_small_block, 1, 1],
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

    fn rules() -> LaunchRules {
        LaunchRules::prod_mxfp4().expect("prod rules parse")
    }

    /// CMP27E: the `decode_pf` knob is OFF in the committed table; ON, exactly the b8 cells the
    /// twins are compiled for move to their plugin, each in its shipped cell's summation class,
    /// and a cell with no twin (b2's tm1) keeps the shipped symbol -- the negative control.
    #[test]
    fn decode_pf_knob_routes_the_b8_cells_to_the_prefetch_twins() {
        let off = rules();
        assert!(!off.knobs.decode_pf);
        let mut on = rules();
        on.knobs.decode_pf = true;
        for (n, k) in [
            (5120, 3072),
            (5120, 8704),
            (17408, 5120),
            (7168, 5120),
            (8192, 5120),
        ] {
            let a = resolve(Op::gemm(), Shape::new(64, n, k), &off).expect("off");
            let b = resolve(Op::gemm(), Shape::new(64, n, k), &on).expect("on");
            assert_eq!(a.grid, b.grid);
            assert_eq!(a.args, b.args);
            if a.kernel.contains("_tm4_")
                && (a.kernel.contains("bk128_ks2") || a.kernel.contains("bk64_ks1"))
            {
                assert_eq!(b.plugin, DECODE_PF_PLUGIN, "{}", a.kernel);
                assert_ne!(a.kernel, b.kernel);
            } else {
                assert_eq!(a.kernel, b.kernel);
            }
            assert_eq!(summation_class(&b.kernel), a.kernel.as_str());
        }
        // the pair at M 64 (qkvz + ba) takes the twin; at M 16 (tm1) it keeps the shipped pair.
        for (m, twin) in [(64, true), (16, false)] {
            let pair = |r: &LaunchRules| {
                let q = resolve(Op::gemm(), Shape::new(m, 8192, 5120), r).expect("qkvz");
                let b = resolve(Op::gemm(), Shape::new(m, 48, 5120), r).expect("ba");
                gemm_decode_pair(
                    (&q, Shape::new(m, 8192, 5120)),
                    (&b, Shape::new(m, 48, 5120)),
                    r,
                )
                .expect("pair")
            };
            let (a, b) = (pair(&off), pair(&on));
            assert_eq!(a.grid, b.grid);
            assert_eq!(a.args, b.args);
            assert_eq!(
                b.kernel == DECODE_PF_PAIR,
                twin,
                "{} -> {}",
                a.kernel,
                b.kernel
            );
            assert_eq!(summation_class(&b.kernel), a.kernel.as_str());
        }
    }

    /// CMP27T: the `decode_pf_tm2` knob is OFF in the committed table; ON, exactly the conc-4
    /// `bk128 ks2 tm2` cell moves to the prefetch body's plugin, in its shipped cell's summation
    /// class; every other cell (tm1, tm4, gate_up's ks1 tm2, the pairs) keeps its symbol.
    #[test]
    fn cmp27t_decode_pf_tm2_knob_routes_only_the_conc4_cells() {
        let off = rules();
        assert!(!off.knobs.decode_pf_tm2);
        let mut on = rules();
        on.knobs.decode_pf_tm2 = true;
        for m in [16, 17, 32, 64] {
            for (n, k) in [
                (5120, 3072),
                (5120, 8704),
                (17408, 5120),
                (7168, 5120),
                (8192, 5120),
            ] {
                let a = resolve(Op::gemm(), Shape::new(m, n, k), &off).expect("off");
                let b = resolve(Op::gemm(), Shape::new(m, n, k), &on).expect("on");
                assert_eq!(a.grid, b.grid);
                assert_eq!(a.args, b.args);
                assert_eq!(a.workspaces.len(), b.workspaces.len());
                let tm2 = a.kernel.contains("_tm2_") && a.kernel.contains("bk128_ks2");
                if m == 32 && n != 17408 {
                    assert!(tm2, "the conc-4 cell M 32 N {n}: {}", a.kernel);
                }
                if tm2 {
                    assert_eq!(b.plugin, DECODE_PF_PLUGIN, "{}", a.kernel);
                    assert_ne!(a.kernel, b.kernel);
                } else {
                    assert_eq!(a.kernel, b.kernel, "M {m} N {n}");
                }
                assert_eq!(summation_class(&b.kernel), a.kernel.as_str());
            }
            let pair = |r: &LaunchRules| {
                let q = resolve(Op::gemm(), Shape::new(m, 8192, 5120), r).expect("qkvz");
                let b = resolve(Op::gemm(), Shape::new(m, 48, 5120), r).expect("ba");
                gemm_decode_pair(
                    (&q, Shape::new(m, 8192, 5120)),
                    (&b, Shape::new(m, 48, 5120)),
                    r,
                )
                .expect("pair")
            };
            let (a, b) = (pair(&off), pair(&on));
            assert_eq!(a.grid, b.grid);
            assert_eq!(a.args, b.args);
            // no tm2 pair twin (measured slower): the pair keeps its shipped symbol
            assert_eq!(a.kernel, b.kernel, "M {m}");
            assert_eq!(summation_class(&b.kernel), a.kernel.as_str());
        }
    }

    /// CMP27X: the `decode_pf_dwn4` knob is OFF in the committed table; ON (with the prefetch
    /// knobs), exactly the N 5120 prefetch cells take the `_dwn4` twin: the same split `ks`, 2x
    /// the column blocks, 4 waves a workgroup, a counter a block, in the DWN 8 cell's summation
    /// class; every other shape keeps its descriptor.
    #[test]
    fn cmp27x_decode_pf_dwn4_knob_routes_only_the_n5120_prefetch_cells() {
        let mut off = rules();
        off.knobs.decode_pf = true;
        off.knobs.decode_pf_tm2 = true;
        assert!(!rules().knobs.decode_pf_dwn4);
        let mut on = off.clone();
        on.knobs.decode_pf_dwn4 = true;
        let mut moved = 0;
        for m in [1, 10, 16, 17, 32, 33, 64] {
            for (n, k) in [
                (5120, 3072),
                (5120, 8704),
                (17408, 5120),
                (7168, 5120),
                (8192, 5120),
            ] {
                let a = resolve(Op::gemm(), Shape::new(m, n, k), &off).expect("off");
                let b = resolve(Op::gemm(), Shape::new(m, n, k), &on).expect("on");
                let tw = a.kernel.starts_with("mxfp4_gemm_decode_pf_bk128_ks2_tm")
                    && (a.kernel.contains("_tm2_") || a.kernel.contains("_tm4_"))
                    && n == 5120;
                if tw {
                    moved += 1;
                    assert_eq!(b.kernel, format!("{}_dwn4", a.kernel));
                    assert_eq!(b.grid, [a.grid[0] * 2, 1, a.grid[2]]);
                    assert_eq!(b.block, [128, 1, 1]);
                    assert_eq!(a.args, b.args);
                    let cnt = |d: &LaunchDescriptor| {
                        d.workspaces
                            .iter()
                            .find(|w| w.name == "decode_counter")
                            .map(|w| w.bytes)
                    };
                    assert_eq!(cnt(&b), Some(u64::from(b.grid[0]) * 4));
                    assert_eq!(summation_class(&b.kernel), summation_class(&a.kernel));
                } else {
                    assert_eq!(a.kernel, b.kernel, "M {m} N {n}");
                    assert_eq!(a.grid, b.grid);
                    assert_eq!(a.block, b.block);
                }
            }
        }
        assert!(moved >= 4, "{moved}");
    }

    #[test]
    fn prod_rules_load_and_check() {
        let r = rules();
        assert_eq!(r.knobs.decode_max_m, 64);
        assert_eq!(r.knobs.a_tiled_min_m, 513);
    }

    #[test]
    fn negative_control_rules_that_break_an_invariant_are_refused() {
        // a_tiled_min_m below the decode band: prod raises at import; so must we.
        let bad = PROD_RULES.replace("a_tiled_min_m = 513", "a_tiled_min_m = 64");
        assert!(LaunchRules::from_toml(&bad).is_err());
    }

    /// `gate_up` at the 1x5 bucket, back on BK=128 since 2026-09-20.
    ///
    /// It was BK=64 for one day, on a sweep that measured this shape's 47.4 MB of weights out of
    /// the Infinity Cache. Measured IC-cold it is the other way round by 10.3% (96.22 vs 106.10
    /// µs), and 106.10 µs is the 450 GB/s H13 measured for the same kernel in chain. The row is
    /// withdrawn in `rules/mxfp4_gfx1201.toml`, where the numbers are.
    #[test]
    fn decode_at_m5_gate_up_tp2() {
        let r = rules();
        let d = resolve(Op::gemm(), Shape::new(5, 17408, 5120), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_decode_bk128_ks1_tm1_wperm1_nt1");
        assert_eq!(d.grid, [136, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        assert!(d.workspaces.is_empty(), "DKS=1 needs no partial buffer");
        assert_eq!(d.n_launches, 1);
    }

    #[test]
    fn split_needs_both_workspaces_and_declares_their_kinds() {
        let r = rules();
        let d = resolve(Op::gemm(), Shape::new(5, 5120, 6144), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1");
        assert_eq!(d.grid, [40, 1, 4]);
        let p = &d.workspaces[0];
        assert_eq!(p.kind, WorkspaceKind::Scratch);
        assert_eq!(p.bytes, 4 * 5 * 5120 * 4);
        let c = &d.workspaces[1];
        assert_eq!(c.kind, WorkspaceKind::Persistent);
        assert!(c.zero_once);
        assert_eq!(c.bytes, 40 * 4);
    }

    #[test]
    fn k_not_a_multiple_of_128_takes_bk64_and_widens_split_2() {
        let r = rules();
        // TP=3 padded down_proj: K = 5824 = 64*91, a multiple of 64 but not 128.
        let d = resolve(Op::gemm(), Shape::new(64, 5120, 5824), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_decode_bk64_ks4_tm4_wperm1_nt1");
        // nblk = 55 is the band where the fill rule answers 2; BK=64 exists only for splits 1
        // and 4, so the split-2 cell widens.
        assert_eq!(r.split_k(55, 24), 2);
        let d = resolve(Op::gemm(), Shape::new(24, 7040, 5824), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_decode_bk64_ks4_tm2_wperm1_nt1");
    }

    /// The per-edge layout obligation, at the kernel plane: told the edge is row-major, the
    /// same shape that would have taken the A-tiled kernel takes the folded tile instead.
    #[test]
    fn a_row_major_edge_does_not_take_the_a_tiled_kernel_however_large_m_is() {
        let r = LaunchRules::prod_mxfp4().unwrap();
        let shape = Shape::new(8192, 5120, 3072);
        // The default: assume a fused producer, so M decides.
        let fused = resolve(Op::gemm(), shape, &r).expect("resolve");
        assert_eq!(fused.plugin, "mxfp4_gemm_atiled");
        // Told otherwise, the same shape takes the row-major family.
        let row = resolve_with_layout(Op::gemm(), shape, &r, ALayout::RowMajor).expect("resolve");
        assert_eq!(row.plugin, "mxfp4_gemm_folded");
        assert_ne!(fused.kernel, row.kernel);
        // Below the threshold the argument changes nothing: everything is row-major there.
        let small = Shape::new(8, 5120, 5120);
        assert_eq!(
            resolve(Op::gemm(), small, &r).unwrap().kernel,
            resolve_with_layout(Op::gemm(), small, &r, ALayout::RowMajor)
                .unwrap()
                .kernel
        );
    }

    /// The two prefill kernels do not share an N tile, and the table says which one a row is
    /// for. Same M, same N, same K down both columns -- only the kernel differs, and the
    /// measurement picks the opposite tile in each (`rules/mxfp4_gfx1201.toml`, 2026-09-19:
    /// folded TN2 328.60 vs TN4 358.76 us, A-tiled TN4 271.66 vs TN2 286.80 us at M=2048).
    #[test]
    fn the_two_prefill_families_do_not_share_an_n_tile() {
        let r = LaunchRules::prod_mxfp4().unwrap();
        for m in [2048, 8192] {
            let shape = Shape::new(m, 5120, 3072);
            let fused = resolve(Op::gemm(), shape, &r).expect("resolve");
            // G3's `_wm8sp` body at the 2048-row chunk and (G4) up to the 8192-row chunk: the
            // A-tiled family is the 512-row tile, same TN.
            let (want, bm) = ("mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp", 512);
            assert_eq!(fused.kernel, want, "the A-tiled family keeps TN=4 at M={m}");
            let row = resolve_with_layout(Op::gemm(), shape, &r, ALayout::RowMajor).expect("row");
            assert_eq!(
                row.kernel, "mxfp4_gemm_folded_tn2_wperm1_epifast1",
                "the folded family takes TN=2 at M={m}"
            );
            // TN=2 is BNF=64, so the same N is covered by twice the column blocks: the "the wide
            // tile cannot fill" half of the argument, in the grid rather than in prose.
            assert_eq!(row.grid, [80, m.div_ceil(256), 1]);
            assert_eq!(fused.grid, [40, m.div_ceil(bm), 1]);
        }
    }

    /// The negative control for the row above: with `a_tiled` dropped from the TN=4 row -- the
    /// table exactly as it shipped before 2026-09-19 -- the row-major cell answers TN=4 again,
    /// which is the launch the measurement rejects. A rule whose removal changes nothing would
    /// not be evidence of anything.
    #[test]
    fn without_the_family_key_the_row_major_cell_goes_back_to_the_slower_tile() {
        let mut r = LaunchRules::prod_mxfp4().unwrap();
        for row in &mut r.folded_tn {
            row.a_tiled = None;
        }
        let shape = Shape::new(2048, 5120, 3072);
        assert_eq!(
            resolve_with_layout(Op::gemm(), shape, &r, ALayout::RowMajor)
                .unwrap()
                .kernel,
            "mxfp4_gemm_folded_tn4_wperm1_epifast1"
        );
        // ... and the A-tiled side is untouched by the key either way, which is why the change
        // costs that family nothing.
        assert_eq!(
            resolve(Op::gemm(), shape, &r).unwrap().kernel,
            "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp"
        );
    }

    /// The split-K fill target is measurably too high for the `ks = 1` step, and the rule stays
    /// anyway. This test is the finding, executable, so it cannot be lost.
    ///
    /// `saber tune`, device 1, prod down, 210 W, 5 warm-up + 20 interleaved rounds, BK=128
    /// medians in µs (`*` = fastest, three runs agree on every cell to within 0.2 µs):
    ///
    /// | nblk | shape | M=5 | M=10 | M=20 | M=40 |
    /// |---|---|---|---|---|---|
    /// | 1 | n48 | ks4* 17.18 (ks1 37.74) | ks4* | ks4* | ks4* |
    /// | 40 | n5120 k3072 | ks4* 23.58 / ks2 23.80 / ks1 27.62 | ks2* | ks4* 27.80 / ks1 32.08 | ks2* |
    /// | 40 | n5120 k8704 | ks2* 46.68 / ks1 59.90 | ks2* | ks2* | ks4* |
    /// | 56 | n7168 | ks2* 40.36 / ks1 41.34 | ks2* 41.56 / ks1 41.92 | ks1* 43.24 / ks2 43.50 | ks2* 47.76 / ks1 49.38 |
    /// | 64 | n8192 | ks1* 44.22 / ks2 44.38 | ks1* 44.96 / ks2 45.50 | ks1* 45.12 / ks2 47.28 | ks1* 51.28 / ks2 51.74 |
    /// | 136 | n17408 | ks4* 83.22 / ks1 84.10 | ks1* | ks1* | ks1* 90.64 / ks2 96.08 |
    ///
    /// The axis is nblk, not M: `ks = 1` wins at every M at nblk 64 and 136, and loses at every M
    /// at nblk 40 (by 15-28%) and at three of four at nblk 56. The boundary therefore sits between
    /// nblk 56 and 64 -- which is the device's 64 CUs, i.e. the width at which one workgroup per
    /// CU already fills the machine and a split only adds partial write-back. The shipped target
    /// (110 at M <= 24, 78 above) is the reference's, and it is too high by that reading.
    ///
    /// It is NOT taken, for two reasons, and both are measured rather than argued:
    ///
    ///  1. A split-K change is the one move in this table that is **not output-preserving**. The
    ///     splits accumulate in fp32 and the reduce adds them, so ks=1 and ks=2 associate the
    ///     k-sum differently and the last bf16 bit moves. Replayed against the captured reference
    ///     output (`the internal reference harness`'s data, no new capture): `gemm 8 10240 5120` **3 bytes
    ///     of 163840** differ, `gemm 8 12288 5120` **2 of 196608**. Every other move Saber has
    ///     made to this table (I9's BK, I10's TN) was byte-identical; this one would trade the
    ///     engine's byte-equality with prod for the win below, and Saber has no end-task gate yet
    ///     to price that trade.
    ///  2. The win is one cell. Of the eight cells in the row, four are already `ks = 1`, three
    ///     are inside the ±2.2% floor, and exactly one is outside it: `layer.gdn.in_proj_qkvz` at
    ///     M=20, 47.28 -> 45.12 µs, ×48 per step = **0.10 ms/step at the `b4s5c8192` bucket**.
    ///
    /// Prod pays the same 2.16 µs × 48 at that bucket, and for prod the trade is priceable (it has
    /// GSM8K). That is where this finding should be spent; see the LEDGER entry.
    #[test]
    fn the_split_k_fill_is_too_high_for_the_ks1_step_and_the_rule_stays_anyway() {
        let r = rules();
        // What ships, at the four decode M the plan produces. nblk 64 is the suboptimal cell.
        for (m, tm) in [(5u32, 1u32), (10, 1), (20, 2), (40, 3)] {
            let d = resolve(Op::gemm(), Shape::new(m, 8192, 5120), &r).expect("resolve");
            assert_eq!(
                d.kernel,
                format!("mxfp4_gemm_decode_bk128_ks2_tm{tm}_wperm1_nt1"),
                "nblk 64 at M={m}: the measured optimum is ks1, and the rule deliberately is not"
            );
            assert_eq!(d.grid, [64, 1, 2]);
        }
        // The cells the measurement says the shipped target gets RIGHT, either side of it.
        assert_eq!(r.split_k(1, 20), 4, "nblk 1: ks1 is 2.2x slower");
        assert_eq!(r.split_k(40, 20), 4, "nblk 40 at M<=24: ks1 is +15%");
        assert_eq!(r.split_k(40, 40), 2);
        assert_eq!(r.split_k(56, 20), 2, "nblk 56: ks2 wins 3 of 4 M");
        assert_eq!(r.split_k(136, 20), 1, "nblk 136 already clears the target");

        // The finding, as an edit: the ks=1 step wants the CU count. One number cannot say that --
        // lowering the target to 64 also drags the ks=2 step down, and nblk 40 at M <= 24 moves
        // off the ks4 the measurement chose. Expressing it would need a per-ladder-step target,
        // which is the schema change this branch does not make.
        let mut lowered = r.clone();
        for row in &mut lowered.split_k_fill {
            row.fill = 64;
        }
        assert_eq!(lowered.split_k(64, 20), 1, "the cell the measurement wants");
        assert_eq!(
            lowered.split_k(40, 20),
            2,
            "... and the collateral: nblk 40 leaves ks4 as well"
        );
    }

    /// The negative control for the row above: the boundary is bounded on BOTH sides by
    /// measurement, so a target that moved it one model width further would be wrong.
    ///
    /// At nblk 56 the measurement says ks2 (40.36 vs 41.34 at M=5, 41.56 vs 41.92 at M=10, 47.76
    /// vs 49.38 at M=40 -- two of those outside the floor). A target of 64 leaves that cell alone;
    /// a target of 56 does not, and would take the slower kernel. If the axis were M, or the
    /// boundary anywhere below 57, this assertion would fail.
    #[test]
    fn the_ks1_boundary_is_bounded_on_both_sides_and_56_is_the_wrong_side() {
        let mut at64 = rules();
        let mut at56 = rules();
        for row in &mut at64.split_k_fill {
            row.fill = 64;
        }
        for row in &mut at56.split_k_fill {
            row.fill = 56;
        }
        for m in [5u32, 10, 20, 40] {
            assert_eq!(at64.split_k(56, m), 2, "nblk 56 keeps ks2 at M={m}");
            assert_eq!(at64.split_k(64, m), 1, "nblk 64 takes ks1 at M={m}");
            // The wrong side: nblk 56 would take the kernel the measurement rejects.
            assert_eq!(at56.split_k(56, m), 1, "the control must actually move it");
        }
    }

    /// The vocabulary GEMM takes the DECODE tile at a decode M. It is the one shape in the model
    /// wider than the split-K scratch (`dec_max_n` 36864), and that cap guards the split arm only.
    ///
    /// Measured paired against the reference's own launch at the TP=2 shard (see the band test's
    /// table): 0.35-0.36x at every decode M, bit-identical, and 511.6 µs at M=1 is 338 MB of
    /// weights at 635 GB/s -- the DRAM roofline, where the folded tile was at 38% of it.
    #[test]
    fn the_vocabulary_gemm_takes_the_decode_tile_at_a_decode_m() {
        let r = rules();
        // TP=2 (124160) and TP=1 (248320). nblk 970 and 1940, both far past the fill target, so
        // the split is 1 and the wide-N band answers BK=64.
        for n in [124_160u32, 248_320] {
            let nblk = n / 128;
            // BK is the reference's own `decode_bk64` again since 2026-09-20 (BK=64 at DTM=4
            // with no split, BK=128 otherwise): cold, the two are a tie at this width -- -1.3% at
            // M=20, -0.7% at M=40, inside the floor -- so nothing overrides it here.
            for (m, tm, bk) in [
                (1u32, 1u32, 128),
                (5, 1, 128),
                (20, 2, 128),
                (40, 3, 128),
                (64, 4, 64),
            ] {
                let d = resolve(Op::gemm(), Shape::new(m, n, 5120), &r).expect("resolve");
                assert_eq!(
                    d.kernel,
                    format!("mxfp4_gemm_decode_bk{bk}_ks1_tm{tm}_wperm1_nt1"),
                    "N={n} M={m}"
                );
                assert_eq!(
                    d.grid,
                    [nblk, 1, 1],
                    "N={n} M={m}: one block per n-tile, no split"
                );
                // The direct arm binds no workspace at all: there is nothing for the cap to guard.
                assert!(d.workspaces.is_empty(), "N={n} M={m}");
            }
            // The control that is NOT a deviation: above `decode_max_m` the folded tile is right
            // and both sides agree, so the change is bounded above by M, not just by N.
            let d = resolve(Op::gemm(), Shape::new(128, n, 5120), &r).expect("resolve");
            assert_eq!(
                d.kernel, "mxfp4_gemm_folded_tn2_wperm1_epifast1",
                "N={n} M=128"
            );
        }
    }

    /// The negative control: with the cap ANDed in unconditionally -- the reference's own test, and
    /// Saber's until 2026-09-19 -- the vocabulary GEMM goes back to the prefill folded tile, which
    /// is the 2.8x slower launch. A rule whose removal changed nothing would prove nothing.
    ///
    /// It also shows the cap still binds where it means something: a shape that needs a SPLIT and
    /// is wider than the scratch is refused the decode tile exactly as before.
    #[test]
    fn without_the_direct_exemption_the_vocabulary_gemm_falls_back_to_the_prefill_tile() {
        let r = rules();
        let capped = |m: u32, n: u32, k: u32| -> String {
            // The pre-change condition, applied on top of the shipped rules: N over the scratch
            // width is out of the band whatever the split is.
            if n > r.knobs.dec_max_n {
                return "mxfp4_gemm_folded_tn2_wperm1_epifast1".to_owned();
            }
            resolve(Op::gemm(), Shape::new(m, n, k), &r).unwrap().kernel
        };
        for n in [124_160u32, 248_320] {
            for m in [1u32, 5, 20, 40, 64] {
                let shipped = resolve(Op::gemm(), Shape::new(m, n, 5120), &r)
                    .unwrap()
                    .kernel;
                assert_ne!(
                    shipped,
                    capped(m, n, 5120),
                    "N={n} M={m}: the exemption must be what moves this cell"
                );
            }
        }
        // And the cap's real job, untouched: at a split the scratch is what bounds N. nblk 40 at
        // M=5 answers ks=4, so 4 * 5 * N floats must fit the declared capacity.
        assert_eq!(r.split_k(40, 5), 4);
        let partial = |m: u64, n: u64| 4 * m * n * 4;
        assert!(partial(5, 5120) <= r.workspace.decode_partial.capacity_bytes);
        // At the vocabulary width a 4-way split outgrows the declared slab at about M=10 (4 x M x
        // 248320 x 4 B against 37,748,736), which is the half of the cap that still binds -- the
        // fill rule never asks for a split that wide, and if it ever did, this is what would stop
        // it. `scratch_ok` is sized by the CHOSEN ks, which is the reference's own wording.
        assert!(partial(9, 248_320) <= r.workspace.decode_partial.capacity_bytes);
        assert!(partial(10, 248_320) > r.workspace.decode_partial.capacity_bytes);
    }

    #[test]
    fn prefill_takes_the_a_tiled_kernel_above_the_threshold() {
        let r = rules();
        let d = resolve(Op::gemm(), Shape::new(2048, 17408, 5120), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp");
        assert_eq!(d.grid, [136, 4, 1]);
        assert_eq!(d.block, [512, 1, 1]);
        let d = resolve(Op::gemm(), Shape::new(8192, 17408, 5120), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp");
        assert_eq!(d.grid, [136, 16, 1]);
        assert_eq!(d.block, [512, 1, 1]);
        let d = resolve(Op::gemm(), Shape::new(1024, 17408, 5120), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_atiled_tn2_lbk128_wperm1");
    }

    /// G3: the tile's bodies are one summation with the reference, and the row's `variant` is what
    /// the descriptor binds.
    /// G5: the mid-M folded heights bind ONLY under the knob, per band and N, each in the
    /// reference folded tile's summation class; the committed table (knob off) is the control --
    /// every rung keeps the 256-row tile, so a plan diff under the knob is exactly these symbols.
    #[test]
    fn the_mid_m_folded_heights_bind_under_the_knob_only() {
        let base = rules();
        assert!(
            !base.knobs.folded_mt,
            "the committed table keeps the reference tile"
        );
        let mut on = base.clone();
        on.knobs.folded_mt = true;
        let k = |r: &LaunchRules, m: u32, n: u32, kk: u32| {
            resolve(Op::gemm(), Shape::new(m, n, kk), r).expect("resolves")
        };
        let reference = "mxfp4_gemm_folded_tn2_wperm1_epifast1";
        let shapes = [
            (17408, 5120),
            (5120, 8704),
            (8192, 5120),
            (7168, 5120),
            (5120, 3072),
            (48, 5120),
        ];
        for m in [129u32, 192, 256, 257, 300, 320, 321, 384, 512] {
            for (n, kk) in shapes {
                assert_eq!(
                    k(&base, m, n, kk).kernel,
                    reference,
                    "knob off, M={m} n={n}"
                );
                let d = k(&on, m, n, kk);
                assert_eq!(
                    summation_class(&d.kernel),
                    reference,
                    "M={m} n={n}: {}",
                    d.kernel
                );
                let want = match (m, n) {
                    (129..=192, _) | (321..=384, _) => Some((3, 4)),
                    (257..=320, 8192..) => Some((5, 4)),
                    (257..=320, 128..) => Some((5, 2)),
                    _ => None,
                };
                match want {
                    Some((tm, wm)) => {
                        assert_eq!(d.plugin, FOLDED_MT_PLUGIN);
                        assert_eq!(
                            d.kernel,
                            format!("mxfp4_gemm_folded_mt_tn2_wperm1_t{tm}w{wm}")
                        );
                        assert_eq!(d.grid, [n.div_ceil(64), m.div_ceil(tm * wm * 16), 1]);
                        assert_eq!(d.block, [wm * 64, 1, 1]);
                        assert_eq!(d.kernarg_bytes, k(&base, m, n, kk).kernarg_bytes);
                    }
                    None => assert_eq!(d.kernel, reference, "M={m} n={n} keeps the reference"),
                }
            }
        }
        // The decode band and the A-tiled family are untouched by the knob.
        for (m, n) in [(64u32, 17408u32), (96, 5120), (2048, 17408), (513, 5120)] {
            assert_eq!(
                k(&on, m, n, 5120).kernel,
                k(&base, m, n, 5120).kernel,
                "M={m}"
            );
        }
        // Negative control: a symbol outside the family is not folded into the class.
        assert_ne!(
            summation_class("mxfp4_gemm_folded_tn4_wperm1_epifast1"),
            reference
        );
    }

    #[test]
    fn the_g3_tile_bodies_are_one_summation_with_the_reference() {
        let base = "mxfp4_gemm_atiled_tn4_lbk64_wperm1";
        for tail in ["", "_wm8", "_wm8s", "_wm8sp"] {
            assert_eq!(summation_class(&format!("{base}{tail}")), base, "{tail}");
        }
        // negative control: a different kernel is not folded into the class
        assert_ne!(summation_class("mxfp4_gemm_atiled_tn2_lbk128_wperm1"), base);
        // and the row's body is what the descriptor binds: the same height without `variant`
        // is G1's body, with the same grid and block.
        let r = rules();
        let d = resolve(Op::gemm(), Shape::new(2048, 17408, 5120), &r).unwrap();
        let mut g1 = r.clone();
        for row in &mut g1.atiled_wm {
            row.variant = None;
        }
        let e = resolve(Op::gemm(), Shape::new(2048, 17408, 5120), &g1).unwrap();
        assert_eq!(e.kernel, format!("{base}_wm8"));
        assert_ne!(d.kernel, e.kernel);
        assert_eq!((d.grid, d.block), (e.grid, e.block));
    }

    /// G1/G2: the 512-row A-tiled tile binds exactly where it was measured -- TN=4, the
    /// fragment-order layout, M in [2048, 8192] (G3's `_wm8sp` body on the whole envelope since G4) (the 2048- and 8192-row chunks and anything a
    /// bucket list could put between) -- and nowhere else; and the row is what binds it (the
    /// control: with the table's `[[atiled_wm]]` rows removed, M = 2048 is the reference's tile).
    #[test]
    fn the_tall_atiled_tile_binds_on_the_2048_to_8192_envelope() {
        let r = rules();
        let at =
            |m: u32, r: &LaunchRules| resolve(Op::gemm(), Shape::new(m, 5120, 8704), r).unwrap();
        assert_eq!(
            at(2048, &r).kernel,
            "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp"
        );
        for m in [2049, 4096, 8191, 8192] {
            assert_eq!(
                at(m, &r).kernel,
                "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp",
                "M={m} is inside G2's envelope, on G3's body since G4"
            );
        }
        // The control: without the second row's `variant`, 2049..8192 goes back to G1's body.
        let mut g2 = r.clone();
        for row in &mut g2.atiled_wm {
            if row.m_min > Some(2048) {
                row.variant = None;
            }
        }
        assert_eq!(
            at(4096, &g2).kernel,
            "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8"
        );
        assert_eq!(
            at(2048, &g2).kernel,
            "mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp"
        );
        for m in [2048, 2049, 4096, 8191, 8192] {
            assert_eq!(at(m, &r).grid, [40, m.div_ceil(512), 1]);
        }
        for m in [1024, 2047, 8193, 16384] {
            assert!(
                !at(m, &r).kernel.contains("_wm8"),
                "M={m} must keep the reference tile"
            );
        }
        let mut none = r.clone();
        none.atiled_wm.clear();
        assert_eq!(at(2048, &none).kernel, "mxfp4_gemm_atiled_tn4_lbk64_wperm1");
        assert_eq!(at(2048, &none).grid, [40, 8, 1]);
        // The checkpoint layout has no tall symbol, so the row cannot bind it.
        let wp0 = LaunchRules::from_toml(&PROD_RULES.replace("wperm = true", "wperm = false"))
            .expect("wperm0 rules");
        assert_eq!(at(2048, &wp0).kernel, "mxfp4_gemm_atiled_tn4_lbk64_wperm0");
    }

    /// G2: `gdn_norm_quant` writes the fragment-tiled layout at the band edge exactly when the
    /// knob says so, and the producer's own answer (`producer_writes_tiled`) and the symbol it
    /// resolves agree at every M. The control is the knob off: the reference's one row-major
    /// instantiation at every M, so a producer that switched on anything else would show here.
    #[test]
    fn the_gdn_norm_writes_the_tiled_layout_only_under_its_knob() {
        let r = rules();
        assert!(
            r.knobs.gdn_norm_tiled,
            "the committed table ships the tiled producer"
        );
        let op = Op::GdnNormQuant {
            zs: 3072,
            eps: 1e-6,
        };
        let edge = a_tiled_min_m();
        for (m, want) in [
            (8, "gdn_norm_quant_t256"),
            (edge - 1, "gdn_norm_quant_t256"),
            (edge, "gdn_norm_quant_t256_tiled1"),
            (2048, "gdn_norm_quant_t256_tiled1"),
            (8192, "gdn_norm_quant_t256_tiled1"),
        ] {
            let d = resolve(op, Shape::new(m, 3072, 0), &r).expect("resolve");
            assert_eq!(d.kernel, want, "M={m}");
            assert_eq!(
                producer_writes_tiled(op, m, &r),
                Some(want.ends_with("_tiled1")),
                "M={m}: the producer's answer and its symbol disagree"
            );
            assert_eq!((d.grid, d.block), ([m, 1, 1], [256, 1, 1]));
        }
        let mut off = r.clone();
        off.knobs.gdn_norm_tiled = false;
        for m in [8, edge, 2048, 8192] {
            let d = resolve(op, Shape::new(m, 3072, 0), &off).expect("resolve");
            assert_eq!(d.kernel, "gdn_norm_quant_t256", "knob off, M={m}");
            assert_eq!(producer_writes_tiled(op, m, &off), Some(false));
        }
    }

    #[test]
    fn between_the_bands_is_the_folded_tile() {
        let r = rules();
        // M=128 is above decode_max_m=64 and below a_tiled_min_m=513.
        let d = resolve(Op::gemm(), Shape::new(128, 5120, 5120), &r).expect("resolve");
        assert_eq!(d.kernel, "mxfp4_gemm_folded_tn2_wperm1_epifast1");
        assert_eq!(d.grid, [80, 1, 1]);
    }

    #[test]
    fn k_alignment_is_refused_not_rounded() {
        let r = rules();
        assert!(resolve(Op::gemm(), Shape::new(8, 5120, 5100), &r).is_err());
        // tiled A needs K % 128
        assert!(resolve(Op::gemm(), Shape::new(2048, 5120, 5120 + 64), &r).is_err());
    }

    #[test]
    fn producers_switch_to_the_tiled_layout_at_the_same_threshold() {
        let r = rules();
        let d = resolve(Op::AddRmsQuant { eps: 1e-6 }, Shape::new(512, 0, 5120), &r).unwrap();
        assert_eq!(d.kernel, "add_rms_quant_t256_tiled0");
        let d = resolve(Op::AddRmsQuant { eps: 1e-6 }, Shape::new(513, 0, 5120), &r).unwrap();
        assert_eq!(d.kernel, "add_rms_quant_t256_tiled1");
        let d = resolve(Op::AddRmsQuant { eps: 1e-6 }, Shape::new(2048, 0, 5120), &r).unwrap();
        assert_eq!(d.kernel, "add_rms_quant_t512_tiled1");
        assert_eq!(d.grid, [2048, 1, 1]);
    }

    #[test]
    fn silu_takes_the_wide_block_when_n_exceeds_the_register_budget() {
        let r = rules();
        let d = resolve(Op::SiluMulQuant, Shape::new(8, 8704, 0), &r).unwrap();
        assert_eq!(d.kernel, "silu_mul_quant_t256_tiled0");
        // TP=1's intermediate is 17408 > 10240: the 512-thread instantiation at any M.
        let d = resolve(Op::SiluMulQuant, Shape::new(8, 17408, 0), &r).unwrap();
        assert_eq!(d.kernel, "silu_mul_quant_t512_tiled0");
    }

    #[test]
    fn gdn_norm_quant_rejects_a_bad_width_and_stride() {
        let r = rules();
        assert!(resolve(
            Op::GdnNormQuant {
                zs: 3072,
                eps: 1e-6
            },
            Shape::new(8, 3000, 0),
            &r
        )
        .is_err());
        assert!(resolve(
            Op::GdnNormQuant { zs: 7, eps: 1e-6 },
            Shape::new(8, 3072, 0),
            &r
        )
        .is_err());
        let d = resolve(
            Op::GdnNormQuant {
                zs: 3072,
                eps: 1e-6,
            },
            Shape::new(8, 3072, 0),
            &r,
        )
        .unwrap();
        assert_eq!(d.kernel, "gdn_norm_quant_t256");
    }

    #[test]
    fn partition_boundaries_must_be_n_tile_aligned() {
        let r = rules();
        let p = Partitions {
            pb1: 100,
            pb2: 1 << 30,
            astride: 0,
        };
        let op = Op::Gemm {
            has_wref: true,
            partitions: p,
        };
        assert!(resolve(op, Shape::new(8, 5120, 5120), &r).is_err());
        let p = Partitions {
            pb1: 1024,
            pb2: 2048,
            astride: 5120 * 8,
        };
        let op = Op::Gemm {
            has_wref: true,
            partitions: p,
        };
        let d = resolve(op, Shape::new(8, 5120, 5120), &r).expect("aligned partitions");
        assert_eq!(
            d.arg("pb1").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 1024 })
        );
        assert_eq!(
            d.arg("astride").map(|s| s.value.clone()),
            Some(ArgValue::I64 { ival: 40960 })
        );
    }

    /// The DIRECT=false control: at the one cell the reference compiles it for, and refused
    /// everywhere else. A control that silently routed somewhere would prove nothing.
    ///
    /// The wide-N BK=64 row took this cell away for a day (2026-09-19) and the test recorded that
    /// it had; withdrawing the row on 2026-09-20 gives it back, and the control is a control
    /// again rather than a refusal.
    #[test]
    fn the_fused_silu_route_takes_the_same_cell_the_plain_decode_kernel_would() {
        // The fusion may not move the tiling: it is the decode band's own kernel with a different
        // A staging, and a cell the band would not have picked is a shape whose fused cost has
        // never been measured. down_proj at TP=2 is N=5120, K=8704.
        let r = rules();
        for m in [1u32, 5, 10, 20, 40] {
            let plain = resolve(Op::gemm(), Shape::new(m, 5120, 8704), &r).expect("plain");
            let fused = gemm_decode_fused_silu(Shape::new(m, 5120, 8704), &r).expect("fused");
            let cell = |k: &str| {
                k.split('_')
                    .filter(|p| p.starts_with("bk") || p.starts_with("ks") || p.starts_with("tm"))
                    .collect::<Vec<_>>()
                    .join("_")
            };
            assert_eq!(cell(&plain.kernel), cell(&fused.kernel), "M={m}");
            assert_eq!(plain.grid, fused.grid, "M={m}");
            // One launch where the plain route spends two -- that is the whole claim.
            assert_eq!(fused.n_launches, 1);
        }
    }

    #[test]
    fn a_shape_the_fused_family_was_not_compiled_for_is_refused() {
        let r = rules();
        // Past the decode band entirely.
        let e = gemm_decode_fused_silu(Shape::new(128, 5120, 8704), &r).unwrap_err();
        assert!(e.to_string().contains("past the decode band"), "{e}");
        // Inside the band but a cell with no fused instantiation: gate_up, not down_proj.
        let e = gemm_decode_fused_silu(Shape::new(5, 17408, 5120), &r).unwrap_err();
        assert!(
            e.to_string()
                .contains("no fused instantiation covers that cell"),
            "{e}"
        );
    }

    #[test]
    fn the_direct_false_decode_control_resolves_one_cell_and_refuses_the_rest() {
        let r = rules();
        // gate_up at M=8: nblk 136 clears the fill target, so DKS=1 and tm = ceil(8/16) = 1.
        let d = gemm_decode_indirect(Shape::new(8, 17408, 5120), &r).expect("resolve");
        assert_eq!(
            d.kernel,
            "mxfp4_gemm_decode_bk128_ks1_tm1_wperm1_nt1_indirect"
        );
        assert_eq!(d.grid, [136, 1, 1]);
        assert_eq!(d.n_launches, 1);
        // The whole point: the slab and the counter the DIRECT arm never touches ARE bound here.
        let names: Vec<_> = d.workspaces.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["decode_partial", "decode_counter"]);
        assert!(matches!(
            d.arg("partial").map(|s| s.value.clone()),
            Some(ArgValue::Buffer { .. })
        ));
        // Anywhere else there is no such instantiation.
        for (m, n, k) in [
            (8u32, 5120u32, 6144u32),
            (64, 17408, 5120),
            (128, 10240, 5120),
        ] {
            let e = gemm_decode_indirect(Shape::new(m, n, k), &r).unwrap_err();
            assert!(
                format!("{e}").contains("DIRECT control"),
                "M={m} N={n} K={k}: {e}"
            );
        }
    }

    /// The WPERM=0 weight layout is a rules-level decision, not a per-call one: the loader packs
    /// the tensor one way and the kernels read the flag, so a mismatch reads the weight as
    /// garbage rather than erroring. Every band has to follow the same flag.
    #[test]
    fn the_weight_layout_knob_moves_every_band_together() {
        let wp0 = PROD_RULES.replace("wperm = true", "wperm = false");
        let r = LaunchRules::from_toml(&wp0).expect("wperm0 rules");
        // NT is honoured only under WPERM, so the row-layout decode arm is nt0 as well.
        assert_eq!(
            resolve(Op::gemm(), Shape::new(5, 17408, 5120), &r)
                .unwrap()
                .kernel,
            "mxfp4_gemm_decode_bk128_ks1_tm1_wperm0_nt0"
        );
        assert_eq!(
            resolve(Op::gemm(), Shape::new(128, 5120, 5120), &r)
                .unwrap()
                .kernel,
            "mxfp4_gemm_folded_tn2_wperm0_epifast1"
        );
        assert_eq!(
            resolve(Op::gemm(), Shape::new(2048, 17408, 5120), &r)
                .unwrap()
                .kernel,
            "mxfp4_gemm_atiled_tn4_lbk64_wperm0"
        );
        // ... and the DIRECT control has no wperm0 instantiation at all.
        assert!(gemm_decode_indirect(Shape::new(8, 17408, 5120), &r).is_err());
    }

    /// The merged-linear partition select at the boundaries prod actually carries.
    #[test]
    fn merged_linear_partitions_bind_the_boundaries_and_the_stride() {
        let r = rules();
        // in_proj_qkvz at TP=2: mixed_qkv 5120 | z 3072, two distinct rotations.
        let p = Partitions {
            pb1: 5120,
            pb2: 1 << 30,
            astride: 8 * 5120,
        };
        let d = resolve(
            Op::Gemm {
                has_wref: true,
                partitions: p,
            },
            Shape::new(8, 8192, 5120),
            &r,
        )
        .expect("two partitions");
        assert_eq!(
            d.arg("pb1").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 5120 })
        );
        assert_eq!(
            d.arg("astride").map(|s| s.value.clone()),
            Some(ArgValue::I64 { ival: 40960 })
        );
        // Merged attention QKV with its output gate: q 3072 | kv 1024 | gate 3072.
        let p = Partitions {
            pb1: 3072,
            pb2: 4096,
            astride: 8 * 5120,
        };
        assert!(resolve(
            Op::Gemm {
                has_wref: true,
                partitions: p
            },
            Shape::new(8, 7168, 5120),
            &r
        )
        .is_ok());
        // NEGATIVE CONTROL: a boundary the loader would never emit. The reference launcher
        // refuses it too ("partition boundaries must be multiples of 128") -- at LAUNCH time,
        // from inside a pybind call. Refusing it here, at plan time, and naming WHICH boundary
        // is the same decision moved somewhere a caller can act on.
        let p = Partitions {
            pb1: 3000,
            pb2: 1 << 30,
            astride: 8 * 5120,
        };
        let e = resolve(
            Op::Gemm {
                has_wref: true,
                partitions: p,
            },
            Shape::new(8, 8192, 5120),
            &r,
        )
        .unwrap_err();
        assert!(
            format!("{e}").contains("pb1=3000 must be a multiple of 128"),
            "{e}"
        );
    }

    #[test]
    fn every_descriptor_arg_lands_where_the_abi_packs_it() {
        let r = rules();
        let d = resolve(Op::gemm(), Shape::new(5, 5120, 6144), &r).unwrap();
        assert_eq!(d.arg("m").map(|s| s.offset), Some(64));
        assert_eq!(d.arg("astride").map(|s| s.offset), Some(88));
        assert_eq!(d.kernarg_bytes, 96);
    }

    // ---- fusion-map row 3 (ws-h38) ----------------------------------------------------------

    fn nq_at(m: u32, k: u32) -> LaunchDescriptor {
        crate::saber::resolve(
            crate::saber::Op::AllReduceExactNq,
            crate::saber::Shape::ArNq(crate::saber::ArNqShape {
                m_rows: m,
                k_cols: k,
                eps_bits: 1e-6f32.to_bits(),
                world_size: 2,
            }),
            &crate::saber::SaberRules::prod_saber().expect("saber rules"),
        )
        .expect("the exact `_nq` epilogue resolves")
    }

    /// The row-parallel GEMMs of one TP=2 layer at `b1s8` (M = 8, N = hidden = 5120): `down_proj`
    /// (K = 8704), the GDN `out_proj` (K = 3072) and attention's `o_proj` (K = 3072). Each takes a
    /// compiled split cell, and the fused descriptor is the GEMM's image, then the epilogue's.
    #[test]
    fn row3_fuses_every_row_parallel_gemm_at_b1s8() {
        let r = rules();
        let nq = nq_at(8, 5120);
        for k in [8704u32, 3072] {
            let shape = Shape::new(8, 5120, k);
            let g = resolve(Op::gemm(), shape, &r).expect("gemm");
            let d = gemm_decode_ar_nq((&g, shape), &nq, &r).expect("row 3 fuses");
            assert_eq!(d.op, "gemm_decode_ar_nq");
            assert_eq!(d.plugin, "mxfp4_gemm_decode_arnq");
            assert_eq!(
                d.kernel,
                g.kernel
                    .replace("mxfp4_gemm_decode_", "mxfp4_gemm_decode_arnq_"),
                "K = {k}"
            );
            assert_eq!(d.grid, g.grid, "one dispatch: the GEMM's own grid");
            assert_eq!(d.block, [256, 1, 1]);
            assert_eq!(d.n_launches, 1);
            assert_eq!(d.kernarg_bytes, 208);
            // The GEMM's fourteen slots, byte for byte.
            assert_eq!(&d.args[..14], &g.args[..]);
            let int = |n: &str| match d.arg(n).map(|s| &s.value) {
                Some(ArgValue::I32 { ival }) => *ival,
                other => panic!("{n}: {other:?}"),
            };
            assert_eq!((int("k16"), int("ep_rows"), int("m")), (640, 8, 8));
            assert_eq!(int("slot_stride16"), 88_080_384 / 16);
            assert_eq!(int("drain"), 3);
            assert_eq!(
                d.arg("eps").map(|s| &s.value),
                nq.arg("eps").map(|s| &s.value)
            );
            let names: Vec<(&str, WorkspaceKind)> = d
                .workspaces
                .iter()
                .map(|w| (w.name.as_str(), w.kind))
                .collect();
            assert_eq!(
                names,
                [
                    ("decode_partial", WorkspaceKind::Scratch),
                    ("decode_counter", WorkspaceKind::Persistent),
                    ("ar_scratch", WorkspaceKind::IpcPeerVisible),
                    ("ar_peer_scratch", WorkspaceKind::IpcPeerVisible),
                    ("ar_flags", WorkspaceKind::IpcPeerVisible),
                    ("ar_peer_flags", WorkspaceKind::IpcPeerVisible),
                    ("ar_seq", WorkspaceKind::IpcPeerVisible),
                    (GEMM_AR_NQ_TILES, WorkspaceKind::Persistent),
                ]
            );
            let tiles = d.workspaces.last().unwrap();
            assert_eq!((tiles.bytes, tiles.zero_once), (8, true));
            // Every buffer slot is either a caller binding or a declared workspace.
            for s in &d.args {
                if let ArgValue::Buffer { buffer } = &s.value {
                    let bound = [
                        "a",
                        "w",
                        "ws",
                        "wref",
                        "as",
                        "c",
                        "residual",
                        "norm_w",
                        "q_out",
                        "scale_out",
                        "res_out",
                    ]
                    .contains(&buffer.as_str());
                    assert!(
                        bound || d.workspaces.iter().any(|w| &w.name == buffer),
                        "{buffer}"
                    );
                }
            }
        }
    }

    /// Row 3 refuses everything it was not compiled for, by name -- each of these would be a
    /// wrong answer or a hang, not a slow path.
    #[test]
    fn row3_refuses_what_it_was_not_compiled_for() {
        let r = rules();
        let nq8 = nq_at(8, 5120);
        let msg = |g: &LaunchDescriptor, s: Shape, nq: &LaunchDescriptor| {
            format!("{}", gemm_decode_ar_nq((g, s), nq, &r).unwrap_err())
        };
        // No split: `gate_up` at M = 8 is ks1, so there is no last-arriving block.
        let s = Shape::new(8, 17408, 5120);
        let g = resolve(Op::gemm(), s, &r).unwrap();
        assert!(g.kernel.contains("_ks1_"), "{}", g.kernel);
        assert!(msg(&g, s, &nq8).contains("not a"), "ks1");
        // The epilogue at a different row count than the GEMM writes.
        let s = Shape::new(8, 5120, 8704);
        let g = resolve(Op::gemm(), s, &r).unwrap();
        assert!(msg(&g, s, &nq_at(5, 5120)).contains("the epilogue is 5 x 5120"));
        // ... or a different width.
        assert!(msg(&g, s, &nq_at(8, 4096)).contains("the epilogue is 8 x 4096"));
        // A reduce that is not the `_nq` epilogue (the plain exact family).
        let plain = crate::saber::resolve(
            crate::saber::Op::AllReduce {
                dtype: crate::saber::ArPayload::Bf16,
            },
            crate::saber::Shape::Ar(crate::saber::ArShape {
                n_elem: 8 * 5120,
                elem_bytes: 2,
                world_size: 2,
            }),
            &crate::saber::SaberRules::prod_saber().unwrap(),
        )
        .unwrap();
        assert_eq!(plain.op, "allreduce_exact");
        assert!(msg(&g, s, &plain).contains("not the exact `_nq` epilogue"));
        // tm2 (M = 20): not compiled, and above the exact band anyway.
        let s = Shape::new(20, 5120, 8704);
        let g = resolve(Op::gemm(), s, &r).unwrap();
        assert!(g.kernel.contains("_tm2_"), "{}", g.kernel);
        assert!(msg(&g, s, &nq_at(20, 5120)).contains("not a compiled row-3 cell"));
        // More rows than n-tiles: the row owners are the last M arrivals.
        let s = Shape::new(12, 1024, 8704);
        let g = resolve(Op::gemm(), s, &r).unwrap();
        assert_eq!(g.kernel, "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1");
        assert!(msg(&g, s, &nq_at(12, 1024)).contains("row owners"));
        // Not a decode descriptor at all.
        let mut bogus = resolve(Op::gemm(), Shape::new(8, 5120, 8704), &r).unwrap();
        bogus.op = "gemm_folded".into();
        assert!(gemm_decode_ar_nq((&bogus, Shape::new(8, 5120, 8704)), &nq8, &r).is_err());
    }
}
