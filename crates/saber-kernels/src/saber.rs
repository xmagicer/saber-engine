//! The reference's launch logic as data — the second plugin.
//!
//! `ref/kernel-builds/b9e42ab-rx6` (the tree the running `qwen_prod_nvfp4` unit points
//! `REF_KERNELS_SO` at) decides, in C++, which instantiation of the paged attention, GDN and quantised
//! all-reduce kernels runs: `prefill_launch` / `decode_launch` / `decode_splits`
//! (`ref_attn_paged_h256_gqa6.hip:57-198`), the `ab_is_bf16` branch in
//! `ref_gdn_fused_update_w4k128v128.hip:286`, the `act` branch in
//! `ref_gdn_gated_rmsnorm_h128_bf16.hip:73-80` and the `dtype` branch in
//! `ref_ar_wht6_launch`. Here each of those is a row in `rules/saber_gfx1201.toml` and [`resolve`]
//! is a pure function over them.
//!
//! Two layers, because the reference has two, and the split is the thing worth copying (see
//! docs/ARCHITECTURE.md, "Kernel plane"):
//!
//! 1. [`Registry`] is `ref_registry.hip`'s `SABERConstraint` table. [`Registry::select`] and
//!    [`Registry::explain`] are `select_kernel()` and `explain()` from `ref_module.hip`,
//!    including their refusal wording. It answers "does a kernel EXIST for this geometry" — a
//!    NECESSARY condition. Strides, contiguity and buffer sizes are per-call and stay in
//!    [`resolve`], which still refuses anything it cannot launch.
//! 2. The instantiation tables turn the entry the registry named plus the plan's knobs into a
//!    SYMBOL. Nothing formats a kernel name: every symbol this module can emit is a table row.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::abi::{self, ArgKind, SigArg};
use crate::rules::{ArgValue, LaunchDescriptor, Slot, WorkspaceKind, WorkspaceReq};
use crate::{Error, Result};

// ---------------------------------------------------------------------------- rules (data)

#[derive(Debug, Clone, Deserialize)]
pub struct SaberRules {
    pub schema_version: u32,
    pub name: String,
    pub knobs: Knobs,
    pub decode_splits: SplitLaw,
    pub gdn: Gdn,
    pub ar: Ar,
    pub ar_nq: ArNq,
    pub registry: Vec<RegistryRow>,
    pub attn_prefill: Vec<PrefillRow>,
    pub attn_decode: Vec<DecodeRow>,
    pub attn_combine: Vec<CombineRow>,
    /// ws-h38: the decode attention whose last-arriving split block runs the combine
    /// ([`attn_decode_declc`]). Absent = the fusion is never offered.
    #[serde(default)]
    pub attn_decode_declc: Vec<DeclcRow>,
    pub gdn_fused_update: Vec<FusedRow>,
    pub gdn_conv_prep: Vec<ConvPrepRow>,
    pub gdn_conv_update: Vec<ConvUpdateRow>,
    pub gdn_recurrent_update: Vec<RecurrentRow>,
    pub gdn_kkt_solve: Vec<KktRow>,
    pub gdn_chunk_scan: Vec<ScanRow>,
    pub gdn_norm: Vec<NormRow>,
    pub ar_wht: Vec<ArRow>,
    pub ar_exact: Vec<ArExactRow>,
    pub ar_exact_nq: Vec<ArNqRow>,
    pub workspace: BTreeMap<String, WorkspaceRule>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Knobs {
    pub attn_head_dim: u32,
    pub attn_gqa: u32,
    pub attn_block_size: u32,
    pub attn_max_decode_rows: u32,
    pub prefill_nwarps: u32,
    pub prefill_block: u32,
    pub prefill_bq: u32,
    pub decode_tile: u32,
    pub decode_rows_per_warp: u32,
    pub decode_max_nw: u32,
    pub combine_block: u32,
    pub combine_partial_f16: bool,
    pub combine_partial_bytes_per_split: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SplitLaw {
    pub fill_target: u32,
    pub ceiling: u32,
    pub floor: u32,
    pub min_tiles_per_segment: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gdn {
    pub head_k: u32,
    pub head_v: u32,
    pub conv_width: u32,
    pub chunk: u32,
    pub fused_block: u32,
    pub fused_max_wg: u32,
    pub fused_max_items: u32,
    pub norm_channels: u32,
    pub norm_rows_per_block: u32,
    pub norm_block: u32,
    pub hpb_start: u32,
    pub hpb_min_workgroups: u64,
    pub conv_chunk: u32,
    pub vs_fill_workgroups: u32,
    pub vs_max: u32,
    pub recurrent_row_threads: u32,
    pub kkt_block: u32,
    pub scan_block: u32,
    pub scan_bv: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ar {
    /// Explicit runtime backend choice; native rule files keep their existing semantics.
    #[serde(skip)]
    pub external_rccl: bool,
    /// With a host-mapped backend, smaller messages retain native fused kernels.
    /// Zero keeps the all-RCCL compatibility profile's existing behavior.
    #[serde(skip)]
    pub external_rccl_min_bytes: u64,
    /// Exact kernels over registered shared host memory, with short bounded waits.
    #[serde(skip)]
    pub host_mapped: bool,
    pub group: u32,
    pub max_blocks: u32,
    pub chunk_groups_even_bits: u32,
    pub chunk_groups_odd_bits: u32,
    pub max_bytes: u64,
    /// The exact-vs-lossy threshold, and the width the policy op asks for. See the rules file for
    /// the two-step rule this reproduces.
    pub quant_min_bytes: u64,
    pub qbits: u32,
    pub min_nb: u32,
    pub max_nb: u32,
    pub max_nb_exact: u32,
    pub threads: u32,
    pub words_per_block: u64,
    pub drain: i64,
    pub acq: i64,
    /// **T4, a diagnostic**: resolve every all-reduce to its RACE-CHECK twin (`<symbol>_chk`, the
    /// served kernel with `hip/ar_race_check.h`'s payload / flag / counter checks inserted and
    /// nothing else changed). Never in a rules file: the engine sets it from
    /// `--engine-race-check` (`saber engine --race-check`), both ranks the same, and the
    /// collective digest (which carries the kernel name) refuses a pair that disagrees.
    #[serde(default)]
    pub race_check: bool,
    /// **CMP27S: the wave-parallel wht6 body** (`--engine-ar-wpar`, `saber_ar_wht_wpar`): a served
    /// wht6 bf16 one-shot whose chunks-per-block fit the LDS word buffer
    /// ([`AR_WPAR_MAX_CHUNKS_PER_BLOCK`]) launches [`AR_WPAR_SYMBOL`] at the SAME grid, block and
    /// kernargs -- the served bytes (payload, scales, pack, output; CMP27R 0 of 132 cases), so a
    /// wpar rank and a served rank interoperate. Never in a rules file: the engine sets it from
    /// the flag on both ranks.
    #[serde(default)]
    pub wpar: bool,
}

/// CMP27S: the wave-parallel twin of the served wht6 bf16 one-shot (`hip/saber_ar_wht_wpar.hip`).
pub const AR_WPAR_SYMBOL: &str = "saber_ar_wpar_2rank_wht6_bf16";
/// Runtime-provided collective implementation, not a HIP code object.
pub const RCCL_PLUGIN: &str = "saber_rccl";

/// Whether this exact message belongs to the external collective backend.
#[must_use]
pub fn uses_rccl(rules: &SaberRules, bytes: u64) -> bool {
    rules.ar.external_rccl && bytes >= rules.ar.external_rccl_min_bytes
}
/// External BF16 sum entry with [`abi::SIG_RCCL_BF16`].
pub const RCCL_BF16_KERNEL: &str = "nccl_allreduce_bf16";
/// The served symbol [`AR_WPAR_SYMBOL`] replaces.
pub const AR_WPAR_SOURCE: &str = "saber_ar_oneshot_2rank_wht6_bf16";
/// `SABER_WPAR_CPB_MAX`: the LDS word buffer's chunks a block (8 x 32 lanes x 12 words = 12 KiB).
/// A call whose `ceil(n_chunks / grid)` exceeds it keeps the served body (counted by
/// [`ar_wpar_route`]'s caller; the kernel traps if launched past it).
pub const AR_WPAR_MAX_CHUNKS_PER_BLOCK: u32 = 8;

/// CMP27S: the body one rotated reduce launches under [`Ar::wpar`] -- `Some(AR_WPAR_SYMBOL)` when
/// `symbol` is the served wht6 bf16 one-shot and its chunks a block fit the LDS buffer, `None`
/// (keep `symbol`) otherwise.
#[must_use]
pub fn ar_wpar_route(
    rules: &SaberRules,
    symbol: &str,
    n_chunks: u32,
    nb: u32,
) -> Option<&'static str> {
    (rules.ar.wpar
        && symbol == AR_WPAR_SOURCE
        && n_chunks.div_ceil(nb.max(1)) <= AR_WPAR_MAX_CHUNKS_PER_BLOCK)
        .then_some(AR_WPAR_SYMBOL)
}

/// CMP27AE: the first `ar_seq` / `ar_flags` word of the GEMM + wht6 fold's family
/// (`ARW_WORD0` in `hip/mxfp4_gemm_decode_arw.hip`): words `AR_FUSE_WORD0..+AR_FUSE_TILES`, one
/// counter and one flag a 64-column tile. The one-shots use words `0..max_nb` (96) and the split
/// form [`AR_SPLIT_FLAG_WORD`] (511), so the family never shares a word with another form.
pub const AR_FUSE_WORD0: u64 = 384;
/// CMP27AE: the fold's tiles a call -- N 5120 / 64 columns. EVERY call of the family launches all
/// of them, so each tile's counter is the family's call count and the slot is the CALL's parity
/// (FN114d's law; a per-block counter shared with the drafter's 29 / 14 / 7-block one-shots would
/// let two forms pick their slots by different parities).
pub const AR_FUSE_TILES: u64 = 80;
/// CMP27AE: the bytes a slot reserves for the fold's payload, just below the scales: 80 tiles x
/// (64 rows x 48 B of symbols + 64 x 2 B of scales) = 256,000 B, rounded up to 256 KiB.
pub const AR_FUSE_RESERVE_BYTES: u64 = 256 * 1024;

/// CMP27AE: where the fold's tiles start inside a scratch slot (`fuse_off`): the
/// [`AR_FUSE_RESERVE_BYTES`] below the scales (`scale_off = max_bytes / 2`). Every one-shot
/// payload is at most `3/8 x max_bytes` (6 bits of a bf16 message that fits the slot), so the
/// reserve is disjoint from every other form's bytes when `max_bytes / 8 >= the reserve`.
///
/// # Errors
/// A slot too small to hold the reserve between the widest one-shot payload and the scales.
pub fn ar_fuse_off(rules: &SaberRules) -> Result<u64> {
    let a = &rules.ar;
    let scale_off = a.max_bytes / 2;
    let widest_payload = a.max_bytes / 2 * 3 / 4; // 6 bits of every bf16 byte pair
    let off = scale_off.saturating_sub(AR_FUSE_RESERVE_BYTES);
    if off < widest_payload || off % 16 != 0 {
        return Err(Error::Rule(format!(
            "--engine-ar-fuse: a {} B slot has no room for the fold's {AR_FUSE_RESERVE_BYTES} B \
             between the widest one-shot payload ({widest_payload} B) and the scales at \
             {scale_off} B",
            a.max_bytes
        )));
    }
    Ok(off)
}

/// The served all-reduce symbols that have a race-check twin (T4). Any other symbol under
/// `race_check` is a refusal by name: a checked run that quietly ran an unchecked reduce would
/// report a clean count for a kernel it never looked at.
pub const AR_CHECKED_SYMBOLS: &[&str] = &[
    "saber_ar_oneshot_2rank_exact_bf16",
    "saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16",
    "saber_ar_oneshot_2rank_wht6_bf16",
    "saber_ar_oneshot_2rank_wht4_bf16",
    AR_WPAR_SYMBOL,
];

/// The twin's suffix.
pub const AR_CHECK_SUFFIX: &str = "_chk";

/// Bytes at the TAIL of each scratch slot that hold the sender's check words (`CHK_RESERVE_BYTES`
/// in `ar_race_check.h`: 512 blocks x 128 pieces x 4 B). A checked message must stay below it.
pub const AR_CHECK_RESERVE_BYTES: u64 = 512 * 128 * 4;

/// Where the checks' diagnostic words start in every `ar_seq` region (`CHK_DIAG` words = 2048 B;
/// the served kernels use the first `max_blocks` words), and how many there are.
pub const AR_CHECK_DIAG_OFFSET: u64 = 512 * 4;
/// The diagnostic words the host reads (`CHK_D_*`, 13 used).
pub const AR_CHECK_DIAG_WORDS: usize = 16;
/// The `ar_seq` region's size under the race check: the served counters, then the diagnostic
/// words. Without the check the rules file's size stands (2048 B), so nothing a served run
/// allocates or keys changes.
pub const AR_CHECK_SEQ_BYTES: u64 = 4096;

/// The kernel a reduce launches: `symbol`, or its race-check twin under [`Ar::race_check`], whose
/// highest byte in the slot (`top_bytes`) must stay below the check words.
///
/// # Errors
/// A symbol with no twin, or a message that would reach the check words.
pub fn ar_kernel(rules: &SaberRules, symbol: &str, top_bytes: u64) -> Result<String> {
    if !rules.ar.race_check {
        return Ok(symbol.to_owned());
    }
    if !AR_CHECKED_SYMBOLS.contains(&symbol) {
        return Err(Error::Rule(format!(
            "--engine-race-check: `{symbol}` has no race-check twin (the checked set is \
             {AR_CHECKED_SYMBOLS:?}); a checked run does not launch an unchecked reduce"
        )));
    }
    let room = rules.ar.max_bytes.saturating_sub(AR_CHECK_RESERVE_BYTES);
    if top_bytes > room {
        return Err(Error::Rule(format!(
            "--engine-race-check: `{symbol}` fills {top_bytes} B of a slot whose last \
             {AR_CHECK_RESERVE_BYTES} B hold the check words ({room} B usable)"
        )));
    }
    Ok(format!("{symbol}{AR_CHECK_SUFFIX}"))
}

/// CMP27S: the row counts `1..=max_rows` of a `rows x hidden` bf16 reduce that [`ar_wpar_route`]
/// sends to the wave-parallel body under `rules` -- `(lowest, highest, how many)`, `None` when
/// none does (the knob off, or no rotated call fits). The boot line and the tests read it.
#[must_use]
pub fn ar_wpar_row_band(rules: &SaberRules, hidden: u64, max_rows: u64) -> Option<(u64, u64, u64)> {
    let mut band: Option<(u64, u64, u64)> = None;
    for rows in 1..=max_rows {
        let d = resolve(
            Op::AllReduce {
                dtype: ArPayload::Bf16,
            },
            Shape::Ar(ArShape {
                n_elem: rows * hidden,
                elem_bytes: 2,
                world_size: 2,
            }),
            rules,
        );
        let wpar = d.is_ok_and(|d| {
            d.kernel.strip_suffix(AR_CHECK_SUFFIX).unwrap_or(&d.kernel) == AR_WPAR_SYMBOL
        });
        if wpar {
            band = Some(band.map_or((rows, rows, 1), |(lo, _, n)| (lo, rows, n + 1)));
        }
    }
    band
}

/// `SABER_ARNQ_T` / `SABER_ARNQ_MAXG`, the `_nq` epilogue's register budget.
#[derive(Debug, Clone, Deserialize)]
pub struct ArNq {
    pub block: u32,
    pub max_k: u32,
    pub k_multiple: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegistryRow {
    pub op: String,
    pub entry: String,
    pub constraints: Vec<Constraint>,
}

/// One predicate on one named parameter — `SABERConstraint` (`ref.h:216-221`).
#[derive(Debug, Clone, Deserialize)]
pub struct Constraint {
    pub key: String,
    pub op: ConstraintOp,
    #[serde(default)]
    pub ival: i64,
    #[serde(default)]
    pub sval: String,
}

/// `SABER_C_*`, spelled the way `constraint_list()` prints them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintOp {
    Eq,
    Le,
    Ge,
    DivisibleBy,
    In,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrefillRow {
    pub entry: String,
    pub fp8_mode: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecodeRow {
    pub entry: String,
    pub nw: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CombineRow {
    pub pf16: bool,
    /// The RAGGED combine (workstream C2): walks `cu_q` for a job's sequence. Absent = false = the
    /// uniform one, which divides by `q_len`.
    #[serde(default)]
    pub ragged: bool,
    pub symbol: String,
}

/// One `[[attn_decode_declc]]` row (ws-h38): the last-arriver decode at one warp count.
#[derive(Debug, Clone, Deserialize)]
pub struct DeclcRow {
    pub entry: String,
    pub nw: u32,
    /// The widest split count its merge holds (the per-wave weight row, in floats).
    pub max_splits: u32,
    pub symbol: String,
    /// The same kernel with the last segment dropped from the merge: the gate's control.
    pub control: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FusedRow {
    pub ab_bf16: bool,
    /// The STORED GDN state is f16 (workstream C1, `--gdn-state-dtype f16`): rx7's
    /// `SABER_ST_F16` instantiation. Absent = false = the fp32 state rx6's kernels read.
    #[serde(default)]
    pub state_f16: bool,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConvPrepRow {
    pub hpb: u32,
    pub ab_bf16: bool,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConvUpdateRow {
    pub hpb: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecurrentRow {
    pub ab_bf16: bool,
    /// The STORED GDN state is f16 (workstream C1, `--gdn-state-dtype f16`): rx7's
    /// `SABER_ST_F16` instantiation. Absent = false = the fp32 state rx6's kernels read.
    #[serde(default)]
    pub state_f16: bool,
    pub norm: bool,
    pub vs: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KktRow {
    pub ratio: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScanRow {
    /// The STORED GDN state is f16 (workstream C1, `--gdn-state-dtype f16`): rx7's
    /// `SABER_ST_F16` instantiation. Absent = false = the fp32 state rx6's kernels read.
    #[serde(default)]
    pub state_f16: bool,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NormRow {
    pub act: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArRow {
    pub entry: String,
    pub bits: u32,
    pub dtype: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArExactRow {
    pub dtype: String,
    /// Elements of the payload type packed into one int4; carried so the row states the geometry
    /// its symbol was compiled for rather than only naming it.
    pub lanes: u32,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArNqRow {
    pub dtype: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceRule {
    pub kind: WorkspaceKind,
    pub capacity_bytes: u64,
}

const PROD_RULES: &str = include_str!("../rules/saber_gfx1201.toml");

/// The same text, for the plan input hash: the policy this module enforces.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

/// **The RAGGED decode attention's rows (workstream C2)**, a table of their own
/// (`rules/saber_ragged_gfx1201.toml`), merged by [`SaberRules::prod_saber`]. Its own table so that the
/// per-table policy digest re-keys only the ragged route's records, never an rx6 route's.
pub const RAGGED_RULES_TEXT: &str = include_str!("../rules/saber_ragged_gfx1201.toml");

/// The rows `saber_ragged_gfx1201.toml` carries: registry entries under
/// [`ATTN_DECODE_RAGGED_OP`] and their instantiations.
#[derive(Debug, Clone, Deserialize)]
struct RaggedTable {
    schema_version: u32,
    #[allow(dead_code)]
    name: String,
    registry: Vec<RegistryRow>,
    attn_decode: Vec<DecodeRow>,
    attn_combine: Vec<CombineRow>,
}

/// The registry op a decode ROW BAND's attention selects (C2); a rectangle's is
/// `attn_decode_paged`. Two names, so neither geometry can ever select the other's entry.
pub const ATTN_DECODE_RAGGED_OP: &str = "attn_decode_paged_ragged";

/// Which `.hsaco` each symbol lives in. One line per plugin, not per symbol: the code objects are
/// named for the reference translation units they were sliced from.
///
/// Order matters, and it bit once: `saber_ar_oneshot_2rank_exact_bf16` also starts with
/// `saber_ar_oneshot_`, so the exact family has to be tested BEFORE the lossy one or a descriptor
/// names the right symbol in the wrong blob and the loader reports "named symbol not found".
/// `tests/saber_rules.rs::every_rules_symbol_lives_in_the_plugin_this_maps_it_to` is the control.
pub fn plugin_for(symbol: &str) -> Result<&'static str> {
    if symbol.starts_with("saber_attn_decode_ragged_")
        || symbol.starts_with("saber_attn_splitkv_combine_ragged_")
    {
        // The ragged pair (C2) BEFORE the uniform attention, whose prefix it shares.
        Ok("saber_attn_ragged")
    } else if symbol.starts_with("saber_attn_") {
        Ok("saber_attn")
    } else if symbol.starts_with("saber_gdn_fused_update_f16s_") {
        // The f16 stored-state plugins (C1) BEFORE their fp32 siblings, whose prefix they share.
        Ok("saber_gdn_fused_update_f16s")
    } else if symbol.starts_with("saber_gdn_recurrent_update_f16s_") {
        Ok("saber_gdn_recurrent_update_f16s")
    } else if symbol.starts_with("saber_gdn_chunk_scan_f16s_") {
        Ok("saber_gdn_chunk_scan_f16s")
    } else if symbol.starts_with("saber_gdn_fused_update_") {
        Ok("saber_gdn_fused_update")
    } else if symbol.starts_with("saber_gdn_conv_prep_")
        || symbol.starts_with("saber_gdn_conv_update_")
    {
        Ok("saber_gdn_conv")
    } else if symbol.starts_with("saber_gdn_recurrent_update_") {
        Ok("saber_gdn_recurrent_update")
    } else if symbol.starts_with("saber_gdn_kkt_solve_") {
        Ok("saber_gdn_kkt_solve")
    } else if symbol.starts_with("saber_gdn_chunk_scan_") {
        Ok("saber_gdn_chunk_scan")
    } else if symbol.starts_with("saber_gdn_gated_rmsnorm_") {
        Ok("saber_gdn_norm")
    } else if symbol.starts_with("saber_ar_oneshot_2rank_exact")
        && symbol.ends_with(AR_CHECK_SUFFIX)
    {
        // T4's race-check twins BEFORE the served objects whose prefix they share.
        Ok("saber_ar_exact_chk")
    } else if symbol.starts_with("saber_ar_wpar_") && symbol.ends_with(AR_CHECK_SUFFIX) {
        // CMP27S: the wave-parallel wht6 body's race-check twin.
        Ok("saber_ar_wht_wpar_chk")
    } else if symbol.starts_with("saber_ar_wpar_") {
        Ok("saber_ar_wht_wpar")
    } else if symbol.starts_with("saber_ar_oneshot_") && symbol.ends_with(AR_CHECK_SUFFIX) {
        Ok("saber_ar_wht_chk")
    } else if symbol.starts_with("saber_ar_split_2rank_wht") {
        // P27-1d: the split rotated all-reduce's own object.
        Ok("saber_ar_wht_split")
    } else if symbol.starts_with("saber_ar_oneshot_2rank_exact") {
        Ok("saber_ar_exact")
    } else if symbol.starts_with("saber_ar_oneshot_") {
        Ok("saber_ar_wht")
    } else {
        Err(Error::Rule(format!("no plugin holds symbol `{symbol}`")))
    }
}

impl SaberRules {
    /// The rules the `qwen_prod_nvfp4` unit runs today (reference rx6, TP=2, MAXSEQS=8), plus the
    /// RAGGED decode attention's own table (C2, [`RAGGED_RULES_TEXT`]).
    pub fn prod_saber() -> Result<Self> {
        Self::from_toml(PROD_RULES)?.with_ragged_toml(RAGGED_RULES_TEXT)
    }

    /// Merge a ragged-attention table (C2) into these rules: its registry rows (op
    /// [`ATTN_DECODE_RAGGED_OP`] only), its decode instantiations and its combines (`ragged =
    /// true` only). A row of another op or kind is refused, so the table cannot reach a route
    /// that is not the ragged one.
    ///
    /// # Errors
    /// A parse failure, another schema, or a row outside the ragged route.
    pub fn with_ragged_toml(mut self, text: &str) -> Result<Self> {
        let t: RaggedTable = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        if t.schema_version != self.schema_version {
            return Err(Error::Rule(format!(
                "ragged rules schema {} beside saber schema {}",
                t.schema_version, self.schema_version
            )));
        }
        if let Some(r) = t.registry.iter().find(|r| r.op != ATTN_DECODE_RAGGED_OP) {
            return Err(Error::Rule(format!(
                "the ragged table names op `{}`; it may carry `{ATTN_DECODE_RAGGED_OP}` only",
                r.op
            )));
        }
        if let Some(r) = t.attn_combine.iter().find(|r| !r.ragged) {
            return Err(Error::Rule(format!(
                "the ragged table carries a uniform combine `{}`",
                r.symbol
            )));
        }
        self.registry.extend(t.registry);
        self.attn_decode.extend(t.attn_decode);
        self.attn_combine.extend(t.attn_combine);
        self.check()?;
        Ok(self)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: SaberRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    /// Invariants the reference asserts in its own sources; a rules file that breaks one is refused here
    /// rather than mis-routing a shape at run time.
    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "saber rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        let k = &self.knobs;
        if k.attn_max_decode_rows != k.decode_rows_per_warp * k.decode_max_nw {
            return Err(Error::Rule(format!(
                "attn_max_decode_rows {} is not {} rows x {} warps: the decode kernel would be \
                 asked for a query tile it has no instantiation for",
                k.attn_max_decode_rows, k.decode_rows_per_warp, k.decode_max_nw
            )));
        }
        if k.prefill_bq * k.attn_gqa != k.prefill_nwarps * 16 {
            return Err(Error::Rule(format!(
                "prefill_bq {} x gqa {} must be nwarps {} x 16",
                k.prefill_bq, k.attn_gqa, k.prefill_nwarps
            )));
        }
        let s = &self.decode_splits;
        if !s.ceiling.is_power_of_two() || !s.floor.is_power_of_two() || s.floor > s.ceiling {
            return Err(Error::Rule(
                "the split law snaps to powers of two between floor and ceiling".into(),
            ));
        }
        if self.ar.max_bytes % 16 != 0 {
            return Err(Error::Rule(
                "ar.max_bytes must be 16 B aligned (the push is uint4)".into(),
            ));
        }
        for (name, w) in &self.workspace {
            if w.capacity_bytes == 0 {
                return Err(Error::Rule(format!("workspace `{name}` has no capacity")));
            }
        }
        if self.registry.is_empty() {
            return Err(Error::Rule("the registry is empty".into()));
        }
        Ok(())
    }

    fn workspace(&self, name: &str) -> Result<&WorkspaceRule> {
        self.workspace
            .get(name)
            .ok_or_else(|| Error::Rule(format!("no workspace rule `{name}`")))
    }

    /// `decode_splits(a)` (`ref_attn_paged_h256_gqa6.hip:126-140`), verbatim.
    pub fn decode_splits(&self, max_ctx: u32, kv_heads: u32, num_seqs: u32, want: u32) -> u32 {
        if want > 0 {
            return want;
        }
        let law = &self.decode_splits;
        let tile = self.knobs.decode_tile;
        let tiles = max_ctx.div_ceil(tile);
        let units = (kv_heads * num_seqs).max(1);
        let raw = law.fill_target / units;
        let mut s = law.ceiling;
        while s > law.floor && s > raw {
            s >>= 1;
        }
        let cap = tiles / law.min_tiles_per_segment;
        while s > law.floor && s > cap {
            s >>= 1;
        }
        if s > tiles {
            s = tiles;
        }
        s.max(1)
    }

    /// `saber_attn_decode_h256_gqa6_scratch_bytes(a)`.
    pub fn decode_scratch_bytes(&self, shape: &AttnShape, splits: u32) -> u64 {
        let rows = shape.num_seqs as u64 * shape.q_len as u64 * shape.q_heads as u64;
        rows * splits as u64 * self.knobs.combine_partial_bytes_per_split
    }

    /// The conv kernels' channel grouping (`ref_gdn_conv_w4_h128_bf16.hip:258-260, 425-431`).
    /// `n_seqs` is `Some` for the DECODE entry, which halves further until the grid covers the
    /// device, and `None` for the prefill one, which does not.
    pub fn conv_hpb(&self, h: u32, hg: u32, n_seqs: Option<u32>) -> u32 {
        let gd = &self.gdn;
        let units = 2 * hg + h;
        let mut hpb = gd.hpb_start;
        while hpb > 1 && (hg % hpb != 0 || h % hpb != 0) {
            hpb >>= 1;
        }
        if let Some(n) = n_seqs {
            while hpb > 1 && n as u64 * (units / hpb) as u64 > 0 && {
                let wg = n as u64 * (units / hpb) as u64;
                wg < gd.hpb_min_workgroups
            } {
                hpb >>= 1;
            }
        }
        hpb
    }

    /// The recurrent update's v-row split (`..._fp32state.hip:196-202`): fill the machine at one
    /// sequence, where the (N, H) grid is only 24 workgroups, but never past `vs_max` -- that is
    /// 64 threads, two waves. The fused row norm needs the whole row, so it forces 1.
    pub fn recurrent_vs(&self, n_seqs: u32, h: u32, norm: bool) -> u32 {
        let gd = &self.gdn;
        if norm {
            return 1;
        }
        let items = (n_seqs * h).max(1);
        let want = gd.vs_fill_workgroups.div_ceil(items);
        let vs = if want >= 4 {
            4
        } else if want >= 2 {
            2
        } else {
            1
        };
        vs.min(gd.vs_max)
    }

    /// The flat chunk-id grid both prefill kernels launch: an UPPER BOUND on the sum of the
    /// per-sequence chunk counts, not the exact count -- a block that falls past the end returns.
    pub fn chunk_upper_bound(&self, t: u32, n_seqs: u32) -> u32 {
        t.div_ceil(self.gdn.conv_chunk) + n_seqs
    }

    /// `_nblocks_q` in `ref_allreduce.py`, then `clamp_blocks` in the kernel's launcher.
    pub fn ar_blocks(&self, nbytes: u64, n_chunks: u32) -> u32 {
        let a = &self.ar;
        let mut nb = (nbytes / (a.words_per_block * 16)) as u32;
        nb = nb.clamp(a.min_nb, a.max_nb);
        nb.clamp(1, n_chunks.max(1)).min(a.max_blocks)
    }

    /// `_nblocks` (the EXACT path): the same target over 16-byte words rather than bytes, and its
    /// own cap -- PCIe saturates with few blocks and extra ones only help the reduce.
    pub fn ar_blocks_exact(&self, n16: u64) -> u32 {
        let a = &self.ar;
        let mut nb = (n16 / a.words_per_block) as u32;
        nb = nb.clamp(a.min_nb, a.max_nb_exact);
        nb.clamp(1, u32::try_from(n16.max(1)).unwrap_or(u32::MAX))
            .min(a.max_blocks)
    }

    /// The exact launcher's thread clamp: [64, 1024] with NO rounding to a multiple of 64. The
    /// quantised one rounds; two kernels, two clamps, both reproduced.
    pub fn ar_threads_exact(&self) -> u32 {
        self.ar.threads.clamp(64, 1024)
    }

    /// `should_custom_ar` + `_quant_ok` (`ref_allreduce.py`), as one answer: does this message
    /// take the LOSSY kernel? `false` means the exact one -- it does NOT mean RCCL, which
    /// [`ar_exact`] decides separately by refusing what the reference cannot take at all.
    pub fn ar_wants_lossy(&self, s: &ArShape, dtype: ArDtype) -> bool {
        let a = &self.ar;
        let nbytes = s.n_elem * s.elem_bytes;
        let _ = dtype; // both lossy dtypes are 16-bit; fp32 never reaches here
        nbytes >= a.quant_min_bytes && s.n_elem % a.group as u64 == 0
    }
}

// ---------------------------------------------------------------------------- the registry

/// A geometry query: the kwargs the reference's `select()` takes, with the same normalisation
/// (`norm_dtype`, and bool-before-int so `causal=True` means 1).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Geometry(Vec<(String, Val)>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Val {
    Num(i64),
    Str(String),
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Num(i) => write!(f, "{i}"),
            Val::Str(s) => write!(f, "{s}"),
        }
    }
}

/// `norm_dtype` (`ref_module.hip:308-330`): a caller may spell a dtype three ways and mean one.
pub fn norm_dtype(s: &str) -> &str {
    match s {
        "torch.bfloat16" | "bfloat16" => "bf16",
        "torch.float16" | "float16" | "half" => "fp16",
        "torch.float32" | "float32" | "float" => "fp32",
        "torch.float8_e4m3fn"
        | "float8_e4m3fn"
        | "torch.float8_e4m3fnuz"
        | "float8_e4m3fnuz"
        | "fp8" => "fp8_e4m3",
        other => other,
    }
}

impl Geometry {
    pub fn new() -> Self {
        Geometry(Vec::new())
    }

    pub fn num(mut self, key: &str, v: i64) -> Self {
        self.0.push((key.to_owned(), Val::Num(v)));
        self
    }

    pub fn dtype(mut self, key: &str, v: &str) -> Self {
        self.0
            .push((key.to_owned(), Val::Str(norm_dtype(v).to_owned())));
        self
    }

    fn get(&self, key: &str) -> Option<&Val> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The query as the log records it: normalised through the same `show(to_val())` the matcher
    /// compares on, so a record reads back as the question that was actually asked.
    pub fn pairs(&self) -> Vec<(String, String)> {
        self.0
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect()
    }
}

/// One candidate's verdict, as `explain()` returns it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub entry: String,
    pub ok: bool,
    pub reason: String,
}

/// One entry of the selection log: a question `select()` was asked and the answer it got.
///
/// `count` folds repeats, because one attention layer asks per instantiation and a 64-layer model
/// asks the same question 64 times -- and "asked 64 times" is how much of the model resolved to
/// that kernel. A refusal is recorded too: it is the reason a fallback is running, and
/// "no kernel" on its own is not something a reader can act on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Selection {
    pub op: String,
    pub query: Vec<(String, String)>,
    /// Empty when nothing matched.
    pub kernel: String,
    /// Empty when something did.
    pub reason: String,
    pub count: u32,
}

/// The reference's constraint registry, plus the log `select()` keeps.
///
/// The log lives here rather than in a global because Saber has no process-wide mutable state:
/// construct one `Registry` for an engine, hand it to [`resolve_with`] for every op it plans, and
/// read [`Registry::selections`] once the plan is built. [`resolve`] makes a throwaway one, which
/// is fine for a single call and useless for a report -- that is the difference between the two.
pub struct Registry<'a> {
    rows: &'a [RegistryRow],
    log: std::cell::RefCell<Vec<Selection>>,
}

impl<'a> Registry<'a> {
    pub fn new(rules: &'a SaberRules) -> Self {
        Registry {
            rows: &rules.registry,
            log: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Every distinct question asked through this registry, in the order first asked.
    pub fn selections(&self) -> Vec<Selection> {
        self.log.borrow().clone()
    }

    fn record(&self, op: &str, g: &Geometry, kernel: &str, reason: &str) {
        let query = g.pairs();
        let mut log = self.log.borrow_mut();
        if let Some(e) = log.iter_mut().find(|e| e.op == op && e.query == query) {
            e.count += 1;
            return;
        }
        log.push(Selection {
            op: op.to_owned(),
            query,
            kernel: kernel.to_owned(),
            reason: reason.to_owned(),
            count: 1,
        });
    }

    /// `matches()` (`ref_module.hip:358-394`). `Ok(())`, or the one constraint that failed and
    /// what it wanted — "no kernel" on its own is not an answer anyone can act on.
    fn matches(row: &RegistryRow, g: &Geometry) -> std::result::Result<(), String> {
        for c in &row.constraints {
            let Some(v) = g.get(&c.key) else {
                return Err(format!("{} not given", c.key));
            };
            if c.op == ConstraintOp::In {
                let hit = matches!(v, Val::Str(s) if c.sval.split(' ').any(|w| w == s));
                if !hit {
                    return Err(format!("{}={}, needs one of: {}", c.key, v, c.sval));
                }
                continue;
            }
            let Val::Num(n) = v else {
                return Err(format!("{}={} is not a number", c.key, v));
            };
            let (ok, rel) = match c.op {
                ConstraintOp::Eq => (*n == c.ival, "== "),
                ConstraintOp::Le => (*n <= c.ival, "<= "),
                ConstraintOp::Ge => (*n >= c.ival, ">= "),
                ConstraintOp::DivisibleBy => (c.ival != 0 && n % c.ival == 0, "divisible by "),
                ConstraintOp::In => unreachable!("handled above"),
            };
            if !ok {
                return Err(format!("{}={}, needs {}{}", c.key, n, rel, c.ival));
            }
        }
        Ok(())
    }

    /// `select_kernel()` (`ref_module.hip:514-535`): the first row of that op whose every
    /// predicate holds, or the refusal the reference would have logged.
    pub fn select(&self, op: &str, g: &Geometry) -> std::result::Result<&'a RegistryRow, String> {
        let mut misses: Vec<String> = Vec::new();
        for row in self.rows.iter().filter(|r| r.op == op) {
            match Self::matches(row, g) {
                Ok(()) => {
                    self.record(op, g, &row.entry, "");
                    return Ok(row);
                }
                Err(why) => {
                    if !misses.contains(&why) {
                        misses.push(why);
                    }
                }
            }
        }
        let reason = if misses.is_empty() {
            format!("no kernel in this build has op '{op}'")
        } else {
            misses.join("; ")
        };
        self.record(op, g, "", &reason);
        Err(reason)
    }

    /// `explain()`: every candidate for that op, matched or not. It deliberately does NOT record:
    /// it asks nothing, it only re-runs a question already recorded.
    pub fn explain(&self, op: &str, g: &Geometry) -> Vec<Verdict> {
        self.rows
            .iter()
            .filter(|r| r.op == op)
            .map(|r| match Self::matches(r, g) {
                Ok(()) => Verdict {
                    entry: r.entry.clone(),
                    ok: true,
                    reason: "ok".into(),
                },
                Err(why) => Verdict {
                    entry: r.entry.clone(),
                    ok: false,
                    reason: why,
                },
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------- ops and shapes

/// The paged-attention call shape: one run of requests with equal query length, which is exactly
/// what `ref_paged_attn.py:_plan()` cuts the batch into before each launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnShape {
    pub num_seqs: u32,
    pub q_len: u32,
    /// Rows each LIVE sequence feeds in this step — the `q_len` kernarg the kernels actually read.
    ///
    /// It is `q_len` for every decode bucket and for a prefill chunk that fills its bucket, and it
    /// is the CHUNK's row count for a padded one: the plan's row count is an arena fact (the
    /// GEMMs run at it, the embedding gathers one id per row of it) and the attention's is a STEP
    /// fact. Keeping them the same number is what made a padded chunk's unfed rows live query rows
    /// computed out of the previous request's leftovers (ws-l16), and the engine answered it with
    /// 1664 memsets a run. [`retarget_attn_prefill`] is how a step moves it.
    pub live_rows: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks: u32,
    /// Host-visible context bound; the split law reads it, the kernel reads `seqused_k`.
    pub max_ctx: u32,
    /// `0` = let the split law choose, which is what prod passes.
    pub splits: u32,
    pub kv_dtype: KvDtype,
    /// Elements between consecutive paged blocks, and between kv heads inside a block. Per-call
    /// (they come from the cache tensor, not from the model), so they are bound here rather than
    /// derived: the header is explicit that strides stay with the entry point.
    pub kv_block_stride: i64,
    pub kv_head_stride: i64,
    /// The softmax scale, as `f32` bits so the shape stays `Eq` — a launch key must compare
    /// exactly, and `f32` does not.
    pub scale_bits: u32,
    /// True when the per-tensor K/V scales are not 1.0 and the caller broadcasts them into a
    /// per-(sequence, head) buffer; false when they fold away and prod passes 0.
    pub has_descales: bool,
    /// **A ROW BAND's decode step (workstream C2): the band's row count, or 0 for the uniform
    /// kernel.**
    ///
    /// Non-zero resolves the RAGGED pair (`saber_attn_decode_ragged_*` +
    /// `saber_attn_splitkv_combine_ragged_*`, `saber_attn_ragged.hsaco`): each sequence's query length
    /// is `cu_q[seq + 1] - cu_q[seq]` (the step's cumulative LIVE-row prefix, bound as `cu_q`),
    /// `q_len` is the PLAN's max verify width (the scratch stride, the NW choice and the
    /// registry's `q_len <= 10`), and the combine runs `ragged_rows * q_heads` jobs — one per row
    /// of the band's arena, the rows past `cu_q[num_seqs]` writing zeros. `num_seqs * q_len >=
    /// ragged_rows` always (a band is clipped to `batch x max width`).
    pub ragged_rows: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    Fp8E4m3,
    Bf16,
}

impl KvDtype {
    fn name(self) -> &'static str {
        match self {
            KvDtype::Fp8E4m3 => "fp8_e4m3",
            KvDtype::Bf16 => "bf16",
        }
    }
}

/// The GDN decode step's shape. `n_seqs * h` is the item count the fused/pair route is keyed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnShape {
    pub n_seqs: u32,
    /// v heads per rank (24 at TP=2 for this model).
    pub h: u32,
    /// k heads per rank (8 at TP=2).
    pub hg: u32,
    pub head_k: u32,
    pub head_v: u32,
    pub conv_width: u32,
    /// The speculative window: `max_query_len`.
    pub maxq: u32,
    /// Padded row count of the output buffer (`core_attn_out.shape[0]`). The kernel zeroes rows
    /// `[cu[N], o_rows)`, which is why the caller may allocate that buffer uninitialised.
    pub o_rows: u32,
    /// Per-call strides, in ELEMENTS, taken from the caller's tensors (`ref_gdn.py:424-441`).
    /// `x` may be a padded view: the qkvz split the layer hands it.
    pub xpitch: i64,
    pub cs_seq: i64,
    pub cs_dim: i64,
    pub cs_tok: i64,
    pub ci_stride: i64,
    pub ab_stride: i64,
    pub st_slot: i64,
    pub st_head: i64,
    pub sidx_stride: i64,
    /// `head_k ** -0.5` and `SOFTPLUS_THRESHOLD`, as `f32` bits (see `AttnShape::scale_bits`).
    pub scale_bits: u32,
    pub softplus_thr_bits: u32,
    /// Prod always passes a `num_accepted_tokens` tensor; a caller that has none passes 0, and
    /// the kernel reads the accepted offset as 0.
    pub has_naccept: bool,
    pub has_bias: bool,
    /// The recurrent state cache is STORED as f16 (C1): the rows' `state_f16`. The strides above
    /// stay ELEMENT counts, so only the symbol changes.
    pub state_f16: bool,
}

/// The GDN PREFILL path's shape: `conv_prep` -> `kkt_solve` -> `chunk_scan` over `t` tokens in
/// `n_seqs` variable-length sequences. All three read the same `cu` and the same head counts; the
/// per-call strides come from the caller's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnPrefillShape {
    pub n_seqs: u32,
    /// Total tokens across the batch.
    pub t: u32,
    pub h: u32,
    pub hg: u32,
    pub head_k: u32,
    pub head_v: u32,
    pub conv_width: u32,
    pub chunk: u32,
    pub xpitch: i64,
    pub cs_seq: i64,
    pub cs_dim: i64,
    pub cs_tok: i64,
    pub ci_stride: i64,
    pub ab_stride: i64,
    /// `f32` bits (see `AttnShape::scale_bits`).
    pub scale_bits: u32,
    pub softplus_thr_bits: u32,
    pub has_bias: bool,
    /// `has_init`: one byte per sequence saying whether its conv state carries over. A chunked
    /// prefill passes it; a fresh one may pass 0.
    pub has_init: bool,
    /// `h0` / `ht` are STORED as f16 (C1): the scan's `state_f16` row.
    pub state_f16: bool,
}

/// The fused row norm the recurrent update can apply in its epilogue. Prod passes none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormGate {
    /// `f32` bits.
    pub eps_bits: u32,
    /// 0 = silu/swish, 1 = sigmoid.
    pub act: u32,
}

/// The gated RMS norm's shape: one row per (token, head).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormShape {
    pub rows: u64,
    pub channels: u32,
    pub xrow: i64,
    pub zrow: i64,
    pub orow: i64,
    /// `f32` bits (see `AttnShape::scale_bits`).
    pub eps_bits: u32,
}

/// One all-reduce message with the decoder layer's fused post-AR epilogue: `m_rows` rows of
/// `k_cols` bf16 channels, one workgroup per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArNqShape {
    pub m_rows: u32,
    pub k_cols: u32,
    /// `f32` bits (see `AttnShape::scale_bits`).
    pub eps_bits: u32,
    pub world_size: u32,
}

/// One all-reduce message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArShape {
    pub n_elem: u64,
    pub elem_bytes: u64,
    pub world_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Attn(AttnShape),
    Gdn(GdnShape),
    Norm(NormShape),
    Ar(ArShape),
    ArNq(ArNqShape),
    GdnPrefill(GdnPrefillShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Paged causal attention, query-tiled. `fp8_mode` is `SABER_ATTN_FP8`; prod runs 3.
    AttnPrefill { fp8_mode: u32 },
    /// Paged causal attention, split-KV. TWO launches: the decode kernel and the combine.
    AttnDecode,
    /// The GDN decode step as one launch (conv -> grid barrier -> recurrent).
    GdnFusedUpdate { ab_bf16: bool },
    /// The GDN decode step as the conv_update + recurrent_update PAIR -- what prod runs at a
    /// batch of two or more, which at TP=2 (H = 24 v heads) is every batch but one. TWO launches:
    /// this resolves the conv, and [`gdn_decode_pair_recurrent`] resolves the update that
    /// consumes its q/k/v.
    GdnDecodePair { ab_bf16: bool },
    /// The GDN prefill preamble: causal conv + qkv split + qk l2 norm + gating + per-chunk cumsum.
    GdnConvPrep { ab_bf16: bool },
    /// `A = (I + strict_lower(diag(beta) K K^T e^dg))^-1` per chunk; the fp32 gram never reaches
    /// HBM.
    GdnKktSolve,
    /// The chunked scan: WY recompute, state recurrence and output in one kernel.
    GdnChunkScan,
    /// The prefill path's gated RMS norm. `act` 0 = silu/swish, 1 = sigmoid.
    GdnGatedRmsNorm { act: u32 },
    /// The WHT-rotated, `bits`-per-symbol all-reduce. bf16 or fp16 payload.
    AllReduceWht { bits: u32, dtype: ArDtype },
    /// The exact one-shot push reduce, fp32 accumulate, bit-identical across ranks.
    AllReduceExact { dtype: ArPayload },
    /// The exact reduce with the fused residual-add + Gemma RMS norm + per-token e4m3 quant
    /// epilogue. bf16 only.
    AllReduceExactNq,
    /// **The policy op.** Picks exact or lossy from the message the way `ref_allreduce.py`
    /// does -- size, dtype and group alignment -- and then lets the registry pick the kernel.
    /// This is the one a planner should call; the two above are for a caller that has already
    /// decided.
    AllReduce { dtype: ArPayload },
}

/// What the LOSSY kernel takes: 16-bit only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArDtype {
    Bf16,
    Fp16,
}

impl ArDtype {
    fn name(self) -> &'static str {
        match self {
            ArDtype::Bf16 => "bf16",
            ArDtype::Fp16 => "fp16",
        }
    }
}

/// What the EXACT kernel takes: the lossy pair plus fp32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArPayload {
    Bf16,
    Fp16,
    Fp32,
}

impl ArPayload {
    fn name(self) -> &'static str {
        match self {
            ArPayload::Bf16 => "bf16",
            ArPayload::Fp16 => "fp16",
            ArPayload::Fp32 => "fp32",
        }
    }

    /// The lossy family has no fp32 arm, so a policy decision can only route these two to it.
    fn lossy(self) -> Option<ArDtype> {
        match self {
            ArPayload::Bf16 => Some(ArDtype::Bf16),
            ArPayload::Fp16 => Some(ArDtype::Fp16),
            ArPayload::Fp32 => None,
        }
    }
}

// ---------------------------------------------------------------------------- kernarg builder

/// Slots for a signature whose arguments are written one per `SigArg` — everything except the two
/// attention kernels, whose first argument is a struct the host fills field by field.
struct KArgs {
    sig: &'static [SigArg],
    values: Vec<ArgValue>,
}

impl KArgs {
    fn new(sig: &'static [SigArg]) -> Self {
        KArgs {
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

    /// A pointer the caller may legitimately pass as 0 (an absent conv bias, an absent
    /// `num_accepted` tensor). Named either way, so the descriptor says WHICH buffer is absent.
    fn maybe_buf(self, name: &str, present: bool) -> Self {
        if present {
            self.buf(name)
        } else {
            self.null_buf(name)
        }
    }

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
        let (slots, size) = abi::pack(self.sig);
        Ok((bind(self.sig, &slots, self.values)?, size))
    }
}

fn kind_of(v: &ArgValue) -> ArgKind {
    match v {
        ArgValue::Buffer { .. } | ArgValue::NullBuffer { .. } => ArgKind::Buffer,
        ArgValue::I32 { .. } => ArgKind::I32,
        ArgValue::I64 { .. } => ArgKind::I64,
        ArgValue::F32 { .. } => ArgKind::F32,
    }
}

fn bind(sig: &[SigArg], slots: &[(u32, u32)], values: Vec<ArgValue>) -> Result<Vec<Slot>> {
    if values.len() != sig.len() {
        return Err(Error::Rule(format!(
            "saber kernarg builder: {} values for a {}-argument signature",
            values.len(),
            sig.len()
        )));
    }
    let mut out = Vec::with_capacity(sig.len());
    for ((s, (off, sz)), v) in sig.iter().zip(slots).zip(values) {
        let want = kind_of(&v);
        if want != s.kind {
            return Err(Error::Rule(format!(
                "saber kernarg builder: `{}` is {:?} in the signature, bound as {want:?}",
                s.name, s.kind
            )));
        }
        out.push(Slot {
            name: s.name.to_owned(),
            offset: *off,
            size: *sz,
            value: v,
        });
    }
    Ok(out)
}

/// The `SABERArgs` fields, laid over the by-value struct at `base`, plus whatever follows it.
///
/// The code object reports ONE 136-byte `by_value` argument; the host has to write nine pointers
/// and eleven scalars into it. Both views are derived here from one signature, which is the whole
/// point: a kernarg layout the host and the kernel disagree about is memory corruption.
fn attn_args(shape: &AttnShape, splits: u32, tile: Option<u32>) -> Result<(Vec<Slot>, u32)> {
    let (field_slots, _) = abi::pack_fields_at(abi::SABER_ARGS_FIELDS, 0);
    let vals = vec![
        ArgValue::Buffer { buffer: "q".into() },
        ArgValue::Buffer {
            buffer: "kv".into(),
        },
        ArgValue::Buffer {
            buffer: "block_table".into(),
        },
        ArgValue::Buffer {
            buffer: "seqused_k".into(),
        },
        ArgValue::Buffer {
            buffer: "out".into(),
        },
        // Prod passes 0 for both descales when the per-tensor scales are 1.0 and fold away
        // (`ref_paged_attn.py:_descale_ptrs`), which the kernels read as "no descale".
        if shape.has_descales {
            ArgValue::Buffer {
                buffer: "k_descale".into(),
            }
        } else {
            ArgValue::NullBuffer {
                buffer: "k_descale".into(),
            }
        },
        if shape.has_descales {
            ArgValue::Buffer {
                buffer: "v_descale".into(),
            }
        } else {
            ArgValue::NullBuffer {
                buffer: "v_descale".into(),
            }
        },
        // The query is bf16 in every variant; `q_descale` is documented unused.
        ArgValue::NullBuffer {
            buffer: "q_descale".into(),
        },
        if splits > 0 {
            ArgValue::Buffer {
                buffer: "attn_decode_scratch".into(),
            }
        } else {
            ArgValue::NullBuffer {
                buffer: "attn_decode_scratch".into(),
            }
        },
        ArgValue::I32 {
            ival: shape.num_seqs as i64,
        },
        // THE KERNARG IS THE STEP'S LIVE ROWS, NOT THE BUCKET'S (ws-h27).
        //
        // Both kernels read this one field three ways: as the row stride into `q`/`out`
        // (`(seq * q_len + qrow) * q_heads`), as the live/padded predicate (`live = qpos < q_len`,
        // and a row past it duplicates row `q_len - 1` and stores nothing), and as the
        // right-alignment of the causal mask (`klimit = ctx - q_len + qpos`). A padded prefill
        // chunk that reported the BUCKET's row count therefore computed its unfed rows out of
        // whatever the arena held and let them steer the live rows' wave-wide softmax rescale
        // (ws-l16). [`AttnShape::live_rows`] is the count the step actually feeds; it equals
        // `q_len` everywhere except a padded prefill chunk, and `attn_decode` refuses any other
        // reading because there the field is also the `sidx` block width.
        ArgValue::I32 {
            ival: shape.live_rows as i64,
        },
        ArgValue::I32 {
            ival: shape.q_heads as i64,
        },
        ArgValue::I32 {
            ival: shape.kv_heads as i64,
        },
        ArgValue::I32 {
            ival: shape.head_dim as i64,
        },
        ArgValue::I32 {
            ival: shape.block_size as i64,
        },
        ArgValue::I32 {
            ival: shape.max_blocks as i64,
        },
        ArgValue::I64 {
            ival: shape.kv_block_stride,
        },
        ArgValue::I64 {
            ival: shape.kv_head_stride,
        },
        ArgValue::F32 {
            fval: f32::from_bits(shape.scale_bits),
        },
        ArgValue::I32 {
            ival: shape.splits as i64,
        },
        ArgValue::I32 {
            ival: shape.max_ctx as i64,
        },
    ];
    let mut out = bind(abi::SABER_ARGS_FIELDS, &field_slots, vals)?;
    let ragged = shape.ragged_rows > 0;
    let sig: &'static [SigArg] = match (tile.is_some(), splits > 0, ragged) {
        (true, _, true) => abi::SIG_SABER_ATTN_COMBINE_RAGGED,
        (true, _, false) => abi::SIG_SABER_ATTN_COMBINE,
        (false, true, true) => abi::SIG_SABER_ATTN_DECODE_RAGGED,
        (false, true, false) => abi::SIG_SABER_ATTN_DECODE,
        (false, false, _) => abi::SIG_SABER_ATTN_PREFILL,
    };
    let (slots, size) = abi::pack(sig);
    if splits > 0 {
        out.push(Slot {
            name: "splits".into(),
            offset: slots[1].0,
            size: slots[1].1,
            value: ArgValue::I32 {
                ival: splits as i64,
            },
        });
    }
    if let Some(t) = tile {
        out.push(Slot {
            name: "tile".into(),
            offset: slots[2].0,
            size: slots[2].1,
            value: ArgValue::I32 { ival: t as i64 },
        });
    }
    if ragged && splits > 0 {
        // The ragged pair's one extra kernarg, the last slot of either signature: the step's
        // cumulative live-row prefix (the `cu` law), which the plan binds per layer.
        let at = sig.len() - 1;
        out.push(Slot {
            name: "cu_q".into(),
            offset: slots[at].0,
            size: slots[at].1,
            value: ArgValue::Buffer {
                buffer: "cu_q".into(),
            },
        });
    }
    Ok((out, size))
}

// ---------------------------------------------------------------------------- resolve

/// The one decision function for saber. Pure: (op, shape, rules) in, descriptor out.
///
/// Every refusal below is either the registry's (the reference's own wording) or a per-call condition
/// the entry point checks — the split the header insists on and the reason the constraint table
/// is a NECESSARY condition only.
pub fn resolve(op: Op, shape: Shape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    resolve_with(op, shape, rules, &Registry::new(rules))
}

/// [`resolve`] against a registry the caller owns, so [`Registry::selections`] ends up holding
/// every question the whole plan asked rather than one throwaway answer.
pub fn resolve_with(
    op: Op,
    shape: Shape,
    rules: &SaberRules,
    reg: &Registry<'_>,
) -> Result<LaunchDescriptor> {
    match (op, shape) {
        (Op::AttnPrefill { fp8_mode }, Shape::Attn(s)) => attn_prefill(s, fp8_mode, reg, rules),
        (Op::AttnDecode, Shape::Attn(s)) => attn_decode(s, reg, rules),
        (Op::GdnFusedUpdate { ab_bf16 }, Shape::Gdn(s)) => gdn_fused_update(s, ab_bf16, reg, rules),
        (Op::GdnDecodePair { ab_bf16 }, Shape::Gdn(s)) => gdn_conv_update(s, ab_bf16, reg, rules),
        (Op::GdnConvPrep { ab_bf16 }, Shape::GdnPrefill(s)) => {
            gdn_conv_prep(s, ab_bf16, reg, rules)
        }
        (Op::GdnKktSolve, Shape::GdnPrefill(s)) => gdn_kkt_solve(s, reg, rules),
        (Op::GdnChunkScan, Shape::GdnPrefill(s)) => gdn_chunk_scan(s, reg, rules),
        (Op::GdnGatedRmsNorm { act }, Shape::Norm(s)) => gdn_norm(s, act, reg, rules),
        (Op::AllReduceWht { bits, dtype }, Shape::Ar(s)) => {
            ar_wht(s, bits, dtype, reg, rules, ArNames::DEFAULT)
        }
        (Op::AllReduceExact { dtype }, Shape::Ar(s)) => {
            ar_exact(s, dtype, reg, rules, ArNames::DEFAULT)
        }
        (Op::AllReduceExactNq, Shape::ArNq(s)) => ar_exact_nq(s, reg, rules),
        (Op::AllReduce { dtype }, Shape::Ar(s)) => {
            // The size decides the `exact` key; the registry decides the kernel. Same split
            // the reference draws, same two-step rule ref_allreduce.py applies per call.
            ar_family(s, dtype, reg, rules, ArNames::DEFAULT)
        }
        (o, _) => Err(Error::Rule(format!(
            "{o:?} was handed a shape of the wrong kind"
        ))),
    }
}

fn collective_plugin(symbol: &str, rules: &SaberRules) -> Result<&'static str> {
    let native = plugin_for(symbol)?;
    if !rules.ar.host_mapped {
        return Ok(native);
    }
    match native {
        "saber_ar_exact" => Ok("saber_ar_host_exact"),
        "saber_ar_exact_chk" => Ok("saber_ar_host_exact_chk"),
        other => Err(Error::Rule(format!(
            "host collectives require bounded exact kernels, got {other}"
        ))),
    }
}

fn attn_geometry(s: &AttnShape, with_q_len: bool, rules: &SaberRules) -> Result<Geometry> {
    if s.kv_heads == 0 || s.q_heads % s.kv_heads != 0 {
        return Err(Error::Rule(format!(
            "q_heads {} is not a whole multiple of kv_heads {}",
            s.q_heads, s.kv_heads
        )));
    }
    let mut g = Geometry::new()
        .num("head_dim", s.head_dim as i64)
        .num("gqa", (s.q_heads / s.kv_heads) as i64)
        .num("block_size", s.block_size as i64)
        .num("causal", 1)
        .dtype("q_dtype", "bf16")
        .dtype("kv_dtype", s.kv_dtype.name());
    if with_q_len {
        g = g.num("q_len", s.q_len as i64);
    }
    // Only a row band states the key: the ragged registry rows test it and the uniform rows do
    // not, so a uniform geometry fails the ragged rows by "ragged not given" and falls through.
    if s.ragged_rows > 0 {
        g = g.num("ragged", 1);
    }
    let _ = rules;
    Ok(g)
}

/// Query-row CTAs for a prefill chunk of `rows` live rows: one workgroup per `bq` query positions.
///
/// The one place this arithmetic is written. [`retarget_attn_prefill`] moves a resolved
/// descriptor's grid with it, so a retarget and a re-resolve cannot disagree.
#[must_use]
fn prefill_grid_x(rows: u32, bq: u32) -> u32 {
    rows.div_ceil(bq.max(1))
}

/// The one condition on [`AttnShape::live_rows`]: `1..=q_len`.
fn check_live_rows(s: &AttnShape, op: &str) -> Result<()> {
    if s.live_rows == 0 || s.live_rows > s.q_len {
        return Err(Error::Rule(format!(
            "{op}: live_rows {} in a {}-row bucket; a step feeds at least one row and is padded \
             UP into its bucket, never truncated",
            s.live_rows, s.q_len
        )));
    }
    Ok(())
}

/// Re-point a resolved `attn_prefill` descriptor at a chunk's live rows.
///
/// **Why this is a rewrite and not a re-resolve.** The prefill leg's registry geometry does not
/// carry `q_len` (`attn_geometry(.., with_q_len = false)`: the instantiation is chosen by head
/// dim, GQA, block size and the KV dtype), and nothing else in the descriptor is a function of the
/// row count — so the only bytes that move are the `q_len` kernarg and `grid[0]`. Held against a
/// full re-resolve by `a_padded_chunk_resolves_at_its_live_rows_and_a_retarget_is_a_re_resolve`,
/// which is what makes this an optimisation of `resolve` rather than a second law.
///
/// `bucket_q_len` is the row count the descriptor was PLANNED at, passed in rather than read back
/// out of the descriptor so that repeated retargets of one plan are all measured against the
/// bucket and never against each other.
///
/// # Errors
/// A descriptor that is not an `attn_prefill` one, a `live_rows` outside `1..=bucket_q_len`, or a
/// descriptor with no `q_len` slot — all three are refusals, because writing the wrong scalar into
/// an attention kernarg is a wrong answer and not an error.
pub fn retarget_attn_prefill(
    desc: &mut LaunchDescriptor,
    live_rows: u32,
    bucket_q_len: u32,
    rules: &SaberRules,
) -> Result<()> {
    if desc.op != "attn_prefill" {
        return Err(Error::Rule(format!(
            "retarget_attn_prefill: `{}` is a `{}` descriptor",
            desc.kernel, desc.op
        )));
    }
    if live_rows == 0 || live_rows > bucket_q_len {
        return Err(Error::Rule(format!(
            "retarget_attn_prefill: {live_rows} live row(s) in a {bucket_q_len}-row bucket"
        )));
    }
    let slot = desc
        .args
        .iter_mut()
        .find(|s| s.name == "q_len")
        .ok_or_else(|| {
            Error::Rule(format!(
                "retarget_attn_prefill: `{}` has no `q_len` kernarg slot",
                desc.kernel
            ))
        })?;
    slot.value = ArgValue::I32 {
        ival: i64::from(live_rows),
    };
    desc.grid[0] = prefill_grid_x(live_rows, rules.knobs.prefill_bq);
    Ok(())
}

fn attn_prefill(
    s: AttnShape,
    fp8_mode: u32,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    let g = attn_geometry(&s, false, rules)?;
    let row = reg.select("attn_prefill_paged", &g).map_err(Error::Rule)?;
    if s.q_len == 0 || s.num_seqs == 0 {
        return Err(Error::Rule("attn_prefill: empty batch".into()));
    }
    check_live_rows(&s, "attn_prefill")?;
    let sym = rules
        .attn_prefill
        .iter()
        .find(|r| r.entry == row.entry && r.fp8_mode == fp8_mode)
        .ok_or_else(|| {
            Error::Rule(format!(
                "{} has no fp8_mode {fp8_mode} instantiation (the 8-bit legs need the fp8 KV \
                 cache)",
                row.entry
            ))
        })?;
    let kn = &rules.knobs;
    let (args, bytes) = attn_args(&s, 0, None)?;
    Ok(LaunchDescriptor {
        op: "attn_prefill".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        // THE GRID COVERS THE LIVE ROWS, not the bucket's. A CTA holds `prefill_bq` query
        // positions, so the rows past the chunk cost a whole workgroup each and compute an answer
        // `if (!live) return;` throws away. What the last CTA still holds past `live_rows` is the
        // kernel's own duplicate of row `live_rows - 1` — a real query at the real key limit —
        // never an arena leftover.
        grid: [
            prefill_grid_x(s.live_rows, kn.prefill_bq),
            s.kv_heads,
            s.num_seqs,
        ],
        block: [kn.prefill_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn attn_decode(s: AttnShape, reg: &Registry<'_>, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let g = attn_geometry(&s, true, rules)?;
    let row = reg.select(decode_op(&s), &g).map_err(Error::Rule)?;
    check_ragged_entry(&s, &row.entry, "attn_decode")?;
    // A DECODE STEP'S LIVE ROWS ARE ITS BUCKET'S, and the kernel needs them to be: `q_len` is the
    // `sidx` block width and the row stride of a step where every live SLOT feeds the full verify
    // width (a shorter one is refused before it is planned). The padding at decode is whole slots,
    // and a padding slot is skipped by `seqused_k = 0` before a lane of it reaches a live row's
    // wave — which is why nothing in ws-h27 touches this leg.
    if s.live_rows != s.q_len {
        return Err(Error::Rule(format!(
            "attn_decode: live_rows {} against q_len {}; a decode step's `q_len` is also the \
             `sidx` block width and the q/out row stride, so the two are one number",
            s.live_rows, s.q_len
        )));
    }
    let kn = &rules.knobs;
    let gqa = s.q_heads / s.kv_heads;
    let rows = s.q_len * gqa;
    // The registry's `q_len <= 10` already covers this at gqa 6; the entry point checks the
    // product, which is the condition that actually binds, and returns -4 rather than running.
    if rows > kn.attn_max_decode_rows {
        return Err(Error::Rule(format!(
            "attn_decode: q_len {} x gqa {gqa} = {rows} rows exceeds the {} a decode workgroup \
             holds",
            s.q_len, kn.attn_max_decode_rows
        )));
    }
    let nw = rows.div_ceil(kn.decode_rows_per_warp).max(1);
    let sym = rules
        .attn_decode
        .iter()
        .find(|r| r.entry == row.entry && r.nw == nw)
        .ok_or_else(|| Error::Rule(format!("{}: no nw={nw} instantiation", row.entry)))?;
    let combine = combine_row(&s, rules)?;
    let splits = rules.decode_splits(s.max_ctx, s.kv_heads, s.num_seqs, s.splits);
    let need = rules.decode_scratch_bytes(&s, splits);
    let ws = rules.workspace("attn_decode_scratch")?;
    if need > ws.capacity_bytes {
        return Err(Error::Rule(format!(
            "attn_decode: {need} B of split-KV partials exceeds the {} B declared for \
             attn_decode_scratch; the buffer is sized once at construction because allocating \
             during a graph capture is not allowed",
            ws.capacity_bytes
        )));
    }
    let (args, bytes) = attn_args(&s, splits, None)?;
    let workspaces = vec![WorkspaceReq {
        name: "attn_decode_scratch".into(),
        kind: ws.kind,
        bytes: need,
        zero_once: false,
    }];
    let _ = combine;
    Ok(LaunchDescriptor {
        op: "attn_decode".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        grid: [splits, s.kv_heads, s.num_seqs],
        block: [nw * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        // The split-KV combine is a SECOND launch, not a fused epilogue: the segments a token's
        // partials live in are spread across `splits` workgroups of the first launch.
        n_launches: 2,
    })
}

/// The second launch of [`attn_decode`]. Separate because a descriptor describes one dispatch;
/// `attn_decode`'s `n_launches = 2` is the declaration that this one follows it.
pub fn attn_decode_combine(s: AttnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let reg = Registry::new(rules);
    let g = attn_geometry(&s, true, rules)?;
    let row = reg.select(decode_op(&s), &g).map_err(Error::Rule)?;
    check_ragged_entry(&s, &row.entry, "attn_decode_combine")?;
    // The same refusal as `attn_decode`, and for the same reason: this reads `q_len` as the row
    // stride of the partials the first launch wrote, so the pair has to agree about it.
    if s.live_rows != s.q_len {
        return Err(Error::Rule(format!(
            "attn_decode_combine: live_rows {} against q_len {}; the combine walks \
             `num_seqs * q_len * q_heads` rows of partials, which is the count the decode kernel \
             wrote",
            s.live_rows, s.q_len
        )));
    }
    let kn = &rules.knobs;
    let combine = combine_row(&s, rules)?;
    let splits = rules.decode_splits(s.max_ctx, s.kv_heads, s.num_seqs, s.splits);
    // One job per (row, q head) of the OUTPUT the step owns: `num_seqs * q_len` rows for the
    // uniform kernel, the band's own rows for the ragged one (C2) — whose arena holds exactly
    // that many, so a `num_seqs * max width` grid would write past it.
    let jobs = if s.ragged_rows > 0 {
        s.ragged_rows * s.q_heads
    } else {
        s.num_seqs * s.q_len * s.q_heads
    };
    let (args, bytes) = attn_args(&s, splits, Some(kn.decode_tile))?;
    let ws = rules.workspace("attn_decode_scratch")?;
    Ok(LaunchDescriptor {
        op: "attn_decode_combine".into(),
        plugin: plugin_for(&combine.symbol)?.into(),
        kernel: combine.symbol.clone(),
        grid: [jobs, 1, 1],
        block: [kn.combine_block, 1, 1],
        // `extern __shared__ float sw[]`: one float per split, sized at launch.
        shared_bytes: splits * 4,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![WorkspaceReq {
            name: "attn_decode_scratch".into(),
            kind: ws.kind,
            bytes: rules.decode_scratch_bytes(&s, splits),
            zero_once: false,
        }],
        n_launches: 1,
    })
}

/// **The decode attention with its split-KV combine done by the LAST-ARRIVING split block**
/// (ws-h38, fusion-map row 6): the [`attn_decode`] + [`attn_decode_combine`] pair in ONE dispatch.
///
/// The split pass is [`attn_decode`]'s own descriptor -- the same grid `(splits, kv_heads,
/// num_seqs)`, block, split count and partial scratch -- under the `[[attn_decode_declc]]` row's
/// symbol for that warp count. Each block stores its partial exactly as before, fences, and adds
/// one to its `(sequence, kv head)` item's counter (`attn_decode_counter`, a Persistent zero_once
/// workspace the last arriver resets); the block that completes the item runs the combine for
/// that item's `q_len * gqa` jobs with the pf1 combine's arithmetic in its own order. No block
/// waits, so there is no grid barrier and no co-residency requirement.
///
/// Refused -- the planner then keeps the pair -- when the split law answers ONE split (nothing to
/// merge), for anything but the fp8 cache with f16 partials (the only instantiation), for a split
/// count past the row's `max_splits`, or for more items than the counter holds.
pub fn attn_decode_declc(s: AttnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let head = resolve(Op::AttnDecode, Shape::Attn(s), rules)?;
    // The pair it replaces must resolve: the same refusals (live rows, the partial format).
    attn_decode_combine(s, rules)?;
    let kn = &rules.knobs;
    if s.kv_dtype != KvDtype::Fp8E4m3 || !kn.combine_partial_f16 {
        return Err(Error::Rule(format!(
            "attn_decode_declc: only the fp8 cache with f16 partials is instantiated ({} KV, \
             f16 partials {})",
            s.kv_dtype.name(),
            kn.combine_partial_f16
        )));
    }
    if kn.decode_tile != 16 || kn.attn_gqa != 6 || kn.attn_head_dim != 256 {
        return Err(Error::Rule(
            "attn_decode_declc: TILE 16, gqa 6 and head_dim 256 are compiled into the merge".into(),
        ));
    }
    let splits = head.grid[0];
    if splits <= 1 {
        return Err(Error::Rule(format!(
            "attn_decode_declc: the split law answers {splits} split(s) at max_ctx {}; one \
             segment has nothing to merge, so the pair stays",
            s.max_ctx
        )));
    }
    let nw = head.block[0] / 32;
    let row = rules
        .attn_decode_declc
        .iter()
        .find(|r| r.nw == nw && head.kernel.ends_with(&format!("_nw{nw}")))
        .ok_or_else(|| {
            Error::Rule(format!(
                "attn_decode_declc: no last-arriver row for `{}` (nw {nw})",
                head.kernel
            ))
        })?;
    if splits > row.max_splits {
        return Err(Error::Rule(format!(
            "attn_decode_declc: {splits} splits exceed the {} the merge's weight row holds",
            row.max_splits
        )));
    }
    let ws = rules.workspace("attn_decode_counter")?;
    let items = u64::from(s.num_seqs) * u64::from(s.kv_heads);
    if items * 4 > ws.capacity_bytes {
        return Err(Error::Rule(format!(
            "attn_decode_declc: {items} (sequence, kv head) items need {} B of counters, the \
             workspace holds {}",
            items * 4,
            ws.capacity_bytes
        )));
    }
    // The decode kernel's own slots (the SABERArgs blob and `splits`), then the counter.
    let (slots, size) = abi::pack(abi::SIG_SABER_ATTN_DECLC);
    let mut args = head.args.clone();
    args.push(Slot {
        name: "attn_decode_counter".into(),
        offset: slots[2].0,
        size: slots[2].1,
        value: ArgValue::Buffer {
            buffer: "attn_decode_counter".into(),
        },
    });
    let mut workspaces = head.workspaces.clone();
    workspaces.push(WorkspaceReq {
        name: "attn_decode_counter".into(),
        kind: ws.kind,
        bytes: ws.capacity_bytes,
        zero_once: true,
    });
    Ok(LaunchDescriptor {
        op: "attn_decode_declc".into(),
        plugin: plugin_for(&row.symbol)?.into(),
        kernel: row.symbol.clone(),
        grid: head.grid,
        block: head.block,
        shared_bytes: 0,
        kernarg_bytes: size,
        args,
        workspaces,
        n_launches: 1,
    })
}

/// The registry op a decode attention shape selects: [`ATTN_DECODE_RAGGED_OP`] for a row band
/// (C2), `attn_decode_paged` for a rectangle.
fn decode_op(s: &AttnShape) -> &'static str {
    if s.ragged_rows > 0 {
        ATTN_DECODE_RAGGED_OP
    } else {
        "attn_decode_paged"
    }
}

/// A ragged shape must resolve to a `_ragged` registry entry and a uniform one to a uniform entry,
/// refused BY NAME either way (C2). The registry's own ordering makes the first impossible and the
/// `ragged` key the second; this is the statement that does not depend on the table's order.
fn check_ragged_entry(s: &AttnShape, entry: &str, op: &str) -> Result<()> {
    let is_ragged_entry = entry.ends_with("_ragged");
    if (s.ragged_rows > 0) != is_ragged_entry {
        return Err(Error::Rule(format!(
            "{op}: a {} shape resolved to `{entry}`; the uniform kernel right-aligns every \
             sequence on ONE q_len and the ragged one reads each sequence's from `cu_q`, so the \
             two are never substitutes",
            if s.ragged_rows > 0 {
                format!("RAGGED ({}-row band)", s.ragged_rows)
            } else {
                "uniform".to_owned()
            }
        )));
    }
    if s.ragged_rows > s.num_seqs * s.q_len {
        return Err(Error::Rule(format!(
            "{op}: a {}-row band over {} sequence(s) at max width {}: the partials hold \
             num_seqs x q_len rows, so a band is clipped to batch x max width",
            s.ragged_rows, s.num_seqs, s.q_len
        )));
    }
    Ok(())
}

/// The combine row for this shape's partial format and kind (C2: ragged or uniform).
fn combine_row<'r>(s: &AttnShape, rules: &'r SaberRules) -> Result<&'r CombineRow> {
    let ragged = s.ragged_rows > 0;
    rules
        .attn_combine
        .iter()
        .find(|r| r.pf16 == rules.knobs.combine_partial_f16 && r.ragged == ragged)
        .ok_or_else(|| {
            Error::Rule(format!(
                "no attn_combine row for the declared partial format (ragged={ragged})"
            ))
        })
}

/// Fusion-map ROW 5 (ws-h42): the one fused-update instantiation compiled with the norm tail --
/// the serving default's f16 stored state and bf16 a/b (`saber_gdn_fused_update_f16s_w4k128v128_ab1`).
pub const GDN_FNQ_FROM: &str = "saber_gdn_fused_update_f16s_w4k128v128_ab1";
/// Row 5's symbol ([`gdn_fused_update_nq`]).
pub const GDN_FNQ_SYMBOL: &str = "saber_gdn_fnq_f16s_w4k128v128_ab1";
/// Row 5's code object (its own: a plugin no registry row binds moves no autotune key).
pub const GDN_FNQ_PLUGIN: &str = "saber_gdn_fnq";
/// Row 5's Persistent arrival counter: monotonic, 64-bit, its own name (row 4's `smq_tiles` law).
pub const GDN_FNQ_TILES: &str = "fnq_tiles";

/// Fusion-map ROW 5 (ws-h42): the decode GDN fused update `fu` and the `gdn_norm_quant` `norm`
/// that reads its `o` as ONE dispatch. The fused update runs exactly as alone (conv -> its own
/// grid barrier -> recurrent -> pad-row zero); then every workgroup takes a ticket on a monotonic
/// counter and the last `ep_rows` arrivals each run the norm's generated body on one row. The
/// norm's operands are renamed `nq_*` (its `x` IS `o`; its `q` / `scale` would collide with the
/// update's own `q`). Refuses anything but the compiled instantiation, the untiled 256-thread
/// norm over `o_rows` rows of `H x head_v` columns, and a grid with fewer workgroups than rows.
///
/// `ep_rows` is the norm's M in production; the gate's `fnq-rows` control passes `M - 1`.
///
/// # Errors
/// [`Error::Rule`] naming the first condition that fails.
pub fn gdn_fused_update_nq(
    fu: &LaunchDescriptor,
    norm: &LaunchDescriptor,
    ep_rows: i64,
) -> Result<LaunchDescriptor> {
    let refuse = |why: String| Error::Rule(format!("gdn_fused_update_nq: {why}"));
    if fu.kernel != GDN_FNQ_FROM {
        return Err(refuse(format!(
            "`{}` is not the compiled row-5 instantiation `{GDN_FNQ_FROM}`",
            fu.kernel
        )));
    }
    if norm.op != "gdn_norm_quant" || norm.kernel != "gdn_norm_quant_t256" || norm.block[0] != 256 {
        return Err(refuse(format!(
            "the producer is `{}` ({} threads), not the untiled 256-thread `gdn_norm_quant`",
            norm.kernel, norm.block[0]
        )));
    }
    let int = |d: &LaunchDescriptor, name: &str| match d.arg(name).map(|x| &x.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => Ok(*ival),
        _ => Err(refuse(format!("`{}` has no integer `{name}`", d.kernel))),
    };
    let rows = i64::from(norm.grid[0]);
    let (o_rows, h, num_wg) = (int(fu, "o_rows")?, int(fu, "h")?, int(fu, "num_wg")?);
    let n = int(norm, "n")?;
    if rows != o_rows || n != h * 128 {
        return Err(refuse(format!(
            "the norm is {rows} x {n} and the update writes {o_rows} x {} (H {h} x 128)",
            h * 128
        )));
    }
    if rows == 0 || num_wg < rows || i64::from(fu.grid[0]) != num_wg {
        return Err(refuse(format!(
            "M = {rows} rows need M row owners among {num_wg} workgroups"
        )));
    }
    let eps = match norm.arg("eps").map(|x| &x.value) {
        Some(ArgValue::F32 { fval }) => *fval,
        _ => return Err(refuse("the norm has no f32 `eps`".into())),
    };
    let mut k = KArgs::new(abi::SIG_SABER_GDN_FNQ);
    k.values = fu.args.iter().map(|a| a.value.clone()).collect();
    let (args, bytes) = k
        .buf("nq_z")
        .i64(int(norm, "zs")?)
        .buf("nq_w")
        .buf("nq_q")
        .buf("nq_scale")
        .i32(n)
        .f32(eps)
        .buf(GDN_FNQ_TILES)
        .i32(ep_rows)
        .finish()?;
    let mut workspaces = fu.workspaces.clone();
    workspaces.push(WorkspaceReq {
        name: GDN_FNQ_TILES.into(),
        kind: WorkspaceKind::Persistent,
        bytes: 8,
        zero_once: true,
    });
    Ok(LaunchDescriptor {
        op: "gdn_fnq".into(),
        plugin: GDN_FNQ_PLUGIN.into(),
        kernel: GDN_FNQ_SYMBOL.into(),
        grid: fu.grid,
        block: fu.block,
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        n_launches: 1,
    })
}

fn gdn_fused_update(
    s: GdnShape,
    ab_bf16: bool,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    let g = Geometry::new()
        .num("conv_width", s.conv_width as i64)
        .num("head_k", s.head_k as i64)
        .num("head_v", s.head_v as i64);
    reg.select("gdn_fused_update", &g).map_err(Error::Rule)?;
    let gd = &rules.gdn;
    if s.hg == 0 || s.h % s.hg != 0 {
        return Err(Error::Rule(format!(
            "gdn_fused_update: H {} is not a whole multiple of Hg {}",
            s.h, s.hg
        )));
    }
    let items = s.n_seqs * s.h;
    // SABER_GDN_FUSED_MAX_ITEMS. Above it prod runs the conv+recurrent PAIR, which is not in
    // this plugin: refuse loudly and name the route rather than launching a fused kernel whose
    // work-loop would serialise the overflow onto FU_MAXWG workgroups.
    if items > gd.fused_max_items {
        return Err(Error::Rule(format!(
            "gdn_fused_update: {} sequences x {} v heads = {items} items exceeds \
             fused_max_items {}; above it the decode step takes the conv_update + \
             recurrent_update pair, which this plugin does not carry",
            s.n_seqs, s.h, gd.fused_max_items
        )));
    }
    let sym = rules
        .gdn_fused_update
        .iter()
        .find(|r| r.ab_bf16 == ab_bf16 && r.state_f16 == s.state_f16)
        .ok_or_else(|| {
            Error::Rule(format!(
                "no gdn_fused_update row for ab_bf16={ab_bf16} state_f16={}",
                s.state_f16
            ))
        })?;
    let num_wg = items.clamp(1, gd.fused_max_wg);
    let ws = rules.workspace("gdn_fused_barrier")?;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_FUSED_UPDATE)
        .buf("x")
        .i64(s.xpitch)
        .buf("wgt")
        .maybe_buf("bias", s.has_bias)
        .buf("cstate")
        .i64(s.cs_seq)
        .i64(s.cs_dim)
        .i64(s.cs_tok)
        // state_len_max = (width - 1) + (max_query_len - 1): the rolling speculative window over
        // the conv state cache (`ref_gdn.py:647`).
        .i32(((s.conv_width - 1) + (s.maxq - 1)) as i64)
        .buf("cache_idx")
        .i64(s.ci_stride)
        .maybe_buf("naccept", s.has_naccept)
        .buf("cu")
        .i32(s.n_seqs as i64)
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .i32(s.maxq as i64)
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("av")
        .buf("bv")
        .i64(s.ab_stride)
        .buf("a_log")
        .buf("dt_bias")
        // The GDN state is CALLER-OWNED (the plan's separate GDN-state allocator), never a
        // workspace: it is the sequence's memory, it is paged, and it outlives every launch.
        .buf("state")
        .i64(s.st_slot)
        .i64(s.st_head)
        .buf("o")
        .buf("sidx")
        .i64(s.sidx_stride)
        .f32(f32::from_bits(s.scale_bits))
        .f32(f32::from_bits(s.softplus_thr_bits))
        .buf("gdn_fused_barrier")
        .i32(num_wg as i64)
        .i32(s.o_rows as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_fused_update".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        grid: [num_wg, 1, 1],
        block: [gd.fused_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![WorkspaceReq {
            name: "gdn_fused_barrier".into(),
            kind: ws.kind,
            bytes: ws.capacity_bytes,
            zero_once: true,
        }],
        // The whole point of this kernel: conv -> grid barrier -> recurrent, ONE launch.
        n_launches: 1,
    })
}

/// The geometry every GDN kernel is asked about, minus the keys a particular op's rows do not
/// carry -- `select()` ignores what a row does not mention, so passing more is harmless and
/// passing less is "<key> not given".
fn gdn_geometry(s: &GdnPrefillShape) -> Geometry {
    Geometry::new()
        .num("conv_width", s.conv_width as i64)
        .num("head_k", s.head_k as i64)
        .num("head_v", s.head_v as i64)
        .num("chunk", s.chunk as i64)
}

/// `H % Hg` is a per-call refusal in three of these entry points (-2), not a registry predicate:
/// it relates two parameters, and a constraint is one predicate on ONE named parameter.
fn heads_divide(h: u32, hg: u32, what: &str) -> Result<u32> {
    if hg == 0 || h % hg != 0 {
        return Err(Error::Rule(format!(
            "{what}: H {h} is not a whole multiple of Hg {hg}"
        )));
    }
    Ok(h / hg)
}

fn gdn_conv_prep(
    s: GdnPrefillShape,
    ab_bf16: bool,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    reg.select("gdn_conv_prep", &gdn_geometry(&s))
        .map_err(Error::Rule)?;
    heads_divide(s.h, s.hg, "gdn_conv_prep")?;
    let hpb = rules.conv_hpb(s.h, s.hg, None);
    let sym = rules
        .gdn_conv_prep
        .iter()
        .find(|r| r.hpb == hpb && r.ab_bf16 == ab_bf16)
        .ok_or_else(|| {
            Error::Rule(format!(
                "gdn_conv_prep: no hpb={hpb} ab_bf16={ab_bf16} instantiation"
            ))
        })?;
    let units = 2 * s.hg + s.h;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_CONV_PREP)
        .buf("x")
        .i64(s.xpitch)
        .buf("wgt")
        .maybe_buf("bias", s.has_bias)
        .buf("cstate")
        .i64(s.cs_seq)
        .i64(s.cs_dim)
        .i64(s.cs_tok)
        .buf("cache_idx")
        .i64(s.ci_stride)
        .maybe_buf("has_init", s.has_init)
        .buf("av")
        .buf("bv")
        .i64(s.ab_stride)
        .buf("a_log")
        .buf("dt_bias")
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("g")
        .buf("beta")
        .buf("cu")
        .i32(s.n_seqs as i64)
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .f32(f32::from_bits(s.softplus_thr_bits))
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_conv_prep".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        // The flat chunk id is an UPPER BOUND on the sum of the per-sequence chunk counts; a block
        // past the end returns rather than the host computing the exact total on the CPU.
        grid: [rules.chunk_upper_bound(s.t, s.n_seqs), units / hpb, 1],
        block: [hpb * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn gdn_kkt_solve(
    s: GdnPrefillShape,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    reg.select("gdn_kkt_solve", &gdn_geometry(&s))
        .map_err(Error::Rule)?;
    let ratio = heads_divide(s.h, s.hg, "gdn_kkt_solve")?;
    let sym = rules
        .gdn_kkt_solve
        .iter()
        .find(|r| r.ratio == ratio)
        .ok_or_else(|| {
            Error::Rule(format!(
                "gdn_kkt_solve: H/Hg = {ratio} has no instantiation (the reference carries \
                 1, 2, 3, 4, 6 and 8 and returns -3 for anything else)"
            ))
        })?;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_KKT)
        .buf("k")
        .buf("beta")
        .buf("g")
        .buf("a_out")
        .buf("cu")
        .i32(s.n_seqs as i64)
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_kkt_solve".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        // One gram per k head, applied to every v head of the group.
        grid: [rules.chunk_upper_bound(s.t, s.n_seqs), s.hg, 1],
        block: [rules.gdn.kkt_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn gdn_chunk_scan(
    s: GdnPrefillShape,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    reg.select("gdn_chunk_scan", &gdn_geometry(&s))
        .map_err(Error::Rule)?;
    heads_divide(s.h, s.hg, "gdn_chunk_scan")?;
    let gd = &rules.gdn;
    let sym = rules
        .gdn_chunk_scan
        .iter()
        .find(|r| r.state_f16 == s.state_f16)
        .ok_or_else(|| {
            Error::Rule(format!(
                "no gdn_chunk_scan row for state_f16={}",
                s.state_f16
            ))
        })?;
    if s.head_v % gd.scan_bv != 0 {
        return Err(Error::Rule(format!(
            "gdn_chunk_scan: head_v {} is not a multiple of the {}-wide v tile",
            s.head_v, gd.scan_bv
        )));
    }
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_CHUNK_SCAN)
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("a_mat")
        .buf("gcs")
        .buf("beta")
        // h0 and ht are the caller's fp32 state, per sequence -- caller-owned, like every other
        // GDN state buffer, and never a workspace.
        .buf("h0")
        .buf("o")
        .buf("ht")
        .buf("cu")
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .f32(f32::from_bits(s.scale_bits))
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_chunk_scan".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        grid: [s.head_v / gd.scan_bv, s.h, s.n_seqs],
        block: [gd.scan_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// The first launch of the decode PAIR. `n_launches = 2`: the recurrent update consumes the q/k/v
/// this kernel writes, and [`gdn_decode_pair_recurrent`] resolves it.
fn gdn_conv_update(
    s: GdnShape,
    ab_bf16: bool,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    let g = Geometry::new()
        .num("conv_width", s.conv_width as i64)
        .num("head_k", s.head_k as i64)
        .num("head_v", s.head_v as i64);
    reg.select("gdn_conv_update", &g).map_err(Error::Rule)?;
    heads_divide(s.h, s.hg, "gdn_conv_update")?;
    // The conv kernel itself does not read the gates, so ab_bf16 discriminates nothing here; it
    // is carried so a caller states the pair's dtype once and both halves agree.
    let _ = ab_bf16;
    let hpb = rules.conv_hpb(s.h, s.hg, Some(s.n_seqs));
    let sym = rules
        .gdn_conv_update
        .iter()
        .find(|r| r.hpb == hpb)
        .ok_or_else(|| Error::Rule(format!("gdn_conv_update: no hpb={hpb} instantiation")))?;
    let units = 2 * s.hg + s.h;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_CONV_UPDATE)
        .buf("x")
        .i64(s.xpitch)
        .buf("wgt")
        .maybe_buf("bias", s.has_bias)
        .buf("cstate")
        .i64(s.cs_seq)
        .i64(s.cs_dim)
        .i64(s.cs_tok)
        .i32(((s.conv_width - 1) + (s.maxq - 1)) as i64)
        .buf("cache_idx")
        .i64(s.ci_stride)
        .maybe_buf("naccept", s.has_naccept)
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("cu")
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .i32(s.maxq as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_conv_update".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        grid: [s.n_seqs, units / hpb, 1],
        block: [hpb * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        // conv_update THEN recurrent_update. Declared here so the pair is budgetable as one
        // decision, the way the fused kernel's single launch is.
        n_launches: 2,
    })
}

/// The second launch of the decode pair. `norm` folds the gated RMS norm into the epilogue and
/// needs the whole 128-wide row, so it forces VS = 1; prod passes `None`.
pub fn gdn_decode_pair_recurrent(
    s: GdnShape,
    ab_bf16: bool,
    norm: Option<NormGate>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    let reg = Registry::new(rules);
    let g = Geometry::new()
        .num("head_k", s.head_k as i64)
        .num("head_v", s.head_v as i64);
    reg.select("gdn_recurrent_update", &g)
        .map_err(Error::Rule)?;
    heads_divide(s.h, s.hg, "gdn_recurrent_update")?;
    let vs = rules.recurrent_vs(s.n_seqs, s.h, norm.is_some());
    let sym = rules
        .gdn_recurrent_update
        .iter()
        .find(|r| {
            r.ab_bf16 == ab_bf16
                && r.norm == norm.is_some()
                && r.vs == vs
                && r.state_f16 == s.state_f16
        })
        .ok_or_else(|| {
            Error::Rule(format!(
                "gdn_recurrent_update: no ab_bf16={ab_bf16} norm={} vs={vs} instantiation \
                 (the fused row norm needs the whole {}-wide row, so it exists at vs=1 only)",
                norm.is_some(),
                s.head_v
            ))
        })?;
    let threads = rules.gdn.recurrent_row_threads / vs;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_RECURRENT)
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("av")
        .buf("bv")
        .i64(s.ab_stride)
        .buf("a_log")
        .buf("dt_bias")
        // Caller-owned paged state, never a workspace.
        .buf("state")
        .i64(s.st_slot)
        .i64(s.st_head)
        .buf("o")
        .buf("cu")
        .buf("sidx")
        .i64(s.sidx_stride)
        .maybe_buf("naccept", s.has_naccept)
        .maybe_buf("zg", norm.is_some())
        .maybe_buf("nw", norm.is_some())
        .f32(norm.map_or(0.0, |n| f32::from_bits(n.eps_bits)))
        .i32(norm.map_or(0, |n| n.act) as i64)
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .f32(f32::from_bits(s.scale_bits))
        .f32(f32::from_bits(s.softplus_thr_bits))
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_recurrent_update".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        grid: [s.n_seqs, s.h, vs],
        block: [threads, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

// ---------------------------------------------------------------------------- P27-W2-5: the replay form

/// Bytes of ONE stash region of the replay form (`p27w2_gdn_replay.hip` `RP_SX_HEAD`): header 16 |
/// a / b at 64 | k [8][128] bf16 at 128 | v [8][128] bf16 at 2,176.
pub const REPLAY_SX_HEAD: u64 = 4224;
/// Rows a sequence the replay form stashes (`RP_TM`): the widest verify.
pub const REPLAY_MAX_ROWS: u32 = 8;
/// The shipped recurrent update the replay form replaces, by symbol prefix (f16 state, bf16 a/b,
/// no fused norm), and the replay symbol prefix (`_vs{1,2,4}` follows both).
pub const REPLAY_FROM_PREFIX: &str = "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs";
/// See [`REPLAY_FROM_PREFIX`].
pub const REPLAY_PREFIX: &str = "p27w2_gdn_replay_f16s_k128_v128_ab1_vs";
/// The materialize launch (VS 1): a pending replay into the base, no rows of its own.
pub const REPLAY_MATERIALIZE: &str = "p27w2_gdn_replay_materialize_f16s_k128_v128_ab1_vs1";
/// CMP27T: the replay form with the shared operands staged once a workgroup in LDS
/// (`hip/p27w2_gdn_replay_lds.hip`, the `--engine-c4-arms replay` ARM): the shipped replay's bytes,
/// the same signature, grid and block; `_vs{1,2,4}` follows.
pub const REPLAY_LDS_PREFIX: &str = "p27w2_gdn_replay_lds_f16s_k128_v128_ab1_vs";
/// CMP27T: the LDS twin's code object (its own: `p27w2_gdn_replay` keeps its bytes).
pub const REPLAY_LDS_PLUGIN: &str = "p27w2_gdn_replay_lds";

/// **CMP27T: the conc-4 arm's replay form by batch** -- `(max sequences, LDS twin, VS)`, first
/// match wins. MEASURED (card-1 probe p1, `tools/capture/cmp27t/probe.py`, IC-cold, r 5, T 8, us a
/// launch): N 2 shipped vs2 20.6 / LDS vs2 18.1; N 4 shipped vs1 39.2 / LDS vs1 30.0 (vs2 31.1);
/// N 8 shipped vs1 73.0 / shipped vs4 62.4 / LDS vs4 65.8 / LDS vs1 66.8 -- but IN SITU (engine
/// gates g1 / g2, harness b8 sweep) VS 4 at 8 sequences ran +0.15 ms a step, so the b8 row keeps
/// the shipped form (VS 1). Every VS of both bodies writes the shipped replay's bytes (168 oracle
/// chains, CMP27G's VS 1 / 2 / 4 oracle).
pub const REPLAY_C4_TABLE: [(u32, bool, u32); 3] =
    [(2, true, 2), (4, true, 1), (u32::MAX, false, 1)];

/// **CMP27T: a replay descriptor in the conc-4 arm's form** ([`REPLAY_C4_TABLE`] by its grid's
/// sequences): the same kernargs; the symbol, code object, grid z (VS) and block follow the row.
///
/// # Errors
/// A descriptor that is not the replay form's verify launch.
pub fn gdn_replay_c4_form(d: &LaunchDescriptor) -> Result<LaunchDescriptor> {
    if d.kernel.strip_prefix(REPLAY_PREFIX).is_none() {
        return Err(Error::Rule(format!(
            "gdn_replay_c4: `{}` is not the replay form's verify launch",
            d.kernel
        )));
    }
    let n = d.grid[0];
    let (_, lds, vs) = REPLAY_C4_TABLE
        .iter()
        .copied()
        .find(|r| n <= r.0)
        .unwrap_or((u32::MAX, false, 1));
    let (plugin, kernel) = if lds {
        (REPLAY_LDS_PLUGIN, format!("{REPLAY_LDS_PREFIX}{vs}"))
    } else {
        ("p27w2_gdn_replay", format!("{REPLAY_PREFIX}{vs}"))
    };
    Ok(LaunchDescriptor {
        plugin: plugin.into(),
        kernel,
        grid: [d.grid[0], d.grid[1], vs],
        block: [256 / vs, 1, 1],
        ..d.clone()
    })
}

/// Stash bytes a slot SET holds on one rank for one GDN layer: two regions a v head.
#[must_use]
pub fn replay_stash_bytes_per_set(heads: u32) -> u64 {
    u64::from(heads) * 2 * REPLAY_SX_HEAD
}

fn slot_i64(d: &LaunchDescriptor, name: &str) -> Result<i64> {
    match d.arg(name).map(|s| &s.value) {
        Some(ArgValue::I64 { ival }) | Some(ArgValue::I32 { ival }) => Ok(*ival),
        _ => Err(Error::Rule(format!(
            "gdn_replay: `{}` has no integer `{name}`",
            d.kernel
        ))),
    }
}

fn slot_f32(d: &LaunchDescriptor, name: &str) -> Result<f32> {
    match d.arg(name).map(|s| &s.value) {
        Some(ArgValue::F32 { fval }) => Ok(*fval),
        _ => Err(Error::Rule(format!(
            "gdn_replay: `{}` has no f32 `{name}`",
            d.kernel
        ))),
    }
}

/// **P27-W2-5: the replay form of a planned decode-pair recurrent update** (or, with
/// `materialize`, the materialize launch at the same shape, VS 1). Built FROM the shipped
/// descriptor so every scalar (strides, heads, scale, softplus threshold) and the grid are the
/// plan's own; only the symbol and the per-sequence tables change: `naccept` -> `rp`, plus the
/// stash pool and the refusal word. `sx_div` is the state slots a sequence's set holds.
///
/// # Errors
/// A descriptor that is not the shipped f16 / bf16-ab / norm-free recurrent update.
pub fn gdn_replay_from_recurrent(
    d: &LaunchDescriptor,
    sx_div: u32,
    materialize: bool,
) -> Result<LaunchDescriptor> {
    let Some(vs) = d.kernel.strip_prefix(REPLAY_FROM_PREFIX) else {
        return Err(Error::Rule(format!(
            "gdn_replay: `{}` is not the f16-state, bf16-ab, norm-free recurrent update",
            d.kernel
        )));
    };
    let vs: u32 = vs
        .parse()
        .map_err(|_| Error::Rule(format!("gdn_replay: `{}`: no VS suffix", d.kernel)))?;
    if !matches!(vs, 1 | 2 | 4) || d.grid[2] != vs {
        return Err(Error::Rule(format!(
            "gdn_replay: `{}` at grid z {} (VS 1 / 2 / 4 only)",
            d.kernel, d.grid[2]
        )));
    }
    if sx_div == 0 {
        return Err(Error::Rule("gdn_replay: a slot set of 0 slots".into()));
    }
    let (args, bytes) = KArgs::new(abi::SIG_P27W2_GDN_REPLAY)
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("av")
        .buf("bv")
        .i64(slot_i64(d, "ab_stride")?)
        .buf("a_log")
        .buf("dt_bias")
        .buf("state")
        .i64(slot_i64(d, "st_slot")?)
        .i64(slot_i64(d, "st_head")?)
        .buf("o")
        .buf("cu")
        .buf("sidx")
        .i64(slot_i64(d, "sidx_stride")?)
        .buf("rp")
        .buf("stash")
        .i64(i64::from(sx_div))
        .i64(REPLAY_SX_HEAD as i64)
        .buf("err")
        .i32(slot_i64(d, "h")?)
        .i32(slot_i64(d, "hg")?)
        .f32(slot_f32(d, "scale")?)
        .f32(slot_f32(d, "sp_thr")?)
        .finish()?;
    let (kernel, grid, block) = if materialize {
        (
            REPLAY_MATERIALIZE.to_owned(),
            [d.grid[0], d.grid[1], 1],
            [256, 1, 1],
        )
    } else {
        (format!("{REPLAY_PREFIX}{vs}"), d.grid, d.block)
    };
    Ok(LaunchDescriptor {
        op: if materialize {
            "gdn_replay_materialize".into()
        } else {
            "gdn_recurrent_update".into()
        },
        plugin: "p27w2_gdn_replay".into(),
        kernel,
        grid,
        block,
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// **CMP27G: the materialize twin of a REPLAY descriptor** (`p27w2_gdn_replay_*_vs*`, the plan's
/// own kernargs): the same signature and immediates, the `_materialize` symbol, one sequence
/// (grid x 1, the descriptor's heads, VS 1), 256 threads. Applies a base's pending replay with no
/// rows of its own (the tree verify reads the base right after it).
///
/// # Errors
/// A descriptor that is not the replay form.
pub fn gdn_replay_materialize_like(d: &LaunchDescriptor) -> Result<LaunchDescriptor> {
    if !d.kernel.starts_with(REPLAY_PREFIX) && !d.kernel.starts_with(REPLAY_LDS_PREFIX) {
        return Err(Error::Rule(format!(
            "gdn_replay_materialize: `{}` is not the replay form",
            d.kernel
        )));
    }
    Ok(LaunchDescriptor {
        op: "gdn_replay_materialize".into(),
        // the shipped object's MODE 1 entry, whichever object the verify launch lives in (CMP27T)
        plugin: "p27w2_gdn_replay".into(),
        kernel: REPLAY_MATERIALIZE.to_owned(),
        grid: [1, d.grid[1], 1],
        block: [256, 1, 1],
        ..d.clone()
    })
}

// ---------------------------------------------------------------------------- CMP27AH: the fused pair

/// CMP27AH: the batched decode pair's conv update symbol prefix the fused form is built from
/// (`_hpb{1,2,4,8}` follows; every hpb writes the same bytes: one wave a 128-channel unit).
pub const CUF_FROM_CONV_PREFIX: &str = "saber_gdn_conv_update_w4_h128_hpb";
/// CMP27AH: the fused pair (`hip/p27w2_gdn_cuf.hip`, `--engine-step-arms cuf`); `_vs{1,2,4}`
/// follows.
pub const CUF_PREFIX: &str = "p27w2_gdn_cuf_f16s_k128_v128_ab1_vs";
/// CMP27AH: the fused pair's code object (its own: the conv and both replay objects keep their
/// bytes).
pub const CUF_PLUGIN: &str = "p27w2_gdn_cuf";

/// **CMP27AH: the fused pair's VS by batch** -- `(max sequences, VS)`, first match wins; VS 0 = the
/// batch keeps the PAIR. MEASURED (card-1 probe p3, `tools/capture/cmp27ah/probe.py`, IC-cold, us a
/// layer, pair as served vs fused): N 2 T 8 27.59 vs 27.72 (VS 2) = tied -> pair; N 4 T 8 38.35 vs
/// 34.48 (VS 1), T 5 30.51 vs 25.51 -> fused VS 1; N 8 T 8 80.22 vs 83.84 (VS 2) / 85.33 (VS 1),
/// T 5 58.61 vs 61.01 -> pair (the k head's q / k conv runs in all 3 x VS workgroups and the
/// saturated b8 grid pays it). Every VS writes the pair's bytes (the oracle at VS 1 / 2 / 4).
/// CMP27AI re-priced the re-cut object (0 SGPR spills; card-1 probe `ws-cmp27ai/p1`, same sets):
/// N 2 T 8 26.63 vs 26.47 (VS 1) = tied; N 4 T 8 38.57 vs 34.07 (VS 1), T 5 30.12 vs 25.51;
/// N 8 T 8 80.26 vs 84.72 (VS 2), T 5 58.60 vs 60.86 -> the same table.
pub const CUF_VS_TABLE: [(u32, u32); 3] = [(2, 0), (4, 1), (u32::MAX, 0)];

/// [`CUF_VS_TABLE`]'s VS at `n` sequences, `None` = the pair stays.
#[must_use]
pub fn cuf_vs(n: u32) -> Option<u32> {
    CUF_VS_TABLE
        .iter()
        .find(|r| n <= r.0)
        .map(|r| r.1)
        .filter(|vs| *vs > 0)
}

/// **CMP27AH: the batched decode PAIR as ONE descriptor** -- the conv update (`conv`, the plan's
/// `saber_gdn_conv_update_*`) and the replay form of its recurrent update (`replay`, the shipped or
/// the LDS replay, already rewritten by [`gdn_replay_from_recurrent`] / [`gdn_replay_c4_form`]).
/// Every scalar is the plan's own (the conv's strides, `state_len_max`, `maxq`; the replay's);
/// the VS is [`CUF_VS_TABLE`]'s at the grid's sequences; `cnt` is the counter region (a new
/// binding). The conv's `cache_idx` must be the replay's `sidx` at the same stride (checked by the
/// caller on the bindings; the strides here).
///
/// # Errors
/// A descriptor that is not the pair's half it is named for, two halves of different grids or
/// heads, a conv `ci_stride` that is not the replay's `sidx_stride`.
pub fn gdn_cuf_from_pair(
    conv: &LaunchDescriptor,
    replay: &LaunchDescriptor,
) -> Result<LaunchDescriptor> {
    if !conv.kernel.starts_with(CUF_FROM_CONV_PREFIX) {
        return Err(Error::Rule(format!(
            "gdn_cuf: `{}` is not the w4 h128 conv update the fused pair is built from",
            conv.kernel
        )));
    }
    if !replay.kernel.starts_with(REPLAY_PREFIX) && !replay.kernel.starts_with(REPLAY_LDS_PREFIX) {
        return Err(Error::Rule(format!(
            "gdn_cuf: `{}` is not the replay form's verify launch",
            replay.kernel
        )));
    }
    let n = replay.grid[0];
    if conv.grid[0] != n {
        return Err(Error::Rule(format!(
            "gdn_cuf: the conv runs {} sequences and the replay {n}",
            conv.grid[0]
        )));
    }
    for (c, r) in [("h", "h"), ("hg", "hg"), ("ci_stride", "sidx_stride")] {
        if slot_i64(conv, c)? != slot_i64(replay, r)? {
            return Err(Error::Rule(format!(
                "gdn_cuf: the conv's `{c}` {} is not the replay's `{r}` {}",
                slot_i64(conv, c)?,
                slot_i64(replay, r)?
            )));
        }
    }
    let present = |d: &LaunchDescriptor, name: &str| -> Result<bool> {
        match d.arg(name).map(|s| &s.value) {
            Some(ArgValue::Buffer { .. }) => Ok(true),
            Some(ArgValue::NullBuffer { .. }) => Ok(false),
            _ => Err(Error::Rule(format!(
                "gdn_cuf: `{}` has no pointer `{name}`",
                d.kernel
            ))),
        }
    };
    let vs = cuf_vs(n).ok_or_else(|| {
        Error::Rule(format!(
            "gdn_cuf: {n} sequences keep the pair (CUF_VS_TABLE): not a fused batch"
        ))
    })?;
    let (args, bytes) = KArgs::new(abi::SIG_P27W2_GDN_CUF)
        .buf("x")
        .i64(slot_i64(conv, "xpitch")?)
        .buf("wgt")
        .maybe_buf("bias", present(conv, "bias")?)
        .buf("cstate")
        .i64(slot_i64(conv, "cs_seq")?)
        .i64(slot_i64(conv, "cs_dim")?)
        .i64(slot_i64(conv, "cs_tok")?)
        .i32(slot_i64(conv, "state_len_max")?)
        .maybe_buf("naccept", present(conv, "naccept")?)
        .i32(slot_i64(conv, "maxq")?)
        .buf("cnt")
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("av")
        .buf("bv")
        .i64(slot_i64(replay, "ab_stride")?)
        .buf("a_log")
        .buf("dt_bias")
        .buf("state")
        .i64(slot_i64(replay, "st_slot")?)
        .i64(slot_i64(replay, "st_head")?)
        .buf("o")
        .buf("cu")
        .buf("sidx")
        .i64(slot_i64(replay, "sidx_stride")?)
        .buf("rp")
        .buf("stash")
        .i64(slot_i64(replay, "sx_div")?)
        .i64(slot_i64(replay, "sx_head")?)
        .buf("err")
        .i32(slot_i64(replay, "h")?)
        .i32(slot_i64(replay, "hg")?)
        .f32(slot_f32(replay, "scale")?)
        .f32(slot_f32(replay, "sp_thr")?)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_cuf".into(),
        plugin: CUF_PLUGIN.into(),
        kernel: format!("{CUF_PREFIX}{vs}"),
        grid: [n, replay.grid[1], vs],
        block: [256 / vs, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

// ---------------------------------------------------------------------------- P27-15d: the tree form

/// The tree verify's rows a sequence: the SABER decode tile (`p27_tree_gdn`'s `TG_TM`).
pub const TREE_ROWS: u32 = 10;

/// Floats in one (sequence, v head) tree record (`TG_REC_FLOATS`: U [10][128] + k^ [10][128] +
/// P [16] + the ancestor words [16]).
pub const TREE_REC_FLOATS: u32 = 2 * TREE_ROWS * 128 + 32;

/// **E4-3**: the 16-row tree objects' rows a sequence (`p27_tree_gdn16`'s `TG_TM`, the 6-warp
/// `p27_tree_attn16`: 96 attention rows = 16 x GQA 6).
pub const TREE16_ROWS: u32 = 16;

/// E4-3: the tree record's floats at `rows` (10 -> [`TREE_REC_FLOATS`], 16 -> 4,128).
#[must_use]
pub const fn tree_rec_floats(rows: u32) -> u32 {
    2 * rows * 128 + 32
}

/// E4-3: the 10-row twin of a tree shape wider than 10 rows (the registry's ragged rows hold 10 a
/// sequence: the twin resolves the grid, the split law and the combine; the real shape's kernargs
/// replace its own).
fn tree_twin10(s: &AttnShape) -> AttnShape {
    let mut t = *s;
    t.q_len = TREE_ROWS;
    t.live_rows = TREE_ROWS;
    if t.ragged_rows > 0 {
        t.ragged_rows = t.num_seqs * TREE_ROWS;
    }
    t
}

/// **E4-3: the 16-row tree verify's attention combine**: the shipped ragged combine at the real
/// shape's rows (`ragged_rows x q_heads` jobs). At <= 10 rows it IS [`attn_decode_combine`].
///
/// # Errors
/// As [`attn_decode_combine`], or a tree past [`TREE16_ROWS`].
pub fn tree_attn_combine(s: AttnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    if s.q_len <= TREE_ROWS {
        return attn_decode_combine(s, rules);
    }
    if s.q_len > TREE16_ROWS || s.ragged_rows == 0 {
        return Err(Error::Rule(format!(
            "attn_decode_tree combine: {} rows a sequence (ragged {}); the 16-row form holds {TREE16_ROWS}",
            s.q_len, s.ragged_rows
        )));
    }
    let mut d = attn_decode_combine(tree_twin10(&s), rules)?;
    let splits = rules.decode_splits(s.max_ctx, s.kv_heads, s.num_seqs, s.splits);
    let (args, bytes) = attn_args(&s, splits, Some(rules.knobs.decode_tile))?;
    d.grid = [s.ragged_rows * s.q_heads, 1, 1];
    d.args = args;
    d.kernarg_bytes = bytes;
    Ok(d)
}

/// **P27-15d: the tree verify's attention head launch**: the ragged decode descriptor this shape
/// resolves to (C2's `saber_attn_decode_ragged_*`: grid, warps, split law, scratch), its symbol
/// swapped for the ancestor-word twin (`p27_attn_decode_anc_*`, the retired source generator
/// ANC_SUBS, in `p27_tree_attn.hsaco`) and the `anc` table appended after `cu_q`
/// ([`abi::SIG_P27_ATTN_DECODE_ANC`]). The combine is the shipped ragged one
/// ([`attn_decode_combine`] of the same shape). A chain's words ARE the causal test, so a chain
/// tree reads what the ragged kernel reads (P27-15 k2: 239 / 239 tiles byte-identical).
///
/// # Errors
/// A uniform (non-ragged) shape, or the ragged resolution's own refusals.
pub fn tree_attn_decode(s: AttnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    if s.q_len > TREE_ROWS {
        return tree16_attn_decode(s, rules);
    }
    if s.ragged_rows == 0 {
        return Err(Error::Rule(
            "attn_decode_tree: the ancestor-word kernel is the RAGGED body's twin; a uniform \
             shape has no `cu_q` to read"
                .into(),
        ));
    }
    let mut d = resolve(Op::AttnDecode, Shape::Attn(s), rules)?;
    let tail = d
        .kernel
        .strip_prefix("saber_attn_decode_ragged_")
        .ok_or_else(|| {
            Error::Rule(format!(
                "attn_decode_tree: the ragged shape resolved to `{}`, not a ragged decode symbol",
                d.kernel
            ))
        })?
        .to_owned();
    d.kernel = format!("p27_attn_decode_anc_{tail}");
    d.plugin = "p27_tree_attn".into();
    d.op = "attn_decode_tree".into();
    let (slots, size) = abi::pack(abi::SIG_P27_ATTN_DECODE_ANC);
    let at = abi::SIG_P27_ATTN_DECODE_ANC.len() - 1;
    d.args.push(Slot {
        name: "anc".into(),
        offset: slots[at].0,
        size: slots[at].1,
        value: ArgValue::Buffer {
            buffer: "anc".into(),
        },
    });
    d.kernarg_bytes = size;
    Ok(d)
}

/// **E4-3: the 16-row tree verify's attention head launch** (`p27_attn_decode_anc16_*_nw6`,
/// `p27_tree_attn16.hsaco`): the 10-row twin's descriptor (grid, split law, workspace) with the
/// 6-warp symbol, 192 threads, the real shape's kernargs (`q_len` 11..16) and the scratch at 16
/// rows.
///
/// # Errors
/// More than 16 rows, a uniform shape, or the twin's own refusals.
fn tree16_attn_decode(s: AttnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    if s.q_len > TREE16_ROWS || s.ragged_rows == 0 || s.live_rows != s.q_len {
        return Err(Error::Rule(format!(
            "attn_decode_tree: {} rows a sequence (live {}, ragged {}); the 16-row form holds \
             {TREE16_ROWS} ragged rows",
            s.q_len, s.live_rows, s.ragged_rows
        )));
    }
    let gqa = s.q_heads / s.kv_heads.max(1);
    if s.q_len * gqa > 96 {
        return Err(Error::Rule(format!(
            "attn_decode_tree: q_len {} x gqa {gqa} exceeds the 6-warp object's 96 rows",
            s.q_len
        )));
    }
    let mut d = tree_attn_decode(tree_twin10(&s), rules)?;
    let tail = d
        .kernel
        .strip_prefix("p27_attn_decode_anc_")
        .and_then(|t| t.rsplit_once("_nw").map(|(a, _)| a.to_owned()))
        .ok_or_else(|| Error::Rule(format!("attn_decode_tree: the twin is `{}`", d.kernel)))?;
    d.kernel = format!("p27_attn_decode_anc16_{tail}_nw6");
    d.plugin = "p27_tree_attn16".into();
    d.block = [6 * 32, 1, 1];
    let splits = rules.decode_splits(s.max_ctx, s.kv_heads, s.num_seqs, s.splits);
    let need = rules.decode_scratch_bytes(&s, splits);
    let ws = rules.workspace("attn_decode_scratch")?;
    if need > ws.capacity_bytes {
        return Err(Error::Rule(format!(
            "attn_decode_tree: {need} B of split-KV partials exceeds the {} B declared",
            ws.capacity_bytes
        )));
    }
    for w in &mut d.workspaces {
        if w.name == "attn_decode_scratch" {
            w.bytes = need;
        }
    }
    let (mut args, _) = attn_args(&s, splits, None)?;
    let (slots, size) = abi::pack(abi::SIG_P27_ATTN_DECODE_ANC);
    // the twin's trailing scalars / buffers past the SABERArgs block (splits, cu_q, anc) are kept
    let base = args.len();
    for a in d.args.iter().skip(base) {
        args.push(a.clone());
    }
    let _ = slots;
    d.args = args;
    d.kernarg_bytes = size;
    Ok(d)
}

/// **P27-15d: the tree verify's conv** (`p27_gdn_conv_tree_w4_h128_hpb*`, `p27_tree_conv.hsaco`):
/// [`gdn_conv_update`]'s geometry with `naccept` replaced by the committed window's cache entries
/// (`hsrc` [n][4]) and every row's older taps (`tsrc` [rows][4]); `cache_idx` is the per-SLOT table
/// (`ci_stride` 1). The walk is the bucket's (`maxq` = the tree's rows) over the graph's stride.
///
/// # Errors
/// A shape the shipped conv refuses, or no instantiation at the hpb the rules pick.
pub fn tree_gdn_conv(s: GdnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let reg = Registry::new(rules);
    let g = Geometry::new()
        .num("conv_width", s.conv_width as i64)
        .num("head_k", s.head_k as i64)
        .num("head_v", s.head_v as i64);
    reg.select("gdn_conv_update", &g).map_err(Error::Rule)?;
    heads_divide(s.h, s.hg, "gdn_conv_tree")?;
    if s.conv_width != 4 || s.head_k != 128 {
        return Err(Error::Rule(format!(
            "gdn_conv_tree: built for conv width 4 and head 128 (w{} h{})",
            s.conv_width, s.head_k
        )));
    }
    let hpb = rules.conv_hpb(s.h, s.hg, Some(s.n_seqs));
    if ![1, 2, 4, 8].contains(&hpb) {
        return Err(Error::Rule(format!(
            "gdn_conv_tree: no hpb={hpb} instantiation"
        )));
    }
    let units = 2 * s.hg + s.h;
    let (args, bytes) = KArgs::new(abi::SIG_P27_GDN_CONV_TREE)
        .buf("x")
        .i64(s.xpitch)
        .buf("wgt")
        .maybe_buf("bias", s.has_bias)
        .buf("cstate")
        .i64(s.cs_seq)
        .i64(s.cs_dim)
        .i64(s.cs_tok)
        .i32(((s.conv_width - 1) + (s.maxq - 1)) as i64)
        .buf("cache_idx")
        .i64(1)
        .buf("hsrc")
        .buf("tsrc")
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("cu")
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .i32(s.maxq as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_conv_tree".into(),
        plugin: "p27_tree_conv".into(),
        kernel: format!("p27_gdn_conv_tree_w4_h128_hpb{hpb}"),
        grid: [s.n_seqs, units / hpb, 1],
        block: [hpb * 32, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// **CMP27L ARM: the tree verify's conv with its rows in parallel**
/// (`p27_gdn_conv_tree_rows_w4_h128_t10`, `p27_tree_rows.hsaco`): [`tree_gdn_conv`]'s kernargs
/// byte for byte, one workgroup a 128-channel unit and one wave a tree row (10 waves; a longer
/// sequence strides by 10), the cache rewrite after a workgroup barrier. The serial kernel's bytes
/// (`tools/capture/cmp27l/probe.py`).
///
/// # Errors
/// As [`tree_gdn_conv`].
pub fn tree_gdn_conv_rows(s: GdnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let mut d = tree_gdn_conv(s, rules)?;
    d.plugin = "p27_tree_rows".into();
    d.kernel = TREE_CONV_ROWS_SYMBOL.into();
    d.grid = [s.n_seqs, 2 * s.hg + s.h, 1];
    d.block = [TREE_CONV_ROWS_WAVES * 32, 1, 1];
    Ok(d)
}

/// CMP27L: the rows-parallel tree conv's symbol and its waves a workgroup.
pub const TREE_CONV_ROWS_SYMBOL: &str = "p27_gdn_conv_tree_rows_w4_h128_t10";
/// CMP27L: one wave a tree row.
pub const TREE_CONV_ROWS_WAVES: u32 = 10;

/// **P27-15d: the tree verify's GDN update** (`p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs1_adv1`,
/// `p27_tree_gdn.hsaco`): one workgroup a (sequence, v head), 256 threads, the committed f16 slot
/// from `tslot` [n], the ancestor words `anc` [rows], the previous tree's accepted row `acc` [n]
/// advanced from `ts_in` as the prologue, this tree's record into `ts_out` (strides in floats:
/// `ts_seq = h x` [`TREE_REC_FLOATS`], `ts_head` = one record). vs1 adv1 is the form P27-15 timed
/// (vs4 loses); the gated row norm (`norm1`) is not taken -- the shipped `gdn_norm_quant` follows.
///
/// # Errors
/// An fp32-state graph (the tree is f16-state only), or a geometry the kernel is not built for.
pub fn tree_gdn_update(s: GdnShape, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let reg = Registry::new(rules);
    let g = Geometry::new()
        .num("head_k", s.head_k as i64)
        .num("head_v", s.head_v as i64);
    reg.select("gdn_recurrent_update", &g)
        .map_err(Error::Rule)?;
    heads_divide(s.h, s.hg, "gdn_tree_update")?;
    if !s.state_f16 || s.head_k != 128 || s.head_v != 128 {
        return Err(Error::Rule(format!(
            "gdn_tree_update: built for the f16 stored state at k128 v128 (f16 {} k{} v{})",
            s.state_f16, s.head_k, s.head_v
        )));
    }
    if s.maxq > TREE16_ROWS {
        return Err(Error::Rule(format!(
            "gdn_tree_update: {} rows a sequence; the widest tree tile is {TREE16_ROWS}",
            s.maxq
        )));
    }
    // E4-3: past 10 rows the 16-row object (`p27_tree_gdn16`: TG_TM 16, its own record stride)
    let wide = s.maxq > TREE_ROWS;
    let rec = i64::from(tree_rec_floats(if wide { TREE16_ROWS } else { TREE_ROWS }));
    let (args, bytes) = KArgs::new(abi::SIG_P27_GDN_TREE)
        .buf("q")
        .buf("k")
        .buf("v")
        .buf("av")
        .buf("bv")
        .i64(s.ab_stride)
        .buf("a_log")
        .buf("dt_bias")
        .buf("state")
        .i64(s.st_slot)
        .i64(s.st_head)
        .buf("slot")
        .buf("o")
        .buf("cu")
        .buf("anc")
        .buf("ts_in")
        .buf("ts_out")
        .i64(i64::from(s.h) * rec)
        .i64(rec)
        .buf("acc")
        .null_buf("zg")
        .null_buf("nw")
        .f32(0.0)
        .i32(0)
        .i32(s.h as i64)
        .i32(s.hg as i64)
        .f32(f32::from_bits(s.scale_bits))
        .f32(f32::from_bits(s.softplus_thr_bits))
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_tree_update".into(),
        plugin: if wide {
            "p27_tree_gdn16"
        } else {
            "p27_tree_gdn"
        }
        .into(),
        kernel: if wide {
            "p27_gdn_tree16_f16s_k128_v128_ab1_norm0_vs1_adv1"
        } else {
            "p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs1_adv1"
        }
        .into(),
        grid: [s.n_seqs, s.h, 1],
        block: [256, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// **P27-15d: the pending advance as its own launch** (the FLUSH form,
/// `p27_gdn_tree_advance_f16s_k128_v128`): `acc` [n]'s state from the record `ts` into
/// `slot_out` [n] (may equal `slot` [n]). Shape as [`tree_gdn_update`].
///
/// # Errors
/// As [`tree_gdn_update`].
pub fn tree_gdn_advance(s: GdnShape) -> Result<LaunchDescriptor> {
    if !s.state_f16 || s.head_k != 128 || s.head_v != 128 {
        return Err(Error::Rule(
            "gdn_tree_advance: built for the f16 stored state at k128 v128".into(),
        ));
    }
    if s.maxq > TREE16_ROWS {
        return Err(Error::Rule(
            "gdn_tree_advance: query span exceeds the 16-row record".into(),
        ));
    }
    let wide = s.maxq > TREE_ROWS;
    let rec = i64::from(tree_rec_floats(if wide { TREE16_ROWS } else { TREE_ROWS }));
    let (args, bytes) = KArgs::new(abi::SIG_P27_GDN_TREE_ADVANCE)
        .buf("state")
        .i64(s.st_slot)
        .i64(s.st_head)
        .buf("slot")
        .buf("slot_out")
        .buf("ts")
        .i64(i64::from(s.h) * rec)
        .i64(rec)
        .buf("acc")
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_tree_advance".into(),
        plugin: if wide {
            "p27_tree_gdn16"
        } else {
            "p27_tree_gdn"
        }
        .into(),
        kernel: if wide {
            "p27_gdn_tree16_advance_f16s_k128_v128"
        } else {
            "p27_gdn_tree_advance_f16s_k128_v128"
        }
        .into(),
        grid: [s.n_seqs, s.h, 1],
        block: [256, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// **P27-15d: the accepted path's K/V compaction** (`p27_tree_kv_compact`, `p27_tree_kv.hsaco`):
/// grid (attention layers, kv heads), 64 threads; `layer_base` [layers] u64 pool bases, `moves`
/// [n][2] absolute token slots applied IN ORDER, `n_moves` by value (a captured form pads to a
/// fixed count with self-moves).
#[must_use]
pub fn tree_kv_compact(
    layers: u32,
    kv_heads: u32,
    n_moves: u32,
    block_stride: i64,
    head_stride: i64,
    row_bytes: u32,
    block_size: u32,
) -> LaunchDescriptor {
    let (args, bytes) = KArgs::new(abi::SIG_P27_TREE_KV_COMPACT)
        .buf("layer_base")
        .buf("moves")
        .i32(i64::from(n_moves))
        .i64(block_stride)
        .i64(head_stride)
        .i32(i64::from(row_bytes))
        .i32(i64::from(block_size))
        .finish()
        .unwrap_or_default();
    LaunchDescriptor {
        op: "tree_kv_compact".into(),
        plugin: "p27_tree_kv".into(),
        kernel: "p27_tree_kv_compact".into(),
        grid: [layers, kv_heads, 1],
        block: [64, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    }
}

/// **P27-W2-4 (R2): the accepted conv window moved to the chain's entries**
/// (`p27_tree_conv_commit`, `p27_tree_flush.hsaco`): grid (GDN layers, ceil(dims / 256)), 256
/// threads; `layer_base` [layers] u64 conv pool bases, `moves` [n][2] entry indices applied in
/// order, `n_moves` / `seq_off` by value.
#[must_use]
pub fn tree_conv_commit(
    layers: u32,
    dims: u32,
    n_moves: u32,
    seq_off: i64,
    cs_dim: i64,
    cs_tok: i64,
) -> LaunchDescriptor {
    let (args, bytes) = KArgs::new(abi::SIG_P27_TREE_CONV_COMMIT)
        .buf("layer_base")
        .buf("moves")
        .i32(i64::from(n_moves))
        .i64(seq_off)
        .i64(cs_dim)
        .i64(cs_tok)
        .i32(i64::from(dims))
        .finish()
        .unwrap_or_default();
    LaunchDescriptor {
        op: "tree_conv_commit".into(),
        plugin: "p27_tree_flush".into(),
        kernel: "p27_tree_conv_commit".into(),
        grid: [layers, dims.div_ceil(256), 1],
        block: [256, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    }
}

/// **CMP27L ARM: the flush's advance for every GDN layer in ONE launch**
/// (`p27_gdn_tree_advance_layers_f16s_k128_v128`, `p27_tree_advl.hsaco`): grid (`layers`, heads,
/// 1), `state_base` / `ts_base` = [layers] u64 tables of each layer's state pool and record, the
/// strides of the per-layer advance `one` ([`tree_gdn_advance_like`]'s; every layer's must agree,
/// the caller checks); the per-layer launch's bytes.
///
/// # Errors
/// `one` is not the per-layer advance, or carries no stride.
pub fn tree_gdn_advance_layers(one: &LaunchDescriptor, layers: u32) -> Result<LaunchDescriptor> {
    if one.kernel != "p27_gdn_tree_advance_f16s_k128_v128" {
        return Err(Error::Rule(format!(
            "gdn_tree_advance_layers: built from the per-layer advance, not `{}`",
            one.kernel
        )));
    }
    let imm = |name: &str| -> Result<i64> {
        match one.arg(name).map(|s| &s.value) {
            Some(ArgValue::I64 { ival } | ArgValue::I32 { ival }) => Ok(*ival),
            _ => Err(Error::Rule(format!("gdn_tree_advance_layers: no `{name}`"))),
        }
    };
    let (args, bytes) = KArgs::new(abi::SIG_P27_GDN_TREE_ADVANCE_LAYERS)
        .buf("state_base")
        .buf("ts_base")
        .i64(imm("st_slot")?)
        .i64(imm("st_head")?)
        .buf("slot")
        .buf("slot_out")
        .i64(imm("ts_seq")?)
        .i64(imm("ts_head")?)
        .buf("acc")
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_tree_advance_layers".into(),
        plugin: "p27_tree_advl".into(),
        kernel: "p27_gdn_tree_advance_layers_f16s_k128_v128".into(),
        grid: [layers, one.grid[1], 1],
        block: [256, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// **P27-W2-4 (R2): the standalone advance at a tree update's own strides** -- the flush form of
/// [`tree_gdn_advance`] built from the planned `p27_gdn_tree_*` descriptor (its `st_slot`,
/// `st_head`, `ts_seq`, `ts_head` and head count), one sequence.
///
/// # Errors
/// An update descriptor without those immediates, or not the f16 k128 v128 form.
pub fn tree_gdn_advance_like(update: &LaunchDescriptor) -> Result<LaunchDescriptor> {
    let imm = |name: &str| -> Result<i64> {
        match update.arg(name).map(|s| &s.value) {
            Some(ArgValue::I64 { ival } | ArgValue::I32 { ival }) => Ok(*ival),
            _ => Err(Error::Rule(format!(
                "gdn_tree_advance: the update `{}` carries no `{name}`",
                update.kernel
            ))),
        }
    };
    let (plugin, kernel) = if update.kernel.starts_with("p27_gdn_tree16_f16s_k128_v128") {
        ("p27_tree_gdn16", "p27_gdn_tree16_advance_f16s_k128_v128")
    } else if update.kernel.starts_with("p27_gdn_tree_f16s_k128_v128") {
        ("p27_tree_gdn", "p27_gdn_tree_advance_f16s_k128_v128")
    } else {
        return Err(Error::Rule(format!(
            "gdn_tree_advance: built for the f16 stored state at k128 v128, not `{}`",
            update.kernel
        )));
    };
    let (args, bytes) = KArgs::new(abi::SIG_P27_GDN_TREE_ADVANCE)
        .buf("state")
        .i64(imm("st_slot")?)
        .i64(imm("st_head")?)
        .buf("slot")
        .buf("slot_out")
        .buf("ts")
        .i64(imm("ts_seq")?)
        .i64(imm("ts_head")?)
        .buf("acc")
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_tree_advance".into(),
        plugin: plugin.into(),
        kernel: kernel.into(),
        grid: [1, update.grid[1], 1],
        block: [256, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn gdn_norm(
    s: NormShape,
    act: u32,
    reg: &Registry<'_>,
    rules: &SaberRules,
) -> Result<LaunchDescriptor> {
    let g = Geometry::new().num("channels", s.channels as i64);
    reg.select("gdn_gated_rmsnorm", &g).map_err(Error::Rule)?;
    let gd = &rules.gdn;
    // One ushort4 per lane on each of x, z and o, and one float4 on w: every row pointer has to
    // be 8-byte aligned in elements. A per-call condition, so it lives here, not in the registry.
    for (name, st) in [("xrow", s.xrow), ("zrow", s.zrow), ("orow", s.orow)] {
        if st % 4 != 0 {
            return Err(Error::Rule(format!(
                "gdn_gated_rmsnorm: {name}={st} must be a multiple of 4 elements (the loads and \
                 the store are one 8-byte access per lane)"
            )));
        }
    }
    let sym = rules
        .gdn_norm
        .iter()
        .find(|r| r.act == act)
        .ok_or_else(|| Error::Rule(format!("no gdn_norm row for act={act}")))?;
    let blocks = s.rows.div_ceil(gd.norm_rows_per_block as u64);
    let blocks = u32::try_from(blocks).map_err(|_| {
        Error::Rule(format!(
            "gdn_gated_rmsnorm: {} rows overflow a grid",
            s.rows
        ))
    })?;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_GDN_NORM)
        .buf("x")
        .buf("z")
        .buf("w")
        .buf("o")
        .i64(s.rows as i64)
        .i64(s.xrow)
        .i64(s.zrow)
        .i64(s.orow)
        .f32(f32::from_bits(s.eps_bits))
        .finish()?;
    Ok(LaunchDescriptor {
        op: "gdn_gated_rmsnorm".into(),
        plugin: plugin_for(&sym.symbol)?.into(),
        kernel: sym.symbol.clone(),
        grid: [blocks, 1, 1],
        block: [gd.norm_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// The per-call conditions `should_custom_ar` checks before the reference kernel is reached at all: anything
/// that fails them goes to RCCL, so a refusal here is a ROUTE, not an error.
fn ar_message_ok(s: &ArShape, rules: &SaberRules, what: &str) -> Result<u64> {
    let a = &rules.ar;
    let nbytes = s.n_elem * s.elem_bytes;
    if nbytes == 0 || nbytes % 16 != 0 {
        return Err(Error::Rule(format!(
            "{what}: {nbytes} B is not a non-zero multiple of 16 (the push is uint4)"
        )));
    }
    if nbytes > a.max_bytes {
        return Err(Error::Rule(format!(
            "{what}: {nbytes} B exceeds the {} B IPC scratch slot; a larger message falls back \
             to RCCL",
            a.max_bytes
        )));
    }
    Ok(nbytes)
}

/// The workspaces both all-reduce families share. `ar_seq` is allocated locally and still carries
/// the peer-visible kind: `slot = seq & 1` picks which half of the double-buffered scratch BOTH
/// ranks use, so its value is cross-rank protocol state.
fn ar_workspaces(rules: &SaberRules, with_loc_pack: bool) -> Result<Vec<WorkspaceReq>> {
    let mut out = Vec::new();
    let mut names: Vec<(&str, &str)> = vec![
        ("ar_scratch", "ar_scratch"),
        ("ar_peer_scratch", "ar_scratch"),
        ("ar_flags", "ar_flags"),
        ("ar_peer_flags", "ar_flags"),
        ("ar_seq", "ar_seq"),
    ];
    if with_loc_pack {
        names.push(("ar_loc_pack", "ar_loc_pack"));
    }
    for (name, rule) in names {
        let w = rules.workspace(rule)?;
        // T4: under the race check `ar_seq` also carries the `_chk` twins' diagnostic words past
        // its per-block counters; the served size (the rules file's) is untouched otherwise, so a
        // run without the check allocates exactly what it always did.
        let bytes = if name == "ar_seq" && rules.ar.race_check {
            w.capacity_bytes.max(AR_CHECK_SEQ_BYTES)
        } else {
            w.capacity_bytes
        };
        out.push(WorkspaceReq {
            name: name.into(),
            kind: w.kind,
            bytes,
            zero_once: w.kind == WorkspaceKind::IpcPeerVisible,
        });
    }
    Ok(out)
}

// ------------------------------------------------------------------------ the split form (P27-1d)

/// The split rotated all-reduce's word in `ar_flags` (and the arrival counter's in `ar_seq`):
/// `SPLIT_FLAG_WORD` / `SPLIT_CNT_WORD` in `hip/saber_ar_wht_split.hip`. The one-shot family uses at
/// most `max_nb` (96) of the 512 words, so a split call and a one-shot call share one region.
pub const AR_SPLIT_FLAG_WORD: u64 = 511;

/// The push's threads per block (P27-1c's pick: N x 1,024 threads).
pub const AR_SPLIT_PUSH_THREADS: u32 = 1024;

/// Whether `desc` is the one-shot rotated reduce the split form is defined against.
fn split_source(desc: &LaunchDescriptor) -> Result<(&'static str, &'static str)> {
    match desc.kernel.as_str() {
        // CMP27S: the wave-parallel body is the same one-shot (same args, chunk ownership and
        // slot law), so its split form is the served one's.
        "saber_ar_oneshot_2rank_wht6_bf16" | "saber_ar_wpar_2rank_wht6_bf16" => Ok((
            "saber_ar_split_2rank_wht6_bf16_push_nb",
            "saber_ar_split_2rank_wht6_bf16_reduce",
        )),
        other => Err(Error::Rule(format!(
            "--ar-form split: `{other}` has no split twin (only the served wht6 bf16 one-shot \
             does; the exact family and the race-check twins stay one-shot)"
        ))),
    }
}

/// **P27-1d: the SPLIT form of one resolved one-shot rotated reduce** -- the PUSH (P27-1c's
/// fewer-blocks kernel: `push_blocks` physical blocks of [`AR_SPLIT_PUSH_THREADS`] walking the
/// one-shot's `nbv = grid.x` virtual blocks) and the REDUCE (P27-1b's half, at the one-shot's own
/// grid and block), the two joined on the host by a command-processor wait on this rank's
/// `ar_flags[AR_SPLIT_FLAG_WORD] >= flag_val` (the peer's last push block raises it).
///
/// Derived, never resolved: every argument the one-shot binds is carried over by name and value
/// (the same buffers, `n_groups`, `n_chunks`, slot stride, scale offset, drain, acq), so the
/// per-chunk arithmetic, the chunk ownership, the slot parity and the counters are the one-shot's
/// -- the byte law P27-1c measured (19/19 sizes, both ranks, 0 elements). `flag_val` is bound 0
/// here and set per call by the executor (monotone per rank, the same sequence on both ranks).
/// Both descriptors declare one launch; the caller budgets the pair.
///
/// # Errors
/// A one-shot that is not the served wht6 bf16 symbol, `push_blocks` outside `1..=nbv`, or a
/// descriptor missing one of the one-shot's arguments.
pub fn ar_wht_split(
    oneshot: &LaunchDescriptor,
    push_blocks: u32,
) -> Result<(LaunchDescriptor, LaunchDescriptor)> {
    let (push_sym, reduce_sym) = split_source(oneshot)?;
    let nbv = oneshot.grid[0];
    if push_blocks == 0 || push_blocks > nbv {
        return Err(Error::Rule(format!(
            "--ar-push-blocks {push_blocks}: the push walks the one-shot's {nbv} virtual blocks \
             with 1..={nbv} physical ones"
        )));
    }
    let value = |name: &str| -> Result<ArgValue> {
        oneshot
            .arg(name)
            .map(|s| s.value.clone())
            .ok_or_else(|| Error::Rule(format!("{}: no `{name}` to carry", oneshot.kernel)))
    };
    let carried = |sig: &'static [SigArg]| -> Result<KArgs> {
        let mut k = KArgs::new(sig);
        for a in abi::SIG_SABER_AR_WHT {
            k.values.push(value(a.name)?);
        }
        Ok(k)
    };
    let (push_args, push_bytes) = carried(abi::SIG_SABER_AR_SPLIT_PUSH_NB)?
        .i32(0)
        .i32(i64::from(nbv))
        .finish()?;
    let (red_args, red_bytes) = carried(abi::SIG_SABER_AR_WHT)?.finish()?;
    let plugin = plugin_for(push_sym)?;
    let push = LaunchDescriptor {
        op: oneshot.op.clone(),
        plugin: plugin.into(),
        kernel: push_sym.into(),
        grid: [push_blocks, 1, 1],
        block: [AR_SPLIT_PUSH_THREADS, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: push_bytes,
        args: push_args,
        workspaces: oneshot.workspaces.clone(),
        n_launches: 1,
    };
    let reduce = LaunchDescriptor {
        op: oneshot.op.clone(),
        plugin: plugin_for(reduce_sym)?.into(),
        kernel: reduce_sym.into(),
        grid: oneshot.grid,
        block: oneshot.block,
        shared_bytes: 0,
        kernarg_bytes: red_bytes,
        args: red_args,
        workspaces: oneshot.workspaces.clone(),
        n_launches: 1,
    };
    Ok((push, reduce))
}

// ------------------------------------------------------------------------ the row bands

/// The two buffer names ONE all-reduce dispatch binds.
///
/// Band 0 of every message keeps `in` and `out`, so a message that fits the slot in one piece —
/// every decode step, every prefill chunk the engine runs — resolves to exactly the descriptor it
/// always did, down to the kernarg image and the cross-rank digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArNames<'a> {
    /// The payload this dispatch reduces.
    pub input: &'a str,
    /// Where it writes the sum. The reduce is in place, so the caller binds one address to both.
    pub output: &'a str,
}

impl ArNames<'static> {
    /// What an unbanded message binds, and what band 0 of a banded one binds.
    pub const DEFAULT: ArNames<'static> = ArNames {
        input: "in",
        output: "out",
    };
}

/// One row band of a chunked all-reduce: `rows` rows starting at `first_row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArBand {
    /// The band's first row in the message.
    pub first_row: u64,
    /// How many rows it carries.
    pub rows: u64,
}

/// The buffer names band `i` binds, as [`ar_bands`] numbers them.
///
/// Band 0 is `in`/`out` deliberately (see [`ArNames::DEFAULT`]); the rest are suffixed, because a
/// descriptor sequence over ONE `ArgBindings` needs one name per address and the planner binds
/// each band at its own offset into the same buffer.
#[must_use]
pub fn ar_band_names(i: usize) -> (String, String) {
    if i == 0 {
        ("in".to_owned(), "out".to_owned())
    } else {
        (format!("in.b{i}"), format!("out.b{i}"))
    }
}

/// **The chunking law**: the row bands a `rows x elems_per_row` message is cut into so that every
/// band fits the peer-visible IPC scratch slot.
///
/// The reduce is ELEMENTWISE, so a `tokens x hidden` message is the concatenation of independent
/// row bands and reducing them one after another is the same arithmetic as reducing the whole —
/// there is no cross-row term to lose. The bands are BALANCED (`ceil(rows / bands)` each, the last
/// one short) rather than greedy: a greedy cut leaves a remainder band that can fall below
/// `quant_min_bytes` and take the other family, and two families in one message is two roundings.
///
/// It is a pure function of the shape and the slot, which is what lets it live inside a captured
/// graph: the band count is known at plan time from the bucket's rows, and both ranks derive the
/// same sequence from the same plan (which is what the one-shot kernel's per-block sequence
/// counters require — `hip/saber_ar_exact.hip`, "both ranks issue identical call sequences, grid
/// size included").
///
/// A message that fits the slot whole returns ONE band, so nothing below 8,601 rows at hidden 5120
/// moves at all.
///
/// # Errors
/// A single row wider than the slot: the reduce cannot be cut below one row, so that message is
/// the one that really does fall back to RCCL and this refuses it by name rather than silently
/// splitting a row across two pushes.
pub fn ar_bands(
    rows: u64,
    elems_per_row: u64,
    elem_bytes: u64,
    max_bytes: u64,
) -> Result<Vec<ArBand>> {
    let row_bytes = elems_per_row * elem_bytes;
    if rows == 0 || row_bytes == 0 {
        return Err(Error::Rule(format!(
            "allreduce: a {rows}-row message of {row_bytes} B rows is not a message"
        )));
    }
    if max_bytes == 0 || row_bytes > max_bytes {
        return Err(Error::Rule(format!(
            "allreduce: one row is {row_bytes} B and the IPC scratch slot is {max_bytes} B; the \
             reduce is elementwise over rows and cannot be cut below one, so a message this wide \
             per row falls back to RCCL"
        )));
    }
    // `max_rows` is an integer, so `ceil(rows / bands) <= max_rows` holds and every band fits.
    // The remainder is spread ONE ROW AT A TIME over the leading bands rather than left on the
    // last one: a short tail band is the thing that can fall below `quant_min_bytes` while its
    // siblings sit above it, and two families in one message is two roundings.
    let max_rows = max_bytes / row_bytes;
    let bands = rows.div_ceil(max_rows);
    let base = rows / bands;
    let rem = rows % bands;
    let mut out = Vec::with_capacity(bands as usize);
    let mut first = 0u64;
    for i in 0..bands {
        let n = base + u64::from(i < rem);
        out.push(ArBand {
            first_row: first,
            rows: n,
        });
        first += n;
    }
    debug_assert_eq!(first, rows);
    Ok(out)
}

/// How many dispatches [`ar_bands`] says this message takes.
///
/// # Errors
/// [`ar_bands`]'s.
pub fn ar_band_count(
    rows: u64,
    elems_per_row: u64,
    elem_bytes: u64,
    max_bytes: u64,
) -> Result<u32> {
    let n = ar_bands(rows, elems_per_row, elem_bytes, max_bytes)?.len();
    u32::try_from(n)
        .map_err(|_| Error::Rule(format!("allreduce: {n} bands overflow a launch count")))
}

/// The family rule (`should_custom_ar` + `_quant_ok`) applied to one band.
fn ar_family(
    s: ArShape,
    dtype: ArPayload,
    reg: &Registry<'_>,
    rules: &SaberRules,
    names: ArNames<'_>,
) -> Result<LaunchDescriptor> {
    if rules.ar.external_rccl || rules.ar.host_mapped {
        return ar_exact(s, dtype, reg, rules, names);
    }
    // The size decides the `exact` key; the registry decides the kernel. Same split the reference draws,
    // same two-step rule ref_allreduce.py applies per call.
    match dtype
        .lossy()
        .filter(|_| rules.ar_wants_lossy(&s, ArDtype::Bf16))
    {
        Some(d) => ar_wht(s, rules.ar.qbits, d, reg, rules, names),
        None => ar_exact(s, dtype, reg, rules, names),
    }
}

/// One `rows x elems_per_row` all-reduce as the SEQUENCE of dispatches it takes.
///
/// The head descriptor declares the whole op's launch count (structural rule 7), exactly the way
/// `attn_decode` declares 2 for the split-KV pair; the rest declare 1 each. A one-band message
/// returns a single descriptor with `n_launches = 1` — the descriptor this op has always had.
///
/// # Errors
/// [`ar_bands`]'s, or any band's own resolve.
pub fn resolve_ar_banded(
    rows: u64,
    elems_per_row: u64,
    elem_bytes: u64,
    dtype: ArPayload,
    world_size: u32,
    rules: &SaberRules,
    reg: &Registry<'_>,
) -> Result<Vec<LaunchDescriptor>> {
    let bands = ar_bands(rows, elems_per_row, elem_bytes, rules.ar.max_bytes)?;
    let mut out = Vec::with_capacity(bands.len());
    for (i, band) in bands.iter().enumerate() {
        let (input, output) = ar_band_names(i);
        let desc = ar_family(
            ArShape {
                n_elem: band.rows * elems_per_row,
                elem_bytes,
                world_size,
            },
            dtype,
            reg,
            rules,
            ArNames {
                input: &input,
                output: &output,
            },
        )?;
        out.push(desc);
    }
    let n = u32::try_from(out.len())
        .map_err(|_| Error::Rule("allreduce: too many bands for a launch count".to_owned()))?;
    if let Some(head) = out.first_mut() {
        head.n_launches = n;
    }
    Ok(out)
}

fn ar_exact(
    s: ArShape,
    dtype: ArPayload,
    reg: &Registry<'_>,
    rules: &SaberRules,
    names: ArNames<'_>,
) -> Result<LaunchDescriptor> {
    if uses_rccl(rules, s.n_elem.saturating_mul(s.elem_bytes)) {
        if dtype.name() != "bf16" || s.elem_bytes != 2 || s.world_size != 2 {
            return Err(Error::Rule(
                "RCCL compatibility currently requires TP=2 BF16 collectives".into(),
            ));
        }
        ar_message_ok(&s, rules, "allreduce")?;
        let count = i64::try_from(s.n_elem)
            .map_err(|_| Error::Rule("RCCL element count overflow".into()))?;
        let (args, bytes) = KArgs::new(abi::SIG_RCCL_BF16)
            .buf(names.input)
            .buf(names.output)
            .i32(count)
            .finish()?;
        return Ok(LaunchDescriptor {
            op: "allreduce_rccl_bf16".into(),
            plugin: RCCL_PLUGIN.into(),
            kernel: RCCL_BF16_KERNEL.into(),
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: bytes,
            args,
            workspaces: Vec::new(),
            n_launches: 1,
        });
    }
    let a = &rules.ar;
    let g = Geometry::new()
        .num("world_size", s.world_size as i64)
        .num("exact", 1)
        .dtype("dtype", dtype.name());
    let row = reg.select("allreduce", &g).map_err(Error::Rule)?;
    let sym = rules
        .ar_exact
        .iter()
        .find(|r| r.dtype == dtype.name())
        .ok_or_else(|| Error::Rule(format!("{}: no {} instantiation", row.entry, dtype.name())))?;
    let nbytes = ar_message_ok(&s, rules, "allreduce")?;
    // An int4 packs `lanes` elements; the kernel walks 16-byte words, so this is the element
    // count the payload width implies and it has to be the one the row states.
    if s.elem_bytes * sym.lanes as u64 != 16 {
        return Err(Error::Rule(format!(
            "allreduce: a {}-byte payload does not pack {} to an int4",
            s.elem_bytes, sym.lanes
        )));
    }
    let n16 = nbytes / 16;
    let nb = rules.ar_blocks_exact(n16);
    let (args, bytes) = KArgs::new(abi::SIG_SABER_AR_EXACT)
        .buf(names.input)
        .buf("ar_peer_scratch")
        .buf("ar_scratch")
        .buf(names.output)
        .buf("ar_peer_flags")
        .buf("ar_flags")
        .buf("ar_seq")
        .i32(n16 as i64)
        // The slot stride is in 16-byte WORDS here and in BYTES in the lossy kernel. Two kernels,
        // two units, both from the reference.
        .i32((a.max_bytes / 16) as i64)
        .i32(a.drain)
        .i32(a.acq)
        .finish()?;
    let kernel = ar_kernel(rules, &sym.symbol, nbytes)?;
    Ok(LaunchDescriptor {
        op: "allreduce_exact".into(),
        plugin: collective_plugin(&kernel, rules)?.into(),
        kernel,
        grid: [nb, 1, 1],
        block: [rules.ar_threads_exact(), 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        // No loc_pack: the exact kernel pushes raw payload and folds the peer's copy in place.
        workspaces: ar_workspaces(rules, false)?,
        n_launches: 1,
    })
}

fn ar_exact_nq(s: ArNqShape, reg: &Registry<'_>, rules: &SaberRules) -> Result<LaunchDescriptor> {
    let a = &rules.ar;
    let nq = &rules.ar_nq;
    let g = Geometry::new()
        .num("world_size", s.world_size as i64)
        .num("exact", 1)
        .dtype("dtype", "bf16");
    let row = reg.select("allreduce_nq", &g).map_err(Error::Rule)?;
    let sym = rules
        .ar_exact_nq
        .iter()
        .find(|r| r.dtype == "bf16")
        .ok_or_else(|| Error::Rule(format!("{}: no bf16 instantiation", row.entry)))?;
    // The three conditions the reference entry point throws on, in its order.
    if s.k_cols % nq.k_multiple != 0 {
        return Err(Error::Rule(format!(
            "ar_exact_nq: K must be a multiple of {}",
            nq.k_multiple
        )));
    }
    if s.k_cols > nq.max_k {
        return Err(Error::Rule(
            "ar_exact_nq: K too large for the register budget".into(),
        ));
    }
    if s.m_rows > a.max_blocks {
        return Err(Error::Rule("ar_exact_nq: too many rows".into()));
    }
    let msg = ArShape {
        n_elem: s.m_rows as u64 * s.k_cols as u64,
        elem_bytes: 2,
        world_size: s.world_size,
    };
    ar_message_ok(&msg, rules, "ar_exact_nq")?;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_AR_EXACT_NQ)
        .buf("in")
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
        .i32((s.k_cols / nq.k_multiple) as i64)
        .i32((a.max_bytes / 16) as i64)
        .f32(f32::from_bits(s.eps_bits))
        .i32(a.drain)
        .i32(a.acq)
        .finish()?;
    let kernel = ar_kernel(rules, &sym.symbol, msg.n_elem * msg.elem_bytes)?;
    Ok(LaunchDescriptor {
        op: "allreduce_exact_nq".into(),
        plugin: collective_plugin(&kernel, rules)?.into(),
        kernel,
        // ONE BLOCK PER ROW: the grid is the row count, not a block-count heuristic, because the
        // epilogue's RMS needs the whole row in one workgroup.
        grid: [s.m_rows, 1, 1],
        block: [nq.block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: ar_workspaces(rules, false)?,
        n_launches: 1,
    })
}

fn ar_wht(
    s: ArShape,
    bits: u32,
    dtype: ArDtype,
    reg: &Registry<'_>,
    rules: &SaberRules,
    names: ArNames<'_>,
) -> Result<LaunchDescriptor> {
    let a = &rules.ar;
    let g = Geometry::new()
        .num("world_size", s.world_size as i64)
        .num("exact", 0)
        .num("bits", bits as i64)
        .dtype("dtype", dtype.name())
        .num("numel", s.n_elem as i64);
    let row = reg.select("allreduce", &g).map_err(Error::Rule)?;
    let sym = rules
        .ar_wht
        .iter()
        .find(|r| r.entry == row.entry && r.bits == bits && r.dtype == dtype.name())
        .ok_or_else(|| Error::Rule(format!("{}: no {} instantiation", row.entry, dtype.name())))?;
    // The rotated payload is 16-bit only (the registry says so); the SHAPE has to agree, or the
    // descriptor would bind a byte count the kernel does not read. Same class of check as the
    // exact path's lanes-per-int4.
    if s.elem_bytes != 2 {
        return Err(Error::Rule(format!(
            "allreduce: a {}-byte payload has no rotated kernel; {} is 16-bit only",
            s.elem_bytes,
            dtype.name()
        )));
    }
    let nbytes = s.n_elem * s.elem_bytes;
    if nbytes % 16 != 0 {
        return Err(Error::Rule(format!(
            "allreduce: {nbytes} B is not 16 B aligned (the push is uint4)"
        )));
    }
    if nbytes > a.max_bytes {
        return Err(Error::Rule(format!(
            "allreduce: {nbytes} B exceeds the {} B IPC scratch slot; a larger message falls \
             back to RCCL",
            a.max_bytes
        )));
    }
    if nbytes < a.quant_min_bytes {
        return Err(Error::Rule(format!(
            "allreduce: {nbytes} B is below quant_min_bytes {}; the exact bf16 kernel wins there \
             and compression only pays once the transfer is bandwidth-bound",
            a.quant_min_bytes
        )));
    }
    let n_groups = (s.n_elem / a.group as u64) as u32;
    let chunk_groups = if bits % 2 == 0 {
        a.chunk_groups_even_bits
    } else {
        a.chunk_groups_odd_bits
    };
    let n_chunks = n_groups.div_ceil(chunk_groups);
    // The packed payload and its bf16 scales share one slot, split at scale_off_bytes.
    let payload = s.n_elem * bits as u64 / 8;
    let scale_off = a.max_bytes / 2;
    if payload > scale_off || scale_off + n_groups as u64 * 2 > a.max_bytes {
        return Err(Error::Rule(format!(
            "allreduce: a {bits}-bit payload of {payload} B plus {} B of scales does not fit the \
             slot's scale_off split at {scale_off} B",
            n_groups * 2
        )));
    }
    let nb = rules.ar_blocks(nbytes, n_chunks);
    let nt = (a.threads.clamp(64, 1024) / 64) * 64;
    let (args, bytes) = KArgs::new(abi::SIG_SABER_AR_WHT)
        .buf(names.input)
        .buf("ar_peer_scratch")
        .buf("ar_scratch")
        .buf("ar_loc_pack")
        .buf(names.output)
        .buf("ar_peer_flags")
        .buf("ar_flags")
        .buf("ar_seq")
        .i32(n_groups as i64)
        .i32(n_chunks as i64)
        .i64(a.max_bytes as i64)
        .i64(scale_off as i64)
        .i32(a.drain)
        .i32(a.acq)
        .finish()?;
    // The flags and the per-block sequence counters are protocol state: zeroed once at
    // construction, then owned by the kernels (the last block of a message resets the flag in
    // place and the counter only ever increments, so replay needs no clearing pass and a re-zero
    // would desynchronise the ranks).
    let workspaces = ar_workspaces(rules, true)?;
    // CMP27S: the wave-parallel body at the same grid when the chunks a block fit its LDS.
    let body = ar_wpar_route(rules, &sym.symbol, n_chunks, nb).unwrap_or(&sym.symbol);
    // T4: the race-check twin, whose check words sit above the scales in the slot.
    let kernel = ar_kernel(rules, body, scale_off + u64::from(n_groups) * 2)?;
    Ok(LaunchDescriptor {
        op: "allreduce_wht".into(),
        plugin: collective_plugin(&kernel, rules)?.into(),
        kernel,
        grid: [nb, 1, 1],
        block: [nt, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces,
        n_launches: 1,
    })
}
