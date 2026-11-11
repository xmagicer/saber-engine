//! The kernarg ABI: what the host writes into a kernel's argument buffer, and where.
//!
//! Offsets are not guessed at the launch site and they are not read out of the code object at run
//! time either. They are *derived* from a declared signature by the same natural-alignment packing
//! the AMDGPU C ABI uses, and then checked against the code object's own metadata (see
//! [`crate::Manifest::validate_against_hsaco`]). Two independent derivations that must agree is
//! the whole point: a kernarg layout the host and the kernel disagree about is memory corruption.

use serde::{Deserialize, Serialize};

/// A scalar slot in the kernarg segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgKind {
    /// A device-visible pointer (`global_buffer` in the code-object metadata).
    Buffer,
    I32,
    I64,
    F32,
    /// A struct passed BY VALUE. The reference's two attention kernels take `const SABERArgs a`, so the
    /// struct's own layout IS most of the kernarg segment and the code object reports it as ONE
    /// `by_value` argument of `sizeof(SABERArgs)`. The host still has to write nine pointers and
    /// eleven scalars into it, so [`STRUCT_SABER_ARGS`] names those fields and
    /// [`pack_fields_at`] gives their absolute offsets; this variant is what the manifest check
    /// compares against the blob.
    Struct {
        bytes: u32,
        align: u32,
    },
}

impl ArgKind {
    pub const fn size(self) -> u32 {
        match self {
            ArgKind::Buffer | ArgKind::I64 => 8,
            ArgKind::I32 | ArgKind::F32 => 4,
            ArgKind::Struct { bytes, .. } => bytes,
        }
    }

    pub const fn align(self) -> u32 {
        match self {
            ArgKind::Struct { align, .. } => align,
            other => other.size(),
        }
    }

    /// The `.value_kind` the code object must report for this slot.
    pub const fn value_kind(self) -> &'static str {
        match self {
            ArgKind::Buffer => "global_buffer",
            ArgKind::I32 | ArgKind::I64 | ArgKind::F32 | ArgKind::Struct { .. } => "by_value",
        }
    }
}

/// One named argument of a kernel signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigArg {
    pub name: &'static str,
    pub kind: ArgKind,
    /// **Bytes per ELEMENT** of a [`ArgKind::Buffer`] slot — the width of the type the kernel's
    /// pointer is declared with (`const __bf16 *` → 2, `const long *` → 8, an e4m3 cache as
    /// `unsigned char *` → 1). `0` on a scalar slot, and `0` on a buffer whose width is not
    /// declared yet.
    ///
    /// The kernarg LAYOUT does not depend on this — every pointer occupies eight bytes — so
    /// nothing about the packing or the code-object check reads it. It exists because a pointer's
    /// element width is part of the kernel's ARITHMETIC and was the last two defects found in this
    /// tree: the walk's `int *` over prod's i64 candidate table (F22) and the selector's `int *`
    /// / `__bf16 *` / `__bf16 *` over an i64 / f32 / f32 triple (F24). Both read the right
    /// addresses and the wrong bytes, and every gate they had was blind to it, because a gate that
    /// converts its reference's dtypes to its subject's is not gating the dtypes.
    ///
    /// `tests/capture_widths.rs` holds these against the dtype each capture actually binds.
    #[serde(default)]
    pub elem: u32,
}

impl SigArg {
    /// The declared element width of a buffer slot, if there is one.
    #[must_use]
    pub const fn elem_bytes(&self) -> Option<u32> {
        if self.elem == 0 {
            None
        } else {
            Some(self.elem)
        }
    }
}

const fn a(name: &'static str, kind: ArgKind) -> SigArg {
    SigArg {
        name,
        kind,
        elem: 0,
    }
}

/// Logical argument contract of the external RCCL BF16 sum implementation.
pub const SIG_RCCL_BF16: &[SigArg] = &[a("in", Buffer), a("out", Buffer), a("count", I32)];

/// A POINTER slot that declares what one element of it is, in bytes. See [`SigArg::elem`].
const fn p(name: &'static str, elem: u32) -> SigArg {
    SigArg {
        name,
        kind: Buffer,
        elem,
    }
}

/// `(offset, size)` per argument, by natural alignment; the total is the explicit kernarg size
/// (the HIP runtime appends its own hidden args after it).
pub fn pack(args: &[SigArg]) -> (Vec<(u32, u32)>, u32) {
    pack_fields_at(args, 0)
}

/// [`pack`] with the first slot forced to `base`. Used to lay a by-value struct's FIELDS over the
/// region the struct occupies in the kernarg segment: the code object sees one `by_value` blob,
/// the host has to write nine pointers into it, and both views have to agree byte for byte.
///
/// The struct's own trailing padding is not part of the field walk, so the returned total is the
/// offset one past the LAST FIELD, not `base + sizeof(struct)`.
pub fn pack_fields_at(args: &[SigArg], base: u32) -> (Vec<(u32, u32)>, u32) {
    let mut out = Vec::with_capacity(args.len());
    let mut off = base;
    for s in args {
        off = off.next_multiple_of(s.kind.align());
        out.push((off, s.kind.size()));
        off += s.kind.size();
    }
    (out, off)
}

use ArgKind::{Buffer, F32, I32, I64};

/// `saber_mxfp4_fp8_gemm_decode` (split-K partials + fused reduce).
pub const SIG_GEMM_DECODE: &[SigArg] = &[
    a("a", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("as", Buffer),
    a("partial", Buffer),
    a("counter", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("pb1", I32),
    a("pb2", I32),
    a("astride", I64),
];

/// The decode PAIR (ws-h37): two decode GEMMs over ONE activation in one dispatch. The shared
/// A side first, then each problem's weight / partial / counter / output, then the shared row
/// and reduction extents and each problem's N, split count and n-block count.
pub const SIG_GEMM_DECODE_PAIR: &[SigArg] = &[
    a("a", Buffer),
    a("as", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("partial", Buffer),
    a("counter", Buffer),
    a("c", Buffer),
    a("w_b", Buffer),
    a("ws_b", Buffer),
    a("wref_b", Buffer),
    a("partial_b", Buffer),
    a("counter_b", Buffer),
    a("c_b", Buffer),
    a("m", I32),
    a("k", I32),
    a("n", I32),
    a("ks", I32),
    a("nb", I32),
    a("n_b", I32),
    a("ks_b", I32),
    a("nb_b", I32),
];

/// Fusion-map row 4 (ws-h40): the gate_up decode GEMM (DIRECT, `ks1`) whose last `ep_rows`
/// arriving n-tiles each run `silu_mul_quant` on one row. The decode GEMM's fourteen arguments
/// unchanged (`c` is the gate_up row the epilogue reads), then the producer's `q` / `scale`, the
/// arrival counter `tiles` and `ep_rows` (`m`, except in the gate's `smq-rows` control).
pub const SIG_GEMM_DECODE_SMQ: &[SigArg] = &[
    a("a", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("as", Buffer),
    a("partial", Buffer),
    a("counter", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("pb1", I32),
    a("pb2", I32),
    a("astride", I64),
    a("q", Buffer),
    a("scale", Buffer),
    a("tiles", Buffer),
    a("ep_rows", I32),
];

/// Fusion-map row 3 (ws-h38): the decode GEMM whose split-K last block also runs the exact
/// two-rank all-reduce and the `_nq` epilogue. The decode GEMM's fourteen arguments unchanged
/// (`c` is ALSO the reduce's local input), then the `_nq` epilogue's minus its `in` (it is `c`)
/// and its row count's source (the GEMM's `m`), plus the arrival counter `tiles` and `ep_rows`
/// (how many rows get an epilogue: `m`, except in the gate's `gemmar-rows` control).
pub const SIG_GEMM_DECODE_AR_NQ: &[SigArg] = &[
    a("a", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("as", Buffer),
    a("partial", Buffer),
    a("counter", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("pb1", I32),
    a("pb2", I32),
    a("astride", I64),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("residual", Buffer),
    a("norm_w", Buffer),
    a("q_out", Buffer),
    a("scale_out", Buffer),
    a("res_out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("tiles", Buffer),
    a("k16", I32),
    a("slot_stride16", I32),
    a("eps", F32),
    a("drain", I32),
    a("acq", I32),
    a("ep_rows", I32),
];

/// CMP27AE: `mxfp4_gemm_decode_arw_*` -- the decode GEMM's fourteen arguments, then the rotated
/// reduce's peer / my scratch, the reduced output, the two flag views, the counters, the slot stride,
/// the family's offset inside a slot, drain and acquire (`hip/mxfp4_gemm_decode_arw.hip`).
pub const SIG_GEMM_DECODE_ARW: &[SigArg] = &[
    a("a", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("as", Buffer),
    a("partial", Buffer),
    a("counter", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("pb1", I32),
    a("pb2", I32),
    a("astride", I64),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("slot_stride_bytes", I64),
    a("fuse_off", I64),
    a("drain", I32),
    a("acq", I32),
];

/// `saber_mxfp4_fp8_gemm_folded` and `_atiled` share one signature.
pub const SIG_GEMM_FOLDED: &[SigArg] = &[
    a("a", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("as", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("pb1", I32),
    a("pb2", I32),
    a("astride", I64),
];

/// `saber_mxfp4_fp8_gemm` (the unfolded path, no per-row reference exponent).
pub const SIG_GEMM_BASE: &[SigArg] = &[
    a("a", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("as", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// `saber_add_rms_quant`.
pub const SIG_ADD_RMS_QUANT: &[SigArg] = &[
    a("y", Buffer),
    a("res", Buffer),
    a("w", Buffer),
    a("q", Buffer),
    a("scale", Buffer),
    a("res_out", Buffer),
    a("k", I32),
    a("eps", F32),
];

/// `saber_silu_mul_quant`.
pub const SIG_SILU_MUL_QUANT: &[SigArg] = &[
    a("gu", Buffer),
    a("q", Buffer),
    a("scale", Buffer),
    a("n", I32),
];

/// FN4's grouped MXFP4 W4A8 GEMM (`moe_grouped_w4a8_*`, `hip/moe_grouped.hip`): the decode
/// body's operands with W/Ws/Wref and M read per tile (`xptr`, the tile tables, `meta[0]`) and the
/// A row through `row_tok` (gather) or the tile row itself (compact).
pub const SIG_MOE_GROUPED: &[SigArg] = &[
    a("a", Buffer),
    a("xptr", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("as", Buffer),
    a("p", Buffer),
    a("cnt", Buffer),
    a("c", Buffer),
    a("n", I32),
    a("k", I32),
    a("pmax", I32),
];

/// FN4's expert-major `silu_mul_quant` (`moe_silu_mul_quant_t256`): the producer's four operands
/// plus the row map and the live row count.
pub const SIG_MOE_SILU: &[SigArg] = &[
    a("gu", Buffer),
    a("q", Buffer),
    a("scale", Buffer),
    a("n", I32),
    a("row_tok", Buffer),
    a("meta", Buffer),
];

/// FN4's router softmax + top-10 + expert-major sort (`moe_route_topk10_sort_e512`).
pub const SIG_MOE_ROUTE: &[SigArg] = &[
    a("logits", Buffer),
    a("t", I32),
    a("e_lo", I32),
    a("e_loc", I32),
    a("tile_blocks", I32),
    a("with_shared", I32),
    a("topk_ids", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("row_tok", Buffer),
    a("row_route", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("expert_cnt", Buffer),
    a("meta", Buffer),
    a("blk_cnt", Buffer),
    a("rank", Buffer),
    a("arrive", Buffer),
];

/// FN4's weighted combine (`moe_combine_bf16`).
pub const SIG_MOE_COMBINE: &[SigArg] = &[
    a("y", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("sgate", Buffer),
    a("meta", Buffer),
    a("out", Buffer),
    a("h", I32),
];

/// FN26's fused BACK HALF of the MoE block (`moe_back_gu{ks}_dn{ks}_p{0,1}`, `hip/moe_grouped.hip`):
/// the shared-expert gate GEMV + grouped gate_up + silu-quant + grouped down + combine as ONE launch
/// (1-D grid `1 + tc * 10 * gu_ks + tc * 20 * dn_ks` workgroups of 256), chained by the device
/// counters in `sync` (`moe::BackTrain`): the operands of the block's launches 2 and 4..7 plus the
/// counter words and the static sizes.
pub const SIG_MOE_BACK: &[SigArg] = &[
    a("a", Buffer),
    a("as", Buffer),
    a("xgu", Buffer),
    a("xdn", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("p_gu", Buffer),
    a("p_dn", Buffer),
    a("cnt_gu", Buffer),
    a("cnt_dn", Buffer),
    a("gu", Buffer),
    a("q", Buffer),
    a("q_scale", Buffer),
    a("y", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("w_sg", Buffer),
    a("bi", Buffer),
    a("sgate", Buffer),
    a("out", Buffer),
    a("sync", Buffer),
    a("t", I32),
    a("pmax", I32),
    a("tc", I32),
];

/// FN114c's fused back half with the SHARED EXPERT SPLIT across the two EP ranks
/// (`moe_back_ss_gu{ks}_dn{ks}_p1`, `hip/moe_grouped_ss.hip`, `--fn-shared-expert split`):
/// [`SIG_MOE_BACK`] + `xsh`, this rank's half table of the layer's shared expert (6 u64: gate_up
/// w / ws / wref, down w / ws / wref at FN80a's half layout).
pub const SIG_MOE_BACK_SS: &[SigArg] = &[
    a("a", Buffer),
    a("as", Buffer),
    a("xgu", Buffer),
    a("xdn", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("p_gu", Buffer),
    a("p_dn", Buffer),
    a("cnt_gu", Buffer),
    a("cnt_dn", Buffer),
    a("gu", Buffer),
    a("q", Buffer),
    a("q_scale", Buffer),
    a("y", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("w_sg", Buffer),
    a("bi", Buffer),
    a("sgate", Buffer),
    a("out", Buffer),
    a("sync", Buffer),
    a("t", I32),
    a("pmax", I32),
    a("tc", I32),
    a("xsh", Buffer),
];

/// FN137 `FarArgs` (`hip/moe_grouped_far.hip`): the fused MoE-out all-reduce's channel -- the
/// peer's / this rank's window (slot 0; slot 1 at + `stride16`), the peer's / this rank's 20 slice
/// flags, the channel counter [calls, this call], the SUM, the diagnostics, the stamp pointer (0
/// served), `stride16`, a pad: eight pointers + two ints, 72 bytes, 8-aligned.
pub const FAR_ARGS_BYTES: u32 = 72;

const FAR_ARGS: ArgKind = ArgKind::Struct {
    bytes: FAR_ARGS_BYTES,
    align: 8,
};

/// FN137 `FarArgs`' fields, in order (laid over the blob by [`pack_fields_at`]).
pub const STRUCT_FAR_ARGS: &[SigArg] = &[
    a("peer", Buffer),
    a("my", Buffer),
    a("pflags", Buffer),
    a("mflags", Buffer),
    a("ctr", Buffer),
    a("red", Buffer),
    a("diag", Buffer),
    a("stamp", Buffer),
    a("stride16", I32),
    a("pad", I32),
];

/// FN137's fused back half + MoE-out all-reduce (`fn137_moe_back_h2_far_gu{ks}_dn{ks}_p1{,_ctl,_chk}`,
/// `hip/moe_grouped_far.hip`, `--fn-moe-ar-push on`): [`SIG_MOE_BACK`] + the channel `FarArgs`.
pub const SIG_MOE_BACK_FAR: &[SigArg] = &[
    a("a", Buffer),
    a("as", Buffer),
    a("xgu", Buffer),
    a("xdn", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("p_gu", Buffer),
    a("p_dn", Buffer),
    a("cnt_gu", Buffer),
    a("cnt_dn", Buffer),
    a("gu", Buffer),
    a("q", Buffer),
    a("q_scale", Buffer),
    a("y", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("w_sg", Buffer),
    a("bi", Buffer),
    a("sgate", Buffer),
    a("out", Buffer),
    a("sync", Buffer),
    a("t", I32),
    a("pmax", I32),
    a("tc", I32),
    a("fa", FAR_ARGS),
];

/// FN137's oracle skew kernel `fn137_delay(unsigned long long ticks)` (probe only).
pub const SIG_FN137_DELAY: &[SigArg] = &[a("ticks", I64)];

/// FN164's fused mixer-out reduce (`fn164_far_<parent>{,_ctl,_chk}`, `hip/flashnext_t8far.hip`,
/// `--fn-mixer-ar-push`): the parent W8A8 GEMM's [`SIG_FN_T8_A8W8_LDC`] + FN137's channel `FarArgs`
/// (the counters are the 40 workgroups' own, the flags one a workgroup).
pub const SIG_FN164_T8_FAR: &[SigArg] = &[
    p("a", 1),
    p("w", 1),
    p("as", 4),
    p("ws", 4),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("ldc", I32),
    a("fa", FAR_ARGS),
];

/// FN84's fused back half under the ORDER LAW (`moe_back_mo{,p}_gu{ks}_dn{ks}_p1`,
/// `hip/moe_grouped.hip`): [`SIG_MOE_BACK`] + the tier's pinned-RAM device range `[ram_lo, ram_hi)`
/// (the RAM predicate: an ordering hint, no byte depends on it) and `kint` (< 0 id order, 0 RAM
/// tiles first, k > 0 one RAM workgroup per k resident ones).
pub const SIG_MOE_BACK_MO: &[SigArg] = &[
    a("a", Buffer),
    a("as", Buffer),
    a("xgu", Buffer),
    a("xdn", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("p_gu", Buffer),
    a("p_dn", Buffer),
    a("cnt_gu", Buffer),
    a("cnt_dn", Buffer),
    a("gu", Buffer),
    a("q", Buffer),
    a("q_scale", Buffer),
    a("y", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("w_sg", Buffer),
    a("bi", Buffer),
    a("sgate", Buffer),
    a("out", Buffer),
    a("sync", Buffer),
    a("t", I32),
    a("pmax", I32),
    a("tc", I32),
    a("ram_lo", I64),
    a("ram_hi", I64),
    a("kint", I32),
];

/// FN31's fused FRONT HALF of the MoE block (`moe_front_e512_nw{4,8}_{u8,u20,s4,s2,s1}`,
/// `hip/moe_route.hip`): the router GEMM (FN5's split-K band body) + the route and sort (FN4's
/// law, the whole step in one workgroup's LDS) as ONE launch (`moe::FrontTrain`): the block input
/// `bi` `[T][2560]` bf16, the router weight `[512][2560]` bf16, the partial slices
/// `[split][T][512]` f32, the one `sync` word (zeroed once; zero again after every launch), the
/// router logits `[T][512]` bf16 (written, the band's bytes), `k` (2,560), then the route's
/// operands in `SIG_MOE_ROUTE`'s order without its three scratch arrays.
pub const SIG_MOE_FRONT: &[SigArg] = &[
    a("bi", Buffer),
    a("wr", Buffer),
    a("part", Buffer),
    a("sync", Buffer),
    a("logits", Buffer),
    a("k", I32),
    a("t", I32),
    a("e_lo", I32),
    a("e_loc", I32),
    a("tile_blocks", I32),
    a("with_shared", I32),
    a("topk_ids", Buffer),
    a("topk_w", Buffer),
    a("route_pos", Buffer),
    a("row_tok", Buffer),
    a("row_route", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("expert_cnt", Buffer),
    a("meta", Buffer),
];

/// FN4's A16 control arm (`moe_grouped_a16_ref_g{0,1}`): bf16 A, the same tile tables.
pub const SIG_MOE_A16: &[SigArg] = &[
    a("a", Buffer),
    a("xptr", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("c", Buffer),
    a("n", I32),
    a("k", I32),
];

/// FN148's NVFP4 oracle (`fn148_a16_nvfp4_ref_g{0,1}`): [`SIG_MOE_A16`] + `gsplit` (the first row of
/// the record's second global: gate_up N / 2, down N).
pub const SIG_FN148_A16: &[SigArg] = &[
    a("a", Buffer),
    a("xptr", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("c", Buffer),
    a("n", I32),
    a("k", I32),
    a("gsplit", I32),
];

/// FN148's NVFP4 W4A8 forms (`fn148_w4a8{b,c}_*`, `hip/moe_fn148.hip`): [`SIG_MOE_GROUPED`] + `gsplit`;
/// `xptr[3e + {0,1,2}]` = the v2 record's `w`, `ws` (E4M3 [K/16, N]) and `gs` (f32).
pub const SIG_FN148_GROUPED: &[SigArg] = &[
    a("a", Buffer),
    a("xptr", Buffer),
    a("tile_expert", Buffer),
    a("tile_row0", Buffer),
    a("tile_rows", Buffer),
    a("row_tok", Buffer),
    a("meta", Buffer),
    a("as", Buffer),
    a("p", Buffer),
    a("cnt", Buffer),
    a("c", Buffer),
    a("n", I32),
    a("k", I32),
    a("pmax", I32),
    a("gsplit", I32),
];

/// `saber_gdn_norm_quant`.
pub const SIG_GDN_NORM_QUANT: &[SigArg] = &[
    a("x", Buffer),
    a("z", Buffer),
    a("zs", I64),
    a("w", Buffer),
    a("q", Buffer),
    a("scale", Buffer),
    a("n", I32),
    a("eps", F32),
];

// ---------------------------------------------------------------------------- saber (plugin 2)

/// `SABERArgs` (saber `ref.h:41-58`), field by field. BOTH paged attention kernels take it by
/// value, so this is the layout the host writes; `crates/saber-kernels/hip/saber_args.h` carries
/// the same struct, sliced verbatim from the reference.
pub const SABER_ARGS_FIELDS: &[SigArg] = &[
    a("q", Buffer),
    a("kv", Buffer),
    a("block_table", Buffer),
    a("seqused_k", Buffer),
    a("out", Buffer),
    a("k_descale", Buffer),
    a("v_descale", Buffer),
    a("q_descale", Buffer),
    a("scratch", Buffer),
    a("num_seqs", I32),
    a("q_len", I32),
    a("q_heads", I32),
    a("kv_heads", I32),
    a("head_dim", I32),
    a("block_size", I32),
    a("max_blocks", I32),
    a("kv_block_stride", I64),
    a("kv_head_stride", I64),
    a("scale", F32),
    a("splits_in", I32),
    a("max_ctx", I32),
];

/// `sizeof(SABERArgs)`: the fields end at 132 and the struct's 8-byte alignment rounds to 136.
pub const SABER_ARGS_BYTES: u32 = 136;

const SABER_ARGS: ArgKind = ArgKind::Struct {
    bytes: SABER_ARGS_BYTES,
    align: 8,
};

/// `saber_attn_prefill_kernel(const SABERArgs a)`.
pub const SIG_SABER_ATTN_PREFILL: &[SigArg] = &[a("args", SABER_ARGS)];

/// `saber_attn_decode_kernel(const SABERArgs a, int splits)`. `splits` is passed a SECOND time
/// outside the struct: the struct's own `splits` is the caller's request (0 = "use the split
/// law") and this one is what the law answered.
pub const SIG_SABER_ATTN_DECODE: &[SigArg] = &[a("args", SABER_ARGS), a("splits", I32)];

/// FN19 `QsaKeys` (flashnext_qsa_attn.hip): one attention CTA's union key list -- three pointers
/// and a long, 32 bytes, 8-aligned.
pub const QSA_KEYS_BYTES: u32 = 32;

const QSA_KEYS: ArgKind = ArgKind::Struct {
    bytes: QSA_KEYS_BYTES,
    align: 8,
};

/// FN124 `fn124_cmp_u32(a, b, n, out)`: the in-situ twin's comparator (out[0] += the differing
/// 32-bit words of a[0, n) vs b).
pub const SIG_FN124_CMP: &[SigArg] = &[p("a", 4), p("b", 4), a("n", I64), p("out", 4)];

/// FN19 `fn_qsa_attn_prefill_*(const SABERArgs a, const QsaKeys qa)`: the gather prefill body.
pub const SIG_FN_QSA_ATTN_PREFILL: &[SigArg] = &[a("args", SABER_ARGS), a("qa", QSA_KEYS)];

/// FN19 `fn_qsa_attn_decode_*(const SABERArgs a, int splits, const QsaKeys qa)`.
pub const SIG_FN_QSA_ATTN_DECODE: &[SigArg] =
    &[a("args", SABER_ARGS), a("splits", I32), a("qa", QSA_KEYS)];

/// `saber_attn_splitkv_combine_kernel(const SABERArgs a, int splits, int tile)`.
pub const SIG_SABER_ATTN_COMBINE: &[SigArg] =
    &[a("args", SABER_ARGS), a("splits", I32), a("tile", I32)];

/// `saber_attn_declc_*` (ws-h38): the decode kernel's arguments plus the per-(sequence, kv head)
/// arrival counter the last-arriving split block elects on before it runs the split-KV merge.
pub const SIG_SABER_ATTN_DECLC: &[SigArg] = &[
    a("args", SABER_ARGS),
    a("splits", I32),
    a("attn_decode_counter", Buffer),
];

/// The RAGGED decode attention (workstream C2): `saber_attn_decode_ragged_*(const SABERArgs a, int
/// splits, const int* cu_q)`. `cu_q` is the step's cumulative LIVE-row prefix (`num_seqs + 1`
/// entries), 8-aligned after `splits`: 136 + 4 -> 144.
pub const SIG_SABER_ATTN_DECODE_RAGGED: &[SigArg] =
    &[a("args", SABER_ARGS), a("splits", I32), a("cu_q", Buffer)];

/// P27-15 `p27_attn_decode_anc_*`: [`SIG_SABER_ATTN_DECODE_RAGGED`] + the per-row u16 ancestor
/// words of the q_len tail (one per live row, indexed like `q`): 144 + 8 -> 152.
pub const SIG_P27_ATTN_DECODE_ANC: &[SigArg] = &[
    a("args", SABER_ARGS),
    a("splits", I32),
    a("cu_q", Buffer),
    p("anc", 2),
];

/// P27-15 `p27_gdn_tree_f16s_*`: the WY-form tree verify of the GDN decode update (rx7's
/// recurrent operands, one committed f16 slot a sequence, the rows' ancestor words, the tree
/// record pair `ts_in` / `ts_out`, the previous tree's accepted row `acc` for the ADV prologue).
pub const SIG_P27_GDN_TREE: &[SigArg] = &[
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("slot", Buffer),
    a("o", Buffer),
    a("cu", Buffer),
    p("anc", 2),
    p("ts_in", 4),
    p("ts_out", 4),
    a("ts_seq", I64),
    a("ts_head", I64),
    a("acc", Buffer),
    a("zg", Buffer),
    a("nw", Buffer),
    a("neps", F32),
    a("nact", I32),
    a("h", I32),
    a("hg", I32),
    a("scale", F32),
    a("sp_thr", F32),
];

/// P27-W2-5 `p27w2_gdn_replay[_materialize|_noround]_f16s_*`: rx7's recurrent operands with ONE
/// base slot a sequence (`sidx[n * stride]`), `rp` [n][3] = {rows to replay from the previous
/// step's stash (0 after a prefill / materialize), the sequence's stash tag, the stash parity}, the
/// stash pool (indexed by the slot SET `(base - 1) / sx_div`, two `sx_head`-byte regions a (set,
/// head)) and the refusal word `err` (bit 0 a stash mismatch, bit 1 a row count past the stash
/// depth, bit 2 a base off the set grid: the sequence's rows are poisoned, never replayed short).
pub const SIG_P27W2_GDN_REPLAY: &[SigArg] = &[
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("cu", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("rp", Buffer),
    a("stash", Buffer),
    a("sx_div", I64),
    a("sx_head", I64),
    p("err", 4),
    a("h", I32),
    a("hg", I32),
    a("scale", F32),
    a("sp_thr", F32),
];

/// CMP27AH `p27w2_gdn_cuf_f16s_*` (`--engine-step-arms cuf`): the batched decode PAIR in one launch
/// -- `saber_gdn_conv_update`'s operands (the conv window read at `naccept - 1` and rewritten shifted;
/// its `cache_idx` IS the replay's `sidx` at the same stride, so it is not repeated), the counter
/// region `cnt` ([n][hg] u32, zero before and after every launch: the LAST of a k head's
/// `h / hg x VS` workgroups to have read the shared q / k window rewrites it), then
/// [`SIG_P27W2_GDN_REPLAY`]'s operands with `q` / `k` / `v` now WRITTEN (the conv's outputs).
pub const SIG_P27W2_GDN_CUF: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("state_len_max", I32),
    a("naccept", Buffer),
    a("maxq", I32),
    p("cnt", 4),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("cu", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("rp", Buffer),
    a("stash", Buffer),
    a("sx_div", I64),
    a("sx_head", I64),
    p("err", 4),
    a("h", I32),
    a("hg", I32),
    a("scale", F32),
    a("sp_thr", F32),
];

/// P27-15 `p27_gdn_tree_advance_f16s_*`: the accepted row's state from a tree record.
pub const SIG_P27_GDN_TREE_ADVANCE: &[SigArg] = &[
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("slot", Buffer),
    a("slot_out", Buffer),
    p("ts", 4),
    a("ts_seq", I64),
    a("ts_head", I64),
    a("acc", Buffer),
];

/// CMP27L `p27_gdn_tree_advance_layers_f16s_*`: the flush's advance for every GDN layer in one
/// launch (`state_base` / `ts_base` = [layers] u64 tables).
pub const SIG_P27_GDN_TREE_ADVANCE_LAYERS: &[SigArg] = &[
    p("state_base", 8),
    p("ts_base", 8),
    a("st_slot", I64),
    a("st_head", I64),
    a("slot", Buffer),
    a("slot_out", Buffer),
    a("ts_seq", I64),
    a("ts_head", I64),
    a("acc", Buffer),
];

/// `saber_attn_splitkv_combine_ragged_*(const SABERArgs a, int splits, int tile, const int* cu_q)`.
pub const SIG_SABER_ATTN_COMBINE_RAGGED: &[SigArg] = &[
    a("args", SABER_ARGS),
    a("splits", I32),
    a("tile", I32),
    a("cu_q", Buffer),
];

/// `saber_gdn_fused_update_kernel<ABF16>`: the decode GDN step as one launch.
pub const SIG_SABER_GDN_FUSED_UPDATE: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("state_len_max", I32),
    a("cache_idx", Buffer),
    a("ci_stride", I64),
    a("naccept", Buffer),
    a("cu", Buffer),
    a("n_seqs", I32),
    a("h", I32),
    a("hg", I32),
    a("maxq", I32),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("scale", F32),
    a("sp_thr", F32),
    a("bar", Buffer),
    a("num_wg", I32),
    a("o_rows", I32),
];

/// Fusion-map ROW 5 (ws-h42): `saber_gdn_fnq_*`, the GDN fused update whose last `ep_rows` arriving
/// workgroups each run `gdn_norm_quant` on one row of `o`. The fused update's thirty-six
/// arguments unchanged, then the norm's (its `x` is `o`): `nq_z`, `nq_zs`, `nq_w`, `nq_q`,
/// `nq_scale`, `nq_n`, `nq_eps`, the arrival counter `tiles` and `ep_rows` (the norm's M, except
/// in the gate's `fnq-rows` control).
pub const SIG_SABER_GDN_FNQ: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("state_len_max", I32),
    a("cache_idx", Buffer),
    a("ci_stride", I64),
    a("naccept", Buffer),
    a("cu", Buffer),
    a("n_seqs", I32),
    a("h", I32),
    a("hg", I32),
    a("maxq", I32),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("scale", F32),
    a("sp_thr", F32),
    a("bar", Buffer),
    a("num_wg", I32),
    a("o_rows", I32),
    a("nq_z", Buffer),
    a("nq_zs", I64),
    a("nq_w", Buffer),
    a("nq_q", Buffer),
    a("nq_scale", Buffer),
    a("nq_n", I32),
    a("nq_eps", F32),
    a("tiles", Buffer),
    a("ep_rows", I32),
];

/// `saber_gdn_gated_rmsnorm_kernel<ACT>`.
pub const SIG_SABER_GDN_NORM: &[SigArg] = &[
    a("x", Buffer),
    a("z", Buffer),
    a("w", Buffer),
    a("o", Buffer),
    a("rows", I64),
    a("xrow", I64),
    a("zrow", I64),
    a("orow", I64),
    a("eps", F32),
];

/// `saber_gdn_conv_prep_kernel<HPB, ABF16>`: the GDN prefill preamble — causal conv, qkv split,
/// qk l2 norm, gating and the per-chunk cumsum, in one kernel.
pub const SIG_SABER_GDN_CONV_PREP: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("cache_idx", Buffer),
    a("ci_stride", I64),
    a("has_init", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("g", Buffer),
    a("beta", Buffer),
    a("cu", Buffer),
    a("n_seqs", I32),
    a("h", I32),
    a("hg", I32),
    a("sp_thr", F32),
];

/// `saber_gdn_conv_update_kernel<HPB>`: the decode conv over the rolling speculative window.
pub const SIG_SABER_GDN_CONV_UPDATE: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("state_len_max", I32),
    a("cache_idx", Buffer),
    a("ci_stride", I64),
    a("naccept", Buffer),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("cu", Buffer),
    a("h", I32),
    a("hg", I32),
    a("maxq", I32),
];

/// P27-15b `p27_tree_kv_compact`: the accepted path's K/V token rows moved from their row slots
/// to their depth slots, in order (`layer_base` [layers] u64 pool bases, `moves` [n][2] i32
/// absolute token slots).
pub const SIG_P27_TREE_KV_COMPACT: &[SigArg] = &[
    p("layer_base", 8),
    p("moves", 4),
    a("n_moves", I32),
    a("block_stride", I64),
    a("head_stride", I64),
    a("row_bytes", I32),
    a("block_size", I32),
];

/// P27-W2-4 `p27_tree_conv_commit`: a tree verify's accepted conv window moved to the chain's
/// entries (`layer_base` [layers] u64 conv pool bases, `moves` [n][2] i32 entry indices).
pub const SIG_P27_TREE_CONV_COMMIT: &[SigArg] = &[
    p("layer_base", 8),
    p("moves", 4),
    a("n_moves", I32),
    a("seq_off", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("dims", I32),
];

/// P27-15b `p27_gdn_conv_tree_*`: [`SIG_SABER_GDN_CONV_UPDATE`] with `naccept` replaced by the
/// committed window's cache entries `hsrc` ([n][4] i32) and every row's older taps `tsrc`
/// ([rows][4] i32: 0..2 = the window, 3 + r = verify row r).
pub const SIG_P27_GDN_CONV_TREE: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("state_len_max", I32),
    a("cache_idx", Buffer),
    a("ci_stride", I64),
    p("hsrc", 4),
    p("tsrc", 4),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("cu", Buffer),
    a("h", I32),
    a("hg", I32),
    a("maxq", I32),
];

/// `saber_gdn_recurrent_update_kernel<ABF16, NORM, VS>`: the decode delta-rule state update.
pub const SIG_SABER_GDN_RECURRENT: &[SigArg] = &[
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("cu", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("naccept", Buffer),
    a("zg", Buffer),
    a("nw", Buffer),
    a("neps", F32),
    a("nact", I32),
    a("h", I32),
    a("hg", I32),
    a("scale", F32),
    a("sp_thr", F32),
];

/// `saber_gdn_kkt_solve_kernel<HG_RATIO>`: the chunk preamble's triangular solve.
pub const SIG_SABER_GDN_KKT: &[SigArg] = &[
    a("k", Buffer),
    a("beta", Buffer),
    a("g", Buffer),
    a("a_out", Buffer),
    a("cu", Buffer),
    a("n_seqs", I32),
    a("h", I32),
    a("hg", I32),
];

/// `saber_gdn_chunk_scan_kernel`: the WY recompute, the state recurrence and the output.
pub const SIG_SABER_GDN_CHUNK_SCAN: &[SigArg] = &[
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("a_mat", Buffer),
    a("gcs", Buffer),
    a("beta", Buffer),
    a("h0", Buffer),
    a("o", Buffer),
    a("ht", Buffer),
    a("cu", Buffer),
    a("h", I32),
    a("hg", I32),
    a("scale", F32),
];

/// `saber_ar_oneshot_2rank_exact_kernel<T, LANES>`: the exact one-shot push reduce. Each block owns
/// a slice of the 16-byte-word space and drives its own flag handshake, so there is no cross-block
/// sync and the grid is free.
pub const SIG_SABER_AR_EXACT: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("n16", I32),
    a("slot_stride16", I32),
    a("drain", I32),
    a("acq", I32),
];

/// FN114b `fn_ar_ll_exact_bf16[_chk|_chkctl]`: the exact reduce with the flag carried in the data
/// words -- the one-shot's arguments with the flag pair replaced by the form's window tails (the
/// wave flags) and `seq_ctrs` its own counters (`flashnext_ar.hip`).
pub const SIG_FN_AR_LL: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("out", Buffer),
    a("peer_tail", Buffer),
    a("my_tail", Buffer),
    a("seq_ctrs", Buffer),
    a("n16", I32),
    a("slot_stride16", I32),
    a("drain", I32),
    a("acq", I32),
];

/// `saber_ar_oneshot_2rank_exact_nq_kernel`: the same reduce with the decoder layer's fused post-AR
/// epilogue (residual add + Gemma RMS norm + per-token e4m3 quant), one block per row. Not
/// templated: bf16 payload only.
pub const SIG_SABER_AR_EXACT_NQ: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("residual", Buffer),
    a("norm_w", Buffer),
    a("q_out", Buffer),
    a("scale_out", Buffer),
    a("res_out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("k16", I32),
    a("slot_stride16", I32),
    a("eps", F32),
    a("drain", I32),
    a("acq", I32),
];

/// `saber_ar_oneshot_2rank_whtq_kernel<T, BITS>`.
pub const SIG_SABER_AR_WHT: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("loc_pack", Buffer),
    a("out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("n_groups", I32),
    a("n_chunks", I32),
    a("slot_stride_bytes", I64),
    a("scale_off_bytes", I64),
    a("drain", I32),
    a("acq", I32),
];

/// P27-1b's split PUSH half (`saber_ar_split_2rank_wht6_bf16_push`): the one-shot's arguments, then
/// the host's monotone `flag_val` the LAST-arriving block raises the peer's split flag to.
pub const SIG_SABER_AR_SPLIT_PUSH: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("loc_pack", Buffer),
    a("out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("n_groups", I32),
    a("n_chunks", I32),
    a("slot_stride_bytes", I64),
    a("scale_off_bytes", I64),
    a("drain", I32),
    a("acq", I32),
    a("flag_val", I32),
];

/// P27-1c's FEWER-BLOCKS push (`..._push_nb`): the push's arguments, then `nbv` -- the one-shot's
/// block count, the VIRTUAL blocks that own the chunks / counters / per-block flags.
pub const SIG_SABER_AR_SPLIT_PUSH_NB: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("loc_pack", Buffer),
    a("out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    a("n_groups", I32),
    a("n_chunks", I32),
    a("slot_stride_bytes", I64),
    a("scale_off_bytes", I64),
    a("drain", I32),
    a("acq", I32),
    a("flag_val", I32),
    a("nbv", I32),
];

// ---------------------------------------------------------------------------- Paro (plugin 3)

/// `pq_add_rms_rot<ROT, I8>`: fused residual-add + Gemma RMSNorm, and in the decode band the
/// pairwise rotation + per-group fp8 quant that feeds the W4A8 GEMM.
pub const SIG_PARO_ADD_RMS_ROT: &[SigArg] = &[
    a("y", Buffer),
    a("res", Buffer),
    a("w", Buffer),
    a("eps", F32),
    a("t", Buffer),
    a("cs", Buffer),
    a("hs", Buffer),
    a("ro", Buffer),
    a("a", Buffer),
    a("asg", Buffer),
    a("rs", Buffer),
    a("m", I32),
    a("k", I32),
    a("krot", I32),
    a("gpw", I32),
];

/// `pq_ew_rot<MODE, ROT, I8>`: silu-mul (0) / attention gate (1) / GDN gated RMSNorm (2), with the
/// same rotate + quant epilogue in the decode band.
pub const SIG_PARO_EW_ROT: &[SigArg] = &[
    a("x", Buffer),
    a("y", Buffer),
    a("ys", I64),
    a("w", Buffer),
    a("eps", F32),
    a("t", Buffer),
    a("cs", Buffer),
    a("hs", Buffer),
    a("a", Buffer),
    a("asg", Buffer),
    a("rs", Buffer),
    a("m", I32),
    a("n", I32),
    a("krot", I32),
    a("gpw", I32),
];

/// `pq_add_rms_rot_tok<W, TILED, IL>`: the same producer with a PER-TOKEN scale instead of a
/// per-group one, which is what the prefill band's A-tiled GEMM reads.
pub const SIG_PARO_ADD_RMS_ROT_TOK: &[SigArg] = &[
    a("y", Buffer),
    a("res", Buffer),
    a("w", Buffer),
    a("eps", F32),
    a("t", Buffer),
    a("cs", Buffer),
    a("hs", Buffer),
    a("ro", Buffer),
    a("a", Buffer),
    a("as", Buffer),
    a("m", I32),
    a("k", I32),
    a("krot", I32),
];

/// `pq_ew_rot_tok<MODE, W, TILED, IL>`.
pub const SIG_PARO_EW_ROT_TOK: &[SigArg] = &[
    a("x", Buffer),
    a("y", Buffer),
    a("ys", I64),
    a("w", Buffer),
    a("eps", F32),
    a("t", Buffer),
    a("cs", Buffer),
    a("hs", Buffer),
    a("a", Buffer),
    a("as", Buffer),
    a("m", I32),
    a("n", I32),
    a("krot", I32),
];

// ------------------------------------------------------------------- the null plugin (bench)

/// `saber_noop()`: no arguments at all. The kernarg segment holds only the HIP runtime's hidden
/// block, so the host writes **zero** bytes and the dispatch is the cheapest one this hardware
/// can be asked to run. An empty signature is a declaration, not an omission: it is what makes
/// the loader's manifest check assert that the blob takes no explicit argument either.
pub const SIG_NOOP: &[SigArg] = &[];

/// `saber_noop_touch(unsigned char *p)`: one pointer, one byte read-modify-written, so a chain of
/// them is a real RAW dependency with the smallest footprint a dependency can have.
pub const SIG_NOOP_TOUCH: &[SigArg] = &[a("p", Buffer)];

// ------------------------------------------------------------------- T6's shadow digest

/// `saber_race_digest_regions(desc, extra, out, extra_bytes, n, parts)`: `n` (address, bytes)
/// regions of the `desc` table plus the per-launch `extra` window, `parts` partial digests each
/// (`hip/race_digest.hip`, `crate::race_digest`).
pub const SIG_RACE_DIGEST_REGIONS: &[SigArg] = &[
    p("desc", 8),
    p("extra", 1),
    p("out", 8),
    a("extra_bytes", I64),
    a("n", I32),
    a("parts", I32),
];

/// `saber_race_digest_kv(block_table, seq_lens, kbase, vbase, out, per_layer, block_bytes, reqs,
/// bt_stride, page_tokens, parts, blocks)`: the drafter K/V pages a proposal read.
pub const SIG_RACE_DIGEST_KV: &[SigArg] = &[
    p("block_table", 4),
    p("seq_lens", 4),
    p("kbase", 1),
    p("vbase", 1),
    p("out", 8),
    a("per_layer", I64),
    a("block_bytes", I64),
    a("reqs", I32),
    a("bt_stride", I32),
    a("page_tokens", I32),
    a("parts", I32),
    a("blocks", I32),
];

// ------------------------------------------------------------------- the vision tower (D1)

/// `vis_cast_f32_bf16(x, y, n)`: the tower's input cast, `pixel_values.to(bf16)`.
pub const SIG_VIS_CAST: &[SigArg] = &[p("x", 4), p("y", 2), a("n", I64)];

/// `vis_gemm_bf16_bias{,_gelu_tanh,_gelu_erf,_residual}(a, w, bias, res, c, m, n, k)`: every
/// linear of the tower (and the patch-embed Conv3d, which is one). `res` is read only by the
/// `_residual` instantiation and may alias `c`; the others are handed a named null.
pub const SIG_VIS_GEMM: &[SigArg] = &[
    p("a", 2),
    p("w", 2),
    p("bias", 2),
    p("res", 2),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// FN5 `fn_hc_down_{mix,expand,combine,final}_mf{1,2}`: the HC mix's first launch (norm [+ fused
/// combine / expand] + the down/inject GEMM over K slices + the last arrival's slice sum, silu and
/// injection logits). `src` is the stream [M, 10240] (the embedding [M, 2560] when expanding).
pub const SIG_FN_HC_DOWN: &[SigArg] = &[
    p("src", 2),
    p("blk", 2),
    p("inj", 2),
    p("hid", 2),
    p("nw", 2),
    p("wdn", 2),
    p("wij", 2),
    p("part", 4),
    p("rr", 4),
    p("cnt", 4),
    p("lora", 2),
    p("inj_out", 2),
    a("m", I32),
    a("sps", I32),
    a("eps", F32),
];

/// FN85 `fn_hc_down_combine_v2_<cfg>_rp` (+ the gate controls `_rps` / `_rpo`): the combine down
/// with the exact two-rank reduce in its prologue -- [`SIG_FN_HC_DOWN`] (`blk` = THIS rank's
/// partial) + the reduce's channel: the peer's / my mailbox, the reduced output (null = not
/// stored), the peer's / my flag words, my channel counter [calls, readers], the slot stride and
/// the one-shot's drain / acquire modes.
pub const SIG_FN_HC_DOWN_RP: &[SigArg] = &[
    p("src", 2),
    p("blk", 2),
    p("inj", 2),
    p("hid", 2),
    p("nw", 2),
    p("wdn", 2),
    p("wij", 2),
    p("part", 4),
    p("rr", 4),
    p("cnt", 4),
    p("lora", 2),
    p("inj_out", 2),
    a("m", I32),
    a("sps", I32),
    a("eps", F32),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    p("out", 2),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("ctl", Buffer),
    a("slot_stride16", I32),
    a("drain", I32),
    a("acq", I32),
];

/// FN5 `fn_hc_up_{bf16,q8,q8t,q8_dbg}_mf{1,2}`: the HC mix's second launch (the K = 320 up GEMM,
/// the sigmoid gate and gated mean -> the bf16 block input [+ the e4m3 per-token quant]).
pub const SIG_FN_HC_UP: &[SigArg] = &[
    p("hid", 2),
    p("rr", 4),
    p("nw", 2),
    p("lora", 2),
    p("wup", 2),
    p("bi", 2),
    p("q8", 1),
    p("scale", 4),
    p("amax", 4),
    p("cnt", 4),
    p("gdbg", 2),
    a("m", I32),
];

/// FN108 `fn_hc_down_<mode>_v2_<cfg>_h8`: [`SIG_FN_HC_DOWN`] with `wdn` the e4m3 record (codes
/// `[320, 10240]` then f32 scales `[320, 80]`, the reference's hc8m down) -- the same kernarg layout.
pub const SIG_FN_HC8_DOWN: &[SigArg] = &[
    p("src", 2),
    p("blk", 2),
    p("inj", 2),
    p("hid", 2),
    p("nw", 2),
    p("wdn", 1),
    p("wij", 2),
    p("part", 4),
    p("rr", 4),
    p("cnt", 4),
    p("lora", 2),
    p("inj_out", 2),
    a("m", I32),
    a("sps", I32),
    a("eps", F32),
];

/// FN108 `fn_hc_up_<kind>_v2_<cfg>_h8`: [`SIG_FN_HC_UP`] with `wup` the e4m3 record (codes
/// `[10240, 320]` then f32 scales `[10240, 4]`, groups of 80).
pub const SIG_FN_HC8_UP: &[SigArg] = &[
    p("hid", 2),
    p("rr", 4),
    p("nw", 2),
    p("lora", 2),
    p("wup", 1),
    p("bi", 2),
    p("q8", 1),
    p("scale", 4),
    p("amax", 4),
    p("cnt", 4),
    p("gdbg", 2),
    a("m", I32),
];

/// FN37 `fn_ar_exact_hcrr_bf16`: the two-rank exact reduce one block per row (the reference's exact
/// push / flag / spin / vadd, word for word) with the NEXT mix's HC combine + stream norm in its
/// epilogue: `out` = the block output, `xo` = x' `[m, 10,240]` bf16, `rro` = rr `[m, 4]` fp32
/// (`_rev_` = the gate's control: the norm's butterfly reversed). Grid m x 320 threads.
pub const SIG_FN_AR_HCRR: &[SigArg] = &[
    a("in", Buffer),
    a("peer_scratch", Buffer),
    a("my_scratch", Buffer),
    a("out", Buffer),
    a("peer_flags", Buffer),
    a("my_flags", Buffer),
    a("seq_ctrs", Buffer),
    p("res", 2),
    p("inj", 2),
    p("xo", 2),
    p("rro", 4),
    a("slot_stride16", I32),
    a("eps", F32),
    a("drain", I32),
    a("acq", I32),
];

/// FN5 `fn_hc_combine(res, blk, inj, out, m)`: the standalone combine (layer 1, before the PLE).
pub const SIG_FN_HC_COMBINE: &[SigArg] = &[
    p("res", 2),
    p("blk", 2),
    p("inj", 2),
    p("out", 2),
    a("m", I32),
];

/// FN5 `fn_wmma_probe(a, b, c, d)`: gate only -- one bf16 WMMA per 16x16 tile.
pub const SIG_FN_WMMA_PROBE: &[SigArg] = &[p("a", 2), p("b", 2), p("c", 4), p("d", 4)];

/// FN21 `fn_hc_sig_probe(direct, viatab)`: gate only -- the up's sigmoid table vs fn_sigmoid over
/// all 65,536 bf16 values.
pub const SIG_FN_HC_SIG_PROBE: &[SigArg] = &[p("direct", 4), p("viatab", 4)];

/// FN5 `fn_bgemm_sk_m{16,32,64}(a, w, part, cnt, c, m, n, k)`: the split-K bf16 band (splits =
/// gridDim.z; `part` f32 [splits, M, N] and `cnt` i32 per output tile when splits > 1).
pub const SIG_FN_BGEMM_SK: &[SigArg] = &[
    p("a", 2),
    p("w", 2),
    p("part", 4),
    p("cnt", 4),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// FN61c `fn_t8_a8w8_bs128_ldc(a, w, as, ws, c, m, n, k, ldc)`: `dflash_gemm_a8w8_bs128_mtn128k128`'s
/// body with the store's row stride an argument (the W8 prefix of a merged projection).
pub const SIG_FN_T8_A8W8_LDC: &[SigArg] = &[
    p("a", 1),
    p("w", 1),
    p("as", 4),
    p("ws", 4),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("ldc", I32),
];

/// FN61g `fn_t8r_a8w8_gather_m16n64(a, w, as, ws, cand, c, m, ncols, k, rows, kk, cstride, ldc)`:
/// the drafter's exact rerank scored from the preshuffled W8 head (each candidate column the W8
/// head GEMM's logit, bit for bit).
pub const SIG_FN_T8R_GATHER: &[SigArg] = &[
    p("a", 1),
    p("w", 1),
    p("as", 4),
    p("ws", 4),
    p("cand", 4),
    p("c", 2),
    a("m", I32),
    a("ncols", I32),
    a("k", I32),
    a("rows", I32),
    a("kk", I32),
    a("cstride", I32),
    a("ldc", I32),
];

/// FN61c `fn_t8_bgemm_m16_ldc(a, w, c, m, n, k, ldc)`: `fn_bgemm_sk_m16` at one split with the
/// store's row stride an argument (the bf16 suffix of a merged projection).
pub const SIG_FN_T8_BGEMM_LDC: &[SigArg] = &[
    p("a", 2),
    p("w", 2),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("ldc", I32),
];

/// FN5 `fn_bgemm_lds_l*(a, w, c, m, n, k)`: the D6 LDS tile with no bias.
pub const SIG_FN_BGEMM: &[SigArg] = &[
    p("a", 2),
    p("w", 2),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// FN5 `fn_gdn_norm_bf16_t256(x, z, zs, w, out, n, eps)`: `gdn_norm_quant`'s pre-quant value, bf16.
pub const SIG_FN_GDN_NORM_BF16: &[SigArg] = &[
    p("x", 2),
    p("z", 2),
    a("zs", I64),
    p("w", 2),
    p("out", 2),
    a("n", I32),
    a("eps", F32),
];

/// FN86 `fn_gdn_norm_q8_sig_t256[_rc](x, z, zs, w, out, q, scales, n, eps)`: the sigmoid-gated
/// norm + `dflash_act_quant_g128`'s quant of its bf16 output (e4m3 `q`, fp32 `scales` a 128 group).
pub const SIG_FN_GDN_NORM_Q8: &[SigArg] = &[
    p("x", 2),
    p("z", 2),
    a("zs", I64),
    p("w", 2),
    p("out", 2),
    p("q", 1),
    p("scales", 4),
    a("n", I32),
    a("eps", F32),
];

/// FN5 `fn_attn_gate_bf16_t256(core, gate, out, n, hd, gs)`: `attn_gate_quant`'s pre-quant value.
pub const SIG_FN_ATTN_GATE_BF16: &[SigArg] = &[
    p("core", 2),
    p("gate", 2),
    p("out", 2),
    a("n", I32),
    a("hd", I32),
    a("gs", I32),
];

/// FN86b `fn_attn_gate_q8_t256[_rc](core, gate, out, q, scales, n, hd, gs)`: the attention gate +
/// `dflash_act_quant_g128`'s quant of its bf16 output (e4m3 `q`, fp32 `scales` a 128 group).
pub const SIG_FN_ATTN_GATE_Q8: &[SigArg] = &[
    p("core", 2),
    p("gate", 2),
    p("out", 2),
    p("q", 1),
    p("scales", 4),
    a("n", I32),
    a("hd", I32),
    a("gs", I32),
];

/// FN86c family 5 `fn86_gdn_fused_update_nq8_fc_*[_rc]`: the fc fused update's thirty-six
/// arguments unchanged, then the gated norm + act quant's (`nz` = the z gate at row stride `nzs`,
/// `nw8` the norm weight, `nout` the norm's bf16 output, `nq` / `nsc` the e4m3 + g128 scales).
pub const SIG_FN86_GDN_FUSED_NQ8: &[SigArg] = &[
    a("x", Buffer),
    a("xpitch", I64),
    a("wgt", Buffer),
    a("bias", Buffer),
    a("cstate", Buffer),
    a("cs_seq", I64),
    a("cs_dim", I64),
    a("cs_tok", I64),
    a("state_len_max", I32),
    a("cache_idx", Buffer),
    a("ci_stride", I64),
    a("naccept", Buffer),
    a("cu", Buffer),
    a("n_seqs", I32),
    a("h", I32),
    a("hg", I32),
    a("maxq", I32),
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("scale", F32),
    a("sp_thr", F32),
    a("bar", Buffer),
    a("num_wg", I32),
    a("o_rows", I32),
    p("nz", 2),
    a("nzs", I64),
    p("nw8", 2),
    p("nout", 2),
    p("nq", 1),
    p("nsc", 4),
    a("neps8", F32),
];

/// FN86c family 5 `fn86_gdn_recurrent_nq8_fc_*[_rc]`: the fc recurrent update's arguments
/// unchanged, then the norm + quant's (as `SIG_FN86_GDN_FUSED_NQ8`).
pub const SIG_FN86_GDN_REC_NQ8: &[SigArg] = &[
    a("q", Buffer),
    a("k", Buffer),
    a("v", Buffer),
    a("av", Buffer),
    a("bv", Buffer),
    a("ab_stride", I64),
    a("a_log", Buffer),
    a("dt_bias", Buffer),
    a("state", Buffer),
    a("st_slot", I64),
    a("st_head", I64),
    a("o", Buffer),
    a("cu", Buffer),
    a("sidx", Buffer),
    a("sidx_stride", I64),
    a("naccept", Buffer),
    a("zg", Buffer),
    a("nw", Buffer),
    a("neps", F32),
    a("nact", I32),
    a("h", I32),
    a("hg", I32),
    a("scale", F32),
    a("sp_thr", F32),
    p("nz", 2),
    a("nzs", I64),
    p("nw8", 2),
    p("nout", 2),
    p("nq", 1),
    p("nsc", 4),
    a("neps8", F32),
];

/// FN86c family 2 `fn86_t8_q8w8_sfx_ldc[_rc](a, w8, ws8, c, m, n8, k, ldc, nbx8, w16, cs, ns)`:
/// the W8 GEMM with the act quant in its A staging (bf16 `a`) at grid x < `nbx8`, family 4's bf16
/// suffix GEMM (`w16`, `ns` rows, into `cs`) at x >= `nbx8`, both at row stride `ldc`.
pub const SIG_FN86_Q8W8_SFX: &[SigArg] = &[
    p("a", 2),
    p("w8", 1),
    p("ws8", 4),
    p("c", 2),
    a("m", I32),
    a("n8", I32),
    a("k", I32),
    a("ldc", I32),
    a("nbx8", I32),
    p("w16", 2),
    p("cs", 2),
    a("ns", I32),
];

/// FN86b `fn86_q8_bgemm_m16_ldc[_rc](a, q, as, m, k, nqb, w, c, n, ldc, nbx)`: the act quant of
/// `a` (workgroups `[0, nqb)`) and `fn_t8_bgemm_m16_ldc` of the same `a` (the rest, x fastest over
/// `nbx`) in ONE launch.
pub const SIG_FN86_Q8_BGEMM_LDC: &[SigArg] = &[
    p("a", 2),
    p("q", 1),
    p("as", 4),
    a("m", I32),
    a("k", I32),
    a("nqb", I32),
    p("w", 2),
    p("c", 2),
    a("n", I32),
    a("ldc", I32),
    a("nbx", I32),
];

/// FN6 `fn_kv_append_ring_h256_bf16`: bf16 K|V into the saber paged layout (+ the QSA indexer's
/// raw-key ring row and its three rope positions when `ring_slot[t] >= 0`).
pub const SIG_FN_KV_APPEND_RING: &[SigArg] = &[
    p("key", 2),
    a("key_stride", I64),
    p("value", 2),
    a("value_stride", I64),
    p("kv", 2),
    p("slot", 8),
    a("block_stride", I64),
    a("page_stride", I64),
    a("head_stride", I64),
    p("ikey", 2),
    a("ikey_stride", I64),
    p("ring", 2),
    p("ring_pos", 8),
    p("ring_slot", 8),
    p("pos", 8),
    a("m", I32),
];

/// FN68 `fn_kv_append_ring_h256_fp8`: [`SIG_FN_KV_APPEND_RING`]'s job with K|V stored as e4m3
/// bytes (value / scale, the hardware convert) -- the page pointer is bytes, and the per-tensor
/// K and V scales ride at the end by value.
pub const SIG_FN_KV_APPEND_RING_FP8: &[SigArg] = &[
    p("key", 2),
    a("key_stride", I64),
    p("value", 2),
    a("value_stride", I64),
    p("kv", 1),
    p("slot", 8),
    a("block_stride", I64),
    a("page_stride", I64),
    a("head_stride", I64),
    p("ikey", 2),
    a("ikey_stride", I64),
    p("ring", 2),
    p("ring_pos", 8),
    p("ring_slot", 8),
    p("pos", 8),
    a("m", I32),
    a("k_scale", F32),
    a("v_scale", F32),
];

/// ATTN-OPT-3 `fn_attnfuse_front_h256_bf16` (`--fn-fuse f6`): `fn_qk_norm_rope_h256_rot64_mi1_v2`'s
/// arguments (sec_t dropped), then `fn_idx_q_compress_h128`'s not already named, then
/// `fn_kv_append_ring_h256_*`'s not already named (the bf16 form ignores the two scales).
pub const SIG_FN_ATTNFUSE_FRONT_BF16: &[SigArg] = &[
    p("qkv", 2),
    p("qw", 2),
    p("kw", 2),
    p("cos_sin", 2),
    p("pos", 8),
    p("q_out", 2),
    p("k_out", 2),
    a("m", I32),
    a("nq", I32),
    a("nkv", I32),
    a("qkv_n", I32),
    a("eps", F32),
    a("sec_h", I32),
    a("sec_w", I32),
    p("iqk", 2),
    a("iqk_stride", I64),
    p("iqw", 2),
    p("ikw", 2),
    p("row_seq", 4),
    p("row_lpos", 8),
    p("seq_row0", 4),
    p("seq_pos0", 8),
    p("ring_row0", 4),
    p("ring", 2),
    p("ring_pos", 8),
    a("ring_rows", I32),
    p("block_table", 4),
    a("max_blocks", I32),
    p("iq_out", 2),
    p("comp", 2),
    p("value", 2),
    a("value_stride", I64),
    p("kv", 2),
    p("slot", 8),
    a("block_stride", I64),
    a("page_stride", I64),
    a("head_stride", I64),
    p("ikey", 2),
    a("ikey_stride", I64),
    p("ring_slot", 8),
    a("k_scale", F32),
    a("v_scale", F32),
];

/// ATTN-OPT-3 `fn_attnfuse_front_h256_fp8`: [`SIG_FN_ATTNFUSE_FRONT_BF16`] over e4m3 K/V pages.
pub const SIG_FN_ATTNFUSE_FRONT_FP8: &[SigArg] = &[
    p("qkv", 2),
    p("qw", 2),
    p("kw", 2),
    p("cos_sin", 2),
    p("pos", 8),
    p("q_out", 2),
    p("k_out", 2),
    a("m", I32),
    a("nq", I32),
    a("nkv", I32),
    a("qkv_n", I32),
    a("eps", F32),
    a("sec_h", I32),
    a("sec_w", I32),
    p("iqk", 2),
    a("iqk_stride", I64),
    p("iqw", 2),
    p("ikw", 2),
    p("row_seq", 4),
    p("row_lpos", 8),
    p("seq_row0", 4),
    p("seq_pos0", 8),
    p("ring_row0", 4),
    p("ring", 2),
    p("ring_pos", 8),
    a("ring_rows", I32),
    p("block_table", 4),
    a("max_blocks", I32),
    p("iq_out", 2),
    p("comp", 2),
    p("value", 2),
    a("value_stride", I64),
    p("kv", 1),
    p("slot", 8),
    a("block_stride", I64),
    a("page_stride", I64),
    a("head_stride", I64),
    p("ikey", 2),
    a("ikey_stride", I64),
    p("ring_slot", 8),
    a("k_scale", F32),
    a("v_scale", F32),
];

/// ATTN-OPT-3 `fn_attnfuse_combine_gate_q8_h256(const SABERArgs a, int splits, int tile, gate, gated,
/// q, scales, n, hd, gs)` (`--fn-fuse f7`): the combine's arguments, then the family-1 gate's
/// (`core` = the struct's `dst`).
pub const SIG_FN_ATTNFUSE_COMBINE_GATE_Q8: &[SigArg] = &[
    a("a", SABER_ARGS),
    a("splits", I32),
    a("tile", I32),
    p("gate", 2),
    p("gated", 2),
    p("q", 1),
    p("scales", 4),
    a("n", I32),
    a("hd", I32),
    a("gs", I32),
];

/// FN6 `fn_idx_q_compress_h128`: the QSA indexer's q norm + rope and the group-of-4 pooled,
/// normed, roped compressed-key store (docs/EXEC.md section 13, the indexer-cache contract).
pub const SIG_FN_IDX_Q_COMPRESS: &[SigArg] = &[
    p("qk", 2),
    a("qk_stride", I64),
    p("qw", 2),
    p("kw", 2),
    p("cos_sin", 2),
    p("pos", 8),
    p("row_seq", 4),
    p("row_lpos", 8),
    p("seq_row0", 4),
    p("seq_pos0", 8),
    p("ring_row0", 4),
    p("ring", 2),
    p("ring_pos", 8),
    a("ring_rows", I32),
    p("block_table", 4),
    a("max_blocks", I32),
    p("q_out", 2),
    p("comp", 2),
    a("m", I32),
    a("sec_h", I32),
    a("sec_w", I32),
];

/// FN19 `fn_qsa_score_r32`: the indexer's block scores (flashnext_qsa.hip).
pub const SIG_FN_QSA_SCORE: &[SigArg] = &[
    p("q", 2),
    p("comp", 2),
    p("bt", 4),
    a("max_blocks", I32),
    p("pos", 8),
    p("scores", 4),
    a("sstride", I64),
    a("m", I32),
    a("rows_per_seq", I32),
    a("rt", I32),
];

/// FN19 `fn_qsa_topk512`: the exact top-512 per row (ties -> the lower group), ascending.
pub const SIG_FN_QSA_TOPK: &[SigArg] = &[
    p("scores", 4),
    a("sstride", I64),
    p("pos", 8),
    p("sel", 4),
    a("m", I32),
];

/// FN19 `fn_qsa_union`: the per-CTA union key list (slot + row mask) and its key count.
pub const SIG_FN_QSA_UNION: &[SigArg] = &[
    p("sel", 4),
    p("pos", 8),
    a("m", I32),
    a("R", I32),
    a("cps", I32),
    p("bt", 4),
    a("max_blocks", I32),
    p("who", 4),
    a("wstride", I64),
    p("kslot", 4),
    p("kmask", 4),
    p("kbits", 1),
    p("nkeys", 4),
    a("kstride", I64),
];

/// FN33 `fn_draft_x_gather(x, sel, xp, rows)`: the selected rows of the head input, zero-padded.
pub const SIG_FN_DRAFT_X_GATHER: &[SigArg] = &[p("x", 2), p("sel", 4), p("xp", 2), a("rows", I32)];

/// FN33 `fn_draft_head_int2_quant(w, wq, s, zs, n)`: the int2 pack of the rank's head half.
pub const SIG_FN_DRAFT_HEAD_INT2_QUANT: &[SigArg] =
    &[p("w", 2), p("wq", 1), p("s", 2), p("zs", 2), a("n", I32)];

/// FN33 `fn_draft_wgather(head, cand, wc, rows, kk, cstride)`: the candidates' head rows.
pub const SIG_FN_DRAFT_WGATHER: &[SigArg] = &[
    p("head", 2),
    p("cand", 4),
    p("wc", 2),
    a("rows", I32),
    a("kk", I32),
    a("cstride", I32),
];

/// FN33 `fn_draft_argmax_cand(sc, cand, sel, part, kk, ldc, cstride, tie_high)`: the reranked
/// candidates' argmax as FN22's split-0 partial.
pub const SIG_FN_DRAFT_ARGMAX_CAND: &[SigArg] = &[
    p("sc", 2),
    p("cand", 4),
    p("sel", 4),
    p("part", 8),
    a("kk", I32),
    a("ldc", I32),
    a("cstride", I32),
    a("tie_high", I32),
];

/// FN86d family 10 `fn86_draft_xg_sum_q8[_rc](x, sel, xp, rows, xs, q, qs)`: `fn_draft_x_gather`,
/// `dflash_head_group_sum_g128` (rows `fn33::XROWS`, stride `NG`) and `dflash_act_quant_g128` of
/// the first `rows` gathered rows in one launch.
pub const SIG_FN86_DRAFT_XG_SUM_Q8: &[SigArg] = &[
    p("x", 2),
    p("sel", 4),
    p("xp", 2),
    a("rows", I32),
    p("xs", 4),
    p("q", 1),
    p("qs", 4),
];

/// FN86d family 10 `fn86_topk_gather_cand[_rc](x, thresh, vals, idx, bidx, cand, rows, n, k)`:
/// `dflash_topk_f32_gather_k80` + `dflash_head_cand_gather`, one workgroup a row.
pub const SIG_FN86_TOPK_K80_CAND: &[SigArg] = &[
    p("x", 4),
    p("thresh", 8),
    p("vals", 4),
    p("idx", 4),
    p("bidx", 4),
    p("cand", 4),
    a("rows", I32),
    a("n", I32),
    a("k", I32),
];

/// FN86d family 10 `fn86_draft_argmax_cand_pack[_rc](sc, cand, sel, part, kk, ldc, cstride, embh,
/// pay, half, rank, tie_high)`: `fn_draft_argmax_cand` + `fn_draft_pack`, one launch.
pub const SIG_FN86_DRAFT_ARGMAX_CAND_PACK: &[SigArg] = &[
    p("sc", 2),
    p("cand", 4),
    p("sel", 4),
    p("part", 8),
    a("kk", I32),
    a("ldc", I32),
    a("cstride", I32),
    p("embh", 2),
    p("pay", 2),
    a("half", I32),
    a("rank", I32),
    a("tie_high", I32),
];

/// LR-N `fn_couple_acp(sc, cand, sel, part, kk, ldc, cstride, embh, pay, half, rank, inb)`: FN86d's
/// fused argmax + pack with the coupled draft's 32-bit key (the chain's input buffer `inb` carries
/// the pass counter + each row's noise key base, position, 1 / T and top_k).
pub const SIG_FN_COUPLE_ACP: &[SigArg] = &[
    p("sc", 2),
    p("cand", 4),
    p("sel", 4),
    p("part", 8),
    a("kk", I32),
    a("ldc", I32),
    a("cstride", I32),
    p("embh", 2),
    p("pay", 2),
    a("half", I32),
    a("rank", I32),
    p("inb", 4),
];

/// LR-N `fn_couple_select(red, emb, ids, ctr, rows)`: the pair's pick over the 32-bit keys.
pub const SIG_FN_COUPLE_SELECT: &[SigArg] = &[
    p("red", 2),
    p("emb", 2),
    p("ids", 4),
    p("ctr", 4),
    a("rows", I32),
];

/// FN22 `fn_draft_argmax_part(lg, sel, part, half, tie_high)`: split s of row `sel[r]`'s argmax.
pub const SIG_FN_DRAFT_ARGMAX: &[SigArg] = &[
    p("lg", 2),
    p("sel", 4),
    p("part", 8),
    a("half", I32),
    a("tie_high", I32),
];

/// FN22 `fn_draft_pack(part, sel, embh, pay, half, rank, tie_high)`: the exchange payload row.
pub const SIG_FN_DRAFT_PACK: &[SigArg] = &[
    p("part", 8),
    p("sel", 4),
    p("embh", 2),
    p("pay", 2),
    a("half", I32),
    a("rank", I32),
    a("tie_high", I32),
];

/// FN22 `fn_draft_select(red, emb, ids, ctr, rows, tie_high)`: the pair's pick.
pub const SIG_FN_DRAFT_SELECT: &[SigArg] = &[
    p("red", 2),
    p("emb", 2),
    p("ids", 4),
    p("ctr", 4),
    a("rows", I32),
    a("tie_high", I32),
];

/// FN57 `fn_pick_top64_bf16(lg, out, rows, n, ld, base)`: each logits row's seam top-64 on the
/// card (512 B a row: 64 x (id + base, f32 bits)).
pub const SIG_FN_PICK_TOP64: &[SigArg] = &[
    p("lg", 2),
    p("out", 4),
    a("rows", I32),
    a("n", I32),
    a("ld", I32),
    a("base", I32),
];

/// FN53c `fn_xmiss_e256(ids, n, gu, dn, lo, mbox, layer)`: the valve's device-side miss flag --
/// a routed record of this rank whose xptr entry is 0 raises the coherent host mailbox, the
/// kernel waits the host's answer and patches the layer's tables (one workgroup).
pub const SIG_FN_XMISS: &[SigArg] = &[
    p("ids", 4),
    a("n", I32),
    p("gu", 8),
    p("dn", 8),
    a("lo", I32),
    p("mbox", 4),
    a("layer", I32),
];

/// FN98 `fn_gate98(tab, idx)`: the in-graph layer gate -- entry `idx` of the gate table (word
/// address u64, value u32, pad u32); one thread spins at system scope until the word reaches the
/// value (0 = pass).
pub const SIG_FN_GATE98: &[SigArg] = &[p("tab", 16), a("idx", I32)];

/// FN119 `fn119_draw_walk(seam_a, seam_b, rp, sq, out, rowout, nseq)`: the device twin of the
/// host sampler's draw + F40's drawn-row walk (`flashnext_dstep.hip`): the two ranks' seams
/// (64 x (id, f32 bits) a row), the row blocks (8 u32), the sequence blocks (8 u32), the result
/// blocks (16 u32), the per-row results (2 u32).
pub const SIG_FN119_DRAW_WALK: &[SigArg] = &[
    p("seam_a", 4),
    p("seam_b", 4),
    p("rp", 4),
    p("sq", 4),
    p("out", 4),
    p("rowout", 4),
    a("nseq", I32),
];

/// FN120 `fn120_res_v<V>_l<L>_w<W>(ctr, out, g, timeout)`: the residency sweep's check-in kernel at
/// a forced budget (`flashnext_df120.hip`; a measurement object, nothing in the engine launches it).
pub const SIG_FN120_RES: &[SigArg] = &[p("ctr", 4), p("out", 4), a("g", I32), a("timeout", I64)];

/// FN120 `fn120_df_run(tab, ntasks, ticket, ctr, checkin, payload, stats, stamps, g, flags,
/// timeout)`: the persistent dataflow runtime (32-byte tasks, u32 counters, u64 REALTIME stamps).
pub const SIG_FN120_DF_RUN: &[SigArg] = &[
    p("tab", 32),
    a("ntasks", I32),
    p("ticket", 4),
    p("ctr", 4),
    p("checkin", 4),
    p("payload", 4),
    p("stats", 4),
    p("stamps", 8),
    a("g", I32),
    a("flags", I32),
    a("timeout", I64),
];

/// FN120 `fn120_df_pair(tab, ntasks, ticket, ctr, checkin, pd, dims, stats, stamps, g, flags,
/// timeout)`: the runtime's loop over the family-1 pair's ROW / TILE tasks (dims = M, N, K, n, hd,
/// gs, ldc).
pub const SIG_FN120_DF_PAIR: &[SigArg] = &[
    p("tab", 32),
    a("ntasks", I32),
    p("ticket", 4),
    p("ctr", 4),
    p("checkin", 4),
    p("pd", 64),
    p("dims", 4),
    p("stats", 4),
    p("stamps", 8),
    a("g", I32),
    a("flags", I32),
    a("timeout", I64),
];

/// FN120c `fn120_df_gdn_b1(st, tk, ntasks, imgs, ctl, nstages, timeout)`: the GDN mixer tail
/// (merged projection -> fused update + norm + quant -> out projection) of a b1 verify as ONE
/// persistent dataflow launch over a host-built table (`flashnext_df120b.hip`; 32-byte stages,
/// 16-byte tasks, the bodies' kernarg images, u32 control words).
pub const SIG_FN120_DF_GDN: &[SigArg] = &[
    p("st", 32),
    p("tk", 16),
    a("ntasks", I32),
    p("imgs", 1),
    p("ctl", 4),
    a("nstages", I32),
    a("timeout", I64),
];

/// FN120 `fn120_stamp(st, node)`: the launch train's comparison node (REALTIME min start / max end).
pub const SIG_FN120_STAMP: &[SigArg] = &[p("st", 8), a("node", I32)];

/// FN119 `fn119_expf_probe(out, base, per)`: the exhaustive expf gate's per-workgroup hash.
pub const SIG_FN119_EXPF_PROBE: &[SigArg] = &[p("out", 8), a("base", I32), a("per", I32)];

/// FN122c (`flashnext_hc8p.hip`): `fn122_hc8_dequant(rec, out, rows, cols, g)` -- an hc8 record
/// (e4m3 codes [rows, cols] + f32 scales [rows, cols / g]) to the bf16 matrix its `_h8` symbols
/// multiply.
pub const SIG_FN122_HC8_DEQUANT: &[SigArg] = &[
    p("rec", 1),
    p("out", 2),
    a("rows", I32),
    a("cols", I32),
    a("g", I32),
];

/// P27-W2-2 (`p27_dstep.hip`, `crate::p27d`): `p27d_cand_seam(vals, ids, win_mine, win_peer,
/// scratch, vocab_base, top_k, rank, call, rows, rows_max)` -- the 27B drafter's candidate seam
/// merged on the card (`--engine-device-step cand`).
pub const SIG_P27D_CAND_SEAM: &[SigArg] = &[
    p("vals", 4),
    p("ids", 8),
    p("win_mine", 8),
    p("win_peer", 8),
    p("scratch", 4),
    a("vocab_base", I64),
    a("top_k", I32),
    a("rank", I32),
    a("call", I32),
    a("rows", I32),
    a("rows_max", I32),
];

/// FN119 round 2 (`flashnext_ahead.hip`): `fn119a_push(seam, win_peer, ctr, rows, rows_max)`.
pub const SIG_FN119A_PUSH: &[SigArg] = &[
    p("seam", 4),
    p("win_peer", 4),
    p("ctr", 4),
    a("rows", I32),
    a("rows_max", I32),
];

/// FN119 round 2: `fn119a_walk(seam_a, win, ctr, rp, sq, out, rowout, nseq, rows_max)`.
pub const SIG_FN119A_WALK: &[SigArg] = &[
    p("seam_a", 4),
    p("win", 4),
    p("ctr", 4),
    p("rp", 4),
    p("sq", 4),
    p("out", 4),
    p("rowout", 4),
    a("nseq", I32),
    a("rows_max", I32),
];

/// FN119 round 2: `fn119a_emb(out, rowout, embh, pay, w, nseq, half, rank)`.
pub const SIG_FN119A_EMB: &[SigArg] = &[
    p("out", 4),
    p("rowout", 4),
    p("embh", 2),
    p("pay", 2),
    a("w", I32),
    a("nseq", I32),
    a("half", I32),
    a("rank", I32),
];

/// FN119 round 2: `fn119a_gather(out, strm, prev, w, rowh)`.
pub const SIG_FN119A_GATHER: &[SigArg] = &[
    p("out", 4),
    p("strm", 2),
    p("prev", 2),
    a("w", I32),
    a("rowh", I32),
];

/// FN119 round 2: `fn119a_sel(out, var, dst, stride16, n16)`.
pub const SIG_FN119A_SEL: &[SigArg] = &[
    p("out", 4),
    p("var", 16),
    p("dst", 16),
    a("stride16", I32),
    a("n16", I32),
];

/// FN119c: `fn119b_copy(src, dst, n16)`.
pub const SIG_FN119B_COPY: &[SigArg] = &[p("src", 16), p("dst", 16), a("n16", I32)];

/// FN119d: `fn119d_vemb(out, ids, embh, pay, w, nseq, half, rank, ids_stride)`.
pub const SIG_FN119D_VEMB: &[SigArg] = &[
    p("out", 4),
    p("ids", 4),
    p("embh", 2),
    p("pay", 2),
    a("w", I32),
    a("nseq", I32),
    a("half", I32),
    a("rank", I32),
    a("ids_stride", I32),
];

/// FN119c: `fn119b_compose(out, var, dst, runs, stride_words)`.
pub const SIG_FN119B_COMPOSE: &[SigArg] = &[
    p("out", 4),
    p("var", 4),
    p("dst", 4),
    p("runs", 4),
    a("stride_words", I32),
];

/// FN119 round 2: `fn119a_backup(rowout, lg, stg, row16)`.
pub const SIG_FN119A_BACKUP: &[SigArg] =
    &[p("rowout", 4), p("lg", 16), p("stg", 16), a("row16", I32)];

/// FN6 `fn_ple_gate_x(x, key, value, wk, wq, wc, gv, xc)`: the PLE gate on a materialized stream.
pub const SIG_FN_PLE_GATE_X: &[SigArg] = &[
    p("x", 2),
    p("key", 2),
    p("value", 2),
    p("wk", 2),
    p("wq", 2),
    p("wc", 2),
    p("gv", 2),
    p("xc", 2),
];

/// FN6 `fn_ple_gate_combine`: the PLE gate with FN5's delayed HC combine fused in (writes `xs`).
pub const SIG_FN_PLE_GATE_COMBINE: &[SigArg] = &[
    p("res", 2),
    p("blk", 2),
    p("inj", 2),
    p("key", 2),
    p("value", 2),
    p("wk", 2),
    p("wq", 2),
    p("wc", 2),
    p("xs", 2),
    p("gv", 2),
    p("xc", 2),
];

/// FN6 `fn_ple_conv`: the dilated short conv, silu, the PLE add into the stream, the next history.
pub const SIG_FN_PLE_CONV: &[SigArg] = &[
    p("xc", 2),
    p("gv", 2),
    p("xs", 2),
    p("cw", 2),
    p("row_seq", 4),
    p("seq_row0", 4),
    p("seq_rows", 4),
    p("st_in", 2),
    p("st_out", 2),
    p("drop", 4),
    p("has_init", 4),
    a("hist", I32),
    p("out", 2),
    p("ple", 2),
];

/// FN6 `fn_ple_rows_fp8(rows, scale, emb, m)`: the GATED fp8 PLE table's rows -> bf16.
pub const SIG_FN_PLE_ROWS_FP8: &[SigArg] = &[p("rows", 1), p("scale", 4), p("emb", 2), a("m", I32)];

/// FN6 `fn_mtp_norms(emb, we, emb_n, hid, wh, hid_n, m)`: MTP's two pre-fc Gemma norms.
pub const SIG_FN_MTP_NORMS: &[SigArg] = &[
    p("emb", 2),
    p("we", 2),
    p("emb_n", 2),
    p("hid", 2),
    p("wh", 2),
    p("hid_n", 2),
    a("m", I32),
];

/// FN6 `fn_mtp_add(fe, fh, x, m)`: x[t, s] = bf16(fc_embedding[t] + fc_hidden[t, s]).
pub const SIG_FN_MTP_ADD: &[SigArg] = &[p("fe", 2), p("fh", 2), p("x", 2), a("m", I32)];

/// FN6 `fn_moe_silu_mul_bf16_t256(gu, act, inter, row_tok, meta)`: the MTP experts' SiluAndMul.
pub const SIG_FN_MOE_SILU_BF16: &[SigArg] = &[
    p("gu", 2),
    p("act", 2),
    a("inter", I32),
    p("row_tok", 4),
    p("meta", 4),
];

/// FN6 `fn_moe_bgemm_sk_m16_g{0,1}`: FN5's split-K bf16 band over FN4's tile tables (the MTP
/// experts, bf16 -- the default).
pub const SIG_FN_MOE_BGEMM: &[SigArg] = &[
    p("a", 2),
    p("xptr", 8),
    p("tile_expert", 4),
    p("tile_row0", 4),
    p("tile_rows", 4),
    p("row_tok", 4),
    p("meta", 4),
    p("part", 4),
    p("cnt", 4),
    p("c", 2),
    a("n", I32),
    a("k", I32),
    a("rows_cap", I32),
];

/// FN6 `fn_moe_a8w8_bs128_g{0,1}`: the drafter's W8A8 block-scale body over FN4's tile tables
/// (the MTP experts as Qwen's first-party fp8 -- a GATED option).
pub const SIG_FN_MOE_A8W8: &[SigArg] = &[
    p("a", 1),
    p("xptr", 8),
    p("tile_expert", 4),
    p("tile_row0", 4),
    p("tile_rows", 4),
    p("row_tok", 4),
    p("meta", 4),
    p("as", 4),
    p("c", 2),
    a("n", I32),
    a("k", I32),
];

/// `vis_layernorm_bf16(x, w, b, y, m, c, eps)`: nn.LayerNorm with weight and bias.
pub const SIG_VIS_LN: &[SigArg] = &[
    p("x", 2),
    p("w", 2),
    p("b", 2),
    p("y", 2),
    a("m", I32),
    a("c", I32),
    a("eps", F32),
];

/// `vis_pos_add(patch, table, idx, wts, out, l, c)`: bilinear pos-embed gather-add. `idx` is
/// i32 `[4, L]`, `wts` fp32 `[4, L]` (host metadata from the grid).
pub const SIG_VIS_POS_ADD: &[SigArg] = &[
    p("patch", 2),
    p("table", 2),
    p("idx", 4),
    p("wts", 4),
    p("out", 2),
    a("l", I32),
    a("c", I32),
];

/// `vis_rope_2d(qkv, cos, sin, l, heads, hd)`: in place on the q and k thirds of qkv.
pub const SIG_VIS_ROPE: &[SigArg] = &[
    p("qkv", 2),
    p("cos", 2),
    p("sin", 2),
    a("l", I32),
    a("heads", I32),
    a("hd", I32),
];

/// `vis_attn_bf16_hd72(qkv, out, cu, heads, scale)`: non-causal attention per image; `cu` is the
/// i32 cu_seqlens `[nseq + 1]`. D2's tiled kernels (`vis_attn_tiled_bf16_hd72`, the hi+lo P arm,
/// and `vis_attn_tiled_p16_bf16_hd72`, P rounded once) take the same five arguments.
pub const SIG_VIS_ATTN: &[SigArg] = &[
    p("qkv", 2),
    p("out", 2),
    p("cu", 4),
    a("heads", I32),
    a("scale", F32),
];

// ------------------------------------------------------------------- the head plugin (P3)

/// `saber_embed_bf16(table, ids, out, hidden, vocab_begin, vocab_end)`: the vocab-parallel
/// embedding lookup. `vocab_begin`/`vocab_end` are this rank's slice, and a token outside it
/// writes a zero row -- the contract the all-reduce after it depends on at TP > 1.
pub const SIG_EMBED: &[SigArg] = &[
    a("table", Buffer),
    a("ids", Buffer),
    a("out", Buffer),
    a("hidden", I32),
    a("vocab_begin", I32),
    a("vocab_end", I32),
];

/// `saber_row_gather_bf16(src, idx, out, cols, src_stride)`: the row selection before `lm_head`,
/// which is what keeps a 2048-token prefill's head one `[1, 5120]` GEMM instead of 2048.
pub const SIG_ROW_GATHER: &[SigArg] = &[
    a("src", Buffer),
    a("idx", Buffer),
    a("out", Buffer),
    a("cols", I32),
    a("src_stride", I64),
];

/// `saber_argmax_bf16(logits, out, cols, stride)`: greedy sampling, ties to the lowest index.
pub const SIG_ARGMAX: &[SigArg] = &[
    a("logits", Buffer),
    a("out", Buffer),
    a("cols", I32),
    a("stride", I64),
];

/// `saber_row_copy_bf16(src, out, cols, src_stride, dst_stride)`: the DFlash2 aux tap — one slab
/// of `rows x cols` bf16 copied into a COLUMN BLOCK of a wider destination.
///
/// The two strides are independent, which is the whole difference from [`SIG_ROW_GATHER`]: that
/// one packs SELECTED rows together (`out + r * cols`), this one writes row `r` at
/// `r * dst_stride` so five taps land side by side in one `rows x 5 x hidden` concatenation.
pub const SIG_ROW_COPY: &[SigArg] = &[
    p("src", 2),
    p("out", 2),
    a("cols", I32),
    a("src_stride", I64),
    a("dst_stride", I64),
];

/// `saber_add_rms_bf16(y, res, w, out, res_out, k, eps)`: the bf16 head's final norm (N2) — the
/// residual add and Gemma RMS norm of `saber_add_rms_quant`, with the bf16 row as the output.
pub const SIG_ADD_RMS_BF16: &[SigArg] = &[
    p("y", 2),
    p("res", 2),
    p("w", 2),
    p("out", 2),
    p("res_out", 2),
    a("k", I32),
    a("eps", F32),
];

/// `saber_widen_i32_i64(src, out, cols, src_stride, dst_stride)`: an i32 table sign-extended to
/// i64, rows `src_stride` / `dst_stride` apart.
///
/// The element widths ARE the point of this kernel (see [`SigArg::elem`]): `dflash_topk_bf16_k16`
/// writes `int` and both `dflash_selector_score_r256_k16` and `dflash_selector_walk_greedy` take
/// their candidate table as `const long *`, so a chain that bound one straight to the other would
/// read the right addresses and the wrong bytes — ws-f24's finding, from the other side.
pub const SIG_WIDEN_I64: &[SigArg] = &[
    p("src", 4),
    p("out", 8),
    a("cols", I32),
    a("src_stride", I64),
    a("dst_stride", I64),
];

/// `saber_device_row_bf16(logits, temps, pen_ids, top_values, top_ids, lse, pen_values, cols,
/// stride, pen_max)`: the sampler's seam — the K largest logits and their ids in the argmax
/// kernel's own total order, the row's log-sum-exp at T = 1 and at its own temperature, and the
/// raw logits at the ids the penalties can move.
///
/// The element widths are the ones the last two defects in this tree were about (`SigArg::elem`):
/// the logits are `__bf16` (2), every index list is `int` (4) and every value written back is
/// `float` (4). `temps` is the one per-ROW kernarg that varies per step, and it is a buffer for
/// exactly that reason — a captured graph's kernargs are fixed at capture and its CONTENTS are
/// what a step rewrites.
pub const SIG_DEVICE_ROW: &[SigArg] = &[
    p("logits", 2),
    p("temps", 4),
    p("pen_ids", 4),
    p("top_values", 4),
    p("top_ids", 4),
    p("lse", 4),
    p("pen_values", 4),
    a("cols", I32),
    a("stride", I64),
    a("pen_max", I32),
];

/// CMP27Y: `saber_device_row_split_g{2,4,8}_bf16` = [`SIG_DEVICE_ROW`] + `sync` (the row's u32 slab:
/// barrier counters, maxima, lse partials, candidates; zeroed once, never poisoned).
pub const SIG_DEVICE_ROW_SPLIT: &[SigArg] = &[
    p("logits", 2),
    p("temps", 4),
    p("pen_ids", 4),
    p("top_values", 4),
    p("top_ids", 4),
    p("lse", 4),
    p("pen_values", 4),
    a("cols", I32),
    a("stride", I64),
    a("pen_max", I32),
    p("sync", 4),
];

// ------------------------------------------------- the attention layer's glue (plugin 4)

/// `qk_norm_rope_*`. Takes the cos/sin CACHE and the mrope positions, not a materialised
/// `[3, T, rd/2]` pair: prod's `cos_sin_cache[positions]` gather is folded into this launch.
pub const SIG_QK_NORM_ROPE: &[SigArg] = &[
    a("qkv", Buffer),
    a("qw", Buffer),
    a("kw", Buffer),
    a("cos_sin", Buffer),
    a("pos", Buffer),
    a("q_out", Buffer),
    a("k_out", Buffer),
    a("m", I32),
    a("nq", I32),
    a("nkv", I32),
    a("qkv_n", I32),
    a("eps", F32),
    a("sec_t", I32),
    a("sec_h", I32),
    a("sec_w", I32),
];

/// `qk_rope_append_*` (ws-h38): `qk_norm_rope` and `kv_append` in ONE dispatch. The rope's own
/// arguments, then the append's minus `key` (the fused kernel reads back the `k_out` its own
/// workgroup wrote).
pub const SIG_QK_ROPE_APPEND: &[SigArg] = &[
    a("qkv", Buffer),
    a("qw", Buffer),
    a("kw", Buffer),
    a("cos_sin", Buffer),
    a("pos", Buffer),
    a("q_out", Buffer),
    a("k_out", Buffer),
    a("m", I32),
    a("nq", I32),
    a("nkv", I32),
    a("qkv_n", I32),
    a("eps", F32),
    a("sec_t", I32),
    a("sec_h", I32),
    a("sec_w", I32),
    a("value", Buffer),
    a("kv", Buffer),
    a("slot", Buffer),
    a("k_scale", Buffer),
    a("v_scale", Buffer),
    a("key_stride", I64),
    a("value_stride", I64),
    a("block_stride", I64),
    a("page_stride", I64),
    a("head_stride", I64),
];

/// `attn_gate_quant_*`. `gate` is the projection's `[q|gate]` BLOCK, read at `gs` row stride and
/// `2*hd` head stride with the kernel adding the `+hd` gate offset itself; prod materialises the
/// same values as a contiguous tensor of its own first, which this does not need.
pub const SIG_ATTN_GATE_QUANT: &[SigArg] = &[
    a("core", Buffer),
    a("gate", Buffer),
    a("q", Buffer),
    a("scale", Buffer),
    a("n", I32),
    a("hd", I32),
    a("gs", I32),
];

/// `kv_append_*`. One launch writes both K and V, as `triton_reshape_and_cache_flash` does; the
/// V base is the K base plus `head_size` elements of the same paged tensor.
pub const SIG_KV_APPEND: &[SigArg] = &[
    a("key", Buffer),
    a("value", Buffer),
    a("kv", Buffer),
    a("slot", Buffer),
    a("k_scale", Buffer),
    a("v_scale", Buffer),
    a("key_stride", I64),
    a("value_stride", I64),
    a("block_stride", I64),
    a("page_stride", I64),
    a("head_stride", I64),
];

/// `mxfp4_gemm_decode_fused_silu_*`: the decode GEMM with `silu_mul_quant` folded into its A
/// staging. No `a` and no `as` -- the A operand IS the bf16 gate_up output, and the per-token
/// scale is computed inside. No partitions either: the fused route is a single-partition MLP edge.
pub const SIG_GEMM_DECODE_SILU: &[SigArg] = &[
    a("gu", Buffer),
    a("w", Buffer),
    a("ws", Buffer),
    a("wref", Buffer),
    a("decode_partial", Buffer),
    a("decode_counter", Buffer),
    a("c", Buffer),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

// ------------------------------------------------- the DFlash2 drafter (plugin 5)

/// `dflash_gemm_a8w8_*`: the drafter's fp8 W8A8 blockscale GEMM. `w` is PRESHUFFLED to
/// `[N/16, K*16]`, which is the layout the reference's kernel un-shuffles; `as` is `[M, K/128]` and `ws`
/// is `[N/128, K/128]`, both fp32.
pub const SIG_DFLASH_GEMM: &[SigArg] = &[
    p("a", 1),
    p("w", 1),
    p("as", 4),
    p("ws", 4),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_draft_head_int2_*`: the reference's int2 draft head. `wq` is `[N, K/4]` quarter-split packed
/// int2, `s`/`zs` are `[N, K/G]` bf16 PREMULTIPLIED scale and zero (`4*sc` and `4*sc + zp*sc`), and
/// `xs` is `[M, K/G]` fp32 per-group sums of `x` — the zero point's other half. `bmax`/`bidx` carry
/// the top `KCAND` of each 64-column block, which is what the rerank draws its candidates from.
pub const SIG_DFLASH_HEAD: &[SigArg] = &[
    p("x", 2),
    p("xs", 4),
    p("wq", 1),
    p("s", 2),
    p("zs", 2),
    p("y", 2),
    p("bmax", 4),
    p("bidx", 4),
    a("m", I32),
    a("n", I32),
    a("k", I32),
    a("stride_wq", I32),
    a("stride_s", I32),
    a("stride_xs", I32),
    a("nblk", I32),
];

/// `dflash_rerank_exact_*`: the exact logit for `r` candidates per draft row, off the head's own
/// e4m3 weight (`w` as raw bytes, `ws` one fp32 scale per row — a compressed-tensors per-channel
/// FP8 `lm_head`). One program per (row, candidate).
/// `out` is `void *` in the C signature and its width is the ARM, not the kernel's opinion: the
/// dense entry point stores `((float *)out)[...]` and `_scatter` stores `((__bf16 *)out)[...]`
/// into the head's own `[M, N]` buffer. That is a per-SYMBOL element width over one kernarg
/// layout, so [`signature_for`] returns the `_scatter` pair below for those two names.
pub const SIG_DFLASH_RERANK: &[SigArg] = &[
    p("x", 2),
    p("w", 1),
    p("ws", 4),
    p("idx", 4),
    p("out", 4),
    a("k", I32),
    a("stride_w", I32),
    a("r", I32),
    // `n` is the vocabulary width, used only by the `_scatter` variant, which writes into the
    // head's own `[M, N]` output instead of a dense `[M, R]` one.
    a("n", I32),
];

/// [`SIG_DFLASH_RERANK`] with the scatter arm's bf16 store.
pub const SIG_DFLASH_RERANK_SCATTER: &[SigArg] = &[
    p("x", 2),
    p("w", 1),
    p("ws", 4),
    p("idx", 4),
    p("out", 2),
    a("k", I32),
    a("stride_w", I32),
    a("r", I32),
    a("n", I32),
];

/// `dflash_rerank_exact_bf16_*`: the same rerank against a **bf16** head — the reference's
/// `FP8=False` arm, which is the one the DFlash2 serve runs because `SABER_NVFP4_LMHEAD=bf16`
/// dequantises `lm_head` at load. There is no `ws`: `_rerank_exact` passes `x` for that argument
/// on this arm and never loads it, so binding a scale here would be a record the pack does not
/// need and a kernarg nothing reads.
pub const SIG_DFLASH_RERANK_BF16: &[SigArg] = &[
    p("x", 2),
    p("w", 2),
    p("idx", 4),
    p("out", 4),
    a("k", I32),
    a("stride_w", I32),
    a("r", I32),
    a("n", I32),
];

/// [`SIG_DFLASH_RERANK_BF16`] with the scatter arm's bf16 store.
pub const SIG_DFLASH_RERANK_BF16_SCATTER: &[SigArg] = &[
    p("x", 2),
    p("w", 2),
    p("idx", 4),
    p("out", 2),
    a("k", I32),
    a("stride_w", I32),
    a("r", I32),
    a("n", I32),
];

/// `dflash_rerank_exact_w8_*` (workstream N6): the bf16 arm's arithmetic over the checkpoint's
/// OWN e4m3 head and its per-row f32 scale — `w8` / `ws` bind `head.rerank.w8` / `head.rerank.ws`,
/// the same bytes as the target's `lm_head.w8` / `lm_head.ws` — expanded in register to
/// `bf16(f32(w8) * f32(scale))`, which is the bf16 record's element bit for bit.
pub const SIG_DFLASH_RERANK_W8: &[SigArg] = &[
    p("x", 2),
    p("w8", 1),
    p("ws", 4),
    p("idx", 4),
    p("out", 4),
    a("k", I32),
    a("stride_w", I32),
    a("r", I32),
    a("n", I32),
];

/// [`SIG_DFLASH_RERANK_W8`] with the scatter arm's bf16 store.
pub const SIG_DFLASH_RERANK_W8_SCATTER: &[SigArg] = &[
    p("x", 2),
    p("w8", 1),
    p("ws", 4),
    p("idx", 4),
    p("out", 2),
    a("k", I32),
    a("stride_w", I32),
    a("r", I32),
    a("n", I32),
];

/// `dflash_grouped_conv_*`: the drafter's 2-tap grouped convolution, bf16 in and out.
pub const SIG_DFLASH_CONV: &[SigArg] = &[
    p("x", 2),
    p("coeff", 2),
    p("base", 2),
    p("out", 2),
    a("t_rows", I32),
    a("h", I32),
    a("groups", I32),
    a("block_size", I32),
    a("side", I32),
];

/// `dflash_grouped_conv_quant_*`: the same convolution with the per-128-group e4m3 quant fused
/// into it, so it writes `q` and `scales` and never materialises the bf16 in between.
pub const SIG_DFLASH_CONV_QUANT: &[SigArg] = &[
    p("x", 2),
    p("coeff", 2),
    p("base", 2),
    p("q", 1),
    p("scales", 4),
    a("t_rows", I32),
    a("h", I32),
    a("groups", I32),
    a("block_size", I32),
    a("side", I32),
];

/// `dflash_bgemm_*`: the drafter's bf16 projections. `w` is `[N, K]` row-major, which is
/// `nn.Linear.weight` as it is stored, so nothing transposes it.
pub const SIG_DFLASH_BGEMM: &[SigArg] = &[
    p("a", 2),
    p("w", 2),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_bgemm_w8_*`: the target's fp8-WEIGHT `lm_head` (workstream N4). `w` is the checkpoint's
/// `[N, K]` e4m3 tensor as stored (one byte per element), `ws` its per-output-row scale as f32
/// (prod's parameter dtype), expanded in register to the bf16 head's bytes before the bf16 WMMA.
pub const SIG_DFLASH_BGEMM_W8: &[SigArg] = &[
    p("a", 2),
    p("w", 1),
    p("ws", 4),
    p("c", 2),
    a("m", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_w8_expand_gate(w, ws, out, n, k, mode)`: GATE-ONLY (N4), the fp8 loader's `expand` over
/// a whole matrix, so the gate can byte-compare it against the bf16 head's record. Not a route.
pub const SIG_DFLASH_W8_EXPAND_GATE: &[SigArg] = &[
    p("w", 1),
    p("ws", 4),
    p("out", 2),
    a("n", I32),
    a("k", I32),
    a("mode", I32),
];

/// `dflash_selector_score_*`: the drafter's edge scoring. The two codebooks are indexed by the
/// kernel itself, so no gathered tensor is ever materialised.
///
/// The WIDTHS are prod's and they are load-bearing: `cand` is `torch.topk`'s **i64**, `anchor` is
/// `input_ids`' i32, `unary` is the exact rerank's **f32**, and `out` is therefore **f32** --
/// `_score_edges` ends in `f32 + bf16`. Three of the four were wrong until 2026-09-21 and were the
/// whole of the proposal gate's selector row (LEDGER F round 24).
pub const SIG_DFLASH_SELECTOR: &[SigArg] = &[
    p("pred_cb", 2),
    p("succ_cb", 2),
    p("cand", 8),
    p("anchor", 4),
    p("unary", 4),
    p("hidden", 2),
    p("out", 4),
    a("bl", I32),
    a("l_len", I32),
    a("r", I32),
    a("k_top", I32),
];

/// `dflash_attn_*`: the drafter's attention. `kc`/`vc` are the reference server's FLASH layout --
/// `[blocks, block_size, kv_heads, head_size]`, SEPARATE caches -- which is a manifest fact, not a
/// convention: the reference's interleaved paged cache cannot be passed here.
pub const SIG_DFLASH_ATTN: &[SigArg] = &[
    p("q", 2),
    p("kc", 2),
    p("vc", 2),
    p("block_table", 4),
    p("seq_lens", 4),
    // SEPARATE tensors, as prod passes them: `[1, kv_heads]` each, of which the reference reads
    // element 0. Packing them as one buffer would take k's scale for v.
    p("k_scale", 4),
    p("v_scale", 4),
    p("out", 2),
    a("rows", I32),
    a("num_q_heads", I32),
    a("num_kv_heads", I32),
    a("block_size", I32),
    a("bt_stride", I32),
    a("window", I32),
    a("scale", F32),
];

/// The same slots over an **e4m3** cache: the two cache pointers are `unsigned char *` and the
/// kernel decodes each byte. Two signatures rather than one, because the kernarg LAYOUT is
/// identical and the element WIDTHS are not — which is precisely the distinction nothing in this
/// tree checked before [`SigArg::elem`].
pub const SIG_DFLASH_ATTN_FP8KV: &[SigArg] = &[
    p("q", 2),
    p("kc", 1),
    p("vc", 1),
    p("block_table", 4),
    p("seq_lens", 4),
    p("k_scale", 4),
    p("v_scale", 4),
    p("out", 2),
    a("rows", I32),
    a("num_q_heads", I32),
    a("num_kv_heads", I32),
    a("block_size", I32),
    a("bt_stride", I32),
    a("window", I32),
    a("scale", F32),
];

/// The BATCHED pair (`..._t32_ns`, ws-s11): the same slots and one more kernarg.
///
/// `q_per_seq` is the drafter's own block and `num_seqs` is `rows / q_per_seq`, which is why only
/// one of the two is a kernarg — the other is the grid's x extent (`rows / BLOCK_Q + num_seqs`),
/// and two numbers that must agree are one number. The element widths are the fp8 pair's, because
/// only the fp8 cache route is bound; the bf16 `_ns` symbol is compiled beside it so the two
/// instantiations cannot drift, and it takes the bf16 widths.
pub const SIG_DFLASH_ATTN_NS: &[SigArg] = &[
    p("q", 2),
    p("kc", 2),
    p("vc", 2),
    p("block_table", 4),
    p("seq_lens", 4),
    p("k_scale", 4),
    p("v_scale", 4),
    p("out", 2),
    a("rows", I32),
    a("num_q_heads", I32),
    a("num_kv_heads", I32),
    a("block_size", I32),
    a("bt_stride", I32),
    a("window", I32),
    a("scale", F32),
    a("q_per_seq", I32),
];

/// The batched pair over an **e4m3** cache. See [`SIG_DFLASH_ATTN_NS`].
pub const SIG_DFLASH_ATTN_FP8KV_NS: &[SigArg] = &[
    p("q", 2),
    p("kc", 1),
    p("vc", 1),
    p("block_table", 4),
    p("seq_lens", 4),
    p("k_scale", 4),
    p("v_scale", 4),
    p("out", 2),
    a("rows", I32),
    a("num_q_heads", I32),
    a("num_kv_heads", I32),
    a("block_size", I32),
    a("bt_stride", I32),
    a("window", I32),
    a("scale", F32),
    a("q_per_seq", I32),
];

/// `dflash_kv_append_flash_*`: the drafter's KV append. `kc`/`vc` are the reference server's FLASH layout --
/// separate `[blocks, block_size, kv_heads, head_size]` caches -- and `slots` is a block-table walk
/// in i64, not a token index.
///
/// `key_stride` and `value_stride` are ELEMENTS PER SOURCE ROW, and they are here because the reference server's
/// own `reshape_and_cache_flash` has them: the caller hands this kernel views. The drafter's k is
/// the rope's contiguous `[M, 8, 128]` output (stride 1024) and its v is a VIEW of the
/// `[q | k | v]` projection (stride 6144, which the capture's manifest records). Assuming
/// contiguity for both was worth every appended v byte (F29).
pub const SIG_DFLASH_KV: &[SigArg] = &[
    p("key", 2),
    p("value", 2),
    p("kc", 2),
    p("vc", 2),
    p("slots", 8),
    p("scales", 4),
    a("rows", I32),
    a("nkv", I32),
    a("head", I32),
    a("block_size", I32),
    a("key_stride", I32),
    a("value_stride", I32),
];

/// ... and into an **e4m3** cache, where the two cache pointers are one byte per element and the
/// two INPUTS are still bf16.
pub const SIG_DFLASH_KV_FP8: &[SigArg] = &[
    p("key", 2),
    p("value", 2),
    p("kc", 1),
    p("vc", 1),
    p("slots", 8),
    p("scales", 4),
    a("rows", I32),
    a("nkv", I32),
    a("head", I32),
    a("block_size", I32),
    a("key_stride", I32),
    a("value_stride", I32),
];

/// `dflash_topk_*`: the selector's candidate selection, one workgroup per row.
///
/// `idx` is **`int64`** and that is the CONSUMER's law: `dflash_selector_score_r256_k16` declares
/// `cand` as `const long *` because prod's candidate ids are `torch.topk`'s int64 (F22 and F24
/// were both a slot declared narrower than the table behind it). Through F27 this kernel wrote
/// i32 and the unseeded chain re-widened on the host between two launches that are meant to be
/// one edge; F28 made the selection's output width prod's.
pub const SIG_DFLASH_TOPK: &[SigArg] = &[
    p("x", 2),
    p("vals", 4),
    p("idx", 8),
    a("rows", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_silu_quant_*`: `silu(gate) * up` and the per-128-group e4m3 quant its consumer needs.
pub const SIG_DFLASH_SILU: &[SigArg] = &[
    p("gate_up", 2),
    p("q", 1),
    p("scales", 4),
    a("t_rows", I32),
    a("n", I32),
];

/// `dflash_act_quant_*`: the blockscale GEMM's OWN activation quant, bf16 -> e4m3 with
/// per-128-group f32 scales. Six launches a decode step at two shapes -- the aux projection's
/// `[ctx, 5 x hidden]` and each layer's `[m, q_heads x head_dim]` attention output -- and the same
/// three slots `silu_quant` has, minus the gate/up split its input carries.
///
/// The widths are the reason this is its own signature rather than `SIG_DFLASH_SILU` renamed: `x`
/// is the producer's MATERIALISED bf16 tensor (2), `q` is e4m3 (1) and `scales` is f32 (4), and
/// `tests/capture_widths.rs` holds each against what the capture binds.
pub const SIG_DFLASH_ACT_QUANT: &[SigArg] = &[
    p("x", 2),
    p("q", 1),
    p("scales", 4),
    a("t_rows", I32),
    a("n", I32),
];

/// `dflash_ctx_kv_layout*`: the context store's `permute(2, 1, 0, 3, 4).contiguous()`, as the
/// permutation it is. `in` is `[num_ctx, L, 2, nkv, hd]` and `out` is `[2, L, num_ctx, nkv, hd]`,
/// both bf16 and both contiguous.
pub const SIG_DFLASH_CTX_KV_LAYOUT: &[SigArg] = &[
    p("in", 2),
    p("out", 2),
    a("num_ctx", I32),
    a("layers", I32),
    a("nkv", I32),
    a("hd", I32),
];

/// `dflash_head_group_sum_g128`: the int2 head's `xs`, `x.reshape(M, ng, G).float().sum(-1)`.
///
/// `x` is the PADDED bf16 head input (2) and `xs` is fp32 (4) -- the widths are the point, because
/// the head kernel below reads `xs` as `const float *` and prod materialises it from a bf16 `x`
/// through an fp32 cast it never stores anywhere a census could see.
pub const SIG_DFLASH_HEAD_GROUP_SUM: &[SigArg] = &[
    p("x", 2),
    p("xs", 4),
    a("rows", I32),
    a("k", I32),
    a("stride_xs", I32),
];

/// `dflash_head_row_pad_*`: `_pow2_at_least`'s row pad, `torch.cat([x, x.new_zeros(M - m, k)])`.
/// Both slots are bf16 (2); `m` is the LIVE rows and `rows` the padded height.
pub const SIG_DFLASH_HEAD_ROW_PAD: &[SigArg] = &[
    p("x", 2),
    p("out", 2),
    a("m", I32),
    a("rows", I32),
    a("k", I32),
];

/// `dflash_row_gather*`: `x[idx]` over whole ROWS, at a width in BYTES rather than in elements.
///
/// `in` and `out` are byte slots (1) precisely because the two sites it serves are different
/// dtypes -- bf16 hidden rows and i32 token ids -- and a gather has no arithmetic to make the
/// dtype matter. `idx` is ATen's `int64` (8), unconverted, because these index tensors are prod's
/// own and narrowing them here would be the F22 defect in reverse.
pub const SIG_DFLASH_ROW_GATHER: &[SigArg] = &[
    p("in", 1),
    p("idx", 8),
    p("out", 1),
    a("rows", I32),
    a("row_bytes", I32),
    a("src_rows", I32),
];

/// `dflash_head_cand_gather*`: `bi.gather(1, bm.topk(RERANK).indices)`, the map from block-maxima
/// POSITIONS to vocabulary COLUMNS. All three slots are i32 (4): `bidx` is what the int2 head
/// wrote, `pos` is what the top-80's gather pass returned, and `out` is what the exact rerank
/// scores. Prod's own tensors are i64 at the boundary and the caller narrows, exactly as it does
/// for the walk's candidates.
pub const SIG_DFLASH_HEAD_CAND_GATHER: &[SigArg] = &[
    p("bidx", 4),
    p("pos", 4),
    p("out", 4),
    a("rows", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_topk_f32_thresh_*`: pass one of the int2 head's top-80, the radix select. `thresh` is
/// ONE `u64` per row -- the k-th largest `(value, ~index)` composite -- and it is eight bytes
/// because the ordering it carries is the pair and not the value.
pub const SIG_DFLASH_TOPK_THRESH: &[SigArg] = &[
    p("x", 4),
    p("thresh", 8),
    a("rows", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_topk_f32_gather_*`: pass two, the gather and the rank. `vals` is fp32 (ATen returns the
/// input dtype and this row is fp32 already) and `idx` is i32, which the caller widens: prod's
/// `bm.topk(...).indices` is i64.
pub const SIG_DFLASH_TOPK_GATHER: &[SigArg] = &[
    p("x", 4),
    p("thresh", 8),
    p("vals", 4),
    p("idx", 4),
    a("rows", I32),
    a("n", I32),
    a("k", I32),
];

/// `dflash_silu_quant_*_amax`: the same kernel with the pre-scale maximum stored as well. A
/// DIAGNOSTIC, never a route -- it exists so "which amax did each side compute?" can be asked
/// directly instead of inferred backwards through `a * (1/448)`, which is lossy.
pub const SIG_DFLASH_SILU_AMAX: &[SigArg] = &[
    p("gate_up", 2),
    p("q", 1),
    p("scales", 4),
    p("amax_out", 4),
    a("t_rows", I32),
    a("n", I32),
];

/// `dflash_conv_add_rms_*_open`: the convolution's `finish` half, the residual add and the RMS
/// norm -- one kernel, as inductor already fuses them -- at the ATTENTION half, which is where
/// prod's compiled region OPENS its fp32 residual (`buf6 = empty_strided_cuda(..., float32)`).
/// The residual comes in as bf16 (the previous layer's, through the eager attention seam) and
/// goes out as FP32.
pub const SIG_DFLASH_CONV_RMS_OPEN: &[SigArg] = &[
    p("x", 2),
    p("coeff", 2),
    p("base", 2),
    p("resid", 2),
    p("weight", 2),
    p("out", 2),
    p("res_out", 4),
    a("t_rows", I32),
    a("h", I32),
    a("groups", I32),
    a("block_size", I32),
    a("side", I32),
    a("eps", F32),
];

/// `dflash_conv_add_rms_*_close`: the same kernel at the MLP half, which CLOSES that residual --
/// prod's `buf24 = buf6` is read fp32 and the region's bf16 output `buf25` is what the next
/// layer's attention sees. Four bytes in, two out; every other slot is the `_open` row's.
///
/// The WIDTHS are the whole of the difference, and they are two facts about prod's graph rather
/// than a choice: a residual rounded to bf16 at both halves is rounded once a layer more than
/// prod rounds it (F29; F28 named it off `drafter_graph.py` and could not gate it).
pub const SIG_DFLASH_CONV_RMS_CLOSE: &[SigArg] = &[
    p("x", 2),
    p("coeff", 2),
    p("base", 2),
    p("resid", 4),
    p("weight", 2),
    p("out", 2),
    p("res_out", 2),
    a("t_rows", I32),
    a("h", I32),
    a("groups", I32),
    a("block_size", I32),
    a("side", I32),
    a("eps", F32),
];

/// `dflash_rms_norm`: the one norm in the drafter with no residual to add and no convolution in
/// front of it to fuse into. `qwen3_dflash2.py:164` takes the `residual is None` branch at layer 0
/// and nowhere else, so this signature is used exactly once a step.
pub const SIG_DFLASH_RMS: &[SigArg] = &[
    p("x", 2),
    p("weight", 2),
    p("out", 2),
    a("t_rows", I32),
    a("h", I32),
    a("eps", F32),
];

/// `dflash_rms_norm_grouped_*`: the RMS norm the drafter's CONTEXT-KV store calls, which is the
/// C++ `upstream::rms_norm_kernel` and not `ir.ops.rms_norm`. `weight` is `[groups, h]` and a row
/// takes group `row / rows_per_group` -- the drafter stacks its five layers' `k_norm.weight` into
/// one buffer so the per-layer K norm is a single grouped launch, and the hidden norm is the same
/// kernel with one group.
pub const SIG_DFLASH_RMS_GROUPED: &[SigArg] = &[
    p("x", 2),
    p("weight", 2),
    p("out", 2),
    a("rows", I32),
    a("h", I32),
    a("rows_per_group", I32),
    a("eps", F32),
];

/// `dflash_rope_neox_k_inplace`: the ONE batched rope of the context store -- K only, in place,
/// over `L * num_ctx` rows with the positions repeated per layer. A second entry point rather than
/// an `attn_glue` band row, because `qk_norm_rope` norms first and writes q and k separately.
pub const SIG_DFLASH_ROPE_CTX: &[SigArg] = &[
    p("positions", 8),
    p("k", 2),
    p("cos_sin", 2),
    a("rows", I32),
    a("heads", I32),
    a("head_size", I32),
    a("rot_dim", I32),
    a("stride", I32),
];

/// `dflash_qk_norm_rope_h128*`: the drafter's QUERY q/k norm + rope, its own kernel because the
/// TARGET's `attn_glue::qk_norm_rope` is a GemmaRMSNorm (`1 + w`) over an ATen reduction with a
/// non-contracting rope, and the drafter's is a plain `RMSNorm` over a Triton reduction with a
/// contracting one. Seven pointers: the `[q | k | v]` projection, the two per-head norm weights
/// (q then k, the order the plan pushes the records in), the shared `[max_position, head_dim]`
/// cos/sin cache, the i64 positions, and q and k to separate outputs.
///
/// `pos` is **`int64`**, as every position table the reference server hands a kernel is; the cache is bf16,
/// because the rotary module holds it at the model dtype and this call -- unlike the context
/// store's -- does not cast it.
pub const SIG_DFLASH_QK_ROPE: &[SigArg] = &[
    p("qkv", 2),
    p("qw", 2),
    p("kw", 2),
    p("cos_sin", 2),
    p("pos", 8),
    p("q_out", 2),
    p("k_out", 2),
    a("m", I32),
    a("nq", I32),
    a("nkv", I32),
    a("qkv_n", I32),
    a("eps", F32),
];

/// `dflash_selector_walk_*`: the seven-step walk from the selector's edge scores to the draft
/// block. One program per request, one wave, as the reference launches it.
///
/// `candidates` and `tokens` are **`int64`**: `compute_candidates` returns `torch.topk`'s indices
/// and ATen types those `int64`, so the Triton reference loads and stores eight bytes per element.
/// `req_state` is `int32` and `realized` is `f32`. The widths are a manifest fact of the tensors
/// prod hands this kernel, and reading `candidates` at four bytes is bit-exact on the scores and
/// wrong on every token (LEDGER, workstream F round 22).
pub const SIG_DFLASH_WALK: &[SigArg] = &[
    p("scores", 4),
    p("candidates", 8),
    p("req_state", 4),
    p("tokens", 8),
    p("realized", 4),
    a("reqs", I32),
    a("num_steps", I32),
    a("top_k", I32),
];

/// FN23: `fn23_offsets`, the card-side pre-flight of the argument offsets the walk unpacks.
pub const SIG_FN23_OFFSETS: &[SigArg] = &[p("out", 4)];

/// The signature a kernel symbol must have, by name. The `.hsaco` metadata carries offsets but
/// not meaning, so this is where an argument gets a NAME: `saber-kbuild` labels the manifest from
/// here and refuses a kernel whose code-object layout does not pack the same way.
pub fn signature_for(kernel: &str) -> Option<&'static [SigArg]> {
    let sig = if kernel.starts_with("dflash_gemm_a8w8") {
        SIG_DFLASH_GEMM
    } else if kernel.starts_with("dflash_selector_walk") {
        SIG_DFLASH_WALK
    } else if kernel.starts_with("dflash_act_quant") {
        SIG_DFLASH_ACT_QUANT
    } else if kernel.starts_with("dflash_ctx_kv_layout") {
        SIG_DFLASH_CTX_KV_LAYOUT
    } else if kernel.starts_with("dflash_row_gather") {
        SIG_DFLASH_ROW_GATHER
    } else if kernel.starts_with("dflash_head_cand_gather") {
        SIG_DFLASH_HEAD_CAND_GATHER
    } else if kernel.starts_with("dflash_head_group_sum") {
        SIG_DFLASH_HEAD_GROUP_SUM
    } else if kernel.starts_with("dflash_head_row_pad") {
        SIG_DFLASH_HEAD_ROW_PAD
    } else if kernel.starts_with("dflash_silu_quant") && kernel.ends_with("_amax") {
        SIG_DFLASH_SILU_AMAX
    } else if kernel.starts_with("dflash_silu_quant") {
        SIG_DFLASH_SILU
    } else if kernel.starts_with("dflash_conv_add_rms_") && kernel.ends_with("_close") {
        SIG_DFLASH_CONV_RMS_CLOSE
    } else if kernel.starts_with("dflash_conv_add_rms_") && kernel.ends_with("_open") {
        // Explicitly, not as a fallback: the two halves differ ONLY in two element widths, so a
        // symbol that names neither must reach no signature at all rather than the nearer one.
        SIG_DFLASH_CONV_RMS_OPEN
    } else if kernel.starts_with("dflash_rms_norm_grouped") {
        SIG_DFLASH_RMS_GROUPED
    } else if kernel.starts_with("dflash_rope_neox") {
        SIG_DFLASH_ROPE_CTX
    } else if kernel.starts_with("dflash_qk_norm_rope") {
        SIG_DFLASH_QK_ROPE
    } else if kernel.starts_with("dflash_rms_norm") {
        SIG_DFLASH_RMS
    } else if kernel.starts_with("dflash_topk_f32_thresh") {
        SIG_DFLASH_TOPK_THRESH
    } else if kernel.starts_with("dflash_topk_f32_gather") {
        SIG_DFLASH_TOPK_GATHER
    } else if kernel.starts_with("dflash_topk") {
        SIG_DFLASH_TOPK
    } else if kernel.starts_with("dflash_kv_append_flash") {
        // The CACHE's element width is the instantiation's, and the kernarg layout is the same
        // either way -- see `SigArg::elem`.
        if kernel.contains("fp8") {
            SIG_DFLASH_KV_FP8
        } else {
            SIG_DFLASH_KV
        }
    } else if kernel.starts_with("dflash_attn") {
        // `_ns` is the batched walk and carries one more kernarg (`q_per_seq`). The suffix is the
        // whole of the difference, and it is read here rather than anywhere else.
        match (kernel.contains("fp8kv"), kernel.ends_with("_ns")) {
            (true, false) => SIG_DFLASH_ATTN_FP8KV,
            (true, true) => SIG_DFLASH_ATTN_FP8KV_NS,
            (false, false) => SIG_DFLASH_ATTN,
            (false, true) => SIG_DFLASH_ATTN_NS,
        }
    } else if kernel.starts_with("dflash_selector_score") {
        SIG_DFLASH_SELECTOR
    } else if kernel.starts_with("dflash_bgemm_w8") || kernel.starts_with("saber_head_w8_m64") {
        SIG_DFLASH_BGEMM_W8
    } else if kernel == "dflash_w8_expand_gate" {
        SIG_DFLASH_W8_EXPAND_GATE
    } else if kernel.starts_with("dflash_bgemm") {
        SIG_DFLASH_BGEMM
    } else if kernel.starts_with("dflash_grouped_conv_quant") {
        SIG_DFLASH_CONV_QUANT
    } else if kernel.starts_with("dflash_grouped_conv") {
        SIG_DFLASH_CONV
    } else if kernel.starts_with("dflash_draft_head_int2") {
        SIG_DFLASH_HEAD
    } else if kernel.starts_with("dflash_rerank_exact_w8") {
        if kernel.ends_with("_scatter") {
            SIG_DFLASH_RERANK_W8_SCATTER
        } else {
            SIG_DFLASH_RERANK_W8
        }
    } else if kernel.starts_with("dflash_rerank_exact_bf16") {
        if kernel.ends_with("_scatter") {
            SIG_DFLASH_RERANK_BF16_SCATTER
        } else {
            SIG_DFLASH_RERANK_BF16
        }
    } else if kernel.starts_with("dflash_rerank_exact") {
        if kernel.ends_with("_scatter") {
            SIG_DFLASH_RERANK_SCATTER
        } else {
            SIG_DFLASH_RERANK
        }
    } else if kernel == "fn23_offsets" {
        SIG_FN23_OFFSETS
    } else if (kernel.starts_with("fn_hc_down_") || kernel.starts_with("fn131_down_"))
        && kernel.ends_with("_h8")
    {
        // FN131: the phase-split decode downs take their parent's arguments
        SIG_FN_HC8_DOWN
    } else if (kernel.starts_with("fn_hc_up_") || kernel.starts_with("fn141_up_"))
        && kernel.ends_with("_h8")
    {
        // FN141: the entry-load decode ups take their parent's arguments
        SIG_FN_HC8_UP
    } else if kernel.starts_with("fn_hc_down_")
        && (kernel.ends_with("_rp") || kernel.ends_with("_rps") || kernel.ends_with("_rpo"))
    {
        SIG_FN_HC_DOWN_RP
    } else if kernel.starts_with("fn_hc_down_") {
        SIG_FN_HC_DOWN
    } else if kernel.starts_with("fn_hc_up_") {
        SIG_FN_HC_UP
    } else if kernel == "fn_hc_combine" {
        SIG_FN_HC_COMBINE
    } else if kernel.starts_with("fn_ar_exact_hcrr_") {
        SIG_FN_AR_HCRR
    } else if kernel == "fn_wmma_probe" {
        SIG_FN_WMMA_PROBE
    } else if kernel == "fn_hc_sig_probe" {
        SIG_FN_HC_SIG_PROBE
    } else if kernel == "fn_t8_a8w8_bs128_ldc"
        || kernel == "fn111_t8_a8w8_n64_ldc"
        || kernel == "fn141_t8_a8w8_n64_pf4_ldc"
        || (kernel.starts_with("fn_t8p_a8w8_") && kernel.ends_with("_ldc"))
    {
        // FN61e: the multi-row W8A8 tiles take FN61c's ldc signature
        SIG_FN_T8_A8W8_LDC
    } else if kernel == "fn_t8r_a8w8_gather_m16n64" {
        SIG_FN_T8R_GATHER
    } else if kernel == "fn_t8_bgemm_m16_ldc" || kernel == "fn_t8p_bgemm_m64_ldc" {
        SIG_FN_T8_BGEMM_LDC
    } else if kernel.starts_with("fn_bgemm_sk_") || kernel.starts_with("fn141_bgemm_sk_") {
        SIG_FN_BGEMM_SK
    } else if kernel.starts_with("fn_bgemm_lds_") {
        SIG_FN_BGEMM
    } else if kernel == "fn_gdn_norm_bf16_t256" || kernel == "fn_gdn_norm_bf16_sig_t256" {
        // FN9: the sigmoid-gated twin (Flash-Next's gated norm) takes the same kernargs.
        SIG_FN_GDN_NORM_BF16
    } else if kernel == "fn_gdn_norm_q8_sig_t256" || kernel == "fn_gdn_norm_q8_sig_t256_rc" {
        SIG_FN_GDN_NORM_Q8
    } else if kernel == "fn_attn_gate_q8_t256" || kernel == "fn_attn_gate_q8_t256_rc" {
        SIG_FN_ATTN_GATE_Q8
    } else if kernel == "fn86_q8_bgemm_m16_ldc" || kernel == "fn86_q8_bgemm_m16_ldc_rc" {
        SIG_FN86_Q8_BGEMM_LDC
    } else if kernel == "fn86_t8_q8w8_sfx_ldc"
        || kernel == "fn86_t8_q8w8_sfx_ldc_rc"
        || kernel == "fn111_t8_q8w8_sfx_n64_ldc"
    {
        // FN111: the 64-column re-tile takes the same kernargs
        SIG_FN86_Q8W8_SFX
    } else if kernel.starts_with("fn86_gdn_fused_update_nq8_fc_") {
        SIG_FN86_GDN_FUSED_NQ8
    } else if kernel.starts_with("fn86_gdn_recurrent_nq8_fc_") {
        SIG_FN86_GDN_REC_NQ8
    } else if kernel == "fn_attn_gate_bf16_t256" || kernel == "fn_attn_gate_bf16_v2_t256" {
        // FN21: the v2 gate (loads first) takes the same kernargs.
        SIG_FN_ATTN_GATE_BF16
    } else if kernel.starts_with("fn_gdn_chunk_scan_xg_")
        || kernel.starts_with("fn_gdn_chunk_scan_xg0_")
        || kernel.starts_with("fn_gdn_chunk_scan_xg2_")
    {
        // FN9b: the exact-gate scan (flashnext_gdn), the certified scan's kernargs unchanged; FN20's
        // xg2 (V' weighted once, byte-identical) likewise. FN20c: `xg` routes xg2's body, `xg0`
        // keeps FN9b's as the control.
        SIG_SABER_GDN_CHUNK_SCAN
    } else if kernel == "fn_draft_argmax_part" {
        SIG_FN_DRAFT_ARGMAX
    } else if kernel == "fn_draft_pack" {
        SIG_FN_DRAFT_PACK
    } else if kernel == "fn_draft_select" {
        SIG_FN_DRAFT_SELECT
    } else if kernel == "fn_pick_top64_bf16" {
        SIG_FN_PICK_TOP64
    } else if kernel == "fn_xmiss_e256" {
        SIG_FN_XMISS
    } else if kernel == "fn_ar_host_exact_bf16" {
        SIG_SABER_AR_EXACT
    } else if kernel.starts_with("fn_ar_ll_exact_bf16") {
        SIG_FN_AR_LL
    } else if kernel == "fn_gate98" {
        SIG_FN_GATE98
    } else if kernel == "fn119_draw_walk" {
        SIG_FN119_DRAW_WALK
    } else if kernel == "fn119_expf_probe" {
        SIG_FN119_EXPF_PROBE
    } else if kernel.starts_with("fn120_res_v") {
        SIG_FN120_RES
    } else if kernel == "fn120_df_run" {
        SIG_FN120_DF_RUN
    } else if kernel == "fn120_stamp" {
        SIG_FN120_STAMP
    } else if kernel == "fn120_df_pair" {
        SIG_FN120_DF_PAIR
    } else if kernel == "fn120_df_gdn_b1"
        || kernel == "fn120_df_gemm_probe"
        || kernel == "fn120_df_nobody_probe"
    {
        SIG_FN120_DF_GDN
    } else if kernel == "fn122_hc8_dequant" {
        SIG_FN122_HC8_DEQUANT
    } else if kernel == "p27d_cand_seam" {
        SIG_P27D_CAND_SEAM
    } else if kernel == "fn119a_push" {
        SIG_FN119A_PUSH
    } else if kernel == "fn119a_walk" {
        SIG_FN119A_WALK
    } else if kernel == "fn119a_emb" {
        SIG_FN119A_EMB
    } else if kernel == "fn119a_gather" {
        SIG_FN119A_GATHER
    } else if kernel == "fn119a_sel" {
        SIG_FN119A_SEL
    } else if kernel == "fn119a_backup" {
        SIG_FN119A_BACKUP
    } else if kernel == "fn119b_copy" {
        SIG_FN119B_COPY
    } else if kernel == "fn119b_compose" {
        SIG_FN119B_COMPOSE
    } else if kernel == "fn119d_vemb" {
        SIG_FN119D_VEMB
    } else if kernel == "fn_couple_acp" {
        SIG_FN_COUPLE_ACP
    } else if kernel == "fn_couple_select" {
        SIG_FN_COUPLE_SELECT
    } else if kernel == "fn_draft_x_gather" {
        SIG_FN_DRAFT_X_GATHER
    } else if kernel == "fn_draft_head_int2_quant" {
        SIG_FN_DRAFT_HEAD_INT2_QUANT
    } else if kernel == "fn_draft_wgather" {
        SIG_FN_DRAFT_WGATHER
    } else if kernel == "fn_draft_argmax_cand" {
        SIG_FN_DRAFT_ARGMAX_CAND
    } else if kernel == "fn86_draft_xg_sum_q8" || kernel == "fn86_draft_xg_sum_q8_rc" {
        SIG_FN86_DRAFT_XG_SUM_Q8
    } else if kernel == "fn86_topk_gather_cand" || kernel == "fn86_topk_gather_cand_rc" {
        SIG_FN86_TOPK_K80_CAND
    } else if kernel == "fn86_draft_argmax_cand_pack" || kernel == "fn86_draft_argmax_cand_pack_rc"
    {
        SIG_FN86_DRAFT_ARGMAX_CAND_PACK
    } else if kernel == "fn_qsa_score_r32" || kernel.starts_with("fn128_qsa_score_s") {
        SIG_FN_QSA_SCORE
    } else if kernel == "fn_qsa_topk512" || kernel.starts_with("fn128_qsa_topk512_k") {
        SIG_FN_QSA_TOPK
    } else if kernel == "fn_qsa_union" || kernel.starts_with("fn128_qsa_union_s") {
        SIG_FN_QSA_UNION
    } else if kernel.starts_with("fn_qsa_attn_prefill_")
        || kernel.starts_with("fn124_qsa_attn_prefill_")
    {
        SIG_FN_QSA_ATTN_PREFILL
    } else if kernel == "fn124_cmp_u32" {
        SIG_FN124_CMP
    } else if kernel.starts_with("fn_qsa_attn_decode_") {
        SIG_FN_QSA_ATTN_DECODE
    } else if kernel.starts_with("fn_attn_prefill_") {
        // FN6: the certified saber attention bodies at GQA 12 over bf16 pages, same kernargs.
        SIG_SABER_ATTN_PREFILL
    } else if kernel.starts_with("fn_attn_declc_") {
        SIG_SABER_ATTN_DECLC
    } else if kernel.starts_with("fn_attn_decode_") {
        SIG_SABER_ATTN_DECODE
    } else if kernel.starts_with("fn_attn_splitkv_combine_") {
        SIG_SABER_ATTN_COMBINE
    } else if kernel == "fn_kv_append_ring_h256_bf16" {
        SIG_FN_KV_APPEND_RING
    } else if kernel == "fn_attnfuse_front_h256_bf16" {
        SIG_FN_ATTNFUSE_FRONT_BF16
    } else if kernel == "fn_attnfuse_front_h256_fp8" {
        SIG_FN_ATTNFUSE_FRONT_FP8
    } else if kernel == "fn_attnfuse_combine_gate_q8_h256" {
        SIG_FN_ATTNFUSE_COMBINE_GATE_Q8
    } else if kernel == "fn_kv_append_ring_h256_fp8" {
        SIG_FN_KV_APPEND_RING_FP8
    } else if kernel == "fn_idx_q_compress_h128" || kernel == "fn_idx_q_compress_h128_v2" {
        // FN21: the v2 indexer (four rows a workgroup) takes the same kernargs.
        SIG_FN_IDX_Q_COMPRESS
    } else if kernel == "fn_qk_norm_rope_h256_rot64_mi1_v2" {
        // FN21: Flash-Next's own qk_norm_rope instantiation, attn_glue's kernargs.
        SIG_QK_NORM_ROPE
    } else if kernel == "fn_ple_gate_x" {
        SIG_FN_PLE_GATE_X
    } else if kernel == "fn_ple_gate_combine" {
        SIG_FN_PLE_GATE_COMBINE
    } else if kernel == "fn_ple_conv" {
        SIG_FN_PLE_CONV
    } else if kernel == "fn_ple_rows_fp8" {
        SIG_FN_PLE_ROWS_FP8
    } else if kernel == "fn_mtp_norms" {
        SIG_FN_MTP_NORMS
    } else if kernel == "fn_mtp_add" {
        SIG_FN_MTP_ADD
    } else if kernel == "fn_moe_silu_mul_bf16_t256" {
        SIG_FN_MOE_SILU_BF16
    } else if kernel.starts_with("fn_moe_bgemm_sk_") {
        SIG_FN_MOE_BGEMM
    } else if kernel.starts_with("fn_moe_a8w8_") {
        SIG_FN_MOE_A8W8
    } else if kernel == "vis_cast_f32_bf16" {
        SIG_VIS_CAST
    } else if kernel.starts_with("vis_gemm_bf16_bias") {
        SIG_VIS_GEMM
    } else if kernel == "vis_layernorm_bf16" {
        SIG_VIS_LN
    } else if kernel == "vis_pos_add" {
        SIG_VIS_POS_ADD
    } else if kernel == "vis_rope_2d" {
        SIG_VIS_ROPE
    } else if kernel == "vis_attn_bf16_hd72"
        || kernel == "vis_attn_tiled_bf16_hd72"
        || kernel == "vis_attn_tiled_p16_bf16_hd72"
    {
        SIG_VIS_ATTN
    } else if kernel == "saber_noop" {
        SIG_NOOP
    } else if kernel == "saber_noop_touch" {
        SIG_NOOP_TOUCH
    } else if kernel == "saber_race_digest_regions" {
        SIG_RACE_DIGEST_REGIONS
    } else if kernel == "saber_race_digest_kv" {
        SIG_RACE_DIGEST_KV
    } else if kernel == "saber_embed_bf16" {
        SIG_EMBED
    } else if kernel == "saber_row_gather_bf16" {
        SIG_ROW_GATHER
    } else if kernel == "saber_argmax_bf16" || kernel == "saber_argmax_pv_bf16" {
        // CMP27T: the pv twin's signature is the argmax's (its `out` is 2 x rows ints)
        SIG_ARGMAX
    } else if kernel == "saber_device_row_bf16"
        || kernel == "saber_device_row_ilp_bf16"
        || kernel.starts_with("saber_device_row_stage_t")
    {
        // CMP27X: the load-ahead twin's signature is the seam row's; CMP27AJ: so is the
        // staged-term form's
        SIG_DEVICE_ROW
    } else if kernel.starts_with("saber_device_row_split_g") {
        // CMP27Y: the split row = the seam row's + its per-row sync slab
        SIG_DEVICE_ROW_SPLIT
    } else if kernel == "saber_row_copy_bf16" {
        SIG_ROW_COPY
    } else if kernel == "saber_widen_i32_i64" {
        SIG_WIDEN_I64
    } else if kernel == "saber_add_rms_bf16" {
        SIG_ADD_RMS_BF16
    } else if kernel.starts_with("mxfp4_gemm_decode_fused_silu") {
        SIG_GEMM_DECODE_SILU
    } else if kernel.starts_with("mxfp4_gemm_decode_pair") {
        SIG_GEMM_DECODE_PAIR
    } else if kernel.starts_with("mxfp4_gemm_decode_arnq") {
        SIG_GEMM_DECODE_AR_NQ
    } else if kernel.starts_with("mxfp4_gemm_decode_arw") {
        SIG_GEMM_DECODE_ARW
    } else if kernel.starts_with("mxfp4_gemm_decode_smq") {
        SIG_GEMM_DECODE_SMQ
    } else if kernel.starts_with("mxfp4_gemm_decode") {
        SIG_GEMM_DECODE
    } else if kernel.starts_with("mxfp4_gemm_folded")
        || kernel.starts_with("mxfp4_gemm_atiled")
        || kernel.starts_with("mxfp4_gemm_skinny")
    {
        SIG_GEMM_FOLDED
    } else if kernel == "mxfp4_gemm_base" {
        SIG_GEMM_BASE
    } else if kernel.starts_with("moe_grouped_w4a8_") {
        SIG_MOE_GROUPED
    } else if kernel.starts_with("moe_back_mo_") || kernel.starts_with("moe_back_mop_") {
        SIG_MOE_BACK_MO
    } else if kernel.starts_with("fn137_moe_back_h2_far_") {
        SIG_MOE_BACK_FAR
    } else if kernel.starts_with("fn152_moe_back_h2_far_") {
        // FN152: FN151's NVFP4 `_p1` body in FN137's far back half (the same launch ABI)
        SIG_MOE_BACK_FAR
    } else if kernel.starts_with("fn164_far_") {
        SIG_FN164_T8_FAR
    } else if kernel == "fn137_delay" || kernel == "fn152_delay" || kernel == "fn164_delay" {
        SIG_FN137_DELAY
    } else if kernel.starts_with("moe_back_ss_") {
        SIG_MOE_BACK_SS
    } else if kernel.starts_with("moe_back_") || kernel.starts_with("fn152_moe_back_") {
        SIG_MOE_BACK
    } else if kernel.starts_with("moe_front_") || kernel.starts_with("fn163_front_") {
        SIG_MOE_FRONT
    } else if kernel.starts_with("moe_grouped_a16_ref_") {
        SIG_MOE_A16
    } else if kernel.starts_with("fn148_a16_nvfp4_ref_") {
        SIG_FN148_A16
    } else if kernel.starts_with("fn148_w4a8b_")
        || kernel.starts_with("fn148_w4a8c_")
        || kernel.starts_with("fn151_w4a8b_")
        || kernel.starts_with("fn153_w4a8b_")
    {
        SIG_FN148_GROUPED
    } else if kernel.starts_with("moe_silu_mul_quant_") {
        SIG_MOE_SILU
    } else if kernel.starts_with("moe_route_topk10_sort_")
        || kernel == "fn111_moe_route_lds_dpp_nt512"
        || kernel.starts_with("fn163_route_")
        || kernel.starts_with("fn163_attr_route_")
    {
        SIG_MOE_ROUTE
    } else if kernel == "moe_combine_bf16" {
        SIG_MOE_COMBINE
    } else if kernel.starts_with("add_rms_quant") {
        SIG_ADD_RMS_QUANT
    } else if kernel.starts_with("silu_mul_quant") {
        SIG_SILU_MUL_QUANT
    } else if kernel.starts_with("gdn_norm_quant") {
        SIG_GDN_NORM_QUANT
    } else if kernel.starts_with("saber_attn_prefill_") {
        SIG_SABER_ATTN_PREFILL
    } else if kernel.starts_with("saber_attn_declc_") {
        SIG_SABER_ATTN_DECLC
    } else if kernel.starts_with("p27_attn_decode_anc_")
        || kernel.starts_with("p27_attn_decode_anc16_")
    {
        SIG_P27_ATTN_DECODE_ANC
    } else if kernel == "p27_tree_kv_compact" {
        SIG_P27_TREE_KV_COMPACT
    } else if kernel == "p27_tree_conv_commit" {
        SIG_P27_TREE_CONV_COMMIT
    } else if kernel.starts_with("p27_gdn_conv_tree_") {
        SIG_P27_GDN_CONV_TREE
    } else if kernel.starts_with("p27_gdn_tree_advance_layers_") {
        SIG_P27_GDN_TREE_ADVANCE_LAYERS
    } else if kernel.starts_with("p27_gdn_tree_advance_")
        || kernel.starts_with("p27_gdn_tree16_advance_")
    {
        SIG_P27_GDN_TREE_ADVANCE
    } else if kernel.starts_with("p27_gdn_tree_") || kernel.starts_with("p27_gdn_tree16_") {
        SIG_P27_GDN_TREE
    } else if kernel.starts_with("p27w2_gdn_replay_") {
        SIG_P27W2_GDN_REPLAY
    } else if kernel.starts_with("p27w2_gdn_cuf_") {
        SIG_P27W2_GDN_CUF
    } else if kernel.starts_with("saber_attn_decode_ragged_") {
        // Before the uniform prefix, which it shares.
        SIG_SABER_ATTN_DECODE_RAGGED
    } else if kernel.starts_with("saber_attn_splitkv_combine_ragged_") {
        SIG_SABER_ATTN_COMBINE_RAGGED
    } else if kernel.starts_with("saber_attn_decode_") {
        SIG_SABER_ATTN_DECODE
    } else if kernel.starts_with("saber_attn_splitkv_combine_") {
        SIG_SABER_ATTN_COMBINE
    } else if kernel.starts_with("saber_gdn_fnq_") {
        SIG_SABER_GDN_FNQ
    } else if kernel.starts_with("saber_gdn_fused_update_")
        || kernel.starts_with("fn_gdn_fused_update_fc_")
    {
        // FN24: the f16-carry body takes the shipped fused update's kernargs unchanged
        SIG_SABER_GDN_FUSED_UPDATE
    } else if kernel.starts_with("saber_gdn_gated_rmsnorm_") {
        SIG_SABER_GDN_NORM
    } else if kernel.starts_with("saber_ar_oneshot_2rank_wht")
        || kernel.starts_with("saber_ar_wpar_2rank_wht")
    {
        // CMP27S: the wave-parallel body takes the served one-shot's kernargs unchanged
        SIG_SABER_AR_WHT
    } else if kernel.starts_with("saber_ar_split_2rank_wht") && kernel.ends_with("_push_nb") {
        // P27-1d: the split all-reduce (P27-1b's halves + P27-1c's fewer-blocks push). The
        // reduce half takes the one-shot's arguments unchanged.
        SIG_SABER_AR_SPLIT_PUSH_NB
    } else if kernel.starts_with("saber_ar_split_2rank_wht") && kernel.ends_with("_push") {
        SIG_SABER_AR_SPLIT_PUSH
    } else if kernel.starts_with("saber_ar_split_2rank_wht") && kernel.ends_with("_reduce") {
        SIG_SABER_AR_WHT
    } else if kernel.starts_with("saber_ar_oneshot_2rank_exact_nq") {
        SIG_SABER_AR_EXACT_NQ
    } else if kernel.starts_with("saber_ar_oneshot_2rank_exact") {
        SIG_SABER_AR_EXACT
    } else if kernel.starts_with("saber_gdn_conv_prep_") {
        SIG_SABER_GDN_CONV_PREP
    } else if kernel.starts_with("saber_gdn_conv_update_") {
        SIG_SABER_GDN_CONV_UPDATE
    } else if kernel.starts_with("saber_gdn_recurrent_update_")
        || kernel.starts_with("fn_gdn_recurrent_update_fc_")
    {
        SIG_SABER_GDN_RECURRENT
    } else if kernel.starts_with("saber_gdn_kkt_solve_") {
        SIG_SABER_GDN_KKT
    } else if kernel.starts_with("saber_gdn_chunk_scan_") {
        SIG_SABER_GDN_CHUNK_SCAN
    } else if kernel.starts_with("paro_add_rms_rot_tok_") {
        SIG_PARO_ADD_RMS_ROT_TOK
    } else if kernel.starts_with("paro_add_rms_rot_") {
        SIG_PARO_ADD_RMS_ROT
    } else if kernel.starts_with("paro_ew_rot_tok_") {
        SIG_PARO_EW_ROT_TOK
    } else if kernel.starts_with("paro_ew_rot_") {
        SIG_PARO_EW_ROT
    } else if kernel.starts_with("qk_norm_rope_") {
        SIG_QK_NORM_ROPE
    } else if kernel.starts_with("qk_rope_append_") {
        SIG_QK_ROPE_APPEND
    } else if kernel.starts_with("attn_gate_quant_") {
        SIG_ATTN_GATE_QUANT
    } else if kernel.starts_with("kv_append_") {
        SIG_KV_APPEND
    } else {
        return None;
    };
    Some(sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_signature_packs_the_way_c_does() {
        let (slots, size) = pack(SIG_GEMM_DECODE);
        // eight pointers, five ints, then a long that must land 8-aligned (84 -> 88).
        assert_eq!(slots[0], (0, 8));
        assert_eq!(slots[7], (56, 8));
        assert_eq!(slots[8], (64, 4)); // M
        assert_eq!(slots[12], (80, 4)); // pb2
        assert_eq!(slots[13], (88, 8)); // astride, padded
        assert_eq!(size, 96);
    }

    #[test]
    fn the_row3_signature_is_the_decode_gemm_then_the_nq_epilogue() {
        let (slots, size) = pack(SIG_GEMM_DECODE_AR_NQ);
        assert_eq!(slots.len(), 31);
        // The decode GEMM's image, unchanged: `astride` still at 88.
        let (gemm, _) = pack(SIG_GEMM_DECODE);
        assert_eq!(&slots[..14], &gemm[..]);
        assert_eq!(slots[14], (96, 8)); // peer_scratch
        assert_eq!(slots[24], (176, 8)); // tiles
        assert_eq!(slots[25], (184, 4)); // k16
        assert_eq!(slots[27], (192, 4)); // eps
        assert_eq!(slots[30], (204, 4)); // ep_rows
        assert_eq!(size, 208);
        assert_eq!(
            signature_for("mxfp4_gemm_decode_arnq_bk64_ks4_tm1_wperm1_nt1"),
            Some(SIG_GEMM_DECODE_AR_NQ)
        );
        assert_eq!(
            signature_for("mxfp4_gemm_decode_bk64_ks4_tm1_wperm1_nt1"),
            Some(SIG_GEMM_DECODE)
        );
    }

    #[test]
    fn saber_args_is_one_by_value_blob_whose_fields_the_host_still_writes() {
        // The code object sees ONE argument; the host writes 21 fields into it.
        let (slots, size) = pack(SIG_SABER_ATTN_DECODE);
        assert_eq!(slots[0], (0, SABER_ARGS_BYTES));
        assert_eq!(slots[1], (SABER_ARGS_BYTES, 4)); // splits, after the struct
        assert_eq!(size, 140);
        let (f, end) = pack_fields_at(SABER_ARGS_FIELDS, 0);
        assert_eq!(f[0], (0, 8)); // q
        assert_eq!(f[8], (64, 8)); // scratch, the ninth pointer
        assert_eq!(f[9], (72, 4)); // num_seqs
        assert_eq!(f[15], (96, 4)); // max_blocks
        assert_eq!(f[16], (104, 8)); // kv_block_stride, 8-aligned past the 4 bytes of padding
        assert_eq!(f[18], (120, 4)); // scale
        assert_eq!(f[20], (128, 4)); // max_ctx
                                     // The fields stop at 132; the struct is 136 because of its own trailing padding, and the
                                     // gap is real memory the host must leave alone.
        assert_eq!(end, 132);
        assert!(end <= SABER_ARGS_BYTES);
        // The combine kernel adds `tile` after `splits`.
        let (slots, size) = pack(SIG_SABER_ATTN_COMBINE);
        assert_eq!(slots[2], (140, 4));
        assert_eq!(size, 144);
        // ws-h38: the last-arriver decode adds its counter after `splits`, realigned to 8.
        let (slots, size) = pack(SIG_SABER_ATTN_DECLC);
        assert_eq!(slots[2], (144, 8));
        assert_eq!(size, 152);
        // ... and the rope+append: the rope's 88 bytes, then five pointers and five longs.
        let (slots, size) = pack(SIG_QK_ROPE_APPEND);
        assert_eq!(slots[15], (88, 8)); // value, 8-aligned past sec_w at 84
        assert_eq!(slots[20], (128, 8)); // key_stride
        assert_eq!(size, 168);
        // The ragged pair (C2): `cu_q` after the scalars, 8-aligned.
        let (slots, size) = pack(SIG_SABER_ATTN_DECODE_RAGGED);
        assert_eq!(slots[1], (136, 4));
        assert_eq!(slots[2], (144, 8));
        assert_eq!(size, 152);
        let (slots, size) = pack(SIG_SABER_ATTN_COMBINE_RAGGED);
        assert_eq!(slots[2], (140, 4));
        assert_eq!(slots[3], (144, 8));
        assert_eq!(size, 152);
    }

    #[test]
    fn saber_fused_update_packs_thirty_six_arguments() {
        let (slots, size) = pack(SIG_SABER_GDN_FUSED_UPDATE);
        assert_eq!(slots.len(), 36);
        assert_eq!(slots[8], (64, 4)); // state_len_max
        assert_eq!(slots[9], (72, 8)); // cache_idx, realigned
        assert_eq!(slots[31], (232, 4)); // scale
        assert_eq!(slots[33], (240, 8)); // bar
        assert_eq!(size, 256);
        let (_, size) = pack(SIG_SABER_AR_WHT);
        assert_eq!(size, 96);
        // The exact reduce: seven pointers then four ints.
        let (slots, size) = pack(SIG_SABER_AR_EXACT);
        assert_eq!(slots[7], (56, 4)); // n16
        assert_eq!(size, 72);
        // ... and the `_nq` epilogue: eleven pointers, two ints, a float, two ints.
        let (slots, size) = pack(SIG_SABER_AR_EXACT_NQ);
        assert_eq!(slots[11], (88, 4)); // k16
        assert_eq!(slots[13], (96, 4)); // eps
        assert_eq!(size, 108);
        let (_, size) = pack(SIG_SABER_GDN_NORM);
        assert_eq!(size, 68);
    }

    #[test]
    fn saber_gdn_prefill_and_decode_pair_signatures_pack_the_way_c_does() {
        let (slots, size) = pack(SIG_SABER_GDN_CONV_PREP);
        assert_eq!(slots[8], (64, 8)); // cache_idx
        assert_eq!(slots[22], (176, 4)); // N
        assert_eq!(slots[25], (188, 4)); // sp_thr
        assert_eq!(size, 192);
        let (slots, size) = pack(SIG_SABER_GDN_CONV_UPDATE);
        assert_eq!(slots[8], (64, 4)); // state_len_max
        assert_eq!(slots[9], (72, 8)); // cache_idx, realigned past the int
        assert_eq!(size, 140);
        let (slots, size) = pack(SIG_SABER_GDN_RECURRENT);
        assert_eq!(slots[5], (40, 8)); // ab_stride
        assert_eq!(slots[18], (144, 4)); // neps
        assert_eq!(size, 168);
        let (_, size) = pack(SIG_SABER_GDN_KKT);
        assert_eq!(size, 52);
        let (slots, size) = pack(SIG_SABER_GDN_CHUNK_SCAN);
        assert_eq!(slots[10], (80, 4)); // H
        assert_eq!(size, 92);
    }

    #[test]
    fn paro_producer_signatures_pad_before_the_rotation_table() {
        // eps is a float in the middle of the pointer block, so the next pointer realigns.
        let (slots, size) = pack(SIG_PARO_ADD_RMS_ROT);
        assert_eq!(slots[3], (24, 4)); // eps
        assert_eq!(slots[4], (32, 8)); // t, 8-aligned past 4 bytes of padding
        assert_eq!(slots[11], (88, 4)); // M
        assert_eq!(slots[14], (100, 4)); // gpw
        assert_eq!(size, 104);
        let (slots, size) = pack(SIG_PARO_EW_ROT);
        assert_eq!(slots[2], (16, 8)); // ys
        assert_eq!(slots[4], (32, 4)); // eps
        assert_eq!(slots[5], (40, 8)); // t
        assert_eq!(size, 104);
        let (slots, size) = pack(SIG_PARO_ADD_RMS_ROT_TOK);
        assert_eq!(slots[9], (72, 8)); // as
        assert_eq!(size, 92);
        let (_, size) = pack(SIG_PARO_EW_ROT_TOK);
        assert_eq!(size, 92);
    }

    #[test]
    fn gdn_signature_pads_before_the_long() {
        let (slots, size) = pack(SIG_GDN_NORM_QUANT);
        assert_eq!(slots[2], (16, 8)); // zs, already 8-aligned behind two pointers
        assert_eq!(slots[6], (48, 4)); // N
        assert_eq!(slots[7], (52, 4)); // eps
        assert_eq!(size, 56);
    }
}
