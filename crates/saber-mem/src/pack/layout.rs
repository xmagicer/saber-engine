//! Layout ids: what a record's bytes actually are.
//!
//! `docs/abi/PACK.md` § Layout ids is the source of truth; this module is its executable form. A
//! layout id names **the transform that produced the bytes together with the kernel variant that
//! consumes them** — not just a dtype, because two records of the same dtype and shape can be laid
//! out differently (an MXFP4 weight in checkpoint order and the same weight in WMMA fragment order
//! are both `[N, K/2]` bytes and only one of them computes the right answer).
//!
//! The reason this is an enum and not a `u32` in a doc comment: a pack built by one version and
//! read by another must disagree *loudly*. [`LayoutId::from_id`] refuses an id this build does not
//! know, and [`check`] refuses a known id in the wrong place — which is the negative control the
//! P2 gate runs (`a record with the wrong layout id for its role is refused at bind`).
//!
//! **Allocating a new id is an ABI change.** It goes in `PACK.md`'s table, in this enum and in the
//! builder that emits it, in one change; the three are diffed against each other by
//! `pack_md_and_this_module_agree` below.

use std::fmt;

use super::PackError;

/// A layout this build knows how to read.
///
/// The discriminants are the wire values and must never be renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum LayoutId {
    /// Raw checkpoint bytes, checkpoint dtype, rank-local shape. No transform but the TP shard.
    Raw = 0,
    /// MXFP4 W4A8 weight in WMMA fragment order: `ref_nvfp4.quant_mxfp4` (the `mse` block
    /// exponent) over the dequantized source, then `ref_mxfp4.permute_w`. Dtype
    /// `FP4E2M1X2`, logical shape `[N, K]`, `N * K / 2` bytes. Read by the `w` argument of every
    /// `mxfp4_gemm_{decode,folded,atiled}_*` variant at `wperm1`.
    Mxfp4Wperm = 1,
    /// The companion block scales of [`LayoutId::Mxfp4Wperm`]: `e8m0`, **transposed** to
    /// `[K/32, N]`, which is the layout the kernel indexes (`ref_mxfp4`'s
    /// `process_weights_after_loading` does the same transpose at load). Read as `ws`.
    Mxfp4BlockScale = 2,
    /// The per-output-row reference exponent, `ref_mxfp4.make_row_ref` = `ws.max(dim 0)`:
    /// `e8m0`, shape `[N]`. Read as `wref`; it is what lets the kernel fold the block exponent.
    Mxfp4RowRef = 3,
    /// [`LayoutId::Mxfp4Wperm`] whose dim 0 is a **concatenation of several checkpoint tensors**
    /// (the merged GDN `in_proj_qkvz` and `in_proj_ba`, the merged attention `qkv`, the merged
    /// MLP `gate_up`). The bytes are the same kind; the id differs because the segment boundaries
    /// are load-bearing for the consumer (`gdn_norm_quant`'s `zs`, the merged-linear partition
    /// select, the q/k/v split), so a merged record handed to a caller expecting a single
    /// projection is a bug this id makes visible.
    Mxfp4WpermMerged = 4,
    /// An `f32` copy of a `bf16` checkpoint tensor. The GDN kernels take `const float*` for
    /// `A_log` and `dt_bias` while the checkpoint stores them `bf16`, so the cast is a load-time
    /// transform like any other and belongs in the pack.
    F32FromBf16 = 5,
    /// A `bf16` matrix whose dim 0 is a concatenation of several checkpoint tensors, with no
    /// requantization: the drafter head's merged `qkv` and `gate_up`. The MTP head stays `bf16` in
    /// this tier (`quantization_config.ignore` ends with `re:^mtp.*`) and an MXFP4 drafter is a
    /// **measured** negative result — it loses acceptance (`docs/LEDGER.md`, "MXFP4 MTP
    /// quantization rejected") — so merging is the only transform it gets.
    Bf16Merged = 6,
    /// The DFlash2 drafter's fp8 W8A8 B operand in the reference's preshuffled order:
    /// `w.view(N/16, 16, K/32, 2, 16).permute(0, 2, 3, 1, 4)`, `e4m3`, shape `[N/16, K*16]`. Read
    /// as the `w` argument of `dflash_gemm_a8w8_bs128_m16n128k128`. The id is distinct from
    /// [`LayoutId::Raw`] because the same bytes in checkpoint order compute a wrong answer and
    /// nothing about the dtype or the length says which order they are in.
    Fp8Preshuffle = 7,
    /// The companion `[N/128, K/128]` block scales of [`LayoutId::Fp8Preshuffle`], cast from the
    /// checkpoint's `bf16` `weight_scale_inv` to the `f32` the kernel reads.
    Fp8BlockScale = 8,
    /// the reference's int2 draft head: asymmetric 2-bit min/max over 128-wide groups, quarter-split
    /// packed to `[N, K/4]` `u8`. Read as `wq`.
    Int2Head = 9,
    /// Its premultiplied group scale, `bf16(4 * sc)`, `[N, K/128]`. Read as `s`.
    Int2HeadScale = 10,
    /// Its premultiplied group zero, `bf16(4 * sc + zp * sc)`, `[N, K/128]`. Read as `zs`.
    Int2HeadZero = 11,
    /// A `bf16` dim-0 concatenation of block-fp8 tensors, dequantized: the drafter's fused
    /// context-KV projection. **Numerically equivalent to prod and not byte-identical to it** —
    /// prod builds the same matrix by pushing the identity through its own quantized layer.
    Bf16FromFp8Block = 12,
    /// A rope `cos|sin` table computed from the checkpoint's rope parameters, `bf16`
    /// `[max_pos, rotary_dim]`. Computed rather than copied, so it is tier 2 by construction.
    RopeCosSin = 13,
    /// A `bf16` matrix dequantized from a **per-channel** fp8 source: `bf16(f32(w8) * f32(scale))`
    /// row by row, which is exactly what `ref_nvfp4.RefFp8ChannelToBf16` leaves bound
    /// under `SABER_NVFP4_LMHEAD=bf16`. The target's `lm_head` is the only one, and it is what
    /// the DFlash2 drafter's exact rerank scores against AND what its int2 head is quantised
    /// from — so the rounding is one step earlier than the checkpoint's own bytes and the id says
    /// so. Distinct from [`LayoutId::Bf16FromFp8Block`], whose source is a `[N/128, K/128]` block
    /// scale and whose consumer is a different kernel.
    Bf16FromFp8Channel = 14,
    /// A **per-channel** e4m3 matrix carried RAW, `[N, K]` in the checkpoint's row order and dim-0
    /// sharded: the fp8-weight head's `lm_head.w8` (workstream N4). Not [`LayoutId::Raw`], because
    /// its consumer expands it in register with the companion [`LayoutId::F32ChannelScale`] to
    /// exactly [`LayoutId::Bf16FromFp8Channel`]'s bytes, and a bf16 reader handed these would
    /// compute garbage from bytes of half the right length.
    Fp8Channel = 15,
    /// The per-output-row scale of [`LayoutId::Fp8Channel`] as `f32` `[N]`: an exact widening of
    /// the checkpoint's `bf16` `weight_scale`, the dtype prod's `ChannelQuantScaleParameter` holds.
    F32ChannelScale = 16,
}

/// The first id no released builder may write. Ids at or above this are for experiments that must
/// not be loadable by a released engine (`PACK.md` § Layout ids).
pub const PRIVATE_BASE: u32 = 0x1000_0000;

impl LayoutId {
    /// Every id this build knows, in wire order.
    pub const ALL: [LayoutId; 17] = [
        LayoutId::Raw,
        LayoutId::Mxfp4Wperm,
        LayoutId::Mxfp4BlockScale,
        LayoutId::Mxfp4RowRef,
        LayoutId::Mxfp4WpermMerged,
        LayoutId::F32FromBf16,
        LayoutId::Bf16Merged,
        LayoutId::Fp8Preshuffle,
        LayoutId::Fp8BlockScale,
        LayoutId::Int2Head,
        LayoutId::Int2HeadScale,
        LayoutId::Int2HeadZero,
        LayoutId::Bf16FromFp8Block,
        LayoutId::RopeCosSin,
        LayoutId::Bf16FromFp8Channel,
        LayoutId::Fp8Channel,
        LayoutId::F32ChannelScale,
    ];

    /// Wire value.
    #[must_use]
    pub fn id(self) -> u32 {
        self as u32
    }

    /// The short name `PACK.md`, `manifest.json` and `explain` all use.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            LayoutId::Raw => "raw",
            LayoutId::Mxfp4Wperm => "mxfp4_wperm",
            LayoutId::Mxfp4BlockScale => "mxfp4_block_scale",
            LayoutId::Mxfp4RowRef => "mxfp4_row_ref",
            LayoutId::Mxfp4WpermMerged => "mxfp4_wperm_merged",
            LayoutId::F32FromBf16 => "f32_from_bf16",
            LayoutId::Bf16Merged => "bf16_merged",
            LayoutId::Fp8Preshuffle => "fp8_preshuffle",
            LayoutId::Fp8BlockScale => "fp8_block_scale",
            LayoutId::Int2Head => "int2_head",
            LayoutId::Int2HeadScale => "int2_head_scale",
            LayoutId::Int2HeadZero => "int2_head_zero",
            LayoutId::Bf16FromFp8Block => "bf16_from_fp8_block",
            LayoutId::RopeCosSin => "rope_cos_sin",
            LayoutId::Bf16FromFp8Channel => "bf16_from_fp8_channel",
            LayoutId::Fp8Channel => "fp8_channel",
            LayoutId::F32ChannelScale => "f32_channel_scale",
        }
    }

    /// Whether the bytes went through the NVFP4/fp8 → MXFP4 requant.
    #[must_use]
    pub fn is_mxfp4_weight(self) -> bool {
        matches!(self, LayoutId::Mxfp4Wperm | LayoutId::Mxfp4WpermMerged)
    }

    /// Decode a wire id.
    ///
    /// # Errors
    /// An id this build does not know — including a private one. A loader that met a record it
    /// cannot interpret must say so; guessing is how "weights read as garbage" happens.
    pub fn from_id(id: u32) -> Result<Self, PackError> {
        Self::ALL
            .into_iter()
            .find(|l| l.id() == id)
            .ok_or(PackError::UnknownLayout(id))
    }

    /// The name of an id that may not be known, for a message.
    #[must_use]
    pub fn describe(id: u32) -> String {
        match Self::from_id(id) {
            Ok(l) => l.name().to_owned(),
            Err(_) if id >= PRIVATE_BASE => format!("private layout {id}"),
            Err(_) => format!("unknown layout {id}"),
        }
    }
}

impl fmt::Display for LayoutId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Refuse a record whose layout is not the one the caller is about to read it as.
///
/// `what` names the record, so the message says which one and both ids. This is the whole check —
/// it is deliberately not a `bool`: a caller that ignores a `bool` compiles, and a caller that
/// ignores a `Result` does not.
///
/// # Errors
/// [`PackError::LayoutMismatch`] when `found` is not `expected`.
pub fn check(what: &str, found: u32, expected: LayoutId) -> Result<(), PackError> {
    if found == expected.id() {
        return Ok(());
    }
    Err(PackError::LayoutMismatch {
        record: what.to_owned(),
        found: LayoutId::describe(found),
        expected: expected.name(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_an_unknown_one_is_refused() {
        for l in LayoutId::ALL {
            assert_eq!(LayoutId::from_id(l.id()).unwrap(), l);
        }
        assert!(LayoutId::from_id(17).is_err(), "17 is not allocated yet");
        assert!(LayoutId::from_id(PRIVATE_BASE).is_err());
        assert!(LayoutId::describe(PRIVATE_BASE).contains("private"));
        assert!(LayoutId::describe(17).contains("unknown"));
    }

    #[test]
    fn check_names_the_record_and_both_layouts() {
        let e = check("layer.0.mlp.gate_up.w", 0, LayoutId::Mxfp4WpermMerged).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("layer.0.mlp.gate_up.w"), "{msg}");
        assert!(msg.contains("raw"), "{msg}");
        assert!(msg.contains("mxfp4_wperm_merged"), "{msg}");
        check("x", LayoutId::Raw.id(), LayoutId::Raw).expect("the matching case passes");
    }
}
