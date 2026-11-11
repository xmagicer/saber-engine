//! The Flash-Next executor's launch records (FN12): kernel declarations from the committed
//! manifests, kernarg images packed by the manifest's own offsets, and the per-rank arena the ops
//! draw their buffers from.
//!
//! A layer is EMITTED first -- a list of [`Rec`]s with concrete arena addresses, host only -- and
//! then executed. Emission needs no card: the host tests build a whole chunk's list against the
//! manifests (launch counts per layer vs `layer_graph`, the kernarg sizes, the seed law) and the
//! card runs the same list.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, ensure, Context};
use saber_kernels::manifest::{KernelDecl, Manifest};
use saber_model::qwen4_exp::layer_graph::Group;

/// The code objects the Flash-Next prefill forward launches (FN4-FN6, FN9b, FN11, the 27B's AR).
pub const PLUGINS: [&str; 57] = [
    "flashnext_hc",
    "flashnext_glue",
    "flashnext_attn",
    "attn_glue",
    "dflash_bgemm",
    "moe_route",
    "moe_grouped",
    // FN80b-1: the TP=2 expert half's DBK-64 down cells + the half-width back half
    "moe_grouped_tp2",
    // FN114c: the fused back half with the shared expert split across the EP ranks
    "moe_grouped_ss",
    // FN137: the half-width back half with the MoE-out all-reduce fused (`--fn-moe-ar-push on`)
    "moe_grouped_far",
    "saber_gdn_conv",
    "saber_gdn_fused_update_f16s",
    "saber_gdn_recurrent_update_f16s",
    "saber_gdn_kkt_solve",
    "saber_gdn_chunk_scan_f16s",
    "flashnext_gdn",
    "flashnext_gdn_fc",
    "flashnext_moe",
    "flashnext_qsa",
    "flashnext_qsa_attn",
    "flashnext_draft",
    // FN33: the drafter's int2 head (the 27B's scorer) and its exact top-k
    "dflash_head",
    "dflash_topk",
    // FN50: the gated fp8 MTP experts -- the per-128 activation quant and the mixed-table GEMMs
    "dflash_mlp",
    "flashnext_mtp8",
    // FN57: the step's seam pick on the card (`--fn-pick device`)
    "flashnext_pick",
    // FN119: the device sampler + acceptance twin (`--fn-device-step check`)
    "flashnext_dstep",
    // FN119 round 2: the issue-ahead's device half (`--fn-device-step ahead`)
    "flashnext_ahead",
    // FN119c: the b > 1 issue-ahead's chain inputs composed per sequence
    "flashnext_ahead2",
    // FN119d: the next verify's embedding rows on the card (`--fn-device-step next`)
    "flashnext_ahead3",
    // LR-N: coupled drafting (`--spec-couple on`): the chain tail drafts with the target's noise
    "flashnext_couple",
    // FN61c: the W8 trunk (`--fn-trunk w8`): the drafter's W8A8 GEMM + the output-stride cells
    "dflash_gemm",
    "flashnext_trunk8",
    // FN61e: the W8 trunk's multi-row tiles (verify 17..64 rows, prefill, the bf16 suffix)
    "flashnext_trunk8p",
    // FN68: the fp8 KV (`--fn-kv-dtype fp8`): the e4m3 append, the dense and gather attention
    "flashnext_glue_kv8",
    "flashnext_attn_kv8",
    "flashnext_qsa_attn_kv8",
    // FN124: the fp8-KV QSA gather prefill with the masked sub-tile skip (`--fn-prefill-tiles qsa`)
    "flashnext_qsa124",
    // FN128: the QSA selection at the decode / verify forms (`--fn-sel128`)
    "flashnext_qsa128",
    // ATTN-OPT-3: the attention-neighbour launch fusions (`--fn-fuse f6` / `f7`)
    "flashnext_attnfuse",
    // FN61g: the drafter's exact rerank on the W8 head (`--fn-trunk w8-head`)
    "flashnext_trunk8r",
    // FN108: the HC mix on fp8 mixing matrices (`--fn-trunk w8-hc8` / `w8-hc8-head`)
    "flashnext_hc8",
    // FN122b: the fp8-mixer HC down's PREFILL tiles (`--fn-prefill-tiles hc`)
    "flashnext_hc8p",
    // FN131: the fp8-mixer HC down at the decode forms re-tiled by phase (`--fn-hc-down split`)
    "flashnext_hc131",
    // FN141: the drained-ring twins (`--fn-k141`)
    "flashnext_hc141",
    "flashnext_t8141",
    // FN164: the out / o W8 GEMM with the mixer-out all-reduce fused (`--fn-mixer-ar-push`)
    "flashnext_t8far",
    // FN163: the parallel route / re-tiled front twins (`--fn-route163`)
    "moe_route163",
    // FN53c: the valve's device-side miss flag (`--fn-mem-valve on`, `--fn-valve-graphs on`)
    "flashnext_xmiss",
    // FN114b: the flag-in-data exact reduce (`--fn-ar ll`)
    "flashnext_ar",
    // FN114d: the global-parity one-shot reduces (`--fn-ar gp`)
    "flashnext_ar_gp",
    // FN120c: the GDN mixer tail as one persistent dataflow launch (`--fn-dataflow gdn`)
    "flashnext_df120b",
    // FN150: the NVFP4 expert cells (FN148 form (b), `--fn-experts-fmt nvfp4`)
    "moe_fn148",
    // FN152: FN151's `_p1` twins (the default NVFP4 route) + the NVFP4 fused back halves
    "moe_fn151",
    "moe_fn152",
    "saber_ar_exact",
    "saber_ar_wht",
];

/// One explicit kernel argument.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// A device address (8 bytes; 0 = null).
    P(u64),
    /// `int`.
    I(i32),
    /// `long` / `size_t` (8 bytes).
    L(i64),
    /// `float`.
    F(f32),
    /// A by-value struct (the attention's `SABERArgs`, 136 bytes).
    Blob(Vec<u8>),
}

/// Every kernel declaration of [`PLUGINS`], host-side (no card).
#[derive(Debug, Clone)]
pub struct Decls {
    by_sym: BTreeMap<String, (&'static str, KernelDecl)>,
}

impl Decls {
    /// Parse the manifests under `prebuilt`.
    pub fn load(prebuilt: &Path) -> anyhow::Result<Decls> {
        Self::load_for_ar(prebuilt, false)
    }

    /// The no-P2P object's declaration is needed only by the registered-host profile.
    pub fn load_for_ar(prebuilt: &Path, host: bool) -> anyhow::Result<Decls> {
        Self::load_plugins(
            prebuilt,
            PLUGINS.into_iter().chain(host.then_some("flashnext_ar_host")),
        )
    }

    pub(super) fn load_plugins(
        prebuilt: &Path,
        plugins: impl IntoIterator<Item = &'static str>,
    ) -> anyhow::Result<Decls> {
        let mut by_sym = BTreeMap::new();
        for p in plugins {
            let m = Manifest::load(prebuilt.join(format!("{p}.manifest.toml")))
                .with_context(|| format!("{p}.manifest.toml"))?;
            for k in m.kernel {
                by_sym.insert(k.name.clone(), (p, k));
            }
        }
        Ok(Decls { by_sym })
    }

    /// The plugin and declaration of `sym`.
    pub fn get(&self, sym: &str) -> anyhow::Result<(&'static str, &KernelDecl)> {
        self.by_sym
            .get(sym)
            .map(|(p, d)| (*p, d))
            .with_context(|| format!("{sym}: in none of the Flash-Next code objects"))
    }

    /// The kernarg image of `sym` for `args`, laid out at the manifest's offsets; every argument's
    /// size must equal the declared one (a pointer where the kernel wants an int is a refusal).
    pub fn pack(&self, sym: &str, args: &[Arg]) -> anyhow::Result<Vec<u8>> {
        let (_, d) = self.get(sym)?;
        let exp: Vec<_> = d
            .arg
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden"))
            .collect();
        ensure!(
            exp.len() == args.len(),
            "{sym}: {} explicit args declared, {} given",
            exp.len(),
            args.len()
        );
        let size = exp.iter().map(|a| a.offset + a.size).max().unwrap_or(0) as usize;
        let mut img = vec![0u8; size];
        for (a, v) in exp.iter().zip(args) {
            let o = a.offset as usize;
            let bytes: Vec<u8> = match v {
                Arg::P(x) => x.to_le_bytes().to_vec(),
                Arg::I(x) => x.to_le_bytes().to_vec(),
                Arg::L(x) => x.to_le_bytes().to_vec(),
                Arg::F(x) => x.to_le_bytes().to_vec(),
                Arg::Blob(b) => b.clone(),
            };
            if bytes.len() != a.size as usize {
                bail!(
                    "{sym}: arg {:?} at {o} is {} B, given {} B ({v:?})",
                    a.name,
                    a.size,
                    bytes.len()
                );
            }
            img[o..o + bytes.len()].copy_from_slice(&bytes);
        }
        Ok(img)
    }
}

/// One launch, ready to run.
#[derive(Debug, Clone)]
pub struct Rec {
    pub plugin: &'static str,
    pub sym: String,
    pub grid: [u32; 3],
    pub block: u32,
    pub args: Vec<u8>,
    pub group: Group,
    /// The decoder layer (-1 = outside the layers: final mixer, head, capture).
    pub layer: i32,
    /// A capture-only launch (the materialized residual): not part of the forward's count.
    pub capture: bool,
    /// Dynamic LDS bytes (the split-KV combine: 4 x splits).
    pub shared: u32,
}

/// A bump region of a device allocation (addresses only; the allocation is the caller's).
#[derive(Debug, Clone, Copy)]
pub struct Bump {
    pub base: u64,
    pub cap: u64,
    pub top: u64,
    pub high: u64,
    /// FN119g: where a reset restarts (0; a work issued on a second stream beside live scratch
    /// starts above it) and the end no take may pass (`u64::MAX`; the other stream's region while
    /// it may still be running) -- both fail loud, never silently alias.
    pub floor: u64,
    pub ceil: u64,
}

/// Every buffer is 256-byte aligned (the arena law of the 27B's plans).
pub const ALIGN: u64 = 256;

impl Bump {
    #[must_use]
    pub fn new(base: u64, cap: u64) -> Bump {
        Bump {
            base,
            cap,
            top: 0,
            high: 0,
            floor: 0,
            ceil: u64::MAX,
        }
    }
    /// `bytes` from the region.
    pub fn take(&mut self, bytes: u64) -> anyhow::Result<u64> {
        let at = self.top.div_ceil(ALIGN) * ALIGN;
        let end = at + bytes.max(4);
        ensure!(
            end <= self.cap,
            "arena region over-committed: {end} B of {} B",
            self.cap
        );
        ensure!(
            end <= self.ceil,
            "arena region past its ceiling (FN119g: a second stream's live region): {end} B of {} B",
            self.ceil
        );
        self.top = end;
        self.high = self.high.max(end);
        Ok(self.base + at)
    }
    pub fn reset(&mut self) {
        self.top = self.floor;
    }
}

/// How a buffer starts: the seed law. Everything an op may read before an op of the same list
/// wrote it is seeded (zeros, or all-ones for `row_tok`'s -1); the rest is plain scratch and the
/// poison arm proves nothing reads it early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seed {
    None,
    Zero,
    Ones,
}

/// The per-rank arena: scratch (reset per layer), the two seeded regions (reset per layer,
/// filled at each flush for what was taken since the last one), and the carry ping-pong (the
/// pending triple a layer hands the next one).
#[derive(Debug, Clone)]
pub struct Arena {
    pub scratch: Bump,
    pub zero: Bump,
    pub ones: Bump,
    pub carry: [Bump; 2],
    /// Seeded ranges `(addr, bytes, value)` taken since the last flush.
    pub pending_seeds: Vec<(u64, u64, u8)>,
}

impl Arena {
    /// Carve one allocation of `total` bytes at `base`: scratch, zero, ones, carry x2.
    #[must_use]
    pub fn carve(base: u64, sizes: ArenaSizes) -> Arena {
        let mut at = base;
        let mut next = |n: u64| {
            let b = Bump::new(at, n);
            at += n.div_ceil(ALIGN) * ALIGN;
            b
        };
        let scratch = next(sizes.scratch);
        let zero = next(sizes.zero);
        let ones = next(sizes.ones);
        let c0 = next(sizes.carry);
        let c1 = next(sizes.carry);
        Arena {
            scratch,
            zero,
            ones,
            carry: [c0, c1],
            pending_seeds: Vec::new(),
        }
    }
    /// A buffer under the seed law.
    pub fn take(&mut self, bytes: u64, seed: Seed) -> anyhow::Result<u64> {
        match seed {
            Seed::None => self.scratch.take(bytes),
            Seed::Zero => {
                let a = self.zero.take(bytes)?;
                self.pending_seeds.push((a, bytes.max(4), 0));
                Ok(a)
            }
            Seed::Ones => {
                let a = self.ones.take(bytes)?;
                self.pending_seeds.push((a, bytes.max(4), 0xff));
                Ok(a)
            }
        }
    }
    /// The start of a layer: every per-layer region restarts.
    pub fn new_layer(&mut self) {
        self.scratch.reset();
        self.zero.reset();
        self.ones.reset();
    }
    /// The seeds to issue now, merged into contiguous ranges of one value.
    pub fn drain_seeds(&mut self) -> Vec<(u64, u64, u8)> {
        let mut v = std::mem::take(&mut self.pending_seeds);
        v.sort_unstable();
        let mut out: Vec<(u64, u64, u8)> = Vec::new();
        for (a, n, val) in v {
            if let Some(last) = out.last_mut() {
                let end = last.0 + last.1;
                if last.2 == val && a <= end.div_ceil(ALIGN) * ALIGN {
                    last.1 = (a + n).max(end) - last.0;
                    continue;
                }
            }
            out.push((a, n, val));
        }
        out
    }
}

/// Region sizes of one rank's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSizes {
    pub scratch: u64,
    pub zero: u64,
    pub ones: u64,
    pub carry: u64,
}

impl ArenaSizes {
    /// FN42b: the lendable range of a WIDE arena (`self`, carved by [`Arena::carve`]: scratch
    /// FIRST) whose narrow passes fit `narrow`: `(offset, bytes)` = the wide scratch past the
    /// narrow scratch, its start rounded up to `align`. Every narrow emission fits below it (the
    /// scratch cap) and the other regions sit above it at the wide arena's own offsets, so the
    /// layout is the permanent cut's byte for byte; zero / ones / carry stay permanent.
    #[must_use]
    pub fn borrow_tail(&self, narrow: &ArenaSizes, align: u64) -> (u64, u64) {
        let at = (narrow.scratch.div_ceil(ALIGN) * ALIGN).div_ceil(align) * align;
        let end = self.scratch.div_ceil(ALIGN) * ALIGN;
        (at, end.saturating_sub(at))
    }
    /// FN44b: the lendable range of a WIDE arena under the TWO-LAYOUT borrow: narrow passes run
    /// in the narrow arena carved at the base (`Arena::carve(base, narrow)` -- the c2048 boot's
    /// layout byte for byte: scratch, zero, ones, carry x2 at the narrow rung) and EVERY byte past
    /// it (its end rounded up to `align`) is lendable; a wide pass reclaims it and runs in the
    /// wide arena carved at the same base (scratch first, as the permanent cut). Every region is
    /// per pass (scratch / zero / ones reset per layer and re-seeded at each flush, carry reset
    /// at its layer), so the two layouts never hand a byte across a pass.
    #[must_use]
    pub fn borrow_arena(&self, narrow: &ArenaSizes, align: u64) -> (u64, u64) {
        let at = narrow.total().div_ceil(align) * align;
        (at, self.total().saturating_sub(at))
    }
    #[must_use]
    pub fn total(&self) -> u64 {
        [self.scratch, self.zero, self.ones, self.carry, self.carry]
            .iter()
            .map(|n| n.div_ceil(ALIGN) * ALIGN)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN51d: THE 262k MTP FAULT'S LAW. Under the two-layout borrow (FN44b) a pass CARVED WIDE
    /// takes its zero / ones / carry buffers past the wide scratch -- inside the tail
    /// [`ArenaSizes::borrow_arena`] lends to the expert tier -- while every buffer of the narrow
    /// carve sits below the tail. A drafter pass captured in the wide layout (the serve plane's
    /// first chain after a prompt whose last chunk is past the narrow rung) replays those baked
    /// addresses after the next narrow step lent the tail: its memset seeds and carry rows land in
    /// expert slots and the tier's promotions overwrite its rows. The boot's sizes (fnborrow:
    /// narrow scratch 662.6 of 2559.1 MiB, narrow arena 779.3, wide 3025.5 MiB).
    #[test]
    fn fn51d_a_pass_carved_wide_bakes_its_seeded_and_carry_rows_into_the_lent_tail() {
        let mib = |x: f64| ((x * (1u64 << 20) as f64) as u64).div_ceil(ALIGN) * ALIGN;
        let narrow = ArenaSizes {
            scratch: mib(662.6),
            zero: mib(16.0),
            ones: mib(4.0),
            carry: mib(48.35),
        };
        let wide = ArenaSizes {
            scratch: mib(2559.1),
            zero: mib(64.0),
            ones: mib(16.0),
            carry: mib(193.2),
        };
        let base = 1u64 << 40;
        let (at, len) = wide.borrow_arena(&narrow, 1 << 21);
        let tail = base + at..base + at + len;
        assert!(len > 2 << 30, "the lent tail ({len} B)");
        // one decode-form drafter pass's buffers: a scratch row, a zero-seeded `who`, a carry row
        let pass = |sizes: ArenaSizes| {
            let mut a = Arena::carve(base, sizes);
            let s = a.take(1 << 20, Seed::None).unwrap();
            let z = a.take(1 << 16, Seed::Zero).unwrap();
            let o = a.take(4096, Seed::Ones).unwrap();
            let c = a.carry[0].take(40 * 20_480).unwrap();
            [s, z, o, c]
        };
        let w = pass(wide);
        assert!(
            !tail.contains(&w[0]),
            "the scratch base is shared by both layouts"
        );
        for (name, addr) in ["zero", "ones", "carry"].iter().zip(&w[1..]) {
            assert!(
                tail.contains(addr),
                "wide {name} at {addr:#x} is in the lent tail"
            );
        }
        for addr in pass(narrow) {
            assert!(
                addr < tail.start,
                "a narrow buffer at {addr:#x} sits below the lent tail"
            );
        }
    }

    #[test]
    fn seeds_merge_only_adjacent_ranges_of_one_value() {
        let mut a = Arena::carve(
            1 << 20,
            ArenaSizes {
                scratch: 4096,
                zero: 4096,
                ones: 4096,
                carry: 4096,
            },
        );
        let z0 = a.take(100, Seed::Zero).unwrap();
        let z1 = a.take(300, Seed::Zero).unwrap();
        let o0 = a.take(64, Seed::Ones).unwrap();
        let s = a.take(64, Seed::None).unwrap();
        assert_eq!(z1, z0 + 256);
        let d = a.drain_seeds();
        assert_eq!(d, vec![(z0, 556, 0), (o0, 64, 0xff)]);
        assert!(s < z0);
        assert!(a.drain_seeds().is_empty());
        assert!(a.take(8192, Seed::None).is_err(), "over-commit refuses");
    }

    #[test]
    fn fn42b_the_borrowed_range_is_the_wide_scratch_past_the_narrow_cap() {
        let wide = ArenaSizes {
            scratch: 1 << 20,
            zero: 5000,
            ones: 300,
            carry: 9000,
        };
        let narrow = ArenaSizes {
            scratch: 200_000,
            zero: 4000,
            ones: 300,
            carry: 4500,
        };
        let base = 1 << 32;
        let mut a = Arena::carve(base, wide);
        let (at, n) = wide.borrow_tail(&narrow, 4096);
        assert_eq!(at % 4096, 0);
        assert!(at >= narrow.scratch && at + n == wide.scratch);
        // the narrow cap: an emission that fits it stays below the range; one past it refuses
        a.scratch.cap = narrow.scratch;
        let x = a.take(narrow.scratch - 64, Seed::None).unwrap();
        assert!(x + narrow.scratch - 64 <= base + at);
        assert!(
            a.take(4096, Seed::None).is_err(),
            "the narrow cap refuses the range"
        );
        for r in [a.zero, a.ones, a.carry[0], a.carry[1]] {
            assert!(
                r.base >= base + at + n,
                "the other regions sit above the range"
            );
        }
    }

    #[test]
    fn fn44b_the_two_layout_borrow_lends_every_byte_past_the_narrow_arena() {
        let wide = ArenaSizes {
            scratch: 1 << 20,
            zero: 5000,
            ones: 300,
            carry: 9000,
        };
        let narrow = ArenaSizes {
            scratch: 200_000,
            zero: 4000,
            ones: 300,
            carry: 4500,
        };
        let base = 1u64 << 32;
        let (at, n) = wide.borrow_arena(&narrow, 4096);
        assert_eq!(at % 4096, 0);
        assert!(at >= narrow.total() && at + n == wide.total());
        // more than the scratch tail: zero / ones / carry at the wide rung are lendable too
        let (_, tail) = wide.borrow_tail(&narrow, 4096);
        assert!(n > tail);
        // the narrow layout is the narrow boot's, and every region of it sits below the range
        let a = Arena::carve(base, narrow);
        for r in [a.scratch, a.zero, a.ones, a.carry[0], a.carry[1]] {
            assert!(
                r.base + r.cap <= base + at,
                "a narrow region in the lent range"
            );
        }
        // the wide layout starts at the same base (scratch first): the reclaimed pass owns it all
        let w = Arena::carve(base, wide);
        assert_eq!(w.scratch.base, base);
        assert_eq!(
            w.carry[1].base + wide.carry.div_ceil(ALIGN) * ALIGN,
            base + wide.total()
        );
    }
}
