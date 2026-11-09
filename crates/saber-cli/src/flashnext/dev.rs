//! One rank on its card: the arena allocation, the eager launch of an emitted list (the seeds
//! first, then every launch in order on the rank's compute stream), the per-op timing arm, and the
//! two-rank reduce on the 27B's certified AR kernels at the 27B's band.

use anyhow::Context;
use saber_exec::loader::FunctionTable;
use saber_hip::{DevicePtr, Event, EventFlags, HipThread, MallocFlags, Stream, StreamFlags};
use saber_model::qwen4_exp::layer_graph::Group;

use super::exec::{Emitter, FarChan, HcPre, Reduce, RpChan, EPS, H, W};
use super::kern::{ArenaSizes, Arg};

/// The 27B's reduce rules (`saber_gfx1201.toml [ar]`), the constants the harness's `ArPair` used.
#[derive(Debug, Clone, Copy)]
pub struct ArRules {
    pub quant_min_bytes: u64,
    pub min_nb: u64,
    pub max_nb: u64,
    pub max_nb_exact: u64,
    pub words_per_block: u64,
    pub max_blocks: u64,
    pub threads: u64,
    pub drain: i32,
    pub acq: i32,
    pub group: u64,
    pub chunk_groups_even_bits: u64,
}

impl ArRules {
    /// FN13's boundary-law arm: every reduce on the EXACT kernel (the wht6 band switched off).
    #[must_use]
    pub fn exact_all(mut self) -> ArRules {
        self.quant_min_bytes = u64::MAX;
        self
    }
    pub fn prod() -> anyhow::Result<ArRules> {
        let t: toml::Value = toml::from_str(saber_kernels::saber::PROD_RULES_TEXT)?;
        let a = t.get("ar").context("[ar]")?;
        let g = |k: &str| -> anyhow::Result<i64> {
            a.get(k)
                .and_then(toml::Value::as_integer)
                .with_context(|| format!("ar.{k}"))
        };
        Ok(ArRules {
            quant_min_bytes: g("quant_min_bytes")? as u64,
            min_nb: g("min_nb")? as u64,
            max_nb: g("max_nb")? as u64,
            max_nb_exact: g("max_nb_exact")? as u64,
            words_per_block: g("words_per_block")? as u64,
            max_blocks: g("max_blocks")? as u64,
            threads: g("threads")? as u64,
            drain: g("drain")? as i32,
            acq: g("acq")? as i32,
            group: g("group")? as u64,
            chunk_groups_even_bits: g("chunk_groups_even_bits")? as u64,
        })
    }
}

/// Mailbox size of the reduce (the harness's 16 MiB).
pub const AR_MB: u64 = 16 << 20;

/// FN38: the mailbox a boot whose widest chunk is `max_chunk` rows needs: [`AR_MB`] holds a wht6
/// payload of 4,096 x 2,560 elements; an 8,192-row chunk's reduce needs twice that. The mailbox's
/// size is a kernel argument (layout, not arithmetic: the reduce's bytes do not depend on it).
#[must_use]
pub fn ar_mb(max_chunk: u32) -> u64 {
    let need = u64::from(max_chunk) * H * 6 / 8;
    let mut mb = AR_MB;
    while need > mb / 2 {
        mb *= 2;
    }
    mb
}

/// The wht6 group (`[ar] group`, elements per f32 scale of the local pack).
pub const AR_WHT_GROUP: u64 = 64;

/// FIX-AR-LOC: the bytes of one rank's wht6 LOCAL PACK (`loc`) at mailbox `mb`. The layout is the
/// kernel ABI (CR2-4a spec 2.3.5 / 3.1): the packed symbols from offset 0, then one f32 scale per
/// group of [`AR_WHT_GROUP`] elements from `scale_off = mb / 2`. The emitter admits every wht6
/// reduce whose 6-bit payload fits half a slot (`n * 6 / 8 <= mb / 2`), so the scale array may reach
/// `4 * floor(mb * 2 / 3 / 64)` bytes -- more than the `mb / 32` the pre-fix size left (the 27B's
/// rule bounds n by its bf16 bytes, `n * 2 <= max_bytes`, and `mb / 32` covers it; Flash-Next's
/// bound is the payload's, so the served 8,192-row chunk's n = 20,971,520 needed 1,310,720 B of
/// scales against 1,052,672 B: 258,048 B past `loc` on each rank, CR2-4b `oob_finding`). Kept:
/// `mb / 32` as the floor and the 4,096 B tail.
#[must_use]
pub fn ar_loc_bytes(mb: u64) -> u64 {
    let max_groups = (mb / 2) * 8 / 6 / AR_WHT_GROUP;
    mb / 2 + (max_groups * 4).max(mb / 32) + 4096
}

/// The `fn.buffers` ledger's reduce term at mailbox `mb` (unchanged by FIX-AR-LOC: its 1 MiB
/// slack covers the flags, the seq counters and the larger local pack; asserted by a host test).
#[must_use]
pub fn ar_ledger_bytes(mb: u64) -> u64 {
    2 * mb + mb / 2 + mb / 32 + (1 << 20)
}

/// FN37: the producer-rr reduce's PRIVATE channel -- its flag / sequence slots start at this block
/// index (the exact kernel uses <= 24, wht6 <= 96, of 512) and its scratch window at this byte
/// offset inside each 16 MiB slot (wht6's payload + scales end below 8.5 MiB); a row per block.
pub const HCRR_FLAG0: u64 = 256;
/// FN38: the offset is 3/4 of a slot of `mb` bytes (12 MiB at the 16 MiB mailbox, FN37's constant;
/// wht6's payload + scales end below `mb / 2 + mb / 32`).
#[must_use]
pub fn hcrr_scratch_off(mb: u64) -> u64 {
    mb / 4 * 3
}
/// FN37: the most rows a producer-rr reduce takes (the decode HC bands; its payload window).
pub const HCRR_MAX_ROWS: u64 = 64;
/// FN85: the prologue reduce's private channel -- flags (the pushers' words, <= 8) and the
/// channel counter `ctl` [calls, readers] in the seq array from this word, its mailbox window 384
/// KiB above FN37's (64 rows x 5,120 B = 320 KiB each, inside every slot of [`AR_MB`] or more).
pub const RP_FLAG0: u64 = 384;
#[must_use]
pub fn rp_scratch_off(mb: u64) -> u64 {
    hcrr_scratch_off(mb) + (384 << 10)
}

/// FN114b (`--fn-ar ll`): the flag-in-data reduce's PRIVATE window inside each slot (above FN85's
/// prologue channel). Its payload capacity a slot is [`LL_PAYLOAD`] (64 rows of bf16 = every
/// exact-band reduce: < 128 KiB, or a decode form of <= [`DECODE_EXACT_MAX_ROWS`] rows), its tail
/// (the wave flags) sits in slot 0 right after the payload, and its own seq counters are words
/// [`LL_SEQ0`]..+32 of `ar_seq` (the shipped kernels use <= 96, FN37's from 256, FN85's 384..386).
/// The window and the counters start zeroed (`ar_alloc`) and only this form touches them.
#[must_use]
pub fn ll_scratch_off(mb: u64) -> u64 {
    rp_scratch_off(mb) + (384 << 10)
}
/// FN114b: the flag-in-data window's payload a slot (`flashnext_ar.hip`'s caller contract).
pub const LL_PAYLOAD: u64 = HCRR_MAX_ROWS * H * 2;
/// FN114b: the tail (the race checker's check flags + words, slot 0) after the payload and its zero-mask
/// words (one u32 a 16-byte word: `flashnext_ar.hip`'s `LL_PAYLOAD16`).
pub const LL_TAIL_OFF: u64 = LL_PAYLOAD + LL_PAYLOAD / 4;
/// FN114b: the flag-in-data form's seq counters (word index into `ar_seq`; 32 of them).
pub const LL_SEQ0: u64 = 448;
/// FN114b: the most blocks a flag-in-data call takes (`LL_NC` in `flashnext_ar.hip`).
pub const LL_MAX_BLOCKS: u64 = 32;
/// FN114b: the flag-in-data form's symbol.
pub const LL_SYMBOL: &str = "fn_ar_ll_exact_bf16";

/// FN114d (`--fn-ar gp`): the global-parity one-shot's own counter AND flag words -- words
/// [`GP_WORD0`]..+[`GP_NC`] of `ar_seq` and of `ar_flags` (the shipped one-shot uses 0..96, FN37's
/// producer-rr from 256, FN85's 384..391, FN114b's seq 448..479). The exact and the wht6 forms SHARE
/// them (one call sequence, one parity), they start zeroed (`ar_alloc`) and only these forms touch
/// them: every one of the [`GP_NC`] counters advances once a call whatever the grid, so the slot is
/// the call's parity (`flashnext_ar_gp.hip`).
pub const GP_WORD0: u64 = 128;
/// FN114d: the global-parity form's counters = the most blocks a call takes (`FN114D_GP_NC`).
pub const GP_NC: u64 = 96;
/// FN114d: the global-parity exact reduce (the shipped one-shot's body, the slot by call parity).
pub const GP_EXACT_SYMBOL: &str = "saber_ar_oneshot_2rank_exact_bf16_gp";
/// FN114d: the global-parity wht6 reduce (the shipped wht6 body, the slot by call parity).
pub const GP_WHT6_SYMBOL: &str = "saber_ar_oneshot_2rank_wht6_bf16_gp";

/// FN137 (`--fn-moe-ar-push on`): the fused MoE-out reduce's PRIVATE channel -- its 20 slice
/// flags are words [`FAR_FLAG0`].. of `ar_flags` (FN114b's flag-in-data form keeps its flags in its
/// window, so 448.. of `ar_flags` are free), its channel counter [calls, this call] and its three
/// diagnostic words (spin timeouts, `_chk`'s flag-range and late-word events) are words
/// [`FAR_CTR0`].. of `ar_seq` (after FN114b's 448..480), its window [`far_scratch_off`] above
/// FN114b's in each slot. They start zeroed (`ar_alloc`) and only this form touches them.
pub const FAR_FLAG0: u64 = 448;
/// FN137: the slices a fused back half exchanges (2,560 columns / 128).
pub const FAR_SLICES: u64 = 20;
/// FN137: the channel counter's first word in `ar_seq` ([calls, this call], then the diagnostics).
pub const FAR_CTR0: u64 = 480;
/// FN137: the diagnostic words after the counter ([`FAR_CTR0`] + 2..+5).
pub const FAR_DIAG0: u64 = FAR_CTR0 + 2;
/// FN137: the fused channel's window inside each slot, 64 KiB above FN119's issue-ahead seam
/// window (`dstep::win_off`, 512 KiB above FN114b's, [`super::dstep::WIN_BYTES`] = 40 KiB) --
/// [`FAR_PAYLOAD`] = 64 rows x 5,120 B. (FN137b h1: a window AT `win_off` overwrote the served
/// device step's seam rows: FN119's twin FATAL.)
#[must_use]
pub fn far_scratch_off(mb: u64) -> u64 {
    super::dstep::win_off(mb) + (64 << 10)
}
/// FN137: the fused channel's payload a slot (the fused back half serves T <= 64).
pub const FAR_PAYLOAD: u64 = HCRR_MAX_ROWS * H * 2;
/// FN137: the fused back half's symbol at a split pair (`moe_grouped_far`).
#[must_use]
pub fn far_symbol(gu_ks: u32, dn_ks: u32) -> String {
    format!("fn137_moe_back_h2_far_gu{gu_ks}_dn{dn_ks}_p1")
}

/// FN164 (`--fn-mixer-ar-push on|b1`): the fused MIXER-OUT reduce's PRIVATE channel -- one flag a
/// GEMV workgroup (the out / o projection's 40 workgroups of 64 columns) at words
/// [`MIX_FLAG0`].. of `ar_flags` (after FN137's 448..468), one call counter a workgroup at words
/// [`MIX_CTR0`].. of `ar_seq` (free: FN85's 384..386, FN114b's 448..480), the three diagnostic
/// words (spin timeouts, `_chk`'s flag-range and late-word events) at [`MIX_DIAG0`].. (after
/// FN137's 482..485), the window [`mix_scratch_off`] above FN137's in each slot. They start zeroed
/// (`ar_alloc`) and only this form touches them.
pub const MIX_FLAG0: u64 = 468;
/// FN164: the workgroups a fused out / o projection exchanges (2,560 columns / 64).
pub const MIX_WGS: u64 = 40;
/// FN164: the workgroups' call counters' first word in `ar_seq`.
pub const MIX_CTR0: u64 = 400;
/// FN164: the diagnostic words' first word in `ar_seq` (+0..+3).
pub const MIX_DIAG0: u64 = 488;
/// FN164: the fused mixer channel's window inside each slot, 384 KiB above FN137's (its payload
/// ends 320 KiB above it) -- [`MIX_PAYLOAD`] = 64 rows x 5,120 B.
#[must_use]
pub fn mix_scratch_off(mb: u64) -> u64 {
    far_scratch_off(mb) + (384 << 10)
}
/// FN164: the fused mixer channel's payload a slot (the decode forms' <= 64 rows).
pub const MIX_PAYLOAD: u64 = HCRR_MAX_ROWS * H * 2;
/// FN164: the parents a fused out / o projection replaces (`flashnext_t8far`'s `fn164_far_<parent>`;
/// the decode cells of `Emitter::w8_gemm_q` at N = ldc = 2,560: FN111's n64 tile and FN141's ring
/// (<= 16 rows), FN61e's d2w4 (<= 32) and d4w4 (<= 64) tiles).
pub const MIX_PARENTS: [&str; 4] = [
    "fn111_t8_a8w8_n64_ldc",
    "fn141_t8_a8w8_n64_pf4_ldc",
    "fn_t8p_a8w8_d2w4_ldc",
    "fn_t8p_a8w8_d4w4_ldc",
];
/// FN164: the fused symbol of a parent cell, or `None` (no twin: the reduce stays standalone).
#[must_use]
pub fn mix_symbol(parent: &str) -> Option<String> {
    MIX_PARENTS
        .contains(&parent)
        .then(|| format!("fn164_far_{parent}"))
}

/// FN164 (`--fn-mixer-ar-push`): where the mixer-out reduce rides the out / o projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MixPush {
    /// The standalone reduce after the GEMV (the control).
    #[default]
    Off,
    /// Every TP = 2 target decode / verify form.
    On,
    /// The batch-1 forms only (a single sequence's decode / verify step).
    B1,
}

/// One rank's reduce state (fine-grained scratch + flags, plain seq counters + the wht6 pack).
#[derive(Debug, Clone, Copy, Default)]
pub struct ArState {
    pub scratch: u64,
    pub flags: u64,
    pub seq: u64,
    pub loc: u64,
}

/// The two-rank reduce as emission (host only: every address is already known).
pub struct ArPair {
    /// All stand-alone reductions use the bounded exact registered-host kernel.
    pub host: bool,
    pub rules: ArRules,
    pub st: [ArState; 2],
    /// FN38: the mailbox size both ranks allocated ([`ar_mb`]).
    pub mb: u64,
    pub calls_exact: u64,
    pub calls_wht6: u64,
    /// FN78, the decode-form band law (`--fn-decode-ar exact`, THE DEFAULT; `band` = the control):
    /// a reduce emitted by a DECODE form (a target decode / verify step, a drafter pass; every
    /// emitter's `ar_decode`) of at most [`DECODE_EXACT_MAX_ROWS`] rows runs on the EXACT kernel
    /// (and the producer-rr twin) whatever its payload -- so a b8 verify at width 4 or more (32+
    /// rows, 160+ KiB, wht6 under the payload band) reduces each row exactly as SPEC 0's b8 step does
    /// (the exact two-rank sum is elementwise: a row's bytes do not depend on the rows beside it).
    /// Every cell of the pre-FN78 served law (<= 24 decode rows = 120 KiB) was exact already.
    pub decode_exact: bool,
    /// FN78 which-path counter: reduces the decode-form law moved from wht6 to the exact kernel.
    pub calls_decode_exact: u64,
    /// FN114b (`--fn-ar ll`): every standalone exact reduce runs the flag-in-data form
    /// ([`LL_SYMBOL`]: the same vadd on the same words, byte-equal to the one-shot by its oracle).
    pub ll: bool,
    /// FN114b which-path counter (`fnar114b`): exact reduces emitted on the flag-in-data form.
    pub calls_ll: u64,
    /// FN114d (`--fn-ar gp`): every standalone exact AND wht6 reduce runs its global-parity twin
    /// ([`GP_EXACT_SYMBOL`] / [`GP_WHT6_SYMBOL`]: the shipped bodies, the slot chosen by the call's
    /// parity over [`GP_NC`] counters at [`GP_WORD0`]; byte-equal to the shipped symbols by the
    /// two-rank oracle). Exclusive with [`Self::ll`].
    pub gp: bool,
    /// FN114d which-path counter (`fnar114b`): reduces (exact + wht6) emitted on the gp form.
    pub calls_gp: u64,
    /// FN137 (`--fn-moe-ar-push on`): a fused tp2 back half's MoE-out reduce runs inside it
    /// ([`Reduce::far_channel`]); off = the standalone reduce (the control).
    pub far: bool,
    /// FN137 which-path counters (`fnar137`): reduces the back half carried / reduces the arm
    /// asked for that stood down to the standalone launch.
    pub calls_far: u64,
    pub far_stood_down: u64,
    /// FN164 (`--fn-mixer-ar-push on|b1`): the out / o projection's mixer-out reduce runs inside
    /// it ([`Reduce::mix_channel`]); off = the standalone reduce (the control).
    pub mix: MixPush,
    /// FN164 which-path counters (`fnar164`): reduces the GEMV carried / reduces the arm asked for
    /// that stood down to the standalone launch.
    pub calls_mix: u64,
    pub mix_stood_down: u64,
}

/// FN78: the most rows a decode-form reduce takes on the exact kernel under the decode-form law
/// (the producer-rr window [`HCRR_MAX_ROWS`]; the widest decode form is b8 x w5 = 40 rows).
pub const DECODE_EXACT_MAX_ROWS: u64 = HCRR_MAX_ROWS;

impl ArPair {
    /// FN78: whether this reduce runs on the lossy (wht6) band: the payload band, except a
    /// decode-form reduce of <= [`DECODE_EXACT_MAX_ROWS`] rows under [`Self::decode_exact`].
    #[must_use]
    pub fn lossy(&self, ems: &[Emitter<'_>], n_elem: u64) -> bool {
        n_elem * 2 >= self.rules.quant_min_bytes && !self.decode_exact_at(ems, n_elem)
    }

    fn decode_exact_at(&self, ems: &[Emitter<'_>], n_elem: u64) -> bool {
        self.decode_exact
            && n_elem % H == 0
            && n_elem / H <= DECODE_EXACT_MAX_ROWS
            && !ems.is_empty()
            && ems.iter().all(|e| e.ar_decode)
    }

    /// FN140: the law a STANDALONE decode-form reduce relies on (FN119's pass-1 embeddings,
    /// FN119d's next-verify embeddings): a reduce of `rows` x [`H`] bf16 rows emitted by a decode
    /// form runs on the EXACT kernel iff it is under the payload band OR the decode-form law
    /// (FN78, [`Self::decode_exact`]) covers it (`rows <= DECODE_EXACT_MAX_ROWS`). Refuses (by
    /// name) a reduce the law would put on wht6: > 64 rows, or past the band under
    /// `--fn-decode-ar band` (the control). FN136 h3's FATAL was the pre-FN140 guard, which checked
    /// the PAYLOAD band only (b8 x w4 = 32 rows = 160 KiB) and so refused a reduce FN78 makes exact.
    pub fn decode_form_exact_law(&self, rows: u64, what: &str) -> anyhow::Result<()> {
        anyhow::ensure!(rows >= 1, "{what}: an empty reduce");
        let payload = rows * H * 2;
        if payload < self.rules.quant_min_bytes {
            return Ok(());
        }
        anyhow::ensure!(
            rows <= DECODE_EXACT_MAX_ROWS,
            "{what}: {rows} rows ({payload} B) is past the decode-form exact law \
             (FN78: <= {DECODE_EXACT_MAX_ROWS} rows) -- the payload band would put it on wht6"
        );
        anyhow::ensure!(
            self.decode_exact,
            "{what}: {rows} rows ({payload} B) is past the payload band and `--fn-decode-ar band` \
             (the control) keeps it there -- this reduce must be EXACT"
        );
        Ok(())
    }

    /// FN140: a STANDALONE decode-form reduce of `rows` x [`H`] bf16 rows on the EXACT kernel
    /// (one emitter: a worker's rank): [`Self::decode_form_exact_law`] first, then the reduce with
    /// the emitter marked a decode form (its `ar_decode` restored after), so FN78's law -- not the
    /// emitter's leftover form -- picks the kernel; a lossy pick after that is refused. Counted in
    /// `calls_decode_exact` when the law (not the band) made it exact.
    pub fn reduce_decode_form(
        &mut self,
        em: &mut Emitter<'_>,
        input: u64,
        rows: u64,
        what: &str,
    ) -> anyhow::Result<u64> {
        self.decode_form_exact_law(rows, what)?;
        let was = em.ar_decode;
        em.ar_decode = true;
        let lossy = self.lossy(std::slice::from_ref(em), rows * H);
        let out = if lossy {
            Err(anyhow::anyhow!("{what}: the reduce would run on wht6"))
        } else {
            self.reduce(std::slice::from_mut(em), &[input], rows * H, false)
        };
        em.ar_decode = was;
        Ok(out?[0])
    }
}

impl Reduce for ArPair {
    fn reduce(
        &mut self,
        ems: &mut [Emitter<'_>],
        ins: &[u64],
        n_elem: u64,
        carry: bool,
    ) -> anyhow::Result<Vec<u64>> {
        let a = self.rules;
        let mb = self.mb;
        let nbytes = n_elem * 2;
        let lossy = self.lossy(ems, n_elem);
        if !lossy && nbytes >= a.quant_min_bytes {
            self.calls_decode_exact += 1;
        }
        let mut outs = Vec::with_capacity(2);
        for (i, em) in ems.iter_mut().enumerate() {
            // the rank is the emitter's own (a worker process emits ONE rank; the one-process form both)
            let r = em.dims.rank as usize;
            let out = if carry {
                let slot = (em.layer.max(0) as usize) % 2;
                em.arena.carry[slot].take(nbytes)?
            } else {
                em.arena.take(nbytes, super::kern::Seed::None)?
            };
            let (s, p) = (self.st[r], self.st[1 - r]);
            if !lossy {
                let n16 = nbytes / 16;
                let nb = a.min_nb.max(a.max_nb_exact.min(n16 / a.words_per_block));
                let nb = if self.host {
                    anyhow::ensure!(nbytes <= mb && n_elem % 8 == 0, "host BF16 reduce does not fit its mailbox");
                    super::host_ar::BLOCKS
                } else {
                    nb.min(n16).max(1).min(a.max_blocks)
                };
                if self.ll {
                    anyhow::ensure!(
                        nbytes <= LL_PAYLOAD && nb <= LL_MAX_BLOCKS,
                        "--fn-ar ll: a {nbytes} B reduce on {nb} blocks is past the form's window"
                    );
                    let off = ll_scratch_off(mb);
                    em.run_pub(
                        LL_SYMBOL,
                        [nb, 1, 1],
                        a.threads.clamp(64, 1024),
                        vec![
                            Arg::P(ins[i]),
                            Arg::P(p.scratch + off),
                            Arg::P(s.scratch + off),
                            Arg::P(out),
                            Arg::P(p.scratch + off + LL_TAIL_OFF),
                            Arg::P(s.scratch + off + LL_TAIL_OFF),
                            Arg::P(s.seq + LL_SEQ0 * 4),
                            Arg::I(n16 as i32),
                            Arg::I((mb / 16) as i32),
                            Arg::I(a.drain),
                            Arg::I(a.acq),
                        ],
                        Group::Ar,
                    )?;
                    outs.push(out);
                    continue;
                }
                // FN114d: the gp twin takes the same arguments with its own flag / counter words
                let (sym, w0) = if self.host {
                    (super::host_ar::SYMBOL, 0)
                } else if self.gp {
                    anyhow::ensure!(nb <= GP_NC, "--fn-ar gp: an exact reduce on {nb} blocks");
                    (GP_EXACT_SYMBOL, GP_WORD0 * 4)
                } else {
                    ("saber_ar_oneshot_2rank_exact_bf16", 0)
                };
                em.run_pub(
                    sym,
                    [nb, 1, 1],
                    a.threads.clamp(64, 1024),
                    vec![
                        Arg::P(ins[i]),
                        Arg::P(p.scratch),
                        Arg::P(s.scratch),
                        Arg::P(out),
                        Arg::P(p.flags + w0),
                        Arg::P(s.flags + w0),
                        Arg::P(s.seq + w0),
                        Arg::I(n16 as i32),
                        Arg::I((mb / 16) as i32),
                        Arg::I(a.drain),
                        Arg::I(a.acq),
                    ],
                    Group::Ar,
                )?;
            } else {
                anyhow::ensure!(
                    n_elem % a.group == 0,
                    "wht6: {n_elem} elements, group {}",
                    a.group
                );
                let n_groups = n_elem / a.group;
                let n_chunks = n_groups.div_ceil(a.chunk_groups_even_bits);
                anyhow::ensure!(n_elem * 6 / 8 <= mb / 2, "wht6 payload over the mailbox");
                anyhow::ensure!(
                    mb / 2 + n_groups * 4 <= ar_loc_bytes(mb),
                    "wht6: {n_groups} scales past the local pack"
                );
                let nb = a
                    .min_nb
                    .max(a.max_nb.min(nbytes / (a.words_per_block * 16)));
                let nb = nb.min(n_chunks.max(1)).max(1).min(a.max_blocks);
                let nt = (a.threads.clamp(64, 1024) / 64) * 64;
                let (sym, w0) = if self.gp {
                    anyhow::ensure!(nb <= GP_NC, "--fn-ar gp: a wht6 reduce on {nb} blocks");
                    (GP_WHT6_SYMBOL, GP_WORD0 * 4)
                } else {
                    ("saber_ar_oneshot_2rank_wht6_bf16", 0)
                };
                em.run_pub(
                    sym,
                    [nb, 1, 1],
                    nt,
                    vec![
                        Arg::P(ins[i]),
                        Arg::P(p.scratch),
                        Arg::P(s.scratch),
                        Arg::P(s.loc),
                        Arg::P(out),
                        Arg::P(p.flags + w0),
                        Arg::P(s.flags + w0),
                        Arg::P(s.seq + w0),
                        Arg::I(n_groups as i32),
                        Arg::I(n_chunks as i32),
                        Arg::L(mb as i64),
                        Arg::L((mb / 2) as i64),
                        Arg::I(a.drain),
                        Arg::I(a.acq),
                    ],
                    Group::Ar,
                )?;
            }
            outs.push(out);
        }
        if !lossy {
            self.calls_exact += 1;
            if self.ll {
                self.calls_ll += 1;
            }
        } else {
            self.calls_wht6 += 1;
        }
        if self.gp {
            self.calls_gp += 1;
        }
        let _ = H;
        Ok(outs)
    }

    fn far_channel(&mut self, ems: &[Emitter<'_>], n_elem: u64) -> Option<Vec<FarChan>> {
        if self.host || !self.far {
            return None;
        }
        // the kernel exchanges FAR_SLICES combine slices of 128 columns (its flag words)
        const { assert!(FAR_SLICES * 128 == H) };
        // the exact decode-form reduce only (FN78's law), at the fused back half's rows
        if self.lossy(ems, n_elem)
            || n_elem % H != 0
            || n_elem / H > HCRR_MAX_ROWS
            || !ems.iter().all(|e| e.ar_decode)
            || far_scratch_off(self.mb) + FAR_PAYLOAD > self.mb
        {
            self.far_stood_down += 1;
            return None;
        }
        let ch = ems
            .iter()
            .map(|em| {
                let r = em.dims.rank as usize;
                let (s, p) = (self.st[r], self.st[1 - r]);
                FarChan {
                    peer: p.scratch + far_scratch_off(self.mb),
                    my: s.scratch + far_scratch_off(self.mb),
                    pflags: p.flags + FAR_FLAG0 * 4,
                    mflags: s.flags + FAR_FLAG0 * 4,
                    ctr: s.seq + FAR_CTR0 * 4,
                    diag: s.seq + FAR_DIAG0 * 4,
                    stride16: self.mb / 16,
                }
            })
            .collect();
        if n_elem * 2 >= self.rules.quant_min_bytes {
            self.calls_decode_exact += 1;
        }
        self.calls_exact += 1;
        self.calls_far += 1;
        Some(ch)
    }

    fn far_stood_down(&mut self) {
        if self.far {
            self.far_stood_down += 1;
        }
    }

    fn mix_arm(&self) -> MixPush {
        self.mix
    }

    fn mix_channel(&mut self, ems: &[Emitter<'_>], n_elem: u64) -> Option<Vec<FarChan>> {
        if self.host || self.mix == MixPush::Off {
            return None;
        }
        // the kernel exchanges MIX_WGS workgroups of 64 columns (its flag / counter words)
        const { assert!(MIX_WGS * 64 == H) };
        // the exact decode-form reduce only (FN78's law), at the decode cells' rows
        if self.lossy(ems, n_elem)
            || n_elem % H != 0
            || n_elem / H > HCRR_MAX_ROWS
            || !ems.iter().all(|e| e.ar_decode)
            || mix_scratch_off(self.mb) + MIX_PAYLOAD > self.mb
        {
            self.mix_stood_down += 1;
            return None;
        }
        let ch = ems
            .iter()
            .map(|em| {
                let r = em.dims.rank as usize;
                let (s, p) = (self.st[r], self.st[1 - r]);
                FarChan {
                    peer: p.scratch + mix_scratch_off(self.mb),
                    my: s.scratch + mix_scratch_off(self.mb),
                    pflags: p.flags + MIX_FLAG0 * 4,
                    mflags: s.flags + MIX_FLAG0 * 4,
                    ctr: s.seq + MIX_CTR0 * 4,
                    diag: s.seq + MIX_DIAG0 * 4,
                    stride16: self.mb / 16,
                }
            })
            .collect();
        if n_elem * 2 >= self.rules.quant_min_bytes {
            self.calls_decode_exact += 1;
        }
        self.calls_exact += 1;
        self.calls_mix += 1;
        Some(ch)
    }

    fn mix_stood_down(&mut self) {
        if self.mix != MixPush::Off {
            self.mix_stood_down += 1;
        }
    }

    fn rp_channel(&mut self, ems: &[Emitter<'_>], n_elem: u64) -> Option<Vec<RpChan>> {
        let a = self.rules;
        if self.host || self.lossy(ems, n_elem)
            || n_elem % H != 0
            || n_elem / H > super::exec::DECODE_RP_MAX_ROWS
            || rp_scratch_off(self.mb) + n_elem * 2 > self.mb
        {
            return None;
        }
        let ch = ems
            .iter()
            .map(|em| {
                let r = em.dims.rank as usize;
                let (s, p) = (self.st[r], self.st[1 - r]);
                RpChan {
                    peer_scr: p.scratch + rp_scratch_off(self.mb),
                    my_scr: s.scratch + rp_scratch_off(self.mb),
                    peer_flags: p.flags + RP_FLAG0 * 4,
                    my_flags: s.flags + RP_FLAG0 * 4,
                    ctl: s.seq + RP_FLAG0 * 4,
                    stride16: self.mb / 16,
                    drain: a.drain,
                    acq: a.acq,
                }
            })
            .collect();
        if n_elem * 2 >= a.quant_min_bytes {
            self.calls_decode_exact += 1;
        }
        self.calls_exact += 1;
        Some(ch)
    }

    fn reduce_hc(
        &mut self,
        ems: &mut [Emitter<'_>],
        ins: &[u64],
        n_elem: u64,
        carry: bool,
        next: &[(u64, u64)],
    ) -> anyhow::Result<(Vec<u64>, Vec<Option<HcPre>>)> {
        let a = self.rules;
        let nbytes = n_elem * 2;
        let m = n_elem / H;
        let on = ems.iter().all(|e| e.rules.hc.rr_producer);
        if self.host || !on || self.lossy(ems, n_elem) || n_elem % H != 0 || m > HCRR_MAX_ROWS {
            let outs = self.reduce(ems, ins, n_elem, carry)?;
            let n = outs.len();
            return Ok((outs, vec![None; n]));
        }
        anyhow::ensure!(next.len() == ems.len(), "one (res, inj) per rank");
        let mut outs = Vec::with_capacity(2);
        let mut pres = Vec::with_capacity(2);
        for (i, em) in ems.iter_mut().enumerate() {
            let r = em.dims.rank as usize;
            let slot = (em.layer.max(0) as usize) % 2;
            let out = if carry {
                em.arena.carry[slot].take(nbytes)?
            } else {
                em.arena.take(nbytes, super::kern::Seed::None)?
            };
            // x' and rr outlive this layer's scratch (the next layer's first mix reads them)
            let x = em.arena.carry[slot].take(m * W * 2)?;
            let rr = em.arena.carry[slot].take(m * 4 * 4)?;
            let (s, p) = (self.st[r], self.st[1 - r]);
            let (res, inj) = next[i];
            em.run_pub(
                "fn_ar_exact_hcrr_bf16",
                [m, 1, 1],
                H / 8,
                vec![
                    Arg::P(ins[i]),
                    Arg::P(p.scratch + hcrr_scratch_off(self.mb)),
                    Arg::P(s.scratch + hcrr_scratch_off(self.mb)),
                    Arg::P(out),
                    Arg::P(p.flags + HCRR_FLAG0 * 4),
                    Arg::P(s.flags + HCRR_FLAG0 * 4),
                    Arg::P(s.seq + HCRR_FLAG0 * 4),
                    Arg::P(res),
                    Arg::P(inj),
                    Arg::P(x),
                    Arg::P(rr),
                    Arg::I((self.mb / 16) as i32),
                    Arg::F(EPS),
                    Arg::I(a.drain),
                    Arg::I(a.acq),
                ],
                Group::Ar,
            )?;
            outs.push(out);
            pres.push(Some(HcPre { x, rr }));
        }
        if nbytes >= a.quant_min_bytes {
            self.calls_decode_exact += 1;
        }
        self.calls_exact += 1;
        Ok((outs, pres))
    }
}

/// A host-only reduce for sizing and launch-count tests (same allocations, same launches).
pub fn fake_pair() -> anyhow::Result<ArPair> {
    Ok(ArPair {
        host: false,
        rules: ArRules::prod()?,
        st: [ArState::default(); 2],
        mb: AR_MB,
        calls_exact: 0,
        calls_wht6: 0,
        decode_exact: true,
        calls_decode_exact: 0,
        ll: false,
        calls_ll: 0,
        gp: false,
        calls_gp: 0,
        far: false,
        calls_far: 0,
        far_stood_down: 0,
        mix: MixPush::Off,
        calls_mix: 0,
        mix_stood_down: 0,
    })
}

/// One rank on its card.
pub struct RankDev<'h> {
    pub hip: &'h HipThread,
    pub rank: u32,
    pub stream: Stream,
    /// The arena (shared: FN42b's borrow lends its wide tail to the expert tier's copy queue).
    pub arena: std::rc::Rc<DevicePtr>,
    /// Per-chunk constant tables.
    pub consts: DevicePtr,
    /// The bf16 cos/sin cache.
    pub cs: DevicePtr,
    /// Fill the scratch region with this byte before every layer (the poison arm).
    pub poison: Option<u8>,
    /// Per-op events (the timing arm): (group, layer, symbol, start, end).
    pub op_timing: bool,
    /// FN54b: the gate timer (`--fn-gate-time`): its event pairs and the pairs used this step.
    pub gate_time: bool,
    pub gate_pool: Vec<(Event, Event)>,
    pub gate_used: usize,
    pub timed: Vec<(Group, i32, String, usize)>,
    pool: Vec<(Event, Event)>,
    pub launches: u64,
    /// Memsets issued by `exec` (seeds + the poison arm): graph nodes beside the launches (FN14).
    pub memsets: u64,
    /// FN14's verify arm: `exec` digests the records and the seeds and launches NOTHING.
    pub dry: bool,
    /// FNV-1a over every record (symbol, grid, block, LDS, kernarg image) and seed range `exec` saw
    /// since the caller last reset it: the captured graph's identity.
    pub digest: u64,
    pub ar: Option<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)>,
    /// FN57: how this rank's host waits for the card (`--host-wait`).
    pub host_wait: saber_exec::HostWait,
    /// FN57: the aimed arm's memory (per wait key, the minimum of the recent waits).
    aims: std::cell::RefCell<saber_exec::AimMemory>,
    /// FN57: per step wait (key, step wall ms, wait ms, excess ms = the wait's return past the
    /// card's end as the host can bound it: issue-to-return minus the device span).
    pub waits: std::cell::RefCell<Vec<(u64, f64, f64, f64)>>,
}

/// FNV-1a, folded (the digest of a launch list).
pub(super) fn fnv(h: &mut u64, b: &[u8]) {
    for x in b {
        *h ^= u64::from(*x);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The digest's start value.
pub const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

impl<'h> RankDev<'h> {
    pub fn new(
        hip: &'h HipThread,
        rank: u32,
        sizes: ArenaSizes,
        cs_bytes: &[u8],
    ) -> anyhow::Result<RankDev<'h>> {
        let stream = hip.create_stream(StreamFlags::NonBlocking)?;
        let arena = std::rc::Rc::new(hip.malloc(sizes.total())?);
        hip.memset(&arena, 0, &stream)?;
        let consts = hip.malloc(8 << 20)?;
        let cs = hip.malloc(cs_bytes.len() as u64)?;
        hip.memcpy_h2d_blocking(&cs, cs_bytes, &stream)?;
        Ok(RankDev {
            hip,
            rank,
            stream,
            arena,
            consts,
            cs,
            poison: None,
            op_timing: false,
            gate_time: false,
            gate_pool: Vec::new(),
            gate_used: 0,
            timed: Vec::new(),
            pool: Vec::new(),
            launches: 0,
            memsets: 0,
            dry: false,
            digest: FNV0,
            ar: None,
            host_wait: super::run::DEFAULT_FN_HOST_WAIT,
            aims: std::cell::RefCell::new(saber_exec::AimMemory::default()),
            waits: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// The reduce's allocations on this card (zeroed; peer access is the caller's).
    pub fn ar_alloc(&mut self, rules: &ArRules, mb: u64) -> anyhow::Result<ArState> {
        let hip = self.hip;
        let scratch = hip.malloc_with_flags(2 * mb, MallocFlags::FineGrained)?;
        let flags = hip.malloc_with_flags(rules.max_blocks * 4, MallocFlags::FineGrained)?;
        let seq = hip.malloc(rules.max_blocks * 4)?;
        anyhow::ensure!(
            rules.group == AR_WHT_GROUP,
            "[ar] group {} != the local pack's {AR_WHT_GROUP}",
            rules.group
        );
        let loc = hip.malloc(ar_loc_bytes(mb))?;
        for d in [&scratch, &flags, &seq, &loc] {
            hip.memset(d, 0, &self.stream)?;
        }
        hip.sync_stream(&self.stream)?;
        let st = ArState {
            scratch: scratch.addr().as_u64(),
            flags: flags.addr().as_u64(),
            seq: seq.addr().as_u64(),
            loc: loc.addr().as_u64(),
        };
        self.ar = Some((scratch, flags, seq, loc));
        Ok(st)
    }

    /// FN57: wait for everything on the compute stream by this boot's arm: `spin` / `block` = the
    /// runtime's `hipStreamSynchronize` (an active wait, or the interrupt sleep under the
    /// BlockingSync schedule set at boot); `aimed` = ws-h32's aimed poll at `key`'s recent waits
    /// first, so the runtime's wait that follows finds the work done.
    ///
    /// # Errors
    /// A HIP call failing, or the aimed poll's deadline.
    pub fn wait_stream(&self, key: u64) -> anyhow::Result<()> {
        if self.host_wait == saber_exec::HostWait::Aimed {
            let aim = self.aims.borrow().aim(key);
            let seen = saber_exec::wait_polled(
                || Ok(self.hip.stream_ready(&self.stream)?),
                "the rank's stream",
                saber_exec::WaitWindow::STEP,
                aim,
            )?;
            self.aims.borrow_mut().observe(key, &seen);
        }
        self.hip.sync_stream(&self.stream)?;
        Ok(())
    }

    /// FN57: [`RankDev::wait_stream`] on one event (a step's end).
    ///
    /// # Errors
    /// A HIP call failing, or the aimed poll's deadline.
    pub fn wait_event(&self, ev: &Event, key: u64) -> anyhow::Result<()> {
        if self.host_wait == saber_exec::HostWait::Aimed {
            let aim = self.aims.borrow().aim(key);
            let seen = saber_exec::wait_polled(
                || Ok(self.hip.query_event(ev)?),
                "the step's end",
                saber_exec::WaitWindow::STEP,
                aim,
            )?;
            self.aims.borrow_mut().observe(key, &seen);
        }
        self.hip.sync_event(ev)?;
        Ok(())
    }

    /// FN57: the end-of-run sleeper line: the step waits' excess past the card (p50 / p99 / max),
    /// the waits' own wall, and the TAIL (steps whose wall passed twice their key's median).
    #[must_use]
    pub fn wait_line(&self) -> String {
        let w = self.waits.borrow();
        let q = |mut v: Vec<f64>, p: f64| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * p).round() as usize]
        };
        let ex: Vec<f64> = w.iter().map(|x| x.3).collect();
        let wt: Vec<f64> = w.iter().map(|x| x.2).collect();
        let mut keys: Vec<u64> = w.iter().map(|x| x.0).collect();
        keys.sort_unstable();
        keys.dedup();
        let mut tail = 0usize;
        let mut worst = 0f64;
        for k in keys {
            let st: Vec<f64> = w.iter().filter(|x| x.0 == k).map(|x| x.1).collect();
            let med = q(st.clone(), 0.5);
            for s in st {
                if s > 2.0 * med {
                    tail += 1;
                }
                if med > 0.0 {
                    worst = worst.max(s / med);
                }
            }
        }
        format!(
            "fnwait57 rank{}: arm {} schedule {:?} | {} step waits | excess ms p50 {:.3} p99 {:.3} max {:.3} | wait ms p50 {:.3} p99 {:.3} | tail {} steps > 2x their key's median (worst {:.2}x)",
            self.rank,
            self.host_wait,
            self.hip.device_schedule().ok().flatten(),
            ex.len(),
            q(ex.clone(), 0.5),
            q(ex.clone(), 0.99),
            q(ex, 1.0),
            q(wt.clone(), 0.5),
            q(wt, 0.99),
            tail,
            worst
        )
    }

    #[must_use]
    pub fn base(&self) -> u64 {
        self.arena.addr().as_u64()
    }

    /// Launch everything emitted since the last flush: the seeds, then the launches, in order.
    pub fn exec(&mut self, table: &FunctionTable<'_>, em: &mut Emitter<'_>) -> anyhow::Result<()> {
        let base = self.base();
        if let Some(p) = self.poison {
            if em.recs.iter().any(|r| r.layer >= 0 && !r.capture) {
                fnv(&mut self.digest, &[0xA5, p]);
                if !self.dry {
                    self.hip.memset_at(
                        &self.arena,
                        em.arena.scratch.base - base,
                        em.arena.scratch.cap,
                        p,
                        &self.stream,
                    )?;
                    self.memsets += 1;
                }
            }
        }
        for (a, n, v) in em.arena.drain_seeds() {
            fnv(&mut self.digest, &a.to_le_bytes());
            fnv(&mut self.digest, &n.to_le_bytes());
            fnv(&mut self.digest, &[v]);
            if !self.dry {
                self.hip
                    .memset_at(&self.arena, a - base, n, v, &self.stream)?;
                self.memsets += 1;
            }
        }
        for mut r in std::mem::take(&mut em.recs) {
            fnv(&mut self.digest, r.sym.as_bytes());
            for g in r.grid {
                fnv(&mut self.digest, &g.to_le_bytes());
            }
            fnv(&mut self.digest, &r.block.to_le_bytes());
            fnv(&mut self.digest, &r.shared.to_le_bytes());
            fnv(&mut self.digest, &r.args);
            if self.dry {
                continue;
            }
            let (f, _) = table
                .get(r.plugin, &r.sym)
                .with_context(|| format!("{}::{} not loaded", r.plugin, r.sym))?;
            let ev = if self.op_timing {
                let i = self.timed.len();
                if self.pool.len() <= i {
                    self.pool.push((
                        self.hip.create_event(EventFlags::Default)?,
                        self.hip.create_event(EventFlags::Default)?,
                    ));
                }
                self.hip.record_event(&self.pool[i].0, &self.stream)?;
                self.timed.push((r.group, r.layer, r.sym.clone(), i));
                Some(i)
            } else {
                None
            };
            self.hip
                .launch(
                    f,
                    r.grid,
                    [r.block, 1, 1],
                    r.shared,
                    &self.stream,
                    &mut r.args,
                )
                .with_context(|| format!("launch {}", r.sym))?;
            if let Some(i) = ev {
                self.hip.record_event(&self.pool[i].1, &self.stream)?;
            }
            self.launches += 1;
        }
        Ok(())
    }

    /// Per-op times (ms) of the timing arm since the last call: (group, layer, symbol, ms).
    pub fn take_times(&mut self) -> anyhow::Result<Vec<(Group, i32, String, f64)>> {
        self.hip.sync_stream(&self.stream)?;
        let mut out = Vec::with_capacity(self.timed.len());
        for (g, l, s, i) in std::mem::take(&mut self.timed) {
            let ms = self.hip.elapsed_ms(&self.pool[i].0, &self.pool[i].1)?;
            out.push((g, l, s, f64::from(ms)));
        }
        Ok(out)
    }
}

impl Emitter<'_> {
    /// `run` for the reduce (outside this module's privacy).
    pub fn run_pub(
        &mut self,
        sym: &str,
        grid: [u64; 3],
        block: u64,
        args: Vec<Arg>,
        g: Group,
    ) -> anyhow::Result<()> {
        self.run(sym, grid, block, args, g)
    }
}

#[cfg(test)]
mod tests {
    use super::{ar_ledger_bytes, ar_loc_bytes, ar_mb, ArRules, AR_MB, AR_WHT_GROUP, H};
    use crate::flashnext::run::{MAX_ROWS, WIDE_ROWS_MAX};

    /// The bytes one wht6 reduce of `n` elements writes into `loc` (packed symbols, then f32
    /// scales from `mb / 2`): the spec's layout (CR2-4a 2.3.5 / 3.1).
    fn loc_need(mb: u64, n: u64) -> u64 {
        mb / 2 + n / AR_WHT_GROUP * 4
    }

    /// The pre-fix size (`mb / 2 + mb / 32 + 4096`): the negative control.
    fn loc_old(mb: u64) -> u64 {
        mb / 2 + mb / 32 + 4096
    }

    /// FIX-AR-LOC (CR2-4b `oob_finding`): the local pack covers the scale area of every reduce the
    /// emitter admits, at every served chunk (the clamp MAX_ROWS..=WIDE_ROWS_MAX, the plan's
    /// `ar_mb(max(m, MAX_ROWS))`, below-clamp chunks), at every mailbox the doubling law can pick,
    /// and the pre-fix size fails at the served 8,192-row chunk by exactly the measured 258,048 B.
    #[test]
    fn the_local_pack_covers_every_admitted_wht6_scale_area() {
        let r = ArRules::prod().unwrap();
        assert_eq!(r.group, AR_WHT_GROUP, "[ar] group");
        // every chunk width a boot can serve, its widest reduce = rows x hidden
        for rows in 1..=WIDE_ROWS_MAX {
            for mb in [ar_mb(rows), ar_mb(rows.max(MAX_ROWS))] {
                let n = u64::from(rows) * H;
                if n * 6 / 8 > mb / 2 {
                    continue; // the emitter refuses it (payload over the mailbox)
                }
                assert!(loc_need(mb, n) <= ar_loc_bytes(mb), "rows {rows} mb {mb}");
            }
        }
        // every mailbox the doubling law can pick (an mb override / a wider cap): the widest
        // message the emitter's payload guard admits still fits
        for k in 0..8 {
            let mb = AR_MB << k;
            let n_max = (mb / 2) * 8 / 6 / AR_WHT_GROUP * AR_WHT_GROUP;
            assert!(n_max * 6 / 8 <= mb / 2);
            assert!(loc_need(mb, n_max) <= ar_loc_bytes(mb), "mb {mb}");
            assert!(ar_loc_bytes(mb) >= loc_old(mb), "only `loc` grows");
        }
        // the finding: the served 8,192-row chunk at mb 32 MiB
        let mb = ar_mb(8192);
        assert_eq!(mb, 32 << 20);
        let n = 8192 * H;
        assert_eq!(n, 20_971_520);
        assert_eq!(loc_old(mb), 17_829_888);
        assert_eq!(loc_need(mb, n), 18_087_936);
        assert_eq!(
            loc_need(mb, n) - loc_old(mb),
            258_048,
            "the old size overflows"
        );
        assert_eq!(ar_loc_bytes(mb), 18_179_412);
        assert_eq!(ar_loc_bytes(mb) - loc_old(mb), 349_524, "extra VRAM a rank");
        assert_eq!(ar_loc_bytes(AR_MB) - loc_old(AR_MB), 174_760);
        // the 27B's rule is safe by its bf16 bound (n * 2 <= max_bytes): unchanged
        let rules = saber_kernels::saber::SaberRules::prod_saber().unwrap();
        let mx = rules.ar.max_bytes;
        let cap = rules.workspace["ar_loc_pack"].capacity_bytes;
        assert!(mx / 2 + (mx / 2) / AR_WHT_GROUP * 4 <= cap);
    }

    /// FIX-AR-LOC: the `fn.buffers` ledger's reduce term (unchanged) still covers what
    /// `ar_alloc` allocates with the larger local pack, so the expert-slot remainder and the FN130
    /// boot window are the same numbers as before the fix.
    #[test]
    fn the_ar_ledger_term_covers_ar_alloc() {
        let r = ArRules::prod().unwrap();
        for rows in [MAX_ROWS, 4096, WIDE_ROWS_MAX] {
            let mb = ar_mb(rows);
            let alloc = 2 * mb + 2 * r.max_blocks * 4 + ar_loc_bytes(mb);
            assert!(alloc <= ar_ledger_bytes(mb), "rows {rows}: {alloc}");
        }
        assert_eq!(
            ar_ledger_bytes(32 << 20),
            2 * (32 << 20) + 17_825_792 + (1 << 20)
        );
    }
}
