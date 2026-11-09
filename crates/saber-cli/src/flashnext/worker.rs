//! The Flash-Next TP=2 engine in the WORKER-PROCESS form (FN13; CLAUDE.md rule 3): one process per
//! rank, each with its ONE `HipThread` on its own card. Rank 0 is the process `saber engine` started;
//! it spawns rank 1 as a child (`--fn-rank 1 --fn-peer-port P`, the same argv otherwise) and the
//! pair meet over `saber_exec::TcpExchange` -- the 27B's transport. What crosses it: the reduce's
//! peer-visible scratch + flags as IPC handles at boot (the one-shot all-reduce writes into the
//! peer's scratch, `hipIpcOpenMemHandle` -- so the reduce is the 27B's certified kernel exactly),
//! and per step each rank's top-20 over its half of the vocabulary (the greedy pick is the merge).
//! Nothing else: both ranks run the same request loop from the same inputs (prompts, PLE rows,
//! embedding rows are host facts each process derives itself).
//!
//! Beyond FN12's prefill this driver runs DECODE at SPEC 0 (greedy, one token a step): per-sequence
//! state POOLS (conv / GDN f16 state at slot `s + 1`, KV pages, indexer ring + compressed store, PLE
//! history ping-pong), the decode-form op list (`exec::gdn_decode`, `exec::attn_decode`), the tier's
//! `end_step` per step, the PLE rows of the fed token gathered while layer 0 runs.

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context};
use saber_exec::loader::PluginSet;
use saber_exec::{IpcExchange, LoadCounters, RankPeer, TcpExchange};
use saber_hip::{DevicePtr, EventFlags, HipThread, Job};
use saber_mem::device_mover::{DeviceMover, HipCopyQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, PathCounts, RouteProfile};
use saber_mem::ple_tier::{PleGate, PleIndex, PleSourceKind, PleTier};
use saber_mem::xtier::XtierFile;
use saber_model::qwen4_exp::ple_hash::{PleContext, PleHash};
use saber_model::qwen4_exp::{LayerKind, Qwen4ExpConfig};
use serde_json::{json, Value};

use super::dev::{ArPair, ArRules, ArState, RankDev};
use super::exec::{
    emit_layer, emit_post3, emit_post3_opt, ChunkConsts, Ctl, Dims, Emitter, HcW, LayerIn,
    LayerState, LayerW, MtpW, PadLaw, Reduce, Resident, H, PLE_HIST, RING, VOCAB, W,
};
use super::kern::{Arena, ArenaSizes, Arg, Seed, PLUGINS};
use super::model::{self, Owned, Src, P};
use super::plan::{self, LAYERS};
use super::run::{
    bf16_to_f32, consts, d2h_in, rope_table, warm, Boot, Cap, Demand, FnFlags, Prompt, TierRank,
    MAX_ROWS, ROPE_POS_MAX,
};
use super::serve::{self, Cmd, Job as SrvJob};
use super::store::{embed_rows, Ckpt, PackRecs};

#[path = "pcache.rs"]
mod pcache;
#[path = "verify.rs"]
pub mod verify;
#[cfg(test)]
pub(super) use pcache::chunk_blocks as pcache_blocks;
pub(super) use pcache::staging_bound as pcache_staging_bound;
pub(super) use pcache::{TreeOpts, TREE_PAGE};
use saber_plan::Phase;
use saber_sched::{SeqId, StepResult};
use saber_serve::engine::SampledLogprobs;

/// Sequences the decode pools hold (the conc-8 batch).
pub const MAX_SEQS: u32 = 8;
/// Top-k carried per row per step (FN3's per-step top-20).
const TOPK: usize = 20;
const EOS: u32 = 248_046;

/// Per-sequence state pools on one rank (decode form: slot `s`; conv / GDN at `s + 1`).
pub(super) struct Pools {
    /// FN96: shared so a lent span's owner can be registered on the copy queue.
    bufs: Vec<std::rc::Rc<DevicePtr>>,
    /// FN96: the sequence-major geometry and region 0's address (None = FN13's layer layout).
    sg: Option<(super::seqpool::SeqGeom, u64)>,
    /// Per layer: (conv pool, GDN pool 0, GDN pool 1) for GDN layers; (kv, ring, ringp, comp) for QSA.
    conv: Vec<u64>,
    g: Vec<[u64; 2]>,
    kv: Vec<u64>,
    ring: Vec<u64>,
    ringp: Vec<u64>,
    comp: Vec<u64>,
    ple: [u64; 2],
    /// Bytes per slot of each pool.
    cb: u64,
    hb: u64,
    kvb: u64,
    /// FN68d: the MTP layer's (48's) flat pool bytes a slot at its own page dtype.
    kvb_mtp: u64,
    pb: u64,
    /// Blocks per sequence.
    nbs: u32,
    /// FN15b: GDN state snapshots per sequence in pool 0 (1 at SPEC 0; `k + 1` under `--fn-mtp k`:
    /// one per verify row, the 27B's law) -- sequence `s`'s snapshot `t` is slot `1 + s * pool_w + t`.
    pool_w: u32,
    /// FN45b2: the tiered KV (None = the flat pools: `kv[l]` per layer).
    pub(super) kvt: super::kvwire::KvCell,
    /// FN51: the index store is PAGED (`kvwire::IxPool`: `comp[l]` = one pool of pages, the step
    /// tables' `bt` rows from the allocator) -- admission by reservation; false = the identity.
    pub(super) ix_paged: bool,
    /// FN68: the main K/V pages' dtype (`--fn-kv-dtype`).
    pub(super) kv_dt: saber_mem::kv_tier::KvDtype,
    /// FN68d: the MTP layer's page dtype (`--fn-kv-mtp-dtype`; == `kv_dt` = one dtype).
    pub(super) kv_mtp_dt: saber_mem::kv_tier::KvDtype,
}

impl Pools {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hip: &HipThread,
        dm: &Dims,
        max_ctx: u32,
        state_f32: bool,
        pool_w: u32,
        conv_l: u64,
        kv_tier: bool,
        ix_pages: u32,
        kv_dt: saber_mem::kv_tier::KvDtype,
        kv_mtp_dt: saber_mem::kv_tier::KvDtype,
        layout: super::seqpool::PoolsLayout,
    ) -> anyhow::Result<Pools> {
        // MAX_SEQS sequences + the spare slot (the serve plane's swap)
        let n = u64::from(MAX_SEQS) + 1;
        let nbs = max_ctx.div_ceil(16) + 1;
        let sb = if state_f32 { 4 } else { 2 };
        let cb = dm.conv * conv_l * 2;
        let hb = dm.hv * 128 * 128 * sb;
        // FN68: the flat pools' pages at the dtype's element bytes (fp8 = half)
        let kvb = u64::from(nbs) * dm.nkv * 16 * 512 * kv_dt.elem_bytes();
        // FN68d: the MTP layer's pool at its own dtype
        let kvb_mtp = u64::from(nbs) * dm.nkv * 16 * 512 * kv_mtp_dt.elem_bytes();
        let pb = PLE_HIST * W * 2;
        let mut bufs = Vec::new();
        let mut take = |b: u64| -> anyhow::Result<u64> {
            let d = hip.malloc(b.max(256))?;
            let a = d.addr().as_u64();
            bufs.push(std::rc::Rc::new(d));
            Ok(a)
        };
        // FN96: the sequence-major regions (one allocation: lead + n regions)
        let sg = match layout {
            super::seqpool::PoolsLayout::Seq => {
                let g = super::seqpool::SeqGeom::new(cb, hb, pool_w);
                let a = take(g.bytes(n))?;
                let r0 = a + g.lead;
                Some((g, r0))
            }
            super::seqpool::PoolsLayout::Layer => None,
        };
        let (mut conv, mut g, mut kv, mut ring, mut ringp, mut comp) = (
            vec![0; 49],
            vec![[0; 2]; 49],
            vec![0; 49],
            vec![0; 49],
            vec![0; 49],
            vec![0; 49],
        );
        // index 48: the MTP layer's own KV pages + indexer ring / store (FN15; a QSA layer)
        for l in 0..=LAYERS as usize {
            if l == LAYERS as usize || plan::kind(l as u32) == LayerKind::Qsa {
                // FN45b2: under the tier the pages live in its slabs (`attach_kv` sets the base)
                if !kv_tier {
                    kv[l] = take(n * if l == LAYERS as usize { kvb_mtp } else { kvb })?;
                }
                ring[l] = take(n * RING * 256)?;
                ringp[l] = take(n * RING * 24)?;
                // FN51: the paged store = `ix_pages` + the pad page (the identity: 9 x the context)
                comp[l] = take(if ix_pages > 0 {
                    u64::from(ix_pages + 1) * 4 * 256
                } else {
                    n * u64::from(nbs) * 4 * 256
                })?;
            } else if let Some((sg, r0)) = sg.as_ref() {
                // FN96: the per-layer bases of the region-strided index (slot 1 + s x S / stride)
                let o = sg.off[l];
                conv[l] = r0 + o[0] - cb;
                g[l] = [r0 + o[2] - hb, r0 + o[1] - hb];
            } else {
                conv[l] = take((n + 1) * cb)?;
                g[l] = [take((1 + n * u64::from(pool_w)) * hb)?, take((n + 1) * hb)?];
            }
        }
        let ple = [take(n * pb)?, take(n * pb)?];
        Ok(Pools {
            bufs,
            sg,
            conv,
            g,
            kv,
            ring,
            ringp,
            comp,
            ple,
            cb,
            hb,
            kvb,
            kvb_mtp,
            pb,
            nbs,
            pool_w,
            kvt: None,
            ix_paged: ix_pages > 0,
            kv_dt,
            kv_mtp_dt,
        })
    }
    /// FN45b2: attach the tier: every QSA layer's (and the MTP layer's) `kv` is the tier's base.
    fn attach_kv(&mut self, k: super::kvwire::KvDev) -> anyhow::Result<()> {
        ensure!(
            k.table.nbs() == self.nbs,
            "the tier's table rows ({}) differ from the pools' blocks ({})",
            k.table.nbs(),
            self.nbs
        );
        // FN68d: the MTP layer's twin tier (its own base / table) when its dtype differs
        ensure!(
            k.geom.mtp_dt() == self.kv_mtp_dt && k.geom.dt == self.kv_dt,
            "the tier's page dtypes ({} / MTP {}) differ from the pools' ({} / {})",
            k.geom.dt.name(),
            k.geom.mtp_dt().name(),
            self.kv_dt.name(),
            self.kv_mtp_dt.name()
        );
        let mtp_base = match k.mtp.as_ref() {
            Some(t) => {
                ensure!(
                    t.table.nbs() == self.nbs && t.row_bytes() == k.row_bytes(),
                    "the MTP twin's table rows differ from the main tier's"
                );
                t.base()
            }
            None => k.base(),
        };
        for l in 0..=LAYERS as usize {
            if l == LAYERS as usize || plan::kind(l as u32) == LayerKind::Qsa {
                ensure!(self.kv[l] == 0, "a flat KV pool beside the tier");
                self.kv[l] = if l == LAYERS as usize {
                    mtp_base
                } else {
                    k.base()
                };
            }
        }
        self.kvb = 0;
        self.kvb_mtp = 0;
        self.kvt = Some(std::cell::RefCell::new(k));
        Ok(())
    }
    fn bytes(&self) -> u64 {
        self.bufs.iter().map(|d| d.len()).sum()
    }
    /// FN96: the step tables' index law (None = FN13's: conv `s + 1`, GDN `1 + s x pool_w + t`).
    pub(super) fn idx_law(&self) -> Option<(u32, u32)> {
        self.sg.as_ref().map(|(g, _)| g.idx())
    }
    /// FN96: the layout's lendable spans with their owners (sequence slots `0..MAX_SEQS`).
    fn lend_spans(&self) -> anyhow::Result<Vec<(std::rc::Rc<DevicePtr>, super::seqpool::Span)>> {
        use super::seqpool::Span;
        let mut out = Vec::new();
        if let Some((g, r0)) = self.sg.as_ref() {
            let (d, off) = self.owner(*r0)?;
            let d = self.rc_of(d)?;
            out.push((
                d,
                Span {
                    off,
                    seq_bytes: g.s,
                },
            ));
            return Ok(out);
        }
        for l in 0..LAYERS as usize {
            if plan::kind(l as u32) == LayerKind::Qsa {
                continue;
            }
            // slot 0 = the null slot; sequence s from slot s + 1 (pool 0: 1 + s x pool_w)
            for (base, sb, first) in [
                (self.g[l][0], u64::from(self.pool_w) * self.hb, self.hb),
                (self.g[l][1], self.hb, self.hb),
                (self.conv[l], self.cb, self.cb),
            ] {
                let (d, off) = self.owner(base)?;
                let d = self.rc_of(d)?;
                out.push((
                    d,
                    Span {
                        off: off + first,
                        seq_bytes: sb,
                    },
                ));
            }
        }
        Ok(out)
    }
    fn rc_of(&self, d: &DevicePtr) -> anyhow::Result<std::rc::Rc<DevicePtr>> {
        self.bufs
            .iter()
            .find(|x| std::ptr::eq(x.as_ref(), d))
            .cloned()
            .context("a pool buffer")
    }
    /// FN68d: layer `l`'s flat KV pool bytes a slot (the MTP layer, 48, at its own dtype; 0
    /// under the tier).
    pub(super) fn kvb_of(&self, l: usize) -> u64 {
        if l == LAYERS as usize {
            self.kvb_mtp
        } else {
            self.kvb
        }
    }
    /// FN18: the bytes [`Pools::new`] allocates for these arguments (the exact ledger's
    /// `fn.pools` class; the boot asserts the allocation equals it).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_bytes(
        dm: &Dims,
        max_ctx: u32,
        state_f32: bool,
        pool_w: u32,
        conv_l: u64,
        kv_tier: bool,
        ix_pages: u32,
        kv_dt: saber_mem::kv_tier::KvDtype,
        kv_mtp_dt: saber_mem::kv_tier::KvDtype,
        layout: super::seqpool::PoolsLayout,
    ) -> u64 {
        let n = u64::from(MAX_SEQS) + 1;
        let nbs = u64::from(max_ctx.div_ceil(16) + 1);
        let sb = if state_f32 { 4 } else { 2 };
        let cb = dm.conv * conv_l * 2;
        let hb = dm.hv * 128 * 128 * sb;
        let kvb = nbs * dm.nkv * 16 * 512 * kv_dt.elem_bytes();
        let kvb_mtp = nbs * dm.nkv * 16 * 512 * kv_mtp_dt.elem_bytes();
        let pb = PLE_HIST * W * 2;
        let t = |b: u64| b.max(256);
        let mut sum = 0;
        for l in 0..=LAYERS as usize {
            if l == LAYERS as usize || plan::kind(l as u32) == LayerKind::Qsa {
                let kvp = if kv_tier {
                    0
                } else if l == LAYERS as usize {
                    t(n * kvb_mtp)
                } else {
                    t(n * kvb)
                };
                let ix = if ix_pages > 0 {
                    u64::from(ix_pages + 1) * 4 * 256
                } else {
                    n * nbs * 4 * 256
                };
                sum += kvp + t(n * RING * 256) + t(n * RING * 24) + t(ix);
            } else if layout == super::seqpool::PoolsLayout::Layer {
                sum += t((n + 1) * cb) + t((1 + n * u64::from(pool_w)) * hb) + t((n + 1) * hb);
            }
        }
        if layout == super::seqpool::PoolsLayout::Seq {
            sum += t(super::seqpool::SeqGeom::new(cb, hb, pool_w).bytes(n));
        }
        sum + 2 * t(n * pb)
    }
    /// Sequence `s`'s own bases (a prefill chunk's view: the FN12 tables at index 0).
    fn seq(&self, s: u32, l: usize) -> LayerState {
        let s = u64::from(s);
        // FN96: a GDN layer's rows inside sequence s's region
        let gdn = |x: usize| -> Option<u64> {
            self.sg
                .as_ref()
                .filter(|_| l < LAYERS as usize && plan::kind(l as u32) != LayerKind::Qsa)
                .map(|(g, r0)| r0 + s * g.s + g.off[l][x])
        };
        LayerState {
            cst: gdn(0).unwrap_or(self.conv[l] + (s + 1) * self.cb),
            h: [
                gdn(2).unwrap_or(self.g[l][0] + (1 + s * u64::from(self.pool_w)) * self.hb),
                gdn(1).unwrap_or(self.g[l][1] + (s + 1) * self.hb),
            ],
            kv: self.kv[l] + s * self.kvb_of(l),
            ring: self.ring[l] + s * RING * 256,
            ringp: self.ringp[l] + s * RING * 24,
            // FN51: a paged store's sequence view is the pool (its `bt` row addresses the pages)
            comp: if self.ix_paged {
                self.comp[l]
            } else {
                self.comp[l] + s * u64::from(self.nbs) * 4 * 256
            },
            ple_st: [self.ple[0] + s * self.pb, self.ple[1] + s * self.pb],
        }
    }
    /// The decode view: every pool at its base (the step's tables index slot `s`); PLE parity `p`.
    fn batch(&self, l: usize, p: usize) -> LayerState {
        LayerState {
            cst: self.conv[l],
            h: [self.g[l][0], self.g[l][1]],
            kv: self.kv[l],
            ring: self.ring[l],
            ringp: self.ringp[l],
            comp: self.comp[l],
            ple_st: [self.ple[p], self.ple[1 - p]],
        }
    }
    /// Zero sequence `s`'s every slot (a sequence's start).
    fn zero(&self, hip: &HipThread, dev: &RankDev<'_>, s: u32) -> anyhow::Result<()> {
        if let Some(k) = self.kvt.as_ref() {
            // FN45b2: the slot's old sequence is reclaimed; the new one is admitted by its first
            // step (no page is read before a step wrote it: the attention reads < ctx)
            k.borrow_mut().start(s, 0)?;
        }
        for l in 0..=LAYERS as usize {
            let v = self.seq(s, l);
            let spans: Vec<(u64, u64)> =
                if l == LAYERS as usize || plan::kind(l as u32) == LayerKind::Qsa {
                    vec![
                        (v.kv, self.kvb_of(l)),
                        (v.ring, RING * 256),
                        (v.ringp, RING * 24),
                        (
                            v.comp,
                            if self.ix_paged {
                                0
                            } else {
                                u64::from(self.nbs) * 4 * 256
                            },
                        ),
                    ]
                } else {
                    vec![
                        (v.cst, self.cb),
                        (v.h[0], u64::from(self.pool_w) * self.hb),
                        (v.h[1], self.hb),
                    ]
                };
            for (a, n) in spans {
                if n > 0 {
                    self.memset(hip, dev, a, n)?;
                }
            }
        }
        let v = self.seq(s, 0);
        self.memset(hip, dev, v.ple_st[0], self.pb)?;
        self.memset(hip, dev, v.ple_st[1], self.pb)
    }
    fn owner(&self, a: u64) -> anyhow::Result<(&DevicePtr, u64)> {
        self.bufs
            .iter()
            .find(|d| a >= d.addr().as_u64() && a < d.addr().as_u64() + d.len())
            .map(|d| (d.as_ref(), a - d.addr().as_u64()))
            .context("an address outside the pools")
    }
    fn memset(&self, hip: &HipThread, dev: &RankDev<'_>, a: u64, n: u64) -> anyhow::Result<()> {
        let (d, off) = self.owner(a)?;
        hip.memset_at(d, off, n, 0, &dev.stream)?;
        Ok(())
    }
    fn copy(
        &self,
        hip: &HipThread,
        dev: &RankDev<'_>,
        dst: u64,
        src: u64,
        n: u64,
    ) -> anyhow::Result<()> {
        let (dd, doff) = self.owner(dst)?;
        let (sd, soff) = self.owner(src)?;
        hip.run_batch(&[Job::MemcpyD2d {
            dst: dd,
            dst_offset: doff,
            src: sd,
            src_offset: soff,
            bytes: n,
            stream: &dev.stream,
        }])?;
        Ok(())
    }
    /// Move sequence `s`'s GDN state to pool 0 (decode runs in place there) and its PLE history to
    /// parity `want` (a decode batch shares one parity).
    fn normalize(
        &self,
        hip: &HipThread,
        dev: &RankDev<'_>,
        s: u32,
        gdn_at: usize,
        ple_at: usize,
        want: usize,
    ) -> anyhow::Result<()> {
        if gdn_at == 1 {
            for l in 0..LAYERS as usize {
                if plan::kind(l as u32) != LayerKind::Qsa {
                    let v = self.seq(s, l);
                    self.copy(hip, dev, v.h[0], v.h[1], self.hb)?;
                }
            }
        }
        if ple_at != want {
            let v = self.seq(s, 0);
            self.copy(hip, dev, v.ple_st[want], v.ple_st[ple_at], self.pb)?;
        }
        Ok(())
    }
    /// FN113: the reverse of [`Self::normalize`] -- move sequence `s`'s GDN state and PLE history
    /// to side `want` (a prefill chunk reads the side of its index's parity, `chunks % 2`).
    fn to_side(
        &self,
        hip: &HipThread,
        dev: &RankDev<'_>,
        s: u32,
        gdn_at: usize,
        ple_at: usize,
        want: usize,
    ) -> anyhow::Result<()> {
        if gdn_at != want {
            for l in 0..LAYERS as usize {
                if plan::kind(l as u32) != LayerKind::Qsa {
                    let v = self.seq(s, l);
                    self.copy(hip, dev, v.h[want], v.h[gdn_at], self.hb)?;
                }
            }
        }
        if ple_at != want {
            let v = self.seq(s, 0);
            self.copy(hip, dev, v.ple_st[want], v.ple_st[ple_at], self.pb)?;
        }
        Ok(())
    }
}

/// A live sequence on this rank.
#[derive(Debug, Clone)]
pub(super) struct Live {
    slot: u32,
    /// Tokens in the KV / state so far.
    len: u32,
    /// Chunks prefilled so far (the next prefill chunk's index: parity + has_init).
    chunks: u32,
    /// Where the GDN / PLE state is after the last op (0 / 1).
    gdn_at: usize,
    ple_at: usize,
    ctx: PleContext,
    /// The next token to feed (the last pick).
    next: u32,
    /// FN15b: tokens the last step committed (the GDN snapshot / conv window the next step starts
    /// from) and the rows it rejected (the PLE history's `drop`); 1 / 0 at SPEC 0.
    naccept: u32,
    drop: u32,
    /// FN16 (the serve plane): the drafts the next verify step feeds (F37: proposed after the
    /// commit), the acceptance EMA (`--fn-mtp-dynamic`), and a multi-chunk prompt's stash (the last
    /// row's target stream of a non-last chunk: the drafter's row there needs the NEXT chunk's
    /// first token).
    drafts: Vec<u32>,
    ema: verify::WidthEma,
    /// FN92: the per-position acceptance estimate (the adaptive depth arm; rank 0 feeds it).
    accq: super::depth::SeqAcc,
    stash: Vec<u8>,
    /// FN45c2: the prompt cache's chained key of the chunks prefilled so far.
    key: [u8; 32],
    /// FN58b (`--fn-ple-overlap layer1`): the committed token's PLE row, gathered under the
    /// drafter round -- (the token, the n-gram context it was hashed from, the row's bytes). The
    /// next decode step uses it only when both still match (else it gathers the row itself).
    pre: Option<(u32, PleContext, Vec<u8>)>,
    /// FN55b (`--fn-vision`): an image sequence's M-RoPE table and tower run (None = text).
    img: Option<std::sync::Arc<super::vision::ImgSeq>>,
    /// FN69 (`--spec-lookup on`): the sequence's prompt-lookup index (prompt + committed ids;
    /// shared by the step's clone of this `Live`) and the source of the drafts it holds.
    lk: Option<Arc<std::sync::Mutex<saber_engine::SeqLookup>>>,
    lk_src: Option<saber_engine::lookup::Source>,
    /// FN113: the cut law's state over the prompt's ids so far, the END node to store in
    /// `Cmd::First` (its first position, its key), and the prompt's deepest hit / a miss after it.
    law: saber_mem::prefix_tree::LawState,
    pt_end: Option<(u32, [u8; 32])>,
    pt_hit: Option<saber_mem::prefix_tree::CutKind>,
    pt_partial: bool,
    /// LR-D: this prompt's walk through the tree (the `fnptreereq` line and `/metrics`).
    pt_req: pcache::PtReq,
}

/// FN69: the merge law at one drafting point -- the lookup's continuation (capped at `cap`)
/// REPLACES `lv.drafts` when its match clears the bar; the source is recorded for the commit's
/// per-source tally. Inert without an index.
fn lookup_merge(lv: &mut Live, cap: usize, counts: &mut saber_engine::LookupCounts) {
    let Some(lk) = lv.lk.as_ref() else {
        return;
    };
    let idx = lk.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match idx.merge(cap, lv.drafts.len(), counts) {
        Some(d) => {
            lv.drafts = d;
            lv.lk_src = Some(saber_engine::lookup::Source::Lookup);
        }
        None => lv.lk_src = Some(saber_engine::lookup::Source::Drafter),
    }
}

/// FN69: append ids to the sequence's lookup index (inert without one).
fn lookup_extend(lv: &Live, ids: &[u32]) {
    if let Some(lk) = lv.lk.as_ref() {
        lk.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(ids);
    }
}

impl Live {
    /// A fresh sequence in pool slot `slot` (nothing prefilled).
    fn fresh(slot: u32, eos: u32) -> Live {
        Live {
            slot,
            len: 0,
            chunks: 0,
            gdn_at: 0,
            ple_at: 0,
            ctx: PleContext::start(eos),
            next: 0,
            naccept: 1,
            drop: 0,
            drafts: Vec::new(),
            ema: verify::WidthEma::default(),
            accq: super::depth::SeqAcc::default(),
            stash: Vec::new(),
            key: [0; 32],
            pre: None,
            lk: None,
            lk_src: None,
            img: None,
            law: saber_mem::prefix_tree::LawState::default(),
            pt_end: None,
            pt_hit: None,
            pt_partial: false,
            pt_req: pcache::PtReq::default(),
        }
    }
}

/// Per-step timing + counters.
#[derive(Debug, Clone, Default)]
pub(super) struct StepStat {
    host_ms: f64,
    dev_ms: f64,
    launches: u64,
    resident: u64,
    ram: u64,
    nvme: u64,
    ple_hits: u64,
    ple_nvme: u64,
    ple_ms: f64,
    /// Demand path: distinct routed local records (the NVMe reads' expected count).
    distinct: u64,
    groups: BTreeMap<String, f64>,
    /// FN24 (`--fn-op-timing`): device ms per SYMBOL this step (the census's cell costs).
    ops: BTreeMap<String, f64>,
    busy_ms: f64,
    /// FN54b (`--fn-gate-time`): device ms the compute stream idled at the decode gates, and the
    /// gates that waited.
    gate_ms: f64,
    gate_n: u64,
    /// FN14: the step's host time by phase (ms): pre (embed, tables, PLE rows), tier (xptr tables +
    /// gates), issue (the graph launch or the eager emission), wait (until the step's last event),
    /// tier_end (fence, route log, LRU), pick (the logits' read + both halves' seam).
    host: BTreeMap<&'static str, f64>,
}

/// A child process that is killed and reaped when dropped.
struct Child(Option<std::process::Child>);

impl Child {
    /// FN73b: wait up to `bound` for the child's own exit and reap it (`None` = no child, or it is
    /// still running: the drop kills it).
    fn reap_within(&mut self, bound: Duration) -> Option<std::process::ExitStatus> {
        let c = self.0.as_mut()?;
        let t0 = Instant::now();
        loop {
            if let Ok(Some(st)) = c.try_wait() {
                self.0 = None;
                return Some(st);
            }
            if t0.elapsed() >= bound {
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_secs(20) {
                if let Ok(Some(_)) = c.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Everything one rank owns (the boot's product).
pub(super) struct Rank<'h, 'x> {
    rank: u32,
    hip: &'h HipThread,
    dev: RankDev<'h>,
    table: saber_exec::loader::FunctionTable<'h>,
    lws: Vec<LayerW>,
    fm: HcW,
    /// The bf16 head half (0 under `--fn-trunk w8-head`: not resident, FN61g).
    head: u64,
    /// FN61g: the W8 head half (weight, scales) under `w8-head`.
    head8: Option<(u64, u64)>,
    tier: TierRank<'h, 'x>,
    pools: Pools,
    peer: RankPeer,
    pair: ArPair,
    /// FN100: the row-scaled buffers -- ONE allocation (`super::bufs::layout`), each buffer's
    /// head (its first `--xtier-lend-arena-top-rows` rows) at its base, its tail after it.
    emb: super::bufs::Sub,
    rows: super::bufs::Sub,
    idlog: super::bufs::Sub,
    /// FN100: the tails lent to the expert tier at decode (`--xtier-lend-buffers-top`).
    buft: Option<BufTop>,
    /// FN47b: `--fn-pregate-capture` (the log, the rows of the step it armed, the tallies).
    pg: Option<PgRank>,
    bar: DevicePtr,
    /// FN30: the fused MoE train's `sync` region (zeroed once at boot; read back at the end of a
    /// run: every word must be zero again).
    moe_sync: DevicePtr,
    /// FN120b (`--fn-glue seeds|fold`): the run-wide seed region (`exec::GlueSeeds`; None = `off`).
    glue: Option<DevicePtr>,
    /// FN120c (`--fn-dataflow gdn`): the persistent GDN-tail tables' region (`exec::DfTables`;
    /// zero at boot; None = `off`).
    df: Option<DevicePtr>,
    /// FN124: the in-situ QSA twin's differing-word counter (None = no twin).
    qsa124: Option<DevicePtr>,
    /// FN68b: the fp8 pages' descale table (`kvscale::KvScales::table`; None = bf16 or FN68's
    /// unit law, the null descale pointers). Held for the boot's lifetime (the emitter reads its
    /// address).
    #[allow(dead_code)]
    kv_desc: Option<DevicePtr>,
    /// FN68b: the boot's K/V scale law (None under bf16; the prompt cache keys on its identity).
    kv_scales: Option<super::kvscale::KvScales>,
    max_ctx: u32,
    /// FN51f: `--fn-control band-boot` -- every decode-form step at the boot's max context (the
    /// pre-FN51f split band, `bt` row length and `bt_rows` table; the control).
    band_boot: bool,
    /// D2D copies issued (the route-id log: one per layer of a complete step; graph nodes).
    copies: u64,
    /// The logits of the last pick (the serve plane's dense fallback reads them).
    last_lg: u64,
    /// FN15: the MTP drafter (`--fn-mtp K`), its buffers and per-slot first drafts.
    mtp: Option<MtpRank>,
    /// FN15b: `--fn-tap-mtp` (the ref arm dumps the MTP pass's per-op taps).
    tap_mtp: bool,
    /// FN17: `--fn-pad-control` -- the pre-FN17 pad law (pad row `i` at pool slot `i`).
    pad_ctl: bool,
    /// FN24: `--fn-row-sha` -- every pick hashes its logits rows (the byte law's record).
    row_sha: bool,
    /// FN18: the route trace (`--fn-route-trace`, rank 0 only; byte-neutral) or FN65's recorder
    /// (`--fn-route-record DIR`: rotated files, written off the rank thread).
    trace: Option<saber_mem::route_trace::RouteSink>,
    /// FN18: what the next `tier_end` records -- (kind, bucket, width, live rows).
    trace_meta: (saber_mem::route_trace::StepKind, u32, u32, u32),
    /// FN18: the insert cap by decode bucket (`--xtier-max-inserts-by-bucket`).
    inserts: saber_mem::route_trace::InsertLaw,
    /// FN30b: the drafter-round diagnostic (per-pass log always; probes / gap env-gated).
    diag: Fn30bDiag,
    /// FN42b: the arena borrow (`--fn-arena-borrow on` with a wide rung).
    borrow: Option<Borrow>,
    /// FN76b: the narrow arena's top lent at decode (`--xtier-lend-arena-top on`).
    top: Option<Top>,
    /// FN96: the idle sequence slots' state lent at decode (`--xtier-lend-seq-pools`).
    seql: Option<SeqLend>,
    /// FN51f-b: ELASTIC VRAM (`--fn-vram-elastic on`): the KV hot pool's lendable top.
    elastic: Option<ElasticRank>,
    /// FN51f-b: `--fn-plane-dump DIR` (the residency view's per-layer resident id sets).
    plane_dump: Option<std::path::PathBuf>,
    /// FN42b: this rank's VRAM plan (the borrow's ledger referee).
    budget: saber_mem::flashnext_budget::Ep2Budget,
    /// FN43: the prefill staging ring (`--xtier-stage`).
    ring: Option<Ring>,
    /// FN44b: fill EMPTY lent slots at a prefill pass's start (`--no-fn-lend-fill` = FN44's law:
    /// they stay empty until decode misses fill them).
    lend_fill: bool,
    /// FN44b: prefill passes that filled lent slots, and their host time (ms).
    fill_passes: u64,
    fill_ms: f64,
    /// FN56: the host levers (`--fn-host-levers`).
    lev: super::run::HostLevers,
    /// FN56 (`defer`): a verify step's tier end waiting for the drafter round's launch -- (the
    /// step's tier was complete, its rows, the step's stat slot for the census).
    tier_pend: Option<(bool, u32)>,
    /// FN53c: the valve's device miss flag (`--fn-mem-valve on` + `--fn-valve-graphs on`).
    xmiss: Option<Xmiss>,
    /// FN56: host ms of the deferred tier ends (the census: run under the drafter round).
    tier_def_ms: f64,
    /// FN59b: tier-end host ms by decode bucket (see [`fnhost59_line`]).
    fn59: BTreeMap<u32, [f64; 7]>,
    /// FN56 (`begin`): the expert pointer tables of the placements the last deferred tier end
    /// left are built and uploaded (the next decode step's begin skips them); (built, used).
    tier_pre: bool,
    tier_pre_n: (u64, u64),
    /// FN56: the last pick's host ms -- (the logits' read, the local seam, the exchange + merge).
    pick_t: [f64; 3],
    /// FN57 (`--fn-pick device`): the seam pick's output region on the card + its counters.
    dpick: Option<DevPick>,
    /// FN112 (`--fn-overlap on`): the step's queued reads (staging, event, counters).
    ov112: Option<super::overlap::Ov112>,
    /// FN117b (`--fn-hand`): the host-hand slices' state (None = `none`, the control).
    hand: Option<super::hand::HandRt>,
    /// FN119 (`--fn-device-step ahead`): the issue-ahead's buffers, counters and step in flight.
    ahead: Option<super::dstep::AheadRt>,
    /// FN119 (`ahead`): the next `mtp_gather` (the round's pass 1) runs on the card from the walk's
    /// result block (out address, verify width, sequences).
    ahead_gather: Option<(u64, u32, u32)>,
    /// FN119c: commit / decode applies of > 20 ms (`fnslow119`).
    slow119: u64,
    /// FN57 (`--fn-pin auto`): the CPUs this rank's pick threads run on.
    pick_cpus: Option<Vec<usize>>,
    /// FN57: every pick of the run -- (picks, rows, rows whose logits were read, read ms, seam ms,
    /// exchange ms).
    pick_acc: (u64, u64, u64, f64, f64, f64),
    /// FN116: the first drafter round after every prompt, by term (`F116_TERMS`, ms; DIAGNOSTIC
    /// counters, every byte unchanged) and the last chunk's end (the device idle before it).
    f116: Vec<[f64; F116_N]>,
    f116_chunk_end: Option<Instant>,
    /// FN116 (`--fn-first-step kv`): first rounds run as the K-side front + a steady round, and the
    /// front's pieces (the which-path counters).
    f116_kv: u64,
    f116_front: u64,
    /// FN119e (`--fn-first-step chunk`): the caller asks `prefill` to stage the K-side front under
    /// the last chunk and issue it after the pick (`front_want`); it ran (`front_done`, taken by
    /// `Cmd::First`); the chunk's last logits row's backup read (the dense fallback reads it).
    front_want: bool,
    front_done: bool,
    lg_backup: Option<saber_hip::Event>,
    front_early: u64,
    /// FN119f (`--fn-first-step dev`): the prompt's first-token device block + the decode parity
    /// (set by `Cmd::Prefill` for its last piece), the first round queued behind the chunk (taken
    /// by `Cmd::First`), and the parity the hook normalized the state to (prefill's end keeps it).
    first_ah: Option<(super::dstep::AheadCmd, usize)>,
    first_dev: Option<FirstDev>,
    first_norm: Option<usize>,
    /// FN119g (`--fn-first-step under`): the second stream the K-side front runs on, and the
    /// front's state for the prompt's last piece (wanted by `run_chunk_as`, issued by the chunk's
    /// tail, joined by the first round).
    front_side: Option<saber_hip::Stream>,
    front_under: Option<FrontUnder>,
    /// FN119f, rank 0: the host's first token (and whether it took the dense row) checked against
    /// the card's walk when its `Cmd::First` is sent -- after the reply (TTFT never waits it).
    first_chk: Option<(u32, bool)>,
    /// FN121 (`--fn-ple-prefetch`, `--fn-ple-trace`): the PLE gate's counters, the prefetch's,
    /// and the per-rank step / round / read records.
    pub(super) f121: super::ple121::F121,
    /// FN58b: `--fn-ple-overlap layer1` (the side stream, the pinned staging, the counters).
    ple_ov: Option<PleOv>,
    /// FN14 / FN64: how a captured decode step is cut into graphs (`--fn-graph-split`).
    gsplit: super::run::GraphSplit,
    /// FN58c: the drafter round's host ms by part since the last census read (`rd_*` keys).
    pub(super) rd_t: BTreeMap<&'static str, f64>,
    /// FN55b (`--fn-vision`): the resident tower.
    vis: Option<super::vision::FnVision<'h>>,
    /// FN55b: each POOL SLOT's image table (synced from the serve state's slots before every
    /// command: the tables a step builds read it by the row's pool slot; None = the text law).
    ropes: Vec<Option<std::sync::Arc<super::vision::ImgSeq>>>,
    /// FN55c (`--fn-vision on|lazy`): image sequences may arrive (the rope tables are kept).
    vis_on: bool,
    /// FN55c (`--fn-vision lazy`): the tower's region, lent to the expert tier while no tower is
    /// resident.
    lazy: Option<LazyVis<'h>>,
}

/// FN57: the device pick's region (`MAX_SEQS x WMAX` rows of 512 B, the ledger class `fn.pick`)
/// and its which-path counters.
pub(super) struct DevPick {
    out: DevicePtr,
    rows_max: usize,
    /// Rows picked on the card / rows also checked against the host seam (the row was read for
    /// the row sha) / rows whose device seam differed from the host's (must stay 0).
    rows: u64,
    checked: u64,
    differ: u64,
}

/// FN57: the bytes of the `fn.pick` class: the device pick's rows, or a 256 B stub (host pick).
#[must_use]
pub fn pick_class_bytes(device: bool) -> u64 {
    if device {
        u64::from(MAX_SEQS) * u64::from(super::exec::WMAX) * 512
    } else {
        256
    }
}

/// FN72: the CU count a `--fn-sdma-lanes split` stream's full mask covers (the device's own
/// `hipDeviceAttributeMultiprocessorCount`, the rank's current device).
fn sdma_lanes_cus(hip: &HipThread) -> anyhow::Result<u32> {
    let dev = hip.current_device()?;
    let n = hip.device_attribute(dev, saber_hip::DeviceAttribute::MultiprocessorCount.id())?;
    ensure!(n > 0, "--fn-sdma-lanes: the device reports {n} CUs");
    Ok(n as u32)
}

/// FN57 (`--fn-pin auto`): this box's placement law -- a Ryzen 9 9900X, 24 logical CPUs = 12
/// cores x SMT 2 (sibling of core c = c + 12), two CCDs of 6 cores (0..5, 6..11). Rank r lives
/// on CCD r: its HIP threads (ours and the runtime's, all named `saber-hip`) on core 6r + 1 and
/// its sibling; its pick threads on the CCD's five other cores and their siblings (10 CPUs).
/// The rank's main thread is left to the scheduler. Refuses on any other CPU count.
fn pin_law(rank: u32) -> anyhow::Result<(Vec<usize>, Vec<usize>)> {
    let all = saber_hip::affinity::current_cpus()?;
    ensure!(
        all.len() == 24 && all.iter().copied().eq(0..24),
        "--fn-pin auto is this box's law (24 logical CPUs, 2 CCDs x 6 cores x SMT 2); this process \
         may run on {all:?}"
    );
    let c0 = 6 * rank as usize;
    let hip = vec![c0 + 1, c0 + 13];
    let pick: Vec<usize> = (c0..c0 + 6)
        .filter(|&c| c != c0 + 1)
        .flat_map(|c| [c, c + 12])
        .collect();
    Ok((hip, pick))
}

/// FN58b: the PLE gather beside the step (`--fn-ple-overlap layer1`): the committed token's rows
/// under the drafter round, the drafts' rows under layer 0; the staged rows cross on `stream`
/// and the layer-1 graph (the PLE layer) waits on `done`.
pub(super) struct PleOv {
    /// The side stream (`None`: FN58c's inline form -- the copy on the step's stream).
    stream: Option<saber_hip::Stream>,
    /// FN58c: whether the pre-hook (half A) runs under the drafter round.
    pub(super) pre_on: bool,
    /// Pinned staging, `MAX_SEQS * WMAX` rows of `H * 2` bytes.
    staged: saber_hip::PinnedBuf,
    /// (layer 0's end, the rows' copy done, the layer-1 graph's start) on the step's stream.
    ev: (saber_hip::Event, saber_hip::Event, saber_hip::Event),
    /// The pre-gathers since the last decode step: (host ms, rows, cache hits, drive reads).
    pre: (f64, u64, u64, u64),
}

/// FN58b: the late rows `(row, ids)` gathered in ONE batch into `staged` at their rows; returns
/// (cache hits, drive reads).
fn ple_late(
    ple: &mut PleTier,
    staged: &mut [u8],
    late: &[(usize, Vec<u64>)],
) -> anyhow::Result<(u64, u64)> {
    if late.is_empty() {
        return Ok((0, 0));
    }
    let rb = (H * 2) as usize;
    let ids: Vec<u64> = late.iter().flat_map(|(_, p)| p.iter().copied()).collect();
    ensure!(
        late.iter().all(|(_, p)| p.len() == late[0].1.len()),
        "a PLE row of {} ids",
        late[0].1.len()
    );
    let mut buf = vec![0u8; late.len() * rb];
    let g = ple
        .gather(&ids, &mut buf)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for (k, (r, _)) in late.iter().enumerate() {
        staged[r * rb..(r + 1) * rb].copy_from_slice(&buf[k * rb..(k + 1) * rb]);
    }
    Ok((g.cache_hits, g.nvme_reads))
}

/// FN121: the PLE gate's reading before a decode step's late gather (a prefetch in flight
/// settled, the rows resident vs not = the counted stall), into the step's census keys.
fn gate121(
    rk: &mut Rank<'_, '_>,
    ple: &mut PleTier,
    late: &[(usize, Vec<u64>)],
    stat: &mut StepStat,
) -> anyhow::Result<()> {
    let ids: Vec<u64> = late.iter().flat_map(|(_, p)| p.iter().copied()).collect();
    let (w, rows, miss) = rk.f121.gate(ple, &ids)?;
    *stat.host.entry("fn121_inflight_ms").or_insert(0.0) += w;
    *stat.host.entry("fn121_rows").or_insert(0.0) += rows as f64;
    *stat.host.entry("fn121_miss_rows").or_insert(0.0) += miss as f64;
    Ok(())
}

/// FN121: a gate that gathered rows from the drive bills the gather's ms to the stall.
fn miss121(f121: &mut super::ple121::F121, stat: &StepStat, g1: Instant) {
    if stat.host.get("fn121_miss_rows").copied().unwrap_or(0.0) > 0.0 {
        f121.miss_ms += g1.elapsed().as_secs_f64() * 1e3;
    }
}

/// FN121 (`--fn-ple-prefetch cand`, THE PLE PREFETCH LAW): the next verify step's candidate
/// tokens' n-gram rows -- every live sequence's committed token and ALL its drafts, hashed exactly
/// as `decode_step` will (from `lv.ctx` over `[next, d1, d2, ..]`) -- submitted to the drive in
/// ONE batch without waiting; they land in the row cache before the step's gate (every gather
/// settles a prefetch first). Byte-neutral: the cache holds the drive's bytes. Returns the rows
/// submitted.
pub(super) fn ple_prefetch_cand<'a>(
    rk: &mut Rank<'_, '_>,
    ple: &mut PleTier,
    ph: &PleHash,
    lives: impl Iterator<Item = &'a Live>,
) -> anyhow::Result<u64> {
    if !rk.f121.form.cand {
        return Ok(0);
    }
    let t = Instant::now();
    let mut ids: Vec<u64> = Vec::new();
    for lv in lives {
        let mut c = lv.ctx;
        ids.extend(ph.ids_step(&mut c, &[lv.next]));
        for &d in &lv.drafts {
            ids.extend(ph.ids_step(&mut c, &[d]));
        }
    }
    let n = ple.prefetch(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    rk.f121.pf_calls += 1;
    rk.f121.pf_rows += n;
    rk.f121.pf_ms += t.elapsed().as_secs_f64() * 1e3;
    Ok(n)
}

/// FN58b: the drafter round's pre-gather hook (host ms, rows, cache hits, drive reads).
pub(super) type PreHook<'a> = &'a mut dyn FnMut() -> anyhow::Result<(f64, u64, u64, u64)>;
/// FN116: work the drafter round runs between its pass 1 and the chain (device chain only).
pub(super) type MidHook<'a> = &'a mut dyn FnMut(&mut Rank<'_, '_>) -> anyhow::Result<()>;
/// FN119f: work queued between the issue-ahead's walk and its round (the first round's front).
pub(super) type PreRound<'a> =
    &'a mut dyn FnMut(&mut Rank<'_, '_>, &mut Emitter<'_>) -> anyhow::Result<()>;
/// FN119f: work queued right behind a prompt's last chunk, before the host waits for it (the
/// chunk's live state, its logits address).
pub(super) type AfterChunk<'a> =
    &'a mut dyn FnMut(&mut Rank<'_, '_>, &mut Emitter<'_>, &mut Live, u64) -> anyhow::Result<()>;

/// FN58b: one row's PLE ids for every live sequence's committed token (`lv.next` hashed on a copy
/// of `lv.ctx`), gathered in ONE batch and stashed on the sequence (`Live::pre`); returns (host
/// ms, rows, cache hits, drive reads). Byte-neutral: the decode step re-hashes the same ids and
/// uses the stash only on an exact (token, context) match.
pub(super) fn ple_pregather<'a>(
    ple: &mut PleTier,
    ph: &PleHash,
    lives: impl Iterator<Item = &'a mut Live>,
) -> anyhow::Result<(f64, u64, u64, u64)> {
    let t = Instant::now();
    let rb = (H * 2) as usize;
    let mut who: Vec<&'a mut Live> = Vec::new();
    let mut ids: Vec<u64> = Vec::new();
    for lv in lives {
        let mut c = lv.ctx;
        ids.extend_from_slice(&ph.ids_step(&mut c, &[lv.next]));
        who.push(lv);
    }
    if who.is_empty() {
        return Ok((0.0, 0, 0, 0));
    }
    let mut buf = vec![0u8; who.len() * rb];
    let g = ple
        .gather(&ids, &mut buf)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let n = who.len() as u64;
    for (i, lv) in who.into_iter().enumerate() {
        lv.pre = Some((lv.next, lv.ctx, buf[i * rb..(i + 1) * rb].to_vec()));
    }
    Ok((
        t.elapsed().as_secs_f64() * 1e3,
        n,
        g.cache_hits,
        g.nvme_reads,
    ))
}

impl Rank<'_, '_> {
    /// FN17: the pad law of a decode step with `live` live sequences (the pad rows at the SPARE
    /// slot; `--fn-pad-control` = FN14's law, pad `i` at slot `i`).
    fn pads(&self, live: usize) -> PadLaw {
        PadLaw {
            live,
            pad: (!self.pad_ctl).then_some(SPARE),
            // FN96: the sequence-major pools' index law
            idx: self.pools.idx_law(),
        }
    }
}

/// FN15: the MTP drafter on one rank.
pub(super) struct MtpRank {
    w: MtpW,
    /// Chain length (passes per drafting point).
    k: u32,
    /// The target's pre-final-mixer stream of the last prefill chunk / decode step `[rows, W]`
    /// (FN100: a buffer of the one `fn.buffers` allocation).
    hid: super::bufs::Sub,
    /// The drafter's own stream at each sequence's drafted row `[MAX_SEQS + 1, W]` (the next pass's
    /// input; row `s` = pool slot `s`, moved with the slot by the serve plane's compaction).
    prev: DevicePtr,
    /// FN16: one row `[W]`: a multi-chunk prompt's stashed target stream, uploaded for the drafter's
    /// row at the chunk boundary.
    stash: DevicePtr,
    /// Per pool slot: the first pass's draft after a prefill and its row position.
    first: Vec<Option<(u32, u32)>>,
    /// Per-pass device times of the last chain (ms).
    pass_ms: Vec<f64>,
    /// The launch count of one pass (prefill form / decode form), asserted.
    launches: BTreeMap<String, u32>,
    /// FN16 (a drafter lever, bit-exact): the decode-form passes CAPTURED on first use per (rows,
    /// width, stream input) and replayed -- `--no-fn-mtp-capture` = the eager control.
    capture: bool,
    graphs: BTreeMap<(u32, u32, u64, bool, u64, u32), MtpGraph>,
    /// FN16 (a drafter lever, bit-exact): the drafts are the rows' ARGMAX only -- one pass per row
    /// per rank and 8 bytes a row across the pair instead of the 64-entry seam (`--no-fn-mtp-argmax`
    /// = the seam control).
    argmax: bool,
    /// FN22: the DEVICE-SIDE chain (the default; `--no-fn-mtp-device-chain` = FN16's host chain).
    chain: Option<DevChain>,
    /// FN22: host time of the drafter rounds by segment (ms, count) -- the 1.8 ms's composition.
    seg: BTreeMap<&'static str, (f64, u64)>,
    /// FN22: the device chain's pass events, read once the chain's ids are on the host.
    evs: Vec<(saber_hip::Event, saber_hip::Event)>,
    /// FN22: `--fn-mtp-chain-graph` -- passes 2..k captured as ONE graph per shape.
    chain_graphs: BTreeMap<(u32, u32, u32, Vec<bool>, u32), MtpGraph>,
    /// FN51d: every DECODE-FORM drafter pass runs under the borrow law's LENT (narrow) arena
    /// ([`borrow_for`] at its rows) -- a pass captured in the wide arena bakes its zero / ones /
    /// carry regions into the tail the next narrow step lends to the expert tier (the 262k
    /// fault). `--fn-control mtp-wide-capture` = the pre-FN51d law (the negative control).
    lend: bool,
}

/// FN22: the drafter's device-side chain on one rank (`flashnext_draft.hip`, `saber_kernels::fn22`).
pub(super) struct DevChain {
    /// The rank's half of `embed_tokens` `[124,160, H]` bf16 (the checkpoint's bytes); FN99: in
    /// VRAM or pinned RAM (`--fn-embed-placement`).
    embh: model::EmbedHalf,
    /// FN99 (`fnembed99`): embedding rows the chain's tails read (one a drafted row, pads too)
    /// and the passes that read them.
    emb_rows: u64,
    emb_passes: u64,
    /// The chain's inputs, ONE upload a chain: the pass counter (u32 at 0), the pass-1 row
    /// selection (i32 x 8 at [`IN_SEL1`]), the chain passes' (x 8 at [`IN_SELC`]), the chain
    /// passes' step tables (one slot of `slot` bytes each from [`IN_STAGE`]).
    inb: DevicePtr,
    /// The chain's ids `[MAX_PASSES, IDS_STRIDE]` u32 (row = the counter at the pass).
    ids: DevicePtr,
    /// One table slot's bytes (the b8 decode tables at the pools' context, 256-aligned).
    slot: u64,
    /// The tie law's control (`--fn-control chain-tie-high`; never served).
    tie_high: bool,
    /// `--fn-mtp-chain-check`: every pass compared with the host law on the same logits.
    check: Option<ChainCheck>,
    /// `--fn-mtp-chain-graph`.
    graph: bool,
    /// FN33: the int2 draft head (`--fn-draft-head int2`): the pack + K.
    int2: Option<Int2Head>,
    /// FN33: the last int2 pass's candidates and scores (the check arm reads them; eager passes).
    i2last: Option<I2Last>,
    /// FN154 (`--fn-mtp-conf-log`): the census arm's confidences (eager passes, no ahead).
    conf: Option<ConfLog>,
    /// LR-N (`--spec-couple on`): coupled drafting's per-step rows + counters; `None` = off.
    couple: Option<CoupleRt>,
}

/// LR-N: coupled drafting's state on one rank.
#[derive(Debug, Default)]
pub(super) struct CoupleRt {
    /// The last `Cmd::Decode`'s rows, one a ready slot (empty = no coupled step).
    cur: Vec<super::couple::CoupleRow>,
    /// Rounds whose inputs carried coupled rows, the rows coupled, the rows left greedy.
    rounds: u64,
    rows: u64,
    greedy: u64,
}

/// FN154: the confidence census -- per drafter round, per pass (draft position), per live row:
/// the merged (both ranks) top-1 id, its candidate-softmax prob, the top-1 / top-2 margin and the
/// candidates' entropy; rank 0 writes one JSON line a verify step.
pub(super) struct ConfLog {
    out: Option<std::io::BufWriter<std::fs::File>>,
    /// The last round's rows: `[pass][row] = (id, p, margin, entropy)`.
    round: Vec<Vec<(u32, f32, f32, f32)>>,
    /// Steps skipped (the round's rows != the step's), sequences whose drafts were not the
    /// round's top-1 ids (dropped).
    skipped: u64,
    mismatched: u64,
}

impl ConfLog {
    fn open(p: &Path, rank: u32) -> anyhow::Result<ConfLog> {
        let out = if rank == 0 {
            let f = std::fs::File::create(p)
                .with_context(|| format!("FN154: --fn-mtp-conf-log {}", p.display()))?;
            Some(std::io::BufWriter::new(f))
        } else {
            None
        };
        Ok(ConfLog {
            out,
            round: Vec::new(),
            skipped: 0,
            mismatched: 0,
        })
    }
}

/// FN154: one drafter pass's confidences of its first `n` output rows (the int2 tail's candidates:
/// each rank's K exact scores reduced to (top-1 id, top-1, top-2, log-sum-exp, mean score), the
/// pair's 20 bytes a row swapped and merged). `first` starts a round. Syncs the stream.
fn conf_pass(rk: &mut Rank<'_, '_>, n: usize, first: bool) -> anyhow::Result<()> {
    rk.hip.sync_stream(&rk.dev.stream)?;
    let l = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .and_then(|c| c.i2last)
        .context("FN154: the conf arm reads the int2 tail's candidates (--fn-draft-head int2)")?;
    let half = u32::try_from(VOCAB / 2)?;
    let (r, k, ldc) = (l.rows as usize, l.k as usize, l.ldc as usize);
    ensure!(n <= r, "FN154: {n} live rows past the tail's {r}");
    let cb = d2h_in(&rk.dev, &rk.dev.arena, l.cand, r * k * 4)?;
    let sb = d2h_in(&rk.dev, &rk.dev.arena, l.sc, r * ldc * 2)?;
    let mut mine = Vec::with_capacity(n * 20);
    for i in 0..n {
        let (mut m1, mut m2, mut id1) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0u32);
        let sc: Vec<f32> = (0..k)
            .map(|j| {
                let o = (i * ldc + i * k + j) * 2;
                f32::from_bits(u32::from(u16::from_le_bytes([sb[o], sb[o + 1]])) << 16)
            })
            .collect();
        for (j, &v) in sc.iter().enumerate() {
            let o = (i * k + j) * 4;
            let c = u32::from_le_bytes([cb[o], cb[o + 1], cb[o + 2], cb[o + 3]]);
            if v > m1 {
                m2 = m1;
                m1 = v;
                id1 = c + rk.rank * half;
            } else if v > m2 {
                m2 = v;
            }
        }
        let z: f64 = sc.iter().map(|&v| f64::from(v - m1).exp()).sum();
        let lse = f64::from(m1) + z.ln();
        let mean: f64 = sc
            .iter()
            .map(|&v| (f64::from(v) - lse).exp() * f64::from(v))
            .sum();
        mine.extend_from_slice(&id1.to_le_bytes());
        mine.extend_from_slice(&m1.to_le_bytes());
        mine.extend_from_slice(&m2.to_le_bytes());
        mine.extend_from_slice(&(lse as f32).to_le_bytes());
        mine.extend_from_slice(&(mean as f32).to_le_bytes());
    }
    let theirs = rk.peer.exchange().xchg_bytes("fn154.conf", &mine)?;
    ensure!(
        theirs.len() == mine.len(),
        "FN154: the peer's conf frame differs in size"
    );
    let word = |b: &[u8], i: usize, q: usize| -> [u8; 4] {
        let o = i * 20 + q * 4;
        [b[o], b[o + 1], b[o + 2], b[o + 3]]
    };
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let side = |b: &[u8]| {
            (
                u32::from_le_bytes(word(b, i, 0)),
                f32::from_le_bytes(word(b, i, 1)),
                f32::from_le_bytes(word(b, i, 2)),
                f64::from(f32::from_le_bytes(word(b, i, 3))),
                f64::from(f32::from_le_bytes(word(b, i, 4))),
            )
        };
        let (a, b) = (side(&mine), side(&theirs));
        // the id is checked against the fed drafts at the step (the tie law is not re-derived)
        let (hi, lo) = if a.1 > b.1 || (a.1 == b.1 && a.0 < b.0) {
            (a, b)
        } else {
            (b, a)
        };
        let t2 = hi.2.max(lo.1);
        let lse = a.3.max(b.3) + (-(a.3 - b.3).abs()).exp().ln_1p();
        let mean = (a.3 - lse).exp() * a.4 + (b.3 - lse).exp() * b.4;
        let p = (f64::from(hi.1) - lse).exp() as f32;
        rows.push((hi.0, p, hi.1 - t2, (lse - mean) as f32));
    }
    let cl = rk
        .mtp
        .as_mut()
        .and_then(|m| m.chain.as_mut())
        .and_then(|c| c.conf.as_mut())
        .context("FN154: the conf arm")?;
    if first {
        cl.round.clear();
    }
    cl.round.push(rows);
    Ok(())
}

/// FN154: one verify step's sequence for the census line (seq id, slot, the drafts it fed, the
/// drafts the walk accepted).
type ConfSeq<'a> = (u64, usize, &'a [u32], u32);

/// FN154: rank 0's line for one verify step.
fn conf_line(
    rk: &mut Rank<'_, '_>,
    n: usize,
    bucket: u32,
    w: u32,
    ms: f64,
    seqs: &[ConfSeq<'_>],
) -> anyhow::Result<()> {
    use std::io::Write as _;
    let Some(cl) = rk
        .mtp
        .as_mut()
        .and_then(|m| m.chain.as_mut())
        .and_then(|c| c.conf.as_mut())
    else {
        return Ok(());
    };
    if cl.out.is_none() {
        return Ok(());
    }
    if cl.round.is_empty() || cl.round[0].len() != n {
        cl.skipped += 1;
        return Ok(());
    }
    let mut rows = Vec::with_capacity(seqs.len());
    for &(seq, slot, d, acc) in seqs {
        if d.is_empty() {
            continue;
        }
        let ok = d.len() <= cl.round.len()
            && d.iter()
                .enumerate()
                .all(|(j, &t)| cl.round[j].get(slot).is_some_and(|r| r.0 == t));
        if !ok {
            cl.mismatched += 1;
            continue;
        }
        let c: Vec<[f32; 3]> = (0..d.len())
            .map(|j| {
                let r = cl.round[j][slot];
                [r.1, r.2, r.3]
            })
            .collect();
        rows.push(json!({"seq": seq, "slot": slot, "acc": acc, "c": c}));
    }
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |x| x.as_secs_f64() * 1e3);
    let line = json!({"t": t, "n": n, "b": bucket, "w": w, "rows": n as u32 * w, "ms": ms,
        "skip": cl.skipped, "mism": cl.mismatched, "s": rows});
    if let Some(out) = cl.out.as_mut() {
        writeln!(out, "{line}")?;
        out.flush()?;
    }
    Ok(())
}

/// FN33: the rank's int2 draft head (`fn_draft_head_int2_quant` of the bf16 head half at boot;
/// the ledger class `fn.draft.int2`).
pub(super) struct Int2Head {
    wq: DevicePtr,
    s: DevicePtr,
    zs: DevicePtr,
    /// Candidates per rank (`--fn-draft-head-k`).
    k: u32,
}

/// FN33: what an int2 tail reads (the pack, the bf16 head half, K; `full_head` = the check arm
/// also runs the full bf16 head for the comparison).
#[derive(Debug, Clone, Copy)]
pub(super) struct I2Args {
    wq: u64,
    s: u64,
    zs: u64,
    head: u64,
    /// FN61g: the W8 head (weight, scales) when the bf16 head is not resident (`w8-head`): the
    /// rerank scores its candidates from these bytes (`fn_t8r_a8w8_gather_m16n64`), the bytes the
    /// verify step's head reads.
    head8: Option<(u64, u64)>,
    k: u32,
    full_head: bool,
}

/// FN61g: the rerank's kernel on the W8 head (one 16-row tile, 64 candidate columns a workgroup).
pub(super) const T8R_GATHER: &str = "fn_t8r_a8w8_gather_m16n64";

/// FN33: an int2 tail's candidates `[rows, k]` i32 (local ids) and exact scores `[rows, ldc]` bf16
/// (row r's candidates at columns `r * k ..`).
#[derive(Debug, Clone, Copy)]
pub(super) struct I2Last {
    cand: u64,
    sc: u64,
    ldc: u32,
    rows: u32,
    k: u32,
}

const IN_SEL1: u64 = 16;
const IN_SELC: u64 = 48;
const IN_STAGE: u64 = 256;

/// FN22: the check arm's counters and records (one JSON line per pass).
#[derive(Default)]
pub(super) struct ChainCheck {
    passes: u64,
    rows: u64,
    id_diff: u64,
    emb_diff: u64,
    /// FN33: candidate scores whose bits differ from the full head's logit (must be 0).
    sc_diff: u64,
    /// FN33: rows whose rank-local bf16 argmax is not among the int2 candidates.
    outside: u64,
    lines: Vec<String>,
}

/// FN22: the tail of a device-chain pass: where its argmax reads its rows and writes its outputs.
#[derive(Debug, Clone, Copy)]
pub(super) struct TailArgs {
    embh: u64,
    ids: u64,
    ctr: u64,
    /// i32 `[rows]` logits row per output row (-1 = a pad row: token 0).
    sel: u64,
    rows: u32,
    /// The next pass's embedding input `[rows, H]`.
    emb: u64,
    tie_high: bool,
    /// FN33: the int2 head's candidates + the exact rerank instead of the full head's argmax.
    i2: Option<I2Args>,
    /// LR-N (`--spec-couple on`): the coupled tail (`flashnext_couple`) reading the chain's
    /// input buffer (`ctr`'s base) for each row's noise key, position, 1 / T and top_k.
    couple: bool,
}

/// FN16: one captured drafter pass (decode form): the graph and the logits / stream it leaves.
struct MtpGraph {
    exec: saber_hip::GraphExec,
    _graph: saber_hip::Graph,
    lg: u64,
    strm: u64,
    /// FN18: the pass's route-id log in the arena (the route trace reads it after the sync).
    ids: u64,
}

/// Rank 0 (or rank 1 when `--fn-rank 1`): the worker-process engine.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn engine_worker(
    hip_lib: &str,
    boot: Boot,
    pack: &Path,
    ckpt: &Path,
    cfg: &Qwen4ExpConfig,
    prebuilt: &Path,
    prompts: &[Prompt],
    out: &Path,
    f: &FnFlags,
) -> anyhow::Result<()> {
    // FN55c: the lazy tower's arena as the plan chose it (the lend capped to pin)
    let fv;
    let f = match boot.vision_patches {
        Some(p) => {
            fv = FnFlags {
                fn_vision_max_patches: Some(p),
                ..f.clone()
            };
            &fv
        }
        None => f,
    };
    let rank = f.fn_rank;
    ensure!(rank < 2, "--fn-rank 0|1");
    let t_boot = Instant::now();
    // ---- the pair: rank 0 spawns rank 1 (same argv + the rank and the port) and accepts it
    let mut child = Child(None);
    // FN73: rank 0 watches rank 1 (an unasked exit = the process's fatal; rank 1's stderr is
    // forwarded through rank 0, its `saber: FATAL rank 1:` line the reason)
    let mut watch: Option<saber_serve::fatal::RankWatch> = None;
    let ex: Arc<dyn IpcExchange> = if rank == 0 {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let exe = std::env::current_exe()?;
        let mut cmd = std::process::Command::new(exe);
        cmd.args(std::env::args().skip(1))
            .arg("--fn-rank")
            .arg("1")
            .arg("--fn-peer-port")
            .arg(port.to_string());
        cmd.stderr(std::process::Stdio::piped());
        let mut c = cmd.spawn().context("spawning rank 1")?;
        watch = Some(saber_serve::fatal::RankWatch::start(
            c.id(),
            1,
            c.stderr.take(),
        ));
        child.0 = Some(c);
        Arc::new(TcpExchange::accept(&listener, Duration::from_secs(300))?)
    } else {
        Arc::new(TcpExchange::connect(
            f.fn_peer_port,
            Duration::from_secs(300),
        )?)
    };
    let peer = RankPeer::new(ex);
    let hip = crate::open_hip(hip_lib)?;
    let r = run_rank(
        &hip, rank, peer, boot, pack, ckpt, cfg, prebuilt, prompts, out, f, t_boot,
    );
    drop(hip);
    if let Err(e) = &r {
        println!("fnworker rank{rank}: FAILED: {e:#}");
        // FN73: rank 1's failure is the pair's fatal (rank 0's serve loop sets its own)
        if rank > 0 {
            saber_serve::fatal::set(rank, format!("{e:#}"));
        }
        if let Some(w) = &watch {
            w.disarm();
        }
        if let Some(c) = child.0.as_mut() {
            let _ = c.kill();
        }
    } else if let Some(w) = &watch {
        // FN73b: the run finished, so rank 0 has sent rank 1 its `Stop`: disarm BEFORE the reap
        // (`Child`'s drop takes the zombie; an armed watch would then see the pid gone and call
        // the exit rank 0 asked for a rank failure), then check the status it exited with -- 0 is
        // the clean stop, a non-zero code or a signal is still rank 1's failure
        w.disarm();
        if let Some(st) = child.reap_within(Duration::from_secs(20)) {
            w.reaped(1, st);
        }
    }
    drop(child);
    r
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_rank(
    hip: &HipThread,
    rank: u32,
    peer: RankPeer,
    boot: Boot,
    pack: &Path,
    ckpt: &Path,
    cfg: &Qwen4ExpConfig,
    prebuilt: &Path,
    prompts: &[Prompt],
    out: &Path,
    f: &FnFlags,
    t_boot: Instant,
) -> anyhow::Result<()> {
    let Boot {
        decls,
        rules,
        dr,
        narrow,
        budgets,
        host,
        laws,
        vision_patches: _,
        l2_capped: _,
        pool_excl,
        xg,
        glue: glue_census,
    } = boot;
    let ri = rank as usize;
    // FN107: the placement this boot RUNS is the plan's (a plain boot may stand down to vram)
    let embed_ram = host.embed_per_rank > 0;
    hip.set_device(rank as i32)?;
    // FN106d: the page form of every pinned allocation, before the first one
    let pinned_form = f.pinned_pages()?;
    hip.set_pinned_pages(pinned_form);
    // FN57: the host-wait arm's device schedule, before the first stream or allocation (`block`:
    // every runtime wait sleeps on the signal's interrupt; `spin` / `aimed` keep the default)
    if f.host_wait == saber_exec::HostWait::Block {
        hip.set_device_schedule(saber_hip::DeviceSchedule::BlockingSync)?;
    }
    peer.agree("fn13.argv", {
        // every plan input the pair shares (the argv minus the rank's own two flags)
        let mut d = saber_exec::ProtocolDigest::new();
        let mut skip = false;
        for a in std::env::args().skip(1) {
            if std::mem::take(&mut skip) {
                continue;
            }
            if a == "--fn-rank" || a == "--fn-peer-port" {
                skip = true;
                continue;
            }
            d.text(&a);
        }
        d.finish()
    })?;
    // FN117b (`--fn-hand`): the slices' state; S6's shared pick line is made here (rank 0 creates
    // the file, rank 1 maps it by the name rank 0 sends, both meet, rank 0 removes the name)
    let hand = hand_boot(f, rank, &peer)?;
    let prof_path = f.profile_path(pack);
    let prof = if f.xtier.no_xtier_profile {
        None
    } else {
        Some(
            RouteProfile::from_json(&std::fs::read_to_string(&prof_path).with_context(|| {
                format!(
                    "the warm-start profile {} (--fn-profile user needs `saber flashnext-profile rebuild` first)",
                    prof_path.display()
                )
            })?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", prof_path.display()))?,
        )
    };
    // FN65: which warm start this boot runs (`--fn-profile pack|user|PATH`)
    if let Some(p) = &prof {
        println!(
            "fnprofile rank{rank}: warm start {} ({} tokens; {})",
            prof_path.display(),
            p.tokens,
            p.source
        );
    }
    let recs = PackRecs::open(pack, ckpt)?;
    // FN121 (`--fn-ple-prefetch embed`): the host embedding gather from a RAM copy of the table
    if f.ple_form()?.embed {
        let t = Instant::now();
        let n = recs.embed_to_ram(64)?;
        println!(
            "fnple121 rank{rank}: embedding table in host RAM -- {n} B (both halves, unpinned; 68 rows checked == the file's) in {:.0} ms; every host embedding gather reads it",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let lc = LoadCounters::new();
    let host_ar = f.fn_ar == "host";
    let plugins: Vec<_> = PLUGINS
        .into_iter()
        .chain(host_ar.then_some("flashnext_ar_host"))
        .collect();
    let ps = PluginSet::load_named(hip, prebuilt, &plugins, &lc)?;
    let table = ps.functions(hip)?;
    // FN51: the table covers the pools' context (byte-identical rows below 65,536)
    let cs = rope_table(super::run::rope_pos(super::run::pools_ctx(f, prompts)));
    let mut dev = RankDev::new(hip, rank, dr.sizes, &cs)?;
    // FN72 (`--fn-sdma-lanes split-both`): the step stream on a hardware queue of its own (its
    // boot work drained first; the same packets, only the queue changes)
    if f.fn_sdma_lanes == "split-both" {
        hip.sync_stream(&dev.stream)?;
        dev.stream = hip.create_stream_own_queue(sdma_lanes_cus(hip)?)?;
    }
    dev.host_wait = f.host_wait;
    // FN57: the device pick's region (the ledger class `fn.pick`; a stub under the host pick)
    let dpick_out = hip.malloc(pick_class_bytes(f.device_pick()))?;
    // FN112 (`--fn-overlap on`): the queued reads' pinned staging (inside the serve plane's pinned
    // reserve) + the event behind them
    let ov112 = if f.overlap()? {
        let lay = super::overlap::Layout::of(
            (MAX_SEQS * super::exec::WMAX) as usize,
            SEAM_K * 8,
            LAYERS as usize,
            40,
            VOCAB as usize / 2,
        );
        let stg = hip.host_malloc(lay.total as u64)?;
        let ev = hip.create_event(EventFlags::Default)?;
        println!(
            "fnoverlap112 rank{rank}: arm on -- the step's seam pick, sampled rows and route log queued behind its graph (staging {:.1} MiB pinned, one wait)",
            lay.total as f64 / 1048576.0
        );
        Some(super::overlap::Ov112::new(lay, stg, ev))
    } else {
        None
    };
    let pick_cpus = if f.pin_threads() {
        let (hc, pc) = pin_law(rank)?;
        let tids = saber_hip::affinity::threads_named("saber-hip");
        for &t in &tids {
            saber_hip::affinity::pin_thread(t, &hc)?;
        }
        println!(
            "fnpin rank{rank}: HIP threads {tids:?} (ours {}) on cpus {hc:?}; pick threads on {pc:?}; \
             the main thread unpinned",
            hip.tid()
        );
        Some(pc)
    } else {
        None
    };
    println!(
        "fnhostwait rank{rank}: {} (device schedule {:?}); pick {}; pin {}",
        f.host_wait,
        hip.device_schedule()?,
        if f.device_pick() {
            "device (fn_pick_top64_bf16)"
        } else {
            "host"
        },
        if pick_cpus.is_some() { "auto" } else { "off" }
    );
    let arules = if f.fn_ar_exact {
        ArRules::prod()?.exact_all()
    } else {
        ArRules::prod()?
    };
    let ar_mb = if host_ar {
        super::host_ar::slot_bytes(f.max_chunk())
    } else {
        super::dev::ar_mb(f.max_chunk())
    };
    let host_regions;
    let (mine, theirs) = if host_ar {
        host_regions = super::host_ar::HostAr::open(hip, &dev.stream, &peer, ar_mb)?;
        println!("fnar host rank{rank}: {ar_mb} bytes per parity slot, {} blocks, system-scope registered host transport", super::host_ar::BLOCKS);
        host_regions.states()
    } else {
        let mine = dev.ar_alloc(&arules, ar_mb)?;
        let a = dev.ar.as_ref().context("ar")?;
        let hs = hip.ipc_export(&a.0)?;
        let hf = hip.ipc_export(&a.1)?;
        let ps = hip.ipc_import(peer.swap_handle("fn.ar.scratch", hs)?)?;
        let pf = hip.ipc_import(peer.swap_handle("fn.ar.flags", hf)?)?;
        (mine, ArState { scratch: ps.as_u64(), flags: pf.as_u64(), seq: 0, loc: 0 })
    };
    let mut st = [ArState::default(); 2];
    st[ri] = mine;
    st[1 - ri] = theirs;
    let pair = ArPair {
        host: host_ar,
        rules: arules,
        st,
        mb: ar_mb,
        calls_exact: 0,
        calls_wht6: 0,
        decode_exact: f.fn_decode_ar == "exact",
        calls_decode_exact: 0,
        ll: super::run::ar_ll(f),
        calls_ll: 0,
        gp: super::run::ar_gp(f),
        calls_gp: 0,
        far: super::run::moe_ar_push(f),
        calls_far: 0,
        far_stood_down: 0,
        mix: super::run::mixer_ar_push(f),
        calls_mix: 0,
        mix_stood_down: 0,
    };
    // ---- weights
    let t_w = Instant::now();
    let mut owned = Owned::new();
    let src = Src::Pack(&recs, rank);
    let mut lws = Vec::new();
    for l in 0..LAYERS {
        // FN61e: `--fn-trunk w8` / `w8-head` drop the bf16 trunk records (every form reads W8)
        lws.push(model::dense_layer(
            &src,
            &mut owned,
            hip,
            &dev.stream,
            l,
            f.trunk8_prefill(),
            f.hc8(),
        )?);
    }
    let fm = model::final_mixer(&src, &mut owned, hip, &dev.stream, f.hc8())?;
    // FN108 (the which-path line): the target's HC mixing matrices at fp8 -- what was read, the
    // kernel path, the HC weight bytes a step (every mix reads its down + up once a step)
    if f.hc8() {
        let pk = recs.dir.as_path();
        let (b8, b16) = (super::hc8::shared_bytes(pk)?, super::hc8::dropped_bf16(pk)?);
        ensure!(
            lws.iter().all(|l| l.ahc.w8 && l.mhc.w8) && fm.w8,
            "--fn-trunk {}: a target connection kept bf16 mixers",
            f.fn_trunk
        );
        println!(
            "fnhc108 rank{rank}: --fn-trunk {} -- {} hc8 records read (shared/hc8, {:.3} GiB), \
             the bf16 target mixers NOT uploaded ({:.3} GiB); path flashnext_hc8 `<sym>_h8` \
             (prefill p4s -> p2s_h8), the MTP layer's mixers bf16; HC mixing bytes a step {:.1} MB \
             (bf16 {:.1} MB)",
            f.fn_trunk,
            super::hc8::RECORDS,
            b8 as f64 / (1u64 << 30) as f64,
            b16 as f64 / (1u64 << 30) as f64,
            b8 as f64 / 1e6,
            b16 as f64 / 1e6
        );
    }
    // FN61g: `w8-head` never uploads the bf16 head (every post and the drafter's rerank read the
    // W8 head); the int2 draft head's pack is then built on the HOST from the pack's bf16 record
    let (head, int2_host) = if f.trunk8_head() {
        let t = Instant::now();
        let ih = if super::run::draft_int2_on(f) {
            let p = super::trunk8::int2_pack_host(&recs.raw(rank, "lm_head.weight")?)?;
            println!(
                "fndrafthead rank{rank}: FN61g -- the int2 pack built on the HOST from the pack's \
                 bf16 head in {:.2} s (the bf16 head is not resident under --fn-trunk w8-head)",
                t.elapsed().as_secs_f64()
            );
            Some(p)
        } else {
            None
        };
        (0, ih)
    } else {
        (model::head(&src, &mut owned, hip, &dev.stream)?, None)
    };
    // FN61c: the fp8 W8 trunk (`--fn-trunk w8`; the decode forms read it, prefill keeps bf16)
    let head8 = if f.trunk8() {
        let b0 = owned.bytes;
        for (l, lw) in lws.iter_mut().enumerate() {
            lw.t8 = Some(model::trunk8_layer(
                &recs,
                rank,
                &mut owned,
                hip,
                &dev.stream,
                l as u32,
                f.trunk8_prefill(),
            )?);
        }
        // FN61d: `w8` keeps the head bf16 (the user's form); `w8-head` = FN61c's W8A8 head
        let h = if f.trunk8_head() {
            Some(model::trunk8_head(
                &recs,
                rank,
                &mut owned,
                hip,
                &dev.stream,
            )?)
        } else {
            None
        };
        println!(
            "fntrunk8 rank{rank}: --fn-trunk {} W8 trunk resident {:.3} GiB (GDN in_proj qkvz + \
             out_proj, QSA q|k|v + o_proj{}; fp8 W8A8 at decode / verify{})",
            f.fn_trunk,
            (owned.bytes - b0) as f64 / (1u64 << 30) as f64,
            if h.is_some() {
                ", the head (the bf16 head NOT resident: every post + the drafter's rerank read \
                 the W8 head, FN61g)"
            } else {
                "; the head bf16"
            },
            if f.trunk8_prefill() {
                " AND prefill; the bf16 trunk records DROPPED (FN61e), the suffix rows uploaded"
            } else {
                "; prefill + the drafter's rerank keep the resident bf16 records (FN61d)"
            }
        );
        h
    } else {
        None
    };
    let mut shared = Vec::new();
    if xg.tp2 {
        // FN80b-2: the shared expert's HALF on both ranks (sliced from the pack's whole records
        // by FN80a's law at boot; the halves join back to the whole, checked)
        let t = Instant::now();
        for l in 0..LAYERS {
            shared.push(super::xtp2::shared_expert_half(
                &recs,
                &mut owned,
                hip,
                &dev.stream,
                l,
                rank,
            )?);
        }
        println!(
            "fnshared rank{rank}: FN80b-2 -- the shared expert's half {rank} of {LAYERS} layers sliced at boot ({} B a layer, the halves join back to the whole: checked) in {:.2} s",
            super::xsplit::half_layout().1,
            t.elapsed().as_secs_f64()
        );
    } else if rank == 0 {
        for l in 0..LAYERS {
            shared.push(model::shared_expert(
                &recs,
                &mut owned,
                hip,
                &dev.stream,
                l,
            )?);
        }
    }
    // FN114c: `--fn-shared-expert split` -- this rank's HALF of every layer's shared expert (FN80a's
    // slicer, the join checked) + the half tables the fused back half reads; rank 0 keeps the
    // whole above for prefill and the drafter
    let shx114 = if let Some(half) = super::shx114::half_of(f, rank) {
        let t = Instant::now();
        let b0 = owned.bytes;
        let a = super::shx114::load(&recs, &mut owned, hip, &dev.stream, half)?;
        println!(
            "fnshared114c rank{rank}: SPLIT -- half {rank} of the shared expert on the fused decode / \
             verify train ({LAYERS} layers, {:.3} GiB + the half tables, the halves join back to the \
             whole: checked) in {:.2} s; prefill + the drafter keep rank 0's whole",
            (owned.bytes - b0) as f64 / (1u64 << 30) as f64,
            t.elapsed().as_secs_f64()
        );
        a
    } else {
        0
    };
    println!(
        "fnweights rank{rank}: {:.3} GiB of dense records in {:.1} s, every record == its sha256",
        owned.bytes as f64 / (1u64 << 30) as f64,
        t_w.elapsed().as_secs_f64()
    );
    // FN55b: the tower (`--fn-vision`): the pack's `vision/` member on this rank, billed as
    // `fn.vision` by the plan (the expert slots are the remainder)
    // (the prompt-set harness boots it too: the text byte law ON vs OFF; images enter on the
    // serve plane only)
    let vmode = f.vision_mode(pack);
    let vps = if vmode != super::run::VisionMode::Off {
        Some(PluginSet::load_named(
            hip,
            prebuilt,
            &[saber_kernels::vision::PLUGIN],
            &lc,
        )?)
    } else {
        None
    };
    // FN55c: `lazy` keeps the plugin's table for the first image's load
    let lazy_table = match vps.as_ref() {
        Some(p) if vmode == super::run::VisionMode::Lazy => Some(p.functions(hip)?),
        _ => None,
    };
    let vis = match vps.as_ref().filter(|_| vmode == super::run::VisionMode::On) {
        Some(p) => {
            let v = super::vision::FnVision::boot(
                hip,
                p.functions(hip)?,
                pack,
                rank,
                f.vision_patches(),
                f.fn_vision_dump.as_deref(),
            )?;
            println!(
                "fnvision rank{rank}: tower RESIDENT (FN55b, class {}): {} records, weights {} B, \
                 arena {} B for {} patches, upload {:.2} s",
                super::vision::CLASS,
                v.tier.records,
                v.tier.weight_bytes,
                v.tier.exec.arena_bytes(),
                f.vision_patches(),
                v.tier.upload_s
            );
            Some(v)
        }
        None => None,
    };
    let law = f.mtp_law()?;
    // FN100: the row-scaled buffers (FN38: they hold the boot's widest chunk) as ONE allocation,
    // each head at its buffer's base: the addresses never move (every graph bakes them)
    let blay = super::bufs::layout(
        f.max_chunk(),
        f.xtier.xtier_lend_arena_top_rows,
        rk_mtp_on(f),
        H,
        W,
        u64::from(LAYERS),
    );
    // `--fn-buffers-layout separate` = the control: one allocation a buffer (the pre-FN100 boot)
    let bsep = f.fn_buffers_layout == "separate";
    let ball = std::rc::Rc::new(hip.malloc(if bsep { 256 } else { blay.total })?);
    let bsub = |n: &str| -> anyhow::Result<super::bufs::Sub> {
        let p = blay
            .part(n)
            .with_context(|| format!("FN100: no {n} in the layout"))?;
        if bsep {
            let own = std::rc::Rc::new(hip.malloc(p.bytes())?);
            let at0 = super::bufs::Part { off: 0, ..*p };
            super::bufs::Sub::new(&own, &at0)
        } else {
            super::bufs::Sub::new(&ball, p)
        }
    };
    let mtp = if f.fn_mtp.on() || f.fn_mtp_ref.is_some() {
        ensure!(
            f.fn_serve.is_some() || f.scripted() || (f.fn_sched == "groups" && f.fn_turns == 1),
            "--fn-mtp runs the prompt-set harness (groups, one turn) or the serve plane (FN16)"
        );
        ensure!(
            (f.fn_serve.is_none() && !f.scripted()) || mtp_verify(f),
            "--fn-mtp in the serve plane is the verify step (not the shadow / reference arms)"
        );
        let t_m = Instant::now();
        let b0 = owned.bytes;
        let w = model::mtp_weights(
            &recs,
            rank,
            &mut owned,
            hip,
            &dev.stream,
            f.fn_control.as_deref() == Some("mtp-swap"),
            f.mtp_x()?,
        )?;
        // FN110: the which-path line of the MTP expert form (the arm's counter)
        println!(
            "fnmtpx110 rank{rank}: form {} (train {}, mix {}) split {}",
            f.mtp_experts(),
            match f.mtp_experts() {
                "slab4" if w.lw.mtp_xtp2 => {
                    "trunk W4A8 seven-launch over the MXFP4 HALF slab at the tp2 dims (FN138)"
                }
                "slab4" => "trunk W4A8 seven-launch over the MXFP4 slab",
                "fp8" => "FN50 mixed W8A8 + bf16",
                _ => "FN6 bf16",
            },
            if w.lw.mtp_x4 { "q8" } else { "bf16" },
            if w.lw.mtp_xtp2 { "tp2" } else { "ep" }
        );
        println!(
            "fnmtp rank{rank}: MTP drafter resident: {:.3} GiB ({} {} routed experts{} + dense + both fc halves) in {:.1} s, every record == its sha256; chain k {}",
            (owned.bytes - b0) as f64 / (1u64 << 30) as f64,
            if w.lw.mtp_xtp2 { 512 } else { 256 },
            if w.lw.mtp_xtp2 {
                "slab4 HALF"
            } else {
                f.mtp_experts()
            },
            if w.lw.mtp_xtp2 {
                " + the shared expert's half"
            } else if rank == 0 {
                " + the bf16 shared expert"
            } else {
                ""
            },
            t_m.elapsed().as_secs_f64(),
            f.mtp_k().max(1)
        );
        if let Some(l) = &law {
            println!("fnmtp rank{rank}: the width law: {l}");
        }
        Some(MtpRank {
            w,
            k: f.mtp_k().max(1),
            hid: bsub("hid")?,
            // FN16: + the spare slot's row (the serve plane's compaction moves a row with its slot)
            prev: hip.malloc(u64::from(MAX_SEQS + 1) * W * 2)?,
            stash: hip.malloc(W * 2)?,
            first: vec![None; MAX_SEQS as usize + 1],
            pass_ms: Vec::new(),
            launches: BTreeMap::new(),
            capture: !f.no_fn_capture && !f.no_fn_mtp_capture && f.fn_mtp_ref.is_none(),
            graphs: BTreeMap::new(),
            argmax: !f.no_fn_mtp_argmax,
            chain: None,
            seg: BTreeMap::new(),
            evs: Vec::new(),
            chain_graphs: BTreeMap::new(),
            lend: f.fn_control.as_deref() != Some("mtp-wide-capture"),
        })
    } else {
        None
    };
    // ---- the tier
    // FN80b-2: the tier file of the expert form (`--fn-experts tp2`: the split member)
    let xt = XtierFile::open(&pack.join(xg.file(rank)), OpenMode::Direct)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let h = xt.header().clone();
    // FN150: the format check -- the tier's tag must be the boot's `--fn-experts-fmt`
    h.require_format(xg.fmt).map_err(|e| anyhow::anyhow!(e))?;
    ensure!(
        h.stride == xg.stride && h.records() == xg.records,
        "rank {rank}: the tier file's geometry ({} B x {}) != the plan's ({} B x {})",
        h.stride,
        h.records(),
        xg.stride,
        xg.records
    );
    if rank == 0 {
        println!("{}", xg.line());
    }
    let mut tp = laws[ri].tier.clone();
    // FN53b: the valve band -- the top `--fn-mem-valve-max-tail` L2 slots as 256-record
    // segments of their own (the same pinned bytes at boot), the bounce past them
    let valve = f.fn_mem_valve == "on";
    ensure!(
        valve || !f.xtier.nvme_p2p(),
        "--xtier-nvme-p2p on puts the valve's bounce in the P2PDMA pool: needs --fn-mem-valve on"
    );
    if valve {
        let seg = saber_mem::expert_tier::VALVE_SEG;
        let cap = (f.fn_mem_valve_max_tail / seg * seg).min(tp.ram_slots / seg * seg);
        ensure!(
            cap > 0 && tp.defer_spares == 0 && tp.refill_spares == 0 && tp.spare_ring == 0,
            "--fn-mem-valve on: needs a band of >= 256 slots and FN8's law (no --xtier-defer, \
             --xtier-demote none or --xtier-mover two-queue)"
        );
        tp.ram_slots -= cap;
        tp.band_slots = cap;
        tp.bounce_slots = super::run::FN_MEM_VALVE_BOUNCE;
    }
    let q = HipCopyQueue::new(
        hip,
        h.stride,
        tp.vram_slots,
        saber_mem::expert_tier::ExpertTier::slab_slots(&tp) - 1,
        h.layers,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut q = q;
    // FN72 (`--fn-sdma-lanes split|split-both`): the ordered copy stream on its own hardware queue
    if f.fn_sdma_lanes != "shared" {
        let cus = sdma_lanes_cus(hip)?;
        q.own_queue(cus).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "fnsdma rank{ri}: {} -- the mover's copy stream on its own hardware queue (CU mask {} CUs){}",
            f.fn_sdma_lanes,
            cus,
            if f.fn_sdma_lanes == "split-both" {
                "; the step stream on its own queue too"
            } else {
                ""
            }
        );
    }
    // FN54b: the two-queue mover (promotions urgent, demotions background; the slot law's events)
    if f.xtier.two_queue() {
        q.enable_two_queue().map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "xtiermover rank{ri}: two-queue (urgent H2D + marks, background D2H, batch marks group {}; {} spare-ring slots)",
            saber_mem::device_mover::DEFAULT_LANE_GROUP,
            tp.spare_ring
        );
    }
    // FN63b: the HSA SDMA copy lane (H2D / D2H on two engines, the signal law)
    if f.xtier.hsa_lane() {
        let desc = q
            .enable_hsa_lane(
                crate::fn63_hsa::hsa_bdfid(hip)?,
                f.xtier.hsa_engines()?,
                h.layers,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "xtiermover rank{ri}: hsa-lane {desc}; {} spare-ring slots",
            tp.spare_ring
        );
    }
    // FN72b: the landed-mark law (`--xtier-mover-marks`)
    let marks = f.xtier.mover_marks()?;
    if marks != saber_mem::device_mover::MarkLaw::Layer {
        q.set_marks(marks, h.layers)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "xtiermarks rank{ri}: {marks:?} (--xtier-mover-marks {}) -- no HIP event per layer batch on the copy stream",
            f.xtier.xtier_mover_marks
        );
    }
    if f.xtier.xtier_mover_census {
        q.set_census().map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    // FN59b: the copy batches through the HIP thread's background lane (`--xtier-plan async`)
    q.set_submit(f.xtier.plan_submit());
    let (vb, rb) = (q.vram_base(), q.ram_device_addr);
    let seg_addrs = if valve {
        let lo = ExpertTier::slab_slots(&tp);
        q.valve_arm(
            lo,
            tp.band_slots / saber_mem::expert_tier::VALVE_SEG,
            lo + tp.band_slots,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        Vec::new()
    };
    let mut tier = ExpertTier::new(&h, tp.clone(), prof.as_ref(), vb, rb)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for (k, &a) in seg_addrs.iter().enumerate() {
        tier.set_band_addr(k as u32, a);
    }
    if valve {
        println!(
            "fnvalve rank{rank}: ARMED -- band {} slots in {} segments of {} past the slab's {} \
             (L2 {} + band), bounce {} slots on the first close",
            tp.band_slots,
            seg_addrs.len(),
            saber_mem::expert_tier::VALVE_SEG,
            ExpertTier::slab_slots(&tp),
            tp.ram_slots,
            tp.bounce_slots
        );
    }
    // FN59b: the cheap victim selection (placement-identical)
    tier.set_fast_plan(f.xtier.plan_fast());
    if rank == 0 {
        println!(
            "xtierplan: {} (fast victim selection{})",
            f.xtier.xtier_plan,
            if f.xtier.plan_submit() {
                " + submitted copy batches"
            } else {
                ""
            }
        );
    }
    // FN42b: THE ARENA BORROW -- the wide arena's scratch past the narrow rung is the tier's
    // lendable slots (INCLUSIVE, FN42's law): lent at boot (decode keeps them), reclaimed before a
    // wide chunk, lent again after it
    // a run whose widest chunk fits the narrow rung (a harness prompt set <= 2,048 tokens) has
    // nothing to lend: the borrow is inert
    // FN44b: the two-layout borrow lends every byte past the narrow ARENA (the default);
    // `--fn-borrow-cut` = FN42b's range (the wide scratch past the narrow scratch)
    let range = |nw: &ArenaSizes| {
        if f.fn_borrow_cut {
            dr.sizes.borrow_tail(nw, BORROW_ALIGN)
        } else {
            dr.sizes.borrow_arena(nw, BORROW_ALIGN)
        }
    };
    let narrow = narrow.filter(|nw| {
        let n = range(nw).1 / h.stride;
        if n == 0 {
            println!(
                "fnborrow rank{rank}: INERT -- this run's widest chunk fits the narrow rung \
                 (arena {:.1} MiB): nothing to lend",
                dr.sizes.total() as f64 / 1048576.0
            );
        }
        n > 0
    });
    let borrow = match narrow {
        Some(nw) => {
            let (off, bytes) = range(&nw);
            let n = bytes / h.stride;
            let base = dev.base() + off;
            let addrs: Vec<u64> = (0..n).map(|i| base + i * h.stride).collect();
            let first = tier
                .register_lendable(&addrs)
                .map_err(|e| anyhow::anyhow!("--fn-arena-borrow: {e}"))?;
            ensure!(
                first == tp.vram_slots,
                "lendable ids start past the static slots"
            );
            q.set_lend(std::rc::Rc::clone(&dev.arena), off, first);
            let ids: Vec<u32> = (first..first + n as u32).collect();
            tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut ledger = saber_mem::ledger::Ledger::new();
            budgets[ri]
                .declare(&mut ledger, 0)
                .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
            let mut b = Borrow {
                ids,
                lent: true,
                narrow_cap: nw.scratch,
                wide_cap: dr.sizes.scratch,
                layouts: (!f.fn_borrow_cut).then_some((dev.base(), nw, dr.sizes)),
                ledger,
                arena_bytes: dr.sizes.total(),
                lend_bytes: n * h.stride,
                reclaims: 0,
                lends: 0,
                dropped: 0,
                reclaim_ms: 0.0,
                lend_ms: 0.0,
                verify: f.fn_verify_slots,
            };
            b.relend(&budgets[ri])?;
            println!(
                "fnborrow rank{rank}: THE ARENA BORROW ({}) -- wide arena {:.1} MiB (narrow scratch {:.1} of \
                 {:.1} MiB; narrow arena {:.1} MiB; {}); tail {:.1} MiB at +{:.1} MiB = {n} \
                 lendable slots (ids {first}..{}), LENT at boot: the tier holds {} + {n} VRAM slots \
                 while no wide chunk runs; ledger fn.arena.prefill {:.1} + xtier.kv_lent {:.1} MiB",
                if f.fn_borrow_cut { "cut layout, FN42b" } else { "two layouts, FN44b" },
                dr.sizes.total() as f64 / 1048576.0,
                nw.scratch as f64 / 1048576.0,
                dr.sizes.scratch as f64 / 1048576.0,
                nw.total() as f64 / 1048576.0,
                if f.fn_borrow_cut {
                    format!(
                        "zero/ones/carry {:.1} MiB permanent",
                        (dr.sizes.total() - dr.sizes.scratch) as f64 / 1048576.0
                    )
                } else {
                    "narrow passes run in the narrow arena, wide ones in the wide arena".into()
                },
                bytes as f64 / 1048576.0,
                off as f64 / 1048576.0,
                first + n as u32,
                tp.vram_slots,
                (dr.sizes.total() - n * h.stride) as f64 / 1048576.0,
                (n * h.stride) as f64 / 1048576.0,
            );
            Some(b)
        }
        None => None,
    };
    // FN76b: THE ARENA TOP -- the narrow carve's scratch past the boundary of every pass of at
    // most `--xtier-lend-arena-top-rows` rows is lendable, LENT at boot (decode keeps it)
    let top = if f.xtier.lend_arena_top().0 {
        top_boot(
            rank,
            f,
            prompts,
            &decls,
            &rules,
            &dr,
            borrow.as_ref(),
            &dev,
            &mut tier,
            &mut q,
            &budgets[ri],
            h.stride,
        )?
    } else {
        None
    };
    // FN43: THE PREFILL STAGING RING -- its memory (ledger line `xtier.stage_ring`, out of the
    // slots) is the tier's LENDABLE slots, lent while no staged pass runs (decode keeps them)
    let ring = match laws[ri].stage {
        Some(law) => Some(ring_boot(
            hip,
            rank,
            law,
            &mut tier,
            &mut q,
            &budgets[ri],
            h.stride,
            f.xtier.fn_ring_idle,
            f.xtier.stage_key(),
        )?),
        None => None,
    };
    // FN55c: THE LAZY TOWER'S REGION -- `fn.vision`'s bytes (the weight upload's arena + the
    // tower's arena) are the tier's lendable slots, LENT at boot; the first image reclaims them
    let lazy = match lazy_table {
        Some(table) => Some(lazy_boot(
            hip,
            rank,
            pack,
            f,
            table,
            &mut tier,
            &mut q,
            &budgets[ri],
            h.stride,
        )?),
        None => None,
    };
    let xptr = hip.malloc(xg.xptr_bytes())?;
    let xhost = hip.host_malloc(xg.xptr_bytes())?;
    let alias = super::run::fn30b_alias(
        f.fn_diag_alias_ram,
        &q,
        h.stride,
        saber_mem::expert_tier::ExpertTier::slab_slots(&tp) - 1,
    );
    let miss_ram = super::run::miss_ram_range(
        q.ram_device_addr,
        saber_mem::expert_tier::ExpertTier::slab_slots(&tp),
        h.stride,
    );
    let mut tier = TierRank {
        tier,
        mover: DeviceMover::new(&xt, q, 32).with_defer(tp.defer_spares > 0),
        xptr,
        xhost,
        shared,
        xn: xg.xn(),
        alias,
    };
    println!("{}", budgets[ri].boot_line());
    println!(
        "{}",
        host.boot_line(
            u64::from(laws[ri].tier.ram_slots),
            // FN107: the exclusive pool slots count as static for the complement
            (xg.records - u64::from(laws[ri].tier.vram_slots)).saturating_sub(pool_excl),
            xg.stride
        )
    );
    println!("{}", tier.tier.boot_line());
    println!(
        "fnmoverhold rank{rank}: {} (FN30c: a verify step's tier copies issued after the drafter round)",
        if f.mover_hold() { "on" } else { "off" }
    );
    println!("{}", laws[ri].boot_line(rank));
    // FN51e: the warm stage (64 slots, 160 MiB pinned) lives for the warm only -- the end-of-run
    // `--fn-verify-slots` check allocates its own ([`verify_stage`])
    // FN150: 64 records of THIS tier's stride (an NVFP4 record is 676 pages, not 639)
    let mut stage = hip.host_malloc(
        64 * xg
            .stride
            .max(saber_mem::flashnext_budget::geometry::EXPERT_STRIDE),
    )?;
    // FN71c: the boot fill through the card's P2PDMA pool (`--xtier-boot-fill`), and the pool
    // reader the valve's bounce keeps (`--xtier-nvme-p2p on`)
    let mut nv = super::run::boot_pool_reader(hip, &xt, &f.xtier, rank as usize)?;
    let fill = match nv.as_mut() {
        Some(nv) if f.xtier.boot_fill_pool().0 => super::run::WarmFill::Pool {
            nv,
            check: if f.fn_verify_slots {
                u32::MAX
            } else {
                super::run::POOL_FILL_CHECK
            },
        },
        _ => super::run::WarmFill::Host,
    };
    let (v, ram, bad) = warm(hip, &xt, &mut tier, &mut stage, fill, rank as usize)?;
    drop(stage);
    if f.xtier.nvme_p2p() {
        let nv = nv
            .take()
            .ok_or_else(|| anyhow::anyhow!("--xtier-nvme-p2p on: no pool reader"))?;
        let (_, bn) = tier.tier.bounce();
        let (gpu, bytes) = tier
            .mover
            .queue_mut()
            .set_pool_bounce(nv, bn)
            .map_err(|e| anyhow::anyhow!("rank {rank}: {e}"))?;
        println!(
            "fnp2p rank{rank}: ARMED -- the valve's bounce ({bn} slots) lives in the P2PDMA pool ({} MiB at {gpu:#x}): a prefill pass's Fetch tail reads drive -> VRAM, read in place by the step (no host DRAM, no UVA)",
            bytes >> 20
        );
    }
    drop(nv);
    println!(
        "fnwarm rank{rank}: {v} VRAM + {ram} RAM records, {bad} differ from the xtier sha256 table"
    );
    ensure!(bad == 0, "rank {rank}: {bad} expert records differ");
    // FN76: THE POOL AS SLOTS -- the fill's reader is gone; the whole pool (less the reserve)
    // becomes lendable expert slots through the BO imported into HIP, lent now, never reclaimed
    let pool_slots = match f.xtier.pool_slots() {
        (true, must) if f.xtier.nvme_p2p() => {
            ensure!(
                !must,
                "--xtier-pool-slots on: the pool is the valve's bounce under --xtier-nvme-p2p on"
            );
            println!(
                "fnpoolslots rank{rank}: off -- the pool is the valve's bounce (--xtier-nvme-p2p on)"
            );
            None
        }
        (true, _) => pool_slots_boot(
            hip,
            rank,
            &f.xtier,
            &mut tier,
            &dev.stream,
            h.stride,
            pool_excl,
        )?,
        _ => None,
    };
    let pool_n = pool_slots.as_ref().map_or(0, |p| p.ids.len() as u32);
    // FN107: an L2 planned without the exclusive slots' records needs exactly those slots; under
    // `on` a pool that gives fewer is refused, under `auto` the boot runs on with the shortfall's
    // records on the drive (demand reads) and says so
    if pool_excl > 0 && u64::from(pool_n) != pool_excl {
        let why = format!(
            "rank {rank}: --xtier-pool-exclusive: the plan counted {pool_excl} exclusive pool slots \
             for the RAM L2's complement but the pool gave {pool_n}: {} records a rank stay on the \
             drive (demand reads); --xtier-pool-exclusive off pins them",
            tier.tier.occupancy().2
        );
        ensure!(!f.xtier.pool_exclusive().1, "{why}");
        println!("fnpoolx107 rank{rank}: SHORT -- {why}");
    }
    // ---- PLE
    let pre = format!("{P}layers.1.ple.ple_embedding.");
    let ck = Ckpt::open(ckpt)?;
    let ph = PleHash::from_buffers(
        &cfg.ple,
        cfg.eos_token_id,
        &ck.raw(&format!("{pre}layer_multipliers"))?,
        &ck.raw(&format!("{pre}ngram_heads_vocab_sizes"))?,
        &ck.raw(&format!("{pre}ngram_heads_offsets"))?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let idx = PleIndex::from_json(
        &std::fs::read_to_string(pack.join("ple/ple_index.json"))?,
        PleSourceKind::CkptBf16,
        PleGate::Default,
        None,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ple = PleTier::new(
        idx,
        OpenMode::Direct,
        f.xtier.ple_qd_prefill,
        ((f.xtier.ple_cache_mib << 20) / 320) as u32,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    // FN121: the per-batch read log under `--fn-ple-trace`
    ple.set_read_log(f.fn_ple_trace.is_some());
    if rank == 0 {
        println!("{}", ple.boot_line());
    }
    // ---- pools + buffers
    // the pools' context (`run::pools_ctx`, the ledger's KV class too): every request + its decode
    // (+ a second turn), at least the 2,048 band; `--fn-max-ctx` under `--fn-serve`
    let max_ctx = super::run::pools_ctx(f, prompts);
    ensure!(
        max_ctx <= ROPE_POS_MAX,
        "the pools' context {max_ctx} past the checkpoint's {ROPE_POS_MAX} positions"
    );
    let pool_w = pools_geometry(f, prompts)?.1;
    // FN80b-2: the TP-split expert form routes all 512 experts as local (the MTP's stay EP)
    let dm = if xg.tp2 {
        Dims::new(2, rank).experts_tp2()
    } else {
        Dims::new(2, rank)
    };
    // FN150: the NVFP4 records' cells (the boot refuses every piece without an NVFP4 body)
    let dm = if xg.fmt == saber_mem::xtier::XtierFormat::Nvfp4E4m3G16 {
        dm.experts_nvfp4()
    } else {
        dm
    };
    let kvg = f.kv_geom_for(max_ctx, super::run::harness_tokens(f, prompts));
    let mut pools = Pools::new(
        hip,
        &dm,
        max_ctx,
        f.fn_gdn_state_f32,
        pool_w,
        u64::from(3 + pool_w - 1),
        kvg.is_some(),
        kvg.map_or(0, |g| g.ix_pages()),
        f.kv_dtype(),
        f.kv_mtp_dtype(),
        f.pools_layout(),
    )?;
    // FN18: the exact ledger's `fn.pools` class is this allocation, byte for byte
    ensure!(
        pools.bytes()
            == Pools::plan_bytes(
                &dm,
                max_ctx,
                f.fn_gdn_state_f32,
                pool_w,
                u64::from(3 + pool_w - 1),
                kvg.is_some(),
                kvg.map_or(0, |g| g.ix_pages()),
                f.kv_dtype(),
                f.kv_mtp_dtype(),
                f.pools_layout()
            ),
        "the pools' allocation differs from the ledger's plan"
    );
    if let Some((kc, kg)) = pools.idx_law() {
        println!(
            "fnpools rank{rank}: SEQUENCE-MAJOR (FN96) -- {} regions of {:.1} MiB (rows {:.1} MiB) \
             + {} B lead = {:.1} MiB; the tables' index: conv 1 + s x {kc}, GDN 1 + s x {kg} + t",
            MAX_SEQS + 1,
            pools.sg.as_ref().map_or(0.0, |(g, _)| g.s as f64 / 1048576.0),
            pools.sg.as_ref().map_or(0.0, |(g, _)| g.content as f64 / 1048576.0),
            pools.sg.as_ref().map_or(0, |(g, _)| g.lead),
            pools.bytes() as f64 / 1048576.0
        );
    }
    if let Some(g) = kvg {
        ensure!(dm.nkv == 1, "the tiered KV needs one KV head a rank (TP=2)");
        let ssd = f.kv_ssd_path(rank);
        let k = super::kvwire::KvDev::new(hip, &dev.stream, g, ssd.as_deref())?;
        println!("{}", k.boot_line(rank));
        pools.attach_kv(k)?;
    }
    // FN51f-b: ELASTIC VRAM -- the hot pool's top as the tier's lendable slots, a region a
    // segment on the copy queue, lent at boot down to the band, filled hottest-first now
    let elastic = match (f.vram_elastic(), pools.kvt.as_ref()) {
        (true, Some(kc)) => Some(elastic_boot(
            hip,
            rank,
            f,
            &dev.stream,
            kc,
            &mut tier,
            &budgets[ri],
            // FN76: the pool's slots are static VRAM slots for the floor (never reclaimed)
            tp.vram_slots + pool_n,
            h.stride,
        )?),
        (true, None) => bail!("--fn-vram-elastic on needs the tiered KV"),
        _ => None,
    };
    // FN96: the idle sequence slots' state regions as the tier's lendable strides
    let seql = if f.xtier.lend_seq_pools().0 {
        seq_lend_boot(rank, f, &pools, &mut tier, &budgets[ri], h.stride)?
    } else {
        None
    };
    // FN22: the drafter's device-side chain (the default under MTP-k through the verify step; the
    // shadow / reference arms keep the host chain): the rank's embedding half resident, the
    // chain's input and id buffers
    let mut mtp = mtp;
    // LR-N: coupled drafting lives in the device chain's tail; any other drafter refuses it
    ensure!(
        !f.spec_couple()? || (mtp.is_some() && mtp_verify(f) && !f.no_fn_mtp_device_chain),
        "--spec-couple on needs the MTP drafter's device chain (MTP-k through the verify step, no \
         --no-fn-mtp-device-chain)"
    );
    if let Some(mt) = mtp.as_mut() {
        if mtp_verify(f) && !f.no_fn_mtp_device_chain {
            let t_e = Instant::now();
            let (embh, nan, sub) = model::embed_half(&recs, rank, hip, &dev.stream, embed_ram)?;
            // the exchange law's premise: x + (-0.0) == x holds for every non-NaN bf16; a
            // subnormal would also depend on the reduce's denormal mode -- none in this checkpoint
            ensure!(
                nan == 0 && sub == 0,
                "the embedding half holds {nan} NaN/Inf and {sub} subnormal patterns: the device \
                 chain's exchange law is not proven for them (--no-fn-mtp-device-chain)"
            );
            let (tb, _) = ChunkConsts::host_verify_at(
                &[0; MAX_SEQS as usize],
                1,
                &[1; MAX_SEQS as usize],
                &[0; MAX_SEQS as usize],
                pools.nbs,
                1,
                PadLaw {
                    live: 1,
                    pad: Some(SPARE),
                    idx: None,
                },
            );
            // FN45b2: under the tier every table carries its per-layer append slots too
            let kvx = if pools.kvt.is_some() {
                u64::from(saber_mem::kv_tier::QSA_LAYERS) * u64::from(MAX_SEQS) * 8 + 256
            } else {
                0
            };
            let slot = (tb.len() as u64 + kvx).div_ceil(256) * 256 + 256;
            let inb = hip.malloc(IN_STAGE + u64::from(saber_kernels::fn22::MAX_PASSES) * slot)?;
            let ids = hip.malloc(
                u64::from(saber_kernels::fn22::MAX_PASSES * saber_kernels::fn22::IDS_STRIDE) * 4,
            )?;
            hip.memset(&inb, 0, &dev.stream)?;
            hip.memset(&ids, 0, &dev.stream)?;
            println!(
                "fnchain rank{rank}: the DEVICE-SIDE drafter chain: embedding half resident in {} ({:.3} GiB, \
                 sha256 == the record, 0 NaN/Inf, 0 subnormal) in {:.1} s; table slot {slot} B; \
                 argmax {} splits + pack + the exact reduce + select a pass{}{}{}",
                embh.placement(),
                embh.len() as f64 / (1u64 << 30) as f64,
                t_e.elapsed().as_secs_f64(),
                saber_kernels::fn22::SPLITS,
                if f.fn_control.as_deref() == Some("chain-tie-high") {
                    "; TIE-HIGH CONTROL"
                } else {
                    ""
                },
                if f.fn_mtp_chain_check { "; CHECK ARM" } else { "" },
                if f.fn_mtp_chain_graph { "; ONE GRAPH A CHAIN" } else { "" },
            );
            println!(
                "fnembed99 rank{rank}: placement {} -- {} B at device address {:#x} ({}); reader \
                 fn_draft_pack (one {} B row a drafted row, captured drafter graphs included)",
                embh.placement(),
                embh.len(),
                embh.addr(),
                if embed_ram {
                    "hipHostMalloc, read in place over UVA; billed in fnhost, its VRAM class -> slots"
                } else {
                    "hipMalloc; the VRAM ledger's embed class"
                },
                H * 2
            );
            mt.chain = Some(DevChain {
                embh,
                emb_rows: 0,
                emb_passes: 0,
                inb,
                ids,
                slot,
                tie_high: f.fn_control.as_deref() == Some("chain-tie-high"),
                check: f.fn_mtp_chain_check.then(ChainCheck::default),
                graph: f.fn_mtp_chain_graph,
                int2: None,
                i2last: None,
                conf: f
                    .fn_mtp_conf_log
                    .as_deref()
                    .map(|p| ConfLog::open(p, rank))
                    .transpose()?,
                couple: f.spec_couple()?.then(CoupleRt::default),
            });
            if f.spec_couple()? {
                // LR-N: the coupled tail is the int2 head's (family 10's fused site), served law
                ensure!(
                    super::run::draft_int2_on(f)
                        && !f.fn_mtp_chain_check
                        && f.fn_mtp_conf_log.is_none()
                        && f.fn_control.as_deref() != Some("chain-tie-high"),
                    "--spec-couple on needs the device chain with the int2 draft head \
                     (`--fn-draft-head int2`), no chain check / confidence log / tie control"
                );
                println!(
                    "fncouple rank{rank}: COUPLED DRAFTING on -- each sampled row's drafts take \
                     the target's own noise at their verify positions (flashnext_couple: \
                     fn_couple_acp + fn_couple_select in family 10's two launches); greedy rows \
                     keep FN22's argmax; the walk and every committed token unchanged"
                );
            }
            if super::run::draft_int2_on(f) {
                let k = f.fn_draft_head_k;
                ensure!(
                    (1..=saber_kernels::fn33::KMAX).contains(&k),
                    "--fn-draft-head-k {k}: 1..={}",
                    saber_kernels::fn33::KMAX
                );
                let n = VOCAB / 2;
                let ng = saber_kernels::fn33::NG as u64;
                let wq = hip.malloc(n * saber_kernels::fn33::ROW_BYTES as u64)?;
                let s = hip.malloc(n * ng * 2)?;
                let zs = hip.malloc(n * ng * 2)?;
                ensure!(
                    wq.len() + s.len() + zs.len() == super::run::draft_int2_class(f),
                    "the int2 draft head's allocation differs from its ledger class"
                );
                if let Some(dc) = mt.chain.as_mut() {
                    dc.int2 = Some(Int2Head { wq, s, zs, k });
                }
            }
        }
    }
    // the decode step's launch law + arena, declared on the host (rule 7)
    let mut dlaw = Vec::new();
    for b in [1u32, 2, 4, 8] {
        let dd = plan::dry_run_decode(&decls, &rules, b, max_ctx)?;
        ensure!(
            dd.sizes.total() <= dr.sizes.total(),
            "the b{b} decode step does not fit the prefill arena"
        );
        // FN42b: under the borrow a decode step runs with the tail LENT: its scratch fits the
        // narrow rung's (the other regions are the wide arena's)
        if let Some(nw) = narrow.as_ref() {
            ensure!(
                dd.sizes.scratch <= nw.scratch,
                "the b{b} decode step's scratch {} B past the borrow's narrow {} B",
                dd.sizes.scratch,
                nw.scratch
            );
            // FN44b: under the two-layout borrow every region runs in the narrow arena
            ensure!(
                f.fn_borrow_cut
                    || (dd.sizes.zero <= nw.zero
                        && dd.sizes.ones <= nw.ones
                        && dd.sizes.carry <= nw.carry),
                "the b{b} decode step's regions {:?} past the borrow's narrow arena {nw:?}",
                dd.sizes
            );
        }
        dlaw.push(format!(
            "b{b} {}",
            dd.per_layer.iter().sum::<u32>() + dd.post
        ));
    }
    if rank == 0 {
        println!(
            "fndecodelaw: launches per decode step (declared, host-emitted): {}",
            dlaw.join(" / ")
        );
    }
    if mtp_verify(f) && rank == 0 {
        // FN15b: gate D's premise at boot -- the ops whose cell moves between a SPEC-0 step of b
        // and a width-w verify of the same b (bytes differ by construction there)
        for b in [1u32, 2, 4, 8] {
            for w in 2..=f.mtp_k() + 1 {
                let rows = plan::verify_census(&rules, b, w, max_ctx, f.fn_verify_class)?;
                let moved: Vec<&str> = rows.iter().filter(|x| !x.same()).map(|x| x.op).collect();
                // FN24: the emitted step's cells against SPEC 0's (the byte law's premise)
                let (nc, diff) =
                    plan::verify_cells(&decls, &rules, b, w, max_ctx, f.fn_verify_class)?;
                let same = diff.is_empty();
                if f.fn_verify_class == super::exec::VerifyClass::Width1 {
                    ensure!(
                        same,
                        "b{b} w{w}: a width-1-class verify must launch SPEC 0's cells: {diff:?}"
                    );
                }
                println!(
                    "fncensus {} b{b} w{w}: {} of {} ops move class [{}]; emitted cells {} SPEC 0's ({nc})",
                    f.fn_verify_class,
                    moved.len(),
                    rows.len(),
                    moved.join("; "),
                    if same { "==" } else { "!=" }
                );
            }
        }
    }
    if rk_mtp_on(f) && rank == 0 {
        let mut v = Vec::new();
        for (m, d) in [(1u32, true), (8, true), (64, false), (512, false)] {
            let (n, _) = plan::dry_run_mtp(&decls, &rules, m, d, max_ctx)?;
            v.push(format!("{}{m} {n}", if d { "d" } else { "p" }));
        }
        println!(
            "fnmtplaw: launches per MTP pass (declared, host-emitted; 4 pre + a QSA layer + mixer 2 + head 1): {}",
            v.join(" / ")
        );
    }
    // FN38: the staging buffers and the route log hold the boot's widest chunk
    // (FN100: buffers of the one allocation above)
    let emb = bsub("emb")?;
    let rows = bsub("rows")?;
    let idlog = bsub("idlog")?;
    // FN100: the tails lent at decode (EMPTY at boot: nothing has written the buffers yet)
    if rank == 0 {
        for l in super::run::buffers_census(f, prompts, h.stride)?.0 {
            println!("{l}");
        }
    }
    let buft = if f.xtier.lend_buffers_top().0 && bsep {
        ensure!(
            !f.xtier.lend_buffers_top().1,
            "--xtier-lend-buffers-top on: refused under --fn-buffers-layout separate"
        );
        println!("fnbuflend rank{rank}: stands down -- --fn-buffers-layout separate (the control)");
        None
    } else if bsep {
        println!(
            "fnbuflend rank{rank}: OFF -- --fn-buffers-layout separate (the pre-FN100 boot: one \
             allocation a buffer, {:.1} MiB)",
            blay.separate_bytes() as f64 / 1048576.0
        );
        None
    } else if f.xtier.lend_buffers_top().0 {
        buf_boot(rank, f, &blay, &ball, &mut tier, &budgets[ri], h.stride)?
    } else {
        println!(
            "fnbuflend rank{rank}: OFF (the control) -- one allocation {:.1} MiB (the separate \
             buffers' {:.1}), heads of {} rows; {} lendable slots unused",
            blay.total as f64 / 1048576.0,
            blay.separate_bytes() as f64 / 1048576.0,
            blay.keep,
            blay.slots(h.stride)
        );
        None
    };
    // FN47b: the pre-gate capture log (2 methods x 48 layers x 64 rows x 512 bf16 = 6 MiB)
    let pg = if f.fn_pregate_capture {
        ensure!(
            f.no_fn_capture,
            "--fn-pregate-capture runs the eager decode: add --no-fn-capture"
        );
        let log = hip.malloc(super::exec::pregate_slot(2, 0))?;
        hip.memset(&log, 0, &dev.stream)?;
        println!(
            "fnpregate rank{rank}: CAPTURE ARM (diagnostic) -- log {} B",
            super::exec::pregate_slot(2, 0)
        );
        Some(PgRank {
            log,
            armed: 0,
            tallies: std::collections::BTreeMap::new(),
        })
    } else {
        None
    };
    let bar = hip.malloc(256)?;
    hip.memset(&bar, 0, &dev.stream)?;
    // FN30: the fused MoE train's sync region, zeroed ONCE (outside the arena and its poison; the
    // ledger class `fn.moe.sync`)
    let moe_sync = hip.malloc(super::run::moe_sync_class(f, rank))?;
    hip.memset(&moe_sync, 0, &dev.stream)?;
    // FN120b (`--fn-glue seeds|fold`): the run-wide seed region, seeded ONCE here (zero part 0,
    // ones part 0xff; the ledger class `fn.glue120`); every captured run restores what it used
    // FN120e: the region the boot's census sized (`run::glue_census`), not a constant
    let glue_form = super::run::glue120(f)?;
    let glue_region = glue_census.region;
    let glue = if glue_form == super::run::Glue120::Off {
        None
    } else {
        let g = hip.malloc(glue_region.total())?;
        hip.memset_at(&g, 0, glue_region.zero, 0, &dev.stream)?;
        hip.memset_at(&g, glue_region.zero, glue_region.ones, 0xff, &dev.stream)?;
        hip.sync_stream(&dev.stream)?;
        Some(g)
    };
    // FN120c (`--fn-dataflow gdn`): the dataflow tables' region, zeroed ONCE (the control words
    // must read 0 at every launch's entry; the kernel leaves them 0); the ledger class `fn.df120`
    // FN124 (`--fn-prefill-tiles ...,qsatwin`, a DIAGNOSTIC): the in-situ twin's differing-word
    // counter (the ledger class `fn.qsa124`), zeroed once, read at the run's end
    let qsa124 = if super::run::prefill122(f)? & saber_kernels::flashnext::prefill122::QSATWIN != 0
    {
        let d = hip.malloc(super::run::QSA124_TWIN_BYTES)?;
        hip.memset(&d, 0, &dev.stream)?;
        hip.sync_stream(&dev.stream)?;
        Some(d)
    } else {
        None
    };
    let df_form = super::run::dataflow120(f)?;
    let df = if df_form == super::run::Dataflow120::Off {
        None
    } else {
        let d = hip.malloc(super::exec::DF120_BYTES)?;
        hip.memset(&d, 0, &dev.stream)?;
        hip.sync_stream(&dev.stream)?;
        Some(d)
    };
    let mut em = Emitter::new(
        &decls,
        &rules,
        dm,
        if let Some(b) = borrow.as_ref() {
            // FN42b: the permanent cut's layout, the scratch capped at the narrow rung while lent
            // (the wide scratch past it is the lent range): an emission past the cap refuses
            // instead of writing a lent slot. FN44b (the default): the narrow ARENA at the base
            // while lent (the c2048 boot's layout), every byte past it lent
            b.boot_arena(dev.base(), dr.sizes)
        } else {
            Arena::carve(dev.base(), dr.sizes)
        },
    );
    if let Some(t) = top.as_ref() {
        // FN76b: lent at boot -- every boot emission (the pre-captured runs) below the boundary
        em.arena.scratch.cap = t.cap;
    }
    em.moe_train = f.fn_moe_train;
    em.shared_split = shx114;
    em.t8_head = head8;
    em.t8_prefill = f.trunk8_prefill();
    em.head_bf16_dropped = f.trunk8_head();
    ensure!(
        (head == 0) == f.trunk8_head() && (head8.is_some() == f.trunk8_head()),
        "the head's form: bf16 resident iff not --fn-trunk w8-head, the W8 head iff it"
    );
    em.route_form = super::run::route_form(f);
    em.miss_order = super::run::miss_order(f);
    em.ar_prologue = super::run::ar_prologue(f);
    em.miss_ram = miss_ram;
    em.moe_sync = moe_sync.addr().as_u64();
    em.glue = glue.as_ref().map(|g| {
        super::exec::GlueSeeds::sized(
            g.addr().as_u64(),
            glue_region,
            glue_form == super::run::Glue120::Fold,
            super::run::glue_strict(f),
        )
    });
    em.pf124_twin = qsa124.as_ref().map(|d| d.addr().as_u64());
    em.df = df.as_ref().map(|d| {
        super::exec::DfTables::new(
            d.addr().as_u64(),
            df_form == super::run::Dataflow120::GdnNoDep,
        )
    });
    if rank == 0 {
        println!(
            "fndf120: form {df_form:?} (gdn = every b1 verify GDN tail -- merged projection, fused \
             update + norm + quant, out projection -- as ONE persistent dataflow launch, {} \
             workgroups; off = the launch train, the control); region {} B a rank",
            super::exec::DF120_GRID,
            if df.is_some() { super::exec::DF120_BYTES } else { 0 }
        );
    }
    if rank == 0 {
        println!(
            "fnglue120: form {glue_form:?} (seeds = the captured runs' seed fills folded into one \
             run-wide region restored at the run's end; fold = + the route-id log as one copy a \
             run; off = per-layer fills + copies, the control); region {} B a rank{}",
            if glue.is_some() {
                glue_region.total()
            } else {
                0
            },
            if glue.is_some() {
                format!(
                    "; {}{}; a run past it {}",
                    glue_census.line(),
                    if f.fn_glue_region_mib.is_some() {
                        " -- the zero part FORCED by --fn-glue-region-mib (a gate control)"
                    } else {
                        ""
                    },
                    if super::run::glue_strict(f) {
                        "REFUSES (strict: --fn-glue fold-strict or a non-served boot)"
                    } else {
                        "FALLS BACK to the per-layer fills from that seed on (counted, logged)"
                    }
                )
            } else {
                String::new()
            }
        );
    }
    if rank == 0 {
        println!(
            "fnmoetrain: {} (fused = router + route + moe_back at decode steps of <= {} rows, 3 launches a block; \
             legacy = the seven); sync region {} B a rank ({} slots x {} B); route {:?} (FN49b: Dpp = the default)",
            f.fn_moe_train,
            saber_kernels::moe::BackTrain::MAX_TOKENS,
            super::exec::moe_sync_bytes(dm.eloc),
            saber_kernels::moe::BackTrain::MAX_TOKENS,
            super::exec::moe_sync_slot_bytes(dm.eloc),
            em.route_form
        );
    }
    dev.poison = f.fn_poison;
    dev.op_timing = f.fn_op_timing;
    em.state_f32 = f.fn_gdn_state_f32;
    em.tap_layer = f.fn_tap_layer;
    em.conv_l = u64::from(3 + pool_w - 1);
    em.qsa = f.fn_qsa;
    if let Some(k) = pools.kvt.as_ref() {
        let k = k.borrow();
        em.kv_table = k.tdev.addr().as_u64();
        em.kv_ls = k.layer_bytes();
        // FN68d: the MTP layer's own table (its twin tier)
        em.kv_mtp_table = k.mtp.as_ref().map_or(0, |t| t.tdev.addr().as_u64());
    }
    em.vclass = f.fn_verify_class;
    em.glue_v1 = f.fn_glue_control;
    em.kv8 = f.kv_dtype() == saber_mem::kv_tier::KvDtype::Fp8;
    em.kv8_mtp = f.kv_mtp_dtype() == saber_mem::kv_tier::KvDtype::Fp8;
    // FN68b: the fp8 pages' scale law -- the append's scalars per layer + the descale table
    let kv_scales = f.kv_scales(pack)?;
    let mut kv_desc = None;
    if let Some(ks) = kv_scales.as_ref() {
        println!("{}", ks.line(rank as usize));
        em.kv_scale = (0..super::kvscale::TABLE_LAYERS)
            .map(|l| ks.at(l, rank as usize))
            .collect();
        // allocated exactly when the ledger declares it (`run::kv_desc_class`)
        if super::run::kv_desc_on(f) {
            let t = ks.table(rank as usize);
            ensure!(
                t.len() as u64 == super::run::kv_desc_class(f),
                "the descale table's bytes vs its ledger class"
            );
            let d = hip.malloc(t.len() as u64)?;
            hip.memcpy_h2d_blocking(&d, &t, &dev.stream)?;
            em.kv_desc = d.addr().as_u64();
            kv_desc = Some(d);
        }
    }
    match f.fn_control.as_deref() {
        None => {}
        Some("swap-arg") => em.ctl = Ctl::SwapArg,
        Some("certified-scan") => em.ctl = Ctl::CertifiedScan,
        Some("drop-carry") => em.ctl = Ctl::DropCarry,
        // FN15: the MTP gate's control lives in the MTP weights (fc_embedding <-> fc_hidden)
        Some("mtp-swap") if rk_mtp_on(f) => {}
        // FN22: the device chain's tie law flipped (the tie gate's control; the ids must differ
        // at exact ties only)
        Some("chain-tie-high") if rk_mtp_on(f) => {}
        // FN51d: the drafter's decode passes keep the arena they find (the pre-FN51d law)
        Some("mtp-wide-capture") if rk_mtp_on(f) => {}
        // FN51f: the decode band at the boot's max context (the pre-FN51f law)
        Some("band-boot") => {}
        // FN54: the host keep-awake arm (a spinning thread in rank 0's process; no byte moves)
        Some("host-keepalive") => {}
        Some(c) => bail!("--fn-control {c}: swap-arg | certified-scan | drop-carry"),
    }
    peer.rendezvous("fn13.boot")?;
    println!(
        "fnboot rank{rank}: WORKER-PROCESS form (pid {}, one HipThread on card {rank}); reduce scratch/flags \
         exchanged as IPC handles; arena {:.1} MiB; pools {:.1} MiB for {MAX_SEQS} sequences x {max_ctx} ctx \
         ({} blocks/seq); AR {}; boot {:.1} s",
        std::process::id(),
        dr.sizes.total() as f64 / 1048576.0,
        pools.bytes() as f64 / 1048576.0,
        pools.nbs,
        if f.fn_ar_exact {
            "EXACT at every size (--fn-ar-exact)"
        } else if f.fn_decode_ar == "exact" {
            "exact < quant_min_bytes, wht6 above; decode forms <= 64 rows EXACT (FN78, --fn-decode-ar exact)"
        } else {
            "exact < quant_min_bytes, wht6 above (--fn-decode-ar band)"
        },
        t_boot.elapsed().as_secs_f64()
    );
    println!(
        "fnpinned106d rank{rank}: boot | {}",
        saber_hip::pinned::stats().line(pinned_form)
    );
    if f.fn_evict_all {
        tier.tier.evict_all();
        println!(
            "fnevict rank{rank}: evict_all -- every placement is Fetch; the demand path serves"
        );
    }
    let mut rk = Rank {
        rank,
        hip,
        dev,
        table,
        lws,
        fm,
        head,
        head8,
        tier,
        pools,
        peer,
        pair,
        emb,
        rows,
        idlog,
        buft,
        pg,
        bar,
        moe_sync,
        glue,
        df,
        qsa124,
        kv_desc,
        kv_scales,
        max_ctx,
        band_boot: f.fn_control.as_deref() == Some("band-boot"),
        copies: 0,
        last_lg: 0,
        lev: f.host_levers()?,
        tier_pend: None,
        xmiss: if f.fn_mem_valve == "on" && f.fn_valve_graphs == "on" && !f.no_fn_capture {
            Some(Xmiss::new(hip, rank)?)
        } else {
            None
        },
        tier_def_ms: 0.0,
        fn59: BTreeMap::new(),
        tier_pre: false,
        tier_pre_n: (0, 0),
        pick_t: [0.0; 3],
        gsplit: f.graph_split()?,
        ple_ov: if let Some(m) = f.ple_overlap()? {
            ensure!(
                super::plan::PLE_LAYER >= 1,
                "--fn-ple-overlap layer1: the PLE layer is layer 0"
            );
            Some(PleOv {
                stream: if m.side {
                    Some(hip.create_stream(saber_hip::StreamFlags::NonBlocking)?)
                } else {
                    None
                },
                pre_on: m.pre,
                staged: hip.host_malloc(u64::from(MAX_SEQS * super::exec::WMAX) * H * 2)?,
                ev: (
                    hip.create_event(EventFlags::Default)?,
                    hip.create_event(EventFlags::Default)?,
                    hip.create_event(EventFlags::Default)?,
                ),
                pre: (0.0, 0, 0, 0),
            })
        } else {
            None
        },
        rd_t: BTreeMap::new(),
        vis,
        ropes: vec![None; MAX_SEQS as usize + 1],
        vis_on: vmode != super::run::VisionMode::Off,
        lazy,
        mtp,
        tap_mtp: f.fn_tap_mtp,
        pad_ctl: f.fn_pad_control,
        row_sha: f.fn_row_sha,
        trace: match (&f.fn_route_trace, &f.fn_route_record, rank) {
            (Some(p), _, 0) => {
                println!("fntrace: recording routes + PLE rows to {}", p.display());
                Some(saber_mem::route_trace::RouteSink::Trace(
                    saber_mem::route_trace::TraceWriter::create(p, LAYERS, 10, 512)?,
                ))
            }
            (None, Some(d), 0) => {
                let cfg = f.record_cfg(d, LAYERS, 10, 512);
                println!(
                    "fnrecord: recording the served routes (target steps + PLE rows, off the rank thread) to {} | rotate every {} steps, ring {:.1} GiB, queue {} MiB",
                    d.display(),
                    cfg.rotate_steps,
                    cfg.max_bytes as f64 / 1073741824.0,
                    cfg.max_inflight >> 20
                );
                Some(saber_mem::route_trace::RouteSink::Record(
                    saber_mem::route_trace::TraceSink::spawn(cfg)?,
                ))
            }
            _ => None,
        },
        trace_meta: (saber_mem::route_trace::StepKind::Prefill, 0, 1, 0),
        inserts: laws[ri].inserts.clone(),
        diag: Fn30bDiag::new(hip, rank, f)?,
        borrow,
        top,
        seql,
        elastic,
        plane_dump: f.fn_plane_dump.clone(),
        budget: budgets[ri].clone(),
        ring,
        lend_fill: !f.no_fn_lend_fill,
        fill_passes: 0,
        fill_ms: 0.0,
        dpick: f.device_pick().then(|| DevPick {
            out: dpick_out,
            rows_max: (MAX_SEQS * super::exec::WMAX) as usize,
            rows: 0,
            checked: 0,
            differ: 0,
        }),
        ov112,
        hand,
        ahead: None,
        ahead_gather: None,
        slow119: 0,
        pick_cpus,
        pick_acc: (0, 0, 0, 0.0, 0.0, 0.0),
        f116: Vec::new(),
        f116_chunk_end: None,
        f116_kv: 0,
        f116_front: 0,
        front_want: false,
        front_done: false,
        lg_backup: None,
        front_early: 0,
        first_ah: None,
        first_dev: None,
        first_norm: None,
        front_side: if super::run::first_step_under(f) {
            Some(hip.create_stream(saber_hip::StreamFlags::NonBlocking)?)
        } else {
            None
        },
        front_under: None,
        first_chk: None,
        f121: super::ple121::F121::new(f.ple_form()?, f.fn_ple_trace.as_deref(), rank)?,
    };
    // FN117b (S1): the round's staging, sized by the boot's buffers (the embedding head rows, the
    // constant region, the device chain's input buffer)
    hand_s1_alloc(&mut rk)?;
    // FN119 (`--fn-device-step ahead`): the issue-ahead's buffers (the serve plane's MTP steps)
    ahead_alloc(&mut rk, f)?;
    // every pool slot starts zeroed, the spare included (FN17: a padded decode row reads and
    // writes the spare slot)
    for s in 0..=MAX_SEQS {
        rk.pools.zero(rk.hip, &rk.dev, s)?;
    }
    rk.hip.sync_stream(&rk.dev.stream)?;
    if rk.dpick.is_some() {
        pick_selftest(&mut rk)?;
    }
    draft_head_boot(&mut rk, &mut em, int2_host)?;
    if f.fn_mtp_chain_selftest {
        ensure!(
            rk.mtp.as_ref().is_some_and(|m| m.chain.is_some()),
            "--fn-mtp-chain-selftest runs the device chain (--fn-mtp k|auto, the verify step)"
        );
        chain_selftest(&mut rk, &mut em, &recs)?;
    }
    let mut gr = Graphs::default();
    if let super::run::GraphSplit::Lazy { q, .. } = rk.gsplit {
        // FN64c: the record's directory and grid (the served steps' runs, `--fn-graph-lazy-record`)
        gr.record.clone_from(&f.fn_graph_lazy_record);
        gr.lazy_grid = (q, rk.ple_ov.is_some().then_some(super::plan::PLE_LAYER));
        // FN64d / FN64e: the billed exec budget's / the charge's run caps
        let bill = f.graph_exec_bill()?;
        gr.exec_cap = bill.run_cap();
        gr.pre_cap = bill.pre_cap();
        gr.bill = bill;
        // FN98: one exec per form, the gates inside (the mover's mark words)
        if f.graph_form_one()? {
            ensure!(
                rk.tier.mover.mark_word_addr(0).is_some(),
                "--fn-graph-form one: the in-graph gate waits the mover's landed-mark words -- the one-stream mover under --xtier-mover-marks word (this boot's mover has none)"
            );
            gr.one = Some(q);
            gr.one_open = f.graph_form_open();
            let bytes = (LAYERS as usize * GATE98_ENT) as u64;
            let dev = rk.hip.malloc(bytes)?;
            rk.hip.memset(&dev, 0, &rk.dev.stream)?;
            gr.gate_tab = Some((dev, rk.hip.host_malloc(bytes)?));
            if gr.one_open {
                println!(
                    "fngraph98 rank{}: CONTROL one-open -- the gate nodes are NEVER armed (a step may read slots mid-copy); not a serving form",
                    rk.rank
                );
            }
            println!(
                "fngraph98 rank{}: ONE graph exec per step form (q {q}: gate blocks at the lazy cut points, a wait node each, set per launch); fn.graph.exec billed",
                rk.rank
            );
        }
    }
    // FN64b: every lazy run captured before the first step (`--fn-graph-lazy-pre`); FN64c: the
    // recorded set (`recorded`)
    lazy_precapture(&mut rk, &mut em, &mut gr, f, &f.lazy_set_dir(pack))?;
    let mut dp = 0usize;
    if f.fn_serve.is_some() || f.scripted() {
        let r = serve_rank(
            &mut rk, &mut em, &mut ple, &ph, &recs, cfg, ckpt, f, &mut gr, &mut dp, prompts, out,
        );
        drop(em);
        tier_end_line(&rk);
        println!("{}", rk.f121.end_line(rk.rank, &ple));
        borrow_line(&rk);
        if let Err(e) = plane_end(&rk) {
            eprintln!("fnplane rank{}: {e:#}", rk.rank);
        }
        ring_line(&rk);
        if f.fn_verify_slots && r.is_ok() {
            let mut stage = verify_stage(rk.hip, xt.header().stride)?;
            verify_slots(&mut rk, &xt, &mut stage)?;
        }
        if let Some(tw) = rk.trace.as_mut() {
            println!("{}", tw.finish()?);
        }
        return r;
    }
    // ---- the request loop
    let mut results = serde_json::Map::new();
    if f.fn_sched == "grow" {
        results.insert(
            "grow".into(),
            grow(
                &mut rk, &mut em, &mut ple, &ph, &recs, cfg, prompts, f, &mut gr, &mut dp,
            )?,
        );
    } else {
        ensure!(f.fn_sched == "groups", "--fn-sched groups|grow");
    }
    let mut hashes = serde_json::Map::new();
    let taps_dir = out.join(format!(
        "taps/chunk{}{}{}",
        f.fn_chunk,
        if f.fn_gdn_state_f32 { "_f32" } else { "" },
        if f.fn_ar_exact { "_arexact" } else { "" }
    ));
    let cap_key = format!(
        "chunk{}{}{}{}",
        f.fn_chunk,
        if f.fn_gdn_state_f32 { "_f32" } else { "" },
        if f.fn_ar_exact { "_arexact" } else { "" },
        f.fn_control
            .as_deref()
            .map(|c| format!("_{c}"))
            .unwrap_or_default()
    );
    for &conc in if f.fn_sched == "grow" {
        &[][..]
    } else {
        &f.fn_conc[..]
    } {
        let conc = conc.clamp(1, MAX_SEQS);
        for pass in 0..f.fn_repeat {
            let key = format!("conc{conc}_pass{pass}");
            let mut runs = Vec::new();
            let mut mtp_first_tops: Vec<Value> = Vec::new();
            for group in prompts.chunks(conc as usize) {
                let mut lives = Vec::new();
                let mut reqs = Vec::new();
                let mut firsts = Vec::new();
                let mut probe0: Vec<(f32, Vec<f32>)> = Vec::new();
                let mut sha0: Vec<Option<String>> = Vec::new();
                for (s, p) in group.iter().enumerate() {
                    let t0 = Instant::now();
                    rk.pools.zero(rk.hip, &rk.dev, s as u32)?;
                    let mut lv = Live::fresh(s as u32, cfg.eos_token_id);
                    let mut cap = f.fn_dump.then(|| Cap::for_prompt(p.ids.len()));
                    let tap = (f.fn_tap_layer >= 0).then(|| taps_dir.join(&p.id));
                    let (pick, st, po) = prefill(
                        &mut rk,
                        &mut em,
                        &mut ple,
                        &ph,
                        &recs,
                        &mut lv,
                        &p.ids,
                        f,
                        cap.as_mut(),
                        tap.as_deref(),
                        Some(&ref_req(Some(p), 0)),
                    )?;
                    lv.next = pick.0[0];
                    if let Some(dir) = &f.fn_prefill_logits_dir {
                        let address = rk.last_lg;
                        let (_, _, logits) = self::pick(&mut rk, address, 1, true)?;
                        if rank == 0 {
                            ensure!(logits.len() == VOCAB as usize, "incomplete next-token logit row");
                            let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
                            std::fs::write(dir.join(format!("{}.f32", p.id)), bytes)?;
                        }
                    }
                    firsts.push(pick.0[0]);
                    // FN80b-2: the mirrored placement after every prompt (a divergence = FATAL)
                    if f.experts_tp2() {
                        mirror_check(&rk, &p.id)?;
                    }
                    if rk.mtp.is_some() {
                        let (d1, top) = mtp_first(
                            &mut rk,
                            &mut em,
                            &recs,
                            &lv,
                            &p.ids,
                            pick.0[0],
                            f.fn_mtp_ref.as_deref(),
                            &p.id,
                            out,
                        )?;
                        mtp_first_tops.push(json!({"id": p.id, "d1": d1, "top": top}));
                    }
                    sha0.push(po.as_ref().and_then(|o| o.sha.clone()));
                    probe0.push(
                        po.map(|o| (o.lse, o.probe))
                            .unwrap_or((f32::NAN, Vec::new())),
                    );
                    if let Some(mut c) = cap {
                        c.logits = pick.2.clone();
                        hashes.insert(p.id.clone(), c.hashes());
                        if rank == 0 && f.fn_prompts.len() <= 4 {
                            c.write(
                                &out.join("capture").join(&cap_key).join(&p.id),
                                p,
                                "saber engine FN13 (worker form, TP=2 / EP=2)",
                            )?;
                        }
                    }
                    reqs.push(json!({"id": p.id, "tokens": p.ids.len(), "prefill_ms": t0.elapsed().as_secs_f64() * 1e3,
                        "prefill_dev_ms": st.dev_ms, "first": pick.0[0], "first_top": pick.1,
                        "tier": [st.resident, st.ram, st.nvme], "demand_distinct": st.distinct, "ple": [st.ple_hits, st.ple_nvme], "launches": st.launches,
                        "busy_ms": st.busy_ms, "groups": st.groups}));
                    lives.push(lv);
                }
                // FN80b-2: a prefill-only run records the first token's probe (the KL gate at
                // t = 0; `fn80b2_report.py kl`)
                if f.fn_decode == 0 {
                    for (i, r) in reqs.iter_mut().enumerate() {
                        r["probe"] = json!([probe0[i].clone()]);
                    }
                }
                if f.fn_decode > 0 {
                    let grp: Vec<&Prompt> = group.iter().collect();
                    let d = decode(
                        &mut rk,
                        &mut em,
                        &mut ple,
                        &ph,
                        &recs,
                        &mut lives,
                        f.fn_decode,
                        f,
                        &mut gr,
                        &mut dp,
                        &grp,
                    )?;
                    for (i, r) in reqs.iter_mut().enumerate() {
                        r["gen"] = json!(d.ids[i]);
                        r["gen_top"] = json!(d.tops[i]);
                        let mut pr = vec![probe0[i].clone()];
                        pr.extend(d.probes[i].iter().cloned());
                        r["probe"] = json!(pr);
                        if rk.row_sha {
                            let mut sh = vec![sha0[i].clone()];
                            sh.extend(d.shas[i].iter().cloned());
                            r["row_sha"] = json!(sh);
                        }
                        if rk.mtp.is_some() {
                            r["mtp_drafts"] = json!(d.mtp[i]);
                        }
                    }
                    let steps: Vec<Value> = d
                        .steps
                        .iter()
                        .map(|s| json!({"host_ms": s.host_ms, "dev_ms": s.dev_ms, "launches": s.launches,
                            "tier": [s.resident, s.ram, s.nvme], "ple": [s.ple_hits, s.ple_nvme, s.ple_ms],
                            "busy_ms": s.busy_ms, "groups": s.groups, "ops": s.ops, "host": s.host,
                            "gate_ms": s.gate_ms, "gate_n": s.gate_n}))
                        .collect();
                    let n = d.steps.len().max(1) as f64;
                    let host: f64 = d.steps.iter().map(|s| s.host_ms).sum::<f64>() / n;
                    let devm: f64 = d.steps.iter().map(|s| s.dev_ms).sum::<f64>() / n;
                    if rank == 0 {
                        println!(
                            "fndecode {key} b{} [{}]: {} steps, {:.2} ms/step host wall, {:.2} ms/step device span, {} launches/step",
                            lives.len(),
                            group.iter().map(|p| p.id.as_str()).collect::<Vec<_>>().join(","),
                            d.steps.len(),
                            host,
                            devm,
                            d.steps.first().map_or(0, |s| s.launches)
                        );
                        // FN56: the host census -- every named host phase's mean ms a step, and
                        // the MTP iteration's wall vs its device spans (verify + drafter round)
                        let mut ph: BTreeMap<&str, f64> = BTreeMap::new();
                        for s in &d.steps {
                            for (k, v) in &s.host {
                                *ph.entry(k).or_insert(0.0) += v / n;
                            }
                        }
                        let it = ph.get("z_iter").copied().unwrap_or(0.0);
                        let rd = ph.get("z_round_dev").copied().unwrap_or(0.0);
                        println!(
                            "fnhost56 {key} b{}: iter {:.3} ms = device {:.3} (verify {:.3} + round {:.3}) + exposed {:.3} | step host-dev {:.3} | tables prebuilt {}/{} | {}",
                            lives.len(),
                            it,
                            devm + rd,
                            devm,
                            rd,
                            it - devm - rd,
                            host - devm,
                            rk.tier_pre_n.0,
                            rk.tier_pre_n.1,
                            ph.iter()
                                .map(|(k, v)| format!("{k} {v:.3}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                        println!(
                            "{}",
                            fnhost59_line(&format!("{key} b{}", lives.len()), &rk.fn59)
                        );
                        rk.fn59.clear();
                        if f.fn_gate_time {
                            let gm: f64 = d.steps.iter().map(|s| s.gate_ms).sum::<f64>() / n;
                            let gn: f64 = d.steps.iter().map(|s| s.gate_n as f64).sum::<f64>() / n;
                            println!(
                                "fngate {key} b{}: {gm:.3} ms/step compute idle at the decode gates ({gn:.2} waiting gates/step; mover {})",
                                lives.len(),
                                f.xtier.xtier_mover
                            );
                        }
                    }
                    // the second turn: [the last pick] + a user turn, prefilled as a continuation,
                    // then decode again; the one-prompt rendering runs as a fresh sequence after it
                    if f.fn_turns > 1 {
                        ensure!(conc == 1, "--fn-turns 2 runs at conc 1");
                        let p = &group[0];
                        let partner = prompts
                            .iter()
                            .cycle()
                            .skip_while(|x| x.id != p.id)
                            .nth(1)
                            .context("partner")?;
                        let mut delta = vec![lives[0].next, EOS, 198];
                        delta.extend_from_slice(&partner.ids);
                        let lv = &mut lives[0];
                        // a continuation: normalize the state parities to 0 (chunk index 2 onward)
                        rk.pools
                            .normalize(rk.hip, &rk.dev, lv.slot, lv.gdn_at, lv.ple_at, 0)?;
                        lv.gdn_at = 0;
                        lv.ple_at = 0;
                        lv.chunks = lv.chunks.max(2).next_multiple_of(2);
                        let (pick2, _, _) = prefill(
                            &mut rk, &mut em, &mut ple, &ph, &recs, lv, &delta, f, None, None, None,
                        )?;
                        lv.next = pick2.0[0];
                        let d2 = decode(
                            &mut rk,
                            &mut em,
                            &mut ple,
                            &ph,
                            &recs,
                            &mut lives,
                            f.fn_decode,
                            f,
                            &mut gr,
                            &mut dp,
                            &[],
                        )?;
                        // the same conversation as ONE prompt
                        let mut one = p.ids.clone();
                        one.push(firsts[0]);
                        one.extend_from_slice(&d.ids[0][..d.ids[0].len() - 1]);
                        one.extend_from_slice(&delta);
                        rk.pools.zero(rk.hip, &rk.dev, 0)?;
                        let mut lv1 = Live::fresh(0, cfg.eos_token_id);
                        let (pick1, _, _) = prefill(
                            &mut rk, &mut em, &mut ple, &ph, &recs, &mut lv1, &one, f, None, None,
                            None,
                        )?;
                        lv1.next = pick1.0[0];
                        let mut v1 = vec![lv1];
                        let d1 = decode(
                            &mut rk,
                            &mut em,
                            &mut ple,
                            &ph,
                            &recs,
                            &mut v1,
                            f.fn_decode,
                            f,
                            &mut gr,
                            &mut dp,
                            &[],
                        )?;
                        reqs[0]["turn2"] = json!({"delta": delta.len(), "partner": partner.id,
                            "two_turn": {"first": pick2.0[0], "gen": d2.ids[0], "gen_top": d2.tops[0]},
                            "one_prompt": {"tokens": one.len(), "first": pick1.0[0], "gen": d1.ids[0], "gen_top": d1.tops[0]}});
                        if rank == 0 {
                            let a: Vec<u32> = std::iter::once(pick2.0[0])
                                .chain(d2.ids[0].iter().copied())
                                .collect();
                            let b: Vec<u32> = std::iter::once(pick1.0[0])
                                .chain(d1.ids[0].iter().copied())
                                .collect();
                            let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                            println!(
                                "fnturn2 {}: two-turn vs one-prompt common prefix {common}/{}",
                                p.id,
                                a.len()
                            );
                        }
                    }
                    runs.push(json!({"prompts": reqs, "steps": steps, "ms_per_step_host": host, "ms_per_step_dev": devm,
                        "mtp_ms": d.mtp_ms, "acc": d.acc, "widths": d.widths}));
                } else {
                    runs.push(json!({"prompts": reqs}));
                }
            }
            if !mtp_first_tops.is_empty() {
                results.insert(format!("{key}_mtp_first"), Value::Array(mtp_first_tops));
            }
            results.insert(key, Value::Array(runs));
        }
    }
    let chain = chain_report(&mut rk, out)?;
    let (sync_words, sync_nz) = moe_sync_check(&rk, &em)?;
    let fn30b = fn30b_report(&mut rk)?;
    let full = json!({
        "fn30b": fn30b,
        "moe_train": f.fn_moe_train.to_string(), "moe_blocks": em.moe_blocks, "moe_front_blocks": em.moe_front_blocks,
        "moe_sync": {"words": sync_words, "nonzero": sync_nz},
        "chain": chain,
        "mode": "engine-worker", "form": "worker-process", "rank": rank, "tp": 2, "chunk": f.fn_chunk,
        "decode": f.fn_decode, "turns": f.fn_turns, "ar_exact": f.fn_ar_exact, "state_f32": f.fn_gdn_state_f32,
        "evict_all": f.fn_evict_all, "control": f.fn_control, "max_ctx": rk.max_ctx,
        "ar_calls": {"exact": rk.pair.calls_exact, "wht6": rk.pair.calls_wht6, "decode_exact": rk.pair.calls_decode_exact},
        "capture": !f.no_fn_capture, "graphs": gr.census(), "declc_as_pair": em.declc_as_pair, "verify_class": f.fn_verify_class.to_string(), "vclass_layers": em.vclass_layers, "poison_step": f.fn_poison_step, "sched": f.fn_sched,
        "runs": results, "hashes": hashes,
        "mtp": rk.mtp.as_ref().map(|m| json!({"k": m.k, "launches": m.launches, "ref": f.fn_mtp_ref})),
        "tier_totals": format!("{:?}", rk.tier.tier.totals()), "mover_counts": format!("{:?}", rk.tier.mover.counts), "mover_hold": f.mover_hold(), "ple_totals": format!("{:?}", ple.totals()),
        "metrics": format!("{}{}", rk.tier.tier.render_prometheus(), ple.render_prometheus()),
    });
    let fn14 = format!(
        "{}{}{}{}",
        if f.no_fn_capture { "_eager" } else { "" },
        if f.fn_sched == "grow" { "_grow" } else { "" },
        f.fn_poison_step
            .map(|p| format!("_poison{p}"))
            .unwrap_or_default(),
        if f.fn_capture_verify { "_verify" } else { "" }
    );
    let fn14 = format!(
        "{fn14}{}{}",
        if f.fn_mtp.on() {
            format!("_mtp{}", f.fn_mtp)
        } else {
            String::new()
        },
        if f.fn_mtp_ref.is_some() {
            "_mtpref"
        } else {
            ""
        }
    );
    let name = format!(
        "worker_d{}_c{}{}{}{}{}{}{fn14}{}_rank{rank}.json",
        f.fn_decode,
        f.fn_chunk,
        if f.fn_gdn_state_f32 { "_f32" } else { "" },
        if f.fn_ar_exact { "_arexact" } else { "" },
        if f.fn_dump { "_cap" } else { "" },
        if f.fn_evict_all { "_evict" } else { "" },
        if f.fn_op_timing { "_optime" } else { "" },
        if f.fn_turns > 1 {
            "_turns".to_string()
        } else {
            f.fn_control
                .as_deref()
                .map(|c| format!("_{c}"))
                .unwrap_or_default()
        }
    );
    std::fs::write(out.join(&name), serde_json::to_string_pretty(&full)?)?;
    println!("fnresult rank{rank}: wrote {}", out.join(&name).display());
    rk.peer.rendezvous("fn13.end")?;
    drop(em);
    if f.fn_kv_pages {
        kv_pages_lines(&rk)?;
    }
    tier_end_line(&rk);
    if f.experts_tp2() {
        mirror_check(&rk, "end")?;
    }
    borrow_line(&rk);
    plane_end(&rk)?;
    ring_line(&rk);
    if f.fn_verify_slots {
        let mut stage = verify_stage(rk.hip, xt.header().stride)?;
        verify_slots(&mut rk, &xt, &mut stage)?;
        // FN80b-2: the placement law -- both ranks' maps EQUAL (after the slot check's settle)
        if f.experts_tp2() {
            mirror_check(&rk, "verify")?;
        }
    }
    if let Some(tw) = rk.trace.as_mut() {
        println!("{}", tw.finish()?);
    }
    Ok(())
}

/// A pick: (per-row greedy ids, per-row merged top-20 (id, logit), row 0's full logits when asked).
type Pick = (Vec<u32>, Vec<Vec<(u32, f32)>>, Vec<f32>);

/// Each rank's top-20 over its vocabulary half per row, swapped, merged: the greedy pick.
fn pick(rk: &mut Rank<'_, '_>, lg: u64, rows: usize, keep_logits: bool) -> anyhow::Result<Pick> {
    let half = VOCAB as usize / 2;
    let raw = d2h_in(&rk.dev, &rk.dev.arena, lg, rows * half * 2)?;
    let mut mine = Vec::with_capacity(rows * TOPK * 8);
    let mut row0 = Vec::new();
    for r in 0..rows {
        let v = bf16_to_f32(&raw[r * half * 2..(r + 1) * half * 2]);
        let mut ix: Vec<usize> = (0..half).collect();
        ix.select_nth_unstable_by(TOPK, |&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
        let mut top: Vec<usize> = ix[..TOPK].to_vec();
        top.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
        for &i in &top {
            mine.extend_from_slice(&((i + rk.rank as usize * half) as u32).to_le_bytes());
            mine.extend_from_slice(&v[i].to_le_bytes());
        }
        if keep_logits && r == 0 {
            row0 = v;
        }
    }
    let theirs = rk.peer.exchange().xchg_bytes("fn.pick", &mine)?;
    ensure!(
        theirs.len() == mine.len(),
        "the peer's pick frame differs in size"
    );
    let theirs_logits = if keep_logits {
        rk.peer
            .exchange()
            .xchg_bytes("fn.logits", &raw[..half * 2])?
    } else {
        Vec::new()
    };
    let rd = |b: &[u8], k: usize| -> (u32, f32) {
        (
            u32::from_le_bytes(b[k * 8..k * 8 + 4].try_into().expect("4")),
            f32::from_le_bytes(b[k * 8 + 4..k * 8 + 8].try_into().expect("4")),
        )
    };
    let mut ids = Vec::with_capacity(rows);
    let mut tops = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut all: Vec<(u32, f32)> = (0..TOPK)
            .flat_map(|k| [rd(&mine, r * TOPK + k), rd(&theirs, r * TOPK + k)])
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(TOPK);
        ids.push(all[0].0);
        tops.push(all);
    }
    let full = if keep_logits {
        let o = bf16_to_f32(&theirs_logits);
        if rk.rank == 0 {
            row0.into_iter().chain(o).collect()
        } else {
            o.into_iter().chain(row0).collect()
        }
    } else {
        Vec::new()
    };
    Ok((ids, tops, full))
}

/// FN42b: the borrow's slot alignment in the arena's tail.
const BORROW_ALIGN: u64 = 1 << 21;

/// FN42b: the arena borrow of one rank (see [`borrow_for`]).
struct Borrow {
    /// The lendable slot ids (the wide arena's tail, one stride each).
    ids: Vec<u32>,
    /// The tail is lent to the tier (no wide chunk owns it).
    lent: bool,
    /// The scratch caps: the narrow rung's (lent) and the wide arena's (reclaimed).
    narrow_cap: u64,
    wide_cap: u64,
    /// FN44b: the TWO-LAYOUT borrow (the default; `--fn-borrow-cut` = FN42b's one layout):
    /// `(base, narrow arena, wide arena)` -- a lent pass runs in the narrow arena, a reclaimed
    /// one in the wide arena, both carved at the base.
    layouts: Option<(u64, ArenaSizes, ArenaSizes)>,
    /// The rank's ledger: `fn.arena.prefill` + `xtier.kv_lent` balance at every flip.
    ledger: saber_mem::ledger::Ledger,
    arena_bytes: u64,
    lend_bytes: u64,
    reclaims: u64,
    lends: u64,
    dropped: u64,
    reclaim_ms: f64,
    lend_ms: f64,
    /// `--fn-verify-slots`: the placement byte law after every reclaim too.
    verify: bool,
}

impl Borrow {
    /// The emitter's arena at boot (lent): FN44b's narrow arena, or the cut's capped scratch.
    fn boot_arena(&self, base: u64, wide: ArenaSizes) -> Arena {
        match self.layouts {
            Some((b, nw, _)) => Arena::carve(b, nw),
            None => {
                let mut a = Arena::carve(base, wide);
                a.scratch.cap = self.narrow_cap;
                a
            }
        }
    }

    /// Flip the emitter's arena to the `lent` / reclaimed state (between passes: nothing seeded
    /// is pending, every region restarts at the next pass's first layer).
    fn flip_arena(&self, em: &mut Emitter<'_>, lent: bool) -> anyhow::Result<()> {
        ensure!(
            em.arena.pending_seeds.is_empty(),
            "the borrow flipped the arena with seeds pending"
        );
        match self.layouts {
            Some((base, nw, wide)) => em.arena = Arena::carve(base, if lent { nw } else { wide }),
            None => em.arena.scratch.cap = if lent { self.narrow_cap } else { self.wide_cap },
        }
        Ok(())
    }

    /// The ledger line of the current state (the referee refuses an imbalance).
    fn relend(&mut self, b: &saber_mem::flashnext_budget::Ep2Budget) -> anyhow::Result<()> {
        let lent = if self.lent { self.lend_bytes } else { 0 };
        b.relend(&mut self.ledger, "fn.arena.prefill", self.arena_bytes, lent)
            .map_err(|e| anyhow::anyhow!("the borrow's ledger: {e}"))
    }
}

/// FN96: THE IDLE SEQUENCE SLOTS' STATE LENT TO THE TIER (`--xtier-lend-seq-pools auto|on`): the
/// pools' lendable spans (sequence slots `0..MAX_SEQS`; the SPARE never) carved into whole
/// strides, each lendable while every sequence slot it overlaps has been free for `hold` decode
/// steps. The serve plane lends at a decode command's start and reclaims a slot's strides before
/// anything writes the slot (an admission's zeroing, a ready move into a free slot): the
/// deferred tier end resolved, the copy stream drained, every lent record flipped to its RAM
/// shadow (dropped, no copy). A lent stride is never written by a pass: the decode tables address
/// live and pad (SPARE) slots only, a prefill addresses its own (held) slot.
struct SeqLend {
    first: u32,
    /// Per lendable stride (`first + i`): the sequence-slot mask it overlaps.
    masks: Vec<u32>,
    lent: Vec<bool>,
    /// Decode commands each sequence slot has been free (saturating at `hold`).
    free_for: [u32; MAX_SEQS as usize],
    hold: u32,
    stride: u64,
    /// The `fn.pools` class's bytes (the ledger referee) and its own ledger.
    pool_bytes: u64,
    ledger: saber_mem::ledger::Ledger,
    lends: u64,
    lent_strides: u64,
    reclaims: u64,
    reclaimed: u64,
    dropped: u64,
    drop_max: u64,
    max_lent: u64,
    lend_ms: f64,
    reclaim_ms: f64,
    reclaim_max_ms: f64,
    /// (strides lendable at conc 0 / 1 / 2 / 4 / 8) -- the boot's census.
    census: [u64; 5],
}

impl SeqLend {
    fn lent_now(&self) -> u64 {
        self.lent.iter().filter(|&&x| x).count() as u64
    }
    fn relend(&mut self, b: &saber_mem::flashnext_budget::Ep2Budget) -> anyhow::Result<()> {
        let lent = self.lent_now() * self.stride;
        b.relend_as(
            &mut self.ledger,
            "fn.pools",
            self.pool_bytes,
            lent,
            "xtier.seq_pools_lent",
        )
        .map_err(|e| anyhow::anyhow!("the seq pools' ledger: {e}"))
    }
    fn line(&self, rank: u32) -> String {
        format!(
            "fnseqlend rank{rank}: THE IDLE SEQUENCE SLOTS (FN96) -- lendable {} strides (conc \
             0/1/2/4/8: {:?}) | lent now {} | max lent {} | lends {} ({} strides, {:.3} ms total) \
             | reclaims {} ({} strides, {} records dropped, max {} a reclaim, {:.3} ms total, max \
             {:.3} ms) | hold {} decode commands",
            self.masks.len(),
            self.census,
            self.lent_now(),
            self.max_lent,
            self.lends,
            self.lent_strides,
            self.lend_ms,
            self.reclaims,
            self.reclaimed,
            self.dropped,
            self.drop_max,
            self.reclaim_ms,
            self.reclaim_max_ms,
            self.hold
        )
    }
}

/// FN96: register the pools' lendable strides on the tier and the copy queue, lend those of the
/// free slots (all of them at boot), EMPTY (the first prefill fills them, after the boot zeroing).
#[allow(clippy::too_many_arguments)]
fn seq_lend_boot(
    rank: u32,
    f: &FnFlags,
    pools: &Pools,
    tier: &mut super::run::TierRank<'_, '_>,
    budget: &saber_mem::flashnext_budget::Ep2Budget,
    stride: u64,
) -> anyhow::Result<Option<SeqLend>> {
    let must = f.xtier.lend_seq_pools().1;
    // the forms it does not model: `on` refuses, `auto` stands down by name
    let why = if f.fn_serve.is_none() && !f.scripted() {
        Some("a harness run (the serve plane's admission drives the lend)")
    } else if f.fn_pad_control {
        Some("--fn-pad-control (pad rows write free slots)")
    } else if f.fn_one_process || f.fn_poison.is_some() || f.fn_poison_step.is_some() {
        Some("the one-process form / the poison arms")
    } else if f.fn_mixed {
        Some("--fn-mixed (the packed mixed step is not modelled)")
    } else {
        None
    };
    if let Some(why) = why {
        ensure!(!must, "--xtier-lend-seq-pools on: refused under {why}");
        println!("fnseqlend rank{rank}: stands down -- {why}");
        return Ok(None);
    }
    let t = Instant::now();
    let nseq = MAX_SEQS;
    let spans = pools.lend_spans()?;
    // the host census's law (seqpool::spans) is the allocation's, span for span
    ensure!(
        spans
            .iter()
            .map(|x| x.1.seq_bytes)
            .eq(
                super::seqpool::spans(f.pools_layout(), pools.cb, pools.hb, pools.pool_w)
                    .iter()
                    .map(|x| x.seq_bytes)
            ),
        "the pools' lendable spans differ from the census's"
    );
    let mut addrs = Vec::new();
    let mut regions = Vec::new();
    let mut masks = Vec::new();
    for (owner, sp) in &spans {
        let (lo, m) = super::seqpool::carve(*sp, nseq, stride);
        if m.is_empty() {
            continue;
        }
        let base = owner.addr().as_u64();
        addrs.extend((0..m.len() as u64).map(|i| base + lo + i * stride));
        regions.push((std::rc::Rc::clone(owner), lo, masks.len() as u32));
        masks.extend(m);
    }
    if masks.is_empty() {
        println!("fnseqlend rank{rank}: INERT -- no whole stride in the pools' spans");
        return Ok(None);
    }
    let first = match tier.tier.register_lendable(&addrs) {
        Ok(x) => x,
        Err(e) => {
            ensure!(!must, "--xtier-lend-seq-pools on: {e}");
            println!("fnseqlend rank{rank}: stands down -- the tier refuses lent slots ({e})");
            return Ok(None);
        }
    };
    for (owner, lo, i) in regions {
        tier.mover.queue_mut().set_lend(owner, lo, first + i);
    }
    let sp: Vec<super::seqpool::Span> = spans.iter().map(|x| x.1).collect();
    let free = |c: u32| (0xffu32 << c) & ((1u32 << nseq) - 1);
    let census = [0, 1, 2, 4, 8].map(|c| super::seqpool::lendable(&sp, nseq, stride, free(c)));
    let n = masks.len();
    let mut ledger = saber_mem::ledger::Ledger::new();
    budget
        .declare(&mut ledger, 0)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    // the class as the ledger declared it (the exact ledger: the allocation, byte for byte)
    let vr = saber_mem::ledger::Pool::Vram(rank);
    let pool_bytes = ledger
        .release(vr, "fn.pools")
        .map_err(|e| anyhow::anyhow!("rank {rank}: the fn.pools class: {e}"))?;
    ledger
        .reserve(vr, "fn.pools", pool_bytes)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    ensure!(
        n as u64 * stride <= pool_bytes,
        "the seq pools' strides past the fn.pools class"
    );
    let hold = f.xtier.xtier_lend_seq_pools_hold;
    let mut sl = SeqLend {
        first,
        masks,
        lent: vec![true; n],
        free_for: [hold; MAX_SEQS as usize],
        hold,
        stride,
        pool_bytes,
        ledger,
        lends: 1,
        lent_strides: n as u64,
        reclaims: 0,
        reclaimed: 0,
        dropped: 0,
        drop_max: 0,
        max_lent: n as u64,
        lend_ms: 0.0,
        reclaim_ms: 0.0,
        reclaim_max_ms: 0.0,
        census,
    };
    let ids: Vec<u32> = (first..first + n as u32).collect();
    tier.tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    sl.relend(budget)?;
    // NO fill here: the boot zeroes every pool slot AFTER this (the Rank's construction), which
    // would overwrite a filled stride's expert bytes (FN96 h1: the first requests read zeroed
    // experts until the strides churned). The strides are lent EMPTY (FN76b's form); the first
    // prefill pass's fill on lend (FN44b) fills them behind the zeroing.
    tier.tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the seq pools' boot lend: {e}"))?;
    println!(
        "fnseqlend rank{rank}: THE IDLE SEQUENCE SLOTS (FN96) -- layout {} | {} spans, {n} \
         lendable strides of {stride} B (ids {first}..{}) = {:.1} MiB of the {:.1} MiB pools | \
         lendable at conc 0/1/2/4/8: {:?} | LENT EMPTY at boot (filled by the first prefill) | an \
         admission reclaims its slot's strides (records dropped to their RAM shadows); a slot \
         free {hold} decode commands is lent again | host {:.0} ms",
        f.pools_layout().name(),
        spans.len(),
        first + n as u32,
        (n as u64 * stride) as f64 / 1048576.0,
        pools.bytes() as f64 / 1048576.0,
        census,
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(Some(sl))
}

/// FN96: before anything writes sequence slot `s` (an admission's zeroing, a ready move into a
/// free slot): reclaim every lent stride that overlaps it.
fn seq_reclaim(rk: &mut Rank<'_, '_>, s: u32) -> anyhow::Result<()> {
    let Some(sl) = rk.seql.as_mut() else {
        return Ok(());
    };
    if s >= MAX_SEQS {
        return Ok(());
    }
    sl.free_for[s as usize] = 0;
    let bit = 1u32 << s;
    let ids: Vec<u32> = sl
        .masks
        .iter()
        .enumerate()
        .filter(|&(i, &m)| m & bit != 0 && sl.lent[i])
        .map(|(i, _)| sl.first + i as u32)
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    tier_flush(rk)?;
    let t0 = Instant::now();
    rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    let before = rk.tier.tier.lent().0;
    let d = rk.tier.tier.reclaim(&ids) as u64;
    ensure!(
        rk.tier.tier.lent().0 + ids.len() as u32 == before,
        "a lent seq-pool stride survived the reclaim"
    );
    rk.tier
        .tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the seq pools' reclaim: {e}"))?;
    let sl = rk.seql.as_mut().context("seql")?;
    for &id in &ids {
        sl.lent[(id - sl.first) as usize] = false;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    sl.reclaims += 1;
    sl.reclaimed += ids.len() as u64;
    sl.dropped += d;
    sl.drop_max = sl.drop_max.max(d);
    sl.reclaim_ms += ms;
    sl.reclaim_max_ms = sl.reclaim_max_ms.max(ms);
    let budget = rk.budget.clone();
    sl.relend(&budget)
}

/// FN96: at a decode command's start: count each sequence slot's free commands (`held` = the
/// serve state's occupied slots) and lend every stride whose slots have all been free `hold`
/// commands (EMPTY: the insert law / the fill on lend refills them behind the step's fence).
fn seq_lend_sync(rk: &mut Rank<'_, '_>, held: &[bool]) -> anyhow::Result<()> {
    let Some(sl) = rk.seql.as_mut() else {
        return Ok(());
    };
    let mut ok = 0u32;
    for s in 0..MAX_SEQS as usize {
        if held.get(s).copied().unwrap_or(false) {
            sl.free_for[s] = 0;
        } else {
            sl.free_for[s] = sl.free_for[s].saturating_add(1).min(sl.hold);
            if sl.free_for[s] >= sl.hold {
                ok |= 1 << s;
            }
        }
    }
    let ids: Vec<u32> = sl
        .masks
        .iter()
        .enumerate()
        .filter(|&(i, &m)| !sl.lent[i] && m & !ok == 0)
        .map(|(i, _)| sl.first + i as u32)
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    let t0 = Instant::now();
    rk.tier
        .tier
        .lend(&ids)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let sl = rk.seql.as_mut().context("seql")?;
    for &id in &ids {
        sl.lent[(id - sl.first) as usize] = true;
    }
    sl.lends += 1;
    sl.lent_strides += ids.len() as u64;
    sl.max_lent = sl.max_lent.max(sl.lent_now());
    sl.lend_ms += t0.elapsed().as_secs_f64() * 1e3;
    let budget = rk.budget.clone();
    sl.relend(&budget)
}

/// FN76b: THE NARROW ARENA'S TOP lent at decode (`--xtier-lend-arena-top on`): the scratch past
/// the boundary `cap` (every pass of at most `rows` rows fits below it: the host-emitted decode
/// forms, a `rows`-row prefill chunk + its drafter pass, + the margin) is the tier's lendable
/// slots while no larger pass runs.
struct Top {
    /// The lendable slot ids (one stride each from `cap` up, inside the scratch).
    ids: Vec<u32>,
    /// The boundary (bytes from the scratch's base): a lent pass's scratch cap.
    cap: u64,
    /// The scratch cap to restore at a reclaim (the carve's own; saved at the lend).
    full_cap: u64,
    /// The largest prefill pass that keeps the top lent.
    rows: u32,
    lent: bool,
    /// The top's lent bytes and the arena class it is carved from (its own ledger referee).
    bytes: u64,
    arena_bytes: u64,
    ledger: saber_mem::ledger::Ledger,
    lends: u64,
    reclaims: u64,
    dropped: u64,
    /// The most records one reclaim dropped.
    drop_max: u64,
    reclaim_ms: f64,
    lend_ms: f64,
    /// The scratch high-water marks measured while lent (bytes): decode-form passes, short
    /// prefill passes (incl. the drafter's prefill-form passes); the next pass's kind.
    high_dec: u64,
    high_pf: u64,
    last: Option<bool>,
}

impl Top {
    fn relend(&mut self, b: &saber_mem::flashnext_budget::Ep2Budget) -> anyhow::Result<()> {
        let lent = if self.lent { self.bytes } else { 0 };
        b.relend_as(
            &mut self.ledger,
            "fn.arena.prefill",
            self.arena_bytes,
            lent,
            "xtier.arena_top_lent",
        )
        .map_err(|e| anyhow::anyhow!("the arena top's ledger: {e}"))
    }

    /// Attribute the scratch high-water mark since the last pass boundary to that pass's kind.
    fn note(&mut self, em: &mut Emitter<'_>) {
        let h = em.arena.scratch.high;
        match self.last {
            Some(true) => self.high_pf = self.high_pf.max(h),
            Some(false) => self.high_dec = self.high_dec.max(h),
            None => {}
        }
        em.arena.scratch.high = 0;
    }

    fn line(&self, rank: u32) -> String {
        let mib = |x: u64| x as f64 / 1048576.0;
        format!(
            "fnlend rank{rank}: THE ARENA TOP (FN76b) -- {} slots at +{:.1} MiB of the scratch \
             (keep <= {} rows) | lends {} ({:.3} ms total) | reclaims {} ({:.3} ms total, {} records \
             dropped, max {} a reclaim) | lent now {} | measured scratch high while lent: decode \
             forms {:.1} MiB, short prefill {:.1} MiB (boundary {:.1})",
            self.ids.len(),
            mib(self.cap),
            self.rows,
            self.lends,
            self.lend_ms,
            self.reclaims,
            self.reclaim_ms,
            self.dropped,
            self.drop_max,
            if self.lent { "LENT" } else { "reclaimed" },
            mib(self.high_dec),
            mib(self.high_pf),
            mib(self.cap)
        )
    }
}

/// FN76b: THE ARENA TOP'S LAW around [`borrow_for`] at a pass of `rows` rows (`prefill`: a prefill
/// chunk or a drafter prefill-form pass; `borrow`: the call site runs the borrow law too). A
/// prefill pass of more than the top's rows RECLAIMS the top first (the deferred tier end
/// resolved, the copy stream drained, every lent record flipped to its RAM shadow -- dropped, no
/// copy -- the scratch cap restored); any other pass LENDS it (EMPTY slots, as the borrow's: the
/// insert law / the fill on lend refills them behind the step's fence) when the borrow's narrow
/// arena is the carve, the scratch capped at the boundary and restarted (every emission path
/// starts at `new_layer`; nothing a pass leaves in scratch is read past it -- the borrow's law).
fn lend_for(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    rows: u32,
    prefill: bool,
    borrow: bool,
) -> anyhow::Result<()> {
    // FN100: the buffers' tails -- lent at a decode-form pass, reclaimed by a wider one
    buf_for(rk, rows, prefill)?;
    let full = match rk.top.as_mut() {
        Some(t) => {
            t.note(em);
            prefill && rows > t.rows
        }
        None => {
            if borrow {
                borrow_for(rk, em, rows)?;
            }
            return Ok(());
        }
    };
    if full && rk.top.as_ref().is_some_and(|t| t.lent) {
        tier_flush(rk)?;
        let t0 = Instant::now();
        rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
        let t = rk.top.as_mut().context("top")?;
        let before = rk.tier.tier.lent().0;
        let d = rk.tier.tier.reclaim(&t.ids) as u64;
        ensure!(
            rk.tier.tier.lent().0 + t.ids.len() as u32 == before,
            "a lent arena-top slot survived the reclaim"
        );
        rk.tier
            .tier
            .check_invariants()
            .map_err(|e| anyhow::anyhow!("after the arena top's reclaim: {e}"))?;
        em.arena.scratch.cap = t.full_cap;
        t.lent = false;
        t.reclaims += 1;
        t.dropped += d;
        t.drop_max = t.drop_max.max(d);
        t.reclaim_ms += t0.elapsed().as_secs_f64() * 1e3;
        let budget = rk.budget.clone();
        t.relend(&budget)?;
    }
    if borrow {
        borrow_for(rk, em, rows)?;
    }
    // the top lives in the narrow carve's scratch: lent only while the borrow (if any) is lent
    let narrow = rk.borrow.as_ref().is_none_or(|b| b.lent);
    let t = rk.top.as_mut().context("top")?;
    // a drafter prefill-form pass (no borrow law) only reclaims
    if !full && narrow && !t.lent && (borrow || !prefill) {
        ensure!(
            em.arena.pending_seeds.is_empty(),
            "the arena top lent with seeds pending"
        );
        let t0 = Instant::now();
        rk.tier
            .tier
            .lend(&t.ids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        t.full_cap = em.arena.scratch.cap;
        // FN80d: the TIER's record stride (a half record under `--fn-experts tp2`: twice the ids
        // in the same bytes), never EP's whole-record constant -- the constant made every tp2 boot's
        // first relend after a wide prefill FATAL (h1)
        ensure!(
            t.cap + t.ids.len() as u64 * rk.tier.tier.stride() <= t.full_cap,
            "the arena top's slots past the carve's scratch"
        );
        em.arena.scratch.reset();
        t.lent = true;
        t.lends += 1;
        t.lend_ms += t0.elapsed().as_secs_f64() * 1e3;
        let budget = rk.budget.clone();
        t.relend(&budget)?;
    }
    if t.lent {
        // a borrow flip re-carves the narrow arena (its cap): the boundary again
        em.arena.scratch.cap = t.cap;
        t.last = Some(prefill);
    } else {
        t.last = None;
    }
    Ok(())
}

/// FN100: THE BUFFERS' TAILS lent at decode (`--xtier-lend-buffers-top auto|on`): the row-scaled
/// buffers' strides past their heads (`super::bufs::layout`) are the tier's lendable slots while
/// no pass of more than `keep` rows writes them. Lent EMPTY at boot (nothing has written the
/// buffers: FN96's h1 law) and at every decode-form pass; a write of more rows (a wider chunk's
/// staging, the drafter's rows over it, a seeded hidden) reclaims them FIRST (the deferred tier
/// end resolved, the copy stream drained, every lent record flipped to its RAM shadow -- dropped,
/// no copy); an in-pass write past the heads while lent REFUSES ([`buf_check`]).
struct BufTop {
    ids: Vec<u32>,
    /// The rows every head holds.
    keep: u32,
    lent: bool,
    /// The lent bytes; the class's (`fn.buffers`) declared bytes and its own ledger referee.
    bytes: u64,
    class_bytes: u64,
    ledger: saber_mem::ledger::Ledger,
    lends: u64,
    reclaims: u64,
    dropped: u64,
    drop_max: u64,
    lend_ms: f64,
    reclaim_ms: f64,
    reclaim_max_ms: f64,
    /// The widest write that reclaimed (rows).
    widest: u32,
    /// Per tail: (buffer, slots).
    tails: Vec<(&'static str, u64)>,
}

impl BufTop {
    fn relend(&mut self, b: &saber_mem::flashnext_budget::Ep2Budget) -> anyhow::Result<()> {
        let lent = if self.lent { self.bytes } else { 0 };
        b.relend_as(
            &mut self.ledger,
            "fn.buffers",
            self.class_bytes,
            lent,
            "xtier.buffers_top_lent",
        )
        .map_err(|e| anyhow::anyhow!("the buffers top's ledger: {e}"))
    }

    fn line(&self, rank: u32) -> String {
        format!(
            "fnbuflend rank{rank}: THE BUFFERS' TAILS (FN100) -- {} slots ({}) past heads of {} \
             rows | lends {} ({:.3} ms total) | reclaims {} ({:.3} ms total, max {:.3} ms, {} \
             records dropped, max {} a reclaim, widest write {} rows) | lent now {}",
            self.ids.len(),
            self.tails
                .iter()
                .map(|(n, s)| format!("{n} {s}"))
                .collect::<Vec<_>>()
                .join(" + "),
            self.keep,
            self.lends,
            self.lend_ms,
            self.reclaims,
            self.reclaim_ms,
            self.reclaim_max_ms,
            self.dropped,
            self.drop_max,
            self.widest,
            if self.lent { "LENT" } else { "reclaimed" }
        )
    }
}

/// FN100: register the buffers' tails as the tier's lendable slots and lend them (EMPTY). `None`
/// = stood down by name (`auto`) or nothing to lend.
fn buf_boot(
    rank: u32,
    f: &FnFlags,
    lay: &super::bufs::Layout,
    own: &std::rc::Rc<DevicePtr>,
    tier: &mut super::run::TierRank<'_, '_>,
    budget: &saber_mem::flashnext_budget::Ep2Budget,
    stride: u64,
) -> anyhow::Result<Option<BufTop>> {
    let must = f.xtier.lend_buffers_top().1;
    // the forms it does not model: `on` refuses, `auto` stands down by name
    let why = if f.fn_one_process || f.fn_poison.is_some() || f.fn_poison_step.is_some() {
        Some("the one-process form / the poison arms")
    } else if f.fn_mixed {
        Some("--fn-mixed (the packed mixed step is not modelled)")
    } else if f.fn_control.as_deref() == Some("mtp-wide-capture") {
        Some("the drafter's wide-capture control")
    } else if f.fn_pregate_capture {
        Some("--fn-pregate-capture (a diagnostic)")
    } else {
        None
    };
    if let Some(why) = why {
        ensure!(!must, "--xtier-lend-buffers-top on: refused under {why}");
        println!("fnbuflend rank{rank}: stands down -- {why}");
        return Ok(None);
    }
    let t = Instant::now();
    let tails = lay.tails(stride);
    if tails.is_empty() {
        println!(
            "fnbuflend rank{rank}: INERT -- no whole stride past the heads of {} rows ({:.1} MiB)",
            lay.keep,
            lay.total as f64 / 1048576.0
        );
        return Ok(None);
    }
    let base = own.addr().as_u64();
    let mut addrs = Vec::new();
    let mut regions = Vec::new();
    for &(_, lo, n) in &tails {
        regions.push((lo, addrs.len() as u32));
        addrs.extend((0..n).map(|i| base + lo + i * stride));
    }
    let first = match tier.tier.register_lendable(&addrs) {
        Ok(x) => x,
        Err(e) => {
            ensure!(!must, "--xtier-lend-buffers-top on: {e}");
            println!("fnbuflend rank{rank}: stands down -- the tier refuses lent slots ({e})");
            return Ok(None);
        }
    };
    for (lo, i) in regions {
        tier.mover
            .queue_mut()
            .set_lend(std::rc::Rc::clone(own), lo, first + i);
    }
    let ids: Vec<u32> = (first..first + addrs.len() as u32).collect();
    tier.tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ledger = saber_mem::ledger::Ledger::new();
    budget
        .declare(&mut ledger, 0)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    let vr = saber_mem::ledger::Pool::Vram(rank);
    let class_bytes = ledger
        .release(vr, "fn.buffers")
        .map_err(|e| anyhow::anyhow!("rank {rank}: the fn.buffers class: {e}"))?;
    ledger
        .reserve(vr, "fn.buffers", class_bytes)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    let mut bt = BufTop {
        bytes: addrs.len() as u64 * stride,
        ids,
        keep: lay.keep,
        lent: true,
        class_bytes,
        ledger,
        lends: 0,
        reclaims: 0,
        dropped: 0,
        drop_max: 0,
        lend_ms: 0.0,
        reclaim_ms: 0.0,
        reclaim_max_ms: 0.0,
        widest: 0,
        tails: tails.iter().map(|x| (x.0, x.2)).collect(),
    };
    bt.relend(budget)?;
    let mib = |x: u64| x as f64 / 1048576.0;
    println!(
        "fnbuflend rank{rank}: THE BUFFERS' TAILS (FN100) -- one allocation {:.1} MiB (the \
         separate buffers' {:.1}), heads of {} rows at each base ({}); tails = {} lendable slots \
         ({:.1} MiB; ids {first}..{}) LENT at boot, EMPTY: a write of > {} rows reclaims them \
         (records dropped to their RAM shadows); host {:.0} ms",
        mib(lay.total),
        mib(lay.separate_bytes()),
        lay.keep,
        lay.parts
            .iter()
            .map(|p| format!("{} +{:.1} head {:.2}", p.name, mib(p.off), mib(p.head)))
            .collect::<Vec<_>>()
            .join(", "),
        bt.ids.len(),
        mib(bt.bytes),
        first + bt.ids.len() as u32,
        lay.keep,
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(Some(bt))
}

/// FN100: a write of `rows` rows into the row-scaled buffers is coming (`what` names it): past
/// the heads while the tails are lent, RECLAIM them first (the deferred tier end resolved, the
/// copy stream drained: no promotion into a tail in flight when the write lands).
fn buf_need(rk: &mut Rank<'_, '_>, rows: u32, what: &str) -> anyhow::Result<()> {
    match rk.buft.as_ref() {
        Some(t) if t.lent && rows > t.keep => {}
        _ => return Ok(()),
    }
    tier_flush(rk)?;
    let t0 = Instant::now();
    rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    let t = rk.buft.as_mut().context("buft")?;
    let before = rk.tier.tier.lent().0;
    let d = rk.tier.tier.reclaim(&t.ids) as u64;
    ensure!(
        rk.tier.tier.lent().0 + t.ids.len() as u32 == before,
        "a lent buffers-tail slot survived the reclaim ({what})"
    );
    rk.tier
        .tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the buffers' reclaim ({what}): {e}"))?;
    t.lent = false;
    t.reclaims += 1;
    t.dropped += d;
    t.drop_max = t.drop_max.max(d);
    t.widest = t.widest.max(rows);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    t.reclaim_ms += ms;
    t.reclaim_max_ms = t.reclaim_max_ms.max(ms);
    let budget = rk.budget.clone();
    t.relend(&budget)
}

/// FN100: an in-pass write of `rows` rows (a D2D into the route log / the hidden, a pass's
/// embedding rows) REFUSES past the heads while the tails are lent: the lend's law is that no
/// pass writes a lent stride (FN51f-b's fault class), proven at every emission.
fn buf_check(rk: &Rank<'_, '_>, rows: u32, what: &str) -> anyhow::Result<()> {
    if let Some(t) = rk.buft.as_ref() {
        ensure!(
            !(t.lent && rows > t.keep),
            "FN100: {what} of {rows} rows past the buffers' heads ({}) while their tails are lent",
            t.keep
        );
    }
    Ok(())
}

/// FN100: a decode-form pass (`prefill` false) of `rows` rows lends the tails again (EMPTY: the
/// insert law fills them behind the step's fence, as FN76b's top); a prefill pass of more rows
/// reclaims (its staging already did).
fn buf_for(rk: &mut Rank<'_, '_>, rows: u32, prefill: bool) -> anyhow::Result<()> {
    let Some(t) = rk.buft.as_ref() else {
        return Ok(());
    };
    if rows > t.keep {
        return buf_need(
            rk,
            rows,
            if prefill {
                "a wide pass"
            } else {
                "a wide decode form"
            },
        );
    }
    if prefill || t.lent {
        return Ok(());
    }
    let t0 = Instant::now();
    let t = rk.buft.as_mut().context("buft")?;
    rk.tier
        .tier
        .lend(&t.ids)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    t.lent = true;
    t.lends += 1;
    t.lend_ms += t0.elapsed().as_secs_f64() * 1e3;
    let budget = rk.budget.clone();
    t.relend(&budget)
}

/// FN100: which row-scaled buffer an H2D fills.
#[derive(Clone, Copy)]
enum BufK {
    Emb,
    Rows,
    Hid,
}

/// FN100: a blocking H2D of `src` at row 0 of buffer `k` (the tails reclaimed first when it is
/// wider than the heads).
fn buf_h2d(rk: &mut Rank<'_, '_>, k: BufK, src: &[u8]) -> anyhow::Result<()> {
    let (row_b, name) = match k {
        BufK::Emb => (H * 2, "the embedding rows"),
        BufK::Rows => (H * 2, "the PLE rows"),
        BufK::Hid => (W * 2, "the seeded hidden"),
    };
    let rows = (src.len() as u64).div_ceil(row_b);
    buf_need(rk, u32::try_from(rows)?, name)?;
    let s = match k {
        BufK::Emb => rk.emb.clone(),
        BufK::Rows => rk.rows.clone(),
        BufK::Hid => rk.mtp.as_ref().context("mtp")?.hid.clone(),
    };
    ensure!(
        src.len() as u64 <= s.len,
        "FN100: {name}: {} B into a {} B buffer",
        src.len(),
        s.len
    );
    rk.hip
        .memcpy_h2d_at_blocking(&s.own, s.off, src, &rk.dev.stream)?;
    Ok(())
}

/// FN42b: THE BORROW LAW at a pass of `rows` rows (before its tables): a WIDE pass (> the narrow
/// rung) takes the arena's tail back -- the copy stream drained (no promotion into a lent slot in
/// flight; every earlier read of a lent slot is ahead of the pass on the compute stream), every
/// lent record flipped to its RAM shadow (no copy), the scratch cap widened, the ledger moved;
/// the tables this step builds point at the shadows. Any other pass lends the tail again (EMPTY
/// slots: the insert law fills them after this step's fence, which orders the copies behind the
/// wide chunk's work on the compute stream).
fn borrow_for(rk: &mut Rank<'_, '_>, em: &mut Emitter<'_>, rows: u32) -> anyhow::Result<()> {
    let Some(b) = rk.borrow.as_mut() else {
        return Ok(());
    };
    let wide = rows > MAX_ROWS;
    if wide == !b.lent {
        return Ok(());
    }
    // FN56: a deferred tier end resolves on the placements it was planned on
    tier_flush(rk)?;
    let Some(b) = rk.borrow.as_mut() else {
        return Ok(());
    };
    let t0 = Instant::now();
    if wide {
        rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
        // FN43: another client's slots (the staging ring's) may stay lent
        let before = rk.tier.tier.lent().0;
        let d = rk.tier.tier.reclaim(&b.ids);
        ensure!(
            rk.tier.tier.lent().0 + b.ids.len() as u32 == before,
            "a lent slot survived the reclaim"
        );
        rk.tier
            .tier
            .check_invariants()
            .map_err(|e| anyhow::anyhow!("after the reclaim: {e}"))?;
        b.flip_arena(em, false)?;
        b.lent = false;
        b.reclaims += 1;
        b.dropped += d as u64;
        b.reclaim_ms += t0.elapsed().as_secs_f64() * 1e3;
    } else {
        // every emission path starts at `new_layer` (a scratch reset), and nothing a pass
        // leaves in scratch is read past it (the wide chunk's pick + stream copy ran)
        rk.tier
            .tier
            .lend(&b.ids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        b.flip_arena(em, true)?;
        b.lent = true;
        b.lends += 1;
        b.lend_ms += t0.elapsed().as_secs_f64() * 1e3;
    }
    let budget = rk.budget.clone();
    b.relend(&budget)?;
    if wide && b.verify {
        let xt = rk.tier.mover.xtier();
        let mut stage = rk.hip.host_malloc(
            64 * xt
                .header()
                .stride
                .max(saber_mem::flashnext_budget::geometry::EXPERT_STRIDE),
        )?;
        verify_slots(rk, xt, &mut stage)?;
    }
    Ok(())
}

/// FN55c: the lazy tower of one rank (`--fn-vision lazy`): its region (the weight upload's
/// arena + the tower's arena, ledger class `fn.vision`) is the tier's lendable slots, LENT while
/// no tower is resident; [`vision_load`] reclaims it and uploads the tower into it,
/// [`vision_release`] lends it back.
struct LazyVis<'h> {
    w: std::rc::Rc<DevicePtr>,
    a: std::rc::Rc<DevicePtr>,
    /// The lendable slot ids (the weight region's whole strides, then the arena's).
    ids: Vec<u32>,
    /// The plugin's table while no tower holds it.
    table: Option<saber_exec::loader::FunctionTable<'h>>,
    pack: std::path::PathBuf,
    max_patches: u32,
    dump: Option<std::path::PathBuf>,
    /// The last tower run's number (a reload continues it).
    epoch: u64,
    ledger: saber_mem::ledger::Ledger,
    bytes: u64,
    lend_bytes: u64,
    lent: bool,
    loads: u64,
    releases: u64,
    load_ms: Vec<f64>,
    reclaim_ms: f64,
    release_ms: f64,
    dropped: u64,
    /// Released towers' (runs, image tokens, rows scattered, tower ms).
    prior: (u64, u64, u64, f64),
}

impl LazyVis<'_> {
    fn relend(&mut self, b: &saber_mem::flashnext_budget::Ep2Budget) -> anyhow::Result<()> {
        let lent = if self.lent { self.lend_bytes } else { 0 };
        b.relend_as(
            &mut self.ledger,
            super::vision::CLASS,
            self.bytes,
            lent,
            "xtier.vision_lent",
        )
        .map_err(|e| anyhow::anyhow!("the lazy tower's ledger: {e}"))
    }

    fn line(&self, rank: u32, resident: bool) -> String {
        let n = self.load_ms.len().max(1) as f64;
        format!(
            "fnvisionlazy rank{rank}: loads {} (ms {:?}, mean {:.1}; reclaim {:.1} ms total, {} \
             records flipped to RAM shadows) | releases {} ({:.1} ms total) | tower {} | region \
             {} slots {} | released towers: {} runs, {} image tokens, {} rows scattered",
            self.loads,
            self.load_ms
                .iter()
                .map(|x| (x * 10.0).round() / 10.0)
                .collect::<Vec<_>>(),
            self.load_ms.iter().sum::<f64>() / n,
            self.reclaim_ms,
            self.dropped,
            self.releases,
            self.release_ms,
            if resident { "RESIDENT" } else { "not resident" },
            self.ids.len(),
            if self.lent { "LENT" } else { "reclaimed" },
            self.prior.0,
            self.prior.1,
            self.prior.2
        )
    }

    fn render(&self, resident: bool) -> String {
        format!(
            "# TYPE saber_fn_vision_loads_total counter\nsaber_fn_vision_loads_total {}\n\
             # TYPE saber_fn_vision_releases_total counter\nsaber_fn_vision_releases_total {}\n\
             # TYPE saber_fn_vision_resident gauge\nsaber_fn_vision_resident {}\n\
             # TYPE saber_fn_vision_load_ms_last gauge\nsaber_fn_vision_load_ms_last {:.3}\n\
             # TYPE saber_fn_vision_lent_slots gauge\nsaber_fn_vision_lent_slots {}\n",
            self.loads,
            self.releases,
            u8::from(resident),
            self.load_ms.last().copied().unwrap_or(0.0),
            if self.lent { self.ids.len() } else { 0 }
        )
    }
}

/// FN55c: allocate the lazy tower's region (two allocations: the weight upload's arena and the
/// tower's arena, the ledger's `fn.vision` bytes), register each one's whole strides as the
/// tier's lendable slots on the copy queue, and LEND them (the tower is not resident at boot).
#[allow(clippy::too_many_arguments)]
fn lazy_boot<'h>(
    hip: &'h HipThread,
    rank: u32,
    pack: &Path,
    f: &FnFlags,
    table: saber_exec::loader::FunctionTable<'h>,
    tier: &mut ExpertTier,
    q: &mut HipCopyQueue<'_>,
    budget: &saber_mem::flashnext_budget::Ep2Budget,
    stride: u64,
) -> anyhow::Result<LazyVis<'h>> {
    let (wb, ab) = super::vision::region_bytes(pack, rank, f.vision_patches())?;
    let w = std::rc::Rc::new(hip.malloc(wb)?);
    let a = std::rc::Rc::new(hip.malloc(ab)?);
    let (nw, na) = (wb / stride, ab / stride);
    let addrs: Vec<u64> = (0..nw)
        .map(|i| w.addr().as_u64() + i * stride)
        .chain((0..na).map(|i| a.addr().as_u64() + i * stride))
        .collect();
    let first = tier
        .register_lendable(&addrs)
        .map_err(|e| anyhow::anyhow!("--fn-vision lazy: {e}"))?;
    q.set_lend(std::rc::Rc::clone(&w), 0, first);
    if na > 0 {
        q.set_lend(std::rc::Rc::clone(&a), 0, first + nw as u32);
    }
    let ids: Vec<u32> = (first..first + (nw + na) as u32).collect();
    tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ledger = saber_mem::ledger::Ledger::new();
    budget
        .declare(&mut ledger, 0)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    let mut lz = LazyVis {
        w,
        a,
        ids,
        table: Some(table),
        pack: pack.to_path_buf(),
        max_patches: f.vision_patches(),
        dump: f.fn_vision_dump.clone(),
        epoch: 0,
        ledger,
        bytes: wb + ab,
        lend_bytes: (nw + na) * stride,
        lent: true,
        loads: 0,
        releases: 0,
        load_ms: Vec::new(),
        reclaim_ms: 0.0,
        release_ms: 0.0,
        dropped: 0,
        prior: (0, 0, 0, 0.0),
    };
    lz.relend(budget)?;
    println!(
        "fnvisionlazy rank{rank}: THE LAZY TOWER (FN55c) -- region {:.1} MiB (weights {:.1} + \
         arena {:.1} MiB, {} patches; ledger fn.vision) = {} lendable slots (ids {first}..{}), \
         LENT at boot: the tier holds them until the first image; idle release after {} s",
        (wb + ab) as f64 / 1048576.0,
        wb as f64 / 1048576.0,
        ab as f64 / 1048576.0,
        f.vision_patches(),
        nw + na,
        first + (nw + na) as u32,
        f.fn_vision_idle_s
    );
    Ok(lz)
}

/// FN55c: THE LOAD -- the copy stream drained and the compute stream synced (every earlier read
/// of a lent slot ran), every lent record flipped to its RAM shadow (no copy), the ledger moved,
/// the prebuilt tables discarded (the next step's tables point at the shadows), then the tower
/// uploaded into the region from the pack's member. Both ranks run it in the same command.
fn vision_load(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    if rk.vis.is_some() || rk.lazy.is_none() {
        return Ok(());
    }
    let t0 = Instant::now();
    tier_flush(rk)?;
    rk.hip.sync_stream(&rk.dev.stream)?;
    rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    let lz = rk.lazy.as_mut().context("lazy")?;
    let before = rk.tier.tier.lent().0;
    let d = rk.tier.tier.reclaim(&lz.ids);
    ensure!(
        rk.tier.tier.lent().0 + lz.ids.len() as u32 == before,
        "a lent slot of the tower's region survived the reclaim"
    );
    rk.tier
        .tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the tower's reclaim: {e}"))?;
    rk.tier_pre = false;
    lz.lent = false;
    lz.dropped += d as u64;
    let budget = rk.budget.clone();
    lz.relend(&budget)?;
    let t_rec = t0.elapsed().as_secs_f64() * 1e3;
    lz.reclaim_ms += t_rec;
    let table = lz.table.take().context("the tower's table is held")?;
    let v = super::vision::FnVision::boot_in(
        rk.hip,
        table,
        &lz.pack,
        rk.rank,
        lz.max_patches,
        lz.dump.as_deref(),
        Some((std::rc::Rc::clone(&lz.w), std::rc::Rc::clone(&lz.a))),
        lz.epoch,
    )?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    lz.loads += 1;
    lz.load_ms.push(ms);
    println!(
        "fnvisionlazy rank{}: LOAD {} -- reclaim {t_rec:.1} ms ({d} records flipped to RAM \
         shadows), tower uploaded ({} records, {} B) in {:.1} ms; {ms:.1} ms in all",
        rk.rank,
        lz.loads,
        v.tier.records,
        v.tier.weight_bytes,
        v.tier.upload_s * 1e3
    );
    rk.vis = Some(v);
    Ok(())
}

/// FN55c: THE RELEASE -- the tower dropped (its sink drains first; every run and scatter was
/// synchronous), the region LENT back to the tier (EMPTY slots: the insert law and the fill on
/// lend refill them), the ledger moved, the prebuilt tables discarded.
fn vision_release(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    let Some(v) = rk.vis.take() else {
        return Ok(());
    };
    let t0 = Instant::now();
    let lz = rk
        .lazy
        .as_mut()
        .context("a release on a boot with no lazy tower")?;
    let (table, epoch, c) = v.release();
    rk.hip.sync_stream(&rk.dev.stream)?;
    lz.table = Some(table);
    lz.epoch = epoch;
    lz.prior = (
        lz.prior.0 + c.0,
        lz.prior.1 + c.1,
        lz.prior.2 + c.2,
        lz.prior.3 + c.3,
    );
    rk.tier
        .tier
        .lend(&lz.ids)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rk.tier_pre = false;
    lz.lent = true;
    lz.releases += 1;
    let budget = rk.budget.clone();
    lz.relend(&budget)?;
    lz.release_ms += t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "fnvisionlazy rank{}: RELEASE {} -- the region's {} slots LENT back to the tier",
        rk.rank,
        lz.releases,
        lz.ids.len()
    );
    Ok(())
}

/// FN55c: whether rank 0 releases the lazy tower now: it is resident, no image request for
/// `idle_s` seconds, and no image sequence is in the pools (an image prompt's chunks scatter the
/// tower's rows until its prompt is in).
fn lazy_due(rk: &Rank<'_, '_>, st: &SrvState, last_img: Instant, idle_s: u64) -> bool {
    rk.lazy.is_some()
        && rk.vis.is_some()
        && last_img.elapsed().as_secs() >= idle_s
        && st.slots.iter().flatten().all(|l| l.img.is_none())
}

/// FN42b: the borrow's end-of-run line (FN76b: + the arena top's).
fn borrow_line(rk: &Rank<'_, '_>) {
    if let Some(t) = rk.top.as_ref() {
        println!("{}", t.line(rk.rank));
    }
    if let Some(s) = rk.seql.as_ref() {
        println!("{}", s.line(rk.rank));
    }
    if let Some(b) = rk.buft.as_ref() {
        println!("{}", b.line(rk.rank));
    }
    if let Some(b) = rk.borrow.as_ref() {
        println!(
            "fnborrowend rank{}: reclaims {} ({:.3} ms total, {} records flipped to RAM shadows) | \
             lends {} ({:.3} ms total) | lent now {} of {} | fill on lend {}: {} slots filled at {} \
             prefill passes ({:.3} ms host)",
            rk.rank,
            b.reclaims,
            b.reclaim_ms,
            b.dropped,
            b.lends,
            b.lend_ms,
            rk.tier.tier.lent().0,
            rk.tier.tier.lent().1,
            if rk.lend_fill { "on" } else { "off" },
            rk.tier.tier.totals().1.lent_fills,
            rk.fill_passes,
            rk.fill_ms
        );
    }
}

/// FN51f-b: ELASTIC VRAM on one rank -- the controller (`saber_mem::elastic`), the tier's first
/// lendable id of the region, its ledger (the KV class less the lent bytes on `xtier.kv_elastic`)
/// and the counters.
struct ElasticRank {
    ctl: saber_mem::elastic::Elastic,
    first: u32,
    bb: u64,
    stride: u64,
    hot_bytes: u64,
    ledger: saber_mem::ledger::Ledger,
    /// KV blocks a lend evicted (no copy), records a reclaim flipped to their RAM shadows.
    evicted: u64,
    dropped: u64,
    lend_ms: f64,
    reclaim_ms: f64,
    t0: Instant,
    /// FN51f-c: `--fn-vram-elastic-refill eager` (the fill on lend), its slots filled and their
    /// host time (the copies run on the copy stream).
    eager: bool,
    refilled: u64,
    refill_ms: f64,
    /// FN130: the demand law's form (`--fn-vram-elastic-demand`) and its reclaims' host time.
    demand: ElasticDemand,
    early_ms: f64,
}

/// FN130: `--fn-vram-elastic-demand` as the worker runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElasticDemand {
    /// The demand law reclaims (a served boot's default).
    On,
    /// A shortfall at a demand site REFUSES loudly (the gates; every non-served boot).
    Strict,
    /// The control: no demand site (the pre-FN130 law; a reload past the cap is the tier's fatal).
    Off,
}

/// FN51f-b: the ledger line of the lent bytes (the referee refuses an imbalance).
fn elastic_relend(
    el: &mut ElasticRank,
    b: &saber_mem::flashnext_budget::Ep2Budget,
) -> anyhow::Result<()> {
    let lent = u64::from(el.ctl.lent * saber_mem::elastic::SEG_SLOTS) * el.stride;
    b.relend_as(
        &mut el.ledger,
        saber_mem::kv_tier::KV_HOT,
        el.hot_bytes,
        lent,
        "xtier.kv_elastic",
    )
    .map_err(|e| anyhow::anyhow!("the elastic region's ledger: {e}"))
}

/// FN51f-b: THE BOOT of the elastic region: the geometry from the KV plan (pages of the dtype,
/// segments of 4 expert slots, the chunk floor kept), every segment's slots registered as the
/// tier's lendable slots (one copy-queue region a segment: segment `i` sits below segment
/// `i - 1`), the idle pool lent down to the band, the lent slots filled hottest-first.
#[allow(clippy::too_many_arguments)]
fn elastic_boot(
    hip: &HipThread,
    rank: u32,
    f: &FnFlags,
    stream: &saber_hip::Stream,
    kc: &std::cell::RefCell<super::kvwire::KvDev>,
    tier: &mut super::run::TierRank<'_, '_>,
    budget: &saber_mem::flashnext_budget::Ep2Budget,
    static_slots: u32,
    stride: u64,
) -> anyhow::Result<ElasticRank> {
    use saber_mem::elastic::{Elastic, ElasticGeom, SEG_SLOTS};
    ensure!(
        static_slots >= f.fn_vram_elastic_min_slots,
        "--fn-vram-elastic: the carve leaves {static_slots} static expert slots, under \
         --fn-vram-elastic-min-slots {} (lower --fn-vram-elastic-kv-max-gib)",
        f.fn_vram_elastic_min_slots
    );
    let mut k = kc.borrow_mut();
    let bb = k.bb;
    let total = k.tier.plan().hot_slots;
    let keep = u32::try_from(k.tier.hot_floor()).context("the KV floor")?;
    let geom = ElasticGeom::new(total, bb, stride, keep);
    // a slot = whole pages (2,617,344 B of experts in 160 x 16,384 B / 320 x 8,192 B): the
    // queue's region law (`offset + i x stride`) does not hold across a slot's page padding, so
    // every slot is its own region (`slot_at` binary-searches them)
    let slot_bytes = u64::from(geom.seg_pages / SEG_SLOTS) * bb;
    ensure!(
        slot_bytes >= stride,
        "the elastic region's slot ({slot_bytes} B of {bb} B pages) < the expert stride {stride}"
    );
    ensure!(
        geom.segs > 0,
        "--fn-vram-elastic: the hot pool holds no segment past its floor"
    );
    let (hot, off) = k.hot_region();
    let base = hot.addr().as_u64() + off;
    let addrs: Vec<u64> = (0..geom.segs)
        .flat_map(|i| (0..SEG_SLOTS).map(move |s| (i, s)))
        .map(|(i, s)| base + geom.slot_offset(i, s, bb))
        .collect();
    let first = tier
        .tier
        .register_lendable(&addrs)
        .map_err(|e| anyhow::anyhow!("--fn-vram-elastic: {e}"))?;
    for i in 0..geom.segs {
        for s in 0..SEG_SLOTS {
            tier.mover.queue_mut().set_lend(
                std::rc::Rc::clone(&hot),
                off + geom.slot_offset(i, s, bb),
                first + i * SEG_SLOTS + s,
            );
        }
    }
    let need = k.tier.hot_floor().max(k.tier.hot_want());
    let n = Elastic::boot_lent(&geom, need);
    let ctl = Elastic::new(geom, n);
    let evicted = k
        .tier
        .shrink_hot(geom.cap(n))
        .map_err(|e| anyhow::anyhow!("the boot lend: {e}"))?;
    for i in 0..n {
        ctl.lend_alias_check(i, &k.tier)
            .map_err(|e| anyhow::anyhow!("the boot lend: {e}"))?;
    }
    let ids: Vec<u32> = (0..n).flat_map(|i| Elastic::seg_ids(first, i)).collect();
    tier.tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ledger = saber_mem::ledger::Ledger::new();
    budget
        .declare(&mut ledger, 0)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    let mut el = ElasticRank {
        ctl,
        first,
        bb,
        stride,
        // FN68d: the `kv.hot` class = both tiers (the MTP twin is never lent)
        hot_bytes: k.geom.hot_alloc_all(),
        ledger,
        evicted: u64::from(evicted),
        dropped: 0,
        lend_ms: 0.0,
        reclaim_ms: 0.0,
        t0: Instant::now(),
        eager: f.fn_vram_elastic_refill == "eager",
        refilled: 0,
        refill_ms: 0.0,
        demand: match super::run::elastic_demand_form(f)? {
            super::run::ElasticDemandForm::Off => ElasticDemand::Off,
            super::run::ElasticDemandForm::On if !super::run::elastic_demand_strict(f) => {
                ElasticDemand::On
            }
            _ => ElasticDemand::Strict,
        },
        early_ms: 0.0,
    };
    elastic_relend(&mut el, budget)?;
    drop(k);
    // the fill: the copy stream after every earlier use of the pages (the boot's zeroing)
    let t = Instant::now();
    tier.mover
        .after(stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let filled = tier
        .tier
        .fill_lent(&mut tier.mover)
        .map_err(|e| anyhow::anyhow!("the elastic boot fill: {e}"))?;
    tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    hip.sync_stream(stream)?;
    tier.tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the elastic boot: {e}"))?;
    let lent: Vec<u32> = (first..first + geom.segs * SEG_SLOTS)
        .filter(|&s| tier.tier.is_lent(s))
        .collect();
    el.ctl
        .owner_law(geom.cap(n), first, &lent)
        .map_err(|e| anyhow::anyhow!("the elastic owner law at boot: {e}"))?;
    println!(
        "fnelastic rank{rank}: REGION {:.3} GiB = {} pages of {bb} B (boot window + {:.2} GiB \
         borrow; ledger {} + xtier.kv_elastic) | {} segments of {SEG_SLOTS} slots ({} pages, {} B \
         a slot) past a {} page floor | lendable ids {first}..{} | static expert slots {} (floor \
         {}) | BOOT LEND {n} segments = {} slots ({:.3} GiB), {} filled in {:.1} ms",
        el.hot_bytes as f64 / f64::from(1u32 << 30),
        geom.total,
        f.fn_vram_elastic_kv_max_gib,
        saber_mem::kv_tier::KV_HOT,
        geom.segs,
        geom.seg_pages,
        slot_bytes,
        keep,
        first + geom.segs * SEG_SLOTS,
        static_slots,
        f.fn_vram_elastic_min_slots,
        n * SEG_SLOTS,
        u64::from(n * SEG_SLOTS) as f64 * stride as f64 / f64::from(1u32 << 30),
        filled,
        t.elapsed().as_secs_f64() * 1e3
    );
    println!(
        "{}",
        el.ctl.line(
            rank,
            "boot",
            -i64::from(n * geom.seg_pages),
            i64::from(n * SEG_SLOTS)
        )
    );
    Ok(el)
}

/// FN51f-b: THE TICK (before a target step's KV step law: a prefill chunk's, a decode step's)
/// on the `borrow_for` pattern. `admit` = a harness sequence's first chunk (its slot and end:
/// the admission the step law would take, taken here first so the law sees its reservation; the
/// serve plane's `Cmd::Zero` already reserved). A reclaim: the deferred tier end lands, the copy
/// stream drains (no promotion into a reclaimed slot in flight; the compute stream's earlier
/// reads of the slots are ahead of any KV write by stream order), the lent records fall back to
/// their RAM shadows, the KV's cap grows. A lend: the KV gives up its top segment (out-of-window
/// blocks evicted with no copy), the alias check, the tier takes the slots EMPTY (the insert law
/// fills them after a tier end's fence). Both moves rebuild the expert tables.
fn elastic_tick(rk: &mut Rank<'_, '_>, admit: Option<(u32, u32)>) -> anyhow::Result<()> {
    use saber_mem::elastic::Move;
    let (Some(el), Some(kc)) = (rk.elastic.as_mut(), rk.pools.kvt.as_ref()) else {
        return Ok(());
    };
    let (m, need) = {
        let mut k = kc.borrow_mut();
        if let Some((s, pend)) = admit {
            if !k.tier.is_live(s) {
                let t = k.geom.first_tokens(pend);
                k.tier
                    .admit(s, t)
                    .map_err(|e| anyhow::anyhow!("kv tier: {e}"))?;
            }
        }
        let need = el.ctl.need(k.tier.hot_floor(), k.tier.hot_want());
        let (lent, segs, geom) = (el.ctl.lent, el.ctl.geom.segs, el.ctl.geom);
        let m = el.ctl.tick(need, || {
            lent < segs && k.tier.can_shrink_hot(geom.cap(lent + 1))
        });
        (m, need)
    };
    if m == Move::Hold {
        return Ok(());
    }
    elastic_apply(rk, m, need, None)
}

/// FN130: THE DEMAND LAW (`--fn-vram-elastic-demand`, default `on`) -- before KV pages are opened
/// for `seqs` (`(pool slot, first position, end position)`, as `KvDev::begin`) at a site the step
/// law's tick may not cover (`site`): the prompt cache / prefix tree's RELOAD of a stored piece
/// (it opens a whole piece's pages with no prefill chunk and so no tick: FN128 h1 s4_A / h3, two
/// stored 8,192-token pieces reloaded back to back past the boot cap of 9,580 pages = rank 1's
/// FATAL `no vram slot for a block that must be resident`), and every target prefill chunk after
/// its tick. The need is the controller's (`need(floor, want)`, the live reservations') and the
/// step's exact shortfall (`KvDev::shortfall`); a positive one reclaims the segments at once, the
/// request waiting inline for the flip (< 1 ms; `fnelastic rank<r>: EARLY RECLAIM`, counted). Both
/// ranks hold the same KV bookkeeping, so both take the same move with no exchange. `strict` (and
/// every non-served boot) REFUSES a reclaim the demand would need, loudly, before any tier state
/// moves; `off` = the control (the pre-FN130 law).
pub(super) fn elastic_demand(
    rk: &mut Rank<'_, '_>,
    seqs: &[(u32, u32, u32)],
    site: &'static str,
) -> anyhow::Result<()> {
    use saber_mem::elastic::Move;
    let (Some(el), Some(kc)) = (rk.elastic.as_mut(), rk.pools.kvt.as_ref()) else {
        return Ok(());
    };
    if el.demand == ElasticDemand::Off {
        return Ok(());
    }
    let (need, short) = {
        let k = kc.borrow();
        (
            el.ctl.need(k.tier.hot_floor(), k.tier.hot_want()),
            k.shortfall(seqs),
        )
    };
    let ticks = el.ctl.counts.ticks;
    let m = el.ctl.demand(need, short);
    let Move::Reclaim(n) = m else {
        return Ok(());
    };
    if el.demand == ElasticDemand::Strict {
        anyhow::bail!(
            "kv tier: FN130 strict (--fn-vram-elastic-demand strict or a non-served boot): {site} \
             needs {n} lent segment(s) back before its pages open (need {need} pages, short \
             {short}, KV cap {} of {}, ticks {ticks}) -- a served boot reclaims them on demand",
            el.ctl.geom.cap(el.ctl.lent),
            el.ctl.geom.total
        );
    }
    elastic_apply(rk, m, need, Some((site, short)))
}

/// FN51f-b / FN130: apply one controller move (`demand`: the site of a DEMAND reclaim, FN130 --
/// counted as an early reclaim, its line `fnelastic rank<r>: EARLY RECLAIM (<site>)`).
fn elastic_apply(
    rk: &mut Rank<'_, '_>,
    m: saber_mem::elastic::Move,
    need: u64,
    demand: Option<(&'static str, u64)>,
) -> anyhow::Result<()> {
    use saber_mem::elastic::{Elastic, Move, SEG_SLOTS};
    tier_flush(rk)?;
    rk.tier_pre = false;
    let t0 = Instant::now();
    let rank = rk.rank;
    let el = rk.elastic.as_mut().context("elastic")?;
    let kc = rk.pools.kvt.as_ref().context("kv")?;
    let geom = el.ctl.geom;
    let lent = el.ctl.lent;
    let (dp, ds, what) = match m {
        Move::Hold => return Ok(()),
        Move::Reclaim(n) => {
            rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
            let ids: Vec<u32> = (lent - n..lent)
                .flat_map(|i| Elastic::seg_ids(el.first, i))
                .collect();
            let d = rk.tier.tier.reclaim(&ids);
            el.dropped += d as u64;
            kc.borrow_mut()
                .tier
                .grow_hot(geom.cap(lent - n))
                .map_err(|e| anyhow::anyhow!("the reclaim: {e}"))?;
            (
                i64::from(n * geom.seg_pages),
                -i64::from(n * SEG_SLOTS),
                "reclaim",
            )
        }
        Move::Lend => {
            {
                let mut k = kc.borrow_mut();
                let ev = k
                    .tier
                    .shrink_hot(geom.cap(lent + 1))
                    .map_err(|e| anyhow::anyhow!("the lend: {e}"))?;
                el.evicted += u64::from(ev);
                el.ctl
                    .lend_alias_check(lent, &k.tier)
                    .map_err(|e| anyhow::anyhow!("the lend: {e}"))?;
            }
            let ids = Elastic::seg_ids(el.first, lent);
            rk.tier
                .tier
                .lend(&ids)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // FN51f-c: THE FILL ON LEND -- the segment's slots take the hottest RAM records now:
            // the copy stream after every earlier use of the pages (the compute stream's
            // attention reads and the KV's flushes of the evicted blocks run on it), one H2D a
            // slot, each layer's gate waits for its copies (law 3); one segment a tick bounds it
            if el.eager && rk.tier.tier.complete() {
                let tf = Instant::now();
                let t = &mut rk.tier;
                t.mover
                    .after(&rk.dev.stream)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let n = t
                    .tier
                    .fill_lent_hot(&ids, &mut t.mover)
                    .map_err(|e| anyhow::anyhow!("the elastic refill: {e}"))?;
                el.refilled += n as u64;
                el.refill_ms += tf.elapsed().as_secs_f64() * 1e3;
            }
            (-i64::from(geom.seg_pages), i64::from(SEG_SLOTS), "lend")
        }
    };
    if demand.is_some() {
        el.ctl.applied_demand(m);
    } else {
        el.ctl.applied(m);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    if demand.is_some() {
        el.early_ms += ms;
    }
    if what == "lend" {
        el.lend_ms += ms;
    } else {
        el.reclaim_ms += ms;
    }
    rk.tier
        .tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the elastic {what}: {e}"))?;
    let lent_ids: Vec<u32> = (el.first..el.first + geom.segs * SEG_SLOTS)
        .filter(|&s| rk.tier.tier.is_lent(s))
        .collect();
    let cap = kc.borrow().tier.hot_cap();
    el.ctl
        .owner_law(cap, el.first, &lent_ids)
        .map_err(|e| anyhow::anyhow!("the elastic owner law: {e}"))?;
    let budget = rk.budget.clone();
    elastic_relend(el, &budget)?;
    let what = match demand {
        Some((site, short)) => format!("EARLY RECLAIM ({site}, short {short} pages)"),
        None => what.to_string(),
    };
    println!(
        "{} | need {need} | t {:.1} s | {:.3} ms | lent empty {}",
        el.ctl.line(rank, &what, dp, ds),
        el.t0.elapsed().as_secs_f64(),
        ms,
        rk.tier.tier.lent_empty()
    );
    Ok(())
}

/// FN51f-b: the end-of-run lines -- the elastic region's (`fnelasticend`) and the residency
/// view (`fnplane`, one a layer; `--fn-plane-dump DIR` writes `plane-rank<r>.json`).
fn plane_end(rk: &Rank<'_, '_>) -> anyhow::Result<()> {
    if let Some(el) = rk.elastic.as_ref() {
        let c = el.ctl.counts;
        println!(
            "fnelasticend rank{}: lent {} of {} segments now (min {} max {}) | ticks {} lends {} \
             ({:.3} ms, {} KV blocks evicted with no copy) reclaims {} ({} segments, {:.3} ms, {} \
             records to RAM shadows) blocked {} | refill {} ({} slots filled on lend, {:.3} ms) \
             | demand {} (FN130: {} checks, {} early reclaims of {} segments, {:.3} ms)",
            rk.rank,
            el.ctl.lent,
            el.ctl.geom.segs,
            c.min_lent,
            c.max_lent,
            c.ticks,
            c.lends,
            el.lend_ms,
            el.evicted,
            c.reclaims,
            c.reclaimed_segs,
            el.reclaim_ms,
            el.dropped,
            c.blocked,
            if el.eager { "eager" } else { "lazy" },
            el.refilled,
            el.refill_ms,
            match el.demand {
                ElasticDemand::On => "on",
                ElasticDemand::Strict => "strict",
                ElasticDemand::Off => "off",
            },
            c.demands,
            c.early_reclaims,
            c.early_segs,
            el.early_ms
        );
    }
    for l in rk.tier.tier.plane_lines() {
        println!("{l}");
    }
    if let Some(d) = rk.plane_dump.as_ref() {
        std::fs::create_dir_all(d)?;
        let p = d.join(format!("plane-rank{}.json", rk.rank));
        std::fs::write(&p, rk.tier.tier.plane_json())
            .with_context(|| format!("--fn-plane-dump {}", p.display()))?;
        println!("fnplane rank{}: dumped {}", rk.rank, p.display());
    }
    Ok(())
}

/// FN51f-b: the elastic region's `/metrics` family (`saber_xtier_elastic_*`) + the tiered KV's
/// RAM-read counters (`saber_fn_kv_*`), rank 0.
fn elastic_metrics(rk: &Rank<'_, '_>) -> String {
    let mut o = format!(
        "# TYPE saber_xtier_steps_total counter\nsaber_xtier_steps_total{{rank=\"{}\"}} {}\n",
        rk.rank,
        rk.tier.tier.steps()
    );
    if let Some(el) = rk.elastic.as_ref() {
        o.push_str(&el.ctl.render_prometheus(rk.rank, el.bb));
    }
    if let Some(k) = rk.pools.kvt.as_ref() {
        o.push_str(&k.borrow().render_prometheus(rk.rank));
    }
    o
}

/// FN43: the prefill staging ring of one rank (see `saber_mem::stage_ring`).
struct Ring {
    law: crate::xtier_flags::StageLaw,
    shape: saber_mem::stage_ring::RingShape,
    /// The ring's memory (`buffers x slots` strides; the tier's lendable slots `ids`).
    mem: std::rc::Rc<DevicePtr>,
    ids: Vec<u32>,
    /// The slots are lent to the tier (no staged pass owns them).
    lent: bool,
    /// Per layer: the arrival mark (copy stream); per buffer: its layer ran its experts (compute).
    arrived: Vec<saber_hip::Event>,
    consumed: Vec<saber_hip::Event>,
    /// The pass's first copy / last arrival (timing, copy stream).
    t0: saber_hip::Event,
    t1: saber_hip::Event,
    /// The current staged pass's plan (None between passes).
    plan: Option<saber_mem::stage_ring::StagePlan>,
    /// LR-H (`--xtier-stage-order`): the order key the ring stages by, and the OTHER key's plan
    /// at the same placements (host-only, never issued: the counterfactual the pass line prices).
    key: saber_mem::stage_ring::StageKey,
    shadow: Option<saber_mem::stage_ring::StagePlan>,
    /// The current pass's which-path counts (tier_end) and the run's.
    pass: saber_mem::stage_ring::StageCounts,
    total: saber_mem::stage_ring::StageCounts,
    /// LR-H: the other key's counts (the shadow plan's), this pass and the run's.
    pass_alt: saber_mem::stage_ring::StageCounts,
    total_alt: saber_mem::stage_ring::StageCounts,
    ledger: saber_mem::ledger::Ledger,
    bytes: u64,
    stride: u64,
    passes: u64,
    /// Passes of >= the law's rows that ran UVA (an incomplete tier: the demand path).
    incomplete: u64,
    reclaims: u64,
    lends: u64,
    dropped: u64,
    reclaim_ms: f64,
    span_ms: f64,
    /// LR-E (`--fn-ring-idle`, off = None): per layer, timing events on the compute stream right
    /// before and right after its wait for the layer's arrivals; the run's summed wait.
    idle: Option<Vec<(saber_hip::Event, saber_hip::Event)>>,
    idle_ms: f64,
}

/// LR-E: a staged pass's compute-stream wait from its per-layer waits (ms): (the sum, the worst
/// layer, its wait). Negative event noise counts as 0.
fn ring_idle_summary(waits: &[f32]) -> (f64, usize, f64) {
    let mut sum = 0.0;
    let (mut at, mut max) = (0usize, 0.0f64);
    for (l, &w) in waits.iter().enumerate() {
        let w = f64::from(w).max(0.0);
        sum += w;
        if w > max {
            (at, max) = (l, w);
        }
    }
    (sum, at, max)
}

/// LR-H: the `fnringpass` / `fnringend` suffix that prices BOTH order keys: the staged key's
/// `misordered` (per layer `min(staged unrouted, uva)`) and, when the pass kept the other key's
/// plan, that plan's counts at the same routes and placements (host-only, never issued).
fn ring_order_suffix(
    key: saber_mem::stage_ring::StageKey,
    c: &saber_mem::stage_ring::StageCounts,
    alt: Option<saber_mem::stage_ring::StageCounts>,
) -> String {
    let mut o = format!(" | order {}: misordered {}", key.name(), c.misordered);
    if let Some(a) = alt {
        o += &format!(
            " | {} (counterfactual): ring {} uva {} staged unrouted {} misordered {}",
            key.other().name(),
            a.ring,
            a.uva,
            a.wasted,
            a.misordered
        );
    }
    o
}

/// LR-E: the `fnringpass` suffix of an armed pass (`span` = its copy span, ms).
fn ring_idle_suffix(waits: &[f32], span: f64) -> String {
    let (sum, at, max) = ring_idle_summary(waits);
    format!(
        " | compute wait {sum:.2} ms ({:.1} % of the span) worst layer {at} {max:.2} ms",
        100.0 * sum / span.max(1e-9)
    )
}

impl Ring {
    fn relend(&mut self, b: &saber_mem::flashnext_budget::Ep2Budget) -> anyhow::Result<()> {
        let lent = if self.lent { self.bytes } else { 0 };
        b.relend_as(
            &mut self.ledger,
            "xtier.stage_ring",
            self.bytes,
            lent,
            "xtier.ring_lent",
        )
        .map_err(|e| anyhow::anyhow!("the ring's ledger: {e}"))
    }
}

/// FN76: the pool's slots (the imported BO keeps the pool; the queue's region holds the Rc).
struct PoolSlots {
    _mem: std::rc::Rc<DevicePtr>,
    ids: Vec<u32>,
}

/// FN76: THE POOL AS SLOTS (`--xtier-pool-slots auto|on`) -- after the boot fill dropped its
/// reader, open the card's whole P2PDMA pool as one BO, import its dma-buf into HIP (copies and
/// kernels address it as device memory), register `(bytes - reserve) / stride` whole strides as
/// the tier's lendable slots on the copy queue, LEND them and fill them hottest-first (the
/// elastic boot's fill). `None` = refused under `auto` (the reason printed). The pool's bytes are
/// the ledger's `kernel.p2pmem` class already: no ledger line moves.
/// FN76b: register the narrow carve's scratch past the boundary as the tier's lendable slots and
/// lend them (the boot line `fnlend rank<r>:`). The boundary = the largest scratch a pass of at
/// most `--xtier-lend-arena-top-rows` rows takes on the host ([`super::run::arena_top_need`]) +
/// the margin, 2 MiB aligned. Needs the two-layout borrow (or none: a run whose widest chunk fits
/// the narrow rung) and refuses the forms whose passes it does not model.
#[allow(clippy::too_many_arguments)]
fn top_boot(
    rank: u32,
    f: &FnFlags,
    prompts: &[Prompt],
    decls: &super::kern::Decls,
    rules: &saber_kernels::flashnext::FlashNextRules,
    dr: &plan::DryRun,
    borrow: Option<&Borrow>,
    dev: &RankDev<'_>,
    tier: &mut ExpertTier,
    q: &mut HipCopyQueue<'_>,
    budget: &saber_mem::flashnext_budget::Ep2Budget,
    stride: u64,
) -> anyhow::Result<Option<Top>> {
    // the forms it does not model: `on` refuses, `auto` (the default) stands down by name
    let must = f.xtier.lend_arena_top().1;
    let why = if f.fn_borrow_cut {
        Some("--fn-borrow-cut (it runs under the two-layout borrow)")
    } else if f.fn_mixed {
        Some("--fn-mixed (the packed mixed step is not modelled)")
    } else if f.fn_control.as_deref() == Some("mtp-wide-capture") {
        Some("the drafter's wide-capture control (it keeps the arena it finds)")
    } else if f.fn_one_process || f.fn_poison.is_some() || f.fn_poison_step.is_some() {
        Some("the one-process form / the poison arms")
    } else if borrow.is_some_and(|b| b.layouts.is_none()) {
        Some("the borrow's cut layout")
    } else {
        None
    };
    if let Some(why) = why {
        ensure!(!must, "--xtier-lend-arena-top on: refused under {why}");
        println!("fnlend rank{rank}: stands down -- {why}");
        return Ok(None);
    }
    let t = Instant::now();
    let rows = f.xtier.xtier_lend_arena_top_rows;
    let pctx = super::run::pools_ctx(f, prompts);
    let (dec, pf, mp) = super::run::arena_top_need(decls, rules, f, pctx, rows)?;
    let need = dec.max(pf).max(mp) + (f.xtier.xtier_lend_arena_top_margin_mib << 20);
    let cap = need.div_ceil(BORROW_ALIGN) * BORROW_ALIGN;
    // the carve the top lives in: the borrow's narrow arena, else the boot's own
    let scratch = match borrow {
        Some(b) => {
            let (_, nw, _) = b
                .layouts
                .ok_or_else(|| anyhow::anyhow!("the arena top needs the two-layout borrow"))?;
            nw.scratch
        }
        None => dr.sizes.scratch,
    };
    let n = scratch.saturating_sub(cap) / stride;
    if n == 0 {
        // a run whose arena is no wider than the boundary (a short harness prompt set): inert
        println!(
            "fnlend rank{rank}: INERT -- the boundary {:.1} MiB leaves no slot in the {:.1} MiB \
             scratch (dec {:.1}, a {rows}-row chunk {:.1}, its drafter pass {:.1} MiB)",
            cap as f64 / 1048576.0,
            scratch as f64 / 1048576.0,
            dec as f64 / 1048576.0,
            pf as f64 / 1048576.0,
            mp as f64 / 1048576.0
        );
        return Ok(None);
    }
    let base = dev.base() + cap;
    let addrs: Vec<u64> = (0..n).map(|i| base + i * stride).collect();
    let first = tier
        .register_lendable(&addrs)
        .map_err(|e| anyhow::anyhow!("--xtier-lend-arena-top: {e}"))?;
    q.set_lend(std::rc::Rc::clone(&dev.arena), cap, first);
    let ids: Vec<u32> = (first..first + n as u32).collect();
    tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ledger = saber_mem::ledger::Ledger::new();
    budget
        .declare(&mut ledger, 0)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    let mut top = Top {
        ids,
        cap,
        full_cap: scratch,
        rows,
        lent: true,
        bytes: n * stride,
        arena_bytes: dr.sizes.total(),
        ledger,
        lends: 0,
        reclaims: 0,
        dropped: 0,
        drop_max: 0,
        reclaim_ms: 0.0,
        lend_ms: 0.0,
        high_dec: 0,
        high_pf: 0,
        last: None,
    };
    top.relend(budget)?;
    let mib = |x: u64| x as f64 / 1048576.0;
    println!(
        "fnlend rank{rank}: THE ARENA TOP (FN76b) -- boundary {:.1} MiB = max(decode forms {:.1}, a \
         {rows}-row prefill chunk {:.1}, its drafter pass {:.1}) + margin {} MiB, 2 MiB aligned, of \
         the {:.1} MiB scratch ({}) = {n} lendable slots (ids {first}..{}), LENT at boot: a prefill \
         pass of > {rows} rows reclaims them (records dropped to their RAM shadows); host {:.0} ms",
        mib(cap),
        mib(dec),
        mib(pf),
        mib(mp),
        f.xtier.xtier_lend_arena_top_margin_mib,
        mib(scratch),
        if borrow.is_some() { "the borrow's narrow arena" } else { "the boot's arena" },
        first + n as u32,
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(Some(top))
}

fn pool_slots_boot(
    hip: &HipThread,
    rank: u32,
    x: &crate::xtier_flags::XtierFlags,
    tier: &mut super::run::TierRank<'_, '_>,
    stream: &saber_hip::Stream,
    stride: u64,
    excl: u64,
) -> anyhow::Result<Option<PoolSlots>> {
    let must = x.pool_slots().1;
    let bdfid = crate::fn63_hsa::hsa_bdfid(hip)?;
    let t = Instant::now();
    let bo = match saber_hsa::pool_bo(bdfid, &saber_hsa::PoolAsk::Auto) {
        Ok(bo) => bo,
        Err(why) => {
            ensure!(
                !must,
                "rank {rank}: --xtier-pool-slots on: the pool is refused ({why})"
            );
            println!("fnpoolslots rank{rank}: the P2PDMA pool is refused ({why}) -- no pool slots");
            return Ok(None);
        }
    };
    let reserve = x.xtier_pool_slots_reserve_mib << 20;
    let n = bo.bytes.saturating_sub(reserve) / stride;
    if n == 0 {
        ensure!(
            !must,
            "rank {rank}: --xtier-pool-slots on: the pool holds no slot"
        );
        println!(
            "fnpoolslots rank{rank}: the pool ({} MiB) holds no slot past the reserve",
            bo.bytes >> 20
        );
        return Ok(None);
    }
    let fd = bo.dmabuf.try_clone().context("dup of the pool's dma-buf")?;
    let mem = std::rc::Rc::new(
        hip.import_dmabuf(std::os::fd::IntoRawFd::into_raw_fd(fd), bo.bytes)
            .context("hipImportExternalMemory of the pool's dma-buf")?,
    );
    drop(bo);
    let base = mem.addr().as_u64();
    let addrs: Vec<u64> = (0..n).map(|i| base + i * stride).collect();
    let first = tier
        .tier
        .register_lendable(&addrs)
        .map_err(|e| anyhow::anyhow!("--xtier-pool-slots: {e}"))?;
    tier.mover
        .queue_mut()
        .set_lend(std::rc::Rc::clone(&mem), 0, first);
    let ids: Vec<u32> = (first..first + n as u32).collect();
    tier.tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t_fill = Instant::now();
    tier.mover
        .after(stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let filled = tier
        .tier
        .fill_lent(&mut tier.mover)
        .map_err(|e| anyhow::anyhow!("the pool slots' fill: {e}"))?;
    tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    hip.sync_stream(stream)?;
    // FN107 (X1): THE POOL SLOTS EXCLUSIVE -- the filled records drop their RAM shadows and the
    // freed RAM slots take the records the smaller L2 left on the drive (one NVMe -> RAM batch):
    // every record is again in VRAM or RAM, and a pool slot's victim is written back from now on
    let mut excl_line = String::new();
    if excl > 0 {
        let t_x = Instant::now();
        let dropped = tier
            .tier
            .make_exclusive(&ids)
            .map_err(|e| anyhow::anyhow!("--xtier-pool-exclusive: {e}"))?;
        let read = tier
            .tier
            .refill_ram(&mut tier.mover)
            .map_err(|e| anyhow::anyhow!("--xtier-pool-exclusive: the RAM refill: {e}"))?;
        let fetch = tier.tier.occupancy().2;
        excl_line = format!(
            "; EXCLUSIVE (FN107 X1, planned {excl}): {dropped} RAM shadows dropped, {read} drive \
             records read into the freed RAM slots in {:.1} ms, {fetch} records on the drive{}",
            t_x.elapsed().as_secs_f64() * 1e3,
            // FN108b: an incomplete tier runs EVERY decode / verify step on the demand path
            if fetch > 0 {
                " -- INCOMPLETE TIER: every step takes the demand path (FN108b)"
            } else {
                ""
            }
        );
    }
    tier.tier
        .check_invariants()
        .map_err(|e| anyhow::anyhow!("after the pool slots' lend: {e}"))?;
    println!(
        "fnpoolslots rank{rank}: THE POOL AS SLOTS (FN76) -- the P2PDMA pool {} MiB imported at \
         {base:#x} (reserve {} MiB) = {n} lendable slots (ids {first}..{}), LENT for the run, {filled} \
         filled in {:.1} ms (open + import {:.1} ms); ledger kernel.p2pmem unchanged{excl_line}",
        mem.len() >> 20,
        reserve >> 20,
        first + n as u32,
        t_fill.elapsed().as_secs_f64() * 1e3,
        (t_fill - t).as_secs_f64() * 1e3
    );
    Ok(Some(PoolSlots { _mem: mem, ids }))
}

/// FN43: allocate the ring (its bytes are the budget's `xtier.stage_ring` line), register it as
/// the tier's lendable slots on the copy queue, and lend it (decode keeps the slots).
#[allow(clippy::too_many_arguments)]
fn ring_boot(
    hip: &HipThread,
    rank: u32,
    law: crate::xtier_flags::StageLaw,
    tier: &mut ExpertTier,
    q: &mut HipCopyQueue<'_>,
    budget: &saber_mem::flashnext_budget::Ep2Budget,
    stride: u64,
    idle: bool,
    key: saber_mem::stage_ring::StageKey,
) -> anyhow::Result<Ring> {
    let shape = saber_mem::stage_ring::RingShape {
        buffers: law.buffers,
        slots: law.slots,
    };
    let n = u64::from(shape.total());
    ensure!(
        budget.stage_slots == n,
        "the ledger's ring ({} slots) != the law's {n}",
        budget.stage_slots
    );
    let mem = std::rc::Rc::new(hip.malloc(n * stride)?);
    let base = mem.addr().as_u64();
    let addrs: Vec<u64> = (0..n).map(|i| base + i * stride).collect();
    let first = tier
        .register_lendable(&addrs)
        .map_err(|e| anyhow::anyhow!("--xtier-stage: {e}"))?;
    q.set_lend(std::rc::Rc::clone(&mem), 0, first);
    let ids: Vec<u32> = (first..first + n as u32).collect();
    tier.lend(&ids).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ledger = saber_mem::ledger::Ledger::new();
    budget
        .declare(&mut ledger, 0)
        .map_err(|e| anyhow::anyhow!("rank {rank}: ledger: {e}"))?;
    let ev = |t: bool| {
        hip.create_event(if t {
            EventFlags::Default
        } else {
            EventFlags::DisableTiming
        })
    };
    let mut r = Ring {
        law,
        shape,
        mem,
        ids,
        lent: true,
        arrived: (0..LAYERS).map(|_| ev(false)).collect::<Result<_, _>>()?,
        consumed: (0..shape.buffers)
            .map(|_| ev(false))
            .collect::<Result<_, _>>()?,
        t0: ev(true)?,
        t1: ev(true)?,
        plan: None,
        key,
        shadow: None,
        pass: Default::default(),
        total: Default::default(),
        pass_alt: Default::default(),
        total_alt: Default::default(),
        ledger,
        bytes: n * stride,
        stride,
        passes: 0,
        incomplete: 0,
        reclaims: 0,
        lends: 0,
        dropped: 0,
        reclaim_ms: 0.0,
        span_ms: 0.0,
        idle: if idle {
            Some(
                (0..LAYERS)
                    .map(|_| Ok((ev(true)?, ev(true)?)))
                    .collect::<anyhow::Result<_>>()?,
            )
        } else {
            None
        },
        idle_ms: 0.0,
    };
    r.relend(budget)?;
    println!(
        "fnring rank{rank}: THE PREFILL STAGING RING -- {} x {} slots ({:.1} MiB, ledger \
         xtier.stage_ring), lendable ids {first}..{}, LENT at boot; chunks >= {} rows stage each \
         layer's RAM records ({} order, <= {} a layer) on the copy stream, the layer's \
         expert half waits for its arrivals; the rest read over UVA",
        shape.buffers,
        shape.slots,
        (n * stride) as f64 / 1048576.0,
        first + n as u32,
        law.above_rows,
        key.name(),
        shape.slots
    );
    Ok(r)
}

/// FN43: THE RING LAW at a pass of `rows` rows (after the borrow, before the tables): a pass the
/// law stages RECLAIMS the ring (copy stream drained, lent records flipped to their RAM shadows,
/// ledger moved); any other pass lends it again (EMPTY: the insert law fills it behind that
/// step's fence, after the ring's copies on the one copy stream). Returns whether the pass stages.
fn ring_for(rk: &mut Rank<'_, '_>, rows: u32, prefill: bool) -> anyhow::Result<bool> {
    let Some(r) = rk.ring.as_mut() else {
        return Ok(false);
    };
    let stage = prefill && r.law.stages(rows);
    if stage == r.lent {
        let t0 = Instant::now();
        if stage {
            rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
            r.dropped += rk.tier.tier.reclaim(&r.ids) as u64;
            rk.tier
                .tier
                .check_invariants()
                .map_err(|e| anyhow::anyhow!("after the ring's reclaim: {e}"))?;
            r.lent = false;
            r.reclaims += 1;
            r.reclaim_ms += t0.elapsed().as_secs_f64() * 1e3;
        } else {
            rk.tier
                .tier
                .lend(&r.ids)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            r.lent = true;
            r.lends += 1;
        }
        let budget = rk.budget.clone();
        r.relend(&budget)?;
    }
    Ok(stage)
}

/// FN43: a staged pass begins: the copy stream waits for every earlier read of the ring's slots
/// (the compute stream so far), then fills the first `buffers` layers.
fn ring_start(rk: &mut Rank<'_, '_>, plan: saber_mem::stage_ring::StagePlan) -> anyhow::Result<()> {
    rk.tier
        .mover
        .queue()
        .fence_after(&rk.dev.stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let r = rk.ring.as_mut().context("ring")?;
    r.plan = Some(plan);
    r.pass = Default::default();
    r.pass_alt = Default::default();
    for l in 0..r.shape.buffers.min(LAYERS) {
        ring_issue(rk, l, None)?;
    }
    Ok(())
}

/// FN43: issue layer `l`'s ring copies (after `wait`: its buffer's previous user ran).
fn ring_issue(rk: &Rank<'_, '_>, l: u32, wait: Option<usize>) -> anyhow::Result<()> {
    let r = rk.ring.as_ref().context("ring")?;
    let p = r.plan.as_ref().context("plan")?;
    let copies: Vec<(u32, u64)> = p.layers[l as usize]
        .copies
        .iter()
        .map(|&(ram, s)| (ram, u64::from(s) * r.stride))
        .collect();
    let q = rk.tier.mover.queue();
    q.stage(
        &r.mem,
        &copies,
        wait.map(|b| &r.consumed[b]),
        (l == 0).then_some(&r.t0),
        &r.arrived[l as usize],
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    if l + 1 == LAYERS {
        rk.hip.record_event(&r.t1, &q.stream)?;
    }
    Ok(())
}

/// FN43: after layer `l`'s launches: its buffer is free once they ran (a compute-stream mark),
/// and layer `l + buffers` fills it.
fn ring_after_layer(rk: &mut Rank<'_, '_>, l: u32) -> anyhow::Result<()> {
    let r = rk.ring.as_ref().context("ring")?;
    let b = r.shape.buffer(l);
    rk.hip
        .record_event(&r.consumed[b as usize], &rk.dev.stream)?;
    let next = l + r.shape.buffers;
    if next < LAYERS {
        ring_issue(rk, next, Some(b as usize))?;
    }
    Ok(())
}

/// FN47b: the capture arm's per-rank state.
struct PgRank {
    log: DevicePtr,
    /// The rows of the step whose 48 layers wrote the log (0 = none armed).
    armed: u32,
    /// Per step rows: [method a (own router on the attention-side stream), method b (the next
    /// layer's router on the MoE input)].
    tallies: std::collections::BTreeMap<u32, [saber_mem::step_ahead::PregateTally; 2]>,
}

/// FN47b: score an armed step's captured predictions against its TRUE routes (`ids` = the route
/// log, `[48][m][10]` i32) and this rank's placements (BEFORE `resolve`: the step's placement).
fn pregate_score_step(rk: &mut Rank<'_, '_>, m: u32, ids: &[u8]) -> anyhow::Result<()> {
    let Some(pg) = rk.pg.as_mut() else {
        return Ok(());
    };
    if pg.armed != m || m == 0 {
        pg.armed = 0;
        return Ok(());
    }
    pg.armed = 0;
    let bytes = super::exec::pregate_slot(2, 0) as usize;
    let raw = d2h_in(&rk.dev, &pg.log, pg.log.addr().as_u64(), bytes)?;
    let mu = m as usize;
    let tier = &rk.tier.tier;
    let t = pg.tallies.entry(m).or_default();
    for method in 0..2u64 {
        for l in 0..LAYERS {
            if method == 1 && l == 0 {
                continue; // nothing precedes layer 0's MoE input
            }
            let at = super::exec::pregate_slot(method, u64::from(l)) as usize;
            let lg: Vec<u16> = raw[at..at + mu * 1024]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let it = l as usize * mu * 40;
            let routed: Vec<u32> = ids[it..it + mu * 40]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            saber_mem::step_ahead::pregate_score(&mut t[method as usize], &lg, mu, &routed, |e| {
                tier.placement(l, e)
                    .map(|p| matches!(p, saber_mem::expert_tier::Placement::Ram(_)))
            });
        }
    }
    Ok(())
}

/// FN47b: the capture arm's end-of-run lines (one per step shape and method).
fn pregate_line(rk: &Rank<'_, '_>) {
    let Some(pg) = rk.pg.as_ref() else {
        return;
    };
    use saber_mem::step_ahead::{PREGATE_BUDGETS, PREGATE_KS};
    for (m, ts) in &pg.tallies {
        for (mi, t) in ts.iter().enumerate() {
            let ls = t.layers.max(1) as f64;
            let ks: Vec<String> = (0..PREGATE_KS.len())
                .map(|k| {
                    format!(
                        "K{} recall {:.1} % copies/layer {:.2} routes-recall {:.1} %",
                        PREGATE_KS[k],
                        100.0 * t.recall(k),
                        t.copies[k] as f64 / ls,
                        100.0 * t.routed_hits[k] as f64 / (t.routed.max(1)) as f64
                    )
                })
                .collect();
            let bs: Vec<String> = (0..PREGATE_BUDGETS.len())
                .map(|b| {
                    format!(
                        "B{} {:.1} % ({:.2} copies)",
                        PREGATE_BUDGETS[b],
                        100.0 * t.budget_hits[b] as f64 / (t.misses.max(1)) as f64,
                        t.budget_copies[b] as f64 / ls
                    )
                })
                .collect();
            println!(
                "fnpregate rank{} rows {m} method {}: layer-steps {} misses/layer {:.2} \
                 routed/layer {:.2} | {} | best-first {} | max copies {}",
                rk.rank,
                ["a:own-router@attn-in", "b:next-router@moe-in"][mi],
                t.layers,
                t.misses as f64 / ls,
                t.routed as f64 / ls,
                ks.join(" | "),
                bs.join(" "),
                t.max_copies
            );
        }
    }
}

/// FN43: a pass ends (tier_end, after the host synchronised the pass): the pass's line.
fn ring_end(rk: &mut Rank<'_, '_>, m: u32) -> anyhow::Result<()> {
    let Some(r) = rk.ring.as_mut() else {
        return Ok(());
    };
    let Some(p) = r.plan.take() else {
        return Ok(());
    };
    rk.hip.sync_event(&r.t1)?;
    let span = f64::from(rk.hip.elapsed_ms(&r.t0, &r.t1)?);
    let c = std::mem::take(&mut r.pass);
    r.total += c;
    // LR-H: the other key's counterfactual counts at the same routes and placements
    let alt = std::mem::take(&mut r.pass_alt);
    r.total_alt += alt;
    let order = ring_order_suffix(r.key, &c, r.shadow.take().map(|_| alt));
    r.passes += 1;
    r.span_ms += span;
    let gb = (p.copies() * r.stride) as f64 / 1e9;
    // LR-E: the armed pass's compute wait (the stream was synchronised by tier_end)
    let idle = match r.idle.as_ref() {
        Some(v) => {
            let w = v
                .iter()
                .map(|(a, b)| rk.hip.elapsed_ms(a, b))
                .collect::<Result<Vec<f32>, _>>()?;
            r.idle_ms += ring_idle_summary(&w).0;
            ring_idle_suffix(&w, span)
        }
        None => String::new(),
    };
    println!(
        "fnringpass rank{}: rows {m} staged {} records ({:.2} GB) max-layer-ram {} | copy span \
         {span:.2} ms = {:.1} GB/s | routed: ring {} uva {} vram {} | staged unrouted {}{idle}{order}",
        rk.rank,
        p.copies(),
        gb,
        p.max_layer_ram(),
        gb / (span / 1e3).max(1e-9),
        c.ring,
        c.uva,
        c.vram,
        c.wasted
    );
    Ok(())
}

/// FN43: the ring's end-of-run line.
fn ring_line(rk: &Rank<'_, '_>) {
    if let Some(r) = rk.ring.as_ref() {
        let t = r.total;
        println!(
            "fnringend rank{}: passes {} (incomplete-tier {}) | staged {} | routed: ring {} uva {} \
             vram {} | staged unrouted {} | copy span {:.1} ms | reclaims {} ({:.3} ms, {} records \
             flipped) lends {} | lent now {}{}{}",
            rk.rank,
            r.passes,
            r.incomplete,
            t.staged,
            t.ring,
            t.uva,
            t.vram,
            t.wasted,
            r.span_ms,
            r.reclaims,
            r.reclaim_ms,
            r.dropped,
            r.lends,
            r.lent,
            if r.idle.is_some() {
                format!(" | compute wait {:.1} ms (LR-E)", r.idle_ms)
            } else {
                String::new()
            },
            ring_order_suffix(r.key, &t, Some(r.total_alt))
        );
    }
}

/// FN43: the xptr hook of a staged layer: the route half runs, the compute stream waits for the
/// layer's ring arrivals, the expert half reads the (retargeted) tables.
struct Arrive<'a, 'h> {
    dev: &'a mut RankDev<'h>,
    table: &'a saber_exec::loader::FunctionTable<'h>,
    arrived: &'a saber_hip::Event,
    lw: &'a [LayerW],
    /// LR-E (`--fn-ring-idle`): the layer's (before, after) wait marks.
    idle: Option<&'a (saber_hip::Event, saber_hip::Event)>,
}

impl super::exec::XptrHook for Arrive<'_, '_> {
    fn tables(
        &mut self,
        ems: &mut [Emitter<'_>],
        _layer: u32,
        _ids: &[u64],
        _m: u64,
    ) -> anyhow::Result<Vec<(u64, u64)>> {
        self.dev.exec(self.table, &mut ems[0])?;
        if let Some((a, _)) = self.idle {
            self.dev.hip.record_event(a, &self.dev.stream)?;
        }
        self.dev
            .hip
            .stream_wait_event(&self.dev.stream, self.arrived)?;
        if let Some((_, b)) = self.idle {
            self.dev.hip.record_event(b, &self.dev.stream)?;
        }
        Ok(self.lw.iter().map(|w| (w.xgu, w.xdn)).collect())
    }
}

/// The tier's tables for a step of `rows` rows (a complete tier fills all 48 layers before the
/// step; FN42b: the arena borrow flips first; FN43: a PREFILL pass may stage the ring).
fn tier_begin(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    rows: u32,
    prefill: bool,
) -> anyhow::Result<bool> {
    tier_begin_up(rk, em, rows, prefill, false)
}

/// [`tier_begin`]; `async_up` (FN119d's early step): a rebuilt table set is uploaded without the
/// stream's sync (the stream holds the drafter round; the next rewrite of the pinned tables is a
/// later step's begin or S2 build, after the host waited this step's verify).
fn tier_begin_up(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    rows: u32,
    prefill: bool,
    async_up: bool,
) -> anyhow::Result<bool> {
    // FN56: a deferred tier end lands before any pass reads the tier
    tier_flush(rk)?;
    let pre = std::mem::take(&mut rk.tier_pre);
    // FN117b (S2): the build queued under the last drafter round (kept below iff its key holds;
    // any other begin -- a prefill pass, a staged one -- drops it)
    let hand_pre = rk.hand.as_mut().and_then(|h| h.pre.take());
    let flips = |rk: &Rank<'_, '_>| {
        (
            rk.borrow.as_ref().map(|b| (b.lends, b.reclaims)),
            rk.top.as_ref().map(|t| (t.lends, t.reclaims)),
            rk.buft.as_ref().map(|t| (t.lends, t.reclaims)),
        )
    };
    let f0 = flips(rk);
    // FN76b: the arena top's law around the borrow's
    lend_for(rk, em, rows, prefill, true)?;
    let staged = ring_for(rk, rows, prefill)?;
    let pre = pre && !prefill && !staged && flips(rk) == f0;
    // FN44b: THE FILL ON LEND -- a lend hands its slots over EMPTY and a prefill chunk's layers
    // are read-through (no insert), so before a prefill pass every empty lent slot (the borrow's,
    // the ring's) takes its layer's next profile / LRFU record: the copy stream waits for every
    // earlier use of the memory (the wide chunk's scratch, the ring's reads), one H2D a slot in
    // layer order, each layer's gate waits for its own copies (law 3) -- the pass's layer l reads
    // VRAM while layer l + 1's copies run
    if prefill
        && rk.lend_fill
        && !staged
        && rk.tier.tier.complete()
        && rk.tier.tier.lent_empty() > 0
    {
        let t0 = Instant::now();
        let t = &mut rk.tier;
        t.mover
            .after(&rk.dev.stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        t.tier
            .fill_lent(&mut t.mover)
            .map_err(|e| anyhow::anyhow!("the fill on lend: {e}"))?;
        rk.fill_passes += 1;
        rk.fill_ms += t0.elapsed().as_secs_f64() * 1e3;
    }
    rk.tier.tier.set_pass_prefill(prefill);
    rk.tier.tier.begin_step();
    let complete = rk.tier.tier.complete();
    // FN53c: an incomplete tier's decode / verify step keeps its graphs under the miss flag
    // (the valve's bounce must exist: the first close allocated it)
    let xm = !complete
        && !prefill
        && !rk.dev.op_timing
        && rk.xmiss.is_some()
        && rk.tier.tier.bounce().1 >= 2;
    rk.tier.tier.set_pass_xmiss(xm);
    if let Some(x) = rk.xmiss.as_mut() {
        x.step = xm;
        x.after_prefill |= prefill;
    }
    if xm {
        // a prefill pass's bounced records may have been promoted from the bounce: those copies
        // land before the miss flag reads into it (once after each prefill run)
        if rk.xmiss.as_ref().is_some_and(|x| x.after_prefill) {
            rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
            if let Some(x) = rk.xmiss.as_mut() {
                x.after_prefill = false;
                x.drains += 1;
            }
        }
        // every layer's tables at the step's placements, Fetch entries 0 (the kernel's flag)
        let mut fetch = 0usize;
        for l in 0..LAYERS {
            let (gu, dn, f) = rk.tier.tables(l, rk.rank);
            fetch += f;
            rk.tier.put(l, &gu, &dn);
        }
        rk.tier.upload(rk.hip, &rk.dev.stream, 0, LAYERS)?;
        if let Some(x) = rk.xmiss.as_mut() {
            x.steps += 1;
            x.fetch_entries = fetch as u64;
        }
        return Ok(true);
    }
    if complete {
        // FN43: a staged pass plans the ring at the pass's (frozen) placements and points the
        // staged records' entries at their ring slots
        // (LR-H: by the ring's order key; the other key's plan at the same placements is kept
        // host-only for the pass line's counterfactual -- never issued)
        let plan = staged.then(|| {
            let r = rk.ring.as_mut().expect("staged => ring");
            let (shape, key) = (r.shape, r.key);
            r.shadow = Some(saber_mem::stage_ring::StagePlan::plan_by(
                &rk.tier.tier,
                LAYERS,
                shape,
                key.other(),
            ));
            saber_mem::stage_ring::StagePlan::plan_by(&rk.tier.tier, LAYERS, shape, key)
        });
        let ring_base = rk.ring.as_ref().map_or(0, |r| r.mem.addr().as_u64());
        let kept = match hand_pre {
            Some(k) if !pre && !prefill && plan.is_none() => hand_s2_keep(rk, &k)?,
            Some(_) => {
                if let Some(h) = rk.hand.as_mut() {
                    h.s2_void += 1;
                }
                false
            }
            None => false,
        };
        if pre && plan.is_none() {
            // FN56 (`begin`): built and uploaded by the deferred tier end
            rk.tier_pre_n.1 += 1;
        } else if kept {
            // FN117b (S2): built and queued under the drafter round, the key unchanged
        } else {
            for l in 0..LAYERS {
                let (mut gu, mut dn, fetch) = rk.tier.tables(l, rk.rank);
                ensure!(fetch == 0, "a complete tier has a Fetch placement");
                if let Some(p) = plan.as_ref() {
                    p.retarget(&rk.tier.tier, l, ring_base, &mut gu, &mut dn);
                }
                rk.tier.put(l, &gu, &dn);
            }
            if async_up && plan.is_none() {
                rk.tier.upload_async(rk.hip, &rk.dev.stream, 0, LAYERS)?;
            } else {
                rk.tier.upload(rk.hip, &rk.dev.stream, 0, LAYERS)?;
            }
        }
        if let Some(p) = plan {
            ring_start(rk, p)?;
        }
    } else if staged {
        if let Some(r) = rk.ring.as_mut() {
            r.incomplete += 1;
        }
    }
    Ok(complete)
}

/// Run one layer at `m` rows (the step's tables `cc`, the states `st`) and log its route ids.
#[allow(clippy::too_many_arguments)]
fn layer(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    l: u32,
    complete: bool,
    st: LayerState,
    input: LayerIn,
    rows: u64,
    cc: ChunkConsts,
    stat: &mut StepStat,
    gate: bool,
) -> anyhow::Result<super::exec::LayerOut> {
    // a captured step gates every layer BEFORE the graph (no host work between its kernels)
    if gate {
        rk.tier
            .mover
            .gate(l, &rk.dev.stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let mut lw = [rk.lws[l as usize]];
    if complete {
        (lw[0].xgu, lw[0].xdn) = rk.tier.addr(l);
    }
    // FN53c: this step runs the device miss flag (one `fn_xmiss` launch a layer)
    let xm = complete && rk.xmiss.as_ref().is_some_and(|x| x.step);
    let m = cc.m;
    let ringed = rk.ring.as_ref().is_some_and(|r| r.plan.is_some());
    // FN47b: the capture arm rides a complete tier's target decode / verify steps
    em.pregate = None;
    if let Some(pg) = rk.pg.as_mut() {
        if complete && !xm && cc.decode && u64::from(m) <= super::exec::PREGATE_ROWS {
            if l == 0 {
                pg.armed = m;
            }
            if pg.armed == m {
                let next = if l + 1 < LAYERS {
                    rk.lws[l as usize + 1].router
                } else {
                    0
                };
                em.pregate = Some((pg.log.addr().as_u64(), next));
            }
        } else {
            pg.armed = 0;
        }
    }
    let outs = {
        let mut ar: Option<&mut dyn Reduce> = Some(&mut rk.pair);
        if complete && ringed {
            // FN43: the layer's route half runs, then the stream waits for the layer's ring
            // arrivals before its expert half (the grouped GEMMs read the ring)
            let r = rk.ring.as_ref().context("ring")?;
            let mut hook = Arrive {
                dev: &mut rk.dev,
                table: &rk.table,
                arrived: &r.arrived[l as usize],
                lw: &lw,
                idle: r.idle.as_ref().map(|v| &v[l as usize]),
            };
            emit_layer(
                std::slice::from_mut(em),
                &lw,
                &[st],
                &[input],
                &[rows],
                &[cc],
                &mut ar,
                &mut hook,
            )?
        } else if xm {
            let mbox = rk.xmiss.as_ref().map_or(0, |x| x.dev);
            let mut hook = MissHook {
                lw: &lw,
                mbox,
                lo: 256 * rk.rank as i32,
            };
            emit_layer(
                std::slice::from_mut(em),
                &lw,
                &[st],
                &[input],
                &[rows],
                &[cc],
                &mut ar,
                &mut hook,
            )?
        } else if complete {
            let mut hook = Resident(&lw);
            emit_layer(
                std::slice::from_mut(em),
                &lw,
                &[st],
                &[input],
                &[rows],
                &[cc],
                &mut ar,
                &mut hook,
            )?
        } else {
            let mut hook = Demand {
                devs: std::slice::from_mut(&mut rk.dev),
                tables: std::slice::from_ref(&rk.table),
                tiers: std::slice::from_mut(&mut rk.tier),
                counts: vec![PathCounts::default(); 1],
                nvme_distinct: vec![0; 1],
                unrouted_fetch: vec![0; 1],
            };
            let o = emit_layer(
                std::slice::from_mut(em),
                &lw,
                &[st],
                &[input],
                &[rows],
                &[cc],
                &mut ar,
                &mut hook,
            )?;
            stat.resident += hook.counts[0].resident;
            stat.ram += hook.counts[0].ram;
            stat.nvme += hook.counts[0].nvme;
            stat.distinct += hook.nvme_distinct[0];
            o
        }
    };
    let form = if cc.decode {
        saber_model::qwen4_exp::layer_graph::Form::Decode {
            // FN15b: the GDN decode form is keyed on the step's ROWS (b x verify width): the fused
            // update runs at one row only. FN24: on its CLASS rows (SPEC 0's b in the width-1 class)
            batch: em.class_rows(&cc),
            attn: saber_kernels::fn6::AttnForm::DecodePair,
        }
    } else {
        saber_model::qwen4_exp::layer_graph::Form::Prefill
    };
    let want = plan::declared_form_qsa(em.rules, l, 2, m, form, em.qsa_sparse(&cc), em.moe_train)
        + if (cc.decode || em.t8_prefill) && lw[0].t8.is_some() {
            super::exec::T8_LAYER_EXTRA
        } else {
            0
        }
        + u32::from(xm)
        // FN85: the reduces the prologue arm folded into their consumers
        - em.ar_rp_saved
        // FN137: the MoE-out reduce the tp2 back half carries
        - em.ar_far_saved
        // FN164: the mixer-out reduce the out / o projection carries
        - em.ar_mix_saved
        // FN86b: the launches the fusion arm folded into a neighbour
        - em.fuse86_saved
        // FN120c: the launches the dataflow form folded into one persistent launch
        - em.df_saved
        // FN122c: the dequant launches the dq family added
        + em.pf122_added;
    ensure!(
        outs[0].launches == want,
        "L{l}: launches {} vs declared {want} ({form:?})",
        outs[0].launches
    );
    rk.dev.exec(&rk.table, em)?;
    em.pregate = None;
    if complete && ringed {
        ring_after_layer(rk, l)?;
    }
    // FN120b: under `--fn-glue fold` the run's ids already sit at the log's offsets (one copy at
    // the run's end, `glue_end`); under `seeds` they sit in the glue region, not the arena
    let glue_run = em.glue.as_ref().filter(|g| g.on);
    let fold_ids = glue_run.is_some_and(|g| g.ids_stride > 0);
    if complete && !rk.dev.dry && !fold_ids {
        rk.copies += 1;
        // FN100: a lent tail is never written (the route log of more rows than the heads refuses)
        buf_check(rk, m, "the route-id log")?;
        // FN120e: by address -- a spilled run's later ids are the arena's
        let (src, src_offset) = match (glue_run.filter(|g| g.holds(outs[0].ids)), rk.glue.as_ref())
        {
            (Some(g), Some(gb)) => (gb, outs[0].ids - g.zero.base),
            _ => (&*rk.dev.arena, outs[0].ids - rk.dev.base()),
        };
        rk.hip.run_batch(&[Job::MemcpyD2d {
            dst: &rk.idlog.own,
            dst_offset: rk.idlog.off + u64::from(l) * u64::from(m) * 40,
            src,
            src_offset,
            bytes: u64::from(m) * 40,
            stream: &rk.dev.stream,
        }])?;
    }
    outs.into_iter().next().context("one rank")
}

/// FN53c: the valve's device miss flag, host half -- the coherent mailbox `fn_xmiss` raises, the
/// last sequence answered, the counters.
pub(super) struct Xmiss {
    mbox: saber_hip::PinnedBuf,
    /// The mailbox's device address (the kernel's `mbox`).
    dev: u64,
    served: u32,
    /// This step runs the miss flag.
    step: bool,
    /// A prefill pass ran since the last miss-flag step (drain once before reading the bounce).
    after_prefill: bool,
    steps: u64,
    drains: u64,
    requests: u64,
    records: u64,
    /// Host ms answering requests (the drive reads inside) and the longest one.
    serve_ms: f64,
    serve_max_ms: f64,
    /// Fetch entries in the last miss-flag step's tables.
    fetch_entries: u64,
}

/// The mailbox layout (bytes): REQ, LAYER, N, DSEQ, ACK words; the list; the u64 addresses.
const XM_REQ: usize = 0;
const XM_LAYER: usize = 4;
const XM_N: usize = 8;
const XM_ACK: usize = 16;
const XM_LIST: usize = 64;
const XM_ADDR: usize = 2048;
const XM_BYTES: u64 = 16 * 1024;

impl Xmiss {
    fn new(hip: &HipThread, rank: u32) -> anyhow::Result<Xmiss> {
        let mbox = hip.host_malloc_coherent(XM_BYTES)?;
        let dev = hip.host_device_addr(&mbox)?;
        println!(
            "fnxmiss rank{rank}: ARMED -- an open valve's decode / verify steps keep their graphs; fn_xmiss a layer raises the coherent mailbox at {dev:#x} ({XM_BYTES} B), the host reads the misses drive -> the bounce while it waits for the step (--fn-valve-graphs off = FN53b's demand path)"
        );
        Ok(Xmiss {
            mbox,
            dev,
            served: 0,
            step: false,
            after_prefill: false,
            steps: 0,
            drains: 0,
            requests: 0,
            records: 0,
            serve_ms: 0.0,
            serve_max_ms: 0.0,
            fetch_entries: 0,
        })
    }

    fn line(&self, rank: u32, vc: &saber_mem::expert_tier::ValveCounts) -> String {
        format!(
            "fnxmiss rank{rank}: steps {} (drains after prefill {}) | requests {} records {} (tier: requests {} records {} routes served {}) | host serve {:.1} ms (mean {:.3} ms, max {:.3} ms) | last step's Fetch entries {}",
            self.steps,
            self.drains,
            self.requests,
            self.records,
            vc.xmiss_requests,
            vc.xmiss_records,
            vc.xmiss_routes,
            self.serve_ms,
            self.serve_ms / (self.requests.max(1) as f64),
            self.serve_max_ms,
            self.fetch_entries
        )
    }
}

/// FN53c: the miss flag's xptr hook -- one `fn_xmiss_e256` launch between the route half and the
/// expert half; the expert half reads the layer's own tables (patched on the card on a miss).
struct MissHook<'a> {
    lw: &'a [LayerW],
    mbox: u64,
    lo: i32,
}

impl super::exec::XptrHook for MissHook<'_> {
    fn tables(
        &mut self,
        ems: &mut [Emitter<'_>],
        layer: u32,
        ids: &[u64],
        m: u64,
    ) -> anyhow::Result<Vec<(u64, u64)>> {
        let w = self.lw[0];
        ems[0].run(
            "fn_xmiss_e256",
            [1, 1, 1],
            256,
            vec![
                Arg::P(ids[0]),
                Arg::I((m * 10) as i32),
                Arg::P(w.xgu),
                Arg::P(w.xdn),
                Arg::I(self.lo),
                Arg::P(self.mbox),
                Arg::I(layer as i32),
            ],
            saber_model::qwen4_exp::layer_graph::Group::Moe,
        )?;
        Ok(self.lw.iter().map(|w| (w.xgu, w.xdn)).collect())
    }
}

/// FN53c: answer the miss flag's pending request, if any: read the layer's missing records
/// drive -> its bounce half, write their addresses, then the ACK (release). Returns whether one
/// was answered.
fn xmiss_serve(rk: &mut Rank<'_, '_>) -> anyhow::Result<bool> {
    let Some(x) = rk.xmiss.as_ref() else {
        return Ok(false);
    };
    let ld = |o: usize| x.mbox.load_u32_acquire(o).context("the mailbox");
    let req = ld(XM_REQ)?;
    if req == x.served {
        return Ok(false);
    }
    let t0 = Instant::now();
    ensure!(
        req == x.served.wrapping_add(1),
        "fn_xmiss: request {req} after {} (one outstanding at most)",
        x.served
    );
    let layer = ld(XM_LAYER)?;
    let n = ld(XM_N)? as usize;
    ensure!(
        (1..=256).contains(&n) && layer < LAYERS,
        "fn_xmiss: layer {layer} n {n}"
    );
    let lo = 256 * rk.rank;
    let mut experts = Vec::with_capacity(n);
    for j in 0..n {
        experts.push(lo + ld(XM_LIST + 4 * j)?);
    }
    let slots = rk
        .tier
        .tier
        .xmiss_slots(layer, &experts)
        .map_err(|e| anyhow::anyhow!("fn_xmiss: {e}"))?;
    let reads: Vec<(u64, u32)> = slots
        .iter()
        .map(|s| (u64::from(s.record), s.slot))
        .collect();
    rk.tier
        .mover
        .nvme_to_bounce(&reads)
        .map_err(|e| anyhow::anyhow!("fn_xmiss: {e}"))?;
    let x = rk.xmiss.as_mut().context("xmiss")?;
    for (j, s) in slots.iter().enumerate() {
        for (k, &a) in s.entries.iter().enumerate() {
            ensure!(
                x.mbox.store_u64_relaxed(XM_ADDR + 8 * (6 * j + k), a),
                "the mailbox's address {j}"
            );
        }
    }
    ensure!(x.mbox.store_u32_release(XM_ACK, req), "the mailbox's ACK");
    x.served = req;
    x.requests += 1;
    x.records += n as u64;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    x.serve_ms += ms;
    x.serve_max_ms = x.serve_max_ms.max(ms);
    Ok(true)
}

/// FN53c: the step's wait under the miss flag -- answer requests until the step's end event
/// completed (the device is parked in `fn_xmiss` until its request is answered).
fn xmiss_wait(rk: &mut Rank<'_, '_>, end: &saber_hip::Event) -> anyhow::Result<()> {
    loop {
        if xmiss_serve(rk)? {
            continue;
        }
        if rk.hip.query_event(end)? {
            // a request raised after the last poll cannot exist: the step has ended
            return Ok(());
        }
        std::hint::spin_loop();
    }
}

/// FN36: the demotion-free law's NVMe refills of the last `end_step`, read while the host waits for
/// the step's device work (the pool slots are touched by no placement and no copy).
fn tier_refill(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    let t = &mut rk.tier;
    t.tier
        .refill(&mut t.mover)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// The tier after a step: the fence, the counters from the logged routes, the LRU's inserts.
fn tier_end(
    rk: &mut Rank<'_, '_>,
    complete: bool,
    m: u32,
    stat: &mut StepStat,
) -> anyhow::Result<()> {
    tier_flush(rk)?;
    // FN59b: the served census -- this tier end's phases by the step's decode bucket
    let bucket = if rk.trace_meta.0 == saber_mem::route_trace::StepKind::Decode {
        rk.trace_meta.1
    } else {
        0
    };
    let keys = ["te_fence", "te_log", "te_resolve", "te_plan", "te_issue"];
    let before: Vec<f64> = keys
        .iter()
        .map(|k| stat.host.get(k).copied().unwrap_or(0.0))
        .collect();
    let t0 = Instant::now();
    let t = Instant::now();
    rk.tier
        .mover
        .after(&rk.dev.stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    host_lap(stat, "te_fence", t);
    tier_finish(rk, complete, m, stat)?;
    let a = rk.fn59.entry(bucket).or_default();
    a[0] += 1.0;
    a[1] += t0.elapsed().as_secs_f64() * 1e3;
    for (i, k) in keys.iter().enumerate() {
        a[2 + i] += stat.host.get(k).copied().unwrap_or(0.0) - before[i];
    }
    Ok(())
}

/// FN59b: the `fnhost59` census line -- a tier end's host ms split by decode bucket (`n` ends,
/// then the means of: the whole end, the fence, the route log read, the resolve, the PLANNING
/// (`end_step` minus the issue), the copy ISSUE, and the rest).
fn fnhost59_line(who: &str, acc: &BTreeMap<u32, [f64; 7]>) -> String {
    let mut out = format!("fnhost59 {who}:");
    for (b, a) in acc {
        let n = a[0].max(1.0);
        let other = a[1] - a[2..7].iter().sum::<f64>();
        out.push_str(&format!(
            " | b{b} n {}: tier_end {:.3} = fence {:.3} + log {:.3} + resolve {:.3} + plan {:.3} + issue {:.3} + other {:.3} (plan {:.0} % of end_step)",
            a[0],
            a[1] / n,
            a[2] / n,
            a[3] / n,
            a[4] / n,
            a[5] / n,
            a[6] / n,
            other / n,
            100.0 * a[5] / (a[5] + a[6]).max(1e-12)
        ));
    }
    out
}

/// FN56: add the host ms since `t` to the step's phase `name`.
fn host_lap(stat: &mut StepStat, name: &'static str, t: Instant) {
    *stat.host.entry(name).or_insert(0.0) += t.elapsed().as_secs_f64() * 1e3;
}

/// FN56 (`defer`): a verify step's tier end, cut at its fence -- the copy stream's fence is
/// recorded at the step's end (as [`tier_end`]), the rest (the route log's read, the resolve, the
/// copy issue) waits in `rk.tier_pend` until [`tier_flush`] runs it under the drafter round.
fn tier_end_fence(rk: &mut Rank<'_, '_>, complete: bool, m: u32) -> anyhow::Result<()> {
    tier_flush(rk)?;
    rk.tier
        .mover
        .after(&rk.dev.stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rk.tier_pend = Some((complete, m));
    Ok(())
}

/// FN56: run a deferred tier end now (none pending: nothing). Every tier pass's begin, every
/// drafter round's launch and the serve plane's no-round commit call it.
/// FN53b: rank 0's valve tick between jobs (one MemAvailable sample every 250 ms); returns the
/// closed-segment count to command when the controller moved.
fn valve_tick(st: &mut SrvState) -> Option<u32> {
    let (v, t0, last) = st.valve.as_mut()?;
    if last.elapsed().as_millis() < 250 {
        return None;
    }
    *last = Instant::now();
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let avail = saber_mem::mem_valve::mem_available_gib(&text)?;
    // FN53c: the engine's own RSS growth since the boot sample is not pressure
    let (pids, rss0) = &st.valve_own;
    let growth = own_rss_gib(pids).map_or(0.0, |r| (r - rss0).max(0.0));
    let s = saber_mem::mem_valve::Signals {
        t_ms: t0.elapsed().as_millis() as u64,
        avail_gib: avail,
        prefill_frac: 0.0,
        own_growth_gib: growth,
    };
    let mv = v.tick(s)?;
    println!(
        "{}",
        v.ledger()
            .last()
            .map(ToString::to_string)
            .unwrap_or_default()
    );
    match mv {
        saber_mem::mem_valve::Move::Demote(k) | saber_mem::mem_valve::Move::Promote(k) => Some(k),
        _ => None,
    }
}

/// FN53c: this engine's processes -- itself and its children (the worker form's rank 1) --
/// found once by a `/proc` scan.
fn own_pids() -> Vec<u32> {
    let me = std::process::id();
    let mut v = vec![me];
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for e in rd.flatten() {
            let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
                continue;
            };
            let Ok(st) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
                continue;
            };
            // `pid (comm) state ppid ...`: the fields after the last ')'
            let ppid = st
                .rsplit_once(')')
                .and_then(|(_, r)| r.split_whitespace().nth(1))
                .and_then(|x| x.parse::<u32>().ok());
            if ppid == Some(me) {
                v.push(pid);
            }
        }
    }
    v
}

/// FN53c: the summed VmRSS (GiB) of `pids` (`None` when none is readable).
fn own_rss_gib(pids: &[u32]) -> Option<f64> {
    let mut sum = None;
    for p in pids {
        if let Some(r) = std::fs::read_to_string(format!("/proc/{p}/status"))
            .ok()
            .and_then(|t| saber_mem::mem_valve::vm_rss_gib(&t))
        {
            *sum.get_or_insert(0.0) += r;
        }
    }
    sum
}

/// FN53b: close / reopen this rank's valve band segments until `want` are closed, between
/// steps: the deferred tier end lands, both streams drain, then each close drops its records
/// to `Fetch`, copies the segment's survivors below it and FREES the segment; each reopen
/// allocates it again and refills it from the drive. Placement only.
fn valve_apply(rk: &mut Rank<'_, '_>, want: u32) -> anyhow::Result<()> {
    use saber_mem::device_mover::CopyQueue as _;
    use saber_mem::expert_tier::{ExpertMover as _, VALVE_SEG};
    let t = Instant::now();
    tier_flush(rk)?;
    rk.tier_pre = false;
    rk.hip.sync_stream(&rk.dev.stream)?;
    let e = |x: saber_mem::expert_tier::TierError| anyhow::anyhow!("valve: {x}");
    rk.tier.mover.queue_mut().drain().map_err(e)?;
    let (_, n, _) = rk.tier.tier.band();
    let closed = |rk: &Rank<'_, '_>| (n - rk.tier.tier.band().2) / VALVE_SEG;
    let (mut drop, mut reloc, mut refill) = (0usize, 0usize, 0usize);
    while closed(rk) < want {
        let (_, bn) = rk.tier.tier.bounce();
        let a = rk.tier.mover.queue_mut().valve_bounce(bn).map_err(e)?;
        rk.tier.tier.set_bounce_addr(a);
        let c = rk.tier.tier.valve_close().map_err(e)?;
        for &(from, to) in &c.relocs {
            rk.tier.mover.queue_mut().ram_copy(from, to).map_err(e)?;
        }
        rk.tier.mover.queue_mut().valve_free(c.seg);
        drop += c.dropped.len();
        reloc += c.relocs.len();
    }
    while closed(rk) > want {
        let k = rk.tier.tier.band().2 / VALVE_SEG;
        let a = rk.tier.mover.queue_mut().valve_alloc(k).map_err(e)?;
        let reads = rk.tier.tier.valve_open(a).map_err(e)?;
        rk.tier.mover.nvme_to_ram(&reads).map_err(e)?;
        refill += reads.len();
    }
    rk.tier
        .tier
        .check_invariants()
        .map_err(|x| anyhow::anyhow!("valve: {x}"))?;
    println!(
        "fnvalve rank{}: closed {want} of {} segments | dropped {drop} relocated {reloc} \
         refilled {refill} | tier {} | {:.1} ms | {:?}",
        rk.rank,
        n / VALVE_SEG,
        if rk.tier.tier.complete() {
            "complete"
        } else {
            if rk.xmiss.is_some() {
                "INCOMPLETE (decode: graphs + the device miss flag; prefill: the demand path)"
            } else {
                "INCOMPLETE (demand path)"
            }
        },
        t.elapsed().as_secs_f64() * 1e3,
        rk.tier.tier.valve_counts()
    );
    Ok(())
}

fn tier_flush(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    if let Some((complete, m)) = rk.tier_pend.take() {
        let t = Instant::now();
        let mut st = StepStat::default();
        tier_finish(rk, complete, m, &mut st)?;
        if rk.lev.begin && complete && rk.tier.tier.complete() {
            // FN56 (`begin`): the tables of the placements this end left (no placement moves
            // before the next step's begin: a refill touches pool slots only; a lend / reclaim /
            // staged pass at that begin discards them). The upload syncs the stream: it runs
            // last, where the round's host part waits for the round anyway.
            for l in 0..LAYERS {
                let (gu, dn, fetch) = rk.tier.tables(l, rk.rank);
                ensure!(fetch == 0, "a complete tier has a Fetch placement");
                rk.tier.put(l, &gu, &dn);
            }
            rk.tier.upload(rk.hip, &rk.dev.stream, 0, LAYERS)?;
            rk.tier_pre = true;
            rk.tier_pre_n.0 += 1;
        }
        rk.tier_def_ms += t.elapsed().as_secs_f64() * 1e3;
    }
    Ok(())
}

/// The tier end past its fence: the route log, the resolve, the LRU's inserts (the copy issue).
fn tier_finish(
    rk: &mut Rank<'_, '_>,
    complete: bool,
    m: u32,
    stat: &mut StepStat,
) -> anyhow::Result<()> {
    if complete {
        let tt = Instant::now();
        // FN119f: a first-token prefill's log, queued behind its chunk (pinned, its event)
        let plog = match rk
            .ahead
            .as_mut()
            .and_then(|a| a.early.as_mut())
            .filter(|e| e.plog_m == Some(m))
        {
            Some(e) => {
                e.plog_m = None;
                rk.hip.sync_event(&e.plog_ev)?;
                Some(e.plog.as_slice()[..(LAYERS * m * 40) as usize].to_vec())
            }
            None => None,
        };
        if plog.is_some() {
            if let Some(a) = rk.ahead.as_mut() {
                a.c.dev_plog += 1;
            }
        }
        // FN112 (`--fn-overlap on`): the log queued behind the step (the same range)
        let log = match plog.or_else(|| rk.ov112.as_mut().and_then(|o| o.take_log(m))) {
            Some(v) => {
                if let Some(o) = rk.ov112.as_mut() {
                    o.host_ms += tt.elapsed().as_secs_f64() * 1e3;
                }
                v
            }
            None => d2h_in(
                &rk.dev,
                &rk.idlog.own,
                rk.idlog.addr().as_u64(),
                (LAYERS * m * 40) as usize,
            )?,
        };
        host_lap(stat, "te_log", tt);
        let tt = Instant::now();
        pregate_score_step(rk, m, &log)?;
        // LR-H: a staged pass decays the served order key once, then notes its routes
        let staged_pass = rk.ring.as_ref().is_some_and(|r| r.plan.is_some());
        if staged_pass {
            rk.tier.tier.decay_prefill_seen();
        }
        for l in 0..LAYERS as usize {
            let at = l * m as usize * 40;
            let routed: Vec<u32> = log[at..at + m as usize * 40]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if let Some(r) = rk.ring.as_mut().filter(|_| staged_pass) {
                let seen = saber_mem::stage_ring::routed_seen(&rk.tier.tier, &routed);
                if let Some(p) = r.plan.as_ref() {
                    r.pass += p.classify_seen(&rk.tier.tier, l as u32, &seen);
                }
                if let Some(p) = r.shadow.as_ref() {
                    r.pass_alt += p.classify_seen(&rk.tier.tier, l as u32, &seen);
                }
                rk.tier.tier.note_prefill_seen(l as u32, &seen);
            }
            let t = &mut rk.tier;
            let c = t
                .tier
                .resolve(l as u32, &routed, &mut t.mover)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            ensure!(
                c.nvme == 0 || rk.xmiss.as_ref().is_some_and(|x| x.step),
                "the complete tier read NVMe"
            );
            stat.resident += c.resident;
            stat.ram += c.ram;
        }
        if let Some(tw) = rk.trace.as_mut() {
            let (kind, bucket, w, live) = rk.trace_meta;
            let live = if kind == saber_mem::route_trace::StepKind::Prefill {
                m
            } else {
                live
            };
            // FN65: the log moves to the recorder (the trace converts it here, as before)
            tw.target_log(kind, m, live, bucket, w, log)?;
        }
        host_lap(stat, "te_resolve", tt);
    }
    let tt = Instant::now();
    // FN18: the step's insert cap by its decode bucket (a prefill chunk: bucket 0 = the default)
    let bucket = if rk.trace_meta.0 == saber_mem::route_trace::StepKind::Decode {
        rk.trace_meta.1
    } else {
        0
    };
    rk.tier.tier.set_max_inserts(rk.inserts.cap(bucket));
    rk.tier.tier.set_promote_budget(rk.inserts.budget(bucket));
    rk.tier
        .tier
        .set_second_chance(rk.inserts.second_chance(bucket));
    rk.trace_meta = (saber_mem::route_trace::StepKind::Prefill, 0, 1, 0);
    ring_end(rk, m)?;
    let t = &mut rk.tier;
    // FN59b: end_step = the planning + the copy issue (the queue's own clock inside `issue`)
    let i0 = t.mover.queue().issue_ns();
    let tp = Instant::now();
    t.tier
        .end_step(&mut t.mover)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let es = tp.elapsed().as_secs_f64() * 1e3;
    let iss = (t.mover.queue().issue_ns() - i0) as f64 * 1e-6;
    *stat.host.entry("te_issue").or_insert(0.0) += iss;
    *stat.host.entry("te_plan").or_insert(0.0) += es - iss;
    host_lap(stat, "te_end", tt);
    Ok(())
}

fn times(rk: &mut Rank<'_, '_>, stat: &mut StepStat) -> anyhow::Result<()> {
    if rk.dev.op_timing {
        for (g, _, sym, ms) in rk.dev.take_times()? {
            *stat.groups.entry(g.name().to_string()).or_insert(0.0) += ms;
            *stat.ops.entry(sym).or_insert(0.0) += ms;
            stat.busy_ms += ms;
        }
    }
    Ok(())
}

/// Prefill `ids` into sequence `lv` (chunks of `--fn-chunk`, continuing its state), returning the
/// pick on the last row.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn prefill(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    lv: &mut Live,
    ids: &[u32],
    f: &FnFlags,
    cap: Option<&mut Cap>,
    tap: Option<&Path>,
    req: Option<&RowReq>,
) -> anyhow::Result<(Pick, StepStat, Option<RowOut>)> {
    prefill_with(rk, em, ple, ph, recs, lv, ids, f, cap, tap, req, None)
}

/// [`prefill`] with `after`: work queued right behind the last chunk before the host waits for it
/// (FN119f's first token on the card); the chunk's pick then reads FN112's queued staging and its
/// tier end reads the route log queued with it (deferred: `tier_flush`).
#[allow(clippy::too_many_arguments)]
fn prefill_with(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    lv: &mut Live,
    ids: &[u32],
    f: &FnFlags,
    mut cap: Option<&mut Cap>,
    tap: Option<&Path>,
    req: Option<&RowReq>,
    mut after: Option<AfterChunk<'_>>,
) -> anyhow::Result<(Pick, StepStat, Option<RowOut>)> {
    let mut stat = StepStat::default();
    let mut hooked = false;
    let m_all = ids.len() as u32;
    let mut staged_front: Option<ChunkConsts> = None;
    // FN19: a prompt past 2,048 tokens runs in 2,048-row PIECES (each staged on its own: the
    // embedding / PLE staging buffers hold 2,048 rows); the chunk law inside a piece is unchanged
    // (FN38: pieces of the boot's widest chunk, `--fn-max-chunk`)
    let wide = f.max_chunk();
    let pieced = m_all > wide;
    ensure!(
        !pieced || rk.mtp.is_none(),
        "{m_all} tokens: a prompt past {wide} runs at SPEC 0 in the harness (the serve plane chunks it)"
    );
    ensure!(
        lv.len + m_all <= rk.max_ctx,
        "context over the pools' {}",
        rk.max_ctx
    );
    // FN55b: the PLE n-gram hash sees the RAW ids (`<|image_pad|>` included: upstream's
    // `requires_raw_input_tokens`); the image rows replace the pads' embedding rows below
    let pids = ph.ids_step(&mut lv.ctx, ids);
    let npid = pids.len() / ids.len().max(1);
    let (img, base) = (lv.img.clone(), lv.len);
    let stage = |rk: &mut Rank<'_, '_>,
                 ple: &mut PleTier,
                 stat: &mut StepStat,
                 a: u32,
                 b: u32|
     -> anyhow::Result<()> {
        let (a, b) = (a as usize, b as usize);
        let mut staged = vec![0u8; (b - a) * 5120];
        let g0 = Instant::now();
        let c = ple
            .gather(&pids[a * npid..b * npid], &mut staged)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some(tw) = rk.trace.as_mut() {
            tw.ple(false, &pids[a * npid..b * npid])?;
        }
        stat.ple_ms += g0.elapsed().as_secs_f64() * 1e3;
        stat.ple_hits += c.cache_hits;
        stat.ple_nvme += c.nvme_reads;
        let e = embed_rows(recs, &ids[a..b])?;
        buf_h2d(rk, BufK::Emb, &e)?;
        if let Some(im) = img.as_ref() {
            // FN55b: the tower's rows over the image tokens' embedding rows, BEFORE layer 0's
            // hyper-connection expansion (upstream merges into `embed_tokens`' output)
            let runs = im.rope.scatter_runs(base + a as u32, (b - a) as u32);
            if !runs.is_empty() {
                let v = rk.vis.as_mut().context("image rows on a text-only boot")?;
                v.scatter(
                    rk.hip,
                    &rk.dev.stream,
                    im.epoch,
                    &runs,
                    &rk.emb.own,
                    rk.emb.off,
                )?;
            }
        }
        buf_h2d(rk, BufK::Rows, &staged)?;
        Ok(())
    };
    if !pieced {
        stage(rk, ple, &mut stat, 0, m_all)?;
    }
    let cut = if pieced {
        if f.fn_chunk > 0 {
            f.fn_chunk.min(wide)
        } else {
            wide
        }
    } else {
        f.fn_chunk
    };
    let cuts: Vec<(u32, u32)> = if cut > 0 && cut < m_all {
        (0..m_all)
            .step_by(cut as usize)
            .map(|a| (a, (a + cut).min(m_all)))
            .collect()
    } else {
        vec![(0, m_all)]
    };
    let ev = (
        rk.hip.create_event(EventFlags::Default)?,
        rk.hip.create_event(EventFlags::Default)?,
    );
    let launches0 = rk.dev.launches;
    let mut out = None;
    for (ci, &(a, b)) in cuts.iter().enumerate() {
        let m = b - a;
        let last = ci + 1 == cuts.len();
        let chunk = lv.chunks;
        // a pieced prompt stages each piece at the buffers' row 0
        let srow = if pieced {
            stage(rk, ple, &mut stat, a, b)?;
            0
        } else {
            a
        };
        // FN51f-b: the elastic law before the chunk's step law (a long request's borrow lands
        // here, before its first chunk: the gather fills the reclaimed slots)
        elastic_tick(rk, Some((lv.slot, lv.len + a + m)))?;
        // FN130: the chunk's exact opens against the cap the tick left (the demand law)
        elastic_demand(
            rk,
            &[(lv.slot, lv.len + a, lv.len + a + m)],
            "prefill chunk",
        )?;
        let cc = kv_consts(rk, m, lv.len + a, chunk, lv.slot)?;
        if em.ctl == Ctl::DropCarry && chunk > 0 {
            for l in 0..LAYERS as usize {
                if plan::kind(l as u32) != LayerKind::Qsa {
                    let v = rk.pools.seq(lv.slot, l);
                    rk.pools
                        .memset(rk.hip, &rk.dev, v.h[(chunk % 2) as usize], rk.pools.hb)?;
                }
            }
        }
        let complete = tier_begin(rk, em, m, true)?;
        rk.hip.record_event(&ev.0, &rk.dev.stream)?;
        let mut input = LayerIn::Embed(rk.emb.addr().as_u64() + u64::from(srow) * H * 2);
        let rows = rk.rows.addr().as_u64() + u64::from(srow) * H * 2;
        for l in 0..LAYERS {
            em.taps.clear();
            em.qsa_sel = None;
            let st = rk.pools.seq(lv.slot, l as usize);
            let o = layer(rk, em, l, complete, st, input, rows, cc, &mut stat, true)?;
            if let (Some(c), Some((sel, iq, rows_))) = (cap.as_deref_mut(), em.qsa_sel.take()) {
                // FN19: the sparse rows' selections (the reference's `idx_sel` law) and the indexer
                // queries they were scored from (the exact selection gate's inputs)
                let sb = d2h_in(&rk.dev, &rk.dev.arena, sel, (rows_ * 512 * 4) as usize)?;
                let qb = d2h_in(&rk.dev, &rk.dev.arena, iq, (rows_ * 512 * 2) as usize)?;
                c.put_sel(l as usize, (lv.len + a) as usize, &sb, &qb);
            }
            if let Some(c) = cap.as_deref_mut() {
                let mm = u64::from(m);
                let rsd = em.arena.take(mm * W * 2, super::kern::Seed::None)?;
                em.materialize(&o.pend, mm, rsd)?;
                rk.dev.exec(&rk.table, em)?;
                let g = |x: u64, n: u64| d2h_in(&rk.dev, &rk.dev.arena, x, n as usize);
                let li = l as usize;
                let a0 = a as usize;
                let keep = c.keep.clone();
                let kp = keep.as_deref();
                Cap::put_rows(
                    &mut c.resid,
                    c.t,
                    kp,
                    li,
                    a0,
                    W as usize * 2,
                    &g(rsd, mm * W * 2)?,
                );
                Cap::put_rows(
                    &mut c.att,
                    c.t,
                    kp,
                    li,
                    a0,
                    H as usize * 2,
                    &g(o.att, mm * H * 2)?,
                );
                if !c.mlp.is_empty() {
                    Cap::put_rows(
                        &mut c.mlp,
                        c.t,
                        kp,
                        li,
                        a0,
                        H as usize * 2,
                        &g(o.pend.mo, mm * H * 2)?,
                    );
                }
                Cap::put_rows(&mut c.ids, c.t, kp, li, a0, 40, &g(o.ids, mm * 40)?);
                if let Some(po) = o.ple {
                    let b = g(po, mm * W * 2)?;
                    Cap::put_rows(&mut c.ple, c.t, kp, 0, a0, W as usize * 2, &b);
                }
            }
            if let (Some(dir), false) = (tap, em.taps.is_empty()) {
                let d = dir.join(format!("rank{}", rk.rank));
                std::fs::create_dir_all(&d)?;
                for (n, adr, bytes) in std::mem::take(&mut em.taps) {
                    let b = d2h_in(&rk.dev, &rk.dev.arena, adr, bytes as usize).or_else(|_| {
                        let (p, off) = rk.pools.owner(adr)?;
                        let mut v = vec![0u8; bytes as usize];
                        rk.hip.sync_stream(&rk.dev.stream)?;
                        rk.hip.memcpy_d2h_addr_blocking(
                            &mut v,
                            p.addr().offset(off).context("off")?,
                            &rk.dev.stream,
                        )?;
                        anyhow::Ok(v)
                    })?;
                    std::fs::write(d.join(format!("c{ci}_a{a}_m{m}_{n}.bin")), b)?;
                }
            }
            input = LayerIn::Pending(o.pend);
        }
        let LayerIn::Pending(pend) = input else {
            unreachable!()
        };
        let under = last
            && cap.is_none()
            && rk.mtp.is_some()
            && matches!(rk.front_under, Some(FrontUnder::Want));
        if under {
            // FN119g: the final mixer, the stream rows kept, the K-side front on the side stream
            // from here, then the head (the same launches and addresses as `emit_post3`)
            let (fx, _, strm) = super::exec::emit_post3_opt(
                em,
                &rk.fm,
                rk.head,
                &pend,
                u64::from(m),
                false,
                false,
            )?;
            rk.dev.exec(&rk.table, em)?;
            ensure!(
                cuts.len() == 1,
                "--fn-mtp: one prefill chunk per prompt this round"
            );
            mtp_keep_stream(rk, strm, m)?;
            let ev_s = rk.hip.create_event(EventFlags::Default)?;
            rk.hip.record_event(&ev_s, &rk.dev.stream)?;
            let lg = super::exec::emit_head(em, rk.head, fx, u64::from(m), false)?;
            rk.dev.exec(&rk.table, em)?;
            front_under_issue(rk, em, recs, lv, ids, &ev_s)?;
            out = Some(lg);
        } else if last || cap.is_some() {
            let (fx, lg, strm) = emit_post3(em, &rk.fm, rk.head, &pend, u64::from(m), false)?;
            rk.dev.exec(&rk.table, em)?;
            if rk.mtp.is_some() {
                ensure!(
                    cuts.len() == 1,
                    "--fn-mtp: one prefill chunk per prompt this round"
                );
                mtp_keep_stream(rk, strm, m)?;
            }
            if let Some(c) = cap.as_deref_mut() {
                let b = d2h_in(&rk.dev, &rk.dev.arena, fx, (u64::from(m) * H * 2) as usize)?;
                let keep = c.keep.clone();
                Cap::put_rows(
                    &mut c.fin,
                    c.t,
                    keep.as_deref(),
                    0,
                    a as usize,
                    H as usize * 2,
                    &b,
                );
            }
            if last {
                out = Some(lg);
            }
        }
        rk.hip.record_event(&ev.1, &rk.dev.stream)?;
        tier_refill(rk)?;
        // FN119e (`--fn-first-step chunk`): the K-side front's inputs staged while the chunk runs
        // (async behind it, after its last kernel); its kernels go after the pick's reads
        if rk.front_want && last && m >= 2 {
            staged_front = Some(front_stage(
                rk,
                recs,
                lv.slot,
                &ids[1..],
                lv.len,
                lv.chunks,
            )?);
        }
        // FN119f: the first token's walk, the front and the first round behind the chunk (the
        // pick's reads and the route log queued first: the host's waits stay the chunk's)
        if last {
            if let (Some(h), Some(lg)) = (after.as_mut(), out) {
                h(rk, em, lv, lg)?;
                hooked = true;
            }
        }
        rk.dev.wait_event(&ev.1, (1u64 << 62) | u64::from(m))?; // FN57: the boot's sleeper
        if hooked {
            // the queued seam / rows landed (FN112's event: right behind the chunk)
            let o = rk.ov112.as_ref().context("FN112's staging")?;
            rk.dev.wait_event(&o.ev, (1u64 << 62) | u64::from(m))?;
        }
        stat.dev_ms += f64::from(rk.hip.elapsed_ms(&ev.0, &ev.1)?);
        // FN30c: a prefill chunk's copies are never held (no drafter round is its consumer)
        rk.tier.mover.set_hold(false);
        if hooked {
            // FN119f: the fence now (the mover after the queued round), the log read later from
            // the copy queued behind the chunk (`tier_finish`)
            tier_end_fence(rk, complete, m)?;
        } else {
            tier_end(rk, complete, m, &mut stat)?;
        }
        times(rk, &mut stat)?;
        lv.chunks += 1;
    }
    lv.len += m_all;
    lv.gdn_at = (lv.chunks % 2) as usize;
    lv.ple_at = (lv.chunks % 2) as usize;
    if let Some(dp) = rk.first_norm.take().filter(|_| hooked) {
        // FN119f: the hook normalized the state behind the chunk (`Cmd::Ready`'s is then a no-op)
        lv.gdn_at = 0;
        lv.ple_at = dp;
    }
    stat.launches = rk.dev.launches - launches0;
    if let Some(c) = cap.as_deref_mut() {
        if c.sel_rows > 0 {
            // FN19: every QSA layer's compressed keys of the whole prompt (the slot's identity
            // pages: group g at row g of the slot's store), the scores' other input. FN51c:
            // under admission the store is PAGED -- block j (4 groups x 256 B) of the slot is
            // page `row[j]` of the layer's pool, read through the allocator's row (runs of
            // consecutive pages in one copy)
            rk.hip.sync_stream(&rk.dev.stream)?;
            let g = (lv.len / 4) as usize;
            const PAGE_B: usize = 4 * 256;
            let runs: Vec<(usize, usize, usize)> = if rk.pools.ix_paged {
                let kv = rk
                    .pools
                    .kvt
                    .as_ref()
                    .context("a paged store without the tier")?;
                let k = kv.borrow();
                let row = k.ix.as_ref().context("the paged index pool")?.row(lv.slot);
                let nb = g.div_ceil(4);
                ensure!(
                    row.len() >= nb,
                    "the slot's index row holds {} of {nb} blocks",
                    row.len()
                );
                // (destination byte, source page, pages)
                let mut runs: Vec<(usize, usize, usize)> = Vec::new();
                for (j, &p) in row[..nb].iter().enumerate() {
                    ensure!(
                        p >= 0 && p != k.ix.as_ref().map_or(-1, |x| x.pad()),
                        "block {j} of slot {} has no page",
                        lv.slot
                    );
                    let p = p as usize;
                    match runs.last_mut() {
                        Some(r) if r.1 + r.2 == p => r.2 += 1,
                        _ => runs.push((j * PAGE_B, p, 1)),
                    }
                }
                runs
            } else {
                Vec::new()
            };
            for l in (3..LAYERS as usize).step_by(4) {
                let st = rk.pools.seq(lv.slot, l);
                let (p, off) = rk.pools.owner(st.comp)?;
                let mut v = vec![0u8; g * 256];
                if rk.pools.ix_paged {
                    let mut tmp = Vec::new();
                    for &(dst, page, n) in &runs {
                        tmp.resize(n * PAGE_B, 0);
                        rk.hip.memcpy_d2h_addr_blocking(
                            &mut tmp,
                            p.addr()
                                .offset(off + (page * PAGE_B) as u64)
                                .context("off")?,
                            &rk.dev.stream,
                        )?;
                        let end = (dst + n * PAGE_B).min(v.len());
                        v[dst..end].copy_from_slice(&tmp[..end - dst]);
                    }
                } else {
                    rk.hip.memcpy_d2h_addr_blocking(
                        &mut v,
                        p.addr().offset(off).context("off")?,
                        &rk.dev.stream,
                    )?;
                }
                c.put_comp(l, &v);
            }
        }
    }
    let lg = out.context("no last chunk")?;
    rk.last_lg = lg;
    if cap.is_some() {
        let pk = pick(rk, lg, 1, true)?;
        return Ok((pk, stat, None));
    }
    // FN14: the seam pick (top-64, log-sum-exps, probes); its first 20 are the old pick's
    let o = pick2(
        rk,
        lg,
        1,
        std::slice::from_ref(req.unwrap_or(&RowReq::default())),
    )?
    .pop()
    .context("one row")?;
    let pk = (vec![o.top[0].0], vec![o.top[..TOPK].to_vec()], Vec::new());
    if let Some(cc) = staged_front {
        // FN119e: the last logits row backed up (the dense fallback may need it after the front
        // reused the arena), then the front's kernels -- running while the host samples
        front_backup(rk, lg)?;
        front_issue(rk, em, lv.slot, m_all - 1, cc)?;
        rk.front_done = true;
        rk.front_early += 1;
    }
    Ok((pk, stat, Some(o)))
}

/// FN119e: the K-side front's inputs for `toks` (the prompt's rows 1.., positions `p0..`) staged
/// in the early pinned buffer and uploaded ASYNC on the step's stream (behind the chunk that is
/// still running): the embedding rows, the chunk constants, the KV step law + the append slots +
/// the page row -- [`mtp_rows_emb`] + [`kv_consts`] with their blocking copies made async.
/// FN119g: the first round's K-side front (the previous piece's stash row + rows 0..m-2) on the
/// side stream, after `ev_s` (the chunk's final mixer + the stream rows kept), while the main
/// stream runs the head, the pick's reads and the walk. Its arena scratch starts above the head's
/// (the floor) and the main stream's takes stay below it until the round joins (the ceiling).
fn front_under_issue(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    lv: &mut Live,
    ids: &[u32],
    ev_s: &saber_hip::Event,
) -> anyhow::Result<()> {
    let m = ids.len() as u32;
    ensure!(m >= 2, "FN119g: a front under the head without rows");
    let (p0, chunk, slot) = (lv.len, lv.chunks, lv.slot);
    // h19: the chunk's tail holds seeded buffers too (their seeds already issued): every
    // per-layer region gets the same floor / ceiling
    ensure!(
        em.arena.pending_seeds.is_empty(),
        "FN119g: seeds pending under the head"
    );
    let al = |x: u64| x.div_ceil(super::kern::ALIGN) * super::kern::ALIGN;
    let floors = [
        al(em.arena.scratch.top),
        al(em.arena.zero.top),
        al(em.arena.ones.top),
    ];
    let mut side = rk.front_side.take().context("FN119g: the side stream")?;
    std::mem::swap(&mut rk.dev.stream, &mut side);
    let r = (|| -> anyhow::Result<saber_hip::Event> {
        rk.hip.stream_wait_event(&rk.dev.stream, ev_s)?;
        em.arena.scratch.floor = floors[0];
        em.arena.zero.floor = floors[1];
        em.arena.ones.floor = floors[2];
        em.arena.new_layer();
        let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
        let prev = std::mem::take(&mut lv.stash);
        let mut base = 0usize;
        if !prev.is_empty() {
            ensure!(p0 >= 1 && chunk >= 1, "a stash without a previous chunk");
            let sa = {
                let mt = rk.mtp.as_ref().context("mtp")?;
                let e = rk
                    .ahead
                    .as_mut()
                    .and_then(|a| a.early.as_mut())
                    .context("FN119f: the early buffers")?;
                e.stin.as_mut_slice()[..prev.len()].copy_from_slice(&prev);
                rk.hip.memcpy_h2d_at(
                    &mt.stash,
                    0,
                    e.stin
                        .slice(0, prev.len())
                        .context("FN119g: the stash row")?,
                    &rk.dev.stream,
                )?;
                mt.stash.addr().as_u64()
            };
            let (cc, nb) = front_stage_at(rk, recs, slot, &ids[..1], p0 - 1, chunk - 1, base)?;
            base = nb;
            front_issue_at(rk, em, slot, 1, cc, sa)?;
        }
        let (cc, _) = front_stage_at(rk, recs, slot, &ids[1..], p0, chunk, base)?;
        front_issue_at(rk, em, slot, m - 1, cc, hid)?;
        let ev = rk.hip.create_event(EventFlags::Default)?;
        rk.hip.record_event(&ev, &rk.dev.stream)?;
        Ok(ev)
    })();
    std::mem::swap(&mut rk.dev.stream, &mut side);
    rk.front_side = Some(side);
    let r2 = r.and_then(|ev| {
        ensure!(
            em.arena.pending_seeds.is_empty(),
            "FN119g: the front left seeds pending"
        );
        Ok(ev)
    });
    // until the join: the main stream below the front's regions, the front's regions intact
    for (b, f) in [
        (&mut em.arena.scratch, floors[0]),
        (&mut em.arena.zero, floors[1]),
        (&mut em.arena.ones, floors[2]),
    ] {
        b.floor = 0;
        b.ceil = f;
        b.top = 0;
    }
    let ev = r2?;
    rk.front_early += 1;
    rk.front_under = Some(FrontUnder::Issued(ev));
    Ok(())
}

/// FN119g: the first round joins the front issued under the head (a stream wait; the arena's
/// ceiling lifted). Returns whether there was one.
fn front_under_join(rk: &mut Rank<'_, '_>, em: &mut Emitter<'_>) -> anyhow::Result<bool> {
    match rk.front_under.take() {
        Some(FrontUnder::Issued(ev)) => {
            rk.hip.stream_wait_event(&rk.dev.stream, &ev)?;
            em.arena.scratch.ceil = u64::MAX;
            em.arena.zero.ceil = u64::MAX;
            em.arena.ones.ceil = u64::MAX;
            Ok(true)
        }
        Some(FrontUnder::Want) => {
            anyhow::bail!("FN119g: a front wanted under the head never issued")
        }
        None => Ok(false),
    }
}

fn front_stage(
    rk: &mut Rank<'_, '_>,
    recs: &PackRecs,
    slot: u32,
    toks: &[u32],
    p0: u32,
    chunk: u32,
) -> anyhow::Result<ChunkConsts> {
    front_stage_at(rk, recs, slot, toks, p0, chunk, 0).map(|x| x.0)
}

/// [`front_stage`] at byte `base` of the early staging (two pieces staged at once); returns the
/// constants and the staging's next free byte.
#[allow(clippy::too_many_arguments)]
fn front_stage_at(
    rk: &mut Rank<'_, '_>,
    recs: &PackRecs,
    slot: u32,
    toks: &[u32],
    p0: u32,
    chunk: u32,
    base: usize,
) -> anyhow::Result<(ChunkConsts, usize)> {
    let m = toks.len() as u32;
    let e = embed_rows(recs, toks)?;
    buf_need(rk, m, "the embedding rows")?;
    let nb = (p0 + m).div_ceil(16) + 1;
    let (cb, offs) = ChunkConsts::host(m, p0, chunk, nb);
    let mut cc = ChunkConsts::at(
        rk.dev.consts.addr().as_u64(),
        &offs,
        m,
        p0,
        chunk,
        nb,
        rk.dev.cs.addr().as_u64(),
    );
    // the KV step law (its own copies on the stream) and the slots / page row
    let mut kv: Option<(Vec<u8>, Option<Vec<u8>>, u64)> = None;
    if let Some(k) = rk.pools.kvt.as_ref() {
        let mut k = k.borrow_mut();
        k.begin(rk.hip, &rk.dev.stream, &[(slot, p0, p0 + m)], true)?;
        let rows: Vec<(u32, u32)> = (0..m).map(|r| (slot, p0 + r)).collect();
        let (ks, _) = super::kvwire::step_tables_of(&k.tables(), &rows, false);
        ensure!(
            KV_PREFILL_OFF + ks.len() as u64 <= rk.dev.consts.len(),
            "a chunk's per-layer slots past the constants"
        );
        let bt = k.ix.as_ref().map(|ix| {
            ix.row(slot)[..cc.nb as usize]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        });
        kv = Some((ks, bt, u64::from(slot) * k.row_bytes()));
    }
    let a = rk.ahead.as_mut().context("FN119: the ahead")?;
    let st = a.early.as_mut().context("FN119e: the early buffers")?;
    let al = |x: usize| x.div_ceil(4096) * 4096;
    let o_e = base;
    let o_c = o_e + al(e.len());
    let o_k = o_c + al(cb.len());
    let (ks_len, bt_len) = kv.as_ref().map_or((0, 0), |(ks, bt, _)| {
        (ks.len(), bt.as_ref().map_or(0, Vec::len))
    });
    let o_b = o_k + al(ks_len);
    ensure!(
        o_b + bt_len <= st.tb.len(),
        "FN119e: the front's inputs ({} B) past the early staging",
        o_b + bt_len
    );
    let b = st.tb.as_mut_slice();
    b[o_e..o_e + e.len()].copy_from_slice(&e);
    b[o_c..o_c + cb.len()].copy_from_slice(&cb);
    if let Some((ks, bt, _)) = kv.as_ref() {
        b[o_k..o_k + ks.len()].copy_from_slice(ks);
        if let Some(bt) = bt {
            b[o_b..o_b + bt.len()].copy_from_slice(bt);
        }
    }
    let emb = rk.emb.clone();
    rk.hip.memcpy_h2d_at(
        &emb.own,
        emb.off,
        st.tb.slice(o_e, e.len()).context("FN119e: the rows")?,
        &rk.dev.stream,
    )?;
    rk.hip.memcpy_h2d_at(
        &rk.dev.consts,
        0,
        st.tb
            .slice(o_c, cb.len())
            .context("FN119e: the constants")?,
        &rk.dev.stream,
    )?;
    if let Some((ks, bt, row0)) = kv.as_ref() {
        rk.hip.memcpy_h2d_at(
            &rk.dev.consts,
            KV_PREFILL_OFF,
            st.tb.slice(o_k, ks.len()).context("FN119e: the slots")?,
            &rk.dev.stream,
        )?;
        if let Some(bt) = bt {
            let off = cc.bt - rk.dev.consts.addr().as_u64();
            rk.hip.memcpy_h2d_at(
                &rk.dev.consts,
                off,
                st.tb.slice(o_b, bt.len()).context("FN119e: the page row")?,
                &rk.dev.stream,
            )?;
        }
        cc.kslot = rk.dev.consts.addr().as_u64() + KV_PREFILL_OFF;
        cc.kv_row0 = *row0;
    }
    Ok((cc, al(o_b + bt_len)))
}

/// FN119e: the chunk's last logits row (this rank's half) into FN112's pinned staging (its event:
/// the dense fallback's read waits on it) before the front reuses the arena.
fn front_backup(rk: &mut Rank<'_, '_>, lg: u64) -> anyhow::Result<()> {
    let half = VOCAB as usize / 2;
    let base = rk.dev.arena.addr().as_u64();
    ensure!(
        lg >= base && lg - base + (half * 2) as u64 <= rk.dev.arena.len(),
        "FN119e: the logits row outside the arena"
    );
    let src = rk
        .dev
        .arena
        .addr()
        .offset(lg - base)
        .context("FN119e: the logits address")?;
    let o = rk.ov112.as_mut().context("FN112's staging")?;
    let at = o.lay.raw_off;
    rk.hip.memcpy_d2h_addr(
        o.stg
            .slice_mut(at, half * 2)
            .context("FN119e: the logits backup")?,
        src,
        &rk.dev.stream,
    )?;
    if rk.lg_backup.is_none() {
        rk.lg_backup = Some(rk.hip.create_event(EventFlags::Default)?);
    }
    let ev = rk.lg_backup.as_ref().context("FN119e: the backup event")?;
    rk.hip.record_event(ev, &rk.dev.stream)?;
    Ok(())
}

/// FN119e: the K-side front's kernels over the staged inputs ([`mtp_kv_rows`]'s one piece).
fn front_issue(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    slot: u32,
    mc: u32,
    cc: ChunkConsts,
) -> anyhow::Result<()> {
    let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
    front_issue_at(rk, em, slot, mc, cc, hid)
}

/// [`front_issue`] over the stream rows at `hid` (the chunk's, or the previous piece's stash).
fn front_issue_at(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    slot: u32,
    mc: u32,
    cc: ChunkConsts,
    hid: u64,
) -> anyhow::Result<()> {
    let st = rk.pools.seq(slot, LAYERS as usize);
    let t = Instant::now();
    lend_for(rk, em, mc, true, false)?;
    seg(rk, "lend", t);
    let t_l = Instant::now();
    let ev = (
        rk.hip.create_event(EventFlags::Default)?,
        rk.hip.create_event(EventFlags::Default)?,
    );
    rk.hip.record_event(&ev.0, &rk.dev.stream)?;
    let w = rk.mtp.as_ref().context("mtp")?.w;
    let n0 = em.forward_launches();
    let x = em.mtp_pre(&w, rk.emb.addr().as_u64(), hid, u64::from(mc))?;
    super::exec::emit_layer_kv_front(em, &w.lw, &st, x, &cc)?;
    let got = em.forward_launches() - n0;
    let want = saber_kernels::fn6::MTP_PRE_LAYER_LAUNCHES as u32
        + plan::declared_qsa_kv_front(
            em.rules,
            3,
            2,
            mc,
            saber_model::qwen4_exp::layer_graph::Form::Prefill,
        );
    ensure!(
        got == want,
        "FN119e: the MTP K-side front m {mc}: {got} launches vs declared {want}"
    );
    rk.dev.exec(&rk.table, em)?;
    rk.hip.record_event(&ev.1, &rk.dev.stream)?;
    pass_done(rk, ev, t_l, false)?;
    rk.f116_front += 1;
    Ok(())
}

/// The decode result: per sequence the picked ids and per-step merged top-20; per step the stats.
pub(super) struct Decoded {
    /// FN15b: per verify step, per sequence: the accepted drafts `a` (the step committed `a + 1`).
    pub(super) acc: Vec<Vec<u32>>,
    /// FN15b: per verify step, the batch's width.
    pub(super) widths: Vec<u32>,
    /// FN15: per sequence, per drafting point (0 = after the prefill), the chain's k drafts.
    mtp: Vec<Vec<Vec<u32>>>,
    /// FN15: per drafting point: (host ms, device ms of the passes).
    mtp_ms: Vec<(f64, f64)>,
    ids: Vec<Vec<u32>>,
    tops: Vec<Vec<Vec<(u32, f32)>>>,
    /// Per sequence per step: (the row's log-sum-exp, the logits at the reference's top-20 ids of
    /// that step -- the KL gate's probe; empty past the reference's steps).
    probes: Vec<Vec<(f32, Vec<f32>)>>,
    /// FN24 (`--fn-row-sha`): per sequence per generated token, the sha256 of the row that chose it.
    shas: Vec<Vec<Option<String>>>,
    steps: Vec<StepStat>,
}

/// The decode buckets a step replays (the 27B's B1 law: the smallest captured batch >= the live
/// count, the rest padding).
pub const BUCKETS: [u32; 4] = [1, 2, 4, 8];

/// The smallest bucket holding `n` live sequences.
#[must_use]
pub fn bucket_for(n: usize) -> u32 {
    BUCKETS
        .iter()
        .copied()
        .find(|&b| b as usize >= n)
        .unwrap_or(MAX_SEQS)
}

/// What a row of a pick asks beyond the greedy id: logits at `probes` (the KL gate's reference
/// ids, the sampler's penalty ids) and the log-sum-exp at `temp` (top_p's tempered mass; 0 = none).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RowReq {
    pub probes: Vec<u32>,
    pub temp: f32,
    /// The whole row's log-sum-exp is wanted (logprobs, a sampled row); off for a greedy row.
    #[serde(default)]
    pub lse: bool,
    /// FN119 (`--fn-device-step lazy`): the row's `top_k` when its `lse_t` is DEFERRED (only a tie
    /// from the cut to the seam's floor can keep the tail alive; the leader fills it late from the
    /// dense row when the merged seam says so, `dstep::late_rows`); 0 = today's law.
    #[serde(default)]
    pub tk: u32,
}

/// bf16 at index `i` of a little-endian bf16 row, as f32.
#[must_use]
pub fn bf16_at(row: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([row[2 * i], row[2 * i + 1]])) << 16)
}

/// The order-preserving integer key of a bf16 (larger key = larger value; -0 < +0).
fn bf16_key(b: u16) -> u16 {
    if b & 0x8000 != 0 {
        !b
    } else {
        b | 0x8000
    }
}

/// The `k` largest entries of a bf16 row as (index, value), value descending, index ascending on
/// a tie -- the total order of the old full sort, found in one pass: candidates at or above a
/// running threshold (the k-th key of the kept set) are buffered and cut back to `k` whenever the
/// buffer passes 4k.
#[must_use]
pub fn seam_top_k(row: &[u8], k: usize) -> Vec<(usize, f32)> {
    let n = row.len() / 2;
    let k = k.min(n).max(1);
    let mut buf: Vec<(u16, u32)> = Vec::with_capacity(8 * k);
    let mut thr = 0u16;
    for (i, c) in row.chunks_exact(2).enumerate() {
        let key = bf16_key(u16::from_le_bytes([c[0], c[1]]));
        if key >= thr {
            buf.push((key, i as u32));
            if buf.len() >= 4 * k {
                buf.select_nth_unstable_by(k - 1, |a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                buf.truncate(k);
                thr = buf.iter().map(|x| x.0).min().unwrap_or(0);
            }
        }
    }
    buf.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    buf.truncate(k);
    buf.iter()
        .map(|&(_, i)| (i as usize, bf16_at(row, i as usize)))
        .collect()
}

/// log-sum-exp of a bf16 row at temperature `t` (terms in f32 about the max, summed in f64).
#[must_use]
pub fn row_lse(row: &[u8], mx: f32, t: f32) -> f32 {
    let m = mx / t;
    let mut acc = 0f64;
    for c in row.chunks_exact(2) {
        let x = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
        acc += f64::from((x / t - m).exp());
    }
    (f64::from(m) + acc.ln()) as f32
}

/// One row of a pick, merged over the two vocabulary halves.
#[derive(Debug, Clone, Default)]
pub struct RowOut {
    /// The row's top-`SEAM_K` (id, raw logit), value descending, id ascending on a tie.
    pub top: Vec<(u32, f32)>,
    /// log-sum-exp of the whole raw row.
    pub lse: f32,
    /// log-sum-exp of the whole row at the request's temperature (None when `temp` was 0).
    pub lse_t: Option<f32>,
    /// The raw logit at each probe id.
    pub probe: Vec<f32>,
    /// FN24 (`--fn-row-sha`): sha256 (hex) of the rank-0 half's sha256 hex + the rank-1 half's --
    /// the whole row's bytes, fingerprinted.
    pub sha: Option<String>,
}

/// The seam's width (the 27B sampler's K: prod's top_k 20 / top_p 0.95 decide inside it).
pub const SEAM_K: usize = 64;

/// FN57: the device pick -- `fn_pick_top64_bf16` over `rows` rows of the logits at `lg` (one
/// workgroup a row), then ONE read of the 512 B a row it wrote: per row SEAM_K x (id + base, f32
/// bits), the host frame's seam bytes exactly.
fn device_top(rk: &mut Rank<'_, '_>, lg: u64, rows: usize) -> anyhow::Result<Vec<u8>> {
    let out = device_top_launch(rk, lg, rows)?;
    let dp = rk.dpick.as_ref().context("the device pick's region")?;
    d2h_in(&rk.dev, &dp.out, out, rows * SEAM_K * 8)
}

/// FN57 / FN112: [`device_top`]'s launch alone (the region's address returned; the caller reads
/// `rows x SEAM_K x 8` bytes there after the launch).
fn device_top_launch(rk: &mut Rank<'_, '_>, lg: u64, rows: usize) -> anyhow::Result<u64> {
    let half = VOCAB as usize / 2;
    let dp = rk.dpick.as_ref().context("the device pick's region")?;
    ensure!(
        rows <= dp.rows_max,
        "a pick of {rows} rows past the device pick's {} (MAX_SEQS x WMAX)",
        dp.rows_max
    );
    let out = dp.out.addr().as_u64();
    let (func, decl) = rk
        .table
        .get(PICK_PLUGIN, PICK_SYMBOL)
        .with_context(|| format!("{PICK_PLUGIN}::{PICK_SYMBOL} not loaded"))?;
    ensure!(
        decl.kernarg_segment_size == 32,
        "{PICK_SYMBOL}: the manifest's kernargs are {} B, the launch packs 32",
        decl.kernarg_segment_size
    );
    let mut args = Vec::with_capacity(32);
    args.extend_from_slice(&lg.to_le_bytes());
    args.extend_from_slice(&out.to_le_bytes());
    for v in [
        rows as i32,
        half as i32,
        half as i32,
        (rk.rank as usize * half) as i32,
    ] {
        args.extend_from_slice(&v.to_le_bytes());
    }
    rk.hip.launch(
        func,
        [rows as u32, 1, 1],
        [256, 1, 1],
        0,
        &rk.dev.stream,
        &mut args,
    )?;
    let dp = rk.dpick.as_mut().context("the device pick's region")?;
    dp.rows += rows as u64;
    Ok(out)
}

/// FN112 (`--fn-overlap on`): queue the step's token-independent reads behind its graph, before
/// the host waits -- the device seam (the same launch [`pick2`] makes), the rows the requests'
/// read law needs ([`pick2`]'s `needs_row`: request parameters only) and the route log of a
/// complete tier's end (`log_m`) -- each a D2H into the staging on the step's stream, then the
/// arm's event. A no-op without the arm.
fn ov112_queue(
    rk: &mut Rank<'_, '_>,
    lg: u64,
    rows: usize,
    reqs: &[RowReq],
    log_m: Option<u32>,
) -> anyhow::Result<()> {
    let Some(lay) = rk.ov112.as_ref().map(|o| o.lay) else {
        return Ok(());
    };
    ensure!(
        rows <= lay.rows_max && log_m.is_none_or(|m| m as usize <= lay.rows_max),
        "FN112: a step of {rows} pick rows / {log_m:?} log rows past the staging's {}",
        lay.rows_max
    );
    let seam = rk.dpick.is_some();
    let raw = !seam || (0..rows).any(|r| pick_needs_row(rk.row_sha, reqs, r));
    let seam_at = if seam {
        Some(device_top_launch(rk, lg, rows)?)
    } else {
        None
    };
    let o = rk.ov112.as_mut().context("FN112: the arm")?;
    if let Some(at) = seam_at {
        let dst = o
            .stg
            .slice_mut(lay.seam_off, rows * lay.seam_row)
            .context("FN112: the seam staging")?;
        let dp = rk.dpick.as_ref().context("the device pick's region")?;
        ensure!(at == dp.out.addr().as_u64(), "FN112: the seam region moved");
        rk.hip.memcpy_d2h_addr(dst, dp.out.addr(), &rk.dev.stream)?;
        o.seam_rows += rows as u64;
    }
    if let Some(m) = log_m {
        let n = lay.log_bytes(m as usize);
        ensure!(
            n as u64 <= rk.idlog.len,
            "FN112: the route log past its buffer"
        );
        let dst = o
            .stg
            .slice_mut(lay.log_off, n)
            .context("FN112: the log staging")?;
        rk.hip
            .memcpy_d2h_addr(dst, rk.idlog.addr(), &rk.dev.stream)?;
        o.logs += 1;
    }
    if raw {
        let n = rows * lay.raw_row;
        let base = rk.dev.arena.addr().as_u64();
        ensure!(
            lg >= base && lg - base + n as u64 <= rk.dev.arena.len(),
            "FN112: the logits read outside the arena"
        );
        let src = rk
            .dev
            .arena
            .addr()
            .offset(lg - base)
            .context("FN112: the logits address")?;
        let dst = o
            .stg
            .slice_mut(lay.raw_off, n)
            .context("FN112: the rows staging")?;
        rk.hip.memcpy_d2h_addr(dst, src, &rk.dev.stream)?;
        o.raw_rows += rows as u64;
    }
    rk.hip.record_event(&o.ev, &rk.dev.stream)?;
    o.steps += 1;
    o.pend = Some(super::overlap::Pend {
        lg,
        rows,
        seam,
        raw,
        log_m,
    });
    Ok(())
}

/// FN56 / FN112: whether a pick reads row `r` whole (the row sha arm, a sampled row, a logprob row,
/// a probe row) -- a function of the requests only.
fn pick_needs_row(row_sha: bool, reqs: &[RowReq], r: usize) -> bool {
    row_sha
        || reqs
            .get(r)
            .is_some_and(|q| super::dstep::wants(q).0 || !q.probes.is_empty())
}

/// FN57: one self-test row of pattern `p` (seeded by `seed`), `n` bf16 values: the cases a seam
/// can get wrong -- ordinary logits, a whole row tied, a thousand ties at the cut, one value so far
/// above the rest that the window holds < 64 (the full radix select), every bit pattern (NaN, inf,
/// subnormals, -0), all negative, exactly 64 finite values among -inf, +0 / -0 ties.
#[must_use]
pub fn pick_selftest_row(p: usize, seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    let mut gauss = || {
        let mut s = 0f32;
        for _ in 0..4 {
            s += (next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        }
        s * 6.0
    };
    let v: Vec<u16> = match p % 8 {
        0 => (0..n).map(|_| bf(gauss())).collect(),
        1 => vec![bf(1.5); n],
        2 => {
            let mut v: Vec<u16> = (0..n).map(|_| bf(gauss().min(5.0))).collect();
            for _ in 0..1000 {
                let i = (next() as usize) % n;
                v[i] = bf(7.0);
            }
            v
        }
        3 => {
            let mut v: Vec<u16> = (0..n).map(|_| bf(gauss() * 0.01)).collect();
            v[(next() as usize) % n] = bf(3.0e38);
            v
        }
        4 => (0..n).map(|_| next() as u16).collect(),
        5 => (0..n).map(|_| bf(-gauss().abs() - 1.0)).collect(),
        6 => {
            let mut v = vec![0xff80u16; n];
            for k in 0..64 {
                v[(k * 1931 + (next() as usize) % 1931) % n] = bf(k as f32 + 0.5);
            }
            v
        }
        _ => {
            let mut v: Vec<u16> = (0..n)
                .map(|i| if i % 2 == 0 { 0x0000 } else { 0x8000 })
                .collect();
            for _ in 0..17 {
                v[(next() as usize) % n] = bf(2.0);
            }
            v
        }
    };
    v.iter().flat_map(|b| b.to_le_bytes()).collect()
}

/// FN57: the device pick's BOOT SELF-TEST -- `MAX_SEQS x WMAX` rows of [`pick_selftest_row`]'s
/// eight patterns uploaded to a transient buffer, picked on the card, compared with the host seam
/// row for row (ids and value bits). A difference refuses the boot.
fn pick_selftest(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    let half = VOCAB as usize / 2;
    let rows = rk.dpick.as_ref().context("the device pick")?.rows_max;
    let mut host = Vec::with_capacity(rows * half * 2);
    for r in 0..rows {
        host.extend_from_slice(&pick_selftest_row(r, 57 + r as u64, half));
    }
    let buf = rk.hip.malloc(host.len() as u64)?;
    rk.hip.memcpy_h2d_blocking(&buf, &host, &rk.dev.stream)?;
    let d = device_top(rk, buf.addr().as_u64(), rows)?;
    let base = rk.rank as usize * half;
    let mut bad = Vec::new();
    for r in 0..rows {
        let h = seam_top_k(&host[r * half * 2..(r + 1) * half * 2], SEAM_K);
        let same = (0..SEAM_K).all(|k| {
            let at = r * SEAM_K * 8 + k * 8;
            let id = u32::from_le_bytes(d[at..at + 4].try_into().expect("4")) as usize;
            let v = u32::from_le_bytes(d[at + 4..at + 8].try_into().expect("4"));
            h[k].0 + base == id && h[k].1.to_bits() == v
        });
        if !same {
            bad.push(r);
        }
    }
    drop(buf);
    if let Some(dp) = rk.dpick.as_mut() {
        dp.rows = 0;
    }
    println!(
        "fnpick57 selftest rank{}: {}/{rows} rows identical to the host seam (8 patterns: logits, \
         all tied, 1,000 ties at the cut, the radix fallback, every bit pattern, negatives, 64 among \
         -inf, +0/-0)",
        rk.rank,
        rows - bad.len()
    );
    ensure!(
        bad.is_empty(),
        "FN57: the device pick differs from the host seam on self-test rows {bad:?}"
    );
    Ok(())
}

/// FN57: the device pick's code object and symbol.
pub const PICK_PLUGIN: &str = "flashnext_pick";
/// FN57: see [`PICK_PLUGIN`].
pub const PICK_SYMBOL: &str = "fn_pick_top64_bf16";

/// FN57: [`row_lse`] with each distinct bf16's term computed ONCE a row (a row of 124,160 logits
/// holds a few thousand distinct values): the same f32 term for the same input, summed in the
/// same order into the same f64 -- the same bytes by construction (gated on the host against
/// [`row_lse`]), at a fraction of the `exp` calls.
#[must_use]
pub fn row_lse_memo(row: &[u8], mx: f32, t: f32) -> f32 {
    thread_local! {
        static MEMO: std::cell::RefCell<(u32, Vec<u32>, Vec<f32>)> =
            std::cell::RefCell::new((0, vec![0; 65536], vec![0.0; 65536]));
    }
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        let (gen, stamp, val) = &mut *m;
        *gen = gen.wrapping_add(1);
        if *gen == 0 {
            stamp.iter_mut().for_each(|s| *s = 0);
            *gen = 1;
        }
        let g = *gen;
        let mm = mx / t;
        let mut acc = 0f64;
        for c in row.chunks_exact(2) {
            let b = usize::from(u16::from_le_bytes([c[0], c[1]]));
            let e = if stamp[b] == g {
                val[b]
            } else {
                let x = f32::from_bits((b as u32) << 16);
                let e = (x / t - mm).exp();
                stamp[b] = g;
                val[b] = e;
                e
            };
            acc += f64::from(e);
        }
        (f64::from(mm) + acc.ln()) as f32
    })
}

/// Each rank's top-`SEAM_K`, log-sum-exps and probe logits over its vocabulary half per row,
/// swapped, merged.
///
/// FN57 (`--fn-pick device`): the seam comes from the card ([`device_top`]); the rows are read
/// only when some row asks for what only its logits answer (a log-sum-exp -- logprobs or a
/// sampled row --, probes, the row sha), and then the log-sum-exps take [`row_lse_memo`]. With the
/// row sha on (the gate arm) every row is read and its device seam CHECKED against the host's.
pub(super) fn pick2(
    rk: &mut Rank<'_, '_>,
    lg: u64,
    rows: usize,
    reqs: &[RowReq],
) -> anyhow::Result<Vec<RowOut>> {
    let half = VOCAB as usize / 2;
    let base = rk.rank as usize * half;
    let t0 = Instant::now();
    let row_sha = rk.row_sha;
    let needs_row = |r: usize| pick_needs_row(row_sha, reqs, r);
    let seam = rk.dpick.is_some();
    let read = !seam || (0..rows).any(needs_row);
    // FN112 (`--fn-overlap on`): the seam and the rows queued behind the step -- the step's one
    // wait covered them; a pick that is not the queued one runs the control's reads
    let pre = rk
        .ov112
        .as_mut()
        .is_some_and(|o| o.take_pick(lg, rows, seam, read));
    let dtop = if !seam {
        None
    } else if pre {
        let o = rk.ov112.as_ref().context("FN112: the arm")?;
        let at = o.lay.seam_off;
        Some(o.stg.as_slice()[at..at + rows * SEAM_K * 8].to_vec())
    } else {
        Some(device_top(rk, lg, rows)?)
    };
    let raw_own = if read && !pre {
        d2h_in(&rk.dev, &rk.dev.arena, lg, rows * half * 2)?
    } else {
        Vec::new()
    };
    let raw: &[u8] = if read && pre {
        let o = rk.ov112.as_ref().context("FN112: the arm")?;
        let at = o.lay.raw_off;
        &o.stg.as_slice()[at..at + rows * half * 2]
    } else {
        &raw_own
    };
    let t1 = Instant::now();
    let memo = dtop.is_some();
    let checked = std::sync::atomic::AtomicU64::new(0);
    let differ = std::sync::atomic::AtomicU64::new(0);
    // one row's frame: its seam (id, logit) x SEAM_K, lse, lse at t, [sha], probes
    let frame = |r: usize, out: &mut Vec<u8>| {
        let row: &[u8] = if read {
            &raw[r * half * 2..(r + 1) * half * 2]
        } else {
            &[]
        };
        let rq = reqs.get(r).cloned().unwrap_or_default();
        let top: Vec<(usize, f32)> = match dtop.as_deref() {
            Some(d) => {
                let o = r * SEAM_K * 8;
                let t: Vec<(usize, f32)> = (0..SEAM_K)
                    .map(|k| {
                        let at = o + k * 8;
                        let id = u32::from_le_bytes(d[at..at + 4].try_into().expect("4"));
                        let v = f32::from_le_bytes(d[at + 4..at + 8].try_into().expect("4"));
                        ((id as usize).wrapping_sub(base), v)
                    })
                    .collect();
                if row_sha {
                    // the gate arm: the card's seam == the host's, id and value bits
                    checked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let h = seam_top_k(row, SEAM_K);
                    let same = h.len() == t.len()
                        && h.iter()
                            .zip(&t)
                            .all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits());
                    if !same {
                        differ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                t
            }
            None => seam_top_k(row, SEAM_K),
        };
        for &(i, v) in &top {
            out.extend_from_slice(&((i + base) as u32).to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mx = top[0].1;
        let lse_of = |t: f32| {
            if memo {
                row_lse_memo(row, mx, t)
            } else {
                row_lse(row, mx, t)
            }
        };
        // FN119: `dstep::wants` -- today's law when `rq.tk == 0` (a deferred row's lse_t is
        // filled late by the leader, its T = 1 log-sum-exp only when logprobs ask)
        let (w1, wt) = super::dstep::wants(&rq);
        let (lse, lt) = if w1 {
            (lse_of(1.0), if wt { lse_of(rq.temp) } else { f32::NAN })
        } else {
            (f32::NAN, f32::NAN)
        };
        out.extend_from_slice(&lse.to_le_bytes());
        out.extend_from_slice(&lt.to_le_bytes());
        if row_sha {
            out.extend_from_slice(saber_kernels::manifest::sha256_hex(row).as_bytes());
        }
        for &id in &rq.probes {
            let i = id as usize;
            let x = if i >= base && i < base + half {
                bf16_at(row, i - base)
            } else {
                f32::NAN
            };
            out.extend_from_slice(&x.to_le_bytes());
        }
    };
    let mut mine: Vec<u8> = Vec::with_capacity(rows * (SEAM_K * 8 + 16));
    let par = if rk.lev.pick { pick_threads(rows) } else { 1 };
    if par > 1 {
        // FN56 (`pick`): contiguous row groups on scoped threads, concatenated in row order (each
        // row's frame is a function of its row only: the same bytes as the serial loop). FN57
        // (`--fn-pin auto`): each thread pins itself to the rank's pick CPUs first
        let per = rows.div_ceil(par);
        let cpus = rk.pick_cpus.clone();
        let parts: Vec<Vec<u8>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..rows)
                .step_by(per)
                .map(|r0| {
                    let frame = &frame;
                    let cpus = cpus.as_deref();
                    sc.spawn(move || {
                        if let Some(c) = cpus {
                            // a refusal leaves the thread where the scheduler put it (counted
                            // nowhere: the boot already proved the law on this box)
                            let _ = saber_hip::affinity::pin_current_thread(c);
                        }
                        let mut v = Vec::with_capacity(per * (SEAM_K * 8 + 16));
                        for r in r0..(r0 + per).min(rows) {
                            frame(r, &mut v);
                        }
                        v
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("a pick thread panicked"))
                .collect()
        });
        for p in parts {
            mine.extend_from_slice(&p);
        }
    } else {
        for r in 0..rows {
            frame(r, &mut mine);
        }
    }
    if let Some(d) = rk.dpick.as_mut() {
        d.checked += checked.into_inner();
        d.differ += differ.into_inner();
        ensure!(
            d.differ == 0,
            "FN57: the device pick's seam differs from the host seam on {} checked rows",
            d.differ
        );
    }
    let t2 = Instant::now();
    // FN117b (S6): the shared pick line under `--fn-hand s6` (the socket = the control)
    let theirs = hand_xchg(rk, &mine)?;
    ensure!(
        theirs.len() == mine.len(),
        "the peer's pick frame differs in size"
    );
    let rdu = |b: &[u8], k: usize| u32::from_le_bytes(b[k..k + 4].try_into().expect("4"));
    let rdf = |b: &[u8], k: usize| f32::from_le_bytes(b[k..k + 4].try_into().expect("4"));
    let mut outs = Vec::with_capacity(rows);
    let mut at = 0usize;
    for r in 0..rows {
        let rq = reqs.get(r).cloned().unwrap_or_default();
        let mut all: Vec<(u32, f32)> = Vec::with_capacity(2 * SEAM_K);
        for k in 0..SEAM_K {
            let o = at + k * 8;
            all.push((rdu(&mine, o), rdf(&mine, o + 4)));
            all.push((rdu(&theirs, o), rdf(&theirs, o + 4)));
        }
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(SEAM_K);
        at += SEAM_K * 8;
        let (a, b) = (rdf(&mine, at), rdf(&theirs, at));
        let lse = if a.is_nan() {
            f32::NAN
        } else {
            saber_serve::sampler::log_add_exp(a, b)
        };
        at += 4;
        let (a, b) = (rdf(&mine, at), rdf(&theirs, at));
        let lse_t = (!a.is_nan()).then(|| saber_serve::sampler::log_add_exp(a, b));
        at += 4;
        let sha = if rk.row_sha {
            let (lo, hi) = if rk.rank == 0 {
                (&mine[at..at + 64], &theirs[at..at + 64])
            } else {
                (&theirs[at..at + 64], &mine[at..at + 64])
            };
            at += 64;
            Some(saber_kernels::manifest::sha256_hex(&[lo, hi].concat()))
        } else {
            None
        };
        let mut probe = Vec::with_capacity(rq.probes.len());
        for _ in &rq.probes {
            let (a, b) = (rdf(&mine, at), rdf(&theirs, at));
            probe.push(if a.is_nan() { b } else { a });
            at += 4;
        }
        outs.push(RowOut {
            top: all,
            lse,
            lse_t,
            probe,
            sha,
        });
    }
    rk.pick_t = [
        (t1 - t0).as_secs_f64() * 1e3,
        (t2 - t1).as_secs_f64() * 1e3,
        t2.elapsed().as_secs_f64() * 1e3,
    ];
    if pre {
        if let Some(o) = rk.ov112.as_mut() {
            o.host_ms += rk.pick_t[0];
        }
    }
    let a = &mut rk.pick_acc;
    a.0 += 1;
    a.1 += rows as u64;
    a.2 += if read { rows as u64 } else { 0 };
    a.3 += rk.pick_t[0];
    a.4 += rk.pick_t[1];
    a.5 += rk.pick_t[2];
    Ok(outs)
}

/// FN56 (`pick`): host threads for a pick of `rows` rows -- one per two rows, at most 6 a rank
/// (the pair's two ranks then hold 12 of the host's 24 threads beside their spinning HIP threads).
fn pick_threads(rows: usize) -> usize {
    rows.div_ceil(2).clamp(1, 6)
}

/// One captured decode step: the graph of a (bucket, PLE parity) and what the capture issued.
struct StepGraph {
    /// The segments in launch order: one per decoder layer + the final mixer / head
    /// (`--fn-graph-split layer`, the default), or the whole step (`step`).
    segs: Vec<(saber_hip::GraphExec, saber_hip::Graph)>,
    digest: u64,
    nodes: usize,
    launches: u64,
    memsets: u64,
    copies: u64,
    lg: u64,
}

/// A step form's graph key: (bucket, width, PLE parity, FN19's short form, the context band).
type GKey = (u32, u32, usize, bool, u32);

/// FN64: one captured run of layers `[a, b)` (or the head, `(LAYERS, LAYERS + 1)`).
struct LazyRun {
    exec: saber_hip::GraphExec,
    _graph: saber_hip::Graph,
    /// The records' digest at capture (a dry re-emission must match it).
    digest: u64,
    launches: u64,
    /// The capture pass (tick) that last used the run (the LRU's order).
    last: u64,
    /// FN64c: the served steps that replayed the run (`--fn-graph-lazy-record`'s use count).
    uses: u64,
    /// FN64d: captured by the boot's pre-capture (the census's held-per-form split).
    pre: bool,
    /// FN98 (`--fn-graph-form one`): the run's gate blocks `(start, end)` -- a `fn_gate98`
    /// launch before each block's first layer reads the gate table's entry `start`.
    gates: Vec<(u32, u32)>,
}

/// FN64: the runs captured for one step form.
#[derive(Default)]
struct LazyKey {
    runs: BTreeMap<(u32, u32), LazyRun>,
    /// The logits' address (the head run's product).
    lg: u64,
    /// FN64b: a served step of this form has re-emitted its runs dry against the captures (the
    /// pre-capture's runs are checked against the first real step of their form).
    seen: bool,
    /// FN64f: this form's served lazy steps and the served capture passes among them (the
    /// census's per-band split: the short mix vs the long rows).
    served: u64,
    passes: u64,
}

/// The decode graphs of one rank (FN14): captured on first use, replayed after.
#[derive(Default)]
pub(super) struct Graphs {
    /// (bucket, width, PLE parity, FN19's short form)
    map: BTreeMap<GKey, StepGraph>,
    captured: u64,
    replays: u64,
    eager: u64,
    verified: u64,
    capture_ms: f64,
    /// FN64 (`--fn-graph-split lazy`): the runs per step form.
    lazy: BTreeMap<GKey, LazyKey>,
    /// FN64: the step counter of the lazy LRU.
    tick: u64,
    /// FN64: runs captured / evicted, capture passes (a step that captured at least one run),
    /// held runs re-emitted dry and checked.
    lazy_captured: u64,
    lazy_evicted: u64,
    lazy_passes: u64,
    lazy_checked: u64,
    /// FN64: graph launches and gated layers (copies in flight at the step's gates) over the
    /// replayed steps -- the census's launches a step.
    seg_launches: u64,
    gated: u64,
    /// The rank (the summary line's).
    rank: Option<u32>,
    /// The split's name (the summary line's).
    split: &'static str,
    /// FN64b: the boot's pre-capture -- runs captured, forms, capture passes, ms, forms left to
    /// the on-demand path (past the cache).
    pre_runs: u64,
    pre_forms: u64,
    pre_passes: u64,
    pre_ms: f64,
    pre_skipped: u64,
    /// FN64c: `--fn-graph-lazy-record`'s directory (the served steps' runs written there).
    record: Option<std::path::PathBuf>,
    /// FN64c: q / the PLE cut of the boot (the record's header; a recorded set of another
    /// grid is refused).
    lazy_grid: (u32, Option<u32>),
    /// FN64d: the billed exec budget's run cap (`--fn-graph-exec-budget-mib` / 2 MiB), `None`
    /// unbilled.
    exec_cap: Option<usize>,
    /// FN64e: the recorded runs the boot pre-captures (most used first), `None` = every recorded
    /// run within the cache and the free-VRAM floor (FN64c, unbilled).
    pre_cap: Option<usize>,
    /// FN64e: the served uses of the runs the LRU evicted, per form (the record keeps them).
    retired: BTreeMap<GKey, BTreeMap<(u32, u32), u64>>,
    /// FN64d / FN64e: how this boot's execs are billed (the census line names it).
    bill: super::run::ExecBill,
    /// FN64d: under `--fn-graph-lazy-record`, every served lazy step's (form, cut mask: bit `a`
    /// for each run `[a, b)` it replays) -- the trace an exec budget's LRU is replayed on
    /// (`lazy_steps.rank<r>.json`, written beside the record).
    trace: Vec<(GKey, u64)>,
    /// FN64d: the drafter's held graph execs (MTP passes + device chains), refreshed each replay.
    mtp_execs: u64,
    /// FN64d: the device's free VRAM (MiB, hipMemGetInfo) at the last census.
    free_mib: u64,
    /// FN98 (`--fn-graph-form one`): the gate quantum (layers a gate block spans), `None` = the
    /// lazy run split.
    one: Option<u32>,
    /// FN98: gate launches captured inside the execs, gate entries armed (blocks whose copies
    /// were in flight at a launch), gated layers among them, the execs' VRAM at capture +
    /// instantiate (hipMemGetInfo delta, bytes) and the execs captured.
    one_nodes: u64,
    one_sets: u64,
    one_gated: u64,
    one_bytes: u64,
    one_execs: u64,
    /// FN98: `--fn-graph-form one-open` (the gate's negative control): gates never armed.
    one_open: bool,
    /// FN98: the gate table (device, read by every `fn_gate98` of the step's execs) and its pinned
    /// staging (written by the host, copied in order on the step's stream before the execs):
    /// [`LAYERS`] entries of 16 bytes (word address u64, value u32, pad), indexed by block start.
    gate_tab: Option<(DevicePtr, saber_hip::PinnedBuf)>,
}

impl Drop for Graphs {
    fn drop(&mut self) {
        if self.replays > 0 {
            println!("{}", self.line());
            if self.one.is_some() {
                println!("{}", self.line98());
            }
            for l in self.form_lines() {
                println!("{l}");
            }
        }
        if let Err(e) = self.write_record() {
            println!(
                "fngraph64c rank{}: the run record was NOT written: {e:#}",
                self.rank.unwrap_or(0)
            );
        }
    }
}

/// FN64c: the recorded lazy run set's format tag (`lazy_runs.rank<r>.json`).
const LAZY_SET_FORMAT: &str = "fn64c-lazy-runs/1";

/// FN64d: the served lazy steps' trace format tag (`lazy_steps.rank<r>.json`).
const LAZY_TRACE_FORMAT: &str = "fn64d-lazy-steps/1";

/// FN64c: a recorded set -- per step form, its runs with their served use counts.
type LazySet = BTreeMap<GKey, Vec<((u32, u32), u64)>>;

/// FN64c: read rank `rank`'s recorded set; refuses another rank's, another grid's (q / PLE cut)
/// or another format's file by name.
fn read_lazy_set(path: &Path, rank: u32, grid: (u32, Option<u32>)) -> anyhow::Result<LazySet> {
    let t = std::fs::read_to_string(path)
        .with_context(|| format!("FN64c: the recorded lazy run set {}", path.display()))?;
    let v: Value = serde_json::from_str(&t).with_context(|| format!("{}", path.display()))?;
    ensure!(
        v["format"] == LAZY_SET_FORMAT,
        "{}: format {} is not {LAZY_SET_FORMAT}",
        path.display(),
        v["format"]
    );
    ensure!(
        v["rank"].as_u64() == Some(u64::from(rank)),
        "{}: rank {} is not this rank ({rank})",
        path.display(),
        v["rank"]
    );
    let ple = v["ple"].as_u64().map(|x| x as u32);
    ensure!(
        v["q"].as_u64() == Some(u64::from(grid.0)) && ple == grid.1,
        "{}: recorded at q {} / PLE cut {:?}, this boot runs q {} / {:?}",
        path.display(),
        v["q"],
        ple,
        grid.0,
        grid.1
    );
    let mut out = LazySet::new();
    for fm in v["forms"].as_array().context("forms")? {
        let g = |k: &str| fm[k].as_u64().with_context(|| format!("form key {k}"));
        let key: GKey = (
            g("bucket")? as u32,
            g("width")? as u32,
            g("parity")? as usize,
            fm["short"].as_bool().context("form key short")?,
            g("band")? as u32,
        );
        let mut runs = Vec::new();
        for r in fm["runs"].as_array().context("runs")? {
            let x = |i: usize| r[i].as_u64().context("run [a, b, uses]");
            runs.push(((x(0)? as u32, x(1)? as u32), x(2)?));
        }
        out.insert(key, runs);
    }
    Ok(out)
}

impl Graphs {
    /// FN64c (`--fn-graph-lazy-record`): write the runs the SERVED steps replayed (use count >
    /// 0), per form, to the record directory (temp file + rename: a reader never sees half).
    fn write_record(&self) -> anyhow::Result<()> {
        let Some(dir) = self.record.as_ref() else {
            return Ok(());
        };
        let rank = self.rank.unwrap_or(0);
        // FN64e: the held runs' uses + the evicted runs' (FN64c's record dropped an evicted run's
        // uses: a boot whose LRU turned over recorded only its last working set)
        let mut all: BTreeMap<GKey, BTreeMap<(u32, u32), u64>> = self.retired.clone();
        for (k, l) in &self.lazy {
            for (s, r) in &l.runs {
                *all.entry(*k).or_default().entry(*s).or_default() += r.uses;
            }
        }
        let forms: Vec<Value> = all
            .iter()
            .filter(|(&(_, _, p, _, _), _)| p < 2)
            .filter_map(|(&(b, w, p, sh, band), k)| {
                let mut runs: Vec<(&(u32, u32), u64)> = k
                    .iter()
                    .filter(|(_, u)| **u > 0)
                    .map(|(s, u)| (s, *u))
                    .collect();
                runs.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(y.0)));
                (!runs.is_empty()).then(|| {
                    json!({"bucket": b, "width": w, "parity": p, "short": sh, "band": band,
                        "runs": runs.iter().map(|(s, u)| json!([s.0, s.1, u])).collect::<Vec<_>>()})
                })
            })
            .collect();
        let n: usize = forms
            .iter()
            .map(|f| f["runs"].as_array().map_or(0, Vec::len))
            .sum();
        let v = json!({"format": LAZY_SET_FORMAT, "rank": rank, "q": self.lazy_grid.0,
            "ple": self.lazy_grid.1, "replays": self.replays, "forms": forms});
        std::fs::create_dir_all(dir)?;
        let path = super::run::lazy_set_file(dir, rank);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&v)?)?;
        std::fs::rename(&tmp, &path)?;
        // FN64d: the served steps' trace (forms indexed; a step = [form, cut mask])
        let keys: Vec<GKey> = self
            .trace
            .iter()
            .map(|t| t.0)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let tv = json!({"format": LAZY_TRACE_FORMAT, "rank": rank, "q": self.lazy_grid.0,
            "ple": self.lazy_grid.1,
            "forms": keys.iter().map(|k| json!([k.0, k.1, k.2, k.3, k.4])).collect::<Vec<_>>(),
            "steps": self.trace.iter().map(|(k, m)| json!([keys.binary_search(k).unwrap_or(0), m])).collect::<Vec<_>>()});
        let tp = dir.join(format!("lazy_steps.rank{rank}.json"));
        let tt = tp.with_extension("json.tmp");
        std::fs::write(&tt, serde_json::to_vec(&tv)?)?;
        std::fs::rename(&tt, &tp)?;
        println!(
            "fngraph64c rank{rank}: recorded {n} served runs of {} forms after {} replays -> {}",
            forms.len(),
            self.replays,
            path.display()
        );
        Ok(())
    }

    /// FN64d: the exec census -- per step form, the lazy runs held (pre-captured / on demand),
    /// the runs its served steps replayed, its steps (the head's uses) and run uses; then the
    /// rank's execs (lazy + per-layer/step graphs + the drafter's) at 2 MiB each.
    fn form_lines(&self) -> Vec<String> {
        let r = self.rank.unwrap_or(0);
        let mut v: Vec<String> = self
            .lazy
            .iter()
            .map(|(&(b, w, p, sh, band), k)| {
                let pre = k.runs.values().filter(|x| x.pre).count();
                let used = k.runs.values().filter(|x| x.uses > 0).count();
                // the head's run (FN98: the exec that carries the head)
                let steps = k
                    .runs
                    .iter()
                    .find(|(s, _)| s.1 > LAYERS)
                    .map_or(0, |(_, x)| x.uses);
                let uses: u64 = k.runs.values().map(|x| x.uses).sum();
                format!(
                    "fngraph64d rank{r}: form b{b} w{w} p{p} short {sh} band {band}: held {} (pre {pre}, on demand {}), used {used}, steps {steps}, run uses {uses}",
                    k.runs.len(),
                    k.runs.len() - pre
                )
            })
            .collect();
        // FN64f: served steps and capture passes by band (the short mix vs the long rows)
        let mut bands: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
        for (k, f) in &self.lazy {
            let e = bands.entry(k.4).or_default();
            e.0 += f.served;
            e.1 += f.passes;
        }
        for (band, (n, p)) in bands {
            v.push(format!(
                "fngraph64f rank{r}: band {band}: {n} served lazy steps, {p} served capture passes ({:.2} per 1,000)",
                1000.0 * p as f64 / n.max(1) as f64
            ));
        }
        let lazy: usize = self.lazy.values().map(|k| k.runs.len()).sum();
        let segs: usize = self.map.values().map(|g| g.segs.len()).sum();
        let all = lazy as u64 + segs as u64 + self.mtp_execs;
        v.push(format!(
            "fngraph64d rank{r}: execs held {all} = lazy {lazy} + step/layer graphs {segs} + drafter {} = {} MiB of kernarg pools ({} MiB a exec) | exec budget {} | evicted {} | VRAM free {} MiB (last census)",
            self.mtp_execs,
            all * super::run::GRAPH_EXEC_POOL_MIB,
            super::run::GRAPH_EXEC_POOL_MIB,
            self.exec_cap.map_or("unbilled".to_string(), |c| format!(
                "{c} runs ({} MiB, {:?})",
                c as u64 * super::run::GRAPH_EXEC_POOL_MIB,
                self.bill
            )),
            self.lazy_evicted,
            self.free_mib
        ));
        v
    }

    /// FN98: the `one` form's census line (`fngraph98`): execs, their bytes at instantiate, the
    /// gates inside, the node updates (blocks with copies in flight at a launch).
    fn line98(&self) -> String {
        let n = self.replays.max(1) as f64;
        let held: usize = self.lazy.values().map(|k| k.runs.len()).sum();
        format!(
            "fngraph98 rank{}: one graph per form (q {}): {} forms, {} execs held ({} captured) + drafter {} | {:.1} MiB at capture + instantiate ({:.2} MiB an exec) | {} fn_gate98 launches inside | {} replays: {:.2} graph launches/step, {:.2} gates armed/step, {:.2} gated layers/step | capture passes {} (pre-capture {}), served {}",
            self.rank.unwrap_or(0),
            self.one.unwrap_or(0),
            self.lazy.len(),
            held,
            self.one_execs,
            self.mtp_execs,
            self.one_bytes as f64 / f64::from(1u32 << 20),
            self.one_bytes as f64 / f64::from(1u32 << 20) / self.one_execs.max(1) as f64,
            self.one_nodes,
            self.replays,
            self.seg_launches as f64 / n,
            self.one_sets as f64 / n,
            self.one_gated as f64 / n,
            self.lazy_passes,
            self.pre_passes,
            self.lazy_passes - self.pre_passes
        )
    }

    /// FN64: the census line -- graph launches and gated layers a replayed step, the lazy runs.
    fn line(&self) -> String {
        let n = self.replays.max(1) as f64;
        format!(
            "fngraph64 rank{}: split {} | {} replays, {:.2} graph launches/step, {:.2} gated layers/step | lazy runs held {} captured {} evicted {} checked {} in {} capture passes | capture {:.1} ms total | pre-captured {} runs of {} forms in {} passes {:.0} ms ({} forms on demand), missed {} in {} served capture passes",
            self.rank.unwrap_or(0),
            self.split,
            self.replays,
            self.seg_launches as f64 / n,
            self.gated as f64 / n,
            self.lazy.values().map(|k| k.runs.len()).sum::<usize>(),
            self.lazy_captured,
            self.lazy_evicted,
            self.lazy_checked,
            self.lazy_passes,
            self.capture_ms,
            self.pre_runs,
            self.pre_forms,
            self.pre_passes,
            self.pre_ms,
            self.pre_skipped,
            self.lazy_captured - self.pre_runs,
            self.lazy_passes - self.pre_passes
        )
    }

    fn census(&self) -> Value {
        json!({"captured": self.captured, "replays": self.replays, "eager_steps": self.eager,
            "verified": self.verified, "capture_ms": self.capture_ms,
            "split": self.split, "graph_launches": self.seg_launches, "gated_layers": self.gated,
            "lazy": {"held": self.lazy.values().map(|k| k.runs.len()).sum::<usize>(),
                "captured": self.lazy_captured, "evicted": self.lazy_evicted,
                "passes": self.lazy_passes, "checked": self.lazy_checked,
                "pre_runs": self.pre_runs, "pre_forms": self.pre_forms, "pre_passes": self.pre_passes,
                "pre_ms": self.pre_ms, "pre_skipped": self.pre_skipped,
                "missed": self.lazy_captured - self.pre_runs},
            "graphs": self.map.iter().map(|((b, w, p, sh, band), g)| json!({"bucket": b, "width": w, "parity": p, "qsa_short": sh, "band": band, "segments": g.segs.len(), "nodes": g.nodes,
                "launches": g.launches, "memsets": g.memsets, "copies": g.copies,
                "digest": format!("{:016x}", g.digest)})).collect::<Vec<_>>()})
    }
}

/// FN98: the gate table's entry bytes (word address u64, value u32, pad u32).
const GATE98_ENT: usize = 16;

/// FN98 (`--fn-graph-form one`): a step form's execs -- the PLE overlap's layer-1 copy point
/// (FN58c: the late rows' copy between layer 0 and layer 1, gathered by the host while layer 0
/// runs) cuts it in two; the head rides the last exec.
fn one_segs(ple: Option<u32>) -> Vec<(u32, u32)> {
    match ple {
        Some(p) if p > 0 && p < LAYERS => vec![(0, p), (p, LAYERS + 1)],
        _ => vec![(0, LAYERS + 1)],
    }
}

/// FN120b: close a captured run's glue -- under `fold` ONE copy of the run's target layers'
/// route ids (already at the log's offsets inside the region) into the route-id log when the
/// tier is complete, then the region's RESTORE: the zero part's used extent to 0 and the ones
/// part's to 0xff, after the run's last reader (stream order). Digested (a held run's dry
/// re-emission must match its capture).
fn glue_end(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    a: u32,
    e: u32,
    complete: bool,
    key: GKey,
    m: u32,
) -> anyhow::Result<()> {
    let Some(g) = em.glue.as_mut() else {
        return Ok(());
    };
    let stride = g.ids_stride;
    let ones_at = g.ones_at();
    let spill = g.spill;
    let (zb, ob) = g.end()?;
    // FN120e: a run past the region fell back to the per-layer fills from that seed on (a
    // served boot; strict refused at the take) -- logged when the run is captured
    if let (Some(sp), false) = (spill, rk.dev.dry) {
        println!(
            "fnglue120 rank{}: FALLBACK form b{} w{} parity {} {} band {} run [{a}, {e}): the {} \
             seeds wanted {} B of {} B -> the rest of the run takes the per-layer fills \
             (spilled runs {}, seeds {})",
            rk.rank,
            key.0,
            key.1,
            key.2,
            if key.3 { "short" } else { "sparse" },
            key.4,
            if sp.ones { "ones" } else { "zero" },
            sp.want,
            sp.cap,
            g.spills,
            g.spilled_seeds
        );
    }
    let l1 = e.min(LAYERS);
    let copy = stride > 0 && complete && a < l1;
    for v in [zb, ob, stride, u64::from(a), u64::from(e), u64::from(copy)] {
        super::dev::fnv(&mut rk.dev.digest, &v.to_le_bytes());
    }
    if rk.dev.dry {
        return Ok(());
    }
    let buf = rk.glue.as_ref().context("FN120b: the glue region")?;
    if copy {
        ensure!(
            stride == u64::from(m) * 40,
            "FN120b: the ids stride is m x 40"
        );
        buf_check(rk, m, "the route-id log")?;
        rk.hip.run_batch(&[Job::MemcpyD2d {
            dst: &rk.idlog.own,
            dst_offset: rk.idlog.off + u64::from(a) * stride,
            src: buf,
            src_offset: u64::from(a) * stride,
            bytes: u64::from(l1 - a) * stride,
            stream: &rk.dev.stream,
        }])?;
        rk.copies += 1;
    }
    if zb > 0 {
        rk.hip.memset_at(buf, 0, zb, 0, &rk.dev.stream)?;
        rk.dev.memsets += 1;
    }
    if ob > 0 {
        rk.hip.memset_at(buf, ones_at, ob, 0xff, &rk.dev.stream)?;
        rk.dev.memsets += 1;
    }
    Ok(())
}

/// FN120c: upload the dataflow sites emitted since the last flush (`exec::DfTables::pending`).
fn df_flush(rk: &mut Rank<'_, '_>, em: &mut Emitter<'_>) -> anyhow::Result<()> {
    let Some(d) = em.df.as_mut() else {
        return Ok(());
    };
    d.armed = false;
    if d.pending.is_empty() {
        return Ok(());
    }
    let buf = rk.df.as_ref().context("FN120c: the dataflow region")?;
    for (off, bytes) in std::mem::take(&mut d.pending) {
        rk.hip
            .memcpy_h2d_at_blocking(buf, off, &bytes, &rk.dev.stream)?;
    }
    Ok(())
}

/// FN120c: the sticky control words of every dataflow site (refusals, dependency timeouts,
/// launches completed), summed -- a refusal or a timeout means a launch ran no bodies.
fn df_sticky(rk: &Rank<'_, '_>, em: &Emitter<'_>) -> anyhow::Result<[u64; 3]> {
    let (Some(d), Some(buf)) = (em.df.as_ref(), rk.df.as_ref()) else {
        return Ok([0; 3]);
    };
    let n = d.used();
    let mut out = [0u64; 3];
    if n == 0 {
        return Ok(out);
    }
    let mut b = vec![0u8; (n * super::exec::DF120_SITE_BYTES) as usize];
    rk.hip.sync_stream(&rk.dev.stream)?;
    rk.hip
        .memcpy_d2h_addr_blocking(&mut b, buf.addr(), &rk.dev.stream)?;
    for site in b.chunks_exact(super::exec::DF120_SITE_BYTES as usize) {
        for (o, &w) in out.iter_mut().zip(&super::exec::DF120_STICKY) {
            *o += u64::from(u32::from_le_bytes([
                site[w * 4],
                site[w * 4 + 1],
                site[w * 4 + 2],
                site[w * 4 + 3],
            ]));
        }
    }
    Ok(out)
}

/// FN98: the gate blocks of the run `[a, e)` -- its first layer and every multiple of `q`
/// inside it (FN64's cut points: a block's gate waits no later than the per-layer form's and at
/// most `q - 1` layers earlier), each `(start, end)`.
fn one_blocks(a: u32, e: u32, q: u32) -> Vec<(u32, u32)> {
    let end = e.min(LAYERS);
    let mut pts: Vec<u32> = (a..end).filter(|&l| l == a || l % q == 0).collect();
    pts.push(end);
    pts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Capture `body` on the rank's stream as one graph: (exec, graph, node count). The capture is
/// closed before any error is raised (a stream left capturing poisons it).
fn capture_seg(
    rk: &mut Rank<'_, '_>,
    body: &mut dyn FnMut(&mut Rank<'_, '_>) -> anyhow::Result<()>,
) -> anyhow::Result<(saber_hip::GraphExec, saber_hip::Graph, usize)> {
    rk.hip
        .begin_capture(&rk.dev.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
    let r = body(rk);
    let g = rk.hip.end_capture(&rk.dev.stream);
    r?;
    let g = g?;
    let n = rk.hip.graph_node_count(&g)?;
    Ok((rk.hip.instantiate(&g)?, g, n))
}

/// FN64 (`--fn-graph-split lazy`): make every run of `segs` (the step's layer runs + the head,
/// `(LAYERS, LAYERS + 1)`) captured for `key`. A pass walks the WHOLE step in order (each layer's
/// emission feeds the next): a missing run is captured, a held one is re-emitted DRY (no launch)
/// and its records' digest checked against its capture -- the by-construction law at every
/// capture pass (and at every step under `--fn-capture-verify`). Past `cache` runs a rank, the
/// least recently used runs of other steps are dropped (only after the pass's stream sync: no
/// dropped graph is in flight). Returns (runs captured, pass ms) when the pass ran.
#[allow(clippy::too_many_arguments)]
fn lazy_capture(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    gr: &mut Graphs,
    key: GKey,
    segs: &[(u32, u32)],
    dp: usize,
    complete: bool,
    cc: ChunkConsts,
    stat: &mut StepStat,
    verify: bool,
    cache: usize,
    pre: bool,
    one: Option<u32>,
) -> anyhow::Result<()> {
    gr.tick += 1;
    let tick = gr.tick;
    if !pre && gr.record.is_some() {
        let mask = segs
            .iter()
            .filter(|s| s.0 < LAYERS)
            .fold(0u64, |m, s| m | (1u64 << s.0));
        gr.trace.push((key, mask));
    }
    let lk = gr.lazy.entry(key).or_default();
    // FN64b: the first served step of a form re-emits its runs dry against the (pre-)captures
    let verify = verify || (!pre && !lk.seen);
    if !pre {
        lk.seen = true;
    }
    let missing = segs.iter().filter(|s| !lk.runs.contains_key(s)).count();
    if !pre {
        lk.served += 1;
        lk.passes += u64::from(missing > 0);
    }
    if missing == 0 && !verify {
        for s in segs {
            if let Some(r) = lk.runs.get_mut(s) {
                r.last = tick;
                r.uses += 1;
            }
        }
        return Ok(());
    }
    let tc = Instant::now();
    if missing > 0 {
        rk.hip.sync_stream(&rk.dev.stream)?;
    }
    let mut input = LayerIn::Embed(rk.emb.addr().as_u64());
    let rows = rk.rows.addr().as_u64();
    let mut nodes = 0usize;
    // FN98: (gate nodes, exec bytes, execs) captured by this pass
    let mut add98 = (0u64, 0u64, 0u64);
    for &(a, e) in segs {
        let held = lk.runs.contains_key(&(a, e));
        rk.dev.digest = super::dev::FNV0;
        let (l0, m0, c0) = (rk.dev.launches, rk.dev.memsets, rk.copies);
        let mut lg = 0u64;
        // FN98: the run's gate blocks (`one`): a `fn_gate98` launch before each block's first
        // layer (emitted dry too: the digest covers it)
        let blocks = one.map_or_else(Vec::new, |q| one_blocks(a, e, q));
        let tab = gr.gate_tab.as_ref().map(|t| t.0.addr().as_u64());
        let mut nw = 0u64;
        let mut body = |rk: &mut Rank<'_, '_>| -> anyhow::Result<()> {
            // FN120b: the run's seeds from the glue region (None = `--fn-glue off`)
            if let Some(g) = em.glue.as_mut() {
                g.begin(u64::from(cc.m))?;
            }
            // FN120c: a captured target run's GDN tails may fold (`--fn-dataflow gdn`)
            if let Some(d) = em.df.as_mut() {
                d.armed = true;
            }
            if a < LAYERS {
                for l in a..e.min(LAYERS) {
                    if blocks.iter().any(|b| b.0 == l) {
                        let tab = tab.context("FN98: the gate table")?;
                        em.run(
                            "fn_gate98",
                            [1, 1, 1],
                            32,
                            vec![Arg::P(tab), Arg::I(l as i32)],
                            saber_model::qwen4_exp::layer_graph::Group::Moe,
                        )?;
                        rk.dev.exec(&rk.table, em)?;
                        nw += 1;
                    }
                    let st = rk.pools.batch(l as usize, dp);
                    let o = layer(rk, em, l, complete, st, input, rows, cc, stat, false)?;
                    input = LayerIn::Pending(o.pend);
                }
                if e > LAYERS {
                    // FN98: the head rides the step's last exec
                    let LayerIn::Pending(pend) = input else {
                        bail!("FN98: the head before any layer");
                    };
                    let (_, lgh, strm) =
                        emit_post3(em, &rk.fm, rk.head, &pend, u64::from(cc.m), true)?;
                    rk.dev.exec(&rk.table, em)?;
                    mtp_keep_stream(rk, strm, cc.m)?;
                    lg = lgh;
                }
            } else {
                let LayerIn::Pending(pend) = input else {
                    bail!("FN64: the head run before any layer");
                };
                let (_, lgh, strm) = emit_post3(em, &rk.fm, rk.head, &pend, u64::from(cc.m), true)?;
                rk.dev.exec(&rk.table, em)?;
                mtp_keep_stream(rk, strm, cc.m)?;
                lg = lgh;
            }
            if let Some(d) = em.df.as_mut() {
                d.armed = false;
            }
            glue_end(rk, em, a, e, complete, key, cc.m)
        };
        if held {
            rk.dev.dry = true;
            let r = body(rk);
            rk.dev.dry = false;
            r?;
            let run = lk.runs.get_mut(&(a, e)).context("FN64: a held run")?;
            ensure!(
                rk.dev.digest == run.digest,
                "FN64 b{} w{} p{dp}: run [{a}, {e}) re-emitted differs from its capture ({:016x} vs {:016x})",
                key.0,
                key.1,
                rk.dev.digest,
                run.digest
            );
            ensure!(
                e <= LAYERS || lg == lk.lg,
                "FN64: the head's logits moved ({lg:#x} vs {:#x})",
                lk.lg
            );
            run.last = tick;
            run.uses += u64::from(!pre);
            gr.lazy_checked += 1;
        } else {
            let free0 = if one.is_some() {
                rk.hip.sync_stream(&rk.dev.stream)?;
                rk.hip.mem_info()?.0
            } else {
                0
            };
            let (exec, g, n) = capture_seg(rk, &mut body)?;
            let (dl, dm, dc) = (rk.dev.launches - l0, rk.dev.memsets - m0, rk.copies - c0);
            ensure!(
                n as u64 == dl + dm + dc,
                "FN64 run [{a}, {e}): the graph holds {n} nodes; the run issued {dl} launches (FN98 gates {nw}) + {dm} memsets + {dc} copies"
            );
            if e > LAYERS {
                lk.lg = lg;
            }
            if nw > 0 {
                add98.0 += nw;
            }
            if one.is_some() {
                rk.hip.sync_stream(&rk.dev.stream)?;
                add98.1 += free0.saturating_sub(rk.hip.mem_info()?.0);
                add98.2 += 1;
            }
            nodes += n;
            lk.runs.insert(
                (a, e),
                LazyRun {
                    exec,
                    _graph: g,
                    digest: rk.dev.digest,
                    launches: dl,
                    last: tick,
                    uses: u64::from(!pre),
                    pre,
                    gates: blocks.clone(),
                },
            );
            gr.lazy_captured += 1;
        }
    }
    // FN120c: the new dataflow sites' tables reach the card after the capture closed and before
    // any graph of this pass launches (stream order); a re-emitted site found its bytes
    df_flush(rk, em)?;
    gr.one_nodes += add98.0;
    gr.one_bytes += add98.1;
    gr.one_execs += add98.2;
    if missing > 0 {
        // the LRU bound: drop the least recently used runs of other steps (the stream is idle)
        let total: usize = gr.lazy.values().map(|k| k.runs.len()).sum();
        if total > cache {
            let mut all: Vec<(u64, GKey, (u32, u32))> = gr
                .lazy
                .iter()
                .flat_map(|(k, v)| v.runs.iter().map(move |(r, x)| (x.last, *k, *r)))
                .filter(|x| x.0 != tick)
                .collect();
            all.sort_unstable();
            for (_, k, r) in all.into_iter().take(total - cache) {
                if let Some(v) = gr.lazy.get_mut(&k) {
                    // FN64e: an evicted run keeps its served uses in the record
                    if let Some(x) = v.runs.remove(&r).filter(|x| x.uses > 0) {
                        *gr.retired.entry(k).or_default().entry(r).or_default() += x.uses;
                    }
                    gr.lazy_evicted += 1;
                }
            }
        }
        gr.lazy_passes += 1;
        gr.capture_ms += tc.elapsed().as_secs_f64() * 1e3;
        if !pre && gr.lazy_passes - gr.pre_passes <= 24 {
            println!(
                "fngraph64 rank{}: b{} w{} parity {dp}: {} run(s) of {} captured, {nodes} nodes ({:.1} ms)",
                rk.rank,
                key.0,
                key.1,
                missing,
                segs.len(),
                tc.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}

/// FN64b: the VRAM the boot's pre-capture may take a rank (graph kernel arguments) and the free
/// VRAM it never goes below; a form past either is captured on demand instead.
const PRE_VRAM_BUDGET: u64 = 192 << 20;
const PRE_VRAM_FLOOR: u64 = 256 << 20;
/// FN64d: the free VRAM a BILLED exec budget's pre-capture never goes below (the ledger holds
/// its bytes; this only keeps the allocator off its last pages).
const PRE_VRAM_FLOOR_BILLED: u64 = 16 << 20;

/// FN64b: the process's resident host memory (MiB; `/proc/self/status` VmRSS, 0 when unread).
fn rss_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb / 1024)
}

/// FN64b (`--fn-graph-split lazy`, `--fn-graph-lazy-pre on`): before the first step, capture
/// every run a decode / verify step of this boot can ask for, so no capture pass lands on a
/// served step. The forms: each bucket (1/2/4/8) at the MTP law's table width (1 at SPEC 0) x
/// both pool parities x short rows (FN19's dense QSA form: every row at position 0) and long
/// rows at each context band (positions just under the band); a form's constants are laid out by
/// the step's own functions ([`band_tables`], [`kv_extend_at`] without the step law,
/// [`step_consts`]) and its key is the step's. Per form, [`super::plan::lazy_cover`]'s plans are
/// walked through [`lazy_capture`] (missing runs captured, held runs re-emitted dry and checked):
/// every reachable run + the head. Nothing executes (captures + dry emissions only); forms that
/// would pass `cache` are left to the on-demand path. The first served step of each form
/// re-emits its runs dry against these captures (`LazyKey::seen`).
fn lazy_precapture(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    gr: &mut Graphs,
    f: &FnFlags,
    set_dir: &Path,
) -> anyhow::Result<()> {
    let super::run::GraphSplit::Lazy { q, cache, pre } = rk.gsplit else {
        return Ok(());
    };
    if pre == super::run::LazyPre::Off || f.no_fn_capture || rk.dev.op_timing || rk.pg.is_some() {
        return Ok(());
    }
    if pre == super::run::LazyPre::Recorded {
        let cap = gr.pre_cap;
        // FN92: under the adaptive depth arm every b1 width is a served form (the recorded set
        // holds the table's): each takes the recorded b1 form's runs
        let b1 = f
            .mtp_law()?
            .filter(|l| l.adaptive)
            .map(|l| (l.table_width(1), l.widths_at(1)));
        return lazy_precapture_recorded(rk, em, gr, set_dir, q, cache, cap, b1);
    }
    let t0 = Instant::now();
    let (free0, _) = rk.hip.mem_info()?;
    let rss0 = rss_mib();
    let law = f.mtp_law()?;
    let ple = rk.ple_ov.is_some().then_some(super::plan::PLE_LAYER);
    let plans = super::plan::lazy_cover(q, ple);
    let per_form = super::plan::lazy_reachable(q, ple).len() + 1;
    // the bands a step can land in (FN51f), ascending
    let mut bands: Vec<u32> = super::kvwire::CTX_BANDS
        .iter()
        .copied()
        .filter(|&b| b < rk.max_ctx)
        .collect();
    bands.push(rk.max_ctx);
    if rk.max_ctx <= super::kvwire::DEFAULT_FN_KV_MAX_CTX || rk.band_boot {
        bands = vec![rk.max_ctx];
    }
    if pre == super::run::LazyPre::FirstBand {
        bands.truncate(1);
    }
    let mut keys: Vec<GKey> = Vec::new();
    let mut stat = StepStat::default();
    let r = (|| -> anyhow::Result<()> {
        for &band_top in &bands {
            for short in [true, false] {
                if short && band_top != bands[0] {
                    continue;
                }
                // FN92: every width a bucket's steps can run (the adaptive arm: b1 at 2..=5)
                let forms: Vec<(u32, u32)> = BUCKETS
                    .iter()
                    .flat_map(|&b| {
                        law.as_ref()
                            .map_or(vec![1], |l| l.widths_at(b))
                            .into_iter()
                            .map(move |w| (b, w.max(1)))
                    })
                    .collect();
                for (bucket, w) in forms {
                    if w > rk.pools.pool_w {
                        continue;
                    }
                    let p = if short {
                        0
                    } else {
                        band_top
                            .saturating_sub(super::kvwire::BAND_MARGIN + w)
                            .min(rk.max_ctx.saturating_sub(w + 1))
                    };
                    let pos = vec![p; bucket as usize];
                    if saber_kernels::fn19::short_rows(&pos, w) != short {
                        continue;
                    }
                    let band = step_band(rk, &pos);
                    let na = vec![1u32; bucket as usize];
                    let dr = vec![0u32; bucket as usize];
                    let (mut tb, mut offs) =
                        band_tables(rk, &pos, w, &na, &dr, rk.pools.pool_w, 0, band);
                    kv_extend_at(rk, &mut tb, &mut offs, &pos, w, 0, band.1, false)?;
                    let cc = step_consts(rk, &offs, bucket, w, band, &pos);
                    for dp in 0..2usize {
                        let key = (bucket, w, dp, cc.qsa_short, cc.max_ctx);
                        if keys.contains(&key) {
                            continue;
                        }
                        let held: usize = gr.lazy.values().map(|k| k.runs.len()).sum();
                        // the cache bound, and the VRAM the graphs' kernel arguments take (the
                        // exact ledger leaves ~360 MiB a card at the served boot): a form past
                        // either is left to the on-demand path
                        let free = rk.hip.mem_info()?.0;
                        if held + per_form > cache
                            || free0.saturating_sub(free) > PRE_VRAM_BUDGET
                            || free < PRE_VRAM_FLOOR
                        {
                            gr.pre_skipped += 1;
                            continue;
                        }
                        keys.push(key);
                        for plan in &plans {
                            let mut v = plan.clone();
                            v.push((LAYERS, LAYERS + 1));
                            lazy_capture(
                                rk, em, gr, key, &v, dp, true, cc, &mut stat, false, cache, true,
                                None,
                            )?;
                        }
                        let n = gr.lazy.get(&key).map_or(0, |k| k.runs.len());
                        ensure!(
                            n == per_form,
                            "FN64b: form b{bucket} w{w} p{dp}: {n} runs held of {per_form}"
                        );
                    }
                }
            }
        }
        Ok(())
    })();
    r?;
    rk.hip.sync_stream(&rk.dev.stream)?;
    let (free1, _) = rk.hip.mem_info()?;
    gr.pre_forms = keys.len() as u64;
    gr.pre_runs = gr.lazy_captured;
    gr.pre_passes = gr.lazy_passes;
    gr.pre_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "fngraph64b rank{}: pre-captured {} runs ({} a form) of {} forms in {} passes ({} plans a form), {:.0} ms; {} forms past the cache ({cache}) left on demand; VRAM free {} -> {} MiB, host RSS {} -> {} MiB",
        rk.rank,
        gr.pre_runs,
        per_form,
        gr.pre_forms,
        gr.pre_passes,
        plans.len(),
        gr.pre_ms,
        gr.pre_skipped,
        free0 >> 20,
        free1 >> 20,
        rss0,
        rss_mib()
    );
    Ok(())
}

/// FN64c (`--fn-graph-lazy-pre recorded`): before the first step, capture exactly the runs a
/// served boot used (`--fn-graph-lazy-record`'s file for this rank, read from `set_dir`): per
/// recorded form (most used first), its constants are laid out by the step's own functions at
/// the form's rows (short = position 0, long = just under the form's band), its key must be the
/// recorded key (a form this boot cannot make is counted, not captured), and
/// [`super::plan::lazy_cover_set`]'s plans are walked through [`lazy_capture`] -- the recorded
/// runs captured, nothing else (a run on no tiling of the set is left on demand). Bounded by the
/// cache and the free-VRAM floor; the first served step of each form re-emits its runs dry
/// against these captures (`LazyKey::seen`), as FN64b's.
#[allow(clippy::too_many_arguments)]
fn lazy_precapture_recorded(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    gr: &mut Graphs,
    set_dir: &Path,
    q: u32,
    cache: usize,
    exec_cap: Option<usize>,
    b1_widths: Option<(u32, Vec<u32>)>,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let (free0, _) = rk.hip.mem_info()?;
    let rss0 = rss_mib();
    let ple = rk.ple_ov.is_some().then_some(super::plan::PLE_LAYER);
    let path = super::run::lazy_set_file(set_dir, rk.rank);
    if !path.exists() {
        // the default's one fallback, said by name (which path: `absent`)
        println!(
            "fngraph64c rank{}: NO recorded set at {} (which path: absent): every run captured on demand",
            rk.rank,
            path.display()
        );
        return Ok(());
    }
    let mut set = read_lazy_set(&path, rk.rank, (q, ple))?;
    if let Some((tw, ws)) = &b1_widths {
        let b1: Vec<_> = set
            .iter()
            .filter(|(k, _)| k.0 == 1 && k.1 == *tw)
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let mut added = 0;
        for (k, v) in b1 {
            for &w in ws {
                let nk = (k.0, w, k.2, k.3, k.4);
                if let std::collections::btree_map::Entry::Vacant(e) = set.entry(nk) {
                    e.insert(v.clone());
                    added += 1;
                }
            }
        }
        println!(
            "fngraph92 rank{}: the adaptive depth arm adds {added} b1 forms (widths {ws:?}, the recorded w{tw} forms' runs)",
            rk.rank
        );
    }
    // FN64d: under a billed exec budget, the most used RUNS across forms within it (the rest on
    // demand, the LRU evicting past it); the ledger billed these bytes, so the floor is only the
    // allocator's (PRE_VRAM_FLOOR_BILLED)
    // FN64e: under the default CHARGE the ledger holds the cap's bytes and the cap bounds the
    // pre-capture (the end state FN64d measured on served traffic: hipMemGetInfo read 0 MiB free
    // while sysfs read ~650 MiB, so its reading is not a floor here) -- no free-VRAM floor
    let budget_sel = exec_cap.map(|c| super::plan::lazy_budget_select(&set, c));
    let floor = match gr.bill {
        super::run::ExecBill::Unbilled => PRE_VRAM_FLOOR,
        super::run::ExecBill::Budget(_) => PRE_VRAM_FLOOR_BILLED,
        super::run::ExecBill::Charge | super::run::ExecBill::One => 0,
    };
    // the most used forms first (the floor / cache cut the tail)
    let mut forms: Vec<_> = set.iter().collect();
    forms.sort_by_key(|(k, v)| (std::cmp::Reverse(v.iter().map(|r| r.1).sum::<u64>()), **k));
    let (mut unmatched, mut orphans, mut floor_cut, mut nforms) = (0u64, 0usize, 0u64, 0u64);
    let mut stat = StepStat::default();
    for (&key, runs) in forms {
        let (bucket, w, dp, short, band_top) = key;
        let p = if short {
            0
        } else {
            band_top
                .saturating_sub(super::kvwire::BAND_MARGIN + w)
                .min(rk.max_ctx.saturating_sub(w + 1))
        };
        let pos = vec![p; bucket as usize];
        if w > rk.pools.pool_w
            || !BUCKETS.contains(&bucket)
            || saber_kernels::fn19::short_rows(&pos, w) != short
        {
            unmatched += 1;
            continue;
        }
        let band = step_band(rk, &pos);
        let na = vec![1u32; bucket as usize];
        let dr = vec![0u32; bucket as usize];
        let (mut tb, mut offs) = band_tables(rk, &pos, w, &na, &dr, rk.pools.pool_w, 0, band);
        kv_extend_at(rk, &mut tb, &mut offs, &pos, w, 0, band.1, false)?;
        let cc = step_consts(rk, &offs, bucket, w, band, &pos);
        if (bucket, w, dp, cc.qsa_short, cc.max_ctx) != key {
            unmatched += 1;
            continue;
        }
        if let Some(q1) = gr.one {
            // FN98: the recorded set names the FORMS; each is captured whole (its execs with the
            // gates inside), the lazy runs are not consulted
            let v = one_segs(ple);
            lazy_capture(
                rk,
                em,
                gr,
                key,
                &v,
                dp,
                true,
                cc,
                &mut stat,
                false,
                cache,
                true,
                Some(q1),
            )?;
            nforms += 1;
            continue;
        }
        let rset: std::collections::BTreeSet<(u32, u32)> = match &budget_sel {
            None => runs.iter().map(|r| r.0).collect(),
            Some(sel) => match sel.get(&key) {
                Some(v) => v.clone(),
                None => {
                    floor_cut += 1;
                    continue;
                }
            },
        };
        let (plans, orph) = super::plan::lazy_cover_set(&rset);
        let want = rset.iter().filter(|r| r.1 <= LAYERS).count() - orph + 1;
        let held: usize = gr.lazy.values().map(|k| k.runs.len()).sum();
        let free = rk.hip.mem_info()?.0;
        if held + want > cache || free < floor {
            floor_cut += 1;
            continue;
        }
        orphans += orph;
        nforms += 1;
        for plan in &plans {
            let mut v = plan.clone();
            v.push((LAYERS, LAYERS + 1));
            lazy_capture(
                rk, em, gr, key, &v, dp, true, cc, &mut stat, false, cache, true, None,
            )?;
        }
        let n = gr.lazy.get(&key).map_or(0, |k| k.runs.len());
        ensure!(
            n == want,
            "FN64c: form b{bucket} w{w} p{dp}: {n} runs held of the {want} recorded"
        );
    }
    rk.hip.sync_stream(&rk.dev.stream)?;
    let (free1, _) = rk.hip.mem_info()?;
    if let (Some(c), Some(sel)) = (exec_cap, &budget_sel) {
        let rec: usize = set.values().map(Vec::len).sum();
        println!(
            "fngraph64d rank{}: exec budget {c} runs ({} MiB, billed, {:?}): selected {} of the {rec} recorded runs over {} of {} forms (most used first); the rest on demand, LRU past {}",
            rk.rank,
            c as u64 * super::run::GRAPH_EXEC_POOL_MIB,
            gr.bill,
            super::plan::lazy_selection_execs(sel),
            sel.len(),
            set.len(),
            gr.exec_cap.unwrap_or(cache)
        );
    }
    gr.pre_forms = nforms;
    gr.pre_runs = gr.lazy_captured;
    gr.pre_passes = gr.lazy_passes;
    gr.pre_ms = t0.elapsed().as_secs_f64() * 1e3;
    gr.pre_skipped = unmatched + floor_cut;
    println!(
        "fngraph64c rank{}: pre-captured the RECORDED set {}: {} runs of {} forms in {} passes, {:.0} ms; {} recorded forms not made by this boot, {} past the cache ({cache}) / VRAM floor, {} runs on no tiling (on demand); VRAM free {} -> {} MiB ({:.2} MiB a run), host RSS {} -> {} MiB",
        rk.rank,
        path.display(),
        gr.pre_runs,
        nforms,
        gr.pre_passes,
        gr.pre_ms,
        unmatched,
        floor_cut,
        orphans,
        free0 >> 20,
        free1 >> 20,
        free0.saturating_sub(free1) as f64 / f64::from(1u32 << 20) / gr.pre_runs.max(1) as f64,
        rss0,
        rss_mib()
    );
    Ok(())
}

/// The step's body: every layer at the pool view of parity `dp`, the final mixer + the head.
/// Returns the logits' address.
#[allow(clippy::too_many_arguments)]
fn step_body(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    b: u32,
    dp: usize,
    complete: bool,
    cc: ChunkConsts,
    stat: &mut StepStat,
    gate: bool,
) -> anyhow::Result<u64> {
    let mut input = LayerIn::Embed(rk.emb.addr().as_u64());
    let rows = rk.rows.addr().as_u64();
    for l in 0..LAYERS {
        let st = rk.pools.batch(l as usize, dp);
        let o = layer(rk, em, l, complete, st, input, rows, cc, stat, gate)?;
        input = LayerIn::Pending(o.pend);
    }
    let LayerIn::Pending(pend) = input else {
        unreachable!()
    };
    let _ = b;
    let (_, lg, strm) = emit_post3(em, &rk.fm, rk.head, &pend, u64::from(cc.m), true)?;
    rk.dev.exec(&rk.table, em)?;
    mtp_keep_stream(rk, strm, cc.m)?;
    Ok(lg)
}

/// Read `n` bytes at a POOL address (the taps' K/V pages).
fn d2h_pool(rk: &Rank<'_, '_>, a: u64, n: usize) -> anyhow::Result<Vec<u8>> {
    let (d, _) = rk.pools.owner(a)?;
    d2h_in(&rk.dev, d, a, n)
}

/// FN15's aux capture: the target's pre-final-mixer stream of this step's `rows` rows, copied to
/// the drafter's input buffer (one D2D copy -- a graph node when captured; none without `--fn-mtp`).
fn mtp_keep_stream(rk: &mut Rank<'_, '_>, strm: u64, rows: u32) -> anyhow::Result<()> {
    // FN100: the hidden of more rows than the heads refuses while the tails are lent
    if rk.mtp.is_some() {
        buf_check(rk, rows, "the drafter's hidden")?;
    }
    let Some(mt) = rk.mtp.as_ref() else {
        return Ok(());
    };
    if rk.dev.dry {
        return Ok(());
    }
    let base = rk.dev.base();
    rk.hip.run_batch(&[Job::MemcpyD2d {
        dst: &mt.hid.own,
        dst_offset: mt.hid.off,
        src: &rk.dev.arena,
        src_offset: strm - base,
        bytes: u64::from(rows) * W * 2,
        stream: &rk.dev.stream,
    }])?;
    rk.copies += 1;
    Ok(())
}

fn rk_mtp_on(f: &FnFlags) -> bool {
    f.fn_mtp.on() || f.fn_mtp_ref.is_some()
}

/// FN15b: MTP-k through the verify step (not the shadow form, not the host-reference arm).
pub(super) fn mtp_verify(f: &FnFlags) -> bool {
    f.fn_mtp.on() && !f.fn_mtp_shadow && f.fn_mtp_ref.is_none()
}

/// FN18: the pools' context and GDN snapshot width for a boot (the worker allocates them; the
/// exact ledger plans them on the host): every request + its decode (+ a second turn), at least the
/// 2,048 band; a serve boot at `--fn-max-ctx`. MTP-k keeps k + 1 GDN snapshots a sequence.
pub(super) fn pools_geometry(f: &FnFlags, prompts: &[Prompt]) -> anyhow::Result<(u32, u32)> {
    let max_ctx = super::run::pools_ctx(f, prompts);
    ensure!(
        max_ctx <= ROPE_POS_MAX,
        "the pools' context {max_ctx} past the checkpoint's {ROPE_POS_MAX} positions"
    );
    let pool_w = if mtp_verify(f) { f.mtp_k() + 1 } else { 1 };
    Ok((max_ctx, pool_w))
}

/// Row `t` of sequence `i`'s verify window: the committed token, then the drafts.
fn tok_of_row(next: u32, drafts: &[Vec<u32>], i: usize, t: usize) -> u32 {
    if t == 0 {
        next
    } else {
        drafts[i][t - 1]
    }
}

/// FN15: the MTP prefill pass's piece (rows): `fc_hidden` runs at 4 x rows, inside the 2,048 band.
const MTP_PIECE: u32 = 512;

/// Upload a decode step's tables for positions `pos` (one row per pool slot, padded) and return
/// the step's constants (FN13's `host_decode` + the context band + the fused update's barrier).
fn decode_consts(rk: &Rank<'_, '_>, pos: &[u32], live: usize) -> anyhow::Result<ChunkConsts> {
    let band = step_band(rk, pos);
    let (tb, offs) = decode_tables(rk, pos, live, band)?;
    rk.hip
        .memcpy_h2d_at_blocking(&rk.dev.consts, 0, &tb, &rk.dev.stream)?;
    Ok(decode_cc(rk, &offs, pos, band))
}

/// FN51f: a decode-form step's band `(segmentation context, bt row length)` for rows starting at
/// `pos` (`kvwire::band_of`; the boot's max context under `--fn-control band-boot`). The `bt` row
/// length follows the band under the tiered KV only (the flat pools' boots are <= 10,240: band ==
/// max context there).
/// A decode / verify step's chunk constants at its tables' layout `offs` (FN64b: one function for
/// the step and the boot's pre-capture, so a pre-captured form's emission is the step's).
fn step_consts(
    rk: &Rank<'_, '_>,
    offs: &[(&'static str, usize)],
    bucket: u32,
    w: u32,
    band: (u32, u32),
    pos: &[u32],
) -> ChunkConsts {
    let mut cc = ChunkConsts::at(
        rk.dev.consts.addr().as_u64(),
        offs,
        bucket * w,
        0,
        2,
        rk.pools.nbs,
        rk.dev.cs.addr().as_u64(),
    );
    cc.max_ctx = band.0;
    cc.bt_nb = band.1;
    cc.bar = rk.bar.addr().as_u64();
    cc.b = bucket;
    cc.w = w;
    cc.pw = rk.pools.pool_w;
    cc.qsa_short = saber_kernels::fn19::short_rows(pos, w);
    cc
}

fn step_band(rk: &Rank<'_, '_>, pos: &[u32]) -> (u32, u32) {
    if rk.band_boot {
        return (rk.max_ctx, rk.pools.nbs);
    }
    let band = super::kvwire::band_of(rk.max_ctx, pos);
    let bt_nb = if rk.pools.kvt.is_some() {
        super::kvwire::band_nb(band).min(rk.pools.nbs)
    } else {
        rk.pools.nbs
    };
    (band, bt_nb)
}

/// FN51f: the step blob's tables at `band` (the tiered KV drops `bt_rows`: its per-row K/V table
/// is `kbtr`; the control keeps it).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn band_tables(
    rk: &Rank<'_, '_>,
    pos: &[u32],
    w: u32,
    naccept: &[u32],
    drop: &[u32],
    pool_w: u32,
    live: usize,
    band: (u32, u32),
) -> (Vec<u8>, Vec<(&'static str, usize)>) {
    let (mut tb, offs) = ChunkConsts::host_verify_band(
        pos,
        w,
        naccept,
        drop,
        rk.pools.nbs,
        band.1,
        rk.pools.kvt.is_none() || rk.band_boot,
        pool_w,
        rk.pads(live),
    );
    fix_pos3(rk, &mut tb, &offs, pos, w, live);
    (tb, offs)
}

/// FN55b: a decode-form step's `pos3` rows of IMAGE sequences at their M-RoPE positions (past the
/// prompt: `index + delta` on every axis); a text sequence's rows -- and every row of a boot
/// without `--fn-vision` -- are untouched (the pre-FN55b bytes).
fn fix_pos3(
    rk: &Rank<'_, '_>,
    tb: &mut [u8],
    offs: &[(&'static str, usize)],
    pos: &[u32],
    w: u32,
    live: usize,
) {
    if !rk.vis_on || rk.ropes.iter().all(Option::is_none) {
        return;
    }
    let Some(&(_, o)) = offs.iter().find(|x| x.0 == "pos3") else {
        return;
    };
    let pads = rk.pads(live);
    let wu = w as usize;
    let m = pos.len() * wu;
    for r in 0..m {
        let s = r / wu;
        let Some(im) = rk.ropes.get(pads.slot(s) as usize).and_then(Option::as_ref) else {
            continue;
        };
        let p = im.rope.at(pos[s] + (r % wu) as u32);
        for (ax, v) in p.iter().enumerate() {
            let at = o + (ax * m + r) * 8;
            tb[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// The tables of a width-1 decode-form step at `pos` (one row per pool slot, padded) -- the
/// drafter's chain passes (FN15) and SPEC 0's shadow pass.
#[allow(clippy::type_complexity)]
fn decode_tables(
    rk: &Rank<'_, '_>,
    pos: &[u32],
    live: usize,
    band: (u32, u32),
) -> anyhow::Result<(Vec<u8>, Vec<(&'static str, usize)>)> {
    decode_tables_at(rk, pos, live, band, true)
}

/// [`decode_tables`]; `begin` false = no KV step law (FN119's variants after one union begin).
#[allow(clippy::type_complexity)]
fn decode_tables_at(
    rk: &Rank<'_, '_>,
    pos: &[u32],
    live: usize,
    band: (u32, u32),
    begin: bool,
) -> anyhow::Result<(Vec<u8>, Vec<(&'static str, usize)>)> {
    let n = pos.len();
    let (mut tb, mut offs) = band_tables(rk, pos, 1, &vec![1; n], &vec![0; n], 1, live, band);
    kv_extend_at(rk, &mut tb, &mut offs, pos, 1, live, band.1, begin)?;
    Ok((tb, offs))
}

/// FN45b2: a decode-form step under the tiered KV (no-op on the flat pools): the step law
/// (`KvDev::begin`: finals + flushes + open pages + the table's changed entries, on the stream
/// before the step) and the step's per-layer tables appended to its constants (`kslot`, and
/// `kbtr` for a dense verify step) -- inside the blob, so the drafter chain's staged copies and
/// the captured graphs carry them at one layout per (bucket, width).
fn kv_extend(
    rk: &Rank<'_, '_>,
    tb: &mut Vec<u8>,
    offs: &mut Vec<(&'static str, usize)>,
    pos: &[u32],
    w: u32,
    live: usize,
    bt_nb: u32,
) -> anyhow::Result<()> {
    kv_extend_at(rk, tb, offs, pos, w, live, bt_nb, true)
}

/// [`kv_extend`]; `begin` false = the tables' LAYOUT only (no step law on the stream: FN64b's
/// pre-capture lays a step form's constants out without a step).
#[allow(clippy::too_many_arguments)]
fn kv_extend_at(
    rk: &Rank<'_, '_>,
    tb: &mut Vec<u8>,
    offs: &mut Vec<(&'static str, usize)>,
    pos: &[u32],
    w: u32,
    live: usize,
    bt_nb: u32,
    begin: bool,
) -> anyhow::Result<()> {
    let Some(k) = rk.pools.kvt.as_ref() else {
        return Ok(());
    };
    let mut k = k.borrow_mut();
    let pads = rk.pads(live);
    let n = pos.len();
    let seqs: Vec<(u32, u32, u32)> = (0..live.min(n))
        .map(|s| (s as u32, pos[s], pos[s] + w))
        .collect();
    if begin {
        k.begin(rk.hip, &rk.dev.stream, &seqs, false)?;
    }
    let wu = w as usize;
    if let Some(ix) = k.ix.as_ref() {
        // FN51: the paged index store -- each batch row's `bt` row is its pool slot's pages
        // (rows = pool slots; a pad row reads the spare slot's row: the pad page)
        // FN51f: the band's row length (the entries a row of the band can address)
        let nb = bt_nb as usize;
        // FN51e: each pool slot's row as bytes ONCE a step, copied whole into every batch row
        // that reads it (the per-entry rewrite was ~0.5 ms a b8 verify step at 16,385 entries)
        let mut rows: Vec<(u32, Vec<u8>)> = Vec::new();
        for (name, per) in [("bt", 1usize), ("bt_rows", wu)] {
            if let Some(&(_, o)) = offs.iter().find(|x| x.0 == name) {
                for r in 0..n * per {
                    let slot = pads.slot(r / per);
                    let i = match rows.iter().position(|x| x.0 == slot) {
                        Some(i) => i,
                        None => {
                            rows.push((slot, super::exec::le32(&ix.row(slot)[..nb])));
                            rows.len() - 1
                        }
                    };
                    let at = o + r * nb * 4;
                    tb[at..at + nb * 4].copy_from_slice(&rows[i].1);
                }
            }
        }
    }
    let rows: Vec<(u32, u32)> = (0..n * wu)
        .map(|r| (pads.slot(r / wu), pos[r / wu] + (r % wu) as u32))
        .collect();
    let per_row = w > 1 && saber_kernels::fn19::short_rows(pos, w);
    let (ks, br) = super::kvwire::step_tables_of(&k.tables(), &rows, per_row);
    super::exec::put_table(tb, offs, "kslot", &ks);
    if per_row {
        super::exec::put_table(tb, offs, "kbtr", &br);
    }
    Ok(())
}

/// FN45b2: [`consts`] for a prefill chunk / drafter piece of the sequence at pool slot `slot`
/// under the tiered KV: the step law, the chunk's per-layer append slots at the constants'
/// [`KV_PREFILL_OFF`], the chunk's row of each layer's page table. The flat pools: `consts`.
fn kv_consts(
    rk: &Rank<'_, '_>,
    m: u32,
    p0: u32,
    chunk: u32,
    slot: u32,
) -> anyhow::Result<ChunkConsts> {
    let mut cc = consts(&rk.dev, 0, m, p0, chunk)?;
    if let Some(im) = rk.ropes.get(slot as usize).cloned().flatten() {
        // FN55b: an image sequence's rows at its M-RoPE positions (the QSA attention's rope, the
        // indexer's queries, and -- through the ring's 3-D positions -- its compressed keys at
        // each group's first token); `pos` (the KV slot, the causal mask) stays the index
        let rows: Vec<_> = (0..m).map(|r| (Some(&im.rope), p0 + r)).collect();
        let p3 = super::vision::pos3_of(&rows);
        let b: Vec<u8> = p3.iter().flat_map(|x| x.to_le_bytes()).collect();
        let off = cc.pos3 - rk.dev.consts.addr().as_u64();
        rk.hip
            .memcpy_h2d_at_blocking(&rk.dev.consts, off, &b, &rk.dev.stream)?;
    }
    let Some(k) = rk.pools.kvt.as_ref() else {
        return Ok(cc);
    };
    let mut k = k.borrow_mut();
    k.begin(rk.hip, &rk.dev.stream, &[(slot, p0, p0 + m)], true)?;
    let rows: Vec<(u32, u32)> = (0..m).map(|r| (slot, p0 + r)).collect();
    let (ks, _) = super::kvwire::step_tables_of(&k.tables(), &rows, false);
    ensure!(
        KV_PREFILL_OFF + ks.len() as u64 <= rk.dev.consts.len(),
        "a chunk's per-layer slots past the constants"
    );
    rk.hip
        .memcpy_h2d_at_blocking(&rk.dev.consts, KV_PREFILL_OFF, &ks, &rk.dev.stream)?;
    if let Some(ix) = k.ix.as_ref() {
        // FN51: the chunk's `bt` (identity 0..nb over the slot's own store) = the slot's pages
        let row: Vec<u8> = ix.row(slot)[..cc.nb as usize]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let off = cc.bt - rk.dev.consts.addr().as_u64();
        rk.hip
            .memcpy_h2d_at_blocking(&rk.dev.consts, off, &row, &rk.dev.stream)?;
    }
    cc.kslot = rk.dev.consts.addr().as_u64() + KV_PREFILL_OFF;
    cc.kv_row0 = u64::from(slot) * k.row_bytes();
    Ok(cc)
}

/// FN45b2: where a prefill chunk's per-layer append slots sit in the constants (past the chunk's
/// own tables: < 1 MiB at 8,192 rows; the slots 13 x 8,192 x 8 B).
const KV_PREFILL_OFF: u64 = 4 << 20;

/// The constants of [`decode_tables`]' layout at the rank's constant region.
fn decode_cc(
    rk: &Rank<'_, '_>,
    offs: &[(&'static str, usize)],
    pos: &[u32],
    band: (u32, u32),
) -> ChunkConsts {
    let mut cc = ChunkConsts::at(
        rk.dev.consts.addr().as_u64(),
        offs,
        pos.len() as u32,
        0,
        2,
        rk.pools.nbs,
        rk.dev.cs.addr().as_u64(),
    );
    cc.max_ctx = band.0;
    cc.bt_nb = band.1;
    cc.bar = rk.bar.addr().as_u64();
    cc.qsa_short = saber_kernels::fn19::short_rows(pos, 1);
    cc
}

/// FN15: ONE MTP pass over `m` rows (EAGER): the pre-layer 4 launches (`fn_mtp_norms`,
/// `fc_embedding`, `fc_hidden` at 4m, `fn_mtp_add`) over the embedding rows already in `rk.emb` and
/// the stream `hid`, the MTP decoder layer (the target's QSA kernels over the drafter's OWN KV /
/// indexer pools, the bf16 MoE), the MTP mixer (combine + final mix) and the shared head (every
/// row in the decode form, the last row in the prefill form). `gather` copies rows of the
/// drafter's own stream to `prev` (`(src row, dst row)`). The launch count is asserted against the
/// layer graph: 4 + a QSA layer's + the mixer 2 + the head 1. Returns the logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn mtp_pass(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    m: u32,
    emb: u64,
    hid: u64,
    cc: ChunkConsts,
    st: LayerState,
    gather: &[(u32, u32)],
    all_rows: bool,
    dump: Option<&Path>,
    tail: Option<TailArgs>,
) -> anyhow::Result<u64> {
    // FN51d: a decode-form pass (captured on first use, its arena addresses baked) runs LENT:
    // after a wide chunk (the serve plane's `Cmd::First` chain of a prompt whose last chunk is
    // past the narrow rung) the arena is still the WIDE layout, whose zero / ones / carry regions
    // sit in the tail the next narrow step lends to the expert tier. Decode passes are narrow.
    let t_lend = Instant::now();
    if cc.decode && rk.mtp.as_ref().context("mtp")?.lend {
        lend_for(rk, em, m, false, true)?;
    } else {
        // FN76b: a prefill-form drafter pass past the top's rows reclaims it (no borrow flip)
        lend_for(rk, em, m, !cc.decode, false)?;
    }
    seg(rk, "lend", t_lend);
    let t_l = Instant::now();
    let w = rk.mtp.as_ref().context("mtp")?.w;
    let ev = (
        rk.hip.create_event(EventFlags::Default)?,
        rk.hip.create_event(EventFlags::Default)?,
    );
    rk.hip.record_event(&ev.0, &rk.dev.stream)?;
    // FN16: a decode-form pass replays its graph (captured on first use); the gather copies (their
    // rows move with the walk) stay outside it. FN22: a pass with the device tail is its own graph
    // (keyed by the tail's row-selection table).
    let gkey = (
        m,
        cc.w,
        hid,
        cc.qsa_short,
        tail.map_or(0, |t| t.sel),
        cc.max_ctx,
    );
    let use_graph = {
        let mt = rk.mtp.as_ref().context("mtp")?;
        mt.capture
            && cc.decode
            && dump.is_none()
            && !rk.dev.op_timing
            && !tail.is_some_and(|t| t.i2.is_some_and(|i| i.full_head))
            && mt.chain.as_ref().is_none_or(|c| c.conf.is_none())
    };
    if use_graph && !rk.mtp.as_ref().context("mtp")?.graphs.contains_key(&gkey) {
        rk.hip.sync_stream(&rk.dev.stream)?;
        let (l0, m0) = (rk.dev.launches, rk.dev.memsets);
        rk.hip
            .begin_capture(&rk.dev.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
        let r = mtp_emit(rk, em, m, emb, hid, cc, st, all_rows, tail);
        let g = rk.hip.end_capture(&rk.dev.stream);
        let (lg, strm, got, ids) = r?;
        let g = g?;
        let nodes = rk.hip.graph_node_count(&g)?;
        let (dl, dm) = (rk.dev.launches - l0, rk.dev.memsets - m0);
        ensure!(
            nodes as u64 == dl + dm,
            "MTP pass m {m} w {}: the graph holds {nodes} nodes; the pass issued {dl} launches + {dm} memsets",
            cc.w
        );
        let exec = rk.hip.instantiate(&g)?;
        if rk.rank == 0 {
            println!(
                "fnmtpgraph: captured the drafter pass m {m} w {} ({got} launches, {nodes} nodes)",
                cc.w
            );
        }
        let mt = rk.mtp.as_mut().context("mtp")?;
        mt.launches.insert(format!("d{m}"), got);
        mt.graphs.insert(
            gkey,
            MtpGraph {
                exec,
                _graph: g,
                lg,
                strm,
                ids,
            },
        );
    }
    if use_graph {
        let (lg, strm) = {
            let mt = rk.mtp.as_ref().context("mtp")?;
            let gph = &mt.graphs[&gkey];
            rk.hip.graph_launch(&gph.exec, &rk.dev.stream)?;
            (gph.lg, gph.strm)
        };
        mtp_gather(rk, strm, gather)?;
        rk.hip.record_event(&ev.1, &rk.dev.stream)?;
        pass_done(rk, ev, t_l, tail.is_none())?;
        let ids = rk.mtp.as_ref().context("mtp")?.graphs[&gkey].ids;
        mtp_trace(rk, cc.decode, m, ids)?;
        return Ok(lg);
    }
    let n0 = em.forward_launches();
    let tap0 = em.tap_layer;
    if dump.is_some() && rk.tap_mtp {
        em.tap_layer = 48;
        em.taps.clear();
    }
    let x = em.mtp_pre(&w, emb, hid, u64::from(m))?;
    let outs = {
        let mut ar: Option<&mut dyn Reduce> = Some(&mut rk.pair);
        let mut hook = Resident(std::slice::from_ref(&w.lw));
        emit_layer(
            std::slice::from_mut(em),
            std::slice::from_ref(&w.lw),
            &[st],
            &[LayerIn::Stream(x)],
            &[0],
            &[cc],
            &mut ar,
            &mut hook,
        )?
    };
    let o = outs.into_iter().next().context("one rank")?;
    let (fin, lg, strm) = emit_post_tail(rk, em, &w.fm, &o.pend, m, all_rows, tail)?;
    let got = em.forward_launches() - n0;
    let form = if cc.decode {
        saber_model::qwen4_exp::layer_graph::Form::Decode {
            batch: cc.b,
            attn: saber_kernels::fn6::AttnForm::DecodePair,
        }
    } else {
        saber_model::qwen4_exp::layer_graph::Form::Prefill
    };
    let want = saber_kernels::fn6::MTP_PRE_LAYER_LAUNCHES as u32
        + plan::declared_form_qsa(
            em.rules,
            3,
            2,
            m,
            form,
            em.qsa_sparse(&cc),
            super::exec::MoeTrain::Legacy,
        )
        + 2
        + post_launches(tail, em.head_bf16_dropped, em.fuse86)
        + fn50_extra(&w.lw);
    ensure!(
        got == want,
        "MTP pass m {m}: {got} launches vs declared {want} ({form:?})"
    );
    rk.dev.exec(&rk.table, em)?;
    mtp_gather(rk, strm, gather)?;
    rk.hip.record_event(&ev.1, &rk.dev.stream)?;
    pass_done(rk, ev, t_l, tail.is_none())?;
    mtp_trace(rk, cc.decode, m, o.ids)?;
    let mt = rk.mtp.as_mut().context("mtp")?;
    mt.launches
        .insert(format!("{}{m}", if cc.decode { "d" } else { "p" }), got);
    em.tap_layer = tap0;
    if let Some(d) = dump {
        std::fs::create_dir_all(d)?;
        for (name, a, n) in std::mem::take(&mut em.taps) {
            let b = d2h_in(&rk.dev, &rk.dev.arena, a, n as usize)
                .or_else(|_| d2h_pool(rk, a, n as usize))?;
            std::fs::write(d.join(format!("tap_{name}.rank{}.bin", rk.rank)), b)?;
        }
        let mm = u64::from(m);
        for (name, a, n) in [
            ("mtp_final", fin, mm * H * 2),
            ("mtp_multi", strm, mm * W * 2),
            ("mtp_attn_out", o.att, mm * H * 2),
            ("mtp_mlp_out", o.pend.mo, mm * H * 2),
            ("mtp_rid", o.ids, mm * 40),
        ] {
            let b = d2h_in(&rk.dev, &rk.dev.arena, a, n as usize)?;
            std::fs::write(d.join(format!("{name}.rank{}.bin", rk.rank)), b)?;
        }
    }
    Ok(lg)
}

/// FN50: the fp8 MTP experts' declared extra launches over the bf16 MoE block (two activation
/// quants + the shared expert's bf16 launch beside each fp8 grouped GEMM).
fn fn50_extra(lw: &LayerW) -> u32 {
    if lw.mtp_fp8 {
        (saber_kernels::fn50::MOE_BLOCK_LAUNCHES_FP8 - saber_kernels::moe::LAUNCHES_PER_MOE_BLOCK)
            as u32
    } else {
        0
    }
}

/// FN16: the emission of one decode-form MTP pass (pre 4 + the QSA layer + mixer 2 + head 1) and
/// its execution -- the body a captured pass records. Returns (logits, the drafter's stream, launches).
#[allow(clippy::too_many_arguments)]
fn mtp_emit(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    m: u32,
    emb: u64,
    hid: u64,
    cc: ChunkConsts,
    st: LayerState,
    all_rows: bool,
    tail: Option<TailArgs>,
) -> anyhow::Result<(u64, u64, u32, u64)> {
    let w = rk.mtp.as_ref().context("mtp")?.w;
    let n0 = em.forward_launches();
    let x = em.mtp_pre(&w, emb, hid, u64::from(m))?;
    let outs = {
        let mut ar: Option<&mut dyn Reduce> = Some(&mut rk.pair);
        let mut hook = Resident(std::slice::from_ref(&w.lw));
        emit_layer(
            std::slice::from_mut(em),
            std::slice::from_ref(&w.lw),
            &[st],
            &[LayerIn::Stream(x)],
            &[0],
            &[cc],
            &mut ar,
            &mut hook,
        )?
    };
    let o = outs.into_iter().next().context("one rank")?;
    let (_, lg, strm) = emit_post_tail(rk, em, &w.fm, &o.pend, m, all_rows, tail)?;
    let got = em.forward_launches() - n0;
    let want = saber_kernels::fn6::MTP_PRE_LAYER_LAUNCHES as u32
        + plan::declared_form_qsa(
            em.rules,
            3,
            2,
            m,
            saber_model::qwen4_exp::layer_graph::Form::Decode {
                batch: cc.b,
                attn: saber_kernels::fn6::AttnForm::DecodePair,
            },
            em.qsa_sparse(&cc),
            super::exec::MoeTrain::Legacy,
        )
        + 2
        + post_launches(tail, em.head_bf16_dropped, em.fuse86)
        + fn50_extra(&w.lw);
    ensure!(
        got == want,
        "MTP pass m {m}: {got} launches vs declared {want}"
    );
    rk.dev.exec(&rk.table, em)?;
    Ok((lg, strm, got, o.ids))
}

/// FN18: record a finished MTP pass's routed ids (the MTP layer's route log in the arena, read after
/// a stream sync -- a device-chain pass (FN22) does not sync on its own; rank 0 with
/// `--fn-route-trace` only).
fn mtp_trace(rk: &mut Rank<'_, '_>, decode: bool, m: u32, ids: u64) -> anyhow::Result<()> {
    // FN65: the recorder never syncs the stream for the drafter's routes
    if !rk.trace.as_ref().is_some_and(|t| t.wants_mtp()) {
        return Ok(());
    }
    rk.hip.sync_stream(&rk.dev.stream)?;
    let b = d2h_in(&rk.dev, &rk.dev.arena, ids, m as usize * 40)?;
    let v: Vec<u32> = b
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ensure!(v.iter().all(|&e| e < 512), "an MTP route id out of range");
    let kind = if decode {
        saber_mem::route_trace::StepKind::MtpDecode
    } else {
        saber_mem::route_trace::StepKind::MtpPrefill
    };
    if let Some(tw) = rk.trace.as_mut() {
        tw.mtp(kind, m, &v)?;
    }
    Ok(())
}

/// FN15: copy rows of the drafter's stream (`strm`) to `prev` (`(src row, dst row)`).
fn mtp_gather(rk: &mut Rank<'_, '_>, strm: u64, gather: &[(u32, u32)]) -> anyhow::Result<()> {
    if let Some((out, w, n)) = rk.ahead_gather.take() {
        // FN119 (`ahead`): the round's pass-1 gather (row s * w + acc_s -> prev row s) on the card,
        // acc from the walk's result block
        ensure!(
            gather.is_empty(),
            "FN119: the device gather replaces the host's"
        );
        let prev = rk.mtp.as_ref().context("mtp")?.prev.addr().as_u64();
        let (func, decl) = rk
            .table
            .get(super::dstep::AHEAD_PLUGIN, "fn119a_gather")
            .context("flashnext_ahead::fn119a_gather not loaded")?;
        let mut args = Vec::with_capacity(40);
        for a in [out, strm, prev] {
            args.extend_from_slice(&a.to_le_bytes());
        }
        args.extend_from_slice(&(w as i32).to_le_bytes());
        args.extend_from_slice(&(W as i32).to_le_bytes());
        args.resize(decl.kernarg_segment_size as usize, 0);
        rk.hip
            .launch(func, [n, 1, 1], [256, 1, 1], 0, &rk.dev.stream, &mut args)?;
        return Ok(());
    }
    if gather.is_empty() {
        return Ok(());
    }
    let base = rk.dev.base();
    let prev = &rk.mtp.as_ref().context("mtp")?.prev;
    // FN22: the chain passes' gather is rows 0..n -> 0..n: one copy of the same bytes
    if gather
        .iter()
        .enumerate()
        .all(|(i, &(s, d))| s == i as u32 && d == i as u32)
    {
        rk.hip.run_batch(&[Job::MemcpyD2d {
            dst: prev,
            dst_offset: 0,
            src: &rk.dev.arena,
            src_offset: strm - base,
            bytes: gather.len() as u64 * W * 2,
            stream: &rk.dev.stream,
        }])?;
        return Ok(());
    }
    let jobs: Vec<Job<'_>> = gather
        .iter()
        .map(|&(src, dst)| Job::MemcpyD2d {
            dst: prev,
            dst_offset: u64::from(dst) * W * 2,
            src: &rk.dev.arena,
            src_offset: strm - base + u64::from(src) * W * 2,
            bytes: W * 2,
            stream: &rk.dev.stream,
        })
        .collect();
    rk.hip.run_batch(&jobs)?;
    Ok(())
}

/// FN16 (a drafter lever): the ARGMAX of each row of the logits at `lg` over both vocabulary
/// halves -- per rank the largest bf16 key, the lowest index on a tie; across the pair the larger
/// value, the lower id on a tie: `pick2(..)[r].top[0].0` exactly (the seam's total order), for 8
/// bytes a row across the pair.
pub(super) fn pick_argmax(rk: &mut Rank<'_, '_>, lg: u64, rows: usize) -> anyhow::Result<Vec<u32>> {
    let half = VOCAB as usize / 2;
    let base = rk.rank as usize * half;
    let t = Instant::now();
    let raw = d2h_in(&rk.dev, &rk.dev.arena, lg, rows * half * 2)?;
    seg(rk, "d2h", t);
    let t = Instant::now();
    let mut mine: Vec<u8> = Vec::with_capacity(rows * 8);
    for r in 0..rows {
        let (i, v) = row_argmax(&raw[r * half * 2..(r + 1) * half * 2]);
        mine.extend_from_slice(&((i + base) as u32).to_le_bytes());
        mine.extend_from_slice(&v.to_le_bytes());
    }
    seg(rk, "argmax", t);
    let t = Instant::now();
    let theirs = rk.peer.exchange().xchg_bytes("fn.pickam", &mine)?;
    seg(rk, "xchg", t);
    ensure!(
        theirs.len() == mine.len(),
        "the peer's argmax frame differs"
    );
    Ok((0..rows)
        .map(|r| merge_argmax(argmax_at(&mine, r), argmax_at(&theirs, r)))
        .collect())
}

fn argmax_at(b: &[u8], r: usize) -> (u32, f32) {
    let k = r * 8;
    (
        u32::from_le_bytes(b[k..k + 4].try_into().expect("4")),
        f32::from_le_bytes(b[k + 4..k + 8].try_into().expect("4")),
    )
}

/// The larger value of two halves' argmaxes; the lower id on a tie (the seam's order).
#[must_use]
pub fn merge_argmax(a: (u32, f32), b: (u32, f32)) -> u32 {
    match b.1.total_cmp(&a.1) {
        std::cmp::Ordering::Greater => b.0,
        std::cmp::Ordering::Less => a.0,
        std::cmp::Ordering::Equal => a.0.min(b.0),
    }
}

/// The largest entry of a bf16 row (the bf16 key order; the lowest index on a tie).
#[must_use]
pub fn row_argmax(row: &[u8]) -> (usize, f32) {
    let mut best = (0usize, 0u16);
    for (i, c) in row.chunks_exact(2).enumerate() {
        let key = bf16_key(u16::from_le_bytes([c[0], c[1]]));
        if i == 0 || key > best.1 {
            best = (i, key);
        }
    }
    (best.0, bf16_at(row, best.0))
}

/// FN16: the drafts of `rows` rows of the logits at `lg`: the argmax lever, or the seam's top-1.
fn draft_ids(rk: &mut Rank<'_, '_>, lg: u64, rows: usize) -> anyhow::Result<Vec<u32>> {
    if rk.mtp.as_ref().is_some_and(|m| m.argmax) {
        pick_argmax(rk, lg, rows)
    } else {
        Ok(pick2(rk, lg, rows, &vec![RowReq::default(); rows])?
            .into_iter()
            .map(|o| o.top[0].0)
            .collect())
    }
}

/// FN15: the MTP chain's FIRST pass after a prefill (one sequence, the prefill form over every
/// prompt row: row p = the target's stream at p + the token at p + 1, the last row the committed
/// first token; the drafter's KV / indexer pools of the sequence's slot fill for positions 0..n).
/// `refd` = the host-reference gate arm: FN3's `resid[47]` (bf16) and FN3's first generated token
/// replace the engine's own, and the pass's outputs are dumped under `out/mtpref/<id>`.
#[allow(clippy::too_many_arguments)]
fn mtp_first(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    lv: &Live,
    ids: &[u32],
    first: u32,
    refd: Option<&Path>,
    pid: &str,
    out: &Path,
) -> anyhow::Result<(u32, Vec<(u32, f32)>)> {
    let m = ids.len() as u32;
    ensure!(m >= 1 && lv.len >= m, "the prefill precedes its MTP pass");
    let p0 = lv.len - m;
    let mut toks: Vec<u32> = ids[1..].to_vec();
    toks.push(first);
    if let Some(d) = refd {
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(d.join(pid).join("meta.json"))?)?;
        let g0 = meta["generated_ids"][0]
            .as_u64()
            .context("FN3's first generated token")? as u32;
        *toks.last_mut().context("rows")? = g0;
        let raw = super::run::npy_payload(&d.join(pid).join("resid.npy"))?;
        let per = m as usize * W as usize * 4;
        ensure!(raw.len() == 48 * per, "resid.npy [48, {m}, 10240] f32");
        let l47 = &raw[47 * per..48 * per];
        let bf: Vec<u8> = l47
            .chunks_exact(4)
            .flat_map(|c| {
                let x = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let r = x.wrapping_add(0x7fff + ((x >> 16) & 1));
                ((r >> 16) as u16).to_le_bytes()
            })
            .collect();
        buf_h2d(rk, BufK::Hid, &bf)?;
    }
    let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
    let dump = refd.map(|_| out.join("mtpref").join(pid));
    let lg = mtp_rows(
        rk,
        em,
        recs,
        lv.slot,
        &toks,
        p0,
        lv.chunks.saturating_sub(1),
        hid,
        true,
        dump.as_deref(),
    )?;
    let o = pick2(rk, lg, 1, &[RowReq::default()])?
        .pop()
        .context("one row")?;
    let d1 = o.top[0].0;
    rk.mtp.as_mut().context("mtp")?.first[lv.slot as usize] = Some((d1, p0 + m - 1));
    Ok((d1, o.top[..TOPK].to_vec()))
}

/// FN15 / FN16: the drafter's PREFILL-form pass over `toks.len()` rows of pool slot `slot`'s own
/// drafter pools at positions `p0..`: row r = the target's stream at `hid + r * W * 2` + the token
/// `toks[r]` (the token at the row's position + 1). Pieces of <= 512 rows (fc_hidden's M = 4 x rows
/// stays in a 2,048 band); a later piece reads the earlier pieces' K/V from the drafter's pool.
/// `gather_last` copies the last row's drafter stream to `prev` row `slot` (the chain's input).
/// Returns the last piece's logits (its last row is the prefill form's head row).
#[allow(clippy::too_many_arguments)]
fn mtp_rows(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    slot: u32,
    toks: &[u32],
    p0: u32,
    chunk: u32,
    hid: u64,
    gather_last: bool,
    dump: Option<&Path>,
) -> anyhow::Result<u64> {
    let m = toks.len() as u32;
    ensure!(m >= 1, "an empty drafter pass");
    mtp_rows_emb(rk, recs, slot, toks, p0)?;
    let st = rk.pools.seq(slot, LAYERS as usize);
    let mut lg = 0;
    for a in (0..m).step_by(MTP_PIECE as usize) {
        let mc = (m - a).min(MTP_PIECE);
        let last = a + mc == m;
        let t = Instant::now();
        let cc = kv_consts(rk, mc, p0 + a, chunk, slot)?;
        seg(rk, "rows_consts", t);
        let dp = dump.map(|d| d.join(format!("piece{a}")));
        let gather: Vec<(u32, u32)> = if last && gather_last {
            vec![(mc - 1, slot)]
        } else {
            Vec::new()
        };
        lg = mtp_pass(
            rk,
            em,
            mc,
            rk.emb.addr().as_u64() + u64::from(a) * H * 2,
            hid + u64::from(a) * W * 2,
            cc,
            st,
            &gather,
            false,
            dp.as_deref(),
            None,
        )?;
    }
    Ok(lg)
}

/// FN15 / FN55b: the embedding rows of a prefill-form drafter pass (`toks` at positions `p0..`)
/// staged into `rk.emb` -- an image prompt's rows read the tower's rows too (row r embeds the token
/// at index p0 + r + 1: upstream's proposer hands the drafter the multimodal `inputs_embeds`).
fn mtp_rows_emb(
    rk: &mut Rank<'_, '_>,
    recs: &PackRecs,
    slot: u32,
    toks: &[u32],
    p0: u32,
) -> anyhow::Result<()> {
    let m = toks.len() as u32;
    let t = Instant::now();
    let e = embed_rows(recs, toks)?;
    buf_h2d(rk, BufK::Emb, &e)?;
    seg(rk, "rows_emb", t);
    if let Some(im) = rk.ropes.get(slot as usize).cloned().flatten() {
        let runs = im.rope.scatter_runs(p0 + 1, m);
        if !runs.is_empty() {
            let v = rk.vis.as_mut().context("image rows on a text-only boot")?;
            v.scatter(
                rk.hip,
                &rk.dev.stream,
                im.epoch,
                &runs,
                &rk.emb.own,
                rk.emb.off,
            )?;
        }
    }
    Ok(())
}

/// FN116 (`--fn-first-step kv`): the drafter's rows `toks` at positions `p0..` of pool slot `slot`
/// through the MTP layer's K-SIDE FRONT only (`exec::emit_layer_kv_front`: pre 4 + the mix + the
/// projection + q / k norm + rope + the indexer + the KV append, EAGER, pieces of <= 512 rows as
/// [`mtp_rows`]): the K / V / indexer state every later drafter pass reads of these rows. Their
/// attention output, MoE, mixer and head are read by nothing (one MTP layer; the chain reads the
/// LAST row's stream only) and do not run. The pieces' device time is event-timed into the
/// device chain's events (read by the round's `finish_chain`).
#[allow(clippy::too_many_arguments)]
fn mtp_kv_rows(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    slot: u32,
    toks: &[u32],
    p0: u32,
    chunk: u32,
    hid: u64,
) -> anyhow::Result<()> {
    let m = toks.len() as u32;
    if m == 0 {
        return Ok(());
    }
    mtp_rows_emb(rk, recs, slot, toks, p0)?;
    let st = rk.pools.seq(slot, LAYERS as usize);
    for a in (0..m).step_by(MTP_PIECE as usize) {
        let mc = (m - a).min(MTP_PIECE);
        let t = Instant::now();
        let cc = kv_consts(rk, mc, p0 + a, chunk, slot)?;
        seg(rk, "rows_consts", t);
        let t = Instant::now();
        // FN76b: a prefill-form drafter pass past the top's rows reclaims it (no borrow flip)
        lend_for(rk, em, mc, true, false)?;
        seg(rk, "lend", t);
        let t_l = Instant::now();
        let ev = (
            rk.hip.create_event(EventFlags::Default)?,
            rk.hip.create_event(EventFlags::Default)?,
        );
        rk.hip.record_event(&ev.0, &rk.dev.stream)?;
        let w = rk.mtp.as_ref().context("mtp")?.w;
        let n0 = em.forward_launches();
        let x = em.mtp_pre(
            &w,
            rk.emb.addr().as_u64() + u64::from(a) * H * 2,
            hid + u64::from(a) * W * 2,
            u64::from(mc),
        )?;
        super::exec::emit_layer_kv_front(em, &w.lw, &st, x, &cc)?;
        let got = em.forward_launches() - n0;
        let want = saber_kernels::fn6::MTP_PRE_LAYER_LAUNCHES as u32
            + plan::declared_qsa_kv_front(
                em.rules,
                3,
                2,
                mc,
                saber_model::qwen4_exp::layer_graph::Form::Prefill,
            );
        ensure!(
            got == want,
            "FN116: the MTP K-side front m {mc}: {got} launches vs declared {want}"
        );
        rk.dev.exec(&rk.table, em)?;
        rk.hip.record_event(&ev.1, &rk.dev.stream)?;
        pass_done(rk, ev, t_l, false)?;
        rk.f116_front += 1;
    }
    Ok(())
}

/// FN15: the MTP chain's passes 2..=k for the batch (pool slots `0..n` padded to `bucket`):
/// pass j reads the drafter's own stream (`prev`) and the previous draft, at `pos[s] + j - 1`.
/// `drafts[s]` holds the first pass's draft on entry; returns with k drafts per live sequence.
fn mtp_chain(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    drafts: &mut [Vec<u32>],
    pos: &[u32],
    bucket: u32,
) -> anyhow::Result<()> {
    let k = rk.mtp.as_ref().context("mtp")?.k;
    mtp_chain_k(rk, em, recs, drafts, pos, bucket, k)
}

/// [`mtp_chain`] to `k` drafts a sequence.
pub(super) fn mtp_chain_k(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    drafts: &mut [Vec<u32>],
    pos: &[u32],
    bucket: u32,
    k: u32,
) -> anyhow::Result<()> {
    if rk.mtp.as_ref().is_some_and(|m| m.chain.is_some()) {
        return chain_from_tokens(rk, em, recs, drafts, pos, bucket, k);
    }
    let n = drafts.len();
    let b = bucket as usize;
    for j in 2..=k {
        let mut toks: Vec<u32> = drafts.iter().map(|d| *d.last().unwrap_or(&0)).collect();
        let mut p: Vec<u32> = pos.iter().map(|x| x + j - 1).collect();
        toks.resize(b, 0);
        p.resize(b, 0);
        let t = Instant::now();
        let e = embed_rows(recs, &toks)?;
        buf_h2d(rk, BufK::Emb, &e)?;
        seg(rk, "emb", t);
        let t = Instant::now();
        let cc = decode_consts(rk, &p, n)?;
        seg(rk, "consts", t);
        let st = rk.pools.batch(LAYERS as usize, 0);
        let hid = rk.mtp.as_ref().context("mtp")?.prev.addr().as_u64();
        let gather: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, i)).collect();
        let emb = rk.emb.addr().as_u64();
        let lg = mtp_pass(rk, em, bucket, emb, hid, cc, st, &gather, true, None, None)?;
        let outs = draft_ids(rk, lg, b)?;
        for (i, d) in drafts.iter_mut().enumerate() {
            d.push(outs[i]);
        }
    }
    Ok(())
}

// ============================================================================ FN30b: the drafter-round diagnostic

/// FN30b probe slots (6 u64 each; one per probe point).
const FN30B_SLOTS: u32 = 16_384;
/// FN30b: the clock probe's dependent FMA count (~20 us at 2.9 GHz).
const FN30B_ITERS: i32 = 12_000;

/// FN30b (DIAGNOSTIC; every target and drafter byte unchanged): the drafter round's per-pass
/// device times (always recorded, `fn30b.passes` in the run JSON) and, under
/// `--fn-diag-probe <fn30b_probe.hsaco>` (tools/capture/fn30b), EAGER probes on the rank's
/// stream at (a) the target step's end, (b) the drafter round's start, (c) the chain's end: the
/// shader clock + a device stamp (`fn30b_clk`) and a 128 MiB VRAM read of the arena (`fn30b_bw`,
/// event-timed). `--fn-diag-gap-us N`: the stream drained + N us of host sleep before every
/// drafter round (the device idle; a transient-state test).
pub(super) struct Fn30bDiag {
    module: Option<saber_hip::Module>,
    rec: Option<DevicePtr>,
    slot: u32,
    gap_us: u64,
    /// The bucket of the round in flight (set at (a) / (b)).
    bucket: u32,
    pend: Vec<(u32, u8, u32, saber_hip::Event, saber_hip::Event)>,
    /// (slot, point, bucket, VRAM read ms).
    probes: Vec<(u32, u8, u32, f64)>,
    /// (bucket, per-pass device ms) of every drafter round after a verify step.
    passes: Vec<(u32, Vec<f64>)>,
}

impl Fn30bDiag {
    fn new(hip: &HipThread, rank: u32, f: &FnFlags) -> anyhow::Result<Self> {
        let gap_us = f.fn_diag_gap_us;
        let (module, rec) = match &f.fn_diag_probe {
            Some(p) => {
                let img =
                    std::fs::read(p).with_context(|| format!("--fn-diag-probe {}", p.display()))?;
                let m = hip.load_module(&img)?;
                let rec = hip.malloc(u64::from(FN30B_SLOTS) * 48)?;
                println!("fn30b rank{rank}: DIAGNOSTIC probes loaded from {} (eager clk + VRAM read at target end / drafter start / chain end); gap {gap_us} us", p.display());
                (Some(m), Some(rec))
            }
            None => {
                if gap_us > 0 {
                    println!("fn30b rank{rank}: DIAGNOSTIC drafter gap {gap_us} us");
                }
                (None, None)
            }
        };
        Ok(Self {
            module,
            rec,
            slot: 0,
            gap_us,
            bucket: 0,
            pend: Vec::new(),
            probes: Vec::new(),
            passes: Vec::new(),
        })
    }
}

/// FN30b: one probe (point 0 = the target's end, 1 = the drafter's start, 2 = the chain's end).
fn fn30b_probe(rk: &mut Rank<'_, '_>, point: u8) -> anyhow::Result<()> {
    if rk.diag.module.is_none() || rk.diag.slot >= FN30B_SLOTS {
        return Ok(());
    }
    let slot = rk.diag.slot;
    rk.diag.slot += 1;
    let bucket = rk.diag.bucket;
    let rec = rk.diag.rec.as_ref().context("fn30b rec")?.addr().as_u64();
    let m = rk.diag.module.as_ref().context("fn30b module")?;
    let clk = rk.hip.get_function(m, "fn30b_clk")?;
    let mut ka = Vec::with_capacity(16);
    ka.extend_from_slice(&rec.to_le_bytes());
    ka.extend_from_slice(&(slot as i32).to_le_bytes());
    ka.extend_from_slice(&FN30B_ITERS.to_le_bytes());
    rk.hip
        .launch(&clk, [1, 1, 1], [64, 1, 1], 0, &rk.dev.stream, &mut ka)?;
    let bw = rk.hip.get_function(m, "fn30b_bw")?;
    let n16 = rk.dev.arena.len().min(128 << 20) / 16;
    let mut kb = Vec::with_capacity(32);
    kb.extend_from_slice(&rk.dev.arena.addr().as_u64().to_le_bytes());
    kb.extend_from_slice(&n16.to_le_bytes());
    kb.extend_from_slice(&rec.to_le_bytes());
    kb.extend_from_slice(&(slot as i32).to_le_bytes());
    kb.extend_from_slice(&0i32.to_le_bytes());
    let e0 = rk.hip.create_event(EventFlags::Default)?;
    let e1 = rk.hip.create_event(EventFlags::Default)?;
    rk.hip.record_event(&e0, &rk.dev.stream)?;
    rk.hip
        .launch(&bw, [1024, 1, 1], [256, 1, 1], 0, &rk.dev.stream, &mut kb)?;
    rk.hip.record_event(&e1, &rk.dev.stream)?;
    rk.diag.pend.push((slot, point, bucket, e0, e1));
    Ok(())
}

/// FN30b: read the pending probes' VRAM-read events (after a sync of the stream).
fn fn30b_resolve(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    for (slot, point, bucket, e0, e1) in std::mem::take(&mut rk.diag.pend) {
        let ms = f64::from(rk.hip.elapsed_ms(&e0, &e1)?);
        rk.diag.probes.push((slot, point, bucket, ms));
    }
    Ok(())
}

/// FN30b: the run JSON's `fn30b` object (the probe records read back once, at the run's end).
fn fn30b_report(rk: &mut Rank<'_, '_>) -> anyhow::Result<Value> {
    fn30b_resolve(rk)?;
    let mut recs = Vec::new();
    if let Some(rec) = rk.diag.rec.as_ref() {
        let n = rk.diag.slot as usize * 48;
        if n > 0 {
            let mut b = vec![0u8; n];
            rk.hip.sync_stream(&rk.dev.stream)?;
            rk.hip.memcpy_d2h_blocking(&mut b, rec, &rk.dev.stream)?;
            let u = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().expect("8"));
            for (slot, point, bucket, ms) in &rk.diag.probes {
                let o = *slot as usize * 6;
                recs.push(json!([
                    point,
                    bucket,
                    u(o),
                    u(o + 1),
                    u(o + 2),
                    u(o + 3),
                    ms
                ]));
            }
        }
    }
    let passes: Vec<Value> = rk.diag.passes.iter().map(|(b, v)| json!([b, v])).collect();
    Ok(json!({"gap_us": rk.diag.gap_us, "passes": passes,
              "probe_cols": ["point", "bucket", "w0", "c0", "w1", "c1", "vram_read_ms"], "probes": recs}))
}

// ============================================================================ FN22: the device chain

/// FN22: a drafter round's host time by segment (the composition of the round trip).
fn seg(rk: &mut Rank<'_, '_>, name: &'static str, t: Instant) {
    if let Some(mt) = rk.mtp.as_mut() {
        let e = mt.seg.entry(name).or_insert((0.0, 0));
        e.0 += t.elapsed().as_secs_f64() * 1e3;
        e.1 += 1;
    }
}

/// FN22: the end of a drafter pass's host part: the host chain waits for the pass (its logits are
/// read next); a device-chain pass keeps its events for the chain's one sync.
fn pass_done(
    rk: &mut Rank<'_, '_>,
    ev: (saber_hip::Event, saber_hip::Event),
    t_l: Instant,
    sync: bool,
) -> anyhow::Result<()> {
    seg(rk, "launch", t_l);
    if sync {
        let t = Instant::now();
        rk.hip.sync_event(&ev.1)?;
        seg(rk, "wait", t);
        let ms = f64::from(rk.hip.elapsed_ms(&ev.0, &ev.1)?);
        rk.mtp.as_mut().context("mtp")?.pass_ms.push(ms);
    } else {
        rk.mtp.as_mut().context("mtp")?.evs.push(ev);
    }
    Ok(())
}

/// FN22: the device tail of a drafter pass (`saber_kernels::fn22`): the argmax of rows `t.sel` of
/// the logits `lg` over this rank's vocabulary half, the exchange payload (the candidate's
/// embedding row + key + id), the certified EXACT reduce (the payload's -0.0 half is the identity),
/// the pair's pick -> the next pass's embedding rows + the chain's id table.
fn emit_tail(
    em: &mut Emitter<'_>,
    pair: &mut ArPair,
    rank: u32,
    lg: u64,
    x: u64,
    m: u64,
    t: TailArgs,
) -> anyhow::Result<Option<I2Last>> {
    use saber_kernels::fn22::{IDS_STRIDE, PAY_ROW, SPLITS};
    use saber_model::qwen4_exp::layer_graph::Group;
    let half = VOCAB / 2;
    ensure!(
        half % 8 == 0 && (1..=IDS_STRIDE).contains(&t.rows),
        "the chain's tail: {} rows over a {half}-entry half",
        t.rows
    );
    let rows = u64::from(t.rows);
    let th = Arg::I(i32::from(t.tie_high));
    let part = em.arena.take(rows * u64::from(SPLITS) * 8, Seed::None)?;
    let mut last = None;
    // FN86d family 10: the int2 tail under the W8 head runs its glue fused (one site a tail)
    let f10 = t.i2.is_some_and(|i| i.head8.is_some())
        && em.fuse86_site(saber_kernels::flashnext::fuse86::F10);
    if let Some(i2) = t.i2 {
        last = Some(emit_int2_part(em, x, m, part, &t, &i2, f10)?);
    } else {
        em.run(
            "fn_draft_argmax_part",
            [u64::from(SPLITS), rows, 1],
            256,
            vec![
                Arg::P(lg),
                Arg::P(t.sel),
                Arg::P(part),
                Arg::I(half as i32),
                th.clone(),
            ],
            Group::Head,
        )?;
    }
    let n = rows * u64::from(PAY_ROW);
    let pay = em.arena.take(n * 2, Seed::None)?;
    // LR-N: the coupled tail replaces family 10's fused argmax + pack (the same launch count)
    ensure!(
        !t.couple || (f10 && last.is_some() && !t.tie_high),
        "--spec-couple on: the coupled tail needs the int2 draft head under FN86d's family 10 \
         (the W8 head, `--fn-fuse` with f10) and the served tie law"
    );
    if let (true, Some(l)) = (t.couple, last) {
        em.run(
            "fn_couple_acp",
            [rows, 1, 1],
            256,
            vec![
                Arg::P(l.sc),
                Arg::P(l.cand),
                Arg::P(t.sel),
                Arg::P(part),
                Arg::I(l.k as i32),
                Arg::I(l.ldc as i32),
                Arg::I(l.k as i32),
                Arg::P(t.embh),
                Arg::P(pay),
                Arg::I(half as i32),
                Arg::I(rank as i32),
                Arg::P(t.ctr),
            ],
            Group::Head,
        )?;
    } else if let (true, Some(l)) = (f10, last) {
        // FN86d family 10: the candidates' argmax + the pack, one launch (byte-equal, card 1)
        em.run(
            saber_kernels::flashnext::fuse86::F10_ACP,
            [rows, 1, 1],
            256,
            vec![
                Arg::P(l.sc),
                Arg::P(l.cand),
                Arg::P(t.sel),
                Arg::P(part),
                Arg::I(l.k as i32),
                Arg::I(l.ldc as i32),
                Arg::I(l.k as i32),
                Arg::P(t.embh),
                Arg::P(pay),
                Arg::I(half as i32),
                Arg::I(rank as i32),
                th.clone(),
            ],
            Group::Head,
        )?;
    } else {
        em.run(
            "fn_draft_pack",
            [rows, 1, 1],
            256,
            vec![
                Arg::P(part),
                Arg::P(t.sel),
                Arg::P(t.embh),
                Arg::P(pay),
                Arg::I(half as i32),
                Arg::I(rank as i32),
                th.clone(),
            ],
            Group::Head,
        )?;
    }
    // FN140: the chain's exchange is EXACT by the PAYLOAD band alone, whatever the verify width:
    // it carries one row a sequence (<= IDS_STRIDE = 8 = the widest decode bucket) of PAY_ROW
    // (2 x H + 16 lanes) bf16 = <= 82,176 B < 128 KiB, so the decode-form law is never needed
    ensure!(
        n * 2 < pair.rules.quant_min_bytes,
        "the chain's exchange ({} B, {rows} rows <= IDS_STRIDE) must ride the EXACT reduce by the \
         payload band",
        n * 2
    );
    let red = pair.reduce(std::slice::from_mut(em), &[pay], n, false)?[0];
    if t.couple {
        em.run(
            "fn_couple_select",
            [1, 1, 1],
            256,
            vec![
                Arg::P(red),
                Arg::P(t.emb),
                Arg::P(t.ids),
                Arg::P(t.ctr),
                Arg::I(t.rows as i32),
            ],
            Group::Head,
        )?;
        return Ok(last);
    }
    em.run(
        "fn_draft_select",
        [1, 1, 1],
        256,
        vec![
            Arg::P(red),
            Arg::P(t.emb),
            Arg::P(t.ids),
            Arg::P(t.ctr),
            Arg::I(t.rows as i32),
            th,
        ],
        Group::Head,
    )?;
    Ok(last)
}

/// FN33: launches a drafter pass's post adds after the MTP mixer: the head (1; 0 on an int2 tail
/// outside the check arm; FN61g: 2 -- act quant + the W8 GEMM -- when `head_w8`, the bf16 head
/// not resident) + the device tail (FN22's 4, or the int2 tail's 9 + FN22's last 3).
fn post_launches(tail: Option<TailArgs>, head_w8: bool, fuse86: u32) -> u32 {
    let head = 1 + u32::from(head_w8);
    match tail {
        None => head,
        Some(t) => match t.i2 {
            None => head + saber_kernels::fn22::TAIL_LAUNCHES,
            Some(i) => {
                // FN86d family 10 (the int2 tail under the W8 head): 8 glue launches -> 4
                let f10 =
                    if i.head8.is_some() && fuse86 & saber_kernels::flashnext::fuse86::F10 != 0 {
                        saber_kernels::flashnext::fuse86::F10_SAVED
                    } else {
                        0
                    };
                head * u32::from(i.full_head)
                    + saber_kernels::fn33::PRE_LAUNCHES
                    + saber_kernels::fn33::POST_LAUNCHES
                    - f10
            }
        },
    }
}

/// FN33: a drafter pass's final mix, its head (skipped on an int2 tail outside the check arm) and
/// its device tail; records the int2 tail's candidate buffers for the check arm.
#[allow(clippy::too_many_arguments)]
fn emit_post_tail(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    fm: &HcW,
    pend: &super::exec::Pending,
    m: u32,
    all_rows: bool,
    tail: Option<TailArgs>,
) -> anyhow::Result<(u64, u64, u64)> {
    let with_head = tail.and_then(|t| t.i2).is_none_or(|i| i.full_head);
    let (fin, lg, strm) = emit_post3_opt(em, fm, rk.head, pend, u64::from(m), all_rows, with_head)?;
    if let Some(t) = tail {
        let last = emit_tail(em, &mut rk.pair, rk.rank, lg, fin, u64::from(m), t)?;
        if let Some(dc) = rk.mtp.as_mut().and_then(|x| x.chain.as_mut()) {
            dc.i2last = last;
        }
    }
    Ok((fin, lg, strm))
}

/// FN33: the int2 tail up to FN22's pack -- the selected rows of the head input `x` (`[m, H]`,
/// all rows) gathered to a zero-padded 16-row block, their group sums, the 27B's int2 scorer over
/// the rank's vocabulary half (block maxima, 8 a 64-column block), the exact top-K of the block
/// maxima, their vocabulary columns, the candidates' bf16 head rows, the head's OWN GEMM cell (the
/// route of `m` rows) on them, and the argmax of each row's exact scores as FN22's split-0 partial.
fn emit_int2_part(
    em: &mut Emitter<'_>,
    x: u64,
    m: u64,
    part: u64,
    t: &TailArgs,
    i2: &I2Args,
    f10: bool,
) -> anyhow::Result<I2Last> {
    use saber_kernels::flashnext::fuse86;
    use saber_kernels::fn33::{nblk, rerank_cols, KC, NG, ROW_BYTES, XBUF_ROWS, XROWS};
    use saber_model::qwen4_exp::layer_graph::Group;
    let half = VOCAB / 2;
    let nb = nblk(half as u32);
    let nbk = u64::from(nb * KC);
    let rows = t.rows;
    let kk = i2.k;
    ensure!(
        rows <= XROWS && x != 0,
        "the int2 tail: {rows} rows over a 16-row tile"
    );
    ensure!(
        !f10 || (i2.head8.is_some() && i2.head == 0),
        "FN86d family 10 runs the W8-head rerank only"
    );
    let r = u64::from(rows);
    let xp = em.arena.take(u64::from(XBUF_ROWS) * H * 2, Seed::None)?;
    if !f10 {
        em.run(
            "fn_draft_x_gather",
            [u64::from(XBUF_ROWS), 1, 1],
            256,
            vec![Arg::P(x), Arg::P(t.sel), Arg::P(xp), Arg::I(rows as i32)],
            Group::Head,
        )?;
    }
    let xs = em
        .arena
        .take(u64::from(XROWS) * NG as u64 * 4, Seed::None)?;
    if !f10 {
        em.run(
            "dflash_head_group_sum_g128",
            [(u64::from(XROWS) * NG as u64).div_ceil(8), 1, 1],
            256,
            vec![
                Arg::P(xp),
                Arg::P(xs),
                Arg::I(XROWS as i32),
                Arg::I(H as i32),
                Arg::I(NG as i32),
            ],
            Group::Head,
        )?;
    }
    let y = em.arena.take(u64::from(XROWS) * half * 2, Seed::None)?;
    let bmax = em.arena.take(u64::from(XROWS) * nbk * 4, Seed::None)?;
    let bidx = em.arena.take(u64::from(XROWS) * nbk * 4, Seed::None)?;
    // FN86d family 10: every buffer the chain takes, in the chain's order (the arena unchanged),
    // then the row gather + group sums + the rerank's act quant as ONE launch
    let f10b = if f10 {
        let thresh = em.arena.take(r * 8, Seed::None)?;
        let vals = em.arena.take(r * u64::from(kk) * 4, Seed::None)?;
        let pos = em.arena.take(r * u64::from(kk) * 4, Seed::None)?;
        let cand = em.arena.take(r * u64::from(kk) * 4, Seed::None)?;
        let q = em.arena.take(r * H, Seed::None)?;
        let qs = em.arena.take(r * (H / 128) * 4, Seed::None)?;
        em.run(
            fuse86::F10_XG,
            [u64::from(XBUF_ROWS), 1, 1],
            256,
            vec![
                Arg::P(x),
                Arg::P(t.sel),
                Arg::P(xp),
                Arg::I(rows as i32),
                Arg::P(xs),
                Arg::P(q),
                Arg::P(qs),
            ],
            Group::Head,
        )?;
        Some((thresh, vals, pos, cand, q, qs))
    } else {
        None
    };
    em.run(
        "dflash_draft_head_int2_m16n64x2g128_lx_topk",
        [u64::from(nb).div_ceil(2), 1, 1],
        128,
        vec![
            Arg::P(xp),
            Arg::P(xs),
            Arg::P(i2.wq),
            Arg::P(i2.s),
            Arg::P(i2.zs),
            Arg::P(y),
            Arg::P(bmax),
            Arg::P(bidx),
            Arg::I(XROWS as i32),
            Arg::I(half as i32),
            Arg::I(H as i32),
            Arg::I(ROW_BYTES as i32),
            Arg::I(NG as i32),
            Arg::I(NG as i32),
            Arg::I(nb as i32),
        ],
        Group::Head,
    )?;
    let thresh = match f10b {
        Some(b) => b.0,
        None => em.arena.take(r * 8, Seed::None)?,
    };
    em.run(
        "dflash_topk_f32_thresh_k80",
        [r, 1, 1],
        1024,
        vec![
            Arg::P(bmax),
            Arg::P(thresh),
            Arg::I(rows as i32),
            Arg::I(nbk as i32),
            Arg::I(kk as i32),
        ],
        Group::Head,
    )?;
    let cand;
    if let Some((_, vals, pos, c, _, _)) = f10b {
        // FN86d family 10: the top-80 gather + the candidate columns, one launch
        cand = c;
        em.run(
            fuse86::F10_TOPK,
            [r, 1, 1],
            1024,
            vec![
                Arg::P(bmax),
                Arg::P(thresh),
                Arg::P(vals),
                Arg::P(pos),
                Arg::P(bidx),
                Arg::P(cand),
                Arg::I(rows as i32),
                Arg::I(nbk as i32),
                Arg::I(kk as i32),
            ],
            Group::Head,
        )?;
    } else {
        let vals = em.arena.take(r * u64::from(kk) * 4, Seed::None)?;
        let pos = em.arena.take(r * u64::from(kk) * 4, Seed::None)?;
        em.run(
            "dflash_topk_f32_gather_k80",
            [r, 1, 1],
            1024,
            vec![
                Arg::P(bmax),
                Arg::P(thresh),
                Arg::P(vals),
                Arg::P(pos),
                Arg::I(rows as i32),
                Arg::I(nbk as i32),
                Arg::I(kk as i32),
            ],
            Group::Head,
        )?;
        cand = em.arena.take(r * u64::from(kk) * 4, Seed::None)?;
        em.run(
            "dflash_head_cand_gather",
            [(r * u64::from(kk)).div_ceil(256), 1, 1],
            256,
            vec![
                Arg::P(bidx),
                Arg::P(pos),
                Arg::P(cand),
                Arg::I(rows as i32),
                Arg::I(nbk as i32),
                Arg::I(kk as i32),
            ],
            Group::Head,
        )?;
    }
    let ncols = u64::from(rerank_cols(rows, kk));
    let sc;
    if let Some((hw, hs)) = i2.head8 {
        // FN61g: the bf16 head is not resident -- the candidates are scored from the W8 head the
        // verify step reads: the rows' per-128 act quant (the W8 head's own producer, row-local)
        // + the gather GEMM (each column the W8 head GEMM's logit of its candidate, bit for bit).
        // 2 launches, as the bf16 form's row gather + head cell.
        use saber_kernels::fn50;
        ensure!(
            i2.head == 0 && ncols % 64 == 0,
            "the W8 rerank: no bf16 head, 64-column tiles"
        );
        let (q, qs) = match f10b {
            Some(b) => (b.4, b.5),
            None => (
                em.arena.take(r * H, Seed::None)?,
                em.arena.take(r * (H / 128) * 4, Seed::None)?,
            ),
        };
        if f10b.is_none() {
            em.run(
                fn50::ACT_QUANT_SYMBOL,
                [fn50::act_quant_blocks(r, H), 1, 1],
                fn50::ACT_QUANT_THREADS,
                vec![
                    Arg::P(xp),
                    Arg::P(q),
                    Arg::P(qs),
                    Arg::I(rows as i32),
                    Arg::I(H as i32),
                ],
                Group::Head,
            )?;
        }
        sc = em.arena.take(r * ncols * 2, Seed::None)?;
        em.run(
            T8R_GATHER,
            [ncols / 64, 1, 1],
            128,
            vec![
                Arg::P(q),
                Arg::P(hw),
                Arg::P(qs),
                Arg::P(hs),
                Arg::P(cand),
                Arg::P(sc),
                Arg::I(rows as i32),
                Arg::I(ncols as i32),
                Arg::I(H as i32),
                Arg::I(rows as i32),
                Arg::I(kk as i32),
                Arg::I(kk as i32),
                Arg::I(ncols as i32),
            ],
            Group::Head,
        )?;
    } else {
        let wc = em.arena.take(ncols * H * 2, Seed::None)?;
        em.run(
            "fn_draft_wgather",
            [ncols, 1, 1],
            256,
            vec![
                Arg::P(i2.head),
                Arg::P(cand),
                Arg::P(wc),
                Arg::I(rows as i32),
                Arg::I(kk as i32),
                Arg::I(kk as i32),
            ],
            Group::Head,
        )?;
        sc = em.arena.take(r * ncols * 2, Seed::None)?;
        em.gemm_cell_at("head.tp2", m, half, xp, wc, sc, r, ncols, H, Group::Head)?;
    }
    if !f10 {
        em.run(
            "fn_draft_argmax_cand",
            [r, 1, 1],
            32,
            vec![
                Arg::P(sc),
                Arg::P(cand),
                Arg::P(t.sel),
                Arg::P(part),
                Arg::I(kk as i32),
                Arg::I(ncols as i32),
                Arg::I(kk as i32),
                Arg::I(i32::from(t.tie_high)),
            ],
            Group::Head,
        )?;
    }
    Ok(I2Last {
        cand,
        sc,
        ldc: ncols as u32,
        rows,
        k: kk,
    })
}

/// FN33: the int2 draft head's pack, built ONCE at boot on the card from the rank's bf16 head half
/// (`fn_draft_head_int2_quant`), then 16 sample rows read back and compared byte for byte with the
/// host law (`saber_kernels::fn33::quant_row`); the `fndrafthead` line.
fn draft_head_boot(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    host: Option<[Vec<u8>; 3]>,
) -> anyhow::Result<()> {
    use saber_kernels::fn33::{quant_row, NG, ROW_BYTES};
    // FN61g: `w8-head` -- no bf16 head on the card: the pack built on the host is uploaded
    if rk.head == 0 {
        let Some(h) = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .and_then(|c| c.int2.as_ref())
        else {
            ensure!(
                host.is_none(),
                "an int2 pack built on the host without an int2 head"
            );
            return Ok(());
        };
        let [c, s, z] = host.context("w8-head: the int2 pack is built on the host")?;
        ensure!(
            (c.len(), s.len(), z.len())
                == (h.wq.len() as usize, h.s.len() as usize, h.zs.len() as usize),
            "the host int2 pack's sizes differ from the card's buffers"
        );
        rk.hip.memcpy_h2d_blocking(&h.wq, &c, &rk.dev.stream)?;
        rk.hip.memcpy_h2d_blocking(&h.s, &s, &rk.dev.stream)?;
        rk.hip.memcpy_h2d_blocking(&h.zs, &z, &rk.dev.stream)?;
        rk.hip.sync_stream(&rk.dev.stream)?;
        println!(
            "fndrafthead rank{}: the int2 draft head uploaded from the HOST law ({:.1} MiB; the \
             ledger class fn.draft.int2); K {} candidates a rank; the rerank scores them from the \
             W8 head ({})",
            rk.rank,
            (c.len() + s.len() + z.len()) as f64 / 1_048_576.0,
            h.k,
            T8R_GATHER
        );
        return Ok(());
    }
    ensure!(
        host.is_none(),
        "a host int2 pack beside a resident bf16 head"
    );
    let Some((wq, s, zs, k)) = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .and_then(|c| c.int2.as_ref())
        .map(|h| {
            (
                h.wq.addr().as_u64(),
                h.s.addr().as_u64(),
                h.zs.addr().as_u64(),
                h.k,
            )
        })
    else {
        return Ok(());
    };
    let t0 = Instant::now();
    let n = (VOCAB / 2) as u32;
    em.arena.new_layer();
    em.run(
        "fn_draft_head_int2_quant",
        [u64::from(n * 5).div_ceil(256), 1, 1],
        256,
        vec![
            Arg::P(rk.head),
            Arg::P(wq),
            Arg::P(s),
            Arg::P(zs),
            Arg::I(n as i32),
        ],
        saber_model::qwen4_exp::layer_graph::Group::Head,
    )?;
    // the sample rows of the bf16 head, gathered on the card by the tail's own row gather
    let sample: Vec<u32> = (0..16u32)
        .map(|i| (i * 7_759 + 3) % n)
        .chain([n - 1])
        .collect();
    let ns = sample.len();
    let cb: Vec<u8> = sample.iter().flat_map(|x| x.to_le_bytes()).collect();
    let cdev = rk.hip.malloc(cb.len() as u64)?;
    rk.hip.memcpy_h2d_blocking(&cdev, &cb, &rk.dev.stream)?;
    let wdev = rk.hip.malloc(ns as u64 * H * 2)?;
    em.run(
        "fn_draft_wgather",
        [ns as u64, 1, 1],
        256,
        vec![
            Arg::P(rk.head),
            Arg::P(cdev.addr().as_u64()),
            Arg::P(wdev.addr().as_u64()),
            Arg::I(1),
            Arg::I(ns as i32),
            Arg::I(ns as i32),
        ],
        saber_model::qwen4_exp::layer_graph::Group::Head,
    )?;
    rk.dev.exec(&rk.table, em)?;
    rk.hip.sync_stream(&rk.dev.stream)?;
    let wall = d2h_in(&rk.dev, &wdev, wdev.addr().as_u64(), ns * H as usize * 2)?;
    let (wqp, sp, zsp) = {
        let h = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .and_then(|c| c.int2.as_ref())
            .context("int2")?;
        (&h.wq, &h.s, &h.zs)
    };
    let mut bad = 0usize;
    for (si, &row) in sample.iter().enumerate() {
        let row = u64::from(row);
        let hb = H as usize * 2;
        let w: Vec<u16> = wall[si * hb..(si + 1) * hb]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let (c, sv, zv) = quant_row(&w);
        let dc = d2h_in(&rk.dev, wqp, wq + row * ROW_BYTES as u64, ROW_BYTES)?;
        let ds = d2h_in(&rk.dev, sp, s + row * NG as u64 * 2, NG * 2)?;
        let dz = d2h_in(&rk.dev, zsp, zs + row * NG as u64 * 2, NG * 2)?;
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        bad += usize::from(dc != c || ds != le(&sv) || dz != le(&zv));
    }
    println!(
        "fndrafthead rank{}: the int2 draft head built on the card in {:.2} s ({:.1} MiB: codes {} B/row + S/ZS {} groups; the ledger class fn.draft.int2); K {k} candidates a rank; host law vs the card on {} sample rows: {} differ",
        rk.rank,
        t0.elapsed().as_secs_f64(),
        saber_kernels::fn33::pack_bytes(u64::from(n)) as f64 / 1_048_576.0,
        ROW_BYTES,
        NG,
        sample.len(),
        bad
    );
    ensure!(
        bad == 0,
        "the int2 draft head's pack differs from the host law"
    );
    drop(cdev);
    drop(wdev);
    Ok(())
}

/// FN99 (`fnembed99`): `passes` chain tails of `rows` rows each read one embedding row a row.
fn emb_count(rk: &mut Rank<'_, '_>, passes: u64, rows: u32) {
    if let Some(dc) = rk.mtp.as_mut().and_then(|m| m.chain.as_mut()) {
        dc.emb_passes += passes;
        dc.emb_rows += passes * u64::from(rows);
    }
}

/// FN22: the tail of a pass over `rows` output rows whose row selection sits at `sel_off` of the
/// chain's input buffer.
fn tail_args(rk: &Rank<'_, '_>, sel_off: u64, rows: u32) -> anyhow::Result<TailArgs> {
    let dc = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("the device chain")?;
    let inb = dc.inb.addr().as_u64();
    Ok(TailArgs {
        embh: dc.embh.addr(),
        ids: dc.ids.addr().as_u64(),
        ctr: inb,
        sel: inb + sel_off,
        rows,
        emb: rk.emb.addr().as_u64(),
        tie_high: dc.tie_high,
        couple: dc.couple.is_some(),
        i2: dc.int2.as_ref().map(|h| I2Args {
            wq: h.wq.addr().as_u64(),
            s: h.s.addr().as_u64(),
            zs: h.zs.addr().as_u64(),
            head: rk.head,
            head8: rk.head8.filter(|_| rk.head == 0),
            k: h.k,
            full_head: dc.check.is_some(),
        }),
    })
}

/// FN22: the chain passes' tables (pass j = 2..=k at `pos[i] + j - 1`, pads at 0) and their one
/// layout, and each pass's sparse flag.
#[allow(clippy::type_complexity)]
fn chain_tables(
    rk: &Rank<'_, '_>,
    pos: &[u32],
    n: usize,
    bucket: u32,
    k: u32,
) -> anyhow::Result<(Vec<(Vec<u8>, Vec<u32>)>, Vec<(&'static str, usize)>)> {
    chain_tables_at(rk, pos, n, bucket, k, true)
}

/// [`chain_tables`]; `begin` false = no KV step law per pass (FN119's acc variants, after one
/// begin over the union of their positions).
#[allow(clippy::type_complexity)]
fn chain_tables_at(
    rk: &Rank<'_, '_>,
    pos: &[u32],
    n: usize,
    bucket: u32,
    k: u32,
    begin: bool,
) -> anyhow::Result<(Vec<(Vec<u8>, Vec<u32>)>, Vec<(&'static str, usize)>)> {
    let mut out = Vec::new();
    let mut lay: Option<Vec<(&'static str, usize)>> = None;
    // FN51f: ONE band a chain (its first pass's rows + the margin): one layout for every pass
    let mut p1: Vec<u32> = pos.iter().map(|x| x + 1).collect();
    p1.resize(bucket as usize, 0);
    let band = step_band(rk, &p1);
    for j in 2..=k {
        let mut p: Vec<u32> = pos.iter().map(|x| x + j - 1).collect();
        p.resize(bucket as usize, 0);
        let (tb, offs) = decode_tables_at(rk, &p, n, band, begin)?;
        if let Some(l) = &lay {
            ensure!(*l == offs, "the chain passes' tables share one layout");
        } else {
            lay = Some(offs);
        }
        out.push((tb, p));
    }
    Ok((out, lay.unwrap_or_default()))
}

/// FN22: ONE upload of a chain's inputs: the pass counter (0), the pass-1 row selection (`sel1`,
/// -1 padded), the chain passes' (rows 0..n, pads -1), the chain passes' tables.
fn chain_inputs(
    rk: &mut Rank<'_, '_>,
    sel1: &[i32],
    n: usize,
    tabs: &[(Vec<u8>, Vec<u32>)],
    acc: Option<&[u32]>,
) -> anyhow::Result<()> {
    let t = Instant::now();
    let b = chain_input_bytes(rk, sel1, n, tabs, acc)?;
    let dc = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("chain")?;
    rk.hip.memcpy_h2d_blocking(&dc.inb, &b, &rk.dev.stream)?;
    seg(rk, "stage", t);
    Ok(())
}

/// FN22: the bytes [`chain_inputs`] uploads (FN117b's S1 stages them instead). LR-N: `acc` = the
/// round's accepted counts (a round after a verify step): under `--spec-couple on` the coupled
/// region carries the last `Cmd::Decode`'s rows at those counts; `None` (a prompt's first round)
/// leaves it zero (every row greedy).
fn chain_input_bytes(
    rk: &Rank<'_, '_>,
    sel1: &[i32],
    n: usize,
    tabs: &[(Vec<u8>, Vec<u32>)],
    acc: Option<&[u32]>,
) -> anyhow::Result<Vec<u8>> {
    let dc = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("the device chain")?;
    let slot = dc.slot;
    let mut b = vec![0u8; (IN_STAGE + tabs.len() as u64 * slot) as usize];
    if let (Some(c), Some(acc)) = (dc.couple.as_ref(), acc) {
        let r = super::couple::couple_region(&c.cur, acc, n);
        let a = super::couple::IN_COUPLE as usize;
        ensure!(
            a + r.len() <= IN_STAGE as usize,
            "the coupled region past the stage"
        );
        b[a..a + r.len()].copy_from_slice(&r);
    }
    let ids = saber_kernels::fn22::IDS_STRIDE as usize;
    ensure!(
        sel1.len() <= ids && n <= ids,
        "the chain's selections hold 8 rows"
    );
    for i in 0..ids {
        let s1 = sel1.get(i).copied().unwrap_or(-1);
        let sc = if i < n { i as i32 } else { -1 };
        let a = IN_SEL1 as usize + 4 * i;
        b[a..a + 4].copy_from_slice(&s1.to_le_bytes());
        let a = IN_SELC as usize + 4 * i;
        b[a..a + 4].copy_from_slice(&sc.to_le_bytes());
    }
    for (j, (tb, _)) in tabs.iter().enumerate() {
        ensure!(tb.len() as u64 <= slot, "a chain table past its slot");
        let a = (IN_STAGE + j as u64 * slot) as usize;
        b[a..a + tb.len()].copy_from_slice(tb);
    }
    Ok(b)
}

/// LR-N: one round's coupled rows counted (`fncouple`'s end line).
fn couple_count(rk: &mut Rank<'_, '_>, n: usize) {
    if let Some(c) = rk
        .mtp
        .as_mut()
        .and_then(|m| m.chain.as_mut())
        .and_then(|d| d.couple.as_mut())
    {
        if c.cur.len() >= n {
            let k = c.cur[..n].iter().filter(|r| r[3] != 0).count() as u64;
            c.rounds += u64::from(k > 0);
            c.rows += k;
            c.greedy += n as u64 - k;
        }
    }
}

/// FN22: the chain passes 2..=k on the card (their inputs uploaded by [`chain_inputs`], the first
/// one's embedding rows already in `rk.emb`): per pass the table slot -> the constant region (a
/// stream-ordered D2D copy), the pass with its tail, the gather of the drafter's stream. Nothing
/// waits. `row0` = the id-table row of the first chain pass (the check arm's index).
#[allow(clippy::too_many_arguments)]
fn chain_passes(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    tabs: &[(Vec<u8>, Vec<u32>)],
    offs: &[(&'static str, usize)],
    n: usize,
    bucket: u32,
    row0: u32,
) -> anyhow::Result<()> {
    if tabs.is_empty() {
        return Ok(());
    }
    let tail = tail_args(rk, IN_SELC, bucket)?;
    emb_count(rk, tabs.len() as u64, bucket);
    let (slot, graph, check) = {
        let dc = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .context("chain")?;
        (dc.slot, dc.graph, dc.check.is_some())
    };
    let conf = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .is_some_and(|c| c.conf.is_some());
    let st = rk.pools.batch(LAYERS as usize, 0);
    let hid = rk.mtp.as_ref().context("mtp")?.prev.addr().as_u64();
    let emb = rk.emb.addr().as_u64();
    let gather: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, i)).collect();
    if graph && !check && !conf {
        return chain_graph(rk, em, tabs, offs, n, bucket, tail);
    }
    for (j, (tb, p)) in tabs.iter().enumerate() {
        let t = Instant::now();
        {
            let dc = rk
                .mtp
                .as_ref()
                .and_then(|m| m.chain.as_ref())
                .context("chain")?;
            rk.hip.run_batch(&[Job::MemcpyD2d {
                dst: &rk.dev.consts,
                dst_offset: 0,
                src: &dc.inb,
                src_offset: IN_STAGE + j as u64 * slot,
                bytes: tb.len() as u64,
                stream: &rk.dev.stream,
            }])?;
        }
        seg(rk, "d2d", t);
        let cc = decode_cc(rk, offs, p, step_band(rk, &tabs[0].1));
        let lg = mtp_pass(
            rk,
            em,
            bucket,
            emb,
            hid,
            cc,
            st,
            &gather,
            true,
            None,
            Some(tail),
        )?;
        if check {
            let sel: Vec<i32> = (0..bucket as i32)
                .map(|i| if (i as usize) < n { i } else { -1 })
                .collect();
            check_pass(rk, recs, lg, bucket as usize, &sel, row0 + j as u32)?;
        }
        if conf && row0 == 1 {
            conf_pass(rk, n, false)?;
        }
    }
    Ok(())
}

/// FN22 (`--fn-mtp-chain-graph`): the chain passes as ONE captured graph per (bucket, live rows,
/// passes, each pass's sparse flag): the table copies, the passes and the gathers as nodes.
#[allow(clippy::too_many_arguments)]
fn chain_graph(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    tabs: &[(Vec<u8>, Vec<u32>)],
    offs: &[(&'static str, usize)],
    n: usize,
    bucket: u32,
    tail: TailArgs,
) -> anyhow::Result<()> {
    // FN51d: the chain's graph runs lent, as every decode-form pass ([`mtp_pass`])
    if rk.mtp.as_ref().context("mtp")?.lend {
        lend_for(rk, em, bucket, false, true)?;
    } else {
        lend_for(rk, em, bucket, false, false)?;
    }
    let t_l = Instant::now();
    let flags: Vec<bool> = tabs
        .iter()
        .map(|(_, p)| saber_kernels::fn19::short_rows(p, 1))
        .collect();
    // FN51f: the chain's one band (chain_tables' law: its first pass)
    let band = step_band(rk, &tabs[0].1);
    let key = (bucket, n as u32, tabs.len() as u32, flags, band.0);
    let slot = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("chain")?
        .slot;
    let st = rk.pools.batch(LAYERS as usize, 0);
    let hid = rk.mtp.as_ref().context("mtp")?.prev.addr().as_u64();
    let emb = rk.emb.addr().as_u64();
    let gather: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, i)).collect();
    if !rk
        .mtp
        .as_ref()
        .context("mtp")?
        .chain_graphs
        .contains_key(&key)
    {
        rk.hip.sync_stream(&rk.dev.stream)?;
        let (l0, m0) = (rk.dev.launches, rk.dev.memsets);
        rk.hip
            .begin_capture(&rk.dev.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
        // FN18: `ids` = the last pass's route log (the one-graph-a-chain ARM does not feed
        // `--fn-route-trace`; the default per-pass graphs do)
        let mut r: anyhow::Result<(u64, u64, u64)> = Ok((0, 0, 0));
        for (j, (tb, p)) in tabs.iter().enumerate() {
            let dc = rk
                .mtp
                .as_ref()
                .and_then(|m| m.chain.as_ref())
                .context("chain")?;
            if let Err(e) = rk.hip.run_batch(&[Job::MemcpyD2d {
                dst: &rk.dev.consts,
                dst_offset: 0,
                src: &dc.inb,
                src_offset: IN_STAGE + j as u64 * slot,
                bytes: tb.len() as u64,
                stream: &rk.dev.stream,
            }]) {
                r = Err(e.into());
                break;
            }
            let cc = decode_cc(rk, offs, p, band);
            match mtp_emit(rk, em, bucket, emb, hid, cc, st, true, Some(tail)) {
                Ok((lg, strm, _, ids)) => {
                    if let Err(e) = mtp_gather(rk, strm, &gather) {
                        r = Err(e);
                        break;
                    }
                    r = Ok((lg, strm, ids));
                }
                Err(e) => {
                    r = Err(e);
                    break;
                }
            }
        }
        let g = rk.hip.end_capture(&rk.dev.stream);
        let (lg, strm, ids) = r?;
        let g = g?;
        let nodes = rk.hip.graph_node_count(&g)?;
        let exec = rk.hip.instantiate(&g)?;
        if rk.rank == 0 {
            println!(
                "fnchaingraph: captured the chain b{bucket} n{n} passes {} ({} launches + {} memsets + {} copies = {nodes} nodes)",
                tabs.len(),
                rk.dev.launches - l0,
                rk.dev.memsets - m0,
                2 * tabs.len()
            );
        }
        rk.mtp.as_mut().context("mtp")?.chain_graphs.insert(
            key.clone(),
            MtpGraph {
                exec,
                _graph: g,
                lg,
                strm,
                ids,
            },
        );
    }
    let ev = (
        rk.hip.create_event(EventFlags::Default)?,
        rk.hip.create_event(EventFlags::Default)?,
    );
    rk.hip.record_event(&ev.0, &rk.dev.stream)?;
    {
        let gph = &rk.mtp.as_ref().context("mtp")?.chain_graphs[&key];
        rk.hip.graph_launch(&gph.exec, &rk.dev.stream)?;
    }
    rk.hip.record_event(&ev.1, &rk.dev.stream)?;
    pass_done(rk, ev, t_l, false)
}

/// FN22: the end of a device chain: ONE D2H of the id table's first `rows` rows (the stream's one
/// wait), the passes' device times read from their events.
fn finish_chain(rk: &mut Rank<'_, '_>, rows: u32) -> anyhow::Result<Vec<u32>> {
    let t = Instant::now();
    let stride = saber_kernels::fn22::IDS_STRIDE;
    let raw = {
        let dc = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .context("chain")?;
        d2h_in(
            &rk.dev,
            &dc.ids,
            dc.ids.addr().as_u64(),
            (rows * stride * 4) as usize,
        )?
    };
    seg(rk, "ids", t);
    let evs = std::mem::take(&mut rk.mtp.as_mut().context("mtp")?.evs);
    for (a, b) in &evs {
        let ms = f64::from(rk.hip.elapsed_ms(a, b)?);
        rk.mtp.as_mut().context("mtp")?.pass_ms.push(ms);
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// FN33's check arm: for an int2 pass (its candidates recorded, the full head also run), per output
/// row: the position of the rank-LOCAL bf16 argmax among the rank's int2 candidates (-2 = absent,
/// -1 = a pad row), the candidates' exact scores vs the full head's logits (bits differing), and
/// the rank-local rerank winner (the candidates' argmax by FN22's law).
#[allow(clippy::type_complexity)]
fn check_int2(
    rk: &mut Rank<'_, '_>,
    lg: u64,
    sel: &[i32],
) -> anyhow::Result<Option<(Vec<i64>, u64, Vec<i64>)>> {
    let Some(l) = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .and_then(|c| c.i2last)
    else {
        return Ok(None);
    };
    let half = VOCAB as usize / 2;
    let (r, k, ldc) = (l.rows as usize, l.k as usize, l.ldc as usize);
    let cb = d2h_in(&rk.dev, &rk.dev.arena, l.cand, r * k * 4)?;
    let cand: Vec<usize> = cb
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let sb = d2h_in(&rk.dev, &rk.dev.arena, l.sc, r * ldc * 2)?;
    let (mut lpos, mut lwin, mut scd) = (Vec::new(), Vec::new(), 0u64);
    for (i, &s) in sel.iter().enumerate() {
        if s < 0 {
            lpos.push(-1);
            lwin.push(-1);
            continue;
        }
        let row = d2h_in(
            &rk.dev,
            &rk.dev.arena,
            lg + s as u64 * half as u64 * 2,
            half * 2,
        )?;
        let (am, _) = row_argmax(&row);
        let cs = &cand[i * k..(i + 1) * k];
        lpos.push(cs.iter().position(|&c| c == am).map_or(-2, |p| p as i64));
        let mut best: Option<(u16, usize)> = None;
        for (j, &c) in cs.iter().enumerate() {
            let o = (i * ldc + i * k + j) * 2;
            let got = u16::from_le_bytes([sb[o], sb[o + 1]]);
            let full = u16::from_le_bytes([row[2 * c], row[2 * c + 1]]);
            scd += u64::from(got != full);
            let key = saber_kernels::fn22::key(got);
            if best.is_none_or(|(bk, bc)| key > bk || (key == bk && c < bc)) {
                best = Some((key, c));
            }
        }
        lwin.push(best.map_or(-1, |b| b.1 as i64));
    }
    Ok(Some((lpos, scd, lwin)))
}

/// FN22's check arm: the pass just enqueued (logits `lg`, `lrows` rows, output row r = logits row
/// `sel[r]`, -1 = pad) vs the HOST chain's law on the same logits -- the ids (row-for-row) and the
/// embedding rows the tail wrote (sha256 vs the checkpoint-file gather of the host's ids).
fn check_pass(
    rk: &mut Rank<'_, '_>,
    recs: &PackRecs,
    lg: u64,
    lrows: usize,
    sel: &[i32],
    row: u32,
) -> anyhow::Result<()> {
    rk.hip.sync_stream(&rk.dev.stream)?;
    let all = pick_argmax(rk, lg, lrows)?;
    let host: Vec<u32> = sel
        .iter()
        .map(|&s| if s < 0 { 0 } else { all[s as usize] })
        .collect();
    let stride = saber_kernels::fn22::IDS_STRIDE as usize;
    let (dev, emb) = {
        let dc = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .context("chain")?;
        let d = d2h_in(
            &rk.dev,
            &dc.ids,
            dc.ids.addr().as_u64() + u64::from(row) * stride as u64 * 4,
            stride * 4,
        )?;
        let e = d2h_in(
            &rk.dev,
            &rk.emb.own,
            rk.emb.addr().as_u64(),
            sel.len() * H as usize * 2,
        )?;
        (d, e)
    };
    let dev: Vec<u32> = dev
        .chunks_exact(4)
        .take(sel.len())
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want = embed_rows(recs, &host)?;
    let i2 = check_int2(rk, lg, sel)?;
    let (ds, hs) = (
        saber_kernels::manifest::sha256_hex(&emb),
        saber_kernels::manifest::sha256_hex(&want),
    );
    let ck = rk
        .mtp
        .as_mut()
        .and_then(|m| m.chain.as_mut())
        .and_then(|c| c.check.as_mut())
        .context("the check arm")?;
    ck.passes += 1;
    ck.rows += sel.len() as u64;
    ck.id_diff += dev.iter().zip(&host).filter(|(a, b)| a != b).count() as u64;
    ck.emb_diff += u64::from(ds != hs);
    if let Some((_, scd, _)) = &i2 {
        ck.sc_diff += *scd;
    }
    let (lpos, scd, lwin) = i2.unwrap_or_default();
    ck.outside += lpos.iter().filter(|&&p| p < -1).count() as u64;
    ck.lines.push(
        json!({"row": row, "rows": sel.len(), "lrows": lrows, "sel": sel, "dev": dev, "host": host,
            "dev_sha": ds, "host_sha": hs, "lpos": lpos, "sc_diff": scd, "lwin": lwin,
            "rank": rk.rank})
        .to_string(),
    );
    Ok(())
}

/// FN22: the chain from host-known first drafts (after a prefill's first pass): pass 2's embedding
/// rows uploaded as the host chain does, then passes 2..=k on the card, one D2H at the end.
fn chain_from_tokens(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    drafts: &mut [Vec<u32>],
    pos: &[u32],
    bucket: u32,
    k: u32,
) -> anyhow::Result<()> {
    if k < 2 {
        return Ok(());
    }
    let n = drafts.len();
    let t = Instant::now();
    let mut toks: Vec<u32> = drafts.iter().map(|d| *d.last().unwrap_or(&0)).collect();
    toks.resize(bucket as usize, 0);
    let e = embed_rows(recs, &toks)?;
    buf_h2d(rk, BufK::Emb, &e)?;
    seg(rk, "emb", t);
    let (tabs, offs) = chain_tables(rk, pos, n, bucket, k)?;
    chain_inputs(rk, &[], n, &tabs, None)?;
    chain_passes(rk, em, recs, &tabs, &offs, n, bucket, 0)?;
    let ids = finish_chain(rk, k - 1)?;
    let stride = saber_kernels::fn22::IDS_STRIDE as usize;
    for (i, d) in drafts.iter_mut().enumerate() {
        for j in 0..(k - 1) as usize {
            d.push(ids[j * stride + i]);
        }
    }
    Ok(())
}

/// FN16 / FN22: the drafter after a verify step's commit, both chains: pass 1 over the verify rows
/// (`toks[s * w + t]` = the token after row t: the accepted drafts, the committed last token, the
/// rejected rows' argmax; positions `pos0[s] + t`), draft 1 = row `s * w + acc[s]`'s argmax, then
/// passes 2..=kn from position `pos0[s] + acc[s]`. Returns `kn` drafts per live sequence. The HOST
/// chain is FN16's code; the DEVICE chain runs pass 1 with its tail, the chain passes, one D2H.
/// FN30c: a verify step's held tier copies are released when the round is done (its drafts read
/// back): the copy stream is fenced behind the whole round, so no promotion overlaps it.
#[allow(clippy::too_many_arguments)]
pub(super) fn mtp_after_verify(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    toks: &[u32],
    pos0: &[u32],
    w: u32,
    acc: &[u32],
    bucket: u32,
    kn: u32,
    pre: Option<PreHook<'_>>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let dr = drafter_round(
        rk, em, recs, toks, pos0, w, acc, bucket, kn, pre, None, false,
    )?;
    let t = Instant::now();
    rk.tier
        .mover
        .release(&rk.dev.stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rd(rk, "rd_release", t);
    Ok(dr)
}

/// FN58c: one part of the drafter round's host time (census key `rd_*`, ms summed per round).
fn rd(rk: &mut Rank<'_, '_>, k: &'static str, t: Instant) {
    *rk.rd_t.entry(k).or_insert(0.0) += t.elapsed().as_secs_f64() * 1e3;
}

#[allow(clippy::too_many_arguments)]
fn drafter_round(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    toks: &[u32],
    pos0: &[u32],
    w: u32,
    acc: &[u32],
    bucket: u32,
    kn: u32,
    mut pre: Option<PreHook<'_>>,
    mid: Option<MidHook<'_>>,
    defer: bool,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let n = acc.len();
    let wu = w as usize;
    // FN58b: the committed tokens' PLE rows, gathered while the round's passes run on the card
    let mut run_pre = |rk: &mut Rank<'_, '_>| -> anyhow::Result<()> {
        if let (Some(h), Some(ov)) = (pre.as_mut(), rk.ple_ov.as_mut()) {
            let t = Instant::now();
            let (ms, r, hits, nvme) = h()?;
            *rk.rd_t.entry("rd_pre").or_insert(0.0) += t.elapsed().as_secs_f64() * 1e3;
            ov.pre.0 += ms;
            ov.pre.1 += r;
            ov.pre.2 += hits;
            ov.pre.3 += nvme;
        }
        Ok(())
    };
    if rk.diag.gap_us > 0 {
        // FN30b's idle arm: the device drained, then idle for gap_us before the drafter round
        rk.hip.sync_stream(&rk.dev.stream)?;
        std::thread::sleep(Duration::from_micros(rk.diag.gap_us));
    }
    rk.diag.bucket = bucket;
    fn30b_probe(rk, 1)?;
    // FN117b (S1): the three inputs below wait in the pinned staging and cross as ONE async batch
    // just before pass 1 (no drain, no blocking copy); only the device chain's round
    let s1 = hand_s1_on(rk);
    let t = Instant::now();
    let e = embed_rows(recs, toks)?;
    rd(rk, "rd_emb_rows", t);
    if s1 {
        // the FN100 tails reclaimed first, as `buf_h2d` does before its copy
        let rows = (e.len() as u64).div_ceil(H * 2);
        buf_need(rk, u32::try_from(rows)?, "the embedding rows")?;
    } else {
        // FN58c: the step's stream drained first (the work still queued when the round starts),
        // then the embeddings' copy -- the blocking copy's two halves
        let t1 = Instant::now();
        rk.hip.sync_stream(&rk.dev.stream)?;
        rd(rk, "rd_emb_drain", t1);
        let t1 = Instant::now();
        buf_h2d(rk, BufK::Emb, &e)?;
        rd(rk, "rd_emb_h2d", t1);
    }
    seg(rk, "emb1", t);
    let t = Instant::now();
    let mut pp = pos0.to_vec();
    pp.resize(bucket as usize, 0);
    let one = vec![1u32; bucket as usize];
    let zero = vec![0u32; bucket as usize];
    let band = step_band(rk, &pp);
    let (mut tb, mut offs) = band_tables(rk, &pp, w, &one, &zero, rk.pools.pool_w, n, band);
    kv_extend(rk, &mut tb, &mut offs, &pp, w, n, band.1)?;
    if !s1 {
        rk.hip
            .memcpy_h2d_at_blocking(&rk.dev.consts, 0, &tb, &rk.dev.stream)?;
    }
    let mut cc = ChunkConsts::at(
        rk.dev.consts.addr().as_u64(),
        &offs,
        bucket * w,
        0,
        2,
        rk.pools.nbs,
        rk.dev.cs.addr().as_u64(),
    );
    cc.max_ctx = band.0;
    cc.bt_nb = band.1;
    cc.bar = rk.bar.addr().as_u64();
    cc.b = bucket;
    cc.w = w;
    cc.qsa_short = saber_kernels::fn19::short_rows(&pp, w);
    seg(rk, "consts1", t);
    rd(rk, "rd_consts", t);
    let t = Instant::now();
    let sv = rk.pools.batch(LAYERS as usize, 0);
    let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
    let gather: Vec<(u32, u32)> = (0..n as u32)
        .map(|i| (i * w + acc[i as usize], i))
        .collect();
    let emb = rk.emb.addr().as_u64();
    let dpos: Vec<u32> = (0..n).map(|i| pos0[i] + acc[i]).collect();
    // LR-N: a round after a verify step carries the coupled rows (a prompt's first round, with its
    // END hook, does not: its drafts stay greedy)
    let cacc = mid.is_none().then_some(acc);
    if rk.mtp.as_ref().is_some_and(|m| m.chain.is_some()) {
        let sel1: Vec<i32> = (0..bucket as usize)
            .map(|i| {
                if i < n {
                    (i as u32 * w + acc[i]) as i32
                } else {
                    -1
                }
            })
            .collect();
        let (tabs, toffs) = chain_tables(rk, &dpos, n, bucket, kn)?;
        if s1 {
            // FN117b (S1): the embedding rows, the pass-1 tables and the chain's inputs in ONE
            // async batch on the step's stream (every KV step-law copy above writes other memory)
            let inb = chain_input_bytes(rk, &sel1, n, &tabs, cacc)?;
            hand_s1_upload(rk, &e, &tb, &inb)?;
        } else {
            chain_inputs(rk, &sel1, n, &tabs, cacc)?;
        }
        if cacc.is_some() {
            couple_count(rk, n);
        }
        let tail = tail_args(rk, IN_SEL1, bucket)?;
        emb_count(rk, 1, bucket);
        let lg = mtp_pass(
            rk,
            em,
            bucket * w,
            emb,
            hid,
            cc,
            sv,
            &gather,
            true,
            None,
            Some(tail),
        )?;
        if rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .is_some_and(|c| c.check.is_some())
        {
            check_pass(rk, recs, lg, (bucket * w) as usize, &sel1, 0)?;
        }
        if rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .is_some_and(|c| c.conf.is_some())
        {
            conf_pass(rk, n, true)?;
        }
        // FN116: the first round's END-node drafter part, between pass 1 and the chain
        if let Some(h) = mid {
            h(rk)?;
        }
        chain_passes(rk, em, recs, &tabs, &toffs, n, bucket, 1)?;
        rd(rk, "rd_issue", t);
        // FN56 (`defer`): the verify step's tier end while the chain runs on the card
        let t = Instant::now();
        tier_flush(rk)?;
        rd(rk, "rd_flush", t);
        // FN117b (S2): the next step's expert-pointer tables while the chain runs on the card
        let t = Instant::now();
        hand_s2_build(rk)?;
        rd(rk, "rd_s2", t);
        run_pre(rk)?;
        fn30b_probe(rk, 2)?;
        if defer {
            // FN119e: the ids read back behind the round; the next step reads them
            queue_ids(rk, kn)?;
            return Ok(Vec::new());
        }
        let t = Instant::now();
        let ids = finish_chain(rk, kn)?;
        rd(rk, "rd_ids", t);
        fn30b_resolve(rk)?;
        let pm = rk.mtp.as_ref().context("mtp")?.pass_ms.clone();
        rk.diag.passes.push((bucket, pm));
        let stride = saber_kernels::fn22::IDS_STRIDE as usize;
        return Ok((0..n)
            .map(|i| (0..kn as usize).map(|j| ids[j * stride + i]).collect())
            .collect());
    }
    ensure!(!s1, "FN117b: S1 stages the device chain's round only");
    let lg = mtp_pass(
        rk,
        em,
        bucket * w,
        emb,
        hid,
        cc,
        sv,
        &gather,
        true,
        None,
        None,
    )?;
    tier_flush(rk)?;
    run_pre(rk)?;
    let o1 = draft_ids(rk, lg, (bucket * w) as usize)?;
    let mut dr: Vec<Vec<u32>> = (0..n).map(|i| vec![o1[i * wu + acc[i] as usize]]).collect();
    mtp_chain_k(rk, em, recs, &mut dr, &dpos, bucket, kn)?;
    Ok(dr)
}

/// FN22's tie-law gate (`--fn-mtp-chain-selftest`, both ranks at boot): the device tail on
/// CONSTRUCTED logits -- 8 output rows over 12 logits rows, out of order: random rows, a unique
/// maximum on rank 1, a three-way tie inside rank 0, a tie across the pair, a flat row, -0 vs +0,
/// a -inf row with one finite entry on rank 1, a pad row -- vs the host chain's law
/// (`row_argmax` + `merge_argmax`, each rank building both halves itself) and the checkpoint-file
/// gather; then the tie-high CONTROL, which must differ exactly where a row's maximum is tied.
fn chain_selftest(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
) -> anyhow::Result<()> {
    let half = VOCAB as usize / 2;
    let lrows = 12usize;
    let sel: [i32; 8] = [9, 2, 4, 0, 11, 6, 7, -1];
    // rank r's half of logits row q
    let mk = |r: usize, q: usize| -> Vec<u16> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ ((q as u64) << 8) ^ r as u64;
        let mut v: Vec<u16> = (0..half)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // finite, |x| < 8: exponent <= 129
                let x = (s >> 40) as u16;
                (x & 0x807f) | ((0x70 + ((x >> 7) & 0x11)) << 7)
            })
            .collect();
        let big = 0x4480u16;
        match q {
            2 if r == 1 => v[77_777] = big,
            4 if r == 0 => {
                v[5_000] = big;
                v[70_000] = big;
                v[half - 1] = big;
            }
            0 => v[if r == 0 { 100_000 } else { 3 }] = big,
            11 => v.iter_mut().for_each(|x| *x = 0x3f80),
            6 => {
                v.iter_mut().for_each(|x| *x = 0x8000);
                if r == 1 {
                    v[7] = 0x0000;
                }
            }
            7 => {
                v.iter_mut().for_each(|x| *x = 0xff80);
                if r == 1 {
                    v[half - 1] = 0xff7f;
                }
            }
            _ => {}
        }
        v
    };
    let rank = rk.rank as usize;
    let mut bytes = Vec::with_capacity(lrows * half * 2);
    for q in 0..lrows {
        for x in mk(rank, q) {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    let lgb = rk.hip.malloc(bytes.len() as u64)?;
    rk.hip.memcpy_h2d_blocking(&lgb, &bytes, &rk.dev.stream)?;
    // the host law, both halves built here
    let row_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut want = Vec::new();
    let mut tied = Vec::new();
    for &q in &sel {
        if q < 0 {
            want.push(0u32);
            tied.push(false);
            continue;
        }
        let (h0, h1) = (mk(0, q as usize), mk(1, q as usize));
        let (i0, v0) = row_argmax(&row_bytes(&h0));
        let (i1, v1) = row_argmax(&row_bytes(&h1));
        want.push(merge_argmax((i0 as u32, v0), ((i1 + half) as u32, v1)));
        let kmax = h0
            .iter()
            .chain(&h1)
            .map(|&b| saber_kernels::fn22::key(b))
            .max()
            .unwrap_or(0);
        tied.push(
            h0.iter()
                .chain(&h1)
                .filter(|&&b| saber_kernels::fn22::key(b) == kmax)
                .count()
                > 1,
        );
    }
    let wemb = embed_rows(recs, &want)?;
    let mut sb = vec![0u8; 64];
    for (i, s) in sel.iter().enumerate() {
        sb[16 + 4 * i..20 + 4 * i].copy_from_slice(&s.to_le_bytes());
    }
    let mut got = Vec::new();
    for tie_high in [false, true] {
        let dc = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .context("chain")?;
        rk.hip.memcpy_h2d_blocking(&dc.inb, &sb, &rk.dev.stream)?;
        let mut t = tail_args(rk, IN_SEL1, 8)?;
        t.tie_high = tie_high;
        t.i2 = None;
        em.arena.new_layer();
        emit_tail(em, &mut rk.pair, rk.rank, lgb.addr().as_u64(), 0, 8, t)?;
        rk.dev.exec(&rk.table, em)?;
        let ids = finish_chain(rk, 1)?;
        let e = d2h_in(
            &rk.dev,
            &rk.emb.own,
            rk.emb.addr().as_u64(),
            8 * H as usize * 2,
        )?;
        got.push((ids[..8].to_vec(), e));
    }
    let (ids, e) = &got[0];
    let ok_ids = ids.iter().zip(&want).filter(|(a, b)| a == b).count();
    let ok_emb = (0..8)
        .filter(|&r| {
            let n = H as usize * 2;
            e[r * n..(r + 1) * n] == wemb[r * n..(r + 1) * n]
        })
        .count();
    let (cids, _) = &got[1];
    let differ: Vec<usize> = (0..8).filter(|&r| cids[r] != want[r]).collect();
    let tie_rows: Vec<usize> = (0..8).filter(|&r| tied[r]).collect();
    println!(
        "fnchaintest rank{}: device law vs host law: ids {ok_ids}/8, embedding rows {ok_emb}/8 byte-identical \
         (ids {:?}); tie-high CONTROL differs on rows {differ:?}, the tied rows are {tie_rows:?}",
        rk.rank, ids
    );
    ensure!(
        ok_ids == 8 && ok_emb == 8 && differ == tie_rows && !differ.is_empty(),
        "FN22's tie-law gate FAILED"
    );
    rk.hip.sync_stream(&rk.dev.stream)?;
    drop(lgb);
    Ok(())
}

/// FN22: the drafter rounds' host segments (per segment: total ms, count, us each) and the check
/// arm's counters (its records -> `chaincheck_rank<r>.jsonl`); the `fnmtpseg` / `fnchaincheck` lines.
/// FN30: read the fused MoE train's `sync` region back after the stream drains: `(words, nonzero)`
/// -- every counter is back at zero after every `moe_back_*` launch (FN26's law), so a nonzero word
/// is a protocol fault. Prints `fnmoesync rank<r>:` with the which-path counter.
fn moe_sync_check(rk: &Rank<'_, '_>, em: &Emitter<'_>) -> anyhow::Result<(u64, u64)> {
    // FN106d: the pinned page census at the run's end (the huge share after the run's pressure)
    println!(
        "fnpinned106d rank{}: end | {}",
        rk.rank,
        saber_hip::pinned::stats().line(rk.hip.pinned_pages())
    );
    let mut b = vec![0u8; rk.moe_sync.len() as usize];
    rk.hip
        .memcpy_d2h_blocking(&mut b, &rk.moe_sync, &rk.dev.stream)?;
    let words = (b.len() / 4) as u64;
    let nz = b
        .chunks_exact(4)
        .filter(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) != 0)
        .count() as u64;
    println!(
        "fnmoesync rank{}: train {}; {} target MoE blocks emitted fused ({} with the front half), {} as the seven; sync {} words read back, {} nonzero{}",
        rk.rank,
        em.moe_train,
        em.moe_blocks[0],
        em.moe_front_blocks,
        em.moe_blocks[1],
        words,
        nz,
        if nz == 0 { "" } else { " -- PROTOCOL FAULT" }
    );
    // FN84's which-path counter: fused back halves by work order
    println!(
        "fnmissorder rank{}: order {}; back halves id {} first {} capped {} cappre {}; RAM range [{:#x}, {:#x})",
        rk.rank,
        em.miss_order.name(),
        em.miss_launches[0],
        em.miss_launches[1],
        em.miss_launches[2],
        em.miss_launches[3],
        em.miss_ram.0,
        em.miss_ram.1
    );
    // FN86b's which-path counter: per family, the sites run fused / as the shipped chain (each
    // fused site removes one launch; ATTN-OPT-3's f6 two)
    println!(
        "fnfuse86 rank{}: arm {}; {}",
        rk.rank,
        saber_kernels::flashnext::fuse86::names(em.fuse86),
        saber_kernels::flashnext::fuse86::NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| saber_kernels::flashnext::fuse86::BUILT & (1 << i) != 0)
            .map(|(i, n)| format!(
                "{n} fused {} chain {}",
                em.fuse86_n[i][0], em.fuse86_n[i][1]
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // FN120e's which-path counter: emitted glue runs that passed the region and fell back to the
    // per-layer fills (a held run's dry re-emission counts again)
    if let Some(g) = em.glue.as_ref() {
        println!(
            "fnglue120 rank{}: region zero {} + ones {} B; spilled runs {}, spilled seeds {}{}",
            rk.rank,
            g.zero.cap,
            g.ones.cap,
            g.spills,
            g.spilled_seeds,
            if g.spills == 0 {
                ""
            } else {
                " -- a form the census missed (FN120e)"
            }
        );
    }
    // FN120c's which-path counter + the sites' sticky words (a refusal = a launch ran no body)
    if let Some(d) = em.df.as_ref() {
        let [refused, timeouts, runs] = df_sticky(rk, em)?;
        println!(
            "fndf120 rank{}: form gdn{}; GDN tails folded {} kept {}; {} sites; launches begun {runs}, refusals {refused}, dependency timeouts {timeouts}{}",
            rk.rank,
            if d.nodep { "-nodep (the CONTROL)" } else { "" },
            d.sites[0],
            d.sites[1],
            d.used(),
            if refused + timeouts == 0 { "" } else { " -- PROTOCOL FAULT" }
        );
    }
    // FN128's which-path counter: per selection kernel, the decode-form emissions on the added
    // flashnext_qsa128 symbol / on the parent
    println!(
        "fnsel128 rank{}: arm {}; score added {} parent {}; topk added {} parent {}; union added {} parent {}",
        rk.rank,
        saber_kernels::flashnext::sel128::names(em.rules.hc.sel128),
        em.sel128_n[0][0],
        em.sel128_n[0][1],
        em.sel128_n[1][0],
        em.sel128_n[1][1],
        em.sel128_n[2][0],
        em.sel128_n[2][1]
    );
    // FN111's which-path counter: per re-tile, the launches its fn111_ symbol ran / the sites that
    // kept the shipped cell
    println!(
        "fntile111 rank{}: arm {}; {}",
        rk.rank,
        saber_kernels::flashnext::tiles111::names(em.tiles111),
        saber_kernels::flashnext::tiles111::NAMES
            .iter()
            .zip(saber_kernels::flashnext::tiles111::SYMS)
            .enumerate()
            .map(|(i, (n, sym))| format!(
                "{n} ({sym}) re-tiled {} shipped {}",
                em.tiles111_n[i][0], em.tiles111_n[i][1]
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // FN141's which-path counter: per family, launches on an fn141_ symbol / on the parent
    println!(
        "fnk141 rank{}: arm {}; up {} / {}, t8 {} / {}, router {} / {} (fn141 / shipped)",
        rk.rank,
        saber_kernels::flashnext::k141::names(em.rules.hc.k141),
        em.k141_n[0][0],
        em.k141_n[0][1],
        em.k141_n[1][0],
        em.k141_n[1][1],
        em.k141_n[2][0],
        em.k141_n[2][1]
    );
    // FN163's which-path counter: per band, launches on an fn163_ symbol / on the shipped one
    println!(
        "fnroute163 rank{}: arm {}; b8 route {} / {}, b1 front {} / {} (fn163 / shipped)",
        rk.rank,
        saber_kernels::flashnext::route163::names(em.rules.hc.route163),
        em.route163_n[0][0],
        em.route163_n[0][1],
        em.route163_n[1][0],
        em.route163_n[1][1]
    );
    // FN131's which-path counter: fp8-mixer decode downs on a flashnext_hc131 symbol / shipped
    println!(
        "fnhcd131 rank{}: arm {}; decode downs on a flashnext_hc131 symbol {}, on the shipped down {}",
        rk.rank,
        saber_kernels::flashnext::down131::names(em.rules.hc.down131),
        em.down131_n[0],
        em.down131_n[1]
    );
    // FN122b's which-path counter: fp8-mixer prefill downs on a flashnext_hc8p tile / shipped cfg
    println!(
        "fnpfx122 rank{}: arm {}; hc prefill downs on a flashnext_hc8p tile {}, on the shipped {} {}; dq downs {} ups {}",
        rk.rank,
        saber_kernels::flashnext::prefill122::names(em.rules.hc.prefill122),
        em.pf122_n[0],
        saber_kernels::flashnext::prefill122::HC_SHIPPED_CFG,
        em.pf122_n[1],
        em.pf122_dq[0],
        em.pf122_dq[1]
    );
    // FN124's which-path counter + the in-situ twin's differing words
    {
        let mut differ = None;
        if let Some(d) = rk.qsa124.as_ref() {
            let mut b = [0u8; 4];
            rk.hip.sync_stream(&rk.dev.stream)?;
            rk.hip
                .memcpy_d2h_addr_blocking(&mut b, d.addr(), &rk.dev.stream)?;
            differ = Some(u32::from_le_bytes(b));
        }
        println!(
            "fnqsa124 rank{}: fp8 QSA prefill gathers on the qsa124 body {}, on the shipped symbol {}; in-situ twins {}{}",
            rk.rank,
            em.pf124_n[0],
            em.pf124_n[1],
            em.pf124_twins,
            match differ {
                Some(n) => format!(", words differing {n}"),
                None => String::new(),
            }
        );
    }
    // FN85's which-path counter: block-output reduces inside a consumer's prologue / standalone
    println!(
        "fnarpro rank{}: arm {}; reduces in a consumer's prologue {}, left standalone {}",
        rk.rank,
        if em.ar_prologue { "norm" } else { "off" },
        em.ar_rp[0],
        em.ar_rp[1]
    );
    // FN114b's which-path counter: standalone exact reduces emitted on the flag-in-data form /
    // on the one-shot
    println!(
        "fnar114b rank{}: form {}; exact reduces emitted flag-in-data {}, one-shot {}; \
         FN114d global-parity (exact + wht6) {}",
        rk.rank,
        if rk.pair.ll {
            "ll"
        } else if rk.pair.gp {
            "gp"
        } else {
            "oneshot"
        },
        rk.pair.calls_ll,
        if rk.pair.gp {
            0
        } else {
            rk.pair.calls_exact - rk.pair.calls_ll
        },
        rk.pair.calls_gp
    );
    // FN137's which-path counter + the channel's words (calls, last call, spin timeouts)
    if rk.pair.far {
        let mut w = [0u32; 6];
        if let Some((_, _, seq, _)) = rk.dev.ar.as_ref() {
            let mut b = vec![0u8; seq.len() as usize];
            rk.hip.memcpy_d2h_blocking(&mut b, seq, &rk.dev.stream)?;
            for (j, x) in w.iter_mut().enumerate() {
                let o = (super::dev::FAR_CTR0 as usize + j) * 4;
                if o + 4 <= b.len() {
                    *x = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                }
            }
        }
        println!(
            "fnar137 rank{}: arm on; MoE-out reduces fused into the back half {} (emitted {}), stood down {}; \
             channel calls {} (last {}), spin timeouts {}{}",
            rk.rank,
            rk.pair.calls_far,
            em.ar_far,
            rk.pair.far_stood_down,
            w[0],
            w[1],
            w[2],
            if w[2] == 0 { "" } else { " -- PROTOCOL FAULT" }
        );
    }
    // FN164's which-path counter + the channel's words (the 40 workgroups' call counters, timeouts)
    if rk.pair.mix != super::dev::MixPush::Off {
        let (mut lo, mut hi, mut to) = (u32::MAX, 0u32, 0u32);
        if let Some((_, _, seq, _)) = rk.dev.ar.as_ref() {
            let mut b = vec![0u8; seq.len() as usize];
            rk.hip.memcpy_d2h_blocking(&mut b, seq, &rk.dev.stream)?;
            let word = |j: u64| -> u32 {
                let o = (j * 4) as usize;
                if o + 4 <= b.len() {
                    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
                } else {
                    0
                }
            };
            for j in 0..super::dev::MIX_WGS {
                let v = word(super::dev::MIX_CTR0 + j);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            to = word(super::dev::MIX_DIAG0);
        }
        println!(
            "fnar164 rank{}: arm {:?}; mixer-out reduces fused into the out / o projection {} (emitted {}), \
             stood down {}; workgroup calls {}..{}, spin timeouts {}{}",
            rk.rank,
            rk.pair.mix,
            rk.pair.calls_mix,
            em.ar_mix,
            rk.pair.mix_stood_down,
            if lo == u32::MAX { 0 } else { lo },
            hi,
            to,
            if to == 0 && (lo == hi || lo == u32::MAX) {
                ""
            } else {
                " -- PROTOCOL FAULT"
            }
        );
    }
    Ok((words, nz))
}

fn chain_report(rk: &mut Rank<'_, '_>, out: &Path) -> anyhow::Result<Value> {
    let rank = rk.rank;
    let Some(mt) = rk.mtp.as_mut() else {
        return Ok(Value::Null);
    };
    let seg: serde_json::Map<String, Value> = mt
        .seg
        .iter()
        .map(|(k, (ms, n))| {
            (
                (*k).to_string(),
                json!({"ms": ms, "n": n, "us_each": 1e3 * ms / (*n).max(1) as f64}),
            )
        })
        .collect();
    println!(
        "fnmtpseg rank{rank} ({} chain): {}",
        if mt.chain.is_some() { "device" } else { "host" },
        mt.seg
            .iter()
            .map(|(k, (ms, n))| format!("{k} {:.1} us x {n}", 1e3 * ms / (*n).max(1) as f64))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(dc) = mt.chain.as_ref() {
        let rb = H * 2;
        println!(
            "fnembed99 rank{rank}: placement {} -- {} B; {} chain tails read {} embedding rows \
             ({:.1} rows a tail, {:.3} MiB{})",
            dc.embh.placement(),
            dc.embh.len(),
            dc.emb_passes,
            dc.emb_rows,
            dc.emb_rows as f64 / dc.emb_passes.max(1) as f64,
            (dc.emb_rows * rb) as f64 / f64::from(1u32 << 20),
            if dc.embh.placement() == "ram" {
                " over the link, read in place"
            } else {
                " from VRAM"
            }
        );
    }
    if let Some(c) = mt.chain.as_ref().and_then(|d| d.couple.as_ref()) {
        println!(
            "fncouple rank{rank}: coupled drafting ON -- {} rounds carried coupled rows: {} rows \
             coupled, {} rows greedy (FN22's argmax)",
            c.rounds, c.rows, c.greedy
        );
    }
    let mut chk = Value::Null;
    if let Some(ck) = mt.chain.as_mut().and_then(|c| c.check.as_mut()) {
        println!(
            "fnchaincheck rank{rank}: {} passes, {} rows: ids differ {}, embedding sha256 differ {} (vs the host chain's law on the same logits); FN33 int2: candidate scores differing from the full head {}, rank-local argmax outside the candidates {}",
            ck.passes, ck.rows, ck.id_diff, ck.emb_diff, ck.sc_diff, ck.outside
        );
        std::fs::create_dir_all(out)?;
        std::fs::write(
            out.join(format!("chaincheck_rank{rank}.jsonl")),
            ck.lines.join("\n") + "\n",
        )?;
        chk = json!({"passes": ck.passes, "rows": ck.rows, "id_diff": ck.id_diff, "emb_diff": ck.emb_diff,
            "sc_diff": ck.sc_diff, "outside": ck.outside});
    }
    Ok(json!({"device": mt.chain.is_some(), "seg": seg, "check": chk}))
}

/// ONE decode step at SPEC 0 for the live sequences `lives` (pool slots `0..n` in order), padded
/// to `bucket` rows (pad row = pool slot `i`, token 0, position 0: its writes land in a FREE
/// slot, zeroed when a sequence is admitted there). Every host input is uploaded BEFORE the step
/// (embedding rows, the step's tables, the fed tokens' PLE rows, the tier's xptr tables) so the
/// step itself is launches only: eager (`--no-fn-capture`), or one hipGraph per (bucket, `dp`)
/// captured on first use and replayed. Returns one [`RowOut`] per LIVE row.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn decode_step(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    lives: &mut [Live],
    bucket: u32,
    dp: usize,
    gr: &mut Graphs,
    f: &FnFlags,
    reqs: &[RowReq],
    w: u32,
    drafts: &[Vec<u32>],
) -> anyhow::Result<(Vec<RowOut>, StepStat)> {
    decode_step_with(
        rk,
        em,
        ple,
        ph,
        recs,
        lives,
        bucket,
        dp,
        gr,
        f,
        reqs,
        w,
        drafts,
        &mut |_| Ok(()),
        None,
        None,
    )
}

/// FN119d (`next`): the next step's reader of the round a commit left pending -- waits for the
/// chain's ids (read back behind the round) and returns each sequence's drafts at the step's width.
pub(super) type EarlyIds<'a> =
    &'a mut dyn FnMut(&mut Rank<'_, '_>) -> anyhow::Result<Vec<Vec<u32>>>;

/// FN119d: resolve the pending round inside a step (the drafts into `dloc`); false = none pending.
fn early_take(
    rk: &mut Rank<'_, '_>,
    early: &mut Option<EarlyIds<'_>>,
    dloc: &mut Vec<Vec<u32>>,
) -> anyhow::Result<bool> {
    match early.take() {
        Some(r) => {
            *dloc = r(rk)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// FN119d: the drafts' PLE rows (rows `i * w + t`, t >= 1) hashed once the ids are read, from each
/// sequence's n-gram context after its committed token (`cafter`), into `late`.
fn early_late(
    rk: &mut Rank<'_, '_>,
    ph: &PleHash,
    cafter: &[PleContext],
    dloc: &[Vec<u32>],
    wu: usize,
    late: &mut Vec<(usize, Vec<u64>)>,
) -> anyhow::Result<()> {
    for (i, c0) in cafter.iter().enumerate() {
        let mut c = *c0;
        for t in 1..wu {
            let pids = ph.ids_step(&mut c, &[dloc[i][t - 1]]);
            if let Some(tw) = rk.trace.as_mut() {
                tw.ple(true, &pids)?;
            }
            late.push((i * wu + t, pids));
        }
    }
    Ok(())
}

/// [`decode_step`] with `under`: host work run while the step's device work is in flight (after
/// the issue, before the host's one wait) -- FN113b: the prefix tree's pending demotions.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_step_with(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    lives: &mut [Live],
    bucket: u32,
    dp: usize,
    gr: &mut Graphs,
    f: &FnFlags,
    reqs: &[RowReq],
    w: u32,
    drafts: &[Vec<u32>],
    under: &mut dyn FnMut(&Rank<'_, '_>) -> anyhow::Result<()>,
    ah: Option<&super::dstep::AheadCmd>,
    early: Option<EarlyIds<'_>>,
) -> anyhow::Result<(Vec<RowOut>, StepStat)> {
    let t0 = Instant::now();
    // FN119d (`next`): the drafts this step feeds (read back under it when `early` holds the
    // pending round's reader)
    let mut early = early;
    let mut dloc: Vec<Vec<u32>> = drafts.to_vec();
    let n = lives.len();
    let b = bucket as usize;
    let wu = w as usize;
    ensure!(
        (1..=super::exec::WMAX).contains(&w) && w <= rk.pools.pool_w,
        "verify width {w} (the pools hold {} snapshots a sequence)",
        rk.pools.pool_w
    );
    ensure!(
        (w == 1 && drafts.is_empty())
            || (drafts.len() == n && drafts.iter().all(|d| d.len() + 1 == wu)),
        "a width-{w} step feeds the committed token + {} drafts a sequence",
        wu - 1
    );
    ensure!(
        n >= 1 && n <= b && b <= MAX_SEQS as usize,
        "{n} live rows in bucket {b}"
    );
    ensure!(
        lives.iter().enumerate().all(|(i, l)| l.slot as usize == i),
        "a decode batch is pool slots 0..n in order"
    );
    ensure!(
        lives.iter().all(|l| l.len + w <= rk.max_ctx),
        "a sequence reached the pools' context"
    );
    let mut stat = StepStat::default();
    // FN119d: the early path's step form (else the round is resolved now and the step runs as
    // `ahead`): the PLE layer-1 split on the one-graph lazy form, nothing that syncs the stream
    let early_form = w > 1
        && rk.ple_ov.is_some()
        && !f.no_fn_capture
        && !rk.dev.op_timing
        && matches!(rk.gsplit, super::run::GraphSplit::Lazy { .. })
        && gr.one.is_some()
        && rk.ring.as_ref().is_none_or(|r| r.plan.is_none())
        && rk.pg.is_none()
        && rk.xmiss.is_none()
        && f.fn_poison_step.is_none()
        && !rk.lev.defer
        && !f.fn_capture_verify;
    if early.is_some() && !early_form {
        early_take(rk, &mut early, &mut dloc)?;
        if let Some(a) = rk.ahead.as_mut() {
            a.c.next_res_form += 1;
        }
    }
    let is_early = early.is_some();
    // row s * w + t: sequence s's committed token (t = 0) or its draft t (FN15b's verify rows);
    // a pad sequence feeds token 0 at positions 0.. of its FREE slot
    let tok_of = |i: usize, t: usize| -> u32 {
        if i >= n {
            0
        } else if t == 0 {
            lives[i].next
        } else {
            dloc[i][t - 1]
        }
    };
    let toks: Vec<u32> = (0..b * wu).map(|r| tok_of(r / wu, r % wu)).collect();
    let mut pos: Vec<u32> = lives.iter().map(|l| l.len).collect();
    let mut na: Vec<u32> = lives.iter().map(|l| l.naccept).collect();
    let mut dr: Vec<u32> = lives.iter().map(|l| l.drop).collect();
    pos.resize(b, 0);
    na.resize(b, 1);
    dr.resize(b, 0);
    // FN56: the host phases before the step's first launch are named (the census)
    let mut tp = Instant::now();
    let mut lap = |stat: &mut StepStat, name: &'static str| {
        let now = Instant::now();
        *stat.host.entry(name).or_insert(0.0) += (now - tp).as_secs_f64() * 1e3;
        tp = now;
    };
    if is_early {
        // FN119d: the rows from the walk's result block + the chain's ids, on the card behind
        // the round (the host has not read the drafts)
        early_emb(rk, em, n, bucket, w)?;
    } else {
        let e = embed_rows(recs, &toks)?;
        buf_h2d(rk, BufK::Emb, &e)?;
    }
    lap(&mut stat, "embed");
    // FN51f-b: the elastic law before the step's KV step law
    elastic_tick(rk, None)?;
    let band = step_band(rk, &pos);
    let (mut tb, mut offs) = band_tables(rk, &pos, w, &na, &dr, rk.pools.pool_w, n, band);
    kv_extend(rk, &mut tb, &mut offs, &pos, w, n, band.1)?;
    let staged_tb = is_early
        && rk
            .ahead
            .as_ref()
            .and_then(|a| a.early.as_ref())
            .is_some_and(|e| tb.len() <= e.tb.len());
    if staged_tb {
        // FN119d: one async copy from the pinned staging (the stream holds the round; the last
        // early step's copy from this staging ran before its verify, which the host waited)
        let e = rk
            .ahead
            .as_mut()
            .and_then(|a| a.early.as_mut())
            .context("FN119d: the early staging")?;
        e.tb.as_mut_slice()[..tb.len()].copy_from_slice(&tb);
        rk.hip.memcpy_h2d_at(
            &rk.dev.consts,
            0,
            e.tb.slice(0, tb.len())
                .context("FN119d: the tables staging")?,
            &rk.dev.stream,
        )?;
    } else {
        rk.hip
            .memcpy_h2d_at_blocking(&rk.dev.consts, 0, &tb, &rk.dev.stream)?;
    }
    let cc = step_consts(rk, &offs, bucket, w, band, &pos);
    lap(&mut stat, "tables");
    // the fed tokens' PLE rows (pad rows: zeros), before the step; the sequence's n-gram context
    // advances over its COMMITTED token only (the caller advances it over the accepted drafts --
    // the rewind to the committed tokens)
    let g0 = Instant::now();
    let ovl = rk.ple_ov.is_some();
    let rb = (H * 2) as usize;
    let nst = b * wu * rb;
    // FN58b: the rows still to gather (row, ids) -- the drafts', and a committed token's whose
    // stash (gathered under the drafter round) does not match
    let mut late: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut cafter = Vec::new();
    let mut staged = vec![0u8; if ovl { 0 } else { nst }];
    if let Some(ov) = rk.ple_ov.as_mut() {
        let pre = std::mem::take(&mut ov.pre);
        *stat.host.entry("ple_pre").or_insert(0.0) += pre.0;
        stat.ple_hits += pre.2;
        stat.ple_nvme += pre.3;
        let stg = &mut ov.staged.as_mut_slice()[..nst];
        stg.fill(0);
        let mut stash = 0u64;
        for (i, lv) in lives.iter_mut().enumerate() {
            let pre = lv.pre.take();
            let mut c = lv.ctx;
            for t in 0..wu {
                if is_early && t >= 1 {
                    // FN119d: the drafts' rows once the ids are read (`early_late`)
                    break;
                }
                let tok = tok_of_row(lv.next, &dloc, i, t);
                let c0 = c;
                let pids = ph.ids_step(&mut c, &[tok]);
                if let Some(tw) = rk.trace.as_mut() {
                    tw.ple(true, &pids)?;
                }
                let r = i * wu + t;
                match &pre {
                    Some((pt, pc, row)) if t == 0 && *pt == tok && *pc == c0 => {
                        stg[r * rb..(r + 1) * rb].copy_from_slice(row);
                        stash += 1;
                    }
                    _ => late.push((r, pids)),
                }
                if t == 0 {
                    lv.ctx = c;
                }
            }
            if is_early {
                cafter.push(c);
            }
        }
        *stat.host.entry("ple_rows_stash").or_insert(0.0) += stash as f64;
        *stat.host.entry("ple_rows_late").or_insert(0.0) += late.len() as f64;
    } else if rk.lev.ple {
        // FN56: every live row's ids in ONE gather (one parallel NVMe batch); the rows are
        // `i * wu + t` in order, so the live rows are the staging's first `n * wu` rows and the
        // bytes are the per-row gathers' (a row's bytes are its ids' records, wherever read)
        let mut all: Vec<u64> = Vec::new();
        for (i, lv) in lives.iter_mut().enumerate() {
            let mut c = lv.ctx;
            for t in 0..wu {
                let pids = ph.ids_step(&mut c, &[tok_of_row(lv.next, &dloc, i, t)]);
                // every row the same id count (the gather checks the total against the rows)
                ensure!(
                    all.is_empty() || all.len() % pids.len() == 0,
                    "a PLE row of {} ids",
                    pids.len()
                );
                if let Some(tw) = rk.trace.as_mut() {
                    tw.ple(true, &pids)?;
                }
                all.extend_from_slice(&pids);
                if t == 0 {
                    lv.ctx = c;
                }
            }
        }
        let g = ple
            .gather(&all, &mut staged[..n * wu * 5120])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        stat.ple_hits += g.cache_hits;
        stat.ple_nvme += g.nvme_reads;
    } else {
        for (i, lv) in lives.iter_mut().enumerate() {
            let mut c = lv.ctx;
            for t in 0..wu {
                let pids = ph.ids_step(&mut c, &[tok_of_row(lv.next, &dloc, i, t)]);
                let r = i * wu + t;
                let g = ple
                    .gather(&pids, &mut staged[r * 5120..(r + 1) * 5120])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if let Some(tw) = rk.trace.as_mut() {
                    tw.ple(true, &pids)?;
                }
                stat.ple_hits += g.cache_hits;
                stat.ple_nvme += g.nvme_reads;
                if t == 0 {
                    lv.ctx = c;
                }
            }
        }
    }
    stat.ple_ms += g0.elapsed().as_secs_f64() * 1e3;
    lap(&mut stat, "ple");
    if !ovl {
        buf_h2d(rk, BufK::Rows, &staged)?;
        lap(&mut stat, "rows");
    }
    if let Some(p) = f.fn_poison_step {
        rk.hip.memset(&rk.dev.arena, p, &rk.dev.stream)?;
    }
    let complete = tier_begin_up(rk, em, bucket * w, false, is_early)?;
    lap(&mut stat, "tier");
    if early.is_some() && !complete {
        early_take(rk, &mut early, &mut dloc)?;
        early_late(rk, ph, &cafter, &dloc, wu, &mut late)?;
        if let Some(a) = rk.ahead.as_mut() {
            a.c.next_res_tier += 1;
        }
    }
    let use_graph = !f.no_fn_capture && complete && !rk.dev.op_timing;
    // FN58b: the late rows land before the layer-1 graph (per-layer graphs only); any other form
    // of the step gathers and uploads them here, before its first launch
    // FN64: the lazy split runs where the step's emission is the plain layer train (no prefill
    // staging ring, no pre-gate capture arm); elsewhere the per-layer form
    let ringed = rk.ring.as_ref().is_some_and(|r| r.plan.is_some());
    let lazy = match rk.gsplit {
        super::run::GraphSplit::Lazy { q, cache, .. }
            if use_graph && !ringed && rk.pg.is_none() =>
        {
            Some((q, cache))
        }
        _ => None,
    };
    let per_layer = lazy.is_none() && rk.gsplit != super::run::GraphSplit::Step;
    let at_l1 = ovl && use_graph && (per_layer || lazy.is_some());
    if early.is_some() && !(at_l1 && lazy.is_some()) {
        early_take(rk, &mut early, &mut dloc)?;
        early_late(rk, ph, &cafter, &dloc, wu, &mut late)?;
        if let Some(a) = rk.ahead.as_mut() {
            a.c.next_res_form += 1;
        }
    }
    if ovl && !at_l1 {
        let g1 = Instant::now();
        gate121(rk, ple, &late, &mut stat)?;
        let ov = rk.ple_ov.as_mut().context("ple_ov")?;
        let c = ple_late(ple, ov.staged.as_mut_slice(), &late)?;
        miss121(&mut rk.f121, &stat, g1);
        stat.ple_hits += c.0;
        stat.ple_nvme += c.1;
        rk.hip.memcpy_h2d_at_blocking(
            &rk.rows.own,
            rk.rows.off,
            &ov.staged.as_slice()[..nst],
            &rk.dev.stream,
        )?;
        stat.ple_ms += g1.elapsed().as_secs_f64() * 1e3;
        lap(&mut stat, "ple_l0");
    }
    let ev = (
        rk.hip.create_event(EventFlags::Default)?,
        rk.hip.create_event(EventFlags::Default)?,
    );
    let launches0 = rk.dev.launches;
    let lg;
    let t_ev0;
    gr.rank = Some(rk.rank);
    gr.split = match (lazy, per_layer) {
        (Some(_), _) => "lazy",
        (None, true) => "layer",
        (None, false) => "step",
    };
    if use_graph {
        // FN64: the layers whose gate waits this step (copies in flight), before any gate
        let gated: Vec<bool> = (0..LAYERS).map(|l| rk.tier.mover.in_flight(l)).collect();
        // FN53c: a miss-flag step's graphs hold one more launch a layer: their own forms
        // (parity + 2; never recorded for the pre-capture)
        let xm_key = if rk.xmiss.as_ref().is_some_and(|x| x.step) {
            2
        } else {
            0
        };
        let key = (bucket, w, dp + xm_key, cc.qsa_short, cc.max_ctx);
        // the replay plan: (first layer, end layer) a graph; the head = (LAYERS, LAYERS + 1)
        let seg_plan: Vec<(u32, u32)> = if let Some((q, cache)) = lazy {
            // FN98: one exec per form (the gates inside); FN64: the runs between gated blocks
            let v = if gr.one.is_some() {
                one_segs(at_l1.then_some(plan::PLE_LAYER))
            } else {
                let mut v =
                    plan::lazy_runs(|l| gated[l as usize], q, at_l1.then_some(plan::PLE_LAYER));
                v.push((LAYERS, LAYERS + 1));
                v
            };
            // FN119d: a run to capture syncs the stream (the round with it): resolve first
            if early.is_some() {
                let ready = gr
                    .lazy
                    .get(&key)
                    .is_some_and(|lk| lk.seen && v.iter().all(|s| lk.runs.contains_key(s)));
                if !ready {
                    early_take(rk, &mut early, &mut dloc)?;
                    early_late(rk, ph, &cafter, &dloc, wu, &mut late)?;
                    if let Some(a) = rk.ahead.as_mut() {
                        a.c.next_res_capture += 1;
                    }
                }
            }
            lazy_capture(
                rk,
                em,
                gr,
                key,
                &v,
                dp,
                complete,
                cc,
                &mut stat,
                f.fn_capture_verify,
                cache,
                false,
                gr.one,
            )?;
            v
        } else if per_layer {
            (0..=LAYERS).map(|l| (l, l + 1)).collect()
        } else {
            vec![(0, LAYERS + 1)]
        };
        if !per_layer && lazy.is_none() {
            // ONE graph: every layer's copies must land before it (law 3, hoisted)
            for l in 0..LAYERS {
                rk.tier
                    .mover
                    .gate(l, &rk.dev.stream)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }
        if lazy.is_none() && !gr.map.contains_key(&key) {
            let tc = Instant::now();
            rk.hip.sync_stream(&rk.dev.stream)?;
            let (l0, m0, c0) = (rk.dev.launches, rk.dev.memsets, rk.copies);
            rk.dev.digest = super::dev::FNV0;
            let mut segs = Vec::new();
            let mut nodes = 0usize;
            #[allow(clippy::needless_late_init)]
            let lgc;
            let cap = capture_seg;
            if per_layer {
                let mut input = LayerIn::Embed(rk.emb.addr().as_u64());
                let rows = rk.rows.addr().as_u64();
                for l in 0..LAYERS {
                    let st = rk.pools.batch(l as usize, dp);
                    let mut out = None;
                    let (e, g, n) = cap(rk, &mut |rk| {
                        out = Some(layer(
                            rk, em, l, complete, st, input, rows, cc, &mut stat, false,
                        )?);
                        Ok(())
                    })?;
                    input = LayerIn::Pending(out.context("layer")?.pend);
                    nodes += n;
                    segs.push((e, g));
                }
                let LayerIn::Pending(pend) = input else {
                    unreachable!()
                };
                let mut lgo = 0;
                let (e, g, n) = cap(rk, &mut |rk| {
                    let (_, lg, strm) =
                        emit_post3(em, &rk.fm, rk.head, &pend, u64::from(cc.m), true)?;
                    rk.dev.exec(&rk.table, em)?;
                    mtp_keep_stream(rk, strm, cc.m)?;
                    lgo = lg;
                    Ok(())
                })?;
                nodes += n;
                segs.push((e, g));
                lgc = lgo;
            } else {
                let mut lgo = 0;
                let (e, g, n) = cap(rk, &mut |rk| {
                    lgo = step_body(rk, em, bucket, dp, complete, cc, &mut stat, false)?;
                    Ok(())
                })?;
                nodes += n;
                segs.push((e, g));
                lgc = lgo;
            }
            let (dl, dm, dc) = (rk.dev.launches - l0, rk.dev.memsets - m0, rk.copies - c0);
            ensure!(
                nodes as u64 == dl + dm + dc,
                "b{bucket} p{dp}: the graphs hold {nodes} nodes; the step issued {dl} launches + {dm} memsets + {dc} copies"
            );
            gr.captured += 1;
            gr.capture_ms += tc.elapsed().as_secs_f64() * 1e3;
            println!(
                "fngraph rank{}: captured b{bucket} w{w} parity {dp}: {} graph(s), {nodes} nodes = {dl} launches + {dm} memsets + {dc} route-log copies ({:.1} ms)",
                rk.rank,
                segs.len(),
                tc.elapsed().as_secs_f64() * 1e3
            );
            gr.map.insert(
                key,
                StepGraph {
                    segs,
                    digest: rk.dev.digest,
                    nodes,
                    launches: dl,
                    memsets: dm,
                    copies: dc,
                    lg: lgc,
                },
            );
        } else if lazy.is_none() && f.fn_capture_verify {
            // the by-construction law, checked: the step re-emitted (no launch) == the graphs
            rk.dev.digest = super::dev::FNV0;
            rk.dev.dry = true;
            let body = step_body(rk, em, bucket, dp, complete, cc, &mut stat, false);
            rk.dev.dry = false;
            let lgv = body?;
            let g = &gr.map[&key];
            ensure!(
                rk.dev.digest == g.digest && lgv == g.lg,
                "b{bucket} p{dp}: the re-emitted step differs from the captured graph ({:016x} vs {:016x})",
                rk.dev.digest,
                g.digest
            );
            gr.verified += 1;
        }
        lap(&mut stat, "tier");
        // FN98: law 3 INSIDE the execs -- every gate block's table entry = its LAST issued mark
        // (the one-stream mover lands its marks in issue order, so that word covers the block's
        // other layers) or value 0 (nothing in flight: the gate passes); the table crosses in
        // order on the step's stream before the first exec
        let mut one98 = (0u64, 0u64);
        if gr.one.is_some() && lazy.is_some() {
            let blocks: Vec<(u32, u32)> = seg_plan
                .iter()
                .flat_map(|s| gr.lazy[&key].runs[s].gates.iter().copied())
                .collect();
            let (tdev, tpin) = gr.gate_tab.as_mut().context("FN98: the gate table")?;
            let stg = tpin.as_mut_slice();
            stg.fill(0);
            for (b0, b1) in blocks {
                let mut best: Option<(saber_hip::DeviceAddr, u32, u64)> = None;
                for l in b0..b1 {
                    if let Some(m) = rk
                        .tier
                        .mover
                        .gate_mark(l, &rk.dev.stream)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                    {
                        one98.1 += 1;
                        if best.is_none_or(|b| m.2 > b.2) {
                            best = Some(m);
                        }
                    }
                }
                if let Some((word, value, _)) = best.filter(|_| !gr.one_open) {
                    let o = b0 as usize * GATE98_ENT;
                    stg[o..o + 8].copy_from_slice(&word.as_u64().to_le_bytes());
                    stg[o + 8..o + 12].copy_from_slice(&value.to_le_bytes());
                    one98.0 += 1;
                }
            }
            let n = LAYERS as usize * GATE98_ENT;
            rk.hip.memcpy_h2d_at(
                tdev,
                0,
                tpin.slice(0, n).context("FN98: the gate table staging")?,
                &rk.dev.stream,
            )?;
        }
        rk.hip.record_event(&ev.0, &rk.dev.stream)?;
        t_ev0 = Instant::now();
        for (si, &(a, e)) in seg_plan.iter().enumerate() {
            if at_l1 && a == super::plan::PLE_LAYER {
                // FN58b: the late rows gathered while the layers before the PLE layer run; the
                // copy crosses on the side stream (after everything before the step: ev.0) and
                // the PLE layer's graph waits on it -- every HIP call on this thread
                lap(&mut stat, "issue");
                let g1 = Instant::now();
                let ov = rk.ple_ov.as_mut().context("ple_ov")?;
                rk.hip.record_event(&ov.ev.0, &rk.dev.stream)?;
                if early.is_some() {
                    // FN119d: the layers before the PLE layer are queued behind the round: the
                    // round's ids now (its copy right behind it), then the drafts' rows
                    let ti = Instant::now();
                    early_take(rk, &mut early, &mut dloc)?;
                    let wait = ti.elapsed().as_secs_f64() * 1e3;
                    *stat.host.entry("early_ids").or_insert(0.0) += wait;
                    early_late(rk, ph, &cafter, &dloc, wu, &mut late)?;
                    early_emb_check(rk, recs, lives, &dloc, n, b, wu)?;
                    if let Some(a) = rk.ahead.as_mut() {
                        a.c.next_early += 1;
                        a.c.next_ids_wait_ms.add(wait);
                    }
                }
                gate121(rk, ple, &late, &mut stat)?;
                let ov = rk.ple_ov.as_mut().context("ple_ov")?;
                let c = ple_late(ple, ov.staged.as_mut_slice(), &late)?;
                miss121(&mut rk.f121, &stat, g1);
                stat.ple_hits += c.0;
                stat.ple_nvme += c.1;
                if let Some(side) = ov.stream.as_ref() {
                    rk.hip.stream_wait_event(side, &ev.0)?;
                    rk.hip.memcpy_h2d_at(
                        &rk.rows.own,
                        rk.rows.off,
                        ov.staged.slice(0, nst).context("the PLE staging")?,
                        side,
                    )?;
                    rk.hip.record_event(&ov.ev.1, side)?;
                    rk.hip.stream_wait_event(&rk.dev.stream, &ov.ev.1)?;
                } else {
                    // FN58c (`layer1-inline`): the copy in order on the step's stream, between
                    // the layer-0 graph and the layer-1 graph -- no second stream
                    rk.hip.memcpy_h2d_at(
                        &rk.rows.own,
                        rk.rows.off,
                        ov.staged.slice(0, nst).context("the PLE staging")?,
                        &rk.dev.stream,
                    )?;
                    rk.hip.record_event(&ov.ev.1, &rk.dev.stream)?;
                }
                rk.hip.record_event(&ov.ev.2, &rk.dev.stream)?;
                stat.ple_ms += g1.elapsed().as_secs_f64() * 1e3;
                lap(&mut stat, "ple_l0");
            }
            // law 3 at the layer, as the eager step: the graph of a layer run waits only for its
            // own layers' copies (FN64: a run's gates all before its graph -- a run has copies in
            // flight only for its first quantum), so the LRU's promotions overlap the layers before
            for i in if (per_layer || lazy.is_some()) && gr.one.is_none() {
                a..e.min(LAYERS)
            } else {
                0..0
            } {
                // FN54b: the gate timer brackets a gate that will wait (copies in flight)
                let timed = rk.dev.gate_time && rk.tier.mover.in_flight(i);
                let k = rk.dev.gate_used;
                if timed {
                    while rk.dev.gate_pool.len() <= k {
                        let a = rk.hip.create_event(EventFlags::Default)?;
                        let b = rk.hip.create_event(EventFlags::Default)?;
                        rk.dev.gate_pool.push((a, b));
                    }
                    rk.hip
                        .record_event(&rk.dev.gate_pool[k].0, &rk.dev.stream)?;
                }
                rk.tier
                    .mover
                    .gate(i, &rk.dev.stream)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if timed {
                    rk.hip
                        .record_event(&rk.dev.gate_pool[k].1, &rk.dev.stream)?;
                    rk.dev.gate_used += 1;
                }
            }
            let exec = if lazy.is_some() {
                &gr.lazy[&key].runs[&(a, e)].exec
            } else {
                &gr.map[&key].segs[si].0
            };
            rk.hip.graph_launch(exec, &rk.dev.stream)?;
        }
        rk.hip.record_event(&ev.1, &rk.dev.stream)?;
        ensure!(
            early.is_none(),
            "FN119d: an early step passed its PLE layer with the round's ids unread"
        );
        gr.one_sets += one98.0;
        gr.one_gated += one98.1;
        if lazy.is_some() {
            let lk = &gr.lazy[&key];
            stat.launches = seg_plan.iter().map(|s| lk.runs[s].launches).sum();
            lg = lk.lg;
        } else {
            let g = &gr.map[&key];
            stat.launches = g.launches;
            lg = g.lg;
        }
        gr.replays += 1;
        gr.seg_launches += seg_plan.len() as u64;
        let ng = gated.iter().filter(|&&x| x).count();
        gr.gated += ng as u64;
        *stat.host.entry("n_graphs").or_insert(0.0) += seg_plan.len() as f64;
        *stat.host.entry("n_gated").or_insert(0.0) += ng as f64;
        gr.mtp_execs = rk
            .mtp
            .as_ref()
            .map_or(0, |m| (m.graphs.len() + m.chain_graphs.len()) as u64);
        if gr.replays % 4096 == 0 {
            gr.free_mib = rk.hip.mem_info()?.0 >> 20;
            println!("{}", gr.line());
            if gr.one.is_some() {
                println!("{}", gr.line98());
            }
            gr.write_record()?;
        }
    } else {
        rk.hip.record_event(&ev.0, &rk.dev.stream)?;
        t_ev0 = Instant::now();
        lg = step_body(rk, em, bucket, dp, complete, cc, &mut stat, true)?;
        rk.hip.record_event(&ev.1, &rk.dev.stream)?;
        stat.launches = rk.dev.launches - launches0;
        gr.eager += 1;
    }
    // FN30b: the target step's end (a no-op without --fn-diag-probe)
    rk.diag.bucket = bucket;
    fn30b_probe(rk, 0)?;
    lap(&mut stat, "issue");
    tier_refill(rk)?;
    lap(&mut stat, "refill");
    let mut rq: Vec<RowReq> = reqs.to_vec();
    rq.resize(b * wu, RowReq::default());
    // FN112 (`--fn-overlap on`): the pick's seam + rows and the tier end's route log queued behind
    // the step (shapes from the requests and the step's rows; never under FN56's `defer`, whose
    // tier end reads the log later)
    let deferred = rk.lev.defer && rk.mtp.is_some() && mtp_verify(f);
    let ov = rk.ov112.is_some();
    if ov {
        let log_m = (complete && !deferred).then_some(cc.m);
        ov112_queue(rk, lg, b * wu, &rq, log_m)?;
    }
    // FN119 (`--fn-device-step ahead`): the device walk + the drafter round issued behind the
    // verify, before the host waits
    if let Some(ah) = ah {
        ensure!(ov, "FN119: the issue-ahead needs FN112's queued pick");
        let t = Instant::now();
        ahead_issue(
            rk,
            em,
            recs,
            ah,
            &pos[..n],
            &dloc,
            w,
            bucket,
            lg,
            None,
            None,
        )?;
        *stat.host.entry("ahead_issue").or_insert(0.0) += t.elapsed().as_secs_f64() * 1e3;
    }
    // FN113b: host-only work under the step's device span (the prefix tree's demotions)
    under(rk)?;
    lap(&mut stat, "under");
    // FN57: the step's one host wait, by the boot's arm; its excess past the card is the issue-to-
    // return wall minus the device span (what the sleeper adds, plus the issue latency it shares)
    let wkey = (1u64 << 63) | (u64::from(bucket) << 8) | u64::from(w);
    let tw = Instant::now();
    if rk.xmiss.as_ref().is_some_and(|x| x.step) {
        xmiss_wait(rk, &ev.1)?;
    }
    match rk.ov112.as_ref() {
        Some(o) if ov => rk.dev.wait_event(&o.ev, wkey)?,
        _ => rk.dev.wait_event(&ev.1, wkey)?,
    }
    let tr = Instant::now();
    lap(&mut stat, "wait");
    stat.dev_ms = f64::from(rk.hip.elapsed_ms(&ev.0, &ev.1)?);
    if let Some(o) = rk.ov112.as_mut() {
        o.dev_ms += f64::from(rk.hip.elapsed_ms(&ev.1, &o.ev)?);
    }
    rk.dev.waits.borrow_mut().push((
        wkey,
        (tr - t0).as_secs_f64() * 1e3,
        (tr - tw).as_secs_f64() * 1e3,
        (tr - t_ev0).as_secs_f64() * 1e3 - stat.dev_ms,
    ));
    for k in 0..std::mem::take(&mut rk.dev.gate_used) {
        let (a, b) = &rk.dev.gate_pool[k];
        stat.gate_ms += f64::from(rk.hip.elapsed_ms(a, b)?);
        stat.gate_n += 1;
    }
    if at_l1 {
        // FN58b: the device idle at the PLE layer's gate (layer 0's end -> the layer-1 graph's
        // start: the late gather, the copy, the host's issue) and where the copy landed against
        // layer 0's end (< 0: before it)
        let ov = rk.ple_ov.as_ref().context("ple_ov")?;
        let gap = f64::from(rk.hip.elapsed_ms(&ov.ev.0, &ov.ev.2)?);
        let cp = f64::from(rk.hip.elapsed_ms(&ov.ev.0, &ov.ev.1)?);
        *stat.host.entry("ple_gate_wait").or_insert(0.0) += gap.max(0.0);
        *stat.host.entry("ple_copy_vs_l0").or_insert(0.0) += cp;
    }
    rk.trace_meta = (
        saber_mem::route_trace::StepKind::Decode,
        bucket,
        w,
        n as u32 * w,
    );
    // FN30c: under MTP-through-verify the drafter round follows this step on the stream: the
    // hold law keeps the step's copies off the device until `mtp_after_verify` releases them
    rk.tier.mover.set_hold(f.mover_hold() && mtp_verify(f));
    if deferred {
        // FN56: the fence now, the rest under the drafter round that follows
        tier_end_fence(rk, complete, cc.m)?;
    } else {
        tier_end(rk, complete, cc.m, &mut stat)?;
    }
    times(rk, &mut stat)?;
    lap(&mut stat, "tier_end");
    rk.last_lg = lg;
    let mut outs = pick2(rk, lg, b * wu, &rq)?;
    // FN112: nothing queued outlives its step
    if let Some(o) = rk.ov112.as_mut() {
        if let Some(p) = o.pend.take() {
            ensure!(
                p.log_m.is_none(),
                "FN112: a queued route log was not consumed by its tier end"
            );
            o.miss += 1;
        }
    }
    outs.truncate(n * wu);
    for (k, x) in ["pick_read", "pick_seam", "pick_xchg"]
        .into_iter()
        .zip(rk.pick_t)
    {
        *stat.host.entry(k).or_insert(0.0) += x;
    }
    lap(&mut stat, "pick");
    stat.host_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((outs, stat))
}

/// The probe of sequence `p` at generated token `t`: FN3's top-20 ids of that step (empty past it).
pub(super) fn ref_req(p: Option<&Prompt>, t: usize) -> RowReq {
    RowReq {
        probes: p
            .and_then(|p| p.ref_top.get(t))
            .cloned()
            .unwrap_or_default(),
        temp: 0.0,
        lse: false,
        tk: 0,
    }
}

/// Decode `n` greedy tokens for every sequence of `lives` together (SPEC 0: one row a sequence,
/// padded to the bucket). Each step feeds the last pick; the sequences share one PLE parity
/// (normalized at the start).
#[allow(clippy::too_many_arguments)]
fn decode(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    lives: &mut [Live],
    n: u32,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    group: &[&Prompt],
) -> anyhow::Result<Decoded> {
    let b = lives.len();
    for lv in lives.iter_mut() {
        rk.pools
            .normalize(rk.hip, &rk.dev, lv.slot, lv.gdn_at, lv.ple_at, *dp)?;
        lv.gdn_at = 0;
        lv.ple_at = *dp;
    }
    if rk.mtp.is_some() && mtp_verify(f) {
        return verify::decode_mtp(rk, em, ple, ph, recs, lives, n, f, gr, dp, group);
    }
    let mut res = Decoded {
        acc: Vec::new(),
        widths: Vec::new(),
        mtp: vec![Vec::new(); b],
        mtp_ms: Vec::new(),
        ids: vec![Vec::new(); b],
        tops: vec![Vec::new(); b],
        probes: vec![Vec::new(); b],
        shas: vec![Vec::new(); b],
        steps: Vec::new(),
    };
    let bucket = bucket_for(b);
    if rk.mtp.is_some() {
        let t0 = Instant::now();
        let mut dr = Vec::new();
        let mut pos = Vec::new();
        for lv in lives.iter() {
            let (d, p) = rk.mtp.as_ref().context("mtp")?.first[lv.slot as usize]
                .context("the prefill's first MTP pass")?;
            dr.push(vec![d]);
            pos.push(p);
        }
        rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
        mtp_chain(rk, em, recs, &mut dr, &pos, bucket)?;
        let dev: f64 = rk.mtp.as_ref().context("mtp")?.pass_ms.iter().sum();
        res.mtp_ms.push((t0.elapsed().as_secs_f64() * 1e3, dev));
        for (i, d) in dr.into_iter().enumerate() {
            res.mtp[i].push(d);
        }
    }
    for t in 1..n {
        let reqs: Vec<RowReq> = (0..b)
            .map(|i| ref_req(group.get(i).copied(), t as usize))
            .collect();
        let (outs, stat) = decode_step(
            rk,
            em,
            ple,
            ph,
            recs,
            lives,
            bucket,
            *dp,
            gr,
            f,
            &reqs,
            1,
            &[],
        )?;
        *dp = 1 - *dp;
        for (i, lv) in lives.iter_mut().enumerate() {
            let id = outs[i].top[0].0;
            lv.len += 1;
            lv.ple_at = *dp;
            lv.next = id;
            res.ids[i].push(id);
            res.tops[i].push(outs[i].top[..TOPK].to_vec());
            res.probes[i].push((outs[i].lse, outs[i].probe.clone()));
            res.shas[i].push(outs[i].sha.clone());
        }
        res.steps.push(stat);
        if rk.mtp.is_some() && t + 1 < n {
            // FN15's shadow drafter: pass 1 on this step's stream + the committed token at the
            // fed row's position, then the chain
            let t0 = Instant::now();
            let mut toks: Vec<u32> = lives.iter().map(|l| l.next).collect();
            let pos: Vec<u32> = lives.iter().map(|l| l.len - 1).collect();
            let mut pp = pos.clone();
            toks.resize(bucket as usize, 0);
            pp.resize(bucket as usize, 0);
            let e = embed_rows(recs, &toks)?;
            buf_h2d(rk, BufK::Emb, &e)?;
            let cc = decode_consts(rk, &pp, b)?;
            let st = rk.pools.batch(LAYERS as usize, 0);
            let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
            let gather: Vec<(u32, u32)> = (0..b as u32).map(|i| (i, i)).collect();
            rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
            let emb = rk.emb.addr().as_u64();
            let lg = mtp_pass(rk, em, bucket, emb, hid, cc, st, &gather, true, None, None)?;
            let o1 = pick2(
                rk,
                lg,
                bucket as usize,
                &vec![RowReq::default(); bucket as usize],
            )?;
            let mut dr: Vec<Vec<u32>> = (0..b).map(|i| vec![o1[i].top[0].0]).collect();
            mtp_chain(rk, em, recs, &mut dr, &pos, bucket)?;
            let dev: f64 = rk.mtp.as_ref().context("mtp")?.pass_ms.iter().sum();
            res.mtp_ms.push((t0.elapsed().as_secs_f64() * 1e3, dev));
            for (i, d) in dr.into_iter().enumerate() {
                res.mtp[i].push(d);
            }
        }
    }
    Ok(res)
}

impl Pools {
    /// Move sequence slot `from`'s every pool region to slot `to` (the compaction of a batch whose
    /// middle sequence left: the live set stays pool slots `0..n`).
    fn move_slot(
        &self,
        hip: &HipThread,
        dev: &RankDev<'_>,
        from: u32,
        to: u32,
    ) -> anyhow::Result<()> {
        if let Some(k) = self.kvt.as_ref() {
            // FN45b2: the tier moves the sequence's page rows, no K/V byte
            k.borrow_mut().move_seq(from, to)?;
        }
        // index 48 = the MTP drafter's own KV / indexer pools (FN15b: they move with the slot)
        for l in 0..=LAYERS as usize {
            let (a, z) = (self.seq(from, l), self.seq(to, l));
            let spans: Vec<(u64, u64, u64)> =
                if l == LAYERS as usize || plan::kind(l as u32) == LayerKind::Qsa {
                    vec![
                        (z.kv, a.kv, self.kvb_of(l)),
                        (z.ring, a.ring, RING * 256),
                        (z.ringp, a.ringp, RING * 24),
                        // FN51: a paged store moves its row (the tier's `move_seq`), no byte
                        (
                            z.comp,
                            a.comp,
                            if self.ix_paged {
                                0
                            } else {
                                u64::from(self.nbs) * 4 * 256
                            },
                        ),
                    ]
                } else {
                    vec![
                        (z.cst, a.cst, self.cb),
                        (z.h[0], a.h[0], u64::from(self.pool_w) * self.hb),
                        (z.h[1], a.h[1], self.hb),
                    ]
                };
            for (d, s, n) in spans {
                if n > 0 {
                    self.copy(hip, dev, d, s, n)?;
                }
            }
        }
        let (a, z) = (self.seq(from, 0), self.seq(to, 0));
        self.copy(hip, dev, z.ple_st[0], a.ple_st[0], self.pb)?;
        self.copy(hip, dev, z.ple_st[1], a.ple_st[1], self.pb)
    }
}

/// FN14's bucket-transition schedule: 8 sequences join one by one (every 3 steps) and leave at
/// staggered lengths (a middle slot first, so the pools compact), so the decode batch walks
/// 1 -> 2 -> 4 -> 8 -> 4 -> 2 -> 1 through every captured bucket and back. The per-sequence ids
/// of a captured run must equal an eager run's (`--no-fn-capture`) of the same schedule.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn grow(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    prompts: &[Prompt],
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
) -> anyhow::Result<Value> {
    const JOIN: [usize; 8] = [0, 3, 6, 9, 12, 15, 18, 21];
    // total tokens per sequence: s5 leaves first, then s7, s4, s6, s1 (a middle slot), s3, s2, s0
    const TARGET: [usize; 8] = [43, 32, 33, 28, 19, 12, 15, 8];
    ensure!(prompts.len() >= 8, "--fn-sched grow takes 8 prompts");
    let mut live: Vec<(usize, Live)> = Vec::new();
    let mut gens: Vec<Vec<u32>> = vec![Vec::new(); 8];
    let mut probes: Vec<Vec<(f32, Vec<f32>)>> = vec![Vec::new(); 8];
    let mut log = Vec::new();
    let mut moves = 0u32;
    let mut step = 0usize;
    while step <= 64 {
        for i in 0..8 {
            if JOIN[i] != step {
                continue;
            }
            let slot = live.len() as u32;
            rk.pools.zero(rk.hip, &rk.dev, slot)?;
            let mut lv = Live::fresh(slot, cfg.eos_token_id);
            let (pk, _, po) = prefill(
                rk,
                em,
                ple,
                ph,
                recs,
                &mut lv,
                &prompts[i].ids,
                f,
                None,
                None,
                Some(&ref_req(Some(&prompts[i]), 0)),
            )?;
            lv.next = pk.0[0];
            gens[i].push(pk.0[0]);
            probes[i].push(
                po.map(|o| (o.lse, o.probe))
                    .unwrap_or((f32::NAN, Vec::new())),
            );
            rk.pools
                .normalize(rk.hip, &rk.dev, lv.slot, lv.gdn_at, lv.ple_at, *dp)?;
            lv.gdn_at = 0;
            lv.ple_at = *dp;
            live.push((i, lv));
        }
        // departures: a sequence that has its tokens leaves; the last slot fills its hole
        while let Some(k) = live.iter().position(|(i, _)| gens[*i].len() >= TARGET[*i]) {
            let last = live.len() - 1;
            if k != last {
                rk.pools.move_slot(rk.hip, &rk.dev, last as u32, k as u32)?;
                if let Some(mt) = rk.mtp.as_ref() {
                    // FN16: the drafter's stream row moves with its slot
                    rk.hip.run_batch(&[Job::MemcpyD2d {
                        dst: &mt.prev,
                        dst_offset: k as u64 * W * 2,
                        src: &mt.prev,
                        src_offset: last as u64 * W * 2,
                        bytes: W * 2,
                        stream: &rk.dev.stream,
                    }])?;
                }
                moves += 1;
            }
            live.swap_remove(k);
            if k < live.len() {
                live[k].1.slot = k as u32;
            }
        }
        if live.is_empty() {
            if JOIN.iter().all(|&j| j < step) {
                break;
            }
            step += 1;
            continue;
        }
        let bucket = bucket_for(live.len());
        let reqs: Vec<RowReq> = live
            .iter()
            .map(|(i, _)| ref_req(Some(&prompts[*i]), gens[*i].len()))
            .collect();
        let mut ls: Vec<Live> = live.iter().map(|x| x.1.clone()).collect();
        let (outs, stat) = decode_step(
            rk,
            em,
            ple,
            ph,
            recs,
            &mut ls,
            bucket,
            *dp,
            gr,
            f,
            &reqs,
            1,
            &[],
        )?;
        *dp = 1 - *dp;
        for (k, x) in live.iter_mut().enumerate() {
            x.1 = ls[k].clone();
            let id = outs[k].top[0].0;
            x.1.len += 1;
            x.1.ple_at = *dp;
            x.1.next = id;
            gens[x.0].push(id);
            probes[x.0].push((outs[k].lse, outs[k].probe.clone()));
        }
        log.push(json!({"step": step, "live": live.len(), "bucket": bucket,
            "seqs": live.iter().map(|x| x.0).collect::<Vec<_>>(),
            "host_ms": stat.host_ms, "dev_ms": stat.dev_ms, "launches": stat.launches}));
        step += 1;
    }
    if rk.rank == 0 {
        let walk: Vec<u32> = log
            .iter()
            .map(|v| v["bucket"].as_u64().unwrap_or(0) as u32)
            .collect();
        let mut w = walk.clone();
        w.dedup();
        println!(
            "fngrow: {} steps, bucket walk {:?}, {moves} compaction moves",
            walk.len(),
            w
        );
    }
    Ok(json!({"moves": moves, "steps": log,
        "seqs": (0..8).map(|i| json!({"id": prompts[i].id, "join": JOIN[i], "target": TARGET[i],
            "gen": gens[i], "probe": probes[i]})).collect::<Vec<_>>()}))
}

/// FN125: what the image-keyed cache counted on a rank (`fnimg125` end line).
#[derive(Debug, Clone, Copy, Default)]
struct Img125 {
    /// Encodes whose tower run was deferred (the pixels held, no tower).
    deferred: u64,
    /// Deferred towers run (a computed chunk touched an image run).
    runs: u64,
    /// Image prompts whose prompt finished with the tower never run (every image run reloaded).
    skipped: u64,
    /// Prefix-tree pieces / prompt-cache prefixes over an image run: reloaded, computed.
    img_hits: u64,
    img_runs: u64,
    /// Image prompts that took the FN55b bypass (no content hashes in their facts, or the arm off).
    bypass: u64,
    /// Image prompts whose prefill finished; prefill rows COMPUTED (every prompt; image prompts').
    prompts: u64,
    rows: u64,
    img_rows: u64,
}

impl Img125 {
    fn line(&self, rank: u32, on: bool) -> String {
        format!(
            "fnimg125 rank{rank}: image cache {} | image prompts {} (bypass {}) | encodes \
             deferred {} | towers run {} | towers skipped {} (every image run reloaded) | image \
             pieces reloaded {} computed {} | prefill rows computed {} (image prompts {})",
            if on { "ON" } else { "off (FN55b bypass)" },
            self.prompts,
            self.bypass,
            self.deferred,
            self.runs,
            self.skipped,
            self.img_hits,
            self.img_runs,
            self.rows,
            self.img_rows
        )
    }

    /// `/metrics` (refreshed after every image prompt's last chunk on a vision boot).
    fn render(&self) -> String {
        let mut s = String::new();
        for (n, v) in [
            ("image_prompts_total", self.prompts),
            // LR-D: image prompts that bypass the tree (computed, never stored)
            ("image_bypass_total", self.bypass),
            ("image_towers_run_total", self.runs),
            ("image_towers_skipped_total", self.skipped),
            ("image_pieces_reloaded_total", self.img_hits),
            ("image_pieces_computed_total", self.img_runs),
            ("prefill_rows_total", self.rows),
            ("image_prefill_rows_total", self.img_rows),
        ] {
            s.push_str(&format!(
                "# TYPE saber_fn125_{n} counter\nsaber_fn125_{n} {v}\n"
            ));
        }
        s
    }
}

/// FN126: the per-image tower-run law's counters ([`super::vision::tower_groups`]): tower runs
/// (one a request's encode), how many were SPLIT into groups (the request's images did not fit
/// the arena together) and the group runs those took. Both ranks count the same runs.
#[derive(Debug, Default, Clone, Copy)]
struct Img126 {
    runs: u64,
    split: u64,
    group_runs: u64,
}

impl Img126 {
    /// One tower run whose encode moved the tower's split / group-run counters by `ds` / `dg`.
    fn note(&mut self, ds: u64, dg: u64) {
        self.runs += 1;
        self.split += ds;
        self.group_runs += dg;
    }

    fn line(&self, rank: u32) -> String {
        format!(
            "fnimg126 rank{rank}: tower runs {} | split into per-image groups {} ({} group runs) \
             -- only the largest single image must fit the arena",
            self.runs, self.split, self.group_runs
        )
    }

    /// `/metrics` (with FN125's families).
    fn render(&self) -> String {
        let mut s = String::new();
        for (n, v) in [
            ("tower_runs_total", self.runs),
            ("tower_split_runs_total", self.split),
            ("tower_group_runs_total", self.group_runs),
        ] {
            s.push_str(&format!(
                "# TYPE saber_fn126_{n} counter\nsaber_fn126_{n} {v}\n"
            ));
        }
        s
    }
}

/// The serve plane's replicated engine state (both ranks apply the same [`Cmd`]s to it).
struct SrvState {
    /// LR-N (`--spec-couple on`): the seed each UNSEEDED sampled request drew at its first step
    /// (one key law a request, so its drafts know the noise its rows will draw with).
    couple_seeds: BTreeMap<SeqId, u64>,
    /// FN55b: rank 0's pixel rows of the next `Cmd::Encode` (f32 LE bytes), and the last encode's
    /// answer (the tower run's epoch, or why it was refused).
    pix: Vec<u8>,
    enc: Result<u64, String>,
    /// FN125 (`--fn-image-cache on`): the last encode's DEFERRED ticket (0 = the tower ran), the
    /// pending pixels both ranks hold until a computed chunk needs the tower (ticket, grids,
    /// pixels), the ticket counter and the round's counters.
    enc_defer: u64,
    tower_px: Option<(u64, Vec<[u32; 3]>, Vec<f32>)>,
    /// FN125: the last deferred run (ticket, the tower run it made).
    tower_ran: Option<(u64, u64)>,
    tickets: u64,
    i125: Img125,
    /// FN126: the per-image tower-run counters.
    i126: Img126,
    /// Per pool slot: the sequence in it (ready ones are slots `0..ready`).
    slots: Vec<Option<Live>>,
    ready: usize,
    /// The last pick's logits and rows (the dense fallback reads them).
    last_lg: u64,
    last_rows: usize,
    moves: u64,
    /// FN53b: rank 0's pressure valve (`--fn-mem-valve on`), its clock and last sample.
    valve: Option<(saber_mem::mem_valve::MemValve, Instant, Instant)>,
    /// FN53c: the engine's pids and their summed RSS at the boot sample (GiB).
    valve_own: (Vec<u32>, f64),
    /// FN16: the MTP width law (`None` at SPEC 0), the verify step awaiting its commit, the last
    /// prompt chunk awaiting its first drafter pass, the counters on `/metrics`.
    law: Option<verify::MtpLaw>,
    pend: Option<Pend>,
    last_chunk: Option<(Vec<u32>, u32, u32)>,
    ms: MtpStats,
    /// FN17: mixed steps run and the chunks they carried.
    mixed_steps: u64,
    mixed_chunks: u64,
    /// FN45c2: the prompt cache (`--fn-kv-pcache-gib`; None = off).
    pc: Option<pcache::Pcache>,
    /// FN45c2: `--fn-chunk` (the cuts a reload replays).
    fn_chunk: u32,
    /// FN69: `--spec-lookup` (None = off; MTP only).
    lookup: Option<saber_engine::LookupConfig>,
    /// FN75: the engine's half of the served-step ledger, since the last step job's end.
    f75: super::stepledger::EngRec,
    /// FN119 (`--fn-device-step`): the form, its counters (rank 0), the check form's device half.
    ds: super::dstep::DeviceStep,
    dsc: super::dstep::Counts,
    dchk: Option<super::dstep::Check>,
    /// FN119d (`next`): the round a commit left pending (its ids not read yet), both ranks.
    rpend: Option<super::dstep::RoundPend>,
}

/// FN16: a verify step between its rows and its commit (both ranks).
struct Pend {
    w: u32,
    /// Per ready slot: the step's first position, the rows' argmaxes, the drafts fed (padded).
    pos0: Vec<u32>,
    am: Vec<Vec<u32>>,
    drafts: Vec<Vec<u32>>,
    /// FN119d: this step was issued early (its drafts read back under it).
    early: bool,
}

/// FN16: the serve plane's MTP counters (`saber_fn_mtp_*` on `/metrics`).
#[derive(Debug, Default)]
struct MtpStats {
    /// Decode steps by verify width.
    steps: BTreeMap<u32, u64>,
    /// Sequence-steps and the tokens they committed (every MTP-mode decode step, width 1 included).
    seq_steps: u64,
    committed: u64,
    /// Drafts verified / accepted, and accepted by draft position (1-based).
    drafted: u64,
    accepted: u64,
    per_pos: [u64; 4],
    /// FN92: verifies that EVALUATED each draft position (accepted + the first rejected).
    eval_pos: [u64; 4],
    /// FN92: the adaptive depth controller (`--fn-mtp-depth adaptive`; rank 0 decides, rank 1
    /// follows the commits' `kn`) and rank 0's iteration clock (the last b1 step's start, its
    /// verify ms, its round ms).
    depth: Option<super::depth::DepthCtl>,
    it: Option<(Instant, f64, f64)>,
    /// Drafts padded (a sequence with fewer drafts than the step's width: its first step, or a
    /// width that grew).
    padded: u64,
    /// Drafter rounds (after a commit, or a prompt's first pass), passes, device and host ms.
    rounds: u64,
    firsts: u64,
    passes: u64,
    dev_ms: f64,
    host_ms: f64,
    /// FN69: the prompt-lookup source's counters (all zero when off).
    lk: saber_engine::LookupCounts,
}

impl MtpStats {
    fn render(&self) -> String {
        let mut o = String::new();
        o.push_str("# HELP saber_fn_mtp_steps_total Flash-Next decode steps by verify width (FN16).\n# TYPE saber_fn_mtp_steps_total counter\n");
        for (w, n) in &self.steps {
            o.push_str(&format!("saber_fn_mtp_steps_total{{width=\"{w}\"}} {n}\n"));
        }
        for (name, v, help) in [
            (
                "seq_steps",
                self.seq_steps as f64,
                "sequence-steps of MTP-mode decode",
            ),
            (
                "committed_tokens",
                self.committed as f64,
                "tokens those sequence-steps committed",
            ),
            ("draft_tokens", self.drafted as f64, "drafts verified"),
            ("accepted_tokens", self.accepted as f64, "drafts accepted"),
            (
                "padded_drafts",
                self.padded as f64,
                "draft rows padded (no proposal)",
            ),
            (
                "drafter_rounds",
                self.rounds as f64,
                "drafter rounds after a commit",
            ),
            (
                "drafter_first_passes",
                self.firsts as f64,
                "drafter first passes after a prompt",
            ),
            ("drafter_passes", self.passes as f64, "drafter passes"),
            (
                "drafter_device_ms",
                self.dev_ms,
                "drafter device ms (events)",
            ),
            (
                "drafter_host_ms",
                self.host_ms,
                "drafter wall ms on the engine thread",
            ),
        ] {
            o.push_str(&format!(
                "# HELP saber_fn_mtp_{name}_total Flash-Next MTP: {help}.\n# TYPE saber_fn_mtp_{name}_total counter\nsaber_fn_mtp_{name}_total {v}\n"
            ));
        }
        let l = &self.lk;
        for (name, v, help) in [
            ("lookups", l.lookups, "drafting points queried"),
            ("hits", l.hits, "queries that found an earlier occurrence"),
            (
                "used",
                l.used,
                "drafting points the merge law gave the lookup",
            ),
            (
                "drafted",
                l.lookup_drafted,
                "draft rows verified from the lookup",
            ),
            (
                "accepted",
                l.lookup_accepted,
                "drafts accepted from the lookup",
            ),
            (
                "drafter_drafted",
                l.drafter_drafted,
                "draft rows verified from the MTP chain",
            ),
            (
                "drafter_accepted",
                l.drafter_accepted,
                "drafts accepted from the MTP chain",
            ),
        ] {
            o.push_str(&format!(
                "# HELP saber_fn_lookup_{name}_total Flash-Next prompt lookup (FN69): {help}.\n# TYPE saber_fn_lookup_{name}_total counter\nsaber_fn_lookup_{name}_total {v}\n"
            ));
        }
        o.push_str("# HELP saber_fn_mtp_accepted_per_pos_total Drafts accepted by draft position.\n# TYPE saber_fn_mtp_accepted_per_pos_total counter\n");
        for (i, n) in self.per_pos.iter().enumerate() {
            o.push_str(&format!(
                "saber_fn_mtp_accepted_per_pos_total{{pos=\"{}\"}} {n}\n",
                i + 1
            ));
        }
        o.push_str("# HELP saber_fn_mtp_evaluated_per_pos_total Verifies that evaluated each draft position (FN92).\n# TYPE saber_fn_mtp_evaluated_per_pos_total counter\n");
        for (i, n) in self.eval_pos.iter().enumerate() {
            o.push_str(&format!(
                "saber_fn_mtp_evaluated_per_pos_total{{pos=\"{}\"}} {n}\n",
                i + 1
            ));
        }
        if let Some(c) = &self.depth {
            o.push_str("# HELP saber_fn_mtp_depth_decisions_total The b1 adaptive depth's decisions by path (FN92, which_path).\n# TYPE saber_fn_mtp_depth_decisions_total counter\n");
            for (p, n) in super::depth::Path::ALL.iter().zip(c.paths) {
                o.push_str(&format!(
                    "saber_fn_mtp_depth_decisions_total{{path=\"{}\"}} {n}\n",
                    p.name()
                ));
            }
            o.push_str("# HELP saber_fn_mtp_depth_steps_total b1 decisions by depth (FN92).\n# TYPE saber_fn_mtp_depth_steps_total counter\n");
            for (i, n) in c.at.iter().enumerate() {
                o.push_str(&format!(
                    "saber_fn_mtp_depth_steps_total{{depth=\"{}\"}} {n}\n",
                    i + 1
                ));
            }
        }
        o
    }
}

/// FN16: `d` fitted to `k` drafts: truncated, or padded with its last draft (or `fill`) -- a pad
/// row is verified like any draft (accepted only if it IS the row's decision).
fn fit_drafts(d: &[u32], k: usize, fill: u32) -> (Vec<u32>, u64) {
    let mut v: Vec<u32> = d.iter().copied().take(k).collect();
    let pad = k.saturating_sub(v.len());
    let f = v.last().copied().unwrap_or(fill);
    v.resize(k, f);
    (v, pad as u64)
}

/// FN16: the chain length the NEXT step of the ready set drafts (the law at the ready set's
/// bucket and EMAs, minus the committed row).
fn next_k(st: &SrvState) -> u32 {
    let Some(law) = &st.law else {
        return 0;
    };
    let n = st.ready.max(1);
    if let (true, Some(c)) = (law.adaptive && bucket_for(n) == 1, &st.ms.depth) {
        // FN92: the depth both ranks hold (rank 0's last decision, rank 1's last commit's `kn`)
        return c.depth() as u32;
    }
    let emas: Vec<verify::WidthEma> = (0..st.ready)
        .filter_map(|i| st.slots[i].as_ref().map(|l| l.ema))
        .collect();
    law.width(bucket_for(n), &emas).saturating_sub(1)
}

/// FN16: after a prompt chunk (both ranks): the drafter's row at the previous chunk's last position
/// (its stashed target stream + this chunk's first token), then -- a non-last chunk -- the rows
/// whose next token is in this chunk, the last row's stream stashed for the next chunk; the last
/// chunk's rows wait for the sampled first token (`Cmd::First`). Returns the last chunk's
/// (ids, first position, chunk index).
fn mtp_after_chunk(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    lv: &mut Live,
    ids: &[u32],
    last: bool,
) -> anyhow::Result<Option<(Vec<u32>, u32, u32)>> {
    let m = ids.len() as u32;
    ensure!(
        m >= 1 && lv.len >= m && lv.chunks >= 1,
        "a chunk precedes its drafter rows"
    );
    let (p0, chunk) = (lv.len - m, lv.chunks - 1);
    if last {
        // everything waits for the sampled token (the prefill's logits stay intact for the
        // sampler's dense fallback): the stash row and the chunk's rows run in `Cmd::First`
        return Ok(Some((ids.to_vec(), p0, chunk)));
    }
    mtp_stash_row(rk, em, recs, lv, ids[0], p0, chunk)?;
    let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
    if m >= 2 {
        mtp_rows(
            rk,
            em,
            recs,
            lv.slot,
            &ids[1..],
            p0,
            chunk,
            hid,
            false,
            None,
        )?;
    }
    let mt = rk.mtp.as_ref().context("mtp")?;
    lv.stash = d2h_in(
        &rk.dev,
        &mt.hid.own,
        hid + u64::from(m - 1) * W * 2,
        (W * 2) as usize,
    )?;
    Ok(None)
}

/// FN16: the drafter's row at a chunk boundary -- the previous chunk's last position `p0 - 1`, its
/// stashed target stream + this chunk's first token `tok` (nothing without a stash).
fn mtp_stash_row(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    lv: &mut Live,
    tok: u32,
    p0: u32,
    chunk: u32,
) -> anyhow::Result<()> {
    if lv.stash.is_empty() {
        return Ok(());
    }
    ensure!(p0 >= 1 && chunk >= 1, "a stash without a previous chunk");
    let sa = {
        let mt = rk.mtp.as_ref().context("mtp")?;
        rk.hip
            .memcpy_h2d_blocking(&mt.stash, &lv.stash, &rk.dev.stream)?;
        mt.stash.addr().as_u64()
    };
    mtp_rows(
        rk,
        em,
        recs,
        lv.slot,
        &[tok],
        p0 - 1,
        chunk - 1,
        sa,
        false,
        None,
    )?;
    lv.stash.clear();
    Ok(())
}

/// The spare pool slot (a swap's third place).
const SPARE: u32 = MAX_SEQS;

/// FN45c2 / FN45c3: one prompt chunk through the PROMPT / CONVERSATION CACHE. (1) While BOTH
/// ranks hold a stored prefix of `lv` + `ids` (the longest first, `pcache::probe`) it is
/// RELOADED (`--fn-kv-pcache-verify`: COMPUTED at the very same cut instead -- the byte law's
/// reference); a whole non-last chunk may vanish that way. (2) The last chunk of a chat prompt is
/// cut at its last `<|im_start|>` (the generation prompt's start, `Pcache::turn_cut`): the part
/// before it runs as a non-last chunk and is STORED -- the prefix the conversation's next turn
/// repeats -- and the few tokens after it run as the last chunk. (3) Every non-last chunk is
/// stored. Without the cache = FN44's single `run_chunk`.
#[allow(clippy::too_many_arguments)]
fn prefill_cached(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    f: &FnFlags,
    st: &mut SrvState,
    lv: &mut Live,
    ids: &[u32],
    req: &RowReq,
    last: bool,
) -> anyhow::Result<RowOut> {
    let mut ids = ids;
    if st.pc.is_none() {
        return run_chunk(rk, em, ple, ph, recs, f, st, lv, ids, req, last);
    }
    if lv
        .img
        .as_ref()
        .is_some_and(|im| im.keys.is_empty() || !super::run::image_cache_on(f))
    {
        // FN55b (FN125's control, `--fn-image-cache off`, and any image prompt whose facts carry
        // no content hashes): a key over the token ids alone would let two images share their
        // `<|image_pad|>` ids -- computed, never stored
        if lv.len == 0 {
            st.i125.bypass += 1;
        }
        return run_chunk_as(rk, em, ple, ph, recs, f, st, lv, ids, req, last, false);
    }
    // FN125: an image prompt with its content hashes meets the tree / the cache like text: every
    // key folds the hashes of the image runs its piece overlaps (`pcache::piece_key`), and the
    // tower runs only before a COMPUTED chunk over an image run (`tower_deferred`)
    if st.pc.as_ref().is_some_and(|pc| pc.tree.is_some()) {
        return prefill_tree(rk, em, ple, ph, recs, f, st, lv, ids, req, last);
    }
    loop {
        let pc = st.pc.as_mut().context("pc")?;
        let Some((t, k)) = pcache::probe(rk, pc, lv, ids, last)? else {
            break;
        };
        let n = (t - lv.len) as usize;
        if pc.verify {
            // the reference: the same cut, computed (and not stored again)
            pc.verify_runs += 1;
            run_chunk_as(
                rk,
                em,
                ple,
                ph,
                recs,
                f,
                st,
                lv,
                &ids[..n],
                req,
                false,
                false,
            )?;
            lv.key = k;
        } else if pcache::reload(rk, pc, lv, &ids[..n], k, st.fn_chunk)? {
            if lv
                .img
                .as_ref()
                .is_some_and(|im| im.keys.iter().any(|x| x.overlaps(t - n as u32, t)))
            {
                st.i125.img_hits += 1;
            }
            if rk.mtp.is_some() {
                st.last_chunk = None;
            }
        } else {
            break;
        }
        ids = &ids[n..];
        if ids.is_empty() {
            return Ok(RowOut::default());
        }
    }
    if last {
        if let Some(c) = st.pc.as_ref().and_then(|pc| pc.turn_cut(lv.len, ids)) {
            run_chunk(rk, em, ple, ph, recs, f, st, lv, &ids[..c], req, false)?;
            st.pc.as_mut().context("pc")?.turn_cuts += 1;
            ids = &ids[c..];
        } else if let Some(c) = st.pc.as_ref().and_then(|pc| pc.warm_cut(ids)) {
            // FN66: a warm prompt -- its head is stored (waiting for a busy writer)
            st.pc.as_mut().context("pc")?.force_store = true;
            run_chunk(rk, em, ple, ph, recs, f, st, lv, &ids[..c], req, false)?;
            let pc = st.pc.as_mut().context("pc")?;
            pc.warm_cuts += 1;
            pc.force_store = false;
            ids = &ids[c..];
        }
    }
    run_chunk(rk, em, ple, ph, recs, f, st, lv, ids, req, last)
}

/// LR-D: a prompt's last piece ran -- its walk into the tree's counters and (rank 0) one
/// `fnptreereq` line: reloaded vs computed, the first miss and its cut kind, the S node's key.
fn pt_req_end(rk: &Rank<'_, '_>, st: &mut SrvState, lv: &Live, end: u32) {
    let Some(tr) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) else {
        return;
    };
    let r = lv.pt_req;
    tr.c.prompt_tokens += u64::from(end);
    tr.c.computed_rows += u64::from(r.rows);
    let new = r.sys.is_some_and(|(_, k)| tr.c.note_sys(&k));
    if rk.rank == 0 {
        let (_, used, all) = tr.ram.usage();
        println!(
            "{}",
            pcache::req_line(rk.rank, end, &r, new, (used, all, tr.ram.len()))
        );
    }
}

/// FN113: one prompt chunk through the TIERED PREFIX TREE (FLASHNEXT-PLAN §16). The cut law
/// splits the chunk into PIECES at the ids' own cuts (prefix-stable: a cold prefill and a hit cut
/// alike); every piece but the prompt's final one is RELOADED when both ranks hold its node (RAM,
/// else drive), else PREFILLED and STORED as a node; the final piece runs (its logits pick the
/// first token) and, when the prompt ends in a generation prompt the next turn will cut at, marks
/// the END node `Cmd::First` stores (SPEC 0: stored after the piece).
#[allow(clippy::too_many_arguments)]
fn prefill_tree(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    f: &FnFlags,
    st: &mut SrvState,
    lv: &mut Live,
    ids: &[u32],
    req: &RowReq,
    last: bool,
) -> anyhow::Result<RowOut> {
    use saber_mem::prefix_tree::{CutKind, CutLaw};
    let p0 = lv.len;
    let end = p0 + ids.len() as u32;
    let cuts = {
        let tr = st
            .pc
            .as_mut()
            .and_then(|pc| pc.tree.as_mut())
            .context("tree")?;
        if p0 == 0 {
            // a prompt admission: the LRFU clock (both ranks alike)
            tr.ram.tick();
            tr.c.prompts += 1;
            lv.law = saber_mem::prefix_tree::LawState::default();
            lv.pt_hit = None;
            lv.pt_partial = false;
            lv.pt_req = pcache::PtReq::default();
        }
        let c = tr.law.feed(&mut lv.law, ids);
        tr.c.law_cuts += c.iter().filter(|x| x.0 < end).count() as u64;
        c
    };
    // FN113b: `--fn-prefix-tree-end off` (diagnostic) never marks an END node
    let end_node = last && f.fn_prefix_tree_end != "off" && CutLaw::end_node(&cuts, &lv.law);
    if !last {
        CutLaw::chunk_end(&mut lv.law);
    }
    // the pieces: (start, end, the kind of the cut at its end)
    let mut bounds: Vec<(u32, CutKind)> = cuts.iter().copied().filter(|c| c.0 < end).collect();
    bounds.push((
        end,
        cuts.last()
            .filter(|c| c.0 == end)
            .map_or(CutKind::Chunk, |c| c.1),
    ));
    let mut out = RowOut::default();
    let mut a = p0;
    for (b, kind) in bounds {
        let pids = &ids[(a - p0) as usize..(b - p0) as usize];
        let fin = last && b == end;
        let pc = st.pc.as_mut().context("pc")?;
        let key = pc.key(lv, pids);
        // LR-D: the S node's key (a system prompt that changes every request misses here)
        if kind == CutKind::Sys && !fin {
            lv.pt_req.sys = Some((b, key));
        }
        if !fin {
            if pcache::probe_piece(rk, pc, lv, pids.len() as u32, &key)? {
                if pc.verify {
                    // the byte law's reference: the same cut, computed (and not stored again)
                    pc.verify_runs += 1;
                    run_chunk_as(rk, em, ple, ph, recs, f, st, lv, pids, req, false, false)?;
                    lv.key = key;
                    lv.pt_hit = Some(kind);
                    a = b;
                    continue;
                }
                if pcache::tree_reload(rk, pc, lv, pids, key, st.fn_chunk)? {
                    if lv
                        .img
                        .as_ref()
                        .is_some_and(|im| im.keys.iter().any(|k| k.overlaps(a, b)))
                    {
                        st.i125.img_hits += 1;
                    }
                    // FN113 (h2's turn-class break): an END node holds the state as `Ready` left
                    // it (GDN on side 0, PLE on the decode parity); the next prefill chunk reads
                    // the side of its index's parity -- move it there (a sys / chunk node is
                    // already on it: stored right after its piece's prefill)
                    let want = (lv.chunks % 2) as usize;
                    if lv.gdn_at != want || lv.ple_at != want {
                        rk.pools
                            .to_side(rk.hip, &rk.dev, lv.slot, lv.gdn_at, lv.ple_at, want)?;
                        lv.gdn_at = want;
                        lv.ple_at = want;
                    }
                    if rk.mtp.is_some() {
                        st.last_chunk = None;
                    }
                    lv.pt_hit = Some(kind);
                    lv.pt_req.hit += b - a;
                    a = b;
                    continue;
                }
            }
            if lv.pt_hit.is_some() {
                lv.pt_partial = true;
            }
            lv.pt_req.rows += b - a;
            if lv.pt_req.first_miss.is_none() {
                lv.pt_req.first_miss = Some((a, b - a, kind));
            }
            if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
                t.c.pieces_run += 1;
            }
            run_chunk_as(rk, em, ple, ph, recs, f, st, lv, pids, req, false, true)?;
        } else {
            if end_node {
                lv.pt_end = Some((a, key));
            }
            if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
                t.c.pieces_run += 1;
                match (lv.pt_hit, lv.pt_partial) {
                    (None, _) => t.c.cls_miss += 1,
                    (Some(_), true) => t.c.cls_partial += 1,
                    (Some(CutKind::Turn), false) => t.c.cls_turn += 1,
                    (Some(CutKind::Sys), false) => t.c.cls_sys += 1,
                    (Some(CutKind::Chunk), false) => t.c.cls_chunk += 1,
                }
            }
            out = run_chunk(rk, em, ple, ph, recs, f, st, lv, pids, req, true)?;
            lv.pt_req.rows += b - a;
            pt_req_end(rk, st, lv, end);
            if rk.mtp.is_none() {
                // SPEC 0: no `Cmd::First` -- the END node now (the stash is empty)
                end_store(rk, st, lv)?;
            }
        }
        a = b;
    }
    Ok(out)
}

/// FN113b: `Cmd::First`'s END block after the drafter's m - 1 row pass was ISSUED (`te` = its
/// start): under `--fn-prefix-tree-end side` the node's pages + the victim's demotion on the host
/// while the pass runs, the stash read (the step's stream drained), the node's copies off the step
/// ([`pcache::end_issue`]); else (or when the RAM tier refuses it) FN113's store on the step.
/// Returns the ms to the drained stash (the pass, or the host work when longer).
#[allow(clippy::too_many_arguments)]
fn end_block(
    rk: &mut Rank<'_, '_>,
    st: &mut SrvState,
    f: &FnFlags,
    lv: &mut Live,
    hid: u64,
    m: u64,
    te: Instant,
) -> anyhow::Result<f64> {
    let prep = match (lv.pt_end, st.pc.as_mut()) {
        (Some((e0, key)), Some(pc)) if f.fn_prefix_tree_end == "side" => {
            Some(pcache::end_prepare(rk, pc, lv, e0, key)?)
        }
        _ => None,
    };
    lv.stash = d2h_in(
        &rk.dev,
        &rk.mtp.as_ref().context("mtp")?.hid.own,
        hid + (m - 1) * W * 2,
        (W * 2) as usize,
    )?;
    let mtp_ms = te.elapsed().as_secs_f64() * 1e3;
    match prep {
        Some(pcache::EndPrep::Plan(plan)) => {
            let (_, key) = lv.pt_end.take().context("the END node")?;
            let pc = st.pc.as_mut().context("pc")?;
            // the drafter part lands before the m-row pass rewrites it; the target part stays in
            // flight under that pass (gated by its word at the end of `Cmd::First`)
            let t0 = Instant::now();
            pcache::end_copy(rk, pc, lv.slot, &plan, true, false)?;
            pcache::end_wait(rk, pc, false)?;
            pcache::end_copy(rk, pc, lv.slot, &plan, false, false)?;
            pcache::end_mark(rk, pc, key)?;
            pcache::end_commit(pc, lv, *plan, t0)?;
            lv.key = key;
            if let Some(t) = pc.tree.as_mut() {
                t.c.end_nodes += 1;
            }
        }
        Some(pcache::EndPrep::Held) => {
            let (_, key) = lv.pt_end.take().context("the END node")?;
            lv.key = key;
            if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
                t.c.end_nodes += 1;
            }
        }
        // FN113's store on the step (`step`, or the RAM tier refused the node)
        Some(pcache::EndPrep::Refused) | None => end_store(rk, st, lv)?,
    }
    Ok(mtp_ms)
}

/// FN113b (`--fn-prefix-tree-end live`), while the prompt's m-row drafter pass is on the card:
/// the END node's pages + the victim's demotion (host), then its TARGET part's copies in flight
/// on the mover's stream (the pass writes only drafter bytes).
fn live_begin(rk: &Rank<'_, '_>, st: &mut SrvState, lv: &Live) -> anyhow::Result<pcache::EndPrep> {
    let t0 = Instant::now();
    let (e0, key) = lv.pt_end.context("the END node")?;
    let pc = st.pc.as_mut().context("pc")?;
    let p = pcache::end_prepare(rk, pc, lv, e0, key)?;
    if let pcache::EndPrep::Plan(plan) = &p {
        pcache::end_copy(rk, pc, lv.slot, plan, false, false)?;
        pcache::end_mark(rk, pc, key)?;
    }
    if let Some(t) = pc.tree.as_mut() {
        t.c.end_ms += t0.elapsed().as_secs_f64() * 1e3;
    }
    Ok(p)
}

/// FN113b live, after the m-row pass (its draft read drained the step's stream): the DRAFTER
/// part's copies ON THE IDLE STEP STREAM (small; never queued behind the target part on the
/// mover) -- the rows as the prompt's own pass left them; NO host wait (the chain that writes
/// the MTP layer next is behind them on the same stream); the node published PENDING until its
/// gate's event completes; a refused node takes FN113's store on the step.
fn live_end(
    rk: &mut Rank<'_, '_>,
    st: &mut SrvState,
    lv: &mut Live,
    p: pcache::EndPrep,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    match p {
        pcache::EndPrep::Plan(plan) => {
            let (_, key) = lv.pt_end.take().context("the END node")?;
            let pc = st.pc.as_mut().context("pc")?;
            pcache::end_copy(rk, pc, lv.slot, &plan, true, true)?;
            pcache::end_commit(pc, lv, *plan, t0)?;
            lv.key = key;
            if let Some(t) = pc.tree.as_mut() {
                t.c.end_nodes += 1;
            }
        }
        pcache::EndPrep::Held => {
            let (_, key) = lv.pt_end.take().context("the END node")?;
            lv.key = key;
            if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
                t.c.end_nodes += 1;
            }
        }
        pcache::EndPrep::Refused => end_store(rk, st, lv)?,
    }
    if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        t.c.end_ms += ms;
        t.c.end_worst_ms = t.c.end_worst_ms.max(ms);
    }
    Ok(())
}

/// FN119g: the K-side front under the chunk's head.
pub(super) enum FrontUnder {
    /// The last piece will issue it (its rows >= 2).
    Want,
    /// Issued on the side stream: its end event (the first round joins it) and the arena floor
    /// it started at (the main stream's takes stay below it until the join).
    Issued(saber_hip::Event),
}

/// FN119f: the first round queued behind a prompt's last chunk (both ranks): the END node's
/// placement (its target part on the mover and its drafter part on the stream already issued),
/// whether the node's stash was read back, the round's chain length, the piece's rows.
pub(super) struct FirstDev {
    prep: Option<pcache::EndPrep>,
    node: bool,
    kn: u32,
}

/// FN119f: [`live_end`]'s host half -- the node published from `lv` as `Cmd::First` leaves it (its
/// stash read back), its drafter part's copies already on the stream between the round's pass 1
/// and its chain.
fn live_commit(
    rk: &mut Rank<'_, '_>,
    st: &mut SrvState,
    lv: &mut Live,
    p: pcache::EndPrep,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    match p {
        pcache::EndPrep::Plan(plan) => {
            let (_, key) = lv.pt_end.take().context("the END node")?;
            let pc = st.pc.as_mut().context("pc")?;
            pcache::end_commit(pc, lv, *plan, t0)?;
            lv.key = key;
            if let Some(t) = pc.tree.as_mut() {
                t.c.end_nodes += 1;
            }
        }
        pcache::EndPrep::Held => {
            let (_, key) = lv.pt_end.take().context("the END node")?;
            lv.key = key;
            if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
                t.c.end_nodes += 1;
            }
        }
        pcache::EndPrep::Refused => end_store(rk, st, lv)?,
    }
    if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        t.c.end_ms += ms;
        t.c.end_worst_ms = t.c.end_worst_ms.max(ms);
    }
    Ok(())
}

/// FN119f: behind a prompt's last chunk (both ranks, before the host waits for it): the pick's
/// reads and the route log queued (FN112's staging + the early buffer), the state normalized to
/// the decode parity (`Cmd::Ready`'s copies, now behind the chunk), the first token's WALK (round
/// 2's, width 1), the K-side front (the previous piece's stash row too), the END node's
/// placement + target part (mover) and its stash read back async, the last row's stream to row 0,
/// the first round with the node's drafter part between its pass 1 and its chain, its ids read
/// back async. Nothing here waits on the host.
#[allow(clippy::too_many_arguments)]
fn first_dev_issue(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    stc: &std::cell::RefCell<&mut SrvState>,
    lv: &mut Live,
    ids: &[u32],
    req: &RowReq,
    ah: &super::dstep::AheadCmd,
    dp: usize,
    lg: u64,
) -> anyhow::Result<()> {
    let m = ids.len() as u32;
    let (p0, chunk, slot) = (lv.len, lv.chunks, lv.slot);
    // 1. the pick's reads + the route log right behind the chunk
    ov112_queue(rk, lg, 1, std::slice::from_ref(req), None)?;
    plog_queue(rk, m)?;
    // 2. the state at the decode parity now (Ready then finds it there)
    let side = ((chunk + 1) % 2) as usize;
    rk.pools.normalize(rk.hip, &rk.dev, slot, side, side, dp)?;
    rk.first_norm = Some(dp);
    // the live state as Ready leaves it (the END node's placement reads it)
    let mut post = lv.clone();
    post.len = p0 + m;
    post.chunks = chunk + 1;
    post.gdn_at = 0;
    post.ple_at = dp;
    let node = lv.pt_end.is_some();
    let prev_stash = std::mem::take(&mut lv.stash);
    let prep: std::cell::RefCell<Option<pcache::EndPrep>> = std::cell::RefCell::new(None);
    let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
    let mut pre = |rk: &mut Rank<'_, '_>, em: &mut Emitter<'_>| -> anyhow::Result<()> {
        // FN119g: the fronts ran under the head on the side stream: join them
        let joined = front_under_join(rk, em)?;
        // the previous piece's stash row (its K-side front: every byte a later pass reads of it)
        let mut base = 0usize;
        if !joined && !prev_stash.is_empty() {
            ensure!(p0 >= 1 && chunk >= 1, "a stash without a previous chunk");
            let sa = {
                let mt = rk.mtp.as_ref().context("mtp")?;
                let e = rk
                    .ahead
                    .as_mut()
                    .and_then(|a| a.early.as_mut())
                    .context("FN119f: the early buffers")?;
                e.stin.as_mut_slice()[..prev_stash.len()].copy_from_slice(&prev_stash);
                rk.hip.memcpy_h2d_at(
                    &mt.stash,
                    0,
                    e.stin
                        .slice(0, prev_stash.len())
                        .context("FN119f: the stash row")?,
                    &rk.dev.stream,
                )?;
                mt.stash.addr().as_u64()
            };
            let (cc, nb) = front_stage_at(rk, recs, slot, &ids[..1], p0 - 1, chunk - 1, base)?;
            base = nb;
            front_issue_at(rk, em, slot, 1, cc, sa)?;
        }
        if m >= 2 && !joined {
            let (cc, _) = front_stage_at(rk, recs, slot, &ids[1..], p0, chunk, base)?;
            front_issue_at(rk, em, slot, m - 1, cc, hid)?;
            rk.front_early += 1;
        }
        if node {
            // the stash (the last prompt row's stream) read back behind the front
            let mt = rk.mtp.as_ref().context("mtp")?;
            let src = mt
                .hid
                .addr()
                .offset(u64::from(m - 1) * W * 2)
                .context("FN119f: the stash row")?;
            let e = rk
                .ahead
                .as_mut()
                .and_then(|a| a.early.as_mut())
                .context("FN119f: the early buffers")?;
            rk.hip.memcpy_d2h_addr(
                e.stash
                    .slice_mut(0, (W * 2) as usize)
                    .context("FN119f: the stash read-back")?,
                src,
                &rk.dev.stream,
            )?;
            rk.hip.record_event(&e.stash_ev, &rk.dev.stream)?;
            // the node's pages + its TARGET part on the mover, after the chunk + the normalize
            rk.tier
                .mover
                .after(&rk.dev.stream)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut stb = stc.borrow_mut();
            *prep.borrow_mut() = Some(live_begin(rk, &mut stb, &post)?);
            if let Some(a) = rk.ahead.as_mut() {
                a.c.dev_end += 1;
            }
        }
        if m >= 2 {
            let mt = rk.mtp.as_ref().context("mtp")?;
            let (own, off) = (mt.hid.own.clone(), mt.hid.off);
            rk.hip.run_batch(&[Job::MemcpyD2d {
                dst: &own,
                dst_offset: off,
                src: &own,
                src_offset: off + u64::from(m - 1) * W * 2,
                bytes: W * 2,
                stream: &rk.dev.stream,
            }])?;
        }
        Ok(())
    };
    let mut mid = |rk: &mut Rank<'_, '_>| -> anyhow::Result<()> {
        // the node's DRAFTER part, on the step's stream between pass 1 and the chain
        // (a node the RAM tier refused: `Cmd::First` re-runs the round and stores it at its mid)
        if let Some(pcache::EndPrep::Plan(plan)) = prep.borrow().as_ref() {
            let mut stb = stc.borrow_mut();
            let pc = stb.pc.as_mut().context("pc")?;
            pcache::end_copy(rk, pc, slot, plan, true, true)?;
        }
        Ok(())
    };
    ahead_issue(
        rk,
        em,
        recs,
        ah,
        &[p0 + m - 1],
        &[Vec::new()],
        1,
        1,
        lg,
        Some(&mut pre),
        Some(&mut mid),
    )?;
    let a = rk.ahead.as_ref().context("FN119: the ahead")?;
    ensure!(
        a.issued.is_some_and(|i| i.round && i.kn == ah.kn),
        "FN119f: the first round was not issued behind the walk"
    );
    queue_ids(rk, ah.kn)?;
    if let Some(a) = rk.ahead.as_mut() {
        a.c.dev_first += 1;
        a.issued = None;
    }
    rk.first_dev = Some(FirstDev {
        prep: prep.into_inner(),
        node,
        kn: ah.kn,
    });
    Ok(())
}

/// FN119f: the chunk's route log (`m` rows) read back behind it into the early pinned buffer (its
/// event); the deferred tier end reads it there.
fn plog_queue(rk: &mut Rank<'_, '_>, m: u32) -> anyhow::Result<()> {
    let n = (LAYERS * m * 40) as usize;
    let src = rk.idlog.addr();
    let e = rk
        .ahead
        .as_mut()
        .and_then(|a| a.early.as_mut())
        .context("FN119f: the early buffers")?;
    ensure!(
        n <= e.plog.len(),
        "FN119f: a {m}-row log past its read-back"
    );
    rk.hip.memcpy_d2h_addr(
        e.plog
            .slice_mut(0, n)
            .context("FN119f: the log read-back")?,
        src,
        &rk.dev.stream,
    )?;
    rk.hip.record_event(&e.plog_ev, &rk.dev.stream)?;
    e.plog_m = Some(m);
    Ok(())
}

/// FN113: store the END node `lv.pt_end` (the post-prefill state of a prompt the next turn
/// extends) -- from `Cmd::First` after the drafter's rows (MTP), or after the final piece (SPEC 0).
fn end_store(rk: &Rank<'_, '_>, st: &mut SrvState, lv: &mut Live) -> anyhow::Result<()> {
    let Some((p0, key)) = lv.pt_end.take() else {
        return Ok(());
    };
    let Some(pc) = st.pc.as_mut() else {
        return Ok(());
    };
    pcache::store(rk, pc, lv, p0, key)?;
    lv.key = key;
    if let Some(t) = pc.tree.as_mut() {
        t.c.end_nodes += 1;
    }
    Ok(())
}

/// FN125: before a COMPUTED chunk of `n` tokens of `lv`: when the sequence's tower run was
/// deferred and the chunk's range overlaps an image run, run it now on both ranks (they hold the
/// same pixels since `Cmd::Encode`; both reach this point in the same command, the range being
/// the same on both) -- the lazy tower loads here, never for a prompt whose image runs all
/// reload. The run is the very run the encode would have made (the same grids, the same pixels,
/// the whole request's images), so the rows -- and every byte after them -- are the eager form's.
fn tower_deferred(
    rk: &mut Rank<'_, '_>,
    st: &mut SrvState,
    lv: &mut Live,
    n: u32,
) -> anyhow::Result<()> {
    let Some(im) = lv.img.clone() else {
        return Ok(());
    };
    if !im.tower_due(lv.len, lv.len + n) {
        return Ok(());
    }
    // a sibling sequence of the same request (n > 1) whose prefill already ran this ticket's
    // tower, the arena still holding that run: its rows, no second run
    let epoch = match st.tower_ran {
        Some((t, e)) if t == im.defer && rk.vis.as_ref().is_some_and(|v| v.holds(e)) => e,
        _ => {
            let (t, grids, pixels) = st
                .tower_px
                .as_ref()
                .context("FN125: a deferred tower run with no pending pixels")?;
            ensure!(
                *t == im.defer,
                "FN125: slot {} needs the tower over ticket {} but the pending pixels are ticket \
                 {t} -- another request's images replaced them before this prompt was in",
                lv.slot,
                im.defer
            );
            vision_load(rk)?;
            let v = rk.vis.as_mut().context("an image on a text-only boot")?;
            let (s0, g0) = (v.split, v.group_runs);
            let r = v.encode(rk.hip, &rk.dev.stream, grids, pixels);
            st.i126.note(v.split - s0, v.group_runs - g0);
            let ok = u8::from(r.is_ok());
            let peer_ok = rk.peer.exchange().xchg_bytes("fn.pix.ok", &[ok])?;
            match (r, peer_ok.first().copied()) {
                (Ok(_), Some(1)) => {}
                (Err(e), _) => {
                    bail!("FN125: the deferred tower run refused after its precheck: {e:#}")
                }
                (Ok(_), _) => bail!("FN125: the peer rank's deferred tower run refused"),
            }
            st.tower_ran = Some((im.defer, v.epoch));
            st.i125.runs += 1;
            v.epoch
        }
    };
    let s = std::sync::Arc::new(im.ran(epoch));
    rk.ropes[lv.slot as usize] = Some(std::sync::Arc::clone(&s));
    lv.img = Some(s);
    Ok(())
}

/// One chunk: the prefill, the drafter's rows (MTP), the cache's store after a non-last chunk.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    f: &FnFlags,
    st: &mut SrvState,
    lv: &mut Live,
    ids: &[u32],
    req: &RowReq,
    last: bool,
) -> anyhow::Result<RowOut> {
    run_chunk_as(rk, em, ple, ph, recs, f, st, lv, ids, req, last, true)
}

#[allow(clippy::too_many_arguments)]
fn run_chunk_as(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    f: &FnFlags,
    st: &mut SrvState,
    lv: &mut Live,
    ids: &[u32],
    req: &RowReq,
    last: bool,
    store: bool,
) -> anyhow::Result<RowOut> {
    // FN125: a computed chunk over an image run needs the tower's rows (the deferred run first)
    tower_deferred(rk, st, lv, ids.len() as u32)?;
    if lv.img.as_ref().is_some_and(|im| {
        im.keys
            .iter()
            .any(|k| k.overlaps(lv.len, lv.len + ids.len() as u32))
    }) {
        st.i125.img_runs += 1;
    }
    st.i125.rows += ids.len() as u64;
    if lv.img.is_some() {
        st.i125.img_rows += ids.len() as u64;
    }
    // FN45c2: a non-last chunk's chained key (stored after it runs)
    let key = st.pc.as_ref().filter(|_| !last).map(|pc| pc.key(lv, ids));
    let p0 = lv.len;
    let t75 = Instant::now();
    // FN119e (`--fn-first-step chunk`): the K-side front under the last chunk, when the first
    // round will be the `kv` form at pool slot 0 (alone in the ready set, a chain to run, no image)
    rk.front_want = last
        && super::run::first_step_front(f)
        && st.ds.next()
        && st.ready == 0
        && lv.slot == 0
        && next_k(st) > 1
        && lv.lk.is_none()
        && ids.len() >= 2
        && ids.len() <= MTP_PIECE as usize + 1
        && ids.len() as u32 <= f.max_chunk()
        && rk.mtp.as_ref().is_some_and(|m| m.chain.is_some())
        && rk.ahead.as_ref().is_some_and(|a| a.early.is_some())
        && rk.ov112.is_some()
        && rk.ropes.get(lv.slot as usize).cloned().flatten().is_none();
    rk.front_done = false;
    // FN119f (`--fn-first-step dev`): the first token walked on the card behind this last piece,
    // the front, the END node's placement + drafter part and the first round queued behind it
    let end_live = f.fn_prefix_tree_end == "live";
    let dev = match rk.first_ah.take() {
        Some(x)
            if last
                && super::run::first_step_dev(f)
                && st.ready == 0
                && lv.slot == 0
                && next_k(st) > 1
                && lv.lk.is_none()
                && !ids.is_empty()
                && ids.len() <= MTP_PIECE as usize + 1
                && ids.len() as u32 <= f.max_chunk()
                && rk.mtp.as_ref().is_some_and(|m| m.chain.is_some())
                && rk.ahead.as_ref().is_some_and(|a| a.early.is_some())
                && rk.ov112.is_some()
                && rk.ropes.get(lv.slot as usize).cloned().flatten().is_none()
                && (lv.pt_end.is_none() || end_live) =>
        {
            Some(x)
        }
        _ => None,
    };
    if dev.is_some() {
        rk.front_want = false;
        // FN119g: the front under the chunk's head (rows to front)
        if super::run::first_step_under(f) && ids.len() >= 2 {
            rk.front_under = Some(FrontUnder::Want);
        }
    }
    let r = match dev {
        Some((ah, dp)) => {
            let stc = std::cell::RefCell::new(&mut *st);
            let mut hook = |rk: &mut Rank<'_, '_>,
                            em: &mut Emitter<'_>,
                            lv: &mut Live,
                            lg: u64|
             -> anyhow::Result<()> {
                first_dev_issue(rk, em, recs, &stc, lv, ids, req, &ah, dp, lg)
            };
            prefill_with(
                rk,
                em,
                ple,
                ph,
                recs,
                lv,
                ids,
                f,
                None,
                None,
                Some(req),
                Some(&mut hook),
            )
        }
        None => prefill(rk, em, ple, ph, recs, lv, ids, f, None, None, Some(req)),
    };
    rk.front_want = false;
    if r.is_ok() {
        ensure!(
            rk.front_under.is_none(),
            "FN119g: the front under the head was not joined by the first round"
        );
    }
    let (_, pstat, o) = r?;
    // FN75: the chunk's device span and wall into the served-step ledger
    st.f75.pf_dev_ms += pstat.dev_ms;
    st.f75.pf_wall_ms += t75.elapsed().as_secs_f64() * 1e3;
    if last {
        rk.f116_chunk_end = Some(Instant::now());
    }
    if rk.mtp.is_some() {
        let tm = Instant::now();
        let n0 = rk.mtp.as_ref().map_or(0, |m| m.pass_ms.len());
        st.last_chunk = mtp_after_chunk(rk, em, recs, lv, ids, last)?;
        if let Some(mt) = rk.mtp.as_ref() {
            st.ms.passes += (mt.pass_ms.len() - n0) as u64;
            st.ms.dev_ms += mt.pass_ms[n0..].iter().sum::<f64>();
        }
        st.ms.host_ms += tm.elapsed().as_secs_f64() * 1e3;
    }
    if let (Some(k), Some(pc)) = (key, st.pc.as_mut()) {
        if store {
            pcache::store(rk, pc, lv, p0, k)?;
        }
        lv.key = k;
    }
    o.context("the pick")
}

/// Apply one command on this rank. Returns the picks it made (prefill: 1, decode: one per ready
/// slot) or the dense rows.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn apply(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    cmd: &Cmd,
) -> anyhow::Result<(Vec<RowOut>, Vec<Vec<f32>>)> {
    // FN113b: no command's device work runs over an END node's copies in flight (`Cmd::First`
    // fences them itself; this is the law's backstop)
    if let Some(pc) = st.pc.as_mut() {
        pcache::end_gate(rk, pc)?;
    }
    // FN55b: the pool slots' image tables as the slots stand now (a command's tables read them
    // by pool slot; Ready / Leave moved the sequences since the last command)
    if rk.vis_on {
        for (i, x) in st.slots.iter().enumerate() {
            rk.ropes[i] = x.as_ref().and_then(|l| l.img.clone());
        }
    }
    // FN119d (`next`): a round a commit left pending is read before any other command
    if st.rpend.is_some() && !matches!(cmd, Cmd::Decode { .. }) {
        resolve_pending(rk, st, ple, ph, false)?;
    }
    match cmd {
        Cmd::Zero { slot, tokens, img } => {
            // FN96: the admission takes its slot's strides back before the zeroing writes it
            seq_reclaim(rk, *slot)?;
            rk.pools.zero(rk.hip, &rk.dev, *slot)?;
            if let Some(k) = rk.pools.kvt.as_ref() {
                k.borrow_mut().start(*slot, *tokens)?;
            }
            let mut lv = Live::fresh(*slot, cfg.eos_token_id);
            if let Some(fx) = img {
                ensure!(rk.vis_on, "an image sequence on a text-only boot");
                lv.img = Some(std::sync::Arc::new(super::vision::ImgSeq::new(fx)));
            }
            if let Some(c) = st.lookup {
                lv.lk = Some(Arc::new(std::sync::Mutex::new(
                    saber_engine::SeqLookup::new(c),
                )));
            }
            rk.ropes[*slot as usize] = lv.img.clone();
            st.slots[*slot as usize] = Some(lv);
        }
        Cmd::Encode { grids } => {
            // both ranks run the tower over the same pixels (rank 0's rows cross as `fn.pix`)
            let mine = if rk.rank == 0 {
                std::mem::take(&mut st.pix)
            } else {
                Vec::new()
            };
            let theirs = rk.peer.exchange().xchg_bytes("fn.pix", &mine)?;
            let px = if rk.rank == 0 { mine } else { theirs };
            let pixels: Vec<f32> = px
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            st.enc_defer = 0;
            if super::run::image_cache_defers(f, st.pc.is_some()) {
                // FN125: the tower is DEFERRED to the first computed chunk that touches an image
                // run (a prompt whose image runs all reload never runs it, and the lazy tower
                // never loads for it); the request is refused HERE where the run would refuse
                let r = super::vision::tower_precheck(grids, pixels.len(), f.vision_patches());
                let ok = u8::from(r.is_ok());
                let peer_ok = rk.peer.exchange().xchg_bytes("fn.pix.ok", &[ok])?;
                st.enc = match (r, peer_ok.first().copied()) {
                    (Ok(()), Some(1)) => {
                        st.tickets += 1;
                        st.enc_defer = st.tickets;
                        st.tower_px = Some((st.tickets, grids.clone(), pixels));
                        st.i125.deferred += 1;
                        Ok(0)
                    }
                    (Err(e), _) => Err(e),
                    (Ok(()), _) => Err("the peer rank refused the images".to_owned()),
                };
                return Ok((Vec::new(), Vec::new()));
            }
            // FN55c: the lazy tower loads on the first image (both ranks, in this command)
            vision_load(rk)?;
            let v = rk.vis.as_mut().context("an image on a text-only boot")?;
            let (s0, g0) = (v.split, v.group_runs);
            let r = v.encode(rk.hip, &rk.dev.stream, grids, &pixels);
            st.i126.note(v.split - s0, v.group_runs - g0);
            // the pair agrees on the answer (one rank's refusal is both ranks')
            let ok = u8::from(r.is_ok());
            let peer_ok = rk.peer.exchange().xchg_bytes("fn.pix.ok", &[ok])?;
            st.enc = match (r, peer_ok.first().copied()) {
                (Ok(_), Some(1)) => Ok(v.epoch),
                (Err(e), _) => Err(format!("{e:#}")),
                (Ok(_), _) => Err("the peer rank's tower refused the images".to_owned()),
            };
        }
        Cmd::Prefill {
            slot,
            ids,
            req,
            last,
            ah,
        } => {
            let mut lv = st.slots[*slot as usize]
                .take()
                .context("prefill into an empty slot")?;
            lookup_extend(&lv, ids);
            // FN119f: the first token's device block, taken by the last piece's chunk
            ensure!(
                rk.first_dev.is_none(),
                "FN119f: a prefill over a first round no `Cmd::First` took"
            );
            rk.first_ah = ah.clone().map(|a| (a, *dp));
            let r = prefill_cached(rk, em, ple, ph, recs, f, st, &mut lv, ids, req, *last);
            rk.first_ah = None;
            if *last && lv.img.is_some() {
                st.i125.prompts += 1;
                // FN125: a prompt in whose prefill no computed chunk touched an image run: the
                // tower never ran (the KV already holds the images)
                // (the pixels stay until the next encode replaces them: a sibling of an n > 1
                // request may still need them)
                if lv
                    .img
                    .as_ref()
                    .is_some_and(|im| im.epoch == 0 && im.defer != 0)
                {
                    st.i125.skipped += 1;
                }
            }
            if rk.first_dev.is_some() {
                // a dense first token reads the walk's backup (one row; the arena was reused)
                st.last_lg = rk.last_lg;
                st.last_rows = 1;
            }
            st.slots[*slot as usize] = Some(lv);
            return Ok((vec![r?], Vec::new()));
        }
        Cmd::Ready { slot, to } => {
            let (s, t) = (*slot, *to);
            ensure!(
                t as usize == st.ready,
                "ready at {t}, the ready set ends at {}",
                st.ready
            );
            {
                let lv = st.slots[s as usize].as_mut().context("ready: empty slot")?;
                rk.pools
                    .normalize(rk.hip, &rk.dev, s, lv.gdn_at, lv.ple_at, *dp)?;
                lv.gdn_at = 0;
                lv.ple_at = *dp;
            }
            if s != t {
                if st.slots[t as usize].is_none() {
                    // FN96: a free slot's strides back before the move writes it
                    seq_reclaim(rk, t)?;
                }
                if st.slots[t as usize].is_some() {
                    // t holds a prefilling sequence: t -> spare, s -> t, spare -> s
                    rk.pools.move_slot(rk.hip, &rk.dev, t, SPARE)?;
                    rk.pools.move_slot(rk.hip, &rk.dev, s, t)?;
                    rk.pools.move_slot(rk.hip, &rk.dev, SPARE, s)?;
                    st.slots.swap(s as usize, t as usize);
                    st.moves += 3;
                } else {
                    rk.pools.move_slot(rk.hip, &rk.dev, s, t)?;
                    st.slots.swap(s as usize, t as usize);
                    st.moves += 1;
                }
                for (i, x) in st.slots.iter_mut().enumerate() {
                    if let Some(l) = x {
                        l.slot = i as u32;
                    }
                }
            }
            st.ready += 1;
        }
        Cmd::Leave { from } => {
            let k = *from as usize;
            ensure!(k < st.ready, "leave from {k}: not a ready slot");
            let last = st.ready - 1;
            st.slots[k] = None;
            if k != last {
                rk.pools.move_slot(rk.hip, &rk.dev, last as u32, k as u32)?;
                st.slots.swap(k, last);
                if let Some(l) = st.slots[k].as_mut() {
                    l.slot = k as u32;
                }
                st.moves += 1;
            } else if let Some(kt) = rk.pools.kvt.as_ref() {
                kt.borrow_mut().release(k as u32)?;
            }
            st.ready -= 1;
        }
        Cmd::Free { slot } => {
            ensure!(*slot as usize >= st.ready, "free {slot}: a ready slot");
            st.slots[*slot as usize] = None;
            if let Some(k) = rk.pools.kvt.as_ref() {
                k.borrow_mut().release(*slot)?;
            }
        }
        Cmd::Decode {
            toks,
            reqs,
            w,
            ah,
            cp,
        } => {
            // LR-N: this step's coupled rows (the round after its verify reads them)
            if let Some(c) = rk
                .mtp
                .as_mut()
                .and_then(|m| m.chain.as_mut())
                .and_then(|d| d.couple.as_mut())
            {
                c.cur.clone_from(cp);
            }
            // FN96: the slots free `hold` decode commands lend their strides (EMPTY)
            if rk.seql.is_some() {
                let held: Vec<bool> = st.slots.iter().map(Option::is_some).collect();
                seq_lend_sync(rk, &held)?;
            }
            let n = st.ready;
            let w = *w;
            ensure!(
                toks.len() == n && reqs.len() == n,
                "decode: {n} ready slots"
            );
            let mtp = rk.mtp.is_some() && st.law.is_some();
            ensure!(w == 1 || mtp, "a width-{w} step without the MTP law");
            // FN119d (`next`): the pending round feeds this step early only at its own shape (the
            // same ready set, a chain long enough for the width); otherwise it is read now
            let early_ok = st
                .rpend
                .as_ref()
                .is_some_and(|p| mtp && w > 1 && p.n == n && p.kn + 1 >= w);
            if st.rpend.is_some() && !early_ok {
                resolve_pending(rk, st, ple, ph, true)?;
            }
            let early_kn = st.rpend.as_ref().map(|p| (p.kn, p.bucket));
            // FN17: the pad rows' slot is never a sequence's home (a swap passes through it)
            ensure!(
                st.slots[SPARE as usize].is_none(),
                "decode: the spare slot holds a sequence"
            );
            let mut lives: Vec<Live> = Vec::with_capacity(n);
            for (i, t) in toks.iter().enumerate() {
                let mut l = st.slots[i]
                    .clone()
                    .context("decode: a hole in the ready set")?;
                // FN16: under MTP the engine knows its committed token (Commit / First): the
                // scheduler must feed exactly that one
                ensure!(
                    !mtp || l.next == *t,
                    "decode: slot {i} is fed {t}, the engine committed {}",
                    l.next
                );
                l.next = *t;
                lives.push(l);
            }
            let wu = w as usize;
            let drafts: Vec<Vec<u32>> = if early_kn.is_some() {
                // FN119d: read under the step (`decode_step_with`'s early reader)
                vec![vec![0u32; wu - 1]; n]
            } else if w > 1 {
                lives
                    .iter()
                    .map(|l| {
                        let (d, pad) = fit_drafts(&l.drafts, wu - 1, l.next);
                        st.ms.padded += pad;
                        d
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let rq: Vec<RowReq> = reqs
                .iter()
                .flat_map(|r| std::iter::repeat_n(r.clone(), wu))
                .collect();
            let pos0: Vec<u32> = lives.iter().map(|l| l.len).collect();
            let bucket = bucket_for(n);
            let t75 = Instant::now();
            let lp90 = gr.lazy_passes;
            // FN113b: the tree's pending demotions run under this step's device span
            let pcr = &mut st.pc;
            let mut got: Option<(Vec<u32>, Vec<Vec<u32>>)> = None;
            let mut reader = |r: &mut Rank<'_, '_>| -> anyhow::Result<Vec<Vec<u32>>> {
                let (kn, bk) = early_kn.context("FN119d: no round pending")?;
                let raw = resolve_ids(r, kn, bk)?;
                let stride = saber_kernels::fn22::IDS_STRIDE as usize;
                let d: Vec<Vec<u32>> = (0..n)
                    .map(|i| (0..wu - 1).map(|j| raw[j * stride + i]).collect())
                    .collect();
                got = Some((raw, d.clone()));
                Ok(d)
            };
            let (outs, stat) = decode_step_with(
                rk,
                em,
                ple,
                ph,
                recs,
                &mut lives,
                bucket,
                *dp,
                gr,
                f,
                &rq,
                w,
                &drafts,
                &mut |r| match pcr.as_mut() {
                    Some(pc) => pcache::demote_pending(r, pc),
                    None => Ok(()),
                },
                ah.as_ref(),
                if early_kn.is_some() {
                    Some(&mut reader)
                } else {
                    None
                },
            )?;
            // FN119d: the round read under the step -- its records, the drafts the step fed
            let early_step = early_kn.is_some();
            let drafts = if early_step {
                let (raw, d) = got
                    .take()
                    .context("FN119d: an early step never read its round's ids")?;
                round_post(rk, st, ple, ph, &raw, true)?;
                d
            } else {
                drafts
            };
            // FN121: this rank's step record (`--fn-ple-trace`; counters always)
            rk.f121.step(
                ple,
                saber_mem::ple_tier::unix_us(t75),
                bucket,
                w,
                n,
                stat.dev_ms,
                stat.host_ms,
                &stat.host,
                stat.ple_nvme,
                stat.ple_hits,
                stat.gate_ms,
            )?;
            // FN75: the target step's spans into the served-step ledger
            {
                let h = |k: &str| stat.host.get(k).copied().unwrap_or(0.0);
                let e = &mut st.f75;
                e.tgt_dev_ms += stat.dev_ms;
                e.tgt_wall_ms += t75.elapsed().as_secs_f64() * 1e3;
                e.tgt_pre_ms += [
                    "embed", "tables", "ple", "rows", "tier", "ple_l0", "issue", "refill",
                ]
                .iter()
                .map(|k| h(k))
                .sum::<f64>();
                e.tgt_wait_ms += h("wait");
                e.tgt_post_ms += h("tier_end") + h("pick");
                // FN93: the phases by name (records only)
                let te = ["te_fence", "te_log", "te_resolve", "te_plan", "te_issue"];
                let pk = ["pick_read", "pick_seam", "pick_xchg"];
                let te_sum: f64 = te.iter().map(|k| h(k)).sum();
                let pk_sum: f64 = pk.iter().map(|k| h(k)).sum();
                let v = [
                    h("te_fence"),
                    h("te_log"),
                    h("te_resolve"),
                    h("te_plan"),
                    h("te_issue"),
                    h("tier_end") - te_sum,
                    h("pick_read"),
                    h("pick_seam"),
                    h("pick_xchg"),
                    h("pick") - pk_sum,
                    h("embed"),
                    h("tables"),
                    h("ple") + h("rows") + h("ple_l0"),
                    h("tier"),
                    h("issue"),
                    h("refill"),
                ];
                for (s, x) in e.sub.iter_mut().zip(v) {
                    *s += x;
                }
                e.w = w;
                e.n = n as u32;
                // FN90a: the tail census's names (the trace writes them; byte-neutral)
                {
                    for (k, n90) in [
                        ("embed", "h_embed"),
                        ("tables", "h_tables"),
                        ("ple", "h_ple"),
                        ("tier", "h_tier"),
                        ("ple_l0", "h_ple_l0"),
                        ("issue", "h_issue"),
                        ("refill", "h_refill"),
                        ("tier_end", "h_tier_end"),
                        ("pick", "h_pick"),
                        ("ple_gate_wait", "ple_gate"),
                    ] {
                        e.fx_add(n90, h(k));
                    }
                    e.fx_add("ple_nvme", stat.ple_nvme as f64);
                    e.fx_add("ple_hits", stat.ple_hits as f64);
                    e.fx_add("x_res", stat.resident as f64);
                    e.fx_add("x_ram", stat.ram as f64);
                    e.fx_add("cap", (gr.lazy_passes - lp90) as f64);
                }
            }
            *dp = 1 - *dp;
            for (i, mut l) in lives.into_iter().enumerate() {
                if !mtp {
                    l.len += 1;
                }
                l.ple_at = *dp;
                st.slots[i] = Some(l);
            }
            if mtp {
                *st.ms.steps.entry(w).or_insert(0) += 1;
                st.pend = Some(Pend {
                    early: early_step,
                    w,
                    pos0,
                    am: (0..n)
                        .map(|i| {
                            outs[i * wu..(i + 1) * wu]
                                .iter()
                                .map(|o| o.top[0].0)
                                .collect()
                        })
                        .collect(),
                    drafts,
                });
            }
            st.last_lg = rk.last_lg;
            st.last_rows = bucket as usize * wu;
            return Ok((outs, Vec::new()));
        }
        Cmd::Commit {
            acc,
            last,
            kn,
            redo,
            dchk,
        } => {
            let pend = st.pend.take().context("a commit without its verify step")?;
            // FN119d: an early step's drafts are each rank's own chain ids: equal on both
            if let Some(d) = dchk {
                ensure!(
                    pend.drafts == *d,
                    "FN119d: rank {} fed the drafts {:?}, rank 0 {d:?}",
                    rk.rank,
                    pend.drafts
                );
            }
            // FN119 (`ahead`): the step issued ahead (its round on the card, or not)
            let issued = rk.ahead.as_mut().and_then(|a| a.issued.take());
            let n = st.ready;
            ensure!(
                acc.len() == n && last.len() == n && pend.pos0.len() == n,
                "commit: {n} ready slots"
            );
            let w = pend.w;
            for i in 0..n {
                let lv = st.slots[i].as_mut().context("commit: a hole")?;
                let a = acc[i];
                ensure!(a < w, "commit: {a} accepted of {} drafts", w - 1);
                let mut c = lv.ctx;
                for &d in pend.drafts.get(i).map_or(&[][..], |d| &d[..a as usize]) {
                    let _ = ph.ids_step(&mut c, &[d]);
                }
                lv.ctx = c;
                lv.len += a + 1;
                lv.next = last[i];
                // FN69: the committed ids into the lookup index, the step's tally by source
                if lv.lk.is_some() {
                    let mut ids: Vec<u32> = pend
                        .drafts
                        .get(i)
                        .map_or(Vec::new(), |d| d[..a as usize].to_vec());
                    ids.push(last[i]);
                    lookup_extend(lv, &ids);
                    if let Some(src) = lv.lk_src.take() {
                        if w > 1 {
                            st.ms.lk.verified(src, u64::from(w - 1), u64::from(a));
                        }
                    }
                }
                if f.fn_mtp_no_rollback {
                    lv.naccept = w;
                    lv.drop = 0;
                } else {
                    lv.naccept = a + 1;
                    lv.drop = w - 1 - a;
                }
                if w > 1 {
                    lv.ema.observe(a, w - 1);
                    st.ms.drafted += u64::from(w - 1);
                    st.ms.accepted += u64::from(a);
                    for p in 0..a as usize {
                        st.ms.per_pos[p.min(3)] += 1;
                    }
                    for p in 0..(a + u32::from(a < w - 1)) as usize {
                        st.ms.eval_pos[p.min(3)] += 1;
                    }
                }
                st.ms.seq_steps += 1;
                st.ms.committed += u64::from(a + 1);
                lv.drafts.clear();
            }
            // FN92: the leader's depth when it sent one (both ranks hold it for the next step)
            if let (Some(k), Some(c)) = (kn, st.ms.depth.as_mut()) {
                c.follow(*k as usize);
            }
            let kn = kn.unwrap_or_else(|| next_k(st));
            if kn == 0 && issued.is_some_and(|i| i.round) {
                // FN119: a round issued ahead that the law no longer wants: drained
                let _ = finish_chain(rk, issued.map_or(0, |i| i.kn))?;
                rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
            }
            if kn == 0 {
                // the law drafts nothing next (the drafter's KV keeps these positions' holes);
                // FN30c: no round follows, the held copies go now (FN56: the deferred tier end
                // first)
                tier_flush(rk)?;
                rk.tier
                    .mover
                    .release(&rk.dev.stream)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                return Ok((Vec::new(), Vec::new()));
            }
            // F37: the drafter after the commit -- pass 1 over the verify rows at q_len w (row t =
            // the target's stream at t + the token at t + 1: the accepted drafts, then the
            // committed last token; rejected rows ride with their argmax), then the chain
            let tm = Instant::now();
            rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
            let bucket = bucket_for(n);
            let wu = w as usize;
            let mut toks: Vec<u32> = Vec::with_capacity(bucket as usize * wu);
            for i in 0..bucket as usize {
                for t in 0..wu {
                    toks.push(if i >= n {
                        0
                    } else if (t as u32) < acc[i] {
                        pend.drafts[i][t]
                    } else if t as u32 == acc[i] {
                        last[i]
                    } else {
                        pend.am[i][t]
                    });
                }
            }
            // FN58b: the committed tokens' PLE rows under the round (`--fn-ple-overlap layer1`)
            let ov = rk.ple_ov.as_ref().is_some_and(|o| o.pre_on);
            // FN119d (`next`): this commit leaves the round's ids pending (the lookup off)
            let next_ok = st.ds.next()
                && rk.ahead.as_ref().is_some_and(|a| a.early.is_some())
                && st.slots[..n].iter().flatten().all(|l| l.lk.is_none());
            let slots = &mut st.slots[..n];
            let rd0 = rk.rd_t.clone();
            let mut hook = || ple_pregather(ple, ph, slots.iter_mut().flatten());
            let round_ahead = issued.is_some_and(|i| i.round);
            let chk = rk.ahead.as_mut().and_then(|a| a.chk.take());
            let dr = if round_ahead && !*redo && issued.is_some_and(|i| i.kn == kn) {
                // FN119c (b > 1): the inputs the card composed == the host's at these counts
                if let Some(c) = chk {
                    let tchk = Instant::now();
                    let dpos: Vec<u32> = (0..n).map(|i| c.pos0[i] + acc[i]).collect();
                    let (tabs, _) = chain_tables_at(rk, &dpos, n, bucket, kn, false)?;
                    let sel1: Vec<i32> = (0..bucket as usize)
                        .map(|i| {
                            if i < n {
                                (i as u32 * w + acc[i]) as i32
                            } else {
                                -1
                            }
                        })
                        .collect();
                    let want = chain_input_bytes(rk, &sel1, n, &tabs, Some(acc))?;
                    ensure!(
                        c.compose(acc) == want,
                        "FN119c: the composed chain inputs differ from the host's at counts {acc:?}"
                    );
                    let ms = tchk.elapsed().as_secs_f64() * 1e3;
                    if let Some(a) = rk.ahead.as_mut() {
                        a.c.composed_checked += 1;
                        a.c.chk_ms.add(ms);
                    }
                    rd(rk, "rd_chk119c", tchk);
                }
                if next_ok {
                    // FN119d: the host half without the ids; the next step reads them
                    ahead_defer(rk, kn, if ov { Some(&mut hook) } else { None })?;
                    st.rpend = Some(super::dstep::RoundPend {
                        kn,
                        n,
                        bucket,
                        tm,
                        rd0,
                        first: false,
                    });
                    if let Some(a) = rk.ahead.as_mut() {
                        a.c.next_pending += 1;
                    }
                    return Ok((Vec::new(), Vec::new()));
                }
                // FN119 (`ahead`): the round is on the card behind the verify -- its host half
                ahead_finish(rk, n, bucket, kn, if ov { Some(&mut hook) } else { None })?
            } else {
                if round_ahead {
                    // the round issued ahead is void: drained, then today's round on these commits
                    let ik = issued.map_or(0, |i| i.kn);
                    let _ = finish_chain(rk, ik)?;
                    rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
                    if let Some(a) = rk.ahead.as_mut() {
                        a.c.redo += 1;
                    }
                }
                mtp_after_verify(
                    rk,
                    em,
                    recs,
                    &toks,
                    &pend.pos0,
                    w,
                    acc,
                    bucket,
                    kn,
                    if ov { Some(&mut hook) } else { None },
                )?
            };
            let cap = st.law.as_ref().map_or(0, |l| l.kmax()) as usize;
            for (i, d) in dr.into_iter().enumerate() {
                if let Some(lv) = st.slots[i].as_mut() {
                    lv.drafts = d;
                    lookup_merge(lv, cap, &mut st.ms.lk);
                }
            }
            // FN121 (`cand`): the next verify's candidate rows on their way before this returns
            let tpf = Instant::now();
            let pf = ple_prefetch_cand(rk, ple, ph, st.slots[..n].iter().flatten())?;
            let pf_ms = tpf.elapsed().as_secs_f64() * 1e3;
            {
                let parts: Vec<(&'static str, f64)> = rk
                    .rd_t
                    .iter()
                    .map(|(k, v)| (*k, v - rd0.get(k).copied().unwrap_or(0.0)))
                    .filter(|(_, v)| *v != 0.0)
                    .collect();
                let dev: f64 = rk.mtp.as_ref().map_or(0.0, |m| m.pass_ms.iter().sum());
                rk.f121.round(
                    ple,
                    saber_mem::ple_tier::unix_us(tm),
                    tm.elapsed().as_secs_f64() * 1e3,
                    dev,
                    &parts,
                    pf,
                    pf_ms,
                )?;
            }
            let mt = rk.mtp.as_ref().context("mtp")?;
            st.ms.rounds += 1;
            st.ms.passes += mt.pass_ms.len() as u64;
            st.ms.dev_ms += mt.pass_ms.iter().sum::<f64>();
            let round_ms = tm.elapsed().as_secs_f64() * 1e3;
            st.ms.host_ms += round_ms;
            // FN92: rank 0 times the b1 round (the leader's clock prices the depths)
            if n == 1 {
                if let (Some(c), Some(it)) = (st.ms.depth.as_mut(), st.ms.it.as_mut()) {
                    c.time_round(kn, round_ms);
                    it.2 = round_ms;
                }
            }
        }
        Cmd::First { slot, tok, redo } => {
            let (ids, p0, chunk) = st
                .last_chunk
                .take()
                .context("a first drafter pass without its prompt chunk")?;
            // FN119e: the logits backup serves this prompt's pick only; the front under its chunk
            rk.lg_backup = None;
            let front_ran = std::mem::take(&mut rk.front_done);
            let s = *slot as usize;
            ensure!(s < st.ready, "first pass: slot {s} is not ready");
            let tm = Instant::now();
            // FN116: the first round's terms (the device idle since the chunk, the segments)
            let f_since = rk
                .f116_chunk_end
                .take()
                .map_or(-1.0, |t| (tm - t).as_secs_f64() * 1e3);
            let f_seg0 = rk.mtp.as_ref().context("mtp")?.seg.clone();
            // FN119f (`dev`): the round queued behind the chunk (its pass events pending)
            let fdev = rk.first_dev.take();
            if fdev.is_none() {
                rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
            }
            {
                let mut lv = st.slots[s].take().context("first pass: an empty slot")?;
                let r = mtp_stash_row(rk, em, recs, &mut lv, ids[0], p0, chunk);
                st.slots[s] = Some(lv);
                r?;
            }
            let f_stash = tm.elapsed().as_secs_f64() * 1e3;
            let f_np0 = rk.mtp.as_ref().context("mtp")?.pass_ms.len();
            let mut toks = ids[1..].to_vec();
            toks.push(*tok);
            let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
            // FN113 (h1's turn-class break, LEDGER FN113): the END node holds the drafter rows in
            // the form the CUT computes them -- a non-last chunk's `ids[1..]` pass of m - 1 rows --
            // not this prompt's m-row pass with the sampled token (its bytes differ: a reload then
            // drafted unlike the computed cut, turn t1/t2 0/8 vs the verify arm). So the m - 1 row
            // pass runs first, the node is stored, and the m-row pass below rewrites every row it
            // wrote (the prompt's own state = the tree off's). The target stream's last prompt row
            // is the stash a later piece's drafter row needs.
            // FN113b (`--fn-prefix-tree-end live`): NO m - 1 row pass -- the node takes the drafter
            // rows as this prompt's own m-row pass leaves them (the tree off's state; drafter-only
            // bytes: the width-1 verify class keeps every target byte, FN24)
            // FN116 (`--fn-first-step kv`): alone in the ready set with a chain to run and no END
            // node: rows 0..m-2 through the MTP layer's K-side front only, then the last row as a
            // STEADY drafter round (its target stream moved to row 0: the round's pass 1 reads row
            // `s * w + acc` = 0; the decode-form pass + the device tail + the chain, one read).
            // With an END node (the tree, FN113b `live` only): its stash before the rows, its
            // pages + TARGET part on the mover under the front, its DRAFTER part on the step stream
            // between the round's pass 1 and the chain (the rows as this form leaves them).
            let end_live = f.fn_prefix_tree_end == "live";
            // FN119e (`kvnext`): the first round's ids left pending (the next decode is queued
            // under it, as `next` does after a commit)
            let fdefer = super::run::first_step_defer(f)
                && st.ds.next()
                && rk.ahead.as_ref().is_some_and(|a| a.early.is_some())
                && st.slots[s].as_ref().is_some_and(|l| l.lk.is_none());
            let kvf = fdev.is_none()
                && super::run::first_step_kv(f)
                && next_k(st) > 1
                && st.ready == 1
                && s == 0
                && rk.mtp.as_ref().is_some_and(|m| m.chain.is_some())
                && st.slots[s]
                    .as_ref()
                    .is_some_and(|l| l.pt_end.is_none() || end_live);
            let mut dev_pending = false;
            let (dr, kn, f_rows, f_np1, f_pick, f_chain) = if let Some(fd) = fdev {
                let m = ids.len() as u32;
                let f_t = Instant::now();
                ensure!(
                    s == 0 && st.ready == 1,
                    "FN119f: a first round off pool slot 0"
                );
                if fd.node {
                    // the stash read back behind the front (its event; no stream wait)
                    let e = rk
                        .ahead
                        .as_ref()
                        .and_then(|a| a.early.as_ref())
                        .context("FN119f: the early buffers")?;
                    rk.hip.sync_event(&e.stash_ev)?;
                    let stash = e.stash.as_slice()[..(W * 2) as usize].to_vec();
                    st.slots[s]
                        .as_mut()
                        .context("first pass: an empty slot")?
                        .stash = stash;
                }
                let kn = next_k(st);
                let refused_node = matches!(fd.prep, Some(pcache::EndPrep::Refused));
                let f_rows = f_t.elapsed().as_secs_f64() * 1e3;
                let f_t = Instant::now();
                let dr = if !*redo && !refused_node && kn == fd.kn {
                    // the walk's token == the host's: the round stands (its ids read later)
                    if let Some(p) = fd.prep {
                        let mut lv = st.slots[s].take().context("first pass: an empty slot")?;
                        let r = live_commit(rk, st, &mut lv, p);
                        lv.stash.clear();
                        st.slots[s] = Some(lv);
                        r?;
                    }
                    // the prompt's deferred tier end here, its host work under the queued round
                    // (h17: 1.0 ms in the first decode's pre otherwise)
                    tier_flush(rk)?;
                    dev_pending = true;
                    Vec::new()
                } else {
                    // refused (or the node's store, or the chain length, needs the round again):
                    // drain the queued round, then the `kv` round on the host's token -- every
                    // byte the queued one wrote is written again before a read
                    {
                        let e = rk
                            .ahead
                            .as_ref()
                            .and_then(|a| a.early.as_ref())
                            .context("FN119f: the early buffers")?;
                        rk.hip.sync_event(&e.ids_ev)?;
                    }
                    rk.mtp.as_mut().context("mtp")?.evs.clear();
                    if let Some(a) = rk.ahead.as_mut() {
                        a.c.redo += 1;
                    }
                    let mut held = fd.prep;
                    let mut ran = false;
                    let dr = {
                        let stm = &mut *st;
                        let mut mid = |rk: &mut Rank<'_, '_>| -> anyhow::Result<()> {
                            ran = true;
                            let Some(p) = held.take() else {
                                return Ok(());
                            };
                            let mut lv =
                                stm.slots[s].take().context("first pass: an empty slot")?;
                            let r = live_end(rk, stm, &mut lv, p);
                            lv.stash.clear();
                            stm.slots[s] = Some(lv);
                            r
                        };
                        drafter_round(
                            rk,
                            em,
                            recs,
                            &[*tok],
                            &[p0 + m - 1],
                            1,
                            &[0],
                            1,
                            kn,
                            None,
                            Some(&mut mid),
                            false,
                        )?
                    };
                    ensure!(ran, "FN119f: the re-run round never reached its mid hook");
                    dr
                };
                let f_chain = f_t.elapsed().as_secs_f64() * 1e3;
                rk.f116_kv += 1;
                (dr, kn, f_rows, f_np0, 0.0, f_chain)
            } else if kvf {
                let m = ids.len() as u32;
                let f_t = Instant::now();
                let node = st.slots[s].as_ref().is_some_and(|l| l.pt_end.is_some());
                if node {
                    let stash = d2h_in(
                        &rk.dev,
                        &rk.mtp.as_ref().context("mtp")?.hid.own,
                        hid + u64::from(m - 1) * W * 2,
                        (W * 2) as usize,
                    )?;
                    st.slots[s]
                        .as_mut()
                        .context("first pass: an empty slot")?
                        .stash = stash;
                }
                if front_ran {
                    // FN119e: the front ran under the chunk (its pass events already queued)
                } else {
                    mtp_kv_rows(rk, em, recs, *slot, &ids[1..], p0, chunk, hid)?;
                }
                let mut held = if node {
                    let lv = st.slots[s].take().context("first pass: an empty slot")?;
                    let r = live_begin(rk, st, &lv);
                    st.slots[s] = Some(lv);
                    Some(r?)
                } else {
                    None
                };
                if m >= 2 {
                    let mt = rk.mtp.as_ref().context("mtp")?;
                    let (own, off) = (mt.hid.own.clone(), mt.hid.off);
                    rk.hip.run_batch(&[Job::MemcpyD2d {
                        dst: &own,
                        dst_offset: off,
                        src: &own,
                        src_offset: off + u64::from(m - 1) * W * 2,
                        bytes: W * 2,
                        stream: &rk.dev.stream,
                    }])?;
                }
                let f_rows = f_t.elapsed().as_secs_f64() * 1e3;
                let kn = next_k(st);
                let f_t = Instant::now();
                let mut ran = false;
                let dr = {
                    let stm = &mut *st;
                    let mut mid = |rk: &mut Rank<'_, '_>| -> anyhow::Result<()> {
                        ran = true;
                        let Some(p) = held.take() else {
                            return Ok(());
                        };
                        let mut lv = stm.slots[s].take().context("first pass: an empty slot")?;
                        let r = live_end(rk, stm, &mut lv, p);
                        lv.stash.clear();
                        stm.slots[s] = Some(lv);
                        r
                    };
                    drafter_round(
                        rk,
                        em,
                        recs,
                        &[*tok],
                        &[p0 + m - 1],
                        1,
                        &[0],
                        1,
                        kn,
                        None,
                        Some(&mut mid),
                        fdefer,
                    )?
                };
                ensure!(ran, "FN116: the round never reached its mid hook");
                let f_chain = f_t.elapsed().as_secs_f64() * 1e3;
                // the K-side front's events lead the round's (both read by its `finish_chain`)
                let np1 = f_np0 + ((m as usize).saturating_sub(1)).div_ceil(MTP_PIECE as usize);
                rk.f116_kv += 1;
                (dr, kn, f_rows, np1, 0.0, f_chain)
            } else {
                let live = f.fn_prefix_tree_end == "live"
                    && st.slots[s].as_ref().is_some_and(|l| l.pt_end.is_some());
                if !live && st.slots[s].as_ref().is_some_and(|l| l.pt_end.is_some()) {
                    let te = Instant::now();
                    let m = ids.len() as u64;
                    if m >= 2 {
                        mtp_rows(rk, em, recs, *slot, &ids[1..], p0, chunk, hid, false, None)?;
                    }
                    let mut lv = st.slots[s].take().context("first pass: an empty slot")?;
                    let r = end_block(rk, st, f, &mut lv, hid, m, te);
                    lv.stash.clear();
                    st.slots[s] = Some(lv);
                    // FN113b: the END block's terms (the m - 1 row pass + the stash read, synced)
                    let mtp_ms = r?;
                    if let Some(t) = st.pc.as_mut().and_then(|pc| pc.tree.as_mut()) {
                        let all = te.elapsed().as_secs_f64() * 1e3;
                        t.c.end_ms += all;
                        t.c.end_mtp_ms += mtp_ms;
                        t.c.end_worst_ms = t.c.end_worst_ms.max(all);
                    }
                }
                if live {
                    // the stash (the target stream's last prompt row) before the pass
                    let m = ids.len() as u64;
                    let stash = d2h_in(
                        &rk.dev,
                        &rk.mtp.as_ref().context("mtp")?.hid.own,
                        hid + (m - 1) * W * 2,
                        (W * 2) as usize,
                    )?;
                    st.slots[s]
                        .as_mut()
                        .context("first pass: an empty slot")?
                        .stash = stash;
                }
                let f_t = Instant::now();
                let lg = mtp_rows(rk, em, recs, *slot, &toks, p0, chunk, hid, true, None)?;
                let f_rows = f_t.elapsed().as_secs_f64() * 1e3;
                let f_np1 = rk.mtp.as_ref().context("mtp")?.pass_ms.len();
                // FN113b live: the node's pages + the victim's demotion on the host and its TARGET
                // part in flight while the m-row pass runs on the card
                let lprep = if live {
                    let lv = st.slots[s].take().context("first pass: an empty slot")?;
                    let r = live_begin(rk, st, &lv);
                    st.slots[s] = Some(lv);
                    Some(r?)
                } else {
                    None
                };
                let f_t = Instant::now();
                let d1 = draft_ids(rk, lg, 1)?[0];
                let f_pick = f_t.elapsed().as_secs_f64() * 1e3;
                if let Some(p) = lprep {
                    let mut lv = st.slots[s].take().context("first pass: an empty slot")?;
                    let r = live_end(rk, st, &mut lv, p);
                    lv.stash.clear();
                    st.slots[s] = Some(lv);
                    r?;
                }
                let mut dr = vec![vec![d1]];
                let kn = next_k(st);
                let f_t = Instant::now();
                if kn > 1 && st.ready == 1 && s == 0 {
                    // alone in the ready set: the whole chain now (else the drafts are padded once)
                    mtp_chain_k(rk, em, recs, &mut dr, &[p0 + toks.len() as u32 - 1], 1, kn)?;
                }
                let f_chain = f_t.elapsed().as_secs_f64() * 1e3;
                (dr, kn, f_rows, f_np1, f_pick, f_chain)
            };
            let mut dr = dr;
            // FN119e: the round left pending -- its result block carries the sampled token as the
            // committed one (the next early step's row 0), the round read at the next command
            let pending = kvf && fdefer;
            if dev_pending {
                // FN119f: the walk wrote the result block (the token as the committed one)
                st.rpend = Some(super::dstep::RoundPend {
                    kn,
                    n: st.ready,
                    bucket: bucket_for(st.ready),
                    tm,
                    rd0: rk.rd_t.clone(),
                    first: true,
                });
            }
            if pending {
                let off = super::dstep::AheadRt::OUT as u64 + s as u64 * 64;
                let a = rk.ahead.as_mut().context("FN119: the ahead")?;
                let e = a.early.as_mut().context("FN119d: the early buffers")?;
                let b = e.out0.as_mut_slice();
                b[..64].fill(0);
                b[8..12].copy_from_slice(&1u32.to_le_bytes());
                b[12..16].copy_from_slice(&tok.to_le_bytes());
                rk.hip.memcpy_h2d_at(
                    &a.dev,
                    off,
                    e.out0.slice(0, 64).context("FN119e: the result block")?,
                    &rk.dev.stream,
                )?;
                a.c.first_pending += 1;
                st.rpend = Some(super::dstep::RoundPend {
                    kn,
                    n: st.ready,
                    bucket: bucket_for(st.ready),
                    tm,
                    rd0: rk.rd_t.clone(),
                    first: true,
                });
            }
            // FN113b: an END node's target part landed before the next command's device work
            if let Some(pc) = st.pc.as_mut() {
                pcache::end_gate(rk, pc)?;
            }
            let cap = st.law.as_ref().map_or(0, |l| l.kmax()) as usize;
            if let Some(lv) = st.slots[s].as_mut() {
                lv.drafts = dr.pop().unwrap_or_default();
                lv.next = *tok;
                // FN69: the prompt's first generated token, then the merge law
                lookup_extend(lv, &[*tok]);
                lookup_merge(lv, cap, &mut st.ms.lk);
            }
            // FN121 (`cand`): the first verify's candidate rows on their way
            ple_prefetch_cand(rk, ple, ph, st.slots[s..=s].iter().flatten())?;
            let mt = rk.mtp.as_ref().context("mtp")?;
            st.ms.firsts += 1;
            st.ms.passes += mt.pass_ms.len() as u64;
            st.ms.dev_ms += mt.pass_ms.iter().sum::<f64>();
            let f_total = tm.elapsed().as_secs_f64() * 1e3;
            st.ms.host_ms += f_total;
            // FN116: the round's record (`F116_TERMS`)
            let d = |k: &str| {
                let a = mt.seg.get(k).map_or(0.0, |v| v.0);
                let b = f_seg0.get(k).map_or(0.0, |v| v.0);
                a - b
            };
            let pm = &mt.pass_ms;
            let rec = [
                toks.len() as f64,
                f_since,
                f_total,
                f_stash,
                f_rows,
                d("rows_emb"),
                d("rows_consts"),
                d("lend"),
                d("launch"),
                d("wait"),
                pm[f_np0.min(pm.len())..f_np1.min(pm.len())].iter().sum(),
                f_pick,
                f_chain,
                pm[f_np1.min(pm.len())..].iter().sum(),
                f64::from(kn),
            ];
            rk.f116.push(rec);
        }
        Cmd::Dense { rows, stage } => {
            st.f75.dense += 1;
            let half = VOCAB as usize / 2;
            let raw = if *stage {
                // FN119 (`ahead`): the walk's refused rows the device backed up into FN112's
                // staging (the round ahead reused the arena): the walk's copy landed first
                let a = rk
                    .ahead
                    .as_mut()
                    .context("FN119: a staged dense without the ahead")?;
                rk.hip.sync_event(&a.ev)?;
                a.c.stage_dense += 1;
                let flags: Vec<u32> = {
                    let b = a.stg.as_slice();
                    (0..st.last_rows)
                        .map(|r| {
                            let o = super::dstep::AheadRt::ROWOUT + r * 16 + 4;
                            u32::from_le_bytes(b[o..o + 4].try_into().expect("4"))
                        })
                        .collect()
                };
                for &r in rows {
                    ensure!(
                        flags.get(r as usize).is_some_and(|&x| x != 0),
                        "FN119: a dense row ({r}) the device walk did not refuse has no backup"
                    );
                }
                let o = rk.ov112.as_ref().context("FN112's staging")?;
                let at = o.lay.raw_off;
                o.stg.as_slice()[at..at + st.last_rows * half * 2].to_vec()
            } else if let Some(ev) = rk.lg_backup.as_ref().filter(|_| st.last_rows == 1) {
                // FN119e: the front reused the arena after the chunk's pick: its backup
                rk.hip.sync_event(ev)?;
                let o = rk.ov112.as_ref().context("FN112's staging")?;
                let at = o.lay.raw_off;
                o.stg.as_slice()[at..at + half * 2].to_vec()
            } else {
                d2h_in(&rk.dev, &rk.dev.arena, st.last_lg, st.last_rows * half * 2)?
            };
            let mut mine = Vec::with_capacity(rows.len() * half * 2);
            for &r in rows {
                let r = r as usize;
                mine.extend_from_slice(&raw[r * half * 2..(r + 1) * half * 2]);
            }
            let theirs = rk.peer.exchange().xchg_bytes("fn.dense", &mine)?;
            let mut out = Vec::new();
            for i in 0..rows.len() {
                let a = bf16_to_f32(&mine[i * half * 2..(i + 1) * half * 2]);
                let b = bf16_to_f32(&theirs[i * half * 2..(i + 1) * half * 2]);
                out.push(if rk.rank == 0 {
                    a.into_iter().chain(b).collect()
                } else {
                    b.into_iter().chain(a).collect()
                });
            }
            return Ok((Vec::new(), out));
        }
        Cmd::VisionRelease => vision_release(rk)?,
        Cmd::Valve { closed } => valve_apply(rk, *closed)?,
        Cmd::Stop | Cmd::Noop => {}
    }
    Ok((Vec::new(), Vec::new()))
}

/// FN16: one MTP decode step on rank 0 -- the law's width for the ready set (the table at the
/// bucket, the EMA cap arm, the pools' context), the verify rows (`Cmd::Decode`), F40's walk per
/// slot (`verify::walk_slot`: the 27B's drawn-row code), the dense rows the seam could not decide,
/// the results; the commit + drafter (`Cmd::Commit`) is queued to run after the reply.
#[allow(clippy::too_many_arguments)]
fn mtp_step(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    batch: &serve::OwnedBatch,
    seq_slot: &BTreeMap<SeqId, u32>,
    sampler: &std::sync::Arc<saber_serve::sampler::Sampler>,
    res: &mut Vec<saber_sched::SeqStep>,
    post: &mut Vec<Cmd>,
) -> anyhow::Result<()> {
    let n = st.ready;
    let law = st.law.clone().context("mtp law")?;
    let mut toks = vec![0u32; n];
    let mut reqs = vec![RowReq::default(); n];
    let mut per: Vec<(usize, bool)> = Vec::with_capacity(batch.slots.len());
    for s in &batch.slots {
        let slot = *seq_slot
            .get(&s.seq)
            .context("decode of an unadmitted sequence")? as usize;
        ensure!(slot < n, "decode of a sequence that is not ready");
        toks[slot] = *s.input.last().context("a decode slot feeds no token")?;
        let (params, pen, mut rq) = serve::row_req(&s.sampling);
        if st.ds.lazy() {
            super::dstep::shape(&mut rq, &params, &pen);
        }
        reqs[slot] = rq;
        per.push((slot, s.verify_width > 0));
    }
    ensure!(
        per.len() == n,
        "the batch holds {} of {n} ready sequences",
        per.len()
    );
    let bucket = bucket_for(n);
    let emas: Vec<verify::WidthEma> = (0..n)
        .filter_map(|i| st.slots[i].as_ref().map(|l| l.ema))
        .collect();
    let room = (0..n)
        .filter_map(|i| {
            st.slots[i]
                .as_ref()
                .map(|l| rk.max_ctx.saturating_sub(l.len))
        })
        .min()
        .unwrap_or(1);
    // FN92: at b1 under the adaptive arm the width is the depth both ranks hold (+ the committed
    // row); the iteration clock: the previous b1 iteration's wall less its verify and round = the
    // host gap every depth pays
    let adaptive = law.adaptive && bucket == 1 && st.ms.depth.is_some();
    let w = if adaptive {
        st.ms.depth.as_ref().map_or(1, |c| c.depth() as u32 + 1)
    } else {
        law.width(bucket, &emas)
    }
    .min(room)
    .max(1);
    let t_step = Instant::now();
    if adaptive {
        if let (Some((t0, v, r)), Some(c)) = (st.ms.it, st.ms.depth.as_mut()) {
            c.time_gap(t0.elapsed().as_secs_f64() * 1e3 - v - r);
        }
        st.ms.it = Some((t_step, 0.0, 0.0));
    } else {
        st.ms.it = None;
    }
    let wu = w as usize;
    // FN119 (`--fn-device-step ahead`): the step issued ahead (b1, a drafting verify, a fixed
    // chain >= 2, a sampling the device walk decides): each row's device block, an unseeded
    // request's key drawn here (the host walk draws with the same one)
    let mut samp: Vec<saber_sched::SlotSampling> =
        batch.slots.iter().map(|s| s.sampling.clone()).collect();
    // LR-N (`--spec-couple on`): each sampled slot's coupled row (the drafts of the round after
    // this verify take its rows' noise keys); an unseeded request draws its seed ONCE
    let couple_on = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .is_some_and(|c| c.couple.is_some());
    let mut cp: Vec<super::couple::CoupleRow> = Vec::new();
    if couple_on {
        st.couple_seeds.retain(|q, _| seq_slot.contains_key(q));
        cp = vec![[0u32; 5]; n];
        for (i, s) in samp.iter_mut().enumerate() {
            let params = saber_serve::sampler::RowParams::resolve(&s.params, VOCAB as usize);
            if params.is_greedy() {
                continue;
            }
            let sd = match s.params.seed {
                Some(sd) => sd,
                None => *st
                    .couple_seeds
                    .entry(batch.slots[i].seq)
                    .or_insert_with(ahead_seed),
            };
            s.params.seed = Some(sd);
            cp[per[i].0] = super::couple::couple_row(&params, sd, s.choice_index, s.position);
        }
    }
    let mut ah: Option<super::dstep::AheadCmd> = None;
    if st.ds.ahead() && rk.ahead.is_some() {
        let kn = next_k(st);
        let chain_ok = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .is_some_and(|c| c.check.is_none() && c.conf.is_none());
        // FN119c: any batch whose sequences all draft (b > 1 composes the chain inputs per sequence)
        let ok = n >= 1
            && w > 1
            && per.iter().all(|p| p.1)
            && !law.dynamic
            && !adaptive
            && kn >= 2
            && chain_ok
            && samp.iter().all(|s| {
                let (p, pen, _) = serve::row_req(s);
                super::dstep::ahead_sampling_ok(&p, &pen)
            });
        if let Some(a) = rk.ahead.as_mut() {
            a.c.steps += 1;
            if !ok {
                a.c.ineligible += 1;
            }
        }
        if ok {
            // the row blocks in pool-slot order (row = slot x w + j)
            let mut blocks = vec![[0u32; 8]; n * wu];
            for (i, s) in samp.iter_mut().enumerate() {
                let slot = per[i].0;
                let params = saber_serve::sampler::RowParams::resolve(&s.params, VOCAB as usize);
                if s.params.seed.is_none() && !params.is_greedy() {
                    s.params.seed = Some(ahead_seed());
                }
                for j in 0..wu {
                    let key = s.params.seed.map_or(0, |sd| {
                        super::dstep::noise_key(sd, s.position + j as u64, s.choice_index)
                    });
                    blocks[slot * wu + j] = super::dstep::row_block(&params, false, 0, key);
                }
            }
            ah = Some(super::dstep::AheadCmd { blocks, kn });
        }
    }
    let ahead_step = ah.is_some();
    let (outs, _) = lead(
        rk,
        em,
        ple,
        ph,
        recs,
        cfg,
        f,
        gr,
        dp,
        st,
        &Cmd::Decode {
            toks,
            reqs: reqs.clone(),
            w,
            ah,
            cp,
        },
    )?;
    let mut outs = outs;
    ds_fill(
        rk, em, ple, ph, recs, cfg, f, gr, dp, st, &mut outs, &reqs, wu, ahead_step,
    )?;
    let fed: Vec<Vec<u32>> = st
        .pend
        .as_ref()
        .map(|p| p.drafts.clone())
        .unwrap_or_default();
    let mut walks = Vec::with_capacity(per.len());
    let mut dense_rows = Vec::new();
    for (i, &(slot, may)) in per.iter().enumerate() {
        let rows = &outs[slot * wu..(slot + 1) * wu];
        let d: &[u32] = if may && w > 1 { &fed[slot] } else { &[] };
        let wk = verify::walk_slot(sampler, rows, d, &samp[i]);
        if let Some(j) = wk.dense {
            dense_rows.push((slot * wu + j) as u32);
        }
        walks.push(wk);
    }
    if !dense_rows.is_empty() {
        let (_, dn) = lead(
            rk,
            em,
            ple,
            ph,
            recs,
            cfg,
            f,
            gr,
            dp,
            st,
            &Cmd::Dense {
                rows: dense_rows.clone(),
                stage: ahead_step,
            },
        )?;
        for (i, &(slot, may)) in per.iter().enumerate() {
            if let Some(j) = walks[i].dense {
                let r = (slot * wu + j) as u32;
                let k = dense_rows
                    .iter()
                    .position(|&x| x == r)
                    .context("dense row")?;
                let d: &[u32] = if may && w > 1 { &fed[slot] } else { &[] };
                verify::dense_decide(
                    sampler,
                    &mut walks[i],
                    &outs[r as usize],
                    dn[k].clone(),
                    d,
                    &samp[i],
                );
            }
        }
    }
    // FN119 (`ahead`): the card's walk (both ranks ran it) == the host's, or the round is re-run
    let mut redo = false;
    if ahead_step {
        let a = rk.ahead.as_mut().context("FN119: the ahead")?;
        rk.hip.sync_event(&a.ev)?;
        for (i, &(slot, _)) in per.iter().enumerate() {
            let (fl, dacc, dtok) = a.result(slot);
            ensure!(
                fl & super::dstep::AHEAD_TIMEOUT == 0,
                "FN119: the peer's seam never landed in the window (flags {fl:#x})"
            );
            let host_dense =
                walks[i].dense.is_some() || dense_rows.iter().any(|&r| r as usize / wu == slot);
            if fl == 0 {
                ensure!(
                    !host_dense && dtok == walks[i].tokens && dacc == walks[i].accepted(),
                    "FN119: the device walk committed {dtok:?} (acc {dacc}), the host {:?} (dense \
                     {host_dense}): the twin's law is broken",
                    walks[i].tokens
                );
                a.c.equal += 1;
            } else {
                a.c.refused += 1;
                redo = true;
            }
        }
    }
    if st.dchk.is_some() {
        let t0 = Instant::now();
        let seqs: Vec<super::dstep::CheckSeq<'_>> = per
            .iter()
            .enumerate()
            .map(|(i, &(slot, may))| super::dstep::CheckSeq {
                row0: slot * wu,
                w: wu,
                draft: if may && w > 1 { &fed[slot] } else { &[] },
                sampling: &batch.slots[i].sampling,
                host_tokens: &walks[i].tokens,
                host_dense: dense_rows.iter().any(|&r| r as usize / wu == slot),
            })
            .collect();
        let seam = rk
            .dpick
            .as_ref()
            .context("the device pick")?
            .out
            .addr()
            .as_u64();
        let dk = st.dchk.as_mut().context("FN119: the check form")?;
        dk.run(
            rk.hip,
            &rk.dev.stream,
            &rk.table,
            seam,
            &outs,
            &seqs,
            &mut st.dsc,
        )?;
        st.dsc.check_ms += t0.elapsed().as_secs_f64() * 1e3;
    }
    if rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .is_some_and(|c| c.conf.is_some())
    {
        let ms = t_step.elapsed().as_secs_f64() * 1e3;
        let seqs: Vec<ConfSeq<'_>> = batch
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (slot, may) = per[i];
                let d: &[u32] = if may && w > 1 { &fed[slot] } else { &[] };
                (s.seq.0, slot, d, walks[i].accepted())
            })
            .collect();
        conf_line(rk, n, bucket, w, ms, &seqs)?;
    }
    let mut acc = vec![0u32; n];
    let mut last = vec![0u32; n];
    for (i, s) in batch.slots.iter().enumerate() {
        let (slot, may) = per[i];
        let wk = &walks[i];
        acc[slot] = wk.accepted();
        last[slot] = *wk.tokens.last().context("a walk commits a token")?;
        let asked = s.sampling.params.logprobs.is_some();
        let lp = if asked {
            wk.lps.iter().cloned().collect::<Option<Vec<_>>>()
        } else {
            None
        };
        res.push(serve::seq_step_k(
            s.seq,
            wk.tokens.clone(),
            if may { w - 1 } else { 0 },
            wk.accepted(),
            lp,
        ));
    }
    // FN92: rank 0 observes the commit, prices the verify, and decides the next depth (sent to
    // rank 1 with the commit: no rank decides from its own clock)
    let mut kn = None;
    if adaptive && n == 1 {
        let verify_ms = t_step.elapsed().as_secs_f64() * 1e3;
        if let Some(it) = st.ms.it.as_mut() {
            it.1 = verify_ms;
        }
        if let (Some(lv), Some(c)) = (st.slots[0].as_mut(), st.ms.depth.as_mut()) {
            if w > 1 && per.first().is_some_and(|p| p.1) {
                lv.accq.observe(w - 1, acc[0]);
            }
            c.time_verify(w, verify_ms);
            kn = Some(c.decide(&lv.accq).0 as u32);
        }
    }
    // FN119d: an early step's drafts (rank 0's read) ride the commit for rank 1's check
    let dchk = st
        .pend
        .as_ref()
        .filter(|p| p.early)
        .map(|p| p.drafts.clone());
    post.push(Cmd::Commit {
        acc,
        last,
        kn,
        redo,
        dchk,
    });
    Ok(())
}

/// FN119 (`ahead`): an unseeded sampled request's key for one step (the host walk and the device
/// draw with it; `EntropyNoise`'s law: the clock and a counter, mixed).
fn ahead_seed() -> u64 {
    static TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let t = TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    super::dstep::mix(nanos ^ super::dstep::mix(t))
}

/// FN119 (`--fn-device-step lazy*`): the deferred rows whose merged seam says their tail can live
/// get their `lse_t` from the dense row (`Cmd::Dense` on the step's logits, both ranks) by the
/// pick's own function before the walk -- the bytes `off` computes, where the sampler reads them.
#[allow(clippy::too_many_arguments)]
fn ds_fill(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    outs: &mut [RowOut],
    reqs: &[RowReq],
    wu: usize,
    stage: bool,
) -> anyhow::Result<()> {
    if !st.ds.lazy() {
        return Ok(());
    }
    // the rows' requests: each slot's request over its w rows (the Decode apply's law)
    let rq: Vec<RowReq> = reqs
        .iter()
        .flat_map(|r| std::iter::repeat_n(r.clone(), wu))
        .collect();
    let live = rq.len().min(outs.len());
    st.dsc.rows += live as u64;
    st.dsc.lse1 += rq[..live].iter().filter(|q| q.lse).count() as u64;
    st.dsc.deferred += rq[..live]
        .iter()
        .filter(|q| q.temp > 0.0 && q.tk > 0)
        .count() as u64;
    let late = super::dstep::late_rows(
        &outs[..live],
        &rq[..live],
        st.ds == super::dstep::DeviceStep::LazyLate,
    );
    if late.is_empty() {
        return Ok(());
    }
    st.dsc.late += late.len() as u64;
    st.dsc.fills += 1;
    let (_, dn) = lead(
        rk,
        em,
        ple,
        ph,
        recs,
        cfg,
        f,
        gr,
        dp,
        st,
        &Cmd::Dense {
            rows: late.iter().map(|&r| r as u32).collect(),
            stage,
        },
    )?;
    ensure!(
        dn.len() == late.len(),
        "FN119: the dense rows of a late fill"
    );
    for (k, &r) in late.iter().enumerate() {
        outs[r].lse_t = Some(super::dstep::fill_lse_t(&dn[k], rq[r].temp));
    }
    Ok(())
}

/// FN119 (`--fn-device-step ahead`): the issue-ahead's buffers on this rank (serve plane, the
/// device chain, FN57's device pick + FN112's queued reads).
fn ahead_alloc(rk: &mut Rank<'_, '_>, f: &FnFlags) -> anyhow::Result<()> {
    use super::dstep::{AheadRt, AHEAD_PLUGIN};
    if !f.device_step()?.ahead() {
        return Ok(());
    }
    let Some(inb) = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .map(|c| c.inb.len())
    else {
        println!(
            "fnahead119 rank{}: STANDS DOWN to lazy -- no device chain (the issue-ahead drives its \
             round)",
            rk.rank
        );
        return Ok(());
    };
    if rk.dpick.is_none() || rk.ov112.is_none() {
        // the default form stands down to `lazy` (FN57's device pick + FN112's queued reads feed it)
        println!(
            "fnahead119 rank{}: STANDS DOWN to lazy -- the device pick and FN112's queued reads \
             (--fn-overlap on) feed it",
            rk.rank
        );
        return Ok(());
    }
    for k in [
        "fn119a_push",
        "fn119a_walk",
        "fn119a_emb",
        "fn119a_gather",
        "fn119a_sel",
        "fn119a_backup",
    ] {
        rk.table
            .get(AHEAD_PLUGIN, k)
            .with_context(|| format!("{AHEAD_PLUGIN}::{k} not loaded"))?;
    }
    for k in ["fn119b_copy", "fn119b_compose"] {
        rk.table
            .get(super::dstep::AHEAD2_PLUGIN, k)
            .with_context(|| format!("flashnext_ahead2::{k} not loaded"))?;
    }
    let mb = rk.pair.mb;
    let woff = super::dstep::win_off(mb);
    ensure!(
        woff + super::dstep::WIN_BYTES <= mb,
        "FN119: the window ({woff} + {} B) past the AR mailbox slot ({mb} B)",
        super::dstep::WIN_BYTES
    );
    let ctr = rk.hip.malloc(256)?;
    let dev = rk.hip.malloc(AheadRt::END as u64)?;
    rk.hip.memset(&ctr, 0, &rk.dev.stream)?;
    rk.hip.memset(&dev, 0, &rk.dev.stream)?;
    let stg = rk.hip.host_malloc(8192)?;
    let stride = (inb as usize).div_ceil(4096) * 4096;
    // >= 2 MiB: the huge-page form (device pointer == host VA, checked by the allocator)
    let runs_off = super::exec::WMAX as usize * stride;
    let vbytes = (runs_off + super::dstep::RUNS_BYTES).max(2 << 20);
    let var = rk.hip.host_malloc(vbytes as u64)?;
    let cs = rk.hip.host_malloc(4 << 20)?;
    let ev = rk.hip.create_event(EventFlags::Default)?;
    // FN119d (`next`): the round's ids read back behind it + the next step's tables' async staging
    let early = if f.device_step()?.next() {
        rk.table
            .get(super::dstep::AHEAD3_PLUGIN, "fn119d_vemb")
            .context("flashnext_ahead3::fn119d_vemb not loaded")?;
        let e = super::dstep::Early {
            ids: rk.hip.host_malloc(4096)?,
            ids_ev: rk.hip.create_event(EventFlags::Default)?,
            tb: rk.hip.host_malloc(rk.dev.consts.len())?,
            out0: rk.hip.host_malloc(4096)?,
            stash: rk.hip.host_malloc(4096.max(W * 2))?,
            stash_ev: rk.hip.create_event(EventFlags::Default)?,
            plog: rk
                .hip
                .host_malloc(u64::from(LAYERS) * (u64::from(MTP_PIECE) + 1) * 40)?,
            plog_ev: rk.hip.create_event(EventFlags::Default)?,
            plog_m: None,
            stin: rk.hip.host_malloc(4096.max(W * 2))?,
        };
        println!(
            "fnahead119 rank{}: NEXT (FN119d) -- the commit leaves the round's ids pending; the \
             next verify's inputs are built and its layers before the PLE layer issued under the \
             round (embedding rows on the card, tables staged {} B pinned)",
            rk.rank,
            e.tb.len()
        );
        Some(e)
    } else {
        None
    };
    rk.hip.sync_stream(&rk.dev.stream)?;
    println!(
        "fnahead119 rank{}: ON -- the device walk on both ranks (seam halves over the P2P window at \
         mailbox +{woff:#x}, {} B), the drafter round's pass-1 embeddings / gather / chain inputs \
         from its result block; chain-input variants {} x {} B pinned, tables 4 MiB pinned",
        rk.rank,
        super::dstep::WIN_BYTES,
        super::exec::WMAX,
        stride
    );
    rk.ahead = Some(AheadRt {
        ctr,
        dev,
        stg,
        var,
        var_stride: stride,
        cs,
        ev,
        issued: None,
        chk: None,
        runs_off,
        early,
        c: super::dstep::AheadCounts::default(),
    });
    Ok(())
}

/// FN119 (`ahead`): one launch of a `flashnext_ahead` symbol (its arguments packed in order, the
/// image padded to the manifest's kernarg size).
fn ahead_launch(
    rk: &Rank<'_, '_>,
    sym: &str,
    grid: [u32; 3],
    block: u32,
    ptrs: &[u64],
    ints: &[i32],
) -> anyhow::Result<()> {
    let (func, decl) = rk
        .table
        .get(super::dstep::AHEAD_PLUGIN, sym)
        .with_context(|| format!("flashnext_ahead::{sym} not loaded"))?;
    let mut args = Vec::with_capacity(decl.kernarg_segment_size as usize);
    for p in ptrs {
        args.extend_from_slice(&p.to_le_bytes());
    }
    for i in ints {
        args.extend_from_slice(&i.to_le_bytes());
    }
    ensure!(
        args.len() <= decl.kernarg_segment_size as usize,
        "{sym}: {} B of arguments for a {} B segment",
        args.len(),
        decl.kernarg_segment_size
    );
    args.resize(decl.kernarg_segment_size as usize, 0);
    rk.hip
        .launch(func, grid, [block, 1, 1], 0, &rk.dev.stream, &mut args)?;
    Ok(())
}

/// FN119c: one launch of a `flashnext_ahead2` symbol (block 256).
fn ahead2_launch(
    rk: &Rank<'_, '_>,
    sym: &str,
    grid: [u32; 3],
    ptrs: &[u64],
    ints: &[i32],
) -> anyhow::Result<()> {
    let (func, decl) = rk
        .table
        .get(super::dstep::AHEAD2_PLUGIN, sym)
        .with_context(|| format!("flashnext_ahead2::{sym} not loaded"))?;
    let mut args = Vec::with_capacity(decl.kernarg_segment_size as usize);
    for p in ptrs {
        args.extend_from_slice(&p.to_le_bytes());
    }
    for i in ints {
        args.extend_from_slice(&i.to_le_bytes());
    }
    ensure!(
        args.len() <= decl.kernarg_segment_size as usize,
        "{sym}: {} B of arguments for a {} B segment",
        args.len(),
        decl.kernarg_segment_size
    );
    args.resize(decl.kernarg_segment_size as usize, 0);
    rk.hip
        .launch(func, grid, [256, 1, 1], 0, &rk.dev.stream, &mut args)?;
    Ok(())
}

/// FN119 (`ahead`): behind the verify (both ranks, before the host waits): the seam pushed to the
/// peer's window, the walk on (own seam, the peer's), the refused rows' logits backed up, the
/// result block read back on its own event; then -- when every accepted count's chain tables share
/// one band and one sparse layout -- the drafter round: pass-1 tables (acc-free), the chain's
/// union KV begin + its variants (the walk's acc picks one on the card), the pass-1 embedding rows
/// (UVA + the exact reduce), pass 1 with its gather on the card, the chain. Every kernel and input
/// byte is today's round's; only the order moves.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ahead_issue(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    recs: &PackRecs,
    ah: &super::dstep::AheadCmd,
    pos0: &[u32],
    drafts: &[Vec<u32>],
    w: u32,
    bucket: u32,
    lg: u64,
    mut pre_round: Option<PreRound<'_>>,
    mid: Option<MidHook<'_>>,
) -> anyhow::Result<()> {
    use super::dstep::{AheadRt, Issued, AHEAD_ROWS};
    use saber_model::qwen4_exp::layer_graph::Group;
    let n = pos0.len();
    let wu = w as usize;
    let rows = bucket as usize * wu;
    let kn = ah.kn;
    // FN119f: width 1 = a prompt's first token (the walk draws the chunk's last row; no drafts)
    ensure!(
        n >= 1
            && n <= bucket as usize
            && (w > 1 || pre_round.is_some())
            && ah.blocks.len() == n * wu
            && drafts.len() == n
            && kn >= 2,
        "FN119: an issue-ahead step drafts at width >= 2 with a chain of >= 2 ({n} sequences in \
         bucket {bucket}, w {w}, {} blocks, chain {kn})",
        ah.blocks.len()
    );
    let half = VOCAB as usize / 2;
    let seam = rk
        .dpick
        .as_ref()
        .context("the device pick")?
        .out
        .addr()
        .as_u64();
    let woff = super::dstep::win_off(rk.pair.mb);
    let r = rk.rank as usize;
    let win_own = rk.pair.st[r].scratch + woff;
    let win_peer = rk.pair.st[1 - r].scratch + woff;
    let raw = {
        let o = rk.ov112.as_ref().context("FN112's staging")?;
        o.stg.addr() as u64 + o.lay.raw_off as u64
    };
    // 1. the row / sequence blocks -> the card (one async copy from the pinned staging)
    {
        let a = rk.ahead.as_mut().context("FN119: the ahead")?;
        a.c.issued += 1;
        let b = a.stg.as_mut_slice();
        b[..AheadRt::OUT].fill(0);
        for (row, blk) in ah.blocks.iter().enumerate() {
            let mut blk = *blk;
            blk[5] = row as u32;
            for (q, x) in blk.iter().enumerate() {
                let o = AheadRt::RP + row * 32 + 4 * q;
                b[o..o + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        for (i, d) in drafts.iter().enumerate() {
            let mut sq = [0u32; 8];
            sq[0] = (i * wu) as u32;
            sq[1] = w;
            sq[2] = (wu - 1) as u32;
            for (q, &t) in d.iter().take(5).enumerate() {
                sq[3 + q] = t;
            }
            for (q, x) in sq.iter().enumerate() {
                let o = AheadRt::SQ + i * 32 + 4 * q;
                b[o..o + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        rk.hip.run_batch(&[Job::MemcpyPinned {
            dev: &a.dev,
            dev_offset: 0,
            host: &a.stg,
            host_offset: 0,
            bytes: AheadRt::OUT as u64,
            kind: saber_hip::MemcpyKind::HostToDevice,
            stream: &rk.dev.stream,
        }])?;
    }
    let (base, ctr) = {
        let a = rk.ahead.as_ref().context("FN119: the ahead")?;
        (a.dev.addr().as_u64(), a.ctr.addr().as_u64())
    };
    let (rp, sq, out, rowout) = (
        base + AheadRt::RP as u64,
        base + AheadRt::SQ as u64,
        base + AheadRt::OUT as u64,
        base + AheadRt::ROWOUT as u64,
    );
    // 2. the seam to the peer, the walk, the refused rows' backup, the result read back
    ahead_launch(
        rk,
        "fn119a_push",
        [1, 1, 1],
        256,
        &[seam, win_peer, ctr],
        // FN119c: EXACTLY the rows the peer's walk reads and zeroes (n x w): a pad row pushed and
        // never read stays nonzero in its parity slot and a later step with more sequences takes
        // it as landed (h8: two b8 walks merged a stale peer seam -> FATAL by the host's check)
        &[super::dstep::push_rows(n, wu) as i32, AHEAD_ROWS as i32],
    )?;
    ahead_launch(
        rk,
        "fn119a_walk",
        [n as u32, 1, 1],
        32 * super::exec::WMAX,
        &[seam, win_own, ctr, rp, sq, out, rowout],
        &[n as i32, AHEAD_ROWS as i32],
    )?;
    ahead_launch(
        rk,
        "fn119a_backup",
        [rows as u32, 8, 1],
        256,
        &[rowout, lg, raw],
        &[(half * 2 / 16) as i32],
    )?;
    {
        let a = rk.ahead.as_ref().context("FN119: the ahead")?;
        rk.hip.run_batch(&[Job::MemcpyPinned {
            dev: &a.dev,
            dev_offset: AheadRt::OUT as u64,
            host: &a.stg,
            host_offset: AheadRt::OUT as u64,
            bytes: (AheadRt::END - AheadRt::OUT) as u64,
            kind: saber_hip::MemcpyKind::DeviceToHost,
            stream: &rk.dev.stream,
        }])?;
        rk.hip.record_event(&a.ev, &rk.dev.stream)?;
    }
    // FN119f: the first round's work between the walk and the round (the K-side front, the END
    // node's placement, the last prompt row's stream moved to row 0)
    if let Some(h) = pre_round.as_mut() {
        h(rk, em)?;
    }
    // 3. the drafter round's acc-free inputs + the chain variants (one per accepted count)
    let mut pp = pos0.to_vec();
    pp.resize(bucket as usize, 0);
    let one = vec![1u32; bucket as usize];
    let zero = vec![0u32; bucket as usize];
    let band = step_band(rk, &pp);
    let (mut tb, mut offs) = band_tables(rk, &pp, w, &one, &zero, rk.pools.pool_w, n, band);
    kv_extend(rk, &mut tb, &mut offs, &pp, w, n, band.1)?;
    let mut cc = ChunkConsts::at(
        rk.dev.consts.addr().as_u64(),
        &offs,
        bucket * w,
        0,
        2,
        rk.pools.nbs,
        rk.dev.cs.addr().as_u64(),
    );
    cc.max_ctx = band.0;
    cc.bt_nb = band.1;
    cc.bar = rk.bar.addr().as_u64();
    cc.b = bucket;
    cc.w = w;
    cc.qsa_short = saber_kernels::fn19::short_rows(&pp, w);
    // the chain's KV step law ONCE over every variant's positions (a sequence's pos0 + 1 ..
    // pos0 + w - 1 + kn - 1)
    if let Some(k) = rk.pools.kvt.as_ref() {
        let seqs: Vec<(u32, u32, u32)> = (0..n)
            .map(|i| (i as u32, pos0[i] + 1, pos0[i] + w + kn - 1))
            .collect();
        k.borrow_mut().begin(rk.hip, &rk.dev.stream, &seqs, false)?;
    }
    // one chain-input build per accepted-count vector (sequence i at acc[i])
    let mut key0: Option<((u32, u32), Vec<bool>)> = None;
    let mut same = true;
    let mut first: Option<(Vec<(Vec<u8>, Vec<u32>)>, Vec<(&'static str, usize)>)> = None;
    let mut build = |rk: &Rank<'_, '_>, acc: &[u32]| -> anyhow::Result<Vec<u8>> {
        let dpos: Vec<u32> = (0..n).map(|i| pos0[i] + acc[i]).collect();
        let (tabs, toffs) = chain_tables_at(rk, &dpos, n, bucket, kn, false)?;
        let key = (
            step_band(rk, &tabs[0].1),
            tabs.iter()
                .map(|(_, p)| saber_kernels::fn19::short_rows(p, 1))
                .collect::<Vec<bool>>(),
        );
        match &key0 {
            None => key0 = Some(key),
            Some(k) => same &= *k == key,
        }
        let sel1: Vec<i32> = (0..bucket as usize)
            .map(|i| {
                if i < n {
                    (i as u32 * w + acc[i]) as i32
                } else {
                    -1
                }
            })
            .collect();
        // LR-N: a prompt's first round (width 1) has no coupled rows
        let b = chain_input_bytes(rk, &sel1, n, &tabs, (w > 1).then_some(acc))?;
        if first.is_none() {
            first = Some((tabs, toffs));
        }
        Ok(b)
    };
    // the uniform variants (every sequence at a), and at b > 1 the single-sequence ones (sequence s
    // at a, every other at 0): the words each sequence owns
    let tbuild = Instant::now();
    let mut vars: Vec<Vec<u8>> = Vec::with_capacity(wu);
    for a in 0..w {
        vars.push(build(rk, &vec![a; n])?);
    }
    let mut runs: Option<Vec<[u32; 3]>> = None;
    if n > 1 {
        let mut single: Vec<Vec<Vec<u8>>> = Vec::with_capacity(n);
        for s in 0..n {
            let mut v = Vec::with_capacity(wu - 1);
            for a in 1..w {
                let mut acc = vec![0u32; n];
                acc[s] = a;
                v.push(build(rk, &acc)?);
            }
            single.push(v);
        }
        runs = super::dstep::owner_runs(&vars[0], &single);
        // the composition law on the uniform vectors (every sequence at a == variant a)
        if let Some(r) = runs.as_ref() {
            let c = super::dstep::ComposeChk {
                pos0: pos0.to_vec(),
                vars: vars.clone(),
                runs: r.clone(),
            };
            if (0..w).any(|a| c.compose(&vec![a; n]) != vars[a as usize]) {
                runs = None;
            }
        }
    }
    let _ = build;
    if n > 1 && runs.is_none() {
        same = false;
    }
    // FN119c: the issue-side builds' host time (behind the verify; `fnahead119`'s end line)
    if let Some(a) = rk.ahead.as_mut() {
        let ms = tbuild.elapsed().as_secs_f64() * 1e3;
        if n > 1 {
            a.c.build_multi_ms.add(ms);
        } else {
            a.c.build_one_ms.add(ms);
        }
    }
    if !same {
        // a variant crosses a band / sparse edge: today's round at the commit
        let a = rk.ahead.as_mut().context("FN119: the ahead")?;
        a.c.no_round += 1;
        a.issued = Some(Issued { kn, round: false });
        return Ok(());
    }
    let (tabs0, toffs0) = first.context("FN119: the chain variants")?;
    if w > 1 {
        couple_count(rk, n);
    }
    // 4. the pass-1 tables -> the constant region (async, behind the verify that reads them), the
    // variants -> the pinned region; the walk's acc picks one into the chain's inputs
    let (inb_dev, inb_len) = {
        let dc = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .context("the device chain")?;
        (dc.inb.addr().as_u64(), vars[0].len())
    };
    {
        let a = rk.ahead.as_mut().context("FN119: the ahead")?;
        ensure!(
            tb.len() <= 4 << 20
                && vars
                    .iter()
                    .all(|v| v.len() == inb_len && v.len() <= a.var_stride),
            "FN119: the round's inputs past the staging"
        );
        a.cs.as_mut_slice()[..tb.len()].copy_from_slice(&tb);
        let stride = a.var_stride;
        let roff = a.runs_off;
        let vb = a.var.as_mut_slice();
        for (i, v) in vars.iter().enumerate() {
            vb[i * stride..i * stride + v.len()].copy_from_slice(v);
        }
        if let Some(r) = runs.as_ref() {
            ensure!(
                r.len() * 12 <= super::dstep::RUNS_BYTES,
                "FN119: {} composition runs past the region",
                r.len()
            );
            for (k, run) in r.iter().enumerate() {
                for (q, x) in run.iter().enumerate() {
                    let o = roff + k * 12 + 4 * q;
                    vb[o..o + 4].copy_from_slice(&x.to_le_bytes());
                }
            }
        }
        rk.hip.run_batch(&[Job::MemcpyPinned {
            dev: &rk.dev.consts,
            dev_offset: 0,
            host: &a.cs,
            host_offset: 0,
            bytes: tb.len() as u64,
            kind: saber_hip::MemcpyKind::HostToDevice,
            stream: &rk.dev.stream,
        }])?;
    }
    let (var_addr, stride, roff) = {
        let a = rk.ahead.as_ref().context("FN119: the ahead")?;
        (a.var.addr() as u64, a.var_stride, a.runs_off)
    };
    if let Some(r) = runs.as_ref() {
        // FN119c (b > 1): variant 0, then each sequence's words from the variant of its count
        let n16 = inb_len.div_ceil(16);
        ahead2_launch(
            rk,
            "fn119b_copy",
            [n16.div_ceil(256) as u32, 1, 1],
            &[var_addr, inb_dev],
            &[n16 as i32],
        )?;
        if !r.is_empty() {
            ahead2_launch(
                rk,
                "fn119b_compose",
                [r.len() as u32, 1, 1],
                &[out, var_addr, inb_dev, var_addr + roff as u64],
                &[(stride / 4) as i32],
            )?;
        }
        let a = rk.ahead.as_mut().context("FN119: the ahead")?;
        a.c.multi += 1;
        a.chk = Some(super::dstep::ComposeChk {
            pos0: pos0.to_vec(),
            vars: vars.clone(),
            runs: r.clone(),
        });
    } else {
        ahead_launch(
            rk,
            "fn119a_sel",
            [1, 1, 1],
            256,
            &[out, var_addr, inb_dev],
            &[(stride / 16) as i32, inb_len.div_ceil(16) as i32],
        )?;
    }
    // 5. the pass-1 embedding rows: this rank's half (UVA) or -0.0, the exact reduce, -> rk.emb
    let embh = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("the device chain")?
        .embh
        .addr();
    buf_need(rk, u32::try_from(rows)?, "the embedding rows")?;
    em.arena.new_layer();
    let pay = em.arena.take(rows as u64 * H * 2, Seed::None)?;
    em.run(
        "fn119a_emb",
        [rows as u64, 1, 1],
        256,
        vec![
            Arg::P(out),
            Arg::P(rowout),
            Arg::P(embh),
            Arg::P(pay),
            Arg::I(w as i32),
            Arg::I(n as i32),
            Arg::I(half as i32),
            Arg::I(rk.rank as i32),
        ],
        Group::Head,
    )?;
    // FN140: a decode-form reduce -- exact by FN78's law at <= 64 rows (b8 x w4 = 32 rows = 160
    // KiB is past the payload band; the pre-FN140 payload guard refused it: FN136 h3's FATAL)
    let red = rk.pair.reduce_decode_form(
        em,
        pay,
        rows as u64,
        "FN119: the pass-1 embeddings must ride the EXACT reduce",
    )?;
    rk.dev.exec(&rk.table, em)?;
    let abase = rk.dev.base();
    rk.hip.run_batch(&[Job::MemcpyD2d {
        dst: &rk.emb.own,
        dst_offset: rk.emb.off,
        src: &rk.dev.arena,
        src_offset: red - abase,
        bytes: rows as u64 * H * 2,
        stream: &rk.dev.stream,
    }])?;
    // 6. pass 1 (its gather on the card) and the chain
    let sv = rk.pools.batch(LAYERS as usize, 0);
    let hid = rk.mtp.as_ref().context("mtp")?.hid.addr().as_u64();
    let emb = rk.emb.addr().as_u64();
    let tail = tail_args(rk, IN_SEL1, bucket)?;
    emb_count(rk, 1, bucket);
    rk.ahead_gather = Some((out, w, n as u32));
    mtp_pass(
        rk,
        em,
        bucket * w,
        emb,
        hid,
        cc,
        sv,
        &[],
        true,
        None,
        Some(tail),
    )?;
    ensure!(
        rk.ahead_gather.is_none(),
        "FN119: pass 1 did not take the device gather"
    );
    // FN119f: the END node's drafter part, stream-only, between pass 1 and the chain
    if let Some(h) = mid {
        h(rk)?;
    }
    chain_passes(rk, em, recs, &tabs0, &toffs0, n, bucket, 1)?;
    let a = rk.ahead.as_mut().context("FN119: the ahead")?;
    a.c.rounds += 1;
    a.issued = Some(Issued { kn, round: true });
    Ok(())
}

/// FN119 (`ahead`): the commit's half of a round issued ahead -- what today's round does on the
/// host while its chain runs (the deferred tier end, FN117b's S2, FN58b's PLE pre-hook), the
/// chain's ids read back, the mover released.
fn ahead_finish(
    rk: &mut Rank<'_, '_>,
    n: usize,
    bucket: u32,
    kn: u32,
    mut pre: Option<PreHook<'_>>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let t = Instant::now();
    tier_flush(rk)?;
    rd(rk, "rd_flush", t);
    let t = Instant::now();
    hand_s2_build(rk)?;
    rd(rk, "rd_s2", t);
    if let (Some(h), Some(ov)) = (pre.as_mut(), rk.ple_ov.as_mut()) {
        let t = Instant::now();
        let (ms, r, hits, nvme) = h()?;
        ov.pre.0 += ms;
        ov.pre.1 += r;
        ov.pre.2 += hits;
        ov.pre.3 += nvme;
        *rk.rd_t.entry("rd_pre").or_insert(0.0) += t.elapsed().as_secs_f64() * 1e3;
    }
    fn30b_probe(rk, 2)?;
    let t = Instant::now();
    let ids = finish_chain(rk, kn)?;
    rd(rk, "rd_ids", t);
    fn30b_resolve(rk)?;
    let pm = rk.mtp.as_ref().context("mtp")?.pass_ms.clone();
    rk.diag.passes.push((bucket, pm));
    let t = Instant::now();
    rk.tier
        .mover
        .release(&rk.dev.stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rd(rk, "rd_release", t);
    let stride = saber_kernels::fn22::IDS_STRIDE as usize;
    Ok((0..n)
        .map(|i| (0..kn as usize).map(|j| ids[j * stride + i]).collect())
        .collect())
}

/// FN119d (`next`): the commit's half of a round issued ahead WITHOUT its ids -- what
/// [`ahead_finish`] runs while the chain runs (the deferred tier end, S2, the PLE pre-hook), the
/// chain's ids queued for a read-back right behind the round (async into pinned memory, its
/// event), the mover released; the next step reads the ids ([`resolve_ids`]) only before the
/// drafts' PLE rows.
fn ahead_defer(rk: &mut Rank<'_, '_>, kn: u32, mut pre: Option<PreHook<'_>>) -> anyhow::Result<()> {
    let t = Instant::now();
    tier_flush(rk)?;
    rd(rk, "rd_flush", t);
    let t = Instant::now();
    hand_s2_build(rk)?;
    rd(rk, "rd_s2", t);
    if let (Some(h), Some(ov)) = (pre.as_mut(), rk.ple_ov.as_mut()) {
        let t = Instant::now();
        let (ms, r, hits, nvme) = h()?;
        ov.pre.0 += ms;
        ov.pre.1 += r;
        ov.pre.2 += hits;
        ov.pre.3 += nvme;
        *rk.rd_t.entry("rd_pre").or_insert(0.0) += t.elapsed().as_secs_f64() * 1e3;
    }
    fn30b_probe(rk, 2)?;
    queue_ids(rk, kn)?;
    let t = Instant::now();
    rk.tier
        .mover
        .release(&rk.dev.stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rd(rk, "rd_release", t);
    Ok(())
}

/// FN119d: the chain's ids queued for a read-back right behind the round (async into the pinned
/// buffer) and their event -- the host reads them later ([`resolve_ids`]), never the stream.
fn queue_ids(rk: &mut Rank<'_, '_>, kn: u32) -> anyhow::Result<()> {
    let stride = saber_kernels::fn22::IDS_STRIDE as usize;
    let src = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("chain")?
        .ids
        .addr();
    let e = rk
        .ahead
        .as_mut()
        .and_then(|a| a.early.as_mut())
        .context("FN119d: the early buffers")?;
    let bytes = kn as usize * stride * 4;
    ensure!(
        bytes <= e.ids.len(),
        "FN119d: {kn} chain passes past the ids read-back"
    );
    rk.hip.memcpy_d2h_addr(
        e.ids
            .slice_mut(0, bytes)
            .context("FN119d: the ids read-back")?,
        src,
        &rk.dev.stream,
    )?;
    rk.hip.record_event(&e.ids_ev, &rk.dev.stream)?;
    Ok(())
}

/// FN119d: the pending round's ids (its read-back's event, then the pinned bytes) and its passes'
/// device times -- [`finish_chain`]'s half for a round [`ahead_defer`] left.
fn resolve_ids(rk: &mut Rank<'_, '_>, kn: u32, bucket: u32) -> anyhow::Result<Vec<u32>> {
    let t = Instant::now();
    let stride = saber_kernels::fn22::IDS_STRIDE as usize;
    let raw: Vec<u32> = {
        let e = rk
            .ahead
            .as_ref()
            .and_then(|a| a.early.as_ref())
            .context("FN119d: the early buffers")?;
        rk.hip.sync_event(&e.ids_ev)?;
        e.ids.as_slice()[..kn as usize * stride * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    seg(rk, "ids", t);
    let evs = std::mem::take(&mut rk.mtp.as_mut().context("mtp")?.evs);
    for (a, b) in &evs {
        let ms = f64::from(rk.hip.elapsed_ms(a, b)?);
        rk.mtp.as_mut().context("mtp")?.pass_ms.push(ms);
    }
    rd(rk, "rd_ids", t);
    fn30b_resolve(rk)?;
    let pm = rk.mtp.as_ref().context("mtp")?.pass_ms.clone();
    rk.diag.passes.push((bucket, pm));
    Ok(raw)
}

/// FN119d: the pending round's ids read -- the commit's tail it skipped: each sequence's drafts,
/// the lookup's merge, FN121's candidate prefetch (not for an early step: its rows were just
/// gathered), the round's records and counters.
fn round_post(
    rk: &mut Rank<'_, '_>,
    st: &mut SrvState,
    ple: &mut PleTier,
    ph: &PleHash,
    raw: &[u32],
    early: bool,
) -> anyhow::Result<()> {
    let rp = st.rpend.take().context("FN119d: no round pending")?;
    let stride = saber_kernels::fn22::IDS_STRIDE as usize;
    let cap = st.law.as_ref().map_or(0, |l| l.kmax()) as usize;
    for i in 0..rp.n {
        if let Some(lv) = st.slots[i].as_mut() {
            lv.drafts = (0..rp.kn as usize).map(|j| raw[j * stride + i]).collect();
            lookup_merge(lv, cap, &mut st.ms.lk);
        }
    }
    let tpf = Instant::now();
    let pf = if early {
        0
    } else {
        ple_prefetch_cand(rk, ple, ph, st.slots[..rp.n].iter().flatten())?
    };
    let pf_ms = tpf.elapsed().as_secs_f64() * 1e3;
    let parts: Vec<(&'static str, f64)> = rk
        .rd_t
        .iter()
        .map(|(k, v)| (*k, v - rp.rd0.get(k).copied().unwrap_or(0.0)))
        .filter(|(_, v)| *v != 0.0)
        .collect();
    let dev: f64 = rk.mtp.as_ref().map_or(0.0, |m| m.pass_ms.iter().sum());
    rk.f121.round(
        ple,
        saber_mem::ple_tier::unix_us(rp.tm),
        rp.tm.elapsed().as_secs_f64() * 1e3,
        dev,
        &parts,
        pf,
        pf_ms,
    )?;
    let mt = rk.mtp.as_ref().context("mtp")?;
    if !rp.first {
        st.ms.rounds += 1;
    }
    st.ms.passes += mt.pass_ms.len() as u64;
    st.ms.dev_ms += mt.pass_ms.iter().sum::<f64>();
    let round_ms = rp.tm.elapsed().as_secs_f64() * 1e3;
    st.ms.host_ms += round_ms;
    if rp.n == 1 && !rp.first {
        if let (Some(c), Some(it)) = (st.ms.depth.as_mut(), st.ms.it.as_mut()) {
            c.time_round(rp.kn, round_ms);
            it.2 = round_ms;
        }
    }
    Ok(())
}

/// FN119d: resolve the pending round before a command the early path does not take (`form`:
/// a decode step of another shape; else another command).
fn resolve_pending(
    rk: &mut Rank<'_, '_>,
    st: &mut SrvState,
    ple: &mut PleTier,
    ph: &PleHash,
    form: bool,
) -> anyhow::Result<()> {
    let Some((kn, bucket)) = st.rpend.as_ref().map(|p| (p.kn, p.bucket)) else {
        return Ok(());
    };
    let raw = resolve_ids(rk, kn, bucket)?;
    round_post(rk, st, ple, ph, &raw, false)?;
    if let Some(a) = rk.ahead.as_mut() {
        if form {
            a.c.next_res_form += 1;
        } else {
            a.c.next_res_cmd += 1;
        }
    }
    Ok(())
}

/// FN119d: the next verify's input embedding rows on the card behind the round (the walk's result
/// block: each sequence's committed token; the chain's ids: its drafts) -> this rank's half or
/// -0.0, the certified EXACT reduce, -> `rk.emb` (== the host gather's rows: x + -0.0 == x).
fn early_emb(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    n: usize,
    bucket: u32,
    w: u32,
) -> anyhow::Result<()> {
    use saber_model::qwen4_exp::layer_graph::Group;
    let rows = bucket as usize * w as usize;
    let half = VOCAB as usize / 2;
    let (embh, ids) = {
        let dc = rk
            .mtp
            .as_ref()
            .and_then(|m| m.chain.as_ref())
            .context("the device chain")?;
        (dc.embh.addr(), dc.ids.addr().as_u64())
    };
    let out = rk
        .ahead
        .as_ref()
        .context("FN119: the ahead")?
        .dev
        .addr()
        .as_u64()
        + super::dstep::AheadRt::OUT as u64;
    buf_need(rk, u32::try_from(rows)?, "the embedding rows")?;
    em.arena.new_layer();
    let pay = em.arena.take(rows as u64 * H * 2, Seed::None)?;
    em.run(
        "fn119d_vemb",
        [rows as u64, 1, 1],
        256,
        vec![
            Arg::P(out),
            Arg::P(ids),
            Arg::P(embh),
            Arg::P(pay),
            Arg::I(w as i32),
            Arg::I(n as i32),
            Arg::I(half as i32),
            Arg::I(rk.rank as i32),
            Arg::I(saber_kernels::fn22::IDS_STRIDE as i32),
        ],
        Group::Head,
    )?;
    // FN140: a decode-form reduce -- exact by FN78's law at <= 64 rows (see `ahead_issue`'s)
    let red = rk.pair.reduce_decode_form(
        em,
        pay,
        rows as u64,
        "FN119d: the verify's embeddings must ride the EXACT reduce",
    )?;
    rk.dev.exec(&rk.table, em)?;
    let abase = rk.dev.base();
    rk.hip.run_batch(&[Job::MemcpyD2d {
        dst: &rk.emb.own,
        dst_offset: rk.emb.off,
        src: &rk.dev.arena,
        src_offset: red - abase,
        bytes: rows as u64 * H * 2,
        stream: &rk.dev.stream,
    }])?;
    Ok(())
}

/// FN119d: the device embedding rows == the host gather's (the first `EARLY_EMB_CHECKS` early
/// steps of a boot; a D2H of the rows after the layers before the PLE layer ran), fail loud.
fn early_emb_check(
    rk: &mut Rank<'_, '_>,
    recs: &PackRecs,
    lives: &[Live],
    dloc: &[Vec<u32>],
    n: usize,
    b: usize,
    wu: usize,
) -> anyhow::Result<()> {
    let done = rk.ahead.as_ref().map_or(u64::MAX, |a| a.c.next_emb_checked);
    if done >= super::dstep::EARLY_EMB_CHECKS {
        return Ok(());
    }
    let toks: Vec<u32> = (0..b * wu)
        .map(|r| {
            let (i, t) = (r / wu, r % wu);
            if i >= n {
                0
            } else if t == 0 {
                lives[i].next
            } else {
                dloc[i][t - 1]
            }
        })
        .collect();
    let want = embed_rows(recs, &toks)?;
    let got = d2h_in(&rk.dev, &rk.emb.own, rk.emb.addr().as_u64(), want.len())?;
    ensure!(
        got == want,
        "FN119d: the card's verify embedding rows differ from the host gather (tokens {toks:?})"
    );
    if let Some(a) = rk.ahead.as_mut() {
        a.c.next_emb_checked += 1;
    }
    Ok(())
}

/// [`apply`] with FN113b's quiet stamp around it (a tree demotion's drive write waits for an
/// engine with no request in flight).
#[allow(clippy::too_many_arguments)]
fn applied(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    cmd: &Cmd,
) -> anyhow::Result<(Vec<RowOut>, Vec<Vec<f32>>)> {
    if let Some(pc) = st.pc.as_ref() {
        pcache::mark_busy(pc);
    }
    // FN119c: a commit's host half by phase when it runs long (the `rd_*` census keys, diffed)
    let watch = matches!(cmd, Cmd::Commit { .. });
    let rd0 = if watch {
        rk.rd_t.clone()
    } else {
        BTreeMap::new()
    };
    let t0 = Instant::now();
    let r = apply(rk, em, ple, ph, recs, cfg, f, gr, dp, st, cmd);
    if watch {
        slow_note(rk, cmd, &rd0, t0.elapsed().as_secs_f64() * 1e3);
    }
    if let Some(pc) = st.pc.as_ref() {
        pcache::mark_after(pc, st.slots.iter().any(Option::is_some));
    }
    r
}

/// FN119c: the commit applies of > 20 ms (both ranks), each with the census phases it
/// added (`rd_*`: the round's host parts, the PLE pre-hook, the chain's ids, the mover's release),
/// the first 64 a rank printed (`fnslow119 rank<r>:`), all counted.
fn slow_note(rk: &mut Rank<'_, '_>, cmd: &Cmd, rd0: &BTreeMap<&'static str, f64>, ms: f64) {
    if ms <= 20.0 {
        return;
    }
    let kind = if matches!(cmd, Cmd::Commit { .. }) {
        "commit"
    } else {
        "command"
    };
    let mut parts: Vec<(&'static str, f64)> = rk
        .rd_t
        .iter()
        .map(|(k, v)| (*k, v - rd0.get(k).copied().unwrap_or(0.0)))
        .filter(|x| x.1 > 0.5)
        .collect();
    parts.sort_by(|a, b| b.1.total_cmp(&a.1));
    rk.slow119 += 1;
    if rk.slow119 <= 64 {
        println!(
            "fnslow119 rank{}: {kind} {ms:.1} ms (ahead {}): {}",
            rk.rank,
            rk.ahead.is_some(),
            parts
                .iter()
                .map(|(k, v)| format!("{k} {v:.1}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

/// Rank 0 sends `cmd` to rank 1, then both apply it.
#[allow(clippy::too_many_arguments)]
fn lead(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    cmd: &Cmd,
) -> anyhow::Result<(Vec<RowOut>, Vec<Vec<f32>>)> {
    let fixed;
    let cmd = match (cmd, rk.first_chk.take()) {
        (Cmd::First { slot, tok, .. }, Some((t, hd))) => {
            fixed = Cmd::First {
                slot: *slot,
                tok: *tok,
                redo: first_check(rk, t, hd)?,
            };
            &fixed
        }
        (_, x) => {
            rk.first_chk = x;
            cmd
        }
    };
    let tx = Instant::now();
    let _ = rk.peer.exchange().xchg_bytes("fn.cmd", &serve::enc(cmd)?)?;
    let xw = tx.elapsed().as_secs_f64() * 1e3;
    if xw > 20.0 {
        // FN119c: rank 1 reached the command exchange late (its previous apply ran long)
        *rk.rd_t.entry("x_cmd_wait").or_insert(0.0) += xw;
        rk.slow119 += 1;
        if rk.slow119 <= 64 {
            println!("fnslow119 rank0: command exchange waited {xw:.1} ms for rank 1");
        }
    }
    applied(rk, em, ple, ph, recs, cfg, f, gr, dp, st, cmd)
}

/// FN119f, rank 0: the card's walk of a prompt's first token (both ranks ran it) == the host's
/// draw `t` (fail loud), or refused (the round re-runs on the host's token: `redo`).
fn first_check(rk: &mut Rank<'_, '_>, t: u32, host_dense: bool) -> anyhow::Result<bool> {
    let a = rk.ahead.as_mut().context("FN119: the ahead")?;
    rk.hip.sync_event(&a.ev)?;
    let (fl, dacc, dtok) = a.result(0);
    ensure!(
        fl & super::dstep::AHEAD_TIMEOUT == 0,
        "FN119f: the peer's seam never landed in the window (flags {fl:#x})"
    );
    if fl == 0 {
        ensure!(
            !host_dense && dtok == [t] && dacc == 0,
            "FN119f: the device walk drew {dtok:?} (acc {dacc}), the host {t} (dense \
             {host_dense}): the twin's law is broken"
        );
        a.c.dev_equal += 1;
        Ok(false)
    } else {
        a.c.dev_refused += 1;
        Ok(true)
    }
}

/// FN14 / FN17: one scheduler batch (a prefill chunk or a decode step) through the commands both
/// ranks apply; its per-sequence results into `res`, what both ranks run after the reply into `post`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_batch(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    seq_slot: &mut BTreeMap<SeqId, u32>,
    sampler: &std::sync::Arc<saber_serve::sampler::Sampler>,
    batch: &serve::OwnedBatch,
    res: &mut Vec<saber_sched::SeqStep>,
    post: &mut Vec<Cmd>,
) -> anyhow::Result<()> {
    if batch.phase == Phase::Prefill {
        let s = batch.slots.first().context("an empty prefill batch")?;
        let slot = *seq_slot
            .get(&s.seq)
            .context("prefill of an unadmitted sequence")?;
        let (params, pen, rq) = serve::row_req(&s.sampling);
        let last = s.chunk.is_none_or(|c| c.last);
        // FN119f (`--fn-first-step dev`): the first token's row block (the request's key drawn
        // here; the host draws with the same one) -- the worker walks it behind the last chunk
        let mut samp = s.sampling.clone();
        let fah = if last
            && super::run::first_step_dev(f)
            && st.ds.next()
            && rk.ahead.as_ref().is_some_and(|a| a.early.is_some())
            && st.ready == 0
            && slot == 0
            && next_k(st) > 1
            && st.law.as_ref().is_some_and(|l| !l.dynamic)
            && rk
                .mtp
                .as_ref()
                .and_then(|m| m.chain.as_ref())
                .is_some_and(|c| c.check.is_none() && c.conf.is_none())
            && super::dstep::ahead_sampling_ok(&params, &pen)
        {
            if samp.params.seed.is_none() && !params.is_greedy() {
                samp.params.seed = Some(ahead_seed());
            }
            let key = samp.params.seed.map_or(0, |sd| {
                super::dstep::noise_key(sd, samp.position, samp.choice_index)
            });
            Some(super::dstep::AheadCmd {
                blocks: vec![super::dstep::row_block(&params, false, 0, key)],
                kn: next_k(st),
            })
        } else {
            None
        };
        let (outs, _) = lead(
            rk,
            em,
            ple,
            ph,
            recs,
            cfg,
            f,
            gr,
            dp,
            st,
            &Cmd::Prefill {
                slot,
                ids: s.input.clone(),
                req: rq,
                last,
                ah: fah,
            },
        )?;
        let fdev = rk.first_dev.is_some();
        if last {
            let mut tok = serve::sample(sampler, &outs[0], None, &params, &pen, &samp);
            let host_dense = tok.is_none();
            if tok.is_none() {
                // the prefill's logits: one row at its own address
                st.last_lg = rk.last_lg;
                st.last_rows = 1;
                let (_, d) = lead(
                    rk,
                    em,
                    ple,
                    ph,
                    recs,
                    cfg,
                    f,
                    gr,
                    dp,
                    st,
                    &Cmd::Dense {
                        rows: vec![0],
                        stage: fdev,
                    },
                )?;
                tok = serve::sample(
                    sampler,
                    &outs[0],
                    d.into_iter().next(),
                    &params,
                    &pen,
                    &samp,
                );
            }
            let (t, lp) = tok.context("sampled")?;
            // FN119f: the card's walk is checked against this draw when `Cmd::First` is sent
            rk.first_chk = fdev.then_some((t, host_dense));
            let to = st.ready as u32;
            lead(
                rk,
                em,
                ple,
                ph,
                recs,
                cfg,
                f,
                gr,
                dp,
                st,
                &Cmd::Ready { slot, to },
            )?;
            if slot != to {
                for v in seq_slot.values_mut() {
                    if *v == to {
                        *v = slot;
                    }
                }
                seq_slot.insert(s.seq, to);
            }
            if st.law.is_some() {
                post.push(Cmd::First {
                    slot: to,
                    tok: t,
                    redo: false,
                });
            }
            res.push(serve::seq_step(s.seq, Some(t), lp));
        } else {
            res.push(serve::seq_step(s.seq, None, None));
        }
    } else if st.law.is_some() {
        mtp_step(
            rk, em, ple, ph, recs, cfg, f, gr, dp, st, batch, seq_slot, sampler, res, post,
        )?;
    } else {
        // the ready set in pool-slot order; the scheduler's slots map onto it
        let n = st.ready;
        let mut toks = vec![0u32; n];
        let mut reqs = vec![RowReq::default(); n];
        let mut per = Vec::with_capacity(batch.slots.len());
        for s in &batch.slots {
            let slot = *seq_slot
                .get(&s.seq)
                .context("decode of an unadmitted sequence")? as usize;
            ensure!(slot < n, "decode of a sequence that is not ready");
            toks[slot] = *s.input.last().context("a decode slot feeds no token")?;
            let (params, pen, mut rq) = serve::row_req(&s.sampling);
            if st.ds.lazy() {
                super::dstep::shape(&mut rq, &params, &pen);
            }
            reqs[slot] = rq;
            per.push((slot, params, pen));
        }
        ensure!(
            per.len() == n,
            "the batch holds {} of {n} ready sequences",
            per.len()
        );
        let (outs, _) = lead(
            rk,
            em,
            ple,
            ph,
            recs,
            cfg,
            f,
            gr,
            dp,
            st,
            &Cmd::Decode {
                toks,
                reqs: reqs.clone(),
                w: 1,
                ah: None,
                cp: Vec::new(),
            },
        )?;
        let mut outs = outs;
        ds_fill(
            rk, em, ple, ph, recs, cfg, f, gr, dp, st, &mut outs, &reqs, 1, false,
        )?;
        let mut picked: Vec<Option<(u32, Option<SampledLogprobs>)>> = Vec::new();
        let mut dense_rows = Vec::new();
        for (i, (slot, params, pen)) in per.iter().enumerate() {
            let t = serve::sample(
                sampler,
                &outs[*slot],
                None,
                params,
                pen,
                &batch.slots[i].sampling,
            );
            if t.is_none() {
                dense_rows.push(*slot as u32);
            }
            picked.push(t);
        }
        if !dense_rows.is_empty() {
            let (_, d) = lead(
                rk,
                em,
                ple,
                ph,
                recs,
                cfg,
                f,
                gr,
                dp,
                st,
                &Cmd::Dense {
                    rows: dense_rows.clone(),
                    stage: false,
                },
            )?;
            for (i, (slot, params, pen)) in per.iter().enumerate() {
                if picked[i].is_none() {
                    let k = dense_rows
                        .iter()
                        .position(|&r| r as usize == *slot)
                        .context("dense row")?;
                    picked[i] = serve::sample(
                        sampler,
                        &outs[*slot],
                        Some(d[k].clone()),
                        params,
                        pen,
                        &batch.slots[i].sampling,
                    );
                }
            }
        }
        for (i, s) in batch.slots.iter().enumerate() {
            let (t, lp) = picked[i].take().context("sampled")?;
            res.push(serve::seq_step(s.seq, Some(t), lp));
        }
    }
    Ok(())
}

/// FN14 / FN17: one scheduler job on rank 0 of the serve plane -- the commands both ranks apply,
/// the pick, the reply; then what runs after the reply (FN16's drafter). The HTTP serve plane and
/// FN17's scripted schedules (`--fn-sched padgate`) drive the same code.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn serve_job(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    seq_slot: &mut BTreeMap<SeqId, u32>,
    sampler: &std::sync::Arc<saber_serve::sampler::Sampler>,
    steps: &mut u64,
    job: SrvJob,
) -> anyhow::Result<()> {
    let (job, img) = match job {
        SrvJob::AdmitImg(seq, tokens, fx) => (SrvJob::Admit(seq, tokens), Some(fx)),
        j => (j, None),
    };
    match job {
        SrvJob::Admit(seq, tokens) => {
            let slot = (st.ready..MAX_SEQS as usize)
                .find(|&i| st.slots[i].is_none())
                .context("no free pool slot (the scheduler admitted a ninth sequence)")?
                as u32;
            seq_slot.insert(seq, slot);
            lead(
                rk,
                em,
                ple,
                ph,
                recs,
                cfg,
                f,
                gr,
                dp,
                st,
                &Cmd::Zero { slot, tokens, img },
            )?;
        }
        SrvJob::AdmitImg(..) => unreachable!("mapped to Admit above"),
        SrvJob::Encode(grids, pixels, reply) => {
            // FN55b: the tower on both ranks (the pixels follow the command as `fn.pix`)
            st.pix = pixels.iter().flat_map(|x| x.to_le_bytes()).collect();
            st.enc = Err("not run".to_owned());
            lead(
                rk,
                em,
                ple,
                ph,
                recs,
                cfg,
                f,
                gr,
                dp,
                st,
                &Cmd::Encode { grids },
            )?;
            let defer = std::mem::take(&mut st.enc_defer);
            let _ = reply.send(std::mem::replace(&mut st.enc, Ok(0)).map(|e| (e, defer)));
        }
        SrvJob::Release(seq) => {
            if let Some(slot) = seq_slot.remove(&seq) {
                if (slot as usize) < st.ready {
                    let last = (st.ready - 1) as u32;
                    lead(
                        rk,
                        em,
                        ple,
                        ph,
                        recs,
                        cfg,
                        f,
                        gr,
                        dp,
                        st,
                        &Cmd::Leave { from: slot },
                    )?;
                    if slot != last {
                        for v in seq_slot.values_mut() {
                            if *v == last {
                                *v = slot;
                            }
                        }
                    }
                } else {
                    lead(
                        rk,
                        em,
                        ple,
                        ph,
                        recs,
                        cfg,
                        f,
                        gr,
                        dp,
                        st,
                        &Cmd::Free { slot },
                    )?;
                }
            }
        }
        SrvJob::Step(batch, reply) => {
            let mut res = Vec::with_capacity(batch.slots.len());
            // FN16: what both ranks run AFTER the reply (the drafter: F37's proposal after
            // the commit, off the step's latency)
            let mut post: Vec<Cmd> = Vec::new();
            run_batch(
                rk, em, ple, ph, recs, cfg, f, gr, dp, st, seq_slot, sampler, &batch, &mut res,
                &mut post,
            )?;
            *steps += 1;
            st.f75.reply = Some(Instant::now());
            let _ = reply.send(StepResult::new(res));
            for c in &post {
                lead(rk, em, ple, ph, recs, cfg, f, gr, dp, st, c)?;
            }
        }
        SrvJob::Mixed(dec, chunks, reply) => {
            // FN17: the unmixed commands of (the decode step, then each chunk step), in that
            // order, as ONE job with one reply. The decode's own post (MTP: the Commit) runs
            // before the chunks, whose last-chunk Ready moves the ready set; the chunks' posts
            // (MTP: the First passes) after the reply, as the unmixed steps run them.
            let mut rd = Vec::with_capacity(dec.slots.len());
            let mut pd: Vec<Cmd> = Vec::new();
            run_batch(
                rk, em, ple, ph, recs, cfg, f, gr, dp, st, seq_slot, sampler, &dec, &mut rd,
                &mut pd,
            )?;
            for c in &pd {
                lead(rk, em, ple, ph, recs, cfg, f, gr, dp, st, c)?;
            }
            // each chunk's own posts (MTP: its First pass) run right after it, before the next
            // chunk's prefill overwrites the target stream the pass reads (hold 4: seven packed
            // chunks with their First passes deferred to the end read the last chunk's stream)
            let mut rcs = Vec::with_capacity(chunks.len());
            for ch in &chunks {
                let mut r = Vec::with_capacity(1);
                let mut pc: Vec<Cmd> = Vec::new();
                run_batch(
                    rk, em, ple, ph, recs, cfg, f, gr, dp, st, seq_slot, sampler, ch, &mut r,
                    &mut pc,
                )?;
                for c in &pc {
                    lead(rk, em, ple, ph, recs, cfg, f, gr, dp, st, c)?;
                }
                rcs.push(StepResult::new(r));
            }
            *steps += 1;
            st.mixed_steps += 1;
            st.mixed_chunks += chunks.len() as u64;
            st.f75.reply = Some(Instant::now());
            let _ = reply.send((StepResult::new(rd), rcs));
        }
    }
    Ok(())
}

/// One sequence's record in a scripted schedule (FN17).
#[derive(Debug, Default, Clone)]
struct GenRec {
    toks: Vec<u32>,
    /// Per token: the sampled logprob + the top-5 (id, logprob), bytes as text.
    lps: Vec<String>,
    /// Per decode step: (drafts verified, accepted).
    acc: Vec<(u32, u32)>,
    /// Decode steps taken.
    steps: u32,
}

/// FN17's PAD-LAW GATE on the card (`--fn-sched padgate`, rank 0; rank 1 follows the commands):
/// the serve plane's own job path (`serve_job`: Zero / Prefill / Ready / Decode / Commit / First /
/// Leave) driven by a script instead of the scheduler. Arm `alone`: the long prompt (> 2,048
/// tokens: two chunks, 2,048 + the rest) prefilled and decoded ALONE (b1). Arm `padded`: three
/// short prompts ready in pool slots 0..3; the long prompt admitted at slot 3 and its FIRST chunk
/// prefilled; the three decode `PADDED` steps at bucket 4 (pad row 3 -- FN14's law wrote slot 3,
/// the long prompt's); the three leave; the long prompt's second chunk, then it decodes alone. Arm
/// `alone2` repeats `alone` (the determinism floor). The long prompt's ids, logprobs and (MTP)
/// per-step acceptance must be equal across the three arms; `--fn-pad-control` must differ.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn padgate(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    st: &mut SrvState,
    prompts: &[Prompt],
    out: &Path,
) -> anyhow::Result<()> {
    const GEN: usize = 64;
    const PADDED: u32 = 8;
    let long = prompts
        .iter()
        .find(|p| p.ids.len() as u32 > MAX_ROWS)
        .context("--fn-sched padgate needs a prompt of > 2,048 tokens (--fn-long 3000)")?;
    let short: Vec<&Prompt> = prompts
        .iter()
        .filter(|p| p.ids.len() as u32 <= MAX_ROWS)
        .take(3)
        .collect();
    ensure!(
        short.len() == 3,
        "--fn-sched padgate needs 3 prompts of <= 2,048 tokens"
    );
    let kw = st.law.as_ref().map_or(1, |l| l.kmax() + 1);
    let sampler = std::sync::Arc::new(saber_serve::sampler::Sampler::new());
    let mut seq_slot: BTreeMap<SeqId, u32> = BTreeMap::new();
    let mut steps = 0u64;
    let mut gens: BTreeMap<u64, GenRec> = BTreeMap::new();
    let samp = |pos: usize| {
        let mut s = saber_sched::SlotSampling::greedy();
        s.params.logprobs = Some(5);
        s.position = pos as u64;
        s
    };
    macro_rules! job {
        ($j:expr) => {
            serve_job(
                rk,
                em,
                ple,
                ph,
                recs,
                cfg,
                f,
                gr,
                dp,
                st,
                &mut seq_slot,
                &sampler,
                &mut steps,
                $j,
            )
        };
    }
    macro_rules! step {
        ($b:expr) => {{
            let (tx, rx) = std::sync::mpsc::channel();
            job!(SrvJob::Step($b, tx))?;
            let r: StepResult = rx.recv().context("the step's reply")?;
            record!(r);
        }};
    }
    macro_rules! record {
        ($r:expr) => {{
            let r: StepResult = $r;
            for q in r.seqs {
                let g = gens.entry(q.seq_id.0).or_default();
                g.toks.extend_from_slice(&q.tokens);
                g.acc.push((q.draft_tokens, q.accepted_tokens));
                if let Some(l) = &q.logprobs {
                    for x in l {
                        g.lps.push(format!(
                            "{}:{:08x}|{}",
                            x.sampled.token_id,
                            x.sampled.logprob.to_bits(),
                            x.top
                                .iter()
                                .map(|t| format!("{}:{:08x}", t.token_id, t.logprob.to_bits()))
                                .collect::<Vec<_>>()
                                .join(",")
                        ));
                    }
                }
            }
        }};
    }
    // one chunk of a prompt (`a..b` of its ids) as the scheduler's one-slot prefill step
    macro_rules! chunk_batch {
        ($seq:expr, $ids:expr, $a:expr, $b:expr) => {{
            let ids: &[u32] = $ids;
            let (a, b): (usize, usize) = ($a, $b);
            serve::OwnedBatch {
                phase: Phase::Prefill,
                slots: vec![serve::OwnedSlot {
                    seq: SeqId($seq),
                    input: ids[a..b].to_vec(),
                    chunk: Some(saber_sched::PrefillChunk {
                        prompt_tokens: (b - a) as u32,
                        tokens_done_before: a as u32,
                        last: b == ids.len(),
                        next_prompt_token: ids.get(b).copied(),
                    }),
                    sampling: samp(0),
                    verify_width: 0,
                }],
            }
        }};
    }
    macro_rules! chunk {
        ($seq:expr, $ids:expr, $a:expr, $b:expr) => {{
            step!(chunk_batch!($seq, $ids, $a, $b));
        }};
    }
    // one decode step of these sequences (the scheduler's width law: none on a first step)
    macro_rules! decode {
        ($seqs:expr) => {{
            let seqs: &[u64] = $seqs;
            let slots: Vec<serve::OwnedSlot> = seqs
                .iter()
                .map(|&q| {
                    let g = gens.get(&q).cloned().unwrap_or_default();
                    serve::OwnedSlot {
                        seq: SeqId(q),
                        input: vec![*g.toks.last().unwrap_or(&0)],
                        chunk: None,
                        sampling: samp(g.toks.len()),
                        verify_width: if g.steps == 0 { 0 } else { kw },
                    }
                })
                .collect();
            let before: Vec<usize> = seqs
                .iter()
                .map(|q| gens.get(q).map_or(0, |g| g.toks.len()))
                .collect();
            step!(serve::OwnedBatch {
                phase: Phase::Decode,
                slots,
            });
            for (i, q) in seqs.iter().enumerate() {
                let g = gens.entry(*q).or_default();
                g.steps += 1;
                ensure!(g.toks.len() > before[i], "a decode step committed nothing");
            }
        }};
    }
    // FN17: one MIXED step -- these sequences' decode + one chunk, as the serve plane's one job
    macro_rules! mixed {
        ($seqs:expr, $seq:expr, $ids:expr, $a:expr, $b:expr) => {{
            let seqs: &[u64] = $seqs;
            let slots: Vec<serve::OwnedSlot> = seqs
                .iter()
                .map(|&q| {
                    let g = gens.get(&q).cloned().unwrap_or_default();
                    serve::OwnedSlot {
                        seq: SeqId(q),
                        input: vec![*g.toks.last().unwrap_or(&0)],
                        chunk: None,
                        sampling: samp(g.toks.len()),
                        verify_width: if g.steps == 0 { 0 } else { kw },
                    }
                })
                .collect();
            let before: Vec<usize> = seqs
                .iter()
                .map(|q| gens.get(q).map_or(0, |g| g.toks.len()))
                .collect();
            let ch = chunk_batch!($seq, $ids, $a, $b);
            let (tx, rx) = std::sync::mpsc::channel();
            job!(SrvJob::Mixed(
                serve::OwnedBatch {
                    phase: Phase::Decode,
                    slots,
                },
                vec![ch],
                tx
            ))?;
            let (rd, rcs): (StepResult, Vec<StepResult>) =
                rx.recv().context("the mixed step's reply")?;
            record!(rd);
            for r in rcs {
                record!(r);
            }
            for (i, q) in seqs.iter().enumerate() {
                let g = gens.entry(*q).or_default();
                g.steps += 1;
                ensure!(g.toks.len() > before[i], "a mixed step committed nothing");
            }
        }};
    }
    let cut = MAX_ROWS as usize;
    let lids = &long.ids;
    let mut arms = serde_json::Map::new();
    if f.fn_sched == "mixgate" {
        // FN17's MIXED-STEP GATE: arrivals one a step (the long prompt 4th: two chunks), every
        // step decodes the ready set; arm `unmixed` sends the decode step then the chunk step
        // (two jobs), arm `mixed` ONE mixed job. Every sequence's ids + logprob bytes must agree.
        const MGEN: usize = 24;
        let mut order: Vec<&Prompt> = prompts
            .iter()
            .filter(|p| p.ids.len() as u32 <= MAX_ROWS)
            .take(7)
            .collect();
        order.insert(3.min(order.len()), long);
        for (ai, arm) in ["unmixed", "mixed"].into_iter().enumerate() {
            let t0 = Instant::now();
            let base = 1000 * (ai as u64 + 1);
            let mut running: Vec<u64> = Vec::new();
            let mut queue: Vec<(u64, &Prompt, usize)> = order
                .iter()
                .enumerate()
                .map(|(i, p)| (base + i as u64, *p, 0usize))
                .collect();
            queue.reverse();
            let mut steps_run = 0u32;
            let mut buckets = Vec::new();
            loop {
                let head = queue.last().copied();
                if let Some((q, _, 0)) = head {
                    job!(SrvJob::Admit(SeqId(q), 0))?;
                }
                let dec = running.clone();
                buckets.push(bucket_for(dec.len().max(1)));
                match head {
                    Some((q, p, a)) => {
                        let b = (a + cut).min(p.ids.len());
                        if dec.is_empty() {
                            chunk!(q, &p.ids, a, b);
                        } else if arm == "mixed" {
                            mixed!(&dec, q, &p.ids, a, b);
                        } else {
                            decode!(&dec);
                            chunk!(q, &p.ids, a, b);
                        }
                        queue.pop();
                        if b == p.ids.len() {
                            running.push(q);
                        } else {
                            queue.push((q, p, b));
                        }
                    }
                    None => {
                        if running
                            .iter()
                            .all(|q| gens.get(q).map_or(0, |g| g.toks.len()) >= MGEN)
                        {
                            break;
                        }
                        decode!(&dec);
                    }
                }
                steps_run += 1;
            }
            for &q in &running {
                job!(SrvJob::Release(SeqId(q)))?;
            }
            arms.insert(
                arm.into(),
                json!({"steps": steps_run, "buckets": buckets, "wall_s": t0.elapsed().as_secs_f64(),
                    "seqs": order.iter().enumerate().map(|(i, p)| {
                        let g = gens.get(&(base + i as u64)).cloned().unwrap_or_default();
                        json!({"id": p.id, "tokens": p.ids.len(), "gen": g.toks, "lps": g.lps, "acc": g.acc})
                    }).collect::<Vec<_>>()}),
            );
        }
        let (u, m) = (&arms["unmixed"]["seqs"], &arms["mixed"]["seqs"]);
        let n = order.len();
        let ids = (0..n).filter(|&i| u[i]["gen"] == m[i]["gen"]).count();
        let lps = (0..n).filter(|&i| u[i]["lps"] == m[i]["lps"]).count();
        let acc = (0..n).filter(|&i| u[i]["acc"] == m[i]["acc"]).count();
        let pass = ids == n && lps == n && acc == n;
        println!(
            "fnmixgate: MTP {}, {n} sequences (the long one {} tokens = 2 chunks beside padded decode steps): mixed vs unmixed ids {ids}/{n}, logprobs {lps}/{n}, acceptance {acc}/{n}; {} steps; mixed steps {} carrying {} chunks -> {}",
            st.law.as_ref().map_or("off".to_string(), ToString::to_string),
            lids.len(),
            arms["mixed"]["steps"],
            st.mixed_steps,
            st.mixed_chunks,
            if pass { "PASS" } else { "DIFFER" }
        );
        let res = json!({"mode": "mixgate", "mtp": f.fn_mtp.to_string(), "arms": arms,
            "equal": {"ids": ids, "lps": lps, "acc": acc, "n": n}, "pass": pass, "graphs": gr.census()});
        let name = format!("mixgate_mtp{}_rank0.json", f.fn_mtp);
        std::fs::write(out.join(&name), serde_json::to_string_pretty(&res)?)?;
        println!("fnresult rank0: wrote {}", out.join(&name).display());
        return Ok(());
    }
    for (arm, lseq) in [("alone", 100u64), ("padded", 101), ("alone2", 102)] {
        let t0 = Instant::now();
        let mut ev = Vec::new();
        if arm == "padded" {
            for (i, p) in short.iter().enumerate() {
                let q = 1 + i as u64;
                job!(SrvJob::Admit(SeqId(q), 0))?;
                chunk!(q, &p.ids, 0, p.ids.len());
            }
            decode!(&[1, 2, 3]);
            decode!(&[1, 2, 3]);
            job!(SrvJob::Admit(SeqId(lseq), 0))?;
            ev.push(format!(
                "long admitted at pool slot {}",
                seq_slot[&SeqId(lseq)]
            ));
            chunk!(lseq, lids, 0, cut);
            for _ in 0..PADDED {
                decode!(&[1, 2, 3]);
            }
            ev.push(format!(
                "{PADDED} decode steps of 3 ready sequences at bucket {} while the long prompt holds slot {} between its chunks",
                bucket_for(3),
                seq_slot[&SeqId(lseq)]
            ));
            for q in 1..=3u64 {
                job!(SrvJob::Release(SeqId(q)))?;
            }
            chunk!(lseq, lids, cut, lids.len());
        } else {
            job!(SrvJob::Admit(SeqId(lseq), 0))?;
            chunk!(lseq, lids, 0, cut);
            chunk!(lseq, lids, cut, lids.len());
        }
        while gens.get(&lseq).map_or(0, |g| g.toks.len()) < GEN {
            decode!(&[lseq]);
        }
        job!(SrvJob::Release(SeqId(lseq)))?;
        let g = gens.get(&lseq).cloned().unwrap_or_default();
        arms.insert(
            arm.into(),
            json!({"seq": lseq, "gen": g.toks, "lps": g.lps, "acc": g.acc, "events": ev,
                "wall_s": t0.elapsed().as_secs_f64(),
                "others": (1..=3u64).map(|q| gens.get(&q).map(|g| g.toks.clone()).unwrap_or_default()).collect::<Vec<_>>()}),
        );
    }
    let cmp = |a: &str, b: &str| -> (usize, usize, bool) {
        let x = |k: &str, f: &str| arms[k][f].clone();
        let (ga, gb) = (x(a, "gen"), x(b, "gen"));
        let ids = ga
            .as_array()
            .into_iter()
            .flatten()
            .zip(gb.as_array().into_iter().flatten())
            .take(GEN)
            .take_while(|(p, q)| p == q)
            .count();
        let (la, lb) = (x(a, "lps"), x(b, "lps"));
        let lps = la
            .as_array()
            .into_iter()
            .flatten()
            .zip(lb.as_array().into_iter().flatten())
            .take(GEN)
            .filter(|(p, q)| p == q)
            .count();
        (ids, lps, x(a, "acc") == x(b, "acc"))
    };
    let (i1, l1, a1) = cmp("alone", "padded");
    let (i2, l2, a2) = cmp("alone", "alone2");
    let pass = i1 == GEN && l1 == GEN && a1;
    println!(
        "fnpadgate: pad law {} (MTP {}), long prompt {} tokens = chunks {cut} + {}: padded vs alone ids {i1}/{GEN}, logprobs {l1}/{GEN}, acceptance {}; alone2 vs alone ids {i2}/{GEN}, logprobs {l2}/{GEN}, acceptance {} -> {}",
        if rk.pad_ctl { "CONTROL (pad i at slot i)" } else { "spare" },
        st.law.as_ref().map_or("off".to_string(), ToString::to_string),
        lids.len(),
        lids.len() - cut,
        if a1 { "equal" } else { "DIFFERS" },
        if a2 { "equal" } else { "DIFFERS" },
        if pass { "PASS" } else { "DIFFER" }
    );
    let res = json!({"mode": "padgate", "pad_control": rk.pad_ctl, "mtp": f.fn_mtp.to_string(),
        "long": {"id": long.id, "tokens": lids.len()}, "short": short.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "arms": arms, "padded_vs_alone": {"ids": i1, "lps": l1, "acc": a1}, "alone2_vs_alone": {"ids": i2, "lps": l2, "acc": a2},
        "pass": pass, "graphs": gr.census(), "moves": st.moves});
    let name = format!(
        "padgate{}_mtp{}_rank0.json",
        if rk.pad_ctl { "_control" } else { "" },
        f.fn_mtp
    );
    std::fs::write(out.join(&name), serde_json::to_string_pretty(&res)?)?;
    println!("fnresult rank0: wrote {}", out.join(&name).display());
    Ok(())
}

/// The serve plane on this rank (FN14): rank 0 runs the scheduler + HTTP plane and turns each of
/// its jobs into [`Cmd`]s; rank 1 applies rank 0's commands until `Stop`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn serve_rank(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    cfg: &Qwen4ExpConfig,
    ckpt: &Path,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    prompts: &[Prompt],
    out: &Path,
) -> anyhow::Result<()> {
    let law = f.mtp_law()?;
    ensure!(
        law.is_none() || rk.mtp.is_some(),
        "--fn-mtp in the serve plane needs the drafter"
    );
    let mut st = SrvState {
        couple_seeds: BTreeMap::new(),
        slots: vec![None; MAX_SEQS as usize + 1],
        ready: 0,
        last_lg: 0,
        last_rows: 0,
        moves: 0,
        valve: (rk.rank == 0 && f.fn_mem_valve == "on").then(|| {
            let segs = f.fn_mem_valve_max_tail / saber_mem::expert_tier::VALVE_SEG;
            let p = saber_mem::mem_valve::ValveParams {
                max_segs: segs,
                ..saber_mem::mem_valve::ValveParams::default()
            };
            // FN53c: the trigger from the boot's free floor, sampled now (the pins settled: the
            // boot, the warm and the pre-captures are done; no request has arrived)
            let boot = std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|t| saber_mem::mem_valve::mem_available_gib(&t));
            let v = match (f.fn_mem_valve_trigger.as_str(), boot) {
                ("boot", Some(b)) => saber_mem::mem_valve::MemValve::at_boot(p, b),
                _ => saber_mem::mem_valve::MemValve::new(p),
            };
            println!(
                "fnvalve: controller ON -- {} (--fn-mem-valve-trigger {}) | {:?} (park / refuse: not offered by this engine)",
                v.thresholds(),
                f.fn_mem_valve_trigger,
                v.params()
            );
            (v, Instant::now(), Instant::now())
        }),
        valve_own: if rk.rank == 0 && f.fn_mem_valve == "on" {
            let pids = own_pids();
            let rss = own_rss_gib(&pids).unwrap_or(0.0);
            println!(
                "fnvalve: own processes {pids:?}, RSS at the boot sample {rss:.2} GiB (its growth is not pressure)"
            );
            (pids, rss)
        } else {
            (Vec::new(), 0.0)
        },
        law,
        pend: None,
        last_chunk: None,
        ms: MtpStats {
            depth: f
                .mtp_law()?
                .and_then(|l| l.adaptive_ceil())
                .map(|c| super::depth::DepthCtl::new(c as usize)),
            ..MtpStats::default()
        },
        mixed_steps: 0,
        mixed_chunks: 0,
        pc: None,
        fn_chunk: f.fn_chunk,
        f75: super::stepledger::EngRec::default(),
        lookup: if f.mtp_law()?.is_some() {
            f.lookup()?
        } else {
            None
        },
        pix: Vec::new(),
        enc: Ok(0),
        enc_defer: 0,
        tower_px: None,
        tower_ran: None,
        tickets: 0,
        i125: Img125::default(),
        i126: Img126::default(),
        ds: f.device_step()?,
        dsc: super::dstep::Counts::default(),
        dchk: None,
        rpend: None,
    };
    if rk.rank == 0 {
        if st.ds == super::dstep::DeviceStep::Check {
            ensure!(
                rk.dpick.is_some(),
                "--fn-device-step check needs the device pick (its seam is the kernel's input)"
            );
            st.dchk = Some(super::dstep::Check::new(rk.hip, &rk.table)?);
        }
        println!(
            "fndstep119 rank0: form {} (FN119; ahead = the default, lazy = the control, off = pre-FN119)",
            st.ds
        );
    }
    match (&st.lookup, f.spec_lookup.as_str()) {
        (Some(c), _) => println!(
            "speclookup     ON: prompt-lookup n-grams {}..{}, merge bar {} (FN69; drafts REPLACE \
             the MTP chain's when confident, verify unchanged)",
            c.nmin, c.nmax, c.min_match
        ),
        (None, "on") => println!(
            "speclookup     INERT: --spec-lookup on needs the MTP law (SPEC 0 drafts nothing)"
        ),
        _ => println!("speclookup     off (FN69's control and the default)"),
    }
    if let Some((dir, budget)) = f.kv_pcache(rk.rank) {
        ensure!(
            rk.pools.kvt.is_some(),
            "--fn-kv-pcache-gib needs the tiered KV"
        );
        let mut model = format!(
            "flashnext-pcache v1|{}|tp2|ctx {}|chunk {}|cut {}|f32 {}|mtp {}",
            ckpt.display(),
            rk.max_ctx,
            f.max_chunk(),
            f.fn_chunk,
            f.fn_gdn_state_f32,
            rk.mtp.is_some()
        );
        // FN68b: fp8 pages key on the dtype and the scale law (a bf16 snapshot, or one written
        // under another law, never reloads into this boot); bf16's identity is unchanged
        if let Some(ks) = rk.kv_scales.as_ref() {
            model.push_str(&format!("|kv fp8 {:016x}", ks.ident()));
        }
        // FN68d: the MTP layer's own page dtype (a one-dtype snapshot never reloads into a split
        // boot, nor the reverse)
        if rk.pools.kv_mtp_dt != rk.pools.kv_dt {
            model.push_str(&format!("|kv mtp {}", rk.pools.kv_mtp_dt.name()));
        }
        // FN113: the tree's nodes are cut by its law (never mixed with FN45c's entries)
        let tree = f.prefix_tree_opts();
        if let Some(t) = tree.as_ref() {
            model.push_str(&format!(
                "|ptree113 gap {} sys {}|ix {}",
                t.min_gap, t.sys_min, rk.pools.ix_paged
            ));
        }
        let pc = pcache::Pcache::open(
            rk.hip,
            &rk.pools,
            &dir,
            budget,
            model,
            rk.max_ctx,
            f.max_chunk(),
            pcache::PcacheOpts {
                sync: f.fn_kv_pcache_sync,
                verify: f.fn_kv_pcache_verify,
                // FN113: the tree's law replaces FN45c3's conversation cut
                turn_tok: if tree.is_some() {
                    None
                } else {
                    pcache::im_start_id(ckpt)
                },
                min_tokens: f.pcache_min_tokens(),
                align: f.fn_kv_pcache_align,
                no_demote: f.fn_prefix_tree_demote == "off",
                warm_end: if f.warm_prompts.is_some() {
                    Some(
                        pcache::special_id(ckpt, "<|im_end|>")
                            .context("--warm-prompts: the tokenizer's <|im_end|>")?,
                    )
                } else {
                    None
                },
            },
            tree,
        )?;
        println!("{} (boot)", pc.line(rk.rank));
        if let Some(t) = pc.tree.as_ref() {
            println!(
                "fnptree113 rank{}: ON -- RAM tier {} pages of 2 MiB ({:.2} GiB pinned), drive \
                 {:.1} GiB, law min gap {} sys min {} cuts {}, LRFU lambda {}; END form {}, gate \
                 word {} KiB fine-grained VRAM (outside the ledger) + {} ring slots",
                rk.rank,
                t.ram.usage().2,
                f64::from(t.ram.usage().2) * 2.0 / 1024.0,
                budget as f64 / f64::from(1u32 << 30),
                t.law.min_gap,
                t.law.sys_min,
                tree.map_or_else(|| "today".into(), |o| o.cuts.name()),
                tree.map_or(0.0, |o| o.lambda),
                f.fn_prefix_tree_end,
                pcache::GATE_BYTES >> 10,
                pcache::GATE_SLOTS
            );
        }
        let mut pc = pc;
        pcache::gate_init(rk, &mut pc)?;
        st.pc = Some(pc);
    }
    if rk.rank == 1 {
        // FN73 (DEBUG): `--fail-rank-after-steps 1:N`
        let fail = f.fail_after(1)?;
        let mut decodes = 0u64;
        loop {
            let b = rk.peer.exchange().xchg_bytes("fn.cmd", &[])?;
            let cmd = serve::dec(&b)?;
            if cmd == Cmd::Stop {
                break;
            }
            if matches!(cmd, Cmd::Decode { .. }) {
                decodes += 1;
                if let Some(n) = fail.filter(|&n| decodes > n) {
                    bail!(
                        "injected failure (--fail-rank-after-steps 1:{n}): rank 1 fails at decode \
                         command {decodes}"
                    );
                }
            }
            applied(rk, em, ple, ph, recs, cfg, f, gr, dp, &mut st, &cmd)?;
        }
        if let Some(pc) = st.pc.as_mut() {
            pc.wait();
            println!("{}", pc.line(rk.rank));

            if let Some(t) = pc.tree.as_ref() {
                println!("{}", t.line(rk.rank));
            }
        }
        if let Some(v) = rk.vis.as_ref() {
            println!("{}", v.line());
        }
        if let Some(l) = rk.lazy.as_ref() {
            println!("{}", l.line(rk.rank, rk.vis.is_some()));
        }
        if rk.vis_on {
            println!(
                "{}",
                st.i125
                    .line(rk.rank, super::run::image_cache_defers(f, st.pc.is_some()))
            );
            println!("{}", st.i126.line(rk.rank));
        }
        if let Some(a) = rk.ahead.as_ref() {
            println!("fnahead119 rank1: {}", a.c);
        }
        println!(
            "fnserve rank1: stopped ({} graphs, {} moves)",
            gr.captured, st.moves
        );
        moe_sync_check(rk, em)?;
        return Ok(());
    }
    if f.scripted() {
        let r = padgate(rk, em, ple, ph, recs, cfg, f, gr, dp, &mut st, prompts, out);
        let _ = rk
            .peer
            .exchange()
            .xchg_bytes("fn.cmd", &serve::enc(&Cmd::Stop)?);
        return r;
    }
    let port = f.fn_serve.context("--fn-serve PORT")?;
    // FN66: the warm prompts (parsed before the server listens: a bad file refuses the boot)
    let warm = match (&f.warm_prompts, st.pc.as_ref()) {
        (Some(path), Some(pc)) => {
            let prompts = saber_serve::warm::load(path)?;
            println!(
                "fnserve: --warm-prompts {}: {} prompt(s); prompt cache ON ({} GiB a rank in {}, \
                 conversation cut floor {} tokens{}), max ctx {}{} -- each head is prefilled \
                 once after the server listens (`warmprompt` lines, then `warmprompts done`)",
                path.display(),
                prompts.len(),
                f.pcache_gib().unwrap_or(0.0),
                f.fn_kv_dir.display(),
                f.pcache_min_tokens(),
                if f.fn_kv_pcache_gib.is_none() {
                    "; turned on by --warm-prompts at the tied settings"
                } else {
                    ""
                },
                rk.max_ctx,
                if f.fn_max_ctx.is_none() {
                    " (--warm-prompts serves without admission by reservation: the cache's law; \
                     --fn-max-ctx raises it up to the host's pinnable ceiling)"
                } else {
                    ""
                },
            );
            let h = pc.handle();
            let probe: saber_serve::warm::CacheProbe = std::sync::Arc::new(move || {
                let c = h.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let used = c.used();
                (
                    used,
                    format!(
                        "rank 0 holds {} entries, {:.1} MiB",
                        c.len(),
                        used as f64 / f64::from(1u32 << 20)
                    ),
                )
            });
            Some((prompts, probe))
        }
        (Some(path), None) => bail!(
            "--warm-prompts {}: the prompt cache is off at this boot",
            path.display()
        ),
        _ => None,
    };
    let started = serve::start(
        port,
        ckpt,
        f.fn_chat_template.as_ref(),
        &f.fn_served_name,
        rk.max_ctx,
        match f.fn_thinking.as_str() {
            "on" => true,
            "off" => false,
            t => bail!("--fn-thinking on|off, not {t}"),
        },
        st.law.as_ref().map_or(1, |l| l.kmax() + 1),
        f.fn_mixed && !f.no_fn_mixed,
        f.max_chunk(),
        f.busy_chunk(),
        // LR-B: decode steps between two prefill chunks (0 = off)
        f.fn_prefill_yield.unwrap_or(0),
        // LR-G: lazy KV growth's headroom (0 = off; only under admission by reservation)
        f.kv_geom(rk.max_ctx)
            .filter(|g| g.admit && g.lazy)
            .map_or(0, |_| f.kv_lazy()),
        // FN51: + admission by reservation (the RAM slab's active tokens)
        f.kv_geom(rk.max_ctx)
            .filter(|g| g.ssd_bytes > 0 || g.admit)
            .map(|g| g.active_tokens()),
        // FN55b: image parts when the tower is resident
        rk.vis_on.then_some(f.vision_patches()),
        // FN125: each image's content hash for the keys (the arm on a boot with a cache)
        rk.vis_on && super::run::image_cache_defers(f, st.pc.is_some()),
        warm,
        f.fn_step_trace.as_deref(),
    )?;
    if let Some(l) = &st.law {
        println!(
            "fnserve: MTP {} (the served default is SPEC 0: --fn-mtp 0)",
            l
        );
    }
    let sampler = std::sync::Arc::new(saber_serve::sampler::Sampler::new());
    let mut seq_slot: BTreeMap<SeqId, u32> = BTreeMap::new();
    let mut steps = 0u64;
    let refresh = |rk: &Rank<'_, '_>, ple: &PleTier, st: &SrvState| {
        if let Ok(mut t) = started.plane.lock() {
            *t = format!(
                "{}{}{}{}{}{}",
                started.ledger.lock().map_or(String::new(), |l| l.render()),
                rk.tier.tier.render_prometheus(),
                ple.render_prometheus(),
                if st.law.is_some() {
                    st.ms.render()
                } else {
                    String::new()
                },
                rk.lazy
                    .as_ref()
                    .map_or(String::new(), |l| l.render(rk.vis.is_some())),
                elastic_metrics(rk)
            );
            // LR-D: the prefix tree's live hit rate and churn
            if let Some(tr) = st.pc.as_ref().and_then(|pc| pc.tree.as_ref()) {
                t.push_str(&tr.render());
            }
            if rk.vis_on {
                t.push_str(&st.i125.render());
                t.push_str(&st.i126.render());
                if let Ok(h) = started.hash125.lock() {
                    t.push_str(&h.render());
                }
            }
        }
    };
    refresh(rk, ple, &st);
    // FN56: the serve plane's handoff census -- per step job: the rank thread's wait for it (the
    // scheduler's turn, the reply's trip) and its own time in the job
    let (mut w_ms, mut j_ms, mut w_n) = (0.0f64, 0.0f64, 0u64);
    // FN53b: with the valve on, an idle server still ticks the controller every second (the
    // heartbeat stays one a minute)
    // FN51f-b: with the elastic region on, an idle server refreshes `/metrics` every second (a
    // phase's counters are read right after it)
    let tick_s = if st.valve.is_some() || rk.elastic.is_some() {
        1
    } else {
        60
    };
    let mut last_noop = Instant::now();
    // FN55c: the last image request (the lazy tower's idle clock)
    let mut last_img = Instant::now();
    // FN73 (DEBUG): `--fail-rank-after-steps 0:N`
    let fail0 = f.fail_after(0)?;
    let r = (|| -> anyhow::Result<()> {
        loop {
            // FN73: a fatal (rank 1 died: the watch; or any other reporter) ends the serve loop
            if let Some(x) = saber_serve::fatal::get() {
                bail!("the pair is down: {}", x.line());
            }
            let tw = Instant::now();
            // FN55c: the lazy tower's idle release (no image for `--fn-vision-idle-s`, no image
            // sequence in the pools): both ranks lend its region back to the expert tier
            if lazy_due(rk, &st, last_img, f.fn_vision_idle_s) {
                lead(
                    rk,
                    em,
                    ple,
                    ph,
                    recs,
                    cfg,
                    f,
                    gr,
                    dp,
                    &mut st,
                    &Cmd::VisionRelease,
                )?;
                refresh(rk, ple, &st);
            }
            let wait = if rk.vis.is_some() && rk.lazy.is_some() {
                std::time::Duration::from_secs(f.fn_vision_idle_s)
                    .saturating_sub(last_img.elapsed())
                    .clamp(
                        std::time::Duration::from_secs(1),
                        std::time::Duration::from_secs(tick_s),
                    )
            } else {
                std::time::Duration::from_secs(tick_s)
            };
            // FN73: the wait is bounded at 1 s so the fatal check above runs while idle (the
            // heartbeat and the valve keep their own clocks)
            let wait = wait.min(std::time::Duration::from_secs(1));
            let job = match started.jobs.recv_timeout(wait) {
                Ok(j) => j,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if rk.elastic.is_some() {
                        refresh(rk, ple, &st);
                    }
                    if let Some(k) = valve_tick(&mut st) {
                        lead(
                            rk,
                            em,
                            ple,
                            ph,
                            recs,
                            cfg,
                            f,
                            gr,
                            dp,
                            &mut st,
                            &Cmd::Valve { closed: k },
                        )?;
                    }
                    if last_noop.elapsed().as_secs() >= 60 {
                        last_noop = Instant::now();
                        let _ = rk
                            .peer
                            .exchange()
                            .xchg_bytes("fn.cmd", &serve::enc(&Cmd::Noop)?)?;
                        refresh(rk, ple, &st);
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let is_step = matches!(job, SrvJob::Step(..));
            if matches!(job, SrvJob::Encode(..)) {
                last_img = Instant::now();
            }
            let tj = Instant::now();
            // FN75: a step job's pick-up and the drafter's counters before it
            let is_step75 = matches!(job, SrvJob::Step(..) | SrvJob::Mixed(..));
            let dr0 = (st.ms.dev_ms, st.ms.host_ms);
            if is_step75 {
                st.f75.pick = Some(tj);
            }
            // FN125: `/metrics` refreshed after every image prompt's last chunk (a turn's counters
            // are read right after it; text traffic never takes this refresh)
            let img0 = st.i125.prompts;
            serve_job(
                rk,
                em,
                ple,
                ph,
                recs,
                cfg,
                f,
                gr,
                dp,
                &mut st,
                &mut seq_slot,
                &sampler,
                &mut steps,
                job,
            )?;
            // FN75: the step job's record into the served-step ledger (a non-step job's wall rides
            // the next step's record)
            {
                let end = Instant::now();
                let e = &mut st.f75;
                if is_step75 {
                    e.dr_dev_ms += st.ms.dev_ms - dr0.0;
                    e.dr_wall_ms += st.ms.host_ms - dr0.1;
                    e.post_ms = e.reply.map_or(0.0, |r| (end - r).as_secs_f64() * 1e3);
                    let rec = std::mem::take(e);
                    if let Ok(mut l) = started.ledger.lock() {
                        l.push_eng(rec);
                    }
                } else {
                    e.other_ms += (end - tj).as_secs_f64() * 1e3;
                    e.other_n += 1;
                }
            }
            if is_step {
                // a wait past 50 ms is an idle server, not a handoff
                let wt = (tj - tw).as_secs_f64() * 1e3;
                if wt < 50.0 {
                    w_ms += wt;
                    j_ms += tj.elapsed().as_secs_f64() * 1e3;
                    w_n += 1;
                }
            }
            if (is_step && steps % 64 == 0) || st.i125.prompts != img0 {
                refresh(rk, ple, &st);
            }
            if let Some(n) = fail0.filter(|&n| is_step && steps >= n) {
                bail!(
                    "injected failure (--fail-rank-after-steps 0:{n}): rank 0 fails after step \
                     {steps}"
                );
            }
            if let Some(k) = valve_tick(&mut st) {
                lead(
                    rk,
                    em,
                    ple,
                    ph,
                    recs,
                    cfg,
                    f,
                    gr,
                    dp,
                    &mut st,
                    &Cmd::Valve { closed: k },
                )?;
            }
        }
        Ok(())
    })();
    println!(
        "fnserve56 rank0: {w_n} step jobs: wait for the job {:.3} ms, in the job {:.3} ms (means); deferred tier ends {:.3} ms total; prebuilt tables {} built / {} used",
        w_ms / w_n.max(1) as f64,
        j_ms / w_n.max(1) as f64,
        rk.tier_def_ms,
        rk.tier_pre_n.0,
        rk.tier_pre_n.1
    );
    println!("{}", fnhost59_line("rank0 served", &rk.fn59));
    if let Ok(mut l) = started.ledger.lock() {
        l.flush();
        println!("{}", l.line());
        println!("{}", l.line93());
    }
    // FN73: a failed serve loop is the process's fatal (after 1 s for rank 1's own reason to
    // arrive through the watch): the HTTP plane stops, every request ends with the reason, no
    // `Stop` is sent to a peer that may be gone, and the job channel is dropped so the step
    // thread's wait on a queued job returns
    if let Err(e) = &r {
        saber_serve::fatal::set_after(0, format!("{e:#}"), std::time::Duration::from_secs(1));
    } else {
        let _ = rk
            .peer
            .exchange()
            .xchg_bytes("fn.cmd", &serve::enc(&Cmd::Stop)?);
    }
    if let Some(pc) = st.pc.as_mut() {
        pc.wait();
        println!("{}", pc.line(rk.rank));

        if let Some(t) = pc.tree.as_ref() {
            println!("{}", t.line(rk.rank));
        }
    }
    if let Some(v) = rk.vis.as_ref() {
        println!("{}", v.line());
    }
    if let Some(l) = rk.lazy.as_ref() {
        println!("{}", l.line(rk.rank, rk.vis.is_some()));
    }
    if rk.vis_on {
        println!(
            "{}",
            st.i125
                .line(rk.rank, super::run::image_cache_defers(f, st.pc.is_some()))
        );
        println!("{}", st.i126.line(rk.rank));
    }
    println!("fndstep119 rank0: form {}; {}", st.ds, st.dsc);
    if let Some(a) = rk.ahead.as_ref() {
        println!("fnahead119 rank0: {}", a.c);
    }
    println!(
        "fnserve rank0: stopped after {steps} steps ({} mixed, carrying {} chunks); graphs {} ({} replays, {} eager); {} compaction moves; {} declc cells run as the pair (emitted); verify class {}: {} layers emitted in SPEC 0's class, {} in their own M's (emitted)",
        st.mixed_steps, st.mixed_chunks, gr.captured, gr.replays, gr.eager, st.moves, em.declc_as_pair,
        f.fn_verify_class, em.vclass_layers[0], em.vclass_layers[1]
    );
    if let Some(c) = &st.ms.depth {
        println!(
            "fnmtpdepth rank0 b1: {} | evaluated by position {:?}, accepted {:?}",
            c.summary(),
            st.ms.eval_pos,
            st.ms.per_pos
        );
    }
    if r.is_ok() {
        moe_sync_check(rk, em)?;
    }
    drop(started.jobs);
    let j = started.join.join();
    r?;
    match j {
        Ok(x) => x,
        Err(_) => bail!("the HTTP thread panicked"),
    }
}

/// FN18: the tier's per-layer which-path totals at the end of a run (the replay self-test's
/// reference: `saber flashnext-profile replay --tier-log` must equal them exactly).
/// FN45b2: the page byte law's record -- every (slot, QSA layer)'s K/V pages in LOGICAL order as
/// one FNV-1a 64 a page (`fnkvpages rank R slot S layer L final F: h0 h1 ...`): under the tier the
/// pages below the sequence's final length (VRAM read back or the RAM slab), on the flat pools
/// every page of the slot (the comparison takes the tier's `F`).
fn kv_pages_lines(rk: &Rank<'_, '_>) -> anyhow::Result<()> {
    let fnv = |b: &[u8]| {
        b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
            (h ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3)
        })
    };
    for slot in 0..=MAX_SEQS {
        for l in 0..=LAYERS as usize {
            let Some(tl) = super::kvwire::tier_layer(l as i32) else {
                continue;
            };
            // FN68d: the MTP layer's pages at its own dtype
            let bb = super::kvwire::page_bytes(if l == LAYERS as usize {
                rk.pools.kv_mtp_dt
            } else {
                rk.pools.kv_dt
            }) as usize;
            let (fin, bytes) = if let Some(k) = rk.pools.kvt.as_ref() {
                let k = k.borrow();
                let f = k.tier.len(slot) / 16;
                if f == 0 {
                    continue;
                }
                (f, k.read_pages(rk.hip, &rk.dev.stream, slot, tl, f)?)
            } else {
                let a = rk.pools.seq(slot, l).kv;
                let (d, off) = rk.pools.owner(a)?;
                let mut v = vec![0u8; rk.pools.kvb_of(l) as usize];
                rk.hip
                    .memcpy_d2h_at_blocking(&mut v, d, off, &rk.dev.stream)?;
                (rk.pools.nbs, v)
            };
            let hs: Vec<String> = bytes
                .chunks(bb)
                .map(|c| format!("{:016x}", fnv(c)))
                .collect();
            println!(
                "fnkvpages rank{} slot {slot} layer {l} final {fin}: {}",
                rk.rank,
                hs.join(" ")
            );
        }
    }
    Ok(())
}

/// FN116: the record of one first drafter round after a prompt (`Rank::f116`), ms unless named.
const F116_TERMS: [&str; 15] = [
    "m",
    "since",
    "total",
    "stash",
    "rows",
    "emb",
    "consts",
    "lend",
    "launch",
    "wait",
    "dev",
    "pick",
    "chain",
    "chain_dev",
    "kn",
];
const F116_N: usize = F116_TERMS.len();

/// FN116: the `fnfirst116 rank<r>:` end lines (p50 per term + every record).
fn f116_line(rk: &Rank<'_, '_>) {
    if rk.f116.is_empty() {
        return;
    }
    let p50 = |i: usize| {
        let mut v: Vec<f64> = rk.f116.iter().map(|r| r[i]).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let terms: Vec<String> = F116_TERMS
        .iter()
        .enumerate()
        .map(|(i, k)| format!("{k} {:.3}", p50(i)))
        .collect();
    println!(
        "fnfirst116 rank{}: {} first rounds ({} as the K-side front + a steady round, {} front pieces, {} \
         fronts under the chunk (FN119e)) | p50 {}",
        rk.rank,
        rk.f116.len(),
        rk.f116_kv,
        rk.f116_front,
        rk.front_early,
        terms.join(" ")
    );
    let recs: Vec<String> = rk
        .f116
        .iter()
        .map(|r| {
            let v: Vec<String> = r.iter().map(|x| format!("{x:.3}")).collect();
            format!("[{}]", v.join(","))
        })
        .collect();
    println!(
        "fnfirst116rec rank{}: {{\"terms\":[{}],\"recs\":[{}]}}",
        rk.rank,
        F116_TERMS
            .iter()
            .map(|k| format!("\"{k}\""))
            .collect::<Vec<_>>()
            .join(","),
        recs.join(",")
    );
}

/// FN117b: the hand slices' state at boot (`none` = None). S6's shared pick line: rank 0 creates
/// the file under `/dev/shm`, sends its name over the socket, both ranks meet once rank 1 mapped
/// it, rank 0 removes the name (the mappings live on; nothing is left on the disk).
fn hand_boot(
    f: &FnFlags,
    rank: u32,
    peer: &RankPeer,
) -> anyhow::Result<Option<super::hand::HandRt>> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let hand = f.hand()?;
    ensure!(
        f.fn_hand_skew_us == 0 || hand.s6,
        "--fn-hand-skew-us is the pick line's race check: it needs --fn-hand s6 (or all)"
    );
    if !hand.any() {
        return Ok(None);
    }
    let mut rt = super::hand::HandRt::new(hand, f.fn_hand_skew_us);
    if hand.s6 {
        use saber_hip::shmline::ShmLine;
        let cap = super::hand::PICK_LINE_CAP;
        let to = Duration::from_secs(300);
        let line = if rank == 0 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let p = std::path::PathBuf::from(format!(
                "/dev/shm/saber-fn117b-{}-{nanos}",
                std::process::id()
            ));
            let l = ShmLine::create(&p, cap, to)
                .map_err(|e| anyhow::anyhow!("FN117b: the pick line: {e}"))?;
            peer.exchange()
                .xchg_bytes("fn117b.line", p.as_os_str().as_bytes())?;
            peer.rendezvous("fn117b.line.mapped")?;
            l.unlink()
                .map_err(|e| anyhow::anyhow!("FN117b: the pick line: {e}"))?;
            l
        } else {
            let name = peer.exchange().xchg_bytes("fn117b.line", &[])?;
            let p = std::path::PathBuf::from(std::ffi::OsString::from_vec(name));
            let l = ShmLine::open(&p, cap, to)
                .map_err(|e| anyhow::anyhow!("FN117b: the pick line: {e}"))?;
            peer.rendezvous("fn117b.line.mapped")?;
            l
        };
        rt.line = Some(line);
    }
    println!(
        "fnhand117b rank{rank}: form {} (boot){}{}",
        hand.form(),
        if hand.s6 {
            format!(
                " | S6 pick line: {} KiB slots x 2 a rank, shared page {} KiB",
                super::hand::PICK_LINE_CAP >> 10,
                saber_hip::shmline::ShmLine::map_bytes(super::hand::PICK_LINE_CAP) >> 10
            )
        } else {
            String::new()
        },
        if f.fn_hand_skew_us > 0 {
            format!(
                " | SKEW 0..{} us before every exchange (race check)",
                f.fn_hand_skew_us
            )
        } else {
            String::new()
        }
    );
    Ok(Some(rt))
}

/// FN117b (S1): the round's pinned staging (inert without the device chain: the host chain's
/// round has no chain inputs to stage).
fn hand_s1_alloc(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    if !rk.hand.as_ref().is_some_and(|h| h.hand.s1) {
        return Ok(());
    }
    let Some(inb) = rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .map(|c| c.inb.len())
    else {
        println!(
            "fnhand117b rank{}: S1 INERT -- no device chain (the round has no chain inputs)",
            rk.rank
        );
        return Ok(());
    };
    let emb = (MAX_SEQS * super::exec::WMAX) as usize * (H * 2) as usize;
    let lay = super::hand::S1Lay::of(emb, 4 << 20, inb as usize);
    let stg = rk.hip.host_malloc(lay.total as u64)?;
    println!(
        "fnhand117b rank{}: S1 staging {:.2} MiB pinned (embedding rows {} B, pass-1 tables {} B, chain inputs {} B)",
        rk.rank,
        lay.total as f64 / 1048576.0,
        lay.emb_cap,
        lay.cs_cap,
        lay.inb_cap
    );
    if let Some(h) = rk.hand.as_mut() {
        h.s1 = Some((lay, stg));
    }
    Ok(())
}

/// FN117b (S1): this round stages its inputs.
fn hand_s1_on(rk: &Rank<'_, '_>) -> bool {
    rk.hand
        .as_ref()
        .is_some_and(|h| h.hand.s1 && h.s1.is_some())
        && rk.mtp.as_ref().is_some_and(|m| m.chain.is_some())
}

/// FN117b (S1): the round's three inputs -- the pass-1 embedding rows (`emb`, into the buffers'
/// embedding head), the pass-1 step tables (`cs`, into the constant region) and the chain's inputs
/// (`inb`) -- laid into the pinned staging and queued as ONE batch of three async copies on the
/// step's stream (the bytes the control's three blocking copies write, before the same launch).
/// A round whose blobs do not fit the staging makes the control's copies (counted `wide`).
fn hand_s1_upload(rk: &mut Rank<'_, '_>, emb: &[u8], cs: &[u8], inb: &[u8]) -> anyhow::Result<()> {
    let t = Instant::now();
    let dst = rk.emb.clone();
    ensure!(
        emb.len() as u64 <= dst.len,
        "FN100: the embedding rows: {} B into a {} B buffer",
        emb.len(),
        dst.len
    );
    let dc_inb = &rk
        .mtp
        .as_ref()
        .and_then(|m| m.chain.as_ref())
        .context("FN117b S1: the device chain")?
        .inb;
    let h = rk.hand.as_mut().context("FN117b: the hand")?;
    let check = h.hand.check;
    let (lay, stg) = h.s1.as_mut().context("FN117b S1: the staging")?;
    let lay = *lay;
    if !lay.fits(emb.len(), cs.len(), inb.len()) {
        h.s1_wide += 1;
        rk.hip
            .memcpy_h2d_at_blocking(&dst.own, dst.off, emb, &rk.dev.stream)?;
        rk.hip
            .memcpy_h2d_at_blocking(&rk.dev.consts, 0, cs, &rk.dev.stream)?;
        rk.hip.memcpy_h2d_blocking(dc_inb, inb, &rk.dev.stream)?;
        return Ok(());
    }
    lay.stage(stg.as_mut_slice(), emb, cs, inb);
    let h2d = saber_hip::MemcpyKind::HostToDevice;
    rk.hip.run_batch(&[
        Job::MemcpyPinned {
            dev: &dst.own,
            dev_offset: dst.off,
            host: stg,
            host_offset: lay.emb_off as u64,
            bytes: emb.len() as u64,
            kind: h2d,
            stream: &rk.dev.stream,
        },
        Job::MemcpyPinned {
            dev: &rk.dev.consts,
            dev_offset: 0,
            host: stg,
            host_offset: lay.cs_off as u64,
            bytes: cs.len() as u64,
            kind: h2d,
            stream: &rk.dev.stream,
        },
        Job::MemcpyPinned {
            dev: dc_inb,
            dev_offset: 0,
            host: stg,
            host_offset: lay.inb_off as u64,
            bytes: inb.len() as u64,
            kind: h2d,
            stream: &rk.dev.stream,
        },
    ])?;
    h.s1_rounds += 1;
    h.s1_ms += t.elapsed().as_secs_f64() * 1e3;
    if check {
        // the oracle: the three regions read back == the host's bytes (the stream drained first)
        let a = d2h_in(&rk.dev, &dst.own, dst.addr().as_u64(), emb.len())?;
        let b = d2h_in(
            &rk.dev,
            &rk.dev.consts,
            rk.dev.consts.addr().as_u64(),
            cs.len(),
        )?;
        let c = d2h_in(&rk.dev, dc_inb, dc_inb.addr().as_u64(), inb.len())?;
        ensure!(
            a == emb && b == cs && c == inb,
            "FN117b S1: a staged round input differs on the card (emb {} tables {} chain {})",
            a == emb,
            b == cs,
            c == inb
        );
        if let Some(h) = rk.hand.as_mut() {
            h.s1_checked += 1;
        }
    }
    seg(rk, "stage", t);
    Ok(())
}

/// FN117b (S2): the next step's expert-pointer tables built from the placements the verify's tier
/// end left, staged in the tier's pinned table buffer and queued behind the round on the step's
/// stream (no sync: the round's ids read drains it before anything rewrites the buffer); the key
/// of the tables' inputs is kept for the next begin. Skipped (counted) where the begin would not
/// rebuild them from a complete tier as today: a deferred tier end pending (FN56), FN56's own
/// prebuilt set, an incomplete tier, a staged ring.
fn hand_s2_build(rk: &mut Rank<'_, '_>) -> anyhow::Result<()> {
    let Some(h) = rk.hand.as_mut() else {
        return Ok(());
    };
    if !h.hand.s2 {
        return Ok(());
    }
    h.pre = None;
    let ok = rk.tier_pend.is_none()
        && !rk.tier_pre
        && rk.tier.tier.complete()
        && rk.ring.as_ref().is_none_or(|r| r.plan.is_none());
    if !ok {
        h.s2_skipped += 1;
        return Ok(());
    }
    let t = Instant::now();
    for l in 0..LAYERS {
        let (gu, dn, fetch) = rk.tier.tables(l, rk.rank);
        ensure!(fetch == 0, "a complete tier has a Fetch placement");
        rk.tier.put(l, &gu, &dn);
    }
    rk.tier.upload_async(rk.hip, &rk.dev.stream, 0, LAYERS)?;
    let key = rk.tier.tier.xptr_key();
    let h = rk.hand.as_mut().context("FN117b: the hand")?;
    h.pre = Some(key);
    h.s2_built += 1;
    h.s2_ms += t.elapsed().as_secs_f64() * 1e3;
    Ok(())
}

/// FN117b (S2): keep the build iff the tables' inputs are unchanged (`check`: the tables rebuilt
/// now == the staged bytes == the card's).
fn hand_s2_keep(
    rk: &mut Rank<'_, '_>,
    k: &saber_mem::expert_tier::XptrKey,
) -> anyhow::Result<bool> {
    let same = rk.tier.tier.xptr_key() == *k;
    let check = rk.hand.as_ref().is_some_and(|h| h.hand.check);
    if same && check {
        let xb = rk.tier.xn * 8;
        let mut want = vec![0u8; LAYERS as usize * 2 * xb];
        for l in 0..LAYERS {
            let (gu, dn, _) = rk.tier.tables(l, rk.rank);
            let base = l as usize * 2 * xb;
            for (i, v) in gu.iter().chain(dn.iter()).enumerate() {
                want[base + i * 8..base + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
        }
        ensure!(
            rk.tier.xhost.as_slice()[..want.len()] == want[..],
            "FN117b S2: the kept build's staged tables differ from the begin's"
        );
        let dev = d2h_in(
            &rk.dev,
            &rk.tier.xptr,
            rk.tier.xptr.addr().as_u64(),
            want.len(),
        )?;
        ensure!(
            dev == want,
            "FN117b S2: the card's tables differ from the begin's"
        );
    }
    let h = rk.hand.as_mut().context("FN117b: the hand")?;
    if same {
        h.s2_kept += 1;
        h.s2_checked += u64::from(check);
    } else {
        h.s2_void += 1;
    }
    Ok(same)
}

/// FN117b (S6): the pick exchange -- the shared line under `--fn-hand s6` (a frame wider than the
/// slot, and every boot without S6, take the socket: the control); `check` runs the socket too and
/// refuses a differing frame.
fn hand_xchg(rk: &mut Rank<'_, '_>, mine: &[u8]) -> anyhow::Result<Vec<u8>> {
    let rank = rk.rank;
    let Some(h) = rk.hand.as_mut().filter(|h| h.line.is_some()) else {
        return Ok(rk.peer.exchange().xchg_bytes("fn.pick2", mine)?);
    };
    let line = h.line.as_ref().context("FN117b S6: the line")?;
    if mine.len() > line.cap() {
        h.s6_wide += 1;
        return Ok(rk.peer.exchange().xchg_bytes("fn.pick2", mine)?);
    }
    let seq = line.seq() + 1;
    let sk = h.skew(rank, seq);
    if sk > 0 {
        std::thread::sleep(Duration::from_micros(sk));
    }
    let t = Instant::now();
    let theirs = h
        .line
        .as_mut()
        .context("FN117b S6: the line")?
        .exchange(mine)
        .map_err(|e| anyhow::anyhow!("FN117b S6: {e}"))?;
    h.s6_calls += 1;
    h.s6_ms += t.elapsed().as_secs_f64() * 1e3;
    if h.hand.check {
        let s = rk.peer.exchange().xchg_bytes("fn.pick2", mine)?;
        ensure!(
            s == theirs,
            "FN117b S6: call {seq}: the line's pick frame differs from the socket's"
        );
        if let Some(h) = rk.hand.as_mut() {
            h.s6_checked += 1;
        }
    }
    Ok(theirs)
}

fn tier_end_line(rk: &Rank<'_, '_>) {
    f116_line(rk);
    println!("{}", rk.dev.wait_line());
    if let Some(o) = rk.ov112.as_ref() {
        println!("{}", o.line(rk.rank));
    }
    if let Some(h) = rk.hand.as_ref() {
        println!("{}", h.line(rk.rank));
    }
    let a = rk.pick_acc;
    let n = a.0.max(1) as f64;
    println!(
        "fnpick57 rank{}: {} picks, {} rows ({} read whole) | ms a pick: read {:.3} seam {:.3} exchange {:.3}{}",
        rk.rank,
        a.0,
        a.1,
        a.2,
        a.3 / n,
        a.4 / n,
        a.5 / n,
        rk.dpick.as_ref().map_or(String::new(), |d| format!(
            " | device: {} rows, {} checked vs the host seam, {} differ",
            d.rows, d.checked, d.differ
        ))
    );
    if let Some(k) = rk.pools.kvt.as_ref() {
        println!("{}", k.borrow().end_line(rk.rank));
    }
    pregate_line(rk);
    // FN63b: the mover's census + the HSA lane's counters
    println!("{}", rk.tier.mover.queue().census_line(rk.rank));
    // FN71c: which route served each drive read (the pool bounce's arm)
    if let Some(pc) = rk.tier.mover.queue().pool_counts() {
        let mc = rk.tier.mover.counts;
        let vc = rk.tier.tier.valve_counts();
        println!(
            "fnp2p rank{}: drive reads by route -- pool {} ({:.2} GB, drive -> VRAM), host {} (drive -> pinned RAM: the L2 / band refills) | bounce prefetched {} hits {} demand {} full {}",
            rk.rank,
            mc.nvme_pool_reads,
            pc.pool_bytes as f64 / 1e9,
            mc.nvme_reads,
            vc.bounce_prefetch,
            vc.bounce_hits,
            vc.bounce_demand,
            vc.bounce_full
        );
    }
    // FN53c: the device miss flag's counters
    if let Some(x) = rk.xmiss.as_ref() {
        println!("{}", x.line(rk.rank, &rk.tier.tier.valve_counts()));
    }
    // FN72b: the landed-mark law's which-path counters
    let (law, mc) = rk.tier.mover.queue().marks();
    println!(
        "xtiermarks rank{}: law {law:?} word_marks {} gate_waits {} guard_host_waits {} step_records {}",
        rk.rank, mc[0], mc[1], mc[2], mc[3]
    );
    let pl = rk.tier.tier.per_layer();
    let j = |f: &dyn Fn(&PathCounts) -> u64| {
        pl.iter()
            .map(|c| f(c).to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let (_, mv) = rk.tier.tier.totals();
    println!(
        "fntierend rank{}: steps {} inserts {} refused {} demotions {} refills {} refill_late {} \
         pool_waits {} | resident [{}] | ram [{}]",
        rk.rank,
        rk.tier.tier.steps(),
        mv.inserts,
        mv.refused,
        mv.demotions,
        mv.refills,
        mv.refill_late,
        mv.hole_waits,
        j(&|c| c.resident),
        j(&|c| c.ram)
    );
    // FN107 (X1): the exclusive pool slots' write-back path (a victim D2H through the spare)
    let nx = rk.tier.tier.exclusive_slots();
    if nx > 0 {
        println!(
            "fnpoolx107 rank{}: end -- {nx} exclusive pool slots, {} victims WRITTEN BACK to RAM \
             (of {} demotions), {} lent drops (inclusive slots), records on the drive {}",
            rk.rank,
            mv.excl_writebacks,
            mv.demotions,
            mv.lent_drops,
            rk.tier.tier.occupancy().2
        );
    }
}

/// FN80b-2: THE MIRRORED PLACEMENT LAW (`--fn-experts tp2`): both ranks hold half r of the SAME
/// records in the SAME slots (one placement, decided on each rank from the replicated routes by the
/// same deterministic policy). The pair agrees on the placement map's digest; a divergence is a
/// FATAL by name, never silent.
fn mirror_check(rk: &Rank<'_, '_>, at: &str) -> anyhow::Result<()> {
    let (d, n) = super::xtp2::placement_digest(rk.tier.tier.placements());
    println!(
        "fnplace rank{} {at}: map {d:016x} (vram {} ram {} fetch {}) steps {}",
        rk.rank,
        n[0],
        n[1],
        n[2],
        rk.tier.tier.steps()
    );
    rk.peer.agree("fn80b2.placement", d).map_err(|e| {
        anyhow::anyhow!(
            "FATAL rank {}: the mirrored expert placement DIVERGED between the ranks at {at}              (--fn-experts tp2, FN80b-2): {e}",
            rk.rank
        )
    })
}

/// FN51e: the pinned staging (64 expert slots) of the end-of-run [`verify_slots`] check, allocated
/// only when it runs (the boot's warm stage is freed after the warm: 160 MiB a rank).
fn verify_stage(hip: &HipThread, stride: u64) -> anyhow::Result<saber_hip::PinnedBuf> {
    // FN150: at the tier's stride (NVFP4 676 pages; never below MXFP4's: the tp2 halves keep it)
    Ok(hip.host_malloc(64 * stride.max(saber_mem::flashnext_budget::geometry::EXPERT_STRIDE))?)
}

/// FN18, THE PLACEMENT BYTE LAW at the end of a run: every VRAM slot (read back through the pinned
/// staging, 64 at a time) and every RAM slot of this rank == its record's sha256 (the xtier table).
fn verify_slots(
    rk: &mut Rank<'_, '_>,
    xt: &XtierFile,
    stage: &mut saber_hip::PinnedBuf,
) -> anyhow::Result<()> {
    rk.hip.sync_stream(&rk.dev.stream)?;
    rk.tier.mover.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    rk.tier.tier.settle();
    tier_refill(rk)?;
    let st = xt.header().stride as usize;
    let (mut checked, mut bad) = (0u64, 0u64);
    {
        let slab = rk.tier.mover.queue().ram.as_slice();
        let slots: Vec<(u32, u32)> = (0..rk.tier.tier.ram_slab_slots())
            .filter_map(|s| rk.tier.tier.ram_record(s).map(|r| (s, r)))
            // FN36: the refilled pool slots of the swaps awaiting publication
            .chain(rk.tier.tier.refill_targets())
            .collect();
        let (c, b) = crate::fn_moe_bench::verify_threaded(xt, slab, st, &slots);
        checked += c;
        bad += b;
    }
    let per = stage.len() / st;
    // FN42b: + the lendable slots (a lent slot's bytes live in the arena's tail)
    let vs = rk.tier.tier.params().vram_slots + rk.tier.tier.lent().1;
    let slots: Vec<(u32, u32)> = (0..vs)
        .filter_map(|s| rk.tier.tier.vram_record(s).map(|r| (s, r)))
        .collect();
    for chunk in slots.chunks(per) {
        let q = rk.tier.mover.queue();
        let jobs: Vec<Job<'_>> = chunk
            .iter()
            .enumerate()
            .map(|(k, &(s, _))| Job::MemcpyPinned {
                dev: q.slot_at(s).0,
                dev_offset: q.slot_at(s).1,
                host: stage,
                host_offset: (k * st) as u64,
                bytes: st as u64,
                kind: saber_hip::MemcpyKind::DeviceToHost,
                stream: &q.stream,
            })
            .chain(std::iter::once(Job::SyncStream { stream: &q.stream }))
            .collect();
        rk.hip.run_batch(&jobs)?;
        let local: Vec<(u32, u32)> = chunk
            .iter()
            .enumerate()
            .map(|(k, &(_, r))| (k as u32, r))
            .collect();
        let (c, b) = crate::fn_moe_bench::verify_threaded(xt, stage.as_slice(), st, &local);
        checked += c;
        bad += b;
    }
    println!(
        "fnslots rank{}: THE PLACEMENT BYTE LAW -- checked {checked} slots (VRAM {} + RAM), {bad} differ from the xtier sha256 table",
        rk.rank,
        slots.len()
    );
    ensure!(
        bad == 0,
        "rank {}: {bad} slots differ from their records",
        rk.rank
    );
    Ok(())
}

#[cfg(test)]
mod lre_tests {
    /// LR-E: the armed `fnringpass` suffix sums the per-layer waits (noise below 0 counts 0),
    /// names the worst layer and prices the wait against the copy span.
    #[test]
    fn lrh_the_ring_line_prices_both_order_keys() {
        use saber_mem::stage_ring::{StageCounts, StageKey};
        let c = StageCounts {
            ring: 100,
            uva: 30,
            vram: 900,
            wasted: 20,
            staged: 120,
            misordered: 15,
        };
        let a = StageCounts {
            ring: 110,
            uva: 20,
            wasted: 10,
            misordered: 4,
            ..c
        };
        assert_eq!(
            super::ring_order_suffix(StageKey::Profile, &c, Some(a)),
            " | order profile: misordered 15 | served (counterfactual): ring 110 uva 20 \
             staged unrouted 10 misordered 4"
        );
        assert_eq!(
            super::ring_order_suffix(StageKey::Served, &c, None),
            " | order served: misordered 15"
        );
    }

    #[test]
    fn lre_ring_idle_sums_the_layer_waits_and_names_the_worst() {
        let w = [0.0f32, 4.0, -0.01, 6.5, 1.5];
        let (sum, at, max) = super::ring_idle_summary(&w);
        assert!((sum - 12.0).abs() < 1e-9);
        assert_eq!((at, max), (3, 6.5));
        assert_eq!(
            super::ring_idle_suffix(&w, 48.0),
            " | compute wait 12.00 ms (25.0 % of the span) worst layer 3 6.50 ms"
        );
        assert_eq!(super::ring_idle_summary(&[]), (0.0, 0, 0.0));
    }
}

#[cfg(test)]
mod pick_tests {
    /// FN98: the one form's execs and gate blocks -- the PLE layer-1 cut makes two execs (the
    /// head in the last), the blocks are the lazy cut points (FN64's grid) and tile the layers.
    #[test]
    fn fn98_one_form_execs_and_gate_blocks() {
        assert_eq!(super::one_segs(None), vec![(0, LAYERS + 1)]);
        assert_eq!(super::one_segs(Some(1)), vec![(0, 1), (1, LAYERS + 1)]);
        assert_eq!(super::one_blocks(0, 1, 4), vec![(0, 1)]);
        let b = super::one_blocks(1, LAYERS + 1, 4);
        assert_eq!(b.first(), Some(&(1, 4)));
        assert_eq!(b.last(), Some(&(44, LAYERS)));
        assert_eq!(b.len(), 12);
        assert!(b.windows(2).all(|w| w[0].1 == w[1].0));
        let all = super::one_blocks(0, LAYERS + 1, 1);
        assert_eq!(all.len(), LAYERS as usize);
        // every block start is a cut point FN64's lazy split can draw from (q 4 + the PLE layer)
        let pts: Vec<u32> = (0..LAYERS).filter(|&l| l % 4 == 0 || l == 1).collect();
        let mut starts: Vec<u32> = super::one_blocks(0, 1, 4)
            .iter()
            .chain(b.iter())
            .map(|x| x.0)
            .collect();
        starts.sort_unstable();
        assert_eq!(starts, pts);
    }

    use super::*;

    #[test]
    fn the_memoized_lse_is_row_lse_bit_for_bit() {
        // FN57: the memo computes each distinct bf16's term once; the bytes must not move -- over
        // the device pick's self-test patterns (NaN / inf / subnormal rows included) at the raw
        // and a sampling temperature
        for p in 0..16 {
            let row = pick_selftest_row(p, 900 + p as u64, 4096);
            let mx = seam_top_k(&row, 1)[0].1;
            for t in [1.0f32, 0.7, 0.3, 1.3] {
                let (a, b) = (row_lse(&row, mx, t), row_lse_memo(&row, mx, t));
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                    "pattern {p} t {t}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn the_device_pick_class_and_selftest_rows_are_as_declared() {
        assert_eq!(pick_class_bytes(false), 256);
        assert_eq!(pick_class_bytes(true), 8 * 5 * 512);
        // eight patterns, each a whole row; pattern 3 puts one value > 32 binades above the rest
        // (the kernel's radix fallback) and pattern 1 ties the whole row
        let n = VOCAB as usize / 2;
        let r1 = pick_selftest_row(1, 1, n);
        assert!(r1.chunks_exact(2).all(|c| c == &r1[..2]));
        let r3 = pick_selftest_row(3, 3, n);
        let top = seam_top_k(&r3, 2);
        assert!(top[0].1 > 1e37 && top[1].1.abs() < 1.0, "{top:?}");
        assert_eq!(pick_selftest_row(6, 6, n).len(), n * 2);
    }

    #[test]
    fn the_pin_law_splits_the_ccds() {
        // rank r on CCD r: the HIP core pair and the pick cpus never overlap, never cross CCDs
        if saber_hip::affinity::current_cpus().map_or(true, |c| c.len() != 24) {
            return; // the law is this box's (24 logical CPUs)
        }
        let (h0, p0) = pin_law(0).unwrap();
        let (h1, p1) = pin_law(1).unwrap();
        assert_eq!(h0, vec![1, 13]);
        assert_eq!(h1, vec![7, 19]);
        assert_eq!(p0.len(), 10);
        for c in &p0 {
            assert!(!h0.contains(c) && (*c % 12) < 6);
        }
        for c in &p1 {
            assert!(!h1.contains(c) && (*c % 12) >= 6);
        }
    }

    #[test]
    fn the_argmax_pick_is_the_seams_top_one() {
        let mut v: Vec<f32> = (0..3000)
            .map(|i| ((i * 7919) % 1000) as f32 * 0.01 - 5.0)
            .collect();
        v[1234] = 9.5;
        v[2345] = 9.5; // a bf16 tie: the lower index
        let r = row(&v);
        assert_eq!(row_argmax(&r).0, seam_top_k(&r, 64)[0].0);
        assert_eq!(row_argmax(&r).0, 1234);
        assert_eq!(
            merge_argmax((10, 2.0), (5, 2.0)),
            5,
            "a cross-half tie: the lower id"
        );
        assert_eq!(merge_argmax((10, 2.0), (5, 1.5)), 10);
        let neg: Vec<f32> = vec![-3.0, -1.0, -1.0, -7.0];
        assert_eq!(row_argmax(&row(&neg)).0, 1, "negative values in key order");
    }

    /// FN22: `saber_kernels::fn22`'s law (the device kernels' model: u64 partials, lanes, the exact
    /// reduce, the select) is the HOST chain's `row_argmax` + `merge_argmax`, ties included.
    #[test]
    fn the_device_chain_law_is_the_host_chains() {
        let half = 2048usize;
        let hu = H as usize;
        let mut s = 11u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 40) as u16
        };
        let e0: Vec<u16> = (0..half * hu).map(|_| rnd() & 0xbfff).collect();
        let e1: Vec<u16> = (0..half * hu).map(|_| rnd() & 0xbfff).collect();
        let b = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for case in 0..64 {
            // few distinct values: dense exact ties inside and across the halves
            let m = if case % 2 == 0 { 0x800f } else { 0xbfff };
            let h0: Vec<u16> = (0..half).map(|_| rnd() & m).collect();
            let h1: Vec<u16> = (0..half).map(|_| rnd() & m).collect();
            let (i0, v0) = row_argmax(&b(&h0));
            let (i1, v1) = row_argmax(&b(&h1));
            let host = merge_argmax((i0 as u32, v0), ((i1 + half) as u32, v1));
            assert_eq!(saber_kernels::fn22::host_pick([&h0, &h1]), host);
            let (dev, row) = saber_kernels::fn22::device_pick([&h0, &h1], [&e0, &e1], false, false);
            assert_eq!(dev, host, "case {case}");
            let src = if (host as usize) < half {
                &e0[host as usize * hu..][..hu]
            } else {
                &e1[(host as usize - half) * hu..][..hu]
            };
            assert_eq!(&row[..], src, "the exchanged row is the checkpoint's");
        }
    }

    /// FN22: a pass's tail is four launches -- argmax, pack, the EXACT 2-rank reduce, select -- at
    /// every decode bucket, and the exchange never reaches the wht6 band.
    #[test]
    fn the_chain_tail_is_four_launches_on_the_exact_reduce() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let d = super::super::kern::Decls::load(&p).expect("manifests");
        let r = saber_kernels::flashnext::FlashNextRules::prod().unwrap();
        for rows in [1u32, 2, 4, 8] {
            let mut em = Emitter::new(&d, &r, Dims::new(2, 1), super::super::plan::unbounded());
            let mut pair = super::super::dev::fake_pair().unwrap();
            let t = TailArgs {
                embh: 1 << 30,
                ids: 1 << 31,
                ctr: 1 << 32,
                sel: 1 << 33,
                rows,
                emb: 1 << 34,
                tie_high: false,
                couple: false,
                i2: None,
            };
            emit_tail(&mut em, &mut pair, 1, 1 << 35, 0, u64::from(rows), t).unwrap();
            let syms: Vec<&str> = em.recs.iter().map(|x| x.sym.as_str()).collect();
            assert_eq!(
                syms,
                [
                    "fn_draft_argmax_part",
                    "fn_draft_pack",
                    "saber_ar_oneshot_2rank_exact_bf16",
                    "fn_draft_select"
                ]
            );
            assert_eq!(em.forward_launches(), saber_kernels::fn22::TAIL_LAUNCHES);
            assert_eq!((pair.calls_exact, pair.calls_wht6), (1, 0));
        }
    }

    /// FN33: an int2 pass's tail is the 9 launches before FN22's pack (x gather, group sums, the
    /// 27B's int2 scorer, the exact top-k pair, the column gather, the weight-row gather, the head's
    /// own GEMM cell, the candidates' argmax) + FN22's last three, at every bucket and pass-1 width;
    /// the rerank GEMM is the head cell's symbol at the pass's rows.
    #[test]
    fn the_int2_tail_is_the_27bs_scorer_plus_the_heads_own_cell() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let d = super::super::kern::Decls::load(&p).expect("manifests");
        let r = saber_kernels::flashnext::FlashNextRules::prod().unwrap();
        for (rows, m) in [(1u32, 1u64), (1, 4), (2, 6), (4, 12), (8, 8), (8, 24)] {
            let mut em = Emitter::new(&d, &r, Dims::new(2, 1), super::super::plan::unbounded());
            let mut pair = super::super::dev::fake_pair().unwrap();
            let t = TailArgs {
                embh: 1 << 30,
                ids: 1 << 31,
                ctr: 1 << 32,
                sel: 1 << 33,
                rows,
                emb: 1 << 34,
                tie_high: false,
                couple: false,
                i2: Some(I2Args {
                    wq: 1 << 36,
                    s: 1 << 37,
                    zs: 1 << 38,
                    head: 1 << 39,
                    head8: None,
                    k: saber_kernels::fn33::DEFAULT_K,
                    full_head: false,
                }),
            };
            let last = emit_tail(&mut em, &mut pair, 1, 0, 1 << 35, m, t)
                .unwrap()
                .expect("an int2 tail records its candidates");
            let (head_sym, _) = r
                .bgemm_route("head.tp2", m as u32, VOCAB as u32 / 2)
                .unwrap();
            let syms: Vec<&str> = em.recs.iter().map(|x| x.sym.as_str()).collect();
            assert_eq!(
                syms,
                [
                    "fn_draft_x_gather",
                    "dflash_head_group_sum_g128",
                    "dflash_draft_head_int2_m16n64x2g128_lx_topk",
                    "dflash_topk_f32_thresh_k80",
                    "dflash_topk_f32_gather_k80",
                    "dflash_head_cand_gather",
                    "fn_draft_wgather",
                    head_sym.symbol.as_str(),
                    "fn_draft_argmax_cand",
                    "fn_draft_pack",
                    "saber_ar_oneshot_2rank_exact_bf16",
                    "fn_draft_select"
                ],
                "rows {rows} m {m}"
            );
            // FN61g: under `w8-head` (no bf16 head) the rerank reads the W8 head: the row gather +
            // the bf16 cell become the W8 head's act quant + the gather GEMM -- same count, and
            // never the bf16 head's address
            let mut e8 = Emitter::new(&d, &r, Dims::new(2, 1), super::super::plan::unbounded());
            let mut pair8 = super::super::dev::fake_pair().unwrap();
            let mut t8 = t;
            t8.i2 = t.i2.map(|i| I2Args {
                head: 0,
                head8: Some((1 << 40, 1 << 41)),
                ..i
            });
            let l8 = emit_tail(&mut e8, &mut pair8, 1, 0, 1 << 35, m, t8)
                .unwrap()
                .expect("an int2 tail records its candidates");
            let s8: Vec<&str> = e8.recs.iter().map(|x| x.sym.as_str()).collect();
            let mut want = syms.clone();
            want[6] = saber_kernels::fn50::ACT_QUANT_SYMBOL;
            want[7] = T8R_GATHER;
            assert_eq!(s8, want, "w8-head rows {rows} m {m}");
            assert_eq!(e8.forward_launches(), post_launches(Some(t8), true, 0));
            assert_eq!((l8.ldc, l8.rows, l8.k), (last.ldc, last.rows, last.k));
            // FN86d family 10: the same tail fused -- 8 glue launches -> 4, every buffer at the
            // chain's address (the rerank GEMM's operands, the payload), the arena unchanged
            {
                use saber_kernels::flashnext::fuse86;
                let mut r10 = r.clone();
                r10.hc.fuse86 = fuse86::ALL;
                let mut e10 =
                    Emitter::new(&d, &r10, Dims::new(2, 1), super::super::plan::unbounded());
                let mut pair10 = super::super::dev::fake_pair().unwrap();
                let l10 = emit_tail(&mut e10, &mut pair10, 1, 0, 1 << 35, m, t8)
                    .unwrap()
                    .expect("an int2 tail records its candidates");
                let s10: Vec<&str> = e10.recs.iter().map(|x| x.sym.as_str()).collect();
                assert_eq!(
                    s10,
                    [
                        fuse86::F10_XG,
                        "dflash_draft_head_int2_m16n64x2g128_lx_topk",
                        "dflash_topk_f32_thresh_k80",
                        fuse86::F10_TOPK,
                        T8R_GATHER,
                        fuse86::F10_ACP,
                        "saber_ar_oneshot_2rank_exact_bf16",
                        "fn_draft_select"
                    ],
                    "f10 rows {rows} m {m}"
                );
                assert_eq!(
                    e10.forward_launches(),
                    post_launches(Some(t8), true, fuse86::ALL)
                );
                assert_eq!(
                    e10.forward_launches() + fuse86::F10_SAVED,
                    e8.forward_launches()
                );
                assert_eq!((l10.cand, l10.sc, l10.ldc), (l8.cand, l8.sc, l8.ldc));
                let args = |e: &Emitter<'_>, k: &str| {
                    e.recs.iter().find(|x| x.sym == k).map(|x| x.args.clone())
                };
                for k in [
                    T8R_GATHER,
                    "dflash_topk_f32_thresh_k80",
                    "dflash_draft_head_int2_m16n64x2g128_lx_topk",
                    "saber_ar_oneshot_2rank_exact_bf16",
                    "fn_draft_select",
                ] {
                    assert_eq!(args(&e10, k), args(&e8, k), "f10 rows {rows}: {k}");
                }
                assert_eq!(e10.arena.scratch.high, e8.arena.scratch.high);
                assert_eq!(e10.fuse86_n[9], [1, 0]);
                // LR-N (`--spec-couple on`): the coupled tail is family 10's chain with the fused
                // argmax + pack and the select swapped for the coupled pair -- the same launches,
                // the same buffers, the chain's input buffer (`ctr`) as the last argument
                let mut ec =
                    Emitter::new(&d, &r10, Dims::new(2, 1), super::super::plan::unbounded());
                let mut pairc = super::super::dev::fake_pair().unwrap();
                let mut tc = t8;
                tc.couple = true;
                emit_tail(&mut ec, &mut pairc, 1, 0, 1 << 35, m, tc).unwrap();
                let sc: Vec<&str> = ec.recs.iter().map(|x| x.sym.as_str()).collect();
                let mut wc = s10.clone();
                wc[5] = "fn_couple_acp";
                wc[7] = "fn_couple_select";
                assert_eq!(sc, wc, "couple rows {rows} m {m}");
                assert_eq!(ec.forward_launches(), e10.forward_launches());
                assert_eq!(ec.arena.scratch.high, e10.arena.scratch.high);
                let ac = args(&ec, "fn_couple_acp").expect("acp");
                let a10 = args(&e10, fuse86::F10_ACP).expect("f10 acp");
                // kernarg bytes: sc .. rank (72 B) equal; then `inb` (= ctr) for `tie_high`
                assert_eq!(
                    ac[..72],
                    a10[..72],
                    "couple rows {rows}: the fused site's operands"
                );
                assert_eq!(
                    ac[72..80],
                    (1u64 << 32).to_le_bytes(),
                    "the chain's input buffer"
                );
                // red, emb, ids, ctr, rows (36 B) equal; no tie law
                assert_eq!(
                    args(&ec, "fn_couple_select").expect("sel")[..36],
                    args(&e10, "fn_draft_select").expect("sel")[..36]
                );
                // without family 10 the coupled tail refuses by name
                let mut en = Emitter::new(&d, &r, Dims::new(2, 1), super::super::plan::unbounded());
                let mut pairn = super::super::dev::fake_pair().unwrap();
                assert!(emit_tail(&mut en, &mut pairn, 1, 0, 1 << 35, m, tc).is_err());
                // the bf16-head rerank (no W8 head) keeps the chain under the same mask
                let mut eb =
                    Emitter::new(&d, &r10, Dims::new(2, 1), super::super::plan::unbounded());
                let mut pb = super::super::dev::fake_pair().unwrap();
                emit_tail(&mut eb, &mut pb, 1, 0, 1 << 35, m, t).unwrap();
                assert_eq!(
                    eb.forward_launches(),
                    post_launches(Some(t), false, fuse86::ALL)
                );
                assert_eq!(eb.forward_launches(), em.forward_launches());
            }
            // NEGATIVE CONTROL: a W8 rerank beside a resident bf16 head refuses
            let mut bad = t8;
            bad.i2 = t8.i2.map(|i| I2Args { head: 1 << 39, ..i });
            let mut eb = Emitter::new(&d, &r, Dims::new(2, 1), super::super::plan::unbounded());
            assert!(emit_tail(&mut eb, &mut pair8, 1, 0, 1 << 35, m, bad).is_err());
            assert_eq!(
                em.forward_launches(),
                post_launches(Some(t), false, 0),
                "rows {rows} m {m}"
            );
            assert_eq!(
                last.ldc,
                saber_kernels::fn33::rerank_cols(rows, saber_kernels::fn33::DEFAULT_K)
            );
            assert_eq!((pair.calls_exact, pair.calls_wht6), (1, 0));
        }
    }

    #[test]
    fn drafts_are_fitted_to_the_step_width() {
        assert_eq!(fit_drafts(&[4, 5, 6], 2, 0), (vec![4, 5], 0));
        assert_eq!(fit_drafts(&[4], 3, 0), (vec![4, 4, 4], 2));
        assert_eq!(fit_drafts(&[], 2, 9), (vec![9, 9], 2));
    }

    fn row(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn the_one_pass_top_k_is_the_full_sort_with_ties_by_index() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for n in [70usize, 1000, 124_160] {
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    st ^= st << 13;
                    st ^= st >> 7;
                    st ^= st << 17;
                    // few distinct values: ties everywhere, negatives and zeros included
                    ((st % 97) as f32 - 60.0) / 4.0
                })
                .collect();
            let r = row(&v);
            let got = seam_top_k(&r, 64);
            let mut ix: Vec<usize> = (0..n).collect();
            let b = |i: usize| bf16_at(&r, i);
            ix.sort_by(|&x, &y| b(y).total_cmp(&b(x)).then(x.cmp(&y)));
            let want: Vec<(usize, f32)> = ix[..64].iter().map(|&i| (i, b(i))).collect();
            assert_eq!(got, want, "n {n}");
        }
    }

    #[test]
    fn row_lse_is_the_f64_formula() {
        let v: Vec<f32> = (0..5000)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let r = row(&v);
        let b: Vec<f64> = (0..v.len()).map(|i| f64::from(bf16_at(&r, i))).collect();
        let mx = b.iter().copied().fold(f64::MIN, f64::max);
        for t in [1.0f32, 0.7] {
            let want = mx / f64::from(t)
                + b.iter()
                    .map(|x| (x / f64::from(t) - mx / f64::from(t)).exp())
                    .sum::<f64>()
                    .ln();
            let got = row_lse(&r, mx as f32, t);
            assert!((f64::from(got) - want).abs() < 1e-5, "{got} vs {want}");
        }
    }
}
