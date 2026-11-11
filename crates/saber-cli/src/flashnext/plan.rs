//! Host-only planning of the Flash-Next prefill forward: the arena is SIZED by emitting the widest
//! chunk (every layer kind, the final mixer and the head) against a region of unbounded capacity and
//! reading each region's high-water mark -- the arena law is the emission itself, not a hand count.
//! The same emission asserts every layer's launch count against `layer_graph` (structural rule 7).

use saber_kernels::flashnext::FlashNextRules;
use saber_model::qwen4_exp::layer_graph::{self, Form, Input, LayerSpec, MoeTrain, Tp};
use saber_model::qwen4_exp::LayerKind;
use std::collections::{BTreeMap, BTreeSet};

use super::dev::fake_pair;
use super::exec::{
    emit_layer, emit_post, ChunkConsts, Dims, Emitter, HcW, LayerIn, LayerState, LayerW, MixerW,
    Pending, PleW, Reduce, Resident, VerifyClass,
};
use super::kern::{Arena, ArenaSizes, Decls};

/// Layers of the target.
pub const LAYERS: u32 = 48;
/// The PLE layer (0-based).
pub const PLE_LAYER: u32 = 1;

/// The kind of decoder layer `l` (QSA at 3, 7, ..., 47).
#[must_use]
pub fn kind(l: u32) -> LayerKind {
    if l % 4 == 3 {
        LayerKind::Qsa
    } else {
        LayerKind::Gdn
    }
}

/// A layer with placeholder addresses (emission only reads them as numbers).
#[must_use]
pub fn fake_layer(l: u32) -> LayerW {
    let hc = HcW {
        w8: false,
        w: 1 << 30,
        wdown: 1 << 30,
        wup: 1 << 30,
        winj: 1 << 30,
    };
    LayerW {
        l,
        ahc: hc,
        mhc: hc,
        mixer: if kind(l) == LayerKind::Qsa {
            MixerW::Qsa {
                wproj: 1 << 30,
                wo: 1 << 30,
                qw: 1 << 30,
                kw: 1 << 30,
                iqw: 1 << 30,
                ikw: 1 << 30,
            }
        } else {
            MixerW::Gdn {
                win: 1 << 30,
                conv: 1 << 30,
                alog: 1 << 30,
                dtb: 1 << 30,
                nw: 1 << 30,
                wout: 1 << 30,
            }
        },
        router: 1 << 30,
        sg: 1 << 30,
        ple: (l == PLE_LAYER).then(PleW::default),
        xgu: 1 << 30,
        xdn: 1 << 30,
        mtp: false,
        mtp_fp8: false,
        mtp_x4: false,
        mtp_xtp2: false,
        t8: None,
    }
}

/// FN30: the fused MoE `sync` region's address in a host dry run (never dereferenced).
pub const DRY_MOE_SYNC: u64 = 1 << 45;

/// An arena of unbounded capacity (for sizing).
#[must_use]
pub fn unbounded() -> Arena {
    let big = 1u64 << 40;
    Arena::carve(
        1 << 44,
        ArenaSizes {
            scratch: big,
            zero: big,
            ones: big,
            carry: big,
        },
    )
}

/// What one host emission of a whole chunk produced.
#[derive(Debug, Clone)]
pub struct DryRun {
    pub sizes: ArenaSizes,
    /// Launches per layer (rank 0).
    pub per_layer: Vec<u32>,
    /// Launches of the final mixer + head.
    pub post: u32,
    /// Launches per symbol over the chunk (rank 0).
    pub by_symbol: std::collections::BTreeMap<String, u32>,
    /// FN24: launches per CELL over the chunk (rank 0): `(symbol, grid x, grid z, threads)` -- the
    /// axes that carry a cell's class (a split-K count, the HC mix's `4 sps`, the fused update's
    /// workgroups), not the rows (grid y).
    pub by_cell: std::collections::BTreeMap<(String, u32, u32, u32), u32>,
    /// FN68b: rank 0's K/V append and attention launches `(layer, symbol, kernarg image)` (the
    /// scale law's kernargs).
    #[cfg_attr(not(test), allow(dead_code))]
    pub kv_args: Vec<(i32, String, Vec<u8>)>,
    /// FN80b-2: layer 0's MoE launches on EVERY rank `(rank, symbol, kernarg image)` (the route's
    /// `e_lo` / `e_loc` / `with_shared`, the expert GEMMs' N / K, the silu's n).
    #[cfg_attr(not(test), allow(dead_code))]
    pub moe_args: Vec<(usize, String, Vec<u8>)>,
    /// FN86: rank 0's forward launches in emission order `(layer, symbol, grid, threads)` (the
    /// launch-fusion census; -1 = the final mixer + head).
    #[cfg_attr(not(test), allow(dead_code))]
    pub seq: Vec<(i32, String, [u32; 3], u32)>,
    /// FN120e: the step as ONE glue-fold run (`fold`: every target layer + the head, the ids
    /// block at the base): the (zero, ones) high-water over both ranks; (0, 0) = not armed.
    pub glue: (u64, u64),
    /// FN120e: the runs that spilled (a non-strict region), summed over the ranks.
    #[cfg_attr(not(test), allow(dead_code))]
    pub glue_spills: u64,
}

/// Emit a whole `m`-row prefill chunk at `tp` on the host and size the arena from it.
pub fn dry_run(d: &Decls, rules: &FlashNextRules, tp: u32, m: u32) -> anyhow::Result<DryRun> {
    dry_run_cap(d, rules, tp, m, false)
}

/// [`dry_run`], with the capture's materialized residual (`m x W` bf16 of scratch after each
/// layer) when `capture`.
pub fn dry_run_cap(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    m: u32,
    capture: bool,
) -> anyhow::Result<DryRun> {
    dry_run_cap_t8(d, rules, tp, m, capture, false, false)
}

/// FN61f: [`dry_run_cap`] under `--fn-trunk w8` / `w8-head` (`t8`: the prefill chunk runs on the
/// W8 records, whose activation quants + fp8 operands take arena bytes the bf16 train does not;
/// `head8`: the W8 head). The boot sizes its arena from THIS when the trunk's prefill is W8 (FN61f
/// card fault: the bf16-sized region over-committed by 203,264 B at the narrow rung and 5,636,096 B
/// at a 2,048-row chunk).
pub fn dry_run_cap_t8(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    m: u32,
    capture: bool,
    t8: bool,
    head8: bool,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m,
        nb: m.div_ceil(16) + 1,
        ..ChunkConsts::default()
    };
    dry_run_cc_t8(
        d,
        rules,
        tp,
        capture,
        cc,
        VerifyClass::default(),
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
    )
}

/// FN80b-2: [`dry_run_cap_t8`] with the expert form (`xtp2`: `--fn-experts tp2`'s route and
/// half-width expert GEMMs; the boot sizes its arena from THIS).
#[allow(clippy::too_many_arguments)]
pub fn dry_run_cap_x(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    m: u32,
    capture: bool,
    t8: bool,
    head8: bool,
    xtp2: bool,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m,
        nb: m.div_ceil(16) + 1,
        ..ChunkConsts::default()
    };
    dry_run_cc_kv(
        d,
        rules,
        tp,
        capture,
        cc,
        VerifyClass::default(),
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
        false,
        (0, Vec::new()),
        xtp2,
        false,
    )
}

/// FN80b-2: [`dry_run_prefill_at_t8`] with the expert form.
#[allow(clippy::too_many_arguments)]
pub fn dry_run_prefill_at_x(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    m: u32,
    p0: u32,
    t8: bool,
    head8: bool,
    xtp2: bool,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m,
        p0,
        nb: (p0 + m).div_ceil(16) + 1,
        ..ChunkConsts::default()
    };
    dry_run_cc_kv(
        d,
        rules,
        tp,
        false,
        cc,
        VerifyClass::default(),
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
        false,
        (0, Vec::new()),
        xtp2,
        false,
    )
}

/// [`dry_run_prefill_at_t8`] on the bf16 trunk (the tests' form).
#[cfg(test)]
pub fn dry_run_prefill_at(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    m: u32,
    p0: u32,
) -> anyhow::Result<DryRun> {
    dry_run_prefill_at_t8(d, rules, tp, m, p0, false, false)
}

/// FN19: a prefill chunk of `m` rows whose first row sits at `p0` (the sparse pipeline when its
/// last row passes 2,050: the arena's selection buffers at a long context).
/// FN61f: with the W8 trunk's prefill (`t8`) / head (`head8`) (the boot's sizing under `w8`).
#[cfg_attr(not(test), allow(dead_code))]
pub fn dry_run_prefill_at_t8(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    m: u32,
    p0: u32,
    t8: bool,
    head8: bool,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m,
        p0,
        nb: (p0 + m).div_ceil(16) + 1,
        ..ChunkConsts::default()
    };
    dry_run_cc_t8(
        d,
        rules,
        tp,
        false,
        cc,
        VerifyClass::default(),
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
    )
}

/// A DECODE step of `b` sequences at TP=2 (FN13): the decode-form ops at `max_ctx`'s band.
pub fn dry_run_decode(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    max_ctx: u32,
) -> anyhow::Result<DryRun> {
    dry_run_verify(d, rules, b, 1, max_ctx)
}

/// FN15b: a VERIFY step of `b` sequences at width `w` (M = b x w rows) at TP=2, the conv state
/// at `3 + WMAX - 1` (a width-1 step is [`dry_run_decode`]'s), in the default verify class.
pub fn dry_run_verify(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    max_ctx: u32,
) -> anyhow::Result<DryRun> {
    dry_run_verify_class(d, rules, b, w, max_ctx, VerifyClass::default())
}

/// [`dry_run_verify`] in verify class `vc` (FN24).
pub fn dry_run_verify_class(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    max_ctx: u32,
    vc: VerifyClass,
) -> anyhow::Result<DryRun> {
    dry_run_verify_train(d, rules, b, w, max_ctx, vc, MoeTrain::default())
}

/// [`dry_run_verify_class`] under MoE train `train` (FN30).
pub fn dry_run_verify_train(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    max_ctx: u32,
    vc: VerifyClass,
    train: MoeTrain,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m: b * w,
        b,
        w,
        decode: true,
        nb: max_ctx.div_ceil(16) + 1,
        max_ctx,
        ..ChunkConsts::default()
    };
    dry_run_cc(
        d,
        rules,
        2,
        false,
        cc,
        vc,
        train,
        saber_kernels::moe::RouteForm::default(),
    )
}

/// [`dry_run_verify_train`] at a route lowering (FN49b: `Shipped` = `--fn-route-control`).
#[cfg(test)]
pub fn dry_run_verify_train_route(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    train: MoeTrain,
    route: saber_kernels::moe::RouteForm,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m: b * w,
        b,
        w,
        decode: true,
        nb: 8192u32.div_ceil(16) + 1,
        max_ctx: 8192,
        ..ChunkConsts::default()
    };
    dry_run_cc(d, rules, 2, false, cc, VerifyClass::Width1, train, route)
}

#[allow(clippy::too_many_arguments)]
fn dry_run_cc(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    capture: bool,
    cc: ChunkConsts,
    vc: VerifyClass,
    train: MoeTrain,
    route: saber_kernels::moe::RouteForm,
) -> anyhow::Result<DryRun> {
    dry_run_cc_t8(d, rules, tp, capture, cc, vc, train, route, false, false)
}

/// FN76b: a decode-form (verify) step of `b` sequences at width `w` at `max_ctx`'s band under the
/// W8 trunk (`t8`) / head (`head8`) -- the arena the served decode step takes (the narrow arena's
/// top above it is lendable at decode).
pub fn dry_run_verify_t8(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    max_ctx: u32,
    t8: bool,
    head8: bool,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m: b * w,
        b,
        w,
        decode: true,
        nb: max_ctx.div_ceil(16) + 1,
        max_ctx,
        ..ChunkConsts::default()
    };
    dry_run_cc_t8(
        d,
        rules,
        2,
        false,
        cc,
        VerifyClass::default(),
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
    )
}

/// FN61c: a decode-form dry run with every target layer's W8 trunk records (placeholder
/// addresses) and (`head8`, `--fn-trunk w8-head`) the W8 head at TP=2; FN61d: `--fn-trunk w8` =
/// `t8` with the head bf16.
#[cfg(test)]
pub fn dry_run_t8(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    t8: bool,
    head8: bool,
) -> anyhow::Result<DryRun> {
    let cc = ChunkConsts {
        m: b * w,
        b,
        w,
        decode: true,
        nb: 8192u32.div_ceil(16) + 1,
        max_ctx: 8192,
        ..ChunkConsts::default()
    };
    dry_run_cc_t8(
        d,
        rules,
        2,
        false,
        cc,
        VerifyClass::Width1,
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
    )
}

#[allow(clippy::too_many_arguments)]
fn dry_run_cc_t8(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    capture: bool,
    cc: ChunkConsts,
    vc: VerifyClass,
    train: MoeTrain,
    route: saber_kernels::moe::RouteForm,
    t8: bool,
    head8: bool,
) -> anyhow::Result<DryRun> {
    dry_run_cc_kv(
        d,
        rules,
        tp,
        capture,
        cc,
        vc,
        train,
        route,
        t8,
        head8,
        false,
        (0, Vec::new()),
        false,
        false,
    )
}

/// FN120c: [`dry_run_cc_t8`] (W8 trunk, W8 head, the fused MoE train) with the dataflow form
/// armed (`--fn-dataflow gdn` inside a captured run): (the dry run, rank 0's tables).
#[cfg(test)]
pub fn dry_run_df(
    d: &Decls,
    rules: &FlashNextRules,
    cc: ChunkConsts,
) -> anyhow::Result<(DryRun, super::exec::DfTables)> {
    DF_PROBE.with(|p| *p.borrow_mut() = Some(None));
    let r = dry_run_cc_kv(
        d,
        rules,
        2,
        false,
        cc,
        VerifyClass::Width1,
        MoeTrain::Fused,
        saber_kernels::moe::RouteForm::default(),
        true,
        true,
        false,
        (0, Vec::new()),
        false,
        true,
    );
    let t = DF_PROBE.with(|p| p.borrow_mut().take().flatten());
    Ok((r?, t.ok_or_else(|| anyhow::anyhow!("FN120c: no tables"))?))
}

#[cfg(test)]
thread_local! {
    /// FN120c: rank 0's dataflow tables after [`dry_run_df`].
    static DF_PROBE: std::cell::RefCell<Option<Option<super::exec::DfTables>>> =
        const { std::cell::RefCell::new(None) };
}

/// FN68: a TP=2 step at `cc` with the fp8 K/V pages (`--fn-kv-dtype fp8`): the emission's
/// symbols, counts and arena under the fp8 append / attention twins.
#[cfg(test)]
pub fn dry_run_kv8(
    d: &Decls,
    rules: &FlashNextRules,
    cc: ChunkConsts,
    kv8: bool,
) -> anyhow::Result<DryRun> {
    dry_run_kv8_law(d, rules, cc, kv8, 0, Vec::new())
}

/// FN68b: [`dry_run_kv8`] with the scale law's descale table at `desc` (0 = null pointers) and
/// rank 0's per-layer (K, V) append scales `scale` (empty = 1.0).
#[cfg(test)]
pub fn dry_run_kv8_law(
    d: &Decls,
    rules: &FlashNextRules,
    cc: ChunkConsts,
    kv8: bool,
    desc: u64,
    scale: Vec<(f32, f32)>,
) -> anyhow::Result<DryRun> {
    let vc = VerifyClass::Width1;
    dry_run_cc_kv(
        d,
        rules,
        2,
        false,
        cc,
        vc,
        MoeTrain::default(),
        saber_kernels::moe::RouteForm::default(),
        false,
        false,
        kv8,
        (desc, scale),
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn dry_run_cc_kv(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    capture: bool,
    cc: ChunkConsts,
    vc: VerifyClass,
    train: MoeTrain,
    route: saber_kernels::moe::RouteForm,
    t8: bool,
    head8: bool,
    kv8: bool,
    ds: (u64, Vec<(f32, f32)>),
    xtp2: bool,
    df: bool,
) -> anyhow::Result<DryRun> {
    dry_run_cc_glue(
        d, rules, tp, capture, cc, vc, train, route, t8, head8, kv8, ds, xtp2, df, None,
    )
}

/// FN120e: ONE served decode / verify form emitted as a captured glue-fold run (`fold`, the ids
/// block reserved, every target layer + the head inside ONE run -- the widest run a step can
/// open: FN64's runs split it, and each run restarts the region) at TP=2 on the host: the
/// region's (zero, ones) high-water, max over both ranks. The forms' axes are the step's
/// constants (`cc`: b, w, the band's `max_ctx`, `qsa_short`) and the boot's emitter flags.
#[allow(clippy::too_many_arguments)]
pub fn dry_run_glue(
    d: &Decls,
    rules: &FlashNextRules,
    cc: ChunkConsts,
    vc: VerifyClass,
    train: MoeTrain,
    t8: bool,
    head8: bool,
    xtp2: bool,
) -> anyhow::Result<(u64, u64)> {
    let big = super::exec::GlueRegion {
        zero: 1 << 40,
        ones: 1 << 40,
    };
    dry_run_glue_at(d, rules, cc, vc, train, t8, head8, xtp2, big)
}

/// FN120e: [`dry_run_glue`] in a SERVED (non-strict) region `region`: the whole dry run (its
/// launches, the arena's per-layer seeded parts the spilled seeds land in, the spill count).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn dry_run_glue_spill(
    d: &Decls,
    rules: &FlashNextRules,
    cc: ChunkConsts,
    t8: bool,
    head8: bool,
    region: super::exec::GlueRegion,
) -> anyhow::Result<DryRun> {
    dry_run_cc_glue(
        d,
        rules,
        2,
        false,
        cc,
        VerifyClass::Width1,
        MoeTrain::Fused,
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
        false,
        (0, Vec::new()),
        false,
        false,
        Some((region, false)),
    )
}

/// [`dry_run_glue`] inside a STRICT region `region` (FN120b's law: a take past it refuses with
/// the engine's own message) -- the census's negative control replays FN120b's fixed region.
#[allow(clippy::too_many_arguments)]
pub fn dry_run_glue_at(
    d: &Decls,
    rules: &FlashNextRules,
    cc: ChunkConsts,
    vc: VerifyClass,
    train: MoeTrain,
    t8: bool,
    head8: bool,
    xtp2: bool,
    region: super::exec::GlueRegion,
) -> anyhow::Result<(u64, u64)> {
    let r = dry_run_cc_glue(
        d,
        rules,
        2,
        false,
        cc,
        vc,
        train,
        saber_kernels::moe::RouteForm::default(),
        t8,
        head8,
        false,
        (0, Vec::new()),
        xtp2,
        false,
        Some((region, true)),
    )?;
    Ok(r.glue)
}

#[allow(clippy::too_many_arguments)]
fn dry_run_cc_glue(
    d: &Decls,
    rules: &FlashNextRules,
    tp: u32,
    capture: bool,
    cc: ChunkConsts,
    vc: VerifyClass,
    train: MoeTrain,
    route: saber_kernels::moe::RouteForm,
    t8: bool,
    head8: bool,
    kv8: bool,
    (desc, scale): (u64, Vec<(f32, f32)>),
    xtp2: bool,
    df: bool,
    glue: Option<(super::exec::GlueRegion, bool)>,
) -> anyhow::Result<DryRun> {
    let m = cc.m;
    let mut ems: Vec<Emitter<'_>> = (0..tp)
        .map(|r| {
            // FN80b-2: the TP-split expert form's route / expert half (its arena)
            let dm = if xtp2 {
                Dims::new(tp, r).experts_tp2()
            } else {
                Dims::new(tp, r)
            };
            let mut e = Emitter::new(d, rules, dm, unbounded());
            e.kv8 = kv8;
            e.kv8_mtp = kv8;
            e.kv_desc = desc;
            e.kv_scale.clone_from(&scale);
            e.vclass = vc;
            e.moe_train = train;
            e.route_form = route;
            e.moe_sync = DRY_MOE_SYNC;
            if t8 && head8 {
                e.t8_head = Some((1 << 30, 1 << 30));
                // FN61g: `w8-head` drops the bf16 head (every post reads the W8 head)
                e.head_bf16_dropped = true;
            }
            // FN61e: `w8` runs prefill on the W8 records too
            e.t8_prefill = t8;
            if df {
                // FN120c: the tables at a fake base, armed as inside a captured run
                let mut t = super::exec::DfTables::new(1 << 41, false);
                t.armed = true;
                e.df = Some(t);
            }
            e
        })
        .collect();
    if let Some((region, strict)) = glue {
        // FN120e: the region at a fake base (the census: strict, unbounded -- it never spills),
        // the run opened as the engine opens it (`fold`: the ids block first)
        for e in &mut ems {
            let mut g = super::exec::GlueSeeds::sized(1 << 42, region, true, strict);
            g.begin(u64::from(m))?;
            e.glue = Some(g);
        }
    }
    if cc.w > 1 {
        for e in &mut ems {
            e.conv_l = u64::from(3 + super::exec::WMAX - 1);
        }
    }
    let st = LayerState::default();
    let mut pair = fake_pair()?;
    // FN38: the mailbox the boot allocates for a chunk of this height
    pair.mb = super::dev::ar_mb(cc.m.max(super::run::MAX_ROWS));
    let mut ins: Vec<LayerIn> = vec![LayerIn::Embed(1 << 31); tp as usize];
    let mut per_layer = Vec::new();
    let mut by_symbol = std::collections::BTreeMap::new();
    let mut by_cell = std::collections::BTreeMap::new();
    let mut kv_args = Vec::new();
    let mut moe_args = Vec::new();
    let mut seq = Vec::new();
    for l in 0..LAYERS {
        let mut lws = vec![fake_layer(l); tp as usize];
        // FN122c: the `dq` family's dequantized matrices take arena bytes only on fp8 mixers --
        // size for them whenever the family is on (a bf16-mixer boot over-sizes by ~13 MiB)
        if rules.hc.prefill122 & saber_kernels::flashnext::prefill122::DQ != 0 {
            for lw in &mut lws {
                lw.ahc.w8 = true;
                lw.mhc.w8 = true;
            }
        }
        if t8 {
            let dm = Dims::new(tp, 0);
            for lw in &mut lws {
                lw.t8 = Some(super::exec::T8W {
                    w: 1 << 30,
                    s: 1 << 30,
                    n: if kind(l) == LayerKind::Qsa {
                        dm.idx_off
                    } else {
                        dm.conv + dm.zn
                    },
                    wo: 1 << 30,
                    so: 1 << 30,
                    sfx: 0,
                });
            }
        }
        let mut ar: Option<&mut dyn Reduce> = if tp == 2 { Some(&mut pair) } else { None };
        let mut hook = Resident(&lws);
        let outs = emit_layer(
            &mut ems,
            &lws,
            &vec![st; tp as usize],
            &ins,
            &vec![1u64 << 31; tp as usize],
            &vec![cc; tp as usize],
            &mut ar,
            &mut hook,
        )?;
        per_layer.push(outs[0].launches);
        if capture {
            for e in &mut ems {
                e.arena
                    .take(u64::from(m) * super::exec::W * 2, super::kern::Seed::None)?;
            }
        }
        ins = outs.iter().map(|o| LayerIn::Pending(o.pend)).collect();
        if l == 0 {
            for (ri, e) in ems.iter().enumerate() {
                for r in e.recs.iter().filter(|r| r.sym.starts_with("moe_")) {
                    moe_args.push((ri, r.sym.clone(), r.args.clone()));
                }
            }
        }
        for r in ems[0].recs.drain(..) {
            if !r.capture {
                seq.push((r.layer, r.sym.clone(), r.grid, r.block));
            }
            *by_cell
                .entry((r.sym.clone(), r.grid[0], r.grid[2], r.block))
                .or_insert(0) += 1;
            if r.sym.contains("kv_append")
                || (r.sym.starts_with("fn_attn_") && !r.sym.starts_with("fn_attn_gate"))
                || r.sym.starts_with("fn_qsa_attn_")
            {
                kv_args.push((r.layer, r.sym.clone(), r.args.clone()));
            }
            *by_symbol.entry(r.sym).or_insert(0) += 1;
        }
        for e in ems.iter_mut().skip(1) {
            e.recs.clear();
        }
    }
    let pend: Vec<Pending> = ins
        .iter()
        .map(|i| match i {
            LayerIn::Pending(p) => *p,
            LayerIn::Embed(_) | LayerIn::Stream(_) => unreachable!(),
        })
        .collect();
    let n0 = ems[0].forward_launches();
    for (r, em) in ems.iter_mut().enumerate() {
        emit_post(
            em,
            &HcW::default(),
            1 << 30,
            &pend[r],
            u64::from(m),
            cc.decode,
        )?;
    }
    let post = ems[0].forward_launches() - n0;
    for r in ems[0].recs.drain(..) {
        if !r.capture {
            seq.push((-1, r.sym.clone(), r.grid, r.block));
        }
        *by_cell
            .entry((r.sym.clone(), r.grid[0], r.grid[2], r.block))
            .or_insert(0) += 1;
        *by_symbol.entry(r.sym).or_insert(0) += 1;
    }
    #[cfg(test)]
    if df {
        DF_PROBE.with(|p| *p.borrow_mut() = Some(ems[0].df.take()));
    }
    let glue = ems
        .iter()
        .filter_map(|e| e.glue.as_ref())
        .fold((0u64, 0u64), |(z, o), g| (z.max(g.zhi), o.max(g.ohi)));
    let glue_spills = ems
        .iter()
        .filter_map(|e| e.glue.as_ref())
        .map(|g| g.spills)
        .sum();
    let a = &ems[0].arena;
    let sizes = ArenaSizes {
        scratch: a.scratch.high,
        zero: a.zero.high,
        ones: a.ones.high,
        carry: a.carry[0].high.max(a.carry[1].high),
    };
    Ok(DryRun {
        sizes,
        per_layer,
        post,
        by_symbol,
        by_cell,
        kv_args,
        moe_args,
        seq,
        glue,
        glue_spills,
    })
}

/// FN15: emit ONE MTP pass at TP=2 on the host (fake weights): `(launches of rank 0, the arena
/// high-water marks)`. The pass = the pre-layer 4 + a QSA layer (the MTP's, `mtp: true`: bf16
/// block input, 16-row MoE tiles) + the MTP mixer 2 + the head 1.
pub fn dry_run_mtp(
    d: &Decls,
    rules: &FlashNextRules,
    m: u32,
    decode: bool,
    max_ctx: u32,
) -> anyhow::Result<(u32, ArenaSizes)> {
    dry_run_mtp_w(d, rules, m, decode, max_ctx, 1)
}

/// [`dry_run_mtp`] of a decode-form pass at `w` rows a sequence (FN15b: the drafter's pass 1 over
/// a verify step's rows, `m / w` sequences at q_len `w`).
pub fn dry_run_mtp_w(
    d: &Decls,
    rules: &FlashNextRules,
    m: u32,
    decode: bool,
    max_ctx: u32,
    w: u32,
) -> anyhow::Result<(u32, ArenaSizes)> {
    dry_run_mtp_arm(
        d,
        rules,
        m,
        decode,
        max_ctx,
        w,
        crate::flashnext::exec::MtpX::Bf16,
    )
}

/// [`dry_run_mtp_w`] with the MTP experts' arm: `Fp8` = FN50's mixed table, `Slab4` = FN110's
/// MXFP4 slab on the trunk's train.
pub fn dry_run_mtp_arm(
    d: &Decls,
    rules: &FlashNextRules,
    m: u32,
    decode: bool,
    max_ctx: u32,
    w: u32,
    x: crate::flashnext::exec::MtpX,
) -> anyhow::Result<(u32, ArenaSizes)> {
    dry_run_mtp_h8(d, rules, m, decode, max_ctx, w, x, false)
}

/// FN61g: [`dry_run_mtp_arm`] with the head's form -- `head8` = `--fn-trunk w8-head` (the bf16 head
/// is not resident: the drafter's head reads the W8 head, its activation quant in the arena).
#[allow(clippy::too_many_arguments)]
pub fn dry_run_mtp_h8(
    d: &Decls,
    rules: &FlashNextRules,
    m: u32,
    decode: bool,
    max_ctx: u32,
    w: u32,
    x: crate::flashnext::exec::MtpX,
    head8: bool,
) -> anyhow::Result<(u32, ArenaSizes)> {
    let (n, a, _) = dry_run_mtp_kv(
        d,
        rules,
        (m, decode, max_ctx, w),
        (x, head8),
        &|_| {},
        &|_| {},
    )?;
    Ok((n, a))
}

/// FN68d: [`dry_run_mtp_h8`] with each emitter and the pass's constants adjusted (`em` / `ccf`:
/// the KV page dtypes, the tier's tables) -> `(launches, arena, rank 0's K/V launches (layer,
/// symbol, kernargs))`.
#[allow(clippy::type_complexity)]
pub fn dry_run_mtp_kv(
    d: &Decls,
    rules: &FlashNextRules,
    mdw: (u32, bool, u32, u32),
    xh: (crate::flashnext::exec::MtpX, bool),
    em: &dyn Fn(&mut Emitter<'_>),
    ccf: &dyn Fn(&mut ChunkConsts),
) -> anyhow::Result<(u32, ArenaSizes, Vec<(i32, String, Vec<u8>)>)> {
    dry_run_mtp_recs(d, rules, mdw, xh, em, ccf, &|s: &str| {
        s.contains("kv_append")
            || (s.starts_with("fn_attn_") && !s.starts_with("fn_attn_gate"))
            || s.starts_with("fn_qsa_attn_")
    })
}

/// FN86d: [`dry_run_mtp_kv`] keeping rank 0's launches whose symbol `keep` accepts (the census
/// keeps every one).
#[allow(clippy::type_complexity)]
pub fn dry_run_mtp_recs(
    d: &Decls,
    rules: &FlashNextRules,
    (m, decode, max_ctx, w): (u32, bool, u32, u32),
    (x, head8): (crate::flashnext::exec::MtpX, bool),
    em: &dyn Fn(&mut Emitter<'_>),
    ccf: &dyn Fn(&mut ChunkConsts),
    keep: &dyn Fn(&str) -> bool,
) -> anyhow::Result<(u32, ArenaSizes, Vec<(i32, String, Vec<u8>)>)> {
    let mut cc = ChunkConsts {
        m,
        b: m / w,
        w,
        decode,
        nb: max_ctx.div_ceil(16) + 1,
        max_ctx,
        ..ChunkConsts::default()
    };
    ccf(&mut cc);
    let mut ems: Vec<Emitter<'_>> = (0..2)
        .map(|r| {
            let mut e = Emitter::new(d, rules, Dims::new(2, r), unbounded());
            // FN138: the half slab's pass runs inside a tp2 target (its MoE at the target's dims)
            if x == crate::flashnext::exec::MtpX::Slab4Tp2 {
                e.dims = e.dims.experts_tp2();
            }
            e.moe_sync = DRY_MOE_SYNC;
            if head8 {
                e.t8_head = Some((1 << 30, 1 << 30));
                e.head_bf16_dropped = true;
            }
            em(&mut e);
            e
        })
        .collect();
    let mut lw = fake_layer(3);
    lw.l = 48;
    lw.mtp = true;
    lw.mtp_fp8 = x == crate::flashnext::exec::MtpX::Fp8;
    lw.mtp_x4 = x.slab4();
    lw.mtp_xtp2 = x == crate::flashnext::exec::MtpX::Slab4Tp2;
    let w = super::exec::MtpW {
        ne: 1 << 30,
        nh: 1 << 30,
        fce: 1 << 30,
        fch: 1 << 30,
        lw,
        fm: HcW::default(),
    };
    let mut ins = Vec::new();
    for em in &mut ems {
        ins.push(LayerIn::Stream(em.mtp_pre(
            &w,
            1 << 31,
            1 << 31,
            u64::from(m),
        )?));
    }
    let mut pair = fake_pair()?;
    let mut ar: Option<&mut dyn Reduce> = Some(&mut pair);
    let lws = vec![lw; 2];
    let mut hook = Resident(&lws);
    let outs = emit_layer(
        &mut ems,
        &lws,
        &[LayerState::default(); 2],
        &ins,
        &[0, 0],
        &[cc; 2],
        &mut ar,
        &mut hook,
    )?;
    for (r, em) in ems.iter_mut().enumerate() {
        super::exec::emit_post3(
            em,
            &HcW::default(),
            1 << 30,
            &outs[r].pend,
            u64::from(m),
            decode,
        )?;
    }
    let kv: Vec<(i32, String, Vec<u8>)> = ems[0]
        .recs
        .iter()
        .filter(|r| keep(&r.sym))
        .map(|r| (r.layer, r.sym.clone(), r.args.clone()))
        .collect();
    let a = &ems[0].arena;
    Ok((
        ems[0].forward_launches(),
        ArenaSizes {
            scratch: a.scratch.high,
            zero: a.zero.high,
            ones: a.ones.high,
            carry: a.carry[0].high.max(a.carry[1].high),
        },
        kv,
    ))
}

/// FN15b / FN24: one op's cell in a SPEC-0 decode step (M = b rows) and in a verify step
/// (M = b x w) of the verify class the census was taken in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CensusRow {
    pub op: &'static str,
    pub spec0: String,
    pub verify: String,
    /// FN24: the verify's cell is keyed on the class rows (`false`: a law the verify class does
    /// not move -- the all-reduce's payload band, the attention's route at q_len w).
    pub routed: bool,
}

impl CensusRow {
    /// The two cells are one class (the verify rows' bytes can equal SPEC-0's at this op).
    #[must_use]
    pub fn same(&self) -> bool {
        self.spec0 == self.verify
    }
}

/// FN15b: gate D's premise, the host census -- for every M-dependent op of the target's decode
/// step, the cell a SPEC-0 step of `b` sequences runs (M = b) against the cell a width-`w` verify
/// step of the same `b` runs (M = b x w) in verify class `vc`: the bf16 GEMM band + split-K, the HC
/// mix `sps` + symbols, the MXFP4 expert cell (split-K / tile), the decode attention's route
/// (symbol + splits), the GDN decode form, the GDN state carry between rows, the all-reduce's
/// exact / wht6 band. FN24: in the width-1 class every routed op reads its cell at M = b (the
/// executor's `Emitter::rm`), and every such cell runs M = b x w (`band_rows` tiles any M, the
/// HC's `sps` is a kernarg, the grouped experts and the fused update work-loop); only the
/// all-reduce's band (a payload law) stays at b x w.
pub fn verify_census(
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    max_ctx: u32,
    vc: VerifyClass,
) -> anyhow::Result<Vec<CensusRow>> {
    use saber_kernels::flashnext::{MixMode, MixOut};
    use saber_kernels::moe::{measured_cell, MoeGemm};
    let dm = Dims::new(2, 0);
    let (m0, m1) = (b, b * w);
    // the class rows of the verify's routed cells
    let c1 = if vc == VerifyClass::Width1 { m0 } else { m1 };
    let mut out = Vec::new();
    let gemm = |shape: &str, m: u32, n: u64| -> anyhow::Result<String> {
        let (l, sp) = rules.bgemm_route(shape, m, n as u32)?;
        Ok(format!("{} sk{sp}", l.symbol))
    };
    for (op, shape, n) in [
        ("gdn in_proj", "gdn.in_proj.tp2", dm.gn),
        ("qsa proj", "qsa.proj.tp2", dm.qn),
        ("out_proj (gdn + qsa)", "gdn.out_proj.tp2", super::exec::H),
        ("ple key_proj", "ple.key_proj", super::exec::W),
        ("ple value_proj", "ple.value_proj", super::exec::H),
        ("moe router", "moe.router", 512),
        ("head", "head.tp2", super::exec::VOCAB / 2),
    ] {
        out.push(CensusRow {
            op,
            spec0: gemm(shape, m0, n)?,
            verify: gemm(shape, c1, n)?,
            routed: true,
        });
    }
    for (op, mode, o) in [
        ("hc mix layer 0 (expand)", MixMode::Expand, MixOut::Bf16),
        ("hc mix attn (combine)", MixMode::Combine, MixOut::Bf16),
        ("hc mix mlp (combine, q8)", MixMode::Combine, MixOut::Q8),
        ("hc mix ple (mix)", MixMode::Mix, MixOut::Bf16),
        ("hc final mixer", MixMode::Final, MixOut::Bf16),
    ] {
        let c = |cm: u32, m: u32| -> anyhow::Result<String> {
            let p = rules.mix_plan_at(mode, o, cm, m)?;
            Ok(format!("{} / {} sps{}", p.down.symbol, p.up.symbol, p.sps))
        };
        out.push(CensusRow {
            op,
            spec0: c(m0, m0)?,
            verify: c(c1, m1)?,
            routed: true,
        });
    }
    for (op, g) in [
        ("moe experts gate_up", MoeGemm::GateUp),
        ("moe experts down", MoeGemm::Down),
    ] {
        let c = |m: u32| {
            let x = measured_cell(g, m);
            format!("ks{} tm{}", x.ks, x.tm)
        };
        out.push(CensusRow {
            op,
            spec0: c(m0),
            verify: c(c1),
            routed: true,
        });
    }
    let att = |q: u32| -> anyhow::Result<String> {
        let r =
            saber_kernels::fn6::decode_route(q, max_ctx, b).map_err(|e| anyhow::anyhow!("{e}"))?;
        // the executor runs a declc cell as the decode + combine pair (FN14's which-path counter)
        let sym = if r.form == saber_kernels::fn6::AttnForm::DecodeDeclc {
            format!(
                "fn_attn_decode_h256_gqa12_bf16kv_nw{}",
                saber_kernels::fn6::ROUTED_DECODE_NW
            )
        } else {
            r.symbol.clone()
        };
        Ok(format!("{sym} splits {} pair", r.splits))
    };
    out.push(CensusRow {
        op: "decode attention",
        spec0: att(1)?,
        verify: att(w)?,
        routed: false,
    });
    let gdn = |m: u32| {
        if u64::from(m) * dm.hv <= super::exec::FUSED_MAX_ITEMS {
            "fused update".to_string()
        } else {
            "conv_update + recurrent_update pair".to_string()
        }
    };
    out.push(CensusRow {
        op: "gdn decode form",
        spec0: gdn(m0),
        verify: gdn(c1),
        routed: true,
    });
    let stored = "f16 store + load between a sequence's tokens";
    out.push(CensusRow {
        op: "gdn state between a sequence's rows",
        spec0: stored.into(),
        verify: if w == 1 || vc == VerifyClass::Width1 {
            // FN24: the `fc` bodies round the carried state to f16 after each row: the same bytes
            stored.into()
        } else {
            "fp32 registers across the window (snapshots stored f16)".into()
        },
        routed: true,
    });
    let ar = |m: u32| {
        if decode_ar_exact(m) {
            "exact".to_string()
        } else {
            "wht6 (lossy)".to_string()
        }
    };
    out.push(CensusRow {
        op: "all-reduce (x2 a layer)",
        spec0: ar(m0),
        verify: ar(m1),
        routed: false,
    });
    Ok(out)
}

/// FN78: whether a DECODE-form reduce of `rows` rows runs on the exact kernel at the default law
/// (`--fn-decode-ar exact`): below the payload band (128 KiB) or at <= 64 rows
/// ([`super::dev::DECODE_EXACT_MAX_ROWS`]) -- every decode form (b8 x w5 = 40 rows at most).
#[must_use]
pub fn decode_ar_exact(rows: u32) -> bool {
    u64::from(rows) * super::exec::H * 2 < 131_072
        || u64::from(rows) <= super::dev::DECODE_EXACT_MAX_ROWS
}

/// FN24: the width-1 class law on the EMITTED step, not the table -- a verify of `b` sequences at
/// width `w` in class `vc` against SPEC 0's step at `b`: every launched cell `(symbol, grid x,
/// grid z, threads)` with its count (the rows axis, grid y, is the only one allowed to differ),
/// the f16-carry GDN bodies counted as the shipped ones they replace (same kernargs, same grid),
/// the attention by its symbol + split count (a width-1-class verify runs it as `b x w` one-row
/// pseudo-sequences at SPEC 0's route) and the reduce left out where its payload band moves
/// (b x w x 5,120 B >= 128 KiB).
/// Returns `(cells compared, the differing cells as text -- empty when equal)`.
pub fn verify_cells(
    d: &Decls,
    rules: &FlashNextRules,
    b: u32,
    w: u32,
    max_ctx: u32,
    vc: VerifyClass,
) -> anyhow::Result<(usize, Vec<String>)> {
    use super::exec::{GDN_FUSED_FC, GDN_RECURRENT_FC};
    type Cells = std::collections::BTreeMap<(String, u32, u32, u32), u32>;
    let ar = decode_ar_exact(b * w);
    let cells = |dr: &DryRun| -> Cells {
        let mut m = Cells::new();
        for (k, n) in &dr.by_cell {
            // FN131: a phase-split decode down is its shipped parent's class (byte-equal by the
            // card-1 oracle; the phase waves are not a class axis)
            let d131 = saber_kernels::flashnext::down131::parent(k.0.as_str());
            let k141p = saber_kernels::flashnext::k141::parent(k.0.as_str());
            let s = match k.0.as_str() {
                _ if d131.is_some() => d131.as_ref().map_or("", |x| x.0.as_str()),
                // FN141: a drained-ring twin is its shipped parent's class (byte-equal by the
                // card-1 oracle)
                _ if k141p.is_some() => k141p.unwrap_or(""),
                // the m16 tile's row-slice twin (the same class per 16-row slice)
                "dflash_bgemm_bf16_mtn16" => "dflash_bgemm_bf16_m16n16",
                GDN_FUSED_FC => "saber_gdn_fused_update_f16s_w4k128v128_ab1",
                GDN_RECURRENT_FC => "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1",
                // FN30: route v2 builds FN4's tables by construction (FN26's 800 cases); its band
                // is by the step's rows, a lever, not a class
                // FN49b: the DPP lowering moves no value (the same partner lane, FN49a's oracle)
                saber_kernels::moe::ROUTE_SYMBOL_V2
                | saber_kernels::moe::ROUTE_SYMBOL_LDS
                | saber_kernels::moe::ROUTE_SYMBOL_LDS_DPP => saber_kernels::moe::ROUTE_SYMBOL,
                // FN111: a re-tile is its shipped cell's class (byte-equal by the card-1 oracle;
                // the tile's columns / waves are not a class axis)
                x if x == saber_kernels::flashnext::tiles111::SYMS[0] => {
                    "dflash_gemm_a8w8_bs128_m16n128k128"
                }
                x if x == saber_kernels::flashnext::tiles111::SYMS[1] => {
                    saber_kernels::flashnext::fuse86::Q8W8_SFX
                }
                x if x == saber_kernels::flashnext::tiles111::SYMS[2] => {
                    saber_kernels::moe::ROUTE_SYMBOL
                }
                // FN163: a route twin is its shipped parent's class (byte-equal by the card-1
                // oracle; its waves are not a class axis) -- the b8 route = FN4's route, the b1
                // front = the split-4 DPP front (counted below as the router + the route)
                x if saber_kernels::flashnext::route163::parent(x).is_some() => {
                    saber_kernels::flashnext::route163::parent(x).unwrap_or(x)
                }
                // FN128: a decode-form selection symbol is its parent's class (byte-equal by the
                // card-1 oracle; its threads are not a class axis)
                x if saber_kernels::flashnext::sel128::parent(x).is_some() => {
                    saber_kernels::flashnext::sel128::parent(x).unwrap_or(x)
                }
                x => x,
            };
            let thr = if let Some((_, t)) = &d131 {
                *t
            } else if s == saber_kernels::moe::ROUTE_SYMBOL
                || saber_kernels::flashnext::sel128::parent(k.0.as_str()).is_some()
            {
                256
            } else {
                k.3
            };
            if !ar && s.starts_with("saber_ar_") {
                continue;
            }
            // FN31's front half IS the router GEMM (the band's body at the split in its name) + the
            // route: count it as those two cells (a verify above T 8 runs them apart)
            if let Some(sp) = s
                .strip_prefix("moe_front_e512_nw8_s")
                .map(|x| x.strip_suffix("_dpp").unwrap_or(x))
                .and_then(|x| x.parse::<u32>().ok())
            {
                let thr = saber_kernels::flashnext::band_rows("fn_bgemm_sk_m16").map_or(0, |b| b.2);
                *m.entry(("fn_bgemm_sk_m16".to_string(), 0, sp, thr))
                    .or_insert(0) += n;
                *m.entry((saber_kernels::moe::ROUTE_SYMBOL.to_string(), 0, 1, 256))
                    .or_insert(0) += n;
                continue;
            }
            // the CLASS axes: the HC down's grid x (4 sps), the attention decode's grid x (splits),
            // a GEMM's grid z (split-K); every other axis is N tiles (the same N) or the rows /
            // sequences (the one a verify moves: its attention runs one row a pseudo-sequence)
            let x = if s.starts_with("fn_hc_down_") || s.starts_with("fn_attn_decode_") {
                k.1
            } else {
                0
            };
            let z = if s.starts_with("fn_attn_") { 0 } else { k.2 };
            *m.entry((s.to_string(), x, z, thr)).or_insert(0) += n;
        }
        m
    };
    let s0 = cells(&dry_run_decode(d, rules, b, max_ctx)?);
    let v = cells(&dry_run_verify_class(d, rules, b, w, max_ctx, vc)?);
    let mut diff = Vec::new();
    for k in s0.keys().chain(v.keys()) {
        let (a, c) = (
            s0.get(k).copied().unwrap_or(0),
            v.get(k).copied().unwrap_or(0),
        );
        if a != c
            && !diff
                .iter()
                .any(|x: &String| x.starts_with(&format!("{k:?}")))
        {
            diff.push(format!("{k:?}: SPEC 0 {a} vs verify {c}"));
        }
    }
    Ok((s0.len(), diff))
}

/// The declared launches of layer `l` at `tp` (`layer_graph`, the pending input from layer 1 on).
#[cfg(test)]
#[must_use]
pub fn declared(rules: &FlashNextRules, l: u32, tp: u32, m: u32) -> u32 {
    declared_form(rules, l, tp, m, Form::Prefill)
}

/// The declared launches of layer `l` at `tp` in `form` (a decode step: `m` = the batch).
#[cfg(test)]
#[must_use]
pub fn declared_form(rules: &FlashNextRules, l: u32, tp: u32, m: u32, form: Form) -> u32 {
    declared_form_qsa(rules, l, tp, m, form, false, MoeTrain::default())
}

/// FN116: the declared launches of QSA layer `l`'s attention-side mix + K-side front (its ops
/// through `qsa.kv_append`, the layer graph's own order) -- what `exec::emit_layer_kv_front` emits.
#[must_use]
pub fn declared_qsa_kv_front(rules: &FlashNextRules, l: u32, tp: u32, m: u32, form: Form) -> u32 {
    let ops = layer_graph::layer_ops(
        rules,
        &LayerSpec {
            layer: l,
            kind: kind(l),
            ple: false,
            input: Input::Pending,
            form,
            tp: if tp == 1 { Tp::One } else { Tp::Two },
            rows: m,
            qsa_sparse: false,
            moe: MoeTrain::default(),
        },
    );
    let end = ops
        .iter()
        .position(|o| o.name.ends_with(".qsa.kv_append"))
        .map_or(0, |i| i + 1);
    layer_graph::launches(&ops[..end])
}

/// [`declared_form`] with the step's QSA form (FN19: `saber_kernels::fn19::sparse_step`).
#[must_use]
pub fn declared_form_qsa(
    rules: &FlashNextRules,
    l: u32,
    tp: u32,
    m: u32,
    form: Form,
    qsa_sparse: bool,
    moe: MoeTrain,
) -> u32 {
    let ops = layer_graph::layer_ops(
        rules,
        &LayerSpec {
            layer: l,
            kind: kind(l),
            ple: l == PLE_LAYER,
            input: if l == 0 { Input::Embed } else { Input::Pending },
            form,
            tp: if tp == 1 { Tp::One } else { Tp::Two },
            rows: m,
            qsa_sparse,
            moe,
        },
    );
    layer_graph::launches(&ops)
}

/// FN64 (`--fn-graph-split lazy`): a captured decode step's runs of layers, `[a, b)` in order
/// covering `0..LAYERS`. The step is cut at layer 0, at `ple` (the PLE layer, when the step's late
/// PLE rows cross before it) and at every layer `gated` names (copies in flight for it: its gate
/// makes the stream wait), quantized DOWN to the largest cut point at or below it -- a multiple of
/// `q`, or `ple`. A gate therefore never runs later than the per-layer form's (it may run up to
/// `q - 1` layers earlier), so every layer still reads its placements after its copies landed; the
/// bytes are the per-layer graphs' (the same records, in the same order, on the same stream). The
/// cut points are drawn from `{0, ple} + multiples of q` only, so a step form owns at most
/// `C(points + 1, 2)` distinct runs (q 4 + the PLE layer: 13 points, 91 runs).
#[must_use]
pub fn lazy_runs(gated: impl Fn(u32) -> bool, q: u32, ple: Option<u32>) -> Vec<(u32, u32)> {
    let q = q.max(1);
    let ple = ple.filter(|&p| p > 0 && p < LAYERS);
    let mut cut = vec![false; LAYERS as usize + 1];
    cut[0] = true;
    cut[LAYERS as usize] = true;
    if let Some(p) = ple {
        cut[p as usize] = true;
    }
    for l in (0..LAYERS).filter(|&l| gated(l)) {
        let mut c = l / q * q;
        if let Some(p) = ple {
            if p <= l && p > c {
                c = p;
            }
        }
        cut[c as usize] = true;
    }
    let pts: Vec<u32> = (0..=LAYERS).filter(|&i| cut[i as usize]).collect();
    pts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// FN64b: the cut points [`lazy_runs`] can draw from (`0`, the PLE layer, every multiple of `q`
/// below `LAYERS`, `LAYERS`) and whether each is a FIXED cut (`0`, the PLE layer, `LAYERS`).
fn lazy_points(q: u32, ple: Option<u32>) -> Vec<(u32, bool)> {
    let q = q.max(1);
    let ple = ple.filter(|&p| p > 0 && p < LAYERS);
    (0..=LAYERS)
        .filter_map(|i| {
            let fixed = i == 0 || i == LAYERS || Some(i) == ple;
            (fixed || i % q == 0).then_some((i, fixed))
        })
        .collect()
}

/// FN64b: every run [`lazy_runs`] can return at `q` / `ple` -- `[a, b)` between two cut points
/// with no FIXED cut strictly inside (each optional cut point is cut by a gated layer of its own
/// quantum, independently of the others). q 4 + the PLE layer: 79 runs (+ the head a form).
#[must_use]
pub fn lazy_reachable(q: u32, ple: Option<u32>) -> BTreeSet<(u32, u32)> {
    let pts = lazy_points(q, ple);
    let mut out = BTreeSet::new();
    for (i, &(a, _)) in pts.iter().enumerate() {
        for &(b, fixed) in &pts[i + 1..] {
            out.insert((a, b));
            if fixed {
                break;
            }
        }
    }
    out
}

/// FN64b: step plans (each a tiling of `0..LAYERS` that [`lazy_runs`] itself returns for some
/// gated set) whose runs together are [`lazy_reachable`] -- the boot's pre-capture walks each
/// plan once (a capture pass captures the plan's missing runs and re-emits the held ones dry),
/// so every run a served step can ask for is held before the first request. Greedy: from each
/// cut point take the nearest run not yet covered, else move to the nearest point (up to the next
/// fixed cut) that still has one.
#[must_use]
pub fn lazy_cover(q: u32, ple: Option<u32>) -> Vec<Vec<(u32, u32)>> {
    let pts = lazy_points(q, ple);
    let mut left = lazy_reachable(q, ple);
    let mut plans = Vec::new();
    while !left.is_empty() {
        let mut plan = Vec::new();
        let mut i = 0usize;
        while pts[i].0 < LAYERS {
            let a = pts[i].0;
            // the candidates: the points after `a` up to the next fixed cut
            let mut cands = Vec::new();
            for (j, &(_, fixed)) in pts.iter().enumerate().skip(i + 1) {
                cands.push(j);
                if fixed {
                    break;
                }
            }
            let j = cands
                .iter()
                .copied()
                .find(|&j| left.contains(&(a, pts[j].0)))
                .or_else(|| {
                    cands
                        .iter()
                        .copied()
                        .find(|&j| left.iter().any(|&(x, _)| x == pts[j].0))
                })
                .unwrap_or(*cands.last().expect("a cut point after a"));
            left.remove(&(a, pts[j].0));
            plan.push((a, pts[j].0));
            i = j;
        }
        plans.push(plan);
    }
    plans
}

/// FN64c: step plans (each a tiling of `0..LAYERS` drawn ONLY from `set`) that together walk
/// every run of `set` lying on such a tiling -- the boot's pre-capture of a RECORDED run set (the
/// runs a served boot used, `--fn-graph-lazy-pre recorded`) captures exactly those runs and no
/// others. Each plan is the tiling with the most not-yet-walked runs (fewest runs on a tie): a
/// longest path over the DAG of cut points, so every plan walks at least one new run. Returns
/// (plans, runs of `set` on no tiling -- left to the on-demand path). Runs outside `0..LAYERS`
/// (the head) are ignored: the caller appends the head to every plan.
#[must_use]
pub fn lazy_cover_set(set: &BTreeSet<(u32, u32)>) -> (Vec<Vec<(u32, u32)>>, usize) {
    let runs: Vec<(u32, u32)> = set
        .iter()
        .copied()
        .filter(|&(a, b)| a < b && b <= LAYERS)
        .collect();
    let n = LAYERS as usize;
    // on a tiling: reachable from 0 by runs of the set AND reaching LAYERS
    let mut from0 = vec![false; n + 1];
    from0[0] = true;
    for &(a, b) in &runs {
        if from0[a as usize] {
            from0[b as usize] = true;
        }
    }
    let mut to_end = vec![false; n + 1];
    to_end[n] = true;
    for &(a, b) in runs.iter().rev() {
        if to_end[b as usize] {
            to_end[a as usize] = true;
        }
    }
    let mut left: BTreeSet<(u32, u32)> = runs
        .iter()
        .copied()
        .filter(|&(a, b)| from0[a as usize] && to_end[b as usize])
        .collect();
    let orphans = runs.len() - left.len();
    let usable: Vec<(u32, u32)> = left.iter().copied().collect();
    let mut plans = Vec::new();
    while !left.is_empty() {
        // best[a] = (new runs, -runs) of the best tiling of [a, LAYERS); next[a] = its first run
        let mut best: Vec<Option<(usize, i64)>> = vec![None; n + 1];
        let mut next: Vec<u32> = vec![0; n + 1];
        best[n] = Some((0, 0));
        for a in (0..n).rev() {
            for &(x, b) in usable.iter().filter(|r| r.0 as usize == a) {
                if let Some((g, c)) = best[b as usize] {
                    let cand = (g + usize::from(left.contains(&(x, b))), c - 1);
                    if best[a].is_none_or(|cur| cand > cur) {
                        best[a] = Some(cand);
                        next[a] = b;
                    }
                }
            }
        }
        let Some((gain, _)) = best[0] else { break };
        if gain == 0 {
            break;
        }
        let mut plan = Vec::new();
        let mut a = 0u32;
        while a < LAYERS {
            let b = next[a as usize];
            left.remove(&(a, b));
            plan.push((a, b));
            a = b;
        }
        plans.push(plan);
    }
    (plans, orphans)
}

/// FN64d: a recorded run with its served use count (`[a, b)`, uses).
pub type UsedRun = ((u32, u32), u64);

/// FN64d: the execs a selection of recorded runs makes the boot capture -- per form, its runs on
/// a tiling drawn from the selection ([`lazy_cover_set`]; orphans are left on demand) + the head
/// when any run is captured.
#[must_use]
pub fn lazy_selection_execs<K: Ord>(sel: &BTreeMap<K, BTreeSet<(u32, u32)>>) -> usize {
    sel.values()
        .map(|set| {
            let body = set.iter().filter(|r| r.1 <= LAYERS).count();
            let on = body - lazy_cover_set(set).1;
            if on > 0 {
                on + 1
            } else {
                0
            }
        })
        .sum()
}

/// FN64d (`--fn-graph-exec-budget-mib`): the recorded runs the boot pre-captures within `cap`
/// execs -- the MOST USED runs across every form (a form's head, used by each of its steps,
/// first on a tie), added one at a time while the execs the selection makes
/// ([`lazy_selection_execs`]) stay within `cap`; a run that would pass it is skipped (a less
/// used one may still fit). The rest are captured on demand and the LRU evicts past `cap`.
#[must_use]
pub fn lazy_budget_select<K: Ord + Copy>(
    forms: &BTreeMap<K, Vec<UsedRun>>,
    cap: usize,
) -> BTreeMap<K, BTreeSet<(u32, u32)>> {
    let mut cands: Vec<(u64, bool, K, (u32, u32))> = forms
        .iter()
        .flat_map(|(k, runs)| runs.iter().map(move |&(r, u)| (u, r.0 < LAYERS, *k, r)))
        .collect();
    cands.sort_by(|x, y| {
        y.0.cmp(&x.0)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
            .then(x.3.cmp(&y.3))
    });
    let mut sel: BTreeMap<K, BTreeSet<(u32, u32)>> = BTreeMap::new();
    for (_, _, k, r) in cands {
        sel.entry(k).or_default().insert(r);
        if lazy_selection_execs(&sel) > cap {
            if let Some(v) = sel.get_mut(&k) {
                v.remove(&r);
            }
        }
    }
    // keep only what the boot captures: the runs on a tiling of the selection + the head
    for v in sel.values_mut() {
        let on: BTreeSet<(u32, u32)> = lazy_cover_set(v).0.into_iter().flatten().collect();
        *v = if on.is_empty() {
            BTreeSet::new()
        } else {
            on.into_iter().chain([(LAYERS, LAYERS + 1)]).collect()
        };
    }
    sel.retain(|_, v| !v.is_empty());
    sel
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// FN64: the lazy split's runs -- cover the step in order, cut at the PLE layer, a gated
    /// layer's cut is at or below it and on the quantum's grid, q 1 with every layer gated is the
    /// per-layer form, nothing gated is one run (+ the PLE cut), and the run set a form can reach
    /// is bounded (q 4 + PLE: 91).
    #[test]
    fn fn64_lazy_runs_cover_the_step_and_cut_at_or_before_every_gate() {
        assert_eq!(lazy_runs(|_| false, 4, None), vec![(0, LAYERS)]);
        assert_eq!(
            lazy_runs(|_| false, 4, Some(PLE_LAYER)),
            vec![(0, 1), (1, LAYERS)]
        );
        let per_layer: Vec<(u32, u32)> = (0..LAYERS).map(|l| (l, l + 1)).collect();
        assert_eq!(lazy_runs(|_| true, 1, Some(PLE_LAYER)), per_layer);
        assert_eq!(
            lazy_runs(|l| l == 6 || l == 13 || l == 15, 4, Some(1)),
            vec![(0, 1), (1, 4), (4, 12), (12, LAYERS)]
        );
        assert_eq!(
            lazy_runs(|l| l == 2 || l == 0, 4, Some(1)),
            vec![(0, 1), (1, LAYERS)]
        );
        assert_eq!(
            lazy_runs(|l| l == 2 || l == 47, 1, None),
            vec![(0, 2), (2, 47), (47, LAYERS)]
        );
        // every gated layer lies in a run that starts at or before it on the grid, every run is
        // non-empty and the runs tile 0..LAYERS; the reachable run set is bounded
        let mut seen = std::collections::BTreeSet::new();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let m = x;
            let g = |l: u32| (m >> (l % 64)) & 1 == 1 && (m >> ((l * 7) % 64)) & 1 == 1;
            let r = lazy_runs(g, 4, Some(PLE_LAYER));
            assert_eq!(r.first().unwrap().0, 0);
            assert_eq!(r.last().unwrap().1, LAYERS);
            for w in r.windows(2) {
                assert_eq!(w[0].1, w[1].0);
            }
            for &(a, b) in &r {
                assert!(a < b);
                assert!(a == 0 || a == PLE_LAYER || a % 4 == 0);
                seen.insert((a, b));
            }
            assert!(r.iter().any(|&(a, _)| a == PLE_LAYER));
            for l in (0..LAYERS).filter(|&l| g(l)) {
                let &(a, _) = r.iter().find(|&&(a, b)| a <= l && l < b).unwrap();
                assert!(l - a < 4, "layer {l} gated in a run from {a}");
            }
        }
        assert!(seen.len() <= 91, "{} runs", seen.len());
    }

    /// FN64b: the pre-capture's cover -- every plan is a step [`lazy_runs`] returns (its interior
    /// optional cuts as the gated layers), every reachable run is reached by some gated set, and
    /// the plans' runs are exactly the reachable set (q 4 + PLE: 79 runs; q 4 alone: 78).
    #[test]
    fn fn64b_lazy_cover_is_every_reachable_run() {
        for (q, ple) in [
            (4, Some(PLE_LAYER)),
            (4, None),
            (2, Some(PLE_LAYER)),
            (1, None),
            (8, Some(5)),
        ] {
            let reach = lazy_reachable(q, ple);
            let fixed = |c: u32| c == 0 || c == LAYERS || Some(c) == ple;
            for &(a, b) in &reach {
                let r = lazy_runs(|l| (l == a && !fixed(a)) || (l == b && !fixed(b)), q, ple);
                assert!(
                    r.contains(&(a, b)),
                    "q {q} ple {ple:?}: run [{a}, {b}) unreachable"
                );
            }
            let plans = lazy_cover(q, ple);
            let mut seen = BTreeSet::new();
            for p in &plans {
                let cuts: Vec<u32> = p.iter().map(|r| r.0).filter(|&c| !fixed(c)).collect();
                assert_eq!(
                    &lazy_runs(|l| cuts.contains(&l), q, ple),
                    p,
                    "q {q}: plan {p:?}"
                );
                seen.extend(p.iter().copied());
            }
            assert_eq!(seen, reach, "q {q} ple {ple:?}");
            assert!(plans.len() <= reach.len());
        }
        assert_eq!(lazy_reachable(4, Some(PLE_LAYER)).len(), 79);
        assert_eq!(lazy_reachable(4, None).len(), 78);
        assert!(lazy_cover(4, Some(PLE_LAYER)).len() <= 60);
    }

    /// FN64d: the budget's selection -- never more execs than the cap, the most used runs first
    /// (a form's head with them), a run past the cap skipped while a less used one still fits,
    /// and a cap at the whole set's execs selects all of it.
    #[test]
    fn fn64d_budget_select_keeps_the_most_used_runs_within_the_cap() {
        let head = (LAYERS, LAYERS + 1);
        let mut forms: BTreeMap<u32, Vec<UsedRun>> = BTreeMap::new();
        // form 1: the no-gate step (one run) used 100x, a 2-run tiling used 10x
        forms.insert(
            1,
            vec![
                (head, 110),
                ((0, LAYERS), 100),
                ((0, 24), 10),
                ((24, LAYERS), 10),
            ],
        );
        // form 2: one run used 50x
        forms.insert(2, vec![(head, 50), ((0, LAYERS), 50)]);
        let all = lazy_selection_execs(
            &forms
                .iter()
                .map(|(k, v)| (*k, v.iter().map(|x| x.0).collect()))
                .collect(),
        );
        assert_eq!(all, 6);
        assert_eq!(
            lazy_budget_select(&forms, 6)
                .values()
                .map(BTreeSet::len)
                .sum::<usize>(),
            6
        );
        for cap in 0..=8 {
            let s = lazy_budget_select(&forms, cap);
            assert!(lazy_selection_execs(&s) <= cap, "cap {cap}");
        }
        let s = lazy_budget_select(&forms, 2);
        assert_eq!(s.len(), 1);
        assert_eq!(s[&1], [head, (0, LAYERS)].into_iter().collect());
        let s = lazy_budget_select(&forms, 4);
        assert_eq!(s[&2], [head, (0, LAYERS)].into_iter().collect());
        // cap 5: (0, 24) alone is an orphan (no exec), (24, LAYERS) then completes the tiling: 6
        // execs > 5 -> skipped, so form 1 keeps its one run
        let s = lazy_budget_select(&forms, 5);
        assert_eq!(lazy_selection_execs(&s), 4);
        assert!(lazy_selection_execs(&lazy_budget_select(&forms, 0)) == 0);
    }

    /// FN64c: a recorded set's cover -- every plan tiles `0..LAYERS` from the set only, the plans
    /// walk exactly the set's runs that lie on a tiling, an orphan run (no tiling through it) is
    /// counted and left out, the head is ignored, and the full reachable set at q 4 is covered
    /// in no more plans than FN64b's cover.
    #[test]
    fn fn64c_lazy_cover_set_walks_exactly_the_recorded_runs() {
        let check = |set: &BTreeSet<(u32, u32)>| -> (BTreeSet<(u32, u32)>, usize, usize) {
            let (plans, orphans) = lazy_cover_set(set);
            let mut seen = BTreeSet::new();
            for p in &plans {
                assert_eq!(p.first().map(|r| r.0), Some(0), "{p:?}");
                assert_eq!(p.last().map(|r| r.1), Some(LAYERS), "{p:?}");
                for w in p.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "{p:?}");
                }
                for r in p {
                    assert!(set.contains(r), "plan run {r:?} not in the set");
                }
                seen.extend(p.iter().copied());
            }
            (seen, orphans, plans.len())
        };
        // the served shape: a few tilings' runs
        let mut set = BTreeSet::new();
        for p in [
            lazy_runs(|_| false, 4, Some(PLE_LAYER)),
            lazy_runs(|l| l == 13 || l == 30, 4, Some(PLE_LAYER)),
            lazy_runs(|l| l == 5 || l == 47, 4, Some(PLE_LAYER)),
        ] {
            set.extend(p);
        }
        let (seen, orphans, _) = check(&set);
        assert_eq!(seen, set);
        assert_eq!(orphans, 0);
        // an orphan: [20, 24) with no run ending at 20 or starting at 24
        let mut o = set.clone();
        o.insert((20, 24));
        o.insert((LAYERS, LAYERS + 1));
        let (seen, orphans, _) = check(&o);
        assert_eq!(seen, set);
        assert_eq!(orphans, 1);
        // the whole reachable set
        let reach = lazy_reachable(4, Some(PLE_LAYER));
        let (seen, orphans, n) = check(&reach);
        assert_eq!((seen, orphans), (reach.clone(), 0));
        assert!(n <= lazy_cover(4, Some(PLE_LAYER)).len(), "{n}");
        assert_eq!(check(&BTreeSet::new()).2, 0);
    }

    /// FN116: the K-side front of a QSA layer = the attention-side mix (down + up) + the merged
    /// projection + q / k norm + rope + the indexer + the KV append -- the prefix of the layer's
    /// own op list, never more than the whole layer.
    #[test]
    fn fn116_the_kv_front_is_the_layers_prefix_through_the_append() {
        let r = FlashNextRules::prod().unwrap();
        for m in [1u32, 2, 63, 150, 511, 512] {
            let f = declared_qsa_kv_front(&r, 3, 2, m, Form::Prefill);
            assert_eq!(f, 6, "m {m}");
            assert!(f < declared_form_qsa(&r, 3, 2, m, Form::Prefill, false, MoeTrain::Legacy));
        }
        // a GDN layer has no K-side front
        assert_eq!(declared_qsa_kv_front(&r, 0, 2, 150, Form::Prefill), 0);
    }

    /// FN61c: under `--fn-trunk w8` every target decode / verify layer is its bf16 train + 3
    /// launches (two activation quants, the W8 prefix + bf16 suffix of the merged projection, the W8
    /// out / o projection), the head + 1, the W8 symbols carry their declared kernargs, and the
    /// bf16 emission (`t8` None) runs none of them.
    #[test]
    fn fn61c_the_w8_trunk_is_the_declared_train_plus_its_quants() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for (b, w) in [(1u32, 1u32), (1, 5), (2, 3), (4, 3), (8, 1), (8, 3)] {
            let bf = dry_run_t8(&d, &r, b, w, false, false).unwrap();
            let w8 = dry_run_t8(&d, &r, b, w, true, true).unwrap();
            // FN61d: `--fn-trunk w8` (head bf16) = the same layers, the bf16 head's own post
            let wh = dry_run_t8(&d, &r, b, w, true, false).unwrap();
            assert_eq!(wh.per_layer, w8.per_layer, "b{b} w{w}");
            assert_eq!(wh.post, bf.post, "b{b} w{w}: the head stays bf16");
            let n = |dr: &DryRun, s: &str| dr.by_symbol.get(s).copied().unwrap_or(0);
            assert_eq!(n(&wh, "dflash_act_quant_g128"), 96, "b{b} w{w}");
            // FN61e: <= 16 rows the FN61c / dflash cells, past them the multi-row tile
            let cell = |m: u64| super::super::exec::t8_cell(m, 2560, 2560).0;
            let out_sym = cell(u64::from(b * w));
            // past 16 rows one tile symbol (it takes ldc) runs the merged prefix AND the out / o
            let per = if b * w <= 16 { 48 } else { 96 };
            assert_eq!(n(&wh, out_sym), per, "b{b} w{w}: the out / o projections");
            for (l, (&a, &c)) in bf.per_layer.iter().zip(&w8.per_layer).enumerate() {
                assert_eq!(c, a + super::super::exec::T8_LAYER_EXTRA, "b{b} w{w} L{l}");
            }
            assert_eq!(w8.post, bf.post + 1, "b{b} w{w} head");
            let n = |dr: &DryRun, s: &str| dr.by_symbol.get(s).copied().unwrap_or(0);
            let pre_sym = super::super::exec::t8_cell(u64::from(b * w), 8192, 8240).0;
            assert!(n(&w8, pre_sym) >= per, "b{b} w{w}");
            assert_eq!(n(&w8, super::super::exec::T8_BGEMM_LDC), 48, "b{b} w{w}");
            assert_eq!(n(&w8, "dflash_act_quant_g128"), 97, "b{b} w{w}");
            // every W8 GEMM (48 merged prefixes + 48 out / o + the head) = one per quant
            let mut fam = vec![
                "dflash_gemm_a8w8_bs128_m16n128k128",
                "dflash_gemm_a8w8_bs128_mtn128k128",
                super::super::exec::T8_A8W8_LDC,
            ];
            fam.extend(super::super::exec::T8_TILES.iter().map(|t| t.1));
            let all: u32 = fam.iter().map(|s| n(&w8, s)).sum();
            assert_eq!(all, 97, "b{b} w{w}");
            assert_eq!(n(&bf, super::super::exec::T8_A8W8_LDC), 0);
            assert_eq!(n(&bf, "dflash_act_quant_g128"), 0);
        }
    }

    /// FN61e: under `--fn-trunk w8` a PREFILL chunk is its bf16 train + 3 launches a layer too:
    /// the merged prefix + out / o on the 64-row prefill tile past 64 rows (the verify tiles at
    /// 17..64), the bf16 suffix on the 64-row cell, one activation quant per W8 GEMM; the post (the
    /// head at the chunk's last row) unchanged; the bf16 emission runs none of them.
    #[test]
    fn fn61e_a_w8_prefill_chunk_is_the_declared_train_plus_its_quants() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for m in [96u32, 2048, 8192] {
            let cc = ChunkConsts {
                m,
                nb: m.div_ceil(16) + 1,
                ..ChunkConsts::default()
            };
            let run = |t8: bool| {
                dry_run_cc_t8(
                    &d,
                    &r,
                    2,
                    false,
                    cc,
                    VerifyClass::default(),
                    MoeTrain::default(),
                    saber_kernels::moe::RouteForm::default(),
                    t8,
                    false,
                )
                .unwrap()
            };
            let (bf, w8) = (run(false), run(true));
            for (l, (&a, &c)) in bf.per_layer.iter().zip(&w8.per_layer).enumerate() {
                assert_eq!(c, a + super::super::exec::T8_LAYER_EXTRA, "m{m} L{l}");
            }
            assert_eq!(w8.post, bf.post, "m{m}: the head stays bf16");
            let n = |dr: &DryRun, s: &str| dr.by_symbol.get(s).copied().unwrap_or(0);
            let tile = super::super::exec::t8_cell(u64::from(m), 2560, 2560).0;
            assert_eq!(tile, "fn_t8p_a8w8_p4w8_ldc");
            assert_eq!(n(&w8, tile), 96, "m{m}");
            assert_eq!(n(&w8, super::super::exec::T8_BGEMM_LDC), 48, "m{m}");
            assert_eq!(n(&w8, super::super::exec::T8_BGEMM_M64), 0, "m{m}");
            assert_eq!(n(&w8, "dflash_act_quant_g128"), 96, "m{m}");
            assert_eq!(n(&bf, tile) + n(&bf, "dflash_act_quant_g128"), 0, "m{m}");
        }
    }

    /// FN61f: the boot's arena under `--fn-trunk w8` is sized from the W8 prefill train (the
    /// activation quants + fp8 operands): every region at least the bf16 train's, the whole MORE,
    /// at the prompt-sized chunks and the widest. The card fault (FN61f h1) was a bf16-sized region
    /// refusing the W8 train: 127,723,008 B of 127,519,744 (25-prompt runs) and 2,689,006,592 B of
    /// 2,683,370,496 (2,048 / 8,192-row chunks).
    #[test]
    fn fn61f_the_w8_arena_is_sized_from_the_w8_prefill_train() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for m in [96u32, 512, 2048, 8192] {
            let bf = dry_run_cap(&d, &r, 2, m, false).unwrap().sizes;
            let w8 = dry_run_cap_t8(&d, &r, 2, m, false, true, false)
                .unwrap()
                .sizes;
            let wh = dry_run_cap_t8(&d, &r, 2, m, false, true, true)
                .unwrap()
                .sizes;
            eprintln!(
                "m{m}: bf16 scratch {} zero {} ones {} carry {} | w8 scratch {} zero {} ones {} \
                 carry {}",
                bf.scratch, bf.zero, bf.ones, bf.carry, w8.scratch, w8.zero, w8.ones, w8.carry
            );
            for (a, b) in [
                (bf.scratch, w8.scratch),
                (bf.zero, w8.zero),
                (bf.ones, w8.ones),
                (bf.carry, w8.carry),
                (w8.scratch, wh.scratch),
            ] {
                assert!(b >= a, "m{m}: {b} < {a}");
            }
            assert!(
                w8.total() > bf.total(),
                "m{m}: the W8 train needs more arena"
            );
        }
    }

    /// FN15b: the verify step at every (b, w) emits the declared train (the GDN decode form keyed
    /// on the step's ROWS) and fits the arena the widest prompt already sizes.
    #[test]
    fn a_verify_step_is_the_declared_train_at_every_bucket_and_width() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for b in [1u32, 2, 4, 8] {
            for (w, vc) in (1..=super::super::exec::WMAX)
                .flat_map(|w| [(w, VerifyClass::Width1), (w, VerifyClass::Wide)])
            {
                let dr = dry_run_verify_class(&d, &r, b, w, 8192, vc).unwrap();
                for (l, &got) in dr.per_layer.iter().enumerate() {
                    // FN24: the GDN decode form is keyed on the CLASS rows
                    let form = Form::Decode {
                        batch: if vc == VerifyClass::Width1 { b } else { b * w },
                        attn: saber_kernels::fn6::AttnForm::DecodePair,
                    };
                    // FN19: the pools' 8,192 context runs the sparse pipeline at every decode step
                    assert_eq!(
                        got,
                        declared_form_qsa(&r, l as u32, 2, b * w, form, true, MoeTrain::default()),
                        "b{b} w{w} {vc} L{l}"
                    );
                }
                // the drafter's pass 1 over the verify rows (q_len w, b sequences)
                let (_, ms) = dry_run_mtp_w(&d, &r, b * w, true, 8192, w).unwrap();
                let widest = dry_run(&d, &r, 2, 2048).unwrap();
                assert!(ms.total() <= widest.sizes.total(), "mtp b{b} w{w}");
                assert!(dr.sizes.total() <= widest.sizes.total(), "b{b} w{w}");
            }
        }
    }

    /// FN15b: gate D's premise, printed (`cargo test -p saber-cli verify_census -- --nocapture`):
    /// a width-1 verify is SPEC 0 everywhere; in the `Wide` control a wider one moves the classes
    /// it lists (FN15b's table: b1 15 of 18 at every w, b2 / b4 1 -- the carry --, b4 w5 and b8
    /// w >= 3 the GEMM bands, b8 w >= 4 the reduce); in the width-1 class (FN24) NOTHING moves but
    /// the all-reduce's payload band at b8 w >= 4.
    #[test]
    fn verify_census_names_every_class_move() {
        let r = FlashNextRules::prod().unwrap();
        for vc in [VerifyClass::Wide, VerifyClass::Width1] {
            for b in [1u32, 2, 4, 8] {
                for w in 1..=super::super::exec::WMAX {
                    let rows = verify_census(&r, b, w, 8192, vc).unwrap();
                    if w == 1 {
                        assert!(
                            rows.iter().all(CensusRow::same),
                            "b{b}: a width-1 verify is SPEC 0"
                        );
                        continue;
                    }
                    let moved: Vec<String> = rows
                        .iter()
                        .filter(|x| !x.same())
                        .map(|x| format!("{}: {} -> {}", x.op, x.spec0, x.verify))
                        .collect();
                    println!(
                        "census {vc} b{b} w{w} (M {} vs {}): {} of {} ops move class",
                        b,
                        b * w,
                        moved.len(),
                        rows.len()
                    );
                    for m in &moved {
                        println!("    {m}");
                    }
                    match vc {
                        VerifyClass::Wide => {
                            assert!(
                                moved.iter().any(|m| m.starts_with("gdn state")),
                                "the control carries the GDN state in fp32 across the window"
                            );
                            if b == 1 {
                                assert_eq!(moved.len(), 15, "FN15b's b1 census");
                            }
                        }
                        VerifyClass::Width1 => {
                            // FN78: the decode-form band law keeps the reduce exact at every
                            // (b, w) -- b8 w >= 4 was FN24's one stated exception
                            assert!(moved.is_empty(), "b{b} w{w}: {moved:?}");
                        }
                    }
                }
            }
        }
    }

    /// FN24: the EMITTED step, not the table (`verify_cells`): a width-1-class verify launches
    /// SPEC 0's cells at every (b, w); the `Wide` control differs at b1.
    #[test]
    fn a_width1_verify_emits_spec0s_cells() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for b in [1u32, 2, 4, 8] {
            for w in 2..=super::super::exec::WMAX {
                let (n, diff) = verify_cells(&d, &r, b, w, 8192, VerifyClass::Width1).unwrap();
                assert!(diff.is_empty() && n > 0, "b{b} w{w}: {n} cells, {diff:#?}");
                let (_, c) = verify_cells(&d, &r, b, w, 8192, VerifyClass::Wide).unwrap();
                if b == 1 {
                    assert!(
                        !c.is_empty(),
                        "the Wide control at b1 w{w} runs M = w's cells"
                    );
                }
            }
        }
    }

    fn decls() -> Decls {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        Decls::load(&p).expect("manifests")
    }

    #[test]
    fn host_ar_emits_one_exact_fixed_grid_for_prefill_and_decode_sizes() {
        use super::super::{exec::Reduce, host_ar};
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let decls = Decls::load_for_ar(&path, true).unwrap();
        let rules = FlashNextRules::prod().unwrap();
        let mut pair = fake_pair().unwrap();
        pair.host = true;
        pair.rules = pair.rules.exact_all();
        pair.mb = host_ar::slot_bytes(8192);
        for (rows, decode) in [(1, true), (64, true), (8192, false), (7, true)] {
            let mut e = Emitter::new(&decls, &rules, Dims::new(2, 0), unbounded());
            e.ar_decode = decode;
            pair.reduce(std::slice::from_mut(&mut e), &[1 << 31], rows * super::super::exec::H, false).unwrap();
            assert_eq!(e.recs.len(), 1);
            let r = &e.recs[0];
            assert_eq!((r.plugin, r.sym.as_str()), ("flashnext_ar_host", host_ar::SYMBOL));
            assert_eq!(r.grid, [host_ar::BLOCKS as u32, 1, 1]);
        }
        assert_eq!((pair.calls_exact, pair.calls_wht6), (4, 0));
    }

    /// FN131: every phase-split decode down (`down131::ROUTE` x the four modes) is in
    /// `flashnext_hc131` with its parent's kernarg layout, takes the routed threads, stays at
    /// <= 192 VGPRs with no scratch (the grid stays resident: 112 / 48 workgroups on 32 WGPs at 2+
    /// a WGP) -- NEGATIVE CONTROL: the parents stay in `flashnext_hc8`.
    #[test]
    fn fn131_every_split_down_has_its_parents_layout() {
        use saber_kernels::flashnext::down131 as d131;
        let d = decls();
        let sig = |k: &saber_kernels::manifest::KernelDecl| -> Vec<(u32, u32)> {
            k.arg.iter().map(|x| (x.offset, x.size)).collect()
        };
        for (cfg, sps, _, _) in d131::ROUTE {
            for mode in d131::MODES {
                let ship = format!("fn_hc_down_{mode}_v2_{cfg}_h8");
                let (new, thr) = d131::route(&ship, sps).unwrap();
                let (pa, da) = d.get(&ship).unwrap();
                let (pb, db) = d.get(&new).unwrap_or_else(|_| panic!("{new}: not built"));
                assert_eq!((pa, pb), ("flashnext_hc8", "flashnext_hc131"), "{new}");
                assert_eq!(sig(da), sig(db), "{new}: kernarg layout");
                assert!(db.max_flat_workgroup_size >= thr, "{new}: {thr} threads");
                assert!(db.vgpr_count <= 192, "{new}: {} VGPRs", db.vgpr_count);
                assert_eq!(db.private_segment_fixed_size, 0, "{new}: scratch");
            }
        }
    }

    /// FN131: `--fn-hc-down split` swaps the routed decode downs' symbol + threads only (launch
    /// counts, the arena and the grids unchanged; every other symbol identical) and `default`
    /// emits no fn131 symbol; the width-1 class law holds with the arm on.
    #[test]
    fn fn131_the_split_arm_swaps_symbols_only_and_default_is_the_control() {
        use saber_kernels::flashnext::{down131 as d131, fuse86, prefill122};
        let d = decls();
        let mut off = FlashNextRules::prod().unwrap();
        off.hc.fuse86 = fuse86::ALL;
        // the dry run plans the fp8 mixers (the `_h8` downs) when the dq family is on
        off.hc.prefill122 = prefill122::ALL;
        let mut on = off.clone();
        on.hc.down131 = d131::ALL;
        let run = |r: &FlashNextRules, b: u32, w: u32| {
            let cc = ChunkConsts {
                m: b * w,
                b,
                w,
                decode: true,
                nb: 8192u32.div_ceil(16) + 1,
                max_ctx: 8192,
                qsa_short: true,
                ..ChunkConsts::default()
            };
            dry_run_cc_t8(
                &d,
                r,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                true,
            )
            .unwrap()
        };
        for (b, w, cfg) in [(1u32, 4u32, "m1w3t1d"), (8, 3, "m1w7t1")] {
            let (x0, x1) = (run(&off, b, w), run(&on, b, w));
            assert_eq!(x0.per_layer, x1.per_layer, "b{b} w{w}: launches a layer");
            assert_eq!(x0.post, x1.post);
            assert_eq!(x0.sizes.scratch, x1.sizes.scratch, "b{b} w{w}: arena");
            assert!(
                !x0.by_symbol.keys().any(|k| k.starts_with("fn131_")),
                "default emits an fn131 symbol"
            );
            let mut moved = 0;
            for (k, n0) in &x0.by_symbol {
                let n1 = x1.by_symbol.get(k).copied().unwrap_or(0);
                if let Some((new, _)) = d131::route(k, if cfg == "m1w3t1d" { 4 } else { 2 }) {
                    assert_eq!(n1, 0, "b{b} w{w}: {k} left on");
                    assert_eq!(x1.by_symbol.get(&new).copied().unwrap_or(0), *n0, "{new}");
                    moved += n0;
                } else {
                    assert_eq!(n1, *n0, "b{b} w{w}: {k}");
                }
            }
            assert!(moved >= 94, "b{b} w{w}: {moved} downs routed");
            for (l, sym, _, thr) in &x1.seq {
                if let Some((_, pthr)) = d131::parent(sym) {
                    assert!(sym.contains(cfg), "L{l}: {sym}");
                    assert!(*thr > pthr, "L{l}: {sym} threads {thr}");
                }
            }
        }
        for (b, w) in [(1u32, 2u32), (1, 4), (2, 3), (4, 3), (8, 3)] {
            let (_, diff) = verify_cells(&d, &on, b, w, 8192, VerifyClass::Width1).unwrap();
            assert!(diff.is_empty(), "b{b} w{w}: {diff:?}");
        }
    }

    /// FN141: `--fn-k141 all` swaps symbols only (launches a layer, the arena, the grids'
    /// counts unchanged), every added symbol is in its new object with its parent's kernarg
    /// layout, `none` emits no fn141 symbol, and the width-1 class law holds with the arm on.
    #[test]
    fn fn141_the_k141_arm_swaps_symbols_only_and_none_is_the_control() {
        use saber_kernels::flashnext::{fuse86, k141, prefill122, tiles111};
        let d = decls();
        let sig = |k: &saber_kernels::manifest::KernelDecl| -> Vec<(u32, u32)> {
            k.arg.iter().map(|x| (x.offset, x.size)).collect()
        };
        for (new, ship, obj) in [
            (k141::UP_SHIP[0].1, k141::UP_SHIP[0].0, "flashnext_hc141"),
            (k141::UP_SHIP[1].1, k141::UP_SHIP[1].0, "flashnext_hc141"),
            (k141::T8_SYM, k141::T8_PARENTS[0], "flashnext_t8141"),
            (k141::ROUTER_SYM, k141::ROUTER_PARENT, "flashnext_hc141"),
        ] {
            let (pb, db) = d.get(new).unwrap_or_else(|_| panic!("{new}: not built"));
            let (_, da) = d.get(ship).unwrap();
            assert_eq!(pb, obj, "{new}");
            assert_eq!(sig(da), sig(db), "{new}: kernarg layout vs {ship}");
            assert_eq!(db.private_segment_fixed_size, 0, "{new}: scratch");
        }
        let mut off = FlashNextRules::prod().unwrap();
        off.hc.fuse86 = fuse86::ALL;
        off.hc.tiles111 = tiles111::ALL;
        off.hc.prefill122 = prefill122::ALL;
        let mut on = off.clone();
        on.hc.k141 = k141::ALL;
        let run = |r: &FlashNextRules, b: u32, w: u32| {
            let cc = ChunkConsts {
                m: b * w,
                b,
                w,
                decode: true,
                nb: 8192u32.div_ceil(16) + 1,
                max_ctx: 8192,
                qsa_short: true,
                ..ChunkConsts::default()
            };
            dry_run_cc_t8(
                &d,
                r,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                true,
            )
            .unwrap()
        };
        for (b, w) in [(1u32, 4u32), (8, 3)] {
            let (x0, x1) = (run(&off, b, w), run(&on, b, w));
            assert_eq!(x0.per_layer, x1.per_layer, "b{b} w{w}: launches a layer");
            assert_eq!(x0.post, x1.post);
            assert_eq!(x0.sizes.scratch, x1.sizes.scratch, "b{b} w{w}: arena");
            assert!(
                !x0.by_symbol.keys().any(|k| k.starts_with("fn141_")),
                "none emits an fn141 symbol"
            );
            let tot = |x: &DryRun| x.by_symbol.values().sum::<u32>();
            assert_eq!(tot(&x0), tot(&x1), "b{b} w{w}: launches");
            let n = |x: &DryRun, s: &str| x.by_symbol.get(s).copied().unwrap_or(0);
            // the up: every s1d up moved
            let ups: u32 = k141::UP_SHIP.iter().map(|r| n(&x1, r.1)).sum();
            assert!(ups >= 96, "b{b} w{w}: {ups} ups routed");
            for (ship, new) in k141::UP_SHIP {
                assert_eq!(n(&x1, ship), 0, "b{b} w{w}: {ship} left on");
                assert_eq!(n(&x1, new), n(&x0, ship), "b{b} w{w}: {new}");
            }
            // the t8 ring: every out / o projection at <= 16 rows (48 a step: 36 GDN out + 12 o);
            // b8 w3's 24 rows keep the d2w4 tile
            assert_eq!(
                n(&x1, k141::T8_SYM),
                if b == 1 { 48 } else { 0 },
                "b{b} w{w}: t8 sites"
            );
            // the router ring: b8 only (b1's router rides the fused front)
            assert_eq!(
                n(&x1, k141::ROUTER_SYM),
                if b == 8 { 48 } else { 0 },
                "b{b} w{w}"
            );
        }
        for (b, w) in [(1u32, 2u32), (1, 4), (2, 3), (4, 3), (8, 3)] {
            let (_, diff) = verify_cells(&d, &on, b, w, 8192, VerifyClass::Width1).unwrap();
            assert!(diff.is_empty(), "b{b} w{w}: {diff:?}");
        }
    }

    /// FN163: `--fn-route163 all` swaps symbols only (launches a layer, the arena, the launch count
    /// unchanged), every added symbol is in `moe_route163` with its parent's kernarg layout, `none`
    /// emits no fn163 symbol, and the width-1 class law holds with the arm on.
    #[test]
    fn fn163_the_route163_arm_swaps_symbols_only_and_none_is_the_control() {
        use saber_kernels::flashnext::{fuse86, k141, prefill122, route163 as r163, tiles111};
        let d = decls();
        let sig = |k: &saber_kernels::manifest::KernelDecl| -> Vec<(u32, u32)> {
            k.arg.iter().map(|x| (x.offset, x.size)).collect()
        };
        for (new, ship) in [
            (r163::ROUTE_SYM, r163::ROUTE_PARENTS[0]),
            (r163::ROUTE_SYM, r163::ROUTE_PARENTS[1]),
            (r163::ROUTE_SYM, r163::ROUTE_PARENTS[2]),
            (r163::FRONT.1, r163::FRONT.0),
            ("fn163_front_e512_nw8_s2", "moe_front_e512_nw8_s2_dpp"),
        ] {
            let (pb, db) = d.get(new).unwrap_or_else(|_| panic!("{new}: not built"));
            let (_, da) = d.get(ship).unwrap();
            assert_eq!(pb, "moe_route163", "{new}");
            assert_eq!(sig(da), sig(db), "{new}: kernarg layout vs {ship}");
            assert_eq!(db.private_segment_fixed_size, 0, "{new}: scratch");
        }
        let mut off = FlashNextRules::prod().unwrap();
        off.hc.fuse86 = fuse86::ALL;
        off.hc.tiles111 = tiles111::ALL;
        off.hc.prefill122 = prefill122::ALL;
        off.hc.k141 = k141::ALL;
        let mut on = off.clone();
        on.hc.route163 = r163::ALL;
        let run = |r: &FlashNextRules, b: u32, w: u32| {
            let cc = ChunkConsts {
                m: b * w,
                b,
                w,
                decode: true,
                nb: 8192u32.div_ceil(16) + 1,
                max_ctx: 8192,
                qsa_short: true,
                ..ChunkConsts::default()
            };
            dry_run_cc_t8(
                &d,
                r,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                true,
            )
            .unwrap()
        };
        // (b, w, b8 route sites, b1 front sites) a step of 48 MoE layers
        for (b, w, nr, nf) in [
            (1u32, 4u32, 0u32, 48u32),
            (1, 1, 0, 48),
            (2, 3, 0, 0),
            (4, 3, 48, 0),
            (8, 2, 48, 0),
            (8, 3, 48, 0),
            (8, 5, 48, 0),
        ] {
            let (x0, x1) = (run(&off, b, w), run(&on, b, w));
            assert_eq!(x0.per_layer, x1.per_layer, "b{b} w{w}: launches a layer");
            assert_eq!(x0.post, x1.post);
            assert_eq!(x0.sizes.scratch, x1.sizes.scratch, "b{b} w{w}: arena");
            assert!(
                !x0.by_symbol.keys().any(|k| k.starts_with("fn163_")),
                "none emits an fn163 symbol"
            );
            let tot = |x: &DryRun| x.by_symbol.values().sum::<u32>();
            assert_eq!(tot(&x0), tot(&x1), "b{b} w{w}: launches");
            let n = |x: &DryRun, s: &str| x.by_symbol.get(s).copied().unwrap_or(0);
            assert_eq!(n(&x1, r163::ROUTE_SYM), nr, "b{b} w{w}: route sites");
            assert_eq!(n(&x1, r163::FRONT.1), nf, "b{b} w{w}: front sites");
            for p in r163::ROUTE_PARENTS {
                if nr > 0 {
                    assert_eq!(n(&x1, p), 0, "b{b} w{w}: {p} left on");
                }
            }
            if nf > 0 {
                assert_eq!(
                    n(&x1, r163::FRONT.0),
                    0,
                    "b{b} w{w}: the shipped front left on"
                );
            }
        }
        for (b, w) in [(1u32, 2u32), (1, 4), (2, 3), (4, 3), (8, 3)] {
            let (_, diff) = verify_cells(&d, &on, b, w, 8192, VerifyClass::Width1).unwrap();
            assert!(diff.is_empty(), "b{b} w{w}: {diff:?}");
        }
    }

    /// FN122b: every routed prefill tile (`prefill122::HC_ROUTE`) has every down mode (+ the
    /// producer-rr twins) in `flashnext_hc8p` with the SHIPPED m2w21t1_h8's kernarg layout, and
    /// it replaces only that cfg; NEGATIVE CONTROL: the shipped cfg is not in the new object.
    #[test]
    fn fn122b_every_routed_prefill_tile_has_the_shipped_layout() {
        use saber_kernels::flashnext::prefill122 as p122;
        let d = decls();
        let sig = |k: &saber_kernels::manifest::KernelDecl| -> Vec<(u32, u32)> {
            k.arg.iter().map(|x| (x.offset, x.size)).collect()
        };
        for (_, cfg, mf, waves, groups) in p122::HC_ROUTE {
            assert!(waves * 32 <= 1024 && groups >= 1 && mf >= 1);
            for mode in ["mix", "expand", "combine", "final", "combinerr", "finalrr"] {
                let ship = format!("fn_hc_down_{mode}_v2_{}_h8", p122::HC_SHIPPED_CFG);
                let new = format!("fn_hc_down_{mode}_v2_{cfg}_h8");
                let (pa, da) = d.get(&ship).unwrap();
                let (pb, db) = d.get(&new).unwrap_or_else(|_| panic!("{new}: not built"));
                assert_eq!((pa, pb), ("flashnext_hc8", "flashnext_hc8p"), "{new}");
                assert_eq!(sig(da), sig(db), "{new}: kernarg layout");
            }
        }
        let (p, _) = d
            .get(&format!(
                "fn_hc_down_combine_v2_{}_h8",
                p122::HC_SHIPPED_CFG
            ))
            .unwrap();
        assert_ne!(
            p, "flashnext_hc8p",
            "the control: the shipped cfg stays in flashnext_hc8"
        );
        assert!(p122::hc_cfg(1023).is_none() && p122::hc_cfg(1024).is_some());
        // FN122c: the dq down is the q body on the BF16 parent's layout (no `_h8`), the dequant is
        // in the same object, and the dq up is the shipped bf16 p4s (flashnext_hc)
        let (cfg, ..) = p122::DQ_DOWN;
        for mode in ["mix", "expand", "combine", "final"] {
            let ship = format!("fn_hc_down_{mode}_v2_{}", p122::HC_SHIPPED_CFG);
            let new = format!("fn_hc_down_{mode}_v2_{cfg}");
            let (pa, da) = d.get(&ship).unwrap();
            let (pb, db) = d.get(&new).unwrap_or_else(|_| panic!("{new}: not built"));
            assert_eq!((pa, pb), ("flashnext_hc", "flashnext_hc8p"), "{new}");
            assert_eq!(sig(da), sig(db), "{new}: kernarg layout");
        }
        assert_eq!(d.get(p122::DEQUANT).unwrap().0, "flashnext_hc8p");
        for kind in ["bf16", "q8", "q8t"] {
            let up = format!("fn_hc_up_{kind}_v2_{}", p122::DQ_UP_CFG);
            assert_eq!(d.get(&up).unwrap().0, "flashnext_hc", "{up}");
        }
    }

    /// FN108: every mix the target can plan (every band, mode, output kind, the producer-rr twin,
    /// the route control's symbols) has its `_h8` twin in `flashnext_hc8` with the SAME kernarg
    /// layout as its parent, and the prefill p4s up runs on p2s_h8 (32-row blocks, counters with
    /// them); NEGATIVE CONTROL: an unrenamed p4s has no `_h8` twin.
    #[test]
    fn fn108_every_planned_hc_mix_has_its_fp8_twin() {
        use super::super::exec::{hc8_symbol, hc8_up_plan, hc_dpp_control, rr_down_symbol};
        use saber_kernels::flashnext::{MixMode, MixOut};
        let (d, r) = (decls(), FlashNextRules::prod().unwrap());
        let same = |a: &str, b: &str| {
            let (pa, da) = d.get(a).unwrap();
            let (pb, db) = d.get(b).unwrap_or_else(|_| panic!("{b}: no fp8 twin"));
            assert_eq!((pa, pb), ("flashnext_hc", "flashnext_hc8"), "{b}");
            let sig = |k: &saber_kernels::manifest::KernelDecl| -> Vec<(u32, u32)> {
                k.arg.iter().map(|x| (x.offset, x.size)).collect()
            };
            assert_eq!(sig(da), sig(db), "{b}: kernarg layout");
        };
        let mut n = 0;
        for m in [1u64, 2, 8, 17, 64, 65, 300, 512, 513, 2048, 8192] {
            for mode in [
                MixMode::Mix,
                MixMode::Expand,
                MixMode::Combine,
                MixMode::Final,
            ] {
                for out in [MixOut::Bf16, MixOut::Q8, MixOut::Q8Tiled] {
                    let Ok(p0) = r.mix_plan_at(mode, out, m as u32, m as u32) else {
                        continue; // the final mixer has no e4m3 output
                    };
                    for ctl in [false, true] {
                        let mut p = p0.clone();
                        if ctl {
                            p.down.symbol = hc_dpp_control(&p.down.symbol);
                            p.up.symbol = hc_dpp_control(&p.up.symbol);
                        }
                        let mut downs = vec![p.down.symbol.clone()];
                        if matches!(mode, MixMode::Combine | MixMode::Final) {
                            downs.push(rr_down_symbol(&p.down.symbol).unwrap());
                        }
                        for ds in downs {
                            same(&ds, &hc8_symbol(&ds, true));
                            n += 1;
                        }
                        let up0 = p.up.symbol.clone();
                        let mut uc = p.up_counters;
                        hc8_up_plan(&mut p.up, &mut uc, m);
                        same(&up0, &hc8_symbol(&p.up.symbol, true));
                        if up0.ends_with("_p4s") {
                            assert!(
                                d.get(&hc8_symbol(&up0, true)).is_err(),
                                "control: no p4s_h8"
                            );
                            assert_eq!(u64::from(p.up.grid[1]), m.div_ceil(32));
                            assert_eq!(uc, if out == MixOut::Bf16 { 0 } else { p.up.grid[1] });
                        }
                        n += 1;
                    }
                }
            }
        }
        assert_eq!(
            hc8_symbol("fn_hc_up_q8_v2_s1d", false),
            "fn_hc_up_q8_v2_s1d"
        );
        assert!(n > 200, "{n}");
    }

    /// FN68b: the scale law reaches every K/V launch -- the append carries the layer's (K, V)
    /// scale as its two last kernargs, every attention / gather / combine launch of a QSA layer
    /// points `k_descale` / `v_descale` (SABERArgs bytes 40..56) at that layer's rows of the descale
    /// table; under bf16 and under FN68's unit law (desc 0) the pointers stay null and the fp8
    /// append's scales stay 1.0 (FN68's kernargs, byte for byte).
    #[test]
    fn fn68b_the_scale_law_reaches_the_append_and_every_attention_launch() {
        use super::super::kvscale::KvScales;
        let (d, r) = (decls(), FlashNextRules::prod().unwrap());
        let dec = |b: u32, w: u32, ctx: u32, short: bool| ChunkConsts {
            m: b * w,
            b,
            w,
            decode: true,
            nb: ctx.div_ceil(16) + 1,
            max_ctx: ctx,
            qsa_short: short,
            ..ChunkConsts::default()
        };
        let pre = |m: u32, p0: u32| ChunkConsts {
            m,
            p0,
            nb: (p0 + m).div_ceil(16) + 1,
            ..ChunkConsts::default()
        };
        let desc = 0x7000_0000u64;
        let scale: Vec<(f32, f32)> = (0..49)
            .map(|l| (0.5 + l as f32, 0.25 / (1.0 + l as f32)))
            .collect();
        let f32_at = |a: &[u8], o: usize| f32::from_le_bytes(a[o..o + 4].try_into().unwrap());
        let u64_at = |a: &[u8], o: usize| u64::from_le_bytes(a[o..o + 8].try_into().unwrap());
        for (what, cc) in [
            ("prefill dense", pre(2048, 0)),
            ("prefill sparse", pre(2048, 8192)),
            ("decode dense", dec(1, 1, 8192, true)),
            ("decode sparse", dec(1, 1, 8192, false)),
            ("verify b8 w4 64k", dec(8, 4, 65_536, false)),
            ("verify b8 w5 short", dec(8, 5, 8192, true)),
        ] {
            let law = dry_run_kv8_law(&d, &r, cc, true, desc, scale.clone()).unwrap();
            let unit = dry_run_kv8_law(&d, &r, cc, true, 0, Vec::new()).unwrap();
            let bf = dry_run_kv8_law(&d, &r, cc, false, desc, scale.clone()).unwrap();
            assert_eq!(law.kv_args.len(), unit.kv_args.len(), "{what}");
            let (mut napp, mut natt) = (0, 0);
            for ((l, sym, a), (_, _, u)) in law.kv_args.iter().zip(&unit.kv_args) {
                let li = usize::try_from(*l).unwrap();
                if sym.contains("kv_append") {
                    napp += 1;
                    let n = a.len();
                    assert_eq!(f32_at(a, n - 8), scale[li].0, "{what} {sym} L{l}");
                    assert_eq!(f32_at(a, n - 4), scale[li].1, "{what} {sym} L{l}");
                    assert_eq!((f32_at(u, n - 8), f32_at(u, n - 4)), (1.0, 1.0), "{what}");
                    assert_eq!(a[..n - 8], u[..n - 8], "{what} {sym}: only the scales move");
                } else {
                    natt += 1;
                    assert_eq!(
                        u64_at(a, 40),
                        desc + KvScales::table_off(li, 0),
                        "{what} {sym} L{l}"
                    );
                    assert_eq!(
                        u64_at(a, 48),
                        desc + KvScales::table_off(li, 1),
                        "{what} {sym} L{l}"
                    );
                    assert_eq!((u64_at(u, 40), u64_at(u, 48)), (0, 0), "{what} {sym}");
                    assert_eq!(a[..40], u[..40], "{what} {sym}");
                    assert_eq!(a[56..], u[56..], "{what} {sym}");
                }
            }
            assert_eq!(napp, 12, "{what}: one append a QSA layer");
            assert!(natt >= 12, "{what}");
            for (_, sym, a) in &bf.kv_args {
                if !sym.contains("kv_append") {
                    assert_eq!((u64_at(a, 40), u64_at(a, 48)), (0, 0), "{what} bf16 {sym}");
                }
            }
        }
    }

    /// FN68: `--fn-kv-dtype fp8` changes WHICH symbols run and nothing else -- every launch of
    /// the bf16 step at the same position with the same grid / threads, the append and every
    /// attention / gather symbol swapped for its fp8 twin (named in the committed manifests: the
    /// emission resolves each through the kernel declarations), the same launch counts and the
    /// same arena; no bf16-KV symbol is left in an fp8 step (dense prefill, sparse prefill, dense
    /// and sparse decode, a b8 x 4 verify at 64k).
    #[test]
    fn fn68_fp8_kv_swaps_the_append_and_attention_for_their_twins_only() {
        let (d, r) = (decls(), FlashNextRules::prod().unwrap());
        let dec = |b: u32, w: u32, ctx: u32, short: bool| ChunkConsts {
            m: b * w,
            b,
            w,
            decode: true,
            nb: ctx.div_ceil(16) + 1,
            max_ctx: ctx,
            qsa_short: short,
            ..ChunkConsts::default()
        };
        let pre = |m: u32, p0: u32| ChunkConsts {
            m,
            p0,
            nb: (p0 + m).div_ceil(16) + 1,
            ..ChunkConsts::default()
        };
        let twin = |s: &str| {
            s.replace("_bf16kv", "_fp8kv")
                .replace("fn_kv_append_ring_h256_bf16", "fn_kv_append_ring_h256_fp8")
        };
        for (what, cc) in [
            ("prefill dense", pre(2048, 0)),
            ("prefill sparse", pre(2048, 8192)),
            ("decode dense", dec(1, 1, 8192, true)),
            ("decode sparse", dec(1, 1, 8192, false)),
            ("verify b8 w4 64k", dec(8, 4, 65_536, false)),
        ] {
            let b = dry_run_kv8(&d, &r, cc, false).unwrap();
            let f = dry_run_kv8(&d, &r, cc, true).unwrap();
            assert_eq!((&f.per_layer, f.post), (&b.per_layer, b.post), "{what}");
            assert_eq!(f.sizes, b.sizes, "{what}");
            let mapped: std::collections::BTreeMap<_, _> = b
                .by_cell
                .iter()
                .map(|((s, x, z, t), n)| ((twin(s), *x, *z, *t), *n))
                .collect();
            assert_eq!(f.by_cell, mapped, "{what}");
            assert!(
                b.by_symbol.contains_key("fn_kv_append_ring_h256_bf16"),
                "{what}"
            );
            assert!(
                f.by_symbol
                    .keys()
                    .all(|s| !s.contains("bf16kv") && !s.ends_with("_h256_bf16")),
                "{what}: {:?}",
                f.by_symbol
                    .keys()
                    .filter(|s| s.contains("bf16"))
                    .collect::<Vec<_>>()
            );
            let swapped = b.by_symbol.keys().filter(|s| twin(s) != **s).count();
            assert!(swapped >= 2, "{what}: the append and an attention symbol");
        }
    }

    #[test]
    fn the_mtp_pass_is_the_declared_train_and_fits_the_prefill_arena() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let big = dry_run(&d, &r, 2, 2048).unwrap().sizes;
        assert!(
            dry_run_mtp(&d, &r, 513, false, 4096).is_err(),
            "fc_hidden at 4 x 513 rows has no band"
        );
        for (m, decode) in [
            (1, true),
            (2, true),
            (4, true),
            (8, true),
            (16, false),
            (295, false),
            (512, false),
        ] {
            let (got, sz) = dry_run_mtp(&d, &r, m, decode, 4096).unwrap();
            let form = if decode {
                Form::Decode {
                    batch: m,
                    attn: saber_kernels::fn6::AttnForm::DecodePair,
                }
            } else {
                Form::Prefill
            };
            // FN19: a decode-form pass at the pools' 4,096 is sparse; a prompt piece at p0 0 is not
            let want = saber_kernels::fn6::MTP_PRE_LAYER_LAUNCHES as u32
                + declared_form_qsa(&r, 3, 2, m, form, decode, MoeTrain::Legacy)
                + 3;
            assert_eq!(got, want, "m {m} decode {decode}");
            assert!(
                sz.scratch <= big.scratch
                    && sz.zero <= big.zero
                    && sz.ones <= big.ones
                    && sz.carry <= big.carry,
                "m {m}: the MTP pass {sz:?} over the prefill arena {big:?}"
            );
        }
        // the pass is fn6's MTP law + the TP=2 reduces (2) + the head (1)
        let (g1, _) = dry_run_mtp(&d, &r, 1, true, 4096).unwrap();
        assert_eq!(
            g1 as usize,
            saber_kernels::fn6::mtp_pass_launches(
                saber_kernels::fn6::AttnForm::DecodePair,
                saber_kernels::fn6::MtpExperts::Bf16
            ) + 2
                + 1
                + saber_kernels::fn19::selection_launches(1) as usize
        );
    }

    /// FN50: the fp8 MTP experts' pass = the bf16 pass + 4 launches (two per-128 activation quants,
    /// the shared expert's bf16 launch beside each fp8 grouped GEMM), names the mixed-table symbols
    /// and the bf16 band never, and fits the prefill arena at every verify row count.
    #[test]
    fn the_fp8_mtp_pass_is_the_bf16_pass_plus_four_declared_launches() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let widest = dry_run(&d, &r, 2, 2048).unwrap();
        for (m, w) in [(1u32, 1u32), (4, 4), (8, 1), (24, 3), (40, 5)] {
            let (b, _) =
                dry_run_mtp_arm(&d, &r, m, true, 8192, w, crate::flashnext::exec::MtpX::Bf16)
                    .unwrap();
            let (f, sz) =
                dry_run_mtp_arm(&d, &r, m, true, 8192, w, crate::flashnext::exec::MtpX::Fp8)
                    .unwrap();
            assert_eq!(
                f as usize,
                b as usize + saber_kernels::fn50::MOE_BLOCK_LAUNCHES_FP8
                    - saber_kernels::moe::LAUNCHES_PER_MOE_BLOCK,
                "m {m} w {w}"
            );
            assert!(sz.total() <= widest.sizes.total(), "m {m} w {w}");
        }
    }

    /// FN110: the slab4 MTP pass (the MXFP4 records on the trunk's seven-launch W4A8 train, a Q8
    /// mix) declares exactly the bf16 pass's launches (the worker's runtime assert needs no extra
    /// term), at every decode / verify row count and the widest prompt piece, and fits the
    /// prefill arena.
    #[test]
    fn the_slab4_mtp_pass_is_the_bf16_pass_count_on_the_trunk_train() {
        use crate::flashnext::exec::MtpX;
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let widest = dry_run(&d, &r, 2, 2048).unwrap();
        for (m, w, decode) in [
            (1u32, 1u32, true),
            (4, 4, true),
            (8, 1, true),
            (32, 4, true),
            (40, 5, true),
            (512, 1, false),
        ] {
            let (b, _) = dry_run_mtp_arm(&d, &r, m, decode, 8192, w, MtpX::Bf16).unwrap();
            let (x, sz) = dry_run_mtp_arm(&d, &r, m, decode, 8192, w, MtpX::Slab4).unwrap();
            assert_eq!(x, b, "m {m} w {w}: slab4 vs bf16 launches");
            assert!(sz.total() <= widest.sizes.total(), "m {m} w {w}");
        }
    }

    /// FN134: the slab4 MTP pass under `--fn-experts tp2` (the emitters at the TP-split dims) is
    /// the EP pass launch for launch, kernarg for kernarg (rank 0's every record + the arena
    /// sizes): the MTP layer runs at `Dims::mtp_form` (EP split, whole shape) whatever the target's
    /// form. The control: the same emitters' TARGET dims differ (eloc 512, the half shape), and the
    /// pass at those dims would route 512 experts (the route record's eloc kernarg).
    #[test]
    fn fn134_the_slab4_mtp_pass_under_tp2_is_the_ep_pass() {
        use crate::flashnext::exec::MtpX;
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for rk in 0..2u32 {
            let e = Dims::new(2, rk);
            let t = e.experts_tp2();
            assert_eq!(e.mtp_form(), e, "rank {rk}: EP's MTP dims are EP's dims");
            assert_eq!(t.mtp_form(), e, "rank {rk}: tp2's MTP dims are EP's dims");
            assert_ne!(t, e, "control: the target's tp2 dims are not EP's");
            assert_ne!(t.xshape(), t.mtp_form().xshape(), "control: the half shape");
        }
        assert_eq!(Dims::new(1, 0).mtp_form(), Dims::new(1, 0));
        let all = |_: &str| true;
        for (m, w, decode) in [
            (1u32, 1u32, true),
            (4, 4, true),
            (24, 3, true),
            (512, 1, false),
        ] {
            let ep = dry_run_mtp_recs(
                &d,
                &r,
                (m, decode, 8192, w),
                (MtpX::Slab4, true),
                &|_| {},
                &|_| {},
                &all,
            )
            .unwrap();
            let tp = dry_run_mtp_recs(
                &d,
                &r,
                (m, decode, 8192, w),
                (MtpX::Slab4, true),
                &|e| e.dims = e.dims.experts_tp2(),
                &|_| {},
                &all,
            )
            .unwrap();
            assert_eq!(tp.0, ep.0, "m {m} w {w}: launches");
            assert_eq!(tp.1, ep.1, "m {m} w {w}: arena sizes");
            assert_eq!(tp.2.len(), ep.2.len(), "m {m} w {w}: records");
            for (i, (a, b)) in tp.2.iter().zip(&ep.2).enumerate() {
                assert_eq!(a, b, "m {m} w {w}: record {i} ({})", b.1);
            }
            assert!(
                ep.2.iter().any(|x| x.1.starts_with("moe_grouped_w4a8")),
                "m {m} w {w}: the slab4 pass runs the trunk's grouped W4A8 train"
            );
            assert!(
                !tp.2.iter().any(|x| x.1.contains("bk64")),
                "m {m} w {w}: no TP-half down cell in the MTP pass"
            );
        }
    }

    /// FN138: the half slab's MTP pass (`--fn-mtp-split tp2`, the emitters at the target's tp2
    /// dims) is the EP slab4 pass's launch count and train, with the MoE at the TP-split form:
    /// the route over 512 local experts + the shared half, the grouped cells at the TpHalf shape
    /// (gate_up N 640, the DBK-64 down at K 320 on the decode rows), every non-MoE record of the
    /// pass identical to EP's (only the MoE moves: the reduce sums other partials = the
    /// envelope's question); its arena fits the boot's sized forms. Control: the half slab
    /// inside an EP target refuses by name (513 entries need the tp2 dims).
    #[test]
    fn fn138_the_tp2_mtp_pass_is_the_ep_pass_with_the_half_moe() {
        use crate::flashnext::exec::{mtp_x4_dims, MtpX};
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let widest = dry_run(&d, &r, 2, 2048).unwrap();
        let all = |_: &str| true;
        let moe = |s: &str| s.starts_with("moe_") || s.starts_with("fn_moe") || s.contains("route");
        for (m, w, decode) in [
            (1u32, 1u32, true),
            (4, 4, true),
            (8, 1, true),
            (24, 3, true),
            (40, 5, true),
            (512, 1, false),
        ] {
            let ep = dry_run_mtp_recs(
                &d,
                &r,
                (m, decode, 8192, w),
                (MtpX::Slab4, true),
                &|e| e.dims = e.dims.experts_tp2(),
                &|_| {},
                &all,
            )
            .unwrap();
            let tp = dry_run_mtp_recs(
                &d,
                &r,
                (m, decode, 8192, w),
                (MtpX::Slab4Tp2, true),
                &|_| {},
                &|_| {},
                &all,
            )
            .unwrap();
            assert_eq!(tp.0, ep.0, "m {m} w {w}: launches");
            assert_eq!(tp.2.len(), ep.2.len(), "m {m} w {w}: records");
            assert!(
                tp.1.total() <= widest.sizes.total(),
                "m {m} w {w}: the half pass fits the widest arena"
            );
            let (mut same, mut moved) = (0, 0);
            for (i, (a, b)) in tp.2.iter().zip(&ep.2).enumerate() {
                if moe(&b.1) {
                    if a != b {
                        moved += 1;
                    }
                } else {
                    // the arena offsets after the MoE may shift with its larger tables; the
                    // symbols of the non-MoE launches are EP's one for one
                    assert_eq!(a.1, b.1, "m {m} w {w}: record {i}");
                    same += 1;
                }
            }
            assert!(
                same > 0 && moved > 0,
                "m {m} w {w}: {same} same / {moved} moved"
            );
            let gu: Vec<&String> =
                tp.2.iter()
                    .filter(|x| x.1.starts_with("moe_grouped"))
                    .map(|x| &x.1)
                    .collect();
            assert_eq!(gu.len(), 2, "m {m} w {w}: gate_up + down grouped ({gu:?})");
            if decode && m <= 64 {
                assert!(
                    gu[1].contains("bk64"),
                    "m {m} w {w}: the TP-half down at DBK 64 ({})",
                    gu[1]
                );
            }
            assert!(
                !ep.2.iter().any(|x| x.1.contains("bk64")),
                "m {m} w {w}: control, the EP slab4 pass has no TP-half cell"
            );
        }
        // the dims: the half slab runs at the target's tp2 dims; an EP target refuses it
        let mut lw = fake_layer(3);
        lw.mtp = true;
        lw.mtp_x4 = true;
        lw.mtp_xtp2 = true;
        for rk in 0..2 {
            let t = Dims::new(2, rk).experts_tp2();
            assert_eq!(mtp_x4_dims(&lw, t).unwrap(), t);
            assert!(mtp_x4_dims(&lw, Dims::new(2, rk))
                .unwrap_err()
                .to_string()
                .contains("--fn-mtp-split tp2"));
        }
        lw.mtp_xtp2 = false;
        assert_eq!(
            mtp_x4_dims(&lw, Dims::new(2, 1).experts_tp2()).unwrap(),
            Dims::new(2, 1)
        );
    }

    /// FN19: a sparse prefill chunk (its last row past 2,050) and a sparse decode step emit the
    /// declared train -- the QSA layer's attention = score + top-k per 256-row slice, the union and
    /// the gather body (+ the combine at decode) -- and fit an arena of < 2 GiB at a 16k context.
    #[test]
    fn a_sparse_chunk_and_step_are_the_declared_train() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for (m, p0) in [(2048u32, 14_336u32), (4, 2048), (1000, 3000)] {
            let dr = dry_run_prefill_at(&d, &r, 2, m, p0).unwrap();
            for (l, &n) in dr.per_layer.iter().enumerate() {
                assert_eq!(
                    n,
                    declared_form_qsa(&r, l as u32, 2, m, Form::Prefill, true, MoeTrain::default()),
                    "m {m} p0 {p0} layer {l}"
                );
            }
            assert_eq!(
                dr.per_layer[3],
                20 - 1 + saber_kernels::fn19::selection_launches(m) + 1,
                "m {m}: the QSA layer"
            );
            assert_eq!(
                dr.by_symbol.get(saber_kernels::fn19::PREFILL_SYMBOL),
                Some(&12)
            );
            assert!(dr.sizes.total() < 2 << 30, "m {m} p0 {p0}: {:?}", dr.sizes);
        }
        // a chunk whose last row is 2,050 is dense (the model's identity selection)
        let dn = dry_run_prefill_at(&d, &r, 2, 2048, 3).unwrap();
        assert_eq!(dn.by_symbol.get(saber_kernels::fn19::PREFILL_SYMBOL), None);
        for b in [1u32, 8] {
            let dr = dry_run_decode(&d, &r, b, 8192).unwrap();
            let form = Form::Decode {
                batch: b,
                attn: saber_kernels::fn6::AttnForm::DecodePair,
            };
            for (l, &n) in dr.per_layer.iter().enumerate() {
                assert_eq!(
                    n,
                    declared_form_qsa(&r, l as u32, 2, b, form, true, MoeTrain::default()),
                    "b {b} layer {l}"
                );
            }
            let total: u32 = dr.per_layer.iter().sum::<u32>() + dr.post;
            if b == 1 {
                assert_eq!(total, 871 + 12 * 3, "a b1 sparse decode step's launches");
            }
        }
    }

    /// FN37: the producer-rr law at every served (b, w) is a symbol swap only -- every EXACT
    /// block-output reduce is the producer, every combine / final down its rr twin (all but layer
    /// 1's PLE mix consume a producer's x' / rr), launch counts unchanged; off = the control.
    #[test]
    fn the_producer_rr_law_swaps_symbols_only_and_off_is_the_control() {
        let d = decls();
        let off = FlashNextRules::prod().unwrap();
        let mut on = off.clone();
        on.hc.rr_producer = true;
        let n = |x: &DryRun, p: &str| -> u32 {
            x.by_symbol
                .iter()
                .filter(|(k, _)| k.starts_with(p))
                .map(|(_, v)| *v)
                .sum()
        };
        for b in [1u32, 2, 4, 8] {
            for w in 1..=super::super::exec::WMAX {
                let a = dry_run_verify(&d, &on, b, w, 8192).unwrap();
                let c = dry_run_verify(&d, &off, b, w, 8192).unwrap();
                assert_eq!(a.per_layer, c.per_layer, "b{b} w{w}");
                assert_eq!(a.post, c.post, "b{b} w{w}");
                assert_eq!(n(&c, "fn_ar_exact_hcrr_"), 0);
                assert_eq!(
                    n(&c, "fn_hc_down_combinerr_") + n(&c, "fn_hc_down_finalrr_"),
                    0
                );
                // FN78: every decode-form reduce is exact at the default law
                let exact = decode_ar_exact(b * w);
                assert!(exact, "b{b} w{w}");
                let prod = n(&a, "fn_ar_exact_hcrr_bf16");
                let rr = n(&a, "fn_hc_down_combinerr_") + n(&a, "fn_hc_down_finalrr_");
                if exact {
                    assert!(prod >= 96, "b{b} w{w}: {prod}");
                    assert_eq!(prod, rr + 1, "b{b} w{w}: the PLE mix alone reads no x'");
                    assert_eq!(n(&a, "fn_hc_down_combine_v2"), 0, "b{b} w{w}");
                    assert_eq!(
                        n(&c, "fn_hc_down_combine_") + n(&c, "fn_hc_down_final_"),
                        rr + n(&a, "fn_hc_down_combine_") + n(&a, "fn_hc_down_final_"),
                        "b{b} w{w}"
                    );
                    assert_eq!(
                        n(&c, "saber_ar_oneshot_2rank_exact_bf16"),
                        n(&a, "saber_ar_oneshot_2rank_exact_bf16") + prod,
                        "b{b} w{w}"
                    );
                } else {
                    assert_eq!(prod + rr, 0, "b{b} w{w}: the wht6 band keeps FN11's path");
                    assert_eq!(a.by_symbol, c.by_symbol, "b{b} w{w}");
                }
            }
        }
        assert_eq!(
            super::super::exec::rr_down_symbol("fn_hc_down_combine_v2_m1w3t1").as_deref(),
            Some("fn_hc_down_combinerr_v2_m1w3t1")
        );
        assert_eq!(
            super::super::exec::rr_down_symbol("fn_hc_down_final_v2_m1w7t1").as_deref(),
            Some("fn_hc_down_finalrr_v2_m1w7t1")
        );
        assert_eq!(
            super::super::exec::rr_down_symbol("fn_hc_down_combine_mf1"),
            None
        );
        assert_eq!(
            super::super::exec::rr_down_symbol("fn_hc_down_mix_v2_m1w3t1"),
            None
        );
    }

    /// FN86: the launch-fusion census (diagnostic, `--ignored --nocapture`): rank 0's verify
    /// launches in emission order at b1 w4 and b8 w3 (W8 trunk, the served defaults).
    #[test]
    #[ignore = "census dump"]
    fn fn86_census_dump() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for (b, w) in [(1u32, 4u32), (8, 3)] {
            // the served step: W8 trunk, fused MoE train, a short context (dense QSA)
            let cc = ChunkConsts {
                m: b * w,
                b,
                w,
                decode: true,
                nb: 8192u32.div_ceil(16) + 1,
                max_ctx: 8192,
                qsa_short: true,
                ..ChunkConsts::default()
            };
            let x = dry_run_cc_t8(
                &d,
                &r,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                false,
            )
            .unwrap();
            let n: u32 = x.per_layer.iter().sum::<u32>() + x.post;
            println!(
                "FN86 b{b} w{w} launches {n} per_layer {:?} post {}",
                x.per_layer, x.post
            );
            for (l, s, g, t) in &x.seq {
                println!("FN86 b{b}w{w} L{l} {s} {}x{}x{} {t}", g[0], g[1], g[2]);
            }
        }
    }

    /// FN111: `--fn-tiles` swaps symbols only -- at b1 w4 the 48 out / o projections run t1's
    /// 64-column cell (twice the workgroups), at b8 w3 the 48 routes run t3's 16-wave LDS route
    /// (512 threads); launch counts and the arena are unchanged and `default` emits no fn111_ symbol.
    #[test]
    fn fn111_the_tiles_arm_swaps_symbols_only_and_default_is_the_control() {
        use saber_kernels::flashnext::{fuse86, tiles111};
        let d = decls();
        let mut off = FlashNextRules::prod().unwrap();
        off.hc.fuse86 = fuse86::ALL;
        let mut on = off.clone();
        on.hc.tiles111 = tiles111::ALL;
        let run = |r: &FlashNextRules, b: u32, w: u32| {
            let cc = ChunkConsts {
                m: b * w,
                b,
                w,
                decode: true,
                nb: 8192u32.div_ceil(16) + 1,
                max_ctx: 8192,
                qsa_short: true,
                ..ChunkConsts::default()
            };
            dry_run_cc_t8(
                &d,
                r,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                true,
            )
            .unwrap()
        };
        let n = |x: &DryRun, k: &str| x.by_symbol.get(k).copied().unwrap_or(0);
        for (b, w, t1, t3) in [(1u32, 4u32, 48u32, 0u32), (8, 3, 0, 48)] {
            let (x0, x1) = (run(&off, b, w), run(&on, b, w));
            assert_eq!(x0.per_layer, x1.per_layer, "b{b} w{w}: launches a layer");
            assert_eq!(x0.post, x1.post);
            assert_eq!(x0.sizes.scratch, x1.sizes.scratch, "b{b} w{w}: arena");
            for sym in tiles111::SYMS {
                assert_eq!(n(&x0, sym), 0, "default emits {sym}");
            }
            assert_eq!(n(&x1, tiles111::SYMS[0]), t1, "b{b} w{w} t1");
            assert_eq!(n(&x1, tiles111::SYMS[1]), 0, "t2 is not in all");
            assert_eq!(n(&x1, tiles111::SYMS[2]), t3, "b{b} w{w} t3");
            assert_eq!(
                n(&x0, "dflash_gemm_a8w8_bs128_m16n128k128")
                    - n(&x1, "dflash_gemm_a8w8_bs128_m16n128k128"),
                t1
            );
            assert_eq!(
                n(&x0, saber_kernels::moe::ROUTE_SYMBOL) - n(&x1, saber_kernels::moe::ROUTE_SYMBOL),
                t3
            );
            for (l, sym, g, thr) in &x1.seq {
                if sym == tiles111::SYMS[0] {
                    assert_eq!(g[0], 40, "L{l}: N 2,560 / 64");
                }
                if sym == tiles111::SYMS[2] {
                    assert_eq!(*thr, 512, "L{l}: 16 waves");
                }
            }
        }
        // the boot's width-1 premise (FN24): every served verify launches SPEC 0's cells with the
        // tiles on (a re-tile is its shipped cell's class)
        for (b, w) in [(1u32, 2u32), (1, 4), (2, 3), (4, 3), (8, 3)] {
            let (_, diff) = verify_cells(&d, &on, b, w, 8192, VerifyClass::Width1).unwrap();
            assert!(diff.is_empty(), "b{b} w{w}: {diff:?}");
        }
    }

    /// FN120c: `--fn-dataflow gdn` folds every b1 verify GDN tail (36 layers: the merged
    /// projection, the fused update + norm + quant, the out projection) into ONE
    /// `fn120_df_gdn_b1` launch over a site whose kernarg images are the train's own bytes; every
    /// other symbol, the arena and the b8 w3 emission (the conv / recurrent pair: not this round)
    /// are unchanged.
    #[test]
    fn fn120c_the_dataflow_arm_folds_the_b1_gdn_tail_and_keeps_every_other_launch() {
        use super::super::exec::{DF120_GRID, DF120_KERNEL, DF120_SITE_BYTES};
        use saber_kernels::flashnext::{fuse86, tiles111};
        let d = decls();
        let mut r = FlashNextRules::prod().unwrap();
        r.hc.fuse86 = fuse86::ALL;
        r.hc.tiles111 = tiles111::ALL;
        let ccf = |b: u32, w: u32| ChunkConsts {
            m: b * w,
            b,
            w,
            decode: true,
            nb: 8192u32.div_ceil(16) + 1,
            max_ctx: 8192,
            qsa_short: true,
            ..ChunkConsts::default()
        };
        let n = |x: &DryRun, k: &str| x.by_symbol.get(k).copied().unwrap_or(0);
        for (b, w, folded) in [(1u32, 4u32, 36u32), (1, 2, 36), (8, 3, 0)] {
            let off = dry_run_cc_t8(
                &d,
                &r,
                2,
                false,
                ccf(b, w),
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                true,
            )
            .unwrap();
            let (on, t) = dry_run_df(&d, &r, ccf(b, w)).unwrap();
            assert_eq!(n(&on, DF120_KERNEL), folded, "b{b} w{w}: folded tails");
            assert_eq!(t.sites[0], u64::from(folded));
            // sites are keyed by content: the dry run's fake layers share weights and arena
            // addresses, so equal tails share a site (the engine's layers differ by weights / state)
            assert!(
                t.used() <= u64::from(folded) && (folded == 0) == (t.used() == 0),
                "b{b} w{w}: {} sites",
                t.used()
            );
            assert_eq!(off.sizes.scratch, on.sizes.scratch, "b{b} w{w}: arena");
            for sym in [fuse86::Q8W8_SFX, fuse86::F5_FUSED] {
                assert_eq!(n(&off, sym) - n(&on, sym), folded, "b{b} w{w}: {sym}");
            }
            assert_eq!(
                n(&off, tiles111::SYMS[0]) - n(&on, tiles111::SYMS[0]),
                folded,
                "b{b} w{w}: the GDN out projections"
            );
            let lo: u32 = off.per_layer.iter().sum();
            let ln: u32 = on.per_layer.iter().sum();
            assert_eq!(lo - ln, 2 * folded, "b{b} w{w}: 2 launches saved a tail");
            for (l, sym, g, thr) in &on.seq {
                if sym == DF120_KERNEL {
                    assert_eq!((g[0], *thr), (DF120_GRID as u32, 256), "L{l}");
                }
            }
            // every other symbol's count is unchanged
            for (k, v) in &off.by_symbol {
                if ![fuse86::Q8W8_SFX, fuse86::F5_FUSED, tiles111::SYMS[0]].contains(&k.as_str()) {
                    assert_eq!(n(&on, k), *v, "b{b} w{w}: {k}");
                }
            }
            // a site: 3 stages (kind, threads, the train's grid), the tasks in stage order
            if let Some((_, s)) = t.pending.first() {
                assert_eq!(s.len() as u64, DF120_SITE_BYTES);
                let u = |o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
                assert_eq!((u(256), u(260)), (1, 128), "stage 0: sfx, mirrored");
                assert_eq!(
                    (u(288), u(292), u(296)),
                    (2, 256, 24),
                    "stage 1: the update, 24 heads"
                );
                assert_eq!(
                    (u(320), u(324), u(328)),
                    (3, 128, 40),
                    "stage 2: n64, N 2,560 / 64"
                );
                assert!(
                    s[..256].iter().all(|&x| x == 0),
                    "the control words start at 0"
                );
            }
        }
    }

    /// FN86b: the fusion arm at every served (b, w) and a W8 prefill chunk: family 1 replaces each
    /// GDN norm (36) and QSA gate (12) + the act quant after it by its `_q8` twin, family 4 each
    /// merged projection's act quant + bf16 suffix (48) by `fn86_q8_bgemm_m16_ldc`; the W8 GEMMs and
    /// every other symbol stay; arena sizes unchanged (the fused forms take the chain's buffers in
    /// its order); `none` (the default) = the control's emission.
    #[test]
    fn fn86b_the_fusion_arm_drops_one_launch_a_site_and_none_is_the_control() {
        use saber_kernels::flashnext::fuse86;
        let d = decls();
        let off = FlashNextRules::prod().unwrap();
        assert_eq!(off.hc.fuse86, 0);
        let n = |x: &DryRun, k: &str| x.by_symbol.get(k).copied().unwrap_or(0);
        let run = |r: &FlashNextRules, cc: ChunkConsts| {
            dry_run_cc_t8(
                &d,
                r,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                false,
            )
            .unwrap()
        };
        let q = "dflash_act_quant_g128";
        let (gf, gr) = (
            super::super::exec::GDN_FUSED_FC,
            super::super::exec::GDN_RECURRENT_FC,
        );
        for mask in [
            fuse86::F1,
            fuse86::F4,
            fuse86::F1 | fuse86::F4,
            fuse86::F1 | fuse86::F5,
            fuse86::F2 | fuse86::F4,
            // the FN86 families' `all` (the f6/f7 flip's control); f6 / f7 are counted below
            fuse86::ALL_FN86D,
        ] {
            let mut on = off.clone();
            on.hc.fuse86 = mask;
            let f1 = mask & fuse86::F1 != 0;
            let f4 = mask & fuse86::F4 != 0;
            let mut forms: Vec<ChunkConsts> = [(1u32, 4u32), (1, 1), (2, 3), (4, 3), (8, 3)]
                .iter()
                .map(|&(b, w)| ChunkConsts {
                    m: b * w,
                    b,
                    w,
                    decode: true,
                    nb: 8192u32.div_ceil(16) + 1,
                    max_ctx: 8192,
                    qsa_short: true,
                    ..ChunkConsts::default()
                })
                .collect();
            forms.push(ChunkConsts {
                m: 512,
                nb: 512u32.div_ceil(16) + 1,
                ..ChunkConsts::default()
            });
            for cc in forms {
                let (a, c) = (run(&on, cc), run(&off, cc));
                let (la, lc) = (
                    a.per_layer.iter().sum::<u32>() + a.post,
                    c.per_layer.iter().sum::<u32>() + c.post,
                );
                // FN86c family 5: every fc decode update (a verify of w > 1) takes the norm-q8
                let f5 = mask & fuse86::F5 != 0 && cc.decode && cc.w > 1;
                let want5 = if f5 { 36 } else { 0 };
                // FN86c family 2: every merged projection of <= F2_MAX_ROWS rows in one launch
                let f2 = mask & fuse86::F2 != 0 && u64::from(cc.m) <= fuse86::F2_MAX_ROWS;
                let want2 = if f2 { 48 } else { 0 };
                let want = if f1 { 48 } else { 0 } + if f4 { 48 } else { 0 };
                assert_eq!(lc - la, want + want5 + want2, "mask {mask} m {}", cc.m);
                assert_eq!(n(&a, fuse86::Q8W8_SFX), want2, "m {}", cc.m);
                assert_eq!(n(&c, q), 96, "m {}", cc.m);
                assert_eq!(n(&a, q), 96 - want, "m {}", cc.m);
                assert_eq!(n(&a, fuse86::GDN_NORM_Q8), if f1 && !f5 { 36 } else { 0 });
                let (b1, fc) = (cc.b == 1, cc.decode && cc.w > 1);
                assert_eq!(n(&a, fuse86::F5_FUSED), if f5 && b1 { 36 } else { 0 });
                assert_eq!(n(&a, fuse86::F5_RECUR), if f5 && !b1 { 36 } else { 0 });
                assert_eq!(n(&c, gf), if fc && b1 { 36 } else { 0 }, "m {}", cc.m);
                assert_eq!(n(&c, gr), if fc && !b1 { 36 } else { 0 }, "m {}", cc.m);
                assert_eq!(
                    n(&a, gf) + n(&a, gr) + n(&a, fuse86::F5_FUSED) + n(&a, fuse86::F5_RECUR),
                    n(&c, gf) + n(&c, gr)
                );
                assert_eq!(n(&a, fuse86::ATTN_GATE_Q8), if f1 { 12 } else { 0 });
                assert_eq!(
                    n(&a, fuse86::GDN_NORM) + n(&a, fuse86::GDN_NORM_Q8) + want5,
                    36
                );
                assert_eq!(n(&a, fuse86::ATTN_GATE) + n(&a, fuse86::ATTN_GATE_Q8), 12);
                assert_eq!(n(&a, fuse86::Q8_BGEMM_LDC), if f4 && !f2 { 48 } else { 0 });
                assert_eq!(
                    n(&a, super::super::exec::T8_BGEMM_LDC),
                    if f4 { 0 } else { 48 }
                );
                assert_eq!(a.sizes, c.sizes, "m {}: the arena", cc.m);
                for (k, v) in &c.by_symbol {
                    if k != q
                        && k != fuse86::GDN_NORM
                        && k != fuse86::ATTN_GATE
                        && k != super::super::exec::T8_BGEMM_LDC
                        && k != gf
                        && k != gr
                        && !(f2
                            && (k == super::super::exec::T8_A8W8_LDC
                                || super::super::exec::T8_TILES.iter().any(|t| t.1 == k)))
                    {
                        assert_eq!(n(&a, k), *v, "m {}: {k}", cc.m);
                    }
                }
            }
        }
        // f6/f7 flip: the default `all` = FN86d's set + ATTN-OPT-3's f6 (the QSA front 3 -> 1 at
        // a target decode / verify of <= F6_MAX_W rows a sequence) + f7 (combine + gate + q8
        // 2 -> 1 at a target decode): 12 QSA layers x (2 + 1) launches fewer at every served
        // (b, w) (the dry run caps w at 5, = F6_MAX_W); a prefill chunk keeps both chains
        let mut all = off.clone();
        all.hc.fuse86 = fuse86::ALL;
        let mut ctl = off.clone();
        ctl.hc.fuse86 = fuse86::ALL_FN86D;
        let mut forms: Vec<ChunkConsts> = [(1u32, 4u32), (1, 1), (2, 3), (4, 3), (8, 3), (1, 5)]
            .iter()
            .map(|&(b, w)| ChunkConsts {
                m: b * w,
                b,
                w,
                decode: true,
                nb: 8192u32.div_ceil(16) + 1,
                max_ctx: 8192,
                qsa_short: true,
                ..ChunkConsts::default()
            })
            .collect();
        forms.push(ChunkConsts {
            m: 512,
            nb: 512u32.div_ceil(16) + 1,
            ..ChunkConsts::default()
        });
        for cc in forms {
            let (a, c) = (run(&all, cc), run(&ctl, cc));
            let (la, lc) = (
                a.per_layer.iter().sum::<u32>() + a.post,
                c.per_layer.iter().sum::<u32>() + c.post,
            );
            let f6 = cc.decode && cc.w.max(1) <= fuse86::F6_MAX_W;
            let f7 = cc.decode;
            let w6 = if f6 { 12 } else { 0 };
            let w7 = if f7 { 12 } else { 0 };
            assert_eq!(
                lc - la,
                w6 * fuse86::F6_SAVED + w7,
                "f6/f7 b {} w {} m {}",
                cc.b,
                cc.w,
                cc.m
            );
            assert_eq!(
                n(&a, fuse86::F6_FRONT_BF16) + n(&a, fuse86::F6_FRONT_FP8),
                w6,
                "m {}",
                cc.m
            );
            assert_eq!(n(&a, fuse86::F7_COMBINE_GATE_Q8), w7, "m {}", cc.m);
            assert_eq!(n(&c, fuse86::F7_COMBINE_GATE_Q8), 0);
            assert_eq!(
                n(&c, fuse86::F6_FRONT_BF16) + n(&c, fuse86::F6_FRONT_FP8),
                0
            );
        }
        assert_eq!(fuse86::parse("none"), Ok(0));
        assert_eq!(fuse86::parse("all"), Ok(fuse86::ALL));
        assert_eq!(fuse86::parse("f4, f1"), Ok(fuse86::F1 | fuse86::F4));
        // ATTN-OPT-3: f6 / f7 are built; f7 composes on f1; since the f6/f7 flip `all` holds them
        assert!(fuse86::parse("f3").unwrap_err().contains("not built"));
        assert!(fuse86::parse("f7").unwrap_err().contains("composes on f1"));
        assert_eq!(fuse86::parse("f6"), Ok(fuse86::F6));
        assert_eq!(fuse86::parse("all,f6,f7"), Ok(fuse86::ALL));
        assert_eq!(
            fuse86::ALL & (fuse86::F6 | fuse86::F7),
            fuse86::F6 | fuse86::F7
        );
        assert_eq!(fuse86::parse("f1,f2,f4,f5,f10"), Ok(fuse86::ALL_FN86D));
        assert!(fuse86::parse("f9").unwrap_err().contains("not built"));
        assert!(fuse86::parse("f11").unwrap_err().contains("unknown"));
        assert_eq!(fuse86::parse("f10"), Ok(fuse86::F10));
        assert!(fuse86::parse("f5").unwrap_err().contains("composes on f1"));
        assert_eq!(fuse86::names(fuse86::ALL), "f1,f2,f4,f5,f6,f7,f10");
        assert_eq!(fuse86::names(fuse86::ALL_FN86D), "f1,f2,f4,f5,f10");
        assert_eq!(fuse86::names(fuse86::ALL_FN86C), "f1,f2,f4,f5");
        assert!(fuse86::parse("f1,f2")
            .unwrap_err()
            .contains("composes on f4"));
        for s in [
            fuse86::GDN_NORM_Q8,
            fuse86::ATTN_GATE_Q8,
            fuse86::Q8_BGEMM_LDC,
            fuse86::F5_FUSED,
            fuse86::F5_RECUR,
            fuse86::Q8W8_SFX,
            fuse86::F10_XG,
            fuse86::F10_TOPK,
            fuse86::F10_ACP,
        ] {
            d.get(s).unwrap();
        }
    }

    /// FN86b census (diagnostic, `--ignored --nocapture`): launches a verify step by family at
    /// b1 w4 / b8 w3 under `--fn-fuse none` and `all`.
    #[test]
    #[ignore = "census dump"]
    fn fn86b_census_dump() {
        let d = decls();
        let off = FlashNextRules::prod().unwrap();
        let mut on = off.clone();
        on.hc.fuse86 = saber_kernels::flashnext::fuse86::ALL;
        for (b, w) in [(1u32, 4u32), (8, 3)] {
            for (name, r) in [("none", &off), ("all", &on)] {
                let cc = ChunkConsts {
                    m: b * w,
                    b,
                    w,
                    decode: true,
                    nb: 8192u32.div_ceil(16) + 1,
                    max_ctx: 8192,
                    qsa_short: true,
                    ..ChunkConsts::default()
                };
                let x = dry_run_cc_t8(
                    &d,
                    r,
                    2,
                    false,
                    cc,
                    VerifyClass::Width1,
                    MoeTrain::Fused,
                    saber_kernels::moe::RouteForm::default(),
                    true,
                    false,
                )
                .unwrap();
                let n: u32 = x.per_layer.iter().sum::<u32>() + x.post;
                println!("FN86B b{b} w{w} {name} launches {n}");
                for (k, v) in &x.by_symbol {
                    println!("FN86B b{b}w{w} {name} {k} {v}");
                }
            }
        }
    }

    /// FN86d census (diagnostic, `--ignored --nocapture`): launches by symbol under the served
    /// `--fn-fuse all` at b1 w4 / b8 w3 verify, a 125-row and a 2,048-row W8 prefill chunk, and
    /// the drafter's MTP layer pass (slab4, the W8 head) at the b1 round's rows (pass 1 over the
    /// verify's 4 rows, passes 2..3 one row) and b8 (24 / 8) -- the device tail is the worker's
    /// (`post_launches`: FN33's int2 9 + FN22's 3).
    #[test]
    #[ignore = "census dump"]
    fn fn86d_census_dump() {
        use crate::flashnext::exec::MtpX;
        let d = decls();
        let mut on = FlashNextRules::prod().unwrap();
        on.hc.fuse86 = saber_kernels::flashnext::fuse86::ALL;
        let mut forms: Vec<(String, ChunkConsts)> = [(1u32, 4u32), (8, 3)]
            .iter()
            .map(|&(b, w)| {
                (
                    format!("b{b}w{w}"),
                    ChunkConsts {
                        m: b * w,
                        b,
                        w,
                        decode: true,
                        nb: 8192u32.div_ceil(16) + 1,
                        max_ctx: 8192,
                        qsa_short: true,
                        ..ChunkConsts::default()
                    },
                )
            })
            .collect();
        for m in [125u32, 2048] {
            forms.push((
                format!("p{m}"),
                ChunkConsts {
                    m,
                    nb: m.div_ceil(16) + 1,
                    ..ChunkConsts::default()
                },
            ));
        }
        for (name, cc) in forms {
            let x = dry_run_cc_t8(
                &d,
                &on,
                2,
                false,
                cc,
                VerifyClass::Width1,
                MoeTrain::Fused,
                saber_kernels::moe::RouteForm::default(),
                true,
                false,
            )
            .unwrap();
            let n: u32 = x.per_layer.iter().sum::<u32>() + x.post;
            println!(
                "FN86D {name} launches {n} per_layer {:?} post {}",
                x.per_layer, x.post
            );
            for (k, v) in &x.by_symbol {
                println!("FN86D {name} sym {k} {v}");
            }
            for (l, s, g, t) in &x.seq {
                println!("FN86D {name} seq L{l} {s} {}x{}x{} {t}", g[0], g[1], g[2]);
            }
        }
        for (m, w) in [(4u32, 4u32), (1, 1), (24, 3), (8, 1)] {
            let (n, _, recs) = dry_run_mtp_recs(
                &d,
                &on,
                (m, true, 8192, w),
                (MtpX::Slab4, true),
                &|_| {},
                &|_| {},
                &|_| true,
            )
            .unwrap();
            println!("FN86D mtp m{m}w{w} launches {n}");
            let mut by: std::collections::BTreeMap<String, u32> = Default::default();
            for (l, s, _) in &recs {
                println!("FN86D mtp m{m}w{w} seq L{l} {s}");
                *by.entry(s.clone()).or_insert(0) += 1;
            }
            for (k, v) in by {
                println!("FN86D mtp m{m}w{w} sym {k} {v}");
            }
        }
    }

    /// FN85: the prologue-reduce arm at every served (b, w): the verify step's block-output
    /// reduces whose consumer is a combine down (94 of 96: every attention output, every MoE output
    /// but the PLE layer's input and the final mixer's) leave the launch list, each such down is
    /// its `_rp` twin (same grid / threads), nothing else moves; prefill and off = the control.
    #[test]
    fn fn85_the_prologue_reduce_folds_94_of_96_reduces_and_off_is_the_control() {
        use super::super::exec::{has_rp_twin, rp_down_symbol};
        let d = decls();
        let off = FlashNextRules::prod().unwrap();
        let mut on = off.clone();
        on.hc.ar_prologue = true;
        let n = |x: &DryRun, p: &str| -> u32 {
            x.by_symbol
                .iter()
                .filter(|(k, _)| k.starts_with(p))
                .map(|(_, v)| *v)
                .sum()
        };
        let ar = "saber_ar_oneshot_2rank_exact_bf16";
        for b in [1u32, 2, 4, 8] {
            for w in 1..=super::super::exec::WMAX {
                let a = dry_run_verify(&d, &on, b, w, 8192).unwrap();
                let c = dry_run_verify(&d, &off, b, w, 8192).unwrap();
                assert_eq!(n(&c, ar), 96, "b{b} w{w}: the control's reduces");
                assert_eq!(n(&a, ar), 2, "b{b} w{w}: the PLE input + the final mixer");
                let rp: u32 = a
                    .by_symbol
                    .iter()
                    .filter(|(k, _)| k.ends_with("_rp"))
                    .map(|(_, v)| *v)
                    .sum();
                assert_eq!(rp, 94, "b{b} w{w}");
                let la: u32 = a.per_layer.iter().sum();
                let lc: u32 = c.per_layer.iter().sum();
                assert_eq!(
                    lc - la,
                    94,
                    "b{b} w{w}: the verify drops the family's count"
                );
                assert_eq!(a.post, c.post, "b{b} w{w}");
                // every other symbol unchanged: the rp twin replaces its combine down one for one
                for (k, v) in &c.by_symbol {
                    let got = a.by_symbol.get(k).copied().unwrap_or(0)
                        + a.by_symbol.get(&rp_down_symbol(k)).copied().unwrap_or(0);
                    let want = if k == ar { *v - 94 } else { *v };
                    assert_eq!(got, want, "b{b} w{w}: {k}");
                }
            }
        }
        // prefill chunks keep the standalone reduce
        let pa = dry_run(&d, &on, 2, 512).unwrap();
        let pc = dry_run(&d, &off, 2, 512).unwrap();
        assert_eq!(pa.by_symbol, pc.by_symbol);
        assert!(has_rp_twin("fn_hc_down_combine_v2_m1w3t1d"));
        assert!(has_rp_twin("fn_hc_down_combine_v2_m1w7t1"));
        assert!(!has_rp_twin("fn_hc_down_final_v2_m1w7t1"));
        assert!(!has_rp_twin("fn_hc_down_combine_v2_m2w21t1"));
        for c in ["m1w3t1d", "m1w3t1", "m1w7t1"] {
            d.get(&rp_down_symbol(&format!("fn_hc_down_combine_v2_{c}")))
                .unwrap();
        }
    }

    /// FN49b: the HC DPP cfgs are routed at the decode bands and the control names their twins.
    #[test]
    fn the_hc_dpp_cfgs_are_routed_and_the_control_strips_them() {
        use super::super::exec::{hc_dpp_control, rr_down_symbol};
        use saber_kernels::moe::RouteForm;
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let dpp =
            dry_run_verify_train_route(&d, &r, 1, 1, MoeTrain::Legacy, RouteForm::Dpp).unwrap();
        let ctl =
            dry_run_verify_train_route(&d, &r, 1, 1, MoeTrain::Legacy, RouteForm::Shipped).unwrap();
        assert!(dpp.by_symbol.keys().any(|k| k.ends_with("_v2_m1w3t1d")));
        assert!(dpp.by_symbol.keys().any(|k| k.ends_with("_v2_s1d")));
        assert!(ctl
            .by_symbol
            .keys()
            .all(|k| !k.ends_with("t1d") && !k.ends_with("s1d")));
        assert!(ctl
            .by_symbol
            .keys()
            .any(|k| k == "fn_hc_down_combine_v2_m1w3t1"));
        assert_eq!(hc_dpp_control("fn_hc_up_q8_v2_s1d"), "fn_hc_up_q8_v2_s1");
        assert_eq!(
            hc_dpp_control("fn_hc_down_final_v2_m1w7t1d"),
            "fn_hc_down_final_v2_m1w7t1"
        );
        assert_eq!(
            hc_dpp_control("fn_hc_down_final_v2_m1w7t1"),
            "fn_hc_down_final_v2_m1w7t1"
        );
        assert_eq!(hc_dpp_control("moe_route_x_lds_dpp"), "moe_route_x_lds_dpp");
        assert_eq!(
            rr_down_symbol("fn_hc_down_combine_v2_m1w3t1d").unwrap(),
            "fn_hc_down_combinerr_v2_m1w3t1"
        );
        // b8 (the 64 band): m1w7t1 stays, the up is s1d
        let b8 =
            dry_run_verify_train_route(&d, &r, 8, 1, MoeTrain::Legacy, RouteForm::Dpp).unwrap();
        assert!(b8.by_symbol.keys().any(|k| k.ends_with("_v2_m1w7t1")));
        assert!(b8.by_symbol.keys().any(|k| k.ends_with("_v2_s1d")));
    }

    /// FN49b: the DPP route is routed in EVERY train at T 1..16 (and the front halves are `_dpp`);
    /// `--fn-route-control` names exactly main's pre-FN49b symbols; launch counts never move.
    #[test]
    fn the_dpp_route_is_the_default_and_the_route_control_is_main_before_fn49b() {
        use saber_kernels::moe::{RouteForm, ROUTE_SYMBOL, ROUTE_SYMBOL_LDS, ROUTE_SYMBOL_LDS_DPP};
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for (b, w) in [
            (1u32, 1u32),
            (1, 4),
            (2, 3),
            (4, 3),
            (8, 1),
            (8, 2),
            (8, 3),
            (4, 5),
        ] {
            let m = b * w;
            for train in [MoeTrain::Legacy, MoeTrain::Fused] {
                let dpp = dry_run_verify_train_route(&d, &r, b, w, train, RouteForm::Dpp).unwrap();
                let ctl =
                    dry_run_verify_train_route(&d, &r, b, w, train, RouteForm::Shipped).unwrap();
                assert_eq!(dpp.per_layer, ctl.per_layer, "b{b} w{w} {train}");
                let front = train == MoeTrain::Fused && m <= 8;
                let (want, was) = if front {
                    let f = saber_kernels::moe::FrontTrain {
                        split: saber_kernels::moe::router_split(b),
                        waves: 8,
                        unrolled: true,
                        route: RouteForm::Dpp,
                    };
                    let c = saber_kernels::moe::FrontTrain {
                        route: RouteForm::Shipped,
                        ..f
                    };
                    (f.symbol(), c.symbol())
                } else if m <= 16 {
                    let was = if train == MoeTrain::Fused {
                        ROUTE_SYMBOL_LDS
                    } else {
                        ROUTE_SYMBOL
                    };
                    (ROUTE_SYMBOL_LDS_DPP.to_string(), was.to_string())
                } else {
                    (ROUTE_SYMBOL.to_string(), ROUTE_SYMBOL.to_string())
                };
                assert_eq!(dpp.by_symbol.get(&want), Some(&48), "b{b} w{w} {train}");
                assert_eq!(ctl.by_symbol.get(&was), Some(&48), "b{b} w{w} {train}");
                assert!(
                    ctl.by_symbol.keys().all(|k| !k.ends_with("_dpp")),
                    "the control names no DPP symbol"
                );
            }
        }
    }

    /// FN30: the fused MoE train at every served (b, w) -- 3 launches a target block, the back half
    /// at the route's CLASS split pair (FN24: a b1 verify keeps T = 1's ks4 / ks2), route v2 at
    /// T 2..16, the gate GEMV gone; `legacy` = the seven (the control), the drafter's pass keeps
    /// its seven either way; the sync region fits every step's words.
    #[test]
    fn the_fused_moe_train_is_emitted_at_every_decode_cell_and_legacy_is_the_control() {
        use saber_kernels::moe::{BackTrain, FrontTrain, ROUTE_SYMBOL, ROUTE_SYMBOL_LDS_DPP};
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for b in [1u32, 2, 4, 8] {
            for w in 1..=super::super::exec::WMAX {
                let f =
                    dry_run_verify_train(&d, &r, b, w, 8192, VerifyClass::Width1, MoeTrain::Fused)
                        .unwrap();
                let l =
                    dry_run_verify_train(&d, &r, b, w, 8192, VerifyClass::Width1, MoeTrain::Legacy)
                        .unwrap();
                let fl: u32 = f.per_layer.iter().sum();
                let ll: u32 = l.per_layer.iter().sum();
                let m = b * w;
                let front = m <= saber_kernels::moe::FrontTrain::MAX_TOKENS;
                assert_eq!(ll - fl, 48 * if front { 5 } else { 4 }, "b{b} w{w}");
                let back = if b == 1 {
                    "moe_back_gu4_dn2_p1"
                } else {
                    "moe_back_gu1_dn1_p1"
                };
                assert_eq!(
                    f.by_symbol.get(back),
                    Some(&48),
                    "b{b} w{w}: {:?}",
                    f.by_symbol
                );
                let sk = |x: &DryRun| x.by_symbol.get("fn_bgemm_sk_m16").copied().unwrap_or(0);
                // the gate GEMV rides the back half; at T <= 8 the router GEMM rides the front
                assert_eq!(sk(&l) - sk(&f), 48 * if front { 2 } else { 1 }, "b{b} w{w}");
                // FN49b: the seven-launch train routes FN49a's DPP LDS route at T 1..16 too
                let legacy_route = if m <= 16 {
                    ROUTE_SYMBOL_LDS_DPP
                } else {
                    ROUTE_SYMBOL
                };
                assert_eq!(l.by_symbol.get(legacy_route), Some(&48), "b{b} w{w}");
                if front {
                    // the router split is the CLASS rows' (b): FN24's b1 verify keeps split 4
                    let fs = FrontTrain {
                        split: saber_kernels::moe::router_split(b),
                        waves: FrontTrain::WAVES,
                        unrolled: true,
                        route: saber_kernels::moe::RouteForm::Dpp,
                    }
                    .symbol();
                    assert!(fs.ends_with("_dpp"), "{fs}");
                    assert_eq!(
                        f.by_symbol.get(&fs),
                        Some(&48),
                        "b{b} w{w}: {:?}",
                        f.by_symbol
                    );
                } else {
                    let route = if m <= 16 {
                        ROUTE_SYMBOL_LDS_DPP
                    } else {
                        ROUTE_SYMBOL
                    };
                    assert_eq!(f.by_symbol.get(route), Some(&48), "b{b} w{w}");
                }
                assert!(l.by_symbol.keys().all(|k| !k.starts_with("moe_back")));
                let tc = super::super::exec::moe_tiles_cap(u64::from(m), 1, 256);
                assert!(
                    BackTrain::sync_words(tc as u32) * 4
                        <= super::super::exec::moe_sync_slot_bytes(256)
                );
            }
        }
        // prefill chunks keep the seven under the default
        let p = dry_run(&d, &r, 2, 64).unwrap();
        assert!(p.by_symbol.keys().all(|k| !k.starts_with("moe_back")));
        // the drafter's pass keeps its seven (bf16 experts): the_mtp_pass_is_the_declared_train_..
        // declares it with `MoeTrain::Legacy` under the default
        let (_, _) = dry_run_mtp(&d, &r, 8, true, 8192).unwrap();
        assert_eq!(super::super::exec::moe_sync_bytes(256), 64 * 3840);
        assert_eq!(super::super::exec::moe_sync_bytes(512), 64 * 6912);
    }

    #[test]
    fn every_layer_emits_the_graphs_launch_count_at_both_tps_and_every_band() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        for tp in [1, 2] {
            for m in [1u32, 12, 64, 65, 128, 295, 2048] {
                let dr = dry_run(&d, &r, tp, m).unwrap();
                for (l, &n) in dr.per_layer.iter().enumerate() {
                    assert_eq!(n, declared(&r, l as u32, tp, m), "tp {tp} m {m} layer {l}");
                }
                // final mixer (2) + head: two launches of 124,160 rows at TP=1, one half at TP=2
                assert_eq!(dr.post, 2 + 2 / tp, "tp {tp} m {m}");
                let total: u32 = dr.per_layer.iter().sum::<u32>() + dr.post;
                let graph = layer_graph::step_graph(
                    &r,
                    &(0..LAYERS).map(kind).collect::<Vec<_>>(),
                    PLE_LAYER,
                    Form::Prefill,
                    if tp == 1 { Tp::One } else { Tp::Two },
                    m,
                );
                // the graph's `pre` is decode's embedding launch; prefill gathers the rows on the host
                assert_eq!(total + 1, graph.launches(), "tp {tp} m {m}");
            }
        }
    }

    #[test]
    fn a_decode_step_emits_the_graphs_decode_form_at_every_batch() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let pre = dry_run(&d, &r, 2, super::super::run::MAX_ROWS).unwrap();
        for b in [1u32, 2, 4, 8] {
            let dr = dry_run_decode(&d, &r, b, 2048).unwrap();
            let form = Form::Decode {
                batch: b,
                attn: saber_kernels::fn6::AttnForm::DecodePair,
            };
            for (l, &n) in dr.per_layer.iter().enumerate() {
                assert_eq!(
                    n,
                    declared_form(&r, l as u32, 2, b, form),
                    "b {b} layer {l}"
                );
            }
            // GDN 17 / PLE 21 / QSA 21 at TP=2 (b1: the fused update; b>1: the conv+recurrent
            // pair) under the default (legacy) MoE train; FN30's fused arm: 5 fewer a layer at
            // T <= 8 (`the_fused_moe_train_is_emitted_..`)
            let gdn = if b == 1 { 17 } else { 18 };
            assert_eq!(
                (dr.per_layer[0], dr.per_layer[1], dr.per_layer[3]),
                (gdn, gdn + 4, 21),
                "b {b}"
            );
            let total: u32 = dr.per_layer.iter().sum::<u32>() + dr.post;
            let graph = layer_graph::step_graph(
                &r,
                &(0..LAYERS).map(kind).collect::<Vec<_>>(),
                PLE_LAYER,
                form,
                Tp::Two,
                b,
            );
            // the embedding rows are a host gather (the graph's `embed` launch is not issued)
            assert_eq!(total + 1, graph.launches(), "b {b}");
            if b == 1 {
                assert_eq!(total, 871, "a b1 decode step's launches");
            }
            assert!(
                dr.sizes.total() <= pre.sizes.total(),
                "the decode step fits the prefill arena"
            );
        }
    }

    #[test]
    fn the_launch_counts_per_kind_are_fn9bs() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let t1 = dry_run(&d, &r, 1, 64).unwrap();
        assert_eq!(
            (
                t1.per_layer[0],
                t1.per_layer[1],
                t1.per_layer[2],
                t1.per_layer[3]
            ),
            (17, 21, 17, 18)
        );
        let t2 = dry_run(&d, &r, 2, 64).unwrap();
        assert_eq!(
            (t2.per_layer[2], t2.per_layer[1], t2.per_layer[3]),
            (19, 23, 20),
            "FN9b: 19/23/20"
        );
        assert_eq!(
            t2.by_symbol.get("fn_gdn_chunk_scan_xg_f16s_k128_v128_c64"),
            Some(&36)
        );
        assert_eq!(
            t2.by_symbol.get(saber_kernels::fn6::PREFILL_SYMBOL),
            Some(&12)
        );
    }

    #[test]
    fn the_arena_is_sized_by_the_widest_chunk_and_grows_with_m() {
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let a = dry_run(&d, &r, 2, 295).unwrap().sizes;
        let b = dry_run(&d, &r, 2, 2048).unwrap().sizes;
        assert!(
            b.scratch >= a.scratch && b.zero >= a.zero && b.carry >= a.carry && b.ones >= a.ones
        );
        assert!(
            b.total() < 2 << 30,
            "one rank's prefill arena at 2,048 rows: {} B",
            b.total()
        );
    }

    #[test]
    fn fn38_a_wide_chunk_plans_and_its_arena_grows_linearly() {
        // FN38: 4,096 / 8,192-row chunks plan (the 2,048 bands' symbols carried up, the reduce's
        // mailbox doubled at 8,192) and the arena is linear in the rows: 763 / 1,497 / 2,963 MiB
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let mib: Vec<u64> = [2048u32, 4096, 8192]
            .iter()
            .map(|&m| {
                dry_run_prefill_at(&d, &r, 2, m, 8192 - m)
                    .unwrap()
                    .sizes
                    .total()
                    >> 20
            })
            .collect();
        assert_eq!(mib, vec![763, 1497, 2963]);
        // the bgemm / HC tables at 8,192 rows are the 2,048 rows' cells (row-local classes): an
        // 8,192-row chunk from position 0 runs the symbols of today's SPARSE second chunk
        let a = dry_run_prefill_at(&d, &r, 2, 2048, 2048).unwrap();
        let b = dry_run_prefill_at(&d, &r, 2, 8192, 0).unwrap();
        let syms = |x: &DryRun| x.by_symbol.keys().cloned().collect::<Vec<_>>();
        assert_eq!(syms(&a), syms(&b));
        // the launches differ only by the QSA layers' score / top-k slices (256 rows a slice)
        let extra = saber_kernels::fn19::selection_launches(8192)
            - saber_kernels::fn19::selection_launches(2048);
        for (l, (x, y)) in a.per_layer.iter().zip(&b.per_layer).enumerate() {
            let q = if kind(l as u32) == LayerKind::Qsa {
                extra
            } else {
                0
            };
            assert_eq!(x + q, *y, "layer {l}");
        }
    }

    /// FN68d: `--fn-kv-mtp-dtype bf16` under fp8 pages moves the MTP layer (48) ALONE -- its
    /// append and attention run the bf16 symbols with the bf16 kernargs (null descales), the same
    /// launches and arena as a bf16 boot's pass, and under the tier its K/V readers bind the
    /// TWIN's table (never the main tier's tier-layer-12 table); the one-dtype control keeps the
    /// fp8 twins and the main table; a target layer's dtype never reads `kv8_mtp`.
    #[test]
    fn fn68d_the_mtp_layer_runs_bf16_pages_from_its_own_table_beside_fp8_targets() {
        let (d, r) = (decls(), FlashNextRules::prod().unwrap());
        let (t, ls, mt, ks) = (0x7000_0000u64, 0x1_0000u64, 0x9000_0000u64, 0x6000_0000u64);
        let u64s = |a: &[u8]| -> Vec<u64> {
            a.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        for (m, decode, ctx, w) in [
            (1u32, true, 8192u32, 1u32),
            (4, true, 8192, 4),
            (8, true, 65_536, 1),
            (512, false, 8192, 1),
        ] {
            let what = format!("m{m} decode {decode} ctx {ctx} w{w}");
            let run = |kv8: bool, kv8_mtp: bool, tier: bool, twin: bool| {
                dry_run_mtp_kv(
                    &d,
                    &r,
                    (m, decode, ctx, w),
                    (crate::flashnext::exec::MtpX::Fp8, false),
                    &|e| {
                        e.kv8 = kv8;
                        e.kv8_mtp = kv8_mtp;
                        e.kv_desc = 0x5000_0000;
                        if tier {
                            e.kv_table = t;
                            e.kv_ls = ls;
                            e.kv_mtp_table = if twin { mt } else { 0 };
                        }
                    },
                    &|cc| {
                        if tier {
                            cc.kslot = ks;
                        }
                    },
                )
                .unwrap()
            };
            let bf = run(false, false, false, false);
            let split = run(true, false, false, false);
            let one = run(true, true, false, false);
            assert!(!bf.2.is_empty() && bf.2.iter().all(|x| x.0 == 48), "{what}");
            // the split pass IS the bf16 pass: symbols, kernargs, launches, arena
            assert_eq!(split.2, bf.2, "{what}");
            assert_eq!((split.0, split.1), (bf.0, bf.1), "{what}");
            // the control keeps FN68b's fp8 twins at layer 48
            assert_eq!((one.0, one.1), (bf.0, bf.1), "{what}");
            assert_eq!(one.2.len(), bf.2.len(), "{what}");
            for ((_, s1, _), (_, sb, _)) in one.2.iter().zip(&bf.2) {
                assert_eq!(
                    *s1,
                    sb.replace("_bf16kv", "_fp8kv")
                        .replace("fn_kv_append_ring_h256_bf16", "fn_kv_append_ring_h256_fp8"),
                    "{what}"
                );
                if sb.contains("bf16") {
                    assert_ne!(
                        s1, sb,
                        "{what}: every bf16-KV launch swaps under the control"
                    );
                }
            }
            assert!(one.2.iter().any(|x| x.1.contains("fp8")), "{what}");
            // the tier: the twin's table, never the main tier's
            let tsplit = run(true, false, true, true);
            let tone = run(true, true, true, false);
            let mut nread = 0;
            for (_, sym, a) in &tsplit.2 {
                let v = u64s(a);
                assert!(
                    !v.iter().any(|&x| (t..t + 13 * ls).contains(&x)),
                    "{what} {sym}: a main-tier table bound at the MTP layer"
                );
                if !sym.contains("kv_append") {
                    nread += 1;
                    assert!(
                        v.iter().any(|&x| (mt..mt + ls).contains(&x)),
                        "{what} {sym}: the twin's table is not bound"
                    );
                }
            }
            assert!(nread >= 1, "{what}");
            for (_, sym, a) in &tone.2 {
                let v = u64s(a);
                assert!(!v.iter().any(|&x| (mt..mt + ls).contains(&x)), "{what}");
                if !sym.contains("kv_append") {
                    assert!(
                        v.iter().any(|&x| (t + 12 * ls..t + 13 * ls).contains(&x)),
                        "{what} {sym}: the control reads tier layer 12 of the main table"
                    );
                }
            }
        }
        // a target layer's dtype is `kv8` whatever `kv8_mtp` says
        let mut e = Emitter::new(&d, &r, Dims::new(2, 0), unbounded());
        e.kv8 = true;
        e.kv8_mtp = false;
        for l in [3, 7, 47] {
            e.layer = l;
            assert!(e.kv8_here(), "layer {l}");
        }
        e.layer = 48;
        assert!(!e.kv8_here());
        assert_eq!(e.kv_eb(), 2);
    }

    /// FN78: the decode-form band law -- every decode / verify step of the served buckets (b8 at
    /// widths 1..5 included: 40 rows) reduces on the EXACT kernel, a 64-row PREFILL chunk keeps the
    /// payload band (wht6), and `--fn-decode-ar band` (the control) puts b8 w >= 4 back on wht6.
    /// FN150: at `Dims::experts_nvfp4` the seven-launch train emits FN148's form (b) cells with the
    /// manifest's kernarg layout (+ `gsplit`) at the measured splits -- T 1: gate_up ks 4 / down
    /// ks 2, T 2..512: ks 1, tm 1 -- the rest of the block unchanged; a > 512-row chunk (the
    /// pf_tm4 prefill tile) and the fused back half refuse by name.
    #[test]
    fn fn150_the_nvfp4_dims_emit_fn148_form_b_cells() {
        use super::super::exec::{HcW, LayerW, MixOutBufs, MixerW};
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let lw = LayerW {
            l: 0,
            ahc: HcW::default(),
            mhc: HcW::default(),
            mixer: MixerW::Gdn {
                win: 0,
                conv: 0,
                alog: 0,
                dtb: 0,
                nw: 0,
                wout: 0,
            },
            router: 1 << 30,
            sg: 1 << 30,
            ple: None,
            xgu: 0,
            xdn: 0,
            mtp: false,
            mtp_fp8: false,
            mtp_x4: false,
            mtp_xtp2: false,
            t8: None,
        };
        let mx = MixOutBufs {
            hid: 1 << 31,
            injo: 1 << 31,
            bi: 1 << 31,
            q8: 1 << 31,
            scale: 1 << 31,
        };
        // FN152: `nv` = FN150's plain route (the control); `emit_p1` = the default P1 route
        let emit_dm = |m: u64, dm: Dims| -> anyhow::Result<Vec<String>> {
            let mut e = Emitter::new(&d, &r, dm, unbounded());
            let rt = e.moe_route(&lw, &mx, m, false)?;
            e.moe_experts(&rt, &mx, m, 1 << 32, 1 << 33, false)?;
            Ok(e.recs.iter().map(|x| x.sym.clone()).collect())
        };
        let emit = |m: u64, nv: bool| -> anyhow::Result<Vec<String>> {
            let dm = Dims::new(2, 0);
            let dm = if nv { dm.experts_nvfp4_plain() } else { dm };
            let mut e = Emitter::new(&d, &r, dm, unbounded());
            let rt = e.moe_route(&lw, &mx, m, false)?;
            e.moe_experts(&rt, &mx, m, 1 << 32, 1 << 33, false)?;
            Ok(e.recs.iter().map(|x| x.sym.clone()).collect())
        };
        for (m, gu, dn) in [(1, 4, 2), (4, 1, 1), (286, 1, 1), (512, 1, 1)] {
            let nv = emit(m, true).unwrap();
            let mx4 = emit(m, false).unwrap();
            assert_eq!(nv.len(), mx4.len(), "m {m}: the launch law");
            let g = format!("fn148_w4a8b_bk128_ks{gu}_tm1_g1");
            let dd = format!("fn148_w4a8b_bk128_ks{dn}_tm1_g0");
            assert!(nv.contains(&g) && nv.contains(&dd), "m {m}: {nv:?}");
            let differ: Vec<_> = nv.iter().zip(&mx4).filter(|(a, b)| a != b).collect();
            assert_eq!(
                differ.len(),
                2,
                "m {m}: only the two grouped GEMMs move: {differ:?}"
            );
            assert!(differ
                .iter()
                .all(|(_, b)| b.starts_with("moe_grouped_w4a8_")));
            // FN152: the default route names FN151's twins of the same two cells, nothing else moves
            let p1 = emit_dm(m, Dims::new(2, 0).experts_nvfp4()).unwrap();
            let want: Vec<String> = nv
                .iter()
                .map(|s| s.replacen("fn148_w4a8b_", "fn151_w4a8b_", 1))
                .collect();
            assert_eq!(p1, want, "m {m}: the P1 route");
        }
        let e = emit(2048, true).unwrap_err().to_string();
        assert!(e.contains("prefill"), "{e}");
        assert!(emit(2048, false).is_ok());
    }

    #[test]
    fn fn78_decode_forms_reduce_exactly_and_band_is_the_control() {
        const WHT6: &str = "saber_ar_oneshot_2rank_wht6_bf16";
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let wht6 = |x: &DryRun| x.by_symbol.get(WHT6).copied().unwrap_or(0);
        for b in [1u32, 2, 4, 8] {
            for w in 1..=super::super::exec::WMAX {
                let x = dry_run_verify(&d, &r, b, w, 8192).unwrap();
                assert_eq!(wht6(&x), 0, "b{b} w{w}: a decode form reduced on wht6");
                assert!(decode_ar_exact(b * w), "b{b} w{w}");
            }
        }
        assert!(
            wht6(&dry_run(&d, &r, 2, 64).unwrap()) > 0,
            "a 64-row prefill chunk is wht6"
        );
        let h = super::super::exec::H;
        let mut pair = fake_pair().unwrap();
        assert!(pair.decode_exact, "the default law");
        let mut e = Emitter::new(&d, &r, Dims::new(2, 0), unbounded());
        e.ar_decode = true;
        let mut ems = vec![e];
        for rows in [24u64, 32, 40, 64] {
            assert!(!pair.lossy(&ems, rows * h), "decode {rows} rows");
        }
        assert!(
            pair.lossy(&ems, 65 * h),
            "past the producer window: the payload band"
        );
        ems[0].ar_decode = false;
        assert!(pair.lossy(&ems, 32 * h), "a prefill form keeps the band");
        assert!(!pair.lossy(&ems, 24 * h));
        ems[0].ar_decode = true;
        pair.decode_exact = false;
        assert!(pair.lossy(&ems, 32 * h), "the control: b8 w4 on wht6");
        assert!(
            !pair.lossy(&ems, 24 * h),
            "the control: b8 w3 exact (today's served cell)"
        );
    }

    /// FN140: FN119's standalone embedding reduces (pass 1, the next verify) ride FN78's
    /// decode-form exact law: a 32-row (b8 x w4) and a 64-row reduce emit the EXACT symbol (the
    /// which-path counter `calls_decode_exact` names the law) whatever form the emitter last
    /// emitted, 65 rows and `--fn-decode-ar band` past the band refuse by name, 24 rows stays
    /// exact by the band alone; control: the pre-FN140 PAYLOAD guard refuses 32 rows (FN136 h3).
    #[test]
    fn fn140_fn119_embedding_reduces_ride_the_decode_form_exact_law() {
        const EXACT: &str = "saber_ar_oneshot_2rank_exact_bf16";
        const WHT6: &str = "saber_ar_oneshot_2rank_wht6_bf16";
        let h = super::super::exec::H;
        let d = decls();
        let r = FlashNextRules::prod().unwrap();
        let what = "FN119: the pass-1 embeddings must ride the EXACT reduce";
        let syms =
            |e: &Emitter<'_>| -> Vec<String> { e.recs.iter().map(|x| x.sym.clone()).collect() };
        for (rows, by_law) in [
            (1u64, false),
            (24, false),
            (32, true),
            (40, true),
            (64, true),
        ] {
            for leftover in [false, true] {
                let mut pair = fake_pair().unwrap();
                let mut e = Emitter::new(&d, &r, Dims::new(2, 0), unbounded());
                e.ar_decode = leftover;
                pair.reduce_decode_form(&mut e, 1 << 31, rows, what)
                    .unwrap();
                assert_eq!(syms(&e), vec![EXACT.to_owned()], "{rows} rows");
                assert_eq!(pair.calls_exact, 1, "{rows} rows");
                assert_eq!(pair.calls_wht6, 0, "{rows} rows");
                assert_eq!(
                    pair.calls_decode_exact,
                    u64::from(by_law),
                    "{rows} rows: which path"
                );
                assert_eq!(
                    e.ar_decode, leftover,
                    "{rows} rows: the emitter's form restored"
                );
            }
        }
        // the guard refuses past the law's window (65 rows would be wht6 by the payload band)
        let mut pair = fake_pair().unwrap();
        let mut e = Emitter::new(&d, &r, Dims::new(2, 0), unbounded());
        let err = pair
            .reduce_decode_form(&mut e, 1 << 31, 65, what)
            .unwrap_err()
            .to_string();
        assert!(err.contains("past the decode-form exact law"), "{err}");
        assert!(e.recs.is_empty() && pair.calls_exact + pair.calls_wht6 == 0);
        assert!(
            pair.decode_form_exact_law(0, what).is_err(),
            "an empty reduce"
        );
        // `--fn-decode-ar band` (FN78's control) keeps 32 rows on the band -> refused, 24 rows exact
        pair.decode_exact = false;
        assert!(pair.reduce_decode_form(&mut e, 1 << 31, 32, what).is_err());
        assert!(e.recs.is_empty());
        pair.reduce_decode_form(&mut e, 1 << 31, 24, what).unwrap();
        assert_eq!(syms(&e), vec![EXACT.to_owned()]);
        // control: the pre-FN140 payload guard (`rows x H x 2 < quant_min_bytes`) refuses 32 rows
        // and passes 24 -- the FATAL of FN136 h3 -- and a bare non-decode reduce of 32 rows is wht6
        let q = fake_pair().unwrap().rules.quant_min_bytes;
        assert!(
            32 * h * 2 >= q,
            "the control: b8 x w4 fails the payload guard"
        );
        assert!(24 * h * 2 < q);
        let mut pair = fake_pair().unwrap();
        let mut e = Emitter::new(&d, &r, Dims::new(2, 0), unbounded());
        e.ar_decode = false;
        pair.reduce(std::slice::from_mut(&mut e), &[1 << 31], 32 * h, false)
            .unwrap();
        assert_eq!(
            syms(&e),
            vec![WHT6.to_owned()],
            "the control: the band's pick"
        );
    }

    /// FN114c: `--fn-shared-expert split` (the emitter's `shared_split` = the half tables) moves
    /// ONLY the target's fused back half: BOTH ranks launch `moe_back_ss_*` with the layer's half
    /// table as the last kernel argument (decode b1 / b8 and verify b1 w4 / b8 w3), the declared
    /// launch counts and the arena equal the control's, and a PREFILL chunk (the seven-launch
    /// train) emits the control's records byte for byte (rank 0's whole shared expert).
    #[test]
    fn fn114c_the_split_moves_only_the_fused_back_half() {
        let (d, r) = (decls(), FlashNextRules::prod().unwrap());
        const SHX: u64 = 0x0B00_0000;
        let u64s = |a: &[u8]| -> Vec<u64> {
            a.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        for (m, w, decode) in [
            (1u32, 1u32, true),
            (4, 4, true),
            (8, 1, true),
            (24, 3, true),
            (64, 1, false),
        ] {
            let run = |ss: u64| {
                let cc = ChunkConsts {
                    m,
                    b: m / w,
                    w,
                    decode,
                    nb: 8192u32.div_ceil(16) + 1,
                    max_ctx: 8192,
                    ..ChunkConsts::default()
                };
                let mut ems: Vec<Emitter<'_>> = (0..2)
                    .map(|rk| {
                        let mut e = Emitter::new(&d, &r, Dims::new(2, rk), unbounded());
                        e.moe_sync = DRY_MOE_SYNC;
                        e.moe_train = MoeTrain::Fused;
                        e.shared_split = ss;
                        e
                    })
                    .collect();
                let lws = vec![fake_layer(3); 2];
                let mut pair = fake_pair().unwrap();
                let mut ar: Option<&mut dyn Reduce> = Some(&mut pair);
                let mut hook = Resident(&lws);
                emit_layer(
                    &mut ems,
                    &lws,
                    &[LayerState::default(); 2],
                    &[LayerIn::Embed(1 << 31), LayerIn::Embed(1 << 31)],
                    &[0, 0],
                    &[cc; 2],
                    &mut ar,
                    &mut hook,
                )
                .unwrap();
                ems.iter()
                    .map(|e| {
                        (
                            e.forward_launches(),
                            e.arena.scratch.high,
                            e.recs
                                .iter()
                                .map(|x| (x.sym.clone(), x.args.clone()))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let (ctl, arm) = (run(0), run(SHX));
            let what = format!("m{m} w{w} decode {decode}");
            for rk in 0..2 {
                assert_eq!(ctl[rk].0, arm[rk].0, "{what} rank {rk}: launches");
                assert_eq!(ctl[rk].1, arm[rk].1, "{what} rank {rk}: arena");
                let back: Vec<_> = arm[rk]
                    .2
                    .iter()
                    .filter(|x| x.0.starts_with("moe_back"))
                    .collect();
                if !decode {
                    assert!(
                        back.is_empty(),
                        "{what}: a prefill chunk is the seven-launch train"
                    );
                    assert_eq!(ctl[rk].2, arm[rk].2, "{what} rank {rk}: prefill moved");
                    continue;
                }
                assert_eq!(back.len(), 1, "{what} rank {rk}");
                assert!(
                    back[0].0.starts_with("moe_back_ss_gu"),
                    "{what}: {}",
                    back[0].0
                );
                let v = u64s(&back[0].1);
                assert_eq!(
                    *v.last().unwrap(),
                    SHX + 3 * super::super::exec::SHX_TABLE_BYTES,
                    "{what} rank {rk}: layer 3's half table is the last argument"
                );
                let cb: Vec<_> = ctl[rk]
                    .2
                    .iter()
                    .filter(|x| x.0.starts_with("moe_back"))
                    .collect();
                assert_eq!(
                    cb[0].0.replace("moe_back_", "moe_back_ss_"),
                    back[0].0,
                    "{what}: the split pair is the control's"
                );
                // every record but the back half and the route (rank 1 now lays out shared rows)
                // is the control's
                let other = |x: &&(String, Vec<u8>)| {
                    !x.0.starts_with("moe_back")
                        && !x.0.starts_with("moe_front")
                        && !x.0.starts_with("moe_route")
                        && !x.0.starts_with("fn111_moe_route")
                };
                let (a, c): (Vec<_>, Vec<_>) = (
                    arm[rk].2.iter().filter(other).collect(),
                    ctl[rk].2.iter().filter(other).collect(),
                );
                assert_eq!(
                    a, c,
                    "{what} rank {rk}: a record outside the MoE route / back moved"
                );
            }
        }
    }

    /// FN137 (`--fn-moe-ar-push on`): at every TARGET decode / verify form of a tp2 boot (b1, b1
    /// w4, b8, b8 w3, b8 w5) the MoE-out reduce rides the fused back half -- BOTH ranks launch
    /// `fn137_moe_back_h2_far_*` at the control's split pair with the control's back-half
    /// arguments + the channel blob whose SUM pointer is the control reduce's output (the layer's
    /// carry), the standalone reduce is gone (one launch fewer), every other record is the
    /// control's byte for byte; a PREFILL chunk and an EP boot (the whole-expert back half) keep
    /// the control's records exactly (the EP decode form counted as a stand-down).
    #[test]
    fn fn137_the_moe_out_reduce_rides_the_tp2_back_half() {
        use super::super::dev::{far_scratch_off, AR_MB, FAR_CTR0, FAR_DIAG0, FAR_FLAG0};
        const AR: &str = "saber_ar_oneshot_2rank_exact_bf16";
        let (d, r) = (decls(), FlashNextRules::prod().unwrap());
        let u64s = |a: &[u8]| -> Vec<u64> {
            a.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        type Recs = Vec<(String, Vec<u8>)>;
        for tp2x in [true, false] {
            for (m, w, decode) in [
                (1u32, 1u32, true),
                (4, 4, true),
                (8, 1, true),
                (24, 3, true),
                (40, 5, true),
                (64, 1, false),
            ] {
                let run = |far: bool| -> (Vec<(u32, Recs)>, (u64, u64)) {
                    let cc = ChunkConsts {
                        m,
                        b: m / w,
                        w,
                        decode,
                        nb: 8192u32.div_ceil(16) + 1,
                        max_ctx: 8192,
                        ..ChunkConsts::default()
                    };
                    let mut ems: Vec<Emitter<'_>> = (0..2)
                        .map(|rk| {
                            let mut e = Emitter::new(&d, &r, Dims::new(2, rk), unbounded());
                            if tp2x {
                                e.dims = e.dims.experts_tp2();
                            }
                            e.moe_sync = DRY_MOE_SYNC;
                            e.moe_train = MoeTrain::Fused;
                            e
                        })
                        .collect();
                    let lws = vec![fake_layer(3); 2];
                    let mut pair = fake_pair().unwrap();
                    pair.far = far;
                    {
                        let mut ar: Option<&mut dyn Reduce> = Some(&mut pair);
                        let mut hook = Resident(&lws);
                        emit_layer(
                            &mut ems,
                            &lws,
                            &[LayerState::default(); 2],
                            &[LayerIn::Embed(1 << 31), LayerIn::Embed(1 << 31)],
                            &[0, 0],
                            &[cc; 2],
                            &mut ar,
                            &mut hook,
                        )
                        .unwrap();
                    }
                    (
                        ems.iter()
                            .map(|e| {
                                (
                                    e.forward_launches(),
                                    e.recs
                                        .iter()
                                        .map(|x| (x.sym.clone(), x.args.clone()))
                                        .collect::<Recs>(),
                                )
                            })
                            .collect(),
                        (pair.calls_far, pair.far_stood_down),
                    )
                };
                let ((ctl, cn), (arm, an)) = (run(false), run(true));
                let what = format!("tp2 {tp2x} m{m} w{w} decode {decode}");
                assert_eq!(cn, (0, 0), "{what}: the control counts nothing");
                if !decode || !tp2x {
                    assert_eq!(ctl, arm, "{what}: the control's records");
                    assert_eq!(an.0, 0, "{what}");
                    assert_eq!(an.1, u64::from(decode), "{what}: stand-downs");
                    continue;
                }
                assert_eq!(an, (1, 0), "{what}: one fused reduce a layer");
                for rk in 0..2 {
                    assert_eq!(arm[rk].0 + 1, ctl[rk].0, "{what} rank {rk}: launches");
                    let ci = ctl[rk]
                        .1
                        .iter()
                        .position(|x| x.0.starts_with("moe_back_h2_gu"))
                        .unwrap();
                    let (cb, ar) = (&ctl[rk].1[ci], &ctl[rk].1[ci + 1]);
                    assert_eq!(ar.0, AR, "{what}: the reduce follows the back half");
                    let ab = &arm[rk].1[ci];
                    assert_eq!(
                        ab.0,
                        format!("fn137_{}", cb.0.replace("moe_back_h2_", "moe_back_h2_far_")),
                        "{what}: the control's split pair"
                    );
                    assert_eq!(
                        &ab.1[..cb.1.len()],
                        &cb.1[..],
                        "{what}: the back half's args"
                    );
                    let b0 = cb.1.len().next_multiple_of(8);
                    assert_eq!(
                        ab.1.len(),
                        b0 + 72,
                        "{what}: the blob after the 8-aligned tail"
                    );
                    let blob = u64s(&ab.1[b0..b0 + 64]);
                    let red = u64s(&ar.1)[3];
                    assert_eq!(blob[5], red, "{what}: the SUM is the reduce's output");
                    assert_eq!(blob[0], far_scratch_off(AR_MB), "{what}: the peer window");
                    assert_eq!(blob[1], far_scratch_off(AR_MB));
                    assert_eq!(blob[2], FAR_FLAG0 * 4);
                    assert_eq!(blob[3], FAR_FLAG0 * 4);
                    assert_eq!(
                        (blob[4], blob[6], blob[7]),
                        (FAR_CTR0 * 4, FAR_DIAG0 * 4, 0)
                    );
                    let mut want = ctl[rk].1.clone();
                    want.remove(ci + 1);
                    want[ci] = ab.clone();
                    assert_eq!(
                        arm[rk].1, want,
                        "{what} rank {rk}: only the back half + reduce moved"
                    );
                }
            }
        }
    }

    /// FN164 (`--fn-mixer-ar-push on|b1`): at every TARGET decode / verify form of a TP = 2 W8
    /// trunk (b1, b1 w4, b8, b8 w3, b8 w5; a GDN and a QSA layer, FN141's ring on and off) the
    /// mixer-out reduce rides the out / o projection -- BOTH ranks launch `fn164_far_<parent>` at
    /// the control's cell with the control's arguments + the channel blob whose SUM pointer is the
    /// control reduce's output, the standalone reduce is gone (one launch fewer), every other
    /// record is the control's byte for byte; `b1` fuses the one-sequence forms only; a PREFILL
    /// chunk keeps the control's records; the m16 cell (FN111's t1 off) stands down, counted.
    #[test]
    fn fn164_the_mixer_out_reduce_rides_the_out_projection() {
        use super::super::dev::{
            mix_scratch_off, MixPush, AR_MB, MIX_CTR0, MIX_DIAG0, MIX_FLAG0, MIX_PARENTS,
        };
        use saber_kernels::flashnext::{k141, tiles111};
        const AR: &str = "saber_ar_oneshot_2rank_exact_bf16";
        let d = decls();
        let u64s = |a: &[u8]| -> Vec<u64> {
            a.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        type Recs = Vec<(String, Vec<u8>)>;
        let mut fused_seen = std::collections::BTreeSet::new();
        for (t1, ring) in [(true, true), (true, false), (false, false)] {
            let mut r = FlashNextRules::prod().unwrap();
            r.hc.tiles111 = if t1 { tiles111::ALL } else { 0 };
            r.hc.k141 = if ring { k141::ALL } else { 0 };
            for l in [2u32, 3] {
                for (m, w, decode) in [
                    (1u32, 1u32, true),
                    (4, 4, true),
                    (8, 1, true),
                    (24, 3, true),
                    (40, 5, true),
                    (64, 1, false),
                ] {
                    let run = |mix: MixPush| -> (Vec<(u32, Recs)>, (u64, u64)) {
                        let cc = ChunkConsts {
                            m,
                            b: if decode { m / w } else { 1 },
                            w,
                            decode,
                            nb: 8192u32.div_ceil(16) + 1,
                            max_ctx: 8192,
                            ..ChunkConsts::default()
                        };
                        let mut ems: Vec<Emitter<'_>> = (0..2)
                            .map(|rk| {
                                let mut e = Emitter::new(&d, &r, Dims::new(2, rk), unbounded());
                                e.moe_sync = DRY_MOE_SYNC;
                                if w > 1 {
                                    e.conv_l = u64::from(3 + super::super::exec::WMAX - 1);
                                }
                                e
                            })
                            .collect();
                        let dm = Dims::new(2, 0);
                        let mut lws = vec![fake_layer(l); 2];
                        for lw in &mut lws {
                            lw.t8 = Some(super::super::exec::T8W {
                                w: 1 << 30,
                                s: 1 << 30,
                                n: if kind(l) == LayerKind::Qsa {
                                    dm.idx_off
                                } else {
                                    dm.conv + dm.zn
                                },
                                wo: 1 << 30,
                                so: 1 << 30,
                                sfx: 0,
                            });
                        }
                        let mut pair = fake_pair().unwrap();
                        pair.mix = mix;
                        {
                            let mut ar: Option<&mut dyn Reduce> = Some(&mut pair);
                            let mut hook = Resident(&lws);
                            emit_layer(
                                &mut ems,
                                &lws,
                                &[LayerState::default(); 2],
                                &[LayerIn::Embed(1 << 31), LayerIn::Embed(1 << 31)],
                                &[1 << 31, 1 << 31],
                                &[cc; 2],
                                &mut ar,
                                &mut hook,
                            )
                            .unwrap();
                        }
                        (
                            ems.iter()
                                .map(|e| {
                                    (
                                        e.forward_launches(),
                                        e.recs
                                            .iter()
                                            .map(|x| (x.sym.clone(), x.args.clone()))
                                            .collect::<Recs>(),
                                    )
                                })
                                .collect(),
                            (pair.calls_mix, pair.mix_stood_down),
                        )
                    };
                    let (ctl, cn) = run(MixPush::Off);
                    assert_eq!(cn, (0, 0), "the control counts nothing");
                    for arm in [MixPush::On, MixPush::B1] {
                        let (got, an) = run(arm);
                        let what =
                            format!("t1 {t1} ring {ring} L{l} m{m} w{w} decode {decode} {arm:?}");
                        let asked = decode && (arm == MixPush::On || m / w == 1);
                        if !asked {
                            assert_eq!(got, ctl, "{what}: the control's records");
                            assert_eq!(an, (0, 0), "{what}");
                            continue;
                        }
                        if !t1 && m <= 16 {
                            assert_eq!(got, ctl, "{what}: the m16 cell stands down");
                            assert_eq!(an, (0, 1), "{what}: one stand-down a layer");
                            continue;
                        }
                        assert_eq!(an, (1, 0), "{what}: one fused reduce a layer");
                        for rk in 0..2 {
                            assert_eq!(got[rk].0 + 1, ctl[rk].0, "{what} rank {rk}: launches");
                            let recs = &ctl[rk].1;
                            let gi = (0..recs.len() - 1)
                                .find(|&i| {
                                    MIX_PARENTS.contains(&recs[i].0.as_str()) && recs[i + 1].0 == AR
                                })
                                .unwrap_or_else(|| {
                                    panic!("{what}: the out projection + its reduce")
                                });
                            let (cg, ca) = (&recs[gi], &recs[gi + 1]);
                            if m <= 16 {
                                let want = if ring {
                                    k141::T8_SYM
                                } else {
                                    tiles111::SYMS[0]
                                };
                                assert_eq!(cg.0, want, "{what}: the control's cell");
                            }
                            let ag = &got[rk].1[gi];
                            assert_eq!(ag.0, format!("fn164_far_{}", cg.0), "{what}: the twin");
                            fused_seen.insert(ag.0.clone());
                            assert_eq!(&ag.1[..cg.1.len()], &cg.1[..], "{what}: the GEMM's args");
                            let b0 = cg.1.len().next_multiple_of(8);
                            assert_eq!(
                                ag.1.len(),
                                b0 + 72,
                                "{what}: the blob after the 8-aligned tail"
                            );
                            let blob = u64s(&ag.1[b0..b0 + 64]);
                            assert_eq!(
                                blob[5],
                                u64s(&ca.1)[3],
                                "{what}: the SUM is the reduce's output"
                            );
                            assert_eq!(
                                (blob[0], blob[1]),
                                (mix_scratch_off(AR_MB), mix_scratch_off(AR_MB))
                            );
                            assert_eq!((blob[2], blob[3]), (MIX_FLAG0 * 4, MIX_FLAG0 * 4));
                            assert_eq!(
                                (blob[4], blob[6], blob[7]),
                                (MIX_CTR0 * 4, MIX_DIAG0 * 4, 0)
                            );
                            let mut want = recs.clone();
                            want.remove(gi + 1);
                            want[gi] = ag.clone();
                            assert_eq!(
                                got[rk].1, want,
                                "{what} rank {rk}: only the GEMM + reduce moved"
                            );
                        }
                    }
                }
            }
        }
        // every parent the decode forms pick has been exercised
        assert_eq!(fused_seen.len(), MIX_PARENTS.len(), "{fused_seen:?}");
    }
}
