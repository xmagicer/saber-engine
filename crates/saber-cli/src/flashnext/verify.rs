//! FN15b: MTP-k through the TARGET's verify step (a child of `worker`: it drives the rank's own
//! step and drafter).
//!
//! One step of MTP-k at width `w = k + 1` (or the dynamic width's):
//!
//! 1. the target verifies `w` rows per sequence -- the committed token + `w - 1` drafts, at
//!    positions `len .. len + w` -- in ONE decode-form forward at M = b x w ([`decode_step`] at
//!    width `w`: attention at q_len w over the target's KV, the GDN conv window `3 + w - 1` from
//!    `naccept - 1` and one GDN snapshot per row, the PLE rows of every fed token with the
//!    previous step's rejected rows dropped from the PLE history, the head over every row);
//! 2. the greedy acceptance walk: draft `t` is accepted iff it equals row `t - 1`'s argmax, `a`
//!    leading drafts accepted, the step commits `a + 1` tokens (the drafts + row `a`'s argmax).
//!    Rollback is POSITIONAL: the rejected rows' KV / indexer entries lie past the new length and
//!    are rewritten by the next step (the attention reads `ctx` keys only), the next step's GDN
//!    state and conv window start from snapshot `a` (`naccept = a + 1`), the PLE history drops
//!    `w - 1 - a` rows, the PLE n-gram context advances over the committed tokens only;
//! 3. the drafter: pass 1 over the verify rows (row t = the target's stream at t + the token at
//!    t + 1 = row t's argmax, at position len + t, in the verify form at q_len w over the drafter's
//!    OWN KV -- rewriting every accepted position from the target's stream), draft 1 = row a's
//!    argmax, then passes 2..w-1 on the drafter's own stream (F37's order: the proposal after the
//!    commit).
//!
//! Greedy only here (the harness): the committed ids are a function of the verify rows' argmaxes.
//! `--fn-mtp-no-rollback` is gate D's control (every step starts from the LAST row's snapshot and
//! keeps the rejected PLE rows, as if all drafts were accepted).

use std::time::Instant;

use anyhow::{ensure, Context};
use saber_mem::ple_tier::PleTier;
use saber_model::qwen4_exp::ple_hash::PleHash;

use super::super::exec::{Emitter, W};
use super::super::run::{FnFlags, FnMtp};
use super::super::store::PackRecs;
use super::{
    bucket_for, decode_step, mtp_after_verify, mtp_chain_k, ref_req, Decoded, Graphs, Live, Prompt,
    Rank, RowOut, RowReq, TOPK,
};

/// FN16: the `--fn-mtp auto` law -- verify WIDTH (rows = k + 1) by decode BUCKET, a measured
/// table (rule 6). b1 = k 3 (FN15b's paired winner, 82.0 vs 52.3 tok/s; FN24's width-1-class
/// re-sweep: k3 86.8 / 84.4, k4 84.1, k2 80.7), b2 / b4 = FN16's paired cells, **b8 = k 2 since FN24**
/// (FN22's device chain + the width-1 class: 163.2 vs SPEC 0's 152.4 tok/s paired, byte-exact to
/// SPEC 0 -- M 24 keeps the exact reduce; k 3 at b8 146.5 and crosses into the wht6 band).
/// FN78: under `--fn-decode-ar exact` k 3 at b8 is byte-exact too (16/16, API 25/25) but loses:
/// harness 356 -> 352 tok/s, BetterBench --quick conc-8 233.4 -> 217.3 -> b8 stays k 2.
pub const AUTO_TABLE: &str = "1:4,2:3,4:3,8:3";

/// The MTP width law of a boot: the table (bucket -> width), the EMA cap arm, the widest chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpLaw {
    table: saber_sched::WidthByBatch,
    /// `--fn-mtp-dynamic`: the step's width is also capped by the batch's widest EMA.
    pub dynamic: bool,
    auto: bool,
    /// FN92 `--fn-mtp-depth adaptive`: bucket 1's width is the depth law's (ceiling = the verify
    /// cap), every other bucket the table's.
    pub adaptive: bool,
}

impl MtpLaw {
    /// The law of `--fn-mtp` / `--fn-mtp-table` / `--fn-mtp-dynamic` (`None` at SPEC 0).
    ///
    /// # Errors
    /// A table the parser refuses, a width past the verify cap (5 rows) or a table on a fixed k.
    pub fn from_flags(f: &FnFlags) -> anyhow::Result<Option<MtpLaw>> {
        let text = match f.fn_mtp {
            FnMtp::Off => {
                ensure!(
                    f.fn_mtp_table.is_none(),
                    "--fn-mtp-table needs --fn-mtp auto"
                );
                return Ok(None);
            }
            FnMtp::K(k) => {
                ensure!(
                    f.fn_mtp_table.is_none(),
                    "--fn-mtp-table needs --fn-mtp auto"
                );
                format!("1:{}", k + 1)
            }
            FnMtp::Auto => f
                .fn_mtp_table
                .clone()
                .unwrap_or_else(|| AUTO_TABLE.to_owned()),
        };
        let mut law = Self::parse(&text, f.fn_mtp_dynamic, f.fn_mtp == FnMtp::Auto)?;
        match f.fn_mtp_depth.as_str() {
            "table" => {}
            "adaptive" => {
                ensure!(
                    f.fn_mtp == FnMtp::Auto,
                    "--fn-mtp-depth adaptive needs --fn-mtp auto"
                );
                ensure!(
                    !f.fn_mtp_dynamic,
                    "--fn-mtp-depth adaptive and --fn-mtp-dynamic are two width laws: pick one"
                );
                law.adaptive = true;
            }
            x => anyhow::bail!("--fn-mtp-depth table | adaptive, not {x}"),
        }
        Ok(Some(law))
    }

    /// FN92: the adaptive depth law's ceiling at bucket 1 (drafts), `None` when the arm is off.
    #[must_use]
    pub fn adaptive_ceil(&self) -> Option<u32> {
        self.adaptive.then_some(super::super::exec::WMAX - 1)
    }

    /// The widths a bucket's steps can run (the boot pre-captures each): the table's, or at
    /// bucket 1 under the adaptive arm every width 2..=the ceiling + 1.
    #[must_use]
    pub fn widths_at(&self, bucket: u32) -> Vec<u32> {
        match self.adaptive_ceil() {
            Some(c) if bucket == 1 => (2..=c + 1).collect(),
            _ => vec![self.table_width(bucket)],
        }
    }

    /// A law from its table text.
    ///
    /// # Errors
    /// As [`Self::from_flags`].
    pub fn parse(text: &str, dynamic: bool, auto: bool) -> anyhow::Result<MtpLaw> {
        let table = saber_sched::WidthByBatch::parse(text).map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            table.rows().iter().all(|r| r.1 <= super::super::exec::WMAX),
            "an MTP width past the verify cap ({} rows): {text}",
            super::super::exec::WMAX
        );
        ensure!(
            table.rows().iter().any(|r| r.1 >= 2),
            "an MTP table that never drafts: {text} (use --fn-mtp 0)"
        );
        Ok(MtpLaw {
            table,
            dynamic,
            auto,
            adaptive: false,
        })
    }

    /// The widest chain any step drafts.
    #[must_use]
    pub fn kmax(&self) -> u32 {
        let t = self.table.widths().last().copied().unwrap_or(1) - 1;
        t.max(self.adaptive_ceil().unwrap_or(0))
    }

    /// The table's width at a decode `bucket` (1/2/4/8).
    #[must_use]
    pub fn table_width(&self, bucket: u32) -> u32 {
        self.table.width_for(bucket)
    }

    /// The step's width at `bucket` with the live sequences' EMAs (the dynamic arm caps the
    /// table's width by the batch's widest EMA cap; a width-1 cell stays SPEC 0).
    #[must_use]
    pub fn width(&self, bucket: u32, emas: &[WidthEma]) -> u32 {
        let t = self.table_width(bucket);
        if !self.dynamic || t <= 1 {
            return t;
        }
        let k = t - 1;
        emas.iter().map(|e| e.width(k)).max().unwrap_or(t).min(t)
    }
}

impl std::fmt::Display for MtpLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} width by bucket {}{}",
            if self.auto { "auto" } else { "fixed" },
            self.table,
            if self.dynamic { " + the EMA cap" } else { "" },
        )?;
        if let Some(c) = self.adaptive_ceil() {
            write!(f, " + the FN92 adaptive depth at b1 (1..={c})")?;
        }
        Ok(())
    }
}

/// FN16: what one slot's verify rows commit in the serve plane (F40's drawn-row walk).
#[derive(Debug, Clone, PartialEq)]
pub struct Walk {
    /// The committed tokens: the accepted drafts + one (the recovery or bonus token).
    pub tokens: Vec<u32>,
    /// One report per committed token (`None`: not asked, or the dense row still to decide).
    pub lps: Vec<Option<saber_serve::engine::SampledLogprobs>>,
    /// The walk's LAST row when the seam could not decide it: its token is the argmax placeholder
    /// until the caller decides the row densely ([`dense_decide`]).
    pub dense: Option<usize>,
}

impl Walk {
    /// Drafts accepted (the committed tokens but the last).
    #[must_use]
    pub fn accepted(&self) -> u32 {
        self.tokens.len() as u32 - 1
    }
}

/// The slot's sampling as row `j` of a verify step sees it (ws-f40): output position
/// `position + j`, the penalties over the output so far + `draft[..j]` -- what a width-1 run has
/// at that position.
#[must_use]
pub fn sampling_at(
    s: &saber_sched::SlotSampling,
    j: usize,
    draft: &[u32],
) -> saber_sched::SlotSampling {
    let mut t = s.clone();
    t.position += j as u64;
    if j > 0 {
        if let Some(h) = &s.penalties {
            let mut out = h.output.clone();
            out.extend_from_slice(&draft[..j.min(draft.len())]);
            t.penalties = Some(std::sync::Arc::new(saber_sched::PenaltyHistory {
                prompt: h.prompt.clone(),
                output: out,
            }));
        }
    }
    t
}

/// FN16: the serve plane's walk over one slot's verify rows -- the 27B's DRAWN-ROW law, its own
/// code (`saber_engine::drawn_rows` over a `saber_engine::SlotDraws`): row `j`'s token is the
/// sampler's decision for output position `position + j` keyed by `(seed, position, choice)`
/// (the argmax for a greedy slot), a draft is accepted iff it IS that decision, the walk stops at
/// the first row whose decision is not the draft (committed: the recovery token) or after the last
/// draft (the bonus). `rows` = the slot's `w` verify rows, `draft` its `w - 1` drafts (none when
/// the slot may not draft this step: its scheduled width is 0).
#[must_use]
pub fn walk_slot(
    sampler: &std::sync::Arc<saber_serve::sampler::Sampler>,
    rows: &[RowOut],
    draft: &[u32],
    s: &saber_sched::SlotSampling,
) -> Walk {
    let (params, pen, _) = super::super::serve::row_req(s);
    let d = &draft[..draft.len().min(rows.len().saturating_sub(1))];
    let draws = saber_engine::SlotDraws::new(std::sync::Arc::clone(sampler), params, s, d.to_vec());
    let am: Vec<u32> = rows.iter().map(|o| o.top[0].0).collect();
    let targets = saber_engine::drawn_rows(&draws, d, &am, &mut |j| {
        rows.get(j)
            .map(|o| super::super::serve::device_row(o, &pen))
    });
    let decided = draws.take();
    let mut lps: Vec<Option<saber_serve::engine::SampledLogprobs>> =
        decided.iter().map(|x| x.logprobs.clone()).collect();
    let dense = (decided.len() < targets.len()).then(|| targets.len() - 1);
    lps.resize(targets.len(), None);
    Walk {
        tokens: targets,
        lps,
        dense,
    }
}

/// FN16: decide a walk's undecided last row from its DENSE logits (both vocabulary halves), keyed
/// as the seam would have been (position + j; the penalties over output + `draft[..j]`).
pub fn dense_decide(
    sampler: &saber_serve::sampler::Sampler,
    walk: &mut Walk,
    row: &RowOut,
    dense: Vec<f32>,
    draft: &[u32],
    s: &saber_sched::SlotSampling,
) {
    let Some(j) = walk.dense.take() else {
        return;
    };
    let sj = sampling_at(s, j, draft);
    let (params, pen, _) = super::super::serve::row_req(s);
    if let Some((t, lp)) =
        super::super::serve::sample(sampler, row, Some(dense), &params, &pen, &sj)
    {
        walk.tokens[j] = t;
        walk.lps[j] = lp;
    }
}

/// The 27B's acceptance-EMA width cap (`saber_sched::width`): `obs = a + (a >= scheduled)`,
/// `ema = 0.35 obs + 0.65 ema`, `w = max(ceil(ema) + 2, 2)`, at most `k + 1`; cold = full width.
#[derive(Debug, Clone, Copy, Default)]
pub struct WidthEma {
    ema: Option<f64>,
}

impl WidthEma {
    pub fn observe(&mut self, accepted: u32, scheduled: u32) {
        let obs = f64::from(accepted + u32::from(accepted >= scheduled));
        self.ema = Some(match self.ema {
            None => obs,
            Some(e) => 0.35 * obs + 0.65 * e,
        });
    }
    #[must_use]
    pub fn width(&self, k: u32) -> u32 {
        match self.ema {
            None => k + 1,
            Some(e) => ((e.ceil() as u32) + 2).max(2).min(k + 1),
        }
    }
}

/// The greedy acceptance walk over one sequence's verify rows: `drafts` (w - 1) against the
/// rows' argmaxes `am` (w). Returns `a` (accepted drafts); the step commits `am[..=a]`.
#[must_use]
pub fn greedy_walk(drafts: &[u32], am: &[u32]) -> u32 {
    let mut a = 0;
    while (a as usize) < drafts.len() && drafts[a as usize] == am[a as usize] {
        a += 1;
    }
    a
}

/// FN15b: decode `n` tokens (the prefill's pick included) for every sequence of `lives` with
/// MTP-k through the verify step. The drafter's first pass ran after each prefill
/// (`MtpRank::first`).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn decode_mtp(
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
    let bucket = bucket_for(b);
    let k = rk.mtp.as_ref().context("mtp")?.k;
    let want = n.saturating_sub(1) as usize;
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
    let mut ema = vec![WidthEma::default(); b];
    // FN16: the boot's width law at this group's bucket (a fixed k: k + 1 at every bucket; auto:
    // the table), capped by the batch's widest EMA under --fn-mtp-dynamic
    let law = f.mtp_law()?.context("the MTP law")?;
    ensure!(
        law.kmax() <= k,
        "the law drafts {} > the drafter's chain {k}",
        law.kmax()
    );
    // FN92: the adaptive depth arm at b1 -- rank 0 decides from its commits and its clock, rank 1
    // takes rank 0's depth over the pair (one 4-byte exchange an iteration)
    let mut ctl = (law.adaptive && bucket == 1)
        .then(|| {
            law.adaptive_ceil()
                .map(|c| super::super::depth::DepthCtl::new(c as usize))
        })
        .flatten();
    let mut accq = super::super::depth::SeqAcc::default();
    let width = |ema: &[WidthEma], ctl: &Option<super::super::depth::DepthCtl>| -> u32 {
        match ctl {
            Some(c) => c.depth() as u32 + 1,
            None => law.width(bucket, ema),
        }
    };
    // the first drafts: pass 1 ran after the prefill; passes 2..w-1 now
    let mut w = width(&ema, &ctl);
    let t0 = Instant::now();
    let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(b);
    let mut dpos = Vec::with_capacity(b);
    for lv in lives.iter() {
        let (d, p) = rk.mtp.as_ref().context("mtp")?.first[lv.slot as usize]
            .context("the prefill's first MTP pass")?;
        drafts.push(vec![d]);
        dpos.push(p);
    }
    rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
    mtp_chain_k(rk, em, recs, &mut drafts, &dpos, bucket, w - 1)?;
    let dev: f64 = rk.mtp.as_ref().context("mtp")?.pass_ms.iter().sum();
    res.mtp_ms.push((t0.elapsed().as_secs_f64() * 1e3, dev));
    while res.ids.iter().any(|v| v.len() < want) {
        // FN56: the iteration's wall (verify step + the walk + the drafter round), its census
        let ti = Instant::now();
        let def0 = rk.tier_def_ms;
        for d in &mut drafts {
            d.truncate(w as usize - 1);
        }
        for (i, d) in drafts.iter().enumerate() {
            res.mtp[i].push(d.clone());
        }
        // the probes: row t of sequence i predicts its generated token `1 + ids + t`
        let wu = w as usize;
        let reqs: Vec<RowReq> = (0..bucket as usize * wu)
            .map(|r| {
                let (i, t) = (r / wu, r % wu);
                ref_req(
                    group.get(i).copied(),
                    1 + res.ids.get(i).map_or(0, Vec::len) + t,
                )
            })
            .collect();
        let pos0: Vec<u32> = lives.iter().map(|l| l.len).collect();
        let (outs, mut stat) = decode_step(
            rk, em, ple, ph, recs, lives, bucket, *dp, gr, f, &reqs, w, &drafts,
        )?;
        *dp = 1 - *dp;
        let mut acc = Vec::with_capacity(b);
        let mut ams = Vec::with_capacity(b);
        for (i, lv) in lives.iter_mut().enumerate() {
            let rows = &outs[i * wu..(i + 1) * wu];
            let am: Vec<u32> = rows.iter().map(|o| o.top[0].0).collect();
            let a = greedy_walk(&drafts[i], &am);
            for t in 0..=a as usize {
                if res.ids[i].len() < want {
                    res.ids[i].push(am[t]);
                    res.tops[i].push(rows[t].top[..TOPK].to_vec());
                    res.probes[i].push((rows[t].lse, rows[t].probe.clone()));
                    res.shas[i].push(rows[t].sha.clone());
                }
            }
            // the PLE n-gram context: the committed token advanced in the step, now the accepted drafts
            let mut c = lv.ctx;
            for &d in &drafts[i][..a as usize] {
                let _ = ph.ids_step(&mut c, &[d]);
            }
            lv.ctx = c;
            lv.len += a + 1;
            lv.next = am[a as usize];
            lv.ple_at = *dp;
            if f.fn_mtp_no_rollback {
                // the control: commit as the walk does, but start the next step as if every
                // draft had been accepted (the state / conv / PLE rollback dropped)
                lv.naccept = w;
                lv.drop = 0;
            } else {
                lv.naccept = a + 1;
                lv.drop = w - 1 - a;
            }
            ema[i].observe(a, w - 1);
            if ctl.is_some() && w > 1 {
                accq.observe(w - 1, a);
            }
            acc.push(a);
            ams.push(am);
        }
        stat.committed_at = Some(Instant::now());
        if let Some(c) = ctl.as_mut() {
            c.time_verify(w, ti.elapsed().as_secs_f64() * 1e3);
            let mine = if rk.rank == 0 {
                c.decide(&accq).0 as u32
            } else {
                0
            };
            let theirs = rk
                .peer
                .exchange()
                .xchg_bytes("fn.depth", &mine.to_le_bytes())?;
            let d = if rk.rank == 0 {
                mine
            } else {
                u32::from_le_bytes(
                    theirs
                        .get(..4)
                        .and_then(|b| b.try_into().ok())
                        .context("FN92: the leader's depth frame")?,
                )
            };
            c.follow(d as usize);
        }
        res.acc.push(acc.clone());
        res.widths.push(w);
        res.steps.push(stat);
        if res.ids.iter().all(|v| v.len() >= want) {
            break;
        }
        // the drafter after the commit: pass 1 over the verify rows at q_len w, then the chain
        // (FN16: none when the next step is SPEC 0's width -- auto's b8 cell)
        if width(&ema, &ctl) == 1 {
            drafts = vec![Vec::new(); b];
            w = 1;
            res.mtp_ms.push((0.0, 0.0));
            iter_census(&mut res, ti, (0.0, 0.0), rk.tier_def_ms - def0);
            continue;
        }
        let tm = Instant::now();
        rk.mtp.as_mut().context("mtp")?.pass_ms.clear();
        let mut toks: Vec<u32> = Vec::with_capacity(bucket as usize * wu);
        for i in 0..bucket as usize {
            for t in 0..wu {
                toks.push(ams.get(i).map_or(0, |am| am[t]));
            }
        }
        let wn = width(&ema, &ctl);
        // FN58b: the committed tokens' PLE rows under the round (`--fn-ple-overlap layer1`)
        let ov = rk.ple_ov.as_ref().is_some_and(|o| o.pre_on);
        let mut hook = || super::ple_pregather(ple, ph, lives.iter_mut());
        drafts = mtp_after_verify(
            rk,
            em,
            recs,
            &toks,
            &pos0,
            w,
            &acc,
            bucket,
            wn - 1,
            if ov { Some(&mut hook) } else { None },
        )?;
        w = wn;
        let dev: f64 = rk.mtp.as_ref().context("mtp")?.pass_ms.iter().sum();
        let dh = tm.elapsed().as_secs_f64() * 1e3;
        if let Some(c) = ctl.as_mut() {
            c.time_round(wn - 1, dh);
        }
        res.mtp_ms.push((dh, dev));
        iter_census(&mut res, ti, (dh, dev), rk.tier_def_ms - def0);
        // FN58c: the round's host parts (`rd_*`)
        if let Some(s) = res.steps.last_mut() {
            for (k, v) in std::mem::take(&mut rk.rd_t) {
                s.host.insert(k, v);
            }
        }
        ensure!(
            lives.iter().all(|l| l.len + w <= rk.max_ctx),
            "MTP: a sequence reached the pools' context"
        );
    }
    if let (Some(c), 0) = (&ctl, rk.rank) {
        println!("fnmtpdepth rank0 b1: {}", c.summary());
    }
    // FN56: a tier end deferred past the last step lands before the run's counters are read
    super::tier_flush(rk)?;
    let _ = W;
    Ok(res)
}

/// FN56: the census of one MTP iteration on its step's record -- the wall from the verify step's
/// start to the drafter round's end (`z_iter`), the round's host / device ms and the deferred
/// tier end's host ms (run under the round).
fn iter_census(res: &mut Decoded, ti: Instant, round: (f64, f64), defer_ms: f64) {
    if let Some(s) = res.steps.last_mut() {
        s.host.insert("z_iter", ti.elapsed().as_secs_f64() * 1e3);
        s.host.insert("z_round_host", round.0);
        s.host.insert("z_round_dev", round.1);
        s.host.insert("z_defer", defer_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flashnext::exec::ChunkConsts;
    use crate::flashnext::exec::PadLaw;

    #[test]
    fn the_greedy_walk_accepts_the_matched_prefix_and_commits_one_more() {
        // drafts d1..d4 vs the rows' argmaxes: row t's argmax is the token after row t
        assert_eq!(greedy_walk(&[5, 6, 7, 8], &[5, 6, 9, 8, 1]), 2);
        assert_eq!(greedy_walk(&[5, 6, 7, 8], &[5, 6, 7, 8, 1]), 4);
        assert_eq!(greedy_walk(&[5, 6, 7, 8], &[4, 6, 7, 8, 1]), 0);
        assert_eq!(greedy_walk(&[], &[3]), 0);
    }

    fn row(top: &[(u32, f32)]) -> RowOut {
        let mut t: Vec<(u32, f32)> = top.to_vec();
        // a full seam: 64 entries, value-desc / id-asc
        let mut id = 100_000;
        while t.len() < super::super::SEAM_K {
            t.push((id, -30.0));
            id += 1;
        }
        let lse = t.iter().map(|x| f64::from(x.1).exp()).sum::<f64>().ln() as f32;
        RowOut {
            top: t,
            lse,
            lse_t: Some(lse),
            probe: Vec::new(),
            sha: None,
        }
    }

    #[test]
    fn the_auto_law_is_width_by_bucket_and_a_fixed_k_is_one_width() {
        let a = MtpLaw::parse(AUTO_TABLE, false, true).unwrap();
        assert_eq!(a.kmax(), 3);
        let w: Vec<u32> = [1, 2, 4, 8].iter().map(|&b| a.width(b, &[])).collect();
        assert_eq!(w[0], 4, "b1 = k 3 (FN15b's paired winner)");
        assert_eq!(
            w[3], 3,
            "b8 = k 2 (FN24: the width-1 class + the device chain, byte-exact at M 24)"
        );
        let k2 = MtpLaw::parse("1:3", false, false).unwrap();
        assert!([1, 2, 4, 8].iter().all(|&b| k2.width(b, &[]) == 3));
        assert!(
            MtpLaw::parse("1:6", false, true).is_err(),
            "past the 5-row cap"
        );
        assert!(MtpLaw::parse("1:1", false, true).is_err(), "never drafts");
        // the EMA arm caps the table's width by the batch's widest EMA; a SPEC-0 cell stays 1
        let d = MtpLaw::parse("1:5,8:1", true, true).unwrap();
        let mut e = WidthEma::default();
        e.observe(0, 4);
        assert_eq!(d.width(1, &[e]), 2);
        assert_eq!(
            d.width(1, &[e, WidthEma::default()]),
            5,
            "a cold sequence: full"
        );
        assert_eq!(d.width(8, &[WidthEma::default()]), 1);
    }

    #[test]
    fn a_narrower_step_resumes_from_its_snapshot_the_stride_is_pool_w() {
        // pool_w 4 (k 3), this step w 1 after a commit of 3: the table holds every snapshot, so
        // sidx[s * 4 + naccept - 1] is sequence s's snapshot 2, not the next sequence's
        let (tb, offs) = ChunkConsts::host_verify(&[10, 3], 1, &[3, 1], &[1, 0], 4, 4);
        let o = offs.iter().find(|x| x.0 == "sidx").unwrap().1;
        let sidx: Vec<i32> = (0..8)
            .map(|i| i32::from_le_bytes(tb[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
            .collect();
        assert_eq!(sidx, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(sidx[3 - 1], 3, "sequence 0 resumes from its own snapshot 2");
    }

    #[test]
    fn the_greedy_walk_through_the_drawn_row_code_is_the_argmax_walk() {
        let sm = std::sync::Arc::new(saber_serve::sampler::Sampler::new());
        let s = saber_sched::SlotSampling::greedy();
        let rows = [
            row(&[(5, 3.0), (1, 2.0)]),
            row(&[(6, 3.0), (1, 2.0)]),
            row(&[(9, 3.0), (7, 2.9)]),
            row(&[(8, 3.0), (1, 2.0)]),
        ];
        let w = walk_slot(&sm, &rows, &[5, 6, 7], &s);
        assert_eq!(
            w.tokens,
            vec![5, 6, 9],
            "two accepted, the recovery token row 2's argmax"
        );
        assert_eq!(w.accepted(), 2);
        assert_eq!(w.dense, None);
        let am: Vec<u32> = rows.iter().map(|o| o.top[0].0).collect();
        assert_eq!(
            greedy_walk(&[5, 6, 7], &am),
            w.accepted(),
            "== the harness walk"
        );
        let full = walk_slot(&sm, &rows, &[5, 6, 9], &s);
        assert_eq!(
            full.tokens,
            vec![5, 6, 9, 8],
            "every draft accepted + the bonus"
        );
        let none = walk_slot(&sm, &rows, &[], &s);
        assert_eq!(
            none.tokens,
            vec![5],
            "a slot that may not draft commits row 0"
        );
    }

    #[test]
    fn a_seeded_walk_commits_the_width_one_draws_and_a_shifted_seed_differs() {
        let sm = std::sync::Arc::new(saber_serve::sampler::Sampler::new());
        let mut s = saber_sched::SlotSampling::greedy();
        s.params.temperature = 1.0;
        s.params.top_k = 20;
        s.params.top_p = 0.95;
        s.params.seed = Some(7);
        s.position = 40;
        let flat: Vec<(u32, f32)> = (0..20).map(|i| (i, 1.0 - 0.01 * i as f32)).collect();
        let rows: Vec<RowOut> = (0..4).map(|_| row(&flat)).collect();
        // the width-1 draws: row j alone at output position 40 + j
        let draw = |seed: u64, j: usize| -> u32 {
            let mut t = s.clone();
            t.params.seed = Some(seed);
            let sj = sampling_at(&t, j, &[]);
            let (p, pen, _) = crate::flashnext::serve::row_req(&sj);
            crate::flashnext::serve::sample(&sm, &rows[j], None, &p, &pen, &sj)
                .unwrap()
                .0
        };
        let d: Vec<u32> = (0..3).map(|j| draw(7, j)).collect();
        let w = walk_slot(&sm, &rows, &d, &s);
        assert_eq!(
            w.tokens,
            vec![d[0], d[1], d[2], draw(7, 3)],
            "drafts == draws: all accepted"
        );
        // a wrong first draft: the recovery token is row 0's own draw
        let wrong = if d[0] == 0 { 1 } else { 0 };
        let w2 = walk_slot(&sm, &rows, &[wrong, d[1], d[2]], &s);
        assert_eq!(w2.tokens, vec![d[0]]);
        // the control: another seed draws another stream
        let other: Vec<u32> = (0..4).map(|j| draw(8, j)).collect();
        assert_ne!(other, w.tokens);
    }

    #[test]
    fn the_width_ema_is_the_27bs_cap() {
        let mut e = WidthEma::default();
        assert_eq!(e.width(4), 5, "cold: full width");
        e.observe(0, 4);
        assert_eq!(e.width(4), 2, "nothing accepted: at least 2");
        for _ in 0..8 {
            e.observe(4, 4);
        }
        assert_eq!(
            e.width(4),
            5,
            "a full accept observes one above and climbs back"
        );
        let mut e = WidthEma::default();
        e.observe(1, 4);
        assert_eq!(e.width(4), 3);
    }

    /// FN17: the pool slots every batch sequence of a decode step touches, read back from the
    /// step's tables (KV slot, ring slot, ring block, block table, conv slot, GDN snapshots, the
    /// PLE history's row -> sequence).
    fn touched(
        tb: &[u8],
        offs: &[(&'static str, usize)],
        b: usize,
        w: usize,
        nb: u32,
        pool_w: u32,
    ) -> Vec<std::collections::BTreeSet<i64>> {
        let get = |k: &str, n: usize, sz: usize| -> Vec<i64> {
            let o = offs.iter().find(|x| x.0 == k).expect(k).1;
            (0..n)
                .map(|i| {
                    let x = &tb[o + i * sz..o + (i + 1) * sz];
                    if sz == 8 {
                        i64::from_le_bytes(x.try_into().unwrap())
                    } else {
                        i64::from(i32::from_le_bytes(x.try_into().unwrap()))
                    }
                })
                .collect()
        };
        let nb = i64::from(nb);
        let ring = crate::flashnext::exec::RING as i64;
        let pwu = pool_w.max(w as u32) as usize;
        let mut t = vec![std::collections::BTreeSet::new(); b];
        for (r, x) in get("slot", b * w, 8).into_iter().enumerate() {
            t[r / w].insert(x / (nb * 16));
        }
        for (r, x) in get("rslot", b * w, 8).into_iter().enumerate() {
            t[r / w].insert(x / ring);
        }
        for (s, x) in get("ring_row0", b, 4).into_iter().enumerate() {
            t[s].insert(x);
        }
        for (s, x) in get("cidx", b, 4).into_iter().enumerate() {
            t[s].insert(x - 1);
        }
        for (r, x) in get("sidx", b * pwu, 4).into_iter().enumerate() {
            t[r / pwu].insert((x - 1) / i64::from(pool_w));
        }
        for (i, x) in get("bt", b * nb as usize, 4).into_iter().enumerate() {
            t[i / nb as usize].insert(x / nb);
        }
        if offs.iter().any(|x| x.0 == "bt_rows") {
            // FN24: the per-row block tables (row r = sequence r / w)
            for (i, x) in get("bt_rows", b * w * nb as usize, 4)
                .into_iter()
                .enumerate()
            {
                t[i / (w * nb as usize)].insert(x / nb);
            }
        }
        let ple = if offs.iter().any(|x| x.0 == "ple_seq") {
            "ple_seq"
        } else {
            "zeros"
        };
        for (r, x) in get(ple, b * w, 4).into_iter().enumerate() {
            t[r / w].insert(x);
        }
        t
    }

    /// FN17, the failing schedule on the host: three ready sequences (pool slots 0..3) decode at the
    /// padded bucket 4 while a > 2,048-token prompt sits in pool slot 3 between its first and its
    /// second chunk. Under the pad law every pad row touches ONLY the spare slot (8) in every
    /// per-sequence table and the live rows' tables are FN16's bytes; the pre-FN17 law (the
    /// control) touches the prefilling sequence's slot 3.
    #[test]
    fn pad_rows_never_touch_a_live_or_prefilling_slot() {
        const SPARE: u32 = crate::flashnext::worker::MAX_SEQS;
        let nb = 513u32;
        // the serve plane's pool: ready 0..3, the long prompt prefilling at 3 (chunk 1 of 2 in)
        let prefilling = 3i64;
        for (w, pool_w, pos) in [
            (1u32, 1u32, vec![700u32, 12, 3000, 0]),
            (3, 5, vec![700, 12, 3000, 0]),
            (1, 4, vec![40, 41, 42, 0, 0, 0, 0, 0]),
        ] {
            let b = pos.len();
            let live = 3usize;
            let wu = w as usize;
            let na = vec![1u32; b];
            let dr = vec![0u32; b];
            let law = PadLaw {
                live,
                pad: Some(SPARE),
                idx: None,
            };
            let (tb, offs) = ChunkConsts::host_verify_at(&pos, w, &na, &dr, nb, pool_w, law);
            let t = touched(&tb, &offs, b, wu, nb, pool_w);
            for (s, set) in t.iter().enumerate() {
                let want = if s < live { s as i64 } else { i64::from(SPARE) };
                assert_eq!(
                    set.iter().copied().collect::<Vec<_>>(),
                    vec![want],
                    "w {w} b {b}: batch sequence {s} touches {set:?}"
                );
                assert!(!set.contains(&prefilling) || s == prefilling as usize && s < live);
            }
            // the PLE pad pseudo-sequence: every pad row inside [seq_row0, seq_row0 + mv) at 8
            let g32 = |k: &str, i: usize| -> i32 {
                let o = offs.iter().find(|x| x.0 == k).expect(k).1;
                i32::from_le_bytes(tb[o + 4 * i..o + 4 * i + 4].try_into().unwrap())
            };
            let sp = SPARE as usize;
            assert_eq!(g32("seq_row0", sp), (live * wu) as i32);
            assert_eq!(g32("mv", sp), ((b - live) * wu) as i32);
            assert_eq!((g32("drop", sp), g32("hi32", sp)), (0, 1));
            // the live rows' tables are FN16's bytes (every entry a live row or sequence reads)
            let (tf, of) = ChunkConsts::host_verify(&pos, w, &na, &dr, nb, pool_w);
            let pwu = pool_w.max(w) as usize;
            for (k, n, sz) in [
                ("pos", live * wu, 8),
                ("slot", live * wu, 8),
                ("zeros", b * wu, 4),
                ("cu", b + 1, 4),
                ("seq_row0", b, 4),
                ("ring_row0", live, 4),
                ("drop", b, 4),
                ("cidx", live, 4),
                ("sidx", live * pwu, 4),
                ("naccept", b, 4),
                ("p0v", b, 8),
                ("hi32", b, 4),
                ("mv", b, 4),
                ("ctx", b, 4),
                ("rslot", live * wu, 8),
                ("bt", live * nb as usize, 4),
            ] {
                let o1 = offs.iter().find(|x| x.0 == k).unwrap().1;
                let o2 = of.iter().find(|x| x.0 == k).unwrap().1;
                assert_eq!(o1, o2, "{k}: the table's offset moved");
                assert_eq!(tb[o1..o1 + n * sz], tf[o2..o2 + n * sz], "{k}");
            }
            // the capture law: a (bucket, width)'s table LAYOUT does not depend on how many of
            // its rows are live (a graph captured on a padded step replays on an unpadded one)
            for l2 in 1..=b {
                let (t2, o2) = ChunkConsts::host_verify_at(
                    &pos,
                    w,
                    &na,
                    &dr,
                    nb,
                    pool_w,
                    PadLaw {
                        live: l2,
                        pad: Some(SPARE),
                        idx: None,
                    },
                );
                assert_eq!(o2, offs, "live {l2}: the table layout moved");
                assert_eq!(t2.len(), tb.len());
                if l2 == b {
                    // no pads: the PLE row table is the row -> sequence table
                    let a = o2.iter().find(|x| x.0 == "ple_seq").unwrap().1;
                    let z = o2.iter().find(|x| x.0 == "zeros").unwrap().1;
                    assert_eq!(t2[a..a + 4 * b * wu], t2[z..z + 4 * b * wu]);
                }
            }
            // the control: FN14's pad law touches the prefilling slot 3
            let ctl = PadLaw {
                live,
                pad: None,
                idx: None,
            };
            let (tc, oc) = ChunkConsts::host_verify_at(&pos, w, &na, &dr, nb, pool_w, ctl);
            let tcl = touched(&tc, &oc, b, wu, nb, pool_w);
            assert!(
                tcl[3].contains(&prefilling),
                "the control reaches the prefilling sequence's slot"
            );
            assert_eq!(
                (tc.clone(), oc.clone()),
                (tf.clone(), of.clone()),
                "control == FN16"
            );
        }
    }

    #[test]
    fn a_width_one_verify_table_is_spec0s() {
        let pos = [7u32, 30, 0, 0];
        let a = ChunkConsts::host_decode(&pos, 9);
        let b = ChunkConsts::host_verify(&pos, 1, &[1; 4], &[0; 4], 9, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn a_verify_table_indexes_rows_snapshots_and_windows() {
        let (tb, offs) = ChunkConsts::host_verify(&[10, 3], 3, &[2, 1], &[1, 2], 4, 5);
        let get = |k: &str, n: usize, sz: usize| -> Vec<i64> {
            let o = offs.iter().find(|x| x.0 == k).expect(k).1;
            (0..n)
                .map(|i| {
                    let b = &tb[o + i * sz..o + (i + 1) * sz];
                    if sz == 8 {
                        i64::from_le_bytes(b.try_into().unwrap())
                    } else {
                        i64::from(i32::from_le_bytes(b.try_into().unwrap()))
                    }
                })
                .collect()
        };
        assert_eq!(get("pos", 6, 8), vec![10, 11, 12, 3, 4, 5]);
        assert_eq!(get("slot", 6, 8), vec![10, 11, 12, 64 + 3, 64 + 4, 64 + 5]);
        assert_eq!(get("zeros", 6, 4), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(get("cu", 3, 4), vec![0, 3, 6]);
        assert_eq!(get("seq_row0", 2, 4), vec![0, 3]);
        // FN16: every snapshot of a sequence (stride pool_w = 5), not only the step's w rows
        assert_eq!(get("sidx", 10, 4), (1..=10).collect::<Vec<i64>>());
        assert_eq!(get("naccept", 2, 4), vec![2, 1]);
        assert_eq!(get("drop", 2, 4), vec![1, 2]);
        assert_eq!(get("cidx", 2, 4), vec![1, 2]);
        assert_eq!(get("mv", 2, 4), vec![3, 3]);
        assert_eq!(get("ctx", 2, 4), vec![13, 6]);
        assert_eq!(get("rslot", 6, 8), vec![2, 3, 4, 8 + 3, 8 + 4, 8 + 5]);
        assert_eq!(get("p0v", 2, 8), vec![10, 3]);
    }
}
