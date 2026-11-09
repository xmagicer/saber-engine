//! P27-12: how the DFlash2 selector RANKS the candidates it already holds.
//!
//! Every proposal carries a lattice: `selector_top_k` (16) candidates at each of the 7 drafted
//! positions (`sel.cand64`, sorted by the drafter's exact-rerank logit) and the edge scores
//! `S[d][pred][cand]` (`sel.scores`, f32: `bf16(edge(pred, cand)) + unary[cand]`). The device's walk
//! (`dflash_selector_walk_greedy`, prod's `_selector_walk_kernel`) takes the argmax of the row its
//! previous pick names, one position at a time. P27-10 priced the oracle (the target's token is in
//! the top-16 at every reached depth, picked first) at +39 % tokens a step over that walk.
//!
//! [`SelectorRank::ExpLen`] re-ranks the same lattice on the HOST, after the walk, by the chain
//! that MAXIMISES THE EXPECTED ACCEPTED LENGTH under a softmax reading of each row at temperature
//! [`EXPLEN_TEMPERATURE`]: with `p_d(c | prev) = softmax(S[d][prev] / T)[c]`, the value of picking
//! `c` at depth `d` is `V_d(c) = p_d(c | prev) * (1 + max_c' V_{d+1}(c' | c))` -- a backward pass
//! over the 7 x 16 x 16 lattice, then the forward read of the policy. It needs no kernel and no
//! code object: the host already reads the walk's tokens back, and the scores and candidates ride
//! the same pinned batch (8 KiB a live request).
//!
//! The law it rests on (LEDGER P27-12): at temperature 0 the target's greedy ids do NOT depend on
//! the proposal -- verify accepts the target's argmax -- so a re-rank moves acceptance and the
//! drafter's later context only; the committed stream stays the greedy stream. Seeded sampling is
//! a different matter (the drawn-row walk draws over the proposed rows): reported, not gated equal.
//!
//! Host-priced on P27-10's 2,427 lattices (`tools/capture/p27_12/rank_sim.py`): walk 5.344 tokens
//! a step; this 5.404 (+1.1 %, prompt-bootstrap 95 % CI +0.019..+0.102 tokens); the unary alone
//! 5.016, Viterbi on raw scores 4.810, the product of row softmaxes 5.346, rank blends and a
//! depth x rank calibration table 5.17..5.36.

/// The temperature the expected-length re-rank reads each selector row at (held out by prompt on
/// P27-10's lattices: the gain is flat over 1.5..3; 2.0 is the round middle).
pub const EXPLEN_TEMPERATURE: f64 = 2.0;

/// Which chain a proposal commits to out of the selector's lattice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectorRank {
    /// The device's greedy walk, prod's (`walk`, THE DEFAULT and the control).
    #[default]
    Walk,
    /// The host's expected-length chain over the same lattice (`explen`, an ARM).
    ExpLen,
    /// Calibrated conditional edges for the native Tandem B16 nine-node verifier.
    TandemB16Calibrated,
}

impl SelectorRank {
    /// The flag's spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Walk => "walk",
            Self::ExpLen => "explen",
            Self::TandemB16Calibrated => "tandem-b16-calibrated",
        }
    }

    /// Parse the flag's value; refused by name otherwise.
    ///
    /// # Errors
    /// A value that is neither form.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "walk" => Ok(Self::Walk),
            "explen" => Ok(Self::ExpLen),
            "tandem-b16-calibrated" => Ok(Self::TandemB16Calibrated),
            o => Err(format!(
                "--engine-selector-rank `{o}`: the forms are `walk` (the device's greedy walk, \
                 the default), `explen` (expected-length chain), and `tandem-b16-calibrated` (calibrated tree)"
            )),
        }
    }

    /// True for expected-length *chain* reranking. Tree ranking uses the separate tree-plan read.
    #[must_use]
    pub const fn reranks(self) -> bool {
        matches!(self, Self::ExpLen)
    }
}

/// Softmax of one row at temperature `t`, in f64.
fn softmax_row(row: &[f32], t: f64, out: &mut [f64]) {
    let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let mut sum = 0.0;
    for (o, &v) in out.iter_mut().zip(row) {
        *o = ((v as f64 - mx) / t).exp();
        sum += *o;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// Index of the maximum, ties to the LOWEST index (the walk's tie law).
fn argmax(v: &[f64]) -> usize {
    let mut bi = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[bi] {
            bi = i;
        }
    }
    bi
}

/// The expected-length chain over one request's lattice.
///
/// `scores` is `sel.scores` for the request, `[steps][k][k]` f32 (`[d][pred][cand]`); the result
/// is the candidate INDEX picked at each of the `steps` positions (the token is
/// `cand[d][index]`). Depth 0's rows all read the anchor, so its row 0 is used.
///
/// # Panics
/// Never on a well-formed lattice; a short slice yields the indices the walk could compute
/// (callers pass exactly `steps * k * k` values).
#[must_use]
pub fn explen_path(scores: &[f32], steps: usize, k: usize, t: f64) -> Vec<usize> {
    if steps == 0 || k == 0 || scores.len() < steps * k * k {
        return Vec::new();
    }
    let row = |d: usize, p: usize| &scores[(d * k + p) * k..(d * k + p + 1) * k];
    // v[c] = the value of having picked candidate c at depth d (filled from the last depth back);
    // pol[d][p] = the best candidate at depth d given candidate p at depth d - 1.
    let mut v = vec![0.0_f64; k];
    let mut pol = vec![vec![0usize; k]; steps];
    let mut sm = vec![0.0_f64; k];
    let mut q = vec![0.0_f64; k];
    for d in (1..steps).rev() {
        let mut vn = vec![0.0_f64; k];
        for p in 0..k {
            softmax_row(row(d, p), t, &mut sm);
            for c in 0..k {
                q[c] = sm[c] * (1.0 + v[c]);
            }
            let b = argmax(&q);
            pol[d][p] = b;
            vn[p] = q[b];
        }
        v = vn;
    }
    softmax_row(row(0, 0), t, &mut sm);
    for c in 0..k {
        q[c] = sm[c] * (1.0 + v[c]);
    }
    let mut c = argmax(&q);
    let mut path = Vec::with_capacity(steps);
    path.push(c);
    for pd in pol.iter().skip(1) {
        c = pd[c];
        path.push(c);
    }
    path
}

/// The device walk's chain over the same lattice (the host reference of
/// `dflash_selector_walk_greedy`): the argmax of the row the previous pick names.
#[must_use]
pub fn walk_path(scores: &[f32], steps: usize, k: usize) -> Vec<usize> {
    let mut path = Vec::with_capacity(steps);
    let mut prev = 0usize;
    for d in 0..steps {
        let r = &scores[(d * k + prev) * k..(d * k + prev + 1) * k];
        let mut bi = 0;
        for (i, &x) in r.iter().enumerate() {
            if x > r[bi] {
                bi = i;
            }
        }
        path.push(bi);
        prev = bi;
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: usize = 4;
    const ST: usize = 3;

    fn lattice(f: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
        let mut s = vec![0.0; ST * K * K];
        for d in 0..ST {
            for p in 0..K {
                for c in 0..K {
                    s[(d * K + p) * K + c] = f(d, p, c);
                }
            }
        }
        s
    }

    #[test]
    fn the_forms_parse_and_walk_is_the_default() {
        assert_eq!(SelectorRank::default(), SelectorRank::Walk);
        assert_eq!(SelectorRank::parse("walk"), Ok(SelectorRank::Walk));
        assert_eq!(SelectorRank::parse("explen"), Ok(SelectorRank::ExpLen));
        assert!(SelectorRank::parse("viterbi").is_err());
        assert!(!SelectorRank::Walk.reranks());
        assert!(SelectorRank::ExpLen.reranks());
        assert!(!SelectorRank::TandemB16Calibrated.reranks());
        for f in [SelectorRank::Walk, SelectorRank::ExpLen, SelectorRank::TandemB16Calibrated] {
            assert_eq!(SelectorRank::parse(f.as_str()), Ok(f));
        }
    }

    /// A lattice with one confident path: both forms pick it.
    #[test]
    fn a_confident_chain_is_both_forms_chain() {
        // candidate 0 at depth 0, then 2, then 1 -- each a clear winner of its row
        let want = [0usize, 2, 1];
        let s = lattice(|d, p, c| {
            let prev_ok = d == 0 || p == want[d - 1];
            if prev_ok && c == want[d] {
                10.0
            } else {
                0.0
            }
        });
        assert_eq!(walk_path(&s, ST, K), want.to_vec());
        assert_eq!(explen_path(&s, ST, K, EXPLEN_TEMPERATURE), want.to_vec());
    }

    /// THE CASE THE RE-RANK EXISTS FOR (and the control: the walk takes the other branch). At depth
    /// 0, candidate 0 leads candidate 1 by a hair, but every row below 0 is flat (the drafter does
    /// not know what follows it) while 1's rows are confident: the expected length prefers 1.
    #[test]
    fn a_near_tie_goes_to_the_branch_with_the_longer_expected_chain() {
        let s = lattice(|d, p, c| match d {
            0 => [1.05, 1.0, -5.0, -5.0][c],
            _ if p == 1 && c == 1 => 12.0,
            _ => 0.0, // flat after candidate 0 (and after anything else)
        });
        assert_eq!(
            walk_path(&s, ST, K),
            vec![0, 0, 0],
            "the control: the walk takes the hair"
        );
        assert_eq!(explen_path(&s, ST, K, EXPLEN_TEMPERATURE), vec![1, 1, 1]);
    }

    /// Ties go to the lowest index, as the walk's.
    #[test]
    fn a_flat_lattice_picks_index_zero() {
        let s = lattice(|_, _, _| 0.0);
        assert_eq!(walk_path(&s, ST, K), vec![0, 0, 0]);
        assert_eq!(explen_path(&s, ST, K, EXPLEN_TEMPERATURE), vec![0, 0, 0]);
    }

    #[test]
    fn a_short_lattice_yields_nothing() {
        assert!(explen_path(&[0.0; 5], ST, K, 2.0).is_empty());
    }
}
