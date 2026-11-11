//! DFlash2 block speculative decoding — the **semantics**, as pure host logic.
//!
//! Nothing here launches a kernel, allocates a buffer or holds a model. It is the set of
//! decisions an engine step makes around a block of draft tokens, ported so they can be tested
//! on the host and so `saber-sched` and `saber-exec` have one place to agree with.
//!
//! Four rules, each with a named reference and each gated against that reference's own output
//! (`the internal reference harness`, `tests` at the bottom of this file):
//!
//! | rule | reference | what this module provides |
//! |---|---|---|
//! | Which draft tokens survive, and what the one extra token is | `upstream/v1/worker/gpu/spec_decode/rejection_sampler_utils.py::_rejection_kernel` + `_insert_resampled_kernel` | [`verify`] |
//! | The scheduler-side verify-width cap | `ref/patch_dynwidth.py` | [`WidthPolicy`], [`AcceptanceEma`] |
//! | The draft-depth ceiling per batch size | `ref/ref_draft.py::_batch_ceil` | [`DraftSchedule`] |
//! | The longest-suffix n-gram continuation | `ref/ref_draft_gpu.py::match_gpu` | [`NgramMatcher`] |
//!
//! plus the `saber:spec_decode_*` counter definitions ([`SpecDecodeCounters`]), which are the
//! numerators `eval_drafter.py` divides and therefore have to mean exactly what prod's mean.
//!
//! # The one sentence the whole module exists to make unambiguous
//!
//! **A verified step commits `accepted_len + 1` tokens, and `accepted_len` counts draft tokens
//! only.** The extra token is never an accepted draft token: it is either the target's own
//! token at the position where the draft was rejected, or — when every draft token survived —
//! the bonus token from the extra logits row the step always computes. Counting it as accepted
//! inflates acceptance-per-draft by exactly 1.0, which is a ~45% overstatement at prod's
//! measured 2.25 (`saber:spec_decode_num_accepted_tokens_total / num_drafts_total`, 2026-09-18).
//!
//! # What is deliberately NOT here
//!
//! * **The proposal.** Which tokens a drafter emits — the DFlash2 block-diffusion pass, the
//!   candidate selector, the selector walk — is a model question and lives in the drafter.
//! * **The replacement token's value.** [`verify`] takes it as an input. Computing it is a
//!   sampler's job: greedy takes the target's argmax, a sampled request draws from the
//!   residual distribution with Gumbel noise keyed on `(seed, position)`.
//! * **Anything about `SABER_DYNAMIC_DRAFT`'s per-slot confidence gate being live.** It is
//!   not, under the unit prod serves — see `docs/abi/DFLASH2-SEMANTICS.md` § "What is inert".
//!   The rules are ported because the controller is the obvious thing to reach for next, and a
//!   port with a gate behind it is worth more than a rediscovery.

use std::collections::BTreeMap;
use std::fmt;

/// A token id. `-1` is the padding a drafter writes into a block position it did not fill, and
/// the verify rule rejects it (`is_valid_draft = draft_sampled >= 0`).
pub type TokenId = i32;

/// The padding value a short draft block carries.
pub const PAD_TOKEN: TokenId = -1;

/// Errors from the draft plane. All of them are shape disagreements, which under a captured
/// graph are the ones that corrupt a stream rather than crash it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DraftError {
    /// `decisions.len()` did not equal the block width.
    #[error("verify: {width} draft tokens but {decisions} target decisions")]
    WidthMismatch {
        /// Draft tokens in the block.
        width: usize,
        /// Target decisions supplied.
        decisions: usize,
    },
    /// A draft schedule string that could not be parsed.
    #[error("draft schedule {spec:?}: {why}")]
    BadSchedule {
        /// The string.
        spec: String,
        /// What was wrong with it.
        why: String,
    },
}

// =====================================================================================
// The draft block
// =====================================================================================

/// A block of draft tokens proposed for one sequence in one step.
///
/// DFlash2 is a **block** drafter: one backbone pass proposes every position at once, so the
/// width is fixed when the graph is captured and the per-step lever is how many of those
/// positions the target is asked to *verify* (`docs/LEDGER.md`, "Dynamic verify width"). The
/// block is therefore allowed to be shorter than the drafter's own width — that is what the
/// scheduler's cap produces — but never longer than the captured shape.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftBlock {
    /// The proposed tokens, in position order. May contain [`PAD_TOKEN`].
    pub tokens: Vec<TokenId>,
}

impl DraftBlock {
    /// A block from proposed tokens.
    #[must_use]
    pub fn new(tokens: impl IntoIterator<Item = TokenId>) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
        }
    }

    /// The empty block: no speculation, one target token per step.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Draft tokens in the block — the `saber:spec_decode_num_draft_tokens_total` contribution
    /// of one draft, padding included (prod counts `len(scheduled_spec_token_ids)`).
    #[must_use]
    pub fn width(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Whether this block proposes nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Truncate to `width` positions. This is what the scheduler's cap does to
    /// `request.spec_token_ids` (`del request.spec_token_ids[w:]`), and it only ever shortens.
    pub fn truncate(&mut self, width: u32) {
        let w = width as usize;
        if w < self.tokens.len() {
            self.tokens.truncate(w);
        }
    }
}

// =====================================================================================
// Verify
// =====================================================================================

/// What the target says about one draft position.
///
/// The two arms are the two arms of `_rejection_kernel`, chosen by `is_greedy = temp == 0.0`.
/// Both carry only what the *rule* reads; the replacement token's value is supplied separately
/// to [`verify`] because it is a sampler output, not a decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetDecision {
    /// `temperature == 0`. Accept iff the draft token equals the target's argmax.
    ///
    /// The kernel writes `draft_sampled if accepted else target_argmax`, so the argmax is both
    /// the test and the replacement — which is why greedy needs no resample pass at a rejected
    /// position and `_insert_resampled_kernel` returns early for it.
    Greedy {
        /// The target's argmax at this position.
        argmax: TokenId,
    },
    /// `temperature > 0`. Leviathan et al.'s ratio test, in the log form the kernel uses:
    /// `log p(x) > log(u) + log q(x)`.
    Sampled {
        /// `log p(draft_token)` under the target's **processed** logits (after temperature,
        /// top-k and friends — `apply_sampling_params` runs before the kernel).
        target_logprob: f32,
        /// `log q(draft_token)` under the drafter's distribution. Exactly `0.0` for a one-hot
        /// draft: with no cached draft logits the kernel hard-codes `log q = 1`'s logarithm.
        draft_logprob: f32,
        /// The uniform draw, `tl_rand32(seed, position, includes_zero=False)` — in `(0, 1]`,
        /// keyed on the request's seed and the token's absolute position so that a draft and
        /// its verification see the same noise.
        u: f32,
    },
}

impl TargetDecision {
    /// Whether this position accepts `draft`.
    ///
    /// A [`PAD_TOKEN`] never accepts: the sampled arm tests `draft >= 0` explicitly, and the
    /// greedy arm gets it for free because an argmax is never negative.
    #[must_use]
    pub fn accepts(&self, draft: TokenId) -> bool {
        match *self {
            TargetDecision::Greedy { argmax } => argmax == draft,
            TargetDecision::Sampled {
                target_logprob,
                draft_logprob,
                u,
            } => {
                draft >= 0
                    && f64::from(target_logprob) > f64::from(u).ln() + f64::from(draft_logprob)
            }
        }
    }
}

/// What a verified step commits for one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    /// Draft tokens accepted, as a **prefix**: acceptance stops at the first rejection and
    /// never resumes (`accepted &= ...` in the kernel, inside an `elif accepted:`).
    pub accepted_len: u32,
    /// The tokens the step commits: the accepted prefix plus exactly one more.
    /// `tokens.len() == accepted_len + 1`, always.
    pub tokens: Vec<TokenId>,
    /// The position that was rejected, when one was. `None` means every draft token survived
    /// and the extra token is the bonus.
    pub rejected_at: Option<u32>,
}

impl Verified {
    /// The extra token — the recovery token, or the bonus when nothing was rejected.
    #[must_use]
    pub fn extra_token(&self) -> TokenId {
        *self
            .tokens
            .last()
            .expect("a verified step always commits one token")
    }

    /// Whether the extra token is the bonus (every draft token accepted).
    #[must_use]
    pub fn is_bonus(&self) -> bool {
        self.rejected_at.is_none()
    }

    /// Draft tokens rejected: `width - accepted_len`. The scheduler rolls `num_computed_tokens`
    /// back by this much.
    #[must_use]
    pub fn rejected(&self, width: u32) -> u32 {
        width.saturating_sub(self.accepted_len)
    }
}

/// Verify a draft block against the target's per-position decisions.
///
/// The rule, transcribed from `_rejection_kernel` and `_insert_resampled_kernel`:
///
/// 1. Walk the block from position 0. `accepted` starts true and is **ANDed** at every step, so
///    the first rejection ends the walk — a later position that would have matched is not
///    reconsidered.
/// 2. `accepted_len` is how many positions accepted.
/// 3. The step commits `accepted_len + 1` tokens. `_insert_resampled_kernel` increments
///    `num_sampled` unconditionally; there is no path on which a verified step commits only the
///    accepted prefix.
/// 4. The extra token is `replacement`: the target's argmax at the rejected position (greedy),
///    a residual-distribution draw there (sampled), or the **bonus** token from the extra
///    logits row when `accepted_len == width`.
///
/// `replacement` is the caller's because it is a sampler output. Passing the wrong one cannot
/// change `accepted_len`; it changes which token the stream continues with, which is what the
/// multi-turn gate is for.
pub fn verify(
    block: &DraftBlock,
    decisions: &[TargetDecision],
    replacement: TokenId,
) -> Result<Verified, DraftError> {
    if decisions.len() != block.tokens.len() {
        return Err(DraftError::WidthMismatch {
            width: block.tokens.len(),
            decisions: decisions.len(),
        });
    }
    let mut accepted_len = 0u32;
    let mut tokens = Vec::with_capacity(block.tokens.len() + 1);
    let mut rejected_at = None;
    for (i, (&draft, decision)) in block.tokens.iter().zip(decisions).enumerate() {
        if !decision.accepts(draft) {
            rejected_at = Some(i as u32);
            break;
        }
        tokens.push(draft);
        accepted_len += 1;
    }
    tokens.push(replacement);
    Ok(Verified {
        accepted_len,
        tokens,
        rejected_at,
    })
}

// =====================================================================================
// Dynamic verify width
// =====================================================================================

/// The scheduler-side verify-width policy, as data.
///
/// Every field is an environment knob `patch_dynwidth.py` reads once at import; [`WidthPolicy::PROD`]
/// carries the values the `qwen_prod_nvfp4` unit runs. Under a DFlash2 drafter the *draft* cost
/// is fixed — the block-diffusion pass emits all 8 positions whether or not they are used — so
/// the only lever is how many of them the target is asked to verify, and the right number is
/// content-dependent with a 2× spread (`docs/LEDGER.md`, "Dynamic verify width": conc-8
/// +11–13% with single-stream unchanged).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WidthPolicy {
    /// `SABER_DYNW_ALPHA`: weight on the newest observation.
    pub alpha: f64,
    /// `SABER_DYNW_MARGIN`: rows above the rounded-up EMA.
    pub margin: u32,
    /// `SABER_DYNW_MIN`: the cap never goes below this.
    pub min_width: u32,
    /// `SABER_DYNW_MIN_BATCH`: the cap applies only at this many running requests or more.
    ///
    /// Below it the batch's M sits in the weight-stream-bound flat zone where narrowing saves
    /// nothing and can only throw away a token that would have been accepted — the measured
    /// conc-2 dip. The **observation** is never gated: history stays warm at every concurrency
    /// so a cap applies the instant a batch forms.
    pub min_batch: u32,
}

impl WidthPolicy {
    /// The policy the `qwen_prod_nvfp4` unit runs (`SABER_DYNAMIC_WIDTH=1`).
    pub const PROD: Self = Self {
        alpha: 0.35,
        margin: 2,
        min_width: 2,
        min_batch: 3,
    };

    /// The cap for a sequence, or `None` when the policy does not bind this step.
    ///
    /// `None` is not "no cap": it is "leave `spec_token_ids` exactly as it is", which is what
    /// `_ref_cap_spec_width`'s three early returns do — no drafts, too small a batch, or
    /// no EMA yet (cold start verifies full width until there is evidence).
    #[must_use]
    pub fn cap(&self, ema: &AcceptanceEma, running: u32, spec_len: u32) -> Option<u32> {
        if spec_len == 0 || running < self.min_batch {
            return None;
        }
        let value = ema.value()?;
        let w = ceil_u32(value) + self.margin;
        let w = w.max(self.min_width);
        (w < spec_len).then_some(w)
    }

    /// Apply the cap to a block in place, returning the width it now has.
    pub fn apply(&self, block: &mut DraftBlock, ema: &AcceptanceEma, running: u32) -> u32 {
        if let Some(w) = self.cap(ema, running, block.width()) {
            block.truncate(w);
        }
        block.width()
    }
}

impl Default for WidthPolicy {
    fn default() -> Self {
        Self::PROD
    }
}

/// `ceil` of a non-negative float into a `u32`, saturating. `math.ceil` in the reference.
fn ceil_u32(v: f64) -> u32 {
    if !v.is_finite() || v <= 0.0 {
        return 0;
    }
    let c = v.ceil();
    if c >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        c as u32
    }
}

/// One sequence's acceptance EMA, in accepted **draft** tokens per draft.
///
/// Fresh sequences have no value at all rather than a prior: `_ref_dynw_observe` seeds the
/// EMA with the first observation (`obs if ema is None else ...`), and until then the cap does
/// not bind.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AcceptanceEma {
    value: Option<f64>,
    samples: u64,
}

impl AcceptanceEma {
    /// A sequence with no history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold in one verified step.
    ///
    /// Two details that are the policy rather than bookkeeping:
    ///
    /// * **A step that drafted nothing is not an observation.** `num_draft_tokens <= 0` returns
    ///   early; a non-drafting step says nothing about the drafter and must not pull the
    ///   average to zero.
    /// * **Full acceptance observes one above the width.** `obs = accepted + (accepted >= drafted)`.
    ///   Without it a cap at 3 could never see evidence for 5: the EMA would be stuck under its
    ///   own shadow, because the only widths it ever observes are the ones it already allows.
    pub fn observe(&mut self, policy: &WidthPolicy, accepted: u32, drafted: u32) {
        if drafted == 0 {
            return;
        }
        let obs = f64::from(accepted) + f64::from(u8::from(accepted >= drafted));
        self.value = Some(match self.value {
            None => obs,
            Some(prev) => policy.alpha * obs + (1.0 - policy.alpha) * prev,
        });
        self.samples += 1;
    }

    /// The current estimate, or `None` before the first drafting step.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Drafting steps folded in.
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.samples
    }
}

// =====================================================================================
// Draft-depth schedule
// =====================================================================================

/// `SABER_DRAFT_SCHEDULE`, the batch-size ceiling on **serial draft forwards**.
///
/// `"bs:max_depth,..."`, carry-forward: the ceiling for a batch is the depth of the largest
/// listed batch size at or below it. It caps how deep the drafter forwards as concurrency
/// rises, so deep serial drafts do not run at batch 8; it does **not** cap the verify width,
/// and under the controller it does not cap a free verbatim n-gram tail either.
///
/// This is the knob whose value the prod boot log prints
/// (`1:8,2:7,4:6,8:5,16:4`) — and which, on the unit prod actually serves, nothing reads. See
/// `docs/abi/DFLASH2-SEMANTICS.md` § "What is inert".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftSchedule {
    /// `(batch_size, max_depth)` pairs, ascending. Empty means "no ceiling".
    entries: Vec<(u32, u32)>,
}

/// The schedule string the `qwen_prod_nvfp4` unit carries.
pub const PROD_DRAFT_SCHEDULE: &str = "1:8,2:7,4:6,8:5,16:4";

/// The ceiling a schedule with no entries reports — `1 << 30` in the reference, i.e. "none".
pub const NO_FORWARD_CEILING: u32 = 1 << 30;

impl DraftSchedule {
    /// Parse `"bs:depth,bs:depth,..."`. An empty string is a schedule with no entries, which
    /// disables the ceiling; whitespace around an entry is ignored, as in the reference.
    pub fn parse(spec: &str) -> Result<Self, DraftError> {
        let mut entries = Vec::new();
        for tok in spec.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let (bs, depth) = tok.split_once(':').ok_or_else(|| DraftError::BadSchedule {
                spec: spec.to_string(),
                why: format!("entry {tok:?} is not `batch:depth`"),
            })?;
            let parse = |s: &str, what: &str| -> Result<u32, DraftError> {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| DraftError::BadSchedule {
                        spec: spec.to_string(),
                        why: format!("{what} in {tok:?}: {e}"),
                    })
            };
            entries.push((parse(bs, "batch size")?, parse(depth, "depth")?));
        }
        entries.sort_unstable();
        Ok(Self { entries })
    }

    /// The schedule prod ships.
    #[must_use]
    pub fn prod() -> Self {
        Self::parse(PROD_DRAFT_SCHEDULE).expect("the shipped schedule parses")
    }

    /// The `(batch_size, depth)` pairs, ascending.
    #[must_use]
    pub fn entries(&self) -> &[(u32, u32)] {
        &self.entries
    }

    /// The maximum serial draft-forward depth at `num_reqs` running requests.
    ///
    /// Carry-forward, exactly as `_batch_ceil` walks it: start at the **first** entry's depth,
    /// take each entry whose batch size `num_reqs` reaches, and stop at the first one it does
    /// not. So a batch smaller than every listed size still gets the first entry's depth — at
    /// `num_reqs = 0` the prod schedule reports 8, not "no drafting".
    #[must_use]
    pub fn forward_ceiling(&self, num_reqs: u32) -> u32 {
        let Some(&(_, first)) = self.entries.first() else {
            return NO_FORWARD_CEILING;
        };
        let mut depth = first;
        for &(bs, v) in &self.entries {
            if num_reqs >= bs {
                depth = v;
            } else {
                break;
            }
        }
        depth.max(1)
    }
}

// =====================================================================================
// The n-gram matcher
// =====================================================================================

/// The longest-suffix n-gram continuation matcher.
///
/// Given a context, find the **longest** earlier occurrence of the context's own current suffix
/// and return what followed it. A hit is a free draft: the tokens cost no forward pass, and the
/// controller only takes one when its next token agrees with the drafter's own top guess.
///
/// Reference: `ref_draft_gpu.py::match_gpu`, two kernels — a scan that packs
/// `(match_len << 20) | end_pos` into an `atomic_max`, and a gather that copies the
/// continuation. Three consequences of that packing, all of them load-bearing:
///
/// * **Longest wins, and among equal lengths the most recent** — `match_len` occupies the high
///   bits, so `max` orders by length first and by end position second.
/// * **Matches shorter than [`Self::min_len`] are discarded** before the max, so a 2-token
///   coincidence never produces a continuation.
/// * **The match length saturates at [`Self::max_len`]** (24): the scan runs a fixed number of
///   comparisons. A longer repeat is found, it is simply not scored any higher.
///
/// The end position must be strictly before the last token (`q < n - 1`), so a suffix cannot
/// match itself and every hit has at least one continuation token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramMatcher {
    /// `_MAXL`: comparisons per candidate, i.e. the longest match length that can be scored.
    pub max_len: u32,
    /// `_MIN`: shorter matches are not candidates at all.
    pub min_len: u32,
}

impl Default for NgramMatcher {
    fn default() -> Self {
        Self::PROD
    }
}

/// A continuation found in the context.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NgramMatch {
    /// The continuation tokens, at most `width` of them. Shorter when the match ran into the
    /// end of the context.
    pub tokens: Vec<TokenId>,
    /// Where the match ended in the context, when there was one.
    pub end_pos: Option<u32>,
    /// How long the match was, saturating at [`NgramMatcher::max_len`].
    pub match_len: u32,
}

impl NgramMatch {
    /// Continuation tokens available — the reference's `clen`.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Whether nothing was found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl NgramMatcher {
    /// The shipped constants: `_MAXL = 24`, `_MIN = 3`.
    pub const PROD: Self = Self {
        max_len: 24,
        min_len: 3,
    };

    /// Find the continuation of the longest earlier occurrence of `ctx`'s suffix.
    ///
    /// `width` is how many continuation tokens are wanted. Returns an empty match when the
    /// context is shorter than 3 tokens — the caller (`_prepare_match_gpu`) does not even
    /// launch the matcher then, which leaves the controller with no continuation, the same
    /// observable state as a match of length zero.
    ///
    /// The Triton gather requires `width` to be a power of two (`tl.arange(0, NSPEC)`); that is
    /// a property of the kernel, not of the rule, and this port has no such restriction.
    #[must_use]
    pub fn find(&self, ctx: &[TokenId], width: u32) -> NgramMatch {
        let n = ctx.len();
        if n < 3 || width == 0 {
            return NgramMatch::default();
        }
        let mut best: Option<(u32, usize)> = None; // (match_len, end_pos)
        for q in 0..n - 1 {
            let mut len = 0u32;
            for k in 0..self.max_len as usize {
                let (Some(sk), Some(qi)) = (n.checked_sub(1 + k), q.checked_sub(k)) else {
                    break;
                };
                if ctx[qi] != ctx[sk] {
                    break;
                }
                len += 1;
            }
            if len < self.min_len {
                continue;
            }
            // `(len << 20) | q` under an atomic max: longer first, then the larger end position.
            if best.is_none_or(|(bl, bq)| (len, q) > (bl, bq)) {
                best = Some((len, q));
            }
        }
        let Some((match_len, end_pos)) = best else {
            return NgramMatch::default();
        };
        let start = end_pos + 1;
        let take = (n - start).min(width as usize);
        NgramMatch {
            tokens: ctx[start..start + take].to_vec(),
            end_pos: Some(end_pos as u32),
            match_len,
        }
    }
}

// =====================================================================================
// The per-slot draft controller (SABER_DYNAMIC_DRAFT)
// =====================================================================================

/// `SABER_DRAFT_TAU`, the confidence-product stop threshold.
///
/// The unit prod serves carries **0.20**. Two other values are in circulation and both are
/// wrong for it: `ref_draft.py`'s code default is `0.28` and its own module docstring
/// says `0.35`. The three are reconciled in `docs/abi/DFLASH2-SEMANTICS.md`.
pub const PROD_DRAFT_TAU: f32 = 0.20;

/// What the controller decides at one draft slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotAction {
    /// Run another drafter forward. The reference's `0`.
    Continue,
    /// Stop forwarding and take the verbatim n-gram continuation from here. The reference's `1`.
    TakeNgram,
    /// Stop forwarding and verify what is drafted. The reference's `2`.
    Verify,
}

/// One sequence's state inside the per-slot draft controller.
///
/// Reference: the `greedy_sample` closure in `ref_draft.py`. The per-slot rule is
/// evaluated *after* slot `j`'s forward, using the running product of the top-1 confidences
/// from slots `0..j` — the product is multiplied by slot `j`'s confidence **after** the
/// decision, so slot 0 always decides with `cum == 1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftGate {
    cum: f32,
    stopped: bool,
    all_agree: bool,
    stop_slot: u32,
    stop_action: SlotAction,
}

impl DraftGate {
    /// A gate for a sequence about to draft `width` positions.
    #[must_use]
    pub fn new(width: u32) -> Self {
        Self {
            cum: 1.0,
            stopped: false,
            all_agree: true,
            stop_slot: width,
            stop_action: SlotAction::Verify,
        }
    }

    /// Evaluate slot `j`.
    ///
    /// `agrees` is whether the n-gram continuation has a token at `j` and it equals the token
    /// the drafter just produced — a free win, taken immediately. `confidence` is the drafter's
    /// top-1 softmax probability at this slot, folded into the running product afterwards.
    pub fn slot(&mut self, tau: f32, j: u32, agrees: bool, confidence: f32) -> SlotAction {
        let action = if agrees {
            SlotAction::TakeNgram
        } else if self.cum >= tau {
            SlotAction::Continue
        } else {
            SlotAction::Verify
        };
        if !self.stopped && action != SlotAction::Continue {
            self.stop_slot = j;
            self.stop_action = action;
        }
        self.stopped |= action != SlotAction::Continue;
        self.cum *= confidence;
        self.all_agree &= agrees || self.stopped;
        action
    }

    /// Apply the forward-count ceiling after slot `j`.
    ///
    /// When the ceiling is reached, sequences that have not stopped do: their draft ends at
    /// `j + 1`. One that has matched the n-gram the whole way keeps drafting **for free** off
    /// the verbatim tail — only the forwards are capped, never the tail.
    pub fn forward_cap(&mut self, j: u32, ceiling: u32) {
        if ceiling == 0 || j + 1 < ceiling || self.stopped {
            return;
        }
        self.stop_slot = j + 1;
        if self.all_agree {
            self.stop_action = SlotAction::TakeNgram;
        }
        self.stopped = true;
    }

    /// Whether this sequence has stopped drafting.
    #[must_use]
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// The slot the sequence stopped at, and what it chose.
    #[must_use]
    pub fn stop(&self) -> (u32, SlotAction) {
        (self.stop_slot, self.stop_action)
    }

    /// Assemble the final block: the drafter's tokens up to the stop slot, plus the verbatim
    /// n-gram tail when that is what was chosen, clipped to `width`.
    ///
    /// Reference: `_postprocess_gpu`. The last rule is the odd one and it is deliberate — a
    /// sequence that would otherwise emit nothing keeps the drafter's first token, because a
    /// zero-width block is not a draft at all and would lose the step's speculation entirely.
    #[must_use]
    pub fn assemble(&self, drafted: &[TokenId], ngram: &NgramMatch, width: u32) -> DraftBlock {
        let k = (self.stop_slot as usize).min(drafted.len());
        let mut out: Vec<TokenId> = drafted[..k].to_vec();
        if self.stop_action == SlotAction::TakeNgram && k < ngram.tokens.len() {
            out.extend_from_slice(&ngram.tokens[k..]);
        }
        if out.is_empty() {
            if let Some(&first) = drafted.first() {
                out.push(first);
            }
        }
        out.truncate(width as usize);
        DraftBlock::new(out)
    }
}

// =====================================================================================
// Counters
// =====================================================================================

/// The `saber:spec_decode_*` counters, with prod's definitions.
///
/// Measured on the live unit 2026-09-18 (`curl localhost:8080/metrics`): 4,533 drafts,
/// 31,731 draft tokens, 10,189 accepted tokens, and `accepted_per_pos` summing to exactly the
/// accepted total over 7 positions. Two identities fall out and both are asserted in the tests:
///
/// * `draft_tokens == sum of block widths` — 31,731 / 4,533 = exactly 7, the full `SPEC=7`
///   width, because that traffic ran at concurrency 1 and `min_batch = 3` kept the width cap
///   from binding.
/// * `sum(accepted_per_pos) == accepted_tokens`, over `width` positions and no more. **The
///   bonus token is in none of them.**
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecDecodeCounters {
    /// `saber:spec_decode_num_drafts_total`: verified draft blocks. One per sequence per step
    /// that drafted anything.
    pub num_drafts: u64,
    /// `saber:spec_decode_num_draft_tokens_total`: draft tokens proposed, i.e. the sum of the
    /// **scheduled** block widths (after the verify-width cap, not the drafter's own width).
    pub num_draft_tokens: u64,
    /// `saber:spec_decode_num_accepted_tokens_total`: accepted **draft** tokens. Never the bonus.
    pub num_accepted_tokens: u64,
    /// `saber:spec_decode_num_accepted_tokens_per_pos_total[i]`: drafts whose accepted prefix
    /// reached past position `i`.
    pub accepted_per_pos: Vec<u64>,
}

impl SpecDecodeCounters {
    /// Fold in one verified step. A step that drafted nothing is not a draft and is not counted
    /// — prod's `if scheduled_spec_token_ids and ...` guard.
    pub fn observe(&mut self, width: u32, verified: &Verified) {
        if width == 0 {
            return;
        }
        self.num_drafts += 1;
        self.num_draft_tokens += u64::from(width);
        self.num_accepted_tokens += u64::from(verified.accepted_len);
        if self.accepted_per_pos.len() < width as usize {
            self.accepted_per_pos.resize(width as usize, 0);
        }
        for slot in self
            .accepted_per_pos
            .iter_mut()
            .take(verified.accepted_len as usize)
        {
            *slot += 1;
        }
    }

    /// Accepted draft tokens per draft — what `eval_drafter.py` reports and what a drafter is
    /// judged on. **Not** tokens per step: that is this plus one.
    #[must_use]
    pub fn acceptance_per_draft(&self) -> f64 {
        if self.num_drafts == 0 {
            return 0.0;
        }
        self.num_accepted_tokens as f64 / self.num_drafts as f64
    }

    /// Tokens committed per drafting step: [`Self::acceptance_per_draft`] plus the one token
    /// every verified step commits.
    #[must_use]
    pub fn tokens_per_step(&self) -> f64 {
        if self.num_drafts == 0 {
            return 0.0;
        }
        self.acceptance_per_draft() + 1.0
    }

    /// The counters as the metric names that carry them, for a `/metrics` handler.
    #[must_use]
    pub fn as_metrics(&self) -> BTreeMap<&'static str, u64> {
        BTreeMap::from([
            ("saber:spec_decode_num_drafts_total", self.num_drafts),
            (
                "saber:spec_decode_num_draft_tokens_total",
                self.num_draft_tokens,
            ),
            (
                "saber:spec_decode_num_accepted_tokens_total",
                self.num_accepted_tokens,
            ),
        ])
    }
}

impl fmt::Display for SpecDecodeCounters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "drafts {} draft_tokens {} accepted {} ({:.3}/draft, {:.3} tokens/step)",
            self.num_drafts,
            self.num_draft_tokens,
            self.num_accepted_tokens,
            self.acceptance_per_draft(),
            self.tokens_per_step()
        )
    }
}

// =====================================================================================
// Tests
// =====================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    // ---------------------------------------------------------------------------------
    // Unit facts
    // ---------------------------------------------------------------------------------

    #[test]
    fn a_verified_step_always_commits_one_more_than_it_accepts() {
        let block = DraftBlock::new([10, 11, 12]);
        let all = [
            TargetDecision::Greedy { argmax: 10 },
            TargetDecision::Greedy { argmax: 11 },
            TargetDecision::Greedy { argmax: 12 },
        ];
        let v = verify(&block, &all, 99).unwrap();
        assert_eq!(v.accepted_len, 3);
        assert_eq!(v.tokens, vec![10, 11, 12, 99]);
        assert!(v.is_bonus());
        assert_eq!(v.rejected(3), 0);

        let mid = [
            TargetDecision::Greedy { argmax: 10 },
            TargetDecision::Greedy { argmax: 77 },
            TargetDecision::Greedy { argmax: 12 },
        ];
        let v = verify(&block, &mid, 77).unwrap();
        assert_eq!(v.accepted_len, 1);
        assert_eq!(v.tokens, vec![10, 77]);
        assert_eq!(v.rejected_at, Some(1));
        assert!(!v.is_bonus());
        assert_eq!(v.rejected(3), 2);
    }

    #[test]
    fn acceptance_is_a_prefix_and_never_resumes() {
        // Position 2 would have matched; the walk ended at position 1 and does not look.
        let block = DraftBlock::new([1, 2, 3]);
        let d = [
            TargetDecision::Greedy { argmax: 1 },
            TargetDecision::Greedy { argmax: 9 },
            TargetDecision::Greedy { argmax: 3 },
        ];
        let v = verify(&block, &d, 9).unwrap();
        assert_eq!(v.accepted_len, 1);
    }

    #[test]
    fn a_pad_token_never_accepts() {
        for decision in [
            TargetDecision::Greedy { argmax: 5 },
            TargetDecision::Sampled {
                target_logprob: 0.0,
                draft_logprob: -50.0,
                u: 1e-6,
            },
        ] {
            assert!(!decision.accepts(PAD_TOKEN), "{decision:?}");
        }
    }

    #[test]
    fn an_empty_block_commits_exactly_the_bonus() {
        let v = verify(&DraftBlock::none(), &[], 42).unwrap();
        assert_eq!(v.accepted_len, 0);
        assert_eq!(v.tokens, vec![42]);
        assert!(v.is_bonus());
    }

    #[test]
    fn a_width_mismatch_is_an_error_not_a_truncation() {
        let err = verify(
            &DraftBlock::new([1, 2, 3]),
            &[TargetDecision::Greedy { argmax: 1 }],
            0,
        )
        .unwrap_err();
        assert_eq!(
            err,
            DraftError::WidthMismatch {
                width: 3,
                decisions: 1
            }
        );
    }

    #[test]
    fn the_prod_schedule_carries_forward() {
        let s = DraftSchedule::prod();
        assert_eq!(s.entries(), &[(1, 8), (2, 7), (4, 6), (8, 5), (16, 4)]);
        for (batch, want) in [
            (0, 8), // below every listed size: the FIRST entry's depth, not "no drafting"
            (1, 8),
            (2, 7),
            (3, 7),
            (4, 6),
            (7, 6),
            (8, 5),
            (15, 5),
            (16, 4),
            (64, 4),
        ] {
            assert_eq!(s.forward_ceiling(batch), want, "batch {batch}");
        }
        assert_eq!(
            DraftSchedule::parse("").unwrap().forward_ceiling(4),
            NO_FORWARD_CEILING
        );
    }

    #[test]
    fn a_malformed_schedule_is_refused() {
        for spec in ["1", "1:x", "a:2", "1:2:3"] {
            assert!(
                DraftSchedule::parse(spec).is_err(),
                "{spec:?} should not parse"
            );
        }
    }

    #[test]
    fn the_width_cap_does_not_bind_below_the_batch_floor() {
        let p = WidthPolicy::PROD;
        let mut ema = AcceptanceEma::new();
        for _ in 0..20 {
            ema.observe(&p, 0, 7);
        }
        // EMA ~ 0, so the cap would be max(0 + 2, 2) = 2 -- but only from 3 running requests.
        assert_eq!(p.cap(&ema, 2, 7), None, "below min_batch");
        assert_eq!(p.cap(&ema, 3, 7), Some(2));
        // Cold start: no EMA, no cap, whatever the batch.
        assert_eq!(p.cap(&AcceptanceEma::new(), 8, 7), None);
        // A block already narrower than the cap is left alone.
        assert_eq!(p.cap(&ema, 8, 2), None);
    }

    #[test]
    fn full_acceptance_lets_the_ema_climb_out_of_its_own_cap() {
        let p = WidthPolicy::PROD;
        let mut capped = AcceptanceEma::new();
        let mut plain = AcceptanceEma::new();
        for _ in 0..30 {
            capped.observe(&p, 3, 3); // accepted everything it was allowed
            plain.observe(&p, 3, 7); // accepted 3 of 7
        }
        assert!((capped.value().unwrap() - 4.0).abs() < 1e-9);
        assert!((plain.value().unwrap() - 3.0).abs() < 1e-9);
        assert_eq!(p.cap(&capped, 8, 7), Some(6));
        assert_eq!(p.cap(&plain, 8, 7), Some(5));
    }

    #[test]
    fn a_step_that_drafted_nothing_is_not_an_observation() {
        let p = WidthPolicy::PROD;
        let mut ema = AcceptanceEma::new();
        ema.observe(&p, 3, 5);
        let before = ema;
        ema.observe(&p, 0, 0);
        assert_eq!(ema, before);
        assert_eq!(ema.samples(), 1);
    }

    #[test]
    fn the_matcher_prefers_the_longest_then_the_most_recent() {
        let m = NgramMatcher::PROD;
        // "5 7 9" repeats; the suffix is 5 7 9 and the longest earlier run ends at index 5.
        let hit = m.find(&[5, 7, 9, 5, 7, 9, 5, 7, 9], 4);
        assert_eq!(hit.tokens, vec![5, 7, 9]);
        assert_eq!(hit.end_pos, Some(5));
        assert_eq!(hit.match_len, 6);

        // Two matches of equal length: the later one wins.
        let hit = m.find(&[1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 2, 3], 2);
        assert_eq!(hit.end_pos, Some(6));

        // Shorter than min_len is not a match at all.
        assert!(m.find(&[4, 1, 2, 9, 9, 1, 2], 3).is_empty());
        // A context under three tokens never matches.
        assert!(m.find(&[7, 7], 4).is_empty());
        assert!(m.find(&[], 4).is_empty());
    }

    #[test]
    fn the_match_length_saturates_but_the_match_still_wins() {
        let mut m = NgramMatcher::PROD;
        m.max_len = 4;
        let ctx: Vec<TokenId> = (0..10).chain(0..10).collect();
        let hit = m.find(&ctx, 3);
        assert_eq!(
            hit.match_len, 4,
            "capped at max_len, though the true run is 10"
        );
        assert_eq!(hit.end_pos, Some(9), "the first copy's last token");
        assert_eq!(
            hit.tokens,
            vec![0, 1, 2],
            "the continuation is the second copy"
        );
    }

    #[test]
    fn the_gate_takes_a_free_ngram_and_otherwise_drafts_while_confidence_holds() {
        let tau = PROD_DRAFT_TAU;
        let mut g = DraftGate::new(8);
        // Slot 0 always decides with cum == 1.0: continue unless the n-gram agrees.
        assert_eq!(g.slot(tau, 0, false, 0.9), SlotAction::Continue);
        assert_eq!(g.slot(tau, 1, false, 0.5), SlotAction::Continue); // cum 0.90 >= 0.20
        assert_eq!(g.slot(tau, 2, false, 0.3), SlotAction::Continue); // cum 0.45 >= 0.20
        assert_eq!(g.slot(tau, 3, false, 0.9), SlotAction::Verify); // cum 0.135 < 0.20
        assert_eq!(g.stop(), (3, SlotAction::Verify));

        let mut g = DraftGate::new(8);
        assert_eq!(g.slot(tau, 0, true, 0.9), SlotAction::TakeNgram);
        assert_eq!(g.stop(), (0, SlotAction::TakeNgram));
        assert!(g.stopped());
    }

    #[test]
    fn the_forward_cap_still_lets_a_matching_sequence_take_the_free_tail() {
        let tau = PROD_DRAFT_TAU;
        let mut agreeing = DraftGate::new(8);
        let mut diverging = DraftGate::new(8);
        // A sequence whose drafter agreed with the n-gram at slot 0 has already stopped;
        // use one that never gets an agreement but is capped while all_agree still holds.
        for j in 0..2 {
            diverging.slot(tau, j, false, 0.9);
        }
        diverging.forward_cap(1, 2);
        assert_eq!(diverging.stop(), (2, SlotAction::Verify));

        agreeing.slot(tau, 0, true, 0.9);
        agreeing.forward_cap(0, 2);
        assert_eq!(
            agreeing.stop(),
            (0, SlotAction::TakeNgram),
            "already stopped: the cap does not move it"
        );
    }

    #[test]
    fn assembly_keeps_the_drafter_prefix_and_appends_the_tail() {
        let g = {
            let mut g = DraftGate::new(6);
            g.slot(PROD_DRAFT_TAU, 0, false, 0.9);
            g.slot(PROD_DRAFT_TAU, 1, true, 0.9);
            g
        };
        assert_eq!(g.stop(), (1, SlotAction::TakeNgram));
        let ngram = NgramMatch {
            tokens: vec![70, 71, 72, 73],
            end_pos: Some(9),
            match_len: 5,
        };
        let block = g.assemble(&[10, 11, 12, 13], &ngram, 6);
        assert_eq!(block.tokens, vec![10, 71, 72, 73]);

        // A gate that stops at slot 0 with nothing to take keeps the drafter's first token
        // rather than emitting an empty block.
        let mut stop0 = DraftGate::new(6);
        stop0.forward_cap(0, 1);
        let block = stop0.assemble(&[10, 11], &NgramMatch::default(), 6);
        assert_eq!(block.tokens, vec![10]);
    }

    #[test]
    fn counters_count_draft_tokens_and_never_the_bonus() {
        let mut c = SpecDecodeCounters::default();
        let block = DraftBlock::new([1, 2, 3, 4]);
        // Two drafts: one accepts 2 of 4, one accepts all 4 (plus a bonus).
        let two = Verified {
            accepted_len: 2,
            tokens: vec![1, 2, 55],
            rejected_at: Some(2),
        };
        let four = Verified {
            accepted_len: 4,
            tokens: vec![1, 2, 3, 4, 55],
            rejected_at: None,
        };
        c.observe(block.width(), &two);
        c.observe(block.width(), &four);
        assert_eq!(c.num_drafts, 2);
        assert_eq!(c.num_draft_tokens, 8);
        assert_eq!(
            c.num_accepted_tokens, 6,
            "6 draft tokens, not 8 with bonuses"
        );
        assert_eq!(c.accepted_per_pos, vec![2, 2, 1, 1]);
        assert_eq!(
            c.accepted_per_pos.iter().sum::<u64>(),
            c.num_accepted_tokens,
            "the per-position histogram sums to the accepted total"
        );
        assert!((c.acceptance_per_draft() - 3.0).abs() < 1e-12);
        assert!((c.tokens_per_step() - 4.0).abs() < 1e-12);
        // A step that drafted nothing is not a draft.
        c.observe(
            0,
            &Verified {
                accepted_len: 0,
                tokens: vec![7],
                rejected_at: None,
            },
        );
        assert_eq!(c.num_drafts, 2);
    }

    #[test]
    fn the_live_prod_counter_identity_holds() {
        // curl localhost:8080/metrics, 2026-09-18, unit qwen_prod_nvfp4.
        let per_pos: [u64; 7] = [3440, 2412, 1600, 1091, 726, 518, 402];
        assert_eq!(per_pos.iter().sum::<u64>(), 10_189, "accepted total");
        assert_eq!(4533 * 7, 31_731, "every draft ran the full SPEC=7 width");
        let c = SpecDecodeCounters {
            num_drafts: 4533,
            num_draft_tokens: 31_731,
            num_accepted_tokens: 10_189,
            accepted_per_pos: per_pos.to_vec(),
        };
        assert!((c.acceptance_per_draft() - 2.2477).abs() < 1e-3);
        assert!((c.tokens_per_step() - 3.2477).abs() < 1e-3);
        assert_eq!(c.as_metrics()["saber:spec_decode_num_drafts_total"], 4533);
    }

    // ---------------------------------------------------------------------------------
    // The gate: every captured case, against prod's own output
    // ---------------------------------------------------------------------------------

    fn cases_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/dflash_cases.jsonl")
    }

    const REGENERATE: &str = "regenerate it with the internal reference harness (the exact \
                              podman line is in that script's module docstring)";

    /// `None` means "cases absent" — the caller prints the loud skip and returns.
    fn load_cases() -> Option<Vec<serde_json::Value>> {
        let path = cases_path();
        if !path.exists() {
            eprintln!(
                "\n=====================================================================\n\
                 SKIPPING THE DFLASH2 SEMANTICS GATE: {} is absent.\n\
                 This gate is what proves saber-model::draft agrees with prod's own\n\
                 rejection kernel, width policy, schedule and n-gram matcher.\n\
                 To run it, {REGENERATE}.\n\
                 =====================================================================\n",
                path.display()
            );
            return None;
        }
        let text = std::fs::read_to_string(&path).expect("read cases");
        let cases: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("case record"))
            .collect();
        assert!(
            !cases.is_empty(),
            "{} is empty; {REGENERATE}",
            path.display()
        );
        Some(cases)
    }

    fn of_kind<'a>(cases: &'a [serde_json::Value], kind: &str) -> Vec<&'a serde_json::Value> {
        cases
            .iter()
            .filter(|c| c["kind"].as_str() == Some(kind))
            .collect()
    }

    fn i32_vec(v: &serde_json::Value) -> Vec<i32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_i64().expect("int") as i32)
            .collect()
    }

    fn decisions_of(case: &serde_json::Value) -> Vec<TargetDecision> {
        let greedy = case["greedy"].as_bool().unwrap();
        case["decisions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                if greedy {
                    TargetDecision::Greedy {
                        argmax: d["argmax"].as_i64().unwrap() as i32,
                    }
                } else {
                    TargetDecision::Sampled {
                        target_logprob: d["target_logprob"].as_f64().unwrap() as f32,
                        draft_logprob: d["draft_logprob"].as_f64().unwrap() as f32,
                        u: d["u"].as_f64().unwrap() as f32,
                    }
                }
            })
            .collect()
    }

    #[test]
    fn verify_matches_the_rejection_kernel_on_every_captured_case() {
        let Some(cases) = load_cases() else { return };
        let verify_cases = of_kind(&cases, "verify");
        assert!(
            verify_cases.len() >= 500,
            "only {} verify cases; {REGENERATE}",
            verify_cases.len()
        );
        let mut greedy = 0usize;
        let mut sampled = 0usize;
        let mut one_hot = 0usize;
        let mut rejections = 0usize;
        let mut bonuses = 0usize;
        for case in &verify_cases {
            let block = DraftBlock::new(i32_vec(&case["draft"]));
            let decisions = decisions_of(case);
            let want_len = case["accepted_len"].as_u64().unwrap() as u32;
            // The replacement is a sampler output and is fed in; what the RULE decides is
            // accepted_len and the prefix.
            let replacement = case["rejected_store"].as_i64().map_or(-7, |x| x as i32);
            let got = verify(&block, &decisions, replacement).unwrap();
            assert_eq!(
                got.accepted_len,
                want_len,
                "case {} ({}): draft {:?}",
                case["id"],
                if case["greedy"].as_bool().unwrap() {
                    "greedy"
                } else {
                    "sampled"
                },
                block.tokens
            );
            let want_prefix = i32_vec(&case["accepted_prefix"]);
            assert_eq!(
                got.tokens[..got.accepted_len as usize],
                want_prefix[..],
                "case {}: accepted prefix",
                case["id"]
            );
            assert_eq!(got.tokens.len(), got.accepted_len as usize + 1);
            if case["greedy"].as_bool().unwrap() {
                greedy += 1;
                // The greedy kernel stores the target argmax at the rejected position itself,
                // so that one IS checkable against prod rather than supplied.
                if let Some(store) = case["rejected_store"].as_i64() {
                    assert_eq!(
                        got.extra_token(),
                        store as i32,
                        "case {}: the recovery token is the target argmax",
                        case["id"]
                    );
                    let at = got.rejected_at.expect("a store implies a rejection");
                    assert_eq!(
                        i64::from(decisions[at as usize].accepts(block.tokens[at as usize])),
                        0
                    );
                }
            } else {
                sampled += 1;
            }
            if case["one_hot"].as_bool().unwrap() {
                one_hot += 1;
            }
            if got.rejected_at.is_some() {
                rejections += 1;
            } else {
                bonuses += 1;
            }
        }
        eprintln!(
            "verify gate: {}/{} cases ({greedy} greedy, {sampled} sampled, {one_hot} one-hot \
             draft, {rejections} rejected a token, {bonuses} took the bonus)",
            verify_cases.len(),
            verify_cases.len()
        );
        assert!(greedy > 50 && sampled > 50, "both arms must be exercised");
        assert!(
            rejections > 50 && bonuses > 50,
            "both outcomes must be exercised"
        );
    }

    /// Structural rule 8's negative control: a gate that cannot fail proves nothing.
    ///
    /// Two mutations, each the mistake this port most plausibly makes: counting the extra token
    /// as an accepted draft token, and letting acceptance resume after a rejection.
    #[test]
    fn negative_control_two_wrong_verify_rules_fail_the_gate() {
        let Some(cases) = load_cases() else { return };
        let verify_cases = of_kind(&cases, "verify");

        fn bonus_counted_as_accepted(block: &DraftBlock, d: &[TargetDecision]) -> u32 {
            let mut n = 0;
            for (i, &tok) in block.tokens.iter().enumerate() {
                if !d[i].accepts(tok) {
                    return n + 1; // the classic error: the recovery token counted as accepted
                }
                n += 1;
            }
            n + 1 // and the bonus counted too
        }

        fn acceptance_resumes(block: &DraftBlock, d: &[TargetDecision]) -> u32 {
            block
                .tokens
                .iter()
                .enumerate()
                .filter(|(i, &tok)| d[*i].accepts(tok))
                .count() as u32
        }

        let mut bonus_wrong = 0usize;
        let mut resume_wrong = 0usize;
        for case in &verify_cases {
            let block = DraftBlock::new(i32_vec(&case["draft"]));
            let d = decisions_of(case);
            let want = case["accepted_len"].as_u64().unwrap() as u32;
            if bonus_counted_as_accepted(&block, &d) != want {
                bonus_wrong += 1;
            }
            if acceptance_resumes(&block, &d) != want {
                resume_wrong += 1;
            }
        }
        assert!(
            bonus_wrong > 100,
            "counting the extra token as accepted must fail loudly; it failed {bonus_wrong} cases"
        );
        assert!(
            resume_wrong > 20,
            "resuming acceptance after a rejection must fail loudly; it failed {resume_wrong} cases"
        );
        eprintln!(
            "negative controls: bonus-counted-as-accepted wrong on {bonus_wrong} cases, \
             acceptance-resumes wrong on {resume_wrong}"
        );
    }

    #[test]
    fn the_width_policy_matches_patch_dynwidth_on_every_captured_sequence() {
        let Some(cases) = load_cases() else { return };
        let width_cases = of_kind(&cases, "width");
        assert!(!width_cases.is_empty(), "no width cases; {REGENERATE}");
        let mut steps_checked = 0usize;
        let mut caps_applied = 0usize;
        for case in &width_cases {
            let k = &case["knobs"];
            let policy = WidthPolicy {
                alpha: k["alpha"].as_f64().unwrap(),
                margin: k["margin"].as_u64().unwrap() as u32,
                min_width: k["min_width"].as_u64().unwrap() as u32,
                min_batch: k["min_batch"].as_u64().unwrap() as u32,
            };
            assert_eq!(
                policy,
                WidthPolicy::PROD,
                "the captured knobs are not the ones WidthPolicy::PROD declares"
            );
            let mut ema = AcceptanceEma::new();
            for step in case["steps"].as_array().unwrap() {
                let running = step["running"].as_u64().unwrap() as u32;
                let before = step["spec_len_before"].as_u64().unwrap() as u32;
                let want_after = step["spec_len_after"].as_u64().unwrap() as u32;
                let mut block = DraftBlock::new(vec![0; before as usize]);
                let got_after = policy.apply(&mut block, &ema, running);
                assert_eq!(got_after, want_after, "cap at running={running}");
                if got_after != before {
                    caps_applied += 1;
                }
                let accepted = step["accepted"].as_u64().unwrap() as u32;
                ema.observe(&policy, accepted, got_after);
                match step["ema_after"].as_f64() {
                    None => assert_eq!(ema.value(), None),
                    Some(want) => {
                        let got = ema.value().expect("an EMA after an observation");
                        assert!(
                            (got - want).abs() < 1e-9,
                            "ema {got} vs {want} at running={running}"
                        );
                    }
                }
                steps_checked += 1;
            }
        }
        eprintln!(
            "width gate: {}/{} sequences, {steps_checked} steps, {caps_applied} of them capped",
            width_cases.len(),
            width_cases.len()
        );
        assert!(caps_applied > 20, "the cap must actually bind somewhere");
    }

    #[test]
    fn the_schedule_matches_batch_ceil_at_every_batch_size() {
        let Some(cases) = load_cases() else { return };
        let sched = of_kind(&cases, "schedule");
        assert!(!sched.is_empty(), "no schedule case; {REGENERATE}");
        let case = sched[0];
        assert_eq!(
            case["env_schedule"].as_str().unwrap(),
            PROD_DRAFT_SCHEDULE,
            "the captured environment does not carry the schedule this module declares"
        );
        // Compared at f32, which is what the constant is: `f64::from(0.20f32)` is
        // 0.20000000298…, so a 1e-9 band on the f64 value would fail for no reason.
        assert_eq!(
            case["tau"].as_f64().unwrap() as f32,
            PROD_DRAFT_TAU,
            "captured tau {} vs PROD_DRAFT_TAU {PROD_DRAFT_TAU} -- the unit's \
             SABER_DRAFT_TAU is not what this module declares",
            case["tau"]
        );
        let mut checked = 0usize;
        for row in case["rows"].as_array().unwrap() {
            let spec = row["spec"].as_str().unwrap();
            let parsed = DraftSchedule::parse(spec).unwrap();
            let want_pairs: Vec<(u32, u32)> = row["parsed"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let a = p.as_array().unwrap();
                    (a[0].as_u64().unwrap() as u32, a[1].as_u64().unwrap() as u32)
                })
                .collect();
            assert_eq!(parsed.entries(), want_pairs, "parse of {spec:?}");
            for pair in row["ceilings"].as_array().unwrap() {
                let a = pair.as_array().unwrap();
                let n = a[0].as_u64().unwrap() as u32;
                let want = a[1].as_u64().unwrap() as u32;
                assert_eq!(
                    parsed.forward_ceiling(n),
                    want,
                    "schedule {spec:?} at batch {n}"
                );
                checked += 1;
            }
        }
        eprintln!("schedule gate: {checked} (schedule, batch) cells");
    }

    #[test]
    fn the_matcher_matches_match_gpu_on_every_captured_context() {
        let Some(cases) = load_cases() else { return };
        let ngram = of_kind(&cases, "ngram");
        assert!(!ngram.is_empty(), "no ngram cases; {REGENERATE}");
        let mut hits = 0usize;
        for case in &ngram {
            let k = &case["knobs"];
            let matcher = NgramMatcher {
                max_len: k["max_len"].as_u64().unwrap() as u32,
                min_len: k["min_len"].as_u64().unwrap() as u32,
            };
            assert_eq!(matcher, NgramMatcher::PROD, "captured matcher knobs");
            let ctx = i32_vec(&case["ctx"]);
            let width = case["nspec"].as_u64().unwrap() as u32;
            let want_len = case["clen"].as_u64().unwrap() as u32;
            let want_cont = i32_vec(&case["cont"]);
            let got = matcher.find(&ctx, width);
            assert_eq!(got.len(), want_len, "clen for ctx {ctx:?} width {width}");
            assert_eq!(
                got.tokens,
                want_cont[..want_len as usize],
                "continuation for ctx {ctx:?} width {width}"
            );
            // The reference pads the unused tail with -1; this port returns a short vector.
            assert!(
                want_cont[want_len as usize..]
                    .iter()
                    .all(|&x| x == PAD_TOKEN),
                "the reference pads with -1"
            );
            if want_len > 0 {
                hits += 1;
            }
        }
        eprintln!(
            "ngram gate: {}/{} contexts, {hits} of them found a continuation",
            ngram.len(),
            ngram.len()
        );
        assert!(hits > 30, "the matcher must actually hit somewhere");
    }

    /// The matcher's negative control: the two rules that make it a *longest-suffix* matcher.
    #[test]
    fn negative_control_a_shortest_match_and_a_no_minimum_matcher_fail_the_gate() {
        let Some(cases) = load_cases() else { return };
        let ngram = of_kind(&cases, "ngram");
        let mut no_min_wrong = 0usize;
        let mut earliest_wrong = 0usize;
        for case in &ngram {
            let ctx = i32_vec(&case["ctx"]);
            let width = case["nspec"].as_u64().unwrap() as u32;
            let want_len = case["clen"].as_u64().unwrap() as u32;
            let want_cont = i32_vec(&case["cont"]);
            let want = &want_cont[..want_len as usize];

            // (a) min_len 1: a one-token coincidence becomes a "match".
            let loose = NgramMatcher {
                max_len: 24,
                min_len: 1,
            };
            if loose.find(&ctx, width).tokens != want {
                no_min_wrong += 1;
            }
            // (b) earliest match instead of the most recent among equal lengths.
            if earliest_match(&ctx, width) != want {
                earliest_wrong += 1;
            }
        }
        assert!(
            no_min_wrong > 20,
            "dropping the 3-token minimum must fail; it failed {no_min_wrong}"
        );
        assert!(
            earliest_wrong > 5,
            "preferring the earliest match must fail; it failed {earliest_wrong}"
        );
        eprintln!(
            "ngram negative controls: min_len=1 wrong on {no_min_wrong}, earliest-match wrong \
             on {earliest_wrong}"
        );
    }

    /// The matcher with the tie broken the other way: earliest end position instead of latest.
    fn earliest_match(ctx: &[TokenId], width: u32) -> Vec<TokenId> {
        let m = NgramMatcher::PROD;
        let n = ctx.len();
        if n < 3 || width == 0 {
            return Vec::new();
        }
        let mut best: Option<(u32, usize)> = None;
        for q in 0..n - 1 {
            let mut len = 0u32;
            for k in 0..m.max_len as usize {
                let (Some(sk), Some(qi)) = (n.checked_sub(1 + k), q.checked_sub(k)) else {
                    break;
                };
                if ctx[qi] != ctx[sk] {
                    break;
                }
                len += 1;
            }
            if len < m.min_len {
                continue;
            }
            if best.is_none_or(|(bl, _)| len > bl) {
                best = Some((len, q));
            }
        }
        match best {
            None => Vec::new(),
            Some((_, q)) => {
                let start = q + 1;
                let take = (n - start).min(width as usize);
                ctx[start..start + take].to_vec()
            }
        }
    }
}
