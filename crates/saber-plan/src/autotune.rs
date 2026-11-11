//! The autotune cache — measurements as data, and a loud miss.
//!
//! `ARCHITECTURE.md` § *Cuts* deletes the planner's cost model outright: **constraint match →
//! autotune lookup → error if not tuned**. This module is the second and third of those.
//!
//! The key is the whole point. It carries:
//!
//! * the **op** and the **shape tuple including M** — the tile and split-K optimum flips per shape
//!   *and per M* (`ar-splitk-cost-is-linear-in-m`: gate_up wants split-K 1, down/out want 4);
//! * the **power cap** — a record measured uncapped is invalid at the enforced 210 W;
//! * the **plan input hash** and the **plugin build hash**.
//!
//! On `plan_hash`: the key holds the plan's *input* hash (manifest × device × model × registry),
//! not the hash of the finished [`crate::ExecutionPlan`]. The finished plan contains the bindings
//! this lookup produces, so keying on it would be circular. See [`AutotuneKey::plan_hash`].

use crate::error::{NotTuned, PlanError, Result};
use crate::provenance::{Provenance, TieBreak};
use crate::shape::ShapeKeys;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

/// What a record is keyed on. Every field is load-bearing; dropping one silently serves a number
/// measured under different conditions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AutotuneKey {
    /// The op, e.g. `gemm_w4a8`.
    pub op: String,
    /// The geometry, **including `m`**.
    pub shape: ShapeKeys,
    /// The enforced power cap in watts at measurement time.
    pub power_cap_w: u32,
    /// The plan **input** hash: manifest × device profile × model spec × registry fingerprint.
    /// Not the finished plan's hash — that depends on the bindings this record decides.
    pub plan_hash: String,
    /// Digest over every declared plugin's build hash.
    pub plugin_build_hash: String,
    /// The launch POLICY behind this record — the digest of the committed rule tables owning the
    /// plugins among this route's candidates, not of all of them.
    ///
    /// It used to be a component of `plan_hash`, which made it true and indiscriminate: one line
    /// changed in any table re-keyed all 85 records and cost a full sweep. Keyed per route, an
    /// edit to `dflash_gfx1201.toml` moves the dflash routes' keys and nobody else's, and
    /// `--autotune-in` then re-measures exactly those (I18).
    #[serde(default)]
    pub policy_digest: String,
}

impl AutotuneKey {
    /// Builds a key, refusing a shape with no `m`.
    pub fn new(
        op: &str,
        shape: ShapeKeys,
        power_cap_w: u32,
        plan_hash: &str,
        plugin_build_hash: &str,
        policy_digest: &str,
    ) -> Result<Self> {
        if shape.m().is_none() {
            return Err(PlanError::ShapeMissingM {
                op: op.to_string(),
                shape,
            });
        }
        Ok(Self {
            op: op.to_string(),
            shape,
            power_cap_w,
            plan_hash: plan_hash.to_string(),
            plugin_build_hash: plugin_build_hash.to_string(),
            policy_digest: policy_digest.to_string(),
        })
    }

    /// The canonical digest used as the cache's index.
    pub fn digest(&self) -> String {
        crate::hash::digest_of(self)
    }
}

impl fmt::Display for AutotuneKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] cap={}W plan={} plugins={}",
            self.op,
            self.shape,
            self.power_cap_w,
            short_digest(&self.plan_hash),
            short_digest(&self.plugin_build_hash)
        )
    }
}

pub(crate) fn short_digest(hash: &str) -> &str {
    if hash.len() > 12 {
        &hash[..12]
    } else {
        hash
    }
}

/// The opaque tile parameters a variant carries. The planner never interprets these beyond
/// `split_k`, which it needs for workspace sizing and launch counting.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VariantParams {
    /// The variant id the plugin knows this configuration by.
    pub variant_id: u32,
    /// Opaque parameters: `split_k`, `tm`, `wperm`, `nt`, tile sizes.
    #[serde(default)]
    pub params: BTreeMap<String, i64>,
}

impl VariantParams {
    /// A variant with no parameters.
    pub fn new(variant_id: u32) -> Self {
        Self {
            variant_id,
            params: BTreeMap::new(),
        }
    }

    /// Builder-style parameter set.
    #[must_use]
    pub fn param(mut self, key: &str, value: i64) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// One parameter.
    pub fn get(&self, key: &str) -> Option<i64> {
        self.params.get(key).copied()
    }

    /// The K-split factor, 1 when unset. Drives workspace sizing and the reduce launch.
    pub fn split_k(&self) -> i64 {
        self.get("split_k").unwrap_or(1).max(1)
    }
}

impl fmt::Display for VariantParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "variant {}", self.variant_id)?;
        if !self.params.is_empty() {
            let params: Vec<String> = self
                .params
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            write!(f, " ({})", params.join(", "))?;
        }
        Ok(())
    }
}

/// The variant that came second, kept so `explain` can print the margin against the noise floor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunnerUp {
    /// Implementation name.
    pub impl_name: String,
    /// Variant id.
    pub variant_id: u32,
    /// Its median, microseconds.
    pub measured_us: f64,
}

/// One tuned answer: which implementation, which variant, how fast, measured by whom.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutotuneRecord {
    /// The implementation that won. Must be one of the candidates the constraint match returns.
    pub impl_name: String,
    /// The winning variant and its opaque parameters.
    pub variant: VariantParams,
    /// Median microseconds after the warm-up discard, or **NaN** when the record carries no
    /// measurement at all (see [`Provenance::not_measured`]). JSON has no NaN, so it round-trips
    /// as `null` rather than as a number that would read as a measurement.
    #[serde(with = "nan_as_null")]
    pub measured_us: f64,
    /// How many samples the median was taken over.
    pub samples_kept: usize,
    /// The runner-up, when the sweep had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_up: Option<RunnerUp>,
    /// Set when two passes over this shape named different winners: the cell is not reproducible,
    /// the binding is the rules' answer rather than the measurement's, and both passes are here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unstable: Option<Unstable>,
    /// Who measured it, when, with which harness.
    pub provenance: Provenance,
}

/// What two independent passes over one shape said, when they did not agree.
///
/// A sweep's margin is usually small, and the tie-break exists for exactly that. What it cannot
/// see is a cell whose MEASUREMENT moves between runs: `attn_decode_paged` at batch 4 measured
/// `nw=2` at 81.320 µs in one sweep and 73.960 µs in the next (I17), 9% apart on a shape whose
/// within-run spread is near 0.5%. The winner flipped, the tie-break decided it silently, and
/// nothing in the cache said the cell was not reproducible.
///
/// So a shape is swept TWICE and the two passes are asked the same question independently. Where
/// they agree the record is what it always was. Where they disagree the cache does not get to
/// decide: the record binds what the RULES would pick (the reference launcher's variant, or the
/// lowest variant id when the reference is not among the candidates) and carries both passes'
/// answers, so the next reader sees a measured fact about the shape rather than a coin flip
/// presented as a measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Unstable {
    /// The first pass's winner and its median.
    pub first: RunnerUp,
    /// The second pass's winner and its median.
    pub second: RunnerUp,
    /// The largest disagreement between the two passes about the SAME variant, in percent of the
    /// smaller. This is the number that says how far outside the noise floor the cell sits.
    pub worst_swing_pct: f64,
}

/// `f64` that survives a JSON round trip when it is NaN.
///
/// `serde_json` writes NaN as `null` and then refuses to read `null` back into an `f64`, so a
/// cache holding an unmeasured route would save and fail to load — the worst possible failure for
/// a file whose whole job is to be re-read.
mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_finite() {
            s.serialize_f64(*v)
        } else {
            s.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

impl AutotuneRecord {
    /// A record for a route this harness **cannot time**, bound to the only variant the rules
    /// permit. `measured_us` is NaN and the provenance says why; nothing else may fill it in.
    pub fn not_measured(
        impl_name: &str,
        variant: VariantParams,
        provenance: Provenance,
        why: &str,
    ) -> Self {
        Self {
            impl_name: impl_name.to_string(),
            variant,
            measured_us: f64::NAN,
            samples_kept: 0,
            runner_up: None,
            unstable: None,
            provenance: provenance.not_measured(why),
        }
    }

    /// Whether this record carries a measurement at all.
    #[must_use]
    pub fn is_measured(&self) -> bool {
        self.provenance.not_measured.is_none()
    }

    /// A record written by hand (a port of an existing measurement, a fixture).
    pub fn new(
        impl_name: &str,
        variant: VariantParams,
        measured_us: f64,
        provenance: Provenance,
    ) -> Self {
        Self {
            impl_name: impl_name.to_string(),
            variant,
            measured_us,
            samples_kept: 1,
            runner_up: None,
            unstable: None,
            provenance,
        }
    }

    /// How much the winner beat the runner-up by, in percent. `None` when there was no runner-up.
    ///
    /// The noise floor on this box is ±2.2% with a ±0.5 µs absolute floor; a margin under it did
    /// not decide anything, and [`Autotuner::record_measurements`] will then have chosen the
    /// winner by [`Provenance::tie_broken_by`] instead. In that case this can be **negative** — a
    /// tied variant that measured a fraction of a microsecond faster is the runner-up. A negative
    /// margin is therefore not a bug; it is a tie, and it is always inside the floor.
    pub fn margin_pct(&self) -> Option<f64> {
        self.runner_up.as_ref().map(|r| {
            if self.measured_us > 0.0 {
                (r.measured_us - self.measured_us) / self.measured_us * 100.0
            } else {
                0.0
            }
        })
    }
}

/// One entry, as persisted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheEntry {
    /// What was measured.
    pub key: AutotuneKey,
    /// The answer.
    pub record: AutotuneRecord,
}

/// The persisted store of measurements.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "CacheFile", into = "CacheFile")]
pub struct AutotuneCache {
    entries: BTreeMap<String, CacheEntry>,
}

/// The on-disk shape: a sorted list, so the file is a stable diff.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct CacheFile {
    schema_version: u32,
    entries: Vec<CacheEntry>,
}

impl From<CacheFile> for AutotuneCache {
    fn from(f: CacheFile) -> Self {
        let entries = f
            .entries
            .into_iter()
            .map(|e| (e.key.digest(), e))
            .collect::<BTreeMap<_, _>>();
        Self { entries }
    }
}

impl From<AutotuneCache> for CacheFile {
    fn from(c: AutotuneCache) -> Self {
        CacheFile {
            schema_version: 1,
            entries: c.entries.into_values().collect(),
        }
    }
}

/// An op whose records do not all carry the same policy digest.
///
/// The cache holds one measurement per (op, shape, cap, plan, plugins, POLICY). Two policies for
/// one op means two measurements of the same route under rule tables that differ, and nothing in
/// the file says which is current — the planner binds whichever key the current tables cut, while
/// `explain` and every census over the file read both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicySplit {
    /// The op name, e.g. `attn_gate_quant`.
    pub op: String,
    /// The distinct policy digests its records carry, in order.
    pub digests: Vec<String>,
}

impl fmt::Display for PolicySplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} carries {} policy digests; one op's routes bind one table set ({})",
            self.op,
            self.digests.len(),
            self.digests
                .iter()
                .map(|d| short_digest(d).to_owned())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for PolicySplit {}

impl AutotuneCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no records.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every entry, in key-digest order.
    pub fn entries(&self) -> impl Iterator<Item = &CacheEntry> {
        self.entries.values()
    }

    /// Inserts or replaces a record.
    pub fn insert(&mut self, key: AutotuneKey, record: AutotuneRecord) {
        self.entries
            .insert(key.digest(), CacheEntry { key, record });
    }

    /// Looks a key up. A miss is [`NotTuned`] — there is no cost model to fall back on.
    pub fn lookup(&self, key: &AutotuneKey) -> std::result::Result<&AutotuneRecord, NotTuned> {
        self.entries
            .get(&key.digest())
            .map(|e| &e.record)
            .ok_or_else(|| NotTuned {
                key: Box::new(key.clone()),
                candidates: Vec::new(),
            })
    }

    /// [`AutotuneCache::lookup`], naming the candidates a tuning run would have to measure.
    pub fn require(&self, key: &AutotuneKey, candidates: &[&str]) -> Result<&AutotuneRecord> {
        self.entries
            .get(&key.digest())
            .map(|e| &e.record)
            .ok_or_else(|| {
                PlanError::NotTuned(NotTuned {
                    key: Box::new(key.clone()),
                    candidates: candidates.iter().map(|c| (*c).to_string()).collect(),
                })
            })
    }

    /// Reads a cache file. A missing file is an empty cache, not an error — but every lookup
    /// against it will then fail loudly, which is the intent.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(|source| PlanError::CacheJson {
                path: path.to_path_buf(),
                source,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(source) => Err(PlanError::Io {
                what: "reading autotune cache",
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Every op in the cache and the policy digests its records carry.
    ///
    /// The uniform 16-query attention pair uses a dedicated head table and the shared
    /// combine table, so it has a separate policy family from narrow attention. Within
    /// a family two digests indicate a merge that retained both sides of an edit. See
    /// [`AutotuneCache::one_policy_digest_per_op`].
    #[must_use]
    pub fn policy_digests_by_op(&self) -> BTreeMap<&str, BTreeSet<&str>> {
        let mut out: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for entry in self.entries.values() {
            let family = if entry.key.op == "attn_decode_paged"
                && entry.key.shape.get("q_len") == Some(&crate::shape::ShapeValue::Num(16))
            {
                "attn_decode_paged[uniform16]"
            } else {
                entry.key.op.as_str()
            };
            out.entry(family)
                .or_default()
                .insert(entry.key.policy_digest.as_str());
        }
        out
    }

    /// **The guard.** An op whose records carry more than one policy digest.
    ///
    /// A record is keyed on the digest of the tables its own route could bind (I18), so a rules
    /// edit moves the keys of the routes that table can reach and of no others. What that left
    /// open is the MERGE: an incremental sweep (`saber tune --autotune-in`) re-measures the routes
    /// whose key moved, and if it also carries the old records forward the op ends up with a
    /// record under each digest — one of them a measurement of a policy that no longer exists,
    /// indistinguishable in the file from the one that does.
    ///
    /// Not hypothetical: it fired on main on 2026-09-21 as `attn_gate_quant carries 2 policy
    /// digests` after F's rope row re-keyed the glue routes, and was worked around with a full
    /// sweep. [`crate::Resolved::carry_forward`] is the fix; this is the check, and both the CI
    /// guard over the committed cache and the merge's own tests call THIS function rather than a
    /// copy of it.
    ///
    /// # Errors
    /// The first op (in name order) carrying more than one digest.
    pub fn one_policy_digest_per_op(&self) -> std::result::Result<(), PolicySplit> {
        for (op, digests) in self.policy_digests_by_op() {
            if digests.len() > 1 {
                return Err(PolicySplit {
                    op: op.to_owned(),
                    digests: digests.into_iter().map(str::to_owned).collect(),
                });
            }
        }
        Ok(())
    }

    /// Writes the cache as pretty JSON, sorted by key digest.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = CacheFile::from(self.clone());
        let text = serde_json::to_string_pretty(&file).map_err(|source| PlanError::CacheJson {
            path: path.to_path_buf(),
            source,
        })?;
        std::fs::write(path, text + "\n").map_err(|source| PlanError::Io {
            what: "writing autotune cache",
            path: path.to_path_buf(),
            source,
        })
    }
}

/// One variant's raw samples from a sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// The implementation measured.
    pub impl_name: String,
    /// The variant measured.
    pub variant: VariantParams,
    /// Raw samples, microseconds, **in the order they were measured**. Order matters: the first
    /// sample of the first variant is the shape's warm-up.
    pub samples: Vec<f64>,
    /// Whether this is the variant the reference launcher picks for this shape. Used **only** to
    /// break a tie the noise floor cannot decide; it never beats a real margin.
    pub is_reference: bool,
}

impl Measurement {
    /// A measurement row.
    pub fn new(impl_name: &str, variant: VariantParams, samples: Vec<f64>) -> Self {
        Self {
            impl_name: impl_name.to_string(),
            variant,
            samples,
            is_reference: false,
        }
    }

    /// Marks this row as the reference launcher's own choice for this shape.
    #[must_use]
    pub fn reference(mut self) -> Self {
        self.is_reference = true;
        self
    }
}

/// The smallest difference at `us` that counts as a result.
///
/// `LEDGER.md`, *Timing*: the ±2.2% band carries an **absolute floor of ±0.5 µs**, because a
/// kernel under ~25 µs is judged on the absolute delta — "any per-launch dispatch cost would have
/// to be one-signed across runs, which it is not". So a delta decides something only when it
/// clears *both*, which is the same as clearing the larger of the two.
#[must_use]
pub fn noise_floor_us(us: f64) -> f64 {
    (us * crate::explain::NOISE_FLOOR_PCT / 100.0).max(crate::explain::NOISE_FLOOR_US)
}

/// Writes measurements into a cache under the warm-up protocol.
///
/// # The warm-up discard, and why it is not optional
///
/// On this box **the first-measured variant per shape reads ~17% inflated**
/// (`mxfp4-kernel-tile-optimum`). A naive measure-in-order autotuner therefore rejects whatever it
/// measured first — systematically, silently, and forever, because the bad number is then cached.
///
/// So [`Autotuner::record_measurements`] discards **the first sample of the first variant offered
/// for a shape** and takes the **median** of what is left, per variant. Callers must pass the
/// variants in measurement order for this to mean anything, and must give the first variant at
/// least two samples.
pub struct Autotuner {
    cache: AutotuneCache,
    provenance: Provenance,
}

impl Autotuner {
    /// A tuner writing into `cache`, stamping every record with `provenance`.
    pub fn new(cache: AutotuneCache, provenance: Provenance) -> Self {
        Self { cache, provenance }
    }

    /// Records a route measured by a **different harness**, naming it.
    ///
    /// The two-rank all-reduce is the case: it cannot be timed in the sweeping process at all —
    /// its workspaces are peer-visible and its kernel spins on a peer's flags — so `saber tune`
    /// runs `saber ar-gate` as two rank processes and brings the number back. The record is a
    /// real measurement and says which harness took it, because "never compare across harnesses"
    /// needs the harness to be written down.
    pub fn record_external(
        &mut self,
        key: AutotuneKey,
        impl_name: &str,
        variant: VariantParams,
        measured_us: f64,
        samples_kept: usize,
        harness: &str,
    ) -> &AutotuneRecord {
        let mut provenance = self.provenance.clone();
        provenance.harness = harness.to_string();
        let record = AutotuneRecord {
            impl_name: impl_name.to_string(),
            variant,
            measured_us,
            samples_kept,
            runner_up: None,
            unstable: None,
            provenance,
        };
        let digest = key.digest();
        self.cache.insert(key, record);
        &self.cache.entries[&digest].record
    }

    /// Records a route the harness cannot time: the rules' only legal variant, no number.
    ///
    /// The planner has no cost model, so an unbindable key is a refusal to start; this is the one
    /// way to bind a route without measuring it, and it is loud at every later step —
    /// `explain` prints NOT MEASURED, and the promotion ladder refuses to lift it to `measured`.
    pub fn record_unmeasured(
        &mut self,
        key: AutotuneKey,
        impl_name: &str,
        variant: VariantParams,
        why: &str,
    ) -> &AutotuneRecord {
        let record = AutotuneRecord::not_measured(impl_name, variant, self.provenance.clone(), why);
        let digest = key.digest();
        self.cache.insert(key, record);
        &self.cache.entries[&digest].record
    }

    /// The cache being written.
    pub fn cache(&self) -> &AutotuneCache {
        &self.cache
    }

    /// Takes the cache back.
    pub fn into_cache(self) -> AutotuneCache {
        self.cache
    }

    /// Records one shape's sweep and returns the record that won.
    ///
    /// `key` carries the shape (including `m`), the power cap and the two hashes;
    /// `measurements` carries each variant's raw samples **in measurement order**. The first
    /// sample of the first variant is dropped as the shape's warm-up — see the type docs.
    pub fn record_measurements(
        &mut self,
        key: AutotuneKey,
        measurements: Vec<Measurement>,
    ) -> Result<&AutotuneRecord> {
        let medians = medians_after_warmup(&measurements)?;

        let mut ranked: Vec<(usize, f64)> = medians.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        let fastest_us = ranked[0].1;
        let (winner_idx, tie_broken_by) = break_tie(&measurements, &ranked, fastest_us);
        let winner = &measurements[winner_idx];
        // The runner-up is the best of the rest, so it can be *faster* than the winner when a tie
        // was broken away from the fastest. That difference is inside the floor by construction.
        let runner_up = ranked
            .iter()
            .find(|(idx, _)| *idx != winner_idx)
            .map(|&(idx, us)| RunnerUp {
                impl_name: measurements[idx].impl_name.clone(),
                variant_id: measurements[idx].variant.variant_id,
                measured_us: us,
            });
        let kept = if winner_idx == 0 {
            winner.samples.len() - 1
        } else {
            winner.samples.len()
        };

        let record = AutotuneRecord {
            impl_name: winner.impl_name.clone(),
            variant: winner.variant.clone(),
            measured_us: medians[winner_idx],
            samples_kept: kept,
            runner_up,
            unstable: None,
            provenance: self.provenance.clone().with_tie_break(tie_broken_by),
        };
        let digest = key.digest();
        self.cache.insert(key, record);
        Ok(&self.cache.entries[&digest].record)
    }

    /// Two independent passes over one shape, and a record that says whether they agreed.
    ///
    /// The passes must carry the same variants in the same order — they are two sweeps of one
    /// enumeration — and each is scored on its own by exactly the protocol
    /// [`Tuner::record_measurements`] uses. Then:
    ///
    /// * **they agree**: the record is the first pass's, unchanged. The second pass cost GPU time
    ///   and bought the claim that the first one is reproducible, which is what it is for.
    /// * **they disagree**: the measurement has no opinion worth binding, so the RULES keep the
    ///   cell. The record binds the reference launcher's variant when one is among the
    ///   candidates, else the lowest variant id — the same fallback a tie takes — and carries
    ///   [`Unstable`] with both passes' winners and medians. `explain` and the sweep's own
    ///   summary print it, so a 9% swing decided by a tie-break can no longer look like a
    ///   measurement (I17's `attn_decode_paged` at batch 4).
    pub fn record_two_passes(
        &mut self,
        key: AutotuneKey,
        first: Vec<Measurement>,
        second: Vec<Measurement>,
    ) -> Result<&AutotuneRecord> {
        if first.len() != second.len()
            || first
                .iter()
                .zip(&second)
                .any(|(a, b)| a.variant.variant_id != b.variant.variant_id)
        {
            return Err(PlanError::Model(
                "the two passes swept different variant sets, so their winners are not comparable"
                    .to_owned(),
            ));
        }
        let a = pick_winner(&first)?;
        let b = pick_winner(&second)?;
        if first[a.0].variant.variant_id == second[b.0].variant.variant_id {
            return self.record_measurements(key, first);
        }

        // The worst the two passes disagreed about ONE variant, which is the number that says how
        // far outside the noise floor the cell sits — the winner flipping is the symptom.
        let worst =
            a.2.iter()
                .zip(&b.2)
                .map(|(x, y)| {
                    let (lo, hi) = if x < y { (*x, *y) } else { (*y, *x) };
                    if lo > 0.0 {
                        (hi - lo) / lo * 100.0
                    } else {
                        0.0
                    }
                })
                .fold(0.0_f64, f64::max);
        let unstable = Unstable {
            first: RunnerUp {
                impl_name: first[a.0].impl_name.clone(),
                variant_id: first[a.0].variant.variant_id,
                measured_us: a.2[a.0],
            },
            second: RunnerUp {
                impl_name: second[b.0].impl_name.clone(),
                variant_id: second[b.0].variant.variant_id,
                measured_us: b.2[b.0],
            },
            worst_swing_pct: worst,
        };
        // The rules keep the cell: everything is treated as tied, so the reference launcher's
        // variant wins if it is among the candidates and the lowest id wins otherwise.
        let all_tied: Vec<(usize, f64)> = (0..first.len()).map(|i| (i, a.2[i])).collect();
        let (idx, tie) = break_tie(&first, &all_tied, f64::INFINITY);
        let winner = &first[idx];
        let runner_up = all_tied
            .iter()
            .filter(|(i, _)| *i != idx)
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|&(i, us)| RunnerUp {
                impl_name: first[i].impl_name.clone(),
                variant_id: first[i].variant.variant_id,
                measured_us: us,
            });
        let record = AutotuneRecord {
            impl_name: winner.impl_name.clone(),
            variant: winner.variant.clone(),
            measured_us: a.2[idx],
            // The same warm-up convention the one-pass path uses: the very first sample of the
            // first variant is the shape's warm-up and is not kept.
            samples_kept: winner.samples.len() - usize::from(idx == 0),
            runner_up,
            unstable: Some(unstable),
            provenance: self.provenance.clone().with_tie_break(tie),
        };
        let digest = key.digest();
        self.cache.insert(key, record);
        Ok(&self.cache.entries[&digest].record)
    }
}

/// One pass's answer: `(winner index, what broke a tie, every variant's median)`.
fn pick_winner(m: &[Measurement]) -> Result<(usize, Option<TieBreak>, Vec<f64>)> {
    let medians = medians_after_warmup(m)?;
    let mut ranked: Vec<(usize, f64)> = medians.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let fastest = ranked[0].1;
    let (idx, tie) = break_tie(m, &ranked, fastest);
    Ok((idx, tie, medians))
}

/// Pick the winner among variants the noise floor cannot separate.
///
/// # Why a tie needs a rule rather than a sort
///
/// Most margins in a real sweep are tiny: of the 32 shapes tuned on this box, 11 were decided by
/// 0.0–1.6%, which the ±2.2%/±0.5 µs floor says is nothing at all. Taking the fastest median
/// anyway makes the binding a coin flip — the same sweep, run three times, would name different
/// kernels, and every artefact keyed on the plan hash would churn with the noise.
///
/// So everything within one noise floor of the fastest is **tied**, and the tie is resolved by a
/// rule that does not depend on the measurement:
///
/// 1. the variant the reference launcher picks, if it is tied — the engine then agrees with
///    production wherever the measurement has no opinion;
/// 2. otherwise the lowest `variant_id`, which is arbitrary but total and stable.
///
/// # Membership is paired, and it needs room on both sides of the boundary
///
/// `CLAUDE.md`: *"paired A/B only, same cache"*. The sweep interleaves variants round-robin
/// precisely so that round `r` holds one sample of every variant taken microseconds apart, which
/// makes the per-round difference a **paired** statistic: drift that moves both variants cancels
/// in it and does not cancel in the difference of two independent medians.
///
/// The threshold is the floor from `LEDGER.md` and nothing else. Two alternatives were tried on
/// this machine and are recorded there as rejected: a quorum of rounds inside the floor (too
/// strict — per-launch jitter is itself of the order of the floor, so it rejected a genuine
/// 0.1 µs tie), and half the floor (an invention, and it only moved which shape flipped). The
/// residual is real and is stated in the ledger: two of the 32 tuned shapes have a
/// reference-versus-fastest gap of 0.6–1.2 floors, so *which side of the floor they land on* is
/// itself not reproducible, and a cold re-sweep can re-decide them. Carrying the cache forward
/// (`saber tune --autotune-in`) is what makes a plan hash reproducible; re-measuring is not.
///
/// Round 0 is dropped from every pairing: it is the round that contains the shape's warm-up
/// sample, which is ~17% inflated for whichever variant was measured first.
///
/// `ranked` must be `(index, median)` sorted ascending. Returns the winner's index and what broke
/// the tie, or `None` when the margin cleared the floor and the measurement decided.
fn break_tie(
    measurements: &[Measurement],
    ranked: &[(usize, f64)],
    fastest_us: f64,
) -> (usize, Option<TieBreak>) {
    let floor = noise_floor_us(fastest_us);
    let fastest = ranked[0].0;
    let tied: Vec<usize> = ranked
        .iter()
        .take_while(|(idx, us)| {
            paired_delta(&measurements[fastest].samples, &measurements[*idx].samples)
                .unwrap_or(us - fastest_us)
                <= floor
        })
        .map(|(idx, _)| *idx)
        .collect();
    if tied.len() < 2 {
        return (ranked[0].0, None);
    }
    if let Some(idx) = tied.iter().copied().find(|i| measurements[*i].is_reference) {
        return (idx, Some(TieBreak::ReferenceLauncher));
    }
    let idx = tied
        .iter()
        .copied()
        .min_by_key(|i| {
            (
                measurements[*i].variant.variant_id,
                measurements[*i].impl_name.clone(),
            )
        })
        .expect("the tied set is non-empty");
    (idx, Some(TieBreak::LowestVariantId))
}

/// `median(b[r] - a[r])` over the rounds both variants have, skipping round 0.
///
/// `None` when the rows cannot be paired — different lengths, or fewer than two usable rounds —
/// in which case the caller falls back to the difference of the medians.
fn paired_delta(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 3 {
        return None;
    }
    let deltas: Vec<f64> = a[1..].iter().zip(&b[1..]).map(|(x, y)| y - x).collect();
    Some(median(&deltas))
}

/// The per-variant medians a sweep resolves to, after the shape's warm-up sample is discarded.
///
/// Exposed so a harness can check the protocol without writing to a cache.
pub fn medians_after_warmup(measurements: &[Measurement]) -> Result<Vec<f64>> {
    if measurements.is_empty() {
        return Err(PlanError::Autotune(
            "a sweep with no variants decides nothing".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(measurements.len());
    for (i, m) in measurements.iter().enumerate() {
        // The first sample of the FIRST variant is the shape's warm-up and is dropped.
        let samples = if i == 0 {
            &m.samples[m.samples.len().min(1)..]
        } else {
            &m.samples[..]
        };
        if samples.is_empty() {
            return Err(PlanError::Autotune(format!(
                "variant {} of `{}` has no samples left after the shape's warm-up discard: the \
                 first variant measured for a shape needs at least two samples (the first reads \
                 ~17% inflated on this box)",
                m.variant.variant_id, m.impl_name
            )));
        }
        out.push(median(samples));
    }
    Ok(out)
}

/// The median of a non-empty sample set.
fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    }
}
