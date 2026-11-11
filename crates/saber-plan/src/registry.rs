//! The constraint registry — which implementation covers which op at which geometry.
//!
//! Modelled on the reference's `SABERConstraint{key, op, ival, sval}` with `C_EQ/C_LE/C_GE/C_DIV/C_IN`, its
//! `select_kernel()` and its `explain()`, and it keeps the reference's split exactly:
//!
//! > A constraint is one predicate on one named parameter. It is a **NECESSARY** condition, not a
//! > sufficient one: it settles whether a kernel exists for a geometry, never whether a particular
//! > call is safe. Strides, contiguity, alignment and buffer sizes are per-call and stay with the
//! > entry point.
//!
//! **Constraints discriminate ops, never tiles.** Tile choice is an opaque `variant_id` plus a
//! parameter blob from the autotune cache ([`crate::AutotuneRecord`]). That single split is what
//! prevents the macro explosion the current stack has (`ks x tm x wperm x nt x bits`).
//!
//! Two differences from saber, both deliberate:
//!
//! * `select()` returns **every** match rather than the first. Zero matches is an error carrying
//!   the closest failing constraint per candidate; several matches are handed to the autotune stage
//!   to break the tie with a measurement.
//! * The selection trace is returned, not logged to a process-global. The reference's `selections()` log
//!   exists because callers ask at import time with no reader attached; here the plan itself is the
//!   record, and `explain` prints it.

use crate::error::{NoImplementation, PlanError, Result};
use crate::provenance::Promotion;
use crate::shape::{ShapeKeys, ShapeValue};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The five predicates, with the reference's semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConstraintOp {
    /// `param == ival`
    Eq,
    /// `param <= ival`
    Le,
    /// `param >= ival`
    Ge,
    /// `param % ival == 0`
    Div,
    /// `param` is one of the space-separated words in `sval`
    In,
}

impl ConstraintOp {
    /// The relation as it is printed in a refusal ("needs <= 128").
    fn rel(self) -> &'static str {
        match self {
            ConstraintOp::Eq => "== ",
            ConstraintOp::Le => "<= ",
            ConstraintOp::Ge => ">= ",
            ConstraintOp::Div => "divisible by ",
            ConstraintOp::In => "one of: ",
        }
    }
}

/// One predicate on one named geometry parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    /// Parameter name, e.g. `head_dim`.
    pub key: String,
    /// Which predicate.
    pub op: ConstraintOp,
    /// Numeric operand (`Eq` / `Le` / `Ge` / `Div`).
    pub ival: i64,
    /// String-set operand, space separated (`In`).
    pub sval: String,
}

impl Constraint {
    /// `key == value`
    pub fn eq(key: &str, value: i64) -> Self {
        Self::num(key, ConstraintOp::Eq, value)
    }

    /// `key <= value`
    pub fn le(key: &str, value: i64) -> Self {
        Self::num(key, ConstraintOp::Le, value)
    }

    /// `key >= value`
    pub fn ge(key: &str, value: i64) -> Self {
        Self::num(key, ConstraintOp::Ge, value)
    }

    /// `key % value == 0`
    pub fn div(key: &str, value: i64) -> Self {
        Self::num(key, ConstraintOp::Div, value)
    }

    /// `key` is one of the space-separated words in `set`.
    pub fn is_in(key: &str, set: &str) -> Self {
        Self {
            key: key.to_string(),
            op: ConstraintOp::In,
            ival: 0,
            sval: set.to_string(),
        }
    }

    fn num(key: &str, op: ConstraintOp, value: i64) -> Self {
        Self {
            key: key.to_string(),
            op,
            ival: value,
            sval: String::new(),
        }
    }

    /// Tests the predicate against a geometry. `Err` carries the reference's refusal wording.
    pub fn check(&self, shape: &ShapeKeys) -> std::result::Result<(), String> {
        let Some(value) = shape.get(&self.key) else {
            return Err(format!("{} not given", self.key));
        };
        if self.op == ConstraintOp::In {
            let ok = matches!(value, ShapeValue::Str(s) if self.sval.split(' ').any(|w| w == s));
            return if ok {
                Ok(())
            } else {
                Err(format!(
                    "{}={}, needs one of: {}",
                    self.key, value, self.sval
                ))
            };
        }
        let Some(v) = value.as_num() else {
            return Err(format!("{}={} is not a number", self.key, value));
        };
        let ok = match self.op {
            ConstraintOp::Eq => v == self.ival,
            ConstraintOp::Le => v <= self.ival,
            ConstraintOp::Ge => v >= self.ival,
            ConstraintOp::Div => self.ival != 0 && v % self.ival == 0,
            ConstraintOp::In => unreachable!("handled above"),
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "{}={}, needs {}{}",
                self.key,
                v,
                self.op.rel(),
                self.ival
            ))
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.op == ConstraintOp::In {
            write!(f, "{} in {{{}}}", self.key, self.sval)
        } else {
            write!(f, "{} {}{}", self.key, self.op.rel(), self.ival)
        }
    }
}

/// What a workspace is for, which is not expressible as a byte count.
///
/// > Prod registers split-K scratch *and* a block counter, and the fused-AR family needs slot
/// > parity to alternate per launch. "N bytes" cannot express that. Retires the
/// > `set_decode_scratch` global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceKind {
    /// Aliasable between ops in the same bucket; contents are dead across launches.
    Scratch,
    /// Survives replay, zeroed once at instantiation (a block counter, a slot-parity flag).
    Persistent,
    /// Exported to the peer rank over IPC (the one-shot all-reduce push buffers).
    IpcPeerVisible,
}

impl WorkspaceKind {
    /// The name `explain` prints.
    pub fn as_str(self) -> &'static str {
        match self {
            WorkspaceKind::Scratch => "scratch",
            WorkspaceKind::Persistent => "persistent",
            WorkspaceKind::IpcPeerVisible => "ipc-peer-visible",
        }
    }
}

impl fmt::Display for WorkspaceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How many bytes a workspace needs at a given shape and variant. Data, not a call into a plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceSizing {
    /// A constant.
    Fixed(u64),
    /// `split_k * m * n * elem_bytes` — the split-K partials, whose cost is linear in M and is why
    /// the split-K rule flips per shape and per M (`ar-splitk-cost-is-linear-in-m`).
    SplitKPartials {
        /// Bytes per partial element (4 for fp32 accumulators).
        elem_bytes: u64,
    },
    /// `m * bytes_per_row` — per-row activation scales and similar.
    PerRow {
        /// Bytes per row.
        bytes_per_row: u64,
    },
    /// The split-KV partials: **allocated** at `capacity`, **used** at
    /// `m * n * splits * bytes_per_split`.
    ///
    /// The two are different on purpose and the difference is the point. The buffer is sized once
    /// at construction for the worst shape the engine can schedule, because allocating during a
    /// graph capture is not allowed and the address has to survive replay; what a *bucket* needs
    /// is the split law's answer at that bucket, which is a fraction of it. Printing only the
    /// allocation hides the law; printing only the law would claim an allocation the engine never
    /// makes. `explain` prints both.
    SplitKvPartials {
        /// What is allocated, from the rules file's `capacity_bytes`.
        capacity: u64,
        /// Bytes one (row, split) partial takes.
        bytes_per_split: u64,
    },
}

impl WorkspaceSizing {
    /// Bytes needed at this shape and variant.
    pub fn bytes(&self, shape: &ShapeKeys, split_k: i64) -> u64 {
        let dim = |key: &str| shape.num(key).unwrap_or(0).max(0) as u64;
        match *self {
            WorkspaceSizing::Fixed(b) => b,
            WorkspaceSizing::SplitKPartials { elem_bytes } => {
                split_k.max(1) as u64 * dim("m") * dim("n") * elem_bytes
            }
            WorkspaceSizing::PerRow { bytes_per_row } => dim("m") * bytes_per_row,
            // The allocation, unchanged: the capacity IS the size law for a buffer that cannot be
            // resized under capture.
            WorkspaceSizing::SplitKvPartials { capacity, .. } => capacity,
        }
    }

    /// What this bucket actually uses, when the law can say. `None` when the sizing has no
    /// per-bucket form, or when the variant carries no split count.
    ///
    /// `splits` comes from the tuned variant's parameters, because the split count is the
    /// measurement's, not the planner's: the decode split law reads the context band, and the
    /// record was taken at whatever it answered.
    #[must_use]
    pub fn used_bytes(&self, shape: &ShapeKeys, splits: Option<i64>) -> Option<u64> {
        match *self {
            WorkspaceSizing::SplitKvPartials {
                bytes_per_split, ..
            } => {
                let dim = |key: &str| shape.num(key).unwrap_or(0).max(0) as u64;
                let splits = u64::try_from(splits?.max(0)).ok()?;
                // rows = tokens x q heads, one partial per (row, split).
                Some(dim("m") * dim("n") * splits * bytes_per_split)
            }
            _ => None,
        }
    }
}

/// One workspace an implementation needs, by kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceReq {
    /// What it is, for `explain` ("split-K partials", "block counter").
    pub name: String,
    /// Aliasable, persistent or peer-visible.
    pub kind: WorkspaceKind,
    /// How it is sized.
    pub sizing: WorkspaceSizing,
}

impl WorkspaceReq {
    /// Declares a workspace.
    pub fn new(name: &str, kind: WorkspaceKind, sizing: WorkspaceSizing) -> Self {
        Self {
            name: name.to_string(),
            kind,
            sizing,
        }
    }
}

/// `n_launches(shape)` as data: the host owns launch logic, so launch count is budgetable ahead of
/// running rather than detectable after the fact (`CLAUDE.md` structural rule 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchModel {
    /// Always this many launches.
    Fixed(u32),
    /// `base`, plus one reduce launch when the bound variant splits K.
    SplitKReduce {
        /// Launches before the reduce.
        base: u32,
    },
}

impl LaunchModel {
    /// Launches for one call of this implementation at a given variant.
    pub fn n_launches(&self, split_k: i64) -> u32 {
        match *self {
            LaunchModel::Fixed(n) => n,
            LaunchModel::SplitKReduce { base } => base + u32::from(split_k > 1),
        }
    }
}

/// One implementation of one op: the row the reference's registry carries, plus the bits the planner needs
/// (graph safety, wave size, workspaces, launch count, promotion state).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Implementation {
    /// Entry-point name, e.g. `gemm_w4a8_mxfp4_decode`.
    pub name: String,
    /// The plugin that ships it; must be declared in the manifest's `[[plugins]]`.
    pub plugin: String,
    /// The op it implements — the key callers select on, and the one thing several rows share.
    pub op: String,
    /// What it computes, for a human.
    pub computes: String,
    /// The geometry it is compiled for, for a human.
    pub shape: String,
    /// Operand dtypes, for a human.
    pub dtypes: String,
    /// The same geometry, testable. Necessary conditions only.
    pub constraints: Vec<Constraint>,
    /// Legal inside a captured graph.
    pub graph_safe: bool,
    /// May synchronise; illegal under capture even when otherwise graph-safe.
    pub may_sync: bool,
    /// The wave size the code object was compiled with (a TU property, read from its metadata).
    pub wave_size: u32,
    /// LDS bytes per workgroup, read from the code object's metadata.
    pub lds_bytes: u32,
    /// Workspaces, by kind.
    pub workspaces: Vec<WorkspaceReq>,
    /// Static launch count.
    pub launches: LaunchModel,
    /// Where it stands on the promotion ladder.
    pub promotion: Promotion,
}

impl Implementation {
    /// A new implementation with everything not named defaulted: graph-safe, no sync, wave32, no
    /// workspaces, one launch, `implemented`.
    pub fn new(name: &str, plugin: &str, op: &str) -> Self {
        Self {
            name: name.to_string(),
            plugin: plugin.to_string(),
            op: op.to_string(),
            computes: String::new(),
            shape: String::new(),
            dtypes: String::new(),
            constraints: Vec::new(),
            graph_safe: true,
            may_sync: false,
            wave_size: 32,
            lds_bytes: 0,
            workspaces: Vec::new(),
            launches: LaunchModel::Fixed(1),
            promotion: Promotion::implemented(),
        }
    }

    /// Sets the human-readable description fields.
    #[must_use]
    pub fn describe(mut self, computes: &str, shape: &str, dtypes: &str) -> Self {
        self.computes = computes.to_string();
        self.shape = shape.to_string();
        self.dtypes = dtypes.to_string();
        self
    }

    /// Adds one necessary condition.
    #[must_use]
    pub fn constrain(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Sets the capture bits.
    #[must_use]
    pub fn capture(mut self, graph_safe: bool, may_sync: bool) -> Self {
        self.graph_safe = graph_safe;
        self.may_sync = may_sync;
        self
    }

    /// Sets the code-object properties read from its metadata.
    #[must_use]
    pub fn code_object(mut self, wave_size: u32, lds_bytes: u32) -> Self {
        self.wave_size = wave_size;
        self.lds_bytes = lds_bytes;
        self
    }

    /// Declares a workspace.
    #[must_use]
    pub fn workspace(mut self, w: WorkspaceReq) -> Self {
        self.workspaces.push(w);
        self
    }

    /// Sets the static launch model.
    #[must_use]
    pub fn launches(mut self, l: LaunchModel) -> Self {
        self.launches = l;
        self
    }

    /// Sets the promotion state.
    #[must_use]
    pub fn promoted(mut self, p: Promotion) -> Self {
        self.promotion = p;
        self
    }

    /// True when every constraint holds. `Err` names the first that did not, and how many held
    /// before it — which is what orders candidates by "closest" in a refusal.
    pub fn matches(&self, shape: &ShapeKeys) -> std::result::Result<(), (String, usize)> {
        for (i, c) in self.constraints.iter().enumerate() {
            if let Err(reason) = c.check(shape) {
                return Err((reason, i));
            }
        }
        Ok(())
    }

    /// Whether this implementation can legally be recorded into a captured graph.
    pub fn capturable(&self) -> bool {
        self.graph_safe && !self.may_sync
    }

    /// Why it cannot be captured, for the refusal message.
    pub fn capture_refusal(&self) -> Option<String> {
        match (self.graph_safe, self.may_sync) {
            (true, false) => None,
            (false, true) => Some("declared !graph_safe and may_sync".to_string()),
            (false, false) => Some("declared !graph_safe".to_string()),
            (true, true) => Some("declared may_sync".to_string()),
        }
    }
}

/// An index into a [`Registry`]. Cheap, stable for the life of the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ImplId(pub usize);

impl fmt::Display for ImplId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// One candidate's verdict in a selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateTrace {
    /// The implementation considered.
    pub impl_name: String,
    /// Whether every constraint held.
    pub ok: bool,
    /// `ok`, or the constraint that refused it, in the reference's wording.
    pub reason: String,
    /// How many constraints held before the refusal (all of them when `ok`).
    pub satisfied: usize,
}

/// A candidate that did *not* match, as carried in a [`NoImplementation`] error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateMiss {
    /// The implementation considered.
    pub impl_name: String,
    /// The constraint that refused it.
    pub reason: String,
    /// How many constraints held before the refusal.
    pub satisfied: usize,
}

/// The structured record of one selection: what was asked, every candidate, and what was chosen.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionTrace {
    /// The op asked for.
    pub op: String,
    /// The geometry asked for.
    pub query: ShapeKeys,
    /// Every implementation carrying that op, in registry order, with its verdict.
    pub candidates: Vec<CandidateTrace>,
}

impl SelectionTrace {
    /// The names that matched, in registry order.
    pub fn chosen(&self) -> Vec<&str> {
        self.candidates
            .iter()
            .filter(|c| c.ok)
            .map(|c| c.impl_name.as_str())
            .collect()
    }
}

/// Every implementation the engine can bind, and the one place a route is chosen.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Registry {
    implementations: Vec<Implementation>,
    /// A digest of the launch-rule tables this registry was derived from, or empty for a registry
    /// built by hand (a test's). See [`Registry::policy_digest`].
    policy_digest: String,
    /// Each committed table's own digest, by file name, so a route can be keyed on the tables it
    /// could actually bind rather than on all of them. Empty for a hand-built registry, which
    /// then falls back to the whole-tier digest above.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    policy_tables: BTreeMap<String, String>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The digest of the launch-rule tables behind this registry, if it was derived from them.
    ///
    /// The rules are the tier's POLICY -- which kernel, tile and split a shape takes -- and an
    /// autotune record is a measurement over the variant set the policy allowed when it ran. A
    /// policy edit can make a variant legal that was never measured (`lm_head` took the decode
    /// tile on 2026-09-19 and the cache went on binding the folded record it had), so the plan
    /// input hash carries this and a policy edit invalidates the cache loudly instead of leaving
    /// stale records to bind quietly. Empty means "not derived from the committed tables", which
    /// is a hand-built registry in a test and hashes as such.
    #[must_use]
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }

    /// Records which policy this registry was derived from. Set once, where it is built.
    pub fn set_policy_digest(&mut self, digest: impl Into<String>) {
        self.policy_digest = digest.into();
    }

    /// Records the committed tables themselves, so routes can be keyed per plugin.
    ///
    /// Sets [`Registry::policy_digest`] to the digest over all of them, which is what `explain`
    /// prints and what a whole-tier comparison wants, and keeps each table's own digest for
    /// [`Registry::policy_digest_for`].
    pub fn set_policy_tables(&mut self, tables: &[(&str, &str)]) {
        self.policy_digest = crate::hash::digest_of(&tables);
        self.policy_tables = tables
            .iter()
            .map(|(name, text)| ((*name).to_owned(), crate::hash::digest_of(text)))
            .collect();
    }

    /// The policy digest for a route whose candidates live in these plugins.
    ///
    /// # Why this is not the whole-tier digest
    ///
    /// Every record in the cache carries a policy digest so that editing a band re-keys the
    /// records measured under the old one (I13, and I17 when two tables turned out to be missing
    /// from it). With ONE digest over all seven tables that is true and indiscriminate: a
    /// one-line change to `dflash_gfx1201.toml` re-keys all 85 records and costs a full sweep,
    /// 84 of which no `dflash` row could have reached. A record is invalidated by the policy its
    /// OWN route could bind, and that is the set of tables owning the plugins among its
    /// candidates.
    ///
    /// A plugin no table claims is keyed on the WHOLE policy. Over-keying costs a re-sweep;
    /// under-keying serves a record measured under a policy that has since changed, and only one
    /// of those is recoverable. `plugin_table_map_covers_every_committed_plugin` is what keeps
    /// the case theoretical.
    #[must_use]
    pub fn policy_digest_for<'p>(&self, plugins: impl IntoIterator<Item = &'p str>) -> String {
        if self.policy_tables.is_empty() {
            return self.policy_digest.clone();
        }
        let mut want: BTreeMap<&str, &str> = BTreeMap::new();
        for plugin in plugins.into_iter().flat_map(|head| {
            std::iter::once(head).chain(saber_kernels::companion_plugins(head).iter().copied())
        }) {
            match saber_kernels::rule_table_for_plugin(plugin) {
                Some(file) => {
                    if let Some((k, v)) = self.policy_tables.get_key_value(file) {
                        want.insert(k.as_str(), v.as_str());
                    }
                }
                // Unclaimed: fall back to everything, which is the conservative direction.
                None => {
                    return self.policy_digest.clone();
                }
            }
        }
        crate::hash::digest_of(&want)
    }

    /// The plugins among a candidate list, for [`Registry::policy_digest_for`].
    #[must_use]
    pub fn plugins_of(&self, candidates: &[ImplId]) -> BTreeSet<&str> {
        candidates
            .iter()
            .filter_map(|id| self.get(*id))
            .map(|imp| imp.plugin.as_str())
            .collect()
    }

    /// Adds an implementation and returns its id.
    pub fn insert(&mut self, imp: Implementation) -> ImplId {
        self.implementations.push(imp);
        ImplId(self.implementations.len() - 1)
    }

    /// Builder-style [`Registry::insert`].
    #[must_use]
    pub fn with(mut self, imp: Implementation) -> Self {
        self.insert(imp);
        self
    }

    /// All implementations, in insertion order.
    pub fn implementations(&self) -> &[Implementation] {
        &self.implementations
    }

    /// Looks up by id.
    pub fn get(&self, id: ImplId) -> Option<&Implementation> {
        self.implementations.get(id.0)
    }

    /// Looks up by entry-point name.
    pub fn by_name(&self, name: &str) -> Option<(ImplId, &Implementation)> {
        self.implementations
            .iter()
            .enumerate()
            .find(|(_, i)| i.name == name)
            .map(|(i, imp)| (ImplId(i), imp))
    }

    /// Every op this registry implements, sorted.
    pub fn ops(&self) -> Vec<&str> {
        let mut ops: Vec<&str> = self.implementations.iter().map(|i| i.op.as_str()).collect();
        ops.sort_unstable();
        ops.dedup();
        ops
    }

    /// Every candidate for an op at a geometry, with the reason each was or was not chosen.
    ///
    /// Asks nothing and records nothing — it re-runs the same walk [`Registry::select`] does.
    pub fn explain(&self, op: &str, shape: &ShapeKeys) -> SelectionTrace {
        let candidates = self
            .implementations
            .iter()
            .filter(|i| i.op == op)
            .map(|i| match i.matches(shape) {
                Ok(()) => CandidateTrace {
                    impl_name: i.name.clone(),
                    ok: true,
                    reason: "ok".to_string(),
                    satisfied: i.constraints.len(),
                },
                Err((reason, satisfied)) => CandidateTrace {
                    impl_name: i.name.clone(),
                    ok: false,
                    reason,
                    satisfied,
                },
            })
            .collect();
        SelectionTrace {
            op: op.to_string(),
            query: shape.clone(),
            candidates,
        }
    }

    /// Every implementation whose necessary conditions hold at this geometry.
    ///
    /// * **Zero** matches is an error carrying every candidate and the constraint that refused it,
    ///   closest first.
    /// * **Several** matches are all returned: the tie is broken by a measurement in the autotune
    ///   stage, never by a cost model and never by registry order.
    pub fn select(&self, op: &str, shape: &ShapeKeys) -> Result<Vec<ImplId>> {
        let hits: Vec<ImplId> = self
            .implementations
            .iter()
            .enumerate()
            .filter(|(_, i)| i.op == op && i.matches(shape).is_ok())
            .map(|(i, _)| ImplId(i))
            .collect();
        if !hits.is_empty() {
            return Ok(hits);
        }
        let mut misses: Vec<CandidateMiss> = self
            .implementations
            .iter()
            .filter(|i| i.op == op)
            .filter_map(|i| {
                i.matches(shape)
                    .err()
                    .map(|(reason, satisfied)| CandidateMiss {
                        impl_name: i.name.clone(),
                        reason,
                        satisfied,
                    })
            })
            .collect();
        // Closest first: the candidate that got furthest through its constraints is the one whose
        // refusal a reader most likely wants to act on.
        misses.sort_by(|a, b| {
            b.satisfied
                .cmp(&a.satisfied)
                .then(a.impl_name.cmp(&b.impl_name))
        });
        Err(PlanError::NoImplementation(NoImplementation {
            op: op.to_string(),
            query: shape.clone(),
            misses,
        }))
    }

    /// Canonical digest of the **whole** registry. Printed by `explain` as the registry's
    /// identity; **not** what the plan's input hash keys on — see
    /// [`Registry::bindable_fingerprint`].
    pub fn fingerprint(&self) -> String {
        crate::hash::digest_of(self)
    }

    /// Canonical digest of the rows a plan can actually **bind**: the union of the candidate sets
    /// the constraint match returns over every (op, geometry) the model resolves.
    ///
    /// # Why the plan hashes this and not the registry
    ///
    /// The input hash is what every autotune key carries, so anything inside it invalidates every
    /// record when it changes. Hashing the whole registry means a plugin added **for another
    /// tier** — eight Paro rows no op of an NVFP4 model can select, 2026-09-19 — moves every
    /// key and makes a freshly measured cache stale, which is a re-sweep of 44 shapes to describe
    /// a change that cannot reach any of them.
    ///
    /// The set is the right granularity rather than a convenient one: a row that is a candidate
    /// can be bound, so its constraints, workspaces, launch count and capture bits are all part
    /// of what a record means; a row that is not a candidate cannot be bound by this model, and
    /// if a later edit makes it one, the *set* changes and the hash moves with it.
    ///
    /// Rows are ordered by name so the digest does not depend on insertion order.
    pub fn bindable_fingerprint(&self, candidates: &[ImplId]) -> String {
        let mut rows: Vec<&Implementation> =
            candidates.iter().filter_map(|id| self.get(*id)).collect();
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        rows.dedup_by(|a, b| a.name == b.name);
        crate::hash::digest_of(&rows)
    }
}
