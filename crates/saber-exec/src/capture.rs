//! Capture a sequence of real launches into one HIP graph, and replay it.
//!
//! This is the execution plane's core. Everything else in this crate builds one dispatch; this
//! builds the thing a step actually is — 1084 of them, recorded once and replayed per token.
//!
//! > Plan → captured HIP graphs per bucket (batch × spec width × ctx band). [...] Launch count is
//! > a budgeted, gated metric — a hipGraph node costs ~2.1 µs and prod runs **1084 launches/step**,
//! > so launch-count regressions fail CI like latency regressions.
//! >
//! > — ARCHITECTURE.md, *4. Execution plane*
//!
//! # The three rules this module exists to enforce
//!
//! 1. **Every workspace is resolved before the capture opens.** An allocation inside a capture is
//!    a capture error; a `hipMemset` inside one is worse, because it *succeeds* and becomes a node
//!    that re-zeroes protocol state on every replay. So [`CapturedBucket::capture`] runs
//!    [`Launcher::prepare_workspaces`] over the whole sequence first — which also sizes the Scratch
//!    arena to the **maximum over the sequence**, so no slab grows (and no address moves) once the
//!    capture is open.
//! 2. **The node count is asserted, not observed.** `hipGraphGetNodes` says how many nodes the
//!    capture produced; native nodes must equal the declared `n_launches`. RCCL mixed graphs
//!    additionally declare one event-record node per collective, with node kinds audited. A mismatch names
//!    the op and refuses (rule 7). This is the check that makes a launch budget a budget.
//! 3. **A `!graph_safe` op cannot be captured — by type.** [`CapturableOp`] is the only thing
//!    [`CapturedBucket::capture`] accepts, and it cannot be built without a [`GraphSafe`] proof.
//!    The bits themselves are the planner's: `saber-plan` carries `graph_safe` and `may_sync` per
//!    implementation, and this is the boundary where that claim enters the execution plane.
//!
//! # What is deliberately not here
//!
//! No bucket selection, no plan, no scheduling. A `CapturedBucket` is a recorded sequence and an
//! instantiated graph; *which* sequence belongs to which bucket is `saber-plan`'s answer, and the
//! backend that replays it is an interface `RetainedReplay` will also implement (P6).

use std::sync::Arc;

use saber_hip::{Graph, GraphExec, Stream, StreamCaptureMode};
use saber_kernels::LaunchDescriptor;

use crate::counters::CaptureCounters;
use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};

/// Proof that an op's implementation may legally be recorded into a graph.
///
/// Zero-sized and unforgeable outside this module's constructor, which is the point: "a
/// `!graph_safe` op cannot be bound into a captured bucket because the type will not permit it"
/// (ARCHITECTURE.md, *Make illegal states unrepresentable*). `saber-plan` has the same proof over
/// the same two bits; this is the same rule restated where the launch happens, because the
/// execution plane must not have to trust that someone else checked.
#[derive(Debug, Clone, Copy)]
pub struct GraphSafe(());

impl GraphSafe {
    /// The planner's two bits. `None` when the op may not be captured.
    ///
    /// `graph_safe` and `may_sync` are different claims and an implementation asserting both is
    /// telling the planner it is capture-legal *and* that it may synchronise — which is how a
    /// bucket ends up captured with a device-wide barrier in it. Both are checked.
    #[must_use]
    pub fn from_flags(graph_safe: bool, may_sync: bool) -> Option<GraphSafe> {
        if graph_safe && !may_sync {
            Some(GraphSafe(()))
        } else {
            None
        }
    }

    /// Why a pair of bits was refused, for the message.
    #[must_use]
    pub fn refusal(graph_safe: bool, may_sync: bool) -> Option<&'static str> {
        match (graph_safe, may_sync) {
            (true, false) => None,
            (false, true) => Some("declared !graph_safe and may_sync"),
            (false, false) => Some("declared !graph_safe"),
            (true, true) => Some("declared may_sync"),
        }
    }
}

/// One op of a capture: a descriptor, its bindings, and the proof it may be recorded.
///
/// There is deliberately no `From<(LaunchDescriptor, ArgBindings)>`. A descriptor says what to
/// launch; it does not say whether the implementation behind it is capture-legal, because that is
/// a property of the *implementation* and lives in the planner's registry. Accepting a bare tuple
/// would make the execution plane answer a control-plane question by assuming the good case.
pub struct CapturableOp<'a> {
    /// The op's dispatch sequence. One descriptor for almost every op; the reference's paged attention
    /// decode is two (the split kernel and its combine) and the head declares both.
    descs: &'a [LaunchDescriptor],
    bindings: &'a ArgBindings,
    _proof: GraphSafe,
}

impl<'a> CapturableOp<'a> {
    /// Bind an op into a capture, given the planner's proof.
    #[must_use]
    pub fn new(
        desc: &'a LaunchDescriptor,
        bindings: &'a ArgBindings,
        proof: GraphSafe,
    ) -> CapturableOp<'a> {
        CapturableOp {
            descs: std::slice::from_ref(desc),
            bindings,
            _proof: proof,
        }
    }

    /// An op that is MORE than one dispatch, as its whole sequence.
    ///
    /// The launch budget is per **op**, not per dispatch: `descs[0]` declares how many launches
    /// the op issues and [`Launcher::launch_op`] checks the sequence against that number. Binding
    /// the descriptors one at a time instead would ask each of them to declare its own count, and
    /// the paged attention decode — one op, two dispatches, `n_launches = 2` on the head — would
    /// be refused for issuing one. This is the same rule ws-h2 moved from per-descriptor to
    /// per-op in the launcher, restated where a capture binds them.
    ///
    /// # Errors
    /// An empty sequence: an op with no dispatch is not an op, and the node-count assertion below
    /// could not name it.
    pub fn sequence(
        descs: &'a [LaunchDescriptor],
        bindings: &'a ArgBindings,
        proof: GraphSafe,
    ) -> Result<CapturableOp<'a>> {
        if descs.is_empty() {
            return Err(Error::Refused(
                "a capturable op needs at least one descriptor".to_owned(),
            ));
        }
        Ok(CapturableOp {
            descs,
            bindings,
            _proof: proof,
        })
    }

    /// Bind an op from the planner's raw bits, refusing (with a counter) when they say no.
    pub fn from_flags(
        desc: &'a LaunchDescriptor,
        bindings: &'a ArgBindings,
        graph_safe: bool,
        may_sync: bool,
        counters: &CaptureCounters,
    ) -> Result<CapturableOp<'a>> {
        match GraphSafe::from_flags(graph_safe, may_sync) {
            Some(proof) => Ok(CapturableOp::new(desc, bindings, proof)),
            None => {
                counters.hit_refused_not_graph_safe();
                Err(Error::NotCapturable {
                    op: desc.op.clone(),
                    kernel: desc.kernel.clone(),
                    reason: GraphSafe::refusal(graph_safe, may_sync)
                        .unwrap_or("not capturable")
                        .to_owned(),
                })
            }
        }
    }

    /// The head descriptor: the one whose `n_launches` is the op's budget.
    #[must_use]
    pub fn descriptor(&self) -> &LaunchDescriptor {
        &self.descs[0]
    }

    /// Every descriptor this op dispatches, in order.
    #[must_use]
    pub fn descriptors(&self) -> &[LaunchDescriptor] {
        self.descs
    }
}

impl std::fmt::Debug for CapturableOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CapturableOp({})",
            self.descs
                .iter()
                .map(|d| d.kernel.as_str())
                .collect::<Vec<_>>()
                .join(" + ")
        )
    }
}

// HIP enum values confirmed against hip_runtime_api.h: Kernel=0, EventRecord=7.
fn rccl_graph_matches(types: &[i32], kernels: usize, collectives: usize) -> bool {
    types.len() == kernels + collectives
        && types.iter().filter(|&&t| t == 0).count() == kernels
        && types.iter().filter(|&&t| t == 7).count() == collectives
}

/// A recorded sequence of launches, instantiated and ready to replay.
///
/// Holds the graph as well as the executable: HIP allows the graph to be freed after
/// instantiation, but keeping it is what lets [`CapturedBucket::node_count`] stay answerable
/// after the fact, and it costs nothing at the sizes a bucket reaches.
pub struct CapturedBucket {
    label: String,
    graph: Graph,
    exec: GraphExec,
    nodes: usize,
    synchronization_nodes: usize,
    declared: u32,
    ops: usize,
    counters: Arc<CaptureCounters>,
}

impl CapturedBucket {
    /// Record `ops` into one graph and instantiate it.
    ///
    /// The whole sequence shares one `Launcher`, which is what makes the Scratch arena aliasable
    /// across the sequence and the Persistent regions shared by it — exactly as they will be in a
    /// real bucket. The launcher's workspaces are prepared for **every** op before the capture
    /// opens, so nothing allocates, grows or memsets while it is open.
    pub fn capture(
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        ops: &[CapturableOp<'_>],
        stream: &Stream,
        label: impl Into<String>,
        counters: &Arc<CaptureCounters>,
    ) -> Result<CapturedBucket> {
        let label = label.into();
        if ops.is_empty() {
            return Err(Error::Refused(format!(
                "capture `{label}`: an empty graph is not a bucket"
            )));
        }

        // ---- everything that allocates, happens here ----------------------------------------
        // Every DESCRIPTOR, not every op: a two-dispatch op's second kernel has workspaces of its
        // own (the attention combine reads the split scratch the decode kernel wrote), and a
        // workspace first seen inside the capture would allocate while it is open.
        for op in ops {
            for desc in op.descs {
                launcher.prepare_workspaces(ctx, desc, op.bindings, stream)?;
            }
        }
        // The Scratch arena is now at its maximum over the sequence and every Persistent region
        // has been zeroed exactly once. Drain anything still in flight before the capture opens.
        ctx.hip.sync_stream(stream)?;

        // Per OP: the head declares what the whole sequence issues.
        let declared: u32 = ops.iter().map(|o| o.descs[0].n_launches).sum();

        // ---- capture ------------------------------------------------------------------------
        // ThreadLocal, and correct by construction rather than by convention: every HIP call in
        // this process runs on the one HipThread, so there is nothing else to serialise against.
        ctx.hip
            .begin_capture(stream, StreamCaptureMode::ThreadLocal)?;
        let mut issued = 0u32;
        let mut first_error = None;
        for op in ops {
            let refs: Vec<&LaunchDescriptor> = op.descs.iter().collect();
            match launcher.launch_op(ctx, &refs, op.bindings, stream) {
                Ok(report) => issued += report.launches,
                Err(e) => {
                    // Do NOT return yet: a stream left in capture mode poisons every later use of
                    // it, so the capture is always closed before the error is raised.
                    first_error = Some((op.descs[0].op.clone(), op.descs[0].kernel.clone(), e));
                    break;
                }
            }
        }
        let graph = ctx.hip.end_capture(stream);
        if let Some((op, kernel, e)) = first_error {
            counters.hit_refused_launch_count();
            let _ = graph;
            return Err(Error::Refused(format!(
                "capture `{label}`: op `{op}` (`{kernel}`) failed while the capture was open: {e}"
            )));
        }
        let graph = graph?;

        // ---- rule 7: the budget is asserted, not observed ------------------------------------
        let nodes = ctx.hip.graph_node_count(&graph)?;
        let rccl_calls = ops
            .iter()
            .flat_map(|o| o.descs)
            .filter(|d| d.plugin == saber_kernels::saber::RCCL_PLUGIN)
            .count();
        if rccl_calls > 0 {
            // RCCL 2.30.4 mixed eager/graph mode records one completion event per call.
            // Count kernel dispatches and synchronization nodes separately; never relax the
            // launch budget or allow arbitrary library-inserted graph work.
            let types = ctx.hip.graph_node_types(&graph)?;
            if issued != declared
                || types.len() != nodes
                || !rccl_graph_matches(&types, declared as usize, rccl_calls)
            {
                counters.hit_refused_launch_count();
                return Err(Error::Refused(format!("capture `{label}`: expected {declared} kernel nodes and {rccl_calls} RCCL event-record nodes; issued {issued}, graph node kinds {types:?}")));
            }
        } else if nodes as u32 != declared || issued != declared {
            counters.hit_refused_launch_count();
            return Err(Error::LaunchBudget {
                label,
                declared,
                issued,
                nodes,
                ops: ops
                    .iter()
                    .map(|o| format!("{} x{}", o.descs[0].kernel, o.descs[0].n_launches))
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }

        let exec = ctx.hip.instantiate(&graph)?;
        counters.hit_captured();
        tracing::debug!(label = %label, nodes, ops = ops.len(), "bucket captured");
        Ok(CapturedBucket {
            label,
            graph,
            exec,
            nodes,
            synchronization_nodes: rccl_calls,
            declared,
            ops: ops.len(),
            counters: Arc::clone(counters),
        })
    }

    /// Submit the graph. Returns as soon as it is queued; the caller orders completion.
    pub fn replay(&self, ctx: &LaunchCtx<'_>, stream: &Stream) -> Result<()> {
        ctx.hip.graph_launch(&self.exec, stream)?;
        self.counters.hit_replayed();
        Ok(())
    }

    /// Nodes the capture produced, including explicitly audited RCCL completion events.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Actual compute nodes, after separately validating external-library completion events.
    #[must_use]
    pub fn kernel_node_count(&self) -> usize {
        self.nodes - self.synchronization_nodes
    }

    /// Explicitly budgeted synchronization nodes in addition to kernel launches.
    #[must_use]
    pub fn synchronization_node_count(&self) -> usize {
        self.synchronization_nodes
    }

    /// The launch count the ops declared.
    #[must_use]
    pub fn declared_launches(&self) -> u32 {
        self.declared
    }

    /// How many ops were recorded.
    #[must_use]
    pub fn ops(&self) -> usize {
        self.ops
    }

    /// The label this bucket was captured under.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The captured graph, before instantiation.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl std::fmt::Debug for CapturedBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CapturedBucket({}, {} ops, {} nodes)",
            self.label, self.ops, self.nodes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_graph_budget_rejects_missing_extra_or_unknown_nodes() {
        assert!(super::rccl_graph_matches(&[0, 7, 0, 0, 7], 3, 2));
        assert!(!super::rccl_graph_matches(&[0, 7, 0, 0], 3, 2));
        assert!(!super::rccl_graph_matches(&[0, 7, 0, 0, 7, 0], 3, 2));
        assert!(!super::rccl_graph_matches(&[0, 7, 0, 0, 3], 3, 2));
        assert!(!super::rccl_graph_matches(&[0, 0, 0, 0, 7], 3, 2));
    }

    #[test]
    fn graph_safety_needs_both_bits_to_agree() {
        assert!(GraphSafe::from_flags(true, false).is_some());
        assert!(GraphSafe::from_flags(false, false).is_none());
        assert!(GraphSafe::from_flags(true, true).is_none());
        assert!(GraphSafe::from_flags(false, true).is_none());
        // The refusal says WHICH bit, because "not capturable" is not something a reader can act
        // on and `explain` has to print what forced a bucket eager.
        assert_eq!(GraphSafe::refusal(true, false), None);
        assert_eq!(GraphSafe::refusal(true, true), Some("declared may_sync"));
        assert_eq!(
            GraphSafe::refusal(false, false),
            Some("declared !graph_safe")
        );
        assert_eq!(
            GraphSafe::refusal(false, true),
            Some("declared !graph_safe and may_sync")
        );
    }

    #[test]
    fn a_proof_cannot_be_forged_from_outside() {
        // `GraphSafe` has a private unit field, so the only way to one is `from_flags`. This is a
        // compile-time property; the assertion here is that the constructor is the only door.
        let proof = GraphSafe::from_flags(true, false).expect("capturable");
        let _ = proof; // and it is Copy, so one proof can bind a whole sequence
        let copied = proof;
        let _ = (proof, copied);
    }
}
