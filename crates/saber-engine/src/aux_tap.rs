//! The aux tap: the five target-layer residual streams a DFlash2 proposal starts from, copied out
//! of the step's own op list.
//!
//! [`crate::draft_inputs`] says WHERE every row of a proposal goes; this module says where the
//! proposal's one non-weight INPUT comes from. `combine_hidden_states` folds five of the target's
//! hidden states into the drafter's projection, and the reference server collects each of them as
//! `hidden_states + residual` after layer `id - 1` — which is exactly the value
//! `layer.{id}.input_layernorm` publishes as `l{id}.resid.a`, the same edge `saber engine`'s
//! per-layer probe digests. So the tap points need no new law and no new name
//! ([`crate::draft_inputs::AUX_TAPS`], gated against the drafter checkpoint's own
//! `dflash_config.target_layer_ids`).
//!
//! # Why this is five copies inside the step and not a read afterwards
//!
//! **The body plan runs with slot reuse ON.** `PlanOptions::reuse_layer_slots` aliases the
//! per-layer intermediates onto a handful of arena buffers, so `l5.resid.a` and `l61.resid.a` are
//! the same address and at the end of a step that address holds layer 63's value. Reading the five
//! edges after the step returns one layer's answer five times — silently, because every read is in
//! range and every byte is a real residual stream. The same fact is why
//! [`crate::device`]'s probe walks the body in SLICES.
//!
//! The alternative is a body plan with `reuse_layer_slots` off, which is an arena this card does
//! not have (64 layers x rows x hidden bf16 of live intermediates). So the copy happens **at the
//! tap point**, as an op in the step's own list: five more dispatches, five more in the declared
//! budget (`CLAUDE.md` rule 7), five more nodes in the captured graph.
//!
//! # The destination is a column block, which is why it is not `row_gather`
//!
//! The drafter reads ONE tensor: `aux.cat`, `[rows, 5 * hidden]` bf16, tap `k` occupying columns
//! `[k * hidden, (k + 1) * hidden)`. So each copy writes `rows` rows of `hidden` elements whose
//! DESTINATION rows are `5 * hidden` apart while its SOURCE rows are `hidden` apart. The head
//! plugin's `row_gather` packs its output rows contiguously (`out + r * cols`) — it selects rows
//! out of a wide source into a narrow destination, which is the opposite arrangement — so the tap
//! is its own op, `head::HeadOp::RowCopy`, with the two strides independent and a destination base
//! that carries the block offset.

use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};

use crate::draft_inputs::{aux_tap_edges, AUX_TAPS};
use crate::forward::ForwardError;

/// Where the tap's copies are issued, relative to the edge each one reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapAt {
    /// At the tap point: immediately after `layer.{id}.input_layernorm`, the op that publishes
    /// `l{id}.resid.a`. This is the law.
    Point,
    /// ONE LAYER LATE — the negative control, and it is a control rather than a different bug
    /// because it differs from the law in exactly the way a mis-sited tap would: the copy runs
    /// after `layer.{id + 1}.input_layernorm`, by which time slot reuse has overwritten the buffer
    /// with the next layer's value. Every launch still issues, every byte is still a real residual
    /// stream, and the drafter would project a plausible five-layer concatenation of the wrong
    /// five layers.
    OneLayerLate,
}

/// The tap as a plan: the edges, the geometry and the launch it issues. No device memory.
#[derive(Clone, Debug)]
pub struct AuxTapPlan {
    /// Model hidden size — the width of one tap and the source row stride.
    pub hidden: u32,
    /// The target layers, in the order the drafter's projection concatenates them.
    pub layers: Vec<u32>,
    /// `l{id}.resid.a` per tap, in the same order.
    pub edges: Vec<String>,
}

impl AuxTapPlan {
    /// The committed tap list at this model's hidden size.
    #[must_use]
    pub fn new(hidden: u32) -> Self {
        AuxTapPlan {
            hidden,
            layers: AUX_TAPS.to_vec(),
            edges: aux_tap_edges(),
        }
    }

    /// A separately owned feature plane, in an explicitly requested checkpoint order.
    pub fn for_layers(hidden: u32, layers: &[u32]) -> Result<Self, ForwardError> {
        if hidden == 0
            || layers.is_empty()
            || layers.iter().any(|x| *x == 0)
            || layers.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(ForwardError::Device(
                "auxiliary layers must be positive, unique and increasing".into(),
            ));
        }
        Ok(Self {
            hidden,
            layers: layers.to_vec(),
            edges: layers.iter().map(|l| format!("l{l}.resid.a")).collect(),
        })
    }

    /// Copies one step issues: one per tapped layer.
    #[must_use]
    pub fn launches(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Elements in one row of `aux.cat`: `5 * hidden`, and the destination row stride.
    #[must_use]
    pub fn row_elems(&self) -> u64 {
        self.layers.len() as u64 * u64::from(self.hidden)
    }

    /// Bytes `aux.cat` holds at `rows` rows, bf16.
    #[must_use]
    pub fn bytes(&self, rows: u32) -> u64 {
        self.row_elems() * u64::from(rows) * 2
    }

    /// Byte offset of tap `k`'s column block within a row.
    #[must_use]
    pub fn block_offset(&self, k: usize) -> u64 {
        k as u64 * u64::from(self.hidden) * 2
    }

    /// The op name tap `k`'s copy is issued AFTER, under `at`.
    ///
    /// # Errors
    /// A control asked for past the model's last layer — which would be a control that silently
    /// did not fire.
    pub fn after(&self, k: usize, at: TapAt, layers_in_model: u32) -> Result<String, ForwardError> {
        let base = *self.layers.get(k).ok_or_else(|| {
            ForwardError::Device(format!(
                "tap {k}: this plan taps {} layers",
                self.layers.len()
            ))
        })?;
        let id = match at {
            TapAt::Point => base,
            TapAt::OneLayerLate => base + 1,
        };
        if id >= layers_in_model {
            return Err(ForwardError::Device(format!(
                "the aux tap's control would be issued after `layer.{id}.input_layernorm` and this \
                 model has {layers_in_model} layer(s): a control whose op does not exist is a \
                 control that never runs"
            )));
        }
        Ok(format!("layer.{id}.input_layernorm"))
    }

    /// The op name tap `k`'s TRUTH is read after — always the tap point, whatever `at` the copies
    /// were issued under, because the control is a claim about the COPY and not about the probe.
    ///
    /// # Errors
    /// A tap index this plan does not have.
    pub fn truth_after(&self, k: usize) -> Result<String, ForwardError> {
        self.after(k, TapAt::Point, u32::MAX)
    }

    /// One copy's descriptor at `rows` rows. Every tap shares it: they differ only in the source
    /// edge and the destination base, both of which are bindings.
    ///
    /// # Errors
    /// A shape the head plugin's rules cannot serve.
    pub fn descriptor(
        &self,
        rows: u32,
        rules: &HeadRules,
    ) -> Result<saber_kernels::rules::LaunchDescriptor, ForwardError> {
        head::resolve(
            HeadOp::RowCopy {
                dst_stride: self.row_elems() as i64,
            },
            HeadShape {
                rows,
                cols: self.hidden,
                src_stride: i64::from(self.hidden),
                vocab: (0, 0),
            },
            rules,
        )
        .map_err(|e| ForwardError::Device(format!("the aux tap at {rows} rows: {e}")))
    }
}

/// One tap's verdict: the copied bytes against the engine's own probe of the edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapVerdict {
    /// The target layer.
    pub layer: u32,
    /// The edge the copy read.
    pub edge: String,
    /// Bytes compared: `rows * hidden * 2`.
    pub bytes: usize,
    /// Bytes that differ.
    pub differing: usize,
}

impl TapVerdict {
    /// Whether this tap carried the edge exactly.
    #[must_use]
    pub fn exact(&self) -> bool {
        self.differing == 0 && self.bytes > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_checkpoint_taps_preserve_order_and_geometry() {
        let p = AuxTapPlan::for_layers(5120, &[1, 16, 31, 46, 61]).unwrap();
        assert_eq!(
            p.edges,
            [
                "l1.resid.a",
                "l16.resid.a",
                "l31.resid.a",
                "l46.resid.a",
                "l61.resid.a"
            ]
        );
        assert_eq!(p.bytes(16), 16 * 5 * 5120 * 2);
        assert_eq!(AuxTapPlan::new(5120).layers, AUX_TAPS);
        for layers in [&[][..], &[0, 1], &[1, 1], &[2, 1]] {
            assert!(AuxTapPlan::for_layers(5120, layers).is_err());
        }
        assert!(AuxTapPlan::for_layers(0, &[1]).is_err());
    }

    /// The geometry the task states, and the stride that makes it one buffer.
    ///
    /// Source rows are `hidden` apart and destination rows are `5 * hidden` apart, which is the
    /// whole reason `row_gather` cannot serve this: its destination is packed.
    #[test]
    fn the_tap_writes_a_column_block_of_one_concatenation() {
        let p = AuxTapPlan::new(5120);
        assert_eq!(p.layers, vec![5, 19, 33, 47, 61]);
        assert_eq!(
            p.edges,
            vec![
                "l5.resid.a",
                "l19.resid.a",
                "l33.resid.a",
                "l47.resid.a",
                "l61.resid.a"
            ]
        );
        assert_eq!(p.launches(), 5);
        assert_eq!(p.row_elems(), 25_600);
        // b1s8 and b4s8, the two shapes the gate captures.
        assert_eq!(p.bytes(8), 409_600);
        assert_eq!(p.bytes(32), 1_638_400);
        for (k, off) in [(0u64, 0u64), (1, 10_240), (4, 40_960)] {
            assert_eq!(p.block_offset(k as usize), off);
        }
        let rules = HeadRules::prod_head().expect("the head rules parse");
        let d = p.descriptor(8, &rules).expect("the tap resolves at b1s8");
        assert_eq!(d.kernel, "saber_row_copy_bf16");
        assert_eq!(d.grid, [8, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert_eq!(
            d.arg("src_stride").map(|s| s.value.clone()),
            Some(saber_kernels::rules::ArgValue::I64 { ival: 5120 })
        );
        assert_eq!(
            d.arg("dst_stride").map(|s| s.value.clone()),
            Some(saber_kernels::rules::ArgValue::I64 { ival: 25_600 })
        );
    }

    /// The copy site is the tap point, the control is one layer on, and the TRUTH point does not
    /// move with the control.
    ///
    /// A control that also moved the probe would compare the late copy with the late edge and
    /// PASS, which is the way this control could have been written wrong.
    #[test]
    fn the_control_moves_the_copy_and_not_the_probe() {
        let p = AuxTapPlan::new(5120);
        for (k, id) in [(0usize, 5u32), (1, 19), (2, 33), (3, 47), (4, 61)] {
            assert_eq!(
                p.after(k, TapAt::Point, 64).expect("a tap point"),
                format!("layer.{id}.input_layernorm")
            );
            assert_eq!(
                p.after(k, TapAt::OneLayerLate, 64).expect("the control"),
                format!("layer.{}.input_layernorm", id + 1)
            );
            assert_eq!(
                p.truth_after(k).expect("the truth point"),
                format!("layer.{id}.input_layernorm")
            );
        }
        // A control past the model's last layer is refused rather than silently not firing.
        let e = p
            .after(4, TapAt::OneLayerLate, 62)
            .expect_err("layer 62 does not exist in a 62-layer model");
        assert!(e.to_string().contains("never runs"), "{e}");
    }
}
