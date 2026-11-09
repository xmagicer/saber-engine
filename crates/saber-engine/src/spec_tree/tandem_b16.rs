//! Frozen conditional-edge calibration for Tandem B16 and the nine-node verifier.
// Depths inside this module are zero-based. Nodes returned to Saber use one-based depth.
use super::{np_sum_f32, Frontier, LatticeView, Node, SpecTree, SpecTreeForm, TreeError, TreePlan};

const K: usize = 16;
const DEPTH: usize = 15;
const FEATURES: usize = 14;

mod model;
use model::{LEXICON, WEIGHTS};

const TARGET_TIER: [u8; 32] = [
    0x9c, 0x2e, 0xfc, 0xec, 0x80, 0xb6, 0x5a, 0x98, 0xaf, 0xde, 0x52, 0x06, 0x7b, 0x0f, 0x2b, 0xbc,
    0xd0, 0x65, 0x83, 0xb4, 0x70, 0xc9, 0x8b, 0x0b, 0xbc, 0xe7, 0x60, 0xf0, 0xf2, 0x39, 0x65, 0x69,
];
const DRAFT_TIER: [u8; 32] = [
    0x9a, 0x32, 0x6e, 0xa5, 0xd8, 0x6b, 0xfe, 0x23, 0x09, 0x4b, 0x63, 0xf9, 0x25, 0x4a, 0x90, 0xbb,
    0xb8, 0xc5, 0x09, 0x77, 0x59, 0xf2, 0x6d, 0x2a, 0xa7, 0x68, 0xfe, 0x55, 0x79, 0x4d, 0xa3, 0x3f,
];

pub(crate) fn supports_packs(target: &[u8; 32], draft: &[u8; 32]) -> bool {
    *target == TARGET_TIER && *draft == DRAFT_TIER
}

pub(crate) fn plan(
    form: SpecTreeForm,
    anchor: i64,
    lat: &LatticeView<'_>,
) -> Result<TreePlan, TreeError> {
    if form != SpecTreeForm::Ancestor9
        || lat.steps != DEPTH
        || lat.k != K
        || !lat.well_formed()
        || lat.cand.iter().any(|id| !(0..248320).contains(id))
    {
        return Err(TreeError::Malformed(
            "tandem-b16-calibrated requires an ancestor9 tree over a 15x16 Tandem lattice".into(),
        ));
    }
    let tree = build(lat, 9)?;
    TreePlan::new(tree, anchor, |n| {
        lat.token(n.depth as usize, n.cand as usize)
    })
}

struct DepthStats {
    p: [[f32; K]; K],
    mean: [f32; K],
    votes: [f32; K],
    peak: [f32; K],
    distance: [f32; K],
}

impl DepthStats {
    fn new(lat: &LatticeView<'_>, d: usize) -> Self {
        let mut out = Self {
            p: [[0.; K]; K],
            mean: [0.; K],
            votes: [0.; K],
            peak: [0.; K],
            distance: [0.; K],
        };
        let parents = if d == 0 { 1 } else { K };
        for prev in 0..parents {
            let row = lat.row(d + 1, prev);
            let mut top = 0;
            for j in 1..K {
                if row[j] > row[top] {
                    top = j;
                }
            }
            let p = &mut out.p[prev];
            for (dest, score) in p.iter_mut().zip(row) {
                *dest = (*score - row[top]).exp();
            }
            let sum = np_sum_f32(p);
            for (mean, val) in out.mean.iter_mut().zip(p.iter_mut()) {
                *val /= sum;
                *mean += *val;
            }
            out.votes[top] += 1.0 / parents as f32;
            out.peak[prev] = p[top];
        }
        for x in &mut out.mean {
            *x /= parents as f32;
        }
        for prev in 0..parents {
            let delta: [f32; K] = std::array::from_fn(|i| (out.p[prev][i] - out.mean[i]).abs());
            out.distance[prev] = 0.5 * np_sum_f32(&delta);
        }
        out
    }
}

struct Frame<'a, 'b> {
    lat: &'a LatticeView<'b>,
    // Only populate a depth the frontier visits and its single look-ahead.
    stats: Vec<DepthStats>,
}

impl Frame<'_, '_> {
    fn offer(
        &mut self,
        heap: &mut Vec<(Frontier, f32)>,
        d: usize,
        prev: usize,
        parent: i64,
        parent_mass: f64,
        parent_lp: f32,
    ) {
        let through = (d + 1).min(DEPTH - 1);
        while self.stats.len() <= through {
            self.stats.push(DepthStats::new(self.lat, self.stats.len()));
        }
        let row = self.lat.row(d + 1, prev);
        let stats = &self.stats[d];
        let p = &stats.p[prev];
        let lp: [f32; K] = std::array::from_fn(|i| p[i].max(1e-12).ln());
        let plp: [f32; K] = std::array::from_fn(|i| p[i] * lp[i]);
        let entropy = -np_sum_f32(&plp) / 16.0_f32.ln();
        let mut order: [usize; K] = std::array::from_fn(|i| i);
        order.sort_unstable_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap().then(i.cmp(&j)));
        let mut rank = [0; K];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        let depth = (((d + 1) as f64).log2() / (DEPTH as f64).log2()) as f32;
        let mut z = [0.; K];
        for i in 0..K {
            let logit = lp[i].clamp(-12., 0.) / 6.;
            let gap = if i == order[0] {
                row[i] - row[order[1]]
            } else {
                row[i] - row[order[0]]
            };
            let token = self.lat.token(d + 1, i);
            let freq = LEXICON
                .binary_search_by_key(&token, |&(id, _)| id)
                .map_or(0., |idx| LEXICON[idx].1);
            let next = self.stats.get(d + 1);
            let x = [
                1.,
                logit,
                gap.clamp(-12., 12.) / 6.,
                1. / (1 + rank[i]) as f32,
                depth,
                if d == 0 { 1. } else { 0. },
                entropy,
                (stats.mean[i].max(1e-12).ln() - lp[i]).clamp(-12., 12.) / 6.,
                stats.votes[i],
                freq,
                (f64::from(parent_lp.max(-12.)) / 6.) as f32,
                depth * logit,
                next.map_or(0., |s| s.peak[i]),
                next.map_or(0., |s| s.distance[i]),
            ];
            z[i] = f64::from(lp[i])
                + x.iter()
                    .zip(WEIGHTS)
                    .map(|(&value, weight)| f64::from(value) * weight)
                    .sum::<f64>();
        }
        let shift = z.iter().copied().fold(0.0_f64, f64::max);
        let e = z.map(|v| (v - shift).exp());
        let denom = e.iter().sum::<f64>() + (-shift).exp(); // outside-candidate class
        order.sort_unstable_by(|&i, &j| z[j].partial_cmp(&z[i]).unwrap().then(i.cmp(&j)));
        for &lane in &order[..3] {
            let edge = Frontier {
                neg_p: -(parent_mass * (e[lane] / denom)),
                depth: d + 1,
                cand: lane,
                parent,
            };
            // Only the ordering key drops 17 mantissa bits. Descendants always
            // multiply the full unrounded probability kept in Frontier.neg_p.
            let pos = heap.partition_point(|(x, _)| {
                let other_mass = (-x.neg_p).to_bits() >> 17;
                let new_mass = (-edge.neg_p).to_bits() >> 17;
                new_mass
                    .cmp(&other_mass)
                    .then(x.depth.cmp(&edge.depth))
                    .then(x.cand.cmp(&edge.cand))
                    .then(x.parent.cmp(&edge.parent))
                    .is_le()
            });
            heap.insert(pos, (edge, lp[lane]));
        }
    }
}

fn build(lat: &LatticeView<'_>, budget: usize) -> Result<SpecTree, TreeError> {
    if lat.scores.iter().any(|s| !s.is_finite()) {
        return Err(TreeError::Malformed(
            "nonfinite B16 selector lattice".into(),
        ));
    }
    let mut frame = Frame {
        lat,
        stats: Vec::with_capacity(DEPTH),
    };
    let mut heap = Vec::with_capacity(2 * budget + 3);
    frame.offer(&mut heap, 0, 0, -1, 1., 0.);
    let mut nodes = Vec::with_capacity(budget);
    while nodes.len() < budget {
        let (e, parent_lp) = heap.remove(0);
        nodes.push(Node {
            depth: e.depth as u32,
            cand: e.cand as u32,
            parent: usize::try_from(e.parent).ok(),
        });
        if nodes.len() < budget && e.depth < DEPTH {
            frame.offer(
                &mut heap,
                e.depth,
                e.cand,
                (nodes.len() - 1) as i64,
                -e.neg_p,
                parent_lp,
            );
        }
    }
    SpecTree::new_limited(nodes, DEPTH as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_tree_requires_the_validated_pair_and_shape() {
        assert!(supports_packs(&TARGET_TIER, &DRAFT_TIER));
        assert!(!supports_packs(&DRAFT_TIER, &TARGET_TIER));
        assert!(!supports_packs(&[0; 32], &DRAFT_TIER));
        assert!(!supports_packs(&TARGET_TIER, &[0; 32]));
        let cand: Vec<_> = (230000..230240).collect();
        let scores = vec![0.; DEPTH * K * K];
        let lat = LatticeView {
            cand: &cand,
            scores: &scores,
            steps: DEPTH,
            k: K,
        };
        assert!(plan(SpecTreeForm::Best16, 1, &lat).is_err());
        let short = LatticeView { steps: 7, ..lat };
        assert!(plan(SpecTreeForm::Ancestor9, 1, &short).is_err());
    }

    #[test]
    fn calibrated_plan_matches_independent_reference_with_tied_and_varied_scores() {
        let cand: Vec<_> = (230000..230240).collect();
        let tied = vec![0.; DEPTH * K * K];
        let varied: Vec<f32> = (0..DEPTH * K * K)
            .map(|i| {
                let (d, p, c) = (i / (K * K), i / K % K, i % K);
                (((d * 11 + p * 7 + c * 13 + c * p * 3) % 41) as i32 - 20) as f32 / 16.
            })
            .collect();
        // Parent row and token ID from the frozen Python model on these synthetic frames.
        for (scores, expected) in [
            (
                &tied,
                [
                    (0, 230015),
                    (0, 230014),
                    (0, 230013),
                    (1, 230031),
                    (1, 230030),
                    (2, 230031),
                    (2, 230030),
                    (1, 230029),
                    (3, 230031),
                ],
            ),
            (
                &varied,
                [
                    (0, 230003),
                    (0, 230006),
                    (0, 230009),
                    (1, 230020),
                    (1, 230031),
                    (1, 230018),
                    (2, 230026),
                    (2, 230022),
                    (2, 230018),
                ],
            ),
        ] {
            let lat = LatticeView {
                cand: &cand,
                scores,
                steps: DEPTH,
                k: K,
            };
            let tree = plan(SpecTreeForm::Ancestor9, 1, &lat).unwrap();
            assert_eq!(tree.rows(), 10);
            for (i, (parent, token)) in expected.iter().enumerate() {
                assert_eq!(
                    (tree.parent_row(i + 1), tree.tokens()[i + 1]),
                    (Some(*parent), *token)
                );
            }
            let stock = TreePlan::from_lattice(SpecTreeForm::Ancestor9, 1, &lat).unwrap();
            assert_ne!(stock, tree);
        }
    }
}
