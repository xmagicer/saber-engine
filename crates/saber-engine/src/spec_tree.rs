//! **P27-10: the tree-speculation law** for the 27B's DFlash2 verify -- the row layout, the
//! state-slot and conv-stride accounting and the acceptance walk -- as host arithmetic, so a tree
//! shape is priced (and refused) before any device code exists for it.
//!
//! # What a tree is here
//!
//! The DFlash2 selector already holds a LATTICE per proposal: `selector_top_k` (16) candidates at
//! each of the block's 7 positions and an edge score from every candidate at position `d - 1` to
//! every candidate at `d` (`sel.cand64`, `sel.scores`; the greedy walk is one path through it). A
//! tree verifies several paths through that lattice at once: a node is `(depth, candidate)` under
//! a parent node (the root = the anchor, the token the last step committed).
//!
//! # The acceptance walk (temperature 0): output-neutral by construction
//!
//! Row `r` of the verify scores the position AFTER node `r`'s token given its ancestors, so node
//! `n` is accepted iff its parent is accepted and its token is the parent row's argmax. The step
//! commits the deepest accepted root-to-node path plus that node's own argmax (the correction, or
//! the bonus at full depth). Every committed token is the target's greedy token given a greedy
//! prefix, so the committed stream IS the SPEC=0 stream whatever the tree -- the same theorem as
//! [`crate::draft`]'s chain law. Two accepted paths of equal depth commit the same tokens; the
//! walk picks the lowest row index, so the pick (and the drafter's restart rows) is deterministic.
//! **Seeded sampling is NOT invariant**: the drawn-row walk (`crate::forward::drawn_rows`) draws
//! per decided position over the step's candidate set, and a tree offers a different set than the
//! chain -- the same class of change as a verify-width change.
//!
//! # The row layouts
//!
//! * [`TreeLayout::AncestorMask`] -- one pseudo-sequence of `1 + nodes` rows; row `r` attends to its
//!   ancestors only. The SABER decode attention holds 64 rows a sequence group = **10 query rows at
//!   GQA 6** ([`SABER_ROWS_PER_SEQ`]), so `1 + nodes <= 10`. NOT runnable on today's kernels: the
//!   attention's mask is causal over the flat row order, and both GDN decode updates and the conv
//!   update carry row `r`'s state from row `r - 1` -- a tree needs a parent index per row in all
//!   three (a kernel round).
//! * [`TreeLayout::SharedPrefixChains`] -- every root-to-leaf path is its own pseudo-sequence,
//!   prefix rows duplicated, so each is a CHAIN the shipped kernels verify as they are; the step is
//!   the `(paths, max_path_rows)` rectangle (2 paths at b1 = the b2s8 graph). Each pseudo-sequence
//!   needs its own GDN snapshot slots, conv window and KV tail page for the rows it writes
//!   ([`chains_state`]).
//!
//! # The accounting
//!
//! GDN: one state slot per candidate ROW per sequence plus the spare (`max_seqs * rows + 1`;
//! 8 x 8 + 1 = 65 slot sets today). Conv: `state_len = conv_width - 1 + q - 1` with `q` the
//! pseudo-sequence's rows ([`conv_state_len`], `docs/EXEC.md` "the conv cache's stride").

use std::fmt;

/// Query rows a sequence holds in the SABER decode attention tile (64 rows / GQA 6, floor).
pub const SABER_ROWS_PER_SEQ: u32 = 10;

/// **E4-3**: the widest tree a sequence verifies -- the 16-row objects (`p27_tree_gdn16`, the
/// 6-warp `p27_tree_attn16`: 96 attention rows = 16 x GQA 6). The u16 ancestor word spans exactly
/// 16 rows (root + 15 nodes), so 16 is also the word's ceiling.
pub const TREE_MAX_ROWS: u32 = 16;

/// The drafter's block positions (block 8 = the anchor + 7).
pub const DRAFT_POSITIONS: u32 = 7;

/// How a tree's nodes become verify rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeLayout {
    /// One pseudo-sequence, a per-row ancestor mask (kernel variants needed).
    AncestorMask,
    /// One pseudo-sequence per root-to-leaf path (shipped kernels).
    SharedPrefixChains,
}

/// One drafted node: its depth (1-based), the selector candidate it takes and its parent node
/// (`None` = the root, i.e. the anchor row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node {
    /// 1..=7.
    pub depth: u32,
    /// The selector candidate index (0..16) at this depth.
    pub cand: u32,
    /// Parent node index, `None` for a depth-1 node.
    pub parent: Option<usize>,
}

/// Why a tree cannot be verified as asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeError {
    /// A parent index that does not precede its child, or a depth that does not follow its parent.
    Malformed(String),
    /// The ancestor-mask layout needs more rows than the attention tile holds.
    TooWide {
        /// Rows asked (root included).
        rows: u32,
    },
    /// A shape string the parser does not know.
    Shape(String),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::Malformed(s) => write!(f, "a malformed tree: {s}"),
            TreeError::TooWide { rows } => write!(
                f,
                "{rows} rows in one sequence: the tree verify holds {TREE_MAX_ROWS} (the 16-row \
                 objects: 96 attention rows / GQA 6, a u16 ancestor word)"
            ),
            TreeError::Shape(s) => write!(f, "unknown tree shape `{s}`"),
        }
    }
}

impl std::error::Error for TreeError {}

/// A speculation tree over one proposal's lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecTree {
    nodes: Vec<Node>,
}

impl SpecTree {
    /// Build from nodes in topological order (a parent before its children).
    ///
    /// # Errors
    /// [`TreeError::Malformed`] when a parent does not precede its child or depths do not chain.
    pub fn new(nodes: Vec<Node>) -> Result<SpecTree, TreeError> {
        Self::new_limited(nodes, DRAFT_POSITIONS)
    }

    fn new_limited(nodes: Vec<Node>, max_depth: u32) -> Result<SpecTree, TreeError> {
        for (i, n) in nodes.iter().enumerate() {
            let want = match n.parent {
                None => 1,
                Some(p) if p < i => nodes[p].depth + 1,
                Some(p) => {
                    return Err(TreeError::Malformed(format!(
                        "node {i}'s parent {p} does not precede it"
                    )))
                }
            };
            if n.depth != want || n.depth > max_depth {
                return Err(TreeError::Malformed(format!(
                    "node {i} at depth {} under a parent that puts it at {want}",
                    n.depth
                )));
            }
        }
        Ok(SpecTree { nodes })
    }

    /// Today's chain: the walk's candidate at each of `k` depths (`cands[d - 1]`).
    ///
    /// # Errors
    /// More candidates than the drafter's positions.
    pub fn chain(cands: &[u32]) -> Result<SpecTree, TreeError> {
        SpecTree::new(
            cands
                .iter()
                .enumerate()
                .map(|(i, c)| Node {
                    depth: i as u32 + 1,
                    cand: *c,
                    parent: i.checked_sub(1),
                })
                .collect(),
        )
    }

    /// The nodes, in row order (row `i + 1` verifies node `i`; row 0 is the anchor).
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Root-to-leaf paths as node-index lists, leaves in node order.
    #[must_use]
    pub fn paths(&self) -> Vec<Vec<usize>> {
        let mut has_child = vec![false; self.nodes.len()];
        for n in &self.nodes {
            if let Some(p) = n.parent {
                has_child[p] = true;
            }
        }
        (0..self.nodes.len())
            .filter(|i| !has_child[*i])
            .map(|leaf| {
                let mut path = vec![leaf];
                let mut at = leaf;
                while let Some(p) = self.nodes[at].parent {
                    path.push(p);
                    at = p;
                }
                path.reverse();
                path
            })
            .collect()
    }

    /// Verify rows a SEQUENCE costs under `layout` (the root row included).
    #[must_use]
    pub fn rows(&self, layout: TreeLayout) -> u32 {
        match layout {
            TreeLayout::AncestorMask => 1 + self.nodes.len() as u32,
            TreeLayout::SharedPrefixChains => {
                let paths = self.paths();
                let longest = paths.iter().map(Vec::len).max().unwrap_or(0) as u32;
                paths.len().max(1) as u32 * (1 + longest)
            }
        }
    }

    /// The ancestor mask, one word per row (bit `j` = row `r` may attend to row `j`; row 0 is the
    /// anchor). Every row sees the root and itself.
    ///
    /// # Errors
    /// [`TreeError::TooWide`] past the attention tile.
    pub fn ancestor_mask(&self) -> Result<Vec<u16>, TreeError> {
        let rows = self.rows(TreeLayout::AncestorMask);
        if rows > TREE_MAX_ROWS {
            return Err(TreeError::TooWide { rows });
        }
        let mut mask = vec![1u16];
        for (i, n) in self.nodes.iter().enumerate() {
            let parent_row = n.parent.map_or(0, |p| p + 1);
            mask.push(mask[parent_row] | (1 << (i + 1)));
        }
        Ok(mask)
    }

    /// **THE ACCEPTANCE WALK** at temperature 0: `token(n)` is node `n`'s token, `argmax(r)` row
    /// `r`'s argmax (row 0 = the anchor's). Returns the accepted node path (root excluded) and the
    /// committed tokens (path tokens + the deepest accepted row's argmax). Equal depths go to the
    /// lowest node index.
    pub fn accept(
        &self,
        token: impl Fn(usize) -> i32,
        argmax: impl Fn(usize) -> i32,
    ) -> (Vec<usize>, Vec<i32>) {
        let mut ok = vec![false; self.nodes.len()];
        let mut best: Option<usize> = None;
        for (i, n) in self.nodes.iter().enumerate() {
            let parent_row = match n.parent {
                None => 0,
                Some(p) if ok[p] => p + 1,
                Some(_) => continue,
            };
            if token(i) == argmax(parent_row) {
                ok[i] = true;
                if best.is_none_or(|b| self.nodes[b].depth < n.depth) {
                    best = Some(i);
                }
            }
        }
        let mut path = Vec::new();
        let mut at = best;
        while let Some(i) = at {
            path.push(i);
            at = self.nodes[i].parent;
        }
        path.reverse();
        let mut committed: Vec<i32> = path.iter().map(|i| token(*i)).collect();
        committed.push(argmax(best.map_or(0, |b| b + 1)));
        (path, committed)
    }
}

/// GDN snapshot slot SETS an arena holds for `max_seqs` sequences of `rows_per_seq` candidate rows:
/// one a row, plus the spare (the pad law's).
#[must_use]
pub fn gdn_slot_sets(max_seqs: u32, rows_per_seq: u32) -> u32 {
    max_seqs * rows_per_seq + 1
}

/// The conv cache's per-sequence state length at verify rows `q` (`docs/EXEC.md`).
#[must_use]
pub fn conv_state_len(conv_width: u32, q: u32) -> u32 {
    conv_width - 1 + q.max(1) - 1
}

/// What one sequence's shared-prefix-chains verify needs beyond a chain's, per extra path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainsState {
    /// Pseudo-sequences the step carries for this sequence.
    pub pseudo_seqs: u32,
    /// Extra GDN snapshot slot sets (one per row of each extra path).
    pub extra_slot_sets: u32,
    /// Extra conv windows (one per extra path; copied from the sequence's before the step).
    pub extra_conv: u32,
    /// Extra KV tail pages (one per extra path per attention layer group: the block holding the
    /// anchor is copied, the path writes its own rows past it).
    pub extra_kv_tail_pages: u32,
}

/// The shared-prefix-chains state bill for one sequence verifying `tree`.
#[must_use]
pub fn chains_state(tree: &SpecTree) -> ChainsState {
    let paths = tree.paths();
    let extra = paths.len().saturating_sub(1) as u32;
    let longest = paths.iter().map(Vec::len).max().unwrap_or(0) as u32;
    ChainsState {
        pseudo_seqs: paths.len().max(1) as u32,
        extra_slot_sets: extra * (1 + longest),
        extra_conv: extra,
        extra_kv_tail_pages: extra,
    }
}

// ---------------------------------------------------------------------------------------------
// P27-15b: THE ARM'S HOST PLAN (`--engine-spec-tree ancestor9`). Item (d) of P27-15's sizing --
// the lattice -> the 10-row verify plan (ancestor words, row -> node map, rope by depth), the
// acceptance verdict and what it moves (the accepted path's KV rows, the drafter's aux rows, the
// conv tail), the GDN tree record pair that replaces P27-10's per-row snapshot slots, and the
// advance law (the accepted row's state is applied by the NEXT verify's prologue, or flushed by
// its own launch before anything else touches the slot). Host arithmetic only: the device half
// (items a, b, c, e) consumes these tables.

/// `--engine-spec-tree`: the verify's row form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpecTreeForm {
    /// Today's chain of 7 (THE DEFAULT and the control).
    #[default]
    Off,
    /// P27-10's best-first 9-node ancestor-mask tree in the 10-row tile (an ARM; numerics: the
    /// WY-form GDN tree is not the chain's summation).
    Ancestor9,
    /// P27-15f: `ancestor9` with the TWO-RECORD control (engine / worker only): the GDN tree
    /// kernel reads record A and writes record B (`ts_in != ts_out`, the oracle-gated form), B
    /// copied to A before each step -- the control for the one-record in-place design.
    Ancestor9Pair,
    /// E4-3: the best-first tree at 15 nodes in the 16-row objects (`p27_tree_gdn16`,
    /// `p27_tree_attn16`) -- over DFlash2's lattice, a 15-node best-first tree (an ARM, default
    /// off).
    Best16,
}

impl SpecTreeForm {
    /// True for either tree form.
    #[must_use]
    pub const fn is_tree(self) -> bool {
        matches!(self, Self::Ancestor9 | Self::Ancestor9Pair | Self::Best16)
    }

    /// Rows a sequence verifies under the form (root + nodes; 0 = the chain).
    #[must_use]
    pub const fn rows(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Ancestor9 | Self::Ancestor9Pair => SABER_ROWS_PER_SEQ as usize,
            Self::Best16 => TREE_MAX_ROWS as usize,
        }
    }

    /// The flag's spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ancestor9 => "ancestor9",
            Self::Ancestor9Pair => "ancestor9-pair",
            Self::Best16 => "best16",
        }
    }

    /// Parse the flag's value; refused by name otherwise.
    ///
    /// # Errors
    /// A value that is neither form.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "ancestor9" => Ok(Self::Ancestor9),
            "ancestor9-pair" => Ok(Self::Ancestor9Pair),
            "best16" => Ok(Self::Best16),
            o => Err(format!(
                "--engine-spec-tree `{o}`: the forms are `off` (today's chain of 7, the default) \
                 , `ancestor9` (the best-first 9-node ancestor-mask tree, P27-15b) and \
                 `best16` (the 16-row tree verify, E4-3)"
            )),
        }
    }

    /// Drafted nodes a sequence verifies (0 = the chain).
    #[must_use]
    pub const fn nodes(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Ancestor9 | Self::Ancestor9Pair => BEST_FIRST_NODES,
            Self::Best16 => TREE_MAX_ROWS as usize - 1,
        }
    }
}

/// The tree's node budget (rows = 1 + nodes = the SABER tile's 10).
pub const BEST_FIRST_NODES: usize = 9;
/// Children a node may expand into (the selector's ranked candidates under it).
pub const BEST_FIRST_CHILDREN: usize = 3;
/// The temperature each selector row is read at (P27-10's priced shape).
pub const BEST_FIRST_TEMPERATURE: f32 = 1.0;

/// One request's selector lattice: `cand` = `sel.cand64` `[steps][k]`, `scores` = `sel.scores`
/// `[steps][k pred][k cand]` (depth 1's rows all read the anchor: row `[0][0]`).
#[derive(Clone, Copy, Debug)]
pub struct LatticeView<'a> {
    /// Candidate ids.
    pub cand: &'a [i64],
    /// Edge scores.
    pub scores: &'a [f32],
    /// Drafted positions (7).
    pub steps: usize,
    /// Candidates a position (16).
    pub k: usize,
}

impl LatticeView<'_> {
    fn well_formed(&self) -> bool {
        self.steps > 0
            && self.k > 0
            && self.cand.len() >= self.steps * self.k
            && self.scores.len() >= self.steps * self.k * self.k
    }

    /// The selector row that scores depth `d` (1-based) after candidate `prev` at `d - 1`.
    fn row(&self, d: usize, prev: usize) -> &[f32] {
        let (dd, p) = if d == 1 { (0, 0) } else { (d - 1, prev) };
        let at = (dd * self.k + p) * self.k;
        &self.scores[at..at + self.k]
    }

    /// Candidate `c`'s token at depth `d` (1-based).
    #[must_use]
    pub fn token(&self, d: usize, c: usize) -> i64 {
        self.cand[(d - 1) * self.k + c]
    }
}

/// numpy's float32 `sum` of a short contiguous row (its pairwise form below 128 elements: eight
/// running sums, combined as a balanced pair tree, then the remainder in order).
fn np_sum_f32(v: &[f32]) -> f32 {
    if v.len() < 8 {
        return v.iter().fold(0.0_f32, |a, &b| a + b);
    }
    let mut r = [0.0_f32; 8];
    r.copy_from_slice(&v[..8]);
    let whole = v.len() - v.len() % 8;
    for i in (8..whole).step_by(8) {
        for j in 0..8 {
            r[j] += v[i + j];
        }
    }
    let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for x in &v[whole..] {
        res += x;
    }
    res
}

/// `tree_sim.softmax` in float32 (the reference's dtype): `e = exp((x - max) / t); e / sum(e)`.
fn softmax_f32(row: &[f32], t: f32, out: &mut Vec<f32>) {
    let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    out.clear();
    out.extend(row.iter().map(|&x| ((x - mx) / t).exp()));
    let s = np_sum_f32(out);
    for o in out.iter_mut() {
        *o /= s;
    }
}

/// Candidate indices by descending score, ties to the lowest index (`argsort(-row, stable)`).
fn ranked(row: &[f32]) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..row.len()).collect();
    ix.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ix
}

/// A frontier entry, ordered as Python's heap orders `(-p, depth, cand, parent)`.
#[derive(Clone, Copy, Debug)]
struct Frontier {
    neg_p: f64,
    depth: usize,
    cand: usize,
    parent: i64,
}

impl Frontier {
    fn key_cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.neg_p
            .partial_cmp(&o.neg_p)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.depth.cmp(&o.depth))
            .then(self.cand.cmp(&o.cand))
            .then(self.parent.cmp(&o.parent))
    }
}

/// **The best-first tree** (the dynamic form, P27-10's +7.8 % shape): expand the frontier
/// node of the highest path-probability proxy (the product of the selector rows' softmaxes along
/// its path) until `budget` nodes, each node offering its `children` best-ranked candidates.
/// Byte-for-byte `tools/capture/p27_10/tree_sim.py`'s `tree_best_first` (float32 rows, float64
/// products, the heap's tuple order); nodes come out in pop order (a parent before its children).
///
/// # Errors
/// A malformed lattice, or a budget the tile does not hold.
pub fn best_first(
    lat: &LatticeView<'_>,
    budget: usize,
    children: usize,
    t: f32,
) -> Result<SpecTree, TreeError> {
    if !lat.well_formed() {
        return Err(TreeError::Malformed(format!(
            "a lattice of {} candidates / {} scores for {} x {}",
            lat.cand.len(),
            lat.scores.len(),
            lat.steps,
            lat.k
        )));
    }
    if 1 + budget as u32 > TREE_MAX_ROWS {
        return Err(TreeError::TooWide {
            rows: 1 + budget as u32,
        });
    }
    let mut heap: Vec<Frontier> = Vec::new();
    let mut sm = Vec::with_capacity(lat.k);
    let push = |heap: &mut Vec<Frontier>, f: Frontier| {
        // a min-heap by the key; the budget is tiny, a sorted insert keeps the pop order exact
        let at = heap.partition_point(|x| x.key_cmp(&f) != std::cmp::Ordering::Greater);
        heap.insert(at, f);
    };
    let first = lat.row(1, 0);
    softmax_f32(first, t, &mut sm);
    for c in ranked(first).into_iter().take(children) {
        push(
            &mut heap,
            Frontier {
                neg_p: -f64::from(sm[c]),
                depth: 1,
                cand: c,
                parent: -1,
            },
        );
    }
    let mut nodes: Vec<Node> = Vec::with_capacity(budget);
    while !heap.is_empty() && nodes.len() < budget {
        let f = heap.remove(0);
        nodes.push(Node {
            depth: f.depth as u32,
            cand: f.cand as u32,
            parent: usize::try_from(f.parent).ok(),
        });
        let me = (nodes.len() - 1) as i64;
        let p = -f.neg_p;
        if f.depth < lat.steps {
            let row = lat.row(f.depth + 1, f.cand);
            softmax_f32(row, t, &mut sm);
            for cc in ranked(row).into_iter().take(children) {
                push(
                    &mut heap,
                    Frontier {
                        neg_p: -(p * f64::from(sm[cc])),
                        depth: f.depth + 1,
                        cand: cc,
                        parent: me,
                    },
                );
            }
        }
    }
    SpecTree::new(nodes)
}

/// Where one conv window element comes from: a verify row's input, or the committed tail's
/// element `j` (0 = the oldest of the `width - 1` the cache holds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvSrc {
    /// The committed conv tail.
    Tail(u32),
    /// A verify row of this step.
    Row(u32),
}

/// **The verify plan of one sequence's tree**: row 0 = the anchor (the token the last step
/// committed), row `i + 1` = node `i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreePlan {
    tree: SpecTree,
    words: Vec<u16>,
    parent_row: Vec<Option<usize>>,
    depth: Vec<u32>,
    tokens: Vec<i64>,
}

impl TreePlan {
    /// The plan of `tree` under `anchor`, node tokens by `token(node)`.
    ///
    /// # Errors
    /// [`TreeError::TooWide`] past the tile.
    pub fn new(
        tree: SpecTree,
        anchor: i64,
        token: impl Fn(&Node) -> i64,
    ) -> Result<TreePlan, TreeError> {
        let words = tree.ancestor_mask()?;
        let mut parent_row = vec![None];
        let mut depth = vec![0];
        let mut tokens = vec![anchor];
        for n in tree.nodes() {
            parent_row.push(Some(n.parent.map_or(0, |p| p + 1)));
            depth.push(n.depth);
            tokens.push(token(n));
        }
        Ok(TreePlan {
            tree,
            words,
            parent_row,
            depth,
            tokens,
        })
    }

    /// Build the external 15-node verifier layout from token IDs and zero-based parent indices.
    /// Each sibling token must be unique so both TP ranks derive the same accepted path.
    pub(crate) fn from_external(
        anchor: u32,
        entries: &[(Option<usize>, u32)],
        vocab: u32,
    ) -> Result<Self, TreeError> {
        if entries.len() != TREE_MAX_ROWS as usize - 1 || anchor >= vocab {
            return Err(TreeError::Malformed(
                "invalid external tree size or anchor".into(),
            ));
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(entries.len());
        let mut siblings = std::collections::BTreeSet::new();
        for (parent, token) in entries.iter().copied() {
            if token >= vocab || !siblings.insert((parent, token)) {
                return Err(TreeError::Malformed(
                    "invalid external token or duplicate sibling".into(),
                ));
            }
            let depth = match parent {
                None => 1,
                Some(parent) => {
                    nodes
                        .get(parent)
                        .ok_or_else(|| {
                            TreeError::Malformed("external parent must precede its child".into())
                        })?
                        .depth
                        + 1
                }
            };
            nodes.push(Node {
                parent,
                depth,
                cand: nodes.len() as u32,
            });
        }
        let tree = SpecTree::new_limited(nodes, TREE_MAX_ROWS - 1)?;
        Self::new(tree, i64::from(anchor), |node| {
            i64::from(entries[node.cand as usize].1)
        })
    }

    /// The best-first plan of `form` over one request's lattice.
    ///
    /// # Errors
    /// `Off` (a chain has no tree plan), a malformed lattice.
    pub fn from_lattice(
        form: SpecTreeForm,
        anchor: i64,
        lat: &LatticeView<'_>,
    ) -> Result<TreePlan, TreeError> {
        if form.nodes() == 0 {
            return Err(TreeError::Shape("off".to_owned()));
        }
        let tree = best_first(
            lat,
            form.nodes(),
            BEST_FIRST_CHILDREN,
            BEST_FIRST_TEMPERATURE,
        )?;
        TreePlan::new(tree, anchor, |n| {
            lat.token(n.depth as usize, n.cand as usize)
        })
    }

    /// The same plan under another anchor token (a lattice plan is built before the anchor is
    /// committed).
    #[must_use]
    pub fn with_anchor(&self, anchor: i64) -> TreePlan {
        let mut p = self.clone();
        p.tokens[0] = anchor;
        p
    }

    /// The tree.
    #[must_use]
    pub fn tree(&self) -> &SpecTree {
        &self.tree
    }

    /// Verify rows (1 + nodes).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.words.len()
    }

    /// One u16 ancestor word a row (the kernels' `anc[row]`).
    #[must_use]
    pub fn words(&self) -> &[u16] {
        &self.words
    }

    /// Row -> parent row (`None` for the anchor).
    #[must_use]
    pub fn parent_row(&self, row: usize) -> Option<usize> {
        self.parent_row[row]
    }

    /// Row -> depth (0 = the anchor).
    #[must_use]
    pub fn depth(&self, row: usize) -> u32 {
        self.depth[row]
    }

    /// The token each row embeds (row 0 = the anchor).
    #[must_use]
    pub fn tokens(&self) -> &[i64] {
        &self.tokens
    }

    /// **Rope by depth**: row `r`'s position is the anchor's + its depth (siblings share one).
    #[must_use]
    pub fn positions(&self, anchor_pos: u32) -> Vec<u32> {
        self.depth.iter().map(|d| anchor_pos + d).collect()
    }

    /// Row `r`'s root path as rows, the anchor first, `r` last.
    #[must_use]
    pub fn path_rows(&self, row: usize) -> Vec<usize> {
        let mut v = vec![row];
        let mut at = row;
        while let Some(p) = self.parent_row[at] {
            v.push(p);
            at = p;
        }
        v.reverse();
        v
    }

    /// **The tree conv window** (item a): the `taps` inputs row `r`'s causal conv reads, oldest
    /// first, ending at `r` itself -- its ancestors' rows, then the committed tail
    /// (`taps - 1` elements) where the path is shorter than the window.
    #[must_use]
    pub fn conv_sources(&self, row: usize, taps: u32) -> Vec<ConvSrc> {
        let tail = taps.saturating_sub(1);
        let mut seq: Vec<ConvSrc> = (0..tail).map(ConvSrc::Tail).collect();
        seq.extend(
            self.path_rows(row)
                .into_iter()
                .map(|r| ConvSrc::Row(r as u32)),
        );
        seq.split_off(seq.len() - taps as usize)
    }

    /// **The verdict** at temperature 0: `argmax(row)` is the target's argmax on verify row `row`.
    #[must_use]
    pub fn verdict(&self, argmax: impl Fn(usize) -> i64) -> TreeVerdict {
        let (path, committed) = self
            .tree
            .accept(|i| self.tokens[i + 1] as i32, |r| argmax(r) as i32);
        let mut path_rows = vec![0];
        path_rows.extend(path.iter().map(|i| i + 1));
        TreeVerdict {
            accepted_row: *path_rows.last().unwrap_or(&0),
            path_rows,
            committed: committed.into_iter().map(i64::from).collect(),
        }
    }

    /// The rows whose parent is `row`, in row order.
    #[must_use]
    pub fn children(&self, row: usize) -> Vec<usize> {
        (row + 1..self.rows())
            .filter(|c| self.parent_row[*c] == Some(row))
            .collect()
    }

    /// **P27-W2-3: THE DRAWN TREE WALK** (the sampled tree acceptance law, `--spec-tree-sampled
    /// on`). `decide(row, depth, path)` is the request's draw for the position `depth` past the
    /// anchor over verify row `row` -- ws-f40's keyed draw (`(seed, position, choice)`, an
    /// exponential race over the row's own distribution), `path` the node tokens accepted so far
    /// (the penalties' history) -- returning the token and whether the request ends on it, or
    /// `None` when the row cannot be decided from its seam. `argmax(row)` is the placeholder a
    /// refused row commits (the executor re-draws it densely under the same key, on the row this
    /// walk names: [`DrawnWalk::undecided_row`]).
    ///
    /// The walk starts at the anchor row; at each row it DRAWS, and descends into the child whose
    /// token IS the draw (the lowest such row), else commits the draw and stops. A leaf's draw is
    /// the bonus. So every committed token is the keyed draw of its own position over the row
    /// that holds exactly its committed prefix -- the width-1 stream's draw for draw (the
    /// temperature-0 theorem with the argmax replaced by the draw): the committed distribution IS
    /// the target's, the seed fixes it, and the tree changes only how many positions one step
    /// decides. Its acceptance at a node is `sum over children p(child)`, recursive rejection
    /// sampling's rate for deterministic distinct children (`docs/EXEC.md`, the sampled tree law).
    pub fn drawn_verdict(
        &self,
        decide: &mut DrawDecider<'_>,
        argmax: &dyn Fn(usize) -> i64,
    ) -> DrawnWalk {
        let mut path_rows = vec![0usize];
        let mut committed: Vec<i64> = Vec::new();
        let mut undecided = false;
        loop {
            let row = *path_rows.last().unwrap_or(&0);
            let depth = path_rows.len() - 1;
            let path: Vec<i64> = path_rows[1..].iter().map(|r| self.tokens[*r]).collect();
            match decide(row, depth, &path) {
                None => {
                    committed.push(argmax(row));
                    undecided = true;
                    break;
                }
                Some((t, ends)) => {
                    committed.push(t);
                    if ends {
                        break;
                    }
                    match self
                        .children(row)
                        .into_iter()
                        .find(|c| self.tokens[*c] == t)
                    {
                        Some(c) => path_rows.push(c),
                        None => break,
                    }
                }
            }
        }
        let decided = committed.len() - usize::from(undecided);
        DrawnWalk {
            verdict: TreeVerdict {
                accepted_row: *path_rows.last().unwrap_or(&0),
                path_rows,
                committed,
            },
            decided,
            undecided,
        }
    }
}

/// The drawn walk's decider: `(row, depth, accepted path) -> (token, request ends on it)` or
/// `None` for a row the seam refuses.
pub type DrawDecider<'a> = dyn FnMut(usize, usize, &[i64]) -> Option<(i64, bool)> + 'a;

/// What [`TreePlan::drawn_verdict`] walked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawnWalk {
    /// The verdict: the accepted path's rows, the committed tokens (the draws, in position
    /// order), the deepest accepted row.
    pub verdict: TreeVerdict,
    /// Rows the decider decided (= committed tokens unless the last row was refused).
    pub decided: usize,
    /// The last committed token is the refused row's ARGMAX placeholder.
    pub undecided: bool,
}

impl DrawnWalk {
    /// The verify row a refused last token must be drawn from densely (the accepted row).
    #[must_use]
    pub fn undecided_row(&self) -> Option<usize> {
        self.undecided.then_some(self.verdict.accepted_row)
    }
}

/// What a verify step's acceptance moves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeVerdict {
    /// The accepted path's verify rows, the anchor (0) first; row `path_rows[j]` holds depth `j`.
    pub path_rows: Vec<usize>,
    /// The committed tokens: the accepted nodes' tokens + the deepest accepted row's argmax.
    pub committed: Vec<i64>,
    /// The deepest accepted row (0 = nothing accepted): the GDN record's advance row.
    pub accepted_row: usize,
}

impl TreeVerdict {
    /// Accepted depth (drafted tokens accepted).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.path_rows.len() - 1
    }

    /// **The KV compaction** (item b): `(from_row, to_row)` for every accepted row whose K/V the
    /// verify wrote at a slot other than its depth's; applied in order (each destination is below
    /// its source and above every earlier destination, so no copy reads a slot already written).
    #[must_use]
    pub fn kv_moves(&self) -> Vec<(usize, usize)> {
        self.path_rows
            .iter()
            .enumerate()
            .filter(|(d, r)| *d != **r)
            .map(|(d, r)| (*r, d))
            .collect()
    }

    /// **The drafter's aux gather** (item c): the verify rows whose `aux.cat` rows the drafter's
    /// context store takes, in position order (a chain takes rows `0..=depth`).
    #[must_use]
    pub fn aux_rows(&self) -> &[usize] {
        &self.path_rows
    }

    /// The committed conv tail after the verdict (`taps - 1` elements, oldest first): the last
    /// inputs along the accepted path, reaching into the old tail when the path is short.
    #[must_use]
    pub fn conv_commit(&self, taps: u32) -> Vec<ConvSrc> {
        let tail = taps.saturating_sub(1);
        let mut seq: Vec<ConvSrc> = (0..tail).map(ConvSrc::Tail).collect();
        seq.extend(self.path_rows.iter().map(|r| ConvSrc::Row(*r as u32)));
        seq.split_off(seq.len() - tail as usize)
    }
}

/// Bytes of one (sequence, v head) GDN tree record: U [10][128] + k^ [10][128] + P [16] + words
/// [16], f32 (`p27_tree_gdn.hip`'s `TG_REC_FLOATS`).
pub const TREE_RECORD_BYTES: u64 = (2 * 10 * 128 + 16 + 16) * 4;

/// E4-3: one record's bytes at `rows` tree rows (10 -> [`TREE_RECORD_BYTES`]; 16 -> 16,512 B,
/// `p27_tree_gdn16.hip`'s `TG_REC_FLOATS` at `TG_TM` 16).
#[must_use]
pub const fn tree_record_bytes(rows: usize) -> u64 {
    (2 * rows as u64 * 128 + 16 + 16) * 4
}

/// **The record pair** a sequence holds a rank: two parities (the verify reads the pending
/// advance from one and writes its own record to the other) x GDN layers x the rank's v heads.
/// It replaces P27-10's `+2` per-row snapshot slot sets a sequence (`gdn_slot_sets(s, 10)` vs
/// `(s, 8)`).
#[must_use]
pub fn tree_record_pair_bytes(gdn_layers: u32, v_heads_per_rank: u32) -> u64 {
    2 * u64::from(gdn_layers) * u64::from(v_heads_per_rank) * TREE_RECORD_BYTES
}

/// A sequence slot's pending GDN advance: the record parity that holds it and the accepted row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingAdvance {
    /// The record (0 / 1) the verify that produced it wrote.
    pub parity: u8,
    /// The accepted row (`TreeVerdict::accepted_row`).
    pub row: u8,
}

/// One verify's record bindings for a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyRecords {
    /// The prologue's advance (`acc[n]` = its row, `ts_in` = its parity), `None` = `acc[n] = -1`.
    pub prologue: Option<PendingAdvance>,
    /// The parity this verify writes (`ts_out`).
    pub out_parity: u8,
}

/// **The advance law** (item d): the committed GDN state of a tree-verified slot is stale until
/// its pending advance runs -- as the next verify's prologue, or as its own launch (a FLUSH)
/// before any other reader or writer of the slot (a prefill chunk, a prefix-cache snapshot, the
/// drafter seeding from the state, a release that hands the slot on is NOT one: the state dies).
#[derive(Clone, Debug, Default)]
pub struct AdvanceLaw {
    pending: Vec<Option<PendingAdvance>>,
    next_parity: Vec<u8>,
    /// Prologue advances issued.
    pub prologues: u64,
    /// Flush launches issued.
    pub flushes: u64,
}

impl AdvanceLaw {
    /// A law over `slots` sequence slots, nothing pending.
    #[must_use]
    pub fn new(slots: usize) -> AdvanceLaw {
        AdvanceLaw {
            pending: vec![None; slots],
            next_parity: vec![0; slots],
            ..AdvanceLaw::default()
        }
    }

    /// The pending advance of `slot`, if any.
    #[must_use]
    pub fn pending(&self, slot: usize) -> Option<PendingAdvance> {
        self.pending[slot]
    }

    /// A tree verify of `slot` is planned: it consumes the pending advance as its prologue and
    /// writes the other parity.
    pub fn verify(&mut self, slot: usize) -> VerifyRecords {
        let prologue = self.pending[slot].take();
        if prologue.is_some() {
            self.prologues += 1;
        }
        let out_parity = prologue.map_or(self.next_parity[slot], |p| p.parity ^ 1);
        self.next_parity[slot] = out_parity ^ 1;
        VerifyRecords {
            prologue,
            out_parity,
        }
    }

    /// The verdict of the verify that wrote `rec`: its accepted row becomes pending.
    pub fn settle(&mut self, slot: usize, rec: VerifyRecords, accepted_row: usize) {
        self.pending[slot] = Some(PendingAdvance {
            parity: rec.out_parity,
            row: accepted_row as u8,
        });
    }

    /// Anything but a tree verify is about to touch `slot`'s state: the advance to flush first.
    pub fn flush(&mut self, slot: usize) -> Option<PendingAdvance> {
        let p = self.pending[slot].take();
        if p.is_some() {
            self.flushes += 1;
        }
        p
    }

    /// The slot is released (its state dies with the sequence): nothing to flush.
    pub fn release(&mut self, slot: usize) {
        self.pending[slot] = None;
        self.next_parity[slot] = 0;
    }
}

/// The arm's counters (`spectree` line): tree steps, nodes verified, accepted depth, steps whose
/// accepted path needed a KV compaction, prologue advances, flush launches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecTreeCounts {
    /// Sequence-steps verified as a tree.
    pub steps: u64,
    /// Nodes verified (excluding the anchor).
    pub nodes: u64,
    /// Accepted drafted tokens.
    pub accepted: u64,
    /// Sequence-steps with at least one KV move.
    pub compactions: u64,
    /// KV rows moved.
    pub moved_rows: u64,
    /// Advances applied as a verify prologue.
    pub prologues: u64,
    /// Advances flushed by their own launch.
    pub flushes: u64,
}

impl SpecTreeCounts {
    /// Count one sequence-step's plan + verdict.
    pub fn record(&mut self, plan: &TreePlan, v: &TreeVerdict) {
        self.steps += 1;
        self.nodes += (plan.rows() - 1) as u64;
        self.accepted += v.depth() as u64;
        let moves = v.kv_moves().len() as u64;
        self.moved_rows += moves;
        self.compactions += u64::from(moves > 0);
    }

    /// The end-of-run line.
    #[must_use]
    pub fn line(&self) -> String {
        let per = |x: u64| {
            if self.steps == 0 {
                0.0
            } else {
                x as f64 / self.steps as f64
            }
        };
        format!(
            "spectree steps {} nodes/step {:.2} accepted/step {:.3} tokens/step {:.3} \
             compactions {} moved rows {} prologues {} flushes {}",
            self.steps,
            per(self.nodes),
            per(self.accepted),
            1.0 + per(self.accepted),
            self.compactions,
            self.moved_rows,
            self.prologues,
            self.flushes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(depth: u32, cand: u32, parent: Option<usize>) -> Node {
        Node {
            depth,
            cand,
            parent,
        }
    }

    /// The chain of 7 + two sibling leaves at depths 1 and 2 = 10 rows: the widest ancestor tree
    /// the tile holds.
    fn chain7_leaves12() -> SpecTree {
        let mut v: Vec<Node> = (0..7)
            .map(|i| n(i as u32 + 1, 0, (i as usize).checked_sub(1)))
            .collect();
        v.push(n(1, 1, None));
        v.push(n(2, 1, Some(0)));
        SpecTree::new(v).expect("well formed")
    }

    #[test]
    fn a_chain_is_the_chain_law_and_commits_the_prefix_plus_one() {
        let t = SpecTree::chain(&[0; 7]).expect("chain");
        assert_eq!(t.rows(TreeLayout::AncestorMask), 8);
        assert_eq!(t.rows(TreeLayout::SharedPrefixChains), 8);
        // tokens 10, 11, .. ; the target agrees through depth 3 then says 99.
        let tok = |i: usize| 10 + i as i32;
        let am = |r: usize| if r < 3 { 10 + r as i32 } else { 99 };
        let (path, committed) = t.accept(tok, am);
        assert_eq!(path, vec![0, 1, 2]);
        assert_eq!(committed, vec![10, 11, 12, 99]);
        // the shared law agrees (row 0 = anchor 7)
        let draft = [7, 10, 11, 12, 13, 14, 15, 16];
        let rows: Vec<i32> = (0..8).map(am).collect();
        let law = saber_exec::forward::accept_prefix(&draft, &rows).expect("law");
        assert_eq!(&law.committed[1..], &committed[..]);
    }

    #[test]
    fn a_sibling_leaf_rescues_a_depth_one_miss() {
        let t = chain7_leaves12();
        assert_eq!(t.rows(TreeLayout::AncestorMask), 10);
        let tok = |i: usize| match i {
            7 => 50, // the depth-1 sibling
            8 => 51, // the depth-2 sibling under the chain's depth-1 node
            _ => 10 + i as i32,
        };
        // the target's depth-1 token is the sibling's; its row then says 77
        let am = |r: usize| match r {
            0 => 50,
            8 => 77,
            _ => -1,
        };
        let (path, committed) = t.accept(tok, am);
        assert_eq!(path, vec![7]);
        assert_eq!(committed, vec![50, 77]);
        // the chain alone commits one token here
        let c = SpecTree::chain(&[0; 7]).expect("chain");
        assert_eq!(c.accept(tok, am).1, vec![50]);
    }

    #[test]
    fn the_committed_stream_is_the_greedy_stream_whatever_the_tree() {
        // A greedy stream g; any tree's committed tokens are a prefix of g (the theorem).
        let g = [5, 6, 7, 8, 9, 10, 11, 12];
        let t = chain7_leaves12();
        for miss in 0..8 {
            // tokens: chain node i = g[i] up to the miss, then wrong; siblings = g where they sit.
            let tok = |i: usize| match i {
                7 => g[0],
                8 => g[1],
                i if i < miss => g[i],
                _ => -5,
            };
            // row r (node r-1 at depth r) predicts g[depth]; any row on a wrong path is wrong too,
            // but the walk never reads one.
            let depth_of = |r: usize| {
                if r == 0 {
                    0
                } else {
                    t.nodes()[r - 1].depth as usize
                }
            };
            let am = |r: usize| g[depth_of(r)];
            let (_, committed) = t.accept(tok, am);
            assert_eq!(&committed[..], &g[..committed.len()], "miss at {miss}");
        }
    }

    #[test]
    fn the_ancestor_mask_and_its_refusal() {
        let t = chain7_leaves12();
        let m = t.ancestor_mask().expect("10 rows fit");
        assert_eq!(m.len(), 10);
        assert_eq!(m[0], 0b1);
        assert_eq!(m[3], 0b1111); // chain node at depth 3: root + rows 1..3
        assert_eq!(m[8], 0b1_0000_0001); // the depth-1 sibling: root + itself
        assert_eq!(m[9], 0b10_0000_0011); // the depth-2 sibling under row 1
        let wide = SpecTree::new(
            (0..7)
                .map(|i| n(i as u32 + 1, 0, (i as usize).checked_sub(1)))
                .chain([n(1, 1, None), n(1, 2, None), n(1, 3, None)])
                .collect(),
        )
        .expect("well formed");
        // E4-3: 11 rows fit the 16-row objects now; 17 do not
        assert_eq!(wide.ancestor_mask().map(|m| m.len()), Ok(11));
        let wider = SpecTree::new(
            (0..7)
                .map(|i| n(i as u32 + 1, 0, (i as usize).checked_sub(1)))
                .chain((1..10).map(|c| n(1, c, None)))
                .collect(),
        )
        .expect("well formed");
        assert_eq!(wider.ancestor_mask(), Err(TreeError::TooWide { rows: 17 }));
    }

    #[test]
    fn shared_prefix_chains_cost_the_rectangle_and_their_state() {
        // chain 7 + a depth-3 deviation walked to depth 7 = two paths of 7 drafts
        let mut v: Vec<Node> = (0..7)
            .map(|i| n(i as u32 + 1, 0, (i as usize).checked_sub(1)))
            .collect();
        v.push(n(3, 1, Some(1)));
        for d in 4..=7 {
            v.push(n(d, 0, Some(v.len() - 1)));
        }
        let t = SpecTree::new(v).expect("well formed");
        assert_eq!(t.paths().len(), 2);
        assert_eq!(
            t.rows(TreeLayout::SharedPrefixChains),
            16,
            "the b2s8 rectangle"
        );
        assert_eq!(t.rows(TreeLayout::AncestorMask), 13);
        // E4-3: 13 rows fit the 16-row objects (P27-15's 10-row tile refused them)
        assert_eq!(t.ancestor_mask().map(|m| m.len()), Ok(13));
        let s = chains_state(&t);
        assert_eq!(s.pseudo_seqs, 2);
        assert_eq!(s.extra_slot_sets, 8);
        assert_eq!((s.extra_conv, s.extra_kv_tail_pages), (1, 1));
    }

    #[test]
    fn the_slot_and_conv_accounting() {
        assert_eq!(gdn_slot_sets(8, 8), 65, "today's arena");
        assert_eq!(
            gdn_slot_sets(8, 10),
            81,
            "a 10-row tree a sequence: +16 slot sets"
        );
        assert_eq!(conv_state_len(4, 8), 10);
        assert_eq!(conv_state_len(4, 10), 12);
        assert_eq!(conv_state_len(4, 1), 3);
    }

    #[test]
    fn malformed_trees_are_refused() {
        assert!(SpecTree::new(vec![n(2, 0, None)]).is_err());
        assert!(SpecTree::new(vec![n(1, 0, Some(0))]).is_err());
        assert!(
            SpecTree::chain(&[0; 8]).is_err(),
            "the drafter has 7 positions"
        );
    }

    // ---- P27-15b: the arm's host plan --------------------------------------------------------

    /// chain 7 + leaves at depth 1 and 2, as a plan: rows 0..=7 the chain, 8 = a depth-1 sibling,
    /// 9 = a depth-2 sibling under row 1.
    fn plan() -> TreePlan {
        TreePlan::new(chain7_leaves12(), 1000, |n| {
            i64::from(n.depth) * 10 + i64::from(n.cand)
        })
        .expect("10 rows")
    }

    #[test]
    fn p27_15b_the_plan_maps_rows_to_nodes_words_and_rope_by_depth() {
        let p = plan();
        assert_eq!(p.rows(), 10);
        assert_eq!(
            p.words(),
            &chain7_leaves12().ancestor_mask().expect("fits")[..]
        );
        assert_eq!(p.tokens()[0], 1000, "row 0 = the anchor");
        assert_eq!(p.tokens()[8], 11, "the depth-1 sibling (cand 1)");
        assert_eq!(p.parent_row(9), Some(1));
        assert_eq!(p.parent_row(8), Some(0));
        assert_eq!(p.parent_row(0), None);
        // siblings share their depth's position
        let pos = p.positions(500);
        assert_eq!(&pos[..8], &[500, 501, 502, 503, 504, 505, 506, 507]);
        assert_eq!((pos[8], pos[9]), (501, 502));
        assert_eq!(p.path_rows(9), vec![0, 1, 9]);
    }

    #[test]
    fn p27_15b_the_conv_window_walks_the_parent_path_into_the_tail() {
        let p = plan();
        use ConvSrc::{Row, Tail};
        // width 4: three inputs before the row + the row
        assert_eq!(
            p.conv_sources(0, 4),
            vec![Tail(0), Tail(1), Tail(2), Row(0)]
        );
        assert_eq!(p.conv_sources(5, 4), vec![Row(2), Row(3), Row(4), Row(5)]);
        // the depth-2 sibling under row 1 reads rows 0, 1 -- never row 2 (its chain neighbour)
        assert_eq!(p.conv_sources(9, 4), vec![Tail(2), Row(0), Row(1), Row(9)]);
        assert_eq!(p.conv_sources(8, 4), vec![Tail(1), Tail(2), Row(0), Row(8)]);
        // a chain's windows are today's sliding window (row r reads r-3..r)
        let c = TreePlan::new(SpecTree::chain(&[0; 7]).expect("chain"), 1, |_| 2).expect("plan");
        for r in 3..8u32 {
            assert_eq!(
                c.conv_sources(r as usize, 4),
                vec![Row(r - 3), Row(r - 2), Row(r - 1), Row(r)]
            );
        }
    }

    #[test]
    fn p27_15b_the_verdict_moves_the_accepted_path_to_its_depths() {
        let p = plan();
        // the target takes the depth-1 chain node (row 1, token 10), then the depth-2 SIBLING
        // (row 9, token 21), then says 77
        let am = |r: usize| match r {
            0 => 10,
            1 => 21,
            9 => 77,
            _ => -1,
        };
        let v = p.verdict(am);
        assert_eq!(v.path_rows, vec![0, 1, 9]);
        assert_eq!(v.committed, vec![10, 21, 77]);
        assert_eq!(v.accepted_row, 9);
        assert_eq!(v.depth(), 2);
        assert_eq!(v.kv_moves(), vec![(9, 2)], "row 9's K/V to depth 2's slot");
        assert_eq!(v.aux_rows(), &[0, 1, 9]);
        use ConvSrc::{Row, Tail};
        assert_eq!(v.conv_commit(4), vec![Row(0), Row(1), Row(9)]);
        // a chain verdict moves nothing and gathers rows 0..=depth (today's law)
        let c = TreePlan::new(SpecTree::chain(&[0; 7]).expect("chain"), 1, |n| {
            i64::from(n.depth)
        })
        .expect("plan");
        let v = c.verdict(|r| if r < 4 { r as i64 + 1 } else { -1 });
        assert_eq!(v.path_rows, vec![0, 1, 2, 3, 4]);
        assert!(v.kv_moves().is_empty());
        assert_eq!(v.conv_commit(4), vec![Row(2), Row(3), Row(4)]);
        let v0 = c.verdict(|_| -1);
        assert_eq!((v0.depth(), v0.accepted_row), (0, 0));
        assert_eq!(v0.conv_commit(4), vec![Tail(1), Tail(2), Row(0)]);
    }

    #[test]
    fn p27_15b_kv_moves_never_read_a_slot_an_earlier_move_wrote() {
        // every accepted path of every plan the tile holds: simulate the copies on slot labels
        let p = plan();
        for leaf in 0..p.rows() {
            let rows = p.path_rows(leaf);
            let v = TreeVerdict {
                accepted_row: leaf,
                committed: Vec::new(),
                path_rows: rows.clone(),
            };
            let mut slot: Vec<usize> = (0..p.rows()).collect();
            for (from, to) in v.kv_moves() {
                slot[to] = slot[from];
            }
            assert_eq!(&slot[..rows.len()], &rows[..], "leaf {leaf}");
        }
    }

    #[test]
    fn p27_15b_the_record_pair_replaces_the_snapshot_slots() {
        assert_eq!(
            TREE_RECORD_BYTES, 10_368,
            "p27_tree_gdn's TG_REC_FLOATS x 4"
        );
        // the 27B at TP=2: 48 GDN layers x 24 v heads a rank
        assert_eq!(
            tree_record_pair_bytes(48, 24),
            23_887_872,
            "23.9 MB a sequence"
        );
        // P27-10's fear: 16 more snapshot slot sets at 8 sequences, 36 MiB each
        let slots = u64::from(gdn_slot_sets(8, 10) - gdn_slot_sets(8, 8)) * 36 * (1 << 20);
        assert_eq!(slots, 576 * (1 << 20));
        assert!(8 * tree_record_pair_bytes(48, 24) < slots / 3);
    }

    #[test]
    fn p27_15b_the_advance_law_prologues_flushes_and_flips_parity() {
        let mut law = AdvanceLaw::new(2);
        let r0 = law.verify(0);
        assert_eq!(r0.prologue, None, "the first verify has nothing pending");
        law.settle(0, r0, 3);
        assert_eq!(
            law.pending(0),
            Some(PendingAdvance {
                parity: r0.out_parity,
                row: 3
            })
        );
        let r1 = law.verify(0);
        assert_eq!(r1.prologue.map(|p| p.row), Some(3));
        assert_eq!(
            r1.prologue.map(|p| p.parity),
            Some(r0.out_parity),
            "the prologue reads the record the previous verify wrote"
        );
        assert_ne!(r1.out_parity, r0.out_parity, "and writes the other parity");
        law.settle(0, r1, 1);
        // a prefill chunk (or a snapshot) touches the slot: flush first, once
        assert_eq!(law.flush(0).map(|p| p.row), Some(1));
        assert_eq!(law.flush(0), None);
        let r2 = law.verify(0);
        assert_eq!(r2.prologue, None, "a flushed advance is not applied twice");
        assert_ne!(r2.out_parity, r1.out_parity);
        law.settle(0, r2, 0);
        // another slot is independent; a release drops the pending advance unflushed
        assert_eq!(law.pending(1), None);
        law.release(0);
        assert_eq!(law.flush(0), None);
        assert_eq!((law.prologues, law.flushes), (1, 1));
    }

    #[test]
    fn p27_15b_best_first_on_a_hand_lattice() {
        // 2 positions x 3 candidates: depth 1 prefers c0 (score 2) over c1 (1) over c2 (0);
        // under c0 depth 2 is flat, under c1 it is sharp.
        let k = 3;
        let cand: Vec<i64> = (0..6).collect();
        let mut s = vec![0.0f32; 2 * k * k];
        s[..3].copy_from_slice(&[2.0, 1.0, 0.0]);
        s[9..12].copy_from_slice(&[0.0, 0.0, 0.0]); // depth 2 after c0
        s[12..15].copy_from_slice(&[0.0, 9.0, 0.0]); // depth 2 after c1
        let lat = LatticeView {
            cand: &cand,
            scores: &s,
            steps: 2,
            k,
        };
        let t = best_first(&lat, 5, 3, 1.0).expect("tree");
        let got: Vec<(u32, u32, Option<usize>)> = t
            .nodes()
            .iter()
            .map(|n| (n.depth, n.cand, n.parent))
            .collect();
        // p(c0)=.665, p(c1)=.245, p(c1->c1)=.245*.9998, p(c0->*)=.222 each, p(c2)=.090
        assert_eq!(
            got,
            vec![
                (1, 0, None),
                (1, 1, None),
                (2, 1, Some(1)),
                (2, 0, Some(0)),
                (2, 1, Some(0))
            ]
        );
        assert!(
            best_first(&lat, 16, 3, 1.0).is_err(),
            "17 rows: past the 16-row tile (E4-3)"
        );
        assert_eq!(
            SpecTreeForm::parse("ancestor9"),
            Ok(SpecTreeForm::Ancestor9)
        );
        assert_eq!(SpecTreeForm::parse("off"), Ok(SpecTreeForm::default()));
        assert!(SpecTreeForm::parse("chains").is_err());
        assert!(TreePlan::from_lattice(SpecTreeForm::Off, 0, &lat).is_err());
    }

    #[test]
    fn p27_15b_the_counters_line() {
        let p = plan();
        let mut c = SpecTreeCounts::default();
        c.record(&p, &p.verdict(|r| if r == 0 { 10 } else { -1 }));
        c.record(
            &p,
            &p.verdict(|r| match r {
                0 => 10,
                1 => 21,
                _ => -1,
            }),
        );
        assert_eq!((c.steps, c.nodes, c.accepted), (2, 18, 3));
        assert_eq!((c.compactions, c.moved_rows), (1, 1));
        assert!(c
            .line()
            .starts_with("spectree steps 2 nodes/step 9.00 accepted/step 1.500"));
    }
}
