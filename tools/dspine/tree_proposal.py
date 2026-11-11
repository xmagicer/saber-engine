"""Offline fixed-budget trees from a DSpine conditional candidate lattice.

Construction reads proposer probabilities only. Canonical target tokens are
supplied separately to the acceptance walk; they never influence the tree.
Fifteen nodes plus the anchor fit a 16-row verifier and a 16-bit ancestor mask.
"""
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class TreeNode:
    token: int
    parent: int
    depth: int
    lane: int
    log_probability: float


def build_best_first_tree(candidates,log_probabilities,max_nodes=15,max_depth=None):
    """Return parent-before-child rows; row zero is the already known anchor."""
    depths=len(candidates)
    if not 1<=depths<=15 or not 1<=max_nodes<=15 or len(log_probabilities)!=depths:
        raise ValueError('A 16-row tree requires 1..15 depths and nodes')
    limit=depths if max_depth is None else max_depth
    if not 1<=limit<=depths:
        raise ValueError('Invalid maximum depth')
    k=len(candidates[0])
    if not 2<=k<=64 or k&(k-1):
        raise ValueError('Candidate count must be a power of two in [2,64]')
    for ids,matrix in zip(candidates,log_probabilities):
        if (len(ids)!=k or any(type(v) is not int or v<0 for v in ids)
                or any(a>=b for a,b in zip(ids,ids[1:])) or len(matrix)!=k):
            raise ValueError('Each candidate row must contain sorted unique token ids')
        for row in matrix:
            if len(row)!=k or any(math.isnan(v) or v>0 for v in row):
                raise ValueError('Invalid conditional log probabilities')
            if not math.isclose(math.fsum(math.exp(v) for v in row),1.,abs_tol=1e-4):
                raise ValueError('Conditional probabilities must sum to one')
    heap=[]

    def offer(parent,depth,previous_lane,prefix):
        if depth>=limit:
            return
        for lane,value in enumerate(log_probabilities[depth][previous_lane]):
            probability=prefix+value
            if math.isfinite(probability):
                # Prefer shallower rows, then smaller token ids on equal mass.
                heapq.heappush(heap,(-probability,depth+1,candidates[depth][lane],parent,lane))

    offer(0,0,0,0.)
    nodes=[]
    while heap and len(nodes)<max_nodes:
        negative,depth,token,parent,lane=heapq.heappop(heap)
        nodes.append(TreeNode(token,parent,depth,lane,-negative))
        offer(len(nodes),depth,lane,-negative)
    return nodes


def ancestor_masks(nodes):
    """Include each node itself and every ancestor, including anchor bit zero."""
    if len(nodes)>15:
        raise ValueError('Ancestor masks have sixteen bits including the anchor')
    masks=[1]
    for index,node in enumerate(nodes,1):
        if not 0<=node.parent<index:
            raise ValueError('Tree rows must follow their parents')
        expected=1 if node.parent==0 else nodes[node.parent-1].depth+1
        if node.depth!=expected:
            raise ValueError('Depth must follow the parent')
        masks.append(masks[node.parent]|(1<<index))
    return masks


def accepted_tree_prefix(nodes,target_tokens):
    """Count only the path whose input tokens match the canonical target prefix."""
    ancestor_masks(nodes)
    children={}
    for index,node in enumerate(nodes,1):
        key=(node.parent,node.token)
        if key in children:
            raise ValueError('Duplicate token under the same parent')
        children[key]=index
    parent=0;accepted=0
    for token in target_tokens:
        child=children.get((parent,token))
        if child is None:
            break
        accepted+=1;parent=child
    return accepted
