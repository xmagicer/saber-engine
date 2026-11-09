import math
import unittest
import tree_proposal as Tree


def logs(a,b):
    return [math.log(a) if a else -math.inf,math.log(b) if b else -math.inf]


class TreeProposalTests(unittest.TestCase):
    def test_best_first_budget_keeps_parents_and_uses_conditional_edges(self):
        nodes=Tree.build_best_first_tree([[10,20],[30,40]],
            [[logs(.55,.45)]*2,[logs(.9,.1),logs(.1,.9)]],max_nodes=4)
        self.assertEqual([(n.token,n.parent) for n in nodes],[(10,0),(30,1),(20,0),(40,3)])
        self.assertEqual(Tree.accepted_tree_prefix(nodes,[10,30]),2)
        self.assertEqual(Tree.accepted_tree_prefix(nodes,[20,40]),2)
        self.assertEqual(Tree.accepted_tree_prefix(nodes,[20,30]),1)
        self.assertEqual(Tree.ancestor_masks(nodes),[1,3,7,9,25])

    def test_full_two_level_tree_covers_each_possible_target_path(self):
        nodes=Tree.build_best_first_tree([[1,2],[3,4]],[[logs(.5,.5)]*2]*2,max_nodes=6)
        for first in [1,2]:
            for second in [3,4]:
                self.assertEqual(Tree.accepted_tree_prefix(nodes,[first,second]),2)
        self.assertEqual(Tree.accepted_tree_prefix(nodes,[99,3]),0)
        self.assertEqual(Tree.accepted_tree_prefix(nodes,[1,3,5]),2)

    def test_fifteen_deep_chain_fits_exactly_sixteen_mask_bits(self):
        nodes=Tree.build_best_first_tree([[2*d,2*d+1] for d in range(15)],
            [[logs(1,0)]*2 for _ in range(15)])
        self.assertEqual([n.parent for n in nodes],list(range(15)))
        self.assertEqual(Tree.ancestor_masks(nodes)[-1],65535)
        self.assertEqual(Tree.accepted_tree_prefix(nodes,list(range(0,30,2))),15)

    def test_ties_are_deterministic_and_malformed_probabilities_refused(self):
        args=([[1,2]],[[logs(.5,.5)]*2])
        self.assertEqual(Tree.build_best_first_tree(*args),Tree.build_best_first_tree(*args))
        for matrix in [[[math.nan,0]]*2,[[0,0]]*2,[[0]]*2]:
            with self.assertRaises(ValueError):Tree.build_best_first_tree([[1,2]],[matrix])
        with self.assertRaises(ValueError):Tree.build_best_first_tree([[2,1]],args[1])


if __name__=='__main__':unittest.main()
