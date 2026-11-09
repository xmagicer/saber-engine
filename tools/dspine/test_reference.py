"""CPU arithmetic/causality tests for the DSpine reference."""
import math
import unittest
try:
    import torch
    import reference as Ref
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch required for numerical reference tests")
class BridgeReferenceTests(unittest.TestCase):
    def bf16(self, values):
        return torch.tensor(values, dtype=torch.bfloat16)

    def test_prepare_read_uses_rms_plus_epsilon_and_bf16_residual_rounding(self):
        raw, normed, rms = Ref.prepare_read(self.bf16([[3, 4]]), eps=1.0)
        self.assertAlmostEqual(float(rms[0]), math.sqrt(12.5), places=6)
        expected = self.bf16([[3/(math.sqrt(12.5)+1), 4/(math.sqrt(12.5)+1)]])
        self.assertTrue(torch.equal(normed, expected))
        x = self.bf16([[1, 1]])
        residual = self.bf16([[1/256, 3/256]])
        raw, _, _ = Ref.prepare_read(x, residual)
        self.assertTrue(torch.equal(raw, self.bf16([[1, 1+4/256]])))

    def test_normalized_codes_use_each_blocks_own_anchor(self):
        p = self.bf16([[9, 9], [3, 4], [9, 9], [0, 0]])
        book = self.bf16([[1, 0], [0, 1]])
        c = Ref.normalize_codes(p, book, torch.tensor([0, 1]), 2)
        self.assertTrue(torch.equal(c[0], self.bf16([math.sqrt(2), 0])))
        self.assertTrue(torch.equal(c[2], self.bf16([0, math.sqrt(2)])))
        self.assertTrue(torch.equal(c[1], self.bf16([math.sqrt(2)*3/5, math.sqrt(2)*4/5])))
        self.assertTrue(torch.equal(c[3], self.bf16([0, 0])))

    def test_predecessor_messages_reset_at_request_boundaries(self):
        p = self.bf16([[0, 0, 2], [0, 0, 900], [0, 0, 20], [0, 0, 800]])
        m = Ref.feature_message(p, self.bf16([0]), 2)
        self.assertTrue(torch.equal(m, self.bf16([[0], [1], [0], [10]])))

    def test_invalid_edges_and_first_rows_never_receive_feature_writes(self):
        raw = self.bf16([[1, 1]]*6)
        delta = self.bf16([[2, 2]]*6)
        out, _ = Ref.feature_finish(raw, delta, torch.ones(6), torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool), 3)
        self.assertTrue(torch.equal(out, self.bf16([[1,1],[3,3],[3,3],[1,1],[1,1],[1,1]])))

    def test_pack_order_and_bf16_tables_match_explicit_small_products(self):
        gate, pred, prev = self.bf16([[1]]), self.bf16([[2]]), self.bf16([[3]])
        self.assertTrue(torch.equal(Ref.pack_bridge_weights(gate,pred,prev), self.bf16([[1],[2],[3]])))
        head, norm, write = self.bf16([[2,0],[0,1]]), self.bf16([1,1]), self.bf16([[-1],[1]])
        tg, tsv = Ref.build_serial_tables(head,norm,write,self.bf16([[0],[1]]),self.bf16([[2],[-2]]),chunk=1)
        self.assertTrue(torch.equal(tg, self.bf16([[-2],[1]])))
        self.assertTrue(torch.equal(tsv, self.bf16([[0,2],[0,-2]])))

    def test_serial_refinement_matches_unfactorized_conditional_greedy(self):
        head = self.bf16([[2,0],[0,1],[1,0],[0,2]])
        norm, write = self.bf16([1,1]), self.bf16([[-1],[1]])
        book, pv = self.bf16([[2],[-2],[4],[-4]]), self.bf16([[0],[1]])
        tg, tsv = Ref.build_serial_tables(head,norm,write,pv,book,chunk=2)
        raw = self.bf16([[0,0],[2,0],[0,0]]*2)
        ids = torch.tensor([2,0,0,3,0,0],dtype=torch.int64)
        cand = torch.tensor([[0,1],[2,3],[0,1],[2,3]],dtype=torch.int64)
        idx, conf = Ref.serial_table(raw,norm,cand,ids,head,tg,tsv,self.bf16([[0,0,0]]*6),torch.ones(6),self.bf16([0]),3)
        output, selected = Ref.serial_chain(idx,cand,3,conf)
        self.assertEqual(output.tolist(), [1,2,0,3])
        expected = []
        for b in range(2):
            previous = int(ids[b*3])
            for pos in range(2):
                # Independent direct residual update and full candidate head, without TG/TSV.
                message = float(book[previous,0]) / 2.0
                state = raw[b*3+1+pos].float() + message*write[:,0].float()
                options = cand[b*2+pos]
                scores = torch.mv(head[options].float(),state)
                previous = int(options[int(scores.argmax())])
                expected.append(previous)
        self.assertEqual(output.tolist(), expected)
        self.assertTrue(bool(((selected > 0) & (selected <= 1)).all()))

    def test_greedy_ties_choose_lower_candidate_lane_and_chains_reset(self):
        idx = torch.tensor([[1,1],[1,0],[0,0],[1,1]],dtype=torch.int64)
        cand = torch.tensor([[10,11],[20,21],[30,31],[40,41]],dtype=torch.int64)
        out, _ = Ref.serial_chain(idx,cand,3)
        self.assertEqual(out.tolist(), [11,20,30,41])
        with self.assertRaises(ValueError):
            Ref.whole_blocks(5,3)


if __name__ == '__main__':
    unittest.main()
