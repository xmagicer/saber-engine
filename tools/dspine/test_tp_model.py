"""Committed-cache regression checks for interleaved TP proposal graph slots."""
import unittest

try:
    import torch
    import tp_graph
    import tp_model
    import test_plain_model as reference_fixture
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None,'PyTorch required')
class TeacherRingChecks(unittest.TestCase):
    def test_absolute_rows_and_payloads_survive_bucket_growth_wrap_and_reuse(self):
        class Teacher:
            h, target_features, layers, window, kheads, dim = 4,1,1,8192,1,4
            device = torch.device('cpu')
            projected_rows = 0

            def project_context(self, features, positions):
                self.projected_rows += len(positions)
                return [features[:,None,:]],[(features*.5)[:,None,:]]

        model = Teacher()
        ring = tp_graph.RingDraft(model,2)
        committed = {}
        rows_total = 0
        for seq, count in [(1,125),(2,130),(1,8),(2,1920),(1,8191),(2,6170),(1,300)]:
            start = len(committed.get(seq,[]))
            positions = torch.arange(start,start+count)
            data = ((positions[:,None] % 97) + 64*seq + torch.arange(4)).bfloat16()
            ring.append(seq,start,data)
            rows_total += count
            committed.setdefault(seq,[]).extend(data.tolist())
            for key,values in committed.items():
                slot = ring.slots[key]
                live = min(len(values),model.window)
                bucket = next(n for n in ring.buckets if n >= live)
                entry = slot.entry(bucket)
                used = entry['positions'] >= 0
                self.assertEqual(int(used.sum()),live)
                where = torch.argsort(entry['positions'][used])
                self.assertEqual(entry['positions'][used][where].tolist(),list(range(len(values)-live,len(values))))
                expected = torch.tensor(values[-live:],dtype=torch.bfloat16)
                self.assertTrue(torch.equal(entry['keys'][0][used][where,0],expected))
                self.assertTrue(torch.equal(entry['values'][0][used][where,0],expected*.5))
        self.assertEqual(model.projected_rows,rows_total)
        with self.assertRaises(ValueError):
            ring.append(1,1,torch.zeros((1,4),dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            ring.append(3,0,torch.zeros((1,4),dtype=torch.bfloat16))
        slot = ring.slots[1]
        ring.clear(1)
        self.assertTrue((slot.positions == tp_graph.INVALID_POSITION).all())
        ring.append(3,0,torch.ones((1,4),dtype=torch.bfloat16))
        self.assertIs(ring.slots[3],slot)
        self.assertEqual(slot.next_position,1)
        self.assertTrue((slot.positions[1:] == tp_graph.INVALID_POSITION).all())

    def test_ring_with_padding_matches_chronological_attention_after_wrap(self):
        model = reference_fixture.PlainModelTests().model()
        ring = tp_graph.RingDraft(model,1)
        features = torch.randn((7,model.h*model.target_features)).bfloat16()
        for start, end in [(0,2),(2,5),(5,7)]:
            model.append_context(9,features[start:end],torch.arange(start,end))
            ring.append(9,start,features[start:end])
            positions = torch.arange(end,end+model.block)
            ids = torch.arange(model.block)
            embeds = torch.randn((model.block,model.h)).bfloat16()
            expected,_ = model.forward_block(9,embeds,ids,positions)
            actual,_ = model.forward_block_tensors(embeds,ids,positions,ring.slots[9].entry(model.window))
            torch.testing.assert_close(actual.float(),expected.float(),atol=.016,rtol=.008)


@unittest.skipUnless(torch is not None,'PyTorch required')
class FusedProjectionChecks(unittest.TestCase):
    def test_qkv_and_gate_up_keep_channel_order_and_batch_stride(self):
        plain = reference_fixture.PlainModelTests().model()
        weights = dict(plain.w)
        layer = 'layers.0.'
        attn = layer + 'self_attn.'
        qkv = torch.cat([weights[attn+part+'_proj.weight'] for part in ('q','k','v')])
        qrows = plain.qheads*plain.dim
        weights[attn+'qkv.weight'] = qkv
        weights[attn+'kv.weight'] = qkv[qrows:]
        weights[layer+'mlp.gate_up.weight'] = torch.cat(
            [weights[layer+'mlp.'+part+'_proj.weight'] for part in ('gate','up')])
        fused = tp_model.FusedProjectionModel(plain.config,weights)
        inputs = torch.eye(plain.h,dtype=torch.bfloat16)
        for method,prefix,rows in [('project_qkv',attn,qkv.shape[0]),
                                   ('project_kv',attn,qkv.shape[0]-qrows),
                                   ('project_gate_up',layer,weights[layer+'mlp.gate_up.weight'].shape[0])]:
            expected = getattr(plain,method)(inputs,prefix)
            actual = getattr(fused,method)(inputs,prefix)
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                self.assertEqual(a.stride(),(rows,1))
                self.assertTrue(torch.equal(a,b),method)
        self.assertEqual(weights[attn+'kv.weight'].untyped_storage().data_ptr(),
                         qkv.untyped_storage().data_ptr())


if __name__ == '__main__':
    unittest.main()
