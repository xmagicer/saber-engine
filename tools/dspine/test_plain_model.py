import copy
import unittest
try:
    import torch
    import plain_model as Model
except ModuleNotFoundError:
    torch=None


@unittest.skipUnless(torch is not None,'PyTorch required')
class PlainModelTests(unittest.TestCase):
    def test_sglang_backbone_norm_multiplies_before_bf16_rounding(self):
        x=torch.tensor([[.030029296875,1.]],dtype=torch.bfloat16)
        weight=torch.tensor([1.296875,1.],dtype=torch.bfloat16)
        actual=Model.rms_norm(x,weight,1e-6)
        self.assertEqual(float(actual[0,0]),.054931640625)
        # Independent analytic scalar result differs from the HF-style early cast.
        self.assertNotEqual(float(actual[0,0]),.05517578125)

    def test_fused_norm_uses_unrounded_sum_and_stores_rounded_residual(self):
        x=torch.tensor([[.030029296875,.030029296875]],dtype=torch.bfloat16)
        residual=torch.tensor([[.01171875,.10009765625]],dtype=torch.bfloat16)
        weight=torch.tensor([1.296875,1.],dtype=torch.bfloat16)
        normalized,stored=Model.add_rms_norm(x,residual,weight,1e-6)
        self.assertEqual(float(normalized[0,0]),.55859375)
        self.assertNotEqual(float(normalized[0,0]),.5625)
        self.assertTrue(torch.equal(stored,torch.tensor([[.041748046875,.130126953125]],dtype=torch.bfloat16)))

    def model(self):
        torch.manual_seed(9)
        h,d,q,k,layers,c,r,v=8,4,2,1,2,4,2,16
        cfg={'hidden_size':h,'num_hidden_layers':layers,'num_attention_heads':q,'num_key_value_heads':k,'head_dim':d,
             'block_size':4,'sliding_window':3,'rope_theta':10000000,'rms_norm_eps':1e-6,
             'code_bridge_second_pass':{'mode':'serial_table','first_exact':layers-1,'head_topk':2},
             'dflash_config':{'draft_block_causal':True,'selected_prefix_terminal':False,
                'selected_prefix_feature_write_mode':'independent','selected_prefix_feature_writes':True,
                'selected_prefix_feature_layers':[1,2],'target_layer_ids':[1,16],
                'selected_prefix_code_bridge':{'code_size':c,'rank':r,'eps':1e-6}}}
        def random(*shape):return (torch.randn(shape)*.05).bfloat16()
        w={'fc.weight':torch.cat((torch.eye(h),torch.zeros(h,h)),1).bfloat16(),
           'hidden_norm.weight':torch.ones(h,dtype=torch.bfloat16),'norm.weight':torch.ones(h,dtype=torch.bfloat16)}
        for i in range(layers):
            p=f'layers.{i}.';a=p+'self_attn.'
            for name,shape in [('q_proj.weight',(q*d,h)),('k_proj.weight',(k*d,h)),('v_proj.weight',(k*d,h)),('o_proj.weight',(h,q*d))]:w[a+name]=random(*shape)
            w[a+'q_norm.weight']=torch.ones(d,dtype=torch.bfloat16);w[a+'k_norm.weight']=torch.ones(d,dtype=torch.bfloat16)
            w[p+'input_layernorm.weight']=torch.ones(h,dtype=torch.bfloat16);w[p+'post_attention_layernorm.weight']=torch.ones(h,dtype=torch.bfloat16)
            for name,shape in [('gate_proj.weight',(12,h)),('up_proj.weight',(12,h)),('down_proj.weight',(h,12))]:w[p+'mlp.'+name]=random(*shape)
        p='selected_prefix_bridge.'
        for name,shape in [('codebook',(v,c)),('early_read_weights',(layers-1,c,h)),('read_weight',(c,h)),
                           ('feature_write_weights',(layers,h,r)),('gate.weight',(r,c)),('predecessor_gate.weight',(r,c)),('prev.weight',(r,c))]:w[p+name]=random(*shape)
        w[p+'gate.bias']=torch.zeros(r,dtype=torch.bfloat16);w[p+'out.weight']=torch.zeros(c,r,dtype=torch.bfloat16)
        for name in ['draft_block_causal_version','last_write_refine_version',p+'independent_write_version',p+'terminal_disabled_version']:w[name]=torch.tensor(1,dtype=torch.int64)
        return Model.TorchDspine(cfg,w)

    def test_context_chunking_and_window_are_consistent(self):
        a,b=self.model(),self.model();aux=torch.randn(7,16).bfloat16()
        a.append_context(1,aux,torch.arange(7));b.append_context(1,aux[:3],torch.arange(3));b.append_context(1,aux[3:],torch.arange(3,7))
        self.assertEqual(a.cache[1]['positions'].tolist(),[4,5,6])
        for ka,kb in zip(a.cache[1]['keys'],b.cache[1]['keys']):self.assertTrue(torch.allclose(ka.float(),kb.float(),atol=.01,rtol=.01))

    def test_future_queries_cannot_change_earlier_rows_or_teacher_cache(self):
        model=self.model();model.append_context(1,torch.randn(2,16).bfloat16(),torch.arange(2))
        x=torch.randn(4,8).bfloat16();ids=torch.tensor([1,3,3,3]);pos=torch.arange(2,6)
        before=[k.clone() for k in model.cache[1]['keys']]
        a,_=model.forward_block(1,x,ids,pos);future=x.clone();future[-1]+=16
        b,_=model.forward_block(1,future,ids,pos)
        self.assertTrue(torch.equal(a[:-1],b[:-1]));self.assertFalse(torch.equal(a[-1],b[-1]))
        self.assertEqual(model.cache[1]['next'],2)
        for old,new in zip(before,model.cache[1]['keys']):self.assertTrue(torch.equal(old,new))

    def test_request_caches_are_isolated_and_gaps_refused(self):
        model=self.model();aux=torch.randn(2,16).bfloat16();model.append_context(10,aux,torch.arange(2));model.append_context(20,-aux,torch.arange(2))
        x=torch.randn(4,8).bfloat16();ids=torch.tensor([1,3,3,3]);pos=torch.arange(2,6)
        a,_=model.forward_block(10,x,ids,pos);model.append_context(20,aux[:1],torch.tensor([2]));b,_=model.forward_block(10,x,ids,pos)
        self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):model.append_context(30,aux,torch.arange(1,3))
        with self.assertRaises(ValueError):model.forward_block(10,x,ids,torch.arange(3,7))
        model.clear(10);self.assertNotIn(10,model.cache);self.assertIn(20,model.cache)

    def test_expired_teacher_rows_cannot_change_next_block(self):
        a,b=self.model(),self.model();aux=torch.randn(8,16).bfloat16();other=aux.clone();other[:5]*=99
        a.append_context(1,aux,torch.arange(8));b.append_context(1,other,torch.arange(8))
        x=torch.randn(4,8).bfloat16();ids=torch.tensor([1,3,3,3]);pos=torch.arange(8,12)
        ya,_=a.forward_block(1,x,ids,pos);yb,_=b.forward_block(1,x,ids,pos)
        self.assertTrue(torch.equal(ya,yb))

    def test_tensor_body_matches_validated_forward_without_mutating_teacher(self):
        model=self.model();model.append_context(1,torch.randn(2,16).bfloat16(),torch.arange(2))
        x=torch.randn(4,8).bfloat16();ids=torch.tensor([1,3,3,3]);pos=torch.arange(2,6)
        expected,stash=model.forward_block(1,x,ids,pos)
        actual,other=model.forward_block_tensors(x,ids,pos,model.cache[1])
        self.assertTrue(torch.equal(actual,expected))
        for key in stash:self.assertTrue(torch.equal(stash[key],other[key]))
        self.assertEqual(model.cache[1]['next'],2)


if __name__=='__main__':unittest.main()
