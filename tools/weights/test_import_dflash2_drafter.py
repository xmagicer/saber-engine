import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import import_dflash2_drafter as pack
except ModuleNotFoundError:
    torch = None


class MemoryCheckpoint(dict):
    def get_tensor(self, path):
        return self[path]


@unittest.skipUnless(torch is not None, 'PyTorch and safetensors required')
class DraftImportTests(unittest.TestCase):
    def test_preshuffle_matches_wmma_lane_addresses(self):
        value=torch.arange(32*64).remainder(251).to(torch.uint8).reshape(32,64)
        result=pack.preshuffle(value).reshape(-1).tolist()
        wanted=[]
        for n in range(0,32,16):
            for k in range(0,64,16):
                for row in range(16):
                    wanted.extend(value[n+row,k:k+16].tolist())
        self.assertEqual(result,wanted)

    def test_quantization_uses_independent_128x128_scales(self):
        x=torch.zeros((256,256),dtype=torch.bfloat16)
        x[:128,:128]=.125
        x[:128,128:]=64
        x[128:,:128]=-4
        q,s=pack.block_quantize(x)
        expected=torch.tensor([[.125/448,64/448],[4/448,1.]])
        torch.testing.assert_close(s,expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(pack.dequantize(q,s),x))
        with self.assertRaises(ValueError):pack.block_quantize(x[:127])
        x[0,0]=float('nan')
        with self.assertRaises(ValueError):pack.block_quantize(x)

    def test_identity_qkv_preserves_the_native_bf16_rounding_boundary(self):
        # FP8 byte 3 is 3/512. The scale is the exact BF16 word 0x39f6.
        # A direct multiply lands at a BF16 midpoint and rounds to 0x3638.
        # Two FP32 scale multiplies in the identity GEMM choose its neighbor.
        q=torch.full((128,128),3,dtype=torch.uint8)
        q[0,1]=128  # E4M3FN negative zero
        s=torch.tensor([[0.000469207763671875]],dtype=torch.float32)
        actual=pack.dequantize(q,s).view(torch.int16)
        self.assertEqual(actual[0,0].item(),0x3639)
        self.assertEqual(actual[0,1].item(),-32768)
        direct=(q.view(torch.float8_e4m3fn).float()*s).bfloat16().view(torch.int16)
        self.assertEqual(direct[0,0].item(),0x3638)

    def test_context_kv_comes_from_same_codes_as_each_local_qkv_slice(self):
        torch.manual_seed(5)
        file=MemoryCheckpoint({key:torch.randn((256,128),dtype=torch.bfloat16) for key in ['q','k','v']})
        for rank in [0,1]:
            src=pack.Source(file,False)
            records=[{'name':'attn.qkv.w','layout_id':7,'shard_kind':'Dim0Segments',
                      'sources':['drafter:q','drafter:k','drafter:v']},
                     {'name':'context_kv.kv_projection.w','layout_id':12,'shard_kind':'Dim0Segments',
                      'sources':['drafter:k','drafter:v']}]
            out=dict(pack.converted(src,records,rank));n=384;k=128
            encoded=out['attn.qkv.w'].view(n//16,k//16,16,16).transpose(1,2).contiguous().view(n,k)
            scales=out['attn.qkv.ws'].repeat_interleave(128,0).repeat_interleave(128,1)
            represented=((448.0*encoded.view(torch.float8_e4m3fn).float())*(scales*(1.0/448.0))).bfloat16()
            self.assertTrue(torch.equal(represented[128:],out['context_kv.kv_projection.w']))
            original=torch.cat([file[key][rank*128:(rank+1)*128] for key in ['q','k','v']])
            self.assertLess(float((original.float()-represented.float()).abs().max()),.2)
            self.assertFalse(torch.equal(original[128:],out['context_kv.kv_projection.w']))
            self.assertEqual(src.used,set(file))

    def test_prequantized_width_shards_keep_exact_codes_and_scale_columns(self):
        codes=torch.arange(128*256).remainder(120).to(torch.uint8).view(128,256)
        scale=torch.tensor([[.125,.5]],dtype=torch.bfloat16)
        src=pack.Source(MemoryCheckpoint({'m':codes.view(torch.float8_e4m3fn),'m_scale_inv':scale}),True)
        left,sl=src.projection('m','Dim1',0);right,sr=src.projection('m','Dim1',1)
        self.assertTrue(torch.equal(torch.cat([left,right],1),codes))
        self.assertTrue(torch.equal(torch.cat([sl,sr],1),scale.float()))

    def test_shared_head_requires_matching_target_payload_as_well_as_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=b'abc';(root/'rank0.pack').write_bytes(data)
            records={};target=[]
            for i,(dst,src) in enumerate(pack.TARGET_RECORDS.items()):
                item={'name':src,'shape':[1],'layout_id':i,'dtype':'U8','length':1,
                      'data_off':i,'sha256':hashlib.sha256(data[i:i+1]).hexdigest()}
                records[dst]={**item,'name':dst};target.append(item)
            for dst in pack.INT2_RECORDS:
                records[dst]={'name':dst,'sources':['target:lm_head.weight'],'sha256':'test'}
            rank={'file':'rank0.pack','records':target}
            self.assertEqual(len(pack.shared_proof(root,rank,records)),6)
            (root/'rank0.pack').write_bytes(b'axc')
            with self.assertRaises(ValueError):pack.shared_proof(root,rank,records)


if __name__=='__main__':unittest.main()
