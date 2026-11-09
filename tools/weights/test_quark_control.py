import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
try:
    import torch
    import quark_control as Q
except ModuleNotFoundError:
    torch=None


@unittest.skipUnless(torch is not None,'PyTorch required')
class QuarkControlTests(unittest.TestCase):
    def test_fragment_order_matches_explicit_lane_address_and_round_trips(self):
        w=torch.arange(32*64,dtype=torch.int64).remainder(251).to(torch.uint8).reshape(32,64)
        actual=Q.fragment_order(w)
        expected=[]
        for nt in range(2):
            for ks in range(8):
                for half in range(2):
                    for row in range(16):expected.extend(w[16*nt+row,8*ks+4*half:8*ks+4*half+4].tolist())
        self.assertEqual(actual.flatten().tolist(),expected)
        self.assertTrue(torch.equal(Q.checkpoint_order(actual),w))

    def test_row_shards_preserve_packed_codes_and_group_exponents(self):
        w=torch.arange(32*64).remainder(251).to(torch.uint8).reshape(32,64)
        s=torch.arange(32*4).remainder(17).to(torch.uint8).reshape(32,4)
        shards=[Q.shard_codes(w,s,'Dim1',r) for r in [0,1]]
        self.assertTrue(torch.equal(torch.cat([a for a,b in shards],1),w))
        self.assertTrue(torch.equal(torch.cat([b for a,b in shards],1),s))

    def test_segmented_qkv_shards_select_each_segment_independently(self):
        w=torch.arange(96*32).remainder(251).to(torch.uint8).reshape(96,32)
        s=torch.arange(96*2).remainder(47).to(torch.uint8).reshape(96,2)
        for rank,indices in [(0,list(range(0,8))+list(range(16,32))+list(range(48,72))),
                             (1,list(range(8,16))+list(range(32,48))+list(range(72,96)))]:
            a,b=Q.shard_codes(w,s,'Dim0Segments',rank,segments=[16,32,48])
            self.assertTrue(torch.equal(a,w[indices]))
            self.assertTrue(torch.equal(b,s[indices]))

    def test_invalid_shapes_and_group_crossing_are_refused(self):
        with self.assertRaises(ValueError):Q.fragment_order(torch.zeros(15,32,dtype=torch.uint8))
        with self.assertRaises(ValueError):Q.shard_codes(torch.zeros(16,16,dtype=torch.uint8),torch.zeros(16,1,dtype=torch.uint8),'Dim1',0)
        with self.assertRaises(ValueError):Q.shard_codes(torch.zeros(16,32,dtype=torch.uint8),torch.zeros(16,2,dtype=torch.uint8),'Dim0Segments',0,segments=[3,13])

    def test_raw_norm_and_rate_parameters_keep_checkpoint_precision(self):
        values=torch.arange(48,dtype=torch.bfloat16)/16
        rec={'name':'model.language_model.layers.0.linear_attn.A_log','shard_kind':'Dim0','shape':[24],'dtype':'F32'}
        for rank in [0,1]:
            actual=Q.raw_shard(values,rec,rank)
            self.assertEqual(actual.dtype,torch.float32)
            self.assertTrue(torch.equal(actual,values[rank*24:(rank+1)*24].float()))
        rec={'name':'norm','shard_kind':'Replicated','shape':[48],'dtype':'BF16'}
        self.assertTrue(torch.equal(Q.raw_shard(values,rec,0),values))

    def test_raw_conv_keeps_qkv_segment_shards_aligned_with_the_linears(self):
        values=torch.arange(10240*4,dtype=torch.float32).reshape(10240,1,4)
        rec={'name':'model.language_model.layers.0.linear_attn.conv1d.weight','shard_kind':'Dim0Segments','shape':[5120,1,4],'dtype':'F32'}
        expected=torch.cat((values[1024:2048],values[3072:4096],values[7168:10240]),0)
        self.assertTrue(torch.equal(Q.raw_shard(values,rec,1),expected))
        with self.assertRaises(ValueError):Q.raw_shard(values,{**rec,'name':'unknown'},1)

    def fixture(self,directory):
        data=bytes(i%251 for i in range(8193));sha=hashlib.sha256(data).digest();name=b'fixture'
        header=bytearray(256);header[:8]=b'SBRPACK1'
        struct.pack_into('<IIII',header,8,2,256,160,1)
        names_off=416;chunk_off=names_off+len(name);chunk_size=4096;n=3
        blob=struct.pack('<II',n,0)+sha+b''.join(hashlib.sha256(data[i:i+chunk_size]).digest() for i in range(0,len(data),chunk_size))
        data_off=4096;file_bytes=16384
        struct.pack_into('<QQQQQ',header,24,256,names_off,len(name),data_off,file_bytes)
        struct.pack_into('<II',header,64,2,0)
        header[112:144]=b'T'*32
        struct.pack_into('<QQ',header,176,chunk_off,len(blob));header[192:224]=hashlib.sha256(blob).digest();struct.pack_into('<I',header,224,chunk_size)
        record=bytearray(160);struct.pack_into('<IIII',record,0,0,len(name),0,0);record[18]=1;struct.pack_into('<I',record,20,1);struct.pack_into('<QQQ',record,24,data_off,len(data),len(data));record[104:136]=sha
        header[144:176]=hashlib.sha256(record+name).digest()
        payload=header+record+name+blob;payload+=bytes(data_off-len(payload));payload+=data;payload+=bytes(file_bytes-len(payload))
        path=Path(directory)/'rank0.pack';path.write_bytes(payload)
        meta={'name':'fixture','data_off':data_off,'length':len(data),'sha256':sha.hex(),'layout_id':0,'shape':[len(data)]}
        return path,meta,data,chunk_off,len(blob)

    def test_record_chunk_and_table_hashes_are_regenerated(self):
        with tempfile.TemporaryDirectory() as d:
            path,meta,data,chunk_off,chunk_len=self.fixture(d);changed=bytes((x+7)%256 for x in data)
            patch=Q.PackPatch(path,[meta]);patch.replace('fixture',memoryview(changed));patch.finish(b'N'*32);patch.close()
            raw=path.read_bytes();self.assertEqual(raw[112:144],b'N'*32)
            self.assertEqual(raw[144:176],hashlib.sha256(raw[256:423]).digest())
            self.assertEqual(raw[192:224],hashlib.sha256(raw[chunk_off:chunk_off+chunk_len]).digest())
            self.assertEqual(raw[360:392],hashlib.sha256(changed).digest())
            for i,start in enumerate(range(0,len(data),4096)):
                at=chunk_off+40+i*32;self.assertEqual(raw[at:at+32],hashlib.sha256(changed[start:start+4096]).digest())
            # A second independent parser/open accepts the new table and chunk identities.
            patch=Q.PackPatch(path,[meta]);patch.finish(b'N'*32);patch.close()

    def test_bad_replacement_does_not_mutate_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            path,meta,data,_,_=self.fixture(d);original=path.read_bytes();patch=Q.PackPatch(path,[meta])
            with self.assertRaisesRegex(ValueError,'byte length'):patch.replace('fixture',b'bad')
            patch.close();self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
