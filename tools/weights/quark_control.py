"""Build a matched-core MXFP4 control from the existing Qwen3.8-27B Quark checkpoint.

This is deliberately not a general pack builder. It replaces the template's
MXFP4 linear weights, block scales and row exponents without requantizing them.
Matching target embeddings, norms, convolution and rate parameters are copied too.
The template head, vision and inactive MTP records remain unchanged. Use a new
output directory; never patch the running model. SBR record/chunk/table digests
and the tier identity are regenerated and verified before publishing a manifest.

Fragment order matches GGZ14/vllm-mxfp4 d6162425, radiance_mxfp4.py:109-138.
Wire format follows Saber docs/abi/PACK.md (SBRPACK1 v1/v2).
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import time

import torch


def fragment_order(packed):
    n,k2=packed.shape
    if packed.dtype!=torch.uint8 or n%16 or k2%8:
        raise ValueError('Fragment weight needs U8 [N,K/2], N/K divisible by 16')
    return packed.reshape(n//16,16,k2//8,2,4).permute(0,2,3,1,4).contiguous().reshape(n,k2)


def checkpoint_order(packed):
    n,k2=packed.shape
    return packed.reshape(n//16,k2//8,2,16,4).permute(0,3,1,2,4).contiguous().reshape(n,k2)


def shard_codes(weight,scales,kind,rank,tp=2,segments=None):
    if weight.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or weight.ndim!=2 or scales.shape!=(weight.shape[0],weight.shape[1]//16):
        raise ValueError('Quark FP4/E8M0 shape or dtype mismatch')
    if not 0<=rank<tp or tp!=2:raise ValueError('This control supports TP=2')
    if kind in ['Dim0','Dim0Segments']:
        parts=segments or [weight.shape[0]]
        if sum(parts)!=weight.shape[0] or any(n%tp for n in parts):raise ValueError('Invalid column-parallel segments')
        ws=[];ss=[];offset=0
        for n in parts:
            start=offset+rank*(n//tp);end=start+n//tp
            ws.append(weight[start:end]);ss.append(scales[start:end]);offset+=n
        return torch.cat(ws).contiguous(),torch.cat(ss).contiguous()
    if kind=='Dim1':
        if weight.shape[1]%(16*tp):raise ValueError('Row-parallel cut crosses a 32-element MX group')
        k2=weight.shape[1]//tp;g=scales.shape[1]//tp
        return weight[:,rank*k2:(rank+1)*k2].contiguous(),scales[:,rank*g:(rank+1)*g].contiguous()
    raise ValueError('Unsupported quantized shard kind: '+kind)


class Checkpoint:
    def __init__(self,directory):
        self.path=Path(directory)/'model.safetensors'
        with self.path.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            if size>16*1024*1024:raise ValueError('Unexpected safetensors header size')
            raw=f.read(size);self.header=json.loads(raw);self.header_sha256=hashlib.sha256(raw).hexdigest();self.base=8+size
        self.mapped=torch.from_file(str(self.path),shared=False,size=self.path.stat().st_size,dtype=torch.uint8)
        config=json.loads((Path(directory)/'config.json').read_text())['quantization_config']
        w=config['global_quant_config']['weight']
        if config['quant_method']!='quark' or (w['dtype'],w['group_size'],w['scale_format'])!=('fp4',32,'e8m0'):
            raise ValueError('Expected Quark group-32 E2M1/E8M0 checkpoint')

    def raw_tensor(self,name):
        spec=self.header[name];dtype={'BF16':torch.bfloat16,'F32':torch.float32}.get(spec['dtype'])
        if dtype is None:raise ValueError('Unsupported floating source tensor: '+name)
        start,end=spec['data_offsets']
        if start<0 or self.base+end>self.mapped.numel():raise ValueError('Raw source tensor range mismatch')
        return self.mapped[self.base+start:self.base+end].view(dtype).reshape(spec['shape'])

    def tensor(self,name):
        spec=self.header[name]
        if spec['dtype']!='U8':raise ValueError('Expected raw quantized U8 tensor: '+name)
        start,end=spec['data_offsets'];n=1
        for d in spec['shape']:n*=d
        if end-start!=n or start<0 or self.base+end>self.mapped.numel():raise ValueError('Checkpoint tensor range mismatch')
        return self.mapped[self.base+start:self.base+end].reshape(spec['shape'])


def replacement(checkpoint,record,rank):
    weights=[];scales=[]
    for source in record['sources']:
        source=source.removesuffix('_packed')
        w=checkpoint.tensor(source);s=checkpoint.tensor(source.removesuffix('.weight')+'.weight_scale')
        parts=[2048,2048,6144] if source.endswith('.linear_attn.in_proj_qkv.weight') else None
        w,s=shard_codes(w,s,record['shard_kind'],rank,segments=parts)
        weights.append(w);scales.append(s)
    w=torch.cat(weights,0);s=torch.cat(scales,0)
    if list(w.shape)!=[record['shape'][0],record['shape'][1]//2] or bool((s==255).any()):raise ValueError('Unsupported weight geometry or NaN exponent: '+record['name'])
    return fragment_order(w),s.T.contiguous(),s.max(dim=1).values.contiguous()


def raw_shard(value,record,rank):
    if rank not in [0,1]:raise ValueError('Invalid raw tensor rank')
    kind=record['shard_kind']
    if kind=='Dim0':
        if value.shape[0]%2:raise ValueError('Unshardable raw dimension')
        n=value.shape[0]//2;value=value[rank*n:(rank+1)*n]
    elif kind=='Dim0Segments':
        if not record['name'].endswith('.linear_attn.conv1d.weight') or value.shape[0]!=10240:raise ValueError('Unknown raw segmented tensor')
        parts=[];offset=0
        for n in [2048,2048,6144]:
            parts.append(value[offset+rank*(n//2):offset+(rank+1)*(n//2)]);offset+=n
        value=torch.cat(parts,0)
    elif kind!='Replicated':raise ValueError('Unsupported raw shard kind: '+kind)
    dtype={'BF16':torch.bfloat16,'F32':torch.float32}.get(record['dtype'])
    if dtype is None or list(value.shape)!=record['shape']:raise ValueError('Raw source/record geometry mismatch: '+record['name'])
    return value.to(dtype).contiguous()


class PackPatch:
    """Own a copied pack; preserve geometry while replacing records and their digest entries."""
    def __init__(self,path,records):
        self.file=Path(path).open('r+b');self.map=mmap.mmap(self.file.fileno(),0);self.records={}
        m=self.map
        if m[:8]!=b'SBRPACK1' or struct.unpack_from('<I',m,8)[0] not in [1,2]:raise ValueError('Unsupported SBR pack')
        count=struct.unpack_from('<I',m,20)[0];names_off,names_len=struct.unpack_from('<QQ',m,32)
        self.table_end=names_off+names_len
        if count!=len(records) or names_off!=256+160*count or hashlib.sha256(m[256:self.table_end]).digest()!=m[144:176]:raise ValueError('Pack table mismatch')
        by_name={a['name']:a for a in records};chunk_cursor=0
        self.chunk_off,self.chunk_len=struct.unpack_from('<QQ',m,176) if struct.unpack_from('<I',m,8)[0]==2 else (0,0)
        self.chunk_bytes=struct.unpack_from('<I',m,224)[0] if self.chunk_len else 0
        if self.chunk_len and hashlib.sha256(m[self.chunk_off:self.chunk_off+self.chunk_len]).digest()!=m[192:224]:raise ValueError('Pack chunk blob mismatch')
        for i in range(count):
            at=256+160*i;no,nl=struct.unpack_from('<II',m,at);name=m[names_off+no:names_off+no+nl].decode();record=by_name[name]
            offset,length=struct.unpack_from('<QQ',m,at+24)
            shape=list(struct.unpack_from('<'+'Q'*m[at+18],m,at+40))
            if ('layout_id' in record and struct.unpack_from('<I',m,at+12)[0]!=record['layout_id']) or ('shape' in record and shape!=record['shape']):raise ValueError('Manifest record geometry mismatch: '+name)
            if offset%4096 or offset+length>len(m):raise ValueError('Pack record data range mismatch')
            if offset!=record['data_off'] or length!=record['length'] or m[at+104:at+136].hex()!=record['sha256']:raise ValueError('Manifest and binary table disagree: '+name)
            chunk=None
            if struct.unpack_from('<I',m,at+20)[0]&1:
                chunk=self.chunk_off+chunk_cursor;n,flags=struct.unpack_from('<II',m,chunk)
                if flags or not self.chunk_bytes or n!=(length+self.chunk_bytes-1)//self.chunk_bytes or m[chunk+8:chunk+40]!=m[at+104:at+136]:raise ValueError('Pack chunk entry mismatch')
                chunk_cursor+=40+32*n
            self.records[name]=(at,record,chunk)
        if chunk_cursor!=self.chunk_len:raise ValueError('Pack chunk blob length mismatch')

    def replace(self,name,data):
        at,record,chunk=self.records[name]
        if len(data)!=record['length']:raise ValueError('Replacement changes record byte length: '+name)
        sha=hashlib.sha256(data).digest()
        if sha.hex()==record['sha256']:return
        start=record['data_off'];self.map[start:start+len(data)]=data
        self.map[at+104:at+136]=sha;record['sha256']=sha.hex()
        if chunk is not None:
            self.map[chunk+8:chunk+40]=sha
            for i,start in enumerate(range(0,len(data),self.chunk_bytes)):
                self.map[chunk+40+i*32:chunk+72+i*32]=hashlib.sha256(memoryview(data)[start:start+self.chunk_bytes]).digest()

    def finish(self,tier):
        struct.pack_into('<Q',self.map,72,int(time.time()))
        self.map[112:144]=tier
        self.map[144:176]=hashlib.sha256(self.map[256:self.table_end]).digest()
        if self.chunk_len:self.map[192:224]=hashlib.sha256(self.map[self.chunk_off:self.chunk_off+self.chunk_len]).digest()
        view=memoryview(self.map)
        for _,record,_ in self.records.values():
            start=record['data_off'];sha=hashlib.sha256(view[start:start+record['length']]).hexdigest()
            if sha!=record['sha256']:raise ValueError('Post-write payload verification failed: '+record['name'])
        view.release();self.map.flush()

    def close(self):
        self.map.close();self.file.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--template',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    template=Path(args.template).resolve();output=Path(args.output).resolve()
    if output.exists() or output==template:raise ValueError('Output must be a new directory')
    checkpoint=Checkpoint(args.checkpoint);manifest=json.loads((template/'manifest.json').read_text())
    if [r['rank'] for r in sorted(manifest['ranks'],key=lambda r:r['rank'])]!=[0,1]:raise ValueError('Expected two-rank template')
    # Check every source before copying any multi-gigabyte pack.
    for rank in manifest['ranks']:
        for record in rank['records']:
            if record['layout_id'] in [1,4]:
                for name in record['sources']:
                    name=name.removesuffix('_packed')
                    for key in [name,name.removesuffix('.weight')+'.weight_scale']:
                        if checkpoint.header.get(key,{}).get('dtype')!='U8':raise ValueError('Missing Quark tensor: '+key)
    output.mkdir(mode=0o700);patches=[]
    try:
        for rank in manifest['ranks']:
            if rank['file']!=f"rank{rank['rank']}.pack":raise ValueError('Unexpected shard file name')
            path=output/rank['file'];shutil.copyfile(template/rank['file'],path)
            patch=PackPatch(path,rank['records']);patches.append(patch);count=0
            for record in rank['records']:
                if record['layout_id'] not in [1,4]:continue
                w,s,ref=replacement(checkpoint,record,rank['rank']);base=record['name'].removesuffix('.w')
                for name,tensor in [(base+'.w',w),(base+'.ws',s),(base+'.wref',ref)]:
                    patch.replace(name,tensor.numpy().tobytes())
                    changed=patch.records[name][1]
                    changed['sources']=[n.removesuffix('_packed') for n in record['sources']]
                    changed['source_origin']='quark-matched-core-control'
                count+=1
                if count%32==0:print(json.dumps({'rank':rank['rank'],'repacked_linears':count}),flush=True)
            raw_count=0
            for record in rank['records']:
                if not record['name'].startswith('model.language_model.') or record['layout_id'] not in [0,5]:continue
                name=record['name']
                if name in checkpoint.header:
                    value=raw_shard(checkpoint.raw_tensor(name),record,rank['rank'])
                    origin='quark-matched-core-control'
                elif name.endswith(('.self_attn.k_scale','.self_attn.v_scale')) and record['shape']==[1]:
                    # The old Quark checkpoint has no KV quantization scales; its FP8 cache uses 1.
                    value=torch.ones(1,dtype=torch.bfloat16);origin='unit-kv-scale'
                else:raise ValueError('Missing matching raw target parameter: '+name)
                data=memoryview(value.view(torch.uint8).flatten().numpy())
                patch.replace(name,data);record['source_origin']=origin;raw_count+=1
            print(json.dumps({'rank':rank['rank'],'matched_raw_parameters':raw_count}),flush=True)

        content=[(rank['rank'],[(rec['name'],rec['sha256']) for rec in rank['records']]) for rank in manifest['ranks']]
        descriptor={'format':'quark-matched-core-control/1','template_tier':manifest['tier']['id'],'checkpoint_header_sha256':checkpoint.header_sha256,'checkpoint_size_bytes':checkpoint.path.stat().st_size,'record_contents_sha256':hashlib.sha256(json.dumps(content,separators=(',',':')).encode()).hexdigest()}
        tier=hashlib.sha256(json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()).digest()
        for patch in patches:patch.finish(tier)
        for patch in patches:patch.close()
        patches=[]
        manifest['tier']={'id':tier.hex(),'descriptor':descriptor}
        manifest['built_unix']=int(time.time())
        manifest['builder']=str(manifest.get('builder',''))+' + tools/weights/quark_control.py'
        manifest['quark_target_control']={'checkpoint':str(checkpoint.path),'checkpoint_header_sha256':checkpoint.header_sha256,'template':str(template),'requantized':False,'head_vision_inactive_mtp':'unchanged template bytes','target_raw_parameters':'matching Quark checkpoint','kv_scale':1.0,'created_unix':int(time.time())}
        for rank in manifest['ranks']:
            with (output/rank['file']).open('rb') as f:rank['file_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (output/'verified.json').write_text(json.dumps({'all_record_digests_verified':True,'tier_id':tier.hex(),'quark_target_control':True},indent=2))
        print(json.dumps({'passed':True,'output':str(output),'tier_id':tier.hex()}),flush=True)
    finally:
        for patch in patches:patch.close()


if __name__=='__main__':main()
