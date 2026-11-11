"""Import a BF16 Qwen3.8-27B-DFlash2 block-8 or block-16 checkpoint into a TP=2 SBR pack.

The supported template is the shipped block-8 DFlash2 W8 rerank pack. Its
record layout, static RoPE table and target-derived head are reusable. Every
learned drafter record is rebuilt. --check-roundtrip exercises the same
transforms using the template's original block-FP8 source as a control.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time

import torch
from safetensors import safe_open

from quark_control import PackPatch


TEMPLATE_TIER = '57d1c407901dd2e16d74ae37b5020dffbcceb8cfc761ca6764b672537c80c2b6'
TARGET_RECORDS = {
    'input_embedding.weight': 'model.language_model.embed_tokens.weight',
    'head.rerank.w8': 'lm_head.w8',
    'head.rerank.ws': 'lm_head.ws',
}
INT2_RECORDS = {'head.int2.wq', 'head.int2.s', 'head.int2.zs'}
ROPE = 'context_kv.rope.cos_sin'


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def record_digest(f, record):
    f.seek(record['data_off'])
    remaining = record['length']
    sha = hashlib.sha256()
    while remaining:
        chunk = f.read(min(remaining, 8 << 20))
        if not chunk:
            raise ValueError(f"Truncated record {record['name']}")
        sha.update(chunk)
        remaining -= len(chunk)
    return sha.hexdigest()


def read_manifest(root):
    manifest = json.loads((root/'manifest.json').read_text())
    ranks = sorted(manifest['ranks'], key=lambda rank:rank['rank'])
    if (manifest['magic'] != 'SBRPACK1' or manifest['format_version'] != 2
            or manifest['tp_size'] != 2 or [s['rank'] for s in ranks] != [0, 1]):
        raise ValueError('Expected a TP=2 SBRPACK1 v2 pack')
    for rank in ranks:
        if rank['file'] != f"rank{rank['rank']}.pack" or (root/rank['file']).stat().st_size != rank['file_bytes']:
            raise ValueError('Unexpected shard file')
        if len({r['name'] for r in rank['records']}) != len(rank['records']):
            raise ValueError('Duplicate record in manifest')
    manifest['ranks'] = ranks
    return manifest


def check_config(config, control):
    expected = {'architectures':['DFlash2DraftModel'], 'hidden_size':5120, 'head_dim':128,
                'intermediate_size':17408, 'num_hidden_layers':5, 'num_attention_heads':32,
                'num_key_value_heads':8, 'num_target_layers':64, 'vocab_size':248320,
                'is_causal':False, 'hidden_act':'silu', 'attention_bias':False,
                'rms_norm_eps':1e-6, 'sliding_window':2048, 'use_sliding_window':True,
                'rope_parameters':{'rope_theta':10000000,'rope_type':'default'}}
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Checkpoint geometry differs from the native DFlash2 profile')
    block = config.get('dflash_config', {}).get('block_size')
    if block not in ((8,) if control else (8, 16)):
        raise ValueError('Unsupported DFlash2 training block size')
    wanted={'block_size':block,'conv_group_size':16,'conv_kernel_size':2,
            'mask_token_id':248070,'selector_rank':256,'selector_top_k':16,
            'target_layer_ids':[5,19,33,47,61]}
    if config.get('dflash_config') != wanted:
        raise ValueError('Unexpected attention taps, selector or convolution configuration')
    quant = config.get('quantization_config', {})
    if control:
        if quant.get('fmt') != 'e4m3' or quant.get('weight_block_size') != [128,128]:
            raise ValueError('Round-trip control requires the original block-FP8 checkpoint')
    elif quant:
        raise ValueError('The conversion input must contain original BF16 parameters')


def shard(value, kind, rank):
    if rank not in (0,1):
        raise ValueError('Invalid TP rank')
    if kind == 'Replicated':
        return value
    if kind not in ('Dim0', 'Dim0Segments', 'Dim1'):
        raise ValueError(f'Unsupported shard kind {kind}')
    dim = 1 if kind == 'Dim1' else 0
    if value.shape[dim] % 2:
        raise ValueError('Uneven tensor-parallel shard')
    width = value.shape[dim] // 2
    return value.narrow(dim, rank * width, width)


def block_quantize(weight):
    """Quantize a logical local matrix, independently per aligned 128x128 tile."""
    n,k = weight.shape
    if weight.dtype != torch.bfloat16 or n % 128 or k % 128:
        raise ValueError('Expected an aligned 2-D BF16 projection')
    quant = torch.empty((n,k),dtype=torch.uint8)
    scales = torch.empty((n//128,k//128),dtype=torch.float32)
    for row in range(0,n,128):
        tile = weight[row:row+128].float().reshape(128,k//128,128)
        maximum = tile.abs().amax(dim=(0,2))
        if not bool(maximum.isfinite().all()):
            raise ValueError('Nonfinite projection weight')
        scale = maximum / 448.0
        scale[maximum == 0] = 1.0
        encoded = (tile / scale[None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        quant[row:row+128] = encoded.view(torch.uint8).reshape(128,k)
        scales[row//128] = scale
    return quant,scales


def preshuffle(quant):
    n,k = quant.shape
    if quant.dtype != torch.uint8 or n % 16 or k % 32:
        raise ValueError('Invalid FP8 WMMA fragment dimensions')
    return quant.reshape(n//16,16,k//32,2,16).permute(0,2,3,1,4).contiguous().reshape(n//16,k*16)


def dequantize(quant,scale):
    n,k = quant.shape
    if scale.shape != (n//128,k//128):
        raise ValueError('Projection scale geometry differs')
    # The upstream fused context KV builder sends a BF16 identity matrix through
    # the quantized QKV projection. The one-hot activation encodes 1 as 448,
    # with FP32 activation scale 1/448. dflash_gemm.hip multiplies the two
    # scales first, then scales the FP32 tile dot before rounding to BF16.
    values = quant.view(torch.float8_e4m3fn).float().reshape(n//128,128,k//128,128)
    identity_dot = values * 448.0
    combined_scale = scale * (1.0 / 448.0)
    return (identity_dot * combined_scale[:,None,:,None]).reshape(n,k).to(torch.bfloat16)


class Source:
    def __init__(self,file,control):
        self.file=file
        self.control=control
        self.used=set()

    def get(self,name,dtype):
        self.used.add(name)
        value=self.file.get_tensor(name)
        if value.dtype != dtype:
            raise ValueError(f'{name}: expected {dtype}, got {value.dtype}')
        return value

    def projection(self,name,kind,rank):
        if not self.control:
            return block_quantize(shard(self.get(name,torch.bfloat16),kind,rank))
        q=shard(self.get(name,torch.float8_e4m3fn),kind,rank).view(torch.uint8)
        scale=shard(self.get(name+'_scale_inv',torch.bfloat16),kind,rank).float()
        if scale.shape != (q.shape[0]//128,q.shape[1]//128):
            raise ValueError(f'Scale shape mismatch for {name}')
        return q,scale


def sources(record):
    names=record['sources']
    if not names or not all(n.startswith('drafter:') for n in names):
        raise ValueError(f"Unknown drafter source for {record['name']}")
    return [n.removeprefix('drafter:') for n in names]


def converted(source,records,rank):
    for record in records:
        name=record['name']; layout=record['layout_id']; kind=record['shard_kind']
        if name in TARGET_RECORDS or name in INT2_RECORDS or name == ROPE:
            continue
        names=sources(record)
        if layout == 7:
            pairs=[source.projection(n,kind,rank) for n in names]
            yield name,preshuffle(torch.cat([x[0] for x in pairs],dim=0))
            yield name+'s',torch.cat([x[1] for x in pairs],dim=0)
        elif layout == 8:
            continue  # emitted together with the packed matrix
        elif name == 'context_kv.kv_projection.w' and layout == 12:
            yield name,torch.cat([dequantize(*source.projection(n,kind,rank)) for n in names],dim=0)
        elif name == 'context_kv.k_norm.weight' and layout == 6 and kind == 'Replicated':
            yield name,torch.stack([source.get(n,torch.bfloat16) for n in names])
        elif layout == 0 and len(names)==1 and kind=='Replicated':
            yield name,source.get(names[0],torch.bfloat16)
        else:
            raise ValueError(f'Unexpected learned record {name}, layout {layout}')


def shared_proof(target, target_rank, records):
    """Prove the immutable template's input embedding and rerank head match the selected target."""
    target_records={r['name']:r for r in target_rank['records']}
    proof={}
    with (target/target_rank['file']).open('rb') as raw:
        for dst,src in TARGET_RECORDS.items():
            a,b=records[dst],target_records[src]
            if any(a[k]!=b[k] for k in ('shape','layout_id','dtype','length','sha256')):
                raise ValueError(f'Template {dst} does not match selected target {src}')
            if record_digest(raw,b)!=a['sha256']:
                raise ValueError(f'Corrupt target bytes for {src}')
            proof[dst]=a['sha256']
    for name in INT2_RECORDS:
        if records[name]['sources'] != ['target:lm_head.weight']:
            raise ValueError('Template int2 records must derive from the verified target head')
        proof[name]=records[name]['sha256']
    return proof


def run(args):
    template,target,checkpoint=(p.resolve() for p in [args.template,args.target_pack,args.checkpoint])
    result=read_manifest(template);target_meta=read_manifest(target)
    if result['tier']['id'] != TEMPLATE_TIER or any(len(x['records'])!=96 for x in result['ranks']):
        raise ValueError('Use the original TP=2 Qwen3.8 DFlash2 FP8-head template')
    config_path=checkpoint/'config.json';check_config(json.loads(config_path.read_text()),args.check_roundtrip)
    weights=checkpoint/'model.safetensors'; source_sha=file_digest(weights)
    if args.expected_sha256 and args.expected_sha256.lower()!=source_sha:raise ValueError('Checkpoint SHA-256 mismatch')
    proof=[]
    for rank,other in zip(result['ranks'],target_meta['ranks']):
        records={a['name']:a for a in rank['records']};proof.append(shared_proof(target,other,records))
    tier_desc={'format':'dflash2-import/1','source_sha256':source_sha,'config_sha256':file_digest(config_path),
               'template_tier':TEMPLATE_TIER,'target_tier':target_meta['tier']['id'],'shared_records':proof,
               'quantization':'fp8-e4m3fn, per 128x128 absmax/448, f32 scales, zero blocks use scale 1'}
    tier=digest(tier_desc);plan=copy.deepcopy(result['plan']['descriptor']);plan['source']['tier_id']=tier
    plan['transforms'][plan['transforms'].index('weight_scale_inv bf16->f32')]='bf16->fp8 block quantization, f32 scales'
    plan_hash=digest(plan)
    out=args.output.resolve() if args.output else None
    if out and (out.exists() or any(out==p or p in out.parents for p in [template,target,checkpoint])):
        raise ValueError('Choose a new output directory outside all input packs')
    if out:out.parent.mkdir(parents=True,exist_ok=True)
    parent=out.parent if out else None
    with tempfile.TemporaryDirectory(prefix='.dflash2-import-',dir=parent) as temp, safe_open(str(weights),framework='pt',device='cpu') as file:
        stage=Path(temp)/'pack'
        if out:stage.mkdir(mode=0o700)
        for rank in result['ranks']:
            index=rank['rank'];records={a['name']:a for a in rank['records']};source=Source(file,args.check_roundtrip)
            seen=set(TARGET_RECORDS)|INT2_RECORDS|{ROPE};patch=None
            try:
                if out:
                    dest=stage/rank['file'];shutil.copyfile(template/rank['file'],dest)
                    patch=PackPatch(dest,rank['records'])
                for name,value in converted(source,rank['records'],index):
                    record=records[name];data=memoryview(value.contiguous().view(torch.uint8).flatten().numpy())
                    if (name in seen or list(value.shape)!=record['shape'] or len(data)!=record['length']):raise ValueError(f'Invalid transformed record {name}')
                    seen.add(name)
                    if patch is None:
                        sha=hashlib.sha256(data).hexdigest()
                        if sha!=record['sha256']:raise ValueError(f'Round-trip mismatch rank {index} {name}: {sha} vs {record["sha256"]}')
                    else:patch.replace(name,data)
                if seen!=set(records):raise ValueError('Unhandled pack records: '+str(sorted(set(records)-seen)))
                if source.used!=set(file.keys()):raise ValueError('Unused or missing learned parameters: '+str(sorted(source.used^set(file.keys()))))
                if patch:
                    patch.map[80:112]=bytes.fromhex(plan_hash)
                    patch.finish(bytes.fromhex(tier));patch.close();patch=None
                    rank['file_sha256']=file_digest(dest)
                print(json.dumps({'rank':index,'learned_tensors':len(source.used),'records_rebuilt':len(seen)-7,'verified_shared_records':6,'rope_reused':True}),flush=True)
            finally:
                if patch:patch.close()
        if out:
            result.update(tier={'id':tier,'descriptor':tier_desc},plan={'hash':plan_hash,'descriptor':plan},
                          builder='tools/weights/import_dflash2_drafter.py',built_unix=int(time.time()))
            (stage/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
            if out.exists():raise FileExistsError(out)
            stage.rename(out)
    print(json.dumps({'passed':True,'roundtrip':args.check_roundtrip,'source_sha256':source_sha,'output':str(out) if out else None,'tier':tier if out else None}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('template','target-pack','checkpoint'):parser.add_argument('--'+name,required=True,type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path);group.add_argument('--check-roundtrip',action='store_true')
    parser.add_argument('--expected-sha256');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads < 1:parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    run(args)


if __name__=='__main__':main()
