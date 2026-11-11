"""Saber int2 candidate preselection followed by exact BF16-weight reranking.

This experimental drafter head never changes target verification. Candidate
preselection is approximate; compare acceptance and recall before enabling it.
"""
from pathlib import Path
import json
import torch
import hip_module
import pack_io


def exact_candidate_topk(hidden,head,pool,k=16):
    """FP32 scores over supplied candidate ids; equal scores prefer lower token ids."""
    pool=pool.sort(dim=-1).values
    selected=head[pool].float()
    scores=(selected*hidden.float()[:,None,:]).sum(-1)
    order=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:k]
    ids=torch.gather(pool,1,order)
    return ids.sort(dim=-1).values


class NativeSparseHead:
    """Use the shipped gfx1201 int2 head code object on a PyTorch HIP stream."""
    def __init__(self,drafter_directory,prebuilt,device=0):
        self.module=hip_module.HipModule(prebuilt,'dflash_head',device)
        self.device=device;self.shards=[];self.vocab=0;self.hidden=5120
        root=Path(drafter_directory);manifest=json.loads((root/'manifest.json').read_text())
        for shard in sorted(manifest['ranks'],key=lambda s:s['rank']):
            path=root/shard['file']
            if shard['file']!=f"rank{shard['rank']}.pack" or path.stat().st_size!=shard['file_bytes']:raise ValueError('Unexpected drafter shard')
            mapped=torch.from_file(str(path),shared=False,size=shard['file_bytes'],dtype=torch.uint8);records={x['name']:x for x in shard['records']}
            wq=pack_io.TargetPack.record(mapped,records['head.int2.wq'],torch.uint8,9).to(device=f'cuda:{device}')
            s=pack_io.TargetPack.record(mapped,records['head.int2.s'],torch.bfloat16,10).to(device=f'cuda:{device}')
            zs=pack_io.TargetPack.record(mapped,records['head.int2.zs'],torch.bfloat16,11).to(device=f'cuda:{device}')
            n=wq.shape[0]
            if wq.shape[1]!=self.hidden//4 or s.shape!=(n,self.hidden//128) or zs.shape!=s.shape or n%64:raise ValueError('Int2 head shape mismatch')
            self.shards.append({'base':self.vocab,'n':n,'wq':wq,'s':s,'zs':zs});self.vocab+=n

    @torch.no_grad()
    def candidates(self,hidden,head,k=16,pool_per_shard=256,coarse_mode="block"):
        if coarse_mode not in ["block","dense"]:raise ValueError("Unknown coarse candidate mode")
        if hidden.ndim!=2 or hidden.shape[1]!=self.hidden or hidden.dtype!=torch.bfloat16 or not hidden.is_cuda:raise ValueError('Expected CUDA BF16 hidden rows')
        live=hidden.shape[0];rows=max(16,1<<(live-1).bit_length())
        x=torch.empty((rows,self.hidden),device=hidden.device,dtype=torch.bfloat16)
        self.module.launch('dflash_head_row_pad_bf16',((rows*self.hidden+2047)//2048,1,1),(256,1,1),{'x':hidden.contiguous(),'out':x,'m':live,'rows':rows,'k':self.hidden})
        xs=torch.empty((rows,self.hidden//128),device=hidden.device,dtype=torch.float32)
        self.module.launch('dflash_head_group_sum_g128',((rows*(self.hidden//128)+7)//8,1,1),(256,1,1),{'x':x,'xs':xs,'rows':rows,'k':self.hidden,'stride_xs':self.hidden//128})
        pools=[]
        for shard in self.shards:
            n=shard['n'];nblk=n//64
            coarse=torch.empty((rows,n),device=hidden.device,dtype=torch.bfloat16)
            maxima=torch.empty((rows,nblk*8),device=hidden.device,dtype=torch.float32)
            indices=torch.empty((rows,nblk*8),device=hidden.device,dtype=torch.int32)
            name='dflash_draft_head_int2_m16n64g128_topk' if rows==16 else 'dflash_draft_head_int2_mtn64g128_topk'
            if coarse_mode=='dense':name=name.removesuffix('_topk')
            self.module.launch(name,(nblk,rows//16,1),(64,1,1),{'x':x,'xs':xs,'wq':shard['wq'],'s':shard['s'],'zs':shard['zs'],'y':coarse,'bmax':maxima,'bidx':indices,'m':rows,'n':n,'k':self.hidden,'stride_wq':self.hidden//4,'stride_s':self.hidden//128,'stride_xs':self.hidden//128,'nblk':nblk})
            if coarse_mode=='dense':
                pools.append(torch.topk(coarse[:live].float(),pool_per_shard,dim=-1).indices+shard['base'])
            else:
                positions=torch.topk(maxima[:live],pool_per_shard,dim=-1).indices
                pools.append(torch.gather(indices[:live].long(),1,positions)+shard['base'])
        return exact_candidate_topk(hidden,head,torch.cat(pools,dim=-1),k)
