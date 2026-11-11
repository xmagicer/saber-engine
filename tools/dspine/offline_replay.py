"""Replay DSpine against actual committed target features, without future leakage.

This is a quality diagnostic, not a serving backend or throughput benchmark.
Requires ROCm PyTorch, Triton, safetensors, target/draft packs and a valid capture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
import capture_reader
import native_head
import native_linear
import pack_io
import plain_model
import reference
import target_layers
import tree_proposal
import triton_bridge
import triton_tail


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','checkpoint','target-pack','draft-pack','prebuilt','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--projection',choices=['bf16','channel','group128'],default='channel')
    p.add_argument('--row-band',type=int,choices=[16,32],default=16)
    p.add_argument('--fused-bridge',action='store_true')
    p.add_argument('--block-size',type=int,choices=[8,16],default=16)
    p.add_argument('--candidate-pool',type=int,default=80,help='Per-shard int2 pool; zero selects the dense BF16 head')
    p.add_argument('--tree-nodes',type=int,choices=[0,9,15],default=0)
    p.add_argument('--max-depth',type=int,default=None)
    p.add_argument('--verify-tail-steps',type=int,default=2)
    return p.parse_args()


@torch.no_grad()
def replay(args):
    if args.output.exists():raise ValueError('Output already exists; use a new evidence file')
    if args.candidate_pool<0 or args.verify_tail_steps<0:raise ValueError('Negative pool/check count')
    if args.max_depth is None:args.max_depth=args.block_size-1
    if not 1<=args.max_depth<args.block_size:raise ValueError('Depth must fit the proposer block')
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    target=pack_io.TargetPack(args.target_pack)
    model=plain_model.TorchDspine.from_checkpoint(args.checkpoint,'cuda')
    if args.projection=='channel':model.linear=native_linear.NativeChannelLinear(model.w,args.prebuilt,row_band=args.row_band)
    elif args.projection=='group128':model.linear=native_linear.NativeLinear(model.w,args.prebuilt)
    if args.fused_bridge:model.bridge=triton_bridge.FusedBridge()
    head=target.load_head('cuda')
    sparse=native_head.NativeSparseHead(args.draft_pack,args.prebuilt) if args.candidate_pool else None
    w=model.w;r=int(model.config['dflash_config']['selected_prefix_code_bridge']['rank'])
    tg,tsv=triton_tail.build_serial_tables(head,w['norm.weight'],w['selected_prefix_bridge.feature_write_weights'][-1],model.packed[r:],w['selected_prefix_bridge.codebook'],chunk=4096)
    proofs=json.loads((args.capture/'validation.json').read_text())['proofs']
    result={'passed':False,'diagnostic_only':True,'serving_integration_complete':False,'throughput_benchmark':False,
            'projection':args.projection,'row_band':args.row_band,'fused_bridge':args.fused_bridge,
            'tree_nodes':args.tree_nodes,'block_size':args.block_size,'max_depth':args.max_depth,
            'candidate_pool_per_shard':args.candidate_pool,'target_future_used_by_proposer':False,
            'capture_layers':model.capture_layer_ids,'checkpoint_config_sha256':hashlib.sha256((args.checkpoint/'config.json').read_bytes()).hexdigest(),
            'torch':torch.__version__,'hip':torch.version.hip,'runs':[]}
    for proof in proofs:
        kind=proof['kind']
        if Path(kind).name!=kind or not kind:raise ValueError('Invalid capture case name')
        response=json.loads((args.capture/(kind+'-response.json')).read_text())
        data=capture_reader.load_capture(args.capture/'shared/capture',proof['stem'],response)
        target_layers.validate_dspine_capture(data['metadata'],model.config['dflash_config']['target_layer_ids'])
        tokens=data['tokens'];features=data['features'];p=data['prompt_tokens'];previous=0;steps=[];t=args.block_size
        while p+t<=tokens.numel() and p<features.shape[0]:
            # Append only features whose target tokens were already committed.
            model.append_context(kind,features[previous:p].to('cuda'),torch.arange(previous,p,device='cuda',dtype=torch.int64))
            ids=torch.full((t,),int(model.config['dflash_config']['mask_token_id']),dtype=torch.int64)
            ids[0]=tokens[p]
            embeds=target.embedding_rows(ids).to('cuda');gpu_ids=ids.to('cuda')
            positions=torch.arange(p,p+t,device='cuda',dtype=torch.int64)
            hidden,stash=model.forward_block(kind,embeds,gpu_ids,positions)
            if sparse is None:cand=torch.topk(torch.mm(hidden[1:],head.T,out_dtype=torch.float32),16,dim=-1).indices.sort(dim=-1).values.contiguous()
            else:cand=sparse.candidates(hidden[1:],head,pool_per_shard=args.candidate_pool).contiguous()
            table=triton_tail.serial_table(stash['raw'],w['norm.weight'],cand,gpu_ids,head,tg,tsv,stash['packed'],stash['rms'],w['selected_prefix_bridge.gate.bias'],t,with_log_probabilities=bool(args.tree_nodes))
            idx,logp=table if args.tree_nodes else (table,None)
            if len(steps)<args.verify_tail_steps:
                expected=reference.serial_table(stash['raw'],w['norm.weight'],cand,gpu_ids,head,tg,tsv,stash['packed'],stash['rms'],w['selected_prefix_bridge.gate.bias'],t,with_log_probabilities=bool(args.tree_nodes))
                if not torch.equal(expected[0],idx):raise AssertionError('GPU conditional greedy differs from reference')
                if logp is not None and not torch.allclose(expected[2],logp,atol=2e-4,rtol=2e-5):raise AssertionError('GPU transition probabilities differ from reference')
            if args.tree_nodes:
                nodes=tree_proposal.build_best_first_tree(cand.cpu().tolist(),logp.cpu().tolist(),max_nodes=args.tree_nodes,max_depth=args.max_depth)
                tree_proposal.ancestor_masks(nodes)
                proposal=[{'token':n.token,'parent':n.parent,'depth':n.depth,'log_probability':n.log_probability} for n in nodes]
            else:
                proposed=torch.empty(t-1,device='cuda',dtype=torch.int64)
                triton_tail.serial_chain(idx,cand,proposed,t)
                proposal=proposed.cpu().tolist()
            # Canonical future is read only AFTER the proposal/tree is complete.
            truth=tokens[p+1:p+t].tolist()
            if args.tree_nodes:accepted=tree_proposal.accepted_tree_prefix(nodes,truth)
            else:
                accepted=0
                for a,b in zip(proposal,truth):
                    if a!=b:break
                    accepted+=1
            steps.append({'position':p,'accepted':accepted,'committed':accepted+1,'proposal':proposal})
            previous=p;p+=accepted+1
        model.clear(kind)
        if not steps:raise ValueError('Capture has no complete proposal block')
        summary={'kind':kind,'rounds':len(steps),'mean_committed':statistics.mean(x['committed'] for x in steps),'covered_output_tokens':p-data['prompt_tokens']}
        print(json.dumps(summary),flush=True)
        result['runs'].append({'summary':summary,'steps':steps})
    result['passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':replay(arguments())
