"""Independent GPU gate for DSpine's conditional transitions and greedy walk."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
import reference as Ref
import triton_tail as Tail


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    torch.cuda.set_device(0);torch.manual_seed(20261109);torch.backends.cuda.matmul.allow_tf32=False
    with safe_open(str(args.checkpoint/'model.safetensors'),framework='pt',device='cpu') as f:
        book=f.get_tensor('selected_prefix_bridge.codebook')[:1024].to('cuda')
        write=f.get_tensor('selected_prefix_bridge.feature_write_weights')[-1].to('cuda')
        norm=f.get_tensor('norm.weight').to('cuda')
        gate=f.get_tensor('selected_prefix_bridge.predecessor_gate.weight').to('cuda')
        prev=f.get_tensor('selected_prefix_bridge.prev.weight').to('cuda')
        bias=f.get_tensor('selected_prefix_bridge.gate.bias').to('cuda')
    v,c=book.shape;h,r=write.shape
    head=(torch.randn((v,h),device='cuda',dtype=torch.bfloat16)*.02).contiguous()
    pv=torch.cat((gate,prev));tg,tsv=Tail.build_serial_tables(head,norm,write,pv,book,chunk=256)
    expected_tg,expected_tsv=Ref.build_serial_tables(head,norm,write,pv,book,chunk=256)
    assert torch.equal(tg,expected_tg) and torch.equal(tsv,expected_tsv),'Static tables differ'
    checks=[];stream=torch.cuda.Stream()
    for t,m,k in [(8,16,2),(8,16,8),(16,32,16),(16,32,32)]:
        n=m//t*(t-1)
        raw=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)
        packed=torch.randn((m,3*r),device='cuda',dtype=torch.bfloat16)
        rms=torch.rand(m,device='cuda')+.2
        ids=torch.arange(m,device='cuda',dtype=torch.int64)
        cand=torch.topk(torch.rand((n,v),device='cuda'),k,dim=-1).indices.sort(dim=-1).values.contiguous()
        confidence=torch.empty((n,k),device='cuda',dtype=torch.float32)
        out=torch.empty(n,device='cuda',dtype=torch.int64)
        def run():
            idx,logp=Tail.serial_table(raw,norm,cand,ids,head,tg,tsv,packed,rms,bias,t,confidence,with_log_probabilities=True)
            Tail.serial_chain(idx,cand,out,t)
            return idx,logp
        for _ in range(3):run()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):actual_idx,actual_logp=run()
        torch.cuda.current_stream().wait_stream(stream)
        worst=0.0
        for repeat in range(5):
            raw.copy_(torch.randn_like(raw));packed.copy_(torch.randn_like(packed));rms.copy_(torch.rand_like(rms)+.2)
            expected_idx,expected_conf,expected_logp=Ref.serial_table(raw,norm,cand,ids,head,tg,tsv,packed,rms,bias,t,with_log_probabilities=True)
            expected_tokens,_=Ref.serial_chain(expected_idx,cand,t)
            g.replay();torch.cuda.synchronize()
            assert torch.equal(expected_idx,actual_idx),'Conditional greedy indices differ'
            assert torch.equal(expected_tokens,out),'Greedy walk differs'
            error=float((expected_logp-actual_logp).abs().max());worst=max(worst,error)
            assert torch.allclose(expected_logp,actual_logp,atol=2e-4,rtol=2e-5),error
            assert torch.allclose(expected_conf,confidence,atol=2e-6,rtol=2e-5),'Confidence differs'
            assert torch.allclose(torch.logsumexp(actual_logp,-1),torch.zeros((n,k),device='cuda'),atol=2e-5),'Transitions not normalized'
        checks.append({'block_size':t,'rows':m,'candidates':k,'changing_graph_inputs':5,'max_logp_error':worst,'indices_and_walk_exact':True})
    print(json.dumps({'passed':True,'checks':checks,'target_future_used':False,'model_acceptance_tested':False},indent=2))


if __name__=='__main__':main()
