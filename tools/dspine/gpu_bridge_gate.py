"""Compare fused bridge arithmetic with the independent PyTorch reference.

Requires ROCm PyTorch, Triton and the downloaded BF16 DSpine checkpoint.
This checks primitives, graph replay and request-block isolation, not model quality.
"""
import argparse
import json
from pathlib import Path
import torch
import triton
from safetensors import safe_open
import reference as Ref
import triton_bridge as Fused


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    torch.cuda.set_device(0);torch.manual_seed(20261109)
    torch.backends.cuda.matmul.allow_tf32=False
    names={'read':'selected_prefix_bridge.early_read_weights','write':'selected_prefix_bridge.feature_write_weights',
           'gate':'selected_prefix_bridge.gate.weight','pred':'selected_prefix_bridge.predecessor_gate.weight',
           'prev':'selected_prefix_bridge.prev.weight','bias':'selected_prefix_bridge.gate.bias',
           'book':'selected_prefix_bridge.codebook','norm':'layers.1.input_layernorm.weight'}
    with safe_open(str(args.checkpoint/'model.safetensors'),framework='pt',device='cpu') as f:
        w={k:f.get_tensor(v).to('cuda') for k,v in names.items()}
    read=w['read'][0];write=w['write'][0];packed_w=Ref.pack_bridge_weights(w['gate'],w['pred'],w['prev'])
    bridge=Fused.FusedBridge();results=[];stream=torch.cuda.Stream()
    for t,m in [(8,8),(8,16),(16,16),(16,32)]:
        x=torch.randn((m,read.shape[1]),device='cuda',dtype=torch.bfloat16)
        residual=torch.randn_like(x)*.1
        ids=torch.arange(m,device='cuda',dtype=torch.int64)
        def call():return bridge(x,residual,ids,read,w['book'],packed_w,w['bias'],write,w['norm'],t)
        for _ in range(3):call()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):actual=call()
        torch.cuda.current_stream().wait_stream(stream)
        worst=0.0
        for iteration in range(10):
            x.copy_(torch.randn_like(x));residual.copy_(torch.randn_like(residual)*.1)
            expected=Ref.bridge_layer(x,residual,ids,read,w['book'],packed_w,w['bias'],write,w['norm'],t)
            eager=call();g.replay();torch.cuda.synchronize()
            for label,a,b in [('normed',expected[0],actual[0]),('residual',expected[1],actual[1])]:
                error=float((a.float()-b.float()).abs().max());worst=max(worst,error)
                assert torch.allclose(a.float(),b.float(),atol=.03125,rtol=.03),(t,m,label,error)
            assert torch.equal(expected[2]['raw'],actual[2]['raw'])
            for a,b in zip(eager[:2],actual[:2]):assert torch.equal(a.view(torch.int16),b.view(torch.int16)),'Changing-input graph mismatch'
            if m>t:
                before=[a[t:].clone() for a in actual[:2]]
                x[:t].add_(2);residual[:t].mul_(3);g.replay();torch.cuda.synchronize()
                for a,b in zip(before,actual[:2]):assert torch.equal(a.view(torch.int16),b[t:].view(torch.int16)),'Cross-request bridge write'
        results.append({'block_size':t,'rows':m,'iterations':10,'max_abs_vs_reference':worst,'graph_bitexact':True,'request_isolation_checked':m>t})
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(0),'cases':results,'model_acceptance_tested':False},indent=2))


if __name__=='__main__':main()
