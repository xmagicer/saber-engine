"""Test two-rank proposal control and exact candidate ranking using CPU Gloo.

Run with `python -m torch.distributed.run --standalone --nproc-per-node=2
tools/dspine/cpu_tp_head_gate.py`.
"""
from types import SimpleNamespace

import torch
import torch.distributed as dist
import tp_service


def main():
    dist.init_process_group("gloo")
    rank=dist.get_rank()
    torch.set_num_threads(1)
    # The exact head favors later/shard-boundary IDs, with identical-score ties.
    ids=torch.arange(160)
    exact=(ids//5 % 7).to(torch.bfloat16)
    head=exact[:,None].contiguous()
    hidden=torch.ones(16,1,dtype=torch.bfloat16)
    correct=sorted(sorted(range(160),key=lambda i:(-float(exact[i]),i))[:16])
    truncated=sorted(sorted([*range(16),*range(80,96)],key=lambda i:(-float(exact[i]),i))[:16])
    assert correct!=truncated
    own=torch.arange(80*rank,80*rank+80,dtype=torch.int64).expand(15,80).contiguous()
    local=tp_service.LocalRank.__new__(tp_service.LocalRank)
    local.vocab=160;local.device=torch.device("cpu");local.rank=rank
    local.tp_head=SimpleNamespace(coarse_pools=lambda h,pool_per_shard: [own])
    local.inputs=SimpleNamespace(head=head,tokens=lambda h,s,i,w,*,with_tree,candidates: candidates.tolist())
    local.forward=lambda *args:(hidden,{},None)
    with tp_service.peer_socket(rank) as peer:
        if rank==1:
            assert tp_service.follow(local,peer)
        else:
            leader=tp_service.Leader(local,peer)
            for case in range(3):
                result=leader.propose(0,100+case,2,15,with_tree=True)
                assert result==[correct]*15
            peer.sendall(tp_service.HEADER.pack(tp_service.STOP,0,0,0,0))
    dist.barrier();dist.destroy_process_group()
    if rank==0:print("PASS two-rank CPU control/ACK, 4800-byte follower broadcast, global dense rerank and lower-ID ties",flush=True)


if __name__=="__main__":
    main()
