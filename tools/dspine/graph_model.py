"""Fixed-shape HIP graphs for experimental DSpine proposal computation.

Teacher caches stay in TorchDspine. Graph staging takes a copy of committed
teacher rows only, with explicitly invalid positions in padded slots. No draft
query K/V is written into a teacher cache. Returned tensors are borrowed until
the next call to the same graph; weights, linear backend and bridge callback must stay fixed.
"""
import torch


class DraftGraph:
    def __init__(self,model,block_size,teacher_capacity):
        if not 2<=block_size<=model.block or not 1<=teacher_capacity<=model.window:
            raise ValueError('Unsupported graph geometry')
        self.model=model;self.block=block_size;self.capacity=teacher_capacity
        self.linear=model.linear
        self.bridge=model.bridge
        device=model.device;h=model.h
        self.embeds=torch.zeros((block_size,h),device=device,dtype=torch.bfloat16)
        self.ids=torch.zeros(block_size,device=device,dtype=torch.int64)
        self.positions=torch.arange(block_size,device=device,dtype=torch.int64)
        self.entry={'positions':torch.full((teacher_capacity,),-(1<<40),device=device,dtype=torch.int64),
                    'keys':[torch.zeros((teacher_capacity,model.kheads,model.dim),device=device,dtype=torch.bfloat16) for _ in range(model.layers)],
                    'values':[torch.zeros((teacher_capacity,model.kheads,model.dim),device=device,dtype=torch.bfloat16) for _ in range(model.layers)]}
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):
                model.forward_block_tensors(self.embeds,self.ids,self.positions,self.entry)
        stream.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream),torch.no_grad():
            self.hidden,self.stash=model.forward_block_tensors(self.embeds,self.ids,self.positions,self.entry)
        torch.cuda.current_stream(device).wait_stream(stream)

    @torch.no_grad()
    def __call__(self,seq,embeds,ids,positions):
        model=self.model;entry=model.cache.get(seq)
        start=0 if entry is None else entry['next']
        if (model.linear is not self.linear or model.bridge is not self.bridge or embeds.shape!=(self.block,model.h)
                or embeds.dtype!=torch.bfloat16 or ids.shape!=(self.block,)
                or positions.shape!=(self.block,) or ids.dtype!=torch.int64 or positions.dtype!=torch.int64
                or any(x.device!=model.device for x in (embeds,ids,positions))):
            raise ValueError('Graph inputs/backend differ from captured geometry')
        if not torch.equal(positions,torch.arange(start,start+self.block,device=model.device,dtype=torch.int64)):
            raise ValueError('Query block must follow committed teacher context')
        live=0 if entry is None else entry['positions'].numel()
        if live>self.capacity:
            raise ValueError('Teacher context exceeds graph capacity')
        self.entry['positions'].fill_(-(1<<40))
        if live:
            self.entry['positions'][:live].copy_(entry['positions'])
            for layer in range(model.layers):
                self.entry['keys'][layer][:live].copy_(entry['keys'][layer])
                self.entry['values'][layer][:live].copy_(entry['values'][layer])
        self.embeds.copy_(embeds);self.ids.copy_(ids);self.positions.copy_(positions)
        self.graph.replay()
        return self.hidden,self.stash
