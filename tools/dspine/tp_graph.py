"""Committed teacher ring and fixed-size draft graphs for an individual TP rank.

Ring addresses stay fixed for each sequence slot. Absolute position -2**40
masks every unused row; after the window fills, physical order can differ from
chronological order. The target's acceptance gate measures any reduction-order
differences. Each pair of TP ranks must execute the same slot/bucket sequence.
"""
import torch


INVALID_POSITION = -(1 << 40)


class TeacherSlot:
    def __init__(self, model):
        self.model = model
        self.next_position = 0
        self.graphs = {}
        self.positions = torch.full((model.window,), INVALID_POSITION,
                                    dtype=torch.int64, device=model.device)
        shape = (model.window,model.kheads,model.dim)
        # Masked values must be finite because attention multiplies them by zero.
        self.keys = [torch.zeros(shape,dtype=torch.bfloat16,device=model.device)
                     for _ in range(model.layers)]
        self.values = [torch.zeros_like(key) for key in self.keys]

    def reset(self):
        self.next_position = 0
        self.positions.fill_(INVALID_POSITION)

    def append(self, position, features):
        model = self.model
        if position != self.next_position:
            raise ValueError('Teacher ring append has a gap or overlaps committed rows')
        if (features.ndim != 2 or features.shape[1] != model.target_features * model.h
                or features.dtype != torch.bfloat16 or features.device != model.device):
            raise ValueError('Invalid teacher feature tensor')
        rows = features.shape[0]
        if not 1 <= rows <= model.window:
            raise ValueError('Append one to window-size committed rows at a time')
        positions = torch.arange(position,position+rows,device=model.device)
        keys,values = model.project_context(features,positions)
        first = position % model.window
        first_rows = min(rows,model.window-first)
        spans = [(slice(first,first+first_rows),slice(0,first_rows))]
        if first_rows < rows:
            spans.append((slice(0,rows-first_rows),slice(first_rows,rows)))
        for dest, source in spans:
            for layer in range(model.layers):
                self.keys[layer][dest].copy_(keys[layer][source])
                self.values[layer][dest].copy_(values[layer][source])
            self.positions[dest].copy_(positions[source])
        self.next_position += rows

    def entry(self, capacity):
        valid = min(self.next_position,self.model.window)
        if not max(1,valid) <= capacity <= self.model.window:
            raise ValueError('Bucket does not cover the live teacher rows')
        return {'positions':self.positions[:capacity],
                'keys':[key[:capacity] for key in self.keys],
                'values':[value[:capacity] for value in self.values]}


class BucketGraph:
    def __init__(self, slot, capacity, embeds, ids, position):
        model = slot.model
        self.embeds = embeds.clone()
        self.ids = ids.clone()
        self.positions = torch.arange(position,position+model.block,device=model.device)
        self.linear = model.linear
        self.bridge = model.bridge
        entry = slot.entry(capacity)
        self.capture_stream = torch.cuda.Stream(device=model.device)
        stream = self.capture_stream
        stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):
                model.forward_block_tensors(self.embeds,self.ids,self.positions,entry)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream),torch.no_grad():
            self.result = model.forward_block_tensors(self.embeds,self.ids,self.positions,entry)
        torch.cuda.current_stream(model.device).wait_stream(stream)

    def replay(self, model, embeds, ids, position):
        if model.linear is not self.linear or model.bridge is not self.bridge:
            raise ValueError('Captured draft projection or bridge changed')
        self.embeds.copy_(embeds)
        self.ids.copy_(ids)
        torch.arange(position,position+model.block,out=self.positions)
        self.graph.replay()
        return self.result


class RingDraft:
    def __init__(self, model, max_seqs):
        self.model = model
        self.max_seqs = max_seqs
        self.slots = {}
        self.free = []
        self.buckets = sorted({min(n,model.window) for n in (128,512,2048,model.window)})

    def append(self, seq, position, features):
        slot = self.slots.get(seq)
        if slot is None:
            if position != 0 or len(self.slots) >= self.max_seqs:
                raise ValueError('Invalid first teacher position or sequence capacity exceeded')
            slot = self.free.pop() if self.free else TeacherSlot(self.model)
            self.slots[seq] = slot
        slot.append(position,features)

    def forward(self, seq, position, embeds, ids):
        slot = self.slots.get(seq)
        if slot is None or slot.next_position != position:
            raise ValueError('Draft query must follow committed teacher rows')
        model = self.model
        if (embeds.shape != (model.block,model.h) or ids.shape != (model.block,)
                or embeds.dtype != torch.bfloat16 or ids.dtype != torch.int64
                or any(x.device != model.device for x in (embeds,ids))):
            raise ValueError('Invalid draft query geometry')
        # Replay generates consecutive positions from the checked committed cursor.
        live = min(position,model.window)
        capacity = next(n for n in self.buckets if n >= live)
        if capacity not in slot.graphs:
            slot.graphs[capacity] = BucketGraph(slot,capacity,embeds,ids,position)
        return slot.graphs[capacity].replay(model,embeds,ids,position)

    def clear(self, seq):
        slot = self.slots.pop(seq,None)
        if slot is not None:
            slot.reset()
            self.free.append(slot)

    def close(self):
        """Release captured graph executables before their TP communicator."""
        for slot in [*self.slots.values(), *self.free]:
            for bucket in slot.graphs.values():
                bucket.graph.reset()
            slot.graphs.clear()
        self.slots.clear()
        self.free.clear()
