"""Run a sharded DSpine proposal server with torchrun on two local gfx1201 GPUs.

The public socket uses live_service's committed-row protocol. A private local
socket sends the same operation and input rows to the other rank. GPU tensor
reductions run only during an actual request, so an idle follower can wait
indefinitely without a pending distributed collective.
"""
import argparse
from contextlib import contextmanager
import datetime
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import tempfile

import torch
import torch.distributed as dist

import live_service
import pack_io
import tp_model
import tp_graph


APPEND, PROPOSE, CLEAR, STATS = range(1,5)
HEADER = struct.Struct('<BQQII')
MEMORY = struct.Struct('<QQQ')
ACK = bytes([1])


def receive(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = sock.recv_into(view[offset:])
        if count == 0:
            raise EOFError('DSpine peer closed the private socket')
        offset += count
    return data


@contextmanager
def peer_socket(rank):
    """Exchange a private same-host control socket location exactly once."""
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix='dspine-') as directory:
            path = str(Path(directory) / 'worker')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(path)
                listener.listen(1)
                dist.broadcast_object_list([path],src=0)
                peer, _ = listener.accept()
                with peer:
                    yield peer
    else:
        addresses = [None]
        dist.broadcast_object_list(addresses,src=0)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.connect(addresses[0])
            yield peer


class LocalRank:
    def __init__(self, args, rank):
        self.device = torch.device('cuda',int(os.environ['LOCAL_RANK']))
        self.model = tp_model.load_tp_model(args.checkpoint,args.prebuilt,self.device.index,
                                             fused_projections=args.fused_projections)
        model = self.model
        self.ring = tp_graph.RingDraft(model,args.max_seqs) if args.graph else None
        target = pack_io.TargetPack(args.target_pack)
        self.vocab = target.vocab
        if (model.capture_layer_ids != live_service.FEATURE_LAYERS
                or model.h != live_service.HIDDEN or model.block != live_service.BLOCK
                or model.h != target.hidden
                or model.w['selected_prefix_bridge.codebook'].shape[0] != target.vocab):
            raise ValueError('Unsupported target/TP draft geometry')
        self.mask_id = int(model.config['dflash_config']['mask_token_id'])
        if rank == 0:
            self.inputs = live_service.ProposalHead(target,model,args.draft_pack,args.prebuilt,self.device,gpu_tree=args.gpu_tree)
        else:
            self.inputs = live_service.ProposalInputs(target,model,self.device)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def append(self, seq, position, rows, payload):
        host = torch.frombuffer(payload,dtype=torch.bfloat16).view(rows,live_service.ROW_BYTES//2)
        for start in range(0,rows,256):
            features = host[start:start+256].to(self.device)
            if self.ring is None:
                positions = torch.arange(position+start,position+start+len(features),device=self.device)
                self.model.append_context(seq,features,positions)
            else:
                self.ring.append(seq,position+start,features)

    def forward(self, seq, position, anchor):
        embeds, ids, positions = self.inputs.inputs(anchor,position)
        if self.ring is None:
            hidden, stash = self.model.forward_block(seq,embeds,ids,positions)
        else:
            hidden, stash = self.ring.forward(seq,position,embeds,ids)
        return hidden,stash,ids

    def clear(self, seq):
        if self.ring is None:
            self.model.clear(seq)
        else:
            self.ring.clear(seq)

    def memory(self):
        return (torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
                torch.cuda.max_memory_allocated(self.device))


class Leader:
    def __init__(self, local, peer):
        self.local = local
        self.peer = peer
        self.vocab = local.vocab
        self.failed = False

    @contextmanager
    def command(self, code, seq=0, position=0, value=0, width=0):
        """A partially dispatched collective cannot be retried on this session."""
        try:
            self.peer.sendall(HEADER.pack(code,seq,position,value,width))
            yield
            if receive(self.peer,1) != ACK:
                raise RuntimeError('Invalid DSpine peer acknowledgment')
        except BaseException as error:
            self.failed = True
            self.peer.close()
            if isinstance(error,(KeyboardInterrupt,SystemExit)):
                raise
            # Leave the listener. torchrun tears down the other rank if it
            # was already in a collective when a local computation failed.
            raise SystemExit(f'DSpine TP request failed: {error}') from error

    def append(self, seq, position, rows, payload):
        with self.command(APPEND,seq,position,rows):
            self.peer.sendall(payload)
            self.local.append(seq,position,rows,payload)

    def propose(self, seq, position, anchor, width, *, with_tree=False):
        with self.command(PROPOSE,seq,position,anchor,width):
            hidden,stash,ids = self.local.forward(seq,position,anchor)
            return self.local.inputs.tokens(hidden,stash,ids,width,with_tree=with_tree)

    def propose_tree(self, seq, position, anchor, width):
        return self.propose(seq,position,anchor,width,with_tree=True)

    def clear(self, seq):
        if self.failed:
            self.local.clear(seq)
            return
        with self.command(CLEAR,seq):
            self.local.clear(seq)

    def memory_stats(self):
        with self.command(STATS):
            follower = MEMORY.unpack(receive(self.peer,MEMORY.size))
        names = ('allocated_bytes','reserved_bytes','peak_allocated_bytes')
        return {'tp':2,'graph':self.local.ring is not None,'fused_projections':isinstance(self.local.model,tp_model.FusedProjectionModel),'gpu_tree':self.local.inputs.tree_selector is not None,'rank_memory':[dict(zip(names,stats)) for stats in (self.local.memory(),follower)]}

    def warmup(self, rows, seqs):
        tile = bytearray(min(256,rows)*live_service.ROW_BYTES)
        boundaries = [rows]
        if self.local.ring is not None:
            boundaries = [n for n in self.local.ring.buckets if n < rows] + [rows]
        for seq in range(seqs):
            position = 0
            for boundary in boundaries:
                while position < boundary:
                    count = min(boundary-position,256)
                    self.append(seq,position,count,memoryview(tile)[:count*live_service.ROW_BYTES])
                    position += count
                self.propose(seq,position,self.local.mask_id,live_service.BLOCK-1,
                             with_tree=self.local.inputs.tree_selector is not None)
        for seq in range(seqs):
            self.clear(seq)
        torch.cuda.synchronize(self.local.device)


def follow(local, peer):
    while True:
        try:
            packet = receive(peer,HEADER.size)
        except EOFError:
            return
        code,seq,position,value,width = HEADER.unpack(packet)
        if code == APPEND:
            if not 1 <= value <= live_service.MAX_FRAME_ROWS:
                raise ValueError('Invalid private append frame')
            local.append(seq,position,value,receive(peer,value*live_service.ROW_BYTES))
        elif code == PROPOSE:
            local.forward(seq,position,value)
        elif code == CLEAR:
            local.clear(seq)
        elif code == STATS:
            peer.sendall(MEMORY.pack(*local.memory()))
        else:
            raise ValueError(f'Invalid private TP opcode {code}')
        peer.sendall(ACK)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('socket','checkpoint','prebuilt','target-pack','draft-pack'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--gpu-tree',action='store_true',help='Select 15-node proposal trees on the leader GPU')
    p.add_argument('--fused-projections',action='store_true',help='Merge local Q/K/V and gate/up projection rows')
    p.add_argument('--graph',action='store_true',help='Keep incremental per-slot teacher rings and fixed HIP graphs')
    p.add_argument('--max-seqs',type=int,default=1)
    p.add_argument('--warmup-rows',type=int,default=8192)
    args = p.parse_args()
    if args.max_seqs < 1 or not 1 <= args.warmup_rows <= 8192:
        p.error('Invalid sequence capacity or warmup length')
    if sys.byteorder != 'little' or int(os.environ.get('WORLD_SIZE','0')) != 2:
        p.error('Use torchrun with two ranks on one little-endian host')
    if os.environ.get('NCCL_P2P_DISABLE','1') != '1':
        p.error('This host requires NCCL_P2P_DISABLE=1')
    os.environ['NCCL_P2P_DISABLE'] = '1'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(minutes=3))
    try:
        rank = dist.get_rank()
        with torch.no_grad(), peer_socket(rank) as peer:
            local = LocalRank(args,rank)
            if rank == 0:
                leader = Leader(local,peer)
                leader.warmup(args.warmup_rows,args.max_seqs)
                live_service.listen(args.socket,leader,args.max_seqs)
            else:
                follow(local,peer)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
