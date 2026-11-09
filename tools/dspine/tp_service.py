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
import traceback

import torch
import torch.distributed as dist

import live_service
import native_head
import pack_io
import tp_model
import tp_graph


APPEND, PROPOSE, CLEAR, STATS, STOP = range(1,6)
HEADER = struct.Struct('<BQQII')
MEMORY = struct.Struct('<QQQ')
ACK = bytes([1])


class StopRequested(SystemExit):
    """The launcher requested worker termination."""


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
        self.rank = rank
        self.tp_head = None
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
            self.inputs = live_service.ProposalHead(target,model,args.draft_pack,args.prebuilt,self.device,
                                                   gpu_tree=args.gpu_tree,sparse_shard=rank if args.tp_sparse_head else None)
            if args.tp_sparse_head:
                self.tp_head = self.inputs.sparse
        else:
            self.inputs = live_service.ProposalInputs(target,model,self.device)
            if args.tp_sparse_head:
                self.tp_head = native_head.NativeSparseHead(args.draft_pack,args.prebuilt,self.device.index,shard_rank=rank)
        if self.tp_head is not None and (self.tp_head.vocab != self.vocab or self.vocab > 2**31):
            raise ValueError('Candidate IDs must fit in the shared signed 32-bit vocabulary')
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

    def joint_pool(self, hidden):
        # The last row-parallel backbone reduction gives both ranks the same hidden
        # states. Each ranks only the same 80 coarse IDs its shard contributed before.
        # Preserve the full 160-ID exact rerank on the leader. Local INT2 top16
        # alone can discard candidates that obtain a high exact BF16-weight score.
        pools = self.tp_head.coarse_pools(hidden[1:],pool_per_shard=80)
        if len(pools) != 1:
            raise RuntimeError('Expected exactly one local INT2 vocabulary shard')
        own = pools[0]
        peer = (own.to(torch.int32).contiguous() if self.rank == 1
                else torch.empty(own.shape,device=self.device,dtype=torch.int32))
        # Same NCCL/RCCL group as backbone, outside its existing graph. The stream
        # waits on the broadcast before rank zero consumes the follower's IDs.
        dist.broadcast(peer,src=1)
        if self.rank == 0:
            return torch.cat((own,peer.to(own.dtype)),dim=-1)
        return None

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
        self.stopping = False

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
            candidates = None
            if self.local.tp_head is not None:
                pool = self.local.joint_pool(hidden)
                candidates = native_head.exact_candidate_topk(hidden[1:],self.local.inputs.head,pool)
            return self.local.inputs.tokens(hidden,stash,ids,width,with_tree=with_tree,candidates=candidates)

    def propose_tree(self, seq, position, anchor, width):
        return self.propose(seq,position,anchor,width,with_tree=True)

    def clear(self, seq):
        if self.failed or self.stopping:
            # serve_connection clears an open session in its finally block.
            # No new command may cross the socket once shutdown has started.
            return
        with self.command(CLEAR,seq):
            self.local.clear(seq)

    def memory_stats(self):
        with self.command(STATS):
            follower = MEMORY.unpack(receive(self.peer,MEMORY.size))
        names = ('allocated_bytes','reserved_bytes','peak_allocated_bytes')
        return {'tp':2,'graph':self.local.ring is not None,'fused_projections':isinstance(self.local.model,tp_model.FusedProjectionModel),'gpu_tree':self.local.inputs.tree_selector is not None,'tp_sparse_head':self.local.tp_head is not None,'rank_memory':[dict(zip(names,stats)) for stats in (self.local.memory(),follower)]}

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
        except StopRequested:
            # Idle between commands. The leader may receive SIGTERM a moment
            # later; require its STOP record before draining the GPU.
            return False
        code,seq,position,value,width = HEADER.unpack(packet)
        if code == STOP:
            if (seq,position,value,width) != (0,0,0,0):
                raise RuntimeError('Malformed DSpine stop record')
            return True
        if code == APPEND:
            if not 1 <= value <= live_service.MAX_FRAME_ROWS:
                raise ValueError('Invalid private append frame')
            local.append(seq,position,value,receive(peer,value*live_service.ROW_BYTES))
        elif code == PROPOSE:
            hidden,_,_ = local.forward(seq,position,value)
            if local.tp_head is not None:
                local.joint_pool(hidden)
        elif code == CLEAR:
            local.clear(seq)
        elif code == STATS:
            peer.sendall(MEMORY.pack(*local.memory()))
        else:
            raise ValueError(f'Invalid private TP opcode {code}')
        peer.sendall(ACK)


def close_idle_pair(local, peer, rank, received_stop):
    # A bounded socket rendezvous detects a failed or interrupted peer without
    # submitting a collective to a communicator that may have unmatched work.
    peer.settimeout(5)
    stop = HEADER.pack(STOP,0,0,0,0)
    if rank == 0:
        peer.sendall(stop)
        if receive(peer,1) != ACK:
            raise RuntimeError('Follower did not acknowledge quiescence')
    else:
        if not received_stop and receive(peer,HEADER.size) != stop:
            raise RuntimeError('Follower stopped during a pending request')
        peer.sendall(ACK)
    torch.cuda.synchronize(local.device)
    if local.ring is not None:
        local.ring.close()
    # The context manager closes the socket only after both graph resets.
    peer.sendall(ACK)
    if receive(peer,1) != ACK:
        raise RuntimeError('Peer did not release DSpine graphs')


def serve_pair(args, rank):
    with torch.no_grad(), peer_socket(rank) as peer:
        local = LocalRank(args,rank)
        received_stop = False
        if rank == 0:
            leader = Leader(local,peer)
            leader.warmup(args.warmup_rows,args.max_seqs)
            def stop_listening(signum, frame):
                leader.stopping = True
                raise StopRequested(0)
            signal.signal(signal.SIGTERM,stop_listening)
            try:
                live_service.listen(args.socket,leader,args.max_seqs)
            except StopRequested:
                if leader.failed:
                    raise
        else:
            received_stop = follow(local,peer)
        # A second SIGTERM while an acknowledged idle stop is already
        # draining must not interrupt graph release.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        close_idle_pair(local,peer,rank,received_stop)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('socket','checkpoint','prebuilt','target-pack','draft-pack'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--gpu-tree',action='store_true',help='Select 15-node proposal trees on the leader GPU')
    p.add_argument('--tp-sparse-head',action='store_true',help='Scan each INT2 vocabulary shard on its owning GPU before the original exact rerank')
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
        raise StopRequested(0)
    signal.signal(signal.SIGTERM,stop)
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(minutes=3))
    rank = dist.get_rank()
    try:
        serve_pair(args,rank)
        dist.destroy_process_group()
        print(f'DSpine rank {rank} closed its process group',flush=True)
    except BaseException:
        # The peer may already be missing from a collective. Exit this
        # torchrun worker without waiting on NCCL or Python GPU finalizers.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)


if __name__=='__main__':
    main()
