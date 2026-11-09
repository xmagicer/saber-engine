"""Tensor-parallel DSpine draft backbone for a pair of gfx1201 GPUs.

Each rank runs its own attention heads and MLP channels. The output and down
projections reduce their partial BF16 outputs through the caller's process
group. Bridge projections and the teacher feature projection are replicated.
The caller supplies identical inputs on all ranks and owns the proposal head.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

import native_linear
import plain_model


COLUMN_PROJECTIONS = (
    'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
    'self_attn.v_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight',
)
ROW_PROJECTIONS = ('self_attn.o_proj.weight', 'mlp.down_proj.weight')


def projection_axis(name):
    """Split an output dimension at axis 0 or an input dimension at axis 1."""
    if name.startswith('layers.') and name.endswith('_proj.weight'):
        suffix = name.split('.', 2)[2]
        if suffix in COLUMN_PROJECTIONS:
            return 0
        if suffix in ROW_PROJECTIONS:
            return 1
        raise ValueError(f'Unknown projection {name}')
    return None


def shard_bounds(size, rank, world):
    if size % world:
        raise ValueError(f'Cannot evenly shard dimension {size} over {world} ranks')
    count = size // world
    return rank * count, (rank + 1) * count


class ShardedLinear(native_linear.NativeChannelLinear):
    """Use packed weight bytes themselves as TorchDspine's projection handles."""
    def __init__(self, prebuilt, device, group, world):
        super().__init__({}, prebuilt, device, row_band=32)
        self.group = group
        self.world = world
        self.reduce_weights = set()

    def register(self, packed, scales):
        self.weights[packed.data_ptr()] = (packed.shape,packed,scales)
        return packed

    @torch.no_grad()
    def load(self, source, axis, rank, destination=None):
        n, k = source.get_shape()
        first_row, last_row = shard_bounds(n, rank, self.world) if axis == 0 else (0,n)
        first_col, last_col = shard_bounds(k, rank, self.world) if axis == 1 else (0,k)
        local_n, local_k = last_row - first_row, last_col - first_col
        if local_n % 16 or local_k % 16:
            raise ValueError('Sharded W8A16 GEMM needs N and K multiples of 16')
        device = torch.device('cuda', self.device)
        if destination is None:
            packed = torch.empty((local_n,local_k),dtype=torch.uint8,device=device)
            scales = torch.empty(local_n,dtype=torch.float32,device=device)
        else:
            packed,scales = destination
            if packed.shape != (local_n,local_k) or scales.shape != (local_n,):
                raise ValueError('Merged projection segment has incorrect dimensions')
        # Read only one local stripe at a time. FC alone has 131M elements.
        for start in range(0, local_n, 256):
            end = min(start + 256, local_n)
            stripe = source[first_row+start:first_row+end, first_col:last_col].to(device)
            if stripe.dtype != torch.bfloat16:
                raise ValueError('Checkpoint projections must be BF16')
            values = stripe.float()
            maximum = values.abs().amax(dim=-1)
            if axis == 1 and self.world > 1:
                # Rank-local scales would change the quantized model itself.
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=self.group)
            scale = (maximum / 448).clamp_min(1 / (448 * 512))
            encoded = (values / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            packed[start:end].copy_(encoded.view(torch.uint8))
            scales[start:end].copy_(scale)
        if destination is None:
            self.register(packed,scales)
        if axis == 1 and self.world > 1:
            self.reduce_weights.add(packed.data_ptr())
        return packed

    def load_columns(self, file, names, rank):
        """Stream local output rows into consecutive ranges of a single allocation."""
        sources = [file.get_slice(name) for name in names]
        shapes = [source.get_shape() for source in sources]
        k = shapes[0][1]
        if any(len(shape) != 2 or shape[0] % self.world or shape[1] != k for shape in shapes):
            raise ValueError('Merged projections need compatible matrix shapes')
        local_rows = [shape[0] // self.world for shape in shapes]
        device = torch.device('cuda',self.device)
        packed = torch.empty((sum(local_rows),k),dtype=torch.uint8,device=device)
        scales = torch.empty(sum(local_rows),dtype=torch.float32,device=device)
        offset = 0
        for source,rows in zip(sources,local_rows):
            end = offset + rows
            self.load(source,0,rank,destination=(packed[offset:end],scales[offset:end]))
            offset = end
        self.register(packed,scales)
        return packed,scales,local_rows

    def __call__(self, inputs, weight):
        projected = super().__call__(inputs, weight)
        if weight.data_ptr() in self.reduce_weights:
            dist.all_reduce(projected, group=self.group)
        return projected


class FusedProjectionModel(plain_model.TorchDspine):
    """Column slices retain their batch-row stride for PyTorch's following ops."""
    def project_kv(self, hidden, attn):
        values = self.linear(hidden,self.w[attn+'kv.weight'])
        return values.split(self.kheads*self.dim,dim=-1)

    def project_qkv(self, hidden, attn):
        values = self.linear(hidden,self.w[attn+'qkv.weight'])
        q = self.qheads*self.dim
        kv = self.kheads*self.dim
        return values.split((q,kv,kv),dim=-1)

    def project_gate_up(self, hidden, prefix):
        return self.linear(hidden,self.w[prefix+'mlp.gate_up.weight']).chunk(2,dim=-1)


def load_fused_columns(file, weights, local, config, rank):
    for layer in range(config['num_hidden_layers']):
        attn = f'layers.{layer}.self_attn.'
        mlp = f'layers.{layer}.mlp.'
        qkv,scales,widths = local.load_columns(
            file,[attn+name+'_proj.weight' for name in ('q','k','v')],rank)
        expected = [config['num_attention_heads']*config['head_dim']//local.world,
                    config['num_key_value_heads']*config['head_dim']//local.world]
        if widths != [expected[0],expected[1],expected[1]]:
            raise ValueError('Local Q/K/V row counts differ from the draft head geometry')
        weights[attn+'qkv.weight'] = qkv
        # The teacher projects only K/V. This suffix shares the same allocation.
        weights[attn+'kv.weight'] = local.register(qkv[widths[0]:],scales[widths[0]:])
        gate_up,_,widths = local.load_columns(
            file,[mlp+name+'_proj.weight' for name in ('gate','up')],rank)
        if widths != [config['intermediate_size']//local.world]*2:
            raise ValueError('Local gate/up row counts differ from the MLP geometry')
        weights[mlp+'gate_up.weight'] = gate_up


def load_tp_model(directory, prebuilt, device, *, group=None, single_rank=False,
                  fused_bridge=True, fused_projections=False):
    """Load rank-local W8A16 weights and construct the existing draft model.

    For TP=2 the process group must already be initialized. single_rank loads a
    full reference on this one device without participating in any collective.
    """
    from safetensors import safe_open

    if single_rank:
        world, rank = 1, 0
    else:
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if world != 2:
            raise ValueError(f'This draft backend requires two ranks, got {world}')
    directory = Path(directory)
    torch.cuda.set_device(device)
    config = json.loads((directory / 'config.json').read_text())
    if any(config[key] % world for key in ('num_attention_heads', 'num_key_value_heads',
                                              'intermediate_size')):
        raise ValueError('Draft head/channel counts must divide evenly across ranks')
    local = ShardedLinear(prebuilt, device, group, world)
    weights = {}
    with safe_open(str(directory / 'model.safetensors'), framework='pt', device='cpu') as file:
        for name in sorted(file.keys()):
            axis = projection_axis(name)
            if fused_projections and axis == 0:
                continue
            if name == 'fc.weight' or axis is not None:
                weights[name] = local.load(file.get_slice(name), axis, rank)
            else:
                weights[name] = file.get_tensor(name).to(torch.device('cuda',device))
        if fused_projections:
            load_fused_columns(file,weights,local,config,rank)
    model_class = FusedProjectionModel if fused_projections else plain_model.TorchDspine
    model = model_class(config,weights)
    model.qheads //= world
    model.kheads //= world
    model.linear = local
    if fused_bridge:
        import triton_bridge
        model.bridge = triton_bridge.FusedBridge()
    return model


def compare_tensor(actual, expected):
    delta = actual.float() - expected.float()
    rms = delta.square().mean().sqrt()
    reference_rms = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return {'max_abs': delta.abs().max().item(),
            'relative_rms': (rms / reference_rms).item(),
            'all_finite': torch.isfinite(delta).all().item()}


def measure(body, device, repeats, graph):
    for _ in range(3):
        body()
    torch.cuda.synchronize(device)
    if graph:
        stream = torch.cuda.Stream(device=device)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            graph_outputs = body()
        torch.cuda.current_stream(device).wait_stream(stream)
        for _ in range(3):
            captured.replay()
        torch.cuda.synchronize(device)
        run = captured.replay
    else:
        run = body
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations)


def verify(args):
    import capture_reader
    import pack_io
    import target_layers

    rank = dist.get_rank()
    local_device = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda',local_device)
    allocated_before = torch.cuda.memory_allocated(device)
    model = load_tp_model(args.checkpoint,args.prebuilt,local_device,fused_projections=args.fused_projections)
    resident = torch.cuda.memory_allocated(device) - allocated_before
    manifest = json.loads((args.capture / 'validation.json').read_text())
    proof = next(p for p in manifest['proofs'] if p['kind'] == args.kind)
    response = json.loads((args.capture / (args.kind+'-response.json')).read_text())
    data = capture_reader.load_capture(args.capture / 'shared/capture', proof['stem'], response)
    target_layers.validate_dspine_capture(data['metadata'], model.config['dflash_config']['target_layer_ids'])
    end = data['prompt_tokens'] + args.offset
    if args.offset < 0 or end >= min(data['features'].shape[0],data['tokens'].numel()):
        raise ValueError('Offset exceeds the captured committed prefix')
    tokens = torch.full((model.block,),model.config['dflash_config']['mask_token_id'],dtype=torch.int64)
    tokens[0] = data['tokens'][end]
    pack = pack_io.TargetPack(args.target_pack)
    embeds = pack.embedding_rows(tokens).to(device)
    ids = tokens.to(device)
    positions = torch.arange(end,end+model.block,device=device)
    aux = data['features'][:end].to(device)
    teacher_positions = torch.arange(end,device=device)
    model.append_context('gate',aux,teacher_positions)
    def run():
        return model.forward_block_tensors(embeds,ids,positions,model.cache['gate'])
    hidden,stash = run()
    torch.cuda.synchronize(device)
    # Every communication participant times the same round. Slower rank counts.
    mode = 'graph' if args.graph else 'eager'
    latency = measure(run,device,args.repeats,args.graph)
    latencies = torch.tensor([latency],device=device,dtype=torch.float32)
    dist.all_reduce(latencies,op=dist.ReduceOp.MAX)
    replica = hidden.clone()
    dist.broadcast(replica,src=0)
    consistent = torch.tensor([int(torch.equal(hidden,replica))],device=device)
    dist.all_reduce(consistent,op=dist.ReduceOp.MIN)
    if not consistent.item():
        raise RuntimeError('TP ranks disagree about the replicated draft output')
    # Rank 1 can wait while rank 0 runs a separate unsharded forward.
    if rank == 0:
        ref = load_tp_model(args.checkpoint,args.prebuilt,local_device,single_rank=True)
        ref.append_context('gate',aux,teacher_positions)
        def run_ref():
            return ref.forward_block_tensors(embeds,ids,positions,ref.cache['gate'])
        expected,expected_stash = run_ref()
        comparisons = {'hidden':compare_tensor(hidden,expected)}
        for name in ('raw','packed','rms'):
            comparisons[name] = compare_tensor(stash[name],expected_stash[name])
        local_heads = model.kheads
        for name in ('keys','values'):
            for layer in range(model.layers):
                actual_kv = model.cache['gate'][name][layer]
                ref_kv = ref.cache['gate'][name][layer][:,:local_heads]
                if not torch.equal(actual_kv,ref_kv):
                    raise RuntimeError(f'Committed teacher {name} differ at layer {layer}')
        ref_latency = measure(run_ref,device,args.repeats,args.graph)
        def digest(value):
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            return hashlib.sha256(raw).hexdigest()
        digests = {name:digest(value) for name,value in stash.items()}
        digests['hidden'] = digest(hidden)
        result = {'backbone_only':True,'rank0_weight_resident_bytes':resident,
                  'fused_projections':args.fused_projections,'output_sha256':digests,
                  'capture_kind':args.kind,'teacher_rows':end,'block':model.block,
                  'mode':mode,'samples':args.repeats,'rank_outputs_exact':True,
                  'rank0_teacher_kv_exact':True,'tp2_ms':latencies.item(),
                  'single_gpu_ms':ref_latency,'comparisons':comparisons}
        print(json.dumps(result),flush=True)
        if any(not v['all_finite'] or v['relative_rms'] > args.max_relative_rms for v in comparisons.values()):
            raise RuntimeError('TP partial-sum drift exceeds gate; see comparison output')
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','prebuilt','capture','target-pack'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--kind',choices=('code','prose'),default='code')
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--max-relative-rms',type=float,default=.1)
    p.add_argument('--fused-projections',action='store_true')
    p.add_argument('--graph',action='store_true')
    args = p.parse_args()
    if args.repeats < 1 or not 0 < args.max_relative_rms <= 1:
        p.error('Invalid timing repeat count or numerical gate')
    if os.environ.get('NCCL_P2P_DISABLE','1') != '1':
        p.error('This host requires NCCL_P2P_DISABLE=1')
    os.environ['NCCL_P2P_DISABLE'] = '1'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(minutes=3))
    try:
        with torch.no_grad():
            verify(args)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
