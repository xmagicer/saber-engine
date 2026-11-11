"""Serve DSpine greedy proposals from committed Saber residual rows over AF_UNIX.

Launch the drafter on one gfx1201 device before starting the target. Each
connection owns its sequence caches. For append, a JSON header ending in LF
is followed by rows * 51200 bytes of row-major little-endian BF16.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys

import tree_proposal


FEATURE_LAYERS = [2, 17, 32, 47, 62]
HIDDEN = 5120
ROW_BYTES = len(FEATURE_LAYERS) * HIDDEN * 2
MAX_FRAME_ROWS = 2048
MAX_HEADER_BYTES = 4096
BLOCK = 16


def uint(header, field, maximum, minimum=0):
    value = header.get(field)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {field}, expected integer in [{minimum}, {maximum}]")
    return value


def read_payload(reader, size):
    payload = bytearray(size)
    view = memoryview(payload)
    offset = 0
    while offset < size:
        count = reader.readinto(view[offset:])
        if not count:
            raise EOFError("Connection closed inside BF16 append frame")
        offset += count
    return payload


def wire_tree(candidates, log_probabilities):
    """Map target-row parents (anchor = 0) to wire parents (anchor = null)."""
    nodes = tree_proposal.build_best_first_tree(
        candidates, log_probabilities, max_nodes=BLOCK - 1, max_depth=BLOCK - 1,
    )
    return [
        {"token": node.token, "parent": None if node.parent == 0 else node.parent - 1}
        for node in nodes
    ]


def validate_tree(tree, vocab):
    """Reject ambiguous siblings and forward references before sending a plan."""
    if not isinstance(tree, list) or len(tree) != BLOCK - 1:
        raise ValueError("DSpine tree requires exactly 15 nodes")
    edges = set()
    for index, node in enumerate(tree):
        if not isinstance(node, dict) or set(node) != {"token", "parent"}:
            raise ValueError("Invalid DSpine tree node")
        token = uint(node, "token", vocab - 1)
        parent = node["parent"]
        if parent is not None and (type(parent) is not int or not 0 <= parent < index):
            raise ValueError("DSpine tree parent must precede its child")
        if (parent, token) in edges:
            raise ValueError("DSpine tree has duplicate sibling tokens")
        edges.add((parent, token))


def serve_connection(connection, backend, max_seqs):
    """Service one ordered target session. A failed frame invalidates the session."""
    next_position = {}
    try:
        with connection.makefile("rb") as reader:
            while True:
                line = reader.readline(MAX_HEADER_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_HEADER_BYTES or not line.endswith(b"\n"):
                    raise ValueError("Invalid DSpine request header length")
                header = json.loads(line)
                if not isinstance(header, dict):
                    raise ValueError("Expected a JSON object")
                seq = uint(header, "seq", (1 << 64) - 1)
                op = header.get("op")
                answer = {"ok": True}
                if op == "clear":
                    backend.clear(seq)
                    next_position.pop(seq, None)
                elif op == "append":
                    position = uint(header, "position", (1 << 32) - 1)
                    rows = uint(header, "rows", MAX_FRAME_ROWS, 1)
                    if position != next_position.get(seq, 0):
                        raise ValueError("DSpine append position differs from committed cursor")
                    if seq not in next_position and len(next_position) >= max_seqs:
                        raise ValueError("DSpine active sequence capacity exceeded")
                    payload = read_payload(reader, rows * ROW_BYTES)
                    # Include it in cleanup even if the backend fails halfway through the append.
                    next_position[seq] = position
                    backend.append(seq, position, rows, payload)
                    next_position[seq] = position + rows
                    del payload
                elif op == "propose":
                    position = uint(header, "position", (1 << 32) - 1)
                    width = uint(header, "width", BLOCK - 1, 1)
                    anchor = uint(header, "anchor", backend.vocab - 1)
                    if position != next_position.get(seq):
                        raise ValueError("DSpine proposal must follow the committed cursor")
                    if "tree_nodes" in header:
                        uint(header, "tree_nodes", BLOCK - 1, BLOCK - 1)
                        if width != BLOCK - 1:
                            raise ValueError("DSpine 15-node tree requires width=15")
                        propose_tree = getattr(backend, "propose_tree", None)
                        if propose_tree is None:
                            raise ValueError("DSpine backend does not support tree proposals")
                        tokens, tree = propose_tree(seq, position, anchor, width)
                        validate_tree(tree, backend.vocab)
                        answer["tree"] = tree
                    else:
                        tokens = backend.propose(seq, position, anchor, width)
                    if len(tokens) != width or any(type(t) is not int or not 0 <= t < backend.vocab for t in tokens):
                        raise RuntimeError("DSpine returned an invalid proposal")
                    answer["tokens"] = tokens
                else:
                    raise ValueError("Unknown DSpine request operation")
                connection.sendall(json.dumps(answer, separators=(",", ":")).encode() + b"\n")
    except (ValueError, OSError, RuntimeError, EOFError) as error:
        message = str(error)
        print(f"DSpine session closed: {message}", file=sys.stderr, flush=True)
        try:
            connection.sendall(json.dumps({"ok": False, "error": message}).encode() + b"\n")
        except OSError:
            pass
    finally:
        for seq in next_position:
            backend.clear(seq)


class ProposalInputs:
    """Load one anchor embedding plus the learned mask block on a target rank."""
    def __init__(self, target, model, device):
        import torch
        self.torch = torch
        self.target = target
        self.device = device
        self.mask_id = int(model.config["dflash_config"]["mask_token_id"])
        self.mask_embedding = self.embedding(self.mask_id)

    def embedding(self, token):
        ids = self.torch.tensor([token], dtype=self.torch.int64)
        return self.target.embedding_rows(ids)[0].to(self.device)

    def inputs(self, anchor, position):
        torch = self.torch
        ids = torch.full((BLOCK,), self.mask_id, device=self.device, dtype=torch.int64)
        ids[0] = anchor
        embeds = self.mask_embedding.expand(BLOCK, -1).clone()
        embeds[0].copy_(self.embedding(anchor))
        positions = torch.arange(position, position + BLOCK, device=self.device)
        return embeds, ids, positions


class ProposalHead(ProposalInputs):
    """DSpine target embeddings, int2 candidate selection and serial refinement."""
    def __init__(self, target, model, draft_pack, prebuilt, device, *, gpu_tree=False, sparse_shard=None):
        import native_head
        import triton_tail

        super().__init__(target, model, device)
        self.tail = triton_tail
        self.tree_selector = None
        if gpu_tree:
            import triton_tree
            self.tree_selector = triton_tree.wire_result
        self.vocab = target.vocab
        self.head = self.target.load_head(self.device)
        self.sparse = native_head.NativeSparseHead(draft_pack, prebuilt, device.index, shard_rank=sparse_shard)
        if self.sparse.vocab != self.vocab:
            raise ValueError("DSpine candidate head vocabulary differs from target")
        weights = model.w
        self.norm = weights["norm.weight"]
        self.bias = weights["selected_prefix_bridge.gate.bias"]
        rank = int(model.config["dflash_config"]["selected_prefix_code_bridge"]["rank"])
        self.tg, self.tsv = triton_tail.build_serial_tables(
            self.head, self.norm, weights["selected_prefix_bridge.feature_write_weights"][-1],
            model.packed[rank:], weights["selected_prefix_bridge.codebook"], chunk=4096,
        )

    def tokens(self, hidden, stash, ids, width, *, with_tree=False, candidates=None):
        torch = self.torch
        if candidates is None:
            candidates = self.sparse.candidates(hidden[1:], self.head, pool_per_shard=80)
        candidates = candidates.contiguous()
        table = self.tail.serial_table(
            stash["raw"], self.norm, candidates, ids, self.head, self.tg, self.tsv,
            stash["packed"], stash["rms"], self.bias, BLOCK, with_log_probabilities=with_tree,
        )
        indices = table[0] if with_tree else table
        tokens = torch.empty(BLOCK - 1, device=self.device, dtype=torch.int64)
        self.tail.serial_chain(indices, candidates, tokens, BLOCK)
        if with_tree and self.tree_selector is not None:
            return self.tree_selector(candidates, table[1], tokens, width)
        chain = tokens[:width].cpu().tolist()
        if with_tree:
            return chain, wire_tree(candidates.cpu().tolist(), table[1].cpu().tolist())
        return chain


class SingleGpuDrafter:
    """Channel-FP8 projections, fused BF16 bridge and a target-pack rerank head."""

    def __init__(self, args):
        import native_linear
        import pack_io
        import plain_model
        import torch
        import triton_bridge

        self.torch = torch
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device("cuda:0")
        self.target = pack_io.TargetPack(args.target_pack)
        self.vocab = self.target.vocab
        self.model = plain_model.TorchDspine.from_checkpoint(args.checkpoint, "cpu")
        model = self.model
        if (model.capture_layer_ids != FEATURE_LAYERS or model.h != HIDDEN
                or model.block != BLOCK or self.target.hidden != HIDDEN
                or model.w["selected_prefix_bridge.codebook"].shape[0] != self.vocab):
            raise ValueError("Unsupported DSpine checkpoint or target vocabulary")
        self.mask_id = int(model.config["dflash_config"]["mask_token_id"])

        # Keep the mapped CPU BF16 matrices as stable keys for NativeChannelLinear.
        # Upload one row group at a time to bound transient quantization allocations.
        linear = native_linear.NativeChannelLinear({}, args.prebuilt, row_band=32)
        for name, weight in model.w.items():
            if name != "fc.weight" and not (name.startswith("layers.") and name.endswith("_proj.weight")):
                model.w[name] = weight.to(self.device)
                continue
            quantized = torch.empty(weight.shape, device=self.device, dtype=torch.uint8)
            scales = torch.empty(weight.shape[0], device=self.device, dtype=torch.float32)
            for start in range(0, weight.shape[0], 256):
                part = weight[start:start + 256].to(self.device)
                values, row_scales = native_linear.quantize_channel_weight(part)
                quantized[start:start + part.shape[0]].copy_(values)
                scales[start:start + part.shape[0]].copy_(row_scales)
            linear.weights[weight.data_ptr()] = (weight.shape, quantized, scales)
        model.device = self.device
        model.inv_freq = model.theta ** (-torch.arange(0, model.dim, 2, device=self.device, dtype=torch.float32) / model.dim)
        model.packed = model.packed.to(self.device)
        model.linear = linear
        model.bridge = triton_bridge.FusedBridge()
        self.proposal_head = ProposalHead(self.target, model, args.draft_pack, args.prebuilt, self.device, gpu_tree=args.gpu_tree)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def append_tensor(self, seq, position, features):
        positions = self.torch.arange(position, position + len(features), device=self.device)
        self.model.append_context(seq, features, positions)

    def append(self, seq, position, rows, payload):
        host = self.torch.frombuffer(payload, dtype=self.torch.bfloat16).reshape(rows, -1)
        # Cache the committed features in bounded pieces, including long chunked prefill.
        for start in range(0, rows, 256):
            self.append_tensor(seq, position + start, host[start:start + 256].to(self.device))

    def propose(self, seq, position, anchor, width, *, with_tree=False):
        embeds, ids, positions = self.proposal_head.inputs(anchor, position)
        hidden, stash = self.model.forward_block(seq, embeds, ids, positions)
        return self.proposal_head.tokens(hidden, stash, ids, width, with_tree=with_tree)

    def propose_tree(self, seq, position, anchor, width):
        return self.propose(seq, position, anchor, width, with_tree=True)

    def clear(self, seq):
        self.model.clear(seq)

    def warmup(self, rows):
        # Reserve/measure the live single-sequence scratch before target KV allocation.
        # Non-integer key keeps warmup outside the wire sequence namespace.
        seq = "warmup"
        tile = self.torch.zeros((min(256, rows), ROW_BYTES // 2), device=self.device,
                                dtype=self.torch.bfloat16)
        try:
            for position in range(0, rows, len(tile)):
                self.append_tensor(seq, position, tile[:min(len(tile), rows - position)])
            self.propose(seq, rows, self.mask_id, BLOCK - 1, with_tree=self.proposal_head.tree_selector is not None)
        finally:
            self.clear(seq)
        self.torch.cuda.synchronize()

    def memory_stats(self):
        return {"gpu_tree": self.proposal_head.tree_selector is not None,
                "allocated_bytes": self.torch.cuda.memory_allocated(),
                "reserved_bytes": self.torch.cuda.memory_reserved(),
                "peak_allocated_bytes": self.torch.cuda.max_memory_allocated()}


def listen(path, backend, max_seqs):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        previous = os.umask(0o177)
        try:
            listener.bind(os.fspath(path))
        finally:
            os.umask(previous)
        bound = path.lstat()
        try:
            listener.listen(2)
            print(json.dumps({"ready": True, "socket": str(path), **backend.memory_stats()}), flush=True)
            while True:
                connection, _ = listener.accept()
                with connection:
                    serve_connection(connection, backend, max_seqs)
        finally:
            if path.exists() and os.path.samestat(bound, path.lstat()):
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("socket", "checkpoint", "target-pack", "draft-pack", "prebuilt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-tree", action="store_true", help="Select 15-node proposal trees on the GPU")
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--warmup-rows", type=int, default=8192)
    args = parser.parse_args()
    if args.max_seqs < 1 or not 1 <= args.warmup_rows <= 8192:
        parser.error("max-seqs must be positive and warmup-rows must be in [1,8192]")
    if sys.byteorder != "little":
        parser.error("Live BF16 payloads require a little-endian server")
    import torch
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    with torch.inference_mode():
        backend = SingleGpuDrafter(args)
        backend.warmup(args.warmup_rows)
        listen(args.socket, backend, args.max_seqs)


if __name__ == "__main__":
    main()
