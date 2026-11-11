"""Check the single-CTA tree selector against the independent Python heap."""
import json
import math
import statistics
import struct
import time

import torch
import live_service
import tree_proposal
import triton_tree


def examples():
    ids = torch.arange(16, dtype=torch.int64).repeat(15, 1) + 100
    for seed in range(6):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn((15, 16, 16), generator=generator) * (1 + seed)
        yield f"random-{seed}", ids + torch.randint(1000, (15, 1), generator=generator), logits.log_softmax(-1)
    yield "uniform-ties", ids, torch.full((15, 16, 16), -math.log(16.))
    sharp = torch.full((15, 16, 16), -math.inf)
    sharp[:, :, 7] = 0.
    yield "depth-15", ids, sharp
    tied = sharp.clone()
    tied[0] = -math.inf
    tied[0, :, :2] = -math.log(2.)
    yield "depth-token-parent-ties", ids, tied
    # The two paths at depth 2 round to the same FP32 cumulative score but
    # remain ordered in FP64. A prefix rounded to FP32 picks the wrong parent.
    precise = tied.clone()
    precise[1, 0, 7] = -1.e-10
    precise[1, 0, 8] = math.log(1.e-10)
    yield "prefix-requires-fp64", ids, precise


def main():
    torch.cuda.set_device(0)
    ids = torch.zeros((15, 16), dtype=torch.int64, device="cuda")
    logp = torch.zeros((15, 16, 16), dtype=torch.float32, device="cuda")
    chain = torch.arange(15, dtype=torch.int64, device="cuda") + 121
    fixtures = list(examples())
    ids.copy_(fixtures[0][1])
    logp.copy_(fixtures[0][2])
    triton_tree.read_rows(triton_tree.select(ids, logp, chain))
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        captured = triton_tree.select(ids, logp, chain)
    passed = []
    for label, ref_ids, ref_logp in fixtures:
        ids.copy_(ref_ids)
        logp.copy_(ref_logp)
        graph.replay()
        rows = triton_tree.read_rows(captured)
        oracle = tree_proposal.build_best_first_tree(ref_ids.tolist(), ref_logp.tolist())
        assert len(oracle) == 15
        for row, node in zip(rows, oracle):
            assert row[:4] == [node.token, node.parent, node.depth, node.lane], (label, row, node)
            score = struct.unpack("=d", struct.pack("=q", row[4]))[0]
            assert score == node.log_probability, (label, score, node.log_probability)
        assert [row[5] for row in rows] == chain.cpu().tolist()
        if label == "depth-15":
            assert tree_proposal.ancestor_masks(oracle)[-1] == 65535
        passed.append(label)
    bad = []
    for kind in ("duplicate", "negative-id", "nan", "positive", "unnormalized", "empty-root"):
        _, ref_ids, ref_logp = fixtures[0]
        ref_ids = ref_ids.clone()
        ref_logp = ref_logp.clone()
        if kind == "duplicate":
            ref_ids[14, 15] = ref_ids[14, 14]
        elif kind == "negative-id":
            ref_ids[0, 0] = -1
        elif kind == "empty-root":
            ref_logp[0, 0] = -math.inf
        else:
            # A bad conditional in the final row is rejected even if unreachable.
            ref_logp[14, 15, 15] = {"nan": math.nan, "positive": 1., "unnormalized": 0.}[kind]
        try:
            tree_proposal.build_best_first_tree(ref_ids.tolist(), ref_logp.tolist())
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid fixture passed CPU oracle: {kind}")
        ids.copy_(ref_ids)
        logp.copy_(ref_logp)
        graph.replay()
        try:
            triton_tree.read_rows(captured)
        except ValueError as error:
            if kind == "empty-root":
                assert "fewer than 15 finite nodes" in str(error)
        else:
            raise AssertionError(f"Invalid fixture passed GPU selector: {kind}")
        bad.append(kind)

    ids.copy_(fixtures[0][1])
    logp.copy_(fixtures[0][2])
    torch.cuda.synchronize()
    def previous():
        return chain.cpu().tolist(), live_service.wire_tree(ids.cpu().tolist(), logp.cpu().tolist())
    def gpu():
        return triton_tree.wire_result(ids, logp, chain, 15)
    assert previous() == gpu()
    times = {"python": [], "gpu": []}
    for repeat in range(32):
        variants = [("python", previous), ("gpu", gpu)]
        for name, fun in (variants if repeat % 2 == 0 else reversed(variants)):
            start = time.perf_counter_ns()
            fun()
            times[name].append((time.perf_counter_ns() - start) * 1.e-3)
    print(json.dumps({"passed": True, "exact_graph_cases": passed, "invalid_refused": bad,
                      "wire_median_us": {k: statistics.median(v) for k, v in times.items()}}, indent=2))


if __name__ == "__main__":
    main()
