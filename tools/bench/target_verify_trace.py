"""Report only complete, rank-local Qwen-27B 542-kernel tree verification graphs."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path


def classify(names, i):
    name = names[i]
    prev = names[i - 1] if i else ""
    following = names[i + 1] if i + 1 < len(names) else ""
    if name.startswith("mxfp4_gemm_decode_arnq"):
        for prefix, label in (("silu_mul_quant", "mlp.down"), ("gdn_norm_quant", "gdn.out"),
                              ("attn_gate_quant", "attn.o")):
            if prev.startswith(prefix):
                return "arnq." + label
        return "arnq.unrecognized"
    if name.startswith("mxfp4_gemm_decode"):
        if following.startswith("silu_mul_quant"):
            return "mlp.gate_up"
        if name.startswith("mxfp4_gemm_decode_pair"):
            return "gdn.in_pair"
        if following.startswith("qk_rope_append"):
            return "attn.qkv"
        if prev.startswith("silu_mul_quant"):
            return "mlp.down_final_unfused"
    if name.startswith("dflash_bgemm") and following.startswith("saber_argmax"):
        return "lm_head"
    return name


def ms_summary(values):
    ms = sorted(x / 1e6 for x in values)
    return {"mean_ms": statistics.mean(ms), "median_ms": statistics.median(ms),
            "p95_ms": ms[math.ceil(len(ms) * .95) - 1], "max_ms": ms[-1]}


def analyze(rows):
    streams = defaultdict(list)
    for agent, queue, stream, start, end, name in rows:
        if end < start:
            raise ValueError("negative kernel duration")
        streams[agent, queue, stream].append((start, end, name))
    graphs = []
    rejected = Counter()
    for stream, trace in streams.items():
        trace.sort()
        for begin, (_, _, start_name) in enumerate(trace):
            if start_name != "saber_embed_bf16":
                continue
            candidate = trace[begin:begin + 542]
            if len(candidate) < 542:
                rejected["incomplete_tail"] += 1
                continue
            names = [r[2] for r in candidate]
            if not names[-1].startswith("saber_device_row_stage") or not names[-2].startswith("saber_argmax"):
                rejected["different_graph_or_truncation"] += 1
                continue
            families = [classify(names, i) for i in range(542)]
            counts = Counter(families)
            required = {"arnq.mlp.down": 63, "arnq.gdn.out": 48, "arnq.attn.o": 16,
                        "mlp.gate_up": 64, "mlp.down_final_unfused": 1, "gdn.in_pair": 48, "attn.qkv": 16}
            if (any(counts[k] != v for k, v in required.items())
                    or sum(n.startswith("p27_gdn_tree") for n in names) != 48
                    or sum(n.startswith("p27_attn_decode_anc") for n in names) != 16
                    or names.count("saber_embed_bf16") != 1):
                rejected["different_body"] += 1
                continue
            durations = [end - start for start, end, _ in candidate]
            graphs.append((stream, candidate[0][0], max(r[1] for r in candidate), names, families, durations))

    base = {"csv_dispatches": sum(map(len, streams.values())), "complete_graphs": len(graphs),
            "nodes_per_graph": 542, "rejected_embed_anchors": dict(rejected)}
    if not graphs:
        return base
    streams_used = {g[0] for g in graphs}
    if len(streams_used) > 1:
        raise ValueError(f"multiple streams contain verification graphs: {streams_used}; supply a single worker trace")
    total = sum(sum(g[-1]) for g in graphs)

    def aggregate(index, *, group_fused=False):
        samples, calls = defaultdict(dict), Counter()
        for j, g in enumerate(graphs):
            per = Counter()
            for name, dur in zip(g[index], g[-1]):
                per[name] += dur
                calls[name] += 1
            if group_fused:
                per["arnq.total"] = sum(v for k, v in per.items() if k.startswith("arnq."))
                calls["arnq.total"] += sum(n.startswith("arnq.") for n in g[index])
            for name, ns in per.items():
                samples[name][j] = ns
        result = {}
        for name, per in sorted(samples.items(), key=lambda kv: -sum(kv[1].values())):
            ns = sum(per.values())
            result[name] = {**ms_summary([per.get(j, 0) for j in range(len(graphs))]),
                            "calls_per_graph": calls[name] / len(graphs),
                            "mean_us_per_call": ns / calls[name] / 1000,
                            "pct_kernel_sum": 100 * ns / total}
        return result

    return {**base, "stream": list(next(iter(streams_used))), "span": ms_summary([g[2] - g[1] for g in graphs]),
            "kernel_sum": ms_summary([sum(g[-1]) for g in graphs]), "families": aggregate(4, group_fused=True),
            "kernels": aggregate(3)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", help="worker rank kernel_trace.csv or - for stdin")
    p.add_argument("--window", type=int, nargs=2, metavar=("START_NS", "END_NS"),
                   help="keep only kernels contained in the request's monotonic time window")
    p.add_argument("--output", type=Path, help="write JSON here (otherwise stdout)")
    args = p.parse_args()
    with (nullcontext(sys.stdin) if args.trace == "-" else open(args.trace, newline="", encoding="utf-8-sig")) as f:
        table = csv.DictReader(f)
        if args.window and args.window[1] < args.window[0]:
            p.error("the end of the window precedes its start")

        def dispatches():
            for r in table:
                start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
                if args.window and (start < args.window[0] or end > args.window[1]):
                    continue
                yield r["Agent_Id"], r["Queue_Id"], r.get("Stream_Id", ""), start, end, r["Kernel_Name"]
        result = analyze(dispatches())
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if not result["complete_graphs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
