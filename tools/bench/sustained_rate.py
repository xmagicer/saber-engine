"""Measure how long client-side rolling one-second token delivery meets a threshold."""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

WINDOW_NS = 1_000_000_000


def measure(result, target):
    timing = result.get("token_arrivals") or {}
    if not result.get("passed") or not timing.get("available") or timing["window_ns"] != WINDOW_NS:
        raise ValueError("A successful --stream-timing run with complete one-second windows is required")
    events = timing["events"]
    times = [event["elapsed_ns"] for event in events]
    if not times or times != sorted(times):
        raise ValueError("Token arrivals are missing or out of order")
    if any(type(event["tokens"]) is not int or event["tokens"] <= 0 for event in events):
        raise ValueError("Expected positive generated-token counts")
    prefix = [0]
    for event in events:
        prefix.append(prefix[-1] + event["tokens"])
    if prefix[-1] != result["usage"]["completion_tokens"] or prefix[-1] != timing["observed_tokens"]:
        raise ValueError("Token counts disagree with the response usage")

    # The window (end - 1 s, end] moves along the decode interval. Its count only
    # changes when an arrival enters at end or leaves at end - 1 s.
    first, last = times[0] + WINDOW_NS, times[-1]
    if last <= first:
        raise ValueError("More than one second of generated-ID arrivals is required to measure duration")
    ends = {first, last}
    for ts in times:
        if first < ts < last:
            ends.add(ts)
        if first < ts + WINDOW_NS < last:
            ends.add(ts + WINDOW_NS)
    bounds = sorted(ends)
    above_ns = streak_ns = longest_ns = integral = peak = 0
    for begin, end in zip(bounds, bounds[1:]):
        count = prefix[bisect_right(times, begin)] - prefix[bisect_right(times, begin - WINDOW_NS)]
        elapsed = end - begin
        integral += count * elapsed
        peak = max(peak, count)
        if count >= target:
            above_ns += elapsed
            streak_ns += elapsed
            longest_ns = max(longest_ns, streak_ns)
        else:
            streak_ns = 0
    final_count = prefix[-1] - prefix[bisect_right(times, last - WINDOW_NS)]
    return {
        "name": result["name"],
        "observable_ns": last - first,
        "at_target_ns": above_ns,
        "longest_at_target_ns": longest_ns,
        "mean_rolling_tok_s": integral / (last - first),
        "max_rolling_tok_s": max(peak, final_count),
        "at_target_percent": 100 * above_ns / (last - first),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="paired_http .result.json files")
    parser.add_argument("--target", type=float, required=True, help="Generated tokens per second")
    args = parser.parse_args()
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error("--target must be finite and positive")
    rows = [measure(json.loads(path.read_text(encoding="utf-8")), args.target) for path in args.results]
    duration = sum(row["observable_ns"] for row in rows)
    print(json.dumps({
        "target_tok_s": args.target,
        "time_weighted_at_target_percent": 100 * sum(row["at_target_ns"] for row in rows) / duration,
        "longest_at_target_seconds": max(row["longest_at_target_ns"] for row in rows) / WINDOW_NS,
        "mean_rolling_tok_s": sum(row["mean_rolling_tok_s"] * row["observable_ns"] for row in rows) / duration,
        "max_rolling_tok_s": max(row["max_rolling_tok_s"] for row in rows),
        "requests": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
