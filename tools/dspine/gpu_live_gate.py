"""Compare the live socket's proposals to a previously measured DSpine trajectory.

The target features and anchors come from the real committed-token capture.
The corresponding fused-bridge chain reference contains one summary and steps
per JSON file. Neither its proposals nor future tokens are sent to the server.
"""
import argparse
import json
from pathlib import Path
import socket

import torch
import capture_reader
import live_service


def exchange(connection, reader, op, seq, payload=None, **fields):
    connection.sendall(json.dumps({"op": op, "seq": seq, **fields}).encode() + b"\n")
    if payload is not None:
        connection.sendall(payload)
    result = json.loads(reader.readline(live_service.MAX_HEADER_BYTES + 1))
    if result.get("ok") is not True:
        raise RuntimeError(f"Live {op} failed: {result}")
    return result


def run(args):
    proofs = {proof["kind"]: proof for proof in json.loads((args.capture / "validation.json").read_text())["proofs"]}
    reports = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(120)
        connection.connect(str(args.socket))
        with connection.makefile("rb") as reader:
            for seq, path in enumerate(args.golden, start=1):
                gold = json.loads(path.read_text())
                summary = gold["summary"]
                if summary.get("tree_node_budget") != 0 or int(summary["block_size"]) != live_service.BLOCK:
                    raise ValueError("Expected a full-block DSpine chain reference")
                kind = summary["kind"]
                proof = proofs[kind]
                response = json.loads((args.capture / (kind + "-response.json")).read_text())
                data = capture_reader.load_capture(args.capture / "shared/capture", proof["stem"], response)
                if data["metadata"].get("target_layer_ids") != live_service.FEATURE_LAYERS:
                    raise ValueError("Capture layer order differs from the live protocol")
                next_row = 0
                differences = []
                steps = gold["steps"][:args.max_rounds]
                if not steps or steps[0]["position"] != data["prompt_tokens"]:
                    raise ValueError("Reference does not start at the captured prompt boundary")
                for step in steps:
                    position = step["position"]
                    if not next_row < position <= len(data["features"]):
                        raise ValueError("Invalid reference trajectory positions")
                    while next_row < position:
                        rows = min(position - next_row, live_service.MAX_FRAME_ROWS)
                        features = data["features"][next_row:next_row + rows].contiguous()
                        payload = memoryview(features.view(torch.uint8).numpy()).cast("B")
                        if len(payload) != rows * live_service.ROW_BYTES:
                            raise ValueError("Wrong capture row geometry")
                        exchange(connection, reader, "append", seq, payload=payload, position=next_row, rows=rows)
                        next_row += rows
                    anchor = int(data["tokens"][position])
                    actual = exchange(connection, reader, "propose", seq, position=position,
                                      anchor=anchor, width=args.width)["tokens"]
                    expected = step["proposal"][:args.width]
                    if actual != expected:
                        differences.append(position)
                exchange(connection, reader, "clear", seq)
                reports.append({"kind": kind, "rounds": len(steps), "width": args.width,
                                "mismatched_rounds": len(differences), "first_mismatches": differences[:8]})
    passed = all(not case["mismatched_rounds"] for case in reports)
    print(json.dumps({"passed": passed, "cases": reports}, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--golden", type=Path, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=7, choices=range(1, live_service.BLOCK))
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("max-rounds must be positive")
    raise SystemExit(0 if run(args) else 1)
