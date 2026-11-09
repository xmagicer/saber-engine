"""Single-session DFlash2 draft socket; all teacher features come from the attached target."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import torch
from draft_model import DraftModel
from prewarm import prewarm_all


def read_exact(f, n):
    buf = bytearray(n)
    view = memoryview(buf)
    while view:
        nr = f.readinto(view)
        if not nr:
            raise EOFError("Truncated feature frame")
        view = view[nr:]
    return buf


def send(f, value):
    f.write((json.dumps(value, separators=(",", ":")) + "\n").encode())
    f.flush()


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def integer(o, name):
    v = o.get(name)
    if type(v) is not int:
        raise ValueError("Missing integer " + name)
    return v


def handle(c, model, stats):
    reset = False
    with c.makefile("rwb", buffering=1024 * 1024) as f:
        while True:
            line = f.readline(8193)
            if not line:
                return
            if len(line) > 8192 or not line.endswith(b"\n"):
                raise ValueError("Invalid request header")
            try:
                msg = json.loads(line)
                op = msg.get("op")
                if op == "reset":
                    model.reset()
                    reset = True
                    stats["reset"] += 1
                    send(f, {"ok": True})
                elif not reset:
                    raise ValueError("Reset required before use")
                elif op == "append":
                    pos = integer(msg, "position")
                    n = integer(msg, "rows")
                    if pos != model.next_pos or not 1 <= n <= 2048:
                        raise ValueError("Noncontiguous/excessive append")
                    buf = read_exact(f, n * 5 * 2560 * 2)
                    x = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(n, 12800)
                    start = time.perf_counter_ns()
                    model.append(pos, x)
                    stats["append"] += 1
                    stats["rows"] += n
                    stats["enqueue_ns"] += time.perf_counter_ns() - start
                    if stats["append"] == 1:
                        stats["first_teacher_frame_sha256"] = hashlib.sha256(
                            buf
                        ).hexdigest()
                    send(f, {"ok": True, "position": model.next_pos})
                elif op == "propose":
                    pos = integer(msg, "position")
                    anchor = integer(msg, "anchor")
                    k = integer(msg, "width")
                    if pos != model.next_pos or not (
                        1 <= k <= 4 and 0 <= anchor < 248320
                    ):
                        raise ValueError("Bad anchor/width/position")
                    start = time.perf_counter_ns()
                    tokens = model.propose(pos, anchor, k)
                    elapsed = time.perf_counter_ns() - start
                    if len(tokens) != k or any(
                        type(t) is not int or not 0 <= t < 248320 for t in tokens
                    ):
                        raise RuntimeError("Bad proposal")
                    stats["proposals"] += 1
                    stats["propose_ns"] += elapsed
                    if stats["proposals"] <= 8:
                        stats["first_rounds"].append(
                            {
                                "position": pos,
                                "width": k,
                                "tokens": tokens,
                                "ms": elapsed / 1e6,
                            }
                        )
                    send(f, {"ok": True, "tokens": tokens})
                elif op == "stats":
                    send(f, {"ok": True, "stats": stats})
                else:
                    raise ValueError("Unknown operation")
            except (ValueError, EOFError, AssertionError) as e:
                stats["errors"] += 1
                with contextlib.suppress(Exception):
                    send(f, {"ok": False, "error": str(e)[:150]})
                return


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser()
    for n in ("weights", "checkpoint", "socket", "ready", "stats"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--vocab", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--warm-width", type=int, default=4)
    p.add_argument(
        "--prewarm-all",
        action="store_true",
        help="Capture all five commit sizes and all twelve proposal shapes and verify against eager before listening",
    )
    a = p.parse_args()
    if a.prewarm_all and (a.eager or not a.device.startswith("cuda")):
        p.error("--prewarm-all requires GPU graphs")
    assert sys.byteorder == "little" and not a.socket.exists()
    torch.set_num_threads(1)
    if a.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(a.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic()
    with torch.inference_mode():
        model = DraftModel(
            a.weights,
            a.checkpoint,
            a.device,
            a.vocab,
            not a.eager and a.device.startswith("cuda"),
        )
        # Exercise initial capture and numerics before accepting a live target; the reset removes all warmup state.
        x = torch.zeros((1, 12800), dtype=torch.bfloat16)
        model.append(0, x)
        output = model.propose(1, 248044, a.warm_width)
        assert len(output) == a.warm_width
        model.reset()
        coverage = prewarm_all(model) if a.prewarm_all else None
        ready = {
            "prewarm_all": coverage,
            "ready": True,
            "model": "incoai/Qwen3.8-Flash-Next-DFlash2",
            "draft_step_rows": 8,
            "hidden": 2560,
            "reference_taps": [5, 15, 25, 35, 45],
            "target_activations": "completed four-HC mean, BF16",
            "device": a.device,
            "graph": model.graphs,
            "vocab_rows": model.vocab.numel(),
            "fp32_head_output": model.fp32_mm,
            "load_and_warm_s": time.monotonic() - start,
            "embed_residency": "CPU-mapped BF16 checkpoint",
            "head_source": "BF16 lm_head.weight from the same Quark target checkpoint",
        }
        if a.device.startswith("cuda"):
            ready.update(
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
            )
        stats = {
            "reset": 0,
            "append": 0,
            "rows": 0,
            "enqueue_ns": 0,
            "proposals": 0,
            "propose_ns": 0,
            "errors": 0,
            "first_rounds": [],
        }

        def stop(sig, frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(a.socket))
            a.socket.chmod(0o600)
            listener.listen(2)
            write(a.ready, ready)
            print(json.dumps(ready), flush=True)
            try:
                while True:
                    conn, _ = listener.accept()
                    try:
                        with conn:
                            handle(conn, model, stats)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        write(a.stats, stats)
            finally:
                write(a.stats, stats)
                a.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
