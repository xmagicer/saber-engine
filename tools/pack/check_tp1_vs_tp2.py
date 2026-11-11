#!/usr/bin/env python3
"""Check a TP=1 pack against the TP=2 pack of the same checkpoint. Host only, no GPU.

Record by record, matched by name (the record SET must be the same):

* ``Replicated`` -- the TP=1 bytes == rank 0's == rank 1's (sha256 from the manifests, and the
  TP=1 sha256 re-taken from the pack file);
* ``Dim0`` row-major layouts (``raw``, ``f32_from_bf16``, ``bf16_from_fp8_channel``) -- the TP=1
  bytes == rank 0's bytes followed by rank 1's (streamed sha256 of the concatenation);
* ``Dim1`` ``raw`` -- every TP=1 row == rank 0's row followed by rank 1's;
* every other sharded layout (the MXFP4 trios, merged segments) -- shape and length law only:
  ``rank0 + rank1`` along the split axis == TP=1, and the byte lengths sum to the TP=1 length
  (the values are checked against the checkpoint by ``check_quark27b.py``).

The file law: ``rank0.pack`` of TP=1 is the sum of the two TP=2 rank payloads plus the
replicated records counted once, within the 4 KiB record alignment.

    tools/pack/check_tp1_vs_tp2.py --tp1 PACK1 --tp2 PACK2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "I32": 4, "U32": 4}
ROW_MAJOR_DIM0 = {"raw", "f32_from_bf16", "bf16_from_fp8_channel", "bf16",
                  "int2_head", "int2_head_scale", "int2_head_zero"}  # row-local layouts
STEP = 64 << 20


def load(pack: str) -> dict:
    with open(os.path.join(pack, "manifest.json")) as f:
        return json.load(f)


def chunks(path: str, off: int, length: int):
    fd = os.open(path, os.O_RDONLY)
    try:
        pos = 0
        while pos < length:
            b = os.pread(fd, min(STEP, length - pos), off + pos)
            if not b:
                raise SystemExit(f"{path}: short read at {off + pos}")
            pos += len(b)
            yield b
    finally:
        os.close(fd)


def sha_of(parts) -> str:
    h = hashlib.sha256()
    for path, off, length in parts:
        for b in chunks(path, off, length):
            h.update(b)
    return h.hexdigest()


def dim1_rows_equal(p1: str, r1: dict, p2: list, r2: list) -> bool:
    n, k = r1["shape"][0], r1["shape"][-1]
    if len(r1["shape"]) != 2:
        return False
    eb = DTYPE_BYTES[r1["dtype"]]
    ks = [r["shape"][-1] for r in r2]
    if sum(ks) != k:
        return False
    rows = max(1, STEP // (k * eb))
    fds = [os.open(p1, os.O_RDONLY)] + [os.open(p, os.O_RDONLY) for p in p2]
    try:
        for a in range(0, n, rows):
            b = min(n, a + rows)
            t = os.pread(fds[0], (b - a) * k * eb, r1["data_off"] + a * k * eb)
            halves = [os.pread(fd, (b - a) * kk * eb, r["data_off"] + a * kk * eb)
                      for fd, r, kk in zip(fds[1:], r2, ks)]
            for i in range(b - a):
                want = b"".join(h[i * kk * eb:(i + 1) * kk * eb] for h, kk in zip(halves, ks))
                if t[i * k * eb:(i + 1) * k * eb] != want:
                    return False
    finally:
        for fd in fds:
            os.close(fd)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tp1", required=True)
    ap.add_argument("--tp2", required=True)
    ap.add_argument("--no-rehash", action="store_true",
                    help="trust the TP=1 manifest's sha256 for replicated records")
    a = ap.parse_args(argv)
    m1, m2 = load(a.tp1), load(a.tp2)
    if m1["tp_size"] != 1 or m2["tp_size"] != 2:
        raise SystemExit(f"tp sizes {m1['tp_size']} / {m2['tp_size']}, want 1 / 2")
    (rk1,) = m1["ranks"]
    rks2 = sorted(m2["ranks"], key=lambda r: r["rank"])
    f1 = os.path.join(a.tp1, rk1["file"])
    f2 = [os.path.join(a.tp2, r["file"]) for r in rks2]
    by1 = {r["name"]: r for r in rk1["records"]}
    by2 = [{r["name"]: r for r in rk["records"]} for rk in rks2]
    names1, names2 = list(by1), list(by2[0])
    if set(names1) != set(names2) or set(by2[0]) != set(by2[1]):
        only1 = sorted(set(names1) - set(names2))[:5]
        only2 = sorted(set(names2) - set(names1))[:5]
        raise SystemExit(f"record sets differ: only TP=1 {only1}, only TP=2 {only2}")
    if names1 != names2:
        print("note: record ORDER differs between TP=1 and TP=2 (names equal)")
    stats: dict = {}
    bad = []
    repl_bytes = 0
    for name in names1:
        r1, r2 = by1[name], [b[name] for b in by2]
        kind, lay = r1["shard_kind"], r1.get("layout")
        if any((r["shard_kind"], r.get("layout"), r["dtype"], r["layout_id"]) !=
               (kind, lay, r1["dtype"], r1["layout_id"]) for r in r2):
            bad.append((name, "kind/layout/dtype differ")); continue
        if kind == "Replicated":
            ok = all(r["sha256"] == r1["sha256"] for r in r2) and \
                all(r["shape"] == r1["shape"] and r["length"] == r1["length"] for r in r2)
            if ok and not a.no_rehash:
                ok = sha_of([(f1, r1["data_off"], r1["length"])]) == r1["sha256"]
            how = "replicated: sha equal"
            repl_bytes += r1["length"]
        elif kind == "Dim0" and lay in ROW_MAJOR_DIM0:
            ok = (r1["shape"][0] == sum(r["shape"][0] for r in r2)
                  and r1["length"] == sum(r["length"] for r in r2)
                  and sha_of([(p, r["data_off"], r["length"]) for p, r in zip(f2, r2)])
                  == r1["sha256"]
                  and (a.no_rehash or sha_of([(f1, r1["data_off"], r1["length"])]) == r1["sha256"]))
            how = "Dim0 rows: TP1 == rank0 || rank1"
        elif kind == "Dim1" and lay == "raw":
            ok = dim1_rows_equal(f1, r1, f2, r2)
            how = "Dim1 rows: TP1 row == rank0 row || rank1 row"
        elif kind == "Dim1" and lay == "mxfp4_row_ref":
            # A K-split trio keeps one e8m0 reference PER OUTPUT ROW on every rank (the rank's own
            # max over its K half), so each rank carries the full [N] and the TP=1 reference (max
            # over the whole K) is the elementwise max of the two ranks' -- per-row bytes, which
            # the payload law counts once like a replicated record.
            import numpy as np
            got = [np.frombuffer(b"".join(chunks(p, r["data_off"], r["length"])), np.uint8)
                   for p, r in [(f1, r1)] + list(zip(f2, r2))]
            ok = (all(r["shape"] == r1["shape"] and r["length"] == r1["length"] for r in r2)
                  and np.array_equal(got[0], np.maximum(got[1], got[2])))
            how = "Dim1 row_ref: TP1 == max(rank0, rank1)"
            repl_bytes += r1["length"]
        else:
            ax = 1 if kind == "Dim1" else 0
            if r1["layout_id"] == 2:  # mxfp4 block scale is stored [K/32, N]: the axes swap
                ax = 1 - ax
            sh_ok = r1["shape"][ax] == sum(r["shape"][ax] for r in r2) and all(
                r["shape"][1 - ax] == r1["shape"][1 - ax] for r in r2) \
                if len(r1["shape"]) == 2 else r1["shape"][0] == sum(r["shape"][0] for r in r2)
            ok = sh_ok and r1["length"] == sum(r["length"] for r in r2)
            how = f"{kind}/{lay}: shape + length law"
        s = stats.setdefault(how, [0, 0])
        s[0] += 1
        s[1] += ok
        if not ok:
            bad.append((name, how))
    pay1 = sum(r["length"] for r in rk1["records"])
    pay2 = [sum(r["length"] for r in rk["records"]) for rk in rks2]
    print(f"records: TP=1 {len(names1)}, TP=2 {len(names2)} a rank")
    for how, (n, ok) in sorted(stats.items()):
        print(f"  {ok:5d}/{n:<5d} {how}")
    print(f"payload: TP=1 {pay1} B; TP=2 rank0 {pay2[0]} + rank1 {pay2[1]} = {sum(pay2)} B; "
          f"replicated + per-row refs {repl_bytes} B counted twice there -> {sum(pay2) - repl_bytes} B "
          f"({'EQUAL' if sum(pay2) - repl_bytes == pay1 else 'DIFFER by ' + str(sum(pay2) - repl_bytes - pay1)})")
    print(f"files:   TP=1 {rk1['file_bytes']} B; TP=2 {[r['file_bytes'] for r in rks2]} B "
          f"(alignment padding TP=1 {rk1['file_bytes'] - pay1} B)")
    if os.path.getsize(f1) != rk1["file_bytes"]:
        bad.append((rk1["file"], f"size {os.path.getsize(f1)} != manifest {rk1['file_bytes']}"))
    if sum(pay2) - repl_bytes != pay1:
        bad.append(("payload", "TP=2 sum less replicated != TP=1"))
    for b in bad[:20]:
        print(f"FAIL {b[0]}: {b[1]}")
    print(f"check_tp1_vs_tp2: {'PASS' if not bad else f'{len(bad)} FAIL'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
