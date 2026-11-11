#!/usr/bin/env python3
"""Compare a pack you built against a reference pack's manifest.json, record by record.

Every Saber pack's ``manifest.json`` lists each record's name, dtype, shape, shard kind, layout,
offset, length and the sha256 of its bytes. A reference manifest is small (about 1 MB) and is
published next to every hosted pack, so you can check a local build without downloading the
reference pack itself:

    hf download Launch80/Qwen3.8-27B-Quark-AWQ-MXFP4-SABER \\
        qwen38-27b-quark-awq-mxfp4-tp1/manifest.json --local-dir ref/
    tools/pack/check_manifest.py $SABER_PACKS/qwen38-27b-quark-awq-mxfp4-tp1 \\
        ref/qwen38-27b-quark-awq-mxfp4-tp1/manifest.json [--rehash]

PASS means: the same tier id (the checkpoint's safetensors headers and config/tokenizer files),
the same plan hash, the same record table, and every record's sha256 equal to the reference's --
so the pack's payload is byte-identical to the reference pack. ``built_unix`` (the build time) is
the only field allowed to differ; it is stamped into each rank file's header, so the rank FILES'
own sha256 differ unless the build passed the reference's time (``--build-unix``).

``--rehash`` also re-reads every record of your rank files and checks it against YOUR manifest
(the builder hashes while it writes; this proves the bytes on disk are those bytes). Numpy is not
needed. Exit 0 on PASS, 1 on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

KEYS = ("dtype", "shape", "shard_kind", "layout_id", "data_off", "length", "sha256")
STEP = 64 << 20


def load(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "manifest.json")
    with open(path) as f:
        return json.load(f)


def rehash(pack: str, man: dict) -> int:
    bad = 0
    for rk in man["ranks"]:
        path = os.path.join(pack, rk["file"])
        fd = os.open(path, os.O_RDONLY)
        try:
            for r in rk["records"]:
                h = hashlib.sha256()
                pos = 0
                while pos < r["length"]:
                    b = os.pread(fd, min(STEP, r["length"] - pos), r["data_off"] + pos)
                    if not b:
                        break
                    h.update(b)
                    pos += len(b)
                if pos != r["length"] or h.hexdigest() != r["sha256"]:
                    bad += 1
                    print(f"REHASH DIFFERS rank {rk['rank']} {r['name']}")
        finally:
            os.close(fd)
        print(f"rehash rank {rk['rank']}: {len(rk['records'])} records read from {rk['file']}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("pack", help="your pack directory (manifest.json + rank<r>.pack)")
    ap.add_argument("reference", help="the reference manifest.json (or a directory holding one)")
    ap.add_argument("--rehash", action="store_true", help="also re-hash your rank files")
    a = ap.parse_args(argv)
    got, want = load(a.pack), load(a.reference)
    bad = 0
    for key, g, w in (("tp_size", got["tp_size"], want["tp_size"]),
                      ("format_version", got["format_version"], want["format_version"]),
                      ("tier id", got["tier"]["id"], want["tier"]["id"]),
                      ("plan hash", got["plan"]["hash"], want["plan"]["hash"])):
        ok = g == w
        bad += not ok
        print(f"{key:15s} {'equal' if ok else 'DIFFERS'}: {str(g)[:16]}{'' if ok else ' vs ' + str(w)[:16]}")
    if got["tier"]["id"] != want["tier"]["id"]:
        gd, wd = got["tier"]["descriptor"], want["tier"]["descriptor"]
        if gd.get("model") != wd.get("model"):
            print(f"  the checkpoint directory's NAME is part of the tier id: yours "
                  f"{gd.get('model')!r}, the reference {wd.get('model')!r}")
        for f in sorted(set(gd.get("extras", {})) | set(wd.get("extras", {}))):
            if gd.get("extras", {}).get(f) != wd.get("extras", {}).get(f):
                print(f"  {f} differs from the reference's")
    if got["tp_size"] != want["tp_size"]:
        print("FAIL: different tensor-parallel sizes")
        return 1
    for rg, rw in zip(sorted(got["ranks"], key=lambda r: r["rank"]),
                      sorted(want["ranks"], key=lambda r: r["rank"])):
        bg = {r["name"]: r for r in rg["records"]}
        bw = {r["name"]: r for r in rw["records"]}
        missing = sorted(set(bw) - set(bg))
        extra = sorted(set(bg) - set(bw))
        diff = [n for n in bw if n in bg and any(bg[n].get(k) != bw[n].get(k) for k in KEYS)]
        order = [r["name"] for r in rg["records"]] == [r["name"] for r in rw["records"]]
        same = len(bw) - len(missing) - len(diff)
        print(f"rank {rw['rank']}: {same}/{len(bw)} records identical"
              f"{'' if order else ', table ORDER differs'}"
              f"{f', {len(missing)} missing' if missing else ''}"
              f"{f', {len(extra)} extra' if extra else ''}"
              f"{f', {len(diff)} DIFFER' if diff else ''}"
              f"; file {rg['file_bytes']} vs {rw['file_bytes']} B")
        for n in (missing + extra + diff)[:10]:
            print(f"   {n}")
        bad += len(missing) + len(extra) + len(diff) + (not order) \
            + (rg["file_bytes"] != rw["file_bytes"])
    if a.rehash:
        bad += rehash(a.pack, got)
    if got.get("built_unix") != want.get("built_unix"):
        print(f"built_unix {got.get('built_unix')} vs {want.get('built_unix')} (informational; "
              f"the rank files' headers carry it)")
    print("check_manifest: PASS" if bad == 0 else f"check_manifest: FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
