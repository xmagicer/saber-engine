#!/usr/bin/env python3
"""Check the Flash-Next distribution against its member indexes, without touching it."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_bytes())


def size_is(path, expected):
    require(
        type(expected) is int and expected > 0 and path.stat().st_size == expected,
        f"{path}: expected {expected} bytes",
    )


def extent(start, length, size, label):
    require(
        type(start) is int
        and type(length) is int
        and start >= 0
        and length > 0
        and start + length <= size,
        f"{label}: extent {start}+{length} outside {size}",
    )


def update_digest(digest, stream, start, length):
    stream.seek(start)
    while length:
        chunk = stream.read(min(length, 4 << 20))
        require(chunk, f"{stream.name}: truncated payload")
        digest.update(chunk)
        length -= len(chunk)


def digest_range(stream, start, length):
    digest = hashlib.sha256()
    update_digest(digest, stream, start, length)
    return digest.hexdigest()


def indexed_member(directory, stem, file_bytes=None):
    """trunk8, mtp_* and hc8 start with tensor bytes; dense may also have checkpoint references."""
    index = load_json(directory / (stem + ".json"))
    pack = directory / (stem + ".pack")
    size = index["bytes"] if file_bytes is None else file_bytes
    size_is(pack, size)
    require(index["records"], f"{pack}: empty index")
    regions = []
    for record in index["records"]:
        store = record["storage"]
        if stem == "dense" and "checkpoint" in store and "pack" not in store:
            continue
        label = f"{pack}:{record['name']}"
        require(store["pack"] == pack.name, f"{label}: unexpected member {store['pack']}")
        start, length = store["data_off"], store["length"]
        extent(start, length, size, label)
        regions.append((start, length, store["sha256"], label))
    require(regions, f"{pack}: no materialized records")
    with pack.open("rb") as stream:
        end = 0
        for start, length, digest, label in sorted(regions):
            require(start >= end, f"{label}: overlaps previous record")
            require(digest_range(stream, start, length) == digest, f"{label}: sha256 mismatch")
            end = start + length


def expert_tier(directory, rank, tier):
    """SBRXPRT1: a header, one digest per full record stride, and the records."""
    path = directory / f"rank{rank}/experts.tp2.xtier"
    require(
        tier["rank"] == rank and tier["file"] == f"rank{rank}/{path.name}",
        f"{path}: mismatched rank or filename",
    )
    size_is(path, tier["file_bytes"])
    with path.open("rb") as stream:
        h = stream.read(4096)
        require(len(h) == 4096 and h[:8] == b"SBRXPRT1", f"{path}: invalid expert header")
        version, header, r, layers, first, experts = struct.unpack_from("<6I", h, 8)
        stride, payload = struct.unpack_from("<2Q", h, 32)
        sub = [list(struct.unpack_from("<2Q", h, 48 + 16 * i)) for i in range(6)]
        require(sub == tier["record"]["sub"], f"{path}: expert sub-records differ from index")
        end = 0
        for offset, length in sub:
            extent(offset, length, payload, str(path))
            require(offset >= end, f"{path}: overlapping expert sub-records")
            end = offset + length
        table, table_bytes, start, size = struct.unpack_from("<4Q", h, 152)
        parts, part = struct.unpack_from("<2I", h, 248)
        count = layers * experts
        require(
            (version, header, r, layers, first, experts, parts, part)
            == (1, 4096, rank, tier["layers"], tier["first_expert"], tier["experts"], 2, rank),
            f"{path}: expert geometry differs from TP2 index",
        )
        require(
            (count, stride, payload, start, size)
            == (
                tier["records"],
                tier["record"]["stride"],
                tier["record"]["payload"],
                tier["data_off"],
                tier["file_bytes"],
            ),
            f"{path}: invalid expert extent metadata",
        )
        require(
            count > 0
            and 0 < payload <= stride
            and table >= header
            and table_bytes == 32 * count
            and table + table_bytes <= start
            and start + count * stride == size,
            f"{path}: invalid table or data bounds",
        )
        stream.seek(table)
        hashes = stream.read(table_bytes)
        require(
            len(hashes) == table_bytes
            and hashlib.sha256(hashes).digest() == h[184:216]
            and h[184:216].hex() == tier["table_sha256"],
            f"{path}: expert table sha256 mismatch",
        )
        for i in range(count):
            require(
                digest_range(stream, start + i * stride, stride)
                == hashes[32 * i : 32 * i + 32].hex(),
                f"{path}: expert record {i} sha256 mismatch",
            )


def vision_member(directory, rank, index):
    path = directory / f"vision/rank{rank}.pack"
    size_is(path, index["file_bytes_per_rank"])
    expected = {r["name"]: (r["length"], r["sha256"]) for r in index["records"]}
    with path.open("rb") as stream:
        h = stream.read(256)
        require(len(h) == 256 and h[:8] == b"SBRPACK1", f"{path}: invalid vision header")
        version, header, stride, count = struct.unpack_from("<4I", h, 8)
        table, names, names_len, start, size = struct.unpack_from("<5Q", h, 24)
        tp, r = struct.unpack_from("<2I", h, 64)
        require(
            version in (1, 2)
            and (header, stride, tp, r, size) == (256, 160, 2, rank, path.stat().st_size)
            and count == len(expected) == len(index["records"])
            and table >= header
            and table + count * stride <= names
            and names + names_len <= start <= size,
            f"{path}: invalid vision table geometry",
        )
        stream.seek(table)
        records = stream.read(count * stride)
        stream.seek(names)
        strings = stream.read(names_len)
        require(
            hashlib.sha256(records + strings).digest() == h[144:176],
            f"{path}: vision table sha256 mismatch",
        )
        if version == 2:
            chunk_off, chunk_bytes = struct.unpack_from("<2Q", h, 176)
            if chunk_bytes:
                require(
                    chunk_off == names + names_len and chunk_off + chunk_bytes <= start,
                    f"{path}: invalid vision chunk metadata bounds",
                )
                require(
                    digest_range(stream, chunk_off, chunk_bytes) == h[192:224].hex(),
                    f"{path}: vision chunk table sha256 mismatch",
                )
            else:
                require(
                    chunk_off == 0 and h[192:224] == bytes(32),
                    f"{path}: inconsistent empty vision chunk table",
                )
        for i in range(count):
            record = records[i * stride : (i + 1) * stride]
            name_off, name_len = struct.unpack_from("<2I", record)
            extent(name_off, name_len, names_len, str(path))
            name = strings[name_off : name_off + name_len].decode()
            offset, length = struct.unpack_from("<2Q", record, 24)
            extent(offset, length, size, f"{path}:{name}")
            require(
                offset >= start and expected.pop(name, None) == (length, record[104:136].hex()),
                f"{path}:{name}: record differs from vision index",
            )
            require(
                digest_range(stream, offset, length) == record[104:136].hex(),
                f"{path}:{name}: sha256 mismatch",
            )


def verify(directory):
    manifest = load_json(directory / "manifest.json")
    indexes = load_json(directory / "experts_tp2.json")["tiers"]
    tiers = {t["rank"]: t for t in indexes}
    require(
        len(indexes) == 2 and set(tiers) == {0, 1}, "expected one expert tier for each of two ranks"
    )
    vision = load_json(directory / "vision/vision.json")
    for rank in (0, 1):
        placement = f"rank{rank}"
        for stem in ("trunk8", "mtp_fp8", "mtp_x4"):
            indexed_member(directory / placement, stem)
        indexed_member(
            directory / placement, "dense", manifest["dense"][placement]["pack_file_bytes"]
        )
        expert_tier(directory, rank, tiers[rank])
        vision_member(directory, rank, vision)
        print(
            f"Flash-Next rank {rank}: member sizes, extents and payload digests verified",
            flush=True,
        )
    indexed_member(directory / "shared", "hc8")
    indexed_member(directory / "shared", "dense", manifest["dense"]["shared"]["pack_file_bytes"])


def verify_checkpoint(directory, checkpoint):
    manifest = load_json(directory / "manifest.json")
    shards = manifest["checkpoint"]["shards"]
    declared = {}
    for s in shards:
        name = s["name"]
        require(
            isinstance(name, str) and Path(name).name == name and name not in declared,
            f"invalid checkpoint shard name: {name}",
        )
        path = checkpoint / name
        size_is(path, s["size_bytes"])
        with path.open("rb") as f:
            raw = f.read(8)
            require(len(raw) == 8, f"{path}: missing safetensors length")
            (n,) = struct.unpack("<Q", raw)
            extent(8, n, s["size_bytes"], str(path))
            require(
                digest_range(f, 0, 8 + n) == s["header_sha256"],
                f"{path}: safetensors header sha256 mismatch",
            )
        declared[name] = s["size_bytes"]
    weights = load_json(checkpoint / "model.safetensors.index.json")["weight_map"]
    require(
        weights and set(weights.values()) <= set(declared),
        "weight_map refers to unavailable shards",
    )
    require((checkpoint / "config.json").is_file(), "missing target config.json")
    for placement in ("rank0", "rank1", "shared"):
        for row in load_json(directory / placement / "dense.json")["records"]:
            store = row["storage"]
            if "checkpoint" not in store:
                continue
            label = f"{placement}:{row['name']}"
            digest = hashlib.sha256()
            total = 0
            for ref in store["checkpoint"]:
                name, start, length = ref["file"], ref["offset"], ref["length"]
                count, stride = ref.get("count", 1), ref.get("stride", 0)
                require(
                    name in declared
                    and type(count) is int
                    and count >= 1
                    and type(stride) is int
                    and stride >= 0,
                    f"{label}: bad checkpoint reference",
                )
                extent(start, length, declared[name], label)
                extent(start + (count - 1) * stride, length, declared[name], label)
                with (checkpoint / name).open("rb") as f:
                    for i in range(count):
                        update_digest(digest, f, start + i * stride, length)
                total += count * length
            require(
                total == row["bytes"] and digest.hexdigest() == store["sha256"],
                f"{label}: checkpoint record size or sha256 mismatch",
            )
    ple = load_json(directory / "ple/ple_index.json")
    source = ple["sources"]["ckpt_bf16"]
    table = ple["table"]
    require(
        len(source["shards"]) == table["shards"] and table["shards"] > 0, "incomplete PLE index"
    )
    first_row = 0
    for shard, row in enumerate(source["shards"]):
        name = row["file"]
        require(
            row["shard"] == shard
            and row["first_row"] == first_row
            and row["rows"] > 0
            and weights.get(row["tensor"]) == name,
            f"{name}: PLE shard order, row coverage or tensor source differs",
        )
        first_row += row["rows"]
        require(
            name in declared and row["bytes"] == row["rows"] * source["row_bytes"],
            f"{name}: invalid PLE record",
        )
        extent(row["offset"], row["bytes"], declared[name], name)
        with (checkpoint / name).open("rb") as f:
            require(
                digest_range(f, row["offset"], row["bytes"]) == row["sha256"],
                f"{row['tensor']}: PLE sha256 mismatch",
            )
    require(first_row == table["rows"], "PLE row coverage differs from table geometry")
    print(
        f"Flash-Next checkpoint: {len(shards)} shard headers, dense references and PLE payloads verified",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="check the original checkpoint against the manifest instead of checking pack members",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        verify(args.pack)
    else:
        verify_checkpoint(args.pack, args.checkpoint)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, struct.error) as err:
        sys.exit(f"Flash-Next verification failed: {err}")
