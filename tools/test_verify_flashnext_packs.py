"""Regression checks for the multiple Flash-Next member formats."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import verify_flashnext_packs as verify


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FlashNextChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, data):
        path = self.root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(data))

    def test_raw_member_has_no_magic_and_checks_the_indexed_extent(self):
        pack = self.root / "trunk8.pack"
        pack.write_bytes(b"raw-bytes")
        store = {"pack": "trunk8.pack", "data_off": 4, "length": 5, "sha256": sha(b"bytes")}
        index = {"bytes": 9, "records": [{"name": "w", "storage": store}]}
        self.write_json("trunk8.json", index)
        verify.indexed_member(self.root, "trunk8")
        pack.write_bytes(b"raw-wrong")
        with self.assertRaisesRegex(ValueError, "sha256"):
            verify.indexed_member(self.root, "trunk8")
        store["length"] = 6
        self.write_json("trunk8.json", index)
        with self.assertRaisesRegex(ValueError, "outside"):
            verify.indexed_member(self.root, "trunk8")

    def test_xtier_checks_table_and_whole_stride(self):
        data = b"x" * 4096
        table = hashlib.sha256(data).digest()
        h = bytearray(4096)
        h[:8] = b"SBRXPRT1"
        struct.pack_into("<6I", h, 8, 1, 4096, 0, 1, 0, 1)
        struct.pack_into("<2Q", h, 32, 4096, 6)
        sub = [[i, 1] for i in range(6)]
        for i, (off, n) in enumerate(sub):
            struct.pack_into("<2Q", h, 48 + 16 * i, off, n)
        struct.pack_into("<4Q", h, 152, 4096, 32, 8192, 12288)
        h[184:216] = hashlib.sha256(table).digest()
        struct.pack_into("<2I", h, 248, 2, 0)
        body = h + table + bytes(4096 - 32) + data
        tier = {
            "rank": 0,
            "file": "rank0/experts.tp2.xtier",
            "layers": 1,
            "experts": 1,
            "first_expert": 0,
            "records": 1,
            "record": {"stride": 4096, "payload": 6, "sub": sub},
            "data_off": 8192,
            "file_bytes": len(body),
            "table_sha256": sha(table),
        }
        path = self.root / tier["file"]
        path.parent.mkdir()
        path.write_bytes(body)
        verify.expert_tier(self.root, 0, tier)
        body[-1] ^= 1  # Padding is included in the expert stride's digest.
        path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, "record 0 sha256"):
            verify.expert_tier(self.root, 0, tier)
        body[4096] ^= 1
        path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, "table sha256"):
            verify.expert_tier(self.root, 0, tier)

    def test_vision_checks_record_against_member_index(self):
        name = b"visual.patch"
        data = b"vision-weight"
        record = bytearray(160)
        struct.pack_into("<2I", record, 0, 0, len(name))
        struct.pack_into("<2Q", record, 24, 4096, len(data))
        record[104:136] = hashlib.sha256(data).digest()
        h = bytearray(256)
        h[:8] = b"SBRPACK1"
        size = 4096 + len(data)
        struct.pack_into("<4I5Q2I", h, 8, 1, 256, 160, 1, 256, 416, len(name), 4096, size, 2, 1)
        h[144:176] = hashlib.sha256(record + name).digest()
        p = self.root / "vision/rank1.pack"
        p.parent.mkdir()
        p.write_bytes(h + record + name + bytes(4096 - 416 - len(name)) + data)
        ix = {
            "file_bytes_per_rank": size,
            "records": [{"name": name.decode(), "length": len(data), "sha256": sha(data)}],
        }
        verify.vision_member(self.root, 1, ix)
        ix["records"][0]["name"] = "a-different-vision-tensor"
        with self.assertRaisesRegex(ValueError, "differs from vision index"):
            verify.vision_member(self.root, 1, ix)

    def test_checkpoint_strided_reference_and_ple(self):
        folder = self.root / "original"
        folder.mkdir()
        header = struct.pack("<Q", 2) + b"{}"
        name = "model-00001.safetensors"
        data = header + b"abXXcd"
        (folder / name).write_bytes(data)
        self.write_json("original/config.json", {})
        self.write_json(
            "original/model.safetensors.index.json", {"weight_map": {"w": name, "ple0": name}}
        )
        self.write_json(
            "manifest.json",
            {
                "checkpoint": {
                    "shards": [
                        {"name": name, "size_bytes": len(data), "header_sha256": sha(header)}
                    ]
                }
            },
        )
        record = {
            "name": "w",
            "bytes": 4,
            "storage": {
                "checkpoint": [{"file": name, "offset": 10, "length": 2, "count": 2, "stride": 4}],
                "sha256": sha(b"abcd"),
            },
        }
        for k in ["rank0", "rank1", "shared"]:
            self.write_json(k + "/dense.json", {"records": [record]})
        ple = {
            "shard": 0,
            "first_row": 0,
            "tensor": "ple0",
            "file": name,
            "offset": 10,
            "bytes": 2,
            "rows": 1,
            "sha256": sha(b"ab"),
        }
        self.write_json(
            "ple/ple_index.json",
            {
                "table": {"shards": 1, "rows": 1},
                "sources": {"ckpt_bf16": {"row_bytes": 2, "shards": [ple]}},
            },
        )
        verify.verify_checkpoint(self.root, folder)
        record["storage"]["checkpoint"][0]["stride"] = 5
        self.write_json("rank0/dense.json", {"records": [record]})
        with self.assertRaisesRegex(ValueError, "outside"):
            verify.verify_checkpoint(self.root, folder)


if __name__ == "__main__":
    unittest.main()
