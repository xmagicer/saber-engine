"""CPU protocol and width checks; real-weight graph parity runs with --prewarm-all."""

import unittest
from prewarm import CAPACITIES, WIDTHS, exercise, validate_prefixes


class FakeDraft:
    def __init__(self):
        self.commits = []
        self.reset()

    def reset(self):
        self.next_pos = 0

    def append(self, position, rows):
        if position != self.next_pos:
            raise AssertionError("noncontiguous append")
        self.commits.append(len(rows))
        self.next_pos += len(rows)

    def propose(self, position, anchor, width):
        if position != self.next_pos:
            raise AssertionError("incorrect anchor position")
        return [(anchor + position + i) % 10000 for i in range(width)]


class PrewarmTest(unittest.TestCase):
    def test_all_shapes_with_compatible_short_proposals(self):
        model = FakeDraft()
        seen = exercise(model, list(range(513)), (5, 7, 11))
        self.assertEqual(model.commits, [5, 1, 4, 2, 3, 114, 384])
        self.assertEqual(set(seen), {(cap, k) for cap in CAPACITIES for k in WIDTHS})
        validate_prefixes(seen)
        self.assertEqual(seen[512, 2], (136, 137))
        seen[512, 2] = (137, 136)
        with self.assertRaisesRegex(RuntimeError, "cap=512, width=2"):
            validate_prefixes(seen)


if __name__ == "__main__":
    unittest.main()
