"""Checks for duration-weighted one-second token windows."""
import unittest

from sustained_rate import measure


def result(events):
    counts = sum(count for _, count in events)
    return {
        "passed": True, "name": "sample", "usage": {"completion_tokens": counts},
        "token_arrivals": {
            "available": True, "window_ns": 1_000_000_000, "observed_tokens": counts,
            "events": [{"elapsed_ns": t, "tokens": n} for t, n in events],
        },
    }


class WindowTests(unittest.TestCase):
    def test_simultaneous_arrivals_and_departures(self):
        data = result([(0, 1), (500_000_000, 3), (1_000_000_000, 2),
                       (1_500_000_000, 2), (1_500_000_000, 3), (2_000_000_000, 1)])
        row = measure(data, 6)
        self.assertEqual(row["observable_ns"], 1_000_000_000)
        self.assertEqual(row["at_target_percent"], 50)
        self.assertEqual(row["longest_at_target_ns"], 500_000_000)
        self.assertEqual(row["mean_rolling_tok_s"], 6)
        self.assertEqual(row["max_rolling_tok_s"], 7)

    def test_short_or_inconsistent_stream_has_no_observable_duration(self):
        with self.assertRaises(ValueError):
            measure(result([(0, 3), (1_000_000_000, 4)]), 4)
        data = result([(0, 1), (2_000_000_000, 4)])
        data["usage"]["completion_tokens"] += 1
        with self.assertRaises(ValueError):
            measure(data, 4)


if __name__ == "__main__":
    unittest.main()
