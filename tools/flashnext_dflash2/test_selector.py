"""Check greedy predecessor conditioning and short terminal proposals on CPU."""

import unittest

import torch

from draft_model import DraftModel


class SelectorTest(unittest.TestCase):
    def test_short_proposals_and_ties(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(909)
        model = DraftModel.__new__(DraftModel)
        model.device = torch.device("cpu")
        model.fp32_mm = False
        model.qid = torch.tensor([47])
        model.vocab = torch.arange(3, 68, 2)
        model.head = torch.randn(len(model.vocab), 32, generator=g)
        model.w = {
            "candidate_selector.hidden_projection.weight": torch.randn(
                8, 32, generator=g
            ),
            "candidate_selector.predecessor_codebook": torch.randn(72, 8, generator=g),
            "candidate_selector.successor_codebook": torch.randn(72, 8, generator=g),
        }
        for hidden in (torch.randn(4, 32, generator=g), torch.zeros(4, 32)):
            expected = []
            pred = 47
            for row in hidden:
                logits = (model.head @ row).tolist()
                coarse = sorted(
                    zip(model.vocab.tolist(), logits), key=lambda v: (-v[1], v[0])
                )[:16]
                proj = model.w["candidate_selector.hidden_projection.weight"] @ row
                query = proj * model.w["candidate_selector.predecessor_codebook"][pred]

                def score(pair):
                    token, unary = pair
                    score = unary + float(
                        model.w["candidate_selector.successor_codebook"][token].dot(
                            query
                        )
                    )
                    return score, -token

                pred = max(coarse, key=score)[0]
                expected.append(pred)
            for width in range(1, 5):
                self.assertEqual(
                    model.select_tokens(hidden[:width]).tolist(), expected[:width]
                )


if __name__ == "__main__":
    unittest.main()
