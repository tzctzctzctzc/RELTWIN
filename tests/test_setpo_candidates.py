#!/usr/bin/env python3
import unittest

from setpo_candidates import (
    RELATION_EXCHANGE_PERMUTATION,
    candidate_qualities,
    ordinary_candidates,
    relation_candidates,
)


class SetPOCandidateTests(unittest.TestCase):
    def test_relation_candidates_are_exchange_symmetric(self):
        ab = [(1.0, 3.0)]
        ba = [(7.0, 9.0)]
        candidates = relation_candidates(ab, ba, 12.0, 0.15)
        quality_ab = candidate_qualities(ab, candidates)
        quality_ba = candidate_qualities(ba, candidates)
        exchanged_ba = [quality_ba[index] for index in RELATION_EXCHANGE_PERMUTATION]
        for first, second in zip(quality_ab, exchanged_ba):
            self.assertAlmostEqual(first, second)
        self.assertEqual(max(range(len(quality_ab)), key=quality_ab.__getitem__), 0)
        self.assertEqual(max(range(len(quality_ba)), key=quality_ba.__getitem__), 1)

    def test_ordinary_candidates_cover_failure_modes(self):
        ground_truth = [(2.0, 4.0), (8.0, 10.0)]
        candidates = ordinary_candidates(ground_truth, 14.0, 0.15)
        qualities = candidate_qualities(ground_truth, candidates)
        self.assertEqual(len(candidates), 6)
        self.assertEqual(qualities[0], 1.0)
        self.assertTrue(all(qualities[0] > value for value in qualities[1:]))
        self.assertEqual(len(candidates[2]), 1)
        self.assertGreater(len(candidates[4]), len(ground_truth))


if __name__ == "__main__":
    unittest.main()
