#!/usr/bin/env python3
import unittest

from interval_metrics import (
    event_f1_iou,
    onset_f1_auto_code,
    parse_auto_aeg_intervals,
    parse_canonical_intervals,
    parse_spotsound_intervals,
    setpo_quality,
    setpo_quality_axes,
    soft_precision_recall,
    temporal_set_iou,
)


class IntervalMetricTests(unittest.TestCase):
    def test_parsers_keep_protocol_difference_visible(self):
        answer = "from 1.250s to 3.750s"
        self.assertEqual(parse_spotsound_intervals(answer, 5.0), [(1.25, 3.75)])
        self.assertEqual(parse_auto_aeg_intervals(answer), [])
        self.assertEqual(parse_canonical_intervals(answer, 5.0), [(1.25, 3.75)])
        self.assertEqual(
            parse_auto_aeg_intervals("<answer>[[1.25, 3.75]]</answer>"),
            [(1.25, 3.75)],
        )

    def test_set_iou_penalizes_extra_interval(self):
        ground_truth = [(0.0, 2.0)]
        prediction = [(0.0, 2.0), (8.0, 10.0)]
        self.assertAlmostEqual(temporal_set_iou(ground_truth, prediction), 0.5)
        precision, recall, soft_f1 = soft_precision_recall(ground_truth, prediction)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(soft_f1, 2 / 3)

    def test_public_ev_f1_is_not_paper_event_f1(self):
        ground_truth = [(0.0, 1.0)]
        prediction = [(0.4, 1.4)]
        self.assertEqual(onset_f1_auto_code(ground_truth, prediction)["f1"], 1.0)
        self.assertEqual(event_f1_iou(ground_truth, prediction, 0.5)["f1"], 0.0)

    def test_setpo_quality_orders_candidates(self):
        ground_truth = [(1.0, 3.0), (7.0, 9.0)]
        exact = setpo_quality(ground_truth, ground_truth)
        dropped = setpo_quality(ground_truth, [(1.0, 3.0)])
        dense = setpo_quality(ground_truth, [(0.0, 10.0)])
        self.assertGreater(exact, dropped)
        self.assertGreater(dropped, dense)

    def test_pareto_axes_separate_boundary_and_cardinality_errors(self):
        ground_truth = [(1.0, 3.0), (7.0, 9.0)]
        exact = setpo_quality_axes(ground_truth, ground_truth, 12.0)
        shifted = setpo_quality_axes(ground_truth, [(0.5, 3.0), (6.5, 9.0)], 12.0)
        dropped = setpo_quality_axes(ground_truth, [(1.0, 3.0)], 12.0)
        self.assertEqual(exact, (1.0, 1.0, 1.0, 1.0))
        self.assertLess(shifted[3], exact[3])
        self.assertEqual(shifted[2], 1.0)
        self.assertLess(dropped[2], exact[2])


if __name__ == "__main__":
    unittest.main()
