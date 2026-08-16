"""Focused tests for Contained Partial Match Rate (CPMR)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/experiments"))

from metrics import (  # noqa: E402
    AMP_LABEL_IDS,
    compute_amp_cpmr,
    contained_partial_match,
    contained_recall,
    labels_to_indicator,
)


class ContainedPartialMatchTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(contained_partial_match({"A", "B"}, {"A", "B"}), 1)
        self.assertEqual(contained_recall({"A", "B"}, {"A", "B"}), 1.0)

    def test_proper_contained_subset(self) -> None:
        reference = {"A", "B", "C"}
        prediction = {"A", "C"}
        self.assertEqual(contained_partial_match(reference, prediction), 1)
        self.assertAlmostEqual(contained_recall(reference, prediction), 2 / 3)

    def test_extra_label_fails_and_recall_is_na(self) -> None:
        self.assertEqual(contained_partial_match({"A", "B"}, {"A", "C"}), 0)
        self.assertIsNone(contained_recall({"A", "B"}, {"A", "C"}))

    def test_empty_prediction_fails_and_recall_is_na(self) -> None:
        self.assertEqual(contained_partial_match({"A", "B"}, set()), 0)
        self.assertIsNone(contained_recall({"A", "B"}, set()))

    def test_empty_reference_is_safe(self) -> None:
        self.assertEqual(contained_partial_match(set(), set()), 0)
        self.assertEqual(contained_partial_match(set(), {"A"}), 0)
        self.assertIsNone(contained_recall(set(), set()))
        self.assertIsNone(contained_recall(set(), {"A"}))

    def test_order_and_duplicate_presentation_do_not_matter(self) -> None:
        reference = ["A", "B", "C"]
        prediction = ["C", "A", "A"]
        self.assertEqual(contained_partial_match(reference, prediction), 1)
        self.assertAlmostEqual(contained_recall(reference, prediction), 2 / 3)

    def test_act_means_and_purpose_are_independent(self) -> None:
        act_a, act_b = AMP_LABEL_IDS[:2]
        means_a, means_b = AMP_LABEL_IDS[5:7]
        purpose_a, purpose_b = AMP_LABEL_IDS[11:13]
        reference = labels_to_indicator(
            [
                [act_a, act_b, means_a, purpose_a, purpose_b],
                [act_a, means_a, means_b, purpose_a],
            ]
        )
        prediction = labels_to_indicator(
            [[act_b, means_b, purpose_a], [means_a, purpose_a]]
        )

        result = compute_amp_cpmr(reference, prediction)

        self.assertEqual(result["test_n"], 2)
        self.assertEqual(
            result["per_case"],
            [
                {
                    "act_cpmr": 1,
                    "act_contained_recall": 0.5,
                    "means_cpmr": 0,
                    "means_contained_recall": None,
                    "purpose_cpmr": 1,
                    "purpose_contained_recall": 0.5,
                },
                {
                    "act_cpmr": 0,
                    "act_contained_recall": None,
                    "means_cpmr": 1,
                    "means_contained_recall": 0.5,
                    "purpose_cpmr": 1,
                    "purpose_contained_recall": 1.0,
                },
            ],
        )
        self.assertEqual(result["by_family"]["ACT"]["cpmr"], 0.5)
        self.assertEqual(result["by_family"]["ACT"]["mean_contained_recall"], 0.5)
        self.assertEqual(result["by_family"]["MEANS"]["cpmr"], 0.5)
        self.assertEqual(result["by_family"]["MEANS"]["mean_contained_recall"], 0.5)
        self.assertEqual(result["by_family"]["PURPOSE"]["cpmr"], 1.0)
        self.assertEqual(
            result["by_family"]["PURPOSE"]["mean_contained_recall"], 0.75
        )

    def test_no_successes_has_na_mean_contained_recall(self) -> None:
        reference = labels_to_indicator([[], []])
        prediction = labels_to_indicator([[], []])
        result = compute_amp_cpmr(reference, prediction)
        for family in ("ACT", "MEANS", "PURPOSE"):
            self.assertEqual(result["by_family"][family]["cpmr"], 0.0)
            self.assertEqual(result["by_family"][family]["success_count"], 0)
            self.assertIsNone(
                result["by_family"][family]["mean_contained_recall"]
            )


if __name__ == "__main__":
    unittest.main()
