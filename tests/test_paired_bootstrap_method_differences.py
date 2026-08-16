from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.metrics import AMP_LABEL_IDS


paired = importlib.import_module("src.experiments.22_generate_paired_bootstrap")


def _data(
    reference: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    m4: np.ndarray,
    *,
    evaluation: str = "A3",
) -> object:
    return paired.EvaluationMatrices(
        evaluation=evaluation,
        reference=reference,
        predictions={"M2": m2, "M3": m3, "M4": m4},
        macro_label_ids=tuple(AMP_LABEL_IDS),
    )


class PairedBootstrapTests(unittest.TestCase):
    def test_paired_identical_methods_have_exact_zero_difference(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.integers(0, 2, size=(55, 17), dtype=np.uint8)
        prediction = rng.integers(0, 2, size=(55, 17), dtype=np.uint8)
        rows = paired.paired_bootstrap_rows(
            _data(reference, prediction, prediction.copy(), prediction.copy()),
            n_resamples=40,
            seed=20260811,
        )
        self.assertEqual(len(rows), 3 * 7)
        self.assertTrue(all(row["point_difference"] == 0.0 for row in rows))
        self.assertTrue(
            all(row["ci_low"] == 0.0 and row["ci_high"] == 0.0 for row in rows)
        )
        self.assertTrue(all(row["ci_excludes_zero"] is False for row in rows))

    def test_exact_set_difference_uses_same_sampled_cases(self) -> None:
        # M3 is exact on the first 27 cases and M2 on the last 28.  Each paired
        # replicate therefore has exact-set differences that are direct signed
        # case means; no independently sampled second-method cases enter.
        reference = np.zeros((55, 17), dtype=np.uint8)
        m2 = np.zeros_like(reference)
        m3 = np.zeros_like(reference)
        m4 = np.zeros_like(reference)
        m2[:27, 0] = 1
        m3[27:, 0] = 1
        rows = paired.paired_bootstrap_rows(
            _data(reference, m2, m3, m4), n_resamples=25, seed=3
        )
        exact = next(
            row
            for row in rows
            if row["comparison"] == "M3 - M2"
            and row["metric"] == "Exact-set accuracy"
        )
        self.assertAlmostEqual(exact["point_difference"], -1 / 55)
        self.assertEqual(
            exact["bootstrap_method"], "PAIRED_CASE_RESAMPLING_PERCENTILE_LINEAR"
        )
        self.assertEqual(exact["resampling_unit"], "CASE_WITH_ALL_17_LABELS")

    def test_fixed_macro_label_set_is_retained_in_zero_support_resamples(self) -> None:
        reference = np.zeros((55, 17), dtype=np.uint8)
        # Every ontology dimension has positive support in the full universe,
        # but rare labels disappear from some small bootstrap resamples.
        for label_index in range(17):
            reference[label_index, label_index] = 1
        prediction = reference.copy()
        rows = paired.paired_bootstrap_rows(
            _data(reference, prediction, prediction, prediction),
            n_resamples=10,
            seed=9,
        )
        self.assertEqual({row["macro_label_count"] for row in rows}, {17})
        self.assertTrue(
            all(row["macro_label_ids_json"].count("ACT_") == 5 for row in rows)
        )

    def test_fails_closed_on_wrong_membership_n(self) -> None:
        reference = np.zeros((54, 17), dtype=np.uint8)
        with self.assertRaisesRegex(paired.PairedBootstrapError, "expected 55"):
            paired.paired_bootstrap_rows(
                _data(reference, reference, reference, reference),
                n_resamples=10,
                seed=1,
            )

    def test_fails_closed_on_prediction_shape_mismatch(self) -> None:
        reference = np.zeros((55, 17), dtype=np.uint8)
        short = np.zeros((54, 17), dtype=np.uint8)
        with self.assertRaisesRegex(paired.PairedBootstrapError, "shape differs"):
            paired.paired_bootstrap_rows(
                _data(reference, reference, short, reference),
                n_resamples=10,
                seed=1,
            )

    def test_paper_final_generation_rejects_protocol_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(paired.PairedBootstrapError, "exactly 1,000"):
                paired.generate(output_path=Path(directory) / "out.csv", n_resamples=999)


if __name__ == "__main__":
    unittest.main()
