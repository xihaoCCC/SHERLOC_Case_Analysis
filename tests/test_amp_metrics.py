"""Focused regression tests for canonical Phase-4 AMP evaluation."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "src/experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from bootstrap import percentile_bootstrap_confidence_intervals  # noqa: E402
from metrics import (  # noqa: E402
    AMP_FAMILY_BY_LABEL,
    AMP_LABEL_IDS,
    ORGAN_REMOVAL_LABEL,
    MetricInputError,
    compute_amp_metrics,
    compute_case_errors,
    labels_to_indicator,
)


def load_evaluator_module():
    path = EXPERIMENTS_DIR / "11_evaluate_amp.py"
    spec = importlib.util.spec_from_file_location("evaluate_amp_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evaluator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_evaluator_module()


class AmpMetricsTest(unittest.TestCase):
    def test_frozen_17_label_schema(self) -> None:
        self.assertEqual(len(AMP_LABEL_IDS), 17)
        self.assertEqual(len(set(AMP_LABEL_IDS)), 17)
        self.assertEqual(
            [sum(AMP_FAMILY_BY_LABEL[label] == family for label in AMP_LABEL_IDS)
             for family in ("ACT", "MEANS", "PURPOSE")],
            [5, 6, 6],
        )
        self.assertEqual(AMP_LABEL_IDS[15], ORGAN_REMOVAL_LABEL)

    def test_known_aggregate_and_case_metrics(self) -> None:
        act_a, act_b = AMP_LABEL_IDS[:2]
        reference = labels_to_indicator([[act_a], [act_b], []])
        predicted = labels_to_indicator([[act_a], [act_a, act_b], []])
        result = compute_amp_metrics(
            reference,
            predicted,
            macro_label_ids=(act_a, act_b),
        )
        self.assertAlmostEqual(result["macro_f1"], 5 / 6)
        self.assertAlmostEqual(result["micro_f1"], 0.8)
        self.assertAlmostEqual(result["exact_set_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["example_jaccard"], 5 / 6)

        errors = compute_case_errors(reference, predicted)
        self.assertEqual(errors[1]["false_positive_labels"], [act_a])
        self.assertEqual(errors[2]["example_jaccard"], 1.0)

    def test_unknown_and_nonbinary_values_are_rejected(self) -> None:
        with self.assertRaises(MetricInputError):
            labels_to_indicator([["NOT_AN_AMP_LABEL"]])
        with self.assertRaises(MetricInputError):
            compute_amp_metrics(np.asarray([[0] * 16]), np.asarray([[0] * 16]))
        with self.assertRaises(MetricInputError):
            compute_amp_metrics(np.asarray([[2] * 17]), np.asarray([[0] * 17]))

    def test_a2_zero_support_organ_is_na_but_still_penalizes_set_metrics(self) -> None:
        supported = tuple(label for label in AMP_LABEL_IDS if label != ORGAN_REMOVAL_LABEL)
        reference = labels_to_indicator([supported])
        predicted = labels_to_indicator([AMP_LABEL_IDS])
        result = compute_amp_metrics(
            reference,
            predicted,
            macro_label_ids=supported,
        )
        organ = next(
            row for row in result["per_label"] if row["label_id"] == ORGAN_REMOVAL_LABEL
        )
        self.assertEqual(organ["support"], 0)
        self.assertIsNone(organ["precision"])
        self.assertIsNone(organ["recall"])
        self.assertIsNone(organ["f1"])
        self.assertEqual(organ["status"], "NO_REFERENCE_SUPPORT")
        self.assertFalse(organ["included_in_macro_f1"])
        self.assertEqual(result["macro_label_count"], 16)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["micro_f1"], 32 / 33)
        self.assertEqual(result["exact_set_accuracy"], 0.0)
        self.assertAlmostEqual(result["example_jaccard"], 16 / 17)

    def test_bootstrap_is_deterministic_and_case_resampled(self) -> None:
        reference = labels_to_indicator(
            [[AMP_LABEL_IDS[0]], [AMP_LABEL_IDS[1]], [], [AMP_LABEL_IDS[0]]]
        )
        predicted = labels_to_indicator(
            [[AMP_LABEL_IDS[0]], [], [], [AMP_LABEL_IDS[0], AMP_LABEL_IDS[1]]]
        )
        kwargs = {
            "macro_label_ids": AMP_LABEL_IDS[:2],
            "n_resamples": 100,
            "seed": 8123,
        }
        first = percentile_bootstrap_confidence_intervals(reference, predicted, **kwargs)
        second = percentile_bootstrap_confidence_intervals(reference, predicted, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {
            "macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard"
        })
        for interval in first.values():
            self.assertLessEqual(interval["ci_lower"], interval["estimate"])
            self.assertLessEqual(interval["estimate"], interval["ci_upper"])
            self.assertEqual(interval["n_resamples"], 100)


class AmpEvaluatorTest(unittest.TestCase):
    def _record(
        self,
        *,
        evaluation: str,
        rank: int,
        labels: tuple[str, ...],
        predicted: tuple[str, ...],
        fold: int | None = None,
        variant: str = "PRIMARY",
        method: str = "M1",
    ):
        return EVALUATOR.PredictionRecord(
            source_path=Path("synthetic.jsonl"),
            source_row=rank,
            method=method,
            evaluation=evaluation,
            fold=fold,
            prediction_variant=variant,
            search_rank=rank,
            case_id=str(rank),
            canonical_url=f"https://example.test/{rank}",
            jurisdiction=f"Jurisdiction {fold or 0}",
            fact_summary=f"Synthetic Fact Summary {rank}.",
            silver_reference_labels=labels,
            predicted_labels=predicted,
            truncated_input=False,
        )

    def test_loader_accepts_canonical_m1_schema_and_fixed_threshold(self) -> None:
        row = {
            "method_id": "M1",
            "evaluation": "A1",
            "search_rank": 7,
            "canonical_url": "https://example.test/7",
            "jurisdiction": "Example",
            "split": "TEST",
            "fact_summary": "A test narrative.",
            "silver_reference_labels": [AMP_LABEL_IDS[0]],
            "predicted_labels": [AMP_LABEL_IDS[0]],
            "predicted_labels_0_50": [],
            "truncated_input": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a1_test_predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = EVALUATOR.load_prediction_files([path])
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record.prediction_variant for record in records},
            {"PRIMARY", "THRESHOLD_0_50"},
        )

    def test_final_completion_gate_rejects_partial_method_set(self) -> None:
        records = [
            self._record(
                evaluation="A1", rank=1, labels=(AMP_LABEL_IDS[0],), predicted=()
            ),
            self._record(
                evaluation="A2",
                fold=1,
                rank=2,
                labels=(AMP_LABEL_IDS[0],),
                predicted=(),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EVALUATOR.EvaluationError):
                EVALUATOR.evaluate_predictions(
                    records,
                    output_root=Path(directory),
                    n_resamples=5,
                    require_complete_primary=True,
                )

    def test_evaluator_writes_a1_a2_and_zero_support_na(self) -> None:
        records = [
            self._record(
                evaluation="A1",
                rank=1,
                labels=AMP_LABEL_IDS,
                predicted=AMP_LABEL_IDS,
            )
        ]
        # Across the pooled A2 sample all labels except Organ Removal have
        # positive silver-reference support.  A false-positive Organ prediction
        # must affect micro/set metrics but not 16-label macro-F1.
        supported = tuple(label for label in AMP_LABEL_IDS if label != ORGAN_REMOVAL_LABEL)
        for fold in (1, 2, 3):
            labels = supported if fold == 1 else (AMP_LABEL_IDS[fold - 1],)
            predicted = (
                tuple(list(labels) + [ORGAN_REMOVAL_LABEL]) if fold == 1 else labels
            )
            records.append(
                self._record(
                    evaluation="A2",
                    rank=10 + fold,
                    fold=fold,
                    labels=labels,
                    predicted=predicted,
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = EVALUATOR.evaluate_predictions(
                records,
                output_root=output,
                n_resamples=25,
                seed=55,
            )
            self.assertEqual(manifest["evaluations"]["A2"]["macro_label_count"], 16)
            self.assertEqual(
                manifest["evaluations"]["A2"]["organ_removal_rule"],
                "N/A_PER_LABEL_F1_AND_EXCLUDED_FROM_MACRO",
            )

            with (output / "a2/amp_per_label.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                per_label = list(csv.DictReader(handle))
            organ = next(row for row in per_label if row["label_id"] == ORGAN_REMOVAL_LABEL)
            self.assertEqual(organ["support"], "0")
            self.assertEqual(organ["f1"], "N/A")
            self.assertEqual(organ["included_in_macro_f1"], "0")

            with (output / "a2/amp_primary_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                primary = list(csv.DictReader(handle))[0]
            self.assertEqual(float(primary["pooled_ood_macro_f1"]), 1.0)
            self.assertLess(float(primary["pooled_micro_f1"]), 1.0)
            self.assertEqual(float(primary["pooled_act_cpmr"]), 1.0)
            self.assertAlmostEqual(float(primary["pooled_means_cpmr"]), 1 / 3)
            self.assertEqual(float(primary["pooled_purpose_cpmr"]), 0.0)

            with (output / "a1/amp_primary_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                a1_primary = list(csv.DictReader(handle))[0]
            self.assertEqual(float(a1_primary["act_cpmr"]), 1.0)
            self.assertEqual(float(a1_primary["means_cpmr"]), 1.0)
            self.assertEqual(float(a1_primary["purpose_cpmr"]), 1.0)

            with (output / "a1/amp_cpmr_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                a1_cpmr = list(csv.DictReader(handle))[0]
            self.assertEqual(float(a1_cpmr["act_cpmr"]), 1.0)
            self.assertEqual(float(a1_cpmr["means_cpmr"]), 1.0)
            self.assertEqual(float(a1_cpmr["purpose_cpmr"]), 1.0)

            with (output / "a2/amp_cpmr_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                a2_cpmr = list(csv.DictReader(handle))[0]
            self.assertEqual(a2_cpmr["scope"], "POOLED_OOD_TEST")
            self.assertEqual(a2_cpmr["purpose_mean_contained_recall"], "N/A")

            with (output / "a2/amp_case_level_errors.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                case_rows = list(csv.DictReader(handle))
            fold_two = next(row for row in case_rows if row["fold"] == "2")
            self.assertEqual(fold_two["act_cpmr"], "1")
            self.assertEqual(fold_two["means_cpmr"], "0")
            self.assertEqual(fold_two["means_contained_recall"], "N/A")

            with (output / "a2/amp_per_fold.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                fold_result = list(csv.DictReader(handle))[0]
            self.assertIn("act_mean_contained_recall", fold_result)
            with (output / "a2/amp_per_jurisdiction.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                jurisdiction_result = list(csv.DictReader(handle))[0]
            self.assertIn("purpose_cpmr", jurisdiction_result)

            self.assertTrue((output / "a2/amp_per_fold.csv").is_file())
            self.assertTrue((output / "a2/amp_per_jurisdiction.csv").is_file())
            self.assertTrue((output / "amp_a1_to_a2_deltas.csv").is_file())
            with (output / "amp_a1_to_a2_deltas.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                delta = list(csv.DictReader(handle))[0]
            self.assertIn("delta_act_cpmr_a2_minus_a1", delta)

    def test_evaluator_writes_unselected_m3_m4_a2_difference_tables(self) -> None:
        supported = tuple(
            label for label in AMP_LABEL_IDS if label != ORGAN_REMOVAL_LABEL
        )
        records = []
        for method in ("M3", "M4"):
            for fold in (1, 2, 3):
                labels = supported if fold == 1 else (AMP_LABEL_IDS[fold - 1],)
                records.append(
                    self._record(
                        method=method,
                        evaluation="A2",
                        rank=20 + fold,
                        fold=fold,
                        labels=labels,
                        predicted=labels if method == "M3" else (),
                    )
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            EVALUATOR.evaluate_predictions(
                records,
                output_root=output,
                n_resamples=10,
                seed=77,
            )
            with (output / "a2/amp_m3_vs_m4_aggregate_deltas.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                aggregate = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate), 7)
            macro = next(row for row in aggregate if row["metric"] == "macro_f1")
            act = next(row for row in aggregate if row["metric"] == "act_cpmr")
            self.assertLess(float(macro["delta_m4_minus_m3"]), 0.0)
            self.assertLess(float(act["delta_m4_minus_m3"]), 0.0)
            self.assertEqual(
                macro["significance_claim"], "NOT_TESTED_DO_NOT_INFER"
            )

            with (output / "a2/amp_m3_vs_m4_per_label_deltas.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                per_label = list(csv.DictReader(handle))
            self.assertEqual(len(per_label), 17)
            organ = next(
                row for row in per_label if row["label_id"] == ORGAN_REMOVAL_LABEL
            )
            self.assertEqual(organ["delta_f1_m4_minus_m3"], "N/A")


if __name__ == "__main__":
    unittest.main()
