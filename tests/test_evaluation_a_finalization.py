"""Focused regression tests for the paper-facing Evaluation A finalizer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "src/experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))


def load_finalizer():
    path = EXPERIMENTS_DIR / "16_finalize_evaluation_a.py"
    spec = importlib.util.spec_from_file_location("evaluation_a_finalizer_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Evaluation A finalizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FINALIZER = load_finalizer()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def hash_tree(paths: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


class EvaluationAFinalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.analysis = cls.root / "analysis"
        cls.figures = cls.root / "figures"
        cls.docs = cls.root / "docs"
        cls.metrics = REPO_ROOT / "outputs/metrics"
        cls.source_paths = [
            cls.metrics / "a1/amp_primary_results.csv",
            cls.metrics / "a2/amp_primary_results.csv",
            cls.metrics / "a1/amp_per_label.csv",
            cls.metrics / "a2/amp_per_label.csv",
            cls.metrics / "a1/amp_case_level_errors.csv",
            cls.metrics / "a2/amp_case_level_errors.csv",
            cls.metrics / "amp_a1_to_a2_deltas.csv",
            cls.metrics / "a2/amp_per_fold.csv",
            cls.metrics / "a2/amp_per_jurisdiction.csv",
            cls.metrics / "a1/amp_llm_api_usage.csv",
            cls.metrics / "a2/amp_llm_api_usage.csv",
        ]
        for path in cls.source_paths:
            if not path.is_file():
                raise RuntimeError(f"Canonical Evaluation A fixture is missing: {path}")
        cls.source_hashes_before = hash_tree(cls.source_paths)
        cls.result = FINALIZER.generate(
            metrics_root=cls.metrics,
            analysis_dir=cls.analysis,
            figure_dir=cls.figures,
            docs_dir=cls.docs,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_generates_exact_contract_without_mutating_canonical_inputs(self) -> None:
        self.assertEqual(self.result["status"], "EVALUATION_A_PAPER_ANALYSIS_READY")
        self.assertEqual(self.result["tables"]["a1_main_comparison.csv"], 4)
        self.assertEqual(self.result["tables"]["a2_main_comparison.csv"], 4)
        self.assertEqual(self.result["tables"]["amp_family_level_metrics.csv"], 24)
        self.assertEqual(self.result["tables"]["prediction_breadth_summary.csv"], 8)
        self.assertEqual(self.result["tables"]["a2_fold_summary.csv"], 12)
        self.assertEqual(self.result["tables"]["a2_jurisdiction_summary.csv"], 72)
        self.assertEqual(hash_tree(self.source_paths), self.source_hashes_before)
        self.assertEqual(
            sorted(path.name for path in self.figures.iterdir()),
            sorted(FINALIZER.FIGURE_FILENAMES),
        )
        for path in self.figures.iterdir():
            self.assertIn("<svg", path.read_text(encoding="utf-8")[:1000])

    def test_main_tables_copy_canonical_values_and_frozen_counts(self) -> None:
        a1 = read_csv(self.analysis / "a1_main_comparison.csv")
        a2 = read_csv(self.analysis / "a2_main_comparison.csv")
        source_a1 = {row["method"]: row for row in read_csv(self.metrics / "a1/amp_primary_results.csv")}
        source_a2 = {row["method"]: row for row in read_csv(self.metrics / "a2/amp_primary_results.csv")}
        self.assertEqual([row["method"] for row in a1], list(FINALIZER.EXPECTED_METHODS))
        self.assertEqual([row["method"] for row in a2], list(FINALIZER.EXPECTED_METHODS))
        for row in a1:
            self.assertEqual(int(row["n"]), 253)
            self.assertEqual(int(row["macro_supported_label_count"]), 17)
            self.assertEqual(row["macro_f1"], source_a1[row["method"]]["macro_f1"])
            self.assertEqual(row["micro_f1"], source_a1[row["method"]]["micro_f1"])
        for row in a2:
            self.assertEqual(int(row["n"]), 861)
            self.assertEqual(int(row["macro_supported_label_count"]), 16)
            self.assertEqual(row["macro_f1"], source_a2[row["method"]]["pooled_ood_macro_f1"])
            self.assertEqual(row["micro_f1"], source_a2[row["method"]]["pooled_micro_f1"])

    def test_zero_support_and_rare_label_rules_remain_explicit(self) -> None:
        rows = read_csv(self.analysis / "rare_label_sensitivity.csv")
        support = {
            (row["evaluation"], row["label_id"]): row
            for row in rows
            if row["record_type"] == "LABEL_SUPPORT"
        }
        self.assertEqual(len(support), 34)
        self.assertEqual(
            support[("A1", FINALIZER.ORGAN_REMOVAL_LABEL)]["label_support"], "2"
        )
        self.assertEqual(
            support[("A1", FINALIZER.ORGAN_REMOVAL_LABEL)]["included_in_official_macro_f1"],
            "1",
        )
        self.assertEqual(
            support[("A2", FINALIZER.ORGAN_REMOVAL_LABEL)]["label_support"], "0"
        )
        self.assertEqual(
            support[("A2", FINALIZER.ORGAN_REMOVAL_LABEL)]["included_in_official_macro_f1"],
            "0",
        )
        sensitivity = [row for row in rows if row["record_type"] == "MACRO_SENSITIVITY"]
        self.assertEqual(len(sensitivity), 8)
        for row in sensitivity:
            if row["evaluation"] == "A2":
                self.assertAlmostEqual(
                    float(row["official_macro_f1"]), float(row["diagnostic_macro_f1"])
                )
                self.assertEqual(float(row["delta_diagnostic_minus_official"]), 0.0)
        family = read_csv(self.analysis / "amp_family_level_metrics.csv")
        a2_purpose = [
            row for row in family if row["evaluation"] == "A2" and row["family"] == "Purpose"
        ]
        self.assertEqual(len(a2_purpose), 4)
        self.assertTrue(all(row["supported_label_count"] == "5" for row in a2_purpose))
        contrasts = read_csv(self.analysis / "m3_vs_m4_per_label_f1.csv")
        organ_a2 = next(
            row
            for row in contrasts
            if row["evaluation"] == "A2" and row["label_id"] == FINALIZER.ORGAN_REMOVAL_LABEL
        )
        self.assertEqual(organ_a2["m3_f1"], "N/A")
        self.assertEqual(organ_a2["m4_f1"], "N/A")
        self.assertEqual(organ_a2["delta_f1_m4_minus_m3"], "N/A")

    def test_outputs_are_byte_deterministic_and_checkable(self) -> None:
        paths = list(self.analysis.iterdir()) + list(self.figures.iterdir()) + list(self.docs.iterdir())
        first_hashes = hash_tree(paths)
        FINALIZER.generate(
            metrics_root=self.metrics,
            analysis_dir=self.analysis,
            figure_dir=self.figures,
            docs_dir=self.docs,
        )
        self.assertEqual(hash_tree(paths), first_hashes)
        checked = FINALIZER.generate(
            metrics_root=self.metrics,
            analysis_dir=self.analysis,
            figure_dir=self.figures,
            docs_dir=self.docs,
            check=True,
        )
        self.assertEqual(checked["mode"], "CHECK")

    def test_report_uses_required_scientific_boundaries(self) -> None:
        report = (self.docs / "evaluation_a_final_report.md").read_text(encoding="utf-8")
        note = (self.docs / "evaluation_a_rare_label_sensitivity.md").read_text(encoding="utf-8")
        self.assertIn("silver-reference labels", report)
        self.assertIn("human-grounded gold", report)
        self.assertIn("not absolute factual correctness", report)
        self.assertIn("No statistical-significance test was designated", report)
        self.assertNotIn("ground truth", report.lower())
        self.assertIn("not a replacement for the canonical metric", note.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
