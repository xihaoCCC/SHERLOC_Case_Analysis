"""Focused offline tests for the deterministic paper-final package builder."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sherloc_paper_mpl"))
REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src/experiments/25_build_paper_final_package.py"


def load_module():
    spec = importlib.util.spec_from_file_location("paper_final_package_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load paper-final package generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PACKAGE = load_module()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def auxiliary_rows(n_by_target=None):
    n_by_target = n_by_target or {target: "55" for target in PACKAGE.AUXILIARY_METRICS}
    rows = []
    for target, metrics in PACKAGE.AUXILIARY_METRICS.items():
        for index, metric in enumerate(metrics, start=1):
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "value": str(0.5 + index / 100),
                    "n": n_by_target[target],
                    "support_json": "{}",
                    "confusion_matrix_json": "{}",
                    "model": "gpt-5.6-luna",
                    "prompt_version": "aux-zero-shot-v1",
                }
            )
    return rows


def paired_rows(resamples="1000"):
    rows = []
    n_by_evaluation = {"A1": "253", "A2": "861", "A3": "55"}
    reference = {
        "A1": "SHERLOC silver reference",
        "A2": "SHERLOC silver reference",
        "A3": "human-grounded narrative reference",
    }
    for evaluation in PACKAGE.EVALUATIONS:
        for comparison, first, second in PACKAGE.PAIRED_COMPARISONS:
            for metric in PACKAGE.PAIRED_METRICS:
                rows.append(
                    {
                        "evaluation": evaluation,
                        "reference_type": reference[evaluation],
                        "comparison": comparison,
                        "first_method": first,
                        "second_method": second,
                        "metric": metric,
                        "n": n_by_evaluation[evaluation],
                        "point_difference": "0.01",
                        "ci_low": "-0.01",
                        "ci_high": "0.03",
                        "confidence_level": "0.95",
                        "bootstrap_resamples": resamples,
                        "seed": "20260811",
                        "resampling_unit": "case",
                        "bootstrap_method": "paired_percentile",
                        "macro_label_count": "17" if evaluation != "A2" else "16",
                        "macro_label_ids_json": "[]",
                        "ci_excludes_zero": "false",
                    }
                )
    return rows


class PaperFinalPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a1 = read_csv(REPO_ROOT / "outputs/metrics/a1/amp_primary_results.csv")
        cls.a2 = read_csv(REPO_ROOT / "outputs/metrics/a2/amp_primary_results.csv")
        cls.a3 = read_csv(
            REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_main_results.csv"
        )
        cls.silver = read_csv(
            REPO_ROOT / "outputs/analysis/evaluation_b/silver_vs_human_summary.csv"
        )
        cls.breadth = read_csv(
            REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_prediction_breadth.csv"
        )
        cls.abstain = read_csv(
            REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_abstain_results.csv"
        )
        cls.reference_comparison = read_csv(
            REPO_ROOT
            / "outputs/analysis/evaluation_b/model_silver_vs_human_metric_comparison.csv"
        )

    def test_master_is_12_rows_and_copies_canonical_strings_exactly(self):
        PACKAGE._validate_primary_sources(self.a1, self.a2, self.a3)
        master = PACKAGE.build_master_rows(self.a1, self.a2, self.a3)
        self.assertEqual(len(master), 12)
        self.assertEqual(
            [(row["evaluation"], row["method"]) for row in master],
            [(evaluation, method) for evaluation in PACKAGE.EVALUATIONS for method in PACKAGE.METHODS],
        )
        a1_m3 = next(row for row in master if row["evaluation"] == "A1" and row["method"] == "M3")
        source_a1_m3 = next(row for row in self.a1 if row["method"] == "M3")
        self.assertEqual(a1_m3["macro_f1"], source_a1_m3["macro_f1"])
        self.assertEqual(a1_m3["macro_f1_ci_low"], source_a1_m3["macro_f1_ci_lower"])
        a2_m4 = next(row for row in master if row["evaluation"] == "A2" and row["method"] == "M4")
        source_a2_m4 = next(row for row in self.a2 if row["method"] == "M4")
        self.assertEqual(a2_m4["purpose_cpmr"], source_a2_m4["pooled_purpose_cpmr"])
        a3_m2 = next(row for row in master if row["evaluation"] == "A3" and row["method"] == "M2")
        source_a3_m2 = next(row for row in self.a3 if row["method"] == "M2")
        self.assertEqual(a3_m2["example_jaccard"], source_a3_m2["jaccard"])

    def test_compact_tables_have_frozen_membership(self):
        master = PACKAGE.build_master_rows(self.a1, self.a2, self.a3)
        main = PACKAGE.build_main_paper_rows(master)
        self.assertEqual(len(main), 12)
        self.assertEqual(tuple(main[0]), PACKAGE.MAIN_PAPER_FIELDS)
        silver = PACKAGE.build_silver_human_rows(self.silver)
        self.assertEqual([row["family"] for row in silver], list(PACKAGE.FAMILIES))
        behavior = PACKAGE.build_behavior_rows(self.a3, self.breadth, self.abstain)
        self.assertEqual([row["method"] for row in behavior], list(PACKAGE.METHODS))
        self.assertEqual({row["N"] for row in behavior}, {"55"})
        self.assertEqual({row["abstain_n"] for row in behavior}, {"6"})

    def test_auxiliary_compact_copies_target_specific_n(self):
        n_by_target = {
            "GEOGRAPHIC_FORM": "55",
            "VICTIM_MULTIPLICITY": "54",
            "CHILD_INVOLVEMENT": "53",
            "ORGANIZED_CRIMINAL_GROUP": "52",
        }
        compact = PACKAGE.build_auxiliary_rows(auxiliary_rows(n_by_target))
        self.assertEqual({row["target"]: row["N"] for row in compact}, n_by_target)
        geographic = compact[0]
        self.assertNotEqual(geographic["macro_f1"], "")
        self.assertNotEqual(geographic["example_jaccard"], "")
        self.assertEqual(geographic["accuracy"], "")

    def test_auxiliary_contract_rejects_missing_metric(self):
        rows = auxiliary_rows()
        rows.pop()
        with self.assertRaisesRegex(PACKAGE.PaperPackageError, "contract mismatch"):
            PACKAGE.build_auxiliary_rows(rows)

    def test_paired_contract_is_exactly_1000_resamples(self):
        rows = paired_rows("1000")
        PACKAGE.validate_paired_rows(rows)
        self.assertEqual(len(rows), 63)
        with self.assertRaisesRegex(PACKAGE.PaperPackageError, "protocol mismatch"):
            PACKAGE.validate_paired_rows(paired_rows("10000"))

    def test_documents_preserve_scientific_boundaries(self):
        claims = PACKAGE.render_claim_map()
        self.assertEqual(sum(f"C{index:02d}" in claims for index in range(1, 14)), 13)
        for phrase in (
            "single-reviewer human-grounded narrative reference",
            "Silver-only labels are not automatically errors",
            "no full-corpus silver auxiliary benchmark",
            "no unplanned p-value",
        ):
            self.assertIn(phrase.lower(), claims.lower())
        plan = PACKAGE.render_nllp_plan()
        self.assertIn("8 pages", plan)
        self.assertIn("planning scaffold only", plan)
        self.assertIn("Figure PF1", plan)
        self.assertIn("Figure PF3", plan)
        readme = PACKAGE.render_readme()
        self.assertIn("does not recompute benchmark metrics", readme)
        self.assertIn("refuses to overwrite", readme)
        figure_readme = PACKAGE.render_figure_readme()
        self.assertIn("figure_pf1_core_performance.svg", figure_readme)
        self.assertIn("figure_pf2_cpmr_by_family.svg", figure_readme)
        self.assertIn("figure_pf3_silver_human_reference_shift.svg", figure_readme)
        self.assertIn("move to appendix first", figure_readme)

    def test_three_figures_are_deterministic_svg(self):
        master = PACKAGE.build_master_rows(self.a1, self.a2, self.a3)
        first = PACKAGE._figure_bytes(master, self.silver, self.reference_comparison)
        second = PACKAGE._figure_bytes(master, self.silver, self.reference_comparison)
        self.assertEqual(set(first), set(PACKAGE.FIGURE_NAMES))
        self.assertEqual(first, second)
        self.assertTrue(all(payload.lstrip().startswith(b"<?xml") for payload in first.values()))

    def test_baseline_validator_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "artifact.txt"
            target.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            aggregate = hashlib.sha256()
            aggregate.update(b"artifact.txt\0" + digest.encode() + b"\0" + str(target.stat().st_size).encode() + b"\n")
            baseline = {
                "schema_version": "test",
                "scopes": {
                    "test": {
                        "file_count": 1,
                        "aggregate_sha256": aggregate.hexdigest(),
                        "files": [
                            {
                                "path": "artifact.txt",
                                "sha256": digest,
                                "size": target.stat().st_size,
                            }
                        ],
                    }
                },
            }
            status, hashes = PACKAGE.validate_evaluation_a_baseline(root, baseline)
            self.assertEqual(status["status"], "PASS_UNCHANGED")
            self.assertEqual(hashes["artifact.txt"], digest)
            target.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(PACKAGE.PaperPackageError, "baseline mismatch"):
                PACKAGE.validate_evaluation_a_baseline(root, baseline)

    def test_write_is_idempotent_and_never_overwrites_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                root / "one.csv": b"a,b\n1,2\n",
                root / "prewriting_freeze_manifest.json": b"{}\n",
            }
            first = PACKAGE.write_package(artifacts)
            second = PACKAGE.write_package(artifacts)
            self.assertTrue(all(row["status"] == "UNCHANGED" for row in first + second))
            (root / "one.csv").write_text("different\n", encoding="utf-8")
            with self.assertRaisesRegex(PACKAGE.PaperPackageError, "Refusing to overwrite"):
                PACKAGE.write_package(artifacts)

    def test_csv_serialization_is_deterministic(self):
        rows = [{"a": "1.2300", "b": "text"}]
        first = PACKAGE._csv_bytes(rows, ("a", "b"))
        second = PACKAGE._csv_bytes(rows, ("a", "b"))
        self.assertEqual(first, second)
        self.assertEqual(first, b"a,b\n1.2300,text\n")


if __name__ == "__main__":
    unittest.main()
