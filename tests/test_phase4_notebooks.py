"""Tests for deterministic Evaluation A views and the Evaluation B template."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "src/experiments/12_generate_analysis_notebooks.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("phase4_notebook_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load notebook generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_generator()
ANALYSIS_FIGURES = (
    "figure_b1_human_grounded_core_performance.svg",
    "figure_b2_human_grounded_cpmr.svg",
    "figure_b3_silver_vs_human_model_scores.svg",
    "figure_b4_silver_human_label_proportions.svg",
)


def all_source(notebook: dict) -> str:
    return "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])


class Phase4NotebookGeneratorTest(unittest.TestCase):
    def test_default_notebook_set_and_unexecuted_structure(self) -> None:
        notebooks = GENERATOR.build_notebooks()
        expected = {
            *GENERATOR.EVALUATION_A_NOTEBOOK_NAMES,
            GENERATOR.HUMAN_GROUNDED_NOTEBOOK_NAME,
        }
        self.assertEqual(set(notebooks), expected)
        for name, notebook in notebooks.items():
            self.assertEqual(notebook["nbformat"], 4, name)
            self.assertGreaterEqual(notebook["nbformat_minor"], 5, name)
            reporting = notebook["metadata"]["sherloc_reporting"]
            self.assertEqual(
                reporting["generator"],
                "src/experiments/12_generate_analysis_notebooks.py",
            )
            self.assertFalse(reporting["cells_executed_by_generator"])
            ids = [cell["id"] for cell in notebook["cells"]]
            self.assertEqual(len(ids), len(set(ids)), name)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

    def test_evaluation_a_notebooks_load_finalized_outputs_only(self) -> None:
        notebooks = GENERATOR.build_notebooks()
        source = "\n".join(
            all_source(notebooks[name]) for name in GENERATOR.EVALUATION_A_NOTEBOOK_NAMES
        )
        self.assertIn('ANALYSIS_ROOT = REPO_ROOT / "outputs/analysis/evaluation_a"', source)
        self.assertIn('FIGURE_ROOT = REPO_ROOT / "outputs/figures/evaluation_a"', source)
        self.assertIn("src/experiments/16_finalize_evaluation_a.py", str(
            notebooks[GENERATOR.EVALUATION_A_NOTEBOOK_NAMES[0]]["metadata"]
        ))
        for table in GENERATOR.ANALYSIS_TABLES:
            self.assertIn(table, source)
        for figure in GENERATOR.CORE_FIGURES:
            self.assertIn(figure, source)

        forbidden = (
            "sklearn.metrics",
            "compute_amp_metrics",
            "compute_amp_cpmr",
            "contained_partial_match(",
            "contained_recall(",
            "f1_score(",
            "jaccard_score(",
            "precision_recall_fscore_support(",
            "percentile_bootstrap_confidence_intervals",
            "bootstrap_resamples",
            "11_evaluate_amp.py --",
            "10_run_llm_amp.py",
        )
        for token in forbidden:
            self.assertNotIn(token, source, token)

    def test_scientific_boundaries_are_explicit(self) -> None:
        notebooks = GENERATOR.build_notebooks()
        a1 = all_source(notebooks["07_a1_amp_results.ipynb"])
        self.assertIn("silver-reference labels", a1)
        self.assertIn("does not establish absolute factual correctness", a1)
        self.assertIn("descriptive sensitivity analysis", a1)
        self.assertIn("does not replace the frozen official Macro-F1", a1)

        a2 = all_source(notebooks["08_a2_amp_results.ipynb"])
        for phrase in (
            "PURPOSE_REMOVAL_OF_ORGANS",
            "zero positive A2 silver-reference support",
            "16 supported labels",
            "Do not rank jurisdictions",
            "statistical significance was not tested",
        ):
            self.assertIn(phrase, a2)

        errors = all_source(notebooks["09_amp_error_analysis.ipynb"])
        for phrase in (
            "false_positive_labels_json",
            "false_negative_labels_json",
            "post-hoc exploratory analysis",
            "not automatically a factual error",
            "Do not use this notebook to tune or rerun Evaluation A",
        ):
            self.assertIn(phrase, errors)

    def test_human_grounded_notebook_is_thin_graceful_and_single_reviewer(self) -> None:
        notebook = GENERATOR.build_notebooks()[GENERATOR.HUMAN_GROUNDED_NOTEBOOK_NAME]
        source = all_source(notebook)
        for phrase in (
            "NOT YET AVAILABLE",
            "single-reviewer human-grounded narrative reference",
            "human_annotation_source_manifest.json",
            "human_annotation_qc_summary.json",
            "human_annotation_qc_report.csv",
            "eval_b_training_exclusion_audit.csv",
            "silver_vs_human_summary.csv",
            "eval_b_main_results.csv",
            "eval_b_bootstrap_cis.csv",
            "eval_b_family_results.csv",
            "eval_b_per_label_results.csv",
            "eval_b_abstain_results.csv",
            "eval_b_prediction_breadth.csv",
            "model_silver_vs_human_metric_comparison.csv",
            "human_grounded_case_level_errors.csv",
            "validate_analysis_manifest",
            "inputs_sha256",
            "outputs_sha256",
            "expected_count",
            "reviewer-to-reviewer reliability is unavailable",
        ):
            self.assertIn(phrase, source)
        for figure in ANALYSIS_FIGURES:
            self.assertIn(figure, source)
        for forbidden in (
            "reliability_sample_100_reference_key",
            "reviewer_annotation_template.csv",
            "compute_reviewer_agreement",
            "build_disagreement_queue",
            "build_human_gold",
            "qc_annotations",
            "kappa",
            "two-reviewer gold",
            "adjudicated gold",
            "compute_amp_metrics",
            "compute_amp_cpmr",
            "bootstrap_intervals",
            "select_reliability_subset",
            "10_run_llm_amp.py",
            "openai",
            "responses.create",
        ):
            self.assertNotIn(forbidden.lower(), source.lower())

    def test_human_grounded_manifest_gate_binds_inputs_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "src/experiments/18_evaluate_evaluation_b.py"
            marker.parent.mkdir(parents=True)
            marker.write_text("# marker\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"SHERLOC_REPO_ROOT": str(root)}):
                namespace: dict = {}
                exec(GENERATOR.HUMAN_SETUP, namespace)
                self.assertFalse(namespace["CANONICAL_ANALYSIS_READY"])

                for relative in (
                    namespace["REQUIRED_ANALYSIS_INPUTS"]
                    | namespace["EXPECTED_ANALYSIS_OUTPUTS"]
                ):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"artifact:{relative}\n", encoding="utf-8")
                inputs = {
                    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                    for relative in namespace["REQUIRED_ANALYSIS_INPUTS"]
                }
                outputs = {
                    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                    for relative in namespace["EXPECTED_ANALYSIS_OUTPUTS"]
                }
                manifest_path = (
                    root
                    / "outputs/analysis/evaluation_b/evaluation_b_analysis_manifest.json"
                )
                manifest_path.write_text(
                    json.dumps(
                        {
                            "status": "COMPLETE",
                            "inputs_sha256": inputs,
                            "outputs_sha256": outputs,
                        }
                    ),
                    encoding="utf-8",
                )
                complete_namespace: dict = {}
                exec(GENERATOR.HUMAN_SETUP, complete_namespace)
                self.assertTrue(complete_namespace["CANONICAL_ANALYSIS_READY"])

                tampered = root / sorted(namespace["EXPECTED_ANALYSIS_OUTPUTS"])[0]
                tampered.write_text("tampered\n", encoding="utf-8")
                stale_namespace: dict = {}
                exec(GENERATOR.HUMAN_SETUP, stale_namespace)
                self.assertFalse(stale_namespace["CANONICAL_ANALYSIS_READY"])
                self.assertIn("hash mismatch", stale_namespace["analysis_gate_detail"])

    def test_optional_auxiliary_notebook_cannot_collide_with_evaluation_b(self) -> None:
        notebooks = GENERATOR.build_notebooks(include_auxiliary=True)
        self.assertIn("10_human_grounded_evaluation.ipynb", notebooks)
        self.assertIn("11_auxiliary_results.ipynb", notebooks)
        self.assertNotIn("10_auxiliary_results.ipynb", notebooks)
        auxiliary = all_source(notebooks["11_auxiliary_results.ipynb"])
        self.assertIn("NOT YET AVAILABLE", auxiliary)
        self.assertIn("No Geographic Form, Multiplicity, Sector, or Child experiment", auxiliary)

    def test_generation_is_deterministic_and_checkable(self) -> None:
        first = GENERATOR.build_notebooks()
        second = GENERATOR.build_notebooks()
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            diagnostics = GENERATOR.write_notebooks(output)
            self.assertTrue(all(item["status"] == "WRITTEN" for item in diagnostics))
            checked = GENERATOR.write_notebooks(output, check=True)
            self.assertTrue(all(item["status"] == "UNCHANGED" for item in checked))
            for name, expected in first.items():
                observed = json.loads((output / name).read_text(encoding="utf-8"))
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
