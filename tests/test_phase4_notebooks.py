"""Tests for the generated Phase-4 reporting notebooks."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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


def all_source(notebook: dict) -> str:
    return "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])


class Phase4NotebookGeneratorTest(unittest.TestCase):
    def test_primary_notebook_set_and_valid_structure(self) -> None:
        notebooks = GENERATOR.build_notebooks()
        self.assertEqual(set(notebooks), set(GENERATOR.PRIMARY_NOTEBOOK_NAMES))
        for name, notebook in notebooks.items():
            self.assertEqual(notebook["nbformat"], 4, name)
            self.assertGreaterEqual(notebook["nbformat_minor"], 5, name)
            self.assertEqual(
                notebook["metadata"]["sherloc_reporting"]["metric_source"],
                "src/experiments/11_evaluate_amp.py",
            )
            ids = [cell["id"] for cell in notebook["cells"]]
            self.assertEqual(len(ids), len(set(ids)), name)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

    def test_notebooks_load_canonical_outputs_without_metric_reimplementation(self) -> None:
        forbidden = (
            "sklearn.metrics",
            "compute_amp_metrics",
            "f1_score(",
            "jaccard_score(",
            "precision_recall_fscore_support(",
            "percentile_bootstrap_confidence_intervals",
        )
        for name, notebook in GENERATOR.build_notebooks().items():
            source = all_source(notebook)
            self.assertIn('METRICS_ROOT = REPO_ROOT / "outputs/metrics"', source, name)
            self.assertIn("11_evaluate_amp.py", source, name)
            self.assertIn("silver reference", source.lower(), name)
            self.assertIn("completion gate", source.lower(), name)
            self.assertNotIn("stage2_m1", source, name)
            self.assertNotIn("stage3_m1", source, name)
            for token in forbidden:
                self.assertNotIn(token, source, f"{name}: {token}")

    def test_required_reporting_sections_and_a2_zero_support_rule(self) -> None:
        notebooks = GENERATOR.build_notebooks()
        a1 = all_source(notebooks["07_a1_amp_results.ipynb"])
        for heading in (
            "Frozen experiment metadata",
            "A1 split composition",
            "Canonical M1–M4 A1 comparison",
            "Canonical per-label",
            "Canonical bootstrap",
            "Fixed 0.50-threshold sensitivity",
        ):
            self.assertIn(heading, a1)

        a2 = all_source(notebooks["08_a2_amp_results.ipynb"])
        for text in (
            "held-out jurisdictions",
            "Canonical per-fold results",
            "Canonical pooled OOD results",
            "Canonical per-jurisdiction results",
            "Canonical A1 → A2 aggregate deltas",
            "PURPOSE_REMOVAL_OF_ORGANS",
            "16 supported labels",
        ):
            self.assertIn(text, a2)

        errors = all_source(notebooks["09_amp_error_analysis.ipynb"])
        for text in (
            "false_positive_labels_json",
            "false_negative_labels_json",
            "TRUNCATED_M2_ONLY",
            "Prediction disagreements among M1–M4",
            "M3 versus M4",
            "Rare-label inspection",
            "post-hoc exploratory analysis",
        ):
            self.assertIn(text, errors)

    def test_optional_auxiliary_notebook_is_explicit_and_separate(self) -> None:
        without = GENERATOR.build_notebooks(include_auxiliary=False)
        self.assertNotIn(GENERATOR.AUXILIARY_NOTEBOOK_NAME, without)
        with_auxiliary = GENERATOR.build_notebooks(include_auxiliary=True)
        self.assertIn(GENERATOR.AUXILIARY_NOTEBOOK_NAME, with_auxiliary)
        source = all_source(with_auxiliary[GENERATOR.AUXILIARY_NOTEBOOK_NAME])
        self.assertIn('METRICS_ROOT / "auxiliary"', source)
        self.assertIn("exploratory silver-reference", source.lower())
        self.assertIn("Primary M1-M4 A1/A2 AMP completion", source)

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
