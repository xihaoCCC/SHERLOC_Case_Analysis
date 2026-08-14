"""Focused protocol tests for the frozen M1 runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "src/experiments/08_run_m1_tfidf.py"
CONFIG_PATH = REPO_ROOT / "config/experiments/m1_tfidf_logreg_amp_v2.yaml"

SPEC = importlib.util.spec_from_file_location("run_m1_tfidf", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {RUNNER_PATH}")
M1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M1
SPEC.loader.exec_module(M1)


class M1RunnerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_phase4_config_and_threshold_grid_are_frozen(self) -> None:
        config_id = self.config["config_id"]
        self.assertEqual(config_id, "m1-tfidf-logreg-amp-v2")
        self.assertEqual(self.config["status"], "FROZEN_FOR_PHASE_4_EXECUTION")
        self.assertEqual(M1.sha256_file(CONFIG_PATH), M1.EXPECTED_CONFIG_SHA256)
        self.assertEqual(M1.THRESHOLD_GRID, tuple(np.arange(0.20, 0.801, 0.05).round(2)))
        self.assertEqual(self.config["thresholding"]["fixed_baseline"], 0.5)
        self.assertEqual(
            self.config["reproducibility"]["split_files"],
            [
                "data/splits/a1_iid_split_final_v1.csv",
                "data/splits/a2_jurisdiction_folds_final_v1.csv",
            ],
        )
        self.assertEqual(config_id, M1.EXPECTED_CONFIG_ID)

    def test_parameter_search_is_the_frozen_twelve_configurations(self) -> None:
        grid = M1.parameter_grid(self.config)
        self.assertEqual(len(grid), 12)
        self.assertEqual(
            grid[0],
            {
                "vectorizer.min_df": 1,
                "base_classifier.C": 0.25,
                "base_classifier.class_weight": None,
            },
        )
        self.assertEqual(
            grid[-1],
            {
                "vectorizer.min_df": 2,
                "base_classifier.C": 4.0,
                "base_classifier.class_weight": "balanced",
            },
        )

    def test_global_threshold_uses_validation_only_and_prefers_point_five(self) -> None:
        y_validation = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
        probabilities = np.asarray(
            [[0.70, 0.30], [0.60, 0.40], [0.40, 0.60], [0.30, 0.70]],
            dtype=np.float64,
        )
        threshold, rows = M1.select_global_threshold(y_validation, probabilities)
        self.assertEqual(threshold, 0.50)
        self.assertEqual(len(rows), 13)
        self.assertEqual(sum(row["threshold"] == threshold for row in rows), 1)

    def test_vectorizer_uses_word_unigrams_and_bigrams(self) -> None:
        parameters = M1.parameter_grid(self.config)[0]
        vectorizer, classifier = M1.build_pipeline(self.config, parameters)
        self.assertEqual(vectorizer.analyzer, "word")
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertTrue(vectorizer.sublinear_tf)
        self.assertEqual(classifier.n_jobs, 1)

    def test_fitted_metadata_is_json_serializable_with_numpy_vocab_indices(self) -> None:
        vectorizer = M1.TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(
            ["recruited victim for labour", "transported victim by force"]
        )
        classifier = M1.OneVsRestClassifier(
            M1.LogisticRegression(solver="liblinear", random_state=20260811)
        )
        classifier.fit(matrix, np.asarray([[1, 0], [0, 1]], dtype=np.int8))
        metadata = M1.fitted_model_metadata(
            {"vectorizer": vectorizer, "classifier": classifier}
        )
        json.dumps(metadata)
        self.assertEqual(metadata["estimator_count"], 2)


if __name__ == "__main__":
    unittest.main()
