"""Focused offline tests for leakage-free Evaluation B M1/M2 runners."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "src/experiments"
sys.path.insert(0, str(EXPERIMENTS))

import evaluation_b_supervised as COMMON  # noqa: E402


def load_numeric(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, EXPERIMENTS / filename)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {filename}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


M1 = load_numeric("evaluation_b_m1_runner_test", "20_run_evaluation_b_m1.py")
M2 = load_numeric("evaluation_b_m2_runner_test", "21_run_evaluation_b_m2.py")


class EvaluationBSupervisedPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = COMMON.prepare_supervised_data()

    def test_exact_frozen_membership_and_leakage_free_training(self) -> None:
        prepared = self.prepared
        self.assertEqual(
            prepared.membership_sha256,
            "2f6cf53ff2e806f4801fb5e59b5551fff681482ab85ce3ce5bd80947f08dfd75",
        )
        self.assertEqual(prepared.retained_n, 61)
        self.assertEqual(
            sum(case.review_status == "SUBSTANTIVE" for case in prepared.target_cases),
            55,
        )
        self.assertEqual(
            sum(case.review_status == "ABSTAIN" for case in prepared.target_cases),
            6,
        )
        self.assertEqual(prepared.retained_primary_cohort_n, 54)
        self.assertEqual(prepared.train_n, 1209)
        retained = {case.search_rank for case in prepared.target_cases}
        training = {int(row["identity"]["search_rank"]) for row in prepared.training_records}
        self.assertFalse(retained & training)
        self.assertEqual(len(training), 1209)
        self.assertTrue(all(value > 0 for value in prepared.training_label_supports.values()))

    def test_exclusion_audit_covers_every_retained_case(self) -> None:
        rows = self.prepared.exclusion_audit
        self.assertEqual(len(rows), 61)
        self.assertEqual(
            {row["reliability_case_id"] for row in rows},
            {case.reliability_case_id for case in self.prepared.target_cases},
        )
        for row in rows:
            self.assertEqual(row["removed_from_eval_b_supervised_training"], "TRUE")
            self.assertEqual(row["removed_from_eval_b_validation"], "TRUE")
            self.assertEqual(row["removed_from_eval_b_threshold_tuning"], "TRUE")
            self.assertEqual(row["removed_from_eval_b_supervised_label_selection"], "TRUE")

    def test_model_inputs_do_not_expose_human_labels(self) -> None:
        forbidden = {
            "acts_human_raw",
            "act_label_ids_json",
            "means_human_raw",
            "means_label_ids_json",
            "purpose_human_raw",
            "purpose_label_ids_json",
            "substantive_amp_evaluable",
        }
        for case in self.prepared.target_cases:
            record = case.model_record()
            serialized = json.dumps(record, sort_keys=True)
            self.assertTrue(forbidden.isdisjoint(serialized.split('"')))
            self.assertEqual(
                record["amp_targets"],
                {
                    "act_ontology_ids": [],
                    "means_ontology_ids": [],
                    "purpose_ontology_ids": [],
                },
            )
            self.assertEqual(
                set(record), {"identity", "text_input", "amp_targets"}
            )

    def test_prediction_contract_accepts_all_retained_cases_without_reference_labels(self) -> None:
        scores = np.zeros((self.prepared.retained_n, len(self.prepared.label_order)))
        scores[:, 0] = 0.25
        rows = COMMON.prediction_rows(
            method_id="M1",
            target_cases=self.prepared.target_cases,
            probabilities=scores,
            label_order=self.prepared.label_order,
            threshold=0.25,
            run_id="offline-test",
            config_sha256="a" * 64,
            membership_sha256=self.prepared.membership_sha256,
            training_membership_sha256="b" * 64,
        )
        self.assertEqual(len(rows), 61)
        self.assertEqual(len({row["search_rank"] for row in rows}), 61)
        self.assertTrue(all(row["predicted_labels"] == ["ACT_RECRUITMENT"] for row in rows))
        self.assertTrue(
            all("human" not in key for row in rows for key in row if key != "human_labels_used_for_training_tuning_or_prediction")
        )
        self.assertTrue(
            all(row["human_labels_used_for_training_tuning_or_prediction"] is False for row in rows)
        )


class EvaluationBFixedRunnerTest(unittest.TestCase):
    def test_m1_config_and_pipeline_are_exact_fixed_transfer(self) -> None:
        config = M1.validate_config(M1.DEFAULT_CONFIG)
        vectorizer, classifier = M1.build_pipeline(config)
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertEqual(vectorizer.min_df, 2)
        self.assertEqual(classifier.estimator.C, 1.0)
        self.assertIsNone(classifier.estimator.class_weight)
        self.assertEqual(M1.FIXED_THRESHOLD, 0.25)
        self.assertNotIn("GridSearch", (EXPERIMENTS / "20_run_evaluation_b_m1.py").read_text())

    def test_m2_config_is_exact_fixed_six_epoch_mps_transfer(self) -> None:
        config = M2.validate_config(M2.DEFAULT_CONFIG)
        fixed = config["fixed_training"]
        self.assertEqual(fixed["learning_rate"], 3e-5)
        self.assertEqual(fixed["weight_decay"], 0.01)
        self.assertEqual(fixed["epochs"], 6)
        self.assertFalse(fixed["early_stopping"])
        self.assertEqual(fixed["physical_train_batch_size"], 1)
        self.assertEqual(fixed["gradient_accumulation_steps"], 16)
        self.assertEqual(config["tokenization"]["max_length"], 2048)
        self.assertEqual(config["tokenization"]["pad_to_multiple_of"], 64)
        self.assertEqual(config["thresholding"]["global_threshold"], 0.20)
        self.assertTrue(config["model"]["fresh_pretrained_initialization"])
        self.assertEqual(config["model"]["revision"], M2.EXPECTED_MODEL_REVISION)

    def test_preflight_path_never_loads_torch_or_model(self) -> None:
        prepared = object()
        summary = {"status": "READY"}
        with (
            patch.object(M2, "prepare_supervised_data", return_value=prepared),
            patch.object(M2, "write_preflight_artifacts", return_value=summary),
            patch.object(M2, "load_phase4_helpers") as load_helpers,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(M2.main(["--preflight"]), 0)
        load_helpers.assert_not_called()

    def test_execute_is_blocked_without_explicit_freeze_confirmation(self) -> None:
        prepared = object()
        with (
            patch.object(M1, "prepare_supervised_data", return_value=prepared),
            patch.object(M1, "write_preflight_artifacts", return_value={"status": "READY"}),
            patch.object(M1, "execute") as execute_m1,
            redirect_stdout(StringIO()),
        ):
            with self.assertRaises(COMMON.EvaluationBSupervisedError):
                M1.main(["--execute"])
        execute_m1.assert_not_called()

        with (
            patch.object(M2, "prepare_supervised_data", return_value=prepared),
            patch.object(M2, "write_preflight_artifacts", return_value={"status": "READY"}),
            patch.object(M2, "execute") as execute_m2,
            patch.object(M2, "load_phase4_helpers") as load_helpers,
            redirect_stdout(StringIO()),
        ):
            with self.assertRaises(COMMON.EvaluationBSupervisedError):
                M2.main(["--execute"])
        execute_m2.assert_not_called()
        load_helpers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
