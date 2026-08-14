"""Regression tests for the M2 A2 fixed-A1-hyperparameter amendment."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "src/experiments/09_run_m2_modernbert.py"
CONFIG_PATH = REPO_ROOT / "config/experiments/m2_modernbert_amp_v2.yaml"

SPEC = importlib.util.spec_from_file_location("run_m2_modernbert_fixed", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {RUNNER_PATH}")
M2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M2
SPEC.loader.exec_module(M2)


def fake_trial_state() -> dict[str, object]:
    return {
        "best_epoch": 3,
        "best_validation_macro_average_precision": 0.8,
        "checkpoint_path": "checkpoint",
        "checkpoint_sha256": "checkpoint-sha",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "mixed_precision_dtype": "bfloat16",
        "gradient_scaler_enabled": False,
        "adamw_optimizer": "torch.optim.AdamW",
        "adamw_foreach_mode": "EXPLICIT_FALSE",
        "adamw_foreach_observed_param_group_value": False,
        "adamw_fused_observed_param_group_value": None,
        "pad_to_multiple_of": 64,
        "tokenizer_padding_side": "right",
        "tokenizer_pad_token_id": 50283,
        "model_config_has_use_cache": False,
        "model_use_cache_action": "NOT_APPLICABLE_CONFIG_HAS_NO_ATTRIBUTE",
        "model_use_cache_during_training": None,
        "accumulation_loss_scaling": "CASE_WEIGHTED_WINDOW_MEAN",
        "effective_train_batch_size_target": 16,
        "final_accumulation_window_cases": 14,
        "training_seconds": 100.0,
    }


class M2A2FixedProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.labels = list(cls.config["targets"]["label_order"])

    def test_fixed_parameters_are_exact_a1_selected_configuration_05(self) -> None:
        self.assertEqual(M2.FIXED_A2_CONFIGURATION_INDEX, 5)
        self.assertEqual(
            M2.parameter_grid(self.config)[4], M2.FIXED_A2_HYPERPARAMETERS
        )
        self.assertEqual(
            M2.sha256_file(M2.DEFAULT_FIXED_A2_AMENDMENT),
            M2.EXPECTED_FIXED_A2_AMENDMENT_SHA256,
        )

    def test_fixed_cli_is_a2_only_and_requires_stable_settings(self) -> None:
        with self.assertRaises(M2.M2ProtocolError):
            M2.main(
                [
                    "--evaluation",
                    "A1",
                    "--plan",
                    "--a2-fixed-hyperparameters-from-a1",
                ]
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.main(
                [
                    "--evaluation",
                    "A2",
                    "--fold",
                    "2",
                    "--plan",
                    "--a2-fixed-hyperparameters-from-a1",
                ]
            )

    def test_same_context_interrupted_run_is_recoverable_only_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "run_metadata.json"
            context = {"context": "fixed"}
            M2.atomic_json(
                metadata_path,
                {
                    "status": "INTERRUPTED",
                    "execution_context_sha256": M2.context_digest(context),
                },
            )
            with self.assertRaises(M2.M2ProtocolError):
                M2.complete_run_is_valid(
                    metadata_path, Path(directory) / "prediction.jsonl", context
                )
            self.assertFalse(
                M2.complete_run_is_valid(
                    metadata_path,
                    Path(directory) / "prediction.jsonl",
                    context,
                    recover_interrupted=True,
                )
            )

    def test_interrupted_fixed_trial_is_archived_before_fresh_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory) / "fixed/grid/configuration_05"
            context = {"scientific_protocol": {"protocol_id": M2.FIXED_A2_PROTOCOL_ID}}
            M2.atomic_json(
                trial / "trial_state.json",
                {
                    "status": "IN_PROGRESS",
                    "configuration_index": 5,
                    "parameters": dict(M2.FIXED_A2_HYPERPARAMETERS),
                    "execution_context_sha256": M2.context_digest(context),
                },
            )
            (trial / "training_log.csv").write_text("epoch\n1\n", encoding="utf-8")
            provenance = M2.prepare_fixed_trial_restart(trial, context)
            self.assertIsNotNone(provenance)
            self.assertFalse(trial.exists())
            archive = M2.resolve_artifact_path(provenance["archive_path"])
            self.assertTrue((archive / "trial_state.json").is_file())

    def test_fixed_fold2_runs_only_configuration_05(self) -> None:
        probabilities = np.full((2, len(self.labels)), 0.1, dtype=np.float64)
        probabilities[:, 0] = (0.9, 0.2)
        labels = np.zeros_like(probabilities, dtype=np.int8)
        labels[0, 0] = 1
        state = fake_trial_state()
        state["best_validation_macro_average_precision"] = M2.macro_average_precision(
            labels, probabilities
        )[0]
        spec = M2.RunSpec(
            evaluation="A2",
            fold=2,
            split_path=Path("unused"),
            model_dir=Path("unused"),
            prediction_path=Path("unused"),
        )
        context = {"technical_execution_options": {"max_length": 2048}}
        with (
            mock.patch.object(M2, "encode_records", return_value=object()),
            mock.patch.object(M2, "prepare_fixed_trial_restart", return_value=None),
            mock.patch.object(
                M2,
                "run_grid_trial",
                return_value=(state, probabilities, labels),
            ) as run_trial,
        ):
            selection, search, _ = M2.fit_and_select_fixed_a2(
                spec,
                {},
                object(),
                [{"identity": {"search_rank": 1}}],
                [{"identity": {"search_rank": 2}}],
                self.labels,
                self.config,
                context,
                SimpleNamespace(type="cpu"),
                Path("fixed"),
                legacy_fold1=None,
                local_files_only=True,
            )
        self.assertEqual(run_trial.call_count, 1)
        self.assertEqual(run_trial.call_args.args[0], 5)
        self.assertEqual(run_trial.call_args.args[1], M2.FIXED_A2_HYPERPARAMETERS)
        self.assertEqual(len(search), 1)
        self.assertEqual(selection["selected_configuration_index"], 5)
        self.assertEqual(
            selection["hyperparameter_selection_source"], "A1_VALIDATION_TRANSFER"
        )
        self.assertFalse(selection["per_fold_hyperparameter_search"])

    def test_fixed_fold1_reuses_c5_without_training(self) -> None:
        probabilities = np.full((2, len(self.labels)), 0.1, dtype=np.float64)
        probabilities[:, 0] = (0.9, 0.2)
        labels = np.zeros_like(probabilities, dtype=np.int8)
        labels[0, 0] = 1
        state = fake_trial_state()
        state["best_validation_macro_average_precision"] = M2.macro_average_precision(
            labels, probabilities
        )[0]
        spec = M2.RunSpec(
            evaluation="A2",
            fold=1,
            split_path=Path("unused"),
            model_dir=Path("unused"),
            prediction_path=Path("unused"),
        )
        with (
            mock.patch.object(M2, "encode_records", return_value=object()),
            mock.patch.object(M2, "run_grid_trial") as run_trial,
        ):
            selection, search, _ = M2.fit_and_select_fixed_a2(
                spec,
                {},
                object(),
                [{"identity": {"search_rank": 1}}],
                [{"identity": {"search_rank": 2}}],
                self.labels,
                self.config,
                {"technical_execution_options": {"max_length": 2048}},
                SimpleNamespace(type="cpu"),
                Path("fixed"),
                legacy_fold1={
                    "state": state,
                    "probabilities": probabilities,
                    "labels": labels,
                },
                local_files_only=True,
            )
        run_trial.assert_not_called()
        self.assertTrue(selection["training_reused_without_retraining"])
        self.assertEqual(search[0]["selection_basis"], "A1_VALIDATION_TRANSFER")

    @unittest.skipUnless(
        (M2.DEFAULT_MODEL_ROOT / "a1/run_metadata.json").is_file()
        and (M2.DEFAULT_MODEL_ROOT / "a2_fold_1/run_metadata.json").is_file(),
        "completed local M2 artifacts are unavailable",
    )
    def test_real_a1_transfer_and_fold1_c5_integrity(self) -> None:
        benchmark, labels, config, _ = M2.validate_static_inputs(
            M2.DEFAULT_BENCHMARK,
            M2.DEFAULT_ONTOLOGY,
            M2.DEFAULT_CONFIG,
            M2.DEFAULT_TOKEN_AUDIT,
        )
        a1 = M2.validate_a1_fixed_transfer_source(M2.DEFAULT_MODEL_ROOT, config)
        self.assertEqual(a1["configuration_index"], 5)
        spec = M2.RunSpec(
            evaluation="A2",
            fold=1,
            split_path=M2.DEFAULT_A2_SPLIT,
            model_dir=M2.DEFAULT_MODEL_ROOT / "a2_fold_1",
            prediction_path=M2.DEFAULT_PREDICTION_ROOT
            / "a2_fold_1_test_predictions.jsonl",
        )
        _, validation, _, split = M2.validate_and_partition_split(
            spec, M2.load_csv(M2.DEFAULT_A2_SPLIT), benchmark, labels
        )
        legacy = M2.validate_legacy_fold1_c5(spec, validation, labels, split)
        self.assertEqual(legacy["state"]["epochs_completed"], 6)
        self.assertEqual(legacy["state"]["best_epoch"], 6)
        self.assertEqual(split["heldout_jurisdiction_leakage_n"], 0)


if __name__ == "__main__":
    unittest.main()

