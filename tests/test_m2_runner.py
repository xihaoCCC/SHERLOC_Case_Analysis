"""Focused offline protocol tests for the frozen M2 ModernBERT runner."""

from __future__ import annotations

import argparse
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

SPEC = importlib.util.spec_from_file_location("run_m2_modernbert", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {RUNNER_PATH}")
M2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M2
SPEC.loader.exec_module(M2)


class M2RunnerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_config_identity_revision_architecture_and_hash(self) -> None:
        self.assertEqual(self.config["config_id"], "m2-modernbert-amp-v2")
        self.assertEqual(self.config["status"], "FROZEN_PHASE4_EXECUTION_READY")
        self.assertEqual(M2.sha256_file(CONFIG_PATH), M2.EXPECTED_CONFIG_SHA256)
        self.assertEqual(
            M2.sha256_file(M2.DEFAULT_TOKEN_AUDIT),
            M2.EXPECTED_TOKEN_AUDIT_SHA256,
        )
        self.assertEqual(
            M2.sha256_file(M2.DEFAULT_A1_SPLIT), M2.EXPECTED_A1_SPLIT_SHA256
        )
        self.assertEqual(
            M2.sha256_file(M2.DEFAULT_A2_SPLIT), M2.EXPECTED_A2_SPLIT_SHA256
        )
        model = self.config["model"]
        self.assertEqual(model["pretrained_model_id"], M2.EXPECTED_MODEL_ID)
        self.assertEqual(model["revision"], M2.EXPECTED_MODEL_REVISION)
        self.assertEqual(model["shared_encoder_count"], 1)
        self.assertEqual(model["classification_head_count"], 1)
        self.assertEqual(model["num_labels"], 17)
        self.assertEqual(self.config["tokenization"]["max_length"], 2048)

    def test_six_configuration_grid_order_is_deterministic(self) -> None:
        grid = M2.parameter_grid(self.config)
        self.assertEqual(len(grid), 6)
        self.assertEqual(
            grid[0], {"learning_rate": 0.00001, "weight_decay": 0.01}
        )
        self.assertEqual(
            grid[-1], {"learning_rate": 0.00003, "weight_decay": 0.05}
        )

    def test_batch_fallback_preserves_effective_sixteen_before_length_change(self) -> None:
        attempts = M2.batch_attempts(self.config)
        self.assertEqual(
            [
                (
                    item.train_batch_size,
                    item.eval_batch_size,
                    item.gradient_accumulation_steps,
                    item.effective_train_batch_size,
                )
                for item in attempts
            ],
            [(4, 8, 4, 16), (2, 4, 8, 16), (1, 2, 16, 16)],
        )
        self.assertEqual(M2.EXPECTED_MAX_LENGTH, 2048)
        batch_one = M2.batch_attempts(
            self.config, initial_train_batch_size=1
        )
        self.assertEqual(
            [
                (
                    item.train_batch_size,
                    item.gradient_accumulation_steps,
                    item.effective_train_batch_size,
                )
                for item in batch_one
            ],
            [(1, 16, 16)],
        )

    def test_global_threshold_is_validation_only_and_prefers_point_five(self) -> None:
        y_validation = np.asarray(
            [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32
        )
        probabilities = np.asarray(
            [[0.70, 0.30], [0.60, 0.40], [0.40, 0.60], [0.30, 0.70]],
            dtype=np.float64,
        )
        threshold, rows = M2.select_global_threshold(y_validation, probabilities)
        self.assertEqual(threshold, 0.50)
        self.assertEqual(len(rows), 13)
        self.assertEqual(M2.configured_threshold_grid(self.config), M2.THRESHOLD_GRID)

    def test_static_validation_reconfirms_nine_truncated_cases(self) -> None:
        benchmark, labels, config, token_info = M2.validate_static_inputs(
            M2.DEFAULT_BENCHMARK,
            M2.DEFAULT_ONTOLOGY,
            M2.DEFAULT_CONFIG,
            M2.DEFAULT_TOKEN_AUDIT,
        )
        self.assertEqual(len(benchmark), 1263)
        self.assertEqual(len(labels), 17)
        self.assertEqual(config["method_id"], "M2")
        self.assertEqual(sum(info.truncated for info in token_info.values()), 9)
        self.assertEqual(
            max(info.original_token_count for info in token_info.values()), 7351
        )

    def test_alternate_config_and_split_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed_config = root / "m2.json"
            changed = dict(self.config)
            changed["config_version"] = "unexpected"
            changed_config.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(M2.M2ProtocolError):
                M2.validate_static_inputs(
                    M2.DEFAULT_BENCHMARK,
                    M2.DEFAULT_ONTOLOGY,
                    changed_config,
                    M2.DEFAULT_TOKEN_AUDIT,
                )

            changed_split = root / "a1.csv"
            changed_split.write_bytes(M2.DEFAULT_A1_SPLIT.read_bytes() + b"\n")
            spec = M2.RunSpec(
                evaluation="A1",
                fold=None,
                split_path=changed_split,
                model_dir=root / "model",
                prediction_path=root / "prediction.jsonl",
            )
            with self.assertRaises(M2.M2ProtocolError):
                M2.validate_and_partition_split(
                    spec,
                    M2.load_csv(changed_split),
                    [],
                    list(self.config["targets"]["label_order"]),
                )

    def test_prediction_schema_is_compatible_and_preserves_truncation(self) -> None:
        labels = list(self.config["targets"]["label_order"])
        record = {
            "identity": {
                "search_rank": 99,
                "unodc_case_number": "X99",
                "canonical_url": "https://example.test/case.html",
                "jurisdiction_country_raw": "Example",
            },
            "text_input": {"english_fact_summary_raw": "A case narrative."},
            "amp_targets": {
                "act_ontology_ids": [labels[0]],
                "means_ontology_ids": [labels[5]],
                "purpose_ontology_ids": [labels[11]],
            },
        }
        spec = M2.RunSpec(
            evaluation="A2",
            fold=2,
            split_path=Path("unused"),
            model_dir=Path("unused"),
            prediction_path=Path("unused"),
        )
        probabilities = np.zeros((1, 17), dtype=np.float64)
        probabilities[0, [0, 5, 11]] = 0.9
        rows = M2.prediction_rows(
            spec,
            [record],
            probabilities,
            labels,
            0.45,
            "run-id",
            {
                "config_sha256": "c",
                "split_membership_sha256": "s",
                "technical_execution_options": {
                    "gradient_checkpointing": True,
                    "max_length": 2048,
                },
            },
            {99: M2.TokenInfo(2500, True, 2048)},
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["method_id"], "M2")
        self.assertEqual(row["split"], "TEST")
        self.assertEqual(row["fold"], 2)
        self.assertEqual(row["predicted_labels"], [labels[0], labels[5], labels[11]])
        self.assertEqual(row["predicted_labels_0_50"], row["predicted_labels"])
        self.assertTrue(row["truncated_input"])
        self.assertEqual(row["original_token_count"], 2500)
        self.assertEqual(row["max_tokens_used"], 2048)
        self.assertTrue(row["gradient_checkpointing_during_training"])

        override_context = {
            "config_sha256": "c",
            "split_membership_sha256": "s",
            "technical_execution_options": {
                "gradient_checkpointing": True,
                "max_length": 1536,
            },
        }
        override_row = M2.prediction_rows(
            spec,
            [record],
            probabilities,
            labels,
            0.45,
            "override-run",
            override_context,
            {99: M2.TokenInfo(2500, True, 2048)},
        )[0]
        self.assertEqual(override_row["max_length"], 1536)
        self.assertEqual(override_row["max_tokens_used"], 1536)
        self.assertTrue(override_row["truncated_input"])

    def test_gradient_checkpointing_addendum_is_explicit_and_disables_cache(self) -> None:
        class FakeConfig:
            use_cache = True

        class FakeModel:
            supports_gradient_checkpointing = True
            is_gradient_checkpointing = False

            def __init__(self) -> None:
                self.config = FakeConfig()
                self.observed_kwargs = None

            def gradient_checkpointing_enable(self, **kwargs: object) -> None:
                self.observed_kwargs = kwargs
                self.is_gradient_checkpointing = True

        model = FakeModel()
        metadata = M2.configure_gradient_checkpointing(model, enabled=True)
        self.assertEqual(
            model.observed_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertFalse(model.config.use_cache)
        self.assertTrue(metadata["gradient_checkpointing"])
        self.assertEqual(
            metadata["gradient_checkpointing_addendum_id"],
            M2.GRADIENT_CHECKPOINTING_ADDENDUM_ID,
        )
        self.assertTrue(metadata["model_config_has_use_cache"])
        self.assertEqual(metadata["model_use_cache_action"], "SET_FALSE")

        class NoCacheConfig:
            pass

        no_cache_model = FakeModel()
        no_cache_model.config = NoCacheConfig()
        no_cache_metadata = M2.configure_gradient_checkpointing(
            no_cache_model, enabled=True
        )
        self.assertFalse(no_cache_metadata["model_config_has_use_cache"])
        self.assertEqual(
            no_cache_metadata["model_use_cache_action"],
            "NOT_APPLICABLE_CONFIG_HAS_NO_ATTRIBUTE",
        )
        self.assertIsNone(
            no_cache_metadata["model_use_cache_during_training"]
        )

    def test_execution_context_separates_checkpointing_modes(self) -> None:
        labels = list(self.config["targets"]["label_order"])
        spec = M2.RunSpec(
            evaluation="A1",
            fold=None,
            split_path=M2.DEFAULT_A1_SPLIT,
            model_dir=Path("unused"),
            prediction_path=Path("unused"),
        )
        common = (
            spec,
            M2.DEFAULT_BENCHMARK,
            M2.DEFAULT_ONTOLOGY,
            M2.DEFAULT_CONFIG,
            M2.DEFAULT_TOKEN_AUDIT,
            {"split_membership_sha256": "membership"},
            labels,
        )
        options = {
            "initial_train_batch_size": 4,
            "execution_environment": {"torch": "test", "hardware": "cpu"},
            "max_length": 2048,
            "max_length_override_acknowledged": False,
            "technical_override_rationale": None,
        }
        regular = M2.execution_context(
            *common, gradient_checkpointing=False, **options
        )
        checkpointed = M2.execution_context(
            *common, gradient_checkpointing=True, **options
        )
        self.assertNotEqual(M2.context_digest(regular), M2.context_digest(checkpointed))
        self.assertFalse(
            regular["technical_execution_options"]["gradient_checkpointing"]
        )
        self.assertTrue(
            checkpointed["technical_execution_options"]["gradient_checkpointing"]
        )
        self.assertEqual(
            checkpointed["technical_execution_options"]["max_length"], 2048
        )
        self.assertFalse(
            checkpointed["technical_execution_options"]["selection_or_tuning_change"]
        )
        self.assertEqual(
            checkpointed["execution_environment_sha256"],
            M2.sha256_text(
                M2.canonical_json(checkpointed["execution_environment"])
            ),
        )
        self.assertEqual(
            checkpointed["runner_source"]["sha256"],
            M2.sha256_file(RUNNER_PATH),
        )

        allocator_env: dict[str, str] = {}
        allocator = M2.configure_mps_allocator(
            "1.0", environ=allocator_env, torch_already_imported=False
        )
        memory_controlled = M2.execution_context(
            *common,
            gradient_checkpointing=True,
            mps_allocator=allocator,
            adamw_foreach_false=True,
            pad_to_multiple_of=64,
            **options,
        )
        self.assertNotEqual(
            M2.context_digest(checkpointed),
            M2.context_digest(memory_controlled),
        )
        technical = memory_controlled["technical_execution_options"]
        self.assertEqual(
            memory_controlled["mps_allocator"]["requested_low_watermark_ratio"],
            1.0,
        )
        self.assertEqual(technical["adamw_foreach_mode"], "EXPLICIT_FALSE")
        self.assertIs(technical["adamw_foreach"], False)
        self.assertEqual(technical["pad_to_multiple_of"], 64)
        self.assertEqual(
            technical["padding_policy"],
            "DYNAMIC_TO_LONGEST_ROUNDED_TO_MULTIPLE_64",
        )
        self.assertTrue(technical["tensor_shape_or_padding_change"])
        self.assertFalse(technical["non_padding_token_content_change"])
        self.assertFalse(technical["truncation_policy_change"])
        self.assertEqual(technical["progress_interval_batches"], 100)
        self.assertFalse(technical["progress_semantic_data_logged"])

    def test_mps_low_watermark_is_explicit_validated_and_runtime_checked(self) -> None:
        environment: dict[str, str] = {}
        provenance = M2.configure_mps_allocator(
            "1.0", environ=environment, torch_already_imported=False
        )
        self.assertEqual(environment[M2.MPS_LOW_WATERMARK_ENV], "1.0")
        self.assertEqual(provenance["application_source"], "CLI_APPLIED")
        self.assertTrue(provenance["applied_before_torch_import"])
        self.assertIsNone(provenance["high_watermark_override"])
        observed = M2.validate_runtime_mps_allocator(
            provenance, environ=environment
        )
        self.assertEqual(observed[M2.MPS_LOW_WATERMARK_ENV], "1.0")
        self.assertIsNone(observed[M2.MPS_HIGH_WATERMARK_ENV])
        with mock.patch.object(M2.os, "environ", environment):
            runtime = M2.runtime_environment(
                {
                    "torch": SimpleNamespace(__version__="2.11.0"),
                    "transformers": SimpleNamespace(__version__="5.5.3"),
                },
                {"backend": "mps"},
                provenance,
            )
        self.assertEqual(
            runtime["mps_allocator_environment"][M2.MPS_LOW_WATERMARK_ENV],
            "1.0",
        )

        preexisting = {M2.MPS_LOW_WATERMARK_ENV: "1.00"}
        confirmed = M2.configure_mps_allocator(
            "1.0", environ=preexisting, torch_already_imported=False
        )
        self.assertEqual(
            confirmed["application_source"], "CLI_CONFIRMED_PREEXISTING"
        )
        self.assertEqual(preexisting[M2.MPS_LOW_WATERMARK_ENV], "1.0")
        with self.assertRaises(M2.M2ProtocolError):
            M2.validate_runtime_mps_allocator(
                provenance,
                environ={M2.MPS_LOW_WATERMARK_ENV: "0.9"},
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.validate_runtime_mps_allocator(
                provenance,
                environ={
                    M2.MPS_LOW_WATERMARK_ENV: "1.0",
                    M2.MPS_HIGH_WATERMARK_ENV: "1.0",
                },
            )

    def test_mps_allocator_rejects_unsafe_or_unrecorded_settings(self) -> None:
        for invalid in ("", "0", "-0.1", "1.4", "2", "nan", "inf", "bad"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(M2.M2ProtocolError):
                    M2.configure_mps_allocator(
                        invalid, environ={}, torch_already_imported=False
                    )
        with self.assertRaises(M2.M2ProtocolError):
            M2.configure_mps_allocator(
                "1.0",
                environ={M2.MPS_HIGH_WATERMARK_ENV: "0.0"},
                torch_already_imported=False,
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.configure_mps_allocator(
                None,
                environ={M2.MPS_LOW_WATERMARK_ENV: "1.0"},
                torch_already_imported=False,
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.configure_mps_allocator(
                "1.0",
                environ={M2.MPS_LOW_WATERMARK_ENV: "0.9"},
                torch_already_imported=False,
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.configure_mps_allocator(
                "1.0", environ={}, torch_already_imported=True
            )

    def test_adamw_foreach_false_is_an_opt_in_implementation_pin(self) -> None:
        self.assertEqual(M2.adamw_execution_kwargs(foreach_false=False), {})
        self.assertEqual(
            M2.adamw_execution_kwargs(foreach_false=True), {"foreach": False}
        )

    def test_padding_multiple_preserves_real_tokens_masks_and_cap(self) -> None:
        class FakeTokenizer:
            padding_side = "right"
            pad_token_id = 99

            def __init__(self) -> None:
                self.last_pad_to_multiple_of = None

            def pad(
                self,
                features: list[dict[str, list[int]]],
                *,
                padding: bool,
                pad_to_multiple_of: int | None,
                return_tensors: str,
            ) -> dict[str, np.ndarray]:
                self.last_pad_to_multiple_of = pad_to_multiple_of
                self_outer = self
                longest = max(len(item["input_ids"]) for item in features)
                if pad_to_multiple_of:
                    longest = (
                        (longest + pad_to_multiple_of - 1)
                        // pad_to_multiple_of
                        * pad_to_multiple_of
                    )
                return {
                    "input_ids": np.asarray(
                        [
                            item["input_ids"]
                            + [self_outer.pad_token_id]
                            * (longest - len(item["input_ids"]))
                            for item in features
                        ]
                    ),
                    "attention_mask": np.asarray(
                        [
                            item["attention_mask"]
                            + [0] * (longest - len(item["attention_mask"]))
                            for item in features
                        ]
                    ),
                }

        class FakeTorch:
            float32 = np.float32

            @staticmethod
            def as_tensor(value: object, dtype: object) -> np.ndarray:
                return np.asarray(value, dtype=dtype)

        tokenizer = FakeTokenizer()
        collate = M2.make_collator(
            tokenizer, FakeTorch, pad_to_multiple_of=64, max_length=2048
        )
        items = [
            {
                "features": {
                    "input_ids": list(range(63)),
                    "attention_mask": [1] * 63,
                },
                "labels": np.zeros(17),
            },
            {
                "features": {
                    "input_ids": list(range(65)),
                    "attention_mask": [1] * 65,
                },
                "labels": np.ones(17),
            },
        ]
        batch = collate(items)
        self.assertEqual(tokenizer.last_pad_to_multiple_of, 64)
        self.assertEqual(batch["input_ids"].shape, (2, 128))
        self.assertEqual(batch["input_ids"][1, :65].tolist(), list(range(65)))
        self.assertEqual(int(batch["attention_mask"][1].sum()), 65)
        self.assertTrue(np.all(batch["attention_mask"][1, 65:] == 0))
        self.assertTrue(np.all(batch["input_ids"][1, 65:] == 99))

        for length, expected in ((2047, 2048), (2048, 2048)):
            one = collate(
                [
                    {
                        "features": {
                            "input_ids": [7] * length,
                            "attention_mask": [1] * length,
                        },
                        "labels": np.zeros(17),
                    }
                ]
            )
            self.assertEqual(one["input_ids"].shape[-1], expected)
            self.assertEqual(int(one["attention_mask"].sum()), length)

        with self.assertRaises(M2.M2ProtocolError):
            collate(
                [
                    {
                        "features": {
                            "input_ids": [7] * 2049,
                            "attention_mask": [1] * 2049,
                        },
                        "labels": np.zeros(17),
                    }
                ]
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.make_collator(
                tokenizer, FakeTorch, pad_to_multiple_of=64, max_length=2000
            )

    def test_mps_uses_bfloat16_autocast_without_gradient_scaler(self) -> None:
        mps = M2.precision_settings(SimpleNamespace(type="mps"), configured=True)
        cuda = M2.precision_settings(SimpleNamespace(type="cuda"), configured=True)
        cpu = M2.precision_settings(SimpleNamespace(type="cpu"), configured=True)
        self.assertEqual(
            mps,
            {
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "gradient_scaler_enabled": False,
            },
        )
        self.assertEqual(cuda["mixed_precision_dtype"], "float16")
        self.assertTrue(cuda["gradient_scaler_enabled"])
        self.assertEqual(cpu["mixed_precision_dtype"], "float32")
        self.assertFalse(cpu["mixed_precision"])

    def test_gradient_checkpointing_cli_is_opt_in_and_plan_visible(self) -> None:
        disabled = M2.parse_args(["--evaluation", "A1", "--plan"])
        enabled = M2.parse_args(
            ["--evaluation", "A1", "--plan", "--gradient-checkpointing"]
        )
        self.assertFalse(disabled.gradient_checkpointing)
        self.assertTrue(enabled.gradient_checkpointing)
        direct_batch_one = M2.parse_args(
            ["--evaluation", "A1", "--plan", "--initial-train-batch-size", "1"]
        )
        self.assertEqual(direct_batch_one.initial_train_batch_size, 1)
        memory_controls = M2.parse_args(
            [
                "--evaluation",
                "A1",
                "--plan",
                "--mps-low-watermark-ratio",
                "1.0",
                "--adamw-foreach-false",
                "--pad-to-multiple-of",
                "64",
            ]
        )
        self.assertEqual(memory_controls.mps_low_watermark_ratio, "1.0")
        self.assertTrue(memory_controls.adamw_foreach_false)
        self.assertEqual(memory_controls.pad_to_multiple_of, 64)

    def test_max_length_override_requires_acknowledgement_and_rationale(self) -> None:
        with self.assertRaises(M2.M2ProtocolError):
            M2.main(
                ["--evaluation", "A1", "--plan", "--max-length-override", "1536"]
            )
        args = M2.parse_args(
            [
                "--evaluation",
                "A1",
                "--plan",
                "--max-length-override",
                "1024",
                "--acknowledge-max-length-reduction",
                "--technical-override-rationale",
                "2048 OOM after clean batch-1 attempt",
            ]
        )
        self.assertEqual(args.max_length_override, 1024)
        self.assertTrue(args.acknowledge_max_length_reduction)

    def test_accumulation_remainder_is_case_weighted(self) -> None:
        # A1 has 884 training cases: 55 full windows of 16 plus 4 cases.
        self.assertEqual(
            M2.accumulation_window_case_count(
                dataset_n=884,
                micro_batch_size=1,
                gradient_accumulation_steps=16,
                batch_index_one_based=881,
            ),
            4,
        )
        self.assertEqual(
            M2.accumulation_window_case_count(
                dataset_n=884,
                micro_batch_size=4,
                gradient_accumulation_steps=4,
                batch_index_one_based=221,
            ),
            4,
        )
        self.assertEqual(
            M2.accumulation_window_case_count(
                dataset_n=884,
                micro_batch_size=1,
                gradient_accumulation_steps=16,
                batch_index_one_based=1,
            ),
            16,
        )

    def test_probability_and_strict_json_guards_reject_nonfinite_values(self) -> None:
        with self.assertRaises(M2.M2ProtocolError):
            M2.validate_probability_matrix(
                np.full((1, 17), np.nan), artifact="test"
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.validate_probability_matrix(
                np.full((1, 17), 1.01), artifact="test"
            )
        with self.assertRaises(M2.M2ProtocolError):
            M2.canonical_json({"bad": float("inf")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"bad": NaN}\n', encoding="utf-8")
            with self.assertRaises(M2.M2ProtocolError):
                M2.load_json(path)

    def test_run_lock_is_exclusive_and_owner_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = M2.RunSpec(
                evaluation="A1",
                fold=None,
                split_path=root / "split.csv",
                model_dir=root / "models/a1",
                prediction_path=root / "predictions/a1.jsonl",
            )
            lock = M2.RunLock.acquire(
                spec,
                break_stale_lock=False,
                execution_options={"initial_train_batch_size": 1},
            )
            self.assertTrue(lock.path.is_file())
            with self.assertRaises(M2.M2ProtocolError):
                M2.RunLock.acquire(
                    spec,
                    break_stale_lock=False,
                    execution_options={},
                )
            lock.release()
            self.assertFalse(lock.path.exists())

    def test_forced_restart_archives_failed_artifacts_recoverably(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = M2.RunSpec(
                evaluation="A1",
                fold=None,
                split_path=root / "split.csv",
                model_dir=root / "models/a1",
                prediction_path=root / "predictions/a1.jsonl",
            )
            trial = spec.model_dir / "grid/configuration_01/trial_state.json"
            metadata = spec.model_dir / "run_metadata.json"
            M2.atomic_json(trial, {"status": "FAILED", "attempts": [1]})
            M2.atomic_json(metadata, {"status": "FAILED"})
            provenance = M2.archive_forced_restart(
                spec,
                context={"context": "new"},
                force_reason="documented hardware-mode restart",
            )
            self.assertFalse(trial.exists())
            self.assertFalse(metadata.exists())
            event = Path(provenance["restart_event_path"])
            self.assertTrue(event.is_file())
            payload = M2.load_json(event)
            self.assertEqual(
                payload["reason"], "documented hardware-mode restart"
            )
            self.assertTrue(payload["archived_artifacts"])
            self.assertEqual(
                payload["previous_statuses"][M2.display_path(metadata)],
                "FAILED",
            )
            self.assertEqual(
                payload["previous_statuses"][M2.display_path(trial)],
                "FAILED",
            )

    def test_exhausted_oom_attempts_receive_terminal_trial_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory) / "configuration_01"
            context = {
                "technical_execution_options": {
                    "gradient_checkpointing": True,
                    "initial_train_batch_size": 1,
                    "max_length": 2048,
                    "batch_fallback_sequence": [1],
                    "mixed_precision_policy": "CUDA_FP16_MPS_BF16_CPU_FP32",
                }
            }
            with mock.patch.object(
                M2,
                "train_attempt",
                side_effect=RuntimeError("MPS backend out of memory"),
            ):
                with self.assertRaises(M2.M2ProtocolError):
                    M2.run_grid_trial(
                        1,
                        {"learning_rate": 0.00001, "weight_decay": 0.01},
                        trial_dir,
                        {"torch": object()},
                        object(),
                        object(),
                        object(),
                        self.config,
                        list(self.config["targets"]["label_order"]),
                        context,
                        SimpleNamespace(type="cpu"),
                        force=False,
                        local_files_only=True,
                    )
            state = M2.load_json(trial_dir / "trial_state.json")
            self.assertEqual(
                state["status"], "FAILED_OOM_BATCH_FALLBACK_EXHAUSTED"
            )
            self.assertEqual(len(state["batch_attempts"]), 1)
            self.assertEqual(state["batch_attempts"][0]["train_batch_size"], 1)

    def test_a2_specs_have_distinct_model_directories_and_checkpoints(self) -> None:
        args = argparse.Namespace(
            evaluation="A2",
            fold=None,
            a1_split=M2.DEFAULT_A1_SPLIT,
            a2_split=M2.DEFAULT_A2_SPLIT,
            model_root=Path("models"),
            prediction_root=Path("predictions"),
        )
        specs = M2.make_specs(args)
        self.assertEqual([spec.fold for spec in specs], [1, 2, 3])
        self.assertEqual(len({spec.model_dir for spec in specs}), 3)
        self.assertEqual(len({spec.prediction_path for spec in specs}), 3)

    def test_only_memory_errors_trigger_batch_fallback(self) -> None:
        self.assertTrue(M2.is_out_of_memory(RuntimeError("CUDA out of memory")))
        self.assertTrue(
            M2.is_out_of_memory(RuntimeError("MPS backend out of memory"))
        )
        self.assertFalse(M2.is_out_of_memory(RuntimeError("shape mismatch")))
        self.assertFalse(M2.is_out_of_memory(ValueError("out of memory")))


if __name__ == "__main__":
    unittest.main()
