#!/usr/bin/env python3
"""Run the dedicated leakage-free fixed-protocol Evaluation B M2 model.

``--preflight`` validates frozen membership and writes the shared exclusion
audit without importing PyTorch.  A real run requires both ``--execute`` and
``--confirm-qc-membership-freeze``.  The training protocol has no validation
partition, search, early stopping, or human-label access: it starts from the
pinned ModernBERT revision, fits exactly six epochs on leakage-free silver
labels, and predicts every retained Evaluation B narrative.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation_b_supervised import (
    DEFAULT_A1_SPLIT,
    DEFAULT_A2_SPLIT,
    DEFAULT_AUDIT,
    DEFAULT_BENCHMARK,
    DEFAULT_HUMAN_REFERENCE,
    DEFAULT_MEMBERSHIP,
    DEFAULT_MEMBERSHIP_FREEZE,
    DEFAULT_ONTOLOGY,
    DEFAULT_PREFLIGHT,
    DEFAULT_RELIABILITY_SAMPLE,
    EvaluationBSupervisedError,
    RunLock,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    canonical_json,
    prediction_rows,
    prepare_supervised_data,
    sha256_directory,
    sha256_file,
    texts,
    training_membership_sha256,
    utc_now,
    write_preflight_artifacts,
)


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/experiments/eval_b_m2_modernbert_v1.yaml"
DEFAULT_M1_CONFIG = REPO_ROOT / "config/experiments/eval_b_m1_tfidf_logreg_v1.yaml"
DEFAULT_PHASE4_RUNNER = REPO_ROOT / "src/experiments/09_run_m2_modernbert.py"
DEFAULT_MODEL_DIR = REPO_ROOT / "outputs/models/evaluation_b/m2"
DEFAULT_PREDICTION = REPO_ROOT / "outputs/predictions/evaluation_b/m2/predictions.jsonl"

EXPECTED_CONFIG_SHA256 = "5a83104cda51b8674ab577ea133be991ebccab99a30915e3fc307b219a64ed7b"
EXPECTED_M1_CONFIG_SHA256 = "5c6a916af3781305926b0cd57bde77e30f7c094a035a313cda95fc391a4046a5"
EXPECTED_MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
FIXED_THRESHOLD = 0.20
FIXED_EPOCHS = 6
FIXED_MAX_LENGTH = 2048
FIXED_TRAIN_BATCH_SIZE = 1
FIXED_PREDICTION_BATCH_SIZE = 2
FIXED_GRADIENT_ACCUMULATION = 16
FIXED_PAD_MULTIPLE = 64
FIXED_MPS_LOW_WATERMARK = "1.0"


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError(f"Cannot load fixed M2 config: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationBSupervisedError("Fixed M2 config must be an object")
    return value


def validate_config(path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_CONFIG_SHA256:
        raise EvaluationBSupervisedError(
            f"Evaluation B M2 config hash mismatch: expected {EXPECTED_CONFIG_SHA256}, got {actual_sha}"
        )
    config = load_config(path)
    if config.get("config_id") != "eval-b-m2-modernbert-v1" or config.get("method_id") != "M2":
        raise EvaluationBSupervisedError("Wrong Evaluation B M2 config identity")
    model = config["model"]
    tokenization = config["tokenization"]
    training = config["fixed_training"]
    expected_training = {
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "epochs": FIXED_EPOCHS,
        "early_stopping": False,
        "physical_train_batch_size": FIXED_TRAIN_BATCH_SIZE,
        "prediction_batch_size": FIXED_PREDICTION_BATCH_SIZE,
        "gradient_accumulation_steps": FIXED_GRADIENT_ACCUMULATION,
        "effective_train_batch_size": 16,
        "gradient_checkpointing": True,
        "mixed_precision_mps": "bfloat16_autocast",
        "gradient_scaler_mps": False,
        "optimizer": "torch.optim.AdamW",
        "adamw_foreach": False,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "mps_low_watermark_ratio": 1.0,
        "mps_high_watermark_override": None,
        "data_seed": 20260811,
    }
    if training != expected_training:
        raise EvaluationBSupervisedError("Evaluation B M2 fixed training settings drifted")
    if (
        model.get("pretrained_model_id") != "answerdotai/ModernBERT-base"
        or model.get("tokenizer_id") != "answerdotai/ModernBERT-base"
        or model.get("revision") != EXPECTED_MODEL_REVISION
        or model.get("tokenizer_revision") != EXPECTED_MODEL_REVISION
        or model.get("num_labels") != 17
        or model.get("fresh_pretrained_initialization") is not True
        or int(tokenization.get("max_length", -1)) != FIXED_MAX_LENGTH
        or tokenization.get("truncation_side") != "right"
        or int(tokenization.get("pad_to_multiple_of", -1)) != FIXED_PAD_MULTIPLE
        or float(config["thresholding"].get("global_threshold", -1)) != FIXED_THRESHOLD
    ):
        raise EvaluationBSupervisedError("Evaluation B M2 model/token/threshold settings drifted")
    scope = config["scope_guard"]
    if any(
        bool(scope.get(key))
        for key in (
            "hyperparameter_grid",
            "threshold_search",
            "best_epoch_selection",
            "early_stopping",
            "human_labels_for_training_or_tuning",
            "evaluation_a_artifacts_mutable",
            "auxiliary_targets",
        )
    ):
        raise EvaluationBSupervisedError("Evaluation B M2 scope guard permits a forbidden operation")
    provenance = config["selection_provenance"]
    source_path = REPO_ROOT / provenance["a1_run_metadata"]
    if sha256_file(source_path) != provenance["a1_run_metadata_sha256"]:
        raise EvaluationBSupervisedError("Frozen A1 M2 selection metadata hash mismatch")
    selected = json.loads(source_path.read_text(encoding="utf-8"))["selection"]
    if (
        selected.get("selected_hyperparameters")
        != {"learning_rate": 3e-5, "weight_decay": 0.01}
        or int(selected.get("selected_best_epoch", -1)) != FIXED_EPOCHS
        or float(selected.get("selected_global_threshold", -1)) != FIXED_THRESHOLD
    ):
        raise EvaluationBSupervisedError("Config does not exactly transfer frozen A1 M2 selection")
    return config


def load_phase4_helpers(path: Path = DEFAULT_PHASE4_RUNNER) -> ModuleType:
    """Load proven ModernBERT helpers lazily; preflight never calls this."""

    specification = importlib.util.spec_from_file_location("sherloc_phase4_m2_helpers", path)
    if specification is None or specification.loader is None:
        raise EvaluationBSupervisedError(f"Cannot load ModernBERT helper runner: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _atomic_training_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_csv(path, rows)


def _complete_is_valid(
    metadata_path: Path,
    *,
    checkpoint_dir: Path,
    prediction_path: Path,
    config_sha256: str,
    membership_sha256: str,
    training_sha256: str,
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError("Existing M2 metadata is unreadable") from exc
    if metadata.get("status") != "COMPLETE":
        raise EvaluationBSupervisedError(
            "Incomplete Evaluation B M2 metadata exists; preserve it and choose a new output path"
        )
    expected = {
        "config_sha256": config_sha256,
        "retained_membership_sha256": membership_sha256,
        "training_membership_sha256": training_sha256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise EvaluationBSupervisedError("Existing M2 completion belongs to different frozen inputs")
    if not checkpoint_dir.is_dir() or sha256_directory(checkpoint_dir) != metadata.get("checkpoint_sha256"):
        raise EvaluationBSupervisedError("Existing Evaluation B M2 checkpoint is damaged")
    if not prediction_path.is_file() or sha256_file(prediction_path) != metadata.get("prediction_sha256"):
        raise EvaluationBSupervisedError("Existing Evaluation B M2 predictions are damaged")
    return True


def _valid_fit_state(
    fit_state_path: Path,
    *,
    checkpoint_dir: Path,
    config_sha256: str,
    membership_sha256: str,
    training_sha256: str,
) -> dict[str, Any] | None:
    if not fit_state_path.is_file():
        return None
    try:
        state = json.loads(fit_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError("Existing Evaluation B M2 fit state is unreadable") from exc
    if state.get("status") != "FIT_COMPLETE":
        raise EvaluationBSupervisedError("Existing Evaluation B M2 fit state is not complete")
    expected = {
        "config_sha256": config_sha256,
        "retained_membership_sha256": membership_sha256,
        "training_membership_sha256": training_sha256,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise EvaluationBSupervisedError("Existing M2 fit state belongs to different frozen inputs")
    if not checkpoint_dir.is_dir() or sha256_directory(checkpoint_dir) != state.get("checkpoint_sha256"):
        raise EvaluationBSupervisedError("Existing Evaluation B M2 fit checkpoint is damaged")
    return state


def _count_original_tokens(tokenizer: Any, cases: Sequence[Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for case in cases:
        encoded = tokenizer(
            case.fact_summary,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        result[case.search_rank] = len(token_ids)
    return result


def _train_fixed_six_epochs(
    *,
    helpers: ModuleType,
    stack: Mapping[str, Any],
    tokenizer: Any,
    config: Mapping[str, Any],
    prepared: Any,
    checkpoint_dir: Path,
    training_log_path: Path,
    device: Any,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any]]:
    torch = stack["torch"]
    fixed = config["fixed_training"]
    seed = int(fixed["data_seed"])
    helpers.seed_everything(stack, seed)
    model, model_commit, checkpointing = helpers.initialize_pretrained_model(
        stack,
        config,
        prepared.label_order,
        device,
        local_files_only=local_files_only,
        gradient_checkpointing=True,
    )
    train_dataset = helpers.encode_records(
        tokenizer,
        prepared.training_records,
        prepared.label_order,
        FIXED_MAX_LENGTH,
    )
    train_loader = helpers.make_loader(
        stack,
        train_dataset,
        tokenizer,
        FIXED_TRAIN_BATCH_SIZE,
        shuffle=True,
        seed=seed,
        device=device,
        pad_to_multiple_of=FIXED_PAD_MULTIPLE,
        max_length=FIXED_MAX_LENGTH,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fixed["learning_rate"]),
        weight_decay=float(fixed["weight_decay"]),
        **helpers.adamw_execution_kwargs(foreach_false=True),
    )
    observed_foreach = optimizer.param_groups[0].get("foreach")
    observed_fused = optimizer.param_groups[0].get("fused")
    if observed_foreach is not False:
        raise EvaluationBSupervisedError("AdamW did not retain foreach=False")
    updates_per_epoch = math.ceil(len(train_loader) / FIXED_GRADIENT_ACCUMULATION)
    total_updates = updates_per_epoch * FIXED_EPOCHS
    warmup_steps = math.ceil(total_updates * float(fixed["warmup_ratio"]))
    scheduler = stack["get_linear_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    precision = helpers.precision_settings(device, True)
    if precision != {
        "mixed_precision": True,
        "mixed_precision_dtype": "bfloat16",
        "gradient_scaler_enabled": False,
    }:
        raise EvaluationBSupervisedError(f"Unexpected MPS precision settings: {precision}")
    scaler = helpers.make_grad_scaler(torch, False)
    epoch_rows: list[dict[str, Any]] = []
    fit_started = time.perf_counter()

    for epoch in range(1, FIXED_EPOCHS + 1):
        epoch_started = time.perf_counter()
        print(f"Evaluation B M2 epoch_start epoch={epoch}/{FIXED_EPOCHS} batches={len(train_loader)}", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_n = 0
        optimizer_steps = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs, labels = helpers.move_batch(batch, device)
            batch_n = int(labels.shape[0])
            window_n = helpers.accumulation_window_case_count(
                dataset_n=len(train_dataset),
                micro_batch_size=FIXED_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=FIXED_GRADIENT_ACCUMULATION,
                batch_index_one_based=batch_index,
            )
            with helpers.autocast_context(torch, device, True):
                output = model(**inputs, labels=labels)
                scaled_loss = output.loss * (batch_n / window_n)
            raw_loss = float(output.loss.detach().float().cpu())
            if not math.isfinite(raw_loss):
                raise EvaluationBSupervisedError("ModernBERT training loss is non-finite")
            scaler.scale(scaled_loss).backward()
            train_loss_sum += raw_loss * batch_n
            train_n += batch_n
            should_step = (
                batch_index % FIXED_GRADIENT_ACCUMULATION == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(fixed["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if batch_index % 100 == 0:
                print(
                    f"Evaluation B M2 progress epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"elapsed_seconds={time.perf_counter() - epoch_started:.1f}",
                    flush=True,
                )
        if train_n != len(train_dataset) or optimizer_steps != updates_per_epoch:
            raise EvaluationBSupervisedError("ModernBERT epoch accounting mismatch")
        epoch_rows.append(
            {
                "epoch": epoch,
                "status": "COMPLETE",
                "train_n": train_n,
                "train_loss": train_loss_sum / train_n,
                "optimizer_steps": optimizer_steps,
                "optimizer_steps_expected": updates_per_epoch,
                "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "validation_run": "FALSE",
                "checkpoint_selection_run": "FALSE",
            }
        )
        _atomic_training_log(training_log_path, epoch_rows)
        print(
            f"Evaluation B M2 epoch_complete epoch={epoch}/{FIXED_EPOCHS} "
            f"train_loss={epoch_rows[-1]['train_loss']:.6f} "
            f"epoch_seconds={epoch_rows[-1]['epoch_seconds']:.1f}",
            flush=True,
        )

    checkpoint_sha = helpers.save_model_checkpoint(model, checkpoint_dir)
    metadata = {
        "fit_seconds": time.perf_counter() - fit_started,
        "epochs_completed": len(epoch_rows),
        "pretrained_model_resolved_commit": model_commit,
        "checkpoint_sha256": checkpoint_sha,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_batch_size": FIXED_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": FIXED_GRADIENT_ACCUMULATION,
        "effective_train_batch_size": 16,
        "optimizer_steps_per_epoch": updates_per_epoch,
        "total_optimizer_steps": total_updates,
        "warmup_steps": warmup_steps,
        "mixed_precision": True,
        "mixed_precision_dtype": "bfloat16",
        "gradient_scaler_enabled": False,
        "adamw_foreach_observed": observed_foreach,
        "adamw_fused_observed": observed_fused,
        "validation_run": False,
        "early_stopping": False,
        "best_epoch_selection": False,
        **checkpointing,
    }
    del optimizer, scheduler, scaler, train_loader, train_dataset
    return model, metadata


def execute(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    prepared: Any,
    model_dir: Path,
    prediction_path: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    config_sha = sha256_file(config_path)
    training_sha = training_membership_sha256(prepared.training_records)
    metadata_path = model_dir / "run_metadata.json"
    fit_state_path = model_dir / "fit_state.json"
    training_log_path = model_dir / "training_log.csv"
    checkpoint_dir = model_dir / "checkpoint_final"
    if _complete_is_valid(
        metadata_path,
        checkpoint_dir=checkpoint_dir,
        prediction_path=prediction_path,
        config_sha256=config_sha,
        membership_sha256=prepared.membership_sha256,
        training_sha256=training_sha,
    ):
        return {"status": "SKIPPED_COMPLETE", "metadata_path": str(metadata_path)}
    if prediction_path.exists() and not metadata_path.exists():
        raise EvaluationBSupervisedError("Uncommitted M2 prediction output exists and will not be overwritten")

    fit_state = _valid_fit_state(
        fit_state_path,
        checkpoint_dir=checkpoint_dir,
        config_sha256=config_sha,
        membership_sha256=prepared.membership_sha256,
        training_sha256=training_sha,
    )
    if fit_state is None and any(path.exists() for path in (checkpoint_dir, training_log_path)):
        raise EvaluationBSupervisedError(
            "Partial Evaluation B M2 fitting artifacts exist and will not be overwritten"
        )

    lock = RunLock.acquire(model_dir)
    started = time.perf_counter()
    started_at = utc_now()
    run_id = os.urandom(12).hex()
    helpers: ModuleType | None = None
    stack: Mapping[str, Any] | None = None
    device: Any = None
    model: Any = None
    try:
        helpers = load_phase4_helpers()
        try:
            allocator = helpers.configure_mps_allocator(FIXED_MPS_LOW_WATERMARK)
            stack = helpers.load_ml_stack()
            runtime_allocator = helpers.validate_runtime_mps_allocator(allocator)
        except helpers.M2ProtocolError as exc:
            raise EvaluationBSupervisedError(str(exc)) from exc
        device, hardware = helpers.select_device(stack["torch"])
        if device.type != "mps":
            raise EvaluationBSupervisedError(
                f"Fixed Evaluation B M2 execution requires Apple MPS, found {device.type!r}"
            )
        tokenizer, tokenizer_commit = helpers.load_tokenizer(
            stack, config, local_files_only=local_files_only
        )
        original_tokens = _count_original_tokens(tokenizer, prepared.target_cases)
        truncated = {
            rank: count > FIXED_MAX_LENGTH for rank, count in original_tokens.items()
        }
        if fit_state is None:
            model, fit_metadata = _train_fixed_six_epochs(
                helpers=helpers,
                stack=stack,
                tokenizer=tokenizer,
                config=config,
                prepared=prepared,
                checkpoint_dir=checkpoint_dir,
                training_log_path=training_log_path,
                device=device,
                local_files_only=local_files_only,
            )
            fit_state = {
                "artifact_schema_version": "sherloc-eval-b-m2-fit-v1",
                "status": "FIT_COMPLETE",
                "completed_at": utc_now(),
                "config_sha256": config_sha,
                "retained_membership_sha256": prepared.membership_sha256,
                "training_membership_sha256": training_sha,
                "train_n": prepared.train_n,
                "checkpoint_path": str(checkpoint_dir.resolve()),
                **fit_metadata,
            }
            atomic_json(fit_state_path, fit_state)
        else:
            model = helpers.load_local_model(stack, checkpoint_dir, device)

        target_records = [case.model_record() for case in prepared.target_cases]
        target_dataset = helpers.encode_records(
            tokenizer, target_records, prepared.label_order, FIXED_MAX_LENGTH
        )
        probabilities, _unused_empty_labels, inference_attempts = helpers.predict_with_batch_fallback(
            stack,
            model,
            target_dataset,
            tokenizer,
            device,
            initial_batch_size=FIXED_PREDICTION_BATCH_SIZE,
            seed=int(config["fixed_training"]["data_seed"]),
            mixed_precision=True,
            max_length=FIXED_MAX_LENGTH,
            pad_to_multiple_of=FIXED_PAD_MULTIPLE,
        )
        rows = prediction_rows(
            method_id="M2",
            target_cases=prepared.target_cases,
            probabilities=probabilities,
            label_order=prepared.label_order,
            threshold=FIXED_THRESHOLD,
            run_id=run_id,
            config_sha256=config_sha,
            membership_sha256=prepared.membership_sha256,
            training_membership_sha256=training_sha,
            truncated_by_rank=truncated,
            original_token_count_by_rank=original_tokens,
        )
        for row in rows:
            row["max_length"] = FIXED_MAX_LENGTH
            row["max_tokens_used"] = min(int(row["original_token_count"]), FIXED_MAX_LENGTH)
            row["truncation_side"] = "right"
            row["tokenizer_model_id"] = "answerdotai/ModernBERT-base"
            row["tokenizer_revision"] = EXPECTED_MODEL_REVISION
        atomic_jsonl(prediction_path, rows)
        metadata = {
            "artifact_schema_version": "sherloc-eval-b-m2-run-v1",
            "runner_version": VERSION,
            "status": "COMPLETE",
            "run_id": run_id,
            "method_id": "M2",
            "evaluation": "B",
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha,
            "source_sha256": dict(prepared.source_hashes),
            "retained_membership_sha256": prepared.membership_sha256,
            "training_membership_sha256": training_sha,
            "source_silver_cohort_n": len(prepared.benchmark),
            "retained_n": prepared.retained_n,
            "retained_primary_cohort_n": prepared.retained_primary_cohort_n,
            "train_n": prepared.train_n,
            "prediction_n": len(rows),
            "training_label_supports": dict(prepared.training_label_supports),
            "label_order": list(prepared.label_order),
            "fixed_hyperparameters": {
                "learning_rate": 3e-5,
                "weight_decay": 0.01,
                "epochs": FIXED_EPOCHS,
                "global_threshold": FIXED_THRESHOLD,
            },
            "selection_policy": "TRANSFERRED_A1_SETTINGS_FIXED_SIX_EPOCHS_NO_EVALUATION_B_SELECTION",
            "human_labels_used_for_training_tuning_or_prediction": False,
            "validation_run": False,
            "threshold_search": False,
            "fit_state_path": str(fit_state_path.resolve()),
            "fit_state_sha256": sha256_file(fit_state_path),
            "checkpoint_path": str(checkpoint_dir.resolve()),
            "checkpoint_sha256": sha256_directory(checkpoint_dir),
            "training_log_path": str(training_log_path.resolve()),
            "training_log_sha256": sha256_file(training_log_path),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "target_truncated_n": sum(truncated.values()),
            "test_inference_batch_attempts": inference_attempts,
            "tokenizer_resolved_commit": tokenizer_commit,
            "hardware": hardware,
            "mps_allocator": allocator,
            "mps_allocator_runtime": runtime_allocator,
            "technical_execution": {
                "max_length": FIXED_MAX_LENGTH,
                "physical_train_batch_size": FIXED_TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": FIXED_GRADIENT_ACCUMULATION,
                "effective_train_batch_size": 16,
                "gradient_checkpointing": True,
                "mixed_precision_dtype": "bfloat16",
                "gradient_scaler_enabled": False,
                "adamw_foreach": False,
                "pad_to_multiple_of": FIXED_PAD_MULTIPLE,
            },
            "fit": fit_state,
            "python": platform.python_version(),
            "hostname": socket.gethostname(),
            "numpy": np.__version__,
            "torch": str(stack["torch"].__version__),
            "transformers": str(stack["transformers"].__version__),
        }
        atomic_json(metadata_path, metadata)
        return {
            "status": "COMPLETE",
            "train_n": prepared.train_n,
            "prediction_n": len(rows),
            "metadata_path": str(metadata_path),
        }
    finally:
        if helpers is not None and stack is not None and device is not None:
            try:
                if model is not None:
                    del model
                helpers.clear_device_memory(stack["torch"], device)
            except Exception:
                pass
        lock.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Write exclusion audit; never import torch")
    mode.add_argument("--execute", action="store_true", help="Fit and predict after explicit freeze confirmation")
    parser.add_argument("--confirm-qc-membership-freeze", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--reliability-sample", type=Path, default=DEFAULT_RELIABILITY_SAMPLE)
    parser.add_argument("--human-reference", type=Path, default=DEFAULT_HUMAN_REFERENCE)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--membership-freeze", type=Path, default=DEFAULT_MEMBERSHIP_FREEZE)
    parser.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    parser.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prediction-output", type=Path, default=DEFAULT_PREDICTION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_config(args.config)
    if sha256_file(DEFAULT_M1_CONFIG) != EXPECTED_M1_CONFIG_SHA256:
        raise EvaluationBSupervisedError("Companion Evaluation B M1 config hash mismatch")
    prepared = prepare_supervised_data(
        benchmark_path=args.benchmark,
        ontology_path=args.ontology,
        reliability_sample_path=args.reliability_sample,
        human_reference_path=args.human_reference,
        membership_path=args.membership,
        membership_freeze_path=args.membership_freeze,
        a1_split_path=args.a1_split,
        a2_split_path=args.a2_split,
    )
    preflight = write_preflight_artifacts(
        prepared,
        audit_path=args.audit_output,
        preflight_path=args.preflight_output,
        config_hashes={
            "M1": sha256_file(DEFAULT_M1_CONFIG),
            "M2": sha256_file(args.config),
        },
    )
    if args.preflight:
        print(canonical_json({"mode": "PREFLIGHT", **preflight}))
        return 0
    if not args.confirm_qc_membership_freeze:
        raise EvaluationBSupervisedError(
            "Execution requires --confirm-qc-membership-freeze after root confirms the freeze"
        )
    result = execute(
        config=config,
        config_path=args.config,
        prepared=prepared,
        model_dir=args.model_dir,
        prediction_path=args.prediction_output,
        local_files_only=bool(args.local_files_only),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except EvaluationBSupervisedError as exc:
        print(f"Evaluation B M2 protocol error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
