#!/usr/bin/env python3
"""Run the frozen M2 ModernBERT joint 17-label AMP benchmark.

The Phase-4 M2 protocol uses one shared ``ModernBERT-base`` encoder with one
17-logit multilabel head.  For A1 and for each fresh A2 fold, the six frozen
learning-rate/weight-decay configurations are trained from the same pinned
pretrained revision.  Checkpoint and global-threshold selection use validation
data only; test probabilities are produced only after ``fit_state.json`` has
made both choices immutable.

The runner is resume-safe at completed grid-trial and completed fit-state
boundaries.  An interrupted grid trial is restarted from the pinned pretrained
model, never from another fold or trial.  Complete artifacts are skipped unless
``--force`` is supplied.  Transformers and PyTorch are imported lazily so that
protocol validation and the focused offline tests do not download a model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import random
import socket
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, f1_score


VERSION = "1.4.1"
ARTIFACT_SCHEMA_VERSION = "sherloc-m2-artifacts-v1"
PREDICTION_SCHEMA_VERSION = "sherloc-amp-predictions-v1"
EXPECTED_CONFIG_ID = "m2-modernbert-amp-v2"
EXPECTED_METHOD_ID = "M2"
EXPECTED_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_N = 1263
EXPECTED_BENCHMARK_SHA256 = (
    "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
)
EXPECTED_CONFIG_SHA256 = (
    "73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3"
)
EXPECTED_TOKEN_AUDIT_SHA256 = (
    "5b05015b44ed98ef6bebefc38ea1f445839cd5835a02a4025c10371efd961dd1"
)
EXPECTED_A1_SPLIT_SHA256 = (
    "63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d"
)
EXPECTED_A2_SPLIT_SHA256 = (
    "75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67"
)
EXPECTED_MODEL_ID = "answerdotai/ModernBERT-base"
EXPECTED_MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
EXPECTED_MAX_LENGTH = 2048
EXPECTED_TRUNCATED_CORPUS_N = 9
GRADIENT_CHECKPOINTING_ADDENDUM_ID = (
    "m2-hardware-memory-gradient-checkpointing-v1"
)
MPS_MEMORY_CONTROLS_ADDENDUM_ID = "m2-hardware-mps-memory-controls-v1"
MPS_LOW_WATERMARK_ENV = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
MPS_HIGH_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
MPS_LOW_WATERMARK_DEFAULT_REFERENCE = Decimal("1.4")
TRAIN_PROGRESS_INTERVAL_BATCHES = 100
BASELINE_THRESHOLD = 0.50
THRESHOLD_GRID = tuple(
    float(Decimal("0.20") + Decimal("0.05") * index) for index in range(13)
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_CONFIG = REPO_ROOT / "config/experiments/m2_modernbert_amp_v2.yaml"
DEFAULT_A1_SPLIT = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
DEFAULT_A2_SPLIT = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
DEFAULT_TOKEN_AUDIT = REPO_ROOT / "outputs/tables/modernbert_token_length_audit.csv"
DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs/models/m2"
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions/m2"
FIXED_A2_PROTOCOL_ID = "m2-modernbert-a2-fixed-a1-hparams-v1"
FIXED_A2_CONFIGURATION_INDEX = 5
FIXED_A2_HYPERPARAMETERS = {
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
}
DEFAULT_FIXED_A2_AMENDMENT = (
    REPO_ROOT / "docs/m2_a2_compute_contingency_amendment_v1.md"
)
EXPECTED_FIXED_A2_AMENDMENT_SHA256 = (
    "b83536e3b2cd8303f03b1977728c733e83a0599e1ed739e93846151ec29899ad"
)
LEGACY_FOLD1_METADATA_ARCHIVE = "legacy_grid_run_metadata_interrupted.json"


class M2ProtocolError(RuntimeError):
    """Raised when an input or artifact violates the frozen M2 protocol."""


def _canonical_mps_low_watermark_ratio(raw_ratio: Any) -> tuple[Decimal, str]:
    """Validate and canonicalize an opt-in MPS low-watermark ratio.

    The accepted interval is deliberately narrower than PyTorch's general
    parser: this runner permits only positive values strictly below the 1.4
    unified-memory default reference.  In particular, zero is rejected because
    PyTorch assigns it special "disable adaptive commit/garbage collection"
    semantics.
    """

    try:
        raw_text = str(raw_ratio).strip()
        if not raw_text:
            raise ValueError("empty value")
        ratio = Decimal(raw_text)
    except (ValueError, ArithmeticError) as error:
        raise M2ProtocolError(
            f"--mps-low-watermark-ratio must be a finite decimal, got {raw_ratio!r}"
        ) from error
    if not ratio.is_finite():
        raise M2ProtocolError("--mps-low-watermark-ratio must be finite")
    if not Decimal("0") < ratio < MPS_LOW_WATERMARK_DEFAULT_REFERENCE:
        raise M2ProtocolError(
            "--mps-low-watermark-ratio must be greater than 0 and strictly "
            "below the 1.4 default reference"
        )
    canonical = format(ratio.normalize(), "f")
    if "." not in canonical:
        canonical += ".0"
    return ratio, canonical


def configure_mps_allocator(
    raw_ratio: Any | None,
    *,
    environ: MutableMapping[str, str] | None = None,
    torch_already_imported: bool | None = None,
) -> dict[str, Any]:
    """Apply the opt-in low watermark before the first PyTorch import.

    Existing allocator environment overrides are rejected unless an existing
    low-watermark value exactly matches the explicit CLI request numerically.
    The high-watermark variable is never set by this runner and is rejected if
    inherited, preventing an unrecorded cap increase.
    """

    target = os.environ if environ is None else environ
    if MPS_HIGH_WATERMARK_ENV in target:
        raise M2ProtocolError(
            f"{MPS_HIGH_WATERMARK_ENV} is set in the environment; M2 prohibits "
            "all high-watermark overrides"
        )
    existing_low = target.get(MPS_LOW_WATERMARK_ENV)
    if raw_ratio is None:
        if MPS_LOW_WATERMARK_ENV in target:
            raise M2ProtocolError(
                f"{MPS_LOW_WATERMARK_ENV} is set without the explicit "
                "--mps-low-watermark-ratio option"
            )
        return {
            "mode": "DISABLED",
            "addendum_id": None,
            "low_watermark_environment_variable": MPS_LOW_WATERMARK_ENV,
            "requested_low_watermark_ratio": None,
            "requested_low_watermark_ratio_text": None,
            "applied_low_watermark_ratio_text": None,
            "application_source": "DISABLED",
            "applied_before_torch_import": None,
            "default_low_watermark_ratio_reference": float(
                MPS_LOW_WATERMARK_DEFAULT_REFERENCE
            ),
            "high_watermark_environment_variable": MPS_HIGH_WATERMARK_ENV,
            "high_watermark_override": None,
            "high_watermark_changes_prohibited": True,
        }

    already_imported = (
        "torch" in sys.modules
        if torch_already_imported is None
        else bool(torch_already_imported)
    )
    if already_imported:
        raise M2ProtocolError(
            "--mps-low-watermark-ratio must be applied before the first torch import; "
            "start a fresh Python process"
        )
    ratio, canonical = _canonical_mps_low_watermark_ratio(raw_ratio)
    source = "CLI_APPLIED"
    if existing_low is not None:
        try:
            existing_ratio = Decimal(existing_low.strip())
        except (ValueError, ArithmeticError) as error:
            raise M2ProtocolError(
                f"Inherited {MPS_LOW_WATERMARK_ENV} is not a valid decimal"
            ) from error
        if not existing_ratio.is_finite() or existing_ratio != ratio:
            raise M2ProtocolError(
                f"Inherited {MPS_LOW_WATERMARK_ENV}={existing_low!r} does not "
                f"match explicit --mps-low-watermark-ratio {canonical}"
            )
        source = "CLI_CONFIRMED_PREEXISTING"
    target[MPS_LOW_WATERMARK_ENV] = canonical
    return {
        "mode": "OPT_IN_LOW_WATERMARK",
        "addendum_id": MPS_MEMORY_CONTROLS_ADDENDUM_ID,
        "low_watermark_environment_variable": MPS_LOW_WATERMARK_ENV,
        "requested_low_watermark_ratio": float(ratio),
        "requested_low_watermark_ratio_text": canonical,
        "applied_low_watermark_ratio_text": target[MPS_LOW_WATERMARK_ENV],
        "application_source": source,
        "applied_before_torch_import": True,
        "default_low_watermark_ratio_reference": float(
            MPS_LOW_WATERMARK_DEFAULT_REFERENCE
        ),
        "high_watermark_environment_variable": MPS_HIGH_WATERMARK_ENV,
        "high_watermark_override": None,
        "high_watermark_changes_prohibited": True,
    }


def validate_runtime_mps_allocator(
    provenance: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify and return the allocator environment observed after torch import."""

    target = os.environ if environ is None else environ
    if MPS_HIGH_WATERMARK_ENV in target:
        raise M2ProtocolError(
            f"{MPS_HIGH_WATERMARK_ENV} appeared after validation; refusing execution"
        )
    expected = provenance.get("applied_low_watermark_ratio_text")
    actual = target.get(MPS_LOW_WATERMARK_ENV)
    if expected is None:
        if MPS_LOW_WATERMARK_ENV in target:
            raise M2ProtocolError(
                f"Unrecorded {MPS_LOW_WATERMARK_ENV} appeared after validation"
            )
    else:
        if actual is None:
            raise M2ProtocolError(
                f"Requested {MPS_LOW_WATERMARK_ENV} disappeared before torch runtime"
            )
        try:
            matches = Decimal(actual.strip()) == Decimal(str(expected))
        except (ValueError, ArithmeticError):
            matches = False
        if not matches:
            raise M2ProtocolError(
                f"Observed {MPS_LOW_WATERMARK_ENV}={actual!r} differs from "
                f"the recorded request {expected!r}"
            )
    return {
        MPS_LOW_WATERMARK_ENV: actual,
        MPS_HIGH_WATERMARK_ENV: None,
        "validated_against_pre_import_request": True,
    }


@dataclass(frozen=True)
class RunSpec:
    evaluation: str
    fold: int | None
    split_path: Path
    model_dir: Path
    prediction_path: Path

    @property
    def key(self) -> str:
        return "a1" if self.evaluation == "A1" else f"a2_fold_{self.fold}"


@dataclass(frozen=True)
class TokenInfo:
    original_token_count: int
    truncated: bool
    max_tokens_used: int


@dataclass(frozen=True)
class BatchAttempt:
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    effective_train_batch_size: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash directory file names and bytes in deterministic order."""

    if not path.is_dir():
        raise M2ProtocolError(f"Checkpoint directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise M2ProtocolError(f"Checkpoint directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M2ProtocolError(f"Value is not strict finite JSON: {error}") from error


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    try:
        payload = json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise M2ProtocolError(f"Refusing non-strict JSON for {path}: {error}") from error
    atomic_text(path, payload)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise M2ProtocolError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise M2ProtocolError(f"Refusing to write empty JSONL: {path}")
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class RunLock:
    path: Path
    token: str

    @classmethod
    def acquire(
        cls,
        spec: RunSpec,
        *,
        break_stale_lock: bool,
        execution_options: Mapping[str, Any],
    ) -> "RunLock":
        spec.model_dir.mkdir(parents=True, exist_ok=True)
        path = spec.model_dir / "run.lock.json"
        if path.exists():
            if not break_stale_lock:
                raise M2ProtocolError(
                    f"M2 run lock already exists for {spec.key}: {path}. "
                    "Confirm no process owns it, then use --break-stale-lock."
                )
            history = (
                spec.model_dir.parent
                / "_run_lock_history"
                / spec.key
                / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
            )
            history.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, history)
        token = os.urandom(16).hex()
        payload = {
            "lock_schema_version": "sherloc-m2-run-lock-v1",
            "token": token,
            "acquired_at": utc_now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "run": spec.key,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "execution_options": dict(execution_options),
        }
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise M2ProtocolError(f"Concurrent M2 run acquired lock: {path}") from error
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return cls(path=path, token=token)

    def release(self) -> None:
        if not self.path.exists():
            raise M2ProtocolError(f"M2 run lock disappeared: {self.path}")
        payload = load_json(self.path)
        if payload.get("token") != self.token:
            raise M2ProtocolError(f"M2 run lock ownership changed: {self.path}")
        self.path.unlink()


def _artifact_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(load_json(path).get("status") or "")
    except M2ProtocolError:
        return "UNREADABLE"


def archive_forced_restart(
    spec: RunSpec,
    *,
    context: Mapping[str, Any],
    force_reason: str,
) -> dict[str, Any]:
    """Recoverably archive an incomplete/failed run before explicit restart."""

    metadata_path = spec.model_dir / "run_metadata.json"
    fit_state_path = spec.model_dir / "fit_state.json"
    trial_states = sorted((spec.model_dir / "grid").glob("*/trial_state.json"))
    statuses = {
        display_path(path): _artifact_status(path)
        for path in (metadata_path, fit_state_path, *trial_states)
        if path.exists()
    }
    completed = [
        path
        for path, status in statuses.items()
        if status in {"COMPLETE", "FIT_AND_VALIDATION_SELECTION_COMPLETE"}
    ]
    if completed:
        raise M2ProtocolError(
            "--force refuses to replace COMPLETE M2 artifacts: "
            + ", ".join(completed)
        )

    history_dir = (
        spec.model_dir.parent
        / "_restart_history"
        / spec.key
        / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"{context_digest(context)[:12]}"
        )
    )
    history_dir.mkdir(parents=True, exist_ok=False)
    candidates = [
        spec.model_dir / "grid",
        spec.model_dir / "tokenizer",
        metadata_path,
        fit_state_path,
        spec.model_dir / "validation_hyperparameter_search.csv",
        spec.model_dir / "validation_threshold_search.csv",
        spec.prediction_path,
    ]
    archived: list[dict[str, Any]] = []
    for source in candidates:
        if not source.exists():
            continue
        digest = sha256_directory(source) if source.is_dir() else sha256_file(source)
        destination = history_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        archived.append(
            {
                "source_path": display_path(source),
                "archived_path": display_path(destination),
                "sha256": digest,
                "kind": "directory" if destination.is_dir() else "file",
            }
        )
    event = {
        "restart_schema_version": "sherloc-m2-forced-restart-v1",
        "recorded_at": utc_now(),
        "run": spec.key,
        "reason": force_reason,
        "previous_statuses": statuses,
        "new_execution_context_sha256": context_digest(context),
        "archived_artifacts": archived,
    }
    event_path = history_dir / "restart_event.json"
    atomic_json(event_path, event)
    return {
        "restart_event_path": display_path(event_path),
        "restart_event_sha256": sha256_file(event_path),
        "reason": force_reason,
    }


def archive_fold1_cpu_inference_for_mps_reexecution(
    spec: RunSpec,
    *,
    current_hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve completed Fold-1 CPU inference before MPS-only re-execution.

    This is intentionally narrower than ``--force``: it may archive only the
    fixed-protocol Fold-1 fit/selection derivatives and test predictions. The
    validated legacy grid, including the official C5 checkpoint, is never
    moved or changed.
    """

    if spec.evaluation != "A2" or spec.fold != 1:
        raise M2ProtocolError(
            "Technical inference re-execution is restricted to A2 Fold 1"
        )
    if (
        current_hardware.get("backend") != "mps"
        or current_hardware.get("mps_available") is not True
    ):
        raise M2ProtocolError(
            "Fold-1 technical inference re-execution requires available Apple MPS"
        )

    metadata_path = spec.model_dir / "run_metadata.json"
    fixed_dir = spec.model_dir / "fixed_a1_hparams_v1"
    prediction_path = spec.prediction_path
    for required in (metadata_path, fixed_dir, prediction_path):
        if not required.exists():
            raise M2ProtocolError(
                f"Fold-1 CPU inference artifact is missing: {required}"
            )

    metadata = load_json(metadata_path)
    protocol = metadata.get("scientific_protocol")
    selection = metadata.get("selection")
    old_hardware = metadata.get("execution_environment", {}).get("hardware", {})
    if (
        metadata.get("status") != "COMPLETE"
        or not isinstance(protocol, Mapping)
        or protocol.get("protocol_id") != FIXED_A2_PROTOCOL_ID
        or not isinstance(selection, Mapping)
        or selection.get("training_reused_without_retraining") is not True
        or selection.get("selected_configuration_index")
        != FIXED_A2_CONFIGURATION_INDEX
        or selection.get("selected_hyperparameters")
        != FIXED_A2_HYPERPARAMETERS
        or metadata.get("test_labels_used_for_selection") is not False
        or old_hardware.get("backend") != "cpu"
        or old_hardware.get("mps_available") is not False
    ):
        raise M2ProtocolError(
            "Existing Fold-1 artifact is not the eligible fixed-protocol CPU "
            "inference attempt"
        )

    fit_state_path = resolve_artifact_path(str(metadata.get("fit_state_path")))
    checkpoint_path = resolve_artifact_path(
        str(selection.get("selected_checkpoint_path"))
    )
    if fit_state_path.parent != fixed_dir or not fit_state_path.is_file():
        raise M2ProtocolError("Fold-1 CPU fit-state path is outside the fixed directory")
    if sha256_file(fit_state_path) != metadata.get("fit_state_sha256"):
        raise M2ProtocolError("Fold-1 CPU fit state is damaged")
    if sha256_file(prediction_path) != metadata.get("prediction_sha256"):
        raise M2ProtocolError("Fold-1 CPU prediction artifact is damaged")
    if (
        not checkpoint_path.is_dir()
        or sha256_directory(checkpoint_path)
        != selection.get("selected_checkpoint_sha256")
    ):
        raise M2ProtocolError("Official Fold-1 C5 checkpoint is missing or damaged")

    old_context = str(metadata.get("execution_context_sha256") or "")
    if len(old_context) != 64:
        raise M2ProtocolError("Fold-1 CPU metadata lacks its execution-context hash")
    history_dir = (
        spec.model_dir
        / "_protocol_history"
        / FIXED_A2_PROTOCOL_ID
        / "technical_reinference"
        / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"{old_context[:12]}"
        )
    )
    history_dir.mkdir(parents=True, exist_ok=False)
    candidates = (fixed_dir, prediction_path, metadata_path)
    archived: list[dict[str, Any]] = []
    for source in candidates:
        digest = sha256_directory(source) if source.is_dir() else sha256_file(source)
        destination = history_dir / source.name
        os.replace(source, destination)
        observed = (
            sha256_directory(destination)
            if destination.is_dir()
            else sha256_file(destination)
        )
        if observed != digest:
            raise M2ProtocolError(
                f"Fold-1 CPU inference archive verification failed: {destination}"
            )
        archived.append(
            {
                "source_path": display_path(source),
                "archived_path": display_path(destination),
                "sha256": digest,
                "kind": "directory" if destination.is_dir() else "file",
            }
        )
    event = {
        "archive_schema_version": "sherloc-m2-technical-reinference-v1",
        "recorded_at": utc_now(),
        "run": spec.key,
        "reason": "SANDBOX_HID_MPS_DURING_FOLD1_TEST_INFERENCE",
        "scope": "FOLD1_FIXED_DERIVATIVES_AND_TEST_INFERENCE_ONLY",
        "training_reexecuted": False,
        "legacy_grid_changed": False,
        "old_execution_context_sha256": old_context,
        "old_backend": "cpu",
        "required_new_backend": "mps",
        "archived_artifacts": archived,
    }
    event_path = history_dir / "archive_event.json"
    atomic_json(event_path, event)
    return {
        "archive_event_path": display_path(event_path),
        "archive_event_sha256": sha256_file(event_path),
        "reason": event["reason"],
        "training_reexecuted": False,
    }


def mark_run_terminal_failure(spec: RunSpec, error: BaseException) -> None:
    """Replace only this runner's IN_PROGRESS marker with a terminal status."""

    metadata_path = spec.model_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return
    try:
        metadata = load_json(metadata_path)
    except M2ProtocolError:
        return
    runner_source = metadata.get("runner_source")
    if (
        metadata.get("status") != "IN_PROGRESS"
        or not isinstance(runner_source, Mapping)
        or runner_source.get("sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        return
    metadata["status"] = (
        "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
    )
    metadata["terminal_at"] = utc_now()
    metadata["terminal_error_type"] = type(error).__name__
    metadata["terminal_error_message"] = str(error)[:1000]
    atomic_json(metadata_path, metadata)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise M2ProtocolError(f"Cannot read JSON document {path}: {error}") from error
    if not isinstance(value, dict):
        raise M2ProtocolError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"Non-finite JSON constant {token}")
                    ),
                )
                if not isinstance(value, dict):
                    raise M2ProtocolError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise M2ProtocolError(f"Cannot read JSONL {path}: {error}") from error
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise M2ProtocolError(f"Cannot read CSV {path}: {error}") from error


def ontology_label_order(ontology: Mapping[str, Any]) -> list[str]:
    order = [
        item["id"]
        for family in ("ACT", "MEANS", "PURPOSE")
        for item in ontology["families"][family]
    ]
    if len(order) != 17 or len(set(order)) != 17:
        raise M2ProtocolError("Ontology is not the frozen 5/6/6 AMP design")
    return order


def record_labels(record: Mapping[str, Any]) -> list[str]:
    targets = record["amp_targets"]
    return list(
        targets["act_ontology_ids"]
        + targets["means_ontology_ids"]
        + targets["purpose_ontology_ids"]
    )


def target_matrix(
    records: Sequence[Mapping[str, Any]], label_order: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [int(label in set(record_labels(record))) for label in label_order]
            for record in records
        ],
        dtype=np.float32,
    )


def validate_probability_matrix(
    probabilities: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
    artifact: str = "probabilities",
) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2 or (expected_shape is None and array.shape[1] != 17):
        raise M2ProtocolError(f"{artifact} has invalid shape {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise M2ProtocolError(
            f"{artifact} has shape {array.shape}; expected {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise M2ProtocolError(f"{artifact} contains NaN or infinite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise M2ProtocolError(f"{artifact} contains values outside [0, 1]")
    return array


def validate_binary_label_matrix(
    labels: Any, *, expected_shape: tuple[int, int], artifact: str
) -> np.ndarray:
    array = np.asarray(labels, dtype=np.float32)
    if array.shape != expected_shape:
        raise M2ProtocolError(
            f"{artifact} has shape {array.shape}; expected {expected_shape}"
        )
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise M2ProtocolError(f"{artifact} is not a finite binary matrix")
    return array


def texts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [str(record["text_input"]["english_fact_summary_raw"]) for record in records]
    if any(not value.strip() for value in result):
        raise M2ProtocolError("M2 encountered an empty English Fact Summary")
    return result


def membership_digest(rows: Iterable[Sequence[Any]]) -> str:
    payload = "".join("\t".join(map(str, row)) + "\n" for row in rows)
    return sha256_text(payload)


def parameter_grid(config: Mapping[str, Any]) -> list[dict[str, float]]:
    raw = config["validation_search"]["grid"]
    if set(raw) != {"learning_rate", "weight_decay"}:
        raise M2ProtocolError(f"Unexpected M2 hyperparameter grid: {sorted(raw)}")
    combinations = [
        {"learning_rate": float(learning_rate), "weight_decay": float(weight_decay)}
        for learning_rate, weight_decay in itertools.product(
            raw["learning_rate"], raw["weight_decay"]
        )
    ]
    maximum = int(config["validation_search"]["maximum_configurations"])
    if len(combinations) != 6 or len(combinations) > maximum:
        raise M2ProtocolError(
            f"Frozen M2 grid must contain exactly six configurations, got {len(combinations)}"
        )
    return combinations


def configured_threshold_grid(config: Mapping[str, Any]) -> tuple[float, ...]:
    raw = config["thresholding"]
    start = Decimal(str(raw["candidate_grid_start"]))
    stop = Decimal(str(raw["candidate_grid_stop"]))
    step = Decimal(str(raw["candidate_grid_step"]))
    return tuple(float(start + step * index) for index in range(int((stop - start) / step) + 1))


def batch_attempts(
    config: Mapping[str, Any], *, initial_train_batch_size: int = 4
) -> list[BatchAttempt]:
    raw = config["training"]
    target = int(raw["effective_train_batch_size_target"])
    initial_eval = int(raw["per_device_eval_batch_size_initial"])
    batches = [int(value) for value in raw["memory_fallback_batch_sizes"]]
    if batches != [4, 2, 1]:
        raise M2ProtocolError("M2 memory fallback order must be 4, 2, 1")
    if initial_train_batch_size not in batches:
        raise M2ProtocolError("Initial M2 train batch size must be one of 4, 2, or 1")
    batches = batches[batches.index(initial_train_batch_size) :]
    result: list[BatchAttempt] = []
    for batch_size in batches:
        if target % batch_size:
            raise M2ProtocolError(
                "Effective batch-size target must be divisible by every fallback batch"
            )
        gradient_accumulation = target // batch_size
        result.append(
            BatchAttempt(
                train_batch_size=batch_size,
                eval_batch_size=min(initial_eval, batch_size * 2),
                gradient_accumulation_steps=gradient_accumulation,
                effective_train_batch_size=batch_size * gradient_accumulation,
            )
        )
    if initial_train_batch_size == 4 and result[0].gradient_accumulation_steps != int(
        raw["gradient_accumulation_steps_initial"]
    ):
        raise M2ProtocolError("Initial gradient accumulation differs from config")
    return result


def accumulation_window_case_count(
    *,
    dataset_n: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    batch_index_one_based: int,
) -> int:
    """Return cases in the current optimizer window, including a remainder."""

    if min(
        dataset_n,
        micro_batch_size,
        gradient_accumulation_steps,
        batch_index_one_based,
    ) <= 0:
        raise M2ProtocolError("Accumulation-window inputs must be positive")
    first_batch = (
        (batch_index_one_based - 1) // gradient_accumulation_steps
    ) * gradient_accumulation_steps
    first_case = first_batch * micro_batch_size
    last_case = min(
        dataset_n,
        (first_batch + gradient_accumulation_steps) * micro_batch_size,
    )
    window_n = last_case - first_case
    if window_n <= 0:
        raise M2ProtocolError("Batch index is outside the accumulation dataset")
    return window_n


def validate_static_inputs(
    benchmark_path: Path,
    ontology_path: Path,
    config_path: Path,
    token_audit_path: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any], dict[int, TokenInfo]]:
    required = (benchmark_path, ontology_path, config_path, token_audit_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise M2ProtocolError(f"Missing frozen M2 inputs: {missing}")
    if sha256_file(benchmark_path) != EXPECTED_BENCHMARK_SHA256:
        raise M2ProtocolError("Frozen benchmark hash changed")
    if sha256_file(ontology_path) != EXPECTED_ONTOLOGY_SHA256:
        raise M2ProtocolError("Frozen ontology hash changed")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise M2ProtocolError("Frozen Phase-4 M2 config hash changed")
    if sha256_file(token_audit_path) != EXPECTED_TOKEN_AUDIT_SHA256:
        raise M2ProtocolError("Frozen ModernBERT token-audit hash changed")

    benchmark = load_jsonl(benchmark_path)
    ontology = load_json(ontology_path)
    config = load_json(config_path)
    label_order = ontology_label_order(ontology)
    if len(benchmark) != EXPECTED_N:
        raise M2ProtocolError(f"Expected {EXPECTED_N} benchmark rows, got {len(benchmark)}")
    if any(row.get("primary_cohort_id") != EXPECTED_COHORT_ID for row in benchmark):
        raise M2ProtocolError("Primary cohort ID mismatch")
    by_rank = {int(row["identity"]["search_rank"]): row for row in benchmark}
    if len(by_rank) != EXPECTED_N:
        raise M2ProtocolError("Benchmark search ranks are not unique")
    if len({row["identity"]["canonical_url"] for row in benchmark}) != EXPECTED_N:
        raise M2ProtocolError("Benchmark canonical URLs are not unique")

    if config.get("config_id") != EXPECTED_CONFIG_ID or config.get("method_id") != EXPECTED_METHOD_ID:
        raise M2ProtocolError("Unexpected M2 config identity")
    if config.get("status") != "FROZEN_PHASE4_EXECUTION_READY":
        raise M2ProtocolError("M2 config is not frozen for Phase-4 execution")
    if config.get("primary_cohort_id") != EXPECTED_COHORT_ID:
        raise M2ProtocolError("M2 config cohort ID mismatch")
    if list(config["targets"]["label_order"]) != label_order:
        raise M2ProtocolError("M2 config label order differs from frozen ontology")
    model = config["model"]
    if (
        model.get("pretrained_model_id") != EXPECTED_MODEL_ID
        or model.get("tokenizer_id") != EXPECTED_MODEL_ID
        or model.get("revision") != EXPECTED_MODEL_REVISION
        or model.get("tokenizer_revision") != EXPECTED_MODEL_REVISION
    ):
        raise M2ProtocolError("M2 model/tokenizer ID or pinned revision changed")
    if (
        int(model.get("shared_encoder_count", 0)) != 1
        or int(model.get("classification_head_count", 0)) != 1
        or int(model.get("num_labels", 0)) != 17
    ):
        raise M2ProtocolError("M2 must use one shared encoder and one 17-logit head")
    tokenization = config["tokenization"]
    if int(tokenization.get("max_length", 0)) != EXPECTED_MAX_LENGTH:
        raise M2ProtocolError("M2 max_length must remain 2048")
    if not tokenization.get("truncation") or tokenization.get("truncation_side") != "right":
        raise M2ProtocolError("Unexpected M2 truncation policy")
    training = config["training"]
    if int(training.get("epochs_max", 0)) != 6 or int(training.get("early_stopping_patience", -1)) != 2:
        raise M2ProtocolError("M2 epoch or patience policy changed")
    if int(training.get("seed", -1)) != int(config["random_seed"]):
        raise M2ProtocolError("M2 seed is inconsistent")
    if config["validation_search"].get("selection_data") != "VALIDATION_ONLY":
        raise M2ProtocolError("M2 hyperparameter selection is not validation-only")
    if config["validation_search"].get("selection_metric") != "macro_average_precision":
        raise M2ProtocolError("M2 checkpoint selection metric changed")
    if not config["validation_search"].get("test_labels_forbidden"):
        raise M2ProtocolError("M2 config does not prohibit test-label tuning")
    if configured_threshold_grid(config) != THRESHOLD_GRID:
        raise M2ProtocolError("M2 threshold grid is not 0.20..0.80 by 0.05")
    if float(config["thresholding"].get("fixed_baseline")) != BASELINE_THRESHOLD:
        raise M2ProtocolError("M2 fixed sensitivity threshold is not 0.50")
    if config["thresholding"].get("test_label_tuning") != "PROHIBITED":
        raise M2ProtocolError("M2 config does not prohibit test threshold tuning")
    if not config["reproducibility"].get("fresh_pretrained_initialization_per_a1_or_a2_fold"):
        raise M2ProtocolError("M2 does not require fresh initialization per evaluation")
    if config["reproducibility"].get("share_weights_across_a2_folds"):
        raise M2ProtocolError("M2 config would share weights across A2 folds")
    expected_splits = {
        "data/splits/a1_iid_split_final_v1.csv",
        "data/splits/a2_jurisdiction_folds_final_v1.csv",
    }
    if set(config["reproducibility"]["split_files"]) != expected_splits:
        raise M2ProtocolError("M2 config does not reference both final split files")
    parameter_grid(config)
    batch_attempts(config)

    audit_rows = load_csv(token_audit_path)
    if len(audit_rows) != EXPECTED_N:
        raise M2ProtocolError(
            f"Frozen token audit must have {EXPECTED_N} rows, got {len(audit_rows)}"
        )
    token_info: dict[int, TokenInfo] = {}
    for row in audit_rows:
        rank = int(row["search_rank"])
        if rank not in by_rank or row["canonical_url"] != by_rank[rank]["identity"]["canonical_url"]:
            raise M2ProtocolError(f"Token-audit identity mismatch at rank {rank}")
        if (
            row["tokenizer_model_id"] != EXPECTED_MODEL_ID
            or row["tokenizer_revision"] != EXPECTED_MODEL_REVISION
        ):
            raise M2ProtocolError(f"Token-audit tokenizer drift at rank {rank}")
        count = int(row["modernbert_token_count_with_special_tokens"])
        truncated = count > EXPECTED_MAX_LENGTH
        if bool(int(row["fully_covered_at_2048"])) == truncated:
            raise M2ProtocolError(f"Token-audit 2048 coverage flag mismatch at rank {rank}")
        token_info[rank] = TokenInfo(count, truncated, min(count, EXPECTED_MAX_LENGTH))
    if len(token_info) != EXPECTED_N:
        raise M2ProtocolError("Token-audit search ranks are not unique")
    if sum(item.truncated for item in token_info.values()) != EXPECTED_TRUNCATED_CORPUS_N:
        raise M2ProtocolError("Frozen corpus no longer has exactly nine truncated cases")
    return benchmark, label_order, config, token_info


def normalize_role(row: Mapping[str, str]) -> str:
    return (row.get("split") or row.get("role") or "").strip().upper()


def validate_and_partition_split(
    spec: RunSpec,
    split_rows: Sequence[dict[str, str]],
    benchmark: Sequence[dict[str, Any]],
    label_order: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not split_rows:
        raise M2ProtocolError(f"Empty final split file: {spec.split_path}")
    expected_split_sha256 = (
        EXPECTED_A1_SPLIT_SHA256
        if spec.evaluation == "A1"
        else EXPECTED_A2_SPLIT_SHA256
    )
    observed_split_sha256 = sha256_file(spec.split_path)
    if observed_split_sha256 != expected_split_sha256:
        raise M2ProtocolError(
            f"{spec.evaluation} frozen split hash changed: "
            f"{observed_split_sha256}"
        )
    selected = (
        [row for row in split_rows if int(row.get("fold_id", "0")) == spec.fold]
        if spec.evaluation == "A2"
        else list(split_rows)
    )
    if len(selected) != EXPECTED_N:
        raise M2ProtocolError(f"{spec.key} must contain {EXPECTED_N} rows, got {len(selected)}")
    statuses = {row.get("split_status", "").strip().upper() for row in selected}
    if not statuses or any("PROVISIONAL" in status or "FINAL" not in status for status in statuses):
        raise M2ProtocolError(f"Refusing non-final split: {spec.split_path}")

    by_rank = {int(row["identity"]["search_rank"]): row for row in benchmark}
    observed_ranks = [int(row["search_rank"]) for row in selected]
    if len(set(observed_ranks)) != EXPECTED_N or set(observed_ranks) != set(by_rank):
        raise M2ProtocolError(f"{spec.key} split membership differs from cohort")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    selected_sorted = sorted(selected, key=lambda row: int(row["search_rank"]))
    for split_row in selected_sorted:
        rank = int(split_row["search_rank"])
        record = by_rank[rank]
        identity = record["identity"]
        if split_row["canonical_url"] != identity["canonical_url"]:
            raise M2ProtocolError(f"Canonical URL mismatch for rank {rank}")
        if split_row["jurisdiction"] != identity["jurisdiction_country_raw"]:
            raise M2ProtocolError(f"Jurisdiction mismatch for rank {rank}")
        actual = set(record_labels(record))
        if any(int(split_row[label]) != int(label in actual) for label in label_order):
            raise M2ProtocolError(f"Split target columns differ at rank {rank}")
        role = normalize_role(split_row)
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "VALIDATION":
            validation.append(record)
        elif role == "TEST":
            test.append(record)
        elif split_row.get("effective_supervised_train", "0").strip() == "1":
            train.append(record)
        else:
            raise M2ProtocolError(f"Rank {rank} has unsupported non-training role {role!r}")
    if not train or not validation or not test:
        raise M2ProtocolError(
            f"{spec.key} has empty train/validation/test partition: "
            f"{len(train)}/{len(validation)}/{len(test)}"
        )
    partition_ranks = [
        {int(row["identity"]["search_rank"]) for row in partition}
        for partition in (train, validation, test)
    ]
    if any(partition_ranks[i] & partition_ranks[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise M2ProtocolError(f"Partition leakage in {spec.key}")

    jurisdiction_sets = {
        name: {
            str(row["identity"]["jurisdiction_country_raw"])
            for row in partition
        }
        for name, partition in (
            ("TRAIN", train),
            ("VALIDATION", validation),
            ("TEST", test),
        )
    }
    heldout_overlap = jurisdiction_sets["TEST"] & (
        jurisdiction_sets["TRAIN"] | jurisdiction_sets["VALIDATION"]
    )
    if spec.evaluation == "A2" and heldout_overlap:
        raise M2ProtocolError(
            f"Held-out jurisdiction leakage in {spec.key}: "
            f"{sorted(heldout_overlap)}"
        )

    digest = membership_digest(
        (
            int(row["search_rank"]),
            row["canonical_url"],
            normalize_role(row),
            row.get("effective_supervised_train", ""),
            row.get("fold_id", ""),
        )
        for row in selected_sorted
    )
    return train, validation, test, {
        "split_file_sha256": sha256_file(spec.split_path),
        "split_membership_sha256": digest,
        "role_counts": dict(sorted(role_counts.items())),
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "train_jurisdictions": sorted(jurisdiction_sets["TRAIN"]),
        "validation_jurisdictions": sorted(jurisdiction_sets["VALIDATION"]),
        "test_jurisdictions": sorted(jurisdiction_sets["TEST"]),
        "heldout_jurisdiction_leakage_n": len(heldout_overlap),
    }


def macro_average_precision(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, list[float | None], list[int]]:
    probabilities = validate_probability_matrix(
        probabilities,
        expected_shape=tuple(y_true.shape),
        artifact="macro-average-precision probabilities",
    )
    y_true = validate_binary_label_matrix(
        y_true,
        expected_shape=tuple(probabilities.shape),
        artifact="macro-average-precision labels",
    )
    supports = y_true.sum(axis=0).astype(int).tolist()
    scores: list[float | None] = []
    for index, support in enumerate(supports):
        scores.append(
            None
            if support == 0
            else float(average_precision_score(y_true[:, index], probabilities[:, index]))
        )
    defined = [score for score in scores if score is not None]
    if not defined:
        raise M2ProtocolError("Macro average precision is undefined for every label")
    macro_ap = float(np.mean(defined))
    if not math.isfinite(macro_ap):
        raise M2ProtocolError("Validation macro average precision is non-finite")
    return macro_ap, scores, supports


def macro_f1(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    probabilities = validate_probability_matrix(
        probabilities,
        expected_shape=tuple(y_true.shape),
        artifact="macro-F1 probabilities",
    )
    prediction = (probabilities >= threshold).astype(np.int8)
    value = float(f1_score(y_true, prediction, average="macro", zero_division=0))
    if not math.isfinite(value):
        raise M2ProtocolError("Validation macro-F1 is non-finite")
    return value


def select_global_threshold(
    y_validation: np.ndarray, probabilities: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    curve = [
        {
            "threshold": threshold,
            "validation_macro_f1": macro_f1(y_validation, probabilities, threshold),
        }
        for threshold in THRESHOLD_GRID
    ]
    winner = min(
        curve,
        key=lambda row: (
            -row["validation_macro_f1"],
            abs(row["threshold"] - BASELINE_THRESHOLD),
            row["threshold"],
        ),
    )
    return float(winner["threshold"]), curve


def execution_context(
    spec: RunSpec,
    benchmark_path: Path,
    ontology_path: Path,
    config_path: Path,
    token_audit_path: Path,
    split_metadata: Mapping[str, Any],
    label_order: Sequence[str],
    *,
    gradient_checkpointing: bool,
    initial_train_batch_size: int,
    execution_environment: Mapping[str, Any],
    max_length: int,
    max_length_override_acknowledged: bool,
    technical_override_rationale: str | None,
    mps_allocator: Mapping[str, Any] | None = None,
    adamw_foreach_false: bool = False,
    pad_to_multiple_of: int | None = None,
    scientific_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attempts = batch_attempts(
        load_json(config_path), initial_train_batch_size=initial_train_batch_size
    )
    environment = dict(execution_environment)
    allocator = dict(
        mps_allocator
        if mps_allocator is not None
        else configure_mps_allocator(
            None, environ={}, torch_already_imported=False
        )
    )
    if pad_to_multiple_of not in {None, 64}:
        raise M2ProtocolError("M2 padding multiple must be absent or exactly 64")
    if pad_to_multiple_of is not None and max_length % pad_to_multiple_of:
        raise M2ProtocolError(
            "M2 max length must be divisible by the requested padding multiple"
        )
    runner_path = Path(__file__).resolve()
    context = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "runner_version": VERSION,
        "runner_source": {
            "path": display_path(runner_path),
            "sha256": sha256_file(runner_path),
        },
        "execution_environment": environment,
        "execution_environment_sha256": sha256_text(canonical_json(environment)),
        "method_id": EXPECTED_METHOD_ID,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "primary_cohort_id": EXPECTED_COHORT_ID,
        "label_order": list(label_order),
        "benchmark_path": display_path(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "ontology_path": display_path(ontology_path),
        "ontology_sha256": sha256_file(ontology_path),
        "config_path": display_path(config_path),
        "config_sha256": sha256_file(config_path),
        "token_audit_path": display_path(token_audit_path),
        "token_audit_sha256": sha256_file(token_audit_path),
        **split_metadata,
        "threshold_grid": list(THRESHOLD_GRID),
        "baseline_threshold": BASELINE_THRESHOLD,
        "test_labels_used_for_selection": False,
        "mps_allocator": allocator,
        "technical_execution_options": {
            "addendum_id": (
                GRADIENT_CHECKPOINTING_ADDENDUM_ID
                if gradient_checkpointing
                else None
            ),
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_checkpointing_kwargs": (
                {"use_reentrant": False} if gradient_checkpointing else None
            ),
            "model_use_cache_policy": (
                "SET_FALSE_IF_CONFIG_ATTRIBUTE_EXISTS_ELSE_NOT_APPLICABLE"
                if gradient_checkpointing
                else "UNCHANGED_MODEL_DEFAULT"
            ),
            "initial_train_batch_size": initial_train_batch_size,
            "batch_fallback_sequence": [
                attempt.train_batch_size for attempt in attempts
            ],
            "effective_train_batch_size_target": 16,
            "max_length": max_length,
            "max_length_reduced": max_length < EXPECTED_MAX_LENGTH,
            "max_length_reduction_addendum_id": (
                "m2-hardware-max-length-reduction-v1"
                if max_length < EXPECTED_MAX_LENGTH
                else None
            ),
            "max_length_override_acknowledged": (
                max_length_override_acknowledged
            ),
            "max_length_override_rationale": technical_override_rationale,
            "selection_or_tuning_change": scientific_protocol is not None,
            "mixed_precision_policy": "CUDA_FP16_MPS_BF16_CPU_FP32",
            "gradient_scaler_policy": "CUDA_ONLY",
            "mps_memory_controls_addendum_id": (
                MPS_MEMORY_CONTROLS_ADDENDUM_ID
                if (
                    allocator["mode"] != "DISABLED"
                    or adamw_foreach_false
                    or pad_to_multiple_of is not None
                )
                else None
            ),
            "mps_allocator": allocator,
            "adamw_optimizer": "torch.optim.AdamW",
            "adamw_foreach_mode": (
                "EXPLICIT_FALSE" if adamw_foreach_false else "PYTORCH_DEFAULT"
            ),
            "adamw_foreach": False if adamw_foreach_false else None,
            "adamw_algorithm_or_hyperparameter_change": False,
            "pad_to_multiple_of": pad_to_multiple_of,
            "padding_policy": (
                "DYNAMIC_TO_LONGEST_ROUNDED_TO_MULTIPLE_64"
                if pad_to_multiple_of == 64
                else "DYNAMIC_TO_LONGEST_IN_BATCH"
            ),
            "padding_side_required": (
                "right" if pad_to_multiple_of is not None else None
            ),
            "non_padding_token_content_change": False,
            "truncation_policy_change": False,
            "tensor_shape_or_padding_change": pad_to_multiple_of is not None,
            "progress_interval_batches": TRAIN_PROGRESS_INTERVAL_BATCHES,
            "progress_fields": ["epoch", "batch", "batch_total", "elapsed_seconds"],
            "progress_semantic_data_logged": False,
        },
    }
    if scientific_protocol is not None:
        context["experiment_tag"] = FIXED_A2_PROTOCOL_ID
        context["scientific_protocol"] = dict(scientific_protocol)
    return context


def context_digest(context: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(context))


def load_ml_stack() -> dict[str, Any]:
    """Load heavyweight training dependencies only for a real execution."""

    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except ImportError as error:  # pragma: no cover - environment-specific.
        raise M2ProtocolError(
            "M2 execution requires torch, transformers, and accelerate in xihao_env"
        ) from error
    return {
        "torch": torch,
        "transformers": transformers,
        "DataLoader": DataLoader,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def select_device(torch: Any) -> tuple[Any, dict[str, Any]]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
        device_name = "Apple Metal Performance Shaders"
    else:
        device = torch.device("cpu")
        device_name = platform.processor() or platform.machine() or "CPU"
    return device, {
        "backend": device.type,
        "device": str(device),
        "device_name": device_name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def seed_everything(stack: Mapping[str, Any], seed: int) -> None:
    torch = stack["torch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    stack["transformers"].set_seed(seed)


def clear_device_memory(torch: Any, device: Any) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def is_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        marker in message
        for marker in (
            "out of memory",
            "mps backend out of memory",
            "not enough memory",
            "can't allocate memory",
            "cannot allocate memory",
        )
    )


class EncodedDataset:
    """Minimal framework-neutral dataset; the collator creates torch tensors."""

    def __init__(self, encodings: Mapping[str, Sequence[Any]], labels: np.ndarray):
        self.encodings = {key: list(value) for key, value in encodings.items()}
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": {key: value[index] for key, value in self.encodings.items()},
            "labels": self.labels[index],
        }


def encode_records(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    label_order: Sequence[str],
    max_length: int,
) -> EncodedDataset:
    encodings = tokenizer(
        texts(records),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return EncodedDataset(encodings, target_matrix(records, label_order))


def make_collator(
    tokenizer: Any,
    torch: Any,
    *,
    pad_to_multiple_of: int | None = None,
    max_length: int | None = None,
) -> Any:
    if pad_to_multiple_of not in {None, 64}:
        raise M2ProtocolError("M2 padding multiple must be absent or exactly 64")
    if pad_to_multiple_of is not None and (
        max_length is None or max_length % pad_to_multiple_of
    ):
        raise M2ProtocolError(
            "Rounded dynamic padding requires a divisible explicit max length"
        )
    if pad_to_multiple_of is not None:
        if getattr(tokenizer, "padding_side", None) != "right":
            raise M2ProtocolError(
                "The 64-token padding mode requires the pinned tokenizer's "
                "right-padding behavior"
            )
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise M2ProtocolError(
                "The 64-token padding mode requires a defined tokenizer pad token"
            )

    def collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if max_length is not None and any(
            len(item["features"]["input_ids"]) > max_length for item in items
        ):
            raise M2ProtocolError("Encoded sequence exceeds the recorded max length")
        batch = tokenizer.pad(
            [item["features"] for item in items],
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        if max_length is not None and int(batch["input_ids"].shape[-1]) > max_length:
            raise M2ProtocolError(
                "Collator padding exceeded the recorded M2 max length"
            )
        batch["labels"] = torch.as_tensor(
            np.stack([item["labels"] for item in items]), dtype=torch.float32
        )
        return batch

    return collate


def make_loader(
    stack: Mapping[str, Any],
    dataset: EncodedDataset,
    tokenizer: Any,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    device: Any,
    pad_to_multiple_of: int | None = None,
    max_length: int | None = None,
) -> Any:
    torch = stack["torch"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return stack["DataLoader"](
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=make_collator(
            tokenizer,
            torch,
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_length,
        ),
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


def move_batch(batch: Mapping[str, Any], device: Any) -> tuple[dict[str, Any], Any]:
    labels = batch["labels"].to(device)
    inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    return inputs, labels


def autocast_context(torch: Any, device: Any, mixed_precision: bool) -> Any:
    if mixed_precision and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision and device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    return nullcontext()


def mixed_precision_dtype(device: Any, mixed_precision: bool) -> str:
    if not mixed_precision:
        return "float32"
    if device.type == "cuda":
        return "float16"
    if device.type == "mps":
        return "bfloat16"
    raise M2ProtocolError(
        f"Mixed precision was enabled on unsupported device type {device.type!r}"
    )


def precision_settings(device: Any, configured: bool) -> dict[str, Any]:
    enabled = bool(configured) and device.type in {"cuda", "mps"}
    return {
        "mixed_precision": enabled,
        "mixed_precision_dtype": mixed_precision_dtype(device, enabled),
        "gradient_scaler_enabled": enabled and device.type == "cuda",
    }


def evaluate_model(
    stack: Mapping[str, Any],
    model: Any,
    loader: Any,
    device: Any,
    *,
    mixed_precision: bool,
    pass_labels_to_model: bool = True,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    torch = stack["torch"]
    model.eval()
    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    loss_sum = 0.0
    case_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = move_batch(batch, device)
            with autocast_context(torch, device, mixed_precision):
                output = (
                    model(**inputs, labels=labels)
                    if pass_labels_to_model
                    else model(**inputs)
                )
            probabilities = torch.sigmoid(output.logits).detach().float().cpu().numpy()
            validate_probability_matrix(
                probabilities,
                expected_shape=(int(labels.shape[0]), 17),
                artifact="model probability batch",
            )
            probability_batches.append(probabilities)
            label_batches.append(labels.detach().float().cpu().numpy())
            batch_n = int(labels.shape[0])
            if pass_labels_to_model:
                batch_loss = float(output.loss.detach().float().cpu())
                if not math.isfinite(batch_loss):
                    raise M2ProtocolError("Model evaluation loss is non-finite")
                loss_sum += batch_loss * batch_n
            case_count += batch_n
    if not case_count:
        raise M2ProtocolError("Evaluation loader is empty")
    combined_probabilities = validate_probability_matrix(
        np.concatenate(probability_batches, axis=0),
        artifact="model probabilities",
    )
    combined_labels = validate_binary_label_matrix(
        np.concatenate(label_batches, axis=0),
        expected_shape=tuple(combined_probabilities.shape),
        artifact="model labels",
    )
    return (
        combined_probabilities,
        combined_labels,
        loss_sum / case_count if pass_labels_to_model else None,
    )


def predict_with_batch_fallback(
    stack: Mapping[str, Any],
    model: Any,
    dataset: EncodedDataset,
    tokenizer: Any,
    device: Any,
    *,
    initial_batch_size: int,
    seed: int,
    mixed_precision: bool,
    max_length: int,
    pad_to_multiple_of: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Predict without labels entering the model, reducing only batch size on OOM."""

    candidates: list[int] = []
    current = max(1, int(initial_batch_size))
    while current >= 1:
        if current not in candidates:
            candidates.append(current)
        if current == 1:
            break
        current = max(1, current // 2)
    attempts: list[dict[str, Any]] = []
    for batch_size in candidates:
        loader = make_loader(
            stack,
            dataset,
            tokenizer,
            batch_size,
            shuffle=False,
            seed=seed,
            device=device,
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_length,
        )
        try:
            probabilities, labels, _ = evaluate_model(
                stack,
                model,
                loader,
                device,
                mixed_precision=mixed_precision,
                pass_labels_to_model=False,
            )
            attempts.append({"batch_size": batch_size, "status": "COMPLETE"})
            return probabilities, labels, attempts
        except Exception as error:
            if not is_out_of_memory(error):
                raise
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "OUT_OF_MEMORY",
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],
                }
            )
            clear_device_memory(stack["torch"], device)
    raise M2ProtocolError(
        f"M2 test inference exhausted batch-size fallback through 1 at "
        f"max_length={max_length}"
    )


def resolved_commit(value: Any) -> str | None:
    candidates = [
        getattr(value, "_commit_hash", None),
        getattr(getattr(value, "config", None), "_commit_hash", None),
    ]
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        candidates.append(init_kwargs.get("_commit_hash"))
        candidates.append(init_kwargs.get("revision"))
    return next((str(item) for item in candidates if item), None)


def configure_gradient_checkpointing(
    model: Any, *, enabled: bool
) -> dict[str, Any]:
    """Apply the frozen hardware-memory addendum without changing model inputs.

    ``use_reentrant=False`` is explicit so checkpoint recomputation behavior is
    stable across PyTorch invocations.  ``use_cache`` is disabled only in this
    mode because decoder-style key/value caches are incompatible with gradient
    checkpointing and provide no benefit to this classification training.
    """

    supports = bool(getattr(model, "supports_gradient_checkpointing", False))
    has_use_cache = hasattr(model.config, "use_cache")
    previous_use_cache = getattr(model.config, "use_cache", None)
    if not enabled:
        return {
            "gradient_checkpointing": False,
            "gradient_checkpointing_addendum_id": None,
            "gradient_checkpointing_kwargs": None,
            "supports_gradient_checkpointing": supports,
            "model_config_has_use_cache": has_use_cache,
            "model_use_cache_action": "UNCHANGED",
            "model_use_cache_before": previous_use_cache,
            "model_use_cache_during_training": previous_use_cache,
        }
    if not supports or not callable(
        getattr(model, "gradient_checkpointing_enable", None)
    ):
        raise M2ProtocolError(
            "Pinned M2 model does not support the requested gradient-checkpointing addendum"
        )
    if has_use_cache:
        model.config.use_cache = False
        use_cache_action = "SET_FALSE"
    else:
        use_cache_action = "NOT_APPLICABLE_CONFIG_HAS_NO_ATTRIBUTE"
    kwargs = {"use_reentrant": False}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    observed = getattr(model, "is_gradient_checkpointing", None)
    if observed is False:
        raise M2ProtocolError("Gradient checkpointing was requested but did not enable")
    return {
        "gradient_checkpointing": True,
        "gradient_checkpointing_addendum_id": GRADIENT_CHECKPOINTING_ADDENDUM_ID,
        "gradient_checkpointing_kwargs": kwargs,
        "supports_gradient_checkpointing": supports,
        "model_reports_gradient_checkpointing": observed,
        "model_config_has_use_cache": has_use_cache,
        "model_use_cache_action": use_cache_action,
        "model_use_cache_before": previous_use_cache,
        "model_use_cache_during_training": getattr(model.config, "use_cache", None),
    }


def initialize_pretrained_model(
    stack: Mapping[str, Any],
    config: Mapping[str, Any],
    label_order: Sequence[str],
    device: Any,
    *,
    local_files_only: bool,
    gradient_checkpointing: bool,
) -> tuple[Any, str | None, dict[str, Any]]:
    raw = config["model"]
    model = stack["AutoModelForSequenceClassification"].from_pretrained(
        raw["pretrained_model_id"],
        revision=raw["revision"],
        num_labels=len(label_order),
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(label_order)},
        label2id={label: index for index, label in enumerate(label_order)},
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if int(model.config.num_labels) != 17:
        raise M2ProtocolError("Loaded M2 model does not have exactly 17 logits")
    checkpointing_metadata = configure_gradient_checkpointing(
        model, enabled=gradient_checkpointing
    )
    commit = resolved_commit(model)
    # The requested revision is itself an immutable commit SHA; some current
    # Transformers classes do not retain it on the instantiated object.
    commit = commit or str(raw["revision"])
    if commit is not None and commit != EXPECTED_MODEL_REVISION:
        raise M2ProtocolError(f"Resolved model commit differs from pinned revision: {commit}")
    model.to(device)
    return model, commit, checkpointing_metadata


def load_tokenizer(
    stack: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    local_files_only: bool,
) -> tuple[Any, str | None]:
    raw = config["model"]
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        raw["tokenizer_id"],
        revision=raw["tokenizer_revision"],
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer.truncation_side = config["tokenization"]["truncation_side"]
    commit = resolved_commit(tokenizer)
    commit = commit or str(raw["tokenizer_revision"])
    if commit is not None and commit != EXPECTED_MODEL_REVISION:
        raise M2ProtocolError(f"Resolved tokenizer commit differs from pinned revision: {commit}")
    return tokenizer, commit


def save_model_checkpoint(model: Any, path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    return sha256_directory(path)


def load_local_model(stack: Mapping[str, Any], checkpoint: Path, device: Any) -> Any:
    model = stack["AutoModelForSequenceClassification"].from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    if int(model.config.num_labels) != 17:
        raise M2ProtocolError("Saved M2 checkpoint no longer has 17 logits")
    model.to(device)
    return model


def make_grad_scaler(torch: Any, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def adamw_execution_kwargs(*, foreach_false: bool) -> dict[str, Any]:
    """Return only the opt-in implementation-path override for AdamW.

    ``foreach=False`` selects PyTorch's single-tensor implementation without
    changing AdamW's equations or any frozen optimizer hyperparameter.  When
    disabled, the keyword is omitted so PyTorch retains its version/device
    default.
    """

    return {"foreach": False} if foreach_false else {}


def train_attempt(
    stack: Mapping[str, Any],
    tokenizer: Any,
    train_dataset: EncodedDataset,
    validation_dataset: EncodedDataset,
    config: Mapping[str, Any],
    parameters: Mapping[str, float],
    label_order: Sequence[str],
    attempt: BatchAttempt,
    checkpoint_dir: Path,
    training_log_path: Path,
    device: Any,
    *,
    local_files_only: bool,
    gradient_checkpointing: bool,
    max_length: int,
    adamw_foreach_false: bool,
    pad_to_multiple_of: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    torch = stack["torch"]
    seed = int(config["training"]["seed"])
    seed_everything(stack, seed)
    model, model_commit, checkpointing_metadata = initialize_pretrained_model(
        stack,
        config,
        label_order,
        device,
        local_files_only=local_files_only,
        gradient_checkpointing=gradient_checkpointing,
    )
    training = config["training"]
    train_loader = make_loader(
        stack,
        train_dataset,
        tokenizer,
        attempt.train_batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_length,
    )
    validation_loader = make_loader(
        stack,
        validation_dataset,
        tokenizer,
        attempt.eval_batch_size,
        shuffle=False,
        seed=seed,
        device=device,
        pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_length,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        **adamw_execution_kwargs(foreach_false=adamw_foreach_false),
    )
    observed_adamw_foreach = optimizer.param_groups[0].get("foreach")
    observed_adamw_fused = optimizer.param_groups[0].get("fused")
    if adamw_foreach_false and observed_adamw_foreach is not False:
        raise M2ProtocolError("AdamW did not retain the explicit foreach=False setting")
    updates_per_epoch = math.ceil(
        len(train_loader) / attempt.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * int(training["epochs_max"])
    warmup_steps = math.ceil(total_updates * float(training["warmup_ratio"]))
    scheduler = stack["get_linear_schedule_with_warmup"](
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    precision = precision_settings(
        device, bool(training["mixed_precision_where_supported"])
    )
    mixed_precision = bool(precision["mixed_precision"])
    precision_dtype = str(precision["mixed_precision_dtype"])
    gradient_scaler_enabled = bool(precision["gradient_scaler_enabled"])
    scaler = make_grad_scaler(torch, gradient_scaler_enabled)
    epoch_rows: list[dict[str, Any]] = []
    best_macro_ap = -math.inf
    best_epoch = 0
    best_probabilities: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, int(training["epochs_max"]) + 1):
        print(
            f"M2 epoch_start epoch={epoch}/{int(training['epochs_max'])} "
            f"batches={len(train_loader)}",
            flush=True,
        )
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_n = 0
        optimizer_steps = 0
        for batch_index, batch in enumerate(train_loader, 1):
            inputs, labels = move_batch(batch, device)
            batch_n = int(labels.shape[0])
            accumulation_window_n = accumulation_window_case_count(
                dataset_n=len(train_dataset),
                micro_batch_size=attempt.train_batch_size,
                gradient_accumulation_steps=attempt.gradient_accumulation_steps,
                batch_index_one_based=batch_index,
            )
            with autocast_context(torch, device, mixed_precision):
                output = model(**inputs, labels=labels)
                # ``output.loss`` is a microbatch mean. Weight by cases in the
                # current optimizer window so a short final window is neither
                # underweighted nor treated as a full effective batch of 16.
                scaled_loss = output.loss * (batch_n / accumulation_window_n)
            raw_loss = float(output.loss.detach().float().cpu())
            if not math.isfinite(raw_loss):
                raise M2ProtocolError("Model training loss is non-finite")
            scaler.scale(scaled_loss).backward()
            train_loss_sum += raw_loss * batch_n
            train_n += batch_n
            should_step = (
                batch_index % attempt.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if batch_index % TRAIN_PROGRESS_INTERVAL_BATCHES == 0:
                print(
                    f"M2 progress epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"elapsed_seconds={time.perf_counter() - epoch_started:.1f}",
                    flush=True,
                )

        print(f"M2 validation_start epoch={epoch}", flush=True)
        probabilities, y_validation, validation_loss = evaluate_model(
            stack,
            model,
            validation_loader,
            device,
            mixed_precision=mixed_precision,
        )
        if validation_loss is None:  # Defensive: validation always supplies labels.
            raise M2ProtocolError("Validation loss was unexpectedly unavailable")
        macro_ap, per_label_ap, supports = macro_average_precision(
            y_validation, probabilities
        )
        improved = macro_ap > best_macro_ap
        if improved:
            best_macro_ap = macro_ap
            best_epoch = epoch
            best_probabilities = probabilities.copy()
            best_labels = y_validation.copy()
            print(f"M2 checkpoint_write_start epoch={epoch}", flush=True)
            save_model_checkpoint(model, checkpoint_dir)
            print(f"M2 checkpoint_write_complete epoch={epoch}", flush=True)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_n,
                "validation_loss": validation_loss,
                "validation_macro_average_precision": macro_ap,
                "validation_defined_ap_labels": sum(value is not None for value in per_label_ap),
                "validation_positive_supports_json": canonical_json(
                    dict(zip(label_order, supports))
                ),
                "validation_per_label_ap_json": canonical_json(
                    dict(zip(label_order, per_label_ap))
                ),
                "optimizer_steps": optimizer_steps,
                "optimizer_steps_expected": updates_per_epoch,
                "accumulation_loss_scaling": "CASE_WEIGHTED_WINDOW_MEAN",
                "full_accumulation_window_cases": attempt.effective_train_batch_size,
                "final_accumulation_window_cases": (
                    len(train_dataset)
                    - (updates_per_epoch - 1) * attempt.effective_train_batch_size
                ),
                "partial_final_accumulation_window": (
                    len(train_dataset) % attempt.effective_train_batch_size != 0
                ),
                "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
                "selected_checkpoint": int(improved),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        # Persist each completed epoch so an interruption leaves diagnostic
        # evidence even though an incomplete trial is intentionally restarted.
        atomic_csv(training_log_path, epoch_rows)
        print(
            f"M2 validation_complete epoch={epoch} val_macro_ap={macro_ap:.6f} "
            f"best_epoch={best_epoch} epoch_seconds={epoch_rows[-1]['epoch_seconds']:.1f}",
            flush=True,
        )
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            break

    if best_probabilities is None or best_labels is None or best_epoch == 0:
        raise M2ProtocolError("M2 training produced no validation checkpoint")
    for row in epoch_rows:
        row["selected_checkpoint"] = int(row["epoch"] == best_epoch)
    checkpoint_hash = sha256_directory(checkpoint_dir)
    metadata = {
        "best_epoch": best_epoch,
        "best_validation_macro_average_precision": best_macro_ap,
        "epochs_completed": len(epoch_rows),
        "stopped_early": len(epoch_rows) < int(training["epochs_max"]),
        "training_seconds": time.perf_counter() - started,
        "pretrained_model_resolved_commit": model_commit,
        "checkpoint_path": display_path(checkpoint_dir),
        "checkpoint_sha256": checkpoint_hash,
        "train_batch_size": attempt.train_batch_size,
        "eval_batch_size": attempt.eval_batch_size,
        "gradient_accumulation_steps": attempt.gradient_accumulation_steps,
        "effective_train_batch_size": attempt.effective_train_batch_size,
        "effective_train_batch_size_target": attempt.effective_train_batch_size,
        "accumulation_loss_scaling": "CASE_WEIGHTED_WINDOW_MEAN",
        "optimizer_steps_per_epoch": updates_per_epoch,
        "full_accumulation_window_cases": attempt.effective_train_batch_size,
        "final_accumulation_window_cases": (
            len(train_dataset)
            - (updates_per_epoch - 1) * attempt.effective_train_batch_size
        ),
        "partial_final_accumulation_window": (
            len(train_dataset) % attempt.effective_train_batch_size != 0
        ),
        "max_length": max_length,
        "mixed_precision": mixed_precision,
        "mixed_precision_dtype": precision_dtype,
        "gradient_scaler_enabled": gradient_scaler_enabled,
        "adamw_optimizer": "torch.optim.AdamW",
        "adamw_foreach_mode": (
            "EXPLICIT_FALSE" if adamw_foreach_false else "PYTORCH_DEFAULT"
        ),
        "adamw_foreach_observed_param_group_value": observed_adamw_foreach,
        "adamw_fused_observed_param_group_value": observed_adamw_fused,
        "pad_to_multiple_of": pad_to_multiple_of,
        "tokenizer_padding_side": getattr(tokenizer, "padding_side", None),
        "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "progress_interval_batches": TRAIN_PROGRESS_INTERVAL_BATCHES,
        **checkpointing_metadata,
    }
    del model, optimizer, scheduler, scaler, train_loader, validation_loader
    clear_device_memory(torch, device)
    return epoch_rows, metadata, best_probabilities, best_labels


def validate_trial_state(
    state: Mapping[str, Any],
    context: Mapping[str, Any],
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    if state.get("status") != "COMPLETE":
        raise M2ProtocolError("Grid-trial state is not complete")
    if state.get("execution_context_sha256") != context_digest(context):
        raise M2ProtocolError("Grid trial belongs to another execution context")
    if state.get("technical_execution_options") != context.get(
        "technical_execution_options"
    ):
        raise M2ProtocolError("Grid trial uses different technical execution options")
    if state.get("parameters") != dict(parameters):
        raise M2ProtocolError("Grid-trial parameters do not match frozen order")
    checkpoint = resolve_artifact_path(str(state["checkpoint_path"]))
    if sha256_directory(checkpoint) != state.get("checkpoint_sha256"):
        raise M2ProtocolError(f"Grid-trial checkpoint is damaged: {checkpoint}")
    probability_path = resolve_artifact_path(str(state["validation_probabilities_path"]))
    if not probability_path.is_file() or sha256_file(probability_path) != state.get(
        "validation_probabilities_sha256"
    ):
        raise M2ProtocolError(f"Grid-trial validation artifact is damaged: {probability_path}")
    with np.load(probability_path, allow_pickle=False) as value:
        probabilities = validate_probability_matrix(
            value["probabilities"], artifact="saved grid-trial validation probabilities"
        )
        labels = validate_binary_label_matrix(
            value["labels"],
            expected_shape=tuple(probabilities.shape),
            artifact="saved grid-trial validation labels",
        )
    return probabilities, labels


def historical_execution_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct and authenticate the immutable context embedded in metadata."""

    fields = (
        "artifact_schema_version",
        "runner_version",
        "runner_source",
        "execution_environment",
        "execution_environment_sha256",
        "method_id",
        "evaluation",
        "fold",
        "primary_cohort_id",
        "label_order",
        "benchmark_path",
        "benchmark_sha256",
        "ontology_path",
        "ontology_sha256",
        "config_path",
        "config_sha256",
        "token_audit_path",
        "token_audit_sha256",
        "split_file_sha256",
        "split_membership_sha256",
        "role_counts",
        "train_n",
        "validation_n",
        "test_n",
        "threshold_grid",
        "baseline_threshold",
        "test_labels_used_for_selection",
        "mps_allocator",
        "technical_execution_options",
    )
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise M2ProtocolError(
            f"Historical M2 metadata lacks context fields: {missing}"
        )
    context = {field: metadata[field] for field in fields}
    if context_digest(context) != metadata.get("execution_context_sha256"):
        raise M2ProtocolError("Historical M2 execution-context digest is invalid")
    return context


def validate_a1_fixed_transfer_source(
    model_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless completed A1 selected the amended A2 parameters."""

    run_path = model_root / "a1" / "run_metadata.json"
    fit_path = model_root / "a1" / "fit_state.json"
    if not run_path.is_file() or not fit_path.is_file():
        raise M2ProtocolError("Fixed A2 mode requires complete M2 A1 artifacts")
    run = load_json(run_path)
    fit = load_json(fit_path)
    if run.get("status") != "COMPLETE":
        raise M2ProtocolError("M2 A1 is not COMPLETE")
    if fit.get("status") != "FIT_AND_VALIDATION_SELECTION_COMPLETE":
        raise M2ProtocolError("M2 A1 fit state is not complete")
    if sha256_file(fit_path) != run.get("fit_state_sha256"):
        raise M2ProtocolError("M2 A1 fit-state hash differs from run metadata")
    context = historical_execution_context(run)
    validate_fit_state(fit, context)
    if fit.get("selection") != run.get("selection"):
        raise M2ProtocolError("M2 A1 run and fit-state selections differ")
    expected = parameter_grid(config)[FIXED_A2_CONFIGURATION_INDEX - 1]
    if expected != FIXED_A2_HYPERPARAMETERS:
        raise M2ProtocolError("Frozen configuration 05 is not 3e-5 / 0.01")
    selection = fit["selection"]
    if (
        int(selection.get("selected_configuration_index", 0))
        != FIXED_A2_CONFIGURATION_INDEX
        or selection.get("selected_hyperparameters") != FIXED_A2_HYPERPARAMETERS
    ):
        raise M2ProtocolError(
            "M2 A1 did not select configuration 05 at 3e-5 / 0.01"
        )
    if (
        run.get("test_labels_used_for_selection") is not False
        or fit.get("test_labels_used_for_selection") is not False
        or selection.get("test_labels_used_for_selection") is not False
    ):
        raise M2ProtocolError("M2 A1 selection is not validation-only")
    prediction_path = resolve_artifact_path(str(run.get("prediction_path", "")))
    if (
        not prediction_path.is_file()
        or sha256_file(prediction_path) != run.get("prediction_sha256")
    ):
        raise M2ProtocolError("M2 A1 prediction artifact is missing or damaged")
    return {
        "selection_source": "A1_VALIDATION_TRANSFER",
        "configuration_index": FIXED_A2_CONFIGURATION_INDEX,
        "hyperparameters": dict(FIXED_A2_HYPERPARAMETERS),
        "selected_epoch": int(selection["selected_best_epoch"]),
        "run_metadata_path": display_path(run_path),
        "run_metadata_sha256": sha256_file(run_path),
        "fit_state_path": display_path(fit_path),
        "fit_state_sha256": sha256_file(fit_path),
        "selected_checkpoint_path": selection["selected_checkpoint_path"],
        "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "completed_at": fit.get("completed_at"),
        "test_labels_used_for_selection": False,
    }


def legacy_fold1_metadata_archive_path(spec: RunSpec) -> Path:
    return (
        spec.model_dir
        / "_protocol_history"
        / FIXED_A2_PROTOCOL_ID
        / LEGACY_FOLD1_METADATA_ARCHIVE
    )


def preserve_legacy_fold1_metadata(spec: RunSpec) -> Path:
    """Preserve the interrupted legacy-grid metadata byte for byte."""

    source = spec.model_dir / "run_metadata.json"
    archive = legacy_fold1_metadata_archive_path(spec)
    if archive.is_file():
        archived = load_json(archive)
        if archived.get("status") != "INTERRUPTED":
            raise M2ProtocolError("Fold 1 legacy metadata archive is not INTERRUPTED")
        historical_execution_context(archived)
        return archive
    if not source.is_file():
        raise M2ProtocolError("Fold 1 legacy run metadata is missing")
    payload = source.read_bytes()
    metadata = load_json(source)
    if metadata.get("status") != "INTERRUPTED":
        raise M2ProtocolError(
            "Fold 1 predecessor metadata must be INTERRUPTED before promotion"
        )
    historical_execution_context(metadata)
    atomic_bytes(archive, payload)
    if archive.read_bytes() != payload:
        raise M2ProtocolError("Fold 1 legacy metadata archive byte check failed")
    return archive


def validate_legacy_fold1_c5(
    spec: RunSpec,
    validation: Sequence[dict[str, Any]],
    label_order: Sequence[str],
    split_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate legacy Fold-1 C5 without weakening new-context checks."""

    archive = legacy_fold1_metadata_archive_path(spec)
    metadata_path = archive if archive.is_file() else spec.model_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise M2ProtocolError("Fold 1 legacy metadata is unavailable")
    metadata = load_json(metadata_path)
    if (
        metadata.get("status") != "INTERRUPTED"
        or metadata.get("evaluation") != "A2"
        or metadata.get("fold") != 1
        or metadata.get("benchmark_sha256") != EXPECTED_BENCHMARK_SHA256
        or metadata.get("ontology_sha256") != EXPECTED_ONTOLOGY_SHA256
        or metadata.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or metadata.get("split_file_sha256") != EXPECTED_A2_SPLIT_SHA256
        or metadata.get("split_membership_sha256")
        != split_metadata.get("split_membership_sha256")
        or int(metadata.get("train_n", -1)) != int(split_metadata["train_n"])
        or int(metadata.get("validation_n", -1))
        != int(split_metadata["validation_n"])
        or int(metadata.get("test_n", -1)) != int(split_metadata["test_n"])
    ):
        raise M2ProtocolError("Fold 1 legacy run metadata failed frozen-context checks")
    legacy_context = historical_execution_context(metadata)
    trial_path = (
        spec.model_dir
        / "grid"
        / f"configuration_{FIXED_A2_CONFIGURATION_INDEX:02d}"
        / "trial_state.json"
    )
    if not trial_path.is_file():
        raise M2ProtocolError("Fold 1 legacy configuration 05 state is missing")
    state = load_json(trial_path)
    probabilities, labels = validate_trial_state(
        state, legacy_context, FIXED_A2_HYPERPARAMETERS
    )
    log_path = resolve_artifact_path(str(state.get("training_log_path", "")))
    if (
        not log_path.is_file()
        or sha256_file(log_path) != state.get("training_log_sha256")
    ):
        raise M2ProtocolError("Fold 1 C5 training log is missing or damaged")
    log_rows = load_csv(log_path)
    expected_validation = target_matrix(validation, label_order)
    if (
        int(state.get("configuration_index", 0)) != FIXED_A2_CONFIGURATION_INDEX
        or int(state.get("epochs_completed", 0)) != 6
        or len(log_rows) != 6
        or [int(row["epoch"]) for row in log_rows] != list(range(1, 7))
        or state.get("fresh_pretrained_initialization") is not True
        or state.get("initialization_source")
        != f"{EXPECTED_MODEL_ID}@{EXPECTED_MODEL_REVISION}"
        or state.get("pretrained_model_resolved_commit") != EXPECTED_MODEL_REVISION
        or not np.array_equal(labels, expected_validation)
    ):
        raise M2ProtocolError("Fold 1 C5 training or validation provenance is invalid")
    technical = state.get("technical_execution_options", {})
    expected_technical = {
        "gradient_checkpointing": True,
        "initial_train_batch_size": 1,
        "max_length": EXPECTED_MAX_LENGTH,
        "adamw_foreach": False,
        "pad_to_multiple_of": 64,
    }
    if any(technical.get(key) != value for key, value in expected_technical.items()):
        raise M2ProtocolError("Fold 1 C5 does not use the stable A1 technical settings")
    if (
        state.get("mixed_precision_dtype") != "bfloat16"
        or state.get("gradient_scaler_enabled") is not False
        or int(state.get("train_batch_size", 0)) != 1
        or int(state.get("eval_batch_size", 0)) != 2
        or int(state.get("gradient_accumulation_steps", 0)) != 16
        or int(state.get("effective_train_batch_size", 0)) != 16
    ):
        raise M2ProtocolError("Fold 1 C5 execution metadata differs from A1 settings")
    macro_ap, _, _ = macro_average_precision(labels, probabilities)
    if not math.isclose(
        macro_ap,
        float(state.get("best_validation_macro_average_precision", math.nan)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise M2ProtocolError("Fold 1 C5 validation macro AP is not reproducible")
    tokenizer_dir = spec.model_dir / "tokenizer"
    tokenizer_sha256 = sha256_directory(tokenizer_dir)
    c6_path = spec.model_dir / "grid/configuration_06/trial_state.json"
    c6 = load_json(c6_path) if c6_path.is_file() else None
    return {
        "state": state,
        "probabilities": probabilities,
        "labels": labels,
        "source": {
            "disposition": "REUSED_OFFICIAL_WITHOUT_RETRAINING",
            "trial_state_path": display_path(trial_path),
            "trial_state_sha256": sha256_file(trial_path),
            "legacy_execution_context_sha256": metadata[
                "execution_context_sha256"
            ],
            "legacy_metadata_path": display_path(metadata_path),
            "legacy_metadata_sha256": sha256_file(metadata_path),
            "training_log_path": display_path(log_path),
            "training_log_sha256": sha256_file(log_path),
            "validation_probabilities_path": state[
                "validation_probabilities_path"
            ],
            "validation_probabilities_sha256": state[
                "validation_probabilities_sha256"
            ],
            "checkpoint_path": state["checkpoint_path"],
            "checkpoint_sha256": state["checkpoint_sha256"],
            "tokenizer_path": display_path(tokenizer_dir),
            "tokenizer_sha256": tokenizer_sha256,
            "configuration_01_through_04_disposition": "HISTORICAL_ONLY",
            "configuration_06_disposition": "ABANDONED_INTERRUPTED_LEGACY_GRID",
            "configuration_06_state_path": (
                display_path(c6_path) if c6_path.is_file() else None
            ),
            "configuration_06_state_sha256": (
                sha256_file(c6_path) if c6_path.is_file() else None
            ),
            "configuration_06_recorded_status": (
                c6.get("status") if c6 is not None else None
            ),
            "validation_macro_average_precision": macro_ap,
        },
    }


def run_grid_trial(
    configuration_index: int,
    parameters: Mapping[str, float],
    trial_dir: Path,
    stack: Mapping[str, Any],
    tokenizer: Any,
    train_dataset: EncodedDataset,
    validation_dataset: EncodedDataset,
    config: Mapping[str, Any],
    label_order: Sequence[str],
    context: Mapping[str, Any],
    device: Any,
    *,
    force: bool,
    local_files_only: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    technical_options = context["technical_execution_options"]
    gradient_checkpointing = bool(technical_options["gradient_checkpointing"])
    initial_train_batch_size = int(
        technical_options["initial_train_batch_size"]
    )
    max_length = int(technical_options["max_length"])
    adamw_foreach_false = technical_options.get("adamw_foreach") is False
    pad_to_multiple_of = technical_options.get("pad_to_multiple_of")
    state_path = trial_dir / "trial_state.json"
    probability_path = trial_dir / "validation_probabilities.npz"
    log_path = trial_dir / "training_log.csv"
    checkpoint_dir = trial_dir / "checkpoint_best"
    if state_path.is_file() and not force:
        state = load_json(state_path)
        if state.get("status") == "COMPLETE":
            probabilities, labels = validate_trial_state(state, context, parameters)
            return state, probabilities, labels

    trial_dir.mkdir(parents=True, exist_ok=True)
    attempts_log: list[dict[str, Any]] = []
    atomic_json(
        state_path,
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "IN_PROGRESS",
            "configuration_index": configuration_index,
            "parameters": dict(parameters),
            "execution_context_sha256": context_digest(context),
            "started_at": utc_now(),
            "initialization_source": f"{EXPECTED_MODEL_ID}@{EXPECTED_MODEL_REVISION}",
            "fresh_pretrained_initialization": True,
            "technical_execution_options": technical_options,
        },
    )
    attempted_batches = batch_attempts(
        config, initial_train_batch_size=initial_train_batch_size
    )
    for attempt_index, attempt in enumerate(attempted_batches, 1):
        # A previous OOM traceback is out of scope by this point; clear cached
        # device allocations before freshly loading the pinned model again.
        clear_device_memory(stack["torch"], device)
        print(
            f"M2 grid={configuration_index}/6 batch={attempt.train_batch_size} "
            f"grad_acc={attempt.gradient_accumulation_steps}",
            flush=True,
        )
        try:
            epoch_rows, training_metadata, probabilities, labels = train_attempt(
                stack,
                tokenizer,
                train_dataset,
                validation_dataset,
                config,
                parameters,
                label_order,
                attempt,
                checkpoint_dir,
                log_path,
                device,
                local_files_only=local_files_only,
                gradient_checkpointing=gradient_checkpointing,
                max_length=max_length,
                adamw_foreach_false=adamw_foreach_false,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            attempts_log.append(
                {
                    "attempt_index": attempt_index,
                    "status": "COMPLETE",
                    "train_batch_size": attempt.train_batch_size,
                    "eval_batch_size": attempt.eval_batch_size,
                    "gradient_accumulation_steps": attempt.gradient_accumulation_steps,
                    "effective_train_batch_size": attempt.effective_train_batch_size,
                    "gradient_checkpointing": gradient_checkpointing,
                    "mixed_precision_dtype": training_metadata[
                        "mixed_precision_dtype"
                    ],
                    "adamw_foreach_mode": training_metadata[
                        "adamw_foreach_mode"
                    ],
                    "pad_to_multiple_of": pad_to_multiple_of,
                }
            )
            atomic_csv(log_path, epoch_rows)
            atomic_npz(
                probability_path,
                probabilities=probabilities,
                labels=labels,
            )
            state = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": "COMPLETE",
                "configuration_index": configuration_index,
                "parameters": dict(parameters),
                "execution_context_sha256": context_digest(context),
                "completed_at": utc_now(),
                "initialization_source": f"{EXPECTED_MODEL_ID}@{EXPECTED_MODEL_REVISION}",
                "fresh_pretrained_initialization": True,
                "technical_execution_options": technical_options,
                "batch_attempts": attempts_log,
                **training_metadata,
                "training_log_path": display_path(log_path),
                "training_log_sha256": sha256_file(log_path),
                "validation_probabilities_path": display_path(probability_path),
                "validation_probabilities_sha256": sha256_file(probability_path),
                "validation_n": int(probabilities.shape[0]),
                "test_labels_used_for_selection": False,
            }
            atomic_json(state_path, state)
            return state, probabilities, labels
        except Exception as error:
            if is_out_of_memory(error):
                attempts_log.append(
                    {
                        "attempt_index": attempt_index,
                        "status": "OUT_OF_MEMORY",
                        "train_batch_size": attempt.train_batch_size,
                        "eval_batch_size": attempt.eval_batch_size,
                        "gradient_accumulation_steps": attempt.gradient_accumulation_steps,
                        "effective_train_batch_size": attempt.effective_train_batch_size,
                        "gradient_checkpointing": gradient_checkpointing,
                        "mixed_precision_policy": technical_options[
                            "mixed_precision_policy"
                        ],
                        "adamw_foreach_mode": technical_options.get(
                            "adamw_foreach_mode", "PYTORCH_DEFAULT"
                        ),
                        "pad_to_multiple_of": pad_to_multiple_of,
                        "mps_low_watermark_ratio": technical_options.get(
                            "mps_allocator", {}
                        ).get("requested_low_watermark_ratio"),
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:500],
                    }
                )
                atomic_json(
                    state_path,
                    {
                        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                        "status": "IN_PROGRESS_BATCH_FALLBACK",
                        "configuration_index": configuration_index,
                        "parameters": dict(parameters),
                        "execution_context_sha256": context_digest(context),
                        "batch_attempts": attempts_log,
                        "max_length": max_length,
                        "max_length_reduced": max_length < EXPECTED_MAX_LENGTH,
                        "technical_execution_options": technical_options,
                    },
                )
                clear_device_memory(stack["torch"], device)
                continue
            atomic_json(
                state_path,
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "status": "FAILED",
                    "configuration_index": configuration_index,
                    "parameters": dict(parameters),
                    "execution_context_sha256": context_digest(context),
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:1000],
                    "batch_attempts": attempts_log,
                    "technical_execution_options": technical_options,
                },
            )
            raise
    mode = "with" if gradient_checkpointing else "without"
    exhausted_batches = [attempt.train_batch_size for attempt in attempted_batches]
    terminal_message = (
        f"M2 exhausted batch sizes {exhausted_batches} at max_length={max_length} {mode} "
        "gradient checkpointing; the frozen protocol prohibits silently reducing "
        "max_length"
    )
    atomic_json(
        state_path,
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "FAILED_OOM_BATCH_FALLBACK_EXHAUSTED",
            "configuration_index": configuration_index,
            "parameters": dict(parameters),
            "execution_context_sha256": context_digest(context),
            "failed_at": utc_now(),
            "failure_reason": terminal_message,
            "batch_attempts": attempts_log,
            "max_length": max_length,
            "max_length_reduced": max_length < EXPECTED_MAX_LENGTH,
            "technical_execution_options": technical_options,
        },
    )
    raise M2ProtocolError(terminal_message)


def grid_result_row(
    configuration_index: int,
    parameters: Mapping[str, float],
    state: Mapping[str, Any],
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_order: Sequence[str],
) -> dict[str, Any]:
    macro_ap, per_label_ap, supports = macro_average_precision(labels, probabilities)
    stored = float(state["best_validation_macro_average_precision"])
    if not math.isclose(macro_ap, stored, rel_tol=0.0, abs_tol=1e-12):
        raise M2ProtocolError("Stored and recomputed validation macro AP differ")
    return {
        "configuration_index": configuration_index,
        "learning_rate": parameters["learning_rate"],
        "weight_decay": parameters["weight_decay"],
        "best_epoch": state["best_epoch"],
        "validation_macro_average_precision": macro_ap,
        "validation_defined_ap_labels": sum(
            value is not None for value in per_label_ap
        ),
        "validation_positive_supports_json": canonical_json(
            dict(zip(label_order, supports))
        ),
        "validation_per_label_ap_json": canonical_json(
            dict(zip(label_order, per_label_ap))
        ),
        "train_batch_size": state["train_batch_size"],
        "gradient_accumulation_steps": state["gradient_accumulation_steps"],
        "gradient_checkpointing": state["gradient_checkpointing"],
        "mixed_precision_dtype": state["mixed_precision_dtype"],
        "adamw_foreach_mode": state["adamw_foreach_mode"],
        "adamw_foreach_observed_param_group_value": state[
            "adamw_foreach_observed_param_group_value"
        ],
        "adamw_fused_observed_param_group_value": state[
            "adamw_fused_observed_param_group_value"
        ],
        "pad_to_multiple_of": state["pad_to_multiple_of"],
        "tokenizer_padding_side": state["tokenizer_padding_side"],
        "tokenizer_pad_token_id": state["tokenizer_pad_token_id"],
        "model_config_has_use_cache": state["model_config_has_use_cache"],
        "model_use_cache_action": state["model_use_cache_action"],
        "model_use_cache_during_training": state[
            "model_use_cache_during_training"
        ],
        "accumulation_loss_scaling": state["accumulation_loss_scaling"],
        "final_accumulation_window_cases": state[
            "final_accumulation_window_cases"
        ],
        "training_seconds": state["training_seconds"],
        "checkpoint_path": state["checkpoint_path"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "selected": 0,
    }


def fit_and_select(
    spec: RunSpec,
    stack: Mapping[str, Any],
    tokenizer: Any,
    train: Sequence[dict[str, Any]],
    validation: Sequence[dict[str, Any]],
    label_order: Sequence[str],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    device: Any,
    *,
    force: bool,
    local_files_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    max_length = int(context["technical_execution_options"]["max_length"])
    train_dataset = encode_records(tokenizer, train, label_order, max_length)
    validation_dataset = encode_records(tokenizer, validation, label_order, max_length)
    grid_rows: list[dict[str, Any]] = []
    trial_states: list[dict[str, Any]] = []
    validation_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for configuration_index, parameters in enumerate(parameter_grid(config), 1):
        state, probabilities, labels = run_grid_trial(
            configuration_index,
            parameters,
            spec.model_dir / "grid" / f"configuration_{configuration_index:02d}",
            stack,
            tokenizer,
            train_dataset,
            validation_dataset,
            config,
            label_order,
            context,
            device,
            force=force,
            local_files_only=local_files_only,
        )
        trial_states.append(state)
        validation_arrays.append((probabilities, labels))
        grid_rows.append(
            grid_result_row(
                configuration_index,
                parameters,
                state,
                probabilities,
                labels,
                label_order,
            )
        )
    selected_index = min(
        range(len(grid_rows)),
        key=lambda index: (
            -grid_rows[index]["validation_macro_average_precision"],
            grid_rows[index]["configuration_index"],
        ),
    )
    grid_rows[selected_index]["selected"] = 1
    selected_state = trial_states[selected_index]
    probabilities, y_validation = validation_arrays[selected_index]
    selected_threshold, threshold_rows = select_global_threshold(
        y_validation, probabilities
    )
    for row in threshold_rows:
        row["selected"] = int(row["threshold"] == selected_threshold)
    selection = {
        "selected_configuration_index": grid_rows[selected_index]["configuration_index"],
        "selected_hyperparameters": {
            "learning_rate": grid_rows[selected_index]["learning_rate"],
            "weight_decay": grid_rows[selected_index]["weight_decay"],
        },
        "selected_best_epoch": selected_state["best_epoch"],
        "selected_checkpoint_path": selected_state["checkpoint_path"],
        "selected_checkpoint_sha256": selected_state["checkpoint_sha256"],
        "selected_training_execution": {
            "gradient_checkpointing": selected_state[
                "gradient_checkpointing"
            ],
            "mixed_precision_dtype": selected_state[
                "mixed_precision_dtype"
            ],
            "gradient_scaler_enabled": selected_state[
                "gradient_scaler_enabled"
            ],
            "adamw_optimizer": selected_state["adamw_optimizer"],
            "adamw_foreach_mode": selected_state["adamw_foreach_mode"],
            "adamw_foreach_observed_param_group_value": selected_state[
                "adamw_foreach_observed_param_group_value"
            ],
            "adamw_fused_observed_param_group_value": selected_state[
                "adamw_fused_observed_param_group_value"
            ],
            "pad_to_multiple_of": selected_state["pad_to_multiple_of"],
            "tokenizer_padding_side": selected_state[
                "tokenizer_padding_side"
            ],
            "tokenizer_pad_token_id": selected_state[
                "tokenizer_pad_token_id"
            ],
            "model_config_has_use_cache": selected_state[
                "model_config_has_use_cache"
            ],
            "model_use_cache_action": selected_state[
                "model_use_cache_action"
            ],
            "model_use_cache_during_training": selected_state[
                "model_use_cache_during_training"
            ],
            "accumulation_loss_scaling": selected_state[
                "accumulation_loss_scaling"
            ],
            "effective_train_batch_size_target": selected_state[
                "effective_train_batch_size_target"
            ],
            "final_accumulation_window_cases": selected_state[
                "final_accumulation_window_cases"
            ],
        },
        "validation_macro_average_precision": grid_rows[selected_index][
            "validation_macro_average_precision"
        ],
        "selected_global_threshold": selected_threshold,
        "validation_macro_f1_selected_threshold": macro_f1(
            y_validation, probabilities, selected_threshold
        ),
        "validation_macro_f1_0_50": macro_f1(
            y_validation, probabilities, BASELINE_THRESHOLD
        ),
        "validation_label_positive_support": dict(
            zip(label_order, y_validation.sum(axis=0).astype(int).tolist())
        ),
        "hyperparameter_tie_break": "earliest_configuration_in_frozen_grid",
        "threshold_tie_break": (
            "max_validation_macro_f1_then_closest_to_0.50_then_lower_threshold"
        ),
        "test_labels_used_for_selection": False,
    }
    return selection, grid_rows, threshold_rows


def prepare_fixed_trial_restart(
    trial_dir: Path,
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Archive a same-context interrupted fixed trial before a fresh restart."""

    state_path = trial_dir / "trial_state.json"
    if not state_path.is_file():
        if trial_dir.exists() and any(trial_dir.iterdir()):
            raise M2ProtocolError(
                f"Fixed A2 trial directory has files but no state: {trial_dir}"
            )
        return None
    state = load_json(state_path)
    status = state.get("status")
    if status == "COMPLETE":
        return None
    if status != "IN_PROGRESS":
        raise M2ProtocolError(
            f"Fixed A2 trial has non-recoverable status {status!r}: {state_path}"
        )
    if (
        state.get("execution_context_sha256") != context_digest(context)
        or state.get("parameters") != FIXED_A2_HYPERPARAMETERS
        or int(state.get("configuration_index", 0))
        != FIXED_A2_CONFIGURATION_INDEX
    ):
        raise M2ProtocolError("Interrupted fixed A2 trial has context/parameter drift")
    history = (
        trial_dir.parent.parent
        / "_attempt_history"
        / (
            f"configuration_{FIXED_A2_CONFIGURATION_INDEX:02d}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"{context_digest(context)[:12]}"
        )
    )
    history.parent.mkdir(parents=True, exist_ok=True)
    os.replace(trial_dir, history)
    print(
        f"M2 fixed protocol archived interrupted trial to {display_path(history)}; "
        "restarting configuration 05 from pretrained initialization",
        flush=True,
    )
    return {
        "status": "ARCHIVED_INTERRUPTED_FIXED_TRIAL",
        "archive_path": display_path(history),
        "archive_sha256": sha256_directory(history),
    }


def fit_and_select_fixed_a2(
    spec: RunSpec,
    stack: Mapping[str, Any],
    tokenizer: Any,
    train: Sequence[dict[str, Any]],
    validation: Sequence[dict[str, Any]],
    label_order: Sequence[str],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    device: Any,
    artifact_dir: Path,
    *,
    legacy_fold1: Mapping[str, Any] | None,
    local_files_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run/adopt only A1-selected configuration 05 for revised M2 A2."""

    if spec.evaluation != "A2" or spec.fold not in (1, 2, 3):
        raise M2ProtocolError("Fixed A1-hyperparameter mode is A2-only")
    max_length = int(context["technical_execution_options"]["max_length"])
    train_dataset = encode_records(tokenizer, train, label_order, max_length)
    validation_dataset = encode_records(
        tokenizer, validation, label_order, max_length
    )
    restart_provenance: dict[str, Any] | None = None
    if spec.fold == 1:
        if legacy_fold1 is None:
            raise M2ProtocolError("Fold 1 fixed mode lacks validated legacy C5")
        state = dict(legacy_fold1["state"])
        probabilities = np.asarray(legacy_fold1["probabilities"])
        labels = np.asarray(legacy_fold1["labels"])
        training_reused = True
    else:
        trial_dir = (
            artifact_dir
            / "grid"
            / f"configuration_{FIXED_A2_CONFIGURATION_INDEX:02d}"
        )
        restart_provenance = prepare_fixed_trial_restart(trial_dir, context)
        print(
            f"M2 A2 Fold {spec.fold}: training fixed configuration 05 "
            "(lr=3e-5, weight_decay=0.01)",
            flush=True,
        )
        state, probabilities, labels = run_grid_trial(
            FIXED_A2_CONFIGURATION_INDEX,
            FIXED_A2_HYPERPARAMETERS,
            trial_dir,
            stack,
            tokenizer,
            train_dataset,
            validation_dataset,
            config,
            label_order,
            context,
            device,
            force=False,
            local_files_only=local_files_only,
        )
        training_reused = False
    row = grid_result_row(
        FIXED_A2_CONFIGURATION_INDEX,
        FIXED_A2_HYPERPARAMETERS,
        state,
        probabilities,
        labels,
        label_order,
    )
    row["selected"] = 1
    row["selection_basis"] = "A1_VALIDATION_TRANSFER"
    selected_threshold, threshold_rows = select_global_threshold(labels, probabilities)
    for threshold_row in threshold_rows:
        threshold_row["selected"] = int(
            threshold_row["threshold"] == selected_threshold
        )
    selection = {
        "selected_configuration_index": FIXED_A2_CONFIGURATION_INDEX,
        "official_configuration_index": FIXED_A2_CONFIGURATION_INDEX,
        "selected_hyperparameters": dict(FIXED_A2_HYPERPARAMETERS),
        "hyperparameter_selection_source": "A1_VALIDATION_TRANSFER",
        "per_fold_hyperparameter_search": False,
        "a2_validation_used_for_hyperparameter_selection": False,
        "selected_best_epoch": state["best_epoch"],
        "selected_checkpoint_path": state["checkpoint_path"],
        "selected_checkpoint_sha256": state["checkpoint_sha256"],
        "selected_training_execution": {
            "gradient_checkpointing": state["gradient_checkpointing"],
            "mixed_precision_dtype": state["mixed_precision_dtype"],
            "gradient_scaler_enabled": state["gradient_scaler_enabled"],
            "adamw_optimizer": state["adamw_optimizer"],
            "adamw_foreach_mode": state["adamw_foreach_mode"],
            "adamw_foreach_observed_param_group_value": state[
                "adamw_foreach_observed_param_group_value"
            ],
            "adamw_fused_observed_param_group_value": state[
                "adamw_fused_observed_param_group_value"
            ],
            "pad_to_multiple_of": state["pad_to_multiple_of"],
            "tokenizer_padding_side": state["tokenizer_padding_side"],
            "tokenizer_pad_token_id": state["tokenizer_pad_token_id"],
            "model_config_has_use_cache": state["model_config_has_use_cache"],
            "model_use_cache_action": state["model_use_cache_action"],
            "model_use_cache_during_training": state[
                "model_use_cache_during_training"
            ],
            "accumulation_loss_scaling": state["accumulation_loss_scaling"],
            "effective_train_batch_size_target": state[
                "effective_train_batch_size_target"
            ],
            "final_accumulation_window_cases": state[
                "final_accumulation_window_cases"
            ],
        },
        "validation_macro_average_precision": row[
            "validation_macro_average_precision"
        ],
        "selected_global_threshold": selected_threshold,
        "validation_macro_f1_selected_threshold": macro_f1(
            labels, probabilities, selected_threshold
        ),
        "validation_macro_f1_0_50": macro_f1(
            labels, probabilities, BASELINE_THRESHOLD
        ),
        "validation_label_positive_support": dict(
            zip(label_order, labels.sum(axis=0).astype(int).tolist())
        ),
        "hyperparameter_tie_break": "NOT_APPLICABLE_FIXED_A1_TRANSFER",
        "threshold_tie_break": (
            "max_validation_macro_f1_then_closest_to_0.50_then_lower_threshold"
        ),
        "epoch_selected_on": "FOLD_VALIDATION_MACRO_AVERAGE_PRECISION",
        "threshold_selected_on": "FOLD_VALIDATION_MACRO_F1",
        "training_reused_without_retraining": training_reused,
        "interrupted_fixed_trial_restart_provenance": restart_provenance,
        "test_labels_used_for_selection": False,
    }
    return selection, [row], threshold_rows


def validate_fit_state(
    fit_state: Mapping[str, Any], context: Mapping[str, Any]
) -> None:
    if fit_state.get("status") != "FIT_AND_VALIDATION_SELECTION_COMPLETE":
        raise M2ProtocolError("M2 fit state is not complete")
    if fit_state.get("execution_context_sha256") != context_digest(context):
        raise M2ProtocolError("M2 fit state belongs to a different execution context")
    if fit_state.get("technical_execution_options") != context.get(
        "technical_execution_options"
    ):
        raise M2ProtocolError("M2 fit state uses different technical execution options")
    selection = fit_state["selection"]
    if float(selection["selected_global_threshold"]) not in THRESHOLD_GRID:
        raise M2ProtocolError("M2 fit-state threshold is outside frozen grid")
    checkpoint = resolve_artifact_path(str(selection["selected_checkpoint_path"]))
    if sha256_directory(checkpoint) != selection.get("selected_checkpoint_sha256"):
        raise M2ProtocolError("Selected M2 checkpoint is missing or damaged")
    tokenizer_dir = resolve_artifact_path(str(fit_state["tokenizer_path"]))
    if sha256_directory(tokenizer_dir) != fit_state.get("tokenizer_sha256"):
        raise M2ProtocolError("Saved M2 tokenizer is missing or damaged")
    for name in ("validation_hyperparameter_search", "validation_threshold_search"):
        artifact = resolve_artifact_path(str(fit_state[f"{name}_path"]))
        if not artifact.is_file() or sha256_file(artifact) != fit_state.get(
            f"{name}_sha256"
        ):
            raise M2ProtocolError(f"M2 fit-state artifact is damaged: {artifact}")
    protocol = context.get("scientific_protocol")
    if protocol is not None:
        if (
            protocol.get("protocol_id") != FIXED_A2_PROTOCOL_ID
            or protocol.get("amendment_sha256")
            != EXPECTED_FIXED_A2_AMENDMENT_SHA256
            or protocol.get("hyperparameter_source") != "A1_VALIDATION_TRANSFER"
            or protocol.get("fixed_hyperparameters") != FIXED_A2_HYPERPARAMETERS
            or protocol.get("official_configuration_index")
            != FIXED_A2_CONFIGURATION_INDEX
            or protocol.get("a2_test_results_used_for_amendment") is not False
        ):
            raise M2ProtocolError("Fixed A2 scientific-protocol provenance is invalid")
        if fit_state.get("scientific_protocol") != protocol:
            raise M2ProtocolError("Fixed A2 fit state lacks exact amendment provenance")
        if (
            selection.get("selected_configuration_index")
            != FIXED_A2_CONFIGURATION_INDEX
            or selection.get("selected_hyperparameters")
            != FIXED_A2_HYPERPARAMETERS
            or selection.get("hyperparameter_selection_source")
            != "A1_VALIDATION_TRANSFER"
            or selection.get("per_fold_hyperparameter_search") is not False
            or selection.get("a2_validation_used_for_hyperparameter_selection")
            is not False
        ):
            raise M2ProtocolError("Fixed A2 fit-state selection is invalid")
        search_rows = load_csv(
            resolve_artifact_path(
                str(fit_state["validation_hyperparameter_search_path"])
            )
        )
        if (
            len(search_rows) != 1
            or int(search_rows[0]["configuration_index"])
            != FIXED_A2_CONFIGURATION_INDEX
            or search_rows[0]["selected"] != "1"
            or search_rows[0].get("selection_basis")
            != "A1_VALIDATION_TRANSFER"
        ):
            raise M2ProtocolError("Fixed A2 search artifact is not a one-row transfer")


def complete_run_is_valid(
    metadata_path: Path,
    prediction_path: Path,
    context: Mapping[str, Any],
    *,
    recover_interrupted: bool = False,
) -> bool:
    if not metadata_path.is_file():
        return False
    metadata = load_json(metadata_path)
    if metadata.get("execution_context_sha256") != context_digest(context):
        raise M2ProtocolError(
            f"Existing M2 run has a different context: {metadata_path}; "
            "use --force to begin the explicitly changed execution mode"
        )
    if metadata.get("status") == "IN_PROGRESS" or (
        recover_interrupted and metadata.get("status") == "INTERRUPTED"
    ):
        return False
    if metadata.get("status") != "COMPLETE":
        raise M2ProtocolError(f"Invalid M2 run status in {metadata_path}")
    if not prediction_path.is_file() or sha256_file(prediction_path) != metadata.get(
        "prediction_sha256"
    ):
        raise M2ProtocolError("Existing M2 prediction artifact is missing or damaged")
    fit_state_path = resolve_artifact_path(str(metadata["fit_state_path"]))
    if not fit_state_path.is_file() or sha256_file(fit_state_path) != metadata.get(
        "fit_state_sha256"
    ):
        raise M2ProtocolError("Existing M2 fit state is missing or damaged")
    validate_fit_state(load_json(fit_state_path), context)
    return True


def prediction_rows(
    spec: RunSpec,
    records: Sequence[dict[str, Any]],
    probabilities: np.ndarray,
    label_order: Sequence[str],
    threshold: float,
    run_id: str,
    context: Mapping[str, Any],
    token_info: Mapping[int, TokenInfo],
) -> list[dict[str, Any]]:
    actual_max_length = int(context["technical_execution_options"]["max_length"])
    probabilities = validate_probability_matrix(
        probabilities,
        expected_shape=(len(records), len(label_order)),
        artifact="M2 test probabilities",
    )
    rows: list[dict[str, Any]] = []
    for record, scores in sorted(
        zip(records, probabilities), key=lambda item: int(item[0]["identity"]["search_rank"])
    ):
        identity = record["identity"]
        rank = int(identity["search_rank"])
        info = token_info[rank]
        reference = set(record_labels(record))
        fact_summary = str(record["text_input"]["english_fact_summary_raw"])
        rows.append(
            {
                "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
                "run_id": run_id,
                "method_id": EXPECTED_METHOD_ID,
                "evaluation": spec.evaluation,
                "fold": spec.fold,
                "search_rank": rank,
                "case_id": identity.get("unodc_case_number") or str(rank),
                "canonical_url": identity["canonical_url"],
                "jurisdiction": identity["jurisdiction_country_raw"],
                "split": "TEST",
                "fact_summary": fact_summary,
                "input_sha256": sha256_text(fact_summary),
                "silver_reference_labels": [
                    label for label in label_order if label in reference
                ],
                "predicted_labels": [
                    label for label, score in zip(label_order, scores) if score >= threshold
                ],
                "predicted_labels_0_50": [
                    label
                    for label, score in zip(label_order, scores)
                    if score >= BASELINE_THRESHOLD
                ],
                "probabilities_by_label": {
                    label: float(score) for label, score in zip(label_order, scores)
                },
                "selected_threshold": threshold,
                "original_token_count": info.original_token_count,
                "max_tokens_used": min(
                    info.original_token_count, actual_max_length
                ),
                "max_length": actual_max_length,
                "truncated_input": (
                    info.original_token_count > actual_max_length
                ),
                "truncation_side": "right",
                "tokenizer_model_id": EXPECTED_MODEL_ID,
                "tokenizer_revision": EXPECTED_MODEL_REVISION,
                "primary_cohort_id": EXPECTED_COHORT_ID,
                "config_sha256": context["config_sha256"],
                "split_membership_sha256": context["split_membership_sha256"],
                "gradient_checkpointing_during_training": context[
                    "technical_execution_options"
                ]["gradient_checkpointing"],
                "mps_low_watermark_ratio": context.get(
                    "mps_allocator", {}
                ).get("requested_low_watermark_ratio"),
                "adamw_foreach_mode": context[
                    "technical_execution_options"
                ].get("adamw_foreach_mode", "PYTORCH_DEFAULT"),
                "pad_to_multiple_of": context[
                    "technical_execution_options"
                ].get("pad_to_multiple_of"),
                "experiment_tag": context.get("experiment_tag"),
                "protocol_amendment_id": context.get(
                    "scientific_protocol", {}
                ).get("protocol_id"),
                "hyperparameter_selection_source": context.get(
                    "scientific_protocol", {}
                ).get("hyperparameter_source"),
            }
        )
    return rows


def runtime_environment(
    stack: Mapping[str, Any],
    hardware: Mapping[str, Any],
    mps_allocator: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": str(stack["torch"].__version__),
        "transformers": str(stack["transformers"].__version__),
        "hostname": socket.gethostname(),
        "hardware": dict(hardware),
        "mps_allocator_environment": validate_runtime_mps_allocator(
            mps_allocator
        ),
    }


def fixed_a2_scientific_protocol(
    spec: RunSpec,
    amendment_path: Path,
    a1_source: Mapping[str, Any],
    legacy_fold1: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not amendment_path.is_file():
        raise M2ProtocolError(f"Fixed A2 amendment is missing: {amendment_path}")
    amendment_sha256 = sha256_file(amendment_path)
    if amendment_sha256 != EXPECTED_FIXED_A2_AMENDMENT_SHA256:
        raise M2ProtocolError("Fixed A2 amendment bytes changed")
    protocol = {
        "protocol_id": FIXED_A2_PROTOCOL_ID,
        "protocol_version": "1.0.0",
        "decision_date": "2026-08-14",
        "affected_experiment": "M2_A2_ONLY",
        "amendment_path": display_path(amendment_path),
        "amendment_sha256": amendment_sha256,
        "reason": "DOCUMENTED_APPLE_MPS_COMPUTE_CONTINGENCY",
        "previous_protocol": "SIX_CONFIGURATION_SEARCH_PER_A2_FOLD",
        "revised_protocol": "FIXED_A1_SELECTED_HYPERPARAMETERS_FOR_ALL_A2_FOLDS",
        "hyperparameter_source": "A1_VALIDATION_TRANSFER",
        "official_configuration_index": FIXED_A2_CONFIGURATION_INDEX,
        "fixed_hyperparameters": dict(FIXED_A2_HYPERPARAMETERS),
        "per_fold_hyperparameter_search": False,
        "fold_validation_selects_epoch": True,
        "fold_validation_selects_global_threshold": True,
        "a2_test_results_used_for_amendment": False,
        "a2_test_labels_used_for_selection": False,
        "a1_unchanged": True,
        "a1_selection_provenance": dict(a1_source),
        "fold": spec.fold,
    }
    if legacy_fold1 is not None:
        protocol["fold1_legacy_c5_reuse"] = dict(legacy_fold1["source"])
    return protocol


def run_one(
    spec: RunSpec,
    benchmark: Sequence[dict[str, Any]],
    label_order: Sequence[str],
    config: Mapping[str, Any],
    token_info: Mapping[int, TokenInfo],
    benchmark_path: Path,
    ontology_path: Path,
    config_path: Path,
    token_audit_path: Path,
    *,
    force: bool,
    plan_only: bool,
    local_files_only: bool,
    gradient_checkpointing: bool,
    initial_train_batch_size: int,
    force_reason: str | None,
    max_length_override: int | None,
    max_length_override_acknowledged: bool,
    technical_override_rationale: str | None,
    mps_allocator: Mapping[str, Any],
    adamw_foreach_false: bool,
    pad_to_multiple_of: int | None,
    fixed_a2_hyperparameters_from_a1: bool = False,
    reexecute_fold1_test_inference_on_mps: bool = False,
    model_root: Path = DEFAULT_MODEL_ROOT,
    amendment_path: Path = DEFAULT_FIXED_A2_AMENDMENT,
) -> dict[str, Any]:
    if not spec.split_path.is_file():
        raise M2ProtocolError(f"Final split does not exist; M2 cannot run: {spec.split_path}")
    train, validation, test, split_metadata = validate_and_partition_split(
        spec, load_csv(spec.split_path), benchmark, label_order
    )
    if fixed_a2_hyperparameters_from_a1 and spec.evaluation != "A2":
        raise M2ProtocolError("Fixed A1-hyperparameter transfer mode is A2-only")
    a1_source = (
        validate_a1_fixed_transfer_source(model_root, config)
        if fixed_a2_hyperparameters_from_a1
        else None
    )
    legacy_fold1 = (
        validate_legacy_fold1_c5(
            spec, validation, label_order, split_metadata
        )
        if fixed_a2_hyperparameters_from_a1 and spec.fold == 1
        else None
    )
    max_length = max_length_override or EXPECTED_MAX_LENGTH
    if max_length not in {EXPECTED_MAX_LENGTH, 1536, 1024}:
        raise M2ProtocolError("M2 max length must be 2048, 1536, or 1024")
    if max_length < EXPECTED_MAX_LENGTH and not max_length_override_acknowledged:
        raise M2ProtocolError("Reduced M2 max length lacks explicit acknowledgement")
    if pad_to_multiple_of not in {None, 64}:
        raise M2ProtocolError("M2 padding multiple must be absent or exactly 64")
    if pad_to_multiple_of is not None and max_length % pad_to_multiple_of:
        raise M2ProtocolError(
            "M2 max length must be divisible by the requested padding multiple"
        )
    plan = {
        "run": spec.key,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "grid_configurations": (
            1 if fixed_a2_hyperparameters_from_a1 else 6
        ),
        "configuration_indices": (
            [FIXED_A2_CONFIGURATION_INDEX]
            if fixed_a2_hyperparameters_from_a1
            else list(range(1, 7))
        ),
        "experiment_tag": (
            FIXED_A2_PROTOCOL_ID
            if fixed_a2_hyperparameters_from_a1
            else None
        ),
        "hyperparameter_selection_source": (
            "A1_VALIDATION_TRANSFER"
            if fixed_a2_hyperparameters_from_a1
            else "CURRENT_FOLD_VALIDATION_GRID"
        ),
        "fold1_c5_reused_without_retraining": bool(
            fixed_a2_hyperparameters_from_a1 and spec.fold == 1
        ),
        "fresh_pretrained_initialization": True,
        "max_length": max_length,
        "max_length_reduced": max_length < EXPECTED_MAX_LENGTH,
        "gradient_checkpointing": gradient_checkpointing,
        "initial_train_batch_size": initial_train_batch_size,
        "mps_allocator": dict(mps_allocator),
        "adamw_foreach_mode": (
            "EXPLICIT_FALSE" if adamw_foreach_false else "PYTORCH_DEFAULT"
        ),
        "pad_to_multiple_of": pad_to_multiple_of,
        "progress_interval_batches": TRAIN_PROGRESS_INTERVAL_BATCHES,
        "batch_fallback_sequence": [
            attempt.train_batch_size
            for attempt in batch_attempts(
                config, initial_train_batch_size=initial_train_batch_size
            )
        ],
        "mixed_precision_policy": "CUDA_FP16_MPS_BF16_CPU_FP32",
        "split_membership_sha256": split_metadata["split_membership_sha256"],
        "prediction_path": str(spec.prediction_path),
        "model_dir": str(spec.model_dir),
    }
    if fixed_a2_hyperparameters_from_a1:
        fixed_a2_scientific_protocol(
            spec,
            amendment_path,
            a1_source or {},
            legacy_fold1,
        )
    if plan_only:
        return {"status": "PLAN_VALIDATED", **plan}

    if fixed_a2_hyperparameters_from_a1 and spec.fold == 1:
        preserve_legacy_fold1_metadata(spec)
        legacy_fold1 = validate_legacy_fold1_c5(
            spec, validation, label_order, split_metadata
        )

    stack = load_ml_stack()
    device, hardware = select_device(stack["torch"])
    if fixed_a2_hyperparameters_from_a1 and device.type != "mps":
        raise M2ProtocolError(
            "Fixed A2 contingency mode requires Apple MPS; observed "
            f"device={device.type}"
        )
    environment = runtime_environment(stack, hardware, mps_allocator)
    scientific_protocol = (
        fixed_a2_scientific_protocol(
            spec,
            amendment_path,
            a1_source or {},
            legacy_fold1,
        )
        if fixed_a2_hyperparameters_from_a1
        else None
    )
    context = execution_context(
        spec,
        benchmark_path,
        ontology_path,
        config_path,
        token_audit_path,
        split_metadata,
        label_order,
        gradient_checkpointing=gradient_checkpointing,
        initial_train_batch_size=initial_train_batch_size,
        execution_environment=environment,
        max_length=max_length,
        max_length_override_acknowledged=max_length_override_acknowledged,
        technical_override_rationale=(
            technical_override_rationale.strip()
            if technical_override_rationale
            else None
        ),
        mps_allocator=mps_allocator,
        adamw_foreach_false=adamw_foreach_false,
        pad_to_multiple_of=pad_to_multiple_of,
        scientific_protocol=scientific_protocol,
    )

    metadata_path = spec.model_dir / "run_metadata.json"
    artifact_dir = (
        spec.model_dir / "fixed_a1_hparams_v1"
        if fixed_a2_hyperparameters_from_a1
        else spec.model_dir
    )
    fit_state_path = artifact_dir / "fit_state.json"
    search_path = artifact_dir / "validation_hyperparameter_search.csv"
    threshold_path = artifact_dir / "validation_threshold_search.csv"
    tokenizer_dir = (
        spec.model_dir / "tokenizer"
        if fixed_a2_hyperparameters_from_a1 and spec.fold == 1
        else artifact_dir / "tokenizer"
    )
    technical_reinference_provenance: dict[str, Any] | None = None
    if reexecute_fold1_test_inference_on_mps:
        technical_reinference_provenance = (
            archive_fold1_cpu_inference_for_mps_reexecution(
                spec,
                current_hardware=hardware,
            )
        )
    restart_provenance: dict[str, Any] | None = None
    if force:
        if not force_reason or not force_reason.strip():
            raise M2ProtocolError("--force requires a non-empty --force-reason")
        restart_provenance = archive_forced_restart(
            spec,
            context=context,
            force_reason=force_reason.strip(),
        )
    if metadata_path.is_file() and not force:
        existing_metadata = load_json(metadata_path)
        is_fold1_legacy_predecessor = bool(
            fixed_a2_hyperparameters_from_a1
            and spec.fold == 1
            and existing_metadata.get("status") == "INTERRUPTED"
            and existing_metadata.get("experiment_tag") is None
            and existing_metadata.get("execution_context_sha256")
            == legacy_fold1["source"]["legacy_execution_context_sha256"]
        )
        if not is_fold1_legacy_predecessor and complete_run_is_valid(
            metadata_path,
            spec.prediction_path,
            context,
            recover_interrupted=fixed_a2_hyperparameters_from_a1,
        ):
            return {"status": "SKIPPED_COMPLETE", **plan}

    overall_started = time.perf_counter()
    started_at = utc_now()
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        metadata_path,
        {
            **context,
            "status": "IN_PROGRESS",
            "started_at": started_at,
            "execution_context_sha256": context_digest(context),
            "environment": environment,
            "forced_restart_provenance": restart_provenance,
            "technical_reinference_provenance": technical_reinference_provenance,
        },
    )

    fit_state: dict[str, Any] | None = None
    if fit_state_path.is_file() and not force:
        candidate = load_json(fit_state_path)
        if candidate.get("status") == "FIT_AND_VALIDATION_SELECTION_COMPLETE":
            validate_fit_state(candidate, context)
            fit_state = candidate

    if fit_state is None:
        if fixed_a2_hyperparameters_from_a1 and spec.fold == 1:
            if (
                legacy_fold1 is None
                or sha256_directory(tokenizer_dir)
                != legacy_fold1["source"]["tokenizer_sha256"]
            ):
                raise M2ProtocolError("Fold 1 legacy tokenizer is missing or damaged")
            tokenizer = stack["AutoTokenizer"].from_pretrained(
                tokenizer_dir,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            tokenizer_commit = EXPECTED_MODEL_REVISION
        else:
            tokenizer, tokenizer_commit = load_tokenizer(
                stack, config, local_files_only=local_files_only
            )
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_dir)
        fit_started = time.perf_counter()
        print(
            f"M2 {spec.key}: fit/selection stage start "
            f"protocol={context.get('experiment_tag', 'FROZEN_GRID')}",
            flush=True,
        )
        if fixed_a2_hyperparameters_from_a1:
            selection, search_rows, threshold_rows = fit_and_select_fixed_a2(
                spec,
                stack,
                tokenizer,
                train,
                validation,
                label_order,
                config,
                context,
                device,
                artifact_dir,
                legacy_fold1=legacy_fold1,
                local_files_only=local_files_only,
            )
        else:
            selection, search_rows, threshold_rows = fit_and_select(
                spec,
                stack,
                tokenizer,
                train,
                validation,
                label_order,
                config,
                context,
                device,
                force=force,
                local_files_only=local_files_only,
            )
        print(
            f"M2 {spec.key}: threshold_selection_complete "
            f"threshold={selection['selected_global_threshold']}",
            flush=True,
        )
        atomic_csv(search_path, search_rows)
        atomic_csv(threshold_path, threshold_rows)
        fit_state = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "FIT_AND_VALIDATION_SELECTION_COMPLETE",
            "completed_at": utc_now(),
            "execution_context_sha256": context_digest(context),
            "selection": selection,
            "fit_seconds": time.perf_counter() - fit_started,
            "tokenizer_path": display_path(tokenizer_dir),
            "tokenizer_sha256": sha256_directory(tokenizer_dir),
            "tokenizer_resolved_commit": tokenizer_commit,
            "technical_execution_options": context[
                "technical_execution_options"
            ],
            "scientific_protocol": context.get("scientific_protocol"),
            "training_reused_without_retraining": selection.get(
                "training_reused_without_retraining", False
            ),
            "forced_restart_provenance": restart_provenance,
            "technical_reinference_provenance": technical_reinference_provenance,
            "validation_hyperparameter_search_path": display_path(search_path),
            "validation_hyperparameter_search_sha256": sha256_file(search_path),
            "validation_threshold_search_path": display_path(threshold_path),
            "validation_threshold_search_sha256": sha256_file(threshold_path),
            "test_labels_used_for_selection": False,
        }
        atomic_json(fit_state_path, fit_state)
    else:
        tokenizer = stack["AutoTokenizer"].from_pretrained(
            resolve_artifact_path(str(fit_state["tokenizer_path"])),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )

    validate_fit_state(fit_state, context)
    selection = fit_state["selection"]
    threshold = float(selection["selected_global_threshold"])
    checkpoint = resolve_artifact_path(str(selection["selected_checkpoint_path"]))

    # Test data are first passed through a fitted model only after the immutable
    # fit-state artifact records both hyperparameter and threshold selection.
    prediction_started = time.perf_counter()
    print(
        f"M2 {spec.key}: test_inference_start n={len(test)} "
        f"threshold={threshold}",
        flush=True,
    )
    model = load_local_model(stack, checkpoint, device)
    test_dataset = encode_records(
        tokenizer, test, label_order, max_length
    )
    selected_batch = int(
        next(
            row["train_batch_size"]
            for row in load_csv(search_path)
            if row["selected"] == "1"
        )
    )
    eval_batch_size = min(
        int(config["training"]["per_device_eval_batch_size_initial"]),
        selected_batch * 2,
    )
    precision = precision_settings(
        device,
        bool(config["training"]["mixed_precision_where_supported"]),
    )
    mixed_precision = bool(precision["mixed_precision"])
    inference_mixed_precision_dtype = str(precision["mixed_precision_dtype"])
    probabilities, observed_test_labels, inference_batch_attempts = predict_with_batch_fallback(
        stack,
        model,
        test_dataset,
        tokenizer,
        device,
        initial_batch_size=eval_batch_size,
        seed=int(config["training"]["seed"]),
        mixed_precision=mixed_precision,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    expected_test_labels = target_matrix(test, label_order)
    if not np.array_equal(observed_test_labels, expected_test_labels):
        raise M2ProtocolError("Test loader changed target order")
    run_id = sha256_text(
        canonical_json(
            {
                "method": EXPECTED_METHOD_ID,
                "evaluation": spec.evaluation,
                "fold": spec.fold,
                "execution_context_sha256": context_digest(context),
                "checkpoint_sha256": selection["selected_checkpoint_sha256"],
                "threshold": threshold,
            }
        )
    )[:24]
    predictions = prediction_rows(
        spec,
        test,
        probabilities,
        label_order,
        threshold,
        run_id,
        context,
        token_info,
    )
    atomic_jsonl(spec.prediction_path, predictions)
    prediction_seconds = time.perf_counter() - prediction_started
    print(
        f"M2 {spec.key}: test_inference_complete rows={len(predictions)} "
        f"seconds={prediction_seconds:.1f}",
        flush=True,
    )
    del model, test_dataset
    clear_device_memory(stack["torch"], device)

    metadata = {
        **context,
        "status": "COMPLETE",
        "execution_context_sha256": context_digest(context),
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds_this_invocation": time.perf_counter() - overall_started,
        "fit_seconds": fit_state["fit_seconds"],
        "prediction_seconds": prediction_seconds,
        "selection": selection,
        "fit_state_path": display_path(fit_state_path),
        "fit_state_sha256": sha256_file(fit_state_path),
        "validation_hyperparameter_search_path": fit_state[
            "validation_hyperparameter_search_path"
        ],
        "validation_hyperparameter_search_sha256": fit_state[
            "validation_hyperparameter_search_sha256"
        ],
        "validation_threshold_search_path": fit_state[
            "validation_threshold_search_path"
        ],
        "validation_threshold_search_sha256": fit_state[
            "validation_threshold_search_sha256"
        ],
        "prediction_path": display_path(spec.prediction_path),
        "prediction_sha256": sha256_file(spec.prediction_path),
        "prediction_rows": len(predictions),
        "test_truncated_n": sum(row["truncated_input"] for row in predictions),
        "test_original_token_count_min": min(
            row["original_token_count"] for row in predictions
        ),
        "test_original_token_count_max": max(
            row["original_token_count"] for row in predictions
        ),
        "test_inference_batch_attempts": inference_batch_attempts,
        "test_inference_mixed_precision_dtype": inference_mixed_precision_dtype,
        "environment": environment,
        "forced_restart_provenance": restart_provenance,
        "technical_reinference_provenance": technical_reinference_provenance,
        "protocol_attestation": {
            "architecture": "one_shared_ModernBERT_encoder_one_17_logit_head",
            "pretrained_model_id": EXPECTED_MODEL_ID,
            "pinned_revision": EXPECTED_MODEL_REVISION,
            "fresh_pretrained_initialization_for_each_grid_trial": True,
            "fresh_pretrained_initialization_for_each_a2_fold": True,
            "weights_shared_across_a2_folds": False,
            "input_field": "text_input.english_fact_summary_raw",
            "max_length": max_length,
            "max_length_reduced": max_length < EXPECTED_MAX_LENGTH,
            "max_length_override_acknowledged": (
                max_length_override_acknowledged
            ),
            "max_length_override_rationale": (
                technical_override_rationale.strip()
                if technical_override_rationale
                else None
            ),
            "gradient_checkpointing": gradient_checkpointing,
            "mps_allocator": dict(mps_allocator),
            "adamw_foreach_mode": selection[
                "selected_training_execution"
            ]["adamw_foreach_mode"],
            "adamw_foreach_observed_param_group_value": selection[
                "selected_training_execution"
            ]["adamw_foreach_observed_param_group_value"],
            "adamw_fused_observed_param_group_value": selection[
                "selected_training_execution"
            ]["adamw_fused_observed_param_group_value"],
            "pad_to_multiple_of": pad_to_multiple_of,
            "tokenizer_padding_side": selection[
                "selected_training_execution"
            ]["tokenizer_padding_side"],
            "tokenizer_pad_token_id": selection[
                "selected_training_execution"
            ]["tokenizer_pad_token_id"],
            "gradient_checkpointing_addendum_id": (
                GRADIENT_CHECKPOINTING_ADDENDUM_ID
                if gradient_checkpointing
                else None
            ),
            "mixed_precision_dtype": inference_mixed_precision_dtype,
            "gradient_scaler_enabled": (
                precision["gradient_scaler_enabled"]
            ),
            "model_config_has_use_cache": selection[
                "selected_training_execution"
            ]["model_config_has_use_cache"],
            "model_use_cache_action": selection[
                "selected_training_execution"
            ]["model_use_cache_action"],
            "model_use_cache_during_training": selection[
                "selected_training_execution"
            ]["model_use_cache_during_training"],
            "accumulation_loss_scaling": selection[
                "selected_training_execution"
            ]["accumulation_loss_scaling"],
            "hyperparameters_selected_on": (
                "A1_VALIDATION_TRANSFER"
                if fixed_a2_hyperparameters_from_a1
                else "VALIDATION_ONLY"
            ),
            "hyperparameter_selection_metric": "macro_average_precision",
            "per_fold_hyperparameter_search": (
                not fixed_a2_hyperparameters_from_a1
            ),
            "official_configuration_index": (
                FIXED_A2_CONFIGURATION_INDEX
                if fixed_a2_hyperparameters_from_a1
                else selection["selected_configuration_index"]
            ),
            "a2_test_results_used_for_protocol_amendment": False,
            "threshold_selected_on": (
                "FOLD_VALIDATION_ONLY"
                if fixed_a2_hyperparameters_from_a1
                else "VALIDATION_ONLY"
            ),
            "threshold_selection_metric": "macro_f1",
            "single_global_threshold": True,
            "per_label_thresholds": False,
            "test_labels_used_for_model_or_threshold_selection": False,
            "test_predictions_created_after_fit_state": True,
            "baseline_threshold": BASELINE_THRESHOLD,
        },
    }
    atomic_json(metadata_path, metadata)
    print(
        f"M2 {spec.key}: COMPLETE epoch={selection['selected_best_epoch']} "
        f"threshold={threshold} predictions={len(predictions)}",
        flush=True,
    )
    return {
        "status": "COMPLETE",
        **plan,
        "run_id": run_id,
        "device": hardware["backend"],
        "selected_hyperparameters": selection["selected_hyperparameters"],
        "selected_best_epoch": selection["selected_best_epoch"],
        "selected_threshold": threshold,
        "validation_macro_average_precision": selection[
            "validation_macro_average_precision"
        ],
        "validation_macro_f1": selection[
            "validation_macro_f1_selected_threshold"
        ],
        "test_truncated_n": metadata["test_truncated_n"],
    }


def make_specs(args: argparse.Namespace) -> list[RunSpec]:
    if args.evaluation == "A1":
        requested: list[tuple[str, int | None]] = [("A1", None)]
    elif args.evaluation == "A2":
        requested = [("A2", fold) for fold in ([args.fold] if args.fold else [1, 2, 3])]
    else:
        if args.fold is not None:
            raise M2ProtocolError("--fold may be used only with --evaluation A2")
        requested = [("A1", None), ("A2", 1), ("A2", 2), ("A2", 3)]
    specs: list[RunSpec] = []
    for evaluation, fold in requested:
        if evaluation == "A1":
            specs.append(
                RunSpec(
                    evaluation="A1",
                    fold=None,
                    split_path=args.a1_split,
                    model_dir=args.model_root / "a1",
                    prediction_path=args.prediction_root / "a1_test_predictions.jsonl",
                )
            )
        else:
            specs.append(
                RunSpec(
                    evaluation="A2",
                    fold=fold,
                    split_path=args.a2_split,
                    model_dir=args.model_root / f"a2_fold_{fold}",
                    prediction_path=args.prediction_root
                    / f"a2_fold_{fold}_test_predictions.jsonl",
                )
            )
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        choices=("A1", "A2", "all"),
        required=True,
        help="Run A1, A2 (all folds unless --fold), or all four evaluations.",
    )
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    parser.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument(
        "--a2-fixed-hyperparameters-from-a1",
        action="store_true",
        help=(
            "Execute the documented M2-A2 compute contingency: reuse Fold-1 "
            "configuration 05 and train only A1-selected configuration 05 for "
            "Folds 2 and 3. Valid only with --evaluation A2 and the known-stable "
            "2048-token MPS execution settings."
        ),
    )
    parser.add_argument(
        "--a2-amendment",
        type=Path,
        default=DEFAULT_FIXED_A2_AMENDMENT,
        help="Pinned compute-contingency amendment used by fixed A2 mode.",
    )
    parser.add_argument(
        "--reexecute-fold1-test-inference-on-mps",
        action="store_true",
        help=(
            "One-time recovery for a fixed-protocol Fold-1 inference attempt "
            "that completed on CPU when a sandbox hid MPS. Byte-archives the "
            "fixed derivatives and predictions, preserves legacy C5, and reruns "
            "selection reconstruction plus test inference on MPS without training."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate frozen inputs and print plans without importing PyTorch or training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recoverably archive an incomplete/failed run and begin fresh. "
            "COMPLETE artifacts are never replaced. Requires --force-reason."
        ),
    )
    parser.add_argument(
        "--force-reason",
        help="Required provenance rationale whenever --force is used.",
    )
    parser.add_argument(
        "--break-stale-lock",
        action="store_true",
        help="Archive an existing run lock after independently confirming it is stale.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned Hugging Face model and tokenizer to be present locally.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable the documented hardware-memory addendum: activation gradient "
            "checkpointing with use_reentrant=false and use_cache=false where that "
            "model-config attribute exists. "
            "This changes no model, input, split, selection, or max-length setting."
        ),
    )
    parser.add_argument(
        "--initial-train-batch-size",
        type=int,
        choices=(4, 2, 1),
        default=4,
        help=(
            "Begin the frozen 4->2->1 batch fallback at this size; gradient "
            "accumulation remains 4/8/16 for effective-batch target 16."
        ),
    )
    parser.add_argument(
        "--mps-low-watermark-ratio",
        help=(
            "Opt in to a positive MPS allocator low-watermark ratio strictly "
            "below 1.4. Applied and verified before the first torch import. "
            "High-watermark overrides remain prohibited."
        ),
    )
    parser.add_argument(
        "--adamw-foreach-false",
        action="store_true",
        help=(
            "Explicitly select torch.optim.AdamW's foreach=False implementation "
            "path without changing its algorithm or frozen hyperparameters."
        ),
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        choices=(64,),
        help=(
            "Round each dynamically padded batch length to 64-token buckets. "
            "Attention masks and the recorded max-length cap remain unchanged."
        ),
    )
    parser.add_argument(
        "--max-length-override",
        type=int,
        choices=(1536, 1024),
        help=(
            "Opt-in hardware override after documented 2048-token failure. The "
            "default remains the frozen 2048."
        ),
    )
    parser.add_argument(
        "--acknowledge-max-length-reduction",
        action="store_true",
        help="Required explicit acknowledgement with --max-length-override.",
    )
    parser.add_argument(
        "--technical-override-rationale",
        help="Required non-empty rationale with --max-length-override.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fold is not None and args.evaluation != "A2":
        raise M2ProtocolError("--fold may be used only with --evaluation A2")
    if args.force and not (args.force_reason and args.force_reason.strip()):
        raise M2ProtocolError("--force requires a non-empty --force-reason")
    if args.force_reason and not args.force:
        raise M2ProtocolError("--force-reason may be used only with --force")
    if args.a2_fixed_hyperparameters_from_a1:
        if args.evaluation != "A2":
            raise M2ProtocolError(
                "--a2-fixed-hyperparameters-from-a1 requires --evaluation A2"
            )
        if args.force:
            raise M2ProtocolError("Fixed A2 mode prohibits --force")
        try:
            requested_low_watermark = Decimal(
                str(args.mps_low_watermark_ratio)
            )
        except ArithmeticError as error:
            raise M2ProtocolError(
                "Fixed A2 mode requires MPS low watermark 1.0"
            ) from error
        stable = {
            "gradient_checkpointing": args.gradient_checkpointing,
            "initial_train_batch_size": args.initial_train_batch_size == 1,
            "mps_low_watermark_ratio": requested_low_watermark == Decimal("1.0"),
            "adamw_foreach_false": args.adamw_foreach_false,
            "pad_to_multiple_of": args.pad_to_multiple_of == 64,
            "max_length_2048": args.max_length_override is None,
        }
        if not all(stable.values()):
            raise M2ProtocolError(
                "Fixed A2 mode requires the proven A1 settings: "
                "--gradient-checkpointing --initial-train-batch-size 1 "
                "--mps-low-watermark-ratio 1.0 --adamw-foreach-false "
                "--pad-to-multiple-of 64 and max_length=2048; observed "
                + canonical_json(stable)
            )
    if args.reexecute_fold1_test_inference_on_mps:
        if not args.a2_fixed_hyperparameters_from_a1:
            raise M2ProtocolError(
                "--reexecute-fold1-test-inference-on-mps requires fixed A2 mode"
            )
        if args.evaluation != "A2" or args.fold != 1:
            raise M2ProtocolError(
                "--reexecute-fold1-test-inference-on-mps requires "
                "--evaluation A2 --fold 1"
            )
        if args.plan:
            raise M2ProtocolError(
                "--reexecute-fold1-test-inference-on-mps is an execution-only mode"
            )
    if args.max_length_override is not None:
        if not args.acknowledge_max_length_reduction:
            raise M2ProtocolError(
                "--max-length-override requires --acknowledge-max-length-reduction"
            )
        if not (
            args.technical_override_rationale
            and args.technical_override_rationale.strip()
        ):
            raise M2ProtocolError(
                "--max-length-override requires a non-empty "
                "--technical-override-rationale"
            )
    elif args.acknowledge_max_length_reduction or args.technical_override_rationale:
        raise M2ProtocolError(
            "Max-length acknowledgement/rationale requires --max-length-override"
        )
    mps_allocator = configure_mps_allocator(args.mps_low_watermark_ratio)
    benchmark, label_order, config, token_info = validate_static_inputs(
        args.benchmark, args.ontology, args.config, args.token_audit
    )
    results: list[dict[str, Any]] = []
    for spec in make_specs(args):
        lock: RunLock | None = None
        if not args.plan:
            lock = RunLock.acquire(
                spec,
                break_stale_lock=args.break_stale_lock,
                execution_options={
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "initial_train_batch_size": args.initial_train_batch_size,
                    "max_length": args.max_length_override or EXPECTED_MAX_LENGTH,
                    "mps_allocator": mps_allocator,
                    "adamw_foreach_mode": (
                        "EXPLICIT_FALSE"
                        if args.adamw_foreach_false
                        else "PYTORCH_DEFAULT"
                    ),
                    "pad_to_multiple_of": args.pad_to_multiple_of,
                    "force": args.force,
                    "experiment_tag": (
                        FIXED_A2_PROTOCOL_ID
                        if args.a2_fixed_hyperparameters_from_a1
                        else None
                    ),
                },
            )
        try:
            try:
                result = run_one(
                    spec,
                    benchmark,
                    label_order,
                    config,
                    token_info,
                    args.benchmark,
                    args.ontology,
                    args.config,
                    args.token_audit,
                    force=args.force,
                    plan_only=args.plan,
                    local_files_only=args.local_files_only,
                    gradient_checkpointing=args.gradient_checkpointing,
                    initial_train_batch_size=args.initial_train_batch_size,
                    force_reason=args.force_reason,
                    max_length_override=args.max_length_override,
                    max_length_override_acknowledged=(
                        args.acknowledge_max_length_reduction
                    ),
                    technical_override_rationale=(
                        args.technical_override_rationale
                    ),
                    mps_allocator=mps_allocator,
                    adamw_foreach_false=args.adamw_foreach_false,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                    fixed_a2_hyperparameters_from_a1=(
                        args.a2_fixed_hyperparameters_from_a1
                    ),
                    reexecute_fold1_test_inference_on_mps=(
                        args.reexecute_fold1_test_inference_on_mps
                    ),
                    model_root=args.model_root,
                    amendment_path=args.a2_amendment,
                )
            except BaseException as error:
                if not args.plan:
                    mark_run_terminal_failure(spec, error)
                raise
            results.append(result)
        finally:
            if lock is not None:
                lock.release()
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M2ProtocolError as error:
        print(f"M2 protocol error: {error}", file=sys.stderr)
        raise SystemExit(2)
