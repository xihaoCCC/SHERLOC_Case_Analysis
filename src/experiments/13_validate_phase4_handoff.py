#!/usr/bin/env python3
"""Read-only completeness validator for the frozen Phase-4 AMP handoff.

This program is intentionally an artifact validator, not an evaluator.  It
does not train a model, call an API, inspect an API-key file, calculate a
scientific metric, or modify any repository artifact.  It checks that the
frozen inputs still have their approved byte hashes, that the canonical test
predictions are complete and share the frozen memberships, that the canonical
evaluator has recorded its final completion gate, and that the generated
reporting notebooks are current.

Exit status is 0 only for a complete handoff, 1 for a safely reported
incomplete handoff, and 2 for a validator invocation error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "1.0.0"
SCHEMA_VERSION = "sherloc-phase4-handoff-validation-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_PREDICTION_SCHEMA = "sherloc-amp-predictions-v1"
METHODS = ("M1", "M2", "M3", "M4")
EXPECTED_A1_TEST_N = 253
EXPECTED_A2_TEST_N_BY_FOLD = {1: 288, 2: 287, 3: 286}
EXPECTED_A2_TEST_N = sum(EXPECTED_A2_TEST_N_BY_FOLD.values())

# These are the exact, approved Phase-4 freeze bytes.  Keeping the hashes here
# makes this check independent of a coordinated edit to the freeze generator
# and its output files.
FROZEN_ARTIFACT_HASHES: dict[str, str] = {
    "data/processed/sherloc_benchmark_v1.jsonl": (
        "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
    ),
    "config/amp_ontology_v1.yaml": (
        "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
    ),
    "data/annotations/demo_bank_review_v2.csv": (
        "c7e793e781c77bde4f99507b66b6ffeb5e37de768c86fd27f58c9e5cdf5e242f"
    ),
    "config/experiments/m1_tfidf_logreg_amp_v2.yaml": (
        "44e80edf844d1589dec8b7236d58a65666f6479f0156d3c7ffff9e9de6d74b46"
    ),
    "config/experiments/m2_modernbert_amp_v2.yaml": (
        "73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3"
    ),
    "config/experiments/demo_bank_amp_v1.yaml": (
        "1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b"
    ),
    "prompts/m3_zero_shot_amp_v2.md": (
        "00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85"
    ),
    "prompts/m4_six_shot_amp_v2.md": (
        "2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37"
    ),
    "config/experiments/llm_extraction_amp_v2.yaml": (
        "5da03305ad97b36723c331ade7092147c828365abb32346b14a36726496d330b"
    ),
    "data/splits/a1_iid_split_final_v1.csv": (
        "63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d"
    ),
    "data/splits/a2_jurisdiction_folds_final_v1.csv": (
        "75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67"
    ),
    "docs/experiment_freeze_v1.md": (
        "b715d9cadb9b45b832597b5fb04c8418762c715064433d707804941bd3f0df95"
    ),
}

PRIMARY_NOTEBOOKS = (
    "07_a1_amp_results.ipynb",
    "08_a2_amp_results.ipynb",
    "09_amp_error_analysis.ipynb",
)
AUXILIARY_NOTEBOOK = "10_auxiliary_results.ipynb"

CANONICAL_METRIC_FILES = (
    "amp_evaluation_manifest.json",
    "a1/amp_primary_results.csv",
    "a1/amp_per_label.csv",
    "a1/amp_bootstrap_cis.csv",
    "a1/amp_case_level_errors.csv",
    "a2/amp_primary_results.csv",
    "a2/amp_per_fold.csv",
    "a2/amp_per_label.csv",
    "a2/amp_per_jurisdiction.csv",
    "a2/amp_bootstrap_cis.csv",
    "a2/amp_case_level_errors.csv",
    "amp_a1_to_a2_deltas.csv",
    "amp_threshold_0_50_sensitivity.csv",
)

SERIALIZED_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SECRET_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?:^|[\"'\s,{])"
    rb"(?:openai_api_key|api_key|authorization|authorization_header|"
    rb"bearer_token|client_secret|access_token|secret_key|api_secret)"
    rb"[\"']?\s*[:=]"
)
FORBIDDEN_SECRET_FIELDS = {
    "openai_api_key",
    "api_key",
    "authorization",
    "authorization_header",
    "bearer_token",
    "client_secret",
    "access_token",
    "secret_key",
    "api_secret",
}
SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "openai_key_like_value",
        re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    ),
    (
        "bearer_credential_like_value",
        re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}"),
    ),
)
SECRET_BEARING_FILENAMES = {
    "api.txt",
    ".env",
    ".env.local",
    ".env.production",
}


class ValidationInputError(RuntimeError):
    """Raised when a validator input cannot be interpreted safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(
    stage_id: str,
    status: str,
    message: str,
    *,
    failures: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"PASSED", "MISSING", "FAILED"}:
        raise ValueError(f"Invalid stage status: {status}")
    return {
        "stage_id": stage_id,
        "status": status,
        "message": message,
        "failures": list(failures),
        "details": dict(details or {}),
    }


def validate_frozen_artifacts(
    repo_root: Path,
    expected_hashes: Mapping[str, str] = FROZEN_ARTIFACT_HASHES,
) -> dict[str, Any]:
    missing: list[str] = []
    mismatches: list[str] = []
    observed: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = repo_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        digest = sha256_file(path)
        observed[relative] = digest
        if digest != expected:
            mismatches.append(relative)
    failures = [
        *(f"missing frozen artifact: {path}" for path in missing),
        *(f"frozen byte hash mismatch: {path}" for path in mismatches),
    ]
    if mismatches:
        status = "FAILED"
    elif missing:
        status = "MISSING"
    else:
        status = "PASSED"
    return _stage(
        "freeze",
        status,
        (
            "All frozen artifacts are present with approved SHA-256 hashes."
            if status == "PASSED"
            else "The deterministic Phase-4 freeze is not intact."
        ),
        failures=failures,
        details={
            "checked_artifact_count": len(expected_hashes),
            "missing_artifacts": missing,
            "hash_mismatches": mismatches,
            "observed_sha256": observed,
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationInputError(f"Cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise ValidationInputError(f"CSV has no data rows: {path}")
    return rows


def load_frozen_test_memberships(
    repo_root: Path,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[int, dict[str, str]]]]:
    """Load A1 and fold-specific A2 TEST identities without reading labels."""

    a1_path = repo_root / "data/splits/a1_iid_split_final_v1.csv"
    a2_path = repo_root / "data/splits/a2_jurisdiction_folds_final_v1.csv"
    if not a1_path.is_file() or not a2_path.is_file():
        raise ValidationInputError("Final A1/A2 split files are unavailable")

    a1: dict[int, dict[str, str]] = {}
    for row in _read_csv(a1_path):
        if str(row.get("split", "")).upper() != "TEST":
            continue
        try:
            rank = int(row["search_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationInputError("A1 TEST row has invalid search_rank") from exc
        if rank in a1:
            raise ValidationInputError(f"Duplicate A1 TEST search_rank {rank}")
        a1[rank] = row
    if len(a1) != EXPECTED_A1_TEST_N:
        raise ValidationInputError(
            f"A1 split has {len(a1)} TEST rows; expected {EXPECTED_A1_TEST_N}"
        )

    a2: dict[int, dict[int, dict[str, str]]] = {1: {}, 2: {}, 3: {}}
    for row in _read_csv(a2_path):
        if str(row.get("role", "")).upper() != "TEST":
            continue
        try:
            fold = int(row["fold_id"])
            rank = int(row["search_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationInputError("A2 TEST row has invalid fold/search_rank") from exc
        if fold not in a2:
            raise ValidationInputError(f"A2 TEST row has invalid fold {fold}")
        if rank in a2[fold]:
            raise ValidationInputError(f"Duplicate A2 Fold {fold} TEST rank {rank}")
        a2[fold][rank] = row
    for fold, expected_n in EXPECTED_A2_TEST_N_BY_FOLD.items():
        if len(a2[fold]) != expected_n:
            raise ValidationInputError(
                f"A2 Fold {fold} has {len(a2[fold])} TEST rows; expected {expected_n}"
            )
    pooled = [rank for fold_rows in a2.values() for rank in fold_rows]
    if len(pooled) != len(set(pooled)):
        raise ValidationInputError("A2 pooled TEST membership contains duplicate ranks")
    return a1, a2


def prediction_path(
    repo_root: Path, method: str, evaluation: str, fold: int | None = None
) -> Path:
    method_dir = repo_root / "outputs/predictions" / method.lower()
    if evaluation == "A1":
        return method_dir / "a1_test_predictions.jsonl"
    if evaluation == "A2" and fold in (1, 2, 3):
        return method_dir / f"a2_fold_{fold}_test_predictions.jsonl"
    raise ValueError(f"Invalid evaluation/fold: {evaluation}/{fold}")


def canonical_prediction_paths(repo_root: Path) -> set[Path]:
    paths: set[Path] = set()
    for method in METHODS:
        paths.add(prediction_path(repo_root, method, "A1"))
        for fold in (1, 2, 3):
            paths.add(prediction_path(repo_root, method, "A2", fold))
    return paths


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationInputError(
                        f"Malformed JSONL at {path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValidationInputError(
                        f"Prediction row is not an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeError) as exc:
        raise ValidationInputError(f"Cannot read prediction file {path}: {exc}") from exc
    if not rows:
        raise ValidationInputError(f"Prediction file has no rows: {path}")
    return rows


def _validate_prediction_file(
    path: Path,
    *,
    method: str,
    evaluation: str,
    fold: int | None,
    expected: Mapping[int, Mapping[str, str]],
) -> tuple[dict[int, int | None], list[str], dict[str, Any]]:
    rows = _read_prediction_rows(path)
    observed: dict[int, int | None] = {}
    failures: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        prefix = f"{path.name}:{line_number}"
        try:
            rank = int(row.get("search_rank"))
        except (TypeError, ValueError):
            failures.append(f"{prefix} has invalid search_rank")
            continue
        if rank in observed:
            failures.append(f"{prefix} duplicates search_rank {rank}")
            continue
        observed_fold = row.get("fold")
        if observed_fold not in (None, ""):
            try:
                observed_fold = int(observed_fold)
            except (TypeError, ValueError):
                failures.append(f"{prefix} has invalid fold")
                continue
        else:
            observed_fold = None
        observed[rank] = observed_fold

        row_method = str(row.get("method_id") or row.get("method") or "").upper()
        if row_method != method:
            failures.append(f"{prefix} method is {row_method!r}, expected {method}")
        if str(row.get("evaluation", "")).upper() != evaluation:
            failures.append(f"{prefix} is not evaluation {evaluation}")
        if str(row.get("split", "")).upper() != "TEST":
            failures.append(f"{prefix} is not a TEST prediction")
        if row.get("prediction_schema_version") != EXPECTED_PREDICTION_SCHEMA:
            failures.append(f"{prefix} has an unexpected prediction schema")
        if row.get("primary_cohort_id") != EXPECTED_COHORT_ID:
            failures.append(f"{prefix} has an unexpected primary cohort ID")
        if observed_fold != fold:
            failures.append(f"{prefix} fold is {observed_fold!r}, expected {fold!r}")
        if "predicted_labels" not in row or "silver_reference_labels" not in row:
            failures.append(f"{prefix} lacks prediction/reference label arrays")
        if method in {"M3", "M4"} and row.get("status") != "SUCCESS_VALIDATED":
            failures.append(f"{prefix} is not a validated LLM success")

        split_row = expected.get(rank)
        if split_row is None:
            continue
        expected_url = str(split_row.get("canonical_url", ""))
        if row.get("canonical_url") != expected_url:
            failures.append(f"{prefix} canonical URL differs from frozen split")
        expected_jurisdiction = str(split_row.get("jurisdiction", ""))
        if row.get("jurisdiction") != expected_jurisdiction:
            failures.append(f"{prefix} jurisdiction differs from frozen split")

    observed_ranks = set(observed)
    expected_ranks = set(expected)
    missing = sorted(expected_ranks - observed_ranks)
    extra = sorted(observed_ranks - expected_ranks)
    if missing:
        failures.append(
            f"missing {len(missing)} frozen TEST ranks (first: {missing[:10]})"
        )
    if extra:
        failures.append(f"contains {len(extra)} extra ranks (first: {extra[:10]})")
    details = {
        "path": str(path),
        "expected_n": len(expected),
        "observed_n": len(rows),
        "unique_search_rank_n": len(observed),
        "missing_rank_n": len(missing),
        "extra_rank_n": len(extra),
        "sha256": sha256_file(path),
    }
    return observed, failures, details


def validate_method_predictions(
    repo_root: Path,
    method: str,
    evaluation: str,
    a1_expected: Mapping[int, Mapping[str, str]],
    a2_expected: Mapping[int, Mapping[int, Mapping[str, str]]],
) -> tuple[dict[str, Any], dict[int, int | None]]:
    missing_paths: list[str] = []
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    pooled_observed: dict[int, int | None] = {}
    folds: Iterable[int | None] = (None,) if evaluation == "A1" else (1, 2, 3)
    for fold in folds:
        path = prediction_path(repo_root, method, evaluation, fold)
        if not path.is_file():
            missing_paths.append(str(path.relative_to(repo_root)))
            continue
        expected = a1_expected if evaluation == "A1" else a2_expected[int(fold)]
        try:
            observed, file_failures, detail = _validate_prediction_file(
                path,
                method=method,
                evaluation=evaluation,
                fold=fold,
                expected=expected,
            )
        except ValidationInputError as exc:
            failures.append(str(exc))
            continue
        overlap = set(pooled_observed) & set(observed)
        if overlap:
            failures.append(
                f"{method} {evaluation} repeats ranks across files: {sorted(overlap)[:10]}"
            )
        pooled_observed.update(observed)
        failures.extend(file_failures)
        files.append(detail)

    if failures:
        status = "FAILED"
    elif missing_paths:
        status = "MISSING"
    else:
        status = "PASSED"
    stage_id = f"{method.lower()}_{evaluation.lower()}_predictions"
    stage = _stage(
        stage_id,
        status,
        (
            f"{method} {evaluation} canonical predictions are complete."
            if status == "PASSED"
            else f"{method} {evaluation} canonical predictions are incomplete."
        ),
        failures=[
            *(f"missing prediction artifact: {path}" for path in missing_paths),
            *failures,
        ],
        details={
            "missing_files": missing_paths,
            "files": files,
            "pooled_unique_search_rank_n": len(pooled_observed),
        },
    )
    return stage, pooled_observed


def validate_common_memberships(
    prediction_stages: Sequence[dict[str, Any]],
    observed: Mapping[tuple[str, str], Mapping[int, int | None]],
) -> dict[str, Any]:
    incomplete = [
        stage["stage_id"] for stage in prediction_stages if stage["status"] != "PASSED"
    ]
    failures: list[str] = []
    for evaluation in ("A1", "A2"):
        baseline = dict(observed.get((METHODS[0], evaluation), {}))
        for method in METHODS[1:]:
            candidate = dict(observed.get((method, evaluation), {}))
            if baseline and candidate and candidate != baseline:
                failures.append(
                    f"{evaluation} TEST/fold membership differs between M1 and {method}"
                )
    if failures:
        status = "FAILED"
    elif incomplete:
        status = "MISSING"
    else:
        status = "PASSED"
    return _stage(
        "common_test_membership",
        status,
        (
            "M1-M4 share the frozen A1 and A2 TEST/fold memberships."
            if status == "PASSED"
            else "Common M1-M4 TEST membership cannot yet be certified."
        ),
        failures=[
            *(f"incomplete prerequisite: {item}" for item in incomplete),
            *failures,
        ],
        details={
            "a1_expected_n": EXPECTED_A1_TEST_N,
            "a2_expected_n": EXPECTED_A2_TEST_N,
            "incomplete_prediction_stages": incomplete,
        },
    )


def _resolve_manifest_input(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def validate_evaluator_completion(repo_root: Path) -> dict[str, Any]:
    metrics_root = repo_root / "outputs/metrics"
    manifest_path = metrics_root / "amp_evaluation_manifest.json"
    missing = [
        relative
        for relative in CANONICAL_METRIC_FILES
        if not (metrics_root / relative).is_file()
    ]
    failures: list[str] = []
    details: dict[str, Any] = {"missing_files": missing}
    if not manifest_path.is_file():
        return _stage(
            "canonical_evaluator_completion_gate",
            "MISSING",
            "The canonical final evaluator manifest is not available.",
            failures=[
                f"missing canonical evaluator artifact: outputs/metrics/{item}"
                for item in missing
            ],
            details=details,
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _stage(
            "canonical_evaluator_completion_gate",
            "FAILED",
            "The canonical evaluator manifest is unreadable.",
            failures=[str(exc)],
            details=details,
        )
    if not isinstance(manifest, dict):
        failures.append("canonical evaluator manifest is not an object")
        manifest = {}

    expected_gate = "PASSED_M1_M2_M3_M4_A1_A2"
    if manifest.get("final_completion_gate") != expected_gate:
        failures.append(f"final_completion_gate is not {expected_gate}")
    evaluations = manifest.get("evaluations", {})
    if not isinstance(evaluations, dict):
        evaluations = {}
        failures.append("evaluator manifest evaluations is not an object")
    for evaluation, expected_n in (
        ("A1", EXPECTED_A1_TEST_N),
        ("A2", EXPECTED_A2_TEST_N),
    ):
        value = evaluations.get(evaluation, {})
        if not isinstance(value, dict):
            value = {}
        if value.get("methods") != list(METHODS):
            failures.append(f"evaluator {evaluation} methods are not M1-M4 in order")
        if value.get("test_n") != expected_n:
            failures.append(f"evaluator {evaluation} test_n is not {expected_n}")
    split_validation = manifest.get("split_validation", {})
    if not isinstance(split_validation, dict):
        split_validation = {}
    if split_validation.get("a1_final_split_validated") is not True:
        failures.append("canonical evaluator did not validate the final A1 split")
    if split_validation.get("a2_final_split_validated") is not True:
        failures.append("canonical evaluator did not validate the final A2 split")

    canonical_inputs = {path.resolve() for path in canonical_prediction_paths(repo_root)}
    recorded_inputs: set[Path] = set()
    for item in manifest.get("input_files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            failures.append("evaluator input_files contains an invalid entry")
            continue
        path = _resolve_manifest_input(repo_root, item["path"]).resolve()
        recorded_inputs.add(path)
        try:
            path.relative_to((repo_root / "outputs/predictions").resolve())
        except ValueError:
            failures.append("evaluator manifest references a non-prediction-root input")
            continue
        if "auxiliary" in {part.casefold() for part in path.parts}:
            failures.append("evaluator manifest references an auxiliary prediction")
        if not path.is_file():
            failures.append(f"recorded evaluator input is missing: {path.name}")
        elif item.get("sha256") != sha256_file(path):
            failures.append(f"recorded evaluator input hash differs: {path.name}")
    if recorded_inputs != canonical_inputs:
        failures.append("evaluator input_files is not the exact 16-file M1-M4 set")

    details.update(
        {
            "final_completion_gate": manifest.get("final_completion_gate"),
            "a1_methods": evaluations.get("A1", {}).get("methods")
            if isinstance(evaluations.get("A1", {}), dict)
            else None,
            "a2_methods": evaluations.get("A2", {}).get("methods")
            if isinstance(evaluations.get("A2", {}), dict)
            else None,
            "recorded_input_file_n": len(recorded_inputs),
        }
    )
    if failures:
        status = "FAILED"
    elif missing:
        status = "MISSING"
    else:
        status = "PASSED"
    return _stage(
        "canonical_evaluator_completion_gate",
        status,
        (
            "Canonical evaluator completion gate and output set are complete."
            if status == "PASSED"
            else "Canonical evaluator completion is not certified."
        ),
        failures=[
            *(f"missing canonical metric artifact: {item}" for item in missing),
            *failures,
        ],
        details=details,
    )


def validate_notebooks(repo_root: Path) -> dict[str, Any]:
    notebook_dir = repo_root / "notebooks"
    generator = repo_root / "src/experiments/12_generate_analysis_notebooks.py"
    missing = [name for name in PRIMARY_NOTEBOOKS if not (notebook_dir / name).is_file()]
    if not generator.is_file():
        missing.append("src/experiments/12_generate_analysis_notebooks.py")
    if missing:
        return _stage(
            "analysis_notebooks",
            "MISSING",
            "Required generated notebooks or their generator are missing.",
            failures=[f"missing notebook artifact: {item}" for item in missing],
            details={"missing_files": missing, "generator_check_ran": False},
        )

    command = [sys.executable, str(generator), "--output-dir", str(notebook_dir), "--check"]
    if (notebook_dir / AUXILIARY_NOTEBOOK).is_file():
        command.append("--include-auxiliary")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _stage(
            "analysis_notebooks",
            "FAILED",
            "The deterministic notebook generator check could not run.",
            failures=[str(exc)],
            details={"generator_check_ran": False},
        )
    if result.returncode != 0:
        # The generator output names files but contains no secrets.  Keep only
        # its last short line so this handoff report cannot become a log dump.
        output = (result.stderr or result.stdout).strip().splitlines()
        diagnostic = output[-1][:500] if output else "generator check returned nonzero"
        return _stage(
            "analysis_notebooks",
            "FAILED",
            "Generated notebook bytes are missing or stale.",
            failures=[diagnostic],
            details={"generator_check_ran": True, "returncode": result.returncode},
        )
    checked = [*PRIMARY_NOTEBOOKS]
    if (notebook_dir / AUXILIARY_NOTEBOOK).is_file():
        checked.append(AUXILIARY_NOTEBOOK)
    return _stage(
        "analysis_notebooks",
        "PASSED",
        "Required notebooks exactly match the deterministic generator.",
        details={"generator_check_ran": True, "checked_notebooks": checked},
    )


def validate_primary_auxiliary_separation(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    prediction_root = repo_root / "outputs/predictions"
    metrics_root = repo_root / "outputs/metrics"
    canonical_predictions = {path.resolve() for path in canonical_prediction_paths(repo_root)}

    for root, auxiliary_root in (
        (prediction_root, prediction_root / "auxiliary"),
        (metrics_root, metrics_root / "auxiliary"),
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative_parts = path.relative_to(root).parts
            has_auxiliary_name = any("auxiliary" in part.casefold() for part in relative_parts)
            under_auxiliary_root = False
            try:
                path.resolve().relative_to(auxiliary_root.resolve())
                under_auxiliary_root = True
            except ValueError:
                pass
            if has_auxiliary_name and not under_auxiliary_root:
                failures.append(
                    f"auxiliary-named artifact is outside {auxiliary_root.relative_to(repo_root)}: "
                    f"{path.relative_to(repo_root)}"
                )

    for path in canonical_predictions:
        if "auxiliary" in {part.casefold() for part in path.parts}:
            failures.append("a canonical primary prediction path is under auxiliary")

    manifest_path = metrics_root / "amp_evaluation_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            manifest = {}
        for item in manifest.get("input_files", []) if isinstance(manifest, dict) else []:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            path = _resolve_manifest_input(repo_root, item["path"]).resolve()
            if "auxiliary" in {part.casefold() for part in path.parts}:
                failures.append("canonical primary evaluator includes an auxiliary input")

    return _stage(
        "primary_auxiliary_separation",
        "FAILED" if failures else "PASSED",
        (
            "Primary AMP and auxiliary artifact namespaces are separated."
            if not failures
            else "Primary and auxiliary artifact namespaces are mixed."
        ),
        failures=failures,
        details={
            "primary_prediction_root": "outputs/predictions/{m1,m2,m3,m4}",
            "auxiliary_prediction_root": "outputs/predictions/auxiliary",
            "primary_metric_root": "outputs/metrics/{a1,a2}",
            "auxiliary_metric_root": "outputs/metrics/auxiliary",
        },
    )


def _iter_serialized_files(repo_root: Path) -> Iterable[Path]:
    roots = [repo_root / "outputs", repo_root / "logs", repo_root / "notebooks"]
    phase4_report = repo_root / "docs/phase4_execution_report.md"
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink():
            yield root
            continue
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            # Do not descend through symlinked directories; they are reported
            # separately by the secret validator without reading their target.
            retained_directories: list[str] = []
            for name in dirnames:
                candidate = Path(directory) / name
                if candidate.is_symlink():
                    yield candidate
                else:
                    retained_directories.append(name)
            dirnames[:] = retained_directories
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.casefold() in SERIALIZED_SUFFIXES or path.name.casefold() in SECRET_BEARING_FILENAMES:
                    yield path
    if phase4_report.is_file() and not phase4_report.is_symlink():
        yield phase4_report


def _normalise_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _walk_forbidden_secret_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalised = _normalise_field_name(key)
            # Tokenizer vocabularies are serialized as token->integer maps and
            # legitimately contain lexical keys such as "authorization".
            # Credential-bearing fields instead hold strings/null/containers.
            if normalised in FORBIDDEN_SECRET_FIELDS and not isinstance(
                child, (int, float, bool)
            ):
                found.add(normalised)
            found.update(_walk_forbidden_secret_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_walk_forbidden_secret_fields(child))
    return found


def _serialized_secret_fields(path: Path, payload: bytes) -> set[str]:
    """Return field names only; never return or render credential values."""

    suffix = path.suffix.casefold()
    found: set[str] = set()
    if suffix in {".json", ".ipynb"}:
        try:
            value = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError):
            if FORBIDDEN_SECRET_ASSIGNMENT_RE.search(payload):
                found.add("unparsed_secret_assignment")
        else:
            found.update(_walk_forbidden_secret_fields(value))
        return found
    if suffix == ".jsonl":
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                if FORBIDDEN_SECRET_ASSIGNMENT_RE.search(line):
                    found.add("unparsed_secret_assignment")
            else:
                found.update(_walk_forbidden_secret_fields(value))
        return found
    if suffix == ".csv":
        try:
            decoded = payload.decode("utf-8-sig")
            header = next(csv.reader(decoded.splitlines()), [])
        except (UnicodeError, csv.Error):
            if FORBIDDEN_SECRET_ASSIGNMENT_RE.search(payload):
                found.add("unparsed_secret_assignment")
        else:
            found.update(
                normalised
                for field in header
                if (normalised := _normalise_field_name(field))
                in FORBIDDEN_SECRET_FIELDS
            )
        return found
    if FORBIDDEN_SECRET_ASSIGNMENT_RE.search(payload):
        found.add("secret_assignment")
    return found


def validate_no_serialized_secrets(repo_root: Path) -> dict[str, Any]:
    """Scan generated serializations without opening the repository api.txt."""

    failures: list[str] = []
    scanned = 0
    total_bytes = 0
    repo_api_path = (repo_root / "api.txt").resolve()
    for path in _iter_serialized_files(repo_root):
        if path.is_symlink():
            failures.append(f"unscanned serialized symlink: {path.relative_to(repo_root)}")
            continue
        if path.name.casefold() in SECRET_BEARING_FILENAMES:
            failures.append(f"secret-bearing filename in generated artifacts: {path.relative_to(repo_root)}")
            continue
        # This explicit guard is defense in depth: api.txt is outside every
        # scan root, and even a future path-list change must not open it.
        if path.resolve() == repo_api_path:
            failures.append("repository api.txt was unexpectedly included in the scan plan")
            continue
        try:
            payload = path.read_bytes()
        except OSError as exc:
            failures.append(f"cannot scan serialized artifact {path.relative_to(repo_root)}: {exc}")
            continue
        scanned += 1
        total_bytes += len(payload)
        secret_fields = _serialized_secret_fields(path, payload)
        if secret_fields:
            failures.append(
                "forbidden secret field serialized in "
                f"{path.relative_to(repo_root)} (field names: {sorted(secret_fields)})"
            )
        for pattern_name, pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(payload):
                failures.append(
                    f"{pattern_name} detected in {path.relative_to(repo_root)}"
                )
    return _stage(
        "api_secret_serialization",
        "FAILED" if failures else "PASSED",
        (
            "No API secret fields or credential-like values were found in generated serializations."
            if not failures
            else "Generated serializations cannot be certified free of API secrets."
        ),
        failures=failures,
        details={
            "scanned_file_n": scanned,
            "scanned_byte_n": total_bytes,
            "repository_api_txt_read": False,
            "scan_roots": ["outputs", "logs", "notebooks", "docs/phase4_execution_report.md"],
        },
    )


def validate_handoff(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    stages: list[dict[str, Any]] = [validate_frozen_artifacts(repo_root)]
    prediction_stages: list[dict[str, Any]] = []
    observed: dict[tuple[str, str], dict[int, int | None]] = {}
    try:
        a1_expected, a2_expected = load_frozen_test_memberships(repo_root)
    except ValidationInputError as exc:
        a1_expected = {}
        a2_expected = {1: {}, 2: {}, 3: {}}
        for method in METHODS:
            for evaluation in ("A1", "A2"):
                stage = _stage(
                    f"{method.lower()}_{evaluation.lower()}_predictions",
                    "MISSING",
                    "Prediction completeness cannot be checked without final splits.",
                    failures=[str(exc)],
                )
                prediction_stages.append(stage)
                observed[(method, evaluation)] = {}
    else:
        for method in METHODS:
            for evaluation in ("A1", "A2"):
                stage, membership = validate_method_predictions(
                    repo_root,
                    method,
                    evaluation,
                    a1_expected,
                    a2_expected,
                )
                prediction_stages.append(stage)
                observed[(method, evaluation)] = membership
    stages.extend(prediction_stages)
    stages.append(validate_common_memberships(prediction_stages, observed))
    stages.append(validate_evaluator_completion(repo_root))
    stages.append(validate_notebooks(repo_root))
    stages.append(validate_primary_auxiliary_separation(repo_root))
    stages.append(validate_no_serialized_secrets(repo_root))

    missing_stages = [stage["stage_id"] for stage in stages if stage["status"] == "MISSING"]
    failed_stages = [stage["stage_id"] for stage in stages if stage["status"] == "FAILED"]
    status = "COMPLETE" if not missing_stages and not failed_stages else "INCOMPLETE"
    return {
        "schema_version": SCHEMA_VERSION,
        "validator_version": VERSION,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "status": status,
        "complete": status == "COMPLETE",
        "missing_stages": missing_stages,
        "failed_stages": failed_stages,
        "stage_counts": {
            state: sum(stage["status"] == state for stage in stages)
            for state in ("PASSED", "MISSING", "FAILED")
        },
        "stages": stages,
        "scope_guards": {
            "scientific_metrics_recomputed": False,
            "api_calls": 0,
            "models_trained": 0,
            "frozen_artifacts_modified": 0,
            "repository_api_txt_read": False,
        },
    }


def render_text_summary(report: Mapping[str, Any]) -> str:
    lines = [f"PHASE 4 HANDOFF: {report['status']}"]
    for stage in report["stages"]:
        lines.append(f"[{stage['status']}] {stage['stage_id']}: {stage['message']}")
        for failure in stage["failures"]:
            lines.append(f"  - {failure}")
    if report["status"] == "INCOMPLETE":
        lines.append(
            "Missing stages: " + (", ".join(report["missing_stages"]) or "none")
        )
        lines.append(
            "Failed stages: " + (", ".join(report["failed_stages"]) or "none")
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Print a concise text report or the full machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.repo_root.is_dir():
        print(f"ERROR: repository root is not a directory: {args.repo_root}", file=sys.stderr)
        return 2
    report = validate_handoff(args.repo_root)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text_summary(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
