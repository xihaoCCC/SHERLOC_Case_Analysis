#!/usr/bin/env python3
"""Evaluate M1--M4 against SHERLOC Legacy AMP silver-reference labels.

This is the canonical A1/A2 evaluation entry point.  It reads completed test
prediction artifacts, validates their identities and references against the
final split files when those files are present, and writes reusable tables for
the analysis notebooks.  It never trains a model, calls an API, or tunes a
threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support package imports and ``python src/experiments/11_...py``.
    from .bootstrap import (
        BOOTSTRAP_METRICS,
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        percentile_bootstrap_confidence_intervals,
    )
    from .metrics import (
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        ORGAN_REMOVAL_LABEL,
        SILVER_REFERENCE_TERM,
        MetricInputError,
        compute_amp_cpmr,
        compute_amp_metrics,
        compute_case_errors,
        labels_to_indicator,
        supported_label_ids,
    )
except ImportError:  # pragma: no cover - used by direct CLI invocation.
    from bootstrap import (  # type: ignore
        BOOTSTRAP_METRICS,
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_BOOTSTRAP_SEED,
        percentile_bootstrap_confidence_intervals,
    )
    from metrics import (  # type: ignore
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        ORGAN_REMOVAL_LABEL,
        SILVER_REFERENCE_TERM,
        MetricInputError,
        compute_amp_cpmr,
        compute_amp_metrics,
        compute_case_errors,
        labels_to_indicator,
        supported_label_ids,
    )


VERSION = "1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/metrics"
DEFAULT_A1_SPLIT = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
DEFAULT_A2_SPLIT = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
DEFAULT_CPMR_ADDENDUM = REPO_ROOT / "docs/cpmr_metric_addendum_v1.md"

PRIMARY_VARIANT = "PRIMARY"
FIXED_050_VARIANT = "THRESHOLD_0_50"
METHOD_ORDER = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
CPMR_FAMILY_KEYS = (("ACT", "act"), ("MEANS", "means"), ("PURPOSE", "purpose"))


class EvaluationError(RuntimeError):
    """Raised when canonical evaluation cannot safely proceed."""


@dataclass(frozen=True)
class PredictionRecord:
    source_path: Path
    source_row: int
    method: str
    evaluation: str
    fold: int | None
    prediction_variant: str
    search_rank: int
    case_id: str
    canonical_url: str
    jurisdiction: str
    fact_summary: str
    silver_reference_labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    truncated_input: bool


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"Malformed JSON at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise EvaluationError(f"Prediction row is not an object at {path}:{line_number}")
                rows.append(value)
        return rows
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise EvaluationError(f"Unsupported prediction artifact type: {path}")


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    return None


def _parse_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return []
    if stripped[0] in "[{\"" or stripped in ("null", "true", "false"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


def _ordered_label_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    value = _parse_json_if_string(value)
    if isinstance(value, Mapping):
        # Strict LLM AMP schema or a wrapper containing it.
        if all(key in value for key in ("acts", "means", "purposes")):
            family_values: list[str] = []
            for family_key, family_name in (
                ("acts", "ACT"),
                ("means", "MEANS"),
                ("purposes", "PURPOSE"),
            ):
                raw_family = _parse_json_if_string(value[family_key])
                if not isinstance(raw_family, Sequence) or isinstance(raw_family, str):
                    raise EvaluationError(f"{field}.{family_key} is not an array")
                for label in raw_family:
                    if AMP_FAMILY_BY_LABEL.get(str(label)) != family_name:
                        raise EvaluationError(
                            f"{field}.{family_key} contains invalid label {label!r}"
                        )
                    family_values.append(str(label))
            value = family_values
        elif "normalized_prediction" in value:
            return _ordered_label_tuple(value["normalized_prediction"], field=field)
        elif "amp" in value:
            return _ordered_label_tuple(value["amp"], field=field)
        else:
            unknown = set(value) - set(AMP_LABEL_IDS)
            if unknown:
                raise EvaluationError(f"{field} map has unknown labels: {sorted(unknown)}")
            selected: list[str] = []
            for label in AMP_LABEL_IDS:
                raw_flag = value.get(label, 0)
                if raw_flag not in (0, 1, False, True, "0", "1"):
                    raise EvaluationError(f"{field}[{label}] is not binary")
                if str(raw_flag) == "1" or raw_flag is True:
                    selected.append(label)
            value = selected

    if isinstance(value, str):
        # A single ontology ID is accepted; delimiter parsing is deliberately
        # not guessed because it could hide malformed API output.
        value = [value]
    if not isinstance(value, Sequence):
        raise EvaluationError(f"{field} is not a label collection or 17-vector")
    values = list(value)
    if len(values) == len(AMP_LABEL_IDS) and all(
        item in (0, 1, False, True, "0", "1") for item in values
    ):
        values = [
            label
            for label, item in zip(AMP_LABEL_IDS, values, strict=True)
            if str(item) == "1" or item is True
        ]
    labels = [str(item) for item in values]
    unknown = set(labels) - set(AMP_LABEL_IDS)
    if unknown:
        raise EvaluationError(f"{field} contains unknown labels: {sorted(unknown)}")
    if len(labels) != len(set(labels)):
        raise EvaluationError(f"{field} contains duplicate labels")
    selected = set(labels)
    return tuple(label for label in AMP_LABEL_IDS if label in selected)


def _parse_bool(value: Any, *, field: str) -> bool:
    if value in (None, "", False, 0, "0", "false", "False", "NO", "No"):
        return False
    if value in (True, 1, "1", "true", "True", "YES", "Yes"):
        return True
    raise EvaluationError(f"{field} is not boolean: {value!r}")


def _infer_token(path: Path, pattern: str) -> str | None:
    match = re.search(pattern, path.as_posix(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _normalise_prediction_row(
    row: Mapping[str, Any], path: Path, row_number: int
) -> list[PredictionRecord]:
    reference_raw = _first(
        row,
        ("silver_reference_labels", "silver_reference_amp", "reference_labels", "y_true"),
    )
    primary_raw = _first(
        row,
        (
            "predicted_labels",
            "predictions_tuned",
            "prediction_labels",
            "normalized_prediction",
            "amp_prediction",
            "prediction",
        ),
    )
    if reference_raw is None and primary_raw is None:
        return []  # A non-prediction JSONL/CSV encountered during auto-discovery.
    if reference_raw is None or primary_raw is None:
        error_status = _first(row, ("error_status", "status", "download_status"))
        raise EvaluationError(
            f"Incomplete test prediction at {path}:{row_number}; "
            f"reference/prediction missing (status={error_status!r})"
        )

    method = str(_first(row, ("method_id", "method")) or "").upper()
    if not method:
        method = _infer_token(path, r"(?:^|[/_-])(m[1-4])(?:[/_.-]|$)") or ""
    if method not in METHOD_ORDER:
        raise EvaluationError(f"Cannot identify M1--M4 method at {path}:{row_number}")

    evaluation = str(_first(row, ("evaluation", "evaluation_id")) or "").upper()
    if not evaluation:
        evaluation = _infer_token(path, r"(?:^|[/_-])(a[12])(?:[/_.-]|$)") or ""
    if evaluation not in ("A1", "A2"):
        raise EvaluationError(f"Cannot identify A1/A2 evaluation at {path}:{row_number}")

    split = _first(row, ("split", "role"))
    if split is not None and str(split).upper() != "TEST":
        raise EvaluationError(
            f"Canonical evaluation accepts TEST rows only; got {split!r} at {path}:{row_number}"
        )

    fold_raw = _first(row, ("fold", "fold_id"))
    if fold_raw is None:
        fold_token = _infer_token(path, r"fold[_-]?([123])")
        fold = int(fold_token) if fold_token else None
    else:
        fold = int(fold_raw)
    if evaluation == "A1" and fold is not None:
        raise EvaluationError(f"A1 prediction unexpectedly has fold={fold}")
    if evaluation == "A2" and fold not in (1, 2, 3):
        raise EvaluationError(f"A2 prediction requires fold 1--3 at {path}:{row_number}")

    rank_raw = _first(row, ("search_rank", "case_id"))
    try:
        search_rank = int(rank_raw)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Missing/integer search_rank at {path}:{row_number}") from exc

    jurisdiction = str(_first(row, ("jurisdiction", "country")) or "").strip()
    if not jurisdiction:
        raise EvaluationError(f"Missing jurisdiction at {path}:{row_number}")
    fact_summary = str(
        _first(row, ("fact_summary", "english_fact_summary", "text", "input_text")) or ""
    )
    if not fact_summary:
        raise EvaluationError(f"Missing Fact Summary at {path}:{row_number}")

    base = PredictionRecord(
        source_path=path,
        source_row=row_number,
        method=method,
        evaluation=evaluation,
        fold=fold,
        prediction_variant=PRIMARY_VARIANT,
        search_rank=search_rank,
        case_id=str(_first(row, ("case_id",)) or search_rank),
        canonical_url=str(_first(row, ("canonical_url", "url")) or ""),
        jurisdiction=jurisdiction,
        fact_summary=fact_summary,
        silver_reference_labels=_ordered_label_tuple(
            reference_raw, field="silver_reference_labels"
        ),
        predicted_labels=_ordered_label_tuple(primary_raw, field="predicted_labels"),
        truncated_input=_parse_bool(
            _first(row, ("truncated_input", "truncated")), field="truncated_input"
        ),
    )
    records = [base]
    fixed_raw = _first(
        row,
        ("predicted_labels_0_50", "predictions_0_50", "prediction_0_50"),
    )
    if fixed_raw is not None:
        records.append(
            replace(
                base,
                prediction_variant=FIXED_050_VARIANT,
                predicted_labels=_ordered_label_tuple(
                    fixed_raw, field="predicted_labels_0_50"
                ),
            )
        )
    return records


def load_prediction_files(paths: Sequence[Path]) -> list[PredictionRecord]:
    """Load, normalize, and validate completed prediction files."""

    records: list[PredictionRecord] = []
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise EvaluationError(f"Prediction artifact does not exist: {path}")
        file_records: list[PredictionRecord] = []
        for row_number, row in enumerate(_read_rows(path), start=1):
            file_records.extend(_normalise_prediction_row(row, path, row_number))
        if file_records:
            records.extend(file_records)
    if not records:
        return []

    seen: dict[tuple[str, str, str, int], PredictionRecord] = {}
    for record in records:
        key = (
            record.method,
            record.evaluation,
            record.prediction_variant,
            record.search_rank,
        )
        if key in seen:
            previous = seen[key]
            raise EvaluationError(
                "Duplicate case prediction for "
                f"{key}: {previous.source_path}:{previous.source_row} and "
                f"{record.source_path}:{record.source_row}"
            )
        seen[key] = record
    return sorted(
        records,
        key=lambda row: (
            row.evaluation,
            METHOD_ORDER[row.method],
            row.prediction_variant,
            row.fold or 0,
            row.search_rank,
        ),
    )


def discover_prediction_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for suffix in ("*.jsonl", "*.csv")
        for path in root.rglob(suffix)
        if path.is_file()
    )


def _load_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_against_final_splits(
    records: Sequence[PredictionRecord],
    *,
    a1_split_path: Path = DEFAULT_A1_SPLIT,
    a2_split_path: Path = DEFAULT_A2_SPLIT,
) -> dict[str, Any]:
    """Validate prediction membership and silver labels against final splits."""

    diagnostics: dict[str, Any] = {
        "a1_final_split_validated": False,
        "a2_final_split_validated": False,
    }
    for evaluation, path in (("A1", a1_split_path), ("A2", a2_split_path)):
        evaluation_records = [
            record
            for record in records
            if record.evaluation == evaluation
            and record.prediction_variant == PRIMARY_VARIANT
        ]
        if not evaluation_records or not path.is_file():
            continue
        rows = _load_split_rows(path)
        expected: dict[int, dict[str, str]] = {}
        for row in rows:
            role = str(row.get("split") or row.get("role") or "").upper()
            if role != "TEST":
                continue
            rank = int(row["search_rank"])
            if evaluation == "A2":
                fold = int(row.get("fold_id") or row.get("fold") or 0)
                if fold not in (1, 2, 3):
                    raise EvaluationError(f"Invalid final A2 fold for rank {rank}")
                row = dict(row)
                row["_fold"] = str(fold)
            if rank in expected:
                raise EvaluationError(f"Duplicate final {evaluation} TEST rank {rank}")
            expected[rank] = row

        for method in sorted({record.method for record in evaluation_records}):
            method_rows = [record for record in evaluation_records if record.method == method]
            observed = {record.search_rank for record in method_rows}
            if observed != set(expected):
                missing = sorted(set(expected) - observed)
                extra = sorted(observed - set(expected))
                raise EvaluationError(
                    f"{method} {evaluation} TEST membership differs from final split: "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
            for record in method_rows:
                split_row = expected[record.search_rank]
                if evaluation == "A2" and record.fold != int(split_row["_fold"]):
                    raise EvaluationError(
                        f"{method} rank {record.search_rank} A2 fold mismatch"
                    )
                if split_row.get("jurisdiction") != record.jurisdiction:
                    raise EvaluationError(
                        f"{method} rank {record.search_rank} jurisdiction mismatch"
                    )
                if record.canonical_url and split_row.get("canonical_url") != record.canonical_url:
                    raise EvaluationError(
                        f"{method} rank {record.search_rank} canonical URL mismatch"
                    )
                if all(label in split_row for label in AMP_LABEL_IDS):
                    expected_labels = tuple(
                        label for label in AMP_LABEL_IDS if split_row[label] == "1"
                    )
                    if record.silver_reference_labels != expected_labels:
                        raise EvaluationError(
                            f"{method} rank {record.search_rank} silver reference differs from final split"
                        )
        diagnostics[f"{evaluation.lower()}_final_split_validated"] = True
        diagnostics[f"{evaluation.lower()}_final_split_path"] = str(path)
        diagnostics[f"{evaluation.lower()}_expected_test_n"] = len(expected)
    return diagnostics


def validate_common_test_membership(records: Sequence[PredictionRecord]) -> None:
    """Ensure all available methods use the same cases (and A2 folds)."""

    for evaluation in ("A1", "A2"):
        by_method: dict[str, dict[int, int | None]] = {}
        for method in sorted({row.method for row in records if row.evaluation == evaluation}):
            subset = [
                row
                for row in records
                if row.evaluation == evaluation
                and row.method == method
                and row.prediction_variant == PRIMARY_VARIANT
            ]
            if subset:
                by_method[method] = {row.search_rank: row.fold for row in subset}
        if len(by_method) < 2:
            continue
        first_method = sorted(by_method, key=METHOD_ORDER.__getitem__)[0]
        expected = by_method[first_method]
        for method, observed in by_method.items():
            if observed != expected:
                raise EvaluationError(
                    f"{evaluation} common TEST/fold membership differs between "
                    f"{first_method} and {method}"
                )


def _matrices(records: Sequence[PredictionRecord]) -> tuple[np.ndarray, np.ndarray]:
    reference = labels_to_indicator(
        [record.silver_reference_labels for record in records]
    )
    predicted = labels_to_indicator([record.predicted_labels for record in records])
    return reference, predicted


def _metric_row(
    method: str,
    variant: str,
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "prediction_variant": variant,
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "exact_set_accuracy": metrics["exact_set_accuracy"],
        "example_jaccard": metrics["example_jaccard"],
        "test_n": metrics["test_n"],
        "macro_label_count": metrics["macro_label_count"],
        "macro_label_ids_json": canonical_json(metrics["macro_label_ids"]),
        "zero_reference_support_label_ids_json": canonical_json(
            metrics["zero_reference_support_label_ids"]
        ),
        **_cpmr_columns(metrics),
        "reference_terminology": SILVER_REFERENCE_TERM,
    }
    for metric in BOOTSTRAP_METRICS:
        row[f"{metric}_ci_lower"] = bootstrap[metric]["ci_lower"] if bootstrap else ""
        row[f"{metric}_ci_upper"] = bootstrap[metric]["ci_upper"] if bootstrap else ""
    return row


PRIMARY_FIELDS = (
    "method",
    "prediction_variant",
    "macro_f1",
    "macro_f1_ci_lower",
    "macro_f1_ci_upper",
    "micro_f1",
    "micro_f1_ci_lower",
    "micro_f1_ci_upper",
    "exact_set_accuracy",
    "exact_set_accuracy_ci_lower",
    "exact_set_accuracy_ci_upper",
    "example_jaccard",
    "example_jaccard_ci_lower",
    "example_jaccard_ci_upper",
    "test_n",
    "macro_label_count",
    "macro_label_ids_json",
    "zero_reference_support_label_ids_json",
    "act_cpmr",
    "act_mean_contained_recall",
    "means_cpmr",
    "means_mean_contained_recall",
    "purpose_cpmr",
    "purpose_mean_contained_recall",
    "reference_terminology",
)


def _per_label_rows(
    method: str,
    metrics: Mapping[str, Any],
    *,
    scope: str,
    fold: int | str = "",
    jurisdiction: str = "",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in metrics["per_label"]:
        output.append(
            {
                "method": method,
                "scope": scope,
                "fold": fold,
                "jurisdiction": jurisdiction,
                "label_id": row["label_id"],
                "family": row["family"],
                "support": row["support"],
                "predicted_positive": row["predicted_positive"],
                "true_positive": row["true_positive"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
                "precision": row["precision"] if row["precision"] is not None else "N/A",
                "recall": row["recall"] if row["recall"] is not None else "N/A",
                "f1": row["f1"] if row["f1"] is not None else "N/A",
                "status": row["status"],
                "included_in_macro_f1": int(row["included_in_macro_f1"]),
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    return output


PER_LABEL_FIELDS = (
    "method",
    "scope",
    "fold",
    "jurisdiction",
    "label_id",
    "family",
    "support",
    "predicted_positive",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "status",
    "included_in_macro_f1",
    "reference_terminology",
)


def _bootstrap_rows(
    method: str,
    evaluation: str,
    intervals: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in BOOTSTRAP_METRICS:
        interval = intervals[metric]
        rows.append(
            {
                "method": method,
                "evaluation": evaluation,
                "scope": "TEST" if evaluation == "A1" else "POOLED_OOD_TEST",
                "metric": metric,
                "estimate": interval["estimate"],
                "ci_lower": interval["ci_lower"],
                "ci_upper": interval["ci_upper"],
                "confidence_level": interval["confidence_level"],
                "n_resamples": interval["n_resamples"],
                "seed": interval["seed"],
                "bootstrap_method": interval["method"],
                "macro_label_count": interval["macro_label_count"],
                "macro_label_ids": canonical_json(interval["macro_label_ids"]),
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    return rows


BOOTSTRAP_FIELDS = (
    "method",
    "evaluation",
    "scope",
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "n_resamples",
    "seed",
    "bootstrap_method",
    "macro_label_count",
    "macro_label_ids",
    "reference_terminology",
)


def _case_error_rows(records: Sequence[PredictionRecord]) -> list[dict[str, Any]]:
    reference, predicted = _matrices(records)
    errors = compute_case_errors(reference, predicted)
    cpmr_cases = compute_amp_cpmr(reference, predicted)["per_case"]
    output: list[dict[str, Any]] = []
    for record, error, cpmr_case in zip(records, errors, cpmr_cases, strict=True):
        output.append(
            {
                "method": record.method,
                "case_id": record.case_id,
                "search_rank": record.search_rank,
                "canonical_url": record.canonical_url,
                "jurisdiction": record.jurisdiction,
                "split": "TEST",
                "fold": record.fold if record.fold is not None else "",
                "fact_summary": record.fact_summary,
                "silver_reference_amp_json": canonical_json(
                    error["silver_reference_labels"]
                ),
                "predicted_amp_json": canonical_json(error["predicted_labels"]),
                "false_positive_labels_json": canonical_json(
                    error["false_positive_labels"]
                ),
                "false_negative_labels_json": canonical_json(
                    error["false_negative_labels"]
                ),
                "exact_set_correct": error["exact_set_correct"],
                "example_jaccard": error["example_jaccard"],
                "act_cpmr": cpmr_case["act_cpmr"],
                "act_contained_recall": (
                    cpmr_case["act_contained_recall"]
                    if cpmr_case["act_contained_recall"] is not None
                    else "N/A"
                ),
                "means_cpmr": cpmr_case["means_cpmr"],
                "means_contained_recall": (
                    cpmr_case["means_contained_recall"]
                    if cpmr_case["means_contained_recall"] is not None
                    else "N/A"
                ),
                "purpose_cpmr": cpmr_case["purpose_cpmr"],
                "purpose_contained_recall": (
                    cpmr_case["purpose_contained_recall"]
                    if cpmr_case["purpose_contained_recall"] is not None
                    else "N/A"
                ),
                "truncated_input": int(record.truncated_input),
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    return output


CASE_ERROR_FIELDS = (
    "method",
    "case_id",
    "search_rank",
    "canonical_url",
    "jurisdiction",
    "split",
    "fold",
    "fact_summary",
    "silver_reference_amp_json",
    "predicted_amp_json",
    "false_positive_labels_json",
    "false_negative_labels_json",
    "exact_set_correct",
    "example_jaccard",
    "act_cpmr",
    "act_contained_recall",
    "means_cpmr",
    "means_contained_recall",
    "purpose_cpmr",
    "purpose_contained_recall",
    "truncated_input",
    "reference_terminology",
)


def _evaluate_group(
    records: Sequence[PredictionRecord],
    *,
    macro_label_ids: Sequence[str] | None,
    n_resamples: int,
    seed: int,
    bootstrap: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
    reference, predicted = _matrices(records)
    metrics = compute_amp_metrics(
        reference, predicted, macro_label_ids=macro_label_ids
    )
    metrics["cpmr"] = compute_amp_cpmr(reference, predicted)
    intervals = (
        percentile_bootstrap_confidence_intervals(
            reference,
            predicted,
            macro_label_ids=metrics["macro_label_ids"],
            n_resamples=n_resamples,
            seed=seed,
        )
        if bootstrap
        else None
    )
    return metrics, intervals


def _cpmr_columns(
    metrics: Mapping[str, Any],
    *,
    include_success_counts: bool = False,
    prefix: str = "",
) -> dict[str, Any]:
    """Return stable wide CPMR columns for a metric group."""

    output: dict[str, Any] = {}
    for family, key in CPMR_FAMILY_KEYS:
        family_result = metrics["cpmr"]["by_family"][family]
        output[f"{prefix}{key}_cpmr"] = family_result["cpmr"]
        output[f"{prefix}{key}_mean_contained_recall"] = (
            family_result["mean_contained_recall"]
            if family_result["mean_contained_recall"] is not None
            else "N/A"
        )
        if include_success_counts:
            output[f"{prefix}{key}_cpmr_success_count"] = family_result["success_count"]
    return output


def _cpmr_result_row(
    method: str, evaluation: str, scope: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "method": method,
        "evaluation": evaluation,
        "scope": scope,
        "test_n": metrics["test_n"],
        **_cpmr_columns(metrics, include_success_counts=True),
        "reference_terminology": SILVER_REFERENCE_TERM,
    }


def _difference_or_na(m4_value: Any, m3_value: Any) -> float | str:
    if m4_value is None or m3_value is None:
        return "N/A"
    return float(m4_value) - float(m3_value)


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_complete_primary: bool = False,
) -> dict[str, Any]:
    """Evaluate normalized predictions and write all available canonical outputs."""

    if not records:
        raise EvaluationError("No prediction rows supplied")
    validate_common_test_membership(records)
    primary = [row for row in records if row.prediction_variant == PRIMARY_VARIANT]
    if not primary:
        raise EvaluationError("No PRIMARY test predictions supplied")
    if require_complete_primary:
        required = set(METHOD_ORDER)
        for evaluation in ("A1", "A2"):
            observed = {row.method for row in primary if row.evaluation == evaluation}
            missing = required - observed
            if missing:
                raise EvaluationError(
                    f"Final completion gate: {evaluation} is missing methods "
                    f"{sorted(missing, key=METHOD_ORDER.__getitem__)}"
                )

    manifest: dict[str, Any] = {
        "evaluator_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "unit": "test_case_with_all_17_labels",
            "method": "percentile_linear",
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "reference_terminology": SILVER_REFERENCE_TERM,
        "secondary_metrics": {
            "contained_partial_match_rate": {
                "status": "SECONDARY_DIAGNOSTIC",
                "families": [family for family, _ in CPMR_FAMILY_KEYS],
                "contained_recall_scope": "CPMR_SUCCESS_CASES_ONLY",
                "addendum_path": str(DEFAULT_CPMR_ADDENDUM.relative_to(REPO_ROOT)),
                "addendum_sha256": (
                    sha256_file(DEFAULT_CPMR_ADDENDUM)
                    if DEFAULT_CPMR_ADDENDUM.is_file()
                    else "MISSING"
                ),
            }
        },
        "final_completion_gate": (
            "PASSED_M1_M2_M3_M4_A1_A2"
            if require_complete_primary
            else "NOT_REQUESTED_PARTIAL_EVALUATION_ALLOWED"
        ),
        "evaluations": {},
    }
    aggregate_by_evaluation_method: dict[tuple[str, str], dict[str, Any]] = {}

    # ---------- A1 ----------
    a1_methods = sorted(
        {row.method for row in primary if row.evaluation == "A1"},
        key=METHOD_ORDER.__getitem__,
    )
    if a1_methods:
        directory = output_root / "a1"
        primary_rows: list[dict[str, Any]] = []
        per_label_rows: list[dict[str, Any]] = []
        bootstrap_rows: list[dict[str, Any]] = []
        case_rows: list[dict[str, Any]] = []
        cpmr_rows: list[dict[str, Any]] = []
        macro_labels: tuple[str, ...] | None = None
        for method in a1_methods:
            subset = sorted(
                [row for row in primary if row.evaluation == "A1" and row.method == method],
                key=lambda row: row.search_rank,
            )
            reference, _ = _matrices(subset)
            current_macro = supported_label_ids(reference)
            if macro_labels is None:
                macro_labels = current_macro
            elif current_macro != macro_labels:
                raise EvaluationError("A1 silver-reference support differs across methods")
            metrics, intervals = _evaluate_group(
                subset,
                macro_label_ids=macro_labels,
                n_resamples=n_resamples,
                seed=seed,
                bootstrap=True,
            )
            assert intervals is not None
            aggregate_by_evaluation_method[("A1", method)] = metrics
            primary_rows.append(_metric_row(method, PRIMARY_VARIANT, metrics, intervals))
            per_label_rows.extend(_per_label_rows(method, metrics, scope="TEST"))
            bootstrap_rows.extend(_bootstrap_rows(method, "A1", intervals))
            case_rows.extend(_case_error_rows(subset))
            cpmr_rows.append(_cpmr_result_row(method, "A1", "TEST", metrics))

        _atomic_csv(directory / "amp_primary_results.csv", primary_rows, PRIMARY_FIELDS)
        _atomic_csv(directory / "amp_per_label.csv", per_label_rows, PER_LABEL_FIELDS)
        _atomic_csv(directory / "amp_bootstrap_cis.csv", bootstrap_rows, tuple(dict.fromkeys(BOOTSTRAP_FIELDS)))
        _atomic_csv(directory / "amp_case_level_errors.csv", case_rows, CASE_ERROR_FIELDS)
        _atomic_csv(
            directory / "amp_cpmr_results.csv", cpmr_rows, tuple(cpmr_rows[0])
        )
        manifest["evaluations"]["A1"] = {
            "methods": a1_methods,
            "test_n": primary_rows[0]["test_n"],
            "macro_label_count": len(macro_labels or ()),
            "macro_label_ids": list(macro_labels or ()),
        }

    # ---------- A2 ----------
    a2_methods = sorted(
        {row.method for row in primary if row.evaluation == "A2"},
        key=METHOD_ORDER.__getitem__,
    )
    if a2_methods:
        directory = output_root / "a2"
        primary_rows = []
        per_label_rows = []
        bootstrap_rows = []
        case_rows = []
        fold_rows: list[dict[str, Any]] = []
        jurisdiction_rows: list[dict[str, Any]] = []
        cpmr_rows = []
        pooled_macro_labels: tuple[str, ...] | None = None
        pooled_zero_support: tuple[str, ...] | None = None
        for method in a2_methods:
            subset = sorted(
                [row for row in primary if row.evaluation == "A2" and row.method == method],
                key=lambda row: (row.fold or 0, row.search_rank),
            )
            reference, _ = _matrices(subset)
            current_macro = supported_label_ids(reference)
            current_zero = tuple(label for label in AMP_LABEL_IDS if label not in current_macro)
            unexpected_zero = set(current_zero) - {ORGAN_REMOVAL_LABEL}
            if unexpected_zero:
                raise EvaluationError(
                    "A2 has unexpected zero-support silver-reference labels: "
                    f"{sorted(unexpected_zero)}"
                )
            if pooled_macro_labels is None:
                pooled_macro_labels = current_macro
                pooled_zero_support = current_zero
            elif current_macro != pooled_macro_labels:
                raise EvaluationError("A2 pooled silver-reference support differs across methods")

            metrics, intervals = _evaluate_group(
                subset,
                macro_label_ids=pooled_macro_labels,
                n_resamples=n_resamples,
                seed=seed,
                bootstrap=True,
            )
            assert intervals is not None
            aggregate_by_evaluation_method[("A2", method)] = metrics

            fold_metrics: dict[int, dict[str, Any]] = {}
            for fold in (1, 2, 3):
                fold_subset = [row for row in subset if row.fold == fold]
                if not fold_subset:
                    raise EvaluationError(f"{method} A2 is missing Fold {fold} predictions")
                fold_result, _ = _evaluate_group(
                    fold_subset,
                    macro_label_ids=pooled_macro_labels,
                    n_resamples=n_resamples,
                    seed=seed,
                    bootstrap=False,
                )
                fold_metrics[fold] = fold_result
                fold_rows.append(
                    {
                        "method": method,
                        "fold": fold,
                        "macro_f1": fold_result["macro_f1"],
                        "micro_f1": fold_result["micro_f1"],
                        "exact_set_accuracy": fold_result["exact_set_accuracy"],
                        "example_jaccard": fold_result["example_jaccard"],
                        **_cpmr_columns(fold_result),
                        "test_n": fold_result["test_n"],
                        "macro_label_count": len(pooled_macro_labels),
                        "reference_terminology": SILVER_REFERENCE_TERM,
                    }
                )

            pooled_row = _metric_row(method, PRIMARY_VARIANT, metrics, intervals)
            primary_rows.append(
                {
                    "method": method,
                    "prediction_variant": PRIMARY_VARIANT,
                    "fold_1_macro_f1": fold_metrics[1]["macro_f1"],
                    "fold_2_macro_f1": fold_metrics[2]["macro_f1"],
                    "fold_3_macro_f1": fold_metrics[3]["macro_f1"],
                    "pooled_ood_macro_f1": pooled_row["macro_f1"],
                    "pooled_ood_macro_f1_ci_lower": pooled_row["macro_f1_ci_lower"],
                    "pooled_ood_macro_f1_ci_upper": pooled_row["macro_f1_ci_upper"],
                    "pooled_micro_f1": pooled_row["micro_f1"],
                    "pooled_micro_f1_ci_lower": pooled_row["micro_f1_ci_lower"],
                    "pooled_micro_f1_ci_upper": pooled_row["micro_f1_ci_upper"],
                    "pooled_exact_set_accuracy": pooled_row["exact_set_accuracy"],
                    "pooled_exact_set_accuracy_ci_lower": pooled_row["exact_set_accuracy_ci_lower"],
                    "pooled_exact_set_accuracy_ci_upper": pooled_row["exact_set_accuracy_ci_upper"],
                    "pooled_example_jaccard": pooled_row["example_jaccard"],
                    "pooled_example_jaccard_ci_lower": pooled_row["example_jaccard_ci_lower"],
                    "pooled_example_jaccard_ci_upper": pooled_row["example_jaccard_ci_upper"],
                    **_cpmr_columns(metrics, prefix="pooled_"),
                    "test_n": metrics["test_n"],
                    "macro_label_count": metrics["macro_label_count"],
                    "macro_label_ids_json": canonical_json(metrics["macro_label_ids"]),
                    "zero_reference_support_label_ids_json": canonical_json(
                        metrics["zero_reference_support_label_ids"]
                    ),
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
            )
            per_label_rows.extend(_per_label_rows(method, metrics, scope="POOLED_OOD_TEST"))
            bootstrap_rows.extend(_bootstrap_rows(method, "A2", intervals))
            case_rows.extend(_case_error_rows(subset))
            cpmr_rows.append(
                _cpmr_result_row(method, "A2", "POOLED_OOD_TEST", metrics)
            )

            for jurisdiction in sorted({row.jurisdiction for row in subset}):
                jurisdiction_subset = [row for row in subset if row.jurisdiction == jurisdiction]
                result, _ = _evaluate_group(
                    jurisdiction_subset,
                    macro_label_ids=pooled_macro_labels,
                    n_resamples=n_resamples,
                    seed=seed,
                    bootstrap=False,
                )
                jurisdiction_rows.append(
                    {
                        "method": method,
                        "jurisdiction": jurisdiction,
                        "fold": jurisdiction_subset[0].fold,
                        "macro_f1": result["macro_f1"],
                        "micro_f1": result["micro_f1"],
                        "exact_set_accuracy": result["exact_set_accuracy"],
                        "example_jaccard": result["example_jaccard"],
                        **_cpmr_columns(result),
                        "test_n": result["test_n"],
                        "macro_label_count": len(pooled_macro_labels),
                        "reference_terminology": SILVER_REFERENCE_TERM,
                    }
                )

        a2_fields = tuple(primary_rows[0])
        _atomic_csv(directory / "amp_primary_results.csv", primary_rows, a2_fields)
        _atomic_csv(directory / "amp_per_label.csv", per_label_rows, PER_LABEL_FIELDS)
        _atomic_csv(directory / "amp_bootstrap_cis.csv", bootstrap_rows, tuple(dict.fromkeys(BOOTSTRAP_FIELDS)))
        _atomic_csv(directory / "amp_per_fold.csv", fold_rows, tuple(fold_rows[0]))
        _atomic_csv(
            directory / "amp_per_jurisdiction.csv",
            jurisdiction_rows,
            tuple(jurisdiction_rows[0]),
        )
        _atomic_csv(directory / "amp_case_level_errors.csv", case_rows, CASE_ERROR_FIELDS)
        _atomic_csv(
            directory / "amp_cpmr_results.csv", cpmr_rows, tuple(cpmr_rows[0])
        )

        if {"M3", "M4"}.issubset(a2_methods):
            m3 = aggregate_by_evaluation_method[("A2", "M3")]
            m4 = aggregate_by_evaluation_method[("A2", "M4")]
            comparison_values: list[tuple[str, float, float]] = [
                ("macro_f1", m3["macro_f1"], m4["macro_f1"]),
                ("micro_f1", m3["micro_f1"], m4["micro_f1"]),
                (
                    "exact_set_accuracy",
                    m3["exact_set_accuracy"],
                    m4["exact_set_accuracy"],
                ),
                (
                    "example_jaccard",
                    m3["example_jaccard"],
                    m4["example_jaccard"],
                ),
            ]
            for family, key in CPMR_FAMILY_KEYS:
                m3_family = m3["cpmr"]["by_family"][family]
                m4_family = m4["cpmr"]["by_family"][family]
                comparison_values.append(
                    (f"{key}_cpmr", m3_family["cpmr"], m4_family["cpmr"])
                )
            comparison_rows = [
                {
                    "metric": metric,
                    "m3_zero_shot": m3_value,
                    "m4_six_shot": m4_value,
                    "delta_m4_minus_m3": m4_value - m3_value,
                    "test_n": m4["test_n"],
                    "significance_claim": "NOT_TESTED_DO_NOT_INFER",
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
                for metric, m3_value, m4_value in comparison_values
            ]
            _atomic_csv(
                directory / "amp_m3_vs_m4_aggregate_deltas.csv",
                comparison_rows,
                tuple(comparison_rows[0]),
            )

            m3_labels = {row["label_id"]: row for row in m3["per_label"]}
            m4_labels = {row["label_id"]: row for row in m4["per_label"]}
            label_delta_rows: list[dict[str, Any]] = []
            for label_id in AMP_LABEL_IDS:
                m3_label = m3_labels[label_id]
                m4_label = m4_labels[label_id]
                if m3_label["support"] != m4_label["support"]:
                    raise EvaluationError(
                        f"M3/M4 A2 silver-reference support differs for {label_id}"
                    )
                label_delta_rows.append(
                    {
                        "comparison": "M4_MINUS_M3",
                        "evaluation": "A2",
                        "scope": "POOLED_OOD_TEST",
                        "label_id": label_id,
                        "family": m3_label["family"],
                        "support": m3_label["support"],
                        "m3_predicted_positive": m3_label["predicted_positive"],
                        "m4_predicted_positive": m4_label["predicted_positive"],
                        "delta_predicted_positive": (
                            m4_label["predicted_positive"]
                            - m3_label["predicted_positive"]
                        ),
                        "m3_precision": (
                            m3_label["precision"]
                            if m3_label["precision"] is not None
                            else "N/A"
                        ),
                        "m4_precision": (
                            m4_label["precision"]
                            if m4_label["precision"] is not None
                            else "N/A"
                        ),
                        "delta_precision": _difference_or_na(
                            m4_label["precision"], m3_label["precision"]
                        ),
                        "m3_recall": (
                            m3_label["recall"]
                            if m3_label["recall"] is not None
                            else "N/A"
                        ),
                        "m4_recall": (
                            m4_label["recall"]
                            if m4_label["recall"] is not None
                            else "N/A"
                        ),
                        "delta_recall": _difference_or_na(
                            m4_label["recall"], m3_label["recall"]
                        ),
                        "m3_f1": (
                            m3_label["f1"] if m3_label["f1"] is not None else "N/A"
                        ),
                        "m4_f1": (
                            m4_label["f1"] if m4_label["f1"] is not None else "N/A"
                        ),
                        "delta_f1_m4_minus_m3": _difference_or_na(
                            m4_label["f1"], m3_label["f1"]
                        ),
                        "significance_claim": "NOT_TESTED_DO_NOT_INFER",
                        "reference_terminology": SILVER_REFERENCE_TERM,
                    }
                )
            _atomic_csv(
                directory / "amp_m3_vs_m4_per_label_deltas.csv",
                label_delta_rows,
                tuple(label_delta_rows[0]),
            )
        manifest["evaluations"]["A2"] = {
            "methods": a2_methods,
            "test_n": primary_rows[0]["test_n"],
            "macro_label_count": len(pooled_macro_labels or ()),
            "macro_label_ids": list(pooled_macro_labels or ()),
            "zero_reference_support_label_ids": list(pooled_zero_support or ()),
            "organ_removal_rule": (
                "N/A_PER_LABEL_F1_AND_EXCLUDED_FROM_MACRO"
                if ORGAN_REMOVAL_LABEL in (pooled_zero_support or ())
                else "SUPPORTED_AND_INCLUDED_IN_MACRO"
            ),
        }

    # ---------- fixed-0.50 threshold sensitivity ----------
    sensitivity_rows: list[dict[str, Any]] = []
    for evaluation in ("A1", "A2"):
        for method in sorted(
            {
                row.method
                for row in records
                if row.evaluation == evaluation
                and row.prediction_variant == FIXED_050_VARIANT
            },
            key=METHOD_ORDER.__getitem__,
        ):
            subset = sorted(
                [
                    row
                    for row in records
                    if row.evaluation == evaluation
                    and row.method == method
                    and row.prediction_variant == FIXED_050_VARIANT
                ],
                key=lambda row: (row.fold or 0, row.search_rank),
            )
            primary_subset = [
                row
                for row in primary
                if row.evaluation == evaluation and row.method == method
            ]
            if {row.search_rank for row in subset} != {
                row.search_rank for row in primary_subset
            }:
                raise EvaluationError(
                    f"{method} {evaluation} 0.50 sensitivity membership is incomplete"
                )
            primary_metrics = aggregate_by_evaluation_method[(evaluation, method)]
            result, _ = _evaluate_group(
                subset,
                macro_label_ids=primary_metrics["macro_label_ids"],
                n_resamples=n_resamples,
                seed=seed,
                bootstrap=False,
            )
            sensitivity_rows.append(
                {
                    "method": method,
                    "evaluation": evaluation,
                    "prediction_variant": FIXED_050_VARIANT,
                    "macro_f1": result["macro_f1"],
                    "micro_f1": result["micro_f1"],
                    "exact_set_accuracy": result["exact_set_accuracy"],
                    "example_jaccard": result["example_jaccard"],
                    "test_n": result["test_n"],
                    "macro_label_count": result["macro_label_count"],
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
            )
    if sensitivity_rows:
        _atomic_csv(
            output_root / "amp_threshold_0_50_sensitivity.csv",
            sensitivity_rows,
            tuple(sensitivity_rows[0]),
        )

    # ---------- A1 -> A2 raw deltas ----------
    delta_rows: list[dict[str, Any]] = []
    for method in sorted(
        set(a1_methods) & set(a2_methods), key=METHOD_ORDER.__getitem__
    ):
        a1 = aggregate_by_evaluation_method[("A1", method)]
        a2 = aggregate_by_evaluation_method[("A2", method)]
        delta_rows.append(
            {
                "method": method,
                "delta_macro_f1_a2_minus_a1": a2["macro_f1"] - a1["macro_f1"],
                "delta_micro_f1_a2_minus_a1": a2["micro_f1"] - a1["micro_f1"],
                "delta_exact_set_a2_minus_a1": (
                    a2["exact_set_accuracy"] - a1["exact_set_accuracy"]
                ),
                "delta_example_jaccard_a2_minus_a1": (
                    a2["example_jaccard"] - a1["example_jaccard"]
                ),
                "delta_act_cpmr_a2_minus_a1": (
                    a2["cpmr"]["by_family"]["ACT"]["cpmr"]
                    - a1["cpmr"]["by_family"]["ACT"]["cpmr"]
                ),
                "delta_means_cpmr_a2_minus_a1": (
                    a2["cpmr"]["by_family"]["MEANS"]["cpmr"]
                    - a1["cpmr"]["by_family"]["MEANS"]["cpmr"]
                ),
                "delta_purpose_cpmr_a2_minus_a1": (
                    a2["cpmr"]["by_family"]["PURPOSE"]["cpmr"]
                    - a1["cpmr"]["by_family"]["PURPOSE"]["cpmr"]
                ),
                "significance_claim": "NOT_TESTED_DO_NOT_INFER",
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    if delta_rows:
        _atomic_csv(
            output_root / "amp_a1_to_a2_deltas.csv",
            delta_rows,
            tuple(delta_rows[0]),
        )

    _atomic_json(output_root / "amp_evaluation_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-file",
        action="append",
        type=Path,
        default=[],
        help="Completed test prediction JSONL/CSV; repeat for multiple artifacts.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DEFAULT_PREDICTION_ROOT,
        help="Recursively discover prediction JSONL/CSV files when no explicit file is given.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    parser.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--skip-split-validation",
        action="store_true",
        help="Testing/development only; canonical project runs should validate final splits.",
    )
    parser.add_argument(
        "--require-complete-primary",
        action="store_true",
        help="Fail unless M1, M2, M3, and M4 are complete for both A1 and A2.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.prediction_file or discover_prediction_files(args.prediction_root)
    if not paths:
        print(f"No prediction artifacts found under {args.prediction_root}; nothing evaluated.")
        return 0
    try:
        records = load_prediction_files(paths)
        if not records:
            print("No completed prediction rows found; nothing evaluated.")
            return 0
        split_diagnostics = (
            validate_against_final_splits(
                records,
                a1_split_path=args.a1_split,
                a2_split_path=args.a2_split,
            )
            if not args.skip_split_validation
            else {"split_validation_skipped": True}
        )
        manifest = evaluate_predictions(
            records,
            output_root=args.output_root,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            require_complete_primary=args.require_complete_primary,
        )
        manifest["split_validation"] = split_diagnostics
        manifest["input_files"] = [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in paths
            if path.is_file()
        ]
        _atomic_json(args.output_root / "amp_evaluation_manifest.json", manifest)
    except (EvaluationError, MetricInputError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "Evaluated "
        + ", ".join(
            f"{evaluation}={','.join(details['methods'])}"
            for evaluation, details in manifest["evaluations"].items()
        )
        + f"; outputs: {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
