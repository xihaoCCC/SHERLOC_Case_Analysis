#!/usr/bin/env python3
"""Generate deterministic, presentation-ready A1 reporting artifacts.

This post-evaluation step consumes the canonical A1 outputs written by
``11_evaluate_amp.py`` plus the completed M2--M4 prediction/provenance files.
It does not call an API, calculate new model metrics, tune a method, or alter
any prediction.  Its calculations are limited to deterministic reshaping,
M4-minus-M3 descriptive differences between already-canonical metrics, and
aggregation of recorded API execution metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

try:  # Support package imports and direct execution.
    from .metrics import AMP_FAMILY_BY_LABEL, AMP_LABEL_IDS
except ImportError:  # pragma: no cover - direct CLI invocation.
    from metrics import AMP_FAMILY_BY_LABEL, AMP_LABEL_IDS  # type: ignore


VERSION = "1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = REPO_ROOT / "outputs/metrics"
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions"
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs/logs/llm"
DEFAULT_OUTPUT_DIR = DEFAULT_METRICS_ROOT / "a1"

EXPECTED_METHODS = ("M1", "M2", "M3", "M4")
LLM_METHODS = ("M3", "M4")
EXPECTED_A1_TEST_N = 253
AGGREGATE_METRICS = (
    "macro_f1",
    "micro_f1",
    "exact_set_accuracy",
    "example_jaccard",
    "act_cpmr",
    "means_cpmr",
    "purpose_cpmr",
)
CPMR_CASE_FIELDS = (
    "act_cpmr",
    "act_contained_recall",
    "means_cpmr",
    "means_contained_recall",
    "purpose_cpmr",
    "purpose_contained_recall",
)
PER_LABEL_METRICS = ("precision", "recall", "f1")
FAILURE_CLASSES = (
    "REFUSAL",
    "SCHEMA_ERROR",
    "API_ERROR",
    "TIMEOUT",
    "RATE_LIMIT_FAILURE",
    "OTHER_FAILURE",
)


class A1ReportingError(RuntimeError):
    """Raised when finalized A1 inputs cannot support a faithful report."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise A1ReportingError(f"Required finalized artifact is missing: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A1ReportingError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise A1ReportingError(f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require_file(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise A1ReportingError(f"Malformed JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise A1ReportingError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    _require_file(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _integer(value: Any, *, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise A1ReportingError(f"{field} is not an integer: {value!r}") from exc
    return result


def _number(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise A1ReportingError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise A1ReportingError(f"{field} is not finite: {value!r}")
    return result


def _parse_json_collection(value: Any, *, field: str) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise A1ReportingError(f"{field} is not valid JSON") from exc
    if not isinstance(value, list):
        raise A1ReportingError(f"{field} is not an array")
    return value


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise A1ReportingError("Cannot summarize an empty latency collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def _validate_evaluator_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    a1 = manifest.get("evaluations", {}).get("A1", {})
    split = manifest.get("split_validation", {})
    if not isinstance(a1, Mapping) or not isinstance(split, Mapping):
        raise A1ReportingError("Canonical evaluator manifest has malformed A1 metadata")
    if tuple(a1.get("methods", ())) != EXPECTED_METHODS:
        raise A1ReportingError("Canonical evaluator manifest is not complete for M1--M4 A1")
    if a1.get("test_n") != EXPECTED_A1_TEST_N:
        raise A1ReportingError("Canonical evaluator manifest A1 test_n is not 253")
    if a1.get("macro_label_count") != len(AMP_LABEL_IDS):
        raise A1ReportingError("Canonical evaluator manifest A1 label count is not 17")
    if tuple(a1.get("macro_label_ids", ())) != tuple(AMP_LABEL_IDS):
        raise A1ReportingError("Canonical evaluator manifest A1 label order changed")
    if (
        split.get("a1_final_split_validated") is not True
        or split.get("a1_expected_test_n") != EXPECTED_A1_TEST_N
    ):
        raise A1ReportingError("Canonical evaluator did not validate the final A1 split")
    return manifest


def _aggregate_delta_rows(primary_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    canonical_rows = [
        row for row in primary_rows if row.get("prediction_variant") == "PRIMARY"
    ]
    method_counts = Counter(str(row.get("method") or "") for row in canonical_rows)
    expected_methods = set(EXPECTED_METHODS)
    if set(method_counts) != expected_methods or any(
        method_counts[method] != 1 for method in EXPECTED_METHODS
    ):
        raise A1ReportingError(
            "Canonical A1 primary table must contain exactly one PRIMARY row "
            "for each of M1--M4"
        )
    selected = {str(row["method"]): row for row in canonical_rows}
    for method, row in selected.items():
        if _integer(row.get("test_n"), field=f"{method}.test_n") != EXPECTED_A1_TEST_N:
            raise A1ReportingError(f"{method} canonical A1 test_n is not 253")
    output: list[dict[str, Any]] = []
    for metric in AGGREGATE_METRICS:
        m3 = _number(selected["M3"].get(metric), field=f"M3.{metric}")
        m4 = _number(selected["M4"].get(metric), field=f"M4.{metric}")
        output.append(
            {
                "metric": metric,
                "m3_zero_shot": m3,
                "m4_six_shot": m4,
                "delta_m4_minus_m3": m4 - m3,
                "test_n": EXPECTED_A1_TEST_N,
                "comparison": "M4_SIX_SHOT_MINUS_M3_ZERO_SHOT",
                "significance_claim": "NOT_TESTED_DO_NOT_INFER",
            }
        )
    return output


def _per_label_outputs(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        method = str(row.get("method") or "")
        label = str(row.get("label_id") or "")
        key = (method, label)
        if key in by_key:
            raise A1ReportingError(f"Duplicate canonical per-label row: {key}")
        by_key[key] = row
    expected = {(method, label) for method in EXPECTED_METHODS for label in AMP_LABEL_IDS}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise A1ReportingError(
            f"Canonical A1 per-label table differs from M1--M4 x 17: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    wide: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for label in AMP_LABEL_IDS:
        method_rows = {method: by_key[(method, label)] for method in EXPECTED_METHODS}
        supports = {
            _integer(row.get("support"), field=f"{method}.{label}.support")
            for method, row in method_rows.items()
        }
        if len(supports) != 1:
            raise A1ReportingError(f"Silver-reference support differs by method for {label}")
        support = supports.pop()
        families = {str(row.get("family")) for row in method_rows.values()}
        if families != {AMP_FAMILY_BY_LABEL[label]}:
            raise A1ReportingError(f"AMP family differs from ontology for {label}")
        wide_row: dict[str, Any] = {
            "label_id": label,
            "family": AMP_FAMILY_BY_LABEL[label],
            "support": support,
        }
        for method in EXPECTED_METHODS:
            for metric in PER_LABEL_METRICS:
                wide_row[f"{method.lower()}_{metric}"] = _number(
                    method_rows[method].get(metric), field=f"{method}.{label}.{metric}"
                )
        wide.append(wide_row)
        deltas.append(
            {
                "label_id": label,
                "family": AMP_FAMILY_BY_LABEL[label],
                "support": support,
                "m3_precision": wide_row["m3_precision"],
                "m4_precision": wide_row["m4_precision"],
                "delta_precision_m4_minus_m3": (
                    wide_row["m4_precision"] - wide_row["m3_precision"]
                ),
                "m3_recall": wide_row["m3_recall"],
                "m4_recall": wide_row["m4_recall"],
                "delta_recall_m4_minus_m3": wide_row["m4_recall"] - wide_row["m3_recall"],
                "m3_f1": wide_row["m3_f1"],
                "m4_f1": wide_row["m4_f1"],
                "delta_f1_m4_minus_m3": wide_row["m4_f1"] - wide_row["m3_f1"],
                "comparison": "M4_SIX_SHOT_MINUS_M3_ZERO_SHOT",
                "significance_claim": "NOT_TESTED_DO_NOT_INFER",
            }
        )
    return wide, deltas


def _prediction_rows_by_rank(path: Path, method: str) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(path)
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("method_id") or row.get("method") or "").upper() != method:
            raise A1ReportingError(f"{path} contains a row not identified as {method}")
        if str(row.get("evaluation") or "").upper() != "A1":
            raise A1ReportingError(f"{path} contains a non-A1 row")
        if str(row.get("split") or "").upper() != "TEST":
            raise A1ReportingError(f"{path} contains a non-TEST row")
        rank = _integer(row.get("search_rank"), field=f"{method}.search_rank")
        if rank in output:
            raise A1ReportingError(f"{path} duplicates search_rank {rank}")
        output[rank] = row
    if len(output) != EXPECTED_A1_TEST_N:
        raise A1ReportingError(f"{method} A1 prediction count is not 253")
    return output


def _case_comparison_rows(
    rows: Sequence[Mapping[str, str]], m2_predictions: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_rank: dict[int, dict[str, Mapping[str, str]]] = {}
    for row in rows:
        method = str(row.get("method") or "")
        if method not in EXPECTED_METHODS:
            raise A1ReportingError(f"Unexpected method in case-error table: {method!r}")
        rank = _integer(row.get("search_rank"), field="case_errors.search_rank")
        method_rows = by_rank.setdefault(rank, {})
        if method in method_rows:
            raise A1ReportingError(f"Duplicate case-error row for {method}, rank {rank}")
        method_rows[method] = row
    if len(by_rank) != EXPECTED_A1_TEST_N:
        raise A1ReportingError("Canonical A1 case-error table does not contain 253 cases")
    if set(by_rank) != set(m2_predictions):
        raise A1ReportingError("M2 prediction membership differs from canonical case errors")

    output: list[dict[str, Any]] = []
    identity_fields = (
        "canonical_url",
        "jurisdiction",
        "fact_summary",
        "silver_reference_amp_json",
    )
    for rank in sorted(by_rank):
        method_rows = by_rank[rank]
        if tuple(method for method in EXPECTED_METHODS if method in method_rows) != EXPECTED_METHODS:
            raise A1ReportingError(f"Rank {rank} lacks one or more M1--M4 case-error rows")
        anchor = method_rows["M1"]
        for field in identity_fields:
            if any(method_rows[method].get(field) != anchor.get(field) for method in EXPECTED_METHODS):
                raise A1ReportingError(f"Rank {rank} has inconsistent {field} across methods")
        # Historical M1 rows predate the canonical UNODC case identity field and
        # use search_rank as case_id.  M2--M4 carry the actual case identity.
        canonical_case_ids = {
            str(method_rows[method].get("case_id") or "")
            for method in ("M2", "M3", "M4")
        }
        if len(canonical_case_ids) != 1 or not next(iter(canonical_case_ids)):
            raise A1ReportingError(f"Rank {rank} has inconsistent M2--M4 case identity")
        narrative = str(anchor.get("fact_summary") or "")
        m2_prediction = m2_predictions[rank]
        expected_m2_identity = {
            "case_id": next(iter(canonical_case_ids)),
            "canonical_url": anchor.get("canonical_url", ""),
            "jurisdiction": anchor.get("jurisdiction", ""),
            "fact_summary": narrative,
        }
        mismatched_m2_identity = [
            field
            for field, expected in expected_m2_identity.items()
            if m2_prediction.get(field) != expected
        ]
        if mismatched_m2_identity:
            raise A1ReportingError(
                f"Rank {rank} M2 prediction identity/provenance differs from canonical "
                f"case errors: {mismatched_m2_identity}"
            )
        m2_truncated = _integer(
            method_rows["M2"].get("truncated_input"), field=f"rank {rank}.m2_truncated"
        )
        if bool(m2_prediction.get("truncated_input")) != bool(m2_truncated):
            raise A1ReportingError(f"Rank {rank} M2 truncation provenance differs")
        case_row: dict[str, Any] = {
            "search_rank": rank,
            "case_id": next(iter(canonical_case_ids)),
            "canonical_url": anchor.get("canonical_url", ""),
            "jurisdiction": anchor.get("jurisdiction", ""),
            "fact_summary": narrative,
            "narrative_word_count": len(re.findall(r"\S+", narrative)),
            "m2_original_token_count": _integer(
                m2_prediction.get("original_token_count"),
                field=f"rank {rank}.m2_original_token_count",
            ),
            "m2_max_tokens_used": _integer(
                m2_prediction.get("max_tokens_used"),
                field=f"rank {rank}.m2_max_tokens_used",
            ),
            "m2_truncated_input": m2_truncated,
            "silver_reference_amp_json": anchor.get("silver_reference_amp_json", "[]"),
        }
        for method in EXPECTED_METHODS:
            raw = method_rows[method]
            prefix = method.lower()
            case_row[f"{prefix}_prediction_amp_json"] = raw.get("predicted_amp_json", "[]")
            case_row[f"{prefix}_false_positive_labels_json"] = raw.get(
                "false_positive_labels_json", "[]"
            )
            case_row[f"{prefix}_false_negative_labels_json"] = raw.get(
                "false_negative_labels_json", "[]"
            )
            case_row[f"{prefix}_exact_set_correct"] = _integer(
                raw.get("exact_set_correct"), field=f"rank {rank}.{prefix}.exact_set_correct"
            )
            case_row[f"{prefix}_example_jaccard"] = _number(
                raw.get("example_jaccard"), field=f"rank {rank}.{prefix}.example_jaccard"
            )
            for field in CPMR_CASE_FIELDS:
                value = raw.get(field)
                if field.endswith("_cpmr"):
                    parsed: int | float | str = _integer(
                        value, field=f"rank {rank}.{prefix}.{field}"
                    )
                    if parsed not in (0, 1):
                        raise A1ReportingError(
                            f"rank {rank}.{prefix}.{field} is not binary: {value!r}"
                        )
                elif value == "N/A":
                    parsed = "N/A"
                else:
                    parsed = _number(value, field=f"rank {rank}.{prefix}.{field}")
                case_row[f"{prefix}_{field}"] = parsed
        output.append(case_row)
    return output


def _validate_llm_prediction_identity(
    method: str,
    predictions: Mapping[int, Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require usage rows to describe the same canonical A1 cases being reported."""

    canonical_by_rank = {
        _integer(row.get("search_rank"), field="case_comparison.search_rank"): row
        for row in case_rows
    }
    if set(predictions) != set(canonical_by_rank):
        missing = sorted(set(canonical_by_rank) - set(predictions))
        extra = sorted(set(predictions) - set(canonical_by_rank))
        raise A1ReportingError(
            f"{method} prediction membership differs from canonical A1 case errors: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    identity_fields = ("case_id", "canonical_url", "jurisdiction", "fact_summary")
    for rank, prediction in predictions.items():
        canonical = canonical_by_rank[rank]
        mismatched = [
            field
            for field in identity_fields
            if prediction.get(field) != canonical.get(field)
        ]
        if mismatched:
            raise A1ReportingError(
                f"{method} rank {rank} identity/provenance differs from canonical "
                f"case errors: {mismatched}"
            )


def _retry_events(row: Mapping[str, Any], *, field: str) -> list[dict[str, Any]]:
    values = _parse_json_collection(row.get("retry_events"), field=field)
    output: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise A1ReportingError(f"{field} contains a non-object event")
        output.append(value)
    return output


def _is_rate_limit_event(event: Mapping[str, Any]) -> bool:
    try:
        if int(event.get("http_status") or 0) == 429:
            return True
    except (TypeError, ValueError):
        pass
    text = f"{event.get('error_type', '')} {event.get('message', '')}".casefold()
    return "ratelimit" in text or "rate limit" in text


def _classify_failure(row: Mapping[str, Any]) -> str:
    error = row.get("error")
    error = error if isinstance(error, Mapping) else {}
    status = error.get("http_status")
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    text = f"{error.get('error_type', '')} {error.get('message', '')}".casefold()
    if "refusal" in text:
        return "REFUSAL"
    if status_code == 429 or "ratelimit" in text or "rate limit" in text:
        return "RATE_LIMIT_FAILURE"
    if status_code == 408 or "timeout" in text or "timed out" in text:
        return "TIMEOUT"
    if any(token in text for token in ("schema", "structured output", "invalid label")):
        return "SCHEMA_ERROR"
    if status_code is not None or any(
        token in text for token in ("apierror", "connectionerror", "servererror")
    ):
        return "API_ERROR"
    return "OTHER_FAILURE"


def _api_usage_row(
    method: str,
    predictions: Mapping[int, Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    if diagnostics.get("status") != "COMPLETE":
        raise A1ReportingError(f"{method} A1 diagnostics are not COMPLETE")
    if diagnostics.get("successful_cases") != EXPECTED_A1_TEST_N:
        raise A1ReportingError(f"{method} A1 diagnostics do not record 253 successes")
    if diagnostics.get("unresolved_failure_cases") != len(failures):
        raise A1ReportingError(f"{method} failure manifest differs from diagnostics")
    if failures:
        raise A1ReportingError(f"{method} has unresolved API failures; reporting is not final")

    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []
    latencies: list[float] = []
    all_events: list[dict[str, Any]] = []
    retry_count = 0
    returned_models: set[str] = set()
    requested_models: set[str] = set()
    sdk_versions: set[str] = set()
    for rank, row in sorted(predictions.items()):
        if row.get("status") != "SUCCESS_VALIDATED":
            raise A1ReportingError(f"{method} rank {rank} is not SUCCESS_VALIDATED")
        usage = row.get("token_usage")
        if not isinstance(usage, Mapping):
            raise A1ReportingError(f"{method} rank {rank} lacks actual API token usage")
        input_value = _integer(usage.get("input_tokens"), field=f"{method}.{rank}.input_tokens")
        output_value = _integer(
            usage.get("output_tokens"), field=f"{method}.{rank}.output_tokens"
        )
        total_value = _integer(usage.get("total_tokens"), field=f"{method}.{rank}.total_tokens")
        if total_value != input_value + output_value:
            raise A1ReportingError(f"{method} rank {rank} token total is inconsistent")
        input_tokens.append(input_value)
        output_tokens.append(output_value)
        total_tokens.append(total_value)
        latencies.append(_number(row.get("latency_seconds"), field=f"{method}.{rank}.latency"))
        retry_count += _integer(row.get("retry_count"), field=f"{method}.{rank}.retry_count")
        all_events.extend(_retry_events(row, field=f"{method}.{rank}.retry_events"))
        if row.get("returned_model_id"):
            returned_models.add(str(row["returned_model_id"]))
        if row.get("effective_requested_model_id") or row.get("requested_model_id"):
            requested_models.add(
                str(row.get("effective_requested_model_id") or row.get("requested_model_id"))
            )
        if row.get("sdk_version"):
            sdk_versions.add(str(row["sdk_version"]))

    failure_counts = Counter(_classify_failure(row) for row in failures)
    failure_summary = {name: failure_counts.get(name, 0) for name in FAILURE_CLASSES}
    return {
        "method": method,
        "evaluation": "A1",
        "request_count": len(predictions) + len(failures),
        "successful_request_count": len(predictions),
        "api_attempt_count_including_retries": len(predictions) + len(failures) + retry_count,
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "total_tokens": sum(total_tokens),
        "median_input_tokens_per_request": float(median(input_tokens)),
        "median_output_tokens_per_request": float(median(output_tokens)),
        "median_latency_seconds": float(median(latencies)),
        "p90_latency_seconds": _linear_percentile(latencies, 0.90),
        "retry_count": retry_count,
        "retry_event_count": len(all_events),
        "rate_limit_event_count": sum(_is_rate_limit_event(event) for event in all_events),
        "api_failure_count": len(failures),
        "failure_history_case_count": len(diagnostics.get("failure_history_search_ranks", [])),
        "failure_class_counts_json": canonical_json(failure_summary),
        "requested_model_ids_json": canonical_json(sorted(requested_models)),
        "returned_model_ids_json": canonical_json(sorted(returned_models)),
        "sdk_versions_json": canonical_json(sorted(sdk_versions)),
        "token_source": "OPENAI_RESPONSES_API_RECORDED_USAGE",
        "latency_scope": "SUCCESSFUL_CASE_REQUEST_END_TO_END_INCLUDING_RETRIES",
        "status": "COMPLETE",
    }


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_reporting_payloads(
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> tuple[dict[str, bytes], list[Path]]:
    """Validate finalized inputs and return deterministic output payloads."""

    manifest_path = metrics_root / "amp_evaluation_manifest.json"
    primary_path = metrics_root / "a1/amp_primary_results.csv"
    per_label_path = metrics_root / "a1/amp_per_label.csv"
    bootstrap_path = metrics_root / "a1/amp_bootstrap_cis.csv"
    case_error_path = metrics_root / "a1/amp_case_level_errors.csv"
    sensitivity_path = metrics_root / "amp_threshold_0_50_sensitivity.csv"
    _validate_evaluator_manifest(manifest_path)
    _require_file(bootstrap_path)
    _require_file(sensitivity_path)

    aggregate_rows = _aggregate_delta_rows(_load_csv(primary_path))
    wide_label_rows, label_delta_rows = _per_label_outputs(_load_csv(per_label_path))
    m2_path = prediction_root / "m2/a1_test_predictions.jsonl"
    m2_predictions = _prediction_rows_by_rank(m2_path, "M2")
    case_rows = _case_comparison_rows(_load_csv(case_error_path), m2_predictions)

    source_paths = [
        manifest_path,
        primary_path,
        per_label_path,
        bootstrap_path,
        case_error_path,
        sensitivity_path,
        m2_path,
    ]
    usage_rows: list[dict[str, Any]] = []
    for method in LLM_METHODS:
        lower = method.lower()
        prediction_path = prediction_root / lower / "a1_test_predictions.jsonl"
        failure_path = log_root / f"{lower}_a1_failures.jsonl"
        diagnostics_path = log_root / f"{lower}_a1_diagnostics.json"
        predictions = _prediction_rows_by_rank(prediction_path, method)
        _validate_llm_prediction_identity(method, predictions, case_rows)
        failures = _load_jsonl(failure_path)
        diagnostics = _load_json(diagnostics_path)
        usage_rows.append(_api_usage_row(method, predictions, failures, diagnostics))
        source_paths.extend((prediction_path, failure_path, diagnostics_path))

    aggregate_fields = (
        "metric",
        "m3_zero_shot",
        "m4_six_shot",
        "delta_m4_minus_m3",
        "test_n",
        "comparison",
        "significance_claim",
    )
    label_wide_fields = (
        "label_id",
        "family",
        "support",
        *(f"{method.lower()}_{metric}" for method in EXPECTED_METHODS for metric in PER_LABEL_METRICS),
    )
    label_delta_fields = tuple(label_delta_rows[0])
    case_fields = tuple(case_rows[0])
    usage_fields = tuple(usage_rows[0])
    payloads = {
        "amp_m3_vs_m4_aggregate_deltas.csv": _csv_bytes(aggregate_rows, aggregate_fields),
        "amp_per_label_comparison.csv": _csv_bytes(wide_label_rows, label_wide_fields),
        "amp_m3_vs_m4_per_label_deltas.csv": _csv_bytes(
            label_delta_rows, label_delta_fields
        ),
        "amp_case_level_comparison.csv": _csv_bytes(case_rows, case_fields),
        "amp_llm_api_usage.csv": _csv_bytes(usage_rows, usage_fields),
    }
    provenance = {
        "schema_version": "sherloc-a1-post-evaluation-reporting-v1",
        "generator_version": VERSION,
        "deterministic": True,
        "a1_test_n": EXPECTED_A1_TEST_N,
        "methods": list(EXPECTED_METHODS),
        "comparison_definition": "M4_SIX_SHOT_MINUS_M3_ZERO_SHOT",
        "significance_claim": "NOT_TESTED_DO_NOT_INFER",
        "narrative_word_count_definition": "COUNT_OF_NON_WHITESPACE_TOKEN_RUNS",
        "inputs": [
            {"path": _relative_or_absolute(path), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "outputs": [
            {"path": name, "sha256": sha256_bytes(payload), "byte_count": len(payload)}
            for name, payload in payloads.items()
        ],
    }
    payloads["amp_a1_post_evaluation_manifest.json"] = _json_bytes(provenance)
    return payloads, source_paths


def generate_reporting_artifacts(
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
    log_root: Path = DEFAULT_LOG_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    check: bool = False,
) -> list[dict[str, Any]]:
    payloads, _ = build_reporting_payloads(
        metrics_root=metrics_root,
        prediction_root=prediction_root,
        log_root=log_root,
    )
    diagnostics: list[dict[str, Any]] = []
    for name, payload in payloads.items():
        path = output_dir / name
        matches = path.is_file() and path.read_bytes() == payload
        if check and not matches:
            raise A1ReportingError(f"Generated A1 reporting artifact is missing or stale: {path}")
        if not check and not matches:
            _atomic_write(path, payload)
        diagnostics.append(
            {
                "path": str(path),
                "status": "UNCHANGED" if matches else "WOULD_WRITE" if check else "WRITTEN",
                "sha256": sha256_bytes(payload),
                "byte_count": len(payload),
            }
        )
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        diagnostics = generate_reporting_artifacts(
            metrics_root=args.metrics_root,
            prediction_root=args.prediction_root,
            log_root=args.log_root,
            output_dir=args.output_dir,
            check=args.check,
        )
    except A1ReportingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"generator_version": VERSION, "artifacts": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
