#!/usr/bin/env python3
"""Summarize recorded M3/M4 A2 API usage without making API calls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs/logs/llm"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/metrics/a2/amp_llm_api_usage.csv"
EXPECTED_FOLD_N = {1: 288, 2: 287, 3: 286}

FIELDS = (
    "method",
    "evaluation",
    "scope",
    "fold",
    "successful_request_count",
    "retry_count",
    "rate_limit_retries",
    "max_output_fallback_attempts",
    "max_output_fallback_cases",
    "unresolved_failures",
    "recorded_input_tokens",
    "recorded_output_tokens",
    "recorded_total_tokens",
    "median_latency_seconds",
    "p90_latency_seconds",
    "runtime_seconds",
    "wall_clock_span_seconds",
    "token_accounting_note",
)


class UsageReportError(RuntimeError):
    """Raised when completed canonical A2 state cannot be summarized safely."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise UsageReportError(f"Expected a JSON object: {path}")
    return value


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _fold_summary(method: str, fold: int, log_root: Path) -> tuple[dict[str, Any], datetime, datetime, list[float]]:
    method_lower = method.lower()
    setting = f"{method_lower}_a2_fold_{fold}"
    diagnostics_path = log_root / f"{setting}_diagnostics.json"
    state_dir = log_root / "state" / method_lower / f"a2_fold_{fold}" / "success"
    if not diagnostics_path.is_file() or not state_dir.is_dir():
        raise UsageReportError(f"Missing completed state for {setting}")
    diagnostics = _read_json(diagnostics_path)
    expected_n = EXPECTED_FOLD_N[fold]
    if (
        diagnostics.get("status") != "COMPLETE"
        or diagnostics.get("expected_cases") != expected_n
        or diagnostics.get("successful_cases") != expected_n
        or diagnostics.get("unresolved_failure_cases") != 0
        or diagnostics.get("missing_unattempted_cases") != 0
    ):
        raise UsageReportError(f"{setting} is not complete: {diagnostics_path}")

    success_paths = sorted(state_dir.glob("*.json"))
    if len(success_paths) != expected_n:
        raise UsageReportError(
            f"{setting} has {len(success_paths)} success records; expected {expected_n}"
        )
    successes = [_read_json(path) for path in success_paths]
    ranks = [int(row.get("search_rank")) for row in successes]
    if len(ranks) != len(set(ranks)):
        raise UsageReportError(f"{setting} contains duplicate success ranks")

    latencies = [float(row["latency_seconds"]) for row in successes]
    retry_events = [event for row in successes for event in row.get("retry_events", [])]
    historical_failures: list[dict[str, Any]] = []
    failure_dir = state_dir.parent / "failures"
    for failure_path in sorted(failure_dir.glob("*.json")) if failure_dir.is_dir() else []:
        document = _read_json(failure_path)
        history = document.get("attempt_history", [])
        if not isinstance(history, list):
            raise UsageReportError(f"Malformed failure history: {failure_path}")
        historical_failures.extend(item for item in history if isinstance(item, dict))
    historical_events = [
        event
        for failure in historical_failures
        for event in failure.get("retry_events", [])
    ]
    retry_events.extend(historical_events)
    rate_limits = sum(
        event.get("http_status") == 429 or event.get("error_type") == "RateLimitError"
        for event in retry_events
    )
    fallback_attempts = sum(
        int(row.get("technical_execution", {}).get("output_token_fallback_attempts_this_invocation", 0))
        for row in successes
    )
    fallback_attempts += sum(
        int(failure.get("technical_execution", {}).get("output_token_fallback_attempts_this_invocation", 0))
        for failure in historical_failures
    )
    fallback_cases = sum(
        bool(row.get("technical_execution", {}).get("output_token_fallback_used", False))
        for row in successes
    )
    input_tokens = sum(int(row.get("token_usage", {}).get("input_tokens", 0)) for row in successes)
    output_tokens = sum(int(row.get("token_usage", {}).get("output_tokens", 0)) for row in successes)
    total_tokens = sum(int(row.get("token_usage", {}).get("total_tokens", 0)) for row in successes)
    lock_periods: list[tuple[datetime, datetime]] = []
    history_dir = state_dir.parent / ".lock_history"
    for history_path in sorted(history_dir.glob("*.json")) if history_dir.is_dir() else []:
        lock = _read_json(history_path)
        if lock.get("acquired_at") and lock.get("released_at"):
            lock_periods.append(
                (_parse_time(str(lock["acquired_at"])), _parse_time(str(lock["released_at"])))
            )
    if lock_periods:
        started = min(period[0] for period in lock_periods)
        updated = max(period[1] for period in lock_periods)
        runtime = sum((end - start).total_seconds() for start, end in lock_periods)
    else:
        started = _parse_time(str(diagnostics["started_at"]))
        updated = _parse_time(str(diagnostics["updated_at"]))
        runtime = (updated - started).total_seconds()
    row = {
        "method": method,
        "evaluation": "A2",
        "scope": "FOLD",
        "fold": fold,
        "successful_request_count": expected_n,
        "retry_count": (
            sum(int(item.get("retry_count", 0)) for item in successes)
            + sum(
                int(failure.get("technical_execution", {}).get("request_attempt_count", 0))
                for failure in historical_failures
            )
        ),
        "rate_limit_retries": int(rate_limits),
        "max_output_fallback_attempts": fallback_attempts,
        "max_output_fallback_cases": fallback_cases,
        "unresolved_failures": 0,
        "recorded_input_tokens": input_tokens,
        "recorded_output_tokens": output_tokens,
        "recorded_total_tokens": total_tokens,
        "median_latency_seconds": median(latencies),
        "p90_latency_seconds": float(np.percentile(latencies, 90)),
        "runtime_seconds": runtime,
        "wall_clock_span_seconds": (updated - started).total_seconds(),
        "token_accounting_note": "SUCCESS_RESPONSE_USAGE_ONLY; FAILED_REQUEST_USAGE_MAY_BE_UNAVAILABLE_AND_TOTALS_ARE_LOWER_BOUNDS",
    }
    return row, started, updated, latencies


def generate_usage_report(log_root: Path = DEFAULT_LOG_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("M3", "M4"):
        method_rows: list[dict[str, Any]] = []
        starts: list[datetime] = []
        ends: list[datetime] = []
        latencies: list[float] = []
        for fold in (1, 2, 3):
            row, started, updated, fold_latencies = _fold_summary(method, fold, log_root)
            method_rows.append(row)
            starts.append(started)
            ends.append(updated)
            latencies.extend(fold_latencies)
        rows.extend(method_rows)
        rows.append(
            {
                "method": method,
                "evaluation": "A2",
                "scope": "POOLED_OOD_TEST",
                "fold": "",
                "successful_request_count": sum(row["successful_request_count"] for row in method_rows),
                "retry_count": sum(row["retry_count"] for row in method_rows),
                "rate_limit_retries": sum(row["rate_limit_retries"] for row in method_rows),
                "max_output_fallback_attempts": sum(row["max_output_fallback_attempts"] for row in method_rows),
                "max_output_fallback_cases": sum(row["max_output_fallback_cases"] for row in method_rows),
                "unresolved_failures": sum(row["unresolved_failures"] for row in method_rows),
                "recorded_input_tokens": sum(row["recorded_input_tokens"] for row in method_rows),
                "recorded_output_tokens": sum(row["recorded_output_tokens"] for row in method_rows),
                "recorded_total_tokens": sum(row["recorded_total_tokens"] for row in method_rows),
                "median_latency_seconds": median(latencies),
                "p90_latency_seconds": float(np.percentile(latencies, 90)),
                "runtime_seconds": sum(row["runtime_seconds"] for row in method_rows),
                "wall_clock_span_seconds": (max(ends) - min(starts)).total_seconds(),
                "token_accounting_note": "SUCCESS_RESPONSE_USAGE_ONLY; FAILED_REQUEST_USAGE_MAY_BE_UNAVAILABLE_AND_TOTALS_ARE_LOWER_BOUNDS",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = generate_usage_report(args.log_root)
    _atomic_csv(args.output, rows)
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
