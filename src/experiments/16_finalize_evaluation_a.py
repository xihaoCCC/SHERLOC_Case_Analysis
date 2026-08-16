#!/usr/bin/env python3
"""Finalize frozen Evaluation A results into paper-facing analysis artifacts.

This script is deliberately post hoc and read-only with respect to model outputs.
It consumes only the frozen prediction-derived canonical metrics under
``outputs/metrics``.  It never trains a model, calls an API, changes a threshold,
or writes into the canonical metric or prediction directories.

The transformations here are deterministic reshaping and explicitly descriptive
summaries: family means from canonical per-label metrics, prediction breadth from
canonical case-level outputs, a prespecified rare-label sensitivity calculation,
and publication figures.  Primary metric values and bootstrap intervals are
copied from the canonical evaluator outputs without recomputation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:  # Support package imports and direct execution.
    from .metrics import (
        AMP_FAMILIES,
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        ORGAN_REMOVAL_LABEL,
        SILVER_REFERENCE_TERM,
    )
except ImportError:  # pragma: no cover - direct CLI invocation.
    from metrics import (  # type: ignore
        AMP_FAMILIES,
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        ORGAN_REMOVAL_LABEL,
        SILVER_REFERENCE_TERM,
    )


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = REPO_ROOT / "outputs/metrics"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs/analysis/evaluation_a"
DEFAULT_FIGURE_DIR = REPO_ROOT / "outputs/figures/evaluation_a"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"

EXPECTED_METHODS: tuple[str, ...] = ("M1", "M2", "M3", "M4")
EXPECTED_TEST_N = {"A1": 253, "A2": 861}
EXPECTED_MACRO_LABEL_N = {"A1": 17, "A2": 16}
EXPECTED_A2_FOLD_N = {1: 288, 2: 287, 3: 286}

DISPLAY_LABELS: dict[str, str] = {
    "ACT_RECRUITMENT": "Recruitment",
    "ACT_TRANSPORTATION": "Transportation",
    "ACT_TRANSFER": "Transfer",
    "ACT_HARBOURING": "Harbouring",
    "ACT_RECEIPT": "Receipt",
    "MEANS_THREAT_FORCE_OR_COERCION": "Threat or force or coercion",
    "MEANS_ABDUCTION": "Abduction",
    "MEANS_FRAUD": "Fraud",
    "MEANS_DECEPTION": "Deception",
    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": "Abuse of power or vulnerability",
    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": "Payments or benefits for control",
    "PURPOSE_SEXUAL_EXPLOITATION": "Sexual exploitation",
    "PURPOSE_FORCED_LABOUR_OR_SERVICES": "Forced labour or services",
    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": "Slavery or similar practices",
    "PURPOSE_SERVITUDE": "Servitude",
    "PURPOSE_REMOVAL_OF_ORGANS": "Removal of organs",
    "PURPOSE_OTHER": "Other",
}

SHORT_LABELS: dict[str, str] = {
    "ACT_RECRUITMENT": "Recruit.",
    "ACT_TRANSPORTATION": "Transport.",
    "ACT_TRANSFER": "Transfer",
    "ACT_HARBOURING": "Harbour.",
    "ACT_RECEIPT": "Receipt",
    "MEANS_THREAT_FORCE_OR_COERCION": "Threat/coercion",
    "MEANS_ABDUCTION": "Abduction",
    "MEANS_FRAUD": "Fraud",
    "MEANS_DECEPTION": "Deception",
    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": "Power/vulnerability",
    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": "Payments/benefits",
    "PURPOSE_SEXUAL_EXPLOITATION": "Sexual exploit.",
    "PURPOSE_FORCED_LABOUR_OR_SERVICES": "Forced labour",
    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": "Slavery/similar",
    "PURPOSE_SERVITUDE": "Servitude",
    "PURPOSE_REMOVAL_OF_ORGANS": "Organ removal",
    "PURPOSE_OTHER": "Other",
}

ANALYSIS_FILENAMES: tuple[str, ...] = (
    "a1_main_comparison.csv",
    "a2_main_comparison.csv",
    "amp_family_level_metrics.csv",
    "prediction_breadth_summary.csv",
    "rare_label_sensitivity.csv",
    "a1_to_a2_distribution_shift.csv",
    "m3_vs_m4_summary.csv",
    "m3_vs_m4_per_label_f1.csv",
    "a2_fold_summary.csv",
    "a2_jurisdiction_summary.csv",
    "amp_label_display_mapping.csv",
)

FIGURE_FILENAMES: tuple[str, ...] = (
    "figure_1_a1_vs_a2_core_performance.svg",
    "figure_2_cpmr_by_amp_family.svg",
    "figure_3_cpmr_vs_contained_recall.svg",
    "figure_4_per_label_f1.svg",
)


class EvaluationAFinalizationError(RuntimeError):
    """Raised when canonical inputs cannot support a faithful final package."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise EvaluationAFinalizationError(f"Required canonical input is missing: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    _require_file(path)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EvaluationAFinalizationError(f"CSV has no header: {path}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvaluationAFinalizationError(f"Could not read canonical CSV: {path}") from exc


def _require_fields(rows: Sequence[Mapping[str, Any]], fields: Iterable[str], *, source: str) -> None:
    if not rows:
        raise EvaluationAFinalizationError(f"Canonical input is empty: {source}")
    missing = set(fields) - set(rows[0])
    if missing:
        raise EvaluationAFinalizationError(
            f"Canonical input {source} is missing fields: {sorted(missing)}"
        )


def _number(value: Any, *, field: str, source: str) -> float:
    if value is None or str(value).strip() in {"", "N/A", "NA", "None"}:
        raise EvaluationAFinalizationError(f"{source}.{field} is unexpectedly undefined")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationAFinalizationError(f"{source}.{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise EvaluationAFinalizationError(f"{source}.{field} is not finite")
    return result


def _optional_number(value: Any, *, field: str, source: str) -> float | None:
    if value is None or str(value).strip() in {"", "N/A", "NA", "None"}:
        return None
    return _number(value, field=field, source=source)


def _integer(value: Any, *, field: str, source: str) -> int:
    result = _number(value, field=field, source=source)
    if not result.is_integer():
        raise EvaluationAFinalizationError(f"{source}.{field} is not an integer: {value!r}")
    return int(result)


def _ordered_method_rows(rows: Sequence[Mapping[str, Any]], *, source: str) -> list[Mapping[str, Any]]:
    by_method: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method in by_method:
            raise EvaluationAFinalizationError(f"Duplicate {method} row in {source}")
        by_method[method] = row
    if set(by_method) != set(EXPECTED_METHODS):
        raise EvaluationAFinalizationError(
            f"{source} methods are {sorted(by_method)}; expected {list(EXPECTED_METHODS)}"
        )
    return [by_method[method] for method in EXPECTED_METHODS]


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> str:
    if not rows:
        raise EvaluationAFinalizationError("Refusing to write an empty analysis table")
    fields = tuple(fieldnames or rows[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, _csv_text(rows))


def _main_table(evaluation: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if evaluation not in EXPECTED_TEST_N:
        raise EvaluationAFinalizationError(f"Unknown evaluation: {evaluation}")
    def canonical_key(metric: str) -> str:
        if evaluation == "A1":
            return metric
        # The canonical evaluator deliberately uses ``pooled_ood`` for macro-F1
        # and ``pooled`` for the all-label and CPMR metrics.
        return f"pooled_ood_{metric}" if metric.startswith("macro_f1") else f"pooled_{metric}"

    required = (
        "method",
        "test_n",
        "macro_label_count",
        *(canonical_key(metric) for metric in (
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
            "act_cpmr",
            "act_mean_contained_recall",
            "means_cpmr",
            "means_mean_contained_recall",
            "purpose_cpmr",
            "purpose_mean_contained_recall",
        )),
    )
    _require_fields(rows, required, source=f"{evaluation} primary results")
    output: list[dict[str, Any]] = []
    for raw in _ordered_method_rows(rows, source=f"{evaluation} primary results"):
        source = f"{evaluation}/{raw['method']}"
        n = _integer(raw["test_n"], field="test_n", source=source)
        macro_n = _integer(raw["macro_label_count"], field="macro_label_count", source=source)
        if n != EXPECTED_TEST_N[evaluation]:
            raise EvaluationAFinalizationError(
                f"{source} N={n}; expected frozen N={EXPECTED_TEST_N[evaluation]}"
            )
        if macro_n != EXPECTED_MACRO_LABEL_N[evaluation]:
            raise EvaluationAFinalizationError(
                f"{source} macro label count={macro_n}; expected {EXPECTED_MACRO_LABEL_N[evaluation]}"
            )
        def get(metric: str) -> float:
            key = canonical_key(metric)
            return _number(raw[key], field=key, source=source)

        row = {
            "method": raw["method"],
            "n": n,
            "macro_f1": get("macro_f1"),
            "macro_f1_ci_lower": get("macro_f1_ci_lower"),
            "macro_f1_ci_upper": get("macro_f1_ci_upper"),
            "micro_f1": get("micro_f1"),
            "micro_f1_ci_lower": get("micro_f1_ci_lower"),
            "micro_f1_ci_upper": get("micro_f1_ci_upper"),
            "exact_set_accuracy": get("exact_set_accuracy"),
            "exact_set_accuracy_ci_lower": get("exact_set_accuracy_ci_lower"),
            "exact_set_accuracy_ci_upper": get("exact_set_accuracy_ci_upper"),
            "example_jaccard": get("example_jaccard"),
            "example_jaccard_ci_lower": get("example_jaccard_ci_lower"),
            "example_jaccard_ci_upper": get("example_jaccard_ci_upper"),
            "act_cpmr": get("act_cpmr"),
            "act_mean_contained_recall": get("act_mean_contained_recall"),
            "means_cpmr": get("means_cpmr"),
            "means_mean_contained_recall": get("means_mean_contained_recall"),
            "purpose_cpmr": get("purpose_cpmr"),
            "purpose_mean_contained_recall": get("purpose_mean_contained_recall"),
            "macro_supported_label_count": macro_n,
            "macro_label_rule": (
                "ALL_17_FROZEN_AMP_LABELS"
                if evaluation == "A1"
                else "16_POSITIVE_SUPPORT_LABELS; ORGAN_REMOVAL_RETAINED_AS_PREDICTION_DIMENSION"
            ),
            "reference_terminology": str(raw.get("reference_terminology") or SILVER_REFERENCE_TERM),
        }
        output.append(row)
    return output


def _index_per_label(
    rows: Sequence[Mapping[str, Any]], *, evaluation: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    _require_fields(
        rows,
        (
            "method",
            "label_id",
            "family",
            "support",
            "precision",
            "recall",
            "f1",
            "status",
            "included_in_macro_f1",
        ),
        source=f"{evaluation} per-label results",
    )
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["label_id"]))
        if key in index:
            raise EvaluationAFinalizationError(f"Duplicate per-label row: {evaluation}/{key}")
        index[key] = row
    expected = {(method, label) for method in EXPECTED_METHODS for label in AMP_LABEL_IDS}
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise EvaluationAFinalizationError(
            f"{evaluation} per-label grid mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return index


def _family_table(
    per_label_by_eval: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for evaluation in ("A1", "A2"):
        index = _index_per_label(per_label_by_eval[evaluation], evaluation=evaluation)
        for method in EXPECTED_METHODS:
            for family in AMP_FAMILIES:
                candidates = [
                    index[(method, label)]
                    for label in AMP_LABEL_IDS
                    if AMP_FAMILY_BY_LABEL[label] == family
                ]
                supported = [
                    row
                    for row in candidates
                    if _integer(row["support"], field="support", source=f"{evaluation}/{method}") > 0
                ]
                values: dict[str, list[float]] = {name: [] for name in ("precision", "recall", "f1")}
                for row in supported:
                    for name in values:
                        value = _optional_number(
                            row[name], field=name, source=f"{evaluation}/{method}/{row['label_id']}"
                        )
                        if value is None:
                            raise EvaluationAFinalizationError(
                                f"Supported label has undefined {name}: {evaluation}/{method}/{row['label_id']}"
                            )
                        values[name].append(value)
                if not supported:
                    raise EvaluationAFinalizationError(
                        f"No supported {family} labels for {evaluation}/{method}"
                    )
                output.append(
                    {
                        "evaluation": evaluation,
                        "method": method,
                        "family": family.title(),
                        "supported_label_count": len(supported),
                        "macro_precision_family": sum(values["precision"]) / len(supported),
                        "macro_recall_family": sum(values["recall"]) / len(supported),
                        "macro_f1_family": sum(values["f1"]) / len(supported),
                        "zero_support_labels_excluded": len(candidates) - len(supported),
                        "status": "DESCRIPTIVE_FROM_CANONICAL_PER_LABEL_METRICS",
                        "reference_terminology": SILVER_REFERENCE_TERM,
                    }
                )
    return output


def _parse_label_array(value: Any, *, source: str) -> tuple[str, ...]:
    try:
        labels = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise EvaluationAFinalizationError(f"Malformed label JSON in {source}") from exc
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise EvaluationAFinalizationError(f"Label JSON is not a string list in {source}")
    if len(labels) != len(set(labels)):
        raise EvaluationAFinalizationError(f"Duplicate label in {source}")
    unknown = set(labels) - set(AMP_LABEL_IDS)
    if unknown:
        raise EvaluationAFinalizationError(f"Unknown labels in {source}: {sorted(unknown)}")
    return tuple(labels)


def _family_counts(labels: Sequence[str]) -> dict[str, int]:
    counts = {family: 0 for family in AMP_FAMILIES}
    for label in labels:
        counts[AMP_FAMILY_BY_LABEL[label]] += 1
    return counts


def _prediction_breadth(
    case_rows_by_eval: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for evaluation in ("A1", "A2"):
        rows = case_rows_by_eval[evaluation]
        _require_fields(
            rows,
            ("method", "search_rank", "silver_reference_amp_json", "predicted_amp_json"),
            source=f"{evaluation} case-level results",
        )
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        references: dict[int, tuple[str, ...]] = {}
        for row in rows:
            method = str(row["method"])
            if method not in EXPECTED_METHODS:
                raise EvaluationAFinalizationError(f"Unexpected method in {evaluation} case rows: {method}")
            grouped[method].append(row)
            rank = _integer(row["search_rank"], field="search_rank", source=f"{evaluation}/{method}")
            reference = _parse_label_array(
                row["silver_reference_amp_json"], source=f"{evaluation}/{method}/rank-{rank}/reference"
            )
            previous = references.setdefault(rank, reference)
            if previous != reference:
                raise EvaluationAFinalizationError(
                    f"Silver-reference mismatch across methods for {evaluation} rank {rank}"
                )
        if set(grouped) != set(EXPECTED_METHODS):
            raise EvaluationAFinalizationError(f"Incomplete case-level methods for {evaluation}")
        if len(references) != EXPECTED_TEST_N[evaluation]:
            raise EvaluationAFinalizationError(
                f"{evaluation} unique case count={len(references)}; expected {EXPECTED_TEST_N[evaluation]}"
            )
        for method in EXPECTED_METHODS:
            method_rows = grouped[method]
            if len(method_rows) != EXPECTED_TEST_N[evaluation]:
                raise EvaluationAFinalizationError(
                    f"{evaluation}/{method} case rows={len(method_rows)}; expected {EXPECTED_TEST_N[evaluation]}"
                )
            predicted_family: dict[str, list[int]] = {family: [] for family in AMP_FAMILIES}
            reference_family: dict[str, list[int]] = {family: [] for family in AMP_FAMILIES}
            predicted_total: list[int] = []
            reference_total: list[int] = []
            seen: set[int] = set()
            for row in method_rows:
                rank = _integer(row["search_rank"], field="search_rank", source=f"{evaluation}/{method}")
                if rank in seen:
                    raise EvaluationAFinalizationError(f"Duplicate rank in {evaluation}/{method}: {rank}")
                seen.add(rank)
                predicted = _parse_label_array(
                    row["predicted_amp_json"], source=f"{evaluation}/{method}/rank-{rank}/prediction"
                )
                reference = references[rank]
                pred_counts = _family_counts(predicted)
                ref_counts = _family_counts(reference)
                for family in AMP_FAMILIES:
                    predicted_family[family].append(pred_counts[family])
                    reference_family[family].append(ref_counts[family])
                predicted_total.append(len(predicted))
                reference_total.append(len(reference))
            output.append(
                {
                    "evaluation": evaluation,
                    "method": method,
                    "n": len(method_rows),
                    "mean_predicted_act_labels": float(np.mean(predicted_family["ACT"])),
                    "mean_predicted_means_labels": float(np.mean(predicted_family["MEANS"])),
                    "mean_predicted_purpose_labels": float(np.mean(predicted_family["PURPOSE"])),
                    "mean_total_predicted_labels": float(np.mean(predicted_total)),
                    "mean_silver_reference_act_labels": float(np.mean(reference_family["ACT"])),
                    "mean_silver_reference_means_labels": float(np.mean(reference_family["MEANS"])),
                    "mean_silver_reference_purpose_labels": float(np.mean(reference_family["PURPOSE"])),
                    "mean_total_silver_reference_labels": float(np.mean(reference_total)),
                    "median_total_predicted_labels": float(median(predicted_total)),
                    "p25_total_predicted_labels": float(np.percentile(predicted_total, 25, method="linear")),
                    "p75_total_predicted_labels": float(np.percentile(predicted_total, 75, method="linear")),
                    "proportion_zero_predicted_act": float(np.mean(np.asarray(predicted_family["ACT"]) == 0)),
                    "proportion_zero_predicted_means": float(np.mean(np.asarray(predicted_family["MEANS"]) == 0)),
                    "proportion_zero_predicted_purpose": float(np.mean(np.asarray(predicted_family["PURPOSE"]) == 0)),
                    "status": "DESCRIPTIVE_NOT_PRIMARY_PERFORMANCE_METRIC",
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
            )
    return output


def _rare_label_sensitivity(
    per_label_by_eval: Mapping[str, Sequence[Mapping[str, Any]]],
    main_by_eval: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    fields = (
        "record_type",
        "evaluation",
        "method",
        "label_id",
        "family",
        "label_support",
        "included_in_official_macro_f1",
        "official_macro_label_count",
        "official_macro_f1",
        "diagnostic_rule",
        "diagnostic_macro_label_count",
        "diagnostic_macro_f1",
        "delta_diagnostic_minus_official",
        "status",
        "reference_terminology",
    )
    output: list[dict[str, Any]] = []
    for evaluation in ("A1", "A2"):
        index = _index_per_label(per_label_by_eval[evaluation], evaluation=evaluation)
        main_index = {str(row["method"]): row for row in main_by_eval[evaluation]}
        # Support is an evaluation property and must be identical across methods.
        for label in AMP_LABEL_IDS:
            supports = {
                _integer(index[(method, label)]["support"], field="support", source=f"{evaluation}/{method}/{label}")
                for method in EXPECTED_METHODS
            }
            inclusions = {
                _integer(
                    index[(method, label)]["included_in_macro_f1"],
                    field="included_in_macro_f1",
                    source=f"{evaluation}/{method}/{label}",
                )
                for method in EXPECTED_METHODS
            }
            if len(supports) != 1 or len(inclusions) != 1:
                raise EvaluationAFinalizationError(
                    f"Support/macro inclusion differs across methods: {evaluation}/{label}"
                )
            output.append(
                dict.fromkeys(fields, "")
                | {
                    "record_type": "LABEL_SUPPORT",
                    "evaluation": evaluation,
                    "label_id": label,
                    "family": AMP_FAMILY_BY_LABEL[label].title(),
                    "label_support": supports.pop(),
                    "included_in_official_macro_f1": inclusions.pop(),
                    "status": "DESCRIPTIVE_SUPPORT_PROFILE",
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
            )
        for method in EXPECTED_METHODS:
            eligible_without_organ: list[float] = []
            for label in AMP_LABEL_IDS:
                row = index[(method, label)]
                support = _integer(row["support"], field="support", source=f"{evaluation}/{method}/{label}")
                if label == ORGAN_REMOVAL_LABEL or support == 0:
                    continue
                value = _optional_number(row["f1"], field="f1", source=f"{evaluation}/{method}/{label}")
                if value is None:
                    raise EvaluationAFinalizationError(
                        f"Supported sensitivity label has undefined F1: {evaluation}/{method}/{label}"
                    )
                eligible_without_organ.append(value)
            official = float(main_index[method]["macro_f1"])
            diagnostic = sum(eligible_without_organ) / len(eligible_without_organ)
            if evaluation == "A2" and not math.isclose(official, diagnostic, rel_tol=0, abs_tol=1e-12):
                raise EvaluationAFinalizationError(
                    f"A2 supported-label Macro-F1 is inconsistent for {method}"
                )
            if evaluation == "A2":
                # The diagnostic rule is exactly the official supported-label
                # rule in A2; preserve an exact zero delta after validating the
                # independently aggregated canonical per-label values above.
                diagnostic = official
            output.append(
                dict.fromkeys(fields, "")
                | {
                    "record_type": "MACRO_SENSITIVITY",
                    "evaluation": evaluation,
                    "method": method,
                    "official_macro_label_count": EXPECTED_MACRO_LABEL_N[evaluation],
                    "official_macro_f1": official,
                    "diagnostic_rule": (
                        "EXCLUDE_PURPOSE_REMOVAL_OF_ORGANS_DESPITE_POSITIVE_SUPPORT"
                        if evaluation == "A1"
                        else "ORGAN_REMOVAL_ALREADY_ZERO_SUPPORT_AND_EXCLUDED_FROM_OFFICIAL_MACRO"
                    ),
                    "diagnostic_macro_label_count": len(eligible_without_organ),
                    "diagnostic_macro_f1": diagnostic,
                    "delta_diagnostic_minus_official": diagnostic - official,
                    "status": "DESCRIPTIVE_SENSITIVITY_NOT_CANONICAL_REPLACEMENT",
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
            )
    return output


def _shift_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _require_fields(
        rows,
        (
            "method",
            "delta_macro_f1_a2_minus_a1",
            "delta_micro_f1_a2_minus_a1",
            "delta_exact_set_a2_minus_a1",
            "delta_example_jaccard_a2_minus_a1",
            "delta_act_cpmr_a2_minus_a1",
            "delta_means_cpmr_a2_minus_a1",
            "delta_purpose_cpmr_a2_minus_a1",
        ),
        source="A1-to-A2 deltas",
    )
    output: list[dict[str, Any]] = []
    for row in _ordered_method_rows(rows, source="A1-to-A2 deltas"):
        output.append(
            {
                "method": row["method"],
                "delta_macro_f1_a2_minus_a1": _number(row["delta_macro_f1_a2_minus_a1"], field="delta_macro_f1_a2_minus_a1", source=str(row["method"])),
                "delta_micro_f1_a2_minus_a1": _number(row["delta_micro_f1_a2_minus_a1"], field="delta_micro_f1_a2_minus_a1", source=str(row["method"])),
                "delta_exact_set_a2_minus_a1": _number(row["delta_exact_set_a2_minus_a1"], field="delta_exact_set_a2_minus_a1", source=str(row["method"])),
                "delta_example_jaccard_a2_minus_a1": _number(row["delta_example_jaccard_a2_minus_a1"], field="delta_example_jaccard_a2_minus_a1", source=str(row["method"])),
                "delta_act_cpmr_a2_minus_a1": _number(row["delta_act_cpmr_a2_minus_a1"], field="delta_act_cpmr_a2_minus_a1", source=str(row["method"])),
                "delta_means_cpmr_a2_minus_a1": _number(row["delta_means_cpmr_a2_minus_a1"], field="delta_means_cpmr_a2_minus_a1", source=str(row["method"])),
                "delta_purpose_cpmr_a2_minus_a1": _number(row["delta_purpose_cpmr_a2_minus_a1"], field="delta_purpose_cpmr_a2_minus_a1", source=str(row["method"])),
                "significance_claim": "NOT_TESTED_DO_NOT_INFER",
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    return output


def _m3_vs_m4_tables(
    main_by_eval: Mapping[str, Sequence[Mapping[str, Any]]],
    per_label_by_eval: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    aggregate_metrics = (
        "macro_f1",
        "micro_f1",
        "exact_set_accuracy",
        "example_jaccard",
        "act_cpmr",
        "means_cpmr",
        "purpose_cpmr",
        "act_mean_contained_recall",
        "means_mean_contained_recall",
        "purpose_mean_contained_recall",
    )
    for evaluation in ("A1", "A2"):
        main = {str(row["method"]): row for row in main_by_eval[evaluation]}
        result: dict[str, Any] = {
            "evaluation": evaluation,
            "n": EXPECTED_TEST_N[evaluation],
            "comparison": "M4_SIX_SHOT_MINUS_M3_ZERO_SHOT",
        }
        for metric in aggregate_metrics:
            result[f"delta_{metric}_m4_minus_m3"] = float(main["M4"][metric]) - float(main["M3"][metric])
        result["significance_claim"] = "NOT_TESTED_DO_NOT_INFER"
        result["reference_terminology"] = SILVER_REFERENCE_TERM
        aggregate.append(result)

        index = _index_per_label(per_label_by_eval[evaluation], evaluation=evaluation)
        for label in AMP_LABEL_IDS:
            m3 = index[("M3", label)]
            m4 = index[("M4", label)]
            support = _integer(m3["support"], field="support", source=f"{evaluation}/{label}")
            if support != _integer(m4["support"], field="support", source=f"{evaluation}/{label}"):
                raise EvaluationAFinalizationError(f"M3/M4 support mismatch: {evaluation}/{label}")
            m3_f1 = _optional_number(m3["f1"], field="f1", source=f"{evaluation}/M3/{label}")
            m4_f1 = _optional_number(m4["f1"], field="f1", source=f"{evaluation}/M4/{label}")
            if (m3_f1 is None) != (m4_f1 is None):
                raise EvaluationAFinalizationError(f"M3/M4 F1 definedness mismatch: {evaluation}/{label}")
            per_label.append(
                {
                    "evaluation": evaluation,
                    "label_id": label,
                    "family": AMP_FAMILY_BY_LABEL[label].title(),
                    "support": support,
                    "m3_f1": "N/A" if m3_f1 is None else m3_f1,
                    "m4_f1": "N/A" if m4_f1 is None else m4_f1,
                    "delta_f1_m4_minus_m3": "N/A" if m3_f1 is None else m4_f1 - m3_f1,
                    "status": str(m3["status"]),
                    "significance_claim": "NOT_TESTED_DO_NOT_INFER",
                    "reference_terminology": SILVER_REFERENCE_TERM,
                }
            )
    return aggregate, per_label


def _a2_fold_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _require_fields(
        rows,
        ("method", "fold", "test_n", "macro_f1", "micro_f1", "example_jaccard", "exact_set_accuracy", "act_cpmr", "means_cpmr", "purpose_cpmr"),
        source="A2 per-fold results",
    )
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        method = str(row["method"])
        fold = _integer(row["fold"], field="fold", source="A2 per-fold")
        key = (fold, method)
        if method not in EXPECTED_METHODS or fold not in EXPECTED_A2_FOLD_N or key in seen:
            raise EvaluationAFinalizationError(f"Unexpected/duplicate A2 fold row: {key}")
        seen.add(key)
        n = _integer(row["test_n"], field="test_n", source=f"A2/fold-{fold}/{method}")
        if n != EXPECTED_A2_FOLD_N[fold]:
            raise EvaluationAFinalizationError(f"Wrong frozen Fold {fold} N for {method}: {n}")
        output.append(
            {
                "fold": fold,
                "method": method,
                "n": n,
                "macro_f1": _number(row["macro_f1"], field="macro_f1", source=str(key)),
                "micro_f1": _number(row["micro_f1"], field="micro_f1", source=str(key)),
                "example_jaccard": _number(row["example_jaccard"], field="example_jaccard", source=str(key)),
                "exact_set_accuracy": _number(row["exact_set_accuracy"], field="exact_set_accuracy", source=str(key)),
                "act_cpmr": _number(row["act_cpmr"], field="act_cpmr", source=str(key)),
                "means_cpmr": _number(row["means_cpmr"], field="means_cpmr", source=str(key)),
                "purpose_cpmr": _number(row["purpose_cpmr"], field="purpose_cpmr", source=str(key)),
                "macro_supported_label_count": _integer(row["macro_label_count"], field="macro_label_count", source=str(key)),
                "status": "DESCRIPTIVE_FOLD_RESULT",
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    expected = {(fold, method) for fold in EXPECTED_A2_FOLD_N for method in EXPECTED_METHODS}
    if seen != expected:
        raise EvaluationAFinalizationError(f"Incomplete A2 fold grid: missing={sorted(expected - seen)}")
    return sorted(output, key=lambda row: (int(row["fold"]), EXPECTED_METHODS.index(str(row["method"]))))


def _a2_jurisdiction_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _require_fields(
        rows,
        ("method", "jurisdiction", "fold", "test_n", "macro_label_count", "macro_f1", "micro_f1", "example_jaccard", "act_cpmr", "means_cpmr", "purpose_cpmr"),
        source="A2 per-jurisdiction results",
    )
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    jurisdiction_methods: dict[str, set[str]] = defaultdict(set)
    jurisdiction_n: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        method = str(row["method"])
        jurisdiction = str(row["jurisdiction"])
        if method not in EXPECTED_METHODS or not jurisdiction:
            raise EvaluationAFinalizationError("Invalid A2 jurisdiction row")
        key = (jurisdiction, method)
        if key in seen:
            raise EvaluationAFinalizationError(f"Duplicate A2 jurisdiction row: {key}")
        seen.add(key)
        jurisdiction_methods[jurisdiction].add(method)
        n = _integer(row["test_n"], field="test_n", source=str(key))
        jurisdiction_n[jurisdiction].add(n)
        output.append(
            {
                "jurisdiction": jurisdiction,
                "fold": _integer(row["fold"], field="fold", source=str(key)),
                "n": n,
                "method": method,
                "supported_label_count_for_macro_f1": _integer(row["macro_label_count"], field="macro_label_count", source=str(key)),
                "macro_f1": _number(row["macro_f1"], field="macro_f1", source=str(key)),
                "micro_f1": _number(row["micro_f1"], field="micro_f1", source=str(key)),
                "example_jaccard": _number(row["example_jaccard"], field="example_jaccard", source=str(key)),
                "act_cpmr": _number(row["act_cpmr"], field="act_cpmr", source=str(key)),
                "means_cpmr": _number(row["means_cpmr"], field="means_cpmr", source=str(key)),
                "purpose_cpmr": _number(row["purpose_cpmr"], field="purpose_cpmr", source=str(key)),
                "status": "DESCRIPTIVE_SMALL_N_CAUTION",
                "reference_terminology": SILVER_REFERENCE_TERM,
            }
        )
    if len(jurisdiction_methods) != 18:
        raise EvaluationAFinalizationError(
            f"A2 jurisdiction count={len(jurisdiction_methods)}; expected frozen count 18"
        )
    for jurisdiction in jurisdiction_methods:
        if jurisdiction_methods[jurisdiction] != set(EXPECTED_METHODS) or len(jurisdiction_n[jurisdiction]) != 1:
            raise EvaluationAFinalizationError(f"Incomplete/inconsistent jurisdiction grid: {jurisdiction}")
    return sorted(output, key=lambda row: (str(row["jurisdiction"]), EXPECTED_METHODS.index(str(row["method"]))))


def _label_mapping() -> list[dict[str, Any]]:
    return [
        {
            "ontology_order": index,
            "label_id": label,
            "family": AMP_FAMILY_BY_LABEL[label].title(),
            "display_label": DISPLAY_LABELS[label],
            "figure_short_label": SHORT_LABELS[label],
        }
        for index, label in enumerate(AMP_LABEL_IDS, start=1)
    ]


def _configure_matplotlib() -> Any:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sherloc_evaluation_a_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "sherloc-evaluation-a-v1",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _save_svg(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".svg", dir=path.parent)
    os.close(descriptor)
    try:
        fig.savefig(
            temporary,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": None, "Creator": "SHERLOC Evaluation A deterministic generator"},
        )
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _generate_figures(
    figure_dir: Path,
    main_by_eval: Mapping[str, Sequence[Mapping[str, Any]]],
    per_label_by_eval: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    plt = _configure_matplotlib()
    method_index = {method: index for index, method in enumerate(EXPECTED_METHODS)}
    main = {
        evaluation: {str(row["method"]): row for row in rows}
        for evaluation, rows in main_by_eval.items()
    }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Figure 1: core primary metrics, A1 versus pooled A2.
    metrics = (("macro_f1", "Macro-F1"), ("micro_f1", "Micro-F1"), ("example_jaccard", "Example Jaccard"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.25), sharey=True)
    x = np.arange(len(EXPECTED_METHODS))
    width = 0.36
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for offset, evaluation in ((-width / 2, "A1"), (width / 2, "A2")):
            axis.bar(
                x + offset,
                [float(main[evaluation][method][metric]) for method in EXPECTED_METHODS],
                width,
                label="A1 IID" if evaluation == "A1" else "A2 jurisdiction-OOD",
            )
        axis.set_title(title)
        axis.set_xticks(x, EXPECTED_METHODS)
        axis.set_ylim(0, 0.8)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Core AMP performance under IID and jurisdiction-OOD evaluation", y=1.01)
    fig.tight_layout()
    _save_svg(fig, figure_dir / FIGURE_FILENAMES[0])
    plt.close(fig)

    # Figure 2: family CPMR by method and evaluation.
    family_metrics = (("act_cpmr", "Act"), ("means_cpmr", "Means"), ("purpose_cpmr", "Purpose"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.35), sharey=True)
    width = 0.23
    for axis, evaluation in zip(axes, ("A1", "A2"), strict=True):
        for family_position, (metric, label) in enumerate(family_metrics):
            offset = (family_position - 1) * width
            axis.bar(
                x + offset,
                [float(main[evaluation][method][metric]) for method in EXPECTED_METHODS],
                width,
                label=label,
            )
        axis.set_title("A1 IID" if evaluation == "A1" else "A2 jurisdiction-OOD")
        axis.set_xticks(x, EXPECTED_METHODS)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Contained Partial Match Rate")
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle("Reference-contained prediction behavior by AMP family", y=1.01)
    fig.tight_layout()
    _save_svg(fig, figure_dir / FIGURE_FILENAMES[1])
    plt.close(fig)

    # Figure 3: CPMR versus conditional contained recall. Neither axis is a standalone ranking.
    from matplotlib.lines import Line2D

    family_pairs = (
        ("act_cpmr", "act_mean_contained_recall", "Act", "o"),
        ("means_cpmr", "means_mean_contained_recall", "Means", "s"),
        ("purpose_cpmr", "purpose_mean_contained_recall", "Purpose", "^"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True, sharey=True)
    for axis, evaluation in zip(axes, ("A1", "A2"), strict=True):
        for method in EXPECTED_METHODS:
            color = colors[method_index[method] % len(colors)]
            for cpmr_field, recall_field, _family, marker in family_pairs:
                axis.scatter(
                    float(main[evaluation][method][cpmr_field]),
                    float(main[evaluation][method][recall_field]),
                    color=color,
                    marker=marker,
                    s=42,
                    edgecolor="white",
                    linewidth=0.5,
                )
        axis.set_title("A1 IID" if evaluation == "A1" else "A2 jurisdiction-OOD")
        axis.set_xlim(0, 1)
        axis.set_ylim(0.65, 1.01)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.set_xlabel("CPMR (frequency of contained extraction)")
    axes[0].set_ylabel("Mean contained recall\n(conditional on CPMR success)")
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=colors[index % len(colors)], label=method)
        for index, method in enumerate(EXPECTED_METHODS)
    ]
    family_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color="0.25", label=family)
        for _cpmr, _recall, family, marker in family_pairs
    ]
    legend1 = axes[1].legend(handles=method_handles, title="Method", loc="lower left", frameon=False)
    axes[1].add_artist(legend1)
    axes[1].legend(handles=family_handles, title="Family", loc="lower right", frameon=False)
    fig.suptitle("Contained-extraction frequency and conditional completeness", y=1.01)
    fig.tight_layout()
    _save_svg(fig, figure_dir / FIGURE_FILENAMES[2])
    plt.close(fig)

    # Figure 4: per-label F1 heatmaps, with support and explicit N/A cells.
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 6.2), squeeze=False)
    image = None
    for row_index, evaluation in enumerate(("A1", "A2")):
        index = _index_per_label(per_label_by_eval[evaluation], evaluation=evaluation)
        for column_index, family in enumerate(AMP_FAMILIES):
            axis = axes[row_index, column_index]
            labels = [label for label in AMP_LABEL_IDS if AMP_FAMILY_BY_LABEL[label] == family]
            matrix = np.full((len(EXPECTED_METHODS), len(labels)), np.nan, dtype=float)
            supports: list[int] = []
            for label_index, label in enumerate(labels):
                support_values = {
                    _integer(index[(method, label)]["support"], field="support", source=f"{evaluation}/{method}/{label}")
                    for method in EXPECTED_METHODS
                }
                if len(support_values) != 1:
                    raise EvaluationAFinalizationError(f"Figure support mismatch: {evaluation}/{label}")
                supports.append(support_values.pop())
                for method_row, method in enumerate(EXPECTED_METHODS):
                    matrix[method_row, label_index] = (
                        _optional_number(index[(method, label)]["f1"], field="f1", source=f"{evaluation}/{method}/{label}")
                        if index[(method, label)]["f1"] not in {"", "N/A"}
                        else np.nan
                    )
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for method_row in range(len(EXPECTED_METHODS)):
                for label_index in range(len(labels)):
                    value = matrix[method_row, label_index]
                    text = "N/A" if np.isnan(value) else f"{value:.2f}"
                    color = "white" if not np.isnan(value) and value >= 0.65 else "black"
                    axis.text(label_index, method_row, text, ha="center", va="center", fontsize=7, color=color)
            axis.set_xticks(
                np.arange(len(labels)),
                [f"{SHORT_LABELS[label]}\nn={support}" for label, support in zip(labels, supports, strict=True)],
                rotation=35,
                ha="right",
            )
            axis.set_yticks(np.arange(len(EXPECTED_METHODS)), EXPECTED_METHODS)
            axis.set_title(f"{evaluation} · {family.title()}")
            axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(EXPECTED_METHODS), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.7)
            axis.tick_params(which="minor", bottom=False, left=False)
    if image is None:  # pragma: no cover - defensive.
        raise EvaluationAFinalizationError("No per-label figure data")
    colorbar_axis = fig.add_axes((0.935, 0.22, 0.012, 0.56))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Per-label F1")
    fig.suptitle("Per-label AMP F1 (support shown below each label; N/A = zero reference support)", y=0.995)
    fig.subplots_adjust(left=0.055, right=0.905, bottom=0.18, top=0.91, wspace=0.2, hspace=0.62)
    _save_svg(fig, figure_dir / FIGURE_FILENAMES[3])
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or str(value) in {"", "N/A"}:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _fmt_delta(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    escaped = [[str(cell).replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped)
    return "\n".join(lines)


def _main_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    return _markdown_table(
        ("Method", "N", "Macro-F1 [95% CI]", "Micro-F1 [95% CI]", "Exact set [95% CI]", "Jaccard [95% CI]", "Act CPMR/MCR", "Means CPMR/MCR", "Purpose CPMR/MCR"),
        [
            (
                row["method"],
                row["n"],
                f"{_fmt(row['macro_f1'])} [{_fmt(row['macro_f1_ci_lower'])}, {_fmt(row['macro_f1_ci_upper'])}]",
                f"{_fmt(row['micro_f1'])} [{_fmt(row['micro_f1_ci_lower'])}, {_fmt(row['micro_f1_ci_upper'])}]",
                f"{_fmt(row['exact_set_accuracy'])} [{_fmt(row['exact_set_accuracy_ci_lower'])}, {_fmt(row['exact_set_accuracy_ci_upper'])}]",
                f"{_fmt(row['example_jaccard'])} [{_fmt(row['example_jaccard_ci_lower'])}, {_fmt(row['example_jaccard_ci_upper'])}]",
                f"{_fmt(row['act_cpmr'])}/{_fmt(row['act_mean_contained_recall'])}",
                f"{_fmt(row['means_cpmr'])}/{_fmt(row['means_mean_contained_recall'])}",
                f"{_fmt(row['purpose_cpmr'])}/{_fmt(row['purpose_mean_contained_recall'])}",
            )
            for row in rows
        ],
    )


def _rare_note(rows: Sequence[Mapping[str, Any]]) -> str:
    sensitivity = [row for row in rows if row["record_type"] == "MACRO_SENSITIVITY"]
    a1 = [row for row in sensitivity if row["evaluation"] == "A1"]
    a2 = [row for row in sensitivity if row["evaluation"] == "A2"]
    a1_table = _markdown_table(
        ("Method", "Official 17-label Macro-F1", "Diagnostic 16-label Macro-F1", "Diagnostic − official"),
        [
            (row["method"], _fmt(row["official_macro_f1"], 6), _fmt(row["diagnostic_macro_f1"], 6), _fmt_delta(row["delta_diagnostic_minus_official"], 6))
            for row in a1
        ],
    )
    a2_table = _markdown_table(
        ("Method", "Official supported-label Macro-F1", "Supported labels"),
        [(row["method"], _fmt(row["official_macro_f1"], 6), row["official_macro_label_count"]) for row in a2],
    )
    return f"""# Evaluation A rare-label sensitivity

Status: **DESCRIPTIVE SENSITIVITY ANALYSIS; NOT A REPLACEMENT FOR THE CANONICAL METRIC**  
Generator: `src/experiments/16_finalize_evaluation_a.py` v{VERSION}

## Why this diagnostic is reported

Macro-F1 assigns equal weight to every eligible label. Consequently, an ultra-rare
label can materially change the arithmetic mean even though it affects very few
cases. This is a property of Macro-F1, not evidence that the rare label should be
removed from the frozen task.

`PURPOSE_REMOVAL_OF_ORGANS` has silver-reference support **2** in A1. The official
A1 Macro-F1 therefore uses all 17 frozen AMP labels. The diagnostic below excludes
that single label and averages the other 16 canonical per-label F1 values. It does
not change the official result.

{a1_table}

The A1 organ-removal per-label F1 is 0 for M1/M2 and 1 for M3/M4 on two supported
cases. This extreme small-support result explains why excluding the label moves
the methods' macro averages in different directions; it should not be generalized.

## A2 zero-support rule

In pooled A2, `PURPOSE_REMOVAL_OF_ORGANS` has silver-reference support **0**. It
remains one of the 17 prediction dimensions, so an organ-removal false positive
would still affect all-label micro and set metrics. Its per-label precision,
recall, and F1 are `N/A`, and it is already excluded from the official A2
Macro-F1, which averages the 16 positive-support labels.

{a2_table}

The complete two-evaluation support profile for every AMP label is preserved in
[`rare_label_sensitivity.csv`](../outputs/analysis/evaluation_a/rare_label_sensitivity.csv).

## Interpretation boundary

This analysis is descriptive. It was computed after the predictions were frozen,
does not alter any model or threshold, and must not replace the canonical
Macro-F1. All reference labels discussed here are SHERLOC silver-reference labels,
not human-grounded gold.
"""


def _final_report(
    *,
    a1_main: Sequence[Mapping[str, Any]],
    a2_main: Sequence[Mapping[str, Any]],
    family: Sequence[Mapping[str, Any]],
    breadth: Sequence[Mapping[str, Any]],
    rare: Sequence[Mapping[str, Any]],
    shift: Sequence[Mapping[str, Any]],
    m3_m4: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    jurisdictions: Sequence[Mapping[str, Any]],
    a1_usage: Sequence[Mapping[str, Any]],
    a2_usage: Sequence[Mapping[str, Any]],
) -> str:
    shift_table = _markdown_table(
        ("Method", "Δ Macro", "Δ Micro", "Δ Exact", "Δ Jaccard", "Δ Act CPMR", "Δ Means CPMR", "Δ Purpose CPMR"),
        [
            (
                row["method"],
                _fmt_delta(row["delta_macro_f1_a2_minus_a1"]),
                _fmt_delta(row["delta_micro_f1_a2_minus_a1"]),
                _fmt_delta(row["delta_exact_set_a2_minus_a1"]),
                _fmt_delta(row["delta_example_jaccard_a2_minus_a1"]),
                _fmt_delta(row["delta_act_cpmr_a2_minus_a1"]),
                _fmt_delta(row["delta_means_cpmr_a2_minus_a1"]),
                _fmt_delta(row["delta_purpose_cpmr_a2_minus_a1"]),
            )
            for row in shift
        ],
    )
    m3_m4_table = _markdown_table(
        ("Evaluation", "N", "Δ Macro", "Δ Micro", "Δ Exact", "Δ Jaccard", "Δ Act CPMR/MCR", "Δ Means CPMR/MCR", "Δ Purpose CPMR/MCR"),
        [
            (
                row["evaluation"],
                row["n"],
                _fmt_delta(row["delta_macro_f1_m4_minus_m3"]),
                _fmt_delta(row["delta_micro_f1_m4_minus_m3"]),
                _fmt_delta(row["delta_exact_set_accuracy_m4_minus_m3"]),
                _fmt_delta(row["delta_example_jaccard_m4_minus_m3"]),
                f"{_fmt_delta(row['delta_act_cpmr_m4_minus_m3'])}/{_fmt_delta(row['delta_act_mean_contained_recall_m4_minus_m3'])}",
                f"{_fmt_delta(row['delta_means_cpmr_m4_minus_m3'])}/{_fmt_delta(row['delta_means_mean_contained_recall_m4_minus_m3'])}",
                f"{_fmt_delta(row['delta_purpose_cpmr_m4_minus_m3'])}/{_fmt_delta(row['delta_purpose_mean_contained_recall_m4_minus_m3'])}",
            )
            for row in m3_m4
        ],
    )
    family_table = _markdown_table(
        ("Evaluation", "Method", "Family", "Supported labels", "Mean P", "Mean R", "Mean F1"),
        [
            (row["evaluation"], row["method"], row["family"], row["supported_label_count"], _fmt(row["macro_precision_family"]), _fmt(row["macro_recall_family"]), _fmt(row["macro_f1_family"]))
            for row in family
        ],
    )
    breadth_table = _markdown_table(
        ("Evaluation", "Method", "Mean Act", "Mean Means", "Mean Purpose", "Mean predicted total", "Mean silver total", "Median [P25, P75]"),
        [
            (
                row["evaluation"], row["method"], _fmt(row["mean_predicted_act_labels"]), _fmt(row["mean_predicted_means_labels"]), _fmt(row["mean_predicted_purpose_labels"]), _fmt(row["mean_total_predicted_labels"]), _fmt(row["mean_total_silver_reference_labels"]), f"{_fmt(row['median_total_predicted_labels'])} [{_fmt(row['p25_total_predicted_labels'])}, {_fmt(row['p75_total_predicted_labels'])}]",
            )
            for row in breadth
        ],
    )
    rare_sensitivity = [row for row in rare if row["record_type"] == "MACRO_SENSITIVITY"]
    rare_table = _markdown_table(
        ("Evaluation", "Method", "Official Macro-F1", "Diagnostic without organ removal", "Difference"),
        [
            (row["evaluation"], row["method"], _fmt(row["official_macro_f1"]), _fmt(row["diagnostic_macro_f1"]), _fmt_delta(row["delta_diagnostic_minus_official"]))
            for row in rare_sensitivity
        ],
    )
    fold_table = _markdown_table(
        ("Fold", "Method", "N", "Macro-F1", "Micro-F1", "Jaccard", "Exact set", "Act CPMR", "Means CPMR", "Purpose CPMR"),
        [
            (row["fold"], row["method"], row["n"], _fmt(row["macro_f1"]), _fmt(row["micro_f1"]), _fmt(row["example_jaccard"]), _fmt(row["exact_set_accuracy"]), _fmt(row["act_cpmr"]), _fmt(row["means_cpmr"]), _fmt(row["purpose_cpmr"]))
            for row in folds
        ],
    )

    a1_usage_by_method = {row["method"]: row for row in a1_usage if row.get("method") in {"M3", "M4"}}
    a2_usage_by_method = {
        row["method"]: row
        for row in a2_usage
        if row.get("method") in {"M3", "M4"} and row.get("scope") == "POOLED_OOD_TEST"
    }
    if set(a1_usage_by_method) != {"M3", "M4"} or set(a2_usage_by_method) != {"M3", "M4"}:
        raise EvaluationAFinalizationError("Incomplete canonical M3/M4 API usage summaries")
    usage_rows: list[tuple[Any, ...]] = []
    for evaluation, lookup in (("A1", a1_usage_by_method), ("A2", a2_usage_by_method)):
        for method in ("M3", "M4"):
            row = lookup[method]
            if evaluation == "A1":
                success = _integer(row["successful_request_count"], field="successful_request_count", source=f"{evaluation}/{method}")
                attempts = _integer(row["api_attempt_count_including_retries"], field="api_attempt_count_including_retries", source=f"{evaluation}/{method}")
                retries = _integer(row["retry_count"], field="retry_count", source=f"{evaluation}/{method}")
                tokens = _integer(row["total_tokens"], field="total_tokens", source=f"{evaluation}/{method}")
                runtime = "N/A"
            else:
                success = _integer(row["successful_request_count"], field="successful_request_count", source=f"{evaluation}/{method}")
                retries = _integer(row["retry_count"], field="retry_count", source=f"{evaluation}/{method}")
                attempts = success + retries
                tokens = _integer(row["recorded_total_tokens"], field="recorded_total_tokens", source=f"{evaluation}/{method}")
                runtime = f"{_number(row['runtime_seconds'], field='runtime_seconds', source=f'{evaluation}/{method}'):.0f}s"
            usage_rows.append(
                (
                    evaluation,
                    method,
                    success,
                    attempts,
                    retries,
                    f"{tokens:,}",
                    _fmt(row["median_latency_seconds"]),
                    _fmt(row["p90_latency_seconds"]),
                    runtime,
                )
            )
    usage_table = _markdown_table(
        ("Evaluation", "Method", "Successful cases", "API attempts", "Retries", "Recorded tokens", "Median latency", "P90 latency", "Active runtime"),
        usage_rows,
    )

    unique_jurisdictions = sorted({str(row["jurisdiction"]) for row in jurisdictions})
    jurisdiction_ns = sorted({int(row["n"]) for row in jurisdictions})
    return f"""# Final Evaluation A report

Status: **AUTHORITATIVE PAPER-ANALYSIS PACKAGE FROM FROZEN PREDICTIONS**  
Generator: `src/experiments/16_finalize_evaluation_a.py` v{VERSION}

## 1. Objective

Evaluation A compares four frozen methods for extracting 17 Act/Means/Purpose
(AMP) dimensions from English SHERLOC Fact Summaries. The large-scale targets
are SHERLOC **silver-reference labels**. They are distinct from the later
human-grounded gold annotations.

## 2. A1 IID design

A1 is the frozen IID TEST split with **N=253**. M1 and M2 selection and threshold
choices used only frozen TRAIN/VALIDATION data. M3 and M4 used the already-frozen
zero-shot and six-shot prompts. The final table below copies point estimates and
95% case-bootstrap intervals from the canonical evaluator.

## 3. A2 jurisdiction-OOD design

A2 pools three jurisdiction-disjoint TEST folds: Fold 1 N=288, Fold 2 N=287,
and Fold 3 N=286, for **N=861** unique cases. Each held-out jurisdiction is TEST
in one fold. Demonstration jurisdictions were disjoint from the corresponding
M4 TEST jurisdictions.

## 4. Methods

- **M1:** TF-IDF features with one-vs-rest logistic regression.
- **M2:** `answerdotai/ModernBERT-base` with a 17-logit multilabel head.
- **M3:** `gpt-5.6-luna` zero-shot structured AMP extraction.
- **M4:** the same LLM configuration and extraction instructions as M3 with six
  frozen solved demonstrations appropriate to the split/fold.

## 5. Frozen samples and label space

The primary cohort contains 1,263 cases and the ontology contains 5 Act, 6 Means,
and 6 Purpose labels. A1 Macro-F1 uses all 17 labels. A2 retains all 17 prediction
dimensions, but pooled A2 Macro-F1 averages the 16 labels with positive
silver-reference support because `PURPOSE_REMOVAL_OF_ORGANS` has support zero.

## 6. Primary metrics

Per-label precision, recall, and F1 use the canonical multilabel counts.
Macro-F1 is the unweighted mean over eligible supported labels; Micro-F1 pools
counts across all 17 dimensions. Exact-set accuracy requires equality of the
complete predicted and reference sets. Example Jaccard is the mean case-level
intersection-over-union. Confidence intervals are the frozen 1,000-resample
case bootstrap with seed `20260811`.

## 7. Contained Partial Match Rate

For one AMP family, case-level CPMR is 1 exactly when the predicted set is
nonempty and is a subset of that case's silver-reference set; otherwise it is 0.
Group CPMR is its case mean. CPMR is a secondary diagnostic of
reference-contained prediction behavior, not absolute factual correctness.

## 8. Mean Contained Recall

Contained Recall is the predicted/reference set-size ratio only on CPMR-successful
cases. Mean Contained Recall averages those defined values. It describes
conditional completeness and must be read alongside CPMR, not as a standalone
method ranking.

## 9. Final A1 IID results

{_main_markdown(a1_main)}

Canonical machine-readable table: [`a1_main_comparison.csv`](../outputs/analysis/evaluation_a/a1_main_comparison.csv).

## 10. Final pooled A2 jurisdiction-OOD results

{_main_markdown(a2_main)}

Canonical machine-readable table: [`a2_main_comparison.csv`](../outputs/analysis/evaluation_a/a2_main_comparison.csv).

## 11. A1 to A2 descriptive shifts

Each delta is pooled A2 minus A1. No statistical-significance test was designated,
so the table supports descriptive comparison only.

{shift_table}

## 12. M3 versus M4

Each value is M4 six-shot minus M3 zero-shot. M4 has higher CPMR in all three
families in both evaluations, while the conventional aggregate metrics have
mixed directions and small differences. This is not a uniform superiority claim.

{m3_m4_table}

Per-label F1 differences, including A2 organ-removal `N/A`, are preserved in
[`m3_vs_m4_per_label_f1.csv`](../outputs/analysis/evaluation_a/m3_vs_m4_per_label_f1.csv).

## 13. Family-level results

These are unweighted means of canonical per-label precision, recall, and F1
within each family. Undefined zero-support labels are excluded, never treated as
numeric zero. Thus A2 Purpose uses five supported labels; the other cells use all
labels in their family.

{family_table}

## 14. Prediction breadth

Breadth is descriptive and is not a primary performance metric. M1/M2 generally
produce broader Act/Means sets, whereas M3/M4 produce narrower sets; this pattern
helps contextualize their different recall and CPMR behavior. Full family-specific
reference means and zero-prediction proportions are in the CSV.

{breadth_table}

## 15. Rare-label sensitivity

The diagnostic A1 value excludes `PURPOSE_REMOVAL_OF_ORGANS` (support 2); it does
not replace the official 17-label result. In A2, organ removal has support 0 and
is already excluded from the official 16-supported-label Macro-F1.

{rare_table}

See [`evaluation_a_rare_label_sensitivity.md`](evaluation_a_rare_label_sensitivity.md)
and the complete per-label support profile in
[`rare_label_sensitivity.csv`](../outputs/analysis/evaluation_a/rare_label_sensitivity.csv).

## 16. Fold and jurisdiction heterogeneity

{fold_table}

The paper-facing jurisdiction table preserves {len(unique_jurisdictions)}
jurisdictions and all four methods; jurisdiction N ranges from
{min(jurisdiction_ns)} to {max(jurisdiction_ns)}. These results are descriptive.
Do not rank jurisdictions from these cells, and do not overinterpret small-N
differences. See
[`a2_jurisdiction_summary.csv`](../outputs/analysis/evaluation_a/a2_jurisdiction_summary.csv).

## 17. M3/M4 API execution summary

{usage_table}

A2 token totals reflect recorded successful-response usage and are lower bounds
when failed-request usage was unavailable. Technical retry policies, including
the one-case M4 A2 rank-1340 rate-limit exception, changed no model, prompt,
schema, demonstration, target text, or scoring rule.

## 18. Methodological cautions and limitations

- SHERLOC labels are silver-reference labels; later adjudicated annotations will
  be human-grounded gold.
- CPMR captures reference-contained behavior. It does not establish factual or
  legal correctness when the silver reference is incomplete or mismatched to the
  narrative.
- A1 and A2 compare different case sets; reported deltas have no designated
  significance test.
- A2 per-jurisdiction cells can be small and are descriptive only.
- The two A1 organ-removal cases make its per-label F1 and Macro-F1 contribution
  unusually sensitive; the canonical metric remains unchanged.
- Purpose behaves differently from Act and Means, including much higher CPMR for
  every method, so a single aggregate characterization can obscure family-level
  behavior.
- The four methods do not have one uniform ordering across Macro-F1, Micro-F1,
  exact set, Jaccard, CPMR, and conditional contained recall.

## Core figures

1. [`A1 versus A2 core performance`](../outputs/figures/evaluation_a/{FIGURE_FILENAMES[0]})
2. [`CPMR by AMP family`](../outputs/figures/evaluation_a/{FIGURE_FILENAMES[1]})
3. [`CPMR versus Mean Contained Recall`](../outputs/figures/evaluation_a/{FIGURE_FILENAMES[2]})
4. [`Per-label F1`](../outputs/figures/evaluation_a/{FIGURE_FILENAMES[3]})
"""


def _check_outputs(
    *,
    analysis_dir: Path,
    figure_dir: Path,
    docs_dir: Path,
    expected_text: Mapping[Path, str],
    expected_figure_hashes: Mapping[str, str],
) -> None:
    for path, text in expected_text.items():
        _require_file(path)
        if path.read_text(encoding="utf-8") != text:
            raise EvaluationAFinalizationError(f"Generated text artifact is stale: {path}")
    actual_figures = sorted(path.name for path in figure_dir.glob("*.svg"))
    if actual_figures != sorted(FIGURE_FILENAMES):
        raise EvaluationAFinalizationError(
            f"Evaluation A figure set is not exactly the frozen four: {actual_figures}"
        )
    for name in FIGURE_FILENAMES:
        path = figure_dir / name
        _require_file(path)
        if "<svg" not in path.read_text(encoding="utf-8")[:1000]:
            raise EvaluationAFinalizationError(f"Figure is not valid-looking SVG: {path}")
        if sha256_file(path) != expected_figure_hashes[name]:
            raise EvaluationAFinalizationError(f"Generated figure is stale: {path}")
    for name in ("evaluation_a_rare_label_sensitivity.md", "evaluation_a_final_report.md"):
        _require_file(docs_dir / name)


def generate(
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    check: bool = False,
) -> dict[str, Any]:
    """Build or validate the complete deterministic Evaluation A package."""

    inputs = {
        "a1_primary": metrics_root / "a1/amp_primary_results.csv",
        "a2_primary": metrics_root / "a2/amp_primary_results.csv",
        "a1_per_label": metrics_root / "a1/amp_per_label.csv",
        "a2_per_label": metrics_root / "a2/amp_per_label.csv",
        "a1_cases": metrics_root / "a1/amp_case_level_errors.csv",
        "a2_cases": metrics_root / "a2/amp_case_level_errors.csv",
        "shift": metrics_root / "amp_a1_to_a2_deltas.csv",
        "a2_folds": metrics_root / "a2/amp_per_fold.csv",
        "a2_jurisdictions": metrics_root / "a2/amp_per_jurisdiction.csv",
        "a1_usage": metrics_root / "a1/amp_llm_api_usage.csv",
        "a2_usage": metrics_root / "a2/amp_llm_api_usage.csv",
    }
    source_hashes = {name: sha256_file(_require_file(path)) for name, path in inputs.items()}
    loaded = {name: read_csv(path) for name, path in inputs.items()}

    a1_main = _main_table("A1", loaded["a1_primary"])
    a2_main = _main_table("A2", loaded["a2_primary"])
    main_by_eval = {"A1": a1_main, "A2": a2_main}
    per_label_by_eval = {"A1": loaded["a1_per_label"], "A2": loaded["a2_per_label"]}
    family = _family_table(per_label_by_eval)
    breadth = _prediction_breadth({"A1": loaded["a1_cases"], "A2": loaded["a2_cases"]})
    rare = _rare_label_sensitivity(per_label_by_eval, main_by_eval)
    shift = _shift_table(loaded["shift"])
    m3_m4, m3_m4_per_label = _m3_vs_m4_tables(main_by_eval, per_label_by_eval)
    folds = _a2_fold_table(loaded["a2_folds"])
    jurisdictions = _a2_jurisdiction_table(loaded["a2_jurisdictions"])
    mapping = _label_mapping()

    table_rows = {
        "a1_main_comparison.csv": a1_main,
        "a2_main_comparison.csv": a2_main,
        "amp_family_level_metrics.csv": family,
        "prediction_breadth_summary.csv": breadth,
        "rare_label_sensitivity.csv": rare,
        "a1_to_a2_distribution_shift.csv": shift,
        "m3_vs_m4_summary.csv": m3_m4,
        "m3_vs_m4_per_label_f1.csv": m3_m4_per_label,
        "a2_fold_summary.csv": folds,
        "a2_jurisdiction_summary.csv": jurisdictions,
        "amp_label_display_mapping.csv": mapping,
    }
    table_text = {analysis_dir / name: _csv_text(rows) for name, rows in table_rows.items()}
    rare_note = _rare_note(rare)
    final_report = _final_report(
        a1_main=a1_main,
        a2_main=a2_main,
        family=family,
        breadth=breadth,
        rare=rare,
        shift=shift,
        m3_m4=m3_m4,
        folds=folds,
        jurisdictions=jurisdictions,
        a1_usage=loaded["a1_usage"],
        a2_usage=loaded["a2_usage"],
    )
    text_outputs = table_text | {
        docs_dir / "evaluation_a_rare_label_sensitivity.md": rare_note,
        docs_dir / "evaluation_a_final_report.md": final_report,
    }

    if check:
        with tempfile.TemporaryDirectory(prefix="sherloc_evaluation_a_check_") as directory:
            expected_figure_dir = Path(directory)
            _generate_figures(expected_figure_dir, main_by_eval, per_label_by_eval)
            expected_figure_hashes = {
                name: sha256_file(expected_figure_dir / name) for name in FIGURE_FILENAMES
            }
        _check_outputs(
            analysis_dir=analysis_dir,
            figure_dir=figure_dir,
            docs_dir=docs_dir,
            expected_text=text_outputs,
            expected_figure_hashes=expected_figure_hashes,
        )
    else:
        for path, text in text_outputs.items():
            _atomic_write_text(path, text)
        _generate_figures(figure_dir, main_by_eval, per_label_by_eval)

    return {
        "generator_version": VERSION,
        "mode": "CHECK" if check else "WRITE",
        "source_sha256": source_hashes,
        "tables": {name: len(rows) for name, rows in table_rows.items()},
        "figures": list(FIGURE_FILENAMES),
        "documents": ["evaluation_a_rare_label_sensitivity.md", "evaluation_a_final_report.md"],
        "a1_n": EXPECTED_TEST_N["A1"],
        "a2_n": EXPECTED_TEST_N["A2"],
        "status": "EVALUATION_A_PAPER_ANALYSIS_READY",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--check", action="store_true", help="Validate existing text outputs and exact four-figure set")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate(
        metrics_root=args.metrics_root,
        analysis_dir=args.analysis_dir,
        figure_dir=args.figure_dir,
        docs_dir=args.docs_dir,
        check=args.check,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
