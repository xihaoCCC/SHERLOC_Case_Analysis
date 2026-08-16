#!/usr/bin/env python3
"""Build the deterministic, fail-closed pre-writing paper artifact package.

This stage is a presentation-only transformer. It reads frozen canonical
Evaluation A and Evaluation B artifacts, the separately generated paired
bootstrap table, and the separately generated auxiliary-extension artifacts.
It never recomputes a benchmark metric, trains a model, calls an API, or edits
an upstream artifact.

The write protocol is deliberately strict:

* every dependency and recorded upstream hash is validated first;
* every output byte is rendered in memory before any path is touched;
* an existing differing output is never overwritten;
* the pre-writing freeze manifest is written last.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ("M1", "M2", "M3", "M4")
EVALUATIONS = ("A1", "A2", "A3")
FAMILIES = ("ACT", "MEANS", "PURPOSE")

MASTER_FIELDS = (
    "evaluation",
    "reference_type",
    "method",
    "N",
    "macro_f1",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "micro_f1",
    "micro_f1_ci_low",
    "micro_f1_ci_high",
    "exact_set_accuracy",
    "exact_set_accuracy_ci_low",
    "exact_set_accuracy_ci_high",
    "example_jaccard",
    "example_jaccard_ci_low",
    "example_jaccard_ci_high",
    "act_cpmr",
    "means_cpmr",
    "purpose_cpmr",
    "act_mean_contained_recall",
    "means_mean_contained_recall",
    "purpose_mean_contained_recall",
)

MAIN_PAPER_FIELDS = (
    "evaluation",
    "reference_type",
    "method",
    "N",
    "macro_f1",
    "micro_f1",
    "example_jaccard",
    "act_cpmr",
    "means_cpmr",
    "purpose_cpmr",
)

SILVER_HUMAN_FIELDS = (
    "family",
    "substantive_n",
    "comparable_n",
    "silver_reference_unavailable_n",
    "exact_set_concordance",
    "mean_jaccard",
    "shared_label_count",
    "silver_only_label_count",
    "human_only_label_count",
    "silver_only_rate_of_silver_labels",
    "human_only_rate_of_human_labels",
    "proportion_silver_labels_supported_by_human",
    "proportion_human_labels_contained_in_silver",
    "availability_policy",
)

BEHAVIOR_FIELDS = (
    "evaluation",
    "reference_type",
    "method",
    "N",
    "mean_predicted_act_labels",
    "mean_predicted_means_labels",
    "mean_predicted_purpose_labels",
    "mean_total_predicted_labels",
    "mean_total_human_labels",
    "macro_f1",
    "micro_f1",
    "example_jaccard",
    "act_cpmr",
    "means_cpmr",
    "purpose_cpmr",
    "abstain_n",
    "abstain_all_amp_empty_rate",
    "abstain_mean_total_predicted_label_count",
)

AUXILIARY_FIELDS = (
    "target",
    "N",
    "macro_f1",
    "micro_f1",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "exact_set_accuracy",
    "example_jaccard",
    "model",
    "prompt_version",
    "reference_type",
)

PAIRED_REQUIRED_FIELDS = (
    "evaluation",
    "reference_type",
    "comparison",
    "first_method",
    "second_method",
    "metric",
    "n",
    "point_difference",
    "ci_low",
    "ci_high",
    "confidence_level",
    "bootstrap_resamples",
    "seed",
    "resampling_unit",
    "bootstrap_method",
    "macro_label_count",
    "macro_label_ids_json",
    "ci_excludes_zero",
)

PAIRED_COMPARISONS = (
    ("M3 - M2", "M3", "M2"),
    ("M4 - M2", "M4", "M2"),
    ("M4 - M3", "M4", "M3"),
)

PAIRED_METRICS = (
    "Macro-F1",
    "Micro-F1",
    "Exact-set accuracy",
    "Example-based Jaccard",
    "Act CPMR",
    "Means CPMR",
    "Purpose CPMR",
)

AUXILIARY_METRICS = {
    "GEOGRAPHIC_FORM": (
        "MACRO_F1",
        "MICRO_F1",
        "EXACT_SET_ACCURACY",
        "EXAMPLE_JACCARD",
    ),
    "VICTIM_MULTIPLICITY": ("ACCURACY", "MACRO_F1"),
    "CHILD_INVOLVEMENT": ("ACCURACY", "MACRO_F1"),
    "ORGANIZED_CRIMINAL_GROUP": (
        "ACCURACY",
        "PRECISION",
        "RECALL",
        "F1",
        "BALANCED_ACCURACY",
    ),
}

FIGURE_NAMES = (
    "figure_pf1_core_performance.svg",
    "figure_pf2_cpmr_by_family.svg",
    "figure_pf3_silver_human_reference_shift.svg",
)


class PaperPackageError(RuntimeError):
    """Raised when an input contract or immutable-output rule is violated."""


@dataclass(frozen=True)
class PackagePaths:
    root: Path
    analysis_dir: Path
    figure_dir: Path
    docs_dir: Path

    @classmethod
    def for_root(
        cls,
        root: Path,
        *,
        analysis_dir: Path | None = None,
        figure_dir: Path | None = None,
        docs_dir: Path | None = None,
    ) -> "PackagePaths":
        resolved = root.resolve()
        return cls(
            resolved,
            (analysis_dir or resolved / "outputs/analysis/paper_final").resolve(),
            (figure_dir or resolved / "outputs/figures/paper_final").resolve(),
            (docs_dir or resolved / "docs").resolve(),
        )


@dataclass(frozen=True)
class PackageSources:
    a1: tuple[dict[str, str], ...]
    a2: tuple[dict[str, str], ...]
    a3: tuple[dict[str, str], ...]
    silver_human: tuple[dict[str, str], ...]
    a3_breadth: tuple[dict[str, str], ...]
    a3_abstain: tuple[dict[str, str], ...]
    reference_comparison: tuple[dict[str, str], ...]
    paired: tuple[dict[str, str], ...]
    auxiliary: tuple[dict[str, str], ...]
    auxiliary_per_class: tuple[dict[str, str], ...]
    auxiliary_case_level: tuple[dict[str, str], ...]
    metadata: Mapping[str, Any]
    dependency_hashes: Mapping[str, str]
    baseline_validation: Mapping[str, Any]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PaperPackageError(f"Artifact is outside repository root: {path}") from exc


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise PaperPackageError(f"Required dependency is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperPackageError(f"Invalid JSON dependency: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperPackageError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path, required: Sequence[str] = ()) -> tuple[dict[str, str], ...]:
    _require_file(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PaperPackageError(f"CSV has no header: {path}")
            missing = set(required) - set(reader.fieldnames)
            if missing:
                raise PaperPackageError(
                    f"CSV {path} is missing required columns: {sorted(missing)}"
                )
            rows = tuple(dict(row) for row in reader)
    except UnicodeError as exc:
        raise PaperPackageError(f"CSV is not valid UTF-8: {path}") from exc
    if not rows:
        raise PaperPackageError(f"CSV has no data rows: {path}")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    if not rows:
        raise PaperPackageError("Refusing to render an empty CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue().encode("utf-8")


def _as_finite(value: str, *, field: str, source: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperPackageError(f"{source}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise PaperPackageError(f"{source}: {field} is not finite: {value!r}")
    return result


def _score(value: str, *, field: str, source: str) -> float:
    result = _as_finite(value, field=field, source=source)
    if not 0.0 <= result <= 1.0:
        raise PaperPackageError(f"{source}: {field} is outside [0,1]: {result}")
    return result


def _validate_method_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source: str,
    expected_n_field: str,
    expected_n: int,
) -> None:
    methods = [str(row.get("method", "")) for row in rows]
    if tuple(methods) != METHODS:
        raise PaperPackageError(f"{source}: expected ordered methods {METHODS}, observed {methods}")
    for row in rows:
        if int(row[expected_n_field]) != expected_n:
            raise PaperPackageError(
                f"{source}: {row['method']} has {expected_n_field}={row[expected_n_field]}, "
                f"expected {expected_n}"
            )


def _validate_scores(rows: Sequence[Mapping[str, str]], fields: Sequence[str], source: str) -> None:
    for row in rows:
        for field in fields:
            value = row.get(field, "")
            if value == "":
                raise PaperPackageError(f"{source}: missing required score {field}")
            _score(value, field=field, source=source)


def build_master_rows(
    a1: Sequence[Mapping[str, str]],
    a2: Sequence[Mapping[str, str]],
    a3: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Copy canonical metric strings into one harmonized 12-row table."""

    output: list[dict[str, str]] = []
    mappings = (
        (
            "A1",
            a1,
            {
                "N": "test_n",
                "macro_f1": "macro_f1",
                "macro_f1_ci_low": "macro_f1_ci_lower",
                "macro_f1_ci_high": "macro_f1_ci_upper",
                "micro_f1": "micro_f1",
                "micro_f1_ci_low": "micro_f1_ci_lower",
                "micro_f1_ci_high": "micro_f1_ci_upper",
                "exact_set_accuracy": "exact_set_accuracy",
                "exact_set_accuracy_ci_low": "exact_set_accuracy_ci_lower",
                "exact_set_accuracy_ci_high": "exact_set_accuracy_ci_upper",
                "example_jaccard": "example_jaccard",
                "example_jaccard_ci_low": "example_jaccard_ci_lower",
                "example_jaccard_ci_high": "example_jaccard_ci_upper",
                "act_cpmr": "act_cpmr",
                "means_cpmr": "means_cpmr",
                "purpose_cpmr": "purpose_cpmr",
                "act_mean_contained_recall": "act_mean_contained_recall",
                "means_mean_contained_recall": "means_mean_contained_recall",
                "purpose_mean_contained_recall": "purpose_mean_contained_recall",
            },
        ),
        (
            "A2",
            a2,
            {
                "N": "test_n",
                "macro_f1": "pooled_ood_macro_f1",
                "macro_f1_ci_low": "pooled_ood_macro_f1_ci_lower",
                "macro_f1_ci_high": "pooled_ood_macro_f1_ci_upper",
                "micro_f1": "pooled_micro_f1",
                "micro_f1_ci_low": "pooled_micro_f1_ci_lower",
                "micro_f1_ci_high": "pooled_micro_f1_ci_upper",
                "exact_set_accuracy": "pooled_exact_set_accuracy",
                "exact_set_accuracy_ci_low": "pooled_exact_set_accuracy_ci_lower",
                "exact_set_accuracy_ci_high": "pooled_exact_set_accuracy_ci_upper",
                "example_jaccard": "pooled_example_jaccard",
                "example_jaccard_ci_low": "pooled_example_jaccard_ci_lower",
                "example_jaccard_ci_high": "pooled_example_jaccard_ci_upper",
                "act_cpmr": "pooled_act_cpmr",
                "means_cpmr": "pooled_means_cpmr",
                "purpose_cpmr": "pooled_purpose_cpmr",
                "act_mean_contained_recall": "pooled_act_mean_contained_recall",
                "means_mean_contained_recall": "pooled_means_mean_contained_recall",
                "purpose_mean_contained_recall": "pooled_purpose_mean_contained_recall",
            },
        ),
        (
            "A3",
            a3,
            {
                "N": "n",
                "macro_f1": "macro_f1",
                "macro_f1_ci_low": "macro_f1_ci_low",
                "macro_f1_ci_high": "macro_f1_ci_high",
                "micro_f1": "micro_f1",
                "micro_f1_ci_low": "micro_f1_ci_low",
                "micro_f1_ci_high": "micro_f1_ci_high",
                "exact_set_accuracy": "exact_set",
                "exact_set_accuracy_ci_low": "exact_set_ci_low",
                "exact_set_accuracy_ci_high": "exact_set_ci_high",
                "example_jaccard": "jaccard",
                "example_jaccard_ci_low": "jaccard_ci_low",
                "example_jaccard_ci_high": "jaccard_ci_high",
                "act_cpmr": "act_cpmr",
                "means_cpmr": "means_cpmr",
                "purpose_cpmr": "purpose_cpmr",
                "act_mean_contained_recall": "act_mean_contained_recall",
                "means_mean_contained_recall": "means_mean_contained_recall",
                "purpose_mean_contained_recall": "purpose_mean_contained_recall",
            },
        ),
    )
    for evaluation, rows, mapping in mappings:
        for source_row in rows:
            row = {
                "evaluation": evaluation,
                "reference_type": source_row["reference_terminology"],
                "method": source_row["method"],
            }
            row.update({target: source_row[source] for target, source in mapping.items()})
            output.append(row)
    if len(output) != 12:
        raise PaperPackageError(f"Master table has {len(output)} rows; expected 12")
    return output


def build_main_paper_rows(master: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [{field: row[field] for field in MAIN_PAPER_FIELDS} for row in master]


def build_silver_human_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_family = {str(row["family"]).upper(): row for row in rows}
    if set(by_family) != set(FAMILIES):
        raise PaperPackageError(
            f"Silver/human summary families differ from {FAMILIES}: {sorted(by_family)}"
        )
    return [
        {field: by_family[family][field] for field in SILVER_HUMAN_FIELDS}
        for family in FAMILIES
    ]


def build_behavior_rows(
    main: Sequence[Mapping[str, str]],
    breadth: Sequence[Mapping[str, str]],
    abstain: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    main_by_method = {row["method"]: row for row in main}
    breadth_by_method = {row["method"]: row for row in breadth}
    abstain_by_method = {row["method"]: row for row in abstain}
    if any(set(mapping) != set(METHODS) for mapping in (main_by_method, breadth_by_method, abstain_by_method)):
        raise PaperPackageError("A3 behavior sources do not contain exactly M1-M4")
    output: list[dict[str, str]] = []
    for method in METHODS:
        metric = main_by_method[method]
        width = breadth_by_method[method]
        abstention = abstain_by_method[method]
        if metric["n"] != width["n"]:
            raise PaperPackageError(f"A3 N mismatch between main and breadth for {method}")
        output.append(
            {
                "evaluation": "A3",
                "reference_type": metric["reference_terminology"],
                "method": method,
                "N": metric["n"],
                "mean_predicted_act_labels": width["mean_predicted_act_labels"],
                "mean_predicted_means_labels": width["mean_predicted_means_labels"],
                "mean_predicted_purpose_labels": width["mean_predicted_purpose_labels"],
                "mean_total_predicted_labels": width["mean_total_predicted_labels"],
                "mean_total_human_labels": width["mean_total_human_labels"],
                "macro_f1": metric["macro_f1"],
                "micro_f1": metric["micro_f1"],
                "example_jaccard": metric["jaccard"],
                "act_cpmr": metric["act_cpmr"],
                "means_cpmr": metric["means_cpmr"],
                "purpose_cpmr": metric["purpose_cpmr"],
                "abstain_n": abstention["abstain_n"],
                "abstain_all_amp_empty_rate": abstention["all_amp_empty_rate"],
                "abstain_mean_total_predicted_label_count": abstention[
                    "mean_total_predicted_label_count"
                ],
            }
        )
    return output


def build_auxiliary_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    indexed: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (str(row["target"]).upper(), str(row["metric"]).upper())
        if key in indexed:
            raise PaperPackageError(f"Duplicate auxiliary target/metric row: {key}")
        indexed[key] = row
    expected = {
        (target, metric)
        for target, metrics in AUXILIARY_METRICS.items()
        for metric in metrics
    }
    if set(indexed) != expected:
        raise PaperPackageError(
            "Auxiliary result target/metric contract mismatch; "
            f"missing={sorted(expected - set(indexed))}, extra={sorted(set(indexed) - expected)}"
        )
    output: list[dict[str, str]] = []
    destination = {
        "MACRO_F1": "macro_f1",
        "MICRO_F1": "micro_f1",
        "ACCURACY": "accuracy",
        "PRECISION": "precision",
        "RECALL": "recall",
        "F1": "f1",
        "BALANCED_ACCURACY": "balanced_accuracy",
        "EXACT_SET_ACCURACY": "exact_set_accuracy",
        "EXAMPLE_JACCARD": "example_jaccard",
    }
    for target, metrics in AUXILIARY_METRICS.items():
        source_rows = [indexed[(target, metric)] for metric in metrics]
        n_values = {row["n"] for row in source_rows}
        models = {row["model"] for row in source_rows}
        prompts = {row["prompt_version"] for row in source_rows}
        if len(n_values) != 1:
            raise PaperPackageError(f"Auxiliary {target} has inconsistent evaluator Ns: {n_values}")
        evaluator_n = next(iter(n_values))
        if not 1 <= int(evaluator_n) <= 55:
            raise PaperPackageError(
                f"Auxiliary {target} evaluator N={evaluator_n}; expected a target-specific N in 1..55"
            )
        if len(models) != 1 or len(prompts) != 1:
            raise PaperPackageError(f"Auxiliary {target} model/prompt metadata are inconsistent")
        row = {field: "" for field in AUXILIARY_FIELDS}
        row.update(
            {
                "target": target,
                "N": evaluator_n,
                "model": next(iter(models)),
                "prompt_version": next(iter(prompts)),
                "reference_type": "single-reviewer human-grounded narrative reference",
            }
        )
        for metric in metrics:
            row[destination[metric]] = indexed[(target, metric)]["value"]
        output.append(row)
    return output


def validate_paired_rows(rows: Sequence[Mapping[str, str]]) -> None:
    expected = {
        (evaluation, comparison, metric)
        for evaluation in EVALUATIONS
        for comparison, _first, _second in PAIRED_COMPARISONS
        for metric in PAIRED_METRICS
    }
    observed: set[tuple[str, str, str]] = set()
    n_by_evaluation = {"A1": 253, "A2": 861, "A3": 55}
    reference_by_evaluation = {
        "A1": "SHERLOC silver reference",
        "A2": "SHERLOC silver reference",
        "A3": "human-grounded narrative reference",
    }
    comparison_methods = {
        comparison: (first, second)
        for comparison, first, second in PAIRED_COMPARISONS
    }
    for row in rows:
        key = (row["evaluation"], row["comparison"], row["metric"])
        if key in observed:
            raise PaperPackageError(f"Duplicate paired-bootstrap row: {key}")
        observed.add(key)
        evaluation, comparison, _metric = key
        if evaluation not in n_by_evaluation or comparison not in comparison_methods:
            raise PaperPackageError(f"Unexpected paired-bootstrap row: {key}")
        if int(row["n"]) != n_by_evaluation[evaluation]:
            raise PaperPackageError(f"Paired-bootstrap N mismatch: {key}")
        if row["reference_type"] != reference_by_evaluation[evaluation]:
            raise PaperPackageError(f"Paired-bootstrap reference mismatch: {key}")
        if (row["first_method"], row["second_method"]) != comparison_methods[comparison]:
            raise PaperPackageError(f"Paired-bootstrap method order mismatch: {key}")
        point = _as_finite(row["point_difference"], field="point_difference", source=str(key))
        low = _as_finite(row["ci_low"], field="ci_low", source=str(key))
        high = _as_finite(row["ci_high"], field="ci_high", source=str(key))
        if low > point or point > high:
            raise PaperPackageError(f"Paired-bootstrap point estimate is outside CI: {key}")
        if row["confidence_level"] != "0.95":
            raise PaperPackageError(f"Paired-bootstrap confidence level differs from 0.95: {key}")
        if row["bootstrap_resamples"] != "1000" or row["seed"] != "20260811":
            raise PaperPackageError(f"Paired-bootstrap protocol mismatch: {key}")
        expected_excludes = low > 0.0 or high < 0.0
        observed_excludes = row["ci_excludes_zero"].strip().lower() in {"1", "true", "yes"}
        if observed_excludes != expected_excludes:
            raise PaperPackageError(f"Paired-bootstrap zero-exclusion flag mismatch: {key}")
    if observed != expected:
        raise PaperPackageError(
            f"Paired-bootstrap contract mismatch; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _validate_auxiliary_detail_tables(
    per_class: Sequence[Mapping[str, str]],
    case_level: Sequence[Mapping[str, str]],
) -> None:
    expected_targets = set(AUXILIARY_METRICS)
    observed_targets = {str(row["target"]).upper() for row in per_class}
    if observed_targets != expected_targets:
        raise PaperPackageError(
            f"Auxiliary per-class targets mismatch: {sorted(observed_targets)}"
        )
    identities: set[str] = set()
    for row in case_level:
        case_id = row["reliability_case_id"]
        if case_id in identities:
            raise PaperPackageError(f"Duplicate auxiliary case-level ID: {case_id}")
        identities.add(case_id)
    if len(case_level) != 55:
        raise PaperPackageError(
            f"Auxiliary case-level table has {len(case_level)} rows; expected 55"
        )


def validate_evaluation_a_baseline(
    root: Path, baseline: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    scopes = baseline.get("scopes")
    if not isinstance(scopes, Mapping) or not scopes:
        raise PaperPackageError("Evaluation A integrity baseline has no scopes")
    dependency_hashes: dict[str, str] = {}
    scope_results: dict[str, Any] = {}
    for scope_name, raw_scope in sorted(scopes.items()):
        if not isinstance(raw_scope, Mapping):
            raise PaperPackageError(f"Invalid Evaluation A baseline scope: {scope_name}")
        files = raw_scope.get("files")
        if not isinstance(files, list) or len(files) != int(raw_scope.get("file_count", -1)):
            raise PaperPackageError(f"Evaluation A baseline file-count mismatch: {scope_name}")
        aggregate = hashlib.sha256()
        for item in sorted(files, key=lambda value: str(value["path"])):
            relative = str(item["path"])
            path = _require_file(root / relative)
            observed_hash = sha256_file(path)
            observed_size = path.stat().st_size
            if observed_hash != item.get("sha256") or observed_size != int(item.get("size", -1)):
                raise PaperPackageError(f"Evaluation A baseline mismatch: {relative}")
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(observed_hash.encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(str(observed_size).encode("ascii"))
            aggregate.update(b"\n")
            dependency_hashes[relative] = observed_hash
        observed_aggregate = aggregate.hexdigest()
        if observed_aggregate != raw_scope.get("aggregate_sha256"):
            raise PaperPackageError(
                f"Evaluation A aggregate mismatch for {scope_name}: {observed_aggregate}"
            )
        scope_results[str(scope_name)] = {
            "status": "PASS_UNCHANGED",
            "file_count": len(files),
            "aggregate_sha256": observed_aggregate,
        }
    return (
        {
            "status": "PASS_UNCHANGED",
            "baseline_schema_version": baseline.get("schema_version"),
            "scopes": scope_results,
        },
        dependency_hashes,
    )


def _validate_manifest_hashes(
    root: Path,
    mapping: Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise PaperPackageError(f"{label} has no artifact hash mapping")
    output: dict[str, str] = {}
    for raw_path, expected_hash in sorted(mapping.items()):
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        relative = _relative(path, root)
        observed = sha256_file(_require_file(path))
        if observed != expected_hash:
            raise PaperPackageError(f"{label} hash mismatch: {relative}")
        output[relative] = observed
    return output


def _validate_split_facts(root: Path) -> dict[str, Any]:
    a1 = _read_csv(root / "data/splits/a1_iid_split_final_v1.csv", ("split",))
    a2 = _read_csv(
        root / "data/splits/a2_jurisdiction_folds_final_v1.csv",
        ("fold_id", "role", "jurisdiction"),
    )
    if len(a1) != 1263:
        raise PaperPackageError(f"A1 split N={len(a1)}; expected 1263")
    a1_roles = ("TRAIN", "VALIDATION", "TEST", "ACTIVE_DEMO", "RESERVE_DEMO")
    a1_counts = {role: sum(row["split"] == role for row in a1) for role in a1_roles}
    if a1_counts != {
        "TRAIN": 876,
        "VALIDATION": 126,
        "TEST": 253,
        "ACTIVE_DEMO": 6,
        "RESERVE_DEMO": 2,
    }:
        raise PaperPackageError(f"A1 TEST N mismatch: {a1_counts}")
    if len(a2) != 1263 * 3:
        raise PaperPackageError(f"A2 split row count={len(a2)}; expected 3789")
    fold_test_n: dict[int, int] = {}
    heldout: set[str] = set()
    heldout_by_fold: dict[int, list[str]] = {}
    for fold in (1, 2, 3):
        test_rows = [row for row in a2 if int(row["fold_id"]) == fold and row["role"] == "TEST"]
        fold_test_n[fold] = len(test_rows)
        jurisdictions = sorted({row["jurisdiction"] for row in test_rows})
        heldout_by_fold[fold] = jurisdictions
        heldout.update(jurisdictions)
    if fold_test_n != {1: 288, 2: 287, 3: 286} or len(heldout) != 18:
        raise PaperPackageError(
            f"A2 fold/jurisdiction contract mismatch: folds={fold_test_n}, jurisdictions={len(heldout)}"
        )
    return {
        "a1_role_counts": a1_counts,
        "a2_fold_test_n": {str(key): value for key, value in fold_test_n.items()},
        "a2_heldout_jurisdictions": {
            str(key): value for key, value in heldout_by_fold.items()
        },
        "a2_pooled_test_n": sum(fold_test_n.values()),
        "a2_heldout_jurisdiction_n": len(heldout),
    }


def _validate_primary_sources(
    a1: Sequence[Mapping[str, str]],
    a2: Sequence[Mapping[str, str]],
    a3: Sequence[Mapping[str, str]],
) -> None:
    _validate_method_rows(a1, source="A1 primary", expected_n_field="test_n", expected_n=253)
    _validate_method_rows(a2, source="A2 primary", expected_n_field="test_n", expected_n=861)
    _validate_method_rows(a3, source="A3 primary", expected_n_field="n", expected_n=55)
    if any(row.get("prediction_variant") != "PRIMARY" for row in (*a1, *a2)):
        raise PaperPackageError("A1/A2 input includes a non-PRIMARY prediction variant")
    if {row.get("macro_label_count") for row in a1} != {"17"}:
        raise PaperPackageError("A1 macro-label count is not exactly 17")
    if {row.get("macro_label_count") for row in a2} != {"16"}:
        raise PaperPackageError("A2 macro-label count is not exactly 16")
    expected_zero = '["PURPOSE_REMOVAL_OF_ORGANS"]'
    if {row.get("zero_reference_support_label_ids_json") for row in a2} != {expected_zero}:
        raise PaperPackageError("A2 zero-support organ-removal rule differs from freeze")
    if {row.get("macro_supported_label_count") for row in a3} != {"17"}:
        raise PaperPackageError("A3 macro-label count is not exactly 17")
    a1_scores = (
        "macro_f1", "macro_f1_ci_lower", "macro_f1_ci_upper", "micro_f1",
        "micro_f1_ci_lower", "micro_f1_ci_upper", "exact_set_accuracy",
        "exact_set_accuracy_ci_lower", "exact_set_accuracy_ci_upper", "example_jaccard",
        "example_jaccard_ci_lower", "example_jaccard_ci_upper", "act_cpmr",
        "means_cpmr", "purpose_cpmr", "act_mean_contained_recall",
        "means_mean_contained_recall", "purpose_mean_contained_recall",
    )
    a2_scores = tuple(
        field.replace("macro_f1", "pooled_ood_macro_f1")
        .replace("micro_f1", "pooled_micro_f1")
        .replace("exact_set_accuracy", "pooled_exact_set_accuracy")
        .replace("example_jaccard", "pooled_example_jaccard")
        .replace("act_cpmr", "pooled_act_cpmr")
        .replace("means_cpmr", "pooled_means_cpmr")
        .replace("purpose_cpmr", "pooled_purpose_cpmr")
        .replace("act_mean_contained_recall", "pooled_act_mean_contained_recall")
        .replace("means_mean_contained_recall", "pooled_means_mean_contained_recall")
        .replace("purpose_mean_contained_recall", "pooled_purpose_mean_contained_recall")
        for field in a1_scores
    )
    a3_scores = (
        "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "micro_f1",
        "micro_f1_ci_low", "micro_f1_ci_high", "exact_set", "exact_set_ci_low",
        "exact_set_ci_high", "jaccard", "jaccard_ci_low", "jaccard_ci_high",
        "act_cpmr", "means_cpmr", "purpose_cpmr", "act_mean_contained_recall",
        "means_mean_contained_recall", "purpose_mean_contained_recall",
    )
    _validate_scores(a1, a1_scores, "A1 primary")
    _validate_scores(a2, a2_scores, "A2 primary")
    _validate_scores(a3, a3_scores, "A3 primary")


def load_sources(paths: PackagePaths) -> PackageSources:
    root = paths.root
    a1_path = root / "outputs/metrics/a1/amp_primary_results.csv"
    a2_path = root / "outputs/metrics/a2/amp_primary_results.csv"
    a3_path = root / "outputs/analysis/evaluation_b/eval_b_main_results.csv"
    silver_path = root / "outputs/analysis/evaluation_b/silver_vs_human_summary.csv"
    breadth_path = root / "outputs/analysis/evaluation_b/eval_b_prediction_breadth.csv"
    abstain_path = root / "outputs/analysis/evaluation_b/eval_b_abstain_results.csv"
    reference_comparison_path = (
        root / "outputs/analysis/evaluation_b/model_silver_vs_human_metric_comparison.csv"
    )
    paired_path = paths.analysis_dir / "paired_bootstrap_method_differences.csv"
    auxiliary_path = root / "outputs/analysis/evaluation_b/auxiliary_llm_human_grounded_results.csv"
    auxiliary_per_class_path = (
        root / "outputs/analysis/evaluation_b/auxiliary_llm_per_class_results.csv"
    )
    auxiliary_case_level_path = (
        root / "outputs/analysis/evaluation_b/auxiliary_llm_case_level.csv"
    )

    a1 = _read_csv(a1_path, ("method", "prediction_variant", "test_n", "reference_terminology"))
    a2 = _read_csv(a2_path, ("method", "prediction_variant", "test_n", "reference_terminology"))
    a3 = _read_csv(a3_path, ("method", "n", "reference_terminology"))
    silver = _read_csv(silver_path, SILVER_HUMAN_FIELDS)
    breadth = _read_csv(
        breadth_path,
        (
            "method", "n", "mean_predicted_act_labels", "mean_predicted_means_labels",
            "mean_predicted_purpose_labels", "mean_total_predicted_labels",
            "mean_total_human_labels",
        ),
    )
    abstain = _read_csv(
        abstain_path,
        ("method", "abstain_n", "all_amp_empty_rate", "mean_total_predicted_label_count"),
    )
    reference_comparison = _read_csv(
        reference_comparison_path,
        (
            "method", "metric_scope", "metric", "n", "human_grounded_value",
            "silver_reference_value", "delta_human_minus_silver",
        ),
    )
    paired = _read_csv(paired_path, PAIRED_REQUIRED_FIELDS)
    auxiliary = _read_csv(
        auxiliary_path,
        ("target", "metric", "value", "n", "model", "prompt_version"),
    )
    auxiliary_per_class = _read_csv(
        auxiliary_per_class_path,
        ("target", "class", "n", "support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"),
    )
    auxiliary_case_level = _read_csv(
        auxiliary_case_level_path,
        ("reliability_case_id", "search_rank", "jurisdiction", "fact_summary"),
    )
    _validate_primary_sources(a1, a2, a3)
    validate_paired_rows(paired)
    build_auxiliary_rows(auxiliary)
    _validate_auxiliary_detail_tables(auxiliary_per_class, auxiliary_case_level)

    dependency_hashes: dict[str, str] = {}
    direct_paths = (
        a1_path, a2_path, a3_path, silver_path, breadth_path, abstain_path,
        reference_comparison_path, paired_path, auxiliary_path, auxiliary_per_class_path,
        auxiliary_case_level_path,
    )
    for path in direct_paths:
        dependency_hashes[_relative(path, root)] = sha256_file(path)

    baseline_path = root / "outputs/analysis/evaluation_b/evaluation_a_integrity_baseline.json"
    baseline = _read_json(baseline_path)
    baseline_validation, baseline_hashes = validate_evaluation_a_baseline(root, baseline)
    dependency_hashes.update(baseline_hashes)
    dependency_hashes[_relative(baseline_path, root)] = sha256_file(baseline_path)

    evaluation_a_manifest_path = root / "outputs/metrics/amp_evaluation_manifest.json"
    evaluation_a_manifest = _read_json(evaluation_a_manifest_path)
    if evaluation_a_manifest.get("final_completion_gate") != "PASSED_M1_M2_M3_M4_A1_A2":
        raise PaperPackageError("Evaluation A final completion gate is not passed")
    dependency_hashes[_relative(evaluation_a_manifest_path, root)] = sha256_file(
        evaluation_a_manifest_path
    )

    evaluation_b_manifest_path = (
        root / "outputs/analysis/evaluation_b/evaluation_b_analysis_manifest.json"
    )
    evaluation_b_manifest = _read_json(evaluation_b_manifest_path)
    if evaluation_b_manifest.get("status") != "COMPLETE":
        raise PaperPackageError("Evaluation B canonical analysis is not COMPLETE")
    dependency_hashes.update(
        _validate_manifest_hashes(
            root, evaluation_b_manifest.get("inputs_sha256"), label="Evaluation B inputs"
        )
    )
    dependency_hashes.update(
        _validate_manifest_hashes(
            root, evaluation_b_manifest.get("outputs_sha256"), label="Evaluation B outputs"
        )
    )
    dependency_hashes[_relative(evaluation_b_manifest_path, root)] = sha256_file(
        evaluation_b_manifest_path
    )

    auxiliary_manifest_path = (
        root / "outputs/analysis/evaluation_b/auxiliary_llm_completion_manifest.json"
    )
    auxiliary_manifest = _read_json(auxiliary_manifest_path)
    if auxiliary_manifest.get("status") != "COMPLETE":
        raise PaperPackageError("Auxiliary LLM completion manifest is not COMPLETE")
    aux_hash_mapping = (
        auxiliary_manifest.get("artifacts_sha256")
        or auxiliary_manifest.get("outputs_sha256")
        or auxiliary_manifest.get("artifact_hashes")
    )
    auxiliary_hashes = _validate_manifest_hashes(
        root, aux_hash_mapping, label="Auxiliary LLM completion manifest"
    )
    required_aux_relatives = {
        _relative(auxiliary_path, root),
        _relative(auxiliary_per_class_path, root),
        _relative(auxiliary_case_level_path, root),
    }
    if not required_aux_relatives.issubset(auxiliary_hashes):
        raise PaperPackageError(
            "Auxiliary completion manifest does not hash all three canonical result tables"
        )
    dependency_hashes.update(auxiliary_hashes)
    dependency_hashes[_relative(auxiliary_manifest_path, root)] = sha256_file(
        auxiliary_manifest_path
    )

    parser_path = root / "logs/parser_diagnostics.json"
    parser = _read_json(parser_path)
    qc_path = root / "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
    qc = _read_json(qc_path)
    if parser.get("corpus_summary", {}).get("total_cases_parsed") != 1590:
        raise PaperPackageError("Parser corpus N differs from 1,590")
    if parser.get("corpus_summary", {}).get("fact_summary", {}).get("with_english") != 1565:
        raise PaperPackageError("Usable English Fact Summary N differs from 1,565")
    expected_qc = {
        "source_row_count": 100,
        "reviewed_n": 74,
        "skip_n": 13,
        "retained_n": 61,
        "substantive_n": 55,
        "abstain_n": 6,
    }
    if any(qc.get(key) != value for key, value in expected_qc.items()):
        raise PaperPackageError(f"Evaluation B human-reference counts differ: {qc}")
    dependency_hashes[_relative(parser_path, root)] = sha256_file(parser_path)
    dependency_hashes[_relative(qc_path, root)] = sha256_file(qc_path)

    split_facts = _validate_split_facts(root)
    ontology_path = root / "config/amp_ontology_v1.yaml"
    ontology = _read_json(ontology_path)
    family_counts = {
        family: len(ontology.get("families", {}).get(family, [])) for family in FAMILIES
    }
    if family_counts != {"ACT": 5, "MEANS": 6, "PURPOSE": 6}:
        raise PaperPackageError(f"Ontology family counts differ: {family_counts}")

    metadata_paths = {
        "ontology": ontology_path,
        "a1_split": root / "data/splits/a1_iid_split_final_v1.csv",
        "a2_split": root / "data/splits/a2_jurisdiction_folds_final_v1.csv",
        "corpus_manifest": root / "data/manifests/case_urls.csv",
        "parsed_corpus": root / "data/interim/sherloc_cases_raw.jsonl",
        "benchmark": root / "data/processed/sherloc_benchmark_v1.jsonl",
        "m1_config": root / "config/experiments/m1_tfidf_logreg_amp_v2.yaml",
        "m2_config": root / "config/experiments/m2_modernbert_amp_v2.yaml",
        "llm_config": root / "config/experiments/llm_extraction_amp_v2.yaml",
        "demo_bank": root / "config/experiments/demo_bank_amp_v1.yaml",
        "m3_prompt": root / "prompts/m3_zero_shot_amp_v2.md",
        "m4_prompt": root / "prompts/m4_six_shot_amp_v2.md",
        "m1_a1_metadata": root / "outputs/models/m1/a1/run_metadata.json",
        "m2_a1_metadata": root / "outputs/models/m2/a1/run_metadata.json",
        "m1_a3_metadata": root / "outputs/models/evaluation_b/m1/run_metadata.json",
        "m2_a3_metadata": root / "outputs/models/evaluation_b/m2/run_metadata.json",
        "m3_a3_diagnostics": root / "outputs/logs/evaluation_b/llm/m3_diagnostics.json",
        "m4_a3_diagnostics": root / "outputs/logs/evaluation_b/llm/m4_diagnostics.json",
    }
    for path in metadata_paths.values():
        _require_file(path)
        dependency_hashes[_relative(path, root)] = sha256_file(path)

    llm_config = _read_json(metadata_paths["llm_config"])
    for method, prompt_key in (("M3", "m3_prompt"), ("M4", "m4_prompt")):
        expected = llm_config["methods"][method]["prompt_sha256"]
        if sha256_file(metadata_paths[prompt_key]) != expected:
            raise PaperPackageError(f"{method} prompt hash differs from frozen LLM config")
    if llm_config.get("api_request", {}).get("store") is not False:
        raise PaperPackageError("Frozen LLM configuration does not set store=false")

    metadata = {
        "parser": parser,
        "human_qc": qc,
        "split_facts": split_facts,
        "ontology": ontology,
        "evaluation_a_manifest": evaluation_a_manifest,
        "evaluation_b_manifest": evaluation_b_manifest,
        "auxiliary_manifest": auxiliary_manifest,
        "m1_config": _read_json(metadata_paths["m1_config"]),
        "m2_config": _read_json(metadata_paths["m2_config"]),
        "llm_config": llm_config,
        "demo_bank": _read_json(metadata_paths["demo_bank"]),
        "m1_a1_metadata": _read_json(metadata_paths["m1_a1_metadata"]),
        "m2_a1_metadata": _read_json(metadata_paths["m2_a1_metadata"]),
        "m1_a3_metadata": _read_json(metadata_paths["m1_a3_metadata"]),
        "m2_a3_metadata": _read_json(metadata_paths["m2_a3_metadata"]),
        "m3_a3_diagnostics": _read_json(metadata_paths["m3_a3_diagnostics"]),
        "m4_a3_diagnostics": _read_json(metadata_paths["m4_a3_diagnostics"]),
        "artifact_paths": {key: _relative(value, root) for key, value in metadata_paths.items()},
    }
    return PackageSources(
        tuple(a1), tuple(a2), tuple(a3), tuple(silver), tuple(breadth), tuple(abstain),
        tuple(reference_comparison), tuple(paired), tuple(auxiliary),
        tuple(auxiliary_per_class), tuple(auxiliary_case_level), metadata,
        dict(sorted(dependency_hashes.items())), baseline_validation,
    )


def _artifact_hash(sources: PackageSources, key: str) -> str:
    relative = str(sources.metadata["artifact_paths"][key])
    try:
        return sources.dependency_hashes[relative]
    except KeyError as exc:
        raise PaperPackageError(f"Missing package dependency hash for {relative}") from exc


def render_methods_factsheet(sources: PackageSources) -> str:
    metadata = sources.metadata
    parser = metadata["parser"]["corpus_summary"]
    qc = metadata["human_qc"]
    split = metadata["split_facts"]
    ontology = metadata["ontology"]
    m1_run = metadata["m1_a1_metadata"]
    m2_run = metadata["m2_a1_metadata"]
    m1_a3_run = metadata["m1_a3_metadata"]
    m2_a3_run = metadata["m2_a3_metadata"]
    llm = metadata["llm_config"]
    demo = metadata["demo_bank"]
    m3 = metadata["m3_a3_diagnostics"]
    m4 = metadata["m4_a3_diagnostics"]
    aux = metadata["auxiliary_manifest"]
    paired_resamples = sorted({row["bootstrap_resamples"] for row in sources.paired})
    paired_seeds = sorted({row["seed"] for row in sources.paired})
    auxiliary_ns = {
        row["target"]: row["N"] for row in build_auxiliary_rows(sources.auxiliary)
    }

    m1_selection = m1_run["selection"]
    m2_selection = m2_run["selection"]
    m2_execution = m2_run["technical_execution_options"]
    if m1_run.get("train_n") != 884 or m2_run.get("train_n") != 884:
        raise PaperPackageError("A1 effective supervised training N differs from 884")
    if m1_a3_run.get("train_n") != 1209 or m2_a3_run.get("train_n") != 1209:
        raise PaperPackageError("A3 leakage-free supervised training N differs from 1,209")
    family_counts = {
        family: len(ontology["families"][family]) for family in FAMILIES
    }
    lines = [
        "# Paper methods factsheet",
        "",
        "Status: deterministic pre-writing factual reference. This is not manuscript prose.",
        "",
        "## Corpus and task",
        "",
        f"- Frozen SHERLOC trafficking-filter corpus: **{parser['total_cases_parsed']:,}** cases.",
        f"- Cases with a usable English Fact Summary: **{parser['fact_summary']['with_english']:,}**; without one: **{parser['fact_summary']['without_usable']:,}**.",
        "- Primary AMP cohort: **1,263** cases with complete Legacy Keywords Act/Means/Purpose fields.",
        "- Primary cohort ID: `sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9`.",
        f"- Corpus manifest SHA-256: `{_artifact_hash(sources, 'corpus_manifest')}`.",
        f"- Primary benchmark JSONL SHA-256: `{_artifact_hash(sources, 'benchmark')}`.",
        "- Unit of analysis: one English Fact Summary per case.",
        "",
        "## Frozen AMP ontology",
        "",
        f"- Act: **{family_counts['ACT']}** labels; Means: **{family_counts['MEANS']}** labels; Purpose: **{family_counts['PURPOSE']}** labels; total: **{sum(family_counts.values())}**.",
        f"- Ontology ID/version: `{ontology['ontology_id']}` / `{ontology['ontology_version']}`.",
        f"- Ontology SHA-256: `{_artifact_hash(sources, 'ontology')}`.",
        "- Primary Evaluation A reference: SHERLOC Legacy Keywords silver reference.",
        "- Evaluation A3 reference: single-reviewer human-grounded narrative reference.",
        "",
        "## Evaluation designs",
        "",
        f"- **A1 IID:** TRAIN={split['a1_role_counts']['TRAIN']}; VALIDATION={split['a1_role_counts']['VALIDATION']}; TEST={split['a1_role_counts']['TEST']}; ACTIVE_DEMO={split['a1_role_counts']['ACTIVE_DEMO']}; RESERVE_DEMO={split['a1_role_counts']['RESERVE_DEMO']}; split SHA-256 `{_artifact_hash(sources, 'a1_split')}`.",
        f"- Effective A1 supervised training N={m1_run['train_n']} (TRAIN + ACTIVE_DEMO + RESERVE_DEMO); validation N=126 and TEST N=253 remained separate.",
        f"- **A2 jurisdiction-OOD:** {split['a2_heldout_jurisdiction_n']} held-out jurisdictions; fold TEST Ns {split['a2_fold_test_n']['1']}/{split['a2_fold_test_n']['2']}/{split['a2_fold_test_n']['3']}; pooled N={split['a2_pooled_test_n']}; split SHA-256 `{_artifact_hash(sources, 'a2_split')}`.",
        f"- A2 Fold 1 held out: {', '.join(split['a2_heldout_jurisdictions']['1'])}.",
        f"- A2 Fold 2 held out: {', '.join(split['a2_heldout_jurisdictions']['2'])}.",
        f"- A2 Fold 3 held out: {', '.join(split['a2_heldout_jurisdictions']['3'])}.",
        "- A2 official Macro-F1 uses the 16 labels with positive pooled reference support. `PURPOSE_REMOVAL_OF_ORGANS` remains a prediction dimension but has zero pooled support, receives per-label N/A, and is excluded from A2 Macro-F1.",
        "- Organized Criminal Group (OCG) is an auxiliary feature and is irrelevant to A2, which evaluates AMP labels only.",
        f"- **A3 single reviewer:** source N={qc['source_row_count']}; reviewed={qc['reviewed_n']}; Skip={qc['skip_n']}; retained={qc['retained_n']}; substantive={qc['substantive_n']}; Abstain={qc['abstain_n']}.",
        "- Only one human reviewer was available for A3; no inter-annotator agreement, second-reviewer adjudication, or dual-reviewed gold reference was produced.",
        "- A3 primary AMP scores use all 55 substantive cases. The six Abstain cases are analyzed separately and are not ordinary all-negative examples.",
        "",
        "## M1-M4 fixed methods",
        "",
        "### M1: TF-IDF plus one-vs-rest logistic regression",
        "",
        f"- Vectorizer: word 1-2 grams; lowercase; no accent stripping or stop-word list; token pattern `(?u)\\b\\w\\w+\\b`; L2 normalization; IDF with smoothing; sublinear TF; `min_df={m1_selection['selected_hyperparameters']['vectorizer.min_df']}`; `max_df=1.0`; `max_features=50000`; float64.",
        f"- Classifier: one-vs-rest logistic regression; L2 penalty; liblinear solver; `C={m1_selection['selected_hyperparameters']['base_classifier.C']}`; `class_weight={m1_selection['selected_hyperparameters']['base_classifier.class_weight']}`; `max_iter=2000`; tolerance `0.0001`; random seed `20260811`; `n_jobs=1` for the wrapper.",
        f"- One global validation-selected threshold: `{m1_selection['selected_global_threshold']}`; no per-label thresholds and no TEST-label tuning.",
        f"- Frozen config SHA-256: `{_artifact_hash(sources, 'm1_config')}`; A1 run-metadata SHA-256: `{_artifact_hash(sources, 'm1_a1_metadata')}`.",
        f"- A3 uses a dedicated leakage-free fit with training N={m1_a3_run['train_n']:,} after excluding all 61 retained human-review cases; no A3 human labels were used for training or selection.",
        "",
        "### M2: ModernBERT multilabel classifier",
        "",
        f"- Model/revision: `answerdotai/ModernBERT-base` / `{metadata['m2_config']['model']['revision']}`.",
        f"- A1-selected settings: learning rate `{m2_selection['selected_hyperparameters']['learning_rate']}`; weight decay `{m2_selection['selected_hyperparameters']['weight_decay']}`; fixed/selected duration `{m2_selection['selected_best_epoch']}` epochs; one global threshold `{m2_selection['selected_global_threshold']}`; no per-label thresholds or TEST-label tuning.",
        "- Objective/optimization: unweighted `BCEWithLogitsLoss`; AdamW; linear learning-rate schedule with 0.1 warmup ratio; maximum gradient norm 1.0; training/data seed `20260811`.",
        f"- Tokenization/execution: max length `{m2_execution['max_length']}` with right truncation; Apple MPS device; physical train batch `{m2_execution['initial_train_batch_size']}`; gradient accumulation `16`; effective batch `{m2_execution['effective_train_batch_size_target']}`; BF16 autocast without gradient scaler; gradient checkpointing `{str(m2_execution['gradient_checkpointing']).lower()}` with `use_reentrant=false`; AdamW `foreach=false`; dynamic right padding rounded to multiple 64; MPS low-watermark ratio 1.0 and no high-watermark override.",
        f"- Frozen config SHA-256: `{_artifact_hash(sources, 'm2_config')}`; A1 run-metadata SHA-256: `{_artifact_hash(sources, 'm2_a1_metadata')}`.",
        f"- A3 uses the fixed six-epoch transferred protocol with training N={m2_a3_run['train_n']:,} after excluding all retained human-review cases; no A3 human labels were used for fitting, tuning, or epoch selection.",
        "",
        "### M3/M4: OpenAI Responses API AMP extraction",
        "",
        f"- Model: `{llm['api_request']['model']}`; reasoning effort `{llm['api_request']['reasoning']['effort']}`; structured-output schema SHA-256 `{llm['structured_output']['schema_sha256']}`.",
        f"- M3: zero-shot; prompt SHA-256 `{llm['methods']['M3']['prompt_sha256']}`; prompt-file SHA-256 `{_artifact_hash(sources, 'm3_prompt')}`.",
        f"- M4: six-shot; prompt SHA-256 `{llm['methods']['M4']['prompt_sha256']}`; prompt-file SHA-256 `{_artifact_hash(sources, 'm4_prompt')}`.",
        f"- Demonstration bank: `{demo['bank_id']}`; active demonstrations={len(demo['roles']['active_six'])}; file SHA-256 `{_artifact_hash(sources, 'demo_bank')}`.",
        f"- A3 completion: M3 {m3['successful_predictions']}/{m3['expected_cases']}; M4 {m4['successful_predictions']}/{m4['expected_cases']}; retained-demo overlap={len(m4.get('demo_overlap_ranks', []))}.",
        "- Each case was an independent request. `store=false`. The target payload contained only the public SHERLOC Fact Summary; no human-reference or SHERLOC silver labels were sent.",
        "- Output-token policy: 512 initially; 2,048 only after explicit `incomplete/max_output_tokens` technical failures.",
        f"- Frozen LLM config SHA-256: `{_artifact_hash(sources, 'llm_config')}`.",
        "",
        "## Metrics and uncertainty",
        "",
        "- Macro-F1: arithmetic mean of per-label F1 over the evaluation's supported-label set (17 labels in A1/A3; 16 in pooled A2).",
        "- Micro-F1: pooled true-positive, false-positive, and false-negative decisions across cases and labels.",
        "- Exact-set accuracy: proportion of cases whose complete predicted AMP set equals the reference set.",
        "- Example-based Jaccard: mean case-level intersection-over-union of predicted and reference AMP sets.",
        "- Family CPMR: proportion of cases with a nonempty predicted family set that is a subset of the reference family set. CPMR is a secondary descriptive diagnostic, not accuracy.",
        "- Mean Contained Recall: mean predicted/reference set-size ratio among CPMR-successful cases only; N/A when there are no successes.",
        "- Unpaired point-metric CIs: 1,000 case-level percentile-bootstrap resamples, seed 20260811.",
        f"- Paired method-difference CIs: {','.join(paired_resamples)} paired case-level resamples, seed {','.join(paired_seeds)}, 95% percentile intervals. `ci_excludes_zero` is descriptive; no unplanned p-value layer is added.",
        "",
        "## Auxiliary extension",
        "",
        "- Human-grounded zero-shot extension only; no supervised auxiliary baselines and no full-corpus silver auxiliary benchmark.",
        f"- Four targets over the substantive A3 source set: {', '.join(f'`{target}` N={n}' for target, n in auxiliary_ns.items())}.",
        "- Targets: Geographic Form, Victim Multiplicity, Child Involvement, and Organized Criminal Group.",
        "- `UNKNOWN` is an evaluable class where defined. `Not Applicable` and explicit non-evaluable masks are excluded target-wise.",
        f"- Auxiliary completion-manifest SHA-256: `{sources.dependency_hashes['outputs/analysis/evaluation_b/auxiliary_llm_completion_manifest.json']}`.",
        "",
        "## Data handling and privacy boundary",
        "",
        "- Model inputs were public SHERLOC English Fact Summaries. No private or nonpublic PII source was introduced.",
        "- OpenAI requests used `store=false`; human annotations and SHERLOC silver-reference labels were excluded from target payloads.",
        "- The benchmark preserves sensitive legal/trafficking narrative content locally; paper reporting should avoid unnecessary reproduction of identifying narrative detail.",
        "",
    ]
    return "\n".join(lines)


def render_claim_map() -> str:
    rows = (
        ("C01", "A1", "On the IID silver-reference test, method differences are metric-specific; LLM methods have higher Macro-F1 point estimates.", "Macro-F1, Micro-F1, Jaccard", "outputs/analysis/paper_final/main_paper_results_table.csv", "evaluation=A1; method=M1..M4", "Silver reference; paired CIs, not point ranks, govern uncertainty claims."),
        ("C02", "A2", "On pooled jurisdiction-OOD cases, LLM methods retain higher Macro-F1 point estimates while other core-score gaps are smaller.", "Macro-F1, Micro-F1, Jaccard", "outputs/analysis/paper_final/main_paper_results_table.csv", "evaluation=A2; method=M1..M4", "Pooled result spans 18 jurisdictions; do not rank individual jurisdictions."),
        ("C03", "A1 versus A2", "A1-to-A2 changes are modest and method/metric dependent rather than a uniform OOD collapse.", "A2 minus A1 deltas", "outputs/analysis/evaluation_a/a1_to_a2_distribution_shift.csv", "method=M1..M4", "Descriptive unpaired shifts; statistical significance was not tested by this table."),
        ("C04", "A3", "Human-grounded point estimates favor M3 on the core aggregate measures, with substantial small-sample uncertainty.", "Macro-F1, Micro-F1, Jaccard", "outputs/analysis/paper_final/master_results.csv", "evaluation=A3; method=M1..M4", "Single reviewer, N=55, wide bootstrap intervals; no gold-standard wording."),
        ("C05", "A1/A2/A3", "M3 versus M2 differences vary across evaluations and metrics.", "Paired core metrics and CPMR", "outputs/analysis/paper_final/paired_bootstrap_method_differences.csv", "comparison=M3 - M2", "Claim a directional difference only where its paired 95% CI excludes zero."),
        ("C06", "A3", "M4 is narrower than M3 and has higher family CPMR point estimates, but this conservatism does not improve every core score.", "Breadth; family CPMR; paired differences", "outputs/analysis/paper_final/model_behavior_summary.csv", "method in {M3,M4}", "CPMR is secondary and cannot be called accuracy; inspect paired CIs for core-score differences."),
        ("C07", "A1/A2/A3", "LLM methods more often return nonempty reference-contained family subsets than supervised baselines.", "Act/Means/Purpose CPMR", "outputs/analysis/paper_final/master_results.csv", "all evaluations; *_cpmr", "Secondary diagnostic introduced by addendum; does not require complete recall."),
        ("C08", "A3", "Supervised methods produce broader AMP label sets than M3/M4 on the human-grounded cases.", "Mean labels per case", "outputs/analysis/paper_final/model_behavior_summary.csv", "method=M1..M4", "Breadth is descriptive, not intrinsically better or worse."),
        ("C09", "A3 dual reference", "SHERLOC silver and narrative-grounded human labels overlap substantially but each contains nonshared labels.", "Exact concordance, Jaccard, shared/silver-only/human-only", "outputs/analysis/paper_final/silver_vs_human_compact.csv", "family=ACT,MEANS,PURPOSE", "Silver-only labels are not automatically errors; SHERLOC metadata can be broader than the Fact Summary."),
        ("C10", "A3 dual reference", "Act labels show a material narrative-versus-structured-reference divergence requiring source-aware interpretation.", "Act shared and nonshared counts/rates", "outputs/analysis/paper_final/silver_vs_human_compact.csv", "family=ACT", "One case lacks comparable silver Act/Means structure; use family-specific comparable N."),
        ("C11", "A3 Abstain", "M3/M4 returned no AMP labels for all six narrative-insufficiency cases, unlike M1/M2.", "All-AMP-empty rate; mean predicted labels", "outputs/analysis/paper_final/model_behavior_summary.csv", "abstain_*; method=M1..M4", "Descriptive insufficiency diagnostic; M1/M2 were not trained with an explicit abstention mechanism."),
        ("C12", "A3 auxiliary", "The zero-shot auxiliary extension provides target-specific human-grounded results for four features.", "Target-appropriate accuracy/F1/Jaccard", "outputs/analysis/paper_final/auxiliary_extension_compact.csv", "target=all four", "No supervised baseline and no full-corpus silver auxiliary benchmark; keep secondary."),
        ("C13", "A1/A2/A3", "Paired case resampling quantifies uncertainty in selected M3/M4-versus-M2 method differences.", "Point difference and paired 95% CI", "outputs/analysis/paper_final/paired_bootstrap_method_differences.csv", "all 63 rows", "Descriptive bootstrap intervals; no multiplicity-adjusted hypothesis testing or p-values."),
    )
    header = (
        "# Paper claim-to-evidence map\n\n"
        "Every proposed claim is intentionally cautious. A3 uses a single-reviewer human-grounded narrative reference. Exact numeric values must be copied from the cited canonical row, never from rounded prose, and no unplanned p-value layer may be inferred from paired confidence intervals.\n\n"
        "| ID | Evaluation | Cautious claim | Metric | Canonical artifact | Row selector | Required caution |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body = "\n".join(
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in rows
    )
    return header + body + "\n"


def render_nllp_plan() -> str:
    return """# NLLP paper artifact plan

Status: planning scaffold only. This file deliberately contains no manuscript prose.

## Format constraint

- Target venue format: ACL/NLLP-style two-column paper.
- Main-text budget: 8 pages excluding references and permitted appendices.
- Keep numerical claims linked to `docs/paper_claim_to_evidence_map.md`.

## Eight-page allocation

| Main page budget | Section function | Primary artifact allocation |
|---|---|---|
| 0.75 | Problem framing and contributions | No table; define narrative extraction and the silver/human distinction |
| 0.75 | Related work and task positioning | No new result artifact |
| 1.00 | Corpus, ontology, and reference construction | Facts from `paper_methods_factsheet.md`; compact corpus/ontology text |
| 1.25 | M1-M4 methods and leakage controls | Method factsheet; no result table |
| 0.75 | A1/A2/A3 evaluation design and metrics | Split facts; CPMR definition; bootstrap protocol |
| 1.50 | Evaluation A results | `main_paper_results_table.csv`; Figure PF1; selective paired-CI statements |
| 1.25 | Human-grounded Evaluation A3 | `silver_vs_human_compact.csv`; Figures PF2/PF3; Abstain diagnostic |
| 0.75 | Auxiliary extension, limitations, and conclusion | `auxiliary_extension_compact.csv` only if space; single-reviewer and scope limitations |

## Main-paper artifacts

- `outputs/analysis/paper_final/main_paper_results_table.csv`
- `outputs/analysis/paper_final/silver_vs_human_compact.csv`
- `outputs/analysis/paper_final/model_behavior_summary.csv` (selected rows/metrics only)
- `outputs/analysis/paper_final/auxiliary_extension_compact.csv` (secondary compact result)
- Figure PF1: core A1/A2/A3 performance
- Figure PF2: Act/Means/Purpose CPMR
- Figure PF3: silver/human mismatch and reference-score shifts

Use no more than three paper-final figures. If space requires one removal, move PF2 to the appendix before removing PF1 or PF3.

## Appendix/supplement artifacts

- Full `master_results.csv` with CIs and Mean Contained Recall
- Full `paired_bootstrap_method_differences.csv` (63 rows)
- A1/A2 fold, jurisdiction, per-label, and rare-label tables
- A3 family/per-label, prediction-breadth, case-level, and Abstain case-level tables
- Full silver-versus-human case/per-label tables
- Auxiliary per-class and case-level tables
- Prompt, schema, demo-bank, execution, and cost/provenance details
- Reproducibility and pre-writing freeze manifests

## Claim discipline

- Use paired-CI language only for comparisons represented in the paired table.
- Do not convert `ci_excludes_zero` into an unplanned p-value or global significance claim.
- Keep CPMR explicitly secondary and descriptive.
- Use family-specific comparable N for silver-versus-human analyses.
- Describe A3 as a single-reviewer human-grounded narrative reference, not adjudicated gold.
- Describe Abstain outcomes as narrative-insufficiency diagnostics.
- Keep the auxiliary extension secondary: four zero-shot targets, no supervised baselines, no full-corpus silver benchmark.
- Do not claim that silver-only labels are erroneous.

## Pre-submission artifact checks

1. Validate `prewriting_freeze_manifest.json` against all cited files.
2. Generate every manuscript number from the frozen CSVs.
3. Cross-check every result sentence against the 13-row claim map.
4. Ensure figure captions identify reference type and N.
5. Confirm no Evaluation A/B canonical artifact changed after the pre-writing freeze.
6. Confirm public-data/privacy wording and `store=false` provenance.
"""


def render_readme() -> str:
    return """# Paper-final analysis package

This directory is a deterministic presentation layer over frozen canonical results. It does not recompute benchmark metrics and does not modify Evaluation A or Evaluation B artifacts.

## Tables

- `master_results.csv`: all A1/A2/A3 × M1-M4 core metrics, confidence intervals, family CPMR, and Mean Contained Recall.
- `main_paper_results_table.csv`: compact 12-row main-paper table.
- `silver_vs_human_compact.csv`: family-level dual-reference comparison with family-specific comparable N.
- `model_behavior_summary.csv`: A3 breadth, human-reference performance, CPMR, and Abstain behavior.
- `auxiliary_extension_compact.csv`: target-appropriate zero-shot auxiliary metrics.
- `paired_bootstrap_method_differences.csv`: separately generated canonical paired method-difference intervals; it is a required dependency and is never rewritten by the package builder.

## Figures

- `figure_pf1_core_performance.svg`: main paper; A1/A2/A3 Macro-F1, Micro-F1, and Jaccard.
- `figure_pf2_cpmr_by_family.svg`: main paper if space permits; otherwise appendix. CPMR is secondary.
- `figure_pf3_silver_human_reference_shift.svg`: main paper; silver/human family mismatch and A3 dual-reference score shifts.

## Main paper versus appendix

Use the compact main table, PF1, PF3, and selected PF2 panels in the main paper. Keep the full master table, all 63 paired-bootstrap rows, detailed family/per-label tables, case-level rows, and provenance manifests in the appendix or supplement.

## Reproduction and integrity

Run:

```bash
python src/experiments/25_build_paper_final_package.py --preflight
python src/experiments/25_build_paper_final_package.py --write
python src/experiments/25_build_paper_final_package.py --check
```

The builder validates every upstream dependency, Evaluation A's unchanged baseline, Evaluation B's analysis manifest, the auxiliary completion manifest, and the paired table before rendering. It refuses to overwrite any differing existing package file. `prewriting_freeze_manifest.json` is written last and hashes every generated artifact except itself.
"""


def render_figure_readme() -> str:
    return """# Paper-final figures

These three deterministic SVGs are presentation-only views of the frozen paper-final tables. They do not recompute benchmark metrics. Captions and manuscript text must identify the evaluation, reference type, and N from the cited table.

| Figure | Contents | Canonical table source | Recommended placement |
|---|---|---|---|
| `figure_pf1_core_performance.svg` | Macro-F1, Micro-F1, and example-based Jaccard for M1-M4 across A1, pooled A2, and A3 | `outputs/analysis/paper_final/master_results.csv` | Main results section; retain in the main paper |
| `figure_pf2_cpmr_by_family.svg` | Act, Means, and Purpose CPMR for M1-M4 across A1, pooled A2, and A3 | `outputs/analysis/paper_final/master_results.csv` | Secondary-behavior section; move to appendix first if space is tight |
| `figure_pf3_silver_human_reference_shift.svg` | Family-level silver/human mismatch and the change in A3 scores under human versus silver references | `outputs/analysis/paper_final/silver_vs_human_compact.csv` and the frozen Evaluation B dual-reference comparison | Human-grounded evaluation section; retain in the main paper |

## Reporting boundaries

- Use PF1 for core performance and PF3 for reference-source effects; CPMR in PF2 is a secondary descriptive diagnostic, not accuracy.
- A1/A2 use the SHERLOC Legacy Keywords silver reference. A3 uses a single-reviewer human-grounded narrative reference.
- Family-specific comparable N varies in the silver-versus-human panel; do not imply that silver-only labels are errors.
- Keep the package to these three paper-final figures. Detailed per-label, fold, jurisdiction, paired-bootstrap, and auxiliary plots belong in the appendix or supplement if later needed.
"""


def _figure_bytes(master: Sequence[Mapping[str, str]], silver: Sequence[Mapping[str, str]], reference_comparison: Sequence[Mapping[str, str]]) -> dict[str, bytes]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise PaperPackageError("matplotlib and numpy are required to render paper figures") from exc

    matplotlib.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "sherloc-paper-final-v1",
            "svg.fonttype": "none",
        }
    )
    colors = {"M1": "#4C78A8", "M2": "#F58518", "M3": "#54A24B", "M4": "#E45756"}

    def save(fig: Any) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": None, "Creator": "SHERLOC paper-final package v1"},
        )
        plt.close(fig)
        return buffer.getvalue()

    indexed = {(row["evaluation"], row["method"]): row for row in master}
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharey=True)
    for axis, evaluation in zip(axes, EVALUATIONS, strict=True):
        for field, label, marker in (
            ("macro_f1", "Macro-F1", "o"),
            ("micro_f1", "Micro-F1", "s"),
            ("example_jaccard", "Jaccard", "^"),
        ):
            axis.plot(
                x,
                [float(indexed[(evaluation, method)][field]) for method in METHODS],
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, 1)
        axis.set_title(f"{evaluation} (N={indexed[(evaluation, 'M1')]['N']})")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Core AMP performance across evaluation designs")
    fig.tight_layout()
    pf1 = save(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharey=True)
    for axis, evaluation in zip(axes, EVALUATIONS, strict=True):
        for family, marker in (("act", "o"), ("means", "s"), ("purpose", "^")):
            axis.plot(
                x,
                [float(indexed[(evaluation, method)][f"{family}_cpmr"]) for method in METHODS],
                marker=marker,
                linewidth=1.5,
                label=family.title(),
            )
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, 1)
        axis.set_title(f"{evaluation} (N={indexed[(evaluation, 'M1')]['N']})")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Contained Partial Match Rate")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Reference-contained predictions by AMP family")
    fig.tight_layout()
    pf2 = save(fig)

    silver_by_family = {row["family"].upper(): row for row in silver}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    family_x = np.arange(3)
    width = 0.36
    axes[0].bar(
        family_x - width / 2,
        [float(silver_by_family[family]["silver_only_rate_of_silver_labels"]) for family in FAMILIES],
        width,
        label="Silver-only / silver labels",
        color="#9ECAE1",
    )
    axes[0].bar(
        family_x + width / 2,
        [float(silver_by_family[family]["human_only_rate_of_human_labels"]) for family in FAMILIES],
        width,
        label="Human-only / human labels",
        color="#FB6A4A",
    )
    axes[0].set_xticks(family_x, [family.title() for family in FAMILIES])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Family mismatch (comparable N varies)")
    axes[0].legend(frameon=False, fontsize=7.5)

    comparison = {
        (row["method"], row["metric"]): row
        for row in reference_comparison
        if row["metric_scope"] == "OVERALL" and row["metric"] in {"macro_f1", "micro_f1", "jaccard"}
    }
    for metric, marker in (("macro_f1", "o"), ("micro_f1", "s"), ("jaccard", "^")):
        axes[1].plot(
            x,
            [float(comparison[(method, metric)]["delta_human_minus_silver"]) for method in METHODS],
            marker=marker,
            linewidth=1.5,
            label=metric.replace("_", " ").title(),
        )
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_xticks(x, METHODS)
    axes[1].set_ylabel("Human-grounded minus silver score")
    axes[1].set_title("A3 dual-reference cases (N=54)")
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Silver-reference versus narrative-grounded human reference")
    fig.tight_layout()
    pf3 = save(fig)
    return {FIGURE_NAMES[0]: pf1, FIGURE_NAMES[1]: pf2, FIGURE_NAMES[2]: pf3}


def build_package_artifacts(
    paths: PackagePaths, sources: PackageSources
) -> dict[Path, bytes]:
    master = build_master_rows(sources.a1, sources.a2, sources.a3)
    main = build_main_paper_rows(master)
    silver = build_silver_human_rows(sources.silver_human)
    behavior = build_behavior_rows(sources.a3, sources.a3_breadth, sources.a3_abstain)
    auxiliary = build_auxiliary_rows(sources.auxiliary)
    figures = _figure_bytes(master, silver, sources.reference_comparison)

    artifacts: dict[Path, bytes] = {
        paths.analysis_dir / "master_results.csv": _csv_bytes(master, MASTER_FIELDS),
        paths.analysis_dir / "main_paper_results_table.csv": _csv_bytes(
            main, MAIN_PAPER_FIELDS
        ),
        paths.analysis_dir / "silver_vs_human_compact.csv": _csv_bytes(
            silver, SILVER_HUMAN_FIELDS
        ),
        paths.analysis_dir / "model_behavior_summary.csv": _csv_bytes(
            behavior, BEHAVIOR_FIELDS
        ),
        paths.analysis_dir / "auxiliary_extension_compact.csv": _csv_bytes(
            auxiliary, AUXILIARY_FIELDS
        ),
        paths.analysis_dir / "README.md": render_readme().encode("utf-8"),
        paths.figure_dir / "README.md": render_figure_readme().encode("utf-8"),
        paths.docs_dir / "paper_methods_factsheet.md": render_methods_factsheet(
            sources
        ).encode("utf-8"),
        paths.docs_dir / "paper_claim_to_evidence_map.md": render_claim_map().encode(
            "utf-8"
        ),
        paths.docs_dir / "nllp_paper_artifact_plan.md": render_nllp_plan().encode(
            "utf-8"
        ),
    }
    for name, payload in figures.items():
        artifacts[paths.figure_dir / name] = payload
    if len(figures) != 3:
        raise PaperPackageError(f"Paper-final figure count={len(figures)}; expected exactly 3")

    generated_hashes = {
        _relative(path, paths.root): sha256_bytes(payload)
        for path, payload in sorted(artifacts.items(), key=lambda item: str(item[0]))
    }
    required_frozen_generated = {
        "outputs/analysis/paper_final/master_results.csv",
        "docs/paper_methods_factsheet.md",
        "docs/paper_claim_to_evidence_map.md",
        "docs/nllp_paper_artifact_plan.md",
    }
    if not required_frozen_generated.issubset(generated_hashes):
        raise PaperPackageError("Pre-writing freeze is missing required generated artifacts")

    manifest = {
        "schema_version": "sherloc-paper-prewriting-freeze-v1",
        "generator": "src/experiments/25_build_paper_final_package.py",
        "generator_version": VERSION,
        "status": "COMPLETE",
        "deterministic": True,
        "canonical_metrics_recomputed": False,
        "upstream_artifacts_modified": False,
        "evaluation_a_unchanged_baseline_validation": sources.baseline_validation,
        "evaluation_b_analysis_status": sources.metadata["evaluation_b_manifest"].get(
            "status"
        ),
        "paired_bootstrap": {
            "path": "outputs/analysis/paper_final/paired_bootstrap_method_differences.csv",
            "row_count": len(sources.paired),
            "resamples": sorted(
                {int(row["bootstrap_resamples"]) for row in sources.paired}
            ),
            "seed": sorted({int(row["seed"]) for row in sources.paired}),
        },
        "auxiliary_extension": {
            "completion_manifest_path": "outputs/analysis/evaluation_b/auxiliary_llm_completion_manifest.json",
            "target_count": len(AUXILIARY_METRICS),
            "supervised_baseline_run": False,
            "full_corpus_silver_benchmark_run": False,
        },
        "dependency_sha256": dict(sorted(sources.dependency_hashes.items())),
        "generated_artifacts_sha256": dict(sorted(generated_hashes.items())),
        "freeze_boundaries": {
            "evaluation_a_reference": "SHERLOC silver reference",
            "evaluation_b_reference": "single-reviewer human-grounded narrative reference",
            "inter_annotator_analysis_performed": False,
            "auxiliary_scope": "HUMAN_GROUNDED_ZERO_SHOT_ONLY",
            "api_or_model_execution_performed_by_generator": False,
        },
    }
    manifest_path = paths.analysis_dir / "prewriting_freeze_manifest.json"
    artifacts[manifest_path] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return artifacts


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def inspect_targets(artifacts: Mapping[Path, bytes]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for path, payload in sorted(artifacts.items(), key=lambda item: str(item[0])):
        if not path.exists():
            status = "MISSING"
            observed = None
        elif not path.is_file():
            status = "CONFLICT_NOT_FILE"
            observed = None
        else:
            observed = sha256_file(path)
            status = "UNCHANGED" if path.read_bytes() == payload else "CONFLICT_DIFFERENT"
        diagnostics.append(
            {
                "path": str(path),
                "status": status,
                "expected_sha256": sha256_bytes(payload),
                "observed_sha256": observed,
                "byte_count": len(payload),
            }
        )
    return diagnostics


def write_package(artifacts: Mapping[Path, bytes]) -> list[dict[str, Any]]:
    before = inspect_targets(artifacts)
    conflicts = [row for row in before if row["status"].startswith("CONFLICT")]
    if conflicts:
        raise PaperPackageError(
            "Refusing to overwrite differing existing paper-final artifacts: "
            + ", ".join(row["path"] for row in conflicts)
        )
    manifest_paths = [
        path for path in artifacts if path.name == "prewriting_freeze_manifest.json"
    ]
    if len(manifest_paths) != 1:
        raise PaperPackageError("Package must contain exactly one pre-writing freeze manifest")
    manifest_path = manifest_paths[0]
    ordered = sorted(
        (path for path in artifacts if path != manifest_path), key=lambda value: str(value)
    ) + [manifest_path]
    for path in ordered:
        if not path.exists():
            _atomic_write_new(path, artifacts[path])
    after = inspect_targets(artifacts)
    if any(row["status"] != "UNCHANGED" for row in after):
        raise PaperPackageError("Paper-final artifact verification failed after write")
    return after


def check_package(artifacts: Mapping[Path, bytes]) -> list[dict[str, Any]]:
    diagnostics = inspect_targets(artifacts)
    failures = [row for row in diagnostics if row["status"] != "UNCHANGED"]
    if failures:
        raise PaperPackageError(
            "Paper-final package is missing or stale: "
            + ", ".join(row["path"] for row in failures)
        )
    return diagnostics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate dependencies and render expected bytes without writing.",
    )
    mode.add_argument(
        "--write", action="store_true", help="Write only missing paper-final artifacts."
    )
    mode.add_argument(
        "--check", action="store_true", help="Require every paper-final artifact to match."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--docs-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = PackagePaths.for_root(
        args.repo_root,
        analysis_dir=args.analysis_dir,
        figure_dir=args.figure_dir,
        docs_dir=args.docs_dir,
    )
    try:
        sources = load_sources(paths)
        artifacts = build_package_artifacts(paths, sources)
        if args.write:
            diagnostics = write_package(artifacts)
            status = "WRITTEN_OR_UNCHANGED"
        elif args.check:
            diagnostics = check_package(artifacts)
            status = "VERIFIED_UNCHANGED"
        else:
            diagnostics = inspect_targets(artifacts)
            status = "PREFLIGHT_PASS_NO_WRITES"
    except PaperPackageError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": status,
                "generator_version": VERSION,
                "dependency_count": len(sources.dependency_hashes),
                "artifact_count": len(artifacts),
                "artifacts": diagnostics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
