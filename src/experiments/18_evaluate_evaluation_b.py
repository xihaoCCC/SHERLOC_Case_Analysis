#!/usr/bin/env python3
"""Canonical single-reviewer Evaluation B analysis and reporting.

This script consumes already-frozen human-reference, silver-reference,
leakage-audit, and M1--M4 prediction artifacts.  It never trains a model,
selects a case, changes a human annotation, or calls an external API.  All
primary model comparisons use one exact common substantive case membership.
Structurally absent Legacy Keyword families remain unavailable rather than
being reinterpreted as negative labels; dual-reference comparisons use only
cases with a complete silver AMP reference.

The source human annotation is intentionally not read here: construction and
QC of ``human_grounded_reference_v1.csv`` are a separate, upstream stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Keep the plotting cache inside the permitted temporary area on machines
# whose user-level Matplotlib directory is unavailable.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sherloc_eval_b_mplconfig")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # package imports
    from .evaluation_b import AMP_ID_BY_RAW_LABEL, AMP_RAW_LABEL_BY_ID, parse_amp_labels
    from .metrics import (
        AMP_FAMILIES,
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        compute_amp_cpmr,
        compute_amp_metrics,
        labels_to_indicator,
    )
except ImportError:  # direct script/test execution
    try:
        from src.experiments.evaluation_b import (
            AMP_ID_BY_RAW_LABEL,
            AMP_RAW_LABEL_BY_ID,
            parse_amp_labels,
        )
        from src.experiments.metrics import (
            AMP_FAMILIES,
            AMP_FAMILY_BY_LABEL,
            AMP_LABEL_IDS,
            compute_amp_cpmr,
            compute_amp_metrics,
            labels_to_indicator,
        )
    except ImportError:
        from evaluation_b import AMP_ID_BY_RAW_LABEL, AMP_RAW_LABEL_BY_ID, parse_amp_labels
        from metrics import (
            AMP_FAMILIES,
            AMP_FAMILY_BY_LABEL,
            AMP_LABEL_IDS,
            compute_amp_cpmr,
            compute_amp_metrics,
            labels_to_indicator,
        )


VERSION = "1.1.0"
METHODS = ("M1", "M2", "M3", "M4")
FAMILIES = ("ACT", "MEANS", "PURPOSE")
FAMILY_PLURAL = {"ACT": "acts", "MEANS": "means", "PURPOSE": "purposes"}
FAMILY_LABEL_IDS = {
    family: tuple(label for label in AMP_LABEL_IDS if AMP_FAMILY_BY_LABEL[label] == family)
    for family in FAMILIES
}
METHOD_COLORS = {"M1": "#4C78A8", "M2": "#72B7B2", "M3": "#F58518", "M4": "#E45756"}

BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20260811
VALID_PREDICTION_STATUSES = {
    "SUCCESS",
    "VALIDATED",
    "SUCCESS_VALIDATED",
    "COMPLETE",
}
REFERENCE_TERM = "single-reviewer human-grounded narrative reference"
SILVER_TERM = "SHERLOC Legacy Keywords silver reference"
FINAL_REPORT_SENTENCE = (
    "Evaluation B is complete as a single-reviewer human-grounded narrative "
    "validation. The substantive human-reference subset and "
    "narrative-insufficiency subset were evaluated separately; no inter-annotator "
    "or adjudication analysis was performed, no retained human case entered "
    "supervised M1/M2 training, Evaluation A remained unchanged, and no A4 "
    "auxiliary model benchmark was run."
)
NOT_APPLICABLE_AUXILIARY_VALUES = {
    "Not Applicable",
    "NOT_APPLICABLE",
    "NOT_APPLICABLE_OUTSIDE_PRIMARY_COHORT",
}
EXPECTED_LLM_PROMPT_SHA256 = {
    "M3": "00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85",
    "M4": "2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37",
}
EXPECTED_LLM_SCHEMA_SHA256 = (
    "d106c4ab1aa5bfcf34a6accd4f8c77df0bd21436cb0761d7828b21d9d87f46da"
)
EXPECTED_SUPERVISED_CONFIG_SHA256 = {
    "M1": "5c6a916af3781305926b0cd57bde77e30f7c094a035a313cda95fc391a4046a5",
    "M2": "5a83104cda51b8674ab577ea133be991ebccab99a30915e3fc307b219a64ed7b",
}
EXPECTED_DEMO_BANK_SHA256 = (
    "1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b"
)
EXPECTED_M4_DEMO_MEMBERSHIP_SHA256 = (
    "0e98d6196d4b7e1a3f15c81186a37e61e00ec34828f4b6bfb7c1398323f02eba"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_REFERENCE = REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
DEFAULT_REFERENCE_KEY = REPO_ROOT / "data/annotations/reliability_sample_100_reference_key.csv"
DEFAULT_MANAGEMENT_SAMPLE = REPO_ROOT / "data/annotations/reliability_sample_100.csv"
DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.csv"
DEFAULT_LEAKAGE_AUDIT = REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_training_exclusion_audit.csv"
DEFAULT_DEMO_BANK = REPO_ROOT / "config/experiments/demo_bank_amp_v1.yaml"
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "outputs/analysis/evaluation_b/human_annotation_source_manifest.json"
DEFAULT_QC_SUMMARY = REPO_ROOT / "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
DEFAULT_MEMBERSHIP_MANIFEST = (
    REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_membership_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/analysis/evaluation_b"
DEFAULT_FIGURE_DIR = REPO_ROOT / "outputs/figures/evaluation_b"
DEFAULT_REPORT = REPO_ROOT / "docs/evaluation_b_human_grounded_report.md"
DEFAULT_EVAL_A_BASELINE = (
    REPO_ROOT / "outputs/analysis/evaluation_b/evaluation_a_integrity_baseline.json"
)
DEFAULT_M1_METADATA = REPO_ROOT / "outputs/models/evaluation_b/m1/run_metadata.json"
DEFAULT_M2_METADATA = REPO_ROOT / "outputs/models/evaluation_b/m2/run_metadata.json"
DEFAULT_M3_DIAGNOSTICS = REPO_ROOT / "outputs/logs/evaluation_b/llm/m3_diagnostics.json"
DEFAULT_M4_DIAGNOSTICS = REPO_ROOT / "outputs/logs/evaluation_b/llm/m4_diagnostics.json"

DEFAULT_PREDICTIONS = {
    "M1": REPO_ROOT / "outputs/predictions/evaluation_b/m1/predictions.jsonl",
    "M2": REPO_ROOT / "outputs/predictions/evaluation_b/m2/predictions.jsonl",
    "M3": REPO_ROOT / "outputs/predictions/evaluation_b/m3/eval_b_predictions.jsonl",
    "M4": REPO_ROOT / "outputs/predictions/evaluation_b/m4/eval_b_predictions.jsonl",
}

EVAL_A_INTEGRITY_SCOPES = (
    "outputs/metrics",
    "outputs/analysis/evaluation_a",
    "outputs/figures/evaluation_a",
    "outputs/predictions/m1",
    "outputs/predictions/m2",
    "outputs/predictions/m3",
    "outputs/predictions/m4",
    "outputs/models/m1",
    "outputs/models/m2",
)

OUTPUT_TABLE_NAMES = (
    "silver_vs_human_summary.csv",
    "silver_vs_human_per_label.csv",
    "silver_vs_human_case_level.csv",
    "auxiliary_silver_vs_human_summary.csv",
    "eval_b_main_results.csv",
    "eval_b_bootstrap_cis.csv",
    "eval_b_family_results.csv",
    "eval_b_per_label_results.csv",
    "eval_b_abstain_results.csv",
    "eval_b_abstain_case_level.csv",
    "eval_b_prediction_breadth.csv",
    "model_silver_vs_human_metric_comparison.csv",
    "human_grounded_case_level_errors.csv",
)

FIGURE_NAMES = (
    "figure_b1_human_grounded_core_performance.svg",
    "figure_b2_human_grounded_cpmr.svg",
    "figure_b3_silver_vs_human_model_scores.svg",
    "figure_b4_silver_human_label_proportions.svg",
)


class EvaluationBAnalysisError(RuntimeError):
    """Raised when an Evaluation B artifact is incomplete or inconsistent."""


@dataclass(frozen=True)
class HumanCase:
    reliability_case_id: str
    search_rank: int
    canonical_url: str
    input_sha256: str
    jurisdiction: str
    fact_summary: str
    review_status: str
    substantive_amp_evaluable: bool
    auxiliary_evaluable: bool
    annotation_notes: str
    labels: Mapping[str, tuple[str, ...]]
    geographic_form: tuple[str, ...]
    organized_criminal_group: bool
    organized_criminal_group_evaluable: bool
    multiplicity: str
    child: str


@dataclass(frozen=True)
class SilverCase:
    reliability_case_id: str
    search_rank: int
    labels: Mapping[str, tuple[str, ...]]
    family_available: Mapping[str, bool]
    geographic_form: tuple[str, ...]
    organized_criminal_group: bool | None
    multiplicity: str | None
    child: str | None
    primary_amp_cohort_member: bool


@dataclass(frozen=True)
class PredictionCase:
    method: str
    reliability_case_id: str
    search_rank: int
    status: str
    labels: tuple[str, ...]

    @property
    def by_family(self) -> dict[str, tuple[str, ...]]:
        return {
            family: tuple(
                label for label in self.labels if AMP_FAMILY_BY_LABEL[label] == family
            )
            for family in FAMILIES
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _evaluation_a_integrity_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Hash every file in the frozen Evaluation A scopes, without modifying it."""

    scopes: dict[str, Any] = {}
    for relative_root in EVAL_A_INTEGRITY_SCOPES:
        root = repo_root / relative_root
        if not root.is_dir():
            raise EvaluationBAnalysisError(
                f"Evaluation A integrity scope is missing: {root}"
            )
        files: list[dict[str, Any]] = []
        aggregate = hashlib.sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(repo_root).as_posix()
            digest = _sha256_file(path)
            size = path.stat().st_size
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(str(size).encode("ascii"))
            aggregate.update(b"\n")
            files.append({"path": relative, "sha256": digest, "size": size})
        scopes[relative_root] = {
            "file_count": len(files),
            "aggregate_sha256": aggregate.hexdigest(),
            "files": files,
        }
    return {
        "schema_version": "sherloc-evaluation-a-integrity-baseline-v1",
        "algorithm": "sorted repo-relative path + NUL + sha256 + NUL + decimal size + newline",
        "scopes": scopes,
    }


def capture_evaluation_a_integrity_baseline(path: Path) -> dict[str, Any]:
    """Create the pre-Evaluation-B baseline once; never update it silently."""

    snapshot = _evaluation_a_integrity_snapshot()
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise EvaluationBAnalysisError(
                f"Evaluation A baseline already exists with different content: {path}"
            )
        return snapshot
    _atomic_text(path, payload)
    return snapshot


def validate_evaluation_a_integrity(path: Path) -> dict[str, Any]:
    baseline = _read_json(path)
    observed = _evaluation_a_integrity_snapshot()
    if baseline != observed:
        expected_scopes = baseline.get("scopes", {})
        observed_scopes = observed.get("scopes", {})
        drift = [
            scope
            for scope in EVAL_A_INTEGRITY_SCOPES
            if expected_scopes.get(scope) != observed_scopes.get(scope)
        ]
        raise EvaluationBAnalysisError(
            f"Evaluation A integrity drift detected in scopes: {drift}"
        )
    return observed


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvaluationBAnalysisError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EvaluationBAnalysisError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationBAnalysisError(f"Required prediction JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvaluationBAnalysisError(
                f"Malformed JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvaluationBAnalysisError(f"Prediction row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationBAnalysisError(f"Required JSON does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationBAnalysisError(f"Malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationBAnalysisError(f"JSON root must be an object: {path}")
    return value


def _column(row: Mapping[str, Any], canonical: str, *aliases: str) -> Any:
    """Resolve one centralized schema alias and reject ambiguous disagreement."""

    candidates = [name for name in (canonical, *aliases) if name in row]
    if not candidates:
        raise EvaluationBAnalysisError(f"Missing required column: {canonical}")
    values = [row[name] for name in candidates]
    normalized = [str(value).strip() for value in values]
    if len(set(normalized)) > 1:
        raise EvaluationBAnalysisError(
            f"Conflicting values across aliases for {canonical}: {candidates}"
        )
    return values[0]


def _strict_int(value: Any, *, field: str) -> int:
    text = str(value).strip()
    try:
        result = int(text)
    except (TypeError, ValueError) as exc:
        raise EvaluationBAnalysisError(f"Invalid integer for {field}: {value!r}") from exc
    if str(result) != text and text not in {f"+{result}", f"{result}.0"}:
        raise EvaluationBAnalysisError(f"Invalid integer syntax for {field}: {value!r}")
    return result


def _strict_bool(value: Any, *, field: str) -> bool:
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    text = str(value).strip()
    if text in {"1", "TRUE", "True", "true"}:
        return True
    if text in {"0", "FALSE", "False", "false"}:
        return False
    raise EvaluationBAnalysisError(f"Invalid Boolean for {field}: {value!r}")


def _optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    return _strict_bool(value, field=field)


def _list_value(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        raise EvaluationBAnalysisError(f"Missing list value for {field}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationBAnalysisError(f"Malformed JSON list for {field}: {value!r}") from exc
    if not isinstance(parsed, list):
        raise EvaluationBAnalysisError(f"Expected JSON list for {field}: {value!r}")
    return parsed


def _amp_labels(value: Any, family: str, *, field: str) -> tuple[str, ...]:
    parsed, errors = parse_amp_labels(value, family)
    if errors:
        raise EvaluationBAnalysisError(f"Invalid {field}: {'; '.join(errors)}")
    if len(parsed) != len(set(parsed)):
        raise EvaluationBAnalysisError(f"Duplicate labels in {field}")
    return tuple(parsed)


def _form_labels(value: Any, *, field: str) -> tuple[str, ...]:
    values = _list_value(value, field=field)
    mapping = {
        "Internal": "Internal",
        "INTERNAL": "Internal",
        "Transnational": "Transnational",
        "TRANSNATIONAL": "Transnational",
    }
    output: list[str] = []
    for item in values:
        text = str(item).strip()
        if text not in mapping:
            raise EvaluationBAnalysisError(f"Invalid Geographic Form in {field}: {item!r}")
        normalized = mapping[text]
        if normalized in output:
            raise EvaluationBAnalysisError(f"Duplicate Geographic Form in {field}: {normalized}")
        output.append(normalized)
    return tuple(label for label in ("Internal", "Transnational") if label in output)


def _silver_form_labels(value: Any, *, field: str) -> tuple[str, ...]:
    """Separate geographic form from the legacy OCG marker without conflation."""

    values = _list_value(value, field=field)
    unexpected = {
        str(item).strip()
        for item in values
        if str(item).strip()
        not in {"Internal", "Transnational", "Organized Criminal Group"}
    }
    if unexpected:
        raise EvaluationBAnalysisError(
            f"Invalid legacy Geographic Form value in {field}: {sorted(unexpected)}"
        )
    return _form_labels(
        [item for item in values if str(item).strip() != "Organized Criminal Group"],
        field=field,
    )


def _silver_auxiliary_value(value: Any) -> str | None:
    """Normalize the frozen outside-cohort sentinel to a non-comparable value."""

    text = "" if value is None else str(value).strip()
    if not text or text in NOT_APPLICABLE_AUXILIARY_VALUES:
        return None
    return text


def _unique_index(rows: Iterable[Any], key, *, artifact: str) -> dict[Any, Any]:
    output: dict[Any, Any] = {}
    for row in rows:
        identity = key(row)
        if identity in output:
            raise EvaluationBAnalysisError(f"Duplicate identity {identity!r} in {artifact}")
        output[identity] = row
    return output


def load_human_reference(path: Path) -> dict[str, HumanCase]:
    rows = _read_csv(path)
    cases: list[HumanCase] = []
    for row in rows:
        case_id = str(_column(row, "reliability_case_id")).strip()
        if not case_id:
            raise EvaluationBAnalysisError("Human reference contains a blank reliability_case_id")
        rank = _strict_int(_column(row, "search_rank"), field=f"{case_id}.search_rank")
        status = str(_column(row, "review_status")).strip().upper()
        if status not in {"SUBSTANTIVE", "ABSTAIN"}:
            raise EvaluationBAnalysisError(
                f"Retained human reference has invalid review_status for {case_id}: {status!r}"
            )
        substantive = _strict_bool(
            _column(row, "substantive_amp_evaluable"),
            field=f"{case_id}.substantive_amp_evaluable",
        )
        if substantive != (status == "SUBSTANTIVE"):
            raise EvaluationBAnalysisError(
                f"review_status/substantive_amp_evaluable mismatch for {case_id}"
            )
        auxiliary_evaluable = _strict_bool(
            _column(row, "auxiliary_evaluable"),
            field=f"{case_id}.auxiliary_evaluable",
        )
        labels = {
            "ACT": _amp_labels(
                _column(row, "acts_human_clean_json"),
                "ACT",
                field=f"{case_id}.acts_human_clean_json",
            ),
            "MEANS": _amp_labels(
                _column(row, "means_human_clean_json"),
                "MEANS",
                field=f"{case_id}.means_human_clean_json",
            ),
            "PURPOSE": _amp_labels(
                _column(row, "purposes_human_clean_json", "purpose_human_clean_json"),
                "PURPOSE",
                field=f"{case_id}.purpose_human_clean_json",
            ),
        }
        if status == "ABSTAIN" and any(labels.values()):
            raise EvaluationBAnalysisError(
                f"ABSTAIN case {case_id} has nonempty cleaned AMP labels"
            )
        geographic_form = _form_labels(
            _column(row, "geographic_form_human_clean_json"),
            field=f"{case_id}.geographic_form_human_clean_json",
        )
        ocg = _strict_bool(
            _column(row, "organized_criminal_group_human"),
            field=f"{case_id}.organized_criminal_group_human",
        )
        ocg_evaluable = _strict_bool(
            _column(row, "organized_criminal_group_evaluable"),
            field=f"{case_id}.organized_criminal_group_evaluable",
        )
        if status == "ABSTAIN" and ocg_evaluable:
            raise EvaluationBAnalysisError(
                f"ABSTAIN case {case_id} cannot be OCG-evaluable"
            )
        multiplicity = str(_column(row, "multiplicity_human_clean")).strip()
        child = str(_column(row, "child_human_clean")).strip()
        if multiplicity not in {"SINGLE", "MULTIPLE", "UNKNOWN", "Not Applicable"}:
            raise EvaluationBAnalysisError(
                f"Invalid cleaned multiplicity for {case_id}: {multiplicity!r}"
            )
        if child not in {"TRUE", "FALSE", "UNKNOWN", "Not Applicable"}:
            raise EvaluationBAnalysisError(f"Invalid cleaned child value for {case_id}: {child!r}")
        fact_summary = str(_column(row, "fact_summary", "english_fact_summary_raw"))
        canonical_url = str(_column(row, "canonical_url")).strip()
        input_sha256 = str(_column(row, "input_sha256")).strip()
        if not canonical_url:
            raise EvaluationBAnalysisError(f"Human reference has a blank canonical_url for {case_id}")
        if input_sha256 != _sha256_text(fact_summary):
            raise EvaluationBAnalysisError(
                f"Human reference input_sha256 does not match Fact Summary for {case_id}"
            )
        cases.append(
            HumanCase(
                reliability_case_id=case_id,
                search_rank=rank,
                canonical_url=canonical_url,
                input_sha256=input_sha256,
                jurisdiction=str(_column(row, "jurisdiction", "jurisdiction_raw")).strip(),
                fact_summary=fact_summary,
                review_status=status,
                substantive_amp_evaluable=substantive,
                auxiliary_evaluable=auxiliary_evaluable,
                annotation_notes=str(row.get("annotation_notes", "")),
                labels=labels,
                geographic_form=geographic_form,
                organized_criminal_group=ocg,
                organized_criminal_group_evaluable=ocg_evaluable,
                multiplicity=multiplicity,
                child=child,
            )
        )
    if not cases:
        raise EvaluationBAnalysisError("Human reference contains no retained cases")
    by_id = _unique_index(cases, lambda item: item.reliability_case_id, artifact="human reference")
    _unique_index(cases, lambda item: item.search_rank, artifact="human reference search_rank")
    return by_id


def _raw_silver_labels(
    row: Mapping[str, Any], family: str, *, case_id: str
) -> tuple[tuple[str, ...], bool]:
    field = {
        "ACT": "legacy_acts_raw_json",
        "MEANS": "legacy_means_raw_json",
        "PURPOSE": "legacy_purposes_raw_json",
    }[family]
    raw_values = _list_value(_column(row, field), field=f"{case_id}.{field}")
    return (
        _amp_labels(raw_values, family, field=f"{case_id}.{field}"),
        bool(raw_values),
    )


def load_silver_reference(
    reference_key_path: Path,
    management_path: Path,
    benchmark_path: Path,
    human_cases: Mapping[str, HumanCase],
) -> dict[str, SilverCase]:
    key_rows = _read_csv(reference_key_path)
    management_rows = _read_csv(management_path)
    benchmark_rows = _read_csv(benchmark_path)
    key_by_id = _unique_index(
        key_rows,
        lambda row: str(_column(row, "reliability_case_id")).strip(),
        artifact="reliability reference key",
    )
    management_by_id = _unique_index(
        management_rows,
        lambda row: str(_column(row, "reliability_case_id")).strip(),
        artifact="reliability management sample",
    )
    benchmark_by_rank = _unique_index(
        benchmark_rows,
        lambda row: _strict_int(_column(row, "search_rank"), field="benchmark.search_rank"),
        artifact="benchmark-v1",
    )
    output: dict[str, SilverCase] = {}
    for case_id, human in human_cases.items():
        if case_id not in key_by_id or case_id not in management_by_id:
            raise EvaluationBAnalysisError(
                f"Retained case {case_id} is missing from reference key or management sample"
            )
        key_row = key_by_id[case_id]
        management = management_by_id[case_id]
        key_rank = _strict_int(_column(key_row, "search_rank"), field=f"{case_id}.key_rank")
        management_rank = _strict_int(
            _column(management, "search_rank"), field=f"{case_id}.management_rank"
        )
        if key_rank != human.search_rank or management_rank != human.search_rank:
            raise EvaluationBAnalysisError(f"search_rank mismatch for {case_id}")
        management_text = str(_column(management, "english_fact_summary_raw"))
        if management_text != human.fact_summary:
            raise EvaluationBAnalysisError(f"Fact Summary mismatch for retained case {case_id}")
        parsed_silver = {
            family: _raw_silver_labels(key_row, family, case_id=case_id)
            for family in FAMILIES
        }
        labels = {family: parsed_silver[family][0] for family in FAMILIES}
        family_available = {
            family: parsed_silver[family][1] for family in FAMILIES
        }
        primary = _strict_bool(
            _column(key_row, "primary_amp_cohort_member"),
            field=f"{case_id}.primary_amp_cohort_member",
        )
        if primary != all(family_available.values()):
            raise EvaluationBAnalysisError(
                f"Primary-cohort/Legacy AMP availability mismatch for {case_id}"
            )
        if primary:
            benchmark = benchmark_by_rank.get(human.search_rank)
            if benchmark is None:
                raise EvaluationBAnalysisError(
                    f"Primary-cohort retained case {case_id} is absent from benchmark-v1"
                )
            benchmark_fields = {
                "ACT": "act_ontology_ids_json",
                "MEANS": "means_ontology_ids_json",
                "PURPOSE": "purpose_ontology_ids_json",
            }
            for family, field in benchmark_fields.items():
                benchmark_labels = _amp_labels(
                    _column(benchmark, field), family, field=f"benchmark.{human.search_rank}.{field}"
                )
                if benchmark_labels != labels[family]:
                    raise EvaluationBAnalysisError(
                        f"Reference-key/benchmark Legacy AMP mismatch for {case_id} {family}"
                    )
        form_raw = key_row.get("legacy_form_raw_json", "")
        geographic_form = (
            _silver_form_labels(form_raw, field=f"{case_id}.legacy_form_raw_json")
            if str(form_raw).strip()
            else ()
        )
        output[case_id] = SilverCase(
            reliability_case_id=case_id,
            search_rank=human.search_rank,
            labels=labels,
            family_available=family_available,
            geographic_form=geographic_form,
            organized_criminal_group=_optional_bool(
                key_row.get("legacy_ocg_present"), field=f"{case_id}.legacy_ocg_present"
            ),
            multiplicity=_silver_auxiliary_value(
                key_row.get("multiplicity_provisional", "")
            ),
            child=_silver_auxiliary_value(key_row.get("child_strict_label", "")),
            primary_amp_cohort_member=primary,
        )
    return output


def _prediction_labels(row: Mapping[str, Any], *, method: str, case_id: str) -> tuple[str, ...]:
    if "predicted_labels" not in row:
        raise EvaluationBAnalysisError(f"{method} {case_id} lacks predicted_labels")
    raw = row["predicted_labels"]
    labels: list[str] = []
    if isinstance(raw, Mapping):
        normalized_keys = {str(key).lower(): value for key, value in raw.items()}
        expected_keys = {"acts", "means", "purposes"}
        if set(normalized_keys) != expected_keys:
            raise EvaluationBAnalysisError(
                f"{method} {case_id} nested predicted_labels must contain acts/means/purposes exactly"
            )
        for family in FAMILIES:
            key = FAMILY_PLURAL[family]
            family_values = _list_value(
                normalized_keys[key], field=f"{method}.{case_id}.predicted_labels.{key}"
            )
            family_labels = _amp_labels(
                family_values,
                family,
                field=f"{method}.{case_id}.predicted_labels.{key}",
            )
            labels.extend(family_labels)
    else:
        values = _list_value(raw, field=f"{method}.{case_id}.predicted_labels")
        labels = [str(value).strip() for value in values]
        unknown = set(labels) - set(AMP_LABEL_IDS)
        if unknown:
            raise EvaluationBAnalysisError(
                f"{method} {case_id} has unknown predicted labels: {sorted(unknown)}"
            )
    if len(labels) != len(set(labels)):
        raise EvaluationBAnalysisError(f"{method} {case_id} has duplicate predicted labels")
    return tuple(label for label in AMP_LABEL_IDS if label in labels)


def load_predictions(
    method: str,
    path: Path,
    human_cases: Mapping[str, HumanCase],
    expected_case_ids: set[str],
    *,
    expected_membership_sha256: str,
    expected_m4_demo_bank_id: str | None = None,
    expected_m4_demo_membership_sha256: str | None = None,
) -> dict[str, PredictionCase]:
    if method not in METHODS:
        raise EvaluationBAnalysisError(f"Unknown method: {method}")
    rows = _read_jsonl(path) if path.suffix.lower() == ".jsonl" else _read_csv(path)
    predictions: list[PredictionCase] = []
    human_by_rank = {case.search_rank: case for case in human_cases.values()}
    for row in rows:
        row_method = str(row.get("method", row.get("method_id", ""))).strip().upper()
        if row_method != method:
            raise EvaluationBAnalysisError(
                f"Prediction method mismatch in {path}: expected {method}, found {row_method!r}"
            )
        case_id = str(row.get("reliability_case_id", "")).strip()
        rank_value = row.get("search_rank", "")
        rank = _strict_int(rank_value, field=f"{method}.search_rank")
        if not case_id:
            human = human_by_rank.get(rank)
            if human is None:
                raise EvaluationBAnalysisError(
                    f"{method} prediction rank {rank} cannot be aligned to retained reference"
                )
            case_id = human.reliability_case_id
        if case_id not in human_cases:
            raise EvaluationBAnalysisError(f"{method} has prediction for non-retained case {case_id}")
        if human_cases[case_id].search_rank != rank:
            raise EvaluationBAnalysisError(f"{method} identity mismatch for {case_id}")
        human = human_cases[case_id]
        if str(row.get("evaluation", "")).strip().upper() != "B":
            raise EvaluationBAnalysisError(f"{method} {case_id} evaluation provenance mismatch")
        if str(row.get("retained_membership_sha256", "")).strip() != expected_membership_sha256:
            raise EvaluationBAnalysisError(
                f"{method} {case_id} retained-membership provenance mismatch"
            )
        if str(row.get("canonical_url", "")) != human.canonical_url:
            raise EvaluationBAnalysisError(f"{method} {case_id} canonical_url mismatch")
        if str(row.get("fact_summary", "")) != human.fact_summary:
            raise EvaluationBAnalysisError(f"{method} {case_id} Fact Summary mismatch")
        if str(row.get("input_sha256", "")).strip() != human.input_sha256:
            raise EvaluationBAnalysisError(f"{method} {case_id} input_sha256 mismatch")
        if method in {"M1", "M2"}:
            if str(row.get("config_sha256", "")) != EXPECTED_SUPERVISED_CONFIG_SHA256[method]:
                raise EvaluationBAnalysisError(f"{method} {case_id} config hash mismatch")
            if row.get("human_labels_used_for_training_tuning_or_prediction") is not False:
                raise EvaluationBAnalysisError(
                    f"{method} {case_id} lacks a false human-label-use provenance flag"
                )
        else:
            if row.get("human_or_silver_labels_sent_to_model") is not False:
                raise EvaluationBAnalysisError(
                    f"{method} {case_id} lacks a false prompt-label-leakage flag"
                )
            if row.get("store") is not False:
                raise EvaluationBAnalysisError(f"{method} {case_id} does not preserve store=false")
            if str(row.get("prompt_sha256", "")) != EXPECTED_LLM_PROMPT_SHA256[method]:
                raise EvaluationBAnalysisError(f"{method} {case_id} prompt hash mismatch")
            if str(row.get("schema_sha256", "")) != EXPECTED_LLM_SCHEMA_SHA256:
                raise EvaluationBAnalysisError(f"{method} {case_id} schema hash mismatch")
            requested_model = str(row.get("effective_requested_model_id", ""))
            if not (
                requested_model == "gpt-5.6-luna"
                or requested_model.startswith("gpt-5.6-luna-")
            ):
                raise EvaluationBAnalysisError(f"{method} {case_id} model provenance mismatch")
            for digest_field in (
                "request_sha256",
                "builder_payload_sha256",
                "builder_metadata_sha256",
            ):
                digest = str(row.get(digest_field, ""))
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise EvaluationBAnalysisError(
                        f"{method} {case_id} has malformed {digest_field}"
                    )
            issued = row.get("api_request_issued_for_evaluation_b")
            reuse_status = str(row.get("reuse_status", ""))
            if not (
                (issued is True and reuse_status == "NEW_EVALUATION_B_REQUEST")
                or (
                    issued is False
                    and reuse_status == "REUSED_IDENTICAL_FROZEN_REQUEST"
                )
            ):
                raise EvaluationBAnalysisError(
                    f"{method} {case_id} has inconsistent request-reuse provenance"
                )
        if method == "M4":
            if str(row.get("demo_bank_id", "")) != str(expected_m4_demo_bank_id or ""):
                raise EvaluationBAnalysisError(f"M4 {case_id} demo-bank ID mismatch")
            if str(row.get("demo_bank_membership_sha256", "")) != str(
                expected_m4_demo_membership_sha256 or ""
            ):
                raise EvaluationBAnalysisError(
                    f"M4 {case_id} demo-bank membership mismatch"
                )
        status = str(row.get("status", "")).strip().upper()
        if status not in VALID_PREDICTION_STATUSES:
            raise EvaluationBAnalysisError(
                f"{method} prediction {case_id} has nonvalidated status {status!r}"
            )
        predictions.append(
            PredictionCase(
                method=method,
                reliability_case_id=case_id,
                search_rank=rank,
                status=status,
                labels=_prediction_labels(row, method=method, case_id=case_id),
            )
        )
    by_id = _unique_index(
        predictions, lambda item: item.reliability_case_id, artifact=f"{method} predictions"
    )
    observed = set(by_id)
    if observed != expected_case_ids:
        raise EvaluationBAnalysisError(
            f"{method} prediction membership mismatch: missing={sorted(expected_case_ids-observed)}, "
            f"extra={sorted(observed-expected_case_ids)}"
        )
    return by_id


def load_demo_overlap(demo_bank_path: Path, human_cases: Mapping[str, HumanCase]) -> set[str]:
    if _sha256_file(demo_bank_path) != EXPECTED_DEMO_BANK_SHA256:
        raise EvaluationBAnalysisError("Frozen demo-bank file SHA-256 has changed")
    config = _read_json(demo_bank_path)
    active = config.get("roles", {}).get("active_six")
    if not isinstance(active, list) or len(active) != 6:
        raise EvaluationBAnalysisError("Frozen demo bank does not contain exactly six active ranks")
    active_ranks = {_strict_int(value, field="demo_bank.active_six") for value in active}
    return {
        case_id for case_id, case in human_cases.items() if case.search_rank in active_ranks
    }


def load_m4_demo_bank_provenance(demo_bank_path: Path) -> tuple[str, str]:
    if _sha256_file(demo_bank_path) != EXPECTED_DEMO_BANK_SHA256:
        raise EvaluationBAnalysisError("Frozen demo-bank file SHA-256 has changed")
    config = _read_json(demo_bank_path)
    bank = config.get("evaluation_banks", {}).get("A1", {})
    if not isinstance(bank, Mapping):
        raise EvaluationBAnalysisError("Frozen demo bank lacks Evaluation-B A1 metadata")
    membership_sha256 = str(bank.get("membership_sha256", "")).strip()
    if membership_sha256 != EXPECTED_M4_DEMO_MEMBERSHIP_SHA256:
        raise EvaluationBAnalysisError("Frozen A1 demo-bank membership hash has changed")
    return "A1", membership_sha256


def validate_leakage_audit(
    path: Path,
    retained_case_ids: set[str],
    *,
    expected_membership_sha256: str,
) -> None:
    rows = _read_csv(path)
    by_id = _unique_index(
        rows,
        lambda row: str(_column(row, "reliability_case_id")).strip(),
        artifact="Evaluation B training exclusion audit",
    )
    if set(by_id) != retained_case_ids:
        raise EvaluationBAnalysisError(
            "Training exclusion audit membership does not equal retained human reference"
        )
    for case_id, row in by_id.items():
        for exclusion_field in (
            "removed_from_eval_b_supervised_training",
            "removed_from_eval_b_validation",
            "removed_from_eval_b_threshold_tuning",
            "removed_from_eval_b_supervised_label_selection",
        ):
            removed = _strict_bool(
                _column(row, exclusion_field),
                field=f"{case_id}.{exclusion_field}",
            )
            if not removed:
                raise EvaluationBAnalysisError(
                    f"Retained case {case_id} was not excluded by {exclusion_field}"
                )
        if str(row.get("retained_membership_sha256", row.get("membership_sha256", ""))).strip() != expected_membership_sha256:
            raise EvaluationBAnalysisError(
                f"Retained case {case_id} has stale leakage-audit membership provenance"
            )
        for optional in ("included_in_m1_training", "included_in_m2_training"):
            if optional in row and str(row[optional]).strip() and _strict_bool(
                row[optional], field=f"{case_id}.{optional}"
            ):
                raise EvaluationBAnalysisError(
                    f"Retained case {case_id} is marked included in supervised training"
                )


def _status_complete(value: Any, *, artifact: str) -> None:
    status = str(value).strip().upper()
    if status not in {"COMPLETE", "SUCCESS", "SUCCESS_VALIDATED", "PASS"}:
        raise EvaluationBAnalysisError(f"{artifact} is not complete: status={status!r}")


def retained_membership_sha256(human_cases: Mapping[str, HumanCase]) -> str:
    """Reproduce the five-field canonical digest frozen by the reference builder."""

    payload = [
        {
            "reliability_case_id": case.reliability_case_id,
            "search_rank": case.search_rank,
            "canonical_url": case.canonical_url,
            "input_sha256": case.input_sha256,
            "review_status": case.review_status,
        }
        for case in sorted(human_cases.values(), key=lambda item: item.search_rank)
    ]
    return _sha256_text(_canonical_json(payload))


def _resolve_frozen_path(value: Any, *, field: str) -> Path:
    if not str(value or "").strip():
        raise EvaluationBAnalysisError(f"Frozen membership manifest lacks {field}")
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_frozen_file(
    entry: Mapping[str, Any],
    actual_path: Path,
    *,
    artifact: str,
) -> None:
    frozen_path = _resolve_frozen_path(entry.get("path"), field=f"{artifact}.path")
    if frozen_path.resolve() != actual_path.resolve():
        raise EvaluationBAnalysisError(f"{artifact} path does not match the frozen manifest")
    frozen_sha = str(entry.get("sha256", "")).strip()
    if not frozen_sha or _sha256_file(actual_path) != frozen_sha:
        raise EvaluationBAnalysisError(f"{artifact} SHA-256 does not match the frozen manifest")


def validate_human_reference_provenance(
    source_manifest: Mapping[str, Any],
    qc_summary: Mapping[str, Any],
    human_cases: Mapping[str, HumanCase],
    *,
    human_reference_path: Path,
    source_manifest_path: Path,
    qc_summary_path: Path,
    membership_manifest: Mapping[str, Any],
) -> Path:
    """Validate the immutable source, QC, reference, and frozen membership chain."""

    if (
        str(membership_manifest.get("status", ""))
        != "FROZEN_FOR_EVALUATION_B_PRE_MODEL_INFERENCE"
    ):
        raise EvaluationBAnalysisError("Evaluation B membership manifest is not frozen")
    if membership_manifest.get("single_reviewer_design") is not True:
        raise EvaluationBAnalysisError("Membership manifest does not preserve single-reviewer design")
    if membership_manifest.get("inter_annotator_statistics_permitted") is not False:
        raise EvaluationBAnalysisError("Membership manifest permits unsupported inter-annotator statistics")

    reference_entry = membership_manifest.get("human_reference", {})
    membership_entry = membership_manifest.get("membership", {})
    source_manifest_entry = membership_manifest.get("source_manifest", {})
    qc_entry = membership_manifest.get("qc", {})
    if not all(
        isinstance(entry, Mapping)
        for entry in (reference_entry, membership_entry, source_manifest_entry, qc_entry)
    ):
        raise EvaluationBAnalysisError("Membership manifest has malformed artifact bindings")
    _validate_frozen_file(
        reference_entry,
        human_reference_path,
        artifact="human reference",
    )
    if str(membership_manifest.get("human_reference_sha256", "")) != str(
        reference_entry.get("sha256", "")
    ):
        raise EvaluationBAnalysisError("Human-reference hashes disagree inside the freeze")
    _validate_frozen_file(
        source_manifest_entry,
        source_manifest_path,
        artifact="annotation source manifest",
    )
    _validate_frozen_file(
        {
            "path": qc_entry.get("summary_path"),
            "sha256": qc_entry.get("summary_sha256"),
        },
        qc_summary_path,
        artifact="human annotation QC summary",
    )
    membership_path = _resolve_frozen_path(
        membership_entry.get("path"), field="membership.path"
    )
    if not membership_path.is_file():
        raise EvaluationBAnalysisError(f"Frozen membership CSV is missing: {membership_path}")
    if _sha256_file(membership_path) != str(membership_entry.get("sha256", "")).strip():
        raise EvaluationBAnalysisError("Frozen membership CSV SHA-256 has changed")
    if str(membership_manifest.get("membership_sha256", "")) != str(
        membership_entry.get("sha256", "")
    ):
        raise EvaluationBAnalysisError("Membership-CSV hashes disagree inside the freeze")

    observed_digest = retained_membership_sha256(human_cases)
    frozen_digests = {
        str(membership_manifest.get("retained_membership_sha256", "")).strip(),
        str(membership_entry.get("retained_membership_sha256", "")).strip(),
    }
    if frozen_digests != {observed_digest}:
        raise EvaluationBAnalysisError(
            "Retained human-reference membership does not match the frozen digest"
        )
    frozen_members = membership_manifest.get("retained_members")
    if not isinstance(frozen_members, list):
        raise EvaluationBAnalysisError("Membership manifest lacks retained_members")
    expected_members = [
        {
            "reliability_case_id": case.reliability_case_id,
            "search_rank": case.search_rank,
            "canonical_url": case.canonical_url,
            "input_sha256": case.input_sha256,
            "review_status": case.review_status,
            "substantive_amp_evaluable": case.substantive_amp_evaluable,
        }
        for case in sorted(human_cases.values(), key=lambda item: item.search_rank)
    ]
    normalized_frozen_members: list[dict[str, Any]] = []
    for index, member in enumerate(frozen_members):
        if not isinstance(member, Mapping):
            raise EvaluationBAnalysisError(f"retained_members[{index}] is not an object")
        normalized_frozen_members.append(
            {
                "reliability_case_id": str(member.get("reliability_case_id", "")),
                "search_rank": _strict_int(
                    member.get("search_rank"), field=f"retained_members[{index}].search_rank"
                ),
                "canonical_url": str(member.get("canonical_url", "")),
                "input_sha256": str(member.get("input_sha256", "")),
                "review_status": str(member.get("review_status", "")),
                "substantive_amp_evaluable": _strict_bool(
                    member.get("substantive_amp_evaluable"),
                    field=f"retained_members[{index}].substantive_amp_evaluable",
                ),
            }
        )
    normalized_frozen_members.sort(key=lambda item: int(item["search_rank"]))
    if normalized_frozen_members != expected_members:
        raise EvaluationBAnalysisError(
            "Retained human-reference identities differ from frozen retained_members"
        )

    source = source_manifest.get("source", {})
    if not isinstance(source, Mapping):
        raise EvaluationBAnalysisError("Annotation source manifest lacks a source object")
    source_path_value = source.get("path", source_manifest.get("source_path"))
    if not source_path_value:
        raise EvaluationBAnalysisError("Annotation source manifest lacks a source path")
    source_path = Path(str(source_path_value))
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    if not source_path.is_file():
        raise EvaluationBAnalysisError(f"Immutable annotation source is missing: {source_path}")
    expected_sha = str(source.get("sha256", source_manifest.get("sha256", ""))).strip()
    if not expected_sha or _sha256_file(source_path) != expected_sha:
        raise EvaluationBAnalysisError("Immutable annotation source SHA-256 has changed")
    _status_complete(qc_summary.get("status"), artifact="human annotation QC")
    if int(qc_summary.get("blocking_issue_count", -1)) != 0:
        raise EvaluationBAnalysisError("Human annotation QC reports blocking issues")
    count_checks = {
        "retained_n": len(human_cases),
        "substantive_n": sum(
            case.review_status == "SUBSTANTIVE" for case in human_cases.values()
        ),
        "abstain_n": sum(case.review_status == "ABSTAIN" for case in human_cases.values()),
    }
    for key, expected in count_checks.items():
        if int(qc_summary.get(key, -1)) != expected:
            raise EvaluationBAnalysisError(
                f"Human annotation QC {key} does not match the retained reference"
            )
        manifest_value = membership_manifest.get(
            key, membership_manifest.get("counts", {}).get(key, -1)
        )
        if int(manifest_value) != expected:
            raise EvaluationBAnalysisError(
                f"Frozen membership manifest {key} does not match the retained reference"
            )
    if str(qc_summary.get("source_sha256", "")).strip() != expected_sha:
        raise EvaluationBAnalysisError("QC/source-manifest annotation hashes disagree")
    return source_path


def validate_execution_metadata(
    m1_path: Path,
    m2_path: Path,
    m3_path: Path,
    m4_path: Path,
    *,
    retained_n: int,
    expected_m4_n: int,
    demo_overlap_ids: set[str],
    demo_overlap_ranks: set[int] | None = None,
    prediction_paths: Mapping[str, Path],
    expected_membership_sha256: str,
    expected_m4_demo_bank_id: str,
) -> dict[str, dict[str, Any]]:
    """Validate artifact-backed completion claims before reporting results."""

    m1 = _read_json(m1_path)
    m2 = _read_json(m2_path)
    m3 = _read_json(m3_path)
    m4 = _read_json(m4_path)
    if set(prediction_paths) != set(METHODS):
        raise EvaluationBAnalysisError("Prediction-file bindings must cover M1 through M4")
    for method, value in (("M1", m1), ("M2", m2)):
        _status_complete(value.get("status"), artifact=f"{method} run metadata")
        observed_method = str(value.get("method_id", value.get("method", ""))).upper()
        if observed_method != method:
            raise EvaluationBAnalysisError(f"{method} run metadata method mismatch")
        if int(value.get("retained_n", -1)) != retained_n:
            raise EvaluationBAnalysisError(f"{method} retained_n does not match reference")
        if int(value.get("prediction_n", -1)) != retained_n:
            raise EvaluationBAnalysisError(f"{method} prediction_n does not match retained N")
        if _strict_bool(
            value.get("human_labels_used_for_training_tuning_or_prediction", ""),
            field=f"{method}.human_labels_used_for_training_tuning_or_prediction",
        ):
            raise EvaluationBAnalysisError(f"{method} metadata reports human-label use")
        if int(value.get("train_n", 0)) <= 0:
            raise EvaluationBAnalysisError(f"{method} train_n is missing or nonpositive")
        if str(value.get("config_sha256", "")) != EXPECTED_SUPERVISED_CONFIG_SHA256[method]:
            raise EvaluationBAnalysisError(f"{method} config hash does not match the freeze")
        prediction_path = prediction_paths[method]
        if not prediction_path.is_file() or str(value.get("prediction_sha256", "")).strip() != _sha256_file(prediction_path):
            raise EvaluationBAnalysisError(
                f"{method} metadata prediction hash does not match the evaluated file"
            )
        recorded_path = _resolve_frozen_path(
            value.get("prediction_path"), field=f"{method}.prediction_path"
        )
        if recorded_path.resolve() != prediction_path.resolve():
            raise EvaluationBAnalysisError(
                f"{method} metadata prediction path does not match the evaluated file"
            )
    m1_hyper = m1.get("fixed_hyperparameters", {})
    if not isinstance(m1_hyper, Mapping) or not (
        int(m1_hyper.get("min_df", -1)) == 2
        and math.isclose(float(m1_hyper.get("C", -1)), 1.0)
        and str(m1_hyper.get("class_weight", "")).lower() in {"none", "null"}
        and math.isclose(float(m1_hyper.get("global_threshold", -1)), 0.25)
    ):
        raise EvaluationBAnalysisError("M1 fixed hyperparameters do not match the protocol")
    m2_hyper = m2.get("fixed_hyperparameters", {})
    m2_technical = m2.get("technical_execution", {})
    m2_max_length = m2_hyper.get(
        "max_length",
        m2_technical.get("max_length") if isinstance(m2_technical, Mapping) else None,
    )
    m2_threshold = m2_hyper.get(
        "global_threshold", m2_hyper.get("threshold", m2_hyper.get("selected_threshold"))
    )
    if not isinstance(m2_hyper, Mapping) or not (
        math.isclose(float(m2_hyper.get("learning_rate", -1)), 3e-5)
        and math.isclose(float(m2_hyper.get("weight_decay", -1)), 0.01)
        and int(m2_hyper.get("epochs", m2_hyper.get("fixed_epochs", -1))) == 6
        and math.isclose(float(m2_threshold if m2_threshold is not None else -1), 0.20)
        and int(m2_max_length if m2_max_length is not None else -1) == 2048
    ):
        raise EvaluationBAnalysisError("M2 fixed hyperparameters do not match the protocol")
    expected_m2_technical = {
        "physical_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "effective_train_batch_size": 16,
        "gradient_checkpointing": True,
        "mixed_precision_dtype": "bfloat16",
        "gradient_scaler_enabled": False,
        "adamw_foreach": False,
        "pad_to_multiple_of": 64,
    }
    if not isinstance(m2_technical, Mapping) or any(
        m2_technical.get(key) != expected
        for key, expected in expected_m2_technical.items()
    ):
        raise EvaluationBAnalysisError(
            "M2 technical execution does not match the fixed MPS protocol"
        )
    for flag in ("validation_run", "threshold_search"):
        if _strict_bool(m2.get(flag, ""), field=f"M2.{flag}"):
            raise EvaluationBAnalysisError(f"M2 metadata reports prohibited {flag}")

    for method, value, expected in (("M3", m3, retained_n), ("M4", m4, expected_m4_n)):
        _status_complete(value.get("status"), artifact=f"{method} diagnostics")
        if str(value.get("method", value.get("method_id", ""))).upper() != method:
            raise EvaluationBAnalysisError(f"{method} diagnostics method mismatch")
        if str(value.get("evaluation", "")).upper() != "B":
            raise EvaluationBAnalysisError(f"{method} diagnostics evaluation mismatch")
        if int(value.get("expected_cases", -1)) != expected:
            raise EvaluationBAnalysisError(f"{method} diagnostics expected_cases mismatch")
        if int(value.get("successful_predictions", -1)) != expected:
            raise EvaluationBAnalysisError(f"{method} diagnostics prediction count mismatch")
        if int(value.get("unresolved_failures", -1)) != 0:
            raise EvaluationBAnalysisError(f"{method} has unresolved failures")
        if int(value.get("missing_unattempted", -1)) != 0:
            raise EvaluationBAnalysisError(f"{method} has missing unattempted cases")
        new_requests = int(value.get("new_api_request_successes", -1))
        reused_requests = int(value.get("reused_identical_requests", -1))
        if new_requests < 0 or reused_requests < 0 or new_requests + reused_requests != expected:
            raise EvaluationBAnalysisError(
                f"{method} diagnostics request-reuse counts do not match predictions"
            )
        if value.get("store") is not False:
            raise EvaluationBAnalysisError(f"{method} diagnostics do not preserve store=false")
        if value.get("human_or_silver_labels_sent_to_model") is not False:
            raise EvaluationBAnalysisError(f"{method} diagnostics report label leakage")
        if value.get("prompt_sha256") != EXPECTED_LLM_PROMPT_SHA256[method]:
            raise EvaluationBAnalysisError(f"{method} prompt hash does not match the freeze")
        if value.get("schema_sha256") != EXPECTED_LLM_SCHEMA_SHA256:
            raise EvaluationBAnalysisError(f"{method} schema hash does not match the freeze")
        model = str(value.get("model", ""))
        if not (model == "gpt-5.6-luna" or model.startswith("gpt-5.6-luna-")):
            raise EvaluationBAnalysisError(f"{method} model does not match the frozen alias")
        prediction_path = prediction_paths[method]
        if not prediction_path.is_file() or str(value.get("prediction_file_sha256", "")).strip() != _sha256_file(prediction_path):
            raise EvaluationBAnalysisError(
                f"{method} diagnostics prediction hash does not match the evaluated file"
            )
        recorded_path = _resolve_frozen_path(
            value.get("prediction_file"), field=f"{method}.prediction_file"
        )
        if recorded_path.resolve() != prediction_path.resolve():
            raise EvaluationBAnalysisError(
                f"{method} diagnostics prediction path does not match the evaluated file"
            )
    for method, value in (("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4)):
        if str(value.get("retained_membership_sha256", "")).strip() != expected_membership_sha256:
            raise EvaluationBAnalysisError(
                f"{method} retained-membership hash does not match the frozen reference"
            )
    if str(m4.get("demo_bank_id", "")) != expected_m4_demo_bank_id:
        raise EvaluationBAnalysisError("M4 diagnostics demo-bank ID does not match the freeze")
    if "demo_overlap_ranks" in m4:
        observed_overlap = {
            _strict_int(value, field="M4.demo_overlap_ranks")
            for value in (m4.get("demo_overlap_ranks") or [])
        }
        if observed_overlap != set(demo_overlap_ranks or set()):
            raise EvaluationBAnalysisError(
                "M4 diagnostics demo-overlap ranks do not match the frozen audit"
            )
    else:
        m4_overlap = m4.get("demo_overlap_case_ids", m4.get("demo_overlap", []))
        if isinstance(m4_overlap, Mapping):
            m4_overlap = m4_overlap.get("case_ids", [])
        if set(m4_overlap or []) != demo_overlap_ids:
            raise EvaluationBAnalysisError(
                "M4 diagnostics demo-overlap set does not match the frozen audit"
            )
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4}


def build_common_membership(
    human_cases: Mapping[str, HumanCase],
    predictions: Mapping[str, Mapping[str, PredictionCase]],
    demo_overlap_ids: set[str],
) -> tuple[list[str], list[str]]:
    substantive = {
        case_id
        for case_id, case in human_cases.items()
        if case.review_status == "SUBSTANTIVE" and case.substantive_amp_evaluable
    }
    abstain = {
        case_id for case_id, case in human_cases.items() if case.review_status == "ABSTAIN"
    }
    expected_common = substantive - demo_overlap_ids
    expected_abstain = abstain - demo_overlap_ids
    for method in METHODS:
        available = set(predictions[method])
        missing = (expected_common | expected_abstain) - available
        if missing:
            raise EvaluationBAnalysisError(
                f"{method} is missing common Evaluation B cases: {sorted(missing)}"
            )
    order = lambda case_id: (human_cases[case_id].search_rank, case_id)
    return sorted(expected_common, key=order), sorted(expected_abstain, key=order)


def _matrix_from_cases(
    case_ids: Sequence[str],
    rows: Mapping[str, HumanCase | SilverCase | PredictionCase],
    *,
    kind: str,
) -> np.ndarray:
    label_rows: list[list[str]] = []
    for case_id in case_ids:
        row = rows[case_id]
        if isinstance(row, PredictionCase):
            labels = list(row.labels)
        else:
            labels = [label for family in FAMILIES for label in row.labels[family]]
        label_rows.append(labels)
    try:
        return labels_to_indicator(label_rows)
    except Exception as exc:
        raise EvaluationBAnalysisError(f"Could not build {kind} label matrix: {exc}") from exc


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _family_diagnostics(
    reference: np.ndarray,
    prediction: np.ndarray,
    family: str,
) -> dict[str, Any]:
    indices = [
        index for index, label in enumerate(AMP_LABEL_IDS) if AMP_FAMILY_BY_LABEL[label] == family
    ]
    family_reference = reference[:, indices]
    family_prediction = prediction[:, indices]
    reference_count = family_reference.sum(axis=1)
    prediction_count = family_prediction.sum(axis=1)
    cpmr_success = np.logical_and(
        prediction_count > 0, np.all(family_prediction <= family_reference, axis=1)
    )
    nonempty = reference_count > 0
    empty = ~nonempty
    recalls = np.divide(
        prediction_count,
        reference_count,
        out=np.zeros(reference.shape[0], dtype=np.float64),
        where=cpmr_success,
    )
    correct_empty = np.logical_and(empty, prediction_count == 0)
    return {
        "cpmr": float(cpmr_success.mean()),
        "cpmr_success_count": int(cpmr_success.sum()),
        "mean_contained_recall": (
            float(recalls[cpmr_success].mean()) if cpmr_success.any() else None
        ),
        "cpmr_nonempty_reference": (
            float(cpmr_success[nonempty].mean()) if nonempty.any() else None
        ),
        "nonempty_reference_n": int(nonempty.sum()),
        "empty_reference_n": int(empty.sum()),
        "empty_reference_correct_empty_count": int(correct_empty.sum()),
        "empty_reference_correct_empty_rate": (
            float(correct_empty.sum() / empty.sum()) if empty.any() else None
        ),
        "per_case_cpmr": cpmr_success.astype(int),
        "per_case_contained_recall": [
            float(recalls[index]) if cpmr_success[index] else None
            for index in range(reference.shape[0])
        ],
        "per_case_empty_reference": empty.astype(int),
        "per_case_correct_empty": correct_empty.astype(int),
    }


def _core_point_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    macro_label_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if macro_label_ids is None:
        macro_label_ids = tuple(
            label
            for label, support in zip(AMP_LABEL_IDS, reference.sum(axis=0), strict=True)
            if support
        )
    if not macro_label_ids:
        raise EvaluationBAnalysisError("Evaluation set has no supported human AMP labels")
    aggregate = compute_amp_metrics(
        reference, prediction, macro_label_ids=tuple(macro_label_ids)
    )
    families = {
        family: _family_diagnostics(reference, prediction, family) for family in FAMILIES
    }
    return aggregate, families


def bootstrap_intervals(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    if n_resamples <= 0:
        raise EvaluationBAnalysisError("Bootstrap resamples must be positive")
    supported = tuple(
        label
        for label, count in zip(AMP_LABEL_IDS, reference.sum(axis=0), strict=True)
        if count
    )
    aggregate, families = _core_point_metrics(
        reference, prediction, macro_label_ids=supported
    )
    point = {
        "macro_f1": aggregate["macro_f1"],
        "micro_f1": aggregate["micro_f1"],
        "exact_set": aggregate["exact_set_accuracy"],
        "jaccard": aggregate["example_jaccard"],
        **{f"{family.lower()}_cpmr": families[family]["cpmr"] for family in FAMILIES},
    }
    samples = {metric: np.empty(n_resamples, dtype=np.float64) for metric in point}
    rng = np.random.default_rng(seed)
    case_n = reference.shape[0]
    for sample_index in range(n_resamples):
        indices = rng.integers(0, case_n, size=case_n)
        sampled_aggregate, sampled_families = _core_point_metrics(
            reference[indices], prediction[indices], macro_label_ids=supported
        )
        values = {
            "macro_f1": sampled_aggregate["macro_f1"],
            "micro_f1": sampled_aggregate["micro_f1"],
            "exact_set": sampled_aggregate["exact_set_accuracy"],
            "jaccard": sampled_aggregate["example_jaccard"],
            **{
                f"{family.lower()}_cpmr": sampled_families[family]["cpmr"]
                for family in FAMILIES
            },
        }
        for metric, value in values.items():
            samples[metric][sample_index] = value
    rows: list[dict[str, Any]] = []
    for metric, values in samples.items():
        low, high = np.percentile(values, (2.5, 97.5), method="linear")
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_low": float(low),
                "ci_high": float(high),
                "confidence_level": 0.95,
                "resamples": n_resamples,
                "seed": seed,
                "resampling_unit": "CASE",
                "bootstrap_method": "PERCENTILE_LINEAR",
                "macro_supported_label_count": len(supported),
            }
        )
    return rows


def _set_category(silver: set[str], human: set[str]) -> str:
    if not silver and not human:
        return "BOTH_EMPTY"
    if silver == human:
        return "EXACT_MATCH"
    if human < silver:
        return "SILVER_BROADER"
    if silver < human:
        return "HUMAN_BROADER"
    if silver & human:
        return "PARTIAL_OVERLAP_BOTH_HAVE_EXTRAS"
    return "NO_OVERLAP"


def compare_silver_human(
    case_ids: Sequence[str],
    human_cases: Mapping[str, HumanCase],
    silver_cases: Mapping[str, SilverCase],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        comparable_ids = [
            case_id
            for case_id in case_ids
            if silver_cases[case_id].family_available[family]
        ]
        exact_values: list[int] = []
        jaccards: list[float] = []
        shared_total = silver_only_total = human_only_total = 0
        label_counts = {
            label: {"silver_support": 0, "human_support": 0, "shared": 0, "agreement": 0}
            for label in FAMILY_LABEL_IDS[family]
        }
        for case_id in case_ids:
            available = silver_cases[case_id].family_available[family]
            silver = set(silver_cases[case_id].labels[family])
            human = set(human_cases[case_id].labels[family])
            if not available:
                case_rows.append(
                    {
                        "reliability_case_id": case_id,
                        "search_rank": human_cases[case_id].search_rank,
                        "jurisdiction": human_cases[case_id].jurisdiction,
                        "family": family,
                        "silver_reference_available": 0,
                        "silver_labels_json": _json([]),
                        "human_labels_json": _json(list(human_cases[case_id].labels[family])),
                        "shared_labels_json": None,
                        "silver_only_labels_json": None,
                        "human_only_labels_json": None,
                        "category": "SILVER_REFERENCE_UNAVAILABLE",
                        "exact_set_concordance": None,
                        "jaccard": None,
                    }
                )
                continue
            shared = silver & human
            silver_only = silver - human
            human_only = human - silver
            union = silver | human
            exact_values.append(int(silver == human))
            jaccards.append(len(shared) / len(union) if union else 1.0)
            shared_total += len(shared)
            silver_only_total += len(silver_only)
            human_only_total += len(human_only)
            for label in FAMILY_LABEL_IDS[family]:
                in_silver = label in silver
                in_human = label in human
                label_counts[label]["silver_support"] += int(in_silver)
                label_counts[label]["human_support"] += int(in_human)
                label_counts[label]["shared"] += int(in_silver and in_human)
                label_counts[label]["agreement"] += int(in_silver == in_human)
            case_rows.append(
                {
                    "reliability_case_id": case_id,
                    "search_rank": human_cases[case_id].search_rank,
                    "jurisdiction": human_cases[case_id].jurisdiction,
                    "family": family,
                    "silver_reference_available": 1,
                    "silver_labels_json": _json(
                        [label for label in FAMILY_LABEL_IDS[family] if label in silver]
                    ),
                    "human_labels_json": _json(
                        [label for label in FAMILY_LABEL_IDS[family] if label in human]
                    ),
                    "shared_labels_json": _json(
                        [label for label in FAMILY_LABEL_IDS[family] if label in shared]
                    ),
                    "silver_only_labels_json": _json(
                        [label for label in FAMILY_LABEL_IDS[family] if label in silver_only]
                    ),
                    "human_only_labels_json": _json(
                        [label for label in FAMILY_LABEL_IDS[family] if label in human_only]
                    ),
                    "category": _set_category(silver, human),
                    "exact_set_concordance": int(silver == human),
                    "jaccard": len(shared) / len(union) if union else 1.0,
                }
            )
        silver_total = shared_total + silver_only_total
        human_total = shared_total + human_only_total
        precision = _safe_ratio(shared_total, silver_total)
        recall = _safe_ratio(shared_total, human_total)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0 if precision is not None and recall is not None else None
        )
        union_label_occurrences = shared_total + silver_only_total + human_only_total
        summary.append(
            {
                "family": family,
                "n": len(comparable_ids),
                "substantive_n": len(case_ids),
                "comparable_n": len(comparable_ids),
                "silver_reference_unavailable_n": len(case_ids) - len(comparable_ids),
                "exact_set_concordance": (
                    float(np.mean(exact_values)) if exact_values else None
                ),
                "mean_jaccard": float(np.mean(jaccards)) if jaccards else None,
                "micro_precision_silver_against_human": precision,
                "micro_recall_silver_against_human": recall,
                "micro_f1_silver_against_human": f1,
                "shared_label_count": shared_total,
                "silver_only_label_count": silver_only_total,
                "human_only_label_count": human_only_total,
                "silver_label_count": silver_total,
                "human_label_count": human_total,
                "silver_only_rate_of_silver_labels": _safe_ratio(
                    silver_only_total, silver_total
                ),
                "human_only_rate_of_human_labels": _safe_ratio(
                    human_only_total, human_total
                ),
                "proportion_silver_labels_supported_by_human": _safe_ratio(
                    shared_total, silver_total
                ),
                "proportion_human_labels_contained_in_silver": _safe_ratio(
                    shared_total, human_total
                ),
                "shared_union_label_rate": _safe_ratio(shared_total, union_label_occurrences),
                "reference_terminology": f"{SILVER_TERM} versus {REFERENCE_TERM}",
                "availability_policy": "EXCLUDE_STRUCTURALLY_ABSENT_SILVER_FAMILY",
            }
        )
        for label, counts in label_counts.items():
            per_label.append(
                {
                    "family": family,
                    "label_id": label,
                    "raw_label": AMP_RAW_LABEL_BY_ID[label],
                    "n": len(comparable_ids),
                    "substantive_n": len(case_ids),
                    "comparable_n": len(comparable_ids),
                    "silver_reference_unavailable_n": len(case_ids) - len(comparable_ids),
                    **counts,
                    "silver_only": counts["silver_support"] - counts["shared"],
                    "human_only": counts["human_support"] - counts["shared"],
                    "raw_agreement": (
                        counts["agreement"] / len(comparable_ids)
                        if comparable_ids
                        else None
                    ),
                }
            )
    return summary, per_label, case_rows


def compare_auxiliary(
    case_ids: Sequence[str],
    human_cases: Mapping[str, HumanCase],
    silver_cases: Mapping[str, SilverCase],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    form_pairs: list[tuple[set[str], set[str]]] = []
    for case_id in case_ids:
        human = human_cases[case_id]
        silver = silver_cases[case_id]
        if human.auxiliary_evaluable:
            form_pairs.append((set(human.geographic_form), set(silver.geographic_form)))
    form_exact = [int(left == right) for left, right in form_pairs]
    form_jaccard = [
        len(left & right) / len(left | right) if left | right else 1.0
        for left, right in form_pairs
    ]
    rows.append(
        {
            "target": "GEOGRAPHIC_FORM",
            "substantive_n": len(case_ids),
            "comparable_n": len(form_pairs),
            "excluded_n": len(case_ids) - len(form_pairs),
            "exact_match_count": sum(form_exact),
            "exact_concordance": float(np.mean(form_exact)) if form_exact else None,
            "mean_jaccard": float(np.mean(form_jaccard)) if form_jaccard else None,
            "status": "DESCRIPTIVE_ONLY",
        }
    )

    for target in ("MULTIPLICITY", "CHILD"):
        pairs: list[tuple[str, str]] = []
        for case_id in case_ids:
            human = human_cases[case_id]
            silver = silver_cases[case_id]
            human_value = human.multiplicity if target == "MULTIPLICITY" else human.child
            silver_value = silver.multiplicity if target == "MULTIPLICITY" else silver.child
            if (
                human.auxiliary_evaluable
                and human_value not in {"UNKNOWN", "Not Applicable", ""}
                and silver_value not in {None, "UNKNOWN", "Not Applicable", ""}
            ):
                pairs.append((human_value, str(silver_value)))
        exact = sum(left == right for left, right in pairs)
        rows.append(
            {
                "target": target,
                "substantive_n": len(case_ids),
                "comparable_n": len(pairs),
                "excluded_n": len(case_ids) - len(pairs),
                "exact_match_count": exact,
                "exact_concordance": exact / len(pairs) if pairs else None,
                "mean_jaccard": None,
                "status": "DESCRIPTIVE_ONLY",
            }
        )

    ocg_pairs: list[tuple[bool, bool]] = []
    for case_id in case_ids:
        human = human_cases[case_id]
        silver = silver_cases[case_id]
        if human.organized_criminal_group_evaluable and silver.organized_criminal_group is not None:
            ocg_pairs.append(
                (human.organized_criminal_group, silver.organized_criminal_group)
            )
    ocg_exact = sum(left == right for left, right in ocg_pairs)
    rows.append(
        {
            "target": "ORGANIZED_CRIMINAL_GROUP",
            "substantive_n": len(case_ids),
            "comparable_n": len(ocg_pairs),
            "excluded_n": len(case_ids) - len(ocg_pairs),
            "exact_match_count": ocg_exact,
            "exact_concordance": ocg_exact / len(ocg_pairs) if ocg_pairs else None,
            "mean_jaccard": None,
            "status": "DESCRIPTIVE_ONLY",
        }
    )
    return rows


def _family_macro_rows(aggregate: Mapping[str, Any], family: str) -> dict[str, Any]:
    supported = [
        row
        for row in aggregate["per_label"]
        if row["family"] == family and row["support"] > 0
    ]
    return {
        "supported_label_count": len(supported),
        "macro_precision_family": (
            float(np.mean([row["precision"] for row in supported])) if supported else None
        ),
        "macro_recall_family": (
            float(np.mean([row["recall"] for row in supported])) if supported else None
        ),
        "macro_f1_family": (
            float(np.mean([row["f1"] for row in supported])) if supported else None
        ),
        "label_supports_json": _json(
            {row["label_id"]: row["support"] for row in aggregate["per_label"] if row["family"] == family}
        ),
    }


def evaluate_methods(
    common_ids: Sequence[str],
    human_cases: Mapping[str, HumanCase],
    silver_cases: Mapping[str, SilverCase],
    predictions: Mapping[str, Mapping[str, PredictionCase]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[dict[str, Any]]]:
    human_matrix = _matrix_from_cases(common_ids, human_cases, kind="human reference")
    dual_reference_ids = [
        case_id
        for case_id in common_ids
        if all(silver_cases[case_id].family_available.values())
    ]
    if not dual_reference_ids:
        raise EvaluationBAnalysisError(
            "No common substantive cases have a complete silver AMP reference"
        )
    dual_human_matrix = _matrix_from_cases(
        dual_reference_ids, human_cases, kind="dual-comparison human reference"
    )
    dual_silver_matrix = _matrix_from_cases(
        dual_reference_ids, silver_cases, kind="complete silver reference"
    )
    main_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    breadth_rows: list[dict[str, Any]] = []
    reference_comparison_rows: list[dict[str, Any]] = []

    human_counts = {
        family: human_matrix[:, [AMP_FAMILY_BY_LABEL[label] == family for label in AMP_LABEL_IDS]].sum(axis=1)
        for family in FAMILIES
    }
    silver_family_available_ids = {
        family: [
            case_id
            for case_id in common_ids
            if silver_cases[case_id].family_available[family]
        ]
        for family in FAMILIES
    }
    mean_silver_family_labels = {
        family: (
            float(
                np.mean(
                    [
                        len(silver_cases[case_id].labels[family])
                        for case_id in silver_family_available_ids[family]
                    ]
                )
            )
            if silver_family_available_ids[family]
            else None
        )
        for family in FAMILIES
    }

    for method in METHODS:
        predicted = _matrix_from_cases(common_ids, predictions[method], kind=f"{method} predictions")
        aggregate, family_diag = _core_point_metrics(human_matrix, predicted)
        ci_rows = bootstrap_intervals(
            human_matrix,
            predicted,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        ci_by_metric = {row["metric"]: row for row in ci_rows}
        for row in ci_rows:
            bootstrap_rows.append({"method": method, "n": len(common_ids), **row})
        main_rows.append(
            {
                "method": method,
                "n": len(common_ids),
                "macro_f1": aggregate["macro_f1"],
                "macro_f1_ci_low": ci_by_metric["macro_f1"]["ci_low"],
                "macro_f1_ci_high": ci_by_metric["macro_f1"]["ci_high"],
                "micro_f1": aggregate["micro_f1"],
                "micro_f1_ci_low": ci_by_metric["micro_f1"]["ci_low"],
                "micro_f1_ci_high": ci_by_metric["micro_f1"]["ci_high"],
                "exact_set": aggregate["exact_set_accuracy"],
                "exact_set_ci_low": ci_by_metric["exact_set"]["ci_low"],
                "exact_set_ci_high": ci_by_metric["exact_set"]["ci_high"],
                "jaccard": aggregate["example_jaccard"],
                "jaccard_ci_low": ci_by_metric["jaccard"]["ci_low"],
                "jaccard_ci_high": ci_by_metric["jaccard"]["ci_high"],
                "macro_supported_label_count": aggregate["macro_label_count"],
                **{
                    f"{family.lower()}_cpmr": family_diag[family]["cpmr"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_cpmr_ci_low": ci_by_metric[f"{family.lower()}_cpmr"]["ci_low"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_cpmr_ci_high": ci_by_metric[f"{family.lower()}_cpmr"]["ci_high"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_mean_contained_recall": family_diag[family]["mean_contained_recall"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_cpmr_nonempty_reference": family_diag[family]["cpmr_nonempty_reference"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_empty_reference_n": family_diag[family]["empty_reference_n"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_empty_reference_correct_empty_count": family_diag[family]["empty_reference_correct_empty_count"]
                    for family in FAMILIES
                },
                **{
                    f"{family.lower()}_empty_reference_correct_empty_rate": family_diag[family]["empty_reference_correct_empty_rate"]
                    for family in FAMILIES
                },
                "reference_terminology": REFERENCE_TERM,
            }
        )
        for family in FAMILIES:
            family_rows.append(
                {
                    "method": method,
                    "family": family,
                    "n": len(common_ids),
                    **_family_macro_rows(aggregate, family),
                    "cpmr": family_diag[family]["cpmr"],
                    "mean_contained_recall": family_diag[family]["mean_contained_recall"],
                    "cpmr_nonempty_reference": family_diag[family]["cpmr_nonempty_reference"],
                    "nonempty_reference_n": family_diag[family]["nonempty_reference_n"],
                    "empty_reference_n": family_diag[family]["empty_reference_n"],
                    "empty_reference_correct_empty_count": family_diag[family]["empty_reference_correct_empty_count"],
                    "empty_reference_correct_empty_rate": family_diag[family]["empty_reference_correct_empty_rate"],
                    "reference_terminology": REFERENCE_TERM,
                }
            )
        for row in aggregate["per_label"]:
            per_label_rows.append(
                {
                    "method": method,
                    "n": len(common_ids),
                    **row,
                    "reference_terminology": REFERENCE_TERM,
                }
            )

        predicted_counts = {
            family: predicted[:, [AMP_FAMILY_BY_LABEL[label] == family for label in AMP_LABEL_IDS]].sum(axis=1)
            for family in FAMILIES
        }
        total_predicted = predicted.sum(axis=1)
        breadth_rows.append(
            {
                "method": method,
                "n": len(common_ids),
                **{
                    f"mean_predicted_{family.lower()}_labels": float(predicted_counts[family].mean())
                    for family in FAMILIES
                },
                "mean_total_predicted_labels": float(total_predicted.mean()),
                "median_total_predicted_labels": float(np.median(total_predicted)),
                **{
                    f"mean_human_{family.lower()}_labels": float(human_counts[family].mean())
                    for family in FAMILIES
                },
                "mean_total_human_labels": float(human_matrix.sum(axis=1).mean()),
                **{
                    f"mean_silver_{family.lower()}_labels": mean_silver_family_labels[family]
                    for family in FAMILIES
                },
                **{
                    f"silver_{family.lower()}_reference_available_n": len(
                        silver_family_available_ids[family]
                    )
                    for family in FAMILIES
                },
                "complete_silver_amp_reference_n": len(dual_reference_ids),
                "incomplete_silver_amp_reference_n": len(common_ids)
                - len(dual_reference_ids),
                "mean_total_silver_labels": float(
                    dual_silver_matrix.sum(axis=1).mean()
                ),
                "status": "DESCRIPTIVE_NOT_PRIMARY_PERFORMANCE_METRIC",
            }
        )

        dual_predicted = _matrix_from_cases(
            dual_reference_ids,
            predictions[method],
            kind=f"{method} dual-reference predictions",
        )
        dual_human_aggregate, dual_human_family_diag = _core_point_metrics(
            dual_human_matrix, dual_predicted
        )
        silver_aggregate, silver_family_diag = _core_point_metrics(
            dual_silver_matrix, dual_predicted
        )
        dual_human_family_macro = {
            family: _family_macro_rows(dual_human_aggregate, family)
            for family in FAMILIES
        }
        silver_family_macro = {
            family: _family_macro_rows(silver_aggregate, family) for family in FAMILIES
        }
        metric_pairs: list[tuple[str, str, float | None, float | None]] = [
            ("OVERALL", "macro_f1", dual_human_aggregate["macro_f1"], silver_aggregate["macro_f1"]),
            ("OVERALL", "micro_f1", dual_human_aggregate["micro_f1"], silver_aggregate["micro_f1"]),
            ("OVERALL", "exact_set", dual_human_aggregate["exact_set_accuracy"], silver_aggregate["exact_set_accuracy"]),
            ("OVERALL", "jaccard", dual_human_aggregate["example_jaccard"], silver_aggregate["example_jaccard"]),
        ]
        for family in FAMILIES:
            metric_pairs.extend(
                [
                    (
                        family,
                        "cpmr",
                        dual_human_family_diag[family]["cpmr"],
                        silver_family_diag[family]["cpmr"],
                    ),
                    (
                        family,
                        "macro_precision_family",
                        dual_human_family_macro[family]["macro_precision_family"],
                        silver_family_macro[family]["macro_precision_family"],
                    ),
                    (
                        family,
                        "macro_recall_family",
                        dual_human_family_macro[family]["macro_recall_family"],
                        silver_family_macro[family]["macro_recall_family"],
                    ),
                ]
            )
        for scope, metric, human_value, silver_value in metric_pairs:
            reference_comparison_rows.append(
                {
                    "method": method,
                    "metric_scope": scope,
                    "metric": metric,
                    "n": len(dual_reference_ids),
                    "human_primary_n": len(common_ids),
                    "dual_reference_n": len(dual_reference_ids),
                    "excluded_incomplete_silver_reference_n": len(common_ids)
                    - len(dual_reference_ids),
                    "human_grounded_value": human_value,
                    "silver_reference_value": silver_value,
                    "delta_human_minus_silver": (
                        human_value - silver_value
                        if human_value is not None and silver_value is not None
                        else None
                    ),
                    "human_supported_label_count": dual_human_aggregate[
                        "macro_label_count"
                    ],
                    "silver_supported_label_count": silver_aggregate["macro_label_count"],
                    "significance_claim": "NOT_TESTED_DO_NOT_INFER",
                }
            )

    grouped_comparisons: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in reference_comparison_rows:
        grouped_comparisons.setdefault(
            (str(row["metric_scope"]), str(row["metric"])), []
        ).append(row)
    for rows in grouped_comparisons.values():
        human_values = sorted(
            {
                float(row["human_grounded_value"])
                for row in rows
                if row["human_grounded_value"] is not None
            },
            reverse=True,
        )
        silver_values = sorted(
            {
                float(row["silver_reference_value"])
                for row in rows
                if row["silver_reference_value"] is not None
            },
            reverse=True,
        )
        for row in rows:
            human_rank = (
                human_values.index(float(row["human_grounded_value"])) + 1
                if row["human_grounded_value"] is not None
                else None
            )
            silver_rank = (
                silver_values.index(float(row["silver_reference_value"])) + 1
                if row["silver_reference_value"] is not None
                else None
            )
            row["human_grounded_dense_rank"] = human_rank
            row["silver_reference_dense_rank"] = silver_rank
            row["rank_changed"] = (
                int(human_rank != silver_rank)
                if human_rank is not None and silver_rank is not None
                else None
            )

    return {
        "main": main_rows,
        "bootstrap": bootstrap_rows,
        "family": family_rows,
        "per_label": per_label_rows,
        "breadth": breadth_rows,
        "reference_comparison": reference_comparison_rows,
    }


def evaluate_abstain(
    abstain_ids: Sequence[str],
    human_cases: Mapping[str, HumanCase],
    predictions: Mapping[str, Mapping[str, PredictionCase]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not abstain_ids:
        raise EvaluationBAnalysisError("Retained reference has no ABSTAIN cases")
    summary: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for method in METHODS:
        totals: list[int] = []
        any_by_family = {family: 0 for family in FAMILIES}
        all_empty = 0
        for case_id in abstain_ids:
            prediction = predictions[method][case_id]
            by_family = prediction.by_family
            total = len(prediction.labels)
            totals.append(total)
            all_empty += int(total == 0)
            for family in FAMILIES:
                any_by_family[family] += int(bool(by_family[family]))
            case_rows.append(
                {
                    "reliability_case_id": case_id,
                    "search_rank": human_cases[case_id].search_rank,
                    "jurisdiction": human_cases[case_id].jurisdiction,
                    "fact_summary": human_cases[case_id].fact_summary,
                    "review_status": "ABSTAIN",
                    "reviewer_note": human_cases[case_id].annotation_notes,
                    "method": method,
                    "predicted_labels_json": _json(list(prediction.labels)),
                    "predicted_acts_json": _json(list(by_family["ACT"])),
                    "predicted_means_json": _json(list(by_family["MEANS"])),
                    "predicted_purposes_json": _json(list(by_family["PURPOSE"])),
                    "predicted_label_count": total,
                    "all_amp_empty": int(total == 0),
                    "interpretation": "NARRATIVE_INSUFFICIENCY_DIAGNOSTIC_NOT_ALL_NEGATIVE_ACCURACY",
                }
            )
        summary.append(
            {
                "method": method,
                "abstain_n": len(abstain_ids),
                "all_amp_empty_count": all_empty,
                "all_amp_empty_rate": all_empty / len(abstain_ids),
                "narrative_insufficiency_safe_rate": all_empty / len(abstain_ids),
                "mean_total_predicted_label_count": float(np.mean(totals)),
                "median_total_predicted_label_count": float(np.median(totals)),
                "cases_with_any_predicted_act": any_by_family["ACT"],
                "cases_with_any_predicted_means": any_by_family["MEANS"],
                "cases_with_any_predicted_purpose": any_by_family["PURPOSE"],
                "total_unsupported_predicted_label_count_under_abstention_interpretation": sum(totals),
                "status": "DESCRIPTIVE_ABSTENTION_DIAGNOSTIC",
            }
        )
    return summary, case_rows


def build_case_level_table(
    common_ids: Sequence[str],
    human_cases: Mapping[str, HumanCase],
    silver_cases: Mapping[str, SilverCase],
    predictions: Mapping[str, Mapping[str, PredictionCase]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in common_ids:
        human = human_cases[case_id]
        silver = silver_cases[case_id]
        row: dict[str, Any] = {
            "reliability_case_id": case_id,
            "search_rank": human.search_rank,
            "jurisdiction": human.jurisdiction,
            "fact_summary": human.fact_summary,
            "review_status": human.review_status,
        }
        for family in FAMILIES:
            key = family.lower()
            human_set = set(human.labels[family])
            silver_set = set(silver.labels[family])
            silver_available = silver.family_available[family]
            row[f"human_{key}_json"] = _json(list(human.labels[family]))
            row[f"silver_{key}_json"] = _json(list(silver.labels[family]))
            row[f"silver_{key}_reference_available"] = int(silver_available)
            row[f"silver_human_{key}_category"] = (
                _set_category(silver_set, human_set)
                if silver_available
                else "SILVER_REFERENCE_UNAVAILABLE"
            )
        combined_human = {label for family in FAMILIES for label in human.labels[family]}
        combined_silver = {label for family in FAMILIES for label in silver.labels[family]}
        complete_silver = all(silver.family_available.values())
        row["complete_silver_amp_reference_available"] = int(complete_silver)
        row["silver_human_combined_category"] = (
            _set_category(combined_silver, combined_human)
            if complete_silver
            else "SILVER_AMP_REFERENCE_INCOMPLETE"
        )
        prediction_signatures: list[tuple[str, ...]] = []
        for method in METHODS:
            prediction = predictions[method][case_id]
            prediction_signatures.append(prediction.labels)
            row[f"{method.lower()}_prediction_json"] = _json(list(prediction.labels))
            predicted_set = set(prediction.labels)
            union_all = combined_human | predicted_set
            row[f"{method.lower()}_exact_set"] = int(predicted_set == combined_human)
            row[f"{method.lower()}_jaccard"] = (
                len(predicted_set & combined_human) / len(union_all) if union_all else 1.0
            )
            by_family = prediction.by_family
            for family in FAMILIES:
                key = family.lower()
                human_set = set(human.labels[family])
                predicted_family = set(by_family[family])
                union = human_set | predicted_family
                contained = bool(predicted_family) and predicted_family.issubset(human_set)
                row[f"{method.lower()}_{key}_prediction_json"] = _json(list(by_family[family]))
                row[f"{method.lower()}_{key}_true_positive_json"] = _json(
                    [label for label in FAMILY_LABEL_IDS[family] if label in predicted_family & human_set]
                )
                row[f"{method.lower()}_{key}_false_positive_json"] = _json(
                    [label for label in FAMILY_LABEL_IDS[family] if label in predicted_family - human_set]
                )
                row[f"{method.lower()}_{key}_false_negative_json"] = _json(
                    [label for label in FAMILY_LABEL_IDS[family] if label in human_set - predicted_family]
                )
                row[f"{method.lower()}_{key}_jaccard"] = (
                    len(predicted_family & human_set) / len(union) if union else 1.0
                )
                row[f"{method.lower()}_{key}_exact_set"] = int(predicted_family == human_set)
                row[f"{method.lower()}_{key}_cpmr"] = int(contained)
                row[f"{method.lower()}_{key}_contained_recall"] = (
                    len(predicted_family) / len(human_set) if contained else None
                )
                row[f"{method.lower()}_{key}_empty_reference"] = int(not human_set)
                row[f"{method.lower()}_{key}_correct_empty"] = int(
                    not human_set and not predicted_family
                )
        row["model_prediction_disagreement"] = int(len(set(prediction_signatures)) > 1)
        rows.append(row)
    return rows


def build_analysis(
    human_cases: Mapping[str, HumanCase],
    silver_cases: Mapping[str, SilverCase],
    predictions: Mapping[str, Mapping[str, PredictionCase]],
    demo_overlap_ids: set[str],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    common_ids, abstain_ids = build_common_membership(
        human_cases, predictions, demo_overlap_ids
    )
    if not common_ids:
        raise EvaluationBAnalysisError("Common substantive Evaluation B set is empty")
    substantive_ids = sorted(
        (
            case_id
            for case_id, case in human_cases.items()
            if case.review_status == "SUBSTANTIVE"
            and case.substantive_amp_evaluable
        ),
        key=lambda case_id: (human_cases[case_id].search_rank, case_id),
    )
    dual_reference_ids = [
        case_id
        for case_id in common_ids
        if all(silver_cases[case_id].family_available.values())
    ]
    silver_summary, silver_per_label, silver_case = compare_silver_human(
        substantive_ids, human_cases, silver_cases
    )
    method_results = evaluate_methods(
        common_ids,
        human_cases,
        silver_cases,
        predictions,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    abstain_summary, abstain_cases = evaluate_abstain(
        abstain_ids, human_cases, predictions
    )
    tables = {
        "silver_vs_human_summary.csv": silver_summary,
        "silver_vs_human_per_label.csv": silver_per_label,
        "silver_vs_human_case_level.csv": silver_case,
        "auxiliary_silver_vs_human_summary.csv": compare_auxiliary(
            substantive_ids, human_cases, silver_cases
        ),
        "eval_b_main_results.csv": method_results["main"],
        "eval_b_bootstrap_cis.csv": method_results["bootstrap"],
        "eval_b_family_results.csv": method_results["family"],
        "eval_b_per_label_results.csv": method_results["per_label"],
        "eval_b_abstain_results.csv": abstain_summary,
        "eval_b_abstain_case_level.csv": abstain_cases,
        "eval_b_prediction_breadth.csv": method_results["breadth"],
        "model_silver_vs_human_metric_comparison.csv": method_results[
            "reference_comparison"
        ],
        "human_grounded_case_level_errors.csv": build_case_level_table(
            common_ids, human_cases, silver_cases, predictions
        ),
    }
    metadata = {
        "schema_version": "sherloc-evaluation-b-analysis-v1",
        "evaluator_version": VERSION,
        "single_reviewer_design": True,
        "inter_annotator_statistics_computed": False,
        "adjudication_performed": False,
        "retained_n": len(human_cases),
        "substantive_n": sum(
            case.review_status == "SUBSTANTIVE" for case in human_cases.values()
        ),
        "abstain_n": sum(case.review_status == "ABSTAIN" for case in human_cases.values()),
        "m4_demo_overlap_case_ids": sorted(demo_overlap_ids),
        "common_substantive_n": len(common_ids),
        "common_substantive_case_ids": list(common_ids),
        "silver_human_substantive_n": len(substantive_ids),
        "silver_human_substantive_case_ids": list(substantive_ids),
        "silver_human_family_comparable_n": {
            family: sum(
                silver_cases[case_id].family_available[family]
                for case_id in substantive_ids
            )
            for family in FAMILIES
        },
        "silver_human_family_unavailable_case_ids": {
            family: [
                case_id
                for case_id in substantive_ids
                if not silver_cases[case_id].family_available[family]
            ]
            for family in FAMILIES
        },
        "dual_reference_complete_amp_n": len(dual_reference_ids),
        "dual_reference_complete_amp_case_ids": list(dual_reference_ids),
        "dual_reference_incomplete_silver_n": len(common_ids)
        - len(dual_reference_ids),
        "dual_reference_incomplete_silver_case_ids": [
            case_id
            for case_id in common_ids
            if not all(silver_cases[case_id].family_available.values())
        ],
        "abstain_evaluation_n": len(abstain_ids),
        "abstain_evaluation_case_ids": list(abstain_ids),
        "bootstrap": {"resamples": bootstrap_resamples, "seed": bootstrap_seed},
        "model_training_performed_by_analysis_stage": False,
        "api_calls_performed_by_analysis_stage": 0,
        "upstream_model_and_api_execution_required": True,
        "auxiliary_model_benchmark_run": False,
        "reference_terminology": REFERENCE_TERM,
    }
    return tables, metadata


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationBAnalysisError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _format(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_format(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def render_report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    qc_summary: Mapping[str, Any],
    execution_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    main = tables["eval_b_main_results.csv"]
    silver = tables["silver_vs_human_summary.csv"]
    abstain = tables["eval_b_abstain_results.csv"]
    breadth = tables["eval_b_prediction_breadth.csv"]
    family_results = tables["eval_b_family_results.csv"]
    auxiliary = tables["auxiliary_silver_vs_human_summary.csv"]
    label_support = [
        {
            "family": row["family"],
            "label_id": row["label_id"],
            "human_support": row["support"],
        }
        for row in tables["eval_b_per_label_results.csv"]
        if row["method"] == METHODS[0]
    ]
    reference_deltas = [
        row
        for row in tables["model_silver_vs_human_metric_comparison.csv"]
        if row["metric_scope"] == "OVERALL" or row["metric"] == "cpmr"
    ]
    reference_family_deltas = [
        row
        for row in tables["model_silver_vs_human_metric_comparison.csv"]
        if row["metric"] in {"macro_precision_family", "macro_recall_family"}
    ]
    mismatch_patterns = sorted(
        tables["silver_vs_human_per_label.csv"],
        key=lambda row: (
            -(int(row["silver_only"]) + int(row["human_only"])),
            str(row["label_id"]),
        ),
    )[:10]
    source_n = source_manifest.get("row_count", source_manifest.get("source_row_count", "N/A"))
    reviewed_n = qc_summary.get("reviewed_n", qc_summary.get("reviewed_count", "N/A"))
    skip_n = qc_summary.get("skip_n", qc_summary.get("skip_count", "N/A"))
    unreviewed_n = qc_summary.get(
        "not_reviewed_n",
        qc_summary.get("unreviewed_n", qc_summary.get("not_reviewed_count", "N/A")),
    )
    lines = [
        "# Evaluation B: single-reviewer human-grounded narrative validation",
        "",
        "## 1. Objective and design",
        "",
        "Evaluation B assesses AMP extraction against a **single-reviewer human-grounded narrative reference** restricted to information recoverable from each English Fact Summary. Primary Evaluation A remains unchanged and uses SHERLOC Legacy Keywords as silver-reference labels.",
        "",
        "Only one human reviewer was available for the completed evaluation, so reviewer-to-reviewer agreement could not be estimated. No second reviewer was fabricated and no reviewer adjudication was performed.",
        "",
        "## 2. Human annotation protocol and retained cases",
        "",
        f"The immutable source contained **{source_n}** rows; **{reviewed_n}** were reviewed, **{unreviewed_n}** remained unreviewed, and **{skip_n}** reviewed cases were marked Skip according to the QC summary. Skip cases were excluded without replacement. The retained reference contains **{metadata['retained_n']}** cases: **{metadata['substantive_n']}** substantive and **{metadata['abstain_n']}** narrative-insufficiency cases.",
        "",
        "Skip indicates that the reviewer judged a case unsuitable for this evaluation. Abstain indicates that the narrative was insufficient for reliable extraction; those cases are retained for a separate diagnostic and are not treated as ordinary all-negative AMP references.",
        "",
        "## 3. Human-reference construction and AMP support",
        "",
        "Human labels were deterministically syntax-normalized and mapped to the frozen AMP ontology without semantic reinterpretation. Organized Criminal Group was separated from Geographic Form. The final primary comparison uses one common substantive membership of **{}** cases.".format(metadata["common_substantive_n"]),
        "",
        _markdown_table(label_support, ("family", "label_id", "human_support")),
        "",
        "## 4. SHERLOC silver reference versus human narrative reference",
        "",
        _markdown_table(
            silver,
            (
                "family",
                "n",
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
                "shared_union_label_rate",
            ),
        ),
        "",
        "Family-specific concordance denominators exclude structurally absent Legacy Keyword families. Unavailable Act or Means metadata is retained in the case audit with `SILVER_REFERENCE_UNAVAILABLE`; it is not scored as an empty label set.",
        "",
        "The largest label-level mismatch counts were:",
        "",
        _markdown_table(
            mismatch_patterns,
            ("family", "label_id", "silver_only", "human_only", "shared"),
        ),
        "",
        "Silver-only labels are not automatically errors. SHERLOC structured metadata may contain information broader than what is directly recoverable from the Fact Summary, while human annotation is intentionally narrative-restricted.",
        "",
        "## 5. Leakage-free model evaluation design",
        "",
        "All retained human cases were required to be excluded from dedicated M1/M2 training, validation, threshold tuning, and supervised selection. M1/M2 used transferred fixed Evaluation A settings; human labels were not used for model tuning. M3/M4 use their frozen AMP-only prompts and technical policy. Any overlap with the active M4 demonstration bank is removed from the common comparison without selecting replacement demonstrations.",
        "",
        f"The dedicated M1 training N was **{execution_metadata['M1'].get('train_n', 'N/A')}** and the dedicated M2 training N was **{execution_metadata['M2'].get('train_n', 'N/A')}**. M3 recorded **{execution_metadata['M3'].get('new_api_request_successes', 'N/A')}** new successful requests and **{execution_metadata['M3'].get('reused_identical_requests', 'N/A')}** identical-request reuses; M4 recorded **{execution_metadata['M4'].get('new_api_request_successes', 'N/A')}** and **{execution_metadata['M4'].get('reused_identical_requests', 'N/A')}**, respectively.",
        "",
        f"M1 fixed settings: `{_json(execution_metadata['M1'].get('fixed_hyperparameters', {}))}`. M2 fixed settings: `{_json(execution_metadata['M2'].get('fixed_hyperparameters', {}))}`, with technical execution `{_json(execution_metadata['M2'].get('technical_execution', {}))}`. M3/M4 used model `{execution_metadata['M3'].get('model', 'N/A')}`, their frozen prompt hashes, the frozen schema hash `{execution_metadata['M3'].get('schema_sha256', 'N/A')}`, and `store=false`.",
        "",
        "## 6. Main human-grounded M1-M4 results",
        "",
        _markdown_table(
            main,
            (
                "method",
                "n",
                "macro_f1",
                "micro_f1",
                "exact_set",
                "jaccard",
                "act_cpmr",
                "means_cpmr",
                "purpose_cpmr",
            ),
        ),
        "",
        f"Confidence intervals use {metadata['bootstrap']['resamples']} deterministic case-level percentile bootstrap resamples with seed {metadata['bootstrap']['seed']}. The small human-reference N warrants cautious interpretation; no statistical-significance claim is made.",
        "",
        "## 7. CPMR, contained recall, and empty-reference behavior",
        "",
        "Standard CPMR retains the previously frozen definition: a nonempty prediction contained in the reference set. CPMR_nonempty_reference is reported separately on cases whose human family reference is nonempty. Empty-reference correct-empty rate measures whether a method leaves a family empty when the human narrative reference is empty. CPMR measures reference-contained behavior, not absolute factual correctness.",
        "",
        _markdown_table(
            family_results,
            (
                "method",
                "family",
                "macro_precision_family",
                "macro_recall_family",
                "macro_f1_family",
                "cpmr",
                "mean_contained_recall",
                "cpmr_nonempty_reference",
                "nonempty_reference_n",
                "empty_reference_n",
                "empty_reference_correct_empty_rate",
            ),
        ),
        "",
        "## 8. Prediction breadth",
        "",
        _markdown_table(
            breadth,
            (
                "method",
                "mean_predicted_act_labels",
                "mean_predicted_means_labels",
                "mean_predicted_purpose_labels",
                "mean_total_predicted_labels",
                "mean_total_human_labels",
                "silver_act_reference_available_n",
                "silver_means_reference_available_n",
                "silver_purpose_reference_available_n",
                "complete_silver_amp_reference_n",
                "mean_total_silver_labels",
            ),
        ),
        "",
        "## 9. Narrative-insufficiency cases",
        "",
        _markdown_table(
            abstain,
            (
                "method",
                "abstain_n",
                "all_amp_empty_rate",
                "mean_total_predicted_label_count",
                "total_unsupported_predicted_label_count_under_abstention_interpretation",
            ),
        ),
        "",
        "The narrative_insufficiency_safe_rate is an operational descriptive diagnostic, not standard accuracy. M1/M2 were not trained with an explicit abstention objective.",
        "",
        "## 10. Silver-scored versus human-scored behavior",
        "",
        f"`model_silver_vs_human_metric_comparison.csv` reports human-grounded minus silver-reference deltas on the identical **{metadata['dual_reference_complete_amp_n']}** common cases with complete silver AMP reference. The primary human-grounded model comparison remains **{metadata['common_substantive_n']}** cases; incomplete silver fields are never interpreted as affirmative empty targets. The core numeric deltas are:",
        "",
        _markdown_table(
            reference_deltas,
            (
                "method",
                "metric_scope",
                "metric",
                "dual_reference_n",
                "excluded_incomplete_silver_reference_n",
                "silver_reference_value",
                "human_grounded_value",
                "delta_human_minus_silver",
                "silver_reference_dense_rank",
                "human_grounded_dense_rank",
                "rank_changed",
            ),
        ),
        "",
        "Family-level macro precision and recall deltas are:",
        "",
        _markdown_table(
            reference_family_deltas,
            (
                "method",
                "metric_scope",
                "metric",
                "dual_reference_n",
                "excluded_incomplete_silver_reference_n",
                "silver_reference_value",
                "human_grounded_value",
                "delta_human_minus_silver",
                "silver_reference_dense_rank",
                "human_grounded_dense_rank",
                "rank_changed",
            ),
        ),
        "",
        "These deltas are descriptive and do not establish that either reference source is universally superior.",
        "",
        "## 11. Auxiliary descriptive comparison",
        "",
        _markdown_table(
            auxiliary,
            ("target", "substantive_n", "comparable_n", "exact_concordance", "mean_jaccard"),
        ),
        "",
        "No auxiliary predictive model was trained or evaluated.",
        "",
        "## 12. Limitations",
        "",
        "- The reference was produced by one reviewer; reviewer-to-reviewer reliability is unavailable.",
        "- Human labels are intentionally limited to the supplied narrative and need not reproduce broader SHERLOC metadata.",
        "- The substantive and abstain samples are small; uncertainty intervals may be wide.",
        "- Silver/human and jurisdiction-specific patterns are descriptive and should not be overinterpreted.",
        "- Evaluation A was not modified, and no A4 auxiliary model benchmark was run.",
        "",
        "## 13. Files",
        "",
        "Canonical tables are under `outputs/analysis/evaluation_b/`; the four figures are under `outputs/figures/evaluation_b/`; the readable unexecuted view is `notebooks/10_human_grounded_evaluation.ipynb`; and this report is `docs/evaluation_b_human_grounded_report.md`.",
        "",
        "## 14. Integrity",
        "",
        f"The immutable raw annotation source still matched SHA-256 `{source_manifest.get('sha256', source_manifest.get('source', {}).get('sha256', 'N/A'))}`. Skip rows were excluded, and Abstain rows were retained only in the narrative-insufficiency diagnostic. No second reviewer was fabricated; reviewer-to-reviewer statistics were not computed. All retained human cases were excluded from supervised M1/M2 training, no active M4 demonstration overlapped the evaluated membership, and human labels were not used for model tuning. Evaluation A passed the preserved hash manifest before and after this analysis. No A4 auxiliary model benchmark was run.",
        "",
        FINAL_REPORT_SENTENCE,
    ]
    return "\n".join(lines)


def _save_svg(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".svg", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    figure.savefig(
        temporary,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "SHERLOC Evaluation B analysis v1"},
    )
    plt.close(figure)
    os.replace(temporary, path)


def generate_figures(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], figure_dir: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "sherloc-evaluation-b-v1",
        }
    )
    main = list(tables["eval_b_main_results.csv"])
    x = np.arange(len(METHODS))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for axis, (column, title) in zip(
        axes,
        (("macro_f1", "Macro-F1"), ("micro_f1", "Micro-F1"), ("jaccard", "Jaccard")),
        strict=True,
    ):
        values = [next(row[column] for row in main if row["method"] == method) for method in METHODS]
        axis.bar(x, values, color=[METHOD_COLORS[method] for method in METHODS])
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, 1)
        axis.set_title(title)
    axes[0].set_ylabel("Score")
    fig.suptitle("Evaluation B: human-grounded AMP performance")
    fig.tight_layout()
    _save_svg(figure_dir / FIGURE_NAMES[0], fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for axis, family in zip(axes, FAMILIES, strict=True):
        column = f"{family.lower()}_cpmr"
        values = [next(row[column] for row in main if row["method"] == method) for method in METHODS]
        axis.bar(x, values, color=[METHOD_COLORS[method] for method in METHODS])
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, 1)
        axis.set_title(family.title())
    axes[0].set_ylabel("CPMR")
    fig.suptitle("Evaluation B: reference-contained predictions by AMP family")
    fig.tight_layout()
    _save_svg(figure_dir / FIGURE_NAMES[1], fig)

    comparison = list(tables["model_silver_vs_human_metric_comparison.csv"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for axis, metric in zip(axes, ("macro_f1", "micro_f1", "jaccard"), strict=True):
        selected = {
            row["method"]: row
            for row in comparison
            if row["metric_scope"] == "OVERALL" and row["metric"] == metric
        }
        width = 0.36
        axis.bar(
            x - width / 2,
            [selected[method]["silver_reference_value"] for method in METHODS],
            width,
            label="Silver reference",
            color="#9ECAE1",
        )
        axis.bar(
            x + width / 2,
            [selected[method]["human_grounded_value"] for method in METHODS],
            width,
            label="Human narrative",
            color="#FB6A4A",
        )
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, 1)
        axis.set_title(metric.replace("_", " ").title())
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Identical cases scored against silver and human references")
    fig.tight_layout()
    _save_svg(figure_dir / FIGURE_NAMES[2], fig)

    silver_summary = list(tables["silver_vs_human_summary.csv"])
    family_x = np.arange(len(FAMILIES))
    silver_only = [
        next(row["silver_only_label_count"] / row["silver_label_count"] if row["silver_label_count"] else 0.0 for row in silver_summary if row["family"] == family)
        for family in FAMILIES
    ]
    human_only = [
        next(row["human_only_label_count"] / row["human_label_count"] if row["human_label_count"] else 0.0 for row in silver_summary if row["family"] == family)
        for family in FAMILIES
    ]
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    width = 0.36
    axis.bar(family_x - width / 2, silver_only, width, label="Silver-only / silver labels", color="#9ECAE1")
    axis.bar(family_x + width / 2, human_only, width, label="Human-only / human labels", color="#FB6A4A")
    axis.set_xticks(family_x, [family.title() for family in FAMILIES])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proportion")
    axis.set_title("Silver-only and human-only narrative-supported label proportions")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_svg(figure_dir / FIGURE_NAMES[3], fig)


def _write_outputs(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    *,
    output_dir: Path,
    figure_dir: Path,
    report_path: Path,
    report_text: str,
    input_hashes: Mapping[str, str],
) -> None:
    for filename in OUTPUT_TABLE_NAMES:
        _atomic_csv(output_dir / filename, tables[filename])
    generate_figures(tables, figure_dir)
    _atomic_text(report_path, report_text)
    output_hashes = {
        str(path.relative_to(REPO_ROOT)): _sha256_file(path)
        for path in [
            *(output_dir / name for name in OUTPUT_TABLE_NAMES),
            *(figure_dir / name for name in FIGURE_NAMES),
            report_path,
        ]
    }
    manifest = {
        **metadata,
        "deterministic": True,
        "inputs_sha256": dict(sorted(input_hashes.items())),
        "outputs_sha256": dict(sorted(output_hashes.items())),
    }
    _atomic_text(
        output_dir / "evaluation_b_analysis_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _parse_prediction_arguments(values: Sequence[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_PREDICTIONS)
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise EvaluationBAnalysisError(
                f"--prediction must use METHOD=PATH syntax: {value!r}"
            )
        method, path = value.split("=", 1)
        method = method.strip().upper()
        if method not in METHODS or method in output:
            raise EvaluationBAnalysisError(f"Invalid or duplicate prediction method: {method}")
        output[method] = Path(path)
    if set(output) != set(METHODS):
        raise EvaluationBAnalysisError("Exactly one prediction path is required for M1, M2, M3, and M4")
    return output


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    validate_evaluation_a_integrity(args.evaluation_a_baseline)
    prediction_paths = _parse_prediction_arguments(args.prediction)
    human = load_human_reference(args.human_reference)
    source_manifest = _read_json(args.source_manifest)
    qc_summary = _read_json(args.qc_summary)
    membership_manifest = _read_json(args.membership_manifest)
    raw_annotation_source = validate_human_reference_provenance(
        source_manifest,
        qc_summary,
        human,
        human_reference_path=args.human_reference,
        source_manifest_path=args.source_manifest,
        qc_summary_path=args.qc_summary,
        membership_manifest=membership_manifest,
    )
    membership_sha256 = retained_membership_sha256(human)
    frozen_membership_path = _resolve_frozen_path(
        membership_manifest["membership"]["path"], field="membership.path"
    )
    silver = load_silver_reference(
        args.reference_key, args.management_sample, args.benchmark, human
    )
    validate_leakage_audit(
        args.leakage_audit,
        set(human),
        expected_membership_sha256=membership_sha256,
    )
    overlap = load_demo_overlap(args.demo_bank, human)
    m4_demo_bank_id, m4_demo_membership_sha256 = load_m4_demo_bank_provenance(
        args.demo_bank
    )
    execution_metadata = validate_execution_metadata(
        args.m1_metadata,
        args.m2_metadata,
        args.m3_diagnostics,
        args.m4_diagnostics,
        retained_n=len(human),
        expected_m4_n=len(human) - len(overlap),
        demo_overlap_ids=overlap,
        demo_overlap_ranks={human[case_id].search_rank for case_id in overlap},
        prediction_paths=prediction_paths,
        expected_membership_sha256=membership_sha256,
        expected_m4_demo_bank_id=m4_demo_bank_id,
    )
    retained = set(human)
    predictions: dict[str, dict[str, PredictionCase]] = {}
    for method in METHODS:
        expected = retained - overlap if method == "M4" else retained
        predictions[method] = load_predictions(
            method,
            prediction_paths[method],
            human,
            expected,
            expected_membership_sha256=membership_sha256,
            expected_m4_demo_bank_id=m4_demo_bank_id,
            expected_m4_demo_membership_sha256=m4_demo_membership_sha256,
        )
    tables, metadata = build_analysis(
        human,
        silver,
        predictions,
        overlap,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    metadata = {
        **metadata,
        "status": "COMPLETE",
        "retained_membership_sha256": membership_sha256,
        "membership_manifest_sha256": _sha256_file(args.membership_manifest),
    }
    report = render_report(
        tables, metadata, source_manifest, qc_summary, execution_metadata
    )
    inputs = {
        str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path): _sha256_file(path)
        for path in (
            args.human_reference,
            args.reference_key,
            args.management_sample,
            args.benchmark,
            args.leakage_audit,
            args.demo_bank,
            args.source_manifest,
            args.qc_summary,
            args.membership_manifest,
            frozen_membership_path,
            args.evaluation_a_baseline,
            args.m1_metadata,
            args.m2_metadata,
            args.m3_diagnostics,
            args.m4_diagnostics,
            raw_annotation_source,
            *prediction_paths.values(),
        )
    }
    _write_outputs(
        tables,
        metadata,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        report_path=args.report,
        report_text=report,
        input_hashes=inputs,
    )
    validate_evaluation_a_integrity(args.evaluation_a_baseline)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-reference", type=Path, default=DEFAULT_HUMAN_REFERENCE)
    parser.add_argument("--reference-key", type=Path, default=DEFAULT_REFERENCE_KEY)
    parser.add_argument("--management-sample", type=Path, default=DEFAULT_MANAGEMENT_SAMPLE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--leakage-audit", type=Path, default=DEFAULT_LEAKAGE_AUDIT)
    parser.add_argument("--demo-bank", type=Path, default=DEFAULT_DEMO_BANK)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--qc-summary", type=Path, default=DEFAULT_QC_SUMMARY)
    parser.add_argument(
        "--membership-manifest", type=Path, default=DEFAULT_MEMBERSHIP_MANIFEST
    )
    parser.add_argument(
        "--evaluation-a-baseline", type=Path, default=DEFAULT_EVAL_A_BASELINE
    )
    parser.add_argument("--m1-metadata", type=Path, default=DEFAULT_M1_METADATA)
    parser.add_argument("--m2-metadata", type=Path, default=DEFAULT_M2_METADATA)
    parser.add_argument("--m3-diagnostics", type=Path, default=DEFAULT_M3_DIAGNOSTICS)
    parser.add_argument("--m4-diagnostics", type=Path, default=DEFAULT_M4_DIAGNOSTICS)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Explicit prediction artifact; provide M1 through M4 or use defaults.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--capture-evaluation-a-baseline",
        action="store_true",
        help="Create the immutable pre-Evaluation-B integrity manifest and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.capture_evaluation_a_baseline:
            snapshot = capture_evaluation_a_integrity_baseline(
                args.evaluation_a_baseline
            )
            print(
                json.dumps(
                    {
                        "path": str(args.evaluation_a_baseline),
                        "scopes": {
                            name: value["aggregate_sha256"]
                            for name, value in snapshot["scopes"].items()
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        metadata = run_analysis(args)
    except EvaluationBAnalysisError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
