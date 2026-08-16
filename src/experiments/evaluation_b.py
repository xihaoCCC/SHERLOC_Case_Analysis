#!/usr/bin/env python3
"""Reusable, label-preserving utilities for the future Evaluation B workflow.

This module prepares human-annotation quality control, two-reviewer agreement,
adjudication, human-grounded-gold construction, silver/human comparison, and
human-gold evaluation.  Importing it has no side effects: it never selects
reliability cases, reads reviewer annotations, evaluates a model, or calls an
external service.

Reviewer-entered cells are always retained verbatim.  Exact frozen machine IDs
and exact SHERLOC-style raw AMP strings are both accepted; validated values are
mapped to machine IDs in frozen ontology order only in separate normalized
fields or downstream derived artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # package/notebook import
    from .metrics import (
        AMP_FAMILIES,
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        compute_amp_cpmr,
        compute_amp_metrics,
        compute_case_errors,
        labels_to_indicator,
    )
except ImportError:  # direct script/test import from src/experiments on sys.path
    from metrics import (
        AMP_FAMILIES,
        AMP_FAMILY_BY_LABEL,
        AMP_LABEL_IDS,
        compute_amp_cpmr,
        compute_amp_metrics,
        compute_case_errors,
        labels_to_indicator,
    )


VERSION = "1.0.0"

EXPECTED_A1_FINAL_SHA256 = "63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d"
EXPECTED_A2_FINAL_SHA256 = "75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67"
EXPECTED_RELIABILITY_SAMPLE_SHA256 = "ff825d1996ff55a72030ef07835c3b71c318df4f43a40ecb79714c27192794bc"

AMP_RAW_LABEL_BY_ID: dict[str, str] = {
    "ACT_RECRUITMENT": "Recruitment",
    "ACT_TRANSPORTATION": "Transportation",
    "ACT_TRANSFER": "Transfer",
    "ACT_HARBOURING": "Harbouring",
    "ACT_RECEIPT": "Receipt",
    "MEANS_THREAT_FORCE_OR_COERCION": (
        "Threat or use of force or other forms of coercion"
    ),
    "MEANS_ABDUCTION": "Abduction",
    "MEANS_FRAUD": "Fraud",
    "MEANS_DECEPTION": "Deception",
    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": (
        "Abuse of power or a position of vulnerability"
    ),
    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": (
        "Giving or receiving payments or benefits to achieve the consent of a "
        "person having control over another person"
    ),
    "PURPOSE_SEXUAL_EXPLOITATION": (
        "Exploitation of the prostitution of others or other forms of sexual "
        "exploitation"
    ),
    "PURPOSE_FORCED_LABOUR_OR_SERVICES": "Forced labour or services",
    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": (
        "Slavery or practices similar to slavery"
    ),
    "PURPOSE_SERVITUDE": "Servitude",
    "PURPOSE_REMOVAL_OF_ORGANS": "Removal of organs",
    "PURPOSE_OTHER": "Other",
}
AMP_ID_BY_RAW_LABEL = {value: key for key, value in AMP_RAW_LABEL_BY_ID.items()}
AMP_LABEL_ORDER = {label: index for index, label in enumerate(AMP_LABEL_IDS)}
AMP_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    family: tuple(
        label for label in AMP_LABEL_IDS if AMP_FAMILY_BY_LABEL[label] == family
    )
    for family in AMP_FAMILIES
}

ANSWERABILITY_VALUES = ("YES", "PARTIAL", "NO")
FORM_VALUES = ("INTERNAL", "TRANSNATIONAL", "BOTH", "UNKNOWN")
MULTIPLICITY_VALUES = ("SINGLE", "MULTIPLE", "UNKNOWN")
CHILD_VALUES = ("TRUE", "FALSE", "UNKNOWN")
SUFFICIENCY_VALUES = ("HIGH", "MODERATE", "LOW")

# The current guideline uses machine values.  These exact aliases accommodate
# the raw reviewer-facing representation described in the Evaluation B plan.
# No case-folding or fuzzy correction is performed.
FORM_VALUE_MAP = {
    **{value: value for value in FORM_VALUES},
    "Internal": "INTERNAL",
    "Transnational": "TRANSNATIONAL",
    "both": "BOTH",
}

FAMILY_COLUMN = {"ACT": "act_labels", "MEANS": "means_labels", "PURPOSE": "purpose_labels"}
FAMILY_ANSWERABILITY_COLUMN = {
    "ACT": "act_answerability",
    "MEANS": "means_answerability",
    "PURPOSE": "purpose_answerability",
}

TARGET_PREFIXES = ("act", "means", "purpose", "form", "multiplicity", "child")
ANSWERABILITY_FIELDS = tuple(f"{target}_answerability" for target in TARGET_PREFIXES)
EVIDENCE_FIELDS = tuple(f"{target}_evidence_sentence_ids" for target in TARGET_PREFIXES)
LABEL_FIELDS = (
    "act_labels",
    "means_labels",
    "purpose_labels",
    "form_label",
    "multiplicity_label",
    "child_label",
)
STRUCTURED_GOLD_FIELDS = LABEL_FIELDS + ANSWERABILITY_FIELDS + EVIDENCE_FIELDS + (
    "overall_narrative_sufficiency",
)

SENTENCE_ID_RE = re.compile(r"S([1-9][0-9]*)\Z")
NUMBERED_SENTENCE_RE = re.compile(
    r"(?:^|\n)\[S([1-9][0-9]*)\]\s*(.*?)(?=(?:\n\[S[1-9][0-9]*\])|\Z)",
    re.DOTALL,
)


class EvaluationBError(RuntimeError):
    """Raised when a future Evaluation B artifact cannot be built safely."""


@dataclass(frozen=True)
class AnnotationQCResult:
    """QC issues plus non-destructive normalized views of reviewer rows."""

    issues: list[dict[str, Any]]
    normalized_rows: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return not any(issue["severity"] == "ERROR" for issue in self.issues)


@dataclass(frozen=True)
class KappaResult:
    value: float | None
    status: str


def load_csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or (list(rows[0]) if rows else []))
    if not columns:
        raise EvaluationBError(f"Cannot write a CSV without a declared schema: {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
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


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _is_blank(value: Any) -> bool:
    return not _as_text(value).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(row: Mapping[str, Any]) -> str:
    for field in ("reliability_case_id", "case_id", "search_rank", "canonical_url"):
        value = _as_text(row.get(field)).strip()
        if value:
            return value
    raise EvaluationBError("Row lacks reliability_case_id, case_id, search_rank, and canonical_url")


def _index_unique(rows: Sequence[Mapping[str, Any]], *, artifact: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = _case_id(row)
        if case_id in result:
            raise EvaluationBError(f"Duplicate case ID {case_id!r} in {artifact}")
        result[case_id] = row
    return result


def _index_on_field(
    rows: Sequence[Mapping[str, Any]], *, field: str, artifact: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = _as_text(row.get(field)).strip()
        if not value:
            raise EvaluationBError(f"Missing {field} in {artifact}")
        if value in result:
            raise EvaluationBError(f"Duplicate {field}={value!r} in {artifact}")
        result[value] = row
    return result


def _align_artifacts(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left_name: str,
    right_name: str,
    allow_right_superset: bool = False,
) -> tuple[str, dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Align artifacts by neutral ID, then rank/URL when predictions lack it."""

    for field in ("reliability_case_id", "search_rank", "canonical_url"):
        if not all(_as_text(row.get(field)).strip() for row in (*left_rows, *right_rows)):
            continue
        left = _index_on_field(left_rows, field=field, artifact=left_name)
        right = _index_on_field(right_rows, field=field, artifact=right_name)
        membership_ok = set(left) <= set(right) if allow_right_superset else set(left) == set(right)
        if membership_ok:
            return field, left, right
    relationship = "a containing" if allow_right_superset else "an exact"
    raise EvaluationBError(
        f"Could not align {left_name} and {right_name} on reliability_case_id, search_rank, "
        f"or canonical_url with {relationship} membership match"
    )


def _split_list_cell(value: Any) -> tuple[list[str], str | None]:
    """Parse a JSON array, Python sequence, or guideline semicolon list."""

    if value is None:
        return [], None
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            return [], "List contains a non-string item"
        return [item.strip() for item in value if item.strip()], None
    if not isinstance(value, str):
        return [], f"Expected a string/list, found {type(value).__name__}"
    raw = value.strip()
    if not raw:
        return [], None
    if raw.startswith("[") or raw.startswith("{"):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [], f"Malformed JSON list: {exc.msg}"
        if not isinstance(decoded, list):
            return [], "JSON multi-label value is not a list"
        if not all(isinstance(item, str) for item in decoded):
            return [], "JSON list contains a non-string item"
        return [item.strip() for item in decoded if item.strip()], None
    return [item.strip() for item in raw.split(";") if item.strip()], None


def parse_amp_labels(value: Any, family: str) -> tuple[list[str], list[str]]:
    """Validate and map one AMP cell without changing the supplied value.

    Returns ``(canonical_ids, error_messages)``.  Both exact machine IDs and
    exact raw SHERLOC strings are accepted.  IDs are returned in frozen order.
    Duplicate presentation is an error even when two different strings map to
    the same machine ID.
    """

    family = family.upper()
    if family not in AMP_IDS_BY_FAMILY:
        raise ValueError(f"Unknown AMP family: {family}")
    items, parse_error = _split_list_cell(value)
    if parse_error:
        return [], [parse_error]
    errors: list[str] = []
    if len(items) != len(set(items)):
        errors.append("Duplicate label presentation")
    mapped: list[str] = []
    allowed = set(AMP_IDS_BY_FAMILY[family])
    for item in items:
        label_id = item if item in AMP_LABEL_ORDER else AMP_ID_BY_RAW_LABEL.get(item)
        if label_id is None:
            errors.append(f"Unknown AMP label: {item}")
        elif label_id not in allowed:
            errors.append(f"AMP label belongs to another family: {item}")
        else:
            mapped.append(label_id)
    if len(mapped) != len(set(mapped)):
        errors.append("Duplicate labels after exact raw-to-ID mapping")
    return sorted(set(mapped), key=AMP_LABEL_ORDER.__getitem__), errors


def parse_evidence_sentence_ids(value: Any) -> list[str]:
    """Parse a semicolon/JSON sentence-ID list and reject malformed entries."""

    if isinstance(value, str) and value.strip().upper() in {"NONE", "[]"}:
        return []
    items, parse_error = _split_list_cell(value)
    if parse_error:
        raise EvaluationBError(parse_error)
    if len(items) != len(set(items)):
        raise EvaluationBError("Duplicate evidence sentence IDs")
    invalid = [item for item in items if not SENTENCE_ID_RE.fullmatch(item)]
    if invalid:
        raise EvaluationBError(f"Malformed evidence sentence IDs: {invalid}")
    numbers = [int(item[1:]) for item in items]
    if numbers != sorted(numbers):
        raise EvaluationBError("Evidence sentence IDs are not in ascending order")
    return items


def extract_numbered_sentences(numbered_text: str) -> dict[str, str]:
    """Return the stable sentence-ID to text mapping in a reviewer narrative."""

    result: dict[str, str] = {}
    for match in NUMBERED_SENTENCE_RE.finditer(_as_text(numbered_text).replace("\r\n", "\n").replace("\r", "\n")):
        sentence_id = f"S{int(match.group(1))}"
        if sentence_id in result:
            raise EvaluationBError(f"Duplicate numbered sentence marker: {sentence_id}")
        result[sentence_id] = " ".join(match.group(2).split())
    return result


def validate_evidence_sentence_ids(
    value: Any, available_sentence_ids: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Return parsed evidence IDs and any missing-ID messages."""

    try:
        parsed = parse_evidence_sentence_ids(value)
    except EvaluationBError as exc:
        return [], [str(exc)]
    available = set(available_sentence_ids)
    missing = [item for item in parsed if item not in available]
    return parsed, ([f"Evidence sentence ID not present in case: {item}" for item in missing])


def map_evidence_ids_to_text(
    evidence_ids: Any, numbered_text_or_mapping: str | Mapping[str, str]
) -> list[dict[str, str]]:
    """Map validated evidence IDs back to supplied narrative sentences."""

    mapping = (
        dict(numbered_text_or_mapping)
        if isinstance(numbered_text_or_mapping, Mapping)
        else extract_numbered_sentences(numbered_text_or_mapping)
    )
    ids, errors = validate_evidence_sentence_ids(evidence_ids, mapping)
    if errors:
        raise EvaluationBError("; ".join(errors))
    return [{"sentence_id": item, "text": mapping[item]} for item in ids]


def answerability_is_abstention(value: Any) -> bool:
    text = _as_text(value).strip()
    if text not in ANSWERABILITY_VALUES:
        raise EvaluationBError(f"Invalid answerability value: {text!r}")
    return text == "NO"


def selective_evaluation_mask(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    *,
    include_answerability: Sequence[str] = ("YES", "PARTIAL"),
) -> list[bool]:
    """Build a family/target-specific mask without inventing new fields."""

    field = f"{target.lower()}_answerability"
    allowed = set(include_answerability)
    unknown = allowed - set(ANSWERABILITY_VALUES)
    if unknown:
        raise EvaluationBError(f"Unknown answerability mask values: {sorted(unknown)}")
    result: list[bool] = []
    for row in rows:
        value = _as_text(row.get(field)).strip()
        if value not in ANSWERABILITY_VALUES:
            raise EvaluationBError(f"Invalid {field} for case {_case_id(row)}: {value!r}")
        result.append(value in allowed)
    return result


def qc_annotations(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str] | None = None,
    sentence_map_by_case: Mapping[str, Iterable[str] | Mapping[str, str]] | None = None,
    reviewer_id_required: bool = True,
) -> AnnotationQCResult:
    """Validate one reviewer file without mutating or overwriting any cell.

    ``expected_case_ids`` is optional so an explicitly chosen final subset of
    any size can be checked later.  It must come from a researcher-supplied
    final list; this function never selects or resamples cases.
    """

    issues: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []

    def issue(
        row_number: int | None,
        case_id: str,
        field: str,
        code: str,
        message: str,
        *,
        severity: str = "ERROR",
        entered_value: Any = "",
    ) -> None:
        issues.append(
            {
                "row_number": row_number if row_number is not None else "",
                "reliability_case_id": case_id,
                "field": field,
                "severity": severity,
                "code": code,
                "message": message,
                "entered_value": _as_text(entered_value),
            }
        )

    seen: dict[str, int] = {}
    observed_ids: list[str] = []
    reviewer_ids: set[str] = set()
    expected = set(expected_case_ids) if expected_case_ids is not None else None

    for row_number, supplied in enumerate(rows, start=2):
        row = dict(supplied)
        case_id = _as_text(row.get("reliability_case_id")).strip()
        if not case_id:
            issue(row_number, "", "reliability_case_id", "MISSING_CASE_ID", "Missing reliability case ID")
            case_id = f"<ROW_{row_number}>"
        elif case_id in seen:
            issue(
                row_number,
                case_id,
                "reliability_case_id",
                "DUPLICATE_CASE_ID",
                f"Duplicate case ID; first seen on CSV row {seen[case_id]}",
                entered_value=case_id,
            )
        else:
            seen[case_id] = row_number
            observed_ids.append(case_id)

        if expected is not None and case_id not in expected:
            issue(
                row_number,
                case_id,
                "reliability_case_id",
                "UNEXPECTED_CASE_ID",
                "Case is not in the supplied final expected-case list",
                entered_value=case_id,
            )

        reviewer_id = _as_text(row.get("reviewer_id")).strip()
        if reviewer_id:
            reviewer_ids.add(reviewer_id)
        elif reviewer_id_required:
            issue(row_number, case_id, "reviewer_id", "MISSING_REVIEWER_ID", "Reviewer ID is required")

        normalized = dict(row)  # reviewer-entered fields remain byte-for-byte values
        for family in AMP_FAMILIES:
            field = FAMILY_COLUMN[family]
            canonical, errors = parse_amp_labels(row.get(field, ""), family)
            normalized[f"{field}_normalized"] = canonical
            for error in errors:
                code = (
                    "MALFORMED_LIST"
                    if error.startswith(("Malformed JSON", "JSON multi-label", "JSON list", "Expected"))
                    else "DUPLICATE_LABEL"
                    if error.startswith("Duplicate")
                    else "INVALID_AMP_LABEL"
                )
                issue(row_number, case_id, field, code, error, entered_value=row.get(field, ""))

            answerability_field = FAMILY_ANSWERABILITY_COLUMN[family]
            answerability = _as_text(row.get(answerability_field)).strip()
            if answerability not in ANSWERABILITY_VALUES:
                issue(
                    row_number,
                    case_id,
                    answerability_field,
                    "INVALID_ANSWERABILITY",
                    f"Expected one of {ANSWERABILITY_VALUES}",
                    entered_value=row.get(answerability_field, ""),
                )
            elif answerability in {"YES", "PARTIAL"} and not canonical:
                issue(
                    row_number,
                    case_id,
                    field,
                    "EMPTY_ANSWERABLE_AMP_FAMILY",
                    f"{family} labels must be nonempty when answerability is {answerability}",
                )
            elif answerability == "NO" and canonical:
                issue(
                    row_number,
                    case_id,
                    field,
                    "LABEL_PRESENT_WHEN_UNANSWERABLE",
                    f"{family} labels must be empty when answerability is NO",
                    entered_value=row.get(field, ""),
                )

        for target in ("form", "multiplicity", "child"):
            label_field = f"{target}_label"
            raw_value = _as_text(row.get(label_field)).strip()
            if target == "form":
                # A blank raw Form cell is accepted only when the reviewer has
                # explicitly marked the target NO, as described in the plan.
                normalized_value = FORM_VALUE_MAP.get(raw_value)
                if raw_value == "" and _as_text(row.get("form_answerability")).strip() == "NO":
                    normalized_value = "UNKNOWN"
                allowed_display = FORM_VALUES
            elif target == "multiplicity":
                normalized_value = raw_value if raw_value in MULTIPLICITY_VALUES else None
                allowed_display = MULTIPLICITY_VALUES
            else:
                normalized_value = raw_value if raw_value in CHILD_VALUES else None
                allowed_display = CHILD_VALUES
            normalized[f"{label_field}_normalized"] = normalized_value
            if normalized_value is None:
                issue(
                    row_number,
                    case_id,
                    label_field,
                    f"INVALID_{target.upper()}_VALUE",
                    f"Expected one of {allowed_display}",
                    entered_value=row.get(label_field, ""),
                )

            answerability_field = f"{target}_answerability"
            answerability = _as_text(row.get(answerability_field)).strip()
            if answerability not in ANSWERABILITY_VALUES:
                issue(
                    row_number,
                    case_id,
                    answerability_field,
                    "INVALID_ANSWERABILITY",
                    f"Expected one of {ANSWERABILITY_VALUES}",
                    entered_value=row.get(answerability_field, ""),
                )
            elif answerability == "NO" and normalized_value != "UNKNOWN":
                issue(
                    row_number,
                    case_id,
                    label_field,
                    "NONUNKNOWN_WHEN_UNANSWERABLE",
                    f"{target.title()} must be UNKNOWN when answerability is NO",
                    entered_value=row.get(label_field, ""),
                )

        for target in TARGET_PREFIXES:
            field = f"{target}_evidence_sentence_ids"
            if sentence_map_by_case is not None and case_id in sentence_map_by_case:
                supplied_map = sentence_map_by_case[case_id]
                available = supplied_map.keys() if isinstance(supplied_map, Mapping) else supplied_map
            else:
                available = extract_numbered_sentences(_as_text(row.get("fact_summary_numbered"))).keys()
            parsed_ids, evidence_errors = validate_evidence_sentence_ids(row.get(field, ""), available)
            normalized[f"{field}_normalized"] = parsed_ids
            for error in evidence_errors:
                code = "EVIDENCE_ID_NOT_IN_CASE" if "not present" in error else "MALFORMED_EVIDENCE_IDS"
                issue(row_number, case_id, field, code, error, entered_value=row.get(field, ""))

        sufficiency = _as_text(row.get("overall_narrative_sufficiency")).strip()
        normalized["overall_narrative_sufficiency_normalized"] = sufficiency or None
        if sufficiency and sufficiency not in SUFFICIENCY_VALUES:
            issue(
                row_number,
                case_id,
                "overall_narrative_sufficiency",
                "INVALID_NARRATIVE_SUFFICIENCY",
                f"Expected one of {SUFFICIENCY_VALUES} or blank",
                entered_value=sufficiency,
            )

        for field, value in row.items():
            lowered = field.lower()
            is_adjudication_field = (
                lowered.startswith("adjudicated_")
                or lowered.startswith("final_adjudicated_")
                or lowered.endswith("_adjudicated")
            )
            if is_adjudication_field and not _is_blank(value):
                issue(
                    row_number,
                    case_id,
                    field,
                    "POPULATED_ADJUDICATION_FIELD",
                    "Reviewer-specific input must not contain adjudicated values",
                    entered_value=value,
                )

        normalized_rows.append(normalized)

    if len(reviewer_ids) > 1:
        issue(
            None,
            "",
            "reviewer_id",
            "MULTIPLE_REVIEWER_IDS",
            f"One reviewer file contains multiple reviewer IDs: {sorted(reviewer_ids)}",
        )

    if expected is not None:
        missing = sorted(expected - set(observed_ids))
        for case_id in missing:
            issue(
                None,
                case_id,
                "reliability_case_id",
                "MISSING_EXPECTED_CASE_ID",
                "Expected case is absent from reviewer file",
            )

    error_count = sum(issue_row["severity"] == "ERROR" for issue_row in issues)
    warning_count = sum(issue_row["severity"] == "WARNING" for issue_row in issues)
    summary = {
        "schema_version": VERSION,
        "row_count": len(rows),
        "unique_case_id_count": len(seen),
        "expected_case_id_count": len(expected) if expected is not None else None,
        "reviewer_ids": sorted(reviewer_ids),
        "error_count": error_count,
        "warning_count": warning_count,
        "issue_count": len(issues),
        "status": "PASS" if error_count == 0 else "FAIL",
        "human_values_modified": False,
    }
    return AnnotationQCResult(issues=issues, normalized_rows=normalized_rows, summary=summary)


QC_ISSUE_COLUMNS = (
    "row_number",
    "reliability_case_id",
    "field",
    "severity",
    "code",
    "message",
    "entered_value",
)


def write_qc_reports(
    result: AnnotationQCResult,
    *,
    machine_path: Path | str,
    human_path: Path | str,
) -> None:
    """Write explicit QC diagnostics; never write back into reviewer input."""

    machine = Path(machine_path)
    human = Path(human_path)
    _atomic_csv(machine, result.issues, fieldnames=QC_ISSUE_COLUMNS)
    lines = [
        "# Evaluation B annotation QC report",
        "",
        f"Status: **{result.summary['status']}**",
        f"Rows: {result.summary['row_count']}",
        f"Unique case IDs: {result.summary['unique_case_id_count']}",
        f"Errors: {result.summary['error_count']}",
        f"Warnings: {result.summary['warning_count']}",
        "",
        "Reviewer-entered values were not modified.",
    ]
    if result.issues:
        lines.extend(["", "## Issues", ""])
        for item in result.issues:
            location = f"case {item['reliability_case_id'] or 'N/A'}, field {item['field']}"
            lines.append(f"- `{item['severity']} {item['code']}` ({location}): {item['message']}")
    else:
        lines.extend(["", "No QC issues were found."])
    _atomic_text(human, "\n".join(lines) + "\n")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> KappaResult:
    """Calculate Cohen's kappa, returning explicit N/A for degenerate data."""

    if len(left) != len(right):
        raise EvaluationBError("Kappa inputs must have the same length")
    if not left:
        return KappaResult(None, "NO_COMMON_CASES")
    categories = sorted(set(left) | set(right), key=str)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[value] / len(left)) * (right_counts[value] / len(right))
        for value in categories
    )
    if math.isclose(1.0 - expected, 0.0, abs_tol=1e-15):
        return KappaResult(None, "DEGENERATE_NO_VARIATION")
    return KappaResult(float((observed - expected) / (1.0 - expected)), "OK")


def _strict_annotation_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized structured values or raise on an invalid row."""

    result: dict[str, Any] = {}
    case_id = _case_id(row)
    for family in AMP_FAMILIES:
        field = FAMILY_COLUMN[family]
        canonical, errors = parse_amp_labels(row.get(field, ""), family)
        if errors:
            raise EvaluationBError(f"Invalid {field} for {case_id}: {'; '.join(errors)}")
        result[field] = tuple(canonical)

    for target, values in (
        ("form", FORM_VALUES),
        ("multiplicity", MULTIPLICITY_VALUES),
        ("child", CHILD_VALUES),
    ):
        field = f"{target}_label"
        raw = _as_text(row.get(field)).strip()
        if target == "form":
            value = FORM_VALUE_MAP.get(raw)
            if raw == "" and _as_text(row.get("form_answerability")).strip() == "NO":
                value = "UNKNOWN"
        else:
            value = raw if raw in values else None
        if value is None:
            raise EvaluationBError(f"Invalid {field} for {case_id}: {raw!r}")
        result[field] = value

    for field in ANSWERABILITY_FIELDS:
        value = _as_text(row.get(field)).strip()
        if value not in ANSWERABILITY_VALUES:
            raise EvaluationBError(f"Invalid {field} for {case_id}: {value!r}")
        result[field] = value

    for field in EVIDENCE_FIELDS:
        result[field] = tuple(parse_evidence_sentence_ids(row.get(field, "")))

    sufficiency = _as_text(row.get("overall_narrative_sufficiency")).strip()
    if sufficiency and sufficiency not in SUFFICIENCY_VALUES:
        raise EvaluationBError(
            f"Invalid overall_narrative_sufficiency for {case_id}: {sufficiency!r}"
        )
    result["overall_narrative_sufficiency"] = sufficiency or None
    _validate_structured_consistency(result, case_id=case_id)
    return result


def _validate_structured_consistency(view: Mapping[str, Any], *, case_id: str) -> None:
    """Fail closed on cross-field rules from the frozen annotation guide."""

    for family in AMP_FAMILIES:
        labels = tuple(view[FAMILY_COLUMN[family]])
        answerability = view[FAMILY_ANSWERABILITY_COLUMN[family]]
        if answerability in {"YES", "PARTIAL"} and not labels:
            raise EvaluationBError(
                f"{family} must be nonempty for answerability {answerability} in {case_id}"
            )
        if answerability == "NO" and labels:
            raise EvaluationBError(
                f"{family} must be empty for answerability NO in {case_id}"
            )
    for target in ("form", "multiplicity", "child"):
        if view[f"{target}_answerability"] == "NO" and view[f"{target}_label"] != "UNKNOWN":
            raise EvaluationBError(
                f"{target} must be UNKNOWN for answerability NO in {case_id}"
            )


def compute_reviewer_agreement(
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two locked reviewer files over their exact common case set."""

    by_a = _index_unique(reviewer_a_rows, artifact="Reviewer A annotations")
    by_b = _index_unique(reviewer_b_rows, artifact="Reviewer B annotations")
    common = sorted(set(by_a) & set(by_b))
    if not common:
        raise EvaluationBError("Reviewer files have no common cases")
    view_a = {case_id: _strict_annotation_view(by_a[case_id]) for case_id in common}
    view_b = {case_id: _strict_annotation_view(by_b[case_id]) for case_id in common}

    summary: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []

    for family in AMP_FAMILIES:
        field = FAMILY_COLUMN[family]
        sets_a = [set(view_a[case_id][field]) for case_id in common]
        sets_b = [set(view_b[case_id][field]) for case_id in common]
        exact = [left == right for left, right in zip(sets_a, sets_b, strict=True)]
        jaccards = [_jaccard(left, right) for left, right in zip(sets_a, sets_b, strict=True)]
        summary.append(
            {
                "target": family,
                "metric_type": "MULTILABEL_SET",
                "common_case_n": len(common),
                "exact_agreement_count": sum(exact),
                "exact_agreement": sum(exact) / len(common),
                "mean_jaccard": sum(jaccards) / len(common),
                "any_disagreement_count": len(common) - sum(exact),
                "any_disagreement_proportion": 1.0 - (sum(exact) / len(common)),
                "cohen_kappa": None,
                "kappa_status": "NOT_APPLICABLE_TO_MULTILABEL_SET",
            }
        )
        for label in AMP_IDS_BY_FAMILY[family]:
            binary_a = [int(label in values) for values in sets_a]
            binary_b = [int(label in values) for values in sets_b]
            both_positive = sum(a == b == 1 for a, b in zip(binary_a, binary_b, strict=True))
            positive_denominator = sum(binary_a) + sum(binary_b)
            kappa = cohen_kappa(binary_a, binary_b)
            per_label.append(
                {
                    "family": family,
                    "label_id": label,
                    "raw_label": AMP_RAW_LABEL_BY_ID[label],
                    "common_case_n": len(common),
                    "reviewer_a_support": sum(binary_a),
                    "reviewer_b_support": sum(binary_b),
                    "raw_agreement": sum(a == b for a, b in zip(binary_a, binary_b, strict=True)) / len(common),
                    "positive_agreement": (
                        (2 * both_positive) / positive_denominator
                        if positive_denominator
                        else None
                    ),
                    "cohen_kappa": kappa.value,
                    "kappa_status": kappa.status,
                }
            )

    categorical_targets = [
        *[(field, ANSWERABILITY_VALUES) for field in ANSWERABILITY_FIELDS],
        ("form_label", FORM_VALUES),
        ("multiplicity_label", MULTIPLICITY_VALUES),
        ("child_label", CHILD_VALUES),
    ]
    for field, categories in categorical_targets:
        values_a = [view_a[case_id][field] for case_id in common]
        values_b = [view_b[case_id][field] for case_id in common]
        exact_count = sum(a == b for a, b in zip(values_a, values_b, strict=True))
        kappa = cohen_kappa(values_a, values_b)
        summary.append(
            {
                "target": field,
                "metric_type": "CATEGORICAL",
                "common_case_n": len(common),
                "exact_agreement_count": exact_count,
                "exact_agreement": exact_count / len(common),
                "mean_jaccard": None,
                "any_disagreement_count": len(common) - exact_count,
                "any_disagreement_proportion": 1.0 - (exact_count / len(common)),
                "cohen_kappa": kappa.value,
                "kappa_status": kappa.status,
            }
        )
        for value_a in categories:
            for value_b in categories:
                confusion.append(
                    {
                        "target": field,
                        "reviewer_a_value": value_a,
                        "reviewer_b_value": value_b,
                        "count": sum(
                            a == value_a and b == value_b
                            for a, b in zip(values_a, values_b, strict=True)
                        ),
                    }
                )

    full_agreement_ids = [
        case_id
        for case_id in common
        if all(view_a[case_id][field] == view_b[case_id][field] for field in STRUCTURED_GOLD_FIELDS)
    ]
    return {
        "metadata": {
            "schema_version": VERSION,
            "total_common_cases": len(common),
            "reviewer_a_only_cases": sorted(set(by_a) - set(by_b)),
            "reviewer_b_only_cases": sorted(set(by_b) - set(by_a)),
            "full_agreement_cases": len(full_agreement_ids),
            "any_disagreement_cases": len(common) - len(full_agreement_ids),
            "full_agreement_case_ids": full_agreement_ids,
            "any_disagreement_case_ids": [item for item in common if item not in set(full_agreement_ids)],
        },
        "summary": summary,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def build_disagreement_queue(
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    *,
    case_context_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a human-only queue containing exactly structured disagreements.

    Silver-reference labels and model predictions are neither accepted nor
    consulted.  Every adjudicated field is blank; this function never chooses
    a reviewer or generates an adjudicated value.
    """

    by_a = _index_unique(reviewer_a_rows, artifact="Reviewer A annotations")
    by_b = _index_unique(reviewer_b_rows, artifact="Reviewer B annotations")
    context = (
        _index_unique(case_context_rows, artifact="reliability case context")
        if case_context_rows is not None
        else {}
    )
    common = sorted(set(by_a) & set(by_b))
    queue: list[dict[str, Any]] = []
    for case_id in common:
        view_a = _strict_annotation_view(by_a[case_id])
        view_b = _strict_annotation_view(by_b[case_id])
        disagreements = [
            field for field in STRUCTURED_GOLD_FIELDS if view_a[field] != view_b[field]
        ]
        if not disagreements:
            continue
        ctx = context.get(case_id, {})
        row: dict[str, Any] = {
            "reliability_case_id": case_id,
            "search_rank": ctx.get("search_rank", ""),
            "jurisdiction": ctx.get("jurisdiction_raw", ctx.get("jurisdiction", "")),
            "fact_summary": ctx.get("english_fact_summary_raw", ""),
            "fact_summary_numbered": ctx.get(
                "fact_summary_numbered", by_a[case_id].get("fact_summary_numbered", "")
            ),
            "disagreement_fields": ";".join(disagreements),
        }
        for field in STRUCTURED_GOLD_FIELDS:
            row[f"reviewer_a_{field}"] = by_a[case_id].get(field, "")
            row[f"reviewer_b_{field}"] = by_b[case_id].get(field, "")
            row[f"adjudicated_{field}"] = ""
        # Notes are evidence context, not automatically adjudicated targets.
        for target in TARGET_PREFIXES:
            notes_field = f"{target}_notes"
            row[f"reviewer_a_{notes_field}"] = by_a[case_id].get(notes_field, "")
            row[f"reviewer_b_{notes_field}"] = by_b[case_id].get(notes_field, "")
        row["reviewer_a_annotation_notes"] = by_a[case_id].get("annotation_notes", "")
        row["reviewer_b_annotation_notes"] = by_b[case_id].get("annotation_notes", "")
        row["adjudication_notes"] = ""
        queue.append(row)
    return queue


def _parse_adjudicated_value(field: str, value: Any, *, case_id: str) -> Any:
    raw = _as_text(value).strip()
    if field in ("act_labels", "means_labels", "purpose_labels"):
        if not raw:
            raise EvaluationBError(
                f"Unresolved {field} disagreement for {case_id}; use [] for an explicit empty set"
            )
        family = field.split("_", 1)[0].upper()
        labels, errors = parse_amp_labels(value, family)
        if errors:
            raise EvaluationBError(
                f"Invalid adjudicated {field} for {case_id}: {'; '.join(errors)}"
            )
        return tuple(labels)
    if field in ANSWERABILITY_FIELDS:
        if raw not in ANSWERABILITY_VALUES:
            raise EvaluationBError(f"Unresolved/invalid adjudicated {field} for {case_id}")
        return raw
    if field == "form_label":
        if not raw or raw not in FORM_VALUE_MAP:
            raise EvaluationBError(f"Unresolved/invalid adjudicated form_label for {case_id}")
        return FORM_VALUE_MAP[raw]
    if field == "multiplicity_label":
        if raw not in MULTIPLICITY_VALUES:
            raise EvaluationBError(f"Unresolved/invalid adjudicated multiplicity_label for {case_id}")
        return raw
    if field == "child_label":
        if raw not in CHILD_VALUES:
            raise EvaluationBError(f"Unresolved/invalid adjudicated child_label for {case_id}")
        return raw
    if field in EVIDENCE_FIELDS:
        if not raw:
            raise EvaluationBError(
                f"Unresolved {field} disagreement for {case_id}; use NONE for explicit no evidence"
            )
        return tuple(parse_evidence_sentence_ids(value))
    if field == "overall_narrative_sufficiency":
        if raw not in SUFFICIENCY_VALUES:
            raise EvaluationBError(
                f"Unresolved/invalid adjudicated overall_narrative_sufficiency for {case_id}"
            )
        return raw
    raise EvaluationBError(f"Unsupported adjudicated field: {field}")


def _serialize_structured_value(field: str, value: Any) -> str:
    if field in ("act_labels", "means_labels", "purpose_labels"):
        return ";".join(value)
    if field in EVIDENCE_FIELDS:
        return ";".join(value)
    return "" if value is None else str(value)


def build_human_gold(
    reviewer_a_rows: Sequence[Mapping[str, Any]],
    reviewer_b_rows: Sequence[Mapping[str, Any]],
    adjudication_rows: Sequence[Mapping[str, Any]],
    *,
    case_context_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Combine reviewer agreement and explicit human adjudication.

    There is deliberately no majority, model-assisted, or silver-assisted
    fallback.  Every structured disagreement must have an explicit human
    adjudicated value before any gold row is emitted.
    """

    by_a = _index_unique(reviewer_a_rows, artifact="Reviewer A annotations")
    by_b = _index_unique(reviewer_b_rows, artifact="Reviewer B annotations")
    by_adj = _index_unique(adjudication_rows, artifact="adjudication rows")
    context = (
        _index_unique(case_context_rows, artifact="reliability case context")
        if case_context_rows is not None
        else {}
    )
    if set(by_a) != set(by_b):
        raise EvaluationBError("Reviewer files must contain the same final case IDs")

    output: list[dict[str, Any]] = []
    for case_id in sorted(by_a):
        view_a = _strict_annotation_view(by_a[case_id])
        view_b = _strict_annotation_view(by_b[case_id])
        disagreement_fields = [
            field for field in STRUCTURED_GOLD_FIELDS if view_a[field] != view_b[field]
        ]
        if disagreement_fields and case_id not in by_adj:
            raise EvaluationBError(f"Missing adjudication row for disagreement case {case_id}")

        ctx = context.get(case_id, {})
        row: dict[str, Any] = {
            "human_gold_schema_version": VERSION,
            "reliability_case_id": case_id,
            "search_rank": ctx.get("search_rank", by_a[case_id].get("search_rank", "")),
            "canonical_url": ctx.get("canonical_url", by_a[case_id].get("canonical_url", "")),
            "numbered_text_sha256": by_a[case_id].get("numbered_text_sha256", ""),
            "reviewer_a_id": by_a[case_id].get("reviewer_id", ""),
            "reviewer_b_id": by_b[case_id].get("reviewer_id", ""),
            "reviewer_a_raw_annotation_json": _canonical_json(dict(by_a[case_id])),
            "reviewer_b_raw_annotation_json": _canonical_json(dict(by_b[case_id])),
        }
        for field in STRUCTURED_GOLD_FIELDS:
            if view_a[field] == view_b[field]:
                final = view_a[field]
                source = "REVIEWER_AGREEMENT"
            else:
                final = _parse_adjudicated_value(
                    field,
                    by_adj[case_id].get(f"adjudicated_{field}", ""),
                    case_id=case_id,
                )
                source = "HUMAN_ADJUDICATION"
            row[field] = _serialize_structured_value(field, final)
            row[f"{field}_provenance"] = source
        final_view = {
            field: (
                tuple(row[field].split(";")) if field in ("act_labels", "means_labels", "purpose_labels") and row[field]
                else tuple() if field in ("act_labels", "means_labels", "purpose_labels")
                else row[field]
            )
            for field in STRUCTURED_GOLD_FIELDS
        }
        _validate_structured_consistency(final_view, case_id=case_id)
        row["adjudication_notes"] = (
            by_adj[case_id].get("adjudication_notes", "") if case_id in by_adj else ""
        )
        output.append(row)
    return output


def _labels_from_row(row: Mapping[str, Any], family: str, *, prediction: bool = False) -> list[str]:
    """Read canonical labels from supported future/common artifact shapes."""

    family = family.upper()
    if prediction and "predicted_labels" in row:
        values = row["predicted_labels"]
        if isinstance(values, str):
            items, error = _split_list_cell(values)
            if error:
                raise EvaluationBError(error)
        else:
            items = list(values)
        unknown = set(items) - set(AMP_LABEL_IDS)
        if unknown:
            raise EvaluationBError(f"Unknown predicted labels: {sorted(unknown)}")
        return [label for label in AMP_IDS_BY_FAMILY[family] if label in set(items)]

    field = FAMILY_COLUMN[family]
    if field in row:
        labels, errors = parse_amp_labels(row.get(field, ""), family)
        if errors:
            raise EvaluationBError(f"Invalid {field}: {'; '.join(errors)}")
        return labels

    raw_candidates = {
        "ACT": ("legacy_acts_raw_json", "act_ontology_ids"),
        "MEANS": ("legacy_means_raw_json", "means_ontology_ids"),
        "PURPOSE": ("legacy_purposes_raw_json", "purpose_ontology_ids"),
    }
    for candidate in raw_candidates[family]:
        if candidate in row:
            labels, errors = parse_amp_labels(row[candidate], family)
            if errors:
                raise EvaluationBError(f"Invalid {candidate}: {'; '.join(errors)}")
            return labels

    if "amp_targets" in row:
        target_key = {
            "ACT": "act_ontology_ids",
            "MEANS": "means_ontology_ids",
            "PURPOSE": "purpose_ontology_ids",
        }[family]
        labels, errors = parse_amp_labels(row["amp_targets"].get(target_key, []), family)
        if errors:
            raise EvaluationBError(f"Invalid amp_targets.{target_key}: {'; '.join(errors)}")
        return labels

    # Frozen split and other wide tables use one binary column per label.
    if all(label in row for label in AMP_IDS_BY_FAMILY[family]):
        labels: list[str] = []
        for label in AMP_IDS_BY_FAMILY[family]:
            value = row[label]
            if value in (1, True, "1", "true", "TRUE"):
                labels.append(label)
            elif value not in (0, False, "0", "false", "FALSE", ""):
                raise EvaluationBError(f"Non-binary value for {label}: {value!r}")
        return labels
    raise EvaluationBError(f"Cannot locate {family} labels in row {_case_id(row)}")


def compare_silver_to_human(
    human_gold_rows: Sequence[Mapping[str, Any]],
    silver_reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare silver-reference AMP with human-grounded narrative AMP."""

    match_field, human_by_id, silver_by_id = _align_artifacts(
        human_gold_rows,
        silver_reference_rows,
        left_name="human-grounded gold",
        right_name="silver reference",
        allow_right_superset=True,
    )
    common = sorted(human_by_id)

    summary: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    case_level: list[dict[str, Any]] = []
    for family in AMP_FAMILIES:
        silver_sets: list[set[str]] = []
        human_sets: list[set[str]] = []
        answerability: list[str] = []
        for case_id in common:
            silver_sets.append(set(_labels_from_row(silver_by_id[case_id], family)))
            human_sets.append(set(_labels_from_row(human_by_id[case_id], family)))
            field = FAMILY_ANSWERABILITY_COLUMN[family]
            value = _as_text(human_by_id[case_id].get(field)).strip()
            if value not in ANSWERABILITY_VALUES:
                raise EvaluationBError(f"Invalid {field} for {case_id}: {value!r}")
            answerability.append(value)

        exact_values: list[int] = []
        jaccards: list[float] = []
        total_shared = total_silver_only = total_human_only = 0
        category_counts: Counter[str] = Counter()
        for case_id, silver, human, answer in zip(
            common, silver_sets, human_sets, answerability, strict=True
        ):
            shared = silver & human
            silver_only = silver - human
            human_only = human - silver
            if answer == "NO":
                category = "HUMAN_FAMILY_UNANSWERABLE"
            elif silver == human:
                category = "SILVER_EQUALS_HUMAN"
            elif silver and human and human < silver:
                category = "SILVER_BROADER_THAN_HUMAN"
            elif silver and human and silver < human:
                category = "HUMAN_BROADER_THAN_SILVER"
            elif shared:
                category = "PARTIAL_OVERLAP_ADDITIONS_BOTH_SIDES"
            else:
                category = "NO_OVERLAP"
            category_counts[category] += 1
            # NO means the narrative cannot supply a human target.  It is not
            # a negative/empty gold set and is excluded from numeric set and
            # label concordance denominators while remaining visible below.
            if answer != "NO":
                exact_values.append(int(silver == human))
                jaccards.append(_jaccard(silver, human))
                total_shared += len(shared)
                total_silver_only += len(silver_only)
                total_human_only += len(human_only)
            case_level.append(
                {
                    "reliability_case_id": human_by_id[case_id].get(
                        "reliability_case_id", ""
                    ),
                    "artifact_match_id": case_id,
                    "family": family,
                    "human_answerability": answer,
                    "silver_labels": ";".join(sorted(silver, key=AMP_LABEL_ORDER.__getitem__)),
                    "human_narrative_labels": ";".join(sorted(human, key=AMP_LABEL_ORDER.__getitem__)),
                    "shared_labels": ";".join(sorted(shared, key=AMP_LABEL_ORDER.__getitem__)),
                    "silver_only_labels": ";".join(sorted(silver_only, key=AMP_LABEL_ORDER.__getitem__)),
                    "human_only_narrative_supported_labels": ";".join(sorted(human_only, key=AMP_LABEL_ORDER.__getitem__)),
                    "exact_set_concordance": int(silver == human),
                    "jaccard": _jaccard(silver, human),
                    "set_relation_category": category,
                }
            )

        tp = total_shared
        fp = total_silver_only  # silver is the comparison prediction here
        fn = total_human_only
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
        silver_total = total_shared + total_silver_only
        human_total = total_shared + total_human_only
        answerable_n = len(exact_values)
        summary.append(
            {
                "family": family,
                "total_case_n": len(common),
                "answerable_or_partial_case_n": answerable_n,
                "unanswerable_case_n": len(common) - answerable_n,
                "numeric_concordance_denominator": answerable_n,
                "exact_set_concordance": (
                    sum(exact_values) / answerable_n if answerable_n else None
                ),
                "mean_jaccard": sum(jaccards) / answerable_n if answerable_n else None,
                "micro_precision_silver_vs_human": precision,
                "micro_recall_silver_vs_human": recall,
                "micro_f1_silver_vs_human": f1,
                "shared_label_count": total_shared,
                "silver_only_label_count": total_silver_only,
                "human_only_narrative_supported_label_count": total_human_only,
                "proportion_silver_labels_supported_by_human": (
                    total_shared / silver_total if silver_total else None
                ),
                "proportion_silver_labels_not_supported_by_human": (
                    total_silver_only / silver_total if silver_total else None
                ),
                "proportion_human_labels_absent_from_silver": (
                    total_human_only / human_total if human_total else None
                ),
                **{
                    f"case_count_{category.lower()}": category_counts[category]
                    for category in (
                        "SILVER_EQUALS_HUMAN",
                        "SILVER_BROADER_THAN_HUMAN",
                        "HUMAN_BROADER_THAN_SILVER",
                        "PARTIAL_OVERLAP_ADDITIONS_BOTH_SIDES",
                        "NO_OVERLAP",
                        "HUMAN_FAMILY_UNANSWERABLE",
                    )
                },
            }
        )

        for label in AMP_IDS_BY_FAMILY[family]:
            answerable_pairs = [
                (silver, human)
                for silver, human, answer in zip(
                    silver_sets, human_sets, answerability, strict=True
                )
                if answer != "NO"
            ]
            silver_positive = [label in silver for silver, _ in answerable_pairs]
            human_positive = [label in human for _, human in answerable_pairs]
            shared = sum(a and b for a, b in zip(silver_positive, human_positive, strict=True))
            silver_only = sum(a and not b for a, b in zip(silver_positive, human_positive, strict=True))
            human_only = sum(b and not a for a, b in zip(silver_positive, human_positive, strict=True))
            per_label.append(
                {
                    "family": family,
                    "label_id": label,
                    "raw_label": AMP_RAW_LABEL_BY_ID[label],
                    "total_case_n": len(common),
                    "answerable_or_partial_case_n": len(answerable_pairs),
                    "numeric_concordance_denominator": len(answerable_pairs),
                    "silver_support": sum(silver_positive),
                    "human_narrative_support": sum(human_positive),
                    "shared_positive": shared,
                    "silver_only": silver_only,
                    "human_only_narrative_supported": human_only,
                    "raw_concordance": (
                        sum(a == b for a, b in zip(silver_positive, human_positive, strict=True))
                        / len(answerable_pairs)
                        if answerable_pairs
                        else None
                    ),
                }
            )
    return {
        "metadata": {
            "schema_version": VERSION,
            "case_n": len(common),
            "artifact_match_field": match_field,
            "human_reference_term": "human-grounded narrative AMP",
            "silver_reference_term": "SHERLOC silver-reference AMP",
        },
        "summary": summary,
        "per_label": per_label,
        "case_level": case_level,
    }


def _all_labels_from_row(row: Mapping[str, Any], *, prediction: bool = False) -> list[str]:
    labels: list[str] = []
    for family in AMP_FAMILIES:
        labels.extend(_labels_from_row(row, family, prediction=prediction))
    return labels


def _family_metrics(
    human_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[str, Any]:
    label_ids = AMP_IDS_BY_FAMILY[family]
    human = labels_to_indicator(
        [_labels_from_row(row, family) for row in human_rows], label_ids=label_ids
    )
    predicted = labels_to_indicator(
        [_labels_from_row(row, family, prediction=True) for row in prediction_rows],
        label_ids=label_ids,
    )
    supported = [label for label, count in zip(label_ids, human.sum(axis=0), strict=True) if count]
    if not supported:
        return {
            "family": family,
            "case_n": len(human_rows),
            "status": "NO_HUMAN_REFERENCE_SUPPORT",
            "supported_label_count": 0,
            "macro_f1": None,
            "micro_f1": None,
            "exact_set_accuracy": float((human == predicted).all(axis=1).mean()),
            "example_jaccard": None,
        }
    metrics = compute_amp_metrics(
        human, predicted, label_ids=label_ids, macro_label_ids=supported
    )
    return {
        "family": family,
        "case_n": len(human_rows),
        "status": "OK",
        "supported_label_count": len(supported),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "exact_set_accuracy": metrics["exact_set_accuracy"],
        "example_jaccard": metrics["example_jaccard"],
    }


def evaluate_human_gold_predictions(
    human_gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate arbitrary-N predictions against human-grounded narrative AMP.

    No fixed reliability-set size is assumed.  Callers must first use the
    experiment-provenance table to decide which method/case pairs are valid
    under their leakage policy.
    """

    match_field, human_by_id, prediction_by_id = _align_artifacts(
        human_gold_rows,
        prediction_rows,
        left_name="human-grounded gold",
        right_name="model predictions",
    )
    case_ids = sorted(human_by_id)
    if not case_ids:
        raise EvaluationBError("Human-gold evaluation requires at least one case")
    human_rows_all = [human_by_id[item] for item in case_ids]
    predicted_rows_all = [prediction_by_id[item] for item in case_ids]
    for case_id, row in zip(case_ids, human_rows_all, strict=True):
        for family in AMP_FAMILIES:
            field = FAMILY_ANSWERABILITY_COLUMN[family]
            value = _as_text(row.get(field)).strip()
            if value not in ANSWERABILITY_VALUES:
                raise EvaluationBError(f"Invalid {field} for {case_id}: {value!r}")
    eligible_indices = [
        index
        for index, row in enumerate(human_rows_all)
        if all(
            _as_text(row.get(FAMILY_ANSWERABILITY_COLUMN[family])).strip() != "NO"
            for family in AMP_FAMILIES
        )
    ]
    if not eligible_indices:
        raise EvaluationBError(
            "No case has answerable/partially answerable Act, Means, and Purpose together"
        )
    eligible_case_ids = [case_ids[index] for index in eligible_indices]
    human_rows = [human_rows_all[index] for index in eligible_indices]
    predicted_rows = [predicted_rows_all[index] for index in eligible_indices]
    human_labels = [_all_labels_from_row(row) for row in human_rows]
    predicted_labels = [_all_labels_from_row(row, prediction=True) for row in predicted_rows]
    human_matrix = labels_to_indicator(human_labels)
    predicted_matrix = labels_to_indicator(predicted_labels)
    supported = [
        label for label, count in zip(AMP_LABEL_IDS, human_matrix.sum(axis=0), strict=True) if count
    ]
    if not supported:
        raise EvaluationBError("No human-grounded AMP labels have positive support")
    aggregate = compute_amp_metrics(
        human_matrix, predicted_matrix, macro_label_ids=supported
    )
    cpmr = compute_amp_cpmr(human_matrix, predicted_matrix)
    errors = compute_case_errors(human_matrix, predicted_matrix)
    case_level: list[dict[str, Any]] = []
    for case_id, error, cpmr_row, human_row in zip(
        eligible_case_ids, errors, cpmr["per_case"], human_rows, strict=True
    ):
        case_level.append(
            {
                "reliability_case_id": human_row.get("reliability_case_id", ""),
                "artifact_match_id": case_id,
                **error,
                **cpmr_row,
                **{
                    field: human_row.get(field, "")
                    for field in (
                        "act_answerability",
                        "means_answerability",
                        "purpose_answerability",
                    )
                },
            }
        )

    strata: list[dict[str, Any]] = []
    per_family: list[dict[str, Any]] = []
    for family in AMP_FAMILIES:
        answer_field = FAMILY_ANSWERABILITY_COLUMN[family]
        family_indices = [
            index
            for index, row in enumerate(human_rows_all)
            if _as_text(row.get(answer_field)).strip() in {"YES", "PARTIAL"}
        ]
        if family_indices:
            per_family.append(
                _family_metrics(
                    [human_rows_all[index] for index in family_indices],
                    [predicted_rows_all[index] for index in family_indices],
                    family,
                )
            )
        else:
            per_family.append(
                {
                    "family": family,
                    "case_n": 0,
                    "status": "NO_ANSWERABLE_OR_PARTIAL_CASES",
                    "supported_label_count": 0,
                    "macro_f1": None,
                    "micro_f1": None,
                    "exact_set_accuracy": None,
                    "example_jaccard": None,
                }
            )
        for answerability in ANSWERABILITY_VALUES:
            indices = [
                index
                for index, row in enumerate(human_rows_all)
                if _as_text(row.get(answer_field)).strip() == answerability
            ]
            if not indices:
                strata.append(
                    {
                        "family": family,
                        "answerability": answerability,
                        "case_n": 0,
                        "status": "NO_CASES",
                        "macro_f1": None,
                        "micro_f1": None,
                        "exact_set_accuracy": None,
                        "example_jaccard": None,
                    }
                )
                continue
            if answerability == "NO":
                strata.append(
                    {
                        "family": family,
                        "answerability": answerability,
                        "case_n": len(indices),
                        "status": "HUMAN_FAMILY_UNANSWERABLE_NUMERIC_METRICS_EXCLUDED",
                        "macro_f1": None,
                        "micro_f1": None,
                        "exact_set_accuracy": None,
                        "example_jaccard": None,
                    }
                )
                continue
            metric = _family_metrics(
                [human_rows_all[index] for index in indices],
                [predicted_rows_all[index] for index in indices],
                family,
            )
            strata.append({"answerability": answerability, **metric})

    return {
        "metadata": {
            "schema_version": VERSION,
            "total_supplied_case_n": len(case_ids),
            "fully_amp_answerable_or_partial_case_n": len(eligible_case_ids),
            "case_n": len(eligible_case_ids),
            "excluded_due_to_any_amp_no_case_n": len(case_ids) - len(eligible_case_ids),
            "reference_term": "human-grounded narrative AMP",
            "artifact_match_field": match_field,
            "requires_prior_provenance_filter": True,
            "unanswerable_families_treated_as_negative": False,
        },
        "aggregate": aggregate,
        "cpmr": cpmr["by_family"],
        "per_label": aggregate["per_label"],
        "per_family": per_family,
        "answerability_strata": strata,
        "case_level": case_level,
    }


def _bool_cell(value: Any) -> int:
    text = _as_text(value).strip()
    if text in {"1", "TRUE", "true", "True"} or value is True or value == 1:
        return 1
    if text in {"0", "FALSE", "false", "False", ""} or value is False or value == 0:
        return 0
    raise EvaluationBError(f"Expected binary split metadata, found {value!r}")


def build_reliability_experiment_provenance(
    reliability_case_rows: Sequence[Mapping[str, Any]],
    a1_split_rows: Sequence[Mapping[str, Any]],
    a2_split_rows: Sequence[Mapping[str, Any]],
    *,
    a1_split_sha256: str = "",
    a2_split_sha256: str = "",
    reliability_sample_sha256: str = "",
) -> list[dict[str, Any]]:
    """Join fixed reliability IDs to frozen A1/A2 exposure metadata.

    All reliability cases are retained.  Cases outside the 1,263-case primary
    AMP cohort receive explicit ``OUTSIDE_PRIMARY_COHORT`` roles instead of
    being dropped.  No annotation or target-label value is read or emitted.
    """

    reliability_by_id = _index_unique(
        reliability_case_rows, artifact="fixed reliability case list"
    )
    a1_by_rank: dict[int, Mapping[str, Any]] = {}
    for row in a1_split_rows:
        rank = int(row["search_rank"])
        if rank in a1_by_rank:
            raise EvaluationBError(f"Duplicate A1 split rank {rank}")
        a1_by_rank[rank] = row

    a2_by_rank_fold: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in a2_split_rows:
        key = (int(row["search_rank"]), int(row["fold_id"]))
        if key in a2_by_rank_fold:
            raise EvaluationBError(f"Duplicate A2 split rank/fold {key}")
        a2_by_rank_fold[key] = row

    output: list[dict[str, Any]] = []
    for case_id, reliability in sorted(
        reliability_by_id.items(),
        key=lambda item: int(_as_text(item[1].get("reviewer_order", "0")) or 0),
    ):
        rank = int(reliability["search_rank"])
        canonical_url = _as_text(reliability.get("canonical_url")).strip()
        a1 = a1_by_rank.get(rank)
        primary = a1 is not None
        stated_primary = _as_text(reliability.get("primary_amp_cohort_member")).strip()
        if stated_primary and _bool_cell(stated_primary) != int(primary):
            raise EvaluationBError(
                f"Primary-cohort flag conflicts with frozen A1 split for {case_id}"
            )
        if a1 is not None and canonical_url and _as_text(a1.get("canonical_url")).strip() != canonical_url:
            raise EvaluationBError(f"Canonical URL mismatch for reliability case {case_id}")

        a1_role = _as_text(a1.get("split")).strip() if a1 else "OUTSIDE_PRIMARY_COHORT"
        row_out: dict[str, Any] = {
            "provenance_schema_version": VERSION,
            "reliability_case_id": case_id,
            "reviewer_order": reliability.get("reviewer_order", ""),
            "search_rank": rank,
            "canonical_url": canonical_url,
            "primary_amp_cohort_member": int(primary),
            "primary_cohort_status": "IN_PRIMARY_AMP_COHORT" if primary else "OUTSIDE_PRIMARY_COHORT",
            "a1_role": a1_role,
            "a1_train": int(a1_role == "TRAIN"),
            "a1_validation": int(a1_role == "VALIDATION"),
            "a1_test": int(a1_role == "TEST"),
            "a1_active_demo": int(a1_role == "ACTIVE_DEMO"),
            "a1_reserve_demo": int(a1_role == "RESERVE_DEMO"),
            "a1_effective_supervised_train": _bool_cell(a1.get("effective_supervised_train", "0")) if a1 else 0,
            "a1_demo_bank_role": a1.get("demo_bank_role", "") if a1 else "",
            "a1_m4_demo": _bool_cell(a1.get("m4_demo", "0")) if a1 else 0,
            "a1_existing_prediction_evaluable_without_split_leakage": int(a1_role == "TEST"),
        }

        test_folds: list[int] = []
        any_a2_supervised_exposure = False
        any_demo_exposure = bool(row_out["a1_active_demo"] or row_out["a1_reserve_demo"])
        for fold in (1, 2, 3):
            a2 = a2_by_rank_fold.get((rank, fold))
            if primary and a2 is None:
                raise EvaluationBError(f"Primary case {case_id} is missing A2 fold {fold}")
            if not primary and a2 is not None:
                raise EvaluationBError(f"Outside-primary case {case_id} unexpectedly occurs in A2")
            if a2 is not None and canonical_url and _as_text(a2.get("canonical_url")).strip() != canonical_url:
                raise EvaluationBError(f"A2 canonical URL mismatch for {case_id}, fold {fold}")
            role = _as_text(a2.get("role")).strip() if a2 else "OUTSIDE_PRIMARY_COHORT"
            supervised = _bool_cell(a2.get("effective_supervised_train", "0")) if a2 else 0
            active_demo = int(role == "ACTIVE_DEMO")
            reserve_demo = int(role == "RESERVE_DEMO")
            m4_demo = _bool_cell(a2.get("m4_demo", "0")) if a2 else 0
            if role == "TEST":
                test_folds.append(fold)
            any_a2_supervised_exposure = any_a2_supervised_exposure or bool(supervised)
            any_demo_exposure = any_demo_exposure or bool(active_demo or reserve_demo)
            prefix = f"a2_fold_{fold}"
            row_out.update(
                {
                    f"{prefix}_role": role,
                    f"{prefix}_train": int(role == "TRAIN"),
                    f"{prefix}_validation": int(role == "VALIDATION"),
                    f"{prefix}_test": int(role == "TEST"),
                    f"{prefix}_active_demo": active_demo,
                    f"{prefix}_reserve_demo": reserve_demo,
                    f"{prefix}_effective_supervised_train": supervised,
                    f"{prefix}_demo_bank_role": a2.get("demo_bank_role", "") if a2 else "",
                    f"{prefix}_approved_demo_pool_role": a2.get("approved_demo_pool_role", "") if a2 else "",
                    f"{prefix}_m4_demo": m4_demo,
                    f"{prefix}_existing_prediction_evaluable_without_split_leakage": int(role == "TEST"),
                }
            )
        row_out.update(
            {
                "a2_test_fold_ids": ";".join(str(value) for value in test_folds),
                "any_effective_supervised_training_exposure": int(
                    bool(row_out["a1_effective_supervised_train"]) or any_a2_supervised_exposure
                ),
                "any_active_or_reserve_demo_exposure": int(any_demo_exposure),
                "evaluation_b_leakage_caution": (
                    "OUTSIDE_PRIMARY_COHORT_NO_EXISTING_M1_M4_EVAL_A_PREDICTION"
                    if not primary
                    else "USE_ONLY_SPLIT_SPECIFIC_TEST_PREDICTIONS"
                ),
                "a1_split_sha256": a1_split_sha256,
                "a2_split_sha256": a2_split_sha256,
                "reliability_sample_sha256": reliability_sample_sha256,
                "annotation_values_used": 0,
                "case_selection_performed": 0,
            }
        )
        output.append(row_out)
    return output


def generate_frozen_reliability_provenance(
    *,
    reliability_sample_path: Path,
    a1_split_path: Path,
    a2_split_path: Path,
    output_path: Path,
    validate_frozen_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Generate the one label-free Evaluation B artifact authorized now."""

    hashes = {
        "reliability": _sha256_file(reliability_sample_path),
        "a1": _sha256_file(a1_split_path),
        "a2": _sha256_file(a2_split_path),
    }
    expected = {
        "reliability": EXPECTED_RELIABILITY_SAMPLE_SHA256,
        "a1": EXPECTED_A1_FINAL_SHA256,
        "a2": EXPECTED_A2_FINAL_SHA256,
    }
    if validate_frozen_hashes:
        mismatches = [key for key in hashes if hashes[key] != expected[key]]
        if mismatches:
            raise EvaluationBError(
                "Frozen provenance inputs changed: "
                + ", ".join(f"{key}={hashes[key]} expected={expected[key]}" for key in mismatches)
            )
    rows = build_reliability_experiment_provenance(
        load_csv_rows(reliability_sample_path),
        load_csv_rows(a1_split_path),
        load_csv_rows(a2_split_path),
        a1_split_sha256=hashes["a1"],
        a2_split_sha256=hashes["a2"],
        reliability_sample_sha256=hashes["reliability"],
    )
    _atomic_csv(output_path, rows)
    return rows


__all__ = [
    "AMP_ID_BY_RAW_LABEL",
    "AMP_RAW_LABEL_BY_ID",
    "ANSWERABILITY_VALUES",
    "AnnotationQCResult",
    "CHILD_VALUES",
    "EvaluationBError",
    "FORM_VALUES",
    "KappaResult",
    "MULTIPLICITY_VALUES",
    "answerability_is_abstention",
    "build_disagreement_queue",
    "build_human_gold",
    "build_reliability_experiment_provenance",
    "cohen_kappa",
    "compare_silver_to_human",
    "compute_reviewer_agreement",
    "evaluate_human_gold_predictions",
    "extract_numbered_sentences",
    "generate_frozen_reliability_provenance",
    "load_csv_rows",
    "load_jsonl_rows",
    "map_evidence_ids_to_text",
    "parse_amp_labels",
    "parse_evidence_sentence_ids",
    "qc_annotations",
    "selective_evaluation_mask",
    "validate_evidence_sentence_ids",
    "write_qc_reports",
]
