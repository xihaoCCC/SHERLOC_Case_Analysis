#!/usr/bin/env python3
"""Freeze the single-reviewer Evaluation B narrative reference.

The raw reviewer CSV is read-only.  Review inclusion is governed exclusively
by ``Done?`` and normalized ``annotation_notes`` status; pre-populated values
on unreviewed rows are never used.  Only deterministic syntax normalization is
performed, with every original cell preserved in the derived reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .evaluation_b import AMP_ID_BY_RAW_LABEL, AMP_RAW_LABEL_BY_ID
except ImportError:
    from evaluation_b import AMP_ID_BY_RAW_LABEL, AMP_RAW_LABEL_BY_ID


VERSION = "1.0.0"
FREEZE_STATUS = "FROZEN_FOR_EVALUATION_B_PRE_MODEL_INFERENCE"
EXPECTED_SOURCE_SHA256 = "7ec0a40ab6a9d64588cf4b6c8b46d2572683cf7e340d786117604bc6f20081af"
EXPECTED_SOURCE_ROWS = 100
EXPECTED_CONTEXT_SHA256 = "ff825d1996ff55a72030ef07835c3b71c318df4f43a40ecb79714c27192794bc"
EXPECTED_A1_SPLIT_SHA256 = "63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d"

ACT_SOURCE_COLUMN = "Acts human labeled "  # trailing space exists in immutable source
MEANS_SOURCE_COLUMN = "Means human labeled"
PURPOSE_SOURCE_COLUMN = "Purpose human labeled"
GEO_SOURCE_COLUMN = "Geographic Form human labeled"
MULTIPLICITY_SOURCE_COLUMN = "multiplicity human labeled"
CHILD_SOURCE_COLUMN = "child human labeled"

REQUIRED_SOURCE_COLUMNS = (
    "reviewer_id",
    "reliability_case_id",
    "sentence_splitter_version",
    "sentence_count",
    "english_fact_summary_raw",
    "Done?",
    ACT_SOURCE_COLUMN,
    MEANS_SOURCE_COLUMN,
    PURPOSE_SOURCE_COLUMN,
    GEO_SOURCE_COLUMN,
    MULTIPLICITY_SOURCE_COLUMN,
    CHILD_SOURCE_COLUMN,
    "annotation_notes",
)

AMP_ALLOWED_BY_FAMILY = {
    "ACT": tuple(
        AMP_RAW_LABEL_BY_ID[label]
        for label in (
            "ACT_RECRUITMENT",
            "ACT_TRANSPORTATION",
            "ACT_TRANSFER",
            "ACT_HARBOURING",
            "ACT_RECEIPT",
        )
    ),
    "MEANS": tuple(
        AMP_RAW_LABEL_BY_ID[label]
        for label in (
            "MEANS_THREAT_FORCE_OR_COERCION",
            "MEANS_ABDUCTION",
            "MEANS_FRAUD",
            "MEANS_DECEPTION",
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
            "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
        )
    ),
    "PURPOSE": tuple(
        AMP_RAW_LABEL_BY_ID[label]
        for label in (
            "PURPOSE_SEXUAL_EXPLOITATION",
            "PURPOSE_FORCED_LABOUR_OR_SERVICES",
            "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
            "PURPOSE_SERVITUDE",
            "PURPOSE_REMOVAL_OF_ORGANS",
            "PURPOSE_OTHER",
        )
    ),
}

GEO_ALLOWED = ("Internal", "Transnational")
OCG_RAW_LABEL = "Organized Criminal Group"
MULTIPLICITY_ALLOWED = ("SINGLE", "MULTIPLE", "UNKNOWN", "Not Applicable")
CHILD_ALLOWED = ("TRUE", "FALSE", "UNKNOWN", "Not Applicable")
AUX_SENTINEL_MAP = {"NOT_APPLICABLE_OUTSIDE_PRIMARY_COHORT": "Not Applicable"}

SMART_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": '"',
        "\u2019": '"',
        "\u201a": '"',
        "\u201b": '"',
    }
)


class HumanReferenceError(RuntimeError):
    """Raised when immutable human input cannot be frozen safely."""


@dataclass(frozen=True)
class ParsedList:
    values: tuple[str, ...]
    smart_quotes_normalized: bool
    non_json_syntax_normalized: bool
    whitespace_normalized: bool
    reordered: bool
    duplicate_values: tuple[str, ...]


@dataclass(frozen=True)
class HumanReferenceBuildResult:
    source_manifest: dict[str, Any]
    qc_summary: dict[str, Any]
    qc_rows: list[dict[str, Any]]
    reference_rows: list[dict[str, Any]]
    exclusion_rows: list[dict[str, Any]]
    membership_rows: list[dict[str, Any]]
    freeze_manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
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


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _timestamp_for_freeze(manifest_path: Path, source_sha256: str) -> str:
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("source", {}).get("sha256") == source_sha256:
                timestamp = existing.get("audit_generated_at_utc")
                if timestamp:
                    return str(timestamp)
        except (json.JSONDecodeError, OSError):
            pass
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def classify_review_status(done_raw: Any, annotation_notes_raw: Any) -> str:
    """Classify solely from Done? and a deterministic note normalization."""

    done = str(done_raw or "").strip()
    note = " ".join(str(annotation_notes_raw or "").split()).casefold()
    if done == "":
        return "NOT_REVIEWED"
    if done != "1":
        raise HumanReferenceError(f"Invalid Done? value: {done!r}")
    if note == "":
        return "SUBSTANTIVE"
    if note == "skip":
        return "SKIP"
    if note == "abstain" or note.startswith("abstain,") or note.startswith("abstain:"):
        return "ABSTAIN"
    raise HumanReferenceError(f"Unknown reviewed annotation_notes value: {annotation_notes_raw!r}")


def parse_human_list(
    raw_value: Any,
    *,
    allowed_values: Sequence[str],
    allow_unquoted_bracket_items: bool = False,
) -> ParsedList:
    """Parse list syntax after quote/whitespace cleanup, without fuzzy mapping."""

    raw = "" if raw_value is None else str(raw_value)
    stripped = raw.strip()
    if not stripped:
        raise HumanReferenceError("Missing list value")
    quote_normalized = stripped.translate(SMART_QUOTE_TRANSLATION)
    smart_changed = quote_normalized != stripped
    non_json_changed = False
    try:
        decoded = json.loads(quote_normalized)
    except json.JSONDecodeError as exc:
        if allow_unquoted_bracket_items and quote_normalized.startswith("[") and quote_normalized.endswith("]"):
            inner = quote_normalized[1:-1].strip()
            if not inner:
                decoded = []
            else:
                tokens = [token.strip().strip('"') for token in inner.split(",")]
                if any(not token for token in tokens):
                    raise HumanReferenceError(f"Malformed list: {raw!r}") from exc
                decoded = tokens
            non_json_changed = True
        else:
            raise HumanReferenceError(f"Malformed JSON-like list: {raw!r}") from exc
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise HumanReferenceError("Human list must be an array of strings")
    collapsed = [" ".join(item.split()) for item in decoded]
    whitespace_changed = collapsed != decoded
    unknown = [item for item in collapsed if item not in allowed_values]
    if unknown:
        raise HumanReferenceError(f"Unknown labels: {unknown}")
    duplicate_values = tuple(
        item for item, count in Counter(collapsed).items() if count > 1
    )
    unique = set(collapsed)
    canonical = tuple(item for item in allowed_values if item in unique)
    reordered = tuple(dict.fromkeys(collapsed)) != canonical
    return ParsedList(
        values=canonical,
        smart_quotes_normalized=smart_changed,
        non_json_syntax_normalized=non_json_changed,
        whitespace_normalized=whitespace_changed,
        reordered=reordered,
        duplicate_values=duplicate_values,
    )


def _normalize_auxiliary(raw_value: Any, *, allowed_values: Sequence[str]) -> tuple[str, bool]:
    raw = str(raw_value or "").strip()
    if raw in allowed_values:
        return raw, False
    if raw in AUX_SENTINEL_MAP:
        return AUX_SENTINEL_MAP[raw], True
    raise HumanReferenceError(f"Invalid auxiliary value: {raw!r}")


def _membership_digest(rows: Sequence[Mapping[str, Any]], *, retained_only: bool = False, substantive_only: bool = False) -> str:
    selected = [
        row
        for row in rows
        if (not retained_only or int(row["retained"]) == 1)
        and (not substantive_only or row["review_status"] == "SUBSTANTIVE")
    ]
    members = [
        {
            "reliability_case_id": row["reliability_case_id"],
            "search_rank": int(row["search_rank"]),
            "canonical_url": row["canonical_url"],
            "input_sha256": row["input_sha256"],
            "review_status": row["review_status"],
        }
        for row in sorted(selected, key=lambda item: int(item["search_rank"]))
    ]
    payload = json.dumps(
        members, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


QC_COLUMNS = (
    "source_row_number",
    "reliability_case_id",
    "review_status",
    "field",
    "severity",
    "blocks_scoring",
    "code",
    "message",
    "raw_value",
    "normalized_value",
)

REFERENCE_COLUMNS = (
    "reference_schema_version",
    "reliability_case_id",
    "reviewer_order",
    "search_rank",
    "case_title",
    "unodc_case_number",
    "canonical_url",
    "jurisdiction",
    "fact_summary",
    "reviewed",
    "review_status",
    "done_raw",
    "acts_human_raw",
    "acts_human_clean_json",
    "act_label_ids_json",
    "act_labels",
    "means_human_raw",
    "means_human_clean_json",
    "means_label_ids_json",
    "means_labels",
    "purpose_human_raw",
    "purpose_human_clean_json",
    "purpose_label_ids_json",
    "purpose_labels",
    "geographic_form_human_raw",
    "geographic_form_human_clean_json",
    "organized_criminal_group_human",
    "organized_criminal_group_evaluable",
    "multiplicity_human_raw",
    "multiplicity_human_clean",
    "child_human_raw",
    "child_human_clean",
    "annotation_notes",
    "substantive_amp_evaluable",
    "auxiliary_evaluable",
    "source_file",
    "source_sha256",
    "source_row_number",
    "input_sha256",
    "context_file",
    "context_sha256",
)

EXCLUSION_COLUMNS = (
    "reliability_case_id",
    "reviewer_order",
    "search_rank",
    "case_title",
    "canonical_url",
    "jurisdiction",
    "reviewed",
    "review_status",
    "exclusion_reason",
    "done_raw",
    "annotation_notes",
    "source_file",
    "source_sha256",
    "source_row_number",
    "input_sha256",
)

MEMBERSHIP_COLUMNS = (
    "reliability_case_id",
    "reviewer_order",
    "search_rank",
    "canonical_url",
    "jurisdiction",
    "input_sha256",
    "reviewed",
    "review_status",
    "retained",
    "substantive_amp_evaluable",
    "abstain_subset",
    "excluded",
    "exclusion_reason",
    "source_row_number",
    "source_sha256",
)


def build_single_reviewer_reference(
    *,
    repo_root: Path,
    source_path: Path,
    context_path: Path,
    a1_split_path: Path,
    source_manifest_path: Path,
    qc_report_path: Path,
    qc_summary_path: Path,
    reference_path: Path,
    exclusions_path: Path,
    membership_path: Path,
    freeze_manifest_path: Path,
    validate_frozen_hashes: bool = True,
) -> HumanReferenceBuildResult:
    """QC and freeze the Done?-gated single-reviewer reference."""

    source_sha_before = sha256_file(source_path)
    context_sha = sha256_file(context_path)
    a1_sha = sha256_file(a1_split_path)
    if validate_frozen_hashes:
        expected = {
            "annotation source": (source_sha_before, EXPECTED_SOURCE_SHA256),
            "reliability context": (context_sha, EXPECTED_CONTEXT_SHA256),
            "A1 split": (a1_sha, EXPECTED_A1_SPLIT_SHA256),
        }
        mismatch = [name for name, values in expected.items() if values[0] != values[1]]
        if mismatch:
            raise HumanReferenceError(
                "Frozen input hash mismatch: "
                + ", ".join(
                    f"{name}={expected[name][0]} expected={expected[name][1]}"
                    for name in mismatch
                )
            )

    columns, rows = _read_csv(source_path)
    _, context_rows = _read_csv(context_path)
    _, a1_rows = _read_csv(a1_split_path)
    audit_timestamp = _timestamp_for_freeze(source_manifest_path, source_sha_before)
    stat = source_path.stat()
    source_manifest: dict[str, Any] = {
        "manifest_schema_version": VERSION,
        "artifact_role": "IMMUTABLE_SINGLE_REVIEWER_HUMAN_ANNOTATION_SOURCE",
        "source": {
            "path": _relative(source_path, repo_root),
            "sha256": source_sha_before,
            "byte_count": stat.st_size,
            "row_count": len(rows),
            "column_count": len(columns),
            "columns_exact": columns,
            "filesystem_modified_at_utc": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).replace(microsecond=0).isoformat(),
        },
        "sha256": source_sha_before,
        "row_count": len(rows),
        "audit_generated_at_utc": audit_timestamp,
        "immutable_raw_source": True,
        "snapshot_copy_created": False,
        "snapshot_note": "The original repository artifact is preserved in place and frozen by SHA-256.",
        "authoritative_review_gate": "Done? == 1",
        "labels_on_done_blank_rows_used": False,
    }
    _atomic_json(source_manifest_path, source_manifest)

    issues: list[dict[str, Any]] = []

    def add_issue(
        *,
        row_number: int | str = "",
        case_id: str = "",
        status: str = "",
        field: str,
        severity: str,
        blocks: bool,
        code: str,
        message: str,
        raw: Any = "",
        normalized: Any = "",
    ) -> None:
        issues.append(
            {
                "source_row_number": row_number,
                "reliability_case_id": case_id,
                "review_status": status,
                "field": field,
                "severity": severity,
                "blocks_scoring": int(blocks),
                "code": code,
                "message": message,
                "raw_value": "" if raw is None else str(raw),
                "normalized_value": "" if normalized is None else str(normalized),
            }
        )

    if len(rows) != EXPECTED_SOURCE_ROWS:
        add_issue(
            field="<file>", severity="ERROR", blocks=True, code="SOURCE_ROW_COUNT_MISMATCH",
            message=f"Expected {EXPECTED_SOURCE_ROWS} rows, found {len(rows)}",
        )
    missing_columns = [column for column in REQUIRED_SOURCE_COLUMNS if column not in columns]
    for column in missing_columns:
        add_issue(
            field=column, severity="ERROR", blocks=True, code="MISSING_REQUIRED_COLUMN",
            message="Required immutable-source column is missing",
        )
    if missing_columns:
        qc_summary = {
            "qc_schema_version": VERSION,
            "status": "BLOCKED",
            "blocking_issue_count": len(missing_columns),
            "blocking_error_count": len(missing_columns),
            "source_sha256": source_sha_before,
        }
        _atomic_csv(qc_report_path, issues, QC_COLUMNS)
        _atomic_json(qc_summary_path, qc_summary)
        raise HumanReferenceError("Human annotation source is missing required columns")

    context_by_id = {row["reliability_case_id"]: row for row in context_rows}
    if len(context_by_id) != len(context_rows):
        add_issue(
            field="reliability_case_id", severity="ERROR", blocks=True,
            code="DUPLICATE_CONTEXT_CASE_ID", message="Reliability context contains duplicate IDs",
        )
    a1_by_rank = {int(row["search_rank"]): row for row in a1_rows}
    active_demo_ranks = {
        rank for rank, row in a1_by_rank.items() if row.get("split") == "ACTIVE_DEMO"
    }

    seen: set[str] = set()
    processed: list[dict[str, Any]] = []
    syntax_normalization_count = 0
    duplicate_normalization_count = 0
    ocg_extraction_count = 0

    for index, raw_row in enumerate(rows, start=2):
        case_id = raw_row["reliability_case_id"].strip()
        done_raw = raw_row["Done?"]
        note_raw = raw_row["annotation_notes"]
        if not case_id:
            add_issue(
                row_number=index, field="reliability_case_id", severity="ERROR", blocks=True,
                code="MISSING_CASE_ID", message="Reliability case ID is blank",
            )
            continue
        if case_id in seen:
            add_issue(
                row_number=index, case_id=case_id, field="reliability_case_id",
                severity="ERROR", blocks=True, code="DUPLICATE_CASE_ID",
                message="Duplicate reliability case ID",
            )
            continue
        seen.add(case_id)
        try:
            status = classify_review_status(done_raw, note_raw)
        except HumanReferenceError as exc:
            status = "INVALID"
            add_issue(
                row_number=index, case_id=case_id, status=status,
                field="Done?/annotation_notes", severity="ERROR", blocks=True,
                code="INVALID_REVIEW_STATUS", message=str(exc), raw=f"{done_raw!r} / {note_raw!r}",
            )

        context = context_by_id.get(case_id)
        if context is None:
            add_issue(
                row_number=index, case_id=case_id, status=status,
                field="reliability_case_id", severity="ERROR", blocks=True,
                code="CONTEXT_CASE_MISSING", message="Case absent from frozen reliability context",
            )
            continue
        fact_summary = raw_row["english_fact_summary_raw"]
        if fact_summary != context["english_fact_summary_raw"]:
            add_issue(
                row_number=index, case_id=case_id, status=status,
                field="english_fact_summary_raw", severity="ERROR", blocks=True,
                code="FACT_SUMMARY_CONTEXT_MISMATCH",
                message="Reviewer narrative differs from frozen reliability context",
            )

        record: dict[str, Any] = {
            "source_row_number": index,
            "raw": raw_row,
            "context": context,
            "case_id": case_id,
            "status": status,
            "input_sha256": sha256_text(fact_summary),
        }
        if status == "NOT_REVIEWED":
            if any(str(raw_row[column] or "").strip() for column in (
                ACT_SOURCE_COLUMN, MEANS_SOURCE_COLUMN, PURPOSE_SOURCE_COLUMN,
                GEO_SOURCE_COLUMN, MULTIPLICITY_SOURCE_COLUMN, CHILD_SOURCE_COLUMN,
            )):
                add_issue(
                    row_number=index, case_id=case_id, status=status, field="Done?",
                    severity="INFO", blocks=False, code="UNREVIEWED_PREPOPULATED_VALUES_IGNORED",
                    message="Done? is blank; all human-label cells are ignored",
                )
            processed.append(record)
            continue

        parsed_amp: dict[str, ParsedList] = {}
        for family, source_column in (
            ("ACT", ACT_SOURCE_COLUMN),
            ("MEANS", MEANS_SOURCE_COLUMN),
            ("PURPOSE", PURPOSE_SOURCE_COLUMN),
        ):
            try:
                parsed = parse_human_list(
                    raw_row[source_column], allowed_values=AMP_ALLOWED_BY_FAMILY[family]
                )
                parsed_amp[family] = parsed
                changed = (
                    parsed.smart_quotes_normalized
                    or parsed.non_json_syntax_normalized
                    or parsed.whitespace_normalized
                    or parsed.reordered
                )
                if changed:
                    syntax_normalization_count += 1
                    add_issue(
                        row_number=index, case_id=case_id, status=status, field=source_column,
                        severity="INFO", blocks=False, code="DETERMINISTIC_SYNTAX_NORMALIZATION",
                        message="Quote/whitespace/list order normalized without semantic mapping",
                        raw=raw_row[source_column], normalized=json.dumps(parsed.values, ensure_ascii=False),
                    )
                if parsed.duplicate_values:
                    duplicate_normalization_count += 1
                    add_issue(
                        row_number=index, case_id=case_id, status=status, field=source_column,
                        severity="WARNING", blocks=False, code="DUPLICATE_LABEL_DEDUPLICATED",
                        message="Duplicate set member removed deterministically",
                        raw=raw_row[source_column], normalized=json.dumps(parsed.values, ensure_ascii=False),
                    )
            except HumanReferenceError as exc:
                add_issue(
                    row_number=index, case_id=case_id, status=status, field=source_column,
                    severity="ERROR" if status != "SKIP" else "WARNING",
                    blocks=status in {"SUBSTANTIVE", "ABSTAIN"}, code="INVALID_AMP_LIST",
                    message=str(exc), raw=raw_row[source_column],
                )

        try:
            geo_parsed = parse_human_list(
                raw_row[GEO_SOURCE_COLUMN],
                allowed_values=(*GEO_ALLOWED, OCG_RAW_LABEL),
                allow_unquoted_bracket_items=True,
            )
            if geo_parsed.smart_quotes_normalized or geo_parsed.non_json_syntax_normalized or geo_parsed.reordered:
                syntax_normalization_count += 1
                add_issue(
                    row_number=index, case_id=case_id, status=status, field=GEO_SOURCE_COLUMN,
                    severity="INFO", blocks=False, code="DETERMINISTIC_SYNTAX_NORMALIZATION",
                    message="Geographic list syntax/order normalized without semantic mapping",
                    raw=raw_row[GEO_SOURCE_COLUMN], normalized=json.dumps(geo_parsed.values, ensure_ascii=False),
                )
            if geo_parsed.duplicate_values:
                duplicate_normalization_count += 1
                add_issue(
                    row_number=index, case_id=case_id, status=status, field=GEO_SOURCE_COLUMN,
                    severity="WARNING", blocks=False, code="DUPLICATE_LABEL_DEDUPLICATED",
                    message="Duplicate geographic set member removed deterministically",
                    raw=raw_row[GEO_SOURCE_COLUMN], normalized=json.dumps(geo_parsed.values, ensure_ascii=False),
                )
            if OCG_RAW_LABEL in geo_parsed.values:
                ocg_extraction_count += int(status in {"SUBSTANTIVE", "ABSTAIN"})
                add_issue(
                    row_number=index, case_id=case_id, status=status, field=GEO_SOURCE_COLUMN,
                    severity="INFO", blocks=False, code="OCG_SPLIT_FROM_GEOGRAPHIC_FORM",
                    message="Organized Criminal Group extracted into its separate binary field",
                    raw=raw_row[GEO_SOURCE_COLUMN],
                    normalized=json.dumps([v for v in geo_parsed.values if v in GEO_ALLOWED]),
                )
            record["geo"] = geo_parsed
        except HumanReferenceError as exc:
            add_issue(
                row_number=index, case_id=case_id, status=status, field=GEO_SOURCE_COLUMN,
                severity="ERROR" if status != "SKIP" else "WARNING",
                blocks=status in {"SUBSTANTIVE", "ABSTAIN"}, code="INVALID_GEOGRAPHIC_FORM",
                message=str(exc), raw=raw_row[GEO_SOURCE_COLUMN],
            )

        for key, source_column, allowed in (
            ("multiplicity", MULTIPLICITY_SOURCE_COLUMN, MULTIPLICITY_ALLOWED),
            ("child", CHILD_SOURCE_COLUMN, CHILD_ALLOWED),
        ):
            try:
                normalized, sentinel = _normalize_auxiliary(raw_row[source_column], allowed_values=allowed)
                record[key] = normalized
                if sentinel:
                    add_issue(
                        row_number=index, case_id=case_id, status=status, field=source_column,
                        severity="WARNING", blocks=False, code="OUTSIDE_COHORT_SENTINEL_NORMALIZED",
                        message="Explicit outside-cohort sentinel normalized to Not Applicable",
                        raw=raw_row[source_column], normalized=normalized,
                    )
            except HumanReferenceError as exc:
                add_issue(
                    row_number=index, case_id=case_id, status=status, field=source_column,
                    severity="ERROR" if status != "SKIP" else "WARNING",
                    blocks=status in {"SUBSTANTIVE", "ABSTAIN"}, code="INVALID_AUXILIARY_VALUE",
                    message=str(exc), raw=raw_row[source_column],
                )

        record["amp"] = parsed_amp
        if status == "SUBSTANTIVE":
            for source_column in (
                ACT_SOURCE_COLUMN, MEANS_SOURCE_COLUMN, PURPOSE_SOURCE_COLUMN,
                GEO_SOURCE_COLUMN, MULTIPLICITY_SOURCE_COLUMN, CHILD_SOURCE_COLUMN,
            ):
                if not str(raw_row[source_column] or "").strip():
                    add_issue(
                        row_number=index, case_id=case_id, status=status, field=source_column,
                        severity="ERROR", blocks=True, code="MISSING_SUBSTANTIVE_HUMAN_VALUE",
                        message="Substantive reviewed case has a missing human value",
                    )
        if status == "ABSTAIN":
            for family, parsed in parsed_amp.items():
                if parsed.values:
                    add_issue(
                        row_number=index, case_id=case_id, status=status,
                        field={"ACT": ACT_SOURCE_COLUMN, "MEANS": MEANS_SOURCE_COLUMN, "PURPOSE": PURPOSE_SOURCE_COLUMN}[family],
                        severity="ERROR", blocks=True, code="ABSTAIN_NONEMPTY_AMP_CONTRADICTION",
                        message="ABSTAIN case must not retain a nonempty AMP set",
                        normalized=json.dumps(parsed.values, ensure_ascii=False),
                    )
            for key, source_column in (
                ("multiplicity", MULTIPLICITY_SOURCE_COLUMN), ("child", CHILD_SOURCE_COLUMN)
            ):
                if record.get(key) not in {None, "Not Applicable"}:
                    add_issue(
                        row_number=index, case_id=case_id, status=status, field=source_column,
                        severity="INFO", blocks=False, code="ABSTAIN_AUXILIARY_PRESERVED_NOT_EVALUATED",
                        message="Auxiliary value is preserved but masked from evaluation for ABSTAIN",
                        raw=raw_row[source_column], normalized=record.get(key),
                    )
        processed.append(record)

    if set(context_by_id) != seen:
        for case_id in sorted(set(context_by_id) - seen):
            add_issue(
                case_id=case_id, field="reliability_case_id", severity="ERROR", blocks=True,
                code="SOURCE_CASE_MISSING", message="Frozen reliability case is absent from source",
            )

    hr61 = next((item for item in processed if item["case_id"] == "HRV1-061"), None)
    hr61_ok = bool(
        hr61
        and hr61["status"] == "ABSTAIN"
        and all(not hr61.get("amp", {}).get(family, ParsedList((), False, False, False, False, ())).values for family in ("ACT", "MEANS", "PURPOSE"))
    )
    if not hr61_ok:
        add_issue(
            case_id="HRV1-061", field="AMP/status", severity="ERROR", blocks=True,
            code="HRV1_061_CORRECTION_NOT_PRESENT",
            message="Expected ABSTAIN with empty Act/Means/Purpose after manual correction",
        )
    sjip_count = sum(
        "sjip" in str(item["raw"].get("annotation_notes", "")).casefold()
        for item in processed
    )
    if sjip_count:
        add_issue(
            field="annotation_notes", severity="ERROR", blocks=True, code="SJIP_TYPO_PRESENT",
            message=f"Found {sjip_count} remaining Sjip annotation-note typo(s)",
        )

    status_counts = Counter(item["status"] for item in processed)
    blocker_count = sum(int(item["blocks_scoring"]) for item in issues)
    severity_counts = Counter(item["severity"] for item in issues)
    qc_summary: dict[str, Any] = {
        "qc_schema_version": VERSION,
        "status": "PASS" if blocker_count == 0 else "BLOCKED",
        "source_path": _relative(source_path, repo_root),
        "source_sha256": source_sha_before,
        "source_row_count": len(rows),
        "unique_case_id_count": len(seen),
        "reviewed_n": status_counts["SUBSTANTIVE"] + status_counts["ABSTAIN"] + status_counts["SKIP"],
        "not_reviewed_n": status_counts["NOT_REVIEWED"],
        "substantive_n": status_counts["SUBSTANTIVE"],
        "abstain_n": status_counts["ABSTAIN"],
        "skip_n": status_counts["SKIP"],
        "retained_n": status_counts["SUBSTANTIVE"] + status_counts["ABSTAIN"],
        "blocking_issue_count": blocker_count,
        "blocking_error_count": blocker_count,
        "error_count": severity_counts["ERROR"],
        "warning_count": severity_counts["WARNING"],
        "info_count": severity_counts["INFO"],
        "deterministic_syntax_normalization_field_count": syntax_normalization_count,
        "duplicate_set_normalization_field_count": duplicate_normalization_count,
        "retained_ocg_positive_n": ocg_extraction_count,
        "hrv1_061_abstain_empty_amp_confirmed": hr61_ok,
        "sjip_typo_count": sjip_count,
        "done_is_sole_review_gate": True,
        "unreviewed_label_cells_used": False,
        "semantic_label_mapping_performed": False,
    }
    _atomic_csv(qc_report_path, issues, QC_COLUMNS)
    _atomic_json(qc_summary_path, qc_summary)
    if blocker_count:
        raise HumanReferenceError(
            f"Human annotation QC has {blocker_count} scoring-blocking issue(s); reference not frozen"
        )

    reference_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    source_rel = _relative(source_path, repo_root)
    context_rel = _relative(context_path, repo_root)
    for item in processed:
        raw_row = item["raw"]
        context = item["context"]
        status = item["status"]
        retained = status in {"SUBSTANTIVE", "ABSTAIN"}
        exclusion_reason = (
            "" if retained else "REVIEWER_SKIP" if status == "SKIP" else "NOT_REVIEWED_DONE_BLANK"
        )
        membership_rows.append(
            {
                "reliability_case_id": item["case_id"],
                "reviewer_order": context["reviewer_order"],
                "search_rank": context["search_rank"],
                "canonical_url": context["canonical_url"],
                "jurisdiction": context["jurisdiction_raw"],
                "input_sha256": item["input_sha256"],
                "reviewed": int(status != "NOT_REVIEWED"),
                "review_status": status,
                "retained": int(retained),
                "substantive_amp_evaluable": int(status == "SUBSTANTIVE"),
                "abstain_subset": int(status == "ABSTAIN"),
                "excluded": int(not retained),
                "exclusion_reason": exclusion_reason,
                "source_row_number": item["source_row_number"],
                "source_sha256": source_sha_before,
            }
        )
        if not retained:
            exclusion_rows.append(
                {
                    "reliability_case_id": item["case_id"],
                    "reviewer_order": context["reviewer_order"],
                    "search_rank": context["search_rank"],
                    "case_title": context["case_title"],
                    "canonical_url": context["canonical_url"],
                    "jurisdiction": context["jurisdiction_raw"],
                    "reviewed": int(status != "NOT_REVIEWED"),
                    "review_status": status,
                    "exclusion_reason": exclusion_reason,
                    "done_raw": raw_row["Done?"],
                    "annotation_notes": raw_row["annotation_notes"],
                    "source_file": source_rel,
                    "source_sha256": source_sha_before,
                    "source_row_number": item["source_row_number"],
                    "input_sha256": item["input_sha256"],
                }
            )
            continue

        amp = item["amp"]
        geo_values = item["geo"].values
        geo_clean = [value for value in GEO_ALLOWED if value in geo_values]
        machine_ids = {
            family: [AMP_ID_BY_RAW_LABEL[value] for value in amp[family].values]
            for family in ("ACT", "MEANS", "PURPOSE")
        }
        reference_rows.append(
            {
                "reference_schema_version": VERSION,
                "reliability_case_id": item["case_id"],
                "reviewer_order": context["reviewer_order"],
                "search_rank": context["search_rank"],
                "case_title": context["case_title"],
                "unodc_case_number": context["unodc_case_number"],
                "canonical_url": context["canonical_url"],
                "jurisdiction": context["jurisdiction_raw"],
                "fact_summary": raw_row["english_fact_summary_raw"],
                "reviewed": 1,
                "review_status": status,
                "done_raw": raw_row["Done?"],
                "acts_human_raw": raw_row[ACT_SOURCE_COLUMN],
                "acts_human_clean_json": json.dumps(amp["ACT"].values, ensure_ascii=False),
                "act_label_ids_json": json.dumps(machine_ids["ACT"], ensure_ascii=False),
                "act_labels": json.dumps(machine_ids["ACT"], ensure_ascii=False),
                "means_human_raw": raw_row[MEANS_SOURCE_COLUMN],
                "means_human_clean_json": json.dumps(amp["MEANS"].values, ensure_ascii=False),
                "means_label_ids_json": json.dumps(machine_ids["MEANS"], ensure_ascii=False),
                "means_labels": json.dumps(machine_ids["MEANS"], ensure_ascii=False),
                "purpose_human_raw": raw_row[PURPOSE_SOURCE_COLUMN],
                "purpose_human_clean_json": json.dumps(amp["PURPOSE"].values, ensure_ascii=False),
                "purpose_label_ids_json": json.dumps(machine_ids["PURPOSE"], ensure_ascii=False),
                "purpose_labels": json.dumps(machine_ids["PURPOSE"], ensure_ascii=False),
                "geographic_form_human_raw": raw_row[GEO_SOURCE_COLUMN],
                "geographic_form_human_clean_json": json.dumps(geo_clean, ensure_ascii=False),
                "organized_criminal_group_human": "TRUE" if OCG_RAW_LABEL in geo_values else "FALSE",
                "organized_criminal_group_evaluable": int(status == "SUBSTANTIVE"),
                "multiplicity_human_raw": raw_row[MULTIPLICITY_SOURCE_COLUMN],
                "multiplicity_human_clean": item["multiplicity"],
                "child_human_raw": raw_row[CHILD_SOURCE_COLUMN],
                "child_human_clean": item["child"],
                "annotation_notes": raw_row["annotation_notes"],
                "substantive_amp_evaluable": int(status == "SUBSTANTIVE"),
                "auxiliary_evaluable": int(status == "SUBSTANTIVE"),
                "source_file": source_rel,
                "source_sha256": source_sha_before,
                "source_row_number": item["source_row_number"],
                "input_sha256": item["input_sha256"],
                "context_file": context_rel,
                "context_sha256": context_sha,
            }
        )

    source_sha_after = sha256_file(source_path)
    if source_sha_after != source_sha_before:
        raise HumanReferenceError("Immutable annotation source changed during processing")

    _atomic_csv(reference_path, reference_rows, REFERENCE_COLUMNS)
    _atomic_csv(exclusions_path, exclusion_rows, EXCLUSION_COLUMNS)
    _atomic_csv(membership_path, membership_rows, MEMBERSHIP_COLUMNS)

    source_manifest_sha = sha256_file(source_manifest_path)
    qc_summary_sha = sha256_file(qc_summary_path)
    reference_sha = sha256_file(reference_path)
    exclusions_sha = sha256_file(exclusions_path)
    membership_sha = sha256_file(membership_path)
    retained_digest = _membership_digest(membership_rows, retained_only=True)
    substantive_digest = _membership_digest(membership_rows, substantive_only=True)
    retained_members = sorted(
        [
            {
                "reliability_case_id": row["reliability_case_id"],
                "search_rank": int(row["search_rank"]),
                "canonical_url": row["canonical_url"],
                "input_sha256": row["input_sha256"],
                "review_status": row["review_status"],
                "substantive_amp_evaluable": bool(row["substantive_amp_evaluable"]),
            }
            for row in membership_rows
            if int(row["retained"]) == 1
        ],
        key=lambda member: member["search_rank"],
    )
    overlap = [
        member for member in retained_members if member["search_rank"] in active_demo_ranks
    ]
    freeze_manifest: dict[str, Any] = {
        "manifest_schema_version": VERSION,
        "freeze_id": "single-reviewer-human-grounded-reference-v1",
        "status": FREEZE_STATUS,
        "frozen_at_utc": audit_timestamp,
        "single_reviewer_design": True,
        "inter_annotator_statistics_permitted": False,
        "authoritative_review_gate": "Done? == 1",
        "source_manifest": {
            "path": _relative(source_manifest_path, repo_root),
            "sha256": source_manifest_sha,
        },
        "retained_n": len(reference_rows),
        "substantive_n": qc_summary["substantive_n"],
        "abstain_n": qc_summary["abstain_n"],
        "retained_membership_sha256": retained_digest,
        "human_reference_path": _relative(reference_path, repo_root),
        "human_reference_sha256": reference_sha,
        "membership_path": _relative(membership_path, repo_root),
        "membership_sha256": membership_sha,
        "human_reference": {
            "path": _relative(reference_path, repo_root),
            "sha256": reference_sha,
            "retained_n": len(reference_rows),
        },
        "exclusions": {
            "path": _relative(exclusions_path, repo_root),
            "sha256": exclusions_sha,
            "excluded_n": len(exclusion_rows),
        },
        "membership": {
            "path": _relative(membership_path, repo_root),
            "sha256": membership_sha,
            "all_source_n": len(membership_rows),
            "retained_membership_sha256": retained_digest,
            "substantive_membership_sha256": substantive_digest,
        },
        "qc": {
            "summary_path": _relative(qc_summary_path, repo_root),
            "summary_sha256": qc_summary_sha,
            "status": qc_summary["status"],
            "blocking_issue_count": qc_summary["blocking_issue_count"],
        },
        "counts": {
            "source_n": len(membership_rows),
            "reviewed_n": qc_summary["reviewed_n"],
            "not_reviewed_n": qc_summary["not_reviewed_n"],
            "skip_n": qc_summary["skip_n"],
            "abstain_n": qc_summary["abstain_n"],
            "substantive_n": qc_summary["substantive_n"],
            "retained_n": qc_summary["retained_n"],
        },
        "retained_members": retained_members,
        "a1_active_m4_demo_overlap_audit": {
            "a1_split_path": _relative(a1_split_path, repo_root),
            "a1_split_sha256": a1_sha,
            "active_demo_search_ranks": sorted(active_demo_ranks),
            "overlap_n": len(overlap),
            "overlap_members": overlap,
            "status": "PASS_NO_OVERLAP" if not overlap else "OVERLAP_REQUIRES_EXCLUSION",
            "human_labels_inspected_for_overlap": False,
        },
        "rules": {
            "skip_excluded": True,
            "abstain_retained_separate": True,
            "abstain_in_ordinary_amp_scoring": False,
            "unreviewed_prepopulated_values_used": False,
            "semantic_normalization_performed": False,
            "organized_criminal_group_split_from_geographic_form": True,
        },
    }
    _atomic_json(freeze_manifest_path, freeze_manifest)
    return HumanReferenceBuildResult(
        source_manifest=source_manifest,
        qc_summary=qc_summary,
        qc_rows=issues,
        reference_rows=reference_rows,
        exclusion_rows=exclusion_rows,
        membership_rows=membership_rows,
        freeze_manifest=freeze_manifest,
    )


__all__ = [
    "ACT_SOURCE_COLUMN",
    "FREEZE_STATUS",
    "HumanReferenceBuildResult",
    "HumanReferenceError",
    "ParsedList",
    "build_single_reviewer_reference",
    "classify_review_status",
    "parse_human_list",
    "sha256_file",
]
