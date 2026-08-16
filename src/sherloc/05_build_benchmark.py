#!/usr/bin/env python3
"""Build the frozen SHERLOC benchmark-v1 and blinded annotation package.

This stage consumes parser-v2 JSONL and performs only versioned reference
construction.  It does not run models, normalize raw SHERLOC AMP labels, or
create train/validation/test splits.  The implementation uses the Python
standard library so the outputs can be rebuilt without an environment-specific
dataframe or YAML dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit


BUILDER_VERSION = "1.0.0"
BENCHMARK_SCHEMA_VERSION = "sherloc-benchmark-v1"
SENTENCE_SPLITTER_VERSION = "sherloc_sentence_splitter_v1"
SNAPSHOT_DATE = "2026-08-09"
BUILD_FREEZE_DATE = "2026-08-11"

PRIMARY_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_SOURCE_SHA256 = (
    "ea0592fcb633a0eee55e5feacb02fc1ef119cfcbb0f594566b4da6420eb184df"
)
EXPECTED_PRIMARY_MEMBERSHIP_SHA256 = (
    "097ce2027171ebc9cac5ad6dfdbf6e854729f81a8ede78e8401086fe5d5ed48c"
)
EXPECTED_RELIABILITY_MEMBERSHIP_SHA256 = (
    "39bb96284b94ac5dd95e89d12c57dfdf09593d1add6b7d6737172ea01c32cd4b"
)
EXPECTED_PRIOR_MANUAL_REVIEW_SHA256 = (
    "d644ee79983f6720c78a92939748f8fcdc7701c5b739f8bb4d13d9278cd4b360"
)
EXPECTED_ALL_CASES = 1590
EXPECTED_ENGLISH_UNIVERSE = 1565
EXPECTED_PRIMARY_COHORT = 1263

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/interim/sherloc_cases_raw.jsonl"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_BENCHMARK_JSONL = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_BENCHMARK_CSV = REPO_ROOT / "data/processed/sherloc_benchmark_v1.csv"
DEFAULT_MULTIPLICITY_QUEUE = (
    REPO_ROOT / "data/annotations/multiplicity_single_review_queue.csv"
)
DEFAULT_RELIABILITY_SAMPLE = (
    REPO_ROOT / "data/annotations/reliability_sample_100.csv"
)
DEFAULT_REVIEWER_TEMPLATE = (
    REPO_ROOT / "data/annotations/reviewer_annotation_template.csv"
)
DEFAULT_REFERENCE_KEY = (
    REPO_ROOT / "data/annotations/reliability_sample_100_reference_key.csv"
)
DEFAULT_REPORT = REPO_ROOT / "docs/benchmark_v1_report.md"
DEFAULT_PRIOR_MANUAL_REVIEW = (
    REPO_ROOT / "outputs/tables/auxiliary_feature_manual_review.csv"
)


class BenchmarkBuildError(RuntimeError):
    """Raised when a frozen invariant or output validation fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def fact_summary(record: dict[str, Any]) -> str:
    return (
        record.get("narrative", {})
        .get("fact_summary", {})
        .get("english_text_raw")
        or ""
    ).strip()


def legacy_values(record: dict[str, Any], key: str) -> list[str]:
    return ordered_unique(
        record.get("legacy_keywords", {})
        .get("core_fields", {})
        .get(key, {})
        .get("values_raw", [])
    )


def sidebar_values(record: dict[str, Any], key: str) -> list[str]:
    return ordered_unique(
        record.get("trafficking_sidebar", {})
        .get("fields", {})
        .get(key, {})
        .get("values_raw", [])
    )


def usable_english(record: dict[str, Any]) -> bool:
    return bool(fact_summary(record))


def in_primary_cohort(record: dict[str, Any]) -> bool:
    return bool(
        usable_english(record)
        and legacy_values(record, "acts")
        and legacy_values(record, "means")
        and legacy_values(record, "exploitative_purposes")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BenchmarkBuildError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return records


def primary_membership_digest(records: Sequence[dict[str, Any]]) -> str:
    payload = "".join(
        f"{int(record['provenance']['search_rank'])}\t"
        f"{record['provenance']['canonical_url']}\n"
        for record in sorted(
            records, key=lambda item: int(item["provenance"]["search_rank"])
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_and_load_ontology(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    # JSON is a strict subset of YAML 1.2; keeping this file JSON-compatible
    # avoids an undeclared PyYAML dependency while preserving the requested
    # .yaml artifact.
    try:
        ontology = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkBuildError(f"Cannot load ontology {path}: {exc}") from exc

    expected_sizes = {"ACT": 5, "MEANS": 6, "PURPOSE": 6}
    maps: dict[str, dict[str, str]] = {}
    all_ids: set[str] = set()
    for family, expected_size in expected_sizes.items():
        labels = ontology.get("families", {}).get(family)
        if not isinstance(labels, list) or len(labels) != expected_size:
            raise BenchmarkBuildError(
                f"Ontology family {family} must contain exactly {expected_size} labels"
            )
        indices = [item.get("index") for item in labels]
        if indices != list(range(expected_size)):
            raise BenchmarkBuildError(
                f"Ontology family {family} indices are not stable zero-based order"
            )
        family_map: dict[str, str] = {}
        for item in labels:
            if item.get("family") != family:
                raise BenchmarkBuildError(f"Ontology family mismatch in {item!r}")
            raw = item.get("raw_sherloc_label")
            label_id = item.get("id")
            if not isinstance(raw, str) or not raw or not isinstance(label_id, str):
                raise BenchmarkBuildError(f"Malformed ontology label: {item!r}")
            if raw in family_map or label_id in all_ids:
                raise BenchmarkBuildError(f"Duplicate ontology mapping: {item!r}")
            family_map[raw] = label_id
            all_ids.add(label_id)
        maps[family] = family_map
    return ontology, maps


def person_sections(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        section
        for section in record.get("participants", {}).get("sections", [])
        if section.get("role_family") == "person_role"
    ]


def person_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for section in person_sections(record)
        for item in (section.get("records") or [])
        if isinstance(item, dict)
    ]


def field_pairs(record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (
            (field.get("label_raw") or "").strip(),
            (field.get("value_raw") or "").strip(),
        )
        for field in (record.get("fields") or [])
        if isinstance(field, dict)
    ]


def role_field(record: dict[str, Any]) -> dict[str, str]:
    matches = [
        field
        for field in (record.get("fields") or [])
        if field.get("class_key") == "victimPlaintiffType"
    ]
    return {
        "label": next(
            (
                (field.get("label_raw") or "").strip()
                for field in matches
                if (field.get("label_raw") or "").strip()
            ),
            "",
        ),
        "value": next(
            (
                (field.get("value_raw") or "").strip()
                for field in matches
                if (field.get("value_raw") or "").strip()
            ),
            "",
        ),
        "raw": (record.get("raw_text") or "").strip(),
    }


# Frozen multiplicity feasibility policy. The expressions are preserved from
# the audited policy rather than re-tuned during benchmark construction.
NUMBER = (
    r"(?:2|3|4|5|6|7|8|9|[1-9][0-9]+|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|hundreds?|thousands?)"
)
QUANT = (
    r"(?:several|multiple|numerous|many|dozens?|various|at\s+least\s+"
    + NUMBER
    + r"|more\s+than\s+"
    + NUMBER
    + r"|over\s+"
    + NUMBER
    + r"|"
    + NUMBER
    + r")"
)
PEOPLE = (
    r"(?:victims?|complainants?|plaintiffs?|individuals?|persons?|people|women|"
    r"girls|men|boys|children|minors?|migrants?|workers?|employees?|sex\s+workers?|"
    r"foreigners?|citizens?|nationals?|females?|males?)"
)
GROUP_RE = re.compile(r"\b" + QUANT + r"\s+(?:anonymous\s+)?" + PEOPLE + r"\b", re.I)
LEADING_ANON_RE = re.compile(r"^\s*" + QUANT + r"\s+anonymous(?:\b|$)", re.I)
PLURAL_RE = re.compile(
    r"\b(?:victims|individuals|persons|people|women|girls|men|boys|children|minors|"
    r"migrants|workers|employees|foreigners|citizens|nationals|females|males)\b",
    re.I,
)
COMMA_RE = re.compile(r"^(?:\s*[A-Z](?:\.[A-Z])?\.?\s*,){2,}", re.I)
TOTAL_RE = re.compile(
    r"\b(?:total\s+)?number\s+of\s+(?:the\s+)?victims?\s*[:=]?\s*"
    + NUMBER
    + r"\b",
    re.I,
)

PROPOSAL_NUMBER = (
    r"(?:2|3|4|5|6|7|8|9|[1-9][0-9]+|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|hundred(?:s)?|thousand(?:s)?)"
)
PROPOSAL_QUANT = (
    r"(?:several|multiple|numerous|many|dozens?|scores?|various|a\s+number\s+of|"
    r"a\s+group\s+of|groups?\s+of|at\s+least\s+"
    + PROPOSAL_NUMBER
    + r"|more\s+than\s+"
    + PROPOSAL_NUMBER
    + r"|over\s+"
    + PROPOSAL_NUMBER
    + r"|approximately\s+"
    + PROPOSAL_NUMBER
    + r"|about\s+"
    + PROPOSAL_NUMBER
    + r"|"
    + PROPOSAL_NUMBER
    + r")"
)
PROPOSAL_PEOPLE = (
    r"(?:victims?|complainants?|plaintiffs?|individuals?|persons?|people|women|girls|"
    r"men|boys|children|minors?|migrants?|workers?|employees?|sex\s+workers?|"
    r"foreigners?|nationals?|females?|males?)"
)
PROPOSAL_GROUP_ROLE_RE = re.compile(
    r"\b" + PROPOSAL_QUANT + r"\s+(?:anonymous\s+)?" + PROPOSAL_PEOPLE + r"\b",
    re.I,
)
PROPOSAL_PLURAL_ROLE_RE = re.compile(
    r"\b(?:anonymous\s+)?(?:victims|individuals|persons|people|women|girls|men|boys|"
    r"children|minors|migrants|workers|employees|foreigners|nationals|females|males)\b",
    re.I,
)
PROPOSAL_TOTAL_RE = re.compile(
    r"\b(?:total\s+)?number\s+of\s+(?:the\s+)?victims?\s*[:=]?\s*(?:"
    + PROPOSAL_NUMBER
    + r")\b",
    re.I,
)
PROPOSAL_LIST_RE = re.compile(r"^(?:\s*[A-Z](?:\.[A-Z])?\.?\s*,){2,}", re.I)
NARRATIVE_MULTIPLE_PATTERNS = [
    re.compile(
        r"\b"
        + PROPOSAL_QUANT
        + r"\s+(?:alleged\s+|potential\s+|identified\s+|trafficking\s+)?victims?\b",
        re.I,
    ),
    re.compile(r"\b(?:the|these|those|both|all)\s+(?:alleged\s+|potential\s+)?victims\b", re.I),
    re.compile(r"\bvictim\s*(?:no\.?\s*)?1\b.{0,250}\bvictim\s*(?:no\.?\s*)?2\b", re.I | re.S),
    re.compile(
        r"\b"
        + PROPOSAL_QUANT
        + r"\s+"
        + PROPOSAL_PEOPLE
        + r"\b.{0,100}\b(?:traffick\w*|exploit\w*|recruit\w*|transport\w*|force\w*|enslav\w*)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\b(?:traffick\w*|exploit\w*|recruit\w*|transport\w*|force\w*|enslav\w*)\b"
        r".{0,100}\b"
        + PROPOSAL_QUANT
        + r"\s+"
        + PROPOSAL_PEOPLE
        + r"\b",
        re.I | re.S,
    ),
    re.compile(
        r"\bvictims\s+(?:were|had|have|who|stated|testified|reported|worked|lived|"
        r"travelled|traveled|arrived|escaped|received|paid|suffered|came|left|entered|"
        r"returned|remained|included|identified)\b",
        re.I,
    ),
]

MULTI_RECORD_SINGLE_EXCEPTIONS = {468, 825, 826}
MULTI_RECORD_UNKNOWN_EXCEPTIONS = {86, 127, 353, 354, 618, 1393}


def direct_group(record: dict[str, Any]) -> bool:
    records = person_records(record)
    if len(records) != 1:
        return False
    role = role_field(records[0])
    if not (role["label"] or role["value"] or role["raw"]):
        return False
    return bool(
        GROUP_RE.search(role["value"])
        or LEADING_ANON_RE.search(role["value"])
        or PLURAL_RE.search(role["value"])
        or COMMA_RE.search(role["value"])
        or TOTAL_RE.search(role["raw"])
    )


def proposal_single(record: dict[str, Any]) -> bool:
    records = person_records(record)
    if len(records) != 1:
        return False
    role = role_field(records[0])
    target = (role["value"] + " " + role["raw"]).strip()
    structured_multiple = bool(
        PROPOSAL_GROUP_ROLE_RE.search(target)
        or PROPOSAL_PLURAL_ROLE_RE.search(role["value"])
        or PROPOSAL_TOTAL_RE.search(role["raw"])
        or PROPOSAL_LIST_RE.search(role["value"])
    )
    text = fact_summary(record)
    narrative_multiple = any(pattern.search(text) for pattern in NARRATIVE_MULTIPLE_PATTERNS)
    victimish = role["label"].lower().rstrip(":") in {"victim", "complainant"}
    singular_mentions = len(re.findall(r"\bthe\s+(?:alleged\s+)?victim\b", text, re.I))
    literal_plural = bool(re.search(r"\bvictims\b", text, re.I))
    return bool(
        not structured_multiple
        and not narrative_multiple
        and victimish
        and singular_mentions >= 2
        and not literal_plural
    )


def multiplicity_classify(record: dict[str, Any]) -> tuple[str, str]:
    rank = int(record["provenance"]["search_rank"])
    count = len(person_records(record))
    if count == 0:
        return "UNKNOWN", "NO_PERSON_SECTION_INDEPENDENT_REFERENCE"
    if count == 1:
        if direct_group(record):
            return "MULTIPLE", "ONE_RECORD_EXPLICIT_AGGREGATE_ROLE_VALUE"
        if proposal_single(record):
            return (
                "SINGLE",
                "ONE_INDIVIDUAL_VICTIM_RECORD_AND_CONSISTENTLY_SINGULAR_SUMMARY",
            )
        return "UNKNOWN", "ONE_RECORD_WITHOUT_CONCLUSIVE_NEGATIVE_OR_GROUP_REFERENCE"
    if rank in MULTI_RECORD_SINGLE_EXCEPTIONS:
        return "SINGLE", "MULTI_RECORD_ROLE_OR_DUPLICATE_EXCEPTION_SUPPORTS_ONE_VICTIM"
    if rank in MULTI_RECORD_UNKNOWN_EXCEPTIONS:
        return "UNKNOWN", "MULTI_RECORD_ROLE_SEMANTICS_AMBIGUOUS_FOR_TRAFFICKING_VICTIMS"
    return (
        "MULTIPLE",
        "MULTI_RECORD_STRUCTURE_SCREENED_WITH_DISTINCT_OR_CORROBORATED_VICTIMS",
    )


def normalized_record_duplicate(record: dict[str, Any]) -> bool:
    normalized = [
        re.sub(r"\W+", " ", (item.get("raw_text") or "")).lower().strip()
        for item in person_records(record)
        if (item.get("raw_text") or "").strip()
    ]
    return len(normalized) != len(set(normalized))


def multiplicity_structural_flags(record: dict[str, Any], label: str) -> list[str]:
    flags: list[str] = []
    records = person_records(record)
    sections = person_sections(record)
    role_labels = {role_field(item)["label"] for item in records}
    rank = int(record["provenance"]["search_rank"])
    if label == "SINGLE":
        flags.append("PROVISIONAL_SINGLE")
    if normalized_record_duplicate(record):
        flags.append("EXACT_DUPLICATE_NORMALIZED_RECORD")
    if {"Victim:", "Plaintiff:"} <= role_labels:
        flags.append("MIXED_VICTIM_PLAINTIFF_ROLE_LABELS")
    if any(
        section.get("visible_section_heading_raw") == "Migrants"
        for section in sections
    ):
        flags.append("MIGRANTS_HEADING")
    if not sections:
        flags.append("PERSON_SECTION_ABSENT")
    if rank in MULTI_RECORD_UNKNOWN_EXCEPTIONS:
        flags.append("MULTI_RECORD_ROLE_SEMANTICS_UNKNOWN")
    return flags


MANDATORY_MULTIPLICITY_QUEUE_FLAGS = {
    "PROVISIONAL_SINGLE",
    "EXACT_DUPLICATE_NORMALIZED_RECORD",
    "MIXED_VICTIM_PLAINTIFF_ROLE_LABELS",
    "MIGRANTS_HEADING",
    "PERSON_SECTION_ABSENT",
    "MULTI_RECORD_ROLE_SEMANTICS_UNKNOWN",
}


# Frozen strict child/minor support policy.
def role_text(record: dict[str, Any]) -> str:
    return " ".join(
        value
        for label, value in field_pairs(record)
        if label in ("Victim:", "Migrant:", "Plaintiff:", "Complainant:")
    )


def grouped_child_record(record: dict[str, Any]) -> bool:
    return bool(
        re.search(
            r"\b(?:several|multiple|numerous|dozens?|many|over|more than|at least|"
            r"unknown number|individuals|persons|citizens|women|men|girls|boys|children|"
            r"victims)\b|(?:^|\b)(?:two|three|four|five|six|seven|eight|nine|ten|\d+)"
            r"\+?\s+(?!year)(?:anonymous|individual|person|citizen|victim|women|men|"
            r"girl|boy|adult)",
            role_text(record),
            re.I,
        )
    )


def base_child_signal(record: dict[str, Any]) -> bool:
    pairs = field_pairs(record)
    return bool(
        any(label == "Gender:" and value.casefold() == "child" for label, value in pairs)
        or any(
            label == "Age:" and re.fullmatch(r"\d{1,2}", value) and int(value) < 18
            for label, value in pairs
        )
        or any(
            label in ("Victim:", "Migrant:")
            and re.search(r"\b(?:child|children|minor|minors|newborn)\b", value, re.I)
            for label, value in pairs
        )
    )


def explicit_adult_signal(record: dict[str, Any]) -> bool:
    pairs = field_pairs(record)
    raw = " ".join(value for _, value in pairs)
    explicit_adult_word = bool(re.search(r"\badults?\b", raw, re.I))
    if grouped_child_record(record):
        return explicit_adult_word
    return bool(
        explicit_adult_word
        or any(
            label == "Age:"
            and re.fullmatch(r"\d{1,3}", value)
            and 18 <= int(value) <= 120
            for label, value in pairs
        )
    )


CHILD_DIRECT_TRUE_ADDITIONS = {21, 39, 202, 285, 510, 635, 898, 1130, 1276, 1581}
CHILD_UNCERTAIN_OVERRIDES = {235, 546, 889, 953, 978, 1462}
CHILD_EXPLICIT_FALSE_OVERRIDES = {850}
CHILD_NARRATIVE_EXCLUSIONS_FROM_ADULT = {370, 643}

CHILD_WORD_RE = re.compile(
    r"\b(?:minor|minors|child|children|underage|under-aged|under aged|juvenile|"
    r"juveniles|newborn|infant|infants)\b",
    re.I,
)
CHILD_AGE_PATTERNS = [
    ("AGE_YEARS_OLD", re.compile(r"\b(\d{1,2})[- ]years?[- ]old\b", re.I)),
    ("AGED_NUMBER", re.compile(r"\baged?\s+(\d{1,2})\b", re.I)),
    (
        "COPULA_NUMBER_YEARS",
        re.compile(r"\b(?:was|is|were)\s+(\d{1,2})\s+years?\s+(?:of age|old)\b", re.I),
    ),
    ("UNDER_AGE_OF_NUMBER", re.compile(r"\bunder the age of\s+(\d{1,2})\b", re.I)),
]
ADULT_WORD_RE = re.compile(
    r"\badult(?:s| women| men| females| males| victims| persons| workers)\b",
    re.I,
)


def child_provisional_sets(
    cohort: Sequence[dict[str, Any]],
) -> dict[str, set[int]]:
    by_rank = {int(item["provenance"]["search_rank"]): item for item in cohort}
    base_true = {
        rank
        for rank, item in by_rank.items()
        if any(base_child_signal(person) for person in person_records(item))
    }
    child_sidebar = {
        rank
        for rank, item in by_rank.items()
        if "Trafficking in children (under 18 years)" in sidebar_values(item, "offences")
    }
    true = (
        base_true | CHILD_DIRECT_TRUE_ADDITIONS | child_sidebar
    ) - CHILD_UNCERTAIN_OVERRIDES - CHILD_EXPLICIT_FALSE_OVERRIDES
    adult_candidates = {
        rank
        for rank, item in by_rank.items()
        if person_records(item)
        and all(explicit_adult_signal(person) for person in person_records(item))
        and not any(base_child_signal(person) for person in person_records(item))
    }
    false = (
        adult_candidates
        - true
        - CHILD_UNCERTAIN_OVERRIDES
        - CHILD_NARRATIVE_EXCLUSIONS_FROM_ADULT
    ) | CHILD_EXPLICIT_FALSE_OVERRIDES
    return {
        "base_true": base_true,
        "child_sidebar": child_sidebar,
        "adult_candidates": adult_candidates,
        "TRUE": true,
        "FALSE": false,
        "UNKNOWN": set(by_rank) - true - false,
    }


def fact_summary_child_support_details(text: str) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    word = CHILD_WORD_RE.search(text)
    if word:
        details.append({"type": "EXPLICIT_CHILD_WORD", "match": word.group(0)})
    for name, pattern in CHILD_AGE_PATTERNS:
        for match in pattern.finditer(text):
            age = int(match.group(1))
            if age < 18:
                details.append({"type": name, "match": match.group(0), "age": age})
    return details


def fact_summary_adult_support(
    record: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    text = fact_summary(record)
    records = person_records(record)
    if not records:
        return False, {"reason": "NO_PERSON_RECORDS"}
    adult_word = ADULT_WORD_RE.search(text)
    if any(grouped_child_record(item) for item in records):
        return bool(adult_word), {
            "reason": "GROUPED_RECORD_REQUIRES_EXPLICIT_ADULT_WORD",
            "match": adult_word.group(0) if adult_word else None,
        }
    ages: list[int] = []
    for item in records:
        item_ages = [
            int(value)
            for label, value in field_pairs(item)
            if label == "Age:"
            and re.fullmatch(r"\d{1,3}", value)
            and 18 <= int(value) <= 120
        ]
        if not item_ages:
            return bool(adult_word), {
                "reason": "AGE_MISSING_REQUIRES_EXPLICIT_ADULT_WORD",
                "match": adult_word.group(0) if adult_word else None,
            }
        ages.extend(item_ages)
    unmatched: list[int] = []
    matches: dict[str, str] = {}
    for age in sorted(set(ages)):
        match = re.search(
            rf"\b(?:aged?|age(?:d)? of|was|is|then)\s*(?:about |approximately )?{age}\b|"
            rf"\b{age}[- ]years?[- ]old\b",
            text,
            re.I,
        )
        if match:
            matches[str(age)] = match.group(0)
        else:
            unmatched.append(age)
    return not unmatched, {
        "reason": "ALL_STRUCTURED_ADULT_AGES_MUST_APPEAR_IN_NARRATIVE",
        "structured_adult_ages": sorted(set(ages)),
        "matched_text": matches,
        "unmatched_ages": unmatched,
    }


def strict_child_labels(
    cohort: Sequence[dict[str, Any]],
) -> tuple[dict[int, str], dict[int, list[dict[str, Any]] | dict[str, Any]]]:
    by_rank = {int(item["provenance"]["search_rank"]): item for item in cohort}
    provisional = child_provisional_sets(cohort)
    labels = {rank: "UNKNOWN" for rank in by_rank}
    support: dict[int, list[dict[str, Any]] | dict[str, Any]] = {}
    for rank in sorted(provisional["TRUE"]):
        details = fact_summary_child_support_details(fact_summary(by_rank[rank]))
        if details:
            labels[rank] = "TRUE"
            support[rank] = details
    for rank in sorted(provisional["FALSE"]):
        supported, details = fact_summary_adult_support(by_rank[rank])
        if supported:
            labels[rank] = "FALSE"
            support[rank] = details
    return labels, support


RELIABILITY_BUCKET_RANKS = {
    "A": [13, 17, 38, 46, 53, 68, 107, 123, 145, 146, 167, 195, 202, 237, 238, 317, 375, 452, 489, 574, 582, 618, 751, 759, 781, 917, 940, 985, 1105, 1111, 1153, 1159, 1238, 1356, 1566],
    "B": [23, 52, 86, 177, 235, 254, 262, 274, 304, 313, 320, 476, 513, 546, 627, 889, 962, 1142, 1229, 1464],
    "C": [29, 34, 58, 266, 409, 418, 480, 509, 604, 633, 824, 842, 898, 999, 1399],
    "D": [25, 79, 474, 545, 615, 628, 640, 728, 769, 817, 933, 934, 984, 1173, 1515],
    "E": [54, 127, 285, 468, 643, 755, 764, 825, 839, 850, 1396, 1406, 1462, 1508, 1517],
}
RELIABILITY_BUCKET_LABELS = {
    "A": "Representative clean cases",
    "B": "Structured metadata present but narrative support incomplete",
    "C": "Structured metadata missing but narrative informative",
    "D": "Narrative genuinely insufficient or abstention candidate",
    "E": "Rare or challenge cases",
}
RELIABILITY_SEED = "sherloc-reliability-v1-2026-08-11"


PROTECTED_ABBREVIATIONS = {
    "art", "arts", "dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms",
    "no", "nos", "para", "paras", "prof", "sec", "secs", "sr", "st",
    "u.k", "u.s", "v", "vs",
}


def _sentence_boundary_allowed(text: str, punctuation_index: int) -> bool:
    punctuation = text[punctuation_index]
    if punctuation == ".":
        before = text[punctuation_index - 1] if punctuation_index else ""
        after = text[punctuation_index + 1] if punctuation_index + 1 < len(text) else ""
        if before.isdigit() and after.isdigit():
            return False
        token_match = re.search(r"([A-Za-z](?:[A-Za-z.]*)?)\.$", text[: punctuation_index + 1])
        token = token_match.group(1).casefold() if token_match else ""
        if token in PROTECTED_ABBREVIATIONS:
            return False
        if len(token) == 1 and token.isalpha():
            return False
    return True


def split_sentences_v1(text: str) -> list[str]:
    """Deterministically segment a SHERLOC Fact Summary for evidence IDs.

    CRLF is normalized; blank-line paragraphs and line-leading list items force
    splits; spaces inside each sentence are collapsed; punctuation splits only
    before a likely new sentence and protects decimals, common abbreviations,
    and initials.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    units: list[str] = []
    list_item_re = re.compile(r"^\s*(?:[-*\u2022]\s+|\(?[A-Za-z0-9]{1,3}[.)]\s+)")
    for paragraph in paragraphs:
        current: list[str] = []
        for line in paragraph.split("\n"):
            if list_item_re.match(line) and current:
                units.append(" ".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            units.append(" ".join(current))
    sentences: list[str] = []
    for unit in units:
        compact = re.sub(r"\s+", " ", unit).strip()
        start = 0
        index = 0
        while index < len(compact):
            if compact[index] not in ".!?" or not _sentence_boundary_allowed(compact, index):
                index += 1
                continue
            end = index + 1
            while end < len(compact) and compact[end] in ".!?\"'\u2019\u201d)]":
                end += 1
            next_index = end
            while next_index < len(compact) and compact[next_index].isspace():
                next_index += 1
            if next_index >= len(compact):
                candidate = compact[start:end].strip()
                if candidate:
                    sentences.append(candidate)
                start = end
                break
            if next_index > end and compact[next_index] in "\"'\u201c\u2018([":
                probe = next_index + 1
                while probe < len(compact) and compact[probe] in "\"'\u201c\u2018([":
                    probe += 1
            else:
                probe = next_index
            if next_index > end and probe < len(compact) and compact[probe].isupper():
                candidate = compact[start:end].strip()
                if candidate:
                    sentences.append(candidate)
                start = next_index
                index = next_index
            else:
                index = end
        tail = compact[start:].strip()
        if tail and (not sentences or sentences[-1] != tail):
            sentences.append(tail)
    return sentences


def numbered_sentences(sentences: Sequence[str]) -> str:
    return "\n".join(f"[S{index}] {sentence}" for index, sentence in enumerate(sentences, 1))


def source_url_year(url: str | None) -> int | None:
    if not url:
        return None
    match = re.search(r"/(19\d{2}|20\d{2})/", urlsplit(url).path)
    return int(match.group(1)) if match else None


def narrative_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['\u2019-]\w+)*\b", text, re.UNICODE))


def compact_excerpt(text: str, limit: int = 900) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    clipped = compact[:limit]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + "\u2026"


def build_benchmark_record(
    source: dict[str, Any],
    ontology_maps: dict[str, dict[str, str]],
    child_labels: dict[int, str],
    child_support: dict[int, list[dict[str, Any]] | dict[str, Any]],
) -> dict[str, Any]:
    rank = int(source["provenance"]["search_rank"])
    text = fact_summary(source)
    acts = legacy_values(source, "acts")
    means = legacy_values(source, "means")
    purposes = legacy_values(source, "exploitative_purposes")
    form = legacy_values(source, "form_of_trafficking")
    internal = "Internal" in form
    transnational = "Transnational" in form
    ocg = "Organized Criminal Group" in form
    mult_label, mult_basis = multiplicity_classify(source)
    mult_flags = multiplicity_structural_flags(source, mult_label)
    child_label = child_labels[rank]

    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_builder_version": BUILDER_VERSION,
        "primary_cohort_id": PRIMARY_COHORT_ID,
        "identity": {
            "search_rank": rank,
            "case_title_raw": source.get("case_identity", {}).get("title_raw"),
            "unodc_case_number": source.get("provenance", {}).get("unodc_case_number"),
            "canonical_url": source.get("provenance", {}).get("canonical_url"),
            "jurisdiction_country_raw": source.get("case_identity", {}).get("country_raw"),
            # Parser v2 does not expose a validated decision/verdict date.  Do
            # not silently reinterpret the URL path year as a decision year.
            "decision_or_verdict_year": None,
            "case_page_url_year": source_url_year(
                source.get("provenance", {}).get("canonical_url")
            ),
        },
        "text_input": {
            "english_fact_summary_raw": text,
            "word_count": narrative_word_count(text),
            "character_count": len(text),
        },
        "amp_targets": {
            "reference_source": "Legacy Keywords",
            "acts_raw": acts,
            "act_ontology_ids": [ontology_maps["ACT"][value] for value in acts],
            "means_raw": means,
            "means_ontology_ids": [ontology_maps["MEANS"][value] for value in means],
            "purposes_raw": purposes,
            "purpose_ontology_ids": [ontology_maps["PURPOSE"][value] for value in purposes],
        },
        "geographic_form": {
            "reference_source": "Legacy Keywords Form of Trafficking",
            "legacy_form_values_raw": form,
            "geographic_form_internal": int(internal),
            "geographic_form_transnational": int(transnational),
            "geographic_form_eligible": int(internal or transnational),
            "organized_criminal_group_present": int(ocg),
            "organized_criminal_group_raw_value": (
                "Organized Criminal Group" if ocg else None
            ),
        },
        "victim_multiplicity": {
            "policy_version": "conservative-feasibility-policy-v1",
            "multiplicity_provisional": mult_label,
            "multiplicity_eligible": int(mult_label in {"SINGLE", "MULTIPLE"}),
            "multiplicity_requires_human_confirmation": int(bool(mult_flags)),
            "provisional_rule": mult_basis,
            "ambiguity_flags": mult_flags,
        },
        "child_minor_exploratory": {
            "policy_version": "strict-narrative-support-policy-v1",
            "child_strict_label": child_label,
            "child_exploratory_eligible": int(child_label in {"TRUE", "FALSE"}),
            "support_basis": child_support.get(rank),
            "status": "EXPLORATORY_NOT_PRIMARY",
        },
        "secondary_metadata": {
            "sidebar_amp_policy": "RAW_SECONDARY_ONLY_NOT_PRIMARY_GROUND_TRUTH",
            "sidebar_acts_raw": sidebar_values(source, "acts"),
            "sidebar_means_raw": sidebar_values(source, "means"),
            "sidebar_exploitative_purposes_raw": sidebar_values(
                source, "exploitative_purposes"
            ),
        },
        "source_provenance": {
            "parser_schema_version": source.get("schema_version"),
            "parser_version": source.get("parser_provenance", {}).get("parser_version"),
            "parser_status": source.get("parser_provenance", {}).get("parse_status"),
            "raw_file_reference": source.get("source_input", {}).get("actual_path")
            or source.get("provenance", {}).get("download_manifest_raw_filename"),
            "raw_sha256": source.get("source_input", {}).get("computed_sha256")
            or source.get("provenance", {}).get("download_manifest_sha256"),
            "download_timestamp": source.get("provenance", {}).get("download_timestamp"),
            "requested_url": source.get("provenance", {}).get("requested_url"),
            "resolved_url": source.get("provenance", {}).get("resolved_url"),
            "api_result_id": source.get("provenance", {}).get("api_result_id"),
            "corpus_snapshot_date": SNAPSHOT_DATE,
        },
    }


def benchmark_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    identity = record["identity"]
    text = record["text_input"]
    amp = record["amp_targets"]
    form = record["geographic_form"]
    mult = record["victim_multiplicity"]
    child = record["child_minor_exploratory"]
    secondary = record["secondary_metadata"]
    provenance = record["source_provenance"]

    def encoded(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    return {
        "search_rank": identity["search_rank"],
        "case_title_raw": identity["case_title_raw"],
        "unodc_case_number": identity["unodc_case_number"],
        "canonical_url": identity["canonical_url"],
        "jurisdiction_country_raw": identity["jurisdiction_country_raw"],
        "decision_or_verdict_year": identity["decision_or_verdict_year"],
        "case_page_url_year": identity["case_page_url_year"],
        "primary_cohort_id": record["primary_cohort_id"],
        "english_fact_summary_raw": text["english_fact_summary_raw"],
        "narrative_word_count": text["word_count"],
        "narrative_character_count": text["character_count"],
        "legacy_acts_raw_json": encoded(amp["acts_raw"]),
        "act_ontology_ids_json": encoded(amp["act_ontology_ids"]),
        "legacy_means_raw_json": encoded(amp["means_raw"]),
        "means_ontology_ids_json": encoded(amp["means_ontology_ids"]),
        "legacy_purposes_raw_json": encoded(amp["purposes_raw"]),
        "purpose_ontology_ids_json": encoded(amp["purpose_ontology_ids"]),
        "legacy_form_values_raw_json": encoded(form["legacy_form_values_raw"]),
        "geographic_form_internal": form["geographic_form_internal"],
        "geographic_form_transnational": form["geographic_form_transnational"],
        "geographic_form_eligible": form["geographic_form_eligible"],
        "organized_criminal_group_present": form["organized_criminal_group_present"],
        "multiplicity_provisional": mult["multiplicity_provisional"],
        "multiplicity_eligible": mult["multiplicity_eligible"],
        "multiplicity_requires_human_confirmation": mult[
            "multiplicity_requires_human_confirmation"
        ],
        "multiplicity_rule": mult["provisional_rule"],
        "multiplicity_ambiguity_flags_json": encoded(mult["ambiguity_flags"]),
        "child_strict_label": child["child_strict_label"],
        "child_exploratory_eligible": child["child_exploratory_eligible"],
        "sidebar_acts_raw_json": encoded(secondary["sidebar_acts_raw"]),
        "sidebar_means_raw_json": encoded(secondary["sidebar_means_raw"]),
        "sidebar_purposes_raw_json": encoded(
            secondary["sidebar_exploitative_purposes_raw"]
        ),
        "raw_file_reference": provenance["raw_file_reference"],
        "raw_sha256": provenance["raw_sha256"],
        "parser_version": provenance["parser_version"],
    }


def relevant_person_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for section in person_sections(record):
        heading = section.get("visible_section_heading_raw") or "[unheaded person section]"
        for item in section.get("records") or []:
            raw = (item.get("raw_text") or "").strip()
            if raw:
                parts.append(f"[{heading}]\n{raw}")
    return "\n\n--- SOURCE PERSON RECORD ---\n\n".join(parts)


def multiplicity_queue_row(
    source: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    mult = benchmark["victim_multiplicity"]
    sections = person_sections(source)
    return {
        "search_rank": benchmark["identity"]["search_rank"],
        "case_title": benchmark["identity"]["case_title_raw"],
        "canonical_url": benchmark["identity"]["canonical_url"],
        "person_section_heading": " | ".join(
            ordered_unique(
                section.get("visible_section_heading_raw") for section in sections
            )
        ),
        "source_person_record_count": len(person_records(source)),
        "relevant_person_role_raw_text": relevant_person_text(source),
        "fact_summary_excerpt": compact_excerpt(fact_summary(source)),
        "provisional_multiplicity_label": mult["multiplicity_provisional"],
        "reason_rule_used": mult["provisional_rule"],
        "ambiguity_flags": "|".join(mult["ambiguity_flags"]),
        "currently_performance_eligible": mult["multiplicity_eligible"],
        "human_confirmation_required": 1,
        "reviewer_final_label": "",
        "reviewer_notes": "",
    }


def load_prior_manual_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    by_rank: dict[int, list[dict[str, str]]] = defaultdict(list)
    if not path.is_file():
        raise BenchmarkBuildError(f"Prior manual-review input is missing: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != EXPECTED_PRIOR_MANUAL_REVIEW_SHA256:
        raise BenchmarkBuildError(
            "Prior manual-review input changed: "
            f"{observed_sha256} != {EXPECTED_PRIOR_MANUAL_REVIEW_SHA256}"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                by_rank[int(row["search_rank"])].append(row)
            except (KeyError, ValueError):
                continue
    return by_rank


def derived_selection_reason(
    bucket: str,
    source: dict[str, Any],
    benchmark_by_rank: dict[int, dict[str, Any]],
    prior_manual: dict[int, list[dict[str, str]]],
) -> str:
    rank = int(source["provenance"]["search_rank"])
    manual = prior_manual.get(rank, [])
    if manual:
        anchors = []
        for row in manual[:3]:
            detail = (
                f"{row.get('feature') or 'feature'} {row.get('structured_narrative_case_type') or ''}"
                f" ({row.get('sample_stratum') or 'prior audit'})"
            ).strip()
            if row.get("ambiguity_reason"):
                detail += f": {row['ambiguity_reason']}"
            anchors.append(detail)
        return "Prior feasibility-audit anchor: " + "; ".join(anchors)

    base = RELIABILITY_BUCKET_LABELS[bucket]
    details: list[str] = []
    if rank in benchmark_by_rank:
        benchmark = benchmark_by_rank[rank]
        mult = benchmark["victim_multiplicity"]
        child = benchmark["child_minor_exploratory"]
        form = benchmark["geographic_form"]
        if mult["ambiguity_flags"]:
            details.append("multiplicity=" + ",".join(mult["ambiguity_flags"]))
        if child["child_strict_label"] != "UNKNOWN":
            details.append("strict-child-support candidate")
        if form["organized_criminal_group_present"]:
            details.append("OCG retained outside geographic target")
    if source.get("parser_provenance", {}).get("warning_count", 0):
        details.append("parser structural warning retained")
    if not in_primary_cohort(source):
        details.append("outside complete-Legacy-AMP cohort")
    return base + ("; " + "; ".join(details) if details else "")


def reliability_membership_digest(cases: Sequence[dict[str, Any]]) -> str:
    payload = "".join(
        f"{int(item['provenance']['search_rank'])}\t"
        f"{item['provenance']['canonical_url']}\n"
        for item in sorted(
            cases, key=lambda value: int(value["provenance"]["search_rank"])
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_reliability_package(
    all_by_rank: dict[int, dict[str, Any]],
    benchmark_by_rank: dict[int, dict[str, Any]],
    prior_manual: dict[int, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bucket_by_rank: dict[int, str] = {}
    for bucket, ranks in RELIABILITY_BUCKET_RANKS.items():
        for rank in ranks:
            if rank in bucket_by_rank:
                raise BenchmarkBuildError(f"Reliability rank {rank} appears in two buckets")
            bucket_by_rank[rank] = bucket
    if len(bucket_by_rank) != 100:
        raise BenchmarkBuildError(
            f"Reliability sample must contain 100 unique ranks, found {len(bucket_by_rank)}"
        )
    expected_bucket_counts = {"A": 35, "B": 20, "C": 15, "D": 15, "E": 15}
    observed_bucket_counts = Counter(bucket_by_rank.values())
    if observed_bucket_counts != Counter(expected_bucket_counts):
        raise BenchmarkBuildError(
            f"Reliability bucket counts changed: {dict(observed_bucket_counts)}"
        )

    missing = sorted(set(bucket_by_rank) - set(all_by_rank))
    if missing:
        raise BenchmarkBuildError(f"Reliability ranks absent from parser JSONL: {missing}")
    selected = [all_by_rank[rank] for rank in bucket_by_rank]
    if not all(usable_english(item) for item in selected):
        bad = [
            int(item["provenance"]["search_rank"])
            for item in selected
            if not usable_english(item)
        ]
        raise BenchmarkBuildError(f"Reliability sample has unusable English text: {bad}")
    digest = reliability_membership_digest(selected)
    if digest != EXPECTED_RELIABILITY_MEMBERSHIP_SHA256:
        raise BenchmarkBuildError(
            f"Reliability sample hash changed: {digest} != "
            f"{EXPECTED_RELIABILITY_MEMBERSHIP_SHA256}"
        )

    ordered = sorted(
        selected,
        key=lambda item: hashlib.sha256(
            (
                f"{RELIABILITY_SEED}|review-order|"
                f"{item['provenance']['canonical_url']}"
            ).encode("utf-8")
        ).hexdigest(),
    )

    project_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for order, source in enumerate(ordered, start=1):
        rank = int(source["provenance"]["search_rank"])
        neutral_id = f"HRV1-{order:03d}"
        bucket = bucket_by_rank[rank]
        sentences = split_sentences_v1(fact_summary(source))
        # A second invocation is a cheap deterministic stability assertion.
        if sentences != split_sentences_v1(fact_summary(source)):
            raise BenchmarkBuildError(f"Sentence segmentation is unstable at rank {rank}")
        numbered = numbered_sentences(sentences)
        sentence_hash = hashlib.sha256(numbered.encode("utf-8")).hexdigest()
        reason = derived_selection_reason(
            bucket, source, benchmark_by_rank, prior_manual
        )
        benchmark = benchmark_by_rank.get(rank)

        project_rows.append(
            {
                "reliability_case_id": neutral_id,
                "reviewer_order": order,
                "search_rank": rank,
                "case_title": source.get("case_identity", {}).get("title_raw"),
                "unodc_case_number": source.get("provenance", {}).get(
                    "unodc_case_number"
                ),
                "canonical_url": source.get("provenance", {}).get("canonical_url"),
                "jurisdiction_raw": source.get("case_identity", {}).get("country_raw"),
                "sampling_bucket": bucket,
                "sampling_bucket_label": RELIABILITY_BUCKET_LABELS[bucket],
                "selection_reason": reason,
                "primary_amp_cohort_member": int(rank in benchmark_by_rank),
                "english_fact_summary_word_count": narrative_word_count(
                    fact_summary(source)
                ),
                "sentence_count": len(sentences),
                "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
                "numbered_text_sha256": sentence_hash,
                "english_fact_summary_raw": fact_summary(source),
                "fact_summary_numbered": numbered,
                "raw_file_reference": source.get("source_input", {}).get("actual_path")
                or source.get("provenance", {}).get(
                    "download_manifest_raw_filename"
                ),
                "raw_sha256": source.get("source_input", {}).get("computed_sha256")
                or source.get("provenance", {}).get("download_manifest_sha256"),
            }
        )

        reviewer_rows.append(
            {
                "reviewer_id": "",
                "reliability_case_id": neutral_id,
                "sentence_splitter_version": SENTENCE_SPLITTER_VERSION,
                "sentence_count": len(sentences),
                "numbered_text_sha256": sentence_hash,
                "fact_summary_numbered": numbered,
                "act_labels": "",
                "act_answerability": "",
                "act_evidence_sentence_ids": "",
                "act_notes": "",
                "means_labels": "",
                "means_answerability": "",
                "means_evidence_sentence_ids": "",
                "means_notes": "",
                "purpose_labels": "",
                "purpose_answerability": "",
                "purpose_evidence_sentence_ids": "",
                "purpose_notes": "",
                "form_label": "",
                "form_answerability": "",
                "form_evidence_sentence_ids": "",
                "form_notes": "",
                "multiplicity_label": "",
                "multiplicity_answerability": "",
                "multiplicity_evidence_sentence_ids": "",
                "multiplicity_notes": "",
                "child_label": "",
                "child_answerability": "",
                "child_evidence_sentence_ids": "",
                "child_notes": "",
                "overall_narrative_sufficiency": "",
                "annotation_notes": "",
            }
        )

        reference_rows.append(
            {
                "reliability_case_id": neutral_id,
                "search_rank": rank,
                "case_title": source.get("case_identity", {}).get("title_raw"),
                "canonical_url": source.get("provenance", {}).get("canonical_url"),
                "jurisdiction_raw": source.get("case_identity", {}).get("country_raw"),
                "sampling_bucket": bucket,
                "sampling_bucket_label": RELIABILITY_BUCKET_LABELS[bucket],
                "selection_reason": reason,
                "primary_amp_cohort_member": int(rank in benchmark_by_rank),
                "legacy_acts_raw_json": json.dumps(
                    legacy_values(source, "acts"), ensure_ascii=False
                ),
                "legacy_means_raw_json": json.dumps(
                    legacy_values(source, "means"), ensure_ascii=False
                ),
                "legacy_purposes_raw_json": json.dumps(
                    legacy_values(source, "exploitative_purposes"), ensure_ascii=False
                ),
                "legacy_form_raw_json": json.dumps(
                    legacy_values(source, "form_of_trafficking"), ensure_ascii=False
                ),
                "legacy_ocg_present": int(
                    "Organized Criminal Group"
                    in legacy_values(source, "form_of_trafficking")
                ),
                "sidebar_acts_raw_json": json.dumps(
                    sidebar_values(source, "acts"), ensure_ascii=False
                ),
                "sidebar_means_raw_json": json.dumps(
                    sidebar_values(source, "means"), ensure_ascii=False
                ),
                "sidebar_purposes_raw_json": json.dumps(
                    sidebar_values(source, "exploitative_purposes"), ensure_ascii=False
                ),
                "sidebar_offences_raw_json": json.dumps(
                    sidebar_values(source, "offences"), ensure_ascii=False
                ),
                "multiplicity_provisional": (
                    benchmark["victim_multiplicity"]["multiplicity_provisional"]
                    if benchmark
                    else "NOT_APPLICABLE_OUTSIDE_PRIMARY_COHORT"
                ),
                "multiplicity_rule": (
                    benchmark["victim_multiplicity"]["provisional_rule"]
                    if benchmark
                    else ""
                ),
                "multiplicity_ambiguity_flags_json": json.dumps(
                    benchmark["victim_multiplicity"]["ambiguity_flags"]
                    if benchmark
                    else [],
                    ensure_ascii=False,
                ),
                "child_strict_label": (
                    benchmark["child_minor_exploratory"]["child_strict_label"]
                    if benchmark
                    else "NOT_APPLICABLE_OUTSIDE_PRIMARY_COHORT"
                ),
                "reference_use_warning": (
                    "RESEARCHER_ONLY; do not show reviewers until both independent "
                    "annotation files are complete. Structured values are comparison "
                    "references, not automatically narrative-grounded gold."
                ),
                "numbered_text_sha256": sentence_hash,
                "raw_file_reference": source.get("source_input", {}).get("actual_path")
                or source.get("provenance", {}).get(
                    "download_manifest_raw_filename"
                ),
                "raw_sha256": source.get("source_input", {}).get("computed_sha256")
                or source.get("provenance", {}).get("download_manifest_sha256"),
            }
        )

    sample_legacy_vocab = {
        "ACT": {
            value
            for source in selected
            for value in legacy_values(source, "acts")
        },
        "MEANS": {
            value
            for source in selected
            for value in legacy_values(source, "means")
        },
        "PURPOSE": {
            value
            for source in selected
            for value in legacy_values(source, "exploitative_purposes")
        },
    }
    diagnostics = {
        "sample_n": len(project_rows),
        "membership_sha256": digest,
        "bucket_counts": dict(Counter(row["sampling_bucket"] for row in project_rows)),
        "jurisdiction_count": len(
            {row["jurisdiction_raw"] for row in project_rows if row["jurisdiction_raw"]}
        ),
        "primary_amp_complete_n": sum(
            int(row["primary_amp_cohort_member"]) for row in project_rows
        ),
        "sample_legacy_vocab": {
            family: sorted(values) for family, values in sample_legacy_vocab.items()
        },
        "challenge_counts": {
            "multiplicity_flagged_primary_cases": sum(
                bool(benchmark_by_rank[rank]["victim_multiplicity"]["ambiguity_flags"])
                for rank in bucket_by_rank
                if rank in benchmark_by_rank
            ),
            "child_uncertainty_or_attempt_anchors": len(
                set(bucket_by_rank)
                & (
                    CHILD_UNCERTAIN_OVERRIDES
                    | CHILD_EXPLICIT_FALSE_OVERRIDES
                    | CHILD_NARRATIVE_EXCLUSIONS_FROM_ADULT
                )
            ),
            "parser_warning_cases": sum(
                bool(all_by_rank[rank].get("parser_provenance", {}).get("warning_count"))
                for rank in bucket_by_rank
            ),
            "outside_primary_amp_cohort": 100
            - sum(int(row["primary_amp_cohort_member"]) for row in project_rows),
        },
    }
    return project_rows, reviewer_rows, reference_rows, diagnostics


FORM_COMBINATION_ORDER = [
    "Internal only",
    "Transnational only",
    "Internal + Transnational",
    "Organized Criminal Group only",
    "Internal + Organized Criminal Group",
    "Transnational + Organized Criminal Group",
    "Internal + Transnational + Organized Criminal Group",
    "Other observed combination",
    "Missing Form",
]


def form_combination_name(values: Sequence[str]) -> str:
    raw = frozenset(values)
    known = {
        frozenset({"Internal"}): "Internal only",
        frozenset({"Transnational"}): "Transnational only",
        frozenset({"Internal", "Transnational"}): "Internal + Transnational",
        frozenset({"Organized Criminal Group"}): "Organized Criminal Group only",
        frozenset({"Internal", "Organized Criminal Group"}): (
            "Internal + Organized Criminal Group"
        ),
        frozenset({"Transnational", "Organized Criminal Group"}): (
            "Transnational + Organized Criminal Group"
        ),
        frozenset(
            {"Internal", "Transnational", "Organized Criminal Group"}
        ): "Internal + Transnational + Organized Criminal Group",
        frozenset(): "Missing Form",
    }
    return known.get(raw, "Other observed combination")


def collect_statistics(
    cohort_sources: Sequence[dict[str, Any]],
    benchmark_records: Sequence[dict[str, Any]],
    queue_rows: Sequence[dict[str, Any]],
    reliability_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    amp_counts = {
        "ACT": Counter(
            value for item in cohort_sources for value in legacy_values(item, "acts")
        ),
        "MEANS": Counter(
            value for item in cohort_sources for value in legacy_values(item, "means")
        ),
        "PURPOSE": Counter(
            value
            for item in cohort_sources
            for value in legacy_values(item, "exploitative_purposes")
        ),
    }
    form_combinations = Counter(
        form_combination_name(legacy_values(item, "form_of_trafficking"))
        for item in cohort_sources
    )
    form_internal = sum(
        item["geographic_form"]["geographic_form_internal"]
        for item in benchmark_records
    )
    form_transnational = sum(
        item["geographic_form"]["geographic_form_transnational"]
        for item in benchmark_records
    )
    form_both = sum(
        item["geographic_form"]["geographic_form_internal"]
        and item["geographic_form"]["geographic_form_transnational"]
        for item in benchmark_records
    )
    form_eligible = sum(
        item["geographic_form"]["geographic_form_eligible"]
        for item in benchmark_records
    )
    mult_counts = Counter(
        item["victim_multiplicity"]["multiplicity_provisional"]
        for item in benchmark_records
    )
    child_counts = Counter(
        item["child_minor_exploratory"]["child_strict_label"]
        for item in benchmark_records
    )
    jurisdictions = Counter(
        item["identity"]["jurisdiction_country_raw"] for item in benchmark_records
    )
    queue_label_counts = Counter(
        row["provisional_multiplicity_label"] for row in queue_rows
    )
    return {
        "amp_counts": {family: dict(counts) for family, counts in amp_counts.items()},
        "form_combination_counts": {
            name: form_combinations.get(name, 0) for name in FORM_COMBINATION_ORDER
        },
        "form": {
            "eligible": form_eligible,
            "internal": form_internal,
            "transnational": form_transnational,
            "both": form_both,
            "ocg_any": sum(
                item["geographic_form"]["organized_criminal_group_present"]
                for item in benchmark_records
            ),
            "ocg_only_excluded": form_combinations[
                "Organized Criminal Group only"
            ],
        },
        "multiplicity": {
            **dict(mult_counts),
            "eligible": mult_counts["SINGLE"] + mult_counts["MULTIPLE"],
            "review_queue": len(queue_rows),
            "queue_label_counts": dict(queue_label_counts),
            "eligible_requiring_confirmation": sum(
                row["currently_performance_eligible"] for row in queue_rows
            ),
        },
        "child": {
            **dict(child_counts),
            "eligible": child_counts["TRUE"] + child_counts["FALSE"],
            "true_to_false_ratio": (
                child_counts["TRUE"] / child_counts["FALSE"]
                if child_counts["FALSE"]
                else None
            ),
        },
        "jurisdiction": {
            "distinct": len(jurisdictions),
            "top_10": jurisdictions.most_common(10),
            "at_least_10": sum(value >= 10 for value in jurisdictions.values()),
            "at_least_20": sum(value >= 20 for value in jurisdictions.values()),
            "at_least_30": sum(value >= 30 for value in jurisdictions.values()),
        },
        "reliability": reliability_diagnostics,
    }


def markdown_frequency_table(counts: dict[str, int], denominator: int = 1263) -> str:
    lines = ["| Raw Legacy label | Cases | Percent |", "|---|---:|---:|"]
    for label, count in counts.items():
        lines.append(f"| `{label}` | {count:,} | {100 * count / denominator:.2f}% |")
    return "\n".join(lines)


def build_report(stats: dict[str, Any]) -> str:
    amp = stats["amp_counts"]
    form = stats["form"]
    mult = stats["multiplicity"]
    child = stats["child"]
    jurisdiction = stats["jurisdiction"]
    reliability = stats["reliability"]
    combo_lines = ["| Raw Legacy Form combination | Cases |", "|---|---:|"]
    for name in FORM_COMBINATION_ORDER:
        combo_lines.append(
            f"| {name} | {stats['form_combination_counts'][name]:,} |"
        )
    jurisdiction_lines = ["| Rank | Jurisdiction/category raw value | Cases |", "|---:|---|---:|"]
    for index, (name, count) in enumerate(jurisdiction["top_10"], start=1):
        jurisdiction_lines.append(f"| {index} | {name} | {count:,} |")
    bucket_lines = ["| Sampling bucket | Cases |", "|---|---:|"]
    for code in "ABCDE":
        bucket_lines.append(
            f"| {code}. {RELIABILITY_BUCKET_LABELS[code]} | "
            f"{reliability['bucket_counts'][code]} |"
        )

    return f"""# SHERLOC benchmark v1 construction report

Build freeze date: `{BUILD_FREEZE_DATE}`  
SHERLOC snapshot: `{SNAPSHOT_DATE}`  
Builder: `src/sherloc/05_build_benchmark.py` version `{BUILDER_VERSION}`

## Decision summary

The frozen primary AMP cohort remains exactly **1,263 cases** under cohort ID
`{PRIMARY_COHORT_ID}`. Legacy Keywords alone define primary Act, Means, and
Purpose targets. Sidebar AMP is retained only as source-separated secondary
metadata. No modeling or evaluation split was created.

| Target | Task type | Reference source | Eligible N | Status |
|---|---|---|---:|---|
| Act | 5-label multi-label | Legacy Keywords Acts | 1,263 | PRIMARY |
| Means | 6-label multi-label | Legacy Keywords Means | 1,263 | PRIMARY |
| Purpose | 6-label multi-label | Legacy Keywords Purpose of Exploitation | 1,263 | PRIMARY |
| Geographic Form | 2-label multi-label | Legacy Form: `Internal`/`Transnational` only | {form['eligible']:,} | AUXILIARY |
| Victim multiplicity | SINGLE/MULTIPLE; UNKNOWN abstains | Conservative feasibility policy v1 | {mult['eligible']:,} provisional | AUXILIARY |
| Child/minor involvement | TRUE/FALSE; UNKNOWN abstains | Strict structured-reference plus narrative-support screen | {child['eligible']:,} | EXPLORATORY |

Sector, exact victim count, and Organized Criminal Group as a geographic Form
label are excluded from benchmark v1.

## 1. Frozen universes and provenance

- Complete parser-v2 corpus: **1,590** cases.
- Usable English Fact Summary universe: **1,565** cases.
- Primary complete-Legacy-AMP cohort: **1,263** cases.
- Primary membership SHA-256: `{EXPECTED_PRIMARY_MEMBERSHIP_SHA256}`.
- Parser-v2 JSONL SHA-256: `{EXPECTED_SOURCE_SHA256}`.
- Prior feasibility-review CSV SHA-256: `{EXPECTED_PRIOR_MANUAL_REVIEW_SHA256}`.
- Every benchmark record retains canonical URL, raw HTML filename/checksum,
  parser version/status, download timestamp, and API identity.

Parser v2 does not expose a validated decision/verdict date. The benchmark
therefore leaves `decision_or_verdict_year` null and separately retains the
case-page URL year where one exists; it does not reinterpret that URL segment.

## 2. Primary Legacy AMP ontology and frequencies

The machine ontology is `sherloc-legacy-amp-v1`, with stable zero-based indices
and exactly **5 Act, 6 Means, and 6 Purpose labels**. Every raw Legacy AMP value
in all 1,263 cases maps exactly once. The ontology preserves raw SHERLOC strings
without merging or relabeling them.

### Act

{markdown_frequency_table(amp['ACT'])}

### Means

{markdown_frequency_table(amp['MEANS'])}

### Purpose

{markdown_frequency_table(amp['PURPOSE'])}

## 3. Geographic Form

{chr(10).join(combo_lines)}

- Geographic eligible N: **{form['eligible']:,}** ({100 * form['eligible'] / 1263:.2f}% of primary cohort).
- INTERNAL: **{form['internal']:,}**.
- TRANSNATIONAL: **{form['transnational']:,}**.
- Both: **{form['both']:,}**.
- Organized Criminal Group appears as raw metadata in **{form['ocg_any']:,}** cases.
- OCG-only cases excluded from the geographic target: **{form['ocg_only_excluded']:,}**.

Internal and Transnational are independent binary labels. Co-occurrence is
preserved. OCG never becomes a geographic label, and OCG-only or missing-Form
cases are not geographic-Form eligible.

## 4. Provisional victim multiplicity

| Provisional label | Cases |
|---|---:|
| SINGLE | {mult['SINGLE']:,} |
| MULTIPLE | {mult['MULTIPLE']:,} |
| UNKNOWN | {mult['UNKNOWN']:,} |
| Eligible SINGLE or MULTIPLE | **{mult['eligible']:,}** |

The compact minimum review queue contains **{mult['review_queue']:,}** cases:
{mult['queue_label_counts'].get('SINGLE', 0)} SINGLE,
{mult['queue_label_counts'].get('MULTIPLE', 0)} flagged MULTIPLE, and
{mult['queue_label_counts'].get('UNKNOWN', 0)} UNKNOWN. Of these,
**{mult['eligible_requiring_confirmation']:,}** are currently performance-eligible
but still require human confirmation. All 183 provisional SINGLE cases are in
the queue. Other flags cover duplicate person records, mixed Victim/Plaintiff
roles, `Migrants` headings, absent person sections, and multi-record semantic
exceptions. UNKNOWN remains outside the main performance cohort.

## 5. Exploratory child/minor target

| Strict label | Cases |
|---|---:|
| TRUE | {child['TRUE']:,} |
| FALSE | {child['FALSE']:,} |
| UNKNOWN | {child['UNKNOWN']:,} |
| Exploratory eligible | **{child['eligible']:,}** |

The eligible class ratio is **{child['true_to_false_ratio']:.2f}:1** TRUE:FALSE
({100 * child['TRUE'] / child['eligible']:.2f}% TRUE). This is an automated
intersection of the conservative structured policy and a strict Fact Summary
support screen, not a fully human-adjudicated focal-victim gold set. It remains
exploratory and does not alter the 1,263-case AMP cohort.

Known role-linkage limitation: search rank **448**
(`ECLI:NL:HR:2011:BP9394`) currently screens `TRUE` because both structured
person metadata and the narrative mention an infant, but the infant is the
manslaughter victim while the focal trafficking victim appears to be the adult
appellant. The frozen automated count is retained here as a candidate ceiling;
this case, and any similar role-ambiguous case, requires human adjudication
before label-level use.

## 6. Jurisdiction composition

The primary cohort contains **{jurisdiction['distinct']}** nonempty exact
jurisdiction/category values. Counts meeting later support thresholds are:
**{jurisdiction['at_least_10']}** with at least 10 cases,
**{jurisdiction['at_least_20']}** with at least 20, and
**{jurisdiction['at_least_30']}** with at least 30.

{chr(10).join(jurisdiction_lines)}

No jurisdiction-held-out or other evaluation split was constructed.

## 7. Blinded 100-case human reliability sample

The deterministic sample is drawn from the 1,565-case usable-English universe,
not only the primary AMP cohort. Membership SHA-256 is
`{reliability['membership_sha256']}`.

{chr(10).join(bucket_lines)}

- Unique jurisdictions/categories: **{reliability['jurisdiction_count']}**.
- Complete primary AMP members: **{reliability['primary_amp_complete_n']}**;
  outside-primary usable-English cases: **{reliability['challenge_counts']['outside_primary_amp_cohort']}**.
- All 5/6/6 Legacy AMP labels occur somewhere in the selected sample.
- Multiplicity-flagged primary cases: **{reliability['challenge_counts']['multiplicity_flagged_primary_cases']}**.
- Child uncertainty/attempt/conflict anchors: **{reliability['challenge_counts']['child_uncertainty_or_attempt_anchors']}**.
- Parser structural-warning cases with usable English: **{reliability['challenge_counts']['parser_warning_cases']}**.

Reviewer order is the stable SHA-256 order of
`sherloc-reliability-v1-2026-08-11|review-order|canonical_url`, exposed only as
neutral IDs `HRV1-001` through `HRV1-100`. The reviewer template contains no
rank, title, URL, jurisdiction, sampling bucket, SHERLOC structured target,
provisional multiplicity/child label, or hidden audit judgment. Reviewers must
not receive the project-management sample or researcher-only reference key
until both independent annotations are complete.

Sentence evidence uses `{SENTENCE_SPLITTER_VERSION}`. Original Fact Summaries
remain unchanged in benchmark/project files; reviewer text adds deterministic
`[S1]`, `[S2]`, ... display identifiers only.

## 8. Remaining blockers before modeling

1. Human-confirm the 220 currently eligible multiplicity queue cases before
   treating the provisional cohort as final gold; preserve UNKNOWN when unclear.
2. Complete both blinded annotations, calculate agreement, and adjudicate before
   creating human-grounded reliability labels.
3. Freeze the evaluation protocol and jurisdiction-support rules only after the
   reviewed target distributions are available.
4. Human-adjudicate focal-victim linkage for the child set, beginning with the
   known rank-448 exception, and keep the 355 cases exploratory unless that
   review supports a stronger claim.

The generated files deliberately contain no model outputs, folds, random split,
held-out-jurisdiction split, or cross-validation assignment.
"""


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise BenchmarkBuildError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validate_frozen_inputs(
    input_path: Path,
    records: Sequence[dict[str, Any]],
    ontology_maps: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    observed_source_sha = sha256_file(input_path)
    if observed_source_sha != EXPECTED_SOURCE_SHA256:
        raise BenchmarkBuildError(
            f"Parser JSONL hash changed: {observed_source_sha} != {EXPECTED_SOURCE_SHA256}"
        )
    if len(records) != EXPECTED_ALL_CASES:
        raise BenchmarkBuildError(
            f"Expected {EXPECTED_ALL_CASES} parser rows, found {len(records)}"
        )
    ranks = [int(item["provenance"]["search_rank"]) for item in records]
    urls = [item["provenance"]["canonical_url"] for item in records]
    if len(set(ranks)) != len(ranks) or len(set(urls)) != len(urls):
        raise BenchmarkBuildError("Parser input ranks or canonical URLs are not unique")
    english = [item for item in records if usable_english(item)]
    if len(english) != EXPECTED_ENGLISH_UNIVERSE:
        raise BenchmarkBuildError(
            f"English universe changed: {len(english)} != {EXPECTED_ENGLISH_UNIVERSE}"
        )
    cohort = sorted(
        (item for item in records if in_primary_cohort(item)),
        key=lambda item: int(item["provenance"]["search_rank"]),
    )
    if len(cohort) != EXPECTED_PRIMARY_COHORT:
        raise BenchmarkBuildError(
            f"Primary cohort changed: {len(cohort)} != {EXPECTED_PRIMARY_COHORT}"
        )
    digest = primary_membership_digest(cohort)
    if digest != EXPECTED_PRIMARY_MEMBERSHIP_SHA256:
        raise BenchmarkBuildError(
            f"Primary membership hash changed: {digest} != "
            f"{EXPECTED_PRIMARY_MEMBERSHIP_SHA256}"
        )

    observed = {
        "ACT": {value for item in cohort for value in legacy_values(item, "acts")},
        "MEANS": {value for item in cohort for value in legacy_values(item, "means")},
        "PURPOSE": {
            value
            for item in cohort
            for value in legacy_values(item, "exploitative_purposes")
        },
    }
    for family in ("ACT", "MEANS", "PURPOSE"):
        configured = set(ontology_maps[family])
        if observed[family] != configured:
            raise BenchmarkBuildError(
                f"Ontology mismatch for {family}; unmatched={sorted(observed[family]-configured)}, "
                f"unobserved={sorted(configured-observed[family])}"
            )
    if not all(
        usable_english(item)
        and legacy_values(item, "acts")
        and legacy_values(item, "means")
        and legacy_values(item, "exploitative_purposes")
        for item in cohort
    ):
        raise BenchmarkBuildError("A primary case lacks English text or complete Legacy AMP")
    return cohort


def validate_computed_package(
    cohort_sources: Sequence[dict[str, Any]],
    benchmark_records: Sequence[dict[str, Any]],
    queue_rows: Sequence[dict[str, Any]],
    project_rows: Sequence[dict[str, Any]],
    reviewer_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    ontology_maps: dict[str, dict[str, str]],
) -> None:
    if len(benchmark_records) != 1263:
        raise BenchmarkBuildError("Benchmark record count is not 1,263")
    expected_form_combinations = {
        "Internal only": 362,
        "Transnational only": 614,
        "Internal + Transnational": 34,
        "Organized Criminal Group only": 30,
        "Internal + Organized Criminal Group": 23,
        "Transnational + Organized Criminal Group": 109,
        "Internal + Transnational + Organized Criminal Group": 14,
        "Other observed combination": 0,
        "Missing Form": 77,
    }
    if stats["form_combination_counts"] != expected_form_combinations:
        raise BenchmarkBuildError(
            f"Legacy Form combinations changed: {stats['form_combination_counts']}"
        )
    if stats["form"] != {
        "eligible": 1156,
        "internal": 433,
        "transnational": 771,
        "both": 48,
        "ocg_any": 176,
        "ocg_only_excluded": 30,
    }:
        raise BenchmarkBuildError(f"Geographic Form totals changed: {stats['form']}")
    for source, benchmark in zip(cohort_sources, benchmark_records):
        amp = benchmark["amp_targets"]
        if amp["reference_source"] != "Legacy Keywords":
            raise BenchmarkBuildError("Non-Legacy source entered primary AMP")
        for family, raw_key, id_key in (
            ("ACT", "acts_raw", "act_ontology_ids"),
            ("MEANS", "means_raw", "means_ontology_ids"),
            ("PURPOSE", "purposes_raw", "purpose_ontology_ids"),
        ):
            expected_ids = [ontology_maps[family][value] for value in amp[raw_key]]
            if amp[id_key] != expected_ids or not amp[raw_key]:
                raise BenchmarkBuildError(
                    f"AMP mapping failure at rank {benchmark['identity']['search_rank']}"
                )
        form_raw = set(benchmark["geographic_form"]["legacy_form_values_raw"])
        eligible = benchmark["geographic_form"]["geographic_form_eligible"]
        if eligible != int(bool(form_raw & {"Internal", "Transnational"})):
            raise BenchmarkBuildError("Geographic Form eligibility is inconsistent")
        if form_raw == {"Organized Criminal Group"} and eligible:
            raise BenchmarkBuildError("OCG-only case entered geographic Form cohort")
        mult = benchmark["victim_multiplicity"]
        if mult["multiplicity_provisional"] == "UNKNOWN" and mult[
            "multiplicity_eligible"
        ]:
            raise BenchmarkBuildError("Multiplicity UNKNOWN marked eligible")
        child = benchmark["child_minor_exploratory"]
        if child["child_exploratory_eligible"] != int(
            child["child_strict_label"] in {"TRUE", "FALSE"}
        ):
            raise BenchmarkBuildError("Child exploratory eligibility is inconsistent")
        provenance = benchmark["source_provenance"]
        if not provenance["raw_file_reference"] or not re.fullmatch(
            r"[0-9a-f]{64}", provenance["raw_sha256"] or ""
        ):
            raise BenchmarkBuildError("Benchmark provenance is not traceable to raw HTML")

    mult = stats["multiplicity"]
    expected_mult = {
        "SINGLE": 183,
        "MULTIPLE": 782,
        "UNKNOWN": 298,
        "eligible": 965,
        "review_queue": 250,
        "queue_label_counts": {"UNKNOWN": 30, "MULTIPLE": 37, "SINGLE": 183},
        "eligible_requiring_confirmation": 220,
    }
    if mult != expected_mult:
        raise BenchmarkBuildError(f"Multiplicity policy changed: {mult}")
    if len(queue_rows) != 250 or len(
        {int(row["search_rank"]) for row in queue_rows}
    ) != 250:
        raise BenchmarkBuildError("Multiplicity review queue is not 250 unique cases")
    if any(not row["human_confirmation_required"] for row in queue_rows):
        raise BenchmarkBuildError("Review queue contains a row not flagged for confirmation")

    child = stats["child"]
    if (
        child.get("TRUE"),
        child.get("FALSE"),
        child.get("UNKNOWN"),
        child.get("eligible"),
    ) != (337, 18, 908, 355):
        raise BenchmarkBuildError(f"Strict child policy changed: {child}")

    if not (
        len(project_rows) == len(reviewer_rows) == len(reference_rows) == 100
    ):
        raise BenchmarkBuildError("Reliability package is not exactly 100 rows per file")
    project_ids = {row["reliability_case_id"] for row in project_rows}
    reviewer_ids = {row["reliability_case_id"] for row in reviewer_rows}
    reference_ids = {row["reliability_case_id"] for row in reference_rows}
    if project_ids != reviewer_ids or project_ids != reference_ids or len(project_ids) != 100:
        raise BenchmarkBuildError("Reliability files do not map 1:1")
    if stats["reliability"]["membership_sha256"] != EXPECTED_RELIABILITY_MEMBERSHIP_SHA256:
        raise BenchmarkBuildError("Reliability sample membership hash changed")
    if stats["reliability"]["primary_amp_complete_n"] != 89:
        raise BenchmarkBuildError("Reliability sample primary/non-primary composition changed")
    if stats["reliability"]["jurisdiction_count"] != 51:
        raise BenchmarkBuildError("Reliability sample jurisdiction diversity changed")
    for family in ("ACT", "MEANS", "PURPOSE"):
        if set(stats["reliability"]["sample_legacy_vocab"][family]) != set(
            ontology_maps[family]
        ):
            raise BenchmarkBuildError(
                f"Reliability sample no longer covers every {family} label"
            )

    reviewer_headers = set(reviewer_rows[0])
    forbidden_headers = {
        "search_rank",
        "case_title",
        "canonical_url",
        "jurisdiction_raw",
        "sampling_bucket",
        "sampling_bucket_label",
        "selection_reason",
        "legacy_acts_raw_json",
        "legacy_means_raw_json",
        "legacy_purposes_raw_json",
        "legacy_form_raw_json",
        "sidebar_acts_raw_json",
        "sidebar_means_raw_json",
        "sidebar_purposes_raw_json",
        "multiplicity_provisional",
        "child_strict_label",
    }
    if reviewer_headers & forbidden_headers:
        raise BenchmarkBuildError(
            f"Reviewer template leaks hidden fields: {sorted(reviewer_headers & forbidden_headers)}"
        )
    annotation_columns = [
        name
        for name in reviewer_rows[0]
        if name not in {
            "reliability_case_id",
            "sentence_splitter_version",
            "sentence_count",
            "numbered_text_sha256",
            "fact_summary_numbered",
        }
    ]
    if any(row[name] != "" for row in reviewer_rows for name in annotation_columns):
        raise BenchmarkBuildError("Reviewer annotation columns are not blank")
    project_by_id = {row["reliability_case_id"]: row for row in project_rows}
    for row in reviewer_rows:
        project = project_by_id[row["reliability_case_id"]]
        sentences = split_sentences_v1(project["english_fact_summary_raw"])
        numbered = numbered_sentences(sentences)
        if row["fact_summary_numbered"] != numbered:
            raise BenchmarkBuildError("Reviewer sentence display is not reproducible")
        if row["numbered_text_sha256"] != hashlib.sha256(
            numbered.encode("utf-8")
        ).hexdigest():
            raise BenchmarkBuildError("Reviewer numbered-text checksum mismatch")


def validate_written_outputs(
    benchmark_jsonl: Path,
    benchmark_csv: Path,
    queue_path: Path,
    sample_path: Path,
    reviewer_path: Path,
    reference_path: Path,
) -> dict[str, str]:
    if len(load_jsonl(benchmark_jsonl)) != 1263:
        raise BenchmarkBuildError("Written benchmark JSONL does not have 1,263 rows")
    expected_csv_rows = {
        benchmark_csv: 1263,
        queue_path: 250,
        sample_path: 100,
        reviewer_path: 100,
        reference_path: 100,
    }
    for path, expected in expected_csv_rows.items():
        with path.open(encoding="utf-8", newline="") as handle:
            observed = sum(1 for _ in csv.DictReader(handle))
        if observed != expected:
            raise BenchmarkBuildError(
                f"Written {path} has {observed} rows, expected {expected}"
            )
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in (
            benchmark_jsonl,
            benchmark_csv,
            queue_path,
            sample_path,
            reviewer_path,
            reference_path,
        )
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen SHERLOC benchmark-v1 and blinded annotation files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--benchmark-jsonl", type=Path, default=DEFAULT_BENCHMARK_JSONL)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument(
        "--multiplicity-review", type=Path, default=DEFAULT_MULTIPLICITY_QUEUE
    )
    parser.add_argument(
        "--reliability-sample", type=Path, default=DEFAULT_RELIABILITY_SAMPLE
    )
    parser.add_argument(
        "--reviewer-template", type=Path, default=DEFAULT_REVIEWER_TEMPLATE
    )
    parser.add_argument("--reference-key", type=Path, default=DEFAULT_REFERENCE_KEY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--prior-manual-review", type=Path, default=DEFAULT_PRIOR_MANUAL_REVIEW
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ontology, ontology_maps = validate_and_load_ontology(args.ontology)
    records = load_jsonl(args.input)
    cohort = validate_frozen_inputs(args.input, records, ontology_maps)
    all_by_rank = {
        int(item["provenance"]["search_rank"]): item for item in records
    }

    child_labels, child_support = strict_child_labels(cohort)
    benchmark_records = [
        build_benchmark_record(item, ontology_maps, child_labels, child_support)
        for item in cohort
    ]
    benchmark_by_rank = {
        int(item["identity"]["search_rank"]): item for item in benchmark_records
    }
    queue_rows = [
        multiplicity_queue_row(all_by_rank[rank], benchmark_by_rank[rank])
        for rank in sorted(benchmark_by_rank)
        if benchmark_by_rank[rank]["victim_multiplicity"]["ambiguity_flags"]
    ]
    prior_manual = load_prior_manual_rows(args.prior_manual_review)
    project_rows, reviewer_rows, reference_rows, reliability_diagnostics = (
        build_reliability_package(all_by_rank, benchmark_by_rank, prior_manual)
    )
    stats = collect_statistics(
        cohort, benchmark_records, queue_rows, reliability_diagnostics
    )
    validate_computed_package(
        cohort,
        benchmark_records,
        queue_rows,
        project_rows,
        reviewer_rows,
        reference_rows,
        stats,
        ontology_maps,
    )

    atomic_write_jsonl(args.benchmark_jsonl, benchmark_records)
    atomic_write_csv(
        args.benchmark_csv, [benchmark_csv_row(item) for item in benchmark_records]
    )
    atomic_write_csv(args.multiplicity_review, queue_rows)
    atomic_write_csv(args.reliability_sample, project_rows)
    atomic_write_csv(args.reviewer_template, reviewer_rows)
    atomic_write_csv(args.reference_key, reference_rows)
    atomic_write_text(args.report, build_report(stats))

    output_hashes = validate_written_outputs(
        args.benchmark_jsonl,
        args.benchmark_csv,
        args.multiplicity_review,
        args.reliability_sample,
        args.reviewer_template,
        args.reference_key,
    )
    output_hashes[str(args.report.relative_to(REPO_ROOT))] = sha256_file(args.report)
    output_hashes[str(args.ontology.relative_to(REPO_ROOT))] = sha256_file(
        args.ontology
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "builder_version": BUILDER_VERSION,
                "source_jsonl_sha256": EXPECTED_SOURCE_SHA256,
                "prior_manual_review_sha256": EXPECTED_PRIOR_MANUAL_REVIEW_SHA256,
                "primary_cohort_id": PRIMARY_COHORT_ID,
                "english_universe_n": EXPECTED_ENGLISH_UNIVERSE,
                "primary_amp_n": len(benchmark_records),
                "ontology_family_sizes": {
                    family: len(ontology["families"][family])
                    for family in ("ACT", "MEANS", "PURPOSE")
                },
                "geographic_form": stats["form"],
                "multiplicity": stats["multiplicity"],
                "child": stats["child"],
                "reliability": stats["reliability"],
                "output_sha256": output_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
