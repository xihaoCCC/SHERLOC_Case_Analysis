#!/usr/bin/env python3
"""Prepare frozen Evaluation-A splits, demo proposals, and token diagnostics.

This stage is deliberately preparation-only: it never trains a model, calls an
LLM API, or evaluates predictions.  Until the researcher approves exactly six
demonstrations, all split artifacts are explicitly provisional and tied to the
top proposed demo set.  Changing that set requires rebuilding every artifact.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import math
import os
import re
import tempfile
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import AutoTokenizer


VERSION = "1.1.0"
PREPARATION_DATE = "2026-08-11"
SEED = 20260811
PRIMARY_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_BENCHMARK_SHA256 = (
    "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
)
EXPECTED_N = 1263
EXPECTED_FORM_N = 1156
EXPECTED_HIGH_JURISDICTION_N = 18
EXPECTED_HIGH_JURISDICTION_CASES = 861

MODEL_ID = "answerdotai/ModernBERT-base"
TOKENIZER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TOKEN_THRESHOLDS = (512, 1024, 1536, 2048, 3072, 4096, 8192)
EXPECTED_PREPARATION_ENVIRONMENT = {
    "python": "3.10",
    "numpy": "2.2.6",
    "scikit-learn": "1.7.2",
    "iterative-stratification": "0.1.9",
    "transformers": "5.5.3",
    "tokenizers": "0.22.2",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_PARSER = REPO_ROOT / "data/interim/sherloc_cases_raw.jsonl"
DEFAULT_A1 = REPO_ROOT / "data/splits/a1_iid_split_v1.csv"
DEFAULT_A2 = REPO_ROOT / "data/splits/a2_jurisdiction_folds_v1.csv"
DEFAULT_DEMO_SETS = REPO_ROOT / "outputs/tables/demo_candidate_sets.csv"
DEFAULT_DEMO_REVIEW = REPO_ROOT / "data/annotations/demo_bank_review.csv"
DEFAULT_TOKEN_AUDIT = REPO_ROOT / "outputs/tables/modernbert_token_length_audit.csv"
DEFAULT_A1_REPORT = REPO_ROOT / "docs/a1_split_report.md"
DEFAULT_A2_REPORT = REPO_ROOT / "docs/a2_jurisdiction_split_report.md"
DEFAULT_PREPARATION_REPORT = REPO_ROOT / "docs/experiment_preparation_report.md"
M1_CONFIG = REPO_ROOT / "config/experiments/m1_tfidf_logreg_v1.yaml"
M2_CONFIG = REPO_ROOT / "config/experiments/m2_modernbert_v1.yaml"
LLM_CONFIG = REPO_ROOT / "config/experiments/llm_extraction_v1.yaml"
M3_PROMPT = REPO_ROOT / "prompts/m3_zero_shot_v1.md"
M4_PROMPT = REPO_ROOT / "prompts/m4_six_shot_v1.md"


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def membership_digest(rows: Iterable[tuple[Any, ...]]) -> str:
    payload = "".join("\t".join(map(str, row)) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise PreparationError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def ontology_ids(ontology: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    ordered: list[str] = []
    family_by_id: dict[str, str] = {}
    for family in ("ACT", "MEANS", "PURPOSE"):
        labels = ontology["families"][family]
        if [item["index"] for item in labels] != list(range(len(labels))):
            raise PreparationError(f"Non-contiguous ontology indices in {family}")
        for item in labels:
            ordered.append(item["id"])
            family_by_id[item["id"]] = family
    if [sum(value == family for value in family_by_id.values()) for family in ("ACT", "MEANS", "PURPOSE")] != [5, 6, 6]:
        raise PreparationError("Ontology is not the frozen 5/6/6 design")
    return ordered, family_by_id


def labels(record: dict[str, Any]) -> list[str]:
    target = record["amp_targets"]
    return (
        target["act_ontology_ids"]
        + target["means_ontology_ids"]
        + target["purpose_ontology_ids"]
    )


def label_matrix(records: Sequence[dict[str, Any]], label_ids: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[int(label in set(labels(record))) for label in label_ids] for record in records],
        dtype=np.int8,
    )


def parser_warning_map(path: Path) -> dict[int, list[str]]:
    """Return substantive warnings affecting demo Fact Summary or Legacy AMP.

    Parser-v2 INFO diagnostics preserve unexpected-but-recovered structure and
    do not imply lost input/target content.  They remain visible in parser
    diagnostics but do not automatically disqualify an otherwise intact demo.
    """
    result: dict[int, list[str]] = {}
    for row in load_jsonl(path):
        rank = int(row["provenance"]["search_rank"])
        fact_warnings = row.get("narrative", {}).get("fact_summary", {}).get("warnings", [])
        legacy_warnings = row.get("legacy_keywords", {}).get("warnings", [])
        result[rank] = [
            f"{source}:{item.get('code', 'UNKNOWN')}"
            for source, warnings in (
                ("FACT_SUMMARY", fact_warnings),
                ("LEGACY_AMP", legacy_warnings),
            )
            for item in warnings
            if item.get("severity", "WARNING").upper() != "INFO"
        ]
    return result


def validate_inputs(
    benchmark_path: Path, ontology_path: Path, records: Sequence[dict[str, Any]]
) -> None:
    if sha256_file(benchmark_path) != EXPECTED_BENCHMARK_SHA256:
        raise PreparationError("Frozen benchmark JSONL hash changed")
    if sha256_file(ontology_path) != EXPECTED_ONTOLOGY_SHA256:
        raise PreparationError("Frozen ontology hash changed")
    if len(records) != EXPECTED_N:
        raise PreparationError(f"Expected {EXPECTED_N} benchmark rows, got {len(records)}")
    if any(item["primary_cohort_id"] != PRIMARY_COHORT_ID for item in records):
        raise PreparationError("Primary cohort ID mismatch")
    ranks = [item["identity"]["search_rank"] for item in records]
    urls = [item["identity"]["canonical_url"] for item in records]
    if len(set(ranks)) != EXPECTED_N or len(set(urls)) != EXPECTED_N:
        raise PreparationError("Benchmark ranks or URLs are not unique")
    if sum(item["geographic_form"]["geographic_form_eligible"] for item in records) != EXPECTED_FORM_N:
        raise PreparationError("Frozen geographic-Form count changed")


def validate_preparation_environment() -> None:
    """Fail on library drift that could change splits or tokenization."""
    import importlib.metadata

    if not sys.version.startswith(EXPECTED_PREPARATION_ENVIRONMENT["python"] + "."):
        raise PreparationError(
            f"Expected Python {EXPECTED_PREPARATION_ENVIRONMENT['python']}.x, got {sys.version.split()[0]}"
        )
    for package, expected in EXPECTED_PREPARATION_ENVIRONMENT.items():
        if package == "python":
            continue
        observed = importlib.metadata.version(package)
        if observed != expected:
            raise PreparationError(
                f"Preparation dependency drift: {package} {observed} != {expected}"
            )


# Candidate ranks are computed from the frozen benchmark below.  The two role
# profiles encode transparent coverage/contrast goals found during the audit;
# they deliberately contain ontology predicates rather than case IDs.  Five
# further proposals come from deterministic weighted set cover, and the last
# two profiles provide concise coverage alternatives.  This keeps the
# shortlist reproducible without making an audited outcome the selection
# mechanism.  Set 01 remains only a provisional split anchor pending human
# verification of every reference label and Form value.
DEMO_CANDIDATE_RANK_SETS: tuple[tuple[int, ...], ...] = ()
PROVISIONAL_DEMO_RANKS: frozenset[int] = frozenset()
PROVISIONAL_DEMO_SET_ID = "demo-bank-proposal-set-01-v1"

DEMO_SET_SIZE = 6
DEMO_SET_COVER_BEAM_WIDTH = 15_000
DEMO_SET_COVER_COUNT = 5

# A role is matched against the frozen ontology and structural metadata.  The
# profiles are intentionally explicit: they represent rare-label examples and
# useful contrasts a human reviewer should see, not assertions that the Legacy
# labels are narratively correct.  `preference=max_labels_then_shortest` is used
# only for a rare-purpose bundle where a compact multi-label example is useful;
# all other roles choose the shortest matching narrative, then search rank.
DEMO_SELECTION_PROFILES: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = (
    (
        "CONCISE_REVIEW_PRIORITY",
        (
            {
                "name": "transnational_forced_labour_slavery_servitude_bundle",
                "required_labels": (
                    "PURPOSE_FORCED_LABOUR_OR_SERVICES",
                    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
                    "PURPOSE_SERVITUDE",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
                "preference": "MAX_LABELS_THEN_SHORTEST",
            },
            {
                "name": "internal_abduction_vulnerability_sex",
                "required_labels": (
                    "MEANS_ABDUCTION",
                    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "INTERNAL_ONLY",
                "min_words": 75,
                "max_words": 200,
                "min_act_count": 3,
                "min_means_count": 3,
            },
            {
                "name": "internal_abduction_vulnerability_other",
                "required_labels": (
                    "MEANS_ABDUCTION",
                    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
                    "PURPOSE_OTHER",
                ),
                "form": "INTERNAL_ONLY",
                "max_words": 150,
            },
            {
                "name": "transnational_fraud_deception_payment_sex",
                "required_labels": (
                    "MEANS_FRAUD",
                    "MEANS_DECEPTION",
                    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 150,
            },
            {
                "name": "internal_transfer_receipt_payment_sex",
                "required_labels": (
                    "ACT_TRANSFER",
                    "ACT_RECEIPT",
                    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "INTERNAL_ONLY",
                "max_words": 150,
            },
            {
                "name": "transnational_organ_removal",
                "required_labels": ("PURPOSE_REMOVAL_OF_ORGANS",),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
            },
        ),
    ),
    (
        "NARRATIVE_SUPPORT_PRIORITY",
        (
            {
                "name": "no_form_receipt_abduction_vulnerability_sex",
                "required_labels": (
                    "ACT_RECEIPT",
                    "MEANS_ABDUCTION",
                    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "UNAVAILABLE",
                "max_words": 250,
            },
            {
                "name": "explicit_servitude_no_form",
                "required_labels": ("PURPOSE_SERVITUDE",),
                "form": "UNAVAILABLE",
                "max_words": 800,
                "text_regex": r"\bservitude\b",
            },
            {
                "name": "simple_internal_recruitment_deception_sex",
                "required_labels": (
                    "ACT_RECRUITMENT",
                    "MEANS_DECEPTION",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "INTERNAL_ONLY",
                "min_words": 150,
                "max_words": 220,
                "exact_total_label_count": 3,
            },
            {
                "name": "explicit_document_fraud_transnational_sex",
                "required_labels": (
                    "MEANS_FRAUD",
                    "MEANS_DECEPTION",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 100,
                "text_regex": r"\bforg(?:ed|ery|ified)\b",
            },
            {
                "name": "payment_slavery_other_transnational",
                "required_labels": (
                    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
                    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
                    "PURPOSE_OTHER",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 250,
            },
            {
                "name": "transnational_organ_removal",
                "required_labels": ("PURPOSE_REMOVAL_OF_ORGANS",),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
            },
        ),
    ),
    (
        "CONCISE_ALTERNATIVE",
        (
            {
                "name": "transnational_forced_labour_slavery_servitude_bundle",
                "required_labels": (
                    "PURPOSE_FORCED_LABOUR_OR_SERVICES",
                    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
                    "PURPOSE_SERVITUDE",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
                "preference": "MAX_LABELS_THEN_SHORTEST",
            },
            {
                "name": "internal_abduction_fraud_deception_sex",
                "required_labels": (
                    "MEANS_ABDUCTION",
                    "MEANS_FRAUD",
                    "MEANS_DECEPTION",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "INTERNAL_ONLY",
                "max_words": 200,
                "min_act_count": 4,
                "exact_means_count": 3,
            },
            {
                "name": "internal_abduction_vulnerability_other",
                "required_labels": (
                    "MEANS_ABDUCTION",
                    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
                    "PURPOSE_OTHER",
                ),
                "form": "INTERNAL_ONLY",
                "max_words": 150,
            },
            {
                "name": "internal_transfer_receipt_payment_sex",
                "required_labels": (
                    "ACT_TRANSFER",
                    "ACT_RECEIPT",
                    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "INTERNAL_ONLY",
                "max_words": 150,
            },
            {
                "name": "transnational_organ_removal",
                "required_labels": ("PURPOSE_REMOVAL_OF_ORGANS",),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
            },
            {
                "name": "transnational_receipt_coercion_deception_sex",
                "required_labels": (
                    "ACT_RECEIPT",
                    "MEANS_THREAT_FORCE_OR_COERCION",
                    "MEANS_DECEPTION",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 150,
                "min_act_count": 4,
                "exclude_organized_criminal_group": True,
            },
        ),
    ),
    (
        "CONCISE_ALTERNATIVE",
        (
            {
                "name": "short_transnational_sex_labour_slavery",
                "required_labels": (
                    "MEANS_DECEPTION",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                    "PURPOSE_FORCED_LABOUR_OR_SERVICES",
                    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 100,
            },
            {
                "name": "transnational_forced_labour_slavery_servitude_bundle",
                "required_labels": (
                    "PURPOSE_FORCED_LABOUR_OR_SERVICES",
                    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
                    "PURPOSE_SERVITUDE",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
                "preference": "MAX_LABELS_THEN_SHORTEST",
            },
            {
                "name": "internal_abduction_vulnerability_sex",
                "required_labels": (
                    "MEANS_ABDUCTION",
                    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "INTERNAL_ONLY",
                "min_words": 75,
                "max_words": 200,
                "min_act_count": 3,
                "min_means_count": 3,
            },
            {
                "name": "concise_transnational_receipt_vulnerability_other",
                "required_labels": (
                    "ACT_RECEIPT",
                    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
                    "PURPOSE_OTHER",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 100,
                "exact_total_label_count": 4,
            },
            {
                "name": "transnational_fraud_deception_payment_sex",
                "required_labels": (
                    "MEANS_FRAUD",
                    "MEANS_DECEPTION",
                    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
                    "PURPOSE_SEXUAL_EXPLOITATION",
                ),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 150,
            },
            {
                "name": "transnational_organ_removal",
                "required_labels": ("PURPOSE_REMOVAL_OF_ORGANS",),
                "form": "TRANSNATIONAL_ONLY",
                "max_words": 200,
            },
        ),
    ),
)

DEMO_CANDIDATE_BASIS = (
    "CONCISE_REVIEW_PRIORITY",
    "NARRATIVE_SUPPORT_PRIORITY",
    "WEIGHTED_SET_COVER",
    "WEIGHTED_SET_COVER",
    "WEIGHTED_SET_COVER",
    "WEIGHTED_SET_COVER",
    "WEIGHTED_SET_COVER",
    "CONCISE_ALTERNATIVE",
    "CONCISE_ALTERNATIVE",
)

DEMO_CANDIDATE_NOTES = (
    "Review Legacy Slavery/Servitude, Transfer, Fraud, Internal Form, and the focal-victim status in rank 1517.",
    "Better narrative support; rank 326 is long, rank 465 is an interrupted scheme, rank 1487 may omit narratively evident Deception, and rank 1517 remains ambiguous.",
    "High weighted coverage score; review sparse support for Legacy labels and rank 1517 focal-victim status.",
    "High weighted coverage score; review sparse support for Legacy labels and Form values.",
    "High weighted coverage score; review sparse support for Legacy labels and Form values.",
    "High weighted coverage score; review sparse support for Legacy labels and rank 1517 focal-victim status.",
    "High weighted coverage score; review sparse support for Legacy labels and Form values.",
    "Concise alternative; review Legacy labels against the narrative and rank 1517 focal-victim status.",
    "Concise alternative; review Legacy labels against the narrative and rank 1517 focal-victim status.",
)


def demo_clarity_penalty(record: dict[str, Any]) -> float:
    """Return a deterministic, deliberately conservative ambiguity proxy.

    This is a ranking aid, not a substitute for expert review.  It penalizes
    text that signals allegations, incomplete merits facts, or evidentiary
    uncertainty, plus extreme length and unusually dense reference labels.
    """

    text = record["text_input"]["english_fact_summary_raw"]
    words = record["text_input"]["word_count"]
    uncertainty_patterns = (
        r"\b(?:alleged|allegedly|unclear|purportedly|reportedly)\b",
        r"\bno direct discussion\b",
        r"\brather than (?:the )?case on the merits\b",
        r"\bnot sufficient evidence\b",
        r"\bevidence (?:was|were|is) (?:deemed )?unreliable\b",
    )
    uncertainty = sum(
        bool(re.search(pattern, text, flags=re.IGNORECASE))
        for pattern in uncertainty_patterns
    )
    label_count = len(labels(record))
    length_penalty = max(0, 50 - words) / 50 + max(0, words - 300) / 300
    density = label_count / max(words / 100, 1.0)
    density_penalty = max(0.0, density - 8.0)
    return 1.5 * uncertainty + 0.5 * length_penalty + 0.25 * density_penalty


def demo_set_cover_quality(record: dict[str, Any]) -> float:
    """Bounded soft quality term used by the broad set-cover search.

    The profile selector above uses the stricter clarity penalty.  Set cover
    needs a softer term so rare-label coverage remains dominant and plausible
    alternatives are retained for human comparison.
    """

    text = record["text_input"]["english_fact_summary_raw"]
    words = record["text_input"]["word_count"]
    label_count = len(labels(record))
    patterns = (
        r"\b(?:alleged|allegedly|unclear|purportedly|reportedly)\b",
        r"\bno direct discussion\b",
        r"\brather than (?:the )?case on the merits\b",
        r"\bnot sufficient evidence\b",
        r"\bevidence (?:was|were|is) (?:deemed )?unreliable\b",
    )
    uncertainty = sum(
        bool(re.search(pattern, text, flags=re.IGNORECASE)) for pattern in patterns
    )
    density = label_count / max(words / 100, 1e-9)
    return (
        1.0
        - 0.35 * uncertainty
        - 0.25 * max(0, 50 - words) / 50
        - 0.25 * max(0, words - 300) / 300
        - 0.20 * max(0, label_count - 9)
        - 0.20 * max(0.0, density - 8.0)
    )


def demo_candidate_pool(
    records: Sequence[dict[str, Any]],
    warnings: dict[int, list[str]],
    high_jurisdictions: set[str],
) -> list[dict[str, Any]]:
    """Apply the non-negotiable demonstration eligibility rules."""

    candidates: list[dict[str, Any]] = []
    for record in records:
        identity = record["identity"]
        rank = identity["search_rank"]
        text = record["text_input"]["english_fact_summary_raw"]
        words = record["text_input"]["word_count"]
        target = record["amp_targets"]
        complete_amp = all(
            key in target
            for key in (
                "act_ontology_ids",
                "means_ontology_ids",
                "purpose_ontology_ids",
            )
        )
        if (
            identity["jurisdiction_country_raw"] in high_jurisdictions
            or warnings.get(rank)
            or not text.strip()
            or not complete_amp
            or not 40 <= words <= 800
        ):
            continue
        candidates.append(record)
    return sorted(candidates, key=lambda item: item["identity"]["search_rank"])


def demo_role_matches(record: dict[str, Any], role: dict[str, Any]) -> bool:
    covered = set(labels(record))
    if not set(role.get("required_labels", ())).issubset(covered):
        return False

    form = record["geographic_form"]
    form_rule = role.get("form")
    if form_rule == "INTERNAL_ONLY" and not (
        form["geographic_form_internal"]
        and not form["geographic_form_transnational"]
    ):
        return False
    if form_rule == "TRANSNATIONAL_ONLY" and not (
        form["geographic_form_transnational"]
        and not form["geographic_form_internal"]
    ):
        return False
    if form_rule == "UNAVAILABLE" and form["geographic_form_eligible"]:
        return False
    if role.get("exclude_organized_criminal_group") and form[
        "organized_criminal_group_present"
    ]:
        return False

    words = record["text_input"]["word_count"]
    target = record["amp_targets"]
    checks = (
        (words, role.get("min_words"), role.get("max_words")),
        (len(target["act_ontology_ids"]), role.get("min_act_count"), None),
        (len(target["means_ontology_ids"]), role.get("min_means_count"), None),
    )
    if any(
        (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
        for value, minimum, maximum in checks
    ):
        return False
    if (
        role.get("exact_means_count") is not None
        and len(target["means_ontology_ids"]) != role["exact_means_count"]
    ):
        return False
    if (
        role.get("exact_total_label_count") is not None
        and len(covered) != role["exact_total_label_count"]
    ):
        return False
    text_regex = role.get("text_regex")
    return not text_regex or bool(
        re.search(
            text_regex,
            record["text_input"]["english_fact_summary_raw"],
            flags=re.IGNORECASE,
        )
    )


def select_demo_role_profile(
    candidates: Sequence[dict[str, Any]], roles: Sequence[dict[str, Any]]
) -> tuple[int, ...]:
    """Select one case per documented coverage role with stable tie breaks."""

    selected: list[dict[str, Any]] = []
    used_jurisdictions: set[str] = set()
    for role in roles:
        matches = [
            record
            for record in candidates
            if record["identity"]["jurisdiction_country_raw"]
            not in used_jurisdictions
            and record not in selected
            and demo_role_matches(record, role)
        ]
        if role.get("preference") == "MAX_LABELS_THEN_SHORTEST":
            matches.sort(
                key=lambda record: (
                    demo_clarity_penalty(record),
                    -len(labels(record)),
                    record["text_input"]["word_count"],
                    record["identity"]["search_rank"],
                )
            )
        else:
            matches.sort(
                key=lambda record: (
                    demo_clarity_penalty(record),
                    record["text_input"]["word_count"],
                    -len(labels(record)),
                    record["identity"]["search_rank"],
                )
            )
        if not matches:
            raise PreparationError(
                f"No eligible demonstration matches role {role['name']!r}"
            )
        chosen = matches[0]
        selected.append(chosen)
        used_jurisdictions.add(chosen["identity"]["jurisdiction_country_raw"])

    if len(selected) != DEMO_SET_SIZE:
        raise PreparationError("A demonstration profile did not select six cases")
    return tuple(sorted(record["identity"]["search_rank"] for record in selected))


def weighted_demo_set_cover(
    candidates: Sequence[dict[str, Any]],
    frequencies: dict[str, int],
    expected_n: int,
) -> list[tuple[int, ...]]:
    """Return diverse top deterministic six-case weighted set-cover solutions."""

    eligible = [
        record
        for record in candidates
        if record["geographic_form"]["geographic_form_eligible"]
        and 40 <= record["text_input"]["word_count"] <= 350
    ]
    rare_weights = {
        label: 1.0 + math.log(expected_n / count)
        for label, count in frequencies.items()
    }
    features = []
    for record in eligible:
        form = record["geographic_form"]
        form_values = frozenset(
            value
            for value, present in (
                ("INTERNAL", form["geographic_form_internal"]),
                ("TRANSNATIONAL", form["geographic_form_transnational"]),
            )
            if present
        )
        quality = demo_set_cover_quality(record)
        features.append(
            (
                record["identity"]["search_rank"],
                record["identity"]["jurisdiction_country_raw"],
                record["text_input"]["word_count"],
                frozenset(labels(record)),
                form_values,
                quality,
            )
        )

    # state = ranks, labels, forms, jurisdictions, quality sum, word sum
    states: list[
        tuple[
            tuple[int, ...],
            frozenset[str],
            frozenset[str],
            frozenset[str],
            float,
            int,
        ]
    ] = [((), frozenset(), frozenset(), frozenset(), 0.0, 0)]

    def score(state: tuple[Any, ...]) -> float:
        _, covered, forms, _, quality_sum, words = state
        return (
            sum(rare_weights[label] for label in covered)
            + 1.5 * len(forms)
            + 0.35 * quality_sum
            - 0.0004 * words
        )

    for _ in range(DEMO_SET_SIZE):
        beam: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
        for state in states:
            ranks, covered, forms, jurisdictions, quality_sum, words = state
            last_rank = ranks[-1] if ranks else -1
            for rank, jurisdiction, word_count, case_labels, case_forms, quality in features:
                if rank <= last_rank or jurisdiction in jurisdictions:
                    continue
                expanded = (
                    ranks + (rank,),
                    covered | case_labels,
                    forms | case_forms,
                    jurisdictions | {jurisdiction},
                    quality_sum + quality,
                    words + word_count,
                )
                # The rank component makes the heap key unique and supplies a
                # stable ascending-rank tie break after score and length.
                key = (
                    score(expanded),
                    -expanded[5],
                    tuple(-value for value in expanded[0]),
                )
                entry = (key, expanded)
                if len(beam) < DEMO_SET_COVER_BEAM_WIDTH:
                    heapq.heappush(beam, entry)
                elif key > beam[0][0]:
                    heapq.heapreplace(beam, entry)
        states = [entry[1] for entry in beam]
        if not states:
            raise PreparationError("Weighted demonstration set-cover beam is empty")

    full_label_count = len(frequencies)
    ranked = sorted(
        (state for state in states if len(state[1]) == full_label_count),
        key=lambda state: (-score(state), state[5], state[0]),
    )
    chosen: list[tuple[int, ...]] = []
    for state in ranked:
        ranks = state[0]
        # Avoid returning near-identical variants that differ by only one case.
        if any(len(set(ranks) & set(existing)) >= 5 for existing in chosen):
            continue
        chosen.append(ranks)
        if len(chosen) == DEMO_SET_COVER_COUNT:
            break
    if len(chosen) != DEMO_SET_COVER_COUNT:
        raise PreparationError("Could not construct five diverse full-cover demo sets")
    return chosen


def select_demo_candidate_sets(
    records: Sequence[dict[str, Any]],
    warnings: dict[int, list[str]],
    high_jurisdictions: set[str],
) -> tuple[tuple[int, ...], ...]:
    candidates = demo_candidate_pool(records, warnings, high_jurisdictions)
    frequencies = Counter(label for record in records for label in labels(record))
    profile_sets = [
        select_demo_role_profile(candidates, roles)
        for _, roles in DEMO_SELECTION_PROFILES
    ]
    weighted_sets = weighted_demo_set_cover(candidates, frequencies, len(records))
    result = tuple(profile_sets[:2] + weighted_sets + profile_sets[2:])
    if len(result) != len(set(result)) or len(result) != 9:
        raise PreparationError("Demonstration shortlist is not nine unique sets")
    by_rank = {record["identity"]["search_rank"]: record for record in candidates}
    for rank_set in result:
        selected = [by_rank[rank] for rank in rank_set]
        if len({item["identity"]["jurisdiction_country_raw"] for item in selected}) != 6:
            raise PreparationError("A demo proposal repeats a jurisdiction")
        if len({label for item in selected for label in labels(item)}) != 17:
            raise PreparationError("A demo proposal does not cover all 17 AMP labels")
    return result


def demo_set_score(
    selected: Sequence[dict[str, Any]],
    label_ids: Sequence[str],
    frequencies: dict[str, int],
) -> dict[str, Any]:
    covered = {label for record in selected for label in labels(record)}
    acts = [value for value in label_ids[:5] if value in covered]
    means = [value for value in label_ids[5:11] if value in covered]
    purposes = [value for value in label_ids[11:] if value in covered]
    rare_score = sum(1.0 / math.sqrt(frequencies[value]) for value in covered)
    form_covered: list[str] = []
    if any(item["geographic_form"]["geographic_form_internal"] for item in selected):
        form_covered.append("INTERNAL")
    if any(item["geographic_form"]["geographic_form_transnational"] for item in selected):
        form_covered.append("TRANSNATIONAL")
    words = sum(item["text_input"]["word_count"] for item in selected)
    return {
        "covered_label_ids": covered,
        "acts": acts,
        "means": means,
        "purposes": purposes,
        "form": form_covered,
        "rare_score": rare_score,
        "words": words,
    }


def build_demo_artifacts(
    records: Sequence[dict[str, Any]],
    label_ids: Sequence[str],
    warnings: dict[int, list[str]],
    tokenizer: Any,
    high_jurisdictions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    global DEMO_CANDIDATE_RANK_SETS, PROVISIONAL_DEMO_RANKS
    DEMO_CANDIDATE_RANK_SETS = select_demo_candidate_sets(
        records, warnings, high_jurisdictions
    )
    PROVISIONAL_DEMO_RANKS = frozenset(DEMO_CANDIDATE_RANK_SETS[0])
    by_rank = {item["identity"]["search_rank"]: item for item in records}
    frequencies = Counter(label for item in records for label in labels(item))
    candidate_rows: list[dict[str, Any]] = []
    memberships: dict[int, list[str]] = {}
    set_diagnostics: list[dict[str, Any]] = []
    for index, rank_set in enumerate(DEMO_CANDIDATE_RANK_SETS, 1):
        set_id = f"demo-bank-proposal-set-{index:02d}-v1"
        selected = [by_rank[rank] for rank in rank_set]
        score = demo_set_score(selected, label_ids, frequencies)
        token_count = sum(
            len(tokenizer(item["text_input"]["english_fact_summary_raw"], add_special_tokens=True)["input_ids"])
            for item in selected
        )
        ambiguity: list[str] = [
            "HUMAN_LABEL_SUPPORT_REVIEW_REQUIRED",
            "REFERENCE_LABELS_MAY_EXCEED_NARRATIVE_SUPPORT",
        ]
        if any(warnings.get(rank) for rank in rank_set):
            ambiguity.append("FACT_SUMMARY_PARSER_WARNING")
        candidate_rows.append(
            {
                "candidate_set_rank": index,
                "candidate_set_id": set_id,
                "status": "PROPOSED_NOT_FROZEN",
                "ranking_basis": DEMO_CANDIDATE_BASIS[index - 1],
                "search_ranks_json": canonical_json(list(rank_set)),
                "case_ids_json": canonical_json(
                    [by_rank[rank]["identity"].get("unodc_case_number") for rank in rank_set]
                ),
                "jurisdictions_json": canonical_json(
                    [by_rank[rank]["identity"]["jurisdiction_country_raw"] for rank in rank_set]
                ),
                "amp_label_ids_covered_json": canonical_json(
                    [value for value in label_ids if value in score["covered_label_ids"]]
                ),
                "act_labels_covered": len(score["acts"]),
                "means_labels_covered": len(score["means"]),
                "purpose_labels_covered": len(score["purposes"]),
                "total_amp_labels_covered": len(score["covered_label_ids"]),
                "form_coverage_json": canonical_json(score["form"]),
                "total_fact_summary_word_count": score["words"],
                "mean_fact_summary_word_count": f"{score['words'] / 6:.3f}",
                "total_modernbert_summary_tokens": token_count,
                "rare_label_coverage_score": f"{score['rare_score']:.9f}",
                "candidate_set_membership_sha256": membership_digest(
                    (rank, by_rank[rank]["identity"]["canonical_url"]) for rank in rank_set
                ),
                "warning_or_ambiguity_flags": "|".join(ambiguity),
                "review_notes": DEMO_CANDIDATE_NOTES[index - 1],
            }
        )
        for rank in rank_set:
            memberships.setdefault(rank, []).append(set_id)
        set_diagnostics.append({"set_id": set_id, "ranks": list(rank_set), **score})

    review_ranks = sorted(
        {rank for rank_set in DEMO_CANDIDATE_RANK_SETS[:3] for rank in rank_set}
    )
    review_rows: list[dict[str, Any]] = []
    for rank in review_ranks:
        item = by_rank[rank]
        amp = item["amp_targets"]
        form = item["geographic_form"]
        review_rows.append(
            {
                "search_rank": rank,
                "case_title": item["identity"]["case_title_raw"],
                "canonical_url": item["identity"]["canonical_url"],
                "jurisdiction": item["identity"]["jurisdiction_country_raw"],
                "english_fact_summary": item["text_input"]["english_fact_summary_raw"],
                "legacy_acts_reference_json": canonical_json(amp["acts_raw"]),
                "legacy_means_reference_json": canonical_json(amp["means_raw"]),
                "legacy_purposes_reference_json": canonical_json(amp["purposes_raw"]),
                "act_ontology_ids_json": canonical_json(amp["act_ontology_ids"]),
                "means_ontology_ids_json": canonical_json(amp["means_ontology_ids"]),
                "purpose_ontology_ids_json": canonical_json(amp["purpose_ontology_ids"]),
                "geographic_form_reference_json": canonical_json(
                    form["legacy_form_values_raw"]
                ),
                "word_count": item["text_input"]["word_count"],
                "modernbert_summary_tokens": len(
                    tokenizer(item["text_input"]["english_fact_summary_raw"], add_special_tokens=True)["input_ids"]
                ),
                "proposed_demo_order": (
                    list(DEMO_CANDIDATE_RANK_SETS[0]).index(rank) + 1
                    if rank in DEMO_CANDIDATE_RANK_SETS[0]
                    else ""
                ),
                "selection_reason": (
                    "Outside all A2 held-out jurisdictions; complete Legacy AMP; "
                    "warning-free usable English; concise set-cover contribution. "
                    "Researcher must verify every displayed label and Form value "
                    "against the narrative before approval."
                ),
                "candidate_set_memberships": "|".join(memberships[rank]),
                "fact_summary_parser_warnings": "|".join(warnings.get(rank, [])),
                "reference_support_review_status": "PENDING_HUMAN_REVIEW",
                "reviewer_approve": "",
                "reviewer_notes": "",
            }
        )

    if any(
        item["identity"]["jurisdiction_country_raw"] in high_jurisdictions
        for rank in PROVISIONAL_DEMO_RANKS
        for item in [by_rank[rank]]
    ):
        raise PreparationError("A proposed demo jurisdiction is in A2 held-out universe")
    return candidate_rows, review_rows, {
        "provisional_set": set_diagnostics[0],
        "candidate_sets": set_diagnostics,
    }


def exact_iterative_split(
    records: Sequence[dict[str, Any]],
    label_ids: Sequence[str],
    test_n: int,
    seed_prefix: int,
) -> tuple[list[int], list[int], int]:
    y = label_matrix(records, label_ids)
    x = np.zeros((len(records), 1), dtype=np.int8)
    for offset in range(10000):
        seed = seed_prefix + offset
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_n / len(records), random_state=seed
        )
        train, test = next(splitter.split(x, y))
        if len(test) == test_n:
            return train.tolist(), test.tolist(), seed
    raise PreparationError(f"Could not obtain exact iterative split of {test_n}")


def build_a1(
    records: Sequence[dict[str, Any]], label_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    demos = [item for item in records if item["identity"]["search_rank"] in PROVISIONAL_DEMO_RANKS]
    pool = [item for item in records if item["identity"]["search_rank"] not in PROVISIONAL_DEMO_RANKS]
    if len(demos) != 6 or len(pool) != 1257:
        raise PreparationError("A1 demo reservation failed")
    non_test_idx, test_idx, test_seed = exact_iterative_split(pool, label_ids, 253, SEED + 1000)
    non_test = [pool[index] for index in non_test_idx]
    train_idx, validation_idx, validation_seed = exact_iterative_split(
        non_test, label_ids, 126, SEED + 2000
    )
    role_by_rank: dict[int, str] = {
        item["identity"]["search_rank"]: "DEMO" for item in demos
    }
    role_by_rank.update(
        {non_test[index]["identity"]["search_rank"]: "TRAIN" for index in train_idx}
    )
    role_by_rank.update(
        {non_test[index]["identity"]["search_rank"]: "VALIDATION" for index in validation_idx}
    )
    role_by_rank.update(
        {pool[index]["identity"]["search_rank"]: "TEST" for index in test_idx}
    )
    rows: list[dict[str, Any]] = []
    for item in records:
        rank = item["identity"]["search_rank"]
        role = role_by_rank[rank]
        selected = set(labels(item))
        row: dict[str, Any] = {
            "search_rank": rank,
            "canonical_url": item["identity"]["canonical_url"],
            "jurisdiction": item["identity"]["jurisdiction_country_raw"],
            "split": role,
            "effective_supervised_train": int(role in {"TRAIN", "DEMO"}),
            "demo_set_id": PROVISIONAL_DEMO_SET_ID,
            "split_status": "PROVISIONAL_PENDING_DEMO_APPROVAL",
            "amp_positive_label_count": len(selected),
            "geographic_form_eligible": item["geographic_form"]["geographic_form_eligible"],
            "geographic_form_internal": item["geographic_form"]["geographic_form_internal"],
            "geographic_form_transnational": item["geographic_form"]["geographic_form_transnational"],
        }
        row.update({label: int(label in selected) for label in label_ids})
        rows.append(row)
    diagnostics = {
        "counts": Counter(role_by_rank.values()),
        "test_seed": test_seed,
        "validation_seed": validation_seed,
        "membership_sha256": membership_digest(
            (row["search_rank"], row["canonical_url"], row["split"]) for row in rows
        ),
        "label_counts": label_counts_by_role(records, role_by_rank, label_ids),
        "form_counts": form_counts_by_role(records, role_by_rank),
    }
    return rows, diagnostics


A2_TEST_JURISDICTIONS = {
    1: (
        "Argentina", "Australia", "Republic of Moldova", "Romania", "Serbia", "Slovakia",
    ),
    2: (
        "Belgium", "Brazil", "Czechia", "India", "Philippines", "Sweden",
    ),
    3: (
        "Canada", "Colombia", "Poland", "Ukraine",
        "United Kingdom of Great Britain and Northern Ireland",
        "United States of America",
    ),
}


def derive_a2_partition(
    records: Sequence[dict[str, Any]], label_ids: Sequence[str]
) -> dict[int, tuple[str, ...]]:
    """Exhaustively reproduce the frozen balanced 6/6/6 jurisdiction groups.

    Every unlabeled group permutation is considered once.  Candidate groups
    must have a test-size range no greater than two cases (the smallest
    practical near-equality constraint selected before label inspection).  The
    deterministic objective then minimizes maximum absolute AMP-prevalence
    deviation, summed squared deviation, size range, and lexicographic groups.
    Purpose-removal is absent from this universe and contributes no deviation.
    """
    import itertools

    counts = Counter(item["identity"]["jurisdiction_country_raw"] for item in records)
    names = tuple(sorted(name for name, count in counts.items() if count >= 20))
    if len(names) != 18:
        raise PreparationError("A2 partition derivation requires 18 jurisdictions")
    sizes = np.asarray([counts[name] for name in names], dtype=np.int16)
    label_counts = np.asarray(
        [
            [
                sum(
                    label in labels(item)
                    for item in records
                    if item["identity"]["jurisdiction_country_raw"] == name
                )
                for label in label_ids
            ]
            for name in names
        ],
        dtype=np.int16,
    )
    combinations = np.asarray(list(itertools.combinations(range(18), 6)), dtype=np.int8)
    masks = np.asarray(
        [sum(1 << int(index) for index in group) for group in combinations],
        dtype=np.uint32,
    )
    group_sizes = sizes[combinations].sum(axis=1)
    group_labels = label_counts[combinations].sum(axis=1)
    full_mask = (1 << 18) - 1
    total_n = int(sizes.sum())
    pooled = label_counts.sum(axis=0)
    observable = pooled > 0
    pooled_prevalence = pooled[observable] / total_n
    mask_to_index = {int(mask): index for index, mask in enumerate(masks)}
    best: tuple[Any, ...] | None = None

    # Anchor the alphabetically first jurisdiction in group 1 and the smallest
    # remaining jurisdiction in group 2 to eliminate fold permutations.
    for first_index in np.flatnonzero((masks & 1) != 0):
        first_mask = int(masks[first_index])
        remaining_mask = full_mask ^ first_mask
        second_anchor = remaining_mask & -remaining_mask
        valid_second = np.flatnonzero(
            ((masks & first_mask) == 0) & ((masks & second_anchor) != 0)
        )
        first_n = int(group_sizes[first_index])
        second_n = group_sizes[valid_second].astype(np.int32)
        third_n = total_n - first_n - second_n
        size_range = (
            np.maximum(np.maximum(first_n, second_n), third_n)
            - np.minimum(np.minimum(first_n, second_n), third_n)
        )
        valid_second = valid_second[size_range <= 2]
        if not len(valid_second):
            continue
        second_n = group_sizes[valid_second].astype(np.int32)
        third_n = total_n - first_n - second_n
        first_labels = group_labels[first_index, observable].astype(float)
        second_labels = group_labels[valid_second][:, observable].astype(float)
        third_labels = pooled[observable] - first_labels - second_labels
        deviations = np.stack(
            [
                np.broadcast_to(first_labels / first_n, second_labels.shape),
                second_labels / second_n[:, None],
                third_labels / third_n[:, None],
            ],
            axis=1,
        ) - pooled_prevalence
        maximum = np.abs(deviations).max(axis=(1, 2))
        squared = (deviations * deviations).sum(axis=(1, 2))
        for candidate_offset in np.flatnonzero(
            np.isclose(maximum, maximum.min(), rtol=0, atol=1e-15)
        ):
            second_index = int(valid_second[candidate_offset])
            third_index = mask_to_index[
                full_mask ^ first_mask ^ int(masks[second_index])
            ]
            groups = tuple(
                tuple(names[index] for index in combinations[group_index])
                for group_index in (first_index, second_index, third_index)
            )
            sizes_here = tuple(
                int(group_sizes[group_index])
                for group_index in (first_index, second_index, third_index)
            )
            key = (
                float(maximum[candidate_offset]),
                float(squared[candidate_offset]),
                max(sizes_here) - min(sizes_here),
                groups,
            )
            if best is None or key < best:
                best = key
    if best is None:
        raise PreparationError("A2 partition search found no near-equal partition")
    return {fold: best[3][fold - 1] for fold in (1, 2, 3)}


def build_a2(
    records: Sequence[dict[str, Any]], label_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    diagnostics: dict[int, dict[str, Any]] = {}
    for fold in (1, 2, 3):
        heldout = set(A2_TEST_JURISDICTIONS[fold])
        demos = [item for item in records if item["identity"]["search_rank"] in PROVISIONAL_DEMO_RANKS]
        test = [item for item in records if item["identity"]["jurisdiction_country_raw"] in heldout]
        pool = [
            item for item in records
            if item["identity"]["jurisdiction_country_raw"] not in heldout
            and item["identity"]["search_rank"] not in PROVISIONAL_DEMO_RANKS
        ]
        train_idx, validation_idx, validation_seed = exact_iterative_split(
            pool, label_ids, 98, SEED + fold * 10000
        )
        role_by_rank = {item["identity"]["search_rank"]: "TEST" for item in test}
        role_by_rank.update({item["identity"]["search_rank"]: "DEMO" for item in demos})
        role_by_rank.update({pool[index]["identity"]["search_rank"]: "TRAIN" for index in train_idx})
        role_by_rank.update(
            {pool[index]["identity"]["search_rank"]: "VALIDATION" for index in validation_idx}
        )
        for item in records:
            rank = item["identity"]["search_rank"]
            role = role_by_rank[rank]
            selected = set(labels(item))
            row: dict[str, Any] = {
                "search_rank": rank,
                "canonical_url": item["identity"]["canonical_url"],
                "jurisdiction": item["identity"]["jurisdiction_country_raw"],
                "fold_id": fold,
                "role": role,
                "heldout_jurisdiction": int(item["identity"]["jurisdiction_country_raw"] in heldout),
                "effective_supervised_train": int(role in {"TRAIN", "DEMO"}),
                "demo_set_id": PROVISIONAL_DEMO_SET_ID,
                "split_status": "PROVISIONAL_PENDING_DEMO_APPROVAL",
                "amp_positive_label_count": len(selected),
                "geographic_form_eligible": item["geographic_form"]["geographic_form_eligible"],
            }
            row.update({label: int(label in selected) for label in label_ids})
            all_rows.append(row)
        diagnostics[fold] = {
            "heldout": tuple(sorted(heldout)),
            "counts": Counter(role_by_rank.values()),
            "validation_seed": validation_seed,
            "membership_sha256": membership_digest(
                (rank, role_by_rank[rank]) for rank in sorted(role_by_rank)
            ),
            "label_counts": label_counts_by_role(records, role_by_rank, label_ids),
        }
    return all_rows, diagnostics


def label_counts_by_role(
    records: Sequence[dict[str, Any]], role_by_rank: dict[int, str], label_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for role in sorted(set(role_by_rank.values())):
        subset = [item for item in records if role_by_rank[item["identity"]["search_rank"]] == role]
        result[role] = {
            label: sum(label in labels(item) for item in subset) for label in label_ids
        }
    return result


def form_counts_by_role(
    records: Sequence[dict[str, Any]], role_by_rank: dict[int, str]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for role in sorted(set(role_by_rank.values())):
        subset = [item for item in records if role_by_rank[item["identity"]["search_rank"]] == role]
        result[role] = {
            "n": len(subset),
            "eligible": sum(item["geographic_form"]["geographic_form_eligible"] for item in subset),
            "internal": sum(item["geographic_form"]["geographic_form_internal"] for item in subset),
            "transnational": sum(item["geographic_form"]["geographic_form_transnational"] for item in subset),
            "both": sum(
                item["geographic_form"]["geographic_form_internal"]
                and item["geographic_form"]["geographic_form_transnational"]
                for item in subset
            ),
        }
    return result


def token_audit(
    records: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lengths: list[int] = []
    rows: list[dict[str, Any]] = []
    for item in records:
        length = len(
            tokenizer(
                item["text_input"]["english_fact_summary_raw"],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        lengths.append(length)
        row: dict[str, Any] = {
            "search_rank": item["identity"]["search_rank"],
            "canonical_url": item["identity"]["canonical_url"],
            "jurisdiction": item["identity"]["jurisdiction_country_raw"],
            "word_count": item["text_input"]["word_count"],
            "modernbert_token_count_with_special_tokens": length,
            "tokenizer_model_id": MODEL_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
        }
        row.update({f"fully_covered_at_{threshold}": int(length <= threshold) for threshold in TOKEN_THRESHOLDS})
        rows.append(row)
    array = np.asarray(lengths)
    stats = {
        "n": len(lengths),
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
        "coverage": {
            threshold: {
                "covered": int((array <= threshold).sum()),
                "percent": float(100 * (array <= threshold).mean()),
                "truncated": int((array > threshold).sum()),
            }
            for threshold in TOKEN_THRESHOLDS
        },
        "recommended_max_length": 2048,
    }
    return rows, stats


def proxy_token_estimates(
    records: Sequence[dict[str, Any]],
    a1_rows: Sequence[dict[str, Any]],
    a2_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Estimate LLM input scale offline with the pinned ModernBERT tokenizer.

    This is intentionally an approximate planning proxy, not OpenAI billing or
    exact model tokenization.  It serializes the actual frozen prompt/schema and
    provisional six-demo payload content without constructing or sending an API
    request.
    """
    import importlib.util

    builder_path = REPO_ROOT / "src/experiments/llm_request_builder.py"
    spec = importlib.util.spec_from_file_location("sherloc_llm_request_builder", builder_path)
    if spec is None or spec.loader is None:
        raise PreparationError("Could not import the offline LLM request builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    contract = builder.load_contract()

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    target_stub = "Extract this case from the supplied evidence only:\n" + builder.canonical_json(
        {"fact_summary": ""}
    )
    fixed = (
        count(contract["developer_instruction"])
        + count(builder.canonical_json(contract["config"]["structured_output"]["schema"]))
        + count(target_stub)
    )
    target_lengths = {
        item["identity"]["search_rank"]: count(item["text_input"]["english_fact_summary_raw"])
        for item in records
    }
    by_rank = {item["identity"]["search_rank"]: item for item in records}
    demo_tokens = 0
    for rank in sorted(PROVISIONAL_DEMO_RANKS):
        item = by_rank[rank]
        form = item["geographic_form"]
        output = {
            "acts": item["amp_targets"]["act_ontology_ids"],
            "means": item["amp_targets"]["means_ontology_ids"],
            "purposes": item["amp_targets"]["purpose_ontology_ids"],
            "geographic_form": {
                "internal": bool(form["geographic_form_internal"]),
                "transnational": bool(form["geographic_form_transnational"]),
            },
        }
        demo_tokens += count(
            "Extract this case from the supplied evidence only:\n"
            + builder.canonical_json(
                {"fact_summary": item["text_input"]["english_fact_summary_raw"]}
            )
        )
        demo_tokens += count(json.dumps(output, ensure_ascii=False, separators=(",", ":")))

    a1_test_ranks = [int(row["search_rank"]) for row in a1_rows if row["split"] == "TEST"]
    a2_test_ranks = [int(row["search_rank"]) for row in a2_rows if row["role"] == "TEST"]

    def summarize(ranks: Sequence[int]) -> dict[str, Any]:
        values = np.asarray([target_lengths[rank] for rank in ranks])
        return {
            "requests": len(ranks),
            "target_median": float(np.median(values)),
            "target_p90": float(np.percentile(values, 90)),
            "target_total": int(values.sum()),
            "m3_input_total": int(fixed * len(ranks) + values.sum()),
            "m4_input_total": int((fixed + demo_tokens) * len(ranks) + values.sum()),
        }

    return {
        "estimator": MODEL_ID,
        "estimator_revision": TOKENIZER_REVISION,
        "fixed_instruction_schema_and_wrapper_tokens": fixed,
        "six_provisional_demo_tokens": demo_tokens,
        "prompt_contract_sha256": contract["marked_block_sha256"],
        "llm_config_sha256": contract["config_sha256"],
        "a1_test": summarize(a1_test_ranks),
        "a2_all_test_assignments": summarize(a2_test_ranks),
    }


def markdown_label_table(label_ids: Sequence[str], counts: dict[str, dict[str, int]]) -> str:
    roles = [role for role in ("TRAIN", "VALIDATION", "TEST", "DEMO") if role in counts]
    lines = ["| Label | " + " | ".join(roles) + " |", "|---|" + "---:|" * len(roles)]
    for label in label_ids:
        lines.append("| `" + label + "` | " + " | ".join(f"{counts[role][label]:,}" for role in roles) + " |")
    return "\n".join(lines)


def build_a1_report(diagnostics: dict[str, Any], label_ids: Sequence[str]) -> str:
    counts = diagnostics["counts"]
    organ = "PURPOSE_REMOVAL_OF_ORGANS"
    organ_values = diagnostics["label_counts"]
    form_lines = ["| Role | N | Eligible | Internal | Transnational | Both |", "|---|---:|---:|---:|---:|---:|"]
    for role in ("TRAIN", "VALIDATION", "TEST", "DEMO"):
        value = diagnostics["form_counts"][role]
        form_lines.append(
            f"| {role} | {value['n']} | {value['eligible']} | {value['internal']} | {value['transnational']} | {value['both']} |"
        )
    return f"""# A1 IID split v1 report

Status: **PROVISIONAL — tied to `{PROVISIONAL_DEMO_SET_ID}` pending human demo approval**  
Generator: `src/experiments/06_prepare_experiments.py` v{VERSION}  
Seed family: `{SEED}`

## Exact allocation

| Role | Cases | Effective supervised training |
|---|---:|---:|
| TRAIN | {counts['TRAIN']} | {counts['TRAIN']} |
| VALIDATION | {counts['VALIDATION']} | 0 |
| TEST | {counts['TEST']} | 0 |
| DEMO | {counts['DEMO']} | {counts['DEMO']} |

The six proposed demonstrations were reserved before splitting. The remaining
1,257 cases were divided by iterative multilabel stratification over all 17 AMP
indicators. Exact splitter seeds were `{diagnostics['test_seed']}` for TEST and
`{diagnostics['validation_seed']}` for VALIDATION. M1/M2 effective training is
TRAIN + DEMO = **{counts['TRAIN'] + counts['DEMO']}**. M4 demonstrations are
excluded from every reported metric.

Membership SHA-256: `{diagnostics['membership_sha256']}`.

## AMP frequencies

{markdown_label_table(label_ids, diagnostics['label_counts'])}

The ten organ-removal cases allocate as TRAIN **{organ_values['TRAIN'][organ]}**,
VALIDATION **{organ_values['VALIDATION'][organ]}**, TEST **{organ_values['TEST'][organ]}**,
and DEMO **{organ_values['DEMO'][organ]}**. Validation and test therefore retain
the rare label.

## Geographic Form audit

{chr(10).join(form_lines)}

Form values were audited after AMP-first splitting. Ineligible Form cases are
not interpreted as two reference negatives.

## Integrity and use restriction

- Exactly 1,263 unique benchmark cases are assigned once.
- DEMO is disjoint from VALIDATION and TEST and is effective supervised training.
- No model result or test performance informed this split.
- A1 test labels may be used only for integrity checks until final evaluation.
- Prompt wording, demonstrations, hyperparameters, and thresholds must not be
  selected from A1 test errors or labels.
- If any proposed demo is rejected or replaced, regenerate this file and record
  a new membership hash before model execution.
"""


def build_a2_report(
    diagnostics: dict[int, dict[str, Any]], label_ids: Sequence[str]
) -> str:
    sections: list[str] = []
    for fold in (1, 2, 3):
        value = diagnostics[fold]
        counts = value["counts"]
        sections.append(
            f"""### Fold {fold}

Held-out jurisdictions ({len(value['heldout'])}): """
            + "; ".join(value["heldout"])
            + f"""

| Role | Cases |
|---|---:|
| TRAIN | {counts['TRAIN']} |
| VALIDATION | {counts['VALIDATION']} |
| TEST | {counts['TEST']} |
| DEMO | {counts['DEMO']} |
| Effective supervised training | {counts['TRAIN'] + counts['DEMO']} |

Membership SHA-256: `{value['membership_sha256']}`. Validation splitter seed:
`{value['validation_seed']}`.

{markdown_label_table(label_ids, value['label_counts'])}
"""
        )
    return f"""# A2 jurisdiction-disjoint folds v1 report

Status: **PROVISIONAL — tied to `{PROVISIONAL_DEMO_SET_ID}` pending human demo approval**  
Generator: `src/experiments/06_prepare_experiments.py` v{VERSION}

## Evaluation universe and fold design

The frozen primary cohort contains exactly **18** jurisdiction/category values
with at least 20 cases: **861** cases total. The remaining **402** smaller-
jurisdiction cases can enter training/validation in every fold. The 18 values
were partitioned into three disjoint six-jurisdiction test groups by an
exhaustive deterministic balance objective. We first impose a near-equal test-
size constraint (range at most two cases), then minimize maximum absolute AMP-
prevalence deviation, summed squared deviation, size range, and lexicographic
group order. The generator recomputes this objective from the frozen benchmark
and fails if it no longer yields the recorded groups.

Test sizes are **288 / 287 / 286** (range 2). All 16 AMP labels observed in the
high-support universe appear in every test fold. All ten
`PURPOSE_REMOVAL_OF_ORGANS` cases occur in smaller jurisdictions, so the A2 test
count is unavoidably **0 / 0 / 0**; A2 cannot estimate jurisdiction-transfer
performance for that label.

{chr(10).join(sections)}

## Leakage checks and use restriction

- Every high-support jurisdiction is TEST in exactly one fold.
- A held-out jurisdiction never appears in TRAIN, VALIDATION, or DEMO in its fold.
- The six proposed demos are outside all 18 high-support jurisdictions, have
  role DEMO in all folds, and are never scored.
- Each fold contains all 1,263 cases exactly once; the long file has 3,789 rows.
- Test labels are restricted to integrity checking until final evaluation and
  must not guide prompts, demos, hyperparameters, or thresholds.
- Demo replacement requires regenerating A2 train/validation assignments and
  hashes; held-out jurisdiction TEST membership remains independently fixed.
"""


def build_preparation_report(
    demo_diagnostics: dict[str, Any],
    token_stats: dict[str, Any],
    proxy_tokens: dict[str, Any],
    a1_diagnostics: dict[str, Any],
    a2_diagnostics: dict[int, dict[str, Any]],
) -> str:
    provisional = demo_diagnostics["provisional_set"]
    coverage_lines = [
        "| Max length | Fully covered | Percent | Truncated |",
        "|---:|---:|---:|---:|",
    ]
    for threshold in TOKEN_THRESHOLDS:
        value = token_stats["coverage"][threshold]
        coverage_lines.append(
            f"| {threshold:,} | {value['covered']:,} | {value['percent']:.3f}% | {value['truncated']:,} |"
        )
    fold_lines = [
        "| Fold | Held-out jurisdictions | TRAIN | VALIDATION | TEST | DEMO |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for fold in (1, 2, 3):
        value = a2_diagnostics[fold]
        counts = value["counts"]
        fold_lines.append(
            f"| {fold} | {'; '.join(value['heldout'])} | {counts['TRAIN']} | "
            f"{counts['VALIDATION']} | {counts['TEST']} | {counts['DEMO']} |"
        )
    a1_tokens = proxy_tokens["a1_test"]
    a2_tokens = proxy_tokens["a2_all_test_assignments"]
    return f"""# Evaluation A experiment-preparation report

Preparation date: `{PREPARATION_DATE}`  
Generator: `src/experiments/06_prepare_experiments.py` v{VERSION}  
Frozen cohort: `{PRIMARY_COHORT_ID}`  
Status: **PREPARATION COMPLETE; DEMO BANK AND DEPENDENT SPLITS PROVISIONAL**

No model was trained, no prediction was generated, and no OpenAI API request
was made in this stage.

## 1. A1 IID split

The six proposal-set-01 cases were reserved first. Iterative multilabel
stratification over all 17 AMP indicators then assigned the remaining 1,257
cases to TRAIN **{a1_diagnostics['counts']['TRAIN']}**, VALIDATION
**{a1_diagnostics['counts']['VALIDATION']}**, and TEST
**{a1_diagnostics['counts']['TEST']}**; DEMO is **6**. M1/M2 effective training
is **{a1_diagnostics['counts']['TRAIN'] + 6}** because the six demos remain in
their supervised training pool. `PURPOSE_REMOVAL_OF_ORGANS` allocates
TRAIN/VALIDATION/TEST/DEMO = **6/1/2/1**. Membership SHA-256 is
`{a1_diagnostics['membership_sha256']}`. All roles are disjoint and every
benchmark row appears once.

## 2. A2 jurisdiction-disjoint folds

The high-support universe is exactly **18 jurisdictions/categories and 861
cases**; 402 smaller-jurisdiction cases remain available to each non-test pool.
The exhaustive deterministic balance search produces:

{chr(10).join(fold_lines)}

All 18 jurisdictions are held out exactly once, no held-out jurisdiction enters
its fold's TRAIN/VALIDATION/DEMO roles, and demos never enter TEST. All ten organ-
removal cases lie outside the high-support universe, making the A2 test count
unavoidably 0 in every fold.

## 3. Six-demonstration shortlist

Nine candidate banks were produced from frozen inputs using explicit eligibility
gates, ontology/Form coverage profiles, and a rare-label-weighted deterministic
set-cover search. Every proposal uses six distinct jurisdictions outside the A2
held-out universe and covers 5 Act, 6 Means, 6 Purpose, INTERNAL, and
TRANSNATIONAL reference values.

Proposal set 01 (the provisional split anchor) uses ranks
**{', '.join(map(str, provisional['ranks']))}** from
**Malta; North Macedonia; Guatemala; Albania; Hungary; Jordan**.
It contains **{provisional['words']:,} words** (mean
**{provisional['words'] / 6:.1f}**) and **758 ModernBERT summary tokens**. Its
reference coverage is **{len(provisional['acts'])}/5 Acts,
{len(provisional['means'])}/6 Means, {len(provisional['purposes'])}/6 Purposes**
and both Form values.

This is not the permanent M4 bank. The researcher/HT expert must review
`data/annotations/demo_bank_review.csv` and confirm that each Fact Summary
actually supports every Legacy reference label and Form value. Known concerns
include Slavery/Servitude, Transfer, Fraud, Internal Form, and focal-victim
status for the organ-removal example. Any replacement requires regeneration of
both split files and all membership hashes.

## 4. ModernBERT token audit

The official `answerdotai/ModernBERT-base` tokenizer was used at pinned revision
`{TOKENIZER_REVISION}`, with special tokens, across all 1,263 summaries.

- Min / median / mean: **{token_stats['min']} / {token_stats['median']:.0f} / {token_stats['mean']:.1f}**
- P75 / P90 / P95 / P99: **{token_stats['p75']:.0f} / {token_stats['p90']:.1f} / {token_stats['p95']:.1f} / {token_stats['p99']:.1f}**
- Maximum: **{token_stats['max']:,}**

{chr(10).join(coverage_lines)}

Recommended M2 `max_length`: **2,048**. It retains **1,254/1,263
({token_stats['coverage'][2048]['percent']:.3f}%)** summaries without truncation
while avoiding the substantially higher compute burden of full 8,192-token
training. The nine truncated case identities remain visible in the audit.

## 5. Frozen M1/M2 plans

M1 uses summary-only word 1-2 gram TF-IDF (sublinear TF, at most 50,000 features)
with one-vs-rest L2 logistic regression. The small validation grid is
`min_df={{1,2}}`, `C={{0.25,1,4}}`, and `class_weight={{null,balanced}}`.

M2 uses one `answerdotai/ModernBERT-base` encoder and one 17-logit multilabel
head, standard BCE-with-logits, max length 2,048, effective batch size 16,
learning rates 1e-5/2e-5/3e-5, weight decay 0.01/0.05, at most six epochs,
patience 2, the model's default mean pooling and zero classifier dropout, and
BF16/FP16/FP32 runtime fallback.

For both methods, 0.5 is retained as a declared baseline. Model selection uses
validation macro average precision; one global threshold may then be selected
on validation macro-F1. Per-label thresholds and all test-label tuning are
disabled in v1. Geographic Form is an LLM-only auxiliary result in the current
freeze: M1/M2 remain the requested 17-output AMP models unless a separate
supervised Form head is preregistered later.

## 6. M3/M4 prompt and request contract

M3 `m3-zero-shot-v1` and M4 `m4-six-shot-v1` use the same byte-identical
instruction block, exact 5/6/6 ontology, AMP-plus-Form targets, target wrapper,
strict JSON Schema, `gpt-5.6-luna`, low reasoning effort, low verbosity, and
512 maximum output tokens. The schema has closed required objects, enum-limited
arrays with family-size limits, and two required Form booleans. Duplicate labels
are prohibited by instruction and rejected by host validation. No rationale,
chain of thought, confidence, evidence spans, multiplicity, child/minor, or
Sector is requested.

Luna is retained because the frozen design explicitly assigns the high-volume,
schema-constrained extraction role to that tier. This is a preregistered method
choice, not a claim that Luna will outperform other GPT-5.6 tiers; changing it
later would create a different experiment configuration.

The only substantive M4 difference is six frozen user/assistant demonstration
pairs inserted between the common developer instruction and target. The
preparation-only builder never imports an API client or sends a request and
fails closed until exactly six unique, ordered, expert-approved and frozen
demos outside the 18 A2 jurisdictions are explicitly supplied.

## 7. Offline input-token scale

These are planning estimates from the pinned ModernBERT tokenizer, not exact GPT
token counts, usage records, billing estimates, or API calls. The serialized
common instruction, schema, and empty target wrapper are approximately
**{proxy_tokens['fixed_instruction_schema_and_wrapper_tokens']:,} proxy tokens**.
The provisional six-demo messages add approximately
**{proxy_tokens['six_provisional_demo_tokens']:,}**.

- A1 TEST: **{a1_tokens['requests']} requests**; target median **{a1_tokens['target_median']:.0f}**,
  P90 **{a1_tokens['target_p90']:.1f}**; M3 total approximately
  **{a1_tokens['m3_input_total']:,}** input tokens and provisional-M4 total
  approximately **{a1_tokens['m4_input_total']:,}**.
- A2: **{a2_tokens['requests']} M3 requests and {a2_tokens['requests']} M4 requests**
  across three held-out folds; M3 approximately **{a2_tokens['m3_input_total']:,}**
  input tokens and provisional-M4 approximately **{a2_tokens['m4_input_total']:,}**.

No dollar cost is estimated because no versioned local price schedule is frozen.

## 8. Methodological and launch guards

- A1/A2 test labels are used now only for split-integrity checks. They must not
  guide prompts, demos, hyperparameters, checkpoints, or thresholds.
- Fit preprocessing, class weights, and supervised models on TRAIN + DEMO only;
  use VALIDATION for selection; never score DEMO.
- Form metrics, when run for M3/M4, use only the 1,156 eligible cases.
- Preserve raw model/API outputs later; classify refusals, incomplete responses,
  API errors, schema errors, and duplicate-label errors explicitly. Never repair
  or silently coerce them into empty labels.

Before execution: (1) approve and freeze six demonstrations and rerun this
generator; (2) decide whether the model alias should be replaced by an available
dated snapshot and record the returned model identifier; (3) configure the API
key in the execution environment, not the repository; and (4) explicitly decide
whether Form remains LLM-only auxiliary or receives a separately preregistered
supervised comparator.

## 9. Reproducibility and validation

Random seed family: `{SEED}`. Frozen benchmark and ontology hashes are checked at
startup. The generator also fails closed on Python, NumPy, scikit-learn,
iterative-stratification, Transformers, or tokenizers version drift. A2 groups
are independently recomputed. The builder records input,
prompt, schema, demonstration-bank, config, and canonical request-payload hashes.
The experiment, request-builder, parser-v2, and benchmark safety tests must pass
before model execution.
"""


def validate_outputs(
    records: Sequence[dict[str, Any]],
    label_ids: Sequence[str],
    a1: Sequence[dict[str, Any]],
    a2: Sequence[dict[str, Any]],
    demo_rows: Sequence[dict[str, Any]],
    review_rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
) -> None:
    if len(a1) != 1263 or Counter(row["split"] for row in a1) != {
        "TRAIN": 878, "VALIDATION": 126, "TEST": 253, "DEMO": 6
    }:
        raise PreparationError("A1 exact counts failed")
    if any(
        int(row["effective_supervised_train"]) != int(row["split"] in {"TRAIN", "DEMO"})
        for row in a1
    ):
        raise PreparationError("A1 effective-supervised flag failed")
    if len(a2) != 3789:
        raise PreparationError("A2 must contain 1,263 x 3 rows")
    by_fold = {fold: [row for row in a2 if row["fold_id"] == fold] for fold in (1, 2, 3)}
    seen_test_jurisdictions: list[str] = []
    for fold, rows in by_fold.items():
        if len(rows) != 1263:
            raise PreparationError(f"A2 fold {fold} does not contain all cases")
        heldout = set(A2_TEST_JURISDICTIONS[fold])
        seen_test_jurisdictions.extend(heldout)
        for row in rows:
            should_test = row["jurisdiction"] in heldout
            if (row["role"] == "TEST") != should_test:
                raise PreparationError(f"A2 jurisdiction leakage in fold {fold}")
            if row["role"] == "DEMO" and row["heldout_jurisdiction"]:
                raise PreparationError("A2 demo entered held-out jurisdiction")
    if len(seen_test_jurisdictions) != 18 or len(set(seen_test_jurisdictions)) != 18:
        raise PreparationError("A2 held-out jurisdictions are not unique")
    if len(demo_rows) < 5 or len(review_rows) < 6 or len(token_rows) != 1263:
        raise PreparationError("Demo/token artifact cardinality failed")
    if any(row["reviewer_approve"] or row["reviewer_notes"] for row in review_rows):
        raise PreparationError("Demo human-review fields are not blank")
    original_ranks = {item["identity"]["search_rank"] for item in records}
    if {row["search_rank"] for row in token_rows} != original_ranks:
        raise PreparationError("Token audit membership differs from benchmark")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--parser-jsonl", type=Path, default=DEFAULT_PARSER)
    parser.add_argument("--a1", type=Path, default=DEFAULT_A1)
    parser.add_argument("--a2", type=Path, default=DEFAULT_A2)
    parser.add_argument("--demo-sets", type=Path, default=DEFAULT_DEMO_SETS)
    parser.add_argument("--demo-review", type=Path, default=DEFAULT_DEMO_REVIEW)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--a1-report", type=Path, default=DEFAULT_A1_REPORT)
    parser.add_argument("--a2-report", type=Path, default=DEFAULT_A2_REPORT)
    parser.add_argument(
        "--preparation-report", type=Path, default=DEFAULT_PREPARATION_REPORT
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_preparation_environment()
    records = sorted(load_jsonl(args.benchmark), key=lambda item: item["identity"]["search_rank"])
    ontology = json.loads(args.ontology.read_text(encoding="utf-8"))
    validate_inputs(args.benchmark, args.ontology, records)
    label_ids, _ = ontology_ids(ontology)
    jurisdictions = Counter(item["identity"]["jurisdiction_country_raw"] for item in records)
    high = {name for name, count in jurisdictions.items() if count >= 20}
    if len(high) != EXPECTED_HIGH_JURISDICTION_N or sum(jurisdictions[name] for name in high) != EXPECTED_HIGH_JURISDICTION_CASES:
        raise PreparationError("A2 high-support universe changed")
    if derive_a2_partition(records, label_ids) != A2_TEST_JURISDICTIONS:
        raise PreparationError("Frozen A2 groups do not match deterministic balance search")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=TOKENIZER_REVISION, local_files_only=True
    )
    warnings = parser_warning_map(args.parser_jsonl)
    demo_rows, review_rows, demo_diagnostics = build_demo_artifacts(
        records, label_ids, warnings, tokenizer, high
    )
    a1_rows, a1_diagnostics = build_a1(records, label_ids)
    a2_rows, a2_diagnostics = build_a2(records, label_ids)
    token_rows, token_stats = token_audit(records, tokenizer)
    proxy_tokens = proxy_token_estimates(records, a1_rows, a2_rows, tokenizer)
    validate_outputs(records, label_ids, a1_rows, a2_rows, demo_rows, review_rows, token_rows)

    atomic_csv(args.a1, a1_rows)
    atomic_csv(args.a2, a2_rows)
    atomic_csv(args.demo_sets, demo_rows)
    atomic_csv(args.demo_review, review_rows)
    atomic_csv(args.token_audit, token_rows)
    atomic_text(args.a1_report, build_a1_report(a1_diagnostics, label_ids))
    atomic_text(args.a2_report, build_a2_report(a2_diagnostics, label_ids))
    atomic_text(
        args.preparation_report,
        build_preparation_report(
            demo_diagnostics,
            token_stats,
            proxy_tokens,
            a1_diagnostics,
            a2_diagnostics,
        ),
    )

    output_paths = (
        args.a1, args.a2, args.demo_sets, args.demo_review, args.token_audit,
        args.a1_report, args.a2_report, args.preparation_report,
    )
    print(json.dumps({
        "status": "PASS",
        "version": VERSION,
        "demo_status": "PROPOSED_NOT_FROZEN",
        "provisional_demo_set_id": PROVISIONAL_DEMO_SET_ID,
        "provisional_demo_ranks": sorted(PROVISIONAL_DEMO_RANKS),
        "a1": a1_diagnostics,
        "a2": a2_diagnostics,
        "modernbert_token_stats": token_stats,
        "llm_proxy_token_estimates": proxy_tokens,
        "frozen_spec_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (M1_CONFIG, M2_CONFIG, LLM_CONFIG, M3_PROMPT, M4_PROMPT)
        },
        "output_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in output_paths},
    }, ensure_ascii=False, indent=2, default=lambda value: dict(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
