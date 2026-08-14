#!/usr/bin/env python3
"""Prepare the second, fidelity-first SHERLOC demonstration review sheet.

This is a review-assistance stage only.  It reads the frozen 1,263-case
benchmark and the parser-v2 corpus, preserves prior human decisions, and writes
one combined review CSV plus a concise search report.  It never calls a model
or API, changes benchmark membership, selects a final demonstration bank, or
regenerates either provisional split.

The evidence mappings below are a manually inspected audit against sentence IDs
from ``sherloc_sentence_splitter_v1``.  CLEAR/POSSIBLE/UNCLEAR are screening
aids, not new ground-truth labels or final human adjudications.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


VERSION = "1.0.0"
PREPARATION_DATE = "2026-08-13"
PRIMARY_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_N = 1263
EXPECTED_HASHES = {
    "data/annotations/demo_bank_review.csv": (
        "1976a0ffc0ca609da231240886988bf2f4297bc94126c435a79900f6374ac1fe"
    ),
    "data/processed/sherloc_benchmark_v1.jsonl": (
        "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
    ),
    "data/processed/sherloc_benchmark_v1.csv": (
        "fd72c0ccd40b74130e2d6c76b989891e6b547b6c9576b2c66ff56c89bd76529b"
    ),
    "data/splits/a1_iid_split_v1.csv": (
        "4b8ea8020556811256f2e273d45a2346df56cf2de2df90a04f10c42ea0f33f5e"
    ),
    "data/splits/a2_jurisdiction_folds_v1.csv": (
        "69ff6ede16fa24ca872e848b79cd4552fd1600e330ac0f5f0639ba45a3318cde"
    ),
    "config/amp_ontology_v1.yaml": (
        "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
    ),
    "data/interim/sherloc_cases_raw.jsonl": (
        "ea0592fcb633a0eee55e5feacb02fc1ef119cfcbb0f594566b4da6420eb184df"
    ),
    "src/sherloc/05_build_benchmark.py": (
        "20605b272e5705aa09fd0f98937be8a21aea20359c9f9faff0248ef399bca047"
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_PARSER = REPO_ROOT / "data/interim/sherloc_cases_raw.jsonl"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_PRIOR_REVIEW = REPO_ROOT / "data/annotations/demo_bank_review.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/annotations/demo_bank_review_v2.csv"
DEFAULT_REPORT = REPO_ROOT / "docs/demo_candidate_search_v2_report.md"
EXPECTED_HASHES_PATHS = tuple(REPO_ROOT / relative for relative in EXPECTED_HASHES)

SCREEN_VALUES = {"CLEAR", "POSSIBLE", "UNCLEAR"}
GEOGRAPHIC_FORM_VALUES = {"Internal", "Transnational"}
FORM_SCREEN_VALUES = SCREEN_VALUES | {"NOT_APPLICABLE"}

RETAINED_EXISTING_RANKS = (
    146,
    154,
    326,
    465,
    625,
    764,
    955,
    1211,
    1293,
    1294,
    1487,
    1494,
    1545,
    1560,
)
EXCLUDED_SKIP_RANKS = (31, 1517)

# The full frozen benchmark was searched, without using provisional A1/A2
# roles.  Rank order here is review priority within each search stratum, not a
# frozen final-demo order.
NEW_RANKS_BY_GROUP: dict[str, tuple[int, ...]] = {
    "NEW_US": (1178, 1477, 1242, 692, 498),
    "NEW_MAJOR_JURISDICTION": (936, 391, 1343, 828, 641),
    "NEW_OTHER": (334, 761, 972, 338, 157),
}

# Best next-review set, deliberately not a final or frozen six-shot bank.  Rank
# 1487 is a previously reviewed Agree seed and supplies rare, clear purposes.
STRONGEST_INSPECTION_RANKS = (1178, 1477, 936, 391, 334, 761, 828, 1487)


class DemoReviewError(RuntimeError):
    """Raised when a frozen invariant or evidence audit check fails."""


def evidence(
    sentence_ids: Sequence[str], screen: str = "CLEAR"
) -> dict[str, Any]:
    return {"sentence_ids": list(sentence_ids), "screen": screen}


def audit(
    *,
    acts: dict[str, dict[str, Any]],
    means: dict[str, dict[str, Any]],
    purposes: dict[str, dict[str, Any]],
    form: dict[str, dict[str, Any]],
    caveat: str,
) -> dict[str, Any]:
    return {
        "ACT": acts,
        "MEANS": means,
        "PURPOSE": purposes,
        "FORM": form,
        "caveat": caveat,
    }


# Manually inspected sentence-level screening records.  Existing Agree/Hold
# rows are included so their prior decisions remain visible while the stricter
# v2 fidelity screen can identify which seeds need renewed scrutiny.
AUDITS: dict[int, dict[str, Any]] = {
    146: audit(
        acts={
            "ACT_RECRUITMENT": evidence([], "UNCLEAR"),
            "ACT_TRANSPORTATION": evidence(["S4"]),
            "ACT_HARBOURING": evidence(["S1", "S3"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S1", "S2"]),
            "MEANS_ABDUCTION": evidence(["S1"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S2"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S2", "S3"])},
        form={"Internal": evidence([], "UNCLEAR")},
        caveat=(
            "Retained prior Agree seed. Recruitment and Internal Form are not "
            "established by the supplied Fact Summary."
        ),
    ),
    154: audit(
        acts={
            "ACT_TRANSPORTATION": evidence(["S4", "S5"]),
            "ACT_HARBOURING": evidence(["S6"]),
            "ACT_RECEIPT": evidence(["S5"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S4", "S5", "S6"]),
            "MEANS_ABDUCTION": evidence(["S1", "S4", "S5"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S2", "S5", "S6"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S6"], "POSSIBLE")},
        form={},
        caveat=(
            "Retained prior Hold. Forced marriage and repeated rape are explicit, "
            "but the summary does not itself call the conduct sexual exploitation."
        ),
    ),
    326: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S3", "S4"], "POSSIBLE"),
            "ACT_HARBOURING": evidence(["S5", "S6"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S8", "S13", "S18"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(
                ["S1", "S2", "S3", "S4", "S6"]
            ),
        },
        purposes={
            "PURPOSE_FORCED_LABOUR_OR_SERVICES": evidence(["S6", "S7", "S25"]),
            "PURPOSE_SERVITUDE": evidence(["S25"]),
        },
        form={},
        caveat=(
            "Retained prior Hold. The aunt and uncle arranged arrival and assumed "
            "guardianship, but active recruitment is only possible from the summary."
        ),
    ),
    465: audit(
        acts={"ACT_RECRUITMENT": evidence(["S1", "S2", "S7"])},
        means={"MEANS_DECEPTION": evidence(["S1", "S2", "S3"], "POSSIBLE")},
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S2"], "POSSIBLE")},
        form={"Internal": evidence(["S1"], "POSSIBLE")},
        caveat=(
            "Retained prior Agree seed. The interrupted approach is clear, but the "
            "proposed work/parties only indirectly support deception and sexual purpose."
        ),
    ),
    625: audit(
        acts={
            "ACT_HARBOURING": evidence([], "UNCLEAR"),
            "ACT_RECEIPT": evidence(["S2"]),
        },
        means={"MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": evidence(["S2"])},
        purposes={"PURPOSE_OTHER": evidence(["S2"], "UNCLEAR")},
        form={"Internal": evidence([], "UNCLEAR")},
        caveat=(
            "Retained prior Agree seed, but the two-sentence summary does not describe "
            "harbouring, an exploitation purpose, or a complete internal route."
        ),
    ),
    764: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1"]),
            "ACT_TRANSPORTATION": evidence(["S2"]),
            "ACT_TRANSFER": evidence([], "UNCLEAR"),
        },
        means={
            "MEANS_FRAUD": evidence(["S2"]),
            "MEANS_DECEPTION": evidence(["S1"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S3"])},
        form={"Transnational": evidence(["S1", "S2"])},
        caveat=(
            "Retained prior Agree seed. Recruitment, movement, forged documents, and "
            "exploitation are explicit; no change of custody/control supports Transfer."
        ),
    ),
    955: audit(
        acts={
            "ACT_TRANSPORTATION": evidence(["S2"]),
            "ACT_TRANSFER": evidence(["S3"], "POSSIBLE"),
            "ACT_HARBOURING": evidence(["S2"]),
        },
        means={
            "MEANS_ABDUCTION": evidence(["S1"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1"]),
        },
        purposes={"PURPOSE_OTHER": evidence(["S3"])},
        form={"Internal": evidence([], "UNCLEAR")},
        caveat=(
            "Retained prior Agree seed. Illegal-adoption purpose is clear, but the "
            "planned sale was interrupted and Internal Form lacks an explicit route."
        ),
    ),
    1211: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S2"], "POSSIBLE"),
            "ACT_TRANSFER": evidence(["S3"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S4"]),
            "MEANS_ABDUCTION": evidence([], "UNCLEAR"),
            "MEANS_FRAUD": evidence(["S4"], "POSSIBLE"),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S4"])},
        form={"Internal": evidence(["S2", "S3"])},
        caveat=(
            "Retained prior Agree seed. The job advertisement does not identify a "
            "recruiting actor, no abduction is narrated, and Fraud is only asserted."
        ),
    ),
    1293: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1", "S2"]),
            "ACT_TRANSPORTATION": evidence(["S2"]),
            "ACT_TRANSFER": evidence(["S1", "S3"], "POSSIBLE"),
            "ACT_HARBOURING": evidence([], "UNCLEAR"),
        },
        means={
            "MEANS_FRAUD": evidence([], "UNCLEAR"),
            "MEANS_DECEPTION": evidence(["S2"]),
            "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": evidence(["S3"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S3"])},
        form={"Transnational": evidence(["S2"], "POSSIBLE")},
        caveat=(
            "Retained prior Hold. Transfer is only implied, Harbouring is absent, "
            "Fraud is not distinct from deception, and the origin country is unstated."
        ),
    ),
    1294: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1"], "POSSIBLE"),
            "ACT_TRANSPORTATION": evidence(["S5"]),
            "ACT_TRANSFER": evidence([], "UNCLEAR"),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S6", "S7"]),
            "MEANS_DECEPTION": evidence([], "UNCLEAR"),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S6", "S7"]),
        },
        purposes={
            "PURPOSE_FORCED_LABOUR_OR_SERVICES": evidence(["S6", "S7"]),
            "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": evidence([], "UNCLEAR"),
            "PURPOSE_SERVITUDE": evidence(["S6", "S7"]),
        },
        form={"Transnational": evidence(["S1", "S5"])},
        caveat=(
            "Retained prior Agree seed. The summary supports coercive domestic work "
            "and servitude but not Transfer, Deception, or a distinct slavery-like label."
        ),
    ),
    1487: audit(
        acts={
            "ACT_TRANSPORTATION": evidence(["S1", "S3"]),
            "ACT_HARBOURING": evidence(["S1"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S3"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S3"]),
            "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": evidence(["S3", "S4"]),
        },
        purposes={
            "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": evidence(["S3", "S4"]),
            "PURPOSE_OTHER": evidence(["S3"]),
        },
        form={"Transnational": evidence(["S1", "S3", "S4"])},
        caveat=(
            "Retained prior Agree seed. The sale of a 12-year-old as a forced bride "
            "provides unusually explicit support for both rare purpose labels."
        ),
    ),
    1494: audit(
        acts={
            "ACT_TRANSFER": evidence(["S4", "S5"]),
            "ACT_RECEIPT": evidence(["S5"]),
        },
        means={"MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": evidence(["S4", "S5"])},
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S3", "S4", "S5"])},
        form={"Internal": evidence([], "UNCLEAR")},
        caveat=(
            "Retained prior Agree seed. All AMP positives are explicit, but no place "
            "or within-country route supports Internal Form."
        ),
    ),
    1545: audit(
        acts={
            "ACT_HARBOURING": evidence(["S1", "S2"]),
            "ACT_RECEIPT": evidence([], "UNCLEAR"),
        },
        means={"MEANS_THREAT_FORCE_OR_COERCION": evidence(["S1"])},
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S2"])},
        form={"Internal": evidence([], "UNCLEAR")},
        caveat=(
            "Retained prior Agree seed. Detention and prostitution are explicit, but "
            "Receipt and Internal Form are not recoverable from the short summary."
        ),
    ),
    1560: audit(
        acts={"ACT_RECRUITMENT": evidence(["S3"])},
        means={"MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S3", "S4"])},
        purposes={"PURPOSE_FORCED_LABOUR_OR_SERVICES": evidence(["S3", "S4"])},
        form={"Internal": evidence([], "UNCLEAR")},
        caveat=(
            "Retained prior Agree seed. The fishing-work AMP chain is concise and "
            "recoverable; the narrative supplies no geography for Internal Form."
        ),
    ),
    1178: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S3"]),
            "ACT_TRANSPORTATION": evidence(["S3"]),
            "ACT_HARBOURING": evidence(["S1"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S2", "S4"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S4"]),
        },
        purposes={
            "PURPOSE_FORCED_LABOUR_OR_SERVICES": evidence(["S2"]),
            "PURPOSE_SERVITUDE": evidence(["S1"]),
        },
        form={"Transnational": evidence(["S1", "S3"], "POSSIBLE")},
        caveat=(
            "Exceptionally explicit despite 83 words; the complete domestic-servitude "
            "AMP chain is stated. Form is only POSSIBLE because the destination country "
            "is not named without outside knowledge about Detroit."
        ),
    ),
    1477: audit(
        acts={
            "ACT_TRANSPORTATION": evidence(["S2"]),
            "ACT_HARBOURING": evidence(["S3"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S4"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S2", "S3", "S4", "S5"]),
        },
        purposes={
            "PURPOSE_FORCED_LABOUR_OR_SERVICES": evidence(["S3", "S4", "S5"]),
            "PURPOSE_SERVITUDE": evidence(["S1", "S3", "S4", "S5"]),
        },
        form={"Transnational": evidence(["S2"])},
        caveat=(
            "The AMP chain is clear; S7-S9 are a modest procedural tail about related "
            "defendants but do not create target ambiguity."
        ),
    ),
    1242: audit(
        acts={"ACT_TRANSPORTATION": evidence(["S2"])},
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S4", "S5"]),
            "MEANS_DECEPTION": evidence(["S3", "S4"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S4", "S6", "S7"])},
        form={"Transnational": evidence(["S2", "S3", "S4"])},
        caveat=(
            "All positive Legacy labels are clear. Job promises and apartments may "
            "suggest additional Acts, but actor attribution is not explicit."
        ),
    ),
    692: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1", "S2", "S3"]),
            "ACT_HARBOURING": evidence(["S5"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(
                ["S4", "S5", "S7", "S8", "S9", "S10", "S14"]
            ),
            "MEANS_DECEPTION": evidence(["S2", "S7"]),
        },
        purposes={
            "PURPOSE_SEXUAL_EXPLOITATION": evidence(["S5", "S7", "S11", "S12", "S13"])
        },
        form={"Transnational": evidence(["S1", "S3", "S4"])},
        caveat=(
            "All positive labels are clear. The narrative also strongly suggests "
            "unreferenced Transportation and vulnerability, so reference completeness "
            "deserves human review."
        ),
    ),
    498: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1"]),
            "ACT_HARBOURING": evidence(["S1", "S5"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S2", "S3", "S5"]),
            "MEANS_DECEPTION": evidence(["S2"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S3", "S5"]),
        },
        purposes={
            "PURPOSE_FORCED_LABOUR_OR_SERVICES": evidence(["S1", "S3", "S4", "S5"]),
            "PURPOSE_SERVITUDE": evidence(["S2", "S5"]),
        },
        form={"Transnational": evidence(["S1", "S3"])},
        caveat=(
            "All displayed labels are clear. Travel arrangements may support an "
            "unreferenced Transportation label; the account uses indictment framing."
        ),
    ),
    936: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1"]),
            "ACT_TRANSPORTATION": evidence(["S1"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S2"]),
            "MEANS_DECEPTION": evidence(["S1"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S2"])},
        form={"Transnational": evidence(["S1"])},
        caveat=(
            "Every displayed AMP and geographic Form value is directly stated; the "
            "raw Organized Criminal Group value remains audit metadata only."
        ),
    ),
    391: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S4"]),
            "ACT_TRANSPORTATION": evidence(["S4"]),
            "ACT_HARBOURING": evidence(["S12", "S14"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(
                ["S5", "S6", "S7", "S11", "S12", "S13", "S14"]
            ),
            "MEANS_DECEPTION": evidence(["S4", "S11"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S3", "S9", "S10"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S4", "S7"])},
        form={"Transnational": evidence(["S4"])},
        caveat=(
            "All displayed AMP and geographic Form values are clear; controlled "
            "confinement supports Harbouring. Organized Criminal Group is audit-only."
        ),
    ),
    1343: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1", "S3"]),
            "ACT_TRANSPORTATION": evidence(["S1", "S4"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S5", "S6", "S7"]),
            "MEANS_DECEPTION": evidence(["S2"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S2", "S3", "S5"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S5", "S6"])},
        form={"Transnational": evidence(["S1", "S4"])},
        caveat=(
            "All displayed labels are clear; vulnerability is established contextually "
            "through age, poverty, lack of resources, and perpetrator control."
        ),
    ),
    641: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1"]),
            "ACT_TRANSPORTATION": evidence(["S3", "S4"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S4"]),
            "MEANS_DECEPTION": evidence(["S2"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S4"])},
        form={"Transnational": evidence(["S1", "S2", "S3", "S4"], "POSSIBLE")},
        caveat=(
            "All AMP positives are clear. Spain is explicit, but the origin country is "
            "not named, so Transnational Form is only POSSIBLE under the strict rule."
        ),
    ),
    828: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S3", "S7"]),
            "ACT_TRANSPORTATION": evidence(["S3", "S7"]),
            "ACT_TRANSFER": evidence(["S8"]),
            "ACT_HARBOURING": evidence(["S3", "S8", "S10"]),
            "ACT_RECEIPT": evidence(["S3", "S8", "S9"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S10", "S12"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S8", "S10", "S12"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S3", "S10"])},
        form={"Transnational": evidence(["S2", "S3", "S10"])},
        caveat=(
            "All displayed AMP and Form values are clear; S8 supplies an explicit "
            "change of controller for Transfer. Organized Criminal Group is audit-only."
        ),
    ),
    334: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1"]),
            "ACT_TRANSPORTATION": evidence(["S1"]),
            "ACT_HARBOURING": evidence(["S1"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S2"]),
            "MEANS_DECEPTION": evidence(["S1"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S2"]),
        },
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S2"])},
        form={"Transnational": evidence(["S1"])},
        caveat=(
            "Awkward English, but all Legacy labels and the border crossing are "
            "directly stated with no material support ambiguity."
        ),
    ),
    761: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1", "S3"]),
            "ACT_TRANSPORTATION": evidence(["S3", "S4", "S5"]),
            "ACT_HARBOURING": evidence(["S6"]),
        },
        means={"MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S2", "S3"])},
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S7"])},
        form={"Transnational": evidence(["S1", "S4", "S5"])},
        caveat=(
            "Every displayed label is clear. Debt repayment weakly cues possible "
            "unreferenced coercion but no threat, restraint, or inability to leave is stated."
        ),
    ),
    972: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S1", "S8"]),
            "ACT_TRANSPORTATION": evidence(["S1"]),
            "ACT_HARBOURING": evidence(["S1", "S2", "S8"]),
        },
        means={"MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S2"])},
        purposes={"PURPOSE_SEXUAL_EXPLOITATION": evidence(["S1", "S2", "S8"])},
        form={"Transnational": evidence(["S1", "S3", "S8"])},
        caveat=(
            "S3 records evidentiary disagreement about individual logistics, but S1 "
            "states the court's findings and S8 independently reinforces the conduct."
        ),
    ),
    338: audit(
        acts={
            "ACT_RECRUITMENT": evidence(["S3", "S7", "S18", "S29"]),
            "ACT_TRANSPORTATION": evidence(["S8", "S21", "S28"]),
            "ACT_TRANSFER": evidence(["S9", "S11", "S20", "S25", "S28"]),
            "ACT_HARBOURING": evidence(["S9", "S12", "S20", "S25", "S31", "S32"]),
            "ACT_RECEIPT": evidence(["S10", "S13", "S15", "S31"]),
        },
        means={
            "MEANS_THREAT_FORCE_OR_COERCION": evidence(["S31", "S32"]),
            "MEANS_DECEPTION": evidence(["S4", "S19", "S24", "S30"]),
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S4", "S19", "S24"]),
        },
        purposes={
            "PURPOSE_SEXUAL_EXPLOITATION": evidence(
                ["S3", "S9", "S20", "S21", "S25", "S26", "S31"]
            )
        },
        form={"Transnational": evidence(["S3", "S8", "S18", "S23"])},
        caveat=(
            "All positives are explicit and all five Acts are covered, but this is the "
            "longest new candidate (704 words). Organized Criminal Group is audit-only."
        ),
    ),
    157: audit(
        acts={"ACT_RECEIPT": evidence(["S3", "S4", "S5", "S6"])},
        means={
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": evidence(["S1", "S4", "S6"]),
            "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": evidence(["S4", "S7", "S8"]),
        },
        purposes={
            "PURPOSE_SEXUAL_EXPLOITATION": evidence(["S4", "S6", "S7", "S8", "S9", "S10"])
        },
        form={"Internal": evidence(["S1", "S3", "S5", "S6"], "POSSIBLE")},
        caveat=(
            "Severe child-sexual-abuse content. All displayed AMP labels are clear. "
            "Internal Form is only POSSIBLE because Georgia and Tbilisi are not explicitly "
            "linked without outside geography; the mother may support unreferenced Transfer."
        ),
    ),
}

SELECTION_REASONS = {
    1178: "Best compact U.S. domestic-servitude narrative; full AMP chain and route are explicit.",
    1477: "Representative U.S. domestic-servitude case in the preferred length range.",
    1242: "Clear U.S. transnational sexual-exploitation contrast with explicit job deception.",
    692: "Detailed U.S. sexual-exploitation narrative with unusually strong coercion evidence.",
    498: "Clear U.S. domestic forced-labour/servitude case in the preferred length range.",
    936: "Exceptionally concise Moldova-to-Russia case with every positive directly stated.",
    391: "Detailed Belgium case with coherent recruitment-to-exploitation facts.",
    1343: "Representative Sweden case with explicit age, vulnerability, control, and movement.",
    641: "Concise Brazil-origin candidate with fully clear AMP labels; Form caveat retained.",
    828: "Strong Colombia case with all five Acts, explicit controller handoff, and clear route.",
    334: "Cleanest geographically diverse reserve; full AMP chain and border crossing are explicit.",
    761: "Low-ambiguity Swiss reserve with explicit recruitment, route, accommodation, and purpose.",
    972: "Court-grounded Danish reserve whose positive references remain clear despite evidentiary detail.",
    338: "High-coverage Chile reserve: all five Acts are structurally and narratively explicit.",
    157: "Rare payment-for-control and Internal-Form reserve; sensitive-content caveat retained.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise DemoReviewError(f"Refusing to write an empty review sheet: {path}")
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


def load_sentence_tools() -> tuple[Any, Any, str]:
    path = REPO_ROOT / "src/sherloc/05_build_benchmark.py"
    spec = importlib.util.spec_from_file_location("sherloc_benchmark_builder_v1", path)
    if spec is None or spec.loader is None:
        raise DemoReviewError(f"Cannot import sentence splitter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_sentences_v1, module.numbered_sentences, module.SENTENCE_SPLITTER_VERSION


def validate_frozen_hashes() -> None:
    for relative, expected in EXPECTED_HASHES.items():
        path = REPO_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise DemoReviewError(
                f"Frozen/input artifact changed: {relative} {observed} != {expected}"
            )


def validate_input_paths(args: argparse.Namespace) -> None:
    expected_by_role = {
        "benchmark": EXPECTED_HASHES["data/processed/sherloc_benchmark_v1.jsonl"],
        "parser_jsonl": EXPECTED_HASHES["data/interim/sherloc_cases_raw.jsonl"],
        "ontology": EXPECTED_HASHES["config/amp_ontology_v1.yaml"],
        "prior_review": EXPECTED_HASHES["data/annotations/demo_bank_review.csv"],
    }
    protected = {path.resolve() for path in EXPECTED_HASHES_PATHS}
    for role, expected in expected_by_role.items():
        path = getattr(args, role)
        if not path.is_file() or sha256_file(path) != expected:
            raise DemoReviewError(
                f"Input override --{role.replace('_', '-')} is not the frozen audited artifact"
            )
        protected.add(path.resolve())
    output_paths = {args.output.resolve(), args.report.resolve()}
    if len(output_paths) != 2 or output_paths & protected:
        raise DemoReviewError("Output paths must be distinct and cannot overwrite protected inputs")


def load_prior_review(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16:
        raise DemoReviewError(f"Expected 16 prior review rows, found {len(rows)}")
    by_rank = {int(row["search_rank"]): row for row in rows}
    if tuple(sorted(rank for rank, row in by_rank.items() if row["reviewer_approve"] == "Skip")) != EXCLUDED_SKIP_RANKS:
        raise DemoReviewError("Prior Skip decisions changed")
    if set(RETAINED_EXISTING_RANKS) != {
        rank for rank, row in by_rank.items() if row["reviewer_approve"] in {"Agree", "Hold"}
    }:
        raise DemoReviewError("Prior Agree/Hold retention set changed")
    return by_rank


def parser_audit(
    path: Path,
    selected: set[int],
    benchmark_by_rank: dict[int, dict[str, Any]],
) -> tuple[dict[int, list[str]], dict[str, Any]]:
    warnings: dict[int, list[str]] = {}
    ekweremadu: dict[str, Any] | None = None
    found: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rank = int(row["provenance"]["search_rank"])
            if rank in selected:
                found.add(rank)
                fact = row.get("narrative", {}).get("fact_summary", {})
                legacy = row.get("legacy_keywords", {})
                core = legacy.get("core_fields", {})
                benchmark = benchmark_by_rank.get(rank)
                if benchmark is None:
                    raise DemoReviewError(
                        f"Selected parser rank {rank} is absent from the frozen benchmark"
                    )
                parser_benchmark_pairs = (
                    (
                        "canonical URL",
                        row.get("provenance", {}).get("canonical_url"),
                        benchmark["identity"]["canonical_url"],
                    ),
                    (
                        "case title",
                        row.get("case_identity", {}).get("title_raw"),
                        benchmark["identity"]["case_title_raw"],
                    ),
                    (
                        "English Fact Summary",
                        fact.get("english_text_raw"),
                        benchmark["text_input"]["english_fact_summary_raw"],
                    ),
                    (
                        "Legacy Acts",
                        core.get("acts", {}).get("values_raw", []),
                        benchmark["amp_targets"]["acts_raw"],
                    ),
                    (
                        "Legacy Means",
                        core.get("means", {}).get("values_raw", []),
                        benchmark["amp_targets"]["means_raw"],
                    ),
                    (
                        "Legacy Purposes",
                        core.get("exploitative_purposes", {}).get("values_raw", []),
                        benchmark["amp_targets"]["purposes_raw"],
                    ),
                    (
                        "Legacy Form",
                        core.get("form_of_trafficking", {}).get("values_raw", []),
                        benchmark["geographic_form"]["legacy_form_values_raw"],
                    ),
                )
                for field, parser_value, benchmark_value in parser_benchmark_pairs:
                    if parser_value != benchmark_value:
                        raise DemoReviewError(
                            f"Rank {rank} parser/benchmark mismatch in {field}"
                        )
                issues: list[str] = []
                if fact.get("status") != "FOUND" or not fact.get("english_text_raw"):
                    issues.append(f"FACT_SUMMARY_STATUS:{fact.get('status', 'MISSING')}")
                for source, entries in (
                    ("FACT_SUMMARY", fact.get("warnings", [])),
                    ("LEGACY_AMP", legacy.get("warnings", [])),
                ):
                    for item in entries:
                        if item.get("severity", "WARNING").upper() != "INFO":
                            issues.append(f"{source}:{item.get('code', 'UNKNOWN')}")
                for field in ("acts", "means", "exploitative_purposes"):
                    status = core.get(field, {}).get("status")
                    if status != "FOUND":
                        issues.append(f"LEGACY_{field.upper()}_STATUS:{status or 'MISSING'}")
                warnings[rank] = issues
            if row.get("case_identity", {}).get("title_raw") == (
                "Rex and Obinna Obeta, Ike and Beatrice Ekweremadu"
            ):
                core = row.get("legacy_keywords", {}).get("core_fields", {})
                ekweremadu = {
                    "search_rank": rank,
                    "canonical_url": row["provenance"]["canonical_url"],
                    "fact_summary_status": row["narrative"]["fact_summary"]["status"],
                    "legacy_acts_status": core.get("acts", {}).get("status"),
                    "legacy_acts": core.get("acts", {}).get("values_raw", []),
                    "legacy_means_status": core.get("means", {}).get("status"),
                    "legacy_means": core.get("means", {}).get("values_raw", []),
                    "legacy_purposes_status": core.get("exploitative_purposes", {}).get("status"),
                    "legacy_purposes": core.get("exploitative_purposes", {}).get("values_raw", []),
                    "legacy_form_status": core.get("form_of_trafficking", {}).get("status"),
                    "legacy_form": core.get("form_of_trafficking", {}).get("values_raw", []),
                    "sidebar_acts": row["trafficking_sidebar"]["fields"]["acts"]["values_raw"],
                    "sidebar_means": row["trafficking_sidebar"]["fields"]["means"]["values_raw"],
                    "sidebar_purposes": row["trafficking_sidebar"]["fields"]["exploitative_purposes"]["values_raw"],
                    "sidebar_form": row["trafficking_sidebar"]["fields"]["form_of_trafficking"]["values_raw"],
                }
    missing = selected - found
    if missing:
        raise DemoReviewError(f"Selected ranks absent from parser JSONL: {sorted(missing)}")
    if ekweremadu is None:
        raise DemoReviewError("Ekweremadu audit case not found in parser JSONL")
    return warnings, ekweremadu


def family_ids(record: dict[str, Any], family: str) -> list[str]:
    target = record["amp_targets"]
    return {
        "ACT": target["act_ontology_ids"],
        "MEANS": target["means_ontology_ids"],
        "PURPOSE": target["purpose_ontology_ids"],
    }[family]


def validate_audit(
    rank: int,
    record: dict[str, Any],
    sentences: Sequence[str],
    item: dict[str, Any],
) -> None:
    for family in ("ACT", "MEANS", "PURPOSE"):
        expected = family_ids(record, family)
        if list(item[family]) != expected:
            raise DemoReviewError(
                f"Rank {rank} {family} audit keys {list(item[family])} != reference {expected}"
            )
        for label_id, support in item[family].items():
            if support["screen"] not in SCREEN_VALUES:
                raise DemoReviewError(f"Rank {rank} invalid screen for {label_id}")
            if support["screen"] == "CLEAR" and not support["sentence_ids"]:
                raise DemoReviewError(f"Rank {rank} CLEAR label {label_id} lacks evidence")
            validate_sentence_ids(rank, support["sentence_ids"], sentences)

    raw_forms = record["geographic_form"]["legacy_form_values_raw"]
    expected_geo = [value for value in raw_forms if value in GEOGRAPHIC_FORM_VALUES]
    if list(item["FORM"]) != expected_geo:
        raise DemoReviewError(
            f"Rank {rank} Form audit keys {list(item['FORM'])} != geographic references {expected_geo}"
        )
    for raw_label, support in item["FORM"].items():
        if support["screen"] not in SCREEN_VALUES:
            raise DemoReviewError(f"Rank {rank} invalid Form screen for {raw_label}")
        if support["screen"] == "CLEAR" and not support["sentence_ids"]:
            raise DemoReviewError(f"Rank {rank} CLEAR Form {raw_label} lacks evidence")
        validate_sentence_ids(rank, support["sentence_ids"], sentences)


def validate_sentence_ids(rank: int, ids: Iterable[str], sentences: Sequence[str]) -> None:
    for sentence_id in ids:
        match = re.fullmatch(r"S([1-9]\d*)", sentence_id)
        if not match or int(match.group(1)) > len(sentences):
            raise DemoReviewError(
                f"Rank {rank} invalid evidence sentence {sentence_id}; has {len(sentences)} sentences"
            )


def evidence_map(item: dict[str, Any], family: str) -> dict[str, list[str]]:
    return {label: support["sentence_ids"] for label, support in item[family].items()}


def screen_map(item: dict[str, Any], family: str) -> dict[str, str]:
    return {label: support["screen"] for label, support in item[family].items()}


def form_screen(item: dict[str, Any]) -> str:
    screens = [support["screen"] for support in item["FORM"].values()]
    if not screens:
        return "NOT_APPLICABLE"
    for value in ("UNCLEAR", "POSSIBLE", "CLEAR"):
        if value in screens:
            return value
    raise DemoReviewError("Unreachable Form screen state")


def fidelity_summary(item: dict[str, Any]) -> str:
    all_support = [
        (label, support["screen"])
        for family in ("ACT", "MEANS", "PURPOSE")
        for label, support in item[family].items()
    ]
    clear = sum(screen == "CLEAR" for _, screen in all_support)
    non_clear = [f"{label}={screen}" for label, screen in all_support if screen != "CLEAR"]
    amp_text = f"{clear}/{len(all_support)} displayed Legacy AMP labels screen CLEAR."
    if non_clear:
        amp_text += " Non-CLEAR: " + ", ".join(non_clear) + "."
    return f"{amp_text} Geographic Form: {form_screen(item)}. {item['caveat']}"


def build_rows(
    records: Sequence[dict[str, Any]],
    prior: dict[int, dict[str, str]],
    parser_warnings: dict[int, list[str]],
    ontology: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], set[str]]:
    split_sentences, numbered_sentences, splitter_version = load_sentence_tools()
    by_rank = {int(row["identity"]["search_rank"]): row for row in records}
    if len(records) != EXPECTED_N or len(by_rank) != EXPECTED_N:
        raise DemoReviewError(f"Expected {EXPECTED_N} unique benchmark rows")
    if any(row["primary_cohort_id"] != PRIMARY_COHORT_ID for row in records):
        raise DemoReviewError("Frozen primary cohort ID mismatch")

    ontology_ids = [
        label["id"]
        for family in ("ACT", "MEANS", "PURPOSE")
        for label in ontology["families"][family]
    ]
    if len(ontology_ids) != 17:
        raise DemoReviewError("Frozen ontology is not 5/6/6")
    frequencies = Counter(
        label
        for record in records
        for label in (
            record["amp_targets"]["act_ontology_ids"]
            + record["amp_targets"]["means_ontology_ids"]
            + record["amp_targets"]["purpose_ontology_ids"]
        )
    )
    rare_ids = {label for label, count in frequencies.items() if count / EXPECTED_N < 0.10}

    ordered: list[tuple[str, int]] = [
        ("EXISTING_REVIEWED", rank) for rank in RETAINED_EXISTING_RANKS
    ]
    for group, ranks in NEW_RANKS_BY_GROUP.items():
        ordered.extend((group, rank) for rank in ranks)
    selected = {rank for _, rank in ordered}
    if selected != set(AUDITS):
        raise DemoReviewError(
            f"Audit/selection mismatch: missing {sorted(selected-set(AUDITS))}; extra {sorted(set(AUDITS)-selected)}"
        )
    if selected & set(EXCLUDED_SKIP_RANKS):
        raise DemoReviewError("A prior Skip case was selected")

    output: list[dict[str, Any]] = []
    for display_order, (group, rank) in enumerate(ordered, start=1):
        record = by_rank.get(rank)
        if record is None:
            raise DemoReviewError(f"Selected rank {rank} is outside the frozen cohort")
        text = record["text_input"]["english_fact_summary_raw"]
        sentences = split_sentences(text)
        item = AUDITS[rank]
        validate_audit(rank, record, sentences, item)
        all_support = [
            support
            for family in ("ACT", "MEANS", "PURPOSE")
            for support in item[family].values()
        ]
        all_ids = [
            *record["amp_targets"]["act_ontology_ids"],
            *record["amp_targets"]["means_ontology_ids"],
            *record["amp_targets"]["purpose_ontology_ids"],
        ]
        prior_row = prior.get(rank, {}) if group == "EXISTING_REVIEWED" else {}
        form_status = form_screen(item)
        warnings = parser_warnings.get(rank, [])
        output.append(
            {
                "candidate_display_order": display_order,
                "search_rank": rank,
                "case_title": record["identity"]["case_title_raw"],
                "unodc_case_number": record["identity"].get("unodc_case_number") or "",
                "jurisdiction": record["identity"]["jurisdiction_country_raw"],
                "canonical_url": record["identity"]["canonical_url"],
                "english_fact_summary": text,
                "word_count": record["text_input"]["word_count"],
                "fact_summary_numbered": numbered_sentences(sentences),
                "sentence_splitter_version": splitter_version,
                "legacy_acts_reference_json": canonical_json(record["amp_targets"]["acts_raw"]),
                "legacy_means_reference_json": canonical_json(record["amp_targets"]["means_raw"]),
                "legacy_purposes_reference_json": canonical_json(record["amp_targets"]["purposes_raw"]),
                "amp_ontology_ids_json": canonical_json(all_ids),
                "legacy_geographic_form_reference_json": canonical_json(
                    record["geographic_form"]["legacy_form_values_raw"]
                ),
                "act_evidence_by_label_json": canonical_json(evidence_map(item, "ACT")),
                "act_support_screen_by_label_json": canonical_json(screen_map(item, "ACT")),
                "means_evidence_by_label_json": canonical_json(evidence_map(item, "MEANS")),
                "means_support_screen_by_label_json": canonical_json(screen_map(item, "MEANS")),
                "purpose_evidence_by_label_json": canonical_json(evidence_map(item, "PURPOSE")),
                "purpose_support_screen_by_label_json": canonical_json(screen_map(item, "PURPOSE")),
                "form_evidence_sentence_ids_json": canonical_json(evidence_map(item, "FORM")),
                "form_support_screen_by_label_json": canonical_json(screen_map(item, "FORM")),
                "form_support_screen": form_status,
                "candidate_group": group,
                "selection_reason": (
                    SELECTION_REASONS[rank]
                    if group != "EXISTING_REVIEWED"
                    else f"Retained prior {prior_row['reviewer_approve']} decision; re-screened under v2."
                ),
                "fidelity_summary": fidelity_summary(item),
                "amp_labels_clear_count": sum(
                    support["screen"] == "CLEAR" for support in all_support
                ),
                "amp_labels_total_count": len(all_support),
                "all_amp_reference_labels_clear": int(
                    all(support["screen"] == "CLEAR" for support in all_support)
                ),
                "geographic_form_clear": int(form_status == "CLEAR"),
                "rare_label_coverage": canonical_json(
                    [label for label in all_ids if label in rare_ids]
                ),
                "any_parser_data_warning": int(bool(warnings)),
                "parser_data_warnings": "|".join(warnings),
                "previous_reviewer_approve": prior_row.get("reviewer_approve", ""),
                "previous_reviewer_notes": prior_row.get("reviewer_notes", ""),
                "reviewer_approve_v2": "",
                "reviewer_notes_v2": "",
            }
        )

    counts = Counter(row["candidate_group"] for row in output)
    expected_counts = {
        "EXISTING_REVIEWED": 14,
        "NEW_US": 5,
        "NEW_MAJOR_JURISDICTION": 5,
        "NEW_OTHER": 5,
    }
    if dict(counts) != expected_counts:
        raise DemoReviewError(f"Unexpected candidate counts: {dict(counts)}")
    new_rows = [row for row in output if row["candidate_group"] != "EXISTING_REVIEWED"]
    if len(new_rows) != 15 or not all(row["all_amp_reference_labels_clear"] for row in new_rows):
        raise DemoReviewError("New candidate fidelity/count invariant failed")
    if sum(row["geographic_form_clear"] for row in new_rows) != 12:
        raise DemoReviewError("Expected 12 new candidates with CLEAR Geographic Form")
    for row in output:
        if row["candidate_group"] == "EXISTING_REVIEWED":
            old = prior[int(row["search_rank"])]
            if row["previous_reviewer_approve"] != old["reviewer_approve"]:
                raise DemoReviewError("Prior reviewer decision was not preserved")
            if row["previous_reviewer_notes"] != old["reviewer_notes"]:
                raise DemoReviewError("Prior reviewer notes were not preserved")
        elif row["previous_reviewer_approve"] or row["previous_reviewer_notes"]:
            raise DemoReviewError("New candidate has a prior human decision")
    return output, dict(counts), rare_ids


def label_coverage(rows: Sequence[dict[str, Any]], ranks: set[int]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        if int(row["search_rank"]) in ranks:
            result.update(json.loads(row["amp_ontology_ids_json"]))
    return result


def render_report(
    rows: Sequence[dict[str, Any]],
    counts: dict[str, int],
    ontology: dict[str, Any],
    ekweremadu: dict[str, Any],
) -> str:
    all_ids = [
        label["id"]
        for family in ("ACT", "MEANS", "PURPOSE")
        for label in ontology["families"][family]
    ]
    new_rows = [row for row in rows if row["candidate_group"] != "EXISTING_REVIEWED"]
    high_fidelity = [row for row in rows if row["all_amp_reference_labels_clear"]]
    high_fidelity_coverage = label_coverage(
        rows, {int(row["search_rank"]) for row in high_fidelity}
    )
    strongest_coverage = label_coverage(rows, set(STRONGEST_INSPECTION_RANKS))
    missing = [label for label in all_ids if label not in high_fidelity_coverage]

    group_labels = {
        "NEW_US": "United States",
        "NEW_MAJOR_JURISDICTION": "Other high-support jurisdictions",
        "NEW_OTHER": "Other/reserve jurisdictions",
    }
    lines = [
        "# Demonstration Candidate Search v2",
        "",
        f"Prepared {PREPARATION_DATE} with `07_prepare_demo_review_v2.py` v{VERSION}. "
        "This is review assistance only: no final six were selected, no model/API was "
        "run, and the frozen benchmark and provisional A1/A2 splits were not changed.",
        "",
        "## Result",
        "",
        f"The combined sheet retains **{counts['EXISTING_REVIEWED']}** prior Agree/Hold "
        f"cases and adds **{len(new_rows)}** new candidates: **{counts['NEW_US']} U.S.**, "
        f"**{counts['NEW_MAJOR_JURISDICTION']} other high-support**, and "
        f"**{counts['NEW_OTHER']} other/reserve**. The two prior Skip cases remain excluded.",
        "",
    ]
    for group in ("NEW_US", "NEW_MAJOR_JURISDICTION", "NEW_OTHER"):
        lines.extend([f"### {group_labels[group]}", ""])
        for row in new_rows:
            if row["candidate_group"] != group:
                continue
            lines.append(
                f"- Rank {row['search_rank']}, **{row['case_title']}** "
                f"({row['jurisdiction']}; {row['word_count']} words): "
                f"AMP {row['amp_labels_clear_count']}/{row['amp_labels_total_count']} CLEAR; "
                f"Form {row['form_support_screen']}."
            )
        lines.append("")

    clear_form = [int(row["search_rank"]) for row in new_rows if row["geographic_form_clear"]]
    lines.extend(
        [
            "## Fidelity and next review",
            "",
            f"All **{len(new_rows)}/{len(new_rows)} new cases** have every displayed "
            "Legacy AMP label screened `CLEAR`. **12/15** have `CLEAR` Geographic Form. "
            "Ranks 641 and 1178 are `POSSIBLE` because only one endpoint's country is "
            "explicit; rank 157 is also `POSSIBLE` because Georgia and Tbilisi are not "
            "explicitly linked without outside geographic knowledge. "
            f"CLEAR-Form ranks: {', '.join(map(str, clear_form))}.",
            "",
            "Recommended strongest cases for the next human inspection are ranks "
            "**1178, 1477, 936, 391, 334, 761, 828**, plus retained Agree seed **1487**. "
            "This eight-case inspection set covers "
            f"**{len(strongest_coverage)}/17 AMP labels**. Rank 1178 is the best compact "
            "U.S. option, with ranks 1477 and 1242 as strong U.S. alternatives. Ranks "
            "334 and 761 are the cleanest reserve options; rank 338 is the high-coverage "
            "reserve. This is not a final bank.",
            "",
            f"Across every screened case whose displayed AMP references are all CLEAR, "
            f"the attainable union is **{len(high_fidelity_coverage)}/17**. Missing labels "
            f"are `{ '`, `'.join(missing) }`. Thus 17/17 does **not** appear achievable "
            "within this reviewed pool without accepting an ambiguous demonstration. "
            "The strict search intentionally did not force rare-label coverage.",
            "",
            "The near-miss Argentina rank 913 was not added: its summary calls victims "
            "vulnerable and notes irregular stay, but does not show clearly how that "
            "vulnerability was used or constrained alternatives.",
            "",
            "## Ekweremadu audit and blockers",
            "",
            "`Rex and Obinna Obeta, Ike and Beatrice Ekweremadu` is **not eligible**. "
            f"It is downloaded/parser rank {ekweremadu['search_rank']} with a usable English "
            "Fact Summary, but it is absent from the frozen 1,263-case cohort because "
            f"Legacy Acts and Means are `{ekweremadu['legacy_acts_status']}` / "
            f"`{ekweremadu['legacy_means_status']}`. Its only Legacy AMP value is "
            "`Removal of organs`, which the narrative clearly supports; Legacy Geographic "
            f"Form is `{ekweremadu['legacy_form_status']}`. Recruitment, vulnerability, and "
            "transnational values exist only in the trafficking sidebar and cannot be "
            "backfilled into the Legacy reference.",
            "",
            "The remaining blockers are human/HT-professional adjudication of positive-label "
            "fidelity, possible source under-labeling, sensitive examples, and later "
            "fold-specific jurisdiction substitutions. Those decisions belong to the "
            "separate finalization step before split regeneration.",
            "",
            "## Reproducibility guardrails",
            "",
            "- Corpus search source: the full frozen N=1,263 benchmark, not provisional split roles.",
            "- Candidate generation: full-cohort filtering followed by a manually curated sentence-level fidelity audit; the frozen audit table rebuild is deterministic, but the qualitative ranking is not model-derived.",
            "- Evidence IDs: `sherloc_sentence_splitter_v1`; all IDs are range-validated.",
            "- Reference source: Legacy Keywords only; sidebar values never define demo targets.",
            "- `Organized Criminal Group` remains in raw Form JSON where supplied but is not a geographic target.",
            "- `rare_label_coverage` uses a pre-screen frequency threshold below 10% of the frozen cohort.",
            "- Candidate screening does not alter silver-reference benchmark membership.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--parser-jsonl", type=Path, default=DEFAULT_PARSER)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--prior-review", type=Path, default=DEFAULT_PRIOR_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_frozen_hashes()
    validate_input_paths(args)
    records = load_jsonl(args.benchmark)
    benchmark_by_rank = {
        int(record["identity"]["search_rank"]): record for record in records
    }
    ontology = json.loads(args.ontology.read_text(encoding="utf-8"))
    prior = load_prior_review(args.prior_review)
    selected = set(RETAINED_EXISTING_RANKS)
    selected.update(rank for ranks in NEW_RANKS_BY_GROUP.values() for rank in ranks)
    warnings, ekweremadu = parser_audit(
        args.parser_jsonl, selected, benchmark_by_rank
    )
    rows, counts, _ = build_rows(records, prior, warnings, ontology)
    report = render_report(rows, counts, ontology, ekweremadu)
    atomic_csv(args.output, rows)
    atomic_text(args.report, report)
    validate_frozen_hashes()
    print(
        canonical_json(
            {
                "output": str(args.output),
                "rows": len(rows),
                "new_candidates": sum(
                    row["candidate_group"] != "EXISTING_REVIEWED" for row in rows
                ),
                "new_all_amp_clear": sum(
                    row["candidate_group"] != "EXISTING_REVIEWED"
                    and row["all_amp_reference_labels_clear"]
                    for row in rows
                ),
                "new_clear_form": sum(
                    row["candidate_group"] != "EXISTING_REVIEWED"
                    and row["geographic_form_clear"]
                    for row in rows
                ),
                "report": str(args.report),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
