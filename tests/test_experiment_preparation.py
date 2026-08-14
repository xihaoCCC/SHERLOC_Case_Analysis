"""Regression tests for frozen Evaluation-A preparation artifacts.

These tests deliberately use only the Python standard library.  They validate
artifact integrity without rerunning tokenization or split generation, keeping
the normal test suite fast and independent of the experiment environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
ONTOLOGY_PATH = REPO_ROOT / "config/amp_ontology_v1.yaml"
A1_PATH = REPO_ROOT / "data/splits/a1_iid_split_v1.csv"
A2_PATH = REPO_ROOT / "data/splits/a2_jurisdiction_folds_v1.csv"
TOKEN_AUDIT_PATH = REPO_ROOT / "outputs/tables/modernbert_token_length_audit.csv"
DEMO_CANDIDATES_PATH = REPO_ROOT / "outputs/tables/demo_candidate_sets.csv"
DEMO_REVIEW_PATH = REPO_ROOT / "data/annotations/demo_bank_review.csv"
DEMO_REVIEW_V2_PATH = REPO_ROOT / "data/annotations/demo_bank_review_v2.csv"
M1_CONFIG_PATH = REPO_ROOT / "config/experiments/m1_tfidf_logreg_v1.yaml"
M2_CONFIG_PATH = REPO_ROOT / "config/experiments/m2_modernbert_v1.yaml"
LLM_CONFIG_PATH = REPO_ROOT / "config/experiments/llm_extraction_v1.yaml"

EXPECTED_N = 1263
EXPECTED_BENCHMARK_SHA256 = (
    "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
)
EXPECTED_MEMBERSHIP_SHA256 = (
    "097ce2027171ebc9cac5ad6dfdbf6e854729f81a8ede78e8401086fe5d5ed48c"
)
EXPECTED_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_TOKENIZER_MODEL = "answerdotai/ModernBERT-base"
EXPECTED_TOKENIZER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TOKEN_THRESHOLDS = (512, 1024, 1536, 2048, 3072, 4096, 8192)

EXPECTED_LABEL_TOTALS = {
    "ACT_RECRUITMENT": 1025,
    "ACT_TRANSPORTATION": 825,
    "ACT_TRANSFER": 489,
    "ACT_HARBOURING": 608,
    "ACT_RECEIPT": 352,
    "MEANS_THREAT_FORCE_OR_COERCION": 664,
    "MEANS_ABDUCTION": 115,
    "MEANS_FRAUD": 304,
    "MEANS_DECEPTION": 673,
    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": 769,
    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": 141,
    "PURPOSE_SEXUAL_EXPLOITATION": 1007,
    "PURPOSE_FORCED_LABOUR_OR_SERVICES": 249,
    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": 64,
    "PURPOSE_SERVITUDE": 72,
    "PURPOSE_REMOVAL_OF_ORGANS": 10,
    "PURPOSE_OTHER": 66,
}

EXPECTED_HIGH_JURISDICTIONS = {
    "Argentina": 75,
    "Australia": 20,
    "Belgium": 24,
    "Brazil": 103,
    "Canada": 22,
    "Colombia": 38,
    "Czechia": 30,
    "India": 26,
    "Philippines": 73,
    "Poland": 21,
    "Republic of Moldova": 57,
    "Romania": 52,
    "Serbia": 36,
    "Slovakia": 48,
    "Sweden": 31,
    "Ukraine": 21,
    "United Kingdom of Great Britain and Northern Ireland": 24,
    "United States of America": 160,
}

EXPECTED_A2_HELDOUT = {
    1: {
        "Argentina",
        "Australia",
        "Republic of Moldova",
        "Romania",
        "Serbia",
        "Slovakia",
    },
    2: {"Belgium", "Brazil", "Czechia", "India", "Philippines", "Sweden"},
    3: {
        "Canada",
        "Colombia",
        "Poland",
        "Ukraine",
        "United Kingdom of Great Britain and Northern Ireland",
        "United States of America",
    },
}
EXPECTED_A1_MEMBERSHIP_SHA256 = (
    "4360fe5100bee298ff3593446e554926f65bb06720a517e732219935962ee8fb"
)
EXPECTED_A2_MEMBERSHIP_SHA256 = {
    1: "6b1b634024fe1800900b5835df0f35172765092b2d5f23ac9271e4a60c3325f2",
    2: "8e1464774d52ced1cbd394feb551f8e4fc83085973ad313527bff39f7a9a73cf",
    3: "3fb90a08c3fed6f15312a27437f4e8ec395e4c6687c1b346e4f70bc3cd94baae",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def benchmark_label_ids(record: dict[str, Any]) -> set[str]:
    targets = record["amp_targets"]
    return set(
        targets["act_ontology_ids"]
        + targets["means_ontology_ids"]
        + targets["purpose_ontology_ids"]
    )


def membership_digest(pairs: list[tuple[int, str]]) -> str:
    payload = "".join(f"{rank}\t{url}\n" for rank, url in sorted(pairs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExperimentPreparationArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            BENCHMARK_PATH,
            ONTOLOGY_PATH,
            A1_PATH,
            A2_PATH,
            TOKEN_AUDIT_PATH,
            DEMO_CANDIDATES_PATH,
            DEMO_REVIEW_PATH,
            DEMO_REVIEW_V2_PATH,
            M1_CONFIG_PATH,
            M2_CONFIG_PATH,
            LLM_CONFIG_PATH,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise AssertionError(f"Missing experiment-preparation artifacts: {missing}")

        cls.benchmark = load_jsonl(BENCHMARK_PATH)
        cls.by_rank = {
            int(record["identity"]["search_rank"]): record
            for record in cls.benchmark
        }
        cls.ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
        cls.label_ids = [
            item["id"]
            for family in ("ACT", "MEANS", "PURPOSE")
            for item in cls.ontology["families"][family]
        ]
        cls.a1 = load_csv(A1_PATH)
        cls.a2 = load_csv(A2_PATH)
        cls.token_rows = load_csv(TOKEN_AUDIT_PATH)
        cls.demo_sets = load_csv(DEMO_CANDIDATES_PATH)
        cls.demo_review = load_csv(DEMO_REVIEW_PATH)
        cls.demo_review_v2 = load_csv(DEMO_REVIEW_V2_PATH)
        cls.m1_config = json.loads(M1_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.m2_config = json.loads(M2_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.llm_config = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_configs_remain_preparation_only_and_provisional(self) -> None:
        for config in (self.m1_config, self.m2_config, self.llm_config):
            self.assertFalse(config["scope_guard"].get("train_now", False))
            self.assertFalse(config["scope_guard"].get("predict_now", False))
            self.assertFalse(config["scope_guard"].get("evaluate_now", False))
            self.assertFalse(config["scope_guard"].get("call_api_now", False))
            self.assertFalse(config["scope_guard"].get("create_predictions_now", False))
        for config in (self.m1_config, self.m2_config):
            reproducibility = config["reproducibility"]
            self.assertEqual(
                reproducibility["split_status"],
                "PROVISIONAL_PENDING_DEMO_APPROVAL",
            )
            self.assertEqual(
                reproducibility["provisional_demo_set_id"],
                "demo-bank-proposal-set-01-v1",
            )
            self.assertTrue(reproducibility["regenerate_if_demo_membership_changes"])
        bank = self.llm_config["demonstration_bank"]
        self.assertEqual(bank["current_status"], "HUMAN_APPROVAL_REQUIRED_NOT_FROZEN")
        self.assertTrue(
            bank["provisional_split_anchor"][
                "not_valid_for_m4_requests_until_human_approved"
            ]
        )

    def test_frozen_benchmark_hash_membership_and_label_totals(self) -> None:
        self.assertEqual(sha256_file(BENCHMARK_PATH), EXPECTED_BENCHMARK_SHA256)
        self.assertEqual(sha256_file(ONTOLOGY_PATH), EXPECTED_ONTOLOGY_SHA256)
        self.assertEqual(len(self.benchmark), EXPECTED_N)
        self.assertEqual(len(self.by_rank), EXPECTED_N)
        self.assertEqual(len(self.label_ids), 17)
        self.assertEqual(self.label_ids, list(EXPECTED_LABEL_TOTALS))

        urls = [record["identity"]["canonical_url"] for record in self.benchmark]
        self.assertEqual(len(set(urls)), EXPECTED_N)
        self.assertEqual(
            {record["primary_cohort_id"] for record in self.benchmark},
            {EXPECTED_COHORT_ID},
        )
        self.assertEqual(
            membership_digest(
                [
                    (
                        int(record["identity"]["search_rank"]),
                        record["identity"]["canonical_url"],
                    )
                    for record in self.benchmark
                ]
            ),
            EXPECTED_MEMBERSHIP_SHA256,
        )

        totals = Counter()
        for record in self.benchmark:
            totals.update(benchmark_label_ids(record))
        self.assertEqual({label: totals[label] for label in self.label_ids}, EXPECTED_LABEL_TOTALS)
        self.assertEqual(
            sum(
                int(record["geographic_form"]["geographic_form_eligible"])
                for record in self.benchmark
            ),
            1156,
        )

    def test_high_support_jurisdiction_universe_is_frozen(self) -> None:
        counts = Counter(
            record["identity"]["jurisdiction_country_raw"]
            for record in self.benchmark
        )
        high = {name: count for name, count in counts.items() if count >= 20}
        self.assertEqual(high, EXPECTED_HIGH_JURISDICTIONS)
        self.assertEqual(len(high), 18)
        self.assertEqual(sum(high.values()), 861)
        self.assertEqual(sum(count for name, count in counts.items() if name not in high), 402)

    def test_a1_membership_roles_labels_and_demo_rules(self) -> None:
        self.assertEqual(len(self.a1), EXPECTED_N)
        self.assertEqual(
            Counter(row["split"] for row in self.a1),
            {"TRAIN": 878, "VALIDATION": 126, "TEST": 253, "DEMO": 6},
        )
        self.assertEqual({int(row["search_rank"]) for row in self.a1}, set(self.by_rank))
        self.assertEqual(len({row["canonical_url"] for row in self.a1}), EXPECTED_N)
        a1_digest = hashlib.sha256(
            "".join(
                f"{row['search_rank']}\t{row['canonical_url']}\t{row['split']}\n"
                for row in self.a1
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(a1_digest, EXPECTED_A1_MEMBERSHIP_SHA256)

        top_demo_set = min(self.demo_sets, key=lambda row: int(row["candidate_set_rank"]))
        top_demo_id = top_demo_set["candidate_set_id"]
        proposed_demo_ranks = set(json.loads(top_demo_set["search_ranks_json"]))
        actual_demo_ranks = {
            int(row["search_rank"]) for row in self.a1 if row["split"] == "DEMO"
        }
        self.assertEqual(actual_demo_ranks, proposed_demo_ranks)
        self.assertEqual(len(actual_demo_ranks), 6)
        self.assertTrue(
            actual_demo_ranks.isdisjoint(
                int(row["search_rank"])
                for row in self.a1
                if row["split"] in {"VALIDATION", "TEST"}
            )
        )

        high = set(EXPECTED_HIGH_JURISDICTIONS)
        for row in self.a1:
            rank = int(row["search_rank"])
            record = self.by_rank[rank]
            selected = benchmark_label_ids(record)
            self.assertEqual(row["canonical_url"], record["identity"]["canonical_url"])
            self.assertEqual(row["jurisdiction"], record["identity"]["jurisdiction_country_raw"])
            self.assertEqual(int(row["effective_supervised_train"]), int(row["split"] in {"TRAIN", "DEMO"}))
            self.assertEqual(row["demo_set_id"], top_demo_id)
            self.assertEqual(row["split_status"], "PROVISIONAL_PENDING_DEMO_APPROVAL")
            self.assertEqual(int(row["amp_positive_label_count"]), len(selected))
            for label in self.label_ids:
                self.assertEqual(int(row[label]), int(label in selected), (rank, label))

            form = record["geographic_form"]
            for column, source in (
                ("geographic_form_eligible", "geographic_form_eligible"),
                ("geographic_form_internal", "geographic_form_internal"),
                ("geographic_form_transnational", "geographic_form_transnational"),
            ):
                self.assertEqual(int(row[column]), int(form[source]), (rank, column))
            if row["split"] == "DEMO":
                self.assertNotIn(row["jurisdiction"], high)

        for label, expected in EXPECTED_LABEL_TOTALS.items():
            self.assertEqual(sum(int(row[label]) for row in self.a1), expected)
            self.assertGreater(sum(int(row[label]) for row in self.a1 if row["split"] == "VALIDATION"), 0)
            self.assertGreater(sum(int(row[label]) for row in self.a1 if row["split"] == "TEST"), 0)

        organ = "PURPOSE_REMOVAL_OF_ORGANS"
        organ_allocation = {
            role: sum(int(row[organ]) for row in self.a1 if row["split"] == role)
            for role in ("TRAIN", "VALIDATION", "TEST", "DEMO")
        }
        self.assertEqual(
            organ_allocation,
            {"TRAIN": 6, "VALIDATION": 1, "TEST": 2, "DEMO": 1},
        )

    def test_a2_fold_membership_and_jurisdiction_leakage(self) -> None:
        self.assertEqual(len(self.a2), EXPECTED_N * 3)
        self.assertEqual({int(row["fold_id"]) for row in self.a2}, {1, 2, 3})
        expected_role_counts = {
            1: {"TRAIN": 871, "VALIDATION": 98, "TEST": 288, "DEMO": 6},
            2: {"TRAIN": 872, "VALIDATION": 98, "TEST": 287, "DEMO": 6},
            3: {"TRAIN": 873, "VALIDATION": 98, "TEST": 286, "DEMO": 6},
        }
        top_demo_set = min(self.demo_sets, key=lambda row: int(row["candidate_set_rank"]))
        top_demo_id = top_demo_set["candidate_set_id"]
        demo_ranks = set(json.loads(top_demo_set["search_ranks_json"]))

        heldout_union: set[str] = set()
        total_test_assignments = 0
        for fold in (1, 2, 3):
            rows = [row for row in self.a2 if int(row["fold_id"]) == fold]
            self.assertEqual(len(rows), EXPECTED_N)
            self.assertEqual({int(row["search_rank"]) for row in rows}, set(self.by_rank))
            self.assertEqual(
                Counter(row["role"] for row in rows), expected_role_counts[fold]
            )
            a2_digest = hashlib.sha256(
                "".join(
                    f"{rank}\t{role}\n"
                    for rank, role in sorted(
                        (int(row["search_rank"]), row["role"]) for row in rows
                    )
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(a2_digest, EXPECTED_A2_MEMBERSHIP_SHA256[fold])

            heldout = EXPECTED_A2_HELDOUT[fold]
            self.assertEqual(len(heldout), 6)
            self.assertTrue(heldout_union.isdisjoint(heldout))
            heldout_union.update(heldout)
            test_jurisdictions = {row["jurisdiction"] for row in rows if row["role"] == "TEST"}
            self.assertEqual(test_jurisdictions, heldout)

            fold_demo_ranks = {
                int(row["search_rank"]) for row in rows if row["role"] == "DEMO"
            }
            self.assertEqual(fold_demo_ranks, demo_ranks)
            total_test_assignments += sum(row["role"] == "TEST" for row in rows)

            for row in rows:
                rank = int(row["search_rank"])
                record = self.by_rank[rank]
                selected = benchmark_label_ids(record)
                is_heldout = row["jurisdiction"] in heldout
                self.assertEqual(row["canonical_url"], record["identity"]["canonical_url"])
                self.assertEqual(row["jurisdiction"], record["identity"]["jurisdiction_country_raw"])
                self.assertEqual(int(row["heldout_jurisdiction"]), int(is_heldout))
                self.assertEqual(row["role"] == "TEST", is_heldout)
                self.assertEqual(int(row["effective_supervised_train"]), int(row["role"] in {"TRAIN", "DEMO"}))
                self.assertEqual(row["demo_set_id"], top_demo_id)
                self.assertEqual(row["split_status"], "PROVISIONAL_PENDING_DEMO_APPROVAL")
                self.assertEqual(int(row["amp_positive_label_count"]), len(selected))
                for label in self.label_ids:
                    self.assertEqual(int(row[label]), int(label in selected), (fold, rank, label))

            self.assertEqual(
                sum(
                    int(row["PURPOSE_REMOVAL_OF_ORGANS"])
                    for row in rows
                    if row["role"] == "TEST"
                ),
                0,
            )

        self.assertEqual(heldout_union, set(EXPECTED_HIGH_JURISDICTIONS))
        self.assertEqual(total_test_assignments, 861)

    def test_token_audit_membership_and_threshold_flags(self) -> None:
        self.assertEqual(len(self.token_rows), EXPECTED_N)
        self.assertEqual(
            {int(row["search_rank"]) for row in self.token_rows}, set(self.by_rank)
        )
        self.assertEqual(len({row["canonical_url"] for row in self.token_rows}), EXPECTED_N)

        for row in self.token_rows:
            rank = int(row["search_rank"])
            record = self.by_rank[rank]
            token_count = int(row["modernbert_token_count_with_special_tokens"])
            self.assertGreater(token_count, 0)
            self.assertEqual(row["canonical_url"], record["identity"]["canonical_url"])
            self.assertEqual(row["jurisdiction"], record["identity"]["jurisdiction_country_raw"])
            self.assertEqual(int(row["word_count"]), int(record["text_input"]["word_count"]))
            self.assertEqual(row["tokenizer_model_id"], EXPECTED_TOKENIZER_MODEL)
            self.assertEqual(row["tokenizer_revision"], EXPECTED_TOKENIZER_REVISION)

            coverage = []
            for threshold in TOKEN_THRESHOLDS:
                flag = int(row[f"fully_covered_at_{threshold}"])
                self.assertIn(flag, (0, 1))
                self.assertEqual(flag, int(token_count <= threshold), (rank, threshold))
                coverage.append(flag)
            self.assertEqual(coverage, sorted(coverage))

        for lower, upper in zip(TOKEN_THRESHOLDS, TOKEN_THRESHOLDS[1:]):
            covered_lower = sum(
                int(row[f"fully_covered_at_{lower}"]) for row in self.token_rows
            )
            covered_upper = sum(
                int(row[f"fully_covered_at_{upper}"]) for row in self.token_rows
            )
            self.assertLessEqual(covered_lower, covered_upper)

    def test_demo_candidate_sets_are_traceable_and_not_frozen(self) -> None:
        self.assertGreaterEqual(len(self.demo_sets), 5)
        self.assertLessEqual(len(self.demo_sets), 10)
        self.assertEqual(
            [int(row["candidate_set_rank"]) for row in self.demo_sets],
            list(range(1, len(self.demo_sets) + 1)),
        )
        self.assertEqual(len({row["candidate_set_id"] for row in self.demo_sets}), len(self.demo_sets))

        global_counts = Counter(
            label for record in self.benchmark for label in benchmark_label_ids(record)
        )
        token_by_rank = {
            int(row["search_rank"]): int(row["modernbert_token_count_with_special_tokens"])
            for row in self.token_rows
        }
        high = set(EXPECTED_HIGH_JURISDICTIONS)

        for index, row in enumerate(self.demo_sets, 1):
            self.assertEqual(row["candidate_set_id"], f"demo-bank-proposal-set-{index:02d}-v1")
            self.assertEqual(row["status"], "PROPOSED_NOT_FROZEN")
            ranks = [int(value) for value in json.loads(row["search_ranks_json"])]
            case_ids = json.loads(row["case_ids_json"])
            jurisdictions = json.loads(row["jurisdictions_json"])
            self.assertEqual(len(ranks), 6)
            self.assertEqual(len(set(ranks)), 6)
            self.assertEqual(len(case_ids), 6)
            self.assertEqual(len(jurisdictions), 6)
            self.assertTrue(set(ranks).issubset(self.by_rank))
            self.assertTrue(set(jurisdictions).isdisjoint(high))

            records = [self.by_rank[rank] for rank in ranks]
            self.assertEqual(
                case_ids,
                [record["identity"].get("unodc_case_number") for record in records],
            )
            self.assertEqual(
                jurisdictions,
                [record["identity"]["jurisdiction_country_raw"] for record in records],
            )
            covered = set().union(*(benchmark_label_ids(record) for record in records))
            covered_ordered = [label for label in self.label_ids if label in covered]
            self.assertEqual(json.loads(row["amp_label_ids_covered_json"]), covered_ordered)
            self.assertEqual(int(row["act_labels_covered"]), sum(label.startswith("ACT_") for label in covered))
            self.assertEqual(int(row["means_labels_covered"]), sum(label.startswith("MEANS_") for label in covered))
            self.assertEqual(int(row["purpose_labels_covered"]), sum(label.startswith("PURPOSE_") for label in covered))
            self.assertEqual(int(row["total_amp_labels_covered"]), len(covered))

            expected_form = []
            if any(record["geographic_form"]["geographic_form_internal"] for record in records):
                expected_form.append("INTERNAL")
            if any(record["geographic_form"]["geographic_form_transnational"] for record in records):
                expected_form.append("TRANSNATIONAL")
            self.assertEqual(json.loads(row["form_coverage_json"]), expected_form)

            total_words = sum(int(record["text_input"]["word_count"]) for record in records)
            self.assertEqual(int(row["total_fact_summary_word_count"]), total_words)
            self.assertAlmostEqual(float(row["mean_fact_summary_word_count"]), total_words / 6, places=3)
            self.assertEqual(
                int(row["total_modernbert_summary_tokens"]),
                sum(token_by_rank[rank] for rank in ranks),
            )
            expected_rare_score = sum(1.0 / math.sqrt(global_counts[label]) for label in covered)
            self.assertAlmostEqual(float(row["rare_label_coverage_score"]), expected_rare_score, places=8)

            digest_payload = "".join(
                f"{rank}\t{self.by_rank[rank]['identity']['canonical_url']}\n"
                for rank in ranks
            )
            self.assertEqual(
                row["candidate_set_membership_sha256"],
                hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
            )
            self.assertTrue(row["warning_or_ambiguity_flags"])

    def test_demo_review_preserves_human_decisions_and_covers_top_three_sets(self) -> None:
        top_three = sorted(self.demo_sets, key=lambda row: int(row["candidate_set_rank"]))[:3]
        expected_review_ranks = {
            int(rank)
            for candidate in top_three
            for rank in json.loads(candidate["search_ranks_json"])
        }
        actual_review_ranks = {int(row["search_rank"]) for row in self.demo_review}
        self.assertEqual(actual_review_ranks, expected_review_ranks)
        self.assertEqual(len(actual_review_ranks), len(self.demo_review))

        token_by_rank = {
            int(row["search_rank"]): int(row["modernbert_token_count_with_special_tokens"])
            for row in self.token_rows
        }
        membership_by_rank: dict[int, list[str]] = {}
        for candidate in self.demo_sets:
            for rank in json.loads(candidate["search_ranks_json"]):
                membership_by_rank.setdefault(int(rank), []).append(candidate["candidate_set_id"])

        for row in self.demo_review:
            rank = int(row["search_rank"])
            record = self.by_rank[rank]
            targets = record["amp_targets"]
            form = record["geographic_form"]
            self.assertEqual(row["case_title"], record["identity"]["case_title_raw"])
            self.assertEqual(row["canonical_url"], record["identity"]["canonical_url"])
            self.assertEqual(row["jurisdiction"], record["identity"]["jurisdiction_country_raw"])
            self.assertEqual(row["english_fact_summary"], record["text_input"]["english_fact_summary_raw"])
            self.assertEqual(json.loads(row["legacy_acts_reference_json"]), targets["acts_raw"])
            self.assertEqual(json.loads(row["legacy_means_reference_json"]), targets["means_raw"])
            self.assertEqual(json.loads(row["legacy_purposes_reference_json"]), targets["purposes_raw"])
            self.assertEqual(json.loads(row["act_ontology_ids_json"]), targets["act_ontology_ids"])
            self.assertEqual(json.loads(row["means_ontology_ids_json"]), targets["means_ontology_ids"])
            self.assertEqual(json.loads(row["purpose_ontology_ids_json"]), targets["purpose_ontology_ids"])
            self.assertEqual(json.loads(row["geographic_form_reference_json"]), form["legacy_form_values_raw"])
            self.assertEqual(int(row["word_count"]), int(record["text_input"]["word_count"]))
            self.assertEqual(int(row["modernbert_summary_tokens"]), token_by_rank[rank])
            self.assertEqual(
                row["candidate_set_memberships"].split("|"), membership_by_rank[rank]
            )
            self.assertTrue(row["selection_reason"])
            self.assertEqual(row["reference_support_review_status"], "PENDING_HUMAN_REVIEW")
            self.assertIn(row["reviewer_approve"], {"Agree", "Hold", "Skip"})

        decisions = Counter(row["reviewer_approve"] for row in self.demo_review)
        self.assertEqual(decisions, {"Agree": 11, "Hold": 3, "Skip": 2})
        self.assertEqual(
            {
                int(row["search_rank"])
                for row in self.demo_review
                if row["reviewer_approve"] == "Skip"
            },
            {31, 1517},
        )

    def test_demo_review_v2_retains_decisions_and_adds_unreviewed_candidates(self) -> None:
        self.assertEqual(len(self.demo_review_v2), 29)
        self.assertEqual(
            Counter(row["candidate_group"] for row in self.demo_review_v2),
            {
                "EXISTING_REVIEWED": 14,
                "NEW_US": 5,
                "NEW_MAJOR_JURISDICTION": 5,
                "NEW_OTHER": 5,
            },
        )
        prior_by_rank = {int(row["search_rank"]): row for row in self.demo_review}
        v2_ranks = {int(row["search_rank"]) for row in self.demo_review_v2}
        self.assertTrue({31, 1517}.isdisjoint(v2_ranks))

        final_decisions = Counter(
            row["reviewer_approve_v2"] for row in self.demo_review_v2
        )
        self.assertEqual(final_decisions, {"Keep": 8, "Skip": 21})
        self.assertEqual(
            {
                int(row["search_rank"])
                for row in self.demo_review_v2
                if row["reviewer_approve_v2"] == "Keep"
            },
            {1487, 1494, 1178, 498, 391, 157, 1343, 936},
        )

        for row in self.demo_review_v2:
            rank = int(row["search_rank"])
            self.assertIn(row["reviewer_approve_v2"], {"Keep", "Skip"})
            self.assertEqual(row["reviewer_notes_v2"], "")
            self.assertIn(row["form_support_screen"], {"CLEAR", "POSSIBLE", "UNCLEAR", "NOT_APPLICABLE"})
            form_evidence = json.loads(row["form_evidence_sentence_ids_json"])
            form_screens = json.loads(row["form_support_screen_by_label_json"])
            self.assertEqual(set(form_evidence), set(form_screens))
            self.assertTrue(
                all(value in {"CLEAR", "POSSIBLE", "UNCLEAR"} for value in form_screens.values())
            )
            self.assertEqual(row["any_parser_data_warning"], "0")
            if row["candidate_group"] == "EXISTING_REVIEWED":
                prior = prior_by_rank[rank]
                self.assertIn(prior["reviewer_approve"], {"Agree", "Hold"})
                self.assertEqual(row["previous_reviewer_approve"], prior["reviewer_approve"])
                self.assertEqual(row["previous_reviewer_notes"], prior["reviewer_notes"])
            else:
                self.assertEqual(row["previous_reviewer_approve"], "")
                self.assertEqual(row["previous_reviewer_notes"], "")
                self.assertEqual(row["all_amp_reference_labels_clear"], "1")

        new_rows = [
            row for row in self.demo_review_v2
            if row["candidate_group"] != "EXISTING_REVIEWED"
        ]
        self.assertEqual(sum(int(row["geographic_form_clear"]) for row in new_rows), 12)


if __name__ == "__main__":
    unittest.main()
