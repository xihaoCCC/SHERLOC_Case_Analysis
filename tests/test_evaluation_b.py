"""Focused offline tests for the unexecuted Evaluation B infrastructure."""

from __future__ import annotations

import sys
import unittest
import csv
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/experiments"))

from evaluation_b import (  # noqa: E402
    EvaluationBError,
    build_disagreement_queue,
    build_human_gold,
    build_reliability_experiment_provenance,
    cohen_kappa,
    compare_silver_to_human,
    compute_reviewer_agreement,
    evaluate_human_gold_predictions,
    map_evidence_ids_to_text,
    parse_amp_labels,
    parse_evidence_sentence_ids,
    qc_annotations,
    selective_evaluation_mask,
)


def annotation_row(case_id: str, reviewer: str = "R-A") -> dict[str, str]:
    return {
        "reviewer_id": reviewer,
        "reliability_case_id": case_id,
        "sentence_splitter_version": "sherloc_sentence_splitter_v1",
        "sentence_count": "2",
        "numbered_text_sha256": "abc",
        "fact_summary_numbered": "[S1] First sentence.\n[S2] Second sentence.",
        "act_labels": "Recruitment;ACT_TRANSPORTATION",
        "act_answerability": "YES",
        "act_evidence_sentence_ids": "S1",
        "act_notes": "",
        "means_labels": "MEANS_DECEPTION",
        "means_answerability": "YES",
        "means_evidence_sentence_ids": "S1",
        "means_notes": "",
        "purpose_labels": "Forced labour or services",
        "purpose_answerability": "YES",
        "purpose_evidence_sentence_ids": "S2",
        "purpose_notes": "",
        "form_label": "Internal",
        "form_answerability": "YES",
        "form_evidence_sentence_ids": "S2",
        "form_notes": "",
        "multiplicity_label": "SINGLE",
        "multiplicity_answerability": "YES",
        "multiplicity_evidence_sentence_ids": "S1",
        "multiplicity_notes": "",
        "child_label": "FALSE",
        "child_answerability": "YES",
        "child_evidence_sentence_ids": "S1",
        "child_notes": "",
        "overall_narrative_sufficiency": "HIGH",
        "annotation_notes": "",
    }


class AnnotationQCTest(unittest.TestCase):
    def test_exact_raw_and_machine_amp_values_map_without_mutating_raw(self) -> None:
        row = annotation_row("HRV1-001")
        result = qc_annotations([row], expected_case_ids=["HRV1-001"])
        self.assertTrue(result.passed, result.issues)
        normalized = result.normalized_rows[0]
        self.assertEqual(normalized["act_labels"], row["act_labels"])
        self.assertEqual(
            normalized["act_labels_normalized"],
            ["ACT_RECRUITMENT", "ACT_TRANSPORTATION"],
        )
        self.assertEqual(normalized["form_label"], "Internal")
        self.assertEqual(normalized["form_label_normalized"], "INTERNAL")

    def test_malformed_json_list_is_reported(self) -> None:
        row = annotation_row("HRV1-001")
        row["means_labels"] = '["Deception"'
        result = qc_annotations([row])
        self.assertIn("MALFORMED_LIST", {issue["code"] for issue in result.issues})

    def test_duplicate_label_is_reported_before_canonicalization(self) -> None:
        row = annotation_row("HRV1-001")
        row["act_labels"] = "Recruitment;ACT_RECRUITMENT"
        result = qc_annotations([row])
        self.assertIn("DUPLICATE_LABEL", {issue["code"] for issue in result.issues})

    def test_invalid_case_fold_is_not_accepted(self) -> None:
        labels, errors = parse_amp_labels("recruitment", "ACT")
        self.assertEqual(labels, [])
        self.assertTrue(errors)

    def test_evidence_parser_and_mapping(self) -> None:
        self.assertEqual(parse_evidence_sentence_ids("S1;S2"), ["S1", "S2"])
        mapped = map_evidence_ids_to_text(
            "S2", "[S1] First sentence.\n[S2] Second sentence."
        )
        self.assertEqual(mapped, [{"sentence_id": "S2", "text": "Second sentence."}])
        with self.assertRaises(EvaluationBError):
            parse_evidence_sentence_ids("S2;S1")

    def test_qc_finds_unknown_evidence_id_and_expected_case_mismatch(self) -> None:
        row = annotation_row("HRV1-999")
        row["act_evidence_sentence_ids"] = "S3"
        result = qc_annotations([row], expected_case_ids=["HRV1-001"])
        codes = {issue["code"] for issue in result.issues}
        self.assertIn("EVIDENCE_ID_NOT_IN_CASE", codes)
        self.assertIn("UNEXPECTED_CASE_ID", codes)
        self.assertIn("MISSING_EXPECTED_CASE_ID", codes)


class ReviewerAgreementTest(unittest.TestCase):
    def test_exact_set_and_jaccard_agreement(self) -> None:
        a1 = annotation_row("HRV1-001", "A")
        a2 = annotation_row("HRV1-002", "A")
        b1 = annotation_row("HRV1-001", "B")
        b2 = annotation_row("HRV1-002", "B")
        b2["act_labels"] = "Recruitment"
        result = compute_reviewer_agreement([a1, a2], [b1, b2])
        act = next(row for row in result["summary"] if row["target"] == "ACT")
        self.assertEqual(act["exact_agreement"], 0.5)
        self.assertEqual(act["mean_jaccard"], 0.75)
        self.assertEqual(result["metadata"]["any_disagreement_cases"], 1)

    def test_kappa_is_na_under_degenerate_categories(self) -> None:
        result = cohen_kappa(["YES", "YES"], ["YES", "YES"])
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "DEGENERATE_NO_VARIATION")

    def test_disagreement_queue_contains_only_disagreement_and_blank_gold(self) -> None:
        a1 = annotation_row("HRV1-001", "A")
        a2 = annotation_row("HRV1-002", "A")
        b1 = annotation_row("HRV1-001", "B")
        b2 = annotation_row("HRV1-002", "B")
        b2["purpose_labels"] = "Other"
        queue = build_disagreement_queue([a1, a2], [b1, b2])
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["reliability_case_id"], "HRV1-002")
        self.assertEqual(queue[0]["adjudicated_purpose_labels"], "")

    def test_gold_builder_refuses_unresolved_disagreement(self) -> None:
        a = annotation_row("HRV1-001", "A")
        b = annotation_row("HRV1-001", "B")
        b["purpose_labels"] = "Other"
        with self.assertRaises(EvaluationBError):
            build_human_gold([a], [b], [])
        unresolved = {
            "reliability_case_id": "HRV1-001",
            "adjudicated_purpose_labels": "",
        }
        with self.assertRaises(EvaluationBError):
            build_human_gold([a], [b], [unresolved])

    def test_gold_builder_flows_agreement_without_adjudication(self) -> None:
        a = annotation_row("HRV1-001", "A")
        b = annotation_row("HRV1-001", "B")
        gold = build_human_gold([a], [b], [])
        self.assertEqual(len(gold), 1)
        self.assertEqual(gold[0]["act_labels_provenance"], "REVIEWER_AGREEMENT")
        self.assertIn('"reviewer_id":"A"', gold[0]["reviewer_a_raw_annotation_json"])

    def test_gold_builder_refuses_cross_field_inconsistency(self) -> None:
        a = annotation_row("HRV1-001", "A")
        b = annotation_row("HRV1-001", "B")
        b["purpose_labels"] = ""
        b["purpose_answerability"] = "NO"
        adjudication = {
            "reliability_case_id": "HRV1-001",
            "adjudicated_purpose_labels": "[]",
            "adjudicated_purpose_answerability": "YES",
        }
        with self.assertRaises(EvaluationBError):
            build_human_gold([a], [b], [adjudication])


class HumanComparisonAndEvaluationTest(unittest.TestCase):
    def test_silver_vs_human_set_categories_and_counts(self) -> None:
        human = annotation_row("HRV1-001")
        human["act_labels"] = "ACT_RECRUITMENT"
        silver = {
            "reliability_case_id": "HRV1-001",
            "act_labels": "ACT_RECRUITMENT;ACT_TRANSPORTATION",
            "means_labels": "MEANS_DECEPTION",
            "purpose_labels": "PURPOSE_FORCED_LABOUR_OR_SERVICES",
        }
        result = compare_silver_to_human([human], [silver])
        act = next(row for row in result["summary"] if row["family"] == "ACT")
        self.assertEqual(act["shared_label_count"], 1)
        self.assertEqual(act["silver_only_label_count"], 1)
        case = next(row for row in result["case_level"] if row["family"] == "ACT")
        self.assertEqual(case["set_relation_category"], "SILVER_BROADER_THAN_HUMAN")

    def test_unanswerable_family_is_not_treated_as_negative_human_gold(self) -> None:
        human = annotation_row("HRV1-001")
        human["means_labels"] = ""
        human["means_answerability"] = "NO"
        silver = {
            "reliability_case_id": "HRV1-001",
            "act_labels": human["act_labels"],
            "means_labels": "MEANS_DECEPTION",
            "purpose_labels": human["purpose_labels"],
        }
        result = compare_silver_to_human([human], [silver])
        means = next(row for row in result["summary"] if row["family"] == "MEANS")
        self.assertEqual(means["answerable_or_partial_case_n"], 0)
        self.assertEqual(means["unanswerable_case_n"], 1)
        self.assertIsNone(means["exact_set_concordance"])
        self.assertEqual(means["silver_only_label_count"], 0)
        case = next(row for row in result["case_level"] if row["family"] == "MEANS")
        self.assertEqual(case["set_relation_category"], "HUMAN_FAMILY_UNANSWERABLE")

    def test_human_gold_evaluator_accepts_arbitrary_n(self) -> None:
        humans = [annotation_row("H-1"), annotation_row("H-2"), annotation_row("H-3")]
        humans[1]["act_labels"] = "ACT_RECEIPT"
        humans[2]["means_labels"] = "MEANS_FRAUD"
        predictions = [
            {
                "reliability_case_id": row["reliability_case_id"],
                "predicted_labels": (
                    parse_amp_labels(row["act_labels"], "ACT")[0]
                    + parse_amp_labels(row["means_labels"], "MEANS")[0]
                    + parse_amp_labels(row["purpose_labels"], "PURPOSE")[0]
                ),
            }
            for row in humans
        ]
        result = evaluate_human_gold_predictions(humans, predictions)
        self.assertEqual(result["metadata"]["case_n"], 3)
        self.assertEqual(result["aggregate"]["exact_set_accuracy"], 1.0)
        self.assertEqual(len(result["case_level"]), 3)

    def test_human_gold_evaluator_can_join_existing_predictions_by_rank(self) -> None:
        human = annotation_row("HRV1-001")
        human["search_rank"] = "237"
        prediction = {
            "search_rank": 237,
            "predicted_labels": (
                parse_amp_labels(human["act_labels"], "ACT")[0]
                + parse_amp_labels(human["means_labels"], "MEANS")[0]
                + parse_amp_labels(human["purpose_labels"], "PURPOSE")[0]
            ),
        }
        result = evaluate_human_gold_predictions([human], [prediction])
        self.assertEqual(result["metadata"]["artifact_match_field"], "search_rank")
        self.assertEqual(result["case_level"][0]["reliability_case_id"], "HRV1-001")

    def test_selective_mask_is_family_specific(self) -> None:
        rows = [annotation_row("H-1"), annotation_row("H-2")]
        rows[1]["means_answerability"] = "NO"
        rows[1]["means_labels"] = ""
        self.assertEqual(selective_evaluation_mask(rows, "means"), [True, False])


class ExperimentProvenanceTest(unittest.TestCase):
    def test_lookup_retains_outside_primary_case(self) -> None:
        reliability = [
            {
                "reliability_case_id": "H-1",
                "reviewer_order": "1",
                "search_rank": "10",
                "canonical_url": "https://example/10",
                "primary_amp_cohort_member": "1",
            },
            {
                "reliability_case_id": "H-2",
                "reviewer_order": "2",
                "search_rank": "20",
                "canonical_url": "https://example/20",
                "primary_amp_cohort_member": "0",
            },
        ]
        a1 = [
            {
                "search_rank": "10",
                "canonical_url": "https://example/10",
                "split": "TEST",
                "effective_supervised_train": "0",
                "demo_bank_role": "",
                "m4_demo": "0",
            }
        ]
        a2 = []
        for fold, role in ((1, "TEST"), (2, "TRAIN"), (3, "VALIDATION")):
            a2.append(
                {
                    "search_rank": "10",
                    "canonical_url": "https://example/10",
                    "fold_id": str(fold),
                    "role": role,
                    "effective_supervised_train": "1" if role == "TRAIN" else "0",
                    "demo_bank_role": "",
                    "approved_demo_pool_role": "",
                    "m4_demo": "0",
                }
            )
        result = build_reliability_experiment_provenance(reliability, a1, a2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["a1_test"], 1)
        self.assertEqual(result[0]["a2_test_fold_ids"], "1")
        self.assertEqual(result[1]["primary_cohort_status"], "OUTSIDE_PRIMARY_COHORT")
        self.assertEqual(result[1]["a2_fold_1_role"], "OUTSIDE_PRIMARY_COHORT")
        self.assertEqual(result[1]["annotation_values_used"], 0)

    def test_generated_frozen_provenance_is_complete_and_label_free(self) -> None:
        path = (
            REPO_ROOT
            / "outputs/analysis/evaluation_b/reliability_case_experiment_provenance.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["reliability_case_id"] for row in rows}), 100)
        self.assertEqual(
            Counter(row["primary_cohort_status"] for row in rows),
            Counter({"IN_PRIMARY_AMP_COHORT": 89, "OUTSIDE_PRIMARY_COHORT": 11}),
        )
        self.assertEqual({row["annotation_values_used"] for row in rows}, {"0"})
        self.assertEqual({row["case_selection_performed"] for row in rows}, {"0"})


if __name__ == "__main__":
    unittest.main()
