"""Regression tests for the frozen single-reviewer Evaluation B reference."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/experiments"))

from evaluation_b_reference import (  # noqa: E402
    ACT_SOURCE_COLUMN,
    AMP_ALLOWED_BY_FAMILY,
    EXPECTED_SOURCE_SHA256,
    FREEZE_STATUS,
    GEO_ALLOWED,
    HumanReferenceError,
    build_single_reviewer_reference,
    classify_review_status,
    parse_human_list,
    sha256_file,
)


SOURCE = REPO_ROOT / "data/annotations/reviewer_annotation_template.csv"
CONTEXT = REPO_ROOT / "data/annotations/reliability_sample_100.csv"
A1 = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
ANALYSIS = REPO_ROOT / "outputs/analysis/evaluation_b"
SOURCE_MANIFEST = ANALYSIS / "human_annotation_source_manifest.json"
QC_REPORT = ANALYSIS / "human_annotation_qc_report.csv"
QC_SUMMARY = ANALYSIS / "human_annotation_qc_summary.json"
REFERENCE = REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
EXCLUSIONS = REPO_ROOT / "data/annotations/human_grounded_reference_exclusions_v1.csv"
MEMBERSHIP = ANALYSIS / "human_grounded_reference_membership_v1.csv"
FREEZE = ANALYSIS / "eval_b_membership_manifest.json"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class SyntaxNormalizationTest(unittest.TestCase):
    def test_done_gate_and_note_status_are_deterministic(self) -> None:
        self.assertEqual(classify_review_status("", "Skip"), "NOT_REVIEWED")
        self.assertEqual(classify_review_status("1", ""), "SUBSTANTIVE")
        self.assertEqual(classify_review_status("1", " sKiP "), "SKIP")
        self.assertEqual(classify_review_status("1", "ABSTAIN: insufficient"), "ABSTAIN")
        self.assertEqual(
            classify_review_status(
                "1", "Abstain, since no trafficking involved, it is voluntary behavior"
            ),
            "ABSTAIN",
        )
        with self.assertRaises(HumanReferenceError):
            classify_review_status("yes", "")

    def test_amp_smart_quotes_and_order_normalize_syntax_only(self) -> None:
        parsed = parse_human_list(
            '[“Receipt”, “Recruitment”, "Harbouring"]',
            allowed_values=AMP_ALLOWED_BY_FAMILY["ACT"],
        )
        self.assertEqual(parsed.values, ("Recruitment", "Harbouring", "Receipt"))
        self.assertTrue(parsed.smart_quotes_normalized)
        self.assertTrue(parsed.reordered)
        with self.assertRaises(HumanReferenceError):
            parse_human_list(
                '["Recruiting"]', allowed_values=AMP_ALLOWED_BY_FAMILY["ACT"]
            )

    def test_geographic_form_unquoted_value_and_ocg_are_structural(self) -> None:
        parsed = parse_human_list(
            "[Internal]",
            allowed_values=(*GEO_ALLOWED, "Organized Criminal Group"),
            allow_unquoted_bracket_items=True,
        )
        self.assertEqual(parsed.values, ("Internal",))
        self.assertTrue(parsed.non_json_syntax_normalized)
        parsed = parse_human_list(
            '["Transnational", "Organized Criminal Group"]',
            allowed_values=(*GEO_ALLOWED, "Organized Criminal Group"),
            allow_unquoted_bracket_items=True,
        )
        self.assertIn("Organized Criminal Group", parsed.values)


class FrozenHumanReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = csv_rows(REFERENCE)
        cls.exclusions = csv_rows(EXCLUSIONS)
        cls.membership = csv_rows(MEMBERSHIP)
        cls.qc = json.loads(QC_SUMMARY.read_text(encoding="utf-8"))
        cls.qc_rows = csv_rows(QC_REPORT)
        cls.source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        cls.freeze = json.loads(FREEZE.read_text(encoding="utf-8"))

    def test_raw_source_is_immutable_and_manifested_exactly(self) -> None:
        self.assertEqual(sha256_file(SOURCE), EXPECTED_SOURCE_SHA256)
        self.assertEqual(self.source_manifest["sha256"], EXPECTED_SOURCE_SHA256)
        self.assertEqual(self.source_manifest["row_count"], 100)
        self.assertTrue(self.source_manifest["immutable_raw_source"])
        self.assertIn(ACT_SOURCE_COLUMN, self.source_manifest["source"]["columns_exact"])

    def test_authoritative_done_gate_counts_and_masks(self) -> None:
        self.assertEqual(self.qc["status"], "PASS")
        self.assertEqual(self.qc["blocking_error_count"], 0)
        expected = {
            "reviewed_n": 74,
            "not_reviewed_n": 26,
            "skip_n": 13,
            "abstain_n": 6,
            "substantive_n": 55,
            "retained_n": 61,
        }
        for field, value in expected.items():
            self.assertEqual(self.qc[field], value)
        self.assertEqual(len(self.reference), 61)
        self.assertEqual(Counter(row["review_status"] for row in self.reference), Counter({"SUBSTANTIVE": 55, "ABSTAIN": 6}))
        self.assertNotIn("SKIP", {row["review_status"] for row in self.reference})
        for row in self.reference:
            expected_mask = "1" if row["review_status"] == "SUBSTANTIVE" else "0"
            self.assertEqual(row["substantive_amp_evaluable"], expected_mask)
            self.assertEqual(row["auxiliary_evaluable"], expected_mask)
            self.assertEqual(row["organized_criminal_group_evaluable"], expected_mask)

    def test_exclusions_preserve_skip_and_unreviewed_reasons(self) -> None:
        self.assertEqual(len(self.exclusions), 39)
        self.assertEqual(
            Counter(row["exclusion_reason"] for row in self.exclusions),
            Counter({"NOT_REVIEWED_DONE_BLANK": 26, "REVIEWER_SKIP": 13}),
        )
        unreviewed = [row for row in self.membership if row["review_status"] == "NOT_REVIEWED"]
        self.assertEqual(len(unreviewed), 26)
        self.assertTrue(all(row["retained"] == "0" for row in unreviewed))

    def test_abstain_amp_is_empty_but_not_interpreted_as_negative(self) -> None:
        abstain = [row for row in self.reference if row["review_status"] == "ABSTAIN"]
        self.assertEqual(len(abstain), 6)
        for row in abstain:
            for field in ("act_labels", "means_labels", "purpose_labels"):
                self.assertEqual(json.loads(row[field]), [])
            self.assertEqual(row["substantive_amp_evaluable"], "0")
        hr61 = next(row for row in abstain if row["reliability_case_id"] == "HRV1-061")
        self.assertEqual(json.loads(hr61["purpose_labels"]), [])
        self.assertTrue(self.qc["hrv1_061_abstain_empty_amp_confirmed"])
        self.assertEqual(self.qc["sjip_typo_count"], 0)

    def test_raw_values_are_preserved_and_clean_sets_are_canonical(self) -> None:
        row72 = next(row for row in self.reference if row["reliability_case_id"] == "HRV1-072")
        self.assertEqual(row72["geographic_form_human_raw"].count("Transnational"), 2)
        self.assertEqual(
            json.loads(row72["geographic_form_human_clean_json"]),
            ["Internal", "Transnational"],
        )
        self.assertEqual(row72["organized_criminal_group_human"], "TRUE")
        ocg_positive = sum(row["organized_criminal_group_human"] == "TRUE" for row in self.reference)
        self.assertEqual(ocg_positive, 6)
        warning_codes = Counter(row["code"] for row in self.qc_rows if row["severity"] == "WARNING")
        self.assertEqual(warning_codes["DUPLICATE_LABEL_DEDUPLICATED"], 1)
        self.assertEqual(warning_codes["OUTSIDE_COHORT_SENTINEL_NORMALIZED"], 2)

    def test_substantive_amp_label_supports_are_frozen(self) -> None:
        substantive = [row for row in self.reference if row["review_status"] == "SUBSTANTIVE"]
        expected = {
            "ACT_RECRUITMENT": 43,
            "ACT_TRANSPORTATION": 32,
            "ACT_TRANSFER": 11,
            "ACT_HARBOURING": 39,
            "ACT_RECEIPT": 33,
            "MEANS_THREAT_FORCE_OR_COERCION": 38,
            "MEANS_ABDUCTION": 5,
            "MEANS_FRAUD": 6,
            "MEANS_DECEPTION": 28,
            "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY": 42,
            "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL": 6,
            "PURPOSE_SEXUAL_EXPLOITATION": 39,
            "PURPOSE_FORCED_LABOUR_OR_SERVICES": 13,
            "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES": 3,
            "PURPOSE_SERVITUDE": 3,
            "PURPOSE_REMOVAL_OF_ORGANS": 2,
            "PURPOSE_OTHER": 4,
        }
        observed = Counter()
        for row in substantive:
            for field in ("act_labels", "means_labels", "purpose_labels"):
                observed.update(json.loads(row[field]))
        self.assertEqual(dict(observed), expected)

    def test_freeze_digest_schema_and_demo_overlap(self) -> None:
        self.assertEqual(self.freeze["status"], FREEZE_STATUS)
        self.assertEqual(self.freeze["retained_n"], 61)
        members = [
            {
                "reliability_case_id": row["reliability_case_id"],
                "search_rank": int(row["search_rank"]),
                "canonical_url": row["canonical_url"],
                "input_sha256": row["input_sha256"],
                "review_status": row["review_status"],
            }
            for row in sorted(
                (row for row in self.membership if row["retained"] == "1"),
                key=lambda item: int(item["search_rank"]),
            )
        ]
        payload = json.dumps(
            members, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(digest, self.freeze["retained_membership_sha256"])
        self.assertEqual(
            self.freeze["a1_active_m4_demo_overlap_audit"]["status"],
            "PASS_NO_OVERLAP",
        )
        self.assertEqual(
            self.freeze["a1_active_m4_demo_overlap_audit"]["overlap_n"], 0
        )

    def test_abstain_nonempty_amp_fails_closed_without_writing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            columns, rows = None, None
            with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                rows = list(reader)
            row61 = next(row for row in rows if row["reliability_case_id"] == "HRV1-061")
            row61["Purpose human labeled"] = '["Other"]'
            modified = root / "reviewer.csv"
            with modified.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            kwargs = {
                "repo_root": REPO_ROOT,
                "source_path": modified,
                "context_path": CONTEXT,
                "a1_split_path": A1,
                "source_manifest_path": root / "manifest.json",
                "qc_report_path": root / "qc.csv",
                "qc_summary_path": root / "qc.json",
                "reference_path": root / "reference.csv",
                "exclusions_path": root / "exclusions.csv",
                "membership_path": root / "membership.csv",
                "freeze_manifest_path": root / "freeze.json",
                "validate_frozen_hashes": False,
            }
            with self.assertRaises(HumanReferenceError):
                build_single_reviewer_reference(**kwargs)
            summary = json.loads((root / "qc.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "BLOCKED")
            self.assertGreaterEqual(summary["blocking_error_count"], 1)
            self.assertFalse((root / "reference.csv").exists())


if __name__ == "__main__":
    unittest.main()
