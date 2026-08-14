"""Offline regression tests for the numeric-named SHERLOC parser-v2 module."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = REPO_ROOT / "src/sherloc/03_parse_pages.py"
MODULE_NAME = "sherloc_parser_v2_under_test"

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, PARSER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"Cannot load parser module from {PARSER_PATH}")
PARSER = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = PARSER
SPEC.loader.exec_module(PARSER)


class SherlocParserV2RegressionTests(unittest.TestCase):
    """Parse fixtures and the deterministic challenge once, then reuse them."""

    @classmethod
    def setUpClass(cls) -> None:
        source_rows, download_by_rank, manifest_validation = (
            PARSER.load_and_validate_manifests(
                PARSER.DEFAULT_CORPUS_MANIFEST,
                PARSER.DEFAULT_DOWNLOAD_MANIFEST,
            )
        )
        cls.source_rows = source_rows
        cls.source_by_rank = {
            int(row["search_rank"]): row for row in source_rows
        }
        cls.download_by_rank = download_by_rank
        cls.manifest_validation = manifest_validation

        identity_to_rank = {
            PARSER.canonical_case_identity(row["canonical_url"]): int(
                row["search_rank"]
            )
            for row in source_rows
        }
        cls.fixture_paths = sorted(PARSER.DEFAULT_FIXTURE_DIR.glob("*.html"))
        cls.fixture_ranks = [
            PARSER.fixture_source_rank(path, identity_to_rank)
            for path in cls.fixture_paths
        ]
        cls.fixture_records = [
            PARSER.parse_case_file(
                path,
                cls.source_by_rank[rank],
                download_by_rank[rank],
                input_kind="manual_regression_fixture",
                parsed_at="2000-01-01T00:00:00Z",
            )
            for path, rank in zip(cls.fixture_paths, cls.fixture_ranks)
        ]
        cls.fixture_by_title = {
            record["case_identity"]["title_raw"]: record
            for record in cls.fixture_records
        }

        cls.challenge_reasons = PARSER.challenge_rank_reasons(source_rows)
        cls.challenge_by_rank = {}
        for rank in sorted(cls.challenge_reasons):
            path = PARSER.resolve_raw_path(
                download_by_rank[rank], PARSER.DEFAULT_RAW_HTML_DIR
            )
            cls.challenge_by_rank[rank] = PARSER.parse_case_file(
                path,
                cls.source_by_rank[rank],
                download_by_rank[rank],
                input_kind="production_raw_html",
                parsed_at="2000-01-01T00:00:00Z",
            )

    def test_status_vocabulary_and_all_19_fixture_joins(self) -> None:
        self.assertEqual(
            PARSER.VALID_STATUSES,
            {"FOUND", "SECTION_ABSENT", "EMPTY", "PARTIAL", "PARSE_ERROR"},
        )
        self.assertEqual(self.manifest_validation["status"], "PASS")
        self.assertEqual(len(self.fixture_paths), 19)
        self.assertEqual(len(self.fixture_records), 19)
        self.assertEqual(len(set(self.fixture_ranks)), 19)

        for path, rank, record in zip(
            self.fixture_paths, self.fixture_ranks, self.fixture_records
        ):
            with self.subTest(fixture=path.name):
                self.assertEqual(record["provenance"]["search_rank"], rank)
                self.assertEqual(
                    record["source_input"]["input_kind"],
                    "manual_regression_fixture",
                )
                self.assertIn(
                    record["case_identity"]["og_url_relation_to_canonical"],
                    {"EXACT_MATCH", "CANONICAL_EQUIVALENT"},
                )
                self.assertEqual(record["case_identity"]["page_locale"], "en")

        validation = PARSER.validate_fixture_records(self.fixture_records)
        self.assertEqual(validation["status"], "PASS", validation["failed_checks"])

    def test_b637_and_causa_multilingual_variants_remain_separate(self) -> None:
        b637 = self.fixture_by_title["B637.L6.961-X7-DF"]
        b637_variants = PARSER.fact_variants(b637)
        self.assertEqual([item["language"] for item in b637_variants], ["en", "fr"])
        self.assertEqual([item["pane_index"] for item in b637_variants], [1, 2])
        self.assertIsNone(b637_variants[0]["tab_label_raw"])
        self.assertEqual(b637_variants[1]["tab_label_raw"], "Français")
        self.assertNotEqual(b637_variants[0]["text_raw"], b637_variants[1]["text_raw"])
        self.assertEqual(
            b637["narrative"]["fact_summary"]["english_text_raw"],
            b637_variants[0]["text_raw"],
        )

        causa = self.fixture_by_title["Causa 2422"]
        self.assertEqual(
            [item["language"] for item in PARSER.fact_variants(causa)],
            ["en", "es"],
        )
        for section_key in (
            "commentary_significant_features",
            "procedural_information",
            "sources_citations",
        ):
            with self.subTest(section=section_key):
                panes = PARSER.section_panes(causa, section_key)
                self.assertEqual({pane["language"] for pane in panes}, {"en", "es"})
                self.assertEqual(len(panes), 2)

    def test_twitter_corporation_is_a_single_respondent_source_record(self) -> None:
        twitter = next(
            record
            for title, record in self.fixture_by_title.items()
            if title and "Twitter" in title
        )
        sections = PARSER.participant_sections(twitter, "defendant_respondent")
        self.assertEqual(len(sections), 1)
        self.assertEqual(PARSER.participant_record_count(twitter, "defendant_respondent"), 1)
        self.assertEqual(sections[0]["dom_role_container_type"], "defendantsRespondents")
        self.assertIn("Twitter, INC.", PARSER.all_record_text(sections[0]["records"][0]))
        self.assertEqual(PARSER.charge_subject_count(twitter), 1)
        self.assertEqual(PARSER.charge_record_count(twitter), 2)

    def test_sidebar_is_strictly_scoped_and_legacy_keywords_are_independent(self) -> None:
        cross_classified = [
            record
            for record in self.fixture_records
            if record["case_identity"]["url_path_crime_type"]
            != "traffickingpersonscrimetype"
        ]
        self.assertTrue(cross_classified)
        for record in cross_classified:
            traffic_ordinals = set(record["trafficking_sidebar"]["badge_ordinals"])
            for field in record["trafficking_sidebar"]["fields"].values():
                for source in field["sources"]:
                    with self.subTest(
                        rank=record["provenance"]["search_rank"],
                        field=source["structural_class"],
                    ):
                        self.assertIn(source["badge_ordinal"], traffic_ordinals)

        b637 = self.fixture_by_title["B637.L6.961-X7-DF"]
        self.assertEqual(b637["trafficking_sidebar"]["badge_ordinals"], [2])
        self.assertEqual(
            PARSER.field_values(b637, "sidebar", "sector"),
            [],
        )
        self.assertEqual(
            PARSER.field_values(b637, "legacy", "sector"),
            ["Commercial sexual exploitation"],
        )
        first_value = b637["trafficking_sidebar"]["fields"]["acts"]["sources"][0][
            "value_records"
        ][0]
        self.assertEqual(first_value["value_raw"], "Recruitment/Hiring")
        self.assertEqual(first_value["source_text_raw"], "• Recruitment/Hiring")
        self.assertEqual(first_value["decorative_prefix_removed"], "•")

    def test_sentencia_preserves_visible_migrants_role(self) -> None:
        sentencia = self.fixture_by_title["Sentencia 298/2015"]
        sections = PARSER.participant_sections(sentencia, "person_role")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["dom_role_container_type"], "victimsPlaintiffs")
        self.assertEqual(sections[0]["visible_section_heading_raw"], "Migrants")
        self.assertEqual(sections[0]["record_count"], 1)
        self.assertIn("Migrant:", PARSER.all_record_text(sections[0]["records"][0]))

    def test_repeated_defendants_charges_and_direct_court_scoping(self) -> None:
        robinson = self.fixture_by_title["United States v Robinson"]
        self.assertEqual(PARSER.participant_record_count(robinson, "defendant_respondent"), 17)
        self.assertEqual(PARSER.charge_subject_count(robinson), 17)
        self.assertEqual(PARSER.charge_record_count(robinson), 63)

        self.assertTrue(PARSER.main_section_present(self.fixture_by_title["B637.L6.961-X7-DF"], "court"))
        self.assertTrue(PARSER.main_section_present(self.fixture_by_title["Rex and Obinna Obeta, Ike and Beatrice Ekweremadu"], "court"))
        absent_titles = {
            title
            for title, record in self.fixture_by_title.items()
            if not PARSER.main_section_present(record, "court")
        }
        self.assertEqual(len(absent_titles), 3)
        self.assertIn("Sentencia 298/2015", absent_titles)
        self.assertTrue(any("Twitter" in title for title in absent_titles))
        self.assertTrue(any("Querela" in title for title in absent_titles))

    def test_all_25_missing_fact_cases_are_section_absent(self) -> None:
        missing = {
            rank
            for rank, record in self.challenge_by_rank.items()
            if record["narrative"]["fact_summary"]["status"]
            == PARSER.STATUS_SECTION_ABSENT
        }
        self.assertEqual(len(PARSER.KNOWN_MISSING_FACT_RANKS), 25)
        self.assertEqual(missing, PARSER.KNOWN_MISSING_FACT_RANKS)
        validation = PARSER.validate_challenge_records(
            list(self.challenge_by_rank.values()),
            self.source_rows,
            self.challenge_reasons,
        )
        self.assertEqual(validation["status"], "PASS", validation["failed_checks"])

    def test_rank_307_duplicate_tab_identity_retains_both_panes(self) -> None:
        record = self.challenge_by_rank[307]
        matching = [
            group
            for _label, group in PARSER.iter_record_tab_groups(record)
            if "DUPLICATE_TAB_PANE_ID"
            in {warning["code"] for warning in group["warnings"]}
        ]
        self.assertEqual(len(matching), 1)
        group = matching[0]
        self.assertEqual(group["status"], PARSER.STATUS_PARTIAL)
        self.assertEqual(group["pane_count"], 2)
        self.assertEqual([pane["pane_index"] for pane in group["panes"]], [1, 2])
        self.assertEqual(len({pane["pane_id_raw"] for pane in group["panes"]}), 1)
        self.assertEqual(len({pane["tab_href_raw"] for pane in group["panes"]}), 1)
        self.assertEqual(len({pane["text_raw"] for pane in group["panes"]}), 2)
        self.assertIn("DUPLICATE_TAB_HREF", PARSER.all_warning_codes(record))

    def test_malformed_nested_charge_subjects_are_not_dropped(self) -> None:
        for rank, expected_subjects in ((63, 4), (1489, 3)):
            with self.subTest(rank=rank):
                record = self.challenge_by_rank[rank]
                self.assertEqual(PARSER.charge_subject_count(record), expected_subjects)
                self.assertEqual(
                    record["charges_claims_decisions"]["orphan_charge_records"],
                    [],
                )
                self.assertIn(
                    "NESTED_CHARGE_SUBJECT_RECORDS",
                    PARSER.all_warning_codes(record),
                )
                nested_fallbacks = [
                    subject
                    for subject in record["charges_claims_decisions"]["subject_records"]
                    if subject["raw_text_excluded_nested_subject_count"]
                ]
                self.assertTrue(nested_fallbacks)
                for subject in nested_fallbacks:
                    self.assertIsNotNone(subject["dom_subtree_text_raw"])
                    self.assertNotEqual(
                        subject["raw_text"], subject["dom_subtree_text_raw"]
                    )

        for rank in (1, 63):
            with self.subTest(nested_court_rank=rank):
                record = self.challenge_by_rank[rank]
                self.assertTrue(PARSER.main_section_present(record, "court"))
                self.assertIn(
                    "NESTED_CASE_LAW_DETAIL",
                    PARSER.all_warning_codes(record),
                )
                charges_fallback = record["charges_claims_decisions"][
                    "non_pane_text_raw"
                ] or ""
                for court in record["main_record_sections"]["court"]:
                    self.assertNotIn(court["non_pane_text_raw"], charges_fallback)

    def test_provenance_checksum_and_coverage_cardinality_helpers(self) -> None:
        record = self.challenge_by_rank[307]
        source = self.source_by_rank[307]
        download = self.download_by_rank[307]
        self.assertEqual(record["provenance"]["search_rank"], 307)
        self.assertEqual(record["provenance"]["api_result_id"], source["api_result_id"])
        self.assertEqual(record["provenance"]["canonical_url"], source["canonical_url"])
        self.assertEqual(record["provenance"]["requested_url"], download["requested_url"])
        self.assertEqual(
            record["source_input"]["computed_sha256"], download["sha256"]
        )
        self.assertEqual(
            record["source_input"]["computed_byte_count"],
            int(download["byte_count"]),
        )
        self.assertEqual(
            record["case_identity"]["og_url_relation_to_canonical"],
            "CANONICAL_EQUIVALENT",
        )

        coverage = [PARSER.coverage_row(item) for item in self.fixture_records]
        self.assertEqual(len(coverage), 19)
        self.assertEqual(len({row["search_rank"] for row in coverage}), 19)
        self.assertEqual(sum(row["has_any_fact_summary"] for row in coverage), 19)
        self.assertEqual(sum(row["person_role_record_count"] for row in coverage), 54)
        self.assertEqual(
            sum(row["defendant_respondent_record_count"] for row in coverage), 45
        )
        self.assertEqual(sum(row["charge_subject_record_count"] for row in coverage), 45)
        self.assertEqual(sum(row["charge_record_count"] for row in coverage), 111)
        self.assertEqual(sum(row["has_court"] for row in coverage), 16)


if __name__ == "__main__":
    unittest.main()
