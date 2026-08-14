"""Focused integrity tests for the final pre-results Phase-4 freeze."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW = REPO_ROOT / "data/annotations/demo_bank_review_v2.csv"
DEMO_BANK = REPO_ROOT / "config/experiments/demo_bank_amp_v1.yaml"
LLM_CONFIG = REPO_ROOT / "config/experiments/llm_extraction_amp_v2.yaml"
M3_PROMPT = REPO_ROOT / "prompts/m3_zero_shot_amp_v2.md"
M4_PROMPT = REPO_ROOT / "prompts/m4_six_shot_amp_v2.md"
A1 = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
A2 = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"

ACTIVE = (1487, 1494, 1178, 498, 391, 157)
RESERVE = (1343, 936)
FOLD_BANKS = {
    1: ACTIVE,
    2: (1487, 1494, 1178, 498, 157, 936),
    3: (1487, 1494, 391, 157, 1343, 936),
}
FOLD_HELDOUT = {
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ExperimentFreezeV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.review = read_csv(REVIEW)
        cls.demo = json.loads(DEMO_BANK.read_text(encoding="utf-8"))
        cls.llm = json.loads(LLM_CONFIG.read_text(encoding="utf-8"))
        cls.a1 = read_csv(A1)
        cls.a2 = read_csv(A2)

    def test_exact_human_approved_bank_and_rejections(self) -> None:
        decisions = {
            int(row["search_rank"]): row["reviewer_approve_v2"]
            for row in self.review
        }
        self.assertEqual(
            {rank for rank, value in decisions.items() if value == "Keep"},
            set(ACTIVE + RESERVE),
        )
        self.assertEqual(decisions[146], "Skip")
        self.assertEqual(decisions[1211], "Skip")
        self.assertEqual(tuple(self.demo["roles"]["active_six"]), ACTIVE)
        self.assertEqual(tuple(self.demo["roles"]["reserve_two"]), RESERVE)
        expected_banks = {
            "A1": ACTIVE,
            **{f"A2_FOLD_{fold}": ranks for fold, ranks in FOLD_BANKS.items()},
        }
        for bank_id, ranks in expected_banks.items():
            self.assertEqual(
                tuple(self.demo["evaluation_banks"][bank_id]["ordered_search_ranks"]),
                ranks,
            )

        approved = self.demo["approved_cases"]
        self.assertEqual(len(approved), 8)
        self.assertEqual({item["search_rank"] for item in approved}, set(ACTIVE + RESERVE))
        for item in approved:
            self.assertEqual(item["human_approval"]["status"], "Keep")
            self.assertIs(item["human_approved"], True)
            self.assertIs(item["frozen"], True)
            self.assertTrue(item["approval_record"])
            self.assertEqual(set(item["output"]), {"acts", "means", "purposes"})
            text_hash = hashlib.sha256(item["fact_summary"].encode("utf-8")).hexdigest()
            self.assertEqual(item["fact_summary_sha256"], text_hash)

    def test_amp_only_prompt_and_strict_schema_contract(self) -> None:
        m3 = M3_PROMPT.read_text(encoding="utf-8")
        m4 = M4_PROMPT.read_text(encoding="utf-8")
        begin = "<!-- SHERLOC_SHARED_INSTRUCTIONS_V2_BEGIN -->"
        end = "<!-- SHERLOC_SHARED_INSTRUCTIONS_V2_END -->"

        def block(text: str) -> str:
            self.assertEqual(text.count(begin), 1)
            self.assertEqual(text.count(end), 1)
            return text[text.index(begin) : text.index(end) + len(end)]

        self.assertEqual(block(m3).encode("utf-8"), block(m4).encode("utf-8"))
        schema = self.llm["structured_output"]["schema"]
        self.assertEqual(schema["required"], ["acts", "means", "purposes"])
        self.assertEqual(set(schema["properties"]), {"acts", "means", "purposes"})
        self.assertIs(schema["additionalProperties"], False)
        self.assertNotIn("geographic_form", json.dumps(schema).lower())
        self.assertIs(self.llm["structured_output"]["strict"], True)
        self.assertEqual(self.llm["api_request"]["model"], "gpt-5.6-luna")
        self.assertIs(self.llm["api_request"]["store"], False)
        self.assertEqual(self.llm["api_request"]["credential_source"], "OPENAI_API_KEY_ENVIRONMENT_ONLY")
        self.assertEqual(
            self.llm["methods"]["M3"]["prompt_sha256"],
            hashlib.sha256(m3.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            self.llm["methods"]["M4"]["prompt_sha256"],
            hashlib.sha256(m4.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            self.llm["methods"]["M4"]["demo_bank_file_sha256"],
            hashlib.sha256(DEMO_BANK.read_bytes()).hexdigest(),
        )

    def test_a1_integrity_and_rare_label_support(self) -> None:
        self.assertEqual(len(self.a1), 1263)
        self.assertEqual(len({row["search_rank"] for row in self.a1}), 1263)
        counts = Counter(row["split"] for row in self.a1)
        self.assertEqual(
            counts,
            Counter(
                TRAIN=876,
                VALIDATION=126,
                TEST=253,
                ACTIVE_DEMO=6,
                RESERVE_DEMO=2,
            ),
        )
        by_rank = {int(row["search_rank"]): row for row in self.a1}
        self.assertEqual(
            {rank for rank, row in by_rank.items() if row["split"] == "ACTIVE_DEMO"},
            set(ACTIVE),
        )
        self.assertEqual(
            {rank for rank, row in by_rank.items() if row["split"] == "RESERVE_DEMO"},
            set(RESERVE),
        )
        for role in ("TRAIN", "VALIDATION", "TEST"):
            positives = sum(
                int(row["PURPOSE_REMOVAL_OF_ORGANS"])
                for row in self.a1
                if row["split"] == role
            )
            self.assertGreater(positives, 0)
        self.assertEqual(sum(int(row["effective_supervised_train"]) for row in self.a1), 884)

    def test_a2_fold_and_demo_jurisdiction_leakage(self) -> None:
        self.assertEqual(len(self.a2), 3789)
        for fold in (1, 2, 3):
            rows = [row for row in self.a2 if int(row["fold_id"]) == fold]
            self.assertEqual(len(rows), 1263)
            self.assertEqual(len({row["search_rank"] for row in rows}), 1263)
            test_rows = [row for row in rows if row["role"] == "TEST"]
            self.assertEqual(
                {row["jurisdiction"] for row in test_rows}, FOLD_HELDOUT[fold]
            )
            demos = [row for row in rows if row["m4_demo"] == "1"]
            self.assertEqual(
                tuple(int(row["search_rank"]) for row in demos),
                tuple(sorted(FOLD_BANKS[fold])),
            )
            self.assertEqual(len(demos), 6)
            self.assertFalse({row["jurisdiction"] for row in demos} & FOLD_HELDOUT[fold])
            self.assertEqual(
                sum(int(row["PURPOSE_REMOVAL_OF_ORGANS"]) for row in test_rows), 0
            )

        pooled_test = [row for row in self.a2 if row["role"] == "TEST"]
        self.assertEqual(len(pooled_test), 861)
        self.assertEqual(len({row["search_rank"] for row in pooled_test}), 861)

    def test_nonused_approved_cases_follow_ordinary_fold_membership(self) -> None:
        by_fold_rank = {
            (int(row["fold_id"]), int(row["search_rank"])): row for row in self.a2
        }
        self.assertEqual(by_fold_rank[(1, 936)]["role"], "TEST")
        self.assertEqual(by_fold_rank[(2, 391)]["role"], "TEST")
        self.assertEqual(by_fold_rank[(2, 1343)]["role"], "TEST")
        self.assertEqual(by_fold_rank[(3, 1178)]["role"], "TEST")
        self.assertEqual(by_fold_rank[(3, 498)]["role"], "TEST")
        for fold, ranks in FOLD_BANKS.items():
            for rank in ranks:
                self.assertIn(
                    by_fold_rank[(fold, rank)]["role"],
                    {"ACTIVE_DEMO", "RESERVE_DEMO"},
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
