"""Offline tests for the preparation-only M3/M4 request builder."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "src/experiments/llm_request_builder.py"
MODULE_NAME = "sherloc_llm_request_builder_under_test"

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, BUILDER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load request builder from {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = BUILDER
SPEC.loader.exec_module(BUILDER)


def target_case() -> dict[str, object]:
    return {
        "case_id": "case-target-999",
        "search_rank": 999,
        "canonical_url": "https://example.test/case/999",
        "fact_summary": "The recruiter deceived and transported the victim for forced labour.",
    }


def demo(index: int) -> dict[str, object]:
    acts = ["ACT_RECRUITMENT"]
    means = ["MEANS_DECEPTION"] if index % 2 else []
    purposes = ["PURPOSE_FORCED_LABOUR_OR_SERVICES"]
    return {
        "demo_id": f"demo-{index}",
        "demo_order": index,
        "search_rank": 100 + index,
        "canonical_url": f"https://example.test/demo/{index}",
        "jurisdiction": f"Outside jurisdiction {index}",
        "fact_summary": f"Demonstration Fact Summary {index}.",
        "output": {
            "acts": acts,
            "means": means,
            "purposes": purposes,
        },
        "human_approved": True,
        "frozen": True,
        "approval_record": f"HT-expert-approval-{index}",
    }


def six_demos() -> list[dict[str, object]]:
    return [demo(index) for index in range(1, 7)]


class LLMRequestBuilderTests(unittest.TestCase):
    def test_shared_marked_instruction_block_is_byte_identical(self) -> None:
        contract = BUILDER.load_shared_prompt_contract()
        m3 = BUILDER.DEFAULT_M3_PROMPT_PATH.read_text(encoding="utf-8")
        m4 = BUILDER.DEFAULT_M4_PROMPT_PATH.read_text(encoding="utf-8")
        m3_block, m3_inner = BUILDER.extract_marked_instruction_block(m3)
        m4_block, m4_inner = BUILDER.extract_marked_instruction_block(m4)

        self.assertEqual(m3_block.encode("utf-8"), m4_block.encode("utf-8"))
        self.assertEqual(m3_inner, m4_inner)
        self.assertEqual(contract["marked_block"], m3_block)
        self.assertEqual(
            contract["marked_block_sha256"],
            hashlib.sha256(m3_block.encode("utf-8")).hexdigest(),
        )

    def test_strict_schema_exactly_matches_frozen_ontology(self) -> None:
        contract = BUILDER.load_contract()
        structured = contract["config"]["structured_output"]
        schema = structured["schema"]
        properties = schema["properties"]

        self.assertEqual(structured["format_type"], "json_schema")
        self.assertIs(structured["strict"], True)
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], ["acts", "means", "purposes"])
        self.assertEqual(set(properties), {"acts", "means", "purposes"})
        self.assertEqual(
            properties["acts"]["items"]["enum"], list(BUILDER.ACT_IDS)
        )
        self.assertEqual(
            properties["means"]["items"]["enum"], list(BUILDER.MEANS_IDS)
        )
        self.assertEqual(
            properties["purposes"]["items"]["enum"], list(BUILDER.PURPOSE_IDS)
        )
        self.assertNotIn("uniqueItems", json.dumps(schema))

    def test_m3_payload_is_independent_zero_shot_and_deterministic(self) -> None:
        first = BUILDER.build_m3_request(target_case())
        second = BUILDER.build_m3_request(target_case())
        payload = first["payload"]

        self.assertEqual(first, second)
        self.assertEqual(len(payload["input"]), 2)
        self.assertEqual([item["role"] for item in payload["input"]], ["developer", "user"])
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertIs(payload["store"], False)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertIs(payload["text"]["format"]["strict"], True)
        self.assertNotIn("tools", payload)
        self.assertNotIn("previous_response_id", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("api_key", json.dumps(first).lower())
        self.assertIsNone(first["metadata"]["demo_bank_version"])
        self.assertIsNone(first["metadata"]["demo_bank_sha256"])

    def test_valid_m4_differs_from_m3_only_by_six_demo_message_pairs(self) -> None:
        case = target_case()
        m3 = BUILDER.build_m3_request(case)
        m4 = BUILDER.build_m4_request(
            case,
            six_demos(),
            demo_bank_version="sherloc-demo-bank-v1-frozen-2026-08-11",
        )
        m4_messages = m4["payload"]["input"]

        self.assertEqual(len(m4_messages), 14)
        self.assertEqual(m4_messages[0], m3["payload"]["input"][0])
        self.assertEqual(m4_messages[-1], m3["payload"]["input"][-1])
        self.assertEqual(
            [m4_messages[0], m4_messages[-1]], m3["payload"]["input"]
        )

        m3_payload_without_input = copy.deepcopy(m3["payload"])
        m4_payload_without_input = copy.deepcopy(m4["payload"])
        del m3_payload_without_input["input"]
        del m4_payload_without_input["input"]
        self.assertEqual(m3_payload_without_input, m4_payload_without_input)

        for offset in range(1, 13, 2):
            self.assertEqual(m4_messages[offset]["role"], "user")
            self.assertEqual(m4_messages[offset + 1]["role"], "assistant")
            parsed = json.loads(m4_messages[offset + 1]["content"])
            self.assertEqual(set(parsed), {"acts", "means", "purposes"})

        self.assertTrue(m4["metadata"]["demo_bank_sha256"])
        self.assertNotEqual(
            m3["metadata"]["request_payload_sha256"],
            m4["metadata"]["request_payload_sha256"],
        )

    def test_m4_fails_closed_without_six_approved_frozen_demos(self) -> None:
        case = target_case()
        invalid_banks: list[tuple[str, object, object]] = [
            ("missing", None, "sherloc-demo-bank-v1"),
            ("five", six_demos()[:5], "sherloc-demo-bank-v1"),
            ("pending version", six_demos(), "demo-bank-proposal-pending"),
        ]
        unapproved = six_demos()
        unapproved[2]["human_approved"] = False
        invalid_banks.append(("unapproved", unapproved, "sherloc-demo-bank-v1"))
        unfrozen = six_demos()
        unfrozen[2]["frozen"] = False
        invalid_banks.append(("unfrozen", unfrozen, "sherloc-demo-bank-v1"))
        missing_record = six_demos()
        missing_record[2]["approval_record"] = ""
        invalid_banks.append(("missing approval record", missing_record, "sherloc-demo-bank-v1"))
        wrong_order = six_demos()
        wrong_order[2]["demo_order"] = 4
        invalid_banks.append(("wrong order", wrong_order, "sherloc-demo-bank-v1"))
        duplicate = six_demos()
        duplicate[5]["canonical_url"] = duplicate[0]["canonical_url"]
        invalid_banks.append(("duplicate", duplicate, "sherloc-demo-bank-v1"))
        for name, bank, version in invalid_banks:
            with self.subTest(name=name):
                with self.assertRaises(BUILDER.RequestBuildError):
                    BUILDER.build_m4_request(
                        case,
                        bank,
                        demo_bank_version=version,
                    )

        heldout = six_demos()
        heldout[0]["jurisdiction"] = "United States of America"
        with self.assertRaises(BUILDER.RequestBuildError):
            BUILDER.build_m4_request(
                case,
                heldout,
                demo_bank_version="sherloc-demo-bank-v1",
                heldout_jurisdictions=["United States of America"],
            )

        # A major-jurisdiction demo is valid when it is not held out in the
        # selected fold; this is required by the frozen fold-specific banks.
        allowed = six_demos()
        allowed[0]["jurisdiction"] = "United States of America"
        BUILDER.build_m4_request(
            case,
            allowed,
            demo_bank_version="sherloc-demo-bank-v1",
            heldout_jurisdictions=["Belgium"],
        )

    def test_output_validation_rejects_duplicates_unknowns_and_wrong_order(self) -> None:
        base = six_demos()[0]["output"]
        invalid_outputs = []
        duplicate = copy.deepcopy(base)
        duplicate["acts"] = ["ACT_RECRUITMENT", "ACT_RECRUITMENT"]
        invalid_outputs.append(duplicate)
        unknown = copy.deepcopy(base)
        unknown["purposes"] = ["PURPOSE_NOT_REAL"]
        invalid_outputs.append(unknown)
        wrong_order = copy.deepcopy(base)
        wrong_order["acts"] = ["ACT_TRANSFER", "ACT_RECRUITMENT"]
        invalid_outputs.append(wrong_order)
        extra_key = copy.deepcopy(base)
        extra_key["confidence"] = 0.9
        invalid_outputs.append(extra_key)
        geographic_form = copy.deepcopy(base)
        geographic_form["geographic_form"] = {
            "internal": True,
            "transnational": False,
        }
        invalid_outputs.append(geographic_form)

        for output in invalid_outputs:
            with self.subTest(output=output):
                with self.assertRaises(BUILDER.RequestBuildError):
                    BUILDER.validate_structured_output(output)

    def test_m4_rejects_target_that_is_a_demonstration(self) -> None:
        demos = six_demos()
        case = target_case()
        case["search_rank"] = demos[0]["search_rank"]
        with self.assertRaises(BUILDER.RequestBuildError):
            BUILDER.build_m4_request(
                case,
                demos,
                demo_bank_version="sherloc-demo-bank-v1",
            )

    def test_prompt_drift_is_rejected(self) -> None:
        original_m3 = BUILDER.DEFAULT_M3_PROMPT_PATH.read_text(encoding="utf-8")
        original_m4 = BUILDER.DEFAULT_M4_PROMPT_PATH.read_text(encoding="utf-8")
        drifted_m4 = original_m4.replace(
            "Return only the schema-constrained result.",
            "Return a schema-constrained result.",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m3_path = root / "m3.md"
            m4_path = root / "m4.md"
            m3_path.write_text(original_m3, encoding="utf-8")
            m4_path.write_text(drifted_m4, encoding="utf-8")
            with self.assertRaises(BUILDER.RequestBuildError):
                BUILDER.load_shared_prompt_contract(m3_path, m4_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
