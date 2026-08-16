from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "eval_b_auxiliary_runner_test", "src/experiments/23_run_evaluation_b_auxiliary_llm.py"
)
evaluator = load_module(
    "eval_b_auxiliary_evaluator_test",
    "src/experiments/24_evaluate_evaluation_b_auxiliary.py",
)


class FakeResponses:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        item = self.sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, sequence):
        self.responses = FakeResponses(sequence)


def completed(output):
    return {
        "id": "resp_test",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [],
        "output_text": json.dumps(output, separators=(",", ":")),
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }


class TransientError(RuntimeError):
    status_code = 500


class AuxiliaryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = runner.prepare()

    def test_frozen_contract_and_exact_membership(self):
        prepared = self.prepared
        self.assertEqual(len(prepared["cases"]), 55)
        self.assertEqual(
            prepared["membership_sha256"],
            "3df23d05c74804ef849e91862c6a4022872b5a3512d90601b53535b591c014bc",
        )
        self.assertEqual(
            prepared["contract"]["prompt_sha256"],
            "b7d9d1c132fdd8454aa16de93fad5d34c24e88218795893e706a16a6d63d6fac",
        )
        self.assertEqual(
            prepared["contract"]["schema_sha256"],
            "3b2a753991f22cfa223e90a93eb7b4c72a802d0b966ec3a38151e543d2362a00",
        )
        self.assertEqual(
            set(prepared["cases"][0]),
            {
                "reliability_case_id",
                "search_rank",
                "canonical_url",
                "jurisdiction",
                "fact_summary",
                "input_sha256",
            },
        )

    def test_payload_is_zero_shot_summary_only_and_store_false(self):
        case = self.prepared["cases"][0]
        payload = runner.build_payload(case, self.prepared["contract"])
        self.assertFalse(payload["store"])
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertEqual(len(payload["input"]), 2)
        self.assertEqual(payload["input"][0]["role"], "developer")
        expected = "Analyze only this supplied Fact Summary:\n" + runner.canonical_json(
            {"fact_summary": case["fact_summary"]}
        )
        self.assertEqual(payload["input"][1], {"role": "user", "content": expected})
        serialized = runner.canonical_json(payload)
        for forbidden in ("canonical_url", "jurisdiction", "search_rank", "acts_human"):
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_geographic_set_is_canonicalized_but_duplicates_rejected(self):
        value = {
            "geographic_form": ["Transnational", "Internal"],
            "multiplicity": "UNKNOWN",
            "child_involvement": "UNKNOWN",
            "organized_criminal_group": "FALSE",
        }
        self.assertEqual(
            runner.validate_output(value)["geographic_form"],
            ["Internal", "Transnational"],
        )
        value["geographic_form"] = ["Internal", "Internal"]
        with self.assertRaises(runner.AuxiliaryLLMError):
            runner.validate_output(value)

    def test_512_incomplete_is_the_only_path_to_2048_fallback(self):
        case = self.prepared["cases"][0]
        payload = runner.build_payload(case, self.prepared["contract"])
        incomplete = {
            "id": "resp_incomplete",
            "model": "gpt-5.6-luna",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"output_tokens": 512},
        }
        final = completed(
            {
                "geographic_form": ["Transnational", "Internal"],
                "multiplicity": "MULTIPLE",
                "child_involvement": "TRUE",
                "organized_criminal_group": "FALSE",
            }
        )
        client = FakeClient([incomplete, final])
        result = runner.invoke_with_retries(
            client, payload, secret="not-a-real-secret", sleeper=lambda _seconds: None
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            [item["max_output_tokens"] for item in client.responses.payloads], [512, 2048]
        )
        self.assertEqual(result["fallback_calls"], 1)
        self.assertEqual(
            result["parsed"]["validated_prediction"]["geographic_form"],
            ["Internal", "Transnational"],
        )

    def test_transient_retry_does_not_change_payload(self):
        case = self.prepared["cases"][0]
        payload = runner.build_payload(case, self.prepared["contract"])
        output = completed(
            {
                "geographic_form": [],
                "multiplicity": "UNKNOWN",
                "child_involvement": "UNKNOWN",
                "organized_criminal_group": "FALSE",
            }
        )
        sleeps = []
        client = FakeClient([TransientError("temporary"), output])
        result = runner.invoke_with_retries(
            client, payload, secret="secret", sleeper=sleeps.append
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(client.responses.payloads), 2)
        self.assertEqual(client.responses.payloads[0], client.responses.payloads[1])
        self.assertEqual(sleeps, [2.0])

    def test_resume_success_preserves_cumulative_call_accounting_and_raw_chain(self):
        case = self.prepared["cases"][0]
        request = self.prepared["requests"][case["search_rank"]]
        raw = {
            "geographic_form": ["Transnational", "Internal"],
            "multiplicity": "MULTIPLE",
            "child_involvement": "UNKNOWN",
            "organized_criminal_group": "FALSE",
        }
        response = completed(raw)
        parsed = runner.parse_response(response)
        result = {
            "response": response,
            "parsed": parsed,
            "request_calls": 1,
            "fallback_calls": 0,
            "initial_incomplete_provenance": None,
            "effective_max_output_tokens": 512,
            "attempts": [],
            "latency_seconds": 0.1,
        }
        prior = {"request_calls": 2, "attempts": [{"error_type": "RateLimitError"}]}
        row = runner._success_record(case, request, result, self.prepared, "2.31.0", prior)
        self.assertEqual(row["prior_request_calls"], 2)
        self.assertEqual(row["request_calls_cumulative"], 3)
        self.assertEqual(len(row["retry_events"]), 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "success.json"
            runner.atomic_json(path, row)
            validated = runner._valid_success(path, case, request, self.prepared)
            self.assertIsNotNone(validated)
            row["raw_structured_response_text"] = "{}"
            runner.atomic_json(path, row)
            with self.assertRaises(runner.AuxiliaryLLMError):
                runner._valid_success(path, case, request, self.prepared)

    def test_plan_is_no_api_and_declares_new_namespace(self):
        plan = runner.build_plan(self.prepared)
        self.assertEqual(plan["status"], "PLAN_ONLY_NO_API_REQUEST")
        self.assertEqual(plan["expected_cases"], 55)
        self.assertEqual(
            plan["existing_validated_successes"] + plan["new_cases_if_executed"],
            55,
        )
        self.assertFalse(plan["human_or_silver_labels_sent_to_model"])


class AuxiliaryEvaluatorTests(unittest.TestCase):
    def test_actual_reference_target_masks_are_55(self):
        references = evaluator.load_reference()
        self.assertEqual(len(references), 55)
        for key in (
            "geo_evaluable",
            "multiplicity_evaluable",
            "child_evaluable",
            "ocg_evaluable",
        ):
            self.assertEqual(sum(bool(row[key]) for row in references), 55)

    def test_target_specific_masks_and_unknown_as_class(self):
        refs = [
            {
                "reliability_case_id": "A",
                "search_rank": 1,
                "canonical_url": "https://example/A",
                "jurisdiction": "X",
                "fact_summary": "One.",
                "input_sha256": "a",
                "geo": ("Internal",),
                "geo_evaluable": True,
                "multiplicity": "UNKNOWN",
                "multiplicity_evaluable": True,
                "child": "Not Applicable",
                "child_evaluable": False,
                "ocg": "FALSE",
                "ocg_evaluable": True,
            },
            {
                "reliability_case_id": "B",
                "search_rank": 2,
                "canonical_url": "https://example/B",
                "jurisdiction": "Y",
                "fact_summary": "Two.",
                "input_sha256": "b",
                "geo": (),
                "geo_evaluable": False,
                "multiplicity": "MULTIPLE",
                "multiplicity_evaluable": False,
                "child": "TRUE",
                "child_evaluable": True,
                "ocg": "TRUE",
                "ocg_evaluable": False,
            },
        ]
        preds = {
            "A": {
                "request_sha256": "ra",
                "validated_prediction": {
                    "geographic_form": ["Internal"],
                    "multiplicity": "UNKNOWN",
                    "child_involvement": "UNKNOWN",
                    "organized_criminal_group": "FALSE",
                },
            },
            "B": {
                "request_sha256": "rb",
                "validated_prediction": {
                    "geographic_form": ["Transnational"],
                    "multiplicity": "SINGLE",
                    "child_involvement": "TRUE",
                    "organized_criminal_group": "TRUE",
                },
            },
        }
        aggregate, per_class, cases = evaluator.evaluate(refs, preds)
        n_by_target = {}
        for row in aggregate:
            n_by_target.setdefault(row["target"], row["n"])
        self.assertEqual(
            n_by_target,
            {
                "GEOGRAPHIC_FORM": 1,
                "VICTIM_MULTIPLICITY": 1,
                "CHILD_INVOLVEMENT": 1,
                "ORGANIZED_CRIMINAL_GROUP": 1,
            },
        )
        unknown = next(
            row
            for row in per_class
            if row["target"] == "VICTIM_MULTIPLICITY" and row["class"] == "UNKNOWN"
        )
        self.assertEqual(unknown["support"], 1)
        self.assertEqual(cases[0]["child_involvement_evaluable"], 0)
        self.assertEqual(cases[0]["child_involvement_correct"], "")

    def test_output_contract_fields_are_stable(self):
        self.assertEqual(
            evaluator.AGGREGATE_FIELDS,
            (
                "target",
                "metric",
                "value",
                "n",
                "support_json",
                "confusion_matrix_json",
                "model",
                "prompt_version",
            ),
        )
        self.assertIn("fact_summary", evaluator.CASE_FIELDS)
        self.assertIn("organized_criminal_group_evaluable", evaluator.CASE_FIELDS)


if __name__ == "__main__":
    unittest.main()
