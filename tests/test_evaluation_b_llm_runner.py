from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src/experiments/19_run_evaluation_b_llm.py"


def load_module():
    name = "_test_evaluation_b_llm_runner"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module()


class EvaluationBLLMRunnerTests(unittest.TestCase):
    def test_membership_digest_is_order_independent_and_status_sensitive(self):
        cases = [
            {
                "reliability_case_id": "HRV1-002",
                "search_rank": 2,
                "canonical_url": "https://example/2",
                "fact_summary": "second",
                "review_status": "ABSTAIN",
            },
            {
                "reliability_case_id": "HRV1-001",
                "search_rank": 1,
                "canonical_url": "https://example/1",
                "fact_summary": "first",
                "review_status": "SUBSTANTIVE",
            },
        ]
        observed = runner.retained_membership_sha256(cases)
        self.assertEqual(observed, runner.retained_membership_sha256(list(reversed(cases))))
        changed = [dict(row) for row in cases]
        changed[0]["review_status"] = "SUBSTANTIVE"
        self.assertNotEqual(observed, runner.retained_membership_sha256(changed))

    def test_reference_loader_preserves_exact_sample_text_and_retained_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.csv"
            reference_path = root / "reference.csv"
            with sample_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "reliability_case_id",
                        "search_rank",
                        "case_title",
                        "unodc_case_number",
                        "canonical_url",
                        "jurisdiction_raw",
                        "english_fact_summary_raw",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "reliability_case_id": "HRV1-001",
                        "search_rank": "7",
                        "case_title": "Title",
                        "unodc_case_number": "CASE-7",
                        "canonical_url": "https://example/7",
                        "jurisdiction_raw": "Exampleland",
                        "english_fact_summary_raw": "Exact narrative.",
                    }
                )
            with reference_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "reliability_case_id",
                        "search_rank",
                        "canonical_url",
                        "jurisdiction_raw",
                        "english_fact_summary_raw",
                        "review_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "reliability_case_id": "HRV1-001",
                        "search_rank": "7",
                        "canonical_url": "https://example/7",
                        "jurisdiction_raw": "Exampleland",
                        "english_fact_summary_raw": "Exact narrative.",
                        "review_status": "SUBSTANTIVE",
                    }
                )
            cases = runner.load_retained_cases(reference_path, sample_path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["case_id"], "CASE-7")
            self.assertEqual(cases[0]["fact_summary"], "Exact narrative.")
            self.assertNotIn("act_labels", cases[0])

    def test_reuse_requires_exact_request_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            records = [
                {
                    "status": "SUCCESS_VALIDATED",
                    "method": "M3",
                    "search_rank": 1,
                    "request_sha256": "exact",
                    "validated_prediction": {"acts": [], "means": [], "purposes": []},
                },
                {
                    "status": "SUCCESS_VALIDATED",
                    "method": "M3",
                    "search_rank": 2,
                    "request_sha256": "stale",
                    "validated_prediction": {"acts": [], "means": [], "purposes": []},
                },
            ]
            path.write_text(
                "".join(runner.canonical_json(row) + "\n" for row in records),
                encoding="utf-8",
            )
            with patch.object(runner, "_prior_prediction_paths", return_value=[path]):
                reusable, rejected = runner.reusable_prediction_index(
                    "M3",
                    {1: {"request_sha256": "exact"}, 2: {"request_sha256": "new"}},
                )
            self.assertEqual(set(reusable), {1})
            self.assertEqual(rejected[0]["reasons"], ["request_sha256"])

    def test_resume_uses_remaining_fallback_only_after_exact_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            case = {
                "reliability_case_id": "HRV1-001",
                "search_rank": 7,
                "canonical_url": "https://example/7",
                "fact_summary": "Exact narrative.",
            }
            request = {"request_sha256": "base"}
            proof = {
                "response_status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "max_output_tokens": 512,
                "actual_request_sha256": "base",
            }
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "sherloc-evaluation-b-llm-failure-history-v1",
                        "method": "M3",
                        "reliability_case_id": "HRV1-001",
                        "search_rank": 7,
                        "request_sha256": "base",
                        "attempt_history": [
                            {
                                "status": "FAILED_NO_PREDICTION",
                                "method": "M3",
                                "evaluation": "B",
                                "reliability_case_id": "HRV1-001",
                                "search_rank": 7,
                                "canonical_url": "https://example/7",
                                "input_sha256": runner.sha256_text("Exact narrative."),
                                "request_sha256": "base",
                                "technical_execution": {
                                    "initial_incomplete_response_provenance": proof,
                                    "cumulative_output_token_fallback_attempts": 1,
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            phase4 = SimpleNamespace(
                INITIAL_MAX_OUTPUT_TOKENS=512,
                MAX_FALLBACK_ATTEMPTS_PER_CASE=2,
            )
            policy = runner.failure_resume_policy(
                path, phase4, case=case, request=request, method="M3"
            )
            self.assertTrue(policy["start_with_fallback"])
            self.assertEqual(policy["prior_fallback_attempts"], 1)
            self.assertEqual(policy["prior_primary_incomplete_provenance"], proof)
            with self.assertRaises(runner.EvaluationBLLMError):
                runner.failure_resume_policy(
                    path,
                    phase4,
                    case=case,
                    request={"request_sha256": "changed"},
                    method="M3",
                )

    def test_existing_success_is_revalidated_before_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "success.json"
            case = {
                "reliability_case_id": "HRV1-001",
                "case_id": "CASE-7",
                "search_rank": 7,
                "case_title": "Title",
                "canonical_url": "https://example/7",
                "jurisdiction": "Exampleland",
                "fact_summary": "Exact narrative.",
                "review_status": "SUBSTANTIVE",
            }
            request = {
                "request_sha256": "request",
                "builder_payload_sha256": "payload",
                "builder_metadata_sha256": "metadata",
            }
            prediction = {"acts": ["ACT_RECRUITMENT"], "means": [], "purposes": []}
            phase4 = SimpleNamespace(
                MODEL_ALIAS="gpt-test",
                builder=SimpleNamespace(validate_structured_output=lambda value: value),
            )
            prepared = {
                "phase4": phase4,
                "config": {
                    "methods": {"M3": {"prompt_sha256": "prompt"}},
                    "structured_output": {"schema_sha256": "schema"},
                },
                "model_marker": {"effective_model_id": "gpt-test-snapshot"},
                "retained_membership_sha256": "membership",
                "demo_metadata": None,
            }
            record = {
                "schema_version": "sherloc-evaluation-b-llm-prediction-v1",
                "status": "SUCCESS_VALIDATED",
                "method": "M3",
                "evaluation": "B",
                "subset": "RETAINED",
                **case,
                "input_sha256": runner.sha256_text("Exact narrative."),
                **request,
                "prompt_sha256": "prompt",
                "schema_sha256": "schema",
                "demo_bank_id": None,
                "demo_bank_membership_sha256": None,
                "requested_model_id": "gpt-test",
                "effective_requested_model_id": "gpt-test-snapshot",
                "retained_membership_sha256": "membership",
                "human_or_silver_labels_sent_to_model": False,
                "store": False,
                "validated_prediction": prediction,
                "normalized_prediction": prediction,
                "predicted_labels": ["ACT_RECRUITMENT"],
                "api_request_issued_for_evaluation_b": False,
                "reuse_status": "REUSED_IDENTICAL_FROZEN_REQUEST",
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                runner._load_valid_success(
                    path,
                    case=case,
                    request=request,
                    method="M3",
                    prepared=prepared,
                ),
                record,
            )
            record["predicted_labels"] = []
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(runner.EvaluationBLLMError):
                runner._load_valid_success(
                    path,
                    case=case,
                    request=request,
                    method="M3",
                    prepared=prepared,
                )

    def test_m4_gate_requires_exact_m3_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_root = root / "predictions"
            log_root = root / "logs"
            (prediction_root / "m3").mkdir(parents=True)
            log_root.mkdir(parents=True)
            (log_root / "m3_diagnostics.json").write_text(
                json.dumps({"status": "COMPLETE", "successful_predictions": 1}),
                encoding="utf-8",
            )
            (prediction_root / "m3/eval_b_predictions.jsonl").write_text(
                runner.canonical_json(
                    {
                        "status": "SUCCESS_VALIDATED",
                        "reliability_case_id": "WRONG",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(runner.EvaluationBLLMError):
                runner.validate_m3_gate(
                    prediction_root,
                    log_root,
                    [
                        {
                            "reliability_case_id": "HRV1-001",
                            "search_rank": 1,
                            "canonical_url": "https://example/1",
                            "fact_summary": "Narrative.",
                            "review_status": "SUBSTANTIVE",
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
