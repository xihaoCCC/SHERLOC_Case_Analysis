"""Offline tests for the A2 API-usage summarizer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src/experiments/15_generate_a2_api_usage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("a2_api_usage", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load A2 usage module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


USAGE = _load_module()


class A2APIUsageTest(unittest.TestCase):
    def test_fold_and_pooled_usage_are_aggregated_from_success_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = USAGE.EXPECTED_FOLD_N
            USAGE.EXPECTED_FOLD_N = {1: 2, 2: 2, 3: 2}
            try:
                for method in ("m3", "m4"):
                    for fold in (1, 2, 3):
                        setting = f"{method}_a2_fold_{fold}"
                        diagnostics = {
                            "status": "COMPLETE",
                            "expected_cases": 2,
                            "successful_cases": 2,
                            "unresolved_failure_cases": 0,
                            "missing_unattempted_cases": 0,
                            "started_at": f"2026-08-14T0{fold}:00:00Z",
                            "updated_at": f"2026-08-14T0{fold}:01:00Z",
                        }
                        (root / f"{setting}_diagnostics.json").write_text(
                            json.dumps(diagnostics), encoding="utf-8"
                        )
                        state = root / "state" / method / f"a2_fold_{fold}" / "success"
                        state.mkdir(parents=True)
                        for index in (1, 2):
                            record = {
                                "search_rank": fold * 10 + index,
                                "latency_seconds": float(index),
                                "retry_count": index - 1,
                                "retry_events": (
                                    [{"error_type": "RateLimitError", "http_status": 429}]
                                    if index == 2
                                    else []
                                ),
                                "token_usage": {
                                    "input_tokens": 100,
                                    "output_tokens": 20,
                                    "total_tokens": 120,
                                },
                                "technical_execution": {
                                    "output_token_fallback_attempts_this_invocation": 1 if index == 2 else 0,
                                    "output_token_fallback_used": index == 2,
                                },
                            }
                            (state / f"{index:06d}.json").write_text(
                                json.dumps(record), encoding="utf-8"
                            )

                rows = USAGE.generate_usage_report(root)
            finally:
                USAGE.EXPECTED_FOLD_N = original

        self.assertEqual(len(rows), 8)
        m3_total = next(
            row for row in rows if row["method"] == "M3" and row["scope"] == "POOLED_OOD_TEST"
        )
        self.assertEqual(m3_total["successful_request_count"], 6)
        self.assertEqual(m3_total["retry_count"], 3)
        self.assertEqual(m3_total["rate_limit_retries"], 3)
        self.assertEqual(m3_total["max_output_fallback_attempts"], 3)
        self.assertEqual(m3_total["recorded_total_tokens"], 720)
        self.assertEqual(m3_total["runtime_seconds"], 180.0)
        self.assertEqual(m3_total["wall_clock_span_seconds"], 7260.0)

    def test_incomplete_diagnostics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "m3_a2_fold_1_diagnostics.json").write_text(
                json.dumps({"status": "PARTIAL"}), encoding="utf-8"
            )
            with self.assertRaises(USAGE.UsageReportError):
                USAGE._fold_summary("M3", 1, root)


if __name__ == "__main__":
    unittest.main()
