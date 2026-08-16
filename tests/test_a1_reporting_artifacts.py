"""Regression tests for deterministic A1 post-evaluation reporting."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "src/experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))


def load_reporter():
    path = EXPERIMENTS_DIR / "14_generate_a1_reporting_artifacts.py"
    spec = importlib.util.spec_from_file_location("a1_reporting_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load A1 reporting generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORTER = load_reporter()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class A1ReportingArtifactsTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        metrics = root / "outputs/metrics"
        predictions = root / "outputs/predictions"
        logs = root / "outputs/logs/llm"
        output = metrics / "a1"
        metrics.mkdir(parents=True)
        manifest = {
            "evaluations": {
                "A1": {
                    "methods": list(REPORTER.EXPECTED_METHODS),
                    "test_n": REPORTER.EXPECTED_A1_TEST_N,
                    "macro_label_count": len(REPORTER.AMP_LABEL_IDS),
                    "macro_label_ids": list(REPORTER.AMP_LABEL_IDS),
                }
            },
            "split_validation": {
                "a1_final_split_validated": True,
                "a1_expected_test_n": REPORTER.EXPECTED_A1_TEST_N,
            },
        }
        (metrics / "amp_evaluation_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        primary: list[dict] = []
        for method_index, method in enumerate(REPORTER.EXPECTED_METHODS, start=1):
            value = method_index / 10
            primary.append(
                {
                    "method": method,
                    "prediction_variant": "PRIMARY",
                    "macro_f1": value,
                    "micro_f1": value + 0.01,
                    "exact_set_accuracy": value + 0.02,
                    "example_jaccard": value + 0.03,
                    "act_cpmr": value + 0.04,
                    "means_cpmr": value + 0.05,
                    "purpose_cpmr": value + 0.06,
                    "test_n": REPORTER.EXPECTED_A1_TEST_N,
                }
            )
        write_csv(output / "amp_primary_results.csv", primary)

        per_label: list[dict] = []
        for method_index, method in enumerate(REPORTER.EXPECTED_METHODS, start=1):
            for label_index, label in enumerate(REPORTER.AMP_LABEL_IDS, start=1):
                value = method_index / 10 + label_index / 1000
                per_label.append(
                    {
                        "method": method,
                        "scope": "TEST",
                        "label_id": label,
                        "family": REPORTER.AMP_FAMILY_BY_LABEL[label],
                        "support": label_index,
                        "precision": value,
                        "recall": value + 0.01,
                        "f1": value + 0.02,
                    }
                )
        write_csv(output / "amp_per_label.csv", per_label)
        write_csv(
            output / "amp_bootstrap_cis.csv",
            [{"method": "M1", "evaluation": "A1", "metric": "macro_f1"}],
        )
        write_csv(
            metrics / "amp_threshold_0_50_sensitivity.csv",
            [{"method": "M1", "evaluation": "A1", "prediction_variant": "FIXED_0_50"}],
        )

        case_errors: list[dict] = []
        m2_predictions: list[dict] = []
        llm_predictions = {"M3": [], "M4": []}
        for rank in range(1, REPORTER.EXPECTED_A1_TEST_N + 1):
            narrative = f"Case narrative {rank}"
            reference = json.dumps([REPORTER.AMP_LABEL_IDS[0]], separators=(",", ":"))
            for method in REPORTER.EXPECTED_METHODS:
                predicted = (
                    [REPORTER.AMP_LABEL_IDS[0]]
                    if method in {"M2", "M4"}
                    else []
                )
                false_negative = [] if predicted else [REPORTER.AMP_LABEL_IDS[0]]
                case_errors.append(
                    {
                        "method": method,
                        "case_id": str(rank) if method == "M1" else f"CASE-{rank}",
                        "search_rank": rank,
                        "canonical_url": f"https://example.test/{rank}",
                        "jurisdiction": "Example",
                        "split": "TEST",
                        "fold": "",
                        "fact_summary": narrative,
                        "silver_reference_amp_json": reference,
                        "predicted_amp_json": json.dumps(predicted, separators=(",", ":")),
                        "false_positive_labels_json": "[]",
                        "false_negative_labels_json": json.dumps(
                            false_negative, separators=(",", ":")
                        ),
                        "exact_set_correct": int(bool(predicted)),
                        "example_jaccard": float(bool(predicted)),
                        "act_cpmr": int(bool(predicted)),
                        "act_contained_recall": 1.0 if predicted else "N/A",
                        "means_cpmr": 0,
                        "means_contained_recall": "N/A",
                        "purpose_cpmr": 0,
                        "purpose_contained_recall": "N/A",
                        "truncated_input": 0,
                    }
                )
            m2_predictions.append(
                {
                    "method_id": "M2",
                    "evaluation": "A1",
                    "split": "TEST",
                    "search_rank": rank,
                    "case_id": f"CASE-{rank}",
                    "canonical_url": f"https://example.test/{rank}",
                    "jurisdiction": "Example",
                    "fact_summary": narrative,
                    "original_token_count": 20 + rank,
                    "max_tokens_used": 20 + rank,
                    "truncated_input": False,
                }
            )
            for method in REPORTER.LLM_METHODS:
                retried = method == "M3" and rank == 1
                events = (
                    [
                        {
                            "attempt": 1,
                            "http_status": 429,
                            "error_type": "RateLimitError",
                            "message": "rate limited",
                        }
                    ]
                    if retried
                    else []
                )
                llm_predictions[method].append(
                    {
                        "method": method,
                        "method_id": method,
                        "evaluation": "A1",
                        "split": "TEST",
                        "search_rank": rank,
                        "case_id": f"CASE-{rank}",
                        "canonical_url": f"https://example.test/{rank}",
                        "jurisdiction": "Example",
                        "fact_summary": narrative,
                        "status": "SUCCESS_VALIDATED",
                        "token_usage": {
                            "input_tokens": 100 + rank,
                            "output_tokens": 10,
                            "total_tokens": 110 + rank,
                        },
                        "latency_seconds": 1.0 + rank / 100,
                        "retry_count": int(retried),
                        "retry_events": events,
                        "requested_model_id": "gpt-5.6-luna",
                        "effective_requested_model_id": "gpt-5.6-luna",
                        "returned_model_id": "gpt-5.6-luna-2026-08-01",
                        "sdk_version": "2.test",
                    }
                )
        write_csv(output / "amp_case_level_errors.csv", case_errors)
        write_jsonl(predictions / "m2/a1_test_predictions.jsonl", m2_predictions)
        for method in REPORTER.LLM_METHODS:
            lower = method.lower()
            write_jsonl(
                predictions / lower / "a1_test_predictions.jsonl",
                llm_predictions[method],
            )
            write_jsonl(logs / f"{lower}_a1_failures.jsonl", [])
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{lower}_a1_diagnostics.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "successful_cases": REPORTER.EXPECTED_A1_TEST_N,
                        "unresolved_failure_cases": 0,
                        "failure_history_search_ranks": [1] if method == "M3" else [],
                    }
                ),
                encoding="utf-8",
            )
        return metrics, predictions, logs, output

    def test_generates_complete_deterministic_reporting_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions, logs, output = self.make_fixture(Path(directory))
            diagnostics = REPORTER.generate_reporting_artifacts(
                metrics_root=metrics,
                prediction_root=predictions,
                log_root=logs,
                output_dir=output,
            )
            self.assertEqual(len(diagnostics), 6)
            self.assertTrue(all(row["status"] == "WRITTEN" for row in diagnostics))

            aggregate = read_csv(output / "amp_m3_vs_m4_aggregate_deltas.csv")
            self.assertEqual(len(aggregate), 7)
            self.assertAlmostEqual(float(aggregate[0]["delta_m4_minus_m3"]), 0.1)
            self.assertEqual(aggregate[0]["significance_claim"], "NOT_TESTED_DO_NOT_INFER")
            cpmr_deltas = {
                row["metric"]: float(row["delta_m4_minus_m3"])
                for row in aggregate
                if row["metric"].endswith("_cpmr")
            }
            self.assertEqual(
                set(cpmr_deltas), {"act_cpmr", "means_cpmr", "purpose_cpmr"}
            )
            for delta in cpmr_deltas.values():
                self.assertAlmostEqual(delta, 0.1)

            wide_labels = read_csv(output / "amp_per_label_comparison.csv")
            label_deltas = read_csv(output / "amp_m3_vs_m4_per_label_deltas.csv")
            self.assertEqual(len(wide_labels), 17)
            self.assertEqual(len(label_deltas), 17)
            self.assertIn("m4_f1", wide_labels[0])
            self.assertAlmostEqual(float(label_deltas[0]["delta_f1_m4_minus_m3"]), 0.1)

            cases = read_csv(output / "amp_case_level_comparison.csv")
            self.assertEqual(len(cases), REPORTER.EXPECTED_A1_TEST_N)
            self.assertEqual(cases[0]["case_id"], "CASE-1")
            self.assertEqual(cases[0]["narrative_word_count"], "3")
            self.assertEqual(cases[0]["m2_original_token_count"], "21")
            self.assertIn("m4_false_negative_labels_json", cases[0])
            self.assertEqual(cases[0]["m3_act_cpmr"], "0")
            self.assertEqual(cases[0]["m3_act_contained_recall"], "N/A")
            self.assertEqual(cases[0]["m4_act_cpmr"], "1")
            self.assertEqual(cases[0]["m4_act_contained_recall"], "1.0")
            self.assertEqual(cases[0]["m4_means_cpmr"], "0")
            self.assertEqual(cases[0]["m4_means_contained_recall"], "N/A")

            usage = {row["method"]: row for row in read_csv(output / "amp_llm_api_usage.csv")}
            self.assertEqual(set(usage), {"M3", "M4"})
            self.assertEqual(usage["M3"]["request_count"], "253")
            self.assertEqual(usage["M3"]["retry_count"], "1")
            self.assertEqual(usage["M3"]["rate_limit_event_count"], "1")
            self.assertEqual(usage["M3"]["api_failure_count"], "0")
            self.assertEqual(usage["M3"]["failure_history_case_count"], "1")

            checked = REPORTER.generate_reporting_artifacts(
                metrics_root=metrics,
                prediction_root=predictions,
                log_root=logs,
                output_dir=output,
                check=True,
            )
            self.assertTrue(all(row["status"] == "UNCHANGED" for row in checked))

    def test_incomplete_llm_predictions_fail_without_writing_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, predictions, logs, output = self.make_fixture(root)
            m4_path = predictions / "m4/a1_test_predictions.jsonl"
            rows = m4_path.read_text(encoding="utf-8").splitlines()
            m4_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            for name in (
                "amp_m3_vs_m4_aggregate_deltas.csv",
                "amp_per_label_comparison.csv",
                "amp_m3_vs_m4_per_label_deltas.csv",
                "amp_case_level_comparison.csv",
                "amp_llm_api_usage.csv",
                "amp_a1_post_evaluation_manifest.json",
            ):
                path = output / name
                if path.exists():
                    path.unlink()
            with self.assertRaises(REPORTER.A1ReportingError):
                REPORTER.generate_reporting_artifacts(
                    metrics_root=metrics,
                    prediction_root=predictions,
                    log_root=logs,
                    output_dir=output,
                )
            self.assertFalse((output / "amp_llm_api_usage.csv").exists())

    def test_mismatched_llm_membership_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions, logs, _ = self.make_fixture(Path(directory))
            m3_path = predictions / "m3/a1_test_predictions.jsonl"
            rows = [
                json.loads(line)
                for line in m3_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["search_rank"] = 9999
            write_jsonl(m3_path, rows)
            with self.assertRaisesRegex(
                REPORTER.A1ReportingError, "membership differs from canonical A1"
            ):
                REPORTER.build_reporting_payloads(
                    metrics_root=metrics,
                    prediction_root=predictions,
                    log_root=logs,
                )

    def test_duplicate_primary_method_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions, logs, output = self.make_fixture(Path(directory))
            primary_path = output / "amp_primary_results.csv"
            rows = read_csv(primary_path)
            write_csv(primary_path, [*rows, dict(rows[-1])])
            with self.assertRaisesRegex(
                REPORTER.A1ReportingError, "exactly one PRIMARY row"
            ):
                REPORTER.build_reporting_payloads(
                    metrics_root=metrics,
                    prediction_root=predictions,
                    log_root=logs,
                )


if __name__ == "__main__":
    unittest.main()
