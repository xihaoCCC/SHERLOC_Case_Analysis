"""Focused tests for the read-only Phase-4 handoff validator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "src/experiments/13_validate_phase4_handoff.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("phase4_handoff_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load handoff validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prediction_row(
    rank: int,
    *,
    method: str = "M1",
    evaluation: str = "A1",
    fold: int | None = None,
) -> dict:
    row = {
        "prediction_schema_version": VALIDATOR.EXPECTED_PREDICTION_SCHEMA,
        "method_id": method,
        "evaluation": evaluation,
        "fold": fold,
        "search_rank": rank,
        "canonical_url": f"https://example.test/{rank}",
        "jurisdiction": "Example",
        "split": "TEST",
        "silver_reference_labels": [],
        "predicted_labels": [],
        "primary_cohort_id": VALIDATOR.EXPECTED_COHORT_ID,
    }
    if method in {"M3", "M4"}:
        row["status"] = "SUCCESS_VALIDATED"
    return row


class Phase4HandoffValidatorTest(unittest.TestCase):
    def test_frozen_hash_check_is_independent_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "frozen.txt"
            path.write_bytes(b"approved bytes")
            original = path.read_bytes()
            expected = {"frozen.txt": hashlib.sha256(original).hexdigest()}

            passed = VALIDATOR.validate_frozen_artifacts(root, expected)
            self.assertEqual(passed["status"], "PASSED")
            self.assertEqual(path.read_bytes(), original)

            path.write_bytes(b"changed")
            failed = VALIDATOR.validate_frozen_artifacts(root, expected)
            self.assertEqual(failed["status"], "FAILED")
            self.assertIn("frozen.txt", failed["details"]["hash_mismatches"])

            path.unlink()
            missing = VALIDATOR.validate_frozen_artifacts(root, expected)
            self.assertEqual(missing["status"], "MISSING")

    def test_prediction_file_validates_identity_and_exact_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a1_test_predictions.jsonl"
            expected = {
                rank: {
                    "canonical_url": f"https://example.test/{rank}",
                    "jurisdiction": "Example",
                }
                for rank in (10, 20)
            }
            write_jsonl(path, [prediction_row(10), prediction_row(20)])
            observed, failures, details = VALIDATOR._validate_prediction_file(
                path,
                method="M1",
                evaluation="A1",
                fold=None,
                expected=expected,
            )
            self.assertFalse(failures)
            self.assertEqual(set(observed), {10, 20})
            self.assertEqual(details["observed_n"], 2)

            bad = prediction_row(10)
            bad["canonical_url"] = "https://wrong.test/10"
            write_jsonl(path, [bad, prediction_row(30)])
            _, failures, _ = VALIDATOR._validate_prediction_file(
                path,
                method="M1",
                evaluation="A1",
                fold=None,
                expected=expected,
            )
            self.assertTrue(any("canonical URL" in item for item in failures))
            self.assertTrue(any("missing 1" in item for item in failures))
            self.assertTrue(any("extra" in item for item in failures))

    def test_llm_prediction_requires_validated_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a1_test_predictions.jsonl"
            row = prediction_row(10, method="M3")
            row["status"] = "FAILED_NO_PREDICTION"
            write_jsonl(path, [row])
            expected = {
                10: {
                    "canonical_url": "https://example.test/10",
                    "jurisdiction": "Example",
                }
            }
            _, failures, _ = VALIDATOR._validate_prediction_file(
                path,
                method="M3",
                evaluation="A1",
                fold=None,
                expected=expected,
            )
            self.assertTrue(any("validated LLM success" in item for item in failures))

    def test_common_membership_distinguishes_missing_and_mismatch(self) -> None:
        stages = [
            VALIDATOR._stage(f"{method.lower()}_a1_predictions", "PASSED", "ok")
            for method in VALIDATOR.METHODS
        ] + [
            VALIDATOR._stage(f"{method.lower()}_a2_predictions", "PASSED", "ok")
            for method in VALIDATOR.METHODS
        ]
        observed = {
            (method, evaluation): {1: None if evaluation == "A1" else 1}
            for method in VALIDATOR.METHODS
            for evaluation in ("A1", "A2")
        }
        self.assertEqual(
            VALIDATOR.validate_common_memberships(stages, observed)["status"],
            "PASSED",
        )

        observed[("M4", "A1")] = {2: None}
        mismatch = VALIDATOR.validate_common_memberships(stages, observed)
        self.assertEqual(mismatch["status"], "FAILED")

        stages[0] = VALIDATOR._stage("m1_a1_predictions", "MISSING", "missing")
        observed[("M4", "A1")] = {1: None}
        incomplete = VALIDATOR.validate_common_memberships(stages, observed)
        self.assertEqual(incomplete["status"], "MISSING")

    def test_canonical_evaluator_gate_requires_exact_inputs_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_files = []
            for path in VALIDATOR.canonical_prediction_paths(root):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                input_files.append(
                    {"path": str(path), "sha256": VALIDATOR.sha256_file(path)}
                )

            metrics_root = root / "outputs/metrics"
            for relative in VALIDATOR.CANONICAL_METRIC_FILES:
                if relative == "amp_evaluation_manifest.json":
                    continue
                path = metrics_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")
            manifest = {
                "final_completion_gate": "PASSED_M1_M2_M3_M4_A1_A2",
                "evaluations": {
                    "A1": {
                        "methods": list(VALIDATOR.METHODS),
                        "test_n": VALIDATOR.EXPECTED_A1_TEST_N,
                    },
                    "A2": {
                        "methods": list(VALIDATOR.METHODS),
                        "test_n": VALIDATOR.EXPECTED_A2_TEST_N,
                    },
                },
                "split_validation": {
                    "a1_final_split_validated": True,
                    "a2_final_split_validated": True,
                },
                "input_files": input_files,
            }
            manifest_path = metrics_root / "amp_evaluation_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                VALIDATOR.validate_evaluator_completion(root)["status"], "PASSED"
            )

            manifest["final_completion_gate"] = "NOT_REQUESTED_PARTIAL_EVALUATION_ALLOWED"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            failed = VALIDATOR.validate_evaluator_completion(root)
            self.assertEqual(failed["status"], "FAILED")

    def test_notebook_check_invokes_generator_check_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator_target = root / "src/experiments/12_generate_analysis_notebooks.py"
            generator_target.parent.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "src/experiments/12_generate_analysis_notebooks.py",
                generator_target,
            )
            notebook_dir = root / "notebooks"
            notebook_dir.mkdir()
            before: dict[str, bytes] = {}
            for name in VALIDATOR.PRIMARY_NOTEBOOKS:
                payload = (REPO_ROOT / "notebooks" / name).read_bytes()
                (notebook_dir / name).write_bytes(payload)
                before[name] = payload

            stage = VALIDATOR.validate_notebooks(root)
            self.assertEqual(stage["status"], "PASSED")
            self.assertTrue(stage["details"]["generator_check_ran"])
            self.assertEqual(
                {name: (notebook_dir / name).read_bytes() for name in before}, before
            )

            with (notebook_dir / VALIDATOR.PRIMARY_NOTEBOOKS[0]).open("ab") as handle:
                handle.write(b"stale")
            self.assertEqual(VALIDATOR.validate_notebooks(root)["status"], "FAILED")

    def test_secret_scan_does_not_read_api_txt_and_reports_no_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # This file deliberately resembles a real credential.  It is not
            # under a generated-artifact scan root and must never be opened.
            (root / "api.txt").write_text(
                "OPENAI_API_KEY=sk-proj-DO-NOT-READ-THIS-012345678901234567890\n",
                encoding="utf-8",
            )
            safe = root / "outputs/predictions/m1/safe.jsonl"
            write_jsonl(safe, [{"status": "SUCCESS", "request_sha256": "a" * 64}])
            passed = VALIDATOR.validate_no_serialized_secrets(root)
            self.assertEqual(passed["status"], "PASSED")
            self.assertFalse(passed["details"]["repository_api_txt_read"])

            unsafe = root / "outputs/logs/unsafe.json"
            unsafe.parent.mkdir(parents=True)
            unsafe.write_text(json.dumps({"api_key": "not-even-a-real-key"}), encoding="utf-8")
            failed = VALIDATOR.validate_no_serialized_secrets(root)
            self.assertEqual(failed["status"], "FAILED")
            self.assertTrue(any("secret field" in item for item in failed["failures"]))
            self.assertTrue(all("not-even-a-real-key" not in item for item in failed["failures"]))

            unsafe.unlink()
            unsafe_csv = root / "outputs/logs/unsafe.csv"
            unsafe_csv.write_text("case_id,OPENAI_API_KEY\n1,placeholder\n", encoding="utf-8")
            csv_failed = VALIDATOR.validate_no_serialized_secrets(root)
            self.assertEqual(csv_failed["status"], "FAILED")
            self.assertTrue(any("openai_api_key" in item for item in csv_failed["failures"]))

    def test_auxiliary_artifacts_must_use_separate_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "outputs/metrics/auxiliary/result.csv"
            allowed.parent.mkdir(parents=True)
            allowed.write_text("x\n", encoding="utf-8")
            self.assertEqual(
                VALIDATOR.validate_primary_auxiliary_separation(root)["status"],
                "PASSED",
            )

            mixed = root / "outputs/metrics/a1/auxiliary_results.csv"
            mixed.parent.mkdir(parents=True)
            mixed.write_text("x\n", encoding="utf-8")
            failed = VALIDATOR.validate_primary_auxiliary_separation(root)
            self.assertEqual(failed["status"], "FAILED")

    def test_text_summary_explicitly_lists_missing_and_failed_stages(self) -> None:
        report = {
            "status": "INCOMPLETE",
            "missing_stages": ["m2_a1_predictions"],
            "failed_stages": ["freeze"],
            "stages": [
                VALIDATOR._stage("freeze", "FAILED", "changed", failures=["hash"]),
                VALIDATOR._stage("m2_a1_predictions", "MISSING", "missing"),
            ],
        }
        text = VALIDATOR.render_text_summary(report)
        self.assertIn("PHASE 4 HANDOFF: INCOMPLETE", text)
        self.assertIn("Missing stages: m2_a1_predictions", text)
        self.assertIn("Failed stages: freeze", text)


if __name__ == "__main__":
    unittest.main()
