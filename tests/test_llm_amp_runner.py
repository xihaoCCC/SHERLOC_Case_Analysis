"""Offline regression tests for the Phase-4 M3/M4 execution runner."""

from __future__ import annotations

import importlib.util
import copy
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "src/experiments/10_run_llm_amp.py"
EVALUATOR_PATH = REPO_ROOT / "src/experiments/11_evaluate_amp.py"
EXPERIMENT_DIR = RUNNER_PATH.parent
MODULE_NAME = "sherloc_llm_amp_runner_under_test"

sys.dont_write_bytecode = True
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, RUNNER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import runner from {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = RUNNER
SPEC.loader.exec_module(RUNNER)

EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "sherloc_amp_evaluator_for_llm_runner_test", EVALUATOR_PATH
)
if EVALUATOR_SPEC is None or EVALUATOR_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import evaluator from {EVALUATOR_PATH}")
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
sys.modules[EVALUATOR_SPEC.name] = EVALUATOR
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


VALID_OUTPUT = {
    "acts": ["ACT_RECRUITMENT"],
    "means": ["MEANS_DECEPTION"],
    "purposes": ["PURPOSE_FORCED_LABOUR_OR_SERVICES"],
}


class FakeResponse:
    def __init__(self, output: object = VALID_OUTPUT) -> None:
        self.id = "resp_test_123"
        self.model = "gpt-5.6-luna-2026-08-01"
        self.status = "completed"
        self.output_text = json.dumps(output, separators=(",", ":"))
        self.output = []
        self.usage = {
            "input_tokens": 101,
            "output_tokens": 17,
            "total_tokens": 118,
        }


class FakeIncompleteResponse:
    def __init__(
        self,
        *,
        status: str = "incomplete",
        reason: str = "max_output_tokens",
        response_id: str = "resp_incomplete_test",
    ) -> None:
        self.id = response_id
        self.model = "gpt-5.6-luna-2026-08-01"
        self.status = status
        self.incomplete_details = {"reason": reason}
        self.output_text = ""
        self.output = []
        self.usage = {
            "input_tokens": 101,
            "output_tokens": 512,
            "total_tokens": 613,
        }


def frozen_base_payload() -> dict[str, object]:
    return {
        "model": "gpt-5.6-luna",
        "input": [
            {"role": "developer", "content": "frozen instructions"},
            {"role": "user", "content": "frozen target"},
        ],
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 512,
        "store": False,
    }


class FakeResponses:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[dict[str, object]] = []

    def create(self, **payload: object) -> FakeResponse:
        self.calls.append(payload)
        outcome = self.outcomes.pop(0) if self.outcomes else FakeResponse()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


class FakeClient:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.responses = FakeResponses(outcomes)


class CoordinatedFatalAndSuccessResponses:
    """Start two paid calls together; one fails fatally and one succeeds later."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.call_count = 0

    def create(self, **_payload: object) -> FakeResponse:
        with self.lock:
            index = self.call_count
            self.call_count += 1
        self.barrier.wait(timeout=2)
        if index == 0:
            raise HTTPFailure(401)
        time.sleep(0.05)
        return FakeResponse()


class CoordinatedSuccessResponses:
    """Ensure both worker requests are in flight before either returns."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.call_count = 0

    def create(self, **_payload: object) -> FakeResponse:
        with self.lock:
            self.call_count += 1
        self.barrier.wait(timeout=2)
        return FakeResponse()


class HTTPFailure(RuntimeError):
    def __init__(self, status: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status}; secret={{API_SECRET}}")
        self.status_code = status
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.response = SimpleNamespace(status_code=status, headers=headers)


def temp_spec(root: Path, method: str = "M3", *, dry_run: bool = True):
    return RUNNER.make_spec(
        method,
        "A1",
        None,
        dry_run=dry_run,
        prediction_root=root / "predictions",
        log_root=root / "logs",
    )


class LLMAMPRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.config, cls.bank = RUNNER.validate_frozen_contract()
        cls.benchmark = RUNNER.load_benchmark_index()

    def test_every_canonical_artifact_hash_and_review_rank_is_pinned(self) -> None:
        RUNNER.validate_canonical_artifact_hashes()
        RUNNER.validate_review_decisions()
        observed = tuple(item["search_rank"] for item in self.bank["approved_cases"])
        self.assertEqual(observed, RUNNER.EXPECTED_APPROVED_RANKS)
        self.assertEqual(
            RUNNER.sha256_file(RUNNER.DEFAULT_CONFIG), RUNNER.EXPECTED_CONFIG_SHA256
        )
        self.assertEqual(
            RUNNER.sha256_file(RUNNER.DEFAULT_BENCHMARK),
            RUNNER.EXPECTED_BENCHMARK_SHA256,
        )
        self.assertEqual(
            RUNNER.sha256_file(RUNNER.DEFAULT_TECHNICAL_AMENDMENT),
            RUNNER.EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        )

    def test_coordinated_config_prompt_substitution_and_alt_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "m3.md"
            prompt_text = RUNNER.DEFAULT_M3_PROMPT.read_text(encoding="utf-8")
            prompt_text += "\n<!-- coordinated-substitution -->\n"
            prompt_path.write_text(prompt_text, encoding="utf-8")
            config = copy.deepcopy(self.config)
            config["methods"]["M3"]["prompt_sha256"] = RUNNER.sha256_text(prompt_text)
            config_path = root / "llm_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.validate_frozen_contract(
                    config_path=config_path,
                    demo_bank_path=RUNNER.DEFAULT_DEMO_BANK,
                    m3_prompt_path=prompt_path,
                    m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                )

            copied_benchmark = root / "benchmark.jsonl"
            copied_benchmark.write_bytes(RUNNER.DEFAULT_BENCHMARK.read_bytes())
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.assert_canonical_input_paths(benchmark_path=copied_benchmark)

    def test_request_builder_receives_the_exact_validated_artifact_paths(self) -> None:
        spec = temp_spec(Path(tempfile.gettempdir()), "M3")
        case = next(iter(self.benchmark.values()))
        custom_config = Path("/validated/config.json")
        custom_m3 = Path("/validated/m3.md")
        custom_m4 = Path("/validated/m4.md")
        fake = {
            "payload": {"model": "gpt-5.6-luna", "input": []},
            "metadata": {},
        }
        with mock.patch.object(
            RUNNER.builder, "build_m3_request", return_value=fake
        ) as called, mock.patch.object(
            RUNNER,
            "validate_builder_result",
            return_value={
                "builder_payload_sha256": RUNNER.sha256_text(
                    RUNNER.canonical_json(fake["payload"])
                ),
                "builder_metadata_sha256": RUNNER.sha256_text("{}"),
            },
        ):
            RUNNER.build_request_for_case(
                spec,
                case,
                demos=None,
                demo_metadata=None,
                heldout_jurisdictions=[],
                effective_model_id="gpt-5.6-luna",
                contract=self.contract,
                config=self.config,
                config_path=custom_config,
                m3_prompt_path=custom_m3,
                m4_prompt_path=custom_m4,
            )
        kwargs = called.call_args.kwargs
        self.assertEqual(kwargs["config_path"], custom_config)
        self.assertEqual(kwargs["m3_prompt_path"], custom_m3)
        self.assertEqual(kwargs["m4_prompt_path"], custom_m4)

    def test_builder_metadata_tampering_is_rejected(self) -> None:
        spec = temp_spec(Path(tempfile.gettempdir()), "M3")
        case = next(iter(self.benchmark.values()))
        target = {
            "case_id": case["case_id"],
            "search_rank": case["search_rank"],
            "canonical_url": case["canonical_url"],
            "fact_summary": case["fact_summary"],
        }
        built = RUNNER.builder.build_m3_request(target)
        built["metadata"]["canonical_url"] = "https://invalid.example/tampered"

        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.validate_builder_result(
                built,
                spec,
                case,
                contract=self.contract,
                config=self.config,
                demos=None,
                demo_metadata=None,
            )

    def test_setting_lock_is_exclusive_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = r"""
import importlib.util
import sys
from pathlib import Path
runner_path = Path(sys.argv[1])
root = Path(sys.argv[2])
spec_obj = importlib.util.spec_from_file_location("lock_child_runner", runner_path)
module = importlib.util.module_from_spec(spec_obj)
sys.modules[spec_obj.name] = module
spec_obj.loader.exec_module(module)
run_spec = module.make_spec("M3", "A1", None, dry_run=True,
                            prediction_root=root / "predictions",
                            log_root=root / "logs")
with module.SettingRunLock(run_spec):
    print("LOCKED", flush=True)
    sys.stdin.readline()
print("RELEASED", flush=True)
"""
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(RUNNER_PATH), str(root)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(process.stdout.readline().strip(), "LOCKED")
                spec = temp_spec(root, "M3")
                with self.assertRaises(RUNNER.LLMProtocolError):
                    RUNNER.SettingRunLock(spec).acquire()
                process.stdin.write("release\n")
                process.stdin.flush()
                self.assertEqual(process.stdout.readline().strip(), "RELEASED")
                self.assertEqual(process.wait(timeout=5), 0, process.stderr.read())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def test_dead_owner_lock_is_archived_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = temp_spec(Path(directory), "M3")
            spec.state_dir.mkdir(parents=True)
            marker_path = spec.state_dir / ".run.lock.json"
            marker_path.write_text(
                json.dumps(
                    {
                        "lock_schema_version": RUNNER.RUN_LOCK_SCHEMA_VERSION,
                        "token": "dead-owner-token",
                        "pid": 99999999,
                        "hostname": socket.gethostname(),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(RUNNER, "_pid_is_alive", return_value=False):
                with RUNNER.SettingRunLock(spec) as acquired:
                    archived = list((spec.state_dir / ".stale_locks").glob("*.json"))
                    self.assertEqual(len(archived), 1)
                    self.assertEqual(
                        json.loads(archived[0].read_text())["token"], "dead-owner-token"
                    )
                    history = RUNNER.load_json(acquired.history_path)
                    self.assertEqual(
                        history["stale_marker_recovery"]["reason"],
                        "SAME_HOST_OWNER_PID_NOT_ALIVE",
                    )
            released_history = RUNNER.load_json(acquired.history_path)
            self.assertEqual(released_history["status"], "RELEASED")
            self.assertFalse(marker_path.exists())

    def test_stage_order_requires_m2_then_m3_and_a1_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m2_gate = {"status": "M2_A1_A2_COMPLETE"}
            metric_gate = {"status": "CANONICAL_A1_M1_M4_METRICS_COMPLETE"}
            complete_gate = {"status": "COMPLETE"}
            with mock.patch.object(
                RUNNER, "validate_m2_completion_gate", return_value=m2_gate
            ) as m2, mock.patch.object(
                RUNNER, "validate_canonical_a1_metrics_gate", return_value=metric_gate
            ) as metrics, mock.patch.object(
                RUNNER, "validate_completed_llm_setting", return_value=complete_gate
            ) as complete:
                dry = temp_spec(root, "M4", dry_run=True)
                RUNNER.validate_stage_prerequisites(
                    dry, prediction_root=root / "predictions", log_root=root / "logs"
                )
                self.assertEqual(m2.call_count, 1)
                metrics.assert_not_called()
                complete.assert_not_called()

                m4_a1 = temp_spec(root, "M4", dry_run=False)
                RUNNER.validate_stage_prerequisites(
                    m4_a1,
                    prediction_root=root / "predictions",
                    log_root=root / "logs",
                )
                self.assertEqual(complete.call_count, 1)
                prior_spec = complete.call_args.args[0]
                self.assertEqual(
                    (prior_spec.method, prior_spec.evaluation, prior_spec.fold),
                    ("M3", "A1", None),
                )

                complete.reset_mock()
                m4_a2 = RUNNER.make_spec(
                    "M4",
                    "A2",
                    2,
                    dry_run=False,
                    prediction_root=root / "predictions",
                    log_root=root / "logs",
                )
                RUNNER.validate_stage_prerequisites(
                    m4_a2,
                    prediction_root=root / "predictions",
                    log_root=root / "logs",
                )
                self.assertEqual(metrics.call_count, 1)
                self.assertEqual(complete.call_count, 5)
                self.assertEqual(
                    [
                        call.args[0].fold
                        for call in complete.call_args_list
                        if call.args[0].evaluation == "A2"
                    ],
                    [1, 2, 3],
                )

    def test_incomplete_m2_blocks_stage_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "models" / "a1" / "run_metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text('{"status":"IN_PROGRESS"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.LLMProtocolError, "M2 a1"):
                RUNNER.validate_m2_completion_gate(
                    model_root=root / "models", prediction_root=root / "predictions"
                )

    def test_complete_m2_a1_and_a2_artifacts_pass_stage_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            prediction_root = root / "predictions"
            for evaluation, fold in (("A1", None), ("A2", 1), ("A2", 2), ("A2", 3)):
                key = "a1" if evaluation == "A1" else f"a2_fold_{fold}"
                cases = [
                    case
                    for case in RUNNER.load_setting_rows(
                        evaluation, fold, self.benchmark
                    )
                    if case["role"] == "TEST"
                ]
                membership, split_sha, _rows = RUNNER._m2_split_membership_sha256(
                    evaluation, fold
                )
                prediction_path = prediction_root / f"{key}_test_predictions.jsonl"
                predictions = [
                    {
                        "method_id": "M2",
                        "evaluation": evaluation,
                        "fold": fold,
                        "split": "TEST",
                        "search_rank": case["search_rank"],
                        "canonical_url": case["canonical_url"],
                        "jurisdiction": case["jurisdiction"],
                        "fact_summary": case["fact_summary"],
                        "input_sha256": RUNNER.sha256_text(case["fact_summary"]),
                        "silver_reference_labels": case["silver_reference_labels"],
                        "predicted_labels": [],
                        "probabilities_by_label": {
                            label: 0.0 for label in RUNNER.AMP_LABEL_IDS
                        },
                        "primary_cohort_id": RUNNER.EXPECTED_COHORT_ID,
                        "config_sha256": RUNNER.EXPECTED_M2_CONFIG_SHA256,
                        "split_membership_sha256": membership,
                    }
                    for case in cases
                ]
                RUNNER.atomic_jsonl(prediction_path, predictions)
                metadata_path = model_root / key / "run_metadata.json"
                RUNNER.atomic_json(
                    metadata_path,
                    {
                        "artifact_schema_version": "sherloc-m2-artifacts-v1",
                        "method_id": "M2",
                        "evaluation": evaluation,
                        "fold": fold,
                        "primary_cohort_id": RUNNER.EXPECTED_COHORT_ID,
                        "label_order": list(RUNNER.AMP_LABEL_IDS),
                        "benchmark_sha256": RUNNER.EXPECTED_BENCHMARK_SHA256,
                        "ontology_sha256": RUNNER.EXPECTED_ONTOLOGY_SHA256,
                        "config_sha256": RUNNER.EXPECTED_M2_CONFIG_SHA256,
                        "split_file_sha256": split_sha,
                        "split_membership_sha256": membership,
                        "test_n": len(cases),
                        "status": "COMPLETE",
                        "prediction_rows": len(cases),
                        "test_labels_used_for_selection": False,
                        "prediction_path": str(prediction_path),
                        "prediction_sha256": RUNNER.sha256_file(prediction_path),
                    },
                )
            result = RUNNER.validate_m2_completion_gate(
                model_root=model_root, prediction_root=prediction_root
            )
            self.assertEqual(result["status"], "M2_A1_A2_COMPLETE")
            self.assertEqual(len(result["settings"]), 4)

    def test_deterministic_dry_run_is_non_test_and_excludes_all_approved_demos(self) -> None:
        rows = RUNNER.load_setting_rows("A1", None, self.benchmark)
        approved = [item["search_rank"] for item in self.bank["approved_cases"]]
        first = RUNNER.deterministic_dry_run_cases(
            rows, count=5, excluded_ranks=approved
        )
        second = RUNNER.deterministic_dry_run_cases(
            rows, count=5, excluded_ranks=approved
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(case["role"] != "TEST" for case in first))
        self.assertFalse({case["search_rank"] for case in first} & set(approved))

    def test_all_frozen_m4_banks_have_order_and_no_heldout_leakage(self) -> None:
        for bank_id in ("A1", "A2_FOLD_1", "A2_FOLD_2", "A2_FOLD_3"):
            with self.subTest(bank_id=bank_id):
                fold = int(bank_id[-1]) if bank_id.startswith("A2") else None
                if fold is None:
                    heldout: set[str] = set()
                else:
                    rows = RUNNER.load_setting_rows("A2", fold, self.benchmark)
                    heldout = {
                        case["jurisdiction"] for case in rows if case["role"] == "TEST"
                    }
                demos, metadata = RUNNER.load_demo_bank_for_setting(
                    bank_id,
                    self.bank,
                    self.config,
                    self.benchmark,
                    actual_test_jurisdictions=heldout,
                )
                self.assertEqual([item["demo_order"] for item in demos], list(range(1, 7)))
                self.assertEqual(len(demos), 6)
                self.assertFalse({item["jurisdiction"] for item in demos} & heldout)
                self.assertEqual(metadata["heldout_test_jurisdictions"], sorted(heldout))

    def test_demo_content_or_output_tampering_fails_before_request_build(self) -> None:
        tampered = copy.deepcopy(self.bank)
        tampered["approved_cases"][0]["fact_summary"] += " altered"
        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.validate_demo_bank_internal_hashes(tampered)

        tampered = copy.deepcopy(self.bank)
        tampered["approved_cases"][0]["output"]["acts"] = []
        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.load_demo_bank_for_setting(
                "A1", tampered, self.config, self.benchmark
            )

    def test_m4_request_is_amp_only_and_records_demo_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = temp_spec(Path(directory), "M4")
            prepared = RUNNER.prepare_run(spec)
            case = prepared["cases"][0]
            request = RUNNER.build_request_for_case(
                spec,
                case,
                demos=prepared["demos"],
                demo_metadata=prepared["demo_metadata"],
                heldout_jurisdictions=[],
                effective_model_id="gpt-5.6-luna",
                contract=prepared["contract"],
                config=prepared["config"],
                config_path=RUNNER.DEFAULT_CONFIG,
                m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
            )

        payload = request["payload"]
        schema = payload["text"]["format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"acts", "means", "purposes"})
        self.assertNotIn("geographic_form", json.dumps(payload).lower())
        self.assertNotIn("api_key", json.dumps(payload).lower())
        self.assertEqual(len(payload["input"]), 14)
        self.assertEqual(
            [item["demo_order"] for item in prepared["demo_metadata"]["demo_order"]],
            list(range(1, 7)),
        )

    def test_prepared_cases_and_demos_are_rederived_from_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = temp_spec(Path(directory), "M4")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            kwargs = {
                "demos": prepared["demos"],
                "demo_metadata": prepared["demo_metadata"],
                "heldout_jurisdictions": prepared["heldout_jurisdictions"],
                "config": prepared["config"],
                "demo_bank": prepared["demo_bank"],
            }
            RUNNER.validate_prepared_run_inputs(spec, prepared["cases"], **kwargs)

            altered_cases = copy.deepcopy(prepared["cases"])
            altered_cases[0]["fact_summary"] += " altered"
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.validate_prepared_run_inputs(spec, altered_cases, **kwargs)

            altered_demos = copy.deepcopy(prepared["demos"])
            altered_demos[0]["fact_summary"] += " altered"
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.validate_prepared_run_inputs(
                    spec,
                    prepared["cases"],
                    **{**kwargs, "demos": altered_demos},
                )

    def test_retry_after_is_respected_and_secret_is_redacted(self) -> None:
        secret = "sk-offline-unit-test-secret"
        failure = HTTPFailure(429, "7")
        failure.args = (f"rate limited {secret}",)
        client = FakeClient([failure, FakeResponse()])
        sleeps: list[float] = []
        result = RUNNER.invoke_with_retries(
            client,
            frozen_base_payload(),
            max_attempts=3,
            base_backoff_seconds=1.0,
            sleeper=sleeps.append,
            secret=secret,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(sleeps, [7.0])
        self.assertNotIn(secret, json.dumps(result["retry_events"]))

    def test_exact_incomplete_trigger_changes_only_output_budget_and_canonicalizes(self) -> None:
        unordered = {
            "acts": ["ACT_TRANSFER", "ACT_RECRUITMENT"],
            "means": ["MEANS_DECEPTION", "MEANS_ABDUCTION"],
            "purposes": ["PURPOSE_OTHER", "PURPOSE_SEXUAL_EXPLOITATION"],
        }
        client = FakeClient(
            [FakeIncompleteResponse(), FakeResponse(unordered)]
        )
        result = RUNNER.invoke_with_retries(
            client,
            frozen_base_payload(),
            max_attempts=5,
            base_backoff_seconds=0.0,
            sleeper=lambda _delay: None,
            secret="sk-offline",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(
            [call["max_output_tokens"] for call in client.responses.calls],
            [512, 2048],
        )
        initial = copy.deepcopy(client.responses.calls[0])
        fallback = copy.deepcopy(client.responses.calls[1])
        del initial["max_output_tokens"]
        del fallback["max_output_tokens"]
        self.assertEqual(initial, fallback)
        self.assertEqual(result["output_token_fallback_attempts_this_invocation"], 1)
        self.assertEqual(result["cumulative_output_token_fallback_attempts"], 1)
        self.assertEqual(
            result["parsed"]["validated_prediction"],
            {
                "acts": ["ACT_RECRUITMENT", "ACT_TRANSFER"],
                "means": ["MEANS_ABDUCTION", "MEANS_DECEPTION"],
                "purposes": ["PURPOSE_SEXUAL_EXPLOITATION", "PURPOSE_OTHER"],
            },
        )
        self.assertEqual(result["parsed"]["raw_structured_response"], unordered)
        trigger = result["retry_events"][0]
        self.assertIs(trigger["technical_fallback_trigger"], True)
        self.assertEqual(trigger["response_status"], "incomplete")
        self.assertEqual(
            trigger["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertEqual(trigger["max_output_tokens"], 512)

    def test_fallback_does_not_activate_for_other_statuses_or_reasons(self) -> None:
        outcomes = (
            FakeIncompleteResponse(reason="content_filter"),
            FakeIncompleteResponse(status="failed", reason="max_output_tokens"),
        )
        for outcome in outcomes:
            with self.subTest(status=outcome.status, reason=outcome.incomplete_details):
                client = FakeClient([outcome, FakeResponse()])
                result = RUNNER.invoke_with_retries(
                    client,
                    frozen_base_payload(),
                    max_attempts=5,
                    base_backoff_seconds=0.0,
                    sleeper=lambda _delay: None,
                    secret="sk-offline",
                )
                self.assertFalse(result["ok"])
                self.assertEqual(len(client.responses.calls), 1)
                self.assertFalse(result["output_token_fallback_used"])

    def test_fallback_has_two_actual_call_ceiling_and_raw_trigger_provenance(self) -> None:
        client = FakeClient(
            [
                FakeIncompleteResponse(response_id="resp_initial"),
                FakeIncompleteResponse(response_id="resp_fallback_1"),
                FakeIncompleteResponse(response_id="resp_fallback_2"),
                FakeResponse(),
            ]
        )
        result = RUNNER.invoke_with_retries(
            client,
            frozen_base_payload(),
            max_attempts=5,
            base_backoff_seconds=0.0,
            sleeper=lambda _delay: None,
            secret="sk-offline",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(len(client.responses.calls), 3)
        self.assertEqual(
            [call["max_output_tokens"] for call in client.responses.calls],
            [512, 2048, 2048],
        )
        self.assertEqual(result["output_token_fallback_attempts_this_invocation"], 2)
        self.assertEqual(result["cumulative_output_token_fallback_attempts"], 2)
        self.assertEqual(
            result["initial_incomplete_response_provenance"]["response_id"],
            "resp_initial",
        )
        self.assertEqual(
            [event["response_id"] for event in result["retry_events"]],
            ["resp_initial", "resp_fallback_1", "resp_fallback_2"],
        )

    def test_direct_fallback_resume_uses_no_new_512_call(self) -> None:
        proof = {
            "source": "PRE_AMENDMENT_PERSISTED_FAILURE_HISTORY",
            "response_status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "max_output_tokens": 512,
            "actual_request_sha256": RUNNER.sha256_text(
                RUNNER.canonical_json(frozen_base_payload())
            ),
        }
        client = FakeClient([FakeResponse()])
        result = RUNNER.invoke_with_retries(
            client,
            frozen_base_payload(),
            max_attempts=5,
            base_backoff_seconds=0.0,
            sleeper=lambda _delay: None,
            secret="sk-offline",
            start_with_fallback=True,
            prior_fallback_attempts=0,
            prior_primary_incomplete_provenance=proof,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(client.responses.calls[0]["max_output_tokens"], 2048)
        self.assertEqual(result["base_request_sha256"], proof["actual_request_sha256"])

        invalid_proof = copy.deepcopy(proof)
        invalid_proof["incomplete_details"] = {"reason": "content_filter"}
        blocked_client = FakeClient([FakeResponse()])
        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.invoke_with_retries(
                blocked_client,
                frozen_base_payload(),
                secret="sk-offline",
                start_with_fallback=True,
                prior_primary_incomplete_provenance=invalid_proof,
            )
        self.assertEqual(blocked_client.responses.calls, [])

    def test_primary_attempt_limit_allows_only_one_rank_551_base_call(self) -> None:
        client = FakeClient([HTTPFailure(429), FakeResponse()])
        result = RUNNER.invoke_with_retries(
            client,
            frozen_base_payload(),
            max_attempts=5,
            primary_attempt_limit=1,
            base_backoff_seconds=0.0,
            sleeper=lambda _delay: None,
            secret="sk-offline",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(client.responses.calls[0]["max_output_tokens"], 512)

    def test_rank_551_primary_reservation_blocks_second_call_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER.RunSpec(
                method="M3",
                evaluation="A1",
                fold=None,
                dry_run=False,
                bank_id=None,
                output_path=root / "predictions.jsonl",
                state_dir=root / "state",
                diagnostics_path=root / "diagnostics.json",
                failure_manifest_path=root / "failures.jsonl",
            )
            payload = frozen_base_payload()
            request = {
                "payload": payload,
                "request_sha256": RUNNER.sha256_text(RUNNER.canonical_json(payload)),
            }
            case = {"search_rank": 551}
            failure = {
                "method": "M3",
                "evaluation": "A1",
                "fold": None,
                "search_rank": 551,
                "request_sha256": request["request_sha256"],
                "recorded_at": "2026-08-15T00:00:00Z",
                "error": {
                    "error_type": "RequestBuildError",
                    "http_status": None,
                    "transient": False,
                    "message": RUNNER.LEGACY_LABEL_ORDER_MESSAGE,
                },
            }
            failure_path = root / "failure.json"
            reservation_path = root / "primary-reservation.json"
            RUNNER.atomic_json(
                failure_path,
                {
                    "status": "UNRESOLVED_FAILURE_HISTORY",
                    "search_rank": 551,
                    "attempt_history": [failure],
                    "latest_failure": failure,
                },
            )
            policy = RUNNER.resolve_failure_resume_policy(
                failure_path,
                case=case,
                request=request,
                spec=spec,
                primary_recovery_reservation_path=reservation_path,
            )
            self.assertEqual(policy["primary_attempt_limit"], 1)
            RUNNER.reserve_primary_recovery_attempt(
                reservation_path,
                case=case,
                request=request,
                spec=spec,
                primary_attempt_number=1,
                actual_payload=payload,
                secret="sk-offline",
            )

            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.resolve_failure_resume_policy(
                    failure_path,
                    case=case,
                    request=request,
                    spec=spec,
                    primary_recovery_reservation_path=reservation_path,
                )

    def test_fallback_reservation_is_cumulative_across_crash_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER.RunSpec(
                method="M3",
                evaluation="A1",
                fold=None,
                dry_run=False,
                bank_id=None,
                output_path=root / "predictions.jsonl",
                state_dir=root / "state",
                diagnostics_path=root / "diagnostics.json",
                failure_manifest_path=root / "failures.jsonl",
            )
            payload = frozen_base_payload()
            base_sha = RUNNER.sha256_text(RUNNER.canonical_json(payload))
            request = {"payload": payload, "request_sha256": base_sha}
            case = {"search_rank": 266}
            failure = {
                "method": "M3",
                "evaluation": "A1",
                "fold": None,
                "search_rank": 266,
                "request_sha256": base_sha,
                "recorded_at": "2026-08-15T00:00:00Z",
                "error": {
                    "error_type": "LLMProtocolError",
                    "http_status": None,
                    "transient": False,
                    "message": RUNNER.LEGACY_MAX_OUTPUT_INCOMPLETE_MESSAGE,
                },
            }
            failure_path = root / "failure.json"
            reservation_path = root / "fallback-reservation.json"
            RUNNER.atomic_json(
                failure_path,
                {
                    "status": "UNRESOLVED_FAILURE_HISTORY",
                    "search_rank": 266,
                    "attempt_history": [failure],
                    "latest_failure": failure,
                },
            )
            fallback_payload = RUNNER.output_token_fallback_payload(payload)
            RUNNER.reserve_fallback_attempt(
                reservation_path,
                case=case,
                request=request,
                spec=spec,
                fallback_attempt_number=1,
                actual_payload=fallback_payload,
                secret="sk-offline",
            )
            policy = RUNNER.resolve_failure_resume_policy(
                failure_path,
                case=case,
                request=request,
                spec=spec,
                fallback_reservation_path=reservation_path,
            )
            self.assertTrue(policy["start_with_fallback"])
            self.assertEqual(policy["prior_fallback_attempts"], 1)
            self.assertEqual(policy["resume_source"], "CRASH_SAFE_FALLBACK_RESERVATION")

            client = FakeClient([FakeResponse()])
            result = RUNNER.invoke_with_retries(
                client,
                payload,
                secret="sk-offline",
                start_with_fallback=True,
                prior_fallback_attempts=1,
                prior_primary_incomplete_provenance=policy[
                    "prior_primary_incomplete_provenance"
                ],
                fallback_attempt_reserver=lambda number, actual: (
                    RUNNER.reserve_fallback_attempt(
                        reservation_path,
                        case=case,
                        request=request,
                        spec=spec,
                        fallback_attempt_number=number,
                        actual_payload=actual,
                        secret="sk-offline",
                    )
                ),
            )
            self.assertTrue(result["ok"])
            self.assertEqual(len(client.responses.calls), 1)
            self.assertEqual(
                RUNNER._fallback_reservation_count(
                    reservation_path, case=case, request=request, spec=spec
                ),
                2,
            )
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.resolve_failure_resume_policy(
                    failure_path,
                    case=case,
                    request=request,
                    spec=spec,
                    fallback_reservation_path=reservation_path,
                )

    def test_rank_1340_exception_allows_only_two_more_identical_2048_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER.RunSpec(
                method="M4",
                evaluation="A2",
                fold=1,
                dry_run=False,
                bank_id="A2_FOLD_1",
                output_path=root / "predictions.jsonl",
                state_dir=root / "state",
                diagnostics_path=root / "diagnostics.json",
                failure_manifest_path=root / "failures.jsonl",
            )
            case = {"search_rank": 1340}
            payload = frozen_base_payload()
            base_sha = RUNNER.sha256_text(RUNNER.canonical_json(payload))
            request = {"payload": payload, "request_sha256": base_sha}
            fallback_payload = RUNNER.output_token_fallback_payload(payload)
            fallback_sha = RUNNER.sha256_text(
                RUNNER.canonical_json(fallback_payload)
            )
            proof = {
                "response_status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "max_output_tokens": 512,
                "actual_request_sha256": base_sha,
            }
            retry_events = [
                {
                    "request_phase": "FALLBACK_2048",
                    "max_output_tokens": 2048,
                    "actual_request_sha256": fallback_sha,
                    "http_status": 429,
                }
                for _ in range(2)
            ]
            failure = {
                "method": "M4",
                "evaluation": "A2",
                "fold": 1,
                "search_rank": 1340,
                "request_sha256": base_sha,
                "retry_events": retry_events,
                "technical_execution": {
                    "initial_incomplete_response_provenance": proof,
                    "cumulative_output_token_fallback_attempts": 2,
                },
            }
            failure_path = root / "failure.json"
            reservation_path = root / "reservation.json"
            RUNNER.atomic_json(
                failure_path,
                {
                    "status": "UNRESOLVED_FAILURE_HISTORY",
                    "search_rank": 1340,
                    "attempt_history": [failure],
                    "latest_failure": failure,
                },
            )
            for number in (1, 2):
                RUNNER.reserve_fallback_attempt(
                    reservation_path,
                    case=case,
                    request=request,
                    spec=spec,
                    fallback_attempt_number=number,
                    actual_payload=fallback_payload,
                    secret="sk-offline",
                )

            policy = RUNNER.resolve_failure_resume_policy(
                failure_path,
                case=case,
                request=request,
                spec=spec,
                fallback_reservation_path=reservation_path,
            )
            self.assertEqual(policy["prior_fallback_attempts"], 2)
            self.assertEqual(policy["fallback_attempt_ceiling"], 4)
            self.assertEqual(
                policy["technical_exception_id"], RUNNER.RANK_1340_EXCEPTION_ID
            )

            client = FakeClient([HTTPFailure(429), FakeResponse()])
            result = RUNNER.invoke_with_retries(
                client,
                payload,
                base_backoff_seconds=0.0,
                sleeper=lambda _delay: None,
                secret="sk-offline",
                start_with_fallback=True,
                prior_fallback_attempts=policy["prior_fallback_attempts"],
                prior_primary_incomplete_provenance=proof,
                fallback_attempt_ceiling=policy["fallback_attempt_ceiling"],
                technical_exception_id=policy["technical_exception_id"],
                technical_exception_sha256=policy["technical_exception_sha256"],
                fallback_attempt_reserver=lambda number, actual: (
                    RUNNER.reserve_fallback_attempt(
                        reservation_path,
                        case=case,
                        request=request,
                        spec=spec,
                        fallback_attempt_number=number,
                        actual_payload=actual,
                        secret="sk-offline",
                        technical_exception_id=policy["technical_exception_id"],
                        technical_exception_sha256=policy[
                            "technical_exception_sha256"
                        ],
                    )
                ),
            )
            self.assertTrue(result["ok"])
            self.assertEqual(
                [call["max_output_tokens"] for call in client.responses.calls],
                [2048, 2048],
            )
            self.assertEqual(result["cumulative_output_token_fallback_attempts"], 4)
            self.assertEqual(
                RUNNER._fallback_reservation_count(
                    reservation_path, case=case, request=request, spec=spec
                ),
                4,
            )

    def test_persisted_legacy_trigger_is_narrow_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER.RunSpec(
                method="M3",
                evaluation="A1",
                fold=None,
                dry_run=False,
                bank_id=None,
                output_path=root / "predictions.jsonl",
                state_dir=root / "state",
                diagnostics_path=root / "diagnostics.json",
                failure_manifest_path=root / "failures.jsonl",
            )
            for rank, expected_direct in ((266, True), (1356, True), (551, False)):
                with self.subTest(rank=rank):
                    case = {"search_rank": rank}
                    request = {
                        "request_sha256": "a" * 64,
                    }
                    message = (
                        RUNNER.LEGACY_MAX_OUTPUT_INCOMPLETE_MESSAGE
                        if expected_direct
                        else "means labels are not in frozen ontology order"
                    )
                    failure = {
                        "method": "M3",
                        "evaluation": "A1",
                        "fold": None,
                        "search_rank": rank,
                        "request_sha256": "a" * 64,
                        "recorded_at": "2026-08-15T00:00:00Z",
                        "error": {
                            "error_type": (
                                "LLMProtocolError" if expected_direct else "RequestBuildError"
                            ),
                            "http_status": None,
                            "transient": False,
                            "message": message,
                        },
                    }
                    path = root / f"{rank}.json"
                    RUNNER.atomic_json(
                        path,
                        {
                            "status": "UNRESOLVED_FAILURE_HISTORY",
                            "search_rank": rank,
                            "attempt_history": [failure],
                            "latest_failure": failure,
                        },
                    )
                    policy = RUNNER.resolve_failure_resume_policy(
                        path, case=case, request=request, spec=spec
                    )
                    self.assertIs(policy["start_with_fallback"], expected_direct)
                    self.assertEqual(
                        policy["primary_attempt_limit"],
                        1 if rank == 551 else None,
                    )
                    self.assertIs(
                        policy["primary_recovery_required"], rank == 551
                    )

            unauthorized = copy.deepcopy(failure)
            unauthorized["search_rank"] = 552
            unauthorized["error"]["error_type"] = "LLMProtocolError"
            unauthorized["error"]["message"] = (
                RUNNER.LEGACY_MAX_OUTPUT_INCOMPLETE_MESSAGE
            )
            path = root / "552.json"
            RUNNER.atomic_json(
                path,
                {
                    "status": "UNRESOLVED_FAILURE_HISTORY",
                    "search_rank": 552,
                    "attempt_history": [unauthorized],
                    "latest_failure": unauthorized,
                },
            )
            policy = RUNNER.resolve_failure_resume_policy(
                path,
                case={"search_rank": 552},
                request={"request_sha256": "a" * 64},
                spec=spec,
            )
            self.assertFalse(policy["start_with_fallback"])

    def test_atomic_success_state_resume_and_secret_nonserialization(self) -> None:
        secret = "sk-offline-never-write-me"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            cases = prepared["cases"]
            client = FakeClient()
            marker = {
                "effective_model_id": "gpt-5.6-luna",
                "marker_sha256": "a" * 64,
            }
            kwargs = {
                "client": client,
                "secret": secret,
                "sdk_version": "2.31.0",
                "contract": prepared["contract"],
                "config": prepared["config"],
                "model_marker": marker,
                "demos": None,
                "demo_metadata": None,
                "heldout_jurisdictions": [],
                "config_path": RUNNER.DEFAULT_CONFIG,
                "m3_prompt_path": RUNNER.DEFAULT_M3_PROMPT,
                "m4_prompt_path": RUNNER.DEFAULT_M4_PROMPT,
                "max_attempts": 2,
                "base_backoff_seconds": 0.0,
                "workers": 2,
                "sleeper": lambda _delay: None,
            }
            with mock.patch.object(
                RUNNER,
                "validate_stage_prerequisites",
                return_value={"status": "OFFLINE_TEST_GATE"},
            ):
                first = RUNNER.execute_cases(spec, cases, **kwargs)
                calls_after_first = len(client.responses.calls)
                second = RUNNER.execute_cases(spec, cases, **kwargs)

            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(second["status"], "COMPLETE")
            self.assertEqual(calls_after_first, 3)
            self.assertEqual(len(client.responses.calls), calls_after_first)
            self.assertEqual(
                second["stage_prerequisite_provenance"],
                first["stage_prerequisite_provenance"],
            )
            rows = RUNNER.load_jsonl(spec.output_path)
            self.assertEqual(len(rows), 3)
            required = {
                "method_id",
                "evaluation",
                "search_rank",
                "jurisdiction",
                "fact_summary",
                "silver_reference_labels",
                "predicted_labels",
                "normalized_prediction",
                "requested_model_id",
                "returned_model_id",
                "request_sha256",
                "response_id",
                "token_usage",
                "latency_seconds",
                "retry_count",
                "status",
            }
            self.assertTrue(required <= set(rows[0]))
            evaluator_row = dict(rows[0])
            evaluator_row["split"] = "TEST"  # Dry-run files are intentionally excluded.
            normalized = EVALUATOR._normalise_prediction_row(
                evaluator_row, spec.output_path, 1
            )
            self.assertEqual(len(normalized), 1)
            self.assertEqual(normalized[0].method, "M3")
            self.assertEqual(
                list(normalized[0].predicted_labels), rows[0]["predicted_labels"]
            )
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes())

    def test_pre_amendment_success_is_accepted_and_never_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            case = prepared["cases"][0]
            request = RUNNER.build_request_for_case(
                spec,
                case,
                demos=None,
                demo_metadata=None,
                heldout_jurisdictions=[],
                effective_model_id="gpt-5.6-luna",
                contract=prepared["contract"],
                config=prepared["config"],
                config_path=RUNNER.DEFAULT_CONFIG,
                m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
            )
            result = RUNNER.invoke_with_retries(
                FakeClient(),
                request["payload"],
                max_attempts=1,
                base_backoff_seconds=0.0,
                sleeper=lambda _delay: None,
                secret="sk-offline",
            )
            record = RUNNER.make_success_record(
                spec,
                case,
                request,
                result,
                contract=prepared["contract"],
                config=prepared["config"],
                model_marker={"effective_model_id": "gpt-5.6-luna"},
                sdk_version="2.31.0",
                demo_metadata=None,
                split_membership_sha256=RUNNER.target_membership_hash(
                    prepared["cases"], spec
                ),
            )
            record["runner_version"] = "1.1.0"
            record.pop("technical_execution")
            record.pop("label_array_canonicalization_applied")
            success_path = RUNNER._state_path(
                spec.state_dir / "success", int(case["search_rank"])
            )
            RUNNER.atomic_json(success_path, record)
            before = success_path.read_bytes()

            loaded = RUNNER._load_existing_success(
                success_path, case=case, request=request, spec=spec
            )

            self.assertEqual(loaded, record)
            self.assertEqual(success_path.read_bytes(), before)

    def test_amended_success_validator_enforces_raw_canonical_and_budget_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            case = prepared["cases"][0]
            request = RUNNER.build_request_for_case(
                spec,
                case,
                demos=None,
                demo_metadata=None,
                heldout_jurisdictions=[],
                effective_model_id="gpt-5.6-luna",
                contract=prepared["contract"],
                config=prepared["config"],
                config_path=RUNNER.DEFAULT_CONFIG,
                m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
            )
            unordered = {
                "acts": ["ACT_TRANSFER", "ACT_RECRUITMENT"],
                "means": ["MEANS_DECEPTION", "MEANS_ABDUCTION"],
                "purposes": ["PURPOSE_OTHER", "PURPOSE_SEXUAL_EXPLOITATION"],
            }
            result = RUNNER.invoke_with_retries(
                FakeClient([FakeResponse(unordered)]),
                request["payload"],
                max_attempts=1,
                secret="sk-offline",
            )
            record = RUNNER.make_success_record(
                spec,
                case,
                request,
                result,
                contract=prepared["contract"],
                config=prepared["config"],
                model_marker={"effective_model_id": "gpt-5.6-luna"},
                sdk_version="2.31.0",
                demo_metadata=None,
                split_membership_sha256=RUNNER.target_membership_hash(
                    prepared["cases"], spec
                ),
            )
            self.assertIs(record["label_array_canonicalization_applied"], True)
            RUNNER.validate_persisted_success_record(
                record, request=request, context="offline amended record"
            )

            bad_flag = copy.deepcopy(record)
            bad_flag["label_array_canonicalization_applied"] = False
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.validate_persisted_success_record(
                    bad_flag, request=request, context="bad flag"
                )

            bad_hash = copy.deepcopy(record)
            bad_hash["technical_execution"]["actual_request_sha256"] = "0" * 64
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.validate_persisted_success_record(
                    bad_hash, request=request, context="bad actual hash"
                )

    def test_canonical_m3_a1_amendment_scope_rejects_any_extra_pending_rank(self) -> None:
        spec = RUNNER.make_spec(
            "M3",
            "A1",
            None,
            dry_run=False,
            prediction_root=RUNNER.DEFAULT_PREDICTION_ROOT,
            log_root=RUNNER.DEFAULT_LOG_ROOT,
        )
        direct = {
            "start_with_fallback": True,
            "prior_fallback_attempts": 0,
            "prior_primary_incomplete_provenance": {"response_status": "incomplete"},
            "primary_attempt_limit": None,
            "primary_recovery_required": False,
        }
        base = {
            "start_with_fallback": False,
            "prior_fallback_attempts": 0,
            "prior_primary_incomplete_provenance": None,
            "primary_attempt_limit": 1,
            "primary_recovery_required": True,
        }
        valid_pending = [
            ({"search_rank": 266}, {}, direct),
            ({"search_rank": 551}, {}, base),
            ({"search_rank": 1356}, {}, direct),
        ]
        RUNNER.validate_canonical_m3_a1_amendment_scope(spec, valid_pending)

        wrong_budget = copy.deepcopy(valid_pending)
        wrong_budget[0][2]["start_with_fallback"] = False
        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.validate_canonical_m3_a1_amendment_scope(spec, wrong_budget)

        extra = valid_pending + [({"search_rank": 777}, {}, base)]
        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.validate_canonical_m3_a1_amendment_scope(spec, extra)

    def test_schema_failure_is_persistent_and_never_becomes_empty_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            cases = prepared["cases"][:1]
            client = FakeClient([FakeResponse({"acts": [], "means": []})])
            with mock.patch.object(
                RUNNER,
                "validate_stage_prerequisites",
                return_value={"status": "OFFLINE_TEST_GATE"},
            ), mock.patch.object(RUNNER, "validate_prepared_run_inputs"):
                diagnostics = RUNNER.execute_cases(
                    spec,
                    cases,
                    client=client,
                    secret="sk-offline",
                    sdk_version="2.31.0",
                    contract=prepared["contract"],
                    config=prepared["config"],
                    model_marker={"effective_model_id": "gpt-5.6-luna"},
                    demos=None,
                    demo_metadata=None,
                    heldout_jurisdictions=[],
                    config_path=RUNNER.DEFAULT_CONFIG,
                    m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                    m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                    max_attempts=2,
                    base_backoff_seconds=0.0,
                    sleeper=lambda _delay: None,
                )

            self.assertEqual(diagnostics["status"], "COMPLETE_WITH_UNRESOLVED_FAILURES")
            self.assertFalse(spec.output_path.exists())
            failures = RUNNER.load_jsonl(spec.failure_manifest_path)
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["status"], "FAILED_NO_PREDICTION")
            self.assertIsNone(failures[0]["validated_prediction"])

    def test_model_snapshot_resolution_prefers_latest_exposed_snapshot(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: [
                    {"id": "gpt-5.6-luna"},
                    {"id": "gpt-5.6-luna-2026-06-01"},
                    {"id": "gpt-5.6-luna-2026-08-01"},
                ],
                retrieve=lambda _model: {"id": "gpt-5.6-luna"},
            )
        )
        result = RUNNER.resolve_model_access(client)

        self.assertEqual(result["effective_model_id"], "gpt-5.6-luna-2026-08-01")
        self.assertEqual(result["selection_basis"], "LATEST_EXPOSED_DATED_SNAPSHOT")

    def test_api_key_is_environment_only_and_never_optional_for_live_stages(self) -> None:
        with self.assertRaises(RUNNER.LLMProtocolError):
            RUNNER.require_api_key({})
        self.assertEqual(
            RUNNER.require_api_key({"OPENAI_API_KEY": "sk-test-only"}),
            "sk-test-only",
        )
        self.assertIs(self.config["api_request"]["api_key_in_config"], False)
        self.assertFalse(RUNNER._contains_secret_key(self.config))

    def test_concurrent_fatal_stop_drains_and_commits_inflight_paid_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            cases = prepared["cases"][:2]
            responses = CoordinatedFatalAndSuccessResponses()
            client = SimpleNamespace(responses=responses)
            with mock.patch.object(
                RUNNER,
                "validate_stage_prerequisites",
                return_value={"status": "OFFLINE_TEST_GATE"},
            ), mock.patch.object(RUNNER, "validate_prepared_run_inputs"):
                diagnostics = RUNNER.execute_cases(
                    spec,
                    cases,
                    client=client,
                    secret="offline-secret",
                    sdk_version="2.31.0",
                    contract=prepared["contract"],
                    config=prepared["config"],
                    model_marker={"effective_model_id": "gpt-5.6-luna"},
                    demos=None,
                    demo_metadata=None,
                    heldout_jurisdictions=[],
                    config_path=RUNNER.DEFAULT_CONFIG,
                    m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                    m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                    max_attempts=1,
                    base_backoff_seconds=0.0,
                    workers=2,
                    sleeper=lambda _delay: None,
                )

            self.assertEqual(responses.call_count, 2)
            self.assertEqual(diagnostics["status"], "BLOCKED_FATAL_API_ACCESS_ERROR")
            self.assertEqual(diagnostics["successful_cases"], 1)
            self.assertEqual(diagnostics["unresolved_failure_cases"], 1)
            self.assertEqual(len(RUNNER.load_jsonl(spec.output_path)), 1)

    def test_concurrent_result_is_unprocessed_until_durable_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            cases = prepared["cases"][:2]
            responses = CoordinatedSuccessResponses()
            client = SimpleNamespace(responses=responses)
            real_atomic_json = RUNNER.atomic_json
            failed_once = False

            def fail_first_success_commit(path, value, **kwargs):
                nonlocal failed_once
                if path.parent.name == "success" and not failed_once:
                    failed_once = True
                    raise OSError("simulated first durable-commit failure")
                return real_atomic_json(path, value, **kwargs)

            with mock.patch.object(
                RUNNER, "atomic_json", side_effect=fail_first_success_commit
            ), mock.patch.object(
                RUNNER,
                "validate_stage_prerequisites",
                return_value={"status": "OFFLINE_TEST_GATE"},
            ), mock.patch.object(RUNNER, "validate_prepared_run_inputs"):
                with self.assertRaisesRegex(OSError, "durable-commit"):
                    RUNNER.execute_cases(
                        spec,
                        cases,
                        client=client,
                        secret="offline-secret",
                        sdk_version="2.31.0",
                        contract=prepared["contract"],
                        config=prepared["config"],
                        model_marker={"effective_model_id": "gpt-5.6-luna"},
                        demos=None,
                        demo_metadata=None,
                        heldout_jurisdictions=[],
                        config_path=RUNNER.DEFAULT_CONFIG,
                        m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                        m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                        max_attempts=1,
                        base_backoff_seconds=0.0,
                        workers=2,
                        sleeper=lambda _delay: None,
                    )

            self.assertEqual(responses.call_count, 2)
            success_files = sorted((spec.state_dir / "success").glob("*.json"))
            self.assertEqual(len(success_files), 2)
            self.assertTrue(
                all(RUNNER.load_json(path)["status"] == "SUCCESS_VALIDATED" for path in success_files)
            )

    def test_post_build_canonical_revalidation_precedes_any_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            client = FakeClient()
            calls = 0

            def canonical_check():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RUNNER.LLMProtocolError("simulated post-build hash drift")

            with mock.patch.object(
                RUNNER,
                "validate_canonical_artifact_hashes",
                side_effect=canonical_check,
            ):
                with self.assertRaisesRegex(RUNNER.LLMProtocolError, "post-build"):
                    RUNNER.execute_cases(
                        spec,
                        prepared["cases"],
                        client=client,
                        secret="offline-secret",
                        sdk_version="2.31.0",
                        contract=prepared["contract"],
                        config=prepared["config"],
                        model_marker={"effective_model_id": "gpt-5.6-luna"},
                        demos=None,
                        demo_metadata=None,
                        heldout_jurisdictions=[],
                        config_path=RUNNER.DEFAULT_CONFIG,
                        m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                        m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                        max_attempts=1,
                        base_backoff_seconds=0.0,
                    )
            self.assertEqual(client.responses.calls, [])

    def test_post_build_builder_metadata_revalidation_precedes_any_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            client = FakeClient()
            with mock.patch.object(
                RUNNER,
                "revalidate_built_request",
                side_effect=RUNNER.LLMProtocolError("post-build builder drift"),
            ):
                with self.assertRaisesRegex(RUNNER.LLMProtocolError, "builder drift"):
                    RUNNER.execute_cases(
                        spec,
                        prepared["cases"],
                        client=client,
                        secret="offline-secret",
                        sdk_version="2.31.0",
                        contract=prepared["contract"],
                        config=prepared["config"],
                        model_marker={"effective_model_id": "gpt-5.6-luna"},
                        demos=None,
                        demo_metadata=None,
                        heldout_jurisdictions=[],
                        config_path=RUNNER.DEFAULT_CONFIG,
                        m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                        m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                        max_attempts=1,
                        base_backoff_seconds=0.0,
                    )
            self.assertEqual(client.responses.calls, [])

    def test_mandatory_stage_gate_runs_before_pending_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            client = FakeClient()
            with mock.patch.object(
                RUNNER,
                "validate_stage_prerequisites",
                return_value={"status": "GATE_PASSED"},
            ) as gate:
                diagnostics = RUNNER.execute_cases(
                    spec,
                    prepared["cases"],
                    client=client,
                    secret="offline-secret",
                    sdk_version="2.31.0",
                    contract=prepared["contract"],
                    config=prepared["config"],
                    model_marker={"effective_model_id": "gpt-5.6-luna"},
                    demos=None,
                    demo_metadata=None,
                    heldout_jurisdictions=[],
                    config_path=RUNNER.DEFAULT_CONFIG,
                    m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                    m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                    max_attempts=1,
                    base_backoff_seconds=0.0,
                )
            gate.assert_called_once()
            gated_spec = gate.call_args.args[0]
            self.assertEqual(gated_spec.setting_id, spec.setting_id)
            self.assertEqual(
                diagnostics["stage_prerequisite_provenance"],
                {"status": "GATE_PASSED"},
            )
            self.assertIn("run_lock", diagnostics)
            history = RUNNER.load_json(Path(diagnostics["run_lock"]["history_path"]))
            self.assertEqual(history["status"], "RELEASED")

    def test_failed_mandatory_stage_gate_sends_no_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = temp_spec(root, "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            client = FakeClient()
            with mock.patch.object(
                RUNNER,
                "validate_stage_prerequisites",
                side_effect=RUNNER.LLMProtocolError("M2 stages incomplete"),
            ):
                with self.assertRaisesRegex(RUNNER.LLMProtocolError, "M2 stages"):
                    RUNNER.execute_cases(
                        spec,
                        prepared["cases"],
                        client=client,
                        secret="offline-secret",
                        sdk_version="2.31.0",
                        contract=prepared["contract"],
                        config=prepared["config"],
                        model_marker={"effective_model_id": "gpt-5.6-luna"},
                        demos=None,
                        demo_metadata=None,
                        heldout_jurisdictions=[],
                        config_path=RUNNER.DEFAULT_CONFIG,
                        m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                        m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                        max_attempts=1,
                        base_backoff_seconds=0.0,
                    )
            self.assertEqual(client.responses.calls, [])

    def test_workers_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = temp_spec(Path(directory), "M3")
            prepared = RUNNER.prepare_run(spec, dry_run_count=3)
            with self.assertRaises(RUNNER.LLMProtocolError):
                RUNNER.execute_cases(
                    spec,
                    prepared["cases"][:1],
                    client=FakeClient(),
                    secret="sk-offline",
                    sdk_version="2.31.0",
                    contract=prepared["contract"],
                    config=prepared["config"],
                    model_marker={"effective_model_id": "gpt-5.6-luna"},
                    demos=None,
                    demo_metadata=None,
                    heldout_jurisdictions=[],
                    config_path=RUNNER.DEFAULT_CONFIG,
                    m3_prompt_path=RUNNER.DEFAULT_M3_PROMPT,
                    m4_prompt_path=RUNNER.DEFAULT_M4_PROMPT,
                    max_attempts=1,
                    base_backoff_seconds=0.0,
                    workers=RUNNER.MAX_WORKERS + 1,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
